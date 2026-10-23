#!/usr/bin/env python3
"""复用收益分析 / Reuse Benefit Analysis.

An opt-in, single-file research utility. It does NOT register serving hooks,
modify inference/scheduling, download data/models, or control GPU holders.
Import and --help use only the Python standard library (Python >= 3.10).
Optional dependencies: tokenizers for replay; torch and this repository's
head_kv package for memory; matplotlib for plot.

Two separate experiments, not an end-to-end speed benchmark:
  1. Bounded-cache replay: exact-text reuse opportunities vs prefix-cache hits.
     Real QA queries can be retrieved with BM25; source-order results must be
     shown alongside seeded shuffles, since QA order is not a production trace.
  2. Real HeadKVManager/SegPaged residency: controlled synthetic K/V fixtures,
     shared local heads, private global heads, COW, descriptor and host metadata.
     This measures storage, NOT model-output equivalence or serving throughput.

Commands (run from repository root; replace dataset/tokenizer paths):
  python test/srt/redknot/reuse_benefit_analysis.py self-test
  python test/srt/redknot/reuse_benefit_analysis.py self-test --with-memory

  python test/srt/redknot/reuse_benefit_analysis.py prepare \\
    --source musique=/data/musique_ans_v1.0_dev.jsonl \\
    --source hotpotqa=/data/LongBench/data/hotpotqa.jsonl --output-dir /tmp/reuse/traces
  python test/srt/redknot/reuse_benefit_analysis.py retrieve \\
    --trace /tmp/reuse/traces/musique.jsonl --top-k 8 --output /tmp/reuse/bm25.jsonl
  python test/srt/redknot/reuse_benefit_analysis.py replay \\
    --trace /tmp/reuse/bm25.jsonl --tokenizer /models/Qwen/tokenizer.json \\
    --token-prefix-control --output /tmp/reuse/cache.json
  python test/srt/redknot/reuse_benefit_analysis.py memory --output /tmp/reuse/memory.json
  python test/srt/redknot/reuse_benefit_analysis.py plot \\
    --cache /tmp/reuse/cache.json --memory /tmp/reuse/memory.json \\
    --output-dir /tmp/reuse/figures --formats png,pdf,svg

Explicit reproduction of the larger storage fixture (requires a free GPU with
adequate memory; coordinate holders externally before/after running):
  python test/srt/redknot/reuse_benefit_analysis.py memory \\
    --device cuda:0 --dtype bfloat16 --layers 32 --kv-heads 8 --local-heads 6 \\
    --head-dim 128 --document-tokens 8192 --prefix-tokens 128 --page-size 256 \\
    --cow-rows 1 --multiplicities 1 2 4 8 16 32 --output /tmp/reuse/gpu_memory.json
For the fixed-slab control, repeat with --fixed-capacity-pages 270528
and a NEW output path. Sharing may lower occupied bytes without reducing a
preallocated slab. --local-heads is a controlled variable, NOT a profiled mask.

Raw input schemas: official MuSiQue paragraphs/question/id or LongBench
Passage N context/input/_id. prepare strips answer/support fields. replay expects
canonical request_id/dataset/query/docs with exact SHA-256 content IDs.
A repeated --trace evaluates workloads independently; never silently pools them.
No automatic data selection, synthetic popularity, or cherry-picking.

Interpretation:
- Cache budgets use unique OBSERVED document working-set bytes. The default
  131072 bytes/token is illustrative conventional KV, not Qwen hybrid/MLA state.
- Document recurrence is only an opportunity; position/context-dependent state
  still needs a correct restoration algorithm before reuse in real inference.
- COW tests exact fixture values at sampled rows, not real-model accuracy.
- CPU metadata is reachable Python-object size, not full process RSS.
- Neither experiment reports accuracy, FLOP savings, TTFT, or QPS.
- Output files are explicit, hash-attributed and no-clobber by default.

Dataset references:
  https://github.com/StonyBrookNLP/musique
  https://github.com/THUDM/LongBench
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import gc
import hashlib
import heapq
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean
import struct
import sys
import tempfile
import time
from typing import Callable, Iterable
import unittest


# ---- Canonical traces and deterministic retrieval -------------------------
TRACE_PASSAGE = re.compile(r"^Passage ([1-9][0-9]*):\r?\n", re.MULTILINE)
TRACE_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
RETRIEVAL_WORD = re.compile(r"\w+", re.UNICODE)


def trace_sha(data):
    return hashlib.sha256(data).hexdigest()


def trace_encode_jsonl(rows):
    return ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                allow_nan=False) for row in rows) + "\n").encode("utf-8")


def trace_write_bytes(path, data, overwrite=False):
    write_bytes(path, data, overwrite)


def trace_write_json(path, value, overwrite=False):
    write_json(path, value, overwrite)


def trace_make_doc(title, text, index=None):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("document title must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("rendered document must be a non-empty string")
    result = {"content_id": trace_sha(text.encode("utf-8")), "text": text, "title": title}
    if index is not None:
        result["source_document_index"] = index
    return result


def trace_extract_musique(row):
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("MuSiQue input requires a non-empty paragraphs list")
    result = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            raise ValueError("paragraph must be an object")
        title, body = paragraph.get("title"), paragraph.get("paragraph_text")
        if not isinstance(title, str) or not title.strip() or not isinstance(body, str) or not body.strip():
            raise ValueError("paragraph title and paragraph_text must be non-empty strings")
        result.append(trace_make_doc(title, title + "\n" + body, index))
    return result


def trace_extract_longbench(row):
    context = row.get("context")
    if not isinstance(context, str):
        raise ValueError("LongBench context must be text")
    matches = list(TRACE_PASSAGE.finditer(context))
    if not matches or context[:matches[0].start()].strip():
        raise ValueError("unsupported LongBench prelude or missing Passage N markers")
    if [int(match[1]) for match in matches] != list(range(1, len(matches) + 1)):
        raise ValueError("LongBench Passage indices must be sequential 1..N")
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        # Remove positional framing and outer delimiter whitespace, but preserve
        # exact title/body text internally; title-only identity is never used.
        text = context[match.end():end].strip()
        title, separator, body = text.partition("\n")
        if not separator or not body.strip():
            raise ValueError("LongBench document must have title followed by body")
        result.append(trace_make_doc(title, text, index))
    return result


def trace_build_rows(dataset, raw):
    if not isinstance(dataset, str) or not TRACE_DATASET_NAME.fullmatch(dataset):
        raise ValueError("dataset name must contain only letters, digits, _, - or . and start with a letter/digit")
    rows, seen, formats = [], set(), set()
    for source_index, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("source JSONL rows must be objects")
        if "paragraphs" in row:
            source_format = "musique"
            source_id, query = row.get("id"), row.get("question")
            docs = trace_extract_musique(row)
        else:
            source_format = "longbench_passages"
            source_id, query = row.get("_id"), row.get("input")
            docs = trace_extract_longbench(row)
        formats.add(source_format)
        if len(formats) > 1:
            raise ValueError("one source file must not mix MuSiQue and LongBench schemas")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("missing non-empty source id/_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("missing non-empty question/input")
        request_id = dataset + ":" + source_id
        if request_id in seen:
            raise ValueError("duplicate source request ID: " + request_id)
        seen.add(request_id)
        # Deliberately whitelist fields; answer/support labels are not exported.
        rows.append({"request_id": request_id, "dataset": dataset, "query": query,
                     "docs": docs, "source_id": source_id,
                     "source_row_index": source_index, "source_format": source_format})
    if not rows:
        raise ValueError("source contains no requests")
    return rows


def trace_validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("canonical trace must contain requests")
    request_ids, texts = set(), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("canonical request must be an object")
        for field in ("request_id", "dataset", "query"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("canonical request requires non-empty " + field)
        if row["request_id"] in request_ids:
            raise ValueError("duplicate canonical request ID")
        request_ids.add(row["request_id"])
        forbidden = {"answer", "answers", "gold_answer", "label", "supporting_facts",
                     "is_supporting", "question_decomposition"}
        if forbidden.intersection(row):
            raise ValueError("answer/label field in canonical trace")
        for doc in row.get("docs", []):
            if isinstance(doc, dict) and forbidden.intersection(doc):
                raise ValueError("answer/label field in canonical document")
        if not isinstance(row.get("docs"), list) or not row["docs"]:
            raise ValueError("canonical request requires a non-empty docs list")
        for doc in row["docs"]:
            if not isinstance(doc, dict):
                raise ValueError("canonical document must be an object")
            for field in ("content_id", "text", "title"):
                if not isinstance(doc.get(field), str) or not doc[field].strip():
                    raise ValueError("canonical document requires non-empty " + field)
            cid, text = doc["content_id"], doc["text"]
            if cid != trace_sha(text.encode("utf-8")):
                raise ValueError("canonical content hash mismatch")
            if cid in texts and texts[cid] != text:
                raise ValueError("canonical content hash collision")
            texts[cid] = text
    return rows


def trace_read_canonical(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return trace_validate_rows(rows), raw


def trace_audit_rows(rows):
    trace_validate_rows(rows)
    seen, prefixes, title_ids = set(), set(), defaultdict(set)
    frequency, histogram = Counter(), Counter()
    accesses = repeats = prefix_repeats = intra = prior_requests = 0
    characters = repeated_characters = prefix_characters = 0
    for row in rows:
        keys, chains, chain, had_prior = [], [], [], False
        for doc in row["docs"]:
            key = doc["content_id"]
            keys.append(key)
            chain.append(key)
            chains.append(tuple(chain))
            accesses += 1
            characters += len(doc["text"])
            frequency[key] += 1
            title_ids[doc["title"]].add(key)
            if key in seen:
                repeats += 1
                repeated_characters += len(doc["text"])
                had_prior = True
                if tuple(chain) in prefixes:
                    prefix_repeats += 1
                    prefix_characters += len(doc["text"])
        intra += len(keys) - len(set(keys))
        prior_requests += int(had_prior)
        histogram[len(keys)] += 1
        # Within-request repetition does not count as a previous-request hit.
        seen.update(keys)
        prefixes.update(chains)
    return {"requests": len(rows), "document_accesses": accesses,
            "unique_exact_documents": len(seen), "unique_titles": len(title_ids),
            "titles_with_multiple_exact_documents": sum(len(value) > 1 for value in title_ids.values()),
            "documents_occurring_more_than_once": sum(value > 1 for value in frequency.values()),
            "max_document_access_count": max(frequency.values()),
            "within_request_duplicate_occurrences": intra,
            "requests_with_prior_request_document": prior_requests,
            "prior_request_document_occurrences": repeats,
            "prior_request_document_occurrence_fraction": repeats / accesses,
            "exact_document_prefix_occurrences": prefix_repeats,
            "exact_document_prefix_occurrence_fraction": prefix_repeats / accesses,
            "nonprefix_document_opportunity_occurrences": repeats - prefix_repeats,
            "rendered_character_accesses": characters,
            "prior_request_document_characters": repeated_characters,
            "exact_document_prefix_characters": prefix_characters,
            "document_count_histogram": dict(sorted(histogram.items())),
            "scope": "unbounded text-repetition opportunity, not bounded-cache hits, validated KV reuse, FLOPs, TTFT or QPS"}


def trace_run(args):
    """Extract answer-free source contexts; preserve source and document order."""
    sources = args.source
    if not sources:
        raise ValueError("at least one --source DATASET=PATH is required")
    output_dir = Path(args.output_dir)
    overwrite = bool(getattr(args, "overwrite", False))
    if output_dir.exists() and not overwrite:
        raise FileExistsError("output directory already exists; choose a new one or pass --overwrite")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output directory is not a directory")
    prepared, names = [], set()
    # Validate every input before writing any output.
    for source in sources:
        dataset, separator, filename = source.partition("=")
        if not separator or not filename or not TRACE_DATASET_NAME.fullmatch(dataset):
            raise ValueError("--source must be a safe DATASET=PATH pair")
        if dataset in names:
            raise ValueError("duplicate dataset name: " + dataset)
        names.add(dataset)
        source_path = Path(filename)
        raw = source_path.read_bytes()
        rows = trace_build_rows(dataset, raw)
        prepared.append((dataset, source_path, raw, rows, trace_encode_jsonl(rows)))
    destinations = [output_dir / (dataset + ".jsonl") for dataset in names] + [output_dir / "manifest.json"]
    input_paths = {source_path.resolve() for _, source_path, _, _, _ in prepared}
    if any(path.resolve() in input_paths for path in destinations):
        raise ValueError("output must not overwrite an input source")
    for destination in destinations:
        check_output(destination, input_paths, overwrite)
    manifest = {"schema": "redknot-answer-free-context-replay-v1",
                "ordering": "source row order and original within-request document order",
                "workload_kind": "dataset QA contexts, NOT production arrivals or executed retrieval",
                "artificial_request_repetition": False, "answer_labels_exported": False,
                "canonicalization": "MuSiQue exact title+newline+paragraph_text; LongBench remove Passage N framing and boundary whitespace only",
                "sources": {}, "limitations": [
                    "No retriever is executed by prepare; supplied evidence/distractor contexts are used.",
                    "File order may reflect dataset construction, not production arrival locality.",
                    "Exact text identity is not proof of context-independent KV correctness.",
                    "Overlapping sources must not be pooled as independent workloads.",
                    "No model quality, inference speed or compute savings are measured."]}
    for dataset, source_path, raw, rows, encoded in prepared:
        target = output_dir / (dataset + ".jsonl")
        trace_write_bytes(target, encoded, overwrite)
        manifest["sources"][dataset] = {"path": str(source_path.resolve()), "sha256": trace_sha(raw),
            "bytes": len(raw), "trace_path": str(target.resolve()), "trace_sha256": trace_sha(encoded),
            "trace_bytes": len(encoded), "audit": trace_audit_rows(rows)}
    trace_write_json(output_dir / "manifest.json", manifest, overwrite)
    print(json.dumps({"event": "prepared", "manifest": str(output_dir / "manifest.json"),
                      "datasets": list(manifest["sources"])}, ensure_ascii=False))
    return manifest


def retrieval_terms(text):
    return RETRIEVAL_WORD.findall(text.casefold())


def retrieval_source_corpus(rows):
    trace_validate_rows(rows)
    by_id = {}
    for row in rows:
        for doc in row["docs"]:
            if doc["content_id"] not in by_id:
                by_id[doc["content_id"]] = {field: doc[field] for field in ("content_id", "text", "title")}
    return list(by_id.values())


class RetrievalBM25:
    """Positive-IDF BM25 with deterministic SHA-256 tie-breaking, not an oracle."""

    def __init__(self, documents, k1=1.2, b=0.75):
        if not documents or not math.isfinite(k1) or k1 < 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("invalid BM25 corpus or parameters")
        self.documents, self.k1, self.b = documents, k1, b
        self.ids = [doc["content_id"] for doc in documents]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate corpus content IDs")
        counts = [Counter(retrieval_terms(doc["text"])) for doc in documents]
        self.lengths = [sum(count.values()) for count in counts]
        self.average_length = sum(self.lengths) / len(self.lengths)
        if not self.average_length:
            raise ValueError("all documents have zero lexical tokens")
        raw = defaultdict(list)
        for index, counts_for_doc in enumerate(counts):
            for term, frequency in counts_for_doc.items():
                raw[term].append((index, frequency))
        self.postings = {}
        for term, entries in raw.items():
            idf = math.log1p((len(documents) - len(entries) + 0.5) / (len(entries) + 0.5))
            self.postings[term] = [(index, idf * frequency * (k1 + 1) /
                (frequency + k1 * (1 - b + b * self.lengths[index] / self.average_length)))
                for index, frequency in entries]
        self.lexical_vocabulary_size = len(self.postings)
        self.posting_count = sum(len(postings) for postings in self.postings.values())

    def scores(self, query):
        scores = [0.0] * len(self.documents)
        for term in sorted(set(retrieval_terms(query))):
            for index, weight in self.postings.get(term, ()):
                scores[index] += weight
        return scores

    def search(self, query, top_k=8):
        if type(top_k) is not int or top_k < 1 or top_k > len(self.documents):
            raise ValueError("top_k must be between 1 and corpus size; no silent truncation")
        scores = self.scores(query)
        indices = heapq.nsmallest(top_k, range(len(scores)), key=lambda index: (-scores[index], self.ids[index]))
        return [(index, scores[index]) for index in indices]


def retrieval_rows(rows, index, top_k):
    for source_row in rows:
        selected = index.search(source_row["query"], top_k)
        docs = [{**index.documents[doc_index], "source_document_index": rank - 1,
                 "corpus_document_index": doc_index, "retrieval_rank": rank,
                 "retrieval_score": score}
                for rank, (doc_index, score) in enumerate(selected, start=1)]
        yield {"request_id": "bm25_top" + str(top_k) + ":" + source_row["request_id"],
               "source_request_id": source_row["request_id"],
               "dataset": source_row["dataset"] + "_bm25_top" + str(top_k),
               "query": source_row["query"], "docs": docs}


def retrieval_run(args):
    """Execute BM25 for every canonical input query; no labels or query filtering."""
    start = time.perf_counter()
    source_path, output_path = Path(args.trace), Path(args.output)
    manifest_path = Path(str(output_path) + ".manifest.json")
    overwrite = bool(getattr(args, "overwrite", False))
    if source_path.resolve() in {output_path.resolve(), manifest_path.resolve()}:
        raise ValueError("retrieval output must not overwrite its input trace")
    for path in (output_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(str(path) + " exists; choose a new path or pass --overwrite")
        if path.exists() and path.is_dir():
            raise ValueError("output path is a directory: " + str(path))
    for destination in (output_path, manifest_path):
        check_output(destination, [source_path], overwrite)
    rows, raw = trace_read_canonical(source_path)
    corpus = retrieval_source_corpus(rows)
    top_k, k1, b = args.top_k, args.k1, args.b
    if type(top_k) is not int or not 1 <= top_k <= len(corpus):
        raise ValueError("top_k must be between 1 and corpus size; no silent truncation")
    index_start = time.perf_counter()
    index = RetrievalBM25(corpus, k1, b)
    index_seconds = time.perf_counter() - index_start
    print(json.dumps({"event": "index_ready", "queries": len(rows), "corpus_documents": len(corpus),
                      "index_seconds": index_seconds}), flush=True)
    retrieval_start = time.perf_counter()
    output = list(retrieval_rows(rows, index, top_k))
    retrieval_seconds = time.perf_counter() - retrieval_start
    encoded = trace_encode_jsonl(output)
    corpus_identity = [(doc["content_id"], len(doc["text"].encode("utf-8"))) for doc in corpus]
    manifest = {"schema": "redknot-executed-bm25-retrieval-v1", "status": "complete",
        "source_path": str(source_path.resolve()), "source_sha256": trace_sha(raw), "source_bytes": len(raw),
        "trace_path": str(output_path.resolve()), "trace_sha256": trace_sha(encoded), "trace_bytes": len(encoded),
        "corpus_documents": len(corpus),
        "corpus_identity_sha256": trace_sha(json.dumps(corpus_identity, separators=(",", ":")).encode("utf-8")),
        "queries": len(rows), "retrieved_documents_per_query": top_k,
        "lexical_tokenizer": {"expression": r"\w+", "preprocess": "str.casefold",
                              "stopwords": False, "stemming": False, "query_tf": "unique terms"},
        "bm25": {"k1": k1, "b": b, "positive_idf": True, "tie_break": "content_id SHA-256 ascending"},
        "vocabulary_size": index.lexical_vocabulary_size, "posting_count": index.posting_count,
        "corpus_lexical_tokens": sum(index.lengths), "corpus_mean_lexical_length": index.average_length,
        "queries_with_zero_score_selected_documents": sum(any(doc["retrieval_score"] == 0 for doc in row["docs"]) for row in output),
        "queries_with_no_positive_score": sum(not any(doc["retrieval_score"] > 0 for doc in row["docs"]) for row in output),
        "audit": trace_audit_rows(output),
        "timing": {"index_seconds": index_seconds, "retrieval_seconds": retrieval_seconds,
                   "total_seconds": time.perf_counter() - start,
                   "scope": "CPU retrieval diagnostics, NOT model inference QPS or TTFT"},
        "limitations": ["Corpus is pooled from the supplied contexts, not a production corpus.",
                        "Real queries are retrieved in file order, without production arrival timestamps.",
                        "Answer/support labels are not used for retrieval or query selection.",
                        "No artificial duplication or popularity sorting is performed.",
                        "Lexical scores are not correctness certificates or measurements of inference speed."]}
    trace_write_bytes(output_path, encoded, overwrite)
    trace_write_json(manifest_path, manifest, overwrite)
    print(json.dumps({"event": "retrieved", "trace": str(output_path), "manifest": str(manifest_path),
                      "queries": len(rows), "trace_sha256": trace_sha(encoded)}), flush=True)
    return manifest


def trace_self_tests():
    """Pure CPU correctness checks; callable from the module's self-test command."""
    source = {"id": "one", "question": "Where?", "answer": "MUST_NOT_EXPORT",
              "paragraphs": [{"title": "Same", "paragraph_text": "First body", "is_supporting": True},
                             {"title": "Same", "paragraph_text": "Second body"}]}
    rows = trace_build_rows("tiny", (json.dumps(source) + "\n").encode())
    assert rows[0]["docs"][0]["content_id"] != rows[0]["docs"][1]["content_id"]
    assert b"MUST_NOT_EXPORT" not in trace_encode_jsonl(rows)
    assert b"is_supporting" not in trace_encode_jsonl(rows)
    assert rows[0]["docs"][0]["text"] == "Same\nFirst body"
    longbench = {"_id": "two", "input": "Question", "answers": ["private"],
                 "context": "Passage 1:\nSame\nFirst body\n\nPassage 2:\nTitle\nA Passage 3: inside a line.\n"}
    lb_rows = trace_build_rows("lb", (json.dumps(longbench) + "\n").encode())
    assert lb_rows[0]["docs"][0]["content_id"] == rows[0]["docs"][0]["content_id"]
    assert lb_rows[0]["docs"][1]["text"] == "Title\nA Passage 3: inside a line."
    assert b'"answers"' not in trace_encode_jsonl(lb_rows)
    assert trace_encode_jsonl(lb_rows) == trace_encode_jsonl(trace_build_rows("lb", (json.dumps(longbench) + "\n").encode()))
    failed = False
    try:
        trace_build_rows("../unsafe", (json.dumps(source) + "\n").encode())
    except ValueError:
        failed = True
    assert failed
    duplicate = [dict(rows[0]), dict(rows[0])]
    failed = False
    try:
        trace_validate_rows(duplicate)
    except ValueError:
        failed = True
    assert failed
    corpus = [trace_make_doc("a", "apple apple pear"), trace_make_doc("b", "pear banana"),
              trace_make_doc("c", "kiwi banana pear")]
    index = RetrievalBM25(corpus)
    counts = [Counter(retrieval_terms(doc["text"])) for doc in corpus]
    reference = []
    for doc_counts in counts:
        score = 0.0
        for term in sorted(set(retrieval_terms("apple pear pear"))):
            tf = doc_counts[term]
            df = sum(term in other for other in counts)
            if tf:
                idf = math.log1p((len(corpus) - df + 0.5) / (df + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * sum(doc_counts.values()) / index.average_length))
        reference.append(score)
    assert all(math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-14)
               for actual, expected in zip(index.scores("apple pear pear"), reference))
    tied = index.search("missing", 3)
    assert [index.ids[i] for i, _ in tied] == sorted(index.ids)
    assert index.scores("apple apple") == index.scores("apple")
    assert all(score == 0 for _, score in tied)
    return {"trace_and_retrieval_self_tests": "passed"}

# ---- Bounded document and token-page prefix policies ----------------------
DEFAULT_FRACTIONS = (0, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1)
DOCUMENT_PREFIX = "[Document]\n"
DOCUMENT_SUFFIX = "\n[/Document]\n"
PROMPT_PREFIX = "Answer the question using the following documents.\n"
QUESTION_PREFIX = "Question: "


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_document(doc: dict) -> str:
    """Title is already part of canonical text; metadata title is not rendered."""
    text = doc.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("each document must contain nonempty text")
    return DOCUMENT_PREFIX + text + DOCUMENT_SUFFIX


@dataclass(frozen=True)
class Document:
    key: str
    tokens: int
    payload_bytes: int


@dataclass(frozen=True)
class Request:
    request_id: str
    dataset: str
    documents: tuple[Document, ...]


def make_document(doc: dict, token_count: int, bytes_per_token: int,
                  page_tokens: int) -> Document:
    if token_count <= 0 or bytes_per_token <= 0 or page_tokens <= 0:
        raise ValueError("token counts, page size, and byte charge must be positive")
    rendered = render_document(doc)
    # The supplied ID is provenance, never a substitute for exact-text identity.
    content_id = doc.get("content_id")
    if isinstance(content_id, str) and len(content_id) == 64:
        try:
            int(content_id, 16)
        except ValueError:
            pass
        else:
            if content_id.lower() != sha256_bytes(doc["text"].encode("utf-8")):
                raise ValueError("content_id does not match canonical text SHA-256")
    charged_tokens = ((token_count + page_tokens - 1) // page_tokens) * page_tokens
    return Document(sha256_bytes(rendered.encode("utf-8")), token_count,
                    charged_tokens * bytes_per_token)


class DocumentLRU:
    """Position-agnostic cache of complete, independently keyed documents."""

    def __init__(self, budget: int):
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        self.budget = budget
        self.entries: OrderedDict[str, int] = OrderedDict()
        self.used = self.peak = self.evictions = self.admissions = self.bypassed = 0

    def lookup(self, docs: tuple[Document, ...]) -> list[bool]:
        # All lookups precede publication: a repeated document within one request
        # cannot turn a miss into a prior-request hit.
        hits = [doc.key in self.entries for doc in docs]
        for doc, hit in zip(docs, hits):
            if hit:
                self.entries.move_to_end(doc.key)
        return hits

    def publish(self, docs: tuple[Document, ...]) -> None:
        seen = set()
        for doc in docs:
            if doc.key in seen:
                continue
            seen.add(doc.key)
            if doc.key in self.entries:
                self.entries.move_to_end(doc.key)
                continue
            if doc.payload_bytes > self.budget:
                self.bypassed += 1
                continue
            while self.used + doc.payload_bytes > self.budget:
                _, size = self.entries.popitem(last=False)
                self.used -= size
                self.evictions += 1
            self.entries[doc.key] = doc.payload_bytes
            self.used += doc.payload_bytes
            self.admissions += 1
            self.peak = max(self.peak, self.used)
        assert self.used <= self.budget


@dataclass
class _Node:
    node_id: int
    key: str
    size: int
    parent: "_Node | None"
    access: int
    children: dict[str, "_Node"] = field(default_factory=dict)
    alive: bool = True


class PrefixTrieLRU:
    """Complete-document exact-prefix trie; only leaf eviction is allowed.

    A node denotes the document AND its complete preceding document chain.
    Thus identical text below different ancestors occupies distinct payload.
    A prefix hit must be a continuous chain from the root. Partial-document
    matches and arbitrary token-level radix sharing are intentionally excluded.
    """

    def __init__(self, budget: int):
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        self.budget = budget
        self.root = _Node(0, "", 0, None, 0)
        self.nodes = {0: self.root}
        self.next_id = 1
        self.clock = 0
        self.leaves: list[tuple[int, int]] = []
        self.used = self.peak = self.evictions = self.admissions = self.bypassed = 0

    def _touch(self, node: _Node) -> None:
        self.clock += 1
        node.access = self.clock
        if node.node_id and not node.children:
            heapq.heappush(self.leaves, (node.access, node.node_id))

    def lookup(self, docs: tuple[Document, ...]) -> list[bool]:
        node = self.root
        hits = []
        missed = False
        for doc in docs:
            child = None if missed else node.children.get(doc.key)
            if child is None:
                missed = True
                hits.append(False)
            else:
                node = child
                self._touch(node)
                hits.append(True)
        return hits

    def _evict_leaf(self, protected: set[int]) -> bool:
        skipped = []
        victim = None
        while self.leaves:
            access, node_id = heapq.heappop(self.leaves)
            node = self.nodes.get(node_id)
            if node is None or node.children or node.access != access:
                continue
            if node_id in protected:
                skipped.append((access, node_id))
                continue
            victim = node
            break
        for item in skipped:
            heapq.heappush(self.leaves, item)
        if victim is None:
            return False
        parent = victim.parent
        assert parent is not None
        del parent.children[victim.key]
        del self.nodes[victim.node_id]
        victim.alive = False
        self.used -= victim.size
        self.evictions += 1
        if parent.node_id and not parent.children:
            heapq.heappush(self.leaves, (parent.access, parent.node_id))
        return True

    def publish(self, docs: tuple[Document, ...]) -> None:
        node = self.root
        protected = {0}
        protected_bytes = 0
        for index, doc in enumerate(docs):
            child = node.children.get(doc.key)
            if child is None:
                # Never discard useful other branches to attempt an impossible
                # admission (this node plus its mandatory ancestors cannot fit).
                if protected_bytes + doc.payload_bytes > self.budget:
                    self.bypassed += len(docs) - index
                    self._compact_heap()
                    return
                while self.used + doc.payload_bytes > self.budget:
                    if not self._evict_leaf(protected):
                        # A suffix cannot be retained without its ancestors.
                        self.bypassed += len(docs) - index
                        self._compact_heap()
                        assert self.used <= self.budget
                        return
                child = _Node(self.next_id, doc.key, doc.payload_bytes, node, 0)
                self.next_id += 1
                node.children[doc.key] = child
                self.nodes[child.node_id] = child
                self.used += doc.payload_bytes
                self.admissions += 1
                self.peak = max(self.peak, self.used)
            node = child
            protected.add(node.node_id)
            protected_bytes += node.size
            self._touch(node)
        self._compact_heap()
        assert self.used <= self.budget

    def _compact_heap(self) -> None:
        # Repeated hits add stale heap entries; keep CPU bookkeeping bounded.
        if len(self.leaves) > max(128, 4 * len(self.nodes)):
            self.leaves = [(n.access, n.node_id) for n in self.nodes.values()
                           if n.node_id and not n.children]
            heapq.heapify(self.leaves)


def policy_state(policy) -> dict:
    return {
        "resident_payload_bytes": policy.used,
        "peak_payload_bytes": policy.peak,
        "resident_units": len(policy.entries) if isinstance(policy, DocumentLRU)
                          else len(policy.nodes) - 1,
        "admissions": policy.admissions,
        "evictions": policy.evictions,
        "bypassed_units": policy.bypassed,
    }


def replay(requests: Iterable[Request], budget: int) -> dict:
    prefix = PrefixTrieLRU(budget)
    document = DocumentLRU(budget)
    counters = {name: {"tokens": 0, "documents": 0} for name in (
        "all", "prefix_hits", "document_hits", "both_hits", "prefix_only_hits",
        "nonprefix_opportunity_hits", "neither_hits", "first_seen",
        "repeated_content", "prefix_repeat_opportunities", "unbounded_nonprefix_opportunities",
        "document_seen_but_not_resident", "prefix_seen_but_not_resident")}
    ever_docs: set[str] = set()
    # A separate unlimited prefix trie measures repetition independent of budget;
    # it stores identities only, not KV payload, and is NOT a serving cache.
    ever_prefix: dict = {}
    request_count = 0
    requests_any_prefix_hit = requests_any_document_hit = 0

    def add(name: str, doc: Document) -> None:
        counters[name]["tokens"] += doc.tokens
        counters[name]["documents"] += 1

    for request in requests:
        request_count += 1
        docs = request.documents
        prefix_hits = prefix.lookup(docs)
        doc_hits = document.lookup(docs)
        requests_any_prefix_hit += int(any(prefix_hits))
        requests_any_document_hit += int(any(doc_hits))
        history_node = ever_prefix
        historical_prefix = []
        chain_exists = True
        for doc in docs:
            if chain_exists and doc.key in history_node:
                history_node = history_node[doc.key]
                historical_prefix.append(True)
            else:
                chain_exists = False
                historical_prefix.append(False)
        for doc, ph, dh, prior_prefix in zip(docs, prefix_hits, doc_hits, historical_prefix):
            add("all", doc)
            if ph:
                add("prefix_hits", doc)
            if dh:
                add("document_hits", doc)
            if ph and dh:
                add("both_hits", doc)
            elif ph:
                add("prefix_only_hits", doc)
            elif dh:
                add("nonprefix_opportunity_hits", doc)
            else:
                add("neither_hits", doc)
            if doc.key in ever_docs:
                add("repeated_content", doc)
                if not dh:
                    add("document_seen_but_not_resident", doc)
                if not prior_prefix:
                    add("unbounded_nonprefix_opportunities", doc)
            else:
                add("first_seen", doc)
            if prior_prefix:
                add("prefix_repeat_opportunities", doc)
                if not ph:
                    add("prefix_seen_but_not_resident", doc)
        prefix.publish(docs)
        document.publish(docs)
        ever_docs.update(d.key for d in docs)
        history_node = ever_prefix
        for doc in docs:
            history_node = history_node.setdefault(doc.key, {})
    for metric in counters.values():
        metric["token_fraction"] = metric["tokens"] / counters["all"]["tokens"] if counters["all"]["tokens"] else 0
        metric["document_fraction"] = metric["documents"] / counters["all"]["documents"] if counters["all"]["documents"] else 0
    return {
        "budget_bytes": budget,
        "request_count": request_count,
        "requests_any_prefix_hit": requests_any_prefix_hit,
        "requests_any_document_hit": requests_any_document_hit,
        "metrics": counters,
        "prefix_policy": policy_state(prefix),
        "document_policy": policy_state(document),
    }


def ordered_requests(requests: list[Request], seed: int | None) -> list[Request]:
    result = list(requests)
    if seed is not None:
        random.Random(seed).shuffle(result)
    return result


def read_trace(path: Path) -> list[dict]:
    records = []
    seen_ids = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            request_id = str(item.get("request_id", ""))
            if not request_id or request_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: missing/duplicate request_id")
            if not isinstance(item.get("query"), str):
                raise ValueError(f"{path}:{line_number}: missing query string")
            if not isinstance(item.get("docs"), list) or not item["docs"]:
                raise ValueError(f"{path}:{line_number}: empty/missing docs")
            if any(k in item for k in ("answer", "answers", "gold_answer", "label")):
                raise ValueError(f"{path}:{line_number}: answer/label field in replay input")
            for doc in item["docs"]:
                render_document(doc)
            seen_ids.add(request_id)
            records.append(item)
    if not records:
        raise ValueError(f"empty trace: {path}")
    trace_validate_rows(records)
    return records


def prepare_trace(records: list[dict], lengths: dict[str, int],
                  bytes_per_token: int, page_tokens: int) -> list[Request]:
    return [Request(str(record["request_id"]), str(record.get("dataset", "unknown")),
                    tuple(make_document(doc, lengths[sha256_bytes(render_document(doc).encode("utf-8"))],
                                        bytes_per_token, page_tokens) for doc in record["docs"]))
            for record in records]


def tokenizer_lengths(record_groups: list[list[dict]], tokenizer_path: Path,
                      cache_path: Path | None) -> tuple[dict[str, int], dict]:
    tokenizer_sha = sha256_bytes(tokenizer_path.read_bytes())
    texts = {}
    for group in record_groups:
        for record in group:
            for doc in record["docs"]:
                rendered = render_document(doc)
                texts[sha256_bytes(rendered.encode("utf-8"))] = rendered
    lengths = {}
    if cache_path and cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved.get("tokenizer_sha256") != tokenizer_sha:
            raise ValueError("token-length cache tokenizer hash mismatch")
        lengths = {k: int(v) for k, v in saved.get("lengths", {}).items()}
        if any(v <= 0 for v in lengths.values()):
            raise ValueError("invalid cached token length")
    missing = [(key, text) for key, text in texts.items() if key not in lengths]
    if missing:
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError("Install the tokenizers package; whitespace token counts are not accepted") from error
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        for start in range(0, len(missing), 64):
            batch = missing[start:start + 64]
            encodings = tokenizer.encode_batch([text for _, text in batch], add_special_tokens=False)
            for (key, _), encoding in zip(batch, encodings):
                lengths[key] = len(encoding.ids)
    if cache_path and (missing or not cache_path.exists()):
        write_json(cache_path, {"tokenizer_sha256": tokenizer_sha, "lengths": lengths},
                   overwrite=cache_path.exists())
    return lengths, {"path": str(tokenizer_path.resolve()), "sha256": tokenizer_sha,
                     "unique_rendered_documents": len(texts), "newly_tokenized_documents": len(missing),
                     "add_special_tokens": False, "truncation": False}


def run_suite(requests: list[Request], fractions: list[float], seeds: list[int]) -> dict:
    unique = {doc.key: doc for request in requests for doc in request.documents}
    corpus_bytes = sum(doc.payload_bytes for doc in unique.values())
    runs = []
    for seed in [None, *seeds]:
        sequence = ordered_requests(requests, seed)
        order_sha = sha256_bytes(json.dumps([r.request_id for r in sequence], separators=(",", ":")).encode())
        for fraction in fractions:
            budget = math.floor(corpus_bytes * fraction)
            result = replay(sequence, budget)
            result.update({"budget_corpus_fraction": fraction,
                           "order": "source_order" if seed is None else "seeded_shuffle",
                           "shuffle_seed": seed, "request_order_sha256": order_sha})
            runs.append(result)
    summaries = []
    for fraction in fractions:
        selected = [r for r in runs if r["budget_corpus_fraction"] == fraction and r["shuffle_seed"] is not None]
        if selected:
            summary = {"budget_corpus_fraction": fraction, "shuffle_count": len(selected), "metrics": {}}
            for name in ("prefix_hits", "document_hits", "nonprefix_opportunity_hits", "repeated_content"):
                values = [r["metrics"][name]["token_fraction"] for r in selected]
                summary["metrics"][name] = {"mean": mean(values), "min": min(values), "max": max(values)}
            summaries.append(summary)
    return {
        "requests": len(requests), "document_occurrences": sum(len(r.documents) for r in requests),
        "unique_documents": len(unique), "unique_document_tokens": sum(d.tokens for d in unique.values()),
        "unique_paged_corpus_payload_bytes": corpus_bytes,
        "datasets": sorted({r.dataset for r in requests}),
        "runs": runs, "shuffle_summary": summaries,
    }


def page_documents(token_ids: list[int], page_tokens: int, bytes_per_token: int):
    if page_tokens <= 0 or bytes_per_token <= 0:
        raise ValueError("positive page size and byte charge required")
    pages = []
    complete = len(token_ids) // page_tokens
    for index in range(complete):
        chunk = token_ids[index * page_tokens:(index + 1) * page_tokens]
        if any(token < 0 or token > 0xffffffff for token in chunk):
            raise ValueError("token ID outside unsigned 32-bit range")
        key = hashlib.sha256(struct.pack("<" + "I" * page_tokens, *chunk)).hexdigest()
        pages.append(Document(key, page_tokens, page_tokens * bytes_per_token))
    return tuple(pages), len(token_ids) - complete * page_tokens


def replay_prefix(requests: list[Request], budget: int, denominator_tokens: int):
    policy = PrefixTrieLRU(budget)
    hits = pages = requests_hit = 0
    for request in requests:
        hit_mask = policy.lookup(request.documents)
        hit_tokens = sum(doc.tokens for doc, hit in zip(request.documents, hit_mask) if hit)
        hits += hit_tokens
        requests_hit += int(any(hit_mask))
        pages += len(request.documents)
        policy.publish(request.documents)
    return {
        "prefix_hit_tokens": hits,
        "prefix_token_fraction": hits / denominator_tokens if denominator_tokens else 0,
        "requests_any_prefix_hit": requests_hit,
        "page_occurrences": pages,
        "denominator_tokens": denominator_tokens,
        "policy": policy_state(policy),
    }


def build_requests(records, tokenizer_path: Path, page_tokens: int, bytes_per_token: int):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    rendered = {}
    for record in records:
        for doc in record["docs"]:
            text = render_document(doc)
            key = sha256_bytes(text.encode())
            rendered[key] = text
    tokenized = {}
    items = list(rendered.items())
    for start in range(0, len(items), 64):
        batch = items[start:start + 64]
        encoded = tokenizer.encode_batch([text for _, text in batch], add_special_tokens=False)
        for (key, _), enc in zip(batch, encoded):
            tokenized[key] = enc.ids
    requests = []
    total_tokens = dropped_tokens = max_dropped = 0
    for record in records:
        ids = []
        for doc in record["docs"]:
            ids.extend(tokenized[sha256_bytes(render_document(doc).encode())])
        pages, discarded = page_documents(ids, page_tokens, bytes_per_token)
        total_tokens += len(ids)
        dropped_tokens += discarded
        max_dropped = max(max_dropped, discarded)
        requests.append(Request(str(record["request_id"]), str(record["dataset"]), pages))
    return requests, {
        "requests": len(requests), "document_token_occurrences": total_tokens,
        "complete_page_tokens": total_tokens - dropped_tokens,
        "discarded_final_partial_tokens": dropped_tokens,
        "max_discarded_per_request": max_dropped,
        "discarded_token_fraction": dropped_tokens / total_tokens if total_tokens else 0,
        "unique_rendered_documents": len(rendered),
    }


# ---- Real HeadKVManager / SegPaged COW residency --------------------------
MEM_CONTRACT = "rag-motivation-unencoded-synthetic-kv-v1"
MEM_CERTIFICATE = (
    "Synthetic unencoded K/V fixture: local payload is supplied identical across "
    "requests and global payload differs. This tests physical storage/COW only; "
    "it does not certify real-model non-prefix reuse or positional equivalence."
)


def mem_load_runtime(repo_root):
    """Load only the selected repository's head_kv package, not its server."""
    root = Path(repo_root).resolve() / "python/sglang/srt/mem_cache"
    if not (root / "head_kv/manager.py").is_file():
        raise FileNotFoundError(f"head_kv runtime missing under {root}")
    existing = sys.modules.get("head_kv")
    if existing is not None and Path(existing.__file__).resolve().parent != root / "head_kv":
        raise RuntimeError("a different head_kv runtime was already imported; use a fresh process")
    sys.path.insert(0, str(root))
    try:
        import head_kv
    finally:
        sys.path.remove(str(root))
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted((root / "head_kv").glob("*.py"))}
    return head_kv, hashes


def mem_api_selection(runtime):
    supports_batch = "batched_writes" in inspect.signature(
        runtime.HeadKVManager.begin_update).parameters
    return {
        "writes": "batched_writes" if supports_batch else "legacy_writes",
        "sharing": ("share_segments_many" if callable(getattr(
            runtime.HeadKVManager, "share_segments_many", None)) else "share_segment"),
        "proof_kind": "policy_approximate",
        "fallback_note": "Compatibility fallback preserves proof and positions; this benchmark reports memory, not publication latency.",
    }


def mem_update(manager, request_id, **kwargs):
    if "writes" in kwargs and "batched_writes" in inspect.signature(
            manager.begin_update).parameters:
        kwargs["batched_writes"] = True
    return manager.update(request_id, **kwargs)


def mem_share_document(manager, target_id, shares):
    if callable(getattr(manager, "share_segments_many", None)):
        return manager.share_segments_many(target_id, [("source-document", shares, True)])
    for source_key, target_key, proof, positions in shares:
        manager.share_segment("source-document", target_id, source_key, target_key,
                              proof=proof, positions=positions, source_is_cache=True)


def mem_validate_args(args):
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        raise ValueError("memory supports cpu or explicit cuda[:N] only; other accelerators are not measured")
    for name in ("layers", "kv_heads", "head_dim", "document_tokens", "prefix_tokens", "page_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.local_heads <= args.kv_heads:
        raise ValueError("local-heads must be in [0, kv-heads]")
    if min(args.protected_first, args.protected_last) < 0 or args.protected_first + args.protected_last > args.layers:
        raise ValueError("invalid protected boundary layer counts")
    if not 0 <= args.cow_rows <= args.document_tokens:
        raise ValueError("cow-rows must be in [0, document-tokens]")
    if not args.multiplicities or min(args.multiplicities) <= 0 or len(set(args.multiplicities)) != len(args.multiplicities):
        raise ValueError("multiplicities must be distinct positive integers")
    if args.fixed_capacity_pages is not None and args.fixed_capacity_pages <= 0:
        raise ValueError("fixed-capacity-pages must be positive")


def mem_cleanup(manager):
    """Reclaim every owned root, including publisher roots on error paths."""
    failures = []
    for transaction in list(manager._pending.values()):
        try:
            transaction.abort()
        except BaseException as exc:
            failures.append(exc)
    for request_id in list(manager._requests):
        try:
            manager.release_request(request_id)
        except BaseException as exc:
            failures.append(exc)
    for cache_key in list(manager._cache):
        try:
            manager.evict(cache_key)
        except BaseException as exc:
            failures.append(exc)
    try:
        mem_synchronize(manager.pool.device)
        manager.pool.collect()
        cleaned = manager.pool.stats()
        if any(cleaned[key] for key in (
                "live_pages", "reserved_pages", "retiring_pages", "pinned_pages")):
            failures.append(AssertionError("page ownership leaked after cleanup"))
    except BaseException as exc:
        failures.append(exc)
    if failures:
        raise RuntimeError("memory benchmark cleanup failed: " +
                           "; ".join(str(exc) for exc in failures)) from failures[0]


def mem_local_pairs(args):
    return {(layer, head) for layer in range(args.protected_first, args.layers - args.protected_last)
            for head in range(args.local_heads)}


def mem_planned_capacity(args, count, arm):
    """Capacity planning only. Results are counted from actual published refs."""
    n_local = len(mem_local_pairs(args))
    n_all = args.layers * args.kv_heads
    doc_pages = math.ceil(args.document_tokens / args.page_size)
    prefix_pages = math.ceil(args.prefix_tokens / args.page_size)
    cow_pages = math.ceil(args.cow_rows / args.page_size)
    if arm == "head_shared":
        before = n_local * doc_pages + count * ((n_all - n_local) * doc_pages + n_all * prefix_pages)
        after = before + count * n_local * cow_pages
    elif arm == "independent":
        before = after = count * n_all * (doc_pages + prefix_pages)
    else:
        raise ValueError("unknown arm")
    reserve = n_local * cow_pages
    needed = max(before, after) + reserve
    capacity = args.fixed_capacity_pages if args.fixed_capacity_pages is not None else needed
    if capacity < needed:
        raise ValueError(f"fixed capacity {capacity} is below declared conservative requirement {needed}")
    return capacity, reserve


def mem_synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.mem_synchronize(device)


def mem_allocator_snapshot(device):
    if torch.device(device).type != "cuda":
        return None
    mem_synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def mem_allocator_delta(after, before):
    if after is None:
        return None
    return {key: value - before[key] for key, value in after.items()}


def mem_python_metadata_size(*roots):
    """Deduplicated reachable CPython object size, never claimed as total RSS.

    Tensor payload storage is excluded. Native CUDA events/locks and other
    extension objects contribute only their reported Python wrapper size.
    Traversal is restricted to built-in containers and head_kv instances.
    """
    seen = set()
    def visit(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        result = sys.getsizeof(value)
        if torch.is_tensor(value):
            return result
        if isinstance(value, Mapping):
            return result + sum(visit(k) + visit(v) for k, v in value.items())
        if isinstance(value, (list, tuple, set, frozenset)):
            return result + sum(visit(v) for v in value)
        if type(value).__module__.startswith("head_kv") and hasattr(value, "__dict__"):
            return result + visit(vars(value))
        return result
    return sum(visit(root) for root in roots)


def mem_descriptor_storage_bytes(adapters):
    seen = set()
    by_device = Counter()
    logical = 0
    for adapter in adapters:
        for _, descriptor, _ in adapter._descriptor_cache.values():
            for name in ("page_slots", "page_lengths", "key_positions"):
                tensor = descriptor[name]
                logical += tensor.numel() * tensor.element_size()
                storage = tensor.untyped_storage()
                identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
                if identity not in seen:
                    seen.add(identity)
                    by_device[tensor.device.type] += storage.nbytes()
    return {"logical_tensor_bytes": logical,
            "cuda_storage_bytes": by_device["cuda"], "cpu_storage_bytes": by_device["cpu"]}


def mem_footprint_snapshot(manager, request_ids, adapters=()):
    mem_synchronize(manager.pool.device)
    manager.pool.collect()
    request_counts = Counter()
    document_refs, prefix_refs = set(), set()
    for request_id in request_ids:
        root = manager.version(request_id)
        request_counts.update({ref for seg in root.segments.values() for ref in seg.pages})
        for key, seg in root.segments.items():
            (document_refs if key[2] == "document" else prefix_refs).update(seg.pages)
    cached_refs = {ref for root in manager._cache.values() for seg in root.segments.values() for ref in seg.pages}
    all_refs = set(request_counts) | cached_refs
    stats = manager.stats()
    if (stats["live_pages"] != len(all_refs) or stats["reserved_pages"]
            or stats["retiring_pages"] or stats["pinned_pages"]):
        raise AssertionError("unaccounted live or in-flight pages after synchronization")
    page_bytes = manager.pool.page_bytes
    private_pages = sum(n == 1 and ref not in cached_refs for ref, n in request_counts.items())
    single_request_cache_shared = sum(n == 1 and ref in cached_refs for ref, n in request_counts.items())
    return {
        "pool": stats,
        "live_owned_page_count": len(all_refs),
        "live_owned_page_bytes": len(all_refs) * page_bytes,
        "active_shared_pages": sum(n >= 2 for n in request_counts.values()),
        "active_private_pages": private_pages,
        "single_request_cache_shared_pages": single_request_cache_shared,
        "active_shared_page_bytes": sum(n >= 2 for n in request_counts.values()) * page_bytes,
        "active_private_page_bytes": private_pages * page_bytes,
        "single_request_cache_shared_page_bytes": single_request_cache_shared * page_bytes,
        "cache_only_pages": len(cached_refs - set(request_counts)),
        "cache_only_page_bytes": len(cached_refs - set(request_counts)) * page_bytes,
        "cache_and_request_shared_pages": len(cached_refs & set(request_counts)),
        "document_unique_pages": len(document_refs),
        "prefix_unique_pages": len(prefix_refs),
        "logical_active_page_references": sum(request_counts.values()),
        "descriptor_metadata": mem_descriptor_storage_bytes(adapters),
        "python_metadata_deep_size_bytes": mem_python_metadata_size(manager, *adapters),
    }


def mem_make_payload(args, layer, head, request_index, rows, kind):
    # CPU fixture allocation is deliberately outside GPU slab residency. Each
    # tensor is supplied, copied and then discarded; no model forward is implied.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + rows)
    base = torch.rand((rows, args.head_dim), generator=generator, dtype=torch.float32)
    offset = layer * 0.0001 + head * 0.01
    if kind != "local":
        offset += (request_index + 1) * 0.125
    if kind == "patch":
        offset += 2.0
    k = (base + offset).to(getattr(torch, args.dtype))
    v = (-base - offset).to(getattr(torch, args.dtype))
    return k, v


def mem_verify_pages(manager, args, request_ids, shared):
    """Read actual physical pages; verify patched and untouched sample rows."""
    checked = 0
    local = mem_local_pairs(args)
    # Per segment, inspect the first, patch-boundary, page-boundary and last row.
    # No historical gather or extra full GPU payload survives this verification.
    for i, request_id in enumerate(request_ids):
        root = manager.version(request_id)
        for (layer, head, occurrence), seg in root.segments.items():
            n = args.document_tokens if occurrence == "document" else args.prefix_tokens
            kind = "local" if occurrence == "document" and (layer, head) in local else "private"
            expected_k, expected_v = mem_make_payload(args, layer, head, i, n, kind)
            if occurrence == "document" and (layer, head) in local and args.cow_rows:
                patch_k, patch_v = mem_make_payload(args, layer, head, i, args.cow_rows, "patch")
                expected_k[:args.cow_rows], expected_v[:args.cow_rows] = patch_k, patch_v
            rows = sorted({0, min(n - 1, args.page_size), min(n - 1, args.cow_rows), n - 1})
            for row in rows:
                page = seg.pages[row // args.page_size]
                k, v = manager.pool.read_page(page)
                torch.testing.assert_close(k[row % args.page_size].cpu(), expected_k[row], rtol=0, atol=0)
                torch.testing.assert_close(v[row % args.page_size].cpu(), expected_v[row], rtol=0, atol=0)
                checked += 1
    if shared and local:
        source = manager._cache["source-document"]
        for (layer, head, _), seg in source.segments.items():
            k, v = manager.pool.read_page(seg.pages[0])
            expected_k, expected_v = mem_make_payload(args, layer, head, 0, args.document_tokens, "local")
            torch.testing.assert_close(k[0].cpu(), expected_k[0], rtol=0, atol=0)
            torch.testing.assert_close(v[0].cpu(), expected_v[0], rtol=0, atol=0)
            checked += 1
    return checked


def mem_run_arm(args, runtime, count, arm):
    mem_validate_args(args)
    capacity, transaction_reserve = mem_planned_capacity(args, count, arm)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing fake GPU measurements")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline_allocator = mem_allocator_snapshot(device)
    started = time.perf_counter()
    pool = runtime.HeadPagePool(capacity, args.page_size, args.head_dim,
                                dtype=getattr(torch, args.dtype), device=device)
    manager = runtime.HeadKVManager(pool)
    request_ids = []
    adapters = []
    local = mem_local_pairs(args)
    source_refs = set()
    cow_new_refs = set()
    try:
        if arm == "head_shared" and local:
            manager.create_request("publisher", context_id="offline-document", contract=MEM_CONTRACT)
            positions = tuple(range(args.document_tokens))
            for layer, head in sorted(local):
                k, v = mem_make_payload(args, layer, head, 0, args.document_tokens, "local")
                mem_update(manager, "publisher", writes=(runtime.SegmentWrite(
                    (layer, head, "document"), k, v, positions, "supplied-local-fixture"),))
            manager.cache("publisher", "source-document")
            manager.release_request("publisher")
            source_refs = {ref for seg in manager._cache["source-document"].segments.values() for ref in seg.pages}
        for index in range(count):
            rid = f"request-{index}"
            request_ids.append(rid)
            context = f"distinct-prefix-{index}"
            manager.create_request(rid, context_id=context, contract=MEM_CONTRACT)
            prefix_positions = tuple(range(args.prefix_tokens))
            # Different preceding text and logical offsets; unencoded synthetic
            # keys deliberately do not purport to implement RoPE relocation.
            offset = args.prefix_tokens + index % 3
            positions = tuple(range(offset, offset + args.document_tokens))
            for layer in range(args.layers):
                writes = []
                for head in range(args.kv_heads):
                    k, v = mem_make_payload(args, layer, head, index, args.prefix_tokens, "private")
                    writes.append(runtime.SegmentWrite((layer, head, "prefix"), k, v, prefix_positions, context))
                    if arm == "head_shared" and (layer, head) in local:
                        continue
                    kind = "local" if (layer, head) in local else "private"
                    k, v = mem_make_payload(args, layer, head, index, args.document_tokens, kind)
                    writes.append(runtime.SegmentWrite((layer, head, "document"), k, v, positions, context))
                mem_update(manager, rid, writes=tuple(writes))
            if arm == "head_shared" and local:
                proof = runtime.ReuseProof("policy_approximate", "supplied-local-fixture", context, MEM_CERTIFICATE)
                shares = [((layer, head, "document"), (layer, head, "document"), proof, positions)
                          for layer, head in sorted(local)]
                mem_share_document(manager, rid, shares)
        before_cow = mem_footprint_snapshot(manager, request_ids)
        copied_before, written_before = pool.copied_bytes, pool.written_bytes
        for index, rid in enumerate(request_ids):
            if not local or not args.cow_rows:
                continue
            previous = manager.version(rid)
            patches = []
            for layer, head in sorted(local):
                k, v = mem_make_payload(args, layer, head, index, args.cow_rows, "patch")
                patches.append(runtime.SegmentPatch((layer, head, "document"), tuple(range(args.cow_rows)),
                                                    k, v, f"request-specific-repair-{index}"))
            current = manager.update(rid, patches=tuple(patches))
            for key, seg in current.segments.items():
                if key[2] == "document" and key[:2] in local:
                    cow_new_refs.update(set(seg.pages) - set(previous.segments[key].pages))
        after_cow = mem_footprint_snapshot(manager, request_ids)
        checked_rows = mem_verify_pages(manager, args, request_ids, arm == "head_shared")
        # Build and retain the exact manager-backed SegPaged descriptor cache for
        # every active request/layer. This includes per-token position metadata.
        for rid in request_ids:
            adapter = runtime.ManagedSegPagedKVCache(manager, rid, num_layers=args.layers, num_kv_heads=args.kv_heads)
            with manager.bind(rid) as lease:
                for layer in range(args.layers):
                    adapter._descriptor(lease, layer)
            adapters.append(adapter)
        # Python references to temporary payload tensors are CPU-only, but remove
        # them from this scope's reporting rationale explicitly via collection.
        mem_synchronize(device)
        gc.collect()
        resident = mem_footprint_snapshot(manager, request_ids, adapters)
        alloc_after = mem_allocator_snapshot(device)
        if cow_new_refs & source_refs:
            raise AssertionError("COW replaced state in-place in the source cache")
        expected_new = count * len(local) * math.ceil(args.cow_rows / args.page_size)
        if len(cow_new_refs) != expected_new:
            raise AssertionError("unexpected COW replacement-page count")
        result = {
            "arm": arm, "active_requests": count,
            "capacity_policy": "fixed_capacity" if args.fixed_capacity_pages else "workload_sized_slab_with_transaction_reserve",
            "transaction_reserve_pages": transaction_reserve,
            "before_cow": before_cow, "after_cow": after_cow, "resident_with_descriptors": resident,
            "cow_new_private_pages": len(cow_new_refs),
            "cow_copy_bytes": pool.copied_bytes - copied_before,
            "cow_write_bytes": pool.written_bytes - written_before,
            "source_immutable_verified": bool(arm == "head_shared" and local),
            "sampled_payload_rows_verified": checked_rows,
            "torch_cuda_before": baseline_allocator,
            "torch_cuda_after": alloc_after,
            "torch_cuda_delta": mem_allocator_delta(alloc_after, baseline_allocator),
            "wall_seconds_not_serving_latency": time.perf_counter() - started,
            "model_accuracy": None, "qps": None, "ttft": None,
        }
    finally:
        original_exception = sys.exc_info()[1]
        adapters.clear()
        try:
            mem_cleanup(manager)
        except BaseException as cleanup_error:
            if original_exception is None:
                raise
            if hasattr(original_exception, "add_note"):
                original_exception.add_note(str(cleanup_error))
    result["cleanup_all_pages_reclaimed"] = True
    return result


def mem_pair_summary(independent, shared):
    a = independent["resident_with_descriptors"]
    b = shared["resident_with_descriptors"]
    def saving(x, y):
        return 1 - y / x if x else None
    result = {
        "active_requests": independent["active_requests"],
        "occupied_kv_page_saving_fraction": saving(a["live_owned_page_bytes"], b["live_owned_page_bytes"]),
        "slab_capacity_saving_fraction": saving(a["pool"]["reserved_slab_bytes"], b["pool"]["reserved_slab_bytes"]),
        "descriptor_metadata_bytes_independent": a["descriptor_metadata"],
        "descriptor_metadata_bytes_shared": b["descriptor_metadata"],
    }
    if independent["torch_cuda_delta"] is not None:
        result["torch_allocated_delta_saving_fraction"] = saving(
            independent["torch_cuda_delta"]["allocated_bytes"], shared["torch_cuda_delta"]["allocated_bytes"])
        result["torch_reserved_delta_saving_fraction"] = saving(
            independent["torch_cuda_delta"]["reserved_bytes"], shared["torch_cuda_delta"]["reserved_bytes"])
    else:
        result["torch_allocated_delta_saving_fraction"] = None
        result["torch_reserved_delta_saving_fraction"] = None
    return result



def mem_run(args):
    """Measure actual storage; importing cache replay never imports torch."""
    global torch
    mem_validate_args(args)
    args.output = Path(args.output)
    if args.output.exists() and not getattr(args, "overwrite", False):
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    import torch
    runtime, hashes = mem_load_runtime(args.repo_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing fake GPU measurements")
    torch.set_num_threads(1)
    payload = {
        "schema": "rag-motivation-actual-page-residency-v1", "complete": False,
        "layout": {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in vars(args).items() if not callable(value)},
        "head_mask_source": "explicit_hypothetical_MHA_GQA_layout_NOT_a_model_profile",
        "local_layer_head_pairs": sorted(mem_local_pairs(args)),
        "head_kv_source_sha256": hashes,
        "runtime_api": mem_api_selection(runtime),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scope": MEM_CERTIFICATE,
        "host_accounting": "Deduplicated reachable CPython objects only; excludes tensor storage, native extension allocations, Python allocator arena slack and process RSS. GPU descriptor storage is exact and separate.",
        "allocator_accounting": "torch allocated/reserved deltas measure this isolated benchmark process, not whole-device nvidia-smi. Slabs are preallocated. Fixed-capacity sharing lowers occupancy, not slab allocation. Workload-sized slabs show the capacity required for the declared workload plus transaction reserve, not dynamic allocator shrinkage.",
        "records": [], "comparisons": [],
    }
    # The first write claims this result path. Later writes only checkpoint the
    # same claimed result; they do not silently replace another prior result.
    write_json(args.output, payload, overwrite=getattr(args, "overwrite", False))
    try:
        for count in args.multiplicities:
            pair = []
            for arm in ("independent", "head_shared"):
                print(json.dumps({"event": "starting", "requests": count, "arm": arm}), flush=True)
                record = mem_run_arm(args, runtime, count, arm)
                payload["records"].append(record)
                pair.append(record)
                write_json(args.output, payload, overwrite=True)
                print(json.dumps({"event": "completed", "requests": count, "arm": arm,
                                  "live_owned_page_bytes": record["resident_with_descriptors"]["live_owned_page_bytes"]}), flush=True)
            payload["comparisons"].append(mem_pair_summary(*pair))
            write_json(args.output, payload, overwrite=True)
        payload["complete"] = True
        write_json(args.output, payload, overwrite=True)
    except BaseException as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(args.output, payload, overwrite=True)
        raise
    return 0


def mem_smoke_test(repo_root):
    """Three CPU fixtures: no sharing, partial sharing/COW, all heads shared.

    Each fixture checks concurrent multiplicities 1 and 2. No CUDA initialization,
    model loading, network calls, or GPU-holder process changes are performed.
    """
    global torch
    import torch
    runtime, hashes = mem_load_runtime(repo_root)
    torch.set_num_threads(1)
    records = []
    for local_heads in (0, 1, 2):
        args = argparse.Namespace(
            repo_root=Path(repo_root), device="cpu", dtype="float32", layers=2,
            kv_heads=2, local_heads=local_heads, protected_first=0, protected_last=0,
            head_dim=4, document_tokens=9, prefix_tokens=3, page_size=4, cow_rows=1,
            multiplicities=[1, 2], fixed_capacity_pages=None, seed=20261022)
        mem_validate_args(args)
        for count in args.multiplicities:
            independent = mem_run_arm(args, runtime, count, "independent")
            shared = mem_run_arm(args, runtime, count, "head_shared")
            a = independent["resident_with_descriptors"]
            b = shared["resident_with_descriptors"]
            n_local = 2 * local_heads
            assert a["live_owned_page_count"] == count * 4 * 4
            assert b["live_owned_page_count"] == n_local * 3 + count * (
                (4 - n_local) * 3 + 4 + n_local)
            for record in (independent, shared):
                footprint = record["resident_with_descriptors"]
                assert footprint["live_owned_page_count"] == sum(
                    footprint[key] for key in (
                        "active_shared_pages", "active_private_pages",
                        "single_request_cache_shared_pages", "cache_only_pages"))
                assert record["cleanup_all_pages_reclaimed"]
                assert record["torch_cuda_delta"] is None
                assert record["model_accuracy"] is None and record["qps"] is None
            assert b["descriptor_metadata"] == a["descriptor_metadata"]
            assert b["descriptor_metadata"]["cpu_storage_bytes"] > 0
            assert b["descriptor_metadata"]["cuda_storage_bytes"] == 0
            assert shared["cow_new_private_pages"] == count * n_local
            if n_local:
                assert shared["source_immutable_verified"]
                assert shared["cow_copy_bytes"] == count * n_local * 3 * 4 * 4 * 2
            else:
                assert a["live_owned_page_count"] == b["live_owned_page_count"]
            records.append({"local_heads": local_heads, "active_requests": count,
                            "checks_passed": True})
    return {"fixtures": 3, "comparisons": records, "runtime_api": mem_api_selection(runtime),
            "head_kv_source_sha256": hashes}


# ---- Scientific plots ---------------------------------------------------
def plot_run(args):
    """Render observed cache opportunities and page residency; never infer speed."""
    import json
    from io import BytesIO
    from pathlib import Path

    inputs = [Path(p).resolve() for p in (args.cache, args.memory) if p is not None]
    if not inputs:
        raise ValueError("plot requires --cache and/or --memory")
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    if not formats or len(set(formats)) != len(formats) or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("formats must be distinct png,pdf,svg values")
    cache = json.loads(Path(args.cache).read_text()) if args.cache else None
    memory = json.loads(Path(args.memory).read_text()) if args.memory else None
    if cache is not None and (cache.get("complete") is False or not cache.get("traces")):
        raise ValueError("cache result is incomplete or has no traces")
    if memory is not None and (not memory.get("complete") or not memory.get("records")):
        raise ValueError("memory result is incomplete or has no records")
    names = (["cache_budget"] if cache is not None else []) + (["memory_residency"] if memory is not None else [])
    output_dir = Path(args.output_dir).resolve()
    paths = [output_dir / (name + "." + ext) for name in names for ext in formats]
    for path in paths:
        check_output(path, inputs, args.overwrite)
        if path.resolve() in inputs or any(path.exists() and path.samefile(p) for p in inputs):
            raise ValueError(f"plot output would overwrite an input: {path}")
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise FileExistsError(f"plot output exists (use --overwrite for a file): {path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plot requires matplotlib; install it in the analysis environment") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"document_hits": "#2264B4", "prefix_hits": "#E5A119", "page": "#637386"}
    def save(fig, name):
        try:
            for ext in formats:
                destination = output_dir / (name + "." + ext)
                check_output(destination, inputs, args.overwrite)
                with BytesIO() as buffer:
                    fig.savefig(buffer, format=ext, dpi=180, bbox_inches="tight")
                    write_bytes(destination, buffer.getvalue(), args.overwrite)
        finally:
            plt.close(fig)
    def curve(ax, rows, getter, label, color, shuffled=False):
        if not rows:
            return
        rows = sorted(rows, key=lambda r: r["budget_corpus_fraction"])
        x = [100 * r["budget_corpus_fraction"] for r in rows]
        values = [getter(r) for r in rows]
        y = [100 * (v["mean"] if shuffled else v) for v in values]
        ax.plot(x, y, "o-", color=color, lw=1.8, ms=3, label=label)
        if shuffled:
            ax.fill_between(x, [100 * v["min"] for v in values],
                            [100 * v["max"] for v in values], color=color, alpha=.14)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"}):
        if cache is not None:
            fig, axes = plt.subplots(len(cache["traces"]), 2, squeeze=False,
                                     figsize=(12, max(4, 3.6 * len(cache["traces"]))))
            for row, trace in enumerate(cache["traces"]):
                title = ", ".join(trace.get("datasets", [])) or f"Trace {row + 1}"
                control = trace.get("token_prefix_control") or {}
                for col, shuffled in enumerate((False, True)):
                    ax = axes[row][col]
                    rows = trace.get("shuffle_summary", []) if shuffled else [
                        r for r in trace["runs"] if r["order"] == "source_order"]
                    for key, label in (("document_hits", "Document opportunity"), ("prefix_hits", "Document-prefix hit")):
                        curve(ax, rows, lambda r, k=key: r["metrics"][k] if shuffled else r["metrics"][k]["token_fraction"],
                              label, colors[key], shuffled)
                    page_rows = control.get("shuffle_summary", []) if shuffled else [
                        r for r in control.get("runs", []) if r["order"] == "source_order"]
                    curve(ax, page_rows, lambda r: r["metrics"]["prefix_token_fraction"] if shuffled else r["prefix_token_fraction"],
                          "Token-page prefix hit", colors["page"], shuffled)
                    ax.set(title=title + (" / shuffled mean" if shuffled else " / source order"),
                           xlabel="Cache budget / observed unique KV working set (%)",
                           ylabel="Document-token hit / opportunity (%)", ylim=(0, 100))
                    ax.grid(alpha=.2)
                    if rows or page_rows:
                        ax.legend(fontsize=8, frameon=False)
                    else:
                        ax.text(.5, .5, "No shuffled runs recorded", ha="center", transform=ax.transAxes)
            fig.suptitle("Bounded-cache reuse opportunities", fontsize=14)
            fig.text(.5, .005, "Shaded range: min–max across recorded shuffles, not confidence intervals. "
                     "Text recurrence does not certify KV correctness or inference speed.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .025, 1, .96))
            save(fig, "cache_budget")
        if memory is not None:
            records = {(r["active_requests"], r["arm"]): r for r in memory["records"]}
            counts = sorted({n for n, _ in records})
            if len(records) != len(memory["records"]) or any((n, arm) not in records for n in counts for arm in ("independent", "head_shared")):
                raise ValueError("memory results must contain exactly one paired record per request count")
            fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(17, 5.4))
            cuda = all(r.get("torch_cuda_delta") is not None for r in records.values())
            for arm, color, label in (("independent", colors["document_hits"], "Independent"), ("head_shared", colors["prefix_hits"], "Head sharing + COW")):
                selected = [records[n, arm] for n in counts]
                ax.plot(counts, [r["resident_with_descriptors"]["live_owned_page_bytes"] / 2**30 for r in selected], "o-", color=color, label=label)
                if cuda:
                    bx.plot(counts, [r["torch_cuda_delta"]["allocated_bytes"] / 2**30 for r in selected], "o-", color=color, label=label)
            ax.set(title="Occupied physical KV pages", ylabel="Owned page payload (GiB)")
            bx.set(title="CUDA allocator delta" if cuda else "CUDA allocation not measured", ylabel="Allocated bytes (GiB)" if cuda else "")
            if not cuda:
                bx.text(.5, .5, "CPU run or incomplete CUDA accounting\nNo GPU allocation or speed inferred", ha="center", va="center", transform=bx.transAxes)
            stacks = (("active_shared_page_bytes", "Multiple active requests", "#E5A119"),
                      ("single_request_cache_shared_page_bytes", "Cache + one active request", "#8CB6D5"),
                      ("active_private_page_bytes", "Private pages (including COW)", "#2264B4"),
                      ("cache_only_page_bytes", "Cache-only source pages", "#529A7C"))
            x, bottom = list(range(len(counts))), [0.] * len(counts)
            for key, label, color in stacks:
                y = [records[n, "head_shared"]["resident_with_descriptors"][key] / 2**30 for n in counts]
                cx.bar(x, y, bottom=bottom, color=color, label=label)
                bottom = [a + b for a, b in zip(bottom, y)]
            cx.set(title="Shared-arm ownership breakdown", ylabel="Owned page payload (GiB)")
            for panel in (ax, bx, cx):
                panel.set_xlabel("Simultaneously resident requests")
                panel.set_ylim(bottom=0)
                panel.grid(axis="y", alpha=.2)
                panel.set_axisbelow(True)
            ax.set_xticks(counts); bx.set_xticks(counts); cx.set_xticks(x, labels=counts)
            ax.legend(fontsize=8, frameon=False); cx.legend(fontsize=8, frameon=False)
            if cuda:
                bx.legend(fontsize=8, frameon=False)
            policies = ", ".join(sorted({r.get("capacity_policy", "unspecified") for r in records.values()}))
            fig.suptitle("Measured HeadKVManager / SegPaged residency", fontsize=14)
            fig.text(.5, .02, "Controlled KV fixture, not model inference. Fixed slabs can reduce occupancy without reducing CUDA allocation.\n"
                     "Ownership panel excludes descriptors; CUDA panel includes allocated descriptors. CPU metadata is separate in JSON. "
                     "Capacity: " + policies, ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .12, 1, .93))
            save(fig, "memory_residency")
    print(json.dumps({"figures": [str(path) for path in paths]}))
    return paths

# ---- Embedded regression tests (stdlib-only unless explicitly requested) --
def d(key, tokens=1, size=1):
    return Document(key, tokens, size)


def r(key, *docs):
    return Request(str(key), "toy", tuple(docs))


class ReplayTests(unittest.TestCase):
    def test_exact_chain_and_order_change(self):
        rows = [r(0, d("A", 2), d("B", 3)),
                r(1, d("A", 2), d("B", 3)),
                r(2, d("B", 3), d("A", 2))]
        out = replay(rows, 10)
        self.assertEqual(out["metrics"]["all"]["tokens"], 15)
        self.assertEqual(out["metrics"]["prefix_hits"]["tokens"], 5)
        self.assertEqual(out["metrics"]["document_hits"]["tokens"], 10)
        self.assertEqual(out["metrics"]["nonprefix_opportunity_hits"]["tokens"], 5)
        self.assertEqual(out["metrics"]["unbounded_nonprefix_opportunities"]["tokens"], 5)

    def test_no_hit_after_prefix_break(self):
        out = replay([r(0, d("A"), d("B")), r(1, d("X"), d("B"))], 10)
        self.assertEqual(out["metrics"]["prefix_hits"]["documents"], 0)
        self.assertEqual(out["metrics"]["nonprefix_opportunity_hits"]["documents"], 1)

    def test_lru_eviction(self):
        policy = DocumentLRU(2)
        policy.publish((d("A"), d("B")))
        self.assertEqual(policy.lookup((d("A"),)), [True])
        policy.publish((d("C"),))
        self.assertEqual(policy.lookup((d("A"), d("B"), d("C"))), [True, False, True])
        self.assertEqual(policy.evictions, 1)

    def test_prefix_leaf_eviction_keeps_ancestors(self):
        policy = PrefixTrieLRU(3)
        policy.publish((d("A"), d("B")))
        policy.publish((d("A"), d("C")))
        policy.lookup((d("A"), d("B")))
        policy.publish((d("X"),))
        self.assertEqual(policy.lookup((d("A"), d("B"))), [True, True])
        self.assertEqual(policy.lookup((d("A"), d("C"))), [True, False])
        self.assertEqual(policy.used, 3)

    def test_prefix_cannot_retain_suffix_without_ancestor(self):
        policy = PrefixTrieLRU(2)
        policy.publish((d("A"), d("B"), d("C")))
        self.assertEqual(policy.lookup((d("A"), d("B"), d("C"))), [True, True, False])
        self.assertEqual(policy.bypassed, 1)

    def test_oversize_never_false_warm(self):
        rows = [r(0, d("A", size=5)), r(1, d("A", size=5))]
        out = replay(rows, 4)
        self.assertEqual(out["metrics"]["document_hits"]["documents"], 0)
        self.assertEqual(out["metrics"]["prefix_hits"]["documents"], 0)
        self.assertEqual(out["metrics"]["repeated_content"]["documents"], 1)
        self.assertEqual(out["document_policy"]["resident_payload_bytes"], 0)

    def test_prefix_oversize_does_not_evict_unrelated_resident_branch(self):
        policy = PrefixTrieLRU(3)
        policy.publish((d("A"), d("B")))
        policy.publish((d("X", size=4),))
        self.assertEqual(policy.lookup((d("A"), d("B"))), [True, True])
        self.assertEqual(policy.evictions, 0)

    def test_zero_budget(self):
        out = replay([r(0, d("A")), r(1, d("A"))], 0)
        for name in ("prefix_policy", "document_policy"):
            self.assertEqual(out[name]["peak_payload_bytes"], 0)
        self.assertEqual(out["metrics"]["document_hits"]["tokens"], 0)

    def test_within_request_duplicates_are_not_prior_hits(self):
        rows = [r(0, d("A"), d("A")), r(1, d("A"), d("A"))]
        out = replay(rows, 10)
        self.assertEqual(out["metrics"]["first_seen"]["documents"], 2)
        self.assertEqual(out["metrics"]["document_hits"]["documents"], 2)
        self.assertEqual(out["document_policy"]["resident_units"], 1)
        self.assertEqual(out["prefix_policy"]["resident_units"], 2)

    def test_metadata_title_does_not_change_rendered_identity(self):
        a = make_document({"text": "Title\nBody", "title": "metadata one"}, 5, 2, 4)
        b = make_document({"text": "Title\nBody", "title": "metadata two"}, 5, 2, 4)
        c = make_document({"text": "Other title\nBody"}, 5, 2, 4)
        self.assertEqual(a.key, b.key)
        self.assertNotEqual(a.key, c.key)
        self.assertEqual(a.payload_bytes, 16)

    def test_content_id_verified(self):
        with self.assertRaisesRegex(ValueError, "content_id"):
            make_document({"text": "A", "content_id": "0" * 64}, 1, 1, 1)

    def test_disjoint_metrics_partition(self):
        rows = [r(i, *(d(x, tokens=ord(x) % 5 + 1) for x in text))
                for i, text in enumerate(["AB", "BA", "AB", "CD", "AB", "CA", "BA"])]
        out = replay(rows, 3)
        m = out["metrics"]
        for unit in ("tokens", "documents"):
            self.assertEqual(sum(m[x][unit] for x in ("both_hits", "prefix_only_hits", "nonprefix_opportunity_hits", "neither_hits")), m["all"][unit])
            self.assertEqual(m["prefix_hits"][unit], m["both_hits"][unit] + m["prefix_only_hits"][unit])
            self.assertEqual(m["document_hits"][unit], m["both_hits"][unit] + m["nonprefix_opportunity_hits"][unit])

    def test_capacity_property_random(self):
        rng = random.Random(12)
        for budget in range(12):
            policies = [DocumentLRU(budget), PrefixTrieLRU(budget)]
            for _ in range(100):
                docs = tuple(d(str(k), size=k % 4 + 1) for k in rng.choices(range(8), k=rng.randint(1, 7)))
                for policy in policies:
                    policy.lookup(docs)
                    policy.publish(docs)
                    self.assertLessEqual(policy.used, budget)
                    self.assertGreaterEqual(policy.used, 0)
                prefix = policies[1]
                self.assertEqual(prefix.used, sum(n.size for n in prefix.nodes.values()))
                for n in prefix.nodes.values():
                    if n.node_id:
                        self.assertIn(n.parent.node_id, prefix.nodes)
                        self.assertIs(n.parent.children[n.key], n)

    def test_order_reproducibility_and_original_unmodified(self):
        rows = [r(i, d(str(i))) for i in range(20)]
        self.assertEqual(ordered_requests(rows, 4), ordered_requests(rows, 4))
        self.assertNotEqual(ordered_requests(rows, 4), ordered_requests(rows, 5))
        self.assertEqual([x.request_id for x in rows], [str(i) for i in range(20)])
        self.assertEqual(ordered_requests(rows, None), rows)

    def test_equal_budget_and_deterministic_suite(self):
        rows = [r(0, d("A"), d("B")), r(1, d("B"), d("A"))]
        a = run_suite(rows, [0, .5, 1], [0, 1])
        b = run_suite(rows, [0, .5, 1], [0, 1])
        self.assertEqual(a, b)
        self.assertEqual(a["unique_paged_corpus_payload_bytes"], 2)
        for run in a["runs"]:
            self.assertLessEqual(run["prefix_policy"]["peak_payload_bytes"], run["budget_bytes"])
            self.assertLessEqual(run["document_policy"]["peak_payload_bytes"], run["budget_bytes"])

    def test_trace_rejects_answers_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            item = {"request_id": "1", "dataset": "toy", "query": "Q?", "docs": [{"text": "A"}], "answers": ["A"]}
            path.write_text(json.dumps(item) + "\n")
            with self.assertRaisesRegex(ValueError, "answer/label"):
                read_trace(path)
            del item["answers"]
            path.write_text((json.dumps(item) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_trace(path)

    def test_no_partial_document_prefix_claim(self):
        a = make_document({"text": "Common prefix, different suffix A"}, 7, 1, 1)
        b = make_document({"text": "Common prefix, different suffix B"}, 7, 1, 1)
        out = replay([r(0, a), r(1, b)], 100)
        self.assertEqual(out["metrics"]["prefix_hits"]["tokens"], 0)


class TokenPrefixTests(unittest.TestCase):
    def test_terminal_partial_discard_retained_in_denominator(self):
        pages, dropped = page_documents(list(range(10)), 4, 2)
        self.assertEqual(dropped, 2)
        self.assertEqual(sum(p.tokens for p in pages), 8)
        self.assertEqual(sum(p.payload_bytes for p in pages), 16)
        rows = [Request("0", "toy", pages), Request("1", "toy", pages)]
        result = replay_prefix(rows, 100, 20)
        self.assertEqual(result["prefix_hit_tokens"], 8)
        self.assertEqual(result["prefix_token_fraction"], .4)

    def test_partial_document_common_prefix_is_recognized(self):
        a, _ = page_documents([1, 2, 3, 4, 5, 6, 7, 8], 4, 1)
        b, _ = page_documents([1, 2, 3, 4, 9, 10, 11, 12], 4, 1)
        result = replay_prefix([Request("a", "toy", a), Request("b", "toy", b)], 100, 16)
        self.assertEqual(result["prefix_hit_tokens"], 4)

    def test_equal_suffix_is_not_prefix_hit(self):
        a, _ = page_documents([1, 2, 3, 4, 5, 6, 7, 8], 4, 1)
        b, _ = page_documents([9, 10, 11, 12, 5, 6, 7, 8], 4, 1)
        result = replay_prefix([Request("a", "toy", a), Request("b", "toy", b)], 100, 16)
        self.assertEqual(result["prefix_hit_tokens"], 0)

    def test_page_boundary_crosses_document_boundary(self):
        left, _ = page_documents([1, 2] + [3, 4, 5, 6], 4, 1)
        right, _ = page_documents([1] + [2, 3, 4, 5, 6], 4, 1)
        self.assertEqual(left, right)

    def test_hard_budget_and_too_short(self):
        pages, dropped = page_documents([1, 2, 3], 4, 1)
        self.assertEqual((pages, dropped), ((), 3))
        pages, _ = page_documents(list(range(20)), 4, 1)
        rows = [Request(str(i), "toy", pages) for i in range(3)]
        for budget in range(25):
            result = replay_prefix(rows, budget, 60)
            self.assertLessEqual(result["policy"]["peak_payload_bytes"], budget)
            self.assertLessEqual(result["prefix_hit_tokens"], 2 * 20)

    def test_unsigned_token_validation(self):
        for token in (-1, 2**32):
            with self.assertRaises(ValueError):
                page_documents([token], 1, 1)


# ---- Shared output safety and CLI orchestration ---------------------------
def check_output(path, inputs=(), overwrite=False):
    """Never clobber input/source files, even when overwrite is requested."""
    path = Path(path)
    protected = [Path(__file__), *(Path(p) for p in inputs if p is not None)]
    for source in protected:
        if path.resolve() == source.resolve() or (
                path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"output would overwrite an input/source: {path}")
    if path.is_symlink():
        raise ValueError(f"refusing symlink output: {path}")
    if path.exists() and (not overwrite or not path.is_file()):
        raise FileExistsError(f"output exists; choose a new path or pass --overwrite: {path}")
    return path


def write_bytes(path, data, overwrite=False):
    path = check_output(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".reuse-analysis-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication, including concurrent writers.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value, overwrite=False):
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode("utf-8"), overwrite)


def replay_page_control(records, tokenizer_path, page_tokens, bytes_per_token, source):
    requests, stats = build_requests(records, tokenizer_path, page_tokens, bytes_per_token)
    runs = []
    for original in source["runs"]:
        denominator = original["metrics"]["all"]["tokens"]
        if denominator != stats["document_token_occurrences"]:
            raise ValueError("token-page denominator differs from document replay")
        sequence = ordered_requests(requests, original["shuffle_seed"])
        digest = sha256_bytes(json.dumps([r.request_id for r in sequence], separators=(",", ":")).encode())
        if digest != original["request_order_sha256"]:
            raise ValueError("token-page request order differs from document replay")
        result = replay_prefix(sequence, original["budget_bytes"], denominator)
        result.update({k: original[k] for k in (
            "budget_bytes", "budget_corpus_fraction", "order", "shuffle_seed", "request_order_sha256")})
        result["document_prefix_token_fraction"] = original["metrics"]["prefix_hits"]["token_fraction"]
        result["position_agnostic_document_token_fraction"] = original["metrics"]["document_hits"]["token_fraction"]
        result["page_minus_document_prefix_pp"] = 100 * (result["prefix_token_fraction"] - result["document_prefix_token_fraction"])
        runs.append(result)
    summaries = []
    for fraction in sorted({r["budget_corpus_fraction"] for r in runs}):
        selected = [r for r in runs if r["budget_corpus_fraction"] == fraction and r["shuffle_seed"] is not None]
        if selected:
            metrics = {}
            for key in ("prefix_token_fraction", "document_prefix_token_fraction",
                        "position_agnostic_document_token_fraction", "page_minus_document_prefix_pp"):
                values = [r[key] for r in selected]
                metrics[key] = {"mean": mean(values), "min": min(values), "max": max(values)}
            summaries.append({"budget_corpus_fraction": fraction, "shuffle_count": len(selected), "metrics": metrics})
    return {"stats": stats, "runs": runs, "shuffle_summary": summaries,
            "protocol": "Exact token-page prefix trie; same budget/order/denominator as document replay. "
                        "Final incomplete pages are not cached but their tokens remain in the denominator. "
                        "Token IDs are encoded per document and concatenated; not a model-serving benchmark."}


def replay_run(args):
    fractions = [float(x) for x in args.budget_fractions.split(",")]
    seeds = [int(x) for x in args.shuffle_seeds.split(",") if x.strip()]
    if not fractions or any(not math.isfinite(x) or not 0 <= x <= 1 for x in fractions):
        raise ValueError("fractions must be finite values between zero and one")
    if len(set(fractions)) != len(fractions) or len(set(seeds)) != len(seeds):
        raise ValueError("duplicate fractions/seeds")
    if args.bytes_per_token <= 0 or args.page_tokens <= 0:
        raise ValueError("positive byte charge and page size required")
    if len({p.resolve() for p in args.trace}) != len(args.trace):
        raise ValueError("duplicate trace path")
    inputs = [*args.trace, args.tokenizer]
    check_output(args.output, [*inputs, args.token_lengths_cache], args.overwrite)
    if args.token_lengths_cache is not None:
        # Cache updates are explicit; an existing cache is reusable without
        # rewriting it when it already contains every requested document.
        check_output(args.token_lengths_cache, [*inputs, args.output], overwrite=True)
    start = time.monotonic()
    groups = [read_trace(path) for path in args.trace]
    lengths, tokenizer = tokenizer_lengths(groups, args.tokenizer, args.token_lengths_cache)
    results = []
    for path, group in zip(args.trace, groups):
        requests = prepare_trace(group, lengths, args.bytes_per_token, args.page_tokens)
        result = run_suite(requests, fractions, seeds)
        result.update({"trace_path": str(path.resolve()), "trace_sha256": sha256_bytes(path.read_bytes())})
        if args.token_prefix_control:
            result["token_prefix_control"] = replay_page_control(
                group, args.tokenizer, args.page_tokens, args.bytes_per_token, result)
        results.append(result)
        print(json.dumps({"completed_trace": str(path), "requests": len(requests),
                          "unique_documents": result["unique_documents"]}), flush=True)
    output = {
        "schema_version": 1, "complete": True,
        "experiment": "bounded_document_cache_opportunity_replay",
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()), "tokenizer": tokenizer,
        "protocol": {
            "cache_start": "empty for every policy, capacity, and order",
            "request_order": "source order plus seeded shuffles; not production arrival timestamps",
            "document_render": DOCUMENT_PREFIX + "{canonical_text}" + DOCUMENT_SUFFIX,
            "prompt_token_construction": "encode each document separately, concatenate; question last and not cached",
            "denominator": "document token occurrences; instruction and query excluded",
            "prefix_policy": "complete-document exact-prefix trie, leaf LRU; ancestors retained",
            "nonprefix_policy": "exact-document position-agnostic LRU; optimistic reuse OPPORTUNITY, not correctness",
            "publication": "all lookups precede publication within each request",
            "payload_bytes_per_token": args.bytes_per_token,
            "default_payload_formula": "2(K,V)*32 layers*8 KV heads*128 dimensions*2 BF16 bytes = 131072 bytes/token",
            "payload_model_caveat": "illustrative conventional full KV; NOT actual Qwen hybrid/MLA allocation",
            "page_tokens": args.page_tokens,
            "budget_denominator": "paged KV of unique observed documents in this trace, not the full retrieval index",
            "budget_hard_cap": True,
            "management_overhead": "excluded from replay; measured separately in memory command",
            "nonprefix_opportunity_hits_definition": "document hit AND prefix miss under independent policies; not a combined serving cache",
            "unbounded_nonprefix_opportunities_definition": "text appeared in an earlier request but the entire ordered prefix did not",
            "shuffle_summary": "mean/min/max, not confidence intervals",
            "token_prefix_control": bool(args.token_prefix_control),
            "not_measured": ["KV correctness", "model accuracy", "FLOPs", "TTFT", "QPS", "production locality", "GPU residency"]},
        "traces": results, "elapsed_cpu_wall_seconds": time.monotonic() - start}
    write_json(args.output, output, args.overwrite)
    print(json.dumps({"output": str(args.output), "runs": sum(len(t["runs"]) for t in results),
                      "elapsed_seconds": output["elapsed_cpu_wall_seconds"]}), flush=True)
    return 0


def default_repo_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "python/sglang/srt/mem_cache/head_kv/manager.py").is_file():
            return candidate
    return Path.cwd()


def self_test_run(args):
    if not __debug__:
        raise RuntimeError("self-tests require assertions; do not use python -O")
    trace_result = trace_self_tests()
    suite = unittest.TestSuite()
    for cls in (ReplayTests, TokenPrefixTests, OutputSafetyTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    report = {**trace_result, "unit_tests": result.testsRun, "status": "passed"}
    if args.with_memory:
        report["memory_cpu"] = mem_smoke_test(args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0


class OutputSafetyTests(unittest.TestCase):
    def test_rejects_unmeasured_accelerator_before_torch_import(self):
        for device in ("mps", "xpu:0", "cuda:-1"):
            with self.assertRaises(ValueError):
                mem_validate_args(argparse.Namespace(device=device))

    def test_output_cannot_equal_source(self):
        with self.assertRaises(ValueError):
            check_output(Path(__file__), overwrite=True)

    def test_atomic_no_clobber_and_input_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.json"
            write_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                write_json(path, {"second": True})
            self.assertEqual(json.loads(path.read_text()), {"first": True})
            with self.assertRaises(ValueError):
                check_output(path, [path], overwrite=True)
            alias = Path(folder) / "alias"
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                check_output(alias, overwrite=True)
            hardlink = Path(folder) / "hardlink"
            os.link(path, hardlink)
            with self.assertRaises(ValueError):
                check_output(hardlink, [path], overwrite=True)
            write_json(path, {"second": True}, overwrite=True)
            self.assertEqual(json.loads(path.read_text()), {"second": True})


def build_parser():
    parser = argparse.ArgumentParser(
        description="复用收益分析 / Standalone RAG reuse-benefit analysis (no serving hooks).",
        epilog="Only memory --device cuda:N touches a GPU. No downloads, credentials, server/holder control, or production monkeypatches.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="extract answer-free MuSiQue / LongBench passage traces")
    prepare.add_argument("--source", action="append", required=True, metavar="DATASET=PATH")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(func=trace_run)
    retrieve = sub.add_parser("retrieve", help="run deterministic BM25 over the supplied document corpus")
    retrieve.add_argument("--trace", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--top-k", type=int, default=8)
    retrieve.add_argument("--k1", type=float, default=1.2)
    retrieve.add_argument("--b", type=float, default=.75)
    retrieve.set_defaults(func=retrieval_run)
    cache_parser = sub.add_parser("replay", help="bounded-cache budget vs prefix/non-prefix hit opportunities")
    cache_parser.add_argument("--trace", type=Path, action="append", required=True)
    cache_parser.add_argument("--tokenizer", type=Path, required=True, help="actual tokenizer.json; not a tokenizer name")
    cache_parser.add_argument("--output", type=Path, required=True)
    cache_parser.add_argument("--token-lengths-cache", type=Path)
    cache_parser.add_argument("--budget-fractions", default=",".join(str(x) for x in DEFAULT_FRACTIONS))
    cache_parser.add_argument("--shuffle-seeds", default="0,1,2,3,4", help="comma-separated seeds; empty string disables shuffles")
    cache_parser.add_argument("--bytes-per-token", type=int, default=131072)
    cache_parser.add_argument("--page-tokens", type=int, default=64)
    cache_parser.add_argument("--token-prefix-control", action="store_true")
    cache_parser.set_defaults(func=replay_run)
    memory = sub.add_parser("memory", help="real HeadKVManager page/COW residency using controlled KV fixtures")
    memory.add_argument("--repo-root", type=Path, default=default_repo_root())
    memory.add_argument("--output", type=Path, required=True)
    memory.add_argument("--device", default="cpu", help="CPU by default; explicit cuda:0 enables GPU allocation")
    memory.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    # Safe small defaults. The docstring documents the explicit large GPU layout.
    for flag, default in (("layers", 2), ("kv-heads", 2), ("local-heads", 1),
                          ("protected-first", 0), ("protected-last", 0), ("head-dim", 8),
                          ("document-tokens", 128), ("prefix-tokens", 16),
                          ("page-size", 16), ("cow-rows", 1)):
        memory.add_argument("--" + flag, type=int, default=default)
    memory.add_argument("--multiplicities", type=int, nargs="+", default=[1, 2, 4, 8])
    memory.add_argument("--fixed-capacity-pages", type=int)
    memory.add_argument("--seed", type=int, default=20261022)
    memory.set_defaults(func=mem_run)
    plots = sub.add_parser("plot", help="plot JSON measurements without inventing speed or accuracy")
    plots.add_argument("--cache", type=Path)
    plots.add_argument("--memory", type=Path)
    plots.add_argument("--output-dir", type=Path, required=True)
    plots.add_argument("--formats", default="png", help="comma-separated png,pdf,svg")
    plots.set_defaults(func=plot_run)
    tests = sub.add_parser("self-test", help="stdlib self-tests; optional tiny CPU page/COW checks")
    tests.add_argument("--with-memory", action="store_true")
    tests.add_argument("--repo-root", type=Path, default=default_repo_root())
    tests.set_defaults(func=self_test_run)
    for command in (prepare, retrieve, cache_parser, memory, plots):
        command.add_argument("--overwrite", action="store_true", help="replace analysis outputs only, never input/source files")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "memory":
            protected = list((args.repo_root / "python/sglang/srt/mem_cache/head_kv").glob("*.py"))
            check_output(args.output, protected, args.overwrite)
        result = args.func(args)
        return result if isinstance(result, int) else 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

