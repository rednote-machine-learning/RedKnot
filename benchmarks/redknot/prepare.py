"""Prepare new, sealed target-tokenizer manifests from frozen SGLang assets.

The original case selection and the source files remain byte-for-byte intact.
Prompt IDs are regenerated for the target tokenizer and prompt protocol. This
is a migrated workload, not a claim of token-identical replay of the 0731 model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from .catalog import ASSETS_ROOT, DATASETS, SOURCE_ASSETS_SHA256, SOURCE_COMMIT, get_model_profile, get_suite
from .contract import digest, file_digest, seal, validate_manifest

MIGRATION_PROTOCOL = "target-token-budget-protected-source-context-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _relative_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe asset path: {name}")
    path = root.joinpath(*relative.parts)
    _require(path.resolve().is_relative_to(root.resolve()), f"asset path escapes its root: {name}")
    return path


def validate_source_assets(
    assets_root: str | Path | None = None,
    *,
    expected_source_digest: str = SOURCE_ASSETS_SHA256,
) -> dict:
    """Check the committed inventory and every original byte before preparation."""
    root = Path(assets_root) if assets_root is not None else ASSETS_ROOT
    source = _json(root / "SOURCE.json")
    _require(source.get("schema") == "redknot-sglang-source-assets-v1", "invalid source inventory schema")
    _require(source.get("commit") == SOURCE_COMMIT, "unexpected SGLang source commit")
    _require(source.get("sha256") == seal(source)["sha256"], "source inventory hash mismatch")
    _require(source["sha256"] == expected_source_digest, "source inventory differs from the pinned snapshot")
    files = source.get("files")
    _require(isinstance(files, dict) and bool(files), "empty source inventory")
    for name, record in files.items():
        path = _relative_path(root, name)
        _require(path.is_file(), f"missing source asset: {name}")
        _require(path.stat().st_size == record["bytes"], f"source asset size mismatch: {name}")
        _require(file_digest(path) == record["sha256"], f"source asset hash mismatch: {name}")
    provenance = _json(root / "datasets/LongBench/PROVENANCE.json")
    for dataset, record in provenance["datasets"].items():
        name = f"datasets/LongBench/data/{dataset}.jsonl"
        _require(name in files, f"dataset is absent from source inventory: {dataset}")
        _require(files[name]["sha256"] == record["sha256"], f"dataset provenance hash mismatch: {dataset}")
        _require(files[name]["bytes"] == record["bytes"], f"dataset provenance size mismatch: {dataset}")
    return source


class _Rows:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache: dict[str, dict[int, dict]] = {}

    def __getitem__(self, dataset: str) -> dict[int, dict]:
        _require(dataset in DATASETS, f"unsupported dataset: {dataset}")
        if dataset not in self.cache:
            rows = {}
            path = self.root / "datasets/LongBench/data" / f"{dataset}.jsonl"
            with path.open("rb") as stream:
                for row_id, line in enumerate(stream):
                    raw = line.rstrip(b"\r\n")
                    if not raw.strip():
                        continue
                    value = json.loads(raw.decode("utf-8"))
                    if not value.get("input") or not value.get("context") or not value.get("answers"):
                        continue
                    _require(isinstance(value["input"], str), f"{dataset}/{row_id}: question must be text")
                    _require(isinstance(value["context"], str), f"{dataset}/{row_id}: context must be text")
                    _require(
                        isinstance(value["answers"], list)
                        and all(isinstance(answer, str) and answer.strip() for answer in value["answers"]),
                        f"{dataset}/{row_id}: reference answers must be nonempty text",
                    )
                    rows[row_id] = {"data": value, "row_id": row_id, "row_sha256": _sha_bytes(raw)}
            _require(bool(rows), f"dataset contains no usable rows: {dataset}")
            self.cache[dataset] = rows
        return self.cache[dataset]


class _Tokens:
    def __init__(self, tokenizer: Any, rows: _Rows) -> None:
        self.tokenizer = tokenizer
        self.rows = rows
        self.cache: dict[tuple[str, int], list[int]] = {}

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        result = list(encoded)
        _require(all(type(token) is int and 0 <= token < 2**31 for token in result), "tokenizer returned invalid token IDs")
        return result

    def decode(self, tokens: list[int]) -> str:
        # tokenizers.Tokenizer and HuggingFace tokenizers have different decode
        # keyword signatures; neither call is allowed to remove special tokens.
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def context(self, dataset: str, row_id: int) -> list[int]:
        key = (dataset, row_id)
        if key not in self.cache:
            _require(row_id in self.rows[dataset], f"{dataset}: missing source row {row_id}")
            self.cache[key] = self.encode(self.rows[dataset][row_id]["data"]["context"])
            _require(bool(self.cache[key]), f"{dataset}/{row_id}: context tokenizes to empty input")
        return self.cache[key]


def _load_release_cases(root: Path, suite: dict, rows: _Rows) -> tuple[list[dict], dict]:
    path = root / suite["path"]
    release = _json(root / "datasets/LongBench/suites/RELEASE_SUITES.json")
    record = release["suites"][suite["name"].upper()]
    _require(record["path"] == path.name, "release suite path mismatch")
    actual = file_digest(path)
    _require(actual == record["sha256"], "release suite hash mismatch")
    _require(path.with_suffix(path.suffix + ".sha256").read_text().split()[0] == actual, "suite SHA sidecar mismatch")
    cases = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            case = json.loads(line)
            _require(case["length"].lower() == suite["name"], "source case belongs to another suite")
            profile_path = (path.parent / case["profile"]).resolve()
            _require(profile_path.is_relative_to(root.resolve()), "source profile escapes assets root")
            profile = _json(profile_path)
            # Old machine absolute paths inside profiles are provenance only.
            # Always resolve the immutable sibling assets in this checkout.
            selection_path = profile_path.parent / "data_selection.json"
            prompt_path = profile_path.parent / "prompt_manifest.json"
            selection = _json(selection_path)
            prompt = _json(prompt_path)
            selection_payload = {key: value for key, value in selection.items() if key != "selection_sha256"}
            _require(selection["selection_sha256"] == digest(selection_payload), "source data selection hash mismatch")
            _require(profile["data_selection_sha256"] == selection["selection_sha256"], "profile selection mismatch")
            dataset = case["dataset"]
            _require(dataset == profile["dataset"] == selection["dataset"]["name"], "source dataset mapping mismatch")
            source_rows = rows[dataset]
            for chunk in selection["selection"]["chunks"]:
                for entry in chunk["rows"]:
                    row_id = entry["row_id"]
                    _require(row_id in source_rows, f"missing source corpus row {row_id}")
                    _require(source_rows[row_id]["row_sha256"] == entry["row_sha256"], "source corpus row hash mismatch")
            query_index = case["query_index"]
            queries = selection["selection"]["queries"]
            _require(0 <= query_index < len(queries), "source query index is out of bounds")
            query = queries[query_index]
            _require(query["row_id"] == case["query_row_id"], "source query row mapping mismatch")
            query_row = source_rows[query["row_id"]]
            _require(query_row["row_sha256"] == query["row_sha256"], "source query row hash mismatch")
            _require(case["question"] == query_row["data"]["input"], "source question differs from dataset row")
            _require(case["answers"] == query_row["data"]["answers"], "source references differ from dataset row")
            _require(query["question_sha256"] == _sha_bytes(case["question"].encode()), "source question hash mismatch")
            _require(query["answers_sha256"] == digest(case["answers"]), "source answer hash mismatch")
            expected_chunk = query["expected_chunk"]
            _require(
                any(entry["row_id"] == query["row_id"] for entry in selection["selection"]["chunks"][expected_chunk]["rows"]),
                "query evidence is absent from its source chunk",
            )
            cases.append(
                {
                    "case": case,
                    "selection": selection,
                    "expected_chunk": expected_chunk,
                    "source": {
                        "kind": "frozen_sglang_release_case",
                        "case_id": case["case_id"],
                        "query_row_id": case["query_row_id"],
                        "query_index": query_index,
                        "row_sha256": query_row["row_sha256"],
                        "original_input_ids_sha256": case["full_input_ids_sha256"],
                        "selection_origin": case.get("selection_origin"),
                        "data_selection_sha256": selection["selection_sha256"],
                        "original_profile": str(profile_path.relative_to(root.resolve())),
                        "profile_file_sha256": file_digest(profile_path),
                        "prompt_manifest_file_sha256": file_digest(prompt_path),
                        "original_prompt_protocol": prompt.get("protocol"),
                        "expected_source_chunk": expected_chunk,
                    },
                }
            )
    _require(len(cases) == release["selection_policy"]["cases_per_length"], "frozen release case count mismatch")
    return cases, release


def _query_instruction(question: str, long_output: int) -> str:
    if long_output:
        return (
            "\n\nAnswer the question using only the documents above. "
            "Put the direct answer alone on the first line. Then give one or "
            "two concise sentences explaining the supporting evidence from "
            "the documents. Do not use bullets or introduce outside facts. "
            f"Keep the complete response between {long_output - 5} and {long_output + 5} tokens "
            f"(target: about {long_output} tokens).\n"
            f"Question: {question}\nAnswer:"
        )
    return (
        "\n\nAnswer the question using only the documents above. "
        "Return the shortest exact answer span only, with no explanation.\n"
        f"Question: {question}\nAnswer:"
    )


def _construct_chunks(
    dataset: str,
    selection: dict,
    token_cache: _Tokens,
    *,
    num_chunks: int,
    chunk_tokens: int,
    excluded_padding_rows: set[int],
) -> tuple[list[list[int]], list[dict]]:
    source_chunks = selection["selection"]["chunks"]
    _require(len(source_chunks) == num_chunks, "source and target document counts differ")
    query_rows = {query["row_id"] for query in selection["selection"]["queries"]}
    source_row_ids = {entry["row_id"] for chunk in source_chunks for entry in chunk["rows"]}
    _require(query_rows <= source_row_ids, "source query row is absent from corpus selection")
    all_rows = token_cache.rows[dataset]
    padding_rows = [row_id for row_id in sorted(all_rows) if row_id not in source_row_ids and row_id not in excluded_padding_rows]
    # Deterministic continuation uses real rows after the source selection,
    # followed by the remaining lower row IDs; rows never repeat as padding.
    last_row = max(source_row_ids, default=-1)
    padding_rows.sort(key=lambda row_id: (row_id <= last_row, row_id))
    padding_cursor = 0
    result, mapping = [], []
    for chunk_index, chunk in enumerate(source_chunks):
        _require(chunk["chunk_index"] == chunk_index, "source chunks are not ordered")
        entries = chunk["rows"]
        ids, spans = [], []
        protected = [entry["row_id"] for entry in entries if entry["row_id"] in query_rows]
        required = sum(len(token_cache.context(dataset, row_id)) for row_id in protected)
        _require(
            required <= chunk_tokens,
            f"{dataset} chunk {chunk_index}: {required} target tokens are required to preserve complete query contexts; "
            f"budget {chunk_tokens} would truncate evidence. Choose a larger context budget.",
        )
        for position, entry in enumerate(entries):
            row_id = entry["row_id"]
            encoded = token_cache.context(dataset, row_id)
            later_required = sum(
                len(token_cache.context(dataset, item["row_id"])) for item in entries[position + 1 :] if item["row_id"] in query_rows
            )
            available = chunk_tokens - len(ids) - later_required
            take = min(len(encoded), max(available, 0))
            if row_id in query_rows:
                _require(take == len(encoded), "target re-tokenization would truncate a query context")
            if take:
                spans.append(
                    {
                        "row_id": row_id,
                        "row_sha256": all_rows[row_id]["row_sha256"],
                        "start": len(ids),
                        "end": len(ids) + take,
                        "target_context_tokens": len(encoded),
                        "complete_context": take == len(encoded),
                        "protected_query_context": row_id in query_rows,
                        "selection": "original_source_row",
                    }
                )
                ids.extend(encoded[:take])
        while len(ids) < chunk_tokens:
            _require(padding_cursor < len(padding_rows), f"{dataset}: insufficient distinct real context rows for the target budget")
            row_id = padding_rows[padding_cursor]
            padding_cursor += 1
            encoded = token_cache.context(dataset, row_id)
            take = min(len(encoded), chunk_tokens - len(ids))
            spans.append(
                {
                    "row_id": row_id,
                    "row_sha256": all_rows[row_id]["row_sha256"],
                    "start": len(ids),
                    "end": len(ids) + take,
                    "target_context_tokens": len(encoded),
                    "complete_context": take == len(encoded),
                    "protected_query_context": False,
                    "selection": "deterministic_real_row_padding",
                }
            )
            ids.extend(encoded[:take])
        _require(len(ids) == chunk_tokens, "target context budget mismatch")
        result.append(ids)
        mapping.append({"chunk_index": chunk_index, "input_ids_sha256": digest(ids), "rows": spans})
    return result, mapping


def _new_selection(dataset: str, query_row: int, num_chunks: int) -> dict:
    return {
        "dataset": {"name": dataset},
        "selection": {
            "chunks": [{"chunk_index": index, "rows": [{"row_id": query_row}] if index == 0 else []} for index in range(num_chunks)],
            "queries": [{"row_id": query_row, "expected_chunk": 0}],
        },
    }


def prepare_manifest(
    suite_name: str,
    model_profile: str | dict,
    tokenizer: Any,
    *,
    tokenizer_identity: dict,
    prompt_renderer: Callable[[str], str] | None = None,
    prompt_protocol: dict | None = None,
    allow_plain_prompt: bool = False,
    warmup_count: int = 1,
    max_cases: int | None = None,
    target_tokens: int | None = None,
    assets_root: str | Path | None = None,
    max_new_tokens: int = 64,
    datasets: Iterable[str] | None = None,
    cases_per_dataset: int = 1,
    expected_source_digest: str = SOURCE_ASSETS_SHA256,
) -> dict:
    """Freeze a migrated release or an explicitly new dataset matrix.

    target_tokens is a document-token budget, excluding inter-document
    delimiters, query instructions and the target chat wrapper. The final
    input_tokens value records the actual once-encoded request length.
    """
    _require(type(warmup_count) is int and warmup_count >= 0, "warmup_count must be nonnegative")
    _require(max_cases is None or (type(max_cases) is int and max_cases > 0), "max_cases must be positive")
    _require(type(max_new_tokens) is int and max_new_tokens > 0, "max_new_tokens must be positive")
    _require(type(cases_per_dataset) is int and cases_per_dataset > 0, "cases_per_dataset must be positive")
    _require(isinstance(tokenizer_identity, dict) and bool(tokenizer_identity.get("fingerprint")), "tokenizer identity is required")
    if prompt_renderer is None:
        _require(allow_plain_prompt, "a target prompt_renderer is required; plain RAG must be explicitly requested")
        prompt_renderer = lambda content: content
        prompt_protocol = {"name": "experimental-plain-rag-v1", "official_chat_protocol": False}
    else:
        _require(isinstance(prompt_protocol, dict) and bool(prompt_protocol.get("name")), "target prompt protocol identity is required")
    root = (Path(assets_root) if assets_root is not None else ASSETS_ROOT).resolve()
    inventory = validate_source_assets(root, expected_source_digest=expected_source_digest)
    profile = get_model_profile(model_profile) if isinstance(model_profile, str) else deepcopy(model_profile)
    _require(isinstance(profile, dict) and profile.get("profile") and profile.get("model_id"), "invalid model profile")
    suite = get_suite(suite_name)
    budget = suite["target_tokens"] if target_tokens is None else target_tokens
    _require(
        type(budget) is int and budget > 0 and budget % suite["num_chunks"] == 0,
        "target_tokens must be positive and divisible by num_chunks",
    )
    chunk_tokens = budget // suite["num_chunks"]
    rows = _Rows(root)
    tokens = _Tokens(tokenizer, rows)
    release_cases, release = _load_release_cases(root, suite, rows)
    excluded_query_rows = {
        dataset: {item["case"]["query_row_id"] for item in release_cases if item["case"]["dataset"] == dataset} for dataset in DATASETS
    }
    matrix_datasets = None
    if datasets is not None:
        matrix_datasets = list(datasets)
        _require(bool(matrix_datasets) and len(matrix_datasets) == len(set(matrix_datasets)), "datasets must be nonempty and unique")
        _require(all(dataset in DATASETS for dataset in matrix_datasets), "unknown matrix dataset")
        source_cases = []
        for dataset in matrix_datasets:
            available = sorted(rows[dataset])
            _require(len(available) >= cases_per_dataset, f"{dataset}: insufficient rows for requested cases")
            for row_id in available[:cases_per_dataset]:
                row = rows[dataset][row_id]
                excluded_query_rows[dataset].add(row_id)
                source_cases.append(
                    {
                        "case": {
                            "case_id": f"matrix-{dataset}-row{row_id}",
                            "query_row_id": row_id,
                            "dataset": dataset,
                            "question": row["data"]["input"],
                            "answers": row["data"]["answers"],
                            "target_output_tokens": 0,
                            "eligible_for_accuracy_aggregate": True,
                        },
                        "selection": _new_selection(dataset, row_id, suite["num_chunks"]),
                        "expected_chunk": 0,
                        "source": {
                            "kind": "output_blind_dataset_matrix",
                            "query_row_id": row_id,
                            "row_sha256": row["row_sha256"],
                            "selection_origin": "physical-row-order-before-inference-v1",
                            "expected_source_chunk": 0,
                        },
                    }
                )
    else:
        source_cases = release_cases
    if max_cases is not None:
        source_cases = source_cases[:max_cases]
    _require(bool(source_cases), "no cases selected")

    def make_case(item: dict, *, role: str) -> dict:
        source_case = item["case"]
        dataset, query_row = source_case["dataset"], source_case["query_row_id"]
        chunks, target_mapping = _construct_chunks(
            dataset,
            item["selection"],
            tokens,
            num_chunks=suite["num_chunks"],
            chunk_tokens=chunk_tokens,
            excluded_padding_rows=set(),
        )
        documents = [tokens.decode(chunk) for chunk in chunks]
        query_context = rows[dataset][query_row]["data"]["context"]
        evidence_document = documents[item["expected_chunk"]]
        _require(
            query_context in evidence_document,
            f"{dataset}/{query_row}: tokenizer decoding did not preserve the complete evidence context",
        )
        long_output = int(source_case.get("target_output_tokens", 0))
        content = "\n\n".join(documents) + _query_instruction(source_case["question"], long_output)
        rendered = prompt_renderer(content)
        _require(isinstance(rendered, str) and content in rendered, "prompt renderer must retain the entire user content")
        final_ids = tokens.encode(rendered)
        _require(bool(final_ids), "rendered prompt is empty")
        references = list(source_case["answers"])
        source = deepcopy(item["source"])
        source.update(
            {
                "repository": inventory["repository"],
                "commit": inventory["commit"],
                "dataset_file_sha256": inventory["files"][f"datasets/LongBench/data/{dataset}.jsonl"]["sha256"],
                "migration_protocol": MIGRATION_PROTOCOL,
                "source_input_ids_reused": False,
                "target_chunks": target_mapping,
                "target_document_token_budget": budget,
                "target_prompt_text_sha256": _sha_bytes(rendered.encode("utf-8")),
                "evidence": {
                    "query_row_id": query_row,
                    "complete_source_context_retained": True,
                    "source_context_sha256": _sha_bytes(query_context.encode("utf-8")),
                    "source_context_characters": len(query_context),
                    "target_document_index": item["expected_chunk"],
                    "rendered_prompt_character_start": rendered.index(query_context),
                },
            }
        )
        return {
            "sample_id": f"{role}:{suite['name']}:{source_case['case_id']}",
            "dataset": dataset,
            "question": source_case["question"],
            "input_ids": final_ids,
            "input_ids_sha256": digest(final_ids),
            "input_tokens": len(final_ids),
            "references": references,
            "references_sha256": digest(references),
            "max_new_tokens": long_output if long_output else (min(max_new_tokens, 8) if role == "warmup" else max_new_tokens),
            "scored": bool(source_case["eligible_for_accuracy_aggregate"]) if role == "case" else False,
            "ignore_eos": bool(long_output),
            "source": source,
        }

    cases = [make_case(item, role="case") for item in source_cases]
    warmup = []
    warmup_datasets = list(dict.fromkeys(item["case"]["dataset"] for item in source_cases))
    candidates = [
        (dataset, row_id)
        for dataset in warmup_datasets
        for row_id in sorted(rows[dataset], reverse=True)
        if row_id not in excluded_query_rows[dataset]
    ]
    _require(len(candidates) >= warmup_count, "insufficient held-out query rows for warmup")
    for dataset, row_id in candidates[:warmup_count]:
        row = rows[dataset][row_id]
        warmup.append(
            make_case(
                {
                    "case": {
                        "case_id": f"heldout-{dataset}-row{row_id}",
                        "dataset": dataset,
                        "query_row_id": row_id,
                        "question": row["data"]["input"],
                        "answers": row["data"]["answers"],
                        "target_output_tokens": 0,
                        "eligible_for_accuracy_aggregate": False,
                    },
                    "selection": _new_selection(dataset, row_id, suite["num_chunks"]),
                    "expected_chunk": 0,
                    "source": {
                        "kind": "heldout_warmup_query",
                        "query_row_id": row_id,
                        "row_sha256": row["row_sha256"],
                        "selection_origin": "reverse-physical-row-order-excluding-all-release-and-matrix-query-rows-v1",
                        "expected_source_chunk": 0,
                    },
                },
                role="warmup",
            )
        )
    result = seal(
        {
            "schema": "redknot-deepinfer-benchmark-v1",
            "model": {
                "profile": profile["profile"],
                "model_id": profile["model_id"],
                "native_supported": profile.get("native_supported", False),
                "tokenizer": deepcopy(tokenizer_identity),
                "prompt_protocol": deepcopy(prompt_protocol),
            },
            "suite": {
                "name": suite["name"],
                "kind": "output_blind_dataset_matrix" if matrix_datasets is not None else "migrated_sglang_release",
                "source_commit": inventory["commit"],
                "source_inventory_sha256": inventory["sha256"],
                "source_suite_sha256": release["suites"][suite["name"].upper()]["sha256"],
                "source_execution_contract": release["execution"],
                "migration_protocol": MIGRATION_PROTOCOL,
                "target_document_token_budget": budget,
                "target_chunk_tokens": chunk_tokens,
                "num_chunks": suite["num_chunks"],
                "source_token_identical": False,
                "actual_input_tokens": [case["input_tokens"] for case in cases],
                "datasets": matrix_datasets or list(dict.fromkeys(case["dataset"] for case in cases)),
                "cases_per_dataset": cases_per_dataset if matrix_datasets is not None else None,
                "case_count": len(cases),
                "scored_case_count": sum(case["scored"] for case in cases),
                "warmup_policy": "heldout-questions-with-distinct-final-input-hashes-v1",
                "warmup_count": len(warmup),
                "notes": (
                    "LongBench-derived constructed RAG, not official leaderboard prompts. "
                    "All source question contexts remain complete. Target chat framing is rendered, then "
                    "the complete request is tokenized once; actual lengths may differ from document budgets. "
                    "Warmup query rows are disjoint from evaluation query rows; background documents may overlap."
                ),
            },
            "warmup": warmup,
            "cases": cases,
        }
    )
    return validate_manifest(result)
