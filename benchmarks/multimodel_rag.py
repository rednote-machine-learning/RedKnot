"""RedKnot multi-model RAG preparation and explicit vLLM benchmark dispatch.

Only RedKnot data, prompt, profile and scoring logic is migrated. Native SGLang
engines, model loaders, GPU allocators and kernels are deliberately not imported.
Default/prepare actions are CPU-only; unsupported backends fail before vLLM import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import string
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks import benchmark_redknot as paired
from benchmarks.flash_reproduction import (
    read_json,
    validate_gpu_environment,
    verify_local_files,
)

HERE = Path(__file__).resolve().parent
PROFILES_SHA256 = "59782efde84a8d5caf2b46d8a9c5a9f440bf2d1063af4fdcc10d03a3ea38e659"
PROFILES = {
    "mistral": {
        "model": "Mistral-7B native-SWA",
        "datasets": ["hotpotqa", "2wikimqa", "musique"],
        "samples": 10,
        "chunk_tokens": 7500,
        "max_context_tokens": 30000,
        "num_chunks": 4,
        "seed": 2026,
        "source_output_tokens": 16,
        "selection": "source_order_middle_truncation_even_split",
        "delimiter": "",
        "query_template": (
            "\n\nAnswer the question based on the given passages. Only give me "
            "the answer and do not output any other words."
            "\n\nQuestion: {q}\nAnswer:"
        ),
        "runtime_blockers": [
            "Native 4096-token SWA reuse runner/offset repair is not connected to vLLM."
        ],
    },
    "llama33": {
        "model": "Llama-3.3-70B-Instruct",
        "datasets": ["2wikimqa", "musique", "multifieldqa_en"],
        "samples": 3,
        "chunk_tokens": 4000,
        "max_context_tokens": 40000,
        "num_chunks": 0,
        "seed": 2026,
        "source_output_tokens": 48,
        "selection": "source_order_leading_cap_drop_tail_under_64",
        "delimiter": "\n\n",
        "query_template": (
            "\n\nAnswer the question based only on the documents above. "
            "Give the shortest exact answer span (a name, entity, number, or short "
            "phrase), with no explanation.\nQuestion: {q}\nAnswer:"
        ),
        "runtime_blockers": [
            "Native Llama-3.3 scaled RoPE is not supported "
            "by the current static-RoPE adapter.",
            "The source INT4 setup is not supported "
            "by the current unquantized MHA adapter.",
        ],
    },
    "qwen3": {
        "model": "Qwen3-32B",
        "datasets": ["multifieldqa_en", "2wikimqa", "hotpotqa"],
        "samples": 20,
        "chunk_tokens": 4000,
        "max_context_tokens": 16000,
        "num_chunks": 0,
        "seed": 2026,
        "source_output_tokens": 32,
        "selection": "longest_char_context_first_leading_cap",
        "delimiter": "\n\n",
        "query_template": (
            "\n\nAnswer the question based on the passages above. Give the shortest "
            "exact answer span (a name, entity, number, or short noun phrase) with no "
            "explanation.\nQuestion: {q}\nAnswer:"
        ),
        "runtime_blockers": [
            "Source local_full/retrieval and SparseFFN semantics "
            "are not automatically translated.",
            "The source INT4 NF4 setup is not supported; an explicit "
            "compatible unquantized vLLM policy is required.",
        ],
    },
    "qwen35_397b": {
        "model": "Qwen3.5-397B-A17B",
        "datasets": ["triviaqa"],
        "samples": 2,
        "chunk_tokens": 8000,
        "max_context_tokens": 32000,
        "num_chunks": 4,
        "seed": 0,
        "source_output_tokens": 24,
        "selection": "seeded_shuffle_round_robin_distractors_exact_target",
        "delimiter": "\n\n",
        "query_template": (
            "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"
        ),
        "runtime_blockers": [
            "vLLM hybrid full-attention/GatedDeltaNet state runner is not connected.",
            "Per-head recurrent relocation and deep token-sparse MoE "
            "need native vLLM integration.",
        ],
    },
}


def source_assets(key: str) -> dict[str, Any]:
    raw = (HERE / "data/multimodel_profiles.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROFILES_SHA256:
        raise ValueError("source profile inventory checksum mismatch")
    assets = json.loads(raw)
    return {
        "model": assets["models"][key],
        "source_commit": assets["source_commit"],
        "source_files_sha256": assets["source_files_sha256"],
    }


def source_policy_plan(key: str, context_tokens: int) -> dict[str, Any]:
    """Read actual per-head labels, not stale human-readable ratio summaries."""
    source = source_assets(key)["model"]
    head = source["head_policy"]
    classes = Counter(
        label for row in head.get("kv_head_classification", []) for label in row
    )
    result = {
        "source_head_classes": dict(classes),
        "source_class_fractions": {
            name: count / sum(classes.values()) for name, count in classes.items()
        },
        "head_identity_candidates": {
            str(i): [j for j, label in enumerate(row) if label == "local"]
            for i, row in enumerate(head.get("kv_head_classification", []))
            if "local" in row
        },
        "untranslated_classes": sorted(set(classes) - {"local", "global"}),
        "source_policy_is_runtime_config": False,
        "source_window": head.get("window"),
        "source_ffn_or_linear": source.get("ffn_or_linear_policy"),
        "source_empirical_claims_validated_on_vllm": False,
    }
    if key == "llama33":
        result["source_effective_window"] = source["ffn_or_linear_policy"][
            "local_window"
        ]
        result["source_window_mode"] = "fixed_ffn_profile_override"
        result["source_disabled_fixed_window_fallback"] = {
            "requires_source_REDKNOT_WINDOW_FIXED": 0,
            "ratio": 0.5,
            "window_at_requested_context": int(context_tokens * 0.5),
            "active_by_default": False,
        }
    if key == "mistral":
        # This benchmark uses native SWA, not the companion head policy file.
        result.update(
            source_native_swa_window=4096,
            source_recompute_ratio=0.20,
            companion_head_profile_used_by_source_benchmark=False,
        )
    if key == "qwen35_397b":
        result.update(
            full_attention_layers=head["full_attention_layers"],
            dense_full_layers=head["dense_full_layers"],
            sparse_full_layers=head["sparse_full_layers"],
            source_frac_global=head["frac_global"],
            source_benchmark_datasets=head["benchmark_datasets"],
        )
    return result


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def short_answer(text: str, key: str) -> str:
    """Symmetric source-style extraction; never consulted during case selection."""
    if key == "mistral":
        text = re.sub(r"</?(?:END|ANS|QUE)>", " ", text, flags=re.I)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"(?i)\bthe answer is\b[:\s]*", "", text)
    text = re.sub(r"(?is)^\s*answer\s*[:：]\s*", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if not re.fullmatch(r"(?i)answer\s*[:：]?", line)]
    candidate = (lines[0] if lines else "").strip().strip('"').strip("'").strip()
    return re.sub(r"\s*[.。]\s*$", "", candidate)


def score_answer(text: str, references: Sequence[str], key: str) -> dict[str, Any]:
    answer = short_answer(text, key)
    valid = [ref for ref in references if ref.strip()]
    return {
        "extracted_answer": answer,
        "f1": paired.reference_f1(answer, valid),
        "em": max(
            (float(normalize(answer) == normalize(ref)) for ref in valid), default=None
        ),
    }


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("answers must be a string or string list")
    return [item for item in value if item.strip()]


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a local LongBench JSONL or explicit RAG JSON/JSONL, no network."""
    content = path.read_text(encoding="utf-8")
    rows = (
        json.loads(content)
        if content.lstrip().startswith("[")
        else [json.loads(line) for line in content.splitlines() if line.strip()]
    )
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("dataset must contain JSON row objects")
    return [{**row, "source_row_index": i} for i, row in enumerate(rows)]


def _ids(encode: Callable[[str], list[int]], text: str) -> list[int]:
    values = list(encode(text))
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("tokenizer must return nonnegative integer token IDs")
    return values


# REDKNOT: RK-RAG-DATA — source-specific RAG preparation without framework code.
def prepare_rows(
    key: str,
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    *,
    samples: int,
    chunk_tokens: int,
    max_context_tokens: int,
    num_chunks: int,
    seed: int,
    instruction_wrap: bool = False,
) -> dict[str, Any]:
    """Port the four source selection/chunking paths, with explicit token audit.

    Native SGLang source did decode/re-encode document fragments. We retain that
    text construction, define BOTH methods' prompt as the same concatenated
    encoded fragments, and report whether single-shot text tokenization differs.
    Source caps/truncation/distractors are declared and counted, never hidden.
    """
    if min(samples, chunk_tokens, max_context_tokens) <= 0 or not 0 <= num_chunks <= 8:
        raise ValueError(
            "positive sample/chunk/context sizes and num_chunks in 0..8 required"
        )
    if key in {"mistral", "qwen35_397b"} and num_chunks == 0:
        raise ValueError("this source profile requires an explicit document count")
    profile = PROFILES[key]
    usable = []
    for row in rows:
        question = row.get("question", row.get("input"))
        context = row.get("context")
        documents = row.get("documents")
        if not isinstance(question, str) or not question.strip():
            continue
        refs = _references(row.get("answers", []))
        if not refs:
            continue
        if documents is None and (not isinstance(context, str) or not context.strip()):
            continue
        if documents is not None and (
            not isinstance(documents, list)
            or not documents
            or any(not isinstance(d, str) or not d for d in documents)
        ):
            raise ValueError("documents must contain nonempty strings")
        usable.append({**row, "question": question.strip(), "answers": refs})
    if key == "qwen3":
        usable.sort(key=lambda row: len(row.get("context", "")), reverse=True)
    if key == "qwen35_397b":
        random.Random(seed).shuffle(usable)
    cases, audit = [], []
    for index, row in enumerate(usable):
        if len(cases) >= samples:
            break
        omitted, distractors = 0, []
        if "documents" in row:
            docs = list(row["documents"])
            original_count = sum(len(_ids(encode, doc)) for doc in docs)
        else:
            tokens = _ids(encode, row["context"])
            original_count = len(tokens)
            if key == "qwen3" and len(tokens) < chunk_tokens:
                continue
            if key == "qwen35_397b":
                target = num_chunks * chunk_tokens
                next_row = (index + 1) % len(usable)
                while len(tokens) < target and next_row != index:
                    extra = usable[next_row]
                    if isinstance(extra.get("context"), str):
                        tokens.extend(_ids(encode, extra["context"]))
                        distractors.append(extra["source_row_index"])
                    next_row = (next_row + 1) % len(usable)
                if len(tokens) < target:
                    continue
                omitted = len(tokens) - target
                tokens = tokens[:target]
            else:
                omitted = max(0, len(tokens) - max_context_tokens)
                if omitted and key == "mistral":
                    half = max_context_tokens // 2
                    tokens = tokens[:half] + tokens[-(max_context_tokens - half) :]
                else:
                    tokens = tokens[:max_context_tokens]
            if key == "mistral":
                if len(tokens) < num_chunks:
                    continue
                bounds = [len(tokens) * i // num_chunks for i in range(num_chunks + 1)]
                parts = [tokens[bounds[i] : bounds[i + 1]] for i in range(num_chunks)]
            else:
                parts = [
                    tokens[i : i + chunk_tokens]
                    for i in range(0, len(tokens), chunk_tokens)
                ]
                if key == "llama33" and parts and len(parts[-1]) < 64:
                    omitted += len(parts.pop())
                if key == "llama33" and len(parts) < 2:
                    continue
            docs = [decode(part) for part in parts]
        if not docs:
            continue
        query_text = profile["query_template"].format(q=row["question"])
        if key == "mistral" and instruction_wrap:
            docs[0] = "[INST] " + docs[0]
            query_text += "[/INST]"
        fragments = [
            doc + (profile["delimiter"] if i < len(docs) - 1 else "")
            for i, doc in enumerate(docs)
        ]
        chunks = [_ids(encode, text) for text in fragments]
        query = _ids(encode, query_text)
        if any(not chunk for chunk in chunks):
            raise ValueError("document tokenization produced an empty chunk")
        prompt = [token for chunk in chunks for token in chunk] + query
        identifier = f"{key}:{dataset}:row{row['source_row_index']}"
        cases.append(
            {
                "id": identifier,
                "chunks": chunks,
                "query": query,
                "references": row["answers"],
            }
        )
        audit.append(
            {
                "id": identifier,
                "dataset": dataset,
                "source_row_index": row["source_row_index"],
                "original_context_tokens": original_count,
                "source_cap_or_tail_omitted_tokens": omitted,
                "distractor_row_indices": distractors,
                "selection_uses_answers": False,
                "chunk_token_counts": list(map(len, chunks)),
                "query_text": query_text,
                "document_fragments": fragments,
                "matches_single_text_encoding": prompt
                == _ids(encode, "".join(fragments) + query_text),
                "prompt_sha256_json": hashlib.sha256(
                    json.dumps(prompt, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    if not cases:
        raise ValueError(
            "no source-profile eligible rows; check corpus size and chunk settings"
        )
    return {
        "cases": cases,
        "preparation": audit,
        "protocol": {
            "model_key": key,
            "source_selection": profile["selection"],
            "seed": seed,
            "source_reproduction_equivalent": False,
            "prompt_definition": (
                "concatenated independently encoded document fragments plus query"
            ),
        },
    }


def local_tokenizer(path: Path):
    """Load tokenizer.json using the optional CPU tokenizers library only."""
    tokenizer_file = path / "tokenizer.json" if path.is_dir() else path
    if not tokenizer_file.is_file():
        raise ValueError("an existing local tokenizer.json is required")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return (
        lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True),
        {
            "path": str(tokenizer_file.resolve()),
            "sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        },
    )


# REDKNOT: RK-RAG-GUARD — never remove source scaling/quantization to pass a run.
def checkpoint_preflight(
    key: str, checkpoint: Mapping[str, Any], policy: dict
) -> dict[str, Any]:
    """Reuse the worker's real CPU guards so known failures precede GPU allocation."""
    if key in {"mistral", "qwen35_397b"}:
        raise ValueError(" ".join(PROFILES[key]["runtime_blockers"]))
    architectures = {"llama33": "LlamaForCausalLM", "qwen3": "Qwen3ForCausalLM"}
    expected = {"llama33": (80, 64, 8), "qwen3": (64, 64, 8)}[key]
    actual = tuple(
        checkpoint.get(k)
        for k in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads")
    )
    if checkpoint.get("architectures") != [architectures[key]] or actual != expected:
        raise ValueError(
            "checkpoint architecture/geometry differs from the named model entrypoint"
        )
    if (
        policy.get("engine_family", "mha") != "mha"
        or not policy["allow_approximate"]
        or not policy["enabled"]
    ):
        raise ValueError("explicit enabled approximate MHA policy required")
    from vllm_redknot.runner import validate_engine_config
    from vllm_redknot.runtime import RedKnotSettings

    configuration = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(**checkpoint),
            enforce_eager=True,
            dtype="torch.bfloat16",
            quantization=checkpoint.get("quantization_config"),
        ),
        parallel_config=SimpleNamespace(),
        use_v2_model_runner=False,
        scheduler_config=SimpleNamespace(max_num_seqs=1, enable_chunked_prefill=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
    )
    settings = validate_engine_config(
        configuration, RedKnotSettings.from_mapping(policy)
    )
    for layer, heads in settings.local_heads.items():
        if (
            not layer.isdigit()
            or str(int(layer)) != layer
            or int(layer) >= expected[0]
            or max(heads) >= expected[2]
        ):
            raise ValueError("explicit canonical in-range layer/KV-head IDs required")
    head_dim = checkpoint.get(
        "head_dim", checkpoint.get("hidden_size", 0) // expected[1]
    )
    groups = expected[1] // expected[2]
    bytes_per_token = sum(
        len(heads) * (2 + groups) * head_dim * 2
        for heads in settings.local_heads.values()
    )
    return {
        "cpu_checkpoint_contract_passed": True,
        "payload_bytes_per_cached_token": bytes_per_token,
        "max_cache_bytes": settings.max_cache_bytes,
        "gpu_verified": False,
        "rope_scaling_changed": False,
    }


def data_preflight(
    document: dict, max_model_len: int, output_tokens: int, cache: dict | None = None
) -> dict:
    blockers = []
    unique = {}
    for case in document["cases"]:
        if not 1 <= len(case["chunks"]) <= 8:
            blockers.append(
                f"{case['id']}: vLLM request contract permits 1..8 chunks only"
            )
        prompt = sum(map(len, case["chunks"])) + len(case["query"])
        if prompt + output_tokens > max_model_len:
            blockers.append(
                f"{case['id']}: prompt plus output exceeds model length; "
                "no runtime truncation"
            )
        for chunk in case["chunks"]:
            unique[tuple(chunk)] = len(chunk)
    required = None
    if cache:
        required = sum(unique.values()) * cache["payload_bytes_per_cached_token"]
        if required > cache["max_cache_bytes"]:
            blockers.append(
                "all unique offline captures exceed configured CPU cache payload budget"
            )
    return {
        "case_count": len(document["cases"]),
        "unique_chunks": len(unique),
        "cached_tokens": sum(unique.values()),
        "required_capture_payload_bytes": required,
        "blockers": blockers,
    }


# REDKNOT: RK-RAG-METRICS — symmetric answer extraction and explicit F1/EM scope.
def migrated_metrics(report: dict, key: str) -> dict:
    refs = {case["id"]: case["references"] for case in report["cases"]}
    rows = [
        {
            "case_id": row["case_id"],
            "pair_index": row["pair_index"],
            "mode": row["mode"],
            **score_answer(row["output"]["text"], refs[row["case_id"]], key),
        }
        for row in report["measurements"]
    ]
    means = {}
    for mode in ("dense", "reuse"):
        means[mode] = {}
        for metric in ("f1", "em"):
            scores = [
                row[metric]
                for row in rows
                if row["mode"] == mode and row[metric] is not None
            ]
            means[mode][metric] = statistics.mean(scores) if scores else None
    dense, reuse = means["dense"]["f1"], means["reuse"]["f1"]
    return {
        "source_style_extraction": True,
        "rows": rows,
        "mean_scores": means,
        "f1_drop_percentage_points": 100 * (dense - reuse)
        if dense is not None and reuse is not None
        else None,
        "raw_generic_metrics_retained": True,
        "qps_measured": False,
        "flops_savings_claimed": False,
    }


def build_parser(key: str) -> argparse.ArgumentParser:
    profile = PROFILES[key]
    parser = argparse.ArgumentParser(
        description=(
            f"{profile['model']} RedKnot vLLM migration. "
            "Default: CPU plan, no model execution."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument(
        "--run",
        action="store_true",
        help="Explicit GPU opt-in, supported native contract only",
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Existing local LongBench JSONL directory"
    )
    parser.add_argument(
        "--rag-file", type=Path, help="Existing JSON/JSONL question/documents/answers"
    )
    parser.add_argument(
        "--dataset", action="append", help="Repeatable; defaults match source profile"
    )
    parser.add_argument(
        "--tokenizer", type=Path, help="Local tokenizer.json or its directory"
    )
    parser.add_argument(
        "--cases", type=Path, help="Previously prepared exact-token cases"
    )
    parser.add_argument("--output", type=Path, help="New JSON path; never overwritten")
    parser.add_argument("--samples", type=int, default=profile["samples"])
    parser.add_argument("--chunk-tokens", type=int, default=profile["chunk_tokens"])
    parser.add_argument(
        "--max-context-tokens", type=int, default=profile["max_context_tokens"]
    )
    parser.add_argument("--num-chunks", type=int, default=profile["num_chunks"])
    parser.add_argument("--seed", type=int, default=profile["seed"])
    parser.add_argument(
        "--mistral-instruction-wrap",
        action="store_true",
        help="Explicit source [INST]...[/INST] wrapping",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Reviewed vLLM policy; source profile is not substituted",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        help="Trusted local {model_revision,files:[{path,size,sha256}]} inventory",
    )
    parser.add_argument("--gpu-confirmed-idle", action="store_true")
    parser.add_argument(
        "--run-compatible-variant",
        action="store_true",
        help=(
            "Acknowledge a different explicit vLLM policy, "
            "not source-equivalent reproduction"
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-pairs", type=int, default=3)
    parser.add_argument("--measured-pairs", type=int, default=10)
    return parser


def prepare_from_args(key: str, args: argparse.Namespace) -> dict | None:
    if args.cases:
        if args.data_dir or args.rag_file:
            raise ValueError("choose prepared --cases OR raw dataset input")
        document = read_json(args.cases)
        paired.parse_cases(document)
        return document
    if not args.data_dir and not args.rag_file:
        return None
    if args.data_dir and args.rag_file:
        raise ValueError("choose --data-dir OR --rag-file")
    if args.tokenizer is None:
        raise ValueError("--tokenizer is required for raw corpus preparation")
    encode, decode, identity = local_tokenizer(args.tokenizer)
    sources = (
        [("explicit_rag", args.rag_file)]
        if args.rag_file
        else [
            (dataset, args.data_dir / f"{dataset}.jsonl")
            for dataset in (args.dataset or PROFILES[key]["datasets"])
        ]
    )
    result = {"cases": [], "preparation": [], "tokenizer": identity, "source_files": []}
    for dataset, path in sources:
        if re.fullmatch(r"[A-Za-z0-9_-]+", dataset) is None:
            raise ValueError("dataset names must not contain paths")
        document = prepare_rows(
            key,
            read_rows(path),
            dataset,
            encode,
            decode,
            samples=args.samples,
            chunk_tokens=args.chunk_tokens,
            max_context_tokens=args.max_context_tokens,
            num_chunks=args.num_chunks,
            seed=args.seed,
            instruction_wrap=args.mistral_instruction_wrap,
        )
        result["cases"].extend(document["cases"])
        result["preparation"].extend(document["preparation"])
        result["protocol"] = document["protocol"]
        result["source_files"].append(
            {
                "dataset": dataset,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return result


def run_gpu(key: str, args: argparse.Namespace, document: dict, plan: dict) -> dict:
    if key in {"mistral", "qwen35_397b"}:
        raise ValueError(" ".join(PROFILES[key]["runtime_blockers"]))
    if not args.run_compatible_variant:
        raise ValueError(
            "--run-compatible-variant is required for the distinct vLLM policy"
        )
    if (
        not args.gpu_confirmed_idle
        or not args.model
        or not args.config
        or not args.model_manifest
    ):
        raise ValueError(
            "--run requires --gpu-confirmed-idle, --model, --config "
            "and --model-manifest"
        )
    validate_gpu_environment(os.environ)
    from vllm_redknot.cli import installed_plugin
    from vllm_redknot.compat import verify_vllm_sources
    from vllm_redknot.config import load_config

    policy = load_config(args.config)
    checkpoint = read_json(args.model / "config.json")
    cache = checkpoint_preflight(key, checkpoint, policy)
    data_plan = data_preflight(document, args.max_model_len, args.output_tokens, cache)
    if data_plan["blockers"]:
        raise ValueError("; ".join(data_plan["blockers"]))
    manifest = read_json(args.model_manifest)
    if manifest.get("model_revision") != policy["model_revision"]:
        raise ValueError("trusted model manifest revision must match explicit policy")
    index_path = args.model / "model.safetensors.index.json"
    if index_path.is_file():
        names = set(read_json(index_path)["weight_map"].values()) | {
            "model.safetensors.index.json"
        }
    else:
        names = {"model.safetensors"}
    declared = {item["path"] for item in manifest["files"]}
    if not names | {"config.json"} <= declared:
        raise ValueError(
            "trusted manifest must hash config, shard index and every checkpoint shard"
        )
    contract = verify_vllm_sources(engine_family="mha")
    if not installed_plugin():
        raise ValueError(
            "the RedKnot plugin must already be installed "
            "in the pinned vLLM environment"
        )
    verified = verify_local_files(args.model, manifest["files"])
    args.dtype = "bfloat16"
    args.max_num_batched_tokens = args.max_model_len
    kwargs, metadata = paired.prepare_engine(args)
    parsed = paired.parse_cases(document)
    vocab_size = checkpoint.get("vocab_size")
    if type(vocab_size) is not int or any(
        token >= vocab_size for case in parsed for token in case.tokens
    ):
        raise ValueError("case token IDs exceed checkpoint vocabulary")
    paired._validate_run(
        parsed,
        args.warmup_pairs,
        args.measured_pairs,
        args.output_tokens,
        args.max_model_len,
        args.max_model_len,
    )
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(**kwargs)
        report = paired.run_benchmark(
            llm,
            SamplingParams,
            parsed,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len,
            warmup_pairs=args.warmup_pairs,
            measured_pairs=args.measured_pairs,
            output_tokens=args.output_tokens,
        )
        report.update(
            engine=kwargs,
            configuration=metadata,
            source_contract=contract,
            model_verification=verified,
            migration_plan=plan,
            input_preparation=document.get("preparation"),
            source_reproduction_equivalent=False,
        )
        report["migrated_source_metrics"] = migrated_metrics(report, key)
        paired.write_report(args.output, report)
        return report
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30)


def main(key: str, argv: Sequence[str] | None = None) -> int:
    parser = build_parser(key)
    args = parser.parse_args(argv)
    try:
        if args.output_tokens < 50 or args.warmup_pairs < 3 or args.measured_pairs < 10:
            raise ValueError(
                "vLLM paired protocol requires >=50 output tokens, "
                "3 warmup and 10 measured pairs"
            )
        if args.max_model_len <= 0 or args.max_context_tokens <= 0:
            raise ValueError("context and model lengths must be positive")
        if args.samples <= 0 or args.chunk_tokens <= 0 or not 0 <= args.num_chunks <= 8:
            raise ValueError(
                "positive samples/chunk size and num_chunks in 0..8 required"
            )
        if args.dataset and len(set(args.dataset)) != len(args.dataset):
            raise ValueError("duplicate dataset selection is not allowed")
        if args.run and key in {"mistral", "qwen35_397b"}:
            raise ValueError(" ".join(PROFILES[key]["runtime_blockers"]))
        if args.run and not args.run_compatible_variant:
            raise ValueError(
                "Original source policies are not runtime-equivalent. "
                "--run-compatible-variant plus a reviewed explicit vLLM policy "
                "is required; no source window/sink/FFN setting is silently dropped."
            )
        if args.prepare_only or args.run:
            if (
                args.output is None
                or not args.output.parent.is_dir()
                or os.path.lexists(args.output)
            ):
                raise ValueError("a new --output in an existing directory is required")
        document = prepare_from_args(key, args)
        plan = {
            "model_key": key,
            "status": "cpu_migration_plan",
            "gpu_initialized": False,
            "defaults": PROFILES[key],
            "source_policy": source_policy_plan(key, args.max_context_tokens),
            "source_assets": {
                name: value
                for name, value in source_assets(key).items()
                if name != "model"
            },
            "source_reproduction_equivalent": False,
            "runtime_assessed": False,
            "automatic_download": False,
            "data_plan": data_preflight(
                document, args.max_model_len, args.output_tokens
            )
            if document
            else None,
        }
        if args.run:
            if not document:
                raise ValueError(
                    "--run requires prepared --cases or explicit local corpus/tokenizer"
                )
            report = run_gpu(key, args, document, plan)
            print(
                json.dumps(
                    {
                        "qualified": report["qualified"],
                        "output": str(args.output),
                        "migrated_source_metrics": report["migrated_source_metrics"],
                    },
                    indent=2,
                )
            )
            return 0 if report["qualified"] else 2
        if args.prepare_only:
            if not document:
                raise ValueError(
                    "--prepare-only requires explicit local data "
                    "and tokenizer or --cases"
                )
            document["migration_plan"] = plan
            paired.write_report(args.output, document)
            print(
                json.dumps(
                    {
                        "status": "prepared_cpu_only",
                        "output": str(args.output),
                        "data_plan": plan["data_plan"],
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"{PROFILES[key]['model']} migration: {error}\n")
    return 0
