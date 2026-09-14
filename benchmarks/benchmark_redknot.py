"""Run an explicit local-model, serial paired warm-prefill benchmark.

Example (only this explicit command initializes vLLM and writes the output):
    python benchmarks/benchmark_redknot.py --model /models/local-model \
        --config /absolute/redknot.json --cases /absolute/cases.json \
        --max-model-len 8192 --max-num-batched-tokens 8192 \
        --output /absolute/new-result.json

Install this project's redknot plugin in the model's vLLM environment first.
Input: {"cases": [{"id": "example", "chunks": [[1, 2], [3]],
                   "query": [4], "references": ["reference answer"]}]}.
Every prompt token is retained. At most eight nonempty chunks are permitted.
Unique chunks are captured independently; capture cost is reported separately.
Each case gets at least three untimed warmup pairs and ten measured pairs, with
alternating dense/reuse order and the same greedy fixed output-token budget.

TTFT is output.metrics.first_token_latency, never a subtraction between wall
and monotonic timestamps. E2E uses perf_counter around generate, excluding RPCs.
English SQuAD-style whitespace token F1 is not a suitable Chinese semantic
accuracy metric; references require a language-appropriate evaluation protocol.
Missing references remain unscored. Ratios describe serial hot TTFT only, not
throughput/QPS or a promised speedup. qualified means measurement/reuse evidence
is complete; it does not imply an accuracy threshold or a performance win.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import string
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    chunks: tuple[tuple[int, ...], ...]
    query: tuple[int, ...]
    references: tuple[str, ...]

    @property
    def tokens(self) -> list[int]:
        return [token for chunk in self.chunks for token in chunk] + list(self.query)

    @property
    def spans(self) -> list[dict[str, int]]:
        spans, offset = [], 0
        for chunk in self.chunks:
            spans.append({"start": offset, "end": offset + len(chunk)})
            offset += len(chunk)
        return spans


def parse_cases(document: Any) -> list[Case]:
    """Validate tokenized cases without tokenization, truncation or deduplication."""
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise ValueError("input must contain a cases array")
    if not document["cases"]:
        raise ValueError("cases must not be empty")

    def tokens(value: Any, label: str, *, nonempty: bool = False) -> tuple[int, ...]:
        if not isinstance(value, list) or (nonempty and not value):
            raise ValueError(f"{label} must be a {'nonempty ' if nonempty else ''}list")
        if any(type(token) is not int or token < 0 for token in value):
            raise ValueError(f"{label} must contain nonnegative integer token IDs")
        return tuple(value)

    result, identifiers = [], set()
    for raw in document["cases"]:
        if not isinstance(raw, dict):
            raise ValueError("each case must be an object")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("each case needs a nonempty string id")
        if identifier in identifiers:
            raise ValueError(f"duplicate case id: {identifier}")
        identifiers.add(identifier)
        chunks = raw.get("chunks")
        if not isinstance(chunks, list) or not 1 <= len(chunks) <= 8:
            raise ValueError("each case needs between one and eight chunks")
        references = raw.get("references", [])
        if not isinstance(references, list) or any(
            not isinstance(reference, str) for reference in references
        ):
            raise ValueError("references must be a list of strings")
        result.append(
            Case(
                identifier,
                tuple(tokens(chunk, "chunk", nonempty=True) for chunk in chunks),
                tokens(raw.get("query"), "query"),
                tuple(references),
            )
        )
    return result


def token_f1(prediction: str, reference: str) -> float:
    """Compute standard English normalized token-overlap F1, including multiplicity."""

    def normalize(text: str) -> list[str]:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted, expected = normalize(prediction), normalize(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2.0 * overlap / (len(predicted) + len(expected))


def reference_f1(prediction: str, references: Sequence[str]) -> float | None:
    """Take the best reference score; absent references are unavailable, not 1.0."""
    return max((token_f1(prediction, ref) for ref in references), default=None)


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Use linear interpolation between sorted sample positions (type 7)."""
    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percentile must be between zero and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_stats(worker: Any) -> dict[str, Any]:
    """RPC callable: read the plugin's actual worker runtime, without GPU work."""
    runner = getattr(worker, "model_runner", None)
    runtime = getattr(runner, "redknot_runtime", None)
    if runtime is None:
        runtime = getattr(runner, "_redknot_runtime", None)
    if runtime is None or not callable(getattr(runtime, "stats", None)):
        return {"available": False, "reason": "redknot_runtime_missing"}
    try:
        snapshot = runtime.stats()
        if not isinstance(snapshot, Mapping):
            raise ValueError("invalid stats")
        counters, cache = snapshot.get("runtime"), snapshot.get("cache")
        if not isinstance(counters, Mapping) or not isinstance(cache, Mapping):
            raise ValueError("invalid counter groups")
        if "hits" not in cache or type(cache["hits"]) is not int or cache["hits"] < 0:
            raise ValueError("cache hit counter missing")
        if any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in counters.items()
        ):
            raise ValueError("invalid runtime counter")
        return {"available": True, "runtime": dict(counters), "cache": dict(cache)}
    except Exception as error:
        return {
            "available": False,
            "reason": f"invalid_worker_stats:{type(error).__name__}",
        }


def _snapshot(llm: Any) -> dict[str, Any]:
    try:
        replies = llm.collective_rpc(read_stats, timeout=10)
    except Exception as error:
        return {"available": False, "reason": f"rpc_failed:{type(error).__name__}"}
    if not isinstance(replies, list) or len(replies) != 1:
        return {"available": False, "reason": "expected_one_tp1_worker"}
    reply = replies[0]
    if not isinstance(reply, dict) or type(reply.get("available")) is not bool:
        return {"available": False, "reason": "invalid_rpc_response"}
    return reply


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not before["available"] or not after["available"]:
        return {"available": False, "reason": "worker_stats_unavailable"}
    runtime = {
        key: after["runtime"].get(key, 0) - before["runtime"].get(key, 0)
        for key in set(before["runtime"]) | set(after["runtime"])
    }
    hits = after["cache"]["hits"] - before["cache"]["hits"]
    if hits < 0 or any(value < 0 for value in runtime.values()):
        return {"available": False, "reason": "worker_counters_reset"}
    return {
        "available": True,
        "cache_hits": hits,
        "reuse_steps": runtime.get("reuse_steps", 0),
        "reused_local_query_rows": runtime.get("reused_local_query_rows", 0),
        "restored_local_kv_rows": runtime.get("restored_local_kv_rows", 0),
        "reused_projected_token_rows": runtime.get("reused_projected_token_rows", 0),
        "native_state_token_rows": runtime.get("native_state_token_rows", 0),
        "launched_sparse_head_rows": runtime.get("launched_sparse_head_rows", 0),
        "runtime": runtime,
    }


def _finite(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        return None
    return float(value)


def _generate(
    llm: Any,
    sampling_factory: Callable[..., Any],
    *,
    tokens: list[int],
    spans: list[dict[str, int]],
    mode: str,
    namespace: str,
    output_tokens: int,
    timed: bool,
    evidence: bool,
    clock: Callable[[], float],
) -> dict[str, Any]:
    request_mode = "recomputed" if mode == "dense" else mode
    params = sampling_factory(
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        extra_args={
            "redknot": {
                "mode": request_mode,
                "namespace": namespace,
                "chunks": spans,
                "allow_approximate": mode == "reuse",
            }
        },
    )
    before = _snapshot(llm) if evidence else None
    started = clock() if timed else None
    outputs = llm.generate(
        [{"prompt_token_ids": list(tokens)}],
        params,
        use_tqdm=False,
    )
    elapsed = clock() - started if timed else None
    after = _snapshot(llm) if evidence else None
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("expected exactly one request and one greedy completion")
    output, completion = outputs[0], outputs[0].outputs[0]
    metrics = getattr(output, "metrics", None)
    raw_prompt = getattr(output, "prompt_token_ids", None)
    generated = list(completion.token_ids)
    return {
        "mode": mode,
        "request_mode": request_mode,
        "prompt_token_ids": list(tokens),
        "returned_prompt_token_ids": list(raw_prompt)
        if raw_prompt is not None
        else None,
        "prompt_preserved": raw_prompt is not None and list(raw_prompt) == tokens,
        "output": {
            "request_id": str(output.request_id),
            "text": completion.text,
            "token_ids": generated,
            "num_tokens": len(generated),
            "finish_reason": completion.finish_reason,
            "stop_reason": getattr(completion, "stop_reason", None),
            "finished": bool(getattr(output, "finished", False)),
        },
        "ttft_seconds": _finite(
            getattr(metrics, "first_token_latency", None),
            positive=True,
        )
        if timed
        else None,
        "e2e_seconds": _finite(elapsed) if timed else None,
        "is_corrupted": bool(getattr(metrics, "is_corrupted", False)),
        "worker_stats_before": before,
        "worker_stats_after": after,
        "reuse_evidence": _delta(before, after) if evidence else None,
    }


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row[field] for row in rows if row[field] is not None]
    complete = len(values) == len(rows) and bool(rows)
    return {
        "samples": len(values),
        "expected_samples": len(rows),
        "p50": percentile(values, 50) if complete else None,
        "p95": percentile(values, 95) if complete else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only supplied measured rows; ratios are dense divided by reuse."""
    by_mode = {
        mode: [row for row in rows if row["mode"] == mode]
        for mode in ("dense", "reuse")
    }
    latency = {
        mode: _distribution(values, "ttft_seconds") for mode, values in by_mode.items()
    }
    f1 = {}
    for mode, values in by_mode.items():
        scores = [row["f1"] for row in values if row["f1"] is not None]
        f1[mode] = statistics.mean(scores) if scores else None
    ratios = {}
    for quantile in ("p50", "p95"):
        dense, reuse = latency["dense"][quantile], latency["reuse"][quantile]
        ratios[quantile] = dense / reuse if dense is not None and reuse else None
    evidence = [row["reuse_evidence"] for row in by_mode["reuse"]]
    complete = all(item is not None and item["available"] for item in evidence)
    totals = {"available": complete, "measured_requests": len(evidence)}
    for field in (
        "cache_hits",
        "reuse_steps",
        "reused_local_query_rows",
        "restored_local_kv_rows",
        "reused_projected_token_rows",
        "native_state_token_rows",
        "launched_sparse_head_rows",
    ):
        totals[field] = sum(item[field] for item in evidence) if complete else None
    return {
        "measured_pairs": len(rows) // 2,
        "mean_f1": f1,
        "scored_outputs": {
            mode: sum(row["f1"] is not None for row in values)
            for mode, values in by_mode.items()
        },
        "f1_drop_percentage_points": (
            100 * (f1["dense"] - f1["reuse"])
            if f1["dense"] is not None and f1["reuse"] is not None
            else None
        ),
        "ttft_seconds": latency,
        "hot_ttft_ratio_dense_over_reuse": ratios,
        "e2e_seconds": {
            mode: _distribution(values, "e2e_seconds")
            for mode, values in by_mode.items()
        },
        "measured_reuse_evidence": totals,
    }


def _validate_run(
    cases: Sequence[Case],
    warmup_pairs: int,
    measured_pairs: int,
    output_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> None:
    for label, value, minimum in (
        ("warmup_pairs", warmup_pairs, 3),
        ("measured_pairs", measured_pairs, 10),
        ("output_tokens", output_tokens, 50),
        ("max_model_len", max_model_len, 1),
        ("max_num_batched_tokens", max_num_batched_tokens, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if not cases:
        raise ValueError("at least one case is required")
    if max_num_batched_tokens < max_model_len:
        raise ValueError(
            "unchunked prefill requires max_num_batched_tokens >= max_model_len"
        )
    for case in cases:
        if len(case.tokens) + output_tokens > max_model_len:
            raise ValueError(
                f"case {case.id!r} plus generated tokens exceeds max_model_len"
            )


def run_benchmark(
    llm: Any,
    sampling_factory: Callable[..., Any],
    cases: Sequence[Case],
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    warmup_pairs: int = 3,
    measured_pairs: int = 10,
    output_tokens: int = 128,
    namespace: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    engine_family: str = "mha",
) -> dict[str, Any]:
    """Run serial captures and fair pairs; injected engines support CPU testing."""
    if engine_family not in {"mha", "deepseek_v4_flash"}:
        raise ValueError("Unsupported benchmark engine_family")
    _validate_run(
        cases,
        warmup_pairs,
        measured_pairs,
        output_tokens,
        max_model_len,
        max_num_batched_tokens,
    )
    namespace = namespace or f"redknot-benchmark-{uuid.uuid4().hex}"
    captures, warmups, measurements = [], [], []
    reasons = set()
    for chunk in dict.fromkeys(chunk for case in cases for chunk in case.chunks):
        record = _generate(
            llm,
            sampling_factory,
            tokens=list(chunk),
            spans=[{"start": 0, "end": len(chunk)}],
            mode="capture",
            namespace=namespace,
            output_tokens=1,
            timed=True,
            evidence=True,
            clock=clock,
        )
        record["chunk_sha256"] = hashlib.sha256(json.dumps(chunk).encode()).hexdigest()
        captures.append(record)
    for case in cases:
        if not case.references:
            reasons.add(f"{case.id}:accuracy_unavailable_no_references")
        for index in range(warmup_pairs + measured_pairs):
            measured = index >= warmup_pairs
            order = ("dense", "reuse") if index % 2 == 0 else ("reuse", "dense")
            for mode in order:
                record = _generate(
                    llm,
                    sampling_factory,
                    tokens=case.tokens,
                    spans=case.spans,
                    mode=mode,
                    namespace=namespace,
                    output_tokens=output_tokens,
                    timed=measured,
                    evidence=measured,
                    clock=clock,
                )
                record.update(
                    case_id=case.id,
                    pair_index=index - warmup_pairs if measured else index,
                    stage="measured" if measured else "warmup",
                    f1=reference_f1(record["output"]["text"], case.references),
                )
                (measurements if measured else warmups).append(record)
                if not measured:
                    continue
                prefix = f"{case.id}:{mode}"
                if record["ttft_seconds"] is None:
                    reasons.add(f"{prefix}:first_token_latency_unavailable")
                if record["e2e_seconds"] is None:
                    reasons.add(f"{prefix}:e2e_clock_invalid")
                if not record["prompt_preserved"]:
                    reasons.add(f"{prefix}:prompt_token_mismatch")
                if record["output"]["num_tokens"] != output_tokens:
                    reasons.add(f"{prefix}:output_token_budget_not_met")
                if not record["output"]["finished"] or record["is_corrupted"]:
                    reasons.add(f"{prefix}:unfinished_or_corrupted_output")
                evidence = record["reuse_evidence"]
                if not evidence["available"]:
                    reasons.add(f"{prefix}:{evidence['reason']}")
                elif mode == "dense" and (
                    evidence["reuse_steps"] or evidence["reused_local_query_rows"]
                ):
                    reasons.add(f"{prefix}:dense_unexpected_reuse")
                elif mode == "reuse":
                    required_evidence = [
                        ("cache_hits", "no_cache_hit"),
                        ("reuse_steps", "no_reuse_step"),
                        (
                            "reused_local_query_rows",
                            "no_selected_query_head_rows_reused",
                        ),
                    ]
                    if engine_family == "deepseek_v4_flash":
                        required_evidence.extend(
                            [
                                ("reused_projected_token_rows", "no_clean_zoff_merge"),
                                (
                                    "native_state_token_rows",
                                    "no_native_state_row_evidence",
                                ),
                                ("launched_sparse_head_rows", "no_sparse_kernel_work"),
                            ]
                        )
                    for field, reason in required_evidence:
                        if evidence[field] <= 0:
                            reasons.add(f"{prefix}:{reason}")
    capture_times = [row["e2e_seconds"] for row in captures]
    return {
        "schema_version": 1,
        "qualified": not reasons,
        "qualification_reasons": sorted(reasons),
        "namespace": namespace,
        "settings": {
            "engine_family": engine_family,
            "warmup_pairs_per_case": warmup_pairs,
            "measured_pairs_per_case": measured_pairs,
            "output_tokens": output_tokens,
            "greedy": True,
            "ignore_eos": True,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "ttft_source": "output.metrics.first_token_latency",
            "e2e_source": "perf_counter around generate; excludes collective_rpc",
            "f1_drop_definition": (
                "100 * (dense mean F1 - reuse mean F1), not relative percent"
            ),
            "row_counter_units": {
                "reused_local_query_rows": (
                    "token * query-head, summed over selected layers"
                ),
                "restored_local_kv_rows": (
                    "token * KV-head, summed over selected layers"
                ),
                "reused_projected_token_rows": "clean token, summed over MLA layers",
                "native_state_token_rows": (
                    "online token, summed over selected MLA attention callsites"
                ),
                "launched_sparse_head_rows": (
                    "token * padded query-head, summed over selected MLA calls"
                ),
            },
        },
        "limitations": [
            "English token F1 is unsuitable for Chinese semantic evaluation.",
            "Serial hot TTFT and E2E do not measure throughput or QPS.",
            "Capture, warmups and engine startup are excluded from hot latencies.",
            "Qualification checks evidence, not an accuracy or speed threshold.",
        ]
        + (
            [
                "Flash reuses local-head z_off only; native KV, compressors, Indexer "
                "and FFNs still run online. Native-state counters describe the "
                "preserved callsite, not independent GPU producer instrumentation.",
                "Masked native wo_a remains full-width: "
                "no wo_a compute saving is claimed.",
                "BF16 partial-projection addition can round differently "
                "from native GEMM.",
                "Independent chunk reuse is approximate even after boundary repair.",
            ]
            if engine_family == "deepseek_v4_flash"
            else []
        ),
        "cases": [
            {
                "id": case.id,
                "chunks": [list(chunk) for chunk in case.chunks],
                "query": list(case.query),
                "references": list(case.references),
            }
            for case in cases
        ],
        "capture_cost": {
            "unique_chunks": len(captures),
            "e2e_seconds_total": sum(capture_times)
            if all(value is not None for value in capture_times)
            else None,
        },
        "captures": captures,
        "warmups": warmups,
        "measurements": measurements,
        "summary": summarize(measurements),
        "per_case": {
            case.id: summarize(
                [row for row in measurements if row["case_id"] == case.id]
            )
            for case in cases
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Existing local model directory"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Explicit opt-in plugin JSON"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON path; never overwritten"
    )
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--warmup-pairs", type=int, default=3)
    parser.add_argument("--measured-pairs", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16"), default="auto"
    )
    return parser


def prepare_engine(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate local opt-in and set worker plugin environment before vLLM import."""
    from vllm_redknot.cli import checkpoint_engine_options
    from vllm_redknot.config import load_config, policy_fingerprint

    model = args.model.expanduser().resolve(strict=True)
    if not model.is_dir():
        raise ValueError("--model must name an existing local model directory")
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_config(config_path)
    if config is None or not config["enabled"] or not config["allow_approximate"]:
        raise ValueError(
            "benchmark requires enabled=true and allow_approximate=true in config"
        )
    backend, dtype = checkpoint_engine_options(model, config, args.dtype)
    family = config.get("engine_family", "mha")
    os.environ["VLLM_REDKNOT_CONFIG"] = str(config_path)
    os.environ["VLLM_PLUGINS"] = "redknot"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return {
        "model": str(model),
        "dtype": dtype,
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "attention_config": {"backend": backend},
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "disable_log_stats": False,
        "generation_config": "vllm",
        "seed": 0,
    }, {
        "config_path": str(config_path),
        "engine_family": family,
        "config": config,
        "policy_fingerprint": policy_fingerprint(config),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write only on an explicit call, refusing existing files and symlinks."""
    serialized = (
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    with path.open("x", encoding="utf-8") as output:
        output.write(serialized)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if os.path.lexists(args.output):
            raise FileExistsError(f"output already exists: {args.output}")
        if not args.output.parent.is_dir():
            raise ValueError("output parent directory must already exist")
        cases = parse_cases(json.loads(args.cases.read_text(encoding="utf-8")))
        _validate_run(
            cases,
            args.warmup_pairs,
            args.measured_pairs,
            args.output_tokens,
            args.max_model_len,
            args.max_num_batched_tokens,
        )
        kwargs, metadata = prepare_engine(args)
        from vllm import LLM, SamplingParams

        llm = LLM(**kwargs)
        report = run_benchmark(
            llm,
            SamplingParams,
            cases,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            warmup_pairs=args.warmup_pairs,
            measured_pairs=args.measured_pairs,
            output_tokens=args.output_tokens,
            engine_family=metadata["engine_family"],
        )
        report["engine"] = kwargs
        report["configuration"] = metadata
        write_report(args.output, report)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"benchmark failed: {error}\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "qualified": report["qualified"],
                "qualification_reasons": report["qualification_reasons"],
            }
        )
    )
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
