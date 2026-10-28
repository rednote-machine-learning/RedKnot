"""Strict paired benchmark comparison; CPU-only and standard-library-only.

An activation record is an explicit adapter attestation, not independent GPU
verification. The current native runner emits baseline runs only. This module
does not synthesize a RedKnot arm or substitute a second baseline for one.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contract import BOUNDARY, digest, read_manifest, seal, validate_manifest, write_json
from .metrics import finite_number, summarize_requests
from .native import ADMISSION_POLICY

RUN_SCHEMA = "redknot-deepinfer-benchmark-run-v1"
MATCH_METADATA = ("model", "tokenizer", "runtime", "settings", "hardware", "sampling", "warmup", "measurement_boundary")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), "report contains a non-finite number")
    elif isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_nonfinite(child)


def _same(actual: Any, expected: Any, name: str) -> None:
    _require(digest(actual) == digest(expected), f"{name} mismatch")


def _same_number(actual: Any, expected: float, name: str) -> None:
    number = finite_number(actual, name)
    _require(math.isclose(number, expected, rel_tol=1e-8, abs_tol=1e-8), f"inconsistent {name}")


def validate_run(report: dict, manifest: dict, *, mode: str, require_warmup: bool = True) -> dict:
    """Recompute metrics and prove every declared rank/iteration has all cases."""
    validate_manifest(manifest)
    _require(isinstance(report, dict), "report must be an object")
    _reject_nonfinite(report)
    if "sha256" in report:
        _same(report["sha256"], seal(report)["sha256"], "report SHA-256")
    _require(report.get("schema") == RUN_SCHEMA, "unsupported run schema")
    _require(report.get("status") == "complete" and not report.get("error"), "run is incomplete or failed")
    _require(mode in ("baseline", "redknot") and report.get("mode") == mode, f"expected a real {mode} run")
    _same(report.get("manifest_sha256"), manifest["sha256"], "manifest")
    metadata = report.get("metadata")
    _require(isinstance(metadata, dict), "run metadata is missing")
    for field in MATCH_METADATA:
        _require(field in metadata and metadata[field] not in (None, {}, [], ""), f"missing comparison metadata: {field}")
    _require(metadata["measurement_boundary"] == BOUNDARY, "unsupported TTFT boundary")
    model = metadata["model"]
    _require(isinstance(model, dict), "model provenance is missing")
    for field in ("profile", "model_id"):
        _same(model.get(field), manifest["model"][field], f"model {field}")
    tokenizer = metadata["tokenizer"]
    _require(isinstance(tokenizer, dict), "tokenizer provenance is missing")
    _same(tokenizer.get("fingerprint"), manifest["model"]["tokenizer"]["fingerprint"], "tokenizer fingerprint")
    settings = metadata["settings"]
    _require(isinstance(settings, dict), "settings must be an object")
    iterations, dp_size = settings.get("iterations"), settings.get("dp_size")
    _require(type(iterations) is int and iterations > 0 and type(dp_size) is int and dp_size > 0, "invalid iteration/DP count")
    _require(settings.get("replication") == "full_cohort_per_dp_rank", "unknown workload replication policy")
    admission_batch_size = settings.get("admission_batch_size")
    expected_batches = None
    if "admission_batch_size" in settings:
        _require(type(admission_batch_size) is int and admission_batch_size > 0, "invalid admission batch size")
        _require(settings.get("admission_policy") == ADMISSION_POLICY, "unknown admission batch policy")
        ordered_ids = [case["sample_id"] for case in manifest["cases"]]
        expected_batches = [ordered_ids[start : start + admission_batch_size] for start in range(0, len(ordered_ids), admission_batch_size)]
    else:
        _require("admission_policy" not in settings, "admission policy requires a batch size")
    warmup = metadata["warmup"]
    _require(isinstance(warmup, dict), "warmup policy is missing")
    _require(type(warmup.get("iterations")) is int and warmup["iterations"] >= 0, "invalid warmup count")
    _require(warmup.get("complete") is (warmup["iterations"] > 0), "warmup did not complete")
    if require_warmup:
        _require(warmup["iterations"] > 0, "hot comparison requires warmup")
        _require(report.get("comparison_eligible", True) is True, "run is explicitly ineligible for comparisons")
    _require(
        type(warmup.get("cases_per_rank")) is int and warmup["cases_per_rank"] == len(manifest["warmup"]),
        "warmup must cover the manifest's calibration cases on every rank",
    )
    if warmup["iterations"]:
        _require(warmup["cases_per_rank"] > 0, "warmup requires calibration cases")
    if admission_batch_size is not None:
        active_warmup_count = len(manifest["warmup"]) if warmup["iterations"] else 0
        _require(
            type(settings.get("max_num_seqs")) is int
            and settings["max_num_seqs"] == min(admission_batch_size, max(active_warmup_count, len(manifest["cases"]))),
            "max_num_seqs differs from the declared admission batch bound",
        )
    activation = report.get("activation")
    _require(isinstance(activation, dict) and activation.get("verified") is True, "missing verified activation attestation")
    _require(activation.get("redknot_enabled") is (mode == "redknot"), "mode and execution activation disagree")
    _require(isinstance(activation.get("implementation"), str) and bool(activation["implementation"].strip()), "implementation missing")
    if mode == "redknot":
        _require(
            isinstance(activation.get("configuration_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", activation["configuration_sha256"]) is not None,
            "RedKnot requires an explicit implementation/configuration attestation; relabeling a baseline is invalid",
        )
    phases = report.get("phases")
    _require(isinstance(phases, list) and len(phases) == iterations, "missing measured phases")
    phase_map, batch_map, elapsed_total = {}, {}, 0.0
    expected_per_phase = len(manifest["cases"]) * dp_size
    for phase in phases:
        _require(isinstance(phase, dict), "invalid phase")
        iteration = phase.get("iteration")
        _require(type(iteration) is int and 0 <= iteration < iterations and iteration not in phase_map, "invalid or duplicate phase")
        rank_elapsed = phase.get("rank_elapsed_s")
        _require(isinstance(rank_elapsed, list) and len(rank_elapsed) == dp_size, "missing rank durations")
        rank_elapsed = [finite_number(value, "rank elapsed_s", positive=True) for value in rank_elapsed]
        duration = finite_number(phase.get("elapsed_s"), "phase elapsed_s", positive=True)
        if expected_batches is None:
            _require("batches" not in phase, "undeclared admission batches")
            _same_number(duration, max(rank_elapsed), "synchronized global phase duration")
        else:
            batches = phase.get("batches")
            _require(isinstance(batches, list) and len(batches) == len(expected_batches), "missing admission batches")
            rank_totals, global_total = [0.0] * dp_size, 0.0
            for batch_index, (batch, sample_ids) in enumerate(zip(batches, expected_batches, strict=True)):
                _require(isinstance(batch, dict), "invalid admission batch")
                _require(
                    type(batch.get("batch_index")) is int and batch["batch_index"] == batch_index, "admission batches are out of order"
                )
                _same(batch.get("sample_ids"), sample_ids, "admission batch manifest order/coverage")
                batch_rank_elapsed = batch.get("rank_elapsed_s")
                _require(isinstance(batch_rank_elapsed, list) and len(batch_rank_elapsed) == dp_size, "missing batch rank durations")
                batch_rank_elapsed = [finite_number(value, "batch rank elapsed_s", positive=True) for value in batch_rank_elapsed]
                batch_duration = finite_number(batch.get("elapsed_s"), "batch elapsed_s", positive=True)
                _same_number(batch_duration, max(batch_rank_elapsed), "synchronized global batch duration")
                _require(
                    type(batch.get("completed_requests")) is int and batch["completed_requests"] == len(sample_ids) * dp_size,
                    "partial admission batch",
                )
                batch_map[iteration, batch_index] = batch_rank_elapsed
                rank_totals = [total + elapsed for total, elapsed in zip(rank_totals, batch_rank_elapsed, strict=True)]
                global_total += batch_duration
            for rank, total in enumerate(rank_totals):
                _same_number(rank_elapsed[rank], total, "rank admission batch duration sum")
            _same_number(duration, global_total, "global sum of synchronized batch durations")
        _require(type(phase.get("completed_requests")) is int and phase["completed_requests"] == expected_per_phase, "partial phase")
        phase_map[iteration] = rank_elapsed
        elapsed_total += duration
    _same_number(report.get("elapsed_s"), elapsed_total, "total global elapsed_s")
    requests = report.get("requests")
    _require(isinstance(requests, list), "request observations missing")
    expected_count = expected_per_phase * iterations
    _require(
        type(report.get("completed_requests")) is int and report["completed_requests"] == len(requests) == expected_count,
        "missing or extra completed requests",
    )
    cases = {case["sample_id"]: case for case in manifest["cases"]}
    identities = set()
    for record in requests:
        _require(isinstance(record, dict), "invalid request record")
        iteration, rank, sample_id = record.get("iteration"), record.get("dp_rank"), record.get("sample_id")
        _require(type(iteration) is int and iteration in phase_map, "request iteration is invalid")
        _require(type(rank) is int and 0 <= rank < dp_size and sample_id in cases, "request rank/sample is invalid")
        identity = (iteration, rank, sample_id)
        _require(identity not in identities, "duplicate request identity")
        identities.add(identity)
        case = cases[sample_id]
        for field in ("dataset", "input_ids_sha256", "references", "max_new_tokens", "scored", "ignore_eos"):
            _same(record.get(field), case[field], f"request {sample_id} {field}")
        latency = finite_number(record.get("latency_s"), "request latency", positive=True)
        if expected_batches is None:
            _require("batch_index" not in record, "undeclared request admission batch")
            _require(latency <= phase_map[iteration][rank] + 1e-8, "request latency exceeds its rank's phase")
        else:
            batch_index = record.get("batch_index")
            _require(type(batch_index) is int and 0 <= batch_index < len(expected_batches), "invalid request admission batch")
            _require(sample_id in expected_batches[batch_index], "request belongs to another admission batch")
            _require(latency <= batch_map[iteration, batch_index][rank] + 1e-8, "request latency exceeds its rank's admission batch")
    expected = {(iteration, rank, sample_id) for iteration in range(iterations) for rank in range(dp_size) for sample_id in cases}
    _require(identities == expected, "incomplete iteration/rank cohort")
    summary = summarize_requests(requests, elapsed_s=elapsed_total)
    if "summary" in report:
        _same(report["summary"], summary, "derived summary")
    return summary


def compare_runs(baseline: dict, redknot: dict, manifest: dict) -> dict:
    """Compare true baseline / RedKnot arms with identical measured workloads."""
    baseline_summary = validate_run(baseline, manifest, mode="baseline")
    redknot_summary = validate_run(redknot, manifest, mode="redknot")
    for field in MATCH_METADATA:
        _same(baseline["metadata"][field], redknot["metadata"][field], f"comparison {field}")
    baseline_records = {(row["iteration"], row["dp_rank"], row["sample_id"]): row for row in baseline["requests"]}
    ratios = [
        baseline_records[(row["iteration"], row["dp_rank"], row["sample_id"])]["ttft_s"] / row["ttft_s"] for row in redknot["requests"]
    ]
    dataset_deltas = {}
    for dataset, quality in baseline_summary["datasets"].items():
        candidate = redknot_summary["datasets"][dataset]
        dataset_deltas[dataset] = {
            "unique_scored_samples": quality["unique_scored_samples"],
            "answer_em_delta_pp": None if quality["answer_em"] is None else 100 * (candidate["answer_em"] - quality["answer_em"]),
            "answer_token_f1_delta_pp": None
            if quality["answer_token_f1"] is None
            else 100 * (candidate["answer_token_f1"] - quality["answer_token_f1"]),
        }
    controlled_qps = baseline_summary["fixed_output_qps_eligible"] and redknot_summary["fixed_output_qps_eligible"]
    return seal(
        {
            "schema": "redknot-deepinfer-benchmark-comparison-v1",
            "manifest_sha256": manifest["sha256"],
            "baseline": baseline_summary,
            "redknot": redknot_summary,
            "ttft_median_paired_speedup": statistics.median(ratios),
            "ttft_p50_ratio": baseline_summary["ttft_s"]["p50"] / redknot_summary["ttft_s"]["p50"],
            "fixed_output_qps_speedup": (
                redknot_summary["completed_requests_per_s"] / baseline_summary["completed_requests_per_s"] if controlled_qps else None
            ),
            "dataset_quality_deltas": dataset_deltas,
            "verified_equal_metadata": list(MATCH_METADATA),
            "limits": [
                "Activation attestations are an explicit report contract, not independent proof of GPU execution.",
                "Natural-EOS and mixed-output suites report observed request QPS, not a controlled fixed-output throughput speedup.",
                "Repeated observations do not increase the independent quality sample count or certify a generalization bound.",
            ],
        }
    )


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("redknot", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_runs(_load(args.baseline), _load(args.redknot), read_manifest(args.manifest))
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"comparison rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
