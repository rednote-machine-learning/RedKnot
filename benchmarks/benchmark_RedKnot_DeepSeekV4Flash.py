#!/usr/bin/env python3
"""vLLM-only RedKnot Flash entrypoint: CPU plan by default, GPU by explicit opt-in.

Examples (run from this repository):
  ./benchmarks/run_deepseek_v4_flash_reproduction.sh --dry-run
  ./benchmarks/run_deepseek_v4_flash_reproduction.sh --suite custom \
      --cases /data/exact-token-cases.json --prepare-only --output /data/new-cases.json
  CUDA_VISIBLE_DEVICES=0 ./benchmarks/run_deepseek_v4_flash_reproduction.sh \
      --suite custom --cases /data/new-cases.json --run --gpu-confirmed-idle \
      --model /workspace/Models/DeepSeek-V4-Flash-0731 --output /data/new-result.json

No model download, dependency installation, SGLang import or GPU keeper changes.
Before a real run, independently hash every pinned model file, then use the
existing paired vLLM benchmark: 3 untimed warmup pairs, 10 measured pairs,
alternating Recomputed/RedKnot, identical greedy fixed-length outputs (128 default).
The original 64K..440K SGLang suite is NOT certified by this entrypoint. Frozen
suite token exports and sufficient CPU/GPU memory are required; the default
8-GiB projected cache cannot hold even a complete 64K frozen case.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Permit direct execution without installing the benchmark directory as a wheel.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import benchmark_redknot as paired
from benchmarks import flash_reproduction as prep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="CPU plan only (default)")
    modes.add_argument(
        "--prepare-only", action="store_true", help="Write exact cases, no GPU"
    )
    modes.add_argument(
        "--run", action="store_true", help="Explicitly initialize one GPU"
    )
    parser.add_argument(
        "--suite", choices=(*prep.LENGTHS, "all", "custom"), default="all"
    )
    parser.add_argument(
        "--cases", type=Path, help="Exact tokenized JSON; required to prepare/run"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=prep.HERE.parent / "examples/deepseek_v4_flash_policy.json",
    )
    parser.add_argument(
        "--model", type=Path, help="Complete local Flash-0731 checkpoint for --run"
    )
    parser.add_argument("--output", type=Path, help="New JSON file; never overwritten")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--warmup-pairs", type=int, default=3)
    parser.add_argument("--measured-pairs", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--gpu-confirmed-idle",
        action="store_true",
        help="Operator confirmed selected GPU is free and its keeper yielded",
    )
    return parser


def new_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("--output is required for --prepare-only/--run")
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists: {path}")
    if not path.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    return path


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from vllm_redknot.config import load_config, policy_fingerprint

    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = args.max_model_len
    if args.warmup_pairs < 3 or args.measured_pairs < 10 or args.output_tokens < 50:
        raise ValueError(
            "requires >=3 warmup pairs, >=10 measured pairs, >=50 output tokens"
        )
    if args.max_model_len < 1 or args.max_num_batched_tokens < args.max_model_len:
        raise ValueError(
            "unchunked prefill requires batched tokens >= max model len > 0"
        )
    policy = load_config(args.config)
    prep.validate_flash_policy(policy)
    catalog = prep.load_catalog(args.suite)
    document = None
    if args.cases:
        document = prep.convert_cases(
            prep.read_json(args.cases), catalog, custom=args.suite == "custom"
        )
    elif args.suite == "custom" or args.prepare_only or args.run:
        raise ValueError(
            "--cases is required; frozen manifests do not contain token IDs"
        )
    parsed = paired.parse_cases(document) if document else None
    plan = prep.cache_plan(
        policy,
        cases=parsed,
        catalog=catalog,
        output_tokens=args.output_tokens,
        max_model_len=args.max_model_len,
    )
    if not document:
        plan["blockers"].insert(0, "exact token exports unavailable; catalog only")
    plan.update(
        {
            "status": "cpu_plan_only",
            "gpu_initialized": False,
            "model_verified": False,
            "suite": args.suite,
            "engine_family": "deepseek_v4_flash",
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "prefix_caching": False,
            "chunked_prefill": False,
            "source_reproduction_equivalent": False,
            "policy_fingerprint": policy_fingerprint(policy),
            "measurement": {
                "warmup_pairs": args.warmup_pairs,
                "measured_pairs": args.measured_pairs,
                "fixed_output_tokens": args.output_tokens,
                "ttft_source": "vLLM output.metrics.first_token_latency",
                "qps_measured": False,
            },
        }
    )
    return plan, document


def aggregate_means(report: dict[str, Any]) -> dict[str, Any]:
    """Explicitly separate ratio-of-means from mean-of-paired-ratios."""
    rows = report["measurements"]
    values = {
        mode: [r["ttft_seconds"] for r in rows if r["mode"] == mode]
        for mode in ("dense", "reuse")
    }
    complete = all(v and all(x is not None for x in v) for v in values.values())
    means = {
        mode: statistics.mean(v) if complete else None for mode, v in values.items()
    }
    pairs: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        pairs.setdefault((row["case_id"], row["pair_index"]), {})[row["mode"]] = row[
            "ttft_seconds"
        ]
    ratios = [
        p["dense"] / p["reuse"]
        for p in pairs.values()
        if set(p) == {"dense", "reuse"} and p["dense"] and p["reuse"]
    ]
    return {
        "mean_ttft_seconds": means,
        "ratio_of_mean_ttft_dense_over_reuse": means["dense"] / means["reuse"]
        if complete
        else None,
        "mean_paired_ttft_ratio_dense_over_reuse": statistics.mean(ratios)
        if complete and len(ratios) == len(pairs)
        else None,
        "f1_drop_percentage_points": report["summary"]["f1_drop_percentage_points"],
        "warmups_and_capture_excluded": True,
        "qps_measured": False,
    }


def run_gpu(args: argparse.Namespace, plan: dict, document: dict) -> dict:
    if not args.gpu_confirmed_idle:
        raise ValueError(
            "--gpu-confirmed-idle is required; "
            "this wrapper never stops other GPU processes"
        )
    if args.model is None:
        raise ValueError("--model is required for --run")
    if plan["blockers"]:
        raise ValueError("GPU preflight blocked: " + "; ".join(plan["blockers"]))
    prep.validate_gpu_environment(os.environ)
    from vllm_redknot.cli import installed_plugin
    from vllm_redknot.compat import verify_vllm_sources

    source = verify_vllm_sources(engine_family="deepseek_v4_flash")
    if not installed_plugin():
        raise ValueError(
            "install this plugin into the pinned vLLM interpreter before running"
        )
    output = new_path(args.output)
    verification_path = new_path(
        output.with_name(output.name + ".model-verification.json")
    )
    manifest = prep.model_file_inventory()
    model_root = args.model.expanduser().absolute()
    verified = prep.verify_local_files(model_root, manifest["files"])
    verification = {
        "verified": True,
        "revision": prep.FLASH_REVISION,
        "model": str(model_root),
        "file_count": len(verified),
        "total_bytes": sum(item["size"] for item in verified),
        "files": verified,
        "source_contract": source,
    }
    paired.write_report(verification_path, verification)
    args.dtype = "bfloat16"
    kwargs, metadata = paired.prepare_engine(args)
    checkpoint = prep.read_json(model_root / "config.json")
    if checkpoint.get("o_lora_rank") != prep.FLASH_O_LORA_RANK:
        raise ValueError(
            "checkpoint output projection rank differs from budget formula"
        )
    parsed = paired.parse_cases(document)
    vocab_size = checkpoint.get("vocab_size")
    if type(vocab_size) is not int or any(
        t >= vocab_size for c in parsed for t in c.tokens
    ):
        raise ValueError("case token IDs exceed checkpoint vocabulary")
    paired._validate_run(
        parsed,
        args.warmup_pairs,
        args.measured_pairs,
        args.output_tokens,
        args.max_model_len,
        args.max_num_batched_tokens,
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
            max_num_batched_tokens=args.max_num_batched_tokens,
            warmup_pairs=args.warmup_pairs,
            measured_pairs=args.measured_pairs,
            output_tokens=args.output_tokens,
            engine_family="deepseek_v4_flash",
        )
        report.update(
            engine=kwargs,
            configuration=metadata,
            cpu_preflight=plan,
            provenance=document["provenance"],
            model_verification=str(verification_path),
            source_reproduction_equivalent=False,
        )
        report["aggregate_means"] = aggregate_means(report)
        paired.write_report(output, report)
        return report
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30)


def print_outputs(report: dict[str, Any]) -> None:
    for case in report["cases"]:
        print(f"\n=== {case['id']} ===")
        rows = [
            r
            for r in report["measurements"]
            if r["case_id"] == case["id"] and r["pair_index"] == 0
        ]
        for mode, label in (("dense", "Recomputed"), ("reuse", "RedKnot")):
            row = next(r for r in rows if r["mode"] == mode)
            print(f"[{label}]\n{row['output']['text']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.prepare_only or args.run:
            new_path(args.output)
        plan, document = prepare(args)
        if args.prepare_only:
            document["cpu_preflight"] = plan
            paired.write_report(args.output, document)
            print(
                json.dumps(
                    {
                        "status": "prepared_not_gpu_verified",
                        "output": str(args.output),
                        "blockers": plan["blockers"],
                    },
                    indent=2,
                )
            )
        elif args.run:
            report = run_gpu(args, plan, document)
            print_outputs(report)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "qualified": report["qualified"],
                        "aggregate_means": report["aggregate_means"],
                        "qualification_reasons": report["qualification_reasons"],
                    },
                    indent=2,
                )
            )
            return 0 if report["qualified"] else 2
        else:
            print(json.dumps(plan, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Flash reproduction failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
