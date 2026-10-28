"""Run with ``python -m benchmarks.redknot --help`` from the repository root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import read_manifest, tokenizer_identity, validate_manifest, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Frozen LongBench RAG inputs and native DeepInfer request measurements")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="show migrated model profiles, native support status, and datasets")

    prepare = commands.add_parser("prepare", help="freeze target-tokenizer inputs from the pinned SGLang release assets (CPU only)")
    prepare.add_argument("--suite", required=True, choices=("64k", "128k", "256k", "440k"))
    prepare.add_argument("--model-profile", default="deepseek-v4.1-flash")
    prepare.add_argument("--tokenizer", required=True, type=Path, help="local target checkpoint/tokenizer directory")
    prepare.add_argument("--tokenizer-revision", help="optional immutable tokenizer commit identifier")
    prepare.add_argument("--trust-remote-code", action="store_true", help="allow the explicitly selected tokenizer's custom Python code")
    prepare.add_argument("--warmup-count", type=int, default=1)
    prepare.add_argument("--max-cases", type=int, help="explicit subset for smoke checks; record it in the manifest")
    prepare.add_argument("--target-tokens", type=int, help="explicit context-budget override for smoke checks")
    prepare.add_argument("--max-new-tokens", type=int, default=64, help="natural-EOS short-answer limit")
    prepare.add_argument(
        "--datasets",
        nargs="+",
        choices=("hotpotqa", "2wikimqa", "musique", "multifieldqa_en", "triviaqa"),
        help="build a new output-blind dataset matrix instead of the source 15-case selection",
    )
    prepare.add_argument("--cases-per-dataset", type=int, default=2)
    prepare.add_argument("--output", required=True, type=Path)

    validate = commands.add_parser("validate", help="check the frozen manifest without importing the GPU runtime")
    validate.add_argument("manifest", type=Path)

    run = commands.add_parser("run", help="run the native full-recompute DeepInfer baseline")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--model-path", required=True, type=Path, help="local checkpoint with weights and matching tokenizer assets")
    run.add_argument("--output-dir", required=True, type=Path, help="new output directory; previous results are never overwritten")
    run.add_argument("--runtime-root", type=Path, help="optional separate, recorded DeepInfer checkout; no implicit backend fallback")
    run.add_argument("--ep", type=int, default=8)
    run.add_argument("--cp", type=int, default=1)
    run.add_argument("--dp", type=int)
    run.add_argument("--warmup-iterations", type=int, default=3)
    run.add_argument("--iterations", type=int, default=10)
    run.add_argument("--max-num-batched-tokens", type=int, default=32768)
    run.add_argument("--memory-utilization", type=float, default=0.85)
    run.add_argument("--seed", type=int, default=0)

    summary = commands.add_parser("summarize", help="summarize completed requests with English QA EM/F1 and native latency")
    summary.add_argument("report", type=Path)
    summary.add_argument("--manifest", required=True, type=Path)

    compare = commands.add_parser("compare", help="strictly compare a qualified native baseline with a real RedKnot implementation")
    compare.add_argument("--manifest", required=True, type=Path)
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--redknot", required=True, type=Path)
    compare.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "catalog":
            from .catalog import DATASETS, MODEL_PROFILES, SUITES

            print(json.dumps({"models": MODEL_PROFILES, "datasets": DATASETS, "suites": SUITES}, indent=2, ensure_ascii=False))
        elif args.command == "prepare":
            from .prepare import prepare_manifest
            from .prompts import prompt_protocol

            identity = tokenizer_identity(args.tokenizer, revision=args.tokenizer_revision)
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ValueError(
                    "prepare requires the target model's Transformers/tokenizers environment; GPU runtime is not required"
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=args.trust_remote_code)
            renderer, protocol = prompt_protocol(args.model_profile, tokenizer)
            manifest = prepare_manifest(
                args.suite,
                args.model_profile,
                tokenizer,
                tokenizer_identity=identity,
                warmup_count=args.warmup_count,
                max_cases=args.max_cases,
                target_tokens=args.target_tokens,
                max_new_tokens=args.max_new_tokens,
                prompt_renderer=renderer,
                prompt_protocol=protocol,
                datasets=args.datasets,
                cases_per_dataset=args.cases_per_dataset,
            )
            validate_manifest(manifest)
            path = write_json(args.output, manifest)
            print(json.dumps({"manifest": str(path), "sha256": manifest["sha256"], "cases": len(manifest["cases"])}))
        elif args.command == "validate":
            manifest = read_manifest(args.manifest)
            print(
                json.dumps(
                    {"valid": True, "sha256": manifest["sha256"], "cases": len(manifest["cases"]), "warmup": len(manifest["warmup"])}
                )
            )
        elif args.command == "run":
            from .runner import run_benchmark

            report = run_benchmark(
                read_manifest(args.manifest),
                model_path=str(args.model_path),
                output_dir=str(args.output_dir),
                ep=args.ep,
                cp=args.cp,
                dp=args.dp,
                warmup_iterations=args.warmup_iterations,
                iterations=args.iterations,
                max_num_batched_tokens=args.max_num_batched_tokens,
                memory_utilization=args.memory_utilization,
                seed=args.seed,
                runtime_root=str(args.runtime_root) if args.runtime_root else None,
            )
            print(json.dumps({"report": str(report)}))
        elif args.command == "summarize":
            from .compare import validate_run

            report = json.loads(args.report.read_text())
            summary = validate_run(report, read_manifest(args.manifest), mode=report["mode"], require_warmup=False)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        elif args.command == "compare":
            from .compare import compare_runs

            comparison = compare_runs(
                json.loads(args.baseline.read_text()), json.loads(args.redknot.read_text()), read_manifest(args.manifest)
            )
            if args.output:
                write_json(args.output, comparison)
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as exc:
        print(f"redknot benchmark: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
