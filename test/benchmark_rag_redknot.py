#!/usr/bin/env python3
"""Replay the 300-question RedKnot Full/S1.6 benchmark from a local checkout."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

from longbench300.dataset import prepare_data, verify_dataset
from longbench300.score_longbench300 import render_reference, score_cohort


ASSETS = Path(__file__).resolve().parent / "longbench300"
DEFAULT_OUTPUT = ASSETS.parents[1] / "benchmark-output" / "redknot-s16-300"


def print_scores(summary):
    print(f"\n{summary['mode']} · {summary['completed']} paired questions")
    print(f"{'Dataset':<13} {'N':>3} {'Full F1':>8} {'S1.6 F1':>8} "
          f"{'Full TTFT/s':>12} {'S1.6 TTFT/s':>12} {'Speedup':>8}")
    for dataset, row in [*summary["by_dataset"].items(), ("All", summary["overall"])]:
        full, sparse = row["Full"], row["S1.6"]
        print(f"{dataset:<13} {row['paired']:>3} {full['score_100']:>8.2f} {sparse['score_100']:>8.2f} "
              f"{full['mean_ttft_s']:>12.3f} {sparse['mean_ttft_s']:>12.3f} "
              f"{row['speedup_ratio_of_mean_ttft']:>7.3f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="validate and inventory all packaged inputs on CPU")
    modes.add_argument("--check-runtime", action="store_true", help="validate the local native runner and its B4 configuration on CPU")
    modes.add_argument("--reference", action="store_true", help="rescore the packaged historical outputs and timings on CPU")
    parser.add_argument("--runtime-root", type=Path, default=os.environ.get("REDKNOT_RUNTIME_ROOT"),
                        help="optional external S1.6 runtime release (default: use the bundled runtime)")
    parser.add_argument("--model-path", type=Path, default=os.environ.get("REDKNOT_MODEL_PATH"),
                        help="local DeepSeek checkpoint (also REDKNOT_MODEL_PATH)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT / "cache", help="verified local dataset cache")
    parser.add_argument("--output", type=Path, help="new directory for one run (default: a unique benchmark-output subdirectory)")
    parser.add_argument("--preset", choices=("two-stage", "single-stage"), default="two-stage",
                        help="RedKnot V2 recipe: two-stage (default, L2->L14 9%%+3%% >=48K / 20%% <48K) "
                             "or single-stage (20%%, zero extra cache VRAM)")
    parser.add_argument("--host-prefix-offload", choices=("auto", "on", "off"), default="auto",
                        help="pinned CPU offload for L3..L8 residuals in two-stage mode (auto: >=192K)")
    args = parser.parse_args()
    from longbench300.local_backend import load_v2_policy_config_class
    v2_config = load_v2_policy_config_class().from_cli(
        preset=args.preset,
        host_prefix_offload=args.host_prefix_offload,
    )
    if not args.check and not args.reference and args.model_path is None:
        parser.error("GPU replay requires local DeepSeek-V4.1-Flash-1017 weights. Set --model-path or REDKNOT_MODEL_PATH. "
                     "For offline historical output comparisons, use --reference.")

    data = prepare_data(ASSETS / "redknot_s16_longbench_300.zip", args.cache.expanduser().resolve())
    if args.check:
        print(json.dumps(verify_dataset(data), indent=2, ensure_ascii=False))
        return

    if args.output:
        run = args.output.expanduser().resolve()
        run.mkdir(parents=True, exist_ok=False)
    else:
        DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
        mode = "reference" if args.reference else ("check" if args.check_runtime else "gpu")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = Path(mkdtemp(prefix=f"{mode}-{timestamp}-", dir=DEFAULT_OUTPUT))
    print(f"Run directory: {run}", flush=True)
    report = run / "report"
    if args.reference:
        summary = render_reference(data, ASSETS / "historical_reference.zip", report)
    else:
        from longbench300.local_backend import run_local
        result = run_local(args.runtime_root, args.model_path, data, run / "results", args.check_runtime, v2_config=v2_config)
        if args.check_runtime:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        summary = score_cohort(data, run / "results", report, expected_contract=result["contract_id"])
    print_scores(summary)
    print(f"\nComplete answers: {report / 'index.html'}\nMachine-readable results: {report / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"Benchmark failed: {exc}")
