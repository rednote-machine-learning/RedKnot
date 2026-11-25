#!/usr/bin/env python3
"""Quick 8xH200 demo: reproduce RedKnot V2 (`two-stage` and `single-stage`) vs Full.

Runs real paired Full vs RedKnot V2 inference on three representative 128K
multi-hop RAG batches (12 questions across HotpotQA, 2WikiMQA, MuSiQue, ~3-5 min
per preset) or all 75 batches (300 questions with `--all`).

Usage:
  # Default optimal two-stage recipe (L2->L14 9%+3%=12% >=48K / 20% <48K):
  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset two-stage

  # Optimal zero-extra-cache single-stage recipe (20% everywhere):
  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset single-stage

  # Run both recipes back-to-back and print a combined 3-way table:
  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset both

  # CPU-only checkpoint & runtime preflight (no GPU launch):
  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --check-runtime
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "test"
ASSETS = TEST_ROOT / "longbench300"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark-output" / "redknot-v2-demo"
DEFAULT_DEMO_GROUPS = ("lb-0002", "lb-0027", "lb-0074")

if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


def score_subset_results(data_dir: Path, results_dir: Path, groups: list[str], preset_label: str) -> dict:
    """Score paired Full vs RedKnot V2 outputs for a subset of B4 groups."""
    from longbench300.score_longbench300 import qa_scores, sealed

    rows = []
    by_dataset: dict[str, list[dict]] = {}
    for gid in groups:
        group_meta = json.loads((data_dir / "groups" / f"{gid}.json").read_bytes())
        meta_by_idx = {}
        for ref, idx in zip(group_meta["cases"], group_meta["indices"]):
            row_path = data_dir / ref["path"]
            opener = gzip.open if row_path.suffix == ".gz" else open
            with opener(row_path, "rt", encoding="utf-8") as src:
                meta_by_idx[idx] = json.load(src)
        full_by_idx = {r["index"]: r for r in sealed(results_dir / "groups" / gid / "Full.json")["requests"]}
        rk_by_idx = {r["index"]: r for r in sealed(results_dir / "groups" / gid / "S1.6.json")["requests"]}
        for idx in group_meta["target_indices"]:
            meta, f_req, r_req = meta_by_idx[idx], full_by_idx[idx], rk_by_idx[idx]
            f_f1, _, _ = qa_scores(f_req["prediction_text"], meta["references"])
            r_f1, _, _ = qa_scores(r_req["prediction_text"], meta["references"])
            item = dict(
                group=gid,
                index=idx,
                dataset=meta["dataset"],
                input_tokens=meta["input_tokens"],
                references=meta["references"],
                full_f1=f_f1 * 100.0,
                redknot_f1=r_f1 * 100.0,
                full_ttft_s=(f_req["first_token_ns"] - f_req["received_ns"]) / 1e9,
                redknot_ttft_s=(r_req["first_token_ns"] - r_req["received_ns"]) / 1e9,
                redknot_pred=r_req["prediction_text"].strip().replace("\n", " ")[:80],
            )
            rows.append(item)
            by_dataset.setdefault(meta["dataset"], []).append(item)

    print(f"\nPaired Results: Full vs RedKnot ({preset_label}) · {len(rows)} questions ({len(groups)} B4 groups)")
    print(f"{'Dataset':<13} {'N':>3} {'Full F1':>9} {'RedKnot F1':>11} {'Delta':>8} {'Full TTFT':>10} {'RK TTFT':>10} {'Speedup':>8}")
    overall = {}
    for ds, items in [*by_dataset.items(), ("All", rows)]:
        n = len(items)
        f_f1 = sum(x["full_f1"] for x in items) / n
        r_f1 = sum(x["redknot_f1"] for x in items) / n
        f_t = sum(x["full_ttft_s"] for x in items) / n
        r_t = sum(x["redknot_ttft_s"] for x in items) / n
        spd = f_t / r_t if r_t > 0 else 0.0
        print(f"{ds:<13} {n:>3} {f_f1:>9.2f} {r_f1:>11.2f} {r_f1 - f_f1:>+8.2f} {f_t:>9.3f}s {r_t:>9.3f}s {spd:>7.2f}x")
        if ds == "All":
            overall = dict(n=n, full_f1=f_f1, redknot_f1=r_f1, delta_f1=r_f1 - f_f1,
                           full_ttft_s=f_t, redknot_ttft_s=r_t, speedup=spd)

    print("\nPer-Question Spot Check:")
    for x in rows:
        print(
            f"  [{x['group']} #{x['index']} {x['dataset']} {x['input_tokens']//1000}K] "
            f"Full={x['full_f1']:5.1f} ({x['full_ttft_s']:.2f}s) | "
            f"RedKnot={x['redknot_f1']:5.1f} ({x['redknot_ttft_s']:.2f}s) | "
            f"Ref={x['references'][0]!r} | RK={x['redknot_pred']!r}"
        )
    return dict(preset=preset_label, groups=groups, overall=overall, rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, default=os.environ.get("REDKNOT_MODEL_PATH"),
                        help="local DeepSeek-V4.1-Flash-1017 checkpoint path")
    parser.add_argument("--preset", choices=("two-stage", "single-stage", "both"), default="two-stage",
                        help="RedKnot V2 recipe: two-stage (default, 9%%+3%%), single-stage (20%%, 0 extra cache), or both")
    parser.add_argument("--host-prefix-offload", choices=("auto", "on", "off"), default="auto",
                        help="pinned CPU offload for L3..L8 residuals in two-stage mode (auto: >=192K)")
    parser.add_argument("--groups", type=str, default=",".join(DEFAULT_DEMO_GROUPS),
                        help="comma-separated B4 group IDs (default: lb-0002,lb-0027,lb-0074 = 12 128K QA)")
    parser.add_argument("--all", action="store_true", help="run all 75 B4 groups (300 questions)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT / "cache", help="local dataset cache")
    parser.add_argument("--output", type=Path, default=None, help="new output directory for this run")
    parser.add_argument("--check-runtime", action="store_true",
                        help="validate checkpoint and runtime preflight on CPU without launching GPUs")
    args = parser.parse_args()

    if args.model_path is None:
        parser.error("Set --model-path /path/to/DeepSeek-V4.1-Flash-1017 (or REDKNOT_MODEL_PATH).")

    from longbench300.dataset import prepare_data
    from longbench300.local_backend import load_v2_policy_config_class, run_local
    from longbench300.score_longbench300 import score_cohort

    v2_cls = load_v2_policy_config_class()
    data = prepare_data(ASSETS / "redknot_s16_longbench_300.zip", args.cache.expanduser().resolve())
    selected_groups = None if args.all else [g.strip() for g in args.groups.split(",") if g.strip()]

    if args.output:
        run_dir = args.output.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(mkdtemp(prefix=f"demo-{args.preset}-{stamp}-", dir=DEFAULT_OUTPUT))

    presets = ("single-stage", "two-stage") if args.preset == "both" else (args.preset,)
    print(f"Demo run directory: {run_dir}", flush=True)
    summaries: dict[str, dict] = {}
    preflight: dict[str, dict] = {}

    for preset_name in presets:
        v2_cfg = v2_cls.from_cli(preset=preset_name, host_prefix_offload=args.host_prefix_offload)
        sub_results = run_dir / (f"results-{preset_name}" if len(presets) > 1 else "results")
        sub_report = run_dir / (f"report-{preset_name}" if len(presets) > 1 else "report")
        print(f"\n[{preset_name}] Policy config: {json.dumps(v2_cfg.to_dict(), ensure_ascii=False)}", flush=True)
        result = run_local(
            None, args.model_path, data, sub_results,
            check_only=args.check_runtime, v2_config=v2_cfg, selected_groups=selected_groups,
        )
        if args.check_runtime:
            preflight[preset_name] = result
            continue
        if selected_groups is None:
            ov = score_cohort(data, sub_results, sub_report, expected_contract=result["contract_id"])["overall"]
            full, sparse = ov["Full"], ov["S1.6"]
            summaries[preset_name] = dict(
                preset=preset_name,
                overall=dict(
                    n=ov["paired"], full_f1=full["score_100"], redknot_f1=sparse["score_100"],
                    delta_f1=sparse["score_100"] - full["score_100"],
                    full_ttft_s=full["mean_ttft_s"], redknot_ttft_s=sparse["mean_ttft_s"],
                    speedup=ov["speedup_ratio_of_mean_ttft"],
                ),
                report=str(sub_report / "index.html"),
            )
        else:
            summaries[preset_name] = score_subset_results(data, sub_results, selected_groups, preset_name)

    if args.check_runtime:
        print(json.dumps(preflight if len(presets) > 1 else preflight[presets[0]], indent=2, ensure_ascii=False))
        return

    if len(summaries) > 1:
        print("\nCombined Comparison: Full vs Single-Stage (20%) vs Two-Stage (9%+3%)")
        print(f"{'Recipe':<14} {'N':>4} {'Full F1':>9} {'RedKnot F1':>11} {'Delta':>8} {'Full TTFT':>10} {'RK TTFT':>10} {'Speedup':>8}")
        for name, item in summaries.items():
            ov = item["overall"]
            print(
                f"{name:<14} {ov['n']:>4} {ov['full_f1']:>9.2f} {ov['redknot_f1']:>11.2f} "
                f"{ov['delta_f1']:>+8.2f} {ov['full_ttft_s']:>9.3f}s {ov['redknot_ttft_s']:>9.3f}s {ov['speedup']:>7.2f}x"
            )
    payload = summaries if len(summaries) > 1 else summaries[presets[0]]
    (run_dir / "demo-summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved JSON summary: {run_dir / 'demo-summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"Demo failed: {exc}")
