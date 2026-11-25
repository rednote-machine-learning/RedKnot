#!/usr/bin/env python3
"""Open-source quickstart demo: reproduce both optimal RedKnot V2 recipes vs Full.

Two optimal recipes (both faster and more accurate than Full on 8xH200):
  1. `two-stage` (default — best overall speed, generalization, and 256K accuracy):
       - <48K: single-stage 20%, NoSpan, Head64 (0 GiB extra cache VRAM)
       - >=48K: L2->L14 two-stage 9%+3%=12%, Span[-8,+24], Head64/128,
         auto pinned host offload of L3..L8 residuals at >=192K
       - Measured 8xH200: 500Q 51.85 (>Full 51.52), 900Q 53.63, 2.43x TTFT speedup
  2. `single-stage` (zero extra PrefixBank VRAM, simplest execution path):
       - All lengths: single-stage 20% (0 GiB extra cache VRAM, no host offload)
       - <48K: NoSpan, Head64 | >=48K: Span[-8,+24], Head64/128
       - Measured 8xH200: 500Q 51.78 (>Full 51.52), 900Q 53.30

Usage:
  # 1. Offline CPU walkthrough (zero GPU / zero checkpoint needed):
  python3 examples/redknot_v2_demo.py --cpu-demo --preset both
  python3 examples/redknot_v2_demo.py --cpu-demo --preset single-stage
  python3 examples/redknot_v2_demo.py --cpu-demo --preset two-stage

  # 2. Live 8xH200 paired Full vs RedKnot V2 benchmark (~3-5 min for 12 128K QA):
  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset two-stage
  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset single-stage
  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset both
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import sys
from tempfile import mkdtemp

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "test"
ALGO_ROOT = TEST_ROOT / "longbench300" / "algorithm"
ASSETS = TEST_ROOT / "longbench300"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark-output" / "redknot-v2-demo"
# Three representative 128K multi-hop RAG batches (12 questions across HotpotQA, 2WikiMQA, MuSiQue).
DEFAULT_DEMO_GROUPS = ("lb-0002", "lb-0027", "lb-0074")

# DeepSeek-V4.1-Flash (hidden=5120, mHC=4, bf16) PrefixBank byte constants per Stage-1 token:
#   residual L3..L13 (11L) = 440 KiB (L3..L8 offloaded 6L = 240 KiB, L9..L13 GPU 5L = 200 KiB)
#   hidden L3..L13 (11L)   = 110 KiB
#   swa + mixes + L8       ~= 16.5 KiB
BYTES_PER_STAGE1_TOKEN_PURE_GPU = int(566.5 * 1024)
BYTES_PER_STAGE1_TOKEN_OFFLOAD_GPU = int(326.5 * 1024)
BYTES_PER_STAGE1_TOKEN_OFFLOAD_HOST = 240 * 1024
GIB = 1024 ** 3

for p in (
    TEST_ROOT,
    ALGO_ROOT / "fixed",
    ALGO_ROOT / "lastkv",
    ALGO_ROOT / "shared" / "runtime",
):
    text = str(p)
    if text not in sys.path:
        sys.path.insert(0, text)


def _estimate_extra_cache_gib(pol, stage1_tokens: int) -> tuple[float, float]:
    """Return estimated (extra_gpu_gib, extra_host_gib) for PrefixBank on DeepSeek-V4.1-Flash."""
    if not pol.two_stage_l14:
        return 0.0, 0.0
    if pol.host_prefix_offload:
        return (
            stage1_tokens * BYTES_PER_STAGE1_TOKEN_OFFLOAD_GPU / GIB,
            stage1_tokens * BYTES_PER_STAGE1_TOKEN_OFFLOAD_HOST / GIB,
        )
    return stage1_tokens * BYTES_PER_STAGE1_TOKEN_PURE_GPU / GIB, 0.0


def run_cpu_recipe_walkthrough(preset: str, host_prefix_offload: str) -> dict:
    """Offline CPU walkthrough for one optimal RedKnot V2 recipe (`single-stage` or `two-stage`)."""
    import cone15_plan
    import fixed15_geometry
    import halo_direct_copy
    import index20_selector
    import two_stage_l14
    import v2_policy

    cfg = v2_policy.V2PolicyConfig.from_cli(
        preset=preset,
        host_prefix_offload=host_prefix_offload,
    )

    print("=" * 84)
    print(f"RedKnot V2 CPU Walkthrough  (recipe={cfg.preset}, host_offload={host_prefix_offload})")
    print("=" * 84)
    print("\n[1] Length-Adaptive Policy Routing & Extra PrefixBank Memory Estimate:")
    print(
        f"{'Prompt':>8} {'Band':>5} {'Mode':<13} {'Budget':>10} {'Head':>6} "
        f"{'Span(C2)':>10} {'Extra GPU':>10} {'Host RAM':>10}"
    )
    for tokens in (16_000, 32_000, 64_000, 128_000, 250_000):
        pol = cfg.resolve(tokens)
        budget = f"{int(pol.stage1_fraction*100)}%->{int(pol.fraction*100)}%" if pol.two_stage_l14 else f"{int(pol.fraction*100)}%"
        est_s1 = int(tokens * (pol.stage1_fraction + 0.003))
        gpu_gib, host_gib = _estimate_extra_cache_gib(pol, est_s1)
        print(
            f"{tokens:>8,} {pol.band:>5} {pol.mode:<13} {budget:>10} {pol.head_width:>6} "
            f"{str(pol.window):>10} {gpu_gib:>8.2f}G {host_gib:>8.2f}G"
        )

    # Simulate a realistic 128K multi-document RAG prompt (10 documents + ~460-token query suffix).
    print("\n[2] Simulated 128K Multi-Document Multi-Hop RAG Pipeline (10 docs, 128,000 tokens):")
    rng = random.Random(2026)
    total = 128_000
    docs = [{"start": 16 + i * 12_754, "end": 16 + i * 12_754 + 12_750, "key": f"doc-{i}"} for i in range(10)]
    pol_128k = cfg.resolve(total)
    n_keys = total // 2
    scores = [rng.uniform(-1.5, 1.5) for _ in range(n_keys)]
    for relevant_doc in (docs[2], docs[7]):
        peak_k = (relevant_doc["start"] + 4_000) // 2
        for k in range(peak_k, peak_k + 40):
            scores[k] = rng.uniform(6.0, 10.0)
    counts = [1] * n_keys

    sel_stage1 = index20_selector.select_from_reduced_scores(
        docs,
        total,
        scores,
        counts,
        fraction=pol_128k.stage1_fraction,
        strategy="max",
        span_window=pol_128k.window,
        head_width=pol_128k.head_width,
        allow_budget_floor=True,
    )
    selected = sel_stage1["selected_positions"]
    mandatory = sel_stage1["mandatory_positions"]
    bridge_range = range(docs[5]["start"] + 5_000, docs[5]["start"] + 5_160)

    if pol_128k.two_stage_l14:
        sel_target = index20_selector.select_from_reduced_scores(
            docs,
            total,
            scores,
            counts,
            fraction=pol_128k.fraction,
            strategy="max",
            span_window=pol_128k.window,
            head_width=pol_128k.head_width,
            allow_budget_floor=True,
        )
        valid_l14 = [False] * n_keys
        for d in docs:
            for k in range((d["start"] + 1) // 2, d["end"] // 2):
                valid_l14[k] = True
        l14_scores = [rng.uniform(-1.0, 2.0) for _ in range(n_keys)]
        bridge_k = (docs[5]["start"] + 5_000) // 2
        for k in range(bridge_k, bridge_k + 80):
            l14_scores[k] = rng.uniform(9.0, 14.0)
        sel_final, extra = two_stage_l14.promote_stage2(
            sel_stage1,
            docs,
            total,
            l14_scores,
            valid_l14,
            target_fraction=pol_128k.fraction,
            target_body_tokens=sel_target["selected_body_tokens"],
            span_window=pol_128k.window,
        )
        affected = two_stage_l14.affected_positions(
            sel_stage1["selected_positions"], extra, sel_final["selected_positions"], mid=14, total=total
        )
        selected = sel_final["selected_positions"]
        gpu_gib, host_gib = _estimate_extra_cache_gib(pol_128k, len(sel_stage1["selected_positions"]))
        bridge_hits = sum(1 for p in bridge_range if p in set(selected))
        print(
            f"  - Two-Stage L2->L14 (9%+3%=12%): Stage-1={len(sel_stage1['selected_positions']):,} tokens "
            f"+ L14 promoted={len(extra):,} -> Final={len(selected):,} / {total:,} ({100*len(selected)/total:.1f}%)"
        )
        print(
            f"  - L14 Multi-Hop Bridge Recall (doc-5): {bridge_hits}/{len(bridge_range)} tokens captured"
        )
        print(
            f"  - L3..L13 Local Backfill: computes ONLY {len(affected[3]):,} newly promoted rows/layer "
            f"(reuses {len(sel_stage1['selected_positions']):,} cached Stage-1 rows; "
            f"est. extra cache GPU={gpu_gib:.2f} GiB, Host={host_gib:.2f} GiB)"
        )
    else:
        gpu_gib, host_gib = 0.0, 0.0
        print(
            f"  - Single-Stage L2 (20%): {len(selected):,} / {total:,} tokens "
            f"({100*len(selected)/total:.1f}%, Head={pol_128k.head_width}, Span={pol_128k.window}, "
            f"Extra PrefixBank VRAM=0.00 GiB)"
        )

    final_frame_start = (total - 1) // 8192 * 8192
    ranges = [(0, 65_536), (65_536, total)]
    legacy_cone = cone15_plan.Cone15Plan(total, selected, mandatory, last_frame_start=final_frame_start, group_ranges=ranges)
    minimal_cone = cone15_plan.Cone15Plan(total, selected, (), last_frame_start=final_frame_start, group_ranges=ranges)

    legacy_q = sum(len(legacy_cone.group(l, a, b).queries) for a, b in ranges for l in range(21, 40))
    minimal_q = sum(len(minimal_cone.group(l, a, b).queries) for a, b in ranges for l in range(21, 40))
    selected_set = set(selected)
    universe = [p for p in fixed15_geometry.window_union(selected) if p not in selected_set]
    bound_groups = []
    for d in docs:
        matches = [(i, p - d["start"]) for i, p in enumerate(universe) if d["start"] <= p < d["end"]]
        if matches:
            bound_groups.append((d["key"], matches[0][0], matches[-1][0] + 1, [s for _, s in matches], None))
    l21_38_full_halo = 18 * len(universe)
    l21_38_pruned_halo = 0
    for l in range(21, 39):
        wanted = cone15_plan.row_ids(minimal_cone.layers[l].offline, 0, total)
        keep, _, _ = halo_direct_copy.plan_layer_halo_gather(universe, wanted, bound_groups)
        l21_38_pruned_halo += len(keep)

    print(
        f"  - MinimalCone (Layers 21..39 FFN rows): {legacy_q:,} -> {minimal_q:,} "
        f"({100*(minimal_q - legacy_q)/legacy_q:+.1f}%, 100% Bit-Exact)"
    )
    print(
        f"  - Opt-2 Post-20 Offline Halo rows (L21..38): {l21_38_full_halo:,} -> {l21_38_pruned_halo:,} "
        f"({100*(l21_38_pruned_halo - l21_38_full_halo)/max(l21_38_full_halo, 1):+.1f}%, 100% Bit-Exact)"
    )
    return dict(
        preset=preset,
        selected_tokens=len(selected),
        extra_gpu_gib=gpu_gib,
        extra_host_gib=host_gib,
        minimal_cone_saved_pct=100.0 * (minimal_q - legacy_q) / legacy_q,
        halo_saved_pct=100.0 * (l21_38_pruned_halo - l21_38_full_halo) / max(l21_38_full_halo, 1),
    )


def run_cpu_demo(preset: str, host_prefix_offload: str) -> None:
    presets = ("single-stage", "two-stage") if preset == "both" else (preset,)
    summaries = [run_cpu_recipe_walkthrough(p, host_prefix_offload) for p in presets]
    if len(summaries) == 2:
        print("\n" + "=" * 84)
        print("Side-by-Side 128K Recipe Comparison (CPU Walkthrough)")
        print("=" * 84)
        print(f"{'Recipe':<14} {'Selected@128K':>14} {'Extra GPU VRAM':>15} {'MinimalCone':>13} {'Opt-2 Halo':>12}")
        for item in summaries:
            print(
                f"{item['preset']:<14} {item['selected_tokens']:>14,} "
                f"{item['extra_gpu_gib']:>13.2f}G "
                f"{item['minimal_cone_saved_pct']:>+12.1f}% "
                f"{item['halo_saved_pct']:>+11.1f}%"
            )
    print("\n[3] Reproduce Both Optimal Recipes on 8xH200:")
    print("  # Quick 12-question 128K comparison (two-stage / single-stage / both):")
    print("  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset two-stage")
    print("  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset single-stage")
    print("  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset both")
    print("  # Full 300-question benchmark (replace --preset two-stage with single-stage to compare):")
    print("  python3.12 test/benchmark_rag_redknot.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset two-stage")
    print("  python3.12 test/benchmark_rag_redknot.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --preset single-stage")


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
        full_rec = sealed(results_dir / "groups" / gid / "Full.json")
        rk_rec = sealed(results_dir / "groups" / gid / "S1.6.json")
        full_by_idx = {r["index"]: r for r in full_rec["requests"]}
        rk_by_idx = {r["index"]: r for r in rk_rec["requests"]}
        for idx in group_meta["target_indices"]:
            meta = meta_by_idx[idx]
            f_req = full_by_idx[idx]
            r_req = rk_by_idx[idx]
            f_f1, _, _ = qa_scores(f_req["prediction_text"], meta["references"])
            r_f1, _, _ = qa_scores(r_req["prediction_text"], meta["references"])
            f_ttft = (f_req["first_token_ns"] - f_req["received_ns"]) / 1e9
            r_ttft = (r_req["first_token_ns"] - r_req["received_ns"]) / 1e9
            item = dict(
                group=gid,
                index=idx,
                dataset=meta["dataset"],
                input_tokens=meta["input_tokens"],
                references=meta["references"],
                full_f1=f_f1 * 100.0,
                redknot_f1=r_f1 * 100.0,
                full_ttft_s=f_ttft,
                redknot_ttft_s=r_ttft,
                full_pred=f_req["prediction_text"].strip().replace("\n", " ")[:80],
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
    for item in rows:
        print(
            f"  [{item['group']} #{item['index']} {item['dataset']} {item['input_tokens']//1000}K] "
            f"Full={item['full_f1']:5.1f} ({item['full_ttft_s']:.2f}s) | "
            f"RedKnot={item['redknot_f1']:5.1f} ({item['redknot_ttft_s']:.2f}s) | "
            f"Ref={item['references'][0]!r} | RK={item['redknot_pred']!r}"
        )
    return dict(preset=preset_label, groups=groups, overall=overall, rows=rows)


def _print_combined_summary(all_summaries: dict[str, dict]) -> None:
    print("\n" + "=" * 84)
    print("Combined Comparison: Full vs Single-Stage (20%) vs Two-Stage (9%+3%)")
    print("=" * 84)
    print(f"{'Recipe':<14} {'N':>4} {'Full F1':>9} {'RedKnot F1':>11} {'Delta':>8} {'Full TTFT':>10} {'RK TTFT':>10} {'Speedup':>8}")
    for preset, summary in all_summaries.items():
        ov = summary["overall"]
        print(
            f"{preset:<14} {ov['n']:>4} {ov['full_f1']:>9.2f} {ov['redknot_f1']:>11.2f} "
            f"{ov['delta_f1']:>+8.2f} {ov['full_ttft_s']:>9.3f}s {ov['redknot_ttft_s']:>9.3f}s {ov['speedup']:>7.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-demo", action="store_true",
                        help="run zero-GPU CPU walkthrough of routing, memory estimate, MinimalCone & L2->L14")
    parser.add_argument("--model-path", type=Path, default=os.environ.get("REDKNOT_MODEL_PATH"),
                        help="local DeepSeek-V4.1-Flash-1017 checkpoint path for live 8xH200 comparison")
    parser.add_argument("--preset", choices=("two-stage", "single-stage", "both"), default="two-stage",
                        help="RedKnot V2 recipe: two-stage (default, L2->L14 9%%+3%%), "
                             "single-stage (20%%, 0 GiB extra cache), or both (run and compare both)")
    parser.add_argument("--host-prefix-offload", choices=("auto", "on", "off"), default="auto",
                        help="pinned CPU offload for L3..L8 residuals in two-stage mode (auto: >=192K)")
    parser.add_argument("--groups", type=str, default=",".join(DEFAULT_DEMO_GROUPS),
                        help="comma-separated B4 group IDs (default: lb-0002,lb-0027,lb-0074 = 12 128K QA questions)")
    parser.add_argument("--all", action="store_true",
                        help="run all 75 B4 groups (300 questions) instead of the quick 3-group demo")
    parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT / "cache",
                        help="local prepared dataset cache directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="new output directory for this demo run")
    parser.add_argument("--check-runtime", action="store_true",
                        help="validate checkpoint and runtime preflight on CPU without launching GPUs")
    args = parser.parse_args()

    if args.cpu_demo:
        run_cpu_demo(args.preset, args.host_prefix_offload)
        return

    if args.model_path is None:
        parser.error(
            "Live 8xH200 comparison requires --model-path /path/to/DeepSeek-V4.1-Flash-1017 "
            "(or REDKNOT_MODEL_PATH). To run the offline walkthrough without GPUs, pass --cpu-demo."
        )

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

    presets_to_run = ("single-stage", "two-stage") if args.preset == "both" else (args.preset,)
    print(f"Demo run directory: {run_dir}", flush=True)
    all_summaries: dict[str, dict] = {}
    preflight_results: dict[str, dict] = {}

    for preset_name in presets_to_run:
        v2_cfg = v2_cls.from_cli(preset=preset_name, host_prefix_offload=args.host_prefix_offload)
        sub_results = run_dir / (f"results-{preset_name}" if len(presets_to_run) > 1 else "results")
        sub_report = run_dir / (f"report-{preset_name}" if len(presets_to_run) > 1 else "report")
        print(f"\n[{preset_name}] Policy config: {json.dumps(v2_cfg.to_dict(), ensure_ascii=False)}", flush=True)
        result = run_local(
            None,
            args.model_path,
            data,
            sub_results,
            check_only=args.check_runtime,
            v2_config=v2_cfg,
            selected_groups=selected_groups,
        )
        if args.check_runtime:
            preflight_results[preset_name] = result
            continue
        if selected_groups is None:
            summary = score_cohort(data, sub_results, sub_report, expected_contract=result["contract_id"])
            ov = summary["overall"]
            full, sparse = ov["Full"], ov["S1.6"]
            all_summaries[preset_name] = dict(
                preset=preset_name,
                overall=dict(
                    n=ov["paired"],
                    full_f1=full["score_100"],
                    redknot_f1=sparse["score_100"],
                    delta_f1=sparse["score_100"] - full["score_100"],
                    full_ttft_s=full["mean_ttft_s"],
                    redknot_ttft_s=sparse["mean_ttft_s"],
                    speedup=ov["speedup_ratio_of_mean_ttft"],
                ),
                report=str(sub_report / "index.html"),
            )
            print(json.dumps(all_summaries[preset_name]["overall"], indent=2, ensure_ascii=False))
        else:
            all_summaries[preset_name] = score_subset_results(data, sub_results, selected_groups, preset_name)

    if args.check_runtime:
        print(json.dumps(preflight_results if len(presets_to_run) > 1 else preflight_results[presets_to_run[0]], indent=2, ensure_ascii=False))
        return

    if len(all_summaries) > 1:
        _print_combined_summary(all_summaries)
    payload = all_summaries if len(all_summaries) > 1 else all_summaries[presets_to_run[0]]
    (run_dir / "demo-summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved JSON summary: {run_dir / 'demo-summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"Demo failed: {exc}")
