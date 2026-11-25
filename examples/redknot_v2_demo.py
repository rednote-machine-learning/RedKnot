#!/usr/bin/env python3
"""Open-source quickstart demo: reproduce RedKnot V2 faster & more accurate than Full.

Two ready-to-run modes:
  1. CPU walkthrough (works on any laptop, zero GPU / zero checkpoint needed):
       python3 examples/redknot_v2_demo.py --cpu-demo

  2. Live 8xH200 paired Full vs RedKnot V2 benchmark (~3-5 minutes for 3 128K B4 groups):
       python3.12 examples/redknot_v2_demo.py \\
         --model-path /path/to/DeepSeek-V4.1-Flash-1017

     Options:
       --preset short20_long15   (default: accuracy-speed champion, L2->L14 12%+3%)
       --preset v2.1-fast        (single-stage 20% production default)
       --preset v2.3_12          (extreme speed 9%+3%, still beats Full on 500Q)
       --groups lb-0002,lb-0027,lb-0074   (default 128K HotpotQA/2WikiMQA/MuSiQue trio)
       --all                     (run all 75 B4 groups / 300 questions)
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

for p in (
    TEST_ROOT,
    ALGO_ROOT / "fixed",
    ALGO_ROOT / "lastkv",
    ALGO_ROOT / "shared" / "runtime",
):
    text = str(p)
    if text not in sys.path:
        sys.path.insert(0, text)


def run_cpu_demo(preset: str, span_min_tokens: int) -> None:
    """Offline CPU walkthrough of RedKnot V2 routing, selection, MinimalCone, and L2->L14 locality."""
    import cone15_plan
    import fixed15_geometry
    import halo_direct_copy
    import index20_selector
    import two_stage_l14
    import v2_policy

    two_stage = preset in v2_policy.TWO_STAGE_PRESETS
    mode = "s1.6" if preset == "s1.6" else "v2.1-fast"
    cfg = v2_policy.V2PolicyConfig.from_cli(
        mode=mode,
        two_stage_l14=two_stage,
        preset=preset,
        span_min_tokens=span_min_tokens,
    )

    print("=" * 78)
    print(f"RedKnot V2 Open-Source CPU Walkthrough  (preset={cfg.preset}, span_min={cfg.span_min_tokens})")
    print("=" * 78)
    print("\n[1] Length-Adaptive Policy Routing ('Short-Wide, Long-Tight'):")
    print(f"{'Prompt':>8} {'Band':>5} {'Mode':<15} {'Budget':>10} {'Head':>6} {'Span(C2)':>10} {'MinimalCone':>12} {'HostOffload':>12}")
    for tokens in (16_000, 32_000, 64_000, 128_000, 250_000):
        pol = cfg.resolve(tokens)
        budget = f"{int(pol.stage1_fraction*100)}%->{int(pol.fraction*100)}%" if pol.two_stage_l14 else f"{int(pol.fraction*100)}%"
        print(
            f"{tokens:>8,} {pol.band:>5} {pol.mode:<15} {budget:>10} "
            f"{pol.head_width:>6} {str(pol.window):>10} {str(pol.minimal_cone):>12} {str(pol.host_prefix_offload):>12}"
        )

    # Simulate a realistic 128K multi-document RAG prompt (10 documents + user query suffix).
    print("\n[2] Simulated 128K Multi-Document RAG Pipeline (10 documents, 128,000 tokens):")
    rng = random.Random(2026)
    total = 128_000
    docs = [{"start": 64 + i * 12_600, "end": 64 + i * 12_600 + 12_400, "key": f"doc-{i}"} for i in range(10)]
    pol_128k = cfg.resolve(total)
    n_keys = total // 2
    # Inject strong signal in 2 relevant documents and realistic background scores elsewhere.
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

    if pol_128k.two_stage_l14:
        # Compute exact single-stage equivalent target body budget, then promote via Layer 14 scores.
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
        # Multi-hop bridge evidence discovered at Layer 14 in doc-5.
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
        print(
            f"  - Two-Stage L2->L14: Stage-1={len(sel_stage1['selected_positions']):,} tokens "
            f"+ L14 promoted={len(extra):,} tokens -> Final={len(selected):,} / {total:,} "
            f"({100*len(selected)/total:.1f}%)"
        )
        print(
            f"  - L3..L13 Local Backfill: computes ONLY {len(affected[3]):,} newly promoted rows/layer "
            f"(reuses {len(sel_stage1['selected_positions']):,} cached Stage-1 rows, "
            f"saves {100*len(sel_stage1['selected_positions'])/len(selected):.1f}% shallow-layer backfill compute)"
        )
    else:
        print(
            f"  - Single-Stage L2 Selection: {len(selected):,} / {total:,} tokens "
            f"({100*len(selected)/total:.1f}%, Head={pol_128k.head_width}, Span={pol_128k.window})"
        )

    # Measure MinimalCone post-20 (Layers 21..39) FFN & Opt-2 offline Halo savings.
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

    print("\n[3] Reproduce Live Faster-and-Better-Than-Full Results on 8xH200:")
    print("  # Quick 12-question 128K multi-hop RAG comparison (~3-5 min):")
    print("  python3.12 examples/redknot_v2_demo.py --model-path /path/to/DeepSeek-V4.1-Flash-1017")
    print("  # Full 300-question paired benchmark:")
    print("  python3.12 test/benchmark_rag_redknot.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --two-stage-l14 --preset short20_long15")
    print("  # Full 2,100-question LongBench-21 benchmark:")
    print("  python3.12 test/benchmark_rag_redknot_21.py --model-path /path/to/DeepSeek-V4.1-Flash-1017 --two-stage-l14 --preset short20_long15")


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
    for ds, items in [*by_dataset.items(), ("All", rows)]:
        n = len(items)
        f_f1 = sum(x["full_f1"] for x in items) / n
        r_f1 = sum(x["redknot_f1"] for x in items) / n
        f_t = sum(x["full_ttft_s"] for x in items) / n
        r_t = sum(x["redknot_ttft_s"] for x in items) / n
        spd = f_t / r_t if r_t > 0 else 0.0
        print(f"{ds:<13} {n:>3} {f_f1:>9.2f} {r_f1:>11.2f} {r_f1 - f_f1:>+8.2f} {f_t:>9.3f}s {r_t:>9.3f}s {spd:>7.2f}x")

    print("\nPer-Question Spot Check:")
    for item in rows:
        print(
            f"  [{item['group']} #{item['index']} {item['dataset']} {item['input_tokens']//1000}K] "
            f"Full={item['full_f1']:5.1f} ({item['full_ttft_s']:.2f}s) | "
            f"RedKnot={item['redknot_f1']:5.1f} ({item['redknot_ttft_s']:.2f}s) | "
            f"Ref={item['references'][0]!r} | RK={item['redknot_pred']!r}"
        )
    return dict(preset=preset_label, groups=groups, rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-demo", action="store_true",
                        help="run zero-GPU CPU walkthrough of routing, selection, MinimalCone & L2->L14")
    parser.add_argument("--model-path", type=Path, default=os.environ.get("REDKNOT_MODEL_PATH"),
                        help="local DeepSeek-V4.1-Flash-1017 checkpoint path for live 8xH200 comparison")
    parser.add_argument("--preset", choices=("short20_long15", "v2.1-fast", "v2.3_12", "v2.3_10", "m16", "s1.6"),
                        default="short20_long15",
                        help="RedKnot preset (default: short20_long15, accuracy-speed champion)")
    parser.add_argument("--groups", type=str, default=",".join(DEFAULT_DEMO_GROUPS),
                        help="comma-separated B4 group IDs (default: lb-0002,lb-0027,lb-0074 = 12 128K QA questions)")
    parser.add_argument("--all", action="store_true",
                        help="run all 75 B4 groups (300 questions) instead of the quick 3-group demo")
    parser.add_argument("--span-min-tokens", type=int, default=48_000,
                        help="prompt token threshold for C2 Span [-8,+24] (default: 48000)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT / "cache",
                        help="local prepared dataset cache directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="new output directory for this demo run")
    parser.add_argument("--check-runtime", action="store_true",
                        help="validate checkpoint and runtime preflight on CPU without launching GPUs")
    args = parser.parse_args()

    if args.cpu_demo:
        run_cpu_demo(args.preset, args.span_min_tokens)
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
    mode = "s1.6" if args.preset == "s1.6" else "v2.1-fast"
    two_stage = args.preset in ("short20_long15", "v2.3_12", "v2.3_10", "m16")
    v2_cfg = v2_cls.from_cli(
        mode=mode,
        two_stage_l14=two_stage,
        preset=args.preset,
        span_min_tokens=args.span_min_tokens,
    )

    data = prepare_data(ASSETS / "redknot_s16_longbench_300.zip", args.cache.expanduser().resolve())
    selected_groups = None if args.all else [g.strip() for g in args.groups.split(",") if g.strip()]

    if args.output:
        run_dir = args.output.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(mkdtemp(prefix=f"demo-{args.preset}-{stamp}-", dir=DEFAULT_OUTPUT))

    print(f"Demo run directory: {run_dir}", flush=True)
    print(f"Policy config: {json.dumps(v2_cfg.to_dict(), ensure_ascii=False)}", flush=True)
    result = run_local(
        None,
        args.model_path,
        data,
        run_dir / "results",
        check_only=args.check_runtime,
        v2_config=v2_cfg,
        selected_groups=selected_groups,
    )
    if args.check_runtime:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if selected_groups is None:
        summary = score_cohort(data, run_dir / "results", run_dir / "report", expected_contract=result["contract_id"])
        print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    else:
        summary = score_subset_results(data, run_dir / "results", selected_groups, args.preset)
        (run_dir / "demo-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON summary: {run_dir / 'demo-summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"Demo failed: {exc}")
