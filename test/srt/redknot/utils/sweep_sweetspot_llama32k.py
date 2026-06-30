#!/usr/bin/env python3
"""Sweet-spot sweep: Llama-3.3-70B @ 32K, RedKnot vs full recompute (dense).

Scans a 3x3 grid of
  * global/local head ratio  (frac_global in {0.05, 0.10, 0.15})
  * Sparse-FFN aggressiveness (mid/deep mass in {(0.2,0.05), (0.3,0.1), (0.5,0.2)})

on a fixed RANDOM sample of N (default 5) triviaqa contexts padded to 32K.
The model is loaded ONCE; the dense baseline (full recompute) is run ONCE
(it is independent of the RedKnot knobs); then each of the 9 RedKnot configs is
evaluated. We report, per config:
  * F1 / EM vs gold (quality)
  * RedKnot TTFT and speedup vs the dense baseline
  * analytic prefill FLOPs saving (compute axis)

Run (8-GPU bf16, matching the validated benchmark path):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    REDKNOT_DTYPE=bf16 REDKNOT_DEVICE_MAP=auto \
    REDKNOT_CUSTOM_FWD=0 REDKNOT_COMPILE=0 \
    REDKNOT_N_SAMPLES=5 REDKNOT_SEED=0 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python test/srt/redknot/sweep_sweetspot_llama32k.py
"""

from __future__ import annotations

import gc
import os
import random
import sys
import time
from pathlib import Path

import torch

# Reuse all the validated helpers from the Llama RAG benchmark.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[3] / "python"))

import benchmark_RedKnot_Llama_RAG as B  # noqa: E402

from sglang.srt.layers.attention.redknot import (  # noqa: E402
    HeadClassConfig,
    SparseFFNSchedule,
    offline_prefill_segments,
    run_redknot_offlinekv,
)

N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "5"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
TARGET_TOKENS = int(os.environ.get("REDKNOT_TARGET_TOKENS", "32000"))
CHUNK = B.CHUNK_TOKENS
MAX_NEW = B.MAX_NEW_TOKENS
WINDOW_FIXED = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))

# ── Grid ──
FRAC_GLOBALS = [0.05, 0.10, 0.15]
FFN_LEVELS = [
    (
        "aggressive",
        dict(
            dense_until=20,
            mass_thresh=0.2,
            deep_layer_start=60,
            mass_thresh_deep=0.05,
            recent_n=512,
        ),
    ),
    (
        "medium",
        dict(
            dense_until=20,
            mass_thresh=0.3,
            deep_layer_start=60,
            mass_thresh_deep=0.1,
            recent_n=512,
        ),
    ),
    (
        "conservative",
        dict(
            dense_until=20,
            mass_thresh=0.5,
            deep_layer_start=60,
            mass_thresh_deep=0.2,
            recent_n=512,
        ),
    ),
]


def _set_frac_global(hc: HeadClassConfig, frac: float) -> float:
    """Rewrite head_class so ~frac of all (layer,kv_head) heads are global.

    Deterministic: pick the global heads by a stable hash of (layer,head) so
    the same frac always yields the same assignment. Local heads keep their
    window; newly-global heads get window=-1.
    """
    L, H = hc.num_layers, hc.num_kv_heads
    total = L * H
    n_global = max(1, round(frac * total))
    # Stable ordering by a pseudo-random but seeded key.
    rng = random.Random(1234)
    coords = [(li, h) for li in range(L) for h in range(H)]
    rng.shuffle(coords)
    global_set = set(coords[:n_global])
    n_set = 0
    for li in range(L):
        for h in range(H):
            if (li, h) in global_set:
                hc.head_class[li][h] = "global"
                hc.head_max_distance[li][h] = -1
                n_set += 1
            else:
                hc.head_class[li][h] = "local"
                if hc.head_max_distance[li][h] <= 0:
                    hc.head_max_distance[li][h] = WINDOW_FIXED
    hc._cached_tensors.clear()
    return n_set / total


def _build_hc(frac_global: float) -> HeadClassConfig:
    hc = HeadClassConfig.from_json(B.HEAD_CFG_JSON)
    hc.merge_retrieval_to_global()
    realized = _set_frac_global(hc, frac_global)
    hc.set_local_window(WINDOW_FIXED)
    return hc, realized


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    W = 100
    print("=" * W)
    print(" SWEET-SPOT SWEEP: Llama-3.3-70B @ 32K — RedKnot vs full recompute")
    print(f" grid: frac_global={FRAC_GLOBALS} x FFN={[n for n, _ in FFN_LEVELS]}")
    print(
        f" samples={N_SAMPLES} (seed={SEED}), window={WINDOW_FIXED}, max_new={MAX_NEW}"
    )
    print("=" * W)

    tok = AutoTokenizer.from_pretrained(B.MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Random sample selection (32K padded triviaqa) ──
    all_samples = B._load_longbench_padded(
        "triviaqa", tok, n_samples=200, target_tokens=TARGET_TOKENS
    )
    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    samples = all_samples[:N_SAMPLES]
    print(f" loaded {len(samples)} random 32K samples")

    dtype_mode = os.environ.get("REDKNOT_DTYPE", "bf16").lower()
    device_map = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
    print(f" loading model (dtype={dtype_mode}, device_map={device_map})...")
    model = AutoModelForCausalLM.from_pretrained(
        B.MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    d = B._model_dims(model.config)

    # ── 1) Dense baseline (full recompute) — run ONCE ──
    print("\n[dense baseline / full recompute] running once over the sample set...")
    base_rows = []
    ctx_lens = []
    for si, s in enumerate(samples):
        qt = B._query_text(s["question"])
        full_text = "\n\n".join(s["docs"])
        tb, ttft_b, dec_b, n_ctx = B.standard_prefill(
            model, tok, full_text, qt, MAX_NEW
        )
        ans_b = B._short_ans(tb)
        base_rows.append(
            {
                "ans": ans_b,
                "golds": s["golds"],
                "ttft": ttft_b,
                "f1": B.f1_max(ans_b, s["golds"]),
                "em": B.em_max(ans_b, s["golds"]),
            }
        )
        ctx_lens.append(n_ctx)
        gc.collect()
        torch.cuda.empty_cache()
    base_f1 = sum(r["f1"] for r in base_rows) / len(base_rows)
    base_em = sum(r["em"] for r in base_rows) / len(base_rows)
    base_ttft = sum(r["ttft"] for r in base_rows) / len(base_rows)
    T = int(sum(ctx_lens) / len(ctx_lens))
    print(
        f"  baseline: F1={base_f1:.3f} EM={base_em:.3f} TTFT={base_ttft:.2f}s (avg ctx={T:,})"
    )

    # ── 2) Sweep RedKnot configs ──
    results = []
    for frac in FRAC_GLOBALS:
        hc, realized = _build_hc(frac)
        summ = hc.summary()
        fg = summ.get("global", 0) / summ["total"]
        for ffn_name, ffn_kw in FFN_LEVELS:
            sched = SparseFFNSchedule(**ffn_kw)
            rk_rows = []
            sel_deep_seen = []
            warm = True
            for s in samples:
                qt = B._query_text(s["question"])
                segs = offline_prefill_segments(
                    model,
                    tok,
                    s["docs"],
                    chunk_size=max(4096, CHUNK + 96),
                    model_id=B.MODEL_PATH,
                )
                if warm:
                    run_redknot_offlinekv(
                        model,
                        tok,
                        segments_offline=segs,
                        query_text=qt,
                        head_cfg=hc,
                        max_new_tokens=3,
                        kernel="fa2",
                        sparse_ffn_schedule=sched,
                        use_compile=False,
                    )
                    warm = False
                stats = []
                t0 = time.perf_counter()
                _, tc, _, ttft_c = run_redknot_offlinekv(
                    model,
                    tok,
                    segments_offline=segs,
                    query_text=qt,
                    head_cfg=hc,
                    max_new_tokens=MAX_NEW,
                    kernel="fa2",
                    sparse_ffn_schedule=sched,
                    sparse_ffn_stats=stats,
                    use_compile=False,
                )
                torch.cuda.synchronize()
                ans_c = B._short_ans(tc)
                sp = [x for x in stats if x.get("mode") == "sparse"]
                if sp:
                    sel_deep_seen.append(sum(x["selected_frac"] for x in sp) / len(sp))
                rk_rows.append(
                    {
                        "f1": B.f1_max(ans_c, s["golds"]),
                        "em": B.em_max(ans_c, s["golds"]),
                        "ttft": ttft_c,
                    }
                )
                del segs
                gc.collect()
                torch.cuda.empty_cache()

            rk_f1 = sum(r["f1"] for r in rk_rows) / len(rk_rows)
            rk_em = sum(r["em"] for r in rk_rows) / len(rk_rows)
            rk_ttft = sum(r["ttft"] for r in rk_rows) / len(rk_rows)
            sel_deep = (
                (sum(sel_deep_seen) / len(sel_deep_seen)) if sel_deep_seen else 1.0
            )
            fl = B.compute_flops(
                d, T, fg, sel_deep, ffn_kw["dense_until"], WINDOW_FIXED
            )
            save = 1.0 - fl["total"][1] / fl["total"][0]
            results.append(
                {
                    "frac": fg,
                    "ffn": ffn_name,
                    "f1": rk_f1,
                    "em": rk_em,
                    "ttft": rk_ttft,
                    "speedup": base_ttft / rk_ttft,
                    "flops_save": save,
                }
            )
            print(
                f"  [frac_g={fg:.3f} ffn={ffn_name:12}] "
                f"F1={rk_f1:.3f} EM={rk_em:.3f} TTFT={rk_ttft:.2f}s "
                f"speedup={base_ttft / rk_ttft:.2f}x save={save * 100:.1f}%"
            )

    # ── 3) Summary table + sweet spot ──
    print("\n" + "=" * W)
    print(
        " SUMMARY  (baseline = full recompute: "
        f"F1={base_f1:.3f} EM={base_em:.3f} TTFT={base_ttft:.2f}s)"
    )
    print("=" * W)
    print(
        f" {'frac_g':>7} {'ffn':>13} {'F1':>6} {'EM':>6} {'TTFT':>7} "
        f"{'speedup':>8} {'FLOPs_save':>11} {'F1_drop':>8}"
    )
    for r in results:
        print(
            f" {r['frac']:7.3f} {r['ffn']:>13} {r['f1']:6.3f} {r['em']:6.3f} "
            f"{r['ttft']:6.2f}s {r['speedup']:7.2f}x {r['flops_save'] * 100:10.1f}% "
            f"{(base_f1 - r['f1']):+8.3f}"
        )

    # Sweet spot: best speedup among configs whose F1 is within 5% of baseline.
    tol = 0.05
    ok = [r for r in results if r["f1"] >= base_f1 - tol]
    pool = ok if ok else results
    best = max(pool, key=lambda r: r["speedup"])
    print("-" * W)
    print(
        f" SWEET SPOT (F1 within {tol:.0%} of baseline, max speedup): "
        f"frac_g={best['frac']:.3f}, ffn={best['ffn']} -> "
        f"F1={best['f1']:.3f} speedup={best['speedup']:.2f}x "
        f"FLOPs_save={best['flops_save'] * 100:.1f}%"
    )
    print("=" * W)


if __name__ == "__main__":
    main()
