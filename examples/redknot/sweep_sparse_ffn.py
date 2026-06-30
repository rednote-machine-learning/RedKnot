#!/usr/bin/env python3
"""Sparse FFN sweet-spot sweep on HotpotQA (single model load).

Loads Qwen3-32B once, builds N samples, computes the dense baseline and the
offline KV per sample once (both independent of the FFN schedule), then sweeps
several (dense_until, mass_thresh) Sparse FFN configs through the *parallel*
online path. Reports, per config: F1 / EM / cosine vs baseline, RedKnot online
TTFT, wall_speedup, and FFN savings -- so you can read off the speed/accuracy
sweet spot directly.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

# Reuse all helpers from the single-run script.
import run_hotpot_4x4k as H  # noqa: E402
from sglang.srt.layers.attention.redknot import (  # noqa: E402
    HeadClassConfig,
    SparseFFNSchedule,
    offline_prefill_segments,
    run_baseline,
    run_redknot_batched,
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-segments", type=int, default=4)
    ap.add_argument("--tokens-per-segment", type=int, default=4000)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--kernel", default="fa2", choices=["fa2", "fa3"])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"tokenizer {H.MODEL}")
    tok = AutoTokenizer.from_pretrained(H.MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hc = HeadClassConfig.from_json(H.HEAD_CFG)
    log(f"head summary {hc.summary()}")

    log("loading Qwen3-32B ...")
    model = AutoModelForCausalLM.from_pretrained(
        H.MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval()

    samples = H.build_samples(
        tok, args.n_samples, args.n_segments, args.tokens_per_segment, args.seed
    )
    ctx = args.n_segments * args.tokens_per_segment
    log(
        f"built {len(samples)} samples ({args.n_segments}x{args.tokens_per_segment} = {ctx} ctx)"
    )

    # ── Precompute baseline + offline KV once per sample (FFN-independent) ──
    per_sample = []
    for si, s in enumerate(samples, 1):
        qt = H.query_text(s["question"])
        prompt = "\n\n".join(s["docs"]) + qt
        log(f"[prep {si}/{len(samples)}] baseline + offline prefill ...")
        bl_logits, bl_text, _, bl_ttft = run_baseline(
            model,
            tok,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            chunk_size=8192,
            attn_impl="sdpa",
        )
        bl_pred = H.short_ans(bl_text)
        segs = offline_prefill_segments(
            model, tok, s["docs"], chunk_size=4096, model_id=H.MODEL
        )
        per_sample.append(
            dict(
                s=s,
                qt=qt,
                bl_logits=bl_logits,
                bl_pred=bl_pred,
                bl_ttft=bl_ttft,
                segs=segs,
            )
        )

    bl_f1 = sum(H.f1(p["bl_pred"], p["s"]["answer"]) for p in per_sample) / len(
        per_sample
    )
    bl_em = sum(H.em(p["bl_pred"], p["s"]["answer"]) for p in per_sample) / len(
        per_sample
    )
    bl_ttft = sum(p["bl_ttft"] for p in per_sample) / len(per_sample)
    log(f"BASELINE: F1={bl_f1:.3f} EM={bl_em:.3f} TTFT={bl_ttft:.2f}s")

    # ── Sweep grid (None = no Sparse FFN; the rest are schedules) ──
    grid = [
        ("no-sparse-ffn", None),
        ("du20_m0.5", SparseFFNSchedule(dense_until=20, mass_thresh=0.5, recent_n=128)),
        ("du8_m0.5", SparseFFNSchedule(dense_until=8, mass_thresh=0.5, recent_n=128)),
        ("du4_m0.5", SparseFFNSchedule(dense_until=4, mass_thresh=0.5, recent_n=128)),
        ("du8_m0.3", SparseFFNSchedule(dense_until=8, mass_thresh=0.3, recent_n=128)),
        ("du4_m0.3", SparseFFNSchedule(dense_until=4, mass_thresh=0.3, recent_n=64)),
    ]

    results = []
    for name, sched in grid:
        log("=" * 70)
        log(f"CONFIG {name}: {sched}")
        agg = collections.defaultdict(list)
        for si, p in enumerate(per_sample, 1):
            ffn_stats = []
            rc_logits, rc_text, qlen, rc_ttft = run_redknot_batched(
                model,
                tok,
                segments_offline=p["segs"],
                query_text=p["qt"],
                head_cfg=hc,
                max_new_tokens=args.max_new_tokens,
                kernel=args.kernel,
                sparse_ffn_schedule=sched,
                sparse_ffn_stats=ffn_stats if sched else None,
            )
            rc_pred = H.short_ans(rc_text)
            gold = p["s"]["answer"]
            agg["f1"].append(H.f1(rc_pred, gold))
            agg["em"].append(H.em(rc_pred, gold))
            agg["cos"].append(
                torch.nn.functional.cosine_similarity(
                    p["bl_logits"].float(), rc_logits.float(), dim=-1
                ).item()
            )
            agg["ttft"].append(rc_ttft)
            if sched and ffn_stats:
                deep = [x for x in ffn_stats if x.get("mode") == "sparse"]
                if deep:
                    agg["dfrac"].append(
                        sum(x["selected_frac"] for x in deep) / len(deep)
                    )
            torch.cuda.empty_cache()
        mean = lambda k: (sum(agg[k]) / len(agg[k])) if agg[k] else 0.0
        rc_ttft_m = mean("ttft")
        results.append(
            dict(
                name=name,
                f1=mean("f1"),
                em=mean("em"),
                cos=mean("cos"),
                ttft=rc_ttft_m,
                wall=bl_ttft / rc_ttft_m if rc_ttft_m else 0.0,
                dfrac=mean("dfrac"),
            )
        )
        r = results[-1]
        log(
            f"  -> F1={r['f1']:.3f} EM={r['em']:.3f} cos={r['cos']:.4f} "
            f"TTFT={r['ttft']:.2f}s wall={r['wall']:.2f}x deep_frac={r['dfrac']:.2f}"
        )

    # ── Summary table ──
    print("\n" + "=" * 86)
    print(
        f"SPARSE FFN SWEET-SPOT SWEEP | HotpotQA {args.n_segments}x{args.tokens_per_segment}={ctx} | n={len(samples)}"
    )
    print(f"BASELINE  F1={bl_f1:.3f}  EM={bl_em:.3f}  TTFT={bl_ttft:.2f}s")
    print("-" * 86)
    print(
        f"{'config':14s} {'F1':>6s} {'EM':>6s} {'cosine':>8s} {'TTFT(s)':>9s} {'wall':>7s} {'deepFrac':>9s}"
    )
    print("-" * 86)
    for r in results:
        print(
            f"{r['name']:14s} {r['f1']:6.3f} {r['em']:6.3f} {r['cos']:8.4f} "
            f"{r['ttft']:9.2f} {r['wall']:6.2f}x {r['dfrac']:9.2f}"
        )
    print("=" * 86)


if __name__ == "__main__":
    main()
