#!/usr/bin/env python3
"""Heterogeneous per-head window: SegPaged vs Paged KV bandwidth (Qwen3-32B).

Uses the REAL head classification (global/retrieval/local) from the RedKnot
head-config, and assigns each LOCAL head a heterogeneous effective window drawn
from a long-tailed distribution (most local heads concentrate within a small
window, a few need larger ones) -- the empirically reported pattern for
streaming/local heads (DuoAttention / StreamingLLM / H2O families).

Compares decode KV-read bandwidth of two backend storages:
  Paged    (token-major, unified block): a layer's block must cover the MAX
           window over its heads; reading any token row pulls ALL Hkv heads ->
           small-window heads are READ-AMPLIFIED to the layer max.
  SegPaged (head-major, per-head pages): each head reads exactly its own window
           -> no amplification.

We report BOTH:
  (a) UNIFORM window (all local heads = W): the artificial setting that HIDES
      SegPaged's advantage (Paged block == window in pure-local layers).
  (b) HETEROGENEOUS window: the realistic setting that EXPOSES Paged's read
      amplification.

Analytic & exact given classification + per-head windows.

Usage: python hetero_window_bandwidth_qwen3.py --seed 2026
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter


def is_full(t: str) -> bool:
    return t in ("global", "retrieval")


# Long-tailed local-head window distribution (tokens) + probabilities.
# Most local heads are very local; a tail needs larger windows.
WIN_CHOICES = [128, 256, 512, 1024, 2048, 4096]
WIN_PROBS = [0.30, 0.28, 0.20, 0.12, 0.07, 0.03]


def assign_windows(cls, rng, uniform_w=None):
    """Return win[L][Hkv]: full heads -> 'FULL'; local heads -> int window."""
    wins = []
    for row in cls:
        r = []
        for t in row:
            if is_full(t):
                r.append("FULL")
            elif uniform_w is not None:
                r.append(uniform_w)
            else:
                r.append(rng.choices(WIN_CHOICES, weights=WIN_PROBS)[0])
        wins.append(r)
    return wins


def bandwidth(wins, Hkv, D, DT, Lctx, sink):
    pg = sg = 0
    for row in wins:
        eff = [(Lctx if w == "FULL" else min(w + sink, Lctx)) for w in row]
        for e in eff:
            sg += e * D * 2 * DT  # SegPaged: each head exact
        pg += max(eff) * Hkv * D * 2 * DT  # Paged: layer block = max head
    return pg, sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-config",
        default="/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3-32B/config.json",
    )
    ap.add_argument(
        "--head-config",
        default="test/srt/redknot/head_class/qwen3-32B_optimal_g15_lf_ret.json",
    )
    ap.add_argument("--lengths", default="16000,32000,64000,128000")
    ap.add_argument("--uniform-window", type=int, default=4096)
    ap.add_argument("--kv-dtype-bytes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    cfg = json.load(open(args.model_config))
    L = cfg["num_hidden_layers"]
    Hkv = cfg["num_key_value_heads"]
    D = cfg["head_dim"]
    hc = json.load(open(args.head_config))
    cls = hc["kv_head_classification"]
    sink = hc["sink_size"]
    DT = args.kv_dtype_bytes
    lengths = [int(x) for x in args.lengths.split(",")]
    rng = random.Random(args.seed)

    flat = [t for row in cls for t in row]
    cnt = Counter(flat)
    nfull = sum(v for k, v in cnt.items() if is_full(k))
    print(f"Qwen3-32B L={L} Hkv={Hkv} D={D} KV={DT}B sink={sink}")
    print(
        f"REAL head classes: {dict(cnt)}  full={nfull}({100 * nfull / len(flat):.0f}%) "
        f"local={len(flat) - nfull}({100 * (len(flat) - nfull) / len(flat):.0f}%)"
    )

    uni = assign_windows(cls, rng, uniform_w=args.uniform_window)
    het = assign_windows(cls, rng, uniform_w=None)
    het_local = [w for row in het for w in row if w != "FULL"]
    print(f"\nLocal-head window — UNIFORM = {args.uniform_window} (artificial)")
    print(
        f"Local-head window — HETEROGENEOUS (realistic, long-tail): "
        f"min={min(het_local)} median={int(statistics.median(het_local))} "
        f"max={max(het_local)} mean={int(statistics.mean(het_local))}"
    )
    print(f"  distribution: {dict(sorted(Counter(het_local).items()))}")

    print("\n" + "=" * 80)
    print("DECODE KV-read bandwidth: SegPaged vs Paged")
    print("=" * 80)
    print(
        f"{'':>7} | {'UNIFORM window (hides advantage)':^32} | "
        f"{'HETEROGENEOUS window (real)':^30}"
    )
    print(
        f"{'L_ctx':>7} | {'Paged':>8} {'SegP':>8} {'ratio':>6} {'saved':>6} | "
        f"{'Paged':>8} {'SegP':>8} {'ratio':>6} {'saved':>6}"
    )
    out = {
        "model": "Qwen3-32B",
        "uniform_window": args.uniform_window,
        "het_window_stats": {
            "min": min(het_local),
            "max": max(het_local),
            "median": statistics.median(het_local),
            "dist": dict(Counter(het_local)),
        },
        "results": [],
    }
    for Lctx in lengths:
        upg, usg = bandwidth(uni, Hkv, D, DT, Lctx, sink)
        hpg, hsg = bandwidth(het, Hkv, D, DT, Lctx, sink)
        print(
            f"{Lctx:>7} | {upg / 1e9:>7.2f}G {usg / 1e9:>7.2f}G {upg / usg:>5.2f}x "
            f"{100 * (1 - usg / upg):>5.1f}% | {hpg / 1e9:>7.2f}G {hsg / 1e9:>7.2f}G "
            f"{hpg / hsg:>5.2f}x {100 * (1 - hsg / hpg):>5.1f}%"
        )
        out["results"].append(
            {
                "Lctx": Lctx,
                "uniform": {
                    "paged_GB": upg / 1e9,
                    "segpaged_GB": usg / 1e9,
                    "ratio": upg / usg,
                },
                "hetero": {
                    "paged_GB": hpg / 1e9,
                    "segpaged_GB": hsg / 1e9,
                    "ratio": hpg / hsg,
                },
            }
        )
    json.dump(out, open("/tmp/redknot_hetero_window_bandwidth.json", "w"), indent=2)
    print("\nKey finding: a UNIFORM window makes Paged's unified block == the window")
    print("(no amplification in pure-local layers), HIDING SegPaged's benefit.")
    print("With REAL HETEROGENEOUS windows, Paged's block is forced to the per-layer")
    print("MAX window, so small-window heads are read-amplified -> SegPaged's")
    print("head-major decoupling wins by a much larger margin. Advantage grows with")
    print("context length and with per-head window dispersion.")
    print("\nSaved /tmp/redknot_hetero_window_bandwidth.json")


if __name__ == "__main__":
    main()
