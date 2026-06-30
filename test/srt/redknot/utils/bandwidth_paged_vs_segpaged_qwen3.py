#!/usr/bin/env python3
"""SegPaged vs Paged attention: PREFILL & DECODE KV bandwidth (Qwen3-32B).

Same RedKnot operation flow (global/retrieval heads attend full context; local
heads attend sink+window). Only the BACKEND KV STORAGE differs:

  Paged    (token-major, shared pages): KV is [token][Hkv][D]; pages are shared
           across all KV heads. Reading any token row pulls ALL Hkv heads. So in
           a layer that mixes full+local heads, the local heads cannot realize
           their per-head sparsity -- they are read over the full range too.
  SegPaged (head-major, per-head pages): each (layer, kv-head) owns its pages;
           local heads physically store/read only sink+window.

We report PREFILL and DECODE bandwidth SEPARATELY because their KV access
patterns differ:

  DECODE  (per step): one query reads the whole KV range it attends to.
          bytes = sum over (layer, head) of (attended_tokens) x D x 2 x dtype.
          Paged: a layer's attended range = max over its heads (full -> Lctx).

  PREFILL (whole prompt, segmented as RedKnot does): for each segment s with
          Lb queries arriving on top of `prefix` already-written tokens, each
          query reads its causal KV range. Aggregate KV-READ work per (layer,
          head):
            global: sum over queries of (prefix + i)  ~ Lb*prefix + Lb(Lb+1)/2
            local : sum over queries of min(window, prefix+i) (sliding window)
          Paged forces local heads in mixed layers onto the global range.
          PREFILL also WRITES every token's KV once (same for both backends);
          we report read-bandwidth (the part where the two differ) and note the
          identical write cost.

All numbers are analytic & exact given head classification + model dims; a real
two-pass run of `run_redknot_offlinekv(use_segpaged=False/True)` yields the same
KV byte counts (only the attention backend storage differs).

Usage:
  python bandwidth_paged_vs_segpaged_qwen3.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def is_full(t: str) -> bool:
    return t in ("global", "retrieval")


def decode_bandwidth(cls, L, Hkv, D, DT, Lctx, win):
    paged = seg = 0
    for li in range(L):
        heads = cls[li]
        cover = Lctx if any(is_full(t) for t in heads) else min(win, Lctx)
        paged += cover * Hkv * D * 2 * DT
        for t in heads:
            tok = Lctx if is_full(t) else min(win, Lctx)
            seg += tok * D * 2 * DT
    return paged, seg


def prefill_read_bandwidth(cls, L, Hkv, D, DT, seg_lens, window, sink):
    """Aggregate KV-read bytes over the whole segmented prefill."""
    paged = seg = 0
    for li in range(L):
        heads = cls[li]
        layer_has_full = any(is_full(t) for t in heads)
        prefix = 0

        # per-head accumulators reset per layer; iterate segments
        # global read work per head:
        def gwork(prefix, Lb):
            return Lb * prefix + Lb * (Lb + 1) // 2

        def lwork(prefix, Lb):
            # sum_{i=0..Lb-1} min(window+sink, prefix+i)
            tot = 0
            for i in range(Lb):
                tot += min(window + sink, prefix + i + 1)
            return tot

        # accumulate over segments
        per_head_global = []
        per_head_local = []
        pfx = 0
        gw = lw = 0
        for Lb in seg_lens:
            gw += gwork(pfx, Lb)
            lw += lwork(pfx, Lb)
            pfx += Lb
        for t in heads:
            full = is_full(t)
            seg += (gw if full else lw) * D * 2 * DT
            # paged: local heads in a mixed layer pay the global read work
            paged += (gw if (full or layer_has_full) else lw) * D * 2 * DT
    return paged, seg


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
    # (n_segments, tokens_per_segment) like RedKnot benchmark
    ap.add_argument("--lengths", default="16K:4x4000,24K:6x4000,32K:8x4000,64K:8x8000")
    ap.add_argument("--kv-dtype-bytes", type=int, default=2)
    args = ap.parse_args()

    cfg = json.load(open(args.model_config))
    L = cfg["num_hidden_layers"]
    Hkv = cfg["num_key_value_heads"]
    D = cfg["head_dim"]
    hc = json.load(open(args.head_config))
    cls = hc["kv_head_classification"]
    window = hc["window"]
    sink = hc["sink_size"]
    DT = args.kv_dtype_bytes

    flat = [t for row in cls for t in row]
    cnt = Counter(flat)
    nfull = sum(v for k, v in cnt.items() if is_full(k))
    nloc = len(flat) - nfull
    pure_local = sum(1 for row in cls if not any(is_full(t) for t in row))
    print(
        f"Model Qwen3-32B  L={L} Hkv={Hkv} D={D} KV={DT}B | window={window} sink={sink}"
    )
    print(
        f"kv-head: {dict(cnt)}  full={nfull}({100 * nfull / len(flat):.0f}%) "
        f"local={nloc}({100 * nloc / len(flat):.0f}%)  pure-local layers={pure_local}/{L}"
    )

    specs = []
    for tok in args.lengths.split(","):
        label, rest = tok.split(":")
        n, per = rest.split("x")
        specs.append((label, int(n), int(per)))

    out = {
        "model": "Qwen3-32B",
        "head_config": args.head_config,
        "decode": [],
        "prefill": [],
    }

    print("\n" + "=" * 78)
    print("DECODE  KV-read bandwidth per step")
    print("=" * 78)
    print(f"{'len':>6} | {'Paged GB':>9} {'SegPaged GB':>11} {'ratio':>6} {'saved':>7}")
    for label, n, per in specs:
        Lctx = n * per
        win = min(window, Lctx) + sink
        pg, sg = decode_bandwidth(cls, L, Hkv, D, DT, Lctx, win)
        print(
            f"{label:>6} | {pg / 1e9:>9.3f} {sg / 1e9:>11.3f} {pg / sg:>5.2f}x {100 * (1 - sg / pg):>6.1f}%"
        )
        out["decode"].append(
            {
                "len": label,
                "Lctx": Lctx,
                "paged_GB": pg / 1e9,
                "segpaged_GB": sg / 1e9,
                "ratio": pg / sg,
                "saved_pct": 100 * (1 - sg / pg),
            }
        )

    print("\n" + "=" * 78)
    print("PREFILL  aggregate KV-read bandwidth (whole segmented prompt)")
    print("=" * 78)
    print(f"{'len':>6} | {'Paged GB':>9} {'SegPaged GB':>11} {'ratio':>6} {'saved':>7}")
    for label, n, per in specs:
        seg_lens = [per] * n
        pg, sg = prefill_read_bandwidth(cls, L, Hkv, D, DT, seg_lens, window, sink)
        print(
            f"{label:>6} | {pg / 1e9:>9.3f} {sg / 1e9:>11.3f} {pg / sg:>5.2f}x {100 * (1 - sg / pg):>6.1f}%"
        )
        out["prefill"].append(
            {
                "len": label,
                "paged_GB": pg / 1e9,
                "segpaged_GB": sg / 1e9,
                "ratio": pg / sg,
                "saved_pct": 100 * (1 - sg / pg),
            }
        )

    print("\nNotes:")
    print(
        "- PREFILL KV-WRITE bytes are identical for both backends (all tokens written"
    )
    print("  once); only KV-READ differs (shown above) -> that is where SegPaged wins.")
    print("- Both phases: SegPaged advantage grows with context length and with the")
    print("  fraction of mixed full+local layers (token-major Paged forces local heads")
    print("  in those layers to be read over the full range).")
    json.dump(out, open("/tmp/redknot_bandwidth_paged_vs_segpaged.json", "w"), indent=2)
    print("\nSaved /tmp/redknot_bandwidth_paged_vs_segpaged.json")


if __name__ == "__main__":
    main()
