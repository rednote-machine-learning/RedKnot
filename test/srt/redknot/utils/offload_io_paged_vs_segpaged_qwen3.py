#!/usr/bin/env python3
"""SegPaged vs Paged for OFFLINE KV stored on CPU/SSD (Qwen3-32B).

Scenario: the prefix KV is generated offline and stored on CPU RAM or SSD
(NOT in GPU HBM). At inference each layer's KV must be read back over a slow
link (PCIe / SSD). Different head classes have different read frequency:
  - global / retrieval heads: read the FULL context every decode step (high freq)
  - local heads: read only sink+window (low freq / small range)

Two on-disk layouts:
  Paged (token-major, unified):  bytes laid out [token][Hkv][D]. The smallest
    addressable read is a TOKEN ROW = all Hkv heads of that token (they are
    physically contiguous). So to serve ANY full head in a layer you must read
    that token's whole row -> you drag in all Hkv heads. local heads get
    "read-amplified" to the union range of the layer.
  SegPaged (head-major, decoupled): bytes laid out per (layer, head). The
    smallest read is a single head's tokens. You read EXACTLY the heads/tokens
    you need; local heads read only their window.

We report, per decode step and per request:
  - IO bytes read from CPU/SSD (Paged vs SegPaged)
  - latency under a slow link bandwidth (PCIe4 ~25 GB/s, SSD ~3-7 GB/s)
The amplification matters here because the link is 100-1000x slower than HBM.

Usage: python offload_io_paged_vs_segpaged_qwen3.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def is_full(t: str) -> bool:
    return t in ("global", "retrieval")


def io_per_decode_step(cls, L, Hkv, D, DT, Lctx, window, sink):
    """Bytes read back from CPU/SSD for one decode step."""
    win = min(window, Lctx) + sink
    paged = seg = 0
    for li in range(L):
        heads = cls[li]
        # SegPaged: read exactly each head's range
        for t in heads:
            seg += (Lctx if is_full(t) else win) * D * 2 * DT
        # Paged token-major: smallest read unit is a token row (all Hkv heads).
        # The layer must read every token any of its heads needs = union range.
        # If a layer has a full head, the union is the full context; every token
        # row read drags in all Hkv heads.
        cover = Lctx if any(is_full(t) for t in heads) else win
        paged += cover * Hkv * D * 2 * DT
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
    ap.add_argument("--lengths", default="16000,32000,64000,128000")
    ap.add_argument("--kv-dtype-bytes", type=int, default=2)
    ap.add_argument("--n-gen", type=int, default=256)
    # link bandwidths (GB/s): GPU HBM ref, PCIe4 x16, NVMe SSD
    ap.add_argument("--links", default="HBM:3000,PCIe4:25,NVMe:5")
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
    lengths = [int(x) for x in args.lengths.split(",")]
    links = [
        (s.split(":")[0], float(s.split(":")[1]) * 1e9) for s in args.links.split(",")
    ]

    flat = [t for row in cls for t in row]
    cnt = Counter(flat)
    nfull = sum(v for k, v in cnt.items() if is_full(k))
    print(f"Qwen3-32B L={L} Hkv={Hkv} D={D} KV={DT}B window={window} sink={sink}")
    print(f"kv-head: {dict(cnt)}  full={nfull}({100 * nfull / len(flat):.0f}%)")
    print(f"Scenario: OFFLINE KV on CPU/SSD, read back per step. n_gen={args.n_gen}")

    print("\n" + "=" * 84)
    print("IO READ per decode step (Paged token-major vs SegPaged head-major)")
    print("=" * 84)
    print(f"{'L_ctx':>7} | {'Paged_GB':>9} {'SegP_GB':>9} {'ratio':>6} {'IO saved':>9}")
    per_step = {}
    for Lctx in lengths:
        pg, sg = io_per_decode_step(cls, L, Hkv, D, DT, Lctx, window, sink)
        per_step[Lctx] = (pg, sg)
        print(
            f"{Lctx:>7} | {pg / 1e9:>9.3f} {sg / 1e9:>9.3f} {pg / sg:>5.2f}x {100 * (1 - sg / pg):>7.1f}%"
        )

    print("\n" + "=" * 84)
    print(f"DECODE latency for {args.n_gen} steps under slow links (read-bound)")
    print("=" * 84)
    hdr = f"{'L_ctx':>7} |"
    for name, _ in links:
        hdr += f" {name + ' Paged':>13} {name + ' SegP':>12}"
    print(hdr)
    for Lctx in lengths:
        pg, sg = per_step[Lctx]
        line = f"{Lctx:>7} |"
        for name, bw in links:
            tp = pg * args.n_gen / bw
            ts = sg * args.n_gen / bw
            line += f" {tp:>12.1f}s {ts:>11.1f}s"
        print(line)

    print("\nKey: ratio is IDENTICAL across links (pure IO-volume ratio), but the")
    print("absolute latency gap explodes on slow links. On HBM the gap is ~ms and")
    print("irrelevant; on PCIe/SSD (100-1000x slower) the same byte ratio becomes")
    print("seconds -> SegPaged's per-head read avoids dragging unused heads off disk.")

    # save
    out = {
        "scenario": "offline_kv_cpu_ssd",
        "per_step_GB": {
            str(k): {"paged": v[0] / 1e9, "segpaged": v[1] / 1e9, "ratio": v[0] / v[1]}
            for k, v in per_step.items()
        },
    }
    json.dump(
        out, open("/tmp/redknot_offload_io_paged_vs_segpaged.json", "w"), indent=2
    )
    print("\nSaved /tmp/redknot_offload_io_paged_vs_segpaged.json")


if __name__ == "__main__":
    main()
