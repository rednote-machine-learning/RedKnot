#!/usr/bin/env python3
"""gen_qwen35_server_head_config.py — convert a Qwen3.5 RedKnot "param" JSON into
a HeadClassConfig that the SGLang server backend can actually load.

The hand-written ``head_class/qwen3.5-*_redknot.json`` files describe the
sweet-spot params (frac_global, window, dense_full_layers, full_attention_layers,
...) but are NOT in the per-(layer,head) matrix format that
``HeadClassConfig.from_json`` expects. The HF-transformers benchmark path builds
the matrix at runtime via ``build_full_attention_head_config``; the SGLang server
needs it on disk.

This script reproduces ``build_full_attention_head_config``'s "random" head
assignment (same seed) for the FULL-ATTENTION layers only, marks the first
``dense_full_layers`` rows as dense (via ``dense_prefix_layers``), and writes a
``*_server.json`` next to the source. The RedKnot backend indexes this config by
full-attention POSITION (0..n_full-1) and maps global layer ids onto it.

Usage:
  python test/srt/redknot/gen_qwen35_server_head_config.py \
      test/srt/redknot/head_class/qwen3.5-397B-A17B_redknot.json
  # -> writes qwen3.5-397B-A17B_redknot_server.json

  # or convert all qwen3.5 param configs in the dir:
  python test/srt/redknot/gen_qwen35_server_head_config.py --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

HEAD_DIR = Path(__file__).resolve().parent / "head_class"


def convert(src: Path, seed: int = 1234) -> Path:
    d = json.load(open(src))

    n_full = int(d["num_full_layers"])
    H = int(d["num_kv_heads"])
    frac_global = float(d["frac_global"])
    window = int(d["window"])
    sink = int(d.get("sink_size", 4))
    dense_full = int(d["dense_full_layers"])

    # Mirror build_full_attention_head_config(global_assign="random").
    total = n_full * H
    n_global = max(1, round(frac_global * total))
    rng = random.Random(seed)
    coords = [(li, h) for li in range(n_full) for h in range(H)]
    rng.shuffle(coords)
    global_set = set(coords[:n_global])

    kv_head_classification = []
    kv_head_max_distance = []
    kv_head_sink_size = []
    for li in range(n_full):
        cls_row, dist_row, sink_row = [], [], []
        for h in range(H):
            if (li, h) in global_set:
                cls_row.append("global")
                dist_row.append(-1)
            else:
                cls_row.append("local")
                dist_row.append(window)
            sink_row.append(sink)
        kv_head_classification.append(cls_row)
        kv_head_max_distance.append(dist_row)
        kv_head_sink_size.append(sink_row)

    out = {
        "model": d.get("model"),
        "model_type": d.get("model_type"),
        "method": d.get("method"),
        "comment": (
            "Loadable HeadClassConfig for the SGLang RedKnot backend. Indexed by "
            "FULL-ATTENTION position (0..%d), NOT global layer id; the backend "
            "maps global full-attn layer ids -> these rows. Generated from the "
            "sweet-spot params (frac_global=%s, window=%s, dense_full_layers=%s)."
            % (n_full - 1, frac_global, window, dense_full)
        ),
        "num_layers": n_full,
        "num_kv_heads": H,
        "dense_prefix_layers": dense_full,
        "kv_head_classification": kv_head_classification,
        "kv_head_max_distance": kv_head_max_distance,
        "kv_head_sink_size": kv_head_sink_size,
        "full_attention_layers": d.get("full_attention_layers"),
        "full_attention_interval": d.get("full_attention_interval"),
        "provenance": {
            k: d[k]
            for k in (
                "frac_global",
                "window",
                "sink_size",
                "dense_full_layers",
                "sparse_full_layers",
            )
            if k in d
        },
    }

    dst = src.with_name(src.stem + "_server.json")
    json.dump(out, open(dst, "w"), indent=2)

    # Verify it loads.
    from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig

    cfg = HeadClassConfig.from_json(str(dst))
    print(
        f"[ok] {src.name} -> {dst.name} "
        f"(num_layers={cfg.num_layers}, num_kv_heads={cfg.num_kv_heads}, "
        f"summary={cfg.summary()})"
    )
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="path to a *_redknot.json param file.")
    ap.add_argument(
        "--all",
        action="store_true",
        help="convert every qwen3.5-*_redknot.json (skipping *_server.json).",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.all:
        srcs = sorted(
            p
            for p in HEAD_DIR.glob("qwen3.5-*_redknot.json")
            if not p.stem.endswith("_server")
        )
        if not srcs:
            raise SystemExit(f"no qwen3.5-*_redknot.json under {HEAD_DIR}")
        for s in srcs:
            convert(s, seed=args.seed)
    elif args.src:
        convert(Path(args.src), seed=args.seed)
    else:
        raise SystemExit("provide a src path or --all")


if __name__ == "__main__":
    main()
