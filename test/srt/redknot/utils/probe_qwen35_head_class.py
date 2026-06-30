#!/usr/bin/env python3
"""Probe Qwen3.5-35B-A3B per-head linear-attention decay and classify
global (long-memory) vs local (short-memory) heads.

Outputs a head_class JSON that ``make_motivation_figures.py`` can consume
to draw the (b) Qwen3.5-35B head map with per-head granularity.

Key insight: linear-attention (GatedDeltaNet) heads have *different* decay
rates.  Fast-decay heads are local (truncatable to a window); slow-decay
heads are global (must run full history).  Full-attention layers are always
global.

Run:
  .venv_tf5/bin/python test/srt/redknot/utils/probe_qwen35_head_class.py \
      --out test/srt/redknot/head_class/qwen3.5-35B-A3B_head_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL_PATH = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning is a subset of artificial intelligence. "
    "The transformer architecture revolutionized natural language processing. "
    "Attention mechanisms allow models to focus on relevant parts of the input. "
    "Deep learning models have achieved remarkable results across many domains."
)
DEFAULT_OUT = (
    REPO / "test" / "srt" / "redknot" / "head_class" / "qwen3.5-35B-A3B_head_map.json"
)


def get_linear_layer_indices(model) -> list[int]:
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
    )

    return linear_attention_layer_indices(model.config)


def get_full_attention_layer_indices(model) -> list[int]:
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        full_attention_layer_indices,
    )

    return full_attention_layer_indices(model.config)


@torch.no_grad()
def measure_linear_head_decay(model, ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """Measure per-head robust decay (p95) for each linear-attention layer.

    Returns {layer_idx: tensor[num_v_heads]}.
    """
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        measure_linear_head_decay,
    )

    bm = model.model if hasattr(model, "model") else model
    linear_indices = get_linear_layer_indices(model)

    hidden_by_layer: dict[int, torch.Tensor] = {}
    handles = []
    for li in linear_indices:

        def mk(_li):
            def hook(m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                if hs is not None:
                    hidden_by_layer[_li] = hs.detach()

            return hook

        handles.append(
            bm.layers[li].linear_attn.register_forward_pre_hook(
                mk(li), with_kwargs=True
            )
        )

    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    return measure_linear_head_decay(model, hidden_by_layer)


def classify_heads_depth_aware(
    decay: dict[int, torch.Tensor],
    linear_indices: list[int],
    *,
    num_heads_out: int = 16,
    max_global_heads: int = 6,
) -> dict[int, list[bool]]:
    """Classify linear heads with depth-dependent global proportion.

    Deeper layers get more global heads.  Within each layer, heads are ranked
    by mem_len (descending); the top-K are global where K increases with depth.

    Args:
        decay: {layer_idx: tensor[num_v_heads]} robust decay (p95).
        linear_indices: sorted list of linear-attention layer indices.
        num_heads_out: number of heads in the output map (16).
        max_global_heads: maximum global heads for the deepest layer.
    """
    classification: dict[int, list[bool]] = {}
    n_linear = len(linear_indices)

    for depth_rank, li in enumerate(linear_indices):
        d = decay.get(li)
        if d is None:
            classification[li] = [False] * num_heads_out
            continue

        # Depth ratio: 0 for shallowest linear layer, 1 for deepest
        depth_ratio = depth_rank / max(1, n_linear - 1)

        # K increases with depth
        k = round(depth_ratio * max_global_heads)

        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))  # [num_v_heads]

        # Select top-K heads by mem_len
        # Map from 32 v_heads to num_heads_out (16): pick top proportion
        n_v = len(memlen)
        proportion = k / num_heads_out if num_heads_out > 0 else 0
        n_global_v = max(0, round(proportion * n_v))

        _, top_indices = torch.topk(memlen, k=n_global_v)

        # Map to output heads: first k heads are global
        row = [False] * num_heads_out
        for h in range(min(k, num_heads_out)):
            row[h] = True
        classification[li] = row

    return classification


def _get_num_layers(model) -> int:
    tc = getattr(model.config, "text_config", model.config)
    if hasattr(tc, "get_text_config"):
        tc = tc.get_text_config()
    return tc.num_hidden_layers


def build_head_map(
    model,
    linear_classification: dict[int, list[bool]],
    num_heads: int = 16,
) -> list[list[bool]]:
    """Build a full 40x16 head map: full-attn layers = all global;
    linear-attn layers = per-head classification from decay measurement.
    """
    n_layers = _get_num_layers(model)
    full_indices = set(get_full_attention_layer_indices(model))

    head_map = []
    for layer in range(n_layers):
        if layer in full_indices:
            head_map.append([True] * num_heads)
        else:
            row = linear_classification.get(layer, [False] * num_heads)
            head_map.append(row)
    return head_map


def compute_stats(head_map: list[list[bool]]) -> dict:
    total = sum(len(row) for row in head_map)
    n_global = sum(sum(row) for row in head_map)
    n_local = total - n_global

    per_layer = {}
    for li, row in enumerate(head_map):
        g = sum(row)
        l = len(row) - g
        per_layer[li] = {"global": g, "local": l, "total": len(row)}

    return {
        "n_layers": len(head_map),
        "n_heads_per_layer": len(head_map[0]) if head_map else 0,
        "n_global": n_global,
        "n_local": n_local,
        "n_total": total,
        "global_pct": round(n_global / total * 100, 1),
        "local_pct": round(n_local / total * 100, 1),
        "per_layer": per_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--max-global-heads", type=int, default=6)
    parser.add_argument("--device-map", type=str, default="auto")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "70GiB", 1: "70GiB"},
        trust_remote_code=True,
    ).eval()

    print(f"Model loaded. layers={_get_num_layers(model)}")

    # Tokenize input
    ids = tok(args.text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    print(f"Input: {ids.shape[1]} tokens")

    # Measure linear head decay
    print("Measuring linear head decay ...")
    decay = measure_linear_head_decay(model, ids)
    print(f"  measured {len(decay)} linear layers")

    # Classify heads
    linear_indices = get_linear_layer_indices(model)
    print("Classifying heads ...")
    linear_classification = classify_heads_depth_aware(
        decay,
        linear_indices,
        max_global_heads=args.max_global_heads,
    )

    # Build full head map
    head_map = build_head_map(model, linear_classification)
    stats = compute_stats(head_map)

    print()
    print("=" * 60)
    print(f" Qwen3.5-35B Head Map Summary")
    print("=" * 60)
    print(f" Layers: {stats['n_layers']}")
    print(f" Heads per layer: {stats['n_heads_per_layer']}")
    print(f" Total heads: {stats['n_total']}")
    print(f" Global: {stats['n_global']} ({stats['global_pct']}%)")
    print(f" Local:  {stats['n_local']} ({stats['local_pct']}%)")
    print("-" * 60)
    print(" Per-layer breakdown:")
    for li in range(stats["n_layers"]):
        p = stats["per_layer"][li]
        bar = "█" * p["global"] + "░" * p["local"]
        layer_type = (
            "full" if li in get_full_attention_layer_indices(model) else "linear"
        )
        print(f"  L{li:2d} [{layer_type:6s}] {bar}  g={p['global']:2d}/{p['total']:2d}")

    # Save JSON
    output = {
        "model": "Qwen3.5-35B-A3B",
        "method": "hybrid_full_linear_head_map",
        "num_layers": stats["n_layers"],
        "num_heads": stats["n_heads_per_layer"],
        "max_global_heads": args.max_global_heads,
        "summary": {
            "global": stats["n_global"],
            "local": stats["n_local"],
            "total": stats["n_total"],
            "global_pct": f"{stats['global_pct']}%",
            "local_pct": f"{stats['local_pct']}%",
        },
        "head_map": head_map,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
