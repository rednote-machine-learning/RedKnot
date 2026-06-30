#!/usr/bin/env python3
"""Analyze per-head decay of Qwen3.5 GatedDeltaNet (linear attention) heads.

Hypothesis (user's "split linear into heads"): different linear heads decay at
different rates. Fast-decay heads forget history quickly -> they effectively
depend only on RECENT tokens -> can be truncated to a window (saving compute).
Slow-decay heads (g.exp()~1) keep long memory -> must run full. If the heads
split into fast/slow groups, linear can do head-class sparsity like full attn.

The per-step decay factor is g.exp() in (0,1), where
    g = -A_log.exp() * softplus(a + dt_bias)
A_log/dt_bias are per-head params; `a` is input-dependent. We run real text and
measure, per (layer, head), the MEAN decay factor and the "effective memory
length" ~ 1/(1-mean_decay) (tokens until state shrinks to 1/e).

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/analyze_linear_head_decay.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DATASET = os.environ.get("REDKNOT_DATASETS", "hotpotqa").split(",")[0]
NTOK = int(os.environ.get("REDKNOT_NTOK", "8000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    bm = model.model if hasattr(model, "model") else model

    # real text
    raw = []
    with open(os.path.join(LB, f"{DATASET}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("context"):
                raw.append(r)
    random.Random(0).shuffle(raw)
    toks = tok(raw[0]["context"], add_special_tokens=False)["input_ids"][:NTOK]
    ids = torch.tensor([toks], device=model.device)
    print(f"analyzing {ids.shape[1]} tokens on {DATASET}")

    # hook each linear_attn to capture its decay factors g.exp() per token/head
    lin_layers = [
        (i, l.linear_attn) for i, l in enumerate(bm.layers) if hasattr(l, "linear_attn")
    ]
    stats = {}  # layer_idx -> mean decay per head [num_v_heads]

    handles = []
    for li, mod in lin_layers:

        def mk(_li, _mod):
            def hook(m, args, kwargs):
                try:
                    hs = None
                    if args and torch.is_tensor(args[0]):
                        hs = args[0]
                    elif "hidden_states" in kwargs:
                        hs = kwargs["hidden_states"]
                    if hs is None:
                        return
                    a = _mod.in_proj_a(hs.to(_mod.in_proj_a.weight.dtype))
                    a = a.to(_mod.A_log.device)
                    g = -_mod.A_log.float().exp() * F.softplus(a.float() + _mod.dt_bias)
                    decay = g.exp()
                    stats[_li] = (
                        decay.reshape(-1, decay.shape[-1]).mean(dim=0).float().cpu()
                    )
                except Exception as e:  # noqa
                    if _li == lin_layers[0][0]:
                        print(
                            f"  [hook err layer {_li}] {type(e).__name__}: {str(e)[:120]}",
                            flush=True,
                        )

            return hook

        handles.append(mod.register_forward_pre_hook(mk(li, mod), with_kwargs=True))
    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    # aggregate
    print("=" * 78)
    print(" Per-layer linear-head decay (mean g.exp over tokens). decay~1 = long")
    print(" memory (must run full); decay<<1 = fast forget (truncatable to window).")
    print("=" * 78)
    all_decay = []
    all_memlen = []
    for li in sorted(stats):
        d = stats[li]  # [H]
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))  # effective memory length (tokens)
        all_decay.append(d)
        all_memlen.append(memlen)
        print(
            f" layer {li:2d}: decay min={d.min():.4f} mean={d.mean():.4f} max={d.max():.4f} "
            f"| mem_len(tok) median={memlen.median():.0f} max={memlen.max():.0f}"
        )
    D = torch.cat(all_decay)
    M = torch.cat(all_memlen)
    print("-" * 78)
    print(f" ALL HEADS: n={D.numel()}")
    print(
        f"   decay   : min={D.min():.4f} p25={D.quantile(0.25):.4f} med={D.median():.4f} p75={D.quantile(0.75):.4f} max={D.max():.4f}"
    )
    print(
        f"   mem_len : med={M.median():.0f}  p90={M.quantile(0.90):.0f}  max={M.max():.0f} tokens"
    )
    # how many heads have short memory (truncatable)?
    for w in [256, 512, 1024, 2048]:
        frac = (M < w).float().mean().item()
        print(
            f"   heads with mem_len < {w:5d} tok: {frac * 100:5.1f}%  (truncatable to window {w})"
        )
    print("=" * 78)
    print(" If a large fraction of heads have short mem_len, linear head-class")
    print(" sparsity (truncate fast heads to a window) can save compute.")
    print("=" * 78)


if __name__ == "__main__":
    main()
