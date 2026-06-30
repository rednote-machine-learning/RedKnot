#!/usr/bin/env python3
"""Can Qwen3.5 full-attention be REALLY sparse (window < context) AND accurate?

window>=context is a fake sweet spot: local heads then see everything, so full
attention saves nothing. The real question: with a TRULY sparse window
(window << context), how much GLOBAL-head budget is needed to recover accuracy,
and how much compute does that still save?

Fixes window=4096 (real sparsity at 16K context) and sweeps frac_global from low
to high. For each: match rate + mean logit diff vs native, plus the ANALYTIC
full-attn FLOPs saving (so we can see the accuracy/compute tradeoff). If accuracy
only recovers at very high frac_global (-> little saving), Qwen3.5 full layers
are hard to sparsify; if it recovers at modest frac_global, there's a sweet spot.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/sweep_qwen35_realsparse.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "4"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))
FRACS = [
    float(x)
    for x in os.environ.get("REDKNOT_FRACS", "0.10,0.25,0.40,0.60,0.80").split(",")
]
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


def load_4chunk(name, tok, n, seed):
    raw = []
    with open(os.path.join(LB, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    target = N_CHUNK * CHUNK
    out, nraw = [], len(raw)
    for i in range(nraw):
        if len(out) >= n:
            break
        toks = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % nraw
        while len(toks) < target and j != i:
            toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % nraw
        toks = toks[:target]
        if len(toks) < target:
            continue
        chunks = [
            tok.decode(toks[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, target, CHUNK)
        ]
        out.append({"q": raw[i]["input"], "chunks": chunks})
    return out


def full_attn_saving(frac_global, window, T):
    """Analytic full-attn FLOPs saving vs dense, for one full layer.
    global head cost ~ T^2/2 ; local head cost ~ T*window. Returns fraction saved.
    """
    dense = T * (T + 1) / 2.0
    hc = frac_global * dense + (1 - frac_global) * T * min(window, T)
    return 1.0 - hc / dense


@torch.no_grad()
def eval_cfg(model, tok, samples, refs, frac, window):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        build_full_attention_head_config,
    )
    from transformers import DynamicCache

    head_cfg = build_full_attention_head_config(
        model.config, frac_global=frac, local_window=window, global_assign="random"
    )
    matches, diffs = [], []
    restore = _install_full_patches(model, head_cfg)
    try:
        for s, ref in zip(samples, refs):
            cache = DynamicCache(config=model.config)
            pos = 0
            last = None
            for piece in s["chunks"] + [s["qt"]]:
                ids = tok(piece, return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ].to(model.device)
                pids = torch.arange(
                    pos, pos + ids.shape[1], device=model.device
                ).unsqueeze(0)
                out = model(
                    input_ids=ids,
                    position_ids=pids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                last = out.logits[0, -1, :].float()
                pos += ids.shape[1]
            matches.append(int(last.argmax() == ref.argmax()))
            diffs.append((last - ref).abs().max().item())
    finally:
        restore()
    return sum(matches) / len(matches), sum(diffs) / len(diffs)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    samples, refs = [], []
    for ds in DATASETS:
        for s in load_4chunk(ds, tok, N, SEED):
            s["qt"] = QP.format(q=s["q"])
            rids = tok(
                "\n\n".join(s["chunks"]) + s["qt"],
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(model.device)
            refs.append(model(input_ids=rids, use_cache=False).logits[0, -1, :].float())
            samples.append(s)
    T = N_CHUNK * CHUNK
    print(f"samples={len(samples)}  context~{T} tok  window={WINDOW} (REAL sparse)")

    W = 78
    print("=" * W)
    print(f" REAL-SPARSE sweep: window={WINDOW} fixed, vary frac_global")
    print(" (match rate / logit diff vs native  +  analytic full-attn FLOPs saving)")
    print("=" * W)
    print(f" {'frac_g':>8} {'match':>7} {'logit_diff':>11} {'full-attn save':>16}")
    for fr in FRACS:
        m, d = eval_cfg(model, tok, samples, refs, fr, WINDOW)
        save = full_attn_saving(fr, WINDOW, T)
        print(f" {fr:>8.2f} {m:>7.2f} {d:>11.3f} {save * 100:>14.1f}%")
    print("=" * W)
    print(" Read: the lowest frac_global with match~1.0 = the real sweet spot;")
    print(" its 'save' column shows whether full attention still saves there.")
    print("=" * W)


if __name__ == "__main__":
    main()
