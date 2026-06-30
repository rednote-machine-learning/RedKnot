#!/usr/bin/env python3
"""Component-level TTFT breakdown for 2K x 10 RAG on Qwen3.5-35B-A3B.

Decomposes prefill TTFT into the three mixers/blocks:
  * full_attention  (self_attn)      — what RedKnot sparsifies
  * linear_attention (linear_attn)   — GatedDeltaNet
  * moe_ffn          (mlp)           — MoE block (unchanged)
plus the residual ("other": norms, embed, lm_head, rope, dispatch).

Times are wall-clock per-module via CUDA events accumulated over the prefill.
Compares STANDARD inference vs RedKnot chunked (full head-class sparse). This
shows where the time actually goes at this scale and whether sparsifying full
attention can move the needle.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/profile_components_2kx10.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch

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
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "10"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "256"))
FRAC = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.10"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


def load_one(name, tok, target, seed=0):
    raw = []
    with open(os.path.join(LB, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    toks = tok(raw[0]["context"], add_special_tokens=False)["input_ids"]
    j = 1
    while len(toks) < target and j < len(raw):
        toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
        j += 1
    toks = toks[:target]
    chunks = [
        tok.decode(toks[k : k + CHUNK], skip_special_tokens=True)
        for k in range(0, target, CHUNK)
    ]
    return {"q": raw[0]["input"], "chunks": chunks}


class Timer:
    """Accumulate per-category CUDA time via module hooks."""

    def __init__(self):
        self.acc = {"full_attn": 0.0, "linear_attn": 0.0, "moe_ffn": 0.0}
        self._ev = {}

    def hook_pre(self, cat):
        def f(mod, inp):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._ev[(cat, id(mod))] = e

        return f

    def hook_post(self, cat):
        def f(mod, inp, out):
            s = self._ev.pop((cat, id(mod)), None)
            if s is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                e.synchronize()
                self.acc[cat] += s.elapsed_time(e)  # ms

        return f


def attach(model, timer):
    bm = model.model if hasattr(model, "model") else model
    handles = []
    for layer in bm.layers:
        if hasattr(layer, "self_attn"):
            handles.append(
                layer.self_attn.register_forward_pre_hook(timer.hook_pre("full_attn"))
            )
            handles.append(
                layer.self_attn.register_forward_hook(timer.hook_post("full_attn"))
            )
        if hasattr(layer, "linear_attn"):
            handles.append(
                layer.linear_attn.register_forward_pre_hook(
                    timer.hook_pre("linear_attn")
                )
            )
            handles.append(
                layer.linear_attn.register_forward_hook(timer.hook_post("linear_attn"))
            )
        if hasattr(layer, "mlp"):
            handles.append(
                layer.mlp.register_forward_pre_hook(timer.hook_pre("moe_ffn"))
            )
            handles.append(layer.mlp.register_forward_hook(timer.hook_post("moe_ffn")))
    return handles


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        build_full_attention_head_config,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    T = N_CHUNK * CHUNK
    s = load_one(DATASET, tok, T)
    qt = QP.format(q=s["q"])

    def report(name, timer, total_ms):
        a = timer.acc
        known = a["full_attn"] + a["linear_attn"] + a["moe_ffn"]
        other = max(0.0, total_ms - known)
        print(f"\n {name}  (total prefill {total_ms:.1f} ms)")
        for k in ["full_attn", "linear_attn", "moe_ffn"]:
            print(f"   {k:12} {a[k]:8.1f} ms  ({a[k] / total_ms * 100:5.1f}%)")
        print(f"   {'other':12} {other:8.1f} ms  ({other / total_ms * 100:5.1f}%)")

    # ── STANDARD: single full 16K forward ──
    t = Timer()
    h = attach(model, t)
    ids = tok(
        "\n\n".join(s["chunks"]) + qt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    std_ms = (time.perf_counter() - t0) * 1000
    for x in h:
        x.remove()
    print("=" * 70)
    print(f" COMPONENT TTFT BREAKDOWN — {Path(MODEL).name} | {N_CHUNK}x{CHUNK}={T} tok")
    print(f" window={WINDOW} frac_global={FRAC}")
    print("=" * 70)
    report("STANDARD (dense full, single pass)", t, std_ms)

    # ── REDKNOT: NON-CHUNKED single forward; full=exact, MoE once, only LINEAR
    #    local heads windowed in-layer (no chunk -> no MoE/full repeat) ──
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        install_linear_token_window,
        linear_attention_layer_indices,
        measure_linear_head_decay,
    )

    bm = model.model if hasattr(model, "model") else model
    # build per-head token windows from decay
    hs_in = {}
    hh = []
    for li in linear_attention_layer_indices(model.config):

        def mk(_li):
            def hook(m, a, k):
                hh_ = a[0] if a and torch.is_tensor(a[0]) else k.get("hidden_states")
                if hh_ is not None:
                    hs_in[_li] = hh_.detach()

            return hook

        hh.append(
            bm.layers[li].linear_attn.register_forward_pre_hook(
                mk(li), with_kwargs=True
            )
        )
    model(input_ids=ids, use_cache=False)
    for h in hh:
        h.remove()
    decay = measure_linear_head_decay(model, hs_in, decay_quantile=0.95)
    win = {}
    for li, d in decay.items():
        if li < 5:
            win[li] = None
            continue
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(4.0 * memlen).long().clamp(min=512)
        wt = torch.where(wt >= T, torch.zeros_like(wt), wt)
        win[li] = wt

    t2 = Timer()
    restore = install_linear_token_window(model, win)
    h2 = attach(model, t2)
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=True)  # SINGLE forward, full 20K
        torch.cuda.synchronize()
        rk_ms = (time.perf_counter() - t0) * 1000
    finally:
        for x in h2:
            x.remove()
        restore()
    report("REDKNOT (non-chunked, linear local windowed)", t2, rk_ms)

    print("\n" + "=" * 70)
    print(f" full_attn: {t.acc['full_attn']:.0f}ms -> {t2.acc['full_attn']:.0f}ms")
    print(
        f" linear_attn: {t.acc['linear_attn']:.0f}ms -> {t2.acc['linear_attn']:.0f}ms"
    )
    print(f" moe_ffn: {t.acc['moe_ffn']:.0f}ms -> {t2.acc['moe_ffn']:.0f}ms")
    print(
        f" TOTAL: {std_ms:.0f}ms -> {rk_ms:.0f}ms  ({std_ms / max(rk_ms, 1e-3):.2f}x)"
    )
    print("=" * 70)
    print(" Read: if full_attn is a SMALL slice of total, sparsifying it can't")
    print(" speed up TTFT much at this scale — the bottleneck is moe_ffn/linear.")
    print("=" * 70)


if __name__ == "__main__":
    main()
