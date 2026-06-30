#!/usr/bin/env python3
"""Sweet-spot sweep for Qwen3.5-397B-A17B (FP8) — loads the model ONCE and
iterates over RedKnot sparsity knobs in-process.

Three knob groups (joint grid):
  1. Full-attention head class:  frac_global  x  full_window
  2. Linear-attention:           (kept at config defaults during MoE sweep)
  3. MoE deep-token-sparse:       deep_moe_start  x  mass_thresh

For each config it measures gsm8k-style F1 vs the STANDARD baseline and the
analytic compute saving (full / linear / moe components), and prints a sweep
table so the accuracy-vs-compute sweet spot can be read off.

Driven entirely by the same driver_qwen35 mechanisms the one-click benchmark
uses; this script only varies the knobs and aggregates.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ["REDKNOT_MODEL_PATH"]
LB = os.environ["REDKNOT_LONGBENCH_DIR"]
DATASET = os.environ.get("REDKNOT_DATASETS", "triviaqa").split(",")[0]
N = int(os.environ.get("REDKNOT_N_SAMPLES", "10"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "4"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK", "8000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"

# Linear-attention knobs (held fixed at sensible defaults during this sweep)
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))
DECAY_Q = float(os.environ.get("REDKNOT_DECAY_QUANTILE", "0.95"))
SAFETY = float(os.environ.get("REDKNOT_SAFETY", "2.0"))
MIN_WINDOW = int(os.environ.get("REDKNOT_MIN_WINDOW", "256"))
LINEAR_SEG = int(os.environ.get("REDKNOT_LINEAR_SEG", "2048"))

COMPONENT_SHARE = {"full": 0.06, "linear": 0.42, "moe": 0.35, "proj_norm": 0.17}

# ---- sweep grids (override via env as comma lists) ----
FRAC_GLOBAL_GRID = [
    float(x) for x in os.environ.get("SWEEP_FRAC_GLOBAL", "0.2,0.4,0.6,0.8").split(",")
]
FULL_WINDOW_GRID = [
    int(x)
    for x in os.environ.get("SWEEP_FULL_WINDOW", "1024,2048,4096,8192").split(",")
]
DEEP_MOE_GRID = [
    int(x) for x in os.environ.get("SWEEP_DEEP_MOE_START", "16,24,32,40").split(",")
]
MASS_THRESH_GRID = [
    float(x) for x in os.environ.get("SWEEP_MASS_THRESH", "0.1,0.3,0.5,0.7").split(",")
]
DENSE_FULL_LAYERS = int(
    os.environ.get("REDKNOT_DENSE_FULL_LAYERS", "9")
)  # 397B: 15 full layers -> ceil(15/2)+1


# ---------- metrics ----------
def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, golds):
    best = 0.0
    for g in golds:
        p, gg = _norm(pred).split(), _norm(g).split()
        if not p or not gg:
            best = max(best, float(p == gg))
            continue
        c = Counter(p) & Counter(gg)
        ns = sum(c.values())
        if ns == 0:
            continue
        prec, rec = ns / len(p), ns / len(gg)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return (lines[0] if lines else t).strip().strip('"').strip("'")


def _fwd_last(model, **kw):
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError as e:
        if "logits_to_keep" not in str(e):
            raise
        return model(**kw)


@torch.no_grad()
def standard_gen(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    out = _fwd_last(model, input_ids=ids, use_cache=True)
    nxt = out.logits[0, -1, :].argmax().view(1, 1)
    past = out.past_key_values
    g = [int(nxt[0, 0])]
    for _ in range(MAX_NEW - 1):
        og = _fwd_last(model, input_ids=nxt, past_key_values=past, use_cache=True)
        past = og.past_key_values
        nxt = og.logits[0, -1, :].argmax().view(1, 1)
        t = int(nxt[0, 0])
        g.append(t)
        if t == tok.eos_token_id:
            break
    return tok.decode(g, skip_special_tokens=True)


def load_samples(tok):
    target = N_CHUNK * CHUNK
    raw = []
    with open(os.path.join(LB, f"{DATASET}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(SEED).shuffle(raw)
    out, nraw = [], len(raw)
    for i in range(nraw):
        if len(out) >= N:
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
        out.append({"q": raw[i]["input"], "golds": raw[i]["answers"], "chunks": chunks})
    return out


@torch.no_grad()
def redknot_gen(
    model, tok, drv, chunks, qt, frac_global, full_window, deep_moe_start, mass_thresh
):
    from transformers import DynamicCache

    device = model.device
    text = "\n\n".join(chunks) + qt
    ids0 = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )

    mass = drv["collect_attention_mass"](model, ids0, deep_full_frac=0.5)
    moe_skip = float((mass < mass_thresh).float().mean().item())

    # linear per-head window (held at fixed linear knobs)
    win, lin_windowed, lin_save = build_win(model, tok, drv, text)

    head_cfg = drv["build_full_attention_head_config"](
        model.config, frac_global=frac_global, local_window=full_window
    )
    rf = drv["_install_full_patches"](
        model, head_cfg, dense_prefix_full_layers=DENSE_FULL_LAYERS
    )
    rl = drv["install_linear_segmented"](model, win, seg=LINEAR_SEG)
    rm = drv["install_moe_token_sparse"](
        model, mass, deep_moe_start_layer=deep_moe_start, mass_thresh=mass_thresh
    )
    try:
        cache = DynamicCache(config=model.config)
        pos = 0
        last = None
        for piece in list(chunks) + [qt]:
            ids = tok(piece, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
            out = _fwd_last(
                model,
                input_ids=ids,
                position_ids=pids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            last = out.logits[0, -1, :]
            pos += ids.shape[1]
        nxt = last.argmax().view(1, 1)
        g = [int(nxt[0, 0])]
        for _ in range(MAX_NEW - 1):
            pids = torch.tensor([[pos]], device=device)
            og = _fwd_last(
                model,
                input_ids=nxt,
                position_ids=pids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            t = int(nxt[0, 0])
            g.append(t)
            pos += 1
            if t == tok.eos_token_id:
                break
        return tok.decode(g, skip_special_tokens=True), lin_windowed, lin_save, moe_skip
    finally:
        rm()
        rl()
        rf()


@torch.no_grad()
def build_win(model, tok, drv, sample_text):
    bm = model.model if hasattr(model, "model") else model
    ids = tok(sample_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    hs_in = {}
    handles = []
    for li in drv["linear_attention_layer_indices"](model.config):

        def mk(_li):
            def hook(m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                if hs is not None:
                    hs_in[_li] = hs.detach()

            return hook

        handles.append(
            bm.layers[li].linear_attn.register_forward_pre_hook(
                mk(li), with_kwargs=True
            )
        )
    _fwd_last(model, input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    decay = drv["measure_linear_head_decay"](model, hs_in, decay_quantile=DECAY_Q)
    win = {}
    nloc = ntot = 0
    ctx = N_CHUNK * CHUNK
    for li, d in decay.items():
        if li < DENSE_PREFIX:
            win[li] = None
            ntot += len(d)
            continue
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(SAFETY * memlen).long().clamp(min=MIN_WINDOW)
        wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
        win[li] = wt
        nloc += int((wt > 0).sum())
        ntot += len(d)
    save_sum = 0.0
    for li, wt in win.items():
        if wt is None:
            continue
        for h in range(wt.numel()):
            w = int(wt[h].item())
            if 0 < w < ctx:
                save_sum += 1.0 - w / ctx
    return win, nloc / max(ntot, 1), save_sum / max(ntot, 1)


def full_attn_save(frac_global, full_window, n_full):
    T = N_CHUNK * CHUNK
    dc = T * (T + 1) / 2.0
    sc = frac_global * dc + (1 - frac_global) * T * min(full_window, T)
    ns = max(0, n_full - DENSE_FULL_LAYERS)
    return 1.0 - (DENSE_FULL_LAYERS * dc + ns * sc) / (n_full * dc)


def total_compute_save(full_s, lin_s, moe_s):
    active = (
        COMPONENT_SHARE["full"] + COMPONENT_SHARE["linear"] + COMPONENT_SHARE["moe"]
    )
    saved = (
        COMPONENT_SHARE["full"] * full_s
        + COMPONENT_SHARE["linear"] * lin_s
        + COMPONENT_SHARE["moe"] * moe_s
    )
    return saved / active


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot import driver_qwen35 as D

    drv = {
        k: getattr(D, k)
        for k in [
            "_install_full_patches",
            "build_full_attention_head_config",
            "collect_attention_mass",
            "full_attention_layer_indices",
            "linear_attention_layer_indices",
            "install_linear_segmented",
            "install_moe_token_sparse",
            "measure_linear_head_decay",
        ]
    }

    print("=" * 100)
    print(
        f"Qwen3.5-397B RedKnot SWEET-SPOT sweep | dataset={DATASET} N={N} ctx={N_CHUNK * CHUNK} MAX_NEW={MAX_NEW}"
    )
    print("=" * 100, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loading model (once)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    n_full = len(drv["full_attention_layer_indices"](model.config))
    print(
        f"n_full_attention_layers={n_full} dense_full_layers={DENSE_FULL_LAYERS}",
        flush=True,
    )

    samples = load_samples(tok)
    print(f"loaded {len(samples)} samples", flush=True)

    # ---- baseline once per sample ----
    print("\n>>> computing STANDARD baseline ...", flush=True)
    base_f1 = 0.0
    base_out = []
    for s in samples:
        txt = "\n\n".join(s["chunks"]) + QP.format(q=s["q"])
        o = short(standard_gen(model, tok, txt))
        base_out.append(o)
        base_f1 += f1(o, s["golds"])
    base_f1 /= max(len(samples), 1)
    print(f"baseline F1 = {base_f1:.3f}", flush=True)

    # ---- joint grid sweep ----
    grid = list(
        itertools.product(
            FRAC_GLOBAL_GRID, FULL_WINDOW_GRID, DEEP_MOE_GRID, MASS_THRESH_GRID
        )
    )
    print(
        f"\n>>> sweeping {len(grid)} configs (frac_global x full_window x deep_moe_start x mass_thresh)",
        flush=True,
    )
    print(
        f"{'fracG':>6} {'window':>7} {'deepMoE':>8} {'mass':>5} | {'rkF1':>6} {'dF1':>6} {'fullSv':>7} {'linSv':>6} {'moeSk':>6} {'totSv%':>7}",
        flush=True,
    )
    print("-" * 100, flush=True)

    results = []
    for fg, fw, dms, mt in grid:
        rkf1 = 0.0
        lin_s = moe_sk = 0.0
        for s in samples:
            qt = QP.format(q=s["q"])
            out, linw, lins, moesk = redknot_gen(
                model, tok, drv, s["chunks"], qt, fg, fw, dms, mt
            )
            rkf1 += f1(short(out), s["golds"])
            lin_s += lins
            moe_sk += moesk
        k = max(len(samples), 1)
        rkf1 /= k
        lin_s /= k
        moe_sk /= k
        n_layers = getattr(
            getattr(model.config, "text_config", model.config), "num_hidden_layers"
        )
        deep_layers = len([i for i in range(n_layers) if i >= dms])
        moe_s = moe_sk * (deep_layers / n_layers)
        full_s = full_attn_save(fg, fw, n_full)
        tot = total_compute_save(full_s, lin_s, moe_s)
        d_f1 = rkf1 - base_f1
        results.append((fg, fw, dms, mt, rkf1, d_f1, full_s, lin_s, moe_sk, tot))
        print(
            f"{fg:>6.2f} {fw:>7d} {dms:>8d} {mt:>5.2f} | {rkf1:>6.3f} {d_f1:>+6.3f} {full_s:>7.3f} {lin_s:>6.3f} {moe_sk:>6.3f} {tot * 100:>6.1f}%",
            flush=True,
        )

    # ---- sweet spot: lossless (dF1>=-0.02) with max compute save ----
    print("=" * 100, flush=True)
    lossless = [r for r in results if r[5] >= -0.02]
    pool = lossless if lossless else results
    best = max(pool, key=lambda r: r[9])
    print(f"baseline F1 = {base_f1:.3f}")
    print(f"SWEET SPOT (lossless, max compute save):")
    print(
        f"  frac_global={best[0]} full_window={best[1]} deep_moe_start={best[2]} mass_thresh={best[3]}"
    )
    print(
        f"  rkF1={best[4]:.3f} (dF1={best[5]:+.3f})  total_compute_save={best[9] * 100:.1f}%"
    )
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
