#!/usr/bin/env python3
"""Data-driven linear head-class windowing on Qwen3.5-35B-A3B.

New criterion (not decay rate): for each (layer,head) directly MEASURE how much
truncating to a sliding window changes that head's output; classify LOCAL only
if the measured error < threshold. Captures information importance.

full=exact. linear: local heads (low truncation error) windowed, rest full.
Dense L0..L4. Reports F1 vs baseline + local-head fraction (theoretical saving).

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/bench_linear_datadriven.py
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_CTX = int(os.environ.get("REDKNOT_MAX_CTX", "8000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "512"))
ERR = float(os.environ.get("REDKNOT_ERR_THRESH", "0.05"))
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


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
        common = Counter(p) & Counter(gg)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec, rec = ns / len(p), ns / len(gg)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def em(pred, golds):
    return max((float(_norm(pred) == _norm(g)) for g in golds), default=0.0)


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    cand = (lines[0] if lines else t).strip().strip('"').strip("'")
    return re.sub(r"\s*[.。]\s*$", "", cand)


def load_ds(name, tok, n, seed):
    raw = []
    with open(os.path.join(LB, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    out = []
    for r in raw[:n]:
        ids = tok(r["context"], add_special_tokens=False)["input_ids"][:MAX_CTX]
        out.append({"q": r["input"], "golds": r["answers"], "ctx_ids": ids, "ds": name})
    return out


@torch.no_grad()
def gen(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    out = model(input_ids=ids, use_cache=True)
    nxt = out.logits[0, -1, :].argmax().view(1, 1)
    past = out.past_key_values
    g = [int(nxt[0, 0])]
    for _ in range(MAX_NEW - 1):
        og = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = og.past_key_values
        nxt = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        g.append(tid)
        if tid == tok.eos_token_id:
            break
    return tok.decode(g, skip_special_tokens=True)


@torch.no_grad()
def capture_hidden(model, tok, ids_t):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
    )

    bm = model.model if hasattr(model, "model") else model
    hs_in = {}
    handles = []
    for li in linear_attention_layer_indices(model.config):

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
    model(input_ids=ids_t, use_cache=False)
    for h in handles:
        h.remove()
    return hs_in


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        classify_linear_heads_by_truncation,
        install_linear_headclass_winmap,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    samples = []
    for ds in DATASETS:
        samples += load_ds(ds, tok, N, SEED)

    W = 92
    print("=" * W)
    print(f" DATA-DRIVEN linear head-class — {Path(MODEL).name}")
    print(
        f" criterion: measured truncation error < {ERR} -> LOCAL window={WINDOW} | dense L0..L{DENSE_PREFIX - 1}"
    )
    print("=" * W)
    bf = be = rf = re_ = 0.0
    nloc = ntot = 0
    per_ds = {}
    last_info = {}
    for s in samples:
        text = tok.decode(s["ctx_ids"], skip_special_tokens=True) + QP.format(q=s["q"])
        bb = short(gen(model, tok, text))
        bF = f1(bb, s["golds"])
        bE = em(bb, s["golds"])
        ids_t = tok(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        hs_in = capture_hidden(model, tok, ids_t)
        win_map, info = classify_linear_heads_by_truncation(
            model, hs_in, WINDOW, err_thresh=ERR, dense_prefix_layers=DENSE_PREFIX
        )
        last_info = info
        restore = install_linear_headclass_winmap(model, win_map)
        try:
            rb = short(gen(model, tok, text))
        finally:
            restore()
        rF = f1(rb, s["golds"])
        rE = em(rb, s["golds"])
        bf += bF
        be += bE
        rf += rF
        re_ += rE
        for li, (l, gg, _) in info.items():
            nloc += l
            ntot += l + gg
        d = per_ds.setdefault(s["ds"], [0.0, 0.0, 0])
        d[0] += bF
        d[1] += rF
        d[2] += 1

    k = len(samples)
    print(f" {'dataset':16} {'base F1':>8} {'hc F1':>8} {'dF1':>7}")
    for ds, (b, r, c) in per_ds.items():
        print(f" {ds:16} {b / c:8.3f} {r / c:8.3f} {(r - b) / c:+7.3f}")
    print("-" * W)
    print(
        f" AVG base F1={bf / k:.3f} headclass F1={rf / k:.3f} dF1={rf / k - bf / k:+.3f} EM {be / k:.3f}->{re_ / k:.3f}"
    )
    print(f" linear local-head frac (windowed) = {nloc / max(ntot, 1) * 100:.1f}%")
    print("=" * W)


if __name__ == "__main__":
    main()
