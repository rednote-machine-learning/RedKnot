#!/usr/bin/env python3
"""Verify chunked linear relay (native kernel + initial_state) == single pass.

Uses run_redknot_qwen35_linear which chunks the input and relays each linear
layer's recurrent state across chunks via the NATIVE chunk_gated_delta_rule
(initial_state). full attention unchanged. If this equals the baseline single
forward, the minimal-invasive chunked relay is faithful (then we can add local-
head windowing on top).

Compares next-token F1 and exact-match of generated text vs baseline.
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

MODEL = os.environ.get("REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B")
LB = os.environ.get("REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data")
DATASETS = os.environ.get("REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa").split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "8"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


def _norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())

def f1(pred, golds):
    best = 0.0
    for g in golds:
        p, gg = _norm(pred).split(), _norm(g).split()
        if not p or not gg: best = max(best, float(p == gg)); continue
        c = Counter(p) & Counter(gg); ns = sum(c.values())
        if ns == 0: continue
        prec, rec = ns/len(p), ns/len(gg); best = max(best, 2*prec*rec/(prec+rec))
    return best

def short(t):
    t = (t or "").strip(); t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    # strip chat-template boundary leakage (assistant/Human/<|...|>/* markers)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return (lines[0] if lines else t).strip().strip('"').strip("'")

def load_nchunk(name, tok, n, seed=0):
    raw = []
    with open(os.path.join(LB, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"): raw.append(r)
    random.Random(seed).shuffle(raw)
    target = N_CHUNK*CHUNK; out, nraw = [], len(raw)
    for i in range(nraw):
        if len(out) >= n: break
        toks = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]; j=(i+1)%nraw
        while len(toks) < target and j != i:
            toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]; j=(j+1)%nraw
        toks = toks[:target]
        if len(toks) < target: continue
        chunks = [tok.decode(toks[k:k+CHUNK], skip_special_tokens=True) for k in range(0, target, CHUNK)]
        out.append({"q": raw[i]["input"], "golds": raw[i]["answers"], "chunks": chunks, "ds": name})
    return out

@torch.no_grad()
def base_gen(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        run_redknot_qwen35_linear, linear_attention_layer_indices)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True).eval()
    # all linear layers in the relay map (global relay -> should == baseline)
    win_by_layer = {li: None for li in linear_attention_layer_indices(model.config)}

    samples = []
    for ds in DATASETS:
        samples += load_nchunk(ds, tok, N)

    W = 80
    print("=" * W)
    print(" VERIFY chunked linear relay (native kernel) == baseline single pass")
    print(f" {N_CHUNK}x{CHUNK} tok | full attention unchanged | global relay all heads")
    print("=" * W)
    bf = rf = 0.0; match = 0
    for s in samples:
        qt = QP.format(q=s["q"])
        bb = short(base_gen(model, tok, "\n\n".join(s["chunks"]) + qt))
        rk, _ = run_redknot_qwen35_linear(
            model, tok, segments=s["chunks"], query_text=qt,
            win_by_layer=win_by_layer, max_new_tokens=MAX_NEW)
        rk = short(rk)
        bF = f1(bb, s["golds"]); rF = f1(rk, s["golds"])
        bf += bF; rf += rF; match += int(bb.strip()[:20] == rk.strip()[:20])
        print(f" {s['ds']:14} base_F1={bF:.3f} relay_F1={rF:.3f} text_match={bb.strip()[:18]!r}=={rk.strip()[:18]!r}")
    k = len(samples)
    print("-" * W)
    print(f" AVG base F1={bf/k:.3f} relay F1={rf/k:.3f} dF1={rf/k-bf/k:+.3f} text_match={match}/{k}")
    print(f" VERDICT: {'PASS (chunked relay faithful)' if abs(rf/k-bf/k)<0.05 else 'FAIL'}")
    print("=" * W)


if __name__ == "__main__":
    main()
