#!/usr/bin/env python3
"""RAG REUSE benchmark: 10 docs x 2K, build doc state ONCE, reuse for queries.

Qwen3.5-35B-A3B. Scenario: 10 document chunks (2K each, 20K total) form a corpus
whose state is built OFFLINE once; multiple queries then REUSE it online. full
attention KV cached natively; linear GLOBAL heads hold full doc state, LOCAL
heads hold only a windowed state (RedKnot). Online cost per query is just the
query tokens — the reuse benefit.

Reports per query: F1/EM vs baseline (full recompute over docs+query) and the
online TTFT of the reuse path (query-only) vs baseline (full prefill).

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/bench_rag_reuse_10x2k.py
"""

from __future__ import annotations

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

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DATASETS = os.environ.get("REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa").split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))  # = #queries (each its own doc set)
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "10"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))
DECAY_Q = float(os.environ.get("REDKNOT_DECAY_QUANTILE", "0.95"))
SAFETY = float(os.environ.get("REDKNOT_SAFETY", "4.0"))
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
        c = Counter(p) & Counter(gg)
        ns = sum(c.values())
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
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return (lines[0] if lines else t).strip().strip('"').strip("'")


def load_nchunk(name, tok, n, seed):
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
        out.append(
            {
                "q": raw[i]["input"],
                "golds": raw[i]["answers"],
                "chunks": chunks,
                "ds": name,
            }
        )
    return out


@torch.no_grad()
def base_gen(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    nxt = out.logits[0, -1, :].argmax().view(1, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
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
    return tok.decode(g, skip_special_tokens=True), ttft


@torch.no_grad()
def build_win(model, tok, sample_text):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
    )

    bm = model.model if hasattr(model, "model") else model
    ids = tok(sample_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
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
    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    decay = measure_linear_head_decay(model, hs_in, decay_quantile=DECAY_Q)
    win = {}
    nloc = ntot = 0
    for li, d in decay.items():
        if li < DENSE_PREFIX:
            win[li] = None
            ntot += len(d)
            continue
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))
        wc = torch.ceil((SAFETY * memlen) / CHUNK).long()
        wc = torch.where(wc >= N_CHUNK, torch.zeros_like(wc), wc)
        win[li] = wc
        nloc += int((wc > 0).sum())
        ntot += len(d)
    return win, nloc / max(ntot, 1)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        rag_build_doc_state,
        rag_query_reuse,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    samples = []
    for ds in DATASETS:
        samples += load_nchunk(ds, tok, N, SEED)

    W = 92
    print("=" * W)
    print(
        f" RAG REUSE: {N_CHUNK} docs x {CHUNK} (build once, reuse per query) — {Path(MODEL).name}"
    )
    print(
        f" full=exact | linear local windowed | safety={SAFETY} dense L0..L{DENSE_PREFIX - 1}"
    )
    print("=" * W)
    bf = be = rf = re_ = 0.0
    bt = rt = 0.0
    flsum = 0.0
    for s in samples:
        qt = QP.format(q=s["q"])
        # baseline: full recompute over docs+query
        bb, bttft = base_gen(model, tok, "\n\n".join(s["chunks"]) + qt)
        bb = short(bb)
        # RAG reuse: build doc state once, then query reuses it
        win, fl = build_win(model, tok, "\n\n".join(s["chunks"]) + qt)
        flsum += fl
        doc = rag_build_doc_state(model, tok, segments=s["chunks"], win_by_layer=win)
        rk, rttft = rag_query_reuse(
            model,
            tok,
            doc_state=doc,
            query_text=qt,
            win_by_layer=win,
            max_new_tokens=MAX_NEW,
        )
        rk = short(rk)
        bF = f1(bb, s["golds"])
        bE = em(bb, s["golds"])
        rF = f1(rk, s["golds"])
        rE = em(rk, s["golds"])
        bf += bF
        be += bE
        rf += rF
        re_ += rE
        bt += bttft
        rt += rttft
        print(
            f" {s['ds']:14} base_F1={bF:.2f} reuse_F1={rF:.2f} | base_TTFT={bttft:.2f}s reuse_TTFT={rttft:.2f}s "
            f"speedup={bttft / max(rttft, 1e-3):.1f}x"
        )
    k = len(samples)
    print("-" * W)
    print(
        f" AVG base F1={bf / k:.3f} reuse F1={rf / k:.3f} dF1={rf / k - bf / k:+.3f} | EM {be / k:.3f}->{re_ / k:.3f}"
    )
    print(
        f" online TTFT: baseline(full prefill)={bt / k:.2f}s  reuse(query-only)={rt / k:.2f}s  speedup={bt / max(rt, 1e-3):.1f}x"
    )
    print(f" linear local-head frac windowed = {flsum / k * 100:.0f}%")
    print("=" * W)
    print(" Note: reuse TTFT excludes the offline doc-state build (amortized across")
    print(" queries). It is the per-query online cost when the corpus is pre-built.")
    print("=" * W)


if __name__ == "__main__":
    main()
