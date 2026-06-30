#!/usr/bin/env python3
"""Full RAG benchmark: RedKnot linear (per-head window relay) vs baseline.

Qwen3.5-35B-A3B, 8-chunk x 2K RAG. full attention UNCHANGED. Linear attention:
GLOBAL heads (long memory) carry full state across chunks; LOCAL heads (short
memory) carry state only over their window (measured in chunks) — out-of-window
history is the decayed prefix already folded into the relayed state, dropped at
window boundaries (validated exact for mem_len<=window). Minimal-invasive: native
chunk kernel preserved.

Reports:
  * accuracy (F1/EM) vs baseline   -> should be ~lossless
  * per-layer local/global split   -> theoretical saving
  * theoretical linear compute saving: LOCAL heads in offline RAG reuse need
    only carry a windowed state (O(window) of cross-chunk relay vs O(T)).

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/bench_rag_linear_full.py
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
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "8"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))
DECAY_Q = float(os.environ.get("REDKNOT_DECAY_QUANTILE", "0.95"))
SAFETY = float(os.environ.get("REDKNOT_SAFETY", "2.0"))
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
    out = model.generate(
        ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def build_win_by_layer(model, tok, sample_text):
    """Per-(layer,head) window IN CHUNKS from robust decay memory length."""
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
    win_by_layer = {}
    info = {}
    for li, d in decay.items():
        if li < DENSE_PREFIX:
            win_by_layer[li] = None  # dense: full relay (global)
            info[li] = (0, len(d), 0)
            continue
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))  # tokens
        win_tok = SAFETY * memlen
        win_chunks = torch.ceil(win_tok / CHUNK).long()  # window in chunks
        # global if window covers whole context
        win_chunks = torch.where(
            win_chunks >= N_CHUNK, torch.zeros_like(win_chunks), win_chunks
        )
        win_by_layer[li] = win_chunks
        nloc = int((win_chunks > 0).sum())
        ngl = int((win_chunks == 0).sum())
        mw = int(win_chunks[win_chunks > 0].float().median().item()) if nloc else 0
        info[li] = (nloc, ngl, mw)
    return win_by_layer, info


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        run_redknot_qwen35_linear,
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
    print(f" RAG linear head-window vs baseline — {Path(MODEL).name}")
    print(
        f" {N_CHUNK}x{CHUNK} | full=exact | safety={SAFETY} decayQ={DECAY_Q} dense L0..L{DENSE_PREFIX - 1}"
    )
    print("=" * W)
    bf = be = rf = re_ = 0.0
    nloc = ntot = 0
    last_info = {}
    per_ds = {}
    for s in samples:
        qt = QP.format(q=s["q"])
        full_text = "\n\n".join(s["chunks"]) + qt
        bb = short(base_gen(model, tok, full_text))
        bF = f1(bb, s["golds"])
        bE = em(bb, s["golds"])
        win_by_layer, info = build_win_by_layer(model, tok, full_text)
        last_info = info
        rk, _ = run_redknot_qwen35_linear(
            model,
            tok,
            segments=s["chunks"],
            query_text=qt,
            win_by_layer=win_by_layer,
            max_new_tokens=MAX_NEW,
        )
        rk = short(rk)
        rF = f1(rk, s["golds"])
        rE = em(rk, s["golds"])
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
    print(f" {'dataset':16} {'base F1':>8} {'rk F1':>8} {'dF1':>7}")
    for ds, (b, r, c) in per_ds.items():
        print(f" {ds:16} {b / c:8.3f} {r / c:8.3f} {(r - b) / c:+7.3f}")
    print("-" * W)
    print(
        f" AVG base F1={bf / k:.3f} RK F1={rf / k:.3f} dF1={rf / k - bf / k:+.3f} | EM {be / k:.3f}->{re_ / k:.3f}"
    )
    fl = nloc / max(ntot, 1)
    print(
        f" linear local-head frac (windowed) = {fl * 100:.1f}%  (these heads relay only a windowed state)"
    )
    print("=" * W)
    print(" per-layer (local#, global#, median_win_chunks):")
    for li in sorted(last_info):
        l, gg, mw = last_info[li]
        print(f"   L{li:2d}: local={l:2d} global={gg:2d} med_win_chunks={mw}")
    print("=" * W)


if __name__ == "__main__":
    main()
