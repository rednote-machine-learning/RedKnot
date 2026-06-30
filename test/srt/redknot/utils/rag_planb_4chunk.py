#!/usr/bin/env python3
"""4-chunk RAG test for Plan-B chunked RedKnot driver (Qwen3.5-35B-A3B).

Real RAG scenario: each sample's context is the gold document padded with
distractor documents to exactly 4 chunks x 4K = ~16K tokens (ordered). We run:
  * native  : full-recompute over the concatenated 16K (reference quality).
  * planb   : Plan-B chunked driver — chunk 1 is the reusable exact prefix;
              chunks 2..4 forward only their own tokens (linear state relayed +
              accumulated, full attention = global+local head-class with prefix
              KV). FFN/MoE unchanged.
Report F1 / EM for both + TTFT speedup.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/rag_planb_4chunk.py
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
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
N = int(os.environ.get("REDKNOT_N_SAMPLES", "4"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "4"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
FRAC = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.10"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))
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


def load_4chunk(name, tok, n, seed):
    """Build samples whose context is gold doc + distractors, tokenised and cut
    into exactly N_CHUNK chunks of CHUNK tokens (ordered, gold first)."""
    raw = []
    with open(os.path.join(LB, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    target = N_CHUNK * CHUNK
    out = []
    nraw = len(raw)
    for i in range(nraw):
        if len(out) >= n:
            break
        base = raw[i]
        toks = tok(base["context"], add_special_tokens=False)["input_ids"]
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
        out.append({"q": base["input"], "golds": base["answers"], "chunks": chunks})
    return out


@torch.no_grad()
def gen_native(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True), dt


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        build_full_attention_head_config,
        run_redknot_qwen35_chunked,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    head_cfg = build_full_attention_head_config(
        model.config, frac_global=FRAC, local_window=WINDOW
    )

    W = 92
    print("=" * W)
    print(f" 4-CHUNK RAG: Plan-B chunked RedKnot vs native — {Path(MODEL).name}")
    print(f" {N_CHUNK} chunks x {CHUNK} tok | frac_global={FRAC} window={WINDOW} N={N}")
    print("=" * W)
    rows = []
    for ds in DATASETS:
        samples = load_4chunk(ds, tok, N, SEED)
        if not samples:
            print(f" [skip] {ds}: no samples")
            continue
        nf = ne = rf = re_ = 0.0
        nt = rt = 0.0
        for s in samples:
            qt = QP.format(q=s["q"])
            nb, ndt = gen_native(model, tok, "\n\n".join(s["chunks"]) + qt)
            nb = short(nb)
            rk_text, rdt = run_redknot_qwen35_chunked(
                model,
                tok,
                segments=s["chunks"],
                query_text=qt,
                head_cfg=head_cfg,
                max_new_tokens=MAX_NEW,
            )
            rk = short(rk_text)
            nf += f1(nb, s["golds"])
            ne += em(nb, s["golds"])
            rf += f1(rk, s["golds"])
            re_ += em(rk, s["golds"])
            nt += ndt
            rt += rdt
        k = len(samples)
        rows.append((ds, k, nf / k, ne / k, rf / k, re_ / k, nt / k, rt / k))
        print(
            f" {ds:16} N={k} | native F1={nf / k:.3f} EM={ne / k:.3f} TTFT={nt / k:.1f}s "
            f"| RedKnot F1={rf / k:.3f} EM={re_ / k:.3f} TTFT={rt / k:.1f}s "
            f"speedup={nt / max(rt, 1e-3):.2f}x"
        )

    print("\n" + "=" * W)
    print(" SUMMARY (4-chunk RAG)")
    print("=" * W)
    print(
        f" {'dataset':16} {'natF1':>6} {'rkF1':>6} {'dF1':>6} {'natEM':>6} {'rkEM':>6} {'speedup':>8}"
    )
    for ds, k, nf, ne, rf, re_, nt, rt in rows:
        print(
            f" {ds:16} {nf:6.3f} {rf:6.3f} {rf - nf:+6.3f} {ne:6.3f} {re_:6.3f} {nt / max(rt, 1e-3):7.2f}x"
        )
    if rows:
        an = sum(r[2] for r in rows) / len(rows)
        ar = sum(r[4] for r in rows) / len(rows)
        print("-" * W)
        print(f" AVG native F1={an:.3f} | RedKnot F1={ar:.3f} | delta={ar - an:+.3f}")
    print("=" * W)


if __name__ == "__main__":
    main()
