#!/usr/bin/env python3
"""8x2K RAG: FULL attention exact (recompute all) + linear chunk1 reuse,
chunk2/3 recompute, chunk4-8 reuse. Qwen3.5-35B-A3B vs standard.

User config (1-indexed chunks):
  * full attention: ALL recomputed (exact, no head-class sparsity).
  * linear attention: chunk1 reuse offline (lossless prefix), chunk2 & chunk3
    RECOMPUTE (near-context fidelity), chunk4..8 reuse offline (accumulate).
Since full is exact, the ONLY accuracy cost is the chunk4-8 linear reuse
mismatch — a clean isolation. Reports F1/EM + TTFT vs standard.

0-based mapping: recompute_chunks={1,2} -> chunk index1,2 == 2nd,3rd chunk.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/bench_fullexact_midrecompute.py
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
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "8"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
# recompute chunk index 1 and 2 (0-based) == 2nd and 3rd chunk
RECOMP = set(
    int(x) for x in os.environ.get("REDKNOT_RECOMPUTE_CHUNKS", "1,2").split(",")
)
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
        out.append({"q": raw[i]["input"], "golds": raw[i]["answers"], "chunks": chunks})
    return out


@torch.no_grad()
def std_infer(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    nxt = out.logits[0, -1, :].argmax().view(1, 1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    past = out.past_key_values
    gen = [int(nxt[0, 0])]
    for _ in range(MAX_NEW - 1):
        og = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = og.past_key_values
        nxt = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        gen.append(tid)
        if tid == tok.eos_token_id:
            break
    return tok.decode(gen, skip_special_tokens=True), ttft


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        build_full_attention_head_config,
        run_redknot_qwen35_planb2,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    head_cfg = build_full_attention_head_config(
        model.config, frac_global=1.0, local_window=4096
    )
    T = N_CHUNK * CHUNK
    W = 96
    print("=" * W)
    print(
        f" 8x2K RAG: FULL EXACT (all recompute) + linear recompute chunks {sorted(RECOMP)} (0-based)"
    )
    print(
        f" {Path(MODEL).name} | {N_CHUNK}x{CHUNK}={T} tok | chunk1 reuse, chunk2/3 recompute, rest reuse"
    )
    print("=" * W)
    rows = []
    for ds in DATASETS:
        samples = load_nchunk(ds, tok, N, SEED)
        if not samples:
            print(f" [skip] {ds}")
            continue
        sf = se = rf = re_ = st = rt = 0.0
        for s in samples:
            qt = QP.format(q=s["q"])
            sb, sttft = std_infer(model, tok, "\n\n".join(s["chunks"]) + qt)
            sb = short(sb)
            rk_text, rttft = run_redknot_qwen35_planb2(
                model,
                tok,
                segments=s["chunks"],
                query_text=qt,
                head_cfg=head_cfg,
                max_new_tokens=MAX_NEW,
                recompute_chunks=RECOMP,
                full_exact=True,
            )
            rk = short(rk_text)
            sf += f1(sb, s["golds"])
            se += em(sb, s["golds"])
            rf += f1(rk, s["golds"])
            re_ += em(rk, s["golds"])
            st += sttft
            rt += rttft
        k = len(samples)
        rows.append((ds, sf / k, se / k, rf / k, re_ / k, st / k, rt / k))
        print(
            f" {ds:16} std F1={sf / k:.3f} EM={se / k:.3f} TTFT={st / k:.2f}s "
            f"| RK F1={rf / k:.3f} EM={re_ / k:.3f} TTFT={rt / k:.2f}s speedup={st / max(rt, 1e-3):.2f}x"
        )
    print("\n" + "=" * W)
    print(" SUMMARY")
    print("=" * W)
    print(f" {'dataset':16} {'stdF1':>6} {'rkF1':>6} {'dF1':>6} {'speedup':>8}")
    for ds, sf, se, rf, re_, st, rt in rows:
        print(
            f" {ds:16} {sf:6.3f} {rf:6.3f} {rf - sf:+6.3f} {st / max(rt, 1e-3):7.2f}x"
        )
    if rows:
        asf = sum(r[1] for r in rows) / len(rows)
        arf = sum(r[3] for r in rows) / len(rows)
        ast = sum(r[5] for r in rows) / len(rows)
        art = sum(r[6] for r in rows) / len(rows)
        print("-" * W)
        print(
            f" AVG std F1={asf:.3f} RK F1={arf:.3f} (d={arf - asf:+.3f}) | TTFT {ast:.2f}->{art:.2f}s ({ast / max(art, 1e-3):.2f}x)"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
