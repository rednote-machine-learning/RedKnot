#!/usr/bin/env python3
"""8-chunk x 2K RAG: window=256 (global recompute + local sink+window) + linear
recompute (Plan-1), sweep frac_global. Qwen3.5-35B-A3B vs standard.

User design for small-chunk (2K) scenario:
  * full attention: global heads RECOMPUTE over full [prefix|chunk]; local heads
    keep only sink + window, window=256 (aggressive real sparsity).
  * linear attention: RECOMPUTE chunk 2,3,... (Plan-1, sees true sparse stream,
    no offline-state coupling loss).
This is the classic RedKnot regime (small window). window=256 saves much more
than 4096 (58-87% at 16K). We sweep frac_global to find the accuracy/compute
sweet spot here.

Metrics per frac_global: F1/EM vs standard, online TTFT speedup, analytic
full-attn FLOPs saving.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/bench_w256_8chunk.py
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
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "256"))
FRACS = [float(x) for x in os.environ.get("REDKNOT_FRACS", "0.10,0.25,0.40").split(",")]
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


def full_attn_saving(frac, window, T):
    dense = T * (T + 1) / 2.0
    return 1.0 - (frac * dense + (1 - frac) * T * min(window, T)) / dense


@torch.no_grad()
def std_infer(model, tok, text):
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
        run_redknot_qwen35_chunked,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    T = N_CHUNK * CHUNK

    # precompute standard once per sample
    allsamp = {}
    for ds in DATASETS:
        allsamp[ds] = load_nchunk(ds, tok, N, SEED)

    W = 96
    print("=" * W)
    print(
        f" 8-CHUNK x 2K, window={WINDOW} (global recompute + local sink+window), linear RECOMPUTE"
    )
    print(
        f" {Path(MODEL).name} | {N_CHUNK}x{CHUNK}={T} tok | sweep frac_global={FRACS} | N={N}"
    )
    print("=" * W)

    # standard baseline (compute once)
    std = {}
    for ds, samples in allsamp.items():
        sf = se = st = 0.0
        for s in samples:
            qt = QP.format(q=s["q"])
            sb, sttft = std_infer(model, tok, "\n\n".join(s["chunks"]) + qt)
            sb = short(sb)
            sf += f1(sb, s["golds"])
            se += em(sb, s["golds"])
            st += sttft
        k = len(samples)
        std[ds] = (sf / k, se / k, st / k)
        print(f" [std] {ds:16} F1={sf / k:.3f} EM={se / k:.3f} TTFT={st / k:.2f}s")

    for fr in FRACS:
        head_cfg = build_full_attention_head_config(
            model.config, frac_global=fr, local_window=WINDOW
        )
        save = full_attn_saving(fr, WINDOW, T)
        print("-" * W)
        print(f" frac_global={fr:.2f}  (full-attn FLOPs save~{save * 100:.0f}%)")
        tot_d = tot_sp = 0.0
        cnt = 0
        for ds, samples in allsamp.items():
            rf = rt = 0.0
            for s in samples:
                qt = QP.format(q=s["q"])
                rk_text, rttft = run_redknot_qwen35_chunked(
                    model,
                    tok,
                    segments=s["chunks"],
                    query_text=qt,
                    head_cfg=head_cfg,
                    max_new_tokens=MAX_NEW,
                )
                rf += f1(short(rk_text), s["golds"])
                rt += rttft
            k = len(samples)
            sF, sE, sT = std[ds]
            d = rf / k - sF
            sp = sT / max(rt / k, 1e-3)
            tot_d += d
            tot_sp += sp
            cnt += 1
            print(
                f"   {ds:16} std_F1={sF:.3f} RK_F1={rf / k:.3f} dF1={d:+.3f} speedup={sp:.2f}x"
            )
        print(
            f"   AVG dF1={tot_d / cnt:+.3f} speedup={tot_sp / cnt:.2f}x save~{save * 100:.0f}%"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
