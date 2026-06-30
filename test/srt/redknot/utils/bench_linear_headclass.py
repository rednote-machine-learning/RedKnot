#!/usr/bin/env python3
"""Real Qwen3.5-35B-A3B: linear head-class windowing vs baseline.

Config (user's design):
  * full attention: EXACT (all computed, unchanged).
  * linear attention: GLOBAL heads (long memory) full-history; LOCAL heads
    (mem_len < window) windowed (state reset every `window` tokens, lossless for
    fast heads). Layers L0..L3 fully dense (no windowing).
  * Decode unchanged (native).

First a SANITY check: window=huge should reproduce baseline (validates the
patched linear forward). Then the real comparison at window W: F1/EM, TTFT,
and the fraction of linear heads windowed (compute proxy).

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/bench_linear_headclass.py
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
import torch.nn.functional as F

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
SAFETY = float(os.environ.get("REDKNOT_SAFETY", "4.0"))  # window = safety * mem_len
WIN_CAP = int(os.environ.get("REDKNOT_WIN_CAP", "8192"))  # >= cap -> global head
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))  # L0..L4
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
        out.append({"q": r["input"], "golds": r["answers"], "ctx_ids": ids})
    return out


@torch.no_grad()
def gen(model, tok, text):
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
def capture_decay(model, tok, ids_tensor):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
    )

    bm = model.model if hasattr(model, "model") else model
    cap = {}
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
    model(input_ids=ids_tensor, use_cache=False)
    for h in handles:
        h.remove()
    return measure_linear_head_decay(model, hs_in)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        install_linear_headclass,
        linear_attention_layer_indices,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    samples = []
    for ds in DATASETS:
        for s in load_ds(ds, tok, N, SEED):
            s["ds"] = ds
            samples.append(s)

    W = 96
    print("=" * W)
    print(f" PER-(LAYER,HEAD) LINEAR WINDOWING vs baseline — {Path(MODEL).name}")
    print(
        f" full=exact | per-head window=safety({SAFETY})*mem_len (cap {WIN_CAP}=global) | dense L0..L{DENSE_PREFIX - 1}"
    )
    print("=" * W)

    print(" NOTE: TTFT not reported (the windowed linear recurrence here is an")
    print(" UNoptimized per-token loop; speed needs a chunked kernel). We report")
    print(" ACCURACY (valid regardless of impl speed) + THEORETICAL saving.\n")

    bf = be = 0.0
    rf = re_ = 0.0
    n_local_total = n_head_total = 0
    per_ds = {}
    for s in samples:
        ctx = tok.decode(s["ctx_ids"], skip_special_tokens=True)
        text = ctx + QP.format(q=s["q"])
        bb, _ = gen(model, tok, text)
        bb = short(bb)
        bF = f1(bb, s["golds"])
        bE = em(bb, s["golds"])
        ids_t = tok(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        decay = capture_decay(model, tok, ids_t)
        restore, info = install_linear_headclass(
            model,
            decay,
            dense_prefix_layers=DENSE_PREFIX,
            safety=SAFETY,
            win_cap=WIN_CAP,
        )
        try:
            rb, _ = gen(model, tok, text)
            rb = short(rb)
        finally:
            restore()
        rF = f1(rb, s["golds"])
        rE = em(rb, s["golds"])
        bf += bF
        be += bE
        rf += rF
        re_ += rE
        nloc = sum(v[0] for v in info.values())
        ntot = sum(len(decay[li]) for li in info)
        n_local_total += nloc
        n_head_total += ntot
        last_info = info
        d = per_ds.setdefault(s["ds"], [0.0, 0.0, 0])
        d[0] += bF
        d[1] += rF
        d[2] += 1

    k = len(samples)
    fl = n_local_total / max(n_head_total, 1)
    print(f" {'dataset':16} {'base F1':>8} {'hc F1':>8} {'dF1':>7}")
    for ds, (b, r, c) in per_ds.items():
        print(f" {ds:16} {b / c:8.3f} {r / c:8.3f} {(r - b) / c:+7.3f}")
    print("-" * W)
    print(
        f" AVG  base F1={bf / k:.3f}  headclass F1={rf / k:.3f}  dF1={rf / k - bf / k:+.3f}  EM {be / k:.3f}->{re_ / k:.3f}"
    )
    print("=" * W)
    # Theoretical saving: of the windowed (non-dense) linear layers, what frac of
    # heads become LOCAL. In chunked RAG, local heads need NOT carry/relay cross-
    # chunk state (window << chunk), so their cross-chunk state work is saved;
    # global heads still relay. Per-token state-update FLOPs are ~equal, but the
    # cross-chunk reuse/relay cost scales with the GLOBAL-head fraction only.
    print(f" THEORETICAL: linear local-head frac (windowed) = {fl * 100:.1f}%")
    print(
        f"   -> cross-chunk linear state relay needed for only {(1 - fl) * 100:.1f}% of heads"
    )
    print(
        f"   -> per-(layer,head) window=safety*mem_len, dense prefix L0..L{DENSE_PREFIX - 1}"
    )
    # per-layer window distribution (from last sample)
    print(" per-layer (local#, global#, median_window):")
    for li in sorted(last_info):
        n_loc, n_glob, mw = last_info[li]
        print(f"   L{li:2d}: local={n_loc:3d} global={n_glob:3d} med_win={mw}")
    print("=" * W)


if __name__ == "__main__":
    main()
