#!/usr/bin/env python3
"""Exactness gate for the approach-2 chunked backend.

GATE 1 (exactness): window=FULL ctx + keep_chunks=0 (full prefix) => the linear
local path degenerates to "recompute everything online from exact prefix", which
must reproduce DENSE output byte-for-byte. If not, the online forward has a bug.

GATE 2 (approx): adaptive window + cap + keep_chunks=2 => F1 vs gold should match
dense (the validated K=2 setting), AND should run on fewer online tokens.
"""

from __future__ import annotations

import os
import re
import string
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR", str(REPO / "test/srt/redknot/datasets/LongBench/data")
)
DS = os.environ.get("REDKNOT_DATASETS", "triviaqa")
CTX = int(os.environ.get("REDKNOT_CTX", "16000"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "16"))
WINDOW_CAP = int(os.environ.get("REDKNOT_WINDOW_CAP", "4096"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DENSE_PREFIX, DECAY_Q, SAFETY, MINW = 5, 0.95, 4.0, 512
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


def _patch():
    import sglang.srt.utils.common as _c

    _o = _c.assert_pkg_version
    _c.assert_pkg_version = lambda *a, **k: (_o(*a, **k) if False else None)


def _n(s):
    s = (s or "").lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(p, gs):
    b = 0.0
    for g in gs:
        a, c = _n(p).split(), _n(g).split()
        if not a or not c:
            b = max(b, float(a == c))
            continue
        common = sum(1 for w in set(a) if w in c)
        if common == 0:
            continue
        pr, rc = common / len(a), common / len(c)
        b = max(b, 2 * pr * rc / (pr + rc))
    return b


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    ls = [x.strip() for x in t.splitlines() if x.strip()]
    return (ls[0] if ls else t).strip().strip('"').strip("'")


def main():
    _patch()
    import json
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
        rag_build_offline_chunked,
        rag_query_reuse_chunked,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    bm = model.model

    raw = []
    with open(os.path.join(LB, f"{DS}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(0).shuffle(raw)
    tk = tok(raw[0]["context"], add_special_tokens=False)["input_ids"]
    j = 1
    while len(tk) < CTX and j < len(raw):
        tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
        j += 1
    tk = tk[:CTX]
    chunks = [
        tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
        for k in range(0, CTX, CHUNK)
    ]
    golds = raw[0]["answers"]
    qt = QP.format(q=raw[0]["input"])
    full = "\n\n".join(chunks) + qt

    @torch.no_grad()
    def dense():
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        cache = DynamicCache(config=model.config)
        pos = 0
        o = None
        for st in range(0, ids.shape[1], 8000):
            piece = ids[:, st : st + 8000]
            pids = torch.arange(pos, pos + piece.shape[1], device=ids.device).unsqueeze(
                0
            )
            o = model(
                input_ids=piece,
                position_ids=pids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = o.past_key_values
            pos += piece.shape[1]
        nx = o.logits[0, -1].argmax().view(1, 1)
        g = [int(nx[0, 0])]
        p = cache
        for _ in range(MAX_NEW - 1):
            pids = torch.tensor([[pos]], device=ids.device)
            og = model(
                input_ids=nx,
                position_ids=pids,
                past_key_values=p,
                use_cache=True,
                logits_to_keep=1,
            )
            p = og.past_key_values
            nx = og.logits[0, -1].argmax().view(1, 1)
            g.append(int(nx[0, 0]))
            pos += 1
        return tok.decode(g, skip_special_tokens=True)

    @torch.no_grad()
    def win_map(cap, force_full=False):
        lin = linear_attention_layer_indices(model.config)
        if force_full:
            win = {}
            for li in lin:
                if li < DENSE_PREFIX:
                    win[li] = None
                else:
                    nh = bm.layers[li].linear_attn.num_v_heads
                    win[li] = torch.full(
                        (nh,), CTX, dtype=torch.long
                    )  # window = full ctx
            return win
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        hs, hh = {}, []
        for li in lin:

            def mk(_li):
                def hook(m, a, k):
                    h = a[0] if a and torch.is_tensor(a[0]) else k.get("hidden_states")
                    if h is not None:
                        hs[_li] = h.detach()

                return hook

            hh.append(
                bm.layers[li].linear_attn.register_forward_pre_hook(
                    mk(li), with_kwargs=True
                )
            )
        model(input_ids=ids, use_cache=False)
        for h in hh:
            h.remove()
        decay = measure_linear_head_decay(model, hs, decay_quantile=DECAY_Q)
        win = {}
        for li, d in decay.items():
            if li < DENSE_PREFIX:
                win[li] = None
                continue
            ml = 1.0 / (1.0 - d.clamp(max=0.99999))
            wt = torch.ceil(SAFETY * ml).long().clamp(min=MINW)
            wt = torch.where(wt >= CTX, torch.zeros_like(wt), wt)
            if cap > 0:
                wt = torch.where(wt > cap, torch.zeros_like(wt), wt)
            win[li] = wt
        return win

    dn = dense()
    print("=" * 80)
    print(f"EXACTNESS GATE  ctx={CTX} cap={WINDOW_CAP} dataset={DS}")
    print("=" * 80)
    print(f"DENSE: {short(dn)[:50]!r}  F1={f1(short(dn), golds):.2f}")

    # GATE 1: window=full + keep_chunks=0 => must equal dense byte-for-byte
    win_full = win_map(0, force_full=True)
    doc1 = rag_build_offline_chunked(
        model, tok, segments=chunks, win_tok_by_layer=win_full
    )
    rk1, _ = rag_query_reuse_chunked(
        model,
        tok,
        doc_state=doc1,
        query_text=qt,
        keep_chunks=0,
        chunk_tokens=CHUNK,
        max_new_tokens=MAX_NEW,
        use_compile=False,
    )
    print(f"[GATE1] window=full keep=ALL  out={short(rk1)[:50]!r}")
    print(f"[GATE1] EXACT MATCH vs dense: {dn.strip() == rk1.strip()}")

    # GATE 2: adaptive window + cap + keep=2 => F1 matches dense, fewer tokens
    win = win_map(WINDOW_CAP)
    Wmax = 0
    for li, wt in win.items():
        if wt is None:
            continue
        loc = wt[(wt > 0)]
        if loc.numel():
            Wmax = max(Wmax, int(loc.max().item()))
    doc2 = rag_build_offline_chunked(model, tok, segments=chunks, win_tok_by_layer=win)
    rk2, ttft2 = rag_query_reuse_chunked(
        model,
        tok,
        doc_state=doc2,
        query_text=qt,
        keep_chunks=2,
        chunk_tokens=CHUNK,
        max_new_tokens=MAX_NEW,
        use_compile=False,
    )
    print(
        f"[GATE2] cap={WINDOW_CAP} keep=2  out={short(rk2)[:50]!r}  "
        f"F1={f1(short(rk2), golds):.2f} Wmax={Wmax} (online~{min(Wmax, CTX)} vs dense {CTX})"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
