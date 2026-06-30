#!/usr/bin/env python3
"""Data-integrity verification for the RedKnot Qwen3.5 reuse path.

Checks, on ONE real RAG sample at a chosen ctx, that the reported speedups are
NOT artifacts:

  CHECK A (baseline fairness): dense baseline TTFT measured two ways —
    (1) chunked prefill (what the benchmark uses), (2) single-shot full prefill.
    If chunked is much slower, the speedup would be inflated. They should match.

  CHECK B (reuse correctness): run rag_query_reuse_v2 with window = FULL CONTEXT
    (i.e. NO offline reuse — every token recomputed online). Its output must
    match the dense baseline byte-for-byte. This proves the online forward path
    itself is numerically exact (no silent corruption / no fake output).

  CHECK C (reuse is real): the normal (capped-window) redknot processes far
    FEWER online tokens than dense — print both counts. The speedup must be
    explained by this token reduction, not by skipped/empty compute.

  CHECK D (output sanity): print raw outputs; redknot output must be non-empty
    and not a copy of the baseline string object.
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
CTX = int(os.environ.get("REDKNOT_CTX", "32000"))
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


def main():
    _patch()
    import json
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
        rag_build_offline_v2,
        rag_query_reuse_v2,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

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
    qt = QP.format(q=raw[0]["input"])
    full = "\n\n".join(chunks) + qt

    @torch.no_grad()
    def dense(chunked):
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = DynamicCache(config=model.config)
        pos = 0
        step = 8000 if chunked else ids.shape[1]
        o = None
        for st in range(0, ids.shape[1], step):
            piece = ids[:, st : st + step]
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
        nx = o.logits[0, -1, :].argmax().view(1, 1)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
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
            nx = og.logits[0, -1, :].argmax().view(1, 1)
            g.append(int(nx[0, 0]))
            pos += 1
        return tok.decode(g, skip_special_tokens=True), ttft, ids.shape[1]

    @torch.no_grad()
    def win_map(cap):
        bm = model.model
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        hs, hh = {}, []
        for li in linear_attention_layer_indices(model.config):

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

    def online_tokens(win):
        Wmax = 0
        for li, wt in win.items():
            if wt is None:
                continue
            loc = wt[(wt > 0)]
            if loc.numel():
                Wmax = max(Wmax, int(loc.max().item()))
        Wmax = min(Wmax, CTX)
        return Wmax, CTX - Wmax

    print("=" * 80)
    print(f"DATA-INTEGRITY CHECK  ctx={CTX} dataset={DS} cap={WINDOW_CAP}")
    print("=" * 80)

    # CHECK A: baseline fairness (chunked vs single-shot)
    txt_c, ttft_c, ntok = dense(chunked=True)
    try:
        txt_s, ttft_s, _ = dense(chunked=False)
        print(
            f"[A] baseline chunked ttft={ttft_c:.2f}s  single-shot ttft={ttft_s:.2f}s  "
            f"ratio={ttft_c / ttft_s:.2f} (≈1.0 => chunking is fair)"
        )
        print(
            f"    chunked_out=={txt_s and (txt_c.strip() == txt_s.strip())} (same output)"
        )
    except Exception as e:
        print(
            f"[A] single-shot OOM/err ({type(e).__name__}); using chunked baseline only"
        )

    # CHECK B: reuse correctness — window=FULL => no reuse => must equal dense
    win_full = {}
    for li in linear_attention_layer_indices(model.config):
        if li < DENSE_PREFIX:
            win_full[li] = None
        else:
            nh = model.model.layers[li].linear_attn.num_v_heads
            win_full[li] = torch.full((nh,), CTX, dtype=torch.long)  # window=full ctx
    Wm_full, reuse_full = online_tokens(win_full)
    doc_full = rag_build_offline_v2(
        model, tok, segments=chunks, win_tok_by_layer=win_full
    )
    rk_full, _ = rag_query_reuse_v2(
        model,
        tok,
        doc_state=doc_full,
        query_text=qt,
        max_new_tokens=MAX_NEW,
        use_compile=False,
    )
    print(
        f"[B] reuse-correctness: window=FULL (no reuse) Wmax={Wm_full} "
        f"reused_tokens={reuse_full}"
    )
    print(f"    dense_out  : {txt_c.strip()[:50]!r}")
    print(f"    redknotFULL: {rk_full.strip()[:50]!r}")
    print(f"    EXACT MATCH (dense==redknot_full): {txt_c.strip() == rk_full.strip()}")

    # CHECK C/D: normal capped redknot — fewer online tokens, real output
    win = win_map(WINDOW_CAP)
    Wmax, reused = online_tokens(win)
    doc = rag_build_offline_v2(model, tok, segments=chunks, win_tok_by_layer=win)
    rk, rttft = rag_query_reuse_v2(
        model,
        tok,
        doc_state=doc,
        query_text=qt,
        max_new_tokens=MAX_NEW,
        use_compile=False,
    )
    print(
        f"[C] capped redknot: online_tokens(Wmax)={Wmax} (+query) "
        f"vs dense_tokens={ntok}  -> {ntok / max(Wmax, 1):.1f}x fewer online tokens"
    )
    print(
        f"[D] redknot_out: {rk.strip()[:60]!r}  (non-empty={len(rk.strip()) > 0}, "
        f"distinct_from_dense={rk.strip() != txt_c.strip()})"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
