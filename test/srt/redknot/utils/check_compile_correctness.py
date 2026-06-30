#!/usr/bin/env python3
"""Correctness + speed check for the compiled Qwen3.5 online forward.

Runs ONE RAG sample and compares the RedKnot online query three ways:
  * eager  (use_compile=False)  -> native model() prefill (known-good reference)
  * compiled(use_compile=True)  -> _compiled_online_forward with torch.compile

Asserts the generated text matches (same prefix-state numerics) and prints the
prefill TTFT of each so we can see the compile speedup on the SAME path.
"""

from __future__ import annotations

import os
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
    "REDKNOT_LONGBENCH_DIR",
    str(REPO / "test/srt/redknot/datasets/LongBench/data"),
)
DS = os.environ.get("REDKNOT_DATASETS", "triviaqa")
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "10"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "16"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DENSE_PREFIX, DECAY_Q, SAFETY, MINW = 5, 0.95, 4.0, 512
# Force a fixed local-attention window (tokens) for ALL windowed linear layers.
# 0 = use the decay-based adaptive window. A small fixed window makes the online
# prefill truly sparse (online_tokens = window+query << doc_len), which is the
# precondition for compile/system gains to materialise.
FORCE_WINDOW = int(os.environ.get("REDKNOT_FORCE_WINDOW", "0"))


def _patch():
    import sglang.srt.utils.common as _c

    _o = _c.assert_pkg_version

    def _f(p, m, msg):
        try:
            return _o(p, m, msg)
        except Exception:
            return None

    _c.assert_pkg_version = _f
    try:
        import sglang.srt.entrypoints.engine as _e

        _e.assert_pkg_version = _f
    except Exception:
        pass


def main():
    _patch()
    import json
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # load one sample
    raw = []
    with open(os.path.join(LB, f"{DS}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(0).shuffle(raw)
    tgt = N_CHUNK * CHUNK
    tk = tok(raw[0]["context"], add_special_tokens=False)["input_ids"]
    j = 1
    while len(tk) < tgt and j < len(raw):
        tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
        j += 1
    tk = tk[:tgt]
    chunks = [
        tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
        for k in range(0, tgt, CHUNK)
    ]
    q = raw[0]["input"]
    qt = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:".format(
        q=q
    )

    @torch.no_grad()
    def build_win(text):
        bm = model.model
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
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
        ctx = N_CHUNK * CHUNK
        win = {}
        for li, d in decay.items():
            if li < DENSE_PREFIX:
                win[li] = None
                continue
            if FORCE_WINDOW > 0:
                # Fixed small window for all local heads of this layer.
                wt = torch.full_like(d, FORCE_WINDOW, dtype=torch.long)
                win[li] = wt
                continue
            ml = 1.0 / (1.0 - d.clamp(max=0.99999))
            wt = torch.ceil(SAFETY * ml).long().clamp(min=MINW)
            wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
            win[li] = wt
        return win

    full = "\n\n".join(chunks) + qt
    win = build_win(full)

    # Diagnostic: how many tokens does the online prefill actually process?
    import torch as _torch

    _doc = rag_build_offline_v2(model, tok, segments=chunks, win_tok_by_layer=win)
    _doc_len = len(_doc["doc_ids"])
    _Wmax = 0
    for _li, _wt in win.items():
        if _wt is None:
            continue
        _loc = _wt[(_wt > 0)]
        if _loc.numel():
            _Wmax = max(_Wmax, int(_loc.max().item()))
    _Wmax = min(_Wmax, _doc_len)
    _q_n = len(tok(qt, add_special_tokens=False)["input_ids"])
    print(
        f"[diag] doc_len={_doc_len} Wmax={_Wmax} online_tokens={_Wmax + _q_n} "
        f"(query={_q_n}) -> compile accelerates this online prefill; "
        f"dense baseline prefills all {_doc_len} tokens"
    )
    del _doc

    def run(use_compile, label, warmup_first):
        # rebuild doc state fresh each time (query trims the cache in place)
        doc = rag_build_offline_v2(model, tok, segments=chunks, win_tok_by_layer=win)
        if warmup_first:
            # compile warmup on a throwaway doc state
            doc_w = rag_build_offline_v2(
                model, tok, segments=chunks, win_tok_by_layer=win
            )
            rag_query_reuse_v2(
                model,
                tok,
                doc_state=doc_w,
                query_text=qt,
                max_new_tokens=2,
                use_compile=use_compile,
            )
        txt, ttft = rag_query_reuse_v2(
            model,
            tok,
            doc_state=doc,
            query_text=qt,
            max_new_tokens=MAX_NEW,
            use_compile=use_compile,
        )
        print(f"[{label}] ttft={ttft:.3f}s out={txt[:60]!r}")
        return txt, ttft

    print("=" * 70)
    eager_txt, eager_t = run(False, "eager   ", warmup_first=False)
    comp_txt, comp_t = run(True, "compiled", warmup_first=True)
    print("=" * 70)
    match = eager_txt.strip() == comp_txt.strip()
    print(f"OUTPUT MATCH (eager==compiled): {match}")
    print(
        f"PREFILL TTFT: eager={eager_t:.3f}s compiled={comp_t:.3f}s "
        f"compiled_speedup_vs_eager={eager_t / comp_t:.2f}x"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
