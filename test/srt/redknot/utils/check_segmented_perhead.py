#!/usr/bin/env python3
"""Per-head segmented offline reuse: correctness + speed.

Compares on ONE RAG sample at a chosen context length:
  * DENSE full recompute (reference for accuracy)
  * rag_query_reuse_v2          (unified bmin prefix)
  * rag_query_reuse_segmented   (per-head segmented snapshots)

Prints output text (eyeball), F1 vs gold, and prefill TTFT for each, plus the
online-token count each path actually processes.
"""

from __future__ import annotations

import os
import re
import string
import sys
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
SEG = int(os.environ.get("REDKNOT_SEG", "2048"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "16"))
FORCE_WINDOW = int(os.environ.get("REDKNOT_FORCE_WINDOW", "0"))
# Promote any local head whose adaptive window exceeds WINDOW_CAP to a GLOBAL head
# (window=0 -> reuses offline S_T, zero online cost). This bounds Wmax so the
# shared online sequence stays short -> per-head segmented reuse can pay off.
WINDOW_CAP = int(os.environ.get("REDKNOT_WINDOW_CAP", "0"))
ADAPTIVE = os.environ.get("REDKNOT_ADAPTIVE", "1") == "1"
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "0") == "1"
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DENSE_PREFIX, DECAY_Q, SAFETY, MINW = 5, 0.95, 4.0, 512
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


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
        common = {}
        for w in a:
            if w in c:
                common[w] = 1
        ncommon = sum(common.values())
        if ncommon == 0:
            continue
        pr, rc = ncommon / len(a), ncommon / len(c)
        b = max(b, 2 * pr * rc / (pr + rc))
    return b


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    ls = [x.strip() for x in t.splitlines() if x.strip()]
    return (ls[0] if ls else t).strip().strip('"').strip("'")


def _clean(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    return " ".join(t.split())[:160]


def main():
    _patch()
    import json
    import random
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
        rag_build_offline_segmented,
        rag_build_offline_v2,
        rag_query_reuse_segmented,
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
    q = raw[0]["input"]
    golds = raw[0]["answers"]
    qt = QP.format(q=q)
    full = "\n\n".join(chunks) + qt

    @torch.no_grad()
    def build_win():
        bm = model.model
        win = {}
        if FORCE_WINDOW > 0:
            for li in linear_attention_layer_indices(model.config):
                if li < DENSE_PREFIX:
                    win[li] = None
                    continue
                nh = bm.layers[li].linear_attn.num_v_heads
                win[li] = torch.full((nh,), FORCE_WINDOW, dtype=torch.long)
            return win
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
        for li, d in decay.items():
            if li < DENSE_PREFIX:
                win[li] = None
                continue
            ml = 1.0 / (1.0 - d.clamp(max=0.99999))
            wt = torch.ceil(SAFETY * ml).long().clamp(min=MINW)
            wt = torch.where(wt >= CTX, torch.zeros_like(wt), wt)
            if WINDOW_CAP > 0:
                # Promote oversized local heads to GLOBAL (window=0 -> reuse S_T).
                wt = torch.where(wt > WINDOW_CAP, torch.zeros_like(wt), wt)
            win[li] = wt
        return win

    @torch.no_grad()
    def std_full():
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        from transformers import DynamicCache

        torch.cuda.synchronize()
        t0 = time.perf_counter()
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
        nx = o.logits[0, -1, :].argmax().view(1, 1)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        p = cache
        g = [int(nx[0, 0])]
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
        return tok.decode(g, skip_special_tokens=True), ttft

    win = build_win()
    # window stats
    allw = []
    for li, wt in win.items():
        if wt is None:
            continue
        for x in wt.tolist():
            allw.append(x)
    import statistics as st_

    loc = [w for w in allw if w > 0]
    print("=" * 80)
    print(
        f"ctx={CTX} seg={SEG} adaptive={ADAPTIVE and FORCE_WINDOW == 0} "
        f"force_window={FORCE_WINDOW} compile={USE_COMPILE}"
    )
    if loc:
        print(
            f"local-head windows: min={min(loc)} max={max(loc)} "
            f"median={int(st_.median(loc))} n_local={len(loc)} "
            f"n_global={len(allw) - len(loc)}"
        )
    print("=" * 80)

    # warmup (compile + autotune)
    d2 = rag_build_offline_v2(model, tok, segments=chunks, win_tok_by_layer=win)
    rag_query_reuse_v2(
        model,
        tok,
        doc_state=d2,
        query_text=qt,
        max_new_tokens=2,
        use_compile=USE_COMPILE,
    )
    dsg = rag_build_offline_segmented(
        model, tok, segments=chunks, win_tok_by_layer=win, seg=SEG
    )
    rag_query_reuse_segmented(
        model,
        tok,
        doc_state=dsg,
        query_text=qt,
        max_new_tokens=2,
        use_compile=USE_COMPILE,
    )

    # measured runs (fresh doc states; query trims cache in place)
    sb, st_ttft = std_full()
    d2 = rag_build_offline_v2(model, tok, segments=chunks, win_tok_by_layer=win)
    v2, v2_ttft = rag_query_reuse_v2(
        model,
        tok,
        doc_state=d2,
        query_text=qt,
        max_new_tokens=MAX_NEW,
        use_compile=USE_COMPILE,
    )
    dsg = rag_build_offline_segmented(
        model, tok, segments=chunks, win_tok_by_layer=win, seg=SEG
    )
    sg, sg_ttft = rag_query_reuse_segmented(
        model,
        tok,
        doc_state=dsg,
        query_text=qt,
        max_new_tokens=MAX_NEW,
        use_compile=USE_COMPILE,
    )

    print(f"Q: {q[:100]}")
    print(f"gold       : {golds[:4]}")
    print(
        f"DENSE      : {_clean(sb)!r}  F1={f1(short(sb), golds):.2f} ttft={st_ttft:.2f}s"
    )
    print(
        f"v2(bmin)   : {_clean(v2)!r}  F1={f1(short(v2), golds):.2f} ttft={v2_ttft:.2f}s "
        f"speedup_vs_dense={st_ttft / v2_ttft:.2f}x"
    )
    print(
        f"segmented  : {_clean(sg)!r}  F1={f1(short(sg), golds):.2f} ttft={sg_ttft:.2f}s "
        f"speedup_vs_dense={st_ttft / sg_ttft:.2f}x"
    )
    print("=" * 80)
    print(f"text match v2==segmented: {short(v2).strip() == short(sg).strip()}")


if __name__ == "__main__":
    main()
