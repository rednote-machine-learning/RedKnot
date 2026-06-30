#!/usr/bin/env python3
"""One-click RAG benchmark: OFFLINE-once, ONLINE-reuse (zero history recompute).

Demonstrates the true RedKnot RAG value: build each document's linear snapshots
+ full-attn KV ONCE (offline, NOT timed), then answer MANY queries by reusing it
with zero history recompute (only window+query is run online, which IS timed).

For each document we run several queries and compare, per query:
  * DENSE  : standard full recompute over the whole doc+query (gold + fair TTFT)
  * REUSE  : rag_query_reuse_from_snapshots (offline snapshots reused, only the
             window+query is computed online)

Reports per-query F1 vs gold and ONLINE TTFT (offline build excluded), plus the
amortised offline build time (shown separately, once per document).

Run:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=python \\
  CUDA_VISIBLE_DEVICES=0,1,2,3 \\
  REDKNOT_DATASETS=hotpotqa,2wikimqa REDKNOT_CTX=32000 \\
  .venv_tf5/bin/python test/srt/redknot/utils/bench_offline_reuse.py
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
DATASETS = os.environ.get("REDKNOT_DATASETS", "hotpotqa,2wikimqa,triviaqa").split(",")
CTX = int(os.environ.get("REDKNOT_CTX", "32000"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
SEG = int(os.environ.get("REDKNOT_SEG", "2000"))
N_DOCS = int(os.environ.get("REDKNOT_N_DOCS", "2"))
N_QUERIES = int(os.environ.get("REDKNOT_N_QUERIES", "4"))
WINDOW_SEGS = int(os.environ.get("REDKNOT_WINDOW_SEGS", "4"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "1") == "1"
MOE_SPARSE = os.environ.get("REDKNOT_MOE_SPARSE", "0") == "1"
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", "20"))
MOE_MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.2"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DENSE_PREFIX, DECAY_Q, SAFETY, MINW = 5, 0.95, 4.0, 512
WINDOW_CAP = int(os.environ.get("REDKNOT_WINDOW_CAP", "8192"))
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
        rag_build_offline_segmented,
        rag_query_reuse_from_snapshots,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    bm = model.model

    # Build documents: each doc = CTX tokens of context; we attach several queries
    # (different questions over the SAME context) to exercise offline reuse.
    def build_docs(name):
        raw = []
        with open(os.path.join(LB, f"{name}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("input") and r.get("context") and r.get("answers"):
                    raw.append(r)
        random.Random(0).shuffle(raw)
        docs = []
        i = 0
        while len(docs) < N_DOCS and i < len(raw):
            tk = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
            qa = [(raw[i]["input"], raw[i]["answers"])]
            j = i + 1
            while (len(tk) < CTX or len(qa) < N_QUERIES) and j < len(raw):
                tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
                if len(qa) < N_QUERIES:
                    qa.append((raw[j]["input"], raw[j]["answers"]))
                j += 1
            if len(tk) < CTX:
                i = j
                continue
            tk = tk[:CTX]
            chunks = [
                tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
                for k in range(0, CTX, CHUNK)
            ]
            docs.append({"ds": name, "chunks": chunks, "qas": qa[:N_QUERIES]})
            i = j
        return docs

    docs = []
    for d in DATASETS:
        docs += build_docs(d)

    @torch.no_grad()
    def dense(full):
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = DynamicCache(config=model.config)
        pos = 0
        o = None
        try:
            o = model(
                input_ids=ids,
                position_ids=torch.arange(0, ids.shape[1], device=ids.device).unsqueeze(
                    0
                ),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = o.past_key_values
            pos = ids.shape[1]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cache = DynamicCache(config=model.config)
            pos = 0
            for st in range(0, ids.shape[1], 8000):
                piece = ids[:, st : st + 8000]
                pids = torch.arange(
                    pos, pos + piece.shape[1], device=ids.device
                ).unsqueeze(0)
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
            nx = og.logits[0, -1].argmax().view(1, 1)
            g.append(int(nx[0, 0]))
            pos += 1
        return tok.decode(g, skip_special_tokens=True), ttft

    @torch.no_grad()
    def win_map(chunks):
        full = "\n\n".join(chunks)
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
            if WINDOW_CAP > 0:
                wt = torch.where(wt > WINDOW_CAP, torch.zeros_like(wt), wt)
            win[li] = wt
        return win

    print("=" * 96)
    print(
        f"OFFLINE-ONCE / ONLINE-REUSE BENCH  ctx={CTX} seg={SEG} window_segs={WINDOW_SEGS} "
        f"compile={USE_COMPILE} moe_sparse={MOE_SPARSE}"
    )
    print(f"  {N_DOCS} docs/dataset x {N_QUERIES} queries each | datasets={DATASETS}")
    print("=" * 96)

    sf = rf = bt = rtt = 0.0
    n = 0
    warm = False
    for doc in docs:
        chunks = doc["chunks"]
        win = win_map(chunks)
        # ---- OFFLINE BUILD (ONCE per doc, NOT counted in online TTFT) ----
        torch.cuda.synchronize()
        ob0 = time.perf_counter()
        ds = rag_build_offline_segmented(
            model, tok, segments=chunks, win_tok_by_layer=win, seg=SEG
        )
        torch.cuda.synchronize()
        offline_t = time.perf_counter() - ob0
        print(
            f"\n[{doc['ds']}] OFFLINE build (amortised, NOT in TTFT): {offline_t:.1f}s "
            f"for {N_QUERIES} queries"
        )
        for q, golds in doc["qas"]:
            qt = QP.format(q=q)
            if not warm:
                try:
                    rag_query_reuse_from_snapshots(
                        model,
                        tok,
                        doc_state=ds,
                        query_text=qt,
                        window_segs=WINDOW_SEGS,
                        max_new_tokens=4,
                        use_compile=USE_COMPILE,
                        moe_sparse=MOE_SPARSE,
                        deep_moe_start_layer=DEEP_MOE_START,
                        moe_mass_thresh=MOE_MASS_THRESH,
                    )
                    dense("\n\n".join(chunks) + qt)
                except Exception as e:
                    print("  [warmup err]", e)
                warm = True
            dn, bttft = dense("\n\n".join(chunks) + qt)
            rk, rttft = rag_query_reuse_from_snapshots(
                model,
                tok,
                doc_state=ds,
                query_text=qt,
                window_segs=WINDOW_SEGS,
                max_new_tokens=MAX_NEW,
                use_compile=USE_COMPILE,
                moe_sparse=MOE_SPARSE,
                deep_moe_start_layer=DEEP_MOE_START,
                moe_mass_thresh=MOE_MASS_THRESH,
            )
            sF, rF = f1(short(dn), golds), f1(short(rk), golds)
            sf += sF
            rf += rF
            bt += bttft
            rtt += rttft
            n += 1
            print(f"   Q: {q[:70]}")
            print(f"     gold={golds[:3]}")
            print(f"     dense  F1={sF:.2f} ttft={bttft:.2f}s '{short(dn)[:36]}'")
            print(
                f"     reuse  F1={rF:.2f} ttft={rttft:.2f}s '{short(rk)[:36]}'  speedup={bttft / max(rttft, 1e-6):.2f}x"
            )
    k = max(n, 1)
    print("\n" + "=" * 96)
    print(
        f" TOTAL {n} queries | dense F1={sf / k:.3f} reuse F1={rf / k:.3f} dF1={rf / k - sf / k:+.3f}"
    )
    print(
        f" ONLINE TTFT (offline build excluded): dense={bt / k:.2f}s reuse={rtt / k:.2f}s "
        f"speedup={(bt / k) / max(rtt / k, 1e-6):.2f}x"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
