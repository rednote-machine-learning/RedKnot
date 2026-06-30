#!/usr/bin/env python3
"""Final fair benchmark: standard FULL recompute baseline vs approach-2 RedKnot.

Baseline = SINGLE-SHOT full-context prefill (no chunking) for fairness; falls
back to chunked only if it OOMs (and flags it). RedKnot = approach-2 chunked
reuse: local heads use K-nearest-chunk prefix, global heads full state, window
recomputed, with optional window-cap (promote big-window heads to global),
torch.compile, and deep-MoE token sparsity.

Reports per-sample F1 (vs gold) and prefill TTFT for both, across datasets and
context lengths, plus a summary table.
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
DATASETS = os.environ.get("REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa").split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
CTX_LENS = [
    int(x) for x in os.environ.get("REDKNOT_CTX_LENS", "16000,32000").split(",")
]
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
WINDOW_CAP = int(os.environ.get("REDKNOT_WINDOW_CAP", "4096"))
KEEP = int(os.environ.get("REDKNOT_KEEP_CHUNKS", "2"))
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "1") == "1"
MOE_SPARSE = os.environ.get("REDKNOT_MOE_SPARSE", "0") == "1"
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", "20"))
MOE_MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.2"))
BASELINE_SINGLE = os.environ.get("REDKNOT_BASELINE_SINGLE", "1") == "1"
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

    def load(name, ctx):
        raw = []
        with open(os.path.join(LB, f"{name}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("input") and r.get("context") and r.get("answers"):
                    raw.append(r)
        random.Random(0).shuffle(raw)
        out = []
        for i in range(len(raw)):
            if len(out) >= N:
                break
            tk = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
            j = i + 1
            while len(tk) < ctx and j < len(raw):
                tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
                j += 1
            tk = tk[:ctx]
            if len(tk) < ctx:
                continue
            ch = [
                tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
                for k in range(0, ctx, CHUNK)
            ]
            out.append(
                {
                    "q": raw[i]["input"],
                    "golds": raw[i]["answers"],
                    "chunks": ch,
                    "ds": name,
                }
            )
        return out

    @torch.no_grad()
    def dense(full):
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        seqlen = ids.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = DynamicCache(config=model.config)
        pos = 0
        o = None
        step = seqlen if BASELINE_SINGLE else 8000
        try:
            for st in range(0, seqlen, step):
                piece = ids[:, st : st + step]
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
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cache = DynamicCache(config=model.config)
            pos = 0
            for st in range(0, seqlen, 8000):
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
    def win_map(full, ctx):
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
            wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
            if WINDOW_CAP > 0:
                wt = torch.where(wt > WINDOW_CAP, torch.zeros_like(wt), wt)
            win[li] = wt
        return win

    def rk(full, chunks, ctx):
        win = win_map(full, ctx)
        doc = rag_build_offline_chunked(
            model, tok, segments=chunks, win_tok_by_layer=win
        )
        return rag_query_reuse_chunked(
            model,
            tok,
            doc_state=doc,
            query_text=QP.format(q=cur_q),
            keep_chunks=KEEP,
            chunk_tokens=CHUNK,
            max_new_tokens=MAX_NEW,
            use_compile=USE_COMPILE,
            moe_sparse=MOE_SPARSE,
            deep_moe_start_layer=DEEP_MOE_START,
            moe_mass_thresh=MOE_MASS_THRESH,
        )

    global cur_q
    print("=" * 92)
    print(
        f"FINAL  baseline={'SINGLE-shot' if BASELINE_SINGLE else 'chunked'} | "
        f"keep={KEEP} cap={WINDOW_CAP} compile={USE_COMPILE} moe_sparse={MOE_SPARSE}"
    )
    print("=" * 92)
    summary = []
    for ctx in CTX_LENS:
        samples = []
        for d in DATASETS:
            samples += load(d, ctx)
        print(f"\n----- ctx={ctx:,} (n={len(samples)}) -----")
        # Warm up FLA-kernel autotune + compile for THIS ctx on the first few
        # samples (different lengths) so the timed loop is steady (no autotune
        # spikes). bucket-quantized Wmax keeps online lengths to a small set.
        for ws in samples[: min(3, len(samples))]:
            cur_q = ws["q"]
            wfull = "\n\n".join(ws["chunks"]) + QP.format(q=cur_q)
            try:
                rk(wfull, ws["chunks"], ctx)
                dense(wfull)
            except Exception:
                pass
        sf = rf = bt = rtt = 0.0
        n = 0
        for s in samples:
            cur_q = s["q"]
            full = "\n\n".join(s["chunks"]) + QP.format(q=cur_q)
            dn, bttft = dense(full)
            rkt, rttft = rk(full, s["chunks"], ctx)
            sF, rF = f1(short(dn), s["golds"]), f1(short(rkt), s["golds"])
            sf += sF
            rf += rF
            bt += bttft
            rtt += rttft
            n += 1
            print(
                f" [{s['ds']:12}] dense F1={sF:.2f}({bttft:.2f}s) | "
                f"redknot F1={rF:.2f}({rttft:.2f}s) speedup={bttft / max(rttft, 1e-6):.2f}x"
            )
            print(f"     dense='{short(dn)[:40]}'  redknot='{short(rkt)[:40]}'")
        k = max(n, 1)
        print(
            f" @ctx={ctx:,}: dense F1={sf / k:.3f} redknot F1={rf / k:.3f} dF1={rf / k - sf / k:+.3f} | "
            f"TTFT dense={bt / k:.2f}s redknot={rtt / k:.2f}s speedup={(bt / k) / max(rtt / k, 1e-6):.2f}x"
        )
        summary.append((ctx, sf / k, rf / k, bt / k, rtt / k, k))
    print("\n" + "=" * 92)
    print(
        f" {'ctx':>8} {'n':>3} {'F1_dense':>9} {'F1_redknot':>11} {'TTFT_dense':>11} {'TTFT_redknot':>13} {'speedup':>8}"
    )
    for ctx, fd, fr, td, tr, n in summary:
        print(
            f" {ctx:>8,} {n:>3} {fd:>9.3f} {fr:>11.3f} {td:>10.2f}s {tr:>12.2f}s {td / max(tr, 1e-6):>7.2f}x"
        )
    print("=" * 92)


cur_q = ""
if __name__ == "__main__":
    main()
