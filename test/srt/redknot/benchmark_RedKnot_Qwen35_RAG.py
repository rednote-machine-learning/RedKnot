#!/usr/bin/env python3
# Copyright 2024-2026 SGLang RedKnot Integration.
"""RedKnot RAG benchmark for Qwen3.5-MoE (hybrid linear/full attention).

End-to-end "build offline ONCE, reuse online" RAG benchmark comparing the
standard full recompute (dense) against RedKnot's offline-snapshot reuse path.

What it does
------------
For each context length (32K / 64K / 128K), it stitches a long document from
fixed-size chunks (4K / 6K / 8K) of a LongBench dataset, builds the RedKnot
offline state ONCE per document (per-segment linear snapshots + full-attn KV;
NOT counted in TTFT), then answers several queries by REUSING that offline state
with zero history recompute (only window+query is run online -> that IS timed).

For every query it reports, side by side:
  * the generated TEXT of dense vs redknot (for eyeball comparison),
  * F1 and EM vs gold,
  * prefill TTFT (dense vs redknot, offline build excluded),
  * speedup and the implied compute (FLOPs) saving from the token reduction.

RedKnot mechanism (all in driver_qwen35.py):
  * linear LOCAL heads: prefix state taken from the offline per-segment snapshot
    at the window boundary (far history dropped; delta-rule makes this lossless),
    only the window is recomputed online;
  * linear GLOBAL heads: full doc state reused;
  * full-attn heads: cached doc KV reused, only window+query KV appended;
  * optional deep-MoE token sparsity (shallow dense / deep sparse);
  * torch.compile of the per-layer static blocks.

One-click run
-------------
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    python test/srt/redknot/benchmark_RedKnot_Qwen35_RAG.py

Everything is configurable via env vars (see CONFIG below); defaults give a full
32K/64K/128K sweep with 4K/6K/8K chunking.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# CONFIG (all overridable via environment variables)
# ──────────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[3]  # .../RedKnotV0.3
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    str(Path(__file__).resolve().parent / "datasets/LongBench/data"),
)
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "hotpotqa,2wikimqa,multifieldqa_en,triviaqa"
).split(",")

# Context lengths (tokens) to sweep and the chunk size used to stitch each.
# CTX_LENS[i] is built from chunks of size CHUNK_TOKENS[i].
CTX_LENS = [
    int(x) for x in os.environ.get("REDKNOT_CTX_LENS", "32000,64000,128000").split(",")
]
CHUNK_TOKENS = [
    int(x) for x in os.environ.get("REDKNOT_CHUNK_TOKENS", "4000,6000,8000").split(",")
]

N_DOCS = int(os.environ.get("REDKNOT_N_DOCS", "1"))  # docs per dataset
N_QUERIES = int(os.environ.get("REDKNOT_N_QUERIES", "4"))  # queries per doc
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))

# RedKnot knobs.
WINDOW_CAP = int(
    os.environ.get("REDKNOT_WINDOW_CAP", "8192")
)  # promote >cap local heads -> global
WINDOW_SEGS = int(
    os.environ.get("REDKNOT_WINDOW_SEGS", "2")
)  # online window = N segments
DECAY_Q = float(os.environ.get("REDKNOT_DECAY_QUANTILE", "0.95"))
SAFETY = float(os.environ.get("REDKNOT_SAFETY", "4.0"))
MINW = int(os.environ.get("REDKNOT_MIN_WINDOW", "512"))
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "1") == "1"
MOE_SPARSE = os.environ.get("REDKNOT_MOE_SPARSE", "0") == "1"
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", "20"))
MOE_MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.2"))

# Baseline dense prefill: single-shot is fair but may OOM at 128K on few GPUs;
# fall back to chunked automatically. BASELINE_PREFILL_CHUNK only used on OOM.
BASELINE_PREFILL_CHUNK = int(os.environ.get("REDKNOT_BASELINE_PREFILL_CHUNK", "16000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")

QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


# ──────────────────────────────────────────────────────────────────────────
# Metrics (LongBench-style F1 / EM)
# ──────────────────────────────────────────────────────────────────────────
def _norm(s):
    s = (s or "").lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, golds):
    best = 0.0
    pn = _norm(pred).split()
    for g in golds:
        gn = _norm(g).split()
        if not pn or not gn:
            best = max(best, float(pn == gn))
            continue
        common = {}
        for w in pn:
            if w in gn:
                common[w] = common.get(w, 0) + 1
        nc = sum(common.values())
        if nc == 0:
            continue
        prec, rec = nc / len(pn), nc / len(gn)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def em_score(pred, golds):
    p = _norm(pred)
    return float(any(_norm(g) == p or _norm(g) in p for g in golds))


def clean_answer(t):
    """Extract the model's short answer for scoring/printing."""
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    return (lines[0] if lines else t).strip().strip('"').strip("'")


def patch_version_check():
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


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    patch_version_check()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
        rag_build_offline_segmented,
        rag_query_reuse_from_snapshots,
    )

    print("=" * 100)
    print(" RedKnot RAG Benchmark — Qwen3.5-MoE  (offline build once, reuse online)")
    print("=" * 100)
    print(f" model       : {MODEL}")
    print(f" datasets    : {DATASETS}")
    print(f" ctx lengths : {CTX_LENS}  (chunk sizes: {CHUNK_TOKENS})")
    print(f" docs/ds={N_DOCS}  queries/doc={N_QUERIES}  max_new={MAX_NEW}")
    print(
        f" window_cap={WINDOW_CAP} window_segs={WINDOW_SEGS} compile={USE_COMPILE} "
        f"moe_sparse={MOE_SPARSE}(deep>={DEEP_MOE_START},thr={MOE_MASS_THRESH})"
    )
    print("=" * 100)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    bm = model.model

    def load_raw(name):
        rows = []
        with open(os.path.join(LONGBENCH_DIR, f"{name}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("input") and r.get("context") and r.get("answers"):
                    rows.append(r)
        random.Random(0).shuffle(rows)
        return rows

    def build_docs(name, ctx, chunk):
        rows = load_raw(name)
        docs, i = [], 0
        while len(docs) < N_DOCS and i < len(rows):
            tk = tok(rows[i]["context"], add_special_tokens=False)["input_ids"]
            qa = [(rows[i]["input"], rows[i]["answers"])]
            j = i + 1
            while (len(tk) < ctx or len(qa) < N_QUERIES) and j < len(rows):
                tk += tok(rows[j]["context"], add_special_tokens=False)["input_ids"]
                if len(qa) < N_QUERIES:
                    qa.append((rows[j]["input"], rows[j]["answers"]))
                j += 1
            if len(tk) < ctx:
                i = j
                continue
            tk = tk[:ctx]
            chunks = [
                tok.decode(tk[k : k + chunk], skip_special_tokens=True)
                for k in range(0, ctx, chunk)
            ]
            docs.append(
                {"ds": name, "chunks": chunks, "qas": qa[:N_QUERIES], "ctx": ctx}
            )
            i = j
        return docs

    @torch.no_grad()
    def dense_generate(full):
        """Standard FULL recompute over docs+query. Returns (text, ttft)."""
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        seqlen = ids.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = DynamicCache(config=model.config)
        pos = 0
        o = None
        # Single-shot full prefill is fairest, but very long contexts (64K/128K)
        # blow up activation memory in one forward (un-catchable OOM inside the
        # accelerate device-map hooks). For seqlen above SINGLE_SHOT_LIMIT we
        # prefill in chunks through the KV cache — this is the SAME dense
        # attention, just memory-bounded; TTFT still times the whole dense prefill.
        SINGLE_SHOT_LIMIT = int(
            os.environ.get("REDKNOT_BASELINE_SINGLE_LIMIT", "48000")
        )
        if seqlen > SINGLE_SHOT_LIMIT:
            for st in range(0, seqlen, BASELINE_PREFILL_CHUNK):
                piece = ids[:, st : st + BASELINE_PREFILL_CHUNK]
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
        else:
            try:  # single-shot full prefill (fair); fall back to chunked on OOM
                pids = torch.arange(0, seqlen, device=ids.device).unsqueeze(0)
                o = model(
                    input_ids=ids,
                    position_ids=pids,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = o.past_key_values
                pos = seqlen
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cache = DynamicCache(config=model.config)
                pos = 0
                for st in range(0, seqlen, BASELINE_PREFILL_CHUNK):
                    piece = ids[:, st : st + BASELINE_PREFILL_CHUNK]
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
        gen = [int(nx[0, 0])]
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
            gen.append(int(nx[0, 0]))
            pos += 1
            if int(nx[0, 0]) == tok.eos_token_id:
                break
        return tok.decode(gen, skip_special_tokens=True), ttft

    @torch.no_grad()
    def window_map(chunks, ctx):
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
            wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
            if WINDOW_CAP > 0:
                wt = torch.where(wt > WINDOW_CAP, torch.zeros_like(wt), wt)
            win[li] = wt
        return win

    def reuse_generate(doc_state, qt):
        return rag_query_reuse_from_snapshots(
            model,
            tok,
            doc_state=doc_state,
            query_text=qt,
            window_segs=WINDOW_SEGS,
            max_new_tokens=MAX_NEW,
            use_compile=USE_COMPILE,
            moe_sparse=MOE_SPARSE,
            deep_moe_start_layer=DEEP_MOE_START,
            moe_mass_thresh=MOE_MASS_THRESH,
        )

    overall = []
    for ctx, chunk in zip(CTX_LENS, CHUNK_TOKENS):
        docs = []
        for d in DATASETS:
            docs += build_docs(d, ctx, chunk)
        n_chunks = ctx // chunk
        print(
            f"\n{'#' * 100}\n# CONTEXT {ctx:,} tokens  =  {n_chunks} chunks x {chunk:,} tokens "
            f"| {len(docs)} docs\n{'#' * 100}"
        )
        # ---- Per-ctx WARMUP: trigger FLA-kernel autotune + torch.compile for
        # this ctx/chunk so the timed queries are steady (no autotune spikes). ----
        if docs:
            _wc = docs[0]["chunks"]
            _win = window_map(_wc, ctx)
            try:
                _wds = rag_build_offline_segmented(
                    model, tok, segments=_wc, win_tok_by_layer=_win, seg=chunk
                )
                for _wq, _ in docs[0]["qas"][: min(2, len(docs[0]["qas"]))]:
                    reuse_generate(_wds, QP.format(q=_wq))
                    dense_generate("\n\n".join(_wc) + QP.format(q=_wq))
                del _wds
            except Exception as e:
                print(f"   [warmup ctx={ctx}] {e}")
        sf = rf = se = re_ = bt = rtt = 0.0
        n = 0
        online_tok_sum = 0
        dense_tok_sum = 0
        for doc in docs:
            chunks = doc["chunks"]
            win = window_map(chunks, ctx)
            # ---- OFFLINE BUILD (ONCE, NOT counted in TTFT) ----
            torch.cuda.synchronize()
            ob = time.perf_counter()
            ds = rag_build_offline_segmented(
                model, tok, segments=chunks, win_tok_by_layer=win, seg=chunk
            )
            torch.cuda.synchronize()
            offline_t = time.perf_counter() - ob
            win_tokens = min(WINDOW_SEGS * chunk, ctx)
            print(
                f"\n[{doc['ds']}] offline build = {offline_t:.1f}s (ONCE, excluded from TTFT)"
                f"  | online window = {win_tokens:,} tok  (dense recomputes {ctx:,} tok)"
            )
            for q, golds in doc["qas"]:
                qt = QP.format(q=q)
                dn, d_ttft = dense_generate("\n\n".join(chunks) + qt)
                rk, r_ttft = reuse_generate(ds, qt)
                da, ra = clean_answer(dn), clean_answer(rk)
                dF, rF = f1_score(da, golds), f1_score(ra, golds)
                dE, rE = em_score(da, golds), em_score(ra, golds)
                sf += dF
                rf += rF
                se += dE
                re_ += rE
                bt += d_ttft
                rtt += r_ttft
                n += 1
                online_tok_sum += win_tokens
                dense_tok_sum += ctx
                print(f"  Q: {q[:80]}")
                print(f"     gold        : {golds[:3]}")
                print(
                    f"     RECOMPUTE   : {da[:46]!r:48} F1={dF:.2f} EM={dE:.0f} ttft={d_ttft:.2f}s"
                )
                print(
                    f"     REDKNOT     : {ra[:46]!r:48} F1={rF:.2f} EM={rE:.0f} ttft={r_ttft:.2f}s"
                    f"  speedup={d_ttft / max(r_ttft, 1e-6):.2f}x"
                )
        k = max(n, 1)
        # compute saving: attention is ~O(L^2); prefilled tokens ratio is the
        # first-order proxy for projection/MLP FLOPs on the prefill.
        tok_ratio = dense_tok_sum / max(online_tok_sum, 1)
        print(f"\n  --- ctx={ctx:,} summary (n={n}) ---")
        print(
            f"   F1   recompute={sf / k:.3f}  redknot={rf / k:.3f}  dF1={rf / k - sf / k:+.3f}"
        )
        print(
            f"   EM   recompute={se / k:.3f}  redknot={re_ / k:.3f}  dEM={re_ / k - se / k:+.3f}"
        )
        print(
            f"   TTFT recompute={bt / k:.2f}s redknot={rtt / k:.2f}s  speedup={(bt / k) / max(rtt / k, 1e-6):.2f}x"
        )
        print(
            f"   online tokens/dense tokens = {online_tok_sum}/{dense_tok_sum} "
            f"=> {tok_ratio:.1f}x fewer prefill tokens (compute saving)"
        )
        overall.append(
            (ctx, chunk, n, sf / k, rf / k, se / k, re_ / k, bt / k, rtt / k, tok_ratio)
        )

    print("\n" + "=" * 100)
    print(" SUMMARY")
    print("=" * 100)
    print(
        f" {'ctx':>8} {'chunk':>6} {'n':>3} {'F1_re':>6} {'F1_rk':>6} {'EM_re':>6} {'EM_rk':>6}"
        f" {'TTFT_re':>8} {'TTFT_rk':>8} {'speedup':>8} {'tok_save':>8}"
    )
    for ctx, chunk, n, f1r, f1k, emr, emk, tr, trk, ratio in overall:
        print(
            f" {ctx:>8,} {chunk:>6,} {n:>3} {f1r:>6.3f} {f1k:>6.3f} {emr:>6.3f} {emk:>6.3f}"
            f" {tr:>7.2f}s {trk:>7.2f}s {tr / max(trk, 1e-6):>7.2f}x {ratio:>7.1f}x"
        )
    print("=" * 100)
    print(" Notes: TTFT excludes the one-time offline build (amortised across all")
    print("        queries of a document). tok_save = dense_prefill_tokens /")
    print("        online_prefill_tokens (first-order compute-saving proxy).")
    print("=" * 100)


if __name__ == "__main__":
    main()
