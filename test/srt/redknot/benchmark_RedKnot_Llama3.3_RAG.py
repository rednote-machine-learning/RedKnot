#!/usr/bin/env python3
"""Benchmark: RedKnot vs standard FA-2 prefill — Llama-3.3-70B, multi-dataset RAG.

Companion to ``benchmark_RedKnot_Qwen.py`` but for **Llama-3.3-70B-Instruct**
(INT4) over **multiple LongBench long-document QA datasets** in a RAG setting.

For each dataset we treat the LongBench ``context`` as the retrieved corpus:
it is split into fixed-size chunks (segments) that RedKnot prefills offline once
and reuses, while the ``input`` is the question. We compare against the standard
fastest dense prefill (one FA-2 forward over the full context).

Reported per dataset (and aggregate): generated answer text, SQuAD F1/EM (the
standard LongBench QA metric uses F1), TTFT + speedup, decode tok/s, and the
analytic prefill FLOPs comparison (attention / FFN / projection + savings).

Why Llama-3.3-70B: native 128K context (rope llama3 scaling), so 16K–64K are all
WITHIN the native range — answer quality is meaningful at every length tested.

Measurement caveats (printed at runtime):
  * RedKnot TTFT excludes the offline prefill (RAG cross-request reuse: that
    cost is amortized across requests); baseline TTFT is full prefill.
  * FLOPs are analytic (no gather/scatter/RoPE-reposition overhead) -> the
    COMPUTE axis, distinct from wall-time.
  * F1/EM use the standard SQuAD normalization; a light answer-span extractor
    is applied identically to both methods (fair).

Usage:
  REDKNOT_COMPILE=1 REDKNOT_DATASETS=2wikimqa,musique,multifieldqa_en \\
    REDKNOT_N_SAMPLES=3 CUDA_VISIBLE_DEVICES=0 \\
    python test/srt/redknot/benchmark_RedKnot_Llama.py
"""

from __future__ import annotations

import gc
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.layers.attention.redknot import (  # noqa: E402
    HeadClassConfig,
    SparseFFNSchedule,
    offline_prefill_segments,
    run_redknot_offlinekv,
)

MODEL_PATH = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/096/models/Llama-3.3-70B-Instruct",
)
HEAD_CFG_JSON = os.environ.get(
    "REDKNOT_HEAD_CFG",
    # Sweet-spot config from the 32K sweep: frac_global=0.10 + local_window=4096
    # paired with aggressive Sparse-FFN (mass 0.2/0.05). 1.83x TTFT, 57% FLOPs
    # saving at F1 within 0.011 of full-recompute baseline.
    str(
        Path(__file__).resolve().parent
        / "head_class/llama-70B_sweetspot_g10_w4096.json"
    ),
)
FFN_CFG_JSON = os.environ.get(
    "REDKNOT_FFN_CFG",
    str(Path(__file__).resolve().parent / "sparse_ffn_params/llama-70B.json"),
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DATASETS = os.environ.get("REDKNOT_DATASETS", "2wikimqa,musique,multifieldqa_en").split(
    ","
)
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
MAX_NEW_TOKENS = int(os.environ.get("REDKNOT_MAX_NEW", "48"))
CHUNK_TOKENS = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
# Adaptive local-window sweet spot: window = ctx * WINDOW_RATIO (Llama: 0.5).
# Set to 0 to use the config's fixed window.
_wr = float(os.environ.get("REDKNOT_WINDOW_RATIO", "0.5"))
WINDOW_RATIO = _wr if _wr > 0 else None
MAX_CTX_TOKENS = int(os.environ.get("REDKNOT_MAX_CTX", "40000"))  # cap context
WINDOW = 256 + 4


# ── Standard SQuAD F1 / EM ──
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_max(pred: str, golds) -> float:
    return max((f1_score(pred, g) for g in golds), default=0.0)


def em_max(pred: str, golds) -> float:
    return max((float(_normalize(pred) == _normalize(g)) for g in golds), default=0.0)


# ── LongBench loading + RAG chunking ──
def _chunk_context(context, tok, chunk_tokens, max_ctx):
    ids = tok(context, add_special_tokens=False)["input_ids"][:max_ctx]
    chunks = []
    for i in range(0, len(ids), chunk_tokens):
        piece = ids[i : i + chunk_tokens]
        if len(piece) < 64:
            break
        chunks.append(tok.decode(piece, skip_special_tokens=True))
    return chunks or [tok.decode(ids, skip_special_tokens=True)]


def _load_longbench(ds_name, tok, n_samples):
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            q = r.get("input", "")
            ctx = r.get("context", "")
            golds = r.get("answers", [])
            if not q or not ctx or not golds:
                continue
            docs = _chunk_context(ctx, tok, CHUNK_TOKENS, MAX_CTX_TOKENS)
            if len(docs) < 2:
                continue  # need >=2 segments for online reuse path
            out.append({"question": q, "golds": golds, "docs": docs})
            if len(out) >= n_samples:
                break
    return out


def _load_longbench_padded(ds_name, tok, n_samples, target_tokens):
    """RAG long-context: the gold question's context goes FIRST, then other
    samples' contexts are appended as distractor documents until the total
    reaches ``target_tokens``. Mirrors real RAG retrieval where the relevant
    passage is mixed into a long pile of retrieved documents. Each segment is
    one chunk of ``CHUNK_TOKENS`` tokens.
    """
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    raw = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    out = []
    n = len(raw)
    # Optional offset so each process can pick a different sample window
    # (lets us run N independent 1-sample processes to avoid cross-sample
    # KV-cache memory accumulation at long context).
    off = int(os.environ.get("REDKNOT_SAMPLE_OFFSET", "0"))
    for i in range(off, n):
        if len(out) >= n_samples:
            break
        base = raw[i]
        q, golds = base["input"], base["answers"]
        # gold context first, then round-robin other samples as distractors
        ctx_tokens = tok(base["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % n
        while len(ctx_tokens) < target_tokens and j != i:
            extra = tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            ctx_tokens = ctx_tokens + extra
            j = (j + 1) % n
        ctx_tokens = ctx_tokens[:target_tokens]
        # split into CHUNK_TOKENS segments
        docs = []
        for k in range(0, len(ctx_tokens), CHUNK_TOKENS):
            piece = ctx_tokens[k : k + CHUNK_TOKENS]
            if len(piece) < 64:
                break
            docs.append(tok.decode(piece, skip_special_tokens=True))
        if len(docs) < 2:
            continue
        out.append({"question": q, "golds": golds, "docs": docs})
    return out


def _query_text(q):
    return (
        "\n\nAnswer the question based only on the documents above. "
        "Give the shortest exact answer span (a name, entity, number, or short "
        "phrase), with no explanation.\nQuestion: " + q + "\nAnswer:"
    )


def _short_ans(t):
    t = t or ""
    if not t.strip():
        return ""
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.sub(r"(?i)\b(the answer is|answer)\b[:\s]*", "", t, count=1)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.fullmatch(r"(?i)answer\s*[:：]?", ln)]
    cand = (lines[0] if lines else t.strip()).strip().strip('"').strip("'").strip()
    cand = re.split(r"\n\s*(?:question|q)\s*[:：]", cand, flags=re.I)[0]
    return re.sub(r"\s*[.。]\s*$", "", cand)


def _head_config():
    hc = HeadClassConfig.from_json(HEAD_CFG_JSON)
    hc.merge_retrieval_to_global()
    return hc


def _ffn_config():
    with open(FFN_CFG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# ── FLOPs accounting (same formulas as the Qwen benchmark) ──
def _model_dims(cfg):
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return {
        "L": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "inter": cfg.intermediate_size,
        "Hq": cfg.num_attention_heads,
        "Hkv": cfg.num_key_value_heads,
        "D": hd,
    }


def _proj_pt(d):
    return (
        2.0 * d["hidden"] * (d["Hq"] + 2 * d["Hkv"]) * d["D"]
        + 2.0 * d["Hq"] * d["D"] * d["hidden"]
    )


def _ffn_pt(d):
    return 6.0 * d["hidden"] * d["inter"]


def _attn_dense(d, T):
    return d["L"] * d["Hq"] * 4.0 * d["D"] * (T * (T + 1) / 2.0)


def _attn_hc(d, T, frac_global, window):
    Hg, Hl = d["Hq"] * frac_global, d["Hq"] * (1 - frac_global)
    kv_g = T * (T + 1) / 2.0
    kv_l = T * min(window, T)
    return d["L"] * 4.0 * d["D"] * (Hg * kv_g + Hl * kv_l)


def compute_flops(d, T, frac_global, sel_deep, dense_until, window):
    proj = d["L"] * T * _proj_pt(d)
    ffn_dense = d["L"] * T * _ffn_pt(d)
    dl = min(dense_until, d["L"])
    ffn_hc = dl * T * _ffn_pt(d) + (d["L"] - dl) * T * _ffn_pt(d) * sel_deep
    # Use the ACTUAL local-head window (adaptive or config), not a hardcoded
    # constant, otherwise attention FLOPs savings are overstated.
    a_d, a_h = _attn_dense(d, T), _attn_hc(d, T, frac_global, window)
    return {
        "attn": (a_d, a_h),
        "ffn": (ffn_dense, ffn_hc),
        "proj": (proj, proj),
        "total": (proj + ffn_dense + a_d, proj + ffn_hc + a_h),
    }


# ── Standard fastest dense prefill baseline (FA-2) ──
@torch.no_grad()
def standard_prefill(model, tok, full_text, query_text, max_new_tokens):
    device = model.device
    ids = tok(full_text + query_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    n_ctx = ids.shape[1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    first = out.logits[0, -1, :].clone()
    nxt = first.argmax().view(1, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    past = out.past_key_values
    gen = [int(nxt[0, 0])]
    t1 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        og = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = og.past_key_values
        nxt = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        gen.append(tid)
        if tid == tok.eos_token_id:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dec_t = max(time.perf_counter() - t1, 1e-3)
    return tok.decode(gen, skip_special_tokens=True), ttft, len(gen) / dec_t, n_ctx


def _trunc(s, n=44):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_compile = os.environ.get("REDKNOT_COMPILE", "1") == "1"
    W = 100
    print("=" * W)
    print(
        " BENCHMARK: RedKnot (head-class KV reuse) vs Standard FlashAttention-2 prefill"
    )
    print(" Model: Llama-3.3-70B-Instruct INT4 NF4 | RAG over LongBench | single GPU")
    print(
        " Caveats: RedKnot TTFT excludes offline prefill (RAG reuse); "
        "FLOPs are analytic (compute axis)."
    )
    print("=" * W)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hc = _head_config()
    # Fixed local-window sweet spot (overrides config & ratio). Llama sweet
    # spot ~4096: small enough to sparsify attention at long ctx, large enough
    # to keep F1. Default comes from the FFN config's "local_window" field
    # (4096); set REDKNOT_WINDOW_FIXED=0 to fall back to the ratio/config.
    _wfix = int(
        os.environ.get(
            "REDKNOT_WINDOW_FIXED", str(_ffn_config().get("local_window", 0))
        )
    )
    if _wfix > 0:
        hc.set_local_window(_wfix)
        global WINDOW_RATIO
        WINDOW_RATIO = None  # fixed window takes precedence over ratio
    summ = hc.summary()
    frac_global = summ.get("global", 0) / summ["total"]
    # Realized local-head window for FLOPs accounting.
    _cfg_wins = {dd for row in hc.head_max_distance for dd in row if dd > 0}
    _orig_cfg_window = max(_cfg_wins) if _cfg_wins else 8192
    # Precision / sharding mode.
    #   REDKNOT_DTYPE=int4 (default): single-GPU bitsandbytes NF4.
    #   REDKNOT_DTYPE=bf16          : multi-GPU bf16 via device_map (tensor /
    #                                 layer sharding). Avoids the INT4 dequant
    #                                 overhead that caps single-GPU speedup, and
    #                                 splits the 70B across GPUs so longer
    #                                 contexts (32K/64K) fit. Set
    #                                 REDKNOT_DEVICE_MAP=auto (default) to let
    #                                 HF balance layers across all visible GPUs.
    dtype_mode = os.environ.get("REDKNOT_DTYPE", "int4").lower()
    device_map = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
    if dtype_mode == "bf16":
        print(f" Loading {MODEL_PATH} (bf16, device_map={device_map})...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    else:
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f" Loading {MODEL_PATH} (INT4)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=qc,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    d = _model_dims(model.config)
    print(
        f" Heads: {summ.get('global', 0)} global + {summ.get('local', 0)} local "
        f"(frac_global={frac_global:.3f}) | window+sink={WINDOW} | layers={d['L']}"
    )

    # Three-tier Sparse-FFN schedule (read straight from the model JSON):
    #   shallow [0, dense_until)            -> dense
    #   mid     [dense_until, deep_start)   -> mass_thresh
    #   deep    [deep_start, num_layers)    -> mass_thresh_deep
    # Llama-3.3-70B sweet spot: 20 dense / mid mass=0.3 / last 20 mass=0.1.
    n_layers_cfg = model.config.num_hidden_layers
    ffn_cfg = _ffn_config()
    dense_until = int(
        os.environ.get("REDKNOT_FFN_DENSE_UNTIL", str(ffn_cfg["dense_until"]))
    )
    ffn_mass = float(os.environ.get("REDKNOT_FFN_MASS", str(ffn_cfg["mass_thresh"])))
    deep_start = int(
        os.environ.get("REDKNOT_FFN_DEEP_START", str(ffn_cfg["deep_layer_start"]))
    )
    mass_deep = float(
        os.environ.get("REDKNOT_FFN_MASS_DEEP", str(ffn_cfg["mass_thresh_deep"]))
    )
    recent_n = int(os.environ.get("REDKNOT_FFN_RECENT_N", str(ffn_cfg["recent_n"])))
    if dense_until >= n_layers_cfg:
        sched = SparseFFNSchedule(dense_until=n_layers_cfg, mass_thresh=1.0)
    else:
        sched = SparseFFNSchedule(
            dense_until=dense_until,
            mass_thresh=ffn_mass,
            deep_layer_start=deep_start,
            mass_thresh_deep=mass_deep,
            recent_n=recent_n,
        )
    print(
        f" Sparse FFN (3-tier): dense<{dense_until}, mid mass={ffn_mass} "
        f"[{dense_until},{deep_start}), deep mass={mass_deep} [{deep_start},"
        f"{n_layers_cfg}), recent_n={recent_n}"
    )

    # Length-sweep mode: REDKNOT_LENGTHS="16K,32K,64K" pads each dataset's
    # RAG context to those token targets (gold passage + distractor docs).
    _len_map = {
        "16K": 16000,
        "32K": 32000,
        "64K": 64000,
        "8K": 8000,
        "24K": 24000,
        "40K": 40000,
    }
    length_labels = [
        x.strip()
        for x in os.environ.get("REDKNOT_LENGTHS", "").split(",")
        if x.strip() in _len_map
    ]

    # Build the list of (label, samples) tasks.
    tasks = []
    if length_labels:
        ds0 = DATASETS[0].strip()
        for lab in length_labels:
            s = _load_longbench_padded(ds0, tok, N_SAMPLES, _len_map[lab])
            tasks.append((f"{ds0}@{lab}", s))
    else:
        for ds_name in DATASETS:
            ds_name = ds_name.strip()
            if ds_name:
                tasks.append((ds_name, _load_longbench(ds_name, tok, N_SAMPLES)))

    overall = {}
    for ds_name, samples in tasks:
        if not samples:
            print(f"\n [skip] {ds_name}: no usable samples")
            continue
        print(f"\n{'=' * W}")
        print(f" TASK: {ds_name}  ({len(samples)} sample(s), chunk={CHUNK_TOKENS})")
        print("=" * W)

        rows, sel_deep_seen, ctx_lens = [], [], []
        for si, s in enumerate(samples):
            qt = _query_text(s["question"])
            golds = s["golds"]
            full_text = "\n\n".join(s["docs"])

            tb, ttft_b, dec_b, n_ctx = standard_prefill(
                model, tok, full_text, qt, MAX_NEW_TOKENS
            )
            ans_b = _short_ans(tb)
            ctx_lens.append(n_ctx)
            gc.collect()
            torch.cuda.empty_cache()

            segs = offline_prefill_segments(
                model,
                tok,
                s["docs"],
                chunk_size=max(4096, CHUNK_TOKENS + 96),
                model_id=MODEL_PATH,
            )
            if si == 0:
                run_redknot_offlinekv(
                    model,
                    tok,
                    segments_offline=segs,
                    query_text=qt,
                    head_cfg=hc,
                    max_new_tokens=3,
                    kernel="fa3_parallel",
                    sparse_ffn_schedule=sched,
                    use_compile=use_compile,
                    window_ratio=WINDOW_RATIO,
                )
            stats = []
            t0 = time.perf_counter()
            _, tc, _, ttft_c = run_redknot_offlinekv(
                model,
                tok,
                segments_offline=segs,
                query_text=qt,
                head_cfg=hc,
                max_new_tokens=MAX_NEW_TOKENS,
                window_ratio=WINDOW_RATIO,
                kernel="fa3_parallel",
                sparse_ffn_schedule=sched,
                sparse_ffn_stats=stats,
                use_compile=use_compile,
            )
            torch.cuda.synchronize()
            tot_c = time.perf_counter() - t0
            ans_c = _short_ans(tc)
            n_dec = len(tok(tc, add_special_tokens=False)["input_ids"]) or 1
            dec_c = n_dec / max(tot_c - ttft_c, 1e-3)
            sp = [x for x in stats if x.get("mode") == "sparse"]
            if sp:
                sel_deep_seen.append(sum(x["selected_frac"] for x in sp) / len(sp))

            rows.append(
                {
                    "f1_b": f1_max(ans_b, golds),
                    "em_b": em_max(ans_b, golds),
                    "f1_c": f1_max(ans_c, golds),
                    "em_c": em_max(ans_c, golds),
                    "ttft_b": ttft_b,
                    "ttft_c": ttft_c,
                    "dec_b": dec_b,
                    "dec_c": dec_c,
                }
            )
            print(f"\n [sample {si}] ctx={n_ctx:,} tok")
            print(f"   Q   : {_trunc(s['question'], 80)}")
            print(f"   gold: {golds[0] if golds else ''}")
            print(
                f"   base: {_trunc(tb, 60)!r} -> {_trunc(ans_b, 28)!r} "
                f"F1={rows[-1]['f1_b']:.2f}"
            )
            print(
                f"   rk  : {_trunc(tc, 60)!r} -> {_trunc(ans_c, 28)!r} "
                f"F1={rows[-1]['f1_c']:.2f}"
            )
            print(
                f"   TTFT: base={ttft_b:.2f}s  rk={ttft_c:.2f}s  "
                f"speedup={ttft_b / ttft_c:.2f}x"
            )

            del segs
            gc.collect()
            torch.cuda.empty_cache()

        if not rows:
            continue
        m = lambda k: sum(r[k] for r in rows) / len(rows)
        sel_deep = (sum(sel_deep_seen) / len(sel_deep_seen)) if sel_deep_seen else 1.0
        T = int(sum(ctx_lens) / len(ctx_lens))
        # Realized local-head window: adaptive (ctx*ratio) if set, else config.
        eff_window = (
            max(256, int(T * WINDOW_RATIO)) if WINDOW_RATIO else _orig_cfg_window
        )
        fl = compute_flops(d, T, frac_global, sel_deep, dense_until, eff_window)
        P = 1e15

        print(f"\n {'-' * (W - 2)}")
        print(f" {ds_name} AGGREGATE ({len(rows)} sample(s), avg ctx={T:,} tok)")
        print(f" {'-' * (W - 2)}")
        print(f"   QUALITY   baseline  F1={m('f1_b'):.3f}  EM={m('em_b'):.3f}")
        print(f"             RedKnot   F1={m('f1_c'):.3f}  EM={m('em_c'):.3f}")
        print(
            f"   TTFT      baseline={m('ttft_b'):.2f}s  RedKnot={m('ttft_c'):.2f}s "
            f" speedup={m('ttft_b') / m('ttft_c'):.2f}x"
        )
        print(
            f"   DECODE    baseline={m('dec_b'):.1f} tok/s  "
            f"RedKnot={m('dec_c'):.1f} tok/s"
        )
        print(f"   COMPUTE (prefill FLOPs, P=1e15):")
        for name in ["attn", "ffn", "proj", "total"]:
            dn, hcv = fl[name]
            sv = (1 - hcv / dn) * 100 if dn > 0 else 0.0
            print(
                f"             {name:6s} dense={dn / P:7.3f}P  RedKnot={hcv / P:7.3f}P "
                f" saving={sv:5.1f}%  ({(dn / hcv) if hcv > 0 else 0:.2f}x)"
            )
        overall[ds_name] = {
            "f1_b": m("f1_b"),
            "f1_c": m("f1_c"),
            "ttft_b": m("ttft_b"),
            "ttft_c": m("ttft_c"),
            "flops_saving": (1 - fl["total"][1] / fl["total"][0]) * 100,
        }

    print(f"\n{'=' * W}")
    print(" SUMMARY  (Llama-3.3-70B INT4, RAG over LongBench)")
    print("=" * W)
    print(
        f" {'dataset':18s} {'base F1':>8s} {'rk F1':>7s} {'base TTFT':>10s} "
        f"{'rk TTFT':>9s} {'speedup':>8s} {'FLOPs save':>11s}"
    )
    for ds_name, o in overall.items():
        print(
            f" {ds_name:18s} {o['f1_b']:>8.3f} {o['f1_c']:>7.3f} "
            f"{o['ttft_b']:>9.2f}s {o['ttft_c']:>8.2f}s "
            f"{o['ttft_b'] / o['ttft_c']:>7.2f}x {o['flops_saving']:>10.1f}%"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
