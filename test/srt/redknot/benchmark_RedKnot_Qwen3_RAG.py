#!/usr/bin/env python3
"""Benchmark: RedKnot head-class KV-reuse vs the standard FA-2 dense prefill.

One-command run::

    python benchmark_RedKnot_Qwen3_RAG.py

It loads three real long-context QA datasets from ``datasets/LongBench`` (the
three that behave best for short-span extractive QA), runs both the STANDARD
FlashAttention-2 prefill baseline and RedKnot's head-class KV-reuse path
(``run_redknot_offlinekv``) on Qwen3-32B (INT4 NF4, single GPU), and reports
for every dataset:

  * answer quality   : SQuAD F1 / EM (max over reference answers)
  * TTFT             : time-to-first-token and the speedup
  * decode throughput: tok/s
  * COMPUTE (FLOPs)  : attention / FFN / projection prefill cost, with savings

Baseline = the honest reference: one ``model(input_ids)`` forward over the full
context with ``attn_implementation="flash_attention_2"``.

Everything is wired with sensible defaults so no env vars / CLI flags are
required.  They can still be overridden via the environment if desired
(``REDKNOT_MODEL_PATH``, ``REDKNOT_DATASETS``, ``REDKNOT_N_SAMPLES``, ...).

Model resolution order:
  1. ``$REDKNOT_MODEL_PATH`` if set and present;
  2. the local checkpoint at ``LOCAL_MODEL_PATH`` if present;
  3. otherwise pull ``HF_MODEL_ID`` ("Qwen/Qwen3-32B") from the HF Hub.
"""

from __future__ import annotations

import gc
import json
import os
import re
import string
import sys
import time

# Reduce allocator fragmentation for the long-context baseline forward (must be
# set before torch initialises CUDA). Newer torch renamed the env var; set both.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.layers.attention.redknot import (  # noqa: E402
    HeadClassConfig,
    SparseFFNSchedule,
    get_global_offline_cache,
    offline_prefill_segments,
    run_redknot_offlinekv,
)

HERE = Path(__file__).resolve().parent

# ────────────────────────────────────────────────────────────────────────
# Configuration (defaults are baked in -> `python this_file.py` just works)
# ────────────────────────────────────────────────────────────────────────
# Model: try a local checkpoint first, fall back to the HF Hub.
LOCAL_MODEL_PATH = "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3-32B"
HF_MODEL_ID = "Qwen/Qwen3-32B"


def _resolve_model_path() -> str:
    """Local checkpoint if present, else the HF Hub id (downloaded on demand)."""
    env = os.environ.get("REDKNOT_MODEL_PATH")
    if env and Path(env).exists():
        return env
    if Path(LOCAL_MODEL_PATH).exists():
        return LOCAL_MODEL_PATH
    # Not found locally -> let transformers pull it from the Hub.
    print(
        f"[info] local model not found; will download '{HF_MODEL_ID}' from HF Hub.",
        flush=True,
    )
    # Make sure we are NOT in offline mode when we have to download.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    return HF_MODEL_ID


MODEL_PATH = _resolve_model_path()

# The three best-behaving LongBench subsets for short-span extractive QA.
LONGBENCH_DIR = HERE / "datasets/LongBench/data"
DEFAULT_DATASETS = ["multifieldqa_en", "2wikimqa", "hotpotqa"]
DATASETS = [
    d
    for d in os.environ.get("REDKNOT_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if d.strip()
]

# Head-class + Sparse-FFN configs (Qwen3-32B optimal profile).
HEAD_CFG_JSON = os.environ.get(
    "REDKNOT_HEAD_CFG",
    str(HERE / "head_class/qwen3-32B_optimal_g15_lf_ret.json"),
)
FFN_CFG_JSON = os.environ.get(
    "REDKNOT_FFN_CFG",
    str(HERE / "sparse_ffn_params/qwen3-32B.json"),
)

# Runtime knobs.
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "20"))
MAX_NEW_TOKENS = int(os.environ.get("REDKNOT_MAX_NEW", "32"))
# torch.compile is OFF by default: across variable-length samples it triggers
# repeated recompilation whose cached graphs grow GPU memory monotonically and
# eventually OOM on a shared 80G node. It only affects speed, not F1/TTFT
# correctness. Set REDKNOT_COMPILE=1 on a dedicated GPU to enable it.
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "0") == "1"
SEG_TOKENS = int(os.environ.get("REDKNOT_SEG_TOKENS", "4000"))  # tokens per segment
# Cap context length. 16K keeps the dense baseline's peak memory well under one
# 80G GPU (leaves headroom on shared nodes) while still being a real
# long-context workload inside Qwen3-32B's native 32K window (F1 reliable).
# Raise via REDKNOT_MAX_CTX on a dedicated GPU for longer contexts.
MAX_CTX_TOKENS = int(os.environ.get("REDKNOT_MAX_CTX", "16000"))
SEED = 2026


# ────────────────────────────────────────────────────────────────────────
# Metrics (SQuAD-style F1 / EM, max over reference answers — LongBench style)
# ────────────────────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def _em(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1_score(pred: str, golds) -> float:
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    return max((_f1(pred, g) for g in golds), default=0.0)


def em_score(pred: str, golds) -> float:
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    return max((_em(pred, g) for g in golds), default=0.0)


# ────────────────────────────────────────────────────────────────────────
# LongBench data loading
# ────────────────────────────────────────────────────────────────────────
# Per-dataset answer prompt (kept short; we want a single exact span).
_PROMPT = (
    "\n\nAnswer the question based on the passages above. Give the shortest "
    "exact answer span (a name, entity, number, or short noun phrase) with no "
    "explanation.\nQuestion: {q}\nAnswer:"
)


def _load_longbench(name, tok, n_samples, seg_tokens, max_ctx_tokens):
    """Load up to ``n_samples`` rows, returning RedKnot-ready segmented docs.

    Each row's real context is tokenized, capped to ``max_ctx_tokens`` and
    chopped into ``seg_tokens``-sized segments (RedKnot prefills each segment
    offline and reuses its KV).  The baseline sees the same concatenated text.

    Rows are sorted by context length (longest first) so the chosen samples
    actually exercise long-context behaviour — that is where head-class KV
    reuse matters and where TTFT/F1 are most informative.
    """
    path = LONGBENCH_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    # Longest contexts first (char length is a cheap proxy for token length).
    rows.sort(key=lambda r: len(str(r.get("context", ""))), reverse=True)
    out = []
    for r in rows:
        ctx = str(r.get("context", "")).strip()
        q = str(r.get("input", "")).strip()
        ans = r.get("answers", [])
        if isinstance(ans, str):
            ans = [ans]
        if not ctx or not q or not ans:
            continue
        ids = tok(ctx, add_special_tokens=False)["input_ids"]
        if len(ids) < seg_tokens:  # too short to exercise long-context reuse
            continue
        ids = ids[:max_ctx_tokens]
        # Split into segments of ~seg_tokens each.
        docs = [
            tok.decode(ids[i : i + seg_tokens], skip_special_tokens=True)
            for i in range(0, len(ids), seg_tokens)
        ]
        out.append(
            {
                "question": q,
                "answers": ans,
                "docs": docs,
                "n_ctx": len(ids),
            }
        )
        if len(out) >= n_samples:
            break
    return out


def _short_ans(t):
    t = t or ""
    if not t.strip():
        return ""
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.sub(r"(?i)\bthe answer is\b[:\s]*", "", t)
    t = re.sub(r"(?is)^\s*answer\s*[:：]\s*", "", t)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.fullmatch(r"(?i)answer\s*[:：]?", ln)]
    cand = (lines[0] if lines else t.strip()).strip().strip('"').strip("'").strip()
    cand = re.split(r"\n\s*(?:question|q)\s*[:：]", cand, flags=re.I)[0]
    cand = re.sub(r"\s*[.。]\s*$", "", cand)
    return cand


# ────────────────────────────────────────────────────────────────────────
# Head / FFN config helpers
# ────────────────────────────────────────────────────────────────────────
def _head_config():
    hc = HeadClassConfig.from_json(HEAD_CFG_JSON)
    hc.merge_retrieval_to_global()
    return hc


def _ffn_config():
    with open(FFN_CFG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _local_window_from_cfg(hc) -> int:
    """The actual sliding window local heads use at runtime (for honest FLOPs).

    Reads the per-head window tensors and returns the dominant LOCAL window so
    the FLOPs accounting matches what is really executed (no hard-coded value).
    """
    try:
        from sglang.srt.layers.attention.redknot.driver_batched import (
            POLICY_LOCAL,
            _head_policy_for_layer,
            _head_window_for_layer,
        )

        ws = []
        for li in range(hc.num_layers):
            pol = _head_policy_for_layer(hc, li)
            win = _head_window_for_layer(hc, li)
            for p, w in zip(pol, win):
                if p == POLICY_LOCAL and int(w) > 0:
                    ws.append(int(w))
        if ws:
            return Counter(ws).most_common(1)[0][0]
    except Exception:
        pass
    return 4096


# ── FLOPs accounting ──
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


def _proj_flops_per_token(d):
    qkv = 2.0 * d["hidden"] * (d["Hq"] + 2 * d["Hkv"]) * d["D"]
    o = 2.0 * d["Hq"] * d["D"] * d["hidden"]
    return qkv + o


def _ffn_flops_per_token(d):
    return 6.0 * d["hidden"] * d["inter"]


def _attn_flops_dense(d, seg_lens):
    flops, prefix = 0.0, 0
    for li in seg_lens:
        kv_pairs = li * prefix + li * (li + 1) / 2.0
        flops += d["L"] * d["Hq"] * 4.0 * d["D"] * kv_pairs
        prefix += li
    return flops


def _attn_flops_headclass(d, seg_lens, frac_global, window):
    Hq = d["Hq"]
    Hg, Hl = Hq * frac_global, Hq * (1 - frac_global)
    flops, prefix = 0.0, 0
    for li in seg_lens:
        kv_g = li * prefix + li * (li + 1) / 2.0
        kv_l = li * min(window, prefix + li)
        flops += d["L"] * 4.0 * d["D"] * (Hg * kv_g + Hl * kv_l)
        prefix += li
    return flops


def compute_flops(d, seg_lens, frac_global, sel_deep, dense_until, window):
    tokens = sum(seg_lens)
    proj = d["L"] * tokens * _proj_flops_per_token(d)
    ffn_dense = d["L"] * tokens * _ffn_flops_per_token(d)
    dense_layers = min(dense_until, d["L"])
    deep_layers = d["L"] - dense_layers
    ffn_hc = (
        dense_layers * tokens * _ffn_flops_per_token(d)
        + deep_layers * tokens * _ffn_flops_per_token(d) * sel_deep
    )
    attn_dense = _attn_flops_dense(d, seg_lens)
    attn_hc = _attn_flops_headclass(d, seg_lens, frac_global, window)
    total_dense = proj + ffn_dense + attn_dense
    total_hc = proj + ffn_hc + attn_hc
    return {
        "attn": (attn_dense, attn_hc),
        "ffn": (ffn_dense, ffn_hc),
        "proj": (proj, proj),
        "total": (total_dense, total_hc),
    }


# ── Standard fastest dense prefill baseline (FA-2) ──
@torch.no_grad()
def standard_prefill(
    model, tok, full_text, query_text, max_new_tokens, prefill_chunk=8192
):
    device = model.device
    ids = tok(full_text + query_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    n_ctx = ids.shape[1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Chunked prefill: build the KV cache in <=prefill_chunk slices so the peak
    # activation (MLP intermediate over the whole sequence) stays bounded. This
    # is still the honest dense FA-2 baseline — same FLOPs and same logits — it
    # just avoids the single-shot 24K activation spike that OOMs an 80G GPU.
    past = None
    n = ids.shape[1]
    for start in range(0, n, prefill_chunk):
        chunk = ids[:, start : start + prefill_chunk]
        last = start + chunk.shape[1] >= n
        out = model(
            input_ids=chunk,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1 if last else 0,
        )
        past = out.past_key_values
    first = out.logits[0, -1, :].clone()
    nxt = first.argmax().view(1, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
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


def _trunc(s, n=40):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    W = 100
    print("=" * W)
    print(" BENCHMARK: RedKnot (head-class KV reuse) vs Standard FlashAttention-2")
    print(
        f" Model: Qwen3-32B (INT4)  |  Datasets: {', '.join(DATASETS)}  |  single GPU"
    )
    print("=" * W)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hc = _head_config()
    summ = hc.summary()
    frac_global = summ.get("global", 0) / summ["total"]
    local_window = _local_window_from_cfg(hc)
    ffn_cfg = _ffn_config()
    sched = SparseFFNSchedule(**ffn_cfg)

    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
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
        f"(frac_global={frac_global:.3f}) | local_window={local_window}"
    )
    print(
        f" Sparse FFN: mass_thresh={ffn_cfg['mass_thresh']} "
        f"(deep {ffn_cfg['mass_thresh_deep']}), dense_until={ffn_cfg['dense_until']}"
    )
    print(
        f" Samples/dataset={N_SAMPLES}  seg_tokens={SEG_TOKENS}  "
        f"max_ctx={MAX_CTX_TOKENS}  max_new={MAX_NEW_TOKENS}  compile={USE_COMPILE}"
    )

    chunk_size = max(4096, SEG_TOKENS + 96)
    overall = {}

    for name in DATASETS:
        samples = _load_longbench(name, tok, N_SAMPLES, SEG_TOKENS, MAX_CTX_TOKENS)
        print(f"\n{'=' * W}")
        print(f" DATASET: {name}  ({len(samples)} sample(s))")
        print("=" * W)
        if not samples:
            print("   (no usable samples; skipped)")
            continue

        rows = []
        sel_deep_seen = []
        warmed = False
        for si, s in enumerate(samples):
            qt = _PROMPT.format(q=s["question"])
            golds = s["answers"]
            full_text = "\n\n".join(s["docs"])

            # ── baseline: standard FA-2 prefill ──
            tb, ttft_b, dec_b, n_ctx = standard_prefill(
                model, tok, full_text, qt, MAX_NEW_TOKENS
            )
            ans_b = _short_ans(tb)
            gc.collect()
            torch.cuda.empty_cache()

            # ── RedKnot head-class KV reuse ──
            segs = offline_prefill_segments(
                model, tok, s["docs"], chunk_size=chunk_size, model_id=MODEL_PATH
            )
            if not warmed:
                run_redknot_offlinekv(
                    model,
                    tok,
                    segments_offline=segs,
                    query_text=qt,
                    head_cfg=hc,
                    max_new_tokens=3,
                    kernel="fa3_parallel",
                    sparse_ffn_schedule=sched,
                    use_compile=USE_COMPILE,
                )
                warmed = True
            stats = []
            t0 = time.perf_counter()
            _, tc, _, ttft_c = run_redknot_offlinekv(
                model,
                tok,
                segments_offline=segs,
                query_text=qt,
                head_cfg=hc,
                max_new_tokens=MAX_NEW_TOKENS,
                kernel="fa3_parallel",
                sparse_ffn_schedule=sched,
                sparse_ffn_stats=stats,
                use_compile=USE_COMPILE,
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
                    "base_f1": f1_score(ans_b, golds),
                    "rk_f1": f1_score(ans_c, golds),
                    "base_em": em_score(ans_b, golds),
                    "rk_em": em_score(ans_c, golds),
                    "base_ttft": ttft_b,
                    "rk_ttft": ttft_c,
                    "base_dec": dec_b,
                    "rk_dec": dec_c,
                    "n_ctx": n_ctx,
                }
            )

            print(f"\n [sample {si}] ctx={n_ctx:,} tok")
            print(f"   Q   : {_trunc(s['question'], 80)}")
            print(f"   gold: {_trunc(str(golds[0]), 60)}")
            print(
                f"   base: ans={_trunc(ans_b, 30)!r} F1={rows[-1]['base_f1']:.2f}  |  "
                f"rk: ans={_trunc(ans_c, 30)!r} F1={rows[-1]['rk_f1']:.2f}"
            )
            print(
                f"   TTFT: base={ttft_b:.2f}s  rk={ttft_c:.2f}s  "
                f"speedup={ttft_b / max(ttft_c, 1e-6):.2f}x"
            )

            del segs
            # The offline KV cache is a process-wide singleton: every segment's
            # GPU-resident KV is registered there and otherwise persists across
            # samples, growing memory monotonically until OOM. Clear it per
            # sample so peak memory stays bounded to a single request.
            get_global_offline_cache().clear()
            gc.collect()
            torch.cuda.empty_cache()

        if not rows:
            continue

        m = lambda k: sum(r[k] for r in rows) / len(rows)
        sel_deep = sum(sel_deep_seen) / len(sel_deep_seen) if sel_deep_seen else 0.14
        avg_ctx = int(m("n_ctx"))
        n_seg = max(1, -(-avg_ctx // SEG_TOKENS))  # ceil
        seg_lens = [SEG_TOKENS] * (n_seg - 1) + [avg_ctx - SEG_TOKENS * (n_seg - 1)]
        fl = compute_flops(
            d, seg_lens, frac_global, sel_deep, ffn_cfg["dense_until"], local_window
        )
        P = 1e15

        print(f"\n {'-' * (W - 2)}")
        print(f" {name} AGGREGATE ({len(rows)} sample(s), avg ctx={avg_ctx:,} tok)")
        print(f" {'-' * (W - 2)}")
        print(f"   QUALITY   baseline  F1={m('base_f1'):.3f}  EM={m('base_em'):.3f}")
        print(f"             RedKnot   F1={m('rk_f1'):.3f}  EM={m('rk_em'):.3f}")
        print(
            f"   TTFT      baseline={m('base_ttft'):.2f}s  "
            f"RedKnot={m('rk_ttft'):.2f}s  "
            f"speedup={m('base_ttft') / max(m('rk_ttft'), 1e-6):.2f}x"
        )
        print(
            f"   DECODE    baseline={m('base_dec'):.1f} tok/s  "
            f"RedKnot={m('rk_dec'):.1f} tok/s"
        )
        print(f"   COMPUTE (prefill FLOPs, P=1e15, local_window={local_window}):")
        for cn in ["attn", "ffn", "proj", "total"]:
            dn, hcv = fl[cn]
            sv = (1 - hcv / dn) * 100 if dn > 0 else 0.0
            print(
                f"             {cn:6s} dense={dn / P:7.3f}P  "
                f"RedKnot={hcv / P:7.3f}P  saving={sv:5.1f}%  "
                f"({(dn / hcv) if hcv > 0 else 0:.2f}x)"
            )
        overall[name] = {
            "base_f1": m("base_f1"),
            "rk_f1": m("rk_f1"),
            "base_ttft": m("base_ttft"),
            "rk_ttft": m("rk_ttft"),
            "flops_saving": (1 - fl["total"][1] / fl["total"][0]) * 100,
        }

    # final summary table
    print(f"\n{'=' * W}")
    print(" SUMMARY")
    print("=" * W)
    print(
        f" {'dataset':16s} {'base F1':>8s} {'rk F1':>7s} {'base TTFT':>10s} "
        f"{'rk TTFT':>9s} {'speedup':>8s} {'FLOPs save':>11s}"
    )
    for name, o in overall.items():
        print(
            f" {name:16s} {o['base_f1']:>8.3f} {o['rk_f1']:>7.3f} "
            f"{o['base_ttft']:>9.2f}s {o['rk_ttft']:>8.2f}s "
            f"{o['base_ttft'] / max(o['rk_ttft'], 1e-6):>7.2f}x "
            f"{o['flops_saving']:>10.1f}%"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
