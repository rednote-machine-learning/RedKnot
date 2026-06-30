#!/usr/bin/env python3
"""Benchmark: RedKnot head-class KV-reuse vs the standard fastest dense prefill.

Compares, on **LongBench** long-context (default 4 x 8K = 32K), the STANDARD
FA-2 prefill baseline against RedKnot's head-class path
(`run_redknot_offlinekv`), reporting for every sample and as aggregates:

  * the generated ANSWER TEXT (baseline vs RedKnot vs gold)
  * answer quality (SQuAD F1 / EM, scored against LongBench's answer list)
  * TTFT (time-to-first-token) and the speedup
  * decode throughput (tok/s)
  * COMPUTE (FLOPs) comparison: attention / FFN / projection, with savings

Pipeline (exactly your spec):
  1. Each request is **>8K per document** (default 4 docs x 8000 tok = 32K).
  2. `offline_prefill_segments` builds the per-document KV OFFLINE (local
     coordinates), once per request.
  3. `run_redknot_offlinekv` runs the RedKnot algorithm online: local-head KV
     is **RoPE-repositioned** to global coordinates and truncated to
     sink+window; global heads are re-prefilled via the sparse online forward;
     then the query attends per-head. (RoPE relocation + the head-class algo
     are internal to that call.)

Baseline = the honest, hard-to-beat reference: one `model(input_ids)` forward
over the full concatenated context with `attn_implementation="flash_attention_2"`
(not RedKnot's per-slot framework, which would carry our own dispatch overhead).

Model: Mistral-7B-Instruct-v0.3 (bf16), single GPU. Local model + bundled
LongBench datasets; falls back to the HuggingFace hub if the model isn't local.

One-click (zero config -> RedKnot defaults: 98%local/2%global head config,
3000 local window, 4x7K RAG context, 3-tier Sparse-FFN, torch.compile):
    python test/srt/redknot/benchmark_RedKnot_Mistral_RAG.py

Optional overrides:
    REDKNOT_N_SAMPLES=4 REDKNOT_DATASETS=triviaqa,hotpotqa \
    REDKNOT_TOKENS_PER_DOC=5000 REDKNOT_LENGTHS=4x \
    python test/srt/redknot/benchmark_RedKnot_Mistral_RAG.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import random
import re
import string
from collections import Counter

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
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Mistral-7B-Instruct-v0.3",
)
# HuggingFace repo id used as a fallback when no local model is found.
HF_MODEL_ID = os.environ.get(
    "REDKNOT_HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"
)


def _resolve_model_source(local_path: str, hf_id: str) -> str:
    """Return a usable model identifier for ``from_pretrained``.

    Prefer the local checkpoint if it exists on disk; otherwise fall back to
    the HuggingFace repo id (``from_pretrained`` will download it).
    """
    if local_path and os.path.isdir(local_path):
        print(f" Using local model: {local_path}")
        return local_path
    print(
        f" Local model not found at {local_path!r}; "
        f"falling back to HuggingFace hub: {hf_id}"
    )
    return hf_id


# LongBench data (jsonl-per-dataset). Each line: {input, context, answers, ...}.
# Default to the datasets bundled next to this benchmark so it works out of the
# box; override with REDKNOT_LONGBENCH_DIR for a different location.
_LOCAL_LONGBENCH = str(Path(__file__).resolve().parent / "datasets/LongBench/data")
LONGBENCH_DIR = os.environ.get("REDKNOT_LONGBENCH_DIR", _LOCAL_LONGBENCH)
# Which LongBench QA datasets to evaluate (comma separated).
DATASETS = [
    s for s in os.environ.get("REDKNOT_DATASETS", "triviaqa").split(",") if s.strip()
]
SEED = 2026


# ────────────────────────────────────────────────────────────────────────
# Self-contained data loading + metrics (no external test-file dependency)
# ────────────────────────────────────────────────────────────────────────
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
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1_max(pred: str, golds) -> float:
    """LongBench answers is a LIST; score against the best-matching gold."""
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    golds = [str(g) for g in golds if str(g).strip()]
    return max((f1_score(pred, g) for g in golds), default=0.0)


def em_max(pred: str, golds) -> float:
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    golds = [str(g) for g in golds if str(g).strip()]
    return max((em_score(pred, g) for g in golds), default=0.0)


def _mean_std_ci(values):
    """Return (mean, sample_std, half_95_ci) for a list of scores.

    Uses the unbiased sample std (ddof=1) and a normal-approx 95% CI on the
    mean (1.96 * std / sqrt(n)). With the tiny n used by these RAG benchmarks
    (e.g. n=4) the CI is intentionally WIDE -- that width is the point: it
    tells the reader how little a single F1 number can be trusted here.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = var**0.5
    half_ci = 1.96 * std / (n**0.5)
    return mean, std, half_ci


def _load_samples(tok, dataset, n_segments, n_samples, tokens_per_segment=8000):
    """Load LongBench samples and build exactly ``n_segments`` documents of
    ``tokens_per_segment`` tokens each (>8K per doc by spec).

    LongBench jsonl lines carry a single long ``context`` and an ``answers``
    list. To reach ``n_segments * tokens_per_segment`` tokens we concatenate
    the context of the chosen sample with the contexts of following samples
    (wrap-around), then slice the token stream into equal ``tokens_per_segment``
    chunks -> these are the ``docs`` whose KV is built offline.
    """
    path = os.path.join(LONGBENCH_DIR, f"{dataset}.jsonl")
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    if not raw:
        raise RuntimeError(f"No usable rows in LongBench dataset {path!r}")
    random.Random(SEED).shuffle(raw)

    target = n_segments * tokens_per_segment
    nraw = len(raw)
    out = []
    for i in range(nraw):
        if len(out) >= n_samples:
            break
        toks = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % nraw
        # Pad with following contexts (distractor-like) until we reach target.
        while len(toks) < target and j != i:
            toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % nraw
        if len(toks) < target:
            continue  # not enough text even after wrap-around
        toks = toks[:target]
        docs = [
            tok.decode(toks[k : k + tokens_per_segment], skip_special_tokens=True)
            for k in range(0, target, tokens_per_segment)
        ]
        out.append(
            {
                "question": raw[i]["input"],
                "gold_answer": raw[i]["answers"],  # LIST of acceptable answers
                "docs": docs,
                "dataset": dataset,
            }
        )
    return out


def _query_text(q):
    return (
        "\n\nUsing only the documents above, give the shortest exact answer "
        "span to the question (a name, entity, number, or short noun phrase). "
        "Answer with the span only, no explanation.\n"
        f"Question: {q}\nAnswer:"
    )


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


def _head_config():
    hc = HeadClassConfig.from_json(HEAD_CFG_JSON)
    hc.merge_retrieval_to_global()
    # Optional explicit local-window override (e.g. 3000 for the parity config).
    _wfix = int(os.environ.get("REDKNOT_WINDOW_FIXED", "3000"))
    if _wfix > 0:
        hc.set_local_window(_wfix)
    return hc


def _ffn_config():
    with open(FFN_CFG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


HEAD_CFG_JSON = os.environ.get(
    "REDKNOT_HEAD_CFG",
    str(
        Path(__file__).resolve().parent
        / "head_class/mistral-7B_parity_98local_w3000.json"
    ),
)
FFN_CFG_JSON = os.environ.get(
    "REDKNOT_FFN_CFG",
    str(Path(__file__).resolve().parent / "sparse_ffn_params/mistral-7B.json"),
)
MAX_NEW_TOKENS = int(os.environ.get("REDKNOT_MAX_NEW", "48"))
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "4"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "3000")) + 4  # local window + sink

# Per-spec: every document must be LARGER than Mistral's sliding window (4096)
# so that local heads actually reuse a truncated window (the RedKnot win). Each
# document defaults to 7000 tokens (>4K window) and only the document COUNT
# changes, so 28K = 4 x 7000, 56K = 8 x 7000, etc. Total stays <= 32K native
# context at 4 docs, keeping F1 valid.
TOKENS_PER_DOC = int(os.environ.get("REDKNOT_TOKENS_PER_DOC", "7000"))
assert TOKENS_PER_DOC > (WINDOW - 4), (
    f"each document ({TOKENS_PER_DOC} tok) must exceed the RedKnot local "
    f"window ({WINDOW - 4}) to exercise local-head window reuse/recompute"
)
_LEN_MAP = {
    "14K": (2, TOKENS_PER_DOC),
    "28K": (4, TOKENS_PER_DOC),
    "42K": (6, TOKENS_PER_DOC),
    "56K": (8, TOKENS_PER_DOC),
    # Fixed 4-document configs (per-doc size set via REDKNOT_TOKENS_PER_DOC):
    #   4x5K  -> REDKNOT_TOKENS_PER_DOC=5000 REDKNOT_LENGTHS=4x
    #   4x7K  -> REDKNOT_TOKENS_PER_DOC=7000 REDKNOT_LENGTHS=4x
    "4x": (4, TOKENS_PER_DOC),
}
LENGTHS = [
    (label, *_LEN_MAP[label])
    for label in os.environ.get("REDKNOT_LENGTHS", "4x").split(",")
    if label in _LEN_MAP
]


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
    """Full causal prefill attention over the whole context."""
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


def compute_flops(d, seg_lens, frac_global, sel_deep, dense_until):
    """Return dense vs head-class prefill FLOPs broken down by component."""
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
    attn_hc = _attn_flops_headclass(d, seg_lens, frac_global, WINDOW)
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


# ── Pretty printing ──
def _trunc(s, n=40):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_compile = os.environ.get("REDKNOT_COMPILE", "1") == "1"
    one_sample = os.environ.get("REDKNOT_ONE_SAMPLE")  # run only this index
    W = 100

    print("=" * W)
    print(
        " BENCHMARK: RedKnot (head-class KV reuse) vs Standard FlashAttention-2 prefill"
    )
    print(
        f" Model: Mistral-7B-Instruct-v0.3 | Dataset: LongBench "
        f"({','.join(DATASETS)}) | docs>8K | single GPU"
    )
    print("=" * W)

    model_src = _resolve_model_source(MODEL_PATH, HF_MODEL_ID)
    tok = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hc = _head_config()
    summ = hc.summary()
    frac_global = summ.get("global", 0) / summ["total"]
    ffn_cfg = _ffn_config()
    # Allow forcing fully-dense FFN (REDKNOT_FFN_DENSE_UNTIL >= num_layers) so an
    # attention-only parity check isolates the local-window recompute from the
    # separate Sparse-FFN approximation.
    _du = int(os.environ.get("REDKNOT_FFN_DENSE_UNTIL", str(ffn_cfg["dense_until"])))
    ffn_cfg = dict(ffn_cfg)
    ffn_cfg["dense_until"] = _du
    if _du >= 32:
        sched = SparseFFNSchedule(dense_until=32, mass_thresh=1.0)
    else:
        sched = SparseFFNSchedule(**ffn_cfg)
    # Mistral-7B is small (~15GB in bf16) so default to bf16 (fastest, exact).
    # Set REDKNOT_DTYPE=int4 to load NF4 4-bit if GPU memory is tight.
    dtype_mode = os.environ.get("REDKNOT_DTYPE", "bf16").lower()
    if dtype_mode == "int4":
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_src,
            quantization_config=qc,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_src,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    d = _model_dims(model.config)
    print(
        f" Heads: {summ.get('global', 0)} global + {summ.get('local', 0)} local "
        f"(frac_global={frac_global:.3f}) | window+sink={WINDOW}"
    )
    print(
        f" Sparse FFN: mass_thresh={ffn_cfg['mass_thresh']} "
        f"(deep {ffn_cfg['mass_thresh_deep']}), "
        f"dense_until={ffn_cfg['dense_until']}"
    )

    overall = {}
    all_rows = []  # every sample across all tasks, for the Top-N similar report

    for dataset in DATASETS:
        for label, n_seg, tps in LENGTHS:
            n = 1 if one_sample is not None else N_SAMPLES
            samples = _load_samples(
                tok,
                dataset,
                n_seg,
                (int(one_sample) + 1) if one_sample else n,
                tokens_per_segment=tps,
            )
            if one_sample is not None:
                samples = [samples[int(one_sample)]]
            if not samples:
                print(
                    f"\n[skip] {dataset}@{label}: no sample reached {n_seg}x{tps} tok"
                )
                continue
            seg_lens_full = [tps] * n_seg
            online_seg_lens = seg_lens_full[1:]
            # chunk_size must be >= tps so each doc becomes exactly one segment.
            chunk_size = max(4096, tps + 96)

            total_tok = n_seg * tps
            # Mistral-7B-Instruct-v0.3 native context = 32K (max_position_embeddings).
            # Beyond that the model extrapolates: TTFT/FLOPs stay meaningful but
            # answer quality (F1) is unreliable. Flag it honestly.
            region = (
                "NATIVE (<=32K, F1+speed valid)"
                if total_tok <= 32000
                else "EXTRAPOLATION (>32K: TTFT/FLOPs valid, F1 unreliable)"
            )
            print(f"\n{'=' * W}")
            print(
                f" {dataset} @ {label}  ({n_seg} x {tps} tokens, "
                f"{len(samples)} sample(s))  [{region}]"
            )
            print("=" * W)

            rows = []
            sel_deep_seen = []
            for si, s in enumerate(samples):
                qt = _query_text(s["question"])
                gold = s["gold_answer"]
                full_text = "\n\n".join(s["docs"])

                # baseline (standard FA-2 prefill)
                tb, ttft_b, dec_b, n_ctx = standard_prefill(
                    model, tok, full_text, qt, MAX_NEW_TOKENS
                )
                ans_b = _short_ans(tb)
                gc.collect()
                torch.cuda.empty_cache()

                # RedKnot
                segs = offline_prefill_segments(
                    model, tok, s["docs"], chunk_size=chunk_size, model_id=model_src
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
                        "q": s["question"],
                        "gold": gold,
                        "base_text": tb,
                        "rk_text": tc,
                        "base_ans": ans_b,
                        "rk_ans": ans_c,
                        "base_f1": f1_max(ans_b, gold),
                        "rk_f1": f1_max(ans_c, gold),
                        "base_em": em_max(ans_b, gold),
                        "rk_em": em_max(ans_c, gold),
                        "base_ttft": ttft_b,
                        "rk_ttft": ttft_c,
                        "base_dec": dec_b,
                        "rk_dec": dec_c,
                        "n_ctx": n_ctx,
                    }
                )

                # Tag this sample for the cross-task Top-N similarity report.
                # Similarity = token-F1 between the RedKnot answer and the
                # baseline answer (1.0 == identical answer span).
                _sim = f1_score(rows[-1]["rk_ans"], rows[-1]["base_ans"])
                _exact = rows[-1]["rk_ans"].strip() == rows[-1]["base_ans"].strip()
                all_rows.append(
                    {
                        **rows[-1],
                        "tag": f"{dataset}@{label}",
                        "sim": _sim,
                        "exact": _exact,
                    }
                )

                # per-sample text dump
                print(f"\n [sample {si}] ctx={rows[-1]['n_ctx']:,} tok")
                print(f"   Q   : {_trunc(s['question'], 80)}")
                print(f"   gold: {_trunc(', '.join(map(str, gold)), 80)}")
                print(
                    f"   base: {_trunc(tb, 70)!r}  -> ans={_trunc(ans_b, 30)!r} "
                    f"F1={rows[-1]['base_f1']:.2f}"
                )
                print(
                    f"   rk  : {_trunc(tc, 70)!r}  -> ans={_trunc(ans_c, 30)!r} "
                    f"F1={rows[-1]['rk_f1']:.2f}"
                )
                print(
                    f"   TTFT: base={ttft_b:.2f}s  rk={ttft_c:.2f}s  "
                    f"speedup={ttft_b / ttft_c:.2f}x"
                )

                del segs
                gc.collect()
                torch.cuda.empty_cache()

            # aggregates
            m = lambda k: sum(r[k] for r in rows) / len(rows)
            sel_deep = (
                sum(sel_deep_seen) / len(sel_deep_seen) if sel_deep_seen else 0.14
            )
            fl = compute_flops(
                d, seg_lens_full, frac_global, sel_deep, ffn_cfg["dense_until"]
            )
            P = 1e15

            print(f"\n {'-' * (W - 2)}")
            tag = f"{dataset}@{label}"
            print(f" {tag} AGGREGATE ({len(rows)} sample(s))")
            print(f" {'-' * (W - 2)}")
            # Quality with dispersion: a single F1 mean from n=4 is near-meaningless,
            # so report sample-std and the 95% CI half-width alongside the mean.
            bf1_m, bf1_sd, bf1_ci = _mean_std_ci([r["base_f1"] for r in rows])
            rf1_m, rf1_sd, rf1_ci = _mean_std_ci([r["rk_f1"] for r in rows])
            bem_m, bem_sd, _ = _mean_std_ci([r["base_em"] for r in rows])
            rem_m, rem_sd, _ = _mean_std_ci([r["rk_em"] for r in rows])
            n_q = len(rows)
            print(
                f"   QUALITY   baseline  F1={bf1_m:.3f}±{bf1_sd:.3f} "
                f"(95%CI ±{bf1_ci:.3f})  EM={bem_m:.3f}±{bem_sd:.3f}"
            )
            print(
                f"             RedKnot   F1={rf1_m:.3f}±{rf1_sd:.3f} "
                f"(95%CI ±{rf1_ci:.3f})  EM={rem_m:.3f}±{rem_sd:.3f}"
            )
            if n_q < 10:
                print(
                    f"             [!] n={n_q}: F1 CI is wide; treat the mean as "
                    f"indicative only (raise REDKNOT_N_SAMPLES for significance)."
                )
            print(
                f"   TTFT      baseline={m('base_ttft'):.2f}s  "
                f"RedKnot={m('rk_ttft'):.2f}s  "
                f"speedup={m('base_ttft') / m('rk_ttft'):.2f}x"
            )
            print(
                f"   DECODE    baseline={m('base_dec'):.1f} tok/s  "
                f"RedKnot={m('rk_dec'):.1f} tok/s"
            )
            print(f"   COMPUTE (prefill FLOPs, P=1e15):")
            for name in ["attn", "ffn", "proj", "total"]:
                dn, hcv = fl[name]
                sv = (1 - hcv / dn) * 100 if dn > 0 else 0.0
                print(
                    f"             {name:6s} dense={dn / P:7.3f}P  "
                    f"RedKnot={hcv / P:7.3f}P  saving={sv:5.1f}%  "
                    f"({(dn / hcv) if hcv > 0 else 0:.2f}x)"
                )
            overall[tag] = {
                "base_f1": m("base_f1"),
                "rk_f1": m("rk_f1"),
                "base_ttft": m("base_ttft"),
                "rk_ttft": m("rk_ttft"),
                "flops_saving": (1 - fl["total"][1] / fl["total"][0]) * 100,
                "n": n_q,
                "base_f1_ci": bf1_ci,
                "rk_f1_ci": rf1_ci,
            }

    # final summary table
    print(f"\n{'=' * W}")
    print(" SUMMARY")
    print("=" * W)
    print(
        f" {'dataset@len':16s} {'n':>3s} {'base F1 (95%CI)':>18s} "
        f"{'rk F1 (95%CI)':>18s} {'base TTFT':>10s} {'rk TTFT':>9s} "
        f"{'speedup':>8s} {'FLOPs save':>11s}"
    )
    for tag, o in overall.items():
        print(
            f" {tag:16s} {o['n']:>3d} "
            f"{o['base_f1']:>10.3f}±{o['base_f1_ci']:<6.3f} "
            f"{o['rk_f1']:>10.3f}±{o['rk_f1_ci']:<6.3f} "
            f"{o['base_ttft']:>9.2f}s {o['rk_ttft']:>8.2f}s "
            f"{o['base_ttft'] / o['rk_ttft']:>7.2f}x {o['flops_saving']:>10.1f}%"
        )
    print("=" * W)

    # ── Top-N examples where RedKnot most closely matches the baseline ──
    # Ranked by answer token-F1 vs baseline (exact-match first), i.e. the
    # clearest evidence that RedKnot's sparse prefill reproduces the dense
    # result. Ties broken by higher gold F1 (correct AND identical).
    TOPN = int(os.environ.get("REDKNOT_TOPN", "10"))
    ranked = sorted(
        all_rows,
        key=lambda r: (r["exact"], r["sim"], r["rk_f1"]),
        reverse=True,
    )[:TOPN]
    print(f"\n{'=' * W}")
    print(f" TOP {len(ranked)} MOST SIMILAR EXAMPLES (RedKnot answer vs baseline)")
    print("=" * W)
    n_exact = sum(1 for r in all_rows if r["exact"])
    print(
        f" {len(all_rows)} samples total | {n_exact} exact-match "
        f"({100 * n_exact / max(len(all_rows), 1):.0f}%) | "
        f"mean answer-similarity={sum(r['sim'] for r in all_rows) / max(len(all_rows), 1):.3f}"
    )
    print("-" * W)
    for i, r in enumerate(ranked, 1):
        mark = "EXACT" if r["exact"] else f"sim={r['sim']:.2f}"
        print(f"\n [{i:2d}] {r['tag']}  ctx={r['n_ctx']:,} tok  [{mark}]")
        print(f"      Q       : {_trunc(r['q'], 84)}")
        print(f"      gold    : {_trunc(', '.join(map(str, r['gold'])), 84)}")
        print(f"      baseline: {_trunc(r['base_ans'], 84)!r}  F1={r['base_f1']:.2f}")
        print(f"      RedKnot : {_trunc(r['rk_ans'], 84)!r}  F1={r['rk_f1']:.2f}")
    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
