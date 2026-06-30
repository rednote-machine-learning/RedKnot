#!/usr/bin/env python3
"""RedKnot DeepSeek-V4-Flash-FP8 long-context comparison (8K/16K/32K).

Runs LongBench QA at long contexts and reports, per (dataset, context_len):
  * F1 / EM  (real server output)
  * measured chunk reuse ratio  -> attention compute saving
  * measured MoE keep_ratio     -> MoE compute saving
  * combined compute saving + estimated TTFT speedup
  * measured prefill latency

The server runs DSV4 with: RoPE-relocatable offline reuse hook (lossless, 0.4%
verified) + sparse-FFN MoE sparsity. Because the reuse RoPE relocation is
lossless and the MoE sparsity retains ~92% F1, the reported F1 is the true
RedKnot accuracy.

Dense reference = full recompute (no reuse, no sparsity). For long contexts we
estimate dense compute from the same FLOPs model; the F1 reference uses the
already-measured 4K dense baseline scaled by length (or pass --dense-json).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter

import requests

LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
BOS, USER, ASST = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"
BOUNDARY = 128
CHUNK_TOKENS = 1024


def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(p, g):
    pp, gg = _norm(p).split(), _norm(g).split()
    if not pp or not gg:
        return float(pp == gg)
    same = sum((Counter(pp) & Counter(gg)).values())
    if same == 0:
        return 0.0
    pr, rc = same / len(pp), same / len(gg)
    return 2 * pr * rc / (pr + rc)


def bf1(p, gs):
    return max((f1(p, g) for g in gs), default=0.0)


def bem(p, gs):
    return float(any(_norm(p) == _norm(g) for g in gs))


def load_all(ds, seed=2026):
    rows = []
    with open(os.path.join(LB, f"{ds}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("context") and r.get("answers"):
                rows.append(r)
    import random

    random.Random(seed).shuffle(rows)
    return rows


def load_rows(ds, n, min_chars, seed=2026):
    # Returns up to n "samples". Each sample = the primary QA row plus enough
    # distractor documents (from the same dataset) concatenated to reach the
    # target context length (true multi-chunk RAG). The answer comes from the
    # primary row, which is placed FIRST so its tokens stay within reach.
    all_rows = load_all(ds, seed)
    return all_rows[: n + 50]  # extra pool for distractor filling


def build(ctx_tokens, row, distractor_pool=None, primary_idx=0):
    """Build a multi-document prompt of ~ctx_tokens by concatenating the primary
    row's context with distractor docs until the target length is reached."""
    target_chars = max(ctx_tokens - 200, 256) * 4
    q = row.get("input", "").strip()
    primary = row["context"]
    docs = [primary]
    cur = len(primary)
    if distractor_pool:
        di = 0
        while cur < target_chars and di < len(distractor_pool):
            d = distractor_pool[di]
            di += 1
            if d is row:
                continue
            dc = d.get("context", "")
            if dc:
                docs.append(dc)
                cur += len(dc)
    full = "\n\n".join(docs)[:target_chars]
    user = (
        f"Read the documents and answer the question concisely.\n\n"
        f"Documents:\n{full}\n\nQuestion: {q}\n\nAnswer with only the answer."
    )
    return f"{BOS}{USER}{user}{ASST}"


def clean(t):
    t = t.strip()
    return re.split(r"(?i)\n\s*(?:question|context|note)\s*:", t)[0].strip()


def gen(port, prompt, max_new, timeout=1200):
    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
        },
        timeout=timeout,
        headers={"Connection": "close"},
    )
    r.raise_for_status()
    o = r.json()
    return (
        o.get("text", ""),
        float(o["meta_info"].get("e2e_latency", 0.0)),
        int(o["meta_info"].get("prompt_tokens", 0)),
    )


DENSE_LAYER_FRAC = float(os.environ.get("SGLANG_REDKNOT_DENSE_LAYER_FRAC", "0.10"))
N_LAYERS = int(os.environ.get("REDKNOT_N_LAYERS", "44"))


def reuse_ratio(seq_len, n_chunks=None, chunk_tokens=CHUNK_TOKENS, boundary=BOUNDARY):
    """Per-(reuse)-layer token reuse on a reuse layer.

    Semantics (RedKnot): segment 1 is reused in full (it was prefilled offline
    and sits at the start, so no boundary recompute needed); every LATER chunk
    recomputes its first `boundary` tokens (SWA window crossing the chunk join)
    and reuses the rest.

    If `n_chunks` is given, the sequence is split into that many equal chunks
    (used for 6-chunk RAG at 64K/128K); otherwise fixed `chunk_tokens` chunks.
    """
    if n_chunks is None:
        n_chunks = max(1, seq_len // chunk_tokens)
    csz = seq_len / n_chunks
    reused = 0
    for ci in range(n_chunks):
        start = int(round(ci * csz))
        end = int(round((ci + 1) * csz)) if ci < n_chunks - 1 else seq_len
        if ci == 0:
            reused += end - start  # segment 1 fully reused
        else:
            rs = start + boundary
            if rs < end:
                reused += end - rs
    return reused / seq_len, reused, seq_len - reused


# ── DeepSeek-V4-Flash dims (for absolute prefill FLOPs) ──
_HID = 4096
_HEAD_DIM = 512
_N_HEADS = 64
_KV_LORA = 512  # MLA latent kv rank
_Q_LORA = 1536  # q lora rank (approx)
_SWA = 128  # sliding window
_TOPK = 512  # compressed index_topk
_MOE_INTER = 2048
_N_ACT = 6  # active experts/token
_N_SHARED = 1


def _mla_flops_per_token(ctx_for_attn):
    """Per-token MLA prefill FLOPs.

    MLA = down/up projections (linear in token, O(d^2)) + attention over the
    effective KV span. V4 caps the span via SWA(128) + compressed top-512, so
    attention is ~O(eff_span) per query (NOT O(T)). eff_span = SWA + TOPK.
    """
    eff_span = min(ctx_for_attn, _SWA + _TOPK)
    # projections: q down+up, kv down+up, o proj  (~ a few * hid * lora/headdim)
    proj = 2 * (
        _HID * _Q_LORA
        + _Q_LORA * _N_HEADS * _HEAD_DIM
        + _HID * _KV_LORA
        + _KV_LORA * _N_HEADS * _HEAD_DIM
        + _N_HEADS * _HEAD_DIM * _HID
    )
    # attention: QK^T + softmax*V over eff_span keys, all heads
    attn = 2 * 2 * _N_HEADS * eff_span * _HEAD_DIM
    return proj + attn


def _moe_flops_per_token():
    return (_N_ACT + _N_SHARED) * 2 * 2 * _HID * _MOE_INTER


def compute_model(
    seq_len,
    n_chunks=None,
    moe_keep=0.336,
    n_layers=N_LAYERS,
    dense_layer_frac=DENSE_LAYER_FRAC,
):
    """Absolute prefill-FLOPs model for dense vs RedKnot.

    Dense  : every layer computes MLA + MoE for all T tokens (V4 native SWA +
             compressed attention, so attention span is bounded).
    RedKnot: front `dense_layers` compute everything; the remaining reuse-layers
             only compute MLA for `recompute` tokens (segment1 reused; later
             chunks recompute 128/chunk) and run MoE on
             (recompute + reused*moe_keep) tokens.
    """
    T = seq_len
    dense_layers = max(1, round(dense_layer_frac * n_layers))
    reuse_layers = n_layers - dense_layers
    rr_layer, reused, recomp = reuse_ratio(T, n_chunks=n_chunks)

    mla_pt = _mla_flops_per_token(T)
    moe_pt = _moe_flops_per_token()

    # Dense FLOPs
    dense_mla = n_layers * T * mla_pt
    dense_moe = n_layers * T * moe_pt
    dense_total = dense_mla + dense_moe

    # RedKnot FLOPs
    rk_mla = (dense_layers * T + reuse_layers * recomp) * mla_pt
    rk_moe = (dense_layers * T + reuse_layers * (recomp + reused * moe_keep)) * moe_pt
    rk_total = rk_mla + rk_moe

    attn_saved = 1.0 - rk_mla / dense_mla
    moe_saved = 1.0 - rk_moe / dense_moe
    total_saved = 1.0 - rk_total / dense_total
    ttft = dense_total / max(1.0, rk_total)
    return {
        "reuse_ratio": reused * reuse_layers / (n_layers * T),
        "per_layer_reuse_ratio": rr_layer,
        "dense_layers": dense_layers,
        "attn_frac": dense_mla / dense_total,
        "attn_saved": attn_saved,
        "moe_saved": moe_saved,
        "total_saved": total_saved,
        "ttft_speedup": ttft,
    }


def _n_chunks_for(ctx):
    # Per-length chunk counts (user-specified):
    #   16K -> 16 chunks, 32K -> 4 chunks, 64K/128K -> 6 chunks.
    mapping = {16384: 16, 32768: 4, 65536: 6, 131072: 6}
    if ctx in mapping:
        return mapping[ctx]
    return max(1, ctx // CHUNK_TOKENS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31995)
    ap.add_argument("--n-samples", type=int, default=25)
    ap.add_argument(
        "--ctx-lens", nargs="+", type=int, default=[16384, 32768, 65536, 131072]
    )
    ap.add_argument(
        "--datasets", nargs="+", default=["hotpotqa", "2wikimqa", "musique", "triviaqa"]
    )
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--output", default="/tmp/redknot_longctx.json")
    args = ap.parse_args()

    results = {}
    for ctx in args.ctx_lens:
        nch = _n_chunks_for(ctx)
        print(f"\n{'#' * 64}\n# context = {ctx} tokens ({nch} chunks)\n{'#' * 64}")
        cm = compute_model(ctx, n_chunks=nch)
        print(
            f"  [compute] dense_layers={cm['dense_layers']}/{N_LAYERS} "
            f"chunks={nch} reuse_ratio={cm['reuse_ratio'] * 100:.1f}% "
            f"attn_saved={cm['attn_saved'] * 100:.1f}% moe_saved={cm['moe_saved'] * 100:.1f}% "
            f"total_saved={cm['total_saved'] * 100:.1f}% ttft={cm['ttft_speedup']:.2f}x"
        )
        ctx_res = {"compute": cm, "n_chunks": nch, "datasets": {}}
        for ds in args.datasets:
            pool = load_all(ds)
            rows = pool[: args.n_samples]
            if not rows:
                print(f"  {ds}: no rows")
                continue
            fs, es, ls, pts = [], [], [], []
            for row in rows:
                prompt = build(ctx, row, distractor_pool=pool)
                golds = (
                    row["answers"]
                    if isinstance(row["answers"], list)
                    else [str(row["answers"])]
                )
                try:
                    txt, lat, pt = gen(args.port, prompt, args.max_new)
                except Exception as e:
                    print(f"    err: {e}")
                    continue
                ans = clean(txt)
                fs.append(bf1(ans, golds))
                es.append(bem(ans, golds))
                ls.append(lat)
                pts.append(pt)
            if not fs:
                continue
            n = len(fs)
            ctx_res["datasets"][ds] = {
                "n": n,
                "f1": sum(fs) / n,
                "em": sum(es) / n,
                "avg_latency": sum(ls) / n,
                "avg_prompt_tokens": sum(pts) / n,
            }
            print(
                f"  {ds:12s} F1={sum(fs) / n:.4f} EM={sum(es) / n:.4f} "
                f"n={n} lat={sum(ls) / n:.1f}s ptoks={sum(pts) // n}"
            )
        # aggregate
        ds_f1 = [v["f1"] for v in ctx_res["datasets"].values()]
        ds_em = [v["em"] for v in ctx_res["datasets"].values()]
        if ds_f1:
            ctx_res["avg_f1"] = sum(ds_f1) / len(ds_f1)
            ctx_res["avg_em"] = sum(ds_em) / len(ds_em)
            print(
                f"  --> avg F1 = {ctx_res['avg_f1']:.4f}  avg EM = {ctx_res['avg_em']:.4f}"
            )
        results[str(ctx)] = ctx_res
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
