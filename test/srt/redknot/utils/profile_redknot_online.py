#!/usr/bin/env python3
"""Profile RedKnot online forward to find the FLOPs->wall-time gap.

Runs the full RedKnot path (offline KV reuse + head-class attention +
3-tier Sparse-FFN) under torch.profiler and aggregates CUDA self-time by
operator category, so we can see whether the bottleneck is:
  * matmul (FFN/proj GEMM)         -> compute bound, expected
  * gather/scatter (index_*)       -> Sparse-FFN selection overhead
  * attention (flash/sdpa)         -> head-class attention
  * elementwise/rope/layernorm     -> memory bound glue

Usage:
  PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 \
    REDKNOT_MODEL_PATH=.../Mistral-7B-Instruct-v0.3 \
    REDKNOT_TOKENS_PER_DOC=7000 \
    python test/srt/redknot/profile_redknot_online.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Mistral-7B-Instruct-v0.3",
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/xiaoyi/RedCacheV0.2/datasets/LongBench/data",
)
HEAD_CFG_JSON = os.environ.get(
    "REDKNOT_HEAD_CFG",
    str(
        Path(__file__).resolve().parent
        / "head_class/mistral-7B_parity_98local_w3000.json"
    ),
)
TOKENS_PER_DOC = int(os.environ.get("REDKNOT_TOKENS_PER_DOC", "7000"))
N_DOCS = int(os.environ.get("REDKNOT_N_DOCS", "4"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "3000"))
# FFN: 0 = full RedKnot 3-tier sparse; >=L = dense (attention-only)
FFN_DENSE_UNTIL = int(os.environ.get("REDKNOT_FFN_DENSE_UNTIL", "4"))


# Map an op name to a coarse category.
def _category(name: str) -> str:
    n = name.lower()
    if any(
        k in n
        for k in (
            "gemm",
            "matmul",
            "addmm",
            "mm",
            "linear",
            "cutlass",
            "cublas",
            "ampere",
            "sgemm",
            "gemv",
        )
    ):
        return "matmul/GEMM"
    if any(
        k in n
        for k in (
            "index_select",
            "index_copy",
            "index_add",
            "gather",
            "scatter",
            "nonzero",
            "masked",
            "take",
        )
    ):
        return "gather/scatter"
    if any(
        k in n
        for k in ("flash", "attention", "sdpa", "scaled_dot", "mem_efficient", "fmha")
    ):
        return "attention"
    if any(k in n for k in ("topk", "sort", "argsort", "cumsum")):
        return "topk/sort"
    if any(k in n for k in ("layer_norm", "rms_norm", "norm")):
        return "norm"
    if any(k in n for k in ("rope", "rotary", "neg", "cat", "roll")):
        return "rope/cat"
    if any(
        k in n
        for k in (
            "mul",
            "add",
            "silu",
            "gelu",
            "sigmoid",
            "elementwise",
            "copy",
            "to_copy",
            "cast",
            "fill",
            "where",
            "softmax",
            "div",
            "sub",
            "exp",
            "vectorized",
        )
    ):
        return "elementwise"
    return "other"


def _load_sample():
    path = os.path.join(LONGBENCH_DIR, "triviaqa.jsonl")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    raw = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    base = raw[0]
    ctx_ids = tok(base["context"], add_special_tokens=False)["input_ids"]
    j = 1
    target = TOKENS_PER_DOC * N_DOCS
    while len(ctx_ids) < target and j < len(raw):
        ctx_ids += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
        j += 1
    ctx_ids = ctx_ids[:target]
    docs = []
    for k in range(0, len(ctx_ids), TOKENS_PER_DOC):
        piece = ctx_ids[k : k + TOKENS_PER_DOC]
        if len(piece) < 64:
            break
        docs.append(tok.decode(piece, skip_special_tokens=True))
    qt = (
        "\n\nAnswer the question based only on the documents above. Give the "
        "shortest exact answer span.\nQuestion: " + base["input"] + "\nAnswer:"
    )
    return tok, docs, qt


def main():
    from transformers import AutoModelForCausalLM

    tok, docs, qt = _load_sample()
    print(
        f"[prof] {len(docs)} docs x {TOKENS_PER_DOC} tok, window={WINDOW}, "
        f"ffn_dense_until={FFN_DENSE_UNTIL}"
    )

    hc = HeadClassConfig.from_json(HEAD_CFG_JSON)
    hc.merge_retrieval_to_global()
    hc.set_local_window(WINDOW)

    L = 32
    if FFN_DENSE_UNTIL >= L:
        sched = SparseFFNSchedule(dense_until=L, mass_thresh=1.0)
        mode = "DENSE-FFN (attention-only)"
    else:
        sched = SparseFFNSchedule(
            dense_until=4,
            mass_thresh=0.2,
            deep_layer_start=20,
            mass_thresh_deep=0.05,
            recent_n=128,
        )
        mode = "3-tier Sparse-FFN (full RedKnot)"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()

    segs = offline_prefill_segments(
        model,
        tok,
        docs,
        chunk_size=max(4096, TOKENS_PER_DOC + 96),
        model_id=MODEL_PATH,
    )

    # warmup (build kernels / autotune)
    run_redknot_offlinekv(
        model,
        tok,
        segments_offline=segs,
        query_text=qt,
        head_cfg=hc,
        max_new_tokens=2,
        kernel="fa3_parallel",
        sparse_ffn_schedule=sched,
        use_compile=False,
    )
    torch.cuda.synchronize()

    print(f"[prof] profiling RedKnot online ({mode}) ...")
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run_redknot_offlinekv(
            model,
            tok,
            segments_offline=segs,
            query_text=qt,
            head_cfg=hc,
            max_new_tokens=2,
            kernel="fa3_parallel",
            sparse_ffn_schedule=sched,
            use_compile=False,
        )
        torch.cuda.synchronize()

    # Aggregate CUDA self-time by category.
    cat_us = defaultdict(float)
    total_us = 0.0
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or getattr(
            evt, "self_cuda_time_total", 0
        )
        if cuda_us <= 0:
            continue
        cat_us[_category(evt.key)] += cuda_us
        total_us += cuda_us

    print("\n" + "=" * 64)
    print(f" RedKnot online CUDA self-time by category ({mode})")
    print("=" * 64)
    for cat, us in sorted(cat_us.items(), key=lambda x: -x[1]):
        print(f"  {cat:18s} {us / 1000:8.2f} ms  {100 * us / total_us:5.1f}%")
    print("-" * 64)
    print(f"  {'TOTAL':18s} {total_us / 1000:8.2f} ms")
    print("=" * 64)

    # Top 12 individual kernels.
    print("\n Top 12 CUDA kernels by self-time:")
    rows = []
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or getattr(
            evt, "self_cuda_time_total", 0
        )
        if cuda_us > 0:
            rows.append((cuda_us, evt.key))
    rows.sort(reverse=True)
    for us, name in rows[:12]:
        print(f"  {us / 1000:8.2f} ms  {100 * us / total_us:5.1f}%  {name[:60]}")


if __name__ == "__main__":
    main()
