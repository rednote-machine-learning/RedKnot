#!/usr/bin/env python3
"""Offline sweet-spot profiler for RedKnot hyper-parameters.

Per the paper, the head-class window, the Sparse-FFN dense/sparse ratio, and
the global/local head split are all things you PROFILE OFFLINE once per model —
not constants. This script sweeps them on a target model + dataset and reports
the accuracy/latency trade-off so you can pick the sweet spot, including an
ADAPTIVE window that scales with the per-request context length (e.g. ctx/2,
ctx/4).

Sweeps (selectable via env):
  1. WINDOW:  fixed {512,1024,2048,4096,8192} and adaptive {ctx/2, ctx/4, ctx/8}
              -> finds the smallest window that keeps F1 == full-attention.
  2. FFN sparsity: dense vs mass_thresh {0.8,0.6,0.4,0.2} with shallow-dense
              tiering -> finds the most aggressive FFN that keeps F1.

For each setting it reports mean F1 (SQuAD), mean TTFT, and the realized
local-head window(s). The "sweet spot" is the most aggressive setting whose F1
is within a tolerance of the full-attention reference.

Usage:
  REDKNOT_MODEL=llama  REDKNOT_DATASET=2wikimqa REDKNOT_N_SAMPLES=5 \\
    REDKNOT_SWEEP=window  CUDA_VISIBLE_DEVICES=0 \\
    python test/srt/redknot/profile_sweet_spot.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "test" / "srt" / "redknot"))

from benchmark_RedKnot_Llama_RAG import (  # noqa: E402  (reuse helpers)
    _load_longbench,
    _query_text,
    _short_ans,
    f1_max,
)
from sglang.srt.layers.attention.redknot import (  # noqa: E402
    HeadClassConfig,
    SparseFFNSchedule,
    offline_prefill_segments,
    run_redknot_offlinekv,
)

MODELS = {
    "llama": (
        "/mnt/tidal-alsh01/dataset/redone/096/models/Llama-3.3-70B-Instruct",
        str(
            Path(__file__).resolve().parent
            / "head_class/llama-70B_optimal_g15_lf_ret.json"
        ),
        "flash_attention_2",
    ),
    "qwen": (
        "/mnt/tidal-alsh01/dataset/redone/096/models/Qwen3-32B",
        str(
            Path(__file__).resolve().parent
            / "head_class/qwen3-32B_optimal_g15_lf_ret.json"
        ),
        "sdpa",
    ),
}

MODEL_KEY = os.environ.get("REDKNOT_MODEL", "llama")
DATASET = os.environ.get("REDKNOT_DATASET", "2wikimqa")
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "5"))
SWEEP = os.environ.get("REDKNOT_SWEEP", "window")  # window | ffn
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "20"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
F1_TOL = float(os.environ.get("REDKNOT_F1_TOL", "0.02"))


def _load_cfg(path):
    hc = HeadClassConfig.from_json(path)
    hc.merge_retrieval_to_global()
    return hc


def _orig_window(hc):
    """The current (config) local-head window value to be overridden."""
    vals = {d for row in hc.head_max_distance for d in row if d > 0}
    return max(vals) if vals else 8192


def _set_window(hc, new_w, orig_w):
    hc.head_max_distance = [
        [new_w if d == orig_w else d for d in row] for row in hc.head_max_distance
    ]
    return hc


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path, cfg_path, attn_impl = MODELS[MODEL_KEY]
    W = 96
    print("=" * W)
    print(
        f" SWEET-SPOT PROFILE  model={MODEL_KEY}  dataset={DATASET}  "
        f"sweep={SWEEP}  n={N_SAMPLES}"
    )
    print("=" * W)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=qc,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).eval()

    samples = _load_longbench(DATASET, tok, N_SAMPLES)
    for s in samples:
        s["ctx_tok"] = len(
            tok("\n\n".join(s["docs"]), add_special_tokens=False)["input_ids"]
        )
    ctx_list = [s["ctx_tok"] for s in samples]
    print(
        f" ctx tokens: min={min(ctx_list)} med={sorted(ctx_list)[len(ctx_list) // 2]} "
        f"max={max(ctx_list)}"
    )
    orig_w = _orig_window(_load_cfg(cfg_path))
    print(f" config local window (to override): {orig_w}")

    def run(window_fn, sched, label):
        f1s, tts, ws = [], [], []
        for s in samples:
            hc = _load_cfg(cfg_path)
            w = window_fn(s["ctx_tok"])
            _set_window(hc, w, orig_w)
            qt = _query_text(s["question"])
            segs = offline_prefill_segments(
                model,
                tok,
                s["docs"],
                chunk_size=max(4096, CHUNK + 96),
                model_id=model_path,
            )
            _, t, _, ttft = run_redknot_offlinekv(
                model,
                tok,
                segments_offline=segs,
                query_text=qt,
                head_cfg=hc,
                max_new_tokens=MAX_NEW,
                kernel="fa3_parallel",
                sparse_ffn_schedule=sched,
                use_compile=False,
            )
            f1s.append(f1_max(_short_ans(t), s["golds"]))
            tts.append(ttft)
            ws.append(w)
            del segs
            gc.collect()
            torch.cuda.empty_cache()
        f1 = sum(f1s) / len(f1s)
        tt = sum(tts) / len(tts)
        print(
            f"  {label:24s} F1={f1:.3f}  ttft={tt:.2f}s  "
            f"win[min={min(ws)},max={max(ws)}]"
        )
        return f1, tt

    dense = SparseFFNSchedule(
        dense_until=model.config.num_hidden_layers, mass_thresh=1.0
    )

    if SWEEP == "window":
        print("\n -- WINDOW sweep (dense FFN, isolate attention) --")
        ref_f1, _ = run(lambda c: orig_w, dense, "full (config window)")
        results = []
        for w in [4096, 2048, 1024, 512]:
            f1, tt = run(lambda c, w=w: min(w, c), dense, f"fixed-{w}")
            results.append((f"fixed-{w}", w, f1, tt))
        for div in [2, 4, 8]:
            f1, tt = run(
                lambda c, d=div: max(256, c // d), dense, f"adaptive ctx/{div}"
            )
            results.append((f"ctx/{div}", div, f1, tt))
        print(f"\n  Reference F1 (full window) = {ref_f1:.3f}  tol={F1_TOL}")
        ok = [r for r in results if r[2] >= ref_f1 - F1_TOL]
        if ok:
            best = min(ok, key=lambda r: r[3])  # fastest among accuracy-OK
            print(f"  SWEET SPOT: {best[0]}  (F1={best[2]:.3f}, ttft={best[3]:.2f}s)")
        else:
            print("  No setting within tolerance — model needs full window.")

    elif SWEEP == "ffn":
        print("\n -- FFN sparsity sweep (config window) --")
        ref_f1, _ = run(lambda c: orig_w, dense, "dense FFN (mass=1.0)")
        n_layers = model.config.num_hidden_layers
        results = []
        for mass in [0.8, 0.6, 0.4, 0.2]:
            sched = SparseFFNSchedule(
                dense_until=max(1, n_layers - 20),
                mass_thresh=mass,
                deep_layer_start=n_layers,
                mass_thresh_deep=None,
                recent_n=512,
            )
            f1, tt = run(lambda c: orig_w, sched, f"mass={mass} (deep-only)")
            results.append((f"mass={mass}", mass, f1, tt))
        print(f"\n  Reference F1 (dense FFN) = {ref_f1:.3f}  tol={F1_TOL}")
        ok = [r for r in results if r[2] >= ref_f1 - F1_TOL]
        if ok:
            best = min(ok, key=lambda r: r[1])  # most aggressive (lowest mass)
            print(f"  SWEET SPOT: {best[0]}  (F1={best[2]:.3f}, ttft={best[3]:.2f}s)")
        else:
            print("  No sparse setting within tolerance — keep dense FFN.")

    print("=" * W)


if __name__ == "__main__":
    main()
