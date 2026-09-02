#!/usr/bin/env python3
"""RAG smoke benchmark for DeepSeek V4 Flash / RedKnot MLA.

The default path uses SGLang Engine twice: baseline ``attention_backend=dsv4``
and RedKnot ``attention_backend=redknot_mla``. The RedKnot path keeps the
physical DeepSeek V4 MLA/FlashMLA cache and applies RedKnot at logical attention
head granularity inside the MLA backend.

Default model is the local DeepSeek-V4-Flash checkpoint because it is small
enough to be the practical smoke target. Override with ``REDKNOT_MODEL_PATH`` for
DeepSeek-V4-Pro or another compatible checkpoint.

Examples:
  # One-sample RAG smoke with standard vs RedKnot MLA output comparison.
  CUDA_VISIBLE_DEVICES=0 REDKNOT_N_SAMPLES=1 REDKNOT_LENGTHS=8K \
    python test/srt/redknot/benchmark_RedKnot_Deepseekv4_RAG.py

  # Tune the logical-head MLA policy.
  REDKNOT_MLA_LOCAL_WINDOW=256 REDKNOT_MLA_GLOBAL_HEAD_STRIDE=8 \
    python test/srt/redknot/benchmark_RedKnot_Deepseekv4_RAG.py
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL_PATH = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash",
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DATASETS = [
    x.strip()
    for x in os.environ.get(
        "REDKNOT_DATASETS", "hotpotqa,2wikimqa,musique,multifieldqa_en"
    ).split(",")
    if x.strip()
]
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "1"))
MAX_NEW_TOKENS = int(os.environ.get("REDKNOT_MAX_NEW", "32"))
CHUNK_TOKENS = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
SEED = int(os.environ.get("REDKNOT_SEED", "2026"))
ENABLE_REUSE = os.environ.get("REDKNOT_ENABLE_REUSE", "0") == "1"
RUNTIME = os.environ.get("REDKNOT_RUNTIME", "sglang").lower()
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "0") == "1"
REUSE_KERNEL = os.environ.get("REDKNOT_KERNEL", "fa3_parallel")
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DTYPE = os.environ.get("REDKNOT_DTYPE", "bf16").lower()
DRY_RUN = os.environ.get("REDKNOT_DRY_RUN", "0") == "1"
TP_SIZE = int(os.environ.get("REDKNOT_TP_SIZE", "1"))
MAX_TOTAL_TOKENS = int(os.environ.get("REDKNOT_MAX_TOTAL_TOKENS", "0"))
DISABLE_CUDA_GRAPH = os.environ.get("REDKNOT_DISABLE_CUDA_GRAPH", "0") == "1"
SKIP_SERVER_WARMUP = os.environ.get("REDKNOT_SKIP_SERVER_WARMUP", "0") == "1"
MOE_RUNNER_BACKEND = os.environ.get("REDKNOT_MOE_RUNNER_BACKEND", "")
MEM_FRACTION_STATIC = os.environ.get("REDKNOT_MEM_FRACTION_STATIC", "")
MLA_DENSE_PREFIX = int(os.environ.get("REDKNOT_MLA_DENSE_PREFIX_LAYERS", "2"))
MLA_LOCAL_WINDOW = int(
    os.environ.get(
        "REDKNOT_MLA_LOCAL_WINDOW", os.environ.get("REDKNOT_LOCAL_WINDOW", "128")
    )
)
MLA_GLOBAL_HEAD_STRIDE = int(os.environ.get("REDKNOT_MLA_GLOBAL_HEAD_STRIDE", "8"))
MLA_GLOBAL_LAYER_STRIDE = int(os.environ.get("REDKNOT_MLA_GLOBAL_LAYER_STRIDE", "0"))

# Offline MLA head-locality profiling (analysis-before-compression step).
PROFILE = os.environ.get("REDKNOT_MLA_PROFILE", "0") == "1"
PROFILE_OUT = os.environ.get(
    "REDKNOT_MLA_PROFILE_OUT",
    str(Path(__file__).resolve().parent / "head_class" / "dsv4_mla_head_config.json"),
)
PROFILE_COVERAGE = float(os.environ.get("REDKNOT_MLA_PROFILE_COVERAGE", "0.95"))
PROFILE_SAMPLE_Q = int(os.environ.get("REDKNOT_MLA_PROFILE_SAMPLE_Q", "256"))
PROFILE_GLOBAL_RATIO = float(os.environ.get("REDKNOT_MLA_PROFILE_GLOBAL_RATIO", "0.5"))
PROFILE_WINDOW_SAFETY = float(
    os.environ.get("REDKNOT_MLA_PROFILE_WINDOW_SAFETY", "1.5")
)

_LEN_MAP = {
    "4K": 4000,
    "8K": 8000,
    "16K": 16000,
    "24K": 24000,
    "32K": 32000,
    "40K": 40000,
    "64K": 64000,
}
LENGTHS = [
    x.strip()
    for x in os.environ.get("REDKNOT_LENGTHS", "8K").split(",")
    if x.strip() in _LEN_MAP
]


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


def f1_max(pred: str, golds: Iterable[str]) -> float:
    return max((f1_score(pred, g) for g in golds), default=0.0)


def em_max(pred: str, golds: Iterable[str]) -> float:
    return max((float(_normalize(pred) == _normalize(g)) for g in golds), default=0.0)


def _short_ans(text: str) -> str:
    text = text or ""
    if not text.strip():
        return ""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"(?i)^\s*(the answer is|answer)\b[:\s]*", "", text, count=1)
    text = re.split(r"(?i)(?:\n\s*question\s*:|\n\s*q\s*:|<\|)", text)[0]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cand = (lines[0] if lines else text.strip()).strip().strip('"').strip("'")
    return re.sub(r"\s*[.。]\s*$", "", cand)


def _trunc(s: str, n: int = 72) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _query_text(question: str) -> str:
    return (
        "\n\nAnswer the question using only the documents above. "
        "Return the shortest exact answer span only, with no explanation.\n"
        f"Question: {question}\nAnswer:"
    )


def _chunk_token_ids(ids: list[int], tok, chunk_tokens: int) -> list[str]:
    docs = []
    for start in range(0, len(ids), chunk_tokens):
        piece = ids[start : start + chunk_tokens]
        if len(piece) < 64:
            break
        docs.append(tok.decode(piece, skip_special_tokens=True))
    return docs


def _load_longbench_padded(ds_name: str, tok, n_samples: int, target_tokens: int):
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("input") and row.get("context") and row.get("answers"):
                raw.append(row)
    rng = random.Random(SEED)
    rng.shuffle(raw)

    out = []
    n = len(raw)
    for i, base in enumerate(raw):
        if len(out) >= n_samples:
            break
        ctx_ids = tok(base["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % n
        while len(ctx_ids) < target_tokens and j != i:
            extra = tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            ctx_ids.extend(extra)
            j = (j + 1) % n
        ctx_ids = ctx_ids[:target_tokens]
        docs = _chunk_token_ids(ctx_ids, tok, CHUNK_TOKENS)
        if len(docs) < 2:
            continue
        out.append(
            {
                "question": base["input"],
                "golds": [str(x) for x in base["answers"]],
                "docs": docs,
            }
        )
    return out


@torch.no_grad()
def standard_prefill(model, tok, full_text: str, query_text: str):
    device = getattr(model, "device", None) or next(model.parameters()).device
    ids = tok(full_text + query_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    next_id = out.logits[0, -1, :].argmax().view(1, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    past = out.past_key_values
    generated = [int(next_id[0, 0])]
    t1 = time.perf_counter()
    for _ in range(MAX_NEW_TOKENS - 1):
        og = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = og.past_key_values
        next_id = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(next_id[0, 0])
        generated.append(tid)
        if tid == tok.eos_token_id:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_t = max(time.perf_counter() - t1, 1e-3)
    return (
        tok.decode(generated, skip_special_tokens=True),
        ttft,
        len(generated) / decode_t,
        ids.shape[1],
    )


def _model_dims(cfg):
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    return {
        "L": int(cfg.num_hidden_layers),
        "hidden": int(cfg.hidden_size),
        "Hq": int(cfg.num_attention_heads),
        "Hkv": int(getattr(cfg, "num_key_value_heads", 1)),
        "D": head_dim,
        "q_lora": int(getattr(cfg, "q_lora_rank", 0) or 0),
        "o_lora": int(getattr(cfg, "o_lora_rank", 0) or 0),
        "moe_inter": int(
            getattr(cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", 0))
        ),
        "experts_per_tok": int(getattr(cfg, "num_experts_per_tok", 1)),
        "shared_experts": int(getattr(cfg, "n_shared_experts", 0)),
    }


def _proj_flops_per_token(d):
    if d["q_lora"] and d["o_lora"]:
        q = 2.0 * d["hidden"] * d["q_lora"] + 2.0 * d["q_lora"] * d["Hq"] * d["D"]
        kv = 2.0 * d["hidden"] * d["D"]
        o = 2.0 * d["Hq"] * d["D"] * d["o_lora"] + 2.0 * d["o_lora"] * d["hidden"]
        return q + kv + o
    return (
        2.0 * d["hidden"] * (d["Hq"] + 2 * d["Hkv"]) * d["D"]
        + 2.0 * d["Hq"] * d["D"] * d["hidden"]
    )


def _ffn_flops_per_token(d):
    if d["moe_inter"]:
        active = max(1, d["experts_per_tok"] + d["shared_experts"])
        return active * 6.0 * d["hidden"] * d["moe_inter"]
    return 0.0


def _attn_dense(d, T):
    return d["L"] * d["Hq"] * 4.0 * d["D"] * (T * (T + 1) / 2.0)


def _attn_hc(d, T, frac_global, window):
    h_global = d["Hq"] * frac_global
    h_local = d["Hq"] - h_global
    full = T * (T + 1) / 2.0
    local = T * min(window, T)
    return d["L"] * 4.0 * d["D"] * (h_global * full + h_local * local)


def compute_flops(d, T, frac_global, ffn_selected, dense_until, window):
    proj = d["L"] * T * _proj_flops_per_token(d)
    ffn_dense = d["L"] * T * _ffn_flops_per_token(d)
    dense_layers = min(dense_until, d["L"])
    ffn_hc = (
        dense_layers * T * _ffn_flops_per_token(d)
        + (d["L"] - dense_layers) * T * _ffn_flops_per_token(d) * ffn_selected
    )
    attn_d = _attn_dense(d, T)
    attn_h = _attn_hc(d, T, frac_global, window)
    return {
        "attn": (attn_d, attn_h),
        "ffn": (ffn_dense, ffn_hc),
        "proj": (proj, proj),
        "total": (proj + ffn_dense + attn_d, proj + ffn_hc + attn_h),
    }


def _looks_redknot_hf_compatible(model) -> tuple[bool, str]:
    base = model.model if hasattr(model, "model") else model
    layers = getattr(base, "layers", None)
    if not layers:
        return False, "model.model.layers is missing"
    attn = getattr(layers[0], "self_attn", None)
    if attn is None:
        return False, "layers[0].self_attn is missing"
    required = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "num_key_value_groups",
        "head_dim",
    ]
    missing = [name for name in required if not hasattr(attn, name)]
    if missing:
        return False, "missing attention attrs: " + ", ".join(missing)
    return True, "compatible"


def _make_default_head_config(cfg):
    from sglang.srt.layers.attention.redknot import (
        DeepSeekV4MLAHeadConfig,
        HeadClassConfig,
        is_deepseek_v4_mla_config,
    )

    n_layers = int(cfg.num_hidden_layers)
    n_kv = int(getattr(cfg, "num_key_value_heads", 1))
    window = int(os.environ.get("REDKNOT_LOCAL_WINDOW", "4096"))
    dense_prefix = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "2"))
    global_stride = max(1, int(os.environ.get("REDKNOT_GLOBAL_LAYER_STRIDE", "8")))
    if is_deepseek_v4_mla_config(cfg):
        return DeepSeekV4MLAHeadConfig.from_model_config(
            cfg,
            dense_prefix_layers=dense_prefix,
            local_window=window,
            global_head_stride=max(
                1, int(os.environ.get("REDKNOT_GLOBAL_HEAD_STRIDE", "8"))
            ),
            global_layer_stride=int(os.environ.get("REDKNOT_GLOBAL_LAYER_STRIDE", "0")),
        )
    head_class = []
    head_distance = []
    for layer in range(n_layers):
        if layer < dense_prefix or layer % global_stride == 0:
            row = ["global"] * n_kv
            dist = [-1] * n_kv
        else:
            row = ["local"] * n_kv
            dist = [window] * n_kv
        head_class.append(row)
        head_distance.append(dist)
    return HeadClassConfig(
        head_class=head_class,
        head_max_distance=head_distance,
        num_layers=n_layers,
        num_kv_heads=n_kv,
        dense_prefix_layers=dense_prefix,
        local_default_window=window,
    )


def _load_head_config(cfg):
    path = os.environ.get("REDKNOT_HEAD_CFG", "")
    if path:
        from sglang.srt.layers.attention.redknot import (
            DeepSeekV4MLAHeadConfig,
            HeadClassConfig,
            is_deepseek_v4_mla_config,
        )

        if is_deepseek_v4_mla_config(cfg):
            return DeepSeekV4MLAHeadConfig.from_json(path)
        hc = HeadClassConfig.from_json(path)
        hc.merge_retrieval_to_global()
        return hc
    return _make_default_head_config(cfg)


def _make_sparse_ffn_schedule():
    from sglang.srt.layers.attention.redknot import SparseFFNSchedule

    dense_until = int(os.environ.get("REDKNOT_FFN_DENSE_UNTIL", "4"))
    mass = float(os.environ.get("REDKNOT_FFN_MASS", "0.30"))
    deep_start = int(os.environ.get("REDKNOT_FFN_DEEP_START", "24"))
    mass_deep = float(os.environ.get("REDKNOT_FFN_MASS_DEEP", "0.10"))
    recent_n = int(os.environ.get("REDKNOT_FFN_RECENT_N", "256"))
    return SparseFFNSchedule(
        dense_until=dense_until,
        mass_thresh=mass,
        deep_layer_start=deep_start,
        mass_thresh_deep=mass_deep,
        recent_n=recent_n,
    )


@torch.no_grad()
def redknot_prefill(model, tok, docs: list[str], query_text: str):
    from sglang.srt.layers.attention.redknot import (
        offline_prefill_segments,
        run_redknot_offlinekv,
    )

    hc = _load_head_config(model.config)
    if hc.__class__.__name__ == "DeepSeekV4MLAHeadConfig":
        raise RuntimeError(
            "DeepSeek V4 MLA needs a FlashMLA-native RedKnot prefill path; "
            "the generic HF offline-KV driver expects materialized k_proj/v_proj KV."
        )
    sched = _make_sparse_ffn_schedule()
    segs = offline_prefill_segments(
        model,
        tok,
        docs,
        chunk_size=max(4096, CHUNK_TOKENS + 96),
        model_id=MODEL_PATH,
    )
    stats = []
    t0 = time.perf_counter()
    _, text, _, ttft = run_redknot_offlinekv(
        model,
        tok,
        segments_offline=segs,
        query_text=query_text,
        head_cfg=hc,
        max_new_tokens=MAX_NEW_TOKENS,
        kernel=REUSE_KERNEL,
        sparse_ffn_schedule=sched,
        sparse_ffn_stats=stats,
        use_compile=USE_COMPILE,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = max(time.perf_counter() - t0, 1e-3)
    n_dec = len(tok(text, add_special_tokens=False)["input_ids"]) or 1
    return text, ttft, n_dec / max(total - ttft, 1e-3), stats


def _load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM

    tok = _load_tokenizer()

    kwargs = {
        "device_map": DEVICE_MAP,
        "trust_remote_code": True,
    }
    if DTYPE == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif DTYPE == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif DTYPE == "fp32":
        kwargs["torch_dtype"] = torch.float32
    print(f"Loading {MODEL_PATH} (dtype={DTYPE}, device_map={DEVICE_MAP})...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs).eval()
    return model, tok


def _engine_kwargs(attention_backend: str, *, sparse_ffn: bool):
    kwargs = {
        "model_path": MODEL_PATH,
        "attention_backend": attention_backend,
        "tp_size": TP_SIZE,
        "random_seed": SEED,
    }
    if MAX_TOTAL_TOKENS > 0:
        kwargs["max_total_tokens"] = MAX_TOTAL_TOKENS
    if DISABLE_CUDA_GRAPH:
        kwargs["disable_cuda_graph"] = True
    if SKIP_SERVER_WARMUP:
        kwargs["skip_server_warmup"] = True
    if MOE_RUNNER_BACKEND:
        kwargs["moe_runner_backend"] = MOE_RUNNER_BACKEND
    if MEM_FRACTION_STATIC:
        kwargs["mem_fraction_static"] = float(MEM_FRACTION_STATIC)
    if attention_backend == "redknot_mla":
        kwargs.update(
            {
                "redknot_sparse_ffn_enable": sparse_ffn,
                "redknot_sparse_ffn_dense_until": int(
                    os.environ.get("REDKNOT_FFN_DENSE_UNTIL", "4")
                ),
                "redknot_sparse_ffn_mass_thresh": float(
                    os.environ.get("REDKNOT_FFN_MASS", "0.30")
                ),
                "redknot_sparse_ffn_deep_start": int(
                    os.environ.get("REDKNOT_FFN_DEEP_START", "24")
                ),
                "redknot_sparse_ffn_mass_thresh_deep": float(
                    os.environ.get("REDKNOT_FFN_MASS_DEEP", "0.10")
                ),
                "redknot_sparse_ffn_recent_n": int(
                    os.environ.get("REDKNOT_FFN_RECENT_N", "256")
                ),
                "redknot_mla_dense_prefix_layers": MLA_DENSE_PREFIX,
                "redknot_mla_local_window": MLA_LOCAL_WINDOW,
                "redknot_mla_global_head_stride": MLA_GLOBAL_HEAD_STRIDE,
                "redknot_mla_global_layer_stride": MLA_GLOBAL_LAYER_STRIDE,
            }
        )
        head_cfg = os.environ.get("REDKNOT_HEAD_CFG", "")
        if head_cfg:
            kwargs["redknot_head_config_path"] = head_cfg
    return kwargs


def _profile_engine_kwargs():
    """Engine kwargs for the offline MLA head-locality analysis run."""
    kwargs = {
        "model_path": MODEL_PATH,
        # Profiling hooks the dsv4 non-fused attention path.
        "attention_backend": "dsv4",
        "tp_size": TP_SIZE,
        "random_seed": SEED,
        "redknot_mla_profile_enable": True,
        "redknot_mla_profile_out": PROFILE_OUT,
        "redknot_mla_profile_coverage": PROFILE_COVERAGE,
        "redknot_mla_profile_sample_queries": PROFILE_SAMPLE_Q,
        "redknot_mla_profile_global_window_ratio": PROFILE_GLOBAL_RATIO,
        "redknot_mla_profile_window_safety": PROFILE_WINDOW_SAFETY,
        "redknot_mla_dense_prefix_layers": MLA_DENSE_PREFIX,
    }
    if MAX_TOTAL_TOKENS > 0:
        kwargs["max_total_tokens"] = MAX_TOTAL_TOKENS
    if DISABLE_CUDA_GRAPH:
        kwargs["disable_cuda_graph"] = True
    if SKIP_SERVER_WARMUP:
        kwargs["skip_server_warmup"] = True
    if MOE_RUNNER_BACKEND:
        kwargs["moe_runner_backend"] = MOE_RUNNER_BACKEND
    if MEM_FRACTION_STATIC:
        kwargs["mem_fraction_static"] = float(MEM_FRACTION_STATIC)
    return kwargs


def _run_profile():
    """Run the analysis-before-compression step and export a head config JSON.

    Prefills one long single-sequence context with the profiler enabled, then
    reads back the exported ``DeepSeekV4MLAHeadConfig`` JSON and prints a
    per-layer summary of global vs local heads and the local window sizes.
    """
    import sglang as sgl

    tok = _load_tokenizer()
    # Build the longest available single context to make distance stats robust.
    length_label = LENGTHS[-1] if LENGTHS else "8K"
    target = _LEN_MAP[length_label]
    sample = None
    for ds_name in DATASETS:
        cand = _load_longbench_padded(ds_name, tok, 1, target)
        if cand:
            sample = cand[0]
            break
    if sample is None:
        print("[profile] no usable sample found; aborting")
        return

    prompt = "\n\n".join(sample["docs"]) + _query_text(sample["question"])
    n_ctx = len(tok(prompt, add_special_tokens=False)["input_ids"])

    W = 108
    print("=" * W)
    print(" REDKNOT MLA HEAD-LOCALITY PROFILE (analysis before compression)")
    print(f" Model: {MODEL_PATH}")
    print(
        f" ctx≈{n_ctx:,} tok | coverage={PROFILE_COVERAGE} "
        f"global_ratio={PROFILE_GLOBAL_RATIO} window_safety={PROFILE_WINDOW_SAFETY}"
    )
    print(f" out: {PROFILE_OUT}")
    print("=" * W)

    os.makedirs(os.path.dirname(PROFILE_OUT) or ".", exist_ok=True)
    engine = sgl.Engine(**_profile_engine_kwargs())
    try:
        # max_new_tokens=1: we only need the prefill pass for analysis.
        engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 1})
    finally:
        engine.shutdown()

    if not os.path.exists(PROFILE_OUT):
        print(f"[profile] expected head config not found at {PROFILE_OUT}")
        return

    from sglang.srt.layers.attention.redknot import DeepSeekV4MLAHeadConfig

    hc = DeepSeekV4MLAHeadConfig.from_json(PROFILE_OUT)
    summary = hc.summary()
    print("\n head classification summary:")
    print(f"   {summary}")
    print("\n per-layer (global / local / dense, local window min..max):")
    from sglang.srt.layers.attention.redknot.head_config import (
        HEAD_DENSE,
        HEAD_GLOBAL,
        HEAD_LOCAL,
    )

    for layer in range(hc.num_layers):
        g = l = d = 0
        wins = []
        for head in range(hc.num_attention_heads):
            t = hc.head_class[layer][head]
            if t == HEAD_GLOBAL:
                g += 1
            elif t == HEAD_LOCAL:
                l += 1
                wins.append(hc.head_max_distance[layer][head])
            else:
                d += 1
        wtxt = f"{min(wins)}..{max(wins)}" if wins else "-"
        print(
            f"   layer {layer:3d}: global={g:3d} local={l:3d} dense={d:3d} win={wtxt}"
        )
    print("=" * W)
    print(f" head config written to: {PROFILE_OUT}")
    print(" Use it via REDKNOT_HEAD_CFG=<path> with attention_backend=redknot_mla")
    print("=" * W)


def _engine_generate_all(
    prompts: list[str], attention_backend: str, *, sparse_ffn: bool
):
    import sglang as sgl

    engine = sgl.Engine(**_engine_kwargs(attention_backend, sparse_ffn=sparse_ffn))
    sampling_params = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
    try:
        return [engine.generate(p, sampling_params) for p in prompts]
    finally:
        engine.shutdown()


def _run_sglang_engine_benchmark():
    tok = _load_tokenizer()
    tasks = []
    for ds_name in DATASETS:
        for length_label in LENGTHS:
            target = _LEN_MAP[length_label]
            samples = _load_longbench_padded(ds_name, tok, N_SAMPLES, target)
            if not samples:
                print(f"\n[skip] {ds_name}@{length_label}: no usable samples")
                continue
            for sample in samples:
                query = _query_text(sample["question"])
                prompt = "\n\n".join(sample["docs"]) + query
                tasks.append((f"{ds_name}@{length_label}", sample, prompt))

    W = 108
    print("=" * W)
    print(
        " BENCHMARK: DeepSeek V4 Flash Standard(dsv4) vs RedKnot(redknot_mla + sparse FFN)"
    )
    print(f" Model: {MODEL_PATH}")
    print(f" tasks={len(tasks)} tp={TP_SIZE} samples/dataset={N_SAMPLES}")
    print(
        " RedKnot MLA policy: "
        f"dense_prefix={MLA_DENSE_PREFIX} local_window={MLA_LOCAL_WINDOW} "
        f"global_head_stride={MLA_GLOBAL_HEAD_STRIDE} "
        f"global_layer_stride={MLA_GLOBAL_LAYER_STRIDE}"
    )
    print("=" * W)
    if not tasks:
        return

    prompts = [p for _, _, p in tasks]
    print("Running baseline engine: attention_backend=dsv4")
    t0 = time.perf_counter()
    base_out = _engine_generate_all(prompts, "dsv4", sparse_ffn=False)
    base_time = time.perf_counter() - t0

    print("Running RedKnot engine: attention_backend=redknot_mla, sparse FFN enabled")
    t0 = time.perf_counter()
    rk_out = _engine_generate_all(prompts, "redknot_mla", sparse_ffn=True)
    rk_time = time.perf_counter() - t0

    rows = []
    for i, ((task_name, sample, prompt), b, r) in enumerate(
        zip(tasks, base_out, rk_out)
    ):
        base_text = b.get("text", "") if isinstance(b, dict) else str(b)
        rk_text = r.get("text", "") if isinstance(r, dict) else str(r)
        base_ans = _short_ans(base_text)
        rk_ans = _short_ans(rk_text)
        row = {
            "task": task_name,
            "base_f1": f1_max(base_ans, sample["golds"]),
            "base_em": em_max(base_ans, sample["golds"]),
            "rk_f1": f1_max(rk_ans, sample["golds"]),
            "rk_em": em_max(rk_ans, sample["golds"]),
        }
        rows.append(row)
        print(
            f"\n [sample {i} {task_name}] ctx≈{len(tok(prompt, add_special_tokens=False)['input_ids']):,} tok"
        )
        print(f"   Q       : {_trunc(sample['question'], 88)}")
        print(f"   gold    : {sample['golds'][0] if sample['golds'] else ''}")
        print(
            f"   standard: {_trunc(base_text, 96)!r} -> {_trunc(base_ans, 40)!r} F1={row['base_f1']:.2f}"
        )
        print(
            f"   redknot : {_trunc(rk_text, 96)!r} -> {_trunc(rk_ans, 40)!r} F1={row['rk_f1']:.2f}"
        )

    print(f"\n{'=' * W}")
    print(" SUMMARY")
    print("=" * W)
    for task_name in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task_name]
        avg = lambda key: sum(r[key] for r in sub) / len(sub)
        print(
            f" {task_name:28s} std F1={avg('base_f1'):.3f} EM={avg('base_em'):.3f} | "
            f"rk F1={avg('rk_f1'):.3f} EM={avg('rk_em'):.3f} dF1={avg('rk_f1') - avg('base_f1'):+.3f}"
        )
    print(f" elapsed: standard={base_time:.1f}s redknot={rk_time:.1f}s")
    print("=" * W)


def main():
    if PROFILE and not DRY_RUN:
        _run_profile()
        return

    if RUNTIME == "sglang" and not DRY_RUN:
        _run_sglang_engine_benchmark()
        return

    W = 108
    print("=" * W)
    print(" BENCHMARK: DeepSeek V4 RAG smoke + optional RedKnot offline-KV reuse")
    print(f" Model: {MODEL_PATH}")
    print(
        " RAG: LongBench padded contexts | "
        f"datasets={','.join(DATASETS)} | lengths={','.join(LENGTHS)} | samples={N_SAMPLES}"
    )
    print(f" RedKnot reuse enabled: {ENABLE_REUSE}")
    print("=" * W)

    if DRY_RUN:
        tok = _load_tokenizer()
        for ds_name in DATASETS:
            for length_label in LENGTHS:
                samples = _load_longbench_padded(
                    ds_name, tok, N_SAMPLES, _LEN_MAP[length_label]
                )
                if not samples:
                    print(f" [dry-run] {ds_name}@{length_label}: no usable samples")
                    continue
                first = samples[0]
                doc_lens = [
                    len(tok(doc, add_special_tokens=False)["input_ids"])
                    for doc in first["docs"]
                ]
                print(
                    f" [dry-run] {ds_name}@{length_label}: samples={len(samples)} "
                    f"docs={len(first['docs'])} doc_tokens={doc_lens[:4]} "
                    f"question={_trunc(first['question'], 64)!r}"
                )
        return

    model, tok = _load_model_and_tokenizer()
    from sglang.srt.layers.attention.redknot import (
        deepseek_v4_mla_cache_descriptor,
        is_deepseek_v4_mla_config,
    )

    dims = _model_dims(model.config)
    print(
        f" Config: layers={dims['L']} hidden={dims['hidden']} Hq={dims['Hq']} "
        f"Hkv={dims['Hkv']} D={dims['D']} q_lora={dims['q_lora']} o_lora={dims['o_lora']}"
    )
    if is_deepseek_v4_mla_config(model.config):
        desc = deepseek_v4_mla_cache_descriptor(model.config)
        mla_cfg = _make_default_head_config(model.config)
        print(f" MLA cache: {desc}")
        print(f" MLA logical-head policy: {mla_cfg.summary()}")

    reuse_ok, reuse_reason = _looks_redknot_hf_compatible(model)
    if ENABLE_REUSE and not reuse_ok:
        print(
            f" [skip RedKnot reuse] Current HF model is not supported: {reuse_reason}"
        )
    elif ENABLE_REUSE:
        print(
            " [RedKnot reuse] HF model exposes generic attention attrs; reuse path will run."
        )

    overall = {}
    for ds_name in DATASETS:
        for length_label in LENGTHS:
            target = _LEN_MAP[length_label]
            samples = _load_longbench_padded(ds_name, tok, N_SAMPLES, target)
            task_name = f"{ds_name}@{length_label}"
            if not samples:
                print(f"\n[skip] {task_name}: no usable samples")
                continue
            print(
                f"\n{'=' * W}\n TASK: {task_name} ({len(samples)} sample(s), chunk={CHUNK_TOKENS})\n{'=' * W}"
            )

            rows, ctx_lens, selected_fracs = [], [], []
            for si, sample in enumerate(samples):
                query = _query_text(sample["question"])
                full_text = "\n\n".join(sample["docs"])
                base_text, base_ttft, base_dec, n_ctx = standard_prefill(
                    model, tok, full_text, query
                )
                base_ans = _short_ans(base_text)
                ctx_lens.append(n_ctx)

                rk_text = rk_ans = ""
                rk_ttft = rk_dec = None
                if ENABLE_REUSE and reuse_ok:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    rk_text, rk_ttft, rk_dec, stats = redknot_prefill(
                        model, tok, sample["docs"], query
                    )
                    rk_ans = _short_ans(rk_text)
                    sparse = [x for x in stats if x.get("mode") == "sparse"]
                    if sparse:
                        selected_fracs.append(
                            sum(x["selected_frac"] for x in sparse) / len(sparse)
                        )

                row = {
                    "base_f1": f1_max(base_ans, sample["golds"]),
                    "base_em": em_max(base_ans, sample["golds"]),
                    "base_ttft": base_ttft,
                    "base_dec": base_dec,
                }
                if rk_ttft is not None:
                    row.update(
                        {
                            "rk_f1": f1_max(rk_ans, sample["golds"]),
                            "rk_em": em_max(rk_ans, sample["golds"]),
                            "rk_ttft": rk_ttft,
                            "rk_dec": rk_dec,
                        }
                    )
                rows.append(row)

                print(f"\n [sample {si}] ctx={n_ctx:,} tok docs={len(sample['docs'])}")
                print(f"   Q   : {_trunc(sample['question'], 88)}")
                print(f"   gold: {sample['golds'][0] if sample['golds'] else ''}")
                print(
                    f"   base: {_trunc(base_text, 64)!r} -> {_trunc(base_ans, 32)!r} "
                    f"F1={row['base_f1']:.2f} TTFT={base_ttft:.2f}s dec={base_dec:.1f} tok/s"
                )
                if rk_ttft is not None:
                    print(
                        f"   rk  : {_trunc(rk_text, 64)!r} -> {_trunc(rk_ans, 32)!r} "
                        f"F1={row['rk_f1']:.2f} TTFT={rk_ttft:.2f}s "
                        f"speedup={base_ttft / rk_ttft:.2f}x"
                    )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not rows:
                continue
            avg = lambda key: sum(r[key] for r in rows if key in r) / max(
                1, sum(1 for r in rows if key in r)
            )
            avg_ctx = int(sum(ctx_lens) / len(ctx_lens))
            selected = (
                sum(selected_fracs) / len(selected_fracs) if selected_fracs else 1.0
            )
            dense_until = int(os.environ.get("REDKNOT_FFN_DENSE_UNTIL", "4"))
            local_window = int(os.environ.get("REDKNOT_LOCAL_WINDOW", "4096"))
            frac_global = 1.0 / max(
                1, int(os.environ.get("REDKNOT_GLOBAL_LAYER_STRIDE", "8"))
            )
            flops = compute_flops(
                dims, avg_ctx, frac_global, selected, dense_until, local_window
            )

            print(f"\n {'-' * (W - 2)}")
            print(
                f" {task_name} AGGREGATE ({len(rows)} sample(s), avg ctx={avg_ctx:,} tok)"
            )
            print(f" {'-' * (W - 2)}")
            print(
                f"   baseline  F1={avg('base_f1'):.3f} EM={avg('base_em'):.3f} TTFT={avg('base_ttft'):.2f}s"
            )
            if any("rk_ttft" in r for r in rows):
                print(
                    f"   RedKnot   F1={avg('rk_f1'):.3f} EM={avg('rk_em'):.3f} "
                    f"TTFT={avg('rk_ttft'):.2f}s speedup={avg('base_ttft') / avg('rk_ttft'):.2f}x"
                )
            print(
                "   analytic FLOPs proxy for candidate RedKnot policy (not wall time):"
            )
            for name in ["attn", "ffn", "proj", "total"]:
                dense, rk = flops[name]
                saving = (1 - rk / dense) * 100 if dense else 0.0
                print(
                    f"             {name:6s} dense={dense / 1e15:7.3f}P "
                    f"rk={rk / 1e15:7.3f}P saving={saving:5.1f}%"
                )
            overall[task_name] = {
                "base_f1": avg("base_f1"),
                "base_ttft": avg("base_ttft"),
                "rk_f1": avg("rk_f1") if any("rk_f1" in r for r in rows) else None,
                "rk_ttft": avg("rk_ttft")
                if any("rk_ttft" in r for r in rows)
                else None,
            }

    print(f"\n{'=' * W}")
    print(" SUMMARY")
    print("=" * W)
    print(
        f" {'task':28s} {'base F1':>8s} {'base TTFT':>10s} {'rk F1':>8s} {'rk TTFT':>10s} {'speedup':>8s}"
    )
    for name, row in overall.items():
        if row["rk_ttft"] is None:
            print(
                f" {name:28s} {row['base_f1']:>8.3f} {row['base_ttft']:>9.2f}s {'-':>8s} {'-':>10s} {'-':>8s}"
            )
        else:
            print(
                f" {name:28s} {row['base_f1']:>8.3f} {row['base_ttft']:>9.2f}s "
                f"{row['rk_f1']:>8.3f} {row['rk_ttft']:>9.2f}s "
                f"{row['base_ttft'] / row['rk_ttft']:>7.2f}x"
            )
    print("=" * W)


if __name__ == "__main__":
    main()
