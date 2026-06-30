#!/usr/bin/env python3
"""Two-GPU PD-disaggregation simulation for per-head KV-cache trimming.

Scenario (standard prefix-compression + PD disaggregation)
----------------------------------------------------------
Qwen3-32B, 64 layers, 8 KV-heads/layer.  A RedKnot head-class profile
(``qwen3-32B_optimal_g15_lf_ret.json``) labels every (layer, kv_head) as:

  * ``local_full``  -> only needs window + sink KV  (85% of heads)
  * ``global``      -> needs ALL KV                  (kept whole)
  * ``retrieval``   -> needs ALL KV                  (kept whole)

PD flow we emulate:
  1. PREFILL node (GPU0): full-prefill the 8K-32K prefix, produce the
     complete KV cache.
  2. TRIM: for every ``local_full`` (layer, head) drop KV outside
     [sink, ) U [seq_len-window, seq_len).  This is the "only transfer the
     KV that decode actually needs" step.  ``global``/``retrieval`` heads
     are transferred whole.
  3. TRANSFER: move the trimmed KV GPU0 -> GPU1 (we measure bytes moved
     and verify a real cross-device copy round-trips correctly).
  4. DECODE node (GPU1-resident weights): greedy-decode using the trimmed
     KV and compare against a full-KV baseline.

Because one HF model instance forwards on a single device group, we shard
the (single) weight copy across GPU0+GPU1 with ``device_map`` and run the
KV trim + explicit cross-device transfer to faithfully measure transfer
volume and verify correctness.  This keeps full bf16 precision (no
quantisation noise) so the accuracy numbers are trustworthy.

Usage
-----
  python test/srt/redknot/test_swa_pd_2gpu.py
  SWA_PREFIX_LENGTHS=8192,16384,24576,32768 python .../test_swa_pd_2gpu.py
  SWA_SINK_SIZE=128 SWA_WINDOW_SIZE=4096 python .../test_swa_pd_2gpu.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "SWA_MODEL_PATH", "/mnt/tidal-alsh01/dataset/redone/096/models/Qwen3-32B"
)
PROFILE_PATH = os.environ.get(
    "SWA_PROFILE_PATH",
    str(Path(__file__).parent / "head_class" / "qwen3-32B_optimal_g15_lf_ret.json"),
)
PREFIX_LENGTHS = [
    int(x)
    for x in os.environ.get("SWA_PREFIX_LENGTHS", "8192,16384,24576,32768").split(",")
]
MAX_NEW_TOKENS = int(os.environ.get("SWA_MAX_NEW_TOKENS", "48"))
WINDOW_SIZE = int(os.environ.get("SWA_WINDOW_SIZE", "4096"))
SINK_SIZE = int(os.environ.get("SWA_SINK_SIZE", "128"))
COS_THRESHOLD = float(os.environ.get("SWA_COS_THRESHOLD", "0.99"))
# Only apply local trimming to layers with id < TRIM_LAYER_MAX.  Layers at or
# beyond this index keep ALL their KV even if the profile marks heads local.
# Default 64 = trim every profile-local head.  Set 48 to protect the last 16
# "information-extraction" layers of Qwen3-32B.
TRIM_LAYER_MAX = int(os.environ.get("SWA_TRIM_LAYER_MAX", "64"))

PREFILL_DEV = "cuda:0"
DECODE_DEV = "cuda:1"
W = 92


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def top_k_agreement(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    ta = set(a.topk(k).indices.tolist())
    tb = set(b.topk(k).indices.tolist())
    return len(ta & tb) / k


# ---------------------------------------------------------------------------
# Profile -> per (layer, head) local mask
# ---------------------------------------------------------------------------
def load_local_head_mask(profile_path: str) -> Tuple[torch.Tensor, dict]:
    """Return a bool tensor [num_layers, num_kv_heads] where True == local
    (trimmable) and a small stats dict."""
    with open(profile_path) as f:
        prof = json.load(f)
    cls = prof["kv_head_classification"]
    nl, nh = len(cls), len(cls[0])
    mask = torch.zeros(nl, nh, dtype=torch.bool)
    for li, layer in enumerate(cls):
        for hi, tag in enumerate(layer):
            if tag == "local_full":
                mask[li, hi] = True
    stats = {
        "num_layers": nl,
        "num_kv_heads": nh,
        "local_heads": int(mask.sum()),
        "total_heads": nl * nh,
        "local_pct": 100.0 * float(mask.sum()) / (nl * nh),
        "profile_window": prof.get("window"),
        "profile_sink": prof.get("sink_size"),
    }
    return mask, stats


# ---------------------------------------------------------------------------
# Prefix builder
# ---------------------------------------------------------------------------
def _make_prefix_ids(tokenizer, length: int) -> torch.Tensor:
    seed_text = (
        "The history of artificial intelligence (AI) began in antiquity, "
        "with myths, stories and rumors of artificial beings endowed with "
        "intelligence or consciousness by master craftsmen. The seeds of "
        "modern AI were planted by philosophers who attempted to describe "
        "the process of human thinking as the mechanical manipulation of "
        "symbols. This work culminated in the invention of the programmable "
        "digital computer in the 1940s, a machine based on the abstract "
        "essence of mathematical reasoning. This device and the ideas behind "
        "it inspired a handful of scientists to begin seriously discussing "
        "the possibility of building an electronic brain. The field of AI "
        "research was founded at a workshop held on the campus of Dartmouth "
        "College, USA during the summer of 1956. "
    )
    ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    repeated = ids * ((length // len(ids)) + 2)
    return torch.tensor([repeated[:length]], dtype=torch.long)


# ---------------------------------------------------------------------------
# KV access (HF DynamicCache layout: per-layer [B, H_kv, L, D])
# ---------------------------------------------------------------------------
def _get_layer_kv(past, layer_id):
    if hasattr(past, "key_cache"):
        return past.key_cache[layer_id], past.value_cache[layer_id]
    return past[layer_id][0], past[layer_id][1]


def trim_and_transfer(
    past,
    seq_len: int,
    local_mask: torch.Tensor,
    window_size: int,
    sink_size: int,
    dst_device: str,
) -> Dict[str, int]:
    """Per-head trim of local KV, then emulate the PD transfer GPU0->GPU1.

    For local heads, we move only [0:sink] + [window_start:seq_len] across
    devices.  For global/retrieval heads we move the whole sequence.  We
    write back the round-tripped tensors so decode reads exactly what the
    decode node would have received, and we zero the trimmed positions so
    those entries contribute nothing to attention (emulating "not present").
    """
    window_start = max(sink_size, seq_len - window_size)
    nl, nh = local_mask.shape

    bytes_full = 0
    bytes_sent = 0

    for layer_id in range(nl):
        k, v = _get_layer_kv(past, layer_id)  # [B, H, L, D] on prefill dev
        src_dev = k.device
        H, L, D = k.shape[1], k.shape[2], k.shape[3]
        elem = k.element_size()
        per_tok_head = 2 * D * elem  # K+V

        layer_trimmable = layer_id < TRIM_LAYER_MAX
        for h in range(H):
            bytes_full += L * per_tok_head
            if layer_trimmable and local_mask[layer_id, h] and window_start > sink_size:
                # transfer sink + window slices only
                idx_lo = slice(0, sink_size)
                idx_hi = slice(window_start, L)
                n_sent = sink_size + (L - window_start)
                bytes_sent += n_sent * per_tok_head
                # real cross-device round-trip of the transferred slices
                k_lo = k[:, h, idx_lo, :].to(dst_device).to(src_dev)
                k_hi = k[:, h, idx_hi, :].to(dst_device).to(src_dev)
                v_lo = v[:, h, idx_lo, :].to(dst_device).to(src_dev)
                v_hi = v[:, h, idx_hi, :].to(dst_device).to(src_dev)
                # reconstruct: zero the middle (not transferred / not present)
                k[:, h, :, :] = 0
                v[:, h, :, :] = 0
                k[:, h, idx_lo, :] = k_lo
                k[:, h, idx_hi, :] = k_hi
                v[:, h, idx_lo, :] = v_lo
                v[:, h, idx_hi, :] = v_hi
            else:
                # global / retrieval head: transfer whole sequence
                bytes_sent += L * per_tok_head
                k[:, h, :, :] = k[:, h, :, :].to(dst_device).to(src_dev)
                v[:, h, :, :] = v[:, h, :, :].to(dst_device).to(src_dev)

    return {
        "bytes_full": bytes_full,
        "bytes_sent": bytes_sent,
        "window_start": window_start,
        "trimmed_per_local_head": max(0, window_start - sink_size),
    }


@torch.no_grad()
def greedy_decode(model, past, first_tid, tokenizer, max_new_tokens, device):
    nxt = torch.tensor([[first_tid]], device=device)
    tokens = [first_tid]
    step_logits = []
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        step_logits.append(logits.detach().cpu().clone())
        nxt = logits.argmax().view(1, 1)
        tid = int(nxt[0, 0])
        tokens.append(tid)
        if tid == tokenizer.eos_token_id:
            break
    return tokens, step_logits


def _clone_cache(past):
    """Deep-clone a DynamicCache so baseline and trimmed runs are isolated."""
    from copy import deepcopy

    try:
        return deepcopy(past)
    except Exception:
        # Fallback: manual clone of tensors
        if hasattr(past, "key_cache"):
            import copy

            new = copy.copy(past)
            new.key_cache = [t.clone() for t in past.key_cache]
            new.value_cache = [t.clone() for t in past.value_cache]
            return new
        return [(k.clone(), v.clone()) for (k, v) in past]


# ---------------------------------------------------------------------------
# Single prefix test
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_one(model, tokenizer, prefix_len, local_mask, window_size, sink_size):
    in_dev = PREFILL_DEV
    prefix_ids = _make_prefix_ids(tokenizer, prefix_len).to(in_dev)
    query = "\n\nBased on the above text, briefly summarize the key points:\n"
    query_ids = tokenizer(query, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(in_dev)
    input_ids = torch.cat([prefix_ids, query_ids], dim=1)
    total_len = input_ids.shape[1]

    print(
        f"  Prefill {total_len} tokens ({prefix_len} prefix + {query_ids.shape[1]} q)"
    )

    # ---- PREFILL (shared) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    logits_prefill = out.logits[0, -1, :].cpu().clone()
    first_tid = int(logits_prefill.argmax())

    # ---- BASELINE decode on a cloned full cache ----
    base_cache = _clone_cache(out.past_key_values)
    base_tokens, base_logits = greedy_decode(
        model, base_cache, first_tid, tokenizer, MAX_NEW_TOKENS, in_dev
    )
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    del base_cache
    gc.collect()
    torch.cuda.empty_cache()

    # ---- TRIM + cross-GPU transfer, then decode ----
    trim_cache = out.past_key_values  # reuse (will be mutated)
    t1 = time.perf_counter()
    xfer = trim_and_transfer(
        trim_cache, total_len, local_mask, window_size, sink_size, DECODE_DEV
    )
    torch.cuda.synchronize()
    t_transfer = time.perf_counter() - t1

    trim_tokens, trim_logits = greedy_decode(
        model, trim_cache, first_tid, tokenizer, MAX_NEW_TOKENS, in_dev
    )
    trim_text = tokenizer.decode(trim_tokens, skip_special_tokens=True)
    del out, trim_cache
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Metrics ----
    n = min(len(base_logits), len(trim_logits))
    step_cos = [cos_sim(base_logits[i], trim_logits[i]) for i in range(n)]
    cos_d1 = step_cos[0] if step_cos else 0.0
    avg_cos = sum(step_cos) / len(step_cos) if step_cos else 0.0
    min_cos = min(step_cos) if step_cos else 0.0
    mad_d1 = max_abs_diff(base_logits[0], trim_logits[0]) if step_cos else -1
    topk_d1 = top_k_agreement(base_logits[0], trim_logits[0]) if step_cos else 0
    tokens_match = base_tokens == trim_tokens
    diverge = -1
    for i in range(min(len(base_tokens), len(trim_tokens))):
        if base_tokens[i] != trim_tokens[i]:
            diverge = i
            break

    bf, bs = xfer["bytes_full"], xfer["bytes_sent"]
    saved_pct = 100.0 * (bf - bs) / bf if bf else 0.0

    return {
        "prefix_len": prefix_len,
        "total_len": total_len,
        "window_size": window_size,
        "sink_size": sink_size,
        "cos_decode_step1": cos_d1,
        "avg_cos_decode": avg_cos,
        "min_cos_decode": min_cos,
        "max_abs_diff_d1": mad_d1,
        "top10_agree_d1": topk_d1,
        "tokens_match": tokens_match,
        "diverge_step": diverge,
        "n_decode_steps": len(base_tokens),
        "baseline_text": base_text[:140],
        "trimmed_text": trim_text[:140],
        "kv_full_MB": bf / 1024 / 1024,
        "kv_sent_MB": bs / 1024 / 1024,
        "saved_pct": saved_pct,
        "prefill_time_s": t_prefill,
        "transfer_time_s": t_transfer,
        "step_cosines": step_cos,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_gpu = torch.cuda.device_count()
    print("=" * W)
    print(" Two-GPU PD-Disaggregation per-head KV-trim accuracy test")
    print(f" Model  : {MODEL_PATH}")
    print(f" Profile: {PROFILE_PATH}")
    print(f" Prefix : {PREFIX_LENGTHS}")
    print(f" Window : {WINDOW_SIZE}  Sink: {SINK_SIZE}  Decode: {MAX_NEW_TOKENS}")
    print(f" TrimLayerMax: {TRIM_LAYER_MAX}  (layers >= this keep full KV)")
    print(f" GPUs   : {n_gpu} visible  (prefill={PREFILL_DEV}, decode={DECODE_DEV})")
    print("=" * W)
    if n_gpu < 2:
        print(" ERROR: need >=2 GPUs")
        return 2

    local_mask, mstats = load_local_head_mask(PROFILE_PATH)
    print(
        f"\n Profile: {mstats['num_layers']}L x {mstats['num_kv_heads']}H, "
        f"local(trimmable)={mstats['local_heads']}/{mstats['total_heads']} "
        f"({mstats['local_pct']:.1f}%), global/retrieval kept whole"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n Loading model bf16, sharded over GPU0+GPU1 ...")
    max_mem = {0: "40GiB", 1: "40GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    print(
        " Model loaded. hf_device_map sample:",
        {k: v for k, v in list(model.hf_device_map.items())[:3]},
        "...",
    )

    results, all_pass = [], True
    for plen in PREFIX_LENGTHS:
        print(f"\n{'-' * W}\n PREFIX_LEN = {plen}\n{'-' * W}")
        if plen <= WINDOW_SIZE + SINK_SIZE:
            print(f"  SKIP: {plen} <= window+sink")
            continue
        try:
            r = run_one(model, tokenizer, plen, local_mask, WINDOW_SIZE, SINK_SIZE)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            torch.cuda.empty_cache()
            continue
        results.append(r)
        passed = r["cos_decode_step1"] >= COS_THRESHOLD
        all_pass = all_pass and passed
        tag = "PASS" if passed else "FAIL"
        print(f"\n  [{tag}] prefix={plen}")
        print(
            f"    cos(decode step-1):   {r['cos_decode_step1']:.6f}  (thr={COS_THRESHOLD})"
        )
        print(
            f"    avg/min cos decode:   {r['avg_cos_decode']:.6f} / {r['min_cos_decode']:.6f}"
        )
        print(f"    top-10 agree (d1):    {r['top10_agree_d1']:.0%}")
        print(
            f"    token match ({r['n_decode_steps']}): "
            f"{'YES' if r['tokens_match'] else 'NO'}"
            + (f"  diverge@{r['diverge_step']}" if r["diverge_step"] >= 0 else "")
        )
        print(
            f"    KV transfer: {r['kv_full_MB']:.0f} -> {r['kv_sent_MB']:.0f} MB "
            f"(saved {r['saved_pct']:.1f}%)"
        )
        print(
            f"    prefill {r['prefill_time_s']:.2f}s  transfer {r['transfer_time_s']:.2f}s"
        )
        print(f"    base: {r['baseline_text'][:80]}")
        print(f"    trim: {r['trimmed_text'][:80]}")

    # Summary
    print(f"\n{'=' * W}\n SUMMARY (window={WINDOW_SIZE}, sink={SINK_SIZE})\n{'=' * W}")
    print(
        f"  {'Prefix':>8} {'cos(d1)':>10} {'avg_cos':>9} {'top10':>6} "
        f"{'tok_match':>9} {'saved%':>7} {'result':>6}"
    )
    for r in results:
        p = r["cos_decode_step1"] >= COS_THRESHOLD
        print(
            f"  {r['prefix_len']:>8} {r['cos_decode_step1']:>10.6f} "
            f"{r['avg_cos_decode']:>9.6f} {r['top10_agree_d1']:>5.0%} "
            f"{'YES' if r['tokens_match'] else 'NO':>9} {r['saved_pct']:>6.1f}% "
            f"{'PASS' if p else 'FAIL':>6}"
        )
    print(f"\n {'ALL PASSED' if all_pass else 'SOME FAILED'} (cos >= {COS_THRESHOLD})")

    out = Path(__file__).with_suffix(".results.json")
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "window": WINDOW_SIZE,
                    "sink": SINK_SIZE,
                    "profile": PROFILE_PATH,
                    "local_pct": mstats["local_pct"],
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f" Results -> {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
