#!/usr/bin/env python3
"""Two-GPU PD-disaggregation benchmark: accuracy + throughput across prefixes.

For each prefix length and each trim-strength we report:

  Accuracy (real, bf16, no quantisation):
    * cos(decode step-1), avg/min cos, top-10 agreement, greedy token match

  Performance:
    * prefill_time_s        -- real wall-clock (single prefill, shared)
    * decode_tps_real       -- real decode tokens/s on HF dense cache
    * transfer_time_s       -- real cross-GPU KV move time (trimmed slices)
    * transfer_saved_pct    -- KV bytes saved vs full transfer
    * e2e_latency_s_real    -- prefill + transfer + decode (real)
    * qps_real              -- 1 / e2e_latency_s_real  (single-stream)
    * decode_tps_est        -- ANALYTICAL estimate of decode speed if the
                               attention cost scaled with the *effective* KV
                               length after per-head trimming (HF dense cache
                               cannot physically shrink per-head, so the real
                               decode does NOT show this speed-up; the estimate
                               models what a true SWA kernel would achieve).
    * qps_est               -- estimated single-stream QPS using decode_tps_est

NOTE on the estimate: decode attention FLOPs ~ sum over (layer,head) of the KV
length each head attends to.  Trimming local heads to (sink+window) reduces the
average KV length, so per-token decode attention time scales by
  ratio = effective_kv_positions / full_kv_positions.
We apply this ratio ONLY to the attention portion; the MLP / projection cost is
assumed constant.  We use a conservative attn_fraction (default 0.5 at these
lengths) so the estimate is not over-optimistic.

Usage
-----
  CUDA_VISIBLE_DEVICES=0,1 python test/srt/redknot/test_swa_pd_bench.py
  SWA_PREFIX_LENGTHS=4096,8192,12288,16384,24576,32768 \
  SWA_TRIM_CONFIGS=0,32,48 python .../test_swa_pd_bench.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_swa_pd_2gpu as B  # reuse helpers

MODEL_PATH = B.MODEL_PATH
PROFILE_PATH = B.PROFILE_PATH
PREFIX_LENGTHS = [
    int(x)
    for x in os.environ.get(
        "SWA_PREFIX_LENGTHS", "4096,8192,12288,16384,24576,32768"
    ).split(",")
]
# Trim configs = list of TRIM_LAYER_MAX values; 0 means baseline (no trim).
TRIM_CONFIGS = [
    int(x) for x in os.environ.get("SWA_TRIM_CONFIGS", "0,32,48").split(",")
]
MAX_NEW_TOKENS = int(os.environ.get("SWA_MAX_NEW_TOKENS", "64"))
WINDOW_SIZE = int(os.environ.get("SWA_WINDOW_SIZE", "4096"))
SINK_SIZE = int(os.environ.get("SWA_SINK_SIZE", "128"))
COS_THRESHOLD = float(os.environ.get("SWA_COS_THRESHOLD", "0.99"))
# Fraction of per-decode-step time spent in attention (vs MLP+proj) at these
# context lengths.  Used only for the analytical decode-speed estimate.
ATTN_FRACTION = float(os.environ.get("SWA_ATTN_FRACTION", "0.5"))
DECODE_WARMUP = int(os.environ.get("SWA_DECODE_WARMUP", "4"))
DECODE_TIMED = int(os.environ.get("SWA_DECODE_TIMED", "32"))

PREFILL_DEV = B.PREFILL_DEV
DECODE_DEV = B.DECODE_DEV
W = 116

cos_sim = B.cos_sim
top_k_agreement = B.top_k_agreement


def effective_kv_positions(
    local_mask: torch.Tensor,
    trim_layer_max: int,
    seq_len: int,
    window_size: int,
    sink_size: int,
) -> Dict[str, float]:
    """Sum of KV positions each (layer,head) attends to, full vs trimmed."""
    nl, nh = local_mask.shape
    window_start = max(sink_size, seq_len - window_size)
    trimmed_per_head = max(0, window_start - sink_size)
    full = nl * nh * seq_len
    sent = 0
    for li in range(nl):
        trimmable = li < trim_layer_max
        for hi in range(nh):
            if trimmable and bool(local_mask[li, hi]) and window_start > sink_size:
                sent += seq_len - trimmed_per_head
            else:
                sent += seq_len
    return {
        "full_positions": full,
        "kept_positions": sent,
        "ratio": sent / full if full else 1.0,
    }


@torch.no_grad()
def timed_decode_tps(model, past_template, first_tid, tokenizer, device, warmup, timed):
    """Measure real decode tokens/s.  Re-uses a deep-copied cache each call so
    we don't mutate the template; runs warmup then timed steps."""
    import copy

    # warmup
    past = B._clone_cache(past_template)
    nxt = torch.tensor([[first_tid]], device=device)
    for _ in range(warmup):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1, :].argmax().view(1, 1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1, :].argmax().view(1, 1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return timed / dt, dt / timed


@torch.no_grad()
def _one_config(
    model,
    tokenizer,
    out_cache,
    total_len,
    first_tid,
    t_prefill,
    base_tokens,
    base_logits,
    local_mask,
    trim_layer_max,
    prefix_len,
    in_dev,
):
    """Evaluate a single trim config on a (freshly cloned) cache.

    out_cache is the pristine full prefill cache; we clone it so the original
    is preserved for the next config.  This keeps peak memory to ~2 KV copies.
    """
    is_baseline = trim_layer_max == 0
    cell_cache = B._clone_cache(out_cache)

    saved_pct = 0.0
    kv_full_mb = kv_sent_mb = 0.0
    t_transfer = 0.0
    eff = {"ratio": 1.0}
    if is_baseline:
        t1 = time.perf_counter()
        xfer = B.trim_and_transfer(
            cell_cache,
            total_len,
            torch.zeros_like(local_mask),
            WINDOW_SIZE,
            SINK_SIZE,
            DECODE_DEV,
        )
        torch.cuda.synchronize()
        t_transfer = time.perf_counter() - t1
        kv_full_mb = xfer["bytes_full"] / 1024 / 1024
        kv_sent_mb = xfer["bytes_sent"] / 1024 / 1024
    else:
        B.TRIM_LAYER_MAX = trim_layer_max
        t1 = time.perf_counter()
        xfer = B.trim_and_transfer(
            cell_cache, total_len, local_mask, WINDOW_SIZE, SINK_SIZE, DECODE_DEV
        )
        torch.cuda.synchronize()
        t_transfer = time.perf_counter() - t1
        kv_full_mb = xfer["bytes_full"] / 1024 / 1024
        kv_sent_mb = xfer["bytes_sent"] / 1024 / 1024
        saved_pct = 100.0 * (kv_full_mb - kv_sent_mb) / kv_full_mb if kv_full_mb else 0
        eff = effective_kv_positions(
            local_mask, trim_layer_max, total_len, WINDOW_SIZE, SINK_SIZE
        )

    tps_real, _ = timed_decode_tps(
        model, cell_cache, first_tid, tokenizer, in_dev, DECODE_WARMUP, DECODE_TIMED
    )
    cell_tokens, cell_logits = B.greedy_decode(
        model, cell_cache, first_tid, tokenizer, MAX_NEW_TOKENS, in_dev
    )
    del cell_cache
    gc.collect()
    torch.cuda.empty_cache()

    n = min(len(base_logits), len(cell_logits))
    step_cos = [cos_sim(base_logits[i], cell_logits[i]) for i in range(n)]
    cos_d1 = step_cos[0] if step_cos else 1.0
    avg_cos = sum(step_cos) / len(step_cos) if step_cos else 1.0
    min_cos = min(step_cos) if step_cos else 1.0
    topk_d1 = top_k_agreement(base_logits[0], cell_logits[0]) if step_cos else 1.0
    tokens_match = base_tokens == cell_tokens
    diverge = -1
    for i in range(min(len(base_tokens), len(cell_tokens))):
        if base_tokens[i] != cell_tokens[i]:
            diverge = i
            break

    # ---- analytical decode-speed estimate ----
    # attention time scales by eff ratio; mlp constant.
    ratio = eff["ratio"]
    speedup_attn = ATTN_FRACTION * ratio + (1 - ATTN_FRACTION)
    tps_est = tps_real / speedup_attn if speedup_attn > 0 else tps_real

    e2e_real = t_prefill + t_transfer + (MAX_NEW_TOKENS / tps_real)
    e2e_est = t_prefill + t_transfer + (MAX_NEW_TOKENS / tps_est)
    qps_real = 1.0 / e2e_real if e2e_real > 0 else 0
    qps_est = 1.0 / e2e_est if e2e_est > 0 else 0

    label = "baseline" if is_baseline else f"trim<{trim_layer_max}"
    return {
        "prefix_len": prefix_len,
        "total_len": total_len,
        "config": label,
        "trim_layer_max": trim_layer_max,
        "cos_d1": cos_d1,
        "avg_cos": avg_cos,
        "min_cos": min_cos,
        "top10_d1": topk_d1,
        "tokens_match": tokens_match,
        "diverge_step": diverge,
        "kv_full_MB": kv_full_mb,
        "kv_sent_MB": kv_sent_mb,
        "transfer_saved_pct": saved_pct,
        "kv_ratio": ratio,
        "prefill_time_s": t_prefill,
        "transfer_time_s": t_transfer,
        "decode_tps_real": tps_real,
        "decode_tps_est": tps_est,
        "e2e_latency_s_real": e2e_real,
        "e2e_latency_s_est": e2e_est,
        "qps_real": qps_real,
        "qps_est": qps_est,
        "passed": bool(cos_d1 >= COS_THRESHOLD) if not is_baseline else True,
    }


@torch.no_grad()
def run_prefix(model, tokenizer, prefix_len, local_mask, trim_configs):
    """Prefill once, baseline-decode once, then evaluate each trim config."""
    in_dev = PREFILL_DEV
    prefix_ids = B._make_prefix_ids(tokenizer, prefix_len).to(in_dev)
    query = "\n\nBased on the above text, briefly summarize the key points:\n"
    query_ids = tokenizer(query, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(in_dev)
    input_ids = torch.cat([prefix_ids, query_ids], dim=1)
    total_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    first_tid = int(out.logits[0, -1, :].argmax())

    # baseline reference decode (full cache, clone so out stays pristine)
    base_cache = B._clone_cache(out.past_key_values)
    base_tokens, base_logits = B.greedy_decode(
        model, base_cache, first_tid, tokenizer, MAX_NEW_TOKENS, in_dev
    )
    del base_cache
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for tlm in trim_configs:
        if tlm != 0 and prefix_len <= WINDOW_SIZE + SINK_SIZE:
            continue
        tag = "baseline" if tlm == 0 else f"trim<{tlm}"
        print(f"   - cfg={tag}")
        r = _one_config(
            model,
            tokenizer,
            out.past_key_values,
            total_len,
            first_tid,
            t_prefill,
            base_tokens,
            base_logits,
            local_mask,
            tlm,
            prefix_len,
            in_dev,
        )
        rows.append(r)

    del out
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_gpu = torch.cuda.device_count()
    print("=" * W)
    print(" PD-Disaggregation benchmark: accuracy + throughput")
    print(f" Model {MODEL_PATH}")
    print(f" Prefixes {PREFIX_LENGTHS}")
    print(f" Trim configs (TRIM_LAYER_MAX) {TRIM_CONFIGS}  (0=baseline)")
    print(
        f" window={WINDOW_SIZE} sink={SINK_SIZE} decode={MAX_NEW_TOKENS} "
        f"attn_frac={ATTN_FRACTION}"
    )
    print(f" GPUs {n_gpu} (prefill={PREFILL_DEV} decode={DECODE_DEV})")
    print("=" * W)
    if n_gpu < 2:
        print(" ERROR need >=2 GPUs")
        return 2

    local_mask, mstats = B.load_local_head_mask(PROFILE_PATH)
    print(
        f" Profile: {mstats['num_layers']}L x {mstats['num_kv_heads']}H, "
        f"local={mstats['local_pct']:.1f}%"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(" Loading model bf16 sharded over 2 GPUs ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    print(" Model loaded.\n")

    rows = []
    for plen in PREFIX_LENGTHS:
        print(f" [prefix={plen}]")
        try:
            rows.extend(run_prefix(model, tokenizer, plen, local_mask, TRIM_CONFIGS))
        except torch.cuda.OutOfMemoryError as e:
            print(f"   OOM at prefix={plen}: {str(e)[:80]}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # ---- Tables ----
    print(f"\n{'=' * W}\n ACCURACY\n{'=' * W}")
    print(
        f"  {'prefix':>7} {'config':>10} {'cos_d1':>9} {'avg_cos':>9} "
        f"{'min_cos':>9} {'top10':>6} {'tok_match':>9} {'diverge':>8} {'res':>5}"
    )
    for r in rows:
        res = "-" if r["config"] == "baseline" else ("PASS" if r["passed"] else "FAIL")
        print(
            f"  {r['prefix_len']:>7} {r['config']:>10} {r['cos_d1']:>9.5f} "
            f"{r['avg_cos']:>9.5f} {r['min_cos']:>9.4f} {r['top10_d1']:>5.0%} "
            f"{'YES' if r['tokens_match'] else 'NO':>9} "
            f"{r['diverge_step']:>8} {res:>5}"
        )

    print(
        f"\n{'=' * W}\n PERFORMANCE  (real wall-clock; QPS = single-stream 1/e2e)\n{'=' * W}"
    )
    print(
        f"  {'prefix':>7} {'config':>10} {'prefill_s':>9} {'dec_tps':>8} "
        f"{'xfer_s':>7} {'saved%':>7} {'e2e_s':>7} {'qps':>6} "
        f"{'tps_est':>8} {'qps_est':>7}"
    )
    for r in rows:
        print(
            f"  {r['prefix_len']:>7} {r['config']:>10} {r['prefill_time_s']:>9.3f} "
            f"{r['decode_tps_real']:>8.1f} {r['transfer_time_s']:>7.3f} "
            f"{r['transfer_saved_pct']:>6.1f}% {r['e2e_latency_s_real']:>7.3f} "
            f"{r['qps_real']:>6.3f} {r['decode_tps_est']:>8.1f} {r['qps_est']:>7.3f}"
        )

    # ---- per-prefix speedup summary (trim vs baseline) ----
    print(f"\n{'=' * W}\n SPEEDUP vs baseline (est e2e, single-stream)\n{'=' * W}")
    by_prefix = {}
    for r in rows:
        by_prefix.setdefault(r["prefix_len"], {})[r["config"]] = r
    print(
        f"  {'prefix':>7} {'config':>10} {'saved%KV':>9} {'qps_est':>8} "
        f"{'speedup':>8} {'acc':>5}"
    )
    for plen, cfgs in by_prefix.items():
        base = cfgs.get("baseline")
        for cfg, r in cfgs.items():
            if cfg == "baseline":
                continue
            sp = r["qps_est"] / base["qps_est"] if base and base["qps_est"] else 0
            print(
                f"  {plen:>7} {cfg:>10} {r['transfer_saved_pct']:>8.1f}% "
                f"{r['qps_est']:>8.3f} {sp:>7.2f}x "
                f"{'OK' if r['passed'] else 'BAD':>5}"
            )

    out = Path(__file__).with_suffix(".results.json")
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "window": WINDOW_SIZE,
                    "sink": SINK_SIZE,
                    "attn_fraction": ATTN_FRACTION,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
                "rows": rows,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n Results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
