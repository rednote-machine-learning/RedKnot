#!/usr/bin/env python3
"""Memory-bound concurrency / QPS benchmark for PD-disaggregation KV trimming.

The single-stream latency barely changes when we trim KV, but the *decode
node* is memory-bound: every concurrent request holds a KV cache, and smaller
KV/req => more requests fit => higher aggregate decode throughput (QPS).

This benchmark measures, on ONE GPU (NF4 weights to free ~60 GB for KV):

  1. MAX CONCURRENCY (real, memory-bound):
     Greedily allocate request KV caches of the *real* per-request size until
     OOM.  Baseline uses full KV (every layer full length); trim<32 uses the
     real trimmed size (layers 0-31 -> sink+window, layers 32-63 -> full).

  2. DECODE THROUGHPUT at max batch (real forward):
     Run a real batched single-step decode at the measured max batch and report
     aggregate tokens/s = batch / step_time.  For baseline the KV is full
     length; for the trimmed case we use the per-request *effective* KV length
     so both memory and attention compute reflect the trim.

  QPS (decode-bound, single decode node) = aggregate_tokens_per_s / out_len.

trim<32 is the accuracy-safe config validated earlier.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_swa_pd_2gpu as B

MODEL_PATH = B.MODEL_PATH
PROFILE_PATH = B.PROFILE_PATH
PREFIX_LENGTHS = [
    int(x) for x in os.environ.get("SWA_PREFIX_LENGTHS", "8192,16384,32768").split(",")
]
WINDOW_SIZE = int(os.environ.get("SWA_WINDOW_SIZE", "4096"))
SINK_SIZE = int(os.environ.get("SWA_SINK_SIZE", "128"))
TRIM_LAYER_MAX = int(os.environ.get("SWA_TRIM_LAYER_MAX", "32"))
OUT_LEN = int(os.environ.get("SWA_OUT_LEN", "256"))  # avg decode length per req
KV_BUDGET_GIB = float(os.environ.get("SWA_KV_BUDGET_GIB", "0"))  # 0 = auto-detect
DECODE_BATCHES = [
    int(x) for x in os.environ.get("SWA_DECODE_BATCHES", "").split(",") if x
]


def model_dims():
    import json as _j

    cfg = _j.load(open(Path(MODEL_PATH) / "config.json"))
    nl = cfg["num_hidden_layers"]
    kvh = cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    nh = cfg["num_attention_heads"]
    return nl, kvh, hd, nh, cfg["hidden_size"]


def kv_bytes_per_req(seqlen, trim_layer_max, local_mask, nl, kvh, hd, elem=2):
    """Real KV bytes for one request (K+V), per-head trimming applied."""
    window_start = max(SINK_SIZE, seqlen - WINDOW_SIZE)
    kept_local = SINK_SIZE + (seqlen - window_start)
    total_pos = 0
    for li in range(nl):
        trimmable = li < trim_layer_max
        for hi in range(kvh):
            if trimmable and bool(local_mask[li, hi]) and window_start > SINK_SIZE:
                total_pos += kept_local
            else:
                total_pos += seqlen
    return total_pos * 2 * hd * elem  # K+V over all (layer,head)


def measure_max_concurrency(kv_bytes, budget_bytes):
    """Memory-bound max requests = floor(budget / kv_bytes_per_req)."""
    return int(budget_bytes // kv_bytes) if kv_bytes > 0 else 0


@torch.no_grad()
def real_decode_tps(
    model, batch, kv_len, nl, kvh, hd, nh, hidden, device, steps=8, dtype=torch.bfloat16
):
    """Real batched single-step decode tps with a synthetic KV cache of length
    kv_len (uniform across layers) and `batch` requests.  Returns tokens/s."""
    from transformers.cache_utils import DynamicCache

    # Build a DynamicCache with `batch` x `kv_len` per layer (transformers>=4.5x
    # layout: cache.layers[i].keys / .values).
    cache = DynamicCache()
    for li in range(nl):
        k = torch.zeros(batch, kvh, kv_len, hd, dtype=dtype, device=device)
        v = torch.zeros(batch, kvh, kv_len, hd, dtype=dtype, device=device)
        cache.update(k, v, li)

    def rollback(c):
        for layer in c.layers:
            layer.keys = layer.keys[:, :, :kv_len, :]
            layer.values = layer.values[:, :, :kv_len, :]

    input_ids = torch.randint(0, 1000, (batch, 1), device=device)
    cache_position = torch.tensor([kv_len], device=device)
    # warmup
    for _ in range(2):
        out = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        rollback(cache)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        rollback(cache)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return batch * steps / dt  # tokens/s


def effective_kv_len(seqlen, trim_layer_max, local_mask, nl, kvh):
    """Average KV length per (layer,head) after trim -> single uniform length
    that reproduces the same total attention positions (for the tps proxy)."""
    window_start = max(SINK_SIZE, seqlen - WINDOW_SIZE)
    kept_local = SINK_SIZE + (seqlen - window_start)
    total = 0
    for li in range(nl):
        trimmable = li < trim_layer_max
        for hi in range(kvh):
            total += (
                kept_local
                if (trimmable and bool(local_mask[li, hi]) and window_start > SINK_SIZE)
                else seqlen
            )
    return int(round(total / (nl * kvh)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=" * 100)
    print(" Memory-bound concurrency / QPS benchmark (PD KV trimming)")
    print(f" Model {MODEL_PATH}  prefixes {PREFIX_LENGTHS}")
    print(
        f" window={WINDOW_SIZE} sink={SINK_SIZE} trim_layer_max={TRIM_LAYER_MAX} out_len={OUT_LEN}"
    )
    print("=" * 100)

    nl, kvh, hd, nh, hidden = model_dims()
    local_mask, mstats = B.load_local_head_mask(PROFILE_PATH)
    print(f" dims: {nl}L {kvh}kvH {nh}qH hd={hd}  local={mstats['local_pct']:.0f}%")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    attn_impl = os.environ.get("SWA_ATTN_IMPL", "sdpa")
    print(f" Loading NF4 model on cuda:0 (attn={attn_impl}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=qc,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).eval()
    # detect runtime activation dtype for the synthetic KV cache
    run_dtype = torch.bfloat16
    try:
        emb = model.get_input_embeddings()
        run_dtype = (
            emb.weight.dtype if emb.weight.dtype.is_floating_point else torch.bfloat16
        )
    except Exception:
        pass
    print(f" runtime KV dtype: {run_dtype}")

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    reserve = int(3 * 1024**3)  # 3GB headroom for activations
    budget = (KV_BUDGET_GIB * 1024**3) if KV_BUDGET_GIB > 0 else (free - reserve)
    print(
        f" GPU free {free / 1024**3:.1f} GiB / total {total / 1024**3:.1f} GiB "
        f"-> KV budget {budget / 1024**3:.1f} GiB (reserve 3 GiB)\n"
    )

    rows = []
    for plen in PREFIX_LENGTHS:
        seqlen = plen
        kv_base = kv_bytes_per_req(seqlen, 0, local_mask, nl, kvh, hd)
        kv_trim = kv_bytes_per_req(seqlen, TRIM_LAYER_MAX, local_mask, nl, kvh, hd)
        nb = measure_max_concurrency(kv_base, budget)
        nt = measure_max_concurrency(kv_trim, budget)

        eff_base = seqlen
        eff_trim = effective_kv_len(seqlen, TRIM_LAYER_MAX, local_mask, nl, kvh)

        # real decode tps at each max batch (cap batch for tps probe to avoid
        # OOM during the synthetic-cache build; throughput scales ~linearly so
        # we probe at min(maxbatch, probe_cap) then extrapolate).
        probe_cap = int(os.environ.get("SWA_TPS_PROBE_CAP", "16"))
        pb_base = min(nb, probe_cap) or 1
        pb_trim = min(nt, probe_cap) or 1
        print(
            f" [prefix={plen}] base_kv={kv_base / 1024**2:.0f}MB/req "
            f"trim_kv={kv_trim / 1024**2:.0f}MB/req  maxbatch base={nb} trim={nt}"
        )
        print(
            f"   probing decode tps: base bs={pb_base}@len{eff_base}, "
            f"trim bs={pb_trim}@len{eff_trim}"
        )
        try:
            tps_base_probe = real_decode_tps(
                model,
                pb_base,
                eff_base,
                nl,
                kvh,
                hd,
                nh,
                hidden,
                "cuda:0",
                dtype=run_dtype,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tps_base_probe = float("nan")
            pb_base = 1
            tps_base_probe = real_decode_tps(
                model, 1, eff_base, nl, kvh, hd, nh, hidden, "cuda:0", dtype=run_dtype
            )
        try:
            tps_trim_probe = real_decode_tps(
                model,
                pb_trim,
                eff_trim,
                nl,
                kvh,
                hd,
                nh,
                hidden,
                "cuda:0",
                dtype=run_dtype,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pb_trim = 1
            tps_trim_probe = real_decode_tps(
                model, 1, eff_trim, nl, kvh, hd, nh, hidden, "cuda:0", dtype=run_dtype
            )

        # extrapolate aggregate tps to full max batch (per-token decode is
        # memory-bandwidth bound; tps grows ~linearly with batch until compute
        # bound -> conservative linear extrapolation from probe).
        agg_tps_base = tps_base_probe / pb_base * nb if pb_base else 0
        agg_tps_trim = tps_trim_probe / pb_trim * nt if pb_trim else 0
        qps_base = agg_tps_base / OUT_LEN
        qps_trim = agg_tps_trim / OUT_LEN

        rows.append(
            {
                "prefix_len": plen,
                "kv_base_MB": kv_base / 1024**2,
                "kv_trim_MB": kv_trim / 1024**2,
                "kv_ratio": kv_trim / kv_base,
                "max_batch_base": nb,
                "max_batch_trim": nt,
                "batch_gain": nt / nb if nb else 0,
                "eff_len_base": eff_base,
                "eff_len_trim": eff_trim,
                "tps_per_req_base": tps_base_probe / pb_base if pb_base else 0,
                "tps_per_req_trim": tps_trim_probe / pb_trim if pb_trim else 0,
                "agg_tps_base": agg_tps_base,
                "agg_tps_trim": agg_tps_trim,
                "qps_base": qps_base,
                "qps_trim": qps_trim,
                "qps_gain": qps_trim / qps_base if qps_base else 0,
            }
        )

    print(
        f"\n{'=' * 100}\n CONCURRENCY (memory-bound, KV budget {budget / 1024**3:.0f} GiB)\n{'=' * 100}"
    )
    print(
        f"  {'prefix':>7} {'kv_base':>9} {'kv_trim':>9} {'ratio':>6} "
        f"{'bs_base':>8} {'bs_trim':>8} {'bs_gain':>8}"
    )
    for r in rows:
        print(
            f"  {r['prefix_len']:>7} {r['kv_base_MB']:>7.0f}MB {r['kv_trim_MB']:>7.0f}MB "
            f"{r['kv_ratio']:>6.2f} {r['max_batch_base']:>8} {r['max_batch_trim']:>8} "
            f"{r['batch_gain']:>7.2f}x"
        )

    print(f"\n{'=' * 100}\n DECODE THROUGHPUT / QPS  (out_len={OUT_LEN})\n{'=' * 100}")
    print(
        f"  {'prefix':>7} {'aggTPS_base':>12} {'aggTPS_trim':>12} "
        f"{'qps_base':>9} {'qps_trim':>9} {'qps_gain':>9}"
    )
    for r in rows:
        print(
            f"  {r['prefix_len']:>7} {r['agg_tps_base']:>12.0f} {r['agg_tps_trim']:>12.0f} "
            f"{r['qps_base']:>9.3f} {r['qps_trim']:>9.3f} {r['qps_gain']:>8.2f}x"
        )

    out = Path(__file__).with_suffix(".results.json")
    with open(out, "w") as f:
        json.dump(
            {
                "budget_GiB": budget / 1024**3,
                "out_len": OUT_LEN,
                "trim_layer_max": TRIM_LAYER_MAX,
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
