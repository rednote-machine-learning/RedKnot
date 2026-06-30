#!/usr/bin/env python3
"""Replay the real musique reuse stream through every KV-cache policy.

Compares (under the SAME byte budget):
  * cache_all      -- store everything forever (current offline approach)
  * lru            -- plain LRU
  * lfu            -- frequency-only
  * redknot_3layer -- admission + value-score eviction + TTL

Outputs hit rate, peak KV memory, token reuse rate, and GPU-seconds saved for
DeepSeek V4 Flash.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(modfile):
    import sys

    spec = importlib.util.spec_from_file_location(modfile.stem, str(modfile))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modfile.stem] = m  # needed for @dataclass module lookup
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/musique_ans_v1.0_dev.jsonl",
    )
    ap.add_argument(
        "--budget-gb", type=float, default=1.0, help="KV cache byte budget (GB)"
    )
    ap.add_argument(
        "--kv-bytes-per-token", type=float, default=49536, help="DeepSeek V4 MLA"
    )
    ap.add_argument(
        "--chunk-tokens", type=int, default=600, help="avg tokens per passage chunk"
    )
    ap.add_argument(
        "--carry-tokens", type=int, default=128, help="RedKnot carry-prefix recompute"
    )
    ap.add_argument(
        "--prefill-ms-per-token",
        type=float,
        default=0.066,
        help="DeepSeek V4 prefill ms per token (265ms/4000tok)",
    )
    ap.add_argument("--r-admit", type=int, default=2)
    ap.add_argument("--tau", type=float, default=200.0)
    ap.add_argument("--out", default=str(HERE / "figures/replay_kv_lifecycle.json"))
    args = ap.parse_args()

    cl = _load(HERE / "chunk_lifecycle.py")
    mgr = _load(HERE / "kv_cache_lifecycle.py")

    stream = cl.load_musique_stream(args.path, None)
    budget = int(args.budget_gb * 1e9)

    # Build per-access list: (chunk_id, n_tokens, kv_bytes)
    # Use a fixed avg chunk size so the byte budget is comparable across policies.
    def chunk_kv(n_tok):
        return int(n_tok * args.kv_bytes_per_token)

    accesses = []
    for rid, chunk_ids, texts in stream:
        for cid in chunk_ids:
            n_tok = args.chunk_tokens
            accesses.append((cid, n_tok, chunk_kv(n_tok)))

    policies = {
        "cache_all": mgr.CacheAll(),
        "lru": mgr.LRU(capacity_bytes=budget),
        "lfu": mgr.LFU(capacity_bytes=budget),
        "redknot_3layer": mgr.KVCacheLifecycleManager(
            capacity_bytes=budget, r_admit=args.r_admit, tau=args.tau
        ),
    }

    for name, pol in policies.items():
        for cid, n_tok, kvb in accesses:
            carry = args.carry_tokens if name == "redknot_3layer" else 0
            pol.access(cid, n_tok, kvb, carry_tokens=carry)

    results = []
    for name, pol in policies.items():
        s = pol.stats()
        # GPU-seconds saved = reused_tokens * prefill_ms_per_token / 1000
        s["gpu_seconds_saved"] = round(
            s["reused_tokens"] * args.prefill_ms_per_token / 1000.0, 1
        )
        # efficiency: GPU-sec saved per GB of peak KV
        s["sec_saved_per_gb"] = round(
            s["gpu_seconds_saved"] / max(s["peak_kv_gb"], 1e-9), 1
        )
        results.append(s)

    print(
        f"\n=== KV-cache policy replay (musique, {len(accesses)} chunk accesses, "
        f"budget={args.budget_gb}GB) ==="
    )
    hdr = (
        f"{'policy':>16} | {'hit_rate':>8} {'peak_KV_GB':>10} {'tok_reuse':>9} "
        f"{'GPU_sec_saved':>13} {'sec/GB':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in results:
        print(
            f"{s['policy']:>16} | {s['hit_rate']:>8.3f} {s['peak_kv_gb']:>10.2f} "
            f"{s['token_reuse_rate']:>9.3f} {s['gpu_seconds_saved']:>13.1f} {s['sec_saved_per_gb']:>8.1f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "budget_gb": args.budget_gb,
            "n_accesses": len(accesses),
            "kv_bytes_per_token": args.kv_bytes_per_token,
            "results": results,
        },
        open(out, "w"),
        indent=2,
    )
    print(f"\nsaved {out}")
    return results


if __name__ == "__main__":
    main()
