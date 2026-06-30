#!/usr/bin/env python3
"""DeepSeek V4 Flash: turn chunk-reuse lifecycle into real cache benefit.

Combines the per-chunk reuse statistics (from chunk_lifecycle.py) with
DeepSeek V4 Flash's real per-chunk prefill cost and MLA KV footprint, so the
abstract "prefills_saved" becomes concrete GPU-seconds saved and GB*seconds
of memory spent.

Per-chunk prefill cost
----------------------
A 4K-token chunk prefill on DeepSeek V4 Flash (MoE, 6/256 experts active).
Active params per token ~= dense(attn) + 6 experts. We use measured prefill
latency if --prefill-ms is given (from a real SGLang `redknot_mla` run),
otherwise estimate from FLOPs / H800 BF16 peak.

Decision rule (your intuition, formalised)
-----------------------------------------
Cache a chunk iff value > 0:
    benefit  = (reuse_count - 1) * prefill_seconds_per_chunk
    cost     = kv_GB_per_chunk * residency_seconds * mem_price
    cache if benefit / cost >= 1   (or simply reuse_count >= R_MIN)
"""

from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# DeepSeek V4 Flash config (from checkpoint config.json)
V4 = dict(
    layers=43,
    hidden=4096,
    n_heads=64,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    n_routed_experts=256,
    experts_per_tok=6,
    shared_experts=1,
    moe_inter=2048,  # per-expert intermediate (typical V4-Flash)
    total_params=340e9,
    active_params=13e9,
)
CHUNK_TOKENS = 4000
H800_BF16_TFLOPS = 989.0  # peak; effective ~0.4
EFF = 0.40


def mla_kv_bytes_per_token(dtype_bytes=2):
    return V4["layers"] * (V4["kv_lora_rank"] + V4["qk_rope_head_dim"]) * dtype_bytes


def estimate_prefill_seconds_per_chunk():
    """FLOPs-based estimate for one 4K chunk prefill (MoE active params)."""
    # 2 * active_params * tokens for the linear/MoE part (fwd)
    flops_linear = 2 * V4["active_params"] * CHUNK_TOKENS
    # attention FLOPs (MLA, full within chunk): ~ 2 * L * n_heads * T^2 * head_dim_eff
    head_dim_eff = V4["kv_lora_rank"] + V4["qk_rope_head_dim"]
    flops_attn = (
        2
        * V4["layers"]
        * V4["n_heads"]
        * (CHUNK_TOKENS**2)
        * head_dim_eff
        / V4["n_heads"]
    )
    total = flops_linear + flops_attn
    return total / (H800_BF16_TFLOPS * 1e12 * EFF)


def load_stats(path, kv_bytes):
    spec = importlib.util.spec_from_file_location(
        "cl", str(HERE / "chunk_lifecycle.py")
    )
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    stream = cl.load_musique_stream(path, None)
    return cl.analyze(stream, None, kv_bytes), len(stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/musique_ans_v1.0_dev.jsonl",
    )
    ap.add_argument(
        "--prefill-ms",
        type=float,
        default=None,
        help="measured per-4K-chunk prefill ms (from real SGLang redknot_mla run)",
    )
    ap.add_argument(
        "--seconds-per-request",
        type=float,
        default=1.0,
        help="wall-clock seconds between requests (residency time scaling)",
    )
    ap.add_argument(
        "--out", default=str(HERE / "figures/chunk_cache_benefit_dsv4.json")
    )
    args = ap.parse_args()

    kv_bytes = mla_kv_bytes_per_token()
    kv_gb_per_chunk = kv_bytes * CHUNK_TOKENS / 1e9
    if args.prefill_ms is not None:
        pf_sec = args.prefill_ms / 1000.0
        src = "measured"
    else:
        pf_sec = estimate_prefill_seconds_per_chunk()
        src = "estimated(FLOPs)"

    print(f"=== DeepSeek V4 Flash cache economics ===")
    print(f"MLA KV bytes/token     : {kv_bytes:,}")
    print(f"KV per 4K chunk        : {kv_gb_per_chunk * 1000:.1f} MB")
    print(f"prefill / 4K chunk     : {pf_sec * 1000:.1f} ms  ({src})")
    print(
        f"  (vs equiv MHA KV     : {V4['layers'] * 64 * 128 * 2 * 2:,} B/token, "
        f"{V4['layers'] * 64 * 128 * 2 * 2 / kv_bytes:.1f}x larger)"
    )

    stats, n_req = load_stats(args.path, kv_bytes)

    def policy(name, keep):
        kept = {c: s for c, s in stats.items() if keep(s)}
        saved_pf = sum(s["reuse_count"] - 1 for s in kept.values())
        gpu_sec_saved = saved_pf * pf_sec
        # memory*time cost: kv_GB * residency_seconds
        mem_gb_sec = sum(
            (s["kv_bytes"] * s["n_tokens"] / 1e9)
            * max(s["residency"], 1)
            * args.seconds_per_request
            for s in kept.values()
        )
        peak_gb = sum(s["kv_bytes"] * s["n_tokens"] / 1e9 for s in kept.values())
        return dict(
            policy=name,
            cached=len(kept),
            pct_cached=round(100 * len(kept) / len(stats), 1),
            prefills_saved=saved_pf,
            gpu_seconds_saved=round(gpu_sec_saved, 1),
            peak_kv_gb=round(peak_gb, 2),
            mem_gb_seconds=round(mem_gb_sec, 1),
            sec_saved_per_gb=round(gpu_sec_saved / max(peak_gb, 1e-9), 2),
        )

    rows = [
        policy("cache_all", lambda s: True),
        policy(
            "prefix_only",
            lambda s: s["non_prefix_ratio"] == 0 and s["reuse_count"] >= 2,
        ),
        policy("redknot(R>=3)", lambda s: s["reuse_count"] >= 3),
        policy("redknot(R>=5)", lambda s: s["reuse_count"] >= 5),
        policy("redknot(R>=10)", lambda s: s["reuse_count"] >= 10),
    ]
    hdr = f"{'policy':>16} | {'%cached':>7} {'prefills_saved':>14} {'GPU_sec_saved':>13} {'peak_KV_GB':>10} {'sec_saved/GB':>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['policy']:>16} | {r['pct_cached']:>6.1f}% {r['prefills_saved']:>14} "
            f"{r['gpu_seconds_saved']:>13.1f} {r['peak_kv_gb']:>10.2f} {r['sec_saved_per_gb']:>12.2f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "model": "DeepSeek-V4-Flash",
            "n_requests": n_req,
            "n_chunks": len(stats),
            "kv_bytes_per_token": kv_bytes,
            "prefill_sec_per_chunk": pf_sec,
            "prefill_source": src,
            "policies": rows,
        },
        open(out, "w"),
        indent=2,
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
