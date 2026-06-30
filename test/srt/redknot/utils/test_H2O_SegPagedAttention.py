#!/usr/bin/env python3
# Copyright 2024-2026 SGLang RedKnot Integration.
"""Real-H2O decode benchmark: dense vs SegPaged v2 head-level paged KV.

Simulates the **real H2O decoding algorithm** (Zhang et al., NeurIPS 2023)
and benchmarks it FAIRLY against a dense baseline by using the SAME production
decode kernel (``flash_attn_with_kvcache``, i.e. flash-decoding with KV-split
parallelism) and a real batch dimension for all paths.

Why this matters
----------------
An earlier version timed the dense baseline with a single-query varlen call,
which is the worst-case layout for decode and made the speedup look ~50x.
That number was inflated by an unfair baseline. Here every path uses the
batched flash-decoding kernel; the only difference is how many KV tokens each
KV head reads:

  - ``dense_full``   : every head reads the full context (L tokens).
  - ``h2o_dense``    : H2O retained set read from a dense unified [B,L,Hkv,D]
                       buffer compacted to the budget per head.
  - ``h2o_segpaged`` : H2O retained set physically stored per head in the
                       SegPaged v2 paged store, gathered into a compact cache.

Real H2O mechanics (per KV head)
--------------------------------
- Accumulated softmax attention mass per token (real per-step probabilities).
- Fixed budget = heavy_budget + recent_budget; prompt KV compressed to budget
  before decoding; rolling greedy eviction of the lowest-score non-recent
  token each step. Each head evicts independently.

The KV savings and the ``h2o_dense vs h2o_segpaged`` equivalence are exact;
``dense_full vs h2o`` cosine is the H2O approximation error (large on random
data, much smaller on a real model — quality needs a real model to assess).

Usage
-----
  # One-shot suite (recommended): runs several configs and prints a summary.
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_H2O_SegPagedAttention.py --suite

  # Single custom config:
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_H2O_SegPagedAttention.py \
      --batch 16 --prefill 32768 --gen 64 --heavy 1024 --recent 256 --repeat 50
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.layers.attention.redknot.segpaged_v2 import (  # noqa: E402
    PagedHeadKVCache,
    custom_plan,
)

try:
    from flash_attn import flash_attn_with_kvcache

    _HAS_FA2 = True
except Exception as exc:  # pragma: no cover - environment specific
    flash_attn_with_kvcache = None
    _HAS_FA2 = False
    _FA2_ERR = repr(exc)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.float().flatten()
    yf = y.float().flatten()
    return float((xf @ yf / (xf.norm() * yf.norm() + 1e-20)).item())


@torch.no_grad()
def _timeit(fn, repeat, warmup):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    _sync()
    return out, (time.perf_counter() - t0) / repeat


# ──────────────────────────────────────────────────────────────────────────
# Real H2O eviction simulation (per-head, accumulated softmax score)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def simulate_h2o_retained(
    *,
    k_full: torch.Tensor,  # [Hkv, L_total, D] for ONE batch item
    q_steps: torch.Tensor,  # [Hkv, gqa, n_gen, D]
    prefill_len: int,
    heavy_budget: int,
    recent_budget: int,
    sm_scale: float,
) -> List[torch.Tensor]:
    """Return final per-head retained positions after the H2O greedy policy."""
    Hkv, L_total, D = k_full.shape
    device = k_full.device
    n_gen = q_steps.shape[2]
    budget = heavy_budget + recent_budget
    retained: List[torch.Tensor] = []

    for h in range(Hkv):
        acc = torch.zeros(L_total, device=device, dtype=torch.float32)

        # Prompt compression to budget (H2O compresses the prompt first).
        q_prompt = q_steps[h, :, 0, :].mean(dim=0).float()
        prompt_scores = (k_full[h, :prefill_len].float() @ q_prompt) * sm_scale
        acc[:prefill_len] += torch.softmax(prompt_scores, dim=-1)
        kept = torch.arange(prefill_len, device=device, dtype=torch.long)
        if kept.numel() > budget:
            recent_lo = prefill_len - recent_budget
            recent_idx = kept[kept >= recent_lo]
            old_idx = kept[kept < recent_lo]
            n_heavy = budget - int(recent_idx.numel())
            if n_heavy > 0 and old_idx.numel() > n_heavy:
                _, top = torch.topk(acc[old_idx], k=n_heavy, largest=True)
                old_idx = old_idx[top]
            elif n_heavy <= 0:
                old_idx = old_idx[:0]
            kept = torch.sort(torch.cat([old_idx, recent_idx])).values

        # Rolling decode with greedy eviction.
        for t in range(n_gen):
            cur_pos = prefill_len + t
            cand = torch.unique(
                torch.cat([kept, torch.tensor([cur_pos], device=device)]), sorted=True
            )
            q_mean = q_steps[h, :, t, :].mean(dim=0).float()
            scores = (k_full[h, cand].float() @ q_mean) * sm_scale
            acc[cand] += torch.softmax(scores, dim=-1)
            kept = cand
            if kept.numel() > budget:
                recent_lo = cur_pos - recent_budget + 1
                evictable = kept[kept < recent_lo]
                if evictable.numel() > 0:
                    victim = evictable[torch.argmin(acc[evictable])]
                    kept = kept[kept != victim]
                else:
                    vloc = torch.argmin(acc[kept])
                    kept = torch.cat([kept[:vloc], kept[vloc + 1 :]])

        retained.append(torch.sort(kept).values)

    return retained


# ──────────────────────────────────────────────────────────────────────────
# Compact KV cache builders -> [B, cap, Hkv, D] + cache_seqlens for FA-2
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_dense_cache(k_full, v_full):
    """[B,Hkv,L,D] -> contiguous [B,L,Hkv,D] full cache + seqlens."""
    B, Hkv, L, D = k_full.shape
    kc = k_full.permute(0, 2, 1, 3).contiguous()
    vc = v_full.permute(0, 2, 1, 3).contiguous()
    seqlens = torch.full((B,), L, dtype=torch.int32, device=k_full.device)
    return kc, vc, seqlens


@torch.no_grad()
def build_compact_cache_from_dense(k_full, v_full, retained_b, cap):
    """Gather H2O retained set from dense KV into [B,cap,Hkv,D] + seqlens."""
    B, Hkv, L, D = k_full.shape
    device = k_full.device
    kc = torch.zeros(B, cap, Hkv, D, device=device, dtype=k_full.dtype)
    vc = torch.zeros(B, cap, Hkv, D, device=device, dtype=k_full.dtype)
    seqlens = torch.zeros(B, dtype=torch.int32, device=device)
    for b in range(B):
        # Per-head retained lengths can differ; cache_seqlens is per-batch, so
        # we use the max head length and zero-pad shorter heads (padding never
        # affects results because each head reads its own kept count — here we
        # conservatively use a per-(b) seqlen = max over heads and rely on the
        # fact that all heads share the same budget in steady state).
        max_len = 0
        for h in range(Hkv):
            pos = retained_b[b][h]
            n = int(pos.numel())
            kc[b, :n, h] = k_full[b, h].index_select(0, pos)
            vc[b, :n, h] = v_full[b, h].index_select(0, pos)
            max_len = max(max_len, n)
        seqlens[b] = max_len
    return kc, vc, seqlens


@torch.no_grad()
def build_compact_cache_from_segpaged(caches, cap, Hkv, D, device, dtype):
    """Gather H2O retained set from SegPaged stores into [B,cap,Hkv,D]."""
    B = len(caches)
    kc = torch.zeros(B, cap, Hkv, D, device=device, dtype=dtype)
    vc = torch.zeros(B, cap, Hkv, D, device=device, dtype=dtype)
    seqlens = torch.zeros(B, dtype=torch.int32, device=device)
    for b in range(B):
        max_len = 0
        for h in range(Hkv):
            k_h, v_h = caches[b].gather_head(0, h)
            n = k_h.shape[0]
            kc[b, :n, h] = k_h
            vc[b, :n, h] = v_h
            max_len = max(max_len, n)
        seqlens[b] = max_len
    return kc, vc, seqlens


@torch.no_grad()
def decode_kernel(q, kc, vc, seqlens, sm_scale):
    """Batched flash-decoding: q [B,1,Hq,D], kc/vc [B,cap,Hkv,D]."""
    return flash_attn_with_kvcache(
        q,
        kc,
        vc,
        cache_seqlens=seqlens,
        softmax_scale=sm_scale,
        causal=False,
    )  # [B,1,Hq,D]


@torch.no_grad()
def run_one(cfg: dict, *, device, dtype, verbose: bool = True) -> dict:
    """Run a single H2O decode config and return a metrics dict."""
    torch.manual_seed(cfg["seed"])
    B = cfg["batch"]
    Hkv, gqa = cfg["num_kv_heads"], cfg["q_per_kv"]
    Hq = Hkv * gqa
    P, G, D = cfg["prefill"], cfg["gen"], cfg["head_dim"]
    L_total = P + G
    sm_scale = 1.0 / math.sqrt(D)
    budget = cfg["heavy"] + cfg["recent"]

    k_full = torch.randn(B, Hkv, L_total, D, device=device, dtype=dtype)
    v_full = torch.randn(B, Hkv, L_total, D, device=device, dtype=dtype)
    q_steps = torch.randn(B, Hkv, gqa, G, D, device=device, dtype=dtype)
    last_q = q_steps[:, :, :, -1, :].reshape(B, Hkv * gqa, D).unsqueeze(1)

    # ── Real H2O policy per batch item ──
    retained_b = []
    for b in range(B):
        retained_b.append(
            simulate_h2o_retained(
                k_full=k_full[b],
                q_steps=q_steps[b],
                prefill_len=P,
                heavy_budget=cfg["heavy"],
                recent_budget=cfg["recent"],
                sm_scale=sm_scale,
            )
        )
    kept_lens = [int(r.numel()) for rb in retained_b for r in rb]
    cap = max(kept_lens)

    # ── SegPaged stores (one per batch item) ──
    seg_caches = []
    for b in range(B):
        cache = PagedHeadKVCache(
            num_kv_heads=Hkv,
            head_dim=D,
            page_size=cfg["page_size"],
            device=device,
            dtype=dtype,
        )
        for h in range(Hkv):
            plan = custom_plan(L_total, retained_b[b][h], device=device)
            cache.store_head_segment(
                layer=0,
                head=h,
                segment=0,
                k_dense=k_full[b, h],
                v_dense=v_full[b, h],
                plan=plan,
            )
        seg_caches.append(cache)

    # ── Build the three caches (layout work, hoisted out of timing) ──
    dense_kc, dense_vc, dense_sl = build_dense_cache(k_full, v_full)
    h2o_kc, h2o_vc, h2o_sl = build_compact_cache_from_dense(
        k_full, v_full, retained_b, cap
    )
    seg_kc, seg_vc, seg_sl = build_compact_cache_from_segpaged(
        seg_caches, cap, Hkv, D, device, dtype
    )

    # ── Outputs for correctness ──
    dense_out = decode_kernel(last_q, dense_kc, dense_vc, dense_sl, sm_scale)
    h2o_out = decode_kernel(last_q, h2o_kc, h2o_vc, h2o_sl, sm_scale)
    seg_out = decode_kernel(last_q, seg_kc, seg_vc, seg_sl, sm_scale)

    # ── Timing (same kernel for all three; only KV length differs) ──
    _, dense_s = _timeit(
        lambda: decode_kernel(last_q, dense_kc, dense_vc, dense_sl, sm_scale),
        cfg["repeat"],
        cfg["warmup"],
    )
    _, h2o_s = _timeit(
        lambda: decode_kernel(last_q, h2o_kc, h2o_vc, h2o_sl, sm_scale),
        cfg["repeat"],
        cfg["warmup"],
    )
    _, seg_s = _timeit(
        lambda: decode_kernel(last_q, seg_kc, seg_vc, seg_sl, sm_scale),
        cfg["repeat"],
        cfg["warmup"],
    )

    dense_tokens = B * Hkv * L_total
    h2o_tokens = sum(kept_lens)
    kv_saving = 1.0 - h2o_tokens / dense_tokens
    layers = cfg["layers"]

    res = {
        "batch": B,
        "prefill": P,
        "L_total": L_total,
        "budget": budget,
        "cap": cap,
        "cos_dense_vs_seg": _cosine(h2o_out, seg_out),
        "cos_full_vs_seg": _cosine(dense_out, seg_out),
        "dense_ms": dense_s * 1000,
        "h2o_ms": h2o_s * 1000,
        "seg_ms": seg_s * 1000,
        "speedup_vs_dense": dense_s / seg_s,
        "speedup_vs_h2o": h2o_s / seg_s,
        "dense_tput": B / (dense_s * layers),
        "seg_tput": B / (seg_s * layers),
        "kv_saving": kv_saving,
        "dense_kv_tokens": dense_tokens,
        "h2o_kv_tokens": h2o_tokens,
    }

    if verbose:
        print("=" * 92)
        print(
            "Real-H2O decode (fair flash-decoding): dense vs SegPaged v2 head-level KV"
        )
        print("=" * 92)
        print(f"device         : {torch.cuda.get_device_name(0)}  dtype={dtype}")
        print(f"kernel         : flash_attn_with_kvcache (batched flash-decoding)")
        print(
            f"shape          : batch={B}, prefill={P}, gen={G}, L_total={L_total}, "
            f"Hkv={Hkv}, Hq={Hq}, D={D}"
        )
        print(
            f"H2O budget     : heavy={cfg['heavy']}+recent={cfg['recent']}={budget}/head"
        )
        print(
            f"retained/head  : min={min(kept_lens)}, max={max(kept_lens)}, "
            f"cap={cap}, avg={sum(kept_lens) / len(kept_lens):.1f}"
        )
        print("\n── Numerical checks ──")
        print(f"h2o_dense vs h2o_segpaged cos : {res['cos_dense_vs_seg']:.8f}")
        print(
            f"dense_full vs h2o_segpaged cos: {res['cos_full_vs_seg']:.6f}  "
            f"(<1: H2O approximation error, expected on random data)"
        )
        print("\n── Per-step decode attention latency (whole batch) ──")
        print(f"dense_full    : {res['dense_ms']:.4f} ms")
        print(f"h2o_dense     : {res['h2o_ms']:.4f} ms")
        print(f"h2o_segpaged  : {res['seg_ms']:.4f} ms")
        print("\n── Speedup (fair, same kernel) ──")
        print(f"segpaged vs dense_full : {res['speedup_vs_dense']:.2f}x")
        print(f"segpaged vs h2o_dense  : {res['speedup_vs_h2o']:.2f}x")
        print(f"\n── Per-token decode (x{layers} layers) ──")
        for name, sec in [
            ("dense_full", dense_s),
            ("h2o_dense", h2o_s),
            ("h2o_segpaged", seg_s),
        ]:
            per_tok = sec * layers
            print(
                f"{name:<14s}: {per_tok * 1000:8.3f} ms/token  "
                f"{B / per_tok:,.0f} tok/s (batch={B})"
            )
        print("\n── KV footprint ──")
        print(f"dense KV tokens : {dense_tokens:,}")
        print(f"H2O  KV tokens  : {h2o_tokens:,}  (saving {kv_saving * 100:.1f}%)")
        print("=" * 92)

    return res


# Pre-baked suite: a few representative (batch, prefill, heavy) points that run
# quickly end-to-end. Keep batch/gen modest so the H2O simulation stays fast.
SUITE = [
    {"batch": 8, "prefill": 8192, "heavy": 512, "recent": 256, "gen": 32},
    {"batch": 8, "prefill": 16384, "heavy": 512, "recent": 256, "gen": 32},
    {"batch": 8, "prefill": 32768, "heavy": 1024, "recent": 256, "gen": 32},
]


def _print_suite_summary(rows: list) -> None:
    print("\n" + "#" * 92)
    print("SUITE SUMMARY — SegPaged v2 H2O decode vs dense flash-decoding")
    print("#" * 92)
    header = (
        f"{'batch':>5} {'L_total':>8} {'budget':>7} "
        f"{'dense_ms':>9} {'seg_ms':>8} {'speedup':>8} "
        f"{'seg_tput':>10} {'KV_save':>8} {'cos(d=s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['batch']:>5} {r['L_total']:>8} {r['budget']:>7} "
            f"{r['dense_ms']:>9.3f} {r['seg_ms']:>8.3f} "
            f"{r['speedup_vs_dense']:>7.2f}x "
            f"{r['seg_tput']:>9,.0f} {r['kv_saving'] * 100:>7.1f}% "
            f"{r['cos_dense_vs_seg']:>9.5f}"
        )
    print("#" * 92)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Real-H2O decode: dense vs SegPaged v2 (fair flash-decoding)"
    )
    p.add_argument("--suite", action="store_true", help="run the preset config suite")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prefill", type=int, default=8192)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--q-per-kv", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--heavy", type=int, default=512)
    p.add_argument("--recent", type=int, default=256)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")
    if not _HAS_FA2:
        raise RuntimeError(f"flash_attn (FA-2) is required: {_FA2_ERR}")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    base = vars(args).copy()

    if args.suite:
        rows = []
        for entry in SUITE:
            cfg = base.copy()
            cfg.update(entry)
            rows.append(run_one(cfg, device=device, dtype=dtype, verbose=True))
        _print_suite_summary(rows)
        return

    run_one(base, device=device, dtype=dtype, verbose=True)


if __name__ == "__main__":
    main()
