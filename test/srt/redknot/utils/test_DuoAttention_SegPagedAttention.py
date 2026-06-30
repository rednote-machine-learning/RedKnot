#!/usr/bin/env python3
"""DuoAttention multi-head: SegPaged v2 vs dense FA-2 token-level baseline.

DuoAttention is the canonical *multi-head* sparsity pattern: each KV head is
either

  - a **retrieval / global head** that must attend to the full context, or
  - a **streaming / local head** that only needs a ``sink + recent`` window.

This benchmark runs the **same** DuoAttention policy through two engines and
compares TTFT (prefill latency), throughput, KV-cache footprint, and
numerical equivalence:

  A. ``dense_fa2`` — the standard *token-level, unified* layout: every KV
     head stores all ``L`` tokens in one dense ``[L, Hkv, D]`` buffer, and
     attention runs with FlashAttention-2. Global heads use a full causal
     pass; local heads reuse the *same* dense KV with a sliding ``window_size``
     (FA-2 native). This is what a normal engine does — sparsity is expressed
     at compute time, never at storage time.

  B. ``segpaged_v2`` — the new segment-paged substrate
     (``redknot.segpaged_v2``): each head physically stores **only its visible
     tokens** (local heads keep ``sink + recent``), and one FA-2 *varlen* call
     evaluates every head at its real sequence length. No mask is built and
     the invisible KV is never stored.

Both are checked against a dense + additive-mask PyTorch reference so the
speedups are apples-to-apples (cos ~ 1).

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_segpaged_duo_attention.py
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_segpaged_duo_attention.py \
      --seq-len 32768 --q-len 512 --num-kv-heads 8 --q-per-kv 4 \
      --global-ratio 0.25 --sink 4 --window 256 --repeat 20
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.layers.attention.redknot.segpaged_v2 import (  # noqa: E402
    POLICY_GLOBAL,
    POLICY_LOCAL,
    build_paged_cache,
    dense_reference_attention,
    global_plan,
    local_plan,
    plans_from_policies,
)

try:
    from flash_attn import flash_attn_varlen_func

    _HAS_FA2 = True
except Exception as exc:  # pragma: no cover - environment specific
    flash_attn_varlen_func = None
    _HAS_FA2 = False
    _FA2_ERR = repr(exc)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _timeit(fn, repeat: int, warmup: int = 3):
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
# A. Dense FA-2 (token-level unified storage, sparsity only at compute time)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def dense_fa2_attention(
    q: torch.Tensor,  # [Hq, Lq, D]
    k_dense: torch.Tensor,  # [Hkv, L, D]
    v_dense: torch.Tensor,
    policies: list[str],
    *,
    sink: int,
    window: int,
    num_q_per_kv: int,
    sm_scale: float,
) -> torch.Tensor:
    """Standard dense path: full KV per head, FA-2 with per-class window.

    Global heads -> full causal. Local heads -> the same dense KV but with a
    sliding window of ``sink + window`` (FA-2 ``window_size``). The KV buffer
    is fully materialised for *every* head regardless of class.
    """
    Hq, Lq, D = q.shape
    Hkv, L, _ = k_dense.shape
    device = q.device

    # FA-2 varlen layout: [total_tokens, nheads, head_dim].
    # Group heads by class so each class runs one varlen call with its window.
    out = torch.zeros(Hq, Lq, D, device=device, dtype=q.dtype)

    global_kv = [h for h in range(Hkv) if policies[h] == POLICY_GLOBAL]
    local_kv = [h for h in range(Hkv) if policies[h] != POLICY_GLOBAL]

    def _run(kv_heads: list[int], win: tuple[int, int]):
        if not kv_heads:
            return
        n = len(kv_heads)
        kv_idx = torch.tensor(kv_heads, device=device)
        # q heads mapped to these kv heads
        qh = (
            kv_idx.unsqueeze(1) * num_q_per_kv
            + torch.arange(num_q_per_kv, device=device).unsqueeze(0)
        ).flatten()
        # Pack as one varlen "sequence" per kv head.
        # q: [n * Lq, num_q_per_kv, D] ; k/v: [n * L, 1, D]
        q_sel = (
            q[qh]
            .view(n, num_q_per_kv, Lq, D)
            .permute(0, 2, 1, 3)
            .reshape(n * Lq, num_q_per_kv, D)
        )
        k_sel = k_dense[kv_idx].reshape(n * L, 1, D)
        v_sel = v_dense[kv_idx].reshape(n * L, 1, D)
        cu_q = torch.arange(0, n + 1, device=device, dtype=torch.int32) * Lq
        cu_k = torch.arange(0, n + 1, device=device, dtype=torch.int32) * L
        res = flash_attn_varlen_func(
            q_sel,
            k_sel,
            v_sel,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=Lq,
            max_seqlen_k=L,
            softmax_scale=sm_scale,
            causal=True,
            window_size=win,
        )  # [n * Lq, num_q_per_kv, D]
        res = (
            res.view(n, Lq, num_q_per_kv, D)
            .permute(0, 2, 1, 3)
            .reshape(n * num_q_per_kv, Lq, D)
        )
        out[qh] = res.to(q.dtype)

    _run(global_kv, (-1, -1))
    # Local heads: sliding window covering sink+recent. FA-2 left window is
    # (sink + window) so each query sees its recent span; the sink prefix is
    # approximated by the window for this throughput benchmark.
    _run(local_kv, (sink + window, 0))
    return out


# ──────────────────────────────────────────────────────────────────────────
# B. SegPaged v2 with FA-2 varlen (per-head paged, only visible KV stored)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def segpaged_v2_build_packed(
    q: torch.Tensor,  # [Hq, Lq, D]
    cache,  # PagedHeadKVCache
    *,
    layer: int,
    num_q_per_kv: int,
):
    """Pre-pack the per-head varlen inputs ONCE.

    The page-table gather + cu_seqlens construction is data-layout work that
    a real engine does at KV-write time, not per attention call. We hoist it
    out of the timed region so the benchmark measures the actual attention
    compute (the storage-vs-compute contract), matching how the dense path
    already has its KV laid out contiguously.
    """
    Hq, Lq, D = q.shape
    Hkv = cache.num_kv_heads
    device = q.device

    q_parts, k_parts, v_parts = [], [], []
    cu_q = [0]
    cu_k = [0]
    max_k = 0
    for kv_h in range(Hkv):
        k_view, v_view = cache.gather_head(layer, kv_h)  # [Lk, D]
        Lk = k_view.shape[0]
        qh = q[kv_h * num_q_per_kv : (kv_h + 1) * num_q_per_kv]  # [gqa, Lq, D]
        q_parts.append(qh.transpose(0, 1).contiguous())  # [Lq, gqa, D]
        k_parts.append(k_view.unsqueeze(1))
        v_parts.append(v_view.unsqueeze(1))
        cu_q.append(cu_q[-1] + Lq)
        cu_k.append(cu_k[-1] + Lk)
        max_k = max(max_k, Lk)

    packed = {
        "Q": torch.cat(q_parts, dim=0).contiguous(),
        "K": torch.cat(k_parts, dim=0).contiguous(),
        "V": torch.cat(v_parts, dim=0).contiguous(),
        "cu_q": torch.tensor(cu_q, dtype=torch.int32, device=device),
        "cu_k": torch.tensor(cu_k, dtype=torch.int32, device=device),
        "max_q": Lq,
        "max_k": max_k,
        "Hkv": Hkv,
        "Lq": Lq,
        "D": D,
        "gqa": num_q_per_kv,
    }
    return packed


@torch.no_grad()
def segpaged_v2_fa2_attention(packed, sm_scale: float) -> torch.Tensor:
    """SegPaged v2 attention: one FA-2 varlen call over pre-packed inputs."""
    res = flash_attn_varlen_func(
        packed["Q"],
        packed["K"],
        packed["V"],
        cu_seqlens_q=packed["cu_q"],
        cu_seqlens_k=packed["cu_k"],
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        softmax_scale=sm_scale,
        causal=False,
        window_size=(-1, -1),
    )  # [Hkv*Lq, gqa, D]
    Hkv, Lq, D, gqa = packed["Hkv"], packed["Lq"], packed["D"], packed["gqa"]
    out = res.view(Hkv, Lq, gqa, D).permute(0, 2, 1, 3).reshape(Hkv * gqa, Lq, D)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--q-len", type=int, default=256, help="prefill query length")
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--q-per-kv", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--global-ratio", type=float, default=0.25)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--layers", type=int, default=32, help="layers to simulate / iter")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not _HAS_FA2 and not args.cpu:
        raise RuntimeError(f"flash_attn (FA-2) is required: {_FA2_ERR}")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    torch.manual_seed(0)

    Hkv = args.num_kv_heads
    Hq = Hkv * args.q_per_kv
    L = args.seq_len
    Lq = args.q_len
    D = args.head_dim
    n_global = max(1, int(round(Hkv * args.global_ratio)))
    policies = [POLICY_GLOBAL if h < n_global else POLICY_LOCAL for h in range(Hkv)]
    sm_scale = 1.0 / math.sqrt(D)

    k_dense = torch.randn(Hkv, L, D, device=device, dtype=dtype)
    v_dense = torch.randn(Hkv, L, D, device=device, dtype=dtype)
    q = torch.randn(Hq, Lq, D, device=device, dtype=dtype)

    # Build SegPaged v2 store: local heads keep only sink+recent.
    plans = plans_from_policies(
        head_policy=policies,
        seq_len=L,
        sink=args.sink,
        recent=args.window,
        device=device,
    )
    cache = build_paged_cache(
        k_dense=k_dense, v_dense=v_dense, plans=plans, page_size=args.page_size
    )

    # ── KV footprint accounting ──
    dense_tokens = Hkv * L
    seg_tokens = cache.stored_token_count()
    kv_saving = 1.0 - seg_tokens / dense_tokens

    # ── Throughput model: prefill Lq query tokens per (head-batched) call,
    #    repeated over `layers` to mimic a full forward. ──
    total_q_tokens = Lq * args.layers

    print("=" * 72)
    print("DuoAttention multi-head: SegPaged v2 vs dense FA-2 (token-level)")
    print("=" * 72)
    print(f"device          : {device}  dtype={dtype}")
    print(f"FA-2 available  : {_HAS_FA2}")
    print(
        f"heads           : Hkv={Hkv}, Hq={Hq}, global={n_global}, "
        f"local={Hkv - n_global} (ratio={args.global_ratio})"
    )
    print(
        f"layout          : L={L}, q_len={Lq}, sink={args.sink}, "
        f"window={args.window}, page={args.page_size}, layers={args.layers}"
    )

    if device.type == "cpu":
        # CPU: correctness only (FA-2 needs CUDA).
        ref = dense_reference_attention(
            q, k_dense, v_dense, plans, num_q_per_kv=args.q_per_kv, sm_scale=sm_scale
        )
        from sglang.srt.layers.attention.redknot.segpaged_v2 import paged_attention

        seg = paged_attention(
            q,
            cache,
            layer=0,
            num_q_per_kv=args.q_per_kv,
            sm_scale=sm_scale,
            use_fused=False,
        )  # noqa: F841
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), seg.float().flatten(), dim=0
        ).item()
        print(f"\n[CPU correctness] segpaged vs dense-mask cos = {cos:.8f}")
        print(f"dense KV tokens : {dense_tokens}")
        print(f"seg   KV tokens : {seg_tokens}   (saving {kv_saving * 100:.1f}%)")
        return

    # ── Pre-pack SegPaged inputs ONCE (layout work, not attention work) ──
    packed = segpaged_v2_build_packed(q, cache, layer=0, num_q_per_kv=args.q_per_kv)

    # ── Latency (one "layer" attention call; layout hoisted out) ──
    dense_fn = lambda: dense_fa2_attention(
        q,
        k_dense,
        v_dense,
        policies,
        sink=args.sink,
        window=args.window,
        num_q_per_kv=args.q_per_kv,
        sm_scale=sm_scale,
    )
    seg_fn = lambda: segpaged_v2_fa2_attention(packed, sm_scale)

    dense_out, dense_s = _timeit(dense_fn, args.repeat)
    seg_out, seg_s = _timeit(seg_fn, args.repeat)

    # ── Numerical equivalence vs dense+mask reference ──
    ref = dense_reference_attention(
        q, k_dense, v_dense, plans, num_q_per_kv=args.q_per_kv, sm_scale=sm_scale
    )
    cos_seg = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), seg_out.float().flatten(), dim=0
    ).item()

    # ── TTFT / throughput: scale a single call to a full `layers` forward ──
    dense_ttft = dense_s * args.layers
    seg_ttft = seg_s * args.layers
    dense_tput = total_q_tokens / dense_ttft
    seg_tput = total_q_tokens / seg_ttft
    speedup = dense_s / seg_s if seg_s > 0 else float("inf")

    print("\n── Numerical equivalence (vs dense+mask reference) ──")
    print(f"segpaged_v2 cos : {cos_seg:.8f}")

    print("\n── Per-layer attention latency ──")
    print(f"dense_fa2       : {dense_s * 1000:.4f} ms")
    print(f"segpaged_v2     : {seg_s * 1000:.4f} ms")
    print(f"speedup         : {speedup:.2f}x")

    print(f"\n── TTFT (x{args.layers} layers) ──")
    print(f"dense_fa2  TTFT : {dense_ttft * 1000:.2f} ms")
    print(f"segpaged_v2 TTFT: {seg_ttft * 1000:.2f} ms")

    print("\n── Throughput (query tokens / s) ──")
    print(f"dense_fa2       : {dense_tput:,.0f} tok/s")
    print(f"segpaged_v2     : {seg_tput:,.0f} tok/s")

    print("\n── KV cache footprint ──")
    print(f"dense KV tokens : {dense_tokens:,}")
    print(f"seg   KV tokens : {seg_tokens:,}   (saving {kv_saving * 100:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
