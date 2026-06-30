# Copyright 2024-2026 SGLang RedKnot Integration.
"""SegPaged v2 attention: mask-free per-head paged attention.

Given a :class:`PagedHeadKVCache` (each KV head physically holds only its
visible tokens) and a query, this runs attention over each head's *real*
sequence length. Because the invisible tokens are not stored, no attention
mask is needed — the head simply attends to fewer keys.

Two execution paths, numerically equivalent up to fp tolerance:

- **fused**: a single FA-3 ``flash_attn_varlen_func`` call packing every
  ``(kv-head, q-head)`` pair as a ragged sequence (``causal=False``: ordering
  is already encoded by the stored positions for the non-causal decode/probe
  case the reference covers).
- **reference**: exact per-head ``softmax(QKᵀ)V`` in PyTorch, CPU-friendly.

This module is self-contained and does not import the existing
``segpaged.py``; current benchmarks are unaffected.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from .page_table import PagedHeadKVCache
from .visible_plan import (
    HeadVisiblePlan,
    global_plan,
    local_plan,
)

logger = logging.getLogger(__name__)

try:
    from sgl_kernel.flash_attn import (
        flash_attn_varlen_func as _fa3_varlen,
        is_fa3_supported as _is_fa3_supported_hw,
    )

    _HAS_FA3 = True
except Exception as exc:  # pragma: no cover - environment specific
    _fa3_varlen = None
    _is_fa3_supported_hw = None
    _HAS_FA3 = False
    logger.info("SegPaged v2: FA-3 fused varlen unavailable (%s).", exc)


def is_fused_varlen_available() -> bool:
    """True iff the fused FA-3 varlen kernel is importable and supported."""
    if not _HAS_FA3:
        return False
    try:
        return bool(_is_fa3_supported_hw())
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Builder: dense per-head KV + per-head plans -> PagedHeadKVCache
# ──────────────────────────────────────────────────────────────────────────
def build_paged_cache(
    *,
    k_dense: torch.Tensor,  # [num_kv_heads, L, head_dim]
    v_dense: torch.Tensor,
    plans: List[HeadVisiblePlan],
    page_size: int = 64,
    layer: int = 0,
    storage=None,
) -> PagedHeadKVCache:
    """Build a paged store for one layer from dense KV + per-head plans.

    ``plans[h]`` declares the visible token positions for KV head ``h``.
    This is the single entry point all head classes share.
    """
    if k_dense.dim() != 3:
        raise ValueError(
            f"build_paged_cache expects [H, L, D] k_dense, got {tuple(k_dense.shape)}"
        )
    H, L, D = k_dense.shape
    if len(plans) != H:
        raise ValueError(f"build_paged_cache: {len(plans)} plans for {H} heads")
    cache = PagedHeadKVCache(
        num_kv_heads=H,
        head_dim=D,
        page_size=page_size,
        storage=storage,
        device=k_dense.device,
        dtype=k_dense.dtype,
    )
    for h in range(H):
        cache.store_head_segment(
            layer=layer,
            head=h,
            segment=0,
            k_dense=k_dense[h],
            v_dense=v_dense[h],
            plan=plans[h].to(k_dense.device),
        )
    return cache


def plans_from_policies(
    *,
    head_policy: List[str],
    seq_len: int,
    sink: int,
    recent: int,
    device: Optional[torch.device] = None,
) -> List[HeadVisiblePlan]:
    """Convenience: build global/local plans from a per-head policy list.

    ``head_policy[h]`` is ``"global"`` -> full context, anything else ->
    ``sink + recent`` window. Custom/retrieval heads should build their own
    plans via :func:`visible_plan.custom_plan`.
    """
    from .visible_plan import POLICY_GLOBAL

    plans: List[HeadVisiblePlan] = []
    for pol in head_policy:
        if pol == POLICY_GLOBAL:
            plans.append(global_plan(seq_len, device=device))
        else:
            plans.append(local_plan(seq_len, sink, recent, device=device))
    return plans


# ──────────────────────────────────────────────────────────────────────────
# Attention execution
# ──────────────────────────────────────────────────────────────────────────
def _sdpa_one_head(
    q: torch.Tensor,  # [Lq, D]
    k: torch.Tensor,  # [Lk, D]
    v: torch.Tensor,  # [Lk, D]
    sm_scale: float,
) -> torch.Tensor:
    if k.shape[0] == 0:
        return torch.zeros_like(q)
    scores = (q.float() @ k.float().transpose(0, 1)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(q.dtype)


@torch.no_grad()
def paged_attention(
    query: torch.Tensor,  # [Hq, Lq, D]
    cache: PagedHeadKVCache,
    *,
    layer: int,
    num_q_per_kv: int,
    sm_scale: float,
    use_fused: bool = True,
) -> torch.Tensor:
    """Per-head paged attention for one layer.

    For each KV head, gather its stored (visible-only) KV and run attention
    for the ``num_q_per_kv`` query heads mapped to it. No mask is built.

    Returns ``[Hq, Lq, D]``.
    """
    Hq, Lq, D = query.shape
    Hkv = cache.num_kv_heads
    if Hq != Hkv * num_q_per_kv:
        raise ValueError(
            f"paged_attention: Hq={Hq} != Hkv({Hkv}) * num_q_per_kv({num_q_per_kv})"
        )

    fused_ok = use_fused and is_fused_varlen_available() and query.is_cuda
    out = torch.zeros_like(query)

    if fused_ok:
        q_parts: List[torch.Tensor] = []
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for kv_h in range(Hkv):
            k_view, v_view = cache.gather_head(layer, kv_h)
            Lk = k_view.shape[0]
            for g in range(num_q_per_kv):
                qh = kv_h * num_q_per_kv + g
                q_parts.append(query[qh])
                k_parts.append(k_view)
                v_parts.append(v_view)
                cu_q.append(cu_q[-1] + Lq)
                cu_k.append(cu_k[-1] + Lk)
                max_q = max(max_q, Lq)
                max_k = max(max_k, Lk)
        q_packed = torch.cat(q_parts, dim=0).unsqueeze(1)
        k_packed = torch.cat(k_parts, dim=0).unsqueeze(1)
        v_packed = torch.cat(v_parts, dim=0).unsqueeze(1)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=query.device)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=query.device)
        result = _fa3_varlen(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens_q=cu_q_t,
            cu_seqlens_k=cu_k_t,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=sm_scale,
            causal=False,
            ver=3,
        )
        packed_out = result[0] if isinstance(result, tuple) else result
        packed_out = packed_out.squeeze(1)
        slot = 0
        for kv_h in range(Hkv):
            for g in range(num_q_per_kv):
                qh = kv_h * num_q_per_kv + g
                out[qh] = packed_out[cu_q[slot] : cu_q[slot + 1]]
                slot += 1
        return out

    # ── Reference path ──
    for kv_h in range(Hkv):
        k_view, v_view = cache.gather_head(layer, kv_h)
        for g in range(num_q_per_kv):
            qh = kv_h * num_q_per_kv + g
            out[qh] = _sdpa_one_head(query[qh], k_view, v_view, sm_scale)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Numerical-equivalence harness
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def dense_reference_attention(
    query: torch.Tensor,  # [Hq, Lq, D]
    k_dense: torch.Tensor,  # [Hkv, L, D]
    v_dense: torch.Tensor,
    plans: List[HeadVisiblePlan],
    *,
    num_q_per_kv: int,
    sm_scale: float,
) -> torch.Tensor:
    """Dense + additive-mask reference (the slow path SegPaged v2 replaces).

    Encodes each head's visible set as a boolean mask over the full dense
    ``[Hkv, L, D]`` KV — the baseline used to validate equivalence.
    """
    Hkv, L, D = k_dense.shape
    out = torch.zeros_like(query)
    for kv_h in range(Hkv):
        visible = torch.zeros(L, dtype=torch.bool, device=query.device)
        visible[plans[kv_h].positions.to(query.device)] = True
        k_h = k_dense[kv_h].float()
        v_h = v_dense[kv_h].float()
        for g in range(num_q_per_kv):
            qh = kv_h * num_q_per_kv + g
            scores = (query[qh].float() @ k_h.transpose(0, 1)) * sm_scale
            scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[qh] = (probs @ v_h).to(query.dtype)
    return out


@torch.no_grad()
def verify_against_dense(
    *,
    num_kv_heads: int,
    num_q_per_kv: int,
    head_dim: int,
    seq_len: int,
    sink: int,
    recent: int,
    global_ratio: float = 0.5,
    page_size: int = 64,
    q_len: int = 1,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Build random KV, run both paths, report cosine + token savings."""
    from .visible_plan import POLICY_GLOBAL, POLICY_LOCAL

    device = device or torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    Hkv = num_kv_heads
    Hq = Hkv * num_q_per_kv
    L = seq_len

    k_dense = torch.randn(Hkv, L, head_dim, generator=gen, device=device, dtype=dtype)
    v_dense = torch.randn(Hkv, L, head_dim, generator=gen, device=device, dtype=dtype)
    query = torch.randn(Hq, q_len, head_dim, generator=gen, device=device, dtype=dtype)
    sm_scale = 1.0 / (head_dim**0.5)

    n_global = max(1, int(round(Hkv * global_ratio)))
    policies = [POLICY_GLOBAL if h < n_global else POLICY_LOCAL for h in range(Hkv)]
    plans = plans_from_policies(
        head_policy=policies, seq_len=L, sink=sink, recent=recent, device=device
    )

    cache = build_paged_cache(
        k_dense=k_dense, v_dense=v_dense, plans=plans, page_size=page_size, layer=0
    )
    seg_out = paged_attention(
        query,
        cache,
        layer=0,
        num_q_per_kv=num_q_per_kv,
        sm_scale=sm_scale,
        use_fused=False,
    )
    ref_out = dense_reference_attention(
        query, k_dense, v_dense, plans, num_q_per_kv=num_q_per_kv, sm_scale=sm_scale
    )

    cos = torch.nn.functional.cosine_similarity(
        seg_out.float().flatten(), ref_out.float().flatten(), dim=0
    ).item()
    max_err = (seg_out.float() - ref_out.float()).abs().max().item()
    dense_tokens = Hkv * L
    seg_tokens = cache.stored_token_count()
    return {
        "cosine": cos,
        "max_abs_err": max_err,
        "dense_kv_tokens": dense_tokens,
        "segpaged_kv_tokens": seg_tokens,
        "kv_token_saving": 1.0 - (seg_tokens / dense_tokens) if dense_tokens else 0.0,
        "n_global_heads": n_global,
        "n_local_heads": Hkv - n_global,
    }


__all__ = [
    "is_fused_varlen_available",
    "build_paged_cache",
    "plans_from_policies",
    "paged_attention",
    "dense_reference_attention",
    "verify_against_dense",
]
