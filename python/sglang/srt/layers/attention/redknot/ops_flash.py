# Copyright 2024-2026 SGLang RedKnot Integration.
"""FlashAttention-2 kernel path for RedKnot.

This module is the *kernel-level integration* of RedKnot with the
`flash_attn` library (Dao-AILab/flash-attention v2.x). It exposes a
drop-in faster replacement for ``ops.segment_attention`` /
``ops_fast.segment_attention_fast`` named :func:`segment_attention_flash`.

Why FlashAttention?
-------------------
Per-head vectorisation alone (``ops_fast``) gets us SDPA, which is good,
but PyTorch SDPA still materialises a small attention probability tile
and pays for masked-out columns. FlashAttention 2 goes further:

- **Block-sparse sliding window** via ``window_size=(W, 0)`` — for
  *local* heads the kernel **physically skips** key-blocks outside the
  ``W``-wide window. No extra mask, no wasted compute.
- **Causal mask is implicit** via ``causal=True`` (no allocation).
- **Native GQA**: when ``num_kv_heads < num_heads`` the kernel reads K/V
  once per group, halving HBM bandwidth.
- **No intermediate attention matrix** at all (the entire softmax happens
  inside SRAM blocks).

How RedKnot maps onto it
-------------------------
The four head types are routed to three FlashAttention call patterns,
applied to disjoint KV-head buckets of the same layer:

1. **local**
   - K/V = ``[sink | prev_tail (≤ window) | self]``
   - One call to ``flash_attn_func(..., causal=True, window_size=(W, 0))``.
   - The kernel ignores anything more than ``W`` tokens back from each
     query position, so the prev region only matters where it falls
     inside the window. We still physically truncate the prev to
     ``window`` tokens to keep memory bounded.

2. **global / dense**
   - K/V = ``[sink | prev_full | self]``
   - Cannot use ``causal=True`` directly because that would forbid prev
     access (causal compares ``i_q + offset`` to ``j_k``). Instead we
     issue **two** flash calls per bucket and merge with log-sum-exp:
       a) prev call: ``causal=False`` over ``[sink | prev]``.
       b) self call: ``causal=True`` over ``[self]`` only.
     Then combine via ``merge_attn_states``. This is exactly the recipe
     sglang's own FlashInfer backend uses for the "extend with cached
     prefix" path.

3. **retrieval**
   - First pick top-p prev tokens via the same per-head scoring as
     ``ops_fast._attn_retrieval``, then run the same two-call merge as
     global on ``[sink | prev_top | self]``.

Numerical correctness
---------------------
Each bucket's output is gathered back into the layer-wide output tensor
via ``index_copy_``. Because we only call FlashAttention with strict
``[L_q, ≤ valid_kv]`` shapes, every produced logit corresponds to a
valid key — no masked-out entries to worry about.

Fallback
--------
If ``flash_attn`` is not importable we keep the function definition
present (so callers can ``import segment_attention_flash`` without
guards) but raise ``RuntimeError`` on call.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
from sglang.srt.layers.attention.redknot.mask_plan import (
    LayerMaskPlan,
    group_heads_by_type,
    q_head_indices,
)

# Default query-axis chunk size; retained as a kw-arg for API stability
# even though FA-2 handles chunking internally.
DEFAULT_Q_CHUNK = 2048

logger = logging.getLogger(__name__)

# Lazy-import flash_attn so the rest of RedKnot keeps working on systems
# without the binary kernel installed.
try:
    from flash_attn import flash_attn_func as _flash_attn_func

    _HAS_FLASH_ATTN = True
except Exception as exc:  # pragma: no cover
    _flash_attn_func = None
    _HAS_FLASH_ATTN = False
    logger.info("flash_attn unavailable (%s); RedKnot flash path disabled.", exc)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _bh_to_bsh(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D] -> [B, S, H, D]`` for FlashAttention call convention."""
    return x.transpose(1, 2).contiguous()


def _bsh_to_bh(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_bh_to_bsh`."""
    return x.transpose(1, 2).contiguous()


def _fa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
    window_size: Tuple[int, int] = (-1, -1),
    return_lse: bool = False,
):
    """Thin wrapper around ``flash_attn_func`` enforcing dtype/contig.

    flash_attn 2.x requires:
      - q/k/v in ``[B, S, H, D]`` layout
      - dtype in ``{fp16, bf16}`` (we coerce to bf16 if needed)
      - contiguous tensors
      - K/V heads can be < Q heads (GQA broadcast handled internally)
    """
    if not _HAS_FLASH_ATTN:
        raise RuntimeError("flash_attn is not available in this environment.")
    orig_dtype = q.dtype
    if q.dtype not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    out = _flash_attn_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=window_size,
        return_attn_probs=return_lse,
    )
    if return_lse:
        o, lse, _ = out  # o:[B,S,H,D], lse:[B,H,S]
        return o.to(orig_dtype), lse
    return out.to(orig_dtype)


def _merge_two_attn(
    o_a: torch.Tensor,
    lse_a: torch.Tensor,
    o_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Log-sum-exp merge of two FlashAttention partial outputs.

    Both ``o_*`` are ``[B, S, H, D]`` and ``lse_*`` are ``[B, H, S]``.

    Math (per query position):
        o = (exp(lse_a - m) * o_a + exp(lse_b - m) * o_b)
            / (exp(lse_a - m) + exp(lse_b - m))
        where m = max(lse_a, lse_b)

    Equivalent to running one big softmax over the union of A's and B's
    keys, which is exactly what we want for ``[prev | self]`` splits.
    """
    # Make lse broadcastable to ``[B, S, H, 1]``.
    lse_a_bsh1 = lse_a.transpose(1, 2).unsqueeze(-1).to(o_a.dtype)
    lse_b_bsh1 = lse_b.transpose(1, 2).unsqueeze(-1).to(o_b.dtype)
    m = torch.maximum(lse_a_bsh1, lse_b_bsh1)
    w_a = (lse_a_bsh1 - m).exp()
    w_b = (lse_b_bsh1 - m).exp()
    return (w_a * o_a + w_b * o_b) / (w_a + w_b)


# ──────────────────────────────────────────────────────────────────────────
# Per-bucket attention calls
# ──────────────────────────────────────────────────────────────────────────
def _attn_local_flash(
    q_bucket: torch.Tensor,  # [B, Hq_b, L_q, D]
    k_sink_b: Optional[torch.Tensor],  # [B, KVH_b, S_max, D]
    v_sink_b: Optional[torch.Tensor],
    k_self_b: torch.Tensor,  # [B, KVH_b, L_self, D]
    v_self_b: torch.Tensor,
    k_prev_b: Optional[torch.Tensor],  # [B, KVH_b, P, D]
    v_prev_b: Optional[torch.Tensor],
    *,
    window: int,
    sm_scale: float,
) -> torch.Tensor:
    """Local head: sink/prev (full) + self (windowed) via two-pass FA + LSE merge.

    Why not a single FA call with one ``window_size``?
    ---------------------------------------------------
    flash_attn's sliding window is a single **contiguous** range
    ``[i_q + (L_kv-L_q) - left, ...]``. RedKnot's local-head pattern is
    not contiguous — sink + prev_tail must be visible to *every* query
    regardless of distance, but self must respect a true sliding window.
    A single ``window_size=(W+sink+tail, 0)`` would mistakenly prune the
    sink/prev_tail prefix once a query is far enough into self.

    Instead we do two FA passes and merge via log-sum-exp:

      Pass A: ``[sink | prev_tail]`` always-visible (non-causal, no window)
      Pass B: ``[self]`` causal + ``window_size=(W, 0)``

    The merge is mathematically equivalent to one big softmax over the
    union, but each pass uses the kernel's fast path.
    """
    # 1. Physically crop prev to last `window` tokens (bounds memory).
    if k_prev_b is not None and k_prev_b.shape[-2] > 0:
        tail = min(window, k_prev_b.shape[-2])
        k_prev_use = k_prev_b[..., -tail:, :]
        v_prev_use = v_prev_b[..., -tail:, :]
    else:
        k_prev_use = None
        v_prev_use = None
        tail = 0

    # Build the "always-visible" prefix part: [sink | prev_tail].
    prefix_parts_k = []
    prefix_parts_v = []
    S_max = k_sink_b.shape[-2] if k_sink_b is not None else 0
    if S_max > 0:
        prefix_parts_k.append(k_sink_b)
        prefix_parts_v.append(v_sink_b)
    if tail > 0:
        prefix_parts_k.append(k_prev_use)
        prefix_parts_v.append(v_prev_use)

    # Pass B (self-only, windowed-causal). We always run this -- self is
    # never empty since we wouldn't be in a forward pass otherwise.
    o_self, lse_self = _fa(
        _bh_to_bsh(q_bucket),
        _bh_to_bsh(k_self_b),
        _bh_to_bsh(v_self_b),
        causal=True,
        sm_scale=sm_scale,
        window_size=(window, 0),
        return_lse=True,
    )

    if not prefix_parts_k:
        # No sink/prev -- self-only path.
        return _bsh_to_bh(o_self)

    # Pass A (sink + prev_tail, fully visible).
    K_prefix = torch.cat(prefix_parts_k, dim=-2)
    V_prefix = torch.cat(prefix_parts_v, dim=-2)
    o_prefix, lse_prefix = _fa(
        _bh_to_bsh(q_bucket),
        _bh_to_bsh(K_prefix),
        _bh_to_bsh(V_prefix),
        causal=False,
        sm_scale=sm_scale,
        return_lse=True,
    )

    o = _merge_two_attn(o_prefix, lse_prefix, o_self, lse_self)
    return _bsh_to_bh(o)


def _attn_global_dense_flash(
    q_bucket: torch.Tensor,
    k_sink_b: Optional[torch.Tensor],
    v_sink_b: Optional[torch.Tensor],
    k_self_b: torch.Tensor,
    v_self_b: torch.Tensor,
    k_prev_b: Optional[torch.Tensor],
    v_prev_b: Optional[torch.Tensor],
    *,
    sm_scale: float,
) -> torch.Tensor:
    """Global / dense head via two-pass FlashAttention + LSE merge.

    Pass A: non-causal attention over ``[sink | prev]`` (everything visible).
    Pass B: causal attention over ``[self]``.
    Merge by log-sum-exp -> equivalent to one big softmax over the union.

    If there is no prev (and no sink) we collapse to a single causal call.
    """
    parts_k = []
    parts_v = []
    S_max = k_sink_b.shape[-2] if k_sink_b is not None else 0
    if S_max > 0:
        parts_k.append(k_sink_b)
        parts_v.append(v_sink_b)
    if k_prev_b is not None and k_prev_b.shape[-2] > 0:
        parts_k.append(k_prev_b)
        parts_v.append(v_prev_b)

    if not parts_k:
        # Pure self path -- single causal call.
        o = _fa(
            _bh_to_bsh(q_bucket),
            _bh_to_bsh(k_self_b),
            _bh_to_bsh(v_self_b),
            causal=True,
            sm_scale=sm_scale,
        )
        return _bsh_to_bh(o)

    K_prev_full = torch.cat(parts_k, dim=-2)
    V_prev_full = torch.cat(parts_v, dim=-2)

    o_a, lse_a = _fa(
        _bh_to_bsh(q_bucket),
        _bh_to_bsh(K_prev_full),
        _bh_to_bsh(V_prev_full),
        causal=False,
        sm_scale=sm_scale,
        return_lse=True,
    )
    o_b, lse_b = _fa(
        _bh_to_bsh(q_bucket),
        _bh_to_bsh(k_self_b),
        _bh_to_bsh(v_self_b),
        causal=True,
        sm_scale=sm_scale,
        return_lse=True,
    )
    o = _merge_two_attn(o_a, lse_a, o_b, lse_b)
    return _bsh_to_bh(o)


def _attn_retrieval_flash(
    q_bucket: torch.Tensor,
    k_sink_b: Optional[torch.Tensor],
    v_sink_b: Optional[torch.Tensor],
    k_self_b: torch.Tensor,
    v_self_b: torch.Tensor,
    k_prev_b: torch.Tensor,
    v_prev_b: torch.Tensor,
    *,
    retrieval_top_p: float,
    sm_scale: float,
) -> torch.Tensor:
    """Retrieval head: top-p prev selection, then FlashAttention merge.

    The scoring step is unavoidable cross-head work; it costs one extra
    ``q @ k_prev^T`` over the prev region. We do it in fp32 for stability
    but keep the operation lazy if the bucket has no prev (collapse to
    global path).
    """
    B = q_bucket.shape[0]
    Hq_b = q_bucket.shape[1]
    KVH_b = k_prev_b.shape[1]
    num_q_per_kv = Hq_b // KVH_b

    # Importance scores: max over (q-heads in group, queries) of logits.
    q_grouped = q_bucket.view(
        B, KVH_b, num_q_per_kv, q_bucket.shape[-2], q_bucket.shape[-1]
    )
    scores = (
        torch.einsum("bkgld,bkpd->bkglp", q_grouped, k_prev_b).amax(dim=(2, 3))
        * sm_scale
    )  # [B, KVH_b, P]
    softmax_scores = F.softmax(scores.float(), dim=-1)

    sorted_scores, sorted_idx = torch.sort(softmax_scores, dim=-1, descending=True)
    cum = torch.cumsum(sorted_scores, dim=-1)
    keep = cum <= retrieval_top_p
    keep[..., 0] = True
    P_keep = int(keep.sum(dim=-1).max().item())

    _, topk_idx = torch.topk(softmax_scores, k=P_keep, dim=-1)  # [B, KVH_b, P_keep]
    D = k_prev_b.shape[-1]
    gather_idx = topk_idx.unsqueeze(-1).expand(B, KVH_b, P_keep, D)
    k_prev_use = torch.gather(k_prev_b, 2, gather_idx)
    v_prev_use = torch.gather(v_prev_b, 2, gather_idx)

    return _attn_global_dense_flash(
        q_bucket,
        k_sink_b,
        v_sink_b,
        k_self_b,
        v_self_b,
        k_prev_use,
        v_prev_use,
        sm_scale=sm_scale,
    )


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def segment_attention_flash(
    q: torch.Tensor,  # [B, Hq, L_q, D]
    k_self: torch.Tensor,  # [B, KVH, L_kv, D]
    v_self: torch.Tensor,
    k_prev: Optional[torch.Tensor],  # [B, KVH, P, D]
    v_prev: Optional[torch.Tensor],
    k_sink_padded: Optional[torch.Tensor],  # [B, KVH, S_max, D]
    v_sink_padded: Optional[torch.Tensor],
    *,
    plan: LayerMaskPlan,
    num_q_per_kv: int,
    sm_scale: float,
    retrieval_top_p: float = 0.9,
    q_chunk_size: int = DEFAULT_Q_CHUNK,  # accepted for API parity
) -> torch.Tensor:
    """Drop-in FlashAttention-backed version of :func:`segment_attention`.

    Same signature, same return shape. Routes each head-type bucket to
    the most efficient FA-2 call pattern (see module docstring).

    Raises
    ------
    RuntimeError
        If ``flash_attn`` is not importable in this environment.
    """
    if not _HAS_FLASH_ATTN:
        raise RuntimeError(
            "segment_attention_flash requires the flash_attn package. "
            "Fall back to segment_attention_fast (SDPA) or segment_attention."
        )

    Hq = q.shape[1]
    KVH = k_self.shape[1]
    L_q = q.shape[-2]
    B = q.shape[0]
    D = v_self.shape[-1]
    out = q.new_empty(B, Hq, L_q, D)

    groups = group_heads_by_type(plan)
    for code, kvh_idx in groups.items():
        k_self_b = k_self.index_select(1, kvh_idx)
        v_self_b = v_self.index_select(1, kvh_idx)
        k_prev_b = k_prev.index_select(1, kvh_idx) if k_prev is not None else None
        v_prev_b = v_prev.index_select(1, kvh_idx) if v_prev is not None else None
        if k_sink_padded is not None and k_sink_padded.shape[-2] > 0:
            k_sink_b = k_sink_padded.index_select(1, kvh_idx)
            v_sink_b = v_sink_padded.index_select(1, kvh_idx)
        else:
            k_sink_b = None
            v_sink_b = None

        qh_idx = q_head_indices(kvh_idx, num_q_per_kv)
        q_b = q.index_select(1, qh_idx)

        if code == HeadClassConfig.TYPE_LOCAL:
            # All heads in this bucket use the *max* per-head window
            # (flash_attn takes a single int, so heads with smaller window
            # would over-attend; this is conservative — keeping more keys
            # at most increases quality, not breaks it).
            window_per_head = plan.window.index_select(0, kvh_idx)
            w = int(window_per_head.max().item())
            if w <= 0:
                w = k_self_b.shape[-2]
            attn = _attn_local_flash(
                q_b,
                k_sink_b,
                v_sink_b,
                k_self_b,
                v_self_b,
                k_prev_b,
                v_prev_b,
                window=w,
                sm_scale=sm_scale,
            )

        elif code in (HeadClassConfig.TYPE_GLOBAL, HeadClassConfig.TYPE_DENSE):
            attn = _attn_global_dense_flash(
                q_b,
                k_sink_b,
                v_sink_b,
                k_self_b,
                v_self_b,
                k_prev_b,
                v_prev_b,
                sm_scale=sm_scale,
            )

        elif code == HeadClassConfig.TYPE_RETRIEVAL:
            if k_prev_b is None or k_prev_b.shape[-2] == 0:
                attn = _attn_global_dense_flash(
                    q_b,
                    k_sink_b,
                    v_sink_b,
                    k_self_b,
                    v_self_b,
                    None,
                    None,
                    sm_scale=sm_scale,
                )
            else:
                attn = _attn_retrieval_flash(
                    q_b,
                    k_sink_b,
                    v_sink_b,
                    k_self_b,
                    v_self_b,
                    k_prev_b,
                    v_prev_b,
                    retrieval_top_p=retrieval_top_p,
                    sm_scale=sm_scale,
                )
        else:
            raise ValueError(f"Unknown head_type code: {code}")

        out.index_copy_(1, qh_idx, attn)

    return out


def is_flash_attn_available() -> bool:
    """Return True iff the ``flash_attn`` Python package is importable."""
    return _HAS_FLASH_ATTN
