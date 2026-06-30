# Copyright 2024-2026 SGLang RedKnot Integration.
"""FlashAttention-3 kernel path for RedKnot.

This module is a *drop-in upgrade* of :mod:`ops_flash` (FA-2) that routes
each head-type bucket to **FlashAttention-3** via ``sgl_kernel.flash_attn``.
The dispatch logic, mask plan, top-p selection and LSE merge are byte-for-
byte the same as FA-2; only the low-level kernel changes.

Why FA-3?
---------
On Hopper-class GPUs (SM 9.0 — H100/H200/L20Y) FA-3 brings:

- **WGMMA + TMA**: warp-group matmul with tensor memory accelerator,
  hiding HBM latency behind compute.
- **Asynchronous softmax/matmul pipelining**.
- **Lower scheduler overhead** via persistent kernel + scheduler metadata.

Whether these translate to wall-clock wins for the RedKnot workload is
exactly what this module lets us measure: a head-to-head ``--kernel fa3``
vs ``--kernel fa2`` comparison on the same model + context.

API
---
Exports :func:`segment_attention_flash3` with the same signature as
:func:`sglang.srt.layers.attention.redknot.ops_flash.segment_attention_flash`,
so callers can swap them with a function-pointer change.

Fallback
--------
If ``sgl_kernel.flash_attn`` (or FA-3 on this GPU) is unavailable,
:func:`is_fa3_available` returns ``False`` and ``segment_attention_flash3``
raises on call.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
from sglang.srt.layers.attention.redknot.mask_plan import (
    LayerMaskPlan,
    group_heads_by_type,
    q_head_indices,
)

# FA-2 default chunk constant for API parity.
DEFAULT_Q_CHUNK = 2048

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Lazy FA-3 import & availability probe
# ──────────────────────────────────────────────────────────────────────────
try:
    from sgl_kernel.flash_attn import (
        flash_attn_varlen_func as _fa3_varlen,
        is_fa3_supported as _is_fa3_supported_hw,
    )

    _HAS_FA3 = True
except Exception as exc:  # pragma: no cover
    _fa3_varlen = None
    _is_fa3_supported_hw = None
    _HAS_FA3 = False
    logger.info("sgl_kernel FA-3 unavailable (%s); RedKnot fa3 path disabled.", exc)


def is_fa3_available() -> bool:
    """Return True iff FA-3 is both importable AND supported on the GPU."""
    if not _HAS_FA3:
        return False
    try:
        return bool(_is_fa3_supported_hw())
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────
def _to_varlen(x_bhsd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """``[B, H, S, D]`` -> packed ``[B*S, H, D]`` + ``cu_seqlens`` + max.

    FA-3 varlen API requires a "ragged" layout: tokens of all sequences
    concatenated along the first dim, with a CSR-style ``cu_seqlens``
    index telling the kernel where each sequence starts. We support
    ``B==1`` (RedKnot's usual case) and produce the trivial ``[0, S]``
    cu_seqlens; multi-batch packing is handled by expanding ``B`` into
    ``cu_seqlens = [0, S, 2S, ...]``.
    """
    B, H, S, D = x_bhsd.shape
    # Permute to [B, S, H, D] and flatten the first two dims.
    x_bshd = x_bhsd.transpose(1, 2).contiguous()
    x_packed = x_bshd.reshape(B * S, H, D)
    cu = torch.arange(0, (B + 1) * S, S, device=x_bhsd.device, dtype=torch.int32)
    return x_packed, cu, S


def _from_varlen(out_packed: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """Inverse of :func:`_to_varlen`. Returns ``[B, H, S, D]``."""
    H = out_packed.shape[-2]
    D = out_packed.shape[-1]
    return out_packed.reshape(B, S, H, D).transpose(1, 2).contiguous()


def _fa3(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
    window_size: Tuple[int, int] = (-1, -1),
    return_lse: bool = False,
):
    """Wrapper that converts ``[B, H, S, D]`` to FA-3 varlen call and back.

    Coerces dtype to bf16 when necessary (FA-3 only accepts bf16/fp16/fp8).
    """
    if not _HAS_FA3:
        raise RuntimeError("FA-3 kernel not available in this environment.")
    orig_dtype = q_bhsd.dtype
    if q_bhsd.dtype not in (torch.float16, torch.bfloat16):
        q_bhsd = q_bhsd.to(torch.bfloat16)
        k_bhsd = k_bhsd.to(torch.bfloat16)
        v_bhsd = v_bhsd.to(torch.bfloat16)

    B_q, H_q, S_q, D_q = q_bhsd.shape
    B_k, H_k, S_k, D_k = k_bhsd.shape
    assert B_q == B_k, "Mismatched batch between q and k"

    q_pack, cu_q, max_q = _to_varlen(q_bhsd)
    k_pack, cu_k, max_k = _to_varlen(k_bhsd)
    v_pack, _, _ = _to_varlen(v_bhsd)

    result = _fa3_varlen(
        q_pack,
        k_pack,
        v_pack,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=window_size,
        return_softmax_lse=return_lse,
        ver=3,
    )

    if return_lse:
        # FA-3 returns (out, lse, ?, ?) for varlen. out shape [B*S_q, H, D],
        # lse shape [H, B*S_q].
        out_packed = result[0]
        lse_packed = result[1]
        out = _from_varlen(out_packed, B_q, S_q).to(orig_dtype)
        # Reshape LSE from [H, B*S_q] to [B, H, S_q].
        lse = lse_packed.view(H_q, B_q, S_q).permute(1, 0, 2).contiguous()
        return out, lse

    out_packed = result if not isinstance(result, tuple) else result[0]
    return _from_varlen(out_packed, B_q, S_q).to(orig_dtype)


def _bh_to_bsh(x_bhsd: torch.Tensor) -> torch.Tensor:
    """Used by the merge helper (which inherits its layout from FA-2 code)."""
    return x_bhsd.transpose(1, 2).contiguous()


def _bsh_to_bh(x_bshd: torch.Tensor) -> torch.Tensor:
    return x_bshd.transpose(1, 2).contiguous()


def _merge_two_attn(
    o_a: torch.Tensor,
    lse_a: torch.Tensor,
    o_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Log-sum-exp merge of two FA-3 partial outputs.

    Inputs (same convention as FA-2 code path so we can share tests):
      o_*  : ``[B, S, H, D]``
      lse_*: ``[B, H, S]``

    Output: ``[B, S, H, D]`` — equivalent to one big softmax over the union.
    """
    lse_a_bsh1 = lse_a.transpose(1, 2).unsqueeze(-1).to(o_a.dtype)
    lse_b_bsh1 = lse_b.transpose(1, 2).unsqueeze(-1).to(o_b.dtype)
    m = torch.maximum(lse_a_bsh1, lse_b_bsh1)
    w_a = (lse_a_bsh1 - m).exp()
    w_b = (lse_b_bsh1 - m).exp()
    return (w_a * o_a + w_b * o_b) / (w_a + w_b)


# ──────────────────────────────────────────────────────────────────────────
# Per-bucket attention calls (FA-3 versions, mirroring ops_flash._attn_*)
# ──────────────────────────────────────────────────────────────────────────
def _attn_local_fa3(
    q_bucket: torch.Tensor,  # [B, Hq_b, L_q, D]
    k_sink_b: Optional[torch.Tensor],
    v_sink_b: Optional[torch.Tensor],
    k_self_b: torch.Tensor,
    v_self_b: torch.Tensor,
    k_prev_b: Optional[torch.Tensor],
    v_prev_b: Optional[torch.Tensor],
    *,
    window: int,
    sm_scale: float,
) -> torch.Tensor:
    """Local head: sink/prev (full) + self (windowed) via 2-pass FA-3 + LSE merge.

    Identical recipe to :func:`ops_flash._attn_local_flash` but kernel ==
    FA-3. The sink/prev_tail prefix is shown to every query (no window);
    the self block uses ``window_size=(W, 0)`` for block-sparse skipping.
    """
    # Crop prev to last `window` tokens.
    if k_prev_b is not None and k_prev_b.shape[-2] > 0:
        tail = min(window, k_prev_b.shape[-2])
        k_prev_use = k_prev_b[..., -tail:, :]
        v_prev_use = v_prev_b[..., -tail:, :]
    else:
        k_prev_use = None
        v_prev_use = None
        tail = 0

    prefix_parts_k: List[torch.Tensor] = []
    prefix_parts_v: List[torch.Tensor] = []
    S_max = k_sink_b.shape[-2] if k_sink_b is not None else 0
    if S_max > 0:
        prefix_parts_k.append(k_sink_b)
        prefix_parts_v.append(v_sink_b)
    if tail > 0:
        prefix_parts_k.append(k_prev_use)
        prefix_parts_v.append(v_prev_use)

    # Pass B: self-only, windowed-causal.
    o_self, lse_self = _fa3(
        q_bucket,
        k_self_b,
        v_self_b,
        causal=True,
        sm_scale=sm_scale,
        window_size=(window, 0),
        return_lse=True,
    )
    o_self_bshd = _bh_to_bsh(o_self)
    lse_self_bhs = lse_self  # already [B, H, S] from _fa3

    if not prefix_parts_k:
        return o_self  # already [B, H, S, D]

    # Pass A: sink + prev_tail, fully visible.
    K_prefix = torch.cat(prefix_parts_k, dim=-2)
    V_prefix = torch.cat(prefix_parts_v, dim=-2)
    o_prefix, lse_prefix = _fa3(
        q_bucket,
        K_prefix,
        V_prefix,
        causal=False,
        sm_scale=sm_scale,
        return_lse=True,
    )
    o_prefix_bshd = _bh_to_bsh(o_prefix)

    merged = _merge_two_attn(o_prefix_bshd, lse_prefix, o_self_bshd, lse_self_bhs)
    return _bsh_to_bh(merged)


def _attn_global_dense_fa3(
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
    """Global / dense head via two-pass FA-3 + LSE merge.

    Pass A: ``[sink | prev]`` non-causal (everything visible).
    Pass B: ``[self]`` causal.
    Merge via log-sum-exp.
    """
    parts_k: List[torch.Tensor] = []
    parts_v: List[torch.Tensor] = []
    S_max = k_sink_b.shape[-2] if k_sink_b is not None else 0
    if S_max > 0:
        parts_k.append(k_sink_b)
        parts_v.append(v_sink_b)
    if k_prev_b is not None and k_prev_b.shape[-2] > 0:
        parts_k.append(k_prev_b)
        parts_v.append(v_prev_b)

    if not parts_k:
        # Self-only path.
        return _fa3(q_bucket, k_self_b, v_self_b, causal=True, sm_scale=sm_scale)

    K_prev_full = torch.cat(parts_k, dim=-2)
    V_prev_full = torch.cat(parts_v, dim=-2)

    o_a, lse_a = _fa3(
        q_bucket,
        K_prev_full,
        V_prev_full,
        causal=False,
        sm_scale=sm_scale,
        return_lse=True,
    )
    o_b, lse_b = _fa3(
        q_bucket,
        k_self_b,
        v_self_b,
        causal=True,
        sm_scale=sm_scale,
        return_lse=True,
    )
    merged = _merge_two_attn(_bh_to_bsh(o_a), lse_a, _bh_to_bsh(o_b), lse_b)
    return _bsh_to_bh(merged)


def _attn_retrieval_fa3(
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
    """Retrieval head: top-p selection (PyTorch) -> FA-3 global/dense path."""
    B = q_bucket.shape[0]
    Hq_b = q_bucket.shape[1]
    KVH_b = k_prev_b.shape[1]
    num_q_per_kv = Hq_b // KVH_b

    q_grouped = q_bucket.view(
        B, KVH_b, num_q_per_kv, q_bucket.shape[-2], q_bucket.shape[-1]
    )
    scores = (
        torch.einsum("bkgld,bkpd->bkglp", q_grouped, k_prev_b).amax(dim=(2, 3))
        * sm_scale
    )
    softmax_scores = F.softmax(scores.float(), dim=-1)

    sorted_scores, sorted_idx = torch.sort(softmax_scores, dim=-1, descending=True)
    cum = torch.cumsum(sorted_scores, dim=-1)
    keep = cum <= retrieval_top_p
    keep[..., 0] = True
    P_keep = int(keep.sum(dim=-1).max().item())

    _, topk_idx = torch.topk(softmax_scores, k=P_keep, dim=-1)
    D = k_prev_b.shape[-1]
    gather_idx = topk_idx.unsqueeze(-1).expand(B, KVH_b, P_keep, D)
    k_prev_use = torch.gather(k_prev_b, 2, gather_idx)
    v_prev_use = torch.gather(v_prev_b, 2, gather_idx)

    return _attn_global_dense_fa3(
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
def segment_attention_flash3(
    q: torch.Tensor,
    k_self: torch.Tensor,
    v_self: torch.Tensor,
    k_prev: Optional[torch.Tensor],
    v_prev: Optional[torch.Tensor],
    k_sink_padded: Optional[torch.Tensor],
    v_sink_padded: Optional[torch.Tensor],
    *,
    plan: LayerMaskPlan,
    num_q_per_kv: int,
    sm_scale: float,
    retrieval_top_p: float = 0.9,
    q_chunk_size: int = DEFAULT_Q_CHUNK,
) -> torch.Tensor:
    """FA-3 powered drop-in replacement for ``segment_attention_flash``.

    Same signature, same return shape ``[B, Hq, L_q, D]``, same head-type
    bucket dispatch. Routes each (layer, kv_head) group to the FA-3
    variant of local / global / retrieval. Raises ``RuntimeError`` if
    FA-3 isn't usable on this GPU.
    """
    if not is_fa3_available():
        raise RuntimeError(
            "segment_attention_flash3 requires FA-3 (sgl_kernel.flash_attn "
            "with is_fa3_supported() == True). Use segment_attention_flash "
            "for the FA-2 path."
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
            window_per_head = plan.window.index_select(0, kvh_idx)
            w = int(window_per_head.max().item())
            if w <= 0:
                w = k_self_b.shape[-2]
            attn = _attn_local_fa3(
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
            attn = _attn_global_dense_fa3(
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
                attn = _attn_global_dense_fa3(
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
                attn = _attn_retrieval_fa3(
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
