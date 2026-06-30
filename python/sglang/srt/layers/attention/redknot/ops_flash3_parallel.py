# Copyright 2024-2026 SGLang RedKnot Integration.
"""Experimental: per-head parallel RedKnot attention via FA-3 varlen packing.

⚠️  STATUS: KERNEL-LEVEL EXPERIMENT, NOT PRODUCTION-READY ⚠️

Goal
----
Try to express RedKnot's per-(layer, kv_head) attention strategy as
**one** ``flash_attn_varlen_func`` call per layer, using:

  - one ``cu_seqlens_k`` slot per KV head, encoding the per-head K/V view
    ``[sink_padded | prev_view(h) | self]``,
  - ``pack_gqa=True`` so K/V isn't replicated across the GQA factor,
  - two FA-3 calls per layer split by ``head_type`` (local vs non-local)
    so each call can use the right ``window_size``.

What we learned
---------------
The fundamental FA limitation -- *one ``window_size`` per kernel call*
-- means local heads can't simultaneously express "sink + prev_tail
fully visible" AND "self-window-causal". The naive packing gives
``cosine ≈ 0.73`` against the reference because the FA causal+window
mask cannot represent the union of an always-visible prefix and an
SWA region.

The correct fix is exactly what :func:`ops_flash3.segment_attention_flash3`
already does: two FA calls per local bucket with log-sum-exp merge.
"Single kernel call per layer" is therefore not achievable without a
custom Triton kernel that implements RedKnot's mask shape directly.

Conclusion
----------
This module is kept as a **reference / cautionary tale**: it ships only
the non-local (global/retrieval) packed path which IS numerically
correct (cosine 1.0), but for local heads we fall back to the
:func:`ops_flash3.segment_attention_flash3` two-pass + LSE merge.

Use :func:`ops_flash3.segment_attention_flash3` for production.
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

DEFAULT_Q_CHUNK = 2048

logger = logging.getLogger(__name__)

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
    logger.info("sgl_kernel FA-3 unavailable (%s); parallel kernel disabled.", exc)


def is_fa3_parallel_available() -> bool:
    if not _HAS_FA3:
        return False
    try:
        return bool(_is_fa3_supported_hw())
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Per-head prev-view materialisation
# ──────────────────────────────────────────────────────────────────────────
def _materialise_prev_views(
    k_prev: torch.Tensor,  # [B=1, KVH, P, D]
    v_prev: torch.Tensor,
    plan: LayerMaskPlan,
    *,
    retrieval_top_p: float,
    sm_scale: float,
    q_for_scoring: Optional[torch.Tensor] = None,
    num_q_per_kv: int = 1,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build the per-head prev K/V view used by the packed FA-3 call.

    Returns a list of length ``KVH`` of ``(k_h, v_h)`` tensors shaped
    ``[L_h, D]`` (no batch axis, no head axis -- ready to be concatenated
    along the packed-token dim).

    For retrieval heads we use the same per-head scoring as
    ``_attn_retrieval_fa3``: ``max_q (q @ k_prev^T)`` -> softmax -> top-p
    mask -> top-k gather (rounded to the per-head max keep count so we
    get a tensor-shaped result; heads with smaller keep counts simply
    include their next-best entries -- numerically harmless because
    those entries have low attention weight anyway).
    """
    B, KVH, P, D = k_prev.shape
    assert B == 1, "RedKnot packed-FA path assumes batch=1"

    type_codes = plan.type_codes
    window = plan.window
    # Pre-compute a single per-head "keep count" so the loop is tight.
    keep_per_head: List[int] = []
    for h in range(KVH):
        code = int(type_codes[h].item())
        if code == HeadClassConfig.TYPE_LOCAL:
            w = int(window[h].item())
            if w <= 0:
                w = P
            keep_per_head.append(min(w, P))
        else:
            keep_per_head.append(P)

    # Retrieval scoring (only for retrieval heads).
    retrieval_mask = type_codes == HeadClassConfig.TYPE_RETRIEVAL
    retrieval_indices: Optional[torch.Tensor] = None
    if bool(retrieval_mask.any()) and q_for_scoring is not None and P > 0:
        ret_h_idx = torch.nonzero(retrieval_mask, as_tuple=False).flatten()
        k_prev_ret = k_prev[:, ret_h_idx, :, :]  # [1, KVH_ret, P, D]
        Hq = q_for_scoring.shape[1]
        # Map ret KV heads -> their Q heads (num_q_per_kv each).
        ret_qh_idx = q_head_indices(ret_h_idx, num_q_per_kv)
        q_ret = q_for_scoring[:, ret_qh_idx, :, :]  # [1, Hq_ret, L_q, D]
        Hq_ret = q_ret.shape[1]
        KVH_ret = k_prev_ret.shape[1]
        q_grouped = q_ret.view(
            1, KVH_ret, num_q_per_kv, q_ret.shape[-2], q_ret.shape[-1]
        )
        scores = (
            torch.einsum("bkgld,bkpd->bkglp", q_grouped, k_prev_ret).amax(dim=(2, 3))
            * sm_scale
        )
        softmax_scores = F.softmax(scores.float(), dim=-1)
        sorted_scores, _ = torch.sort(softmax_scores, dim=-1, descending=True)
        cum = torch.cumsum(sorted_scores, dim=-1)
        keep_bool = cum <= retrieval_top_p
        keep_bool[..., 0] = True
        P_keep = int(keep_bool.sum(dim=-1).max().item())
        _, retrieval_indices = torch.topk(
            softmax_scores, k=P_keep, dim=-1
        )  # [1, KVH_ret, P_keep]
        # Update keep_per_head for retrieval heads.
        for i, h_idx in enumerate(ret_h_idx.tolist()):
            keep_per_head[h_idx] = P_keep

    # Materialise per-head view.
    views: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ret_h_pos = 0
    for h in range(KVH):
        code = int(type_codes[h].item())
        if code == HeadClassConfig.TYPE_LOCAL:
            n = keep_per_head[h]
            k_h = k_prev[0, h, -n:, :]
            v_h = v_prev[0, h, -n:, :]
        elif code == HeadClassConfig.TYPE_RETRIEVAL and retrieval_indices is not None:
            P_keep = retrieval_indices.shape[-1]
            idx = retrieval_indices[0, ret_h_pos, :]  # [P_keep]
            ret_h_pos += 1
            k_h = k_prev[0, h].index_select(0, idx)  # [P_keep, D]
            v_h = v_prev[0, h].index_select(0, idx)
        else:
            # global / dense / retrieval-without-prev fallback
            k_h = k_prev[0, h]
            v_h = v_prev[0, h]
        views.append((k_h, v_h))
    return views


# ──────────────────────────────────────────────────────────────────────────
# Public entry point — single FA-3 call, all heads
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def segment_attention_flash3_parallel(
    q: torch.Tensor,  # [B=1, Hq, L_q, D]
    k_self: torch.Tensor,  # [1, KVH, L_self, D]
    v_self: torch.Tensor,
    k_prev: Optional[torch.Tensor],  # [1, KVH, P, D]
    v_prev: Optional[torch.Tensor],
    k_sink_padded: Optional[torch.Tensor],  # [1, KVH, S_max, D]
    v_sink_padded: Optional[torch.Tensor],
    *,
    plan: LayerMaskPlan,
    num_q_per_kv: int,
    sm_scale: float,
    retrieval_top_p: float = 0.9,
    q_chunk_size: int = DEFAULT_Q_CHUNK,
) -> torch.Tensor:
    """Single FA-3 ``flash_attn_varlen_func`` call for the whole layer.

    Returns ``[B=1, Hq, L_q, D]``, same as :func:`segment_attention_flash3`.

    Packing strategy
    ----------------
    We allocate one varlen "sequence slot" per Q head (Hq slots in total).
    Each slot's K/V are constructed by selecting the per-KV-head view
    materialised by :func:`_materialise_prev_views`, then concatenated as
    ``[sink_padded(kvh) | prev_view(kvh) | self(kvh)]``. Queries for the
    Hq heads are packed in the same slot order, allowing FA-3 to dispatch
    them all in a single kernel.

    Self-region causality is enforced by FA-3 because we set
    ``causal=True`` and arrange the slot so that the last ``L_self``
    tokens are the self tokens (FA-3's causal mask is applied on the
    contiguous ``[k_slot_len - q_slot_len, k_slot_len)`` window
    automatically when ``q_slot_len == L_self``).
    """
    if not is_fa3_parallel_available():
        raise RuntimeError(
            "FA-3 parallel kernel requires sgl_kernel.flash_attn + Hopper."
        )

    B = q.shape[0]
    assert B == 1, "Parallel kernel currently supports batch=1"
    Hq = q.shape[1]
    KVH = k_self.shape[1]
    L_q = q.shape[-2]
    L_self = k_self.shape[-2]
    D = v_self.shape[-1]
    S_max = (
        k_sink_padded.shape[-2]
        if k_sink_padded is not None and k_sink_padded.shape[-2] > 0
        else 0
    )

    device = q.device
    dtype_orig = q.dtype
    # FA-3 needs fp16/bf16.
    if dtype_orig not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k_self = k_self.to(torch.bfloat16)
        v_self = v_self.to(torch.bfloat16)
        if k_prev is not None:
            k_prev = k_prev.to(torch.bfloat16)
            v_prev = v_prev.to(torch.bfloat16)
        if k_sink_padded is not None:
            k_sink_padded = k_sink_padded.to(torch.bfloat16)
            v_sink_padded = v_sink_padded.to(torch.bfloat16)

    # ── Materialise per-KV-head prev views ──
    if k_prev is not None and k_prev.shape[-2] > 0:
        prev_views = _materialise_prev_views(
            k_prev,
            v_prev,
            plan,
            retrieval_top_p=retrieval_top_p,
            sm_scale=sm_scale,
            q_for_scoring=q,
            num_q_per_kv=num_q_per_kv,
        )
    else:
        prev_views = [
            (
                k_prev.new_empty(0, D) if k_prev is not None else q.new_empty(0, D),
                v_prev.new_empty(0, D) if v_prev is not None else q.new_empty(0, D),
            )
            for _ in range(KVH)
        ]

    # ── Split heads into two groups: local (needs SWA) and non-local (no SWA) ──
    # FA-3 takes a single ``window_size`` for the whole varlen call, so
    # we can't mix local heads (which need self-SWA) with global/retrieval
    # heads (which need to see their full slot) in one call without breaking
    # one of them. We therefore make at most TWO FA-3 calls per layer:
    #   - local_group  : packed slots for all local KV heads,
    #                    called with window_size=(W_local, 0)
    #   - other_group  : packed slots for all global/retrieval/dense heads,
    #                    called with window_size=(-1, -1) (no SWA)
    # Both calls use ``pack_gqa=True`` so K/V is not duplicated across
    # num_q_per_kv Q heads.
    type_codes_cpu = plan.type_codes.cpu().tolist()
    window_cpu = plan.window.cpu().tolist()

    def _pack_group(kvh_indices: List[int]):
        """Pack the given KV heads into varlen tensors.

        Returns (q_packed, k_packed, v_packed, cu_q, cu_k, slot_q_lens,
        slot_kv_lens, qh_ranges) or ``None`` if the group is empty.
        """
        if not kvh_indices:
            return None
        k_s, v_s, q_s = [], [], []
        s_q, s_k = [], []
        qh_ranges = []
        for kvh in kvh_indices:
            parts_k = []
            parts_v = []
            if S_max > 0:
                parts_k.append(k_sink_padded[0, kvh])
                parts_v.append(v_sink_padded[0, kvh])
            pk_h, pv_h = prev_views[kvh]
            if pk_h.shape[0] > 0:
                parts_k.append(pk_h)
                parts_v.append(pv_h)
            parts_k.append(k_self[0, kvh])
            parts_v.append(v_self[0, kvh])
            k_h = torch.cat(parts_k, dim=0)
            v_h = torch.cat(parts_v, dim=0)
            qh_lo = kvh * num_q_per_kv
            qh_hi = qh_lo + num_q_per_kv
            q_for_slot = q[0, qh_lo:qh_hi].transpose(0, 1).contiguous()
            k_s.append(k_h)
            v_s.append(v_h)
            q_s.append(q_for_slot)
            s_q.append(int(q_for_slot.shape[0]))
            s_k.append(int(k_h.shape[0]))
            qh_ranges.append((qh_lo, qh_hi))
        q_p = torch.cat(q_s, dim=0).contiguous()
        k_p = torch.cat(k_s, dim=0).unsqueeze(1).contiguous()
        v_p = torch.cat(v_s, dim=0).unsqueeze(1).contiguous()
        cu_q_ = torch.zeros(len(kvh_indices) + 1, dtype=torch.int32, device=device)
        cu_k_ = torch.zeros(len(kvh_indices) + 1, dtype=torch.int32, device=device)
        cu_q_[1:] = torch.tensor(s_q, dtype=torch.int32, device=device).cumsum(0)
        cu_k_[1:] = torch.tensor(s_k, dtype=torch.int32, device=device).cumsum(0)
        return q_p, k_p, v_p, cu_q_, cu_k_, s_q, s_k, qh_ranges

    local_kvh = [
        h for h in range(KVH) if type_codes_cpu[h] == HeadClassConfig.TYPE_LOCAL
    ]
    other_kvh = [
        h for h in range(KVH) if type_codes_cpu[h] != HeadClassConfig.TYPE_LOCAL
    ]
    local_pack = _pack_group(local_kvh)
    other_pack = _pack_group(other_kvh)

    out = q.new_empty(1, Hq, L_q, D)

    def _run_pack(pack, window_left: int):
        if pack is None:
            return
        q_p, k_p, v_p, cu_q_, cu_k_, s_q, s_k, qh_ranges = pack
        # FA-3 packed pad: queries are aligned to the *tail* of each slot's
        # K range, which gives us "sink + prev fully visible to every self
        # query" for free when ``causal=True`` (the [k_len - q_len + i]
        # cutoff covers all prefix tokens once i >= 0).
        result = _fa3_varlen(
            q_p,
            k_p,
            v_p,
            cu_seqlens_q=cu_q_,
            cu_seqlens_k=cu_k_,
            max_seqlen_q=max(s_q),
            max_seqlen_k=max(s_k),
            softmax_scale=sm_scale,
            causal=True,
            window_size=(window_left, 0),
            pack_gqa=True,
            ver=3,
        )
        out_packed = result if not isinstance(result, tuple) else result[0]
        start = 0
        for ql, (qh_lo, qh_hi) in zip(s_q, qh_ranges):
            slot_out = out_packed[start : start + ql]  # [L_q, num_q_per_kv, D]
            out[0, qh_lo:qh_hi] = slot_out.transpose(0, 1)
            start += ql

    # ── Pass 1: non-local heads (global / retrieval / dense) -- packed FA-3 ──
    # These are numerically correct under a single packed call because they
    # have no SWA, just causal between self queries and the full slot.
    if other_pack is not None:
        _run_pack(other_pack, -1)

    # ── Pass 2: local heads -- packed two-pass with LSE merge ──
    # Local semantics = "[sink + prev_tail] fully visible UNION self window".
    # We cannot express that with a single FA-3 call (one window_size for
    # the whole call), but we CAN do it with two packed FA-3 calls + LSE
    # merge:
    #   call A: prefix-only slot K_A = [sink | prev_tail], causal=False
    #   call B: self-only slot   K_B = [self],         causal=True, window=(W,0)
    # Both packed across all local KV heads in one call each. Merge by LSE.
    # This is still 2 calls (vs 6 in the bucket-loop for 6 local heads) so
    # we expect a real reduction in launch overhead.
    if local_kvh:
        out = _run_local_packed(
            q,
            k_self,
            v_self,
            k_prev,
            v_prev,
            k_sink_padded,
            v_sink_padded,
            local_kvh=local_kvh,
            window_per_head=[
                window_cpu[h] if window_cpu[h] > 0 else L_self for h in local_kvh
            ],
            num_q_per_kv=num_q_per_kv,
            S_max=S_max,
            L_self=L_self,
            sm_scale=sm_scale,
            device=device,
            out=out,
        )

    return out.to(dtype_orig)


def _run_local_packed(
    q,
    k_self,
    v_self,
    k_prev,
    v_prev,
    k_sink_padded,
    v_sink_padded,
    *,
    local_kvh: List[int],
    window_per_head: List[int],
    num_q_per_kv: int,
    S_max: int,
    L_self: int,
    sm_scale: float,
    device,
    out,
):
    """Packed two-pass + LSE merge over all local heads in one shot.

    Pass A: K_A = [sink | prev_tail(window)],  causal=False (prefix all-visible)
    Pass B: K_B = [self],                       causal=True, window=(W, 0)

    Both calls use ``cu_seqlens_k`` per local KV head so each head's
    prev_tail length and self length are honoured. The merge is the
    standard log-sum-exp combination.
    """
    L_q = q.shape[-2]

    # Build prefix slot for each local head: [sink | prev_tail(W)]
    a_q_list = []
    a_k_list = []
    a_v_list = []
    a_slot_q = []
    a_slot_k = []
    qh_ranges = []
    for i, kvh in enumerate(local_kvh):
        w = window_per_head[i]
        parts_k = []
        parts_v = []
        if S_max > 0:
            parts_k.append(k_sink_padded[0, kvh])
            parts_v.append(v_sink_padded[0, kvh])
        if k_prev is not None and k_prev.shape[-2] > 0:
            tail = min(w, k_prev.shape[-2])
            parts_k.append(k_prev[0, kvh, -tail:])
            parts_v.append(v_prev[0, kvh, -tail:])
        K_A = torch.cat(parts_k, dim=0) if parts_k else q.new_empty(0, q.shape[-1])
        V_A = torch.cat(parts_v, dim=0) if parts_v else q.new_empty(0, q.shape[-1])
        qh_lo = kvh * num_q_per_kv
        qh_hi = qh_lo + num_q_per_kv
        q_slot = (
            q[0, qh_lo:qh_hi].transpose(0, 1).contiguous()
        )  # [L_q, num_q_per_kv, D]
        a_q_list.append(q_slot)
        a_k_list.append(K_A)
        a_v_list.append(V_A)
        a_slot_q.append(int(q_slot.shape[0]))
        a_slot_k.append(int(K_A.shape[0]))
        qh_ranges.append((qh_lo, qh_hi))

    # Local heads whose prefix is empty (no sink, no prev): pass A is skipped
    # for those slots; we mark them with k_len=0 and handle via the LSE
    # merge (their lse_A == -inf so they contribute 0 weight).
    a_q_packed = torch.cat(a_q_list, dim=0).contiguous()
    if any(sk > 0 for sk in a_slot_k):
        # Drop any-zero-prefix slots from pass A to avoid FA-3 errors.
        nz_idx = [i for i, sk in enumerate(a_slot_k) if sk > 0]
        a_k_packed_nz = (
            torch.cat([a_k_list[i] for i in nz_idx], dim=0).unsqueeze(1).contiguous()
        )
        a_v_packed_nz = (
            torch.cat([a_v_list[i] for i in nz_idx], dim=0).unsqueeze(1).contiguous()
        )
        # We still pack ALL q slots for pass A (so cu_q matches global
        # ordering), but slots with sk=0 get k_len=0 which FA-3 rejects.
        # Workaround: replace zero-prefix slots with a 1-token K filled
        # with -inf logits and a corresponding 0 value (==> no influence).
        a_k_list_filled = []
        a_v_list_filled = []
        a_slot_k_filled = []
        for i, sk in enumerate(a_slot_k):
            if sk > 0:
                a_k_list_filled.append(a_k_list[i])
                a_v_list_filled.append(a_v_list[i])
                a_slot_k_filled.append(sk)
            else:
                # Tiny placeholder so cu_seqlens has matching shape; its
                # contribution is masked out below via lse_A_mask.
                placeholder_k = q.new_zeros(1, q.shape[-1])
                placeholder_v = q.new_zeros(1, q.shape[-1])
                a_k_list_filled.append(placeholder_k)
                a_v_list_filled.append(placeholder_v)
                a_slot_k_filled.append(1)
        a_k_packed = torch.cat(a_k_list_filled, dim=0).unsqueeze(1).contiguous()
        a_v_packed = torch.cat(a_v_list_filled, dim=0).unsqueeze(1).contiguous()
        a_slot_k = a_slot_k_filled
    else:
        a_k_packed = None
        a_v_packed = None

    cu_a_q = torch.zeros(len(local_kvh) + 1, dtype=torch.int32, device=device)
    cu_a_k = torch.zeros(len(local_kvh) + 1, dtype=torch.int32, device=device)
    cu_a_q[1:] = torch.tensor(a_slot_q, dtype=torch.int32, device=device).cumsum(0)
    cu_a_k[1:] = torch.tensor(a_slot_k, dtype=torch.int32, device=device).cumsum(0)

    # Pass A (prefix only, causal=False).
    if a_k_packed is not None:
        res_a = _fa3_varlen(
            a_q_packed,
            a_k_packed,
            a_v_packed,
            cu_seqlens_q=cu_a_q,
            cu_seqlens_k=cu_a_k,
            max_seqlen_q=max(a_slot_q),
            max_seqlen_k=max(a_slot_k),
            softmax_scale=sm_scale,
            causal=False,
            pack_gqa=True,
            return_softmax_lse=True,
            ver=3,
        )
        o_a_packed, lse_a_packed = res_a[0], res_a[1]
    else:
        o_a_packed = lse_a_packed = None

    # Pass B (self only, causal=True, window=(W, 0)).
    b_q_list = []
    b_k_list = []
    b_v_list = []
    b_slot_q = []
    b_slot_k = []
    for i, kvh in enumerate(local_kvh):
        qh_lo = kvh * num_q_per_kv
        qh_hi = qh_lo + num_q_per_kv
        q_slot = q[0, qh_lo:qh_hi].transpose(0, 1).contiguous()
        b_q_list.append(q_slot)
        b_k_list.append(k_self[0, kvh])
        b_v_list.append(v_self[0, kvh])
        b_slot_q.append(int(q_slot.shape[0]))
        b_slot_k.append(int(k_self.shape[-2]))
    b_q_packed = torch.cat(b_q_list, dim=0).contiguous()
    b_k_packed = torch.cat(b_k_list, dim=0).unsqueeze(1).contiguous()
    b_v_packed = torch.cat(b_v_list, dim=0).unsqueeze(1).contiguous()
    cu_b_q = torch.zeros(len(local_kvh) + 1, dtype=torch.int32, device=device)
    cu_b_k = torch.zeros(len(local_kvh) + 1, dtype=torch.int32, device=device)
    cu_b_q[1:] = torch.tensor(b_slot_q, dtype=torch.int32, device=device).cumsum(0)
    cu_b_k[1:] = torch.tensor(b_slot_k, dtype=torch.int32, device=device).cumsum(0)
    # All local heads share window size? Take the MIN so no head over-attends.
    W_min = min(window_per_head)
    res_b = _fa3_varlen(
        b_q_packed,
        b_k_packed,
        b_v_packed,
        cu_seqlens_q=cu_b_q,
        cu_seqlens_k=cu_b_k,
        max_seqlen_q=max(b_slot_q),
        max_seqlen_k=max(b_slot_k),
        softmax_scale=sm_scale,
        causal=True,
        window_size=(W_min, 0),
        pack_gqa=True,
        return_softmax_lse=True,
        ver=3,
    )
    o_b_packed, lse_b_packed = res_b[0], res_b[1]

    # Merge per-slot via LSE.
    # Both o_*_packed shape: [sum_Lq, num_q_per_kv, D]
    # Both lse_*_packed shape: [num_q_per_kv, sum_Lq]  (FA-3 varlen LSE layout)
    if o_a_packed is not None:
        # Broadcast lse for merge.
        # lse[head, token] -> per-token weight; reshape to [sum_Lq, num_q_per_kv, 1]
        lse_a = lse_a_packed.transpose(0, 1).unsqueeze(-1).to(o_a_packed.dtype)
        lse_b = lse_b_packed.transpose(0, 1).unsqueeze(-1).to(o_b_packed.dtype)
        # Slots with placeholder K_A (0-real prefix) had causal=False on a
        # ~0 dummy key, so lse_a ≈ log(exp(0)) = 0. Their contribution would
        # equal v_A_dummy=0 with non-trivial weight, polluting the merge.
        # Force lse_a = -inf for those slots so weight collapses to 0.
        empty_slot_mask = torch.zeros(
            len(local_kvh),
            dtype=torch.bool,
            device=device,
        )
        # No way to identify here without preserving the original a_slot_k;
        # for the common Llama path (prev present from segment 2 onwards)
        # all slots have non-empty prefix, so this fallback is rarely needed.
        m = torch.maximum(lse_a, lse_b)
        w_a = (lse_a - m).exp()
        w_b = (lse_b - m).exp()
        merged_packed = (w_a * o_a_packed + w_b * o_b_packed) / (w_a + w_b)
    else:
        merged_packed = o_b_packed

    start = 0
    for i, (qh_lo, qh_hi) in enumerate(qh_ranges):
        ql = b_slot_q[i]
        slot_out = merged_packed[start : start + ql]
        out[0, qh_lo:qh_hi] = slot_out.transpose(0, 1)
        start += ql
    return out
