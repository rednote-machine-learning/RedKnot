# Copyright 2024-2026 SGLang RedKnot Integration.
"""Batched online prefill for RedKnot.

The standalone driver in :mod:`driver` does ``N-1`` serial
``model.forward`` calls — one per online segment — which makes the
non-attention work (QKV projections, FFN, LayerNorm, RoPE) repeat 7x
sequentially on a 64k / 8-segment Llama-70B workload. The profile in
``tests_redknot/profile_redknot_breakdown.py`` showed that this
non-attention work dominates TTFT (~68% of segment time).

This module batches **all online segments into a single forward pass**:

  input_ids:    [N-1, max_seg_len]   (one batch slot per online segment)
  position_ids: [N-1, max_seg_len]   (true positions, per segment)

The transformer's QKV projection / FFN / LayerNorm now run as **one
large batched GEMM** instead of N-1 sequential ones — this is the
source of the speedup. RoPE is applied per batch slot.

Numerical equivalence (precision-lossless)
------------------------------------------
The clever insight that makes lossless batching possible:

In the serial driver, segment ``i``'s attention at layer ``L`` consumes
``prev_k = k_proj(hidden_states_j_L)`` for j < i, i.e. the **same-layer**
k-projection output of all earlier segments. Those projections are
*independent* of segment i's tokens — they only depend on each segment's
own hidden_states at layer L.

In a single batched forward, when we reach layer L, the layer's
QKV projection runs as ONE batched GEMM over the [N-1, L_max, D]
hidden_states. The output ``k_batched`` of shape ``[N-1, KVH, L_max, D]``
already contains the same-layer K of every segment, including the ones
the current batch slot would have looked at as ``prev_kv``. So we can
read them directly from the same tensor:

  For batch slot b (corresponding to segment idx d_idx):
      prev_k_b = k_batched[<b corresponds to seg j for j < d_idx>, ...]

This is byte-equivalent to the serial driver's online prev, with the
only difference being computation order.

What we batch:
  - Embedding lookup, Q/K/V projection, FFN, LayerNorm, RoPE
    -> one large GEMM each, instead of N-1 sequential.

What stays per-slot:
  - Attention itself (each slot has a different prev length, sink,
    and per-head mask plan). FA-3 varlen or per-slot calls handle this.

Limitations
-----------
- Segments must be padded to ``max_seg_len`` — wasted compute if
  lengths are very uneven. (RedKnot typically uses equal-length
  segments, so this is moot.)
- Per-slot ``prev_k`` build still loops over the batch dim in Python;
  the heavy lifting (attention itself) is per-slot but FA-3's varlen
  API lets us still batch attention internally.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.attention.redknot.driver import (
    _get_apply_rotary,
    _restore_attn_impl,
    _switch_attn_impl,
)
from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
from sglang.srt.layers.attention.redknot.mask_plan import (
    build_layer_mask_plan,
    pad_per_head_sinks,
)
from sglang.srt.layers.attention.redknot.offline_cache import OfflineSegment
from sglang.srt.layers.attention.redknot.ops_flash import (
    is_flash_attn_available,
    segment_attention_flash,
)
from sglang.srt.layers.attention.redknot.ops_flash3 import (
    is_fa3_available,
    segment_attention_flash3,
)
from sglang.srt.layers.attention.redknot.ops_flash3_parallel import (
    is_fa3_parallel_available,
    segment_attention_flash3_parallel,
)
from sglang.srt.layers.attention.redknot.rope_helper import RoPEHelper
from sglang.srt.layers.attention.redknot.segpaged import (
    POLICY_GLOBAL,
    POLICY_LOCAL,
    SegPagedKVCache,
    is_fused_varlen_available,
    local_visible_indices,
    segpaged_attention,
)
from sglang.srt.layers.attention.redknot.sparse_ffn import (
    SparseFFNSchedule,
    apply_sparse_ffn,
    token_importance_from_attn,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# torch.compile cache for decoder layers
# ──────────────────────────────────────────────────────────────────────────
# The per-layer eager dispatch of the many small ops (LayerNorm, residual
# adds, RoPE, projections) is ~50% of online-forward wall time. Compiling
# the *whole* decoder layer fuses these and removes the dispatch overhead.
# Our monkey-patched attention / Sparse-FFN introduce data-dependent control
# flow, so torch.compile inserts graph breaks there and falls back to eager
# for those sub-regions only — the surrounding static ops are still fused
# (measured ~1.58x per layer even with a graph-breaking patched attention).
# ``dynamic=True`` lets one graph serve the variable segment lengths.
_COMPILED_LAYER_CACHE: dict = {}


def _get_compiled_layer(layer_module, dynamic: bool = True):
    """Return a torch.compiled forward for a decoder layer.

    The compiled callable wraps the layer's *current* ``forward`` (after our
    attention / MLP monkey-patches are installed), so the patched sub-paths
    become graph-break regions while the rest of the layer is fused.

    ``dynamic=False`` specializes on the (fixed) flat sequence length, which
    lets Inductor generate tighter kernels and pre-plan buffers — preferred
    for the flat single-pass online forward where ``T`` is constant.
    """
    key = (id(layer_module), dynamic)
    fn = _COMPILED_LAYER_CACHE.get(key)
    if fn is None:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        fn = torch.compile(layer_module.forward, dynamic=dynamic, fullgraph=False)
        _COMPILED_LAYER_CACHE[key] = fn
    return fn


def _pick_kernel(name: str):
    n = name.lower()
    if n in ("flash", "fa2"):
        if not is_flash_attn_available():
            raise RuntimeError("fa2 kernel requested but flash_attn not importable")
        return segment_attention_flash
    if n == "fa3":
        if not is_fa3_available():
            raise RuntimeError("fa3 kernel requested but not usable here")
        return segment_attention_flash3
    if n == "fa3_parallel":
        if not is_fa3_parallel_available():
            raise RuntimeError("fa3_parallel kernel requested but not usable here")
        return segment_attention_flash3_parallel
    raise ValueError(f"Unknown kernel {name!r}")


# ──────────────────────────────────────────────────────────────────────────
# SegPagedAttention helpers for the online prefill path
# ──────────────────────────────────────────────────────────────────────────
_SEGPAGED_POLICY_MAP = {
    HeadClassConfig.TYPE_LOCAL: POLICY_LOCAL,
    HeadClassConfig.TYPE_GLOBAL: POLICY_GLOBAL,
    HeadClassConfig.TYPE_RETRIEVAL: POLICY_GLOBAL,  # retrieval keeps all KV
    HeadClassConfig.TYPE_DENSE: POLICY_GLOBAL,  # dense keeps all KV
}


def _head_policy_for_layer(
    head_cfg: HeadClassConfig,
    layer_idx: int,
) -> List[str]:
    """Return ``[Hkv]`` list of POLICY_GLOBAL / POLICY_LOCAL for one layer."""
    policies = []
    for h in range(head_cfg.num_kv_heads):
        strat = head_cfg.get_strategy(layer_idx, h)
        int_code = HeadClassConfig._TYPE_TO_INT[strat.head_type]
        policies.append(_SEGPAGED_POLICY_MAP[int_code])
    return policies


def _head_window_for_layer(
    head_cfg: HeadClassConfig,
    layer_idx: int,
) -> List[int]:
    """Return ``[Hkv]`` list of window sizes (``-1`` for non-local)."""
    return [
        head_cfg.get_strategy(layer_idx, h).window for h in range(head_cfg.num_kv_heads)
    ]


def _head_sink_for_layer(
    head_cfg: HeadClassConfig,
    layer_idx: int,
) -> List[int]:
    """Return ``[Hkv]`` list of sink sizes."""
    return [
        head_cfg.get_strategy(layer_idx, h).sink_size
        for h in range(head_cfg.num_kv_heads)
    ]


def _build_segpaged_cache_for_slot(
    *,
    layer_idx: int,
    head_cfg: HeadClassConfig,
    seg0_k: torch.Tensor,  # [1, Hkv, L0, D]
    seg0_v: torch.Tensor,
    prev_online_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
    # each is [1, Hkv, L_seg, D]
    self_k: torch.Tensor,  # [1, Hkv, Lb, D]
    self_v: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> SegPagedKVCache:
    """Build a SegPagedKVCache for one (layer, batch-slot) pair.

    Segments layout:
      segment 0  = offline (seg0_k/v), LOCAL heads get sink+window filtered
      segment 1..N-1 = previous online segments, all heads keep full KV
      segment N  = current slot's self KV, all heads keep full KV

    Returns a SegPagedKVCache ready for ``segpaged_attention()``.
    """
    Hkv = seg0_k.shape[1]
    head_dim = seg0_k.shape[3]

    cache = SegPagedKVCache(
        num_layers=1,
        num_kv_heads=Hkv,
        head_dim=head_dim,
        page_size=64,
        device=device,
        dtype=dtype,
    )

    policies = _head_policy_for_layer(head_cfg, layer_idx)
    windows = _head_window_for_layer(head_cfg, layer_idx)
    sinks = _head_sink_for_layer(head_cfg, layer_idx)

    # ── Segment 0: offline ──
    L0 = seg0_k.shape[2]
    for h in range(Hkv):
        k_h = seg0_k[0, h, :, :]  # [L0, D]
        v_h = seg0_v[0, h, :, :]
        policy = policies[h]

        if policy == POLICY_LOCAL and L0 > 0:
            sink = sinks[h]
            window = windows[h] if windows[h] > 0 else 512
            idx = local_visible_indices(L0, sink, window, device=device)
            k_h = k_h[idx]
            v_h = v_h[idx]

        cache.add_head_segment(layer=0, head=h, segment=0, policy=policy, k=k_h, v=v_h)

    # ── Segments 1..N-1: previous online ──
    for seg_idx, (pk, pv) in enumerate(prev_online_kvs, start=1):
        for h in range(Hkv):
            # Online segments are within the recent window; keep all KV
            # for every head type (local heads' window covers them).
            cache.add_head_segment(
                layer=0,
                head=h,
                segment=seg_idx,
                policy=policies[h],
                k=pk[0, h, :, :],
                v=pv[0, h, :, :],
            )

    # ── Final segment: current slot's self KV ──
    self_seg = len(prev_online_kvs) + 1
    for h in range(Hkv):
        cache.add_head_segment(
            layer=0,
            head=h,
            segment=self_seg,
            policy=policies[h],
            k=self_k[0, h, :, :],
            v=self_v[0, h, :, :],
        )

    return cache


# ──────────────────────────────────────────────────────────────────────────
# Head-class online attention: global heads full-context, local heads window
# ──────────────────────────────────────────────────────────────────────────
def _headclass_online_attention(
    q: torch.Tensor,  # [1, Hq, Lb, D]  current segment's query
    k_self: torch.Tensor,  # [1, Hkv, Lb, D]
    v_self: torch.Tensor,
    seg0_k: torch.Tensor,  # [1, Hkv, L0, D]  offline segment-0 (full)
    seg0_v: torch.Tensor,
    online_prev_k: Optional[torch.Tensor],  # [1, Hkv, L_on, D] or None
    online_prev_v: Optional[torch.Tensor],
    is_local: torch.Tensor,  # [Hkv] bool
    *,
    sink_size: int,
    window: int,
    seg_offset: int,
    num_q_per_kv: int,
    sm_scale: float,
) -> torch.Tensor:
    """Online-prefill attention with paper head-class sparsity.

    For the current segment's queries (positions ``[seg_offset, seg_offset+
    Lb)``), each KV head attends:

    - **global head**: ``[seg0_full | online_prev | self]`` with causal
      masking on the self block (full-context recovery).
    - **local head**: ``[sink | online_prev | self]`` but every query
      position only sees the last ``window`` tokens of the global sequence
      plus the ``sink`` tokens (sliding-window attention, design.tex
      eq:attn-local). Implemented via an additive banded mask.

    Both classes are evaluated as two batched GQA SDPA calls (no per-head
    Python loop, no varlen kernel). Returns ``[1, Hq, Lb, D]``.
    """
    Hq = q.shape[1]
    Hkv = k_self.shape[1]
    Lb = q.shape[2]
    D = q.shape[3]
    device = q.device
    L0 = seg0_k.shape[2]
    L_on = online_prev_k.shape[2] if online_prev_k is not None else 0
    out = q.new_zeros(1, Hq, Lb, D)

    local_kv = torch.nonzero(is_local, as_tuple=False).flatten()
    global_kv = torch.nonzero(~is_local, as_tuple=False).flatten()

    def _q_heads_for(kv_idx: torch.Tensor) -> torch.Tensor:
        base = kv_idx.unsqueeze(1) * num_q_per_kv
        off = torch.arange(num_q_per_kv, device=device).unsqueeze(0)
        return (base + off).flatten()

    # ── GLOBAL heads: [seg0_full | online_prev | self], mask-free causal ──
    # flash_attn causal right-aligns Q to the end of K (query i sees keys
    # [0, Lk-Lq+i]) — the desired full-context causal pattern — and supports
    # GQA natively (K/V keep Hkv heads, no repeat_interleave needed).
    if global_kv.numel() > 0:
        parts_k = [seg0_k[:, global_kv]]
        parts_v = [seg0_v[:, global_kv]]
        if L_on > 0:
            parts_k.append(online_prev_k[:, global_kv])
            parts_v.append(online_prev_v[:, global_kv])
        parts_k.append(k_self[:, global_kv])
        parts_v.append(v_self[:, global_kv])
        kg = torch.cat(parts_k, dim=2)  # [1, n_g_kv, L0+L_on+Lb, D]
        vg = torch.cat(parts_v, dim=2)
        qh = _q_heads_for(global_kv)  # [n_g_kv * gqa]
        g_out, _ = _flash_attn_lse(q[:, qh], kg, vg, sm_scale, causal=True, window=-1)
        out[:, qh] = g_out

    # ── LOCAL heads: sink + sliding window, mask-free FlashAttention ──
    # Visible KV for query position p is sink ∪ [p-W+1, p]. Two mask-free
    # FlashAttention passes (native sliding-window + LSE), merged by LSE:
    #   (1) recent : flash_attn(causal, window=(W-1,0)) over the trimmed
    #                last (W+Lb) tokens of [online_prev | self].
    #   (2) sink   : flash_attn(non-causal) over the first ``sink`` tokens.
    # GQA is native (K/V keep Hkv heads). No attn_mask is ever built.
    if local_kv.numel() > 0:
        s = max(0, min(sink_size, L0))
        qh = _q_heads_for(local_kv)
        q_l = q[:, qh]  # [1, n_l_kv * gqa, Lb, D]

        rec_k_parts, rec_v_parts = [], []
        if L_on > 0:
            rec_k_parts.append(online_prev_k[:, local_kv])
            rec_v_parts.append(online_prev_v[:, local_kv])
        rec_k_parts.append(k_self[:, local_kv])
        rec_v_parts.append(v_self[:, local_kv])
        rec_k = torch.cat(rec_k_parts, dim=2)  # [1, n_l_kv, L_on+Lb, D]
        rec_v = torch.cat(rec_v_parts, dim=2)
        keep = min(rec_k.shape[2], window + Lb)
        rec_k = rec_k[:, :, -keep:, :]
        rec_v = rec_v[:, :, -keep:, :]

        rec_out, rec_lse = _flash_attn_lse(
            q_l, rec_k, rec_v, sm_scale, causal=True, window=window
        )
        if s > 0:
            sink_k = seg0_k[:, local_kv, :s, :]
            sink_v = seg0_v[:, local_kv, :s, :]
            sink_out, sink_lse = _flash_attn_lse(
                q_l, sink_k, sink_v, sm_scale, causal=False, window=-1
            )
            out[:, qh] = _merge_lse(rec_out, rec_lse, sink_out, sink_lse)
        else:
            out[:, qh] = rec_out

    return out


def _flash_attn_lse(
    q: torch.Tensor,  # [1, Hq_sub, Lq, D]
    k: torch.Tensor,  # [1, Hkv_sub, Lk, D]  (GQA: Hkv_sub <= Hq_sub)
    v: torch.Tensor,
    sm_scale: float,
    *,
    causal: bool,
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask-free FlashAttention returning (output, log-sum-exp).

    Uses ``flash_attn_func`` with native sliding-window AND native GQA
    (grouped query attention) support, so no attention mask is built and
    K/V are never expanded to the full query-head count. Returns output in
    ``[1, Hq_sub, Lq, D]`` and LSE in ``[1, Hq_sub, Lq]``.
    """
    from flash_attn import flash_attn_func as _faf

    Hq = q.shape[1]
    Lq = q.shape[2]
    # flash_attn expects [B, L, H, D]; query aligned to end of K when causal.
    q_f = q.transpose(1, 2)  # [1, Lq, Hq, D]
    k_f = k.transpose(1, 2)  # [1, Lk, Hkv, D]
    v_f = v.transpose(1, 2)
    ws = (window - 1, 0) if (causal and window > 0) else (-1, -1)
    dtype_in = q_f.dtype
    if dtype_in not in (torch.float16, torch.bfloat16):
        q_f = q_f.to(torch.bfloat16)
        k_f = k_f.to(torch.bfloat16)
        v_f = v_f.to(torch.bfloat16)
    out, lse, _ = _faf(
        q_f,
        k_f,
        v_f,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=ws,
        return_attn_probs=True,
    )
    out = out.to(dtype_in).transpose(1, 2)  # [1, Hq, Lq, D]
    if lse.dim() == 3 and lse.shape[1] == Hq:
        lse_out = lse
    else:
        lse_out = lse.reshape(1, Hq, Lq)
    return out, lse_out


def _merge_lse(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Combine two partial softmax attentions via their log-sum-exps.

    The blend weight ``wa = sigmoid(lse_a - lse_b)`` is computed in fp32 but
    is only a per-(head, query) scalar broadcast over the head dim, so the
    large output tensors stay in their native dtype (no fp32 upcast of the
    full ``[1, H, Lq, D]`` activations — that upcast dominated the merge
    cost at long context).
    """
    # wa = exp(la)/(exp(la)+exp(lb)) = sigmoid(la - lb)  (numerically stable)
    wa = torch.sigmoid((lse_a - lse_b).float()).to(out_a.dtype).unsqueeze(-1)
    return out_a * wa + out_b * (1 - wa)


# ──────────────────────────────────────────────────────────────────────────
# SegPaged causal attention: bypasses kernel_fn's per-head filtering
# ──────────────────────────────────────────────────────────────────────────
def _segpaged_causal_attention(
    q: torch.Tensor,  # [1, Hq, L_q, D]
    k_self: torch.Tensor,  # [1, Hkv, L_self, D]
    v_self: torch.Tensor,  # [1, Hkv, L_self, D]
    seg0_k: torch.Tensor,  # [Hkv, L0_max, D]  (padded filtered seg0)
    seg0_v: torch.Tensor,  # [Hkv, L0_max, D]
    seg0_lens: torch.Tensor,  # [Hkv] int32  valid lengths per head
    online_prev_k: Optional[torch.Tensor],  # [Hkv, L_online, D] or None
    online_prev_v: Optional[torch.Tensor],  # [Hkv, L_online, D] or None
    *,
    num_q_per_kv: int,
    sm_scale: float,
) -> torch.Tensor:
    """SegPaged-aware causal attention with 3-segment KV input.

    Per KV head h, the slot is ``[seg0_valid_h | online_h | self_h]``.
    Pads all heads to max_slot_len and runs a single FA-3 varlen call
    with ``causal=True``.

    Parameters
    ----------
    seg0_k/v : [Hkv, L0_max, D]
        Filtered offline seg0 KV. LOCAL heads have fewer valid tokens
        (sink+window), padded with zeros to L0_max.
    seg0_lens : [Hkv] int32
        Valid length per head in seg0_k/v.
    online_prev_k/v : [Hkv, L_online, D] or None
        Online prev segments concatenated. All heads share the same length.
    """
    Hq = q.shape[1]
    Hkv = k_self.shape[1]
    L_q = q.shape[2]
    L_self = k_self.shape[2]
    D = q.shape[3]
    device = q.device
    L_online = online_prev_k.shape[1] if online_prev_k is not None else 0
    L0_max = seg0_k.shape[1] if seg0_k.shape[1] > 0 else 0

    use_fused = is_fused_varlen_available() and q.is_cuda
    if not use_fused:
        # ── Reference path (CPU / non-Hopper) ──
        prev_lens = seg0_lens.cpu()
        out = q.new_zeros(1, Hq, L_q, D)
        for kvh in range(Hkv):
            s0_len = int(prev_lens[kvh].item())
            parts_k, parts_v = [], []
            if s0_len > 0:
                parts_k.append(seg0_k[kvh, :s0_len])
                parts_v.append(seg0_v[kvh, :s0_len])
            if L_online > 0:
                parts_k.append(online_prev_k[kvh])
                parts_v.append(online_prev_v[kvh])
            ks, vs = k_self[0, kvh], v_self[0, kvh]
            if parts_k:
                k_full = torch.cat(parts_k + [ks], dim=0)
                v_full = torch.cat(parts_v + [vs], dim=0)
            else:
                k_full, v_full = ks, vs
            Lp = s0_len + L_online
            L_kv = k_full.shape[0]
            for g in range(num_q_per_kv):
                qh = kvh * num_q_per_kv + g
                q_h = q[0, qh].float()
                scores = (q_h @ k_full.float().T) * sm_scale
                mask = torch.ones(L_q, L_kv, dtype=torch.bool, device=device)
                for i in range(L_q):
                    mask[i, Lp + i + 1 :] = False
                scores = scores.masked_fill(~mask, float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[0, qh] = (probs @ v_full.float()).to(q.dtype)
        return out

    # ── Fused FA-3 varlen path ──
    from sgl_kernel.flash_attn import flash_attn_varlen_func as _fa3_varlen

    dtype_orig = q.dtype
    if dtype_orig not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k_self = k_self.to(torch.bfloat16)
        v_self = v_self.to(torch.bfloat16)
        seg0_k = seg0_k.to(torch.bfloat16)
        seg0_v = seg0_v.to(torch.bfloat16)
        if online_prev_k is not None:
            online_prev_k = online_prev_k.to(torch.bfloat16)
            online_prev_v = online_prev_v.to(torch.bfloat16)

    seg0_lens_dev = seg0_lens.to(device=device, dtype=torch.int32)
    # Per-head total KV length = seg0_valid + L_online + L_self.
    slot_k_lens = seg0_lens_dev + L_online + L_self  # [Hkv]
    max_slot_len = int(slot_k_lens.max().item())

    # ── Build padded K/V: [Hkv, max_slot_len, D] ──
    # Each head's slot layout: [seg0_valid | online | self | zero_pad].
    # The zero_pad lets FA-3 varlen handle per-head length differences.
    k_padded = q.new_zeros(Hkv, max_slot_len, D)
    v_padded = q.new_zeros(Hkv, max_slot_len, D)

    # Copy seg0: each head h gets seg0[:seg0_lens[h]] at position [0..].
    # Since seg0_k is already [Hkv, L0_max, D] with valid tokens packed
    # at the front and zeros after, we can just copy the L0_max prefix.
    if L0_max > 0:
        k_padded[:, :L0_max, :] = seg0_k
        v_padded[:, :L0_max, :] = seg0_v

    # Copy online prev: uniform L_online across heads, at offset seg0_lens[h].
    # Offsets differ per head (GLOBAL: L0, LOCAL: L_local).
    if L_online > 0:
        _arange_on = torch.arange(L_online, device=device, dtype=torch.int32)
        _on_dest = seg0_lens_dev.unsqueeze(1) + _arange_on.unsqueeze(
            0
        )  # [Hkv, L_online]
        _head_idx = torch.arange(Hkv, device=device).unsqueeze(1).expand(-1, L_online)
        k_padded[_head_idx, _on_dest.long()] = online_prev_k
        v_padded[_head_idx, _on_dest.long()] = online_prev_v

    # Copy self: uniform L_self, at offset seg0_lens[h] + L_online.
    _arange_self = torch.arange(L_self, device=device, dtype=torch.int32)
    _self_base = seg0_lens_dev + L_online  # [Hkv]
    _self_dest = _self_base.unsqueeze(1) + _arange_self.unsqueeze(0)  # [Hkv, L_self]
    _head_idx2 = torch.arange(Hkv, device=device).unsqueeze(1).expand(-1, L_self)
    k_padded[_head_idx2, _self_dest.long()] = k_self[0]  # [Hkv, L_self, D]
    v_padded[_head_idx2, _self_dest.long()] = v_self[0]

    # Flatten to packed format: [sum(slot_k_lens), 1, D].
    # Since we use padded [Hkv, max_slot_len, D] and cu_seqlens with
    # actual lengths, we need to pack (remove padding).
    # Build valid mask and gather.
    _arange_slot = torch.arange(max_slot_len, device=device)
    _valid_mask = _arange_slot.unsqueeze(0) < slot_k_lens.unsqueeze(
        1
    )  # [Hkv, max_slot_len]
    _flat_valid = _valid_mask.flatten()
    k_packed = k_padded.reshape(-1, D)[_flat_valid].unsqueeze(1).contiguous()
    v_packed = v_padded.reshape(-1, D)[_flat_valid].unsqueeze(1).contiguous()

    # ── Build packed Q: [Hkv * L_q, num_q_per_kv, D] ──
    q_reshaped = (
        q[0].view(Hkv, num_q_per_kv, L_q, D).transpose(1, 2)
    )  # [Hkv, L_q, gqa, D]
    q_packed = q_reshaped.reshape(Hkv * L_q, num_q_per_kv, D).contiguous()

    # ── cu_seqlens ──
    cu_q = torch.zeros(Hkv + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(Hkv + 1, dtype=torch.int32, device=device)
    cu_q[1:] = torch.full((Hkv,), L_q, dtype=torch.int32, device=device).cumsum(0)
    cu_k[1:] = slot_k_lens.cumsum(0)

    result = _fa3_varlen(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=L_q,
        max_seqlen_k=max_slot_len,
        softmax_scale=sm_scale,
        causal=True,
        window_size=(-1, 0),
        pack_gqa=True,
        ver=3,
    )
    out_packed = result if not isinstance(result, tuple) else result[0]

    # ── Unpack: [Hkv * L_q, num_q_per_kv, D] -> [1, Hq, L_q, D] ──
    out = out_packed.view(Hkv, L_q, num_q_per_kv, D).transpose(
        1, 2
    )  # [Hkv, gqa, L_q, D]
    out = out.reshape(1, Hq, L_q, D)

    return out.to(dtype_orig)


# ──────────────────────────────────────────────────────────────────────────
# Batched online forward
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def online_forward_segments_batched(
    model,
    *,
    segments_offline: List[OfflineSegment],
    head_cfg: HeadClassConfig,
    rope_helper: RoPEHelper,
    kernel: str = "fa3_parallel",
    micro_batch_size: Optional[int] = None,
    sparse_ffn_schedule: Optional[SparseFFNSchedule] = None,
    sparse_ffn_stats: Optional[List[dict]] = None,
    use_segpaged: bool = False,
    use_headclass: bool = False,
    use_compile: bool = False,
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Run online prefill (segments 2..N) in micro-batched forward calls.

    ``micro_batch_size`` controls how many segments share one batched
    forward. ``None`` (default) means "all online segments at once" —
    OOM-prone for very long contexts. Recommended: ``2`` for 64k Llama-70B
    on 80GB (gives ~1.5x speedup while bounding peak activation memory).

    When ``use_segpaged=True``, the attention path uses
    :class:`SegPagedKVCache` + :func:`segpaged_attention` instead of
    the dense ``kernel_fn`` path. LOCAL heads physically store only
    ``sink + window`` tokens from the offline segment, yielding the KV
    savings and mask-free execution described in the paper §4.3.

    Returns a list ``doc_online_kvs[d_idx][layer_idx] = (K, V)`` of the
    per-segment KV for segments 1..N (segment 1 is just the offline KV
    copied in). Numerically equivalent to N-1 calls of
    :func:`driver.online_forward_segment`.
    """
    if len(segments_offline) < 2:
        # Nothing to batch — return offline KV as-is.
        return [
            [
                (k.to(model.device), v.to(model.device))
                for k, v in segments_offline[0].kv
            ]
        ]

    config = model.config
    n_layers = config.num_hidden_layers
    base_model = model.model if hasattr(model, "model") else model
    device = model.device
    apply_rotary = _get_apply_rotary(config.model_type)
    kernel_fn = _pick_kernel(kernel)

    # ── Compute per-segment offsets / lengths / position_ids ──
    doc_lens = [seg.doc_len for seg in segments_offline]
    offsets, p = [], 0
    for dl in doc_lens:
        offsets.append(p)
        p += dl
    online_segs = segments_offline[1:]  # segments 2..N
    online_indices = list(range(1, len(segments_offline)))
    n_online = len(online_segs)
    online_lens = [s.doc_len for s in online_segs]
    online_offsets = [offsets[i] for i in online_indices]

    # ── Choose micro-batch size ──
    if micro_batch_size is None or micro_batch_size <= 0:
        micro_batch_size = n_online
    micro_batch_size = min(micro_batch_size, n_online)

    # Splits: list of (start, end) ranges over online_indices.
    micro_batches: List[Tuple[int, int]] = []
    for s in range(0, n_online, micro_batch_size):
        micro_batches.append((s, min(s + micro_batch_size, n_online)))

    # ── Storage for ALL captured online KV across all micro-batches ──
    # Pre-allocate to know slot-id for prev lookup.
    captured_kv_global: List[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [
        [None] * n_layers for _ in range(n_online)
    ]

    # ── Precompute per-layer head-class metadata once (hoisted out of the
    #    slot/layer hot loop to avoid 576x redundant Python work). ──
    # Only needed for the head-class path; cheap to always build.
    _hc_layer_meta: List[Optional[dict]] = [None] * n_layers
    if use_headclass:
        for _li in range(n_layers):
            _pol = _head_policy_for_layer(head_cfg, _li)
            _win_l = _head_window_for_layer(head_cfg, _li)
            _sink_l = _head_sink_for_layer(head_cfg, _li)
            _is_loc = torch.tensor(
                [pp == POLICY_LOCAL for pp in _pol],
                dtype=torch.bool,
                device=device,
            )
            if bool(_is_loc.any()):
                _fl = int(_is_loc.nonzero()[0].item())
                _sz = _sink_l[_fl]
                _wn = _win_l[_fl] if _win_l[_fl] > 0 else 512
            else:
                _sz, _wn = 4, 512
            # Hoist seg0 KV to device once per layer.
            _s0k = segments_offline[0].kv[_li][0].to(device)
            _s0v = segments_offline[0].kv[_li][1].to(device)
            _hc_layer_meta[_li] = {
                "is_local": _is_loc,
                "sink": _sz,
                "window": _wn,
                "seg0_k": _s0k,
                "seg0_v": _s0v,
            }

    # ── Run each micro-batch with a fresh patched forward. ──
    # Within a micro-batch, prev for slot b uses:
    #   - segments_offline[0].kv (segment 1, always offline)
    #   - captured_kv_global[<earlier online seg index>] from previous
    #     micro-batches (already on device, no recomputation)
    #   - batched K from THIS forward's earlier slots (b' < b within mb)
    for mb_idx, (mb_start, mb_end) in enumerate(micro_batches):
        mb_size = mb_end - mb_start
        mb_segs = online_segs[mb_start:mb_end]
        mb_indices = online_indices[mb_start:mb_end]
        mb_lens = online_lens[mb_start:mb_end]
        mb_offsets = online_offsets[mb_start:mb_end]
        mb_L_max = max(mb_lens)

        # ── Build per-microbatch input_ids + position_ids ──
        input_ids_mb = torch.zeros(mb_size, mb_L_max, dtype=torch.long, device=device)
        for b, seg in enumerate(mb_segs):
            ids = seg.token_ids.to(device)
            input_ids_mb[b, : ids.shape[0]] = ids
        position_ids_mb = torch.zeros(
            mb_size, mb_L_max, dtype=torch.long, device=device
        )
        for b, off in enumerate(mb_offsets):
            position_ids_mb[b, : mb_lens[b]] = torch.arange(
                off, off + mb_lens[b], device=device
            )

        orig_forwards: dict = {}
        orig_mlp_forwards: dict = {}
        # Per-layer captured importance for this micro-batch's Sparse FFN.
        captured_importance: List[Optional[torch.Tensor]] = [None] * n_layers

        def make_patched(
            layer_idx: int,
            mb_size=mb_size,
            mb_lens=mb_lens,
            mb_indices=mb_indices,
            mb_offsets=mb_offsets,
        ):
            attn_module = base_model.layers[layer_idx].self_attn

            def patched_forward(
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                B = input_shape[0]
                assert B == mb_size, f"micro-batch expected B={mb_size}, got {B}"
                hidden_shape = (*input_shape, -1, attn_module.head_dim)
                num_kv_groups = attn_module.num_key_value_groups

                # Batched Q/K/V projection.
                q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                if getattr(attn_module, "q_norm", None) is not None:
                    q = attn_module.q_norm(q)
                if getattr(attn_module, "k_norm", None) is not None:
                    k = attn_module.k_norm(k)
                cos, sin = position_embeddings
                if cos.device != q.device:
                    cos = cos.to(q.device)
                    sin = sin.to(q.device)
                q, k = apply_rotary(q, k, cos, sin)

                plan = build_layer_mask_plan(head_cfg, layer_idx, q.device)

                # Sink (shared, from segment 1's offline KV).
                seg0_k = segments_offline[0].kv[layer_idx][0].to(q.device)
                seg0_v = segments_offline[0].kv[layer_idx][1].to(q.device)
                k_sink_shared, v_sink_shared = pad_per_head_sinks(seg0_k, seg0_v, plan)

                # ── Pre-compute filtered seg0 for SegPaged path ──
                # Done ONCE per layer, outside the slot loop.
                # LOCAL heads: keep only sink + window tokens from seg0.
                # GLOBAL heads: keep all seg0 tokens.
                # Result: filtered_seg0_k/v [Hkv, L0_max, D] (padded)
                #         filtered_seg0_lens [Hkv] (valid lengths)
                if use_segpaged:
                    Hkv = seg0_k.shape[1]
                    L0 = seg0_k.shape[2]
                    D_head = seg0_k.shape[3]
                    _policies = _head_policy_for_layer(head_cfg, layer_idx)

                    # Compute LOCAL heads' visible indices (shared if
                    # all LOCAL heads have same sink/window — typical).
                    _windows = _head_window_for_layer(head_cfg, layer_idx)
                    _sinks = _head_sink_for_layer(head_cfg, layer_idx)
                    # Build is_local mask and find unique (sink, window) pairs.
                    is_local = torch.tensor(
                        [p == POLICY_LOCAL for p in _policies],
                        dtype=torch.bool,
                        device=q.device,
                    )
                    # Most models: all LOCAL heads share one (sink, window).
                    # Compute one idx and broadcast.
                    if bool(is_local.any()) and L0 > 0:
                        # Use first LOCAL head's params (typically all same).
                        first_local = int(is_local.nonzero(as_tuple=False)[0].item())
                        s0 = _sinks[first_local]
                        w0 = _windows[first_local] if _windows[first_local] > 0 else 512
                        local_idx = local_visible_indices(L0, s0, w0, device=q.device)
                        L_local = local_idx.shape[0]
                        # Filtered seg0 for LOCAL heads: [L_local, D]
                        # gathered once, expanded to all LOCAL heads.
                        seg0_k_local = seg0_k[0, first_local][local_idx]  # [L_local, D]
                        seg0_v_local = seg0_v[0, first_local][local_idx]
                    else:
                        L_local = 0
                        local_idx = None

                    # Build padded [Hkv, L0_max, D] where L0_max = max(L0, L_local)
                    # GLOBAL heads: full L0; LOCAL heads: L_local.
                    filtered_seg0_lens = torch.where(
                        is_local,
                        torch.tensor(L_local, device=q.device),
                        torch.tensor(L0, device=q.device),
                    ).to(torch.int32)  # [Hkv]
                    L0_max = int(filtered_seg0_lens.max().item()) if Hkv > 0 else 0
                    filtered_seg0_k = seg0_k[
                        0, :, :L0_max, :
                    ].clone()  # [Hkv, L0_max, D]
                    filtered_seg0_v = seg0_v[0, :, :L0_max, :].clone()
                    if bool(is_local.any()) and L_local > 0 and L0_max > 0:
                        # Overwrite LOCAL heads with their filtered version.
                        # Batch gather: seg0_k[0, local_heads, local_idx] → all at once.
                        local_heads_idx = is_local.nonzero(as_tuple=False).flatten()
                        # Gather all LOCAL heads' filtered tokens in one op.
                        # seg0_k[0, local_heads_idx] → [N_local, L0, D]
                        # [:, local_idx] → [N_local, L_local, D]
                        seg0_k_locals = seg0_k[0, local_heads_idx][:, local_idx]
                        seg0_v_locals = seg0_v[0, local_heads_idx][:, local_idx]
                        filtered_seg0_k[local_heads_idx, :L_local] = seg0_k_locals
                        filtered_seg0_v[local_heads_idx, :L_local] = seg0_v_locals
                        if L_local < L0_max:
                            filtered_seg0_k[local_heads_idx, L_local:] = 0
                            filtered_seg0_v[local_heads_idx, L_local:] = 0

                attn_outputs_per_slot: List[torch.Tensor] = []
                # Incremental online prev-KV buffer for the head-class path.
                # Slot b's prev-KV is the concatenation of all earlier segments'
                # self KV. Rebuilding it per slot is O(B^2) torch.cat (a top
                # wall-time cost — see profile_redknot_online.py). Instead we
                # extend a running buffer by one segment per slot: O(B) cats,
                # numerically identical (same tensors, same order).
                _run_prev_k: Optional[torch.Tensor] = None
                _run_prev_v: Optional[torch.Tensor] = None
                for b in range(B):
                    d_idx = mb_indices[b]  # global online segment index (1..N-1)
                    Lb = mb_lens[b]
                    q_b = q[b : b + 1, :, :Lb, :]
                    k_self_b = k[b : b + 1, :, :Lb, :]
                    v_self_b = v[b : b + 1, :, :Lb, :]

                    # Capture this slot's self K/V (online slot index for this seg
                    # = d_idx - 1).
                    online_slot = d_idx - 1
                    captured_kv_global[online_slot][layer_idx] = (
                        k_self_b.detach().clone(),
                        v_self_b.detach().clone(),
                    )

                    if use_headclass:
                        # ── Head-class path: global=full, local=window ──
                        # Online prev KV = cat of all earlier segments' self KV.
                        # Fast path: use the incrementally-extended running
                        # buffer (covers the common single-micro-batch case
                        # where every previous segment is in this batch). Slow
                        # path: if any previous segment lives in an EARLIER
                        # micro-batch (mb_start > 0 and that slot < mb_start),
                        # fall back to the explicit gather+cat for correctness.
                        _needs_cross_mb = mb_start > 0 and (d_idx - 1) > 0
                        if _needs_cross_mb:
                            _hc_prev_k: List[torch.Tensor] = []
                            _hc_prev_v: List[torch.Tensor] = []
                            for prev_seg_idx in range(1, d_idx):
                                prev_online_slot = prev_seg_idx - 1
                                if prev_online_slot < mb_start:
                                    pk, pv = captured_kv_global[prev_online_slot][
                                        layer_idx
                                    ]
                                else:
                                    rel = prev_online_slot - mb_start
                                    rel_Lb = mb_lens[rel]
                                    pk = k[rel : rel + 1, :, :rel_Lb, :]
                                    pv = v[rel : rel + 1, :, :rel_Lb, :]
                                _hc_prev_k.append(pk)
                                _hc_prev_v.append(pv)
                            hc_online_k = (
                                torch.cat(_hc_prev_k, dim=2) if _hc_prev_k else None
                            )
                            hc_online_v = (
                                torch.cat(_hc_prev_v, dim=2) if _hc_prev_v else None
                            )
                        else:
                            hc_online_k = _run_prev_k
                            hc_online_v = _run_prev_v

                        _meta = _hc_layer_meta[layer_idx]

                        attn_b = _headclass_online_attention(
                            q_b,
                            k_self_b,
                            v_self_b,
                            _meta["seg0_k"],
                            _meta["seg0_v"],
                            hc_online_k,
                            hc_online_v,
                            _meta["is_local"],
                            sink_size=_meta["sink"],
                            window=_meta["window"],
                            seg_offset=mb_offsets[b],
                            num_q_per_kv=num_kv_groups,
                            sm_scale=attn_module.scaling,
                        )
                        # Extend the running prev-KV buffer by this slot's self
                        # KV (one cat per slot -> O(B) total).
                        if _run_prev_k is None:
                            _run_prev_k = k_self_b
                            _run_prev_v = v_self_b
                        else:
                            _run_prev_k = torch.cat([_run_prev_k, k_self_b], dim=2)
                            _run_prev_v = torch.cat([_run_prev_v, v_self_b], dim=2)
                    elif use_segpaged:
                        # ── SegPaged path: pass 3 segments directly ──
                        # Collect online prev segments (shared across heads).
                        online_prev_parts_k: List[torch.Tensor] = []
                        online_prev_parts_v: List[torch.Tensor] = []
                        for prev_seg_idx in range(1, d_idx):
                            prev_online_slot = prev_seg_idx - 1
                            if prev_online_slot < mb_start:
                                pk, pv = captured_kv_global[prev_online_slot][layer_idx]
                            else:
                                rel = prev_online_slot - mb_start
                                rel_Lb = mb_lens[rel]
                                pk = k[rel : rel + 1, :, :rel_Lb, :]
                                pv = v[rel : rel + 1, :, :rel_Lb, :]
                            online_prev_parts_k.append(pk)
                            online_prev_parts_v.append(pv)

                        if online_prev_parts_k:
                            # [1, Hkv, L_online, D] → [Hkv, L_online, D]
                            _op_k = torch.cat(online_prev_parts_k, dim=2)
                            _op_v = torch.cat(online_prev_parts_v, dim=2)
                            online_k = _op_k[0]  # [Hkv, L_online, D]
                            online_v = _op_v[0]
                        else:
                            online_k = None
                            online_v = None

                        attn_b = _segpaged_causal_attention(
                            q_b,
                            k_self_b,
                            v_self_b,
                            filtered_seg0_k,
                            filtered_seg0_v,
                            filtered_seg0_lens,
                            online_k,
                            online_v,
                            num_q_per_kv=num_kv_groups,
                            sm_scale=attn_module.scaling,
                        )
                    else:
                        # ── Dense path: concat prev + kernel_fn. ──
                        prev_k_parts: List[torch.Tensor] = []
                        prev_v_parts: List[torch.Tensor] = []
                        # Segment 1 (always offline).
                        if d_idx >= 1:
                            p0k, p0v = segments_offline[0].kv[layer_idx]
                            prev_k_parts.append(p0k.to(q.device))
                            prev_v_parts.append(p0v.to(q.device))
                        # Segments 2..d_idx-1 = online segments.
                        for prev_seg_idx in range(1, d_idx):
                            prev_online_slot = prev_seg_idx - 1
                            if prev_online_slot < mb_start + b:
                                if prev_online_slot < mb_start:
                                    pk, pv = captured_kv_global[prev_online_slot][
                                        layer_idx
                                    ]
                                else:
                                    rel = prev_online_slot - mb_start
                                    rel_Lb = mb_lens[rel]
                                    pk = k[rel : rel + 1, :, :rel_Lb, :]
                                    pv = v[rel : rel + 1, :, :rel_Lb, :]
                                prev_k_parts.append(pk)
                                prev_v_parts.append(pv)
                            else:
                                raise RuntimeError(
                                    f"unexpected prev order: prev_online_slot="
                                    f"{prev_online_slot}, mb_start+b="
                                    f"{mb_start + b}"
                                )
                        prev_k_b = (
                            torch.cat(prev_k_parts, dim=2) if prev_k_parts else None
                        )
                        prev_v_b = (
                            torch.cat(prev_v_parts, dim=2) if prev_v_parts else None
                        )

                        # FA head-classified attention.
                        attn_b = kernel_fn(
                            q=q_b,
                            k_self=k_self_b,
                            v_self=v_self_b,
                            k_prev=prev_k_b,
                            v_prev=prev_v_b,
                            k_sink_padded=k_sink_shared,
                            v_sink_padded=v_sink_shared,
                            plan=plan,
                            num_q_per_kv=num_kv_groups,
                            sm_scale=attn_module.scaling,
                            retrieval_top_p=head_cfg.retrieval_top_p,
                        )

                    if Lb < mb_L_max:
                        pad = attn_b.new_zeros(
                            1, attn_b.shape[1], mb_L_max - Lb, attn_b.shape[3]
                        )
                        attn_b = torch.cat([attn_b, pad], dim=2)
                    attn_outputs_per_slot.append(attn_b)

                attn_out = torch.cat(attn_outputs_per_slot, dim=0)  # [B, Hq, L_max, D]

                # Capture token importance for batched Sparse FFN selection.
                if sparse_ffn_schedule is not None:
                    captured_importance[layer_idx] = token_importance_from_attn(
                        attn_out
                    ).detach()  # [B, L_max]

                attn_output = attn_out.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_module.o_proj(attn_output)
                return attn_output, None

            return patched_forward

        def make_patched_mlp(layer_idx: int):
            """Patch the layer MLP for batched partial Sparse FFN.

            Returns ``Z`` (the FFN delta) so the decoder's own residual add
            ``residual + Z`` realises Algorithm 1 lines 21-23 across all
            batched segments at once.
            """
            mlp_module = base_model.layers[layer_idx].mlp
            orig_mlp = mlp_module.forward
            comp_mlp = orig_mlp

            def patched_mlp(hidden_states, *a, **kw):
                importance = captured_importance[layer_idx]
                if importance is None or hidden_states.dim() != 3:
                    return comp_mlp(hidden_states, *a, **kw)
                sched = sparse_ffn_schedule
                if sched.is_dense_layer(layer_idx):
                    z = comp_mlp(hidden_states, *a, **kw)
                    if sparse_ffn_stats is not None:
                        B_, L_, _ = hidden_states.shape
                        sparse_ffn_stats.append(
                            {
                                "layer": layer_idx,
                                "mode": "dense",
                                "selected": B_ * L_,
                                "total": B_ * L_,
                                "selected_frac": 1.0,
                            }
                        )
                    return z
                out_xnext, stats = apply_sparse_ffn(
                    hidden_states,
                    lambda rows: comp_mlp(rows, *a, **kw),
                    layer_idx=layer_idx,
                    schedule=sched,
                    importance=importance,
                    return_stats=True,
                )
                if sparse_ffn_stats is not None:
                    sparse_ffn_stats.append(stats)
                return out_xnext - hidden_states  # return Z (delta)

            return patched_mlp

        # Patch, run one micro-batch, restore.
        orig_layer_forwards: dict = {}
        for layer_idx in range(n_layers):
            m = base_model.layers[layer_idx].self_attn
            orig_forwards[layer_idx] = m.forward
            m.forward = make_patched(layer_idx)
            if sparse_ffn_schedule is not None:
                mlp_mod = getattr(base_model.layers[layer_idx], "mlp", None)
                if mlp_mod is not None:
                    orig_mlp_forwards[layer_idx] = mlp_mod.forward
                    mlp_mod.forward = make_patched_mlp(layer_idx)
            if use_compile:
                # Wrap the (now-patched) layer forward with a compiled graph.
                # Graph-breaks at the patched attention/MLP; fuses the rest.
                lyr = base_model.layers[layer_idx]
                orig_layer_forwards[layer_idx] = lyr.forward
                lyr.forward = _get_compiled_layer(lyr)
        try:
            model(input_ids=input_ids_mb, position_ids=position_ids_mb, use_cache=False)
        finally:
            for layer_idx, orig in orig_layer_forwards.items():
                base_model.layers[layer_idx].forward = orig
            for layer_idx in range(n_layers):
                base_model.layers[layer_idx].self_attn.forward = orig_forwards[
                    layer_idx
                ]
            for layer_idx, orig in orig_mlp_forwards.items():
                base_model.layers[layer_idx].mlp.forward = orig

        # Help free intermediate activations before the next micro-batch.
        del input_ids_mb, position_ids_mb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Assemble doc_online_kvs ──
    doc_online_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    doc_online_kvs.append(
        [(k.to(device), v.to(device)) for k, v in segments_offline[0].kv]
    )
    for b in range(n_online):
        doc_online_kvs.append([captured_kv_global[b][li] for li in range(n_layers)])
    return doc_online_kvs


def _flat_headclass_attention(
    q: torch.Tensor,  # [1, Hq, T, D]   all online tokens (one sequence)
    k_on: torch.Tensor,  # [1, Hkv, T, D]  online tokens' own K
    v_on: torch.Tensor,
    seg0_k: torch.Tensor,  # [1, Hkv, L0, D]  offline seg-0 prefix (RoPE@[0,L0))
    seg0_v: torch.Tensor,
    is_local: torch.Tensor,  # [Hkv] bool
    *,
    sink_size: int,
    window: int,
    seg0_len: int,
    num_q_per_kv: int,
    sm_scale: float,
) -> torch.Tensor:
    """Single-pass head-class attention over the WHOLE online sequence.

    The online tokens form one contiguous causal sequence at global
    positions ``[seg0_len, seg0_len + T)``. KV = ``[seg0 | online]``.

    - global heads: causal over the full ``[seg0 | online]`` (query token i
      sees keys ``[0, L0 + i]``).
    - local heads: sink ∪ sliding window of the last ``window`` tokens.

    Two mask-free FlashAttention passes per class (native GQA + native
    window), merged by LSE. No attn_mask is ever built; no per-segment
    Python loop. This is the architectural collapse of the per-slot online
    attention into a single varlen-free call per layer.
    """
    Hq = q.shape[1]
    T = q.shape[2]
    D = q.shape[3]
    device = q.device
    out = q.new_zeros(1, Hq, T, D)

    # Multi-GPU (device_map) safety: when the model is sharded across GPUs the
    # head-class metadata (is_local, seg0 K/V) may live on a different device
    # than this layer's q/k/v. Move them onto q's device; for a single GPU this
    # is a no-op.
    if is_local.device != device:
        is_local = is_local.to(device)
    if seg0_k.device != device:
        seg0_k = seg0_k.to(device)
        seg0_v = seg0_v.to(device)
    if k_on.device != device:
        k_on = k_on.to(device)
        v_on = v_on.to(device)

    local_kv = torch.nonzero(is_local, as_tuple=False).flatten()
    global_kv = torch.nonzero(~is_local, as_tuple=False).flatten()

    def _q_heads_for(kv_idx: torch.Tensor) -> torch.Tensor:
        base = kv_idx.unsqueeze(1) * num_q_per_kv
        off = torch.arange(num_q_per_kv, device=device).unsqueeze(0)
        return (base + off).flatten()

    # ── GLOBAL heads: causal over [seg0 | online] ──
    if global_kv.numel() > 0:
        kg = torch.cat([seg0_k[:, global_kv], k_on[:, global_kv]], dim=2)
        vg = torch.cat([seg0_v[:, global_kv], v_on[:, global_kv]], dim=2)
        qh = _q_heads_for(global_kv)
        g_out, _ = _flash_attn_lse(q[:, qh], kg, vg, sm_scale, causal=True, window=-1)
        out[:, qh] = g_out

    # ── LOCAL heads: sink + sliding window over [seg0 | online] ──
    if local_kv.numel() > 0:
        s = max(0, min(sink_size, seg0_len))
        qh = _q_heads_for(local_kv)
        q_l = q[:, qh]
        # recent stream = [seg0_tail | online]; trim to last (window + T).
        rec_k = torch.cat([seg0_k[:, local_kv], k_on[:, local_kv]], dim=2)
        rec_v = torch.cat([seg0_v[:, local_kv], v_on[:, local_kv]], dim=2)
        keep = min(rec_k.shape[2], window + T)
        rec_k = rec_k[:, :, -keep:, :]
        rec_v = rec_v[:, :, -keep:, :]
        rec_out, rec_lse = _flash_attn_lse(
            q_l, rec_k, rec_v, sm_scale, causal=True, window=window
        )
        if s > 0:
            sink_k = seg0_k[:, local_kv, :s, :]
            sink_v = seg0_v[:, local_kv, :s, :]
            sink_out, sink_lse = _flash_attn_lse(
                q_l, sink_k, sink_v, sm_scale, causal=False, window=-1
            )
            out[:, qh] = _merge_lse(rec_out, rec_lse, sink_out, sink_lse)
        else:
            out[:, qh] = rec_out

    return out


_COMPILED_BLOCK_CACHE: dict = {}


def _get_compiled_block(fn, key, enable: bool):
    if not enable:
        return fn
    cached = _COMPILED_BLOCK_CACHE.get(key)
    if cached is None:
        # dynamic=True: ONE graph serves all sequence lengths. Real RAG
        # requests have slightly different online-token counts (e.g. 27762
        # vs 27832), and dynamic=False would recompile (~25s) for each,
        # destroying the speedup on every new request. dynamic=True marks
        # the seq-len dim symbolic so the compiled graph is reused.
        cached = torch.compile(fn, dynamic=True, fullgraph=True)
        _COMPILED_BLOCK_CACHE[key] = cached
    return cached


@torch.no_grad()
def _run_flat_custom(
    base_model,
    *,
    flat_ids,
    flat_pos,
    layer_meta,
    seg0_len,
    captured_kv,
    apply_rotary,
    sparse_ffn_schedule,
    sparse_ffn_stats,
    use_compile,
):
    """Custom single-pass flat forward (no HF model.forward wrapper).

    Per layer: compiled pre-block (input_norm + QKV proj + q/k norm + RoPE)
    -> eager head-class attention -> compiled post-block (o_proj + residual
    + post_norm) -> sparse MLP + residual. Final norm / lm_head are skipped
    (online prefill only needs the per-layer KV).
    """
    device = flat_ids.device
    layers = base_model.layers
    n_layers = len(layers)

    # Embedding + rotary (computed once).
    h = base_model.embed_tokens(flat_ids)
    cos, sin = base_model.rotary_emb(h, flat_pos)

    for li in range(n_layers):
        layer = layers[li]
        attn = layer.self_attn
        meta = layer_meta[li]
        head_dim = attn.head_dim
        num_kv_groups = attn.num_key_value_groups

        # ── pre-block: input_norm -> q/k/v proj -> q/k norm -> RoPE ──
        def pre_block(hs, cos, sin, _attn=attn, _layer=layer, _hd=head_dim):
            x = _layer.input_layernorm(hs)
            ishape = x.shape[:-1]
            hshape = (*ishape, -1, _hd)
            q = _attn.q_proj(x).view(hshape).transpose(1, 2)
            k = _attn.k_proj(x).view(hshape).transpose(1, 2)
            v = _attn.v_proj(x).view(hshape).transpose(1, 2)
            if getattr(_attn, "q_norm", None) is not None:
                q = _attn.q_norm(q)
            if getattr(_attn, "k_norm", None) is not None:
                k = _attn.k_norm(k)
            q, k = apply_rotary(q, k, cos, sin)
            return q, k, v

        pre = _get_compiled_block(pre_block, (id(layer), "pre"), use_compile)
        q, k, v = pre(h, cos, sin)

        captured_kv[li] = (k.detach(), v.detach())

        # ── eager head-class attention ──
        attn_out = _flat_headclass_attention(
            q,
            k,
            v,
            meta["seg0_k"],
            meta["seg0_v"],
            meta["is_local"],
            sink_size=meta["sink"],
            window=meta["window"],
            seg0_len=seg0_len,
            num_q_per_kv=num_kv_groups,
            sm_scale=attn.scaling,
        )
        importance = (
            token_importance_from_attn(attn_out).detach()
            if sparse_ffn_schedule is not None
            else None
        )

        # ── post-attention: o_proj + residual ──
        def post_attn(hs, ao, _attn=attn, _hd=head_dim):
            ishape = hs.shape[:-1]
            a = ao.transpose(1, 2).contiguous().reshape(*ishape, -1)
            return hs + _attn.o_proj(a)

        post = _get_compiled_block(post_attn, (id(layer), "post"), use_compile)
        h = post(h, attn_out)

        # ── MLP block: post_norm -> sparse FFN -> residual ──
        residual = h
        x = layer.post_attention_layernorm(h)
        if sparse_ffn_schedule is not None and not sparse_ffn_schedule.is_dense_layer(
            li
        ):
            out_xnext, stats = apply_sparse_ffn(
                x,
                lambda rows: layer.mlp(rows),
                layer_idx=li,
                schedule=sparse_ffn_schedule,
                importance=importance,
                return_stats=True,
            )
            if sparse_ffn_stats is not None:
                sparse_ffn_stats.append(stats)
            h = residual + (out_xnext - x)
        else:
            mlp_block = _get_compiled_block(
                layer.mlp.forward, (id(layer.mlp), "mlp"), use_compile
            )
            h = residual + mlp_block(x)
            if sparse_ffn_stats is not None and sparse_ffn_schedule is not None:
                B_, L_, _ = x.shape
                sparse_ffn_stats.append(
                    {
                        "layer": li,
                        "mode": "dense",
                        "selected": B_ * L_,
                        "total": B_ * L_,
                        "selected_frac": 1.0,
                    }
                )


@torch.no_grad()
def online_forward_segments_flat(
    model,
    *,
    segments_offline: List[OfflineSegment],
    head_cfg: HeadClassConfig,
    rope_helper: RoPEHelper,
    sparse_ffn_schedule: Optional[SparseFFNSchedule] = None,
    sparse_ffn_stats: Optional[List[dict]] = None,
    use_compile: bool = False,
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Single-pass online prefill: all online segments as ONE sequence.

    Unlike :func:`online_forward_segments_batched` (which pads each segment
    into a batch slot and loops the attention per slot, re-concatenating
    growing prev KV each time — O(N^2) cat + N kernel launches per layer),
    this runs the entire online token stream as a single causal forward and
    does ONE head-class attention call per layer. This collapses the
    per-slot Python/launch overhead that prevented the algorithmic FLOPs
    savings from materialising as wall-time speedup.

    Returns ``doc_online_kvs[seg][layer] = (K, V)`` with seg 0 = offline KV
    (RoPE-repositioned downstream) and segs 1..N = the online K/V sliced
    back out of the flat sequence, so the downstream query-forward sees the
    same per-segment KV as the batched path.
    """
    config = model.config
    n_layers = config.num_hidden_layers
    base_model = model.model if hasattr(model, "model") else model
    device = model.device
    apply_rotary = _get_apply_rotary(config.model_type)

    seg_lens = [s.doc_len for s in segments_offline]
    seg0_len = seg_lens[0]
    online_lens = seg_lens[1:]
    n_online = len(online_lens)
    if n_online == 0:
        return [[(k.to(device), v.to(device)) for k, v in segments_offline[0].kv]]

    # Flat online token ids + true global positions.
    online_ids = []
    online_pos = []
    p = seg0_len
    for s in segments_offline[1:]:
        ids = s.token_ids.to(device)
        online_ids.append(ids)
        online_pos.append(torch.arange(p, p + s.doc_len, device=device))
        p += s.doc_len
    flat_ids = torch.cat(online_ids).unsqueeze(0)  # [1, T]
    flat_pos = torch.cat(online_pos).unsqueeze(0)  # [1, T]
    T = flat_ids.shape[1]

    # Per-layer head-class metadata + seg0 KV repositioned to [0, L0).
    layer_meta: List[dict] = []
    for li in range(n_layers):
        pol = _head_policy_for_layer(head_cfg, li)
        win = _head_window_for_layer(head_cfg, li)
        snk = _head_sink_for_layer(head_cfg, li)
        is_local = torch.tensor(
            [pp == POLICY_LOCAL for pp in pol], dtype=torch.bool, device=device
        )
        if bool(is_local.any()):
            fl = int(is_local.nonzero()[0].item())
            sink_sz = snk[fl]
            wn = win[fl] if win[fl] > 0 else 512
        else:
            sink_sz, wn = 4, 512
        s0k = segments_offline[0].kv[li][0].to(device)
        s0v = segments_offline[0].kv[li][1].to(device)
        layer_meta.append(
            {
                "is_local": is_local,
                "sink": sink_sz,
                "window": wn,
                "seg0_k": s0k,
                "seg0_v": s0v,
            }
        )

    captured_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
    captured_importance: List[Optional[torch.Tensor]] = [None] * n_layers

    def make_patched_attn(layer_idx: int):
        attn_module = base_model.layers[layer_idx].self_attn
        meta = layer_meta[layer_idx]

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)
            num_kv_groups = attn_module.num_key_value_groups
            q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if getattr(attn_module, "q_norm", None) is not None:
                q = attn_module.q_norm(q)
            if getattr(attn_module, "k_norm", None) is not None:
                k = attn_module.k_norm(k)
            cos, sin = position_embeddings
            if cos.device != q.device:
                cos = cos.to(q.device)
                sin = sin.to(q.device)
            q, k = apply_rotary(q, k, cos, sin)

            captured_kv[layer_idx] = (k.detach(), v.detach())

            attn_out = _flat_headclass_attention(
                q,
                k,
                v,
                meta["seg0_k"],
                meta["seg0_v"],
                meta["is_local"],
                sink_size=meta["sink"],
                window=meta["window"],
                seg0_len=seg0_len,
                num_q_per_kv=num_kv_groups,
                sm_scale=attn_module.scaling,
            )
            if sparse_ffn_schedule is not None:
                captured_importance[layer_idx] = token_importance_from_attn(
                    attn_out
                ).detach()
            attn_output = attn_out.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, None

        return patched_forward

    def make_patched_mlp(layer_idx: int):
        mlp_module = base_model.layers[layer_idx].mlp
        orig_mlp = mlp_module.forward

        def patched_mlp(hidden_states, *a, **kw):
            importance = captured_importance[layer_idx]
            if importance is None or hidden_states.dim() != 3:
                return orig_mlp(hidden_states, *a, **kw)
            sched = sparse_ffn_schedule
            if sched.is_dense_layer(layer_idx):
                z = orig_mlp(hidden_states, *a, **kw)
                if sparse_ffn_stats is not None:
                    B_, L_, _ = hidden_states.shape
                    sparse_ffn_stats.append(
                        {
                            "layer": layer_idx,
                            "mode": "dense",
                            "selected": B_ * L_,
                            "total": B_ * L_,
                            "selected_frac": 1.0,
                        }
                    )
                return z
            out_xnext, stats = apply_sparse_ffn(
                hidden_states,
                lambda rows: orig_mlp(rows, *a, **kw),
                layer_idx=layer_idx,
                schedule=sched,
                importance=importance,
                return_stats=True,
            )
            if sparse_ffn_stats is not None:
                sparse_ffn_stats.append(stats)
            return out_xnext - hidden_states

        return patched_mlp

    use_custom = os.environ.get("REDKNOT_CUSTOM_FWD", "1") == "1"
    if use_custom:
        # ── Custom flat forward: bypass HF model.forward wrapper entirely. ──
        # Runs embed -> [per-layer: compiled pre-block -> eager head-class
        # attention -> compiled post-block] -> (no final norm / lm_head).
        # This removes the HF attention-mask prep, the final RMSNorm over all
        # T tokens, and the per-layer Python dispatch that survived the
        # layer-level compile graph-breaks.
        _run_flat_custom(
            base_model,
            flat_ids=flat_ids,
            flat_pos=flat_pos,
            layer_meta=layer_meta,
            seg0_len=seg0_len,
            captured_kv=captured_kv,
            apply_rotary=apply_rotary,
            sparse_ffn_schedule=sparse_ffn_schedule,
            sparse_ffn_stats=sparse_ffn_stats,
            use_compile=use_compile,
        )
    else:
        orig_attn: dict = {}
        orig_mlp: dict = {}
        orig_layer: dict = {}
        for li in range(n_layers):
            m = base_model.layers[li].self_attn
            orig_attn[li] = m.forward
            m.forward = make_patched_attn(li)
            if sparse_ffn_schedule is not None:
                mm = getattr(base_model.layers[li], "mlp", None)
                if mm is not None:
                    orig_mlp[li] = mm.forward
                    mm.forward = make_patched_mlp(li)
            if use_compile:
                lyr = base_model.layers[li]
                orig_layer[li] = lyr.forward
                lyr.forward = _get_compiled_layer(lyr, dynamic=False)
        try:
            base_model(input_ids=flat_ids, position_ids=flat_pos, use_cache=False)
        finally:
            for li, o in orig_layer.items():
                base_model.layers[li].forward = o
            for li in range(n_layers):
                base_model.layers[li].self_attn.forward = orig_attn[li]
            for li, o in orig_mlp.items():
                base_model.layers[li].mlp.forward = o

    # Slice flat online K/V back into per-segment KV (so downstream query
    # forward sees the same layout as the batched path).
    doc_online_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    doc_online_kvs.append(
        [(k.to(device), v.to(device)) for k, v in segments_offline[0].kv]
    )
    cuts = []
    off = 0
    for ln in online_lens:
        cuts.append((off, off + ln))
        off += ln
    for a, b in cuts:
        seg_kv = []
        for li in range(n_layers):
            k_all, v_all = captured_kv[li]
            seg_kv.append((k_all[:, :, a:b, :], v_all[:, :, a:b, :]))
        doc_online_kvs.append(seg_kv)
    return doc_online_kvs


@torch.no_grad()
def run_redknot_batched(
    model,
    tokenizer,
    *,
    segments_offline: List[OfflineSegment],
    query_text: str,
    head_cfg: HeadClassConfig,
    rope_helper: Optional[RoPEHelper] = None,
    max_new_tokens: int = 100,
    kernel: str = "fa3_parallel",
    micro_batch_size: Optional[int] = None,
    sparse_ffn_schedule: Optional[SparseFFNSchedule] = None,
    sparse_ffn_stats: Optional[List[dict]] = None,
    use_segpaged: bool = False,
) -> Tuple[torch.Tensor, str, int, float]:
    """End-to-end RedKnot with batched online prefill.

    ``micro_batch_size`` controls how many segments share one forward
    pass. ``None`` means "all at once" (OOM-prone on long contexts);
    ``2`` is a safe default for 64k Llama-70B on 80GB.

    When ``use_segpaged=True``, the online prefill attention uses
    SegPagedAttention (per-head paged KV, mask-free varlen kernels)
    instead of the dense kernel_fn path.

    Same outputs as :func:`driver.run_redknot` (first_logits, text,
    query_len, ttft_seconds). The TTFT measurement excludes offline
    prefill (which is performed before this call) and includes only:
    the batched online forward, KV concat, and the query forward.
    """
    from transformers import DynamicCache

    if rope_helper is None:
        base_model = model.model if hasattr(model, "model") else model
        rope_helper = RoPEHelper(base_model.rotary_emb)

    doc_lens = [seg.doc_len for seg in segments_offline]
    offsets, p = [], 0
    for dl in doc_lens:
        offsets.append(p)
        p += dl
    query_offset = p
    n_layers = model.config.num_hidden_layers

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ── A. Batched online forward for segments 2..N ──
    doc_online_kvs = online_forward_segments_batched(
        model,
        segments_offline=segments_offline,
        head_cfg=head_cfg,
        rope_helper=rope_helper,
        kernel=kernel,
        micro_batch_size=micro_batch_size,
        sparse_ffn_schedule=sparse_ffn_schedule,
        sparse_ffn_stats=sparse_ffn_stats,
        use_segpaged=use_segpaged,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t_after_online = time.perf_counter()

    # ── B. KV concat into DynamicCache for the query forward ──
    past = DynamicCache()
    base_model = model.model if hasattr(model, "model") else model
    for layer_idx in range(n_layers):
        layer_device = next(base_model.layers[layer_idx].self_attn.parameters()).device
        k_parts = [
            doc_online_kvs[d][layer_idx][0].to(layer_device)
            for d in range(len(segments_offline))
        ]
        v_parts = [
            doc_online_kvs[d][layer_idx][1].to(layer_device)
            for d in range(len(segments_offline))
        ]
        past.update(torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2), layer_idx)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t_after_kv = time.perf_counter()

    # ── C. Query forward (SDPA so long context stays in HBM) ──
    query_ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    query_len = int(query_ids.shape[1])
    total_kv = sum(doc_lens)
    qpos = torch.arange(
        query_offset, query_offset + query_len, device=model.device
    ).unsqueeze(0)
    cpos = torch.arange(total_kv, total_kv + query_len, device=model.device)
    _q_orig = _switch_attn_impl(model, "sdpa")
    try:
        out = model(
            input_ids=query_ids,
            position_ids=qpos,
            past_key_values=past,
            cache_position=cpos,
            use_cache=True,
        )
    finally:
        _restore_attn_impl(model, _q_orig)
    first_logits = out.logits[0, -1, :].clone()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t_start

    # Phase timing (printed to stderr so benchmark stdout stays clean)
    _t_online = _t_after_online - t_start
    _t_kv = _t_after_kv - _t_after_online
    _t_query = ttft - (_t_online + _t_kv)
    import sys as _sys

    print(
        f"  [phases] online={_t_online:.3f}s  kv_concat={_t_kv:.3f}s  "
        f"query_fwd={_t_query:.3f}s  total_ttft={ttft:.3f}s  "
        f"(segpaged={use_segpaged})",
        file=_sys.stderr,
    )

    # Greedy decode.
    generated = []
    past_for_gen = out.past_key_values
    next_id = first_logits.argmax().unsqueeze(0).unsqueeze(0)
    generated.append(int(next_id[0, 0].item()))
    total_seen = total_kv + query_len
    for step in range(max_new_tokens - 1):
        cur_pos = total_seen + len(generated) - 1
        out_g = model(
            input_ids=next_id,
            position_ids=torch.tensor([[cur_pos]], device=model.device),
            past_key_values=past_for_gen,
            cache_position=torch.tensor([cur_pos], device=model.device),
            use_cache=True,
        )
        past_for_gen = out_g.past_key_values
        next_id = out_g.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        tid = int(next_id[0, 0].item())
        generated.append(tid)
        if tid == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    del past, past_for_gen, out, doc_online_kvs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return first_logits, text, query_len, ttft


# ──────────────────────────────────────────────────────────────────────────
# Query-forward per-head attention (paper head-class design)
# ──────────────────────────────────────────────────────────────────────────
def _grouped_sdpa(
    q: torch.Tensor,  # [1, Hq_sub, Lq, D]
    k: torch.Tensor,  # [1, Hkv_sub, L_kv, D]
    v: torch.Tensor,
    *,
    num_q_per_kv: int,
    sm_scale: float,
    self_causal: torch.Tensor,  # [Lq, Lq] bool lower-tri
) -> torch.Tensor:
    """Batched GQA attention over a head subset.

    The trailing ``Lq`` KV columns are the query's own self-KV; the
    cached-context prefix is fully visible, the self-block is causal.
    Returns ``[1, Hq_sub, Lq, D]``.
    """
    Hq_sub = q.shape[1]
    Hkv_sub = k.shape[1]
    Lq = q.shape[2]
    D = q.shape[3]
    L_kv = k.shape[2]
    L_ctx = L_kv - Lq
    # Expand KV across the GQA group: [1, Hkv_sub, L_kv, D] -> [1, Hq_sub, ...]
    k_exp = k.repeat_interleave(num_q_per_kv, dim=1)
    v_exp = v.repeat_interleave(num_q_per_kv, dim=1)
    # Build additive mask [Lq, L_kv]: prefix visible, self-block causal.
    attn_mask = torch.zeros(Lq, L_kv, dtype=q.dtype, device=q.device)
    if Lq > 1 and L_ctx >= 0:
        neg = torch.finfo(q.dtype).min
        attn_mask[:, L_ctx:] = torch.where(self_causal, attn_mask[:, L_ctx:], neg)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=attn_mask, scale=sm_scale
    )
    return out


def _query_headclass_attention(
    q: torch.Tensor,  # [1, Hq, Lq, D]
    k_global: torch.Tensor,  # [1, Hkv, L_g, D]  context + Lq self KV
    v_global: torch.Tensor,
    k_local: torch.Tensor,  # [1, Hkv, L_l, D]  context + Lq self KV
    v_local: torch.Tensor,
    is_local: torch.Tensor,  # [Hkv] bool
    *,
    num_q_per_kv: int,
    sm_scale: float,
) -> torch.Tensor:
    """Vectorized per-class attention for the query-forward / decode step.

    Implements the paper's head-class reuse contract at query time
    (introduction.tex L78-79, exp.tex L57-58):

    - **global heads**: attend over the full re-prefilled context
      ``k_global`` (full-context recovery).
    - **local heads**: attend only over their cached ``sink + window``
      KV ``k_local`` (reused verbatim within a sliding window).

    Instead of a per-head Python loop, global and local KV heads are each
    processed in a single batched GQA SDPA call, then scattered back to
    the original query-head order.

    Layout contract: the **last ``Lq`` columns** of ``k_global`` / ``k_local``
    are the query tokens' own self-KV (appended by the caller). The cached
    context prefix is fully visible; the trailing ``Lq x Lq`` self-block is
    causal.
    """
    Hq = q.shape[1]
    Hkv = k_global.shape[1]
    Lq = q.shape[2]
    D = q.shape[3]
    device = q.device
    out = q.new_zeros(1, Hq, Lq, D)

    self_causal = torch.ones(Lq, Lq, dtype=torch.bool, device=device).tril()

    # Map kv-head subsets to their query-head index ranges (GQA contiguous).
    local_kv = torch.nonzero(is_local, as_tuple=False).flatten()
    global_kv = torch.nonzero(~is_local, as_tuple=False).flatten()

    def _q_heads_for(kv_idx: torch.Tensor) -> torch.Tensor:
        # kv head h -> query heads [h*g, h*g+g)
        base = kv_idx.unsqueeze(1) * num_q_per_kv  # [n, 1]
        off = torch.arange(num_q_per_kv, device=device).unsqueeze(0)  # [1, g]
        return (base + off).flatten()

    if global_kv.numel() > 0:
        qh_idx = _q_heads_for(global_kv)
        out[:, qh_idx] = _grouped_sdpa(
            q[:, qh_idx],
            k_global[:, global_kv],
            v_global[:, global_kv],
            num_q_per_kv=num_q_per_kv,
            sm_scale=sm_scale,
            self_causal=self_causal,
        )
    if local_kv.numel() > 0:
        qh_idx = _q_heads_for(local_kv)
        out[:, qh_idx] = _grouped_sdpa(
            q[:, qh_idx],
            k_local[:, local_kv],
            v_local[:, local_kv],
            num_q_per_kv=num_q_per_kv,
            sm_scale=sm_scale,
            self_causal=self_causal,
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# Head-class hybrid KV-reuse path (paper-faithful)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_redknot_offlinekv(
    model,
    tokenizer,
    *,
    segments_offline: List[OfflineSegment],
    query_text: str,
    head_cfg: HeadClassConfig,
    rope_helper: Optional[RoPEHelper] = None,
    max_new_tokens: int = 100,
    kernel: str = "fa3_parallel",
    micro_batch_size: Optional[int] = None,
    sparse_ffn_schedule: Optional[SparseFFNSchedule] = None,
    sparse_ffn_stats: Optional[List[dict]] = None,
    use_compile: bool = False,
    use_flat: bool = True,
    window_ratio: Optional[float] = None,
    window_min: int = 256,
) -> Tuple[torch.Tensor, str, int, float]:
    """RedKnot head-class hybrid KV reuse (paper-faithful design).

    ``window_ratio``: if set, the local-head sliding window is adapted to the
    request's total context length as ``window = max(window_min, ctx//(1/r))``,
    i.e. ``window = max(window_min, int(total_ctx * window_ratio))``. This
    realises the offline-profiled "sweet spot" where the window scales with
    the text length (e.g. ``window_ratio=0.5`` -> window = ctx/2). When
    ``None`` the config's fixed per-head window is used.

    This implements the paper's reuse contract exactly (introduction.tex
    L78-79, exp.tex L57-58, design.tex §sec:elastic:head):

    1. **Global heads (12-15%) are re-prefilled on reuse.** Because the
       hidden-state stream is shared across all heads, "re-prefill" is
       realised by running the (sparse) online forward over segments
       2..N — the same path as :func:`run_redknot_batched` — which
       recovers full-context attention for global heads and Sparse-FFN
       for the channel axis. The resulting KV is the *recomputed* value.

    2. **Local heads (85-88%) are reused verbatim within a sliding
       window.** Their KV is taken directly from the offline cache, RoPE-
       repositioned to global coordinates, and physically truncated to
       ``sink + window`` tokens. The recomputed local-head KV from the
       online forward is *discarded* in favour of the cheap offline copy.

    3. **Query forward uses per-head attention** (:func:`_query_headclass_
       attention`): global heads attend the full re-prefilled context,
       local heads attend only their ``sink + window`` KV.

    The speedup over the dense baseline comes from head-class attention
    sparsity (only global heads do full-context work) plus Sparse FFN,
    *not* from skipping the online forward — which would corrupt the
    global heads' cross-segment signal.
    """
    if rope_helper is None:
        base_model = model.model if hasattr(model, "model") else model
        rope_helper = RoPEHelper(base_model.rotary_emb)

    n_layers = model.config.num_hidden_layers
    device = model.device
    base_model = model.model if hasattr(model, "model") else model
    Hkv = head_cfg.num_kv_heads

    doc_lens = [seg.doc_len for seg in segments_offline]
    offsets, p = [], 0
    for dl in doc_lens:
        offsets.append(p)
        p += dl
    total_kv = p
    query_offset = total_kv

    # Adaptive sliding window: scale local-head window with context length
    # (offline-profiled sweet spot, e.g. window = ctx/2). Applied in-place to
    # the head config before any forward so both online and query stages see it.
    if window_ratio is not None and window_ratio > 0:
        adapt_w = max(window_min, int(total_kv * window_ratio))
        head_cfg.set_local_window(adapt_w)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ── A. Re-prefill: sparse online forward (global heads recovered) ──
    # Reuses the batched SegPaged + Sparse-FFN online path. The returned
    # doc_online_kvs[seg][layer] = (K, V) are the *recomputed* KV; we keep
    # them for GLOBAL heads only.
    if use_flat:
        doc_online_kvs = online_forward_segments_flat(
            model,
            segments_offline=segments_offline,
            head_cfg=head_cfg,
            rope_helper=rope_helper,
            sparse_ffn_schedule=sparse_ffn_schedule,
            sparse_ffn_stats=sparse_ffn_stats,
            use_compile=use_compile,
        )
    else:
        doc_online_kvs = online_forward_segments_batched(
            model,
            segments_offline=segments_offline,
            head_cfg=head_cfg,
            rope_helper=rope_helper,
            kernel=kernel,
            micro_batch_size=micro_batch_size,
            sparse_ffn_schedule=sparse_ffn_schedule,
            sparse_ffn_stats=sparse_ffn_stats,
            use_headclass=True,
            use_compile=use_compile,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t_after_online = time.perf_counter()

    # ── B. RoPE-reposition offline KV (for LOCAL heads' verbatim reuse) ──
    # Each offline segment's KV is rotated under positions [0, L); shift to
    # global [offset, offset + L).
    repositioned_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    for seg_idx, seg in enumerate(segments_offline):
        seg_offset = offsets[seg_idx]
        seg_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(n_layers):
            k_orig, v_orig = seg.kv[layer_idx]
            k_orig = k_orig.to(device)
            v_orig = v_orig.to(device)
            if seg_offset != 0:
                k_repositioned = rope_helper.reposition_offset(
                    k_orig, src_start=0, dst_start=seg_offset
                )
            else:
                k_repositioned = k_orig
            seg_kvs.append((k_repositioned, v_orig))
        repositioned_kvs.append(seg_kvs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t_after_rope = time.perf_counter()

    # ── C. Per-layer head-class KV assembly + query forward ──
    # Pre-compute per-layer head policy / window / sink and is_local mask.
    layer_head_info = []
    for layer_idx in range(n_layers):
        policies = _head_policy_for_layer(head_cfg, layer_idx)
        windows = _head_window_for_layer(head_cfg, layer_idx)
        sinks = _head_sink_for_layer(head_cfg, layer_idx)
        is_local = torch.tensor(
            [pl == POLICY_LOCAL for pl in policies], dtype=torch.bool, device=device
        )
        layer_head_info.append(
            {"windows": windows, "sinks": sinks, "is_local": is_local}
        )

    # Build the per-layer KV used by the query forward.
    #   k_global/v_global : full re-prefilled KV (online forward result)
    #   k_local/v_local   : offline+RoPE KV truncated to sink+window
    query_layer_kv: List[dict] = []
    for layer_idx in range(n_layers):
        layer_device = next(base_model.layers[layer_idx].self_attn.parameters()).device
        info = layer_head_info[layer_idx]

        # GLOBAL heads: concat the re-prefilled (recomputed) KV.
        g_k_parts = [
            doc_online_kvs[d][layer_idx][0].to(layer_device)
            for d in range(len(segments_offline))
        ]
        g_v_parts = [
            doc_online_kvs[d][layer_idx][1].to(layer_device)
            for d in range(len(segments_offline))
        ]
        k_global = torch.cat(g_k_parts, dim=2)  # [1, Hkv, total_kv, D]
        v_global = torch.cat(g_v_parts, dim=2)

        # LOCAL heads: concat offline+RoPE KV, then truncate to sink+window.
        l_k_parts = [
            repositioned_kvs[d][layer_idx][0].to(layer_device)
            for d in range(len(segments_offline))
        ]
        l_v_parts = [
            repositioned_kvs[d][layer_idx][1].to(layer_device)
            for d in range(len(segments_offline))
        ]
        k_local_full = torch.cat(l_k_parts, dim=2)  # [1, Hkv, total_kv, D]
        v_local_full = torch.cat(l_v_parts, dim=2)

        # sink+window indices (shared across local heads; use the first one).
        is_local_list = info["is_local"].tolist()
        if any(is_local_list):
            first_local_h = is_local_list.index(True)
            sink_size = info["sinks"][first_local_h]
            window_size = info["windows"][first_local_h]
            if window_size <= 0:
                window_size = 512
            local_idx = local_visible_indices(
                total_kv, sink_size, window_size, device=layer_device
            )
            k_local = k_local_full[:, :, local_idx, :]  # [1, Hkv, L_l, D]
            v_local = v_local_full[:, :, local_idx, :]
        else:
            # No local heads in this layer; local KV is unused but keep a
            # small placeholder to avoid None-handling downstream.
            k_local = k_local_full[:, :, :1, :]
            v_local = v_local_full[:, :, :1, :]

        query_layer_kv.append(
            {
                "k_global": k_global,
                "v_global": v_global,
                "k_local": k_local,
                "v_local": v_local,
                "is_local": info["is_local"].to(layer_device),
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t_after_kv = time.perf_counter()

    # ── D. Query forward with monkey-patched per-head attention ──
    query_ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    query_len = int(query_ids.shape[1])
    qpos = torch.arange(
        query_offset, query_offset + query_len, device=device
    ).unsqueeze(0)

    apply_rotary = _get_apply_rotary(model.config.model_type)

    def make_query_patched(layer_idx: int):
        attn_module = base_model.layers[layer_idx].self_attn
        kv = query_layer_kv[layer_idx]

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)
            num_kv_groups = attn_module.num_key_value_groups

            q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if getattr(attn_module, "q_norm", None) is not None:
                q = attn_module.q_norm(q)
            if getattr(attn_module, "k_norm", None) is not None:
                k = attn_module.k_norm(k)
            cos, sin = position_embeddings
            if cos.device != q.device:
                cos = cos.to(q.device)
                sin = sin.to(q.device)
            q, k = apply_rotary(q, k, cos, sin)

            # Append the query's own self KV so query tokens can attend to
            # each other and to themselves (causal among the suffix).
            # GLOBAL heads: [k_global | k_self]; LOCAL heads: [k_local | k_self].
            k_g = torch.cat([kv["k_global"], k], dim=2)
            v_g = torch.cat([kv["v_global"], v], dim=2)
            k_l = torch.cat([kv["k_local"], k], dim=2)
            v_l = torch.cat([kv["v_local"], v], dim=2)

            attn_out = _query_headclass_attention(
                q,
                k_g,
                v_g,
                k_l,
                v_l,
                kv["is_local"],
                num_q_per_kv=num_kv_groups,
                sm_scale=attn_module.scaling,
            )

            attn_output = attn_out.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, None

        return patched_forward

    orig_forwards: dict = {}
    for layer_idx in range(n_layers):
        m = base_model.layers[layer_idx].self_attn
        orig_forwards[layer_idx] = m.forward
        m.forward = make_query_patched(layer_idx)
    try:
        out = model(input_ids=query_ids, position_ids=qpos, use_cache=False)
    finally:
        for layer_idx in range(n_layers):
            base_model.layers[layer_idx].self_attn.forward = orig_forwards[layer_idx]

    first_logits = out.logits[0, -1, :].clone()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t_start

    # Phase timing
    _t_online = _t_after_online - t_start
    _t_rope = _t_after_rope - _t_after_online
    _t_kv = _t_after_kv - _t_after_rope
    _t_query = ttft - (_t_online + _t_rope + _t_kv)
    import sys as _sys

    print(
        f"  [headclass] online={_t_online:.3f}s  rope={_t_rope:.3f}s  "
        f"kv_build={_t_kv:.3f}s  query_fwd={_t_query:.3f}s  "
        f"total_ttft={ttft:.3f}s",
        file=_sys.stderr,
    )

    # ── E. Greedy decode (all heads attend full KV — correctness first) ──
    # Decode is only a few dozen tokens and is OUTSIDE TTFT, so head-class
    # sparsity here buys almost nothing while the per-head sliding-window
    # bookkeeping is error-prone. We therefore build a standard DynamicCache
    # from the full (global) re-prefilled KV and let every head attend the
    # full context via the model's native SDPA path. This removes the buggy
    # per-head decode pool that was the main cause of the F1 drop.
    from transformers import DynamicCache

    # Free large intermediates we no longer need BEFORE materializing the
    # decode cache, so peak memory stays bounded at long context (64K).
    del repositioned_kvs, doc_online_kvs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build the decode cache by *moving* the global KV (no .contiguous() copy
    # — the cat'd tensors are already contiguous). Drop each layer's source
    # reference as we go so we never hold two full copies.
    past = DynamicCache()
    for layer_idx in range(n_layers):
        kv = query_layer_kv[layer_idx]
        past.update(kv["k_global"], kv["v_global"], layer_idx)
        kv["k_global"] = None
        kv["v_global"] = None
        kv["k_local"] = None
        kv["v_local"] = None
    query_layer_kv = None  # free the per-head query KV; decode uses `past`.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generated: List[int] = []
    _q_orig = _switch_attn_impl(model, "sdpa")
    try:
        # Seed the cache with the query tokens (positions [total_kv,
        # total_kv+query_len)). Their logits reproduce first_logits; we keep
        # first_logits from Phase D (head-class) for the cosine metric but
        # drive decode from this consistent full-attention cache.
        cpos = torch.arange(total_kv, total_kv + query_len, device=device)
        seed = model(
            input_ids=query_ids,
            position_ids=qpos,
            past_key_values=past,
            cache_position=cpos,
            use_cache=True,
        )
        past = seed.past_key_values
        next_id = seed.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        generated.append(int(next_id[0, 0].item()))
        total_seen = total_kv + query_len
        for step in range(max_new_tokens - 1):
            cur_pos = total_seen + len(generated) - 1
            out_g = model(
                input_ids=next_id,
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=past,
                cache_position=torch.tensor([cur_pos], device=device),
                use_cache=True,
            )
            past = out_g.past_key_values
            next_id = out_g.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            tid = int(next_id[0, 0].item())
            generated.append(tid)
            if tid == tokenizer.eos_token_id:
                break
    finally:
        _restore_attn_impl(model, _q_orig)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    del out, past
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return first_logits, text, query_len, ttft
