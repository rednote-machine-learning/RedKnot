# Copyright 2024-2026 SGLang RedKnot Integration.
"""Per-layer head-classification mask plan for RedKnot.

This is the metadata RedKnot attaches to one layer so the attention
kernel can dispatch each KV head to the right strategy (local / global /
retrieval / dense). It is small (a couple of int32 tensors of shape
``[KVH]``) and built once per layer per forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig


@dataclass
class LayerMaskPlan:
    """Per-head dispatch table for one layer.

    Attributes
    ----------
    type_codes : ``[KVH]`` int32 — head type per KV head (see
        ``HeadClassConfig.TYPE_*``).
    window : ``[KVH]`` int32 — local sliding-window per head (``-1`` for
        non-local).
    sink_keep : ``[KVH, S_max]`` bool — which padded sink slots are
        actually owned by each head (sink sizes can be heterogeneous).
    sink_max : largest sink size across all heads of this layer.
    """

    type_codes: torch.Tensor
    window: torch.Tensor
    sink_keep: torch.Tensor
    sink_max: int


def build_layer_mask_plan(
    head_cfg: HeadClassConfig,
    layer_idx: int,
    device: torch.device,
    kv_head_start: int = 0,
    kv_head_count: Optional[int] = None,
) -> LayerMaskPlan:
    """Materialise the per-head mask plan for one layer.

    Under tensor parallelism each rank owns only a contiguous slice of the
    KV heads. ``kv_head_start`` / ``kv_head_count`` select that slice from the
    (global) head config so the resulting plan tensors are sized to this rank's
    local KV heads (matching ``layer.tp_k_head_num``). Defaults select all heads
    (single-rank behaviour).
    """
    tensors = head_cfg.as_tensors(device, dtype=torch.int32)
    type_codes = tensors["head_type"][layer_idx]  # [KVH_global]
    window = tensors["window"][layer_idx]
    sinks = tensors["sink_size"][layer_idx]

    if kv_head_count is not None:
        end = kv_head_start + kv_head_count
        type_codes = type_codes[kv_head_start:end]
        window = window[kv_head_start:end]
        sinks = sinks[kv_head_start:end]

    sink_max = int(sinks.max().item()) if sinks.numel() else 0
    if sink_max < 0:
        sink_max = 0
    if sink_max > 0:
        col = torch.arange(sink_max, device=device).unsqueeze(0)
        sink_keep = col < sinks.unsqueeze(1)
    else:
        sink_keep = torch.empty((sinks.shape[0], 0), dtype=torch.bool, device=device)
    return LayerMaskPlan(
        type_codes=type_codes,
        window=window,
        sink_keep=sink_keep,
        sink_max=sink_max,
    )


def pad_per_head_sinks(
    k_first_offline: torch.Tensor,
    v_first_offline: torch.Tensor,
    plan: LayerMaskPlan,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Pad/crop the first-segment K/V to ``plan.sink_max`` along time axis.

    Heads with smaller sinks have the trailing slots masked via
    ``plan.sink_keep`` at attention time.
    """
    S_max = plan.sink_max
    if S_max == 0:
        return k_first_offline[:, :, :0, :], v_first_offline[:, :, :0, :]
    L1 = k_first_offline.shape[2]
    if L1 >= S_max:
        return (
            k_first_offline[:, :, :S_max, :].contiguous(),
            v_first_offline[:, :, :S_max, :].contiguous(),
        )
    pad_k = k_first_offline.new_zeros(
        k_first_offline.shape[0],
        k_first_offline.shape[1],
        S_max - L1,
        k_first_offline.shape[3],
    )
    pad_v = v_first_offline.new_zeros(
        v_first_offline.shape[0],
        v_first_offline.shape[1],
        S_max - L1,
        v_first_offline.shape[3],
    )
    return (
        torch.cat([k_first_offline, pad_k], dim=2),
        torch.cat([v_first_offline, pad_v], dim=2),
    )


def group_heads_by_type(plan: LayerMaskPlan) -> Dict[int, torch.Tensor]:
    """Return ``{type_code: kv_head_indices}`` for one layer."""
    type_codes = plan.type_codes
    out: Dict[int, torch.Tensor] = {}
    for code in (
        HeadClassConfig.TYPE_LOCAL,
        HeadClassConfig.TYPE_GLOBAL,
        HeadClassConfig.TYPE_RETRIEVAL,
        HeadClassConfig.TYPE_DENSE,
    ):
        mask = type_codes == code
        if bool(mask.any()):
            out[code] = torch.nonzero(mask, as_tuple=False).flatten()
    return out


def q_head_indices(kv_head_idx: torch.Tensor, num_q_per_kv: int) -> torch.Tensor:
    """Expand KV head indices to corresponding Q head index range.

    E.g. with ``num_q_per_kv=4`` and ``kv_head_idx=[0, 2]``
    -> ``[0,1,2,3, 8,9,10,11]``.
    """
    base = kv_head_idx.unsqueeze(-1) * num_q_per_kv
    offsets = torch.arange(num_q_per_kv, device=kv_head_idx.device).unsqueeze(0)
    return (base + offsets).flatten()
