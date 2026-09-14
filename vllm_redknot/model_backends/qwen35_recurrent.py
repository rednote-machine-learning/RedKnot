# REDKNOT-MODEL: explicit hybrid-state buffer/capture/restore adapter boundary.
# Copyright 2024-2026 SGLang RedKnot Integration.
"""Portable tensor operations for the strict Qwen3.5 offline-prefix contract.

Native vLLM must supply correctly shaped recurrent buffers, physical slot ids,
cache lookups, and feature configuration. No native request pool or model code
is copied. These helpers are not automatically installed into any engine.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass

import torch

from vllm_redknot.core.offline_cache import OfflineRecurrentState

from .qwen35_reuse_contract import (
    Qwen35PrefixReusePlan,
    Qwen35ReuseConfig,
    plan_qwen35_prefix_reuse,
)


@dataclass(frozen=True)
class Qwen35RecurrentBuffers:
    """Explicit per-rank tensors with source layout ``[layer, slot, ...]``."""

    conv: Sequence[torch.Tensor]
    temporal: torch.Tensor


def capture_qwen35_recurrent_state(
    buffers: Qwen35RecurrentBuffers, state_slot: int
) -> OfflineRecurrentState:
    """Clone the conv window and GDN state; the slot mapping is caller-owned."""
    _validate_slot(buffers, state_slot)
    conv = [state[:, state_slot].contiguous().clone() for state in buffers.conv]
    temporal = buffers.temporal[:, state_slot].contiguous().clone()
    return OfflineRecurrentState(conv=conv, temporal=temporal)


def _validate_slot(buffers: Qwen35RecurrentBuffers, slot: int) -> None:
    if type(slot) is not int or slot < 0:
        raise ValueError("recurrent slot must be a non-negative built-in integer")
    for tensor in tuple(buffers.conv) + (buffers.temporal,):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            raise ValueError("recurrent buffers require [layer, slot, ...] tensors")
        if slot >= tensor.shape[1]:
            raise ValueError("recurrent slot is outside a state buffer")


def _validate_recurrent_state(
    buffers: Qwen35RecurrentBuffers, slot: int, state: OfflineRecurrentState
) -> None:
    _validate_slot(buffers, slot)
    if not isinstance(state, OfflineRecurrentState):
        raise ValueError("offline segment has no compatible conv/GDN state")
    if len(buffers.conv) != len(state.conv):
        raise ValueError("offline and live conv buffer counts differ")
    for dst, src in zip(buffers.conv, state.conv):
        if not isinstance(src, torch.Tensor) or src.shape != dst[:, slot].shape:
            raise ValueError("offline and live conv state shapes differ")
    if (
        not isinstance(state.temporal, torch.Tensor)
        or state.temporal.shape != buffers.temporal[:, slot].shape
    ):
        raise ValueError("offline and live GDN state shapes differ")


@torch.no_grad()
def prepare_qwen35_offline_reuse(
    *,
    buffers: Qwen35RecurrentBuffers,
    segments: Sequence[Sequence[str] | str | None],
    state_slots: Sequence[int],
    seq_lens: Sequence[int],
    prefix_lens: Sequence[int],
    loaded_slots: MutableMapping[int, str],
    cleared_slots: Sequence[int] = (),
    get_segment: Callable[[str], object | None],
    is_prefill: bool,
    config: Qwen35ReuseConfig,
) -> Qwen35PrefixReusePlan:
    """Restore a complete hybrid bundle once, after batch-wide preflight.

    Unlike the source wrapper's pool/global lookups, cache lookup and state-slot
    identity are explicit. All geometry, context, feature, and capacity checks
    run before the first buffer write. A device-copy failure invalidates affected
    loaded-slot receipts and raises; it does not claim transactional CUDA rollback.
    The owning engine must abort/reinitialize that request on such a failure.
    """
    for slot in cleared_slots:
        if type(slot) is not int or slot < 0:
            raise ValueError("cleared slots must be non-negative integers")
    # Clearing is a native lifecycle event, not part of a speculative restore.
    # A later preflight failure must not leave an old request's receipt usable.
    for slot in cleared_slots:
        loaded_slots.pop(slot, None)
    if not isinstance(config, Qwen35ReuseConfig):
        raise TypeError("an explicit Qwen35ReuseConfig is required")
    for raw in segments:
        ids = [] if raw is None else ([raw] if isinstance(raw, str) else list(raw))
        if any(
            isinstance(sid, str) and not sid.startswith("__RKBUILD__:") for sid in ids
        ):
            config.validate()
            break
    candidate_loaded = dict(loaded_slots)
    cached = {}
    lengths = {}
    for raw in segments:
        ids = [] if raw is None else ([raw] if isinstance(raw, str) else list(raw))
        for sid in ids:
            if not isinstance(sid, str) or sid.startswith("__RKBUILD__:"):
                continue
            if sid not in cached:
                segment = get_segment(sid)
                if segment is None:
                    raise ValueError(f"offline segment not found: {sid}")
                if getattr(segment, "recurrent_state", None) is None:
                    raise ValueError(
                        "offline full-attention KV is missing conv/GDN state"
                    )
                cached[sid] = segment
                lengths[sid] = getattr(segment, "doc_len", None)
    plan = plan_qwen35_prefix_reuse(
        segments=segments,
        state_slots=state_slots,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        document_lengths=lengths,
        loaded_slots=candidate_loaded,
        is_prefill=is_prefill,
        config=config,
    )
    for row in plan.restore_rows:
        _validate_recurrent_state(
            buffers,
            plan.state_slots[row],
            cached[plan.segment_ids[row]].recurrent_state,
        )
    try:
        for row in plan.restore_rows:
            slot = plan.state_slots[row]
            state = cached[plan.segment_ids[row]].recurrent_state
            for dst, src in zip(buffers.conv, state.conv):
                dst[:, slot].copy_(src.to(device=dst.device, dtype=dst.dtype))
            buffers.temporal[:, slot].copy_(
                state.temporal.to(
                    device=buffers.temporal.device, dtype=buffers.temporal.dtype
                )
            )
    except Exception:
        for slot in cleared_slots:
            loaded_slots.pop(slot, None)
        for row in plan.restore_rows:
            loaded_slots.pop(plan.state_slots[row], None)
        raise
    for slot in cleared_slots:
        loaded_slots.pop(slot, None)
    for row in plan.restore_rows:
        loaded_slots[plan.state_slots[row]] = plan.segment_ids[row]
    return plan


def apply_qwen35_position_offsets(
    positions: torch.Tensor,
    *,
    position_offsets: Sequence[int],
    is_prefill: bool,
    extend_seq_lens: Sequence[int] | None = None,
) -> torch.Tensor:
    """Apply bundle length to packed text or multi-axis native RoPE positions."""
    if positions is None:
        return positions
    if positions.ndim < 1:
        raise ValueError("positions need a packed token dimension")
    if type(is_prefill) is not bool:
        raise TypeError("is_prefill must be a boolean")
    if any(type(offset) is not int or offset < 0 for offset in position_offsets):
        raise ValueError("position offsets must be non-negative built-in integers")
    offsets = torch.tensor(position_offsets, dtype=torch.long, device=positions.device)
    if offsets.ndim != 1 or offsets.numel() == 0:
        raise ValueError("position offsets must contain one value per request")
    if bool(torch.any(offsets < 0)):
        raise ValueError("position offsets must be non-negative")
    if is_prefill:
        if extend_seq_lens is None or len(extend_seq_lens) != offsets.numel():
            raise ValueError("prefill lengths must align with request offsets")
        if any(type(length) is not int or length < 0 for length in extend_seq_lens):
            raise ValueError("prefill lengths must be non-negative built-in integers")
        lengths = torch.tensor(extend_seq_lens, dtype=torch.long, device=offsets.device)
        if bool(torch.any(lengths < 0)):
            raise ValueError("prefill lengths must be non-negative")
        packed_offsets = torch.repeat_interleave(offsets, lengths)
    else:
        token_count = int(positions.shape[-1])
        if token_count % offsets.numel() != 0:
            raise ValueError("packed decode positions do not divide across requests")
        packed_offsets = torch.repeat_interleave(
            offsets, token_count // offsets.numel()
        )
    if packed_offsets.numel() != positions.shape[-1]:
        raise ValueError("packed positions and offsets have different token counts")
    shape = (1,) * (positions.ndim - 1) + (packed_offsets.numel(),)
    return positions + packed_offsets.to(dtype=positions.dtype).view(shape)
