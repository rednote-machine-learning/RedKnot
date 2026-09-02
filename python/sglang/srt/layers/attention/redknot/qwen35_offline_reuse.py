# Copyright 2024-2026 SGLang RedKnot Integration.
"""Correctness helpers for Qwen3.5 hybrid offline-prefix reuse.

Qwen3.5 interleaves full-attention layers with Gated DeltaNet (GDN) layers.
Reusing only the full-attention KV is therefore incomplete: every GDN layer
also needs its causal-convolution window and recurrent state.  This module
implements the conservative production contract used by the SGLang adapter:

* one offline segment represents one ordered, immutable document bundle;
* the bundle is a logical prefix of the online query;
* its GDN state is restored once when a fresh request starts;
* full-attention RoPE positions are shifted by the bundle token length.

Independent recurrent segments cannot be concatenated by copying their final
states because GDN state is order-dependent.  Such plans are rejected instead
of silently producing an invalid result.  The paper's more general multi-chunk
adapter requires checkpoint/replay or recomputation and is intentionally kept
outside this correctness-first path.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch

from sglang.srt.layers.attention.redknot.offline_cache import (
    OfflineKVCache,
    OfflineRecurrentState,
    get_global_offline_cache,
)


RKBUILD_PREFIX = "__RKBUILD__:"


def capture_qwen35_recurrent_state(
    req_to_token_pool, req_pool_idx: int
) -> Optional[OfflineRecurrentState]:
    """Clone all GDN conv/recurrent layers for one live request slot.

    Returns ``None`` for a non-hybrid request pool so the generic RedKnot KV
    snapshot path remains usable by dense-attention models.
    """

    mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
    if mamba_pool is None or not hasattr(mamba_pool, "mamba_cache"):
        return None

    mapping = getattr(req_to_token_pool, "req_index_to_mamba_index_mapping", None)
    if mapping is not None:
        mamba_idx = int(mapping[req_pool_idx].item())
    else:
        device = getattr(mamba_pool.mamba_cache.temporal, "device", torch.device("cpu"))
        req_idx = torch.tensor([req_pool_idx], dtype=torch.long, device=device)
        mamba_idx = int(req_to_token_pool.get_mamba_indices(req_idx)[0].item())

    cache = mamba_pool.mamba_cache
    conv = [state[:, mamba_idx].contiguous().clone() for state in cache.conv]
    temporal = cache.temporal[:, mamba_idx].contiguous().clone()
    return OfflineRecurrentState(conv=conv, temporal=temporal)


def _normalise_segment_ids(ids) -> List[str]:
    if ids is None:
        return []
    if isinstance(ids, str):
        return [ids]
    return [sid for sid in ids if sid is not None]


def _drop_cleared_slots(
    loaded_slots: Dict[int, str], cleared_slots: Optional[Iterable[int]]
) -> None:
    if cleared_slots is None:
        return
    if isinstance(cleared_slots, torch.Tensor):
        cleared_slots = cleared_slots.detach().to("cpu").tolist()
    for slot in cleared_slots:
        loaded_slots.pop(int(slot), None)


def _restore_recurrent_state(req_to_token_pool, mamba_idx: int, state) -> None:
    live = req_to_token_pool.mamba_pool.mamba_cache
    if len(live.conv) != len(state.conv):
        raise RuntimeError(
            "RedKnot Qwen3.5 recurrent-state mismatch: "
            f"snapshot has {len(state.conv)} conv buffers, runtime has {len(live.conv)}."
        )

    for buffer_idx, (dst, src) in enumerate(zip(live.conv, state.conv)):
        expected = dst[:, mamba_idx].shape
        if src.shape != expected:
            raise RuntimeError(
                "RedKnot Qwen3.5 conv-state shape mismatch at buffer "
                f"{buffer_idx}: snapshot={tuple(src.shape)}, runtime={tuple(expected)}."
            )
        dst[:, mamba_idx].copy_(src.to(device=dst.device, dtype=dst.dtype))

    expected = live.temporal[:, mamba_idx].shape
    if state.temporal.shape != expected:
        raise RuntimeError(
            "RedKnot Qwen3.5 recurrent-state shape mismatch: "
            f"snapshot={tuple(state.temporal.shape)}, runtime={tuple(expected)}."
        )
    live.temporal[:, mamba_idx].copy_(
        state.temporal.to(device=live.temporal.device, dtype=live.temporal.dtype)
    )


def prepare_qwen35_offline_reuse(
    forward_batch,
    req_to_token_pool,
    *,
    loaded_slots: Dict[int, str],
    cleared_slots: Optional[Sequence[int]] = None,
    offline_cache: Optional[OfflineKVCache] = None,
) -> None:
    """Validate a Qwen3.5 offline plan and restore fresh request states.

    The function attaches two tensors to ``forward_batch``:

    ``redknot_position_offsets``
        One logical-prefix length per request.  Full-attention layers add it
        to their packed positions before Qwen's native partial/mRoPE code.
    ``redknot_recurrent_loaded_mask``
        Requests whose recurrent state was restored in this step.  GDN uses
        it to mark the convolution as having an initial state.
    """

    _drop_cleared_slots(loaded_slots, cleared_slots)
    plan = getattr(forward_batch, "redknot_offline_segments", None)
    if plan is None:
        forward_batch.redknot_position_offsets = None
        forward_batch.redknot_recurrent_loaded_mask = None
        return
    batch_size = int(forward_batch.req_pool_indices.shape[0])
    device = forward_batch.req_pool_indices.device
    offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
    restored = torch.zeros(batch_size, dtype=torch.bool, device=device)
    forward_batch.redknot_position_offsets = offsets
    forward_batch.redknot_recurrent_loaded_mask = restored

    if len(plan) != batch_size:
        raise RuntimeError(
            "RedKnot Qwen3.5 offline plan must contain one entry per request: "
            f"plan={len(plan)}, batch={batch_size}."
        )

    normalised = [_normalise_segment_ids(ids) for ids in plan]
    active = [ids for ids in normalised if ids and not ids[0].startswith(RKBUILD_PREFIX)]
    if not active:
        return

    # The eager decode/offline-state path is intentionally fail-closed until
    # graph capture and Radix cache carry the virtual-prefix metadata.
    try:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
    except Exception:
        server_args = None
    if server_args is not None:
        required_flags = {
            "disable_cuda_graph": "--disable-cuda-graph",
            "disable_piecewise_cuda_graph": "--disable-piecewise-cuda-graph",
            "disable_radix_cache": "--disable-radix-cache",
        }
        missing = [
            cli
            for attr, cli in required_flags.items()
            if not bool(getattr(server_args, attr, False))
        ]
        if missing:
            raise RuntimeError(
                "RedKnot Qwen3.5 fixed-prefix reuse currently requires "
                + ", ".join(missing)
                + "."
            )
        if getattr(server_args, "enable_multimodal", None) is not False:
            raise RuntimeError(
                "RedKnot Qwen3.5 fixed-prefix reuse is currently text-only; "
                "launch through server/launch_redknot_qwen35_text.py."
            )
        if int(getattr(server_args, "pp_size", 1)) != 1:
            raise RuntimeError(
                "RedKnot Qwen3.5 fixed-prefix reuse does not yet support "
                "pipeline parallelism."
            )
    try:
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        if is_dp_attention_enabled():
            raise RuntimeError(
                "RedKnot Qwen3.5 fixed-prefix reuse does not yet support "
                "DP-attention."
            )
    except ImportError:
        pass

    mode = forward_batch.forward_mode
    if mode.is_target_verify() or mode.is_draft_extend_v2():
        raise RuntimeError(
            "RedKnot Qwen3.5 offline-prefix reuse does not yet support speculative "
            "decode; launch without speculative decoding."
        )

    cache = offline_cache or get_global_offline_cache()
    mamba_indices = req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
    is_extend = mode.is_extend(include_draft_extend_v2=True)
    prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)

    for req_idx, ids in enumerate(normalised):
        if not ids or ids[0].startswith(RKBUILD_PREFIX):
            continue
        if len(ids) != 1:
            raise RuntimeError(
                "RedKnot Qwen3.5 accepts exactly one ordered offline document "
                "bundle per request. Independent GDN states cannot be composed; "
                "build the ordered documents as one bundle instead."
            )

        sid = ids[0]
        segment = cache.to_device(sid, device)
        if segment is None:
            raise RuntimeError(f"RedKnot Qwen3.5 offline segment not found: {sid}")
        if segment.recurrent_state is None:
            raise RuntimeError(
                "RedKnot Qwen3.5 segment has full-attention KV but no GDN state. "
                "Rebuild it with the Qwen3.5 hybrid adapter before querying."
            )

        offsets[req_idx] = int(segment.doc_len)
        mamba_idx = int(mamba_indices[req_idx].item())
        already_loaded = loaded_slots.get(mamba_idx) == sid

        if is_extend and not already_loaded:
            prefix_len = int(prefix_lens[req_idx].item()) if prefix_lens is not None else 0
            if prefix_len != 0:
                raise RuntimeError(
                    "RedKnot Qwen3.5 cannot combine an offline GDN prefix with a "
                    "Radix/live prefix hit. Launch with --disable-radix-cache; "
                    "continuing chunked prefill is supported only after the same "
                    "offline bundle has already been loaded into this request slot."
                )
            _restore_recurrent_state(
                req_to_token_pool, mamba_idx, segment.recurrent_state
            )
            loaded_slots[mamba_idx] = sid
            restored[req_idx] = True
        elif not is_extend and not already_loaded:
            raise RuntimeError(
                "RedKnot Qwen3.5 reached decode without its offline recurrent "
                f"state (segment={sid[:12]}..., mamba_slot={mamba_idx}, "
                f"slot currently holds {loaded_slots.get(mamba_idx)!r}). Disable "
                "CUDA graphs, ensure query prefill uses the same "
                "redknot_offline_segments plan, and make sure the caller does not "
                "drop the slot from loaded_slots between prefill and decode "
                "(deferred mamba clears only execute on extend batches)."
            )

    logical_seq_lens = forward_batch.seq_lens.to(
        device=device, dtype=torch.long
    ) + offsets
    forward_batch.redknot_logical_seq_lens = logical_seq_lens
    context_length = (
        getattr(server_args, "context_length", None)
        if server_args is not None
        else None
    )
    if context_length is not None and bool(torch.any(logical_seq_lens > context_length)):
        largest = int(logical_seq_lens.max().item())
        raise RuntimeError(
            "RedKnot Qwen3.5 logical sequence exceeds the configured context "
            f"length: logical={largest}, limit={context_length}."
        )


def apply_qwen35_position_offsets(
    positions: torch.Tensor, forward_batch
) -> torch.Tensor:
    """Add each request's offline-prefix length to packed Qwen positions.

    Supports normal one-dimensional text positions and Qwen's multi-axis
    position tensor by broadcasting along every dimension except the packed
    token dimension.
    """

    offsets = getattr(forward_batch, "redknot_position_offsets", None)
    if offsets is None or positions is None:
        return positions

    batch_size = int(offsets.numel())
    mode = forward_batch.forward_mode
    if mode.is_extend(include_draft_extend_v2=True):
        lengths = forward_batch.extend_seq_lens.to(
            device=offsets.device, dtype=torch.long
        )
        packed_offsets = torch.repeat_interleave(offsets, lengths)
    else:
        token_count = int(positions.shape[-1])
        if token_count % batch_size != 0:
            raise RuntimeError(
                "RedKnot Qwen3.5 cannot map packed decode positions to requests: "
                f"tokens={token_count}, batch={batch_size}."
            )
        packed_offsets = torch.repeat_interleave(offsets, token_count // batch_size)

    if packed_offsets.numel() != positions.shape[-1]:
        raise RuntimeError(
            "RedKnot Qwen3.5 packed position length mismatch: "
            f"offsets={packed_offsets.numel()}, positions={positions.shape[-1]}."
        )
    view_shape = (1,) * (positions.ndim - 1) + (packed_offsets.numel(),)
    return positions + packed_offsets.to(
        device=positions.device, dtype=positions.dtype
    ).view(view_shape)
