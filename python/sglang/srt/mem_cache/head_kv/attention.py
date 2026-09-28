"""Attention over independently allocated KV-head pages.

The pools contain *one* independently addressable KV head per page; GQA maps
query head ``h`` to KV head ``h // num_q_per_kv`` without duplicating K/V.
Page-table order need not be token order. ``key_positions`` are target causal
positions, not physical offsets or source positions before a RoPE transform.

The caller must hold a read lease on every referenced page until the operation
has completed on its CUDA stream. This module does not own page lifetimes.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Optional, Sequence, Union

import torch


HeadPolicy = Optional[Union[torch.Tensor, Sequence[int]]]
_INDEX_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _head_policy(value: HeadPolicy, heads: int, device, name: str) -> torch.Tensor:
    if value is None:
        return torch.zeros(heads, dtype=torch.int64, device=device)
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise ValueError(f"{name} must be on the query device")
        if value.dtype not in _INDEX_DTYPES:
            raise TypeError(f"{name} must have int32 or int64 dtype")
        if value.shape != (heads,):
            raise ValueError(f"{name} must have shape [{heads}]")
        return value
    if not isinstance(value, (list, tuple)) or len(value) != heads:
        raise ValueError(f"{name} must contain one integer per KV head")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in value):
        raise TypeError(f"{name} must contain integers")
    return torch.tensor(value, dtype=torch.int64, device=device)


def validate_paged_attention_inputs(
    query,
    k_pool,
    v_pool,
    page_slots,
    page_lengths,
    key_positions,
    *,
    query_positions,
    num_q_per_kv=1,
    scale=None,
    causal=True,
    windows=None,
    sinks=None,
):
    """Validate descriptors and return normalized positions/policies/scale.

    CUDA value checks intentionally synchronize once: out-of-bounds page slots
    must fail before the kernel can access memory. A scheduler may validate an
    immutable descriptor once and call the internal launch function while its
    read lease remains held. Shape validation alone is insufficient for safety.
    """
    named_tensors = {
        "query": query,
        "k_pool": k_pool,
        "v_pool": v_pool,
        "page_slots": page_slots,
        "page_lengths": page_lengths,
        "key_positions": key_positions,
        "query_positions": query_positions,
    }
    for name, value in named_tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{name} must be a strided tensor")
        if value.device != query.device:
            raise ValueError(f"{name} must be on the query device")
    if query.device.type not in ("cpu", "cuda"):
        raise ValueError("paged attention supports CPU and CUDA tensors")
    if query.ndim != 3 or k_pool.ndim != 3 or v_pool.shape != k_pool.shape:
        raise ValueError("query must be [Hq,Nq,D]; pools must both be [capacity,B,D]")
    hq, nq, dim = query.shape
    capacity, page_size, pool_dim = k_pool.shape
    if hq <= 0 or dim <= 0 or page_size <= 0 or dim != pool_dim:
        raise ValueError("head count, page size and matching head dimension must be positive")
    if query.dtype not in _FLOAT_DTYPES:
        raise TypeError("query must have a supported floating dtype")
    if k_pool.dtype != query.dtype or v_pool.dtype != query.dtype:
        raise TypeError("query, K and V must have identical dtypes")
    if any(x.requires_grad for x in (query, k_pool, v_pool)):
        raise ValueError("paged attention is an inference operation; gradients are unsupported")
    if isinstance(num_q_per_kv, bool) or not isinstance(num_q_per_kv, Integral):
        raise TypeError("num_q_per_kv must be a positive integer")
    if num_q_per_kv <= 0 or hq % num_q_per_kv:
        raise ValueError("query head count must be divisible by num_q_per_kv")
    hkv = hq // num_q_per_kv
    if page_slots.ndim != 2 or page_slots.shape[0] != hkv:
        raise ValueError("page_slots must have shape [Hkv,max_pages]")
    if page_lengths.shape != page_slots.shape:
        raise ValueError("page_lengths must match page_slots")
    if key_positions.shape != (*page_slots.shape, page_size):
        raise ValueError("key_positions must have shape [Hkv,max_pages,B]")
    for name in ("page_slots", "page_lengths"):
        if named_tensors[name].dtype not in _INDEX_DTYPES:
            raise TypeError(f"{name} must have int32 or int64 dtype")
    if key_positions.dtype != torch.int64 or query_positions.dtype != torch.int64:
        raise TypeError("key_positions and query_positions must have int64 dtype")
    if query_positions.shape == (nq,):
        normalized_qpos = query_positions.unsqueeze(0).expand(hq, nq)
    elif query_positions.shape == (hq, nq):
        normalized_qpos = query_positions
    else:
        raise ValueError("query_positions must have shape [Nq] or [Hq,Nq]")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")
    if scale is None:
        scale = dim**-0.5
    elif isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
        raise ValueError("scale must be a finite real number")
    win = _head_policy(windows, hkv, query.device, "windows")
    sink = _head_policy(sinks, hkv, query.device, "sinks")

    # The flags are downloaded in one synchronization, not one per check/page.
    valid_offsets = torch.arange(page_size, device=query.device)[None, None, :] < page_lengths[..., None]
    flags = torch.stack(
        [
            ((page_slots < -1) | (page_slots >= capacity)).any(),
            ((page_lengths < 0) | (page_lengths > page_size)).any(),
            ((page_slots == -1) & (page_lengths != 0)).any(),
            ((key_positions < 0) & valid_offsets).any(),
            (normalized_qpos < 0).any(),
            (sink < 0).any(),
        ]
    ).cpu().tolist()
    errors = (
        "page_slots contains an invalid physical slot",
        "page_lengths must be between zero and page size",
        "an absent page (-1 slot) must have zero length",
        "active key positions must be nonnegative target positions",
        "query positions must be nonnegative",
        "sink counts must be nonnegative",
    )
    for invalid, message in zip(flags, errors):
        if invalid:
            raise ValueError(message)
    return normalized_qpos, win, sink, float(scale)


def _torch_paged_attention(
    query,
    k_pool,
    v_pool,
    page_slots,
    page_lengths,
    key_positions,
    query_positions,
    windows,
    sinks,
    num_q_per_kv,
    scale,
    causal,
):
    """Pagewise online-softmax oracle; never materializes the full KV sequence."""
    hq, nq, dim = query.shape
    acc_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    output = torch.zeros_like(query)
    if nq == 0:
        return output
    for q_head in range(hq):
        kv_head = q_head // num_q_per_kv
        q = query[q_head].to(acc_dtype)
        qpos = query_positions[q_head]
        running_max = torch.full((nq,), -torch.inf, dtype=acc_dtype, device=query.device)
        denominator = torch.zeros(nq, dtype=acc_dtype, device=query.device)
        numerator = torch.zeros((nq, dim), dtype=acc_dtype, device=query.device)
        window, sink = int(windows[kv_head]), int(sinks[kv_head])
        for page in range(page_slots.shape[1]):
            length = int(page_lengths[kv_head, page])
            if not length:
                continue
            slot = int(page_slots[kv_head, page])
            k = k_pool[slot, :length].to(acc_dtype)
            v = v_pool[slot, :length].to(acc_dtype)
            pos = key_positions[kv_head, page, :length]
            visible = torch.ones((nq, length), device=query.device, dtype=torch.bool)
            if causal:
                visible &= pos[None, :] <= qpos[:, None]
            if window > 0:
                visible &= (pos[None, :] >= qpos[:, None] - window + 1) | (pos[None, :] < sink)
            scores = (q @ k.transpose(0, 1)) * scale
            scores.masked_fill_(~visible, -torch.inf)
            new_max = torch.maximum(running_max, scores.amax(dim=1))
            # All-masked rows must not evaluate (-inf)-(-inf) in the recurrence.
            safe_max = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
            old_weight = torch.exp(running_max - safe_max)
            weights = torch.exp(scores - safe_max[:, None])
            numerator = numerator * old_weight[:, None] + weights @ v
            denominator = denominator * old_weight + weights.sum(dim=1)
            running_max = new_max
        output[q_head] = (numerator / denominator.clamp_min(torch.finfo(acc_dtype).tiny)[:, None]).to(query.dtype)
    return output


def paged_attention(
    query,
    k_pool,
    v_pool,
    page_slots,
    page_lengths,
    key_positions,
    *,
    query_positions,
    num_q_per_kv=1,
    scale=None,
    causal=True,
    windows=None,
    sinks=None,
    backend="auto",
):
    """Compute attention directly over head pages, returning ``[Hq,Nq,D]``.

    ``windows[h] <= 0`` disables the window for that KV head. Otherwise the
    visible set is ``position >= query_position-window+1 OR position < sinks[h]``,
    intersected with ``position <= query_position`` when ``causal=True``.
    Consequently noncausal mode does not impose a right window bound.
    Empty and all-masked rows return zero. Duplicate logical positions are
    distinct attention entries; the manager must avoid unintended alias entries.

    ``auto`` uses the direct Triton kernel for CUDA and the pagewise torch oracle
    for CPU. Triton is imported only for CUDA execution; it is not a CPU dependency.
    No backend gathers, concatenates, or expands KV heads into a dense KV tensor.
    Pools must already use the target position/encoding basis declared by the
    manager's reuse/repair plan: this operator does not relocate RoPE keys.
    """
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    qpos, win, sink, scale = validate_paged_attention_inputs(
        query, k_pool, v_pool, page_slots, page_lengths, key_positions,
        query_positions=query_positions, num_q_per_kv=num_q_per_kv, scale=scale,
        causal=causal, windows=windows, sinks=sinks,
    )
    if backend == "triton" and query.device.type != "cuda":
        raise ValueError("Triton attention requires CUDA tensors")
    if backend == "triton" or (backend == "auto" and query.device.type == "cuda"):
        if query.dtype == torch.float64:
            raise TypeError("Triton attention supports float16, bfloat16 and float32")
        try:
            from .triton_attention import triton_paged_attention
        except ImportError as exc:
            raise RuntimeError("CUDA paged attention requires Triton; use backend='torch' for the oracle") from exc
        return triton_paged_attention(
            query, k_pool, v_pool, page_slots, page_lengths, key_positions,
            qpos, win, sink, num_q_per_kv, scale, causal,
        )
    return _torch_paged_attention(
        query, k_pool, v_pool, page_slots, page_lengths, key_positions,
        qpos, win, sink, num_q_per_kv, scale, causal,
    )
