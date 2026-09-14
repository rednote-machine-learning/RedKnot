"""Torch reference operators for RedKnot; importing this module needs no Torch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _head_ids(values: Sequence[int], total: int, name: str) -> tuple[int, ...]:
    ids = tuple(values)
    if any(type(value) is not int or not 0 <= value < total for value in ids):
        raise ValueError(f"{name} must contain integer head IDs in [0, {total})")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate heads")
    return ids


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _positions(positions: torch.Tensor, length: int, device, name: str):
    import torch

    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if positions.ndim != 1 or positions.numel() != length:
        raise ValueError(f"{name} must have shape [{length}]")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must have int32 or int64 dtype")
    if bool((positions < 0).any()):
        raise ValueError(f"{name} must be nonnegative")
    return positions.to(device=device)


def _floating_tensor(value, rank: int, name: str):
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != rank or value.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError(f"{name} must be a rank-{rank} floating-point tensor")
    return value


def relocate_rope(
    keys: torch.Tensor,
    source_positions: torch.Tensor,
    target_positions: torch.Tensor,
    rotary_dim: int,
    theta: float,
    interleaved: bool = False,
    position_scale: float = 1.0,
) -> torch.Tensor:
    """Move static-RoPE keys to new logical positions, without changing magnitude.

    Args:
        keys: Already-rotated keys shaped [tokens, heads, head_dim].
        source_positions: Original nonnegative integer positions, [tokens].
        target_positions: Destination nonnegative integer positions, [tokens].
        rotary_dim: Even rotary prefix width; the remaining dimensions pass through.
        theta: Fixed RoPE base used to produce the cached keys.
        interleaved: Whether rotary pairs are adjacent, rather than half-split.
        position_scale: Constant divisor applied to positions (linear scaling).

    Returns:
        A new tensor applying R(target) R(source)^-1 to the rotary prefix only.
        The caller must reject dynamic, multimodal or other nonstandard RoPE;
        this operator neither infers nor approximates those frequency schedules.
    """
    import torch

    _floating_tensor(keys, 3, "keys")
    _positive_int(rotary_dim, "rotary_dim")
    if rotary_dim % 2 or rotary_dim > keys.shape[-1]:
        raise ValueError("rotary_dim must be even and no larger than head_dim")
    if keys.shape[1] <= 0:
        raise ValueError("keys must have at least one head")
    if type(interleaved) is not bool:
        raise ValueError("interleaved must be a boolean")
    theta = _positive_real(theta, "theta")
    position_scale = _positive_real(position_scale, "position_scale")
    source = _positions(
        source_positions, keys.shape[0], keys.device, "source_positions"
    )
    target = _positions(
        target_positions, keys.shape[0], keys.device, "target_positions"
    )
    compute_dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    inv_freq = theta ** (
        -torch.arange(0, rotary_dim, 2, device=keys.device, dtype=compute_dtype)
        / rotary_dim
    )
    # Compute the integer difference before float conversion at long positions.
    delta = (target.to(torch.int64) - source.to(torch.int64)).to(compute_dtype)
    phase = (delta / position_scale).unsqueeze(-1) * inv_freq
    cos = phase.cos().unsqueeze(1)
    sin = phase.sin().unsqueeze(1)
    rotary = keys[..., :rotary_dim].to(compute_dtype)
    if interleaved:
        left, right = rotary[..., 0::2], rotary[..., 1::2]
        rotated = torch.stack((left * cos - right * sin, right * cos + left * sin), -1)
        rotated = rotated.flatten(-2)
    else:
        left, right = rotary.chunk(2, dim=-1)
        rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
    result = keys.clone()
    result[..., :rotary_dim] = rotated.to(keys.dtype)
    return result


def head_partition(
    num_query_heads: int,
    num_kv_heads: int,
    local_kv_heads: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Expand KV-head policy to contiguous GQA groups, preserving local order."""
    _positive_int(num_query_heads, "num_query_heads")
    _positive_int(num_kv_heads, "num_kv_heads")
    if num_query_heads % num_kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    local = _head_ids(local_kv_heads, num_kv_heads, "local_kv_heads")
    local_set = set(local)
    global_kv = tuple(head for head in range(num_kv_heads) if head not in local_set)
    group_size = num_query_heads // num_kv_heads

    def expand(heads):
        return tuple(
            head * group_size + offset for head in heads for offset in range(group_size)
        )

    return expand(local), expand(global_kv), global_kv


def merge_heads(
    local_output: torch.Tensor,
    global_output: torch.Tensor,
    local_ids: Sequence[int],
    global_ids: Sequence[int],
    total_heads: int,
) -> torch.Tensor:
    """Restore independent head outputs [T, H_subset, D] to original head order.

    Local/global heads are disjoint output channels, not two softmax partitions.
    Consequently their outputs are scattered, never LSE-weighted together.
    """
    import torch

    _positive_int(total_heads, "total_heads")
    _floating_tensor(local_output, 3, "local_output")
    _floating_tensor(global_output, 3, "global_output")
    local = _head_ids(local_ids, total_heads, "local_ids")
    global_ = _head_ids(global_ids, total_heads, "global_ids")
    overlap = set(local) & set(global_)
    complete = set(local) | set(global_) == set(range(total_heads))
    if overlap or not complete:
        raise ValueError("local/global head IDs must partition every head exactly once")
    if local_output.shape[1] != len(local) or global_output.shape[1] != len(global_):
        raise ValueError("output head axes must match their head IDs")
    if (
        local_output.shape[0] != global_output.shape[0]
        or local_output.shape[2] != global_output.shape[2]
        or local_output.dtype != global_output.dtype
        or local_output.device != global_output.device
    ):
        raise ValueError(
            "head outputs must share token/dimension sizes, dtype and device"
        )
    result = local_output.new_empty(
        (local_output.shape[0], total_heads, local_output.shape[2])
    )
    result.index_copy_(
        1, torch.tensor(local, device=result.device, dtype=torch.long), local_output
    )
    result.index_copy_(
        1, torch.tensor(global_, device=result.device, dtype=torch.long), global_output
    )
    return result


def causal_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    scale: float | None = None,
    query_block_size: int = 64,
    key_block_size: int = 256,
) -> torch.Tensor:
    """Compute GQA causal attention with bounded score-block memory.

    Q/K/V use [tokens, heads, dimensions]. Logical positions determine visibility:
    a key is visible exactly when key_position <= query_position. Position order
    need not match storage order. All-masked queries and empty KV produce zeros.
    Float16/BFloat16 accumulation uses FP32; float64 inputs retain FP64.
    """
    import torch

    for value, name in ((q, "q"), (k, "k"), (v, "v")):
        _floating_tensor(value, 3, name)
    _positive_int(query_block_size, "query_block_size")
    _positive_int(key_block_size, "key_block_size")
    if q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if q.shape[2] <= 0 or v.shape[2] <= 0:
        raise ValueError("head dimensions must be positive")
    if q.shape[2] != k.shape[2] or k.shape[:2] != v.shape[:2]:
        raise ValueError("Q/K dimensions and K/V token/head axes must agree")
    if not (q.dtype == k.dtype == v.dtype and q.device == k.device == v.device):
        raise ValueError("Q, K and V must have identical dtype and device")
    qp = _positions(query_positions, q.shape[0], q.device, "query_positions")
    kp = _positions(key_positions, k.shape[0], q.device, "key_positions")
    scale = q.shape[-1] ** -0.5 if scale is None else _positive_real(scale, "scale")
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    groups = q.shape[1] // k.shape[1]
    result = q.new_zeros((q.shape[0], q.shape[1], v.shape[-1]))
    for q_start in range(0, q.shape[0], query_block_size):
        q_end = min(q_start + query_block_size, q.shape[0])
        queries = q[q_start:q_end].transpose(0, 1).to(compute_dtype)
        running_max = torch.full(
            queries.shape[:2], -torch.inf, device=q.device, dtype=compute_dtype
        )
        denominator = torch.zeros_like(running_max)
        numerator = torch.zeros(
            (*queries.shape[:2], v.shape[-1]), device=q.device, dtype=compute_dtype
        )
        for k_start in range(0, k.shape[0], key_block_size):
            k_end = min(k_start + key_block_size, k.shape[0])
            keys = k[k_start:k_end].transpose(0, 1).to(compute_dtype)
            values = v[k_start:k_end].transpose(0, 1).to(compute_dtype)
            keys = keys.repeat_interleave(groups, dim=0)
            values = values.repeat_interleave(groups, dim=0)
            scores = torch.matmul(queries, keys.transpose(-1, -2)) * scale
            visible = kp[k_start:k_end].unsqueeze(0) <= qp[q_start:q_end].unsqueeze(1)
            scores = scores.masked_fill(~visible.unsqueeze(0), -torch.inf)
            block_max = scores.amax(dim=-1)
            next_max = torch.maximum(running_max, block_max)
            finite_max = torch.where(torch.isfinite(next_max), next_max, 0.0)
            old_weight = torch.exp(running_max - finite_max)
            weights = torch.exp(scores - finite_max.unsqueeze(-1))
            numerator = numerator * old_weight.unsqueeze(-1) + torch.matmul(
                weights, values
            )
            denominator = denominator * old_weight + weights.sum(dim=-1)
            running_max = next_max
        divisor = denominator.clamp_min(torch.finfo(compute_dtype).tiny).unsqueeze(-1)
        output = numerator / divisor
        result[q_start:q_end] = output.transpose(0, 1).to(q.dtype)
    return result
