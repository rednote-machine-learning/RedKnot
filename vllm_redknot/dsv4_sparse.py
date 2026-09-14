"""Selected-row/head DeepSeek V4 sparse MLA, without altering native KV state.

The gathered native KV serves as both K and V. Candidate indices already encode
visibility: this module adds no causal mask and never deduplicates candidates.
Importing the module does not import Torch, Triton, vLLM, or initialize CUDA.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

HEAD_DIM = 512
HEAD_TILE = 16
KV_TILE = 64


def _selection(values: Sequence[int], size: int, name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a CPU sequence of Python integers")
    result = tuple(values)
    if any(type(value) is not int or not 0 <= value < size for value in result):
        raise ValueError(f"{name} must contain integer IDs in [0, {size})")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate selections")
    return result


def sparse_workload(num_rows: int, num_heads: int, num_candidates: int) -> dict:
    """Report launched head padding, not an inferred FLOP count or speedup.

    Candidate capacity is rounded to 64. Individual programs stop at their
    clamped row length; they still evaluate invalid slots within their last tile.
    """
    for name, value in (
        ("num_rows", num_rows),
        ("num_heads", num_heads),
        ("num_candidates", num_candidates),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative Python integer")
    head_tiles = (num_heads + HEAD_TILE - 1) // HEAD_TILE
    return {
        "selected_rows": num_rows,
        "selected_heads": num_heads,
        "head_tile": HEAD_TILE,
        "candidate_tile": KV_TILE,
        "programs": num_rows * head_tiles,
        "logical_query_head_rows": num_rows * num_heads,
        "padded_heads_per_row": head_tiles * HEAD_TILE,
        "padded_query_head_rows": num_rows * head_tiles * HEAD_TILE,
        "candidate_capacity": num_candidates,
        "padded_candidate_capacity": (
            (num_candidates + KV_TILE - 1) // KV_TILE * KV_TILE
        ),
    }


def _validate(q, kv, indices, lens, sink, rows, head_ids, scale):
    import torch

    for name, value in (
        ("q", q),
        ("kv", kv),
        ("indices", indices),
        ("lens", lens),
        ("sink", sink),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if q.ndim != 3 or q.shape[2] != HEAD_DIM or q.shape[1] < 1:
        raise ValueError("q must have shape [T, H, 512] with H > 0")
    if kv.ndim != 3 or kv.shape[1:] != (1, HEAD_DIM):
        raise ValueError("kv must have shape [S, 1, 512]")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("q must have a floating-point dtype")
    if kv.dtype != q.dtype:
        raise ValueError("q and kv must have the same dtype")
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0, :]
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError("indices must have shape [T, K] or [T, 1, K]")
    if lens.ndim != 1 or lens.shape[0] != q.shape[0]:
        raise ValueError("lens must have shape [T]")
    if sink.ndim != 1 or sink.shape[0] != q.shape[1]:
        raise ValueError("sink must have shape [H] in original head order")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must have int32 or int64 dtype")
    if lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("lens must have int32 or int64 dtype")
    if sink.dtype not in (torch.float32, torch.float64):
        raise ValueError("sink must have float32 or float64 dtype")
    if any(value.device != q.device for value in (kv, indices, lens, sink)):
        raise ValueError("all input tensors must be on the same device")
    row_ids = _selection(rows, q.shape[0], "rows")
    heads = _selection(head_ids, q.shape[1], "head_ids")
    if row_ids:
        selected_lengths = lens[list(row_ids)]
        if bool(((selected_lengths < 0) | (selected_lengths > indices.shape[1])).any()):
            raise ValueError("selected rows' lens must be in [0, K]")
    if scale is None:
        scale = HEAD_DIM**-0.5
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise ValueError("scale must be a finite positive number")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")
    return indices, row_ids, heads, scale


def selected_sparse_mla_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor,
    rows: Sequence[int],
    head_ids: Sequence[int],
    scale: float | None = None,
) -> torch.Tensor:
    """Small Torch oracle, including zero-valued sink and repeated candidates.

    This intentionally slow implementation may synchronize devices. It is for
    correctness checks, never a runtime fallback. Float64 inputs compute in
    float64, otherwise in float32. Output has q's dtype and selected-order shape.
    """
    import torch

    indices, rows, heads, scale = _validate(
        q, kv, indices, lens, sink, rows, head_ids, scale
    )
    result = q.new_zeros((len(rows), len(heads), HEAD_DIM))
    if not rows or not heads:
        return result
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    head_tensor = torch.tensor(heads, device=q.device, dtype=torch.long)
    selected_sink = sink[head_tensor].to(dtype)
    positive_infinity = torch.isposinf(selected_sink)
    # The +inf sink has zero value and wins over all finite attention scores.
    finite_sink = selected_sink.masked_fill(positive_infinity, 0)
    for out_row, original_row in enumerate(rows):
        length = int(lens[original_row])
        candidate = indices[original_row, :length].long()
        candidate = candidate[(candidate >= 0) & (candidate < kv.shape[0])]
        values = kv[candidate, 0, :].to(dtype)
        scores = (q[original_row, head_tensor].to(dtype) @ values.T) * scale
        logits = torch.cat((scores, finite_sink[:, None]), dim=-1)
        all_masked = torch.isneginf(logits).all(dim=-1)
        logits[all_masked] = 0
        probabilities = logits.softmax(-1)[:, :-1]
        output = probabilities @ values
        output[all_masked | positive_infinity] = 0
        result[out_row] = output.to(q.dtype)
    return result


@lru_cache(maxsize=1)
def _kernel():
    # Triton resolves constexpr annotations against the function's globals.
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def selected_kernel(
        Q,
        KV,
        Indices,
        Lengths,
        Sink,
        Rows,
        Heads,
        Out,
        q_row: tl.constexpr,
        q_head: tl.constexpr,
        q_dim: tl.constexpr,
        kv_row: tl.constexpr,
        kv_dim: tl.constexpr,
        idx_row: tl.constexpr,
        idx_col: tl.constexpr,
        len_stride: tl.constexpr,
        sink_stride: tl.constexpr,
        S: tl.constexpr,
        K: tl.constexpr,
        H: tl.constexpr,
        scale: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
        D: tl.constexpr,
    ):
        selected_row = tl.program_id(0).to(tl.int64)
        selected_heads = tl.program_id(1) * BH + tl.arange(0, BH)
        original_row = tl.load(Rows + selected_row).to(tl.int64)
        original_heads = tl.load(Heads + selected_heads, selected_heads < H, 0).to(
            tl.int64
        )
        dims = tl.arange(0, D)
        query = tl.load(
            Q
            + original_row * q_row
            + original_heads[:, None] * q_head
            + dims[None, :] * q_dim,
            selected_heads[:, None] < H,
            0,
        )
        sink = tl.load(Sink + original_heads * sink_stride, selected_heads < H, 0)
        infinite_sink = sink == float("inf")
        maximum = tl.where(infinite_sink, 0.0, sink).to(tl.float32)
        denominator = tl.where(sink == -float("inf"), 0.0, 1.0)
        accumulator = tl.full((BH, D), 0, tl.float32)
        length = tl.minimum(
            tl.maximum(tl.load(Lengths + original_row * len_stride), 0), K
        )
        for tile in range(tl.cdiv(length, BK)):
            offsets = tile * BK + tl.arange(0, BK)
            candidates = tl.load(
                Indices + original_row * idx_row + offsets * idx_col,
                offsets < length,
                -1,
            ).to(tl.int64)
            valid = (offsets < length) & (candidates >= 0) & (candidates < S)
            values = tl.load(
                KV + candidates[:, None] * kv_row + dims[None, :] * kv_dim,
                valid[:, None],
                0,
            )
            scores = tl.dot(query, tl.trans(values)) * scale
            scores = tl.where(valid[None, :], scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            # Avoid -inf - -inf in fully masked tiles, including a -inf sink.
            alpha = tl.exp(
                tl.where(
                    maximum == -float("inf"), -float("inf"), maximum - next_maximum
                )
            )
            safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
            probabilities = tl.exp(scores - safe_maximum[:, None])
            denominator = denominator * alpha + tl.sum(probabilities, axis=1)
            accumulator = accumulator * alpha[:, None]
            accumulator += tl.dot(probabilities.to(values.dtype), values)
            maximum = next_maximum
        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        output = accumulator / safe_denominator[:, None]
        output = tl.where(infinite_sink[:, None], 0.0, output)
        tl.store(
            Out + (selected_row * H + selected_heads[:, None]) * D + dims[None, :],
            output,
            selected_heads[:, None] < H,
        )

    return selected_kernel


def selected_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor,
    rows: Sequence[int],
    head_ids: Sequence[int],
    scale: float | None = None,
) -> torch.Tensor:
    """Compute only selected original query rows and heads using Triton.

    Args:
        q: Finite BF16/FP16 CUDA queries, [T, H, 512].
        kv: Native gathered shared keys/values, [S, 1, 512], same dtype/device.
        indices: Native candidates, int32/int64 [T, K] or [T, 1, K]. Invalid
            entries (< 0 or >= S) are masked; duplicates retain multiplicity.
        lens: Candidate lengths [T]. Selected lengths outside [0, K] are rejected
            with a device-to-host synchronization before any kernel launch.
        sink: Original-order float32 head logits [H]; +/- infinity supported.
        rows: Unique original row IDs, in desired output order (CPU sequence).
        head_ids: Unique arbitrary original head IDs (CPU sequence).
        scale: Positive attention scale, default 1/sqrt(512).

    Returns:
        A new contiguous [len(rows), len(head_ids), 512] tensor. It does not write
        q, native gathered KV, candidate indices, or any native cache/state.

    Actual head work is rounded to 16 per query, not to FlashMLA's 64-head
    minimum. Empty selections launch nothing. This is not a full RedKnot cache
    implementation and does not skip native QKV, compressor, indexer, or wo_a.
    """
    import torch

    indices, rows, heads, scale = _validate(
        q, kv, indices, lens, sink, rows, head_ids, scale
    )
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("selected_sparse_mla requires CUDA BF16/FP16 q and kv")
    if sink.dtype != torch.float32:
        raise ValueError("selected_sparse_mla requires float32 sink logits")
    output = q.new_empty((len(rows), len(heads), HEAD_DIM))
    if not rows or not heads:
        return output
    selected_rows = torch.tensor(rows, dtype=torch.int32, device=q.device)
    selected_heads = torch.tensor(heads, dtype=torch.int32, device=q.device)
    grid = (len(rows), (len(heads) + HEAD_TILE - 1) // HEAD_TILE)
    with torch.cuda.device(q.device):
        _kernel()[grid](
            q,
            kv,
            indices,
            lens,
            sink,
            selected_rows,
            selected_heads,
            output,
            *q.stride(),
            kv.stride(0),
            kv.stride(2),
            *indices.stride(),
            lens.stride(0),
            sink.stride(0),
            kv.shape[0],
            indices.shape[1],
            len(heads),
            scale,
            HEAD_TILE,
            KV_TILE,
            HEAD_DIM,
            num_warps=8,
            num_stages=1,
        )
    return output
