"""Three-launch pointer-table restore kernels for DSV4 shared latent KV.

This module is intentionally separate from the controller and SGLang binding
layers.  It consumes the persistent int64 descriptor rows produced by
``DeviceRestoreBatchWorkspace`` and launches exactly one Triton kernel for
each semantic family:

* packed SWA/C4/C128: copy positionless FP8/scales and apply destination RoPE
  to the canonical BF16[64] field;
* Indexer: apply destination RoPE to canonical pre-Hadamard BF16[128], perform
  the native normalized 128-point Hadamard transform, and quantize the whole
  row to e4m3fn with one FP32 scale;
* state: copy opaque C4/C128 attention/Indexer restart records into their
  certified physical terminal-group slots.

Every layer cache remains a separate allocation.  Kernels chase the certified
uint64 addresses in the descriptor table, so no padded ``[layers,rows,...]``
activation or per-layer CUDA launch is materialized.  Production construction
requires an unforgeable in-process :class:`OracleCertificate` issued only
after these exact sources JIT and match the native callbacks on the serving
target SM89/SM90+ GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
    DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS,
    DeviceRestoreBatchKernel,
    DeviceRestoreBatchPreflightKernel,
    RESTORE_FAMILY_INDEXER,
    RESTORE_FAMILY_PACKED,
    RESTORE_FAMILY_STATE,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_sglang import (
    SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS,
)


# Common descriptor columns (owned by dsv4_shared_latent_gpu).
# Triton 3.5 no longer captures ordinary Python globals from ``@triton.jit``
# functions.  These descriptor ordinals and arithmetic constants are genuine
# compile-time values, so instantiate them as ``tl.constexpr`` instead of
# relying on the temporary TRITON_ALLOW_NON_CONSTEXPR_GLOBALS escape hatch.
_D_INPUT = tl.constexpr(0)
_D_DOMAIN = tl.constexpr(1)
_D_LAYER = tl.constexpr(2)
_D_COUNT = tl.constexpr(3)
_D_SOURCE_BANK_PTR = tl.constexpr(4)
_D_SOURCE_INDICES_PTR = tl.constexpr(5)
_D_TARGET_CACHE_PTR = tl.constexpr(6)
_D_TARGET_SLOTS_PTR = tl.constexpr(7)
_D_OUTPUT_ROWS_PTR = tl.constexpr(8)
_D_POSITIONS_PTR = tl.constexpr(9)
_D_RECORD_BYTES = tl.constexpr(10)
_D_SOURCE_ROWS = tl.constexpr(11)
_D_VECTOR_ROWS = tl.constexpr(12)

# SGLang model suffix columns.
_MODEL_BEGIN = len(DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS)
_D_TARGET_PAGE_SIZE = tl.constexpr(_MODEL_BEGIN + 0)
_D_TARGET_ROW_BYTES = tl.constexpr(_MODEL_BEGIN + 1)
_D_STATE_GROUP_WIDTH = tl.constexpr(_MODEL_BEGIN + 2)
_D_STATE_REQUIRED_GROUPS = tl.constexpr(_MODEL_BEGIN + 3)
_D_FREQS_CIS_PTR = tl.constexpr(_MODEL_BEGIN + 4)
_D_FREQS_CIS_ROW_BYTES = tl.constexpr(_MODEL_BEGIN + 5)
_D_FREQS_CIS_ROWS = tl.constexpr(_MODEL_BEGIN + 6)
_D_TARGET_PHYSICAL_ROWS = tl.constexpr(_MODEL_BEGIN + 7)
_REQUIRED_DESCRIPTOR_WIDTH = _MODEL_BEGIN + len(
    SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS
)

_PACKED_RECORD_BYTES = tl.constexpr(584)
_PACKED_NOPE_BYTES = tl.constexpr(448)
_PACKED_NOPE_ROPE_BYTES = tl.constexpr(576)
_PACKED_SCALE_BYTES = tl.constexpr(8)
_INDEXER_RECORD_BYTES = tl.constexpr(256)  # canonical BF16[128]
_INDEXER_DIM = tl.constexpr(128)
_ROPE_DIM = tl.constexpr(64)
_FP8_E4M3_MAX = tl.constexpr(448.0)
_QUANT_AMAX_FLOOR = tl.constexpr(1.0e-4)
_INV_SQRT_128 = tl.constexpr(0.08838834764831845)

_VALIDATION_SOURCE_BOUNDS = tl.constexpr(1)
_VALIDATION_OUTPUT_BOUNDS = tl.constexpr(2)
_VALIDATION_TARGET_BOUNDS = tl.constexpr(4)
_VALIDATION_POSITION_BOUNDS = tl.constexpr(8)
_VALIDATION_TARGET_OVERLAP = tl.constexpr(16)
_VALIDATION_HASH_EXHAUSTED = tl.constexpr(32)


@triton.jit
def _descriptor(Descriptors, job, stride: tl.constexpr, column: tl.constexpr):
    return tl.load(Descriptors + job * stride + column)


@triton.jit
def _as_u8_pointer(address):
    return tl.cast(address, tl.pointer_type(tl.uint8))


@triton.jit
def _as_i64_pointer(address):
    return tl.cast(address, tl.pointer_type(tl.int64))


@triton.jit
def _as_bf16_pointer(address):
    return tl.cast(address, tl.pointer_type(tl.bfloat16))


@triton.jit
def _as_f32_pointer(address):
    return tl.cast(address, tl.pointer_type(tl.float32))


@triton.jit
def _as_fp8_pointer(address):
    return tl.cast(address, tl.pointer_type(tl.float8e4nv))


@triton.jit
def _destination_rope64(
    canonical,
    position,
    freqs_address,
    freqs_row_bytes,
    active,
):
    """Apply adjacent-pair DSV4 RoPE and return BF16-rounded values."""

    paired = tl.reshape(canonical, 32, 2)
    real, imaginary = tl.split(paired)
    pair = tl.arange(0, 32)
    freq = _as_f32_pointer(freqs_address)
    safe_position = tl.where(active, position, 0)
    freq_row = (safe_position * freqs_row_bytes) // 4
    cosine = tl.load(
        freq + freq_row + pair * 2, mask=active, other=1.0
    )
    sine = tl.load(
        freq + freq_row + pair * 2 + 1, mask=active, other=0.0
    )
    rotated = tl.join(
        real * cosine - imaginary * sine,
        real * sine + imaginary * cosine,
    )
    # The native apply_rotary_emb_triton path writes BF16 before Indexer H.
    return tl.reshape(rotated, _ROPE_DIM).to(tl.bfloat16)


@triton.jit
def _fwht_stage_128(x, GROUPS: tl.constexpr, WIDTH: tl.constexpr):
    paired = tl.reshape(x, GROUPS, 2, WIDTH)
    split_last = tl.permute(paired, 0, 2, 1)
    left, right = tl.split(split_last)
    joined = tl.join(left + right, left - right)
    paired = tl.permute(joined, 0, 2, 1)
    return tl.reshape(paired, _INDEXER_DIM)


@triton.jit
def _normalized_fwht128(x):
    # Static shapes keep the complete transform in one program's registers.
    x = _fwht_stage_128(x, 64, 1)
    x = _fwht_stage_128(x, 32, 2)
    x = _fwht_stage_128(x, 16, 4)
    x = _fwht_stage_128(x, 8, 8)
    x = _fwht_stage_128(x, 4, 16)
    x = _fwht_stage_128(x, 2, 32)
    x = _fwht_stage_128(x, 1, 64)
    return x * _INV_SQRT_128


@triton.jit
def _bounds_preflight_pointer_table_kernel(
    Descriptors,
    Status,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
):
    """Validate every dynamic dereference before any target-cache launch."""

    linear = tl.program_id(0)
    job = linear // MAX_ROWS
    row = linear - job * MAX_ROWS
    count = _descriptor(Descriptors, job, DESCRIPTOR_WIDTH, _D_COUNT)
    active = (job < NUM_JOBS) & (row < count)

    source_indices_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_INDICES_PTR
    )
    output_rows_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    target_slots_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    positions_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_POSITIONS_PTR
    )
    source_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_ROWS
    )
    vector_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_VECTOR_ROWS
    )
    page_size = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PAGE_SIZE
    )
    group_width = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_GROUP_WIDTH
    )
    required_groups = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_REQUIRED_GROUPS
    )
    freqs_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_ROWS
    )
    target_physical_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PHYSICAL_ROWS
    )

    source_index = tl.load(
        _as_i64_pointer(source_indices_address) + row,
        mask=active,
        other=0,
    )
    output_row = tl.load(
        _as_i64_pointer(output_rows_address) + row,
        mask=active,
        other=0,
    )
    source_valid = (source_index >= 0) & (source_index < source_rows)
    output_valid = (output_row >= 0) & (output_row < vector_rows)
    selected = active & output_valid
    safe_output_row = tl.where(selected, output_row, 0)
    target_slot = tl.load(
        _as_i64_pointer(target_slots_address) + safe_output_row,
        mask=selected,
        other=0,
    )
    position = tl.load(
        _as_i64_pointer(positions_address) + safe_output_row,
        mask=selected,
        other=0,
    )

    is_state = group_width > 0
    positioned_target_valid = (
        (target_slot >= 0)
        & (page_size > 0)
        & (target_slot < target_physical_rows * page_size)
    )
    safe_group_width = tl.maximum(group_width, 1)
    state_group_count = target_physical_rows // safe_group_width
    state_target_valid = (
        (required_groups > 0)
        & (target_slot >= required_groups - 1)
        & (target_slot < state_group_count)
    )
    target_valid = tl.where(
        is_state, state_target_valid, positioned_target_valid
    )
    position_valid = is_state | (
        (freqs_rows > 0) & (position >= 0) & (position < freqs_rows)
    )

    tl.atomic_or(
        Status,
        _VALIDATION_SOURCE_BOUNDS,
        mask=active & (~source_valid),
    )
    tl.atomic_or(
        Status,
        _VALIDATION_OUTPUT_BOUNDS,
        mask=active & (~output_valid),
    )
    tl.atomic_or(
        Status,
        _VALIDATION_TARGET_BOUNDS,
        mask=selected & (~target_valid),
    )
    tl.atomic_or(
        Status,
        _VALIDATION_POSITION_BOUNDS,
        mask=selected & (~position_valid),
    )


_BOUNDS_PREFLIGHT_BLOCK_ROWS = 32
_DESTINATION_MATERIALIZER_BLOCK_ROWS = 32
_DESTINATION_MATERIALIZER_MAX_GROUPS = 2


@triton.jit
def _bounds_preflight_pointer_table_block32_kernel(
    Descriptors,
    Status,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Vectorized equivalent of the scalar pointer-table bounds proof.

    One program owns one job and up to ``BLOCK_ROWS`` rows.  The status
    predicates and bits are identical to the scalar oracle; only launch
    geometry changes.  Exact physical destination aliasing remains covered by
    the independent sort proof below.
    """

    job = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    count = _descriptor(Descriptors, job, DESCRIPTOR_WIDTH, _D_COUNT)
    active = (job < NUM_JOBS) & (rows < count)

    source_indices_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_INDICES_PTR
    )
    output_rows_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    target_slots_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    positions_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_POSITIONS_PTR
    )
    source_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_ROWS
    )
    vector_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_VECTOR_ROWS
    )
    page_size = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PAGE_SIZE
    )
    group_width = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_GROUP_WIDTH
    )
    required_groups = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_REQUIRED_GROUPS
    )
    freqs_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_ROWS
    )
    target_physical_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PHYSICAL_ROWS
    )

    source_index = tl.load(
        _as_i64_pointer(source_indices_address) + rows,
        mask=active,
        other=0,
    )
    output_row = tl.load(
        _as_i64_pointer(output_rows_address) + rows,
        mask=active,
        other=0,
    )
    source_valid = (source_index >= 0) & (source_index < source_rows)
    output_valid = (output_row >= 0) & (output_row < vector_rows)
    selected = active & output_valid
    safe_output_row = tl.where(selected, output_row, 0)
    target_slot = tl.load(
        _as_i64_pointer(target_slots_address) + safe_output_row,
        mask=selected,
        other=0,
    )
    position = tl.load(
        _as_i64_pointer(positions_address) + safe_output_row,
        mask=selected,
        other=0,
    )

    is_state = group_width > 0
    positioned_target_valid = (
        (target_slot >= 0)
        & (page_size > 0)
        & (target_slot < target_physical_rows * page_size)
    )
    safe_group_width = tl.maximum(group_width, 1)
    state_group_count = target_physical_rows // safe_group_width
    state_target_valid = (
        (required_groups > 0)
        & (target_slot >= required_groups - 1)
        & (target_slot < state_group_count)
    )
    target_valid = tl.where(
        is_state, state_target_valid, positioned_target_valid
    )
    position_valid = is_state | (
        (freqs_rows > 0) & (position >= 0) & (position < freqs_rows)
    )

    status_pointers = Status + tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
    tl.atomic_or(
        status_pointers,
        tl.full((BLOCK_ROWS,), _VALIDATION_SOURCE_BOUNDS, dtype=tl.int32),
        mask=active & (~source_valid),
    )
    tl.atomic_or(
        status_pointers,
        tl.full((BLOCK_ROWS,), _VALIDATION_OUTPUT_BOUNDS, dtype=tl.int32),
        mask=active & (~output_valid),
    )
    tl.atomic_or(
        status_pointers,
        tl.full((BLOCK_ROWS,), _VALIDATION_TARGET_BOUNDS, dtype=tl.int32),
        mask=selected & (~target_valid),
    )
    tl.atomic_or(
        status_pointers,
        tl.full((BLOCK_ROWS,), _VALIDATION_POSITION_BOUNDS, dtype=tl.int32),
        mask=selected & (~position_valid),
    )


@triton.jit
def _materialize_destination_keys_block32_kernel(
    Descriptors,
    JobOffsets,
    DestinationKeys,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Materialize every certified destination address with one launch.

    The previous implementation built one address vector per Python job and
    therefore launched hundreds of eager tensor kernels before the exact sort
    proof.  This kernel preserves the same address and unique-negative-sentinel
    semantics while packing all jobs into one flat buffer.
    """

    job = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    group = tl.program_id(2)
    count = _descriptor(Descriptors, job, DESCRIPTOR_WIDTH, _D_COUNT)
    required_groups = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_REQUIRED_GROUPS
    )
    destination_units = tl.maximum(required_groups, 1)
    active = (
        (job < NUM_JOBS)
        & (rows < count)
        & (group < destination_units)
    )

    output_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    slots_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    target_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_CACHE_PTR
    )
    vector_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_VECTOR_ROWS
    )
    page_size = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PAGE_SIZE
    )
    target_row_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_ROW_BYTES
    )
    group_width = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_GROUP_WIDTH
    )
    record_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_RECORD_BYTES
    )
    target_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PHYSICAL_ROWS
    )

    output_row = tl.load(
        _as_i64_pointer(output_address) + rows,
        mask=active,
        other=0,
    )
    output_valid = (output_row >= 0) & (output_row < vector_rows)
    selected = active & output_valid
    safe_output = tl.where(selected, output_row, 0)
    target_slot = tl.load(
        _as_i64_pointer(slots_address) + safe_output,
        mask=selected,
        other=0,
    )

    is_state = group_width > 0
    safe_group_width = tl.maximum(group_width, 1)
    state_group_count = target_rows // safe_group_width
    state_valid = (
        (required_groups > 0)
        & (target_slot >= required_groups - 1)
        & (target_slot < state_group_count)
    )
    positioned_valid = (
        (target_slot >= 0)
        & (page_size > 0)
        & (target_slot < target_rows * page_size)
    )
    destination_valid = selected & tl.where(
        is_state, state_valid, positioned_valid
    )

    safe_page_size = tl.maximum(page_size, 1)
    page = target_slot // safe_page_size
    token = target_slot - page * safe_page_size
    slot_bytes = tl.where(record_bytes == _PACKED_RECORD_BYTES, 576, 128)
    positioned_destination = (
        target_address + page * target_row_bytes + token * slot_bytes
    )
    state_group = target_slot - (required_groups - 1 - group)
    state_destination = (
        target_address + state_group * group_width * target_row_bytes
    )
    destination = tl.where(
        is_state, state_destination, positioned_destination
    )

    job_offset = tl.load(JobOffsets + job)
    packed_ordinal = job_offset + rows * destination_units + group
    destination = tl.where(
        destination_valid, destination, -(packed_ordinal + 1)
    )
    tl.store(
        DestinationKeys + packed_ordinal,
        destination,
        mask=active,
    )


@triton.jit
def _state_overlap_preflight_pointer_table_kernel(
    Descriptors,
    Status,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
    TOTAL_ENTRIES: tl.constexpr,
):
    """Reject intersecting state-history byte intervals across the batch."""

    pair = tl.program_id(0)
    left_entry = pair // TOTAL_ENTRIES
    right_entry = pair - left_entry * TOTAL_ENTRIES
    ordered_pair = left_entry < right_entry
    left_job = left_entry // MAX_ROWS
    left_row = left_entry - left_job * MAX_ROWS
    right_job = right_entry // MAX_ROWS
    right_row = right_entry - right_job * MAX_ROWS

    left_count = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_COUNT
    )
    right_count = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_COUNT
    )
    active = (
        ordered_pair
        & (left_job < NUM_JOBS)
        & (right_job < NUM_JOBS)
        & (left_row < left_count)
        & (right_row < right_count)
    )

    left_output_address = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    right_output_address = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    left_vector_rows = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_VECTOR_ROWS
    )
    right_vector_rows = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_VECTOR_ROWS
    )
    left_output = tl.load(
        _as_i64_pointer(left_output_address) + left_row,
        mask=active,
        other=0,
    )
    right_output = tl.load(
        _as_i64_pointer(right_output_address) + right_row,
        mask=active,
        other=0,
    )
    outputs_valid = (
        (left_output >= 0)
        & (left_output < left_vector_rows)
        & (right_output >= 0)
        & (right_output < right_vector_rows)
    )
    comparable = active & outputs_valid
    left_output = tl.where(comparable, left_output, 0)
    right_output = tl.where(comparable, right_output, 0)

    left_slots_address = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    right_slots_address = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    left_terminal = tl.load(
        _as_i64_pointer(left_slots_address) + left_output,
        mask=comparable,
        other=0,
    )
    right_terminal = tl.load(
        _as_i64_pointer(right_slots_address) + right_output,
        mask=comparable,
        other=0,
    )

    left_target = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_TARGET_CACHE_PTR
    )
    right_target = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_TARGET_CACHE_PTR
    )
    left_width = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_STATE_GROUP_WIDTH
    )
    right_width = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_STATE_GROUP_WIDTH
    )
    left_row_bytes = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_TARGET_ROW_BYTES
    )
    right_row_bytes = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_TARGET_ROW_BYTES
    )
    left_required = _descriptor(
        Descriptors, left_job, DESCRIPTOR_WIDTH, _D_STATE_REQUIRED_GROUPS
    )
    right_required = _descriptor(
        Descriptors, right_job, DESCRIPTOR_WIDTH, _D_STATE_REQUIRED_GROUPS
    )
    left_group_bytes = left_width * left_row_bytes
    right_group_bytes = right_width * right_row_bytes
    left_begin = (left_terminal - (left_required - 1)) * left_group_bytes
    left_end = (left_terminal + 1) * left_group_bytes
    right_begin = (right_terminal - (right_required - 1)) * right_group_bytes
    right_end = (right_terminal + 1) * right_group_bytes
    overlap = (
        comparable
        & (left_target == right_target)
        & (left_begin < right_end)
        & (right_begin < left_end)
    )
    tl.atomic_or(Status, _VALIDATION_TARGET_OVERLAP, mask=overlap)


@triton.jit
def _target_collision_preflight_pointer_table_kernel(
    Descriptors,
    Status,
    CollisionTable,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
):
    """Insert every physical destination into an exact owner-id hash set.

    The table deliberately stores a 32-bit, one-based program/owner id rather
    than the 64-bit destination address.  Triton 3.5's NVIDIA backend can
    mis-lower a scalar i64 ``atomic_cas`` on SM89/SM90, while i32 CAS is a
    native and well-tested path.  A non-empty bucket is still exact: its owner
    id is decoded back into ``(job, row, group)`` and the immutable descriptor
    table is used to recompute the complete 64-bit physical address before a
    duplicate is reported.
    """

    linear = tl.program_id(0)
    group_ordinal = linear % MAX_GROUPS
    job_row = linear // MAX_GROUPS
    row = job_row % MAX_ROWS
    job = job_row // MAX_ROWS
    count = _descriptor(Descriptors, job, DESCRIPTOR_WIDTH, _D_COUNT)
    output_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    slots_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    vector_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_VECTOR_ROWS
    )
    group_width = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_GROUP_WIDTH
    )
    required_groups = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_REQUIRED_GROUPS
    )
    is_state = group_width > 0
    destination_units = tl.where(is_state, required_groups, 1)
    active = (
        (job < NUM_JOBS)
        & (row < count)
        & (group_ordinal < destination_units)
    )
    output_row = tl.load(
        _as_i64_pointer(output_address) + row, mask=active, other=0
    )
    output_valid = (output_row >= 0) & (output_row < vector_rows)
    selected = active & output_valid
    safe_output = tl.where(selected, output_row, 0)
    target_slot = tl.load(
        _as_i64_pointer(slots_address) + safe_output,
        mask=selected,
        other=0,
    )

    target_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_CACHE_PTR
    )
    target_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PHYSICAL_ROWS
    )
    target_row_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_ROW_BYTES
    )
    page_size = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PAGE_SIZE
    )
    record_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_RECORD_BYTES
    )

    safe_group_width = tl.maximum(group_width, 1)
    state_group_count = target_rows // safe_group_width
    state_valid = (
        (target_slot >= required_groups - 1)
        & (target_slot < state_group_count)
    )
    positioned_valid = (
        (page_size > 0)
        & (target_slot >= 0)
        & (target_slot < target_rows * page_size)
    )
    destination_valid = selected & tl.where(
        is_state, state_valid, positioned_valid
    )

    state_group = target_slot - (required_groups - 1 - group_ordinal)
    state_key = (
        target_address
        + state_group * safe_group_width * target_row_bytes
    )
    page = target_slot // tl.maximum(page_size, 1)
    token = target_slot - page * tl.maximum(page_size, 1)
    positioned_slot_bytes = tl.where(
        record_bytes == _PACKED_RECORD_BYTES, _PACKED_NOPE_ROPE_BYTES, 128
    )
    positioned_key = (
        target_address
        + page * target_row_bytes
        + token * positioned_slot_bytes
    )
    key = tl.where(is_state, state_key, positioned_key)

    table_mask = TABLE_SIZE - 1
    table_slot = ((key >> 4) ^ (key >> 32)) & table_mask
    owner_id = linear + 1
    # Keep every operand of atomic_cas i32.  In particular, never put ``key``
    # (an i64 address) in the table: the target backend currently emits an
    # invalid shared-memory store for scalar i64 CAS on SM89/SM90.
    empty_owner = owner_id ^ owner_id
    probing = destination_valid
    probe = 0
    while probing & (probe < TABLE_SIZE):
        prior = tl.atomic_cas(
            CollisionTable + table_slot,
            empty_owner,
            owner_id,
        )

        occupied = probing & (prior != empty_owner)
        owner_in_range = occupied & (
            (prior > 0)
            & (prior <= NUM_JOBS * MAX_ROWS * MAX_GROUPS)
        )
        corrupt_owner = occupied & (~owner_in_range)
        tl.atomic_or(
            Status, _VALIDATION_HASH_EXHAUSTED, mask=corrupt_owner
        )

        # Decode the incumbent and independently rebuild its full i64 key.
        # ``safe_owner_linear`` keeps all descriptor pointer arithmetic valid
        # for the empty-bucket and defensive-corruption paths.
        safe_owner_linear = tl.where(owner_in_range, prior - 1, 0)
        owner_group_ordinal = safe_owner_linear % MAX_GROUPS
        owner_job_row = safe_owner_linear // MAX_GROUPS
        owner_row = owner_job_row % MAX_ROWS
        owner_job = owner_job_row // MAX_ROWS
        owner_output_address = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_OUTPUT_ROWS_PTR,
        )
        owner_slots_address = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_TARGET_SLOTS_PTR,
        )
        owner_output_row = tl.load(
            _as_i64_pointer(owner_output_address) + owner_row,
            mask=owner_in_range,
            other=0,
        )
        owner_target_slot = tl.load(
            _as_i64_pointer(owner_slots_address) + owner_output_row,
            mask=owner_in_range,
            other=0,
        )
        owner_target_address = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_TARGET_CACHE_PTR,
        )
        owner_target_row_bytes = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_TARGET_ROW_BYTES,
        )
        owner_page_size = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_TARGET_PAGE_SIZE,
        )
        owner_record_bytes = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_RECORD_BYTES,
        )
        owner_group_width = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_STATE_GROUP_WIDTH,
        )
        owner_required_groups = _descriptor(
            Descriptors,
            owner_job,
            DESCRIPTOR_WIDTH,
            _D_STATE_REQUIRED_GROUPS,
        )
        owner_is_state = owner_group_width > 0
        safe_owner_group_width = tl.maximum(owner_group_width, 1)
        owner_state_group = owner_target_slot - (
            owner_required_groups - 1 - owner_group_ordinal
        )
        owner_state_key = (
            owner_target_address
            + owner_state_group
            * safe_owner_group_width
            * owner_target_row_bytes
        )
        safe_owner_page_size = tl.maximum(owner_page_size, 1)
        owner_page = owner_target_slot // safe_owner_page_size
        owner_token = (
            owner_target_slot - owner_page * safe_owner_page_size
        )
        owner_positioned_slot_bytes = tl.where(
            owner_record_bytes == _PACKED_RECORD_BYTES,
            _PACKED_NOPE_ROPE_BYTES,
            128,
        )
        owner_positioned_key = (
            owner_target_address
            + owner_page * owner_target_row_bytes
            + owner_token * owner_positioned_slot_bytes
        )
        owner_key = tl.where(
            owner_is_state, owner_state_key, owner_positioned_key
        )

        duplicate = owner_in_range & (owner_key == key)
        tl.atomic_or(Status, _VALIDATION_TARGET_OVERLAP, mask=duplicate)
        installed = (prior == empty_owner) | duplicate | corrupt_owner
        probing = probing & (~installed)
        table_slot = (table_slot + 1) & table_mask
        probe += 1
    tl.atomic_or(Status, _VALIDATION_HASH_EXHAUSTED, mask=probing)


@triton.jit
def _packed_restore_pointer_table_kernel(
    Descriptors,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
):
    linear = tl.program_id(0)
    job = linear // MAX_ROWS
    row = linear - job * MAX_ROWS
    count = _descriptor(Descriptors, job, DESCRIPTOR_WIDTH, _D_COUNT)
    active = (job < NUM_JOBS) & (row < count)

    source_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_BANK_PTR
    )
    source_indices_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_INDICES_PTR
    )
    target_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_CACHE_PTR
    )
    target_slots_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    output_rows_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    positions_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_POSITIONS_PTR
    )
    record_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_RECORD_BYTES
    )
    page_size = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PAGE_SIZE
    )
    target_row_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_ROW_BYTES
    )
    freqs_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_PTR
    )
    freqs_row_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_ROW_BYTES
    )
    freqs_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_ROWS
    )
    target_physical_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PHYSICAL_ROWS
    )

    source_indices = _as_i64_pointer(source_indices_address)
    output_rows = _as_i64_pointer(output_rows_address)
    target_slots = _as_i64_pointer(target_slots_address)
    positions = _as_i64_pointer(positions_address)
    source_index = tl.load(source_indices + row, mask=active, other=0)
    output_row = tl.load(output_rows + row, mask=active, other=0)
    target_slot = tl.load(target_slots + output_row, mask=active, other=0)
    position = tl.load(positions + output_row, mask=active, other=0)
    destination_valid = (
        (target_slot >= 0)
        & (target_slot < target_physical_rows * page_size)
        & (position >= 0)
        & (position < freqs_rows)
    )
    tl.device_assert((~active) | destination_valid, "packed restore bounds")
    active = active & destination_valid
    page = target_slot // page_size
    token = target_slot - page * page_size

    source = _as_u8_pointer(source_address)
    target = _as_u8_pointer(target_address)
    source_row = source + source_index * record_bytes
    target_kv_row = target + page * target_row_bytes + token * _PACKED_NOPE_ROPE_BYTES

    byte_offsets = tl.arange(0, 512)
    nope_mask = active & (byte_offsets < _PACKED_NOPE_BYTES)
    nope = tl.load(source_row + byte_offsets, mask=nope_mask, other=0)
    tl.store(target_kv_row + byte_offsets, nope, mask=nope_mask)

    rope_offsets = tl.arange(0, _ROPE_DIM)
    canonical_rope = tl.load(
        _as_bf16_pointer(
            source_address
            + source_index * record_bytes
            + _PACKED_NOPE_BYTES
        )
        + rope_offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    positioned_rope = _destination_rope64(
        canonical_rope,
        position,
        freqs_address,
        freqs_row_bytes,
        active,
    )
    target_rope = _as_bf16_pointer(
        target_address
        + page * target_row_bytes
        + token * _PACKED_NOPE_ROPE_BYTES
        + _PACKED_NOPE_BYTES
    )
    tl.store(target_rope + rope_offsets, positioned_rope, mask=active)

    scale_offsets = tl.arange(0, _PACKED_SCALE_BYTES)
    scales = tl.load(
        source_row + _PACKED_NOPE_ROPE_BYTES + scale_offsets,
        mask=active,
        other=0,
    )
    target_scale = (
        target
        + page * target_row_bytes
        + page_size * _PACKED_NOPE_ROPE_BYTES
        + token * _PACKED_SCALE_BYTES
    )
    tl.store(target_scale + scale_offsets, scales, mask=active)


@triton.jit
def _indexer_restore_pointer_table_kernel(
    Descriptors,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
):
    linear = tl.program_id(0)
    job = linear // MAX_ROWS
    row = linear - job * MAX_ROWS
    count = _descriptor(Descriptors, job, DESCRIPTOR_WIDTH, _D_COUNT)
    active = (job < NUM_JOBS) & (row < count)

    source_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_BANK_PTR
    )
    source_indices_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_INDICES_PTR
    )
    target_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_CACHE_PTR
    )
    target_slots_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    output_rows_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    positions_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_POSITIONS_PTR
    )
    record_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_RECORD_BYTES
    )
    page_size = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PAGE_SIZE
    )
    target_row_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_ROW_BYTES
    )
    freqs_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_PTR
    )
    freqs_row_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_ROW_BYTES
    )
    freqs_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_FREQS_CIS_ROWS
    )
    target_physical_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PHYSICAL_ROWS
    )

    source_indices = _as_i64_pointer(source_indices_address)
    output_rows = _as_i64_pointer(output_rows_address)
    target_slots = _as_i64_pointer(target_slots_address)
    positions = _as_i64_pointer(positions_address)
    source_index = tl.load(source_indices + row, mask=active, other=0)
    output_row = tl.load(output_rows + row, mask=active, other=0)
    target_slot = tl.load(target_slots + output_row, mask=active, other=0)
    position = tl.load(positions + output_row, mask=active, other=0)
    destination_valid = (
        (target_slot >= 0)
        & (target_slot < target_physical_rows * page_size)
        & (position >= 0)
        & (position < freqs_rows)
    )
    tl.device_assert((~active) | destination_valid, "Indexer restore bounds")
    active = active & destination_valid

    offsets = tl.arange(0, _INDEXER_DIM)
    canonical = tl.load(
        _as_bf16_pointer(source_address + source_index * record_bytes) + offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    canonical_halves = tl.reshape(canonical, 2, 64)
    canonical_interleaved = tl.permute(canonical_halves, 1, 0)
    canonical_nope, canonical_rope = tl.split(canonical_interleaved)
    positioned_rope = _destination_rope64(
        canonical_rope,
        position,
        freqs_address,
        freqs_row_bytes,
        active,
    ).to(tl.float32)
    positioned_interleaved = tl.join(canonical_nope, positioned_rope)
    positioned_halves = tl.permute(positioned_interleaved, 1, 0)
    pre_hadamard = tl.reshape(positioned_halves, _INDEXER_DIM)
    post_hadamard = _normalized_fwht128(pre_hadamard)

    amax = tl.max(tl.abs(post_hadamard), axis=0)
    amax = tl.maximum(amax, _QUANT_AMAX_FLOOR)
    scale = amax / _FP8_E4M3_MAX
    scaled = post_hadamard / scale
    scaled = tl.maximum(tl.minimum(scaled, _FP8_E4M3_MAX), -_FP8_E4M3_MAX)
    quantized = scaled.to(tl.float8e4nv)

    page = target_slot // page_size
    token = target_slot - page * page_size
    target_key_address = (
        target_address + page * target_row_bytes + token * _INDEXER_DIM
    )
    tl.store(
        _as_fp8_pointer(target_key_address) + offsets,
        quantized,
        mask=active,
    )
    target_scale_address = (
        target_address
        + page * target_row_bytes
        + page_size * _INDEXER_DIM
        + token * 4
    )
    tl.store(_as_f32_pointer(target_scale_address), scale, mask=active)


@triton.jit
def _state_restore_pointer_table_kernel(
    Descriptors,
    NUM_JOBS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    DESCRIPTOR_WIDTH: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    linear = tl.program_id(0)
    chunk = linear % MAX_CHUNKS
    job_row = linear // MAX_CHUNKS
    row = job_row % MAX_ROWS
    job = job_row // MAX_ROWS
    count = _descriptor(Descriptors, job, DESCRIPTOR_WIDTH, _D_COUNT)
    active = (job < NUM_JOBS) & (row < count)

    source_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_BANK_PTR
    )
    source_indices_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_SOURCE_INDICES_PTR
    )
    target_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_CACHE_PTR
    )
    target_slots_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_SLOTS_PTR
    )
    output_rows_address = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_OUTPUT_ROWS_PTR
    )
    record_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_RECORD_BYTES
    )
    target_row_bytes = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_ROW_BYTES
    )
    group_width = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_GROUP_WIDTH
    )
    required_groups = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_STATE_REQUIRED_GROUPS
    )
    target_physical_rows = _descriptor(
        Descriptors, job, DESCRIPTOR_WIDTH, _D_TARGET_PHYSICAL_ROWS
    )

    source_indices = _as_i64_pointer(source_indices_address)
    output_rows = _as_i64_pointer(output_rows_address)
    target_slots = _as_i64_pointer(target_slots_address)
    source_index = tl.load(source_indices + row, mask=active, other=0)
    output_row = tl.load(output_rows + row, mask=active, other=0)
    terminal_slot = tl.load(target_slots + output_row, mask=active, other=0)
    target_group_count = target_physical_rows // group_width
    destination_valid = (
        (terminal_slot >= required_groups - 1)
        & (terminal_slot < target_group_count)
    )
    tl.device_assert((~active) | destination_valid, "state restore bounds")
    active = active & destination_valid
    bytes_per_group = group_width * target_row_bytes

    record_offsets = chunk * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    copy_mask = active & (record_offsets < record_bytes)
    values = tl.load(
        _as_u8_pointer(source_address + source_index * record_bytes)
        + record_offsets,
        mask=copy_mask,
        other=0,
    )
    group_index = record_offsets // bytes_per_group
    offset_in_group = record_offsets - group_index * bytes_per_group
    destination_group = terminal_slot - (required_groups - 1 - group_index)
    destination = _as_u8_pointer(target_address) + (
        destination_group * bytes_per_group + offset_in_group
    )
    tl.store(destination, values, mask=copy_mask)


def _validate_callback_inputs(
    *,
    family: str,
    expected_family: str,
    jobs: Sequence[object],
    descriptors: torch.Tensor,
    descriptor_columns: Sequence[str],
) -> tuple[int, int, int]:
    if family != expected_family:
        raise ValueError("batch restore callback received another family")
    if not jobs:
        raise ValueError("batch restore callback has no jobs")
    if (
        not isinstance(descriptors, torch.Tensor)
        or descriptors.device.type != "cuda"
        or descriptors.dtype != torch.int64
        or descriptors.ndim != 2
        or not descriptors.is_contiguous()
        or int(descriptors.shape[0]) != len(jobs)
        or int(descriptors.shape[1]) < _REQUIRED_DESCRIPTOR_WIDTH
    ):
        raise ValueError("batch restore callback descriptor geometry changed")
    if tuple(descriptor_columns) != tuple(
        DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS
    ):
        raise ValueError("batch restore common descriptor ABI changed")
    max_rows = max(int(getattr(job, "count", 0)) for job in jobs)
    if max_rows <= 0:
        raise ValueError("batch restore callback jobs contain no rows")
    return len(jobs), max_rows, int(descriptors.shape[1])


def _sort_destination_overlap_preflight(
    *,
    jobs: Sequence[object],
    descriptors: torch.Tensor,
    device_status: torch.Tensor,
    device_collision_table: torch.Tensor,
    max_rows: int,
) -> None:
    """Detect exact target-byte aliasing without a probing hash table.

    The old scalar collision kernel launched one program per virtual
    ``job,row,group`` and used an atomic open-addressing table.  On the real
    147-job/8K geometry that proof alone takes about ten seconds.  Here we
    materialize the same complete destination byte addresses from the already
    validated immutable jobs, sort them, and flag adjacent equality.  Invalid
    destinations receive unique negative sentinels; the bounds kernel remains
    authoritative for their status bits.
    """

    if (
        not isinstance(descriptors, torch.Tensor)
        or descriptors.device != device_status.device
        or descriptors.dtype != torch.int64
        or descriptors.ndim != 2
        or not descriptors.is_contiguous()
        or int(descriptors.shape[0]) != len(jobs)
        or int(descriptors.shape[1]) < _REQUIRED_DESCRIPTOR_WIDTH
    ):
        raise ValueError("sort preflight descriptor geometry changed")

    job_offsets = []
    required_entries = 0
    for job in jobs:
        count = int(getattr(job, "count", 0))
        metadata = getattr(job, "metadata", None)
        target_cache = getattr(job, "target_cache", None)
        target_slots = getattr(job, "target_slots", None)
        output_rows = getattr(job, "output_rows", None)
        positions = getattr(job, "positions", None)
        if (
            count <= 0
            or metadata is None
            or not isinstance(target_cache, torch.Tensor)
            or not isinstance(target_slots, torch.Tensor)
            or not isinstance(output_rows, torch.Tensor)
            or not isinstance(positions, torch.Tensor)
            or target_cache.device != device_status.device
            or target_slots.device != device_status.device
            or output_rows.device != device_status.device
            or positions.device != device_status.device
            or target_slots.dtype != torch.int64
            or output_rows.dtype != torch.int64
            or positions.dtype != torch.int64
            or target_slots.ndim != 1
            or output_rows.ndim != 1
            or positions.ndim != 1
            or int(output_rows.numel()) != count
        ):
            raise ValueError("sort preflight job tensor geometry changed")

        vector_rows = int(positions.numel())
        if vector_rows <= 0 or int(target_slots.numel()) != vector_rows:
            raise ValueError("sort preflight slot/position vectors diverged")

        page_size = int(getattr(metadata, "target_page_size", 0))
        target_row_bytes = int(getattr(metadata, "target_row_bytes", 0))
        target_physical_rows = int(
            getattr(metadata, "target_physical_rows", 0)
        )
        group_width = int(getattr(metadata, "state_group_width", 0))
        required_groups = int(
            getattr(metadata, "state_required_groups", 0)
        )
        record_bytes = int(getattr(job, "record_bytes", 0))
        if target_row_bytes <= 0 or target_physical_rows <= 0:
            raise ValueError("sort preflight target geometry is absent")
        is_state = group_width > 0
        if is_state:
            if (
                required_groups <= 0
                or required_groups > _DESTINATION_MATERIALIZER_MAX_GROUPS
                or target_physical_rows % group_width
            ):
                raise ValueError("sort preflight state geometry changed")
            destination_units = required_groups
        else:
            if page_size <= 0 or record_bytes not in (256, 584):
                raise ValueError("sort preflight positioned geometry changed")
            destination_units = 1
        job_offsets.append(required_entries)
        required_entries += count * destination_units

    if required_entries <= 0:
        raise ValueError("sort preflight has no destination addresses")
    if (
        not isinstance(device_collision_table, torch.Tensor)
        or device_collision_table.device != device_status.device
        or device_collision_table.dtype != torch.int32
        or device_collision_table.ndim != 1
        or not device_collision_table.is_contiguous()
        or int(device_collision_table.numel()) < 2 * required_entries
    ):
        raise MemoryError("sort preflight key workspace is undersized")
    offsets = torch.tensor(
        job_offsets, dtype=torch.int64, device=device_status.device
    )
    destination_keys = device_collision_table.view(torch.int64)[
        :required_entries
    ]
    _materialize_destination_keys_block32_kernel[
        (
            len(jobs),
            triton.cdiv(
                int(max_rows), _DESTINATION_MATERIALIZER_BLOCK_ROWS
            ),
            _DESTINATION_MATERIALIZER_MAX_GROUPS,
        )
    ](
        descriptors,
        offsets,
        destination_keys,
        NUM_JOBS=len(jobs),
        MAX_ROWS=int(max_rows),
        DESCRIPTOR_WIDTH=int(descriptors.shape[1]),
        BLOCK_ROWS=_DESTINATION_MATERIALIZER_BLOCK_ROWS,
        num_warps=1,
        num_stages=1,
    )
    ordered = torch.sort(destination_keys).values
    duplicate = torch.any(ordered[1:] == ordered[:-1])
    device_status.bitwise_or_(
        duplicate.to(torch.int32) * int(_VALIDATION_TARGET_OVERLAP)
    )


class _BatchPreflightCallback:
    """One bounds phase plus one exact sort-overlap phase and status read."""

    @torch.no_grad()
    def __call__(
        self,
        *,
        jobs,
        descriptors,
        descriptor_columns,
        device_status,
        host_status,
        device_collision_table,
        max_validation_entries,
        plan_digest,
    ) -> None:
        del plan_digest
        if not jobs:
            raise ValueError("batch restore preflight has no jobs")
        if tuple(descriptor_columns) != tuple(
            DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS
        ):
            raise ValueError("batch restore preflight descriptor ABI changed")
        if (
            not isinstance(descriptors, torch.Tensor)
            or descriptors.device.type != "cuda"
            or descriptors.dtype != torch.int64
            or descriptors.ndim != 2
            or not descriptors.is_contiguous()
            or int(descriptors.shape[0]) != len(jobs)
            or int(descriptors.shape[1]) < _REQUIRED_DESCRIPTOR_WIDTH
        ):
            raise ValueError("batch restore preflight descriptors changed")
        if (
            not isinstance(device_status, torch.Tensor)
            or device_status.device != descriptors.device
            or device_status.dtype != torch.int32
            or tuple(device_status.shape) != (1,)
            or not device_status.is_contiguous()
            or not isinstance(host_status, torch.Tensor)
            or host_status.device.type != "cpu"
            or host_status.dtype != torch.int32
            or tuple(host_status.shape) != (1,)
            or not isinstance(device_collision_table, torch.Tensor)
            or device_collision_table.device != descriptors.device
            or device_collision_table.dtype != torch.int32
            or device_collision_table.ndim != 1
            or not device_collision_table.is_contiguous()
        ):
            raise ValueError("batch restore preflight status workspace changed")

        num_jobs = len(jobs)
        max_rows = max(int(getattr(job, "count", 0)) for job in jobs)
        width = int(descriptors.shape[1])
        _bounds_preflight_pointer_table_block32_kernel[
            (
                num_jobs,
                triton.cdiv(max_rows, _BOUNDS_PREFLIGHT_BLOCK_ROWS),
            )
        ](
            descriptors,
            device_status,
            NUM_JOBS=num_jobs,
            MAX_ROWS=max_rows,
            DESCRIPTOR_WIDTH=width,
            BLOCK_ROWS=_BOUNDS_PREFLIGHT_BLOCK_ROWS,
            num_warps=1,
            num_stages=1,
        )

        required_entries = sum(
            int(getattr(job, "count", 0))
            * max(
                1,
                int(
                    getattr(
                        getattr(job, "metadata", None),
                        "state_required_groups",
                        getattr(job, "required_groups", 0),
                    )
                ),
            )
            for job in jobs
        )
        if (
            type(max_validation_entries) is not int
            or required_entries > max_validation_entries
            or num_jobs * max_rows * 2 > 2_147_483_647
            or int(device_collision_table.numel()) < 2 * required_entries
            or int(device_collision_table.numel())
            & (int(device_collision_table.numel()) - 1)
        ):
            raise MemoryError(
                "restore batch collision-proof workspace is undersized"
            )
        _sort_destination_overlap_preflight(
            jobs=jobs,
            descriptors=descriptors,
            device_status=device_status,
            device_collision_table=device_collision_table,
            max_rows=max_rows,
        )

        host_status.copy_(device_status, non_blocking=True)
        torch.cuda.current_stream(descriptors.device).synchronize()


class _PackedBatchCallback:
    @torch.no_grad()
    def __call__(
        self,
        *,
        family,
        jobs,
        descriptors,
        descriptor_columns,
        plan_digest,
    ) -> None:
        del plan_digest
        num_jobs, max_rows, width = _validate_callback_inputs(
            family=family,
            expected_family=RESTORE_FAMILY_PACKED,
            jobs=jobs,
            descriptors=descriptors,
            descriptor_columns=descriptor_columns,
        )
        _packed_restore_pointer_table_kernel[(num_jobs * max_rows,)](
            descriptors,
            NUM_JOBS=num_jobs,
            MAX_ROWS=max_rows,
            DESCRIPTOR_WIDTH=width,
            num_warps=4,
            num_stages=1,
        )


class _IndexerBatchCallback:
    @torch.no_grad()
    def __call__(
        self,
        *,
        family,
        jobs,
        descriptors,
        descriptor_columns,
        plan_digest,
    ) -> None:
        del plan_digest
        num_jobs, max_rows, width = _validate_callback_inputs(
            family=family,
            expected_family=RESTORE_FAMILY_INDEXER,
            jobs=jobs,
            descriptors=descriptors,
            descriptor_columns=descriptor_columns,
        )
        _indexer_restore_pointer_table_kernel[(num_jobs * max_rows,)](
            descriptors,
            NUM_JOBS=num_jobs,
            MAX_ROWS=max_rows,
            DESCRIPTOR_WIDTH=width,
            num_warps=4,
            num_stages=1,
        )


class _StateBatchCallback:
    @torch.no_grad()
    def __call__(
        self,
        *,
        family,
        jobs,
        descriptors,
        descriptor_columns,
        plan_digest,
    ) -> None:
        del plan_digest
        num_jobs, max_rows, width = _validate_callback_inputs(
            family=family,
            expected_family=RESTORE_FAMILY_STATE,
            jobs=jobs,
            descriptors=descriptors,
            descriptor_columns=descriptor_columns,
        )
        block_bytes = 1024
        record_widths = map(
            lambda current: int(getattr(current, "record_bytes")), jobs
        )
        max_record_bytes = max(record_widths)
        max_chunks = (max_record_bytes + block_bytes - 1) // block_bytes
        _state_restore_pointer_table_kernel[
            (num_jobs * max_rows * max_chunks,)
        ](
            descriptors,
            NUM_JOBS=num_jobs,
            MAX_ROWS=max_rows,
            MAX_CHUNKS=max_chunks,
            DESCRIPTOR_WIDTH=width,
            BLOCK_BYTES=block_bytes,
            num_warps=8,
            num_stages=1,
        )


_ORACLE_SEAL = object()
_ORACLE_LOCK = RLock()
_ORACLE_CACHE: dict[tuple[object, ...], "OracleCertificate"] = {}
_ISSUED_CERTIFICATE_IDS: set[int] = set()
_ORACLE_SEED = 20260815
DSV4_RESTORE_JOBS_PER_REQUEST = 18 * 5 + 19 * 3
DSV4_RESTORE_EXTRA_DESCRIPTOR_COLUMNS = 8
if len(SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS) != DSV4_RESTORE_EXTRA_DESCRIPTOR_COLUMNS:
    raise RuntimeError("DSV4 restore model descriptor ABI changed")


def dsv4_restore_workspace_requirements(
    *, max_requests: int, max_batch_rows: int
) -> Mapping[str, int]:
    """Conservative capacity for one *whole-batch* chunked forward.

    ``max_batch_rows`` is already summed across requests.  Request count only
    enlarges the job table and the rounding slack for compressed/checkpoint
    rows; it must never multiply the full row term a second time.
    """

    if type(max_requests) is not int or max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if type(max_batch_rows) is not int or max_batch_rows <= 0:
        raise ValueError("max_batch_rows must be positive")

    def summed_ceil(divisor: int) -> int:
        return (
            max_batch_rows + (divisor - 1) * max_requests
        ) // divisor

    c4_rows = summed_ceil(4)
    c128_rows = summed_ceil(128)
    checkpoints = summed_ceil(512)
    max_validation_entries = (
        37 * max_batch_rows
        + 36 * c4_rows
        + 19 * c128_rows
        + 72 * checkpoints
        + 19 * checkpoints
    )
    return MappingProxyType(
        {
            "max_jobs": DSV4_RESTORE_JOBS_PER_REQUEST * max_requests,
            "max_extra_descriptor_columns": (
                DSV4_RESTORE_EXTRA_DESCRIPTOR_COLUMNS
            ),
            "max_validation_entries": max_validation_entries,
        }
    )


def _source_sha256(path: object) -> str:
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


def _freeze_report(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_report(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_report(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"oracle report contains unsupported {type(value)!r}")


def _thaw_report(value):
    if isinstance(value, Mapping):
        return {str(key): _thaw_report(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_report(item) for item in value]
    return value


def _normalize_cuda_device(device: object) -> torch.device:
    result = torch.device(device)
    if result.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("batch restore certification requires a CUDA device")
    index = torch.cuda.current_device() if result.index is None else int(result.index)
    return torch.device("cuda", index)


def _device_fingerprint(device: torch.device) -> tuple[object, ...]:
    properties = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    return (
        int(device.index),
        str(properties.name),
        capability,
        str(getattr(properties, "uuid", "")),
        int(getattr(properties, "total_memory", 0)),
    )


@dataclass(frozen=True)
class OracleCertificate:
    """Process-local proof for one exact source/runtime/target-GPU tuple."""

    kernel_source_sha256: str
    oracle_source_sha256: str
    torch_version: str
    triton_version: str
    cuda_runtime_version: str
    device_index: int
    device_name: str
    device_capability: tuple[int, int]
    device_uuid: str
    device_total_memory: int
    seed: int
    report: Mapping[str, Any]
    common_provider_digest: str
    manifest_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _ORACLE_SEAL:
            raise ValueError("OracleCertificate cannot be constructed externally")
        if not bool(self.report.get("strict_pass", False)):
            raise ValueError("OracleCertificate requires a strict oracle pass")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kernel_source_sha256": self.kernel_source_sha256,
            "oracle_source_sha256": self.oracle_source_sha256,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "device_capability": list(self.device_capability),
            "device_uuid": self.device_uuid,
            "device_total_memory": self.device_total_memory,
            "seed": self.seed,
            "common_provider_digest": self.common_provider_digest,
            "manifest_sha256": self.manifest_sha256,
            "report": _thaw_report(self.report),
        }


@dataclass(frozen=True)
class TritonBatchRestoreProvider:
    certificate: OracleCertificate
    kernels: Mapping[str, DeviceRestoreBatchKernel]
    preflight_kernel: DeviceRestoreBatchPreflightKernel
    max_extra_descriptor_columns: int
    jobs_per_request: int = DSV4_RESTORE_JOBS_PER_REQUEST

    @property
    def provider_token(self) -> str:
        """Rank-local identity; includes CUDA index and device UUID."""

        return self.certificate.manifest_sha256

    @property
    def rank_local_provider_token(self) -> str:
        return self.certificate.manifest_sha256

    @property
    def common_provider_digest(self) -> str:
        """TP-common identity; excludes rank-local index/UUID."""

        return self.certificate.common_provider_digest

    @property
    def common_provider_token(self) -> str:
        return self.certificate.common_provider_digest

    def workspace_max_jobs(self, *, max_requests: int) -> int:
        if type(max_requests) is not int or max_requests <= 0:
            raise ValueError("max_requests must be positive")
        return self.jobs_per_request * max_requests

    def workspace_requirements(
        self, *, max_requests: int, max_batch_rows: int
    ) -> Mapping[str, int]:
        requirements = dsv4_restore_workspace_requirements(
            max_requests=max_requests, max_batch_rows=max_batch_rows
        )
        if (
            requirements["max_extra_descriptor_columns"]
            != self.max_extra_descriptor_columns
        ):
            raise AssertionError("provider descriptor ABI changed")
        return requirements


def _oracle_module():
    from sglang.srt.layers.attention.redknot import (
        probe_dsv4_shared_latent_batch_kernels as oracle,
    )

    return oracle


def _oracle_environment(device: torch.device) -> tuple[object, ...]:
    oracle = _oracle_module()
    return (
        _source_sha256(__file__),
        _source_sha256(oracle.__file__),
        str(torch.__version__),
        str(triton.__version__),
        str(torch.version.cuda or ""),
    ) + _device_fingerprint(device)


def get_cached_target_gpu_batch_restore_oracle(
    device: object,
) -> Optional[OracleCertificate]:
    device = _normalize_cuda_device(device)
    key = _oracle_environment(device)
    with _ORACLE_LOCK:
        return _ORACLE_CACHE.get(key)


def run_target_gpu_batch_restore_oracle(device: object) -> OracleCertificate:
    """JIT all five launches and issue a source/runtime/device-bound proof."""

    device = _normalize_cuda_device(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (8, 9) and capability[0] < 9:
        raise RuntimeError("batch restore oracle allows only SM89 or SM90+")
    key = _oracle_environment(device)
    with _ORACLE_LOCK:
        cached = _ORACLE_CACHE.get(key)
        if cached is not None:
            return cached

        callbacks = {
            RESTORE_FAMILY_PACKED: _PackedBatchCallback(),
            RESTORE_FAMILY_INDEXER: _IndexerBatchCallback(),
            RESTORE_FAMILY_STATE: _StateBatchCallback(),
        }
        preflight_callback = _BatchPreflightCallback()
        oracle = _oracle_module()
        with torch.cuda.device(device):
            report = oracle.run_oracle_callbacks(
                callbacks=callbacks,
                preflight_callback=preflight_callback,
                device=device,
                seed=_ORACLE_SEED,
            )
        if not bool(report.get("strict_pass", False)):
            raise RuntimeError("target-GPU batch restore strict oracle failed")
        report = dict(report)
        report["provider_contract"] = {
            "common_descriptor_columns": tuple(
                DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS
            ),
            "model_descriptor_columns": tuple(
                SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS
            ),
            "jobs_per_request": DSV4_RESTORE_JOBS_PER_REQUEST,
            "restore_families": tuple(
                (
                    RESTORE_FAMILY_PACKED,
                    RESTORE_FAMILY_INDEXER,
                    RESTORE_FAMILY_STATE,
                )
            ),
            "restore_launches": 3,
            "preflight_logical_phases": 2,
            "preflight_launch_accounting": (
                "logical_bounds_and_exact_address_sort_phases; framework sort "
                "may use multiple internal CUDA kernels"
            ),
            "preflight_collision_algorithm": "exact_physical_address_sort_v1",
            "preflight_destination_materializer": (
                "fused_triton_block32_v1"
            ),
            "validation_memsets": 1,
        }

        environment = {
            "kernel_source_sha256": key[0],
            "oracle_source_sha256": key[1],
            "torch_version": key[2],
            "triton_version": key[3],
            "cuda_runtime_version": key[4],
            "device_index": key[5],
            "device_name": key[6],
            "device_capability": key[7],
            "device_uuid": key[8],
            "device_total_memory": key[9],
            "seed": _ORACLE_SEED,
        }
        frozen_report = _freeze_report(report)
        evidence = _thaw_report(frozen_report)
        common_environment = tuple(
            (name, value)
            for name, value in environment.items()
            if name not in ("device_index", "device_uuid")
        )
        common_provider_digest = "sha256:" + sha256(
            repr((common_environment, evidence)).encode("utf-8")
        ).hexdigest()
        manifest_payload = repr(
            (tuple(environment.items()), common_provider_digest, evidence)
        ).encode("utf-8")
        certificate = OracleCertificate(
            **environment,
            report=frozen_report,
            common_provider_digest=common_provider_digest,
            manifest_sha256="sha256:" + sha256(manifest_payload).hexdigest(),
            _seal=_ORACLE_SEAL,
        )
        _ISSUED_CERTIFICATE_IDS.add(id(certificate))
        _ORACLE_CACHE[key] = certificate
        return certificate


def get_cached_hopper_batch_restore_oracle(
    device: object,
) -> Optional[OracleCertificate]:
    """Compatibility alias; target certification also supports SM89."""

    return get_cached_target_gpu_batch_restore_oracle(device)


def run_hopper_batch_restore_oracle(device: object) -> OracleCertificate:
    """Compatibility alias for :func:`run_target_gpu_batch_restore_oracle`."""

    return run_target_gpu_batch_restore_oracle(device)


def _validate_oracle_certificate(
    certificate: OracleCertificate, device: object
) -> torch.device:
    device = _normalize_cuda_device(device)
    if (
        not isinstance(certificate, OracleCertificate)
        or certificate._seal is not _ORACLE_SEAL
        or id(certificate) not in _ISSUED_CERTIFICATE_IDS
    ):
        raise ValueError("a live in-process OracleCertificate is required")
    key = _oracle_environment(device)
    expected = (
        certificate.kernel_source_sha256,
        certificate.oracle_source_sha256,
        certificate.torch_version,
        certificate.triton_version,
        certificate.cuda_runtime_version,
        certificate.device_index,
        certificate.device_name,
        certificate.device_capability,
        certificate.device_uuid,
        certificate.device_total_memory,
    )
    if key != expected or not bool(certificate.report.get("strict_pass", False)):
        raise ValueError("OracleCertificate does not match this runtime/device")
    return device


def build_triton_batch_restore_kernels(
    *, certificate: OracleCertificate, device: object = None
) -> Mapping[str, DeviceRestoreBatchKernel]:
    """Return three certified callbacks; a bool cannot bypass the oracle."""

    if not isinstance(certificate, OracleCertificate):
        raise ValueError("a live in-process OracleCertificate is required")
    if device is None:
        device = torch.device("cuda", certificate.device_index)
    _validate_oracle_certificate(certificate, device)
    return {
        RESTORE_FAMILY_PACKED: DeviceRestoreBatchKernel(
            family=RESTORE_FAMILY_PACKED,
            callback=_PackedBatchCallback(),
            production_certified=True,
            max_launches_per_call=1,
        ),
        RESTORE_FAMILY_INDEXER: DeviceRestoreBatchKernel(
            family=RESTORE_FAMILY_INDEXER,
            callback=_IndexerBatchCallback(),
            production_certified=True,
            max_launches_per_call=1,
        ),
        RESTORE_FAMILY_STATE: DeviceRestoreBatchKernel(
            family=RESTORE_FAMILY_STATE,
            callback=_StateBatchCallback(),
            production_certified=True,
            max_launches_per_call=1,
        ),
    }


def build_triton_batch_restore_preflight(
    *, certificate: OracleCertificate, device: object = None
) -> DeviceRestoreBatchPreflightKernel:
    if not isinstance(certificate, OracleCertificate):
        raise ValueError("a live in-process OracleCertificate is required")
    if device is None:
        device = torch.device("cuda", certificate.device_index)
    _validate_oracle_certificate(certificate, device)
    return DeviceRestoreBatchPreflightKernel(
        callback=_BatchPreflightCallback(),
        production_certified=True,
        max_launches_per_call=2,
    )


def build_triton_batch_restore_provider(
    *, certificate: OracleCertificate, device: object = None
) -> TritonBatchRestoreProvider:
    if device is None:
        device = torch.device("cuda", certificate.device_index)
    kernels = build_triton_batch_restore_kernels(
        certificate=certificate, device=device
    )
    preflight_kernel = build_triton_batch_restore_preflight(
        certificate=certificate, device=device
    )
    return TritonBatchRestoreProvider(
        certificate=certificate,
        kernels=MappingProxyType(dict(kernels)),
        preflight_kernel=preflight_kernel,
        max_extra_descriptor_columns=len(
            SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS
        ),
    )


__all__ = [
    "DSV4_RESTORE_JOBS_PER_REQUEST",
    "DSV4_RESTORE_EXTRA_DESCRIPTOR_COLUMNS",
    "OracleCertificate",
    "TritonBatchRestoreProvider",
    "build_triton_batch_restore_kernels",
    "build_triton_batch_restore_preflight",
    "build_triton_batch_restore_provider",
    "dsv4_restore_workspace_requirements",
    "get_cached_hopper_batch_restore_oracle",
    "get_cached_target_gpu_batch_restore_oracle",
    "run_hopper_batch_restore_oracle",
    "run_target_gpu_batch_restore_oracle",
]
