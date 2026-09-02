#!/usr/bin/env python3
"""Target-GPU JIT/numerical oracle for the batched restore kernels.

Run this inside the target SGLang environment after the three implementation
modules have been installed under ``sglang.srt.layers.attention.redknot``::

    TRITON_DEBUG=1 python probe_dsv4_shared_latent_batch_kernels.py --mode strict

The probe makes two or three independent layer allocations per family, runs
the existing per-layer callbacks as the oracle, then runs one pointer-table
callback for the complete family.  It therefore exercises dynamic source,
target, index-vector, position-vector, and RoPE-table pointers as well as
different row counts and state record widths.  ``strict`` requires complete
target buffers to be bitwise identical.  ``tolerance`` is diagnostic only: it
still requires all copied bytes and state bytes to be exact, but allows the
BF16 RoPE and dequantized Indexer values the tolerances printed in the JSON.

This file deliberately does not set ``oracle_verified=True``.  Certification
is a deployment decision made only after this process exits successfully on
the exact SM89/SM90+/PyTorch/Triton/SGLang build used for serving.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import torch
import triton

from sglang.srt.layers.attention.redknot.dsv4_shared_latent_cache import (
    INDEXER_POSITION_SEMANTICS,
    PACKED_LATENT_POSITION_SEMANTICS,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
    DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS,
    DOMAIN_C128_ATTENTION_STATE,
    DOMAIN_C4,
    DOMAIN_C4_ATTENTION_STATE,
    DOMAIN_INDEXER,
    DOMAIN_INDEXER_STATE,
    DOMAIN_SWA,
    RESTORE_FAMILY_INDEXER,
    RESTORE_FAMILY_PACKED,
    RESTORE_FAMILY_STATE,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_sglang import (
    ATTENTION_STATE_POSITION_SEMANTICS,
    INDEXER_STATE_POSITION_SEMANTICS,
    SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS,
    _indexer_restore_kernel,
    _packed_restore_kernel,
    _state_restore_kernel,
)


_PACKED_BYTES = 584
_PACKED_KV_BYTES = 576
_PACKED_NOPE_BYTES = 448
_INDEXER_SOURCE_BYTES = 256
_DESCRIPTOR_WIDTH = len(DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS) + len(
    SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS
)


@dataclass
class _Case:
    domain: str
    layer_id: int
    source: torch.Tensor
    source_indices: torch.Tensor
    target: torch.Tensor
    reference: torch.Tensor
    target_slots: torch.Tensor
    output_rows: torch.Tensor
    positions: torch.Tensor
    page_size: int = 0
    group_width: int = 0
    required_groups: int = 0
    freqs: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return int(self.source_indices.numel())

    @property
    def record_bytes(self) -> int:
        return int(self.source.shape[1])

    @property
    def target_row_bytes(self) -> int:
        return int(self.target.shape[1]) * int(self.target.element_size())

    @property
    def selected_slots(self) -> torch.Tensor:
        return self.target_slots.index_select(0, self.output_rows)


def _frequency_table(
    *, rows: int, device: torch.device, phase_offset: float
) -> torch.Tensor:
    position = torch.arange(rows, device=device, dtype=torch.float32)[:, None]
    pair = torch.arange(32, device=device, dtype=torch.float32)[None, :]
    inv_frequency = torch.exp(-pair * (9.210340371976184 / 32.0))
    phase = position * inv_frequency + float(phase_offset)
    return torch.polar(torch.ones_like(phase), phase).to(torch.complex64).contiguous()


def _padded_packed_page_bytes(page_size: int) -> int:
    raw = int(page_size) * _PACKED_BYTES
    return ((raw + _PACKED_KV_BYTES - 1) // _PACKED_KV_BYTES) * _PACKED_KV_BYTES


def _source_packed(*, rows: int, device: torch.device) -> torch.Tensor:
    result = torch.randint(
        0, 256, (rows, _PACKED_BYTES), dtype=torch.uint8, device=device
    )
    rope = (0.20 * torch.randn((rows, 64), device=device)).to(torch.bfloat16)
    result[:, _PACKED_NOPE_BYTES:_PACKED_KV_BYTES].copy_(
        rope.contiguous().view(torch.uint8).view(rows, -1)
    )
    return result


def _source_indexer(*, rows: int, device: torch.device) -> torch.Tensor:
    canonical = (0.20 * torch.randn((rows, 128), device=device)).float()
    canonical[0].zero_()
    canonical[1].zero_()
    canonical[1, 0] = 1.0
    canonical[2] = torch.where(
        torch.arange(128, device=device) % 2 == 0,
        torch.ones(128, device=device),
        -torch.ones(128, device=device),
    )
    canonical[3].fill_(1.0e-6)
    canonical[4] = torch.linspace(-1.0, 1.0, 128, device=device)
    canonical = canonical.to(torch.bfloat16)
    return canonical.contiguous().view(torch.uint8).view(rows, -1).clone()


def _descriptor_row(case: _Case, ordinal: int) -> List[int]:
    freqs_ptr = int(case.freqs.data_ptr()) if case.freqs is not None else 0
    freqs_row_bytes = (
        int(case.freqs.stride(0)) * int(case.freqs.element_size())
        if case.freqs is not None
        else 0
    )
    freqs_rows = int(case.freqs.shape[0]) if case.freqs is not None else 0
    return [
        int(ordinal),
        0,
        int(case.layer_id),
        int(case.count),
        int(case.source.data_ptr()),
        int(case.source_indices.data_ptr()),
        int(case.target.data_ptr()),
        int(case.target_slots.data_ptr()),
        int(case.output_rows.data_ptr()),
        int(case.positions.data_ptr()),
        int(case.record_bytes),
        int(case.source.shape[0]),
        int(case.positions.numel()),
        int(case.page_size),
        int(case.target_row_bytes),
        int(case.group_width),
        int(case.required_groups),
        freqs_ptr,
        freqs_row_bytes,
        freqs_rows,
        int(case.target.shape[0]),
    ]


def _family_descriptors(cases: Sequence[_Case]) -> torch.Tensor:
    rows = [_descriptor_row(case, ordinal) for ordinal, case in enumerate(cases)]
    if any(len(row) != _DESCRIPTOR_WIDTH for row in rows):
        raise AssertionError("oracle descriptor ABI does not match production")
    return torch.tensor(
        rows, dtype=torch.int64, device=cases[0].target.device
    ).contiguous()


def _family_jobs(cases: Sequence[_Case]) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            count=case.count,
            record_bytes=case.record_bytes,
            family=(
                RESTORE_FAMILY_STATE
                if case.group_width
                else (
                    RESTORE_FAMILY_INDEXER
                    if case.domain == DOMAIN_INDEXER
                    else RESTORE_FAMILY_PACKED
                )
            ),
            required_groups=case.required_groups,
            target_cache=case.target,
            target_slots=case.target_slots,
            output_rows=case.output_rows,
            positions=case.positions,
            metadata=SimpleNamespace(
                target_page_size=case.page_size,
                target_row_bytes=case.target_row_bytes,
                target_physical_rows=int(case.target.shape[0]),
                state_group_width=case.group_width,
                state_required_groups=case.required_groups,
                freqs_cis=case.freqs,
            ),
        )
        for case in cases
    )


def _run_preflight(preflight_callback, cases: Sequence[_Case]) -> int:
    descriptors = _family_descriptors(cases)
    status_device = torch.zeros(1, dtype=torch.int32, device=descriptors.device)
    status_host = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
    required_entries = sum(
        case.count * max(1, case.required_groups) for case in cases
    )
    table_size = 1
    while table_size < 2 * required_entries:
        table_size *= 2
    collision_device = torch.zeros(
        table_size, dtype=torch.int32, device=descriptors.device
    )
    preflight_callback(
        jobs=_family_jobs(cases),
        descriptors=descriptors,
        descriptor_columns=DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS,
        device_status=status_device,
        host_status=status_host,
        device_collision_table=collision_device,
        max_validation_entries=required_entries,
        plan_digest="hopper-oracle-preflight",
    )
    return int(status_host[0])


def _run_owner_id_hash_collision_oracle(
    device: torch.device, preflight_callback
) -> Dict[str, Any]:
    """Force one hash collision whose complete destination keys differ."""

    page_size = 4
    target = torch.full(
        (4, _padded_packed_page_bytes(page_size)),
        0xCD,
        dtype=torch.uint8,
        device=device,
    )
    table_size = 4  # two destination entries at the certified <= 0.5 load
    buckets: Dict[int, tuple[int, int]] = {}
    collision: tuple[int, int, int, int] | None = None
    target_address = int(target.data_ptr())
    target_row_bytes = int(target.shape[1])
    for slot in range(int(target.shape[0]) * page_size):
        page, token = divmod(slot, page_size)
        key = (
            target_address
            + page * target_row_bytes
            + token * _PACKED_KV_BYTES
        )
        bucket = ((key >> 4) ^ (key >> 32)) & (table_size - 1)
        incumbent = buckets.get(bucket)
        if incumbent is not None and incumbent[1] != key:
            collision = (incumbent[0], slot, incumbent[1], key)
            break
        buckets[bucket] = (slot, key)
    if collision is None:
        raise AssertionError("failed to construct an owner-id hash collision")

    left_slot, right_slot, left_key, right_key = collision
    case = _Case(
        domain=DOMAIN_SWA,
        layer_id=3,
        source=_source_packed(rows=2, device=device),
        source_indices=torch.tensor([0, 1], device=device),
        target=target,
        reference=target.clone(),
        target_slots=torch.tensor([left_slot, right_slot], device=device),
        output_rows=torch.tensor([0, 1], device=device),
        positions=torch.tensor([0, 1], device=device),
        page_size=page_size,
        freqs=_frequency_table(rows=2, device=device, phase_offset=0.0),
    )
    status = _run_preflight(preflight_callback, (case,))
    return {
        "status": status,
        "distinct_full_keys": bool(left_key != right_key),
        "same_initial_bucket": bool(
            (((left_key >> 4) ^ (left_key >> 32)) & (table_size - 1))
            == (((right_key >> 4) ^ (right_key >> 32)) & (table_size - 1))
        ),
        "pass": bool(status == 0 and left_key != right_key),
    }


def _run_batched_family(
    family: str,
    cases: Sequence[_Case],
    callbacks: Dict[str, Any],
    preflight_callback,
) -> None:
    descriptors = _family_descriptors(cases)
    jobs = _family_jobs(cases)
    status = _run_preflight(preflight_callback, cases)
    if status:
        raise AssertionError(f"positive oracle preflight failed with code {status}")
    callback = callbacks[family]
    callback(
        family=family,
        jobs=jobs,
        descriptors=descriptors,
        descriptor_columns=DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS,
        plan_digest="hopper-oracle",
    )


def _gather_packed(
    cache: torch.Tensor, slots: torch.Tensor, page_size: int
) -> torch.Tensor:
    flat = cache.view(torch.uint8).flatten()
    slots = slots.long()
    page = slots // page_size
    token = slots % page_size
    row_bytes = int(cache.shape[1]) * int(cache.element_size())
    kv_base = page * row_bytes + token * _PACKED_KV_BYTES
    scale_base = page * row_bytes + page_size * _PACKED_KV_BYTES + token * 8
    kv_index = kv_base[:, None] + torch.arange(
        _PACKED_KV_BYTES, device=cache.device
    )
    scale_index = scale_base[:, None] + torch.arange(8, device=cache.device)
    return torch.cat((flat[kv_index], flat[scale_index]), dim=1)


def _gather_indexer(
    cache: torch.Tensor, slots: torch.Tensor, page_size: int
) -> torch.Tensor:
    flat = cache.view(torch.uint8).flatten()
    slots = slots.long()
    page = slots // page_size
    token = slots % page_size
    row_bytes = int(cache.shape[1]) * int(cache.element_size())
    key_base = page * row_bytes + token * 128
    scale_base = page * row_bytes + page_size * 128 + token * 4
    key_index = key_base[:, None] + torch.arange(128, device=cache.device)
    scale_index = scale_base[:, None] + torch.arange(4, device=cache.device)
    return torch.cat((flat[key_index], flat[scale_index]), dim=1)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def _bool(value: torch.Tensor | bool) -> bool:
    return bool(value if isinstance(value, bool) else value.item())


def _packed_cases(device: torch.device) -> List[_Case]:
    cases = []
    specifications = (
        (DOMAIN_SWA, 3, 3, [5, 1, 4], [0, 3, 6]),
        (DOMAIN_C4, 4, 2, [2, 0], [1, 5]),
    )
    for ordinal, (domain, layer, count, source_index, output_row) in enumerate(
        specifications
    ):
        page_size = 4
        target = torch.full(
            (4, _padded_packed_page_bytes(page_size)),
            0xCD,
            dtype=torch.uint8,
            device=device,
        )
        cases.append(
            _Case(
                domain=domain,
                layer_id=layer,
                source=_source_packed(rows=6, device=device),
                source_indices=torch.tensor(source_index, device=device),
                target=target,
                reference=target.clone(),
                target_slots=torch.tensor(
                    [0, 5, 2, 11, 4, 14, 9], device=device
                ),
                output_rows=torch.tensor(output_row, device=device),
                positions=torch.tensor(
                    [0, 7, 13, 19, 23, 31, 47], device=device
                ),
                page_size=page_size,
                freqs=_frequency_table(
                    rows=64, device=device, phase_offset=ordinal * 0.03125
                ),
            )
        )
        if cases[-1].count != count:
            raise AssertionError("packed oracle count changed")
    return cases


def _run_packed_oracle(
    device: torch.device, callbacks: Dict[str, Any], preflight_callback
) -> Dict[str, Any]:
    cases = _packed_cases(device)
    native = _packed_restore_kernel(
        {case.layer_id: case.freqs for case in cases},
        {(case.domain, case.layer_id): case.page_size for case in cases},
    )
    for case in cases:
        native(
            domain=case.domain,
            layer_id=case.layer_id,
            source_bank=case.source,
            source_indices=case.source_indices,
            target_cache=case.reference,
            target_slots=case.target_slots,
            output_rows=case.output_rows,
            positions=case.positions,
            scratch=torch.empty(
                (case.count, case.record_bytes), dtype=torch.uint8, device=device
            ),
            slot_scratch=torch.empty(case.count, dtype=torch.long, device=device),
            position_semantics=PACKED_LATENT_POSITION_SEMANTICS,
        )
    _run_batched_family(
        RESTORE_FAMILY_PACKED, cases, callbacks, preflight_callback
    )
    torch.cuda.synchronize(device)

    full_exact = True
    copied_exact = True
    rope_close = True
    max_rope_error = 0.0
    for case in cases:
        full_exact &= torch.equal(case.target, case.reference)
        got = _gather_packed(case.target, case.selected_slots, case.page_size)
        expected = _gather_packed(
            case.reference, case.selected_slots, case.page_size
        )
        copied_exact &= torch.equal(
            got[:, :_PACKED_NOPE_BYTES], expected[:, :_PACKED_NOPE_BYTES]
        ) and torch.equal(got[:, _PACKED_KV_BYTES:], expected[:, _PACKED_KV_BYTES:])
        got_rope = (
            got[:, _PACKED_NOPE_BYTES:_PACKED_KV_BYTES]
            .contiguous()
            .view(torch.bfloat16)
        )
        expected_rope = (
            expected[:, _PACKED_NOPE_BYTES:_PACKED_KV_BYTES]
            .contiguous()
            .view(torch.bfloat16)
        )
        max_rope_error = max(max_rope_error, _max_abs(got_rope, expected_rope))
        rope_close &= torch.allclose(
            got_rope,
            expected_rope,
            rtol=0.0078125,
            atol=0.00390625,
            equal_nan=True,
        )
    return {
        "jobs": len(cases),
        "rows": sum(case.count for case in cases),
        "full_target_bitwise": bool(full_exact),
        "nope_and_scale_bitwise": bool(copied_exact),
        "rope_close": bool(rope_close),
        "rope_max_abs_error": max_rope_error,
        "tolerance_pass": bool(copied_exact and rope_close),
    }


def _indexer_cases(device: torch.device) -> List[_Case]:
    cases = []
    specifications = (
        (5, [5, 0, 3, 2], [0, 2, 4, 6]),
        (6, [1, 4], [1, 5]),
    )
    for ordinal, (layer, source_index, output_row) in enumerate(specifications):
        page_size = 4
        target = torch.full(
            (4, page_size * 132), 0xCD, dtype=torch.uint8, device=device
        )
        cases.append(
            _Case(
                domain=DOMAIN_INDEXER,
                layer_id=layer,
                source=_source_indexer(rows=6, device=device),
                source_indices=torch.tensor(source_index, device=device),
                target=target,
                reference=target.clone(),
                target_slots=torch.tensor(
                    [1, 6, 2, 13, 4, 10, 15], device=device
                ),
                output_rows=torch.tensor(output_row, device=device),
                positions=torch.tensor(
                    [2, 9, 15, 22, 29, 37, 55], device=device
                ),
                page_size=page_size,
                freqs=_frequency_table(
                    rows=64, device=device, phase_offset=ordinal * 0.046875
                ),
            )
        )
    return cases


def _dequantize_indexer(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key = packed[:, :128].contiguous().view(torch.float8_e4m3fn).float()
    scale = packed[:, 128:].contiguous().view(torch.float32).reshape(-1, 1)
    return key * scale, scale


def _run_indexer_oracle(
    device: torch.device, callbacks: Dict[str, Any], preflight_callback
) -> Dict[str, Any]:
    cases = _indexer_cases(device)
    native = _indexer_restore_kernel(
        {case.layer_id: case.freqs for case in cases},
        {case.layer_id: case.page_size for case in cases},
    )
    for case in cases:
        native(
            domain=case.domain,
            layer_id=case.layer_id,
            source_bank=case.source,
            source_indices=case.source_indices,
            target_cache=case.reference,
            target_slots=case.target_slots,
            output_rows=case.output_rows,
            positions=case.positions,
            scratch=torch.empty(
                (case.count, case.record_bytes), dtype=torch.uint8, device=device
            ),
            slot_scratch=torch.empty(case.count, dtype=torch.long, device=device),
            position_semantics=INDEXER_POSITION_SEMANTICS,
        )
    _run_batched_family(
        RESTORE_FAMILY_INDEXER, cases, callbacks, preflight_callback
    )
    torch.cuda.synchronize(device)

    full_exact = True
    key_exact = True
    scale_close = True
    dequant_close = True
    max_scale_error = 0.0
    max_dequant_error = 0.0
    for case in cases:
        full_exact &= torch.equal(case.target, case.reference)
        got = _gather_indexer(case.target, case.selected_slots, case.page_size)
        expected = _gather_indexer(
            case.reference, case.selected_slots, case.page_size
        )
        key_exact &= torch.equal(got[:, :128], expected[:, :128])
        got_value, got_scale = _dequantize_indexer(got)
        expected_value, expected_scale = _dequantize_indexer(expected)
        max_scale_error = max(max_scale_error, _max_abs(got_scale, expected_scale))
        max_dequant_error = max(
            max_dequant_error, _max_abs(got_value, expected_value)
        )
        scale_close &= torch.allclose(
            got_scale, expected_scale, rtol=1.0e-5, atol=1.0e-7
        )
        dequant_close &= torch.allclose(
            got_value, expected_value, rtol=0.03, atol=0.02
        )
    return {
        "jobs": len(cases),
        "rows": sum(case.count for case in cases),
        "full_target_bitwise": bool(full_exact),
        "fp8_key_bitwise": bool(key_exact),
        "scale_close": bool(scale_close),
        "scale_max_abs_error": max_scale_error,
        "dequant_close": bool(dequant_close),
        "dequant_max_abs_error": max_dequant_error,
        "tolerance_pass": bool(scale_close and dequant_close),
    }


def _state_source(
    *, rows: int, group_width: int, required_groups: int, state_width: int,
    device: torch.device
) -> torch.Tensor:
    values = torch.randn(
        (rows, required_groups * group_width, state_width),
        dtype=torch.float32,
        device=device,
    )
    return values.contiguous().view(torch.uint8).view(rows, -1).clone()


def _state_cases(device: torch.device) -> List[_Case]:
    specifications = (
        (DOMAIN_C4_ATTENTION_STATE, 7, 4, 2, 129, [3, 0, 4], [0, 2, 5]),
        (DOMAIN_INDEXER_STATE, 8, 4, 2, 17, [1, 4], [1, 4]),
        (DOMAIN_C128_ATTENTION_STATE, 9, 128, 1, 257, [2, 0], [0, 3]),
        (DOMAIN_C128_ATTENTION_STATE, 10, 1, 1, 257, [1, 3], [1, 4]),
    )
    cases = []
    for domain, layer, group_width, required, state_width, source_idx, rows in specifications:
        group_count = 16 if group_width == 4 else 4
        target = torch.full(
            (group_count * group_width, state_width),
            -3.25,
            dtype=torch.float32,
            device=device,
        )
        q_rows = 6
        if group_width == 4:
            slots = [3, 5, 7, 9, 11, 14]
        else:
            slots = [1, 2, 3, 2, 1, 3]
        cases.append(
            _Case(
                domain=domain,
                layer_id=layer,
                source=_state_source(
                    rows=5,
                    group_width=group_width,
                    required_groups=required,
                    state_width=state_width,
                    device=device,
                ),
                source_indices=torch.tensor(source_idx, device=device),
                target=target,
                reference=target.clone(),
                target_slots=torch.tensor(slots, device=device),
                output_rows=torch.tensor(rows, device=device),
                positions=torch.arange(q_rows, device=device, dtype=torch.long),
                group_width=group_width,
                required_groups=required,
            )
        )
    return cases


def _run_state_oracle(
    device: torch.device, callbacks: Dict[str, Any], preflight_callback
) -> Dict[str, Any]:
    cases = _state_cases(device)
    native = _state_restore_kernel()
    for case in cases:
        semantics = (
            INDEXER_STATE_POSITION_SEMANTICS
            if case.domain == DOMAIN_INDEXER_STATE
            else ATTENTION_STATE_POSITION_SEMANTICS
        )
        native(
            domain=case.domain,
            layer_id=case.layer_id,
            source_bank=case.source,
            source_indices=case.source_indices,
            target_cache=case.reference,
            target_slots=case.target_slots,
            output_rows=case.output_rows,
            positions=case.positions,
            scratch=torch.empty(
                (case.count, case.record_bytes), dtype=torch.uint8, device=device
            ),
            slot_scratch=torch.empty(case.count, dtype=torch.long, device=device),
            position_semantics=semantics,
        )
    _run_batched_family(
        RESTORE_FAMILY_STATE, cases, callbacks, preflight_callback
    )
    torch.cuda.synchronize(device)
    exact = all(torch.equal(case.target, case.reference) for case in cases)
    return {
        "jobs": len(cases),
        "rows": sum(case.count for case in cases),
        "record_bytes": [case.record_bytes for case in cases],
        "max_chunks_1024": max(
            (case.record_bytes + 1023) // 1024 for case in cases
        ),
        "full_target_bitwise": bool(exact),
        "tolerance_pass": bool(exact),
    }


def _run_negative_preflight_oracles(
    device: torch.device, preflight_callback
) -> Dict[str, Any]:
    """Prove invalid descriptors are rejected before any target mutation."""

    packed = _packed_cases(device)
    packed_target_before = tuple(case.target.clone() for case in packed)
    first = packed[0]
    selected_row = int(first.output_rows[0].item())

    original_position = int(first.positions[selected_row].item())
    first.positions[selected_row] = int(first.freqs.shape[0])
    position_status = _run_preflight(preflight_callback, packed)
    first.positions[selected_row] = original_position

    original_slot = int(first.target_slots[selected_row].item())
    first.target_slots[selected_row] = -1
    target_status = _run_preflight(preflight_callback, packed)
    first.target_slots[selected_row] = original_slot

    original_source = int(first.source_indices[0].item())
    first.source_indices[0] = int(first.source.shape[0])
    source_status = _run_preflight(preflight_callback, packed)
    first.source_indices[0] = original_source

    state = _state_cases(device)[0]
    cross_job = _Case(
        domain=state.domain,
        layer_id=state.layer_id,
        source=state.source,
        source_indices=torch.tensor([1], device=device),
        target=state.target,
        reference=state.reference,
        target_slots=torch.tensor([4, 8, 10, 12, 14, 15], device=device),
        output_rows=torch.tensor([0], device=device),
        positions=state.positions,
        group_width=state.group_width,
        required_groups=state.required_groups,
    )
    state_single = _Case(
        domain=state.domain,
        layer_id=state.layer_id,
        source=state.source,
        source_indices=state.source_indices[:1],
        target=state.target,
        reference=state.reference,
        target_slots=state.target_slots,
        output_rows=state.output_rows[:1],
        positions=state.positions,
        group_width=state.group_width,
        required_groups=state.required_groups,
    )
    state_target_before = state.target.clone()
    overlap_status = _run_preflight(
        preflight_callback, (state_single, cross_job)
    )

    targets_unchanged = all(
        torch.equal(case.target, before)
        for case, before in zip(packed, packed_target_before)
    ) and torch.equal(state.target, state_target_before)
    passed = (
        bool(position_status & 8)
        and bool(target_status & 4)
        and bool(source_status & 1)
        and bool(overlap_status & 16)
        and targets_unchanged
    )
    return {
        "position_status": position_status,
        "target_status": target_status,
        "source_status": source_status,
        "cross_job_state_overlap_status": overlap_status,
        "targets_unchanged": bool(targets_unchanged),
        "pass": bool(passed),
    }


def run_oracle_callbacks(
    *,
    callbacks: Dict[str, Any],
    preflight_callback,
    device: torch.device,
    seed: int = 20260815,
) -> Dict[str, Any]:
    """JIT and compare raw production callbacks; called by certificate issuer."""

    expected = {
        RESTORE_FAMILY_PACKED,
        RESTORE_FAMILY_INDEXER,
        RESTORE_FAMILY_STATE,
    }
    if set(callbacks) != expected or any(
        not callable(callbacks[family]) for family in expected
    ):
        raise ValueError("oracle requires the exact three raw callbacks")
    if not callable(preflight_callback):
        raise TypeError("oracle requires the aggregate preflight callback")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    results = {
        "packed": _run_packed_oracle(device, callbacks, preflight_callback),
        "indexer": _run_indexer_oracle(device, callbacks, preflight_callback),
        "state": _run_state_oracle(device, callbacks, preflight_callback),
        "owner_id_hash_collision": _run_owner_id_hash_collision_oracle(
            device, preflight_callback
        ),
        "negative_preflight": _run_negative_preflight_oracles(
            device, preflight_callback
        ),
    }
    results["strict_pass"] = all(
        results[family]["full_target_bitwise"]
        for family in ("packed", "indexer", "state")
    ) and (
        results["owner_id_hash_collision"]["pass"]
        and results["negative_preflight"]["pass"]
    )
    results["tolerance_pass"] = all(
        results[family]["tolerance_pass"]
        for family in ("packed", "indexer", "state")
    ) and (
        results["owner_id_hash_collision"]["pass"]
        and results["negative_preflight"]["pass"]
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("strict", "tolerance"),
        default="strict",
        help="strict requires complete bitwise equality; tolerance is diagnostic",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the batch-kernel oracle requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (8, 9) and major < 9:
        raise RuntimeError("the certification oracle allows only SM89 or SM90+")
    from sglang.srt.layers.attention.redknot.dsv4_shared_latent_batch_kernels import (
        run_target_gpu_batch_restore_oracle,
    )

    certificate = run_target_gpu_batch_restore_oracle(device)
    results = dict(certificate.report)
    selected = (
        bool(results["strict_pass"])
        if args.mode == "strict"
        else bool(results["tolerance_pass"])
    )
    payload = certificate.as_dict()
    payload["selected_mode"] = args.mode
    payload["selected_mode_pass"] = selected
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
