#!/usr/bin/env python3
"""Strict B300/SM103 oracle for Pro-0813 packed-cache RoPE relocation.

This exercises the active-architecture JIT offset/read/write kernels and the
native DeepSeek-V4 rotary kernel.  All cache-layout operations are compared to
independent Torch byte-address references; rotary results are compared to an
FP32 adjacent-pair complex reference with explicit BF16 rounding at each native
in-place kernel boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from sglang.jit_kernel.dsv4 import attn as jit_attn
from sglang.srt.layers import deepseek_v4_rope
from sglang.srt.layers.attention.redknot import dsv4_rope_reloc as reloc
from sglang.srt.layers.attention.redknot.pro0813 import profile


PRO_ROOT = Path("/workspace/RedKnot_Pro0813").resolve()
EXPECTED_PTXAS = Path("/usr/local/cuda/bin/ptxas").resolve()
PAGE_STRIDE_BYTES = 576
NOPE_BYTES = 448
ROPE_DIM = 64
ROPE_BYTES = 128
SCALE_BYTES = 8
PACKED_BYTES = 584


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _module_path(module: object) -> Path:
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    if not path.is_file() or PRO_ROOT not in path.parents:
        raise RuntimeError(f"module escaped isolated Pro checkout: {path}")
    return path


def _page_bytes(page_size: int) -> int:
    raw = page_size * PACKED_BYTES
    return ((raw + PAGE_STRIDE_BYTES - 1) // PAGE_STRIDE_BYTES) * PAGE_STRIDE_BYTES


def _indices(
    loc: torch.Tensor,
    *,
    page_size: int,
    row_bytes: int,
    relative_offset: int,
    width: int,
) -> torch.Tensor:
    slots = loc.long()
    page = slots // page_size
    token = slots % page_size
    base = page * row_bytes + token * PAGE_STRIDE_BYTES + relative_offset
    return base[:, None] + torch.arange(width, device=loc.device)


def _scale_indices(
    loc: torch.Tensor, *, page_size: int, row_bytes: int
) -> torch.Tensor:
    slots = loc.long()
    page = slots // page_size
    token = slots % page_size
    base = page * row_bytes + page_size * PAGE_STRIDE_BYTES + token * SCALE_BYTES
    return base[:, None] + torch.arange(SCALE_BYTES, device=loc.device)


def _read_rope_reference(
    buf: torch.Tensor, loc: torch.Tensor, page_size: int
) -> torch.Tensor:
    index = _indices(
        loc,
        page_size=page_size,
        row_bytes=int(buf.shape[1]),
        relative_offset=NOPE_BYTES,
        width=ROPE_BYTES,
    )
    return (
        buf.flatten()[index]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, ROPE_DIM)
    )


def _write_rope_reference(
    buf: torch.Tensor,
    loc: torch.Tensor,
    rope: torch.Tensor,
    page_size: int,
) -> None:
    index = _indices(
        loc,
        page_size=page_size,
        row_bytes=int(buf.shape[1]),
        relative_offset=NOPE_BYTES,
        width=ROPE_BYTES,
    )
    buf.flatten()[index] = (
        rope.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(-1, ROPE_BYTES)
    )


def _read_packed_reference(
    buf: torch.Tensor, loc: torch.Tensor, page_size: int
) -> torch.Tensor:
    row_bytes = int(buf.shape[1])
    kv_index = _indices(
        loc,
        page_size=page_size,
        row_bytes=row_bytes,
        relative_offset=0,
        width=PAGE_STRIDE_BYTES,
    )
    return torch.cat(
        (buf.flatten()[kv_index], buf.flatten()[_scale_indices(
            loc, page_size=page_size, row_bytes=row_bytes
        )]),
        dim=-1,
    )


def _write_packed_reference(
    buf: torch.Tensor,
    loc: torch.Tensor,
    packed: torch.Tensor,
    page_size: int,
) -> None:
    row_bytes = int(buf.shape[1])
    flat = buf.flatten()
    flat[_indices(
        loc,
        page_size=page_size,
        row_bytes=row_bytes,
        relative_offset=0,
        width=PAGE_STRIDE_BYTES,
    )] = packed[:, :PAGE_STRIDE_BYTES]
    flat[_scale_indices(loc, page_size=page_size, row_bytes=row_bytes)] = packed[
        :, PAGE_STRIDE_BYTES:
    ]


def _frequency_table(rows: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(rows, device=device, dtype=torch.float32)[:, None]
    pairs = torch.arange(32, device=device, dtype=torch.float32)[None, :]
    inv_frequency = torch.exp(-pairs * (9.210340371976184 / 32.0))
    phase = positions * inv_frequency + 0.03125
    return torch.polar(torch.ones_like(phase), phase).to(torch.complex64).contiguous()


def _rotate_reference(
    values: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    selected = freqs.index_select(0, positions.long()).to(torch.complex64)
    cosine = selected.real.float()
    sine = selected.imag.float()
    if inverse:
        sine = -sine
    paired = values.float().reshape(-1, 32, 2)
    real = paired[..., 0]
    imaginary = paired[..., 1]
    result = torch.stack(
        (
            real * cosine - imaginary * sine,
            real * sine + imaginary * cosine,
        ),
        dim=-1,
    )
    return result.reshape(-1, ROPE_DIM).to(torch.bfloat16)


def _assert_geometry() -> dict[str, object]:
    geometry = profile.PRO0813_TP8_GEOMETRY
    observed = {
        "variant": geometry.variant,
        "official_config_sha256": geometry.official_config_sha256,
        "geometry_digest": geometry.geometry_digest,
        "num_target_layers": geometry.num_target_layers,
        "num_attention_heads": geometry.num_attention_heads,
        "qk_rope_head_dim": geometry.qk_rope_head_dim,
        "q_lora_rank": geometry.q_lora_rank,
        "tp_size": geometry.tp_size,
        "heads_per_tp_rank": geometry.heads_per_tp_rank,
    }
    expected = {
        "variant": profile.PRO0813_VARIANT,
        "official_config_sha256": profile.PRO0813_OFFICIAL_CONFIG_SHA256,
        "geometry_digest": profile.PRO0813_TP8_GEOMETRY_DIGEST,
        "num_target_layers": 61,
        "num_attention_heads": 128,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1536,
        "tp_size": 8,
        "heads_per_tp_rank": 16,
    }
    if observed != expected:
        raise RuntimeError(f"official Pro-0813 geometry changed: {observed!r}")
    if (
        reloc.NOPE_DIM != NOPE_BYTES
        or reloc.ROPE_DIM != ROPE_DIM
        or reloc.ROPE_BYTES != ROPE_BYTES
        or reloc.NOPE_ROPE_BYTES != PAGE_STRIDE_BYTES
        or reloc.SCALE_PADDED != SCALE_BYTES
        or reloc.BYTES_PER_TOKEN != PACKED_BYTES
    ):
        raise RuntimeError("packed Pro-0813 cache layout changed")
    return observed


def _run_layout_case(
    *, device: torch.device, page_size: int, seed: int
) -> dict[str, object]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    pages = 3
    row_bytes = _page_bytes(page_size)
    buf = torch.randint(
        0,
        256,
        (pages, row_bytes),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    raw_slots = (0, page_size - 1, page_size, 2 * page_size + page_size // 2)
    loc = torch.tensor(raw_slots, dtype=torch.int32, device=device)

    expected_rope = _read_rope_reference(buf, loc, page_size)
    actual_rope = reloc.read_rope_bf16(buf, loc, page_size)
    if not torch.equal(
        actual_rope.contiguous().view(torch.uint8),
        expected_rope.contiguous().view(torch.uint8),
    ):
        raise AssertionError("JIT rope gather differs from byte-address reference")

    expected_packed = _read_packed_reference(buf, loc, page_size)
    actual_packed = reloc.read_packed_kv(buf, loc, page_size)
    if not torch.equal(actual_packed, expected_packed):
        raise AssertionError("JIT packed gather differs from byte-address reference")

    expected_offsets = (
        loc.long() // page_size * row_bytes
        + loc.long() % page_size * PAGE_STRIDE_BYTES
    )
    actual_kv_base, actual_scale_base = reloc._packed_offsets_u8(
        loc, page_size, row_bytes
    )
    expected_scale_base = (
        loc.long() // page_size * row_bytes
        + page_size * PAGE_STRIDE_BYTES
        + loc.long() % page_size * SCALE_BYTES
    )
    if not torch.equal(actual_kv_base.long(), expected_offsets):
        raise AssertionError("JIT packed KV offsets differ from reference")
    if not torch.equal(actual_scale_base.long(), expected_scale_base):
        raise AssertionError("JIT packed scale offsets differ from reference")

    new_rope = (
        torch.randn(
            (loc.numel(), ROPE_DIM), device=device, generator=generator
        )
        * 0.125
    ).to(torch.bfloat16)
    actual_write = buf.clone()
    expected_write = buf.clone()
    reloc.write_rope_bf16(actual_write, loc, new_rope, page_size)
    _write_rope_reference(expected_write, loc, new_rope, page_size)
    if not torch.equal(actual_write, expected_write):
        raise AssertionError("JIT rope scatter changed non-RoPE bytes")

    new_packed = torch.randint(
        0,
        256,
        (loc.numel(), PACKED_BYTES),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    actual_write = buf.clone()
    expected_write = buf.clone()
    reloc.write_packed_kv(actual_write, loc, new_packed, page_size)
    _write_packed_reference(expected_write, loc, new_packed, page_size)
    if not torch.equal(actual_write, expected_write):
        raise AssertionError("JIT packed scatter differs from byte-address reference")

    return {
        "page_size": page_size,
        "page_row_bytes": row_bytes,
        "slots": list(raw_slots),
        "rope_gather_bitwise": True,
        "packed_gather_bitwise": True,
        "offsets_bitwise": True,
        "rope_scatter_full_buffer_bitwise": True,
        "packed_scatter_full_buffer_bitwise": True,
    }


def _run_relocation_case(device: torch.device) -> dict[str, object]:
    torch.manual_seed(1030813)
    positions = torch.tensor(
        [0, 1, 127, 128, 511, 512, 8191, 32768, 65535],
        dtype=torch.long,
        device=device,
    )
    destinations = torch.tensor(
        [65535, 8192, 128, 1, 512, 32768, 0, 511, 127],
        dtype=torch.long,
        device=device,
    )
    freqs = _frequency_table(65_536, device)
    canonical = (torch.randn((positions.numel(), ROPE_DIM), device=device) * 0.2).to(
        torch.bfloat16
    )
    source_rotated = _rotate_reference(
        canonical, positions, freqs, inverse=False
    )
    expected = _rotate_reference(
        _rotate_reference(source_rotated, positions, freqs, inverse=True),
        destinations,
        freqs,
        inverse=False,
    )
    actual = reloc.reposition_rope(
        source_rotated, positions, destinations, freqs
    )
    error = (actual.float() - expected.float()).abs()
    max_abs = float(error.max().item())
    mean_abs = float(error.mean().item())
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.0078125,
        atol=0.00390625,
    )

    page_size = 128
    row_bytes = _page_bytes(page_size)
    buf = torch.randint(
        0, 256, (2, row_bytes), dtype=torch.uint8, device=device
    )
    loc = torch.tensor(
        [0, 1, 63, 64, 127, 128, 191, 255, 129],
        dtype=torch.int32,
        device=device,
    )
    _write_rope_reference(buf, loc, source_rotated, page_size)
    before = buf.clone()
    expected_buf = buf.clone()
    _write_rope_reference(expected_buf, loc, expected, page_size)
    reloc.reposition_slots(
        buf, loc, positions, destinations, freqs, page_size
    )
    actual_slots = _read_rope_reference(buf, loc, page_size)
    expected_slots = _read_rope_reference(expected_buf, loc, page_size)
    slot_error = (actual_slots.float() - expected_slots.float()).abs()
    slot_max_abs = float(slot_error.max().item())
    torch.testing.assert_close(
        actual_slots.float(),
        expected_slots.float(),
        rtol=0.0078125,
        atol=0.00390625,
    )

    rope_indices = _indices(
        loc,
        page_size=page_size,
        row_bytes=row_bytes,
        relative_offset=NOPE_BYTES,
        width=ROPE_BYTES,
    ).flatten()
    changed = torch.zeros(buf.numel(), dtype=torch.bool, device=device)
    changed[rope_indices] = True
    if not torch.equal(buf.flatten()[~changed], before.flatten()[~changed]):
        raise AssertionError("reposition_slots changed non-RoPE cache bytes")
    return {
        "rows": int(positions.numel()),
        "max_position": int(positions.max().item()),
        "direct_max_abs": max_abs,
        "direct_mean_abs": mean_abs,
        "slots_max_abs": slot_max_abs,
        "non_rope_bytes_bitwise": True,
        "fp32_reference_pass": True,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = tuple(torch.cuda.get_device_capability(device))
    device_name = str(torch.cuda.get_device_name(device))
    if capability != (10, 3) or "B300" not in device_name:
        raise RuntimeError(
            f"this certificate is B300/SM103-only, got {device_name} SM{capability}"
        )
    ptxas = Path(os.environ.get("TRITON_PTXAS_PATH", "")).resolve()
    if ptxas != EXPECTED_PTXAS or not ptxas.is_file():
        raise RuntimeError(
            f"TRITON_PTXAS_PATH must resolve to {EXPECTED_PTXAS}, got {ptxas}"
        )
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS") != "1":
        raise RuntimeError("SGLANG_USE_JIT_PACKED_OFFSETS=1 is required")

    geometry = _assert_geometry()
    layout_cases = [
        _run_layout_case(device=device, page_size=4, seed=1030400),
        _run_layout_case(device=device, page_size=128, seed=1031280),
    ]
    relocation_case = _run_relocation_case(device)

    modules = {
        "rope_relocation": reloc,
        "deepseek_v4_rope": deepseek_v4_rope,
        "jit_dsv4_attn": jit_attn,
        "pro0813_profile": profile,
    }
    source_paths = {name: _module_path(module) for name, module in modules.items()}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    deployment_oracle = Path(__file__).resolve()
    if PRO_ROOT not in deployment_oracle.parents:
        raise RuntimeError("deployment oracle escaped isolated Pro checkout")
    source_hashes["deployment_oracle"] = _sha256(deployment_oracle)
    manifest_payload = {
        "schema": "redknot-pro0813-b300-rope-relocation-oracle-v1",
        "geometry_digest": profile.PRO0813_TP8_GEOMETRY_DIGEST,
        "source_hashes": source_hashes,
    }
    source_manifest_sha256 = "sha256:" + hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "status": "pass",
                "device": device_name,
                "compute_capability": "sm_103",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "ptxas": str(ptxas),
                "geometry": geometry,
                "layout_cases": layout_cases,
                "relocation_case": relocation_case,
                "source_hashes": source_hashes,
                "source_manifest_sha256": source_manifest_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
