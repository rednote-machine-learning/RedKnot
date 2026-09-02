# Copyright 2024-2026 SGLang RedKnot Integration.
"""In-place RoPE repositioning for DeepSeek-V4 packed SWA KV cache.

The DSV4 SWA KV pool stores each token as a packed 584-byte record inside a
page-major uint8 buffer::

    [ nope_fp8 (448 bytes) | rope_bf16 (64 * 2 = 128 bytes) | scale (7+1 bytes) ]

Only the ``rope_bf16`` portion (the last ``qk_rope_head_dim=64`` dims of the
latent key) carries rotary position information. For offline KV reuse we must
re-rotate that portion from the *offline* positions ``[0, L)`` to the *online*
absolute positions ``[offset, offset+L)``.

This module provides:
  * :func:`read_rope_bf16`  – gather the rope[64] bf16 vectors for a set of slots
  * :func:`write_rope_bf16` – scatter rope[64] bf16 vectors back into the buffer
  * :func:`reposition_slots` – read -> RoPE realign -> write, for given slots

The nope (FP8) and scale bytes are *never touched* (position-independent), so
offline reuse is lossless on those and only incurs BF16 rotation precision on
the 64 rope dims (negligible).

The RoPE math uses DeepSeek's compress_rope_theta / rope_theta. DSV4 applies
RoPE to the rope dims with a standard (NeoX/GPT-style interleaved-half) rotation
derived from ``inv_freq``. We re-derive cos/sin from the model's rotary module
so this stays consistent with whatever YaRN/scaling the model uses.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# Layout constants (match deepseek_v4_memory_pool.DeepSeekV4SingleKVPool).
# IMPORTANT: within a page, the (nope_fp8 | rope_bf16) records are packed
# CONTIGUOUSLY with stride = NOPE_DIM + ROPE_BYTES = 576 bytes/token. The
# per-token scales live SEPARATELY in a region at the end of the page (after
# page_size * 576 bytes). So the rope offset uses stride 576, NOT 584.
NOPE_DIM = 448  # fp8 bytes (1 byte per elem)
ROPE_DIM = 64  # bf16 elems
ROPE_BYTES = ROPE_DIM * 2
SCALE_DIM = 7
SCALE_PADDED = SCALE_DIM + 1
NOPE_ROPE_BYTES = NOPE_DIM + ROPE_BYTES  # 576: nope+rope packing stride
BYTES_PER_TOKEN = NOPE_ROPE_BYTES + SCALE_PADDED  # 584 (total incl. scale)


def _loc_to_int64(loc: torch.Tensor) -> torch.Tensor:
    """Use an active-architecture JIT cast when requested.

    CUDA 12.9 Triton emits an SM103-compatible kernel on B300, avoiding the
    precompiled PyTorch cast kernel selected by some DeepSeek-V4 worker paths.
    """
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_cast_int64

        return triton_cast_int64(loc)
    return loc.to(torch.int64)


def _rope_offsets_bf16(
    loc: torch.Tensor, page_size: int, buf_numel_per_page: int
) -> torch.Tensor:
    """Return, for each slot in ``loc``, the bf16 element offset of its rope[0].

    Mirrors the write formula in index_buf_accessor._set_k_and_s_torch:
        rope_offset = page_index * (buf_numel_per_page // 2)
                      + (token_off * (nope_dim + rope_dim*2) + nope_dim) // 2
    where buf is uint8 viewed as bf16 (so // 2 element strides) and the
    nope+rope packing stride is (nope_dim + rope_dim*2) = 576 bytes.
    """
    loc = _loc_to_int64(loc)
    page_index = loc // page_size
    token_off = loc % page_size
    rope_off = (
        page_index * (buf_numel_per_page // 2)
        + (token_off * NOPE_ROPE_BYTES + NOPE_DIM) // 2
    )
    return rope_off  # [N], element index into buf.view(bf16).flatten()


@torch.no_grad()
def read_rope_bf16(
    buf: torch.Tensor, loc: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Gather rope[64] bf16 vectors for slots ``loc``.

    buf: [num_pages, buf_numel_per_page] uint8 (the layer's kv_buffer)
    loc: [N] int slot indices
    returns: [N, 64] bfloat16
    """
    if buf.dtype != torch.uint8:
        buf = buf.view(torch.uint8)
    num_pages, buf_numel_per_page = buf.shape
    buf_bf16 = buf.view(torch.bfloat16).flatten()
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_read_rope_bf16

        return triton_read_rope_bf16(
            buf_bf16,
            loc,
            page_size,
            buf_numel_per_page,
            NOPE_ROPE_BYTES,
            NOPE_DIM,
            ROPE_DIM,
        )
    base = _rope_offsets_bf16(loc, page_size, buf_numel_per_page)  # [N]
    idx = base[:, None] + torch.arange(ROPE_DIM, device=buf.device)[None, :]  # [N,64]
    return buf_bf16[idx]  # [N,64] bf16


@torch.no_grad()
def write_rope_bf16(
    buf: torch.Tensor, loc: torch.Tensor, rope: torch.Tensor, page_size: int
) -> None:
    """Scatter rope[64] bf16 vectors back into ``buf`` at slots ``loc``."""
    if buf.dtype != torch.uint8:
        buf = buf.view(torch.uint8)
    num_pages, buf_numel_per_page = buf.shape
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_write_rope_bf16

        triton_write_rope_bf16(
            buf.view(torch.bfloat16).flatten(),
            loc,
            rope.contiguous(),
            page_size,
            buf_numel_per_page,
            NOPE_ROPE_BYTES,
            NOPE_DIM,
            ROPE_DIM,
        )
        return
    # Same rationale as ``write_packed_kv``: address the page/token axes with two
    # 1-D index tensors instead of building an ``[N, 64]`` index matrix.  In bf16
    # element units each record spans ``NOPE_ROPE_BYTES // 2`` elements and its
    # rope block starts at ``NOPE_DIM // 2``.
    record_elems = NOPE_ROPE_BYTES // 2
    rope_start = NOPE_DIM // 2
    loc = _loc_to_int64(loc)
    page_index = loc // page_size
    token_off = loc % page_size
    records = (
        buf.view(torch.bfloat16)[:, : page_size * record_elems]
        .unflatten(1, (page_size, record_elems))
    )
    rope_region = records[:, :, rope_start : rope_start + ROPE_DIM]
    rope_region[page_index, token_off] = rope.to(torch.bfloat16)


def _packed_offsets_u8(
    loc: torch.Tensor, page_size: int, buf_numel_per_page: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_packed_offsets_u8

        return triton_packed_offsets_u8(
            loc,
            page_size,
            buf_numel_per_page,
            NOPE_ROPE_BYTES,
            SCALE_PADDED,
        )
    loc = _loc_to_int64(loc)
    page_index = loc // page_size
    token_off = loc % page_size
    kv_base = page_index * buf_numel_per_page + token_off * NOPE_ROPE_BYTES
    scale_base = (
        page_index * buf_numel_per_page
        + page_size * NOPE_ROPE_BYTES
        + token_off * SCALE_PADDED
    )
    return kv_base, scale_base


@torch.no_grad()
def read_packed_kv(buf: torch.Tensor, loc: torch.Tensor, page_size: int) -> torch.Tensor:
    """Gather complete DSV4 SWA records as ``[N, 584]`` uint8 tensors."""
    # Ensure uint8 view — get_key_buffer may return a float8 view of the
    # underlying uint8 storage (same byte layout).
    if buf.dtype != torch.uint8:
        buf = buf.view(torch.uint8)
    _, buf_numel_per_page = buf.shape
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_read_packed_kv_u8

        return triton_read_packed_kv_u8(
            buf,
            loc,
            page_size,
            buf_numel_per_page,
            NOPE_ROPE_BYTES,
            SCALE_PADDED,
        )
    flat = buf.flatten()
    kv_base, scale_base = _packed_offsets_u8(loc, page_size, buf_numel_per_page)
    kv_idx = kv_base[:, None] + torch.arange(NOPE_ROPE_BYTES, device=buf.device)
    scale_idx = scale_base[:, None] + torch.arange(SCALE_PADDED, device=buf.device)
    # Bounds check
    max_kv = kv_idx.max().item() if kv_idx.numel() > 0 else -1
    max_sc = scale_idx.max().item() if scale_idx.numel() > 0 else -1
    if max_kv >= flat.numel() or max_sc >= flat.numel():
        import logging
        logging.error(
            f"[read_packed_kv OOB] buf={list(buf.shape)} page_size={page_size} "
            f"loc=[{loc.min().item()}..{loc.max().item()}] n={loc.numel()} "
            f"flat={flat.numel()} max_kv_idx={max_kv} max_scale_idx={max_sc} "
            f"buf_numel_per_page={buf_numel_per_page}"
        )
    return torch.cat((flat[kv_idx], flat[scale_idx]), dim=1)


@torch.no_grad()
def write_packed_kv(
    buf: torch.Tensor, loc: torch.Tensor, packed: torch.Tensor, page_size: int
) -> None:
    """Scatter complete ``[N, 584]`` DSV4 SWA records into cache slots."""
    if packed.shape != (loc.numel(), BYTES_PER_TOKEN):
        raise ValueError(
            f"packed shape must be {(loc.numel(), BYTES_PER_TOKEN)}, got {packed.shape}"
        )
    # Ensure uint8 view — get_key_buffer may return a float8 view of the
    # underlying uint8 storage (same byte layout).
    if buf.dtype != torch.uint8:
        buf = buf.view(torch.uint8)
    _, buf_numel_per_page = buf.shape
    packed = packed.to(device=buf.device, dtype=torch.uint8)
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_write_packed_kv_u8

        triton_write_packed_kv_u8(
            buf,
            loc,
            packed.contiguous(),
            page_size,
            buf_numel_per_page,
            NOPE_ROPE_BYTES,
            SCALE_PADDED,
        )
        return
    # Index the page/token axes directly instead of materializing an
    # ``[N, record_bytes]`` int64 index matrix.  The old form allocated eight
    # bytes of index per payload byte and forced a fully general scatter, which
    # dominated restore time.  Each page stores ``page_size`` contiguous
    # nope+rope records followed by ``page_size`` contiguous scale records, so
    # both regions are exact views and the innermost bytes stay contiguous.
    loc = _loc_to_int64(loc)
    page_index = loc // page_size
    token_off = loc % page_size
    kv_region = buf[:, : page_size * NOPE_ROPE_BYTES].unflatten(
        1, (page_size, NOPE_ROPE_BYTES)
    )
    scale_region = buf[
        :,
        page_size * NOPE_ROPE_BYTES : page_size * (NOPE_ROPE_BYTES + SCALE_PADDED),
    ].unflatten(1, (page_size, SCALE_PADDED))
    kv_region[page_index, token_off] = packed[:, :NOPE_ROPE_BYTES]
    scale_region[page_index, token_off] = packed[:, NOPE_ROPE_BYTES:]


@torch.no_grad()
def reposition_rope(
    rope: torch.Tensor,
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Re-rotate rope[N,64] from src_pos[N] to dst_pos[N] using DeepSeek-V4's
    OWN rotary kernel (complex / adjacent-pair rotation with YaRN freqs_cis).

    This guarantees the unapply/apply cancel exactly the way the model wrote the
    cache: we call ``apply_rotary_emb_triton`` with ``inverse=True`` at the
    source positions (undo local RoPE) then ``inverse=False`` at the
    destination positions (apply global RoPE).

    Parameters
    ----------
    rope: [N, 64] bf16/any — the cached rope key vectors (local-rotated).
    src_pos, dst_pos: [N] long — source (local) and destination (global) positions.
    freqs_cis: complex tensor [max_pos, 32] — the model's precomputed RoPE table
        (YaRN-adjusted), shared by the layer that produced ``rope``.
    """
    from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton

    device = rope.device
    # The kernel writes in-place and expects float-ish 2D [N, rope_dim].
    work = rope.to(torch.bfloat16).clone().contiguous()
    src = src_pos.to(device=device, dtype=torch.int32).contiguous()
    dst = dst_pos.to(device=device, dtype=torch.int32).contiguous()
    fc = freqs_cis.to(device)
    # 1) undo the local-position rotation
    apply_rotary_emb_triton(work, fc, positions=src, inverse=True)
    # 2) apply the global-position rotation
    apply_rotary_emb_triton(work, fc, positions=dst, inverse=False)
    return work.to(rope.dtype)


@torch.no_grad()
def reposition_slots(
    buf: torch.Tensor,
    loc: torch.Tensor,
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    freqs_cis: torch.Tensor,
    page_size: int,
) -> None:
    """Read rope at slots, re-rotate from src_pos to dst_pos, write back."""
    rope = read_rope_bf16(buf, loc, page_size)  # [N,64]
    rope2 = reposition_rope(rope, src_pos, dst_pos, freqs_cis)
    write_rope_bf16(buf, loc, rope2, page_size)
