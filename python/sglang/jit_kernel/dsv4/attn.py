from typing import Literal, Tuple

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_metadata_module():
    return load_jit(
        make_name("metadata"),
        cuda_files=["deepseek_v4/paged_mqa_metadata.cuh"],
        cuda_wrappers=[("run", "IndexerMetadataKernel::run")],
    )


@cache_once
def _jit_fused_store_module(
    name: Literal["flashmla", "indexer"],
    input_dtype: torch.dtype,
    index_dtype: torch.dtype,
    page_size: int,
):
    args = make_cpp_args(input_dtype, index_dtype, page_size, is_arch_support_pdl())
    cname = "FlashMLA" if name == "flashmla" else "Indexer"
    kernel_class = f"FusedStoreCache{cname}Kernel<{args}>"
    return load_jit(
        make_name("store_" + name),
        *args,
        cuda_files=["deepseek_v4/store.cuh"],
        cuda_wrappers=[("run", f"{kernel_class}::run")],
    )


def get_paged_mqa_logits_metadata(seq_lens: torch.Tensor, page_size: int, num_sm: int):
    assert page_size == 64
    seq_lens = seq_lens.view(-1).to(torch.int32)
    metadata = seq_lens.new_empty(num_sm + 1, 2)
    module = _jit_metadata_module()
    module.run(seq_lens, metadata)
    return metadata


def fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    if is_hip_runtime():
        from sglang.jit_kernel.triton_store_cache import triton_fused_store_cache

        triton_fused_store_cache(input, cache, indices, page_size=page_size, type=type)
    else:
        module = _jit_fused_store_module(
            name=type,
            input_dtype=input.dtype,
            index_dtype=indices.dtype,
            page_size=page_size,
        )
        module.run(input, cache, indices)


@triton.jit
def translate_loc_kernel(
    mapping_ptr,
    indices_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0).to(tl.int64)
    values = tl.load(mapping_ptr + indices, mask=mask, other=0).to(tl.int32)
    tl.store(out_ptr + offsets, values, mask=mask)


def triton_translate_loc(
    full_to_swa_index_mapping: torch.Tensor,
    kv_indices: torch.Tensor,
) -> torch.Tensor:
    """Translate FULL-cache slots without relying on a prebuilt CUDA index kernel."""
    kv_indices = kv_indices.contiguous()
    out = torch.empty(kv_indices.shape, dtype=torch.int32, device=kv_indices.device)
    numel = kv_indices.numel()
    if numel == 0:
        return out
    block = 256
    translate_loc_kernel[(triton.cdiv(numel, block),)](
        full_to_swa_index_mapping,
        kv_indices,
        out,
        numel,
        BLOCK=block,
    )
    return out


@triton.jit
def cast_int64_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(input_ptr + offsets, mask=mask, other=0)
    tl.store(output_ptr + offsets, values.to(tl.int64), mask=mask)


def triton_cast_int64(input: torch.Tensor) -> torch.Tensor:
    """Cast a contiguous CUDA tensor to int64 using Triton's active toolchain.

    This is used by the DeepSeek-V4 cache relocation path on Blackwell systems
    where the PyTorch wheel can otherwise dispatch a prebuilt cast kernel that
    has no SM103 image.  Triton compiles this tiny kernel for the active GPU.
    """
    if input.dtype == torch.int64:
        return input
    if not input.is_cuda:
        return input.to(torch.int64)
    if not input.is_contiguous():
        raise ValueError("triton_cast_int64 requires a contiguous input tensor")
    output = torch.empty(input.shape, dtype=torch.int64, device=input.device)
    numel = input.numel()
    if numel == 0:
        return output
    block = 256
    cast_int64_kernel[(triton.cdiv(numel, block),)](
        input,
        output,
        numel,
        BLOCK=block,
    )
    return output


@triton.jit
def packed_offsets_u8_kernel(
    loc_ptr,
    kv_base_ptr,
    scale_base_ptr,
    numel,
    page_size,
    buf_numel_per_page,
    nope_rope_bytes,
    scale_padded,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    loc = tl.load(loc_ptr + offsets, mask=mask, other=0).to(tl.int64)
    page_index = loc // page_size
    token_offset = loc % page_size
    page_base = page_index * buf_numel_per_page
    kv_base = page_base + token_offset * nope_rope_bytes
    scale_base = (
        page_base + page_size * nope_rope_bytes + token_offset * scale_padded
    )
    tl.store(kv_base_ptr + offsets, kv_base, mask=mask)
    tl.store(scale_base_ptr + offsets, scale_base, mask=mask)


def triton_packed_offsets_u8(
    loc: torch.Tensor,
    page_size: int,
    buf_numel_per_page: int,
    nope_rope_bytes: int,
    scale_padded: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build packed-cache byte offsets using an active-architecture kernel."""
    if not loc.is_cuda:
        raise ValueError("triton_packed_offsets_u8 requires a CUDA tensor")
    if not loc.is_contiguous():
        raise ValueError("triton_packed_offsets_u8 requires contiguous locations")
    kv_base = torch.empty(loc.shape, dtype=torch.int64, device=loc.device)
    scale_base = torch.empty_like(kv_base)
    numel = loc.numel()
    if numel == 0:
        return kv_base, scale_base
    block = 256
    packed_offsets_u8_kernel[(triton.cdiv(numel, block),)](
        loc,
        kv_base,
        scale_base,
        numel,
        page_size,
        buf_numel_per_page,
        nope_rope_bytes,
        scale_padded,
        BLOCK=block,
    )
    return kv_base, scale_base


@triton.jit
def read_packed_kv_u8_kernel(
    buf_ptr,
    loc_ptr,
    output_ptr,
    num_records,
    page_size,
    buf_numel_per_page,
    nope_rope_bytes: tl.constexpr,
    scale_padded: tl.constexpr,
    total_bytes: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    record = tl.program_id(0)
    byte_offsets = tl.arange(0, BLOCK)
    mask = (record < num_records) & (byte_offsets < total_bytes)
    loc = tl.load(loc_ptr + record, mask=record < num_records, other=0).to(tl.int64)
    page_index = loc // page_size
    token_offset = loc % page_size
    page_base = page_index * buf_numel_per_page
    is_kv = byte_offsets < nope_rope_bytes
    kv_address = page_base + token_offset * nope_rope_bytes + byte_offsets
    scale_address = (
        page_base
        + page_size * nope_rope_bytes
        + token_offset * scale_padded
        + byte_offsets
        - nope_rope_bytes
    )
    address = tl.where(is_kv, kv_address, scale_address)
    value = tl.load(buf_ptr + address, mask=mask, other=0)
    tl.store(output_ptr + record * total_bytes + byte_offsets, value, mask=mask)


def triton_read_packed_kv_u8(
    buf: torch.Tensor,
    loc: torch.Tensor,
    page_size: int,
    buf_numel_per_page: int,
    nope_rope_bytes: int,
    scale_padded: int,
) -> torch.Tensor:
    """Gather complete packed KV records without generic CUDA indexing."""
    if buf.dtype != torch.uint8 or not buf.is_cuda:
        raise ValueError("triton_read_packed_kv_u8 requires a CUDA uint8 buffer")
    if not buf.is_contiguous() or not loc.is_contiguous():
        raise ValueError("triton_read_packed_kv_u8 requires contiguous tensors")
    total_bytes = nope_rope_bytes + scale_padded
    output = torch.empty(
        (loc.numel(), total_bytes), dtype=torch.uint8, device=buf.device
    )
    if loc.numel() == 0:
        return output
    read_packed_kv_u8_kernel[(loc.numel(),)](
        buf,
        loc,
        output,
        loc.numel(),
        page_size,
        buf_numel_per_page,
        nope_rope_bytes=nope_rope_bytes,
        scale_padded=scale_padded,
        total_bytes=total_bytes,
        BLOCK=1024,
    )
    return output


@triton.jit
def write_packed_kv_u8_kernel(
    buf_ptr,
    loc_ptr,
    packed_ptr,
    num_records,
    page_size,
    buf_numel_per_page,
    nope_rope_bytes: tl.constexpr,
    scale_padded: tl.constexpr,
    total_bytes: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    record = tl.program_id(0)
    byte_offsets = tl.arange(0, BLOCK)
    mask = (record < num_records) & (byte_offsets < total_bytes)
    loc = tl.load(loc_ptr + record, mask=record < num_records, other=0).to(tl.int64)
    page_index = loc // page_size
    token_offset = loc % page_size
    page_base = page_index * buf_numel_per_page
    is_kv = byte_offsets < nope_rope_bytes
    kv_address = page_base + token_offset * nope_rope_bytes + byte_offsets
    scale_address = (
        page_base
        + page_size * nope_rope_bytes
        + token_offset * scale_padded
        + byte_offsets
        - nope_rope_bytes
    )
    address = tl.where(is_kv, kv_address, scale_address)
    value = tl.load(packed_ptr + record * total_bytes + byte_offsets, mask=mask)
    tl.store(buf_ptr + address, value, mask=mask)


def triton_write_packed_kv_u8(
    buf: torch.Tensor,
    loc: torch.Tensor,
    packed: torch.Tensor,
    page_size: int,
    buf_numel_per_page: int,
    nope_rope_bytes: int,
    scale_padded: int,
) -> None:
    """Scatter complete packed KV records without generic CUDA indexing."""
    if buf.dtype != torch.uint8 or packed.dtype != torch.uint8 or not buf.is_cuda:
        raise ValueError("triton_write_packed_kv_u8 requires CUDA uint8 tensors")
    if not buf.is_contiguous() or not loc.is_contiguous() or not packed.is_contiguous():
        raise ValueError("triton_write_packed_kv_u8 requires contiguous tensors")
    total_bytes = nope_rope_bytes + scale_padded
    if packed.shape != (loc.numel(), total_bytes):
        raise ValueError("packed shape does not match the location count")
    if loc.numel() == 0:
        return
    write_packed_kv_u8_kernel[(loc.numel(),)](
        buf,
        loc,
        packed,
        loc.numel(),
        page_size,
        buf_numel_per_page,
        nope_rope_bytes=nope_rope_bytes,
        scale_padded=scale_padded,
        total_bytes=total_bytes,
        BLOCK=1024,
    )


@triton.jit
def read_rope_bf16_kernel(
    buf_bf16_ptr,
    loc_ptr,
    output_ptr,
    num_records,
    page_size,
    buf_numel_per_page_u8,
    nope_rope_bytes: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    record = tl.program_id(0)
    dims = tl.arange(0, BLOCK)
    mask = (record < num_records) & (dims < rope_dim)
    loc = tl.load(loc_ptr + record, mask=record < num_records, other=0).to(tl.int64)
    page_index = loc // page_size
    token_offset = loc % page_size
    base = (
        page_index * (buf_numel_per_page_u8 // 2)
        + (token_offset * nope_rope_bytes + nope_dim) // 2
    )
    value = tl.load(buf_bf16_ptr + base + dims, mask=mask, other=0.0)
    tl.store(output_ptr + record * rope_dim + dims, value, mask=mask)


def triton_read_rope_bf16(
    buf_bf16: torch.Tensor,
    loc: torch.Tensor,
    page_size: int,
    buf_numel_per_page_u8: int,
    nope_rope_bytes: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Gather packed RoPE vectors without generic CUDA indexing."""
    if buf_bf16.dtype != torch.bfloat16 or not buf_bf16.is_cuda:
        raise ValueError("triton_read_rope_bf16 requires a CUDA bf16 buffer")
    if not buf_bf16.is_contiguous() or not loc.is_contiguous():
        raise ValueError("triton_read_rope_bf16 requires contiguous tensors")
    output = torch.empty(
        (loc.numel(), rope_dim), dtype=torch.bfloat16, device=buf_bf16.device
    )
    if loc.numel() == 0:
        return output
    read_rope_bf16_kernel[(loc.numel(),)](
        buf_bf16,
        loc,
        output,
        loc.numel(),
        page_size,
        buf_numel_per_page_u8,
        nope_rope_bytes=nope_rope_bytes,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        BLOCK=64,
    )
    return output


@triton.jit
def write_rope_bf16_kernel(
    buf_bf16_ptr,
    loc_ptr,
    rope_ptr,
    num_records,
    page_size,
    buf_numel_per_page_u8,
    nope_rope_bytes: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    record = tl.program_id(0)
    dims = tl.arange(0, BLOCK)
    mask = (record < num_records) & (dims < rope_dim)
    loc = tl.load(loc_ptr + record, mask=record < num_records, other=0).to(tl.int64)
    page_index = loc // page_size
    token_offset = loc % page_size
    base = (
        page_index * (buf_numel_per_page_u8 // 2)
        + (token_offset * nope_rope_bytes + nope_dim) // 2
    )
    value = tl.load(rope_ptr + record * rope_dim + dims, mask=mask)
    tl.store(buf_bf16_ptr + base + dims, value, mask=mask)


def triton_write_rope_bf16(
    buf_bf16: torch.Tensor,
    loc: torch.Tensor,
    rope: torch.Tensor,
    page_size: int,
    buf_numel_per_page_u8: int,
    nope_rope_bytes: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """Scatter packed RoPE vectors without generic CUDA indexing."""
    if buf_bf16.dtype != torch.bfloat16 or not buf_bf16.is_cuda:
        raise ValueError("triton_write_rope_bf16 requires a CUDA bf16 buffer")
    if not buf_bf16.is_contiguous() or not loc.is_contiguous() or not rope.is_contiguous():
        raise ValueError("triton_write_rope_bf16 requires contiguous tensors")
    if rope.shape != (loc.numel(), rope_dim):
        raise ValueError("rope shape does not match the location count")
    if loc.numel() == 0:
        return
    write_rope_bf16_kernel[(loc.numel(),)](
        buf_bf16,
        loc,
        rope,
        loc.numel(),
        page_size,
        buf_numel_per_page_u8,
        nope_rope_bytes=nope_rope_bytes,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        BLOCK=64,
    )


@triton.jit
def compute_compressed_slots_kernel(
    full_slots_ptr,
    full_to_swa_ptr,
    output_ptr,
    num_output,
    compress_ratio: tl.constexpr,
    swa_page_size,
    ring_size,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_output
    boundary = (offsets + 1) * compress_ratio - 1
    full_slot = tl.load(full_slots_ptr + boundary, mask=mask, other=0).to(tl.int64)
    swa_loc = tl.load(full_to_swa_ptr + full_slot, mask=mask, other=0).to(tl.int32)
    state_loc = (swa_loc // swa_page_size) * ring_size + swa_loc % ring_size
    tl.store(output_ptr + offsets, state_loc // compress_ratio, mask=mask)


def triton_compute_compressed_slots(
    full_slots: torch.Tensor,
    full_to_swa: torch.Tensor,
    seq_len: int,
    compress_ratio: int,
    swa_page_size: int,
    ring_size: int,
) -> torch.Tensor:
    """Compute compressor-state slots using an SM103 JIT kernel."""
    num_output = seq_len // compress_ratio
    output = torch.empty(
        (num_output,), dtype=torch.int32, device=full_slots.device
    )
    if num_output == 0:
        return output
    block = 256
    compute_compressed_slots_kernel[(triton.cdiv(num_output, block),)](
        full_slots,
        full_to_swa,
        output,
        num_output,
        compress_ratio=compress_ratio,
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        BLOCK=block,
    )
    return output


@triton.jit
def compute_paged_compressed_slots_kernel(
    page_table_ptr,
    output_ptr,
    num_output,
    req_idx,
    compressed_start,
    compressed_page_size: tl.constexpr,
    stride_page_table_0,
    stride_page_table_1,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_output
    compressed_offset = compressed_start + offsets
    page_index = compressed_offset // compressed_page_size
    offset_in_page = compressed_offset % compressed_page_size
    page_value = tl.load(
        page_table_ptr
        + req_idx * stride_page_table_0
        + page_index * stride_page_table_1,
        mask=mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        output_ptr + offsets,
        page_value * compressed_page_size + offset_in_page,
        mask=mask,
    )


def triton_compute_paged_compressed_slots(
    page_table: torch.Tensor,
    req_idx: int,
    seq_len: int,
    compress_ratio: int,
    compressed_page_size: int,
    token_offset: int,
) -> torch.Tensor:
    """Map compressed sequence offsets through a paged table on SM103."""
    num_output = seq_len // compress_ratio
    output = torch.empty(
        (num_output,), dtype=torch.int32, device=page_table.device
    )
    if num_output == 0:
        return output
    block = 256
    compute_paged_compressed_slots_kernel[(triton.cdiv(num_output, block),)](
        page_table,
        output,
        num_output,
        req_idx,
        token_offset // compress_ratio,
        compressed_page_size=compressed_page_size,
        stride_page_table_0=page_table.stride(0),
        stride_page_table_1=page_table.stride(1),
        BLOCK=block,
    )
    return output


@triton.jit
def index_select_rows_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    row_size,
    BLOCK: tl.constexpr,
) -> None:
    output_row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < row_size
    input_row = tl.load(indices_ptr + output_row).to(tl.int64)
    values = tl.load(input_ptr + input_row * row_size + offsets, mask=mask)
    tl.store(output_ptr + output_row * row_size + offsets, values, mask=mask)


def triton_index_select_rows(
    input: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Select rows from a contiguous tensor with an active-architecture kernel."""
    if not input.is_cuda or not input.is_contiguous() or not indices.is_contiguous():
        raise ValueError("triton_index_select_rows requires contiguous CUDA tensors")
    output_shape = (indices.numel(), *input.shape[1:])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if indices.numel() == 0:
        return output
    row_size = input[0].numel()
    block = 256
    index_select_rows_kernel[(indices.numel(), triton.cdiv(row_size, block))](
        input,
        indices,
        output,
        row_size,
        BLOCK=block,
    )
    return output


@triton.jit
def index_copy_rows_kernel(
    output_ptr,
    indices_ptr,
    input_ptr,
    row_size,
    BLOCK: tl.constexpr,
) -> None:
    input_row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < row_size
    output_row = tl.load(indices_ptr + input_row).to(tl.int64)
    values = tl.load(input_ptr + input_row * row_size + offsets, mask=mask)
    tl.store(output_ptr + output_row * row_size + offsets, values, mask=mask)


def triton_index_copy_rows(
    output: torch.Tensor,
    indices: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """Copy rows into a contiguous tensor with an active-architecture kernel."""
    if (
        not output.is_cuda
        or not output.is_contiguous()
        or not indices.is_contiguous()
        or not input.is_contiguous()
    ):
        raise ValueError("triton_index_copy_rows requires contiguous CUDA tensors")
    if input.shape != (indices.numel(), *output.shape[1:]):
        raise ValueError("input row shape does not match the destination")
    if indices.numel() == 0:
        return
    row_size = output[0].numel()
    block = 256
    index_copy_rows_kernel[(indices.numel(), triton.cdiv(row_size, block))](
        output,
        indices,
        input,
        row_size,
        BLOCK=block,
    )


@triton.jit
def hc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    output_ptr,
    hidden_size,
    stride_x_0,
    stride_x_1,
    stride_residual_0,
    stride_residual_1,
    stride_residual_2,
    stride_post_0,
    stride_post_1,
    stride_comb_0,
    stride_comb_1,
    stride_comb_2,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    HC_MULT: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    token_channel = tl.program_id(0)
    token = token_channel // HC_MULT
    output_channel = token_channel % HC_MULT
    hidden_offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = hidden_offsets < hidden_size

    x = tl.load(
        x_ptr + token * stride_x_0 + hidden_offsets * stride_x_1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    post = tl.load(
        post_ptr + token * stride_post_0 + output_channel * stride_post_1
    ).to(tl.float32)
    value = post * x
    for input_channel in tl.static_range(HC_MULT):
        residual = tl.load(
            residual_ptr
            + token * stride_residual_0
            + input_channel * stride_residual_1
            + hidden_offsets * stride_residual_2,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        coefficient = tl.load(
            comb_ptr
            + token * stride_comb_0
            + input_channel * stride_comb_1
            + output_channel * stride_comb_2
        ).to(tl.float32)
        value += coefficient * residual
    tl.store(
        output_ptr
        + token * stride_output_0
        + output_channel * stride_output_1
        + hidden_offsets * stride_output_2,
        value,
        mask=mask,
    )


def triton_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """DeepSeek-V4 HC post-mix compiled for the active CUDA architecture."""
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[1]
    if residual.shape != (num_tokens, hc_mult, hidden_size):
        raise ValueError("residual shape does not match HC post inputs")
    if comb.shape != (num_tokens, hc_mult, hc_mult):
        raise ValueError("comb shape does not match HC post inputs")
    output = torch.empty_like(residual)
    if num_tokens == 0:
        return output
    block = 256
    hc_post_kernel[(num_tokens * hc_mult, triton.cdiv(hidden_size, block))](
        x,
        residual,
        post,
        comb,
        output,
        hidden_size,
        *x.stride(),
        *residual.stride(),
        *post.stride(),
        *comb.stride(),
        *output.stride(),
        HC_MULT=hc_mult,
        BLOCK=block,
    )
    return output


@triton.jit
def cast_fp32_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values.to(tl.float32), mask=mask)


def triton_cast_fp32(input: torch.Tensor) -> torch.Tensor:
    if input.dtype == torch.float32:
        return input
    if not input.is_cuda or not input.is_contiguous():
        raise ValueError("triton_cast_fp32 requires a contiguous CUDA tensor")
    output = torch.empty(input.shape, dtype=torch.float32, device=input.device)
    numel = input.numel()
    if numel == 0:
        return output
    block = 256
    cast_fp32_kernel[(triton.cdiv(numel, block),)](
        input, output, numel, BLOCK=block
    )
    return output


@triton.jit
def hc_scale_mixes_rms_kernel(
    residual_ptr,
    gemm_ptr,
    mixes_ptr,
    num_tokens,
    hc_hidden_size,
    rms_eps,
    MIX_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_MIX: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hc_hidden_size
    residual = tl.load(
        residual_ptr + token * hc_hidden_size + k_offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(residual * residual, axis=0) / hc_hidden_size
    inv_rms = tl.rsqrt(mean_square + rms_eps)
    mix_offsets = tl.arange(0, BLOCK_MIX)
    mix_mask = mix_offsets < MIX_DIM
    values = tl.load(
        gemm_ptr + token * MIX_DIM + mix_offsets,
        mask=mix_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        mixes_ptr + token * MIX_DIM + mix_offsets,
        values * inv_rms,
        mask=mix_mask,
    )


@triton.jit
def hc_split_sinkhorn_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    eps,
    SINKHORN_ITERS: tl.constexpr,
    HC_MULT: tl.constexpr,
    MIX_DIM: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    channels = tl.arange(0, HC_MULT)
    pre_logits = (
        tl.load(mixes_ptr + token * MIX_DIM + channels)
        * tl.load(scale_ptr + 0)
        + tl.load(base_ptr + channels)
    )
    pre = 1.0 / (1.0 + tl.exp(-pre_logits)) + eps
    post_logits = (
        tl.load(mixes_ptr + token * MIX_DIM + HC_MULT + channels)
        * tl.load(scale_ptr + 1)
        + tl.load(base_ptr + HC_MULT + channels)
    )
    post = 2.0 / (1.0 + tl.exp(-post_logits))

    comb_offsets = tl.arange(0, HC_MULT * HC_MULT)
    comb_logits_flat = (
        tl.load(mixes_ptr + token * MIX_DIM + 2 * HC_MULT + comb_offsets)
        * tl.load(scale_ptr + 2)
        + tl.load(base_ptr + 2 * HC_MULT + comb_offsets)
    )
    comb_logits = tl.reshape(comb_logits_flat, (HC_MULT, HC_MULT))
    row_max = tl.max(comb_logits, axis=1)
    comb = tl.exp(comb_logits - row_max[:, None])
    comb /= tl.sum(comb, axis=1)[:, None]
    comb += eps
    comb /= tl.sum(comb, axis=0)[None, :] + eps
    for _ in tl.static_range(SINKHORN_ITERS - 1):
        comb /= tl.sum(comb, axis=1)[:, None] + eps
        comb /= tl.sum(comb, axis=0)[None, :] + eps

    tl.store(pre_ptr + token * HC_MULT + channels, pre)
    tl.store(post_ptr + token * HC_MULT + channels, post)
    tl.store(
        comb_ptr + token * HC_MULT * HC_MULT + comb_offsets,
        tl.reshape(comb, (HC_MULT * HC_MULT,)),
    )


@triton.jit
def hc_pre_weighted_sum_kernel(
    residual_ptr,
    pre_ptr,
    output_ptr,
    hidden_size,
    HC_MULT: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = hidden_offsets < hidden_size
    value = tl.zeros((BLOCK,), tl.float32)
    for channel in tl.static_range(HC_MULT):
        weight = tl.load(pre_ptr + token * HC_MULT + channel).to(tl.float32)
        residual = tl.load(
            residual_ptr
            + (token * HC_MULT + channel) * hidden_size
            + hidden_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value += weight * residual
    tl.store(output_ptr + token * hidden_size + hidden_offsets, value, mask=mask)


def triton_hc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    sinkhorn_iters: int,
    hc_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 HC pre-mix using cuBLAS plus SM103 Triton epilogues."""
    if residual.ndim != 3 or not residual.is_contiguous():
        raise ValueError("triton_hc_pre requires a contiguous [T, C, H] tensor")
    num_tokens, hc_mult, hidden_size = residual.shape
    hc_hidden_size = hc_mult * hidden_size
    mix_dim = (2 + hc_mult) * hc_mult
    if fn.shape != (mix_dim, hc_hidden_size):
        raise ValueError("HC projection shape does not match residual")
    residual_flat = residual.view(num_tokens, hc_hidden_size)
    residual_fp32 = triton_cast_fp32(residual_flat)
    gemm = torch.nn.functional.linear(residual_fp32, fn)
    mixes = torch.empty_like(gemm)
    if num_tokens == 0:
        return (
            residual.new_empty((0, hidden_size)),
            gemm.new_empty((0, hc_mult)),
            gemm.new_empty((0, hc_mult, hc_mult)),
        )
    block_k = triton.next_power_of_2(hc_hidden_size)
    block_mix = triton.next_power_of_2(mix_dim)
    hc_scale_mixes_rms_kernel[(num_tokens,)](
        residual,
        gemm,
        mixes,
        num_tokens,
        hc_hidden_size,
        rms_eps,
        MIX_DIM=mix_dim,
        BLOCK_K=block_k,
        BLOCK_MIX=block_mix,
        num_warps=8,
    )
    pre = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(pre)
    comb = torch.empty(
        (num_tokens, hc_mult, hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )
    hc_split_sinkhorn_kernel[(num_tokens,)](
        mixes,
        hc_scale,
        hc_base,
        pre,
        post,
        comb,
        hc_eps,
        SINKHORN_ITERS=sinkhorn_iters,
        HC_MULT=hc_mult,
        MIX_DIM=mix_dim,
    )
    output = torch.empty(
        (num_tokens, hidden_size), dtype=residual.dtype, device=residual.device
    )
    block = 256
    hc_pre_weighted_sum_kernel[
        (num_tokens, triton.cdiv(hidden_size, block))
    ](
        residual,
        pre,
        output,
        hidden_size,
        HC_MULT=hc_mult,
        BLOCK=block,
    )
    return output, post, comb


@triton.jit
def create_paged_compress_data_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    req_to_token_ptr,
    full_to_swa_index_mapping_ptr,
    out_0_ptr,
    out_1_ptr,
    batch_size,
    stride_req_to_token_0,
    stride_req_to_token_1: tl.constexpr,
    stride_out_1_0,
    stride_out_1_1: tl.constexpr,
    compress_ratio: tl.constexpr,
    is_overlap: tl.constexpr,
    swa_page_size: tl.constexpr,
    ring_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size

    rid = tl.load(req_pool_indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    extend_len = tl.load(extend_seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    prefix_len = seq_len - extend_len

    cr = compress_ratio
    write_pos = ((seq_len - 1) // cr) * cr
    load_pos = ((prefix_len - 1) // cr) * cr
    write_overlap_pos = write_pos - cr
    load_overlap_pos = load_pos - cr
    v0 = tl.zeros([BLOCK], tl.int32)
    v1 = tl.zeros([BLOCK], tl.int32)
    v2 = tl.zeros([BLOCK], tl.int32)
    v3 = tl.zeros([BLOCK], tl.int32)

    for i in tl.static_range(4):
        if i == 0:
            pos = load_pos
        elif i == 1:
            pos = write_pos
        elif i == 2:
            pos = load_overlap_pos
        else:
            pos = write_overlap_pos
        pos = tl.maximum(pos, 0)
        loc = tl.load(
            req_to_token_ptr
            + rid.to(tl.int64) * stride_req_to_token_0
            + pos.to(tl.int64) * stride_req_to_token_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        swa_loc = tl.load(full_to_swa_index_mapping_ptr + loc, mask=mask, other=0).to(
            tl.int32
        )
        swa_page = swa_loc // swa_page_size
        state_loc = swa_page * ring_size + (swa_loc % ring_size)
        state_loc = state_loc // cr
        if i == 0:
            v0 = state_loc
        elif i == 1:
            v1 = state_loc
        elif i == 2:
            v2 = state_loc
        else:
            v3 = state_loc

    tl.store(out_0_ptr + offs, v1, mask=mask)

    if is_overlap:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v2, mask=mask)
        tl.store(base + 1 * stride_out_1_1, v0, mask=mask)
        tl.store(base + 2 * stride_out_1_1, v3, mask=mask)
        tl.store(base + 3 * stride_out_1_1, write_pos.to(tl.int32), mask=mask)
    else:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v0, mask=mask)


def triton_create_paged_compress_data(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = req_pool_indices.shape[0]
    out_dim = 4 if is_overlap else 1
    device_args: dict = dict(device=req_pool_indices.device, dtype=torch.int32)
    out_0 = torch.empty((batch_size,), **device_args)
    out_1 = torch.empty((batch_size, out_dim), **device_args)
    grid = (triton.cdiv(batch_size, block),)
    create_paged_compress_data_kernel[grid](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        req_to_token,
        full_to_swa_index_mapping,
        out_0,
        out_1,
        batch_size=batch_size,
        stride_req_to_token_0=req_to_token.stride(0),
        stride_req_to_token_1=req_to_token.stride(1),  # type: ignore
        stride_out_1_0=out_1.stride(0),
        stride_out_1_1=out_1.stride(1),  # type: ignore
        compress_ratio=compress_ratio,  # type: ignore
        is_overlap=1 if is_overlap else 0,  # type: ignore
        swa_page_size=swa_page_size,  # type: ignore
        ring_size=ring_size,  # type: ignore
        BLOCK=block,  # type: ignore
    )

    if not is_overlap:
        out_1.squeeze_(1)
    return out_0, out_1
