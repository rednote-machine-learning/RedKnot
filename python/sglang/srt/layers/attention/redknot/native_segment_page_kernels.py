"""GPU metadata kernels for native RedKnot segment pages.

The first production use is deliberately narrow: compact an existing Indexer
Top-K set while enforcing a per-offline-document cap.  The kernel moves only
indices, never MLA/KV records.  Online suffix positions are retained without a
document cap.  Attention math is unchanged until the direct native-bank path
is separately certified.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.redknot.native_segment_pages import (
    NativeIndexerBucketPolicy,
)


@triton.jit
def _compact_indexer_topk_by_document_kernel(
    RawIndices,
    PhysicalIndices,
    InputLengths,
    OutputPhysical,
    OutputRaw,
    BucketCounts,
    OutputLengths,
    query_rows,
    stride_raw_q,
    stride_raw_k,
    stride_physical_q,
    stride_physical_k,
    stride_output_q,
    stride_output_k,
    stride_bucket_q,
    stride_bucket_document,
    compressed_rows_per_document,
    TOPK: tl.constexpr,
    DOCUMENTS: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_K + tl.arange(0, BLOCK_K)
    input_length = tl.load(InputLengths + query, mask=query < query_rows, other=0)
    active = (query < query_rows) & (offsets < TOPK) & (offsets < input_length)
    raw = tl.load(
        RawIndices + query * stride_raw_q + offsets * stride_raw_k,
        mask=active,
        other=-1,
    )
    physical = tl.load(
        PhysicalIndices
        + query * stride_physical_q
        + offsets * stride_physical_k,
        mask=active,
        other=-1,
    )
    offline_extent = DOCUMENTS * compressed_rows_per_document
    document = raw // compressed_rows_per_document
    online = raw >= offline_extent
    valid = active & (raw >= 0) & (physical >= 0)
    bucket = tl.where(online, DOCUMENTS, document)
    valid = valid & (bucket >= 0) & (bucket <= DOCUMENTS)
    safe_bucket = tl.where(valid, bucket, 0)
    bucket_address = (
        BucketCounts
        + query * stride_bucket_q
        + safe_bucket * stride_bucket_document
    )
    bucket_ordinal = tl.atomic_add(
        bucket_address, 1, mask=valid, sem="relaxed"
    )
    bucket_cap = tl.where(online, TOPK, CAP)
    keep = valid & (bucket_ordinal < bucket_cap)
    # Every active lane contributes to the same per-query compact length.  The
    # explicit zero-offset vector keeps the pointer/value/mask block shapes
    # identical across Triton versions.
    output_length_address = OutputLengths + query + offsets * 0
    output_ordinal = tl.atomic_add(
        output_length_address, 1, mask=keep, sem="relaxed"
    )
    in_range = keep & (output_ordinal < TOPK)
    output_address = (
        OutputPhysical + query * stride_output_q + output_ordinal * stride_output_k
    )
    raw_output_address = (
        OutputRaw + query * stride_output_q + output_ordinal * stride_output_k
    )
    tl.store(output_address, physical, mask=in_range)
    tl.store(raw_output_address, raw, mask=in_range)


@dataclass
class NativeIndexerBucketWorkspace:
    output_physical: torch.Tensor
    output_raw: torch.Tensor
    bucket_counts: torch.Tensor
    output_lengths: torch.Tensor

    def compatible(self, raw_indices: torch.Tensor, documents: int) -> bool:
        return bool(
            self.output_physical.device == raw_indices.device
            and self.output_physical.dtype == torch.int32
            and self.output_physical.shape == raw_indices.shape
            and self.output_raw.device == raw_indices.device
            and self.output_raw.dtype == torch.int32
            and self.output_raw.shape == raw_indices.shape
            and self.bucket_counts.device == raw_indices.device
            and self.bucket_counts.dtype == torch.int32
            and self.bucket_counts.shape
            == (int(raw_indices.shape[0]), int(documents) + 1)
            and self.output_lengths.device == raw_indices.device
            and self.output_lengths.dtype == torch.int32
            and self.output_lengths.shape == (int(raw_indices.shape[0]),)
        )


def allocate_native_indexer_bucket_workspace(
    raw_indices: torch.Tensor, *, documents: int
) -> NativeIndexerBucketWorkspace:
    if raw_indices.ndim != 2 or raw_indices.dtype != torch.int32:
        raise ValueError("native Indexer raw indices must be a 2-D int32 tensor")
    query_rows, width = (int(value) for value in raw_indices.shape)
    if query_rows <= 0 or width <= 0:
        raise ValueError("native Indexer raw indices cannot be empty")
    return NativeIndexerBucketWorkspace(
        output_physical=torch.empty_like(raw_indices),
        output_raw=torch.empty_like(raw_indices),
        bucket_counts=torch.empty(
            (query_rows, int(documents) + 1),
            dtype=torch.int32,
            device=raw_indices.device,
        ),
        output_lengths=torch.empty(
            (query_rows,), dtype=torch.int32, device=raw_indices.device
        ),
    )


def compact_indexer_topk_by_document(
    *,
    raw_indices: torch.Tensor,
    physical_indices: torch.Tensor,
    input_lengths: torch.Tensor,
    policy: NativeIndexerBucketPolicy,
    workspace: NativeIndexerBucketWorkspace | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    NativeIndexerBucketWorkspace,
]:
    """Return compact physical/raw Top-K plus lengths and exact bucket counts."""

    if not isinstance(policy, NativeIndexerBucketPolicy):
        raise TypeError("policy must be NativeIndexerBucketPolicy")
    if (
        raw_indices.ndim != 2
        or physical_indices.shape != raw_indices.shape
        or raw_indices.dtype != torch.int32
        or physical_indices.dtype != torch.int32
        or raw_indices.device != physical_indices.device
    ):
        raise ValueError("native Indexer raw/physical tensors are incompatible")
    if input_lengths.shape != (int(raw_indices.shape[0]),):
        raise ValueError("native Indexer length tensor has the wrong shape")
    if input_lengths.dtype != torch.int32 or input_lengths.device != raw_indices.device:
        raise ValueError("native Indexer lengths must be device-local int32")
    if int(raw_indices.shape[1]) < policy.indexer_topk:
        raise ValueError("native Indexer buffer is narrower than policy Top-K")
    if workspace is None or not workspace.compatible(raw_indices, policy.documents):
        workspace = allocate_native_indexer_bucket_workspace(
            raw_indices, documents=policy.documents
        )
    workspace.output_physical.fill_(-1)
    workspace.output_raw.fill_(-1)
    workspace.bucket_counts.zero_()
    workspace.output_lengths.zero_()
    block_k = 128
    _compact_indexer_topk_by_document_kernel[
        (
            int(raw_indices.shape[0]),
            triton.cdiv(policy.indexer_topk, block_k),
        )
    ](
        raw_indices,
        physical_indices,
        input_lengths,
        workspace.output_physical,
        workspace.output_raw,
        workspace.bucket_counts,
        workspace.output_lengths,
        int(raw_indices.shape[0]),
        raw_indices.stride(0),
        raw_indices.stride(1),
        physical_indices.stride(0),
        physical_indices.stride(1),
        workspace.output_physical.stride(0),
        workspace.output_physical.stride(1),
        workspace.bucket_counts.stride(0),
        workspace.bucket_counts.stride(1),
        policy.document_compressed_rows,
        TOPK=policy.indexer_topk,
        DOCUMENTS=policy.documents,
        CAP=policy.per_document_cap,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    lengths = torch.clamp(
        workspace.output_lengths, min=1, max=policy.indexer_topk
    )
    return (
        workspace.output_physical,
        workspace.output_raw,
        lengths,
        workspace.bucket_counts,
        workspace,
    )
