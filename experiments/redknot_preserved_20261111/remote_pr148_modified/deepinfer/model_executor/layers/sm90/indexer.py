"""FP4 index attention with dense/candidate and contiguous/paged KV addressing."""

import torch
import triton
import triton.language as tl

from .quant import decode_e2m1, decode_ue8m0


@triton.jit
def _row_offset(row: tl.tensor, stride: tl.constexpr):
    # Cast BEFORE multiplication: 8192 queries with a context just above256K
    # exceed2**31 BF16 elements after the512-element stride padding.
    return row.to(tl.int64) * stride


@triton.jit
def _mqa_logits(
    Q: tl.tensor,
    QS: tl.tensor,
    K: tl.tensor,
    KS: tl.tensor,
    W: tl.tensor,
    START: tl.tensor,
    END: tl.tensor,
    BLOCKS: tl.tensor,
    TABLE: tl.tensor,
    QUERY_SEQ: tl.tensor,
    OUT: tl.tensor,
    H: tl.constexpr,
    NH: tl.constexpr,
    QM: tl.constexpr,
    QH: tl.constexpr,
    QSM: tl.constexpr,
    QSH: tl.constexpr,
    WM: tl.constexpr,
    KM: tl.constexpr,
    KSM: tl.constexpr,
    NK: tl.constexpr,
    WIDTH: tl.constexpr,
    OM: tl.constexpr,
    CANDIDATE: tl.constexpr,
    CBM: tl.constexpr,
    CBS: tl.constexpr,
    PAGED: tl.constexpr,
    PAGE: tl.constexpr,
    KB: tl.constexpr,
    TABLE_M: tl.constexpr,
    TABLE_N: tl.constexpr,
    SEQUENCES: tl.constexpr,
    BT: tl.constexpr = 64,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BT + tl.arange(0, BT)
    heads = tl.arange(0, NH)
    dims = tl.arange(0, 128)
    end = tl.load(END + row).to(tl.int64)
    start = 0 if PAGED else tl.load(START + row).to(tl.int64)
    logical = columns
    valid = columns < WIDTH
    if CANDIDATE:
        candidate = tl.load(BLOCKS + row * CBM + columns // CBS, valid, -1).to(tl.int64)
        logical = candidate * CBS + columns % CBS
        valid &= candidate >= 0
    valid &= (logical >= 0) & (logical < end - start)
    if PAGED:
        sequence = tl.load(QUERY_SEQ + row).to(tl.int64)
        valid &= (sequence >= 0) & (sequence < SEQUENCES) & (logical // PAGE < TABLE_N)
        page = tl.load(TABLE + _row_offset(sequence, TABLE_M) + logical // PAGE, valid, -1).to(tl.int64)
        valid &= (page >= 0) & (page < NK)
        token = logical % PAGE
        key_base = _row_offset(page, KB) + token * 64
        scale_base = _row_offset(page, KB) + PAGE * 64 + token * 4
        packed_k = tl.load(K + key_base[None, :] + dims[:, None] // 2, valid[None, :], 0).to(tl.uint8)
        kc = tl.load(K + scale_base[None, :] + dims[:, None] // 32, valid[None, :], 0)
    else:
        token = start + logical
        valid &= (token >= 0) & (token < NK)
        packed_k = tl.load(K + token[None, :] * KM + dims[:, None] // 2, valid[None, :], 0).to(tl.uint8)
        kp = tl.load(KS + token * KSM, valid, 0)
        kc = (kp[None, :] >> (8 * (dims[:, None] // 32))) & 255
    key = decode_e2m1((packed_k >> (4 * (dims[:, None] % 2))) & 15) * decode_ue8m0(kc)
    packed_q = tl.load(Q + row * QM + heads[:, None] * QH + dims[None, :] // 2, heads[:, None] < H, 0).to(tl.uint8)
    qp = tl.load(QS + row * QSM + heads * QSH, heads < H, 0)
    qc = (qp[:, None] >> (8 * (dims[None, :] // 32))) & 255
    query = decode_e2m1((packed_q >> (4 * (dims[None, :] % 2))) & 15) * decode_ue8m0(qc)
    scores = tl.dot(query.to(tl.bfloat16), key.to(tl.bfloat16))
    weights = tl.load(W + row * WM + heads, heads < H, 0).to(tl.float32)
    result = tl.sum(tl.maximum(scores, 0) * weights[:, None], 0)
    result = tl.where(valid, result, float("-inf"))
    tl.store(OUT + _row_offset(row, OM) + columns, result, columns < WIDTH)


def mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor | None,
    ends: torch.Tensor,
    max_seqlen: int,
    *,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    block_table: torch.Tensor | None = None,
    query_indices: torch.Tensor | None = None,
    active_columns_bound: int | None = None,
) -> torch.Tensor:
    """Return compact BF16 logits, masking invalid candidates before top-k.

    ``active_columns_bound`` is an explicit native-prefill contract: contiguous
    non-candidate KV and all sequence starts nonnegative. The caller guarantees
    the latter from CPU metadata construction; this function never reads device
    starts back to verify it. The bound must cover min(logical_width, loaded_KV
    rows). The logical shape/stride and masked -Inf tail remain unchanged while
    unnecessary column tiles are omitted. Default None preserves the general
    interface, including negative starts, paged KV, and candidate coordinates.
    """
    x, xs = q
    m, h, packed_dim = x.shape
    if packed_dim != 64 or x.dtype != torch.int8 or xs.dtype != torch.int32:
        raise ValueError("SM90 indexer expects E2M1 queries of head dimension 128 with UE8M0 scales")
    paged = isinstance(kv, torch.Tensor)
    k, ks = (kv, None) if paged else kv
    if paged and (block_table is None or query_indices is None):
        raise ValueError("Paged index attention requires block tables and query-to-sequence indices")
    if not paged and starts is None:
        raise ValueError("Contiguous index attention requires sequence start offsets")
    width = max_seqlen if candidate_blocks is None else candidate_blocks.shape[1] * candidate_block_size
    launch_width = width
    if active_columns_bound is not None:
        if paged or candidate_blocks is not None:
            raise ValueError("active_columns_bound requires contiguous non-candidate KV with nonnegative starts")
        if type(active_columns_bound) is not int or not 0 <= active_columns_bound <= width:
            raise ValueError("active_columns_bound must be an integer between zero and logical width")
        if active_columns_bound < min(width, k.shape[0]):
            raise ValueError("active_columns_bound must cover at least min(logical width, loaded KV rows)")
        launch_width = active_columns_bound
    stride = triton.cdiv(width, 512) * 512
    out = torch.empty((m, stride), dtype=torch.bfloat16, device=x.device)[:, :width]
    if m == 0 or width == 0:
        return out
    if launch_width < width:
        # Keep downstream top-k shape, indexing, and tie behavior identical.
        # With start>=0, logical>=NK implies start+logical>=NK, always invalid.
        out.fill_(float("-inf"))
    if launch_width == 0:
        return out
    xs = xs.reshape(m, h)
    _mqa_logits[(m, triton.cdiv(launch_width, 64))](
        x,
        xs,
        k,
        ks,
        weights,
        starts,
        ends,
        candidate_blocks,
        block_table,
        query_indices,
        out,
        h,
        max(16, triton.next_power_of_2(h)),
        x.stride(0),
        x.stride(1),
        *xs.stride(),
        weights.stride(0),
        k.stride(0),
        ks.stride(0) if ks is not None else 0,
        k.shape[0],
        width,
        out.stride(0),
        candidate_blocks is not None,
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        candidate_block_size,
        paged,
        k.shape[1] if paged else 1,
        k.stride(0),
        block_table.stride(0) if paged else 0,
        block_table.shape[1] if paged else 0,
        block_table.shape[0] if paged else 0,
        num_warps=4,
    )
    return out
