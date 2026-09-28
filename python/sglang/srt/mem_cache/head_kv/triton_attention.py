"""Direct head-paged CUDA attention; imported lazily by attention.py.

One program computes one query/head/page partition. K/V are loaded directly
from slab pages, with online softmax within partitions and a second reduction
across partitions. This initial kernel targets decode and small repair/query
batches, not high-throughput dense prefill.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _head_paged_attention(
    Q, K, V, SLOTS, LENGTHS, KPOS, QPOS, WINDOWS, SINKS, OUT, PARTIAL,
    q_h: tl.constexpr, q_n: tl.constexpr, q_d: tl.constexpr,
    k_s: tl.constexpr, k_b: tl.constexpr, k_d: tl.constexpr,
    v_s: tl.constexpr, v_b: tl.constexpr, v_d: tl.constexpr,
    slot_h: tl.constexpr, slot_p: tl.constexpr,
    len_h: tl.constexpr, len_p: tl.constexpr,
    pos_h: tl.constexpr, pos_p: tl.constexpr, pos_b: tl.constexpr,
    qp_h: tl.constexpr, qp_n: tl.constexpr,
    win_h: tl.constexpr, sink_h: tl.constexpr,
    out_h: tl.constexpr, out_n: tl.constexpr, out_d: tl.constexpr,
    DIM: tl.constexpr, PAGE_SIZE: tl.constexpr, NUM_PAGES: tl.constexpr,
    NUM_QUERIES: tl.constexpr, NUM_SPLITS: tl.constexpr, PAGES_PER_SPLIT: tl.constexpr,
    GROUP_SIZE: tl.constexpr, SCALE: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_B: tl.constexpr,
):
    head = tl.program_id(0)
    query_idx = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head // GROUP_SIZE
    dims = tl.arange(0, BLOCK_D)
    tokens = tl.arange(0, BLOCK_B)
    query = tl.load(Q + head * q_h + query_idx * q_n + dims * q_d, dims < DIM, 0).to(tl.float32)
    query_pos = tl.load(QPOS + head * qp_h + query_idx * qp_n)
    window = tl.load(WINDOWS + kv_head * win_h)
    sink = tl.load(SINKS + kv_head * sink_h)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((BLOCK_D,), 0.0, tl.float32)
    for local_page in range(PAGES_PER_SPLIT):
        page = split * PAGES_PER_SPLIT + local_page
        slot = tl.load(SLOTS + kv_head * slot_h + page * slot_p, page < NUM_PAGES, -1).to(tl.int64)
        length = tl.load(LENGTHS + kv_head * len_h + page * len_p, page < NUM_PAGES, 0)
        if (slot >= 0) & (length > 0):
            for start in range(0, PAGE_SIZE, BLOCK_B):
                offsets = start + tokens
                active = (offsets < length) & (offsets < PAGE_SIZE)
                positions = tl.load(KPOS + kv_head * pos_h + page * pos_p + offsets * pos_b, active, -1)
                visible = active
                if CAUSAL:
                    visible = visible & (positions <= query_pos)
                visible = visible & ((window <= 0) | (positions >= query_pos - window + 1) | (positions < sink))
                keys = tl.load(K + slot * k_s + offsets[:, None] * k_b + dims[None, :] * k_d,
                               active[:, None] & (dims[None, :] < DIM), 0).to(tl.float32)
                scores = tl.sum(keys * query[None, :], axis=1) * SCALE
                scores = tl.where(visible, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, axis=0))
                safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
                old_weight = tl.exp(maximum - safe_max)
                weights = tl.exp(scores - safe_max)
                values = tl.load(V + slot * v_s + offsets[:, None] * v_b + dims[None, :] * v_d,
                                 active[:, None] & (dims[None, :] < DIM), 0).to(tl.float32)
                numerator = numerator * old_weight + tl.sum(weights[:, None] * values, axis=0)
                denominator = denominator * old_weight + tl.sum(weights, axis=0)
                maximum = new_max
    if NUM_SPLITS == 1:
        result = tl.where(denominator > 0, numerator / tl.maximum(denominator, 1.0e-30), 0.0)
        tl.store(OUT + head * out_h + query_idx * out_n + dims * out_d, result, dims < DIM)
    else:
        base = ((head * NUM_QUERIES + query_idx) * NUM_SPLITS + split) * (DIM + 2)
        tl.store(PARTIAL + base + dims, numerator, dims < DIM)
        tl.store(PARTIAL + base + DIM, maximum)
        tl.store(PARTIAL + base + DIM + 1, denominator)


@triton.jit
def _merge_partitions(
    PARTIAL, OUT,
    DIM: tl.constexpr, NUM_QUERIES: tl.constexpr, NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    head = tl.program_id(0)
    query_idx = tl.program_id(1)
    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, BLOCK_D)
    base = ((head * NUM_QUERIES + query_idx) * NUM_SPLITS + splits) * (DIM + 2)
    maxima = tl.load(PARTIAL + base + DIM, splits < NUM_SPLITS, -float("inf"))
    denominators = tl.load(PARTIAL + base + DIM + 1, splits < NUM_SPLITS, 0)
    maximum = tl.max(maxima, axis=0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(maxima - safe_max)
    partial = tl.load(PARTIAL + base[:, None] + dims[None, :],
                      (splits[:, None] < NUM_SPLITS) & (dims[None, :] < DIM), 0)
    numerator = tl.sum(partial * weights[:, None], axis=0)
    denominator = tl.sum(denominators * weights, axis=0)
    result = tl.where(denominator > 0, numerator / tl.maximum(denominator, 1.0e-30), 0.0)
    tl.store(OUT + (head * NUM_QUERIES + query_idx) * DIM + dims, result, dims < DIM)


def triton_paged_attention(
    query, k_pool, v_pool, page_slots, page_lengths, key_positions,
    query_positions, windows, sinks, num_q_per_kv, scale, causal,
):
    """Launch with *already validated* descriptors under a live caller read lease.

    This internal entry point intentionally omits the public API's synchronizing
    value checks. Mutating descriptors or reusing physical slots after validation
    invalidates its safety contract. Record a CUDA completion event before
    releasing the manager's read lease.
    """
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if query.shape[1] == 0:
        return output
    block_d = triton.next_power_of_2(query.shape[2])
    block_b = min(32, triton.next_power_of_2(k_pool.shape[1]))
    # Long-context decode needs more parallelism than one CTA per query head.
    # Partition only page traversal; the workspace contains O(Hq*Nq*splits*D)
    # online-softmax statistics, never a gathered KV sequence or score matrix.
    row_count = query.shape[0] * query.shape[1]
    desired_splits = min(32, max(1, triton.cdiv(256, row_count)))
    num_splits = min(desired_splits, max(1, triton.cdiv(page_slots.shape[1], 8)))
    pages_per_split = triton.cdiv(page_slots.shape[1], num_splits)
    partial = (torch.empty((query.shape[0], query.shape[1], num_splits, query.shape[2] + 2),
                           dtype=torch.float32, device=query.device)
               if num_splits > 1 else output)
    _head_paged_attention[(query.shape[0], query.shape[1], num_splits)](
        query, k_pool, v_pool, page_slots, page_lengths, key_positions,
        query_positions, windows, sinks, output, partial,
        *query.stride(), *k_pool.stride(), *v_pool.stride(),
        *page_slots.stride(), *page_lengths.stride(), *key_positions.stride(),
        *query_positions.stride(), windows.stride(0), sinks.stride(0),
        *output.stride(),
        query.shape[2], k_pool.shape[1], page_slots.shape[1],
        query.shape[1], num_splits, pages_per_split,
        num_q_per_kv, scale, causal, block_d, block_b,
        num_warps=4 if block_b * block_d <= 8192 else 8,
    )
    if num_splits > 1:
        _merge_partitions[(query.shape[0], query.shape[1])](
            partial, output, query.shape[2], query.shape[1], num_splits,
            block_d, triton.next_power_of_2(num_splits), num_warps=4,
        )
    return output
