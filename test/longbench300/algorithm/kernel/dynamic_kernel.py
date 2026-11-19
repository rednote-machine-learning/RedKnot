"""SM90 per-32 GEMM with runtime row count and activation scale strides."""
import inspect

import triton
import triton.language as tl

from deepinfer.model_executor.layers.sm90.quant import decode_e2m1, decode_ue8m0

DYNAMIC_ARGUMENTS = ("M", "ASM", "ASK")
JIT_OPTIONS = {"do_not_specialize": list(DYNAMIC_ARGUMENTS)}
if "do_not_specialize_on_alignment" in inspect.signature(triton.jit).parameters:
    JIT_OPTIONS["do_not_specialize_on_alignment"] = list(DYNAMIC_ARGUMENTS)


@triton.jit(**JIT_OPTIONS)
def _gemm_dynamic_m(
    A: tl.tensor,
    AS: tl.tensor,
    B: tl.tensor,
    BS: tl.tensor,
    D: tl.tensor,
    C: tl.tensor,
    PSUM: tl.tensor,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    ASM,
    ASK,
    BG: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BSG: tl.constexpr,
    BSN: tl.constexpr,
    BSK: tl.constexpr,
    DM: tl.constexpr,
    DN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    GROUPS: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    FP4: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BM: tl.constexpr = 32,
    BN_TILE: tl.constexpr = 64,
):
    row_start = tl.program_id(0) * BM
    group = 0
    end = M
    start = 0
    if GROUPS:
        end = tl.load(PSUM)
        while (row_start >= end) & (group + 1 < GROUPS):
            start = tl.cdiv(end, ALIGNMENT) * ALIGNMENT
            group += 1
            end = tl.load(PSUM + group)
    rows = row_start + tl.arange(0, BM)
    cols = tl.program_id(1) * BN_TILE + tl.arange(0, BN_TILE)
    valid_rows = (rows >= start) & (rows < end) & (rows < M)
    acc = tl.full((BM, BN_TILE), 0, tl.float32)
    if row_start < end:
        for block in range(tl.cdiv(K, 32)):
            ks = block * 32 + tl.arange(0, 32)
            av = tl.load(A + rows[:, None] * AM + ks[None, :] * AK, valid_rows[:, None] & (ks[None, :] < K), 0.0)
            ac = tl.load(AS + rows * ASM + (block // 4) * ASK, valid_rows, 0)
            asc = decode_ue8m0((ac >> (8 * (block % 4))) & 255)
            if FP4:
                bv = tl.load(
                    B + group * BG + cols[None, :] * BN + (ks[:, None] // 2) * BK,
                    (cols[None, :] < N) & (ks[:, None] < K),
                    0,
                ).to(tl.uint8)
                bv = decode_e2m1((bv >> (4 * (ks[:, None] % 2))) & 15)
            else:
                bv = tl.load(
                    B + group * BG + cols[None, :] * BN + ks[:, None] * BK,
                    (cols[None, :] < N) & (ks[:, None] < K),
                    0.0,
                ).to(tl.float32)
            bc = tl.load(BS + group * BSG + cols * BSN + (block // 4) * BSK, cols < N, 0)
            bsc = decode_ue8m0((bc >> (8 * (block % 4))) & 255)
            # E2M1/E4M3 values times a power of two are exactly representable
            # in BF16 throughout the model's finite representable range.
            av = (av.to(tl.float32) * asc[:, None]).to(tl.bfloat16)
            bv = (bv * bsc[None, :]).to(tl.bfloat16)
            acc = tl.dot(av, bv, acc)
    if ACCUMULATE:
        acc += tl.load(C + rows[:, None] * CM + cols[None, :] * CN, (rows[:, None] < M) & (cols[None, :] < N), 0).to(tl.float32)
    tl.store(D + rows[:, None] * DM + cols[None, :] * DN, acc, (rows[:, None] < M) & (cols[None, :] < N))
