"""Bitwise SWA row transfer for a single RedKnot owner, group and layer."""
import triton
import triton.language as tl


@triton.jit
def fill_online_and_keep(CUR, PREV, CURRENT_DST, CURRENT_SRC, PREVIOUS_DST,
                         PREVIOUS_SRC, KEPT, OUT, TAIL, ncur, nprev, prior_len,
                         current_stride, previous_stride, output_stride,
                         CURRENT_COL: tl.constexpr, PREVIOUS_COL: tl.constexpr,
                         OUT_COL: tl.constexpr, ELEMENT_BYTES: tl.constexpr,
                         WIDTH: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    # No floating-point instructions. Preserve signed zeros and NaN payloads.
    if ELEMENT_BYTES == 2:
        cur = CUR.to(tl.pointer_type(tl.uint16))
        prev = PREV.to(tl.pointer_type(tl.uint16))
        out = OUT.to(tl.pointer_type(tl.uint16))
        tail = TAIL.to(tl.pointer_type(tl.uint16))
    else:
        cur = CUR.to(tl.pointer_type(tl.uint32))
        prev = PREV.to(tl.pointer_type(tl.uint32))
        out = OUT.to(tl.pointer_type(tl.uint32))
        tail = TAIL.to(tl.pointer_type(tl.uint32))
    if row < ncur:
        src = tl.load(CURRENT_SRC + row).to(tl.int64)
        dst = tl.load(CURRENT_DST + row).to(tl.int64)
        value = tl.load(cur + src * current_stride + col * CURRENT_COL)
        tl.store(out + dst * output_stride + col * OUT_COL, value)
    elif row < ncur + nprev:
        slot = row - ncur
        src = tl.load(PREVIOUS_SRC + slot).to(tl.int64)
        dst = tl.load(PREVIOUS_DST + slot).to(tl.int64)
        value = tl.load(prev + src * previous_stride + col * PREVIOUS_COL)
        tl.store(out + dst * output_stride + col * OUT_COL, value)
    else:
        slot = row - ncur - nprev
        src = tl.load(KEPT + slot).to(tl.int64)
        if src < prior_len:
            value = tl.load(prev + src * previous_stride + col * PREVIOUS_COL)
        else:
            value = tl.load(cur + (src-prior_len) * current_stride + col * CURRENT_COL)
        tl.store(tail + slot * WIDTH + col, value)
