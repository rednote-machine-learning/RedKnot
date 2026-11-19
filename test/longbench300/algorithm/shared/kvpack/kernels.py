"""One launch moves actual KV values across every requested layer/document.

Pointer tables point at existing banks. No concatenated whole-bank allocation,
floating arithmetic, normalization, RoPE, projection, or attention is performed.
"""
import triton
import triton.language as tl


@triton.jit
def gather_rows(SOURCE_PTRS, DOC_IDS, LOCAL_ROWS, OUT,
                R: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
                BLOCK: tl.constexpr):
    layer = tl.program_id(1)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = x // W, x % W
    valid = row < R
    document = tl.load(DOC_IDS + row, valid, other=0)
    source_row = tl.load(LOCAL_ROWS + row, valid, other=0)
    address = tl.load(SOURCE_PTRS + layer * D + document)
    source = address.to(tl.pointer_type(OUT.dtype.element_ty))
    value = tl.load(source + source_row * W + column, valid, other=0)
    tl.store(OUT + layer * R * W + x, value, valid)


@triton.jit
def scatter_rows(SOURCE_PTRS, SOURCE_ROWS, DESTINATION_ROWS, OUT,
                 R: tl.constexpr, M: tl.constexpr, W: tl.constexpr,
                 BLOCK: tl.constexpr):
    layer = tl.program_id(1)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = x // W, x % W
    valid = row < R
    source = tl.load(SOURCE_PTRS + layer).to(tl.pointer_type(OUT.dtype.element_ty))
    source_row = tl.load(SOURCE_ROWS + row, valid, other=0)
    destination = tl.load(DESTINATION_ROWS + row, valid, other=0)
    value = tl.load(source + source_row * W + column, valid, other=0)
    tl.store(OUT + layer * M * W + destination * W + column, value, valid)


def launch_gather(pointers, documents, rows, output):
    layers, count, width = output.shape
    if count:
        gather_rows[(triton.cdiv(count * width, 1024), layers)](
            pointers, documents, rows, output, count,
            pointers.numel() // layers, width, BLOCK=1024, num_warps=4)


def launch_scatter(pointers, source_rows, destinations, output, rows):
    layers, memory, _, width = output.shape
    if rows:
        scatter_rows[(triton.cdiv(rows * width, 1024), layers)](
            pointers, source_rows, destinations, output, rows, memory, width, BLOCK=1024, num_warps=4)
