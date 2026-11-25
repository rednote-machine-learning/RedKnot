"""Opt-2 Triton DirectCopy and pre-bound halo gather planning.

This module deliberately performs NO top-level torch/triton/numpy imports so
CPU preflight (`prepare_native`) never loads GPU libraries before release.
"""
from __future__ import annotations

from bisect import bisect_left

_COPY_ROWS_KERNEL = None


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _get_copy_rows_kernel():
    global _COPY_ROWS_KERNEL
    if _COPY_ROWS_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _copy_rows(source, inverse, dst_rows, src_rows, output, width: tl.constexpr):
            row = tl.program_id(0)
            dest = tl.load(dst_rows + row).to(tl.int64)
            src = tl.load(inverse + tl.load(src_rows + row).to(tl.int64)).to(tl.int64)
            cols = tl.arange(0, width)
            payload = tl.load(source + tl.maximum(src, 0) * width + cols, mask=src >= 0, other=0)
            tl.store(output + dest * width + cols, payload)

        _COPY_ROWS_KERNEL = _copy_rows
    return _COPY_ROWS_KERNEL


def plan_layer_halo_gather(universe, wanted_positions, bound_groups):
    """Build CPU reachability & per-document slice/index plan for a single layer.

    Args:
        universe: Sorted sequence of all offline halo positions for the request.
        wanted_positions: None (all universe rows, L3..L20) or sorted sequence of
            reachable offline positions (L21..L38 under MinimalCone).
        bound_groups: Sequence of `(doc_key, lo, hi, src_list, src_gpu)` tuples
            where `[lo, hi)` is the contiguous destination slice in `universe`.

    Returns:
        `(keep, inverse, doc_ops)` where:
          - `keep`: list[int] of indices into `universe` that must be materialized.
          - `inverse`: list[int] of length `len(universe)` mapping universe index
            -> compacted index in `0..len(keep)-1`, or `-1` if unreachable.
          - `doc_ops`: list of `(doc_key, target_slice, mode, payload)` tuples where
            `mode` is `'slice'` (`payload` is `slice(a, b)`), `'full_gpu'` (`payload`
            is `src_gpu`), or `'sub_gpu'` (`payload` is `(src_gpu, local_ids)`).
    """
    n_universe = len(universe)
    if wanted_positions is None:
        keep = list(range(n_universe))
    else:
        keep = []
        for pos in wanted_positions:
            idx = bisect_left(universe, pos)
            require(idx < n_universe and universe[idx] == pos,
                    f"Reachable offline position {pos} absent from bound halo universe")
            keep.append(idx)

    inverse = [-1] * n_universe
    for compact_idx, univ_idx in enumerate(keep):
        inverse[univ_idx] = compact_idx

    doc_ops = []
    for key, lo, hi, src_list, src_gpu in bound_groups:
        left = bisect_left(keep, lo)
        right = bisect_left(keep, hi)
        if left == right:
            continue
        local_ids = [keep[i] - lo for i in range(left, right)]
        sub_src = [src_list[i] for i in local_ids]
        if sub_src[-1] - sub_src[0] + 1 == len(sub_src) and all(
            a < b for a, b in zip(sub_src, sub_src[1:])
        ):
            doc_ops.append((key, slice(left, right), "slice", slice(int(sub_src[0]), int(sub_src[-1]) + 1)))
        elif len(sub_src) == len(src_list):
            doc_ops.append((key, slice(left, right), "full_gpu", src_gpu))
        else:
            doc_ops.append((key, slice(left, right), "sub_gpu", (src_gpu, local_ids)))
    return keep, inverse, doc_ops


def fill_direct_halos(halos, inverse, mapping, output, *, torch_module=None):
    """Copy normalized/rotated live halo rows directly into PackedAttention output.

    Unreachable rows (`inverse[src] < 0`) are zero-filled in-place.
    Uses Triton `_copy_rows` on CUDA tensors and a byte-equivalent Torch fallback
    on CPU tensors for offline testing.
    """
    require(output.is_contiguous() and halos.is_contiguous()
            and tuple(output.shape[1:]) == tuple(halos.shape[1:]) == (1, 512),
            "DirectCopy halo buffers must be contiguous [rows, 1, 512]")
    if getattr(output, "is_cuda", False):
        kernel = _get_copy_rows_kernel()
        for kind, dst, src in mapping["transfers"]:
            if kind == "halo" and dst.numel():
                kernel[(dst.numel(),)](
                    halos if halos.numel() else output,
                    inverse,
                    dst,
                    src,
                    output,
                    512,
                    num_warps=4,
                )
        return
    # CPU fallback (identical zero-fill and copy semantics for offline verification).
    t = torch_module
    if t is None:
        import torch as t
    for kind, dst, src in mapping["transfers"]:
        if kind != "halo" or not dst.numel():
            continue
        mapped = inverse.index_select(0, src.long())
        live = mapped >= 0
        if live.any():
            output.index_copy_(0, dst[live].long(), halos.index_select(0, mapped[live].long()))
        dead = ~live
        if dead.any():
            output.index_fill_(0, dst[dead].long(), 0)
