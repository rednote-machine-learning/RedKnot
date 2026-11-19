"""Fill a layer's live rows and retain its next-group SWA tail in one launch."""


def online_counters():
    return dict(layers=0, kernel_launches=0, original_online_copy_ops=0,
                cat_destination_bytes_avoided=0, gather_destination_bytes_avoided=0,
                retained_tail_bytechecks=0)


def assemble_online(torch, out, current, previous, mapping, counters):
    from fixed15_online_triton import fill_online_and_keep

    if any(t.device != out.device or t.dtype != out.dtype or tuple(t.shape[1:]) != (1, 512)
           for t in (out, current, previous)) or not out.is_cuda or out.dtype not in (
               torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('Fused online fill requires same-owner CUDA floating [rows,1,512] inputs')
    kept = mapping['kept']
    # A real (possibly zero-length) tensor from the same owner can stand in for a
    # missing transfer. A zero count makes that branch unreachable in the kernel.
    current_dst = current_src = previous_dst = previous_src = kept
    sizes = {}
    for kind, dst, src in mapping['transfers']:
        if kind == 'halo':
            continue
        if kind in sizes or kind not in ('current', 'previous'):
            raise ValueError('Invalid online transfer in original mapping')
        if dst.shape != src.shape:
            raise ValueError('Original source/destination index shapes differ')
        sizes[kind] = dst.numel()
        if kind == 'current':
            current_dst, current_src = dst, src
        else:
            previous_dst, previous_src = dst, src
    for index in (current_dst, current_src, previous_dst, previous_src, kept):
        if index.device != out.device or index.ndim != 1 or not index.is_contiguous() or index.dtype not in (
                torch.int32, torch.int64):
            raise ValueError('Original device row mapping is not a contiguous owner-local integer vector')
    ncur, nprev = sizes.get('current', 0), sizes.get('previous', 0)
    tail = current.new_empty((kept.numel(), 1, 512))
    rows = ncur + nprev + kept.numel()
    if rows:
        # Empty torch tensors may use address zero. Pass a real typed pointer
        # to each unreachable branch, avoiding any null-specialization issue.
        any_index = kept if kept.numel() else (current_src if ncur else previous_src)
        if not ncur:
            current_dst = current_src = any_index
        if not nprev:
            previous_dst = previous_src = any_index
        fill_online_and_keep[(rows,)](
            current if current.numel() else out, previous if previous.numel() else out,
            current_dst, current_src, previous_dst, previous_src,
            kept if kept.numel() else any_index, out if out.numel() else current,
            tail if tail.numel() else out, ncur, nprev, previous.shape[0], current.stride(0),
            previous.stride(0), out.stride(0), current.stride(2), previous.stride(2),
            out.stride(2), current.element_size(), 512, num_warps=4)
    counters['layers'] += 1
    counters['kernel_launches'] += int(rows > 0)
    counters['original_online_copy_ops'] += 2 * len(sizes) + 2
    row_bytes = 512 * current.element_size()
    counters['cat_destination_bytes_avoided'] += (current.shape[0] + previous.shape[0]) * row_bytes
    counters['gather_destination_bytes_avoided'] += (ncur + nprev) * row_bytes
    return tail
