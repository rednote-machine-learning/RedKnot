"""Build just the two inputs consumed by the native layer-39 replay projection."""
from bisect import bisect_left
from types import MethodType


def packed_only(self, start, end, selected, online_packed, online_bf16):
    from index20_lastkv_cache import copy_native_fp8_rows
    from index20_plan import require

    torch = self.t
    width = self.lastkv_input_width
    require(0 <= start < end <= self.total, 'Invalid last-KV replay interval')
    require(isinstance(online_packed, tuple) and len(online_packed) == 2, 'Expected native packed input and scales')
    data, scales = online_packed
    require(tuple(data.shape) == tuple(online_bf16.shape) == (len(selected), width)
            and tuple(scales.shape) == (len(selected), width // 128)
            and data.dtype == torch.float8_e4m3fn and scales.dtype == torch.int32
            and online_bf16.dtype == torch.bfloat16
            and data.device == scales.device == online_bf16.device == self.device,
            'Last-KV input geometry or dtype changed')
    n = end - start
    output = torch.empty((n, width), device=self.device, dtype=data.dtype)
    scale_storage = torch.zeros((width // 128, ((n + 3) // 4) * 4), device=self.device, dtype=scales.dtype)
    output_scales = scale_storage[:, :n].transpose(0, 1)
    lo = bisect_left(selected, start)
    hi = bisect_left(selected, end, lo)
    # The controller binds ordered, unique selection; verify it once per group.
    if getattr(self, '_pack15_last_ordered', None) is not selected:
        require(list(selected) == sorted(set(selected)) and all(type(p) is int and 0 <= p < self.total for p in selected), 'Invalid live selection')
        self._pack15_last_ordered = selected
    doc_rows = overlap = 0
    previous_end = start
    for doc in self.docs:
        a, z = max(start, doc['start']), min(end, doc['end'])
        if a >= z:
            continue
        require(a >= previous_end, 'Overlapping document intervals')
        previous_end = z
        target = slice(a - start, z - start)
        source = slice(a - doc['start'], z - doc['start'])
        # Retain the original bank version/scope validation, without copying the unused BF16 rows.
        self.entry(doc['key'], 'last_attn_bf16', 39)
        output[target].copy_(self.entry(doc['key'], 'last_attn_packed', 39)[source])
        output_scales[target].copy_(self.entry(doc['key'], 'last_attn_sf', 39)[source])
        doc_rows += z - a
        overlap += bisect_left(selected, z, lo, hi) - bisect_left(selected, a, lo, hi)
    live = hi - lo
    require(doc_rows + live - overlap == n, 'Missing noncached online attention input')
    if live:
        dst = self._indices([p - start for p in selected[lo:hi]], self.device)
        src = torch.arange(lo, hi, device=self.device, dtype=torch.long)
        copy_native_fp8_rows(output, dst, data, src, torch)
        output_scales.index_copy_(0, dst, scales.index_select(0, src))
    self._pack15_last_replayed_rows += n
    return (output, output_scales), None, dict(start=start, end=end, full_rows=n,
        online_input_rows=live, offline_input_rows=n-live, source_coverage_complete=True,
        post_mhc_attn_norm_input=True, native_packed_input_preserved=True, bf16_requantized=False)


def install_lastkv(cache, enabled):
    """Toggle only the per-instance replay source for the optimized arm."""
    if enabled and '_pack15_last_original' not in vars(cache):
        cache._pack15_last_original = cache.materialize_last_attention_inputs
        cache._pack15_last_replayed_rows = 0
        cache.materialize_last_attention_inputs = MethodType(packed_only, cache)
    if not enabled and '_pack15_last_original' in vars(cache):
        cache.materialize_last_attention_inputs = vars(cache).pop('_pack15_last_original')
        vars(cache).pop('_pack15_last_ordered', None)
