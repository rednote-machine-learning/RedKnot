"""S1 global assembly without full current-chunk raw temporaries.

The immutable bank, phase base, compressor, paged stores and decoder state all
retain the original implementation. Only the source rows supplied to those
operations change. Qualification deliberately also builds the independent full
current chunk used by the original numerical checks.
"""
from __future__ import annotations

from .core import require
from .s1_cache import GLOBAL_SOURCES, S1Cache


class S1SystemCache(S1Cache):
    def __init__(self, *, enabled=False, **kwargs):
        super().__init__(**kwargs)
        require(type(enabled) is bool, "system cache enabled must be bool")
        self._configured_enabled = enabled
        self._phase_enabled = False
        self.system_global_update_events = []

    def configure(self, *, enabled):
        """Set the next phase's implementation; never switch an active phase."""
        require(type(enabled) is bool, "system cache enabled must be bool")
        self._configured_enabled = enabled

    def begin_phase(self, phase, *, build_global_base=True):
        self._phase_enabled = self._configured_enabled
        self.system_global_update_events = []
        return super().begin_phase(phase, build_global_base=build_global_base)

    def _current_rows(self, layer, positions, online_raw, selected_lookup, ctx, counts):
        """Gather just requested absolute rows, preserving raw FP32 column order."""
        positions = list(positions)
        require(positions == sorted(set(positions)), "system raw rows must be sorted and unique")
        width = 512 if layer == 20 else 1024
        dtype = self.t.bfloat16 if layer == 20 else self.t.float32
        out = self.t.empty((len(positions), width), device=self.device, dtype=dtype)
        online_dst, online_src, offline = [], [], {}
        carry_dst = []
        for dst, position in enumerate(positions):
            if position == ctx.start - 1 and ctx.start % 2 and layer != 20:
                carry_dst.append(dst)
            elif position in selected_lookup:
                online_dst.append(dst)
                online_src.append(selected_lookup[position])
            else:
                require(ctx.start <= position < ctx.end, "system raw row outside current chunk")
                document = next((d for d in self.docs if d["start"] <= position < d["end"]), None)
                require(document is not None, "system non-document row has no online projection")
                rows = offline.setdefault(document["key"], ([], []))
                rows[0].append(dst)
                rows[1].append(position-document["start"])
        for key, (dst, src) in offline.items():
            values = self.entry(key, "global_raw", layer)
            out.index_copy_(0, self._indices(dst, out.device),
                            values.index_select(0, self._indices(src, values.device)))
            counts["offline_raw_rows_gathered"] += len(src)
        if online_dst:
            out.index_copy_(0, self._indices(online_dst, out.device),
                            online_raw.index_select(0, self._indices(online_src, online_raw.device)))
            counts["online_raw_rows_gathered"] += len(online_src)
        if carry_dst:
            carry = self.global_carry.get(layer)
            require(len(carry_dst) == 1 and carry is not None and carry[0] == ctx.start-1,
                    "S1 missing C2 phase carry")
            out[carry_dst[0]].copy_(carry[1])
            counts["previous_carry_rows_gathered"] += 1
        return out

    def global_update(self, mod, online_raw, kv_cache, full_meta, ctx):
        if not self._phase_enabled:
            return super().global_update(mod, online_raw, kv_cache, full_meta, ctx)
        layer, ratio = mod.layer_idx, mod.kv_compress_ratio
        require(layer in GLOBAL_SOURCES and self.next_chunk[layer] == ctx.start,
                "S1 global source/chunk order")
        require(ratio == (1 if layer == 20 else 2), "system global source compression ratio")
        require(layer in self.global_bases, "S1 phase global base not prepared")
        require(0 <= ctx.start < ctx.end <= self.total, "system global chunk bounds")
        selected = list(ctx.positions_cpu)
        require(selected == [p for p in self.selected if ctx.start <= p < ctx.end],
                "global online selection differs")
        require(tuple(online_raw.shape) == (len(selected), 512 if ratio == 1 else 1024),
                "global projected shape")
        require(online_raw.dtype == (self.t.bfloat16 if ratio == 1 else self.t.float32),
                "S1 source raw projection precision changed")
        require(online_raw.device == self.device, "system online projection device differs")
        begin_id, end_id = ctx.start//ratio, ctx.end//ratio
        canonical = self.global_bases[layer][begin_id:end_id].clone()
        affected = sorted({p//ratio for p in selected if p//ratio < end_id})
        if ratio == 2 and ctx.start % 2 and ctx.start-1 in self.selected_set:
            affected = sorted(set(affected) | {ctx.start//2})
        counts = dict(layer=layer, phase_generation=self.phase_generation,
            chunk_start=ctx.start, chunk_end=ctx.end, enabled=True,
            full_current_raw_rows_materialized=0, qualification_full_raw_rows=0,
            canonical_rows_cloned=end_id-begin_id, affected_pair_member_rows=0,
            offline_raw_rows_gathered=0, online_raw_rows_gathered=0,
            previous_carry_rows_gathered=0, compressor_tail_rows_gathered=0,
            source20_direct_overlay_rows=0, affected_pair_compressor_calls=0,
            qualification_compressor_calls=0, qualification_native_store_checks=0,
            native_compressor_state_update_calls=0,
            full_canonical_native_store_rows=max(1, end_id-begin_id))
        lookup = {p: row for row, p in enumerate(selected)}
        if ratio == 1:
            if selected:
                canonical.index_copy_(0, self._indices([p-ctx.start for p in selected], canonical.device), online_raw)
            counts["source20_direct_overlay_rows"] = len(selected)
        elif affected:
            positions = [p for pair in affected for p in (pair*2, pair*2+1)]
            members = self._current_rows(layer, positions, online_raw, lookup, ctx, counts)
            # The exact native call and [gate512, latent512] FP32 pair order
            # match the original affected-pair path, including its row count.
            updated = self.ops.compress_canonical(mod, members.contiguous())
            canonical.index_copy_(0, self._indices([p-begin_id for p in affected], canonical.device), updated)
            counts["affected_pair_member_rows"] = len(positions)
            counts["affected_pair_compressor_calls"] = 1

        checks, current = 0, None
        if getattr(ctx, "verify", False) or getattr(ctx, "mode", "") == "reference_sparse":
            # This intentionally retains the original independent full-current
            # oracle and original native paged payload/SF verification below.
            current = self._offline_raw(layer, ctx.start, ctx.end)
            current.index_copy_(0, self._indices([p-ctx.start for p in selected], current.device), online_raw)
            whole = current
            if ctx.start % ratio:
                carry = self.global_carry.get(layer)
                require(carry is not None and carry[0] == ctx.start-1, "S1 reference carry absent")
                whole = self.t.cat((carry[1].unsqueeze(0), current), dim=0)
            expected = self.ops.compress_canonical(mod, whole[:(end_id-begin_id)*ratio])
            require(self.t.equal(canonical, expected), "S1 affected-only/global full compression byte parity failed")
            counts["qualification_full_raw_rows"] = ctx.end-ctx.start
            counts["full_current_raw_rows_materialized"] = ctx.end-ctx.start
            counts["qualification_compressor_calls"] = 1
            checks = 1

        tail = None
        if ratio == 2:
            keep = min(ctx.end-ctx.start, kv_cache.ca_state.shape[1])
            require(keep > 0, "system native compressor state has no tail capacity")
            tail = self._current_rows(layer, range(ctx.end-keep, ctx.end), online_raw, lookup, ctx, counts)
            counts["compressor_tail_rows_gathered"] = keep
        if ctx.end % ratio:
            self.global_carry[layer] = (ctx.end-1, tail[-1].clone())
        else:
            self.global_carry.pop(layer, None)
        # Retain native's reserved -1 padding slot for an incomplete C2 pair.
        stored = canonical if end_id > begin_id else self.t.zeros((1, 512), device=self.device, dtype=self.t.bfloat16)
        require(stored.shape[0] == full_meta.positions_c[mod.ratio_key].numel(),
                "global native compressed metadata rows")
        self.ops.store_main_and_index(mod, stored, kv_cache, full_meta)
        if checks:
            self.ops.verify_native_store(mod, current, kv_cache, full_meta)
            counts["qualification_native_store_checks"] = 1
            checks += 1
        if ratio == 2:
            # NativeCacheOps itself retains min(rows, state_size) rows and uses
            # absolute end to compute carry metadata. Passing exactly that tail
            # produces identical arguments to its state-writing kernel.
            self.ops.update_compressor_carry(mod, tail, kv_cache, full_meta, ctx.end)
            counts["native_compressor_state_update_calls"] = 1
        self.next_chunk[layer] = ctx.end
        self.global_events.append(dict(layer=layer, phase_generation=self.phase_generation,
            chunk_start=ctx.start, chunk_end=ctx.end, compression_ratio=ratio,
            online_projection_rows=len(selected), full_raw_rows=ctx.end-ctx.start,
            online_projection_positions=selected, complete_compressed_ids=list(range(begin_id, end_id)),
            affected_compressed_ids=affected, offline_base_rows=end_id-begin_id-len(affected), checks=checks,
            max_absolute_error=0.0 if checks else None, source2_untouched=True,
            main_and_index_k_native_store=True, global_base_representation="compressed_prenorm_bf16",
            packed_main_and_index_store_checked=bool(checks), native_compressor_carry_updated=ratio == 2))
        self.system_global_update_events.append(counts)
        return stored

    def report(self):
        result = super().report()
        result["system_global_update_diagnostics"] = dict(
            schema="s1-system-global-update-diagnostics-v1", enabled=self._phase_enabled,
            baseline_global_update_unchanged=not self._phase_enabled,
            original_global_event_schema_preserved=True,
            full_raw_rows_event_is_logical_chunk_extent=True,
            numerical_qualification_uses_original_full_current_oracle=True,
            native_paged_store_and_compressor_state_kernels_unchanged=True,
            events=list(self.system_global_update_events))
        return result
