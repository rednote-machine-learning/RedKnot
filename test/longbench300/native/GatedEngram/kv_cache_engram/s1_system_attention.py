"""Phase-local storage optimization of the original S1 attention policy.

Projection, normalization/RoPE/casting, joint attention, global key ownership,
cross-chunk KV values and final physical ring writes remain in S1Attention.
Only immutable integer geometry and ephemeral local-memory storage are reused.
Configure before reset/begin_phase; all preparation occurs inside that phase.
"""
from __future__ import annotations

from .core import require
from .s1_attention import S1Attention, local_memory_plan


class S1SystemAttention(S1Attention):
    def __init__(self, cache, *, torch_module=None, allow_sync=None, enabled=False):
        require(type(enabled) is bool, "system attention enabled must be bool")
        self._configured_enabled = enabled
        self._system_generation = 0
        super().__init__(cache, torch_module=torch_module, allow_sync=allow_sync)

    def configure(self, *, enabled):
        """Choose the next phase's implementation; never switch an active phase."""
        require(type(enabled) is bool, "system attention enabled must be bool")
        self._configured_enabled = enabled
        return self

    def begin_phase(self):
        super().begin_phase()
        self.enabled = self._configured_enabled
        self._system_generation += 1
        self._system_scope = None
        self._device_indices, self._device_geometry, self._workspaces = {}, {}, {}
        self._system_counts = dict(index_allocations=0, index_reuses=0,
            device_geometry_builds=0, device_geometry_reuses=0,
            local_workspace_allocations=0, local_workspace_reuses=0,
            optimized_memory_calls=0)

    def _storage_key(self, device, dtype):
        # Reuse on one stream is ordered by CUDA. Different streams must never
        # write the same scratch storage without an explicit dependency.
        kind = getattr(device, "type", str(device).split(":", 1)[0])
        stream = (int(self.t.cuda.current_stream(device=device).cuda_stream)
                  if kind == "cuda" else None)
        return str(device), str(dtype), stream

    def _indices(self, values, device, dtype=None):
        if not self.enabled:
            return super()._indices(values, device, dtype)
        dtype = self.t.long if dtype is None else dtype
        values = tuple(values)
        key = self._storage_key(device, dtype), values
        if key not in self._device_indices:
            # Original transfer/allow_sync behavior, only on the first use.
            self._device_indices[key] = super()._indices(values, device, dtype)
            self._system_counts["index_allocations"] += 1
        else:
            self._system_counts["index_reuses"] += 1
        return self._device_indices[key]

    def _bind_geometry_scope(self):
        scope = (self.cache.total, tuple((d["key"], d["start"], d["end"])
                                        for d in self.cache.docs))
        if self._system_scope is None:
            self._system_scope = scope
        require(self._system_scope == scope, "system attention input changed without begin_phase")

    def _mapping(self, prepared, meta, ctx, previous_positions):
        self._bind_geometry_scope()
        positions_cpu = tuple(ctx.positions_cpu)
        previous_positions = tuple(previous_positions)
        # Unlike a cache keyed only by lengths, this cannot alias two different
        # absolute query layouts or a different cross-chunk previous set.
        host_key = ctx.start, ctx.end, positions_cpu, previous_positions
        if host_key not in self.geometry:
            self.geometry[host_key] = local_memory_plan(self.cache.docs, self.cache.total,
                ctx.start, ctx.end, positions_cpu, previous_positions)
        event = self.geometry[host_key]
        storage = self._storage_key(prepared.device, meta.positions.dtype)
        key = host_key, storage
        if key in self._device_geometry:
            self._system_counts["device_geometry_reuses"] += 1
            mapping = self._device_geometry[key]
        else:
            t = self.t
            transfers = []
            for kind in ("current", "previous", "halo"):
                dst = [i for i, source in enumerate(event["source_kinds"]) if source == kind]
                if dst:
                    src = [event["source_rows"][i] for i in dst]
                    transfers.append((kind, self._indices(dst, prepared.device),
                                      self._indices(src, prepared.device)))
            candidates = previous_positions + positions_cpu
            kept = [i for i, p in enumerate(candidates) if p >= ctx.end-127]
            memory_positions = self._indices(event["memory_positions"], prepared.device, meta.positions.dtype)
            queries = self._indices(positions_cpu, prepared.device, meta.positions.dtype)
            # Keep the original integer GPU construction and coverage check.
            # Its result is immutable and shared across layers with this layout.
            low = t.clamp(queries-127, min=0)
            wanted = low[:, None] + t.arange(128, device=prepared.device, dtype=queries.dtype)[None, :]
            valid = wanted <= queries[:, None]
            if memory_positions.numel():
                slots = t.searchsorted(memory_positions, wanted.contiguous())
                safe = slots.clamp(max=memory_positions.numel()-1)
                t._assert_async(((~valid) | ((slots < memory_positions.numel()) &
                    (memory_positions[safe] == wanted))).all(), "S1 GPU local-window key missing")
                indices = t.where(valid, slots, -1).to(t.int32).contiguous()
            else:
                indices = t.empty((0, 128), device=prepared.device, dtype=t.int32)
            mapping = dict(event=event, transfers=transfers, positions=memory_positions,
                queries=queries, indices=indices, kept=self._indices(kept, prepared.device),
                kept_positions=[candidates[i] for i in kept])
            self._device_geometry[key] = mapping
            self._system_counts["device_geometry_builds"] += 1
        require(tuple(meta.positions.shape) == (len(positions_cpu),), "S1 system query geometry shape")
        # Do not silently trust CPU positions if a caller changes device metadata.
        self.t._assert_async((meta.positions == mapping["queries"]).all(),
                             "S1 system query absolute positions changed")
        return mapping

    def _memory(self, mod, prepared, meta, ctx):
        if not self.enabled:
            return super()._memory(mod, prepared, meta, ctx)
        t, layer = self.t, mod.layer_idx
        previous_positions, previous = self.previous.get(layer, ([], prepared[:0]))
        mapping = self._mapping(prepared, meta, ctx, previous_positions)
        event = mapping["event"]
        halo = self._halo(mod, prepared.device, meta.positions.dtype)
        shape = len(event["memory_positions"]), 1, 512
        key = self._storage_key(prepared.device, prepared.dtype), shape
        if key not in self._workspaces:
            self._workspaces[key] = prepared.new_empty(shape)
            self._system_counts["local_workspace_allocations"] += 1
        else:
            self._system_counts["local_workspace_reuses"] += 1
        out = self._workspaces[key]
        sources = dict(current=prepared, previous=previous, halo=halo)
        # Same gather/copy order as the original. Every output row is assigned
        # once by the full local-memory plan; no previous layer's values survive.
        for kind, dst, src in mapping["transfers"]:
            out.index_copy_(0, dst, sources[kind].index_select(0, src))
        combined = t.cat((previous, prepared), dim=0)
        # Own a new tensor: the state for the next chunk never aliases scratch.
        self.previous[layer] = (list(mapping["kept_positions"]),
                                combined.index_select(0, mapping["kept"]))
        self._system_counts["optimized_memory_calls"] += 1
        # The caller consumes this scratch on the same stream before the next
        # _memory call. _joint copies it; neither previous nor the offline bank
        # retains an alias. The original joint/ring implementations are inherited.
        return out, mapping["positions"], mapping["indices"], event

    def report(self):
        result = super().report()
        result["system_optimization"] = dict(schema="s1-system-attention-diagnostics-v1",
            enabled=self.enabled, phase_generation=self._system_generation,
            counters=dict(self._system_counts),
            device_geometry_entries=len(self._device_geometry),
            device_index_entries=len(self._device_indices),
            local_workspace_entries=len(self._workspaces),
            local_workspace_bytes=sum(x.numel()*x.element_size() for x in self._workspaces.values()),
            preparation_within_online_phase=True, cross_phase_device_reuse=False,
            offline_bank_mutated=False, original_projection_and_joint_attention=True,
            original_previous127_and_final_ring128=True,
            baseline_dispatches_original_implementation=not self.enabled,
            timing_measurements=None)
        return result
