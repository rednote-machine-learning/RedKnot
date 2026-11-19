"""Windowed cross-layer offline SWA movement; original online math is unchanged.

Only original immutable prepared halos are moved together. Fresh online KV,
projection, RoPE, attention, global compression, ring writes and layer39 remain
layer ordered. No tensors from another request/owner enter a pack lease.
"""
from types import MethodType
from fixed15_online import assemble_online, online_counters
from kvpack.pack import KVPackSession, gather_plan_from_cache

STAGES = ((3, 8), (9, 14), (15, 20), (21, 39))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def window_layers(layer, stage, memory_rows, width_bytes, budget_bytes):
    first, last = STAGES[stage]
    last = min(last, 38)
    require(first <= layer <= last, 'Layer is outside its completed selection stage')
    # Include worst-case src/dst metadata for every memory row. At least two
    # layers must fit, otherwise explicitly delegate to the original code.
    per_layer = memory_rows * width_bytes + 8
    metadata = memory_rows * 16
    fit = min(4, (budget_bytes-metadata)//max(per_layer, 1), last-layer+1)
    count = 4 if fit >= 4 else 2 if fit >= 2 else 0
    return tuple(range(layer, layer+count))


class PackedAttention:
    def __init__(self, controller):
        self.controller = controller
        self.attention = controller.attention
        self.original_memory = self.attention._memory
        self.original_begin_phase = self.attention.begin_phase
        self.enabled = False
        self.budget_bytes = 512 * 1024**2
        self.checks = None
        self.session = self.lease = self.group = None
        self.stage = None
        self.events = []
        self.fallbacks = []
        self.generation = 0
        self._lease_geometry = None
        self._lease_extra = {}
        self._cached_geometry = None
        self._stage_gather_plan = None
        self._gather_stage = None
        self.gather_events = []
        self._closed = False
        self.binding_generation = 0
        self.binding_events = []
        self.memory_hook = self.phase_hook = None
        self._attach(self.attention, reason='controller_install')

    def _attach(self, attention, *, reason):
        require(not self._closed and self.group is None and self.lease is None
                and getattr(self, 'fixed_group', None) is None,
                'Cannot attach KV pack during an active group or after close')
        require(callable(getattr(attention, '_memory', None)) and
                callable(getattr(attention, 'begin_phase', None)), 'Invalid native attention replacement')
        if hasattr(self.controller, 'cache'):
            require(attention.cache is self.controller.cache,
                    'Current attention does not own the current controller cache')
        self.attention = attention
        self.original_memory = attention._memory
        self.original_begin_phase = attention.begin_phase
        # Resolve self.memory dynamically. FixedPackedAttention overrides it;
        # binding PackedAttention.memory directly would silently bypass its
        # original fixed-group lifecycle and layer-order checks.
        self.memory_hook = MethodType(lambda unused, *a, **k: self.memory(*a, **k), attention)
        self.phase_hook = MethodType(lambda unused, *a, **k: self.begin_phase(*a, **k), attention)
        attention._memory, attention.begin_phase = self.memory_hook, self.phase_hook
        self.binding_generation += 1
        self.binding_events.append(dict(generation=self.binding_generation, reason=reason,
            attention_type=type(attention).__name__, controller_attention_is_current=True,
            memory_dispatch='dynamic_helper_override', active_group=False))

    def _assert_binding(self):
        require(not self._closed and self.attention is self.controller.attention,
                'KV pack is bound to a replaced native attention object')
        require(self.attention.__dict__.get('_memory') is self.memory_hook and
                self.attention.__dict__.get('begin_phase') is self.phase_hook,
                'KV pack attention hooks were replaced by another owner')
        if hasattr(self.controller, 'cache'):
            require(self.attention.cache is self.controller.cache,
                    'KV pack attention/controller cache identity differs')

    def _restore_hooks(self):
        require(self.attention.__dict__.get('_memory') is self.memory_hook and
                self.attention.__dict__.get('begin_phase') is self.phase_hook,
                'Cannot remove KV pack hooks owned by another component')
        self.attention._memory = self.original_memory
        self.attention.begin_phase = self.original_begin_phase

    def ensure_attached(self):
        """Rebind only at phase configuration, after original startup replacement.

        Native multiowner startup deliberately replaces both cache and attention
        after controller.install(). Hooks installed earlier otherwise remain on
        an abandoned object. No native startup or cache policy is modified here.
        """
        require(not self._closed and self.group is None and self.lease is None
                and getattr(self, 'fixed_group', None) is None,
                'Cannot rebind KV pack while a group is active or after close')
        if self.attention is not self.controller.attention:
            self._restore_hooks()
            if self.session is not None:
                self.session.close()
            self.session = None
            self._cached_geometry = self._stage_gather_plan = self._gather_stage = None
            self._attach(self.controller.attention, reason='native_attention_replaced_before_phase')
        self._assert_binding()

    def configure(self, *, enabled, budget_bytes=512*1024**2):
        require(type(enabled) is bool and type(budget_bytes) is int and budget_bytes > 0, 'Invalid KV pack configuration')
        require(self.group is None and self.lease is None and getattr(self, 'fixed_group', None) is None,
                'Cannot reconfigure an active pack group')
        self.ensure_attached()
        self.enabled, self.budget_bytes = enabled, budget_bytes
        self.fuse_live_rows = enabled and getattr(self.controller, '_fixed_pack_method', None) in ('Fixed15Opt','Cone15')
        if hasattr(self.controller, '_fixed_pack_method'):
            from lastkv_packed_only import install_lastkv
            install_lastkv(self.controller.cache, enabled=self.fuse_live_rows)

    def begin_phase(self, *args, **kwargs):
        self._assert_binding()
        require(self.group is None and self.lease is None, 'Previous pack group was not released')
        if self.session is not None:
            self.session.close()
        self.session = None
        self.online_stats = online_counters()
        self.events, self.fallbacks = [], []
        self.gather_events = []
        self._stage_gather_plan = self._gather_stage = None
        self.generation += 1
        return self.original_begin_phase(*args, **kwargs)

    def begin_group(self, stage, group):
        self._assert_binding()
        require(self.group is None and self.lease is None, 'Overlapping stage groups')
        if not self.enabled:
            return
        self.group, self.stage = group, stage
        self._cached_geometry = None
        if self.session is None:
            c = self.controller
            owner = '|'.join(map(str, (c.rid, c.input_sha, c.phase, self.generation)))
            self.session = KVPackSession(owner, max_bytes=self.budget_bytes, torch_module=c.t)
        if self._gather_stage != stage:
            if self._gather_stage is not None:
                self.session.release_gather_geometry(str(self._gather_stage))
            self._stage_gather_plan = None
            self._gather_stage = stage

    def _release(self):
        if self.lease is not None:
            event = self.lease.report()
            event.update(self._lease_extra)
            event['assembled_layers'] = sorted(self.lease.assembled)
            self.events.append(event)
            self.lease.release()
            self.lease = self._lease_geometry = None
            self._lease_extra = {}

    def end_group(self, stage, group):
        if not self.enabled:
            return
        require(self.group is group and self.stage == stage, 'KV group owner changed')
        self._release()
        self.group = self.stage = None
        self._cached_geometry = None

    def memory(self, mod, prepared, meta, ctx):
        self._assert_binding()
        a, c, layer = self.attention, self.controller, mod.layer_idx
        if not self.enabled or layer == 39 or layer < 3:
            return self.original_memory(mod, prepared, meta, ctx)
        require(self.group is not None and self.session is not None and a.enabled,
                'Packed memory called outside the configured progressive stage/group')
        require((ctx.start, ctx.end, tuple(ctx.positions_cpu)) ==
                (self.group.start, self.group.end, tuple(self.group.positions_cpu)), 'Packed group geometry changed')
        a._assert_selection_binding()
        previous_positions, previous = a.previous.get(layer, ([], prepared[:0]))
        mapping = a._mapping(prepared, meta, ctx, previous_positions)
        event = mapping['event']
        geometry = (ctx.start, ctx.end, tuple(ctx.positions_cpu), tuple(previous_positions))
        if self.lease is None:
            if not any(kind == 'halo' for kind, dst, src in mapping['transfers']):
                self.fallbacks.append(dict(layer=layer,stage=self.stage,start=ctx.start,end=ctx.end,
                    reason='no_offline_rows_to_move'))
                return self.original_memory(mod, prepared, meta, ctx)
            layers = window_layers(layer, self.stage, len(event['memory_positions']),
                                   512*prepared.element_size(), self.budget_bytes)
            # Bound both the staged destination and halos pulled forward from
            # later layers. Previously materialized baseline halos are not new
            # allocations caused by this window.
            bytes_per_row = 512*prepared.element_size()
            offline_count = len(a.cache.index20_offline_positions)
            def additional_bytes(window):
                prefetched = sum(l != layer and l not in a.halos for l in window) * offline_count * bytes_per_row
                staged = len(window)*len(event['memory_positions'])*bytes_per_row
                metadata = (len(window)+2*len(event['memory_positions']))*8
                normalized = sum(l not in a.halos for l in window)*offline_count*bytes_per_row
                # Document/local-row geometry remains cached for this stage.
                geometry_bytes = 2*offline_count*8
                peak = max(normalized+prefetched+geometry_bytes,
                           staged+prefetched+metadata+geometry_bytes)
                return prefetched, staged+metadata, normalized, peak
            while layers and additional_bytes(layers)[-1] > self.budget_bytes:
                layers = layers[:2] if len(layers) == 4 else ()
            if not layers:
                self.fallbacks.append(dict(layer=layer, stage=self.stage, start=ctx.start, end=ctx.end,
                                           reason='two_layers_exceed_memory_budget_or_single_remaining_layer'))
                return self.original_memory(mod, prepared, meta, ctx)
            # Future offline halos are independent of live hidden. Restrict the
            # window to this stage so no future selection barrier owns a halo.
            prefetch_bytes, staged_bytes, normalized_bytes, peak_bytes = additional_bytes(layers)
            missing = tuple(l for l in layers if l not in a.halos)
            if missing:
                if self._stage_gather_plan is None:
                    first,last = STAGES[self.stage]
                    self._stage_gather_plan = gather_plan_from_cache(a.cache,range(first,min(last,38)+1),
                        owner_key=self.session.owner_key,stage_key=str(self.stage),group_key='stage-offline-halo')
                with c.allow_sync():
                    gathered = self.session.gather_cache(a.cache,self._stage_gather_plan.for_layers(missing))
                try:
                    absolute = a._indices(a.cache.index20_offline_positions,prepared.device,meta.positions.dtype)
                    for l in missing:
                        canonical = gathered.normalized(l)
                        if self.checks is not None:
                            with c.allow_sync():
                                positions,reference = a.cache.halo(l)
                                self.checks.check_kv(reference,canonical,kind='normalized_gather',layer=l,
                                    stage=self.stage,group_key=gathered.plan.group_key,
                                    positions=positions,source_plan_sha256=gathered.plan.signature)
                                del reference
                        a.halos[l] = a.cache.ops.prepare_swa(c.layers[l].self_attn,canonical,absolute)
                    del canonical
                    self.gather_events.append(gathered.report())
                finally:
                    gathered.release()
            halos = {l:a.halos[l] for l in layers}
            if self._cached_geometry is None:
                require(all(kind in ('current','previous','halo') for kind in event['source_kinds']),
                        'Unknown/uncovered SWA memory source')
                self._cached_geometry = self.session.geometry_from_mapping(mapping,
                    source_positions=a.cache.index20_offline_positions,stage_key=str(self.stage),
                    group_key=str(ctx.start)+':'+str(ctx.end))
            with c.allow_sync():
                self.lease = self.session.scatter_cached_halos(layers, halos, self._cached_geometry)
            self._lease_extra = dict(early_future_halo_bytes=prefetch_bytes,
                normalized_temporary_bytes=normalized_bytes,
                bounded_window_additional_bytes=peak_bytes)
            self._lease_geometry = geometry
        lease = self.lease
        require(geometry == self._lease_geometry and layer in lease.plan.layers,
                'Cross-layer geometry or stage differs from the packed lease')
        lease._check()
        require(layer not in lease.assembled, 'Layer consumed its staged memory twice')
        out = lease.memory[lease.plan.layers.index(layer)]
        # Reuse the existing device index tensors: never rebuild/copy row
        # indices per layer. Only offline transfers are replaced by fused copy.
        sources = dict(current=prepared, previous=previous)
        if self.fuse_live_rows:
            retained = assemble_online(c.t, out, prepared, previous, mapping, self.online_stats)
        else:
            for kind, dst, src in mapping['transfers']:
                if kind != 'halo':
                    out.index_copy_(0, dst, sources[kind].index_select(0, src))
        lease.assembled.add(layer)
        if self.checks is not None:
            with c.allow_sync():
                reference = prepared.new_empty(out.shape)
                original_sources = dict(sources, halo=a._halo(mod, prepared.device, meta.positions.dtype))
                for kind, dst, src in mapping['transfers']:
                    reference.index_copy_(0, dst, original_sources[kind].index_select(0, src))
                self.checks.check_kv(reference, out, kind='assembled_memory', layer=layer,
                    stage=self.stage, group_key=lease.plan.group_key,
                    positions=event['memory_positions'], source_plan_sha256=lease.plan.signature)
                del reference
        if not self.fuse_live_rows:
            combined = c.t.cat((previous, prepared), dim=0)
            retained = combined.index_select(0, mapping['kept'])
        elif self.checks is not None:
            with c.allow_sync():
                expected = c.t.cat((previous, prepared), dim=0).index_select(0, mapping['kept'])
                require(c.t.equal(retained.contiguous().view(c.t.uint8), expected.contiguous().view(c.t.uint8)),
                        'Fused kept SWA rows differ from native cat/index_select')
                self.online_stats['retained_tail_bytechecks'] += 1
        a.previous[layer] = (list(mapping['kept_positions']), retained)
        a._system_counts['optimized_memory_calls'] += 1
        result = out, mapping['positions'], mapping['indices'], event
        if layer == lease.plan.layers[-1]:
            # The returned out still owns storage. Same CUDA stream ordering
            # protects it until the original attention consumes it.
            self._release()
        return result

    def report(self):
        return dict(schema='s13-packed-attention-runtime-v1', enabled=self.enabled,
            fixed15_online=dict(getattr(self, 'online_stats', online_counters())),
            attention_binding_generation=self.binding_generation,
            attention_binding_events=list(self.binding_events),
            attention_is_current=self.attention is self.controller.attention,
            memory_hook_is_owned=self.attention.__dict__.get('_memory') is self.memory_hook,
            phase_hook_is_owned=self.attention.__dict__.get('begin_phase') is self.phase_hook,
            phase_generation=self.generation, max_new_pack_bytes=self.budget_bytes,
            high_water_new_pack_bytes=0 if self.session is None else self.session.high_water_bytes,
            actual_fused_kernel_launches=sum(e['scatter_data_kernel_launches'] for e in self.events),
            actual_fused_bank_gather_launches=sum(e['gather_data_kernel_launches'] for e in self.gather_events),
            actual_layers_moved=sum(len(e['layers']) for e in self.events),
            document_bank_gathers=list(self.gather_events),
            leases=list(self.events), explicit_single_layer_fallbacks=list(self.fallbacks),
            selected_online_fill_uses_original_device_indices=True,
            original_native_prepare_swa_and_layer39=True, cross_owner_tensor_sharing=False,
            pure_layer_dependency_order_preserved=True, active_lease=self.lease is not None)

    def close(self):
        if self._closed:
            return
        self._release()
        if self.session is not None:
            self.session.close()
        self.group = self.stage = None
        self._restore_hooks()
        self._closed = True
