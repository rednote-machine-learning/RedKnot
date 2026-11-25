"""Drive the existing fused-copy helper from an unchanged fixed-budget group loop.

The four ranges below are copy-window boundaries only. They are NOT selection
stages: fixed-budget obtains one native layer2 max selection and retains it to layer39.
"""
from packed_attention import PackedAttention, STAGES


def require(value, message):
    if not value:
        raise RuntimeError(message)


class FixedPackedAttention(PackedAttention):
    def __init__(self, controller):
        self.fixed_group = None
        self.fixed_geometry = None
        self.fixed_layers = []
        self.fixed_group_events = []
        super().__init__(controller)

    def begin_phase(self, *args, **kwargs):
        require(self.fixed_group is None, 'Unreleased fixed-budget execution group')
        self.fixed_group_events = []
        return super().begin_phase(*args, **kwargs)

    def enter_fixed_group(self, group):
        require(self.enabled and self.fixed_group is None and self.group is None,
                'Fixed pack execution group overlaps or is disabled')
        self.fixed_group = group
        self.fixed_geometry = (group.start, group.end, tuple(group.positions_cpu))
        self.fixed_layers = []

    def _end_window_region(self):
        if self.group is not None:
            self.end_group(self.stage, self.group)

    def exit_fixed_group(self, *, completed):
        group = self.fixed_group
        require(group is not None, 'No fixed execution group to release')
        try:
            self._end_window_region()
            geometry = (group.start, group.end, tuple(group.positions_cpu))
            require(geometry == self.fixed_geometry, 'Fixed selection/group changed while executing')
            if completed:
                expected = list(range(3, 40)) if group.positions_cpu else []
                require(self.fixed_layers == expected,
                        'Fixed20 group did not consume exactly its original layer3..39 memories')
            self.fixed_group_events.append(dict(start=group.start, end=group.end,
                selected_positions=list(group.positions_cpu),
                actual_memory_layers=list(self.fixed_layers), completed=completed,
                layer_order='original_group_major_3_through_39',
                selection_changed=False, groups_repartitioned=False))
        finally:
            if self.direct_halo_copy and getattr(self.controller, 'ids', None) is not None and group.end == len(self.controller.ids):
                self._release_direct_halos()
            self.fixed_group = self.fixed_geometry = None
            self.fixed_layers = []

    def memory(self, mod, prepared, meta, ctx):
        layer = mod.layer_idx
        if not self.enabled or layer < 3:
            return super().memory(mod, prepared, meta, ctx)
        require(self.fixed_group is not None, 'Packed fixed memory outside original deep group')
        require((ctx.start, ctx.end, tuple(ctx.positions_cpu)) == self.fixed_geometry,
                'Packed fixed memory request geometry differs')
        require(layer == 3 + len(self.fixed_layers) and layer <= 39,
                'Original fixed-budget layer order was skipped, repeated or reordered')
        if layer == 39:
            # LastKV all-input regeneration and its ring assembly stay native.
            self._end_window_region()
        else:
            stage = next(i for i, (first, last) in enumerate(STAGES) if first <= layer <= last)
            if self.stage != stage:
                self._end_window_region()
                self.begin_group(stage, self.fixed_group)
        result = super().memory(mod, prepared, meta, ctx)
        self.fixed_layers.append(layer)
        return result

    def report(self):
        result = super().report()
        policy = getattr(self.controller, '_v2_policy', None)
        sel_layers = [2, policy.two_stage_mid_layer] if (policy is not None and policy.two_stage_l14) else [2]
        result.update(fixed_group_active=self.fixed_group is not None,
            fixed_group_events=list(self.fixed_group_events),
            copy_window_ranges=[[a, min(z, 38)] for a, z in STAGES],
            copy_windows_are_selection_stages=False,
            algorithm_selection_layers=sel_layers, regroup_enabled=False)
        return result

    def close(self):
        if self.fixed_group is not None:
            self.exit_fixed_group(completed=False)
        return super().close()
