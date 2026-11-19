"""Cone15: fixed15 native max with S1.4 offline KV movement fusion.

This factory accepts the installed LastKV controller, optionally wrapped by
the original multiowner controller. All arithmetic and decoder iteration remain
in that parent class. No stage collectors, progressive pruning or regroup exist.
"""
import hashlib
import json
from fixed_pack_policy import ARM_FRACTIONS, budget_admission


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    require('sha256' not in value, 'Cannot reseal evidence')
    return dict(value, sha256=digest(value))


def fixed_pack_controller_class(baseclass):
    class FixedPackController(baseclass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._configured_fixed_pack = False
            self._fixed_pack_budget = 1024**3
            self._fixed_pack_method = 'Cone15'
            self._fixed_pack_fraction = .15
            self._fixed_pack_budget_admission = None
            self.fixed_pack_enabled = False
            self.fixed_pack_attention = None
            self.fixed_pack_checks = None
            self._fixed_pack_selection = None

        def install(self):
            super().install()
            try:
                from fixed_pack_attention import FixedPackedAttention
                require(self.fixed_pack_attention is None, 'Fixed KV pack already installed')
                self.fixed_pack_attention = FixedPackedAttention(self)
                return self
            except BaseException:
                self.close()
                raise

        def configure_fixed_pack(self, enabled, *, method='Cone15', kv_pack_budget_bytes=1024**3):
            require(type(enabled) is bool and self.state in ('NEW', 'IDLE', 'COMPLETE'),
                    'Cannot configure an active fixed pack request')
            require(type(kv_pack_budget_bytes) is int and kv_pack_budget_bytes > 0,
                    'Bounded positive KV pack budget required')
            require(not enabled or method in ARM_FRACTIONS, 'Explicit approved fixed-ratio arm required')
            self._configured_fixed_pack = enabled
            self._fixed_pack_budget = kv_pack_budget_bytes
            self._fixed_pack_method = method if enabled else None
            self._fixed_pack_fraction = ARM_FRACTIONS[method] if enabled else .2
            from fixed15_geometry import configure_geometry
            configure_geometry(self, enabled and method in ARM_FRACTIONS)
            return self

        def configure_fixed_pack_checks(self, checker=None):
            require(self.state in ('NEW', 'IDLE', 'COMPLETE') and
                (checker is None or all(callable(getattr(checker, name, None))
                    for name in ('check_kv', 'report'))), 'Invalid fixed KV checker/lifecycle')
            self.fixed_pack_checks = checker
            return self

        def reset(self, *args, **kwargs):
            self.fixed_pack_enabled = self._configured_fixed_pack
            if self.fixed_pack_enabled:
                mode = args[1] if len(args) > 1 else kwargs.get('mode')
                require(mode == 'sensitive' and self._configured_index20 and
                    self._configured_lastkv and self._index20_strategy == 'max' and
                    self._index20_fraction == self._fixed_pack_fraction,
                    'Cone15 requires explicit native max budget and all-input LastKV')
                require(not getattr(self, '_configured_progressive', False),
                        'A progressive controller cannot masquerade as fixed15')
            helper = self.fixed_pack_attention
            require(not self.fixed_pack_enabled or helper is not None,
                    'Fixed pack helper must install before reset')
            if helper is not None:
                helper.checks = self.fixed_pack_checks
                helper.configure(enabled=self.fixed_pack_enabled, budget_bytes=self._fixed_pack_budget)
            self._replay_start = getattr(self.cache, '_pack15_last_replayed_rows', 0)
            self._fixed_pack_selection = None
            self._fixed_pack_budget_admission = None
            if self.fixed_pack_enabled:
                # Validate the same natural documents BEFORE the parent creates
                # a collector or configures request cache state.
                from rag_natural_documents import natural_cache_documents
                ids = kwargs['input_ids']
                docs = natural_cache_documents(ids, kwargs.get('documents') or ())
                self._fixed_pack_budget_admission = budget_admission(docs, len(ids), self._fixed_pack_fraction)
            return super().reset(*args, **kwargs)

        def _run_deep_group(self, model, caches, group):
            if not self.fixed_pack_enabled:
                return super()._run_deep_group(model, caches, group)
            selected = tuple(self._index20_selection['selected_positions'])
            if self._fixed_pack_selection is None:
                self._fixed_pack_selection = selected
            require(selected == self._fixed_pack_selection and
                list(group.positions_cpu) == [p for p in selected if group.start <= p < group.end],
                'Fixed15 selection or original group ownership drifted')
            helper = self.fixed_pack_attention
            helper.enter_fixed_group(group)
            completed = False
            try:
                result = super()._run_deep_group(model, caches, group)
                require(tuple(self._index20_selection['selected_positions']) == selected,
                        'Fixed15 selection changed after layer2')
                completed = True
                return result
            finally:
                helper.exit_fixed_group(completed=completed)

        def finish(self):
            report = super().finish()
            helper = self.fixed_pack_attention
            pack = None if helper is None else helper.report()
            if self.fixed_pack_enabled:
                require(pack is not None and not pack['active_lease'] and not pack['fixed_group_active'],
                        'Unreleased fixed KV pack memory')
                require(self._fixed_pack_selection == tuple(report['index20']['selection']['selected_positions']),
                        'Final fixed15 selection differs from executed groups')
                original = report['system_batch_execution']['groups']
                actual = pack['fixed_group_events']
                require([(x['start'], x['end'], x['selected_positions']) for x in original] ==
                        [(x['start'], x['end'], x['selected_positions']) for x in actual] and
                        all(x['completed'] for x in actual), 'Original fixed15 group evidence differs')
            report['system_pack_fixed_v1'] = seal(dict(schema='redknot-fixed15-system-pack-v1',
                enabled=self.fixed_pack_enabled, canonical_method=self._fixed_pack_method,
                reference_method='Full', selector_strategy='max', body_fraction=self._fixed_pack_fraction,
                budget_admission=self._fixed_pack_budget_admission,
                geometry_calls=dict(self._fixed15_geometry.calls) if getattr(self,'_fixed15_geometry',None) else {},
                packed_only_replay_rows=getattr(self.cache,'_pack15_last_replayed_rows',0)-self._replay_start,
                algorithm_changed_from_fixed20=self.fixed_pack_enabled and self._fixed_pack_fraction != .2,
                selector_layers=[2], selected_ids_retained_through_layer39=True,
                layer_order='original_group_major_3_through_39', regroup_enabled=False,
                original_first3_global_lastkv_decode_unchanged=self._fixed_pack_method!='Cone15',
                full_selection_is_outer_wrapper_layout=True,
                post20_live_row_execution=self._fixed_pack_method=='Cone15',
                kv_pack_budget_bytes=self._fixed_pack_budget, kv_pack=pack,
                tensor_checks=None if self.fixed_pack_checks is None else self.fixed_pack_checks.report(),
                mathematical_quality_equivalence_claim=False, gpu_speedup_claim=False))
            self.last_report = report
            return report

        def close(self):
            helper = self.fixed_pack_attention
            self.fixed_pack_attention = None
            try:
                if helper is not None:
                    helper.close()
            finally:
                super().close()

    return FixedPackController
