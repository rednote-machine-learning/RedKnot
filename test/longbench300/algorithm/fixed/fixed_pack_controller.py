"""Cone15 / RedKnot V2 controller with Opt-1 static metadata caching and V2 policy routing.

This factory accepts the installed LastKV controller, optionally wrapped by
the original multiowner controller. Legacy S1.6 mode executes the exact
original path; RedKnot V2 enables MinimalCone, suffix-only Max scoring,
Online_Head, length-adaptive Span, Opt-1/2/3, and optional L2->L14 two-stage.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import time
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
            self._v2_policy_config = None
            self._v2_policy = None
            self._v2_two_stage_callback = None
            self._v2_two_stage_state = None
            self._v2_two_stage_summary = None
            self._v2_static_meta = {}
            self._v2_reset_stats = None

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

        def configure_v2_policy(self, policy_config=None, *, two_stage_callback=None):
            require(self.state in ('NEW', 'IDLE', 'COMPLETE'),
                    'Cannot configure RedKnot V2 policy during an active request')
            self._v2_policy_config = policy_config
            self._v2_two_stage_callback = two_stage_callback
            return self

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

        def clear_cache(self, *args, **kwargs):
            self._v2_static_meta.clear()
            return super().clear_cache(*args, **kwargs)

        @contextmanager
        def _opt1_static_reset_scope(self, ids_tuple, docs_raw):
            """Opt-1: cache immutable CPU request metadata while retaining all live checks."""
            import rag_natural_documents as rag

            started = time.perf_counter()
            cache = self.cache
            description = json.dumps(docs_raw, sort_keys=True, separators=(',', ':'))
            scope_desc = json.dumps(cache.scope, sort_keys=True, separators=(',', ':'))
            key = (ids_tuple, description, scope_desc, self.case_input, self.case_binding)
            cache_state = self._v2_static_meta
            hit = cache_state.get('key') == key
            if not hit:
                cache_state.clear()
            stats = {'route': 'hit' if hit else 'miss', 'natural_hits': 0, 'plan_hits': 0}

            with ExitStack() as inside:
                original_natural = rag.natural_documents

                def cached_natural(callids, docs):
                    if tuple(callids) != ids_tuple or json.dumps(docs, sort_keys=True, separators=(',', ':')) != description:
                        return original_natural(callids, docs)
                    if 'natural' not in cache_state:
                        cache_state['natural'] = original_natural(callids, docs)
                    else:
                        stats['natural_hits'] += 1
                    return [dict(d) for d in cache_state['natural']]

                rag.natural_documents = cached_natural
                inside.callback(setattr, rag, 'natural_documents', original_natural)

                reset_cls = next(
                    (cls for cls in type(self).__mro__
                     if 'reset' in cls.__dict__
                     and 'mandatory_dflash_prepared' in cls.__dict__['reset'].__code__.co_names),
                    None,
                )
                if reset_cls is not None:
                    reset_func = reset_cls.__dict__['reset']
                    table = reset_func.__globals__
                    old_planned = table['mandatory_dflash_prepared']

                    def cached_planned(documents, n):
                        shape = (tuple((v['start'], v['end'], v['key']) for v in documents), n)
                        if cache_state.get('plan_shape') != shape:
                            cache_state['plan'] = old_planned(documents, n)
                            cache_state['plan_shape'] = shape
                        else:
                            stats['plan_hits'] += 1
                        return deepcopy(cache_state['plan'])

                    table['mandatory_dflash_prepared'] = cached_planned
                    inside.callback(table.__setitem__, 'mandatory_dflash_prepared', old_planned)

                    if reset_func.__closure__ and 'token_digest' in reset_func.__code__.co_freevars:
                        cell = dict(zip(reset_func.__code__.co_freevars, reset_func.__closure__))['token_digest']
                        old_token = cell.cell_contents

                        def cached_token_digest(seq):
                            if tuple(seq) != ids_tuple:
                                return old_token(seq)
                            if hit and 'token_sha' in cache_state:
                                return cache_state['token_sha']
                            result = old_token(seq)
                            cache_state['token_sha'] = result
                            return result

                        cell.cell_contents = cached_token_digest
                        inside.callback(setattr, cell, 'cell_contents', old_token)

                entry_cls = next(
                    (cls for cls in type(cache).__mro__
                     if 'entry' in cls.__dict__
                     and 'digest' in cls.__dict__['entry'].__code__.co_names),
                    None,
                )
                if entry_cls is not None:
                    g = entry_cls.__dict__['entry'].__globals__
                    original_digest = g['digest']
                    scope_value = original_digest(cache.scope)

                    def cached_scope_digest(x):
                        return scope_value if x is cache.scope else original_digest(x)

                    g['digest'] = cached_scope_digest
                    inside.callback(g.__setitem__, 'digest', original_digest)

                yield
            cache_state['key'] = key
            stats['seconds'] = time.perf_counter() - started
            self._v2_reset_stats = stats

        def reset(self, *args, **kwargs):
            self.fixed_pack_enabled = self._configured_fixed_pack
            self._v2_policy = None
            self._v2_two_stage_state = None
            self._v2_two_stage_summary = None
            self._v2_reset_stats = None
            if self.fixed_pack_enabled:
                from v2_policy import resolve_request_policy
                ids = kwargs['input_ids']
                policy = resolve_request_policy(self._v2_policy_config, len(ids))
                self._v2_policy = policy
                if policy is not None and policy.is_v2:
                    self._fixed_pack_fraction = float(policy.fraction)
                    self._index20_fraction = float(policy.fraction)
                mode = args[1] if len(args) > 1 else kwargs.get('mode')
                require(mode == 'sensitive' and self._configured_index20 and
                    self._configured_lastkv and self._index20_strategy == 'max' and
                    self._index20_fraction == self._fixed_pack_fraction,
                    'Cone15 requires explicit native max budget and all-input LastKV')
                require(not getattr(self, '_configured_progressive', False),
                        'A progressive controller cannot masquerade as fixed15')
            collector = getattr(self, 'index20_collector', None)
            if collector is not None:
                if self.fixed_pack_enabled and self._v2_policy is not None and self._v2_policy.is_v2:
                    p = self._v2_policy
                    collector.configure_v2(dict(
                        suffix_only=p.suffix_only,
                        head_width=p.head_width,
                        span_window=p.window,
                        gpu_select=p.gpu_select,
                        stage1_fraction=p.stage1_fraction if p.two_stage_l14 else None,
                        two_stage_callback=self._v2_two_stage_callback,
                    ))
                else:
                    collector.configure_v2(None)
            helper = self.fixed_pack_attention
            require(not self.fixed_pack_enabled or helper is not None,
                    'Fixed pack helper must install before reset')
            if helper is not None:
                helper.checks = self.fixed_pack_checks
                use_direct = bool(self.fixed_pack_enabled and self._v2_policy is not None and self._v2_policy.direct_halo_copy)
                audit_halo = bool(self.fixed_pack_enabled and self._v2_policy is not None and self._v2_policy.audit_halo)
                helper.configure(enabled=self.fixed_pack_enabled, budget_bytes=self._fixed_pack_budget,
                                 direct_halo_copy=use_direct, audit_halo=audit_halo)
            self._replay_start = getattr(self.cache, '_pack15_last_replayed_rows', 0)
            self._fixed_pack_selection = None
            self._fixed_pack_budget_admission = None
            if self.fixed_pack_enabled:
                from rag_natural_documents import natural_cache_documents
                ids = kwargs['input_ids']
                docs_raw = kwargs.get('documents') or ()
                use_opt1 = bool(self._v2_policy is not None and self._v2_policy.static_metadata_cache)
                if use_opt1:
                    with self._opt1_static_reset_scope(tuple(ids), list(docs_raw)):
                        docs = natural_cache_documents(ids, docs_raw)
                        self._fixed_pack_budget_admission = budget_admission(
                            docs, len(ids), self._fixed_pack_fraction, policy=self._v2_policy)
                        return super().reset(*args, **kwargs)
                # Validate the same natural documents BEFORE the parent creates
                # a collector or configures request cache state.
                docs = natural_cache_documents(ids, docs_raw)
                self._fixed_pack_budget_admission = budget_admission(
                    docs, len(ids), self._fixed_pack_fraction, policy=self._v2_policy)
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
            is_v2 = bool(self._v2_policy is not None and self._v2_policy.is_v2)
            selector_layers = [2, self._v2_policy.two_stage_mid_layer] if (is_v2 and self._v2_policy.two_stage_l14) else [2]
            pack_payload = dict(schema='redknot-fixed15-system-pack-v1',
                enabled=self.fixed_pack_enabled, canonical_method=self._fixed_pack_method,
                reference_method='Full', selector_strategy='max', body_fraction=self._fixed_pack_fraction,
                budget_admission=self._fixed_pack_budget_admission,
                geometry_calls=dict(self._fixed15_geometry.calls) if getattr(self,'_fixed15_geometry',None) else {},
                packed_only_replay_rows=getattr(self.cache,'_pack15_last_replayed_rows',0)-self._replay_start,
                algorithm_changed_from_fixed20=self.fixed_pack_enabled and self._fixed_pack_fraction != .2,
                selector_layers=selector_layers, selected_ids_retained_through_layer39=True,
                layer_order='original_group_major_3_through_39', regroup_enabled=False,
                original_first3_global_lastkv_decode_unchanged=self._fixed_pack_method!='Cone15',
                full_selection_is_outer_wrapper_layout=True,
                post20_live_row_execution=self._fixed_pack_method=='Cone15',
                kv_pack_budget_bytes=self._fixed_pack_budget, kv_pack=pack,
                tensor_checks=None if self.fixed_pack_checks is None else self.fixed_pack_checks.report(),
                mathematical_quality_equivalence_claim=False, gpu_speedup_claim=False)
            if is_v2:
                pack_payload['v2_policy'] = self._v2_policy.to_dict()
                pack_payload['v2_reset_stats'] = self._v2_reset_stats
            report['system_pack_fixed_v1'] = seal(pack_payload)
            if self._v2_two_stage_summary is not None:
                report['two_stage_l14'] = self._v2_two_stage_summary
            self.last_report = report
            return report

        def close(self):
            self._v2_static_meta.clear()
            self._v2_two_stage_state = None
            helper = self.fixed_pack_attention
            self.fixed_pack_attention = None
            try:
                if helper is not None:
                    helper.close()
            finally:
                super().close()

    return FixedPackController
