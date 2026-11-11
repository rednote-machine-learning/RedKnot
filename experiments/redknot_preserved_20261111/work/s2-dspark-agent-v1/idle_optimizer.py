"""Opt-in world-idle inner-forward bypass; native Worker and sampler remain.

This is an executable candidate, NOT a declaration of GPU qualification. A
fresh real-request hidden/mHC/cache/output parity gate is required by runner.
Only the authenticated eight-rank scheduler frame can authorize a bypass.
"""
from __future__ import annotations

import hashlib
import json

IDLE_METADATA_CONTRACT = dict(num_reqs=1, total_q=0, canonical_request_binding_present=False,
    input_ids_shape=[1], positions_shape=[1], state_block_idx_shape=[1], valid_token_mask_shape=[1])


def idle_metadata_observation(meta):
    """Only host scalar fields/shapes: never read tensor values or synchronize."""
    out = dict(num_reqs=getattr(meta, 'num_reqs', None), total_q=getattr(meta, 'total_q', None),
        canonical_request_binding_present=getattr(meta, 'canonical_request_binding', None) is not None)
    for name in ('input_ids', 'positions', 'state_block_idx', 'valid_token_mask'):
        value = getattr(meta, name, None)
        out[name + '_shape'] = list(value.shape) if value is not None else None
    return out


def require(ok, message):
    if not ok:
        raise ValueError(message)


def seal(value):
    out = dict(value)
    out['sha256'] = hashlib.sha256(json.dumps(out, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return out


def native_capabilities(engine, model):
    """Read actual loaded native configuration; do not trust CLI assertions."""
    from deepinfer import envs
    from deepinfer.model_executor import eplb
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    runner = engine.driver_worker.model_runner
    manager = eplb.current()
    require(manager is not None and manager._load_balancers, 'actual EPLB manager required')
    values = dict(cp_size=PG.context_parallel_group().size(), dp_size=PG.data_parallel_group().size(),
        ep_size=PG.expert_parallel_group().size(), eager=runner.use_cuda_graph is False,
        nextn=int(runner.nextn), worker_nextn=int(engine.driver_worker.nextn),
        model_load_nextn=int(model.load_nextn), dflash_input_size=int(model.dflash_input_size),
        model_num_layers=int(model.num_layers),
        mtp_varlen=bool(envs.DEEPINFER_ENABLE_MTP_VARLEN),
        rearrangement_interval=int(envs.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL),
        rearrange_after_prefill=bool(envs.DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE),
        force_imbalance_ratio=float(envs.DEEPINFER_FORCE_IMBALANCE_RATIO),
        actual_balancer_intervals=[int(lb._rearrangement_interval) for lb in manager._load_balancers],
        actual_balancer_after_prefill=[bool(lb._schedule_rearrangement_after_prefill)
                                      for lb in manager._load_balancers])
    require(values['cp_size'] == 1 and values['dp_size'] == 8 and values['ep_size'] == 8,
            'world-idle adapter is restricted to pinned CP1/DP8/EP8')
    require(values['model_num_layers'] == 40, 'idle bypass requires the pinned forty-layer model')
    require(values['eager'] and values['nextn'] == 0 and values['worker_nextn'] == 0
            and values['model_load_nextn'] == 0 and values['dflash_input_size'] == 0
            and not values['mtp_varlen'], 'idle bypass requires actual eager non-speculative model')
    require(values['rearrangement_interval'] == 0 and not values['rearrange_after_prefill']
            and not any(values['actual_balancer_intervals'])
            and not any(values['actual_balancer_after_prefill'])
            and values['force_imbalance_ratio'] == 0., 'adaptive/random expert routing is unsupported')
    return values


class Candidate:
    def __init__(self, model, controller, engine, rank, batch_worker, torch_module, allow_sync):
        require(type(rank) is int and 0 <= rank < 8, 'rank must be 0..7')
        require(batch_worker is not None and batch_worker.rank == rank
                and batch_worker.model is model and batch_worker.engine is engine,
                'candidate must share the existing all-rank scheduler owner')
        require((rank == 0) == (controller is not None), 'only rank zero owns request state')
        self.model, self.controller, self.engine, self.rank = model, controller, engine, rank
        self.worker, self.t, self.allow_sync = batch_worker, torch_module, allow_sync
        self.enabled, self.installed, self.closed = False, False, False
        self.phase, self.request_id, self.events = None, None, []
        self.active_state_nonzero_checks = 0
        self.capabilities, self.original, self.hook = None, None, None

    def install(self):
        require(not self.installed and not self.closed, 'candidate cannot be reinstalled')
        self.capabilities = native_capabilities(self.engine, self.model)
        self.own = 'forward' in self.model.__dict__
        self.original = self.model.forward
        def forward(kv_caches, input_metadata, hidden_states=None, dflash=None):
            return self.forward(kv_caches, input_metadata, hidden_states, dflash)
        self.hook = forward
        self.model.forward = forward
        self.installed = True
        return self

    def configure(self, *, enabled, phase, request_id):
        require(self.installed and not self.closed and type(enabled) is bool,
                'candidate configuration requires installed owner and boolean enable')
        require(self.worker._world_frame is None, 'cannot reconfigure during an engine step')
        require(isinstance(phase, str) and phase and isinstance(request_id, str) and request_id,
                'phase and request identity required')
        self.enabled, self.phase, self.request_id, self.events = enabled, phase, request_id, []
        self.active_state_nonzero_checks = 0

    def forward(self, caches, meta, hidden, dflash):
        if not self.enabled:
            return self.original(caches, meta, hidden, dflash)
        frame, schedule = self.worker._world_frame, self.worker.schedule
        require(schedule is not None and schedule['request_id'] == self.request_id
                and schedule['phase'] == self.phase and not self.worker._phase_done,
                'candidate request/phase lacks admitted world schedule')
        require(frame is not None, 'candidate forward lacks global scheduler evidence')
        # Active prefill/decode, including locally empty companions, is untouched.
        if frame['world_phase'] != 'idle':
            require(frame['world_phase'] in ('prefill', 'decode'), 'unknown world scheduler phase')
            if self.rank == 0:
                require(frame['local_is_empty'] is False and frame['rank0_request_id'] == self.request_id,
                        'active rank-zero metadata differs from admitted request')
                require(meta.num_reqs == 1 and meta.state_block_idx is not None,
                        'real request state metadata missing')
                # Native block/state slot zero is reserved for placeholders.
                # This assertion does not transfer device metadata to the host.
                self.t._assert_async((meta.state_block_idx > 0).all(), 'real request aliases placeholder state zero')
                self.active_state_nonzero_checks += 1
            return self.original(caches, meta, hidden, dflash)
        require(frame['local_is_empty'] is True and frame['rank0_request_id'] is None
                and frame['start'] is None and frame['end'] is None,
                'idle classification contradicts real request metadata')
        require(frame['scheduler_prompt_run'] is False and frame['local_prompt_run'] is False
                and meta.is_prompt is False, 'only native globally empty decode rounds are supported')
        require(not frame['model_consumed'] and dflash is None, 'idle model consumed twice or speculative')
        # Native InputMetadata.total_q is a prefill-only CP-mirror field.
        # Eager decode keeps its dataclass default 0. ModelRunner separately
        # returns local total_q=1 for the physical placeholder; it does not
        # populate this prefill-only field. Assert actual row shapes below.
        observed_metadata = idle_metadata_observation(meta)
        require(observed_metadata == IDLE_METADATA_CONTRACT,
                'world-idle native decode metadata mismatch: ' + json.dumps(dict(
                    expected=IDLE_METADATA_CONTRACT, observed=observed_metadata), sort_keys=True))
        require(tuple(meta.input_ids.shape) == (1,) and tuple(meta.positions.shape) == (1,),
                'native world-idle placeholder is not one row')
        require(meta.state_block_idx is not None and tuple(meta.state_block_idx.shape) == (1,),
                'native placeholder state metadata missing')
        require(meta.valid_token_mask is not None and tuple(meta.valid_token_mask.shape) == (1,),
                'native placeholder valid-token mask missing')
        # These stay device-side under native ERROR sync-debug mode.
        self.t._assert_async((~meta.valid_token_mask).all(), 'idle placeholder contains a valid token')
        self.t._assert_async((meta.positions == 0).all(), 'idle placeholder absolute position differs')
        self.t._assert_async((meta.input_ids == 0).all(), 'idle placeholder input differs')
        self.t._assert_async((meta.state_block_idx == 0).all(), 'idle placeholder targets a real state')
        require(isinstance(frame['scheduler_observations_sha256'], str)
                and len(frame['scheduler_observations_sha256']) == 64,
                'unbound world scheduler evidence')
        frame['model_consumed'] = True
        result = self.model.allocate_output(1)
        self.events.append(dict(ordinal=frame['ordinal'], world_phase='idle',
            schedule_sha256=schedule['sha256'],
            scheduler_observations_sha256=frame['scheduler_observations_sha256'],
            phase=self.phase, request_id=self.request_id,
            idle_forward_skipped=True, native_decoder_calls=0,
            returned_placeholder_rows=1, real_query_rows=0,
            worker_forward_postprocess_sample_round_preserved=True,
            sampler_and_eplb_step_preserved=True, dummy_output_is_real_model_output=False,
            real_request_cache_accessed_by_bypass=False,
            native_decode_metadata=observed_metadata,
            total_q_is_prefill_only_default=True))
        return result

    def report(self):
        require(self.installed, 'candidate was never installed')
        return seal(dict(schema='s12-world-idle-forward-candidate-v1', rank=self.rank,
            enabled=self.enabled, phase=self.phase, request_id=self.request_id,
            capabilities=self.capabilities, events=list(self.events),
            bypassed_world_idle_steps=len(self.events),
            active_real_state_nonzero_checks=self.active_state_nonzero_checks,
            placeholder_state_zero_asserted=True,
            source_derived_avoided_decoder_entries=self.capabilities['model_num_layers'] * len(self.events),
            source_derived_entries_are_not_measured_time=True,
            existing_world_model_execution_completed_means_callback_return_not_decoder_execution=True,
            output_values_for_discarded_placeholder_changed=bool(self.events),
            placeholder_expert_load_statistics_may_change=bool(self.events),
            routing_rearrangement_disabled_by_actual_capabilities=True,
            real_prefill_decode_forward_unchanged=True,
            gpu_numerically_qualified=False, qualification_is_external_fresh_gate=True))

    def restore(self):
        if self.closed:
            return
        require(self.installed and self.model.forward is self.hook, 'candidate forward hook ownership changed')
        if self.own:
            self.model.forward = self.original
        else:
            delattr(self.model, 'forward')
        self.closed = True
