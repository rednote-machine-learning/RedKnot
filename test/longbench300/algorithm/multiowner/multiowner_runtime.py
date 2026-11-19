"""One resident EP8/DP8 engine with actual, independent DP request owners.

This module is an execution candidate, not evidence of GPU qualification. It
imports no Torch/DeepInfer until create_runtime()/initialize() is called. Each
active DP rank owns one real request; global B4 therefore means four distinct
native requests scheduled on four ranks, not local num_reqs=4 or four queued
serial requests. Frozen sources are imported, never edited.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib
import json
import os
import socket
import sys
import time
from types import MethodType


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, sha256=digest(value))


def file_ref(path, value=None):
    path = Path(path).resolve()
    result = dict(path=str(path), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if value is not None:
        result['seal_sha256'] = value['sha256']
    return result


def canonical_method(method, selector_strategy='mean'):
    require(selector_strategy in ('mean', 'max', 'softmax_mean'), 'Unknown selector strategy')
    if method == 'Full':
        return 'Full', False, selector_strategy
    if method in ('S1.3', 'S13'):
        return 'S1.3', True, selector_strategy
    if method.startswith('Index20LastKV.'):
        strategy = method.split('.', 1)[1]
        require(strategy in ('mean', 'max', 'softmax_mean'), 'Unknown LastKV strategy')
        return 'S1.3', True, strategy
    raise ValueError('Initial multiowner candidate supports Full and S1.3 only; S1.2 is not enabled')


def checked_rows(rows, request_batch_size, installed_ranks):
    """Keep real input IDs and frame geometry; never pad or alter a question."""
    require(type(request_batch_size) is int and request_batch_size in (1, 4, 8),
        'Supported global DP request batches are 1, 4, 8')
    require(isinstance(rows, (list, tuple)) and len(rows) == request_batch_size,
        'Exactly one real input row is required per active DP owner')
    active = tuple(installed_ranks[:request_batch_size])
    require(len(active) == request_batch_size and len(set(active)) == request_batch_size,
        'Requested global batch exceeds installed owner capacity')
    identities = set()
    frame_counts = set()
    for row in rows:
        ids = row['input_ids']
        require(isinstance(ids, (list, tuple)) and ids and
            all(type(v) is int and 0 <= v < 2**31 for v in ids), 'Invalid original token IDs')
        require(row['input_tokens'] == len(ids) and row['input_ids_sha256'] == digest(list(ids)),
            'Original complete input identity changed')
        require(type(row['index']) is int and row['index'] not in identities,
            'Every batch member must be a distinct original case')
        identities.add(row['index'])
        frame_counts.add((len(ids) + 8191) // 8192)
    require(len(frame_counts) == 1,
        'Initial multiowner candidate requires equal native chunk counts; no padding/truncation')
    return active


def world_execution_evidence(workers, spec, batch_id):
    """Preserve raw reports and derive active rows from actual scheduler events."""
    from world_plan_coordinator import validate_worker_reports
    collective = validate_worker_reports(workers, spec)
    require(collective['all_eight_workers_agreed'] is True and
        collective['world_plan_sha256'] == spec['sha256'], 'Actual eight-rank worker reports failed collective validation')
    owners = {o['rank']:o for o in spec['owners']}
    steps = []
    for event in workers[0]['world_step_trace']:
        if event['world_phase'] == 'idle':
            continue
        rank_rows = []
        for rank in range(8):
            interval = event['owner_ranges'].get(str(rank))
            requests = []
            if interval is not None:
                require(rank in owners and interval[1] > interval[0], 'Foreign or empty executed owner interval')
                requests.append(dict(request_id=owners[rank]['request_id'], scheduled_token_rows=interval[1]-interval[0]))
            rank_rows.append(dict(rank=rank, requests=requests))
        steps.append(dict(step_id=event['ordinal'], kind=event['world_phase'], rank_rows=rank_rows))
    world = seal(dict(schema='s13-multiowner-world-trace-v1', batch_id=batch_id, steps=steps,
        original_worker_reports=workers, source='actual_native_engine_scheduler_callbacks',
        queued_requests_are_not_active_batch_evidence=True))
    proof = seal(dict(schema='s13-multiowner-collective-proof-v1', status='passed', batch_id=batch_id,
        world_plan_sha256=spec['sha256'], actual_validation=collective,
        all_eight_ranks_same_order=True, empty_ep_participation_verified=True,
        proof_basis='validated_actual_eight_rank_scheduler_and_executed_prefill_layer_reports'))
    return world, proof


class MultiOwnerScheduleObserver:
    """Observe each local native owner, preserving the original engine hook."""
    def __init__(self, engine, rank):
        self.engine, self.rank = engine, rank
        self.previous_override = engine.__dict__.get('_execute_model_async_step')
        self.original = engine._execute_model_async_step
        self.request = None
        self.active = False
        owner = self

        def observe(module, seqs, outputs):
            frames = [dict(request_id=s.request_id, q_start=s.seqlen_k-s.seqlen_q,
                q_end=s.seqlen_k, rows=s.seqlen_q) for s in seqs if not s.is_placeholder]
            if owner.active:
                if owner.request is None:
                    require(not frames, 'Inactive owner rank received a real native request')
                else:
                    require(all(f['request_id'] == owner.request.request_id for f in frames),
                        'Local native scheduled RID differs from this owner')
                    require(len(frames) <= 1, 'One local sequence per DP owner is required')
                if outputs.prompt_run:
                    owner.ranges.extend(frames)
                else:
                    owner.decode_steps += bool(frames)
            return owner.original(seqs, outputs)

        self.hook = MethodType(observe, engine)
        engine._execute_model_async_step = self.hook

    def configure(self, request):
        require(not self.active, 'Previous local schedule observation is unfinished')
        self.request, self.ranges, self.decode_steps, self.active = request, [], 0, True

    def finish(self):
        require(self.active, 'No active local schedule observation')
        position = 0
        for frame in self.ranges:
            require(frame['q_start'] == position and frame['q_end']-frame['q_start'] == frame['rows'],
                'Actual local prefill is not a contiguous complete original prompt')
            position = frame['q_end']
        if self.request is None:
            require(not self.ranges and self.decode_steps == 0, 'Inactive rank performed real native work')
        else:
            require(position == len(self.request.prompt_token_ids), 'Native owner omitted actual input rows')
        result = seal(dict(schema='s13-multiowner-native-scheduled-rows-v1', rank=self.rank,
            request_id=None if self.request is None else self.request.request_id,
            input_tokens=0 if self.request is None else len(self.request.prompt_token_ids),
            initial_cached_tokens=0, scheduled_prefill_rows=sum(f['rows'] for f in self.ranges),
            first_prefill_q_start=None if not self.ranges else self.ranges[0]['q_start'],
            prefill_ranges=list(self.ranges), decode_steps=self.decode_steps))
        self.request, self.active = None, False
        return result

    def close(self):
        require(not self.active, 'Cannot remove an active local schedule observer')
        require(self.engine.__dict__.get('_execute_model_async_step') is self.hook,
            'Multiowner schedule observer hook ownership changed')
        if self.previous_override is None:
            del self.engine._execute_model_async_step
        else:
            self.engine._execute_model_async_step = self.previous_override


def validate_owner_decode(phase, owner_rank, *, require_real_decode=True):
    """Use original all-rank storage/graph checks and explicitly index the owner."""
    from redknot.decode_evidence import validate_decode_phase
    validate_decode_phase(phase, require_real_decode=False)
    scheduled = phase['scheduled_rows_reports'][owner_rank]
    graph = phase['decode_graph_evidence'][owner_rank]
    require(scheduled['rank'] == graph['rank'] == owner_rank and
        scheduled['request_id'] == phase['record']['request_id'], 'Decode evidence belongs to another owner')
    if require_real_decode:
        require(scheduled['decode_steps'] > 0 and len(phase['record']['output_ids']) > 1 and
            all(graph['successful_replay_deltas'][name] > 0 for name in ('main', 'dflash')),
            'This actual owner must execute real decode and both captured graph runners')
    return dict(owner_rank=owner_rank, actual_native_decode_steps=scheduled['decode_steps'],
        successful_main_replays=graph['successful_replay_deltas']['main'],
        successful_dflash_replays=graph['successful_replay_deltas']['dflash'],
        native_idle_bypass_installed=False)


def runtime_class(parent):
    """Build on real validated B1 startup, then install independent owner hooks."""
    from index20_lastkv_bench_runtime import bench_runtime_class
    base = bench_runtime_class(parent)

    class MultiOwnerRuntime(base):
        def __init__(self, config, arm, output_dir, rank):
            require(arm == 's2', 'Full and S1.3 share one multiowner-capable model instance')
            super().__init__(config, arm, Path(output_dir), rank)
            self.installed_owner_ranks = tuple(config.get('multiowner_active_ranks', (0, 1, 2, 3)))
            require(self.installed_owner_ranks == tuple(range(len(self.installed_owner_ranks))) and
                len(self.installed_owner_ranks) in (1, 4, 8), 'Explicit contiguous DP owner ranks required')
            self.multiowner_enabled = False
            self.multiowner_sequence = 0
            self.local_case = None
            self.local_case_plan = None
            self.input_preparation = None
            self.identity_qualification = {}
            self.case_qualifications = {}

        def initialize(self):
            # Original qualification executes unchanged, including D-Spark and
            # both CUDA graphs. It is not reused as another owner's identity.
            ready = super().initialize()
            self.actual_runtime_reports = []
            for reference in self.startup['runtime']:
                observed = json.loads(Path(reference['path']).read_text())
                require(observed['sha256'] == reference['sha256'] ==
                    digest({k:v for k,v in observed.items() if k != 'sha256'}), 'Actual native runtime provenance changed')
                require(observed['observed']['nextn'] == 5 and observed['observed']['use_cuda_graph'] is True and
                    observed['observed']['dflash_use_cuda_graph'] is True, 'Actual native D-Spark/graphs are not enabled')
                self.actual_runtime_reports.append(observed)
            self._install_multiowner()
            self.clock_evidence_ref = self._clock_evidence()
            self._qualify_owner_identity()
            return dict(ready, multiowner_ready=True, active_owner_capacity=len(self.installed_owner_ranks),
                clock_evidence_ref=self.clock_evidence_ref,
                actual_global_B4_B8_quality_qualified=False)

        def _install_multiowner(self):
            from multiowner_controller import multiowner_controller_class
            from multiowner_worker import install_multiowner_worker_schedule
            from index20_lastkv_runtime import index20_lastkv_controller_class
            from rag_natural_documents import natural_controller_class
            from rag_capture_policy import capture_policy_cache_class
            from service.canonical_request_binding_dspark import install_request_binding
            require(self.prefix_cache is None and self.binding is None and self.current_request is None,
                'Multiowner installation requires a drained non-prefix engine')
            cap = self.batch_worker.capability
            for name, method in (('observer', 'close'), ('batch_worker', 'close'),
                                 ('controller', 'close'), ('binder', 'restore')):
                obj = getattr(self, name)
                if obj is not None:
                    getattr(obj, method)()
                    setattr(self, name, None)
            # A binding is local to its rank. It is never broadcast from rank0.
            self.binder = install_request_binding(self.model_runner, lambda: self.binding)
            if self.rank in self.installed_owner_ranks:
                cls = multiowner_controller_class(index20_lastkv_controller_class(natural_controller_class()))
                self.controller = cls(self.layers, allow_sync=self.allow_sync, native_model=self.model,
                    expected_document_count=0, model_sha256=self.scope['model_sha256'],
                    tokenizer_sha256=self.scope['tokenizer_sha256'], execution_sha256=self.scope['execution_sha256'],
                    dependency_root=Path(self.config['experiment_root']), empty_ep_capability=cap, enable_dflash_tail=True)
                self.controller.install()
                c = self.controller
                cache_cls = capture_policy_cache_class(type(c.cache), persist_offline_tensors=False)
                c.cache = cache_cls(layers=c.layers, scope=c.scope, torch_module=c.t, allow_sync=c.allow_sync)
                c.attention = type(c.attention)(c.cache, torch_module=c.t, allow_sync=c.allow_sync)
                c.configure_lastkv(False)
                c.configure_index20(enabled=False, fraction=.2)
            self.batch_worker = install_multiowner_worker_schedule(self.model, rank=self.rank,
                torch_module=self.torch, empty_ep_capability=cap, controller=self.controller,
                engine=self.driver.engine, phase_gather=self.driver.gather, allow_native_decode_graphs=True,
                active_ranks=self.installed_owner_ranks)
            self.observer = MultiOwnerScheduleObserver(self.driver.engine, self.rank)
            self.multiowner_enabled = True
            self.driver.barrier()
            self.multiowner_enabled = True

        def _clock_evidence(self):
            start = self.driver.gather(time.perf_counter_ns() if self.rank == 0 else None)[0]
            clock = time.get_clock_info('perf_counter')
            boot_path = Path('/proc/sys/kernel/random/boot_id')
            sample = dict(rank=self.rank, hostname=socket.gethostname(),
                boot_id=boot_path.read_text().strip() if boot_path.is_file() else None,
                implementation=clock.implementation, monotonic=clock.monotonic,
                resolution=clock.resolution, sample_ns=time.perf_counter_ns())
            samples = self.driver.gather(sample)
            end = self.driver.gather(time.perf_counter_ns() if self.rank == 0 else None)[0]
            require(len(samples) == 8 and [s['rank'] for s in samples] == list(range(8)), 'Eight actual clock samples required')
            require(len({(s['hostname'], s['boot_id'], s['implementation']) for s in samples}) == 1 and
                samples[0]['boot_id'] is not None and all(s['monotonic'] and start <= s['sample_ns'] <= end for s in samples),
                'Owner monotonic timestamps do not prove a common host/kernel clock')
            body = seal(dict(schema='s13-multiowner-clock-evidence-v1', passed=True,
                clock_domain='same_host_perf_counter_ns', rank0_send_ns=start, rank0_receive_ns=end,
                samples=samples, timestamps_are_original_owner_samples=True, timestamps_rewritten=False))
            ref = self.publish('multiowner-clock-evidence', body)
            return file_ref(ref['path'], body)

        def _configure_many(self, requests, phase, mode, method, strategy, verify_sparse=False):
            from world_plan_coordinator import make_world_plan
            from redknot.decode_evidence import cache_storage_snapshot
            local = requests.get(self.rank)
            require(self.binding is None and self.prefix_cache is None, 'Only a drained fresh-request phase is admitted')
            self.phase, self.phase_mode = phase, mode
            self.native_admission = None
            self.graph_phase_before = self.graph_meter.snapshot()
            self.cache_storage_before = cache_storage_snapshot(self.driver.engine)
            require(self.cache_storage_before == self.cache_storage_baseline, 'Native captured-graph cache storage changed')
            self.observer.configure(local)
            schedule = None
            if local is not None:
                c = self.controller
                require(c is not None, 'An actual owner must have its independent installed controller')
                index20 = method == 'S1.3'
                c.configure_lastkv(False)
                c.configure_index20(enabled=index20, fraction=.2, strategy=strategy,
                    diagnostic_fingerprints=bool(getattr(self, '_diagnostic', False)))
                if index20:
                    c.configure_lastkv(True)
                c.configure(system_enabled=mode in ('reference_sparse', 'sensitive'),
                    batch_group_chunks=17 if mode == 'sensitive' else 1)
                arm = ('s1' if mode in ('reference_sparse', 'sensitive') else
                    'qualification' if mode in ('qualified_native', 'all_selected') else
                    'offline' if mode == 'capture_independent' else 'recomputed')
                with self.allow_sync():
                    c.reset(local.request_id, mode or 'native', arm=arm, input_ids=local.prompt_token_ids,
                        documents=None if mode == 'capture_independent' else list(local.documents), phase=phase,
                        verify_sparse=verify_sparse,
                        index_device=self.layers[0].self_attn.q_a_kv_swa_proj.weight.device
                            if mode in ('reference_sparse', 'sensitive', 'all_selected') else None)
                schedule = c.batch_schedule
            owners = self.driver.gather(None if local is None else dict(rank=self.rank, request_id=local.request_id,
                input_ids_sha256=digest(list(local.prompt_token_ids)), input_tokens=len(local.prompt_token_ids), schedule=schedule))
            self.schedule = make_world_plan(batch_id=phase, phase=phase, arm=method,
                owners=[x for x in owners if x is not None])
            self.batch_receipts = self.driver.gather(self.batch_worker.configure(self.schedule))
            if local is not None:
                self.controller.bind_multiowner(self.schedule, self.batch_receipts)
            return dict(multiowner_phase_id=phase, multiowner_method=method,
                world_plan_sha256=self.schedule['sha256'], owner_rank=self.rank if local is not None else None,
                original_input_ids_unchanged=True)

        def _finish_many(self, requests):
            from redknot.decode_evidence import cache_storage_snapshot
            from world_plan_coordinator import validate_worker_reports
            local = requests.get(self.rank)
            workers = self.driver.gather(self.batch_worker.finish_phase())
            world_proof = validate_worker_reports(workers, self.schedule)
            report = None
            if local is not None:
                require(self.controller is not None and self.binding is not None and
                    self.binding['request_id'] == local.request_id, 'Local owner/controller/binding identity changed')
                with self.allow_sync():
                    report = self.controller.finish()
            else:
                require(self.binding is None, 'Inactive owner retained a request binding')
            reports = self.driver.gather(report)
            require(all((r in requests) == (reports[r] is not None) for r in range(8)), 'Controller reports do not match actual owner map')
            self.binding = None
            scheduled = self.driver.gather(self.observer.finish())
            admissions = self.driver.gather(self.native_admission)
            require(all((r in requests) == (admissions[r] is not None) for r in range(8)), 'Actual native submissions do not match owner map')
            after = self.graph_meter.snapshot()
            graph = dict(rank=self.rank, profile=self.profile.name, before=self.graph_phase_before, after=after,
                successful_replay_deltas={key: after[key]['successful_replay_calls']-
                    self.graph_phase_before[key]['successful_replay_calls'] for key in ('main', 'dflash')},
                includes_native_idle=True, speculative_acceptance_rate=None)
            graphs = self.driver.gather(graph)
            storage_after = cache_storage_snapshot(self.driver.engine)
            require(storage_after == self.cache_storage_before == self.cache_storage_baseline,
                'A multiowner request replaced native graph-bound cache storage')
            storage = self.driver.gather(dict(rank=self.rank, baseline_sha256=self.cache_storage_baseline['sha256'],
                before_sha256=self.cache_storage_before['sha256'], after_sha256=storage_after['sha256'],
                native_cache_storage_unchanged=True))
            return dict(controllers_by_rank=reports, batch_worker_reports=workers, batch_schedule=self.schedule,
                batch_configuration_receipts=self.batch_receipts, scheduled_rows_reports=scheduled,
                native_admissions_by_rank=admissions, decode_graph_evidence=graphs,
                native_cache_storage_evidence=storage, native_idle_bypass_installed=False,
                world_scheduler=world_proof)

        def _execute_many(self, requests, phase, mode, method, *, strategy='mean', diagnostic=False, benchmark=False):
            from multiowner_transport import run_multiowner_requests
            require(self.multiowner_enabled and requests and set(requests) <= set(self.installed_owner_ranks),
                'Requests exceed installed independent owner capacity')
            self._diagnostic = bool(diagnostic)
            local = run_multiowner_requests(self.port, requests, rank=self.rank, active_ranks=tuple(sorted(requests)),
                barrier=self.driver.barrier, gather=self.driver.gather,
                prepare=lambda request: self._configure_many(requests, phase, mode, method, strategy),
                finish=lambda request: self._finish_many(requests), emit=lambda event: None)
            records = self.driver.gather(local)
            require(all((r in requests) == (records[r] is not None) for r in range(8)), 'Returned native records differ from owner map')
            phases = {}
            for rank in sorted(requests):
                record = dict(records[rank])
                completion = record.pop('completion')
                preparation = record.pop('preparation')
                record['text'] = self.driver.engine.tokenizer.decode(record['output_ids'], skip_special_tokens=False)
                record['prediction_text'] = self.driver.engine.tokenizer.decode(record['output_ids'], skip_special_tokens=True)
                controller = completion['controllers_by_rank'][rank]
                body = seal(dict(schema='s13-multiowner-native-phase-v1', name=phase+'.owner'+str(rank),
                    mode=mode, method=method, owner_rank=rank, record=record, preparation=preparation,
                    controller=controller, benchmark_request=bool(benchmark), qualification_only=not benchmark,
                    **completion))
                ref = self.publish(body['name'], body)
                phases[rank] = dict(phase=body, phase_ref=file_ref(ref['path'], body))
            return phases

        def _qualify_owner_identity(self):
            from service.engine_bridge import TokenRequest
            ids = tuple(self.config['qualification_input_ids'])
            require(len(ids) == 16384, 'Each new owner requires actual fresh 16K identity evidence')
            results = []
            for mode in ('qualified_native', 'all_selected'):
                phase = 'multiowner-startup-'+mode
                received = self.driver.gather(time.perf_counter() if self.rank == 0 else None)[0]
                requests = {r: TokenRequest(phase+'.owner'+str(r), ids, 32, 's2', received)
                    for r in self.installed_owner_ranks}
                results.append(self._execute_many(requests, phase, mode, 'Full'))
            for rank in self.installed_owner_ranks:
                a, b = results[0][rank]['phase'], results[1][rank]['phase']
                ac, bc = a['controller'], b['controller']
                require(ac['identity_check_count'] == bc['identity_check_count'] == 80 and bc['identity_exact'] is True and
                    ac['identity_fingerprints'] == bc['identity_fingerprints'] and
                    a['record']['output_ids'] == b['record']['output_ids'], 'This owner failed actual native/all-selected identity')
                decode = validate_owner_decode(b, rank, require_real_decode=len(b['record']['output_ids']) > 1)
                proof = seal(dict(schema='s13-multiowner-native-identity-v1', passed=True, owner_rank=rank,
                    native_ref=results[0][rank]['phase_ref'], all_selected_ref=results[1][rank]['phase_ref'],
                    identity_check_count=80, complete32_output_ids_equal=True, decode=decode,
                    identity_qualified_from_own_actual_execution=True, qualification_flags_copied=False))
                ref = self.publish('multiowner-identity.owner'+str(rank), proof)
                self.identity_qualification[rank] = file_ref(ref['path'], proof)

        def prepare_inputs(self, rows):
            """Separate exact-document bank creation; excluded from online TTFT."""
            from rag_natural_documents import plan_case
            from service.engine_bridge import TokenRequest
            active = checked_rows(rows, len(rows), self.installed_owner_ranks)
            self.multiowner_sequence += 1
            preparation_generation = self.multiowner_sequence
            began = self.driver.gather(time.perf_counter_ns() if self.rank == 0 else None)[0]
            plans = [plan_case(r['input_ids'], r['documents']) for r in rows]
            require(all(p['effective_route'] == 's12' for p in plans),
                'Initial S1.3 multiowner pilot requires cacheable original inputs; no silent Full fallback')
            local_row = rows[active.index(self.rank)] if self.rank in active else None
            if self.controller is not None:
                with self.allow_sync():
                    self.controller.clear_cache(reset_sparse_qualification=True)
                self.controller.set_cache_output_dir(self.out/('multiowner-offline.prep'+
                    str(preparation_generation)+'.owner'+str(self.rank)))
            self.driver.barrier()
            builds = []
            for owner, row, plan in zip(active, rows, plans):
                for ordinal, doc in enumerate(plan['unique_capture_documents']):
                    self.multiowner_sequence += 1
                    phase = 'multiowner-offline-'+str(self.multiowner_sequence)+'.owner'+str(owner)+'.doc'+str(ordinal)
                    received = self.driver.gather(time.perf_counter() if self.rank == 0 else None)[0]
                    ids = tuple(row['input_ids'][doc['input_start']:doc['input_end']])
                    request = TokenRequest(phase, ids, 1, 's2', received)
                    builds.append(self._execute_many({owner: request}, phase, 'capture_independent', 'Full')[owner]['phase_ref'])
            if local_row is not None:
                self.local_case = local_row
                self.local_case_plan = plans[active.index(self.rank)]
                with self.allow_sync():
                    self.controller.prepare_case(input_ids=local_row['input_ids'], documents=self.local_case_plan['effective_documents'],
                        index_device=self.layers[0].self_attn.q_a_kv_swa_proj.weight.device)
            self.driver.barrier()
            identities = [dict(rank=r, index=row['index'], input_ids_sha256=row['input_ids_sha256']) for r, row in zip(active, rows)]
            ended = self.driver.gather(time.perf_counter_ns() if self.rank == 0 else None)[0]
            body = seal(dict(schema='s13-multiowner-input-preparation-v1', status='ready',
                indices=[r['index'] for r in rows], input_hashes=[r['input_ids_sha256'] for r in rows],
                active_ranks=list(active), inputs=identities,
                input_plans=plans, offline_build_refs=builds, offline_build_elapsed_s=(ended-began)/1e9,
                offline_build_in_online_ttft=False, persist_offline_tensors=False,
                independent_owner_banks=True, exact_original_input_ids=True, padding_tokens=0, truncation_tokens=0))
            ref = self.publish('multiowner-input-preparation-'+str(self.multiowner_sequence), body)
            self.input_preparation = body
            self.input_preparation_ref = file_ref(ref['path'], body)
            self.case_qualifications = {}
            return dict(body, preparation_ref=self.input_preparation_ref)

        def run_batch(self, method, input_rows, *, phase, request_batch_size, timing_repeat=None,
                      selector_strategy='mean', output_dir=None):
            from service.engine_bridge import TokenRequest
            arm, index20, strategy = canonical_method(method, selector_strategy)
            checked_rows(input_rows, request_batch_size, self.installed_owner_ranks)
            require(self.input_preparation is not None, 'Call prepare_inputs before running a batch')
            assignments = {(a['index'], a['input_ids_sha256']): a['rank'] for a in self.input_preparation['inputs']}
            keys = [(row['index'], row['input_ids_sha256']) for row in input_rows]
            require(all(key in assignments for key in keys), 'Original input is not resident in its prepared owner bank')
            active = tuple(assignments[key] for key in keys)
            require(len(set(active)) == request_batch_size and list(active) == sorted(active),
                'Sub-batches must retain original prepared owner assignments and order')
            plans_by_rank = dict(zip(self.input_preparation['active_ranks'], self.input_preparation['input_plans']))
            world_key = tuple((owner, row['input_ids_sha256']) for owner, row in zip(active, input_rows))
            require(phase in ('qualification', 'timing', 'quality'), 'Unknown benchmark phase')
            missing = [(arm, strategy, owner, row['input_ids_sha256'], world_key) not in self.case_qualifications
                for owner, row in zip(active, input_rows)]
            if phase != 'qualification' and any(missing):
                # Actual fresh bounded execution uses this exact owner set.
                # Its work occurs before the measured request admission clock.
                self.run_batch(method, input_rows, phase='qualification', request_batch_size=request_batch_size,
                    selector_strategy=selector_strategy)
            self.multiowner_sequence += 1
            name = 'batch'+str(self.multiowner_sequence)+'.'+arm+'.'+strategy+'.'+phase
            if timing_repeat is not None:
                name += '.repeat'+str(timing_repeat)
            started_ns = self.driver.gather(time.perf_counter_ns() if self.rank == 0 else None)[0]
            requests = {}
            for owner, row in zip(active, input_rows):
                plan = plans_by_rank[owner]
                tokens = 270336-len(row['input_ids'])-16 if phase == 'quality' else 32
                require(tokens > 0, 'Original model context leaves no output capacity')
                requests[owner] = TokenRequest(name+'.owner'+str(owner), tuple(row['input_ids']), tokens,
                    's2' if index20 else 'full', started_ns/1e9, documents=tuple(plan['effective_documents']))
            phases = self._execute_many(requests, name, 'sensitive' if index20 else None, arm,
                strategy=strategy, diagnostic=phase == 'qualification', benchmark=phase != 'qualification')
            finished_ns = self.driver.gather(time.perf_counter_ns() if self.rank == 0 else None)[0]
            answers = []
            qualification_refs = []
            for owner, row in zip(active, input_rows):
                item = phases[owner]
                raw, record = item['phase'], item['phase']['record']
                decode = validate_owner_decode(raw, owner, require_real_decode=len(record['output_ids']) > 1)
                proof = None
                if index20:
                    from index20_lastkv_contract import validate_lastkv_phase
                    proof = validate_lastkv_phase(raw, row)
                key = (arm, strategy, owner, row['input_ids_sha256'], world_key)
                if phase == 'qualification':
                    qualification = seal(dict(schema='s13-multiowner-case-qualification-v1', status='passed', passed=True,
                        request_id=record['request_id'], owner_rank=owner, method=arm, strategy=strategy, input_ids_sha256=row['input_ids_sha256'],
                        phase_ref=item['phase_ref'], native_identity_ref=self.identity_qualification[owner],
                        decode=decode, lastkv_execution=proof, output_ids=list(record['output_ids']),
                        claim='actual_own_arm_execution_not_Full_vs_S13_numerical_equivalence'))
                    ref = self.publish(name+'.qualification.owner'+str(owner), qualification)
                    self.case_qualifications[key] = (qualification, file_ref(ref['path'], qualification))
                else:
                    require(key in self.case_qualifications, 'Fresh own-method qualification required before measured requests')
                    qualification, _ = self.case_qualifications[key]
                    expected = qualification['output_ids']
                    require(record['output_ids'][:len(expected)] == expected,
                        'Measured request differs from its own actual bounded qualification')
                qualification, warm_ref = self.case_qualifications[key]
                if phase == 'qualification':
                    qualification_ref = warm_ref
                else:
                    execution = seal(dict(schema='s13-multiowner-measured-request-execution-v1', status='passed',
                        request_id=record['request_id'], owner_rank=owner, method=arm, strategy=strategy,
                        input_ids_sha256=row['input_ids_sha256'], actual_phase_ref=item['phase_ref'],
                        actual_native_identity_ref=self.identity_qualification[owner], actual_same_owner_set_warm_ref=warm_ref,
                        own_method_bounded_output_prefix_exact=True, decode=decode, lastkv_execution=proof,
                        Full_vs_S13_equivalence_claim=False))
                    execution_ref = self.publish(name+'.execution.owner'+str(owner), execution)
                    qualification_ref = file_ref(execution_ref['path'], execution)
                qualification_refs.append(qualification_ref)
                answers.append(dict(index=row['index'], request_id=record['request_id'], owner_rank=owner,
                    input_ids_sha256=row['input_ids_sha256'], input_tokens=row['input_tokens'],
                    received_ns=round(record['received_s']*1e9), admitted_ns=round(record['admitted_s']*1e9),
                    prepared_ns=None if record['prepared_s'] is None else round(record['prepared_s']*1e9),
                    submitted_ns=round(record['submitted_s']*1e9), first_token_ns=round(record['first_raw_token_s']*1e9),
                    completed_ns=round(record['terminal_s']*1e9), drained_ns=round(record['drained_s']*1e9),
                    output_ids=record['output_ids'], text=record['text'], prediction_text=record['prediction_text'],
                    finish_reason=record['finish_reason'], phase_ref=item['phase_ref'], qualification_ref=qualification_ref))
            raw_phase = phases[active[0]]['phase']
            workers = raw_phase['batch_worker_reports']
            world, collective = world_execution_evidence(workers, raw_phase['batch_schedule'], name)
            world_ref = self.publish(name+'.world-trace', world)
            collective_ref = self.publish(name+'.collective-proof', collective)
            graphs = raw_phase['decode_graph_evidence']
            storage = raw_phase['native_cache_storage_evidence']
            graph_counters = [dict(rank=r, main_captured_sizes=graphs[r]['after']['main']['captured_graph_sizes'],
                draft_captured_sizes=graphs[r]['after']['dflash']['captured_graph_sizes'],
                main_replay_delta=graphs[r]['successful_replay_deltas']['main'],
                draft_replay_delta=graphs[r]['successful_replay_deltas']['dflash'],
                native_cache_storage_unchanged=storage[r]['native_cache_storage_unchanged']) for r in range(8)]
            evidence = seal(dict(schema='s13-multiowner-native-evidence-v1', status='passed',
                batch_id=name, request_ids=[a['request_id'] for a in answers],
                request_batch_size=request_batch_size, active_ranks=list(active),
                all_request_qualifications_passed=True, qualification_refs=qualification_refs,
                native_idle_bypass_installed=False, clock_evidence_ref=self.clock_evidence_ref,
                decode_graph_evidence=graphs, native_cache_storage_evidence=storage,
                graph_replays_by_rank=graph_counters, actual_runtime_reports=self.actual_runtime_reports,
                native_dspark_nextn=self.actual_runtime_reports[0]['observed']['nextn'],
                main_graphs_enabled=self.actual_runtime_reports[0]['observed']['use_cuda_graph'],
                draft_graphs_enabled=self.actual_runtime_reports[0]['observed']['dflash_use_cuda_graph'],
                multiple_owner_speedup_claim=False, B1_vs_B4_output_parity_checked_by_benchmark=False))
            evidence_ref = self.publish(name+'.native-evidence', evidence)
            body = seal(dict(schema='s13-native-multiowner-batch-v1', status='complete', batch_id=name, method=arm,
                selector_strategy=strategy, phase=phase, request_batch_size=request_batch_size,
                input_indices=[r['index'] for r in input_rows], clock_domain='same_host_perf_counter_ns',
                clock_evidence_ref=self.clock_evidence_ref, batch_started_ns=started_ns, batch_finished_ns=finished_ns,
                batch_finish_includes_controller_reporting=True, requests=answers,
                input_preparation_ref=self.input_preparation_ref,
                native_evidence_ref=file_ref(evidence_ref['path'], evidence),
                world_trace_ref=file_ref(world_ref['path'], world), collective_trace_ref=file_ref(collective_ref['path'], collective),
                global_real_request_count=request_batch_size, local_requests_per_active_DP_rank=1,
                ep_size=8, dp_size=8, concurrent_model_copies=1, quality_artificial_limit=None if phase == 'quality' else 32,
                original_input_ids_unchanged=True, actual_gpu_executed=True))
            self.publish(name+'.batch', body)
            return body

    return MultiOwnerRuntime


@contextmanager
def create_runtime(cfg, *, rank, output_dir=None):
    """Called on all eight original launch() ranks; never starts processes itself.

    cfg supplies service_config, native_source_root, frozen_runtime_root and
    rag_runtime_root. Parent benchmark owns source hashes, original model/token
    admission, process lifecycle and the shared exclusive GPU lock.
    """
    cfg = dict(cfg)
    if 'frozen_runtime_root' not in cfg:
        cfg['frozen_runtime_root'] = cfg['frozen_lastkv_runtime_root']
    for key in ('native_source_root', 'frozen_runtime_root', 'rag_runtime_root'):
        require(Path(cfg[key]).is_dir(), 'Missing preserved runtime dependency: '+key)
    for path in (Path(cfg['native_source_root']), Path(cfg['native_source_root'])/'GatedEngram',
                 Path(cfg['rag_runtime_root']), Path(cfg['frozen_runtime_root'])):
        sys.path.insert(0, str(path.resolve()))
    parent = importlib.import_module('service.native_runtime')
    require(Path(parent.__file__).resolve() == Path(cfg['native_source_root']).resolve()/'service/native_runtime.py',
        'Imported native runtime differs from the preserved source root')
    if 'service_config' in cfg:
        service = dict(cfg['service_config'])
    else:
        from rag_dspark_runtime import read_ref
        service = dict(read_ref(cfg['service_config_ref']))
    service.pop('sha256', None)
    root = Path(cfg['native_source_root']).resolve()
    service.update(native_repo=str(root/'NativeSnapshot'), gated_repo=str(root/'GatedEngram'), stage=str(root),
        context_tokens=270336, gpu_memory_utilization=.60, rag_suite_global_base_cache=False,
        rag_suite_persist_offline_tensors=False, rag_suite_derived_layout_cache_policy='cold_per_online_request',
        index20_fraction=.2, index20_strategy=cfg.get('selector_strategy', 'mean'),
        index20_strategies=[cfg.get('selector_strategy', 'mean')])
    service['scope'] = dict(service['scope'], execution_sha256=digest(dict(base_scope=service['scope'],
        multiowner_config_sha256=cfg['sha256'])))
    service['multiowner_active_ranks'] = list(cfg.get('multiowner_active_ranks',
        tuple(range(max(cfg.get('target_batch_sizes', [4]))))))
    service = seal(service)
    out = Path(output_dir or cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    runtime = runtime_class(parent)(service, 's2', out, rank)
    try:
        runtime.initialize()
        yield runtime
    finally:
        runtime.close()
