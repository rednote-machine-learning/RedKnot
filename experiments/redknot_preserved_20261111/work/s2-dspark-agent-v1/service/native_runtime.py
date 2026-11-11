"""Persistent native EP8/DP8 execution; GPU imports occur only in initialize().

Full owns no RedKnot hook. S2 captures missing natural documents once, and the
strict path performs fresh same-policy qualification for every new prompt.
Those operations are inside the live request's preparation/TTFT clock.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys
import time

from .engine_bridge import NativeEnginePort, TokenRequest, run_serial_request, token_digest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, sha256=digest(value))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def runtime_args(config, output_dir):
    return SimpleNamespace(repo=Path(config['native_repo']), model_path=Path(config['model_path']),
        experiment_root=Path(config['experiment_root']), out_dir=Path(output_dir),
        ep=8, cp=1, dp=8, batch_invariant=1, prefill_attempt_interval=1, sampler_force_topk=1024,
        max_model_len=270336, max_num_batched_tokens=8192, max_num_seqs=1,
        gpu_memory_utilization=.60, mode='quality', max_new_tokens=32, seed=0,
        diagnostic_progress_limit=0)


def load_runtime(config):
    root = Path(config['experiment_root'])
    for folder in ('benchmark', 'strong_native_20261029', 'kv_recompression_20261029',
                   'direct_global_20261029', 'cross_layer_mla_repair_20261030',
                   'kv_compute_20261029', 'moe_stage_20261028'):
        sys.path.insert(0, str(root/folder))
    sys.path.insert(0, str(config['gated_repo']))
    import pr148_native_runner as base
    import strong_native_runner as native
    from native_profile import NativeProfile, scoped_base_profile
    return base, native, NativeProfile, scoped_base_profile


def validate_candidate(reports, schedule, workers, *, phase, request_id):
    """Audit actual batch scheduling with native MTP5 idle, never old bypass."""
    from gpu_pilot.read_s11_results import audit_world_scheduler
    require(len(reports) == len(workers) == 8 and all(v is None for v in reports)
        and schedule is not None, 'D-Spark candidate must retain native idle on all eight ranks')
    require(schedule['request_id'] == request_id, 'candidate schedule belongs to another request')
    for rank, worker in enumerate(workers):
        require(worker.get('rank') == rank and worker.get('enabled') is True
            and worker.get('schedule_sha256') == schedule['sha256'], 'worker schedule identity differs')
        steps = worker.get('steps')
        require(isinstance(steps, list) and len(steps) == len(schedule['steps']), 'batch step coverage missing')
        for actual, expected in zip(steps, schedule['steps']):
            require(all(actual[k] == expected[k] for k in ('start','end','flush','layers')),
                    'actual batch EP order differs')
    world = audit_world_scheduler(workers, schedule)
    return dict(world, schema='s2-dspark-native-idle-scheduler-v1', phase=phase,
        request_id=request_id, schedule_sha256=schedule['sha256'],
        idle_steps_use_original_native_execution=True, actual_idle_steps_bypassed=0,
        mtp0_idle_candidate_installed=False, native_decode_graphs_allowed=True)


def reuse_coverage(input_ids, source_documents, effective_documents, resident_keys, prefix_history):
    """Disjoint actual document occurrences, with exact historical prefix proof."""
    prior = set(resident_keys)
    origins = {(d['input_start'],d['input_end']):d.get('origin',{}).get('source_kind','unknown')
               for d in source_documents}
    hit_tokens = same_prefix = hit_occurrences = 0
    pending, categories = [], {}
    for doc in effective_documents:
        key = doc['document_id']
        prefix_sha = token_digest(input_ids[:doc['input_end']])
        pending.append((key, prefix_sha))
        kind = origins[(doc['input_start'],doc['input_end'])]
        category = categories.setdefault(kind, dict(eligible_document_tokens=0, hit_document_tokens=0,
            prefix_compatible_hit_document_tokens=0, nonprefix_hit_document_tokens=0))
        category['eligible_document_tokens'] += doc['tokens']
        if key in prior:
            hit_tokens += doc['tokens']; hit_occurrences += 1
            category['hit_document_tokens'] += doc['tokens']
            if prefix_sha in prefix_history.get(key, set()):
                same_prefix += doc['tokens']
                category['prefix_compatible_hit_document_tokens'] += doc['tokens']
            else:
                category['nonprefix_hit_document_tokens'] += doc['tokens']
    value = dict(eligible_hit_document_tokens=hit_tokens, hit_occurrences=hit_occurrences,
        prefix_compatible_hit_document_tokens=same_prefix, nonprefix_hit_document_tokens=hit_tokens-same_prefix,
        input_tokens=len(input_ids), nonprefix_hit_document_fraction=(hit_tokens-same_prefix)/len(input_ids),
        by_source_kind=categories,
        match_basis='exact_cached_token_ids_plus_complete_input_prefix_sha_through_document_end',
        scope='prior_completed_native_requests_in_this_persistent_service',
        all_document_kv_bypassed_claim=False, first_three_layers_recomputed=True)
    return value, pending


@dataclass(frozen=True)
class NativeInitializer:
    config: dict
    arm: str
    output_dir: str

    def record_failure(self, rank, error):
        write_json(Path(self.output_dir)/f'failure.rank{rank}.json',
            seal(dict(status='failed', rank=rank, error=repr(error))))

    def __call__(self, rank):
        runtime = NativeRuntime(self.config, self.arm, Path(self.output_dir), rank)
        try:
            metadata = runtime.initialize()
            return runtime.port, runtime.driver.barrier, runtime.prepare, runtime.finish, runtime.close, metadata
        except BaseException:
            # rank_worker records a fatal marker before potentially blocking
            # native shutdown; outer owned-process launcher drains peers.
            raise


class BoundPort(NativeEnginePort):
    def __init__(self, runtime):
        super().__init__(runtime.driver.engine)
        self.runtime = runtime

    def submit(self, request):
        owner = self.runtime
        require(owner.binding is None, 'overlapping native binding')
        owner.binding = dict(request_id=request.request_id, phase=owner.phase,
                             token_ids=list(request.prompt_token_ids))
        if owner.prefix_cache is not None:
            sequence = owner.prefix_cache.submit(request, self)
            owner.native_admission = dict(request_id=request.request_id,
                native_prefix_reuse_enabled=owner.prefix_reuse_enabled,
                initial_num_cached_tokens=sequence.num_cached_tokens, initial_cached_hit_tokens=sequence.cached_hit_tokens,
                native_sequence_created_fresh=True, native_fork_used=False, gpu_copy_prefix=True)
            require(sequence.num_cached_tokens == owner.observer.cached, 'prefix copy and scheduled-row admission differ')
            return sequence
        sequence = super().submit(request)
        require(sequence.num_cached_tokens == 0 and sequence.cached_hit_tokens == 0,
                'fresh Full/S2 request unexpectedly received a native prefix hit')
        owner.native_admission = dict(request_id=request.request_id, native_prefix_reuse_enabled=False,
            initial_num_cached_tokens=sequence.num_cached_tokens, initial_cached_hit_tokens=sequence.cached_hit_tokens,
            native_sequence_created_fresh=True, native_fork_used=False)
        return sequence


class NativeRuntime:
    def __init__(self, config, arm, output_dir, rank):
        self.config, self.arm, self.out, self.rank = config, arm, output_dir, rank
        self.llm = self.controller = self.binder = self.batch_worker = self.instrumentation = None
        self.stack = ExitStack()
        self.binding, self.phase, self.current_request = None, None, None
        self.schedule = self.batch_receipts = None
        self.startup = {}
        self.observer = None
        self.prefix_cache = None
        self.prefix_initialized = False
        self.prefix_reuse_enabled = True
        self.prefix_diagnostic = False
        self.prefix_history = {}
        self.session_certificate = None
        self.last_qualification = None
        self.graph_meter = None
        self.graph_phase_before = None
        self.profile = None
        self.cache_storage_baseline = None
        self.cache_storage_before = None

    def initialize(self):
        cfg, rank = self.config, self.rank
        self.args = runtime_args(cfg, self.out)
        base, native, _, scoped = load_runtime(cfg)
        from native_decode_profile import DecodeProfile, GraphReplayMeter, actual_runtime
        profile = self.profile = DecodeProfile('dspark')
        self.stack.enter_context(scoped(base, profile, args=self.args, apply_environment=True))
        sys.path.insert(0, cfg['native_repo'])
        sys.path.insert(0, str(Path(cfg['native_repo'])/'examples'))
        from _config import create_llm, setup_common_env
        setup_common_env(model_name=cfg['model_path'], sampler_force_topk=1024,
                         set_override_experts=False, enable_eplb=False)
        started = time.perf_counter()
        self.llm = create_llm(cfg['model_path'], rank, ep_size=8, cp_size=1,
            env={'DEEPINFER_LOGGING_INTERVAL_STEP':'0'}, data_parallel_size=8,
            max_num_batched_tokens=8192, max_model_len=270336, max_num_seqs=1,
            gpu_memory_utilization=.60, **profile.llm_kwargs())
        import torch
        import deepinfer
        from deepinfer import envs
        from deepinfer.utils import allow_cuda_sync
        require(Path(deepinfer.__file__).resolve().is_relative_to(Path(cfg['native_repo']).resolve()),
                'DeepInfer was imported from another tree')
        self.torch, self.allow_sync = torch, allow_cuda_sync
        torch.backends.cuda.matmul.allow_tf32 = False
        self.driver = base.NativeDriver(self.llm, self.args, diagnostic_rank=rank)
        require(self.driver.dp_size == 8, 'native collective world is not DP8')
        self.model_runner = self.driver.engine.driver_worker.model_runner
        self.model = self.model_runner.model.model
        self.layers = list(self.model.layers)
        require([m.layer_idx for m in self.layers] == list(range(40)), 'native40 model required')
        observed, validation = actual_runtime(native, profile, self.args, self.model_runner, torch,
            envs, cache_config=self.driver.engine.cache_config)
        self.graph_meter = GraphReplayMeter(self.model_runner)
        from redknot.decode_evidence import cache_storage_snapshot
        self.cache_storage_baseline = cache_storage_snapshot(self.driver.engine)
        self.startup['runtime'] = self.publish('native-runtime',
            dict(rank=rank, observed=observed, validation=validation, init_s=time.perf_counter()-started,
                graph_capture=self.graph_meter.snapshot(), native_idle_bypass_installed=False,
                native_cache_storage=self.cache_storage_baseline,
                mtp_module_type=type(self.model_runner.model.MTP_module).__name__), per_rank=True)
        self.startup['runtime'] = self.driver.gather(self.startup['runtime'])
        self.port = BoundPort(self)
        from .prefix_cache import ScheduleObserver
        self.observer = ScheduleObserver(self.driver.engine, rank)
        self.scope = cfg['scope']
        # Each arm has the same native warmup before any formal HTTP request.
        ids = tuple(cfg['qualification_input_ids'])
        warm = TokenRequest('startup-native-warmup', ids, 32, self.arm, time.perf_counter())
        self._internal(warm, 'startup-native-warmup', None)
        if self.arm == 's2':
            self._initialize_s2()
        elif self.arm == 'prefix':
            self._initialize_prefix()
        else:
            require(self.controller is None and self.binder is None and self.batch_worker is None,
                    'Full must not install RedKnot/controller/batch/idle hooks')
        self.driver.barrier()
        return dict(schema='s2-persistent-worker-ready-v1', rank=rank, arm=self.arm,
            config_sha256=cfg['sha256'], native_runtime=self.startup['runtime'][rank],
            startup=self.startup, online_request_started=False,
            redknot_hooks_installed=self.arm == 's2', persistent_engine=True)

    def publish(self, name, value, *, per_rank=False):
        if 'sha256' in value:
            require(value['sha256'] == digest({k:v for k,v in value.items() if k != 'sha256'}), 'artifact seal differs')
        else:
            value = seal(value)
        path = self.out/(name + (f'.rank{self.rank}' if per_rank else '') + '.json')
        if self.rank == 0 or per_rank:
            write_json(path, value)
        # Publish-before-consume is mandatory even on a shared filesystem.
        self.driver.barrier()
        return dict(path=str(path), sha256=value['sha256'])

    def _initialize_prefix(self):
        from .build_prefix_bridge import import_bridge
        from .prefix_cache import PrefixCache
        from redknot.admission import uniform_bank_budget
        with self.allow_sync():
            free = int(self.torch.cuda.mem_get_info()[0])
        free8 = self.driver.gather(free)
        cap = uniform_bank_budget(free8)
        proof = self.config['prefix_host_bridge']
        require(proof['torch_version'] == self.torch.__version__
            and proof['torch_cxx11_abi'] == self.torch._C._GLIBCXX_USE_CXX11_ABI,
            'prefix descriptor was compiled for another native Torch ABI')
        if self.rank == 0:
            bridge = import_bridge(proof)
            self.prefix_cache = PrefixCache(self.driver.engine, bridge, self.torch, self.allow_sync, cap)
            self.observer.prefix = self.prefix_cache
        self.prefix_initialized = True
        self.startup['prefix_budget'] = self.publish('prefix-bank-budget',dict(
            actual_free_bytes_by_rank=free8,bank_budget_bytes=cap,formula='min(4GiB,min(all8free)//5)',
            eviction_policy='whole_snapshot_LRU',native_state_slots_unchanged=True))
        self._qualify_prefix()

    def _qualify_prefix(self):
        from tokenizers import Tokenizer
        from .checkpoint_codec import CheckpointCodec
        from redknot.prefix_gate import compare_full_prefix
        from redknot.decode_evidence import validate_decode_phase
        cases = None
        if self.rank == 0:
            codec = CheckpointCodec(self.config['encoder_path'],Tokenizer.from_file(self.config['tokenizer_path']),
                thinking_mode=self.config['thinking_mode'],reasoning_effort=self.config['reasoning_effort'])
            root = Path(self.config['native_repo'])
            source_path = max(root.rglob('*.py'), key=lambda p:(p.stat().st_size,str(p.relative_to(root))))
            source = source_path.read_text()
            def render(system):
                return codec.render(dict(messages=[dict(role='system',content=system),
                    dict(role='user',content=[dict(type='text',text=source),
                        dict(type='text',text='Explain one concrete constraint briefly.')])]))
            original, changed = render('Inspect this code.'), render('A different system instruction.')
            ids, other = list(original['input_ids']),list(changed['input_ids'])
            require(len(ids) > 10000 and len(other) > 8194,
                    'natural source too short for controlled native checkpoint qualification')
            cold = ids[:8194]
            suffix_at = next((i for i in range(8194,len(ids)-1) if ids[i:i+2] != cold[-2:]),None)
            require(suffix_at is not None, 'distinct actual source suffix required')
            sources = dict(path=str(source_path.relative_to(root)), source_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
                rendered_source_input_sha256=original['input_ids_sha256'],
                controlled_token_boundary_diagnostic=True, no_padding=True, official_task=False)
            cases = [dict(case_id='cold',input_ids=cold,documents=[],source=sources),
                dict(case_id='hit',input_ids=ids[:10000],documents=[],source=sources),
                dict(case_id='changed_prefix',input_ids=other[:8194],documents=[],
                    source=dict(sources,rendered_source_input_sha256=changed['input_ids_sha256'])),
                dict(case_id='repeated_short_suffix',input_ids=cold[:8192]+ids[suffix_at:suffix_at+2],
                    documents=[],source=dict(sources,suffix_source_start=suffix_at))]
        cases=self.driver.gather(cases)[0]
        comparisons=[]
        source_snapshots={}
        self.prefix_diagnostic=True
        for case in cases:
            request=TokenRequest('prefix-qualification.'+case['case_id']+'.full',tuple(case['input_ids']),32,
                'prefix',time.perf_counter(),documents=tuple(case['documents']))
            self.prefix_reuse_enabled=False
            reference=self._internal(request,request.request_id,None)
            self.prefix_reuse_enabled=True
            request=replace(request,request_id='prefix-qualification.'+case['case_id']+'.prefix')
            candidate=self._internal(request,request.request_id,None)
            reference_decode = validate_decode_phase(reference)
            candidate_decode = validate_decode_phase(candidate)
            result=compare_full_prefix(case['input_ids'],reference['record'],candidate['record'],
                candidate['prefix_report'],candidate['scheduled_rows_reports'],reference['scheduled_rows_reports'],
                source_snapshots=source_snapshots)
            require(candidate['prefix_report']['cache_hit'] is (case['case_id'] in ('hit','repeated_short_suffix')),
                    'prefix qualification did not exercise required cold/hit/changed-prefix-miss')
            if case['case_id'] == 'repeated_short_suffix':
                require(len(case['input_ids'])-candidate['prefix_report']['reused_prefix_tokens'] == 2,
                        'actual Prefix resume must exercise exactly two uncached suffix tokens')
            comparisons.append(dict(case_id=case['case_id'],source=case['source'],comparison=result,
                reference_decode=reference_decode, candidate_decode=candidate_decode,
                reference_sha256=reference['sha256'],candidate_sha256=candidate['sha256']))
            for receipt in candidate['prefix_report']['captures']:
                if 'snapshot_id' in receipt:
                    source_snapshots[receipt['snapshot_id']] = dict(receipt=receipt,input_ids=case['input_ids'])
        self.prefix_diagnostic=False
        self.prefix_reuse_enabled=True
        if self.prefix_cache is not None:self.prefix_cache.clear()
        self.startup['prefix_qualification']=self.publish('prefix-qualification',dict(
            schema='s2-prefix-copy-fresh-qualification-v1',passed=True,comparisons=comparisons,
            actual_cold_miss=True,actual_prefix_hit=True,actual_changed_prefix_miss=True,
            actual_repeat_after_unrelated_request=True,actual_two_token_suffix=True,
            native_decode_profile='dspark',native_idle_bypass_installed=False,
            strict_complete32_or_eos_parity=True,benchmark_bank_cleared=True,
            representative_gpu_qualification=True,universal_input_equivalence_claim=False))

    def _initialize_s2(self):
        from gpu_pilot.cuda_mask_qualification import run_cuda_mask_qualification, validate_report
        from gpu_pilot.index_hit20_empty_ep_qualification import (
            run_empty_ep_qualification, build_all_ranks_report, EmptyEPCapability)
        from kv_cache_engram.s1_runtime import run_s1_runtime_qualification
        from gpu_pilot.read_s11_results import validate_runtime_qualification
        from kv_cache_engram.s1_system_batch import run_s1_system_metadata_qualification
        from kv_cache_engram.s1_system_runtime import install_worker_batch_schedule
        from canonical_request_binding import install_request_binding
        from redknot.controller import adapted_controller_class
        from redknot.admission import uniform_bank_budget
        t, allow, g, rank = self.torch, self.allow_sync, self.driver.gather, self.rank
        masks = g(run_cuda_mask_qualification(t, allow))
        require(len(masks) == 8, 'all8 mask qualification required')
        for report in masks:
            validate_report(report)
        self.startup['mask'] = self.publish('qualification-mask', dict(passed=True, reports=masks))
        ep = build_all_ranks_report(g(run_empty_ep_qualification(self.layers, rank, t, allow,
            execution_sha256=self.scope['execution_sha256'], model_sha256=self.scope['model_sha256'])),
            execution_sha256=self.scope['execution_sha256'], model_sha256=self.scope['model_sha256'])
        self.startup['empty_ep'] = self.publish('qualification-empty-ep', ep)
        rt = seal(dict(schema='s1-runtime-all-ranks-qualification-v1', passed=True,
            execution_configuration_sha256=self.scope['execution_sha256'], model_sha256=self.scope['model_sha256'],
            reports=g(run_s1_runtime_qualification(self.layers, rank, t, allow,
                execution_sha256=self.scope['execution_sha256'], model_sha256=self.scope['model_sha256'])),
            empty_ep_qualification_sha256=ep['sha256'], quality_eligible=False, timing_valid=False))
        validate_runtime_qualification(rt, dict(execution_configuration_sha256=self.scope['execution_sha256'],
            checkpoint_index_sha256=self.scope['model_sha256']), ep['sha256'])
        self.startup['operators'] = self.publish('qualification-runtime', rt)
        md = g(dict(rank=rank, operator_report=run_s1_system_metadata_qualification(t, allow,
            device='cuda', group_chunks=(17,))))
        for i, item in enumerate(md):
            r = item['operator_report']
            require(item['rank'] == i and r.get('sha256') == digest({k:v for k,v in r.items() if k != 'sha256'})
                and r.get('passed') is True and r.get('cuda_exercised') is True
                and r.get('sync_debug_mode_during') == 2 and r.get('requested_batch_groups') == [17]
                and r.get('sync_debug_mode_before') == r.get('sync_debug_mode_after'),
                'fresh all8 ERROR-mode group17 CUDA metadata qualification failed')
        self.startup['metadata17'] = self.publish('qualification-metadata17', dict(passed=True, reports=md))
        capability = EmptyEPCapability(self.layers, ep, execution_sha256=self.scope['execution_sha256'],
                                     model_sha256=self.scope['model_sha256'])
        self.binder = install_request_binding(self.model_runner, lambda: self.binding if rank == 0 else None)
        if rank == 0:
            self.controller = adapted_controller_class()(self.layers, allow_sync=allow, native_model=self.model,
                model_sha256=self.scope['model_sha256'], tokenizer_sha256=self.scope['tokenizer_sha256'],
                execution_sha256=self.scope['execution_sha256'], dependency_root=Path(self.config['experiment_root']),
                empty_ep_capability=capability, enable_dflash_tail=True)
            self.controller.install()
        # The frozen batch installer requires an unwrapped native engine. The
        # same read-only meter is then installed outside it and removed first.
        self.observer.close()
        self.observer = None
        self.batch_worker = install_worker_batch_schedule(self.model, rank=rank, torch_module=t,
            empty_ep_capability=capability, controller=self.controller, engine=self.driver.engine,
            phase_gather=g, allow_native_decode_graphs=True)
        from .prefix_cache import ScheduleObserver
        self.observer = ScheduleObserver(self.driver.engine, rank)
        ids = tuple(self.config['qualification_input_ids'])
        req = TokenRequest('startup-native-identity', ids, 32, 's2', time.perf_counter())
        ref = self._internal(req, 'startup-qualified-native', 'qualified_native')
        candidate = self._internal(replace(req, request_id='startup-all-selected'),
                                   'startup-all-selected', 'all_selected')
        from redknot.decode_evidence import validate_decode_phase
        native_decode = validate_decode_phase(ref)
        all_selected_decode = validate_decode_phase(candidate)
        r, c = ref['controller'], candidate['controller']
        require(r['identity_check_count'] == c['identity_check_count'] == 80
            and c['identity_exact'] is True and r['identity_fingerprints'] == c['identity_fingerprints']
            and r['identity_fingerprints_sha256'] == c['identity_fingerprints_sha256']
            and ref['record']['output_ids'] == candidate['record']['output_ids'],
            'fresh16K native/all-selected exact identity failed')
        self.startup['native_identity'] = self.publish('qualification-native-identity', dict(passed=True,
            reference_sha256=ref['sha256'], candidate_sha256=candidate['sha256'], identity_check_count=80,
            benchmark_data=False, input_ids_sha256=token_digest(ids), full_prompt_equivalence_claim=False,
            generation_budget=32, generated_tokens=len(ref['record']['output_ids']),
            native_decode=native_decode, all_selected_decode=all_selected_decode,
            mtp5_decode_exercised=True, natural_eos_qualification=False))
        with allow():
            free = int(t.cuda.mem_get_info()[0])
        free8 = g(free)
        self.bank_budget = uniform_bank_budget(free8)
        self.startup['bank_budget'] = self.publish('shared-bank-budget', dict(actual_free_bytes_by_rank=free8,
            bank_budget_bytes=self.bank_budget, formula='min(4GiB,min(all8free)//5)', eviction=False))
        proc = dict(rank=rank, pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
            argv=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v])
        self.startup['startup_session'] = self.publish('startup-session', dict(
            schema='s2-native-session-admission-v1', session_id=self.config['session_id'],
            config_sha256=self.config['sha256'], scope=self.scope, rank_processes=g(proc),
            startup_refs=dict(self.startup), fresh_process_qualification=True))
        if self.config['per_prompt_qualification'] == 'representative_session':
            self._qualify_session()

    def _qualify_session(self):
        from tokenizers import Tokenizer
        from .checkpoint_codec import CheckpointCodec
        from redknot.qualification_cases import rendered_requests
        from redknot.session import issue_certificate, validate_certificate
        started = time.perf_counter()
        cases = None
        if self.rank == 0:
            codec = CheckpointCodec(self.config['encoder_path'], Tokenizer.from_file(self.config['tokenizer_path']),
                thinking_mode=self.config['thinking_mode'], reasoning_effort=self.config['reasoning_effort'])
            cases = rendered_requests(codec, source_root=Path(self.config['native_repo']))
        cases = self.driver.gather(cases)[0]
        evidence = []
        for case in cases:
            request = TokenRequest('session.'+case['case_id'], tuple(case['input_ids']), 32, 's2',
                time.perf_counter(), documents=tuple(case['documents']))
            if case['expected_route'] == 's2':
                preparation = self.prepare(request, qualification_only=True)
                require(self.last_qualification is not None, 'actual sparse qualification missing')
                # This is the actual candidate phase, not a fabricated formal request.
                record = self.last_qualification['candidate_phase']['record']
            else:
                local = run_serial_request(self.port, request, rank=self.rank, barrier=self.driver.barrier,
                    prepare=self.prepare, finish=self.finish, emit=lambda event:None)
                record = self.driver.gather(local)[0]
                preparation = record['preparation']
            require(preparation['effective_arm'] == ('s2' if case['expected_route'] == 's2' else 'full'),
                    'representative qualification unexpectedly fell back')
            self.publish('session-case.'+case['case_id'], dict(schema='s2-dspark-representative-actual-case-v1',
                case_id=case['case_id'], source_refs=case['source_refs'], record=record,
                preparation=preparation, redundant_formal_request_omitted=case['expected_route'] == 's2',
                input_ids_sha256=case['input_ids_sha256'], actual_gpu_executed=True, official_task=False,
                qualified=self.last_qualification))
            if self.last_qualification is not None:
                evidence.append(dict(case_id=case['case_id'], **self.last_qualification))
        source_map = dict(self.config['source_sha256']['external'])
        for prefix, section in (('GatedEngram/', 'gated'), ('NativeSnapshot/', 'native'), ('runtime/', 'runtime')):
            source_map.update({prefix+k:v for k,v in self.config['source_sha256'][section].items()})
        cert = issue_certificate(scope=self.scope, session_id=self.config['session_id'], source_sha256=source_map,
            startup_refs=self.startup, case_evidence=evidence)
        validate_certificate(cert, scope=self.scope, session_id=self.config['session_id'], source_sha256=source_map)
        cert_ref = self.publish('session-certificate', cert)
        # Diagnostic corpus must not prewarm or consume the benchmark bank.
        if self.controller is not None:
            with self.allow_sync():
                self.controller.clear_cache(reset_sparse_qualification=True)
                require(self.controller.resident_bank_state()['resident_bank_bytes'] == 0,
                        'qualification corpus remained resident before formal tasks')
                self.controller.set_session_certificate(cert, session_id=self.config['session_id'], source_sha256=source_map)
        self.prefix_history.clear()
        self.session_certificate = cert
        self.startup['session_certificate'] = cert_ref
        self.startup['session_qualification_cost'] = self.publish('session-qualification-cost', dict(
            setup_s=time.perf_counter()-started, official_requests_executed=0,
            diagnostic_bank_cleared=True, included_in_whole_experiment_wall_time=True,
            repeated_in_each_online_request=False))
        self.driver.barrier()

    def _configure(self, request, phase, mode, *, verify_sparse=False):
        self.phase = phase
        self.phase_mode = mode
        self.graph_phase_before = None if self.graph_meter is None else self.graph_meter.snapshot()
        if self.cache_storage_baseline is not None:
            from redknot.decode_evidence import cache_storage_snapshot
            self.cache_storage_before = cache_storage_snapshot(self.driver.engine)
            require(self.cache_storage_before == self.cache_storage_baseline,
                    'native cache storage changed before request after graph capture')
        cached = 0
        if self.prefix_initialized:
            cached = self.prefix_cache.begin(request, diagnostic=self.prefix_diagnostic,
                enabled=self.prefix_reuse_enabled) if self.prefix_cache is not None else None
            cached = self.driver.gather(cached)[0]
        if self.observer is not None:
            self.observer.configure(request, cached)
        enabled = mode == 'sensitive'
        c = self.controller
        if self.batch_worker is None:
            require(mode is None and not verify_sparse, 'Full cannot enter a RedKnot phase')
            return
        if c is not None:
            with self.allow_sync():
                c.configure(system_enabled=mode in ('reference_sparse','sensitive'),
                            batch_group_chunks=17 if enabled else 1)
                arm = ('s1' if mode in ('reference_sparse','sensitive') else
                       'qualification' if mode in ('qualified_native','all_selected') else
                       'offline' if mode == 'capture_independent' else 'recomputed')
                c.reset(request.request_id, mode or 'native', arm=arm, input_ids=request.prompt_token_ids,
                    documents=None if mode == 'capture_independent' else list(request.documents), phase=phase, verify_sparse=verify_sparse,
                    index_device=self.layers[0].self_attn.q_a_kv_swa_proj.weight.device
                        if mode in ('reference_sparse','sensitive','all_selected') else None)
        self.schedule = self.driver.gather(c.batch_schedule if c is not None else None)[0]
        self.batch_receipts = self.driver.gather(self.batch_worker.configure(self.schedule))
        if self.schedule is not None and c is not None:
            c.bind_batch_schedule(self.schedule, self.batch_receipts)
        require(self.instrumentation is None, 'MTP0 idle bypass must not be installed under D-Spark')

    def _finish_phase(self, request):
        workers, reports, instruments, scheduling = [], [], [], None
        if self.batch_worker is not None:
            workers = self.driver.gather(self.batch_worker.finish_phase())
            report = None
            if self.controller is not None:
                with self.allow_sync():
                    report = self.controller.finish()
            reports = self.driver.gather(report)
            require(reports[0] is not None and all(v is None for v in reports[1:]), 'only rank0 controller allowed')
            require(self.instrumentation is None, 'unexpected idle bypass instrumentation')
            instruments = self.driver.gather(None)
            if self.phase_mode == 'sensitive':
                scheduling = validate_candidate(instruments, self.schedule, workers,
                    phase=self.phase, request_id=request.request_id)
            else:
                require(self.schedule is None and all(v is None for v in instruments)
                    and all(w['enabled'] is False for w in workers), 'noncandidate phase installed batching/idle')
        if self.rank == 0:
            require(self.binding is not None and self.binding['request_id'] == request.request_id,
                    'request binding ownership changed')
        self.binding = None
        scheduled = self.driver.gather(self.observer.finish()) if self.observer is not None else []
        prefix = (self.driver.gather(self.prefix_cache.finish() if self.prefix_cache is not None else None)[0]
                  if self.prefix_initialized else None)
        admissions = self.driver.gather(self.native_admission if self.rank == 0 else None)
        require(all(item is None for item in admissions[1:]), "inactive ranks submitted a request")
        graph = None
        if self.graph_meter is not None:
            after = self.graph_meter.snapshot()
            require(self.graph_phase_before is not None, 'graph phase start missing')
            graph = dict(rank=self.rank, profile=self.profile.name, before=self.graph_phase_before, after=after,
                successful_replay_deltas={key:after[key]['successful_replay_calls']-
                    self.graph_phase_before[key]['successful_replay_calls'] for key in ('main','dflash')},
                includes_native_idle=True, speculative_acceptance_rate=None)
        graph_reports = self.driver.gather(graph)
        storage = None
        if self.cache_storage_baseline is not None:
            from redknot.decode_evidence import cache_storage_snapshot
            after = cache_storage_snapshot(self.driver.engine)
            require(after == self.cache_storage_before == self.cache_storage_baseline,
                    'request replaced native cache storage used by captured graphs')
            storage = dict(rank=self.rank, baseline_sha256=self.cache_storage_baseline['sha256'],
                before_sha256=self.cache_storage_before['sha256'], after_sha256=after['sha256'],
                native_cache_storage_unchanged=True)
        return dict(decode_graph_evidence=graph_reports, native_cache_storage_evidence=self.driver.gather(storage),
            native_idle_bypass_installed=False,
            native_prefix_admission=admissions[0], scheduled_rows_reports=scheduled, prefix_report=prefix, controller=reports[0] if reports else None, batch_schedule=self.schedule,
            batch_configuration_receipts=self.batch_receipts, batch_worker_reports=workers,
            idle_reports=instruments, world_scheduler=scheduling)

    def _internal(self, request, phase, mode, *, verify_sparse=False):
        local = run_serial_request(self.port, request, rank=self.rank, barrier=self.driver.barrier,
            prepare=lambda r:self._configure(r, phase, mode, verify_sparse=verify_sparse),
            finish=self._finish_phase, emit=lambda event:None)
        record = self.driver.gather(local)[0]
        record['text'] = self.driver.engine.tokenizer.decode(record['output_ids'], skip_special_tokens=False)
        body = seal(dict(schema='s2-internal-native-phase-v1', name=phase, mode=mode,
            record={k:v for k,v in record.items() if k not in ('preparation','completion')},
            **record['completion'], qualification_only=True, benchmark_request=False))
        self.publish(phase, body)
        return body

    def prepare(self, request, *, qualification_only=False):
        require(request.arm == self.arm and self.current_request is None, 'service arm/request ownership differs')
        self.current_request = request.request_id
        began = time.perf_counter()
        if self.arm in ('full','prefix'):
            self._configure(request, request.request_id+'.formal', None)
            return dict(requested_arm=self.arm, effective_arm=self.arm, config_sha256=self.config['sha256'], redknot_hooks_installed=False,
                        cold_capture_s=0.0, online_qualification_s=0.0)
        from redknot.admission import plan_request
        from redknot.gate import verify_sparse_pair
        c, plan = self.controller, None
        if c is not None:
            plan = plan_request(request.prompt_token_ids, request.documents, self.scope,
                bank_budget_bytes=self.bank_budget, session_certificate=self.session_certificate, **c.resident_bank_state())
            missing = c.begin_request(plan, request_id=request.request_id)
        else:
            missing = None
        plan, missing = self.driver.gather((plan, missing))[0]
        plan_ref = self.publish(request.request_id+'.plan', plan)
        capture_started = time.perf_counter()
        captures, gate_ref = [], None
        if plan['route'] == 's2':
            for ordinal, key in enumerate(missing):
                doc = next(d for d in plan['effective_documents'] if d['document_id'] == key)
                ids = tuple(request.prompt_token_ids[doc['input_start']:doc['input_end']])
                capture = TokenRequest(request.request_id+f'.capture{ordinal}', ids, 1, 's2', request.received_s)
                phase = self._internal(capture, capture.request_id, 'capture_independent')
                captures.append(dict(document_id=key, phase_sha256=phase['sha256']))
            if c is not None:
                with self.allow_sync():
                    c.prepare_case(input_ids=request.prompt_token_ids, documents=plan['effective_documents'],
                        index_device=self.layers[0].self_attn.q_a_kv_swa_proj.weight.device)
            self.driver.barrier()
        capture_s = time.perf_counter()-capture_started
        qualify_started = time.perf_counter()
        self.last_qualification = None
        if plan['route'] == 's2' and self.session_certificate is None:
            qualified_request = replace(request, request_id=request.request_id+'.dense32', max_tokens=32,
                temperature=0.0, top_p=1.0, seed=0, documents=tuple(plan['effective_documents']))
            reference = self._internal(qualified_request, qualified_request.request_id, 'reference_sparse')
            qualified_request = replace(qualified_request, request_id=request.request_id+'.candidate32')
            candidate = self._internal(qualified_request, qualified_request.request_id, 'sensitive', verify_sparse=True)
            gate = verify_sparse_pair(plan, request.prompt_token_ids, reference['controller'], candidate['controller'],
                reference['record'], candidate['record'])
            gate_ref = self.publish(request.request_id+'.sparse-gate', gate)
            self.last_qualification = dict(plan=plan, input_ids=list(request.prompt_token_ids),
                reference_phase=reference, candidate_phase=candidate)
        qualification_s = time.perf_counter()-qualify_started
        actual = self.driver.gather(c.resident_bank_state() if c is not None else None)[0]
        require(actual['resident_bank_bytes'] <= self.bank_budget, 'actual resident bank exceeded admitted capacity')
        if qualification_only:
            require(plan['route'] == 's2' and self.last_qualification is not None,
                    'qualification-only requires actual fresh dense/candidate phases')
            self.current_request = None
        else:
            self._configure(replace(request, documents=tuple(plan['effective_documents'])),
                request.request_id+'.formal', 'sensitive' if plan['route'] == 's2' else None)
        self.request_plan = plan
        reuse, self.pending_prefixes = reuse_coverage(request.prompt_token_ids, request.documents,
            plan['effective_documents'], plan['resident_document_ids'], self.prefix_history)
        return dict(requested_arm='s2', config_sha256=self.config['sha256'], effective_arm=plan['effective_arm'], fallback_reason=plan['fallback_reason'],
            plan=plan_ref, numerical_gate=gate_ref, startup_admission=self.startup, reuse=reuse,
            cold_capture_s=capture_s, online_qualification_s=qualification_s,
            online_preparation_so_far_s=time.perf_counter()-began,
            unique_misses=len(missing) if plan['route'] == 's2' else 0, captures=captures,
            prior_bank_bytes=plan['resident_bank_bytes'], bank_budget_bytes=self.bank_budget,
            actual_resident_bank=actual, redknot_hooks_installed=True,
            strict_per_prompt_qualification=self.session_certificate is None,
            numerical_admission_scope='representative_session' if self.session_certificate is not None else 'strict_request',
            session_certificate_sha256=None if self.session_certificate is None else self.session_certificate['sha256'],
            current_prompt_numerical_equivalence_claim=self.session_certificate is None and plan['route'] == 's2',
            qualification_excluded_from_ttft=False, qualification_only=qualification_only,
            redundant_formal_qualification_request_omitted=qualification_only, native_idle_bypass_installed=False)

    def finish(self, request):
        evidence = self._finish_phase(request)
        ref = self.publish(request.request_id+'.formal', dict(schema='s2-live-formal-evidence-v1',
            request_id=request.request_id, arm=self.arm, input_ids_sha256=token_digest(request.prompt_token_ids),
            **evidence))
        if self.arm == 's2':
            for key, prefix in self.pending_prefixes:
                self.prefix_history.setdefault(key, set()).add(prefix)
        self.current_request = None
        return dict(formal_evidence=ref, completed_all8_rank_admission=True)

    def close(self):
        failures = []
        for obj, method in ((self.instrumentation,'restore'), (self.graph_meter,'close'), (self.observer,'close'), (self.batch_worker,'close'),
                            (self.controller,'close'), (self.binder,'restore'), (self.llm,'close')):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except BaseException as error:
                    failures.append(repr(error))
        self.stack.close()
        if failures:
            raise RuntimeError('owned native shutdown failed: '+repr(failures))
