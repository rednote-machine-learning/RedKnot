"""Isolated native CSA2 + YOCO-compact qualification, not a RedKnot result.

Model loading/JIT are excluded. Quality uses natural EOS; performance uses
warmup followed by repeated, fixed-output, closed-loop request waves. Every
rank owns at most one live sequence. A C2 pilot is NOT saturated-server QPS.
Existing runner, checkpoint, native source and frozen experiments are unchanged.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for subdir in ('benchmark', 'kv_recompression_20261029'):
    sys.path.insert(0, str(ROOT / subdir))
import pr148_native_runner as base
from native_profile import NativeProfile, scoped_base_profile, validate_runtime


def checkpoint_maps(config):
    text = config.get('text_config', config)
    n = text['num_hidden_layers']
    if type(n) is not int or n < 1:
        raise ValueError('invalid checkpoint main layer count')
    def expand(ids):
        if (not isinstance(ids, list) or not ids
                or any(type(i) is not int or not 0 <= i < n for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('invalid checkpoint CSA2 source layers')
        # HF _source_map sorts source ids; layers preceding the first source
        # are absent in its sparse map and self-reference after normalization.
        ids = sorted(ids)
        return [max((s for s in ids if s <= i), default=i) for i in range(n)]
    candidate = text['candidate_source_layer_id']
    if candidate is not None and (type(candidate) is not int or not 0 <= candidate < n):
        raise ValueError('invalid checkpoint candidate source layer')
    return {
        'layer_kv_map': expand(text['kv_source_layer_ids']),
        'layer_index_module_map': expand(text['index_source_layer_ids']),
        'layer_index_k_map': expand(text.get('index_k_source_layer_ids', text['kv_source_layer_ids'])),
        'layer_candidate_source_map': {} if candidate is None else {str(i): candidate for i in range(candidate, n)},
    }


def stage_sources():
    # Other independent adapters/tests in this directory may still evolve.
    # Freeze only executable dependencies of this native qualification path.
    return {name: base.file_digest(HERE / name) for name in
            ('strong_native_runner.py', 'native_profile.py')}


def gpu_runtime_imported():
    return any(name == 'torch' or name.startswith('torch.') or name == 'deepinfer'
               or name.startswith('deepinfer.') for name in sys.modules)


def quality_eligible(manifest, rows):
    return (manifest.get('artifact_kind') == 'canonical-KV-quality-pilot'
            and manifest.get('task_quality_eligible') is True
            and bool(rows) and all(row.get('continuation_is_quality_evidence') is True
                and row.get('trace_kind') != 'teacher_forced_answer'
                and row['input_tokens'] >= 262144 for row in rows))


def phase_plan(args):
    plan = [dict(name='quality', active=1, mode='quality', tokens=args.max_new_tokens,
                 repeats=1, quality_shapes=True)]
    if not args.quality_only:
        for c in args.performance_concurrencies:
            plan.extend((dict(name=f'warmup-c{c}', active=c, mode='throughput',
                              tokens=args.performance_tokens, repeats=1, quality_shapes=False),
                         dict(name=f'perf-c{c}', active=c, mode='throughput',
                              tokens=args.performance_tokens, repeats=args.measure_repeats,
                              quality_shapes=False)))
    return plan


def phase_records(records, tokenizer, mode, name, manifest, rows):
    converted = [base.canonical_record(record, tokenizer, mode, name) for record in records]
    eligible = mode == 'quality' and quality_eligible(manifest, rows)
    for record in converted:
        record['task_quality_eligible'] = eligible
        if not eligible:
            record.pop('answer_em', None)
            record.pop('answer_token_f1', None)
    return converted


def verify_frozen_inputs(args, manifest, identity, profile, *, native_source=False):
    """Recheck CPU artifacts before native import; reject stale preflight identity."""
    if identity.get('stage_sources') != stage_sources():
        raise ValueError('native runner/profile changed after CPU preflight')
    if identity.get('adapter_sha256') != base.file_digest(Path(base.__file__)):
        raise ValueError('frozen base runner changed after CPU preflight')
    if identity.get('owned_launcher_sha256') != base.file_digest(ROOT / 'kv_recompression_20261029' / 'quality_process_group.py'):
        raise ValueError('owned launcher changed after CPU preflight')
    if identity.get('native_profile') != profile.identity(args) or identity.get('fixed_env') != base.effective_env(args):
        raise ValueError('worker profile/environment differs from CPU preflight')
    for filename, key in (('config.json', 'checkpoint_config_sha256'),
                          ('model.safetensors.index.json', 'checkpoint_index_sha256'),
                          ('tokenizer.json', 'tokenizer_json_sha256')):
        if base.file_digest(args.model_path / filename) != identity.get(key):
            raise ValueError(f'checkpoint {filename} changed after CPU preflight')
    base.validate_manifest(manifest)
    disk_manifest = base.read_json(args.manifest)
    base.validate_manifest(disk_manifest)
    if disk_manifest['sha256'] != manifest['sha256']:
        raise ValueError('manifest changed after CPU preflight')
    if native_source:
        current = base.source_identity(args.repo)
        if any(current[key] != identity.get(key) for key in current):
            raise ValueError('native source changed after CPU preflight')


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--native-mtp-tokens', choices=(0, 5), type=int, default=5)
    p.add_argument('--native-keep-tokens', type=int, default=140)
    p.add_argument('--case-indices', type=int, nargs='+', default=[0])
    p.add_argument('--performance-concurrencies', type=int, nargs='+', default=[1, 2])
    p.add_argument('--performance-tokens', type=int, default=32)
    p.add_argument('--measure-repeats', type=int, default=2)
    p.add_argument('--quality-only', action='store_true')
    extra, rest = p.parse_known_args(argv)
    args = base.parse_args(rest)
    for key, value in vars(extra).items():
        setattr(args, key, value)
    return args


def validate(args, manifest):
    if (args.arm != 'recomputed' or args.mode != 'quality' or args.concurrency != 1
            or args.ep != 8 or args.cp != 1 or args.dp != 8 or args.max_num_seqs != 1
            or not args.skip_warmup or args.repeat != 1 or args.sample_id or args.limit
            or args.head_profile or args.intervention_policy or args.cache_dir
            or args.evaluate_calibration
            or args.max_new_tokens > 128 or args.max_num_batched_tokens > 8192):
        raise ValueError('native pilot requires C1 quality, CP1/DP8/EP8, one sequence/rank, explicit composite warmup')
    if (not args.case_indices or len(set(args.case_indices)) != len(args.case_indices)
            or any(not 0 <= i < len(manifest['cohort']) for i in args.case_indices)
            or not args.performance_concurrencies
            or tuple(sorted(set(args.performance_concurrencies))) != tuple(args.performance_concurrencies)
            or not set(args.performance_concurrencies) <= {1, 2, 4, 8}
            or not 1 <= args.measure_repeats <= 10 or not 1 <= args.performance_tokens <= 128):
        raise ValueError('invalid frozen cases, concurrency ramp or output/repetition budget')
    if any((args.diagnostic_progress, args.diagnostic_ep_sync, args.diagnostic_finite,
            args.diagnostic_stacks_seconds, args.diagnostic_prefill_breakdown)):
        raise ValueError('diagnostic timing is not eligible for this native performance pilot')
    if manifest.get('artifact_kind') not in ('canonical-KV-quality-pilot', 'canonical-KV-plumbing-smoke'):
        raise ValueError('use the frozen real-RAG or explicit plumbing-smoke manifest')
    for i in args.case_indices:
        base.document_specs(manifest['cohort'][i])
        if args.max_model_len < manifest['cohort'][i]['input_tokens'] + max(args.max_new_tokens, args.performance_tokens):
            raise ValueError('max_model_len does not cover every phase output budget')
    selected = [manifest['cohort'][i] for i in args.case_indices]
    if manifest['artifact_kind'] == 'canonical-KV-quality-pilot' and not quality_eligible(manifest, selected):
        raise ValueError('256K quality pilot requires intact natural heldout rows and explicit quality eligibility')
    if manifest['artifact_kind'] == 'canonical-KV-plumbing-smoke' and manifest.get('task_quality_eligible') is not False:
        raise ValueError('plumbing smoke must explicitly reject task accuracy claims')


def actual_runtime(profile, args, runner, torch, envs, *, cache_config):
    model = runner.model
    n = model.cfg.num_hidden_layers
    topology = model.shared_topology
    maps = {}
    for name in ('layer_kv_map', 'layer_index_module_map', 'layer_index_k_map'):
        full_map = list(getattr(topology, name))
        if (len(full_map) < n or any(type(value) is not int for value in full_map)
                or any(value != layer for layer, value in enumerate(full_map[n:], start=n))):
            raise ValueError('native topology must cover main layers and self-reference MTP layers')
        maps[name] = full_map[:n]
    maps['layer_candidate_source_map'] = {str(k): v for k, v in topology.layer_candidate_source_map.items() if k < n}
    if any(type(k) is not int or type(v) is not int or not 0 <= k < n or not 0 <= v <= k
           for k, v in topology.layer_candidate_source_map.items()):
        raise ValueError('candidate sharing is invalid or includes MTP layers')
    text = base.read_json(args.model_path / 'config.json')
    text = text.get('text_config', text)
    if n != text['num_hidden_layers'] or len(model.model.layers) != n:
        raise ValueError('loaded main layer count does not match checkpoint')
    expected_boundary = next((i for i, ratio in enumerate(text['compress_ratios']) if ratio == 1), None)
    if (expected_boundary is None or not 0 < expected_boundary < n
            or model.model.yoco_boundary_layer_idx != expected_boundary):
        raise ValueError('native YOCO boundary does not match checkpoint compression ratios')
    if (runner.parallel_config.context_parallel_size != 1
            or runner.parallel_config.data_parallel_size != 8
            or runner.parallel_config.expert_parallel_size != 8
            or runner.scheduler_config.chunked_prefill is not True
            or runner.scheduler_config.prefill_attempt_interval != args.prefill_attempt_interval
            or runner.scheduler_config.prefill_any_attempt_interval != args.prefill_attempt_interval):
        raise ValueError('native layout or scheduler differs from the disclosed profile')
    observed = dict(nextn=runner.nextn, use_cuda_graph=runner.use_cuda_graph,
        dflash_use_cuda_graph=runner.dflash_use_cuda_graph,
        yoco_prefill_truncation_mode=runner.yoco_prefill_truncation_mode,
        window_size=runner.window_size, keep_tokens=envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS,
        max_backtracking_tokens=envs.DEEPINFER_MAX_BACKTRACKING_TOKENS,
        mtp_varlen=envs.DEEPINFER_ENABLE_MTP_VARLEN, batch_invariant=envs.DEEPINFER_BATCH_INVARIANT,
        lookahead_tokens=cache_config.lookahead_tokens,
        mtp_module_present=model.MTP_module is not None, cp_size=runner.parallel_config.context_parallel_size,
        zero_overhead_scheduling=envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING,
        compute_capability=list(torch.cuda.get_device_capability()),
        engine_type=runner.parallel_config.engine_type.name,
        csa2_maps=maps, expected_csa2_maps=checkpoint_maps(text),
        native_yoco_boundary=model.model.yoco_boundary_layer_idx,
        loaded_main_layers=len(model.model.layers))
    return observed, validate_runtime(profile, observed, args=args)


def worker(args, manifest, identity, dp_size, rank):
    profile = NativeProfile(mtp_tokens=args.native_mtp_tokens, keep_tokens=args.native_keep_tokens)
    with scoped_base_profile(base, profile, args=args, apply_environment=True):
        if gpu_runtime_imported():
            raise RuntimeError('worker imported GPU runtime before its guarded native entry')
        verify_frozen_inputs(args, manifest, identity, profile)
        return worker_with_profile(args, manifest, identity, dp_size, rank, profile)


def worker_with_profile(args, manifest, identity, dp_size, rank, profile):
    sys.path.insert(0, str(args.repo / 'examples'))
    sys.path.insert(0, str(args.repo))
    llm = None
    handles = []
    try:
        from _config import create_llm, setup_common_env
        setup_common_env(model_name=str(args.model_path), sampler_force_topk=args.sampler_force_topk,
                         set_override_experts=False, enable_eplb=False)
        init_start = time.perf_counter()
        llm = create_llm(str(args.model_path), rank, ep_size=args.ep, cp_size=args.cp,
            env={'DEEPINFER_LOGGING_INTERVAL_STEP': '0'}, data_parallel_size=dp_size,
            max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
            max_num_seqs=1, gpu_memory_utilization=args.gpu_memory_utilization, **profile.llm_kwargs())
        init_s = time.perf_counter() - init_start
        import torch
        from deepinfer import envs
        driver = base.NativeDriver(llm, args, diagnostic_rank=rank)
        if driver.dp_size != dp_size or dp_size != 8:
            raise ValueError('native DP process group differs from the declared layout')
        runner = driver.engine.driver_worker.model_runner
        observed, validation = actual_runtime(profile, args, runner, torch, envs,
                                               cache_config=driver.engine.cache_config)
        base.write_json(args.out_dir / f'native-runtime.rank{rank}.json', base.seal({
            'rank': rank, 'observed': observed, 'validation': validation, 'profile': profile.identity(args)}))
        shape_rows = {}
        def observe_query(module, positional, keyword):
            meta = keyword.get('input_metadata')
            if meta is not None and meta.is_prompt:
                key = str(module.layer_idx)
                item = shape_rows.setdefault(key, {'calls': 0, 'query_rows': 0, 'max_query_rows': 0})
                item['calls'] += 1
                item['query_rows'] += meta.total_q
                item['max_query_rows'] = max(item['max_query_rows'], meta.total_q)
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]
        phases = {}
        requested_plan = phase_plan(args)
        original_mode, original_tokens = args.mode, args.max_new_tokens

        def run(name, rows, *, active, mode, tokens, repeats, quality_shapes=False):
            args.mode, args.max_new_tokens = mode, tokens
            jobs = [(f'{name}-rank{rank}-r{r}-i{i}', row) for r in range(repeats)
                    for i, row in enumerate(rows)] if rank < active else []
            if quality_shapes:
                shape_rows.clear()
                for layer in runner.model.model.layers:
                    handles.append(layer.self_attn.register_forward_pre_hook(observe_query, with_kwargs=True))
            driver.barrier()
            if rank == 0:
                print(json.dumps({'phase': name, 'status': 'starting', 'active_ranks': active}), flush=True)
            events = []
            try:
                records, window = base.run_phase(driver, jobs, int(rank < active), rank, epoch, tokens, events.append)
            finally:
                for h in handles:
                    h.remove()
                handles.clear()
            expected = active * repeats * len(rows)
            if len(records) != expected or any(not record.success(mode) for record in records):
                raise RuntimeError(f'{name}: incomplete generation; refusing performance claims')
            converted = phase_records(records, driver.engine.tokenizer, mode, name, manifest, rows)
            for event in events:
                if 'logprob' in event and not math.isfinite(event['logprob']):
                    event['logprob'] = None
            base.write_json(args.out_dir / f'{name}.events.rank{rank}.json', base.seal({'events': events}))
            summary = base.summarize(converted, window)
            summary['timing_valid'] = mode == 'throughput' and not quality_shapes
            summary['pilot_timing_available'] = summary['timing_valid'] and name.startswith('perf-')
            summary['qps_comparison_eligible'] = (summary['pilot_timing_available']
                and manifest['artifact_kind'] != 'canonical-KV-plumbing-smoke')
            summary['task_quality_eligible'] = mode == 'quality' and quality_eligible(manifest, rows)
            summary['saturated_server_qps'] = False
            entry = {'records': converted, 'summary': summary, 'window': window,
                     'sampling': base.sampling(args),
                     'concurrency': active, 'native_query_rows': driver.gather(copy.deepcopy(shape_rows)) if quality_shapes else None}
            phases[name] = entry
            if rank == 0:
                base.write_json(args.out_dir / f'{name}.json', base.seal(entry))
                print(json.dumps({'phase': name, 'status': 'complete', 'summary': summary}), flush=True)
            driver.barrier()

        rows = [manifest['cohort'][i] for i in args.case_indices]
        try:
            for step in requested_plan:
                # Same prompts warm kernel/allocator shapes, never prefix reuse.
                run(rows=rows, **step)
        finally:
            args.mode, args.max_new_tokens = original_mode, original_tokens
        memory = driver.gather({'rank': rank, 'max_allocated_bytes': torch.cuda.max_memory_allocated(),
                                'max_reserved_bytes': torch.cuda.max_memory_reserved()})
        if rank == 0:
            result = dict(schema='redknot-strong-native-pilot-v1', status='complete', arm='recomputed',
                identity=identity, profile=profile.identity(args), phases=phases,
                manifest_sha256=manifest['sha256'], case_indices=args.case_indices,
                engine_init_s=init_s, memory=memory, measurement_boundary=base.BOUNDARY,
                redknot_enabled=False, whole_model_compute_measured=False,
                quality_eligible=quality_eligible(manifest, rows),
                strongest_profile_certified=False,
                measured_concurrencies=[] if args.quality_only else args.performance_concurrencies,
                phase_plan=requested_plan,
                limitations=['native profile qualification, not an incremental RedKnot speedup',
                    'closed-loop finite workload, not saturated production QPS',
                    'quality shape instrumentation is excluded from performance phases',
                    'same frozen prompts used for explicit shape warmup; not offline model calibration',
                    'active ranks repeat identical frozen inputs; not a production arrival trace'])
            base.write_json(args.out_dir / 'run.json', base.seal(result))
    except BaseException as exc:
        p = args.out_dir / f'failure.rank{rank}.json'
        if not p.exists():
            base.write_json(p, base.seal({'status': 'failed', 'rank': rank, 'error': f'{type(exc).__name__}: {exc}'}))
        raise
    finally:
        for h in handles:
            h.remove()
        if llm is not None:
            llm.close()


def main(argv=None):
    args = parse_args(argv)
    profile = NativeProfile(mtp_tokens=args.native_mtp_tokens, keep_tokens=args.native_keep_tokens)
    with scoped_base_profile(base, profile, args=args):
        manifest, identity = base.preflight(args)
        validate(args, manifest)
        identity.update(native_profile=profile.identity(args), expected_csa2_maps=checkpoint_maps(base.read_json(args.model_path/'config.json')),
            stage_sources=stage_sources(),
            owned_launcher_sha256=base.file_digest(ROOT/'kv_recompression_20261029'/'quality_process_group.py'),
            phase_plan=phase_plan(args),
            no_redknot=True, existing_runner_modified=False)
        if gpu_runtime_imported():
            raise RuntimeError('CPU preflight imported GPU runtime')
        if args.check_only:
            print(json.dumps({'status': 'cpu_preflight_passed', 'profile': profile.identity(args),
                              'case_tokens': [manifest['cohort'][i]['input_tokens'] for i in args.case_indices],
                              'gpu_initialized': False}))
            return 0
        args.out_dir.mkdir(parents=True, exist_ok=False)
        base.write_json(args.out_dir / 'preflight.json', base.seal({'identity': identity, 'manifest_sha256': manifest['sha256']}))
        base.wait_for_release(args, manifest['sha256'])
        verify_frozen_inputs(args, manifest, identity, profile, native_source=True)
        from quality_process_group import launch
        launch(functools.partial(worker, args, manifest, identity), args.dp, args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
