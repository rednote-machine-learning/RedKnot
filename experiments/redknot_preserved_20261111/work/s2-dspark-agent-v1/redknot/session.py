"""Explicit S2 representative-GPU session admission, never a per-prompt proof."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import FunctionType

from kv_cache_engram.core import digest, require, validate_digest
from gpu_pilot.cuda_mask_qualification import validate_report as validate_mask
from gpu_pilot.index_hit20_empty_ep_qualification import validate_all_ranks_report
from gpu_pilot.read_s11_results import validate_runtime_qualification
from gpu_pilot.read_tail_link_results import _fingerprint
from profile_group_contract import group_contract
from .decode_evidence import validate_decode_phase
from .admission import validate_plan, validate_scope, uniform_bank_budget
from .gate import verify_sparse_pair

SCHEMA = 's2-dspark-representative-session-certificate-v1'
GEOMETRY_KEYS = ('max_input_tokens', 'max_document_occurrences', 'max_document_tokens', 'max_group_selected_rows')
REQUIRED_SOURCES = ('redknot/admission.py', 'redknot/controller.py', 'redknot/gate.py',
    'redknot/session.py', 'redknot/qualification_cases.py', 'redknot/decode_evidence.py',
    'native_decode_profile.py', 'dflash_tail_adapter.py', 'profile_group_contract.py',
    'GatedEngram/kv_cache_engram/s1_runtime.py', 'GatedEngram/kv_cache_engram/s1_system_runtime.py',
    'GatedEngram/kv_cache_engram/s1_dflash_bridge.py',
    'GatedEngram/kv_cache_engram/s1_plan.py', 'GatedEngram/kv_cache_engram/s1_system_batch.py',
    'GatedEngram/kv_cache_engram/s1_cache.py', 'GatedEngram/kv_cache_engram/s1_system_cache.py',
    'GatedEngram/kv_cache_engram/s1_attention.py', 'GatedEngram/kv_cache_engram/s1_system_attention.py',
    'GatedEngram/kv_cache_engram/s11_qualification.py', 'GatedEngram/kv_cache_engram/native_reuse.py',
    'GatedEngram/kv_cache_engram/tail_link_plan.py')


def sealed(value):
    require(type(value) is dict and value.get('sha256') == digest({k:v for k,v in value.items() if k != 'sha256'}),
            'invalid sealed qualification artifact')
    return value


def read_reference(ref):
    path = Path(ref['path'])
    require(path.is_file() and not path.is_symlink(), 'qualification evidence must exist as a regular file')
    raw = path.read_bytes();value = sealed(json.loads(raw))
    require(value['sha256'] == ref['sha256'], 'qualification reference seal changed')
    require('file_sha256' not in ref or ref['file_sha256'] == hashlib.sha256(raw).hexdigest(),
            'qualification reference bytes changed')
    return value, dict(path=str(path.resolve()), sha256=value['sha256'], file_sha256=hashlib.sha256(raw).hexdigest())


def validate_sources(sources):
    require(type(sources) is dict and set(REQUIRED_SOURCES) <= set(sources), 'complete adapter/core source pins required')
    for value in sources.values(): validate_digest(value, 'session source pin')


def verify_all_rank_phase(plan, phase, *, require_idle_exercised):
    sealed(phase)
    control = phase['controller']
    row = dict(input_tokens=plan['input_tokens'], input_ids_sha256=plan['input_ids_sha256'])
    contract = group_contract(17)
    original = contract.audit_system_execution
    namespace = dict(original.__globals__)
    namespace['documents'] = lambda ignored: [(d['input_start'], d['input_end'], d['document_id'])
                                              for d in plan['effective_documents']]
    audit = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    adapted = dict(phase, system_enabled=control['system_enabled'], batch_group_chunks=control['batch_group_chunks'],
                   verify_sparse=phase['mode']=='sensitive', system_batch_schedule=phase['batch_schedule'],
                   system_batch_configuration_receipts=phase['batch_configuration_receipts'],
                   system_batch_worker_reports=phase['batch_worker_reports'])
    audit(adapted, row)
    require(len(phase['idle_reports']) == 8 and all(v is None for v in phase['idle_reports']),
            'new D-Spark stage cannot reuse the MTP0 idle bypass')
    validate_decode_phase(phase)
    if phase['mode'] != 'sensitive':
        return
    world = phase.get('world_scheduler')
    require(world.get('schema') == 's2-dspark-native-idle-scheduler-v1'
        and world.get('phase') == phase['name'] and world.get('request_id') == phase['record']['request_id']
        and world.get('schedule_sha256') == phase['batch_schedule']['sha256']
        and world.get('mtp0_idle_candidate_installed') is False
        and world.get('actual_idle_steps_bypassed') == 0
        and world.get('idle_steps_use_original_native_execution') is True
        and world.get('native_decode_graphs_allowed') is True,
        'actual native-idle scheduling evidence missing')
    if require_idle_exercised:
        require(world['actual_idle_steps'] > 0, 'qualification must exercise native world-idle')


def _startup(refs, session_id, scope):
    evidence = {}
    values = {}
    for key in ('mask', 'empty_ep', 'operators', 'metadata17', 'native_identity', 'bank_budget', 'startup_session'):
        values[key], evidence[key] = read_reference(refs[key])
    owner = values['startup_session']
    require(owner.get('schema') == 's2-native-session-admission-v1' and owner.get('session_id') == session_id
            and owner.get('scope') == scope and owner.get('fresh_process_qualification') is True,
            'startup evidence belongs to another service session')
    processes = owner.get('rank_processes')
    require(type(processes) is list and len(processes) == 8
            and [p.get('rank') for p in processes] == list(range(8))
            and len({p.get('pid') for p in processes}) == 8
            and all(type(p.get('pid')) is int and p['pid'] > 1 and type(p.get('start_ticks')) is int
                    and p['start_ticks'] > 0 and type(p.get('argv')) is list and p['argv'] for p in processes),
            'fresh all-rank process identities required')
    for key in ('mask', 'empty_ep', 'operators', 'metadata17', 'native_identity', 'bank_budget', 'runtime'):
        require(owner['startup_refs'][key] == refs[key], 'session receipt detached from raw startup evidence')
    require(len(refs['runtime']) == 8, 'all eight native runtime observations required')
    evidence['runtime'] = []
    evidence['native_cache_storage_sha256_by_rank'] = []
    for rank, ref in enumerate(refs['runtime']):
        value, item = read_reference(ref)
        require(value.get('rank') == rank and value['validation'].get('validated_runtime_configuration') is True,
                'actual native runtime configuration not verified')
        require(value['validation'].get('schema') == 's2-native-decode-runtime-validation-v1'
            and all(value['observed'].get(k) == v for k,v in dict(nextn=5,mtp_varlen=True,
                use_cuda_graph=True,dflash_use_cuda_graph=True,mtp_module_present=True).items())
            and value.get('native_idle_bypass_installed') is False, 'actual unified D-Spark profile required')
        sealed(value['native_cache_storage'])
        require(value['native_cache_storage']['layer_count'] > 40,
                'native main and DFlash cache address baseline missing')
        evidence['native_cache_storage_sha256_by_rank'].append(value['native_cache_storage']['sha256'])
        evidence['runtime'].append(item)
    masks = values['mask']['reports'];require(len(masks) == 8, 'all eight CUDA masks required')
    for mask in masks: validate_mask(mask)
    ep = values['empty_ep']
    validate_all_ranks_report(ep, execution_sha256=scope['execution_sha256'], model_sha256=scope['model_sha256'])
    validate_runtime_qualification(values['operators'], dict(execution_configuration_sha256=scope['execution_sha256'],
        checkpoint_index_sha256=scope['model_sha256']), ep['sha256'])
    metadata = values['metadata17']['reports'];require(len(metadata) == 8, 'all eight group17 metadata reports required')
    expected = [dict(name='inactive_empty_cu', passed=True, rows=0, cu=[0,0]),
        dict(name='active_queue_group17', passed=True, input_tokens=139265, member_chunks=18,
             deep_groups=2, empty_groups=0, selected_rows=128)]
    groups = [dict(group_chunks=17, start=0, end=139264, selected_rows=127, empty=False,
        sampler_output_rows=139, exact_metadata=True, strict_sync_debug_mode=2),
        dict(group_chunks=17, start=139264, end=139265, selected_rows=1, empty=False,
        sampler_output_rows=1, exact_metadata=True, strict_sync_debug_mode=2)]
    for rank, row in enumerate(metadata):
        report = sealed(row['operator_report'])
        require(row.get('rank') == rank and report.get('passed') is True and report.get('cuda_exercised') is True
                and report.get('sync_debug_mode_during') == 2 and report.get('requested_batch_groups') == [17]
                and report.get('sync_debug_mode_before') == report.get('sync_debug_mode_after')
                and report.get('checks') == expected and report.get('active_queue_groups') == groups,
                'fresh CUDA group17 metadata coverage differs')
    identity = values['native_identity']
    require(identity.get('passed') is True and identity.get('identity_check_count') == 80, 'fresh16K native identity missing')
    parent = Path(refs['native_identity']['path']).parent
    phases = []
    for name, key in (('startup-qualified-native', 'reference_sha256'), ('startup-all-selected', 'candidate_sha256')):
        phase, item = read_reference(dict(path=str(parent/(name+'.json')), sha256=identity[key]))
        phases.append(phase);evidence[name] = item
    for phase in phases:
        validate_decode_phase(phase)
        require([v['baseline_sha256'] for v in phase['native_cache_storage_evidence']] ==
            evidence['native_cache_storage_sha256_by_rank'], 'native identity detached from graph cache addresses')
    require(identity.get('generation_budget') == 32 and identity.get('mtp5_decode_exercised') is True,
            'native identity must include actual MTP5 decoding')
    left, right = [p['controller'] for p in phases]
    expected_keys = {(a, a+8192, layer) for a in (0,8192) for layer in range(40)}
    for report in (left, right):
        rows = report.get('identity_fingerprints')
        require(type(rows) is list and len(rows) == 80 and {tuple(r['key']) for r in rows} == expected_keys
                and report.get('identity_fingerprints_sha256') == digest(rows), 'native full80 fingerprint coverage differs')
        for item in rows:
            value = item['value']
            require(set(value) == {'output', 'lazy'} and value['output'] is not None
                    and set(value['lazy']) == {'residual', 'post_layer_mix', 'comb_res_mix', 'pre_layer_mix'},
                    'native output/four-mHC fingerprints missing')
            _fingerprint(value['output'])
            for field in value['lazy'].values(): _fingerprint(field)
    require(right.get('identity_exact') is True and left['identity_fingerprints'] == right['identity_fingerprints']
            and phases[0]['record']['output_ids'] == phases[1]['record']['output_ids'], 'native numerical identity failed')
    budget = values['bank_budget']
    require(budget['bank_budget_bytes'] == uniform_bank_budget(budget['actual_free_bytes_by_rank']), 'uniform actual memory cap differs')
    return evidence, budget['bank_budget_bytes']


def _case_file(parent, case_id):
    path = parent/('session-case.'+case_id+'.json')
    require(path.is_file() and not path.is_symlink(), 'actual qualification case receipt missing')
    value = sealed(json.loads(path.read_bytes()))
    value, ref = read_reference(dict(path=str(path), sha256=value['sha256']))
    require(value.get('schema') == 's2-dspark-representative-actual-case-v1' and value.get('case_id') == case_id
            and value.get('actual_gpu_executed') is True and value.get('official_task') is False,
            'qualification case must be an actual diagnostic request')
    return value, ref


def _bind_qualified(actual, case):
    # Native member_chunks contains tuples in memory and JSON lists on disk;
    # canonical sealed encoding is the equality contract at this boundary.
    require(digest(actual.get('qualified')) == digest({k:v for k,v in case.items() if k != 'case_id'})
            and actual['input_ids_sha256'] == case['plan']['input_ids_sha256'],
            'numerical case detached from actual published request')


def _short_fallback(parent):
    value, ref = _case_file(parent, 'short_only')
    record = value['record'];prep = record['preparation']
    require(value.get('qualified') is None and record.get('live_engine') is True and record.get('replay') is False
            and prep.get('requested_arm') == 's2' and prep.get('effective_arm') == 'full'
            and prep.get('fallback_reason') == 'no_cacheable_document'
            and prep.get('numerical_gate') is None and prep.get('unique_misses') == 0 and prep.get('captures') == [],
            'short-only case did not exercise actual native Full fallback')
    plan, plan_ref = read_reference(prep['plan'])
    require(plan.get('route') == 'full_fallback' and plan.get('effective_documents') == []
            and plan.get('input_ids_sha256') == value['input_ids_sha256'] == record['input_ids_sha256'],
            'short-only plan/input ownership differs')
    formal, formal_ref = read_reference(record['completion']['formal_evidence'])
    require(record['completion'].get('completed_all8_rank_admission') is True
            and formal.get('request_id') == record['request_id'] and formal.get('batch_schedule') is None
            and len(formal.get('batch_worker_reports', [])) == 8
            and all(w == dict(schema='s1-system-worker-execution-v1', rank=rank, enabled=False, steps=[])
                    for rank, w in enumerate(formal['batch_worker_reports']))
            and len(formal.get('idle_reports', [])) == 8 and all(x is None for x in formal['idle_reports'])
            and formal['controller'].get('mode') == 'native', 'short-only native all-rank fallback differs')
    require(0 < len(record.get('output_ids', [])) <= 32 and record.get('finish_reason') in ('length','stop'),
            'short-only actual bounded generation did not close')
    return dict(case=ref, plan=plan_ref, formal=formal_ref, complete_quality_claim=False)


def issue_certificate(*, scope, session_id, source_sha256, startup_refs, case_evidence):
    validate_scope(scope);validate_sources(source_sha256)
    require(type(session_id) is str and session_id, 'actual unique service session ID required')
    startup, cap = _startup(startup_refs, session_id, scope)
    require([v['case_id'] for v in case_evidence] == ['natural_cold', 'cross_prefix_repeated', 'group17_partial'],
            'minimal deterministic natural qualification suite required')
    comparisons = [];plans = []
    parent = Path(startup_refs['startup_session']['path']).parent
    for case in case_evidence:
        actual, actual_ref = _case_file(parent, case['case_id'])
        _bind_qualified(actual, case)
        require(actual.get('redundant_formal_request_omitted') is True
            and actual.get('record') == case['candidate_phase']['record'],
            'qualification closure must bind the actual candidate record')
        plan = validate_plan(case['plan'], scope=scope, input_ids=case['input_ids'])
        require(plan['bank_budget_bytes'] == cap, 'qualification must use actual initialized bank budget')
        reference, candidate = case['reference_phase'], case['candidate_phase']
        verify_all_rank_phase(plan, reference, require_idle_exercised=False)
        verify_all_rank_phase(plan, candidate, require_idle_exercised=True)
        for phase in (reference,candidate):
            require([v['baseline_sha256'] for v in phase['native_cache_storage_evidence']] ==
                startup['native_cache_storage_sha256_by_rank'], 'sparse phase detached from initialized cache addresses')
        comparison = verify_sparse_pair(plan, case['input_ids'], reference['controller'], candidate['controller'],
                                        reference['record'], candidate['record'])
        comparisons.append(dict(case_id=case['case_id'], actual_case_ref=actual_ref, comparison=comparison,
            reference_phase_sha256=reference['sha256'], candidate_phase_sha256=candidate['sha256']))
        plans.append(plan)
    cold, repeated, long = plans
    require(bool(cold['short_documents']) and bool(repeated['resident_document_ids']), 'mixed-short and persistent-hit evidence required')
    common = set(cold['unique_document_ids']) & set(repeated['resident_document_ids'])
    require(any(sum(d['document_id'] == key for d in repeated['effective_documents']) >= 2 for key in common),
            'same resident document must be reused at distinct occurrences')
    require(cold['input_ids_sha256'] != repeated['input_ids_sha256'] and long['input_tokens'] > 17*8192,
            'different prompt and group17-plus-tail coverage required')
    require(any(any(any(e['world_phase'] == 'decode' for e in worker['world_step_trace'])
                    for worker in case['candidate_phase']['batch_worker_reports']) for case in case_evidence),
            'actual candidate decode execution required')
    geometry = dict(max_input_tokens=max(p['input_tokens'] for p in plans),
        max_document_occurrences=max(len(p['effective_documents']) for p in plans),
        max_document_tokens=max(d['tokens'] for p in plans for d in p['effective_documents']),
        max_group_selected_rows=max(n for p in plans for n in p['group_selected_rows']))
    body = dict(schema=SCHEMA, status='passed', scope=scope, session_id=session_id, source_sha256=source_sha256,
        startup_evidence=startup, numerical_cases=comparisons, short_fallback_evidence=_short_fallback(parent),
        supported_geometry=geometry, bank_budget_bytes=cap,
        fresh_gpu_representative_qualification=True, per_request_numeric_equivalence_claim=False,
        native_decode_profile='dspark', native_idle_bypass_installed=False,
        fixed32_or_early_eos_qualification=True, full_natural_eos_qualification=False,
        formal_agent_natural_eos_evaluated_separately=True,
        universal_input_equivalence_proven=False, mathematical_policy_changed=False,
        admission_policy='representative_session_with_structural_checks_and_explicit_out_of_domain_full_fallback',
        qualification_cost='setup_included_in_whole_experiment_wall_time_not_repeated_in_each_request')
    return dict(body, sha256=digest(body))


def validate_certificate(value, *, scope, session_id, source_sha256):
    sealed(value);validate_sources(source_sha256)
    require(value.get('schema') == SCHEMA and value.get('status') == 'passed'
            and value.get('scope') == scope and value.get('session_id') == session_id
            and value.get('source_sha256') == source_sha256 and value.get('fresh_gpu_representative_qualification') is True
            and value.get('per_request_numeric_equivalence_claim') is False
            and value.get('universal_input_equivalence_proven') is False
            and value.get('native_decode_profile') == 'dspark'
            and value.get('native_idle_bypass_installed') is False, 
            'session certificate scope/source/admission differs')
    geometry = value.get('supported_geometry')
    require(type(geometry) is dict and set(geometry) == set(GEOMETRY_KEYS)
            and all(type(n) is int and n > 0 for n in geometry.values()), 'explicit supported geometry upper bounds required')
    require(len(value.get('numerical_cases', [])) == 3 and value.get('startup_evidence'), 'session numerical evidence missing')
    require(value.get('short_fallback_evidence'), 'actual short-only fallback evidence missing')
    return value


def validate_installation_process(value):
    """A passed receipt from a previous process cannot admit a new controller."""
    owner, _ = read_reference(value['startup_evidence']['startup_session'])
    process = owner['rank_processes'][0]
    require(process['pid'] == os.getpid()
            and process['start_ticks'] == int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
            and process['argv'] == [p.decode() for p in Path('/proc/self/cmdline').read_bytes().split(b'\0') if p],
            'session admission belongs to another native controller process')
