"""Fresh Full/S1.2 reproduction on original natural 64K/128K questions.

GPU execution is launched only by an explicit user invocation of the package
entrypoint. The frozen model/policy/group17/idle implementation is unchanged;
all numerical, EOS and clean timing qualification is fresh within this run.
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
STAGE=HERE/'GatedEngram'
sys.path.insert(0,str(STAGE))
sys.path.insert(0,str(HERE))
from gpu_pilot import run_s11_pilot as parent
from gpu_pilot import run_pilot as support
from gpu_pilot import s11_protocol as inherited_protocol

SCHEMA='s12-reproduction-run-v1'
PHASE_SCHEMA=parent.PHASE_SCHEMA
ARMS=('recomputed','s1_system')
POLICY_ARMS=('s1_system',)
CANDIDATE_VARIANTS={'idle_group17':17}
BAND_INDICES={'64k':tuple(range(3,22,2)),'128k':tuple(range(2,21,2))}
BAND_LIMITS={'64k':(57344,73728),'128k':(114688,147456)}
DISPLAY_NAMES={'recomputed':'Full','s1_system':'S1.2'}
LIFECYCLE_TIMEOUT_SECONDS=7200
FROZEN_STAGE_SOURCES_SHA256='0321d533deceb984c126b07d24a099316cc7c8dcc769e470ccbcbc7095fa4d36'
EOS_TOKEN_ID=1
UPDATED_SYSTEM_SOURCES=('GatedEngram/kv_cache_engram/s1_system_batch.py',
    'GatedEngram/kv_cache_engram/s1_system_runtime.py')
ORIGINAL_INPUT_SHA256='e505b2cd56fce4727d8f6e65a6f264c30dcf9d32a6180c11b22707d46d518e30'
ORIGINAL_MANIFEST_SHA256='fb6bba62937fb8005f05df9a3a213729c172932a30e5509b136000692881aa67'
FROZEN_SYSTEM_SHA256={
    UPDATED_SYSTEM_SOURCES[0]:'28acd73b2d1c3b4b8bd31aef030fd60606a58b4a68d4160040a749c919c433b9',
    UPDATED_SYSTEM_SOURCES[1]:'ea566b45ea85b657bc347264bd81cd94248085b9e549b98abce728d908d825e2'}
FROZEN_EXTERNAL_SHA256={
    'idle_optimizer.py':'be34a00fde1262176d6eb69fa50289609343a26b6cf0b5d5fc6d9d4c203e58b3',
    'profile_group_contract.py':'855fbb8e8c02eb7eae9f935472778e53d72b97843e83f3ca0b0736e3074fde35'}
require=parent.require
write_json=parent.write_json
evidence_ref=parent.evidence_ref
case_scope=parent.case_scope
phase_policy=parent.phase_policy
quality_token_budget=parent.quality_token_budget
metadata_qualification_report=parent.metadata_qualification_report
qualification_row=parent.qualification_row
qualification_passed=parent.qualification_passed
document_jobs=parent.document_jobs
verify_shared_bank=parent.verify_shared_bank
verify_native_imports=parent.verify_native_imports


def digest(value):return parent._digest(value)


def read_sealed(path):
    value=json.loads(Path(path).read_bytes())
    require(type(value) is dict and value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),
        'sealed JSON mismatch: '+str(path))
    return value


def file_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root,config):
    """Revalidate every inherited pin, overriding only two explicitly frozen system modules."""
    pins=config['system_source_override_sha256']
    require(pins==FROZEN_SYSTEM_SHA256,'frozen S1.2 system implementation changed')
    require(inherited_protocol.DEPENDENCIES and all(k in inherited_protocol.DEPENDENCIES
        for k in inherited_protocol.REQUIRED_NEW_DEPENDENCIES),'inherited dependency pins missing')
    hashes=support.source_hashes(root)
    for name,previous in inherited_protocol.DEPENDENCIES.items():
        path=STAGE/name.removeprefix('GatedEngram/') if name.startswith('GatedEngram/') else Path(root)/name
        actual=file_digest(path)
        require(actual==pins.get(name,previous),'frozen dependency changed: '+name)
        hashes[name]=actual
    return hashes


def phase_execution(arm,*,batch_group_chunks=None):
    if arm=='s1_system':
        size=17 if batch_group_chunks is None else batch_group_chunks
        require(type(size) is int and size in (1,17),'S1.2 admits only qualified group17 or dense group1 oracle')
        return dict(controller_arm='s1',system_enabled=True,batch_group_chunks=size)
    return parent.phase_execution(arm,batch_group_chunks=batch_group_chunks)


def candidate_contract(variant):
    require(variant in CANDIDATE_VARIANTS,'unregistered candidate variant')
    from reproduction_contract import reproduction_contract
    return reproduction_contract()


def verify_phase(value,**kwargs):
    require(value.get('diagnostic_gpu_profile') is False,'profiler is forbidden in this comparison')
    variant=value.get('candidate_variant')
    if variant is not None:
        require(value.get('candidate_enabled') is True
            and value.get('batch_group_chunks')==CANDIDATE_VARIANTS.get(variant),
            'candidate variant and actual batch group differ')
        require(value.get('arm')=='s1_system' and value.get('method_name')=='S1.2',
            'S1.2 method identity differs')
        return candidate_contract(variant).verify_phase(value,**kwargs)
    require(value.get('candidate_enabled') is not True,'candidate phase lost its variant identity')
    if value.get('arm')=='s1_system':
        require(value.get('mode')=='reference_sparse' and value.get('batch_group_chunks')==1
            and value.get('qualification_only') is True and not kwargs.get('timing') and not kwargs.get('quality'),
            'retired S1.1 formal phase is forbidden')
        return candidate_contract('idle_group17').verify_phase(value,**kwargs)
    return parent.verify_phase(value,**kwargs)


def selected_rows(manifest,*,length,cases_per_length):
    require(manifest.get('sha256')==ORIGINAL_MANIFEST_SHA256,'original input manifest identity changed')
    require(length in ('64k','128k','both') and type(cases_per_length) is int
        and 1<=cases_per_length<=10,'length and cases-per-length selection must be explicit')
    cohort=manifest.get('cohort')
    require(type(cohort) is list and len({row['index'] for row in cohort})==len(cohort),'duplicate original case index')
    by_index={row['index']:row for row in cohort}
    selected=[]
    for band in (('64k','128k') if length=='both' else (length,)):
        require(tuple(sorted(row['index'] for row in cohort if row.get('length_band')==band))==BAND_INDICES[band],
            'original length cohort changed')
        lo,hi=BAND_LIMITS[band]
        for index in BAND_INDICES[band][:cases_per_length]:
            row=by_index[index]
            require(row['length_band']==band and type(row['input_tokens']) is int and lo<=row['input_tokens']<=hi
                and len(row['input_ids'])==row['input_tokens']
                and row['input_ids_sha256']==digest(row['input_ids']),'original selected token input changed')
            parent.document_lengths(row)
            selected.append(row)
    require(len({row['input_ids_sha256'] for row in selected})==len(selected),'duplicate selected token input')
    return selected


def case_pin(row):
    fields=('index','sample_id','source_id','source_index','source_row_sha256','source_qa_sha256',
        'context_sha256','input_ids_sha256','input_tokens','length_band','raw_answer_sha256')
    return dict({key:row[key] for key in fields},documents_sha256=digest(row['documents']))


def build_config(manifest_path,pilot_protocol_path,*,length='128k',cases_per_length=1,candidate_enabled=True):
    """CPU-only builder; the accepted S1.2 implementation cannot be disabled."""
    from gpu_pilot.launch_pilot import source_files
    from gpu_pilot.launch_index_hit20_pilot import snapshot_sources
    manifest=read_sealed(manifest_path);inputs=read_sealed(pilot_protocol_path)
    require(candidate_enabled is True,'S1.2 cannot be replaced by retired S1.1')
    rows=selected_rows(manifest,length=length,cases_per_length=cases_per_length)
    actual_system={name:file_digest(HERE/name) for name in UPDATED_SYSTEM_SOURCES}
    require(actual_system==FROZEN_SYSTEM_SHA256,'accepted group17 implementation bytes changed')
    require({name:file_digest(HERE/name) for name in FROZEN_EXTERNAL_SHA256}==FROZEN_EXTERNAL_SHA256,
        'accepted idle/contract implementation bytes changed')
    stage_sha256=digest(source_files(STAGE))
    require(stage_sha256==FROZEN_STAGE_SOURCES_SHA256,'accepted GatedEngram tree bytes changed')
    names=['run_s12_reproduction.py','reproduction_contract.py',*FROZEN_EXTERNAL_SHA256]
    body=dict(schema='s12-reproduction-config-v1',length=length,cases_per_length=cases_per_length,
        evaluation_case_indices=[row['index'] for row in rows],
        case_pins=[case_pin(row) for row in rows],manifest_sha256=manifest['sha256'],input_protocol_sha256=inputs['sha256'],
        stage_sources_sha256=stage_sha256,native_sources_sha256=digest(snapshot_sources(HERE/'NativeSnapshot')),
        external_source_sha256={name:file_digest(HERE/name) for name in names},
        system_source_override_sha256=actual_system,
        arms=list(ARMS),display_names=DISPLAY_NAMES,method_name='S1.2',
        internal_policy_arm='s1_system',formal_s11_retired=True,
        clean_quality_natural_eos=True,clean_warmups_per_arm=1,clean_timing_repeats=3,clean_timing_output_tokens=32,
        profiler_installed=False,candidate_enabled=True,candidate_name='idle_group17',
        candidate_variants=CANDIDATE_VARIANTS,lifecycle_timeout_seconds=LIFECYCLE_TIMEOUT_SECONDS,
        timeout_origin='actual_gpu_release_of_this_explicit_reproduction',
        group17_metadata_qualification_required=True,
        per_case_same_policy_qualification_token_budget=32,qualification_eos_enabled=True,eos_token_id=EOS_TOKEN_ID,
        qualification_complete_ids_text_stop_reason_exact=True,full_model_output_parity_required=False,
        qualification='fresh_native16K_and_each_case_dense_group1_vs_S1.2_group17',
        selection='first_n_original_rows_per_length_band_64k_before_128k',
        no_padding=True,no_truncation=True,global_timing_ordinal_offset=0,
        case_publication_barrier_before_all_rank_reference=True,
        historical_64k_gpu_measurements_used=False,
        selection_uses_model_outcomes=False,historical_outputs_used_for_admission=False,
        other_evaluation_cases_executed=False,original_s1_executed=False,
        core_source_sha256=inherited_protocol.CORE_SOURCE_SHA256,core_policy_sha256=inherited_protocol.CORE_POLICY_SHA256)
    return dict(body,sha256=digest(body))


def validate_config(args,manifest):
    value=read_sealed(args.profile_config)
    require(type(value.get('candidate_enabled')) is bool,'candidate opt-in must be explicit')
    require(value==build_config(args.manifest,args.pilot_protocol,length=value['length'],
        cases_per_length=value['cases_per_length'],candidate_enabled=value['candidate_enabled']),
        'profile config/input/source contract differs')
    return value


def verify_frozen(args,manifest,identity,profile,base,*,native_source=False):
    require(validate_config(args,manifest)['sha256']==identity['profile_config_sha256'],'profile config changed')
    require(source_hashes(args.experiment_root,args.profile_config_value)==identity['pilot_source_sha256'],'frozen parent source changed')
    require(profile.identity(args)==identity['native_profile'] and base.effective_env(args)==identity['fixed_env'],
        'native execution configuration drift')
    require(parent.native_yoco_swa_configuration(args.repo,args.model_path)==identity['native_yoco_swa_configuration'],
        'native YOCO source/config changed')
    for name,key in (('config.json','checkpoint_config_sha256'),('tokenizer.json','tokenizer_json_sha256'),
                     ('model.safetensors.index.json','checkpoint_index_sha256')):
        require(base.file_digest(args.model_path/name)==identity[key],'model/tokenizer changed')
    if native_source:
        require(all(identity.get(k)==v for k,v in parent.frozen_core_identity(args.repo,inherited_protocol.s11_protocol()).items()),
            'native/core changed')
        verify_native_imports(args.repo)
        require(all(identity[k]==v for k,v in base.source_identity(args.repo).items()),'native source identity changed')


def install_candidate(model,layers,controller,engine,rank,torch,allow_sync,batch_worker,config):
    require(config['candidate_enabled'],'unregistered optimizer phase')
    from idle_optimizer import Candidate
    return Candidate(model,controller,engine,rank,batch_worker,torch,allow_sync).install()


def instrumentation_report(instrumentation,*,candidate):
    """Collect only after the phase finally-block has restored its hook owner."""
    if instrumentation is None:
        return None
    return instrumentation.report() if candidate else instrumentation.finish()


def validate_candidate_receipts(reports,schedule,worker_reports,*,name,rid):
    from idle_optimizer import IDLE_METADATA_CONTRACT
    require(type(reports) is list and len(reports)==8 and len(worker_reports)==8
        and schedule is not None,'candidate requires eight admitted worker reports')
    for rank,(report,worker) in enumerate(zip(reports,worker_reports)):
        require(type(report) is dict and report.get('sha256')==digest({k:v for k,v in report.items() if k!='sha256'})
            and report.get('schema')=='s12-world-idle-forward-candidate-v1'
            and report.get('rank')==rank and report.get('enabled') is True
            and report.get('phase')==name and report.get('request_id')==rid,
            'candidate rank/phase/enable receipt differs')
        require(worker.get('rank')==rank and worker.get('enabled') is True
            and worker.get('schedule_sha256')==schedule['sha256'],'candidate worker schedule differs')
        idle=[event for event in worker.get('world_step_trace',[]) if event['world_phase']=='idle']
        events=report.get('events')
        require(type(events) is list and len(events)==len(idle)>0
            and report.get('bypassed_world_idle_steps')==len(events),
            'candidate must exercise every actual world-idle callback')
        for event,world in zip(events,idle):
            require(event.get('ordinal')==world['ordinal'] and event.get('world_phase')=='idle'
                and event.get('scheduler_observations_sha256')==world['scheduler_observations_sha256']
                and event.get('schedule_sha256')==schedule['sha256']
                and event.get('phase')==name and event.get('request_id')==rid
                and event.get('idle_forward_skipped') is True and event.get('native_decoder_calls')==0
                and event.get('real_query_rows')==0
                and event.get('worker_forward_postprocess_sample_round_preserved') is True
                and event.get('native_decode_metadata')==IDLE_METADATA_CONTRACT
                and event.get('total_q_is_prefill_only_default') is True,
                'candidate bypass event detached from actual world scheduler')
    return True


def require_natural_eos(value):
    record=verify_phase(value,index=value['index'],arm=value['arm'],timing=False,quality=True)
    require(record.get('finish_reason')=='stop' and record.get('quality_output_complete') is True
        and record.get('truncated') is False,'quality did not reach natural EOS')
    return record


def group17_metadata_report(reports,identity):
    require(type(reports) is list and len(reports)==8,'group17 metadata gate needs all eight ranks')
    checks=[dict(name='inactive_empty_cu',passed=True,rows=0,cu=[0,0]),
        dict(name='active_queue_group17',passed=True,input_tokens=139265,member_chunks=18,
            deep_groups=2,empty_groups=0,selected_rows=128)]
    groups=[dict(group_chunks=17,start=0,end=139264,selected_rows=127,empty=False,
        sampler_output_rows=139,exact_metadata=True,strict_sync_debug_mode=2),
        dict(group_chunks=17,start=139264,end=139265,selected_rows=1,empty=False,
        sampler_output_rows=1,exact_metadata=True,strict_sync_debug_mode=2)]
    for rank,item in enumerate(reports):
        require(type(item) is dict and set(item)=={'rank','operator_report'} and item['rank']==rank,
            'group17 metadata rank identity differs')
        report=item['operator_report']
        require(report.get('sha256')==digest({k:v for k,v in report.items() if k!='sha256'})
            and report.get('schema')=='s1-system-cuda-metadata-qualification-v1'
            and report.get('passed') is True and report.get('cuda_exercised') is True
            and report.get('device_type')=='cuda' and report.get('sync_debug_mode_during')==2
            and report.get('gpu_numerical_parity_claim') is False
            and report.get('requested_batch_groups')==[17]
            and report.get('checks')==checks and report.get('active_queue_groups')==groups,
            'group17 strict CUDA metadata coverage differs')
        before,after=report.get('sync_debug_mode_before'),report.get('sync_debug_mode_after')
        require(type(before) is int and before in (0,1,2) and type(after) is int and before==after,
            'group17 metadata gate changed the CUDA guard')
    body=dict(schema='s12-group17-all-rank-metadata-v1',passed=True,
        execution_configuration_sha256=identity['execution_configuration_sha256'],
        model_sha256=identity['checkpoint_index_sha256'],reports=reports,
        operator_source_sha256={name:identity['pilot_source_sha256'][name] for name in UPDATED_SYSTEM_SOURCES},
        quality_eligible=False,timing_valid=False,gpu_numerical_parity_claim=False)
    return dict(body,sha256=digest(body))


def timing_schedule(index,evaluation_case_indices):
    require(index in evaluation_case_indices,'unregistered comparison case')
    ordinal=evaluation_case_indices.index(index)
    return [(repeat,ARMS[(ordinal+repeat+slot)%2]) for repeat in range(3) for slot in range(2)]


def summarize_timing(phases,index,evaluation_case_indices):
    require([(p['repeat'],p['arm']) for p in phases]==timing_schedule(index,evaluation_case_indices),'paired three-repeat order differs')
    summaries={};bindings=set()
    for arm in ARMS:
        records=[]
        for phase in (p for p in phases if p['arm']==arm):
            records.append(verify_phase(phase,index=index,arm=arm,timing=True,quality=False))
            bindings.add(tuple(phase[k] for k in ('input_ids_sha256','execution_configuration_sha256',
                'manifest_sha256','s1_protocol_sha256')))
        samples=[r['online_ttft_s'] for r in records]
        summaries[arm]=dict(online_ttft_samples_s=samples,online_ttft_p50_s=statistics.median(samples),
            repetitions=3,fixed_output_tokens=32,display_name=DISPLAY_NAMES[arm])
    require(len(bindings)==1,'paired timing execution/input differs')
    by_repeat={(p['repeat'],p['arm']):p['record']['online_ttft_s'] for p in phases}
    return dict(arms=summaries,comparisons={'s1_system':dict(
        ttft_speedup=summaries['recomputed']['online_ttft_p50_s']/summaries['s1_system']['online_ttft_p50_s'],
        paired_repeat_speedups=[by_repeat[(r,'recomputed')]/by_repeat[(r,'s1_system')] for r in range(3)])},
        concurrency=1,active_dp_ranks=1,participating_ep_ranks=8,online_full_bank_gpu_resident=True,
        online_gather_scatter_and_fallback_included=True,model_load_included=False,offline_build_included=False,
        saturated_capacity_claim=False,fresh_paired_baseline=True,historical_results_used_for_admission=False)


def qualification_record(phase):
    require(phase.get('qualification_only') is True and phase.get('qualification_budget') is True
        and phase.get('eos_enabled') is True and phase.get('generation_token_budget')==32
        and phase.get('quality_eligible') is False and phase.get('timing_valid') is False,
        'same-policy reference is bounded qualification only')
    record=verify_phase(phase,index=phase['index'],arm='s1_system',timing=False,quality=False)
    ids=record.get('output_ids');reason=record.get('finish_reason')
    require(type(ids) is list and 1<=len(ids)<=32 and all(type(x) is int and x>=0 for x in ids)
        and record.get('natural_EOS') is True and type(record.get('text')) is str
        and reason in ('stop','length') and (reason!='length' or len(ids)==32)
        and record.get('truncated') is (reason=='length'),'invalid bounded natural-EOS qualification output')
    require((reason=='stop' and ids[-1]==EOS_TOKEN_ID and ids.count(EOS_TOKEN_ID)==1)
        or (reason=='length' and EOS_TOKEN_ID not in ids),'qualification EOS/stop reason differs')
    return record


def sparse_identity32(reference,candidate,row,qualification_sha256):
    # Adapt only the generation-count contract; preserve actual raw phases,
    # full IDs, all inherited cache/source2 audits and exact per-member tensors.
    from gpu_pilot import read_s11_results as audit
    contract=candidate_contract('idle_group17')
    audit.validate_digest(qualification_sha256,'fresh native qualification seal')
    records=[qualification_record(p) for p in (reference,candidate)]
    require(reference['mode']=='reference_sparse' and reference['batch_group_chunks']==1
        and reference['candidate_enabled'] is False and reference['verify_sparse'] is False,
        'fresh oracle must be dense group1 without the candidate')
    require(candidate['mode']=='sensitive' and candidate['batch_group_chunks']==17
        and candidate['candidate_enabled'] is True and candidate['verify_sparse'] is True,
        'candidate must exercise S1.2 group17 and idle bypass')
    chunks=contract.chunk_oracle(contract.documents(row),row['input_tokens'])
    expected={(c['chunk_start'],c['chunk_end'],layer):c for c in chunks for layer in range(40)}
    banks=[]
    for phase in (reference,candidate):
        audit._fields(phase,dict(schema=PHASE_SCHEMA,index=row['index'],arm='s1_system',fraction=None,
            input_ids_sha256=row['input_ids_sha256'],diagnostic=False,natural_boundary_inputs=True,
            s1_policy=True,native_yoco_swa_share_hidden=False,**phase_policy('s1_system')),
            'same-policy raw qualification scope')
        audit._fields(phase['record'],dict(success=True,rank=0,input_ids_sha256=row['input_ids_sha256'],
            input_tokens=row['input_tokens'],sample_id=row['sample_id'],dataset=row['dataset'],
            phase=phase['name'],request_id=f"s1:{row['index']}:{phase['name']}:{row['sample_id']}"),
            'same-policy raw qualification request identity')
        control=phase['controller']
        contract.audit_controller(control,row,phase,contract.documents(row))
        banks.append(contract.comparison_scope(control,phase,spans=contract.documents(row)))
        values=control.get('sparse_identity_fingerprints')
        require(type(values) is list and len(values)==len(expected)
            and control.get('sparse_identity_check_count')==len(expected)
            and control.get('sparse_identity_fingerprints_sha256')==digest(values),
            'full-member 40-layer fingerprints missing')
        seen=set()
        for item in values:
            key=tuple(item.get('key',[]));value=item.get('value',{})
            require(key in expected and key not in seen and set(value)=={'output','lazy'}
                and set(value['lazy'])=={'residual','post_layer_mix','comb_res_mix','pre_layer_mix'},
                'output/all-four-mHC member coverage differs')
            seen.add(key);require(value['output'] is not None,'selected decoder output missing')
            audit._fingerprint(value['output'])
            n=key[1]-key[0] if key[2]<3 else len(expected[key]['selected_positions'])
            require(value['output']['shape'][0]==n,'missing/extra selected output rows')
            for field in value['lazy'].values():
                audit._fingerprint(field)
                require(field is None or len(field['shape'])>=2 and field['shape'][:2]==[1,n],
                    'mHC selected token axis differs')
    for key in ('manifest_sha256','execution_configuration_sha256','s1_protocol_sha256',
        'empty_ep_qualification_sha256','runtime_qualification_sha256',
        'system_metadata_qualification_sha256','group17_metadata_qualification_sha256'):
        require(reference.get(key)==candidate.get(key),'dense/candidate frozen execution differs')
    require(banks[0]==banks[1] and reference['controller']['phase_generation']!=candidate['controller']['phase_generation'],
        'dense/candidate reused phase state or changed bank')
    require(candidate['controller'].get('sparse_identity_exact') is True
        and reference['controller']['sparse_identity_fingerprints']==candidate['controller']['sparse_identity_fingerprints'],
        'selected output/all-four-mHC exact byte parity failed')
    require(all(records[0][key]==records[1][key] for key in ('output_ids','output_ids_sha256','text','finish_reason')),
        'same-policy complete bounded output/stop reason differs')
    body=dict(schema='s12-multi-sparse-qualification-v1',passed=True,index=row['index'],
        method_name='S1.2',arm='s1_system',batch_group_chunks=17,system_enabled=True,
        input_tokens=row['input_tokens'],input_ids_sha256=row['input_ids_sha256'],
        qualification_sha256=qualification_sha256,reference_sha256=reference['sha256'],candidate_sha256=candidate['sha256'],
        identity_check_count=len(expected),identity_fingerprints_sha256=candidate['controller']['sparse_identity_fingerprints_sha256'],
        quality_eligible=False,timing_valid=False,exact_same_policy_reference=True,equivalence_to_full_model_claim=False,
        generation_token_budget=32,eos_enabled=True,generated_tokens=len(records[0]['output_ids']),
        output_ids_sha256=records[0]['output_ids_sha256'],output_text_sha256=hashlib.sha256(records[0]['text'].encode()).hexdigest(),
        finish_reason=records[0]['finish_reason'],complete_generated_ids_text_stop_reason_equal=True,
        old_qualification_reused=False,shared_bank_identity=banks[0],
        empty_chunk_exercised=any(not c['selected_positions'] for c in chunks))
    return dict(body,sha256=digest(body))


def verify_output_prefix(reference,quality,warmup,measurements):
    ref=qualification_record(reference);prefix=ref['output_ids']
    values=[quality,warmup,*measurements]
    require(all(p['record']['output_ids'][:len(prefix)]==prefix for p in values),
        'formal S1.2 output prefix differs from fresh same-policy reference')
    if ref['finish_reason']=='stop':
        require(quality['record']['output_ids']==prefix and quality['record']['text']==ref['text'],
            'formal S1.2 complete early-EOS answer differs from fresh reference')
    require(all(p['record']['output_ids']==warmup['record']['output_ids'] for p in measurements),
        'S1.2 fixed32 repetitions differ')
    return True


def worker(args,manifest,identity,dp_size,rank):
    parent.bind_native_snapshot(args)
    base,native,NativeProfile,scoped_base_profile=support.load_runtime(args.experiment_root)
    profile=NativeProfile(mtp_tokens=0,keep_tokens=140)
    with scoped_base_profile(base,profile,args=args,apply_environment=True):
        require(not native.gpu_runtime_imported(),'GPU imported before guarded worker')
        verify_frozen(args,manifest,identity,profile,base)
        return worker_impl(args,manifest,identity,dp_size,rank,profile,base,native)


def worker_impl(args, manifest, identity, dp_size, rank, profile, base, native):
    sys.path.insert(0, str(args.repo/"examples")); sys.path.insert(0, str(args.repo))
    llm = controller = binder = batch_worker = None
    completed = []
    try:
        from _config import create_llm, setup_common_env
        setup_common_env(model_name=str(args.model_path), sampler_force_topk=args.sampler_force_topk,
                         set_override_experts=False, enable_eplb=False)
        started = time.perf_counter()
        llm = create_llm(str(args.model_path), rank, ep_size=8, cp_size=1,
            env={"DEEPINFER_LOGGING_INTERVAL_STEP": "0"}, data_parallel_size=8,
            max_num_batched_tokens=8192, max_model_len=270336, max_num_seqs=1,
            gpu_memory_utilization=.60, **profile.llm_kwargs())
        init_s = time.perf_counter()-started
        verify_native_imports(args.repo)
        import torch
        from deepinfer import envs
        from deepinfer.utils import allow_cuda_sync
        from canonical_request_binding import install_request_binding
        from kv_cache_engram.s1_system_runtime import install_worker_batch_schedule
        from kv_cache_engram.s11_qualification import S11QualificationController
        torch.backends.cuda.matmul.allow_tf32 = False

        class Driver(base.NativeDriver):
            request_binding = None
            phase = "native"

            def submit(self, rid, ids):
                require(self.request_binding is None, "overlapping request state")
                self.request_binding = dict(request_id=rid, phase=self.phase, token_ids=list(ids))
                return super().submit(rid, ids)

            def finish(self, rid):
                super().finish(rid)
                require(self.request_binding is not None and self.request_binding["request_id"] == rid,
                        "request completion ownership mismatch")
                self.request_binding = None

        driver = Driver(llm, args, diagnostic_rank=rank)
        model_runner = driver.engine.driver_worker.model_runner
        runtime, validation = native.actual_runtime(profile, args, model_runner, torch, envs,
                                                    cache_config=driver.engine.cache_config)
        write_json(args.out_dir/f"native-runtime.rank{rank}.json", base.seal(dict(observed=runtime, validation=validation)))
        from gpu_pilot.cuda_mask_qualification import run_cuda_mask_qualification, validate_report
        mask_reports = driver.gather(run_cuda_mask_qualification(torch, allow_cuda_sync))
        require(len(mask_reports) == 8, "all eight ranks require strict CUDA mask qualification")
        for report in mask_reports:
            validate_report(report)
        cuda_mask_qualification = base.seal(dict(
            schema="ffn-only-cuda-mask-all-ranks-v1", passed=True,
            execution_configuration_sha256=identity["execution_configuration_sha256"],
            reports=[dict(rank=i, report=report) for i, report in enumerate(mask_reports)],
            quality_eligible=False, timing_valid=False))
        if rank == 0:
            write_json(args.out_dir/"cuda_mask_qualification.json", cuda_mask_qualification)
            print(json.dumps(dict(phase="cuda_mask_qualification", status="complete", ranks=8)), flush=True)
        driver.barrier()
        layers = list(model_runner.model.model.layers)
        require([m.layer_idx for m in layers] == list(range(40)) and dp_size == 8, "native40 layer DP8 geometry required")
        from gpu_pilot.index_hit20_empty_ep_qualification import (
            run_empty_ep_qualification, build_all_ranks_report, EmptyEPCapability,
        )
        empty_ep_reports = driver.gather(run_empty_ep_qualification(
            layers, rank, torch, allow_cuda_sync,
            execution_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"]))
        empty_ep_qualification = build_all_ranks_report(empty_ep_reports,
            execution_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"])
        if rank == 0:
            write_json(args.out_dir/"empty_ep_qualification.json", empty_ep_qualification)
        driver.barrier()
        from kv_cache_engram.s1_runtime import run_s1_runtime_qualification
        from gpu_pilot.read_s11_results import validate_runtime_qualification
        runtime_reports=driver.gather(run_s1_runtime_qualification(layers,rank,torch,allow_cuda_sync,
            execution_sha256=identity["execution_configuration_sha256"],model_sha256=identity["checkpoint_index_sha256"]))
        runtime_qualification=base.seal(dict(schema="s1-runtime-all-ranks-qualification-v1",passed=True,
            execution_configuration_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"],reports=runtime_reports,
            empty_ep_qualification_sha256=empty_ep_qualification["sha256"],quality_eligible=False,timing_valid=False))
        validate_runtime_qualification(runtime_qualification,identity,empty_ep_qualification["sha256"])
        if rank==0:write_json(args.out_dir/"s1_runtime_qualification.json",runtime_qualification)
        driver.barrier()
        from kv_cache_engram.s1_system_batch import run_s1_system_metadata_qualification
        metadata_reports = driver.gather(dict(rank=rank, operator_report=
            run_s1_system_metadata_qualification(torch, allow_cuda_sync, device="cuda")))
        metadata_qualification = metadata_qualification_report(metadata_reports, identity)
        if rank == 0:
            write_json(args.out_dir/"s1_system_metadata_qualification.json", metadata_qualification)
            print(json.dumps(dict(phase="s1_system_metadata_qualification", status="complete", ranks=8)), flush=True)
        driver.barrier()
        metadata17_reports=driver.gather(dict(rank=rank,operator_report=
            run_s1_system_metadata_qualification(torch,allow_cuda_sync,device='cuda',group_chunks=(17,))))
        metadata17_qualification=group17_metadata_report(metadata17_reports,identity)
        if rank==0:
            write_json(args.out_dir/'s12_group17_metadata_qualification.json',metadata17_qualification)
            print(json.dumps(dict(phase='s12_group17_metadata_qualification',status='complete',ranks=8)),flush=True)
        driver.barrier()
        binder = install_request_binding(model_runner, lambda: driver.request_binding if rank == 0 else None)
        empty_ep_capability = EmptyEPCapability(layers, empty_ep_qualification,
            execution_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"])
        if rank == 0:
            controller = S11QualificationController(layers, allow_sync=allow_cuda_sync,
                native_model=model_runner.model.model, expected_document_count=10,
                model_sha256=identity["checkpoint_index_sha256"], tokenizer_sha256=identity["tokenizer_json_sha256"],
                execution_sha256=identity["execution_configuration_sha256"], dependency_root=args.experiment_root,
                empty_ep_capability=empty_ep_capability)
            controller.install()
        batch_worker = install_worker_batch_schedule(model_runner.model.model, rank=rank,
            torch_module=torch, empty_ep_capability=empty_ep_capability, controller=controller,
            engine=driver.engine, phase_gather=driver.gather)
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]

        def phase(row, name, mode, *, arm, quality=False, tokens=None, timing=False, repeat=None, fraction=None,
                  partial_smoke=False, verify_sparse=False, diagnostic=False, batch_group_chunks=None, candidate=None,
                  candidate_variant=None, qualification_budget=False):
            require(diagnostic is False,'profiling is not part of this two-arm comparison')
            should_enable=arm=='s1_system' and mode=='sensitive'
            if candidate is None:candidate=should_enable
            require(type(candidate) is bool and candidate is should_enable,'S1.2 idle bypass cannot silently fallback')
            if candidate:
                require(candidate_variant in (None,'idle_group17'),'unregistered candidate variant')
                candidate_variant='idle_group17'
                if batch_group_chunks is None:batch_group_chunks=17
            require((candidate and batch_group_chunks==17) or (not candidate and candidate_variant is None),
                'S1.2 candidate group registration differs')
            require(not qualification_budget or (not timing and not quality and arm=='s1_system'
                and tokens==32 and (mode=='reference_sparse' or verify_sparse)),
                'bounded generation is numerical qualification only')
            # Per-request candidate installation/capability checks are online
            # preparation and therefore included in the candidate TTFT clock.
            began = time.perf_counter()
            instrumentation=None
            if candidate:
                instrumentation=install_candidate(model_runner.model.model,layers,controller,driver.engine,rank,torch,
                    allow_cuda_sync,batch_worker,args.profile_config_value)
            execution = phase_execution(arm, batch_group_chunks=batch_group_chunks)
            required_group=CANDIDATE_VARIANTS[candidate_variant] if candidate else None
            require(not (timing or quality) or execution == phase_execution(arm,batch_group_chunks=required_group),
                "formal system execution cannot fallback")
            if quality:
                require(tokens is None, "quality cannot receive an artificial output-token cap")
                tokens = quality_token_budget(row["input_tokens"], args.max_model_len)
            elif tokens is None:
                tokens = 1
            args.mode, args.max_new_tokens = ("quality" if quality or qualification_budget else "throughput"), tokens
            driver.phase = name
            rid = f"s1:{row.get('index', 'doc')}:{name}:{row['sample_id']}"
            try:
                if controller is not None:
                    # Fresh phase state, full-bank lookup and online memory preparation
                    # belong to TTFT: clock starts before this reset.
                    index_device = layers[0].self_attn.q_a_kv_swa_proj.weight.device if mode in ("sensitive", "reference_sparse", "all_selected") else None
                    with allow_cuda_sync():
                        controller.configure(system_enabled=execution["system_enabled"], batch_group_chunks=execution["batch_group_chunks"])
                        special = dict(system_empty_group_qualification=True,
                            source_full_input_ids_sha256=row['source_full_input_ids_sha256']) if row.get('system_empty_group_qualification') else {}
                        controller.reset(rid, mode, arm=execution["controller_arm"], input_ids=row["input_ids"], documents=row.get("documents"),
                                         phase=name, fraction=fraction, index_device=index_device,
                                         verify_sparse=verify_sparse,**special)
                schedule = driver.gather(controller.batch_schedule if controller is not None else None)[0]
                batch_receipts = driver.gather(batch_worker.configure(schedule))
                if schedule is not None and controller is not None:
                    controller.bind_batch_schedule(schedule, batch_receipts)
                if candidate:
                    instrumentation.configure(enabled=True,phase=name,request_id=rid)
                driver.barrier()
                if rank == 0:
                    print(json.dumps(dict(case=row.get("index"), phase=name, status="starting", mode=mode, arm=arm)), flush=True)
                records, window = base.run_phase(driver, [(rid, row)] if rank == 0 else [],
                                                 int(rank == 0), rank, epoch, tokens, lambda _: None)
                batch_worker_reports = driver.gather(batch_worker.finish_phase())
                report = None
                if controller is not None:
                    with allow_cuda_sync():
                        report = controller.finish()
                require(len(records) == 1 and records[0].rank == 0 and records[0].success(args.mode),
                        "phase lacks exactly one complete native request")
                rec = base.canonical_record(records[0], driver.engine.tokenizer, args.mode, name)
                require(quality or qualification_budget or len(rec["output_ids"]) == tokens, "fixed generation length mismatch")
                start = driver.gather(began-epoch)[0]
                rec.update(online_prepare_to_submit_s=rec["submit_s"]-start,
                    online_ttft_s=rec["submit_s"]+rec["ttft_s"]-start,
                    online_elapsed_s=rec["finish_s"]-start, natural_EOS=bool(quality or qualification_budget),
                    truncated=bool((quality or qualification_budget) and rec.get("finish_reason") == "length"))
                quality_scope = {}
                if quality:
                    quality_scope = dict(quality_requested_max_new_tokens=None, technical_max_new_tokens=tokens)
                    rec.update(technical_max_new_tokens=tokens,
                        quality_output_complete=rec.get("finish_reason") == "stop",
                        quality_limit_reason="engine_context_limit" if rec.get("finish_reason") == "length" else None)
            finally:
                if instrumentation is not None:instrumentation.restore()
            observed_instrumentation=None
            if instrumentation is not None:
                with allow_cuda_sync():
                    observed_instrumentation=instrumentation_report(instrumentation,candidate=candidate)
            instrumentation_reports=driver.gather(observed_instrumentation)
            if candidate:
                validate_candidate_receipts(instrumentation_reports,schedule,batch_worker_reports,name=name,rid=rid)
            reports = driver.gather(report)
            require(isinstance(reports[0], dict) and all(p is None for p in reports[1:]),
                    "only rank0 may own natural-boundary state")
            require((not diagnostic and not candidate) == all(v is None for v in instrumentation_reports),
                'instrumentation must be absent from clean phases and present on all instrumented ranks')
            result = base.seal(dict(schema=PHASE_SCHEMA, name=name, mode=mode, arm=arm, repeat=repeat,
                fraction=fraction, record=rec, controller=reports[0], timing_valid=timing, quality_eligible=quality,
                diagnostic_gpu_profile=diagnostic, candidate_enabled=candidate,
                candidate_variant=candidate_variant,
                method_name='S1.2' if candidate else ('same_policy_dense_reference' if mode=='reference_sparse' else DISPLAY_NAMES.get(arm,arm)),
                display_name=DISPLAY_NAMES.get(arm,arm),
                qualification_only=mode in ('qualified_native','all_selected','reference_sparse') or verify_sparse,
                qualification_budget=qualification_budget,eos_enabled=bool(quality or qualification_budget),
                generation_token_budget=tokens,
                profile_config_sha256=identity['profile_config_sha256'],
                instrumentation_reports=instrumentation_reports,
                **execution,
                system_batch_schedule=schedule, system_batch_configuration_receipts=batch_receipts,
                system_batch_worker_reports=batch_worker_reports,
                system_empty_group_qualification=row.get('system_empty_group_qualification',False),
                source_original_case_index=row.get('source_original_case_index'),
                source_full_input_ids_sha256=row.get('source_full_input_ids_sha256'),
                **quality_scope,
                partial_smoke=partial_smoke, diagnostic=diagnostic, natural_boundary_inputs=True,
                verify_sparse=verify_sparse, s1_policy=(arm in POLICY_ARMS),
                native_yoco_swa_share_hidden=identity["native_yoco_swa_share_hidden"],
                **phase_policy(arm),
                online_global_window_s=window["end_s"]-start, index=row.get("index"), dataset=row["dataset"],
                **case_scope(row),
                input_ids_sha256=row["input_ids_sha256"], manifest_sha256=manifest["sha256"],
                execution_configuration_sha256=identity["execution_configuration_sha256"],
                empty_ep_qualification_sha256=empty_ep_qualification["sha256"],
                runtime_qualification_sha256=runtime_qualification["sha256"],
                system_metadata_qualification_sha256=metadata_qualification["sha256"],
                group17_metadata_qualification_sha256=metadata17_qualification["sha256"],
                s1_protocol_sha256=identity["s1_protocol_sha256"],
                online_full_bank_gpu_resident=(arm in ARMS), online_clock_started_before_reset=True,
                phase_local_memory_rebuilt=True, offline_build_included=False,
                native_first3_source2_preserved=True))
            if rank == 0:
                write_json(args.out_dir/f"{name}.json", result)
                print(json.dumps(dict(case=row.get("index"), phase=name, status="complete", arm=arm,
                    em=rec.get("answer_em") if quality else None, f1=rec.get("answer_token_f1") if quality else None,
                    online_ttft_s=rec["online_ttft_s"] if timing else None)), flush=True)
            driver.barrier()
            return result

        check_row = qualification_row(manifest)
        reference = phase(check_row, "qualification.native", "qualified_native", arm="qualification")
        candidate = phase(check_row, "qualification.all_selected", "all_selected", arm="qualification")
        qualification_passed(reference, candidate)
        qualification = base.seal(dict(schema="s11-qualification-v1", passed=True,
            cuda_mask_qualification_sha256=cuda_mask_qualification["sha256"],
            empty_ep_qualification_sha256=empty_ep_qualification["sha256"],
            runtime_qualification_sha256=runtime_qualification["sha256"],
            system_metadata_qualification_sha256=metadata_qualification["sha256"],
            group17_metadata_qualification_sha256=metadata17_qualification["sha256"],
            reference_sha256=reference["sha256"], candidate_sha256=candidate["sha256"],
            identity_check_count=candidate["controller"]["identity_check_count"], quality_claim=False,
            qualification_input_tokens=16384, full_input_numerical_equivalence_claim=False,
            old_qualification_reused=False))
        if rank == 0:
            write_json(args.out_dir/"qualification.json", qualification)
        for row in selected_rows(manifest,length=args.profile_config_value['length'],
                cases_per_length=args.profile_config_value['cases_per_length']):
            index=row['index'];prefix=f'case{index}'
            if controller is not None:
                # Both the resident bank and every numerical admission are fresh
                # for this case. prepare_case alone would not clear old proofs.
                controller.clear_cache(reset_sparse_qualification=True)
                controller.set_cache_output_dir(args.out_dir/f'{prefix}.offline-kv')
            build_start=time.perf_counter()
            builds=[phase(doc,f"{prefix}.offline_doc{doc['offline_document_index']}","capture_independent",arm="offline")
                for doc in document_jobs(base,row)]
            if controller is not None:
                with allow_cuda_sync():
                    controller.prepare_case(input_ids=row['input_ids'],documents=row['documents'],
                        index_device=layers[0].self_attn.q_a_kv_swa_proj.weight.device)
            driver.barrier()
            build_s=max(driver.gather(time.perf_counter()-build_start))
            reference=phase(row,f'{prefix}.s12_dense_reference32','reference_sparse',arm='s1_system',
                batch_group_chunks=1,tokens=32,qualification_budget=True)
            qualified=phase(row,f'{prefix}.s12_group17_qualification32','sensitive',arm='s1_system',
                verify_sparse=True,tokens=32,qualification_budget=True)
            sparse=sparse_identity32(reference,qualified,row,qualification['sha256'])
            if rank==0:write_json(args.out_dir/f'{prefix}.sparse-qualification.json',sparse)
            quality={};warmups={}
            for arm in ARMS:
                quality[arm]=phase(row,f'{prefix}.{arm}_quality','native' if arm=='recomputed' else 'sensitive',
                    arm=arm,quality=True)
                require_natural_eos(quality[arm])
            for arm in ARMS:
                warmups[arm]=phase(row,f'{prefix}.{arm}_warmup','native' if arm=='recomputed' else 'sensitive',
                    arm=arm,tokens=32)
            measurements=[phase(row,f'{prefix}.{arm}_measure{repeat}','native' if arm=='recomputed' else 'sensitive',
                arm=arm,tokens=32,timing=True,repeat=repeat) for repeat,arm in timing_schedule(index,args.profile_config_value['evaluation_case_indices'])]
            shared=verify_shared_bank([*quality.values(),*warmups.values(),*measurements],row)
            require(shared==sparse['shared_bank_identity'],'formal requests changed the numerically qualified bank')
            timing_summary=summarize_timing(measurements,index,args.profile_config_value['evaluation_case_indices'])
            for value in measurements:
                require(value['record']['output_ids']==warmups[value['arm']]['record']['output_ids'],
                    'clean deterministic fixed32 outputs differ across repetitions')
            verify_output_prefix(reference,quality['s1_system'],warmups['s1_system'],
                [value for value in measurements if value['arm']=='s1_system'])
            clean=base.seal(dict(schema='s12-multi-clean-two-arm-case-v1',index=index,method_name='S1.2',
                display_names=DISPLAY_NAMES,length_band=row['length_band'],input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
                context_sha256=row['context_sha256'],profile_config_sha256=identity['profile_config_sha256'],
                qualification_sha256=qualification['sha256'],sparse_qualification_sha256=sparse['sha256'],
                sparse_qualification=evidence_ref(args.out_dir,f'{prefix}.sparse-qualification.json',sparse['sha256']),
                reference=evidence_ref(args.out_dir,reference['name']+'.json',reference['sha256']),
                candidate_qualification=evidence_ref(args.out_dir,qualified['name']+'.json',qualified['sha256']),
                shared_bank_identity=shared,offline_build_s=build_s,
                offline_phases=[evidence_ref(args.out_dir,v['name']+'.json',v['sha256']) for v in builds],
                quality={a:evidence_ref(args.out_dir,v['name']+'.json',v['sha256']) for a,v in quality.items()},
                warmups={a:evidence_ref(args.out_dir,v['name']+'.json',v['sha256']) for a,v in warmups.items()},
                timing_phases=[evidence_ref(args.out_dir,v['name']+'.json',v['sha256']) for v in measurements],
                timing_summary=timing_summary,profiling_installed=False,
                s12_group_chunks=17,s12_idle_bypass_enabled=True,formal_s11_retired=True,
                same_policy_qualification_prefix_matches_formal=True,
                full_model_output_equality_claim=False,historical_measurements_used=False,
                all_output_ids_and_text_retained=True))
            if rank==0:write_json(args.out_dir/f'{prefix}.clean-paired.json',clean)
            # Rank zero must finish publishing before any other rank reads the
            # artifact to construct its completed-case receipt.
            driver.barrier()
            completed.append(dict(index=index,clean_case_sha256=clean['sha256'],
                clean_case=evidence_ref(args.out_dir,f'{prefix}.clean-paired.json',clean['sha256'])))
            driver.barrier()
        # Restore the fixed bootstrap/preflight identity after requests have ended.
        # This value never supplies the actual quality request budget above.
        args.mode, args.max_new_tokens = "quality", 64
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        if rank==0:
            write_json(args.out_dir/"run.json",base.seal(dict(schema=SCHEMA,status="complete",identity=identity,
                manifest_sha256=manifest["sha256"],profile_config_sha256=identity["profile_config_sha256"],
                evaluation_case_indices=args.profile_config_value['evaluation_case_indices'],
                length=args.profile_config_value['length'],cases_per_length=args.profile_config_value['cases_per_length'],completed_case_indices=[r['index'] for r in completed],
                completed_cases=completed,method_name='S1.2',display_names=DISPLAY_NAMES,engine_init_s=init_s,
                group17_metadata_qualification_sha256=metadata17_qualification['sha256'],
                other_cases_executed=False,automatic_retry=False,profiling_in_formal_timing=False,
                formal_s11_retired=True,s12_group_chunks=17,s12_idle_bypass_enabled=True)))
        driver.barrier()
    except BaseException as exc:
        # These are already-host Python scope fields, not tensor reads. Keep
        # the exact failed row layout without adding a synchronization path.
        failure_context = None
        if controller is not None:
            failure_context = getattr(getattr(controller, "adapter", None), "last_failure_context", None)
        write_json(args.out_dir/f"failure.rank{rank}.json", base.seal(dict(schema=SCHEMA, status="failed", rank=rank,
            error=repr(exc), traceback=traceback.format_exc()[-32768:], completed_indices=[r["index"] for r in completed],
            completed_cases=completed,
            ffn_failure_context=failure_context)))
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup = []
        for obj, method in ((batch_worker, "close"), (controller, "close"), (binder, "restore"), (llm, "close")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except BaseException as exc:
                    cleanup.append(f"{method}: {exc!r}")
        if cleanup and not active_error:
            raise RuntimeError("natural-boundary cleanup failed: " + "; ".join(cleanup))



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s12-config',dest='profile_config',type=Path,required=True)
    known,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    args=support.parse_args(rest);vars(args).update(vars(known))
    parent.bind_native_snapshot(args)
    base,native,NativeProfile,scoped_base_profile=support.load_runtime(args.experiment_root)
    profile=NativeProfile(mtp_tokens=0,keep_tokens=140)
    with scoped_base_profile(base,profile,args=args):
        support.validate_args(args)
        manifest,identity=base.preflight(args)
        from gpu_pilot.loong_inputs import validate_inputs
        validate_inputs(base,manifest,base.read_json(args.pilot_protocol),input_dir=args.manifest.parent)
        config=validate_config(args,manifest);args.profile_config_value=config
        rows=selected_rows(manifest,length=config['length'],cases_per_length=config['cases_per_length'])
        original=inherited_protocol.s11_protocol()
        identity.update(parent.frozen_core_identity(args.repo,original))
        parent.verify_diagnostic_native_identity(identity,original)
        identity.update(pilot_source_sha256=source_hashes(args.experiment_root,config),native_yoco_swa_share_hidden=False,
            native_yoco_swa_configuration=parent.native_yoco_swa_configuration(args.repo,args.model_path),
            native_profile=profile.identity(args),s1_protocol_sha256=config['sha256'],profile_config_sha256=config['sha256'],
            profile_external_source_sha256=config['external_source_sha256'],workflow_stage=SCHEMA,no_redknot=False,
            experiment_root=str(args.experiment_root.resolve()),isolated_native_snapshot=str(args.repo.resolve()))
        identity['execution_configuration_sha256']=base.digest(dict(native_profile=identity['native_profile'],
            fixed_env=identity['fixed_env'],native_source=base.source_identity(args.repo),
            pilot_source_sha256=identity['pilot_source_sha256'],profile_config_sha256=config['sha256'],
            native_yoco_swa_configuration=identity['native_yoco_swa_configuration']))
        require(not native.gpu_runtime_imported(),'CPU preflight imported GPU runtime')
        preflight=base.seal(dict(schema=SCHEMA,identity=identity,manifest_sha256=manifest['sha256'],profile_config=config,
            original_manifest_path=str(args.manifest.resolve()),evaluation_case_indices=config['evaluation_case_indices'],
            length=config['length'],cases_per_length=config['cases_per_length'],
            audit_cases=[{k:v for k,v in row.items() if k not in ('input_ids','prompt','context_text')} for row in rows],
            input_geometry=[parent.case_geometry(row) for row in rows],native_yoco_swa_share_hidden=False,
            qualification_input_tokens=16384,other_evaluation_cases_executed=False,
            method_name='S1.2',display_names=DISPLAY_NAMES,profiling_installed=False,
            lifecycle_timeout_seconds=LIFECYCLE_TIMEOUT_SECONDS,
            formal_s11_retired=True))
        if args.check_only:
            print(json.dumps(dict(status='cpu_preflight_passed',gpu_initialized=False,preflight=preflight)))
            return 0
        args.out_dir.mkdir(parents=True,exist_ok=False)
        write_json(args.out_dir/'preflight.json',preflight)
        write_json(args.out_dir/'selected-inputs.json',base.seal(dict(schema='s12-selected-original-inputs-v1',rows=rows,
            original_manifest_sha256=manifest['sha256'],profile_config_sha256=config['sha256'])))
        base.wait_for_release(args,manifest['sha256'])
        verify_frozen(args,manifest,identity,profile,base,native_source=True)
        from quality_process_group import launch
        launch(functools.partial(worker,args,manifest,identity),args.dp,args.out_dir)
    return 0


if __name__=='__main__':raise SystemExit(main())
