"""Independent CPU readout for future pinned 64K/128K Full / S1.2 reproduction.

Frozen S1.2 group17/idle runtime and original algorithm sources are untouched.
Only whole cases with fresh numerical/first32 gates, natural EOS in both arms,
and clean warmup plus three TTFT measurements enter performance results.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
SOURCE_ROOT=HERE if (HERE/'GatedEngram').is_dir() else HERE.parent
sys.path.insert(0,str(SOURCE_ROOT/'GatedEngram'))
sys.path.insert(0,str(SOURCE_ROOT))
from gpu_pilot import read_s11_results as inherited
from gpu_pilot import run_s11_pilot as contracts
from gpu_pilot.token_divergence import TokenizerDecoder,audit_decoded_record
from reproduction_contract import reproduction_contract,BAND_INDICES

def group_contract(size):
    inherited.require(size==17,"reproduction only admits S1.2 group17")
    return reproduction_contract()

SCHEMA='s12-reproduction-independent-readout-v1'
ARMS=('recomputed','s1_system')
DISPLAY_NAMES={'recomputed':'Full','s1_system':'S1.2'}
ORIGINAL_MANIFEST_SHA256='fb6bba62937fb8005f05df9a3a213729c172932a30e5509b136000692881aa67'
FROZEN_STAGE_SHA256='0321d533deceb984c126b07d24a099316cc7c8dcc769e470ccbcbc7095fa4d36'
FROZEN_IDLE_SHA256='be34a00fde1262176d6eb69fa50289609343a26b6cf0b5d5fc6d9d4c203e58b3'
FROZEN_GROUP_CONTRACT_SHA256='855fbb8e8c02eb7eae9f935472778e53d72b97843e83f3ca0b0736e3074fde35'
SCHEME_DEFINITION_SHA256='5b3574185974c4d39b5bde217a61a506f539f2b017abcb1536670eaa5a003704'
digest=inherited.digest
require=inherited.require

def seal(body):
    body = {k: v for k, v in body.items() if k != 'sha256'}
    return dict(body, sha256=digest(body))


def fields(value, expected, label):
    require(type(value) is dict and all(value.get(k) == v for k, v in expected.items()), label)


def finite(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), label)
    return value


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.references = {}
        self.cache = {}

    def read(self, name, reference=None):
        require(type(name) is str and Path(name).name == name and name.endswith('.json'),
                'evidence must be a plain JSON filename')
        if name not in self.cache:
            path = self.directory / name
            require(path.is_file() and not path.is_symlink(), 'missing/linked evidence: ' + name)
            raw = path.read_bytes()
            value = json.loads(raw)
            inherited.sealed(value)
            self.references[name] = dict(file_sha256=hashlib.sha256(raw).hexdigest(), seal_sha256=value['sha256'])
            self.cache[name] = value
        value = self.cache[name]
        if reference is not None:
            require(reference == dict(file=name, **self.references[name]), 'raw reference differs: ' + name)
        return value

    def ref(self, reference):
        require(type(reference) is dict and set(reference) == {'file', 'file_sha256', 'seal_sha256'},
                'incomplete evidence reference')
        return self.read(reference['file'], reference)


def candidate_reports(phase):
    """Bind each true bypass to the original eight-rank world scheduler trace."""
    fields(phase, dict(candidate_enabled=True, diagnostic=False, diagnostic_gpu_profile=False),
           'candidate phase is absent or diagnostic')
    reports = phase.get('instrumentation_reports')
    workers = phase.get('system_batch_worker_reports')
    schedule = phase.get('system_batch_schedule')
    require(type(reports) is list and len(reports) == 8 and type(workers) is list and len(workers) == 8
            and type(schedule) is dict, 'candidate needs actual eight-rank instrumentation/schedule')
    inherited.sealed(schedule)
    counts = []
    ordinals = None
    for rank, (report, worker) in enumerate(zip(reports, workers)):
        inherited.sealed(report)
        fields(report, dict(schema='s12-world-idle-forward-candidate-v1', rank=rank, enabled=True,
            phase=phase['name'], request_id=phase['record']['request_id'],
            source_derived_entries_are_not_measured_time=True,
            existing_world_model_execution_completed_means_callback_return_not_decoder_execution=True,
            real_prefill_decode_forward_unchanged=True, gpu_numerically_qualified=False,
            qualification_is_external_fresh_gate=True, placeholder_state_zero_asserted=True,
            placeholder_expert_load_statistics_may_change=True,
            routing_rearrangement_disabled_by_actual_capabilities=True), 'candidate rank/enable/request differs')
        capabilities = report.get('capabilities')
        fields(capabilities, dict(cp_size=1, dp_size=8, ep_size=8, eager=True, nextn=0,
            dflash_input_size=0, mtp_varlen=False, rearrangement_interval=0,
            rearrange_after_prefill=False, force_imbalance_ratio=0., worker_nextn=0,
            model_load_nextn=0, model_num_layers=40), 'candidate native capabilities differ')
        intervals = capabilities.get('actual_balancer_intervals')
        after_prefill = capabilities.get('actual_balancer_after_prefill')
        require(type(intervals) is list and bool(intervals) and all(type(v) is int and v == 0 for v in intervals)
                and type(after_prefill) is list and len(after_prefill) == len(intervals)
                and all(v is False for v in after_prefill), 'candidate actual adaptive EPLB is enabled/unknown')
        trace = worker.get('world_step_trace')
        require(worker.get('rank') == rank and worker.get('schedule_sha256') == schedule['sha256']
                and type(trace) is list, 'candidate worker trace binding differs')
        idle = {event['ordinal']: event for event in trace if event.get('world_phase') == 'idle'}
        require(report.get('active_real_state_nonzero_checks') ==
                (sum(event.get('world_phase') in ('prefill', 'decode') for event in trace) if rank == 0 else 0),
                'candidate real request state-zero separation checks differ')
        events = report.get('events')
        require(type(events) is list and len(events) > 0 and len(events) == len(idle)
                and report.get('bypassed_world_idle_steps') == len(events)
                and report.get('source_derived_avoided_decoder_entries') == 40 * len(events)
                and report.get('output_values_for_discarded_placeholder_changed') is True,
                'candidate did not actually bypass every admitted world-idle step')
        seen = set()
        for event in events:
            ordinal = event.get('ordinal')
            require(type(ordinal) is int and ordinal in idle and ordinal not in seen,
                    'candidate bypass points outside unique world-idle trace')
            seen.add(ordinal)
            original = idle[ordinal]
            fields(original, dict(local_is_empty=True, rank0_request_id=None, start=None, end=None,
                scheduler_prompt_run=False, local_prompt_run=False, model_execution_completed=True),
                'candidate bypass classified a real request as idle')
            fields(event, dict(world_phase='idle', schedule_sha256=schedule['sha256'],
                scheduler_observations_sha256=original.get('scheduler_observations_sha256'),
                phase=phase['name'], request_id=phase['record']['request_id'],
                idle_forward_skipped=True, native_decoder_calls=0, returned_placeholder_rows=1,
                real_query_rows=0, worker_forward_postprocess_sample_round_preserved=True,
                sampler_and_eplb_step_preserved=True, dummy_output_is_real_model_output=False,
                real_request_cache_accessed_by_bypass=False, total_q_is_prefill_only_default=True,
                native_decode_metadata=dict(num_reqs=1,total_q=0,canonical_request_binding_present=False,
                    input_ids_shape=[1],positions_shape=[1],state_block_idx_shape=[1],valid_token_mask_shape=[1])),
                'candidate idle event differs from actual trace')
        require(ordinals is None or seen == ordinals, 'candidate world-idle bypass disagrees across ranks')
        ordinals = seen
        counts.append(len(events))
    return dict(actual_bypassed_world_idle_steps_per_rank=counts,
        source_derived_avoided_decoder_entries_per_rank=[40 * n for n in counts],
        source_derived_entries_are_not_measured_time=True,
        original_worker_model_execution_completed_is_callback_only=True)


def check_complete_quality(phase, decoder):
    record = phase['record']
    require(record.get('finish_reason') == 'stop' and record.get('natural_EOS') is True
            and record.get('truncated') is False and record.get('quality_output_complete') is True,
            'complete natural-EOS output required for performance acceptance')
    return audit_decoded_record(record, decoder)


def audit_group17_metadata(value, identity):
    """Independently validate real eight-rank strict-CUDA group17 operator proof."""
    inherited.sealed(value)
    sources=('GatedEngram/kv_cache_engram/s1_system_batch.py','GatedEngram/kv_cache_engram/s1_system_runtime.py')
    fields(value,dict(schema='s12-group17-all-rank-metadata-v1',passed=True,
        execution_configuration_sha256=identity['execution_configuration_sha256'],
        model_sha256=identity['checkpoint_index_sha256'],
        operator_source_sha256={name:identity['pilot_source_sha256'][name] for name in sources},
        quality_eligible=False,timing_valid=False,gpu_numerical_parity_claim=False),
        'group17 actual CUDA metadata proof binding differs')
    checks=[dict(name='inactive_empty_cu',passed=True,rows=0,cu=[0,0]),
        dict(name='active_queue_group17',passed=True,input_tokens=139265,member_chunks=18,
            deep_groups=2,empty_groups=0,selected_rows=128)]
    groups=[dict(group_chunks=17,start=0,end=139264,selected_rows=127,empty=False,
        sampler_output_rows=139,exact_metadata=True,strict_sync_debug_mode=2),
        dict(group_chunks=17,start=139264,end=139265,selected_rows=1,empty=False,
        sampler_output_rows=1,exact_metadata=True,strict_sync_debug_mode=2)]
    reports=value.get('reports')
    require(type(reports) is list and len(reports)==8,'group17 metadata proof needs all eight ranks')
    for rank,item in enumerate(reports):
        require(type(item) is dict and set(item)=={'rank','operator_report'} and type(item['rank']) is int
                and item['rank']==rank,'group17 metadata rank identity differs')
        report=item['operator_report'];inherited.sealed(report)
        fields(report,dict(schema='s1-system-cuda-metadata-qualification-v1',passed=True,
            cuda_exercised=True,device_type='cuda',sync_debug_mode_during=2,gpu_numerical_parity_claim=False,
            requested_batch_groups=[17],checks=checks,active_queue_groups=groups),
            'group17 CUDA metadata/operator coverage differs')
        before,after=report.get('sync_debug_mode_before'),report.get('sync_debug_mode_after')
        require(type(before) is int and before in (0,1,2) and type(after) is int and before==after,
                'group17 metadata gate failed to restore strict synchronization guard')
    return value


def qualification_output(phase):
    """A 32-token, EOS-enabled gate is distinct from full answer quality."""
    fields(phase,dict(timing_valid=False,quality_eligible=False,diagnostic=False,
        diagnostic_gpu_profile=False,qualification_only=True,qualification_budget=True,eos_enabled=True,generation_token_budget=32),
        'same-policy qualification must use explicit EOS-enabled budget32')
    record=phase['record'];ids=record.get('output_ids');finish=record.get('finish_reason')
    require(type(ids) is list and 1<=len(ids)<=32 and all(type(v) is int and v>=0 for v in ids)
        and record.get('output_ids_sha256')==digest(ids) and record.get('success') is True
        and record.get('rank')==0 and record.get('natural_EOS') is True
        and finish in ('stop','length') and record.get('truncated') is (finish=='length'),
        'qualification raw output IDs/termination differ')
    require((finish=='stop' and ids[-1]==1 and ids.count(1)==1)
        or (finish=='length' and len(ids)==32 and 1 not in ids),
        'qualification EOS/32-token truncation binding differs')
    require(type(record.get('text')) is str,'qualification complete recorded text required')
    return ids


def sparse_identity32(reference,candidate,row,qualification_sha256):
    """Full-layer output/mHC/cache proof plus complete bounded first32 trajectory."""
    inherited.validate_digest(qualification_sha256,'fresh native qualification seal')
    contract=group_contract(17);spans=inherited.documents(row)
    chunks=inherited.chunk_oracle(spans,row['input_tokens'])
    expected={(c['chunk_start'],c['chunk_end'],layer):c for c in chunks for layer in range(40)}
    banks=[];outputs=[]
    for phase,mode,group,enabled in ((reference,'reference_sparse',1,False),(candidate,'sensitive',17,True)):
        inherited.sealed(phase)
        fields(phase,dict(schema=inherited.PHASE_SCHEMA,index=row['index'],arm='s1_system',mode=mode,
            input_ids_sha256=row['input_ids_sha256'],fraction=None,system_enabled=True,batch_group_chunks=group,
            s1_policy=True,natural_boundary_inputs=True,native_yoco_swa_share_hidden=False,candidate_enabled=enabled),
            'raw S1.2 qualification policy/group/input differs')
        ids=qualification_output(phase);outputs.append(ids)
        fields(phase['record'],dict(input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
            phase=phase['name'],sample_id=row['sample_id'],dataset=row['dataset'],
            request_id=f"s1:{row['index']}:{phase['name']}:{row['sample_id']}"),
            'qualification complete request identity differs')
        control=phase['controller'];contract.audit_controller(control,row,phase,spans)
        banks.append(contract.comparison_scope(control,phase,spans=spans))
        values=control.get('sparse_identity_fingerprints')
        require(type(values) is list and len(values)==len(expected)
            and control.get('sparse_identity_check_count')==len(expected)
            and control.get('sparse_identity_fingerprints_sha256')==digest(values),
            'all original layer/chunk sparse fingerprints required')
        seen=set()
        for item in values:
            key=tuple(item.get('key',[]));value=item.get('value',{})
            require(key in expected and key not in seen and set(value)=={'output','lazy'}
                and set(value['lazy'])=={'residual','post_layer_mix','comb_res_mix','pre_layer_mix'},
                'qualification output/all-four-mHC coverage differs')
            seen.add(key);require(value['output'] is not None,'qualification decoder output missing')
            inherited._fingerprint(value['output'])
            n=key[1]-key[0] if key[2]<3 else len(expected[key]['selected_positions'])
            require(value['output']['shape'][0]==n,'qualification retained decoder rows differ')
            for field in value['lazy'].values():
                inherited._fingerprint(field)
                require(field is None or len(field['shape'])>=2 and field['shape'][:2]==[1,n],
                        'qualification mHC row-axis/layout differs')
    for key in ('manifest_sha256','execution_configuration_sha256','s1_protocol_sha256','profile_config_sha256',
                'empty_ep_qualification_sha256','runtime_qualification_sha256',
                'system_metadata_qualification_sha256','group17_metadata_qualification_sha256'):
        require(reference.get(key)==candidate.get(key),'same-policy reference/candidate execution differs: '+key)
    require(banks[0]==banks[1] and reference['controller']['phase_generation']!=candidate['controller']['phase_generation'],
            'qualification reused phase state or changed the offline bank')
    require(candidate.get('verify_sparse') is True and reference.get('verify_sparse') is False
        and candidate['controller'].get('sparse_identity_exact') is True
        and reference['controller']['sparse_identity_fingerprints']==candidate['controller']['sparse_identity_fingerprints'],
        'S1.2 per-layer output/four-mHC numerical parity failed')
    require(outputs[0]==outputs[1] and reference['record']['text']==candidate['record']['text']
        and reference['record']['finish_reason']==candidate['record']['finish_reason'],
        'S1.2 complete first32/EOS reference trajectory differs')
    return dict(index=row['index'],input_tokens=row['input_tokens'],input_ids_sha256=row['input_ids_sha256'],
        qualification_sha256=qualification_sha256,reference_sha256=reference['sha256'],candidate_sha256=candidate['sha256'],
        identity_check_count=len(expected),identity_fingerprints_sha256=candidate['controller']['sparse_identity_fingerprints_sha256'],
        batch_group_chunks=17,exact_same_policy_reference=True,equivalence_to_full_model_claim=False,
        quality_eligible=False,timing_valid=False,output_ids=list(outputs[1]),
        output_ids_sha256=digest(outputs[1]),output_tokens=len(outputs[1]),finish_reason=candidate['record']['finish_reason'],
        first32_or_earlier_eos_ids_and_text_equal=True,complete_eos_reference_quality_not_required=True,
        shared_bank_identity=banks[0])


def check_qualification_prefix(record,qualification_record,*,natural_quality):
    reference=qualification_record['output_ids'];ids=record['output_ids']
    require(ids[:len(reference)]==reference,'S1.2 fresh phase changed the qualified first32/EOS prefix')
    if natural_quality and qualification_record['finish_reason']=='stop':
        require(ids==reference,'early-EOS qualification does not match complete S1.2 quality output')
    return True


def case_pin(row):
    names=('index','sample_id','source_id','source_index','source_row_sha256','source_qa_sha256',
           'context_sha256','input_ids_sha256','input_tokens','length_band','raw_answer_sha256')
    return dict({name:row[name] for name in names},documents_sha256=digest(row['documents']))


def selected_indices(length,cases_per_length):
    require(length in ('64k','128k','both') and type(cases_per_length) is int and 1<=cases_per_length<=10,
        'reproduction length/case count must be 64k, 128k or both, with 1..10 per length')
    bands=('64k','128k') if length=='both' else (length,)
    return [index for band in bands for index in BAND_INDICES[band][:cases_per_length]]


def original_selection(manifest,config):
    inherited.sealed(manifest)
    require(manifest['sha256']==ORIGINAL_MANIFEST_SHA256,'pinned original input manifest differs')
    cohort=manifest.get('cohort')
    require(type(cohort) is list and len({row['index'] for row in cohort})==len(cohort),'duplicate source input indices')
    indices=selected_indices(config.get('length'),config.get('cases_per_length'))
    by_index={row['index']:row for row in cohort}
    require(all(index in by_index for index in indices),'selected pinned input is missing')
    selected=[by_index[index] for index in indices]
    reproduction_contract().audit_length_inputs(selected)
    require(config.get('evaluation_case_indices')==indices,'declared execution order differs from deterministic input selection')
    return selected


def audit_sources(config,identity):
    from gpu_pilot.launch_pilot import source_files
    from gpu_pilot.launch_index_hit20_pilot import snapshot_sources
    fields(config,dict(stage_sources_sha256=FROZEN_STAGE_SHA256,
        native_sources_sha256=inherited.NATIVE_SNAPSHOT_SHA256,
        core_source_sha256=inherited.CORE_SHA256,core_policy_sha256=inherited.CORE_POLICY_SHA256),
        'S1.2 frozen algorithm/runtime/native pins differ')
    require(digest(source_files(SOURCE_ROOT/'GatedEngram'))==FROZEN_STAGE_SHA256
            and digest(snapshot_sources(SOURCE_ROOT/'NativeSnapshot'))==inherited.NATIVE_SNAPSHOT_SHA256,
            'actual unchanged frozen stage/native source bytes differ')
    pins=config.get('external_source_sha256')
    require(type(pins) is dict and set(pins)=={'run_s12_reproduction.py','reproduction_contract.py','idle_optimizer.py','profile_group_contract.py'}
        and pins['idle_optimizer.py']==FROZEN_IDLE_SHA256
        and pins['profile_group_contract.py']==FROZEN_GROUP_CONTRACT_SHA256,
        'formal S1.2 idle/group17 implementation changed')
    require(all(hashlib.sha256((SOURCE_ROOT/name).read_bytes()).hexdigest()==key for name,key in pins.items()),
            'new wrapper/frozen external source bytes differ')
    fields(identity,dict(profile_external_source_sha256=pins,core_source_sha256=inherited.CORE_SHA256,
        core_policy_sha256=inherited.CORE_POLICY_SHA256,native_snapshot_sources_sha256=inherited.NATIVE_SNAPSHOT_SHA256),
        'actual native/algorithm execution identity differs')
    overrides=config.get('system_source_override_sha256')
    require(type(overrides) is dict and set(overrides)=={
        'GatedEngram/kv_cache_engram/s1_system_batch.py','GatedEngram/kv_cache_engram/s1_system_runtime.py'},
        'only frozen system grouping source overrides are allowed')
    require(all(identity['pilot_source_sha256'].get(name)==key==hashlib.sha256((SOURCE_ROOT/name).read_bytes()).hexdigest()
        for name,key in overrides.items()),'actual system execution source identity differs')


def audit_native_qualification(evidence,preflight,metadata17):
    identity=preflight['identity'];mask=evidence.read('cuda_mask_qualification.json')
    from gpu_pilot.cuda_mask_qualification import validate_report
    fields(mask,dict(schema='ffn-only-cuda-mask-all-ranks-v1',passed=True,
        execution_configuration_sha256=identity['execution_configuration_sha256'],quality_eligible=False,timing_valid=False),
        'CUDA mask proof detached from this run')
    require(type(mask.get('reports')) is list and len(mask['reports'])==8
        and [r.get('rank') for r in mask['reports']]==list(range(8)),'all eight CUDA mask ranks required')
    for item in mask['reports']:validate_report(item['report'])
    from gpu_pilot.index_hit20_empty_ep_qualification import validate_all_ranks_report
    empty=evidence.read('empty_ep_qualification.json')
    validate_all_ranks_report(empty,execution_sha256=identity['execution_configuration_sha256'],
        model_sha256=identity['checkpoint_index_sha256'])
    runtime=evidence.read('s1_runtime_qualification.json')
    inherited.validate_runtime_qualification(runtime,identity,empty['sha256'])
    metadata=evidence.read('s1_system_metadata_qualification.json')
    inherited.validate_metadata_qualification(metadata,identity)
    reference=evidence.read('qualification.native.json');candidate=evidence.read('qualification.all_selected.json')
    closure=evidence.read('qualification.json')
    fields(closure,dict(schema='s11-qualification-v1',passed=True,cuda_mask_qualification_sha256=mask['sha256'],
        empty_ep_qualification_sha256=empty['sha256'],runtime_qualification_sha256=runtime['sha256'],
        system_metadata_qualification_sha256=metadata['sha256'],reference_sha256=reference['sha256'],
        candidate_sha256=candidate['sha256'],identity_check_count=80,qualification_input_tokens=16384,
        quality_claim=False,full_input_numerical_equivalence_claim=False,old_qualification_reused=False),
        'fresh native/runtime qualification closure differs')
    for phase in (reference,candidate):
        inherited.binding(phase,preflight);inherited.empty_ep_binding(phase,closure)
        fields(phase,dict(profile_config_sha256=preflight['profile_config']['sha256'],
            runtime_qualification_sha256=runtime['sha256'],group17_metadata_qualification_sha256=metadata17['sha256'],
            diagnostic_gpu_profile=False,candidate_enabled=False,instrumentation_reports=[None]*8),
            'native qualification is not a clean freshly bound phase')
        require(phase['controller']['cache_report'].get('offline_bank_entry_count')==0,'native gate used offline bank')
    inherited.native_identity(reference,candidate)
    return closure


def timing_schedule(index,case_indices):
    require(index in case_indices,"foreign reproduction case index")
    ordinal=case_indices.index(index)
    return [(repeat,ARMS[(ordinal+repeat+slot)%2]) for repeat in range(3) for slot in range(2)]


def audit_sparse_closure(closure,computed,reference,candidate,row):
    keys=('index','input_tokens','input_ids_sha256','qualification_sha256','reference_sha256',
        'candidate_sha256','identity_check_count','identity_fingerprints_sha256','batch_group_chunks',
        'exact_same_policy_reference','equivalence_to_full_model_claim','quality_eligible','timing_valid',
        'output_ids_sha256','finish_reason','shared_bank_identity')
    fields(closure,{key:computed[key] for key in keys},'same-policy numerical closure differs from independent recomputation')
    fields(closure,dict(schema='s12-multi-sparse-qualification-v1',passed=True,method_name='S1.2',
        arm='s1_system',system_enabled=True,generation_token_budget=32,eos_enabled=True,
        generated_tokens=computed['output_tokens'],
        output_text_sha256=hashlib.sha256(candidate['record']['text'].encode()).hexdigest(),
        complete_generated_ids_text_stop_reason_equal=True,old_qualification_reused=False,
        empty_chunk_exercised=any(not c['selected_positions'] for c in inherited.chunk_oracle(
            inherited.documents(row),row['input_tokens']))),'same-policy bounded output closure differs')


def expected_timing_summary(measurements,index,case_indices):
    require([(p.get('repeat'),p.get('arm')) for p in measurements]==timing_schedule(index,case_indices),
        'exactly six clean measurements in registered rotated order required')
    samples={arm:[p['record']['online_ttft_s'] for p in measurements if p['arm']==arm] for arm in ARMS}
    summary=inherited.expected_timing_summary(samples)
    for arm in ARMS:summary['arms'][arm]['display_name']=DISPLAY_NAMES[arm]
    return summary


def audit_case(evidence,row,preflight,qualification,metadata17,decoder):
    index=row['index'];prefix=f'case{index}';config=preflight['profile_config'];identity=preflight['identity']
    contract=group_contract(17);spans=inherited.documents(row);bank=None;generations=set();bypass={}
    clean=evidence.read(prefix+'.clean-paired.json')
    fields(clean,dict(schema='s12-multi-clean-two-arm-case-v1',index=index,method_name='S1.2',length_band=row['length_band'],
        display_names=DISPLAY_NAMES,input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
        context_sha256=row['context_sha256'],profile_config_sha256=config['sha256'],
        qualification_sha256=qualification['sha256'],profiling_installed=False,s12_group_chunks=17,
        s12_idle_bypass_enabled=True,formal_s11_retired=True,same_policy_qualification_prefix_matches_formal=True,
        full_model_output_equality_claim=False,historical_measurements_used=False,
        all_output_ids_and_text_retained=True),'complete two-arm case scope differs')

    def read_role(ref,name):
        require(ref.get('file')==name+'.json','case evidence filename/role differs')
        phase=evidence.ref(ref)
        require(phase.get('name')==name,'raw phase name differs from reference')
        return phase

    def bind(phase,*,formal=False):
        nonlocal bank
        contract.binding(phase,preflight);inherited.empty_ep_binding(phase,qualification)
        fields(phase,dict(profile_config_sha256=config['sha256'],
            runtime_qualification_sha256=qualification['runtime_qualification_sha256'],
            group17_metadata_qualification_sha256=metadata17['sha256'],diagnostic=False,
            diagnostic_gpu_profile=False,partial_smoke=False),'phase fresh qualification/config binding differs')
        candidate=phase['arm']=='s1_system' and phase['mode']!='reference_sparse'
        if candidate:
            fields(phase,dict(candidate_enabled=True,candidate_variant='idle_group17',batch_group_chunks=17,
                method_name='S1.2',display_name='S1.2'),'S1.2 actual idle/group17 implementation differs')
            bypass[phase['name']]=candidate_reports(phase)
        else:
            fields(phase,dict(candidate_enabled=False,candidate_variant=None,instrumentation_reports=[None]*8),
                'unoptimized Full/dense reference contains candidate or profiler')
        if formal:
            fields(phase,dict(verify_sparse=False,qualification_only=False,qualification_budget=False),
                'formal measurements contain qualification instrumentation')
        observed=contract.comparison_scope(phase['controller'],phase,spans=spans,
            expected_scope=inherited.bank_scope_sha256(spans,identity))
        require(bank is None or bank==observed,'same-case resident bank changed across fresh phases')
        bank=observed;generation=phase['controller']['phase_generation']
        require(type(generation) is int and generation not in generations,'phase reset/generation reused')
        generations.add(generation)

    reference=read_role(clean['reference'],prefix+'.s12_dense_reference32')
    candidate=read_role(clean['candidate_qualification'],prefix+'.s12_group17_qualification32')
    for phase in (reference,candidate):
        bind(phase)
        qualification_output(phase)
        require(decoder.decode(phase['record']['output_ids'],skip_special_tokens=True)==phase['record']['text'],
            'bounded qualification text differs from actual complete token decode')
    computed=sparse_identity32(reference,candidate,row,qualification['sha256'])
    require(clean['sparse_qualification'].get('file')==prefix+'.sparse-qualification.json',
        'case sparse qualification filename differs')
    sparse=evidence.ref(clean['sparse_qualification'])
    audit_sparse_closure(sparse,computed,reference,candidate,row)
    require(clean.get('sparse_qualification_sha256')==sparse['sha256'],'case sparse closure is detached')
    quality={};warmups={};quality_output={}
    require(set(clean.get('quality',{}))==set(ARMS) and set(clean.get('warmups',{}))==set(ARMS),
        'both formal arms require complete quality and fresh warmup')
    for arm in ARMS:
        for kind,target in (('quality',quality),('warmup',warmups)):
            phase=read_role(clean['quality' if kind=='quality' else 'warmups'][arm],f'{prefix}.{arm}_{kind}')
            require(phase.get('repeat') is None,'quality/warmup carries a timing repetition')
            contract.audit_phase(phase,row,preflight,arm=arm,timing=False,quality=kind=='quality')
            bind(phase,formal=True);target[arm]=phase
            if kind=='quality':
                value=inherited.quality_record(phase['record'],phase,row)
                value['output_tokenization']=check_complete_quality(phase,decoder)
                value['online_elapsed_s']=phase['record']['online_elapsed_s']
                value['online_ttft_s']=phase['record']['online_ttft_s']
                value['complete_answer_elapsed_is_actual_single_request']=True
                value['display_name']=DISPLAY_NAMES[arm];quality_output[arm]=value
            if arm=='s1_system':check_qualification_prefix(phase['record'],candidate['record'],natural_quality=kind=='quality')
    measurements=[evidence.ref(ref) for ref in clean.get('timing_phases',[])]
    summary=expected_timing_summary(measurements,index,config['evaluation_case_indices'])
    for phase,ref in zip(measurements,clean['timing_phases']):
        arm,repeat=phase['arm'],phase['repeat']
        require(ref['file']==f'{prefix}.{arm}_measure{repeat}.json' and phase['name']==f'{prefix}.{arm}_measure{repeat}',
            'formal timing phase identity differs')
        contract.audit_phase(phase,row,preflight,arm=arm,timing=True,quality=False);bind(phase,formal=True)
        require(phase['record']['output_ids']==warmups[arm]['record']['output_ids'],
            'fixed32 trajectory changed between warmup and timing repetition')
        if arm=='s1_system':check_qualification_prefix(phase['record'],candidate['record'],natural_quality=False)
    require(clean.get('timing_summary')==summary and clean.get('shared_bank_identity')==bank==computed['shared_bank_identity'],
        'clean timing medians/paired ratios or bank closure differs')
    finite(clean.get('offline_build_s'),'offline build duration',positive=True)
    offline=[evidence.ref(ref) for ref in clean.get('offline_phases',[])]
    require(len(offline)==10 and [p.get('name') for p in offline]==[f'{prefix}.offline_doc{i}' for i in range(10)],
        'all ten case-local offline bank captures required')
    for phase in offline:
        contract.binding(phase,preflight);inherited.empty_ep_binding(phase,qualification)
        fields(phase,dict(mode='capture_independent',arm='offline',timing_valid=False,quality_eligible=False,
            candidate_enabled=False,diagnostic_gpu_profile=False,instrumentation_reports=[None]*8,
            profile_config_sha256=config['sha256'],group17_metadata_qualification_sha256=metadata17['sha256']),
            'offline capture scope/binding differs')
        require(phase['controller']['bank_generation']==bank['bank_generation'],'offline phase belongs to another bank')
    return dict(index=index,status='complete',performance_accepted=True,length_band=row['length_band'],
        target_input_tokens=row['target_input_tokens'],input_tokens=row['input_tokens'],
        input_ids_sha256=row['input_ids_sha256'],context_sha256=row['context_sha256'],source_id=row['source_id'],
        question=row['question'],instruction=row['instruction'],raw_answer=row['raw_answer'],
        quality=quality_output,clean_timing=summary,full_over_s12_speedup=summary['comparisons']['s1_system']['ttft_speedup'],
        shared_bank_identity=bank,all_rank_actual_bypass=bypass,clean_case_sha256=clean['sha256'],
        sparse_qualification_sha256=sparse['sha256'],qualification_generated_tokens=computed['output_tokens'],
        qualification_finish_reason=computed['finish_reason'],fresh_first32_or_earlier_eos_parity=True,
        natural_eos_both_arms=True,full_model_output_parity_claim=False,
        same_policy_numerical_equivalence_is_not_full_accuracy=True,phase_generations=sorted(generations))


def terminal_closure(evidence,completed,preflight,metadata17):
    config=preflight['profile_config'];indices=[c['index'] for c in completed]
    expected=[]
    for result in completed:
        name=f"case{result['index']}.clean-paired.json"
        expected.append(dict(index=result['index'],clean_case_sha256=result['clean_case_sha256'],
            clean_case=dict(file=name,**evidence.references[name])))
    if (evidence.directory/'run.json').exists():
        require(indices==config['evaluation_case_indices'],'complete run is missing an independently closed whole case')
        run=evidence.read('run.json')
        fields(run,dict(schema='s12-reproduction-run-v1',status='complete',identity=preflight['identity'],
            manifest_sha256=config['manifest_sha256'],profile_config_sha256=config['sha256'],
            evaluation_case_indices=config['evaluation_case_indices'],completed_case_indices=indices,completed_cases=expected,
            length=config['length'],cases_per_length=config['cases_per_length'],
            method_name='S1.2',display_names=DISPLAY_NAMES,group17_metadata_qualification_sha256=metadata17['sha256'],
            other_cases_executed=False,automatic_retry=False,profiling_in_formal_timing=False,
            formal_s11_retired=True,s12_group_chunks=17,s12_idle_bypass_enabled=True),'terminal complete run scope differs')
        require(not list(evidence.directory.glob('failure.rank*.json')),'complete run coexists with rank failures')
        return dict(status='complete',raw_run_status='complete',source_run_sha256=run['sha256'],raw_failure=None)
    path=evidence.directory.parent/'supervisor.json'
    require(path.is_file() and not path.is_symlink(),'partial acceptance requires actual terminal supervisor')
    raw=path.read_bytes();supervisor=json.loads(raw)
    fields(supervisor,dict(schema='redknot-glm53-gpu-lifecycle-v1',status='failed',holder_restored=True),
        'partial acceptance requires original failed supervisor with holder restoration')
    require(supervisor.get('finished_unix') and supervisor.get('error') and not supervisor.get('cleanup_error'),
        'partial run has no successful terminal cleanup')
    evidence.references['../supervisor.json']=dict(file_sha256=hashlib.sha256(raw).hexdigest())
    failures=[]
    for path in sorted(evidence.directory.glob('failure.rank*.json')):
        value=evidence.read(path.name);rank=value.get('rank')
        require(type(rank) is int and rank in range(8) and path.name==f'failure.rank{rank}.json','foreign failure rank')
        fields(value,dict(schema='s12-reproduction-run-v1',status='failed',completed_indices=indices,
            completed_cases=expected),'raw failure disagrees with independently closed cases')
        require(type(value.get('error')) is str and value['error'],'rank failure lacks error')
        failures.append(dict(rank=rank,error=value['error'],seal_sha256=value['sha256']))
    return dict(status='partial_complete' if completed else 'failed',raw_run_status='failed',source_run_sha256=None,
        raw_failure=dict(supervisor_file_sha256=hashlib.sha256(raw).hexdigest(),supervisor_error=supervisor['error'],
            rank_failures=failures,failure_not_rewritten_as_success=True,
            holder_restoration_full_physical_proof_audited=False))


def summarize_lengths(cases):
    result={}
    for band in ('64k','128k'):
        selected=[case for case in cases if case['length_band']==band]
        if not selected:continue
        ratios=[case['full_over_s12_speedup'] for case in selected]
        result[band]=dict(completed_case_indices=[case['index'] for case in selected],evaluated_questions=len(selected),
            distinct_contexts=len({case['context_sha256'] for case in selected}),
            full_over_s12_geomean_speedup=math.exp(sum(map(math.log,ratios))/len(ratios)),
            summed_medians_full_over_s12=sum(case['clean_timing']['arms']['recomputed']['online_ttft_p50_s'] for case in selected)/
                sum(case['clean_timing']['arms']['s1_system']['online_ttft_p50_s'] for case in selected))
    return result


def audit(directory,*,tokenizer=None,manifest_path=None):
    evidence=Evidence(directory);preflight=evidence.read('preflight.json');config=preflight['profile_config']
    inherited.sealed(config);identity=preflight['identity']
    fields(config,dict(schema='s12-reproduction-config-v1',
        manifest_sha256=ORIGINAL_MANIFEST_SHA256,arms=list(ARMS),display_names=DISPLAY_NAMES,
        method_name='S1.2',internal_policy_arm='s1_system',formal_s11_retired=True,
        clean_quality_natural_eos=True,clean_warmups_per_arm=1,clean_timing_repeats=3,clean_timing_output_tokens=32,
        profiler_installed=False,candidate_enabled=True,candidate_name='idle_group17',candidate_variants={'idle_group17':17},
        lifecycle_timeout_seconds=7200,timeout_origin='actual_gpu_release_of_this_explicit_reproduction',
        group17_metadata_qualification_required=True,per_case_same_policy_qualification_token_budget=32,
        qualification_eos_enabled=True,qualification_complete_ids_text_stop_reason_exact=True,
        full_model_output_parity_required=False,
        selection='first_n_original_rows_per_length_band_64k_before_128k',
        no_padding=True,no_truncation=True,global_timing_ordinal_offset=0,
        case_publication_barrier_before_all_rank_reference=True,historical_64k_gpu_measurements_used=False,
        selection_uses_model_outcomes=False,historical_outputs_used_for_admission=False,
        other_evaluation_cases_executed=False,original_s1_executed=False),'future reproduction Full/S1.2 configuration differs')
    fields(preflight,dict(schema='s12-reproduction-run-v1',manifest_sha256=ORIGINAL_MANIFEST_SHA256,
        evaluation_case_indices=config['evaluation_case_indices'],other_evaluation_cases_executed=False,
        length=config['length'],cases_per_length=config['cases_per_length'],
        method_name='S1.2',display_names=DISPLAY_NAMES,profiling_installed=False,formal_s11_retired=True),
        'preflight cohort/display scope differs')
    fields(identity,dict(profile_config_sha256=config['sha256'],s1_protocol_sha256=config['sha256']),
        'actual preflight configuration differs')
    audit_sources(config,identity)
    manifest_path=Path(manifest_path or preflight['original_manifest_path'])
    manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw);rows=original_selection(manifest,config)
    selected=evidence.read('selected-inputs.json')
    fields(selected,dict(schema='s12-selected-original-inputs-v1',original_manifest_sha256=ORIGINAL_MANIFEST_SHA256,
        profile_config_sha256=config['sha256'],rows=rows),'executed cases differ from original deterministic selection')
    require(config.get('case_pins')==[case_pin(row) for row in rows],'config case pin set differs')
    require(preflight.get('audit_cases')==[{k:v for k,v in row.items() if k not in ('input_ids','prompt','context_text')}
        for row in rows],'preflight selected case metadata differs')
    decoder=TokenizerDecoder(tokenizer or Path(identity['checkpoint_path'])/'tokenizer.json',identity['tokenizer_json_sha256'])
    adapted=dict(preflight,protocol=config)
    metadata17=audit_group17_metadata(evidence.read('s12_group17_metadata_qualification.json'),identity)
    qualification=audit_native_qualification(evidence,adapted,metadata17)
    completed=[];pending=[];missing=False
    for row in rows:
        if (evidence.directory/f"case{row['index']}.clean-paired.json").exists():
            require(not missing,'completed whole cases must follow declared execution order')
            completed.append(audit_case(evidence,row,adapted,qualification,metadata17,decoder))
        else:
            missing=True;pending.append(row['index'])
    require(len({c['shared_bank_identity']['bank_generation'] for c in completed})==len(completed),
        'different cases reused the same offline bank generation')
    terminal=terminal_closure(evidence,completed,preflight,metadata17)
    ratios=[c['full_over_s12_speedup'] for c in completed]
    return seal(dict(schema=SCHEMA,**terminal,errors=[],method_name='S1.2',display_names=DISPLAY_NAMES,
        evaluation_case_indices=config['evaluation_case_indices'],completed_case_indices=[c['index'] for c in completed],
        pending_case_indices=pending,completed_case_count=len(completed),evaluated_questions=len(completed),length=config['length'],cases_per_length=config['cases_per_length'],
        by_length_band=summarize_lengths(completed),
        distinct_contexts=len({c['context_sha256'] for c in completed}),cases=completed,
        performance_accepted=bool(completed),all_execution_cases_audited=not pending,raw_run_status_preserved=True,
        full_over_s12_geomean_speedup=math.exp(sum(map(math.log,ratios))/len(ratios)) if ratios else None,
        summed_medians_full_over_s12=(sum(c['clean_timing']['arms']['recomputed']['online_ttft_p50_s'] for c in completed)/
            sum(c['clean_timing']['arms']['s1_system']['online_ttft_p50_s'] for c in completed)) if completed else None,
        whole_case_acceptance_only=True,historical_anchor_included=False,historical_results_used_for_admission=False,
        formal_s11_retired=True,original_s1_executed=False,
        qualification_is_bounded_32_or_earlier_eos=True,full_model_accuracy_not_proven_by_same_policy_parity=True,
        selected_cases_do_not_establish_population_accuracy=True,standard_retriever_evaluation=False,formal_ttft_derived_from_profile=False,
        preflight_sha256=preflight['sha256'],profile_config_sha256=config['sha256'],
        original_manifest_file_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        original_manifest_sha256=manifest['sha256'],method_definition_origin_sha256=SCHEME_DEFINITION_SHA256,
        evidence=evidence.references,tokenizer=decoder.identity,
        reader_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reader_is_external_post_run_cpu_audit=True,offline_payload_bytes_rehashed=False,gpu_lifecycle_cleanup_audited=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path)
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args(argv)
    require(not args.output.exists(),'reader output already exists; preserve immutable prior audit')
    try:result=audit(args.directory,tokenizer=args.tokenizer,manifest_path=args.manifest)
    except Exception as exc:
        result=seal(dict(schema=SCHEMA,status='invalid',errors=[repr(exc)],performance_accepted=False,
            whole_case_acceptance_only=True,gpu_lifecycle_cleanup_audited=False))
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],sha256=result['sha256'],errors=result['errors'])))
    return 0 if result['status'] in ('complete','partial_complete') else 1


if __name__=='__main__':raise SystemExit(main())
