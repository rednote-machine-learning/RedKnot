"""Validate S1 operator qualifications and eight-rank scheduler observations."""

from kv_cache_engram.core import digest, require, validate_digest

def sealed(value):
    require(type(value) is dict and value.get("sha256") == digest(
        {key: item for key, item in value.items() if key != "sha256"}), "canonical seal mismatch")
    return value

def _fields(actual, expected, context):
    for field, value in expected.items():
        matches = actual.get(field) is value if type(value) is bool else actual.get(field) == value
        require(matches, context + ": " + field + " differs from independent expectation")

RUNTIME_CHECK_NAMES=('decoder_hidden_gather','four_mhc_fields_axis1','query_metadata_preserves_key_axes',
    'empty_packed_sf_abi','noncontiguous_absolute_positions','sampler_final_row_rebind',
    'engram_embedding_gather_and_empty_hook_consumption')


def validate_runtime_qualification(value,identity,empty_ep_sha256):
    sealed(value)
    _fields(value,dict(schema='s1-runtime-all-ranks-qualification-v1',passed=True,
        execution_configuration_sha256=identity['execution_configuration_sha256'],
        model_sha256=identity['checkpoint_index_sha256'],empty_ep_qualification_sha256=empty_ep_sha256,
        quality_eligible=False,timing_valid=False),'actual eight-rank S1 runtime qualification')
    reports=value.get('reports')
    require(type(reports) is list and len(reports)==8,'all eight actual GPU runtime reports required')
    for rank,report in enumerate(reports):
        sealed(report)
        _fields(report,dict(schema='s1-runtime-rank-qualification-v1',rank=rank,world_size=8,passed=True,
            execution_configuration_sha256=identity['execution_configuration_sha256'],model_sha256=identity['checkpoint_index_sha256'],
            cuda_exercised=True,quality_eligible=False,timing_valid=False,native_empty_ep_requires_separate_all_ranks_proof=True),
            'S1 rank-specific GPU ABI qualification')
        checks=report.get('checks')
        require(type(checks) is list and [e.get('name') for e in checks]==list(RUNTIME_CHECK_NAMES),
            'S1 runtime qualification coverage differs')
        for e in checks:
            require(e.get('passed') is True and e.get('actual_rows')==(0 if e['name']=='empty_packed_sf_abi' else 3)
                and type(e.get('cuda_tensor_device')) is str and e['cuda_tensor_device'].startswith('cuda'),
                'S1 runtime check did not exercise actual expected CUDA rows')
        validate_digest(report.get('output_sha256'),'S1 runtime gathered GPU output')
    return value


def audit_world_scheduler(reports,schedule):
    """Rebuild phase/progress from CPU scheduler observations, not local tensors."""
    traces=[report.get('world_step_trace') for report in reports]
    require(len(traces)==8 and all(type(t) is list for t in traces)
        and len(traces[0])>=len(schedule['steps'])
        and all(len(t)==len(traces[0]) for t in traces),'all-rank world scheduler coverage differs')
    completed=0;counts=dict(idle=0,prefill=0,decode=0)
    for ordinal,first in enumerate(traces[0]):
        samples=first.get('scheduler_observations')
        require(type(samples) is list and len(samples)==8,'raw eight-rank scheduler observations required')
        for rank,sample in enumerate(samples):
            _fields(sample,dict(rank=rank,schedule_sha256=schedule['sha256'],ordinal=ordinal,
                prefill_completed=completed),'raw scheduler rank/phase binding')
            require(type(sample.get('prompt_run')) is bool and type(sample.get('is_empty')) is bool
                and type(sample.get('sequences')) is list
                and sample['is_empty']==(not sample['sequences']),'raw scheduler emptiness differs')
        prompt=samples[0]['prompt_run'];seqs=samples[0]['sequences']
        require(all(sample['prompt_run'] is prompt for sample in samples)
            and all(sample['is_empty'] for sample in samples[1:]),'native DP world phase/ownership differs')
        kind='idle';start=end=rid=None
        if seqs:
            require(len(seqs)==1,'single active request required')
            seq=seqs[0];rid=seq.get('request_id')
            require(rid==schedule['request_id'] and seq.get('is_prompt') is prompt
                and type(seq.get('seqlen_q')) is int and seq['seqlen_q']>0
                and type(seq.get('seqlen_k')) is int and seq['seqlen_k']>=seq['seqlen_q'],
                'scheduler active request identity/length differs')
            if prompt:
                kind='prefill'
                require(completed<len(schedule['steps']),'extra world prefill')
                start,end=seq['seqlen_k']-seq['seqlen_q'],seq['seqlen_k']
                step=schedule['steps'][completed]
                require((start,end)==(step['start'],step['end']),'world prefill chunk order differs')
            else:
                kind='decode'
                require(completed==len(schedule['steps']),'real decode before final batch flush')
        expected=dict(ordinal=ordinal,world_phase=kind,prefill_completed_before=completed,
            start=start,end=end,scheduler_prompt_run=prompt,rank0_request_id=rid,
            scheduler_observations_sha256=digest(samples),model_execution_completed=True)
        for rank,trace in enumerate(traces):
            event=trace[ordinal]
            _fields(event,dict(**expected,local_prompt_run=prompt,
                local_is_empty=samples[rank]['is_empty']),'actual rank world step differs')
            if 'scheduler_observations' in event:
                require(event['scheduler_observations']==samples,'raw world observations differ across ranks')
        counts[kind]+=1
        if kind=='prefill':completed+=1
    require(completed==len(schedule['steps']),'world prefill coverage incomplete')
    return dict(actual_world_steps=len(traces[0]),actual_idle_steps=counts['idle'],
        actual_prefill_steps=counts['prefill'],actual_decode_steps=counts['decode'],
        idle_steps_use_original_native_execution=True,idle_steps_advance_batch=False)
