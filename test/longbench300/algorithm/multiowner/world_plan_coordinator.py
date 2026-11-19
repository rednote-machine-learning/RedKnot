"""Fail-closed CPU protocol for real multi-owner DP8 / EP8 model rounds.

Each owner has one request in its local native scheduler. Initial admission uses
one common ceil(N/8192) bucket so native final chunks and deep flush coincide.
No input padding, request concatenation, or local multi-request relabeling.
"""
import hashlib,json

SCHEMA='s13-multiowner-world-plan-v1'
CHUNK=8192
WORLD=8
DEEP=list(range(3,40))


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seal(value):return dict(value,sha256=digest(value))


def validate_world_plan(spec):
    require(type(spec)is dict and spec.get('schema')==SCHEMA
        and spec.get('sha256')==digest({k:v for k,v in spec.items()if k!='sha256'}),'World plan schema/seal differs')
    owners=spec['owners'];ranks=[o['rank']for o in owners]
    require(len(owners)in(1,4,8) and ranks==sorted(set(ranks)) and all(type(r)is int and 0<=r<WORLD for r in ranks)
        and spec['active_ranks']==ranks and spec['request_batch_size']==len(ranks),'B1/B4/B8 unique owner ranks required')
    require(all(type(spec[k])is str and spec[k]for k in ('batch_id','phase','arm')),'World phase identity missing')
    require(len({o['request_id']for o in owners})==len(owners),'Cross-owner request IDs must be unique')
    chunks=[];kinds=[]
    for owner in owners:
        n=owner['input_tokens'];rid=owner['request_id'];sha=owner['input_ids_sha256'];schedule=owner['schedule']
        require(type(n)is int and n>0 and type(rid)is str and rid
            and type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'Owner request identity invalid')
        chunks.append((n+CHUNK-1)//CHUNK);kinds.append('native'if schedule is None else'index20')
        if schedule is not None:
            require(schedule.get('schema')=='index20-first3-finalflush-v1'
                and schedule.get('sha256')==digest({k:v for k,v in schedule.items()if k!='sha256'})
                and schedule['request_id']==rid and schedule['full_input_tokens']==n
                and schedule['input_ids_sha256']==sha and schedule['phase']==spec['phase']
                and schedule['dynamic_deep_groups']is True,'Owner candidate schedule identity differs')
            require(len(schedule['steps'])==chunks[-1],'Native chunk coverage differs')
            for j,step in enumerate(schedule['steps']):
                last=j==chunks[-1]-1
                require(step['start']==j*CHUNK and step['end']==min((j+1)*CHUNK,n)
                    and step['flush']is last and step['layers']==(list(range(40))if last else[0,1,2]),
                    'Candidate first3/final-flush order differs')
    require(len(set(chunks))==1 and spec['native_prefill_rounds']==chunks[0],
        'First prototype requires equal native chunk counts; no implicit input padding')
    require(len(set(kinds))==1 and spec['execution_kind']==kinds[0],
        'A world round cannot mix native Full and sparse candidate schedules')
    return spec


def make_world_plan(batch_id,phase,arm,owners):
    owners=sorted(owners,key=lambda o:o['rank'])
    require(owners,'No active owners')
    result=seal(dict(schema=SCHEMA,batch_id=batch_id,phase=phase,arm=arm,owners=owners,
        active_ranks=[o['rank']for o in owners],request_batch_size=len(owners),world_size=WORLD,
        local_max_requests=1,native_chunk_tokens=CHUNK,native_prefill_rounds=(owners[0]['input_tokens']+CHUNK-1)//CHUNK,
        execution_kind='native'if owners[0]['schedule']is None else'index20',
        input_padding=False,request_concatenation=False))
    return validate_world_plan(result)


def classify_world_step(samples,spec,ordinal,prefill_completed):
    validate_world_plan(spec);owners={o['rank']:o for o in spec['owners']}
    require(type(samples)is list and len(samples)==WORLD,'Eight actual scheduler observations required')
    for rank,sample in enumerate(samples):
        require(sample.get('rank')==rank and sample.get('world_plan_sha256')==spec['sha256']
            and sample.get('ordinal')==ordinal and sample.get('prefill_completed')==prefill_completed,
            'World scheduler identity/progress differs')
        require(type(sample.get('prompt_run'))is bool and type(sample.get('is_empty'))is bool
            and type(sample.get('sequences'))is list and sample['is_empty']==(len(sample['sequences'])==0),
            'Native scheduler emptiness differs from actual sequences')
        require(len(sample['sequences'])<=1,'Local request batching is not this DP-owner prototype')
        require(rank in owners or sample['is_empty'],'Foreign active request on an unowned rank')
    require(len({s['prompt_run']for s in samples})==1,'Native DP vote yielded mixed prompt/decode rounds')
    active=[s for s in samples if not s['is_empty']];kind='idle';ranges={}
    for sample in active:
        owner=owners[sample['rank']];seq=sample['sequences'][0]
        require(seq['request_id']==owner['request_id'] and seq['is_prompt']is sample['prompt_run']
            and type(seq['seqlen_q'])is int and seq['seqlen_q']>0
            and type(seq['seqlen_k'])is int and seq['seqlen_k']>=seq['seqlen_q'],'Foreign request or native query geometry')
        ranges[str(sample['rank'])]=[seq['seqlen_k']-seq['seqlen_q'],seq['seqlen_k']]
    if active:
        kind='prefill'if samples[0]['prompt_run']else'decode'
        if kind=='prefill':
            require(prefill_completed<spec['native_prefill_rounds'] and {s['rank']for s in active}==set(owners),
                'All owner requests must be admitted together before prefill; staggered admission unsupported')
            for rank,owner in owners.items():
                require(ranges[str(rank)]==[prefill_completed*CHUNK,min((prefill_completed+1)*CHUNK,owner['input_tokens'])],
                    'Actual owner prefill chunk differs from its plan')
        else:
            require(prefill_completed==spec['native_prefill_rounds'],'Decode before all-owner final flush')
    return dict(ordinal=ordinal,world_phase=kind,prefill_completed_before=prefill_completed,
        scheduler_prompt_run=samples[0]['prompt_run'],owner_ranges=ranges,
        active_owner_ranks=[s['rank']for s in active],scheduler_observations_sha256=digest(samples),
        world_plan_sha256=spec['sha256'])


def validate_installation(spec,receipts):
    validate_world_plan(spec);owners={o['rank']:o for o in spec['owners']}
    require(len(receipts)==WORLD,'Eight worker installation receipts required')
    for rank,item in enumerate(receipts):
        owner=owners.get(rank)
        require(item.get('schema')=='s13-multiowner-worker-install-v1' and item.get('rank')==rank
            and item.get('world_plan_sha256')==spec['sha256'] and item.get('enabled')is True
            and item.get('owner_request_id')==(owner['request_id']if owner else None),
            'Worker installed another world/owner request')
    return seal(dict(schema='s13-multiowner-installation-proof-v1',world_plan_sha256=spec['sha256'],receipts=receipts))


def agree_deep_groups(items,spec):
    validate_world_plan(spec);require(spec['execution_kind']=='index20','Flush agreement on a native-only round')
    owners={o['rank']:o for o in spec['owners']};counts=[]
    require(len(items)==WORLD,'Eight flush proposals required')
    for rank,item in enumerate(items):
        owner=owners.get(rank);count=item.get('local_groups')
        require(item.get('rank')==rank and item.get('world_plan_sha256')==spec['sha256']
            and item.get('request_id')==(owner['request_id']if owner else None),'Flush proposal owner differs')
        require(type(count)is int and (1<=count<=spec['native_prefill_rounds']if owner else count==0),
            'Invalid owner or inactive deep-group count')
        counts.append(count)
    maximum=max(counts)
    return seal(dict(schema='s13-multiowner-flush-v1',world_plan_sha256=spec['sha256'],
        local_groups_by_rank=counts,world_groups=maximum,empty_groups_by_rank=[maximum-n for n in counts],
        deep_ep_calls_per_rank=37*maximum,all_eight_proposals_valid=True))


def expected_world_layers(spec,round_index,world_groups=None):
    require(0<=round_index<spec['native_prefill_rounds'],'Prefill ordinal outside plan')
    if spec['execution_kind']=='native':return list(range(40))
    if round_index<spec['native_prefill_rounds']-1:return [0,1,2]
    require(type(world_groups)is int and world_groups>0,'Final sparse flush lacks agreement')
    return [0,1,2]+DEEP*world_groups


def validate_worker_reports(reports,spec):
    validate_world_plan(spec);require(len(reports)==WORLD,'Eight actual worker execution reports required')
    reference=None;flush=None
    for rank,report in enumerate(reports):
        require(report.get('sha256')==digest({k:v for k,v in report.items()if k!='sha256'})
            and report.get('schema')=='s13-multiowner-worker-execution-v1'and report.get('rank')==rank
            and report.get('enabled')is True and report.get('world_plan_sha256')==spec['sha256']
            and report.get('prefill_rounds')==spec['native_prefill_rounds'],'Worker execution identity/coverage differs')
        sequence=[(e['ordinal'],e['world_phase'],e['scheduler_observations_sha256'])for e in report['world_step_trace']]
        if reference is None:reference=sequence
        require(sequence==reference,'Ranks observed different native model rounds')
        require(sum(e[1]=='prefill'for e in sequence)==spec['native_prefill_rounds'],'Native prefill count differs')
        agreement=report['flush_agreement']
        if spec['execution_kind']=='index20':
            require(agreement is not None and agreement['world_plan_sha256']==spec['sha256'],'Sparse report missing flush proof')
            if flush is None:flush=agreement
            require(agreement==flush,'Ranks used different actual deep-group counts')
        else:require(agreement is None,'Native baseline unexpectedly ran sparse flush')
        traces=report['prefill_trace'];require(len(traces)==spec['native_prefill_rounds'],'Missing per-prefill execution trace')
        for j,event in enumerate(traces):
            require(event['round']==j and event['layers']==expected_world_layers(spec,j,agreement['world_groups']if agreement else None)
                and event['real_owner']is(rank in spec['active_ranks']),'Actual per-rank layer order/owner differs')
            if agreement and j==spec['native_prefill_rounds']-1:
                require(event['local_deep_groups']==agreement['local_groups_by_rank'][rank]
                    and event['empty_deep_groups']==agreement['empty_groups_by_rank'][rank],'Actual padding group count differs')
    return seal(dict(schema='s13-multiowner-world-execution-proof-v1',world_plan_sha256=spec['sha256'],
        status='passed',passed=True,all_eight_workers_agreed=True,actual_request_batch_size=len(spec['owners']),local_max_requests=1,
        native_prefill_rounds=spec['native_prefill_rounds'],flush_agreement=flush,
        scheduler_observation_rounds=len(reference),gpu_quality_or_performance_claim=False))
