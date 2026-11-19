"""Independent all-rank transport over the actual NativeEnginePort API.

All eight ranks enter every native unfinished/step/flush collective. Each active
owner submits and consumes exactly one local request. This module does not
change an engine, model, controller, scheduler or graph. Any failure is fatal to
the owned worker group; the original supervisor must drain/clean that group.
"""
from __future__ import annotations
import hashlib
import json
import math
import time

from service.engine_bridge import RequestMetrics,token_digest

WORLD_SIZE=8


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _error(error):return dict(error_type=type(error).__name__,error=str(error))


def _requests(requests_by_rank,active_ranks):
    given=tuple(active_ranks)
    require(len(given)in (1,2,4,8)and len(set(given))==len(given)
        and all(type(owner)is int and 0<=owner<WORLD_SIZE for owner in given),
        'Declare unique valid owners with cardinality1/2/4/8')
    active=tuple(sorted(given))
    if isinstance(requests_by_rank,dict):
        require(all(type(k)is int for k in requests_by_rank),'Integer request owner ranks required')
        table={k:v for k,v in requests_by_rank.items()if v is not None}
    else:
        require(isinstance(requests_by_rank,(list,tuple))and len(requests_by_rank)==WORLD_SIZE,
                'Request table must be owner mapping or eight-rank sequence')
        table={rank:request for rank,request in enumerate(requests_by_rank)if request is not None}
    require(set(table)==set(active),'Exactly one request per declared active owner required')
    descriptors=[]
    for owner in active:
        request=table[owner];request.validate()
        require(type(request.request_id)is str and request.request_id,'Actual native request ID must be nonempty text')
        descriptor=dict(owner_rank=owner,request_id=request.request_id,arm=request.arm,
            input_tokens=len(request.prompt_token_ids),input_ids_sha256=token_digest(request.prompt_token_ids),
            max_tokens=request.max_tokens,received_s=request.received_s,temperature=request.temperature,
            top_p=request.top_p,seed=request.seed,documents_sha256=digest(request.documents),
            explicit_resolution_sha256=digest(request.explicit_resolution))
        descriptors.append(descriptor)
    require(len({d['request_id']for d in descriptors})==len(active),'Owner request IDs must be globally unique')
    require(len({d['arm']for d in descriptors})==1,'All owners must execute the same native arm')
    return active,table,descriptors


def _gather(gather,sample,*,stage):
    reports=gather(sample)
    require(type(reports)is list and len(reports)==WORLD_SIZE,'Actual eight-rank '+stage+' gather required')
    require(all(type(report)is dict and report.get('rank')==rank for rank,report in enumerate(reports)),
            'Actual '+stage+' reports must be in rank order')
    errors=[dict(rank=r['rank'],**r['error'])for r in reports if r.get('passed')is not True]
    require(not errors,'Multiowner '+stage+' failed: '+json.dumps(errors,ensure_ascii=False))
    return reports


def run_multiowner_requests(port,requests_by_rank,*,rank,active_ranks,barrier,gather,
                            prepare,finish,emit,clock=time.perf_counter):
    """Return this owner's original request report, or None on inactive ranks.

    requests_by_rank is identical on all ranks. prepare(local_request_or_None)
    and finish(local_request_or_None) are called by EVERY rank. prepare must
    return a dict containing common multiowner_phase_id and multiowner_method;
    native arm='s2' alone does not distinguish S1.2 from S1.3 strategies.
    emit is a local nonblocking callback. It must not wait on a network peer.

    There are three host all-gathers (identity, preparation, final-drain proof),
    never an added host collective per generated token. NativeEnginePort's
    has_unfinished already performs the real world all-reduce.
    """
    require(type(rank)is int and 0<=rank<WORLD_SIZE,'Actual local DP rank required')
    require(all(callable(fn)for fn in (barrier,gather,prepare,finish,emit,clock)),
            'Actual world coordination and local callbacks required')
    admitted=clock();local_error=None;active=table=descriptors=None
    try:active,table,descriptors=_requests(requests_by_rank,active_ranks)
    except Exception as error:local_error=_error(error)
    identity_samples=_gather(gather,dict(rank=rank,passed=local_error is None,error=local_error,
        table_sha256=None if descriptors is None else digest(descriptors),
        active_ranks=None if active is None else list(active)),stage='request_identity')
    require(len({sample['table_sha256']for sample in identity_samples})==1
        and all(sample['active_ranks']==list(active)for sample in identity_samples),
        'All ranks must use the same owner request identities and active-rank map')
    request=table.get(rank)
    metrics=None if request is None else RequestMetrics(request.request_id,request.arm,request.received_s,admitted)
    preparation=None;local_error=None
    try:
        preparation=prepare(request)
        require(type(preparation)is dict,'All ranks must return preparation metadata')
        for name in ('multiowner_phase_id','multiowner_method'):
            require(type(preparation.get(name))is str and preparation[name],
                    'Preparation must bind common '+name)
        digest(preparation)  # Require transferable metadata, without copying payloads.
    except Exception as error:local_error=_error(error)
    preparation_samples=_gather(gather,dict(rank=rank,passed=local_error is None,error=local_error,
        phase=preparation.get('multiowner_phase_id')if type(preparation)is dict else None,
        method=preparation.get('multiowner_method')if type(preparation)is dict else None),stage='preparation')
    require(len({sample['phase']for sample in preparation_samples})==1
        and len({sample['method']for sample in preparation_samples})==1,
        'All ranks must execute one common logical method and phase')
    barrier()
    if metrics is not None:
        metrics.prepared_s=clock();metrics.submitted_s=clock();port.submit(request)
    # Native has_unfinished is collective: nobody may enter before all local
    # owner enqueues finish. Inactive ranks submit nothing but still step.
    barrier()
    terminal=False;steps=0;flushes=0
    while True:
        unfinished=port.has_unfinished()
        require(type(unfinished)is bool,'Actual native world unfinished predicate must be bool')
        outputs=port.advance(unfinished);arrived=clock();steps+=int(unfinished);flushes+=int(not unfinished)
        require(isinstance(outputs,(list,tuple)),'Native output must be an already copied local list')
        require(request is not None or not outputs,'Inactive rank emitted a real request')
        for output in outputs:
            require(output['request_id']==request.request_id and not terminal,'Foreign, cross-owner or late native output')
            require(output['prompt_len']==len(request.prompt_token_ids),'Native owner prompt length changed')
            require(type(output['token_id'])is int and 0<=output['token_id']<2**31,'Native output token is not int32')
            require(type(output['finished'])is bool and math.isfinite(output['logprob']),'Invalid native output status/logprob')
            if metrics.first_raw_token_s is None:metrics.first_raw_token_s=arrived
            metrics.output_ids.append(output['token_id'])
            require(len(metrics.output_ids)<=request.max_tokens,'Native owner output exceeded its original budget')
            emit(dict(event='raw_token',owner_rank=rank,arrival_s=arrived,**output))
            if output['finished']:
                terminal=True;metrics.terminal_s=arrived;metrics.finish_reason=output['finish_reason']
        if not unfinished:break  # The real native ZOS flush ran on every rank.
    local_ok=request is None or (terminal and metrics.finish_reason in ('stop','length'))
    drain_samples=_gather(gather,dict(rank=rank,passed=local_ok,
        error=None if local_ok else dict(error_type='IncompleteOwner',error='World drained without a successful local terminal'),
        owner_rank=rank if request is not None else None,request_id=None if request is None else request.request_id,
        native_steps=steps,final_flush_calls=flushes,terminal=terminal,
        output_tokens=0 if metrics is None else len(metrics.output_ids)),stage='global_drain')
    require(len({(sample['native_steps'],sample['final_flush_calls'])for sample in drain_samples})==1
        and flushes==1,'World ranks did not execute the same native step/final flush sequence')
    require([sample['owner_rank']for sample in drain_samples if sample['owner_rank']is not None]==list(active),
            'World drained without all active owners')
    for sample in drain_samples:
        if sample['owner_rank']is not None:
            require(sample['request_id']==table[sample['owner_rank']].request_id,
                    'Global drain report belongs to another owner request')
    completion=finish(request)
    require(completion is None or type(completion)is dict,'Finish metadata must be mapping or None')
    barrier()
    if metrics is None:return None
    metrics.drained_s=clock();result=metrics.report(request)
    result.update(preparation=preparation,completion=completion,multiowner_transport=dict(
        schema='s13-multiowner-native-transport-v1',owner_rank=rank,active_ranks=list(active),
        request_table_sha256=identity_samples[0]['table_sha256'],request_id_by_rank={str(d['owner_rank']):d['request_id']for d in descriptors},
        global_phase_id=preparation_samples[0]['phase'],global_method=preparation_samples[0]['method'],
        native_world_steps=steps,final_native_flush_calls=flushes,all_rank_global_drain_verified=True,
        global_drain_reports=drain_samples,extra_per_decode_step_host_gathers=0,
        owner_terminal_latency_s=metrics.terminal_s-request.received_s,
        request_elapsed_includes_world_drain_and_finish=True,saturated_qps_claim=False,
        actual_gpu_multiowner_qualification_claimed=False))
    emit(dict(event='request_complete',owner_rank=rank,report=result));return result
