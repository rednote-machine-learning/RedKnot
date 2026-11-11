"""Read actual S2/Full/Prefix artifacts without executing services or evaluators.

No result is produced until a sealed triplet terminal exists. Missing measurements
remain null; independent agent trajectories are never converted to causal speedups.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS={'Full':'full','Prefix':'prefix','S2':'s2'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def sealed(value):
    require(type(value) is dict and value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'artifact seal differs')
    return value


def ref(path,value=None):
    path=Path(path);raw=path.read_bytes()
    result=dict(path=str(path.resolve()),file_sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value,dict) and 'sha256' in value:result['seal_sha256']=sealed(value)['sha256']
    return result


def read(path,*,verify=True):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular artifact missing: '+str(path))
    value=json.loads(path.read_bytes())
    if verify:sealed(value)
    return value


def number(value):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,'invalid measurement')
    return float(value)


def distribution(values):
    if not values:return dict(n=0,p50=None,p95=None,min=None,max=None,quantile='linear_interpolation')
    values=sorted(map(number,values));position=(len(values)-1)*.95;low=int(position);high=min(low+1,len(values)-1)
    return dict(n=len(values),p50=statistics.median(values),p95=values[low]+(values[high]-values[low])*(position-low),
                min=values[0],max=values[-1],quantile='linear_interpolation')


def nullable_sum(values):
    return None if any(v is None for v in values) else sum(values)


def token_sha(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def formal_for(record,service):
    reference=record.get('completion',{}).get('formal_evidence')
    if not reference:return None,None
    expected=record['request_id']+'.formal.json'
    require(Path(reference['path']).name==expected,'formal reference belongs to another request')
    path=service/expected
    if not path.is_file():return None,dict(missing=str(path),expected_seal_sha256=reference['sha256'])
    value=read(path)
    require(value['sha256']==reference['sha256'] and value.get('request_id')==record['request_id']
            and value.get('input_ids_sha256')==record['input_ids_sha256'],'formal request binding differs')
    return value,ref(path,value)


def actual_rows(formal,record):
    if formal is None or not formal.get('scheduled_rows_reports'):return None,None
    reports=formal['scheduled_rows_reports'];require(len(reports)==8,'all8 actual row receipts required')
    for rank,r in enumerate(reports):
        sealed(r)
        require(r.get('schema')=='s2-native-scheduled-rows-v1' and r.get('rank')==rank
                and r.get('request_id')==record['request_id'] and r.get('input_tokens')==record['input_tokens'],
                'rank native row receipt detached')
        if rank:require(r['scheduled_prefill_rows']==0 and r['initial_cached_tokens']==0 and r['prefill_ranges']==[],
                        'inactive rank reports real query rows')
    r=reports[0];cursor=r['initial_cached_tokens'];require(type(cursor) is int and cursor>=0,'invalid actual native prefix count')
    for interval in r['prefill_ranges']:
        require(interval['q_start']==cursor and interval['q_end']>cursor and interval['rows']==interval['q_end']-cursor,
                'native query ranges overlap or leave a gap')
        cursor=interval['q_end']
    require(cursor==record['input_tokens'] and r['scheduled_prefill_rows']==cursor-r['initial_cached_tokens'],
            'native query row sum differs')
    return r['scheduled_prefill_rows'],r['initial_cached_tokens']


def complete_request(record,service):
    require(record.get('schema')=='s2-live-native-request-v1' and record.get('live_engine') is True
            and record.get('replay') is False,'not an actual online native request')
    ids=record['output_ids'];require(type(ids) is list and ids and all(type(x) is int and x>=0 for x in ids), 'actual output IDs missing')
    require(record['output_tokens']==len(ids) and record['output_ids_sha256']==token_sha(ids),'actual output token digest/count differs')
    times=[number(record[k]) for k in ('received_s','admitted_s','prepared_s','submitted_s','first_raw_token_s','terminal_s','drained_s')]
    require(times==sorted(times),'actual native clocks are not ordered')
    for key,actual in [('raw_ttft_s',times[4]-times[0]),('engine_raw_ttft_s',times[4]-times[3]),('request_elapsed_s',times[6]-times[0])]:
        require(abs(number(record[key])-actual)<1e-5,'latency summary disagrees with raw timestamps')
    formal,evidence=formal_for(record,service);rows,cached=actual_rows(formal,record)
    if record['arm'] in ('full','s2') and cached is not None:require(cached==0,'native prefix appeared in disabled arm')
    prep=record.get('preparation') or {};reuse=prep.get('reuse') or {};prefix=None if formal is None else formal.get('prefix_report')
    captures=restore_bytes=None
    if prefix is not None:
        sealed(prefix);require(prefix.get('request_id')==record['request_id'] and prefix['input_ids_sha256']==record['input_ids_sha256'],
                               'prefix report detached')
        require(cached is None or cached==prefix['reused_prefix_tokens'],'native rows disagree with prefix copy count')
        captures=sum(c['kv_bytes']+c['state_bytes'] for c in prefix['captures'] if c.get('copy_complete') is True)
        restore=prefix.get('restore');restore_bytes=0 if restore is None else restore['kv_bytes']+restore['state_bytes']
    groups=None if formal is None else (formal.get('controller') or {}).get('system_batch_execution')
    selected_rows=None if not groups else sum(g['selected_rows'] for g in groups['groups'])
    actual=prep.get('actual_resident_bank') or {}
    growth=None
    if type(actual.get('resident_bank_bytes')) is int and type(prep.get('prior_bank_bytes')) is int:
        growth=actual['resident_bank_bytes']-prep['prior_bank_bytes'];require(growth>=0,'S2 no-eviction bank shrank')
    native_reason=record.get('native_finish_reason')
    # Older producer overwrote native reason with the parsed chat reason. The
    # output IDs still directly identify EOS, without inventing a native reason.
    eos=ids[-1]==1 and ids.count(1)==1
    budget=record.get('generation_token_budget');reason=native_reason or record.get('finish_reason')
    capped=reason=='length'
    cap_kind=('native_context_capacity' if record.get('generation_budget_source')=='remaining_native_context_capacity'
              else 'explicit_client_token_cap') if capped else None
    def latency(key):
        value=record.get(key)
        if value is None:return None
        value=number(value)-times[0];require(value>=0,'transport timestamp precedes admission');return value
    return dict(request_id=record['request_id'],arm=record['arm'],input_ids_sha256=record['input_ids_sha256'],
        input_tokens=record['input_tokens'],output_tokens=len(ids),output_ids_sha256=record['output_ids_sha256'],
        native_ttft_s=record['raw_ttft_s'],engine_ttft_s=record['engine_raw_ttft_s'],
        native_request_elapsed_s=record['request_elapsed_s'],first_parsed_delta_s=latency('first_delta_s'),
        first_parsed_content_s=latency('first_content_delta_s'),server_first_http_flush_s=latency('first_http_flush_s'),
        client_first_byte_s=None,client_complete_response_s=None,received_s=times[0],drained_s=times[-1],
        native_finish_reason=native_reason,chat_finish_reason=record.get('chat_finish_reason',record.get('finish_reason')),
        natural_eos_observed_in_ids=eos,truncated_by_token_budget=capped,token_budget_kind=cap_kind,
        generation_token_budget=budget,effective_arm=prep.get('effective_arm'),fallback_reason=prep.get('fallback_reason'),
        native_scheduled_prefill_rows=rows,native_cached_prefix_rows=cached,
        s2_group_selected_query_rows=selected_rows,
        s2_eligible_document_tokens=sum(d['tokens'] for d in record.get('document_spans',[]) if d['tokens']>=256) if record['arm']=='s2' else None,
        s2_hit_document_tokens=reuse.get('eligible_hit_document_tokens'),
        s2_prefix_compatible_hit_document_tokens=reuse.get('prefix_compatible_hit_document_tokens'),
        s2_nonprefix_hit_document_tokens=reuse.get('nonprefix_hit_document_tokens'),s2_reuse_by_source_kind=reuse.get('by_source_kind'),
        s2_cold_capture_s=prep.get('cold_capture_s') if record['arm']=='s2' else None,
        online_qualification_s=prep.get('online_qualification_s'),s2_retained_bank_growth_bytes=growth,
        offline_build_copied_bytes=None,prefix_snapshot_capture_bytes=captures,prefix_restore_bytes=restore_bytes,
        prefix_copy_s=None,unique_document_misses=prep.get('unique_misses'),
        prefix_bank_bytes=None if prefix is None else prefix['bank_bytes_after'],
        s2_bank_bytes=actual.get('resident_bank_bytes'),formal_evidence=evidence)


def official_grade(path,instance_id,arm,agent_patch_hashes):
    if path is None:return dict(resolved=None,status='not_supplied',source=None)
    value=read(path,verify=False)
    require(type(value) is dict and instance_id in value,'official per-instance result missing')
    result=value[instance_id]
    require(type(result) is dict and type(result.get('resolved')) is bool,'official resolved boolean missing')
    receipt_path=Path(path).parent/'offline-grader-receipt.json';receipt=None;verified=False;prediction_bound=False;prediction_ref=None
    if receipt_path.is_file():
        receipt=read(receipt_path)
        require(receipt.get('schema')=='s2-official-offline-grader-v1' and receipt.get('arm')==arm
                and receipt.get('instance_id')==instance_id,'grader receipt belongs to another arm/task')
        require(receipt.get('reports_sha256')==ref(path)['file_sha256'],'official grade bytes detached from receipt')
        verified=(receipt.get('status')=='official_evaluation_returned' and receipt.get('evaluator')=='swebench==4.1.0'
                  and receipt.get('official_grading_functions_unchanged') is True)
        if receipt.get('predictions_path'):
            prediction=Path(receipt['predictions_path'])
            if not prediction.is_file():prediction=Path(path).parent/prediction.name
            if prediction.is_file():
                prediction_ref=ref(prediction)
                require(prediction_ref['file_sha256']==receipt['predictions_sha256'],'graded prediction bytes changed')
                rows=[json.loads(line) for line in prediction.read_text().splitlines() if line.strip()]
                require(len(rows)==1 and rows[0].get('instance_id')==instance_id and type(rows[0].get('model_patch')) is str,
                        'graded prediction task/patch missing')
                prediction_bound=hashlib.sha256(rows[0]['model_patch'].encode()).hexdigest() in agent_patch_hashes
                require(not agent_patch_hashes or prediction_bound,'official grader evaluated a different agent patch')
    return dict(resolved=result['resolved'] if verified and prediction_bound else None,reported_resolved=result['resolved'],
                status='official_report_verified' if verified and prediction_bound else 'grader_provenance_or_prediction_binding_incomplete',source=ref(path),
                prediction_binding_verified=prediction_bound,graded_prediction=prediction_ref,
                grader_receipt=ref(receipt_path,receipt) if receipt else None,
                instance_id=instance_id,test_results=result.get('tests_status'),
                provenance_limit='Reads supplied official per-instance report; does not rerun tests or infer success from agent FinishAction.')


def read_arm(root,label,arm_record,instance_id,grade_path=None):
    service=root/'arms'/label/'service';journal=service/'requests.jsonl'
    records=[];failures=[];admissions=[];unknown=[];sources=[];trailing_fragment=False;chat_complete=0
    if journal.is_file():
        sources.append(ref(journal));seen=set()
        text=journal.read_text();lines=text.splitlines()
        for ordinal,line in enumerate(lines):
            if not line.strip():continue
            try:value=sealed(json.loads(line))
            except json.JSONDecodeError:
                if ordinal==len(lines)-1 and not text.endswith('\n'):
                    trailing_fragment=True;break
                raise
            schema=value.get('schema')
            require(value.get('arm')==ARMS[label],'request arm differs')
            if schema=='s2-live-native-request-v1':
                require(value['request_id'] not in seen,'duplicate closed native request')
                seen.add(value['request_id']);records.append(dict(complete_request(value,service),chat_transport_failed=False));chat_complete+=1
            elif schema=='s2-live-chat-failure-v1':
                failures.append(value)
                if value.get('native_report'):
                    native=value['native_report'];require(native['request_id']==value['request_id'] and native['arm']==ARMS[label],
                                                          'failed transport embeds another native request')
                    require(native['request_id'] not in seen,'duplicate failure/native completion')
                    seen.add(native['request_id']);records.append(dict(complete_request(native,service),chat_transport_failed=True))
            elif schema=='s2-live-chat-admission-v1':admissions.append(value)
            else:unknown.append(schema)
    ready=terminal=None
    for name in ('service-ready.json','service-terminal.json'):
        path=service/name
        if path.is_file():
            value=read(path);require(value.get('arm')==ARMS[label],'service arm differs');sources.append(ref(path,value))
            if name=='service-ready.json':ready=value
            else:terminal=value
    window=None
    if ready is not None and terminal is not None:
        window=number(terminal['finished_unix'])-number(ready['ready_unix']);require(window>=0,'service wall interval reversed')
    whole_wall=None
    if 'started_unix' in arm_record and 'finished_unix' in arm_record:
        whole_wall=number(arm_record['finished_unix'])-number(arm_record['started_unix']);require(whole_wall>=0,'arm clock reversed')
    metrics={key:distribution([r[key] for r in records if r[key] is not None]) for key in (
        'native_ttft_s','engine_ttft_s','native_request_elapsed_s','first_parsed_delta_s',
        'first_parsed_content_s','server_first_http_flush_s')}
    totals={key:nullable_sum([r[key] for r in records]) for key in ('input_tokens','output_tokens',
        'native_scheduled_prefill_rows','native_cached_prefix_rows','s2_hit_document_tokens',
        's2_prefix_compatible_hit_document_tokens','s2_nonprefix_hit_document_tokens','s2_cold_capture_s',
        's2_retained_bank_growth_bytes','prefix_snapshot_capture_bytes','prefix_restore_bytes','s2_group_selected_query_rows')}
    denom=totals['input_tokens']
    fractions={key:(totals[key]/denom if denom and totals[key] is not None else None) for key in (
        'native_cached_prefix_rows','s2_hit_document_tokens','s2_prefix_compatible_hit_document_tokens','s2_nonprefix_hit_document_tokens')}
    predictions=[];patch_hashes=[]
    for path in sorted((root/'agent_runs'/label).rglob('output.jsonl')) if (root/'agent_runs'/label).exists() else []:
        output=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(all(row.get('instance_id')==instance_id for row in output),'agent output instance differs')
        for row in output:
            patch=row.get('model_patch')
            if patch is None and type(row.get('test_result')) is dict:patch=row['test_result'].get('git_patch')
            if type(patch) is str:patch_hashes.append(hashlib.sha256(patch.encode()).hexdigest())
        predictions.append(dict(source=ref(path),records=len(output),
            errors=[row.get('error') for row in output],model_patch_sha256=list(patch_hashes)))
    startup=None
    if ready is not None and terminal is not None and terminal.get('started_unix') is not None:
        startup=number(ready['ready_unix'])-number(terminal['started_unix']);require(startup>=0,'service startup interval reversed')
    return dict(arm=label,service_arm=ARMS[label],status=arm_record.get('status','not_started'),instance_id=instance_id,
        closed_native_requests=len(records),logged_chat_complete_requests=chat_complete,logged_failures=len(failures),
        failures_without_closed_native_report=sum(not f.get('native_report') for f in failures),
        incomplete_trailing_journal_fragment=trailing_fragment,
        admitted_requests=len(admissions) if admissions else None,
        unlogged_inflight_at_stop_possible=not bool(admissions),unknown_journal_schemas=unknown,
        latency=metrics,totals=totals,token_fractions=fractions,
        token_totals_scope='Observed closed native requests, including fully drained native work whose chat transport failed; unlogged/incomplete native requests are not estimated.',
        reuse_count_limit='S2 hit-document tokens describe prior resident document coverage, not skipped work in every layer: first three layers and selected tail/non-document rows still execute. Group selected-query rows come from group execution receipts and are not a model-FLOPs estimate.',
        fallback_requests=sum(r['fallback_reason'] is not None for r in records),
        fallback_reasons={reason:sum(r['fallback_reason']==reason for r in records) for reason in sorted({r['fallback_reason'] for r in records if r['fallback_reason']})},
        natural_eos_requests=sum(r['natural_eos_observed_in_ids'] for r in records),
        token_capped_requests=sum(r['truncated_by_token_budget'] for r in records),
        arm_timed_out=arm_record.get('status')=='timed_out',
        service_observation_window_s=window,service_startup_including_model_and_qualification_s=startup,
        whole_arm_including_setup_cleanup_s=whole_wall,precise_agent_task_wall_s=(arm_record['agent_finished_unix']-arm_record['agent_started_unix']) if all(k in arm_record for k in ('agent_finished_unix','agent_started_unix')) else None,
        completed_native_requests_per_service_second=len(records)/window if window else None,
        qps_definition='Closed native requests / (service-terminal.finished_unix - service-ready.ready_unix); includes tool/client idle, errors, tail time and shutdown. Not saturated serving capacity or verified client completion rate.',
        completed_native_requests_per_whole_arm_second=len(records)/whole_wall if whole_wall else None,
        client_first_byte_latency=None,client_end_to_end_latency=None,
        offline_build_copied_bytes=None,prefix_copy_time_s=None,
        unsupported_measurements=['client first-byte and complete-response latency','separate Prefix(copy) copy time','S2 actual offline copy bytes'],
        official_grade=official_grade(grade_path,instance_id,label,patch_hashes),agent_predictions=predictions,
        raw_failures=failures,requests=records,sources=sources)


def read_triplet(directory,grades=None):
    root=Path(directory).resolve();path=root/'triplet-terminal.json';terminal=read(path)
    require(terminal.get('schema')=='s2-agent-triplet-terminal-v1','not an actual triplet terminal')
    grades=grades or {};known={a['arm']:a for a in terminal['arms']}
    require(set(known)<=set(ARMS) and len(known)==len(terminal['arms']),'duplicate or unknown arm terminal')
    arms={label:read_arm(root,label,known.get(label,{}),terminal['instance_id'],grades.get(label)) for label in ARMS}
    body=dict(schema='s2-agent-triplet-independent-readout-v1',status='read_complete',
        raw_triplet_status=terminal['status'],instance_id=terminal['instance_id'],executed_ordinal=terminal['executed_ordinal'],
        requested_arms=list(ARMS),arms=arms,source_triplet=ref(path,terminal),
        original_round_released_unix=terminal.get('released_unix'),absolute_deadline_unix=terminal.get('absolute_deadline_unix'),
        cross_arm_causal_speedups=None,paired_same_input_speedups=None,
        comparison_limit='Independent live agent trajectories may differ. No same-input causal speedup is computed; paired input-ID, order and sampling bindings would be required.',
        single_task_not_benchmark_wide=True,official_success_inferred_from_finish_action=False,
        gpu_executed_by_reader=False,evaluator_rerun=False)
    return dict(body,sha256=digest(body))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grade',action='append',default=[],metavar='ARM=OFFICIAL_REPORT_JSON')
    args=p.parse_args(argv);grades={}
    for item in args.grade:
        arm,path=item.split('=',1);require(arm in ARMS and arm not in grades,'duplicate or unknown grade arm');grades[arm]=Path(path)
    result=read_triplet(args.directory,grades)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],sha256=result['sha256'],raw_triplet_status=result['raw_triplet_status'])))


if __name__=='__main__':main()
