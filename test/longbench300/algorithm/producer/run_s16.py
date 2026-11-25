"""Resident DeepSeek V4.1 Flash B4 runner for the matched Full/S1.6 LongBench workload."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import MethodType

ARTIFACT = '9b98d633db6e003a8d0af58f1af1c4497bec9a26e66daa1b503dbe315409a2d1'
ARMS = {'Full':'Full', 'S1.6':'Cone15'}
CHUNK = 8192
LIMIT = 270336


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    data=dict(obj,sha256=digest(obj));tmp=p.with_name(f'{p.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False))
    tmp.replace(p)
    return data


def load_template(p):
    cfg=read(p);seal=cfg.pop('sha256');check(seal==digest(cfg)==ARTIFACT,'Expected frozen four-finance S1.6 source')
    check(cfg['arms']==['Full','Fixed15Opt.max','Cone15.max'] and cfg['dspark_nextn']==5 and cfg['main_and_draft_graphs_required'] and cfg['multiowner_active_ranks']==[0,1,2,3] and not cfg['padding_allowed'],'Changed native S1.6 profile')
    cfg['sha256']=seal
    return cfg


def group_rows(data,path):
    g=read(path); gid=g['group_id'];refs=g['cases'];keys=g['indices'];targets=g['target_indices']
    check(re.fullmatch(r'[A-Za-z0-9_-]+',gid) and Path(path).stem==gid and len(refs)==len(keys)==len(set(keys))==4 and targets and len(targets)==len(set(targets)) and set(targets)<=set(keys),'Invalid B4 descriptor')
    rows=[]
    for ref,idx in zip(refs,keys):
        f=(Path(data)/ref['path']).resolve();check(f.is_relative_to(Path(data).resolve()) and sha(f)==ref['file_sha256'],'Changed packed row')
        opener=gzip.open if f.suffix=='.gz' else open
        with opener(f,'rt',encoding='utf-8') as src:row=json.load(src)
        n=row['input_tokens'];check(row['index']==idx and n==len(row['input_ids']) and n<LIMIT-16 and row['input_ids_sha256']==digest(row['input_ids']) and 5<=len(row['documents'])<=15,'Original full LongBench input mismatch')
        eligible=[d for d in row['documents'] if d['input_end']-d['input_start']>=256]
        body=sum(d['input_end']-d['input_start'] for d in eligible)
        check(eligible and (15*body+99)//100>=128*len(eligible),'15% body budget cannot include every eligible tail')
        rows.append(row)
    check(len({(r['input_tokens']+CHUNK-1)//CHUNK for r in rows})==1,'Sparse B4 requires equal native frame counts')
    return g,rows


def capture_waves(plans):
    # The bank is content-addressed. Captures commute within an owner. Only combine native
    # requests with equal frame count so the full-model EP step sequence stays aligned.
    by_frame={}
    for owner,p in enumerate(plans):
        for doc in p['unique_capture_documents']:
            frame=(doc['tokens']+CHUNK-1)//CHUNK
            by_frame.setdefault(frame,[[] for _ in plans])[owner].append(doc)
    for buckets in by_frame.values():
        while all(buckets):
            yield {owner:q.pop(0) for owner,q in enumerate(buckets)}
        for owner,q in enumerate(buckets):
            for doc in q:yield {owner:doc}


def compact_controller(report):
    sparse=report.get('cone15_ffn');idx=report.get('index20',{})
    out={k:report[k] for k in ('input_ids_sha256','full_input_tokens','mode','model_boundary_restored','decode_native')}
    if sparse:
        choice=idx['selection']; pack=report['system_pack_fixed_v1']
        v2=pack.get('v2_policy')
        if v2 and v2.get('mode')!='s1.6':
            expected_fraction=float(v2['fraction'])
            check(sparse['method']=='Cone15' and sparse['new_and_tail_full_ffn_each_layer']
                  and sparse['all_input_lastkv_physical_replay']
                  and sparse['qualification_cache_bytes'] is None
                  and choice['strategy']=='max'
                  and abs(float(choice['fraction'])-expected_fraction)<1e-9
                  and bool(v2.get('two_stage_l14',False))==bool(choice.get('two_stage_l14',False))
                  and pack['enabled'] and pack['canonical_method']=='Cone15',
                  'Wrong RedKnot V2 online path')
        else:
            check(sparse['method']=='Cone15' and sparse['new_and_tail_full_ffn_each_layer']
                  and sparse['all_input_lastkv_physical_replay']
                  and sparse['qualification_cache_bytes'] is None
                  and choice['strategy']=='max' and choice['fraction']==.15
                  and pack['enabled'] and pack['canonical_method']=='Cone15',
                  'Wrong S1.6 online path')
        groups=sparse['groups'];q=0;selected=0
        for g in groups:
            check([l['layer'] for l in g['layers']]==list(range(21,40)),'Missing post-source20 rows')
            for l in g['layers']:
                check(l['actual_ffn_rows']==l['dispatch_rows']==l['live_ffn_rows'] and l['core']['actual_attention_rows']==l['actual_ffn_rows'],'Incomplete live execution')
                selected+=l['core']['input_axis_rows'];q+=l['actual_ffn_rows']
        out.update(sparse_method='Cone15',selected_tokens=len(choice['selected_positions']),selected_sha256=digest(choice['selected_positions']),post20_ffn_rows=q,original_post20_ffn_rows=selected,mandatory_tail128_complete=True)
        if v2:
            out['v2_policy']=v2
            out['minimal_cone']=bool(sparse.get('minimal_cone',False))
            if 'two_stage_l14' in report:
                out['two_stage_l14']=report['two_stage_l14']
    return out


@contextmanager
def compact_completion(runtime):
    # The original .finish() runs in full. Compact after local verification and before the
    # eight-rank host gather, avoiding gigabytes of repeated Python objects on each B4.
    ctrl=runtime.controller
    if ctrl is None:
        yield;return
    method=ctrl.finish;present='finish' in vars(ctrl);prev=vars(ctrl).get('finish')
    def finish(self):
        orig=method()
        return compact_controller(orig)
    ctrl.finish=MethodType(finish,ctrl)
    try:yield
    finally:
        if present:ctrl.finish=prev
        else:del ctrl.finish


def execute_phase(native,rows_by_owner,phase,method,mode,max_by_owner,documents):
    from multiowner_transport import run_multiowner_requests
    from multiowner_runtime import validate_owner_decode
    from service.engine_bridge import TokenRequest
    now=native.driver.gather(time.perf_counter() if native.rank==0 else None)[0]
    reqs={r:TokenRequest(f'{phase}.owner{r}',tuple(row['input_ids']),max_by_owner[r],
              's2' if mode in ('sensitive','capture_independent') else 'full',now,
              documents=tuple(documents.get(r,()))) for r,row in rows_by_owner.items()}
    evidence={}
    def finish(_):
        completion=native._finish_many(reqs)
        check(completion['world_scheduler']['all_eight_workers_agreed'] and completion['world_scheduler']['actual_request_batch_size']==len(reqs),'Missing real multiowner EP proof')
        if native.rank in reqs:
            r=native.rank;controller=completion['controllers_by_rank'][r];scheduled=completion['scheduled_rows_reports'][r]
            check(controller['input_ids_sha256']==rows_by_owner[r]['input_ids_sha256'] and controller['full_input_tokens']==len(reqs[r].prompt_token_ids) and controller['model_boundary_restored'] and controller['decode_native'],'Wrong owner model/sequence')
            check((controller.get('sparse_method')=='Cone15')==(mode=='sensitive'),'Changed arm during model execution')
            check(scheduled['scheduled_prefill_rows']==len(reqs[r].prompt_token_ids),'Different native prefill length')
            controls={k:completion[k] for k in ('scheduled_rows_reports','decode_graph_evidence','native_cache_storage_evidence','native_idle_bypass_installed')}
            evidence.update(controls=controls,controller=controller,world=completion['world_scheduler'])
        return None
    native._diagnostic=False
    try:
        local=run_multiowner_requests(native.port,reqs,rank=native.rank,active_ranks=tuple(sorted(reqs)),
            barrier=native.driver.barrier,gather=native.driver.gather,
            prepare=lambda _:native._configure_many(reqs,phase,mode,method,'max'),finish=finish,emit=lambda event:None)
    finally:
        coord=getattr(native,'_two_stage_l14',None)
        if coord is not None:
            coord.end_phase()
    answer=None
    if local is not None:
        r=native.rank;row=rows_by_owner[r];ids=local['output_ids'];quality=mode!='capture_independent'
        proof=validate_owner_decode(dict(evidence['controls'],record=local),r,require_real_decode=quality and len(ids)>1)
        world=evidence['world'];answer=dict(owner_rank=r,request_id=local['request_id'],
            input_ids_sha256=row['input_ids_sha256'],input_tokens=len(reqs[r].prompt_token_ids),
            actual_active_request_count=world['actual_request_batch_size'],world_plan_sha256=world['world_plan_sha256'],
            decode=proof,controller=evidence['controller'],finish_reason=local['finish_reason'])
        check(ids and len(ids)<=max_by_owner[r] and (local['finish_reason']=='stop' or (local['finish_reason']=='length' and len(ids)==max_by_owner[r])),'Native generation did not finish')
        if quality:
            answer.update(index=row['index'],dataset=row['dataset'],source_id=row.get('source_id'),output_ids=ids,
                          text=native.driver.engine.tokenizer.decode(ids,skip_special_tokens=False),
                          prediction_text=native.driver.engine.tokenizer.decode(ids,skip_special_tokens=True),
                          output_tokens=len(ids),model_context_output_capacity=max_by_owner[r])
            for a,b in (('received','received'),('admitted','admitted'),('prepared','prepared'),('submitted','submitted'),('first_token','first_raw_token'),('completed','terminal'),('drained','drained')):
                answer[a+'_ns']=round(local[b+'_s']*1e9)
            check(answer['received_ns']<=answer['submitted_ns']<=answer['first_token_ns']<=answer['completed_ns'],'Invalid native receipt-to-first clocks')
    all_records=native.driver.gather(answer)
    check([r for r,x in enumerate(all_records) if x is not None]==sorted(reqs),'Missing completed native owners')
    return [all_records[r] for r in sorted(reqs)]


def prepare(native,gid,rows):
    from rag_natural_documents import plan_case
    plans=[plan_case(r['input_ids'],r['documents']) for r in rows]
    check(all(p['effective_route']=='s12' for p in plans),'Unsupported natural source layout')
    if native.controller is not None:
        controller=native.controller
        with native.allow_sync():controller.clear_cache(reset_sparse_qualification=True)
        check(not controller.cache.bank and controller.case_input is None and controller.case_binding is None,'Previous GPU document bank was not released')
        native.controller.set_cache_output_dir(native.out/(gid+'.bank.owner'+str(native.rank)))
    native.driver.barrier();started=time.monotonic();count=0;batches=[]
    for j,docs in enumerate(capture_waves(plans)):
        inputs={owner:dict(input_ids=rows[owner]['input_ids'][d['input_start']:d['input_end']],input_ids_sha256=d['token_ids_sha256']) for owner,d in docs.items()}
        execute_phase(native,inputs,f'{gid}.capture{j}','Full','capture_independent',{owner:1 for owner in docs},{})
        count+=len(docs);batches.append(len(docs))
    if native.controller is not None:
        r=native.rank;native.local_case=rows[r];native.local_case_plan=plans[r]
        with native.allow_sync():
            native.controller.prepare_case(input_ids=rows[r]['input_ids'],documents=plans[r]['effective_documents'],index_device=native.layers[0].self_attn.q_a_kv_swa_proj.weight.device)
    native.driver.barrier()
    return {r:p['effective_documents'] for r,p in enumerate(plans)},dict(captures=count,capture_batch_sizes=batches,wall_s=time.monotonic()-started,offline_work_in_ttft=False)


def closed(path,g,rows,method,contract_id):
    if not path.is_file():return False
    b=read(path);sealed=b.pop('sha256');check(sealed==digest(b) and b['contract_id']==contract_id and b['group_id']==g['group_id'] and b['method']==method and b['input_indices']==g['indices'] and b['target_indices']==g['target_indices'] and b['actual_gpu_executed'] and b['status']=='complete','Existing result differs from this experiment')
    check(len(b['requests'])==4 and all(x['index']==r['index'] and x['input_ids_sha256']==r['input_ids_sha256'] and x['output_ids'] for x,r in zip(b['requests'],rows)),'Invalid existing B4 result')
    return True


def next_group(data,done,selected_groups=None):
    if selected_groups is not None:
        for gid in selected_groups:
            if gid not in done:
                path=data/'groups'/(gid+'.json')
                check(path.is_file(),f'Requested group not found: {gid}')
                return ('group',path.name)
        return ('complete',dict(group_count=len(selected_groups),target_count=None,selected_groups=list(selected_groups)))
    for f in sorted((data/'groups').glob('*.json')):
        if f.stem not in done and not f.name.startswith('.'):
            return ('group',f.name)
    if (data/'complete.json').exists():
        complete=read(data/'complete.json')
        if len(done)<complete['group_count']:
            # The filesystem view may lag the terminal manifest briefly.
            return ('wait',None)
        check(len(done)==complete['group_count'],'Input producer and executor disagree on number of groups')
        return ('complete',complete)
    if (data/'failed.json').is_file():raise RuntimeError(f'Input producer failed: {data}/failed.json')
    return ('wait',None)


def process(native,data,out,contract_id,selected_groups=None):
    done=set();targets=set();t0=time.time();count_arms=0
    with compact_completion(native):
        while True:
            task=native.driver.gather(next_group(data,done,selected_groups) if native.rank==0 else None)[0]
            if task[0]=='wait':time.sleep(2);continue
            if task[0]=='complete':
                if selected_groups is None:
                    check(len(targets)==task[1]['target_count'],'Number of distinct completed targets differs from manifest')
                else:
                    check(len(done)==len(selected_groups),'Selected subset group count mismatch')
                if native.rank==0:
                    run_payload=dict(status='complete',contract_id=contract_id,target_count=len(targets),group_count=len(done),complete_source=task[1],finished=time.time(),began=t0)
                    if selected_groups is not None:
                        run_payload['selected_groups']=list(selected_groups)
                    save(out/'run.json',run_payload)
                return
            group,rows=group_rows(data,data/'groups'/task[1]);gid=group['group_id'];area=out/'groups'/gid
            check(not set(group['target_indices'])&targets,'Repeated target detected')
            sequence=['Full','S1.6'] if len(done)%2==0 else ['S1.6','Full']
            pending=native.driver.gather([m for m in sequence if not closed(area/(m+'.json'),group,rows,m,contract_id)] if native.rank==0 else None)[0]
            if pending:
                if native.rank==0:save(out/'progress.json',dict(phase='offline_banks',group_id=gid,closed_groups=len(done),closed_targets=len(targets),updated=time.time()))
                docs,prep=prepare(native,gid,rows)
                for m in pending:
                    if native.rank==0:save(out/'progress.json',dict(phase='online_b4',method=m,group_id=gid,closed_groups=len(done),closed_targets=len(targets),updated=time.time()))
                    active=dict(enumerate(rows));native_method=ARMS[m]
                    answers=execute_phase(native,active,f'{gid}.{native_method}.max.quality',native_method,None if m=='Full' else 'sensitive',
                                          {r:LIMIT-row['input_tokens']-16 for r,row in active.items()},docs)
                    if native.rank==0:
                        v2_cfg=getattr(native,'_v2_policy_config',None)
                        record_payload=dict(schema='s16-longbench-b4-result-v1',status='complete',contract_id=contract_id,
                            group_id=gid,method=m,native_method=native_method,phase='quality',
                            input_indices=group['indices'],target_indices=group['target_indices'],request_batch_size=4,ep_size=8,
                            selector_strategy='max',actual_gpu_executed=True,global_real_request_count=4,main_and_draft_graphs_enabled=True,
                            quality_artificial_limit=None,ttft_scope='online receipt to actual first raw token, warm model and separately prepared document banks',offline_preparation=prep,requests=answers)
                        if m!='Full' and v2_cfg is not None:
                            record_payload['redknot_v2']=v2_cfg.to_dict()
                        save(area/(m+'.json'),record_payload)
                    native.driver.barrier();count_arms+=1
            targets.update(group['target_indices']);done.add(gid)
            if native.rank==0:
                save(area/'group.json',dict(status='complete',contract_id=contract_id,group_id=gid,indices=group['indices'],
                     target_indices=group['target_indices'],methods=['Full','S1.6'],completed_targets=len(targets)))
                save(out/'progress.json',dict(phase='group_complete',group_id=gid,closed_groups=len(done),closed_targets=len(targets),updated=time.time()))
            native.driver.barrier()


def worker(template,data,out,attempt,contract_id,world,rank):
    out,attempt,data=Path(out),Path(attempt),Path(data)
    try:
        check(world==8,'Expected actual EP8/DP8 world')
        cfg=load_template(template)
        sys.path.insert(0,cfg['fixed_pack_runtime_root']);import fixed_pack_runtime_adapter as adapter
        with adapter.create_runtime(cfg,rank=rank,output_dir=attempt/'native') as rt:
            native=getattr(rt,'runtime',rt)
            installed=native.driver.gather(dict(rank=rank,owners=list(native.installed_owner_ranks),profile=native.profile.name,
                dspark=native.actual_runtime_reports[rank]['observed']['nextn'],
                main_graph=native.actual_runtime_reports[rank]['observed']['use_cuda_graph'],
                draft_graph=native.actual_runtime_reports[rank]['observed']['dflash_use_cuda_graph']))
            check(all(s['rank']==k and s['owners']==[0,1,2,3] and s['profile']=='dspark' and s['dspark']==5 and s['main_graph'] and s['draft_graph'] for k,s in enumerate(installed)),'Actual model does not match graph/D-Spark contract')
            if rank==0:save(out/'resident.json',dict(status='native_model_ready',contract_id=contract_id,observed=time.time(),attempt=str(attempt),worker_proofs=installed))
            process(native,data,out,contract_id,selected_groups=cfg.get('selected_groups'))
    except BaseException as e:
        p=attempt/('failure.rank'+str(rank)+'.json')
        if not p.is_file():save(p,dict(status='failed',rank=rank,error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        raise


def main():
    import fcntl
    parser=argparse.ArgumentParser()
    for k in ('template','manifest','output-dir'):parser.add_argument('--'+k,required=True,type=Path)
    parser.add_argument('--ready-file',type=Path);parser.add_argument('--release-file',type=Path)
    parser.add_argument('--preset',choices=('two-stage','single-stage'),default=None)
    parser.add_argument('--host-prefix-offload',choices=('auto','on','off'),default='auto')
    action=parser.add_mutually_exclusive_group(required=True);action.add_argument('--check-only',action='store_true');action.add_argument('--run',action='store_true')
    a=parser.parse_args();p=a.template.resolve();data=a.manifest.resolve();out=a.output_dir.resolve();cfg=load_template(p)
    if a.preset is not None or a.host_prefix_offload!='auto':
        sys.path.insert(0,cfg['fixed_pack_runtime_root']);from v2_policy import V2PolicyConfig
        v2_obj=V2PolicyConfig.from_cli(preset=a.preset or 'two-stage',host_prefix_offload=a.host_prefix_offload)
        cfg['redknot_v2']=v2_obj.to_dict()
    contract_id=digest(dict(template=ARTIFACT,runner_sha256=sha(__file__),data_root=str(data),methods=ARMS,owner_count=4,redknot_v2=cfg.get('redknot_v2')))
    check(data.is_dir(),'Manifest data directory must exist')
    if cfg.get('strict_source_hashes', False):
        check(sha(cfg['adapter_ref']['path'])==cfg['adapter_ref']['file_sha256'] and sha(cfg['service_config_ref']['path'])==cfg['service_config_ref']['file_sha256'],'Changed native template dependencies')
        for src,h in cfg['source_sha256'].items():check(sha(src)==h,'Changed frozen source: '+src)
    sys.path.insert(0,cfg['fixed_pack_runtime_root']);import fixed_pack_runtime_adapter as adapter
    preserved,_=adapter.load_dependencies(cfg);adapter.private_preserved_module(preserved)
    check('torch' not in sys.modules and 'deepinfer' not in sys.modules,'GPU library import before native release')
    if a.check_only:
        first=next(iter(sorted((data/'groups').glob('*.json'))),None)
        if first:group_rows(data,first)
        print(json.dumps(dict(status='cpu_pass',contract_id=contract_id,source_files=len(cfg['source_sha256']),first_valid_group=first.stem if first else None,model_calls=0)),flush=True);return
    out.mkdir(exist_ok=True,parents=True)
    with (out/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(not (out/'run.json').exists(),'Experiment is already complete')
        sys.path.insert(0,cfg['rag_runtime_root']);from run_rag_dspark_pair import wait_release
        wait_release(a.ready_file,a.release_file,contract_id)
        service=read(cfg['service_config_ref']['path']);root=Path(service['experiment_root'])/'kv_recompression_20261029';launcher=root/'quality_process_group.py'
        check(sha(launcher)==cfg['source_sha256'][str(launcher)],'Native launcher changed')
        sys.path.insert(0,str(root));from quality_process_group import launch
        attempt=out/'attempts'/f'run-{time.time_ns()}';attempt.mkdir(parents=True)
        launch(functools.partial(worker,str(p),str(data),str(out),str(attempt),contract_id),8,attempt)


if __name__=='__main__':main()
