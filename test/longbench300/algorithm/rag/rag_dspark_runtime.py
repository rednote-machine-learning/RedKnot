"""RAG-only direct token adapter over the preserved D-Spark native service.

No HTTP, agent scaffold, explicit block markers, adaptive selection, or hidden
state algorithm is introduced. The original 10 cache blocks and fixed group17
selected-row policy are retained. All GPU imports remain inside initialize().
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seal(value):return dict(value,sha256=digest(value))


def require(value,message):
    if not value:raise ValueError(message)


def read_ref(ref):
    path=Path(ref['path']);raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==ref['file_sha256'],'Referenced input changed: '+str(path))
    value=json.loads(raw)
    require(value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'Referenced JSON seal changed')
    return value


def original_rows(manifest,indices):
    require(manifest.get('sha256')=='fb6bba62937fb8005f05df9a3a213729c172932a30e5509b136000692881aa67',
            'Expected preserved original Loong 64K/128K manifest')
    require(type(indices) is list and indices and len(set(indices))==len(indices), 'Unique explicit case indices required')
    by_id={r['index']:r for r in manifest['cohort']}
    rows=[]
    for index in indices:
        r=by_id[index]
        require(r['input_ids_sha256']==digest(r['input_ids']) and len(r['input_ids'])==r['input_tokens'],
                'Original input token IDs changed')
        docs=r['documents'];require(len(docs)==10,'Original RAG policy uses exactly ten natural cache blocks')
        previous=0
        for d in docs:
            a,z=d['input_start'],d['input_end']
            require(type(a) is type(z) is int and previous<=a<z<=len(r['input_ids']),'Unordered/overlapping original document spans')
            require(d['tokens']==z-a and d['token_ids_sha256']==digest(r['input_ids'][a:z]),
                    'Original cache block token span changed')
            previous=z
        rows.append(r)
    return rows


def load_config(path):
    cfg=json.loads(Path(path).read_bytes())
    require(cfg.get('sha256')==digest({k:v for k,v in cfg.items() if k!='sha256'}),'RAG config seal changed')
    require(cfg.get('schema')=='rag-dspark-pair-config-v1','Wrong RAG runner configuration')
    require(type(cfg.get('timing_repeats')) is int and 1<=cfg['timing_repeats']<=5,'Explicit 1..5 repeats required')
    require(type(cfg.get('quality')) is bool and type(cfg.get('global_base_cache')) is bool,'Explicit output/cache policy required')
    require(cfg['global_base_cache'] is False,'Global-base optimization has a separate, not yet installed adapter')
    for p,h in cfg['source_sha256'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'RAG source changed: '+p)
    require(str(Path(__file__).resolve()) in cfg['source_sha256'],'Runtime adapter source pin missing')
    manifest=read_ref(cfg['manifest_ref']);rows=original_rows(manifest,cfg['indices']);base=read_ref(cfg['service_config_ref'])
    native_root=Path(cfg['native_source_root']).resolve()
    for relative in ('service/native_runtime.py','GatedEngram/kv_cache_engram/s1_dflash_bridge.py','NativeSnapshot/deepinfer/__init__.py',
                     'native_decode_profile.py','dflash_tail_adapter.py','prefix_decode_contract.py'):
        require((native_root/relative).is_file(),'Preserved native D-Spark asset missing: '+relative)
    service=dict(base);service.pop('sha256',None)
    service.update(native_repo=str(native_root/'NativeSnapshot'),gated_repo=str(native_root/'GatedEngram'),
                   stage=str(native_root),context_tokens=270336,gpu_memory_utilization=.60)
    service['scope']=dict(base['scope'],execution_sha256=digest(dict(base_scope=base['scope'],rag_config_sha256=cfg['sha256'])))
    service=seal(service)
    require(len(service['qualification_input_ids'])==16384,'Preserved native warmup input must be 16K tokens')
    return cfg,service,manifest,rows


def bind_assets(cfg,service):
    root=Path(cfg['native_source_root']).resolve()
    for path in (root,root/'GatedEngram'):
        sys.path.insert(0,str(path))
    runtime=importlib.import_module('service.native_runtime')
    require(Path(runtime.__file__).resolve()==root/'service/native_runtime.py','Conflicting service package was imported')
    return runtime


def runtime_class(parent):
    """Private subclass; preserved NativeRuntime and module globals stay intact."""
    class RagNativeRuntime(parent.NativeRuntime):
        def _internal(self,request,phase,mode,*,verify_sparse=False,qualification_only=True):
            local=parent.run_serial_request(self.port,request,rank=self.rank,barrier=self.driver.barrier,
                prepare=lambda r:self._configure(r,phase,mode,verify_sparse=verify_sparse),
                finish=self._finish_phase,emit=lambda event:None)
            record=self.driver.gather(local)[0]
            record['text']=self.driver.engine.tokenizer.decode(record['output_ids'],skip_special_tokens=False)
            body=parent.seal(dict(schema='rag-dspark-native-phase-v1',name=phase,mode=mode,
                record={k:v for k,v in record.items() if k not in ('preparation','completion')},
                **record['completion'],qualification_only=qualification_only,
                benchmark_request=not qualification_only,execution_purpose='numerical_qualification' if qualification_only else 'rag_measurement',
                agent_framework_executed=False,original_rag_input_unchanged=True))
            self.publish(phase,body)
            return body

        def _initialize_s2(self):
            # Retain the exact S1/S1.2 operator capability checks used by the
            # live D-Spark adapter, without S2's agent/marker/adaptive admission.
            from gpu_pilot.cuda_mask_qualification import run_cuda_mask_qualification,validate_report
            from gpu_pilot.index_hit20_empty_ep_qualification import run_empty_ep_qualification,build_all_ranks_report,EmptyEPCapability
            from kv_cache_engram.s1_runtime import run_s1_runtime_qualification
            from gpu_pilot.read_s11_results import validate_runtime_qualification
            from kv_cache_engram.s1_system_batch import run_s1_system_metadata_qualification
            from kv_cache_engram.s1_system_runtime import install_worker_batch_schedule
            from kv_cache_engram.s11_qualification import S11QualificationController
            from service.canonical_request_binding_dspark import install_request_binding
            t,allow,g,rank=self.torch,self.allow_sync,self.driver.gather,self.rank
            masks=g(run_cuda_mask_qualification(t,allow))
            for r in masks:validate_report(r)
            self.startup['mask']=self.publish('qualification-mask',dict(passed=True,reports=masks))
            ep=build_all_ranks_report(g(run_empty_ep_qualification(self.layers,rank,t,allow,
                execution_sha256=self.scope['execution_sha256'],model_sha256=self.scope['model_sha256'])),
                execution_sha256=self.scope['execution_sha256'],model_sha256=self.scope['model_sha256'])
            self.startup['empty_ep']=self.publish('qualification-empty-ep',ep)
            rt=parent.seal(dict(schema='s1-runtime-all-ranks-qualification-v1',passed=True,
                execution_configuration_sha256=self.scope['execution_sha256'],model_sha256=self.scope['model_sha256'],
                reports=g(run_s1_runtime_qualification(self.layers,rank,t,allow,
                    execution_sha256=self.scope['execution_sha256'],model_sha256=self.scope['model_sha256'])),
                empty_ep_qualification_sha256=ep['sha256'],quality_eligible=False,timing_valid=False))
            validate_runtime_qualification(rt,dict(execution_configuration_sha256=self.scope['execution_sha256'],
                checkpoint_index_sha256=self.scope['model_sha256']),ep['sha256'])
            self.startup['operators']=self.publish('qualification-runtime',rt)
            md=g(dict(rank=rank,operator_report=run_s1_system_metadata_qualification(t,allow,device='cuda',group_chunks=(17,))))
            for i,item in enumerate(md):
                r=item['operator_report']
                require(item['rank']==i and r['passed'] is True and r['cuda_exercised'] is True
                    and r['requested_batch_groups']==[17] and r['sync_debug_mode_during']==2,
                    'Original all-rank group17 metadata qualification failed')
            self.startup['metadata17']=self.publish('qualification-metadata17',dict(passed=True,reports=md))
            cap=EmptyEPCapability(self.layers,ep,execution_sha256=self.scope['execution_sha256'],model_sha256=self.scope['model_sha256'])
            self.binder=install_request_binding(self.model_runner,lambda:self.binding if rank==0 else None)
            if rank==0:
                self.controller=S11QualificationController(self.layers,allow_sync=allow,native_model=self.model,
                    expected_document_count=10,model_sha256=self.scope['model_sha256'],tokenizer_sha256=self.scope['tokenizer_sha256'],
                    execution_sha256=self.scope['execution_sha256'],dependency_root=Path(self.config['experiment_root']),
                    empty_ep_capability=cap,enable_dflash_tail=True)
                self.controller.install()
            self.observer.close();self.observer=None
            self.batch_worker=install_worker_batch_schedule(self.model,rank=rank,torch_module=t,empty_ep_capability=cap,
                controller=self.controller,engine=self.driver.engine,phase_gather=g,allow_native_decode_graphs=True)
            from service.prefix_cache import ScheduleObserver
            self.observer=ScheduleObserver(self.driver.engine,rank)
            from service.engine_bridge import TokenRequest
            req=TokenRequest('startup-native-identity',tuple(self.config['qualification_input_ids']),32,'s2',time.perf_counter())
            ref=self._internal(req,'startup-qualified-native','qualified_native')
            cand=self._internal(replace(req,request_id='startup-all-selected'),'startup-all-selected','all_selected')
            r,c=ref['controller'],cand['controller']
            require(r['identity_check_count']==c['identity_check_count']==80 and c['identity_exact'] is True
                and r['identity_fingerprints']==c['identity_fingerprints']
                and ref['record']['output_ids']==cand['record']['output_ids'],'Original native/all-selected identity failed')
            from redknot.decode_evidence import validate_decode_phase
            self.startup['native_identity']=self.publish('qualification-native-identity',dict(passed=True,
                reference_sha256=ref['sha256'],candidate_sha256=cand['sha256'],
                native_decode=validate_decode_phase(ref),candidate_decode=validate_decode_phase(cand),
                full_prompt_equivalence_claim=False))

        def run_phase(self,row,name,mode,*,tokens=32,qualification=False,verify_sparse=False):
            from service.engine_bridge import TokenRequest
            req=TokenRequest(name,tuple(row['input_ids']),tokens,'full' if mode is None else 's2',time.perf_counter(),
                documents=tuple(row.get('documents',())))
            return self._internal(req,name,mode,verify_sparse=verify_sparse,qualification_only=qualification)

        def capture_case(self,row,prefix):
            if self.controller is not None:
                self.controller.clear_cache(reset_sparse_qualification=True)
                self.controller.set_cache_output_dir(self.out/(prefix+'.offline-kv'))
            self.driver.barrier();began=time.perf_counter();phases=[]
            for i,d in enumerate(row['documents']):
                offline=dict(input_ids=row['input_ids'][d['input_start']:d['input_end']])
                phases.append(self.run_phase(offline,prefix+'.offline_doc'+str(i),'capture_independent',tokens=1,qualification=True))
            if self.controller is not None:
                with self.allow_sync():self.controller.prepare_case(input_ids=row['input_ids'],documents=row['documents'],
                    index_device=self.layers[0].self_attn.q_a_kv_swa_proj.weight.device)
            self.driver.barrier()
            return max(self.driver.gather(time.perf_counter()-began)),phases
    return RagNativeRuntime


def verify_pair(reference,candidate,row):
    """Check existing full-member equality receipts; never admit from flags alone."""
    from kv_cache_engram.s1_plan import all_chunks
    from kv_cache_engram.s1_runtime import s1_documents
    from redknot.gate import verify_dflash_tail_pair
    chunks=all_chunks(s1_documents(row['input_ids'],row['documents'],expected_document_count=10),len(row['input_ids']))
    expected={(c['chunk_start'],c['chunk_end'],layer) for c in chunks for layer in range(40)}
    r,c=reference['controller'],candidate['controller']
    for control,mode,group in ((r,'reference_sparse',1),(c,'sensitive',17)):
        require(control['mode']==mode and control['system_enabled'] is True and control['batch_group_chunks']==group,
                'Original dense/group17 execution mode changed')
        require(control['input_ids_sha256']==row['input_ids_sha256'] and control['full_input_tokens']==row['input_tokens'],
                'Numerical comparison input changed')
        fp=control['sparse_identity_fingerprints']
        require(control['sparse_identity_check_count']==len(expected) and len(fp)==len(expected)
            and {tuple(v['key']) for v in fp}==expected and control['sparse_identity_fingerprints_sha256']==digest(fp),
            'Incomplete original-chunk/layer tensor fingerprints')
        require(control['first_three_full'] is True and control['decode_native'] is True
            and control['source2_native_cache_preserved'] is True and control['mathematical_policy_changed'] is False,
            'Original S1.2 mathematical policy changed')
        require(len(control['source2_identity_events'])==len(chunks)
            and all(v['exact'] is True and v['before']==v['after'] for v in control['source2_identity_events']),
            'Original source2 native cache changed')
    require(c['sparse_identity_exact'] is True and c['system_numerically_qualified'] is True
        and r['sparse_identity_fingerprints']==c['sparse_identity_fingerprints'],'Actual selected hidden/mHC equality failed')
    for k in ('bank_generation','offline_bank_scope_sha256'):
        require(r[k]==c[k],'Numerical comparison changed immutable bank')
    require(r['phase_generation']!=c['phase_generation'],'Fresh phase state required')
    for k in ('output_ids','output_ids_sha256','text','finish_reason'):
        require(reference['record'][k]==candidate['record'][k],'Complete same-policy bounded output differs')
    tail=verify_dflash_tail_pair(r,c,input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'])
    return seal(dict(schema='rag-dspark-same-policy-qualification-v1',passed=True,input_ids_sha256=row['input_ids_sha256'],
        original_chunk_layer_checks=len(expected),reference_sha256=reference['sha256'],candidate_sha256=candidate['sha256'],
        dflash_tail=tail,full_model_equivalence_claim=False,selected_policy_unchanged=True))
