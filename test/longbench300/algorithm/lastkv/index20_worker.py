"""All-rank final-flush companion for the independent Index20 experiment.

Baseline Full/S1.2 calls delegate to the frozen worker. Only Index20 schedules
defer deep layers until all native first-three-layer chunks have completed.
"""
from types import SimpleNamespace as NS
from kv_cache_engram.core import digest, require
from kv_cache_engram.s1_system_runtime import WorkerBatchSchedule, inactive_empty_query_cu
from kv_cache_engram.s1_runtime import compact_metadata

SCHEMA = 'index20-first3-finalflush-v1'


def is_index20(schedule):
    return isinstance(schedule, dict) and schedule.get('schema') == SCHEMA


def validate_schedule(schedule):
    require(is_index20(schedule), 'Index20 schedule schema')
    require(schedule.get('sha256') == digest({k:v for k,v in schedule.items() if k!='sha256'}),
            'Index20 schedule seal')
    n=schedule['full_input_tokens']; steps=schedule['steps']
    require(type(n) is int and n>0 and len(steps)==(n+8191)//8192, 'Index20 native chunk count')
    require(isinstance(schedule['request_id'],str) and schedule['request_id'], 'Index20 request identity')
    require(schedule.get('dynamic_deep_groups') is True,'Index20 final flush must declare dynamic groups')
    for i,step in enumerate(steps):
        require(step['start']==i*8192 and step['end']==min(n,(i+1)*8192)
                and step['flush'] is (i==len(steps)-1), 'Index20 first3/final flush geometry')
        require(step['layers']==(list(range(40)) if step['flush']else [0,1,2]),
                'Index20 first3/final dynamic layer descriptor')


def validate_agreement(items,schedule):
    validate_schedule(schedule)
    require(isinstance(items,list) and len(items)==8,'Index20 eight-rank flush agreement')
    for rank,item in enumerate(items):
        require(item.get('rank')==rank and item.get('schedule_sha256')==schedule['sha256']
                and item.get('request_id')==schedule['request_id'], 'Index20 flush identity mismatch')
        if rank:require(item.get('num_groups') is None,'Inactive rank proposed Index20 groups')
    count=items[0].get('num_groups')
    require(type(count) is int and 1<=count<=len(schedule['steps']), 'Index20 invalid deep group count')
    return count


class Index20WorkerSchedule(WorkerBatchSchedule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.index20_group_count=None
        if self.controller is not None:
            self.controller.index20_flush_agreement=self.agree_flush

    def configure(self,schedule):
        self.index20_group_count=None
        if not is_index20(schedule):return super().configure(schedule)
        require(not self.closed and self._phase_done,'Index20 worker prior phase incomplete')
        validate_schedule(schedule)
        self.schedule,self.trace,self.world_trace=schedule,[],[]
        self._phase_done,self._world_frame=False,None
        if self.rank==0:
            require(self.controller.batch_schedule==schedule and self.controller.state=='RUNNING',
                    'Index20 controller/worker disagreement')
        return dict(schema='index20-worker-install-v1',rank=self.rank,enabled=True,
            schedule_sha256=schedule['sha256'],worker_loop_installed=True,
            native_first3_per_chunk=True,deep_only_on_final_flush=True,
            dynamic_flush_count_agreement=True,world_scheduler_phase_bound=True)

    def agree_flush(self,count):
        require(is_index20(self.schedule) and self.index20_group_count is None,
                'Index20 duplicate/unbound final agreement')
        require((self.rank==0 and type(count)is int) or (self.rank!=0 and count is None),
                'Index20 flush proposal ownership')
        frame=self._world_frame
        require(frame is not None and frame['world_phase']=='prefill'
                and frame['end']==self.schedule['full_input_tokens'], 'Index20 agreement outside final native prefill')
        items=self.phase_gather(dict(rank=self.rank,num_groups=count,
            schedule_sha256=self.schedule['sha256'],request_id=self.schedule['request_id']))
        self.index20_group_count=validate_agreement(items,self.schedule)
        return self.index20_group_count

    def forward(self,caches,meta,hidden,dflash):
        if not is_index20(self.schedule):return super().forward(caches,meta,hidden,dflash)
        frame=self._world_frame
        require(frame is not None and not frame['model_consumed'],'Index20 inactive frame authorization')
        require(meta.is_prompt is frame['local_prompt_run'],'Index20 native placeholder step type')
        frame['model_consumed']=True
        if frame['world_phase']!='prefill':return self.original(caches,meta,hidden,dflash)
        require(meta.is_prompt is True and frame['local_is_empty'] is True and self.rank!=0,
                'Index20 inactive prefill owner')
        require(not self._phase_done and len(self.trace)<len(self.schedule['steps']),
                'Index20 extra inactive prefill')
        require(meta.num_reqs==1 and meta.total_q==1
                and getattr(meta,'canonical_request_binding',None)is None,'Index20 placeholder geometry')
        t,g,model=self.t,self.symbols,self.model
        feature_size=getattr(model,'dflash_input_size',0)
        require((dflash is not None)==bool(feature_size) and (not feature_size or self.allow_native_decode_graphs),
                'Index20 D-Spark native decode admission')
        for name in ('input_ids','positions','state_block_idx'):
            t._assert_async((getattr(meta,name)==0).all(),'Index20 inactive real row: '+name)
        require(model.engram_runtime is not None and meta.ngram_token_ids is not None,
                'Index20 original Engram unavailable')
        step=self.schedule['steps'][len(self.trace)]
        hidden=model.embed_tokens(meta.input_ids) if hidden is None else hidden
        require(hidden.ndim==2 and tuple(hidden.shape)==(1,model.layers[0].mlp.hidden_size)
                and hidden.dtype==t.bfloat16 and hidden.device==meta.input_ids.device and hidden.is_contiguous(),
                'Index20 original inactive BF16 embedding')
        features=hidden.new_empty((1,feature_size))if feature_size else None
        residual,cross=None,g['CrossLayerRuntime']()
        hashes=g['tile_kernels'].engram.engram_hash(meta.ngram_token_ids,model.ngram_multipliers,
            model.ngram_vocab_sizes,model.ngram_offsets,getattr(meta,'images_seq_mask',None))
        hooks=model.engram_runtime.fetch(hashes);layers_run=[]
        for i in range(3):
            hidden,residual=model.layers[i](g['Movable'](hidden),caches[i],meta,residual,
                embed_hook=hooks.get(i),cross_layer_runtime=cross,dflash_input=features)
            layers_run.append(i)
        require(callable(hooks.get(14)),'Index20 Engram14 fetch missing');hooks[14]()
        groups=self.agree_flush(None) if step['flush']else 0
        if groups:
            rows=t.empty((0,),device=hidden.device,dtype=t.long)
            compact=compact_metadata(meta,rows,inactive_empty_query_cu(meta,t))
            for _ in range(groups):
                for i in range(3,40):
                    layer=model.layers[i]
                    self.capability.permits(layer,meta,NS(full_rows=1,selected_count=0))
                    bf16=hidden.new_empty((0,layer.mlp.hidden_size),dtype=t.bfloat16)
                    sf=t.zeros((layer.mlp.hidden_size//128,4),device=hidden.device,dtype=t.int32)
                    packed=t.empty_like(bf16,dtype=t.float8_e4m3fn),sf[:,:0].transpose(0,1)
                    result=layer.mlp(packed,bf16,compact)
                    require(result.shape==bf16.shape,'Index20 empty native EP shape')
                    layers_run.append(i)
        self.trace.append(dict(start=step['start'],end=step['end'],flush=step['flush'],layers=layers_run,
            original_placeholder_rows=1,deep_native_ep_calls=37*groups,empty_deep_native_ep_calls=37*groups,
            original_engram14_fetch_consumed=True,index20_deep_groups=groups))
        return model.allocate_output(1)

    def finish_phase(self):
        if not is_index20(self.schedule):return super().finish_phase()
        require(not self.closed and not self._phase_done,'Index20 worker phase not active')
        trace=self.controller._batch_step_trace if self.rank==0 else self.trace
        require(len(trace)==len(self.schedule['steps']) and self.index20_group_count is not None,
                'Index20 worker coverage/agreement incomplete')
        for event,step in zip(trace,self.schedule['steps']):
            expected=[0,1,2]+(list(range(3,40))*self.index20_group_count if step['flush']else [])
            require(all(event[k]==step[k]for k in ('start','end','flush')) and event['layers']==expected,
                    'Index20 actual native EP loop differs')
            require(event['deep_native_ep_calls']==(37*self.index20_group_count if step['flush']else 0),
                    'Index20 EP call count differs')
        self._phase_done=True
        return dict(schema='index20-worker-execution-v1',rank=self.rank,enabled=True,
            schedule_sha256=self.schedule['sha256'],steps=list(trace),
            actual_deep_ep_calls=37*self.index20_group_count,
            actual_empty_deep_ep_calls=37*self.index20_group_count if self.rank else 0,
            index20_deep_groups=self.index20_group_count,world_scheduler_phase_bound=True,
            world_step_trace=list(self.world_trace),gpu_numerical_parity_claim=False)


def install_worker_batch_schedule(native_model,**kwargs):
    return Index20WorkerSchedule(native_model,**kwargs)


def validate_candidate(reports,schedule,workers,*,phase,request_id):
    if not is_index20(schedule):
        from service.native_runtime import validate_candidate as original
        return original(reports,schedule,workers,phase=phase,request_id=request_id)
    validate_schedule(schedule)
    require(schedule['request_id']==request_id and len(reports)==8 and all(x is None for x in reports),
            'Index20 candidate identity or idle bypass')
    require(len(workers)==8,'Index20 missing worker reports')
    groups=workers[0]['index20_deep_groups']
    require(type(groups) is int and 1<=groups<=len(schedule['steps']),
            'Index20 report group count outside admitted bounds')
    for rank,w in enumerate(workers):
        require(w.get('schema')=='index20-worker-execution-v1' and w.get('rank')==rank
                and w.get('enabled') is True and w.get('schedule_sha256')==schedule['sha256']
                and w.get('index20_deep_groups')==groups and w.get('actual_deep_ep_calls')==37*groups
                and w.get('world_scheduler_phase_bound') is True,'Index20 worker agreement failure')
        require(w.get('actual_empty_deep_ep_calls')==(37*groups if rank else 0),'Index20 empty EP ownership')
        require(len(w['steps'])==len(schedule['steps']),'Index20 missing native first3 chunks')
        for event,step in zip(w['steps'],schedule['steps']):
            expected=[0,1,2]+(list(range(3,40))*groups if step['flush']else [])
            require(all(event[k]==step[k]for k in ('start','end','flush'))and event['layers']==expected,
                    'Index20 worker step trace mismatch')
        trace=w['world_step_trace']
        require(sum(e['world_phase']=='prefill' for e in trace)==len(schedule['steps']),
                'Index20 scheduler prefill coverage')
    require(all([e['world_phase']for e in w['world_step_trace']]==
                [e['world_phase']for e in workers[0]['world_step_trace']]for w in workers),
            'Index20 world phase order differs')
    return dict(schema='index20-world-scheduler-proof-v1',phase=phase,request_id=request_id,
        schedule_sha256=schedule['sha256'],all_eight_workers_agreed=True,
        native_first3_executed_once_per_chunk=True,index20_deep_groups=groups,
        deep_native_ep_calls_per_rank=37*groups,algorithm_changed=True,
        original_s12_mathematical_equivalence_claim=False,native_idle_bypass_installed=False)
