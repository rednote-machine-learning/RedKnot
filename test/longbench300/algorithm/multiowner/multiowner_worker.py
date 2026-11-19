"""Actual native scheduler observation and same-order multi-owner EP execution.

Owner computation delegates to its frozen controller. An inactive rank executes
the original placeholder first3, then only qualified zero-local-row native MoE
calls for each agreed deep group. No dummy real tokens enter the request cache.
"""
from types import MethodType,SimpleNamespace as NS
from world_plan_coordinator import (require,seal,validate_world_plan,validate_installation,
    classify_world_step,agree_deep_groups,expected_world_layers)


class MultiOwnerWorkerSchedule:
    def __init__(self,model,*,rank,torch_module,empty_ep_capability,controller=None,engine=None,
                 phase_gather=None,allow_native_decode_graphs=False,active_ranks=(0,)):
        from gpu_pilot.index_hit20_empty_ep_qualification import EmptyEPCapability
        require(type(rank)is int and 0<=rank<8 and list(active_ranks)==sorted(set(active_ranks)),
            'Invalid multi-owner rank inventory')
        require((rank in active_ranks)==(controller is not None),'Controller ownership differs from configured ranks')
        require(isinstance(empty_ep_capability,EmptyEPCapability),'Actual native empty-EP qualification required')
        require(engine is not None and callable(phase_gather) and type(allow_native_decode_graphs)is bool,
            'Native engine, CPU world gather and explicit graph policy required')
        require('_execute_model_async_step'not in engine.__dict__,'Foreign native scheduler hook')
        if controller is None:require('forward'not in model.__dict__,'Foreign inactive model hook')
        else:require(controller.native_model is model and controller.state!='NEW','Owner controller not installed')
        self.model,self.rank,self.t,self.capability=model,rank,torch_module,empty_ep_capability
        self.controller,self.engine,self.phase_gather=controller,engine,phase_gather
        self.active_ranks=list(active_ranks);self.allow_native_decode_graphs=allow_native_decode_graphs
        self.original=model.forward;self.engine_original=engine._execute_model_async_step
        self.symbols=controller.native_symbols if controller is not None else self.original.__func__.__globals__
        self.closed=False;self._phase_done=True;self.spec=None;self._world_frame=None
        self.world_trace=[];self.prefill_trace=[];self.prefill_completed=0;self.flush_agreement=None
        owner=self
        def forward(module,kv_caches,input_metadata,hidden_states=None,dflash=None):
            return owner.forward(kv_caches,input_metadata,hidden_states,dflash)
        def execute(module,seq_metadata_list,scheduler_outputs):
            return owner.execute_step(seq_metadata_list,scheduler_outputs)
        self.hook=MethodType(forward,model);model.forward=self.hook
        self.engine_hook=MethodType(execute,engine);engine._execute_model_async_step=self.engine_hook
        if controller is not None:
            controller.multiowner_worker=self;controller.index20_flush_agreement=self.agree_flush

    def configure(self,spec):
        require(not self.closed and self._phase_done,'Previous multi-owner phase is unfinished')
        self.spec=spec;self.world_trace=[];self.prefill_trace=[];self.prefill_completed=0
        self.flush_agreement=None;self._world_frame=None;self._phase_done=False
        if spec is None:return dict(schema='s13-multiowner-worker-disabled-v1',rank=self.rank,enabled=False)
        validate_world_plan(spec)
        require(set(spec['active_ranks'])<=set(self.active_ranks),'World request assigned to a rank without controller')
        owner=next((o for o in spec['owners']if o['rank']==self.rank),None)
        if owner:
            require(self.controller.state=='RUNNING' and self.controller.rid==owner['request_id']
                and self.controller.batch_schedule==owner['schedule'],'Worker owner is not prepared')
        return dict(schema='s13-multiowner-worker-install-v1',rank=self.rank,enabled=True,
            world_plan_sha256=spec['sha256'],owner_request_id=owner['request_id']if owner else None,
            real_owner=owner is not None,native_scheduler_observer_installed=True,
            full_first3_and_local_request_state=True,world_flush_max_groups=True)

    def execute_step(self,seqs,outputs):
        if self.spec is None:return self.engine_original(seqs,outputs)
        require(not self._phase_done and self._world_frame is None,'Multi-owner scheduler phase inactive/reentrant')
        sample=dict(rank=self.rank,world_plan_sha256=self.spec['sha256'],ordinal=len(self.world_trace),
            prefill_completed=self.prefill_completed,prompt_run=outputs.prompt_run,is_empty=outputs.is_empty(),
            sequences=[dict(request_id=s.request_id,is_prompt=s.is_prompt,seqlen_q=s.seqlen_q,seqlen_k=s.seqlen_k)for s in seqs])
        observations=self.phase_gather(sample)
        event=classify_world_step(observations,self.spec,len(self.world_trace),self.prefill_completed)
        frame=dict(event,local_prompt_run=sample['prompt_run'],local_is_empty=sample['is_empty'],model_consumed=False)
        self._world_frame=frame
        try:
            result=self.engine_original(seqs,outputs)
            if event['world_phase']=='prefill':
                require(frame['model_consumed'],'Actual prefill bypassed multi-owner model dispatch')
                groups=self.flush_agreement['world_groups']if self.flush_agreement else None
                expected=expected_world_layers(self.spec,self.prefill_completed,groups)
                require(frame.get('executed_world_layers')==expected,'Actual local EP execution order differs')
                self.prefill_trace.append(dict(round=self.prefill_completed,layers=expected,
                    real_owner=not sample['is_empty'],owner_range=event['owner_ranges'].get(str(self.rank)),
                    local_deep_groups=frame.get('local_deep_groups'),empty_deep_groups=frame.get('empty_deep_groups')))
                self.prefill_completed+=1
            self.world_trace.append(dict(event,local_is_empty=sample['is_empty'],model_execution_completed=True,
                python_model_callback_consumed=frame['model_consumed'],native_decode_graphs_allowed=self.allow_native_decode_graphs))
            return result
        finally:self._world_frame=None

    def agree_flush(self,count):
        require(self.spec is not None and self.spec['execution_kind']=='index20' and self.flush_agreement is None,
            'Unbound/duplicate multi-owner flush agreement')
        frame=self._world_frame
        require(frame is not None and frame['world_phase']=='prefill'
            and self.prefill_completed==self.spec['native_prefill_rounds']-1,'Flush outside final all-owner prefill')
        owner=next((o for o in self.spec['owners']if o['rank']==self.rank),None)
        local=count if owner else 0
        require((owner is not None and type(count)is int)or(owner is None and count is None),'Flush proposal ownership')
        items=self.phase_gather(dict(rank=self.rank,world_plan_sha256=self.spec['sha256'],
            request_id=owner['request_id']if owner else None,local_groups=local))
        self.flush_agreement=agree_deep_groups(items,self.spec)
        # The unchanged local controller requires its own group count. Padding
        # is executed after its last real group, using the separate world max.
        return local

    def run_empty_deep(self,meta,groups):
        from kv_cache_engram.s1_runtime import compact_metadata
        from kv_cache_engram.s1_system_runtime import inactive_empty_query_cu
        require(type(groups)is int and groups>=0,'Negative padding group count')
        if groups==0:return []
        t=self.t;rows=t.empty((0,),device=meta.positions.device,dtype=t.long)
        compact=compact_metadata(meta,rows,inactive_empty_query_cu(meta,t));executed=[]
        for _ in range(groups):
            for i in range(3,40):
                layer=self.model.layers[i]
                self.capability.permits(layer,meta,NS(full_rows=meta.total_q,selected_count=0))
                bf16=meta.positions.new_empty((0,layer.mlp.hidden_size),dtype=t.bfloat16)
                sf=t.zeros((layer.mlp.hidden_size//128,4),device=meta.positions.device,dtype=t.int32)
                packed=t.empty_like(bf16,dtype=t.float8_e4m3fn),sf[:,:0].transpose(0,1)
                result=layer.mlp(packed,bf16,compact)
                require(result.shape==bf16.shape,'Native zero-local-row MLP returned foreign shape')
                executed.append(i)
        return executed

    def forward(self,caches,meta,hidden,dflash):
        if self.spec is None:return self.original(caches,meta,hidden,dflash)
        frame=self._world_frame
        require(frame is not None and not frame['model_consumed'] and meta.is_prompt is frame['local_prompt_run'],
            'Native model lacks one world-scheduler authorization')
        frame['model_consumed']=True
        if frame['world_phase']!='prefill':return self.original(caches,meta,hidden,dflash)
        if self.spec['execution_kind']=='native':
            result=self.original(caches,meta,hidden,dflash)
            frame['executed_world_layers']=list(range(40));return result
        if not frame['local_is_empty']:
            require(self.controller is not None and self.controller.multiowner_spec==self.spec,'Active owner not bound')
            result=self.original(caches,meta,hidden,dflash)
            local_event=self.controller._batch_step_trace[-1]
            if self.prefill_completed<self.spec['native_prefill_rounds']-1:
                require(local_event['layers']==[0,1,2],'Owner early chunk executed deep layers')
                frame['executed_world_layers']=[0,1,2]
            else:
                agreement=self.flush_agreement;local=agreement['local_groups_by_rank'][self.rank]
                padding=self.controller.multiowner_padding
                require(local_event['layers']==[0,1,2]+list(range(3,40))*local and len(padding)==1,
                    'Owner actual local deep trace/padding coverage differs')
                require(padding[0]['layers']==list(range(3,40))*(agreement['world_groups']-local),
                    'Owner actual empty-EP padding trace differs')
                frame['executed_world_layers']=local_event['layers']+padding[0]['layers']
                frame['local_deep_groups']=local;frame['empty_deep_groups']=agreement['world_groups']-local
            return result
        require(meta.is_prompt is True and meta.num_reqs==1 and meta.total_q==1
            and getattr(meta,'canonical_request_binding',None)is None,'Inactive rank must receive native placeholder only')
        t,g,model=self.t,self.symbols,self.model
        for name in ('input_ids','positions','state_block_idx'):
            t._assert_async((getattr(meta,name)==0).all(),'Inactive rank received real data: '+name)
        hidden=model.embed_tokens(meta.input_ids)if hidden is None else hidden
        require(hidden.ndim==2 and tuple(hidden.shape)==(1,model.layers[0].mlp.hidden_size)
            and hidden.dtype==t.bfloat16 and hidden.is_contiguous(),'Inactive native embedding geometry')
        feature_size=getattr(model,'dflash_input_size',0)
        require((dflash is not None)==bool(feature_size)and(not feature_size or self.allow_native_decode_graphs),
            'Native D-Spark placeholder feature policy')
        features=hidden.new_empty((1,feature_size))if feature_size else None
        lazy,cross=None,g['CrossLayerRuntime']()
        require(model.engram_runtime is not None and meta.ngram_token_ids is not None,'Native Engram unavailable')
        hashes=g['tile_kernels'].engram.engram_hash(meta.ngram_token_ids,model.ngram_multipliers,
            model.ngram_vocab_sizes,model.ngram_offsets,getattr(meta,'images_seq_mask',None))
        hooks=model.engram_runtime.fetch(hashes);executed=[]
        for i in range(3):
            hidden,lazy=model.layers[i](g['Movable'](hidden),caches[i],meta,lazy,
                embed_hook=hooks.get(i),cross_layer_runtime=cross,dflash_input=features);executed.append(i)
        require(callable(hooks.get(14)),'Inactive Engram14 fetch missing');hooks[14]()
        if self.prefill_completed==self.spec['native_prefill_rounds']-1:
            self.agree_flush(None);groups=self.flush_agreement['world_groups']
            executed+=self.run_empty_deep(meta,groups)
            frame['local_deep_groups']=0;frame['empty_deep_groups']=groups
        frame['executed_world_layers']=executed
        return model.allocate_output(1)

    def finish_phase(self):
        require(not self.closed and not self._phase_done,'Worker phase already closed')
        self._phase_done=True
        if self.spec is None:return dict(schema='s13-multiowner-worker-execution-v1',rank=self.rank,enabled=False)
        require(self.prefill_completed==self.spec['native_prefill_rounds'],'Not every native owner prefill completed')
        require(self.spec['execution_kind']=='native'or self.flush_agreement is not None,'Sparse world never agreed final flush')
        return seal(dict(schema='s13-multiowner-worker-execution-v1',rank=self.rank,enabled=True,
            world_plan_sha256=self.spec['sha256'],prefill_rounds=self.prefill_completed,
            prefill_trace=self.prefill_trace,world_step_trace=self.world_trace,flush_agreement=self.flush_agreement,
            native_scheduler_observed=True,local_max_requests=1,
            native_full_layer_sequence_delegated_to_original_model=self.spec['execution_kind']=='native',
            sparse_layer_sequence_observed_from_actual_controller_and_empty_ep=self.spec['execution_kind']=='index20',
            gpu_numerical_parity_claim=False))

    def close(self):
        if self.closed:return
        require(self.model.__dict__.get('forward')is self.hook
            and self.engine.__dict__.get('_execute_model_async_step')is self.engine_hook,'Worker hook ownership changed')
        if self.controller is None:del self.model.forward
        else:
            self.model.forward=self.original;self.controller.multiowner_worker=None
        del self.engine._execute_model_async_step
        self.closed=True


def install_multiowner_worker_schedule(model,**kwargs):return MultiOwnerWorkerSchedule(model,**kwargs)
