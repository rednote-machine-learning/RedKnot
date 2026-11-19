"""Cone15 uses the original B4/D-Spark/EP runtime, changing only live post-20 output rows."""
from contextlib import ExitStack, contextmanager
from copy import copy
import hashlib
from types import MethodType, SimpleNamespace

from cone15_plan import Cone15Plan, local_indices, row_receipt
from cone15_ffn_native_proof import private_tail_proof, require

CANDIDATE = 'Cone15'
REFERENCE = 'Fixed15Opt'
LAYERS = tuple(range(21,40))


def collect_world_guard(runtime):
    from deepinfer import PLATFORM, envs
    from deepinfer.model_executor import eplb
    from deepinfer.model_executor.models.v4_modules.mlp import DeepseekMoE
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    cap=runtime.batch_worker.capability
    require(PLATFORM.is_sm90 and PG.expert_parallel_group().size()==8 and PG.context_parallel_group().size()==1
            and envs.DEEPINFER_BATCH_INVARIANT and envs.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL==0
            and not envs.DEEPINFER_FORCE_LOAD_BALANCE and envs.DEEPINFER_FORCE_IMBALANCE_RATIO==0,'Unsupported routing')
    for i in LAYERS:
        mod=runtime.layers[i].mlp
        balancer,_=eplb.current()._layer_refs[i]
        require(type(mod) is DeepseekMoE and mod.layer_idx==i and mod.ep_size==8 and mod.use_fp8_act
                and not mod.use_mega_moe and mod.num_extra_experts==0 and balancer._rearrangement_interval==0
                and not balancer._schedule_rearrangement_after_prefill and cap.bindings[i][0] is runtime.layers[i]
                and cap.bindings[i][1] is mod and runtime.layers[i].hc_enabled,'Unqualified post-20 MLP/mHC')
    local=dict(rank=runtime.rank,world=8,cp=1,sm=90,layers=list(LAYERS),rearrangement_interval=0,
               num_extra_experts=0,random_routing=False,native_empty_ep=cap.sha256)
    world=runtime.driver.gather(local)
    require([v['rank'] for v in world]==list(range(8)) and len({v['native_empty_ep'] for v in world})==1,
            'Incomplete eight-rank expert-parallel guard')
    if runtime.controller is not None:
        runtime.controller._cone_world=world


def fingerprint(t, value):
    raw = value.detach().contiguous().view(t.uint8).cpu().numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())


@contextmanager
def temporary_method(obj, name, call):
    """Restore the particular module's original override/descriptor on every exit."""
    existed = name in obj.__dict__
    old = obj.__dict__.get(name)
    bound = MethodType(call, obj)
    setattr(obj,name,bound)
    try:
        yield
    finally:
        require(obj.__dict__.get(name) is bound, 'Terminal method ownership changed')
        if existed:
            setattr(obj,name,old)
        else:
            delattr(obj,name)


def gpu_index_bank(t, device, groups):
    """One transfer of unique sorted CPU maps for this single owner's group."""
    offsets={};combined=[]
    for rows in groups:
        if rows not in offsets:
            offsets[rows]=(len(combined),len(rows));combined.extend(rows)
    host=t.tensor(combined,dtype=t.int64,device='cpu',pin_memory=True)
    device_buffer=host.to(device=device,non_blocking=True)
    return host,{rows:device_buffer.narrow(0,a,n) for rows,(a,n) in offsets.items()}


def expand_hidden(compact, total, ids):
    full=compact.new_zeros((total,*compact.shape[1:]))
    full.index_copy_(0,ids,compact)
    return full


def expand_lazy(compact, total, ids, *, placeholder):
    from kv_cache_engram.s1_runtime import LAZY
    full=copy(compact)
    for field in LAZY:
        value=getattr(compact,field)
        if value is None:
            continue
        require(value.ndim>=2 and value.shape[0]==1,'Unknown native mHC row layout')
        expanded=value.new_zeros((1,total,*value.shape[2:]))
        if not placeholder:
            expanded.index_copy_(1,ids,value)
        setattr(full,field,expanded)
    return full


@contextmanager
def compact_ffn(controller, layer, group, rows, live, ids, reference, qualify, evidence):
    from kv_cache_engram.index_hit20_adapter import TorchFFNOnlyOps, make_selected_mlp_metadata
    from kv_cache_engram.s1_runtime import gather_lazy
    t=controller.t;moe=layer.mlp;original=moe.forward;native_norm=layer.ffn_norm_forward
    count,keep=len(rows),len(live);norm_count=0;mlp_count=0
    packed_norm_result=None;ops=TorchFFNOnlyOps()
    evidence.update(layer=layer.layer_idx,live_ffn_rows=keep,live_positions_sha256=row_receipt(live)['sha256'],
                    actual_ffn_rows=None,dispatch_rows=None,combine_calls=0,actual_ffn_norm_rows=0,
                    actual_ffn_mhc_rows=0,norm_placeholder_rows=0,qualification_live_ffn_sha256=None,
                    qualification_live_ffn_shape=None,qualification_row_ids_sha256=None)

    def norm(_,hidden,residual):
        nonlocal norm_count,packed_norm_result
        norm_count+=1
        require(norm_count==1 and count and hidden.shape[0]==count,'Unexpected full attention output/mHC shape')
        if reference or keep==count:
            evidence['actual_ffn_norm_rows']=count
            result=native_norm(hidden,residual)
        else:
            # Existing zero-local-row native MoE supports no local sender. For an
            # entirely dead owner group with nonempty old geometry, compute one actual
            # norm row for native dtype/shape discovery, then discard its output.
            indices=ids if keep else t.zeros((1,),device=hidden.device,dtype=t.long)
            evidence['norm_placeholder_rows']=int(not keep)
            evidence['actual_ffn_norm_rows']=indices.numel()
            compact=native_norm(hidden.index_select(0,indices),gather_lazy(residual,indices))
            packed,bf16,lazy=compact
            if not keep:
                packed,bf16=ops.gather(packed,ids),ops.gather(bf16,ids)
            result=packed,bf16,expand_lazy(lazy,count,ids,placeholder=not keep)
        evidence['actual_ffn_mhc_rows']=evidence['actual_ffn_norm_rows']
        packed_norm_result=result[1]
        return result

    def forward(_,packed,bf16,meta):
        nonlocal mlp_count,packed_norm_result
        mlp_count+=1
        physical=count if reference else keep
        require(mlp_count==1 and controller._active_deep_batch is group and meta is group.compact_meta
                and meta.is_prompt and meta.num_reqs==1 and bf16.shape[0]==physical
                and (not count or bf16 is packed_norm_result),'Unexpected cone FFN native input')
        packed_norm_result=None
        require(not count or layer.layer_idx not in (37,38,39) or controller.dflash_bridge.frame==(layer.layer_idx,rows),
                'D-Spark did not capture the original selected input frame')
        if qualify:
            with controller.allow_sync():
                require(meta.gate_force_random is None or not bool(meta.gate_force_random.any().item()),'Random gate active')
                gpu_positions=meta.positions.index_select(0,ids).cpu().tolist()
                require(gpu_positions==list(live),'Physical device row IDs differ from planned live rows')
                evidence['qualification_row_ids_sha256']=row_receipt(gpu_positions)['sha256']
        plan=SimpleNamespace(full_rows=count,selected_count=keep,selected_rows=ids)
        if count and not keep and not reference:
            controller.empty_ep_capability.permits(layer,meta,plan)
        active=meta if reference or keep==count else make_selected_mlp_metadata(layer,meta,plan)
        dispatch,combine=moe.dispatch,moe.combine
        def send(_,x,idx,weights,m,evt):
            require(evidence['dispatch_rows'] is None and m is active and idx.shape[0]==weights.shape[0]==physical,
                    'Native EP dispatch got a different number of rows')
            evidence['dispatch_rows']=idx.shape[0]
            return dispatch(x,idx,weights,m,evt)
        def receive(_,x,shared,handle,evt):
            evidence['combine_calls']+=1
            require(evidence['combine_calls']==1,'Duplicate native EP combine')
            return combine(x,shared,handle,evt)
        with temporary_method(moe,'dispatch',send),temporary_method(moe,'combine',receive):
            output=original(packed,bf16,active)
        require(evidence['dispatch_rows']==physical and evidence['combine_calls']==1 and output.shape==bf16.shape,
                'Incomplete original EP execution')
        evidence['actual_ffn_rows']=bf16.shape[0]
        if qualify:
            with controller.allow_sync():
                result=output.index_select(0,ids) if reference else output
                require(result.dtype==t.bfloat16,'Expected native BF16 FFN result')
                value=fingerprint(t,result)
                evidence['qualification_live_ffn_sha256']=value['sha256']
                evidence['qualification_live_ffn_shape']=value['shape']
        return output if physical==count else expand_hidden(output,count,ids)

    with temporary_method(layer,'ffn_norm_forward',norm),temporary_method(moe,'forward',forward):
        yield
    require(mlp_count==1 and norm_count==int(count>0),'Native FFN/norm count differs')


def enabled(c):
    return getattr(c,'fixed_pack_enabled',False) and c._fixed_pack_method in (REFERENCE,CANDIDATE)


def cone15_controller_class(base):
    private_tail_proof(base)

    class ConeController(base):
        def reset(self,*args,**kwargs):
            self._cone_plan=None;self._cone_groups=[];self._cone_device_ids=None
            self._cone_layer_state=None;self._cone_qualified_cache=None
            return super().reset(*args,**kwargs)

        def _run_deep_group(self,model,caches,group):
            parent=super()._run_deep_group
            if not enabled(self):
                return parent(model,caches,group)
            from cone_input import compact_inputs
            from cone_query import compact_core
            from index20_runtime import partition_native_frames
            require(self._cone_device_ids is None and self._cone_world,'Missing world guard or overlapping owner group')
            n=len(self.ids)
            if self._cone_plan is None:
                final=(n-1)//8192*8192
                sel=self._index20_selection
                chunks=[(a,min(n,a+8192)) for a in range(0,n,8192)]
                ranges=[(units[0][0],units[-1][1]) for units in partition_native_frames(chunks,sel['selected_positions'])]
                self._cone_plan=Cone15Plan(n,sel['selected_positions'],sel['mandatory_positions'],last_frame_start=final,group_ranges=ranges)
            plan=self._cone_plan;rows=tuple(group.positions_cpu);a,b=group.start,group.end
            require(plan.selection(a,b)==rows,'Frozen group was not bound to real selected rows')
            qualifies=self.phase.endswith('.qualification');reference=self._fixed_pack_method==REFERENCE
            geometry={l:plan.group(l,a,b) for l in LAYERS}
            maps={l:local_indices(rows,v.queries) for l,v in geometry.items()}
            input_maps={l:local_indices(rows,v.inputs) for l,v in geometry.items()}
            host,device=gpu_index_bank(self.t,group.compact_meta.positions.device,[*maps.values(),*input_maps.values()])
            self._cone_device_ids=device
            reports={l:{} for l in LAYERS}
            self._cone_layer_state=geometry,maps,reports
            try:
                with ExitStack() as stack:
                    for i in LAYERS:
                        layer=self.layers[i];positions=geometry[i].queries
                        stack.enter_context(compact_inputs(self,layer,group,input_maps[i],qualifies,reports[i]))
                        stack.enter_context(compact_core(self,layer,group,maps[i],qualifies,reports[i]))
                        stack.enter_context(compact_ffn(self,layer,group,rows,positions,device[maps[i]],reference,qualifies,reports[i]))
                    out=parent(model,caches,group)
                sampler=None
                if qualifies and b==n:
                    require(out is not None,'Missing terminal sampler')
                    with self.allow_sync():sampler=fingerprint(self.t,out)
                self._cone_groups.append(dict(start=a,end=b,layers=[reports[i] for i in LAYERS],
                    qualification_sampler_sha256=None if sampler is None else sampler['sha256'],
                    qualification_sampler_shape=None if sampler is None else sampler['shape']))
                return out
            finally:
                self._cone_layer_state=self._cone_device_ids=None
                del host

        def _observe_layer(self,layer,start,end,result,meta):
            first=len(self.decoder_events)
            super()._observe_layer(layer,start,end,result,meta)
            if layer not in LAYERS or not enabled(self):
                return
            geom,_,reports=self._cone_layer_state
            live=geom[layer].queries;ev=reports[layer];selected=tuple(self._active_deep_batch.positions_cpu)
            reference=self._fixed_pack_method==REFERENCE
            physical=selected if reference else live
            require(ev['actual_ffn_rows']==len(physical) and ev['dispatch_rows']==len(physical),
                    'Decoder row observer ran before original MoE')
            attention=self.attention.events[-1]
            require(attention['layer']==layer and (attention['chunk_start'],attention['chunk_end'])==(start,end)
                    and tuple(attention['selected_positions'])==selected,'Incorrect original SWA group event')
            input_stats=ev['inputs']
            in_rows=selected if reference or input_stats['empty_live_uses_native'] else geom[layer].inputs
            require(input_stats['actual_attn_norm_rows']==input_stats['actual_q_a_swa_rows']==len(in_rows),
                    'Physical input normalization/projection row count differs')
            require(ev['core']['actual_attention_rows']==len(physical),'Real native attention core rows differ')
            attention.update(actual_attention_rows=len(physical),selected_wrapper_rows=len(selected),
                             actual_swa_projection_rows=len(in_rows),actual_query_projection_rows=len(physical),
                             cone15_core_positions=list(physical),cone15_input_positions=list(in_rows))
            for event in self.decoder_events[first:]:
                a,b=event['chunk_start'],event['chunk_end'];positions=[p for p in physical if a<=p<b]
                normrows=len(positions)+int(ev['norm_placeholder_rows']==1 and a<=selected[0]<b)
                inputs=[p for p in in_rows if a<=p<b]
                # A selected group with no live output can need one real norm row as a shape witness.
                mhc=sorted(set(inputs)|({selected[0]} if ev['norm_placeholder_rows'] and a<=selected[0]<b else set()))
                event.update(actual_ffn_rows=len(positions),actual_attention_rows=len(positions),cone15_ffn_positions=positions,
                    cone15_core_positions=positions,cone15_input_positions=inputs,cone15_mhc_positions=mhc,
                    selected_wrapper_rows=event['selected_rows'],actual_query_projection_rows=len(positions),
                    actual_swa_projection_rows=len(inputs),actual_residual_rows=len(mhc),actual_mhc_rows=len(mhc),
                    actual_attn_norm_rows=len(inputs),actual_ffn_norm_rows=normrows,actual_ffn_mhc_rows=normrows)

        def _commit_dflash_tail(self,meta,dflash,caches,end):
            if not (enabled(self) and self.phase.endswith('.qualification') and end==len(self.ids)):
                return super()._commit_dflash_tail(meta,dflash,caches,end)
            collector=self.dflash_bridge.collector
            require(tuple(collector.targets)==(37,38,39) and collector.buffer is not None and
                    all(len(collector.coverage[l])==len(collector.positions) for l in collector.targets),
                    'Incomplete input features for draft model')
            with self.allow_sync():features=fingerprint(self.t,collector.buffer)
            result=super()._commit_dflash_tail(meta,dflash,caches,end)
            with self.allow_sync():
                state=int(meta.state_block_idx[0].item())
                positions=self.t.arange(collector.start,end,device=meta.positions.device,dtype=self.t.long)
                rings=[]
                for i in [*LAYERS,*[layer.layer_idx for layer in dflash.layers]]:
                    cache=caches[i].swa_cache
                    valid=cache[state].index_select(0,positions%cache.shape[1])
                    rings.append(dict(layer=i,valid_swa=fingerprint(self.t,valid)))
                self._cone_qualified_cache=dict(dflash_features=features,rings=rings)
            return result

        def finish(self):
            report=super().finish()
            if not enabled(self):return report
            from index20_selector import seal
            plan=self._cone_plan
            require(plan is not None and [(g['start'],g['end']) for g in self._cone_groups]==list(plan.group_ranges),
                    'Missing native Cone15 group evidence')
            mode=self._fixed_pack_method
            if mode==CANDIDATE:
                info=dict(report['index20_lastkv']);info.pop('sha256')
                live=sum(len(plan.group(39,a,b).inputs) for a,b in plan.group_ranges)
                info.update(selected_hidden_ffn_attention_pruning=True,
                    hidden_policy='Fixed15 selected layout; post-source20 hidden regenerated only for dependency cone',
                    selected_rows_replayed_from_dependency_complete_inputs=live,
                    selected_rows_replayed_from_dead_placeholder_inputs=len(plan.selected)-live,
                    legacy_live_selected_rows_field='selected positions in physical replay buffer',
                    all_input_kv_bytes_equal_to_fixed15_reference_claim=False,
                    decode_observable_swa_rows_per_layer=min(len(self.ids),128))
                report['index20_lastkv']=seal(info)
            report['cone15_ffn']=dict(method=mode,worker_guards=self._cone_world,groups=self._cone_groups,
                qualification_cache_bytes=self._cone_qualified_cache,extra_native_collectives=0,
                expert_statistics='physical_dispatched_rows',query_axes='absolute selected position order',
                new_and_tail_full_ffn_each_layer=True,all_input_lastkv_physical_replay=True,
                raw_dead_intermediate_and_final_kv_numerical_equality_claim=False)
            return report
    return ConeController
