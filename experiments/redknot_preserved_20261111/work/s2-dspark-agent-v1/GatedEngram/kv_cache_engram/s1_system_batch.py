"""Actual tensor/metadata assembly for deferred S1 deep-layer microbatches.

Native engine chunks and layers0..2 remain 8192-token calls. This module queues
their selected outputs, consumes the original Engram14 fetch, and materializes
one contiguous-key/noncontiguous-query deep frame. It does not install a model
hook or claim distributed qualification. All EP ranks must first acknowledge
the same installed early/flush schedule; rank0-only deferral is rejected.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

from .core import digest, require, validate_digest
from .s1_plan import plan
from .s1_runtime import LAZY, QUERY_FIELDS, QUERY_MAPS, compact_metadata, gather_hidden, gather_lazy

CHUNK = 8192
REGISTERED_BATCH_GROUPS = (2, 4, 17)


def device_integer_values(values, template, torch):
    """Build exact integer geometry with GPU range kernels, never a host copy.

    Selection and sampler maps consist of a few contiguous document/tail runs.
    One arange kernel writes each whole run; this is not one kernel per token.
    Empty vectors allocate no elements. No CUDA value is inspected by Python.
    """
    values=list(values)
    require(all(type(value) is int for value in values),"device index values must be host integers")
    require(template.dtype in (torch.int32,torch.int64),"device integer template dtype")
    out=torch.empty((len(values),),device=template.device,dtype=template.dtype)
    start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[end]==values[end-1]+1:end+=1
        torch.arange(values[start],values[start]+end-start,out=out[start:end])
        start=end
    return out


def device_cumulative_pair(template, end, torch):
    require(type(end) is int and end>=0,"nonnegative cumulative length required")
    require(template.dtype in (torch.int32,torch.int64),"cumulative length template dtype")
    out=torch.zeros((2,),device=template.device,dtype=template.dtype)
    out[1:].fill_(end)
    return out


def _metadata_qualification_inputs(torch, device, start, end):
    """Small-width diagnostic tensors; no model or numerical-parity claim."""
    n=end-start;t=torch;p=t.arange(start,end,device=device,dtype=t.int64)
    proto=t.empty((0,),device=device,dtype=t.int32)
    keys=p[:,None]-t.arange(127,-1,-1,device=device,dtype=t.int64)[None,:]
    positions_c={};slots={};compressed_cu={}
    for ratio in (1,2):
        key=str(ratio);count=end//ratio-start//ratio
        positions_c[key]=(t.arange(start//ratio,end//ratio,device=device,dtype=t.int32)*ratio
                          if count else t.zeros((1,),device=device,dtype=t.int32))
        slots[key]=(t.arange(start//ratio,end//ratio,device=device,dtype=t.int32)
                    if count else t.full((1,),-1,device=device,dtype=t.int32))
        compressed_cu[key]=device_cumulative_pair(proto,max(1,count),t)
    meta=SimpleNamespace(is_prompt=True,num_reqs=1,total_q=n,input_ids=p.clone(),positions=p,
        valid_token_mask=t.ones((n,),device=device,dtype=t.bool),
        ngram_token_ids=p[:,None].repeat(1,4),positions_c=positions_c,slot_mapping_store_ca=slots,
        cu_compressed_seqlens_q=compressed_cu,cu_seqlens_q=device_cumulative_pair(proto,n,t),
        cu_seqlens_k=device_cumulative_pair(proto,end,t),
        kv_swa_full_index=t.where(keys>=0,keys-start+128,-1).to(t.int32),
        total_k_swa=128,load_segments_swa=t.empty((0,4),device=device,dtype=t.int32),
        total_k_compressed={'1':end,'2':max(1,end//2)},
        load_segments_ca={k:t.empty((0,4),device=device,dtype=t.int32) for k in ('1','2')},
        canonical_request_binding=dict(request_id='metadata-proof',phase='metadata-proof',start=start,end=end))
    hidden=p[:,None].to(t.float32).repeat(1,4)
    lazy=SimpleNamespace(**{name:hidden[None,:,None,:].clone()+i for i,name in enumerate(LAZY)})
    cross=SimpleNamespace(topk_indices=p[:,None].repeat(1,512).to(t.int32),
        block_candidate_idxs=None,candidate_mqa_schedule_metadata={},
        loaded_prefill_kv=t.zeros((128+end//2,1,1),device=device,dtype=t.bfloat16),
        loaded_prefill_kv_main_offset=128)
    return meta,hidden,lazy,cross


def run_s1_system_metadata_qualification(torch_module, allow_sync, *, device='cuda', group_chunks=(2, 4)):
    """Exercise real CUDA batch metadata operators with sync-debug ERROR.

    Only fixture creation, native-independent integer/tensor queue assembly and
    the actual inactive-worker cumulative-length helper are tested. This does
    not replace native multi-rank/FP8/attention/FFN numerical qualification.
    The execution region never calls allow_sync; result readback is afterwards.
    """
    from .s1_system_runtime import inactive_empty_query_cu
    require(isinstance(group_chunks, (tuple, list)) and group_chunks
            and all(type(width) is int and width in REGISTERED_BATCH_GROUPS for width in group_chunks)
            and len(set(group_chunks)) == len(group_chunks), "metadata qualification group registration")
    t=torch_module
    require(t.cuda.is_initialized(),"native model must initialize CUDA before metadata qualification")
    device=t.device(device)
    if device.index is None:device=t.device(device.type,t.cuda.current_device())
    require(device.type=='cuda',"metadata qualification requires actual CUDA")
    before=t.cuda.get_sync_debug_mode();checks=[];group_records=[]
    try:
        t.cuda.set_sync_debug_mode('error')
        with t.inference_mode():
            proto=t.empty((2,),device=device,dtype=t.int32)
            empty_cu=inactive_empty_query_cu(SimpleNamespace(cu_seqlens_q=proto),t)
            t._assert_async((empty_cu==0).all(),"inactive empty cumulative query offsets")
        require(t.cuda.get_sync_debug_mode()==2,"metadata strict mode changed during inactive construction")
        with allow_sync():
            t.cuda.synchronize(device)
            require(empty_cu.tolist()==[0,0],"inactive CUDA cumulative query values")
        checks.append(dict(name='inactive_empty_cu',passed=True,rows=0,cu=[0,0]))
        for width in group_chunks:
            t.cuda.set_sync_debug_mode('error')
            # The new width must exercise an actual full17-member flush, then
            # the original odd one-token tail. A five-chunk fixture alone
            # cannot qualify the wider metadata lifetime/merge geometry.
            input_tokens = 17*CHUNK+1 if width == 17 else 32769
            selected_start = input_tokens-128
            member_chunks = (input_tokens+CHUNK-1)//CHUNK
            docs=[dict(start=0,end=input_tokens,key='metadata-proof')]
            s=batch_schedule(docs,input_tokens,request_id='metadata-proof',phase='metadata-proof',
                phase_generation=1,input_ids_sha256='a'*64,group_chunks=width)
            receipts=[dict(schema='s1-system-batch-rank-installation-v1',rank=rank,
                schedule_sha256=s['sha256'],worker_loop_installed=True,native_first3_per_chunk=True,
                deep_only_on_flush=True,empty_deep_ep_on_flush=True) for rank in range(8)]
            q=CrossChunkDeepBatch(torch_module=t,documents=docs,full_input_tokens=input_tokens,
                request_id='metadata-proof',phase='metadata-proof',phase_generation=1,
                input_ids_sha256='a'*64,group_chunks=width,all_rank_receipts=receipts)
            batches=[];consumed=[]
            with t.inference_mode():
                for step in s['steps']:
                    a,b=step['start'],step['end']
                    meta,hidden,lazy,cross=_metadata_qualification_inputs(t,device,a,b)
                    selected=list(range(max(a,selected_start),b)) if b>selected_start else []
                    rows=device_integer_values([p-a for p in selected],meta.positions,t)
                    def hook(h=hidden):
                        consumed.append(True)
                        return h+100
                    batch=q.push(full_meta=meta,hidden_after_layer2=hidden,lazy_after_layer2=lazy,
                        cross_after_layer2=cross,selected_rows=rows,selected_positions_cpu=selected,
                        engram14_hook=hook)
                    if batch is None:continue
                    expected=t.arange(min(batch.end,max(batch.start,selected_start)),batch.end,device=device,dtype=t.int64)
                    t._assert_async((batch.compact_meta.positions==expected).all(),"batch absolute positions")
                    t._assert_async((batch.hidden[:,0]==expected).all(),"batch selected hidden")
                    for i,name in enumerate(LAZY):
                        t._assert_async((getattr(batch.lazy,name)[0,:,0,0]==expected+i).all(),"batch lazy fields")
                    t._assert_async((batch.cross.topk_indices[:,0]==expected).all(),"batch source2 query topk")
                    t._assert_async((batch.engram14_hook()[:,0]==expected+100).all(),"batch Engram embedding")
                    positions=t.arange(batch.start,batch.end,device=device,dtype=t.int64)
                    local_keys=positions[:,None]-t.arange(127,-1,-1,device=device,dtype=t.int64)[None,:]
                    mask=t.where(local_keys>=0,local_keys-batch.start+128,-1).to(t.int32)
                    t._assert_async((batch.full_meta.kv_swa_full_index==mask).all(),"batch native SWA rebasing")
                    batches.append(batch)
                q.finish()
            require(t.cuda.get_sync_debug_mode()==2,"metadata strict mode changed during active queue")
            with allow_sync():
                t.cuda.synchronize(device)
                require(len(consumed)==member_chunks,"metadata original Engram fetch count")
                for batch in batches:
                    count=len(batch.positions_cpu)
                    require(batch.full_meta.cu_seqlens_q.tolist()==[0,batch.end-batch.start]
                        and batch.full_meta.cu_seqlens_k.tolist()==[0,batch.end]
                        and batch.compact_meta.cu_seqlens_q.tolist()==[0,count],"batch cumulative query/key offsets")
                    for ratio in (1,2):
                        key=str(ratio);ids=list(range(batch.start//ratio,batch.end//ratio))
                        require(batch.full_meta.positions_c[key].tolist()==([i*ratio for i in ids] or [0])
                            and batch.full_meta.slot_mapping_store_ca[key].tolist()==(ids or [-1])
                            and batch.full_meta.cu_compressed_seqlens_q[key].tolist()==[0,max(1,len(ids))],
                            "batch compressed offsets/C2 odd placeholder")
                    expected_dst=(list(range(12,139)) if count==127 else ([0] if count==1 else []))
                    require(batch.prepared['sample_src'].tolist()==list(range(count))
                        and batch.prepared['sample_dst'].tolist()==expected_dst
                        and batch.prepared['sampler_output_rows']==(139 if count==127 else 1),
                        "batch original sampler positions")
                    group_records.append(dict(group_chunks=width,start=batch.start,end=batch.end,
                        selected_rows=count,empty=count==0,sampler_output_rows=batch.prepared['sampler_output_rows'],
                        exact_metadata=True,strict_sync_debug_mode=2))
            checks.append(dict(name='active_queue_group'+str(width),passed=True,input_tokens=input_tokens,
                member_chunks=member_chunks,deep_groups=len(batches),empty_groups=sum(b.selected_count==0 for b in batches),
                selected_rows=sum(b.selected_count for b in batches)))
    finally:
        t.cuda.set_sync_debug_mode(before)
    after=t.cuda.get_sync_debug_mode()
    require(after==before,"metadata qualification changed caller sync guard")
    report=dict(schema='s1-system-cuda-metadata-qualification-v1',passed=True,cuda_exercised=True,
        device_type=device.type,device_index=device.index,sync_debug_mode_during=2,
        sync_debug_mode_before=before,sync_debug_mode_after=after,checks=checks,
        active_queue_groups=group_records,gpu_numerical_parity_claim=False,
        quality_eligible=False,timing_valid=False)
    if tuple(group_chunks) != (2, 4):
        report['requested_batch_groups'] = list(group_chunks)
    return dict(report,sha256=digest(report))


def batch_schedule(documents, full_input_tokens, *, request_id, phase,
                   phase_generation, input_ids_sha256, group_chunks=2):
    require(type(group_chunks) is int and group_chunks in REGISTERED_BATCH_GROUPS,
            "batch group width must be explicitly registered")
    require(type(full_input_tokens) is int and full_input_tokens > 0, "positive batch input length")
    require(isinstance(request_id,str) and request_id and isinstance(phase,str) and phase,
            "batch request/phase identity")
    require(type(phase_generation) is int and phase_generation > 0, "batch phase generation")
    validate_digest(input_ids_sha256, "batch input IDs")
    # Reuse only the existing pure S1 selection definition; no new selection.
    selection = plan(documents,full_input_tokens,0,full_input_tokens)["selected_positions"]
    steps=[];group_start=0
    for ordinal,start in enumerate(range(0,full_input_tokens,CHUNK)):
        end=min(start+CHUNK,full_input_tokens)
        flush=(ordinal+1)%group_chunks==0 or end==full_input_tokens
        steps.append(dict(start=start,end=end,group_start=group_start,flush=flush,
            layers=list(range(40)) if flush else [0,1,2],
            selected_rows=sum(start<=p<end for p in selection),
            final=end==full_input_tokens,nonfinal_sample_discarded=end<full_input_tokens))
        if flush:group_start=end
    value=dict(schema="s1-system-batch-schedule-v1",request_id=request_id,phase=phase,
        phase_generation=phase_generation,input_ids_sha256=input_ids_sha256,
        full_input_tokens=full_input_tokens,documents_sha256=digest(documents),
        group_chunks=group_chunks,native_chunk_tokens=CHUNK,steps=steps,
        first_three_native_per_chunk=True,all_rank_same_ep_layer_order_required=True,
        prompt_logprobs_supported=False,speculative_decode_supported=False)
    return dict(value,sha256=digest(value))


def validate_schedule_agreement(schedule, receipts, *, world_size=8):
    """Validate acknowledgments from actual hook installation, not GPU parity.

    Caller obtains these by an all-rank collective after installing its worker
    loop. A collection of receipts proves only the declared installation scope;
    actual EP/empty-group execution must additionally pass GPU qualification.
    """
    require(schedule.get("sha256")==digest({k:v for k,v in schedule.items() if k!="sha256"}),"batch schedule seal")
    require(type(world_size) is int and world_size==8,"batch requires all eight current EP ranks")
    require(isinstance(receipts,(list,tuple)) and len(receipts)==world_size,"all-rank schedule acknowledgment missing")
    ranks=set()
    for r in receipts:
        require(type(r) is dict and type(r.get("rank")) is int and r["rank"] not in ranks,"batch repeated rank")
        require(r.get("schema")=="s1-system-batch-rank-installation-v1"
                and r.get("schedule_sha256")==schedule["sha256"]
                and r.get("worker_loop_installed") is True
                and r.get("native_first3_per_chunk") is True
                and r.get("deep_only_on_flush") is True
                and r.get("empty_deep_ep_on_flush") is True,"batch rank loop installation differs")
        ranks.add(r["rank"])
    require(ranks==set(range(world_size)),"batch rank scope differs")
    return dict(schedule_sha256=schedule["sha256"],rank_count=world_size,
                installation_receipts_sha256=digest(receipts),gpu_execution_parity_claim=False)


def _clone(value):
    if value is None:return None
    if isinstance(value,dict):return {k:_clone(v) for k,v in value.items()}
    if hasattr(value,"clone"):return value.clone()
    return value


def snapshot_metadata(meta):
    """Own all row/store descriptors that may be reused by a later engine step."""
    require(meta.is_prompt is True and meta.num_reqs==1,"batch is single-request prompt only")
    out=copy(meta)
    names=set(QUERY_FIELDS+QUERY_MAPS+("input_ids","positions_c","slot_mapping_store_ca",
        "cu_seqlens_q","cu_seqlens_k","cu_compressed_seqlens_q","compressed_batch_mapping",
        "kv_swa_full_index","kv_swa_full_index_pivots","load_segments_ca","load_segments_swa",
        "state_block_idx","block_tables"))
    for name in names:
        if hasattr(meta,name):setattr(out,name,_clone(getattr(meta,name)))
    if getattr(meta,"total_k_compressed",None) is not None:
        out.total_k_compressed=dict(meta.total_k_compressed)
    out.canonical_request_binding=dict(meta.canonical_request_binding)
    out.flash_mla_sched_meta_dict=None
    out.mlp_relu_schedule_metadata=None
    return out


def _cat_optional(values, torch, *, dim=0, label="tensor"):
    if all(v is None for v in values):return None
    require(all(v is not None for v in values),"inconsistent optional "+label)
    first=values[0]
    require(all(v.dtype==first.dtype and v.device==first.device for v in values),"mixed dtype/device "+label)
    return torch.cat(values,dim=dim)


def concat_packed(values, torch):
    """Concatenate dense or FP8/SF token rows, preserving native SF stride ABI."""
    require(values,"empty tensor list")
    if not isinstance(values[0],tuple):
        require(all(not isinstance(v,tuple) for v in values),"mixed packed/dense rows")
        return _cat_optional(values,torch,label="dense output")
    require(all(isinstance(v,tuple) and len(v)==2 for v in values),"mixed packed rows")
    data=_cat_optional([v[0] for v in values],torch,label="packed data")
    sf=_cat_optional([v[1] for v in values],torch,label="packed scale")
    rows,columns=sf.shape
    padded=max(4,((rows+3)//4)*4)
    storage=torch.zeros((columns,padded),dtype=sf.dtype,device=sf.device)
    out_sf=storage[:,:rows].transpose(0,1)
    out_sf.copy_(sf)
    return data,out_sf


def concat_lazy(values, torch):
    require(values and all(all(hasattr(v,k) for k in LAZY) for v in values),"four lazy fields required")
    out=copy(values[0])
    for name in LAZY:
        parts=[getattr(v,name) for v in values]
        require(all(v is None or (v.ndim>=2 and v.shape[0]==1) for v in parts),"lazy token axis changed")
        setattr(out,name,_cat_optional(parts,torch,dim=1,label=name))
    return out


def merge_full_metadata(frames, torch):
    """Merge native chunk descriptors without changing absolute key coordinates.

    Query maps concatenate. Complete compressed-store rows concatenate (a
    final odd singleton's native -1 placeholder is omitted when the group has
    real rows). Prefix key-load descriptors come from the last native chunk.
    Native SWA indices are translated from each member's [history128|current]
    coordinates into the group's [history128|current] coordinates.
    """
    require(frames,"empty metadata batch")
    first,last=frames[0],frames[-1];start,end=first.start,last.end
    require(all(a.end==b.start for a,b in zip(frames,frames[1:])),"noncontiguous native chunks")
    metas=[f.meta for f in frames];out=copy(metas[-1]);out.total_q=end-start
    for name in set(QUERY_FIELDS+("input_ids",)):
        parts=[getattr(m,name,None) for m in metas]
        require(all(v is None or v.shape[0]==f.end-f.start for v,f in zip(parts,frames)),"foreign query axis "+name)
        if hasattr(out,name) or any(v is not None for v in parts):
            setattr(out,name,_cat_optional(parts,torch,label=name))
    for name in QUERY_MAPS:
        maps=[getattr(m,name,None) for m in metas]
        if all(m is None for m in maps):setattr(out,name,None);continue
        require(all(isinstance(m,dict) and set(m)==set(maps[0]) for m in maps),"query-map keys changed "+name)
        setattr(out,name,{k:_cat_optional([m[k] for m in maps],torch,label=name+":"+k) for k in maps[0]})
    out.positions_c={};out.slot_mapping_store_ca={};out.cu_compressed_seqlens_q={}
    ratios=set(metas[-1].positions_c)
    require(all(set(m.positions_c)==ratios and set(m.slot_mapping_store_ca)==ratios for m in metas),"compression ratio maps changed")
    for key in ratios:
        ratio=int(key);count=end//ratio-start//ratio
        real=[(m,f) for m,f in zip(metas,frames) if f.end//ratio-f.start//ratio>0]
        source=real if count else [(metas[-1],frames[-1])]
        for m,f in source:
            expected=max(1,f.end//ratio-f.start//ratio)
            require(m.positions_c[key].shape[0]==expected and m.slot_mapping_store_ca[key].shape[0]==expected,
                    "compressed store shape differs")
        out.positions_c[key]=torch.cat([m.positions_c[key] for m,_ in source],dim=0)
        out.slot_mapping_store_ca[key]=torch.cat([m.slot_mapping_store_ca[key] for m,_ in source],dim=0)
        require(out.positions_c[key].shape[0]==max(1,count),"merged compressed row count")
        out.cu_compressed_seqlens_q[key]=device_cumulative_pair(metas[-1].cu_compressed_seqlens_q[key],max(1,count),torch)
    out.cu_seqlens_q=device_cumulative_pair(metas[-1].cu_seqlens_q,end-start,torch)
    out.cu_seqlens_k=device_cumulative_pair(metas[-1].cu_seqlens_k,end,torch)
    matrices=[]
    for m,f in zip(metas,frames):
        indices=m.kv_swa_full_index
        require(indices is not None and tuple(indices.shape)==(f.end-f.start,128),"native SWA matrix missing")
        matrices.append(torch.where(indices>=0,indices+(f.start-start),indices))
    out.kv_swa_full_index=torch.cat(matrices,dim=0)
    out.kv_swa_full_index_pivots=None
    out.compressed_batch_mapping=None
    out.load_segments_swa=_clone(metas[0].load_segments_swa)
    out.total_k_swa=metas[0].total_k_swa
    require(out.total_k_swa==128 and all(m.total_k_swa==128 for m in metas),"CP1 native history width changed")
    out.flash_mla_sched_meta_dict=None;out.mlp_relu_schedule_metadata=None
    marker=dict(metas[-1].canonical_request_binding);marker.update(start=start,end=end)
    out.canonical_request_binding=marker
    out.s1_system_native_member_chunks=[(f.start,f.end) for f in frames]
    out.s1_system_deep_batch=True
    return out


@dataclass
class DeepBatch:
    start: int
    end: int
    member_chunks: list
    hidden: object
    lazy: object
    full_meta: object
    compact_meta: object
    cross: object
    selected_rows: object
    positions_cpu: list
    prepared: dict
    plan: dict
    engram14_hook: object
    report: dict
    engram14_consumption: list
    member_metadata: list
    member_selected_counts: list

    @property
    def selected_count(self):
        return len(self.positions_cpu)


class CrossChunkDeepBatch:
    def __init__(self, *, torch_module, documents, full_input_tokens, request_id,
                 phase, phase_generation, input_ids_sha256, group_chunks=2,
                 all_rank_receipts=None, prompt_logprobs=False, speculative=False):
        require(prompt_logprobs is False and speculative is False,"batch forbids prompt logprobs/speculation")
        self.t=torch_module;self.docs=documents
        self.schedule=batch_schedule(documents,full_input_tokens,request_id=request_id,phase=phase,
            phase_generation=phase_generation,input_ids_sha256=input_ids_sha256,group_chunks=group_chunks)
        self.agreement=validate_schedule_agreement(self.schedule,all_rank_receipts)
        self.frames=[];self.next_step=0;self.state="ACTIVE";self.events=[];self.consumptions=[]

    def _step(self,meta,positions):
        require(self.state=="ACTIVE" and self.next_step<len(self.schedule["steps"]),"batch queue is closed/exhausted")
        step=self.schedule["steps"][self.next_step];marker=getattr(meta,"canonical_request_binding",None)
        require(isinstance(marker,dict) and all(marker.get(k)==self.schedule[k] for k in ("request_id","phase")),"batch foreign request/phase")
        require((marker.get("start"),marker.get("end"))==(step["start"],step["end"]),"batch native step order")
        require(meta.is_prompt is True and meta.num_reqs==1 and meta.total_q==step["end"]-step["start"],"batch native input geometry")
        expected=plan(self.docs,self.schedule["full_input_tokens"],step["start"],step["end"])["selected_positions"]
        require(list(positions)==expected,"batch selection changed")
        return step

    def push(self, *, full_meta, hidden_after_layer2, lazy_after_layer2,
             cross_after_layer2, selected_rows, selected_positions_cpu, engram14_hook):
        step=self._step(full_meta,selected_positions_cpu);t=self.t
        require(selected_rows.numel()==len(selected_positions_cpu),"batch selected gather count")
        expected_rows=device_integer_values([p-step["start"] for p in selected_positions_cpu],selected_rows,t)
        t._assert_async((selected_rows==expected_rows).all(),"batch selected row identity")
        require(callable(engram14_hook),"batch must consume original Engram14 fetch")
        require(cross_after_layer2.topk_indices is not None
            and tuple(cross_after_layer2.topk_indices.shape)==(full_meta.total_q,512),"native layer2 topk rows")
        require(cross_after_layer2.block_candidate_idxs is None,"candidate selection appeared before layer20")
        # Gather owns fresh tensors, including all lazy fields. Native Engram
        # buffer is consumed now before the next engine chunk may reuse it.
        hidden=gather_hidden(hidden_after_layer2,selected_rows,t)
        lazy=gather_lazy(lazy_after_layer2,selected_rows)
        embedding=gather_hidden(engram14_hook(),selected_rows,t)
        meta=snapshot_metadata(full_meta)
        cross=copy(cross_after_layer2)
        cross.topk_indices=cross_after_layer2.topk_indices.index_select(0,selected_rows)
        cross.candidate_mqa_schedule_metadata={}
        frame=SimpleNamespace(start=step["start"],end=step["end"],hidden=hidden,lazy=lazy,embedding=embedding,
            meta=meta,cross=cross,rows=selected_rows.clone(),positions=list(selected_positions_cpu))
        self.frames.append(frame);self.next_step+=1
        if not step["flush"]:return None
        return self._flush()

    def _flush(self):
        frames=self.frames;t=self.t;first,last=frames[0],frames[-1]
        full=merge_full_metadata(frames,t);positions=[p for f in frames for p in f.positions]
        require(len(positions)<=8192,"merged selected rows exceed the unchanged native EP/Engram buffer capacity")
        rows=t.cat([f.rows+(f.start-first.start) for f in frames])
        pos=t.cat([f.meta.positions.index_select(0,f.rows) for f in frames])
        cu=device_cumulative_pair(full.cu_seqlens_q,len(positions),t)
        compact=compact_metadata(full,rows,cu)
        cross=copy(last.cross);cross.topk_indices=t.cat([f.cross.topk_indices for f in frames])
        cross.shared_hidden=None;cross.consumer_full_hidden=None;cross.consumer_full_metadata=None
        cross.candidate_mqa_lens=None;cross.candidate_mqa_schedule_metadata={}
        # Source2 loaded main/index data belong to the last original chunk and
        # cover the entire prefix through end. Earlier topk row decisions above
        # remain unchanged and retain their individual causal bounds.
        p=plan(self.docs,self.schedule["full_input_tokens"],first.start,last.end)
        sampler=plan(self.docs,self.schedule["full_input_tokens"],last.start,last.end)
        lookup={p:i for i,p in enumerate(positions)}
        pairs=[(i,lookup[q]) for i,q in enumerate(sampler["sampler_output_positions"]) if q in lookup]
        prepared=dict(rows=rows,positions=pos,cu_q=cu,
            sample_dst=device_integer_values([a for a,_ in pairs],rows,t),
            sample_src=device_integer_values([b for _,b in pairs],rows,t),
            sampler_output_rows=len(sampler["sampler_output_positions"]))
        prepared["plan"]=p
        embedding=concat_packed([f.embedding for f in frames],t);consumed=[False]
        def hook():
            require(not consumed[0],"merged Engram14 consumed twice")
            consumed[0]=True
            return embedding
        report=dict(schema="s1-system-deep-batch-v1",schedule_sha256=self.schedule["sha256"],
            phase_generation=self.schedule["phase_generation"],start=first.start,end=last.end,
            member_chunks=[(f.start,f.end) for f in frames],selected_positions=positions,
            selected_rows=len(positions),first3_native_calls=len(frames),deep_layer_calls=37,
            deep_layer_calls_are_planned_not_measured=True,
            all_rank_installation=self.agreement,original_engram14_fetches_consumed=len(frames),
            no_unselected_deep_rows=True,numerical_parity_claim=False,
            sampler_output_rows=prepared["sampler_output_rows"])
        report["sha256"]=digest(report)
        out=DeepBatch(first.start,last.end,report["member_chunks"],concat_packed([f.hidden for f in frames],t),
            concat_lazy([f.lazy for f in frames],t),full,compact,cross,rows,positions,prepared,p,hook,report,consumed,
            [f.meta for f in frames],[len(f.positions) for f in frames])
        self.frames=[];self.events.append(report);self.consumptions.append(consumed)
        return out

    def finish(self):
        require(self.state=="ACTIVE" and not self.frames and self.next_step==len(self.schedule["steps"]),
                "batch final flush missing")
        require(all(x[0] for x in self.consumptions),"merged Engram14 hook was not consumed by deep execution")
        self.state="COMPLETE"
        return dict(schedule=self.schedule,groups=self.events,all_rank_installation=self.agreement,
                    numerical_parity_claim=False)

    def discard(self):
        self.frames=[];self.state="FAILED"


def deferred_sampler_output(reference, *, full_input_tokens, start, end,
                            prompt_logprobs=False, speculative=False):
    """Only an ignored nonfinal sample may receive a placeholder output slot."""
    require(prompt_logprobs is False and speculative is False and 0<=start<end<full_input_tokens,
            "cannot defer final sample/prompt-logprobs/speculative work")
    clip=max(0,min(full_input_tokens-140-start,end-start-1))
    return reference.new_zeros((end-start-clip,reference.shape[-1]))
