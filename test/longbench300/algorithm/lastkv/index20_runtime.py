"""Independent online Index20 controller, retaining native first-three layers.

Every native first-three-layer chunk executes once.  Its hidden/mHC/Engram
state is owned until the full native index-logit reduction has selected rows.
Only selected rows enter layers 3..39; chronological groups obey the unchanged
8192-row EP/Engram capacity.  No frozen S1.2 source is patched.
"""
from __future__ import annotations

from copy import copy
from types import SimpleNamespace as NS

from index20_selector import NativeIndexScoreCollector, mandatory_positions, require, seal
from index20_plan import make_plan


def mandatory_dflash_prepared(documents,total):
    mandatory=mandatory_positions(documents,total)
    return {(a,min(a+8192,total)):dict(
        plan=make_plan(documents,total,mandatory,a,min(a+8192,total)),
        purpose="mandatory_dflash_admission_only_not_final_deep_selection")
        for a in range(0,total,8192)}


def partition_native_frames(chunks, selected, *, capacity=8192):
    """Partition whole, consecutive native frames; never split causal state."""
    require(type(capacity) is int and capacity > 0, "invalid deep row capacity")
    require(chunks and chunks[0][0] == 0 and
            all(a[1] == b[0] for a,b in zip(chunks,chunks[1:])), "noncontiguous frames")
    require(selected == sorted(set(selected)) and all(0 <= p < chunks[-1][1] for p in selected),
            "invalid selected positions")
    groups=[]; current=[]; count=0
    for a,z in chunks:
        n=sum(a <= p < z for p in selected)
        require(0 < z-a <= 8192 and n <= capacity, "native frame exceeds deep capacity")
        if current and count+n > capacity:
            groups.append(current); current=[]; count=0
        current.append((a,z)); count+=n
    if current: groups.append(current)
    require([c for g in groups for c in g] == list(chunks), "partition lost original frames")
    return groups


def index20_schedule(documents, total, *, request_id, phase, phase_generation, input_ids_sha256):
    require(type(total) is int and total > 0, "positive input length required")
    steps=[]
    for a in range(0,total,8192):
        z=min(a+8192,total); final=z==total
        steps.append(dict(start=a,end=z,flush=final,layers=list(range(40)) if final else [0,1,2],
            final=final,nonfinal_sample_discarded=not final,
            deep_layers_dynamic_after_selection=final))
    return seal(dict(schema="index20-first3-finalflush-v1",algorithm="index20",
        request_id=request_id,phase=phase,phase_generation=phase_generation,
        input_ids_sha256=input_ids_sha256,full_input_tokens=total,
        documents_sha256=seal({"documents":documents})["sha256"],
        native_chunk_tokens=8192,group_chunks=len(steps),steps=steps,
        first_three_native_per_chunk=True,all_rank_same_ep_layer_order_required=True,
        max_deep_group_selected_tokens=8192,dynamic_deep_groups=True,selection_available_after_all_first3=True,
        prompt_logprobs_supported=False,native_speculative_decode_preserved=True))


def validate_installation(schedule, receipts):
    require(isinstance(receipts,(list,tuple)) and len(receipts)==8,
            "Index20 requires eight installed worker loops")
    for rank,item in enumerate(receipts):
        require(item.get("schema")=="index20-worker-install-v1" and item.get("rank")==rank
            and item.get("schedule_sha256")==schedule["sha256"]
            and item.get("worker_loop_installed") is True
            and item.get("native_first3_per_chunk") is True
            and item.get("deep_only_on_final_flush") is True
            and item.get("dynamic_flush_count_agreement") is True
            and item.get("world_scheduler_phase_bound") is True,
            "Index20 worker installation receipt mismatch")
    return dict(schedule_sha256=schedule["sha256"],rank_count=8,
        installation_receipts_sha256=seal({"receipts":receipts})["sha256"],
        gpu_execution_parity_claim=False)


def mandatory_tail_update_proof(documents,total,decoder_events,global_events,attention_events):
    """Validate actual per-layer reports, including complete C2 pair updates."""
    tails=sorted({p for d in documents for p in range(d["end"]-128,d["end"])})
    checked=[]
    for layer in range(40):
        events=[e for e in decoder_events if e["layer"]==layer]
        require(len(events)==(total+8191)//8192,"mandatory tail decoder coverage")
        seen=[]
        for e in events:
            wanted=[p for p in tails if e["chunk_start"]<=p<e["chunk_end"]]
            require(set(wanted)<=set(e["selected_positions"]),"mandatory tail skipped a decoder layer")
            require(e["actual_ffn_rows"]==e["selected_rows"]
                and e["actual_residual_rows"]==e["selected_rows"]
                and e["actual_mhc_rows"]==e["selected_rows"],"mandatory tail hidden pipeline incomplete")
            seen.extend(wanted)
        require(sorted(seen)==tails,"mandatory tails have missing/duplicate decoder coverage")
    for layer in range(3,40):
        events=[e for e in attention_events if e["layer"]==layer];seen=[]
        for e in events:
            wanted=[p for p in tails if e["chunk_start"]<=p<e["chunk_end"]]
            require(set(wanted)<=set(e["selected_positions"])
                and e["actual_attention_rows"]==e["selected_rows"]
                and e["actual_native_swa_indices_checked"] is True,
                "mandatory online tail SWA recomputation missing")
            seen.extend(wanted)
        require(sorted(seen)==tails,"mandatory tail SWA coverage")
    for layer in (8,14,20):
        events=[e for e in global_events if e["layer"]==layer];seen=[];cursor=0
        ratio=1 if layer==20 else 2
        for e in events:
            require(e["chunk_start"]==cursor and e["compression_ratio"]==ratio,
                    "mandatory tail global source order")
            cursor=e["chunk_end"]
            wanted=[p for p in tails if e["chunk_start"]<=p<e["chunk_end"]]
            required={p//ratio for p in wanted if p//ratio<e["chunk_end"]//ratio}
            pending=[p for p in wanted if p//ratio>=e["chunk_end"]//ratio]
            require(set(wanted)<=set(e["online_projection_positions"])
                and required<=set(e["affected_compressed_ids"])
                and e["main_and_index_k_native_store"] is True
                and e["source2_untouched"] is True,"mandatory tail global/main/index refresh missing")
            require(not pending or (ratio==2 and e["native_compressor_carry_updated"] is True),
                    "mandatory tail incomplete C2 pair carry missing")
            seen.extend(wanted)
            checked.append(dict(layer=layer,start=e["chunk_start"],end=e["chunk_end"],
                mandatory_tail_rows=len(wanted),affected_complete_blocks=len(required),
                incomplete_pair_tail_rows_in_native_carry=len(pending),main_and_index_refreshed=True))
        require(cursor==total and sorted(seen)==tails,"mandatory tail global coverage incomplete")
    return seal(dict(schema="index20-mandatory-tail-updates-v1",mandatory_tail_tokens=len(tails),
        every_decoder_layer_full_tail_hidden_pipeline=True,online_tail_swa_layers=list(range(3,40)),
        source2_native_full_first3=True,deep_global_sources=[8,14,20],
        complete_compressed_block_refresh=True,source2_and_deep_index_k_refresh=True,
        mandatory_tail_global_updates=True,actual_event_checks=checked))


def index20_controller_class(natural_class):
    """Create a private subclass; baseline methods retain their original code."""
    from kv_cache_engram.core import digest, token_digest
    from kv_cache_engram.s1_runtime import (S1Controller, gather_hidden, gather_lazy,
        compact_metadata, rebind_sampler_output)
    from kv_cache_engram.s1_system_cache import S1SystemCache
    from kv_cache_engram.s1_system_attention import S1SystemAttention
    from kv_cache_engram.index_hit20 import fingerprint
    from kv_cache_engram.s1_system_batch import (DeepBatch, snapshot_metadata,
        merge_full_metadata, concat_packed, concat_lazy, device_integer_values,
        device_cumulative_pair, deferred_sampler_output)
    from rag_natural_documents import natural_cache_documents
    from index20_cache import index20_cache_class
    from index20_attention import index20_attention_class

    class Index20Controller(natural_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._configured_index20=False; self._index20_fraction=.2
            self._index20_strategy="mean"
            self._index20_diagnostic=False
            self.index20_enabled=False; self.index20_collector=None
            self.index20_flush_agreement=None
            self._index20_frames=[]

        def install(self):
            super().install()
            try:
                require(not self.cache.bank and self.case_binding is None,
                        "Index20 must install before any offline capture")
                self.cache=index20_cache_class(S1SystemCache)(layers=self.layers,scope=self.scope,
                    torch_module=self.t,allow_sync=self.allow_sync)
                self.attention=index20_attention_class(S1SystemAttention)(self.cache,
                    torch_module=self.t,allow_sync=self.allow_sync)
                module=self.layers[2].self_attn.index_attn
                self.index20_collector=NativeIndexScoreCollector(module,torch_module=self.t,
                    allow_sync=self.allow_sync)
                return self
            except BaseException:
                self.close(); raise

        def configure_index20(self,enabled,*,fraction=.2,diagnostic_fingerprints=False,strategy="mean"):
            require(type(enabled) is bool and type(fraction) in (int,float) and 0<=fraction<=1,
                    "Index20 enabled/fraction configuration")
            require(self.state in ("NEW","IDLE","COMPLETE"),"cannot reconfigure active Index20")
            require(type(diagnostic_fingerprints) is bool,"Index20 diagnostic flag")
            require(strategy in ("mean","max","softmax_mean"),"Index20 strategy")
            self._configured_index20=enabled; self._index20_fraction=float(fraction)
            self._index20_diagnostic=diagnostic_fingerprints
            self._index20_strategy=strategy
            return self

        def reset(self,rid,mode,*,arm,input_ids,documents=None,phase,fraction=None,
                  index_device=None,verify_sparse=False,**kwargs):
            if not self._configured_index20:
                self.index20_enabled=False
                self.cache.bind_selection(None)
                return super().reset(rid,mode,arm=arm,input_ids=input_ids,documents=documents,
                    phase=phase,fraction=fraction,index_device=index_device,
                    verify_sparse=verify_sparse,**kwargs)
            require(not kwargs and mode=="sensitive" and arm=="s1" and fraction is None
                    and verify_sparse is False,"Index20 uses its own algorithm qualification")
            require(self.state in ("IDLE","COMPLETE") and self.ctx is None
                    and self.capture_frame is None and not self._index20_frames,
                    "unclean Index20 request")
            require(self.native_model.yoco_boundary_layer_idx==self.original_boundary,
                    "previous model boundary not restored")
            require(self.qualified and index_device is not None,
                    "Index20 requires actual native identity qualification and device")
            require(type(rid) is str and rid and type(phase) is str and phase,
                    "Index20 request identity")
            self.rid,self.mode,self.arm,self.phase=rid,mode,arm,phase
            self.ids=tuple(input_ids); self.input_sha=token_digest(self.ids)
            self.docs=natural_cache_documents(self.ids,documents or ())
            self.expected_document_count=len(self.docs)
            require(self.case_input==self.input_sha and self.case_binding is not None,
                    "Index20 current exact-document bank not prepared")
            require(tuple((key,id(e["tensor"]),e["version"]) for key,e in sorted(self.cache.bank.items()))
                    ==self.case_binding,"Index20 bank identity changed")
            for key in self.cache.bank: self.cache.entry(*key)
            self.verify_sparse=False; self.coverage=0
            self.observed,self.sparse_observed={},{}
            self.decoder_events,self.source2_events,self.engram_events,self.capture_sampler_events=[],[],[],[]
            # D-Spark's existing bridge must collect early target layers now,
            # before index scores can select additional body rows. Its begin
            # contract needs only a guaranteed superset of the final128 window.
            self.prepared=mandatory_dflash_prepared(self.docs,len(self.ids))
            self._index20_frames=[]; self._index20_groups=[]
            self._index20_selection=None; self._index20_agreement=None
            self._index20_installation=None; self._index20_buffer_bytes=0
            self._index20_fingerprints={}; self._index20_source2=[]
            self.index20_enabled=True; self.system_enabled=True
            self.batch_group_chunks=(len(self.ids)+8191)//8192
            self._batch=self._active_deep_batch=None
            self._batch_step_trace,self._source2_before_batch=[],[]
            self.cache.bind_selection(None)
            self.cache.configure(enabled=True); self.attention.configure(enabled=True)
            self.cache.begin_phase(phase,build_global_base=True)
            self.attention.begin_phase()
            if self.dflash_bridge is not None:self.dflash_bridge.reset()
            self.batch_schedule=index20_schedule(self.docs,len(self.ids),request_id=rid,
                phase=phase,phase_generation=self.cache.phase_generation,input_ids_sha256=self.input_sha)
            self.index20_collector.begin(self.docs,len(self.ids),index_device,self._index20_fraction,self._index20_strategy)
            self.state="RUNNING"

        def bind_batch_schedule(self,schedule,receipts):
            if not self.index20_enabled:return super().bind_batch_schedule(schedule,receipts)
            require(self.state=="RUNNING" and self.coverage==0 and self._index20_installation is None
                    and schedule==self.batch_schedule,"Index20 schedule installation order")
            self._index20_installation=validate_installation(schedule,receipts)

        def _observe_layer(self,layer,start,end,result,meta):
            super()._observe_layer(layer,start,end,result,meta)
            if not self.index20_enabled or not self._index20_diagnostic:return
            hidden,lazy=result; group=self._active_deep_batch
            members=[(start,end)] if group is None or layer<3 else group.member_chunks
            offset=0
            for a,z in members:
                if layer<3:h,l=hidden,lazy
                else:
                    n=sum(a<=p<z for p in group.positions_cpu)
                    rows=self.t.arange(offset,offset+n,device=hidden.device,dtype=self.t.long)
                    h,l=gather_hidden(hidden,rows,self.t),gather_lazy(lazy,rows)
                    offset+=n
                key=(a,z,layer)
                require(key not in self._index20_fingerprints,"duplicate diagnostic layer observation")
                with self.allow_sync():
                    self._index20_fingerprints[key]=dict(output=fingerprint(h,self.t),
                        lazy={k:fingerprint(getattr(l,k),self.t) for k in
                            ("residual","post_layer_mix","comb_res_mix","pre_layer_mix")})

        def _retain_first3(self,start,end,meta,hidden,lazy,cross,hook):
            t=self.t
            require(callable(hook) and cross.topk_indices is not None
                    and tuple(cross.topk_indices.shape)==(end-start,512)
                    and cross.block_candidate_idxs is None,"Index20 native source2/Engram contract")
            rows=t.arange(end-start,device=meta.positions.device,dtype=t.long)
            # Fresh gather storage owns every row until future queries decide
            # whether it survives. Drain distributed Engram fetch before reuse.
            owned_hidden=gather_hidden(hidden,rows,t); owned_lazy=gather_lazy(lazy,rows)
            embedding=gather_hidden(hook(),rows,t)
            owned_cross=copy(cross)
            owned_cross.topk_indices=cross.topk_indices.clone()
            owned_cross.candidate_mqa_schedule_metadata={}
            frame=NS(start=start,end=end,meta=snapshot_metadata(meta),hidden=owned_hidden,
                lazy=owned_lazy,embedding=embedding,cross=owned_cross)
            def tensor_bytes(value):
                if value is None:return 0
                if isinstance(value,tuple):return sum(tensor_bytes(x) for x in value)
                return value.numel()*value.element_size()
            self._index20_buffer_bytes+=tensor_bytes(owned_hidden)+tensor_bytes(embedding)+sum(
                tensor_bytes(getattr(owned_lazy,k)) for k in ("residual","post_layer_mix","comb_res_mix","pre_layer_mix"))
            self._index20_frames.append(frame)

        def _make_deep_group(self,frames,selected):
            t=self.t; first,last=frames[0],frames[-1]
            full=merge_full_metadata(frames,t)
            positions=[p for p in selected if first.start<=p<last.end]
            require(len(positions)<=8192,"Index20 selected group exceeds native capacity")
            rows=device_integer_values([p-first.start for p in positions],full.positions,t).long()
            pos=full.positions.index_select(0,rows)
            cu=device_cumulative_pair(full.cu_seqlens_q,len(positions),t)
            compact=compact_metadata(full,rows,cu)
            parts=[]
            for f in frames:
                ps=[p for p in positions if f.start<=p<f.end]
                rs=device_integer_values([p-f.start for p in ps],f.meta.positions,t).long()
                parts.append(NS(hidden=gather_hidden(f.hidden,rs,t),lazy=gather_lazy(f.lazy,rs),
                    embedding=gather_hidden(f.embedding,rs,t),topk=f.cross.topk_indices.index_select(0,rs),count=len(ps)))
            cross=copy(last.cross); cross.topk_indices=t.cat([p.topk for p in parts])
            cross.shared_hidden=None;cross.consumer_full_hidden=None;cross.consumer_full_metadata=None
            cross.candidate_mqa_lens=None;cross.candidate_mqa_schedule_metadata={}
            plan=make_plan(self.docs,len(self.ids),selected,first.start,last.end)
            sampler=make_plan(self.docs,len(self.ids),selected,last.start,last.end)
            lookup={p:i for i,p in enumerate(positions)}
            pairs=[(i,lookup[q]) for i,q in enumerate(sampler["sampler_output_positions"]) if q in lookup]
            prepared=dict(plan=plan,rows=rows,positions=pos,cu_q=cu,
                sample_dst=device_integer_values([a for a,_ in pairs],rows,t),
                sample_src=device_integer_values([b for _,b in pairs],rows,t),
                sampler_output_rows=len(sampler["sampler_output_positions"]))
            embedding=concat_packed([p.embedding for p in parts],t); consumed=[False]
            def hook():
                require(not consumed[0],"Index20 merged Engram14 consumed twice")
                consumed[0]=True; return embedding
            report=seal(dict(schema="index20-deep-group-v1",start=first.start,end=last.end,
                member_chunks=[(f.start,f.end)for f in frames],selected_positions=positions,
                selected_rows=len(positions),first3_native_calls=len(frames),deep_layer_calls=37,
                original_engram14_fetches_consumed=len(frames),no_unselected_deep_rows=True,
                source2_prefix_bound_to_group_end=True,numerical_parity_claim=False))
            return DeepBatch(first.start,last.end,report["member_chunks"],
                concat_packed([p.hidden for p in parts],t),concat_lazy([p.lazy for p in parts],t),
                full,compact,cross,rows,positions,prepared,plan,hook,report,consumed,
                [f.meta for f in frames],[p.count for p in parts])

        def _run_deep_group(self,model,caches,group):
            g=self.native_symbols
            self._active_deep_batch=group
            self.ctx=NS(full_meta=group.full_meta,compact_meta=group.compact_meta,start=group.start,
                end=group.end,positions=group.compact_meta.positions,positions_cpu=group.positions_cpu,
                selected_rows=group.selected_rows,selected_count=len(group.positions_cpu),docs=self.docs,
                input_tokens=len(self.ids),mode=self.mode,phase=self.phase,rid=self.rid,plan=group.plan,
                prepared=group.prepared,verify_sparse=False,verify=False,reference_hidden=None,
                reference_bf16=None,reference_selected_rows=group.selected_rows)
            hidden,lazy,cross=group.hidden,group.lazy,group.cross
            try:
                for i in range(3,40):
                    layer=self.layers[i]
                    if not group.positions_cpu:
                        if i==14:group.engram14_hook()
                        hidden,lazy=self._empty_decoder(layer,caches[i],self.ctx,cross,hidden,lazy)
                        self._observe_layer(i,group.start,group.end,(hidden,lazy),group.compact_meta)
                    else:
                        hidden,lazy=layer(g["Movable"](hidden),caches[i],group.compact_meta,lazy,
                            embed_hook=group.engram14_hook if i==14 else None,
                            cross_layer_runtime=cross,dflash_input=None)
                require(group.engram14_consumption[0],"Index20 deep Engram14 not consumed")
                for (a,z),n in zip(group.member_chunks,group.member_selected_counts):
                    self.engram_events.append(dict(layer=14,chunk_start=a,chunk_end=z,
                        original_hash_rows=z-a,selected_positions=[p for p in group.positions_cpu if a<=p<z],
                        actual_engram_rows=n,hook_consumed_once=True,original_ngram_hashes_preserved=True,
                        index20_first3_embedding_retained=True,system_deferred_gate=True))
                self._index20_groups.append(seal(dict({k:v for k,v in group.report.items() if k!="sha256"},
                    actual_deep_layer_calls=37,engram14_consumed=True,group_execution_completed=True)))
                if group.end!=len(self.ids):return None
                require(group.positions_cpu and group.positions_cpu[-1]==len(self.ids)-1,
                        "Index20 final sampler token must survive")
                if hidden.dim()==2:hidden=g["HCLayerMixin"].hc_post(hidden,lazy)
                require(hidden.ndim==3,"Index20 final mHC trunk")
                hidden=g["mhc_pre_apply_mix"](hidden,lazy.pre_layer_mix) if model.hc_shifted else model.hc_head(hidden)
                hidden=model.output_norm(hidden)
                p=group.prepared
                return rebind_sampler_output(hidden,p["sampler_output_rows"],p["sample_dst"],p["sample_src"])
            finally:
                self.ctx=self._active_deep_batch=None

        def _policy_model_forward(self,model,caches,meta,hidden,dflash):
            if not self.index20_enabled:return super()._policy_model_forward(model,caches,meta,hidden,dflash)
            require(self._index20_installation is not None and callable(self.index20_flush_agreement)
                    and self.ctx is None,"Index20 all-rank execution not installed")
            self._check_dflash(dflash)
            start,end=self._marker(meta.canonical_request_binding)
            require(meta.total_q==end-start and meta.num_reqs==1,"Index20 native chunk geometry")
            g=self.native_symbols
            hidden=model.embed_tokens(meta.input_ids) if hidden is None else hidden
            lazy,cross=None,g["CrossLayerRuntime"]()
            require(model.engram_runtime is not None and meta.ngram_token_ids is not None,
                    "Index20 native Engram history required")
            hashes=g["tile_kernels"].engram.engram_hash(meta.ngram_token_ids,model.ngram_multipliers,
                model.ngram_vocab_sizes,model.ngram_offsets,getattr(meta,"images_seq_mask",None))
            hooks=model.engram_runtime.fetch(hashes)
            for i in range(3):
                hidden,lazy=self.layers[i](g["Movable"](hidden),caches[i],meta,lazy,
                    embed_hook=hooks.get(i),cross_layer_runtime=cross,dflash_input=None)
            self._retain_first3(start,end,meta,hidden,lazy,cross,hooks.get(14))
            if self._index20_diagnostic:
                with self.allow_sync():before=self.attention.source2_fingerprint(caches[2],meta,cross)
                self._index20_source2.append((start,end,before,self._index20_frames[-1].meta))
            if end<len(self.ids):
                self._batch_step_trace.append(dict(start=start,end=end,flush=False,layers=[0,1,2],
                    deep_native_ep_calls=0,index20_deep_groups=0))
                return g["CausalLMForwardOutput"](hidden_states=deferred_sampler_output(hidden,
                    full_input_tokens=len(self.ids),start=start,end=end))
            self._index20_selection=self.index20_collector.finish()
            selected=self._index20_selection["selected_positions"]
            self.cache.bind_selection(selected)
            self.attention.begin_phase()
            groups=partition_native_frames([(f.start,f.end)for f in self._index20_frames],selected)
            require(self.index20_flush_agreement(len(groups))==len(groups),"Index20 final group agreement")
            self._index20_agreement=dict(num_groups=len(groups),all_ranks_agreed=True)
            by_start={f.start:f for f in self._index20_frames}; output=None
            self._index20_frames=[]
            for chunks in groups:
                group=self._make_deep_group([by_start[a]for a,z in chunks],selected)
                output=self._run_deep_group(model,caches,group)
                # Release full retained rows as soon as the corresponding deep
                # group no longer needs them; only online KV remains.
                for a,z in chunks:by_start.pop(a)
            self._index20_frames=[]
            if self._index20_diagnostic:
                for a,z,before,saved in self._index20_source2:
                    with self.allow_sync():after=self.attention.source2_fingerprint(caches[2],saved)
                    require(before==after,"Index20 changed a native source2 prefix")
                    self.source2_events.append(dict(chunk_start=a,chunk_end=z,before=before,after=after,exact=True))
                self._index20_source2=[]
            require(output is not None,"Index20 final output missing")
            self._commit_dflash_tail(meta,dflash,caches,end)
            self._batch_step_trace.append(dict(start=start,end=end,flush=True,
                layers=[0,1,2]+list(range(3,40))*len(groups),deep_native_ep_calls=37*len(groups),
                index20_deep_groups=len(groups)))
            return g["CausalLMForwardOutput"](hidden_states=output)

        def finish(self):
            if not self.index20_enabled:return super().finish()
            require(not self._index20_frames and self._index20_selection is not None
                    and self._index20_agreement is not None,"Index20 final selection/flush incomplete")
            # Original base checks all40 logical events per native chunk and
            # D-Spark/ring/capture report ownership; no old sparse gate is set.
            report=S1Controller.finish(self)
            tail_proof=mandatory_tail_update_proof(self.docs,len(self.ids),self.decoder_events,
                report["cache_report"]["global_events"],report["attention_report"]["attention_events"])
            for document in self.docs:
                for layer in range(3,40):
                    halo=self.cache.entry(document["key"],"swa_halo",layer)
                    require(tuple(halo.shape)==(128,512),"mandatory preceding128 SWA bank missing")
            tail_proof=seal(dict({k:v for k,v in tail_proof.items() if k!="sha256"},
                mandatory_preceding128_offline_swa_bank=True))
            fingerprints=[dict(key=list(k),value=v)for k,v in sorted(self._index20_fingerprints.items())]
            if self._index20_diagnostic:
                require(len(fingerprints)==40*((len(self.ids)+8191)//8192),"Index20 diagnostic coverage")
            report.update(schema="index20-controller-v1",algorithm="index20",
                mathematical_policy_changed=True,system_policy_changes=True,
                original_s12_mathematical_equivalence_claim=False,
                system_enabled=True,batch_group_chunks=self.batch_group_chunks,
                system_numerically_qualified=None,system_original_per_chunk_fingerprints=False,
                system_batch_step_trace=list(self._batch_step_trace),
                system_batch_execution=dict(schedule=self.batch_schedule,groups=self._index20_groups,
                    all_rank_installation=self._index20_installation,final_agreement=self._index20_agreement,
                    numerical_parity_claim=False),
                index20=dict(selection=self._index20_selection,
                    first3_retained_hidden_mhc_engram_bytes=self._index20_buffer_bytes,
                    retained_byte_scope="hidden_four_lazy_mhc_and_engram14_only_excludes_metadata_and_source2_loaded_KV",
                    total_retained_state_bytes=None,dflash_admission_uses_guaranteed_mandatory_subset=True,
                    mandatory_tail_updates=tail_proof,
                    first3_executed_once=True,online_selection_every_phase=True,
                    selected_tokens=len(self._index20_selection["selected_positions"]),
                    actual_deep_groups=len(self._index20_groups),
                    actual_deep_native_ep_calls=37*len(self._index20_groups),
                    diagnostic_fingerprints=fingerprints,
                    diagnostic_fingerprints_sha256=digest(fingerprints) if fingerprints else None,
                    diagnostic_fingerprints_collected=self._index20_diagnostic,
                    diagnostic_source2_byte_identity_checked=self._index20_diagnostic,
                    full_swa_dependency_bank=True,original_native_window=128,
                    independent_candidate_diagnostic_qualification_required=True))
            self.last_report=report
            return report

        def close(self):
            if self.index20_collector is not None:
                self.index20_collector.close();self.index20_collector=None
            self._index20_frames=[]
            return super().close()

    return Index20Controller
