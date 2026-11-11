import copy
import unittest
from types import SimpleNamespace

from kv_cache_engram import s1_runtime as rt
from kv_cache_engram.core import token_digest
from kv_cache_engram.s1_plan import plan, all_chunks

try:
    import torch
except ImportError:
    torch = None


class S1GeometryTests(unittest.TestCase):
    def test_tail128_and_nondoc_only(self):
        p = plan([dict(start=3,end=600,key="a"),dict(start=607,end=900,key="b")],910,0,910)
        expected = list(range(3))+list(range(472,607))+list(range(772,910))
        self.assertEqual(p["selected_positions"],expected)
        self.assertFalse(p["extra_cross_document_links"])
        self.assertFalse(p["native_yoco140_query_intersection"])

    def test_halo_covers_exact_native_local_keys_for_all_selected_queries(self):
        docs = [dict(start=2,end=500,key="a"),dict(start=511,end=911,key="b")]
        p = plan(docs,950,0,950)
        selected = set(p["selected_positions"])
        halo = set(range(244,372)) | set(range(655,783))
        observed = set()
        for q in selected:
            missing = set(range(max(0,q-127),q+1))-selected
            self.assertLessEqual(missing,halo)
            observed.update(missing)
        self.assertEqual(observed,set(p["required_offline_halo_positions"]))

    def test_large_doc_has_genuine_empty_first_chunk(self):
        ps = all_chunks([dict(start=0,end=16384,key="a")],16384)
        self.assertEqual(ps[0]["selected_positions"],[])
        self.assertEqual(ps[1]["selected_positions"],list(range(16256,16384)))
        self.assertEqual(ps[0]["sampler_source_rows"],[])
        self.assertEqual(ps[1]["sampler_source_rows"][-1],127)

    def test_natural_long_block_crosses8192_without_truncation(self):
        ps = all_chunks([dict(start=4,end=8998,key="a")],9011)
        self.assertEqual(ps[0]["selected_positions"],list(range(4)))
        self.assertIn(8997,ps[1]["selected_positions"])
        self.assertEqual(ps[1]["sampler_output_positions"][-1],9010)

    def test_small_or_overlapping_documents_rejected(self):
        for docs in ([dict(start=0,end=255)], [dict(start=0,end=300),dict(start=299,end=600)]):
            with self.assertRaises(ValueError):
                plan(docs,700,0,700)

    def test_single_document_qualification_validates_real_ids(self):
        ids = list(range(16384))
        d = dict(input_start=0,input_end=16384,token_ids_sha256=token_digest(ids))
        self.assertEqual(rt.s1_documents(ids,[d]),(dict(start=0,end=16384,key=token_digest(ids)),))
        d["token_ids_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            rt.s1_documents(ids,[d])

    def test_mhc_missing_fields_rejected_without_tensor_operations(self):
        with self.assertRaises(ValueError):
            rt.gather_lazy(SimpleNamespace(residual=None),None)


class S1LifecycleTests(unittest.TestCase):
    def test_request_marker_rejects_foreign_request_and_chunk(self):
        owner=object.__new__(rt.S1Controller)
        owner.state="RUNNING";owner.rid="r";owner.phase="p";owner.ids=range(17000)
        self.assertEqual(owner._marker(dict(request_id="r",phase="p",start=0,end=8192)),(0,8192))
        for change in (dict(request_id="foreign"),dict(phase="foreign"),dict(start=1),dict(end=8191)):
            value=dict(request_id="r",phase="p",start=0,end=8192);value.update(change)
            with self.assertRaises(ValueError):owner._marker(value)

    def test_close_restores_only_owned_hooks_and_capture_boundary(self):
        owner=object.__new__(rt.S1Controller)
        obj=SimpleNamespace();hook=object();obj.forward=hook
        owner.native_model=SimpleNamespace(yoco_boundary_layer_idx=None)
        owner.original_boundary=20;owner.handles=[(obj,"forward",hook)];owner.state="FAILED"
        owner.close()
        self.assertEqual(owner.native_model.yoco_boundary_layer_idx,20)
        self.assertNotIn("forward",obj.__dict__)
        self.assertEqual(owner.state,"CLOSED")
        owner.close()

    def test_mutated_hook_is_not_silently_overwritten(self):
        owner=object.__new__(rt.S1Controller)
        obj=SimpleNamespace(forward=object())
        owner.handles=[(obj,"forward",object())];owner.state="IDLE"
        with self.assertRaisesRegex(ValueError,"ownership"):
            owner.close()

    def test_bank_clear_preserves_or_explicitly_resets_qualification(self):
        owner=object.__new__(rt.S1Controller)
        owner.state="COMPLETE";owner.bank_generation=2
        owner.cache=SimpleNamespace(bank={1:1},capture_reports=[1],global_bases={1:1},global_carry={1:1},
            global_events=[1],preparation_s=2.0)
        owner.case_binding=1;owner.case_input=1
        owner.sparse_reference={1:1};owner.sparse_reference_input="a";owner.sparse_qualified=True
        owner.clear_cache()
        self.assertTrue(owner.sparse_qualified)
        self.assertEqual(owner.bank_generation,3)
        self.assertEqual(owner.cache.bank,{})
        owner.clear_cache(reset_sparse_qualification=True)
        self.assertFalse(owner.sparse_qualified)
        self.assertEqual(owner.sparse_reference,{})


@unittest.skipUnless(torch is not None,"Torch unavailable in local CPU environment")
class S1TensorTests(unittest.TestCase):
    def test_installed_capture_wrapper_preserves_full40_then_rebinds_sampler(self):
        # Exercise the actual installed wrapper with a CPU model double. Its
        # full-row layer/head operations finish before the wrapper may slice.
        for total, expected_rows in ((256,[140]),(8192,[140]),(8193,[139,1]),
                                     (8331,[1,139]),(8994,[1,140]),(16384,[1,140])):
            with self.subTest(total=total):
                owner=object.__new__(rt.S1Controller)
                owner.state="RUNNING";owner.mode="capture_independent"
                owner.ids=range(total);owner.rid="r";owner.phase="p";owner.handles=[]
                owner.capture_sampler_events=[]
                trace=[]
                class Model:
                    yoco_boundary_layer_idx=None
                    dflash_input_size=0  # Native no-MTP fixture, matching DeepseekModel.
                    def forward(self, kv_caches, meta, hidden_states=None, dflash=None):
                        self.assert_metadata=meta
                        n=meta.total_q
                        start=meta.canonical_request_binding["start"]
                        h=torch.arange(start,start+n,dtype=torch.float32).reshape(n,1)
                        before=len(owner.capture_sampler_events)
                        for layer in range(40):
                            trace.append((layer,h.shape[0]))
                            h=h+layer
                            # No output rebinding event is allowed during capture.
                            if len(owner.capture_sampler_events)!=before:
                                raise AssertionError("capture rows were rebound before all layers")
                        return SimpleNamespace(hidden_states=h*2)
                model=Model();owner.native_model=model
                owner._install_model_forward(model.forward)
                for chunk, start in enumerate(range(0,total,8192)):
                    end=min(start+8192,total)
                    meta=SimpleNamespace(is_prompt=True,num_reqs=1,total_q=end-start,
                        canonical_request_binding=dict(request_id="r",phase="p",start=start,end=end))
                    output=model.forward(None,meta)
                    self.assertIs(model.assert_metadata,meta)
                    self.assertEqual(trace[-40:],[(layer,end-start) for layer in range(40)])
                    self.assertEqual(output.hidden_states.shape,(expected_rows[chunk],1))
                    # This is the exact sampler index produced by native C++.
                    self.assertEqual(output.hidden_states[expected_rows[chunk]-1,0].item(),2*(end-1+780))
                    event=owner.capture_sampler_events[-1]
                    self.assertEqual(event,dict(chunk_start=start,chunk_end=end,
                        full_output_rows=end-start,native_sampler_rows=expected_rows[chunk],
                        output_local_start=end-start-expected_rows[chunk],output_last_absolute_position=end-1,
                        source="full40_post_head_norm",rebinding_after_capture=True))
                from gpu_pilot.read_s1_results import audit_capture_sampler
                audit_capture_sampler(dict(capture_sampler_events=owner.capture_sampler_events),total)
                owner.close()

    def test_capture_wrapper_rejects_already_compacted_model_output(self):
        owner=object.__new__(rt.S1Controller)
        owner.state="RUNNING";owner.mode="capture_independent";owner.ids=range(256)
        owner.rid="r";owner.phase="p";owner.handles=[];owner.capture_sampler_events=[]
        class Model:
            yoco_boundary_layer_idx=None
            dflash_input_size=0  # Native no-MTP fixture, matching DeepseekModel.
            def forward(self,*args):
                return SimpleNamespace(hidden_states=torch.zeros(140,1))
        model=Model();owner.native_model=model;owner._install_model_forward(model.forward)
        meta=SimpleNamespace(is_prompt=True,num_reqs=1,total_q=256,
            canonical_request_binding=dict(request_id="r",phase="p",start=0,end=256))
        with self.assertRaisesRegex(ValueError,"every original chunk row"):
            model.forward(None,meta)
        self.assertEqual(owner.state,"FAILED")
        self.assertEqual(owner.capture_sampler_events,[])
        owner.close()

    def test_capture_wrapper_decode_passthrough_does_not_rebind(self):
        owner=object.__new__(rt.S1Controller)
        owner.mode="capture_independent";owner.handles=[];owner.capture_sampler_events=[]
        result=SimpleNamespace(hidden_states=torch.ones(1,1))
        class Model:
            def forward(self,*args): return result
        model=Model();owner.native_model=model;owner._install_model_forward(model.forward)
        self.assertIs(model.forward(None,SimpleNamespace(is_prompt=False,canonical_request_binding={})),result)
        self.assertEqual(owner.capture_sampler_events,[])
        owner.state="IDLE";owner.close()

    def test_layer3_refresh_reads_lazy_first3_fields_without_gathering_key_axis(self):
        rows=torch.tensor([1,4],dtype=torch.long)
        full=SimpleNamespace(is_prompt=True,num_reqs=1,total_q=6,positions=torch.arange(6),
            klen=None,compressed_batch_mapping=None,total_k_compressed=None)
        cu=torch.tensor([0,2],dtype=torch.int32)
        early=rt.compact_metadata(full,rows,cu)
        self.assertIsNone(early.klen)
        # Simulate the native first3 compute_attn_info assignment after the
        # early context was made, not a mutation of an already-shared dict.
        full.klen={"2":torch.arange(6,dtype=torch.int32)}
        full.compressed_batch_mapping={"2":torch.zeros(3,dtype=torch.int32)}
        full.total_k_compressed={"2":3}
        ctx=SimpleNamespace(full_meta=full,selected_rows=rows,compact_meta=early,
            prepared=dict(cu_q=cu,positions=full.positions[rows]))
        active=rt.refresh_compact_metadata(ctx,torch)
        self.assertEqual(active.klen["2"].tolist(),[1,4])
        self.assertIs(active.compressed_batch_mapping,full.compressed_batch_mapping)
        self.assertIs(active.total_k_compressed,full.total_k_compressed)
        self.assertIsNot(active,early)

    def test_all_four_lazy_fields_gather_axis1_and_preserve_none(self):
        rows = torch.tensor([1,4],dtype=torch.long)
        values = {k:torch.arange(1*5*4*2).reshape(1,5,4,2).clone() for k in rt.LAZY}
        lazy = SimpleNamespace(**values)
        got = rt.gather_lazy(lazy,rows)
        for key in rt.LAZY:
            self.assertTrue(torch.equal(getattr(got,key),values[key][:,[1,4]]))
            self.assertNotEqual(getattr(got,key).data_ptr(),values[key].data_ptr())
        lazy.pre_layer_mix=None
        self.assertIsNone(rt.gather_lazy(lazy,rows).pre_layer_mix)

    def test_compact_metadata_does_not_gather_key_axes_or_mutate_source(self):
        rows = torch.tensor([0,4,7],dtype=torch.long)
        full = SimpleNamespace(is_prompt=True,num_reqs=1,total_q=8,positions=torch.arange(30,38),
            gate_force_random=torch.zeros(8,dtype=torch.bool),ks={"2":torch.arange(8)},ke={"2":torch.arange(8)+1},
            positions_c={"2":torch.arange(4)},slot_mapping_store_ca={"2":torch.arange(4)},
            cu_seqlens_k=torch.tensor([0,38]))
        active=rt.compact_metadata(full,rows,torch.tensor([0,3]))
        self.assertEqual(active.positions.tolist(),[30,34,37])
        self.assertIs(active.positions_c,full.positions_c)
        self.assertIs(active.slot_mapping_store_ca,full.slot_mapping_store_ca)
        self.assertIs(active.cu_seqlens_k,full.cu_seqlens_k)
        self.assertEqual(active.ke["2"].tolist(),[1,5,8])
        self.assertEqual(full.total_q,8)
        self.assertIsNone(active.flash_mla_sched_meta_dict)

    def test_cross_runtime_gathers_only_query_arrays(self):
        rows=torch.tensor([1,4])
        keys=torch.arange(24).reshape(6,4)
        cross=SimpleNamespace(topk_indices=torch.arange(30).reshape(5,6),block_candidate_idxs=torch.arange(10).reshape(5,2),
            pre_rope_kv=keys,loaded_prefill_kv=keys,loaded_prefill_index_kv=(keys,keys),
            candidate_mqa_lens=torch.arange(5),candidate_mqa_schedule_metadata={"old":1})
        rt.compact_cross_queries(cross,rows)
        self.assertEqual(cross.topk_indices.shape,(2,6))
        self.assertIs(cross.pre_rope_kv,keys)
        self.assertIs(cross.loaded_prefill_kv,keys)
        self.assertEqual(cross.candidate_mqa_schedule_metadata,{})
        self.assertIsNone(cross.candidate_mqa_lens)

    def test_empty_packed_rows_preserve_sf_stride_without_dummy(self):
        hidden=(torch.zeros((5,128),dtype=torch.float8_e4m3fn),torch.zeros((5,1),dtype=torch.int32))
        actual=rt.gather_hidden(hidden,torch.empty(0,dtype=torch.long),torch)
        self.assertEqual(actual[0].shape,(0,128))
        self.assertEqual(actual[1].shape,(0,1))
        self.assertEqual(actual[1].stride(),(1,4))

    def test_engram_full_hash_embedding_gather_and_hook_once(self):
        source=torch.arange(8*128,dtype=torch.float32).reshape(8,128)
        rows=torch.tensor([2,7])
        calls=[]
        def original():
            calls.append(True)
            return source
        consumed=[False]
        hook=rt.selected_embedding_hook(original,rows,torch,consumed)
        self.assertTrue(torch.equal(hook(),source[[2,7]]))
        self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError): hook()
        empty=rt.selected_embedding_hook(original,torch.empty(0,dtype=torch.long),torch,[False])
        self.assertEqual(empty().shape,(0,128))
        self.assertEqual(len(calls),2)

    def test_sampler_rebind_preserves_last_selected_row_and_empty_intermediate(self):
        hidden=torch.arange(128*8).reshape(128,8)
        result=rt.rebind_sampler_output(hidden,140,torch.arange(12,140),torch.arange(128))
        self.assertTrue(torch.equal(result[-1],hidden[-1]))
        self.assertEqual(torch.count_nonzero(result[:12]).item(),0)
        empty=rt.rebind_sampler_output(hidden[:0],1,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))
        self.assertEqual(empty.shape,(1,8))
        self.assertEqual(torch.count_nonzero(empty).item(),0)

    def test_runtime_gpu_qualification_rejects_cpu_claim(self):
        layer=SimpleNamespace(self_attn=SimpleNamespace(q_a_kv_swa_proj=SimpleNamespace(weight=torch.zeros(1))))
        with self.assertRaisesRegex(ValueError,"actual CUDA"):
            rt.run_s1_runtime_qualification([layer],0,torch,rt.nullcontext,
                execution_sha256="a"*64,model_sha256="b"*64)


@unittest.skipUnless(torch is not None,"Torch unavailable in local CPU environment")
class S1ModelLoopTests(unittest.TestCase):
    def build(self,total,start,end,mode="sensitive"):
        owner=object.__new__(rt.S1Controller)
        owner.t=torch;owner.mode=mode;owner.state="RUNNING";owner.verify_sparse=False
        owner.ids=tuple(range(total));owner.rid="r";owner.phase="p";owner.ctx=None
        owner.docs=[dict(start=0,end=total,key="a")]
        owner.decoder_events=[];owner.engram_events=[];owner.source2_events=[];owner.coverage=start
        owner.allow_sync=rt.nullcontext;owner.empty_ep_capability=SimpleNamespace(permits=lambda *args:True)
        width=128
        owner.attention=SimpleNamespace(forward=lambda mod,packed,bf16,*args:bf16.clone())
        calls=[];fetches=[]
        class Movable:
            def __init__(self,value): self.value=value
            def move(self): return self.value
        class Layer:
            def __init__(self,index):
                self.layer_idx=index;self.engram=object() if index in (1,14) else None
                self.self_attn=SimpleNamespace(layer_idx=index)
                self.mlp=SimpleNamespace(hidden_size=width)
                def mlp(packed,bf16,metadata):
                    calls.append((index,"empty_mlp",bf16.shape[0]))
                    return bf16.clone()
                class MLP:
                    hidden_size=width
                    def __call__(self,*args): return mlp(*args)
                self.mlp=MLP()
            def __call__(self,hidden,cache,metadata,lazy,*,embed_hook=None,cross_layer_runtime=None,dflash_input=None):
                h=rt.unwrap(hidden)
                calls.append((self.layer_idx,"decoder",h.shape[0]))
                if self.layer_idx==0: owner.coverage=end
                if self.engram is not None:
                    embedding=embed_hook()
                    self.assert_embedding_shape=h.shape[0]==embedding.shape[0]
                    if not self.assert_embedding_shape: raise AssertionError("Engram shape mismatch")
                if self.layer_idx==2:
                    cross_layer_runtime.topk_indices=torch.zeros((h.shape[0],512),dtype=torch.int32)
                out=h+1
                lazy=SimpleNamespace(**{key:torch.zeros((1,len(out),4,1)) for key in rt.LAZY})
                owner._observe_layer(self.layer_idx,start,end,(out,lazy),metadata)
                return out,lazy
        owner.layers=[Layer(i) for i in range(40)]
        p=plan(owner.docs,total,start,end)
        ids=lambda values,dtype=torch.long:torch.tensor(values,dtype=dtype)
        owner.prepared={(start,end):dict(plan=p,rows=ids(p["selected_local_rows"]),positions=ids(p["selected_positions"]),
            cu_q=ids([0,p["selected_rows"]],torch.int32),sample_positions=ids(p["sampler_output_positions"]),
            sample_dst=ids(p["sampler_destination_rows"]),sample_src=ids(p["sampler_source_rows"]))}
        def fetch(hashes):
            def one(layer):
                def hook():
                    fetches.append(layer)
                    return torch.zeros((end-start,width))
                return hook
            return {1:one(1),14:one(14)}
        class Norm:
            weight=torch.ones(width)
            def __call__(self,x): return x
        model=SimpleNamespace(dflash_input_size=0,embed_tokens=lambda x:torch.zeros((len(x),width)),
            engram_runtime=SimpleNamespace(fetch=fetch),ngram_multipliers=None,ngram_vocab_sizes=None,ngram_offsets=None,
            hc_shifted=True,output_norm=Norm())
        owner.native_model=model
        owner.dflash_bridge=None
        owner.native_symbols=dict(CrossLayerRuntime=lambda:SimpleNamespace(topk_indices=None),Movable=Movable,
            CausalLMForwardOutput=lambda **kwargs:SimpleNamespace(**kwargs),
            tile_kernels=SimpleNamespace(engram=SimpleNamespace(engram_hash=lambda *args:None)),
            HCLayerMixin=SimpleNamespace(hc_post=lambda h,lazy:h[:,None,:].expand(-1,4,-1)),
            mhc_pre_apply_mix=lambda h,m:h.mean(1))
        meta=SimpleNamespace(is_prompt=True,num_reqs=1,total_q=end-start,positions=torch.arange(start,end),
            input_ids=torch.arange(start,end),ngram_token_ids=torch.zeros((end-start,4),dtype=torch.int32),
            canonical_request_binding=dict(request_id="r",phase="p",start=start,end=end),
            yoco_compaction_enabled=True,yoco_consumer_total_q=len(p["sampler_output_positions"]),
            yoco_consumer_positions=ids(p["sampler_output_positions"]))
        return owner,model,meta,calls,fetches,p

    def test_nonempty_actual_loop_first3_full_then37_compact_with_engram14(self):
        owner,model,meta,calls,fetches,p=self.build(768,0,768)
        out=owner._policy_model_forward(model,[None]*40,meta,None,None)
        self.assertEqual([n for _,kind,n in calls if kind=="decoder"], [768]*3+[128]*37)
        self.assertEqual(fetches,[1,14])
        self.assertEqual(out.hidden_states.shape,(140,128))
        self.assertEqual(len(owner.decoder_events),40)
        self.assertTrue(owner.engram_events[0]["hook_consumed_once"])
        self.assertEqual(owner.engram_events[0]["actual_engram_rows"],128)

    def test_empty_actual_loop_keeps37_native_mlp_calls_drains_engram_no_fake_hidden(self):
        owner,model,meta,calls,fetches,p=self.build(16384,0,8192)
        out=owner._policy_model_forward(model,[None]*40,meta,None,None)
        self.assertEqual([n for _,kind,n in calls if kind=="decoder"],[8192]*3)
        self.assertEqual([n for _,kind,n in calls if kind=="empty_mlp"],[0]*37)
        self.assertEqual(fetches,[1,14])
        self.assertEqual(out.hidden_states.shape,(1,128))
        self.assertEqual(torch.count_nonzero(out.hidden_states).item(),0)
        self.assertTrue(owner.engram_events[0]["empty_hook_consumed_without_fake_rows"])
        self.assertTrue(all(e["actual_mhc_rows"]==0 for e in owner.decoder_events[3:]))


if __name__=="__main__":
    unittest.main()
