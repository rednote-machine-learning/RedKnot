"""CPU geometry and actual tensor queue/metadata regression for deep batching."""
import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from kv_cache_engram import s1_system_batch as b
from kv_cache_engram.core import digest

try:
    import torch
except ImportError:
    torch=None


def schedule(n=32769,group=2):
    docs=[dict(start=0,end=n,key='doc')]
    s=b.batch_schedule(docs,n,request_id='r',phase='p',phase_generation=1,
        input_ids_sha256='a'*64,group_chunks=group)
    return docs,s


def acknowledgments(s):
    return [dict(schema='s1-system-batch-rank-installation-v1',rank=rank,
        schedule_sha256=s['sha256'],worker_loop_installed=True,native_first3_per_chunk=True,
        deep_only_on_flush=True,empty_deep_ep_on_flush=True) for rank in range(8)]


class BatchScheduleTests(unittest.TestCase):
    def test_batch_and_runtime_have_no_hotpath_host_tensor_constructors(self):
        for name in ('s1_system_batch.py','s1_system_runtime.py'):
            tree=ast.parse((Path(b.__file__).parent/name).read_text())
            forbidden=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute) and node.func.attr in ('new_tensor','tensor','as_tensor')]
            self.assertEqual(forbidden,[],name)

    def test_cuda_gate_expected_sampler_geometry_matches_existing_policy(self):
        docs,_=schedule()
        for start,end,rows,dst in ((8192,16384,1,[]),(24576,32768,139,list(range(12,139))),
                                   (32768,32769,1,[0])):
            p=b.plan(docs,32769,start,end)
            self.assertEqual(len(p['sampler_output_positions']),rows)
            self.assertEqual([i for i,q in enumerate(p['sampler_output_positions']) if q>=32641],dst)

    def test_two_chunk_schedule_keeps_first3_and_final_flush(self):
        _,s=schedule()
        self.assertEqual([x['flush'] for x in s['steps']],[False,True,False,True,True])
        self.assertEqual([x['group_start'] for x in s['steps']],[0,0,16384,16384,32768])
        self.assertEqual(s['steps'][0]['layers'],[0,1,2])
        self.assertEqual(s['steps'][1]['layers'],list(range(40)))
        self.assertEqual(s['steps'][0]['selected_rows'],0)
        self.assertEqual(s['steps'][-1]['selected_rows'],1)
        self.assertFalse(s['steps'][-1]['nonfinal_sample_discarded'])

    def test_four_chunk_schedule_does_not_drop_partial_final_group(self):
        _,s=schedule(40961,4)
        self.assertEqual([i for i,v in enumerate(s['steps']) if v['flush']],[3,5])
        self.assertEqual(s['steps'][-1]['end'],40961)

    def test_no_rank_zero_only_installation(self):
        _,s=schedule();receipts=acknowledgments(s)
        self.assertFalse(b.validate_schedule_agreement(s,receipts)['gpu_execution_parity_claim'])
        for bad in (None,receipts[:1],receipts[:-1],receipts+[receipts[0]]):
            with self.assertRaises(ValueError):b.validate_schedule_agreement(s,bad)

    def test_foreign_rank_schedule_and_skipped_empty_ep_rejected(self):
        _,s=schedule()
        for change in ({'rank':0},{'schedule_sha256':'b'*64},{'empty_deep_ep_on_flush':False},
                       {'deep_only_on_flush':False},{'worker_loop_installed':False}):
            bad=acknowledgments(s);bad[-1].update(change)
            with self.assertRaises(ValueError):b.validate_schedule_agreement(s,bad)
        bad=copy.deepcopy(s);bad['steps'][0]['flush']=True
        with self.assertRaises(ValueError):b.validate_schedule_agreement(bad,acknowledgments(s))

    def test_batch_scope_binds_original_input_and_phase(self):
        docs,s=schedule();changed=b.batch_schedule(docs,32769,request_id='other',phase='p',
            phase_generation=1,input_ids_sha256='a'*64)
        self.assertNotEqual(s['sha256'],changed['sha256'])
        with self.assertRaises(ValueError):b.validate_schedule_agreement(changed,acknowledgments(s))

    def test_invalid_group_and_auxiliary_output_contract_rejected_before_tensors(self):
        for group in (0,1,3,8,True):
            with self.assertRaises(ValueError):schedule(group=group)
        docs,s=schedule()
        for option in ({'prompt_logprobs':True},{'speculative':True}):
            with self.assertRaises(ValueError):
                b.CrossChunkDeepBatch(torch_module=None,documents=docs,full_input_tokens=32769,
                    request_id='r',phase='p',phase_generation=1,input_ids_sha256='a'*64,
                    all_rank_receipts=acknowledgments(s),**option)

    def test_incomplete_queue_cannot_finish(self):
        docs,s=schedule();q=b.CrossChunkDeepBatch(torch_module=None,documents=docs,full_input_tokens=32769,
            request_id='r',phase='p',phase_generation=1,input_ids_sha256='a'*64,
            all_rank_receipts=acknowledgments(s))
        with self.assertRaises(ValueError):q.finish()
        q.discard();self.assertEqual(q.state,'FAILED')


def metadata(start,end,total):
    n=end-start;p=torch.arange(start,end,dtype=torch.int64)
    key=p[:,None]-torch.arange(127,-1,-1,dtype=torch.int64)[None,:]
    swa=torch.where(key>=0,key-start+128,-1).to(torch.int32)
    pc={};slots={};cu={}
    for r in (1,2):
        count=end//r-start//r;k=str(r)
        pc[k]=torch.arange(start//r,end//r,dtype=torch.int32)*r if count else torch.tensor([0],dtype=torch.int32)
        slots[k]=torch.arange(start//r,end//r,dtype=torch.int32) if count else torch.tensor([-1],dtype=torch.int32)
        cu[k]=torch.tensor([0,max(1,count)],dtype=torch.int32)
    return SimpleNamespace(is_prompt=True,num_reqs=1,total_q=n,input_ids=p.clone(),positions=p,
        valid_token_mask=torch.ones(n,dtype=torch.bool),gate_force_random=torch.zeros(n,dtype=torch.bool),
        images_seq_mask=None,ngram_token_ids=p[:,None].repeat(1,4),
        slot_mapping_store_swa=torch.where(p>=end-128,p%128,-1).to(torch.int32),
        query_indices=None,context_lens=None,ks={str(r):torch.zeros(n,dtype=torch.int32) for r in (1,2)},
        ke={str(r):((p+1)//r).to(torch.int32) for r in (1,2)},
        klen={str(r):((p+1)//r).to(torch.int32) for r in (1,2)},kv_ca_index={},kv_ca_index_len={},kv_ca_index_offset_len={},
        positions_c=pc,slot_mapping_store_ca=slots,cu_compressed_seqlens_q=cu,
        cu_seqlens_q=torch.tensor([0,n],dtype=torch.int32),cu_seqlens_k=torch.tensor([0,end],dtype=torch.int32),
        kv_swa_full_index=swa,kv_swa_full_index_pivots=None,compressed_batch_mapping={},
        total_k_swa=128,load_segments_swa=torch.tensor([[0,0,128,0]],dtype=torch.int32),
        total_k_compressed={'1':end,'2':max(1,end//2)},
        load_segments_ca={'1':torch.tensor([[0,0,end,0]],dtype=torch.int32),
                          '2':torch.tensor([[0,0,max(1,end//2),0]],dtype=torch.int32)},
        state_block_idx=torch.tensor([1],dtype=torch.int32),
        canonical_request_binding=dict(request_id='r',phase='p',start=start,end=end))


def queue(n=16385):
    docs,s=schedule(n)
    return b.CrossChunkDeepBatch(torch_module=torch,documents=docs,full_input_tokens=n,request_id='r',phase='p',
        phase_generation=1,input_ids_sha256='a'*64,all_rank_receipts=acknowledgments(s))


def push(q,start,end):
    meta=metadata(start,end,q.schedule['full_input_tokens'])
    selected=b.plan(q.docs,q.schedule['full_input_tokens'],start,end)['selected_positions']
    rows=torch.tensor([p-start for p in selected],dtype=torch.int64)
    h=meta.positions[:,None].to(torch.float32).repeat(1,4)
    lazy=SimpleNamespace(**{k:h[None,:,None,:].clone()+i for i,k in enumerate(b.LAZY)})
    topk=meta.positions[:,None].repeat(1,512).to(torch.int32)
    cross=SimpleNamespace(topk_indices=topk,block_candidate_idxs=None,candidate_mqa_schedule_metadata={},
        loaded_prefill_kv=torch.zeros(128+(end//2),1,2),loaded_prefill_kv_main_offset=128,
        loaded_prefill_index_kv=(torch.zeros(end//2,1),torch.zeros(end//2,1)),pre_rope_kv=None)
    consumed=[]
    def embedding():consumed.append(True);return h+100
    batch=q.push(full_meta=meta,hidden_after_layer2=h,lazy_after_layer2=lazy,cross_after_layer2=cross,
        selected_rows=rows,selected_positions_cpu=selected,engram14_hook=embedding)
    return batch,meta,h,lazy,consumed


@unittest.skipUnless(torch is not None,'Torch unavailable in local CPU environment')
class BatchTensorTests(unittest.TestCase):
    def test_device_integer_run_construction_matches_irregular_and_empty_values(self):
        for dtype in (torch.int32,torch.int64):
            proto=torch.empty(0,dtype=dtype)
            for values in ([],[0],[0,0],[-1,0,1,8,9,10,4],list(range(32768))):
                actual=b.device_integer_values(values,proto,torch)
                self.assertEqual(actual.tolist(),values)
                self.assertEqual(actual.dtype,dtype)
            for n in (0,1,8192,32769):
                self.assertEqual(b.device_cumulative_pair(proto,n,torch).tolist(),[0,n])

    def test_actual_cuda_gate_metadata_fixture_runs_complete_queue_on_cpu(self):
        # Same fixture builder and same queue entry points as the GPU gate;
        # this CPU test makes no strict-CUDA or native numerical parity claim.
        for width in (2,4):
            docs,s=schedule(32769,width)
            q=b.CrossChunkDeepBatch(torch_module=torch,documents=docs,full_input_tokens=32769,
                request_id='metadata-proof',phase='metadata-proof',phase_generation=1,
                input_ids_sha256='a'*64,group_chunks=width,
                all_rank_receipts=acknowledgments(b.batch_schedule(docs,32769,request_id='metadata-proof',
                    phase='metadata-proof',phase_generation=1,input_ids_sha256='a'*64,group_chunks=width)))
            groups=[]
            for step in s['steps']:
                a,z=step['start'],step['end']
                meta,h,lazy,cross=b._metadata_qualification_inputs(torch,torch.device('cpu'),a,z)
                selected=list(range(max(a,32641),z)) if z>32641 else []
                rows=b.device_integer_values([p-a for p in selected],meta.positions,torch)
                batch=q.push(full_meta=meta,hidden_after_layer2=h,lazy_after_layer2=lazy,
                    cross_after_layer2=cross,selected_rows=rows,selected_positions_cpu=selected,
                    engram14_hook=lambda h=h:h+100)
                if batch is not None:
                    self.assertEqual(batch.hidden[:,0].tolist(),batch.positions_cpu)
                    self.assertEqual(batch.prepared['sample_src'].tolist(),list(range(batch.selected_count)))
                    batch.engram14_hook();groups.append(batch)
            q.finish()
            self.assertEqual([v.selected_count for v in groups],[0,127,1] if width==2 else [127,1])

    def test_real_queue_concatenates_hidden_all_lazy_and_layer2_topk(self):
        q=queue();a,meta,h,lazy,used=push(q,0,8192)
        self.assertIsNone(a);self.assertEqual(used,[True])
        batch,*_=push(q,8192,16384)
        self.assertEqual(batch.positions_cpu,list(range(16257,16384)))
        self.assertEqual(batch.hidden[:,0].tolist(),batch.positions_cpu)
        for i,name in enumerate(b.LAZY):
            self.assertEqual(getattr(batch.lazy,name)[0,:,0,0].tolist(),[p+i for p in batch.positions_cpu])
        self.assertEqual(batch.cross.topk_indices[:,0].tolist(),batch.positions_cpu)
        self.assertEqual(batch.engram14_hook()[:,0].tolist(),[p+100 for p in batch.positions_cpu])
        with self.assertRaises(ValueError):batch.engram14_hook()

    def test_meta_native_swa_rebased_and_causal_keys_never_shift(self):
        q=queue();push(q,0,8192);batch,*_=push(q,8192,16384)
        full=batch.full_meta
        self.assertEqual(full.total_q,16384)
        self.assertEqual(full.cu_seqlens_k.tolist(),[0,16384])
        self.assertEqual(full.total_k_compressed,{'1':16384,'2':8192})
        self.assertEqual(full.positions_c['2'].tolist(),list(range(0,16384,2)))
        self.assertEqual(full.kv_swa_full_index[8192].tolist(),list(range(8193,8321)))
        self.assertEqual(batch.compact_meta.ke['2'].tolist(),[(p+1)//2 for p in batch.positions_cpu])
        self.assertEqual(batch.compact_meta.positions.tolist(),batch.positions_cpu)

    def test_final_odd_singleton_retains_native_zero_compression_placeholder(self):
        q=queue();push(q,0,8192);batch,*_=push(q,8192,16384);batch.engram14_hook()
        final,*_=push(q,16384,16385)
        self.assertEqual(final.full_meta.positions_c['2'].tolist(),[0])
        self.assertEqual(final.full_meta.slot_mapping_store_ca['2'].tolist(),[-1])
        self.assertEqual(final.prepared['sampler_output_rows'],1)
        self.assertEqual(final.prepared['sample_src'].tolist(),[0]);final.engram14_hook()
        self.assertEqual(len(q.finish()['groups']),2)

    def test_partial_final_group_removes_only_spurious_compressed_placeholder(self):
        q=queue(8193);push(q,0,8192);batch,*_=push(q,8192,8193)
        self.assertEqual(batch.full_meta.positions_c['2'].shape[0],4096)
        self.assertEqual(batch.full_meta.slot_mapping_store_ca['2'][-1].item(),4095)
        self.assertEqual(batch.prepared['sampler_output_rows'],1)
        self.assertEqual(batch.prepared['sample_src'].tolist(),[127])
        self.assertEqual(batch.plan['sampler_output_positions'][-1],8192)
        batch.engram14_hook();q.finish()

    def test_empty_group_has_no_phantom_queries_and_still_needs_deep_execution(self):
        q=queue(32769);push(q,0,8192);batch,*_=push(q,8192,16384)
        self.assertEqual(batch.selected_count,0);self.assertEqual(batch.hidden.shape,(0,4))
        self.assertEqual(batch.compact_meta.total_q,0)
        self.assertEqual(batch.report['deep_layer_calls'],37)
        self.assertEqual(batch.engram14_hook().shape,(0,4))

    def test_metadata_snapshot_does_not_alias_future_native_step(self):
        q=queue();_,meta,h,lazy,_=push(q,0,8192)
        saved=q.frames[0].meta;meta.positions.fill_(999);meta.ks['2'].fill_(999)
        self.assertEqual(saved.positions[0].item(),0);self.assertEqual(saved.ks['2'][0].item(),0)
        self.assertNotEqual(saved.positions.data_ptr(),meta.positions.data_ptr())

    def test_packed_concat_preserves_scale_column_stride_with_zero_member(self):
        values=[(torch.ones(n,8),torch.ones(n,2,dtype=torch.int32)) for n in (0,3,2)]
        data,sf=b.concat_packed(values,torch)
        self.assertEqual(data.shape,(5,8));self.assertEqual(sf.shape,(5,2));self.assertEqual(sf.stride(),(1,8))
        _,empty=b.concat_packed(values[:1],torch)
        self.assertEqual(empty.shape,(0,2));self.assertEqual(empty.stride(),(1,4))

    def test_nonfinal_sampler_shape_and_final_deferral_refusal(self):
        h=torch.ones(3,4)
        self.assertEqual(b.deferred_sampler_output(h,full_input_tokens=16385,start=0,end=8192).shape,(1,4))
        self.assertEqual(b.deferred_sampler_output(h,full_input_tokens=8193,start=0,end=8192).shape,(139,4))
        with self.assertRaises(ValueError):b.deferred_sampler_output(h,full_input_tokens=8193,start=8192,end=8193)

    def test_flush_without_consuming_deep_engram_cannot_claim_finish(self):
        q=queue(8193);push(q,0,8192);push(q,8192,8193)
        with self.assertRaisesRegex(ValueError,'Engram14'):q.finish()


if __name__=='__main__':unittest.main()
