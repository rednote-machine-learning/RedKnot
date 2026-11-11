"""CPU contract and mocked native-call tests; not GPU numerical qualification."""
import importlib.util
import subprocess
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch

from kv_cache_engram.core import ValidationError
from kv_cache_engram.tail_link import TailLinkController, tail_scope, padded_memory_width, make_memory_indices
from kv_cache_engram.tail_link_staging import TailStaging


def controller():
    layers = [NS(layer_idx=i) for i in range(40)]
    return TailLinkController(layers, model_sha256='a'*64, execution_sha256='b'*64,
                              tokenizer_sha256='c'*64, native_model=NS(layers=layers,yoco_boundary_layer_idx=20))


class InterfaceTests(unittest.TestCase):
    def test_import_never_loads_torch_or_native(self):
        result = subprocess.run([sys.executable,'-I','-B','-c',
            'import sys;sys.path.insert(0,'+repr(str(Path(__file__).resolve().parents[1]))+');'
            'import kv_cache_engram.tail_link;assert "torch" not in sys.modules;'
            'assert not any(k.startswith("deepinfer") for k in sys.modules)'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_scope_contains_original_tail_position_not_compact_zero(self):
        scope=tail_scope({'execution_sha256':'a'*64},'b'*64,39)
        self.assertEqual(scope['source_local_start'],32640)
        self.assertEqual(scope['source_document_tokens'],32768)
        self.assertEqual(scope['tail_tokens'],128)
        self.assertEqual(scope['representation'],'swa-tail128-postnorm-prerope-bf16')

    def test_memory_padding_kernel_alignment(self):
        self.assertEqual([padded_memory_width(n) for n in (0,1,127,128,129,1024,1075)],
                         [128,128,128,128,256,1024,1152])
        for n in (-1,1.0,True):
            with self.assertRaises(ValidationError): padded_memory_width(n)

    def test_no_percentage_silently_accepted(self):
        c=controller()
        with self.assertRaises(ValidationError):
            c.reset('r','sensitive',input_ids=[1],phase='p',fraction=.2)

    def test_cleanup_clears_staging_and_restored_on_reset(self):
        c=controller();calls=[]
        c.staging=NS(close=lambda:calls.append('closed'))
        c.restored={3:object()};c.nondoc={3:object()};c.ring={3:object()}
        c._clear_phase_gpu()
        self.assertEqual(calls,['closed'])
        self.assertIsNone(c.staging)
        self.assertEqual((c.restored,c.nondoc,c.ring),({},{},{}))

    def test_cannot_clear_running_bank(self):
        c=controller();c.state='RUNNING'
        with self.assertRaises(ValidationError):c.clear_cache()

    def test_invalid_active_reset_does_not_clear_live_buffers(self):
        c=controller();c.state='RUNNING';original=object();c.restored={3:original}
        with self.assertRaises(ValidationError):c.reset('r','sensitive',input_ids=[1],phase='p')
        self.assertIs(c.restored[3],original)

    def test_cleanup_uses_explicit_sync_scope(self):
        from contextlib import contextmanager
        c=controller();entered=[]
        @contextmanager
        def allowed():
            entered.append(True)
            try:yield
            finally:entered.pop()
        c.allow_sync=allowed;c.staging=NS(close=lambda:self.assertEqual(entered,[True]))
        c._clear_phase_gpu()

    def test_host_index_upload_has_bounded_explicit_sync_scope(self):
        from contextlib import contextmanager
        c=controller();entered=[]
        @contextmanager
        def allowed():
            entered.append(True)
            try:yield
            finally:entered.pop()
        def tensor(values,device,dtype):
            self.assertEqual(entered,[True]);return (values,device,dtype)
        c.t=NS(tensor=tensor,long='long');c.allow_sync=allowed
        self.assertEqual(c._upload_indices([1,7],'cuda:0'),([1,7],'cuda:0','long'))
        self.assertEqual(entered,[])

    def test_native_control_delegates_unchanged(self):
        from kv_cache_engram.index_hit20 import IndexHit20Controller
        c=controller();c.mode='all_selected'
        args=[object() for _ in range(6)]
        with patch.object(IndexHit20Controller,'_selected_attention',return_value='native') as call:
            self.assertEqual(c._selected_attention(*args),'native')
            call.assert_called_once_with(*args)


class FakeEvent:
    def __init__(self,log,enable_timing=False):self.log=log;self.ready=True
    def record(self,stream):self.log.append(('record',id(self),stream.name))
    def query(self):return self.ready
    def elapsed_time(self,other):return .25
    def synchronize(self):self.log.append(('sync_event',id(self)))


class FakeStream:
    def __init__(self,log,name):self.log,self.name=log,name
    def wait_event(self,event):self.log.append(('wait',self.name,id(event)))
    def wait_stream(self,stream):self.log.append(('wait_stream',self.name,stream.name))
    def synchronize(self):self.log.append(('sync_stream',self.name))


class FakeTensor:
    ndim=2;shape=(1024,512);dtype='bf16'
    def __init__(self,log,device='cpu'):self.log=log;self.device=NS(type=device)
    def is_pinned(self):return True
    def numel(self):return 1024*512
    def element_size(self):return 2
    def copy_(self,source,non_blocking=False):self.log.append(('copy',id(self),non_blocking));return self


def fake_torch():
    log=[];main=FakeStream(log,'main');prefetch=FakeStream(log,'prefetch')
    cuda=NS(Stream=lambda device:prefetch,Event=lambda enable_timing=False:FakeEvent(log,enable_timing),
            current_stream=lambda device:main,stream=lambda stream:nullcontext())
    t=NS(cuda=cuda,bfloat16='bf16',empty=lambda shape,dtype,device:FakeTensor(log,'cuda'))
    return t,log,{layer:FakeTensor(log) for layer in range(3,40)}


class StagingTests(unittest.TestCase):
    def test_exactly_two_initial_prefetches(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=9)
        self.assertEqual(set(s.pending),{3,4});self.assertEqual(len(s.buffers),2)
        self.assertEqual(sum(x[0]=='copy' for x in log),2)
        first_copy=next(i for i,event in enumerate(log) if event[0]=='copy')
        self.assertLess(log.index(('wait_stream','prefetch','main')),first_copy)
        s.close()

    def test_all_layers_have_one_measured_copy_with_phase_scope(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=9)
        for layer in range(3,40):s.acquire(layer);s.release(layer)
        report=s.report()
        self.assertEqual(report['copy_count'],37)
        self.assertEqual(report['copy_bytes'],37*1024*1024)
        self.assertTrue(all(e['generation']==9 for e in report['copies']))
        self.assertFalse(report['cross_phase_gpu_reuse'])
        self.assertEqual([e['slot'] for e in report['copies']],[i%2 for i in range(37)])
        s.close()

    def test_overwrite_waits_for_consumer_release_event(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=1)
        s.acquire(3);s.release(3)
        event=s.release_events[0]
        wait=log.index(('wait','prefetch',id(event)))
        final_copy=max(i for i,row in enumerate(log) if row[0]=='copy')
        self.assertLess(wait,final_copy)
        self.assertIn(('record',id(event),'main'),log)
        s.close()

    def test_read_requires_copy_event_wait(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=1)
        event=s.pending[3][1];s.acquire(3)
        self.assertIn(('wait','main',id(event)),log)
        s.close()

    def test_duplicate_acquire_release_and_missing_layer_fail(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=1)
        with self.assertRaises(ValidationError):s.acquire(5)
        with self.assertRaises(ValidationError):s.release(3)
        s.acquire(3)
        with self.assertRaises(ValidationError):s.acquire(3)
        s.release(3)
        with self.assertRaises(ValidationError):s.release(3)
        s.close()

    def test_missing_or_unfinished_copy_fails_report(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=1)
        with self.assertRaises(ValidationError):s.report()
        for layer in range(3,40):s.acquire(layer);s.release(layer)
        s.records[0]['end'].ready=False
        with self.assertRaises(ValidationError):s.report()
        s.close()

    def test_close_waits_before_dropping_pinned_banks(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=1)
        s.acquire(3);s.release(3);s.close()
        self.assertIn(('sync_stream','prefetch'),log)
        self.assertTrue(any(x[0]=='sync_event' for x in log))
        self.assertEqual(s.banks,{})
        with self.assertRaises(ValidationError):s.acquire(4)

    def test_exception_before_release_waits_owner_stream(self):
        t,log,banks=fake_torch();s=TailStaging(t,banks,device='cuda:0',generation=1)
        s.acquire(3);s.close()
        self.assertIn(('sync_stream','main'),log)

    def test_wrong_layer_and_nonpinned_banks_rejected(self):
        t,log,banks=fake_torch();banks.pop(39)
        with self.assertRaises(ValidationError):TailStaging(t,banks,device='cuda:0',generation=1)

    def test_constructor_failure_completes_first_copy_before_releasing_bank(self):
        t,log,banks=fake_torch();original=TailStaging._schedule
        def fail_second(stage,layer):
            if layer == 4:raise RuntimeError('injected second-copy setup failure')
            return original(stage,layer)
        with patch.object(TailStaging,'_schedule',fail_second),self.assertRaises(RuntimeError):
            TailStaging(t,banks,device='cuda:0',generation=1)
        self.assertIn(('sync_stream','prefetch'),log)
        t,log,banks=fake_torch();banks[3].is_pinned=lambda:False
        with self.assertRaises(ValidationError):TailStaging(t,banks,device='cuda:0',generation=1)


TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH,'requires CPU torch runtime; must run unskipped on qualification server')
class TorchForwardTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch

    def fixture(self,mode='sensitive',source=False,positions=None):
        t=self.t;c=controller();c.t=t;c.state='RUNNING';c.mode=mode;c.verify_sparse=False
        c.phase='p';c.rid='r';c.ids=(1,)*32771;c.docs=({'start':1,'end':32769,'key':'a'*64},)
        c.layouts={};c.plans={};c.nondoc={};c.ring={};c.swa_events=[];c.attention_events=[]
        c.tail_memory_events=[];c.memory_assembly_events=[];c.decode_ring_events=[]
        c.cpu_plan_oracles=set()
        c.tail_positions_cpu=list(range(32641,32769));calls=[]
        positions=[32638,32639,32641,32768,32769] if positions is None else positions
        p=t.tensor(positions,dtype=t.int32);n=len(positions)
        meta=NS(total_q=n,num_reqs=1,positions=p,is_prompt=True,
                canonical_request_binding={'request_id':'r','phase':'p','start':24576,'end':32770},
                slot_mapping_store_swa=t.arange(n,dtype=t.int32),total_k_compressed={'2':16385},
                ks={'2':t.zeros(n,dtype=t.int32)},
                load_segments_ca={'2':None},compute_attn_info=lambda *args:None)
        mod=NS(layer_idx=8 if source else 3,use_swa=True,has_packed_kv=True,is_sparse=True,
               max_model_len=270336,window_size=128,kv_compress_ratio=2,ratio_key='2',
               shared_hidden_kv_proj=None,has_local_index=source,is_kv_source=source,use_shared_kv=True,
               yoco_swa_share_hidden=False,num_heads=1,qk_head_dim=512,q_a_norm=lambda x:x,
               _q_b_norm_rope=lambda x,p:x,_resolve_shared_hidden=lambda x,s:s)
        def global_project(x):calls.append(('global_project',len(x)));return (None,None)
        def global_index(*args):calls.append(('global_index',meta.total_q));return t.zeros((n,512),dtype=t.int32)
        mod._project_current_hidden_inputs=global_project;mod._compute_kv_ca_index=global_index
        def qa(x):calls.append(('qa',len(x)));return x[:,:4]
        def swa(x):calls.append(('swa',len(x)));return x[:,:512].clone()
        def qb(x):calls.append(('qb',len(x)));return x[:,:1].repeat(1,512)
        c.projections[mod.layer_idx]=(qa,swa);c.q_b_projections[mod.layer_idx]=(qb,lambda x:x)
        c.native_prepares[mod.layer_idx]=lambda raw,cache,md,cp:(raw.reshape(-1,1,512),None)
        tails=t.arange(128,dtype=t.bfloat16)[:,None,None].repeat(1,1,512)
        c._restored_tails=lambda mod,meta:tails
        c.ops=NS(rows=lambda x:len(x),identity_error=lambda a,b:self.assertTrue(t.equal(a,b)))
        def joint(q,kv,g,s,cached,current,metadata):
            calls.append(('joint',len(q),current,s.clone(),kv.clone()));return q.clone()
        mod._prefill_attention=joint;mod._wv_and_o_proj=lambda q,p:q[:,0,:].repeat(1,10)
        cache=NS(ca_cache=t.zeros(1,1,512),swa_cache=t.zeros(1,32,1,512))
        cross=NS(shared_hidden=None,consumer_full_metadata=None,
                 loaded_prefill_kv=t.full((16485,1,512),77,dtype=t.bfloat16),loaded_prefill_kv_main_offset=100)
        hidden=(p.to(t.float32)%100+1).to(t.bfloat16)[:,None].repeat(1,5120)
        load=ModuleType('load');load.load_paged_kv_cache=lambda cache,segments,rows,out:out.fill_(77)
        store=ModuleType('store');store.store_kv_cache=lambda value,cache,slots:calls.append(('store',value.clone(),slots.clone()))
        modules={'deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache':load,
                 'deepinfer.model_executor.layers.tilelang_ops.store_kv_cache':store}
        return c,mod,meta,cache,cross,hidden,calls,modules

    def run_fixture(self,**kwargs):
        c,mod,meta,cache,cross,h,calls,modules=self.fixture(**kwargs)
        with patch.dict(sys.modules,modules):out=c._selected_attention(mod,h,cache,meta,h,cross)
        return c,out,calls,cross

    def test_actual_forward_only_tail_and_nondoc_mainqueries(self):
        c,out,calls,cross=self.run_fixture()
        self.assertIn(('qa',3),calls);self.assertIn(('qb',3),calls)
        self.assertIn(('swa',1),calls)
        self.assertIn(('global_index',5),calls);self.assertIn(('global_project',5),calls)
        self.assertTrue(self.t.equal(out[:2],self.t.zeros_like(out[:2])))
        self.assertEqual(c.tail_memory_events[0]['selected_positions'],[32641,32768,32769])
        self.assertEqual(c.swa_events[0]['document_online_rows'],0)
        self.assertEqual(cross.loaded_prefill_kv_main_offset,129)

    def test_source_index_q_stays_full_but_main_q_is_selected(self):
        c,out,calls,_=self.run_fixture(source=True)
        self.assertIn(('qa',5),calls);self.assertIn(('qb',3),calls)
        self.assertEqual(c.attention_events[0]['index_q_rows'],5)

    def test_dense_samepolicy_reference_masks_after_all_query_work(self):
        sparse,out,calls,_=self.run_fixture()
        ref,reference,rcalls,_=self.run_fixture(mode='reference_sparse')
        self.assertTrue(self.t.equal(out,reference))
        self.assertIn(('qb',5),rcalls)
        self.assertEqual(ref.tail_memory_events[0]['actual_joint_query_rows'],5)
        self.assertEqual(len(ref.memory_assembly_events),1)

    def test_all_unselected_rows_no_main_q_but_global_runs(self):
        c,out,calls,_=self.run_fixture(positions=[26000,26001])
        self.assertIn(('global_index',2),calls)
        self.assertFalse(any(row[0] in ('qa','qb','joint','swa') for row in calls))
        self.assertTrue(self.t.equal(out,self.t.zeros_like(out)))

    def test_yoco_single_nonfinal_placeholder_is_not_forced_selected(self):
        c,out,calls,_=self.run_fixture(positions=[26000])
        self.assertEqual(c.tail_memory_events[0]['selected_rows'],0)
        self.assertIn(('global_index',1),calls)
        store=next(row for row in calls if row[0]=='store')
        self.assertEqual(store[1].shape[0],0)
        self.assertEqual(store[2].numel(),0)
        self.assertTrue(self.t.equal(out,self.t.zeros_like(out)))

    def test_ring_writes_only_real_selected_kv(self):
        c,out,calls,_=self.run_fixture()
        store=next(row for row in calls if row[0]=='store')
        self.assertEqual(store[2].tolist(),[2,3,4])
        self.assertTrue(self.t.equal(store[1][0],self.t.zeros_like(store[1][0])))
        self.assertEqual(float(store[1][1,0,0]),127)

    def test_foreign_global_index_future_rejected(self):
        c,mod,meta,cache,cross,h,calls,modules=self.fixture()
        mod._compute_kv_ca_index=lambda *args:self.t.full((5,512),20000,dtype=self.t.int32)
        with patch.dict(sys.modules,modules),self.assertRaises(RuntimeError):
            c._selected_attention(mod,h,cache,meta,h,cross)

    def test_nonzero_request_offset_rejected(self):
        c,mod,meta,cache,cross,h,calls,modules=self.fixture()
        meta.ks['2'].fill_(1)
        with patch.dict(sys.modules,modules),self.assertRaises(RuntimeError):
            c._selected_attention(mod,h,cache,meta,h,cross)

    def test_global_index_outside_materialized_bank_rejected(self):
        c,mod,meta,cache,cross,h,calls,modules=self.fixture()
        meta.total_k_compressed['2']=1
        # Group1 is causally valid for these positions, but not in the bank.
        mod._compute_kv_ca_index=lambda *args:self.t.ones((5,512),dtype=self.t.int32)
        with patch.dict(sys.modules,modules),self.assertRaises(RuntimeError):
            c._selected_attention(mod,h,cache,meta,h,cross)

    def test_gpu_causal_indices_match_absolute_positions_not_slots(self):
        t=self.t;mem=t.tensor([1,10,20],dtype=t.int32);q=t.tensor([0,10,30],dtype=t.int32)
        out=make_memory_indices(t,mem,q)
        self.assertEqual(tuple(out.shape),(3,128))
        self.assertEqual(out[0].tolist(),[-1]*128)
        self.assertEqual(out[1,:4].tolist(),[0,1,-1,-1])
        self.assertEqual(out[2,:4].tolist(),[0,1,2,-1])

    def test_empty_memory_and_empty_query_index_geometry(self):
        t=self.t;out=make_memory_indices(t,t.tensor([],dtype=t.int32),t.tensor([1],dtype=t.int32))
        self.assertEqual(out.tolist(),[[-1]*128])
        out=make_memory_indices(t,t.tensor([1],dtype=t.int32),t.tensor([],dtype=t.int32))
        self.assertEqual(tuple(out.shape),(0,128))

    def test_changed_native_positions_same_length_rejected(self):
        c,mod,meta,*_=self.fixture();m=meta.canonical_request_binding
        c._current_positions(meta,m);meta.positions=meta.positions+1
        with self.assertRaises(RuntimeError):c._current_positions(meta,m)

    def ring_fixture(self):
        t=self.t;c=controller();c.t=t;c.ids=(1,)*200;c.ring={};c.decode_ring_events=[]
        mod=NS(layer_idx=3);cache=NS(swa_cache=t.zeros((1,128,1,512),dtype=t.bfloat16))
        positions=list(range(72,200));slots=t.arange(128,dtype=t.int32)
        meta=NS(slot_mapping_store_swa=slots,canonical_request_binding={'end':200})
        values=t.arange(128,dtype=t.bfloat16)[:,None,None].repeat(1,1,512)
        def store(value,cache,selected_slots):
            valid=selected_slots>=0
            cache.reshape(128,1,512).index_copy_(0,selected_slots[valid].long(),value[valid])
        def load(cache,segments,count,out):
            indices=segments[:,0]*128+segments[:,1]
            out.copy_(cache.reshape(128,1,512).index_select(0,indices.long()))
        return c,mod,cache,meta,positions,values,store,load

    def test_final_ring_physical_readback_covers_exact_last128(self):
        c,mod,cache,meta,p,v,store,load=self.ring_fixture()
        c._store_and_check_ring(mod,cache,meta,p,self.t.arange(128),v,store,load)
        event=c.decode_ring_events[0]
        self.assertEqual(event['positions'],p)
        self.assertTrue(event['physical_readback_exact'])
        self.assertEqual(event['physical_slots'].tolist(),list(range(128)))

    def test_missing_final_ring_position_fails_before_decode(self):
        c,mod,cache,meta,p,v,store,load=self.ring_fixture()
        with self.assertRaises(ValidationError):
            c._store_and_check_ring(mod,cache,meta,p[:-1],self.t.arange(127),v[:-1],store,load)

    def test_final_ring_padding_or_colliding_slots_fail(self):
        for bad in (-1,0):
            c,mod,cache,meta,p,v,store,load=self.ring_fixture()
            meta.slot_mapping_store_swa[-1]=bad
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):
                c._store_and_check_ring(mod,cache,meta,p,self.t.arange(128),v,store,load)

    def test_final_ring_readback_corruption_fails(self):
        c,mod,cache,meta,p,v,store,load=self.ring_fixture()
        with self.assertRaises(RuntimeError):
            c._store_and_check_ring(mod,cache,meta,p,self.t.arange(128),v,store,
                                    lambda cache,segments,count,out:out.zero_())

    def test_final_ring_coverage_can_span_two_prompt_chunks(self):
        c,mod,cache,meta,p,v,store,load=self.ring_fixture()
        first=NS(slot_mapping_store_swa=meta.slot_mapping_store_swa[:78],canonical_request_binding={'end':150})
        c._store_and_check_ring(mod,cache,first,p[:78],self.t.arange(78),v[:78],store,load)
        self.assertEqual(c.decode_ring_events,[])
        second=NS(slot_mapping_store_swa=meta.slot_mapping_store_swa[78:],canonical_request_binding={'end':200})
        c._store_and_check_ring(mod,cache,second,p[78:],self.t.arange(50),v[78:],store,load)
        self.assertEqual(c.decode_ring_events[0]['positions'],p)


if __name__ == '__main__':unittest.main()
