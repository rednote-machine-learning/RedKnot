"""CPU specification/actual controller-path tests, not GPU qualification."""
import importlib.util
import inspect
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from kv_cache_engram.core import ValidationError
from kv_cache_engram.index_hit_selector import select_by_hitcounts, raw_index_hitcounts
from kv_cache_engram.index_hit20 import IndexHit20Controller, FFN_LAYERS
from kv_cache_engram.index_hit20_adapter import IndexHit20Adapter
from tests.test_ffn_only_adapter import Layer, Rows, Ops, selected_metadata


class SelectorTests(unittest.TestCase):
    def test_import_is_cpu_only(self):
        proc = subprocess.run([sys.executable, "-B", "-c", "import sys; import kv_cache_engram.index_hit20; assert 'torch' not in sys.modules; assert not any(x.startswith('deepinfer') for x in sys.modules)"],
                              cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_first_three_only_are_full(self):
        self.assertEqual(FFN_LAYERS, tuple(range(3,40)))

    def test_budget_is_ceil_twenty_of_documents_plus_protected(self):
        result = select_by_hitcounts([0,1,8,7,6,5,4,3,2,0], list(range(10)), [(1,9)])
        self.assertEqual(result['selected_local_rows'], [0,2,3,9])
        self.assertEqual((result['optional_rows'],result['mandatory_rows']), (8,2))

    def test_zero_hits_tie_by_absolute_position(self):
        result = select_by_hitcounts([0]*11, list(range(20,31)), [(20,31)])
        self.assertEqual(result['selected_local_rows'], [0,1,2])
        self.assertEqual(result['zero_hit_document_rows'],11)
        self.assertEqual(result['selected_zero_hit_document_rows'],3)

    def test_compact_decoder_ranks_current_survivors(self):
        result = select_by_hitcounts([2,100,4,3,5,7], [1,7,100,200,201,260], [(0,250)])
        self.assertEqual(result['selected_local_rows'], [1,5])

    def test_single_surviving_document_row_not_forced_to_zero(self):
        self.assertEqual(select_by_hitcounts([0],[8191],[(0,32768)])['selected_local_rows'],[0])

    def test_empty_current_query_layout(self):
        self.assertEqual(select_by_hitcounts([],[],[(0,32768)])['selected_local_rows'],[])

    def test_all_selected_control_has_no_mask(self):
        self.assertEqual(select_by_hitcounts([0]*8,list(range(8)),[(0,8)],all_selected=True)['selected_local_rows'],list(range(8)))

    def test_frequency_not_intersection(self):
        counts=raw_index_hitcounts([[0,1,-1],[1,2,-1],[1,3,-1]], [7,8,9], list(range(8)),ratio=2,global_kv_rows=5)
        self.assertEqual(counts,[1,1,3,3,1,1,1,1])

    def test_c2_alignment_is_request_not_document_start(self):
        counts=raw_index_hitcounts([[1]], [7], [2,3,4],ratio=2,global_kv_rows=4)
        self.assertEqual(counts,[1,1,0])

    def test_offset_is_subtracted(self):
        self.assertEqual(raw_index_hitcounts([[11]], [3], [0,1,2],ratio=1,global_kv_rows=4,request_offset=10),[0,1,0])

    def test_incomplete_c2_group_gets_zero_hits(self):
        self.assertEqual(raw_index_hitcounts([[0]], [2], [0,1,2],ratio=2,global_kv_rows=1),[1,1,0])

    def test_noncausal_and_duplicate_indices_rejected(self):
        for rows in ([[2]],[[0,0]],[[-2]],[[4]]):
            with self.subTest(rows=rows), self.assertRaises(ValidationError):
                raw_index_hitcounts(rows,[3],[0],ratio=2,global_kv_rows=3)

    def test_invalid_positions_and_scores_rejected(self):
        for scores,pos in (([1,2],[2,1]),([1,2],[1,1]),([-.1],[0]),([True],[0]),([1],[0,1])):
            with self.subTest(scores=scores,pos=pos), self.assertRaises(ValidationError):
                select_by_hitcounts(scores,pos,[(0,5)])

    def test_many_budget_sizes(self):
        rng=random.Random(12)
        for n in range(1,100):
            hits=[rng.randrange(9) for _ in range(n)]
            result=select_by_hitcounts(hits,list(range(n)),[(0,n)])
            self.assertEqual(len(result['selected_local_rows']),(n+4)//5)
            self.assertEqual(result['selected_local_rows'],sorted(sorted(range(n),key=lambda i:(-hits[i],i))[:(n+4)//5]))


class AdapterTests(unittest.TestCase):
    def fixture(self, index=3, selected=(1,), reference=False, empty=False):
        layer=Layer(index)
        meta=NS(is_prompt=True,num_reqs=1,total_q=5,positions=list(range(5)),
                canonical_request_binding=dict(request_id='r',phase='p',start=0,end=5),
                valid_token_mask=[True]*5,gate_force_random=None,images_seq_mask=None)
        plan=NS(full_rows=5,selected_count=len(selected),selected_rows=list(selected),current_positions=meta.positions,
                request_id='r',phase='p',chunk_start=0,chunk_end=5,plan_sha256='a'*64,reference_dense=reference)
        adapter=IndexHit20Adapter([layer],lambda *args:plan,selected_metadata,ops=Ops(),
                                  allow_empty_native=(lambda *args: True) if empty else None).install()
        return layer,meta,adapter

    def test_layers_three_through_last_can_select(self):
        for index in (3,4,5,8,14,20,39):
            layer,meta,a=self.fixture(index)
            self.assertEqual(layer.mlp.forward(Rows(range(5)),Rows(range(5)),meta).values,[0,5,0,0,0])
            self.assertEqual(a.report()['counts']['actual_mlp_rows'],1)
            a.close()

    def test_first_three_cannot_select(self):
        for index in range(3):
            layer,meta,a=self.fixture(index)
            with self.assertRaises(ValueError): layer.mlp.forward(Rows(range(5)),Rows(range(5)),meta)
            a.close(allow_incomplete=True)

    def test_reference_executes_all_then_masks(self):
        layer,meta,a=self.fixture(reference=True)
        out=layer.mlp.forward(Rows(range(5)),Rows(range(5)),meta)
        self.assertEqual(out.values,[0,5,0,0,0])
        self.assertEqual(layer.mlp.calls[0][1].shape[0],5)
        self.assertEqual(a.report()['counts']['actual_mlp_rows'],5)
        a.close()

    def test_empty_still_enters_native_mlp(self):
        layer,meta,a=self.fixture(selected=(),empty=True)
        self.assertEqual(layer.mlp.forward(Rows(range(5)),Rows(range(5)),meta).values,[0]*5)
        self.assertEqual(layer.mlp.calls[0][1].shape[0],0)
        self.assertEqual(a.report()['counts']['empty_native_mlp_calls'],1)
        a.close()

    def test_empty_without_capability_fails(self):
        layer,meta,a=self.fixture(selected=())
        with self.assertRaises(ValueError): layer.mlp.forward(Rows(range(5)),Rows(range(5)),meta)
        self.assertEqual(len(layer.mlp.calls),0)
        a.close(allow_incomplete=True)


TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH,'Torch CPU runtime unavailable')
class TorchSelectionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch

    def controller(self,positions=None,mode='sensitive',ratio=2):
        t=self.t
        layers=tuple(NS(layer_idx=i,self_attn=NS(has_local_index=i in (2,8,14,20,24,28,32,36))) for i in range(40))
        c=IndexHit20Controller(layers,model_sha256='a'*64,execution_sha256='b'*64,tokenizer_sha256='c'*64,
                              native_model=NS(layers=layers,yoco_boundary_layer_idx=20))
        c.t=t;c.mode=mode;c.rid='r';c.phase='p';c.ids=(1,)*16;c.input_sha='d'*64;c.state='RUNNING'
        c.docs=(dict(start=1,end=15,key='e'*64),)
        c.plans={};c.layouts={};c.index_groups={};c.live_indices=[];c.live_index_chunk=None;c.verify_sparse=False
        c.prepared={(0,16):dict(optional_mask=t.tensor([False]+[True]*14+[False]))}
        pos=t.arange(16) if positions is None else t.tensor(positions)
        meta=NS(total_q=pos.numel(),positions=pos,num_reqs=1,is_prompt=True,
                canonical_request_binding=dict(request_id='r',phase='p',start=0,end=16),
                ks={str(ratio):t.zeros(pos.numel(),dtype=t.int32)},total_k_compressed={str(ratio):16//ratio})
        mod=NS(layer_idx=3,kv_compress_ratio=ratio,ratio_key=str(ratio))
        return c,mod,meta

    def index(self,meta,ratio=2):
        t=self.t
        value=t.full((meta.total_q,512),-1,dtype=t.int32)
        for i,p in enumerate(meta.positions.tolist()):
            count=(p+1)//ratio
            value[i,:count]=t.arange(count,dtype=t.int32)
        return value

    def test_gpu_style_histogram_equals_cpu_oracle(self):
        c,mod,meta=self.controller();idx=self.index(meta)
        plan=c._index_plan(mod,meta,idx)
        expected=raw_index_hitcounts(idx.tolist(),meta.positions.tolist(),meta.positions.tolist(),ratio=2,global_kv_rows=8)
        self.assertEqual(plan['hits'].tolist(),expected)
        self.assertEqual(plan['ids'].tolist(),select_by_hitcounts(expected,list(range(16)),[(1,15)])['selected_local_rows'])

    def test_shared_index_reuses_one_group(self):
        c,mod,meta=self.controller();idx=self.index(meta)
        c._index_plan(mod,meta,idx);mod.layer_idx=4;c._index_plan(mod,meta,idx)
        self.assertEqual(len(c.index_groups),1)
        self.assertEqual(next(iter(c.index_groups.values()))['source_layer'],2)

    def test_formal_does_not_transfer_full_raw_indices(self):
        c,mod,meta=self.controller();c._index_plan(mod,meta,self.index(meta))
        self.assertIsNone(next(iter(c.index_groups.values()))['tensor'])

    def test_all_selected_selects_all_and_has_no_histogram_dump(self):
        c,mod,meta=self.controller(mode='all_selected');plan=c._index_plan(mod,meta,self.index(meta))
        self.assertEqual(plan['ids'].tolist(),list(range(16)))
        self.assertEqual(c.index_groups,{})

    def test_upper_compact_budget_uses_only_surviving_rows(self):
        c,mod,meta=self.controller(positions=[1,3,7,8,10,15]);mod.layer_idx=39
        plan=c._index_plan(mod,meta,self.index(meta))
        self.assertEqual(plan['optional_rows'],5)
        self.assertEqual(plan['mandatory_rows'],1)
        self.assertEqual(plan['ids'].numel(),2)

    def test_zero_query_layout_is_legal(self):
        c,mod,meta=self.controller(positions=[]);meta.positions=meta.positions.long()
        plan=c._index_plan(mod,meta,self.index(meta))
        self.assertEqual(plan['ids'].numel(),0)

    def test_nonzero_offset_rejected(self):
        c,mod,meta=self.controller();meta.ks['2'].fill_(1)
        with self.assertRaises(RuntimeError):c._index_plan(mod,meta,self.index(meta))

    def test_duplicate_layer_plan_rejected(self):
        c,mod,meta=self.controller();idx=self.index(meta);c._index_plan(mod,meta,idx)
        with self.assertRaises(ValidationError):c._index_plan(mod,meta,idx)

    def test_qualification_raw_copy_requires_output_dir(self):
        c,mod,meta=self.controller(mode='reference_sparse')
        with self.assertRaises(ValidationError):c._index_plan(mod,meta,self.index(meta))

    def test_qualification_copies_raw_indices_once(self):
        c,mod,meta=self.controller(mode='reference_sparse');c.index_output_dir=Path(tempfile.gettempdir())
        idx=self.index(meta);c._index_plan(mod,meta,idx)
        self.assertTrue(self.t.equal(next(iter(c.index_groups.values()))['tensor'],idx))

    def forward_fixture(self, mode='sensitive', source=False, positions=None):
        t=self.t;c,mod,meta=self.controller(positions=positions,mode=mode)
        if source:mod.layer_idx=8
        c.swa_events=[];c.attention_events=[];c.projection_events=[];c.projection_checks=0;c.restore_events=[]
        c.index_output_dir=Path(tempfile.gettempdir()) if mode=='reference_sparse' else None
        meta.total_k_swa=0;meta.load_segments_ca={'2':None};meta.load_segments_swa=None;meta.slot_mapping_store_swa=None
        meta.kv_swa_full_index=t.full((meta.total_q,128),-1,dtype=t.int32)
        meta.compute_attn_info=lambda *args:None
        calls=[]
        def project_q(x):calls.append(('qa',x.shape[0]));return x[:,:4].clone()
        def project_swa(x):calls.append(('swa',x.shape[0]));return x[:,:512].clone()
        def project_main(x):calls.append(('qb',x.shape[0]));return x[:,:1].repeat(1,512)
        def project_idx(x):calls.append(('indexq',x.shape[0]));return x[:,:1].clone()
        mod.use_swa=mod.has_packed_kv=mod.is_sparse=True;mod.window_size=128;mod.max_model_len=270336
        mod.use_shared_kv=True;mod.is_kv_source=source;mod.has_local_index=source;mod.yoco_swa_share_hidden=False
        mod.num_heads=1;mod.qk_head_dim=512;mod.q_lora_rank=4;mod.shared_hidden_kv_proj=None
        mod._resolve_shared_hidden=lambda bf16,shared:shared
        def global_project(x):calls.append(('global_project',x.shape[0]));return (x[:,:1],x[:,:1]) if source else (None,None)
        mod._project_current_hidden_inputs=global_project
        idx=self.index(meta)
        def global_index(waiter,weights,indexq,cache,metadata,cross):
            calls.append(('global_index',metadata.total_q));return idx
        mod._compute_kv_ca_index=global_index
        mod.q_a_norm=lambda x:x
        mod._q_b_norm_rope=lambda x,p:x
        mod._project_q_and_swa=lambda h,s:(h[:,:4].clone(),h[:,:512].clone())
        mod._project_q_b=lambda q,p:(q[:,:1].repeat(1,512).reshape(-1,1,512),q[:,:1].clone() if source else None)
        mod._prepare_prefill_kv=lambda rows,offset,device,cross:t.zeros(rows,1,512,dtype=t.bfloat16)
        def joint(q,kv,g,s,cached,current,metadata):
            calls.append(('joint',q.shape[0]));return q.clone()
        mod._prefill_attention=joint
        def output(q,p):calls.append(('output',q.shape[0]));return q[:,0,:].repeat(1,10)
        mod._wv_and_o_proj=output
        c.projections[mod.layer_idx]=(project_q,project_swa)
        c.q_b_projections[mod.layer_idx]=(project_main,project_idx if source else None)
        c.native_prepares[mod.layer_idx]=lambda raw,cache,metadata,cp:(raw.reshape(-1,1,512),None)
        c._canonical=lambda layer,marker,p:t.full((p.numel(),512),-7,dtype=t.bfloat16)
        c.ops=NS(rows=lambda x:x.shape[0],postprocess=lambda m,x,md:(x.reshape(-1,1,512),None),normalize=lambda m,x:x,
                 identity_error=lambda a,b:self.assertTrue(t.equal(a,b)))
        cache=NS(ca_cache=t.zeros(1,1,512),swa_cache=None)
        cross=NS(shared_hidden=None,consumer_full_metadata=None,loaded_prefill_kv=None,loaded_prefill_kv_main_offset=None)
        bf16=(meta.positions+1).to(t.bfloat16)[:,None].repeat(1,5120)
        load=ModuleType('load');load.load_paged_kv_cache=lambda cache,segments,rows,out:out.fill_(77)
        store=ModuleType('store');store.store_kv_cache=lambda swa,cache,slots:calls.append(('stored_swa',swa.clone()))
        modules={'deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache':load,
                 'deepinfer.model_executor.layers.tilelang_ops.store_kv_cache':store}
        return c,mod,meta,cache,cross,bf16,calls,modules

    def run_fixture(self,mode='sensitive',source=False,positions=None):
        c,mod,meta,cache,cross,bf16,calls,modules=self.forward_fixture(mode,source,positions)
        with patch.dict(sys.modules,modules):out=c._selected_attention(mod,bf16,cache,meta,bf16,cross)
        return c,mod,meta,out,calls

    def test_actual_forward_computes_only_selected_query_and_swa_rows(self):
        c,mod,meta,out,calls=self.run_fixture()
        selected=next(iter(c.plans.values()))['ids'];k=selected.numel()
        self.assertIn(('qa',k),calls);self.assertIn(('qb',k),calls);self.assertIn(('swa',k),calls)
        self.assertIn(('joint',k),calls);self.assertIn(('output',k),calls)
        self.assertIn(('global_index',16),calls)
        inactive=self.t.ones(16,dtype=self.t.bool);inactive[selected]=False
        self.assertTrue(self.t.equal(out[inactive],self.t.zeros_like(out[inactive])))
        stored=next(v for tag,v in calls if tag=='stored_swa')[:,0,:]
        self.assertTrue(self.t.equal(stored[inactive],self.t.full_like(stored[inactive],-7)))
        self.assertEqual(c.swa_events[0]['reused_rows'],16-k)

    def test_source_global_and_index_remain_full_even_when_attention_is_sparse(self):
        c,mod,meta,out,calls=self.run_fixture(source=True)
        k=next(iter(c.plans.values()))['ids'].numel()
        self.assertIn(('global_project',16),calls);self.assertIn(('qa',16),calls);self.assertIn(('indexq',16),calls)
        self.assertIn(('qb',k),calls);self.assertIn(('joint',k),calls)

    def test_dense_reference_and_sparse_outputs_match_in_cpu_fixture(self):
        _,_,_,sparse,_=self.run_fixture()
        c,_,_,reference,calls=self.run_fixture(mode='reference_sparse')
        self.assertTrue(self.t.equal(sparse,reference));self.assertIn(('joint',16),calls)
        self.assertEqual(c.attention_events[0]['joint_attention_rows'],16)

    def test_all_selected_actual_path_runs_projection_and_restore_checks(self):
        c,_,_,_,calls=self.run_fixture(mode='all_selected',source=True)
        self.assertIn(('joint',16),calls);self.assertEqual(c.projection_checks,5)
        self.assertEqual(c.restore_events[0]['checks'],2)

    def test_zero_query_actual_forward_never_fabricates_token(self):
        c,mod,meta,cache,cross,bf16,calls,modules=self.forward_fixture(positions=[])
        meta.positions=meta.positions.long()
        with patch.dict(sys.modules,modules):out=c._selected_attention(mod,bf16,cache,meta,bf16,cross)
        self.assertEqual(tuple(out.shape),(0,5120));self.assertNotIn(('joint',0),calls)
        self.assertEqual(c.swa_events[0]['full_rows'],0)
