"""CPU policy and Torch mock regressions; these are not GPU qualification."""
import copy
import importlib.util
import sys
import unittest
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch

from kv_cache_engram.core import ValidationError
from kv_cache_engram.index_hit20 import IndexHit20Controller
from kv_cache_engram.headtail_link import HeadTailLinkController, make_union_indices, check_native_window
from kv_cache_engram.headtail_link_plan import plan, materialize_indices


def controller():
    layers = [NS(layer_idx=i, self_attn=NS(yoco_swa_share_hidden=False)) for i in range(40)]
    return HeadTailLinkController(layers, model_sha256='a' * 64,
        execution_sha256='b' * 64, tokenizer_sha256='c' * 64)


class HeadTailControllerPolicyTests(unittest.TestCase):
    def test_install_requires_actual_false_model_flags_before_hooks(self):
        c = controller()
        with patch.object(IndexHit20Controller, 'install', return_value=c): c.install()
        self.assertIs(c.native_yoco_swa_share_hidden, False)
        for flag in (True, None, 0, 'false'):
            c = controller(); c.layers[20].self_attn.yoco_swa_share_hidden = flag
            with patch.object(IndexHit20Controller, 'install') as install, self.assertRaises(ValidationError):
                c.install()
            install.assert_not_called()

    def test_report_rechecks_config_drift_and_preserves_native_qualification(self):
        c = controller(); c.native_yoco_swa_share_hidden = False; c.mode = 'all_selected'
        with patch.object(IndexHit20Controller, 'finish', return_value={'identity_check_count': 80}):
            r = c.finish()
            self.assertEqual(r['identity_check_count'], 80)
            self.assertEqual(r['dynamic_refresh_start_layer'], 3)
            self.assertFalse(r['dynamic_document_swa'])
            c.layers[39].self_attn.yoco_swa_share_hidden = True
            with self.assertRaises(ValidationError): c.finish()

    def test_phase_reset_clears_all_overlays_and_oracles(self):
        c = controller()
        c.boundaries[3] = object(); c.boundary_refreshed[3] = {3}
        c.ring[3] = object(); c.union_oracle_cache['x'] = object()
        c._clear_phase()
        for name in ('boundaries', 'boundary_refreshed', 'boundary_chunk_end', 'ring', 'union_oracle_cache'):
            self.assertEqual(getattr(c, name), {})

    def test_yoco_missing_boundaries_emit_explicit_offline_fallback_without_torch(self):
        c = controller(); c.phase_generation = 9
        c.boundary_positions_cpu = [0, 1, 98, 99, 100, 101, 198, 199]
        c.boundary_lookup = {p: i for i, p in enumerate(c.boundary_positions_cpu)}
        meta = NS(canonical_request_binding={'start': 0, 'end': 91})
        c._refresh_boundaries(NS(layer_idx=20), meta, [90], [], None, None, checked=True)
        e = c.headtail_swa_events[0]
        self.assertEqual(e['refreshed_positions'], [])
        self.assertEqual(e['visible_offline_fallback_positions'], [0, 1])
        self.assertEqual(e['checks'], 0)
        self.assertEqual(e['source'], 'current_layer_attention_input')

    def test_layer_local_chunk_order_is_enforced(self):
        c = controller(); meta = NS(canonical_request_binding={'start': 0, 'end': 4})
        c._refresh_boundaries(NS(layer_idx=3), meta, [3], [], None, None, checked=False)
        with self.assertRaises(ValidationError):
            c._refresh_boundaries(NS(layer_idx=3), meta, [3], [], None, None, checked=False)
        c._refresh_boundaries(NS(layer_idx=4), meta, [3], [], None, None, checked=False)
        with self.assertRaises(ValidationError):
            c._refresh_boundaries(NS(layer_idx=2), meta, [3], [], None, None, checked=False)

    def test_real_no_refresh_event_roundtrips_independent_reader(self):
        from gpu_pilot.read_headtail_link_results import audit_refresh_event, boundary_positions
        spans = [(i*32768,(i+1)*32768,str(i)*64) for i in range(8)]
        for layer in (20,24,39):
            c = controller(); c.phase_generation = 7
            c.boundary_positions_cpu = boundary_positions(spans)
            c.boundary_lookup = {p:i for i,p in enumerate(c.boundary_positions_cpu)}
            meta = NS(canonical_request_binding={'start':0,'end':8192})
            c._refresh_boundaries(NS(layer_idx=layer),meta,[8191],[],None,None,checked=True)
            count = audit_refresh_event(c.headtail_swa_events[-1],[8191],spans,set(),generation=7,qualification=True)
            self.assertEqual(count,0)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,
                     'requires CPU Torch; must pass unskipped remotely')
class HeadTailTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch

    def test_union_gpu_algorithm_matches_scalar_native_boundary_yoco_oracles(self):
        t = self.t
        docs = [dict(start=0, end=32768), dict(start=32768, end=65536)]
        for layer, start, end, total in ((3, 32760, 32780, 65540), (20, 32760, 32780, 65540),
                                        (24, 65400, 65540, 65540), (39, 65536, 65540, 65540)):
            from kv_cache_engram.headtail_link_plan import native_positions
            current = native_positions(start, end, total, layer)
            e = plan(docs, current, start, end, total, layer=layer)
            positions = t.tensor(current, dtype=t.int32)
            full = t.tensor(materialize_indices(e), dtype=t.int32)
            native = full[:, :128].contiguous()
            check_native_window(t, native, positions, e)
            for chosen in (list(range(len(current))), e['selected_local_rows']):
                active = t.tensor(chosen, dtype=t.long)
                actual = make_union_indices(t, native, positions, active, e)
                self.assertTrue(t.equal(actual, full.index_select(0, active)))
            bad = native.clone(); bad[0, 0] = 0 if bad[0, 0] != 0 else 1
            with self.assertRaises(RuntimeError): check_native_window(t, bad, positions, e)

    def test_cpu_vector_oracle_cache_is_rechecked_and_chunk_bounded(self):
        t = self.t; c = controller(); c.t = t
        docs = [dict(start=0, end=32768)]
        e = plan(docs, list(range(32768, 32772)), 32768, 32772, 32772, layer=3)
        expected = t.tensor(materialize_indices(e), dtype=t.int32)
        c._check_union_oracle(e, e['current_positions'], expected)
        self.assertEqual(len(c.union_oracle_cache), 1)
        bad = expected.clone(); bad[0, 0] = -1
        with self.assertRaises(RuntimeError): c._check_union_oracle(e, e['current_positions'], bad)
        e2 = plan(docs, [32771], 32771, 32772, 32772, layer=20)
        c._check_union_oracle(e2, e2['current_positions'], t.tensor(materialize_indices(e2), dtype=t.int32))
        self.assertEqual(len(c.union_oracle_cache), 1)

    def test_layer3_refresh_persists_across_chunks_but_not_layers_or_phases(self):
        t = self.t; c = controller(); c.t = t; c.phase_generation = 3
        c.boundary_positions_cpu = [0, 1, 8, 9, 10, 11]
        c.boundary_lookup = {p: i for i, p in enumerate(c.boundary_positions_cpu)}
        overlay = t.arange(6, dtype=t.bfloat16).reshape(6, 1, 1)
        meta = NS(canonical_request_binding={'start': 0, 'end': 5})
        c._refresh_boundaries(NS(layer_idx=3), meta, list(range(5)), [0, 1],
            t.tensor([50, 51], dtype=t.bfloat16).reshape(2, 1, 1), overlay, checked=True)
        self.assertEqual(overlay.flatten().tolist(), [50, 51, 2, 3, 4, 5])
        meta.canonical_request_binding = {'start': 5, 'end': 12}
        c._refresh_boundaries(NS(layer_idx=3), meta, list(range(5, 12)), [3, 4, 5, 6],
            t.tensor([58, 59, 60, 61], dtype=t.bfloat16).reshape(4, 1, 1), overlay, checked=True)
        self.assertEqual(overlay.flatten().tolist(), [50, 51, 58, 59, 60, 61])
        self.assertEqual(c.headtail_swa_events[-1]['available_refreshed_positions'], [0, 1, 8, 9, 10, 11])
        other = t.full_like(overlay, 70)
        meta.canonical_request_binding = {'start': 0, 'end': 5}
        c._refresh_boundaries(NS(layer_idx=4), meta, list(range(5)), [0, 1],
            t.tensor([80, 81], dtype=t.bfloat16).reshape(2, 1, 1), other, checked=False)
        self.assertEqual(other.flatten().tolist(), [80, 81, 70, 70, 70, 70])
        self.assertEqual(overlay.flatten().tolist(), [50, 51, 58, 59, 60, 61])
        c._clear_phase()
        self.assertEqual(c.boundary_refreshed, {})

    def test_yoco_missing_tail_preserves_layer_offline_value(self):
        t = self.t; c = controller(); c.t = t
        c.boundary_positions_cpu = [0, 1, 8, 9]
        c.boundary_lookup = {p: i for i, p in enumerate(c.boundary_positions_cpu)}
        overlay = t.tensor([10, 11, 18, 19], dtype=t.bfloat16).reshape(4, 1, 1)
        c._refresh_boundaries(NS(layer_idx=24), NS(canonical_request_binding={'start': 0, 'end': 10}),
            [9], [0], t.tensor([99], dtype=t.bfloat16).reshape(1, 1, 1), overlay, checked=False)
        self.assertEqual(overlay.flatten().tolist(), [10, 11, 18, 99])
        self.assertEqual(c.headtail_swa_events[0]['visible_offline_fallback_positions'], [0, 1, 8])

    def forward_fixture(self, mode='sensitive', *, source=False):
        t = self.t; c = controller(); c.t = t; c.state = 'RUNNING'; c.mode = mode; c.verify_sparse = False
        c.phase_generation = 5; c.phase = 'p'; c.rid = 'r'; c.ids = (1,) * 262148
        c.docs = tuple(dict(start=i*32768,end=(i+1)*32768,key=str(i)*64) for i in range(8))
        c.boundary_positions_cpu = [p for d in c.docs for p in
            list(range(d['start'],d['start']+128))+list(range(d['end']-128,d['end']))]
        c.boundary_lookup = {p:i for i,p in enumerate(c.boundary_positions_cpu)}
        c.tail_positions_cpu = [p for d in c.docs for p in range(d['end']-128,d['end'])]
        c.tail_boundary_rows = [c.boundary_lookup[p] for p in c.tail_positions_cpu]
        layer = 8 if source else 3
        start, end = 32635, 32780
        c.boundary_chunk_end[layer] = start
        c.layouts = {}; c.plans = {}; c.swa_events = []; c.attention_events = []
        positions = t.arange(start,end,dtype=t.int32); n = len(positions)
        e = plan(c.docs, positions.tolist(), start,end,len(c.ids),layer=layer)
        native = t.tensor(materialize_indices(e),dtype=t.int32)[:,:128].contiguous()
        meta = NS(total_q=n,num_reqs=1,is_prompt=True,positions=positions,
            canonical_request_binding=dict(request_id='r',phase='p',start=start,end=end),
            total_k_swa=128,kv_swa_full_index=native,load_segments_swa=None,
            slot_mapping_store_swa=t.arange(n,dtype=t.int32),total_k_compressed={'2':17},
            ks={'2':t.zeros(n,dtype=t.int32)},load_segments_ca={'2':None},compute_attn_info=lambda *a:None)
        calls = []
        mod = NS(layer_idx=layer,use_swa=True,has_packed_kv=True,is_sparse=True,
            max_model_len=270336,window_size=128,kv_compress_ratio=2,ratio_key='2',
            shared_hidden_kv_proj=None,has_local_index=source,is_kv_source=source,use_shared_kv=True,
            yoco_swa_share_hidden=False,num_heads=1,qk_head_dim=512,q_a_norm=lambda x:x,
            _q_b_norm_rope=lambda x,p:x,_resolve_shared_hidden=lambda x,s:s)
        def projection(x): calls.append(('swa',len(x))); return x.clone()
        def qa(x): calls.append(('qa',len(x))); return x[:,:4]
        qb = lambda x:x[:,:1].repeat(1,512)
        c.projections[layer] = (qa,projection); c.q_b_projections[layer] = (qb,lambda x:x)
        def prepare(raw,cache,md,cp):
            return (raw*2+(md.positions%7)[:,None]).reshape(-1,1,512),None
        c.native_prepares[layer] = prepare
        mod._project_q_and_swa = lambda h,shared:(h[:,:4],h.clone())
        mod._project_q_b = lambda q,p:(qb(q).reshape(-1,1,512),q if source else None)
        def identity(a,b):
            if not t.equal(a,b): raise ValidationError('mock independent byte parity failed')
        c.ops = NS(identity_error=identity,normalize=lambda m,x:x*2,
            postprocess=lambda m,x,md:((x+(md.positions%7)[:,None]).reshape(-1,1,512),None))
        c._canonical = lambda *args:t.full((n,512),37,dtype=t.bfloat16)
        boundaries = t.full((2048,1,512),44,dtype=t.bfloat16)
        c._restored_boundaries = lambda *args:boundaries
        def global_project(x): calls.append(('global_project',len(x))); return None,None
        def global_index(*args): calls.append(('global_index',n)); return t.zeros((n,512),dtype=t.int32)
        mod._project_current_hidden_inputs = global_project; mod._compute_kv_ca_index = global_index
        def attention(q,kv,g,indices,cached,current,md):
            calls.append(('attention',len(q),cached,current,indices.clone(),kv.clone()))
            # A deterministic per-query operation makes dense/sparse numerical
            # equality sensitive to changed key sets, not only query gather.
            values = kv[:,0,0].float().index_select(0,indices.clamp(min=0).long().flatten()).reshape(indices.shape)
            sums = t.where(indices>=0,values,0).sum(dim=1).to(t.bfloat16)
            return q + sums[:,None,None]
        mod._prefill_attention = attention; mod._wv_and_o_proj = lambda x,p:x[:,0,:]
        cache = NS(ca_cache=t.zeros((1,1,512)),swa_cache=t.zeros((1,128,1,512)))
        cross = NS(shared_hidden=t.full((n,512),99,dtype=t.bfloat16),consumer_full_metadata=None,
            loaded_prefill_kv=t.full((128+n+17,1,512),77,dtype=t.bfloat16),loaded_prefill_kv_main_offset=128+n)
        hidden = (positions%13+1).to(t.bfloat16)[:,None].repeat(1,512)
        load = ModuleType('load')
        def load_cache(cache_arg,segments,rows,out):
            calls.append(('load',rows)); out.fill_(77 if cache_arg is cache.ca_cache else 33)
        load.load_paged_kv_cache = load_cache
        store = ModuleType('store')
        store.store_kv_cache = lambda values,cache_arg,slots:calls.append(('store',values.clone(),slots.clone()))
        modules = {'deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache':load,
                   'deepinfer.model_executor.layers.tilelang_ops.store_kv_cache':store}
        return c,mod,meta,cache,cross,hidden,calls,modules

    def run_forward(self,mode='sensitive',**kwargs):
        c,m,md,cache,cross,h,calls,modules = self.forward_fixture(mode,**kwargs)
        with patch.dict(sys.modules,modules): out = c._selected_attention(m,h,cache,md,h+20,cross)
        return c,out,calls,cross

    def test_actual_forward_uses_current_hidden_keeps_middle_kv_and_native_global(self):
        c,out,calls,cross = self.run_forward()
        self.assertIn(('global_project',145),calls); self.assertIn(('global_index',145),calls)
        self.assertIn(('swa',140),calls)
        self.assertTrue(self.t.equal(out[:5],self.t.zeros_like(out[:5])))
        stored = next(v for name,*rest in calls if name=='store' for v in rest[:1])
        expected_middle = 37+(self.t.arange(32635,32640)%7)
        self.assertEqual(stored[:5,0,0].tolist(),expected_middle.tolist())
        self.assertEqual(stored[5,0,0].item(),(32640%13+1)*2+32640%7)
        self.assertEqual(cross.loaded_prefill_kv_main_offset,128+145+1024)
        self.assertTrue(self.t.equal(cross.loaded_prefill_kv[-17:],self.t.full((17,1,512),77,dtype=self.t.bfloat16)))
        self.assertLess(next(i for i,v in enumerate(calls) if v[0]=='load'),
                        next(i for i,v in enumerate(calls) if v[0]=='store'))
        event = c.headtail_memory_events[0]
        self.assertEqual(event['native_cache_store_rows'],145)
        self.assertEqual(c.headtail_swa_events[0]['refreshed_positions'],list(range(32640,32780)))

    def test_actual_dense_reference_matches_sparse_masked_output_and_global_source_rows(self):
        c,sparse,_,_ = self.run_forward(source=True)
        dense,reference,calls,_ = self.run_forward('reference_sparse',source=True)
        self.assertTrue(self.t.equal(sparse,reference))
        self.assertEqual(c.attention_events[0]['index_q_rows'],145)
        self.assertIn(('qa',145),calls)
        self.assertEqual(dense.headtail_memory_events[0]['actual_joint_query_rows'],145)
        self.assertEqual(c.headtail_memory_events[0]['actual_joint_query_rows'],140)
        self.assertEqual(dense.headtail_swa_events[0]['checks'],2)

    def test_wrong_subset_projection_and_wrong_native_position_transform_fail_qualification(self):
        for corruption in ('projection','prepare'):
            c,m,md,cache,cross,h,calls,modules = self.forward_fixture('reference_sparse')
            if corruption=='projection': c.projections[m.layer_idx]=(c.projections[m.layer_idx][0],lambda x:x+1)
            else: c.native_prepares[m.layer_idx]=lambda raw,cache,md,cp:(raw.reshape(-1,1,512),None)
            with patch.dict(sys.modules,modules),self.assertRaises(ValidationError):
                c._selected_attention(m,h,cache,md,h,cross)


if __name__ == '__main__':
    unittest.main()
