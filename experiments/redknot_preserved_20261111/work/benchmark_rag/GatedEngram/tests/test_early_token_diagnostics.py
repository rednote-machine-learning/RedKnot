import copy
from contextlib import nullcontext
import importlib.util
from types import SimpleNamespace as NS
import unittest
import numpy as np
from kv_cache_engram.core import ValidationError
from kv_cache_engram.early_token_diagnostics import (EarlyTokenDiagnostics, LAYERS, pack_rows, unpack_rows, scalar_summary, rank_comparison)
from kv_cache_engram.headtail_link_plan import native_positions


def native_layers(routing=(2,), ratio=2, sparse=True):
    return [NS(layer_idx=i,self_attn=NS(layer_idx=i,has_packed_kv=i in routing,
        kv_compress_ratio=ratio if i in routing else 0,is_sparse=sparse if i in routing else False,
        layer_cfg={'index_topk':2 if sparse and i in routing else 0})) for i in range(3)]


def finish_fixture(routing=()):
    total=32810
    doc=dict(key='d',start=17,end=32785)
    c=NS(mode='native',arm='recomputed',diagnostic=True,docs=[doc],layers=native_layers(routing),
        ids=tuple(range(total)),phase='diagnostic',input_sha='a'*64,_entry=lambda key,layer:dict(version=0))
    observer=EarlyTokenDiagnostics(controller=c)
    for layer in LAYERS:
        for start in range(0,total,8192):
            end=min(start+8192,total)
            observer.chunks.add((layer,start,end))
            p=np.array([x-doc['start'] for x in native_positions(start,end,total,layer)
                        if doc['start'] <= x < doc['end']],dtype=np.int64)
            if len(p):
                v=np.ones((len(p),3),np.float32)
                observer.rows.setdefault((layer,0),[]).append((p,v))
    return observer

class DiagnosticEvidenceTests(unittest.TestCase):
    def test_lossless_evidence_and_corruption_detection(self):
        p=np.array([0,127,128,32767]);v=np.array([[0,1,0],[4,4,.5],[2,8,1],[1,1,2]],dtype=np.float32)
        record=pack_rows(p,v);pp,vv=unpack_rows(record)
        np.testing.assert_array_equal(pp,p);np.testing.assert_array_equal(vv,v)
        for field,value in [('raw_sha256','0'*64),('rows',5)]:
            bad=copy.deepcopy(record);bad[field]=value
            with self.assertRaises(ValidationError):unpack_rows(bad)

    def test_invalid_scalar_evidence_rejected(self):
        for p,v in [([1,1],[[1,1,1],[1,1,1]]),([1,2],[[float('nan'),1,1],[1,1,1]]),([1,2],[[-1,1,1],[1,1,1]])]:
            with self.assertRaises(ValidationError):unpack_rows(pack_rows(p,v))

    def test_boundary_propagation_beyond_first128(self):
        p=np.arange(32768);v=np.zeros((len(p),3),np.float32);v[:,1]=1;v[:384,0]=1
        s=scalar_summary(p,v)
        self.assertAlmostEqual(s['boundaries']['head128']['error_energy_fraction'],1/3)
        self.assertAlmostEqual(s['boundaries']['head256']['error_energy_fraction'],2/3)
        self.assertEqual(s['boundaries']['head384']['error_energy_fraction'],1)
        self.assertEqual(s['boundaries']['tail128']['error_energy_fraction'],0)
        self.assertEqual(s['boundaries']['head128_tail128']['rows'],256)

    def test_zero_error_does_not_claim_energy_coverage(self):
        p=np.arange(300);v=np.zeros((300,3));v[:,1]=1;s=scalar_summary(p,v)
        self.assertIsNone(s['boundaries']['head128']['error_energy_fraction'])
        self.assertEqual(s['relative_frobenius'],0)
        self.assertIsNone(rank_comparison(p,v,p,v)['spearman'])

    def test_sparse_yoco_uses_observed_common_positions(self):
        p=np.array([0,200,32639,32640,32767]);v=np.ones((5,3));s=scalar_summary(p,v)
        self.assertEqual(s['observed_rows'],5);self.assertEqual(s['boundaries']['tail128']['rows'],2)
        self.assertEqual(s['coverage'],5/32768)
        b=np.array([200,32767]);w=np.array([[1,1,0],[4,1,0]])
        self.assertEqual(rank_comparison(p,v,b,w)['common_rows'],2)

    def test_relative_error_rank_ties(self):
        p=np.arange(300);v=np.zeros((300,3));v[:,0]=np.arange(300)//2;v[:,1]=1
        a=rank_comparison(p,v,p,v)
        self.assertAlmostEqual(a['spearman'],1);self.assertEqual(a['top256_overlap_fraction'],1)
        reverse=v.copy();reverse[:,0]=v[::-1,0]
        self.assertAlmostEqual(rank_comparison(p,v,p,reverse)['spearman'],-1)

    def test_no_observer_in_formal_or_candidate_phase(self):
        for mode,arm,enabled in [('sensitive','headtail128_link',True),('native','recomputed',False)]:
            with self.assertRaises(ValidationError):EarlyTokenDiagnostics(controller=NS(mode=mode,arm=arm,diagnostic=enabled))

    def test_exact_native_coverage_and_placeholder_groups(self):
        report=finish_fixture().finish()
        early=next(r for r in report['records'] if r['layer']==0)
        late=next(r for r in report['records'] if r['layer']==20)
        self.assertEqual(early['native_row_groups']['full_layout']['rows'],32768)
        self.assertIsNone(early['native_yoco_rectangle_lo'])
        self.assertEqual(late['summary']['observed_rows'],118)
        self.assertEqual(late['native_yoco_rectangle_lo'],32670)
        self.assertEqual(late['native_row_groups']['yoco_placeholder']['rows'],3)
        self.assertEqual(late['native_row_groups']['yoco_final_rectangle']['rows'],115)
        self.assertEqual(late['native_row_groups']['full_layout']['rows'],0)
        self.assertIsNone(late['native_row_groups']['full_layout']['relative_frobenius'])
        self.assertTrue(report['native_position_coverage_checked'])
        self.assertEqual(report['question_routing_layers'],[])

    def test_missing_chunk_or_surviving_yoco_position_rejected(self):
        observer=finish_fixture();observer.chunks.remove((39,0,8192))
        with self.assertRaisesRegex(ValidationError,'layer/chunk coverage'):observer.finish()
        observer=finish_fixture();observer.rows[39,0]=observer.rows[39,0][1:]
        with self.assertRaisesRegex(ValidationError,'position coverage'):observer.finish()
        observer=finish_fixture();p,v=observer.rows[20,0][0]
        observer.rows[20,0][0]=(p-1,v)
        with self.assertRaisesRegex(ValidationError,'position coverage'):observer.finish()

    def test_actual_routing_topology_not_assumed_all_first_three(self):
        observer=finish_fixture(routing=(2,))
        with self.assertRaisesRegex(ValidationError,'question routing coverage'):observer.finish()
        observer=finish_fixture(routing=(2,))
        observer.index_events=[dict(layer=2,chunk_start=32768,chunk_end=32810,
                                   query_absolute_positions=list(range(32785,32810)))]
        report=observer.finish()
        self.assertEqual(report['question_routing_layers'],[2])
        self.assertEqual(report['question_routing_compress_ratios'],{'2':2})
        self.assertEqual(report['question_routing_selection_kinds'],{'2':'sparse_topk'})

    def test_duplicate_question_routing_chunk_rejected(self):
        observer=finish_fixture(routing=(2,))
        event=dict(layer=2,chunk_start=32768,chunk_end=32810,
                   query_absolute_positions=list(range(32785,32810)))
        observer.index_events=[event,event]
        with self.assertRaisesRegex(ValidationError,'question routing coverage'):observer.finish()

@unittest.skipUnless(importlib.util.find_spec('torch'), 'remote CPU torch validation required')
class ObserverCPUTests(unittest.TestCase):
    def test_gapped_positions_and_no_mutation(self):
        import torch
        doc=dict(key='d',start=100,end=32868)
        bank=torch.arange(32768,dtype=torch.float32)[:,None].expand(-1,512).to(torch.bfloat16).clone()
        entry=dict(tensor=bank,version=bank._version)
        c=NS(mode='native',arm='recomputed',diagnostic=True,t=torch,docs=[doc],layers=native_layers(),
             _entry=lambda key,layer:entry,allow_sync=nullcontext,
             _current_positions=lambda meta,marker:meta.positions.tolist(),
             _upload_indices=lambda ids,device:torch.tensor(ids,device=device,dtype=torch.long),
             ops=NS(normalize=lambda module,raw:raw.clone()))
        observer=EarlyTokenDiagnostics(controller=c)
        pos=torch.tensor([99,100,101,32740,32867,32868]);raw=torch.zeros((6,512),dtype=torch.bfloat16)
        raw[1:5]=bank[torch.tensor([0,1,32640,32767])];raw[2]+=2
        before=raw.clone();version=bank._version
        meta=NS(is_prompt=True,positions=pos,canonical_request_binding=dict(start=0,end=40000))
        observer.observe_swa(NS(layer_idx=20),raw,meta);p,v=observer.rows[20,0][0]
        np.testing.assert_array_equal(p,[0,1,32640,32767]);self.assertEqual(v[1,0],2048)
        self.assertEqual(v[[0,2,3],0].sum(),0)
        self.assertTrue(torch.equal(raw,before));self.assertEqual(bank._version,version)
        with self.assertRaises(ValidationError):observer.observe_swa(NS(layer_idx=20),raw,meta)

    def routing_fixture(self, *, sparse=True):
        import torch
        c=NS(mode='native',arm='recomputed',diagnostic=True,t=torch,docs=[dict(key='d',start=0,end=4)],
             layers=native_layers(sparse=sparse),ops=NS(PG=NS(context_parallel_group=lambda:NS(size=lambda:1))),
             _entry=lambda key,layer:dict(version=0),allow_sync=nullcontext,
             _current_positions=lambda meta,marker:[2,3,4,5],
             _upload_indices=lambda ids,device:torch.tensor(ids,device=device,dtype=torch.long))
        observer=EarlyTokenDiagnostics(controller=c)
        indices=torch.tensor([[0,-1],[0,1],[1,-1],[0,2]],dtype=torch.int32)
        meta=NS(is_prompt=True,num_reqs=1,total_q=4,canonical_request_binding=dict(start=0,end=6),
                ks={'2':torch.tensor([0,0,0,0],dtype=torch.int32)},
                ke={'2':torch.tensor([1,2,2,3],dtype=torch.int32)},
                kv_ca_index_offset_len={'2':torch.tensor([[0,1],[0,2],[0,2],[0,3]],dtype=torch.int32)})
        return observer,c.layers[2].self_attn,indices,meta

    def test_question_suffix_routing_only(self):
        observer,module,indices,meta=self.routing_fixture()
        observer.observe_global_index(module,indices,meta)
        event=observer.index_events[0]
        self.assertEqual(event['query_absolute_positions'],[4,5]);self.assertEqual(event['native_global_indices'],[[1,-1],[0,2]])
        self.assertEqual(event['native_compressed_key_starts'],[0,0])
        self.assertEqual(event['native_compressed_key_ends'],[2,3])
        self.assertEqual(event['token_interval'],'[index*ratio,(index+1)*ratio)')
        self.assertEqual(event['selection_kind'],'sparse_topk')

    def test_noncausal_invalid_or_foreign_routing_rejected(self):
        import torch
        for bad in (-2,2):
            observer,module,indices,meta=self.routing_fixture();indices[2,0]=bad
            with self.assertRaisesRegex(ValidationError,'invalid or noncausal'):
                observer.observe_global_index(module,indices,meta)
        observer,module,indices,meta=self.routing_fixture();meta.num_reqs=2
        with self.assertRaisesRegex(ValidationError,'CP1 single-request'):
            observer.observe_global_index(module,indices,meta)
        observer,module,indices,meta=self.routing_fixture();meta.ks['2'][2]=1
        with self.assertRaisesRegex(ValidationError,'absolute blocks'):
            observer.observe_global_index(module,indices,meta)
        observer,module,indices,meta=self.routing_fixture()
        with self.assertRaisesRegex(ValidationError,'diagnostic layout'):
            observer.observe_global_index(module,indices.float(),meta)
        observer,module,indices,meta=self.routing_fixture()
        with self.assertRaisesRegex(ValidationError,'index width'):
            observer.observe_global_index(module,indices[:,:1],meta)
        observer,module,indices,meta=self.routing_fixture()
        observer.c.ops.PG=NS(context_parallel_group=lambda:NS(size=lambda:2))
        with self.assertRaisesRegex(ValidationError,'CP1 single-request'):
            observer.observe_global_index(module,indices,meta)

    def test_dense_routing_has_distinct_label_and_native_bounds(self):
        observer,module,indices,meta=self.routing_fixture(sparse=False)
        observer.observe_global_index(module,indices,meta)
        self.assertEqual(observer.index_events[0]['selection_kind'],'dense_causal_enumeration')
        observer,module,indices,meta=self.routing_fixture(sparse=False)
        meta.kv_ca_index_offset_len['2'][2,1]=3
        with self.assertRaisesRegex(ValidationError,'absolute blocks'):
            observer.observe_global_index(module,indices,meta)

if __name__=='__main__':unittest.main()
