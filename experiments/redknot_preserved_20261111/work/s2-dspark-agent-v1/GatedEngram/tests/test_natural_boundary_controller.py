"""Natural token boundaries are independent of native 8192-token chunks."""
import copy
import importlib.util
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from kv_cache_engram.core import ValidationError, token_digest
from kv_cache_engram.native_reuse import document_plan
from kv_cache_engram.headtail_comparison import ComparisonController
from kv_cache_engram.index_hit20 import IndexHit20Controller
from kv_cache_engram.headtail_link_plan import plan, native_positions, materialize_indices, query_key_positions
from kv_cache_engram.tail_link_plan import plan as static_plan
from kv_cache_engram.early_token_diagnostics import EarlyTokenDiagnostics, LAYERS, pack_rows, unpack_rows, scalar_summary


def controller(natural=True):
    layers = [NS(layer_idx=i, self_attn=NS(yoco_swa_share_hidden=False)) for i in range(40)]
    return ComparisonController(layers, model_sha256='a'*64, execution_sha256='b'*64,
        tokenizer_sha256='c'*64, natural_boundary_inputs=natural)


def geometry(lengths=(301, 901, 1701, 2377, 2903, 1109, 3401, 2003), gap=3):
    cursor, docs = 17, []
    for i, length in enumerate(lengths):
        docs.append(dict(start=cursor, end=cursor+length, key=str(i)*64))
        cursor += length+gap
    return docs, cursor+61


class NaturalBoundaryPolicyTests(unittest.TestCase):
    def test_natural_install_uses_new_protocol_and_same_single_owner(self):
        from tests.test_headtail_comparison import controller as configured_controller
        c=configured_controller();c.natural_boundary_inputs=True
        with patch.object(IndexHit20Controller,'install',return_value=c) as install:
            self.assertIs(c.install(),c)
        install.assert_called_once_with()
        self.assertIs(c.native_yoco_swa_share_hidden,False)

    def test_natural_native_and_capture_resets_bind_exact_spans_and_restore_native_chunk_policy(self):
        c=controller();c.state='IDLE';c.native_model=NS(yoco_boundary_layer_idx=20)
        c.original_yoco_boundary=20;c.adapter=NS(poisoned=False,in_call=False,events=[],counts={})
        docs,total=geometry();ids=tuple(range(total))
        supplied=[dict(input_start=d['start'],input_end=d['end'],
            token_ids_sha256=token_digest(ids[d['start']:d['end']])) for d in docs]
        with patch.object(c,'_bind_case_bank'):
            c.reset('native','native',arm='recomputed',input_ids=ids,documents=supplied,phase='native')
        self.assertEqual(c._comparison_fields()['document_token_lengths'],[d['end']-d['start'] for d in docs])
        self.assertEqual(c.native_model.yoco_boundary_layer_idx,20)
        c.state='COMPLETE';c.cache_output_dir=object()
        c.reset('capture','capture_independent',arm='offline',input_ids=ids[17:318],phase='capture')
        self.assertEqual(c._comparison_fields()['document_token_lengths'],[301])
        self.assertIsNone(c.native_model.yoco_boundary_layer_idx)
        c._restore_offline_layout()
        self.assertEqual(c.native_model.yoco_boundary_layer_idx,20)

    def test_explicit_optin_preserves_legacy_document_validation(self):
        docs, total = geometry()
        ids = list(range(total))
        supplied = [dict(input_start=d['start'], input_end=d['end'],
            token_ids_sha256=token_digest(ids[d['start']:d['end']])) for d in docs]
        parsed = document_plan(ids, supplied, natural_boundary_inputs=True)
        self.assertEqual([(d['start'],d['end']) for d in parsed], [(d['start'],d['end']) for d in docs])
        with self.assertRaises(ValidationError): document_plan(ids, supplied)
        for invalid in (0, 1, None, 'true'):
            with self.assertRaises(ValidationError): document_plan(ids, supplied, natural_boundary_inputs=invalid)
            with self.assertRaises(ValidationError): controller(invalid)
        for change in ('overlap', 'hash'):
            bad = copy.deepcopy(supplied)
            if change == 'overlap': bad[1]['input_start'] = bad[0]['input_end']-1
            else: bad[1]['token_ids_sha256'] = '0'*64
            with self.assertRaises(ValidationError): document_plan(ids, bad, natural_boundary_inputs=True)

    def test_independent_capture_uses_exact_length_without_padding(self):
        for length in (1,127,256,4093,8193,10003):
            ids = tuple(range(length))
            self.assertEqual(document_plan(ids,None,independent=True,natural_boundary_inputs=True),
                ({'start':0,'end':length,'key':token_digest(ids)},))
            with self.assertRaises(ValidationError): document_plan(ids,None,independent=True)
        with self.assertRaises(ValidationError): document_plan([],None,independent=True,natural_boundary_inputs=True)

    def test_short_boundary_union_and_variable_bank_metadata(self):
        c=controller(); docs,total=geometry((1,127,128,129,255,256,257,8193))
        c.docs=docs; c._bind_document_lengths(docs); c._set_boundary_positions(docs)
        expected=sorted({p for d in docs for p in list(range(d['start'],min(d['end'],d['start']+128)))
                        +list(range(max(d['start'],d['end']-128),d['end']))})
        self.assertEqual(c.boundary_positions_cpu,expected)
        self.assertEqual([expected[i] for i in c.tail_boundary_rows],c.tail_positions_cpu)
        self.assertEqual(len(expected),sum(min(256,d['end']-d['start']) for d in docs))
        c.bank={(d['key'],l):dict(scope=c._entry_scope(d['key'],l)) for d in docs for l in range(40)}
        fields=c._comparison_fields(); rows=40*sum(d['end']-d['start'] for d in docs)
        self.assertEqual(fields['offline_bank_rows'],rows)
        self.assertEqual(fields['offline_bank_bytes'],rows*1024)
        self.assertEqual(fields['document_token_lengths'],[d['end']-d['start'] for d in docs])
        self.assertEqual(fields['comparison_schema'],'headtail-natural-boundary-comparison-v1')
        self.assertEqual(c._entry_scope(docs[0]['key'],3)['source_document_tokens'],1)
        changed=dict(docs[0],end=docs[0]['end']+1)
        with self.assertRaises(ValidationError): c._bind_document_lengths([changed])

    def test_unaligned_boundaries_same_chunk_links_causal_union_and_native_ring_coordinates(self):
        docs,total=geometry((4011,4321,301,901,1701,2377,2903,1109))
        self.assertTrue(any(d['start']//8192 != (d['end']-1)//8192 for d in docs))
        points={p+delta for d in docs for p in (d['start'],d['end']-128,d['end']) for delta in (-1,0,1)}
        for layer in (3,19,20,39):
            for start in range(0,total,8192):
                end=min(start+8192,total); positions=native_positions(start,end,total,layer)
                event=plan(docs,positions,start,end,total,layer=layer)
                queries=sorted(set(positions)&(points|{positions[0],positions[-1]}))
                for q,indices in zip(queries,materialize_indices(event,queries)):
                    low=max(0,q-127,total-140 if layer>=20 else 0)
                    local=set(range(low,q+1))
                    extra={p for d in docs if d['end']<=q for p in range(d['end']-128,d['end'])}
                    actual=query_key_positions(event,q)
                    self.assertEqual(set(actual),local|extra)
                    self.assertEqual(len(actual),len(set(actual)))
                    self.assertTrue(all(p<=q for p in actual))
                    decoded=[start-128+i if i<128 else positions[i-128]
                             if i<128+len(positions) else event['tail_bank_positions'][i-event['tail_extra_offset']]
                             for i in indices if i>=0]
                    self.assertEqual(decoded,actual)
                self.assertTrue(all(p in positions for p in event['selected_positions']))

    def test_diagnostic_summary_uses_actual_tail_and_clipped_bins(self):
        p=np.arange(4093); v=np.ones((4093,3),np.float32)
        summary=scalar_summary(p,v,4093)
        self.assertEqual(summary['possible_rows'],4093)
        self.assertEqual(summary['boundaries']['tail128']['rows'],128)
        self.assertEqual(summary['boundaries']['head128_tail128']['rows'],256)
        self.assertEqual(summary['bins'][-1]['local_end'],4093)
        self.assertEqual(summary['bins'][-1]['local_start'],3965)
        short=scalar_summary(np.arange(193),np.ones((193,3)),193)
        self.assertEqual(short['boundaries']['head128_tail128']['rows'],193)

    def test_empty_scalar_evidence_has_no_fabricated_observations(self):
        payload=pack_rows(np.empty(0,dtype=np.int64),np.empty((0,3),dtype=np.float32))
        p,v=unpack_rows(payload)
        self.assertEqual(payload['rows'],0); self.assertEqual(payload['raw_bytes'],0)
        self.assertEqual(p.shape,(0,)); self.assertEqual(v.shape,(0,3))
        with self.assertRaises(ValidationError): scalar_summary(p,v,4093)

    def diagnostic_fixture(self):
        docs,total=geometry()
        layers=[NS(layer_idx=i,self_attn=NS(has_packed_kv=False,is_sparse=False)) for i in range(3)]
        c=NS(mode='native',arm='recomputed',diagnostic=True,natural_boundary_inputs=True,
            docs=docs,layers=layers,ids=tuple(range(total)),phase='diag',input_sha='a'*64,
            _entry=lambda key,layer:dict(version=0))
        observer=EarlyTokenDiagnostics(controller=c)
        for layer in LAYERS:
            for start in range(0,total,8192):
                end=min(start+8192,total); observer.chunks.add((layer,start,end))
                surviving=native_positions(start,end,total,layer)
                for ordinal,d in enumerate(docs):
                    p=np.asarray([x-d['start'] for x in surviving if d['start']<=x<d['end']],dtype=np.int64)
                    if len(p): observer.rows.setdefault((layer,ordinal),[]).append((p,np.ones((len(p),3),np.float32)))
        return observer

    def test_yoco_empty_document_is_explicitly_unavailable_and_exact_coverage_remains_strict(self):
        report=self.diagnostic_fixture().finish()
        self.assertEqual(report['schema'],'early-token-diagnostics-natural-boundary-v1')
        self.assertEqual(len(report['records']),72)
        empty=[r for r in report['records'] if not r['available']]
        self.assertTrue(empty)
        for record in empty:
            self.assertIn(record['layer'],(20,39)); self.assertEqual(record['expected_rows'],0)
            self.assertIsNone(record['summary']); self.assertIsNone(record['native_row_groups'])
            self.assertEqual(record['unavailable_reason'],'no_native_surviving_rows')
            self.assertEqual(unpack_rows(record['scalars'])[0].size,0)
        for record in report['records']:
            if record['available']:
                self.assertEqual(record['summary']['possible_rows'],record['document_tokens'])
                self.assertIsNone(record['unavailable_reason'])
        self.assertTrue(any(not value['available'] and value['common_rows']==0
                            for value in report['early_late_rank_comparisons']))
        bad=self.diagnostic_fixture(); del bad.rows[3,0]
        with self.assertRaisesRegex(ValidationError,'missing diagnostic'): bad.finish()
        bad=self.diagnostic_fixture(); bad.rows[20,0]=[(np.asarray([0]),np.ones((1,3)))]
        with self.assertRaisesRegex(ValidationError,'position coverage'): bad.finish()


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'CPU Torch required; remote gate must run unskipped')
class NaturalBoundaryTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch

    def test_variable_capture_and_canonical_cross_document_chunk_map(self):
        t=self.t; c=controller(); c.t=t
        c.ops=NS(allocate=lambda rows,like:like.new_empty((rows,512)))
        docs,total=geometry((301,901,1701,2377,2903,1109,3401,10003)); c._bind_document_lengths(docs)
        for d in docs:
            length=d['end']-d['start']; c.docs=[dict(d,start=0,end=length)]; c.ids=tuple(range(length))
            c.capture_coverage={3:0}
            for start in range(0,length,8192):
                end=min(start+8192,length)
                values=t.full((end-start,512),float(d['start']),dtype=t.bfloat16)
                c._capture(3,dict(start=start,end=end),values)
            self.assertEqual(tuple(c._entry(d['key'],3)['tensor'].shape),(length,512))
        c.docs=docs
        for start in range(0,total,8192):
            end=min(start+8192,total); positions=t.arange(start,end,dtype=t.int32)
            actual=c._canonical(3,dict(start=start,end=end),positions)
            expected=t.zeros((end-start,512),dtype=t.bfloat16)
            for d in docs:
                lo,hi=max(start,d['start']),min(end,d['end'])
                if lo<hi: expected[lo-start:hi-start].fill_(float(d['start']))
            self.assertTrue(t.equal(actual,expected))
            compact=positions[-1:]
            self.assertTrue(t.equal(c._canonical(3,dict(start=start,end=end),compact),expected[-1:]))
        entry=c.bank[docs[0]['key'],3]; entry['scope']['source_document_tokens']+=1
        with self.assertRaises(ValidationError): c._entry(docs[0]['key'],3)

    def test_variable_short_static_tail_assembly_matches_absolute_position_reference(self):
        t=self.t;c=controller();c.t=t;c.mode='reference_sparse';c.verify_sparse=False
        docs,total=geometry((1,127,128,129,255,256,257,513),gap=1)
        c.docs=docs;c._bind_document_lengths(docs);c._set_boundary_positions(docs)
        tails=t.tensor(c.tail_positions_cpu,dtype=t.bfloat16)[:,None,None].expand(-1,1,512).clone()
        nondoc=[p for p in range(total) if not any(d['start']<=p<d['end'] for d in docs)]
        other=t.tensor(nondoc,dtype=t.bfloat16)[:,None,None].expand(-1,1,512).clone()
        c.ops=NS(identity_error=lambda a,b:self.assertTrue(t.equal(a,b)))
        event=static_plan(docs,list(range(total)),0,total,total,available_nondoc_positions=nondoc)
        event['layer']=3
        actual=c._assemble_memory(event,tails,nondoc,other)
        expected=t.tensor(event['memory_positions'],dtype=t.bfloat16)[:,None,None].expand(-1,1,512)
        self.assertTrue(t.equal(actual,expected))
        self.assertEqual(c.memory_assembly_events[-1]['checks'],1)

    def test_variable_boundary_restore_deduplicates_and_preserves_canonical_bank(self):
        t=self.t;c=controller();c.t=t;c.mode='reference_sparse';c.verify_sparse=False
        docs,total=geometry((1,127,128,129,255,256,257,4093));c.docs=docs
        c._bind_document_lengths(docs);c._set_boundary_positions(docs)
        for d in docs:
            length=d['end']-d['start']
            value=t.arange(length,dtype=t.float32).to(t.bfloat16)[:,None].expand(-1,512).clone()
            c.bank[d['key'],3]=dict(tensor=value,complete=True,version=value._version,scope=c._entry_scope(d['key'],3))
        checks=[]
        def postprocess(mod,value,meta):
            result=value.clone().reshape(-1,1,512)
            result[:,:,0]=meta.positions.to(t.bfloat16)[:,None]
            return result,None
        c.ops=NS(postprocess=postprocess,identity_error=lambda a,b:checks.append(t.equal(a,b)))
        meta=NS(positions=t.arange(1,dtype=t.int32),total_q=1)
        before={key:entry['tensor'].clone() for key,entry in c.bank.items()}
        result=c._restored_boundaries(NS(layer_idx=3),meta)
        self.assertEqual(len(result),sum(min(256,d['end']-d['start']) for d in docs))
        self.assertTrue(t.equal(result[:,0,0],t.tensor(c.boundary_positions_cpu,dtype=t.bfloat16)))
        self.assertEqual(checks,[True])
        for key,expected in before.items(): self.assertTrue(t.equal(c._entry(*key)['tensor'],expected))

    def test_variable_static_tail_restore_uses_each_document_length_and_absolute_rope(self):
        from tests.test_headtail_comparison import ComparisonTorchTests
        c,mod,meta,entries,calls=ComparisonTorchTests.restore_fixture(self)
        c.natural_boundary_inputs=True
        c.docs,_=geometry((1,127,128,129,255,256,257,4093))
        c._bind_document_lengths(c.docs);c._set_boundary_positions(c.docs)
        entries.clear()
        for d in c.docs:
            length=d['end']-d['start']
            entries[d['key']]=dict(tensor=(self.t.arange(length)%31).to(self.t.bfloat16)[:,None].expand(-1,512))
        versions={key:entry['tensor']._version for key,entry in entries.items()}
        result=c._restored_tails(mod,meta)
        expected=self.t.cat([entry['tensor'][-128:] for entry in entries.values()])
        expected=(expected+(self.t.tensor(c.tail_positions_cpu)%7)[:,None]).reshape(-1,1,512)
        self.assertTrue(self.t.equal(result,expected))
        event=c.gpu_tail_restore_events[0]
        self.assertEqual(event['source_rows_per_document'],[d['end']-d['start'] for d in c.docs])
        self.assertEqual(event['source_local_start'],[max(0,d['end']-d['start']-128) for d in c.docs])
        self.assertEqual(event['rows'],len(c.tail_positions_cpu))
        self.assertEqual(versions,{key:entry['tensor']._version for key,entry in entries.items()})

    def test_actual_headtail_forward_uses_variable_tail_offset_and_same_policy_dense_parity(self):
        from tests.test_headtail_link_controller import HeadTailTorchTests
        outputs=[]
        for mode in ('sensitive','reference_sparse'):
            c,mod,meta,cache,cross,hidden,calls,modules=HeadTailTorchTests.forward_fixture(self,mode)
            c.natural_boundary_inputs=True
            c.docs,_=geometry((1,4321,6301,4901,7701,9462,8903,7109),gap=0)
            c._bind_document_lengths(c.docs);c._set_boundary_positions(c.docs)
            boundary=self.t.full((len(c.boundary_positions_cpu),1,512),44,dtype=self.t.bfloat16)
            c._restored_boundaries=lambda *args:boundary
            with patch.dict(sys.modules,modules):
                outputs.append(c._selected_attention(mod,hidden,cache,meta,hidden+20,cross))
            event=c.headtail_memory_events[0]
            self.assertEqual(event['global_index_offset'],128+meta.total_q+len(c.tail_positions_cpu))
            self.assertEqual(cross.loaded_prefill_kv_main_offset,event['global_index_offset'])
            attention=next(call for call in calls if call[0]=='attention')
            self.assertEqual(attention[3],meta.total_q+len(c.tail_positions_cpu))
            self.assertEqual(event['native_cache_store_rows'],meta.total_q)
            self.assertEqual(c.headtail_swa_events[0]['layer'],3)
        self.assertTrue(self.t.equal(*outputs))


if __name__=='__main__': unittest.main()
