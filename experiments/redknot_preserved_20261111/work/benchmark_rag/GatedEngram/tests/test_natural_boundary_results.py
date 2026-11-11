"""Independent fresh20 reader, static GPU-bank, and lossless diagnostic oracles."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from gpu_pilot import read_natural_boundary_results as r
from kv_cache_engram.core import digest
from kv_cache_engram import early_token_diagnostics as producer
from kv_cache_engram.tail_link_plan import plan as tail_plan
from kv_cache_engram.headtail_link_plan import plan as union_plan,query_key_positions,materialize_indices
from types import SimpleNamespace as NS
from tests.test_natural_boundary_controller import controller
from tests import test_headtail_link_results as head_fixture


def sealed(body):
    value={k:v for k,v in body.items() if k!='sha256'}
    return dict(value,sha256=digest(value))


def spans():
    lengths=(900,1300,1900,2700,600,1700,800,1000)
    out=[];start=5
    for i,n in enumerate(lengths):
        out.append((start,start+n,str(i)*64));start+=n+3+i
    return out


def row():
    return dict(index=2,input_tokens=spans()[-1][1]+45,input_ids_sha256='a'*64,
        documents=[dict(input_start=a,input_end=b,token_ids_sha256=k) for a,b,k in spans()])


def scope(phase,generation=7,bank_generation=2):
    c=controller();c.arm=phase['arm'];c.mode=phase['mode'];c.active_policy=c.arm
    c.phase_generation=generation;c.bank_generation=bank_generation
    c.docs=[dict(start=a,end=b,key=k) for a,b,k in spans()];c._bind_document_lengths(c.docs)
    c.bank={(key,layer):dict(scope=dict(document_sha256=key,layer=layer))
        for _,_,key in spans() for layer in range(40)}
    return c._comparison_fields()


def static_fixture(qualified=False,layer=20):
    from tests.test_tail_link_pilot import ReaderTests
    c,_,phase=ReaderTests().controller_fixture();data=row();total=data['input_tokens']
    phase.update(diagnostic=False,verify_sparse=qualified)
    c.update(scope(phase),schema='tail-link-gpu-bank-controller-v1',head_tokens=0,tail_tokens=128,staging=None,
        full_input_tokens=total,ffn_layers=[layer],static_tail_source='shared_gpu_full_bank_tail128_slice',
        static_swa_projection_reused=True,native_local_window_preserved=False,native_local_mask_preserved=False,
        full_bank_retained_but_static_attention_tail_only=True,tail_bank_source_shapes=[[b-a,512] for a,b,_ in spans()],
        diagnostic=False,early_token_diagnostics=None,tail_restore_identity_events=[],memory_assembly_identity_events=[])
    for name in ('selection','ffn_layouts','swa_events','attention_events','tail_memory_events'):c[name]=[]
    c['adapter']['events']=[];full=selected=0;available=set()
    docs=[dict(start=a,end=b,key=k) for a,b,k in spans()]
    for start in range(0,total,8192):
        end=min(start+8192,total);p=r.head.native_positions_reference(start,end,total,layer)
        available.update(q for q in p if not any(a<=q<b for a,b,_ in spans()))
        event=tail_plan(docs,p,start,end,total,available_nondoc_positions=sorted(available))
        n,k=len(p),event['selected_rows'];full+=n;selected+=k;ids=event['selected_local_rows']
        base=dict(layer=layer,chunk_start=start,chunk_end=end)
        selection=dict(base,start=start,end=end,current_positions=p,selected_local_rows=ids,
            selected_positions=[p[i] for i in ids],full_rows=n,selected_rows=k,plan_sha256=event['plan_sha256'])
        c['selection'].append(selection);c['ffn_layouts'].append(dict(selection))
        c['adapter']['events'].append(dict(base,full_rows=n,selected_rows=k,actual_mlp_rows=k,reference_dense=False,
            plan_sha256=event['plan_sha256'],request_id=phase['record']['request_id'],phase=phase['name'],
            inactive_ffn_output='zero',empty_native_collective=k==0 and n>0))
        non=event['mandatory_nondoc_local_rows'];tails=event['document_tail_local_rows']
        c['swa_events'].append(dict(base,current_positions=p,full_rows=n,document_online_rows=0,
            document_tail_rows=len(tails),non_document_online_rows=len(non),reused_rows=len(tails),
            online_local_rows=non,online_rows=len(non),compact=layer>=20))
        local_index=layer in (8,14,20,24,28,32,36)
        c['attention_events'].append(dict(base,full_rows=n,selected_rows=k,main_q_a_rows=n if local_index else k,index_q_rows=n if local_index else 0,
            main_q_b_rows=k,joint_attention_rows=k,output_projection_rows=k,global_compute_called=True,
            native_joint_attention=True,reference_dense=False,is_global_source=layer in (8,14,20),global_kv_rows=end,
            plan_sha256=event['plan_sha256']))
        event.update(layer=layer,query_positions=[p[i] for i in ids],query_memory_lengths=event['selected_memory_lengths'],
            actual_joint_query_rows=k,gpu_causal_oracle_checked=qualified)
        c['tail_memory_events'].append(event)
        if qualified:c['memory_assembly_identity_events'].append(dict(base,
            schema='tail-link-memory-assembly-identity-v1',rows=event['memory_count'],checks=1,max_absolute_error=0.0))
    c['adapter']['counts']=dict(full_rows=full,selected_rows=selected,actual_mlp_rows=selected,inactive_ffn_zero_rows=full-selected)
    tail_sha=digest([p for _,hi,_ in spans() for p in range(hi-128,hi)])
    names=['gpu_full_bank_tail_slice','absolute_rope_nativecast']
    c['gpu_tail_restore_events']=[dict(layer=layer,rows=1024,source='shared_gpu_full_canonical_bank',
        source_rows_per_document=[b-a for a,b,_ in spans()],source_local_start=[b-a-128 for a,b,_ in spans()],checked=qualified,checks=2 if qualified else 0,
        checks_named=names if qualified else [],absolute_positions_sha256=tail_sha,
        phase_generation=c['phase_generation'],host_to_device_copy=False,max_absolute_error=0.0 if qualified else None)]
    if qualified:c['tail_restore_identity_events']=[dict(layer=layer,rows=1024,checks=2,checks_named=names,
        absolute_positions_sha256=tail_sha,max_absolute_error=0.0)]
    c['decode_ring_positions']=[dict(layer=layer,positions=list(range(total-128,total)),
        physical_readback_exact=True,checks=128,physical_slots=list(range(128)))]
    return c,data,phase,spans()


class StaticOracleTests(unittest.TestCase):
    def test_scalar_set_oracle_and_producer_agree_full_and_yoco_all_chunks(self):
        data=row();docs=[dict(start=a,end=b,key=k) for a,b,k in spans()]
        for layer in (3,20):
            available=set()
            for start in range(0,data['input_tokens'],8192):
                end=min(start+8192,data['input_tokens']);p=r.head.native_positions_reference(start,end,data['input_tokens'],layer)
                available.update(q for q in p if not any(a<=q<b for a,b,_ in spans()))
                actual=r.static_memory_plan(p,start,end,data['input_tokens'],spans(),available)
                self.assertEqual(actual,tail_plan(docs,p,start,end,data['input_tokens'],available_nondoc_positions=sorted(available)))
                memory={q for _,hi,_ in spans() for q in range(hi-128,hi) if q<=p[-1]}|available
                self.assertEqual(actual['memory_positions'],sorted(memory))
                for i in sorted({0,len(p)//2,len(p)-1}):
                    self.assertEqual(actual['reference_memory_lengths'][i],sum(q<=p[i] for q in memory))

    def check(self,*args):
        with patch.object(r,'FFN_LAYERS',tuple(args[0]['ffn_layers'])):return r.audit_controller(*args)

    def test_static_full_gpu_contract_has_real_consumption_without_dynamic_refresh(self):
        result=self.check(*static_fixture())
        self.assertGreater(result['consumed_cached_pairs'],0)
        self.assertGreater(result['ffn_bypassed_rows'],0)
        self.assertEqual(result['actual_partial_chunks'],0)  # This fixture's final rectangle is tail plus suffix.
        self.assertEqual(result['dynamic_refreshed_rows'],0)
        self.assertEqual(result['offline_kv_h2d_bytes'],0)
        encoder=self.check(*static_fixture(layer=3))
        self.assertGreater(encoder['actual_partial_chunks'],0)

    def test_qualified_gpu_slice_and_rope_are_independent_exact_checks(self):
        fixture=static_fixture(True);self.check(*fixture)
        for field,value in [('checks_named',['pinned_cpu_tail_h2d','absolute_rope_nativecast']),
                            ('max_absolute_error',1e-9),('source_rows_per_document',128),('host_to_device_copy',True)]:
            bad=copy.deepcopy(fixture);bad[0]['gpu_tail_restore_events'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(*bad)

    def test_static_cannot_claim_current_hidden_refresh_revival_or_foreign_bank(self):
        mutations=[lambda c:c.update(dynamic_document_swa=True),lambda c:c.update(offline_bank_bytes=40*1024**2),
            lambda c:c.update(cross_phase_overlay_reuse=True),lambda c:c['swa_events'][0].update(document_online_rows=1),
            lambda c:c['selection'][0]['current_positions'].append(0),
            lambda c:c['tail_memory_events'][-1]['memory_doc_token_offsets'].__setitem__(0,0),
            lambda c:c['gpu_tail_restore_events'][0].update(phase_generation=6),
            lambda c:c['decode_ring_positions'][0]['physical_slots'].__setitem__(127,0),
            lambda c:c['attention_events'][0].update(global_compute_called=False)]
        for mutate in mutations:
            fixture=static_fixture();mutate(fixture[0])
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):self.check(*fixture)

    def test_scope_hash_independently_binds_all_documents_and_layers(self):
        identity=dict(checkpoint_index_sha256='a'*64,tokenizer_json_sha256='c'*64,execution_configuration_sha256='b'*64)
        phase=dict(arm='recomputed',mode='native');c=controller();c.arm=phase['arm'];c.mode=phase['mode']
        c.bank_generation=1;c.phase_generation=4
        c.docs=[dict(start=a,end=b,key=k) for a,b,k in spans()];c._bind_document_lengths(c.docs)
        for lo,hi,key in spans():
            for layer in range(40):
                c.bank[key,layer]=dict(scope=dict(model_sha256='a'*64,tokenizer_sha256='c'*64,
                    execution_sha256='b'*64,document_sha256=key,layer=layer,representation='swa-postnorm-prerope-bf16',source_document_tokens=hi-lo,document_length_policy='natural_disjoint_token_spans'))
        report=c._comparison_fields();expected=r.bank_scope_sha256(spans(),identity)
        self.assertEqual(report['offline_bank_scope_sha256'],expected)
        r.comparison_scope(report,phase,expected_scope=expected,spans=spans())
        report['offline_bank_scope_sha256']='f'*64
        with self.assertRaises(ValueError):r.comparison_scope(report,phase,expected_scope=expected,spans=spans())

    def test_full_native_keeps_bank_resident_without_any_sparse_or_diagnostic_work(self):
        data=row();phase=dict(arm='recomputed',mode='native',name='case2.recomputed_quality',fraction=None,
            native_yoco_swa_share_hidden=False,diagnostic=False)
        control=dict(scope(phase),schema='headtail-link-controller-v1',mode='native',phase=phase['name'],
            input_ids_sha256=data['input_ids_sha256'],full_input_tokens=data['input_tokens'],
            head_tokens=0,tail_tokens=0,dynamic_refresh_start_layer=None,dynamic_document_swa=False,
            native_yoco_swa_share_hidden=False,staging=None,phase_local_boundary_overlay=False,
            boundary_overlay_rows_per_layer=0,projection_identity_checks=0,cache_restore_identity_checks=0,
            diagnostic=False,early_token_diagnostics=None,adapter=dict(events=[],counts=dict(full_rows=0,selected_rows=0,actual_mlp_rows=0)))
        for name in ('selection','index_groups','swa_events','ffn_layouts','attention_events',
                     'projection_identity_events','cache_restore_identity_events','headtail_swa_events',
                     'headtail_memory_events','union_oracle_events','boundary_restore_identity_events','decode_ring_positions'):
            control[name]=[]
        result=r.audit_controller(control,data,phase,spans())
        self.assertFalse(result['actual_rows_instrumented'])
        self.assertEqual(result['ffn_selected_rows'],result['ffn_full_rows'])
        self.assertEqual(result['consumed_cached_pairs'],0)
        for update in (dict(offline_bank_entry_count=0),dict(cached_document_kv_consumed=True),
                       dict(early_token_diagnostics={'records':[]})):
            with self.assertRaises(ValueError):r.audit_controller(dict(control,**update),data,phase,spans())


class DiagnosticScalarTests(unittest.TestCase):
    def test_real_producer_bytes_statistics_groups_and_ranks_recomputed(self):
        rng=np.random.default_rng(1907)
        for n,layer in ((900,3),(140,20),(1,39),(513,19)):
            p=np.arange(1000-n,1000,dtype=np.uint32)
            v=np.column_stack((rng.integers(0,9,n),rng.integers(1,5,n),rng.random(n)*2)).astype(np.float32)
            decoded=r.unpack_diagnostic_scalars(producer.pack_rows(p,v),p.tolist())
            np.testing.assert_array_equal(decoded[1],v)
            r.compare_statistics(producer.scalar_summary(p,v,1000),r.diagnostic_summary(*decoded,1000))
            r.compare_statistics(producer.native_row_groups(p,v,document_start=229381,total_tokens=262194,layer=layer),
                r.diagnostic_row_groups(p,v,229381,layer,262194))
            r.compare_statistics(producer.rank_comparison(p,v,p,v[::-1].copy()),
                {k:v for k,v in r.diagnostic_rank_comparison(p,v,p,v[::-1].copy()).items() if k!='available'})

    def test_zero_energy_ties_and_empty_groups_keep_null_mean(self):
        p=np.arange(4,dtype=np.uint32);v=np.zeros((4,3),np.float32)
        result=r.diagnostic_summary(p,v,900)
        self.assertIsNone(result['boundaries']['tail128']['mean_relative_l2'])
        self.assertIsNone(result['boundaries']['head128']['error_energy_fraction'])
        ranks=r.diagnostic_rank_comparison(p,v,p,v)
        self.assertIsNone(ranks['spearman']);self.assertEqual(ranks['top256_overlap_fraction'],1)
        r.compare_statistics(producer.scalar_summary(p,v,900),result)

    def test_lossless_stream_rejects_checksum_padding_bomb_positions_and_nonfinite(self):
        p=np.arange(6,dtype=np.uint32);v=np.ones((6,3),np.float32);good=producer.pack_rows(p,v)
        mutations=[lambda b:b.update(raw_sha256='f'*64),lambda b:b.update(rows=7),
            lambda b:b.update(data=b['data']+'!'),
            lambda b:b.update(data=base64.b64encode(base64.b64decode(b['data'])+b'junk').decode()),
            lambda b:b.update(data=base64.b64encode(zlib.compress(b'0'*100000)).decode())]
        for mutate in mutations:
            bad=copy.deepcopy(good);mutate(bad)
            with self.assertRaises(ValueError):r.unpack_diagnostic_scalars(bad,p.tolist())
        for values in (np.full((6,3),np.nan,np.float32),np.full((6,3),-1,np.float32),np.full((6,3),3,np.float32)):
            with self.assertRaises(ValueError):r.unpack_diagnostic_scalars(producer.pack_rows(p,values),p.tolist())
        with self.assertRaises(ValueError):r.unpack_diagnostic_scalars(good,[0,1,2,3,4,7])
        with self.assertRaises(ValueError):r.compare_statistics({'energy':1.01},{'energy':1.0})

    def routing_fixture(self):
        data=row();pin=r.natural_boundary_protocol()['diagnostics']
        preflight=dict(protocol=dict(diagnostics=pin),identity=dict(checkpoint_config_sha256=pin['checkpoint_config_sha256'],
            runtime_source_files_sha256=pin['native_source_files_sha256']))
        suffix=spans()[-1][1];start=(suffix//8192)*8192;end=data['input_tokens'];queries=list(range(suffix,end))
        event=dict(layer=2,chunk_start=start,chunk_end=end,query_absolute_positions=queries,
            question_suffix_start=suffix,kv_compress_ratio=2,selection_kind='sparse_topk',
            coordinate_system='loaded_compressed_cache_rows_single_request',invalid_index=-1,
            token_interval='[index*ratio,(index+1)*ratio)',context_parallel_size=1,num_reqs=1,
            native_compressed_key_starts=[0]*len(queries),native_compressed_key_ends=[(q+1)//2 for q in queries],
            interpretation='native_compressed_block_routing_proxy_not_attention_weights',
            includes_question_suffix_template_tokens=True,native_global_indices=[[0]+[-1]*511 for q in queries])
        report=dict(question_routing_layers=[2],question_routing_compress_ratios={'2':2},
            question_routing_selection_kinds={'2':'sparse_topk'},question_routing_events=[event])
        return report,data,preflight

    def test_routing_binds_actual_prefix_topology_and_compressed_native_bounds(self):
        fixture=self.routing_fixture();self.assertTrue(r.audit_question_routing(*fixture))
        mutations=[lambda x:x['question_routing_events'][0]['native_global_indices'][0].__setitem__(0,262194),
            lambda x:x['question_routing_events'][0]['native_global_indices'][0].pop(),
            lambda x:x.update(question_routing_layers=[0,1,2]),
            lambda x:x['question_routing_events'][0].update(selection_kind='dense_causal_enumeration'),
            lambda x:x['question_routing_events'][0]['native_compressed_key_starts'].__setitem__(0,1)]
        for mutate in mutations:
            bad=copy.deepcopy(fixture);mutate(bad[0])
            with self.assertRaises(ValueError):r.audit_question_routing(*bad)
        bad=copy.deepcopy(fixture);bad[2]['identity']['checkpoint_config_sha256']='a'*64
        with self.assertRaises(ValueError):r.audit_question_routing(*bad)

    def test_full72record_report_recomputed_compacted_and_missing_coverage_rejected(self):
        routing,data,preflight=self.routing_fixture();phase=dict(name='case2.diagnostic_recomputed')
        records=[];values_by_key={}
        for layer in r.DIAGNOSTIC_LAYERS:
            all_p=[p for s in range(0,data['input_tokens'],8192)
                for p in r.head.native_positions_reference(s,min(s+8192,data['input_tokens']),data['input_tokens'],layer)]
            for ordinal,(lo,hi,key) in enumerate(spans()):
                p=np.asarray([q-lo for q in all_p if lo<=q<hi],dtype=np.uint32)
                v=np.column_stack((p%7,np.ones(len(p))*3,np.ones(len(p))*.25)).astype(np.float32)
                values_by_key[layer,ordinal]=p,v
                records.append(dict(layer=layer,document_ordinal=ordinal,document_sha256=key,document_start=lo,
                    document_end=hi,native_yoco_rectangle_lo=None if layer<20 else data['input_tokens']-140,
                    document_tokens=hi-lo,expected_rows=len(p),available=bool(len(p)),
                    unavailable_reason=None if len(p) else 'no_native_surviving_rows',
                    native_row_groups=producer.native_row_groups(p,v,document_start=lo,total_tokens=data['input_tokens'],layer=layer) if len(p) else None,
                    summary=producer.scalar_summary(p,v,hi-lo) if len(p) else None,scalars=producer.pack_rows(p,v)))
        correlations=[dict(early_layer=a,late_layer=b,document_ordinal=i,
            available=len(np.intersect1d(values_by_key[a,i][0],values_by_key[b,i][0]))>=2,
            **producer.rank_comparison(*values_by_key[a,i],*values_by_key[b,i]))
            for a in (1,2,3) for b in (7,13,19,20,39) for i in range(8)]
        report=sealed(dict(routing,schema='early-token-diagnostics-natural-boundary-v1',complete=True,phase=phase['name'],
            input_ids_sha256=data['input_ids_sha256'],mode='native',arm='recomputed',included_in_formal_timing=False,
            representation='swa-postnorm-prerope-bf16',layers=list(r.DIAGNOSTIC_LAYERS),
            metrics=['error_l2_squared','reference_l2_squared','cosine_distance'],full_kv_vectors_exported=False,
            attention_weights_collected=False,only_actual_native_surviving_rows=True,offline_bank_mutated=False,
            natural_boundary_inputs=True,document_token_lengths=[hi-lo for lo,hi,_ in spans()],
            bin_edges=list(r.DIAGNOSTIC_EDGES),bin_edges_policy='clip_to_document_length_include_tail128',native_position_coverage_checked=True,prompt_chunk_tokens=8192,
            native_yoco_keep_tokens=140,records=records,early_late_rank_comparisons=correlations,limitations=[]))
        self._last_report_fixture=(report,data,phase,preflight)
        compact=r.audit_observer_report(report,data,phase,preflight)
        self.assertEqual(len(compact['records']),72)
        self.assertEqual(compact['original_observer_sha256'],report['sha256'])
        self.assertNotIn('data',compact['records'][0]['scalar_evidence'])
        self.assertNotIn('scalars',compact['records'][0])
        self.assertEqual(compact['records'][-1]['native_row_groups']['full_layout']['rows'],0)
        for mutate in (lambda x:x['records'].pop(),
            lambda x:x['records'][0].update(document_sha256='f'*64),
            lambda x:x['records'][0]['summary'].update(error_energy=999),
            lambda x:x['records'][0].update(native_yoco_rectangle_lo=0)):
            bad=copy.deepcopy(report);mutate(bad)
            with self.assertRaises(ValueError):r.audit_observer_report(sealed(bad),data,phase,preflight)


class PairedClosureTests(unittest.TestCase):
    def test_three_fresh_samples_all_arms_rebuild_median_and_pair_ratios(self):
        samples={'recomputed':[90.,60.,80.],'tail128_link':[30.,30.,40.],'headtail128_link':[45.,40.,50.]}
        result=r.expected_timing_summary(samples)
        self.assertEqual(result['arms']['recomputed']['online_ttft_p50_s'],80)
        self.assertEqual(result['comparisons']['tail128_link']['ttft_speedup'],80/30)
        self.assertEqual(result['comparisons']['headtail128_link']['paired_repeat_speedups'],[2,1.5,1.6])
        self.assertTrue(result['fresh_paired_baseline'])
        for bad in ({**samples,'recomputed':[90.]},{**samples,'tail128_link':[float('nan')]*3}):
            with self.assertRaises(ValueError):r.expected_timing_summary(bad)

    def test_pooled_median_uses_raw_samples_and_row_fraction_is_not_flops(self):
        cases=[]
        for samples in ([1.,2.,100.],[3.,4.,5.]):
            cases.append(dict(online_ttft_samples_s={arm:[x/(i+1) for x in samples] for i,arm in enumerate(r.ARMS)},
                quality={arm:dict(answer_em=0,answer_token_f1=.5,truncated=False) for arm in r.ARMS},
                actual_rows_and_swa_reuse={arm:dict(ffn_full_rows=100,ffn_selected_rows=100 if i==0 else 10,
                    attention_full_rows=100,attention_selected_rows=100 if i==0 else 10,
                    dynamic_refreshed_rows=50 if i==2 else 0,boundary_slot_capacity=100)
                    for i,arm in enumerate(r.ARMS)}))
        result=r.summarize_cases(cases)
        self.assertEqual(result['arms']['recomputed']['online_ttft_p50_s'],3.5)
        self.assertEqual(result['comparisons']['tail128_link']['ttft_speedup'],2)
        self.assertAlmostEqual(result['headtail_vs_static']['ttft_speedup'],1.5)
        self.assertEqual(result['actual_rates_by_arm']['headtail128_link']['dynamic_refreshed_boundary_fraction'],.5)
        self.assertTrue(result['actual_rates_by_arm']['headtail128_link']['not_model_flops_fraction'])
        self.assertFalse(result['actual_rates_by_arm']['recomputed']['actual_rows_instrumented'])

    def closure_fixture(self,directory):
        preflight=dict(identity={'run':'new20'},manifest_sha256='a'*64);qual={'sha256':'b'*64}
        smokes={arm:dict(sha256=str(i+1)*64) for i,arm in enumerate(r.POLICY_ARMS)}
        closures=[];evidence={};refs=[];diags=[]
        for index in range(2,22):
            diagnostic=dict(file=f'case{index}.diagnostic_recomputed.json',file_sha256=str(index%10)*64,seal_sha256='d'*64)
            c=sealed(dict(index=index,shared_bank_identity=dict(bank_generation=index),diagnostic_recomputed=diagnostic))
            p=Path(directory)/f'case{index}.paired.json';p.write_text(json.dumps(c))
            loaded=r.read(p,evidence);closures.append(loaded)
            refs.append(dict(index=index,file=p.name,file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seal_sha256=c['sha256']))
            diags.append(dict(index=index,**diagnostic))
        run=sealed(dict(schema=r.RUN_SCHEMA,status='complete',completed_case_indices=list(range(2,22)),
            identity=preflight['identity'],manifest_sha256=preflight['manifest_sha256'],qualification=qual,
            partial_smoke_sha256={arm:smokes[arm]['sha256'] for arm in r.POLICY_ARMS},
            diagnostic_excluded_from_formal_metrics=True,cases=refs,diagnostics=diags))
        return run,preflight,qual,smokes,closures,[],directory,evidence

    def test_full20_closure_binds_each_fresh_case_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.closure_fixture(directory);self.assertTrue(r.validate_closed_run(*args))
            for mutate in (lambda x:x['completed_case_indices'].pop(),lambda x:x['diagnostics'].pop(),
                lambda x:x['diagnostics'][1].update(file='historical.json'),lambda x:x['cases'][0].update(file_sha256='f'*64)):
                bad=list(copy.deepcopy(args));mutate(bad[0]);bad[0]=sealed(bad[0])
                with self.assertRaises(ValueError):r.validate_closed_run(*bad)
            bad=list(args);bad[5]=['raw case failed']
            with self.assertRaises(ValueError):r.validate_closed_run(*bad)

    def test_diagnostic_output_and_order_cannot_enter_formal_metrics(self):
        phase=dict(name='case2.diagnostic_recomputed',repeat=None,arm='recomputed',mode='native',sha256='a'*64,
            record=dict(output_ids=[7]),controller=dict(phase_generation=18,early_token_diagnostics={'schema':'early-token-diagnostics-v1'}))
        with patch.object(r,'empty_ep_binding'),patch.object(r,'audit_phase') as phase_check,\
             patch.object(r,'comparison_scope',return_value={'bank':1}),patch.object(r,'audit_observer_report',return_value={'records':[]}):
            result=r.audit_diagnostic(phase,row(),{},'qual',7,{'bank':1},set(range(3,18)))
            self.assertTrue(result['excluded_from_formal_ttft_and_quality'])
            self.assertEqual(phase_check.call_args.kwargs,dict(arm='recomputed',timing=False,quality=False,fixed_tokens=1,diagnostic=True))
            for mutate in (lambda x:x['record'].update(output_ids=[8]),lambda x:x['controller'].update(phase_generation=4)):
                bad=copy.deepcopy(phase);mutate(bad)
                with self.assertRaises(ValueError):r.audit_diagnostic(bad,row(),{},'qual',7,{'bank':1},set(range(3,18)))


def headtail_fixture():
    c=controller();c.native_yoco_swa_share_hidden=False;c.phase_generation=7
    spans=globals()['spans']();row=globals()['row']();total=row['input_tokens']
    c.docs=[dict(start=a,end=b,key=k) for a,b,k in spans];c._bind_document_lengths(c.docs);c._set_boundary_positions(c.docs)
    c.bank={(key,layer):dict(scope=c._entry_scope(key,layer)) for _,_,key in spans for layer in range(40)}
    phase=dict(arm=r.ARM,mode='sensitive',name='case2.headtail128_link_quality',fraction=None,
        dynamic_document_swa=True,native_yoco_swa_share_hidden=False,verify_sparse=False,
        diagnostic=False,record=dict(request_id='natural-boundary:2:quality:sample'))
    control=dict(c._report_fields(True),**scope(phase),diagnostic=False,early_token_diagnostics=None,mode=phase['mode'],phase=phase['name'],
        input_ids_sha256=row['input_ids_sha256'],full_input_tokens=total,ffn_layers=[20],
        first_three_full=True,actual_full_row_native_mhc=True,yoco_layout_preserved=True,
        global_kv_native_online=True,decode_native=True,formal_admission_unchanged=True,
        not_model_flops_fraction=True,selected_tokens_forced_through_yoco=False,
        selection=[],ffn_layouts=[],swa_events=[],attention_events=[])
    adapter=dict(schema='kv-cache-engram-index-hit20-adapter-v1',decoder_forward_unchanged=True,
        attention_forward_unchanged=True,native_mhc_unchanged=True,native_yoco_layout_unchanged=True,
        all_selected_delegates_native=True,empty_selection_skips_ep_collective=False,
        cached_hidden_or_ffn_output=False,accuracy_guarantee=False,speedup_claim=False,events=[])
    previous=set();full=selected=0
    for start in range(0,total,8192):
        end=min(start+8192,total);positions=r.head.native_positions_reference(start,end,total,20)
        event=union_plan([dict(start=a,end=b,key=k) for a,b,k in spans],positions,start,end,total,layer=20)
        ids=event['selected_local_rows'];n=len(positions);k=len(ids);full+=n;selected+=k
        base=dict(chunk_start=start,chunk_end=end,layer=20)
        selection=dict(start=start,end=end,layer=20,current_positions=positions,selected_local_rows=ids,
            selected_positions=[positions[i] for i in ids],full_rows=n,selected_rows=k,plan_sha256=event['plan_sha256'])
        control['selection'].append(selection);control['ffn_layouts'].append(dict(selection))
        adapter['events'].append(dict(base,full_rows=n,selected_rows=k,actual_mlp_rows=k,reference_dense=False,
            plan_sha256=event['plan_sha256'],request_id=phase['record']['request_id'],phase=phase['name'],
            inactive_ffn_output='zero',empty_native_collective=k==0 and n>0))
        control['attention_events'].append(dict(base,full_rows=n,selected_rows=k,main_q_a_rows=n,index_q_rows=n,
            main_q_b_rows=k,joint_attention_rows=k,output_projection_rows=k,inactive_attention_zero_rows=n-k,
            global_compute_called=True,native_joint_attention=True,is_global_source=True,reference_dense=False,
            native_local_window_preserved=True,extra_completed_document_tail_links=True,
            plan_sha256=event['plan_sha256'],global_kv_rows=0))
        control['swa_events'].append(dict(base,current_positions=positions,online_local_rows=ids,online_rows=k,
            reused_rows=n-k,full_rows=n,compact=True,native_cache_store_rows=n,
            head_rows=len(event['document_head_local_rows']),tail_rows=len(event['document_tail_local_rows']),
            global_index_offset=event['global_index_offset']))
        event.update(query_positions=[positions[i] for i in ids],actual_joint_query_rows=k,reference_dense=False,
            native_cache_store_rows=n,global_source_is_native=True,
            tail_slots_absolute_positions_sha256=digest(event['tail_bank_positions']),
            gpu_native_window_checked=True,gpu_union_oracle_checked=False,
            native_valid_start_binding='native_indices_verified_zero_start')
        control['headtail_memory_events'].append(event)
        e=head_fixture.refresh(positions,spans,20,start,end,previous,qualified=False)
        previous.update(e['refreshed_positions']);control['headtail_swa_events'].append(e)
    adapter['counts']=dict(full_rows=full,selected_rows=selected,actual_mlp_rows=selected,inactive_ffn_zero_rows=full-selected)
    control['adapter']=adapter
    control['boundary_restore_identity_events']=[dict(layer=20,rows=2048,checked=False,checks=0,
        absolute_positions_sha256=digest(r.boundary_positions(spans)),max_absolute_error=None)]
    control['decode_ring_positions']=[dict(layer=20,positions=list(range(total-128,total)),
        physical_readback_exact=True,checks=128,physical_slots=list(range(128)))]
    return control,row,phase,spans

class UnlimitedQualityTests(unittest.TestCase):
    def fixture(self,tokens=200,finish='stop',input_tokens=256):
        policy=dict(max_tokens=None,stop_policy='natural_eos_or_engine_context_limit',
            engine_max_model_len=1000,model_max_position_embeddings=1048576)
        budget=1000-input_tokens
        phase=dict(quality_requested_max_new_tokens=None,technical_max_new_tokens=budget)
        record=dict(output_ids=list(range(tokens)),natural_EOS=True,technical_max_new_tokens=budget,
            finish_reason=finish,truncated=finish=='length',quality_output_complete=finish=='stop',
            quality_limit_reason='engine_context_limit' if finish=='length' else None)
        return record,phase,dict(input_tokens=input_tokens),policy

    def test_natural_eos_beyond_old_cap_and_exact_engine_limit_are_distinct(self):
        result=r.audit_quality_completion(*self.fixture())
        self.assertTrue(result['quality_output_complete']);self.assertIsNone(result['quality_limit_reason'])
        result=r.audit_quality_completion(*self.fixture(tokens=744,finish='length'))
        self.assertFalse(result['quality_output_complete']);self.assertEqual(result['quality_limit_reason'],'engine_context_limit')
        self.assertEqual(result['technical_max_new_tokens'],744)
        # A short budget is valid only when the input actually fills the engine context.
        self.assertEqual(r.audit_quality_completion(*self.fixture(tokens=64,finish='length',input_tokens=936))['technical_max_new_tokens'],64)

    def test_old_cap_early_length_stop_and_forged_technical_scope_are_rejected(self):
        with self.assertRaises(ValueError):r.audit_quality_completion(*self.fixture(tokens=64,finish='length'))
        with self.assertRaises(ValueError):r.audit_quality_completion(*self.fixture(tokens=745))
        for target,field,value in ((1,'quality_requested_max_new_tokens',64),(1,'technical_max_new_tokens',64),
            (0,'technical_max_new_tokens',64),(0,'quality_output_complete',False),(0,'quality_limit_reason','engine_context_limit'),
            (0,'truncated',True),(0,'natural_EOS',False),(3,'max_tokens',64),(3,'engine_max_model_len',2000000)):
            fixture=list(self.fixture());fixture[target][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.audit_quality_completion(*fixture)
        fixture=list(self.fixture());del fixture[1]['quality_requested_max_new_tokens']
        with self.assertRaises(ValueError):r.audit_quality_completion(*fixture)


class VariableGeometryTests(unittest.TestCase):
    def test_source_spans_preserve_gaps_and_reject_overlap_length_and_hash_alias(self):
        data=row();self.assertEqual(r.documents(data),spans())
        geometry=r.case_geometry(data)
        self.assertEqual(geometry['layer_chunk_checks'],80)
        self.assertEqual(geometry['offline_bank_bytes'],40*sum(b-a for a,b,_ in spans())*1024)
        self.assertGreater(geometry['nondocument_tokens'],45)
        for mutate in (lambda x:x['documents'][1].update(input_start=x['documents'][0]['input_end']-1),
                       lambda x:x['documents'][1].update(token_ids_sha256=x['documents'][0]['token_ids_sha256']),
                       lambda x:x['documents'][1].update(tokens=32768),
                       lambda x:x['documents'][-1].update(input_end=x['input_tokens']+1)):
            bad=copy.deepcopy(data);mutate(bad)
            with self.assertRaises(ValueError):r.documents(bad)

    def test_short_blocks_and_gaps_union_match_independent_set_and_real_producer(self):
        start=7;docs=[]
        for i,n in enumerate((1,127,128,129,255,256,257,913)):
            docs.append((start,start+n,str(i)*64));start+=n+3+i
        total=start+45
        boundaries=sorted({p for a,b,_ in docs for p in list(range(a,min(a+128,b)))+list(range(max(a,b-128),b))})
        self.assertEqual(r.boundary_positions(docs),boundaries)
        for layer in (3,19,20,39):
            positions=r.head.native_positions_reference(0,total,total,layer)
            produced=union_plan([dict(start=a,end=b,key=k) for a,b,k in docs],positions,0,total,total,layer=layer)
            expected=r.expected_plan(positions,0,total,total,layer,docs)
            self.assertEqual(produced,expected)
            for i,q in enumerate(positions):
                keys=set(range(max(0,q-127,total-140 if layer>=20 else 0),q+1))
                for a,b,_ in docs:
                    if b<=q:keys.update(range(max(a,b-128),b))
                self.assertEqual(set(query_key_positions(produced,q)),keys)
                self.assertEqual(expected['union_lengths'][i],len(keys))
            r.head.audit_observed_indices(expected,positions[::31],materialize_indices(produced,positions[::31]))
            available={p for p in positions if not any(a<=p<b for a,b,_ in docs)}
            self.assertEqual(r.static_memory_plan(positions,0,total,total,docs,available),
                tail_plan([dict(start=a,end=b,key=k) for a,b,k in docs],positions,0,total,total,
                          available_nondoc_positions=sorted(available)))

    def test_variable_canonical_payload_requires_actual_length_finite_and_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'canonical.bf16';payload=b'\0'*(257*512*2);p.write_bytes(payload)
            sha=hashlib.sha256(payload).hexdigest()
            self.assertEqual(r.audit_bank_payload(p,sha,257)['bytes'],len(payload))
            for length in (128,256,32768):
                with self.assertRaises(ValueError):r.audit_bank_payload(p,sha,length)
            with self.assertRaises(ValueError):r.audit_bank_payload(p,'f'*64,257)
            link=Path(directory)/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):r.audit_bank_payload(link,sha,257)
            p.write_bytes(b'\x80\x7f'+payload[2:])
            with self.assertRaisesRegex(ValueError,'nonfinite'):r.audit_bank_payload(p,sha,257)

    def test_real_headtail_report_variable_union_refresh_and_offline_fallback(self):
        fixture=headtail_fixture()
        with patch.object(r,'FFN_LAYERS',(20,)):
            result=r.audit_controller(*fixture)
        self.assertGreater(result['dynamic_refreshed_rows'],0)
        self.assertGreater(result['consumed_cached_pairs'],0)
        self.assertGreater(result['unrefreshed_boundary_slots'],0)
        self.assertLess(result['dynamic_refreshed_boundary_fraction'],1)
        self.assertEqual(result['boundary_slot_capacity'],2048)
        for mutate in (lambda c:c.update(offline_bank_rows=40*8*32768),
                       lambda c:c.update(boundary_overlay_rows_per_layer=1024),
                       lambda c:c['headtail_swa_events'][0].update(source='previous_layer_kv'),
                       lambda c:c['headtail_memory_events'][-1].update(global_index_offset=999),
                       lambda c:c['headtail_swa_events'][-1].update(available_refreshed_positions=[]),
                       lambda c:c['headtail_memory_events'][0].update(native_local_mask_preserved=False)):
            bad=copy.deepcopy(fixture);mutate(bad[0])
            with patch.object(r,'FFN_LAYERS',(20,)),self.assertRaises(ValueError):r.audit_controller(*bad)


class AbsentDiagnosticTests(unittest.TestCase):
    def test_empty_scalars_are_valid_but_not_zero_error_observations(self):
        p=np.empty(0,np.uint32);v=np.empty((0,3),np.float32)
        decoded=r.unpack_diagnostic_scalars(producer.pack_rows(p,v),[])
        self.assertEqual(decoded[1].shape,(0,3))
        self.assertIsNone(r.diagnostic_summary(*decoded,900))
        self.assertIsNone(r.diagnostic_row_groups(*decoded,5,20,11000))
        self.assertEqual(r.diagnostic_rank_comparison(*decoded,*decoded),
            dict(available=False,common_rows=0,spearman=None,top256_overlap_fraction=None))
        with self.assertRaises(ValueError):r.unpack_diagnostic_scalars(producer.pack_rows([0],np.zeros((1,3))),[])
        with self.assertRaises(ValueError):r.compare_statistics(dict(error_energy=0),None)

    def test_real_observer_finish_matches_reader_with_empty_docs_and_routing(self):
        routing,data,preflight=DiagnosticScalarTests().routing_fixture()
        docs=[dict(start=a,end=b,key=k) for a,b,k in spans()]
        layers=[NS(layer_idx=i,self_attn=NS(has_packed_kv=i==2,is_sparse=i==2,kv_compress_ratio=2 if i==2 else 0)) for i in range(3)]
        c=NS(mode='native',arm='recomputed',diagnostic=True,natural_boundary_inputs=True,
            docs=docs,layers=layers,ids=range(data['input_tokens']),phase='case2.diagnostic_recomputed',
            input_sha=data['input_ids_sha256'],_entry=lambda key,layer:dict(version=0))
        observer=producer.EarlyTokenDiagnostics(controller=c)
        for layer in r.DIAGNOSTIC_LAYERS:
            for start in range(0,data['input_tokens'],8192):
                end=min(start+8192,data['input_tokens']);observer.chunks.add((layer,start,end))
                positions=r.head.native_positions_reference(start,end,data['input_tokens'],layer)
                for ordinal,(lo,hi,_) in enumerate(spans()):
                    p=np.asarray([q-lo for q in positions if lo<=q<hi],np.uint32)
                    if len(p):observer.rows.setdefault((layer,ordinal),[]).append((p,np.ones((len(p),3),np.float32)))
        observer.index_events=routing['question_routing_events'];report=observer.finish()
        compact=r.audit_observer_report(report,data,dict(name=c.phase),preflight)
        empty=[x for x in compact['records'] if not x['available']]
        self.assertGreater(len(empty),0)
        self.assertTrue(all(x['summary'] is None and x['native_row_groups'] is None for x in empty))
        index=next(i for i,x in enumerate(report['records']) if not x['available'])
        rank_index=next(i for i,x in enumerate(report['early_late_rank_comparisons']) if not x['available'])
        for mutate in (lambda x:x['records'][index].update(available=True),
                       lambda x:x['records'][index].update(summary=dict(error_energy=0)),
                       lambda x:x['records'][index].update(unavailable_reason=None),
                       lambda x:x['records'][index].update(document_tokens=32768),
                       lambda x:x['records'][index].update(expected_rows=1),
                       lambda x:x.update(bin_edges_policy='fixed32768'),
                       lambda x:x['early_late_rank_comparisons'][rank_index].update(available=True)):
            bad=copy.deepcopy(report);mutate(bad)
            with self.assertRaises(ValueError):r.audit_observer_report(sealed(bad),data,dict(name=c.phase),preflight)


if __name__=='__main__':unittest.main()
