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
from gpu_pilot import read_headtail_scale20_results as r
from kv_cache_engram.core import digest
from kv_cache_engram import early_token_diagnostics as producer
from kv_cache_engram.tail_link_plan import plan as tail_plan
from tests.test_headtail_comparison import controller
from tests import test_headtail_link_results as head_fixture


def sealed(body):
    value={k:v for k,v in body.items() if k!='sha256'}
    return dict(value,sha256=digest(value))


def spans():
    return [(5+i*32768,5+(i+1)*32768,str(i)*64) for i in range(8)]


def row():
    return dict(index=2,input_tokens=262194,input_ids_sha256='a'*64,
        documents=[dict(input_start=a,input_end=b,token_ids_sha256=k) for a,b,k in spans()])


def scope(phase,generation=7,bank_generation=2):
    c=controller();c.arm=phase['arm'];c.mode=phase['mode'];c.active_policy=c.arm
    c.phase_generation=generation;c.bank_generation=bank_generation
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
        full_bank_retained_but_static_attention_tail_only=True,tail_bank_source_shape=[32768,512],
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
        source_rows_per_document=32768,source_local_start=32640,checked=qualified,checks=2 if qualified else 0,
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
        self.assertEqual(result['actual_partial_chunks'],0)  # YOCO layout has single placeholder rows.
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

    def test_headtail_real_report_survives_dispatch_and_shared_bank_binding(self):
        c,data,phase,docs=head_fixture.ControllerAuditTests().fixture();phase['diagnostic']=False
        c.update(scope(phase),diagnostic=False,early_token_diagnostics=None)
        with patch.object(r.head,'FFN_LAYERS',(20,)):
            result=r.audit_controller(c,data,phase,docs)
        self.assertGreater(result['dynamic_refreshed_rows'],0)
        self.assertGreater(result['consumed_cached_pairs'],0)
        c['early_token_diagnostics']={'schema':'early-token-diagnostics-v1'}
        with self.assertRaises(ValueError):r.audit_controller(c,data,phase,docs)

    def test_scope_hash_independently_binds_all_documents_and_layers(self):
        identity=dict(checkpoint_index_sha256='a'*64,tokenizer_json_sha256='c'*64,execution_configuration_sha256='b'*64)
        phase=dict(arm='recomputed',mode='native');c=controller();c.arm=phase['arm'];c.mode=phase['mode']
        c.bank_generation=1;c.phase_generation=4
        for _,_,key in spans():
            for layer in range(40):
                c.bank[key,layer]=dict(scope=dict(model_sha256='a'*64,tokenizer_sha256='c'*64,
                    execution_sha256='b'*64,document_sha256=key,layer=layer,representation='swa-postnorm-prerope-bf16'))
        report=c._comparison_fields();expected=r.bank_scope_sha256(spans(),identity)
        self.assertEqual(report['offline_bank_scope_sha256'],expected)
        r.comparison_scope(report,phase,expected_scope=expected)
        report['offline_bank_scope_sha256']='f'*64
        with self.assertRaises(ValueError):r.comparison_scope(report,phase,expected_scope=expected)

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
        for n,layer in ((32768,3),(140,20),(1,39),(513,19)):
            p=np.arange(32768-n,32768,dtype=np.uint32)
            v=np.column_stack((rng.integers(0,9,n),rng.integers(1,5,n),rng.random(n)*2)).astype(np.float32)
            decoded=r.unpack_diagnostic_scalars(producer.pack_rows(p,v),p.tolist())
            np.testing.assert_array_equal(decoded[1],v)
            r.compare_statistics(producer.scalar_summary(p,v),r.diagnostic_summary(*decoded))
            r.compare_statistics(producer.native_row_groups(p,v,document_start=229381,total_tokens=262194,layer=layer),
                r.diagnostic_row_groups(p,v,229381,layer,262194))
            r.compare_statistics(producer.rank_comparison(p,v,p,v[::-1].copy()),
                r.diagnostic_rank_comparison(p,v,p,v[::-1].copy()))

    def test_zero_energy_ties_and_empty_groups_keep_null_mean(self):
        p=np.arange(4,dtype=np.uint32);v=np.zeros((4,3),np.float32)
        result=r.diagnostic_summary(p,v)
        self.assertIsNone(result['boundaries']['tail128']['mean_relative_l2'])
        self.assertIsNone(result['boundaries']['head128']['error_energy_fraction'])
        ranks=r.diagnostic_rank_comparison(p,v,p,v)
        self.assertIsNone(ranks['spearman']);self.assertEqual(ranks['top256_overlap_fraction'],1)
        r.compare_statistics(producer.scalar_summary(p,v),result)

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
        data=row();pin=r.headtail_scale20_protocol()['diagnostics']
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
                    native_row_groups=producer.native_row_groups(p,v,document_start=lo,total_tokens=data['input_tokens'],layer=layer),
                    summary=producer.scalar_summary(p,v),scalars=producer.pack_rows(p,v)))
        correlations=[dict(early_layer=a,late_layer=b,document_ordinal=i,
            **producer.rank_comparison(*values_by_key[a,i],*values_by_key[b,i]))
            for a in (1,2,3) for b in (7,13,19,20,39) for i in range(8)]
        report=sealed(dict(routing,schema='early-token-diagnostics-v1',complete=True,phase=phase['name'],
            input_ids_sha256=data['input_ids_sha256'],mode='native',arm='recomputed',included_in_formal_timing=False,
            representation='swa-postnorm-prerope-bf16',layers=list(r.DIAGNOSTIC_LAYERS),
            metrics=['error_l2_squared','reference_l2_squared','cosine_distance'],full_kv_vectors_exported=False,
            attention_weights_collected=False,only_actual_native_surviving_rows=True,offline_bank_mutated=False,
            bin_edges=list(r.DIAGNOSTIC_EDGES),native_position_coverage_checked=True,prompt_chunk_tokens=8192,
            native_yoco_keep_tokens=140,records=records,early_late_rank_comparisons=correlations,limitations=[]))
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
            result=r.audit_diagnostic(phase,{'index':2},{},'qual',7,{'bank':1},set(range(3,18)))
            self.assertTrue(result['excluded_from_formal_ttft_and_quality'])
            self.assertEqual(phase_check.call_args.kwargs,dict(arm='recomputed',timing=False,quality=False,fixed_tokens=1,diagnostic=True))
            for mutate in (lambda x:x['record'].update(output_ids=[8]),lambda x:x['controller'].update(phase_generation=4)):
                bad=copy.deepcopy(phase);mutate(bad)
                with self.assertRaises(ValueError):r.audit_diagnostic(bad,{'index':2},{},'qual',7,{'bank':1},set(range(3,18)))


if __name__=='__main__':unittest.main()
