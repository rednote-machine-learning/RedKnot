"""Independent CPU oracle/mutation tests; fixtures are not GPU evidence."""
import copy
import json
import random
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from kv_cache_engram.core import digest
from gpu_pilot import read_s1_results as r
from gpu_pilot import s1_protocol as protocol
from tests.test_natural_boundary_runner import row as _original_row

def row(lengths=(256,300,511,512,800,1024,1200,8193,400,600)):
    if len(lengths)==8:lengths=tuple(lengths)+(400,600)
    data=_original_row(lengths);data["dataset"]="musique"
    return data


SHA='a'*64


def seal(value):
    body={k:v for k,v in value.items() if k!='sha256'}
    return dict(body,sha256=digest(body))


def fixture(*,reference=False,qualified=True,empty=False):
    data=row((256,301,513,1000,4000,260,8193,257))
    if empty:
        data=dict(index=0,dataset='hotpotqa',input_tokens=16384,input_ids_sha256=SHA,
            s1_empty_chunk_qualification=True,documents=[dict(input_start=0,input_end=16384,token_ids_sha256=SHA)])
    spans=r.documents(data);total=data['input_tokens'];chunks=r.chunk_oracle(spans,total)
    mode='reference_sparse' if reference else 'sensitive';generation=1 if reference else 2
    phase=dict(schema=r.PHASE_SCHEMA,index=data['index'],name='fixture',mode=mode,arm='s1',fraction=None,
        input_ids_sha256=data['input_ids_sha256'],timing_valid=False,quality_eligible=False,
        verify_sparse=qualified and not reference,diagnostic=False,natural_boundary_inputs=True,s1_policy=True,
        native_yoco_swa_share_hidden=False,**protocol.phase_policy('s1'))
    events=[];attention=[];global_events=[];engram=[];fingerprints=[];previous=[]
    selected=r.position_oracle(spans,total)['selected_positions']
    for c in chunks:
        start,end=c['chunk_start'],c['chunk_end'];positions=c['selected_positions'];n=len(positions)
        memory=r.local_memory_oracle(spans,total,c,previous)
        engram.append(dict(layer=14,chunk_start=start,chunk_end=end,original_hash_rows=end-start,
            selected_positions=positions,actual_engram_rows=end-start if reference else n,
            hook_consumed_once=True,original_ngram_hashes_preserved=True,
            empty_hook_consumed_without_fake_rows=n==0 and not reference))
        for layer in range(40):
            current=list(range(start,end)) if layer<3 else positions
            actual=end-start if layer<3 or reference else n
            events.append(dict(layer=layer,chunk_start=start,chunk_end=end,full_rows=end-start,
                selected_positions=current,selected_rows=len(current),actual_attention_rows=len(current),
                actual_ffn_rows=actual,actual_residual_rows=actual,actual_mhc_rows=actual,
                dense_reference_only=reference and layer>=3,original_absolute_positions=True,
                empty_native_ep_call=layer>=3 and actual==0))
            nr=len(current);fp=lambda shape:dict(shape=shape,dtype='torch.bfloat16',bytes=2*nr*4,sha256=SHA)
            fingerprints.append(dict(key=[start,end,layer],value=dict(output=fp([nr,4]),
                lazy={k:fp([1,nr,4]) for k in ('residual','post_layer_mix','comb_res_mix','pre_layer_mix')})))
            if layer<3:continue
            attention.append(dict(layer=layer,chunk_start=start,chunk_end=end,selected_positions=positions,
                selected_rows=n,full_rows=end-start,memory_positions=memory['memory_positions'],
                offline_halo_rows=memory['source_kinds'].count('halo'),actual_attention_rows=n,
                projection_identity_checks=3 if reference else 0,max_absolute_error=0.0 if reference else None,
                gpu_local_window_oracle_checked=True,actual_native_swa_indices_checked=True,global_causal_checked=True,
                no_unselected_hidden_fallback=True,no_extra_swa_links=True,plan_sha256=memory['sha256']))
            if layer in (8,14,20):
                ratio=1 if layer==20 else 2;begin,last=start//ratio,end//ratio
                affected=sorted({p//ratio for p in selected if begin*ratio<=p<last*ratio})
                global_events.append(dict(layer=layer,chunk_start=start,chunk_end=end,phase_generation=generation,
                    compression_ratio=ratio,online_projection_rows=n,online_projection_positions=positions,full_raw_rows=end-start,
                    complete_compressed_ids=list(range(begin,last)),affected_compressed_ids=affected,
                    offline_base_rows=last-begin-len(affected),checks=2 if qualified else 0,
                    max_absolute_error=0.0 if qualified else None,source2_untouched=True,
                    main_and_index_k_native_store=True,global_base_representation='compressed_prenorm_bf16',
                    packed_main_and_index_store_checked=qualified,native_compressor_carry_updated=ratio==2))
        previous=[p for p in previous+positions if p>=end-127]
    cache=dict(schema='s1-cache-report-v1',full_swa_bank_persisted=False,hidden_or_ffn_bank_persisted=False,
        phase_preparation_in_ttft_required=True,source2_native_prepass_unchanged=True,
        phase_generation=generation,phase_preparation_s=.01,global_events=global_events)
    rings=[dict(layer=l,positions=list(range(total-128,total)),physical_store=True,
        physical_readback_exact=qualified,checks=128 if qualified else 0) for l in range(3,40)]
    sources=[]
    if qualified:
        for c in chunks:
            fp=dict(schema='s1-source2-cache-fingerprint-v1',rows=c['chunk_end']//2,main_sha256=SHA,index_k_sha256=SHA)
            sources.append(dict(chunk_start=c['chunk_start'],chunk_end=c['chunk_end'],before=fp,after=copy.deepcopy(fp),exact=True))
    geometry=r.case_geometry(data)
    control=dict(schema='s1-controller-v1',mode=mode,arm='s1',phase='fixture',input_ids_sha256=data['input_ids_sha256'],expected_document_count=10,
        full_input_tokens=total,first_three_full=True,native_yoco_swa_share_hidden=False,decode_native=True,
        model_boundary_restored=True,extra_cross_document_tail_links=False,native_local_window=128,
        source2_native_cache_preserved=True,source2_formal_byte_hash_claim=False,
        phase_generation=generation,cache_report=cache,online_cache_prepare_s=.01,
        selected_survive_layer39=True,yoco_layout_preserved=False,unselected_deep_residual_mhc=reference,
        dense_reference_only=reference,decoder_events=events,source2_identity_events=sources,engram_events=engram,
        capture_sampler_events=[],
        attention_report=dict(schema='s1-attention-report-v1',full_swa_bank_used=False,only_tail_and_nondoc_queries=True,
            native_global_attention_kernel=True,selected_queries_survive_layers3_through39=True,
            attention_events=attention,decode_ring_events=rings),
        sparse_identity_fingerprints=fingerprints,sparse_identity_fingerprints_sha256=digest(fingerprints),
        sparse_identity_check_count=len(fingerprints),sparse_identity_exact=not reference and qualified,
        offline_gpu_resident=True,gpu_bank_constant_across_arms=True,bank_generation=1,
        offline_bank_scope_sha256=SHA,cache_storage_policy='same_s1_offline_gpu_bank_resident_both_arms',
        **{k:v for k,v in geometry.items() if k.startswith('offline_') or k=='document_token_lengths'})
    phase.update(controller=control,record=dict(success=True,rank=0,input_ids_sha256=data['input_ids_sha256'],
        input_tokens=total,natural_EOS=False,output_ids=[1],output_ids_sha256=digest([1])))
    return data,seal(phase)


class S1ResultsTests(unittest.TestCase):
    def test_reader_import_is_cpu_only(self):
        result=subprocess.run([sys.executable,'-c',"import sys; from gpu_pilot import read_s1_results; assert 'torch' not in sys.modules and not any(x.startswith('deepinfer') for x in sys.modules)"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_real_producer_geometry_matches_independent_oracle(self):
        from kv_cache_engram.s1_plan import all_chunks
        from kv_cache_engram.s1_attention import local_memory_plan
        data=row();spans=r.documents(data);docs=[dict(start=a,end=b,key=k) for a,b,k in spans]
        expected=r.chunk_oracle(spans,data['input_tokens']);produced=all_chunks(docs,data['input_tokens'])
        previous=[]
        for a,b in zip(expected,produced):
            self.assertEqual(a['selected_positions'],b['selected_positions'])
            self.assertEqual(local_memory_plan(docs,data['input_tokens'],a['chunk_start'],a['chunk_end'],a['selected_positions'],previous),
                r.local_memory_oracle(spans,data['input_tokens'],a,previous))
            previous=[p for p in previous+a['selected_positions'] if p>=a['chunk_end']-127]

    def test_actual_controller_finish_report_passes_reader_contract(self):
        from kv_cache_engram.s1_runtime import S1Controller
        data,phase=fixture(qualified=False);expected=phase['controller'];spans=r.documents(data)
        controller=object.__new__(S1Controller)
        controller.state='RUNNING';controller.ctx=None;controller.coverage=data['input_tokens']
        controller.ids=tuple(range(data['input_tokens']));controller.input_sha=data['input_ids_sha256']
        controller.mode='sensitive';controller.arm='s1';controller.phase='fixture';controller.verify_sparse=False
        controller.native_model=SimpleNamespace(yoco_boundary_layer_idx=None);controller.original_boundary=20
        controller.docs=[dict(start=a,end=b,key=k) for a,b,k in spans]
        controller.expected_document_count=10
        controller.observed={};controller.sparse_observed={};controller.qualified=True;controller.sparse_qualified=True
        controller.decoder_events=expected['decoder_events'];controller.source2_events=[]
        controller.engram_events=expected['engram_events'];controller.bank_generation=1
        controller.capture_sampler_events=[]
        controller.scope=dict(model_sha256=SHA,execution_sha256=SHA,tokenizer_sha256=SHA)
        controller.empty_ep_capability=None
        cache=copy.deepcopy(expected['cache_report'])
        cache.update({k:v for k,v in r.case_geometry(data).items() if k.startswith('offline_')})
        cache['capture_reports']=[dict(document_sha256=k,document_tokens=b-a,payloads=[]) for a,b,k in spans]
        controller.cache=SimpleNamespace(report=lambda:cache)
        controller.attention=SimpleNamespace(report=lambda:expected['attention_report'])
        actual=controller.finish();phase['controller']=actual
        result=r.audit_controller(actual,data,phase,spans)
        r.comparison_scope(actual,phase,spans=spans)
        self.assertGreater(result['residual_bypassed_rows'],0)
        self.assertEqual(controller.native_model.yoco_boundary_layer_idx,20)

    def test_zero_deep_chunk_preserves_late_selected_rows(self):
        data,phase=fixture(empty=True)
        result=r.audit_controller(phase['controller'],data,phase,r.documents(data))
        self.assertEqual(result['actual_empty_chunks'],37)
        self.assertEqual(result['ffn_selected_rows'],37*128)
        self.assertEqual(result['residual_bypassed_rows'],37*(16384-128))
        self.assertGreater(result['consumed_cached_pairs'],0)

    def test_dense_and_compact_compare_same_selected_fingerprints(self):
        data,reference=fixture(reference=True,empty=True)
        _,candidate=fixture(empty=True)
        result=r.sparse_identity(reference,candidate,data,SHA)
        self.assertTrue(result['passed']);self.assertTrue(result['empty_chunk_exercised'])
        self.assertFalse(result['equivalence_to_full_model_claim'])
        self.assertEqual(result['identity_check_count'],80)

    def test_skip_all_residual_mhc_is_enforced(self):
        for field in ('actual_ffn_rows','actual_residual_rows','actual_mhc_rows'):
            data,phase=fixture();event=phase['controller']['decoder_events'][3];event[field]=event['full_rows']
            with self.subTest(field=field),self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_future_or_missing_swa_key_and_yoco_intersection_rejected(self):
        for mutate in (lambda c:c['attention_report']['attention_events'][0]['memory_positions'].append(999999),
                       lambda c:c.update(selected_survive_layer39=False),
                       lambda c:c['decoder_events'][39]['selected_positions'].pop(),
                       lambda c:c['attention_report']['attention_events'][0].update(global_causal_checked=False)):
            data,phase=fixture();mutate(phase['controller'])
            with self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_global_compression_groups_and_current_member_coverage_are_bound(self):
        for mutate in (lambda e:e['affected_compressed_ids'].pop(),lambda e:e.update(compression_ratio=1),
                       lambda e:e.update(online_projection_rows=0),lambda e:e.update(source2_untouched=False),
                       lambda e:e.update(checks=0)):
            data,phase=fixture();mutate(phase['controller']['cache_report']['global_events'][0])
            with self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_source2_main_and_index_bytes_must_survive(self):
        for field in ('main_sha256','index_k_sha256'):
            data,phase=fixture();phase['controller']['source2_identity_events'][0]['after'][field]='b'*64
            with self.subTest(field=field),self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_final_physical_decode_ring_is_required(self):
        data,phase=fixture();phase['controller']['attention_report']['decode_ring_events'][0]['physical_readback_exact']=False
        with self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_empty_engram_hook_consumes_without_fake_rows(self):
        data,phase=fixture(empty=True);phase['controller']['engram_events'][0]['empty_hook_consumed_without_fake_rows']=False
        with self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_formal_phases_do_not_claim_qualification_sync(self):
        data,phase=fixture(qualified=False)
        result=r.audit_controller(phase['controller'],data,phase,r.documents(data));self.assertGreater(result['ffn_bypassed_rows'],0)
        phase['controller']['source2_identity_events']=[{}]
        with self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_sparse_fingerprint_shape_and_phase_generation_mutations_fail(self):
        data,reference=fixture(reference=True)
        for mutate in (lambda x:x['controller'].update(phase_generation=1),
                       lambda x:x['controller']['sparse_identity_fingerprints'][3]['value']['output']['shape'].__setitem__(0,999),
                       lambda x:x['record'].update(output_ids=[2],output_ids_sha256=digest([2]))):
            _,candidate=fixture();mutate(candidate);candidate=seal(candidate)
            with self.assertRaises(ValueError):r.sparse_identity(reference,candidate,data,SHA)

    def test_halo_and_global_storage_are_not_full_swa_bank(self):
        data=row();g=r.case_geometry(data);s=sum(g['document_token_lengths'])
        self.assertEqual(g['offline_bank_entry_count'],400)
        self.assertEqual(g['offline_bank_bytes'],37*10*128*1024+s*9216)
        self.assertNotEqual(g['offline_bank_bytes'],40*s*1024)

    def test_musique_ten_documents_rejects_old_eight_or_wrong_controller_scope(self):
        data=row();self.assertEqual(len(r.documents(data)),10)
        bad=copy.deepcopy(data);bad['documents']=bad['documents'][:8]
        with self.assertRaises(ValueError):r.documents(bad)
        data,phase=fixture();phase['controller']['expected_document_count']=8
        with self.assertRaises(ValueError):r.audit_controller(phase['controller'],data,phase,r.documents(data))

    def test_deadline_partial_prefix_is_never_whole_run_complete(self):
        cases=[dict(index=i,shared_bank_identity=dict(bank_generation=i+1)) for i in range(2,7)]
        value=r.completion_state(cases,[],complete=False,allow_partial=True,partial_reason='hard_deadline')
        self.assertEqual(value['status'],'partial_complete');self.assertEqual(value['planned_cases'],20)
        self.assertEqual(value['complete_cases'],5);self.assertEqual(value['completed_case_indices'],list(range(2,7)))
        self.assertEqual(value['missing_case_indices'],list(range(7,22)))
        self.assertEqual(value['partial_completion_reason'],'hard_deadline')
        self.assertTrue(value['all_completed_cases_audited'])
        for key in ('overall_run_complete','final_run_audited','all_registered_cases_audited'):
            self.assertFalse(value[key])
        self.assertEqual(r.completion_state(cases,[],complete=False)['status'],'incomplete')

    def test_zero_case_deadline_partial_has_no_fabricated_summary(self):
        value=r.completion_state([],[],complete=False,allow_partial=True,partial_reason='hard_deadline')
        self.assertEqual(value['status'],'partial_complete');self.assertEqual(value['complete_cases'],0)
        self.assertEqual(value['missing_case_indices'],list(range(2,22)));self.assertIsNone(r.summarize_cases([]))

    def test_partial_reason_preserves_runner_failure_and_bounded_window_distinction(self):
        for reason in ('runner_failure','bounded_window'):
            value=r.completion_state([],[],complete=False,allow_partial=True,partial_reason=reason)
            self.assertEqual(value['status'],'partial_complete')
            self.assertEqual(value['partial_completion_reason'],reason)
            self.assertFalse(value['overall_run_complete'])
        with self.assertRaises(ValueError):r.completion_state([],[],complete=False,
            allow_partial=True,partial_reason='success')

    def test_partial_rejects_gap_errors_fullcount_without_run_and_missing_reason(self):
        def case(i,g=None):return dict(index=i,shared_bank_identity=dict(bank_generation=g or i+1))
        for cases in ([case(2),case(4)],[case(2),case(3,3)],[case(i) for i in range(2,22)]):
            with self.assertRaises(ValueError):r.completion_state(cases,[],complete=False,
                allow_partial=True,partial_reason='hard_deadline')
        for kwargs in (dict(allow_partial=True),dict(partial_reason='hard_deadline'),
                       dict(allow_partial=True,partial_reason='hard_deadline',case_index=2)):
            with self.assertRaises(ValueError):r.completion_state([case(2)],[],complete=False,**kwargs)
        value=r.completion_state([case(2)],[dict(index=3,error='bad seal')],complete=False,
            allow_partial=True,partial_reason='hard_deadline')
        self.assertEqual(value['status'],'invalid');self.assertFalse(value['all_completed_cases_audited'])
        with self.assertRaises(ValueError):r.completion_state([case(2)],[],complete=True)

    def test_six_case_complete_requires_exact_indices_and_no_errors(self):
        cases=[dict(index=i,shared_bank_identity=dict(bank_generation=i+1)) for i in range(2,22)]
        value=r.completion_state(cases,[],complete=True)
        self.assertEqual(value['status'],'complete');self.assertTrue(value['overall_run_complete'])
        self.assertEqual(value['missing_case_indices'],[]);self.assertIsNone(value['partial_completion_reason'])
        with self.assertRaises(ValueError):r.completion_state(cases,[dict(error='bad')],complete=True)
        subset=r.completion_state(cases[:1],[],complete=False,case_index=2)
        self.assertEqual(subset['status'],'subset_complete')
        self.assertFalse(subset['all_completed_cases_audited'])

    def test_capture_sampler_layout_matches_original_positions(self):
        for total in (256,4096,8192,8193,8300,8332,16384,16400):
            events=[]
            for start in range(0,total,8192):
                end=min(total,start+8192)
                # Independent position-set oracle, including the early placeholder.
                positions=list(range(start,end));survivors=[p for p in positions if p>=total-140]
                if not survivors:survivors=[positions[-1]]
                events.append(dict(chunk_start=start,chunk_end=end,full_output_rows=len(positions),
                    native_sampler_rows=len(survivors),output_local_start=survivors[0]-start,
                    output_last_absolute_position=survivors[-1],source='full40_post_head_norm',
                    rebinding_after_capture=True))
            with self.subTest(total=total):
                self.assertEqual(r.audit_capture_sampler(dict(capture_sampler_events=events),total),events)

    def test_capture_sampler_wrong_first_or_last_row_and_missing_chunk_rejected(self):
        total=16384
        good=dict(capture_sampler_events=[
            dict(chunk_start=0,chunk_end=8192,full_output_rows=8192,native_sampler_rows=1,
                output_local_start=8191,output_last_absolute_position=8191,
                source='full40_post_head_norm',rebinding_after_capture=True),
            dict(chunk_start=8192,chunk_end=16384,full_output_rows=8192,native_sampler_rows=140,
                output_local_start=8052,output_last_absolute_position=16383,
                source='full40_post_head_norm',rebinding_after_capture=True)])
        for mutate in (lambda e:e[1].update(output_last_absolute_position=8331),
                       lambda e:e[1].update(output_local_start=0),
                       lambda e:e[0].update(native_sampler_rows=140),
                       lambda e:e[1].update(source='before_native_head_norm'),
                       lambda e:e[1].update(rebinding_after_capture=False),lambda e:e.pop()):
            control=copy.deepcopy(good);mutate(control['capture_sampler_events'])
            with self.assertRaises(ValueError):r.audit_capture_sampler(control,total)

    def test_payload_hash_extent_and_nonfinite_values(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'raw.bin';raw=struct.pack('<ff',1.,2.);p.write_bytes(raw)
            self.assertTrue(r.audit_payload(p,hashlib.sha256(raw).hexdigest(),8,'float32')['finite'])
            for raw,dtype in ((struct.pack('<f',float('nan')),'float32'),(struct.pack('<H',0x7f80),'bfloat16')):
                p.write_bytes(raw)
                with self.assertRaises(ValueError):r.audit_payload(p,hashlib.sha256(raw).hexdigest(),len(raw),dtype)
            p.write_bytes(b'xx')
            with self.assertRaises(ValueError):r.audit_payload(p,SHA,4,'bfloat16')

    def test_full_quality_output_above64_and_technical_limit_only(self):
        text='unchanged full answer '*200+'FINAL MARKER';ids=list(range(200));n=10000;budget=270336-n
        rec=dict(text=text,output_ids=ids,output_ids_sha256=digest(ids),natural_EOS=True,technical_max_new_tokens=budget,
            finish_reason='stop',truncated=False,quality_output_complete=True,quality_limit_reason=None,
            answer_em=1.,answer_token_f1=1.)
        phase=dict(quality_requested_max_new_tokens=None,technical_max_new_tokens=budget)
        value=r.quality_record(rec,phase,dict(input_tokens=n,references=[text],dataset='loong',raw_answer=text))
        self.assertEqual(value['text'],text);self.assertTrue(value['text'].endswith('FINAL MARKER'))
        rec.update(finish_reason='length',truncated=True,quality_output_complete=False,quality_limit_reason='engine_context_limit')
        with self.assertRaises(ValueError):r.quality_record(rec,phase,dict(input_tokens=n,references=[text],dataset='loong',raw_answer=text))

    def test_paired_timing_has_two_arms_and_three_real_samples(self):
        samples=dict(recomputed=[3.,6.,9.],s1=[2.,3.,4.]);summary=r.expected_timing_summary(samples)
        self.assertEqual(summary['comparisons']['s1']['ttft_speedup'],2.)
        self.assertEqual(summary['comparisons']['s1']['paired_repeat_speedups'],[1.5,2.,2.25])
        with self.assertRaises(ValueError):r.expected_timing_summary(dict(recomputed=[3.],s1=[2.]))


if __name__=='__main__':unittest.main()
