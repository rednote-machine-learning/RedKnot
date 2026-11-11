"""CPU admission regressions for fresh three-arm twenty-case measurement."""
import copy
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

from gpu_pilot import natural_boundary_protocol as protocol
from gpu_pilot import run_natural_boundary_pilot as runner


def seal(value):
    body={k:v for k,v in value.items() if k!='sha256'}
    return dict(body,sha256=runner._digest(body))


def phase(arm='headtail128_link',repeat=0,*,quality=False,diagnostic=False,tokens=32):
    ids=list(range(tokens)); ttft={'recomputed':80.,'tail128_link':50.,'headtail128_link':60.}[arm]+repeat
    budget=runner.quality_token_budget(8450)
    quality_scope=dict(quality_requested_max_new_tokens=None,technical_max_new_tokens=budget) if quality else {}
    quality_record=dict(technical_max_new_tokens=budget,quality_output_complete=True,quality_limit_reason=None) if quality else {}
    return seal(dict(schema=runner.PHASE_SCHEMA,index=2,dataset='hotpotqa',arm=arm,
        name=f'case2.{arm}_measure{repeat}',repeat=repeat,mode='native' if arm=='recomputed' else 'sensitive',
        fraction=None,natural_boundary_inputs=True,dynamic_document_swa=arm=='headtail128_link',native_yoco_swa_share_hidden=False,
        **protocol.phase_policy(arm),diagnostic=diagnostic,partial_smoke=False,
        timing_valid=not quality and not diagnostic,quality_eligible=quality,
        input_ids_sha256='a'*64,manifest_sha256='b'*64,execution_configuration_sha256='c'*64,
        natural_boundary_protocol_sha256='d'*64,online_global_window_s=110.,**quality_scope,
        record=dict(success=True,rank=0,input_tokens=8450,input_ids_sha256='a'*64,output_ids=ids,**quality_record,
            output_ids_sha256=runner._digest(ids),natural_EOS=quality,online_ttft_s=ttft,
            online_elapsed_s=ttft+10,online_prepare_to_submit_s=.2,
            finish_reason='stop' if quality else 'length',truncated=False,answer_em=0.,answer_token_f1=0.)))


def row(lengths=(256,300,511,512,800,1024,1200,8193)):
    ids=list(range(11));documents=[]
    for n in lengths:
        start=len(ids);ids.extend(range(start,start+n));end=len(ids)
        sha=runner._digest(ids[start:end])
        documents.append(dict(input_start=start,input_end=end,tokens=n,online_position=start,
            offline_position=0,token_ids_sha256=sha,document_id=sha))
        ids.extend(range(len(ids),len(ids)+7))
    ids.extend(range(len(ids),len(ids)+53))
    return dict(index=2,dataset='hotpotqa',input_ids=ids,input_tokens=len(ids),
        input_ids_sha256=runner._digest(ids),documents=documents)


class NaturalBoundaryProtocolTests(unittest.TestCase):
    def test_quality_has_only_engine_context_bound_and_timing_remains_fixed32(self):
        p=protocol.natural_boundary_protocol()
        self.assertIsNone(p['quality']['max_tokens'])
        self.assertEqual(p['quality']['stop_policy'],'natural_eos_or_engine_context_limit')
        self.assertEqual(p['quality']['engine_max_model_len'],270336)
        self.assertEqual(p['quality']['model_max_position_embeddings'],1048576)
        self.assertFalse(p['quality']['bootstrap_is_quality_output_cap'])
        self.assertEqual(p['timing']['output_tokens'],32)
        self.assertEqual(runner.quality_token_budget(15964),254372)
        for value in (None,True,0,270336,270337):
            with self.subTest(input=value),self.assertRaises(ValueError):runner.quality_token_budget(value)

    def test_fixed_twenty_fresh_three_arms_and_common_bank(self):
        p=protocol.natural_boundary_protocol()
        self.assertEqual(p['cases'],list(range(2,22)))
        self.assertEqual(p['arms'],['recomputed','tail128_link','headtail128_link'])
        self.assertTrue(p['fresh_all_twenty_cases_all_three_arms'])
        self.assertFalse(p['historical_measurements_used'])
        self.assertEqual(p['cache_storage_policy'],'shared_gpu_full_canonical_bank_all_arms')
        self.assertIsNone(p['free_disk_minimum_bytes'])
        self.assertTrue(p['full_original_context_preserved'])
        self.assertEqual(p['minimum_cache_block_tokens'],256)
        self.assertEqual(p['sparse_parity']['arms'],list(runner.POLICY_ARMS))

    def test_cancelled_ninety_minutes_is_not_infinite_timeout(self):
        p=protocol.natural_boundary_protocol()
        self.assertTrue(p['previous_ninety_minute_limit_cancelled'])
        self.assertEqual(p['supervisor_timeout_seconds'],86400)
        self.assertEqual(p['prepare_timeout_seconds'],300)
        self.assertIn('failure backstop',p['timeout_purpose'])

    def test_policy_geometry_is_explicit_per_arm(self):
        self.assertEqual(protocol.phase_policy('recomputed'),dict(head_tokens=0,tail_tokens=0,dynamic_refresh_start_layer=None))
        self.assertEqual(protocol.phase_policy('tail128_link'),dict(head_tokens=0,tail_tokens=128,dynamic_refresh_start_layer=None))
        self.assertEqual(protocol.phase_policy('headtail128_link'),dict(head_tokens=128,tail_tokens=128,dynamic_refresh_start_layer=3))
        with self.assertRaises(ValueError):protocol.phase_policy('tail128_dynamic')

    def test_counterbalance_every_arm_visits_every_repeat_slot(self):
        for index in range(2,22):
            schedule=runner.timing_schedule(index)
            self.assertEqual(len(schedule),9)
            for repeat in range(3):
                self.assertEqual({a for r,a in schedule if r==repeat},set(runner.ARMS))
            for slot in range(3):
                self.assertEqual({schedule[3*r+slot][1] for r in range(3)},set(runner.ARMS))
        for index in (True,0,1,22):
            with self.assertRaises(ValueError):runner.timing_schedule(index)

    def test_frozen_inputs_and_source_pins_are_required(self):
        with mock.patch.object(protocol,'DEPENDENCIES',{}),self.assertRaisesRegex(ValueError,'must be frozen'):
            runner.source_hashes(Path('/unused'))

    def test_diagnostic_native_pins_fail_before_gpu_admission(self):
        p=protocol.natural_boundary_protocol();pins=p['diagnostics']
        identity=dict(checkpoint_config_sha256=pins['checkpoint_config_sha256'],
            runtime_source_files_sha256=dict(pins['native_source_files_sha256']))
        runner.verify_diagnostic_native_identity(identity,p)
        changed=copy.deepcopy(identity);changed['checkpoint_config_sha256']='0'*64
        with self.assertRaises(ValueError):runner.verify_diagnostic_native_identity(changed,p)
        for name in pins['native_source_files_sha256']:
            changed=copy.deepcopy(identity);changed['runtime_source_files_sha256'][name]='0'*64
            with self.subTest(name=name),self.assertRaises(ValueError):runner.verify_diagnostic_native_identity(changed,p)
        changed=copy.deepcopy(p);changed['diagnostics']['native_index_topk']=256
        with self.assertRaises(ValueError):runner.verify_diagnostic_native_identity(identity,changed)

    def test_formal_clock_gate_and_diagnostic_order(self):
        source=Path(runner.__file__).read_text()
        start=source.index('        def phase(');end=source.index('        check_row =',start)
        body=source[start:end]
        self.assertLess(body.index('began = time.perf_counter()'),body.index('controller.reset('))
        self.assertIn('arm=arm, diagnostic=diagnostic',body)
        self.assertLess(source.index('set(partial_smokes) == set(POLICY_ARMS)'),source.index('            quality = {}'))
        self.assertLess(source.index('timing_summary = summarize_timing'),source.index('diagnostic = phase('))
        self.assertNotIn('support.validate_inputs(',source)


class NaturalBoundaryGeometryTests(unittest.TestCase):
    def test_actual_spans_drive_bank_and_layer_chunks(self):
        r=row();g=runner.case_geometry(r)
        self.assertEqual(g['document_token_lengths'],[256,300,511,512,800,1024,1200,8193])
        self.assertEqual(g['offline_bank_rows'],40*sum(g['document_token_lengths']))
        self.assertEqual(g['offline_bank_bytes'],g['offline_bank_rows']*1024)
        self.assertEqual(g['native_prefill_chunks'],(len(r['input_ids'])+8191)//8192)
        self.assertEqual(g['layer_chunk_checks'],40*g['native_prefill_chunks'])
        self.assertGreater(g['nondocument_tokens'],0)

    def test_bad_overlap_identity_or_short_block_rejected(self):
        for field,value in [('input_start',0),('tokens',32768),('document_id','x'*64)]:
            r=row();r['documents'][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):runner.case_geometry(r)
        with self.assertRaises(ValueError):runner.case_geometry(row((255,)*8))

    def test_offline_jobs_use_actual_complete_ids(self):
        r=row()
        class Base:
            @staticmethod
            def document_specs(item):
                return [dict(token_ids=item['input_ids'][d['input_start']:d['input_end']],
                    token_ids_sha256=d['token_ids_sha256']) for d in item['documents']]
        jobs=list(runner.document_jobs(Base,r))
        self.assertEqual([j['input_tokens'] for j in jobs],runner.document_lengths(r))
        self.assertEqual([j['input_ids_sha256'] for j in jobs],[d['token_ids_sha256'] for d in r['documents']])

    def test_storage_budget_comes_from_all_twenty_real_inputs(self):
        rows=[dict(row(),index=i,dataset=('hotpotqa' if i%2==0 else '2wikimqa')) for i in range(2,22)]
        g=runner.manifest_geometry(dict(cohort=rows),protocol.natural_boundary_protocol())
        self.assertEqual(len(g['cases']),20)
        self.assertEqual(g['offline_payload_bytes_all_cases'],sum(runner.case_geometry(r)['offline_bank_bytes'] for r in rows))
        self.assertGreaterEqual(g['free_disk_required_bytes'],g['offline_payload_bytes_all_cases']+8*1024**3)
        self.assertEqual(g['sparse_qualification']['layer_chunk_checks'],runner.case_geometry(rows[0])['layer_chunk_checks'])

    def test_native_qualification_remains_exact_two_chunks(self):
        cal=dict(row(),index=0);cal['input_ids']=list(range(17000))
        q=runner.qualification_row(dict(warmup=[cal]))
        self.assertEqual(q['input_tokens'],16384)
        cal['input_ids']=list(range(16000))
        with self.assertRaises(ValueError):runner.qualification_row(dict(warmup=[cal]))


class NaturalBoundaryPhaseTests(unittest.TestCase):
    def samples(self):return [phase(arm,r) for r,arm in runner.timing_schedule(2)]

    def test_fresh_three_arm_summary_and_paired_repeat_ratios(self):
        result=runner.summarize_timing(self.samples(),2)
        self.assertTrue(result['fresh_paired_baseline'])
        self.assertEqual(result['arms']['recomputed']['online_ttft_p50_s'],81.)
        self.assertEqual(result['comparisons']['tail128_link']['ttft_speedup'],81/51)
        self.assertEqual(result['comparisons']['headtail128_link']['paired_repeat_speedups'],[80/60,81/61,82/62])

    def test_missing_arm_repeat_or_wrong_order_rejected(self):
        for values in (self.samples()[:-1],list(reversed(self.samples()))):
            with self.assertRaises(ValueError):runner.summarize_timing(values,2)

    def test_policy_or_native_config_mutation_rejected(self):
        for arm in runner.ARMS:
            for update in (dict(head_tokens=256),dict(tail_tokens=256),dict(native_yoco_swa_share_hidden=True),
                           dict(dynamic_document_swa=arm!='headtail128_link')):
                with self.subTest(arm=arm,update=update),self.assertRaises(ValueError):
                    runner.verify_phase(seal(dict(phase(arm),**update)),index=2,arm=arm,timing=True,quality=False)

    def test_diagnostic_cannot_enter_quality_or_timing(self):
        with self.assertRaises(ValueError):
            runner.verify_phase(seal(dict(phase(),diagnostic=True)),index=2,arm='headtail128_link',timing=True,quality=False)

    def test_quality_preserves_natural_eos_output_beyond_64_tokens(self):
        value=phase('recomputed',quality=True,tokens=257)
        record=runner.verify_phase(value,index=2,arm='recomputed',timing=False,quality=True)
        self.assertEqual(len(record['output_ids']),257)
        self.assertTrue(record['quality_output_complete'])
        self.assertIsNone(record['quality_limit_reason'])

    def test_length_finish_requires_the_actual_context_budget(self):
        value=phase('recomputed',quality=True,tokens=64)
        value['record'].update(finish_reason='length',truncated=True,
            quality_output_complete=False,quality_limit_reason='engine_context_limit')
        with self.assertRaises(ValueError):
            runner.verify_phase(seal(value),index=2,arm='recomputed',timing=False,quality=True)
        # A deliberately near-capacity prompt reaches the real remaining context.
        value['record']['input_tokens']=270272
        value['record']['technical_max_new_tokens']=value['technical_max_new_tokens']=64
        result=runner.verify_phase(seal(value),index=2,arm='recomputed',timing=False,quality=True)
        self.assertTrue(result['truncated'])
        self.assertFalse(result['quality_output_complete'])

    def test_quality_application_cap_or_missing_completion_binding_rejected(self):
        value=phase('recomputed',quality=True,tokens=5)
        for update in (dict(quality_requested_max_new_tokens=64),dict(technical_max_new_tokens=4096)):
            with self.subTest(update=update),self.assertRaises(ValueError):
                runner.verify_phase(seal(dict(value,**update)),index=2,arm='recomputed',timing=False,quality=True)
        for update in (dict(quality_limit_reason='engine_context_limit'),dict(quality_output_complete=False)):
            bad=copy.deepcopy(value);bad['record'].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):
                runner.verify_phase(seal(bad),index=2,arm='recomputed',timing=False,quality=True)

    def test_diagnostic_native_first_token_and_identity_gate(self):
        q=phase('recomputed',quality=True,tokens=5)
        d=seal(dict(phase('recomputed',diagnostic=True,tokens=1),name='case2.diagnostic_recomputed',repeat=None,
            controller=dict(diagnostic=True,early_token_diagnostics={'schema':'early-token-diagnostics-natural-boundary-v1'})))
        self.assertTrue(runner.verify_diagnostic(d,q,2))
        for field,value in (('input_ids_sha256','x'*64),('mode','sensitive'),('diagnostic',False)):
            with self.subTest(field=field),self.assertRaises(ValueError):runner.verify_diagnostic(seal(dict(d,**{field:value})),q,2)
        changed=copy.deepcopy(d);changed['record']['output_ids']=[99];changed['record']['output_ids_sha256']=runner._digest([99])
        with self.assertRaises(ValueError):runner.verify_diagnostic(seal(changed),q,2)

    def test_same_physical_bank_required_for_all_fifteen_formal_phases(self):
        values=[]
        for arm in runner.ARMS:
            for i in range(5):
                values.append(dict(arm=arm,controller=dict(comparison_schema='headtail-natural-boundary-comparison-v1',
                    comparison_arm=arm,cache_storage_policy='shared_gpu_full_canonical_bank_all_arms',
                    gpu_bank_constant_across_arms=True,offline_gpu_resident=True,bank_generation=3,
                    offline_bank_scope_sha256='e'*64,offline_bank_entry_count=320,
                    natural_boundary_inputs=True, document_token_lengths=runner.document_lengths(row()),
                    offline_bank_rows=runner.case_geometry(row())['offline_bank_rows'],
                    offline_bank_bytes=runner.case_geometry(row())['offline_bank_bytes'])))
        self.assertEqual(runner.verify_shared_bank(values,row())['bank_generation'],3)
        values[-1]['controller']['bank_generation']=4
        with self.assertRaises(ValueError):runner.verify_shared_bank(values,row())
        with self.assertRaises(ValueError):runner.verify_shared_bank(values[:-1],row())

    def test_each_policy_smoke_requires_its_own_sparse_and_real_reuse(self):
        fake=ModuleType('gpu_pilot.read_natural_boundary_results')
        fake.documents=lambda row:[]
        fake.audit_controller=lambda *args:dict(ffn_bypassed_rows=100,swa_reused_rows=100,
            consumed_cached_pairs=100,dynamic_refreshed_rows=100)
        case=row();case['input_ids_sha256']='a'*64
        q=dict(passed=True,identity_check_count=80,sha256='q'*64,empty_ep_qualification_sha256='e'*64)
        with mock.patch.dict(sys.modules,{'gpu_pilot.read_natural_boundary_results':fake}):
            for arm in runner.POLICY_ARMS:
                p=seal(dict(phase(arm,tokens=1),timing_valid=False,partial_smoke=True,
                    empty_ep_qualification_sha256='e'*64,controller=dict(dynamic_document_swa=arm=='headtail128_link')))
                sparse=dict(passed=True,arm=arm,input_ids_sha256='a'*64,candidate_sha256=p['sha256'],
                    qualification_sha256=q['sha256'],sha256='s'*64,
                    input_tokens=case['input_tokens'],identity_check_count=runner.case_geometry(case)['layer_chunk_checks'])
                closure=runner.verify_partial_smoke([p],case,q,sparse,arm=arm)
                self.assertEqual(closure['arm'],arm)
                with self.assertRaises(ValueError):
                    runner.verify_partial_smoke([p],case,q,dict(sparse,arm='foreign'),arm=arm)
                with self.assertRaises(ValueError):
                    runner.verify_partial_smoke([p],case,q,dict(sparse,candidate_sha256='f'*64),arm=arm)
            fake.audit_controller=lambda *args:dict(ffn_bypassed_rows=100,swa_reused_rows=0,
                consumed_cached_pairs=0,dynamic_refreshed_rows=0)
            with self.assertRaises(ValueError):runner.verify_partial_smoke([p],case,q,sparse,arm=arm)


if __name__=='__main__':unittest.main()
