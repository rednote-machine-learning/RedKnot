"""CPU admission tests for the independent single-candidate headtail experiment."""
import ast
import copy
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

from gpu_pilot import headtail_link_protocol as protocol
from gpu_pilot import run_headtail_link_pilot as runner


def seal(value):
    body = {k:v for k,v in value.items() if k != 'sha256'}
    return dict(body, sha256=runner._digest(body))


def phase(repeat=0, *, quality=False, tokens=32):
    arm = runner.ARMS[0]
    ids = list(range(tokens))
    return seal(dict(schema=runner.PHASE_SCHEMA, index=2, name=f'case2.{arm}_measure{repeat}',
        arm=arm, repeat=repeat, mode='sensitive', fraction=None,
        dynamic_document_swa=True, native_yoco_swa_share_hidden=False,
        head_tokens=128, tail_tokens=128, dynamic_refresh_start_layer=3,
        timing_valid=not quality, quality_eligible=quality,
        input_ids_sha256='a'*64, manifest_sha256='b'*64,
        execution_configuration_sha256='c'*64, headtail_link_protocol_sha256='d'*64,
        online_global_window_s=70.,
        record=dict(success=True, rank=0, input_ids_sha256='a'*64, output_ids=ids,
            output_ids_sha256=runner._digest(ids), natural_EOS=quality,
            online_ttft_s=55.+repeat, online_elapsed_s=60.+repeat,
            online_prepare_to_submit_s=.1, finish_reason='stop', truncated=False,
            answer_em=0., answer_token_f1=0.)))


class HeadTailProtocolTests(unittest.TestCase):
    def test_full_bank_union_scope_and_immediate_layer3_refresh(self):
        p = protocol.headtail_link_protocol()
        self.assertEqual(p['arms'], ['headtail128_link'])
        self.assertEqual((p['head_tokens'], p['tail_tokens'], p['dynamic_refresh_start_layer']), (128,128,3))
        self.assertEqual(p['first_full_layers'], [0,1,2])
        self.assertFalse(p['dynamic_layer3_static_link_seed'])
        self.assertIn('native local SWA window', p['memory'])
        self.assertIn('prior-document tail128', p['memory'])
        self.assertEqual(p['offline_shape'], [32768,512])
        self.assertEqual(p['offline_payload_bytes_per_case'], 10*1024**3)
        self.assertIs(p['native_yoco_swa_share_hidden'], False)
        self.assertTrue(p['global_kv_complete_online'])
        self.assertTrue(p['preserve_native_global_consumer_sharing'])
        self.assertFalse(p['forced_survivor_retention'])

    def test_timeout_is_frozen_with_separate_preparation_budget(self):
        p = protocol.headtail_link_protocol()
        self.assertEqual(p['supervisor_timeout_seconds'], 5400)
        self.assertEqual(p['prepare_timeout_seconds'], 300)
        self.assertEqual(p['supervisor_timeout_start'], 'after GPU release')

    def test_fresh_full_input_and_native_qualifications_required(self):
        p = protocol.headtail_link_protocol()
        self.assertEqual(p['new_parity']['layer_chunk_checks'], 80)
        self.assertEqual(p['sparse_parity']['layer_chunk_checks'], 1320)
        self.assertEqual(p['sparse_parity']['arms'], list(runner.ARMS))
        self.assertFalse(p['sparse_parity']['equivalence_to_full_model_claim'])

    def test_independent_source_pins_are_required(self):
        with mock.patch.object(protocol, 'DEPENDENCIES', {}), self.assertRaisesRegex(ValueError, 'must be frozen'):
            runner.source_hashes(Path('/unused'))

    def test_formal_clock_precedes_reset_and_gate_precedes_quality(self):
        source = Path(runner.__file__).read_text()
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == 'phase')
        body = ast.get_source_segment(source, node)
        self.assertLess(body.index('began = time.perf_counter()'), body.index('controller.reset('))
        self.assertNotIn('dynamic_document_swa=', body[body.index('controller.reset('):body.index('driver.barrier()')])
        self.assertLess(source.index('partial_smoke = driver.gather(smoke_result)[0]'), source.index('quality = {}'))
        self.assertNotIn('dynamic_partial_smoke', source)
        self.assertIn('headtail-link-case-v1', source)


class HeadTailPhaseTests(unittest.TestCase):
    def test_three_candidate_repeats_have_no_fresh_baseline_ratio(self):
        p = [phase(i) for i in range(3)]
        result = runner.summarize_timing(p, 2)
        self.assertEqual(result['arms']['headtail128_link']['online_ttft_p50_s'], 56.)
        self.assertNotIn('ttft_speedup', result['arms']['headtail128_link'])
        self.assertFalse(result['fresh_paired_baseline'])
        self.assertFalse(result['historical_results_used_for_admission'])

    def test_schedule_is_exactly_one_candidate_three_times(self):
        for index in (2,3,4,5):
            self.assertEqual(runner.timing_schedule(index), protocol.timing_schedule(index))
            self.assertEqual(runner.timing_schedule(index), [(i,'headtail128_link') for i in range(3)])
        for index in (True,0,6):
            with self.assertRaises(ValueError): runner.timing_schedule(index)

    def test_missing_repeat_or_historical_arm_rejected(self):
        with self.assertRaises(ValueError): runner.summarize_timing([phase(0),phase(1)], 2)
        p = [phase(i) for i in range(3)]; p[1] = seal(dict(p[1], arm='tail128_dynamic'))
        with self.assertRaises(ValueError): runner.summarize_timing(p, 2)

    def test_seed_layer_or_head_scope_drift_rejected(self):
        for update in (dict(dynamic_refresh_start_layer=4), dict(head_tokens=0), dict(tail_tokens=256),
                       dict(dynamic_document_swa=False), dict(native_yoco_swa_share_hidden=True)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                runner.verify_phase(seal(dict(phase(), **update)), index=2, arm='headtail128_link', timing=True, quality=False)

    def test_quality_is_natural_eos_separate_from_fixed_timing(self):
        p = phase(quality=True, tokens=5)
        runner.verify_phase(p, index=2, arm='headtail128_link', timing=False, quality=True)
        p['record']['natural_EOS'] = False
        with self.assertRaises(ValueError):
            runner.verify_phase(seal(p), index=2, arm='headtail128_link', timing=False, quality=True)

    def test_foreign_execution_or_input_cannot_mix_repeats(self):
        for field in ('execution_configuration_sha256','manifest_sha256','headtail_link_protocol_sha256'):
            p = [phase(i) for i in range(3)]; p[-1] = seal(dict(p[-1], **{field:'e'*64}))
            with self.subTest(field=field), self.assertRaises(ValueError): runner.summarize_timing(p, 2)


class HeadTailSmokeTests(unittest.TestCase):
    def fixture(self):
        p = phase(tokens=1)
        p.update(timing_valid=False, partial_smoke=True, empty_ep_qualification_sha256='e'*64,
                 controller=dict(dynamic_document_swa=True))
        p = seal(p)
        row = dict(index=2,input_ids_sha256=p['input_ids_sha256'],input_tokens=262194,documents=[{}]*8)
        qualification = dict(passed=True,identity_check_count=80,sha256='f'*64,empty_ep_qualification_sha256='e'*64)
        sparse = dict(passed=True,input_ids_sha256=row['input_ids_sha256'],
                      candidate_sha256=p['sha256'],qualification_sha256=qualification['sha256'],sha256='1'*64)
        return p,row,qualification,sparse

    def check(self, args, actual):
        reader = ModuleType('gpu_pilot.read_headtail_link_results')
        reader.documents = lambda row: []
        reader.audit_controller = lambda *args: actual
        with mock.patch.dict(sys.modules,{'gpu_pilot.read_headtail_link_results':reader}):
            return runner.verify_partial_smoke([args[0]],*args[1:])

    def test_smoke_requires_consumed_cached_kv_and_actual_refresh(self):
        args=self.fixture()
        actual=dict(ffn_bypassed_rows=20,consumed_cached_pairs=30,dynamic_refreshed_rows=10)
        result=self.check(args,actual)
        self.assertTrue(result['passed']);self.assertFalse(result['timing_valid'])
        for field in actual:
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.check(args,dict(actual,**{field:0}))

    def test_smoke_cannot_borrow_historical_or_other_candidate_gate(self):
        args=self.fixture();args[3]['candidate_sha256']='9'*64
        with self.assertRaisesRegex(ValueError,'own fresh'):
            self.check(args,dict(ffn_bypassed_rows=20,consumed_cached_pairs=30,dynamic_refreshed_rows=10))


if __name__ == '__main__': unittest.main()
