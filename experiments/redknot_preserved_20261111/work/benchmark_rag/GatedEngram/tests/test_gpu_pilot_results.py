"""CPU-only sealed fixture readout tests, not actual accelerator measurements."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import pilot_protocol as pp
from gpu_pilot import read_results as reader
from kv_cache_engram.core import ValidationError, digest
from test_gpu_pilot_protocol import cases, report, reports, scope


def seal(value):
    return pp.seal_record({key: item for key, item in value.items() if key != 'sha256'})


def write(directory, name, value):
    (directory/name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


class Fixture:
    def __init__(self, path, policy, *, selected=True):
        self.path, self.policy, self.selected = path, policy, selected
        self.identity = dict(checkpoint_index_sha256='a'*64, tokenizer_json_sha256='b'*64,
                             execution_configuration_sha256='c'*64)
        self.expected = []
        for index, case in enumerate(cases('calibration')+cases('evaluation')):
            self.expected.append(dict(case, index=index, sample_id=case['case_id'], question='Which item?',
                                      references=['Gold'], input_tokens=262180))
        self.preflight = seal(dict(identity=self.identity, manifest_sha256='f'*64, protocol=pp.pilot_protocol(),
                                   audit_cases=self.expected))
        write(path, 'preflight.json', self.preflight)
        calibration = reports('calibration', [policy])
        if not selected:
            event = calibration[0]['events'][0]
            event.update(candidate_absolute_max=.1, full_absolute_max=.1, bad_candidate_rows=1,
                         oracle_bad_rows=1, oracle_bad_reused_rows=1)
            calibration[0] = seal(calibration[0])
        self.frozen = pp.fit_shadow(calibration, cases('calibration'), scope())
        write(path, 'calibration.frozen.json', self.frozen)
        self.paired = {}

    def phase(self, row, suffix, mode, *, quality, gated_text='Wrong', ttft=100., truncated=False):
        name = f"case{row['index']}.{suffix}"
        shadow = report(row, [self.policy])
        hook = {key: deepcopy(shadow[key]) for key in ('complete', 'native_execution_declared', 'scope', 'oracle',
                    'input_sha256', 'request_id', 'events')}
        hook.update(schema='kv-cache-engram-native-hook-v1', mode=mode, phase=name,
            request_id=f"kce:{row['index']}:{name}:{row['sample_id']}",
            backend='deepinfer-sm90-native-bfloat16-kv-only-v1', full_q_attention_moe_preserved=True,
            coverage={layer: row['input_tokens'] for layer in pp.LAYERS}, input_tokens=row['input_tokens'])
        for event in hook['events']:
            event['phase'] = name
            event['request_id'] = hook['request_id']
        if mode == 'native':
            hook['events'] = []
        if mode == 'gated':
            hook['events'] = []
            if self.selected:
                candidates = 512-128-self.policy.probe_count
                hook['events'] = [dict(layer=3, document_sha256=row['documents'][0]['document_sha256'],
                    document_local_start=0, document_local_end=512, candidate_id=digest(asdict(self.policy)),
                    candidate_sha256=digest(asdict(self.policy)), request_id=hook['request_id'], phase=name,
                    input_sha256=row['input_sha256'], accepted=True, actual_reused_rows=candidates,
                    candidate_reusable_rows=candidates, probe_relative_max=0., probe_absolute_max=0.,
                    probe_count=self.policy.probe_count, mandatory_count=128,
                    oracle_measured=False, current_hybrid_hidden_probes=True)]
        hook['counts'] = {}
        for layer in pp.LAYERS:
            reused = sum(event['actual_reused_rows'] for event in hook['events'] if event['layer'] == layer)
            split = 8192 if mode == 'gated' and self.selected else 0
            hook['counts'][layer] = dict(q_rows=row['input_tokens'], native_fused_rows=row['input_tokens']-split,
                split_q_rows=split, online_kv_rows=row['input_tokens']-reused, reused_kv_rows=reused,
                probe_rows=self.policy.probe_count if split else 0, fallback_rows=0)
        text = gated_text if mode == 'gated' else 'Gold'
        rec = dict(success=True, rank=0, phase=name, input_ids_sha256=row['input_sha256'], text=text,
            request_id=hook['request_id'], sample_id=row['sample_id'], dataset=row['dataset'], input_tokens=row['input_tokens'],
            references=row['references'], output_ids=list(range(64 if truncated else 2)) if quality else list(range(32)),
            finish_reason='length' if not quality or truncated else 'stop', natural_EOS=quality,
            truncated=truncated if quality else False, online_ttft_s=ttft, online_elapsed_s=ttft+2,
            online_prepare_to_submit_s=.2, ttft_s=ttft-.2, elapsed_s=ttft+1.8,
            submit_s=1., finish_s=ttft+2.8, **reader._scores(text, row['references']))
        rec['output_ids_sha256'] = digest(rec['output_ids'])
        value = seal(dict(schema='kv-cache-engram-native-phase-v1', name=name, mode=mode,
            index=row['index'], dataset=row['dataset'], split='heldout', manifest_sha256='f'*64,
            execution_configuration_sha256='c'*64, input_ids_sha256=row['input_sha256'], record=rec, hook=hook,
            timing_valid=not quality, quality_eligible=quality, online_gate_and_plan_included=True,
            offline_build_included=False, online_global_window_s=ttft+3))
        write(self.path, name+'.json', value)
        return value

    def case(self, index, *, gated_text='Wrong', gated_ttft=80., truncated=False):
        row = self.expected[index]
        reference = self.phase(row, 'reference_shadow', 'shadow_full', quality=True, truncated=truncated)
        gated = self.phase(row, 'gated_quality', 'gated', quality=True, gated_text=gated_text, truncated=truncated)
        measured = []
        for repeat in range(3):
            for arm in (reader.ARMS if (index+index//2+repeat) % 2 == 0 else reader.ARMS[::-1]):
                raw = self.phase(row, f'{arm}_measure{repeat}', 'native' if arm == 'recomputed' else 'gated',
                                 quality=False, ttft=100. if arm == 'recomputed' else gated_ttft)
                measured.append(seal(dict(raw, arm=arm, repeat=repeat, raw_phase_sha256=raw['sha256'])))
        timing = reader._timing(dict(recomputed=[100.]*3, gated=[gated_ttft]*3))
        value = seal(dict(index=index, dataset=row['dataset'], sample_id=row['sample_id'], references=row['references'],
            question=row['question'], input_tokens=row['input_tokens'], reference=reference, gated=gated,
            timing_phases=measured, timing_summary=timing, offline_build_s=123., calibration_sha256=self.frozen['sha256'],
            all_negative_results_retained=True, no_native_Q_attention_MoE_skipped=True))
        self.paired[index] = value
        write(self.path, f'case{index}.paired.json', value)
        return value

    def update_raw_and_pair(self, index, suffix, mutate):
        name = f'case{index}.{suffix}'
        raw = json.loads((self.path/(name+'.json')).read_text())
        mutate(raw)
        raw = seal(raw)
        write(self.path, name+'.json', raw)
        pair = deepcopy(self.paired[index])
        if suffix in ('reference_shadow', 'gated_quality'):
            pair['reference' if suffix == 'reference_shadow' else 'gated'] = raw
        else:
            for pos, nested in enumerate(pair['timing_phases']):
                if nested['name'] == name:
                    pair['timing_phases'][pos] = seal(dict(raw, arm=nested['arm'], repeat=nested['repeat'],
                                                         raw_phase_sha256=raw['sha256']))
        self.paired[index] = seal(pair)
        write(self.path, f'case{index}.paired.json', self.paired[index])


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.policy = pp.candidate_policies()[0]
        self.layer_patch = patch.object(pp, 'LAYERS', (3,))
        self.grid_patch = patch.object(pp, 'candidate_policies', return_value=(self.policy,))
        self.layer_patch.start(); self.grid_patch.start()
        self.addCleanup(self.layer_patch.stop); self.addCleanup(self.grid_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.fixture = Fixture(self.path, self.policy)

    def test_first_complete_pair_is_available_before_batch_finishes(self):
        self.fixture.case(2)
        result = reader.read_results(self.path)
        self.assertEqual(result['status'], 'running')
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['planned'], 4)
        self.assertEqual(result['by_dataset']['hotpotqa']['completed'], 1)
        self.assertEqual(result['by_dataset']['2wikimqa']['completed'], 0)
        summary = result['global_summary']
        self.assertEqual(summary['loss_percentage_points'], dict(answer_em=100., answer_token_f1=100.))
        self.assertEqual(summary['c1_online_timing']['ttft_speedup'], 1.25)
        self.assertEqual(summary['c1_online_timing']['ttft_reduction_s'], 20.)
        self.assertAlmostEqual(summary['c1_online_timing']['ttft_reduction_percent'], 20.)
        self.assertEqual(summary['baseline_correct_to_gated_wrong'], 1)
        self.assertEqual(result['cases'][0]['quality']['gated']['text'], 'Wrong')
        self.assertEqual(result['calibration_cases_excluded_from_quality'], 2)

    def test_completed_four_cases_reports_each_dataset_two(self):
        for index in range(2, 6):
            self.fixture.case(index, gated_text='Gold' if index == 3 else 'Wrong')
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 4)
        self.assertEqual([result['by_dataset'][d]['completed'] for d in pp.DATASETS], [2, 2])
        self.assertEqual(result['global_summary']['quality']['gated']['answer_em'], .25)
        self.assertEqual(len(result['global_summary']['c1_online_timing']['gated']['online_ttft_samples_s']), 12)

    def test_negative_speed_and_length_results_are_retained(self):
        self.fixture.case(2, gated_ttft=120., truncated=True)
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 1)
        self.assertLess(result['global_summary']['c1_online_timing']['ttft_speedup'], 1)
        self.assertEqual(result['global_summary']['length_counts'], dict(recomputed=1, gated=1))
        self.assertEqual(result['cases'][0]['quality']['gated']['output_tokens'], 64)

    def test_partial_case_with_missing_raw_phase_is_excluded_and_visible(self):
        self.fixture.case(2)
        (self.path/'case2.gated_measure2.json').unlink()
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['status'], 'evidence_invalid')
        self.assertIsNone(result['global_summary'])
        self.assertIn('missing', result['audit_errors'][0]['error'])

    def test_within_dataset_start_order_is_balanced_and_old_order_rejected(self):
        starts = {}
        for index in range(2, 6):
            pair = self.fixture.case(index)
            starts[index] = pair['timing_phases'][0]['arm']
        self.assertNotEqual(starts[2], starts[4])
        self.assertNotEqual(starts[3], starts[5])
        pair = deepcopy(self.fixture.paired[2])
        pair['timing_phases'][0], pair['timing_phases'][1] = pair['timing_phases'][1], pair['timing_phases'][0]
        write(self.path, 'case2.paired.json', seal(pair))
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 3)
        self.assertIn('order', result['audit_errors'][0]['error'])

    def test_raw_or_nested_seal_tamper_cannot_enter_metrics(self):
        self.fixture.case(2)
        path = self.path/'case2.gated_measure1.json'
        raw = json.loads(path.read_text())
        raw['record']['online_ttft_s'] = .1
        write(self.path, path.name, raw)
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 0)
        self.assertIn('seal', result['audit_errors'][0]['error'])

    def test_nested_phase_cannot_substitute_for_missing_raw_evidence(self):
        pair = self.fixture.case(2)
        pair['timing_phases'][0]['record']['online_ttft_s'] = 1.
        pair['timing_phases'][0] = seal(pair['timing_phases'][0])
        write(self.path, 'case2.paired.json', seal(pair))
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 0)
        self.assertIn('differs from raw', result['audit_errors'][0]['error'])

    def test_fixed_gold_and_scoring_are_independently_verified(self):
        self.fixture.case(2)
        self.fixture.update_raw_and_pair(2, 'gated_quality', lambda raw: raw['record'].update(answer_em=1.))
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 0)
        self.assertIn('fixed gold', result['audit_errors'][0]['error'])

    def test_request_binding_and_online_preparation_cannot_be_substituted(self):
        for fault in ('request', 'online_time'):
            with self.subTest(fault=fault):
                self.fixture.case(2)
                def mutate(raw):
                    if fault == 'request':
                        raw['record']['request_id'] = 'wrong-request'
                    else:
                        raw['record']['online_prepare_to_submit_s'] += .1
                self.fixture.update_raw_and_pair(2, 'gated_measure0', mutate)
                result = reader.read_results(self.path)
                self.assertEqual(result['completed'], 0)
                self.assertEqual(result['status'], 'evidence_invalid')

    def test_runtime_counters_and_frozen_gate_decisions_are_checked(self):
        for fault in ('counters', 'threshold', 'native'):
            with self.subTest(fault=fault):
                self.fixture.case(2)
                def mutate(raw):
                    if fault == 'counters':
                        raw['hook']['counts']['3']['online_kv_rows'] += 1
                    elif fault == 'threshold':
                        raw['hook']['events'][0]['probe_absolute_max'] = 1.
                    else:
                        raw['hook']['native_execution_declared'] = False
                self.fixture.update_raw_and_pair(2, 'gated_quality', mutate)
                result = reader.read_results(self.path)
                self.assertEqual(result['completed'], 0)
                self.assertEqual(result['status'], 'evidence_invalid')

    def test_all_native_fallback_is_not_reported_as_reuse(self):
        self.fixture = Fixture(self.path, self.policy, selected=False)
        self.fixture.case(2, gated_text='Gold', gated_ttft=102.)
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 1)
        counts = result['global_summary']['gated_formal_row_accounting']
        self.assertEqual(counts['actual_reused_kv_rows'], 0)
        self.assertTrue(counts['all_full_recompute'])
        self.assertFalse(result['cases'][0]['shadow_full_audit']['any_selected_layer'])

    def test_actual_reuse_denominator_is_only_targeted_layer_rows(self):
        self.fixture.case(2)
        counts = reader.read_results(self.path)['global_summary']['gated_formal_row_accounting']
        self.assertEqual(counts['actual_reused_kv_rows'], 3*(512-128-32))
        self.assertEqual(counts['online_kv_rows']+counts['actual_reused_kv_rows'], 3*262180)
        self.assertTrue(counts['not_full_model_compute_fraction'])

    def test_shadow_false_accept_is_visible_without_filtering_bad_quality(self):
        self.fixture.case(2)
        def mutate(raw):
            raw['hook']['events'][0].update(candidate_relative_max=.2, full_relative_max=.2,
                bad_candidate_rows=1, oracle_bad_rows=1, oracle_bad_reused_rows=1)
        self.fixture.update_raw_and_pair(2, 'reference_shadow', mutate)
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['global_summary']['false_accept_blocks_shadow'], 1)
        self.assertFalse(result['cases'][0]['shadow_full_audit']['selected_candidate_kv_validation_pass'])

    def test_failure_preserves_only_completed_case_statistics(self):
        self.fixture.case(2)
        write(self.path, 'failure.rank0.json', seal(dict(schema='kv-cache-engram-native-gpu-pilot-run-v1',
            status='failed', rank=0, error='fixture failure', completed_indices=[2])))
        result = reader.read_results(self.path)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['failures'][0]['error'], 'fixture failure')

    def test_cli_output_is_exclusive_and_reading_does_not_change_inputs(self):
        self.fixture.case(2)
        before = {p.name: p.read_bytes() for p in self.path.iterdir()}
        output = self.path/'readout.json'
        self.assertEqual(reader.main([str(self.path), '--output', str(output)]), 0)
        with self.assertRaises(FileExistsError):
            reader.main([str(self.path), '--output', str(output)])
        self.assertEqual(before, {name: (self.path/name).read_bytes() for name in before})

    def test_terminal_complete_requires_all_paired_raw_closures(self):
        for index in range(2, 6):
            self.fixture.case(index)
        audit = pp.validate_shadow(self.fixture.frozen, reports('evaluation', [self.policy]), cases('evaluation'))
        audit['evaluation_report_sha256'] = [case['shadow_full_audit']['source_shadow_report_sha256']
                                             for case in reader.read_results(self.path)['cases']]
        audit = seal(audit)
        run = seal(dict(schema='kv-cache-engram-native-gpu-pilot-run-v1', status='complete',
            identity=self.fixture.identity, manifest_sha256='f'*64, calibration=self.fixture.frozen,
            heldout_shadow_audit=audit, cases=list(self.fixture.paired.values())))
        write(self.path, 'run.json', run)
        self.assertEqual(reader.read_results(self.path)['status'], 'complete')
        (self.path/'case5.paired.json').unlink()
        with self.assertRaises(ValidationError):
            reader.read_results(self.path)

    def test_foreign_preflight_or_frozen_calibration_fails_closed(self):
        self.fixture.frozen['scope']['execution_sha256'] = 'e'*64
        write(self.path, 'calibration.frozen.json', seal(self.fixture.frozen))
        with self.assertRaises(ValidationError):
            reader.read_results(self.path)


if __name__ == '__main__':
    unittest.main()
