"""Synthetic CPU reader contracts, not model/GPU validation evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import pilot_protocol as pp
from gpu_pilot import read_results as reader
from gpu_pilot import sensitivity_protocol as sp
from kv_cache_engram.core import ValidationError, digest
from test_gpu_pilot_results import Fixture, seal, write
from test_sensitivity_protocol import observer


class SensitivityFixture(Fixture):
    def __init__(self, path, policy):
        super().__init__(path, policy, selected=False)
        protocol = sp.sensitivity_protocol()
        self.identity['sensitivity_protocol_sha256'] = protocol['sha256']
        self.preflight = seal(dict(self.preflight, identity=self.identity, sensitivity_protocol=protocol))
        write(path, 'preflight.json', self.preflight)
        self.qualification()
        reports, sensitivity = [], []
        for row in self.expected[:2]:
            raw = self.phase(row, 'reference_shadow', 'shadow_full', quality=True)
            raw['split'] = 'calibration'
            obs = observer(row)
            obs.update(input_tokens=row['input_tokens'], phase=raw['name'], request_id=raw['record']['request_id'],
                       coverage={str(layer): row['input_tokens'] for layer in sp.LAYERS})
            for record in obs['records']:
                record['end'] = row['input_tokens']
                record['attention_update']['rows'] = row['input_tokens']
                for head in record['heads']:
                    head['rows'] = row['input_tokens']
            raw.update(sensitivity=seal(obs), diagnostic_phase_wall_s=2., diagnostic_wall_not_online_ttft=True)
            raw = seal(raw)
            write(path, raw['name']+'.json', raw)
            wrapped = reader._shadow_wrapper(raw, row, reader._scope(self.preflight))
            reports.append(wrapped)
            write(path, f"case{row['index']}.calibration.json", seal(dict(reference_sha256=raw['sha256'],
                report=wrapped, offline_build_s=1., raw_pair_sha256=digest(['raw', row['index']]))))
            note = sp.wrap_phase(raw, reader._projection(row), manifest_sha256='f'*64, execution_sha256='c'*64)
            sensitivity.append(note)
            write(path, f"case{row['index']}.sensitivity.json", note)
        self.numerical = pp.fit_shadow(reports, [reader._projection(row) for row in self.expected[:2]],
                                      reader._scope(self.preflight))
        self.admission = sp.freeze_admission_plan(sensitivity,
            [reader._projection(row) for row in self.expected[:2]], numerical_calibration_sha256=self.numerical['sha256'])
        effective = deepcopy(self.numerical)
        for layer in self.admission['mandatory_full_layers']:
            if str(layer) in effective['layers']:
                effective['layers'][str(layer)].update(selected=None, mode='full_recompute', sensitivity_protected=True)
        self.frozen = seal(dict(effective, numerical_calibration_sha256=self.numerical['sha256'],
                               sensitivity_admission_plan_sha256=self.admission['sha256']))
        write(path, 'calibration.numerical.json', self.numerical)
        write(path, 'sensitivity.admission.json', self.admission)
        write(path, 'calibration.frozen.json', self.frozen)

    def qualification(self):
        row = self.expected[0]
        raw = self.phase(row, 'qualification_unused', 'native', quality=False)
        (self.path/(raw['name']+'.json')).unlink()
        name = 'qualification_split_identity'
        rid = f"kce:0:{name}:{row['sample_id']}"
        raw.update(name=name, mode='split_identity', split='calibration', timing_valid=False)
        raw['record'].update(phase=name, request_id=rid, output_ids=[1], output_ids_sha256=digest([1]))
        raw['hook'].update(phase=name, request_id=rid, mode='split_identity', identity_exact=True,
            numerical_equivalence_claim=True, qualification=[dict(layer=layer, start=0,
                end=row['input_tokens'], exact=True, errors={key: 0. for key in
                    ('q', 'kv_raw', 'kv_canonical', 'gather32', 'gather128', 'post_rope_cast',
                     'post_rope_gather32', 'post_rope_gather128', 'post_rope_gather384', 'post_rope_gather512')})
                for layer in pp.LAYERS])
        self.qualified = seal(raw)
        write(self.path, name+'.json', self.qualified)
        write(self.path, 'qualification.json', seal(dict(passed=True,
            phase_sha256=self.qualified['sha256'], quality_claim=False)))

    def case(self, index, **kwargs):
        value = super().case(index, **kwargs)
        value = seal(dict(value, sensitivity_admission_plan_sha256=self.admission['sha256']))
        self.paired[index] = value
        write(self.path, f'case{index}.paired.json', value)
        return value

    def terminal(self):
        reports = []
        for row in self.expected[2:]:
            raw = json.loads((self.path/f"case{row['index']}.reference_shadow.json").read_text())
            reports.append(reader._shadow_wrapper(raw, row, reader._scope(self.preflight)))
        audit = pp.validate_shadow(self.frozen, reports, [reader._projection(row) for row in self.expected[2:]])
        value = seal(dict(schema='kv-cache-engram-native-gpu-pilot-run-v1', status='complete',
            identity=self.identity, manifest_sha256='f'*64, calibration=self.frozen,
            sensitivity_admission=self.admission, heldout_shadow_audit=audit, cases=list(self.paired.values())))
        write(self.path, 'run.json', value)
        return value


class SensitivityResultsTests(unittest.TestCase):
    def setUp(self):
        self.policy = pp.candidate_policies()[0]
        # All fourteen observer layers are still checked; only numeric-grid event
        # fixtures are narrowed to one layer to keep filesystem tests CPU-small.
        for patcher in (patch.object(pp, 'LAYERS', (3,)),
                        patch.object(pp, 'candidate_policies', return_value=(self.policy,))):
            patcher.start(); self.addCleanup(patcher.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.fixture = SensitivityFixture(self.path, self.policy)

    def mutate(self, filename, callback):
        value = json.loads((self.path/filename).read_text())
        callback(value)
        write(self.path, filename, seal(value))

    def test_valid_new_mode_reconstructs_all_permissions_from_raw_calibration(self):
        self.fixture.case(2, gated_text='Gold', gated_ttft=103.)
        result = reader.read_results(self.path)
        self.assertTrue(result['sensitivity_enabled'])
        self.assertTrue(result['numerical_qualification']['passed'])
        self.assertEqual(result['sensitivity_admission'], self.fixture.admission)
        self.assertEqual(result['completed'], 1)
        self.assertTrue(result['global_summary']['gated_formal_row_accounting']['all_full_recompute'])
        self.assertLess(result['global_summary']['c1_online_timing']['ttft_speedup'], 1)
        self.assertEqual(result['by_dataset']['hotpotqa']['completed'], 1)
        self.assertEqual(result['calibration_cases_excluded_from_quality'], 2)

    def test_old_mode_still_accepts_complete_pair_without_sensitivity_files(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Fixture(Path(directory), self.policy, selected=False)
            old.case(2)
            self.assertEqual(reader.read_results(directory)['completed'], 1)

    def test_qualification_is_required_even_if_passed_closure_exists(self):
        self.fixture.case(2)
        (self.path/'qualification_split_identity.json').unlink()
        with self.assertRaisesRegex(ValidationError, 'missing'):
            reader.read_results(self.path)

    def test_qualification_rejects_nonzero_incomplete_or_false_identity(self):
        original = deepcopy(self.fixture.qualified)
        for fault in ('nonzero', 'missingcheck', 'coverage', 'identity', 'diagnostic'):
            with self.subTest(fault=fault):
                raw = deepcopy(original)
                if fault == 'nonzero': raw['hook']['qualification'][0]['errors']['gather128'] = .001
                elif fault == 'missingcheck': raw['hook']['qualification'][0]['errors'].pop('post_rope_cast')
                elif fault == 'coverage': raw['hook']['qualification'][0]['end'] -= 1
                elif fault == 'identity': raw['hook']['identity_exact'] = False
                else: raw['sensitivity'] = {}
                raw = seal(raw)
                write(self.path, 'qualification_split_identity.json', raw)
                write(self.path, 'qualification.json', seal(dict(passed=True, phase_sha256=raw['sha256'], quality_claim=False)))
                with self.assertRaises(ValidationError): reader.read_results(self.path)

    def test_numeric_summary_cannot_replace_full_raw_events(self):
        self.mutate('case0.reference_shadow.json', lambda raw: raw['hook']['events'].pop())
        with self.assertRaises(ValidationError): reader.read_results(self.path)

    def test_saved_calibration_note_and_sensitivity_note_are_raw_bound(self):
        for filename, field in (('case0.calibration.json', 'reference_sha256'),
                                ('case0.sensitivity.json', 'source_phase_sha256')):
            original = (self.path/filename).read_bytes()
            self.mutate(filename, lambda value: value.update({field: '0'*64}))
            with self.assertRaises(ValidationError): reader.read_results(self.path)
            (self.path/filename).write_bytes(original)

    def test_numerical_candidate_statistics_recomputed_before_protection(self):
        self.mutate('calibration.numerical.json', lambda value:
            value['layers']['3']['candidates'][0]['summary'].update(would_reuse_rows=1))
        with self.assertRaisesRegex(ValidationError, 'independently reconstructed'):
            reader.read_results(self.path)

    def test_admission_cannot_unprotect_or_pick_layers_without_calibration(self):
        self.mutate('sensitivity.admission.json', lambda value: value.update(mandatory_full_layers=[]))
        with self.assertRaisesRegex(ValidationError, 'full calibration evidence'):
            reader.read_results(self.path)

    def test_effective_calibration_cannot_reselect_protected_layer(self):
        self.mutate('calibration.frozen.json', lambda value:
            value['layers']['3'].update(selected=self.fixture.numerical['layers']['3']['selected'], mode='pilot_gated_kv'))
        with self.assertRaisesRegex(ValidationError, 'protected numerical calibration'):
            reader.read_results(self.path)

    def test_heldout_or_timed_diagnostics_never_enter_paired_metrics(self):
        for suffix in ('reference_shadow', 'gated_quality', 'gated_measure1'):
            self.fixture.case(2)
            self.fixture.update_raw_and_pair(2, suffix, lambda raw: raw.update(sensitivity={}))
            result = reader.read_results(self.path)
            self.assertEqual(result['completed'], 0)
            self.assertIn('diagnostics forbidden', result['audit_errors'][0]['error'])

    def test_paired_case_requires_frozen_plan_binding(self):
        self.fixture.case(2)
        self.mutate('case2.paired.json', lambda value: value.update(sensitivity_admission_plan_sha256='0'*64))
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'], 0)
        self.assertIn('admission binding', result['audit_errors'][0]['error'])

    def test_terminal_plan_binding_and_four_heldout_closures(self):
        for index in range(2, 6): self.fixture.case(index)
        self.fixture.terminal()
        self.assertEqual(reader.read_results(self.path)['status'], 'complete')
        self.mutate('run.json', lambda value: value.update(sensitivity_admission=None))
        with self.assertRaisesRegex(ValidationError, 'terminal sensitivity admission'):
            reader.read_results(self.path)

    def test_protocol_identity_cannot_be_dropped_to_bypass_new_audit(self):
        self.mutate('preflight.json', lambda value: value.pop('sensitivity_protocol'))
        with self.assertRaisesRegex(ValidationError, 'protocol/execution identity'):
            reader.read_results(self.path)


if __name__ == '__main__':
    unittest.main()
