"""CPU-only runner wiring and immutable diagnostic evidence contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import run_repair_pilot as runner
from gpu_pilot import repair_protocol as rp
from gpu_pilot import pilot_protocol as pp
from test_pilot_runner import valid_args


def fixtures():
    rows = []
    for index in range(2, 6):
        docs = [dict(document_id=f'{index*10+j:064x}', token_ids_sha256=f'{index*10+j:064x}',
                     tokens=32768, input_start=40+j*32768) for j in range(8)]
        rows.append(dict(index=index, dataset=pp.DATASETS[index%2], split='heldout',
            sample_id=f'cpu-case-{index}', input_ids_sha256=f'{index:064x}', input_tokens=262200,
            question='CPU fixture, not quality evidence?', references=['fixture'], documents=docs,
            context_segments=[dict(kind='primary', context_sha256=f'{index+100:064x}')]))
    base = SimpleNamespace(document_specs=lambda r: [dict(d, token_ids=[1]*32768) for d in r['documents']])
    return base, dict(cohort=rows, sha256='a'*64)


class RepairRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root/'repair.json'
        self.path.write_text(json.dumps(rp.repair_protocol()))
        self.args = valid_args(repair_protocol=self.path, sensitivity_protocol=self.root/'sensitivity.json',
            experiment_root=self.root, out_dir=self.root/'not-created', manifest=self.root/'manifest.json')
        self.base = SimpleNamespace(read_json=lambda p: json.loads(p.read_text()))

    def test_parse_keeps_repair_flag_out_of_legacy_parser(self):
        with mock.patch.object(runner.old, 'parse_args', return_value=SimpleNamespace()) as parse:
            result = runner.parse_args(['--repair-protocol', str(self.path), '--legacy', 'value'])
        self.assertEqual(result.repair_protocol, self.path)
        parse.assert_called_once_with(['--legacy', 'value'])

    def test_only_frozen_protocol_and_required_original_binding(self):
        value = runner.validate_repair(self.args, self.base)
        self.assertEqual(value, rp.repair_protocol())
        with self.assertRaises(ValueError):
            runner.validate_repair(self.args, self.base, {'repair_protocol_sha256':'0'*64})
        self.args.sensitivity_protocol = None
        with self.assertRaises(ValueError):
            runner.validate_repair(self.args, self.base)

    def test_changed_protocol_even_resealed_is_rejected(self):
        value = rp.repair_protocol(); value.pop('sha256'); value['performance_measured'] = True
        self.path.write_text(json.dumps(pp.seal_record(value)))
        with self.assertRaises(ValueError):
            runner.validate_repair(self.args, self.base)

    def test_audit_cases_explicit_exposure_four_questions_exact_geometry(self):
        base, manifest = fixtures()
        result = runner.audit_cases(base, manifest)
        self.assertEqual([r['index'] for r in result], [2,3,4,5])
        for row in result:
            self.assertTrue(row['previously_exposed_exploratory_cohort'])
            self.assertEqual(sum(d['tokens'] for d in row['document_spans']), 262144)
        manifest['cohort'][0]['documents'][0]['input_start'] += 1
        with self.assertRaises(ValueError):
            runner.audit_cases(base, manifest)

    def test_incomplete_or_duplicate_cohort_rejected(self):
        base, manifest = fixtures()
        for rows in (manifest['cohort'][:-1], [manifest['cohort'][0]]*4):
            with self.assertRaises(ValueError):
                runner.audit_cases(base, dict(manifest, cohort=rows))

    def test_triple_requires_same_plan_input_and_all_three_arms(self):
        _, manifest = fixtures(); row = manifest['cohort'][0]
        identity={'execution_configuration_sha256':'b'*64}; plan={'sha256':'c'*64}
        phases = [pp.seal_record(dict(arm=arm, name=f'case2.{arm}',
            plan_sha256=plan['sha256'], input_ids_sha256=row['input_ids_sha256'],
            timing_valid=False, quality_eligible=True)) for arm in rp.ARMS]
        triple=runner.build_triple(row,manifest,identity,rp.repair_protocol(),plan,phases,1.0)
        self.assertEqual(list(triple['phase_sha256']),list(rp.ARMS))
        self.assertTrue(triple['formal_admission_unchanged']); self.assertTrue(triple['no_ttft_claim'])
        for bad in (phases[:-1], list(reversed(phases))):
            with self.assertRaises(ValueError):
                runner.build_triple(row,manifest,identity,rp.repair_protocol(),plan,bad,1.0)
        for key,value in [('plan_sha256','f'*64),('timing_valid',True),('input_ids_sha256','f'*64)]:
            bad=copy.deepcopy(phases); item=bad[1]; item.pop('sha256'); item[key]=value
            bad[1]=pp.seal_record(item)
            with self.assertRaises(ValueError):
                runner.build_triple(row,manifest,identity,rp.repair_protocol(),plan,bad,1.0)

    def test_check_only_no_launch_release_output_or_accelerator(self):
        fixture_base, manifest = fixtures()
        identity=dict(fixed_env={'CPU':'fixture'})
        base=SimpleNamespace(preflight=mock.Mock(return_value=(manifest,identity)),
            read_json=mock.Mock(return_value=pp.pilot_protocol()),
            source_identity=mock.Mock(return_value={'source':'cpu'}),digest=pp.digest,
            document_specs=fixture_base.document_specs,
            wait_for_release=mock.Mock(side_effect=AssertionError('no release')),
            write_json=mock.Mock(side_effect=AssertionError('no output')))
        native=SimpleNamespace(gpu_runtime_imported=mock.Mock(return_value=False))
        profile=SimpleNamespace(identity=lambda args:{'profile':'cpu'})
        with mock.patch.object(runner,'parse_args',return_value=self.args), \
             mock.patch.object(runner.old,'load_runtime',return_value=(base,native,lambda **kw:profile,
                lambda *a,**kw:contextlib.nullcontext())), \
             mock.patch.object(runner.old,'validate_inputs'), \
             mock.patch.object(runner.old,'validate_sensitivity_inputs',return_value={'sha256':'d'*64}), \
             mock.patch.object(runner,'validate_repair',return_value=rp.repair_protocol()), \
             mock.patch.object(runner.old,'source_hashes',return_value={'source':'e'*64}), \
             mock.patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=9*1024**3)), \
             mock.patch.object(runner,'worker',side_effect=AssertionError('no worker')) as worker, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main([]),0)
        value=json.loads(output.getvalue())
        self.assertFalse(value['gpu_initialized']); self.assertTrue(value['no_ttft_claim'])
        self.assertEqual([r['index'] for r in value['cases']],[2,3,4,5])
        self.assertFalse(self.args.out_dir.exists())
        base.wait_for_release.assert_not_called(); base.write_json.assert_not_called(); worker.assert_not_called()

    def test_freeze_verifies_both_legacy_and_new_protocol(self):
        with mock.patch.object(runner.old,'verify_frozen') as legacy, \
             mock.patch.object(runner,'validate_repair') as repair:
            runner.verify_frozen(self.args,{}, {'id':1}, None, self.base,native_source=True)
        legacy.assert_called_once_with(self.args,{}, {'id':1}, None,self.base,native_source=True)
        repair.assert_called_once_with(self.args,self.base,{'id':1})


if __name__ == '__main__':
    unittest.main()
