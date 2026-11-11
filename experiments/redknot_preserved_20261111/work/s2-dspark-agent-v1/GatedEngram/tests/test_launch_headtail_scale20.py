"""Finite lifecycle, source and input binding for the twenty-case comparison."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import headtail_scale20_protocol as protocol
from gpu_pilot import launch_headtail_scale20_pilot as launch


class HeadTailScale20LauncherTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.stage=self.root/'experiment/GatedEngram'
        self.native=self.stage.parent/'NativeSnapshot';(self.root/'results').mkdir()
        names=('gpu_pilot/run_pilot.py','gpu_pilot/launch_pilot.py','gpu_pilot/headtail_scale20_protocol.py',
            'gpu_pilot/headtail_scale20_inputs.py','gpu_pilot/run_headtail_scale20_pilot.py',
            'gpu_pilot/read_headtail_scale20_results.py','gpu_pilot/launch_headtail_scale20_pilot.py',
            'kv_cache_engram/headtail_comparison.py','kv_cache_engram/headtail_link.py',
            'kv_cache_engram/headtail_link_plan.py','kv_cache_engram/early_token_diagnostics.py',
            'tests/test_launch_headtail_scale20.py')
        for name in names:
            p=self.stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
        for name in ('setup.py','README.md','deepinfer/model_executor/models/v4_modules/attn.py'):
            p=self.native/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
        self.manifest=self.stage/'manifest.json';self.inputs=self.stage/'input-protocol.json'
        self.comparison=self.stage/'headtail-scale20-protocol.json'
        self.write(self.inputs,dict(schema='headtail-scale20-input-protocol-v1'))
        data=dict(schema='redknot-deepinfer-real-rag-manifest-v1',artifact_kind='headtail_scale20_inputs_v1',
            input_protocol_sha256=launch.lifecycle.read_sealed(self.inputs)['sha256'])
        self.write(self.manifest,data)
        patch=mock.patch.object(protocol,'INPUT_MANIFEST_SHA256',launch.lifecycle.read_sealed(self.manifest)['sha256'])
        patch.start();self.addCleanup(patch.stop)
        self.comparison.write_bytes(launch.lifecycle.canonical(protocol.headtail_scale20_protocol()))
        for name in launch.INPUT_COMPANIONS:self.write(self.stage/name,dict(schema='fixture',name=name))
        self.previous=dict(name=launch.PREDECESSOR,previous_workers_exited=True,previous_scores_reused=False)

    def write(self,path,value):path.write_bytes(launch.lifecycle.canonical(launch.lifecycle.seal(value)))

    def proof(self):
        families=('test_headtail_link_plan','test_headtail_comparison','test_headtail_scale20_runner',
            'test_headtail_scale20_results','test_launch_headtail_scale20','test_headtail_scale20_inputs',
            'test_index_hit20_empty_ep_qualification','test_early_token_diagnostics')
        ids=[f'tests.{f}.Fixture.test_required' for f in families]
        ids += [f'tests.fixture.Fixture.test_{i}' for i in range(1000-len(ids))]
        return dict(schema='kv-cache-engram-headtail-scale20-cpu-proof-v1',successful=True,
            cuda_initialized=False,cuda_init_guard_installed=True,failures=0,errors=0,skipped=0,
            tests=len(ids),test_ids=ids,source_sha256=launch.lifecycle.source_files(self.stage))

    def plan(self):
        self.write(self.stage/'HEADTAIL_SCALE20_CPU_TESTS-v1.json',self.proof())
        with mock.patch.object(launch,'predecessor_evidence',return_value=self.previous):
            return launch.build_plan(self.root,self.stage,self.manifest,self.inputs,self.comparison)

    def verify(self,value):
        with mock.patch.object(launch,'predecessor_evidence',return_value=self.previous):launch.verify_plan(value)

    def test_original_supervisor_has_24hour_backstop_and_new_runner(self):
        p=self.plan();self.verify(p);args=p['argv']
        self.assertEqual(args[args.index('--timeout')+1],'86400')
        self.assertEqual(args[args.index('--prepare-timeout')+1],'300')
        self.assertEqual(p['evaluation_indices'],list(range(2,22)))
        self.assertEqual(p['evaluation_arms'],list(protocol.ARMS))
        self.assertIsNone(p['sensitivity_protocol']);self.assertNotIn('--sensitivity-protocol',args)
        self.assertEqual(args[args.index('--repo')+1],str(self.native))
        self.assertTrue(p['fresh_all_twenty_cases_all_three_arms'])

    def test_timeout_and_full_twenty_contract_cannot_change_with_resealing(self):
        p=self.plan()
        for update in (dict(timeout=5400),dict(timeout=0),dict(completion_case_count=4),
                       dict(evaluation_indices=[2,3,4,5]),dict(evaluation_arms=['headtail128_link']),
                       dict(previous_ninety_minute_limit_cancelled=False)):
            bad=launch.lifecycle.seal({**{k:v for k,v in p.items() if k!='sha256'},**update})
            with self.subTest(update=update),self.assertRaises(ValueError):self.verify(bad)
        bad=copy.deepcopy(p);bad['argv'][bad['argv'].index('--timeout')+1]='5400'
        bad=launch.lifecycle.seal({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaises(ValueError):self.verify(bad)

    def test_companion_and_native_sources_are_frozen(self):
        p=self.plan();(self.stage/'source-plan.json').write_text('{}')
        with self.assertRaises(ValueError):self.verify(p)

    def test_cpu_proof_requires_new_families_no_cuda_or_skips(self):
        p=self.plan()
        for update in (dict(skipped=1),dict(cuda_initialized=True),dict(tests=999),dict(source_sha256={})):
            proof=self.proof();proof.update(update);self.write(Path(p['cpu_proof_path']),proof)
            with self.subTest(update=update),self.assertRaises(ValueError):launch.verify_cpu_proof(p)

    def test_check_only_never_launches_or_queries_holder(self):
        p=self.plan()
        with mock.patch.object(launch,'build_plan',return_value=p), \
             mock.patch.object(launch,'predecessor_evidence',return_value=self.previous), \
             mock.patch.object(launch,'run_plan',side_effect=AssertionError('launch forbidden')), \
             mock.patch.object(launch.lifecycle,'holder_snapshot',side_effect=AssertionError('holder forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(['--check-only']),0)

    def test_diagnostic_test_family_is_required(self):
        p=self.plan();proof=self.proof()
        proof['test_ids']=[name.replace('test_early_token_diagnostics.', 'test_unrelated.') for name in proof['test_ids']]
        self.write(Path(p['cpu_proof_path']),proof)
        with self.assertRaises(ValueError):launch.verify_cpu_proof(p)

    def test_previous_gpu_completion_does_not_replace_workflow_terminal(self):
        old_stage=self.root/'DeepInfer-PR148/experiments/old/GatedEngram'
        work=old_stage.parent/'Workflow';work.mkdir(parents=True)
        self.write(self.root/'results'/(launch.PREDECESSOR+'.launch-plan.json'),dict(schema=launch.lifecycle.PLAN_SCHEMA,name=launch.PREDECESSOR,
            root=str(self.root),stage_dir=str(old_stage),stage_sources={
                'gpu_pilot/run_headtail_link_pilot.py':launch.PREDECESSOR_RUNNER_SHA}))
        with self.assertRaises(ValueError):launch.verify_previous_workflow_terminal(self.root)
        terminal=dict(schema='headtail-link-workflow-v1',experiment_name=launch.PREDECESSOR,
            status='complete',finished_at='2026-11-06T13:00:00+08:00',holder_final_sha256='f'*64)
        self.write(work/'terminal.json',terminal)
        self.assertIn('workflow_terminal_sha256',launch.verify_previous_workflow_terminal(self.root))
        for update in (dict(status='running'),dict(finished_at=None),dict(experiment_name='foreign'),
                       dict(holder_error='missing holder'),dict(holder_final_sha256=None)):
            self.write(work/'terminal.json',dict(terminal,**update))
            with self.subTest(update=update),self.assertRaises(ValueError):launch.verify_previous_workflow_terminal(self.root)
        self.write(work/'terminal.json',dict(terminal,status='failed',holder_on_error_sha256='e'*64))
        self.assertIn('workflow_terminal_sha256',launch.verify_previous_workflow_terminal(self.root))

    def test_failed_supervisor_is_not_retried_and_holder_is_checked(self):
        p=self.plan();child=SimpleNamespace(pid=4321,wait=mock.Mock(return_value=1))
        snapshot=launch.lifecycle.seal(dict(schema='holder-fixture'))
        with mock.patch.object(launch,'verify_plan'),mock.patch.object(launch,'verify_previous_fullmodel_terminal'), \
             mock.patch.multiple(launch,ROOT=self.root,REMOTE_STAGE=self.stage,STAGE=self.stage), \
             mock.patch.object(launch.lifecycle,'verify_lifecycle'), \
             mock.patch.object(launch.lifecycle,'holder_snapshot',return_value=snapshot) as holder, \
             mock.patch.object(launch.lifecycle,'proc_identity',return_value=dict(pid=4321,argv=p['argv'])), \
             mock.patch.object(launch.subprocess,'Popen',return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            result=launch.run_plan(p)
        self.assertFalse(result['successful']);self.assertFalse(result['automatic_retry'])
        self.assertTrue(result['holder_physically_verified']);self.assertEqual(holder.call_count,2)
        self.assertEqual(popen.call_count,1)


if __name__=='__main__':unittest.main()
