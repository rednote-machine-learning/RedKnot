"""Finite lifecycle, source and input binding for the twenty-case comparison."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import natural_boundary_protocol as protocol
from gpu_pilot import launch_natural_boundary_pilot as launch


class NaturalBoundaryLauncherTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.stage=self.root/'experiment/GatedEngram'
        self.native=self.stage.parent/'NativeSnapshot';(self.root/'results').mkdir()
        names=('validate_natural_boundary_cpu.py','gpu_pilot/run_pilot.py','gpu_pilot/launch_pilot.py','gpu_pilot/natural_boundary_protocol.py',
            'gpu_pilot/natural_boundary_inputs.py','gpu_pilot/run_natural_boundary_pilot.py',
            'gpu_pilot/read_natural_boundary_results.py','gpu_pilot/launch_natural_boundary_pilot.py',
            'kv_cache_engram/headtail_comparison.py','kv_cache_engram/headtail_link.py',
            'kv_cache_engram/headtail_link_plan.py','kv_cache_engram/early_token_diagnostics.py',
            'tests/test_launch_natural_boundary.py')
        for name in names:
            p=self.stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
        for name in ('setup.py','README.md','deepinfer/model_executor/models/v4_modules/attn.py'):
            p=self.native/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
        self.manifest=self.stage/'manifest.json';self.inputs=self.stage/'input-protocol.json'
        self.comparison=self.stage/'natural-boundary-protocol.json'
        self.write(self.inputs,dict(schema='natural-boundary-input-protocol-v1'))
        data=dict(schema='di-sg-real-rag-manifest-v1',artifact_kind='natural_boundary_inputs_v1',
            contract_schema='kv-cache-engram-natural-boundary-input-manifest-v1',
            input_protocol_sha256=launch.lifecycle.read_sealed(self.inputs)['sha256'])
        self.write(self.manifest,data)
        patch=mock.patch.object(protocol,'INPUT_MANIFEST_SHA256',launch.lifecycle.read_sealed(self.manifest)['sha256'])
        patch.start();self.addCleanup(patch.stop)
        self.comparison.write_bytes(launch.lifecycle.canonical(protocol.natural_boundary_protocol()))
        for name in launch.INPUT_COMPANIONS:self.write(self.stage/name,dict(schema='fixture',name=name))
        self.previous=dict(name=launch.PREDECESSOR,previous_workers_exited=True,previous_scores_reused=False)

    def write(self,path,value):path.write_bytes(launch.lifecycle.canonical(launch.lifecycle.seal(value)))

    def proof(self):
        families=('test_headtail_link_plan','test_headtail_comparison','test_natural_boundary_runner',
            'test_natural_boundary_results','test_launch_natural_boundary','test_natural_boundary_inputs',
            'test_index_hit20_empty_ep_qualification','test_early_token_diagnostics','test_natural_boundary_controller')
        ids=[f'tests.{f}.Fixture.test_required' for f in families]
        ids += [f'tests.fixture.Fixture.test_{i}' for i in range(1000-len(ids))]
        return dict(schema='kv-cache-engram-natural-boundary-cpu-proof-v1',successful=True,
            cuda_initialized=False,cuda_init_guard_installed=True,failures=0,errors=0,skipped=0,
            tests=len(ids),test_ids=ids,source_sha256=launch.lifecycle.source_files(self.stage))

    def plan(self):
        self.write(self.stage/'NATURAL_BOUNDARY_CPU_TESTS-v1.json',self.proof())
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
        p=self.plan();(self.stage/'selection-plan.json').write_text('{}')
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

    def previous_receipt(self, mutate=None):
        old=self.root/'DeepInfer-PR148/experiments/previous/GatedEngram'
        result=self.root/'results'/launch.PREDECESSOR
        argv=['python','original-supervisor','--',str(old/'gpu_pilot/run_headtail_scale20_pilot.py')]
        seal=launch.lifecycle.seal
        plan=seal(dict(schema=launch.lifecycle.PLAN_SCHEMA,name=launch.PREDECESSOR,
            root=str(self.root),stage_dir=str(old),argv=argv,
            stage_sources={'gpu_pilot/run_headtail_scale20_pilot.py':launch.PREDECESSOR_RUNNER_SHA}))
        identity=dict(pid=900002,start_ticks=12345,pgid=900002,sid=900002,argv=argv)
        started=seal(dict(schema='tail-link-original-supervisor-started-v1',plan_sha256=plan['sha256'],**identity))
        marker=dict(schema='user-authorized-natural-context-transition-v1',old_results_preserved=True,identity=identity.copy())
        sup=dict(schema='redknot-glm53-gpu-lifecycle-v1',status='failed',finished_unix=10.,holder_restored=True,
            runner_sha256=launch.PREDECESSOR_RUNNER_SHA,supervisor_sha256=launch.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launch.lifecycle.HELPER_SHA,child_environment_sha256=launch.lifecycle.ENV_SHA,
            gpu_ids=list(range(8)),error="KeyboardInterrupt('received signal 15')",child_pid=900003,
            cleanup_warnings=['authenticated auxiliary ownership warning preserved'])
        contexts=[dict(pid=910000+i,uuid='gpu'+str(i),memory_mib=1000) for i in range(8)]
        holder=dict(schema='fixture-holder',metadata=dict(worker_pids=[c['pid'] for c in contexts]),contexts=contexts)
        terminal=dict(schema='headtail-scale20-workflow-v1',experiment_name=launch.PREDECESSOR,
            status='failed',finished_at='2026-11-06T14:34:21+08:00')
        outcome=dict(schema='tail-link-launch-outcome-v1',successful=False,supervisor_started=True,
            holder_physically_verified=True,plan_sha256=plan['sha256'],finished_unix=11.)
        values=dict(launch_plan=plan,supervisor_started=started,stop_request=marker,supervisor=sup,
                    holder_on_error=holder,workflow_terminal=terminal,launch_outcome=outcome)
        if mutate:mutate(values)
        values['holder_on_error']=seal(values['holder_on_error'])
        values['workflow_terminal']['holder_on_error_sha256']=values['holder_on_error']['sha256']
        for key in ('stop_request','workflow_terminal','launch_outcome'):values[key]=seal(values[key])
        paths=dict(stop_request=self.root/'results'/(launch.PREDECESSOR+'.natural4k-stop-request.json'),
            supervisor=result/'supervisor.json',launch_plan=self.root/'results'/(launch.PREDECESSOR+'.launch-plan.json'),
            supervisor_started=self.root/'results'/(launch.PREDECESSOR+'.supervisor-started.json'),
            launch_outcome=self.root/'results'/(launch.PREDECESSOR+'.launch-outcome.json'),
            workflow_terminal=old.parent/'Workflow/terminal.json',holder_on_error=old.parent/'Workflow/holder-on-error.json')
        artifacts={}
        for key,path in paths.items():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(launch.lifecycle.canonical(values[key]))
            artifacts[key]=dict(path=str(path),file_sha256=launch.lifecycle.file_digest(path),value=values[key])
        receipt=seal(dict(schema='natural-boundary-predecessor-restoration-v1',prior_experiment=launch.PREDECESSOR,
            termination_reason='user_replaced_artificial_cohort',all_old_owned_pids_absent=True,
            only_restored_holder_contexts=True,old_owned_pids=list(range(900000,900021)),contexts=contexts,artifacts=artifacts))
        path=self.root/'predecessor-evidence.json';path.write_bytes(launch.lifecycle.canonical(receipt))
        (self.root/'proc').mkdir(exist_ok=True)
        return receipt

    @contextlib.contextmanager
    def bind_previous(self,receipt):
        values={key:item['value'] for key,item in receipt['artifacts'].items()}
        with mock.patch.multiple(launch,PREDECESSOR_RECEIPT_SHA=receipt['sha256'],
            STOP_REQUEST_SHA=values['stop_request']['sha256'],
            PREDECESSOR_WORKFLOW_SHA=values['workflow_terminal']['sha256'],
            PREDECESSOR_HOLDER_SHA=values['holder_on_error']['sha256'],
            PREDECESSOR_RECEIPT_PATH=self.root/'predecessor-evidence.json',PROC_ROOT=self.root/'proc'):
            yield

    def test_authenticated_cancelled_run_does_not_require_complete_or_group_json(self):
        receipt=self.previous_receipt()
        with self.bind_previous(receipt):
            value=launch.verify_previous_fullmodel_terminal(self.root)
        self.assertTrue(value['user_requested_cancellation'])
        self.assertTrue(value['previous_workers_exited'])
        self.assertEqual(value['prior_status'],'failed')
        self.assertFalse((self.root/'results'/launch.PREDECESSOR/'benchmark/process-group.json').exists())

    def test_stop_identity_restoration_and_workflow_are_independent_gates(self):
        mutations=[lambda v:v['stop_request']['identity'].update(start_ticks=999),
            lambda v:v['supervisor'].update(holder_restored=False),
            lambda v:v['supervisor'].update(error='unrelated failure'),
            lambda v:v['workflow_terminal'].update(status='complete'),
            lambda v:v['workflow_terminal'].update(holder_error='missing'),
            lambda v:v['launch_outcome'].update(holder_physically_verified=False)]
        for mutate in mutations:
            receipt=self.previous_receipt(mutate)
            with self.subTest(mutate=mutate),self.bind_previous(receipt),self.assertRaises(ValueError):
                launch.verify_previous_fullmodel_terminal(self.root)

    def test_live_old_pid_or_old_audit_refuses_new_admission(self):
        receipt=self.previous_receipt()
        with self.bind_previous(receipt):
            pid=self.root/'proc/900002';pid.mkdir()
            with self.assertRaises(ValueError):launch.verify_previous_fullmodel_terminal(self.root)
            pid.rmdir()
            audit=self.root/'proc/999999';audit.mkdir()
            (audit/'cmdline').write_bytes(b'python\0read_headtail_scale20_results.py\0'+
                str(self.root/'results'/launch.PREDECESSOR/'benchmark').encode()+b'\0')
            with self.assertRaises(ValueError):launch.verify_previous_fullmodel_terminal(self.root)

    def test_predecessor_source_artifact_cannot_change_after_receipt(self):
        receipt=self.previous_receipt()
        Path(receipt['artifacts']['supervisor']['path']).write_text('{}')
        with self.bind_previous(receipt),self.assertRaises(ValueError):launch.verify_previous_fullmodel_terminal(self.root)

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
