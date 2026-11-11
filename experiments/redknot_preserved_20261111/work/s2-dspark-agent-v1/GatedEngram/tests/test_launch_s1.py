"""Finite lifecycle, source and input binding for the natural Loong multi-document RAG comparison."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import s1_protocol as protocol
from gpu_pilot import launch_s1_pilot as launch


class S1LauncherTests(unittest.TestCase):
    def setUp(self):
        clock=mock.patch.object(launch.time,"time",return_value=1794050000)
        clock.start();self.addCleanup(clock.stop)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.stage=self.root/'experiment/GatedEngram'
        (self.root/'pr148-native-env.json').write_text('{}')
        self.native=self.stage.parent/'NativeSnapshot';(self.root/'results').mkdir()
        names=('validate_s1_cpu.py','gpu_pilot/run_pilot.py','gpu_pilot/launch_pilot.py','gpu_pilot/s1_protocol.py',
            'gpu_pilot/loong_inputs.py','gpu_pilot/loong_metrics.py','gpu_pilot/token_divergence.py','gpu_pilot/run_s1_pilot.py',
            'gpu_pilot/read_s1_results.py','gpu_pilot/launch_s1_pilot.py',
            'kv_cache_engram/s1_runtime.py',
            'tests/test_launch_s1.py')
        for name in names:
            p=self.stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
        for name in ('setup.py','README.md','deepinfer/model_executor/models/v4_modules/attn.py'):
            p=self.native/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
        self.manifest=self.stage/'manifest.json';self.inputs=self.stage/'input-protocol.json'
        self.comparison=self.stage/'s1-protocol.json'
        self.write(self.inputs,dict(schema='loong-input-protocol-v1'))
        data=dict(schema='di-sg-real-rag-manifest-v1',artifact_kind='loong_inputs_v1',
            contract_schema='kv-cache-engram-loong-input-manifest-v1',
            input_protocol_sha256=launch.lifecycle.read_sealed(self.inputs)['sha256'])
        self.write(self.manifest,data)
        patch=mock.patch.object(protocol,'INPUT_MANIFEST_SHA256',launch.lifecycle.read_sealed(self.manifest)['sha256'])
        patch.start();self.addCleanup(patch.stop)
        self.comparison.write_bytes(launch.lifecycle.canonical(protocol.s1_protocol()))
        for name in launch.INPUT_COMPANIONS:self.write(self.stage/name,dict(schema='fixture',name=name))
        self.previous=dict(name=launch.PREDECESSOR,previous_workers_exited=True,previous_scores_reused=False)
        sources=launch.lifecycle.source_files(self.stage)
        core={'kv_cache_engram/s1_runtime.py':sources['kv_cache_engram/s1_runtime.py']}
        for key,value in [('CORE_SOURCE_SHA256',core),('NATIVE_SNAPSHOT_SHA256',launch.lifecycle.digest(launch.snapshot_sources(self.native)))]:
            patch=mock.patch.object(protocol,key,value);patch.start();self.addCleanup(patch.stop)
        self.comparison.write_bytes(launch.lifecycle.canonical(protocol.s1_protocol()))

    def write(self,path,value):path.write_bytes(launch.lifecycle.canonical(launch.lifecycle.seal(value)))

    def proof(self):
        families=('test_s1_runner','test_s1_results','test_launch_s1','test_s1_runtime',
                  'test_s1_cache_attention','test_token_divergence','test_loong_results','test_loong_metrics',
            'test_loong_inputs','test_musique30_s1_geometry','test_index_hit20_empty_ep_qualification')
        ids=[f'tests.{f}.Fixture.test_required' for f in families]
        ids += [f'tests.fixture.Fixture.test_{i}' for i in range(1000-len(ids))]
        return dict(schema='kv-cache-engram-s1-cpu-proof-v1',successful=True,
            cuda_initialized=False,cuda_init_guard_installed=True,failures=0,errors=0,skipped=0,
            tests=len(ids),test_ids=ids,source_sha256=launch.lifecycle.source_files(self.stage))

    def plan(self):
        self.write(self.stage/'S1_CPU_TESTS-v1.json',self.proof())
        with mock.patch.object(launch,'predecessor_evidence',return_value=self.previous):
            return launch.build_plan(self.root,self.stage,self.manifest,self.inputs,self.comparison)

    def verify(self,value):
        with mock.patch.object(launch,'predecessor_evidence',return_value=self.previous):launch.verify_plan(value)

    def test_original_supervisor_binds_two_hours_from_gpu_release(self):
        p=self.plan();self.verify(p);args=p['argv']
        self.assertEqual(args[args.index('--timeout')+1],'7200')
        self.assertEqual(args[args.index('--prepare-timeout')+1],'300')
        self.assertEqual(p['evaluation_indices'],list(protocol.EVALUATION_INDICES))
        self.assertEqual(p['evaluation_arms'],list(protocol.ARMS))
        self.assertIsNone(p['sensitivity_protocol']);self.assertNotIn('--sensitivity-protocol',args)
        self.assertEqual(args[args.index('--repo')+1],str(self.native))
        self.assertTrue(p['fresh_all_cases_both_arms'])

    def test_timeout_and_long_context_contract_cannot_change_with_resealing(self):
        p=self.plan()
        for update in (dict(timeout=5400),dict(timeout=0),dict(completion_case_count=4),
                       dict(evaluation_indices=[2,3,4,5]),dict(evaluation_arms=['headtail128_link']),
                       dict(hard_deadline_unix=1793970588)):
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

    def test_s1_runtime_test_family_is_required(self):
        p=self.plan()
        for family in ('test_s1_runtime.','test_token_divergence.','test_loong_results.'):
            proof=self.proof()
            proof['test_ids']=[name.replace(family, 'test_unrelated.') for name in proof['test_ids']]
            self.write(Path(p['cpu_proof_path']),proof)
            with self.subTest(family=family),self.assertRaises(ValueError):launch.verify_cpu_proof(p)

    def test_plan_timeout_is_relative_and_unchanged_by_prelaunch_delay(self):
        p=self.plan()
        for later in (1794050000+120,1794050000+86400):
            with mock.patch.object(launch.time,'time',return_value=later):self.verify(p)
        self.assertEqual(p['timeout'],7200)
        self.assertEqual(p['timeout_origin'],'original supervisor GPU release')
        self.assertIsNone(p['absolute_deadline_unix'])
        self.assertFalse(any(k in p for k in ('hard_deadline_unix','stop_request_unix')))
        self.assertEqual(p['completion_case_count'],len(protocol.EVALUATION_INDICES))
        self.assertEqual(p['document_count'],10)

    def test_readonly_git_environment_is_idempotent_and_keeps_other_settings(self):
        env=dict(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0='/native',
                 GIT_OPTIONAL_LOCKS='1',LD_LIBRARY_PATH='pinned-libs')
        launch.protect_git_environment(env);once=dict(env);launch.protect_git_environment(env)
        self.assertEqual(env,once);self.assertEqual(env['GIT_OPTIONAL_LOCKS'],'0')
        self.assertEqual(env['GIT_CONFIG_COUNT'],'2')
        self.assertEqual(env['GIT_CONFIG_KEY_1'],'diff.autoRefreshIndex')
        self.assertEqual(env['GIT_CONFIG_VALUE_1'],'false')
        self.assertEqual(env['LD_LIBRARY_PATH'],'pinned-libs')
        for bad in (dict(GIT_CONFIG_COUNT='bad'),dict(GIT_CONFIG_COUNT='1'),dict(GIT_CONFIG_PARAMETERS='arbitrary')):
            with self.assertRaises(ValueError):launch.protect_git_environment(bad)

    def test_child_environment_cannot_reenable_git_index_writes(self):
        path=self.root/'pr148-native-env.json'
        for value in (dict(GIT_OPTIONAL_LOCKS='1'),dict(GIT_CONFIG_COUNT='0'),dict(GIT_CONFIG_PARAMETERS='x')):
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):launch.supervisor_environment(self.root)
        path.write_text('{}');env=launch.supervisor_environment(self.root)
        self.assertEqual(env['GIT_OPTIONAL_LOCKS'],'0')
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1')

    def test_latest_complete6_predecessor_and_worker_absence_are_required(self):
        previous=self.root/'DeepInfer-PR148/experiments/kce_longcontext64_128_20261109.qzae9zix/Workflow'
        previous.mkdir(parents=True);proc=self.root/'proc';proc.mkdir()
        audit=launch.lifecycle.seal(dict(schema='longcontext-s1-independent-readout-v1',status='complete',errors=[],
            complete_cases=6,planned_cases=6,completed_case_indices=list(range(2,8)),
            final_run_audited=True,overall_run_complete=True,all_registered_cases_audited=True))
        holder=launch.lifecycle.seal(dict(schema='exact-restored-holder-live-verification-v1',status='passed',
            experiment_contexts_present=False,elapsed_s=launch.PREDECESSOR_HOLDER_ELAPSED_S,samples=[{}]*37))
        terminal=launch.lifecycle.seal(dict(schema='longcontext-workflow-v1',status='complete',experiment_name=launch.PREDECESSOR,
            holder_physically_verified=True,launcher_successful=True,supervisor_exit=0,final_audit_sha256=audit['sha256'],
            complete_cases=6,planned_cases=6,completed_case_indices=list(range(2,8)),holder_live_verification_sha256=holder['sha256']))
        for name,value in [('terminal',terminal),('final-audit',audit),('holder-live-verification',holder)]:
            (previous/(name+'.json')).write_text(json.dumps(value))
        with mock.patch.multiple(launch,PROC_ROOT=proc,PREDECESSOR_WORKFLOW_SHA=terminal['sha256'],
                PREDECESSOR_AUDIT_SHA=audit['sha256'],PREDECESSOR_HOLDER_SHA=holder['sha256']):
            self.assertEqual(launch.verify_previous_fullmodel_terminal(self.root)['prior_status'],'complete')
            (proc/'42').mkdir();(proc/'42/cmdline').write_bytes(str(previous.parent/'GatedEngram/gpu_pilot/run_s1_pilot.py').encode()+b'\0')
            with self.assertRaisesRegex(ValueError,'still active'):launch.verify_previous_fullmodel_terminal(self.root)
            (proc/'42/cmdline').unlink();(proc/'42').rmdir()
            terminal['status']='timeout_partial';terminal=launch.lifecycle.seal({k:v for k,v in terminal.items() if k!='sha256'})
            (previous/'terminal.json').write_text(json.dumps(terminal))
            with mock.patch.object(launch,'PREDECESSOR_WORKFLOW_SHA',terminal['sha256']),self.assertRaises(ValueError):
                launch.verify_previous_fullmodel_terminal(self.root)

    def test_core_or_native_drift_rejected_even_if_plan_is_resealed(self):
        p=self.plan();expected=protocol.s1_protocol()
        launch.verify_core_identity(p['stage_sources'],p['native_snapshot_sources_sha256'],expected)
        altered=dict(p['stage_sources']);altered['kv_cache_engram/s1_runtime.py']='0'*64
        for sources,native in ((altered,p['native_snapshot_sources_sha256']),(p['stage_sources'],'0'*64)):
            with self.assertRaisesRegex(ValueError,'core modules'):launch.verify_core_identity(sources,native,expected)

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
        self.assertEqual(popen.call_args.kwargs['env']['GIT_OPTIONAL_LOCKS'],'0')
        self.assertEqual(popen.call_args.args[0][popen.call_args.args[0].index('--timeout')+1],'7200')

    def test_success_accepts_frozen_twenty_case_run_and_publishes_sealed_outcome(self):
        from gpu_pilot.run_s1_pilot import SCHEMA
        p=self.plan();result_dir=Path(p['paths']['result_dir'])
        (result_dir/'benchmark').mkdir(parents=True)
        self.write(result_dir/'benchmark/run.json',dict(schema=SCHEMA,status='complete',
            completed_case_indices=list(protocol.EVALUATION_INDICES)))
        (result_dir/'supervisor.json').write_text('{}')
        child=SimpleNamespace(pid=4321,wait=mock.Mock(return_value=0))
        snapshot=launch.lifecycle.seal(dict(schema='holder-fixture'))
        with mock.patch.object(launch,'verify_plan'),mock.patch.object(launch,'verify_previous_fullmodel_terminal'), \
             mock.patch.object(launch,'verify_terminal'),mock.patch.object(launch.lifecycle,'require_unused'), \
             mock.patch.multiple(launch,ROOT=self.root,REMOTE_STAGE=self.stage,STAGE=self.stage), \
             mock.patch.object(launch.lifecycle,'verify_lifecycle'), \
             mock.patch.object(launch.lifecycle,'holder_snapshot',return_value=snapshot), \
             mock.patch.object(launch.lifecycle,'proc_identity',return_value=dict(pid=4321,argv=p['argv'])), \
             mock.patch.object(launch.subprocess,'Popen',return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            returned=launch.run_plan(p)
        self.assertTrue(returned['successful']);self.assertTrue(returned['holder_physically_verified'])
        actual=launch.lifecycle.read_sealed(p['paths']['outcome'])
        self.assertEqual({k:v for k,v in actual.items() if k!='sha256'},returned)
        self.assertEqual(returned['run_sha256'],launch.lifecycle.read_sealed(result_dir/'benchmark/run.json')['sha256'])
        self.assertEqual(popen.call_count,1)


class ActualLongContextPredecessorTests(unittest.TestCase):
    def test_actual_sealed_terminal_audit_and_180038ms_holder_observation(self):
        # Use the complete captured artifacts, not reconstructed seal fixtures.
        # Local preparation has the verified retrieval; server admission has the
        # same immutable predecessor Workflow. Missing evidence fails the test.
        relative=Path('DeepInfer-PR148/experiments/kce_longcontext64_128_20261109.qzae9zix/Workflow')
        local=Path(__file__).resolve().parents[4]/'outputs/longcontext-v1-final'
        source=local if local.is_dir() else Path(launch.ROOT)/relative
        names=('terminal.json','final-audit.json','holder-live-verification.json')
        original={name:(source/name).read_bytes() for name in names}
        actual={name:launch.lifecycle.read_sealed(source/name) for name in names}
        self.assertEqual(actual['terminal.json']['sha256'],launch.PREDECESSOR_WORKFLOW_SHA)
        self.assertEqual(actual['final-audit.json']['sha256'],launch.PREDECESSOR_AUDIT_SHA)
        holder=actual['holder-live-verification.json']
        self.assertEqual(holder['sha256'],launch.PREDECESSOR_HOLDER_SHA)
        self.assertEqual(holder['elapsed_s'],180.03817653656006)
        self.assertEqual(len(holder['samples']),37)
        self.assertEqual(holder['elapsed_s'],holder['samples'][-1]['observed_unix']-holder['samples'][0]['observed_unix'])
        for sample in holder['samples']:
            self.assertEqual([gpu['index'] for gpu in sample['gpus']],list(range(8)))
            self.assertTrue(all(70<=gpu['utilization']<=90 for gpu in sample['gpus']))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();previous=root/relative;previous.mkdir(parents=True)
            proc=root/'proc';proc.mkdir()
            for name,data in original.items():(previous/name).write_bytes(data)
            with mock.patch.object(launch,'PROC_ROOT',proc):
                evidence=launch.verify_previous_fullmodel_terminal(root)
                self.assertEqual(evidence['completed_case_indices'],list(range(2,8)))
                self.assertEqual(evidence['holder_observation_elapsed_s'],holder['elapsed_s'])
                self.assertEqual(evidence['holder_sample_count'],37)
                # A different duration, even resealed, cannot replace this proof.
                changed=launch.lifecycle.seal({**{k:v for k,v in holder.items() if k!='sha256'},'elapsed_s':180.0})
                (previous/'holder-live-verification.json').write_bytes(launch.lifecycle.canonical(changed))
                with self.assertRaisesRegex(ValueError,'restoration proof differs'):
                    launch.verify_previous_fullmodel_terminal(root)
        for name,data in original.items():self.assertEqual((source/name).read_bytes(),data)


if __name__=='__main__':unittest.main()
