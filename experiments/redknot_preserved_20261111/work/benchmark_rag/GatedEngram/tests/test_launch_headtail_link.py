"""CPU-only checks of finite original supervisor and independent headtail stage."""
import copy
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import launch_headtail_link_pilot as launch
from gpu_pilot.headtail_link_protocol import headtail_link_protocol
from gpu_pilot.sensitivity_protocol import sensitivity_protocol


class HeadTailLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage = self.root/'independent/GatedEngram'; self.native = self.stage.parent/'NativeSnapshot'
        (self.root/'results').mkdir()
        for name in ('gpu_pilot/run_pilot.py','gpu_pilot/launch_pilot.py',
                     'gpu_pilot/run_headtail_link_pilot.py','gpu_pilot/read_headtail_link_results.py',
                     'gpu_pilot/headtail_link_protocol.py','gpu_pilot/launch_headtail_link_pilot.py',
                     'kv_cache_engram/headtail_link.py','kv_cache_engram/headtail_link_plan.py',
                     'tests/test_launch_headtail_link.py'):
            path=self.stage/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('# fixture\n')
        for name in ('setup.py','README.md','deepinfer/model_executor/models/v4_modules/attn.py'):
            path=self.native/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('# fixture\n')
        self.manifest=self.stage/'manifest.json'; self.protocol=self.stage/'protocol.json'
        self.sensitivity=self.stage/'sensitivity-protocol.json'; self.headtail=self.stage/'headtail-link-protocol.json'
        self.write(self.protocol,dict(schema=launch.lifecycle.PROTOCOL_SCHEMA))
        self.sensitivity.write_bytes(launch.lifecycle.canonical(sensitivity_protocol()))
        self.headtail.write_bytes(launch.lifecycle.canonical(headtail_link_protocol()))
        self.write(self.manifest,dict(schema='fixture',artifact_kind='kv_cache_engram_native_pilot',
            pilot_protocol_sha256=launch.lifecycle.read_json(self.protocol)['sha256'],
            sensitivity_protocol_sha256=sensitivity_protocol()['sha256']))
        self.previous=dict(name=launch.PREDECESSOR, previous_workers_exited=True, previous_scores_reused=False)

    def write(self,path,value):
        path.write_bytes(launch.lifecycle.canonical(launch.lifecycle.seal(value)))

    def proof(self):
        families=('test_headtail_link_plan','test_headtail_link_controller','test_headtail_link_runner',
                  'test_headtail_link_results','test_launch_headtail_link','test_index_hit20_empty_ep_qualification')
        ids=[f'tests.{name}.Fixture.test_required' for name in families]
        ids += [f'tests.fixture.Test.test_{i}' for i in range(1000-len(ids))]
        return dict(schema='kv-cache-engram-headtail-link-cpu-proof-v1',successful=True,cuda_initialized=False,
            cuda_init_guard_installed=True,failures=0,errors=0,skipped=0,tests=len(ids),test_ids=ids,
            source_sha256=launch.lifecycle.source_files(self.stage))

    def plan(self):
        self.write(self.stage/'HEADTAIL_LINK_CPU_TESTS-v1.json',self.proof())
        with mock.patch.object(launch,'predecessor_evidence',return_value=self.previous):
            return launch.build_plan(self.root,self.stage,self.manifest,self.protocol,self.headtail,self.sensitivity)

    def verify(self,plan):
        with mock.patch.object(launch,'predecessor_evidence',return_value=self.previous): launch.verify_plan(plan)

    def test_original_supervisor_5400_real_argv_and_private_snapshot(self):
        p=self.plan(); self.verify(p); a=p['argv']
        self.assertEqual(a[a.index('--timeout')+1],'5400')
        self.assertEqual(a[a.index('--prepare-timeout')+1],'300')
        self.assertEqual(a[a.index('--repo')+1],str(self.native))
        self.assertEqual(a[a.index('--')+1],str(self.stage/'gpu_pilot/run_headtail_link_pilot.py'))
        self.assertEqual(p['evaluation_arms'],['headtail128_link'])
        self.assertEqual(p['dynamic_refresh_start_layer'],3)
        self.assertTrue(p['no_automatic_retry']); self.assertTrue(p['full_canonical_gpu_bank'])

    def test_timeout_cannot_be_extended_even_with_new_seal(self):
        p=self.plan()
        for change in ('plan','argv','both'):
            bad=copy.deepcopy(p)
            if change in ('plan','both'): bad['timeout']=21600
            if change in ('argv','both'): bad['argv'][bad['argv'].index('--timeout')+1]='21600'
            bad=launch.lifecycle.seal({k:v for k,v in bad.items() if k!='sha256'})
            with self.subTest(change=change),self.assertRaises(ValueError): self.verify(bad)

    def test_policy_and_source_mutation_rejected(self):
        p=self.plan()
        for field,value in (('head_tokens',0),('dynamic_refresh_start_layer',4),('native_yoco_swa_share_hidden',True),
                            ('evaluation_arms',['recomputed','headtail128_link'])):
            bad=copy.deepcopy(p);bad[field]=value
            bad=launch.lifecycle.seal({k:v for k,v in bad.items() if k!='sha256'})
            with self.subTest(field=field),self.assertRaises(ValueError): self.verify(bad)
        (self.native/'README.md').write_text('changed')
        with self.assertRaises(ValueError): self.verify(p)

    def test_new_test_families_no_skip_no_cuda_and_source_binding_required(self):
        p=self.plan()
        for update in (dict(skipped=1),dict(cuda_initialized=True),dict(successful=False),dict(source_sha256={}),dict(tests=999)):
            proof=self.proof();proof.update(update);self.write(Path(p['cpu_proof_path']),proof)
            with self.subTest(update=update),self.assertRaises(ValueError): launch.verify_cpu_proof(p)

    def test_check_only_cannot_launch_or_touch_holder(self):
        p=self.plan()
        with mock.patch.object(launch,'build_plan',return_value=p), \
             mock.patch.object(launch,'predecessor_evidence',return_value=self.previous), \
             mock.patch.object(launch,'run_plan',side_effect=AssertionError('launch forbidden')), \
             mock.patch.object(launch.lifecycle,'holder_snapshot',side_effect=AssertionError('GPU forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(['--check-only']),0)

    def test_timeout_failure_is_not_retried_and_holder_audited(self):
        p=self.plan(); child=SimpleNamespace(pid=1234,wait=mock.Mock(return_value=1))
        snapshot=launch.lifecycle.seal(dict(schema='holder-fixture'))
        with mock.patch.object(launch,'verify_plan'),mock.patch.object(launch,'verify_previous_fullmodel_terminal'), \
             mock.patch.multiple(launch,ROOT=self.root,REMOTE_STAGE=self.stage,STAGE=self.stage), \
             mock.patch.object(launch.lifecycle,'verify_lifecycle'), \
             mock.patch.object(launch.lifecycle,'holder_snapshot',return_value=snapshot) as holder, \
             mock.patch.object(launch.lifecycle,'proc_identity',return_value=dict(pid=1234,argv=p['argv'])), \
             mock.patch.object(launch.subprocess,'Popen',return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            result=launch.run_plan(p)
        self.assertFalse(result['successful']);self.assertFalse(result['automatic_retry'])
        self.assertTrue(result['holder_physically_verified']);self.assertEqual(holder.call_count,2)
        self.assertEqual(popen.call_count,1)


if __name__ == '__main__': unittest.main()
