"""CPU-only evidence and failure-path tests; never invokes NVIDIA/holder code."""
import copy
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import profile_lifecycle as lifecycle


def holder():
    parent=120;workers=list(range(121,129));script='/holder/gpu_hold_target.py'
    identities={str(parent):dict(pid=parent,ppid=1,pgid=parent,sid=parent,start_ticks=100,
        argv=[lifecycle.HOLDER_PYTHON,'-u',script,'--run','--target','80'])}
    identities.update({str(pid):dict(pid=pid,ppid=parent,pgid=parent,sid=parent,start_ticks=101,
        argv=[lifecycle.HOLDER_PYTHON,'-c','multiprocessing.spawn']) for pid in workers})
    return dict(metadata=dict(pid=parent,pgid=parent,worker_pids=workers,gpu_ids=list(range(8)),
        target_util_percent=80,variation_amplitude_percent=6,target_range_percent=[74,86],script=script),
        process_identities=identities,snapshot=dict(observed_unix=10,gpus=[dict(index=i,uuid=f'GPU-{i}',utilization=80,memory_mib=1009) for i in range(8)],
        compute_apps=[dict(pid=121+i,uuid=f'GPU-{i}',memory_mib=1009) for i in range(8)]))


def configuration():
    paths=['/native/glm53_gpu_lifecycle.py','/holder/run_segpaged_experiment.py',
        '/holder/gpu_hold_target.py','/env.json','/runner.py',str(Path(lifecycle.__file__).resolve())]
    return lifecycle.seal(dict(schema='s12-system-profile-lifecycle-config-v1',timeout_seconds=7200,
        prepare_timeout_seconds=600,holder_python=lifecycle.HOLDER_PYTHON,lock_file=lifecycle.SHARED_LOCK,
        holder_policy='restore_original_holder_80_plusminus6_on_exit',independent_watchdog=False,
        sigkill_or_host_loss_recovery_claim=False,runner_argv=['/runner.py','--profile-case','20'],
        original_supervisor=paths[0],holder_dir='/holder',holder_helper=paths[1],holder_script=paths[2],
        child_env_json=paths[3],child_python='/runtime/python',cwd='/stage',output_dir='/results/wrapper',
        source_sha256={p:'a'*64 for p in paths},admission_refs=[dict(path='/proof.json',file_sha256='b'*64,
            seal_sha256='c'*64,required_fields={'status':'passed','cuda_initialized':False})],expected_holder=holder()))


def reseal(value):
    return lifecycle.seal({k:v for k,v in value.items() if k!='sha256'})


class ProfileLifecycleTests(unittest.TestCase):
    def test_only_original_interpreter_and_7200_second_supervisor(self):
        config=configuration();self.assertTrue(lifecycle.validate_config(config,check_files=False))
        command=lifecycle.supervisor_command(config)
        self.assertEqual(command[0],'/root/miniconda3/bin/python')
        self.assertEqual(command[command.index('--timeout')+1],'7200')
        self.assertEqual(command[command.index('--prepare-timeout')+1],'600')
        self.assertEqual(command[command.index('--lock-file')+1],lifecycle.SHARED_LOCK)
        self.assertEqual(command[-len(config['runner_argv']):],config['runner_argv'])
        self.assertNotIn('--watchdog',command)

    def test_root_fixed_result_path_is_respected(self):
        config=configuration();config['supervisor_result_dir']='/newbase/ProfileRun'
        command=lifecycle.supervisor_command(config)
        self.assertEqual(command[command.index('--result-dir')+1],'/newbase/ProfileRun')

    def test_model_environment_cannot_contaminate_holder(self):
        env=lifecycle.clean_holder_environment(dict(HOME='/root',PATH='/evil',PYTHONPATH='/model',
            LD_LIBRARY_PATH='/compat',LD_PRELOAD='/hook.so',CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/cuda',
            PYTHONHOME='/bad',CONDA_PREFIX='/child',SPT_NOENV='1'))
        for key in ('PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_VISIBLE_DEVICES','CUDA_HOME','PYTHONHOME','CONDA_PREFIX','SPT_NOENV'):
            self.assertNotIn(key,env)
        self.assertEqual(env['HOME'],'/root');self.assertTrue(env['PATH'].startswith('/root/miniconda3/bin:'))

    def test_deadline_and_unimplemented_watchdog_claim_rejected(self):
        for key,value in [('timeout_seconds',86400),('prepare_timeout_seconds',3600),('independent_watchdog',True),
                          ('sigkill_or_host_loss_recovery_claim',True),('lock_file','/tmp/other.lock')]:
            with self.subTest(key=key):
                c=configuration();c[key]=value
                with self.assertRaises(ValueError):lifecycle.validate_config(reseal(c),check_files=False)

    def test_barrier_paths_cannot_be_overridden_by_runner(self):
        for flag in ('--output-dir','--ready-file','--release-file','--out-dir','--cpu-ready-file'):
            c=configuration();c['runner_argv'] += [flag+'=/evil']
            with self.assertRaises(ValueError):lifecycle.validate_config(reseal(c),check_files=False)

    def test_same_pid_new_starttime_or_argv_is_rejected(self):
        original=holder()
        for key,value in [('start_ticks',999),('argv',['/evil/python'])]:
            changed=copy.deepcopy(original);changed['process_identities']['121'][key]=value
            with self.assertRaises(ValueError):lifecycle.validate_holder_capture(changed,'/holder',original)

    def test_missing_foreign_duplicate_or_wrong_uuid_context_rejected(self):
        for mode in ('missing','foreign','duplicate','uuid'):
            c=holder()
            if mode=='missing':c['snapshot']['compute_apps'].pop()
            elif mode=='foreign':c['snapshot']['compute_apps'][0]['pid']=999
            elif mode=='duplicate':c['snapshot']['compute_apps'][0]['uuid']='GPU-1'
            else:
                c['snapshot']['gpus'][0]['uuid']='NEW-GPU';c['snapshot']['compute_apps'][0]['uuid']='NEW-GPU'
            with self.assertRaises(ValueError):lifecycle.validate_holder_capture(c,'/holder',holder())

    def test_holder_center_amplitude_and_all_workers_strict(self):
        for mutation in ('target','amplitude','parent'):
            c=holder()
            if mutation=='target':c['metadata']['target_util_percent']=90
            elif mutation=='amplitude':c['metadata']['variation_amplitude_percent']=0
            else:c['process_identities']['121']['ppid']=999
            with self.assertRaises(ValueError):lifecycle.validate_holder_capture(c,'/holder')

    def test_pidfd_does_not_signal_recycled_identity(self):
        expected=holder()['process_identities']['120'];changed=dict(expected,start_ticks=999)
        with patch.object(lifecycle.os,'pidfd_open',return_value=44,create=True),\
             patch.object(lifecycle.signal,'pidfd_send_signal',create=True) as send,\
             patch.object(lifecycle.os,'close') as close,patch.object(lifecycle,'proc_identity',return_value=changed):
            with self.assertRaises(ValueError):lifecycle.signal_exact(expected,signal.SIGTERM)
            send.assert_not_called();close.assert_called_once_with(44)

    def test_actual_37_samples_and_no_restore_call(self):
        now=[0.];calls=[]
        def sleep(seconds):now[0]+=seconds
        def sample(config,expected=None):
            c=holder();c['snapshot']['observed_unix']=now[0];calls.append(expected);return c
        with patch.object(lifecycle,'capture_holder',side_effect=sample):
            result=lifecycle.verify_restored(configuration(),sleep=sleep,clock=lambda:now[0])
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['samples']),37)
        self.assertEqual(result['elapsed_s'],180);self.assertEqual(len(calls),39)
        self.assertTrue(all(x['in_band']==37 for x in result['summary'].values()))

    def test_outside_band_is_not_reported_passed(self):
        now=[0.];n=[0]
        def sleep(seconds):now[0]+=seconds
        def sample(config,expected=None):
            c=holder();c['snapshot']['observed_unix']=now[0];n[0]+=1
            if n[0]==8:c['snapshot']['gpus'][3]['utilization']=95
            return c
        with patch.object(lifecycle,'capture_holder',side_effect=sample):
            result=lifecycle.verify_restored(configuration(),sleep=sleep,clock=lambda:now[0])
        self.assertEqual(result['status'],'outside_band');self.assertEqual(result['summary']['3']['in_band'],36)

    def test_supervisor_failure_still_verifies_restoration(self):
        for code in (0,1):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as directory:
                config=configuration();config['output_dir']=directory+'/wrapper';config['cwd']=directory
                original=lifecycle.supervisor_directory(config)
                class Child:
                    pid=999;returncode=code
                    def poll(self):return self.returncode
                def spawn(command,**kwargs):
                    original.mkdir();(original/'supervisor.json').write_text(json.dumps(dict(holder_restored=True,
                        holder_helper_sha256=lifecycle.ORIGINAL_PINS['run_segpaged_experiment.py'],
                        supervisor_sha256=lifecycle.ORIGINAL_PINS['glm53_gpu_lifecycle.py'])))
                    self.assertNotIn('LD_LIBRARY_PATH',kwargs['env']);return Child()
                expected=dict(pid=999,ppid=1,pgid=999,sid=999,start_ticks=100,argv=[])
                def identity(pid):return dict(expected,argv=lifecycle.supervisor_command(config))
                with patch.object(lifecycle,'SHARED_LOCK',directory+'/lock'),\
                     patch.object(lifecycle,'HOLDER_PYTHON',sys.executable),\
                     patch.object(lifecycle,'validate_config',return_value=True),\
                     patch.object(lifecycle,'capture_holder',return_value=holder()),\
                     patch.object(lifecycle.subprocess,'Popen',side_effect=spawn),\
                     patch.object(lifecycle,'proc_identity',side_effect=identity),\
                     patch.object(lifecycle,'verify_restored',return_value=dict(status='passed')) as verify:
                    result=lifecycle.run(config)
                terminal=lifecycle.read_json(Path(config['output_dir'])/'terminal.json',sealed=True)
                self.assertEqual(result,0 if code==0 else 1);verify.assert_called_once()
                self.assertTrue(terminal['holder_restored']);self.assertTrue(terminal['holder_live_verification_passed'])
                self.assertEqual(terminal['status'],'complete' if code==0 else 'failed_with_holder_restored')


if __name__=='__main__':unittest.main()
