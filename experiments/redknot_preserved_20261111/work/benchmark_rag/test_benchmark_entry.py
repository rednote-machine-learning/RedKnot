"""Admission/launch separation tests; all future execution tools are mocked."""
from contextlib import redirect_stdout
import io
import json
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

import benchmark_rag_redknot as entry


class EntryTests(unittest.TestCase):
    def test_interrupt_targets_original_supervisor_from_current_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'Lifecycle').mkdir()
            wrapper=dict(pid=10,argv=['holder-python','wrapper'])
            supervisor=dict(pid=11,ppid=10,argv=['holder-python','original-supervisor'])
            config={'sha256':'a'*64}
            entry.write_sealed(root/'Lifecycle/supervisor-started.json',dict(config_sha256=config['sha256'],identity=supervisor))
            lifecycle=NS(proc_identity=Mock(return_value=wrapper),supervisor_command=Mock(return_value=supervisor['argv']),
                signal_exact=Mock(return_value=True))
            process=NS(poll=Mock(return_value=None))
            self.assertTrue(entry.request_supervisor_stop(process,wrapper,config,root,lifecycle))
            lifecycle.signal_exact.assert_called_once_with(supervisor,signal.SIGTERM)
            receipt=entry.read_sealed(root/'interrupt-request.json')
            self.assertFalse(receipt['wrapper_signaled']);self.assertTrue(receipt['signal_sent_by_inference_python'])
            self.assertEqual(receipt['supervisor_identity']['pid'],11)

    def test_interrupt_rejects_unbound_supervisor_before_signaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'Lifecycle').mkdir()
            wrapper={'pid':10};config={'sha256':'a'*64}
            entry.write_sealed(root/'Lifecycle/supervisor-started.json',dict(config_sha256='b'*64,
                identity=dict(pid=11,ppid=10,argv=['original'])))
            lifecycle=NS(proc_identity=Mock(return_value=wrapper),signal_exact=Mock())
            with self.assertRaisesRegex(ValueError,'different run'):
                entry.request_supervisor_stop(NS(poll=lambda:None),wrapper,config,root,lifecycle)
            lifecycle.signal_exact.assert_not_called()

    def test_import_and_default_help_never_launch_or_read_runtime(self):
        code='import sys; import benchmark_rag_redknot; assert "torch" not in sys.modules and "deepinfer" not in sys.modules'
        subprocess.run([sys.executable,'-B','-c',code],cwd=entry.HERE,check=True,capture_output=True)
        with patch.object(entry,'load_definition',side_effect=AssertionError('default touched runtime')), \
            redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry.main([]),0)
        self.assertIn('--check-only',output.getvalue());self.assertIn('--run',output.getvalue())

    def test_check_only_cannot_reach_holder_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            definition={'child_python':sys.executable}
            prepared={'config':{'evaluation_case_indices':[3,2]}}
            with patch.object(entry,'load_definition',return_value=definition), \
                patch.object(entry,'cpu_preflight',return_value=prepared) as preflight, \
                patch.object(entry,'explicit_run',side_effect=AssertionError('check-only touched holder')), \
                redirect_stdout(io.StringIO()) as output:
                self.assertEqual(entry.main(['--length','both','--cases','1','--check-only',
                    '--output-root',directory]),0)
            value=json.loads(output.getvalue())
            self.assertFalse(value['gpu_initialized']);self.assertFalse(value['gpu_released'])
            self.assertEqual(value['evaluation_case_indices'],[3,2]);preflight.assert_called_once()

    def test_explicit_run_reaches_real_run_orchestrator_after_cpu_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            definition={'child_python':sys.executable};prepared={'config':{'evaluation_case_indices':[2]}}
            order=[]
            with patch.object(entry,'load_definition',return_value=definition), \
                patch.object(entry,'cpu_preflight',side_effect=lambda *a:(order.append('cpu'),prepared)[1]), \
                patch.object(entry,'explicit_run',side_effect=lambda *a:(order.append('run'),0)[1]) as run:
                self.assertEqual(entry.main(['--run','--output-root',directory]),0)
            self.assertEqual(order,['cpu','run']);run.assert_called_once()

    def test_bootstrap_argv_retains_native_profile_and_barrier_ownership(self):
        definition={key:'/pinned/'+key for key in ('child_python','experiment_root','manifest',
            'pilot_protocol','model_path')}
        argv=entry.native_argv(definition,Path('/pinned/config.json'))
        for key,value in {'--ep':'8','--dp':'8','--cp':'1','--max-model-len':'270336',
            '--max-num-batched-tokens':'8192','--batch-invariant':'1','--sampler-force-topk':'1024',
            '--s12-config':'/pinned/config.json'}.items():
            self.assertEqual(argv[argv.index(key)+1],value)
        for key in ('--ready-file','--release-file','--output-dir','--round-deadline-unix'):
            self.assertNotIn(key,argv)

    def test_future_gpu_run_requires_passed_source_bound_cpu_proof(self):
        with self.assertRaisesRegex(ValueError,'focused CPU proof'):
            entry.admitted_package_cpu_proof({})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'proof.json';pins={'source.py':'a'*64}
            entry.write_sealed(path,dict(schema='benchmark-rag-focused-cpu-proof-v1',passed=True,
                tests=7,failures=0,errors=0,skips=0,cuda_initialized=False,source_sha256=pins))
            definition=dict(package_cpu_proof_ref=entry.reference(path),package_source_sha256=pins)
            self.assertEqual(entry.admitted_package_cpu_proof(definition)['required_fields']['passed'],True)
            definition['package_source_sha256']={'source.py':'b'*64}
            with self.assertRaisesRegex(ValueError,'detached'):entry.admitted_package_cpu_proof(definition)

    def test_missing_cpu_proof_fails_before_any_gpu_or_holder_query(self):
        import profile_lifecycle
        with patch.object(profile_lifecycle,'prepare_config',side_effect=AssertionError('queried holder')):
            with self.assertRaisesRegex(ValueError,'focused CPU proof'):
                entry.explicit_run({},NS(),Path('/unused'),{})


if __name__=='__main__':unittest.main()
