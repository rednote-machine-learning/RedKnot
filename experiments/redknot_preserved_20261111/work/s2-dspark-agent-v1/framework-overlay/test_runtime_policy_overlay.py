import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import run_official_retry as bootstrap

LOCAL_FRAMEWORK = Path(__file__).resolve().parents[2] / 's2-agent-v1/framework'
FRAMEWORK = LOCAL_FRAMEWORK if LOCAL_FRAMEWORK.is_dir() else bootstrap.FROZEN_FRAMEWORK_ROOT


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = bootstrap.load_official(FRAMEWORK)
        cls.policy = sys.modules['runtime_policy']

    def official_forward_names(self):
        source = ast.parse((FRAMEWORK / 'benchmarks/benchmarks/utils/laminar.py').read_text())
        return next(ast.literal_eval(node.value) for node in source.body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LMNR_ENV_VARS' for t in node.targets))

    def test_actual_official_forward_signature_filtered_without_mutating_input(self):
        names = self.official_forward_names()
        self.assertEqual(set(names), {'LMNR_PROJECT_API_KEY', 'LMNR_SPAN_CONTEXT'})
        kwargs = dict(forward_env=[*names, 'DEBUG'], working_dir='/workspace')
        with mock.patch.dict(os.environ, {'LMNR_SPAN_CONTEXT': '{"trace_id":"cpu-fixture"}'}, clear=True):
            with self.policy.admitted_environment() as receipt:
                actual = self.policy.workspace_kwargs(kwargs, 0)
                self.assertFalse(receipt['external_tracing_enabled'])
        self.assertEqual(kwargs['forward_env'], [*names, 'DEBUG'])
        self.assertEqual(actual['forward_env'], ['DEBUG', 'SESSION_API_KEY', 'PYTHONPATH', 'LITELLM_LOCAL_MODEL_COST_MAP'])
        self.assertFalse(actual['enable_gpu']); self.assertFalse(actual['use_fakeroot'])
        self.assertEqual(actual['extra_bind_mounts'], [str(FRAMEWORK / 'runtime_guard') + ':/s2_runtime_guard:ro'])

    def test_real_tracing_key_rejected_both_before_and_after_admission(self):
        with mock.patch.dict(os.environ, {'LMNR_PROJECT_API_KEY': 'fake-key'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'LMNR_PROJECT_API_KEY'):
                with self.policy.admitted_environment():
                    self.fail('tracing key accepted')
        with mock.patch.dict(os.environ, {}, clear=True), self.policy.admitted_environment():
            os.environ['LMNR_PROJECT_API_KEY'] = 'late-key'
            with self.assertRaisesRegex(ValueError, 'External Laminar'):
                self.policy.workspace_kwargs({'forward_env': self.official_forward_names()}, 0)

    def test_unknown_forward_and_mount_permissions_remain_rejected(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.policy.admitted_environment():
            for kwargs in ({'forward_env': ['SOME_PROVIDER_KEY']}, {'enable_gpu': True},
                           {'use_fakeroot': True}, {'extra_bind_mounts': ['/host:/host']}):
                with self.assertRaises(ValueError):
                    self.policy.workspace_kwargs(kwargs, 0)

    def test_frozen_original_paths_configs_and_thread_worker_share_overlay(self):
        self.assertEqual(self.official.B, FRAMEWORK)
        self.assertEqual(self.policy.B, FRAMEWORK)
        self.assertEqual(self.policy.GUARD, FRAMEWORK / 'runtime_guard')
        self.assertIs(self.official.workspace_kwargs, self.policy.workspace_kwargs)
        self.assertEqual(self.policy.validate_config(FRAMEWORK / 'config/llm.Full.json')['max_output_tokens'], None)
        source = ast.parse((FRAMEWORK / 'benchmarks/benchmarks/utils/evaluation.py').read_text())
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'ThreadPoolExecutor' for node in ast.walk(source)))
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'prepare_workspace' and any(k.arg == 'forward_env'
            and isinstance(k.value, ast.Name) and k.value.id == 'LMNR_ENV_VARS' for k in node.keywords)
            for node in ast.walk(source)))
        with mock.patch.dict(os.environ, {}, clear=True), self.policy.admitted_environment(), ThreadPoolExecutor(max_workers=1) as workers:
            result = workers.submit(self.official.workspace_kwargs, {'forward_env': self.official_forward_names()}, 0).result()
        self.assertNotIn('LMNR_SPAN_CONTEXT', result['forward_env'])

    def test_bootstrap_passes_cli_without_new_flags_or_main_patch(self):
        argv = ['--arm', 'Full', '--ordinal', '0', '--run']
        with mock.patch.object(bootstrap, 'FROZEN_FRAMEWORK_ROOT', FRAMEWORK), mock.patch.object(self.official, 'main', return_value='called') as main:
            self.assertEqual(bootstrap.main(argv), 'called')
        main.assert_called_once_with(argv)

    def test_preimported_other_policy_is_rejected(self):
        foreign = types.SimpleNamespace(__file__=str(FRAMEWORK / 'runtime_policy.py'))
        with mock.patch.dict(sys.modules, {'runtime_policy': foreign}):
            with self.assertRaisesRegex(RuntimeError, 'before the overlay'):
                bootstrap.install_overlay(FRAMEWORK)


if __name__ == '__main__':
    unittest.main()
