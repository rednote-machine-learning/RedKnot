import copy
import types
import unittest
from unittest import mock

import run_official_retry_greedy as entry


class GreedyAttemptTests(unittest.TestCase):
    def test_all_arms_same_public_flag_original_plan_unchanged(self):
        for arm in ('Full', 'S2', 'Prefix'):
            source = {'arm': arm, 'argv': ['swebench-infer', 'pinned-llm.json', '--max-retries', '3']}
            before = copy.deepcopy(source)
            actual = entry.single_attempt_plan(source)
            self.assertEqual(source, before)
            self.assertEqual(actual['argv'], before['argv'] + ['--n-critic-runs', '1'])
            self.assertTrue(actual['single_attempt_greedy'])
            self.assertEqual(actual['same_policy_for_arms'], ['Full', 'S2', 'Prefix'])

    def test_existing_explicit_flag_is_not_overridden(self):
        for flag in ('--n-critic-runs', '--n-critic-runs=3'):
            with self.assertRaises(ValueError):
                entry.single_attempt_plan({'argv': ['swebench-infer', flag, '3']})
            with mock.patch.object(entry, 'load_official') as loader:
                with self.assertRaises(ValueError):
                    entry.main([flag, '3'])
                loader.assert_not_called()

    def test_main_calls_original_and_restores_plan_on_success_and_failure(self):
        for failure in (False, True):
            original = lambda args: {'argv': ['swebench-infer', 'pinned.json']}
            official = types.SimpleNamespace(build_plan=original)
            def main(argv):
                self.assertEqual(argv, ['--arm', 'Full'])
                result = official.build_plan(None)
                self.assertEqual(result['argv'][-2:], ['--n-critic-runs', '1'])
                if failure:
                    raise RuntimeError('original-main-fixture-error')
                return result
            official.main = main
            with mock.patch.object(entry, 'load_official', return_value=official):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'original-main-fixture-error'):
                        entry.main(['--arm', 'Full'])
                else:
                    self.assertTrue(entry.main(['--arm', 'Full'])['single_attempt_greedy'])
            self.assertIs(official.build_plan, original)


if __name__ == '__main__':
    unittest.main()
