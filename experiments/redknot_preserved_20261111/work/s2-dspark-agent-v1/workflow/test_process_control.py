"""CPU-only regression for a real /proc enumeration -> exit -> environ race."""
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

import process_control as p


def sample():
    return dict(pid=1234, ppid=12, pgid=12, sid=12, start_ticks=987,
                boot_id='boot', cmdline_bytes_sha256='sha', exe='/fixture/compiler')


class MarkerAdmissionTests(unittest.TestCase):
    def test_gpu_queries_share_absolute_monotonic_budget(self):
        with mock.patch.object(p.time, 'monotonic', side_effect=[100., 108.]), mock.patch.object(p.subprocess, 'check_output', side_effect=['0, gpu0, 0\n', '']) as query:
            self.assertTrue(p.gpu_snapshot({'0': 'gpu0'}, deadline_monotonic=110.)['idle'])
        self.assertEqual([c.kwargs['timeout'] for c in query.call_args_list], [10, 2.])
        with mock.patch.object(p.time, 'monotonic', return_value=110.), mock.patch.object(p.subprocess, 'check_output') as query:
            with self.assertRaises(subprocess.TimeoutExpired):
                p.gpu_snapshot({'0': 'gpu0'}, deadline_monotonic=110.)
            query.assert_not_called()

    def test_live_marker_required_and_identity_diagnostic(self):
        value = sample()
        with mock.patch.object(p, 'identity', return_value=value), mock.patch.object(p, 'has_marker', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'stable live process.*"pid": 1234'):
                p.live_marked_identity(value, 'secret-marker-not-in-message')

    def test_exit_during_environ_read_is_not_a_live_unowned_process(self):
        value = sample()
        with mock.patch.object(p, 'identity', side_effect=[value, ProcessLookupError(1234)]), mock.patch.object(p, 'has_marker', return_value=False):
            self.assertIsNone(p.live_marked_identity(value, 'fixture'))

    def test_pid_reuse_does_not_become_authorized(self):
        value = sample()
        with mock.patch.object(p, 'identity', side_effect=[value, {**value, 'start_ticks': 988}]), mock.patch.object(p, 'has_marker', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'identity changed after'):
                p.live_marked_identity(value, 'fixture')

    def test_permission_failure_is_not_treated_as_exit(self):
        value = sample()
        with mock.patch.object(p, 'identity', return_value=value), mock.patch.object(p, 'has_marker', side_effect=PermissionError('fixture')):
            with self.assertRaises(PermissionError):
                p.live_marked_identity(value, 'fixture')


@unittest.skipUnless(sys.platform == 'linux' and Path('/proc/self/stat').exists(), 'actual Linux /proc regression requires Linux')
class LinuxRealProcessTests(unittest.TestCase):
    def child(self, marked=True):
        marker = 's2-real-process-cpu-' + str(os.getpid())
        env = dict(os.environ)
        env.pop('REDKNOT_GLM_RUN_ID', None)
        if marked:
            env['REDKNOT_GLM_RUN_ID'] = marker
        child = subprocess.Popen([sys.executable, '-B', '-c',
            "import sys; print('ready',flush=True); sys.stdin.buffer.read(1)"],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(child.stdout.readline(), b'ready\n')
        self.addCleanup(self.close_child, child)
        return child, marker

    @staticmethod
    def close_child(child):
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()

    def test_actual_marker_missing_live_child_still_fails_closed(self):
        child, marker = self.child(marked=False)
        with self.assertRaisesRegex(RuntimeError, 'Original lifecycle marker absent from stable live process'):
            p.descendants(p.identity(os.getpid()), marker)
        self.assertIsNone(child.poll())

    def test_actual_marked_child_is_authorized_without_changing_signal_scope(self):
        child, marker = self.child()
        found = p.descendants(p.identity(os.getpid()), marker)
        self.assertIn(child.pid, [v['pid'] for v in found])
        self.assertIsNone(child.poll())

    def test_actual_child_exits_between_enumeration_and_marker_read(self):
        child, marker = self.child()
        original = p.has_marker
        seen = []
        def exit_then_read(pid, expected_marker):
            if pid == child.pid:
                child.stdin.write(b'x'); child.stdin.flush()
                child.wait(timeout=5)
                result = original(pid, expected_marker)
                seen.append(result)
                return result
            return original(pid, expected_marker)
        with mock.patch.object(p, 'has_marker', side_effect=exit_then_read):
            found = p.descendants(p.identity(os.getpid()), marker)
        self.assertEqual(seen, [False])  # The old unbracketed predicate raised here.
        self.assertNotIn(child.pid, [v['pid'] for v in found])

    def test_actual_zombie_between_enumeration_and_marker_read(self):
        child, marker = self.child()
        original = p.has_marker
        seen = []
        def zombie_then_read(pid, expected_marker):
            if pid == child.pid:
                child.stdin.write(b'x'); child.stdin.flush()
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    state = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
                    if state == 'Z':
                        break
                    time.sleep(.001)
                self.assertEqual(state, 'Z')
                result = original(pid, expected_marker)
                seen.append(result)
                return result
            return original(pid, expected_marker)
        with mock.patch.object(p, 'has_marker', side_effect=zombie_then_read):
            found = p.descendants(p.identity(os.getpid()), marker)
        self.assertEqual(seen, [False])
        self.assertNotIn(child.pid, [v['pid'] for v in found])


if __name__ == '__main__':
    unittest.main()
