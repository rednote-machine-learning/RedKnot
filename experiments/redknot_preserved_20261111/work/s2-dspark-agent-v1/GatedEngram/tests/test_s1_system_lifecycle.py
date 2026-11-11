"""Owned-session cleanup and the no-holder policy, without GPU/process side effects."""
import unittest
from pathlib import Path
from unittest.mock import patch
from gpu_pilot import s1_system_lifecycle as module


class LifecycleTests(unittest.TestCase):
    def test_retry_uses_remaining_round_time_not_fresh_two_hours(self):
        plan={'timeout_seconds':7200,'absolute_deadline_unix':10000.}
        self.assertEqual(module.release_deadlines(plan,9000.,20000.),(10000.,21000.))

    def test_expired_original_round_cannot_release_gpu(self):
        plan={'timeout_seconds':7200,'absolute_deadline_unix':10000.}
        for now in (10000.,10001.):
            with self.assertRaises(TimeoutError):module.release_deadlines(plan,now,20000.)

    def test_relative_maximum_still_caps_longer_absolute_deadline(self):
        plan={'timeout_seconds':7200,'absolute_deadline_unix':50000.}
        self.assertEqual(module.release_deadlines(plan,9000.,20000.),(16200.,27200.))

    def member(self,pid=100,start=20,sid=100):
        return dict(pid=pid,ppid=1,pgid=sid,sid=sid,start_ticks=start,state='S',argv=[])

    def test_no_holder_execution_branch(self):
        source=Path(module.__file__).read_text()
        self.assertNotIn('start_holder(',source)
        self.assertNotIn('restore_holder(',source)
        self.assertEqual(module.POLICY,'leave_stopped_never_restore')

    def test_identity_reuse_refused(self):
        old=self.member()
        with patch.object(module,'identity',return_value=self.member(start=21)):
            self.assertFalse(module.same_process(old))
            self.assertEqual(module.owned_members(old),[])

    def test_signal_rechecks_start_ticks(self):
        with patch.object(module,'identity',return_value=self.member(start=21)),patch.object(module.os,'kill') as kill:
            module.signal_member(self.member(),15)
            kill.assert_not_called()

    def test_signal_only_same_session(self):
        with patch.object(module,'identity',return_value=self.member(sid=101)),patch.object(module.os,'kill') as kill:
            module.signal_member(self.member(),15)
            kill.assert_not_called()

    def test_owned_signal(self):
        with patch.object(module,'identity',return_value=self.member()),patch.object(module.os,'pidfd_open',return_value=501,create=True),patch.object(module.signal,'pidfd_send_signal',create=True) as send,patch.object(module.os,'close') as close:
            module.signal_member(self.member(),15)
            send.assert_called_once_with(501,15)
            close.assert_called_once_with(501)

    def test_pid_recycled_between_checks_cannot_receive_signal(self):
        with patch.object(module,'identity',side_effect=[self.member(),self.member(start=21)]),patch.object(module.os,'pidfd_open',return_value=501,create=True),patch.object(module.signal,'pidfd_send_signal',create=True) as send,patch.object(module.os,'close') as close:
            module.signal_member(self.member(),15)
            send.assert_not_called()
            close.assert_called_once_with(501)

    def test_session_leader_required(self):
        with self.assertRaises(ValueError):module.cleanup(self.member(pid=101,sid=100))

    def test_cleanup_reaps_only_recorded_session(self):
        child=self.member();worker=self.member(pid=102,start=21)
        with patch.object(module,'owned_members',side_effect=[[worker],[],[],[],[]]),patch.object(module,'signal_member') as send:
            result=module.cleanup(child)
        self.assertEqual(result,[dict(pid=102,start_ticks=21,sid=100)])
        send.assert_called_once_with(worker,module.signal.SIGTERM)

    def test_empty_compute_context_is_required_for_idle(self):
        rows='\n'.join(f'{i}, GPU-{i}, NVIDIA L20X, 0, 0' for i in range(8))
        with patch.object(module.subprocess,'check_output',side_effect=[rows,'99, GPU-0, python\n']):
            self.assertFalse(module.gpu_snapshot()['idle'])

    def test_wrong_gpu_set_rejected(self):
        with patch.object(module.subprocess,'check_output',side_effect=['0, GPU-0, NVIDIA L20X, 0, 0','']):
            with self.assertRaises(ValueError):module.gpu_snapshot()
