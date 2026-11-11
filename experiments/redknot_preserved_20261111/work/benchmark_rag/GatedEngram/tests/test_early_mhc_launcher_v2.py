"""CPU checks for the explicitly authorized v2 repair, never a generic retry."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import launch_early_mhc_pilot as launch


class AuthorizedRepairTests(unittest.TestCase):
    def fixtures(self):
        pause = dict(exit_verified=True, no_gpu_supervisor_started=True)
        outcome = dict(schema="early-mhc-launch-outcome-v1", sha256=launch.PREDECESSOR_OUTCOME_SEAL,
                       successful=False, supervisor_exit=1, holder_physically_verified=True)
        failure = dict(sha256=launch.PREDECESSOR_FAILURE_SEAL,
                       error="RuntimeError('called a synchronizing CUDA operation')", completed_indices=[])
        state = dict(schema="redknot-glm53-gpu-lifecycle-v1", status="failed", worker_exit=1,
            holder_restored=True, gpu_ids=list(range(8)), finished_unix=1.,
            supervisor_sha256=launch.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launch.lifecycle.HELPER_SHA,
            child_environment_sha256=launch.lifecycle.ENV_SHA,
            runner_sha256=launch.PREDECESSOR_RUNNER_SHA,
            error="RuntimeError('GLM benchmark failed: exit=1')",
            cleanup_warnings=["_UnstableOwnership('refusing to signal unauthenticated process-group member 3550938')"])
        started = dict(pid=3550690, start_ticks=2868345386)
        return [pause, outcome, failure, started, dict(started, pid=3550711)], state

    def verify(self, values, state, current=None):
        with tempfile.TemporaryDirectory(prefix="early-repair-launch-") as d:
            with patch.object(launch.lifecycle, "read_sealed", side_effect=values), \
                 patch.object(launch.lifecycle, "read_json", return_value=state), \
                 patch.object(launch.lifecycle, "proc_identity", side_effect=FileNotFoundError() if current is None else None,
                              return_value=current):
                launch.verify_predecessor(Path(d))

    def test_only_known_failure_with_restored_holder_is_accepted(self):
        self.verify(*self.fixtures())

    def test_unknown_or_successful_predecessor_is_not_generic_retry_permission(self):
        for update in ({"sha256":"a"*64}, {"successful":True}, {"holder_physically_verified":False},
                       {"supervisor_exit":0}):
            values, state = self.fixtures(); values[1].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError): self.verify(values, state)

    def test_failure_marker_or_old_completed_case_must_not_change(self):
        for update in ({"sha256":"a"*64}, {"completed_indices":[2]}, {"error":"different failure"}):
            values, state = self.fixtures(); values[2].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError): self.verify(values, state)

    def test_unrestored_wrong_runner_or_new_cleanup_error_blocks(self):
        for update in ({"holder_restored":False}, {"runner_sha256":"a"*64}, {"cleanup_error":"bad"},
                       {"cleanup_warnings":[]}, {"status":"running"}, {"gpu_ids":[0]}):
            values, state = self.fixtures(); state.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError): self.verify(values, state)

    def test_alive_authenticated_old_launcher_blocks_without_signals(self):
        values, state = self.fixtures()
        with self.assertRaisesRegex(ValueError, "still alive"):
            self.verify(values, state, dict(start_ticks=2868345386))

    def test_pause_must_remain_verified(self):
        values, state = self.fixtures(); values[0]["exit_verified"] = False
        with self.assertRaises(ValueError): self.verify(values, state)

    def test_new_unique_stage_and_original_supervisor_only(self):
        self.assertEqual(launch.NAME, "kce-early-mhc-20261102-v2")
        self.assertEqual(launch.REMOTE_STAGE.name, "GatedEngram")
        self.assertNotIn("DeepInfer-PR148", launch.REMOTE_STAGE.parts)
        self.assertEqual(launch.PREDECESSOR, "kce-early-mhc-20261102-v1")


if __name__ == "__main__":
    unittest.main()
