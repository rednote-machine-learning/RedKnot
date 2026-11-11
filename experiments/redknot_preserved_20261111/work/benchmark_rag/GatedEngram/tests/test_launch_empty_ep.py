"""CPU-only v3 launch fencing; this file never launches a GPU job."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from gpu_pilot import launch_ffn_only_pilot as launcher
from tests import test_ffn_only_pilot as pilot_tests


class EmptyEPLaunchProofTests(unittest.TestCase):
    def fixture(self):
        names = ["tests.test_ffn_only_pilot.Contracts.test_" + str(i) for i in range(6)]
        names += ["tests.test_empty_ep_qualification.Contracts.test_" + str(i) for i in range(4)]
        sources = {"gpu_pilot/empty_ep_qualification.py": "a" * 64}
        plan = dict(cpu_proof_path="/independent/new/FFN_ONLY_CPU_TESTS.json", stage_sources=sources)
        proof = dict(successful=True, cuda_initialized=False, failures=0, errors=0,
                     skipped=0, source_sha256=copy.deepcopy(sources), tests=len(names), test_ids=names)
        return plan, proof

    def test_fresh_cpu_proof_includes_empty_ep_contract_tests(self):
        plan, proof = self.fixture()
        with patch.object(launcher.lifecycle, "read_sealed", return_value=proof):
            launcher.verify_cpu_proof(plan)

    def test_generic_ffn_cpu_proof_without_empty_ep_tests_is_rejected(self):
        plan, proof = self.fixture()
        proof["test_ids"] = [name.replace("test_empty_ep_qualification", "test_generic")
                             for name in proof["test_ids"]]
        with patch.object(launcher.lifecycle, "read_sealed", return_value=proof), self.assertRaises(ValueError):
            launcher.verify_cpu_proof(plan)

    def test_stale_skipped_or_gpu_initialized_proof_is_rejected(self):
        for field, value in (("source_sha256", {}), ("skipped", 1), ("cuda_initialized", True),
                             ("successful", False), ("errors", 1), ("failures", 1)):
            plan, proof = self.fixture()
            proof[field] = value
            with self.subTest(field=field), patch.object(launcher.lifecycle, "read_sealed", return_value=proof), \
                 self.assertRaises(ValueError):
                launcher.verify_cpu_proof(plan)

    def test_duplicate_test_ids_cannot_inflate_proof_count(self):
        plan, proof = self.fixture()
        proof["test_ids"][-1] = proof["test_ids"][-2]
        with patch.object(launcher.lifecycle, "read_sealed", return_value=proof), self.assertRaises(ValueError):
            launcher.verify_cpu_proof(plan)


class EmptyEPPredecessorFenceTests(unittest.TestCase):
    def test_changed_supervisor_file_hash_rejected_before_any_launch(self):
        root, sealed, state, *_ = pilot_tests.FailedPredecessorTests().fixture()
        with patch.object(launcher.lifecycle, "read_sealed", side_effect=lambda path: sealed[Path(path)]), \
             patch.object(launcher.lifecycle, "read_json", return_value=state), \
             patch.object(launcher.lifecycle, "file_digest", return_value="changed"), \
             patch.object(launcher.subprocess, "Popen") as popen, \
             self.assertRaisesRegex(ValueError, "cleanup state"):
            launcher.verify_failed_predecessor(root)
        popen.assert_not_called()

    def test_unqualified_present_old_worker_pid_is_not_signalled_or_ignored(self):
        root, sealed, state, *_ = pilot_tests.FailedPredecessorTests().fixture()
        def proc(pid):
            if pid == 3803509:
                return dict(start_ticks=123456789)
            raise FileNotFoundError(pid)
        with patch.object(launcher.lifecycle, "read_sealed", side_effect=lambda path: sealed[Path(path)]), \
             patch.object(launcher.lifecycle, "read_json", return_value=state), \
             patch.object(launcher.lifecycle, "file_digest", return_value="s" * 64), \
             patch.object(launcher.lifecycle, "proc_identity", side_effect=proc), \
             patch.object(launcher.subprocess, "Popen") as popen, \
             self.assertRaisesRegex(ValueError, "ownership unqualified"):
            launcher.verify_failed_predecessor(root)
        popen.assert_not_called()

    def test_original_cleanup_warning_cannot_be_silently_erased(self):
        fixture = pilot_tests.FailedPredecessorTests()
        with self.assertRaisesRegex(ValueError, "cleanup state"):
            fixture.check(lambda values: values[2].update(cleanup_warnings=[]))

    def test_new_name_and_stage_do_not_reuse_prior_output_identity(self):
        self.assertEqual(launcher.NAME, "kce-ffn-only-20261104-v3")
        self.assertEqual(launcher.REMOTE_STAGE,
                         launcher.ROOT / "gated_engram_ffn_only_20261104_v3/GatedEngram")
        self.assertNotEqual(launcher.NAME, launcher.FAILED_PREDECESSOR)


if __name__ == "__main__":
    unittest.main()
