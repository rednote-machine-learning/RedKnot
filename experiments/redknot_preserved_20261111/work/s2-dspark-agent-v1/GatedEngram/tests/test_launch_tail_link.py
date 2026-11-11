"""Independent CPU launcher review: fixtures only; all process/GPU APIs mocked."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_pilot import launch_tail_link_pilot as launch
from gpu_pilot.sensitivity_protocol import sensitivity_protocol
from gpu_pilot.tail_link_protocol import tail_link_protocol


class TailLinkLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tail-link-launch-cpu-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage = self.root / "independent-tail/GatedEngram"
        self.native = self.stage.parent / "NativeSnapshot"
        (self.root / "results").mkdir()
        for relative in (
            "gpu_pilot/run_pilot.py", "gpu_pilot/launch_pilot.py",
            "gpu_pilot/run_tail_link_pilot.py", "gpu_pilot/read_tail_link_results.py",
            "gpu_pilot/tail_link_protocol.py", "gpu_pilot/launch_tail_link_pilot.py",
            "kv_cache_engram/tail_link.py", "kv_cache_engram/tail_link_plan.py",
            "tests/test_launch_tail_link.py",
        ):
            path = self.stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# CPU fixture, never executed\n")
        for relative in ("setup.py", "README.md", "deepinfer/model_executor/models/v4_modules/attn.py"):
            path = self.native / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# immutable native fixture\n")
        self.manifest = self.stage / "manifest.json"
        self.protocol = self.stage / "protocol.json"
        self.sensitivity = self.stage / "sensitivity-protocol.json"
        self.tail_protocol = self.stage / "tail-link-protocol.json"
        self.write_sealed(self.protocol, dict(schema=launch.lifecycle.PROTOCOL_SCHEMA))
        self.sensitivity.write_bytes(launch.lifecycle.canonical(sensitivity_protocol()))
        self.tail_protocol.write_bytes(launch.lifecycle.canonical(tail_link_protocol()))
        self.write_sealed(self.manifest, dict(schema="tail-cpu-input-v1",
            artifact_kind="kv_cache_engram_native_pilot",
            pilot_protocol_sha256=launch.lifecycle.read_json(self.protocol)["sha256"],
            sensitivity_protocol_sha256=sensitivity_protocol()["sha256"]))
        self.predecessor = dict(name=launch.PREDECESSOR, terminal=True, model_results_reused=False,
                               state_file_sha256="state", result_file_sha256="result",
                               result_seal_sha256=launch.PREDECESSOR_RESULT_SEAL)

    @staticmethod
    def write_sealed(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(launch.lifecycle.canonical(launch.lifecycle.seal(body)))

    @staticmethod
    def reseal(value):
        return launch.lifecycle.seal({key: item for key, item in value.items() if key != "sha256"})

    def proof(self):
        families = ["test_tail_link_plan", "test_tail_link_controller", "test_tail_link_pilot",
                    "test_tail_link_results", "test_launch_tail_link", "test_index_hit20_empty_ep_qualification",
                    "test_dynamic_tail_controller", "test_dynamic_tail_runner", "test_dynamic_tail_results"]
        names = [f"tests.{family}.Fixture.test_required" for family in families]
        names.extend(f"tests.test_other.Fixture.test_{i}" for i in range(1160 - len(names)))
        return dict(schema="tail-link-fullmodel-cpu-proof-v1", successful=True,
                    cuda_initialized=False, cuda_init_guard_installed=True,
                    failures=0, errors=0, skipped=0, tests=len(names), test_ids=names,
                    source_sha256=launch.lifecycle.source_files(self.stage))

    def plan(self):
        self.write_sealed(self.stage / "DYNAMIC_TAIL_CPU_TESTS-v2.json", self.proof())
        with mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor), \
             mock.patch.object(launch, "verify_previous_dynamic_terminal", return_value={"status": "failed"}):
            return launch.build_plan(self.root, self.stage, self.manifest, self.protocol,
                                     self.tail_protocol, self.sensitivity)

    def verify(self, plan):
        with mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor), \
             mock.patch.object(launch, "verify_previous_dynamic_terminal", return_value={"status": "failed"}):
            launch.verify_plan(plan)

    def terminal(self, plan):
        return dict(schema="redknot-glm53-gpu-lifecycle-v1",
            status="runner_completed; consult benchmark quality/performance gates",
            worker_exit=0, holder_restored=True, finished_unix=123., gpu_ids=list(range(8)),
            supervisor_sha256=launch.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launch.lifecycle.HELPER_SHA,
            child_environment_sha256=launch.lifecycle.ENV_SHA,
            runner_sha256=plan["stage_sources"]["gpu_pilot/run_tail_link_pilot.py"])

    def test_build_plan_pins_finite_original_lifecycle_and_private_repo(self):
        plan = self.plan()
        self.verify(plan)
        argv = plan["argv"]
        self.assertEqual(argv[:2], ["/root/miniconda3/bin/python", str(launch.lifecycle.SUPERVISOR)])
        self.assertEqual(argv[argv.index("--repo") + 1], str(self.native))
        self.assertEqual(argv[argv.index("--") + 1], str(self.stage / "gpu_pilot/run_tail_link_pilot.py"))
        self.assertEqual(argv[argv.index("--gpus") + 1], "0,1,2,3,4,5,6,7")
        self.assertEqual(plan["timeout"], 21600)
        self.assertEqual(plan["prepare_timeout"], 300)
        self.assertTrue(plan["no_automatic_retry"])
        self.assertTrue(plan["live_workspace_not_execution_source"])
        self.assertEqual(plan["first_full_layers"], [0, 1, 2])
        self.assertEqual(plan["tail_layers"], list(range(3, 40)))
        self.assertEqual(plan["evaluation_indices"], [2, 3, 4, 5])
        self.assertEqual(plan["evaluation_arms"], ["recomputed", "tail128_link", "tail128_dynamic"])
        self.assertEqual(plan["dynamic_refresh_start_layer"], 4)
        self.assertNotIn(str(self.root / "DeepInfer-PR148"), argv)

    def test_check_only_does_not_query_gpu_launch_or_publish(self):
        plan = self.plan()
        with mock.patch.object(launch, "build_plan", return_value=plan), \
             mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor), \
             mock.patch.object(launch, "verify_previous_dynamic_terminal", return_value={"status": "failed"}), \
             mock.patch.object(launch, "run_plan", side_effect=AssertionError("forbidden launch")) as run, \
             mock.patch.object(launch.subprocess, "Popen", side_effect=AssertionError("forbidden process")) as popen, \
             mock.patch.object(launch.lifecycle, "holder_snapshot", side_effect=AssertionError("forbidden GPU")) as holder, \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(launch.main(["--check-only"]), 0)
        self.assertEqual(json.loads(stream.getvalue())["sha256"], plan["sha256"])
        run.assert_not_called(); popen.assert_not_called(); holder.assert_not_called()
        self.assertFalse(Path(plan["paths"]["plan"]).exists())

    def test_cpu_proof_rejects_failed_skipped_cuda_stale_or_bad_count(self):
        plan = self.plan()
        for update in (dict(successful=False), dict(cuda_initialized=True),
                       dict(cuda_init_guard_installed=False), dict(failures=1), dict(errors=1),
                       dict(skipped=1), dict(source_sha256={}), dict(tests=1159), dict(tests=True),
                       dict(test_ids=["duplicate"] * 1160)):
            proof = self.proof(); proof.update(update)
            self.write_sealed(Path(plan["cpu_proof_path"]), proof)
            with self.subTest(update=update), self.assertRaises(ValueError):
                launch.verify_cpu_proof(plan)

    def test_cpu_proof_requires_each_new_test_family(self):
        plan = self.plan()
        for family in ("test_tail_link_plan.", "test_tail_link_controller.", "test_tail_link_pilot.",
                       "test_tail_link_results.", "test_launch_tail_link.", "test_index_hit20_empty_ep_qualification."):
            proof = self.proof()
            proof["test_ids"] = [name.replace(family, "removed.") for name in proof["test_ids"]]
            self.write_sealed(Path(plan["cpu_proof_path"]), proof)
            with self.subTest(family=family), self.assertRaises(ValueError):
                launch.verify_cpu_proof(plan)

    def test_verify_terminal_accepts_only_success_and_pinned_identities(self):
        plan = self.plan(); good = self.terminal(plan)
        launch.verify_terminal(good, plan)
        updates = [dict(status="failed"), dict(schema="wrong"), dict(worker_exit=1),
                   dict(holder_restored=False), dict(finished_unix=None), dict(gpu_ids=[0]),
                   dict(error="failure"), dict(cleanup_error="failure"), dict(cleanup_warnings=["warning"])]
        updates += [{key: "foreign"} for key in ("supervisor_sha256", "holder_helper_sha256",
                                               "child_environment_sha256", "runner_sha256")]
        for update in updates:
            state = dict(good); state.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                launch.verify_terminal(state, plan)

    def test_private_snapshot_all_nongit_files_bound(self):
        plan = self.plan()
        self.assertIn("README.md", plan["native_snapshot_sources"])
        (self.native / ".git").mkdir(); (self.native / ".git/config").write_text("fixture")
        self.verify(plan)
        (self.native / "README.md").write_text("foreign content")
        with self.assertRaises(ValueError):
            self.verify(plan)

    def test_private_snapshot_new_or_removed_file_blocks(self):
        plan = self.plan()
        new = self.native / "added.bin"; new.write_bytes(b"new")
        with self.assertRaises(ValueError):
            self.verify(plan)
        new.unlink()
        (self.native / "setup.py").unlink()
        with self.assertRaises(ValueError):
            self.verify(plan)

    def test_private_symlink_or_live_repo_substitution_blocks(self):
        plan = self.plan()
        changed = copy.deepcopy(plan); changed["native_snapshot"] = str(self.root / "DeepInfer-PR148")
        with self.assertRaises(ValueError):
            self.verify(self.reseal(changed))
        link = self.native / "live.py"; link.symlink_to(self.native / "setup.py")
        with self.assertRaises(ValueError):
            self.verify(plan)

    def test_source_mutation_blocks_before_any_process_operation(self):
        plan = self.plan()
        (self.stage / "kv_cache_engram/tail_link.py").write_text("# altered\n")
        with mock.patch.object(launch.subprocess, "Popen") as popen, self.assertRaises(ValueError):
            launch.run_plan(plan)
        popen.assert_not_called()

    def test_dynamic_policy_cannot_be_changed_after_plan_freeze(self):
        original = self.plan()
        for key, value in (("dynamic_refresh_start_layer", 3),
                           ("native_yoco_swa_share_hidden", True),
                           ("cpu_proof_path", str(self.stage / "DYNAMIC_TAIL_CPU_TESTS-v1.json")),
                           ("evaluation_arms", ["recomputed", "tail128_dynamic"])):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(self.reseal(changed))

    def test_previous_fullmodel_workers_and_audit_must_be_finished(self):
        state = self.terminal(self.plan())
        state["runner_sha256"] = "c53bc6a0a7461f1d93ab2f8a71b802c1be6e97ad33e0e23efca853c4329e2586"
        run = dict(status="complete", completed_case_indices=[2, 3, 4, 5])
        group = dict(status="complete", owned_ranks_still_alive_after_cleanup=[],
                     workers=[dict(pid=2147483500 + i, alive=False, exitcode=0) for i in range(8)])
        with mock.patch.object(launch.lifecycle, "read_json", return_value=state), \
             mock.patch.object(launch.lifecycle, "read_sealed", side_effect=[run, group]), \
             mock.patch.object(Path, "exists", return_value=False), \
             mock.patch.object(Path, "iterdir", return_value=iter([])):
            result = launch.verify_previous_fullmodel_terminal(self.root)
        self.assertTrue(result["previous_workers_exited"])
        self.assertFalse(result["previous_scores_reused"])
        proc = mock.MagicMock()
        proc.name = "12345"
        proc.__truediv__.return_value.read_bytes.return_value = (
            b"python\0/read_tail_link_results.py\0--result-dir\0"
            + str(self.root / "results/kce-tail-link-fullmodel-20261105-v1/benchmark").encode() + b"\0")
        with mock.patch.object(launch.lifecycle, "read_json", return_value=state), \
             mock.patch.object(launch.lifecycle, "read_sealed", side_effect=[run, group]), \
             mock.patch.object(Path, "exists", return_value=False), \
             mock.patch.object(Path, "iterdir", return_value=iter([proc])), \
             self.assertRaisesRegex(ValueError, "CPU audit is still active"):
            launch.verify_previous_fullmodel_terminal(self.root)

    def test_required_tail_source_missing_blocks(self):
        (self.stage / "kv_cache_engram/tail_link_plan.py").unlink()
        with self.assertRaises(ValueError):
            self.plan()

    def failed_dynamic_fixture(self):
        state = self.terminal(self.plan())
        state.update(status="failed", worker_exit=1, child_pid=2147483400,
                     runner_sha256=launch.PREVIOUS_DYNAMIC_RUNNER_SHA,
                     cleanup_warnings=["_UnstableOwnership('refusing to signal unauthenticated process-group member 4022298')"])
        group = dict(schema="canonical-quality-owned-process-group-v1", status="failed",
                     reason="worker_failure_marker", owned_ranks_still_alive_after_cleanup=[],
                     only_owned_process_handles_signaled=True, sha256="group",
                     workers=[dict(rank=i, pid=2147483500+i, alive=False, exitcode=-15) for i in range(8)])
        outcome = dict(successful=False, automatic_retry=False, supervisor_exit=1, finished_unix=123.,
                       holder_physically_verified=True, holder_audit_sha256="after", sha256="outcome")
        after = dict(sha256="after")
        return state, group, outcome, after

    def check_failed_dynamic(self, state, group, outcome, after, *, exists=False):
        with mock.patch.object(launch.lifecycle, "read_json", return_value=state), \
             mock.patch.object(launch.lifecycle, "read_sealed", side_effect=[group, outcome, after]), \
             mock.patch.object(launch.lifecycle, "file_digest", return_value="state"), \
             mock.patch.object(Path, "exists", return_value=exists):
            return launch.verify_previous_dynamic_terminal(self.root)

    def test_failed_v1_requires_drained_workers_and_bound_physical_restoration(self):
        fixture = self.failed_dynamic_fixture()
        result = self.check_failed_dynamic(*fixture)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["previous_workers_exited"])
        self.assertFalse(result["previous_scores_reused"])
        self.assertEqual(result["original_cleanup_warnings"], fixture[0]["cleanup_warnings"])
        with self.assertRaisesRegex(ValueError, "still present"):
            self.check_failed_dynamic(*fixture, exists=True)

    def test_failed_v1_cannot_be_admitted_with_missing_cleanup_or_forged_success(self):
        fixture = self.failed_dynamic_fixture()
        for index, field, value in ((0, "holder_restored", False), (0, "status", "running"),
                                    (0, "runner_sha256", "foreign"), (0, "cleanup_warnings", ["new warning"]),
                                    (1, "owned_ranks_still_alive_after_cleanup", [0]),
                                    (2, "holder_physically_verified", False), (2, "holder_audit_sha256", "foreign"),
                                    (2, "successful", True)):
            changed = copy.deepcopy(fixture)
            changed[index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check_failed_dynamic(*changed)

    def test_changed_tail_protocol_even_resealed_blocks(self):
        plan = self.plan()
        body = tail_link_protocol(); body.pop("sha256"); body["unauthorized_change"] = True
        self.write_sealed(self.tail_protocol, body)
        with self.assertRaises(ValueError):
            self.verify(plan)

    def test_existing_evidence_forbids_retry_without_overwrite(self):
        plan = self.plan()
        for key, raw in plan["paths"].items():
            path = Path(raw); path.write_text("original evidence")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "already used"):
                self.plan()
            self.assertEqual(path.read_text(), "original evidence")
            path.unlink()

    def test_plan_seal_runner_repo_protocol_suffix_verified(self):
        original = self.plan()
        bad = copy.deepcopy(original); bad["timeout"] += 1
        with self.assertRaises(ValueError): self.verify(bad)
        for flag in ("--", "--repo", "--tail-link-protocol"):
            bad = copy.deepcopy(original); bad["argv"][bad["argv"].index(flag) + 1] = "/foreign/path"
            with self.subTest(flag=flag), self.assertRaises(ValueError): self.verify(self.reseal(bad))

    def test_resealed_full_argv_mutations_rejected(self):
        original = self.plan()
        for flag, value in (("--dp", "1"), ("--gpus", "0"), ("--timeout", "0"),
                            ("--max-num-seqs", "2"), ("--holder-helper", "/foreign/helper.py")):
            bad = copy.deepcopy(original); bad["argv"][bad["argv"].index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError): self.verify(self.reseal(bad))
        for index in (0, 1):
            bad = copy.deepcopy(original); bad["argv"][index] = "/foreign/program"
            with self.subTest(index=index), self.assertRaises(ValueError): self.verify(self.reseal(bad))

    def test_resealed_evidence_paths_and_retry_policy_rejected(self):
        original = self.plan()
        for key, value in (("no_automatic_retry", False), ("supervisor_owns_cpu_release_and_holder", False)):
            bad = copy.deepcopy(original); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.verify(self.reseal(bad))
        bad = copy.deepcopy(original); bad["paths"]["result_dir"] = str(self.root / "results/other")
        with self.assertRaises(ValueError): self.verify(self.reseal(bad))

    def predecessor_fixture(self):
        directory = self.root / "results" / launch.PREDECESSOR
        directory.mkdir(exist_ok=True)
        state = self.terminal(self.plan())
        state["runner_sha256"] = launch.PREDECESSOR_RUNNER_SHA
        result = launch.lifecycle.seal(dict(schema="tail-memory-operator-result-v1", status="complete",
            synthetic_operator_only=True, model_accuracy_measured=False, end_to_end_ttft_measured=False,
            lifecycle_identity=dict(parent_pid=2147483601, pid=2147483602)))
        (directory / "supervisor.json").write_text(json.dumps(state))
        (directory / "benchmark").mkdir(exist_ok=True)
        (directory / "benchmark/operator-result.json").write_bytes(launch.lifecycle.canonical(result))
        return directory, state, result

    def test_predecessor_operator_is_not_model_accuracy_evidence(self):
        directory, state, result = self.predecessor_fixture()
        with mock.patch.object(launch, "PREDECESSOR_RESULT_SEAL", result["sha256"]):
            evidence = launch.predecessor_evidence(self.root)
        self.assertTrue(evidence["terminal"])
        self.assertFalse(evidence["model_results_reused"])
        result.pop("sha256"); result["model_accuracy_measured"] = True
        result = launch.lifecycle.seal(result)
        (directory / "benchmark/operator-result.json").write_bytes(launch.lifecycle.canonical(result))
        with mock.patch.object(launch, "PREDECESSOR_RESULT_SEAL", result["sha256"]), self.assertRaises(ValueError):
            launch.predecessor_evidence(self.root)

    def test_predecessor_requires_successful_status_and_schema(self):
        directory, state, result = self.predecessor_fixture()
        for update in (dict(status="failed"), dict(schema="foreign"), dict(worker_exit=1),
                       dict(holder_restored=False), dict(runner_sha256="foreign")):
            bad = dict(state); bad.update(update)
            (directory / "supervisor.json").write_text(json.dumps(bad))
            with self.subTest(update=update), mock.patch.object(launch, "PREDECESSOR_RESULT_SEAL", result["sha256"]), self.assertRaises(ValueError):
                launch.predecessor_evidence(self.root)

    def test_run_failure_is_single_attempt_and_restoration_evidence_checked(self):
        plan = self.plan()
        child = SimpleNamespace(pid=2147483610, wait=mock.Mock(return_value=1))
        snapshot = launch.lifecycle.seal(dict(schema="holder-fixture-v1", physical=True))
        with mock.patch.object(launch, "verify_plan"), \
             mock.patch.object(launch, "verify_previous_fullmodel_terminal"), \
             mock.patch.multiple(launch, ROOT=self.root, REMOTE_STAGE=self.stage, STAGE=self.stage), \
             mock.patch.object(launch.lifecycle, "verify_lifecycle"), \
             mock.patch.object(launch.lifecycle, "holder_snapshot", return_value=snapshot) as holder, \
             mock.patch.object(launch.lifecycle, "proc_identity", return_value=dict(pid=child.pid, argv=plan["argv"])), \
             mock.patch.object(launch.subprocess, "Popen", return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            outcome = launch.run_plan(plan)
        self.assertFalse(outcome["successful"])
        self.assertFalse(outcome["automatic_retry"])
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(child.wait.call_count, 1)
        self.assertEqual(holder.call_count, 2)
        self.assertTrue(outcome["holder_physically_verified"])
        self.assertTrue(Path(plan["paths"]["outcome"]).is_file())

    def test_run_failure_never_claims_restoration_when_physical_audit_fails(self):
        plan = self.plan()
        child = SimpleNamespace(pid=2147483610, wait=mock.Mock(return_value=1))
        snapshot = launch.lifecycle.seal(dict(schema="holder-fixture-v1"))
        with mock.patch.object(launch, "verify_plan"), \
             mock.patch.object(launch, "verify_previous_fullmodel_terminal"), \
             mock.patch.multiple(launch, ROOT=self.root, REMOTE_STAGE=self.stage, STAGE=self.stage), \
             mock.patch.object(launch.lifecycle, "verify_lifecycle"), \
             mock.patch.object(launch.lifecycle, "holder_snapshot", side_effect=[snapshot, ValueError("GPU coverage missing")]), \
             mock.patch.object(launch.lifecycle, "proc_identity", return_value=dict(pid=child.pid, argv=plan["argv"])), \
             mock.patch.object(launch.subprocess, "Popen", return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            outcome = launch.run_plan(plan)
        self.assertFalse(outcome["successful"])
        self.assertFalse(outcome["holder_physically_verified"])
        self.assertIn("GPU coverage", outcome["holder_error"])
        self.assertEqual(popen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
