"""CPU-only isolated-launch contracts; every process/GPU entry is mocked."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import launch_index_hit20_pilot as launch
from gpu_pilot.sensitivity_protocol import sensitivity_protocol
from gpu_pilot.index_hit20_protocol import index_hit20_protocol


class IndexHit20LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="index-hit20-launch-cpu-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage = self.root / "gated_engram_index_hit20_b_20261104_v1/GatedEngram"
        self.native = self.stage.parent / "NativeSnapshot"
        self.results = self.root / "results"
        self.results.mkdir()
        for relative in (
            "gpu_pilot/run_pilot.py", "gpu_pilot/launch_pilot.py",
            "gpu_pilot/run_index_hit20_pilot.py", "gpu_pilot/read_index_hit20_results.py",
            "gpu_pilot/launch_index_hit20_pilot.py", "gpu_pilot/index_hit20_protocol.py",
            "gpu_pilot/index_hit20_empty_ep_qualification.py", "kv_cache_engram/index_hit20.py",
            "kv_cache_engram/index_hit20_adapter.py",
            "tests/test_launch_index_hit20.py",
        ):
            path = self.stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# CPU fixture, never executed\n")
        for relative in ("setup.py", "deepinfer/model_executor/models/v4_modules/attn.py", "README.md"):
            path = self.native / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen native CPU fixture\n")
        self.manifest = self.stage / "manifest.json"
        self.protocol = self.stage / "protocol.json"
        self.sensitivity = self.stage / "sensitivity-protocol.json"
        self.swa = self.stage / "index-hit20-protocol.json"
        self.write_sealed(self.protocol, dict(schema=launch.lifecycle.PROTOCOL_SCHEMA))
        self.sensitivity.write_bytes(launch.lifecycle.canonical(sensitivity_protocol()))
        self.swa.write_bytes(launch.lifecycle.canonical(index_hit20_protocol()))
        self.write_sealed(self.manifest, dict(schema="cpu-input-v1",
            artifact_kind="kv_cache_engram_native_pilot",
            pilot_protocol_sha256=launch.lifecycle.read_json(self.protocol)["sha256"],
            sensitivity_protocol_sha256=sensitivity_protocol()["sha256"]))
        self.predecessor = dict(name=launch.PREDECESSOR, status="complete_preserved")
        self.old = self.results / launch.PREDECESSOR
        self.old.mkdir()
        self.state = dict(schema="redknot-glm53-gpu-lifecycle-v1",
            status="runner_completed; consult benchmark quality/performance gates",
            worker_exit=0, holder_restored=True, gpu_ids=list(range(8)),
            finished_unix=123., child_pid=2147483600,
            supervisor_sha256=launch.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launch.lifecycle.HELPER_SHA,
            child_environment_sha256=launch.lifecycle.ENV_SHA)
        self.audit = launch.lifecycle.seal(dict(schema="swa-reuse-qonly-independent-readout-v1",
            status="complete", complete_cases=4, completed_case_indices=[2, 3, 4, 5],
            final_run_audited=True, overall_run_complete=True, all_registered_cases_audited=True))
        self.audit_patch = mock.patch.object(launch, "PREDECESSOR_AUDIT_SHA", self.audit["sha256"])
        self.audit_patch.start()
        self.addCleanup(self.audit_patch.stop)
        self.save_predecessor()

    @staticmethod
    def write_sealed(path, body):
        path.write_bytes(launch.lifecycle.canonical(launch.lifecycle.seal(body)))

    def save_predecessor(self, outcome_updates=None):
        (self.old / "supervisor.json").write_text(json.dumps(self.state))
        outcome = dict(schema="old-outcome-v1", successful=True, automatic_retry=False,
                       supervisor_exit=0, holder_physically_verified=True)
        outcome.update(outcome_updates or {})
        self.write_sealed(self.results / (launch.PREDECESSOR + ".launch-outcome.json"), outcome)
        (self.results / launch.PREDECESSOR_AUDIT).write_bytes(launch.lifecycle.canonical(self.audit))
        for index, suffix in enumerate(("launcher-started", "supervisor-started")):
            self.write_sealed(self.results / (launch.PREDECESSOR + "." + suffix + ".json"),
                              dict(schema="old-started-v1", pid=2147483601 + index, start_ticks=900 + index))

    def cpu_proof(self):
        names = ["tests.test_launch_index_hit20.test_fixed_command",
                 "tests.test_index_hit20_controller.test_cache_use",
                 "tests.test_index_hit20_pilot.test_complete_audit",
                 "tests.test_index_hit20_empty_ep_qualification.test_zero_rank"]
        names.extend("tests.test_fixture.test_%d" % n for n in range(1041))
        return dict(schema="index-hit20-cpu-proof-v1", successful=True, cuda_initialized=False,
                    failures=0, errors=0, skipped=0, tests=len(names), test_ids=names,
                    source_sha256=launch.lifecycle.source_files(self.stage))

    def plan(self):
        self.write_sealed(self.stage / "INDEX_HIT20_CPU_TESTS.json", self.cpu_proof())
        with mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor):
            return launch.build_plan(self.root, self.stage, self.manifest,
                                     self.protocol, self.swa, self.sensitivity)

    def verify(self, plan):
        with mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor):
            launch.verify_plan(plan)

    def test_fixed_finite_original_supervisor_and_private_native(self):
        plan = self.plan()
        self.verify(plan)
        argv = plan["argv"]
        self.assertEqual(plan["timeout"], 21600)
        self.assertEqual(plan["prepare_timeout"], 300)
        self.assertEqual(argv[:2], ["/root/miniconda3/bin/python", str(launch.lifecycle.SUPERVISOR)])
        self.assertEqual(argv[argv.index("--") + 1], str(self.stage / "gpu_pilot/run_index_hit20_pilot.py"))
        self.assertEqual(argv[argv.index("--repo") + 1], str(self.native))
        self.assertEqual(argv[argv.index("--gpus") + 1], "0,1,2,3,4,5,6,7")
        self.assertEqual(plan["evaluation_indices"], [2, 3, 4, 5])
        self.assertTrue(plan["live_native_workspace_is_not_execution_source"])
        self.assertTrue(plan["native_joint_attention_preserved"])
        self.assertTrue(plan["cached_document_swa_requires_raw_consumption_audit"])
        self.assertNotIn("zero_post6_swa_attention_is_ablation", plan)
        self.assertNotIn("effective_swa_attention_reuse_claim", plan)
        self.assertEqual(plan["predecessor"], "kce-swa-reuse-qonly-20261104-v1")
        self.assertEqual(plan["first_full_layers"], [0, 1, 2])
        self.assertEqual(plan["sparse_layers"], list(range(3, 40)))
        self.assertEqual(plan["optional_fraction"], .20)
        self.assertEqual(plan["empty_ep_layer_count"], 37)
        self.assertEqual(plan["empty_ep_rank_pattern_checks"], 1184)
        self.assertTrue(plan["global_kv_native_online"])
        self.assertTrue(plan["sparse_reference_byteexact_required"])
        self.assertNotIn(str(self.root / "DeepInfer-PR148"), argv)

    def test_check_only_does_not_launch_query_holder_or_publish(self):
        plan = self.plan()
        with mock.patch.object(launch, "build_plan", return_value=plan), \
             mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor), \
             mock.patch.object(launch, "run_plan", side_effect=AssertionError("no launch")) as run, \
             mock.patch.object(launch.lifecycle, "holder_snapshot", side_effect=AssertionError("no GPU")) as holder, \
             mock.patch.object(launch.subprocess, "Popen", side_effect=AssertionError("no process")) as popen, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.main(["--check-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["sha256"], plan["sha256"])
        run.assert_not_called(); holder.assert_not_called(); popen.assert_not_called()
        self.assertFalse(Path(plan["paths"]["plan"]).exists())

    def test_all_native_non_git_source_files_are_bound(self):
        sources = launch.snapshot_sources(self.native)
        self.assertIn("README.md", sources)
        self.assertIn("setup.py", sources)
        self.assertIn("deepinfer/model_executor/models/v4_modules/attn.py", sources)
        (self.native / ".git").mkdir()
        (self.native / ".git/config").write_text("fixture")
        self.assertEqual(sources, launch.snapshot_sources(self.native))

    def test_native_changed_added_or_removed_file_blocks(self):
        for kind in ("changed", "added", "removed"):
            with self.subTest(kind=kind):
                plan = self.plan()
                target = self.native / ("new.json" if kind == "added" else "README.md")
                before = target.read_bytes() if target.exists() else None
                if kind == "removed":
                    target.unlink()
                else:
                    target.write_text("changed native bytes")
                with self.assertRaisesRegex(ValueError, "native snapshot changed"):
                    self.verify(plan)
                if before is None:
                    target.unlink()
                else:
                    target.write_bytes(before)

    def test_native_leaf_or_parent_symlink_blocks(self):
        linked = self.native / "linked.py"
        linked.symlink_to(self.native / "setup.py")
        with self.assertRaises(ValueError):
            launch.snapshot_sources(self.native)
        linked.unlink()
        alias = self.root / "native-alias"
        alias.symlink_to(self.native, target_is_directory=True)
        with self.assertRaises(ValueError):
            launch.snapshot_sources(alias)

    def test_incomplete_native_snapshot_blocks(self):
        (self.native / "setup.py").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete native snapshot"):
            self.plan()

    def test_new_required_source_missing_blocks(self):
        (self.stage / "gpu_pilot/index_hit20_empty_ep_qualification.py").unlink()
        with self.assertRaisesRegex(ValueError, "modules absent"):
            self.plan()

    def test_source_mutation_blocks_without_gpu_calls(self):
        plan = self.plan()
        (self.stage / "kv_cache_engram/index_hit20.py").write_text("# changed\n")
        with mock.patch.object(launch.subprocess, "Popen") as popen, self.assertRaises(ValueError):
            launch.run_plan(plan)
        popen.assert_not_called()

    def test_existing_any_evidence_forbids_duplicate_launch(self):
        plan = self.plan()
        for name, raw in plan["paths"].items():
            path = Path(raw)
            path.write_text("preserve evidence")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "already used"):
                self.plan()
            self.assertEqual(path.read_text(), "preserve evidence")
            path.unlink()

    def test_protocol_changed_even_if_resealed_blocks(self):
        plan = self.plan()
        body = index_hit20_protocol()
        body.pop("sha256")
        body["optional_fraction"] = .25
        self.write_sealed(self.swa, body)
        with self.assertRaises(ValueError):
            self.verify(plan)

    def test_cpu_proof_flags_counts_and_required_test_families(self):
        plan = self.plan()
        mutations = [dict(successful=False), dict(cuda_initialized=True), dict(failures=1),
                     dict(errors=1), dict(skipped=1), dict(tests=877), dict(tests=True),
                     dict(source_sha256={}), dict(test_ids=["test"] * 878)]
        for family in ("test_launch_index_hit20.", "test_index_hit20_controller.",
                       "test_index_hit20_pilot.", "test_index_hit20_empty_ep_qualification."):
            proof = self.cpu_proof()
            proof["test_ids"] = [name.replace(family, "removed_family") for name in proof["test_ids"]]
            mutations.append(dict(test_ids=proof["test_ids"]))
        for update in mutations:
            proof = self.cpu_proof(); proof.update(update)
            self.write_sealed(self.stage / "INDEX_HIT20_CPU_TESTS.json", proof)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "CPU proof"):
                launch.verify_cpu_proof(plan)

    def test_plan_seal_and_command_runner_repo_protocol_are_checked(self):
        original = self.plan()
        value = copy.deepcopy(original); value["timeout"] = 999999
        with self.assertRaisesRegex(ValueError, "plan seal"):
            self.verify(value)
        for flag in ("--", "--repo", "--index-hit20-protocol"):
            value = copy.deepcopy(original)
            value["argv"][value["argv"].index(flag) + 1] = "/unapproved/source"
            value.pop("sha256"); value = launch.lifecycle.seal(value)
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "wrong runner/native/protocol"):
                self.verify(value)

    def test_predecessor_terminal_identity_exit_and_restore_required(self):
        with mock.patch.object(launch.lifecycle, "proc_identity", side_effect=FileNotFoundError()):
            evidence = launch.predecessor_evidence(self.root)
        self.assertEqual(evidence["status"], "complete_preserved")
        self.assertEqual(len(evidence["original_process_exits"]), 2)
        self.assertEqual(evidence["audit_sha256"], self.audit["sha256"])
        for update in (dict(status="running"), dict(worker_exit=1), dict(holder_restored=False),
                       dict(finished_unix=None), dict(cleanup_error="failure"),
                       dict(cleanup_warnings=["unresolved"]), dict(gpu_ids=list(range(7))),
                       dict(supervisor_sha256="bad"), dict(holder_helper_sha256="bad"),
                       dict(child_environment_sha256="bad")):
            saved = dict(self.state); self.state.update(update); self.save_predecessor()
            with self.subTest(update=update), self.assertRaises(ValueError):
                launch.predecessor_evidence(self.root)
            self.state = saved
        self.save_predecessor()

    def test_predecessor_failure_retry_or_unrestored_outcome_blocks(self):
        for update in (dict(successful=False), dict(automatic_retry=True), dict(supervisor_exit=1),
                       dict(holder_physically_verified=False)):
            self.save_predecessor(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                launch.predecessor_evidence(self.root)

    def test_predecessor_raw_audit_missing_or_changed_blocks(self):
        path = self.results / launch.PREDECESSOR_AUDIT
        for update in (dict(status="partial"), dict(complete_cases=3),
                       dict(completed_case_indices=[2, 3, 4]), dict(final_run_audited=False),
                       dict(overall_run_complete=False), dict(all_registered_cases_audited=False),
                       dict(errors=["bank mismatch"])):
            value = {key: val for key, val in self.audit.items() if key != "sha256"}
            value.update(update)
            self.write_sealed(path, value)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "final raw/cache audit"):
                launch.predecessor_evidence(self.root)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing or oversized JSON"):
            launch.predecessor_evidence(self.root)

    def test_alive_same_identity_blocks_without_signals(self):
        with mock.patch.object(launch.lifecycle, "proc_identity", return_value=dict(start_ticks=900)), \
             self.assertRaisesRegex(ValueError, "still alive"):
            launch.predecessor_evidence(self.root)

    def terminal(self, plan):
        return dict(schema="redknot-glm53-gpu-lifecycle-v1",
            status="runner_completed; consult benchmark quality/performance gates", holder_restored=True,
            worker_exit=0, gpu_ids=list(range(8)), finished_unix=321.,
            supervisor_sha256=launch.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launch.lifecycle.HELPER_SHA,
            child_environment_sha256=launch.lifecycle.ENV_SHA,
            runner_sha256=plan["stage_sources"]["gpu_pilot/run_index_hit20_pilot.py"])

    def test_terminal_rejects_error_warning_bad_owner_and_missing_gpu(self):
        plan = self.plan()
        launch.verify_terminal(self.terminal(plan), plan)
        for update in (dict(status="failed"), dict(status="running"),
                       dict(holder_restored=False), dict(worker_exit=1), dict(gpu_ids=list(range(7))),
                       dict(finished_unix=None), dict(error="failed"), dict(cleanup_error="failed"),
                       dict(cleanup_warnings=["cannot authenticate"]), dict(supervisor_sha256="bad"),
                       dict(holder_helper_sha256="bad"), dict(child_environment_sha256="bad"),
                       dict(runner_sha256="bad")):
            state = self.terminal(plan); state.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                launch.verify_terminal(state, plan)

    def run_fake(self, plan, *, exit_code=0, identity_error=False, holder_error=False,
                 final_indices=None, terminal_update=None, final_schema=None):
        events = []
        def wait():
            events.append("wait")
            result = Path(plan["paths"]["result_dir"])
            (result / "benchmark").mkdir(parents=True)
            state = self.terminal(plan); state.update(terminal_update or {})
            (result / "supervisor.json").write_text(json.dumps(state))
            self.write_sealed(result / "benchmark/run.json", dict(schema=launch.RUN_SCHEMA if final_schema is None else final_schema, status="complete",
                completed_case_indices=[2, 3, 4, 5] if final_indices is None else final_indices))
            return exit_code
        child = SimpleNamespace(pid=98765, wait=wait)
        def holder(_root):
            events.append("holder")
            if holder_error and events.count("holder") == 2:
                raise ValueError("physical eight-GPU audit failed")
            return launch.lifecycle.seal(dict(schema="cpu-holder-v1", read_only=True))
        with mock.patch.object(launch, "ROOT", self.root), mock.patch.object(launch, "STAGE", self.stage), \
             mock.patch.object(launch, "REMOTE_STAGE", self.stage), \
             mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor), \
             mock.patch.object(launch.lifecycle, "verify_lifecycle"), \
             mock.patch.object(launch.lifecycle, "holder_snapshot", side_effect=holder), \
             mock.patch.object(launch.lifecycle, "proc_identity",
                side_effect=ValueError("identity failed") if identity_error else None,
                return_value=dict(pid=child.pid, start_ticks=100, argv=plan["argv"])), \
             mock.patch.object(launch.subprocess, "Popen", return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            outcome = launch.run_plan(plan)
        return outcome, events, popen

    def test_success_waits_then_authenticates_all_holder_evidence(self):
        outcome, events, popen = self.run_fake(self.plan())
        self.assertTrue(outcome["successful"])
        self.assertTrue(outcome["holder_physically_verified"])
        self.assertTrue(outcome["independent_raw_audit_required"])
        self.assertFalse(outcome["automatic_retry"])
        self.assertEqual(events, ["holder", "wait", "holder"])
        self.assertEqual(popen.call_count, 1)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_post_popen_identity_failure_still_waits_before_holder_check(self):
        outcome, events, popen = self.run_fake(self.plan(), identity_error=True)
        self.assertFalse(outcome["successful"])
        self.assertTrue(outcome["holder_physically_verified"])
        self.assertEqual(events, ["holder", "wait", "holder"])
        self.assertEqual(popen.call_count, 1)

    def test_worker_failure_does_not_skip_holder_or_retry(self):
        outcome, events, popen = self.run_fake(self.plan(), exit_code=1)
        self.assertFalse(outcome["successful"])
        self.assertTrue(outcome["holder_physically_verified"])
        self.assertFalse(outcome["automatic_retry"])
        self.assertEqual(events, ["holder", "wait", "holder"])
        self.assertEqual(popen.call_count, 1)

    def test_physical_holder_failure_overrides_success_claim(self):
        outcome, events, _ = self.run_fake(self.plan(), holder_error=True)
        self.assertFalse(outcome["successful"])
        self.assertFalse(outcome["holder_physically_verified"])
        self.assertIn("holder_error", outcome)
        self.assertEqual(events, ["holder", "wait", "holder"])

    def test_partial_or_wrong_cases_cannot_claim_completed_run(self):
        outcome, _, _ = self.run_fake(self.plan(), final_indices=[2, 3, 4])
        self.assertFalse(outcome["successful"])
        self.assertIn("four registered cases", outcome["error"])

    def test_cleanup_warning_kept_as_failed_even_with_worker_zero(self):
        outcome, _, _ = self.run_fake(self.plan(), terminal_update=dict(cleanup_warnings=["unknown process"]))
        self.assertFalse(outcome["successful"])
        self.assertTrue(outcome["holder_physically_verified"])

    def test_foreign_workflow_run_cannot_claim_success(self):
        outcome, _, _ = self.run_fake(self.plan(), final_schema="swa-skip-run-v1")
        self.assertFalse(outcome["successful"])
        self.assertTrue(outcome["holder_physically_verified"])

    def test_resealed_foreign_attention_plan_is_rejected(self):
        for field in ("native_joint_attention_preserved", "cached_document_swa_requires_raw_consumption_audit",
                      "global_kv_native_online", "sparse_reference_byteexact_required"):
            plan = self.plan()
            plan.pop("sha256")
            plan[field] = False
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "wrong bounded launch"):
                self.verify(launch.lifecycle.seal(plan))

    def test_resealed_algorithm_budget_layer_or_empty_ep_mutation_rejected(self):
        for field, value in (("optional_fraction", .15), ("first_full_layers", list(range(6))),
                             ("sparse_layers", list(range(6, 40))), ("empty_ep_layer_count", 34),
                             ("empty_ep_rank_pattern_checks", 1088)):
            plan = self.plan()
            plan.pop("sha256")
            plan[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "wrong bounded launch"):
                self.verify(launch.lifecycle.seal(plan))

    def test_actual_local_launch_rejected_before_holder_query(self):
        plan = self.plan()
        with mock.patch.object(launch, "predecessor_evidence", return_value=self.predecessor), \
             mock.patch.object(launch.lifecycle, "holder_snapshot") as holder, \
             mock.patch.object(launch.subprocess, "Popen") as popen, \
             self.assertRaisesRegex(ValueError, "authorized independent remote stage"):
            launch.run_plan(plan)
        holder.assert_not_called(); popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

