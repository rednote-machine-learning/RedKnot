"""CPU-only orchestration fixtures; no real subprocess or GPU is launched."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import launch_pilot as old
from gpu_pilot import launch_repair_pilot as launch
from gpu_pilot import pilot_protocol as pp
from gpu_pilot.repair_protocol import repair_protocol
from gpu_pilot.sensitivity_protocol import sensitivity_protocol


class RepairLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "results").mkdir()
        self.old_stage = self.root / launch.PREDECESSOR_STAGE
        self.stage = self.root / "gated_engram_repair_20261102_v3/GatedEngram"
        for stage in (self.old_stage, self.stage):
            for name in ("gpu_pilot/run_pilot.py", "gpu_pilot/launch_pilot.py", "kv_cache_engram/core.py",
                         "gpu_pilot/read_results.py"):
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# CPU fixture " + name + "\n")
            inputs = stage / "inputs-sensitivity-v1"
            inputs.mkdir()
            self.write(inputs / "protocol.json", pp.pilot_protocol())
            self.write(inputs / "sensitivity-protocol.json", sensitivity_protocol())
            self.write(inputs / "manifest.json", old.seal(dict(schema="cpu-fixture-manifest-v1",
                artifact_kind="kv_cache_engram_native_pilot",
                pilot_protocol_sha256=pp.pilot_protocol()["sha256"],
                sensitivity_protocol_sha256=sensitivity_protocol()["sha256"])))
        for name in ("gpu_pilot/run_repair_pilot.py", "gpu_pilot/launch_repair_pilot.py"):
            (self.stage / name).write_text("# new CPU-only fixture\n")
        self.repair_path = self.stage / "repair_protocol.json"
        self.write(self.repair_path, repair_protocol())
        self.reader_patch = mock.patch.object(launch, "READER_SHA", old.file_digest(self.old_stage / "gpu_pilot/read_results.py"))
        self.reader_patch.start(); self.addCleanup(self.reader_patch.stop)
        self.old_inputs = self.old_stage / "inputs-sensitivity-v1"
        self.inputs = self.stage / "inputs-sensitivity-v1"
        self.old_plan = old.build_plan(self.root, self.old_stage, launch.WAIT_FOR,
            self.old_inputs / "manifest.json", self.old_inputs / "protocol.json",
            sensitivity_protocol=self.old_inputs / "sensitivity-protocol.json")
        self.old_paths = old.evidence_paths(self.root, launch.WAIT_FOR)
        self.write(self.old_paths["plan"], self.old_plan)
        self.launcher_process = dict(pid=301, start_ticks=5001,
            argv=["python", str(self.old_stage / "gpu_pilot/launch_pilot.py"), "--name", launch.WAIT_FOR])
        self.write(self.root / "results" / (launch.WAIT_FOR + ".launcher-started.json"), old.seal(dict(
            schema="kv-cache-engram-sensitivity-launcher-started-v1", process=self.launcher_process,
            argv=self.launcher_process["argv"], source_sha256=self.old_plan["stage_sources_sha256"])))
        self.write(self.old_paths["started"], old.seal(dict(
            schema="kv-cache-engram-pilot-supervisor-started-v1", pid=302,
            plan_sha256=self.old_plan["sha256"], argv=self.old_plan["argv"])))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True))

    def plan(self, **changes):
        args = dict(root=self.root, stage_dir=self.stage, name=launch.NAME,
            manifest=self.inputs / "manifest.json", protocol=self.inputs / "protocol.json",
            sensitivity_protocol=self.inputs / "sensitivity-protocol.json",
            repair_protocol_path=self.repair_path, wait_for=launch.WAIT_FOR)
        args.update(changes)
        return launch.build_plan(**args)

    def supervisor(self, plan=None):
        plan = plan or self.old_plan
        runner = launch.RUNNER if plan.get("schema") == launch.PLAN_SCHEMA else "gpu_pilot/run_pilot.py"
        return dict(schema="redknot-glm53-gpu-lifecycle-v1", child_pid=303,
            holder_restored=True, worker_exit=0, finished_unix=12345,
            gpu_ids=list(range(8)), supervisor_sha256=old.SUPERVISOR_SHA,
            holder_helper_sha256=old.HELPER_SHA, child_environment_sha256=old.ENV_SHA,
            runner_sha256=plan["stage_sources"][runner])

    def holder(self):
        return old.seal(dict(schema="kv-cache-engram-pilot-holder-audit-v1", fixture="CPU only"))

    def finish_old(self):
        self.old_paths["result_dir"].mkdir(exist_ok=True)
        self.write(self.old_paths["result_dir"] / "supervisor.json", self.supervisor())
        after = self.holder()
        self.write(self.old_paths["after"], after)
        self.write(self.old_paths["outcome"], old.seal(dict(
            schema="kv-cache-engram-pilot-launch-outcome-v1", successful=True,
            plan_sha256=self.old_plan["sha256"], supervisor_pid=302, supervisor_exit=0,
            supervisor_started=True, holder_physically_verified=True, holder_audit_sha256=after["sha256"])))

    def raw_audit(self):
        return old.seal(dict(schema="kv-cache-engram-native-pilot-readout-v1", status="complete",
            completed=4, planned=4, sensitivity_enabled=True, failures=[], audit_errors=[], pending_case_indices=[],
            cases=[dict(index=i) for i in (2, 3, 4, 5)],
            by_dataset={dataset: dict(completed=2, planned=2) for dataset in ("hotpotqa", "2wikimqa")}))

    def admission(self, plan):
        with mock.patch.object(launch, "process_state", return_value=None):
            terminal = launch.predecessor_terminal(plan)
        return old.seal(dict(schema="kv-cache-engram-repair-predecessor-admission-v1",
            plan_sha256=plan["sha256"], physically_verified=True, terminal=terminal,
            raw_audit=self.raw_audit(), current_holder=self.holder()))

    def test_plan_has_new_runner_frozen_inputs_and_wait_identity(self):
        plan = self.plan()
        self.assertEqual(plan["schema"], launch.PLAN_SCHEMA)
        self.assertEqual(plan["predecessor"]["reader_sha256"], launch.READER_SHA)
        self.assertEqual(plan["predecessor"]["plan_sha256"], self.old_plan["sha256"])
        self.assertIn(str(self.stage / launch.RUNNER), plan["argv"])
        self.assertNotIn(str(self.stage / "gpu_pilot/run_pilot.py"), plan["argv"])
        self.assertEqual(plan["argv"][-2:], ["--repair-protocol", str(self.repair_path)])
        self.assertEqual(plan["poll_seconds"], 30)
        self.assertEqual(plan["wait_timeout"], 21600)
        for key in ("manifest", "protocol", "sensitivity_protocol"):
            self.assertEqual(plan[key]["file_sha256"], self.old_plan[key]["file_sha256"])
        launch.verify_plan(plan)

    def test_only_named_finite_queue_and_distinct_stage_are_allowed(self):
        changes = (dict(name="kce-other"), dict(wait_for="sparse-query20-rag50-v8"),
                   dict(wait_timeout=0), dict(wait_timeout=21601), dict(wait_timeout=True),
                   dict(timeout=21601), dict(stage_dir=self.old_stage))
        for kwargs in changes:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)

    def test_input_bytes_and_repair_protocol_rules_cannot_drift(self):
        path = self.inputs / "manifest.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "exact already-exposed"):
            self.plan()
        path.write_bytes((self.old_inputs / "manifest.json").read_bytes())
        value = repair_protocol(); value.pop("sha256"); value["no_ttft_claim"] = False
        self.write(self.repair_path, old.seal(value))
        with self.assertRaisesRegex(ValueError, "CPU-tested"):
            self.plan()

    def test_new_or_predecessor_stage_drift_invalidates_frozen_plan(self):
        plan = self.plan()
        (self.stage / launch.RUNNER).write_text("# changed new code\n")
        with self.assertRaisesRegex(ValueError, "repair stage changed"):
            launch.verify_plan(plan)
        (self.stage / launch.RUNNER).write_text("# new CPU-only fixture\n")
        (self.old_stage / "gpu_pilot/read_results.py").write_text("# changed old reader\n")
        with self.assertRaises(ValueError):
            launch.verify_plan(plan)

    def test_check_only_never_queries_gpu_or_launches_or_claims_queue_name(self):
        with mock.patch.object(old, "holder_snapshot", side_effect=AssertionError("no GPU")) as holder, \
             mock.patch.object(launch, "run_queue", side_effect=AssertionError("no queue")) as queue, \
             mock.patch.object(launch.subprocess, "Popen", side_effect=AssertionError("no launch")) as popen, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            result = launch.main(["--root", str(self.root), "--stage-dir", str(self.stage),
                "--manifest", str(self.inputs / "manifest.json"),
                "--pilot-protocol", str(self.inputs / "protocol.json"),
                "--sensitivity-protocol", str(self.inputs / "sensitivity-protocol.json"),
                "--repair-protocol", str(self.repair_path), "--wait-for", launch.WAIT_FOR, "--check-only"])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output.getvalue())["gpu_initialized"])
        holder.assert_not_called(); queue.assert_not_called(); popen.assert_not_called()
        self.assertFalse(any(path.exists() for path in launch.paths_for(self.root, launch.NAME).values()))

    def test_pending_predecessor_does_not_query_holder_or_run_reader(self):
        plan = self.plan()
        now, sleeps = [0.], []
        def sleep(seconds):
            sleeps.append(seconds); now[0] += seconds
        value = dict(plan, wait_timeout=60)
        value.pop("sha256"); value = old.seal(value)
        with mock.patch.object(old, "holder_snapshot", side_effect=AssertionError("no GPU")) as holder, \
             mock.patch.object(launch, "process_state", return_value=dict(self.launcher_process, state="S")), \
             mock.patch.object(launch, "audit_predecessor", side_effect=AssertionError("not ready")) as audit, \
             self.assertRaisesRegex(ValueError, "timed out"):
            launch.wait_until_admitted(value, clock=lambda: now[0], sleep=sleep)
        self.assertEqual(sleeps, [30, 30])
        holder.assert_not_called(); audit.assert_not_called()

    def test_lost_launcher_without_terminal_outcome_fails_closed(self):
        plan = self.plan()
        with mock.patch.object(launch, "process_state", return_value=None), \
             self.assertRaisesRegex(ValueError, "exited without a terminal outcome"):
            launch.predecessor_terminal(plan)

    def test_predecessor_failure_is_not_polled_or_retried(self):
        plan = self.plan(); self.finish_old()
        outcome = old.read_json(self.old_paths["outcome"]); outcome.pop("sha256")
        outcome["successful"] = False
        self.write(self.old_paths["outcome"], old.seal(outcome))
        with self.assertRaisesRegex(ValueError, "launcher failed"):
            launch.predecessor_terminal(plan)
        self.old_paths["outcome"].unlink()
        failed = self.supervisor(); failed["worker_exit"] = 1
        self.write(self.old_paths["result_dir"] / "supervisor.json", failed)
        with self.assertRaisesRegex(ValueError, "worker failed"):
            launch.predecessor_terminal(plan)

    def test_successful_outcome_still_waits_for_launcher_supervisor_and_child_exit(self):
        plan = self.plan(); self.finish_old()
        states = {
            301: dict(self.launcher_process, state="S"),
            302: dict(pid=302, state="S", start_ticks=5002, argv=self.old_plan["argv"]),
            303: dict(pid=303, state="S", start_ticks=5003, argv=["python", "run_pilot.py"]),
        }
        for alive in (301, 302, 303):
            with self.subTest(alive=alive), mock.patch.object(launch, "process_state",
                    side_effect=lambda pid: states[pid] if pid == alive else None):
                self.assertIsNone(launch.predecessor_terminal(plan))
        with mock.patch.object(launch, "process_state", return_value=None):
            self.assertTrue(launch.predecessor_terminal(plan)["processes_exited"])

    def test_launcher_success_cannot_override_failed_supervisor_restoration(self):
        plan = self.plan(); self.finish_old()
        status = self.supervisor(); status["holder_restored"] = False
        self.write(self.old_paths["result_dir"] / "supervisor.json", status)
        with self.assertRaisesRegex(ValueError, "holder restoration"):
            launch.predecessor_terminal(plan)

    def test_pid_reuse_means_original_exited_but_is_never_signalled(self):
        state = dict(self.launcher_process, state="S", start_ticks=6000, argv=["unrelated CPU process"])
        with mock.patch.object(launch, "process_state", return_value=state):
            self.assertTrue(launch._exited(301, expected=self.launcher_process))
        state = dict(self.launcher_process, state="Z")
        with mock.patch.object(launch, "process_state", return_value=state):
            self.assertTrue(launch._exited(301, expected=self.launcher_process))

    def test_live_launcher_argv_change_fails_closed_without_signals(self):
        plan = self.plan(); self.finish_old()
        state = dict(self.launcher_process, state="S", argv=["foreign", "process"])
        with mock.patch.object(launch, "process_state", return_value=state), \
             self.assertRaisesRegex(ValueError, "identity changed"):
            launch.predecessor_terminal(plan)

    def test_raw_audit_subprocess_uses_pinned_old_reader_and_complete_coverage(self):
        plan = self.plan()
        result = SimpleNamespace(returncode=0, stdout=json.dumps(self.raw_audit()), stderr="")
        with mock.patch.object(launch.subprocess, "run", return_value=result) as run:
            audit = launch.audit_predecessor(plan)
        self.assertEqual(audit["completed"], 4)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [str(old.RUNTIME / "di-venv/bin/python"), "-B", "-s"])
        self.assertEqual(command[3], str(self.old_stage / "gpu_pilot/read_results.py"))
        self.assertEqual(command[4], str(self.old_paths["result_dir"] / "benchmark"))
        self.assertEqual(run.call_args.kwargs["timeout"], 600)

    def test_raw_audit_rejects_partial_bad_seal_errors_stderr_and_nonzero_exit(self):
        plan = self.plan()
        partial = self.raw_audit(); partial.pop("sha256"); partial["completed"] = 3
        invalid = self.raw_audit(); invalid["completed"] = 3
        results = (
            SimpleNamespace(returncode=1, stdout="", stderr="failure"),
            SimpleNamespace(returncode=0, stdout=json.dumps(self.raw_audit()), stderr="unexpected warning"),
            SimpleNamespace(returncode=0, stdout=json.dumps(old.seal(partial)), stderr=""),
            SimpleNamespace(returncode=0, stdout=json.dumps(invalid), stderr=""),
        )
        for result in results:
            with self.subTest(result=result), mock.patch.object(launch.subprocess, "run", return_value=result), \
                 self.assertRaises(ValueError):
                launch.audit_predecessor(plan)

    def test_physical_holder_checked_only_after_raw_and_terminal_audits(self):
        plan = self.plan(); self.finish_old()
        calls = []
        with mock.patch.object(launch, "process_state", return_value=None), \
             mock.patch.object(launch, "audit_predecessor", side_effect=lambda *a, **k: calls.append("raw") or self.raw_audit()), \
             mock.patch.object(old, "holder_snapshot", side_effect=lambda root: calls.append("physical") or self.holder()):
            value = launch.wait_until_admitted(plan, clock=lambda: 0, sleep=lambda _: self.fail("already ready"))
        self.assertEqual(calls, ["raw", "physical"])
        self.assertTrue(value["physically_verified"])
        self.assertEqual(value["plan_sha256"], plan["sha256"])

    def test_physical_holder_failure_prevents_admission(self):
        plan = self.plan(); self.finish_old()
        with mock.patch.object(launch, "process_state", return_value=None), \
             mock.patch.object(launch, "audit_predecessor", return_value=self.raw_audit()), \
             mock.patch.object(old, "holder_snapshot", side_effect=ValueError("foreign GPU context")), \
             self.assertRaisesRegex(ValueError, "foreign GPU"):
            launch.wait_until_admitted(plan, clock=lambda: 0)

    def test_raw_audit_timeout_is_capped_by_remaining_cpu_wait_budget(self):
        self.finish_old()
        for wait_timeout, expected_audit_timeout in ((21600, 600), (75, 75)):
            plan = self.plan(wait_timeout=wait_timeout)
            with self.subTest(wait_timeout=wait_timeout), \
                 mock.patch.object(launch, "process_state", return_value=None), \
                 mock.patch.object(launch, "audit_predecessor", return_value=self.raw_audit()) as audit, \
                 mock.patch.object(old, "holder_snapshot", return_value=self.holder()):
                launch.wait_until_admitted(plan, clock=lambda: 0)
            audit.assert_called_once_with(plan, timeout=expected_audit_timeout)

    def test_queue_claim_is_exclusive_and_failure_is_published_without_launch(self):
        plan = self.plan()
        own = dict(pid=999, start_ticks=888, argv=["python", "launch_repair_pilot.py"])
        with mock.patch.object(old, "proc_identity", return_value=own), \
             mock.patch.object(launch, "wait_until_admitted", side_effect=ValueError("old failed")), \
             mock.patch.object(launch, "run_supervisor", side_effect=AssertionError("must not launch")) as run:
            self.assertEqual(launch.run_queue(plan), 1)
            with self.assertRaisesRegex(ValueError, "already used"):
                launch.run_queue(plan)
        paths = launch.paths_for(self.root, launch.NAME)
        intent = old.read_sealed(paths["intent"])
        started = old.read_sealed(paths["queue_started"])
        outcome = old.read_sealed(paths["queue_outcome"])
        self.assertEqual(intent["process"], own)
        self.assertEqual(started["process"], own)
        self.assertEqual(started["intent_sha256"], intent["sha256"])
        self.assertFalse(outcome["successful"])
        self.assertFalse(outcome["gpu_supervisor_invoked"])
        run.assert_not_called()

    def test_admitted_queue_invokes_supervisor_once_and_saves_raw_gate(self):
        plan = self.plan(); self.finish_old(); admission = self.admission(plan)
        with mock.patch.object(old, "proc_identity", return_value=dict(pid=999, start_ticks=888)), \
             mock.patch.object(launch, "wait_until_admitted", return_value=admission), \
             mock.patch.object(launch, "run_supervisor", return_value=0) as run:
            self.assertEqual(launch.run_queue(plan), 0)
        run.assert_called_once_with(plan, admission)
        paths = launch.paths_for(self.root, launch.NAME)
        self.assertEqual(old.read_sealed(paths["predecessor_audit"]), admission)
        self.assertTrue(old.read_sealed(paths["queue_outcome"])["successful"])

    def test_actual_supervisor_path_still_requires_after_holder_and_new_runner_sha(self):
        plan = self.plan(); self.finish_old(); admission = self.admission(plan)
        paths = launch.paths_for(self.root, launch.NAME)
        def wait():
            paths["result_dir"].mkdir()
            self.write(paths["result_dir"] / "supervisor.json", self.supervisor(plan))
            return 0
        child = SimpleNamespace(pid=901, wait=mock.Mock(side_effect=wait))
        with mock.patch.object(old, "verify_lifecycle"), \
             mock.patch.object(launch, "process_state", return_value=None), \
             mock.patch.object(old, "holder_snapshot", side_effect=[self.holder(), ValueError("not restored")]), \
             mock.patch.object(launch.subprocess, "Popen", return_value=child) as popen:
            self.assertEqual(launch.run_supervisor(plan, admission), 1)
        popen.assert_called_once()
        child.wait.assert_called_once()
        outcome = old.read_sealed(paths["outcome"])
        self.assertFalse(outcome["successful"])
        self.assertFalse(outcome["holder_physically_verified"])
        self.assertIn("not restored", outcome["holder_verification_error"])

    def test_tampered_admission_cannot_invoke_original_supervisor(self):
        plan = self.plan(); self.finish_old(); admission = self.admission(plan)
        admission["tampered"] = True
        with mock.patch.object(old, "verify_lifecycle"), \
             mock.patch.object(launch, "process_state", return_value=None), \
             mock.patch.object(launch.subprocess, "Popen", side_effect=AssertionError("no launch")) as process, \
             self.assertRaisesRegex(ValueError, "no longer admitted"):
            launch.run_supervisor(plan, admission)
        process.assert_not_called()

    def test_original_runner_hash_cannot_stand_in_for_new_repair_runner(self):
        plan = self.plan()
        status = self.supervisor(plan)
        status["runner_sha256"] = self.old_plan["stage_sources"]["gpu_pilot/run_pilot.py"]
        with self.assertRaisesRegex(ValueError, "source or environment"):
            launch._verify_repair_supervisor(status, plan)


if __name__ == "__main__":
    unittest.main()
