"""One isolated three-hour FFN-only pilot under the original holder supervisor.

Check-only does not query GPUs. No signals, automatic retry, queue resurrection,
or holder replacement are implemented here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
import sys
import time

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from gpu_pilot import launch_pilot as lifecycle

ROOT = lifecycle.REMOTE_ROOT
REMOTE_STAGE = ROOT/"gated_engram_ffn_only_20261104_v3/GatedEngram"
NAME = "kce-ffn-only-20261104-v3"
PREDECESSOR = "kce-early-mhc-20261102-v2"
FAILED_PREDECESSOR = "kce-ffn-only-20261104-v2"
FAILED_AUDIT_SHA = "66488fa51296ef6b0be00923b3d41b8e5cc3b7259e2b64eeb6bf15b77a5969d8"
FAILED_OUTCOME_SHA = "bf1719e62f6a97e7ca1441984e62980ba830bc202c554853d68a90a9da73fbea"
FAILED_RAW_SHA = "f0af332acb1de62cf9f188a387c4a6110979f0fff77201dddb1ae639264e1a04"
FAILED_HOLDER_SHA = "4c70c5a5ca85d3aed616a615aaa03a02f8d56ae731ec01fd6946ff00d5d69e8a"
FAILED_STAGE_SHA = "729ebaebe42e543589a2108391e91279806842be6dde535dbab583eadd0064f3"
FAILED_ERROR = "ValueError('empty local FFN selection requires native EP qualification; collective cannot be skipped')"
FAILED_PROCESS_IDS = (3803437, 3803505, 3803509, 3803613, 3803614, 3803615,
                      3803616, 3803617, 3803622, 3803623, 3803624, 3803765)
FAILED_KNOWN_START_TICKS = {3803437: 2883926072, 3803505: 2883926672}
WORKFLOW_SCHEMA = "kv-cache-engram-ffn-only-launch-v1"
require = lifecycle.require


def build_plan(root, stage, manifest, pilot_protocol, early_protocol, sensitivity_protocol=None):
    from gpu_pilot.ffn_only_protocol import ffn_only_protocol, DEPENDENCIES
    root, stage = lifecycle.absolute_no_symlink(root), lifecycle.absolute_no_symlink(stage)
    early_protocol = lifecycle.absolute_no_symlink(early_protocol)
    require(stage in early_protocol.parents and early_protocol not in (Path(manifest), Path(pilot_protocol), sensitivity_protocol),
            "early protocol must be a distinct file inside independent stage")
    declared = lifecycle.read_sealed(early_protocol)
    require(declared == ffn_only_protocol(), "early protocol is not the CPU-tested fixed protocol")
    old = lifecycle.build_plan(root, stage, NAME, manifest, pilot_protocol,
                               timeout=10800, prepare_timeout=300, sensitivity_protocol=sensitivity_protocol)
    body = {k: v for k, v in old.items() if k != "sha256"}
    required = {"gpu_pilot/run_ffn_only_pilot.py", "gpu_pilot/launch_ffn_only_pilot.py",
                "gpu_pilot/ffn_only_protocol.py", "gpu_pilot/empty_ep_qualification.py",
                "kv_cache_engram/ffn_only.py", "tests/test_ffn_only_pilot.py",
                "tests/test_empty_ep_qualification.py"}
    require(required <= body["stage_sources"].keys(), "new early-mHC modules/tests absent")
    argv = list(body["argv"])
    separator = argv.index("--")
    argv[separator+1] = str(stage/"gpu_pilot/run_ffn_only_pilot.py")
    argv.extend(["--ffn-only-protocol", str(early_protocol)])
    body.update(workflow_schema=WORKFLOW_SCHEMA, argv=argv,
        ffn_only_protocol=dict(path=str(early_protocol), sha256=declared["sha256"],
                                file_sha256=lifecycle.file_digest(early_protocol)),
        predecessor=PREDECESSOR, failed_predecessor=FAILED_PREDECESSOR,
        failed_predecessor_audit_sha256=FAILED_AUDIT_SHA,
        evaluation_indices=[2, 3, 4, 5], qualification_input_tokens=16384,
        external_sources={str(root/name): lifecycle.file_digest(root/name) for name in DEPENDENCIES},
        cpu_proof_path=str(stage/"FFN_ONLY_CPU_TESTS.json"))
    require(body["external_sources"] == {str(root/name): sha for name, sha in DEPENDENCIES.items()},
            "original sparse operator dependency changed")
    return lifecycle.seal(body)


def verify_cpu_proof(plan):
    proof = lifecycle.read_sealed(plan["cpu_proof_path"])
    names = proof.get("test_ids", [])
    require(proof.get("successful") is True and proof.get("cuda_initialized") is False
            and proof.get("failures") == 0 and proof.get("errors") == 0 and proof.get("skipped") == 0
            and proof.get("source_sha256") == plan["stage_sources"]
            and type(proof.get("tests")) is int and proof["tests"] >= 10
            and len(names) == len(set(names)) == proof["tests"]
            and sum("test_ffn_only_pilot." in name for name in names) >= 6
            and any("test_empty_ep_qualification." in name for name in names),
            "new early-mHC CPU validation missing, stale, incomplete or skipped")


def verify_plan(plan):
    from gpu_pilot.ffn_only_protocol import DEPENDENCIES
    lifecycle.verify_plan_inputs(plan)
    require(plan.get("workflow_schema") == WORKFLOW_SCHEMA and plan.get("name") == NAME
            and plan.get("timeout") == 10800 and plan.get("prepare_timeout") == 300
            and plan.get("predecessor") == PREDECESSOR
            and plan.get("failed_predecessor") == FAILED_PREDECESSOR
            and plan.get("failed_predecessor_audit_sha256") == FAILED_AUDIT_SHA
            and plan.get("evaluation_indices") == [2, 3, 4, 5], "wrong bounded early-mHC launch plan")
    item = plan["ffn_only_protocol"]
    require(lifecycle.file_digest(item["path"]) == item["file_sha256"]
            and lifecycle.read_sealed(item["path"])["sha256"] == item["sha256"], "early protocol changed")
    argv = plan["argv"]
    require(argv[argv.index("--")+1] == str(Path(plan["stage_dir"])/"gpu_pilot/run_ffn_only_pilot.py")
            and argv[-2:] == ["--ffn-only-protocol", item["path"]], "wrong pilot runner/protocol command")
    expected = {str(Path(plan["root"])/name): sha for name, sha in DEPENDENCIES.items()}
    require(plan.get("external_sources") == expected
            and all(lifecycle.file_digest(path) == sha for path, sha in expected.items()),
            "original sparse operator dependency changed")
    verify_cpu_proof(plan)


def verify_failed_predecessor(root):
    """Require the exact failed v2 handoff; this is not an automatic retry.

    The predecessor passed its original qualification but did not pass the
    full-length smoke. Those results authorize neither v3 numerical execution
    nor an empty native EP call; v3 must establish its own qualifications.
    """
    root = Path(root)
    results = root/"results"
    audit = lifecycle.read_sealed(results/(FAILED_PREDECESSOR+".terminal-audit-20261104-1210.json"))
    require(audit.get("sha256") == FAILED_AUDIT_SHA
            and audit.get("schema") == "kce-ffn-only-terminal-smoke-failure-audit-v1"
            and audit.get("name") == FAILED_PREDECESSOR
            and audit.get("status") == "failed_during_partial_smoke"
            and audit.get("quality_completed_indices") == []
            and audit.get("partial_8x32k_smoke_passed") is False
            and audit.get("cuda_mask_gpu_qualification_passed") is True
            and audit.get("strict_16k_all_selected_gpu_qualification_passed") is True
            and audit.get("stage_sources_sha256") == FAILED_STAGE_SHA
            and audit.get("error") == FAILED_ERROR
            and audit.get("holder_physically_verified") is True
            and audit.get("automatic_retry") is False,
            "exact failed FFN-v2 terminal audit required")
    outcome_path = results/(FAILED_PREDECESSOR+".launch-outcome.json")
    raw_path = results/FAILED_PREDECESSOR/"benchmark/failure.rank0.json"
    outcome, raw = lifecycle.read_sealed(outcome_path), lifecycle.read_sealed(raw_path)
    require(outcome.get("sha256") == FAILED_OUTCOME_SHA
            and outcome.get("successful") is False and outcome.get("automatic_retry") is False
            and outcome.get("supervisor_started") is True and outcome.get("supervisor_exit") == 1
            and outcome.get("holder_physically_verified") is True
            and raw.get("sha256") == FAILED_RAW_SHA
            and raw.get("status") == "failed" and raw.get("completed_indices") == []
            and raw.get("error") == FAILED_ERROR
            and audit.get("source_seals", {}).get(outcome_path.name) == outcome["sha256"]
            and audit.get("source_seals", {}).get(raw_path.name) == raw["sha256"],
            "failed FFN-v2 raw/outcome binding differs")
    state_path = results/FAILED_PREDECESSOR/"supervisor.json"
    state = lifecycle.read_json(state_path)
    require(lifecycle.file_digest(state_path) == audit.get("supervisor_file_sha256")
            and state.get("status") == "failed"
            and state.get("holder_restored") is True and state.get("worker_exit") == 1
            and state.get("finished_unix") and not state.get("cleanup_error")
            and state.get("cleanup_warnings") == audit.get("supervisor_cleanup_warnings")
            and state.get("supervisor_sha256") == lifecycle.SUPERVISOR_SHA
            and state.get("holder_helper_sha256") == lifecycle.HELPER_SHA
            and state.get("child_environment_sha256") == lifecycle.ENV_SHA,
            "failed FFN-v2 cleanup state changed or incomplete")
    holder_path = results/(FAILED_PREDECESSOR+".terminal-holder-audit-20261104-1208.json")
    holder = lifecycle.read_sealed(holder_path)
    require(audit.get("source_seals", {}).get(holder_path.name) == FAILED_HOLDER_SHA
            and audit.get("holder_audit_sha256") == FAILED_HOLDER_SHA
            and holder.get("sha256") == FAILED_HOLDER_SHA,
            "failed FFN-v2 physical holder restoration evidence differs")
    lifecycle.validate_physical_holder(holder["metadata"], holder["gpus"], holder["contexts"])
    # The original supervisor correctly refused to signal an unauthenticated
    # cleanup member. Its recorded absence is preserved, not used to signal a
    # historical PID. run_plan also requires a fresh eight-GPU holder snapshot.
    require(audit.get("all_original_process_ids_absent") == list(FAILED_PROCESS_IDS),
            "failed FFN-v2 process exits and cleanup warning absence were not audited")
    for pid in FAILED_PROCESS_IDS:
        try:
            current = lifecycle.proc_identity(pid)
        except FileNotFoundError:
            continue
        # Compare identities where the original start time is recorded. For
        # other former worker/child IDs, fail closed on present PIDs rather than
        # guessing ownership or issuing signals. No process is ever stopped.
        require(pid in FAILED_KNOWN_START_TICKS
                and current["start_ticks"] != FAILED_KNOWN_START_TICKS[pid],
                "failed FFN-v2 process still alive or former PID ownership unqualified")


def verify_predecessor(root):
    """Both stopped and failed predecessors stay terminal; neither is resumed."""
    root = Path(root)
    verify_failed_predecessor(root)
    paused = lifecycle.read_sealed(root/"results"/(PREDECESSOR+".user-pause-final-20261102.json"))
    require(paused.get("sha256") == "33e20e7b6b1ef85f3991c367200edeb25acbf4e306300118fc9cc924f856a54e"
            and paused.get("user_requested_stop") is True
            and paused.get("status") == "stopped_by_user"
            and paused.get("holder_physically_verified") is True
            and paused.get("automatic_retry") is False,
            "explicit old-version stop and physical restoration evidence required")
    expected = {"launcher_identity": (3578467,2869924333),
                "supervisor_identity": (3578484,2869924944),
                "child_identity": (3578487,2869924987)}
    for label, (pid, start_ticks) in expected.items():
        require(paused["old_process_exit_checks"][label].get("exit_verified") is True,
                "predecessor exit was not audited")
        try:
            current = lifecycle.proc_identity(pid)
        except FileNotFoundError:
            continue
        require(current["start_ticks"] != start_ticks, "predecessor process still alive")
    state = lifecycle.read_json(root/"results"/PREDECESSOR/"supervisor.json")
    require(state.get("holder_restored") is True and state.get("finished_unix")
            and state.get("error") == "KeyboardInterrupt('received signal 15')"
            and not state.get("cleanup_error")
            and state.get("supervisor_sha256") == lifecycle.SUPERVISOR_SHA
            and state.get("holder_helper_sha256") == lifecycle.HELPER_SHA
            and state.get("child_environment_sha256") == lifecycle.ENV_SHA,
            "predecessor stop/cleanup identity mismatch")
    # Preserve the old warning, but do not signal its PID. The exact sealed
    # post-stop audit records that it is gone; the fresh holder audit in
    # run_plan independently rejects any foreign GPU context.
    require(state.get("cleanup_warnings") == paused.get("cleanup_warnings"),
            "predecessor cleanup warnings changed")
    oracle_pause = lifecycle.read_sealed(root/"results/kce-oracle-repair-20261102-v3.user-paused-20261102.json")
    require(oracle_pause.get("exit_verified") is True and oracle_pause.get("no_gpu_supervisor_started") is True,
            "other oracle experiment must remain paused")
    require(not (root/"results/kce-oracle-repair-20261102-v3.supervisor-started.json").exists(),
            "paused experiment unexpectedly started")


def verify_terminal(state, plan):
    require(state.get("schema") == "redknot-glm53-gpu-lifecycle-v1"
            and state.get("holder_restored") is True and state.get("worker_exit") == 0
            and state.get("gpu_ids") == list(range(8))
            and type(state.get("finished_unix")) in (int, float)
            and not state.get("error") and not state.get("cleanup_error") and not state.get("cleanup_warnings")
            and state.get("supervisor_sha256") == lifecycle.SUPERVISOR_SHA
            and state.get("holder_helper_sha256") == lifecycle.HELPER_SHA
            and state.get("child_environment_sha256") == lifecycle.ENV_SHA
            and state.get("runner_sha256") == plan["stage_sources"]["gpu_pilot/run_ffn_only_pilot.py"],
            "original supervisor reported failed/incomplete early-mHC work or cleanup")


def run_plan(plan):
    verify_plan(plan)
    root, stage = Path(plan["root"]), Path(plan["stage_dir"])
    require(root == ROOT and stage == REMOTE_STAGE and STAGE == REMOTE_STAGE,
            "only the independent authorized remote stage may launch")
    lifecycle.verify_lifecycle(root)
    verify_predecessor(root)
    paths = {k: Path(v) for k, v in plan["paths"].items()}
    lifecycle.require_unused(paths)
    lifecycle.write_exclusive(paths["plan"], plan)
    outcome = dict(schema="ffn-only-launch-outcome-v1", plan_sha256=plan["sha256"],
        started_unix=time.time(), successful=False, automatic_retry=False, supervisor_started=False,
        holder_physically_verified=False)
    try:
        lifecycle.write_exclusive(paths["before"], lifecycle.holder_snapshot(root))
        verify_plan(plan); verify_predecessor(root)
        with paths["log"].open("x") as log:
            child = subprocess.Popen(plan["argv"], stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            outcome["supervisor_started"] = True
            try:
                identity = lifecycle.proc_identity(child.pid)
                require(identity["argv"] == plan["argv"], "owned original supervisor argv differs")
                lifecycle.write_exclusive(paths["started"], lifecycle.seal(dict(
                    schema="ffn-only-supervisor-started-v1", **identity, started_unix=time.time(),
                    plan_sha256=plan["sha256"])))
                print(json.dumps(dict(name=NAME, supervisor_pid=child.pid)), flush=True)
            finally:
                # Never inspect/restart the holder while the owned supervisor
                # is still running, even after post-Popen evidence failure.
                outcome["supervisor_exit"] = child.wait()
        require(outcome["supervisor_exit"] == 0, "original supervisor failed")
        verify_terminal(lifecycle.read_json(paths["result_dir"]/"supervisor.json"), plan)
        verify_plan(plan)
        final = lifecycle.read_sealed(paths["result_dir"]/"benchmark/run.json")
        require(final.get("schema") == "kv-cache-engram-ffn-only-pilot-run-v1" and final.get("status") == "complete"
                and final.get("completed_case_indices") == [2, 3, 4, 5], "runner did not finish all four registered cases")
        outcome.update(successful=True, run_sha256=final["sha256"], independent_raw_audit_required=True)
    except BaseException as exc:
        outcome["error"] = repr(exc)
    finally:
        try:
            after = lifecycle.holder_snapshot(root)
            lifecycle.write_exclusive(paths["after"], after)
            outcome.update(holder_physically_verified=True, holder_audit_sha256=after["sha256"])
        except BaseException as exc:
            outcome.update(successful=False, holder_physically_verified=False, holder_error=repr(exc))
        outcome["finished_unix"] = time.time()
        lifecycle.write_exclusive(paths["outcome"], lifecycle.seal(outcome))
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=STAGE/"inputs-sensitivity-v1/manifest.json")
    parser.add_argument("--pilot-protocol", type=Path, default=STAGE/"inputs-sensitivity-v1/protocol.json")
    parser.add_argument("--sensitivity-protocol", type=Path, default=STAGE/"inputs-sensitivity-v1/sensitivity-protocol.json")
    parser.add_argument("--ffn-only-protocol", type=Path, default=STAGE/"ffn-only-protocol.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(ROOT, STAGE, args.manifest, args.pilot_protocol,
                      args.ffn_only_protocol, args.sensitivity_protocol)
    verify_plan(plan); lifecycle.verify_lifecycle(ROOT); verify_predecessor(ROOT)
    if args.check_only:
        print(json.dumps(dict(status="cpu_launch_plan_validated", gpu_initialized=False, plan=plan)))
        return 0
    return 0 if run_plan(plan)["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

