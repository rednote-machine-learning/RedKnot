"""One isolated three-hour early-mHC pilot under the original holder supervisor.

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
REMOTE_STAGE = ROOT/"gated_engram_early_mhc_20261102_v2/GatedEngram"
NAME = "kce-early-mhc-20261102-v2"
PREDECESSOR = "kce-early-mhc-20261102-v1"
PREDECESSOR_RUNNER_SHA = "0be3fba63dc03431d89069ce5888c48bd63f9b5375f3f6f10907cd340180a901"
PREDECESSOR_OUTCOME_SEAL = "15fffa59284005a35db885cab94fb3b18d79fbdb0ac069289faf8622ab458644"
PREDECESSOR_FAILURE_SEAL = "c841e4d60b44f759b82e62fd19a4847b55378bfb041979ef686be841b9646658"
WORKFLOW_SCHEMA = "kv-cache-engram-early-mhc-launch-v2"
require = lifecycle.require


def build_plan(root, stage, manifest, pilot_protocol, early_protocol, sensitivity_protocol=None):
    from gpu_pilot.early_mhc_protocol import early_mhc_protocol, DEPENDENCIES
    root, stage = lifecycle.absolute_no_symlink(root), lifecycle.absolute_no_symlink(stage)
    early_protocol = lifecycle.absolute_no_symlink(early_protocol)
    require(stage in early_protocol.parents and early_protocol not in (Path(manifest), Path(pilot_protocol), sensitivity_protocol),
            "early protocol must be a distinct file inside independent stage")
    declared = lifecycle.read_sealed(early_protocol)
    require(declared == early_mhc_protocol(), "early protocol is not the CPU-tested fixed protocol")
    old = lifecycle.build_plan(root, stage, NAME, manifest, pilot_protocol,
                               timeout=10800, prepare_timeout=300, sensitivity_protocol=sensitivity_protocol)
    body = {k: v for k, v in old.items() if k != "sha256"}
    required = {"gpu_pilot/run_early_mhc_pilot.py", "gpu_pilot/launch_early_mhc_pilot.py",
                "gpu_pilot/early_mhc_protocol.py", "kv_cache_engram/early_mhc.py", "tests/test_early_mhc_pilot.py"}
    require(required <= body["stage_sources"].keys(), "new early-mHC modules/tests absent")
    argv = list(body["argv"])
    separator = argv.index("--")
    argv[separator+1] = str(stage/"gpu_pilot/run_early_mhc_pilot.py")
    argv.extend(["--early-mhc-protocol", str(early_protocol)])
    body.update(workflow_schema=WORKFLOW_SCHEMA, argv=argv,
        early_mhc_protocol=dict(path=str(early_protocol), sha256=declared["sha256"],
                                file_sha256=lifecycle.file_digest(early_protocol)),
        predecessor=PREDECESSOR, evaluation_indices=[2, 3, 4, 5], qualification_input_tokens=16384,
        external_sources={str(root/name): lifecycle.file_digest(root/name) for name in DEPENDENCIES},
        cpu_proof_path=str(stage/"EARLY_MHC_CPU_TESTS.json"))
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
            and sum("test_early_mhc_pilot." in name for name in names) >= 6,
            "new early-mHC CPU validation missing, stale, incomplete or skipped")


def verify_plan(plan):
    from gpu_pilot.early_mhc_protocol import DEPENDENCIES
    lifecycle.verify_plan_inputs(plan)
    require(plan.get("workflow_schema") == WORKFLOW_SCHEMA and plan.get("name") == NAME
            and plan.get("timeout") == 10800 and plan.get("prepare_timeout") == 300
            and plan.get("evaluation_indices") == [2, 3, 4, 5], "wrong bounded early-mHC launch plan")
    item = plan["early_mhc_protocol"]
    require(lifecycle.file_digest(item["path"]) == item["file_sha256"]
            and lifecycle.read_sealed(item["path"])["sha256"] == item["sha256"], "early protocol changed")
    argv = plan["argv"]
    require(argv[argv.index("--")+1] == str(Path(plan["stage_dir"])/"gpu_pilot/run_early_mhc_pilot.py")
            and argv[-2:] == ["--early-mhc-protocol", item["path"]], "wrong pilot runner/protocol command")
    expected = {str(Path(plan["root"])/name): sha for name, sha in DEPENDENCIES.items()}
    require(plan.get("external_sources") == expected
            and all(lifecycle.file_digest(path) == sha for path, sha in expected.items()),
            "original sparse operator dependency changed")
    verify_cpu_proof(plan)


def verify_predecessor(root):
    root = Path(root)
    pause = lifecycle.read_sealed(root/"results/kce-oracle-repair-20261102-v3.user-paused-20261102.json")
    require(pause.get("exit_verified") is True and pause.get("no_gpu_supervisor_started") is True,
            "oracle repair must remain paused")
    require(not (root/"results/kce-oracle-repair-20261102-v3.supervisor-started.json").exists(), "paused v3 unexpectedly launched")
    old = lifecycle.read_sealed(root/"results"/(PREDECESSOR+".launch-outcome.json"))
    require(old.get("schema") == "early-mhc-launch-outcome-v1"
            and old.get("sha256") == PREDECESSOR_OUTCOME_SEAL
            and old.get("successful") is False and old.get("supervisor_exit") == 1
            and old.get("holder_physically_verified") is True,
            "only the exact reported failed predecessor may precede this authorized repair")
    failure = lifecycle.read_sealed(root/"results"/PREDECESSOR/"benchmark/failure.rank0.json")
    require(failure.get("sha256") == PREDECESSOR_FAILURE_SEAL
            and failure.get("error") == "RuntimeError('called a synchronizing CUDA operation')"
            and failure.get("completed_indices") == [], "predecessor failure identity changed")
    state = lifecycle.read_json(root/"results"/PREDECESSOR/"supervisor.json")
    require(state.get("schema") == "redknot-glm53-gpu-lifecycle-v1"
            and state.get("status") == "failed" and state.get("worker_exit") == 1
            and state.get("holder_restored") is True
            and state.get("gpu_ids") == list(range(8))
            and type(state.get("finished_unix")) in (int, float)
            and state.get("supervisor_sha256") == lifecycle.SUPERVISOR_SHA
            and state.get("holder_helper_sha256") == lifecycle.HELPER_SHA
            and state.get("child_environment_sha256") == lifecycle.ENV_SHA
            and state.get("runner_sha256") == PREDECESSOR_RUNNER_SHA
            and state.get("error") == "RuntimeError('GLM benchmark failed: exit=1')"
            and not state.get("cleanup_error")
            and state.get("cleanup_warnings") == ["_UnstableOwnership('refusing to signal unauthenticated process-group member 3550938')"],
            "predecessor original supervisor is not the exact known terminal failure")
    for suffix in ("launcher-started", "supervisor-started"):
        started = lifecycle.read_sealed(root/"results"/(PREDECESSOR+"."+suffix+".json"))
        try:
            current = lifecycle.proc_identity(started["pid"])
        except FileNotFoundError:
            continue
        require(current["start_ticks"] != started["start_ticks"], "predecessor process still alive")
    require(not Path("/proc/3550718").exists(), "old worker PID still exists; do not signal it")


def verify_terminal(state, plan):
    require(state.get("schema") == "redknot-glm53-gpu-lifecycle-v1"
            and state.get("holder_restored") is True and state.get("worker_exit") == 0
            and state.get("gpu_ids") == list(range(8))
            and type(state.get("finished_unix")) in (int, float)
            and not state.get("error") and not state.get("cleanup_error") and not state.get("cleanup_warnings")
            and state.get("supervisor_sha256") == lifecycle.SUPERVISOR_SHA
            and state.get("holder_helper_sha256") == lifecycle.HELPER_SHA
            and state.get("child_environment_sha256") == lifecycle.ENV_SHA
            and state.get("runner_sha256") == plan["stage_sources"]["gpu_pilot/run_early_mhc_pilot.py"],
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
    outcome = dict(schema="early-mhc-launch-outcome-v2", plan_sha256=plan["sha256"],
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
                    schema="early-mhc-supervisor-started-v2", **identity, started_unix=time.time(),
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
        require(final.get("schema") == "kv-cache-engram-early-mhc-pilot-run-v2" and final.get("status") == "complete"
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
    parser.add_argument("--early-mhc-protocol", type=Path, default=STAGE/"early-mhc-protocol.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(ROOT, STAGE, args.manifest, args.pilot_protocol,
                      args.early_mhc_protocol, args.sensitivity_protocol)
    verify_plan(plan); lifecycle.verify_lifecycle(ROOT); verify_predecessor(ROOT)
    if args.check_only:
        print(json.dumps(dict(status="cpu_launch_plan_validated", gpu_initialized=False, plan=plan)))
        return 0
    return 0 if run_plan(plan)["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
