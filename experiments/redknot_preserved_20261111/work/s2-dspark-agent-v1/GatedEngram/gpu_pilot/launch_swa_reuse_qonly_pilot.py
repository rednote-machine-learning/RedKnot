"""Finite independent cached-SWA/Q-only FFN15 experiment; original GPU owner only.

Check-only is CPU-only. This launcher never signals a process or operates the
holder itself. A private native snapshot insulates the run from editor changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from gpu_pilot import launch_pilot as lifecycle

ROOT = lifecycle.REMOTE_ROOT
REMOTE_STAGE = ROOT / "gated_engram_swa_reuse_qonly_20261104_v1/GatedEngram"
NATIVE_SNAPSHOT = REMOTE_STAGE.parent / "NativeSnapshot"
NAME = "kce-swa-reuse-qonly-20261104-v1"
PREDECESSOR = "kce-swa-skip-20261104-v1"
WORKFLOW_SCHEMA = "kce-swa-reuse-qonly-launch-v1"
RUN_SCHEMA = "kv-cache-engram-swa-reuse-qonly-pilot-run-v1"
TIMEOUT = 21600
PREPARE_TIMEOUT = 300
require = lifecycle.require


def snapshot_sources(repo):
    repo = lifecycle.absolute_no_symlink(repo)
    require(repo.is_dir() and (repo / "deepinfer/model_executor/models/v4_modules/attn.py").is_file(),
            "independent native source tree missing")
    result = {}
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if any(part in (".git", "__pycache__", ".pytest_cache") for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError("native snapshot must not link back to live sources: " + str(path))
        if path.is_file():
            result[str(rel)] = lifecycle.file_digest(path)
    require(result and "setup.py" in result, "incomplete native snapshot")
    return result


def predecessor_evidence(root):
    root = Path(root)
    results = root / "results"
    state_path = results / PREDECESSOR / "supervisor.json"
    state = lifecycle.read_json(state_path)
    require(state.get("status") == "failed" and state.get("worker_exit") == 1
            and state.get("holder_restored") is True and state.get("finished_unix")
            and not state.get("cleanup_error")
            and state.get("supervisor_sha256") == lifecycle.SUPERVISOR_SHA
            and state.get("holder_helper_sha256") == lifecycle.HELPER_SHA
            and state.get("child_environment_sha256") == lifecycle.ENV_SHA,
            "previous experiment is not terminal with authenticated restoration")
    outcome_path = results / (PREDECESSOR + ".launch-outcome.json")
    outcome = lifecycle.read_sealed(outcome_path)
    require(outcome.get("successful") is False and outcome.get("automatic_retry") is False
            and outcome.get("supervisor_exit") == 1
            and outcome.get("holder_physically_verified") is True,
            "previous failure/restoration outcome missing")
    identities = []
    for suffix in ("launcher-started", "supervisor-started"):
        path = results / (PREDECESSOR + "." + suffix + ".json")
        saved = lifecycle.read_sealed(path)
        try:
            current = lifecycle.proc_identity(saved["pid"])
        except FileNotFoundError:
            current = None
        require(current is None or current["start_ticks"] != saved["start_ticks"],
                "previous owned process still alive")
        identities.append(dict(file=str(path), file_sha256=lifecycle.file_digest(path),
                               pid=saved["pid"], start_ticks=saved["start_ticks"], exit_verified=True))
    # Historical child/cleanup-warning PIDs have no sealed start ticks. Never
    # signal them or infer new ownership from a PID: fail closed if still present.
    for pid in (state["child_pid"], 3838413):
        require(not Path("/proc", str(pid)).exists(), "previous child/warning PID still present")
    return dict(name=PREDECESSOR, status="failed_preserved", state_file=str(state_path),
                state_file_sha256=lifecycle.file_digest(state_path),
                outcome_file=str(outcome_path), outcome_file_sha256=lifecycle.file_digest(outcome_path),
                cleanup_warnings=state.get("cleanup_warnings", []), original_process_exits=identities)


def build_plan(root, stage, manifest, pilot_protocol, swa_protocol, sensitivity_protocol):
    from gpu_pilot.swa_reuse_qonly_protocol import swa_reuse_qonly_protocol
    root, stage = lifecycle.absolute_no_symlink(root), lifecycle.absolute_no_symlink(stage)
    swa_protocol = lifecycle.absolute_no_symlink(swa_protocol)
    require(stage in swa_protocol.parents, "protocol must reside in new independent stage")
    declared = lifecycle.read_sealed(swa_protocol)
    require(declared == swa_reuse_qonly_protocol(), "SWA-reuse-Q-only protocol differs from tested rules")
    original = lifecycle.build_plan(root, stage, NAME, manifest, pilot_protocol,
        timeout=TIMEOUT, prepare_timeout=PREPARE_TIMEOUT, sensitivity_protocol=sensitivity_protocol)
    body = {k: v for k, v in original.items() if k != "sha256"}
    required = {"gpu_pilot/run_swa_reuse_qonly_pilot.py", "gpu_pilot/read_swa_reuse_qonly_results.py",
                "gpu_pilot/launch_swa_reuse_qonly_pilot.py", "gpu_pilot/swa_reuse_qonly_protocol.py",
                "kv_cache_engram/swa_reuse_qonly.py", "gpu_pilot/empty_ep_qualification.py"}
    require(required <= body["stage_sources"].keys(), "new independent modules absent")
    native = stage.parent / "NativeSnapshot"
    native_sources = snapshot_sources(native)
    argv = list(body["argv"])
    argv[argv.index("--") + 1] = str(stage / "gpu_pilot/run_swa_reuse_qonly_pilot.py")
    argv[argv.index("--repo") + 1] = str(native)
    argv.extend(["--swa-reuse-qonly-protocol", str(swa_protocol)])
    body.update(workflow_schema=WORKFLOW_SCHEMA, argv=argv,
        native_snapshot=str(native), native_snapshot_sources=native_sources,
        native_snapshot_sources_sha256=lifecycle.digest(native_sources),
        swa_reuse_qonly_protocol=dict(path=str(swa_protocol), sha256=declared["sha256"],
                               file_sha256=lifecycle.file_digest(swa_protocol)),
        predecessor=PREDECESSOR, predecessor_evidence=predecessor_evidence(root),
        evaluation_indices=[2, 3, 4, 5], cpu_proof_path=str(stage / "SWA_REUSE_QONLY_CPU_TESTS.json"),
        native_joint_attention_preserved=True, cached_document_swa_requires_raw_consumption_audit=True,
        live_native_workspace_is_not_execution_source=True)
    return lifecycle.seal(body)


def verify_cpu_proof(plan):
    proof = lifecycle.read_sealed(plan["cpu_proof_path"])
    names = proof.get("test_ids", [])
    require(proof.get("successful") is True and proof.get("cuda_initialized") is False
            and proof.get("failures") == 0 and proof.get("errors") == 0 and proof.get("skipped") == 0
            and proof.get("source_sha256") == plan["stage_sources"]
            and type(proof.get("tests")) is int and proof["tests"] >= 978
            and len(names) == len(set(names)) == proof["tests"]
            and all(any(family in name for name in names) for family in (
                "test_swa_reuse_qonly_launcher.", "test_swa_reuse_qonly_controller.",
                "test_swa_reuse_qonly_pilot."))
            and any("test_empty_ep_qualification." in name for name in names),
            "new CPU proof is missing, stale, failed, or incomplete")


def verify_plan(plan):
    from gpu_pilot.swa_reuse_qonly_protocol import swa_reuse_qonly_protocol
    lifecycle.verify_plan_inputs(plan)
    require(plan.get("workflow_schema") == WORKFLOW_SCHEMA and plan.get("name") == NAME
            and plan.get("timeout") == TIMEOUT and plan.get("prepare_timeout") == PREPARE_TIMEOUT
            and plan.get("evaluation_indices") == [2, 3, 4, 5]
            and plan.get("predecessor") == PREDECESSOR
            and plan.get("native_joint_attention_preserved") is True
            and plan.get("cached_document_swa_requires_raw_consumption_audit") is True,
            "wrong bounded launch plan")
    item = plan["swa_reuse_qonly_protocol"]
    require(lifecycle.file_digest(item["path"]) == item["file_sha256"]
            and lifecycle.read_sealed(item["path"]) == swa_reuse_qonly_protocol(), "protocol changed")
    stage = Path(plan["stage_dir"])
    native = stage.parent / "NativeSnapshot"
    require(str(native) == plan["native_snapshot"]
            and snapshot_sources(native) == plan["native_snapshot_sources"]
            and lifecycle.digest(plan["native_snapshot_sources"]) == plan["native_snapshot_sources_sha256"],
            "native snapshot changed after freeze")
    argv = plan["argv"]
    require(argv[argv.index("--") + 1] == str(stage / "gpu_pilot/run_swa_reuse_qonly_pilot.py")
            and argv[argv.index("--repo") + 1] == str(native)
            and argv[-2:] == ["--swa-reuse-qonly-protocol", item["path"]], "wrong runner/native/protocol command")
    require(predecessor_evidence(Path(plan["root"])) == plan["predecessor_evidence"],
            "predecessor evidence changed")
    verify_cpu_proof(plan)


def verify_terminal(state, plan):
    require(state.get("schema") == "redknot-glm53-gpu-lifecycle-v1"
            and state.get("status") == "runner_completed; consult benchmark quality/performance gates"
            and state.get("holder_restored") is True and state.get("worker_exit") == 0
            and state.get("gpu_ids") == list(range(8)) and state.get("finished_unix")
            and not state.get("error") and not state.get("cleanup_error") and not state.get("cleanup_warnings")
            and state.get("supervisor_sha256") == lifecycle.SUPERVISOR_SHA
            and state.get("holder_helper_sha256") == lifecycle.HELPER_SHA
            and state.get("child_environment_sha256") == lifecycle.ENV_SHA
            and state.get("runner_sha256") == plan["stage_sources"]["gpu_pilot/run_swa_reuse_qonly_pilot.py"],
            "original supervisor reported failed/incomplete experiment or cleanup")


def run_plan(plan):
    verify_plan(plan)
    root, stage = Path(plan["root"]), Path(plan["stage_dir"])
    require(root == ROOT and stage == REMOTE_STAGE and STAGE == REMOTE_STAGE,
            "launch only from the authorized independent remote stage")
    lifecycle.verify_lifecycle(root)
    paths = {k: Path(v) for k, v in plan["paths"].items()}
    lifecycle.require_unused(paths)
    lifecycle.write_exclusive(paths["plan"], plan)
    outcome = dict(schema="kce-swa-reuse-qonly-launch-outcome-v1", plan_sha256=plan["sha256"],
        started_unix=time.time(), successful=False, automatic_retry=False, supervisor_started=False,
        holder_physically_verified=False)
    try:
        lifecycle.write_exclusive(paths["before"], lifecycle.holder_snapshot(root))
        verify_plan(plan)
        with paths["log"].open("x") as log:
            child = subprocess.Popen(plan["argv"], stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            outcome["supervisor_started"] = True
            try:
                identity = lifecycle.proc_identity(child.pid)
                require(identity["argv"] == plan["argv"], "owned original supervisor argv mismatch")
                lifecycle.write_exclusive(paths["started"], lifecycle.seal(dict(
                    schema="kce-swa-reuse-qonly-supervisor-started-v1", **identity, started_unix=time.time(),
                    plan_sha256=plan["sha256"])))
                print(json.dumps(dict(name=NAME, supervisor_pid=child.pid)), flush=True)
            finally:
                outcome["supervisor_exit"] = child.wait()
        require(outcome["supervisor_exit"] == 0, "original supervisor failed")
        verify_terminal(lifecycle.read_json(paths["result_dir"] / "supervisor.json"), plan)
        verify_plan(plan)
        final = lifecycle.read_sealed(paths["result_dir"] / "benchmark/run.json")
        require(final.get("schema") == RUN_SCHEMA and final.get("status") == "complete"
                and final.get("completed_case_indices") == [2, 3, 4, 5],
                "runner did not finish all four registered cases")
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
    parser.add_argument("--manifest", type=Path, default=STAGE / "inputs-sensitivity-v1/manifest.json")
    parser.add_argument("--pilot-protocol", type=Path, default=STAGE / "inputs-sensitivity-v1/protocol.json")
    parser.add_argument("--sensitivity-protocol", type=Path, default=STAGE / "inputs-sensitivity-v1/sensitivity-protocol.json")
    parser.add_argument("--swa-reuse-qonly-protocol", type=Path, default=STAGE / "swa-reuse-qonly-protocol.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(ROOT, STAGE, args.manifest, args.pilot_protocol, args.swa_reuse_qonly_protocol, args.sensitivity_protocol)
    verify_plan(plan)
    if args.check_only:
        print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
        return 0
    outcome = run_plan(plan)
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

