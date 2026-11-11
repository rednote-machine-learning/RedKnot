"""Finite CPU queue for the independently authorized oracle-repair diagnostic.

Never interrupts the predecessor or allocates a GPU. Only the pinned original
supervisor may release/restore the holder after successful, raw-audited 4/4
completion, process exit verification, and physical eight-GPU holder admission.
All publication is exclusive: a failed/partial name is never retried in place.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_pilot import launch_pilot as original
from gpu_pilot.repair_protocol import repair_protocol


NAME = "kce-oracle-repair-20261102-v3"
WAIT_FOR = "kce-sensitivity-20261102-v2"
PREDECESSOR_STAGE = "gated_engram_sensitivity_20261102_v2/GatedEngram"
READER_SHA = "bcfc7a6c1a0f32103906c349f09fe99053fafc73f6f112660c1a0ad51d493ebf"
PLAN_SCHEMA = "kv-cache-engram-oracle-repair-launch-v1"
RUNNER = "gpu_pilot/run_repair_pilot.py"
POLL_SECONDS = 30
MAX_WAIT_SECONDS = 21600
require = original.require


def paths_for(root, name):
    paths = original.evidence_paths(root, name)
    paths.update(intent=root / "results" / (name + ".queue-intent.json"),
                 queue_started=root / "results" / (name + ".queue-started.json"),
                 queue_outcome=root / "results" / (name + ".queue-outcome.json"),
                 predecessor_audit=root / "results" / (name + ".predecessor-audit.json"))
    return paths


def predecessor_inputs(root, wait_for):
    require(wait_for == WAIT_FOR, "only the explicitly authorized sensitivity predecessor may be awaited")
    paths = original.evidence_paths(root, wait_for)
    plan = original.read_sealed(paths["plan"], original.PLAN_SCHEMA)
    stage = root / PREDECESSOR_STAGE
    require(plan.get("name") == wait_for and plan.get("root") == str(root)
            and plan.get("stage_dir") == str(stage), "foreign predecessor plan/root/stage")
    require(plan.get("paths") == {key: str(value) for key, value in paths.items()},
            "predecessor paths do not belong to this exact run")
    original.verify_plan_inputs(plan)
    reader = stage / "gpu_pilot/read_results.py"
    require(plan["stage_sources"].get("gpu_pilot/read_results.py") == READER_SHA
            and original.file_digest(reader) == READER_SHA, "predecessor reader identity changed")
    launcher_path = root / "results" / (wait_for + ".launcher-started.json")
    launcher = original.read_sealed(launcher_path, "kv-cache-engram-sensitivity-launcher-started-v1")
    started = original.read_sealed(paths["started"], "kv-cache-engram-pilot-supervisor-started-v1")
    require(launcher.get("source_sha256") == plan["stage_sources_sha256"]
            and launcher.get("process", {}).get("argv") == launcher.get("argv")
            and str(stage / "gpu_pilot/launch_pilot.py") in launcher.get("argv", [])
            and wait_for in launcher["argv"], "predecessor launcher binding mismatch")
    process = launcher["process"]
    require(type(process.get("pid")) is int and process["pid"] > 1
            and type(process.get("start_ticks")) is int and process["start_ticks"] > 0,
            "predecessor launcher requires PID/start-time identity")
    require(started.get("plan_sha256") == plan["sha256"] and started.get("argv") == plan["argv"]
            and type(started.get("pid")) is int and started["pid"] > 1,
            "predecessor supervisor start is not bound to its plan")
    return dict(name=wait_for, plan_path=str(paths["plan"]), plan_sha256=plan["sha256"],
        plan_file_sha256=original.file_digest(paths["plan"]), stage_dir=str(stage),
        reader_path=str(reader), reader_sha256=READER_SHA,
        launcher_started_path=str(launcher_path), launcher_started_sha256=launcher["sha256"],
        supervisor_started_path=str(paths["started"]), supervisor_started_sha256=started["sha256"]), plan


def build_plan(root, stage_dir, name, manifest, protocol, sensitivity_protocol, repair_protocol_path,
               *, wait_for=WAIT_FOR, timeout=21600, prepare_timeout=300, wait_timeout=21600):
    require(name == NAME, "only the authorized oracle-repair run name is accepted")
    require(type(wait_timeout) is int and 1 <= wait_timeout <= MAX_WAIT_SECONDS,
            "finite CPU wait timeout 1..21600 required")
    base = original.build_plan(root, stage_dir, name, manifest, protocol, timeout, prepare_timeout,
                               sensitivity_protocol=sensitivity_protocol)
    root, stage = Path(base["root"]), Path(base["stage_dir"])
    require(root / "DeepInfer-PR148" not in stage.parents,
            "repair GPU stage must be independent of the native repository")
    require(RUNNER in base["stage_sources"] and "gpu_pilot/launch_repair_pilot.py" in base["stage_sources"],
            "independent repair runner and queue sources are required")
    repair_path = original.absolute_no_symlink(repair_protocol_path)
    require(stage in repair_path.parents and str(repair_path) not in
            (base["manifest"]["path"], base["protocol"]["path"], base["sensitivity_protocol"]["path"]),
            "repair protocol must be a separate frozen file inside the new stage")
    repair = original.read_sealed(repair_path, "kv-cache-engram-oracle-repair-protocol-v1")
    require(repair == repair_protocol(), "repair protocol differs from CPU-tested declarations")
    predecessor, old_plan = predecessor_inputs(root, wait_for)
    require(stage != Path(old_plan["stage_dir"]), "never modify or run inside the active predecessor stage")
    for key in ("manifest", "protocol", "sensitivity_protocol"):
        require(base[key]["sha256"] == old_plan[key]["sha256"]
                and base[key]["file_sha256"] == old_plan[key]["file_sha256"],
                "repair must use the exact already-exposed predecessor input bytes: " + key)
    paths = paths_for(root, name)
    original.require_unused(paths)
    value = {key: item for key, item in base.items() if key != "sha256"}
    argv = list(base["argv"])
    argv[argv.index(str(stage / "gpu_pilot/run_pilot.py"))] = str(stage / RUNNER)
    argv.extend(["--repair-protocol", str(repair_path)])
    value.update(schema=PLAN_SCHEMA, argv=argv, runner=RUNNER,
        paths={key: str(path) for key, path in paths.items()}, predecessor=predecessor,
        repair_protocol=dict(path=str(repair_path), sha256=repair["sha256"],
                             file_sha256=original.file_digest(repair_path)),
        wait_timeout=wait_timeout, poll_seconds=POLL_SECONDS, waiting_is_cpu_only=True,
        predecessor_complete_and_physical_holder_required=True,
        check_only_scope="immutable CPU queue/launch plan only; no process, GPU query, allocation or release")
    return original.seal(value)


def verify_plan(plan):
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("name") == NAME
            and plan.get("sha256") == original.digest({k: v for k, v in plan.items() if k != "sha256"}),
            "repair launch plan seal/schema mismatch")
    require(original.source_files(Path(plan["stage_dir"])) == plan["stage_sources"],
            "repair stage changed after freeze")
    for key in ("manifest", "protocol", "sensitivity_protocol", "repair_protocol"):
        item = plan[key]
        require(original.file_digest(item["path"]) == item["file_sha256"]
                and original.read_sealed(item["path"])["sha256"] == item["sha256"],
                key + " changed after freeze")
    current, _ = predecessor_inputs(Path(plan["root"]), plan["predecessor"]["name"])
    require(current == plan["predecessor"], "predecessor immutable identity changed")


def process_state(pid):
    """Read-only /proc inspection; never signal a process or guess reused PIDs."""
    require(type(pid) is int and pid > 1, "invalid process PID")
    folder = Path("/proc") / str(pid)
    try:
        stat = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        argv = [os.fsdecode(value) for value in (folder / "cmdline").read_bytes().split(b"\0") if value]
    except FileNotFoundError:
        return None
    return dict(pid=pid, state=stat[0], start_ticks=int(stat[19]), ppid=int(stat[1]),
                pgid=int(stat[2]), sid=int(stat[3]), argv=argv)


def _exited(pid, *, expected=None, required_argv=None):
    current = process_state(pid)
    if current is None or current["state"] in ("Z", "X", "x"):
        return True
    if expected is not None and current["start_ticks"] != expected["start_ticks"]:
        return True  # the recorded original process has exited; never touch its replacement
    require(required_argv is None or current["argv"] == required_argv,
            "process PID/argv ownership mismatch; no action permitted")
    if expected is not None:
        require(current["argv"] == expected["argv"], "predecessor launcher identity changed")
    return False


def predecessor_terminal(plan):
    """None means healthy waiting. Failed or inconsistent evidence raises."""
    root = Path(plan["root"])
    dep = plan["predecessor"]
    paths = original.evidence_paths(root, dep["name"])
    old_plan = original.read_sealed(paths["plan"], original.PLAN_SCHEMA)
    require(old_plan["sha256"] == dep["plan_sha256"], "predecessor plan changed")
    supervisor_path = paths["result_dir"] / "supervisor.json"
    supervisor = original.read_json(supervisor_path) if supervisor_path.exists() else None
    if supervisor is not None:
        require(not supervisor.get("error") and not supervisor.get("cleanup_error")
                and not supervisor.get("cleanup_warnings"), "predecessor reported failure or cleanup warning")
        require(supervisor.get("worker_exit") in (None, 0), "predecessor worker failed; never retry")
    if not paths["outcome"].exists():
        launcher = original.read_sealed(dep["launcher_started_path"])
        require(not _exited(launcher["process"]["pid"], expected=launcher["process"]),
                "predecessor launcher exited without a terminal outcome; no retry or launch")
        return None
    outcome = original.read_sealed(paths["outcome"], "kv-cache-engram-pilot-launch-outcome-v1")
    require(outcome.get("plan_sha256") == old_plan["sha256"]
            and outcome.get("successful") is True and outcome.get("supervisor_exit") == 0
            and outcome.get("supervisor_started") is True
            and outcome.get("holder_physically_verified") is True
            and not outcome.get("error") and not outcome.get("holder_verification_error"),
            "predecessor launcher failed or lacks successful holder handoff")
    require(supervisor is not None, "successful predecessor lacks supervisor evidence")
    original.verify_supervisor_result(supervisor, old_plan)
    launcher = original.read_sealed(dep["launcher_started_path"])
    started = original.read_sealed(dep["supervisor_started_path"])
    require(started["pid"] == outcome.get("supervisor_pid"), "supervisor PID differs across evidence")
    if not _exited(launcher["process"]["pid"], expected=launcher["process"]):
        return None
    if not _exited(started["pid"], required_argv=old_plan["argv"]):
        return None
    child_pid = supervisor.get("child_pid")
    require(type(child_pid) is int and child_pid > 1, "predecessor child identity missing")
    if not _exited(child_pid):
        return None
    after = original.read_sealed(paths["after"], "kv-cache-engram-pilot-holder-audit-v1")
    require(outcome.get("holder_audit_sha256") == after["sha256"], "predecessor holder audit is not bound")
    return dict(outcome=outcome, supervisor=supervisor,
                processes_exited=True, previous_holder_audit_sha256=after["sha256"])


def audit_predecessor(plan, *, timeout=600):
    """Execute only the SHA-verified predecessor's standard-library reader."""
    verify_plan(plan)
    dep = plan["predecessor"]
    directory = Path(plan["root"]) / "results" / dep["name"] / "benchmark"
    require(original.file_digest(dep["reader_path"]) == READER_SHA, "reader changed before execution")
    result = subprocess.run([str(original.RUNTIME / "di-venv/bin/python"), "-B", "-s",
                             dep["reader_path"], str(directory)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=False)
    require(result.returncode == 0 and not result.stderr.strip(), "predecessor read-only raw audit failed")
    value = json.loads(result.stdout, object_pairs_hook=original._unique_object)
    require(type(value) is dict and value.get("sha256") == original.digest(
        {key: item for key, item in value.items() if key != "sha256"}), "raw reader output seal mismatch")
    require(value.get("schema") == "kv-cache-engram-native-pilot-readout-v1"
            and value.get("status") == "complete" and value.get("completed") == 4
            and value.get("planned") == 4 and value.get("sensitivity_enabled") is True
            and not value.get("failures") and not value.get("audit_errors")
            and not value.get("pending_case_indices"), "predecessor lacks complete audited 4/4 evidence")
    require([case.get("index") for case in value.get("cases", [])] == [2, 3, 4, 5]
            and all(value.get("by_dataset", {}).get(dataset, {}).get("completed") == 2
                    for dataset in ("hotpotqa", "2wikimqa")), "predecessor dataset/case coverage mismatch")
    verify_plan(plan)
    return value


def wait_until_admitted(plan, *, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + plan["wait_timeout"]
    while True:
        verify_plan(plan)
        require(clock() < deadline, "finite predecessor wait timed out; no GPU launch")
        terminal = predecessor_terminal(plan)
        if terminal is not None:
            remaining = deadline - clock()
            require(remaining > 0, "finite predecessor wait timed out")
            audit = audit_predecessor(plan, timeout=min(600, remaining))
            require(clock() < deadline, "predecessor audit exceeded finite wait budget")
            require(predecessor_terminal(plan) == terminal, "predecessor changed during raw audit")
            holder = original.holder_snapshot(Path(plan["root"]))
            require(clock() < deadline, "holder verification exceeded finite wait budget")
            return original.seal(dict(schema="kv-cache-engram-repair-predecessor-admission-v1",
                plan_sha256=plan["sha256"], predecessor=plan["predecessor"],
                terminal=terminal, raw_audit=audit, current_holder=holder,
                physically_verified=True, admitted_unix=time.time()))
        sleep(min(plan["poll_seconds"], max(0, deadline - clock())))


def _verify_repair_supervisor(value, plan):
    # Reuse the unchanged lifecycle verification with only its runner lookup
    # redirected to the new, separately frozen executable.
    adapted = dict(plan, stage_sources=dict(plan["stage_sources"],
        **{"gpu_pilot/run_pilot.py": plan["stage_sources"][RUNNER]}))
    original.verify_supervisor_result(value, adapted)


def run_supervisor(plan, admission):
    verify_plan(plan)
    root, paths = Path(plan["root"]), {key: Path(value) for key, value in plan["paths"].items()}
    original.verify_lifecycle(root)
    require(admission.get("schema") == "kv-cache-engram-repair-predecessor-admission-v1"
            and admission.get("sha256") == original.digest(
                {key: value for key, value in admission.items() if key != "sha256"})
            and admission.get("plan_sha256") == plan["sha256"]
            and admission.get("physically_verified") is True
            and predecessor_terminal(plan) == admission["terminal"], "predecessor no longer admitted")
    outcome = dict(schema="kv-cache-engram-pilot-launch-outcome-v1", plan_sha256=plan["sha256"],
        started_unix=time.time(), supervisor_started=False, successful=False,
        holder_physically_verified=False, automatic_retry=False)
    try:
        before = original.holder_snapshot(root)
        original.write_exclusive(paths["before"], before)
        verify_plan(plan)
        with paths["log"].open("x") as log:
            child = subprocess.Popen(plan["argv"], stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            outcome.update(supervisor_started=True, supervisor_pid=child.pid)
            # Always wait for the known original restorer, even if evidence
            # publication fails; do not let a logging failure orphan cleanup.
            try:
                identity = process_state(child.pid)
                original.write_exclusive(paths["started"], original.seal(dict(
                    schema="kv-cache-engram-pilot-supervisor-started-v1", pid=child.pid,
                    process=identity, plan_sha256=plan["sha256"], argv=plan["argv"], started_unix=time.time())))
            finally:
                outcome["supervisor_exit"] = child.wait()
        require(outcome["supervisor_exit"] == 0, "original supervisor returned nonzero")
        _verify_repair_supervisor(original.read_json(paths["result_dir"] / "supervisor.json"), plan)
        require(_exited(child.pid), "supervisor remains active after wait")
        verify_plan(plan)
        outcome["successful"] = True
    except BaseException as exc:
        outcome["error"] = repr(exc)
    finally:
        try:
            after = original.holder_snapshot(root)
            original.write_exclusive(paths["after"], after)
            outcome.update(holder_physically_verified=True, holder_audit_sha256=after["sha256"])
        except BaseException as exc:
            outcome.update(successful=False, holder_verification_error=repr(exc))
        outcome["finished_unix"] = time.time()
        original.write_exclusive(paths["outcome"], original.seal(outcome))
    return 0 if outcome["successful"] else 1


def run_queue(plan):
    verify_plan(plan)
    paths = {key: Path(value) for key, value in plan["paths"].items()}
    original.require_unused(paths)
    process = original.proc_identity(os.getpid())
    # The intent is the exclusive name claim; no second queue can get past it.
    intent = original.seal(dict(schema="kv-cache-engram-repair-queue-intent-v1",
        plan_sha256=plan["sha256"], process=process, predecessor=plan["predecessor"],
        wait_timeout=plan["wait_timeout"], poll_seconds=plan["poll_seconds"],
        no_automatic_retry=True, other_experiments_resumed=False, created_unix=time.time()))
    original.write_exclusive(paths["intent"], intent)
    original.write_exclusive(paths["plan"], plan)
    original.write_exclusive(paths["queue_started"], original.seal(dict(
        schema="kv-cache-engram-repair-queue-started-v1", intent_sha256=intent["sha256"],
        plan_sha256=plan["sha256"], process=process, started_unix=time.time(), cpu_only_wait=True)))
    outcome = dict(schema="kv-cache-engram-repair-queue-outcome-v1", plan_sha256=plan["sha256"],
        process=process, successful=False, gpu_supervisor_invoked=False, automatic_retry=False)
    try:
        admission = wait_until_admitted(plan)
        original.write_exclusive(paths["predecessor_audit"], admission)
        outcome["gpu_supervisor_invoked"] = True
        code = run_supervisor(plan, admission)
        outcome["successful"] = code == 0
        outcome["launch_exit"] = code
    except BaseException as exc:
        outcome["error"] = repr(exc)
    finally:
        outcome["finished_unix"] = time.time()
        original.write_exclusive(paths["queue_outcome"], original.seal(outcome))
    return 0 if outcome["successful"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--root", type=Path, default=original.REMOTE_ROOT)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-protocol", type=Path, required=True)
    parser.add_argument("--sensitivity-protocol", type=Path, required=True)
    parser.add_argument("--repair-protocol", type=Path, required=True)
    parser.add_argument("--wait-for", required=True)
    parser.add_argument("--wait-timeout", type=int, default=21600)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--prepare-timeout", type=int, default=300)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(args.root, args.stage_dir, args.name, args.manifest, args.pilot_protocol,
        args.sensitivity_protocol, args.repair_protocol, wait_for=args.wait_for,
        timeout=args.timeout, prepare_timeout=args.prepare_timeout, wait_timeout=args.wait_timeout)
    if args.check_only:
        print(json.dumps(dict(status="CPU immutable queue plan validated; no launch or GPU query",
                             gpu_initialized=False, lifecycle_identity_checked=False, plan=plan)))
        return 0
    return run_queue(plan)


if __name__ == "__main__":
    raise SystemExit(main())
