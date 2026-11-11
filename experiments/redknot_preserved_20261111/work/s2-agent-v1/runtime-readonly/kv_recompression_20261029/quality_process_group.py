"""Single-host owned-process launcher for the isolated canonical quality runner.

Unlike sequential unbounded joins, monitor every owned rank and each exact
out-dir failure.rankN.json. A worker may publish failure then block in llm.close;
the marker is sufficient to stop all other owned workers. No process search,
process-group signal, GPU API, remote command, or production patch is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable


class ProcessGroupFailure(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        exits = [{"rank": item["rank"], "pid": item["pid"], "exitcode": item["exitcode"],
                  "alive": item["alive"]} for item in report["workers"]]
        super().__init__(f"Owned quality process group failed: {report['reason']}; worker facts: {exits}")


def _duration(value, name, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum} seconds")
    return float(value)


def _snapshot(record):
    process = record["process"]
    pid = process.pid
    alive = bool(process.is_alive()) if pid is not None else False
    return {"rank": record["rank"], "pid": pid, "started": record["started"],
            "start_attempted": record["start_attempted"], "exitcode": process.exitcode,
            "alive": alive, "actions": list(record["actions"]), "startup_error": record.get("startup_error")}


def _markers(root, dp_size):
    """Only exact expected rank paths; malformed/partial markers still fail closed."""
    found = []
    for rank in range(dp_size):
        path = root / f"failure.rank{rank}.json"
        if not path.exists() and not path.is_symlink():
            continue
        fact = {"rank": rank, "path": str(path)}
        try:
            if path.is_symlink() or not path.is_file():
                fact["diagnostic"] = "failure marker exists but is not a regular nonsymlink file"
            else:
                fact["bytes"] = path.stat().st_size
                if fact["bytes"] > 65536:
                    fact["diagnostic"] = "failure marker exceeds 64KiB read limit"
                else:
                    value = json.loads(path.read_text())
                    if isinstance(value, dict):
                        fact["status"] = str(value.get("status", ""))[:256]
                        fact["error"] = str(value.get("error", ""))[:4096]
                    else:
                        fact["diagnostic"] = "failure marker JSON is not an object"
        except (OSError, ValueError, UnicodeError) as exc:
            fact["diagnostic"] = f"unreadable/partial failure marker: {type(exc).__name__}: {exc}"
        found.append(fact)
    return found


def _cleanup_owned(records, terminate_grace, kill_grace, poll_interval):
    """Signal only processes represented by handles constructed in this launch."""
    def live():
        return [r for r in records if r["process"].pid is not None and r["process"].is_alive()]

    def signal_all(method):
        for record in live():
            try:
                getattr(record["process"], method)()
                record["actions"].append({"action": method, "sent": True})
            except (OSError, ValueError, ProcessLookupError) as exc:
                record["actions"].append({"action": method, "sent": False,
                                          "error": f"{type(exc).__name__}: {exc}"})

    def drain(seconds):
        deadline = time.monotonic() + seconds
        while live() and time.monotonic() < deadline:
            # Never join one rank indefinitely while another rank has failed.
            for record in records:
                if record["process"].pid is not None:
                    record["process"].join(timeout=0)
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval, remaining))
        for record in records:
            if record["process"].pid is not None:
                record["process"].join(timeout=0)

    signal_all("terminate")
    drain(terminate_grace)
    if live():
        signal_all("kill")
        drain(kill_grace)
    return [r["rank"] for r in live()]


def _write_report(path, report):
    payload = json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    sealed = {**report, "sha256": hashlib.sha256(payload).hexdigest()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(sealed, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sealed


def launch(
    entry: Callable[[int, int], None],
    dp_size: int,
    out_dir,
    *,
    poll_interval: float = 0.2,
    terminate_grace: float = 3.0,
    kill_grace: float = 3.0,
    _context=None,
) -> dict[str, Any]:
    """Spawn entry(dp_size, rank), fail-fast, clean owned children, then raise.

    RANK/WORLD_SIZE are the outer single-host node coordinates (0/1), not each
    child's DP rank, exactly matching the existing worker entry contract. The
    explicit rank argument identifies the worker. Missing node env values are
    set only for this launch and restored afterwards. Existing incompatible
    distributed env is rejected before any worker starts.

    A success report is returned; a failed report is attached to
    ProcessGroupFailure and saved as process-group.json in the exact out_dir.
    An external supervisor still owns overall run timeouts and descendants;
    this launcher never searches for or signals processes it did not create.
    _context exists solely for deterministic CPU unit tests.
    """
    if not callable(entry) or type(dp_size) is not int or not 1 <= dp_size <= 8:
        raise ValueError("entry must be callable and dp_size an integer in [1,8]")
    poll_interval = _duration(poll_interval, "poll_interval", maximum=1.0)
    terminate_grace = _duration(terminate_grace, "terminate_grace", maximum=30.0)
    kill_grace = _duration(kill_grace, "kill_grace", maximum=30.0)
    requested = Path(out_dir)
    if requested.is_symlink():
        raise ValueError("out_dir must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("out_dir must already be a directory")
    report_path = root / "process-group.json"
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("process-group.json already exists; launch requires fresh output directory")
    for key, expected in (("RANK", "0"), ("WORLD_SIZE", "1")):
        if key in os.environ and os.environ[key] != expected:
            raise ValueError(f"single-host launcher requires {key}={expected}")
    context = _context or multiprocessing.get_context("spawn")
    records, reason, parent_error = [], None, None
    failures_before_cleanup, nonzero_before_cleanup = [], []
    survivors = []
    previous_env = {key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE")}
    started_at = time.monotonic()
    try:
        os.environ["RANK"], os.environ["WORLD_SIZE"] = "0", "1"
        failures_before_cleanup = _markers(root, dp_size)
        if failures_before_cleanup:
            reason = "preexisting_worker_failure_marker"
        else:
            for rank in range(dp_size):
                process = context.Process(target=entry, args=(dp_size, rank), daemon=False,
                                          name=f"canonical-quality-rank-{rank}")
                record = {"rank": rank, "process": process, "started": False,
                          "start_attempted": True, "actions": []}
                records.append(record)
                try:
                    process.start()
                    record["started"] = True
                except BaseException as exc:
                    record["startup_error"] = f"{type(exc).__name__}: {exc}"
                    raise
            while True:
                failures_before_cleanup = _markers(root, dp_size)
                snapshots = [_snapshot(record) for record in records]
                nonzero_before_cleanup = [s for s in snapshots if s["exitcode"] is not None and s["exitcode"] != 0]
                if failures_before_cleanup or nonzero_before_cleanup:
                    reason = "worker_failure_marker" if failures_before_cleanup else "worker_nonzero_exit"
                    break
                if not any(s["alive"] for s in snapshots):
                    for record in records:
                        record["process"].join(timeout=0)
                    # Close the marker/exit observation race at group completion.
                    failures_before_cleanup = _markers(root, dp_size)
                    nonzero_before_cleanup = [s for s in map(_snapshot, records)
                                              if s["exitcode"] is not None and s["exitcode"] != 0]
                    reason = ("worker_failure_marker" if failures_before_cleanup else
                              "worker_nonzero_exit" if nonzero_before_cleanup else "all_workers_exited_zero")
                    break
                time.sleep(poll_interval)
    except BaseException as exc:
        parent_error = f"{type(exc).__name__}: {exc}"
        reason = "parent_or_partial_start_failure"
    finally:
        if reason != "all_workers_exited_zero":
            survivors = _cleanup_owned(records, terminate_grace, kill_grace, poll_interval)
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    report = {
        "schema": "canonical-quality-owned-process-group-v1",
        "status": "complete" if reason == "all_workers_exited_zero" else "failed",
        "reason": reason,
        "node_rank": 0, "node_world_size": 1, "dp_size": dp_size,
        "multiprocessing_start_method": "spawn",
        "out_dir": str(root), "report_path": str(report_path),
        "poll_interval_seconds": poll_interval,
        "terminate_grace_seconds": terminate_grace, "kill_grace_seconds": kill_grace,
        "elapsed_seconds": time.monotonic()-started_at,
        "parent_error": parent_error,
        "failure_markers_before_cleanup": failures_before_cleanup,
        "nonzero_exits_before_cleanup": nonzero_before_cleanup,
        "failure_markers_after_cleanup": _markers(root, dp_size),
        "workers": [_snapshot(record) for record in records],
        "owned_ranks_still_alive_after_cleanup": survivors,
        "only_owned_process_handles_signaled": True,
        "external_supervisor_responsible_for_descendants_and_overall_timeout": True,
    }
    try:
        sealed = _write_report(report_path, report)
    except (OSError, ValueError) as exc:
        report["report_write_error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "failed"
        raise ProcessGroupFailure(report) from exc
    finally:
        for record in records:
            process = record["process"]
            if process.pid is not None and not process.is_alive():
                process.close()
    if sealed["status"] != "complete":
        raise ProcessGroupFailure(sealed)
    return sealed
