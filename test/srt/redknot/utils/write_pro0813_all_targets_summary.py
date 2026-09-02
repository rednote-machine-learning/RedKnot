#!/usr/bin/env python3
"""Write the machine-readable completion record for the five-target sweep."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path


TARGETS = (
    ("64k", 65536),
    ("128k", 131072),
    ("256k", 262144),
    ("440k", 450560),
    ("512k", 524288),
)
RESTORED = re.compile(
    r"holder_restarted pid=[1-9][0-9]* workers=8 "
    r"util_threshold_pct=[0-9]+ util_good_samples=[0-9]+/[0-9]+ "
    r"mapping=.+\n"
)
RETAINED = re.compile(
    r"holder_retained pid=[1-9][0-9]* workers=8 "
    r"util_threshold_pct=[0-9]+ util_good_samples=[0-9]+/[0-9]+ "
    r"mapping=.+\n"
)
FAILED = re.compile(r"holder_restore_failed pid=[1-9][0-9]*\n")


def _holder_outcome(run_dir: Path) -> dict[str, object]:
    receipt = run_dir / "holder_restore_status"
    try:
        metadata = receipt.lstat()
    except FileNotFoundError:
        return {
            "state": "no_handoff_receipt",
            "receipt": str(receipt),
            "proven": False,
        }
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        return {
            "state": "invalid_receipt",
            "receipt": str(receipt),
            "proven": False,
        }
    if metadata.st_size <= 0 or metadata.st_size > 16384:
        return {
            "state": "invalid_receipt",
            "receipt": str(receipt),
            "proven": False,
        }
    try:
        content = receipt.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {
            "state": "invalid_receipt",
            "receipt": str(receipt),
            "proven": False,
        }
    if RESTORED.fullmatch(content):
        state = "restored"
        proven = True
    elif RETAINED.fullmatch(content):
        state = "retained"
        proven = True
    elif FAILED.fullmatch(content):
        state = "restore_failed"
        proven = False
    else:
        state = "invalid_receipt"
        proven = False
    return {"state": state, "receipt": str(receipt), "proven": proven}


def build_summary(root_run_dir: Path, exit_codes: list[int]) -> dict:
    if not root_run_dir.is_absolute() or root_run_dir == Path("/"):
        raise ValueError("root run directory must be absolute and non-root")
    if len(exit_codes) != len(TARGETS):
        raise ValueError("exactly five target exit codes are required")
    if any(type(value) is not int or not 0 <= value <= 255 for value in exit_codes):
        raise ValueError("target exit codes must be integers in [0, 255]")
    records = []
    overall_exit_code = 0
    for ordinal, ((label, target_tokens), exit_code) in enumerate(
        zip(TARGETS, exit_codes), start=1
    ):
        run_dir = root_run_dir / label
        holder = _holder_outcome(run_dir)
        benchmark_success = exit_code == 0 and holder["state"] == "restored"
        if overall_exit_code == 0 and exit_code != 0:
            overall_exit_code = exit_code
        # A successful benchmark necessarily crossed the holder handoff.  Its
        # exit cannot be considered successful without the supervisor's own
        # authenticated final-holder receipt.
        if exit_code == 0 and not benchmark_success and overall_exit_code == 0:
            overall_exit_code = 74
        records.append(
            {
                "ordinal": ordinal,
                "label": label,
                "target_tokens": target_tokens,
                "run_dir": str(run_dir),
                "exit_code": exit_code,
                "success": benchmark_success,
                "holder": holder,
            }
        )
    return {
        "format": "redknot_pro0813_all_targets_summary_v1",
        "root_run_dir": str(root_run_dir),
        "target_count": len(records),
        "continued_after_target_failure": True,
        "targets": records,
        "overall_exit_code": overall_exit_code,
        "pass": overall_exit_code == 0,
    }


def write_summary(path: Path, summary: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing summary: {path}")
    payload = (
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while publishing summary")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run-dir", required=True, type=Path)
    parser.add_argument(
        "--exit-code", action="append", type=int, required=True, dest="exit_codes"
    )
    args = parser.parse_args()
    root_run_dir = args.root_run_dir
    summary = build_summary(root_run_dir, args.exit_codes)
    output = root_run_dir / "all_targets_summary.json"
    write_summary(output, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
