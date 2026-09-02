#!/usr/bin/env python3
"""Visible Pro-0813 benchmark alias; all implementation stays isolated.

This file lives beside the untouched Flash benchmark for discovery, but it
delegates only to the Pro-specific entry point in this repository.  It has no
fallback to a Flash model and does not mutate the Flash benchmark path.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


PYTHON = Path("/workspace/RedKnot/.venv_sm103/bin/python")
PRO_ENTRY = Path(
    "/workspace/RedKnot/test/srt/redknot/utils/"
    "benchmark-redknot-deepseekv4-pro.py"
)
PRO_PYTHONPATH = (
    "/workspace/RedKnot/python:/data/temp/FlashMLA-sm103-src"
)


def _require_interpreter(path: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"required Pro interpreter is unavailable: {path}") from error
    if not resolved.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"required Pro interpreter is not executable: {path}")


def _require_regular(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"required Pro benchmark path is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"required Pro benchmark path is not regular: {path}")


def _environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("PYTHON")
    }
    environment.update(
        {
            "PYTHONPATH": PRO_PYTHONPATH,
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def main() -> None:
    _require_interpreter(PYTHON)
    _require_regular(PRO_ENTRY)
    os.execve(
        str(PYTHON),
        [str(PYTHON), str(PRO_ENTRY), *sys.argv[1:]],
        _environment(),
    )
    raise RuntimeError("execve unexpectedly returned")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"{Path(sys.argv[0]).name}: {error}", file=sys.stderr)
        raise SystemExit(2) from None
