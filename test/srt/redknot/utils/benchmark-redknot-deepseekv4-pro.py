#!/usr/bin/env python3
"""Stable Pro-only CLI for the DeepSeek-V4-Pro-0813 RedKnot benchmark.

The informal ``Pro-0831`` label is treated as ``Pro-0813``: there is no
separate 0831 checkpoint accepted by this entry point.  Benchmark behavior and
all command-line options remain owned by
``benchmark_dsv4_pro0813_redknot_http.py``.  This file only verifies that the
canonical entry point is still bound to the certified Pro-0813 model, driver,
and launcher before delegating to its ``main`` function.

Run ``benchmark-redknot-deepseekv4-pro.py --help`` for the canonical CLI.  The
wrapper has no generic or Flash fallback.
"""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
CANONICAL_ENTRYPOINT = HERE / "benchmark_dsv4_pro0813_redknot_http.py"
PINNED_MODEL = Path("/workspace/Models/DeepSeek-V4-Pro-0813")
PINNED_VARIANT = "deepseek_v4_pro_0813"
PINNED_CONFIG_SHA256 = (
    "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
)
PINNED_GEOMETRY_DIGEST = (
    "sha256:adca138e64f2da316e94dd62394a51bbf5a89ab0651475579ce1977c59497819"
)
PINNED_TP_SIZE = 8
PINNED_NUM_LAYERS = 61
PINNED_NUM_HEADS = 128
PINNED_INDEX_TOPK = 1024
PINNED_DRIVER = HERE / "benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py"
PINNED_LAUNCHER = HERE.parents[3] / "server/start_server_redknot_pro0813.sh"

_CANONICAL_MODULE_NAME = "_redknot_deepseek_v4_pro0813_canonical_cli"
_IDENTITY_FIELDS = (
    ("MODEL", PINNED_MODEL),
    ("PRO0813_VARIANT", PINNED_VARIANT),
    ("PRO0813_CONFIG_SHA256", PINNED_CONFIG_SHA256),
    ("PRO0813_GEOMETRY_DIGEST", PINNED_GEOMETRY_DIGEST),
    ("PRO0813_TP_SIZE", PINNED_TP_SIZE),
    ("PRO0813_NUM_LAYERS", PINNED_NUM_LAYERS),
    ("PRO0813_NUM_HEADS", PINNED_NUM_HEADS),
    ("PRO0813_INDEX_TOPK", PINNED_INDEX_TOPK),
    ("DRIVER", PINNED_DRIVER),
    ("LAUNCHER", PINNED_LAUNCHER),
)


class CanonicalEntrypointError(RuntimeError):
    """The canonical Pro-0813 CLI is missing or no longer identity-bound."""


def _require_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CanonicalEntrypointError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CanonicalEntrypointError(f"{label} must be a regular file")


def _load_canonical() -> ModuleType:
    _require_regular_file(CANONICAL_ENTRYPOINT, "canonical Pro-0813 entry point")
    spec = importlib.util.spec_from_file_location(
        _CANONICAL_MODULE_NAME, CANONICAL_ENTRYPOINT
    )
    if spec is None or spec.loader is None:
        raise CanonicalEntrypointError("cannot load canonical Pro-0813 entry point")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _validate_canonical_identity(module: ModuleType) -> None:
    module_path = getattr(module, "__file__", None)
    if module_path is None or Path(module_path).resolve() != CANONICAL_ENTRYPOINT:
        raise CanonicalEntrypointError(
            "loaded canonical module does not come from the pinned entry point"
        )
    for field, expected in _IDENTITY_FIELDS:
        if getattr(module, field, None) != expected:
            raise CanonicalEntrypointError(
                f"canonical Pro-0813 identity field {field} differs from the pin"
            )
    if not callable(getattr(module, "main", None)):
        raise CanonicalEntrypointError("canonical Pro-0813 main is unavailable")
    _require_regular_file(PINNED_DRIVER, "canonical Pro-0813 benchmark driver")
    _require_regular_file(PINNED_LAUNCHER, "canonical Pro-0813 server launcher")


def main() -> None:
    canonical = _load_canonical()
    _validate_canonical_identity(canonical)
    canonical.main()


if __name__ == "__main__":
    try:
        main()
    except CanonicalEntrypointError as error:
        print(f"{Path(sys.argv[0]).name}: {error}", file=sys.stderr)
        raise SystemExit(2) from None
