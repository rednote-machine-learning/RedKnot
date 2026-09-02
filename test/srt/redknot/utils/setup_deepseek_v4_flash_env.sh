#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
HARDWARE_PROFILE="${REDKNOT_HARDWARE_PROFILE:-h200}"
if [[ "$HARDWARE_PROFILE" == "b300" ]]; then
  DEFAULT_BOOTSTRAP_PYTHON=/usr/bin/python3.12
  DEFAULT_VENV="${REPO_ROOT}/.venv_sm103"
else
  DEFAULT_BOOTSTRAP_PYTHON=python3.11
  DEFAULT_VENV="${REPO_ROOT}/.venv_tf5"
fi
PYTHON_BIN="${REDKNOT_BOOTSTRAP_PYTHON:-$DEFAULT_BOOTSTRAP_PYTHON}"
VENV="${REDKNOT_VENV:-$DEFAULT_VENV}"
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "missing environment: ${VENV}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
profile = os.environ.get("REDKNOT_HARDWARE_PROFILE", "h200")
allowed = {(3, 11), (3, 12)} if profile == "b300" else {(3, 11)}
if sys.version_info[:2] not in allowed:
    raise SystemExit(
        f"RedKnot {profile} requires CPython {sorted(allowed)}, got {sys.version}"
    )
PY
  "${PYTHON_BIN}" -m venv "${VENV}"
fi

if [[ "${CHECK_ONLY}" == "0" ]]; then
  "${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV}/bin/python" -m pip install \
    -r "${SCRIPT_DIR}/requirements-deepseek-v4-flash.txt"
  "${VENV}/bin/python" -m pip install --no-deps -e "${REPO_ROOT}/python"
fi

PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VENV}/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
import sys

profile = os.environ.get("REDKNOT_HARDWARE_PROFILE", "h200")
allowed_python = {(3, 11), (3, 12)} if profile == "b300" else {(3, 11)}
if sys.version_info[:2] not in allowed_python:
    raise SystemExit(
        f"expected {profile} CPython {sorted(allowed_python)}, got {sys.version}"
    )

import deep_gemm
import flash_mla
import sgl_kernel
import torch

required_flash_mla = (
    "get_mla_metadata",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
)
missing = [name for name in required_flash_mla if not hasattr(flash_mla, name)]
if missing:
    raise SystemExit(f"FlashMLA API mismatch: missing {missing}")
expected_cuda = "12.9" if profile == "b300" else "12.8"
if torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"expected {profile} PyTorch CUDA {expected_cuda}, got {torch.version.cuda}"
    )

expected = {
    "torch": "2.9.1+cu129" if profile == "b300" else "2.9.1",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "safetensors": "0.8.0",
    "flashinfer-python": "0.5.3",
    "sgl-kernel": "0.3.20",
    "triton": "3.5.1",
}
observed = {name: metadata.version(name) for name in expected}
mismatch = {
    name: {"expected": expected[name], "observed": observed[name]}
    for name in expected
    if observed[name] != expected[name]
}
if mismatch:
    raise SystemExit(f"dependency version mismatch: {mismatch}")

try:
    flash_mla_version = metadata.version("flash-mla")
except metadata.PackageNotFoundError:
    if profile != "b300":
        raise
    # The B300 build is imported from its audited SM103 source tree rather
    # than installed as a wheel, so it intentionally has no dist-info record.
    flash_mla_version = getattr(flash_mla, "__version__", "sm103-source-tree")

print(json.dumps({
    "status": "ready",
    "hardware_profile": profile,
    "python": platform.python_version(),
    "torch_cuda": torch.version.cuda,
    "packages": observed,
    "flash_mla": flash_mla_version,
    "deep_gemm": deep_gemm.__file__,
    "sgl_kernel": sgl_kernel.__file__,
}, sort_keys=True))
PY
