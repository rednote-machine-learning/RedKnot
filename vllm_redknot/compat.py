"""Source-level compatibility checks, without importing vLLM or torch."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from .config import ENGINE_FAMILIES


def verify_vllm_sources(
    vllm_root: str | Path | None = None, *, engine_family: str = "mha"
) -> dict[str, str]:
    """Fail closed on API drift in files used by the integration.

    Args:
        vllm_root: Optional path to the Python ``vllm/`` package, not its parent.
        engine_family: Explicit adapter profile. DSV4 verifies its native model,
            metadata, projection and built FlashMLA Python interface, not the
            unrelated MHA model/backend files. This does not attest CUDA binary
            provenance or model weights; those need separate qualification.
    """
    if not isinstance(engine_family, str) or engine_family not in ENGINE_FAMILIES:
        raise ValueError("engine_family must be mha or deepseek_v4_flash")
    if vllm_root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vLLM is not installed in the selected interpreter")
        package = Path(spec.origin).parent
    else:
        package = Path(vllm_root)
    manifest = json.loads(Path(__file__).with_name("pinned_files.json").read_text())
    if engine_family == "mha":
        files = manifest.get("files")
    else:
        files = manifest.get("family_files", {}).get(engine_family)
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"No source contract for engine family {engine_family}")
    mismatches = []
    for relative, expected in files.items():
        candidate = package / relative
        if not candidate.is_file():
            mismatches.append(f"{relative}: missing")
        elif hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            mismatches.append(f"{relative}: hash mismatch")
    if mismatches:
        raise RuntimeError(
            "Unsupported vLLM source drift; review before enabling RedKnot: "
            + "; ".join(mismatches)
        )
    return {
        "status": "SOURCE_CONTRACT_OK",
        "commit": manifest["commit"],
        "package": str(package),
        "engine_family": engine_family,
    }
