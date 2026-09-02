#!/usr/bin/env python3
"""Strict B300/SM103 gate for Pro-0813 shared-latent batch restore.

The package-level oracle owns the independent Torch numerical comparisons.
This deployment wrapper additionally binds the result to the isolated Pro
checkout, the official 61-layer/128-head/rope64/q_lora1536/TP8 geometry, the
correct 27-C4/28-C128 workspace sizing, and a source manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from sglang.srt.layers import deepseek_v4_rope
from sglang.srt.layers.attention.redknot import (
    dsv4_offline_reuse_v2,
    dsv4_rope_reloc,
    dsv4_shared_latent_batch_kernels as batch_kernels,
    dsv4_shared_latent_cache,
    dsv4_shared_latent_sglang,
    probe_dsv4_shared_latent_batch_kernels as oracle,
)
from sglang.srt.layers.attention.redknot.pro0813 import profile


PRO_ROOT = Path("/workspace/RedKnot_Pro0813").resolve()
EXPECTED_PTXAS = Path("/usr/local/cuda/bin/ptxas").resolve()
EXPECTED_JOBS_PER_REQUEST = 27 * 5 + 28 * 3


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _module_path(module: object) -> Path:
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    if not path.is_file() or PRO_ROOT not in path.parents:
        raise RuntimeError(f"module escaped isolated Pro checkout: {path}")
    return path


def _expected_validation_entries(max_requests: int, max_batch_rows: int) -> int:
    def summed_ceil(divisor: int) -> int:
        return (
            max_batch_rows + (divisor - 1) * max_requests
        ) // divisor

    return (
        55 * max_batch_rows
        + 54 * summed_ceil(4)
        + 28 * summed_ceil(128)
        + 108 * summed_ceil(512)
        + 28 * summed_ceil(512)
    )


def _require_official_geometry() -> dict[str, object]:
    geometry = profile.PRO0813_TP8_GEOMETRY
    observed = {
        "variant": geometry.variant,
        "official_config_sha256": geometry.official_config_sha256,
        "geometry_digest": geometry.geometry_digest,
        "num_target_layers": geometry.num_target_layers,
        "num_attention_heads": geometry.num_attention_heads,
        "qk_rope_head_dim": geometry.qk_rope_head_dim,
        "q_lora_rank": geometry.q_lora_rank,
        "tp_size": geometry.tp_size,
        "heads_per_tp_rank": geometry.heads_per_tp_rank,
        "reusable_layer_ids": tuple(geometry.reusable_layer_ids),
    }
    expected = {
        "variant": profile.PRO0813_VARIANT,
        "official_config_sha256": profile.PRO0813_OFFICIAL_CONFIG_SHA256,
        "geometry_digest": profile.PRO0813_TP8_GEOMETRY_DIGEST,
        "num_target_layers": 61,
        "num_attention_heads": 128,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1536,
        "tp_size": 8,
        "heads_per_tp_rank": 16,
        "reusable_layer_ids": tuple(range(3, 58)),
    }
    if observed != expected:
        raise RuntimeError(
            f"official Pro-0813 geometry mismatch: {observed!r} != {expected!r}"
        )
    ratios = dsv4_shared_latent_cache.DSV4_PRO0813_TARGET_COMPRESS_RATIOS
    c4 = tuple(layer for layer in range(3, 58) if ratios[layer] == 4)
    c128 = tuple(layer for layer in range(3, 58) if ratios[layer] == 128)
    if len(c4) != 27 or len(c128) != 28:
        raise RuntimeError("Pro-0813 compression topology is not 27-C4/28-C128")
    return {**observed, "c4_layer_ids": c4, "c128_layer_ids": c128}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = tuple(torch.cuda.get_device_capability(device))
    device_name = str(torch.cuda.get_device_name(device))
    if capability != (10, 3) or "B300" not in device_name:
        raise RuntimeError(
            f"this certificate is B300/SM103-only, got {device_name} SM{capability}"
        )
    ptxas = Path(os.environ.get("TRITON_PTXAS_PATH", "")).resolve()
    if ptxas != EXPECTED_PTXAS or not ptxas.is_file():
        raise RuntimeError(
            f"TRITON_PTXAS_PATH must resolve to {EXPECTED_PTXAS}, got {ptxas}"
        )

    geometry = _require_official_geometry()
    if batch_kernels.DSV4_RESTORE_JOBS_PER_REQUEST != EXPECTED_JOBS_PER_REQUEST:
        raise RuntimeError("Pro-0813 restore job workspace is not 219 jobs/request")
    max_requests, max_batch_rows = 3, 65_539
    requirements = dict(
        batch_kernels.dsv4_restore_workspace_requirements(
            max_requests=max_requests, max_batch_rows=max_batch_rows
        )
    )
    if requirements != {
        "max_jobs": EXPECTED_JOBS_PER_REQUEST * max_requests,
        "max_extra_descriptor_columns": 8,
        "max_validation_entries": _expected_validation_entries(
            max_requests, max_batch_rows
        ),
    }:
        raise RuntimeError(f"Pro-0813 workspace capacity mismatch: {requirements}")

    certificate = batch_kernels.run_target_gpu_batch_restore_oracle(device)
    report = dict(certificate.report)
    contract = dict(report.get("pro0813_geometry_contract", {}))
    if not bool(report.get("strict_pass", False)):
        raise AssertionError("shared-latent strict oracle did not pass")
    if not all(
        bool(dict(report[family]).get("torch_reference_pass", False))
        for family in ("packed", "indexer", "state")
    ):
        raise AssertionError("an independent Torch reference did not pass")
    if (
        contract.get("geometry_digest") != profile.PRO0813_TP8_GEOMETRY_DIGEST
        or contract.get("official_config_sha256")
        != profile.PRO0813_OFFICIAL_CONFIG_SHA256
        or contract.get("jobs_per_request") != EXPECTED_JOBS_PER_REQUEST
    ):
        raise AssertionError("certificate is not bound to the Pro-0813 geometry")

    modules = {
        "batch_kernel": batch_kernels,
        "batch_oracle": oracle,
        "deepseek_v4_rope": deepseek_v4_rope,
        "offline_reuse_v2": dsv4_offline_reuse_v2,
        "rope_relocation": dsv4_rope_reloc,
        "shared_latent_cache": dsv4_shared_latent_cache,
        "shared_latent_sglang": dsv4_shared_latent_sglang,
        "pro0813_profile": profile,
    }
    source_paths = {name: _module_path(module) for name, module in modules.items()}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    deployment_oracle = Path(__file__).resolve()
    if PRO_ROOT not in deployment_oracle.parents:
        raise RuntimeError("deployment oracle escaped isolated Pro checkout")
    source_hashes["deployment_oracle"] = _sha256(deployment_oracle)
    if certificate.kernel_source_sha256 != source_hashes["batch_kernel"]:
        raise AssertionError("certificate batch-kernel source hash diverged")
    if certificate.oracle_source_sha256 != source_hashes["batch_oracle"]:
        raise AssertionError("certificate oracle source hash diverged")
    manifest_payload = {
        "schema": "redknot-pro0813-b300-shared-latent-oracle-v1",
        "geometry_digest": profile.PRO0813_TP8_GEOMETRY_DIGEST,
        "source_hashes": source_hashes,
        "certificate_manifest_sha256": certificate.manifest_sha256,
    }
    source_manifest_sha256 = "sha256:" + hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "status": "pass",
                "device": device_name,
                "compute_capability": "sm_103",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "ptxas": str(ptxas),
                "geometry": geometry,
                "workspace_requirements": requirements,
                "source_hashes": source_hashes,
                "source_manifest_sha256": source_manifest_sha256,
                "certificate": certificate.as_dict(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
