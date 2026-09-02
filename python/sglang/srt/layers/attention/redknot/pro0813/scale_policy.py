"""Geometry-derived RedKnot sizing policy for DeepSeek-V4-Pro-0813.

The Pro checkpoint is not a uniformly scaled Flash checkpoint.  This module
therefore exposes component-specific ratios and the few dimensionless knobs
that deliberately remain unchanged.  It is CPU-only and may be imported by
preflight tooling before torch or a model checkpoint is available.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


PRO0813_SCALE_POLICY_VERSION = "pro0813_b300_component_scale_v1"

FLASH0731_WEIGHT_BYTES = 166_878_536_440
PRO0813_WEIGHT_BYTES = 892_727_580_904
PRO0813_WEIGHT_SHARDS = 66
TP_SIZE = 8

FLASH0731_LAYERS = 43
FLASH0731_DENSE_LAYERS = 6
FLASH0731_REUSABLE_LAYERS = 37
PRO0813_LAYERS = 61
PRO0813_DENSE_LAYERS = 6
PRO0813_REUSABLE_LAYERS = 55

FLASH0731_ATTENTION_HEADS = 64
PRO0813_ATTENTION_HEADS = 128
FLASH0731_O_GROUPS_PER_TP_RANK = 1
PRO0813_O_GROUPS_PER_TP_RANK = 2
O_LORA_RANK = 1024
ZOFF_BYTES_PER_ELEMENT = 2

FLASH0731_INDEX_TOPK = 512
PRO0813_INDEX_TOPK = 1024
FLASH0731_Q_LORA_RANK = 1024
PRO0813_Q_LORA_RANK = 1536
FLASH0731_ROUTED_EXPERTS = 256
PRO0813_ROUTED_EXPERTS = 384
FLASH0731_MOE_INTERMEDIATE = 2048
PRO0813_MOE_INTERMEDIATE = 3072
NATIVE_EXPERTS_PER_TOKEN = 6
PHYSICAL_MLA_PACKED_ROW_BYTES = 584
FLASH0731_C4_REUSABLE_LAYERS = 18
FLASH0731_C128_REUSABLE_LAYERS = 19
PRO0813_C4_REUSABLE_LAYERS = 27
PRO0813_C128_REUSABLE_LAYERS = 28

# These are reference Flash row budgets, not runtime imports from the Flash
# checkout.  They are token-coverage ratios within each reusable layer.  Model
# bytes and layer count do not supply evidence for changing them, so they are
# retained only as an explicit Pro calibration grid, never geometry-scaled.
FLASH0731_STRONG_ACTIVE_RATIO = 0.15
FLASH0731_STANDARD_ACTIVE_RATIO = 0.20
FLASH0731_DIFFUSE_ACTIVE_RATIO = 0.25

# The official B300 server reports this exact physical capacity on every GPU.
B300_TOTAL_MEMORY_MIB = 275_040
B300_RUNTIME_RESERVE_MIB = 32 * 1024
PRO0813_MIN_FREE_BEFORE_LAUNCH_MIB = 240_000

# z_off is device-resident only through 256K.  440K and 512K use the
# bounded, CPU-authoritative one-layer-at-a-time restore path.
PRO0813_DEVICE_ZOFF_CAP_MIB: Mapping[int, int] = {
    65_536: 16 * 1024,
    131_072: 32 * 1024,
    262_144: 64 * 1024,
    450_560: 0,
    524_288: 0,
}


PRO0813_STRONG_ACTIVE_RATIO = FLASH0731_STRONG_ACTIVE_RATIO
PRO0813_STANDARD_ACTIVE_RATIO = FLASH0731_STANDARD_ACTIVE_RATIO
PRO0813_DIFFUSE_ACTIVE_RATIO = FLASH0731_DIFFUSE_ACTIVE_RATIO

# Layer-index policies are scaled by depth, not by checkpoint bytes.
FLASH0731_PROGRESSIVE_TOPK_SCHEDULE = "0-11:6,12-27:5,28-42:4"
PRO0813_PROGRESSIVE_TOPK_SCHEDULE = "0-15:6,16-39:5,40-60:4"
FLASH0731_TOKEN_SPARSE_DEEP_START = 24
PRO0813_TOKEN_SPARSE_DEEP_START = round(
    FLASH0731_TOKEN_SPARSE_DEEP_START * PRO0813_LAYERS / FLASH0731_LAYERS
)

# These controls are dimensionless or belong to the token/chunk geometry.
# Model byte size is not a sound reason to alter them.  They remain subject to
# the mandatory Pro quality/throughput sweep rather than being called
# geometry-certified.
PRO0813_ADAPTIVE_TOPK_MASS = 0.50
PRO0813_ADAPTIVE_TOPK_BUCKETS = (3, 4, 5, 6)
PRO0813_ADAPTIVE_TOPK_FORMAL_ENABLED = False
PRO0813_QUERY_PROTECTION_TOKENS = 8192
PRO0813_SEGMENT_CAP_RATIO = 0.50
PRO0813_TOKEN_SPARSE_MASS = 0.95
PRO0813_TOKEN_SPARSE_DEEP_MASS = 0.90
PRO0813_TOKEN_SPARSE_MIN_FULL_RATIO = 0.10
PRO0813_TOKEN_SPARSE_MAX_FULL_RATIO = 0.50

# The selected-row runtime represents independently restorable spans with one
# descriptor per checkpoint cell.  The bounded online allocator may consume at
# most ``checkpoint_stride`` tokens per descriptor, so the old global cap of 64
# descriptors could not realize a 20% row budget beyond 128K.  Keep this Pro
# capacity policy separate from the dimensionless row-quality ratios above.
PRO0813_CHECKPOINT_BLOCK_TOKENS = 128
PRO0813_CHECKPOINT_STRIDE_TOKENS = 512
PRO0813_CHECKPOINT_DESCRIPTOR_LIMIT = 256


def pro0813_required_checkpoint_islands(
    target_tokens: int,
    active_ratio: float,
    *,
    checkpoint_stride: int = PRO0813_CHECKPOINT_STRIDE_TOKENS,
) -> int:
    """Conservative descriptor count for one request-global row budget.

    Mandatory boundary/query rows do not consume descriptors, so charging the
    complete requested budget here intentionally overestimates the requirement.
    This makes the CPU preflight independent of a particular prompt while
    guaranteeing that descriptor capacity cannot be the reason for undershoot.
    """

    target_tokens = int(target_tokens)
    checkpoint_stride = int(checkpoint_stride)
    active_ratio = float(active_ratio)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if checkpoint_stride < 512 or checkpoint_stride % 512:
        raise ValueError("checkpoint_stride must be a multiple of 512")
    if not math.isfinite(active_ratio) or not 0.0 < active_ratio < 1.0:
        raise ValueError("active_ratio must be finite and in (0, 1)")
    requested_tokens = math.ceil(target_tokens * active_ratio)
    return math.ceil(requested_tokens / checkpoint_stride)


def pro0813_min_realized_active_ratio(
    target_tokens: int,
    requested_ratio: float,
    *,
    checkpoint_stride: int = PRO0813_CHECKPOINT_STRIDE_TOKENS,
) -> float:
    """Fail-close floor allowing at most one checkpoint cell of undershoot."""

    target_tokens = int(target_tokens)
    checkpoint_stride = int(checkpoint_stride)
    requested_ratio = float(requested_ratio)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if checkpoint_stride <= 0:
        raise ValueError("checkpoint_stride must be positive")
    if not math.isfinite(requested_ratio) or not 0.0 < requested_ratio < 1.0:
        raise ValueError("requested_ratio must be finite and in (0, 1)")
    return max(0.0, requested_ratio - checkpoint_stride / target_tokens)


def pro0813_realized_active_ratio_gate(
    requested_ratio: float,
    min_realized_ratio: float,
    online_row_saving: float | None,
    *,
    required: bool,
) -> dict[str, float | bool | None]:
    """Return the formal row-budget evidence gate used by Pro results.

    ``online_row_saving`` is the runtime's measured inactive-row fraction, so
    the corresponding realized active ratio is ``1 - online_row_saving``.
    Missing, non-finite, or out-of-range runtime evidence fails closed whenever
    the combined row-sparse path is required.
    """

    requested_ratio = float(requested_ratio)
    min_realized_ratio = float(min_realized_ratio)
    if not math.isfinite(requested_ratio) or not 0.0 < requested_ratio < 1.0:
        raise ValueError("requested_ratio must be finite and in (0, 1)")
    if (
        not math.isfinite(min_realized_ratio)
        or not 0.0 < min_realized_ratio <= requested_ratio
    ):
        raise ValueError(
            "min_realized_ratio must be finite and in (0, requested_ratio]"
        )

    actual_ratio: float | None = None
    if online_row_saving is not None:
        candidate = 1.0 - float(online_row_saving)
        if math.isfinite(candidate) and 0.0 <= candidate <= 1.0:
            actual_ratio = candidate
    evidence_pass = actual_ratio is not None and (
        actual_ratio + 1e-12 >= min_realized_ratio
    )
    return {
        "requested_active_ratio": requested_ratio,
        "min_realized_active_ratio": min_realized_ratio,
        "actual_realized_active_ratio": actual_ratio,
        "realized_active_ratio_pass": bool(evidence_pass or not required),
        "realized_active_ratio_required": bool(required),
    }


def _checkpoint_capacity_for_target(target_tokens: int) -> int:
    required = pro0813_required_checkpoint_islands(
        target_tokens, PRO0813_DIFFUSE_ACTIVE_RATIO
    )
    capacity = max(64, 1 << (required - 1).bit_length())
    if capacity > PRO0813_CHECKPOINT_DESCRIPTOR_LIMIT:
        raise ValueError(
            "Pro-0813 row policy exceeds the checkpoint descriptor limit"
        )
    return capacity


PRO0813_CHECKPOINT_MAX_ISLANDS: Mapping[int, int] = {
    target: _checkpoint_capacity_for_target(target)
    for target in PRO0813_DEVICE_ZOFF_CAP_MIB
}


def pro0813_zoff_bytes_per_rank(total_tokens: int) -> int:
    return (
        PRO0813_REUSABLE_LAYERS
        * int(total_tokens)
        * PRO0813_O_GROUPS_PER_TP_RANK
        * O_LORA_RANK
        * ZOFF_BYTES_PER_ELEMENT
    )


def flash0731_zoff_bytes_per_rank(total_tokens: int) -> int:
    return (
        FLASH0731_REUSABLE_LAYERS
        * int(total_tokens)
        * FLASH0731_O_GROUPS_PER_TP_RANK
        * O_LORA_RANK
        * ZOFF_BYTES_PER_ELEMENT
    )


def pro0813_cpu_reservation_bytes_per_rank(total_tokens: int) -> int:
    """Host reservation including token metadata and per-layer validity."""

    return int(total_tokens) * (
        9 + PRO0813_REUSABLE_LAYERS * (4096 + 1)
    )


def pro0813_mem_fraction_static(target_tokens: int) -> float:
    """Target-specific B300 static fraction with a 32-GiB runtime reserve.

    Device z_off is outside SGLang's static allocator.  Round down to two
    decimals so the configured allocation never consumes the reserve.
    CPU-authoritative 440K/512K paths are capped at 0.80 to retain transfer
    and kernel workspace even though the arithmetic ceiling is higher.
    """

    target_tokens = int(target_tokens)
    if target_tokens not in PRO0813_DEVICE_ZOFF_CAP_MIB:
        raise ValueError(f"unsupported Pro-0813 target: {target_tokens}")
    exact_device_mib = (
        pro0813_zoff_bytes_per_rank(target_tokens) // (1024**2)
        if PRO0813_DEVICE_ZOFF_CAP_MIB[target_tokens]
        else 0
    )
    ceiling = (
        B300_TOTAL_MEMORY_MIB
        - B300_RUNTIME_RESERVE_MIB
        - exact_device_mib
    ) / B300_TOTAL_MEMORY_MIB
    return math.floor(min(0.80, ceiling) * 100.0) / 100.0


PRO0813_MEM_FRACTION_STATIC: Mapping[int, float] = {
    target: pro0813_mem_fraction_static(target)
    for target in PRO0813_DEVICE_ZOFF_CAP_MIB
}


def _component_ratios() -> dict[str, float]:
    return {
        "logical_weight_bytes": PRO0813_WEIGHT_BYTES / FLASH0731_WEIGHT_BYTES,
        "layers": PRO0813_LAYERS / FLASH0731_LAYERS,
        "reusable_layers": PRO0813_REUSABLE_LAYERS / FLASH0731_REUSABLE_LAYERS,
        "attention_heads": PRO0813_ATTENTION_HEADS / FLASH0731_ATTENTION_HEADS,
        "o_groups_per_tp_rank": (
            PRO0813_O_GROUPS_PER_TP_RANK / FLASH0731_O_GROUPS_PER_TP_RANK
        ),
        "index_topk": PRO0813_INDEX_TOPK / FLASH0731_INDEX_TOPK,
        "q_lora_rank": PRO0813_Q_LORA_RANK / FLASH0731_Q_LORA_RANK,
        "routed_experts": PRO0813_ROUTED_EXPERTS / FLASH0731_ROUTED_EXPERTS,
        "moe_intermediate": PRO0813_MOE_INTERMEDIATE / FLASH0731_MOE_INTERMEDIATE,
        "zoff_bytes_per_token": (
            PRO0813_REUSABLE_LAYERS * PRO0813_O_GROUPS_PER_TP_RANK
        )
        / (FLASH0731_REUSABLE_LAYERS * FLASH0731_O_GROUPS_PER_TP_RANK),
    }


def scale_policy_audit() -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target, device_cap_mib in PRO0813_DEVICE_ZOFF_CAP_MIB.items():
        zoff_bytes = pro0813_zoff_bytes_per_rank(target)
        cpu_bytes = pro0813_cpu_reservation_bytes_per_rank(target)
        cpu_cap_bytes = {
            65_536: 16 * 1024**3,
            131_072: 32 * 1024**3,
            262_144: 64 * 1024**3,
            450_560: 112 * 1024**3,
            524_288: 128 * 1024**3,
        }[target]
        if cpu_cap_bytes < cpu_bytes:
            raise ValueError("Pro-0813 CPU artifact cap is below reservation")
        targets[str(target)] = {
            "zoff_bytes_per_rank": zoff_bytes,
            "cpu_reservation_bytes_per_rank": cpu_bytes,
            "cpu_cap_bytes_per_rank": cpu_cap_bytes,
            "cpu_cap_headroom_bytes": cpu_cap_bytes - cpu_bytes,
            "device_cap_bytes_per_rank": device_cap_mib * 1024**2,
            "storage_mode": (
                "device_resident"
                if device_cap_mib
                else "cpu_authoritative_layer_stream"
            ),
            "mem_fraction_static": PRO0813_MEM_FRACTION_STATIC[target],
            "checkpoint_stride_tokens": PRO0813_CHECKPOINT_STRIDE_TOKENS,
            "checkpoint_max_islands": PRO0813_CHECKPOINT_MAX_ISLANDS[target],
            "checkpoint_capacity_tokens": (
                PRO0813_CHECKPOINT_MAX_ISLANDS[target]
                * PRO0813_CHECKPOINT_STRIDE_TOKENS
            ),
            "checkpoint_islands_required_standard": (
                pro0813_required_checkpoint_islands(
                    target, PRO0813_STANDARD_ACTIVE_RATIO
                )
            ),
            "checkpoint_islands_required_diffuse": (
                pro0813_required_checkpoint_islands(
                    target, PRO0813_DIFFUSE_ACTIVE_RATIO
                )
            ),
            "min_realized_standard_active_ratio": (
                pro0813_min_realized_active_ratio(
                    target, PRO0813_STANDARD_ACTIVE_RATIO
                )
            ),
        }
    return {
        "version": PRO0813_SCALE_POLICY_VERSION,
        "calibration_status": (
            "geometry_sized; dimensionless_quality_knobs_require_pro_measurement"
        ),
        "weights": {
            "flash0731_bytes": FLASH0731_WEIGHT_BYTES,
            "pro0813_bytes": PRO0813_WEIGHT_BYTES,
            "pro0813_shards": PRO0813_WEIGHT_SHARDS,
            "pro0813_ideal_tp8_bytes_per_rank": math.ceil(
                PRO0813_WEIGHT_BYTES / TP_SIZE
            ),
        },
        "component_ratios_pro_over_flash": _component_ratios(),
        "component_geometry": {
            "physical_mla": {
                "flash0731_packed_row_bytes": PHYSICAL_MLA_PACKED_ROW_BYTES,
                "pro0813_packed_row_bytes": PHYSICAL_MLA_PACKED_ROW_BYTES,
                "row_width_ratio": 1.0,
                "note": (
                    "physical KV heads/head_dim/RoPE width are unchanged; "
                    "logical attention-head count must not scale this row"
                ),
            },
            "compressed_reusable_rows_per_input_token": {
                "flash0731": (
                    FLASH0731_C4_REUSABLE_LAYERS / 4
                    + FLASH0731_C128_REUSABLE_LAYERS / 128
                ),
                "pro0813": (
                    PRO0813_C4_REUSABLE_LAYERS / 4
                    + PRO0813_C128_REUSABLE_LAYERS / 128
                ),
                "ratio": (
                    PRO0813_C4_REUSABLE_LAYERS / 4
                    + PRO0813_C128_REUSABLE_LAYERS / 128
                )
                / (
                    FLASH0731_C4_REUSABLE_LAYERS / 4
                    + FLASH0731_C128_REUSABLE_LAYERS / 128
                ),
            },
            "indexer": {
                "persistent_c4_layer_ratio": (
                    PRO0813_C4_REUSABLE_LAYERS
                    / FLASH0731_C4_REUSABLE_LAYERS
                ),
                "online_topk_ratio": PRO0813_INDEX_TOPK / FLASH0731_INDEX_TOPK,
            },
            "moe": {
                "activated_experts_per_token_flash0731": NATIVE_EXPERTS_PER_TOKEN,
                "activated_experts_per_token_pro0813": NATIVE_EXPERTS_PER_TOKEN,
                "activated_expert_count_ratio": 1.0,
                "expert_intermediate_width_ratio": (
                    PRO0813_MOE_INTERMEDIATE / FLASH0731_MOE_INTERMEDIATE
                ),
            },
        },
        "layer_domains": {
            "flash0731": {
                "total": FLASH0731_LAYERS,
                "dense": FLASH0731_DENSE_LAYERS,
                "reusable": FLASH0731_REUSABLE_LAYERS,
            },
            "pro0813": {
                "total": PRO0813_LAYERS,
                "dense": PRO0813_DENSE_LAYERS,
                "reusable": PRO0813_REUSABLE_LAYERS,
            },
        },
        "row_policy": {
            "derivation": (
                "dimensionless_per_layer_token_coverage_not_geometry_scaled"
            ),
            "flash_reference": {
                "strong": FLASH0731_STRONG_ACTIVE_RATIO,
                "standard": FLASH0731_STANDARD_ACTIVE_RATIO,
                "diffuse": FLASH0731_DIFFUSE_ACTIVE_RATIO,
            },
            "pro0813": {
                "strong": PRO0813_STRONG_ACTIVE_RATIO,
                "standard": PRO0813_STANDARD_ACTIVE_RATIO,
                "diffuse": PRO0813_DIFFUSE_ACTIVE_RATIO,
            },
            "query_protection_tokens": PRO0813_QUERY_PROTECTION_TOKENS,
            "segment_cap_ratio": PRO0813_SEGMENT_CAP_RATIO,
            "checkpoint_descriptor_limit": PRO0813_CHECKPOINT_DESCRIPTOR_LIMIT,
            "checkpoint_capacity_derivation": (
                "next_power_of_two_at_least_diffuse_ratio_times_target_over_512"
            ),
            "realized_ratio_gate": (
                "requested_ratio_minus_at_most_one_512_token_checkpoint_cell"
            ),
            "calibration_status": "candidate_grid_requires_pro_measurement",
        },
        "moe_policy": {
            "native_experts_per_token": NATIVE_EXPERTS_PER_TOKEN,
            "progressive_topk_schedule": PRO0813_PROGRESSIVE_TOPK_SCHEDULE,
            "adaptive_mass": PRO0813_ADAPTIVE_TOPK_MASS,
            "adaptive_buckets": list(PRO0813_ADAPTIVE_TOPK_BUCKETS),
            "adaptive_formal_enabled": PRO0813_ADAPTIVE_TOPK_FORMAL_ENABLED,
            "adaptive_calibration_status": "requires_pro_router_histogram",
            "token_sparse_deep_start": PRO0813_TOKEN_SPARSE_DEEP_START,
            "dimensionless_controls_preserved_for_measurement": {
                "token_sparse_mass": PRO0813_TOKEN_SPARSE_MASS,
                "token_sparse_deep_mass": PRO0813_TOKEN_SPARSE_DEEP_MASS,
                "token_sparse_min_full_ratio": PRO0813_TOKEN_SPARSE_MIN_FULL_RATIO,
                "token_sparse_max_full_ratio": PRO0813_TOKEN_SPARSE_MAX_FULL_RATIO,
            },
        },
        "b300": {
            "total_memory_mib": B300_TOTAL_MEMORY_MIB,
            "runtime_reserve_mib": B300_RUNTIME_RESERVE_MIB,
            "min_free_before_launch_mib": PRO0813_MIN_FREE_BEFORE_LAUNCH_MIB,
        },
        "zoff": {
            "flash0731_bytes_per_token_per_rank": (
                FLASH0731_REUSABLE_LAYERS
                * FLASH0731_O_GROUPS_PER_TP_RANK
                * O_LORA_RANK
                * ZOFF_BYTES_PER_ELEMENT
            ),
            "pro0813_bytes_per_token_per_rank": (
                PRO0813_REUSABLE_LAYERS
                * PRO0813_O_GROUPS_PER_TP_RANK
                * O_LORA_RANK
                * ZOFF_BYTES_PER_ELEMENT
            ),
            "pro_over_flash_ratio": 110 / 37,
            "formula": "55 * tokens * 2 * 1024 * 2",
            "cpu_reservation_formula": "tokens * (9 + 55 * (4096 + 1))",
        },
        "targets": targets,
    }


def scale_policy_digest() -> str:
    payload = json.dumps(
        scale_policy_audit(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


PRO0813_SCALE_POLICY_DIGEST = scale_policy_digest()


__all__ = [name for name in globals() if name.startswith("PRO0813_")] + [
    "flash0731_zoff_bytes_per_rank",
    "pro0813_cpu_reservation_bytes_per_rank",
    "pro0813_min_realized_active_ratio",
    "pro0813_realized_active_ratio_gate",
    "pro0813_mem_fraction_static",
    "pro0813_required_checkpoint_islands",
    "pro0813_zoff_bytes_per_rank",
    "scale_policy_audit",
    "scale_policy_digest",
]
