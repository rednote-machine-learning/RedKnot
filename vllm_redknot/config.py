"""Validated, explicit opt-in settings; no framework import is needed."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

VLLM_COMMIT = "e52be1a62d3879b1202f4f355d3c3472b560c6f2"
REDKNOT_COMMIT = "55ee4e8401603f8d2612877e4053e18b37b1c1bd"
ENGINE_FAMILIES = frozenset({"mha", "deepseek_v4_flash"})
_ALLOWED = {
    "schema_version",
    "engine_family",
    "enabled",
    "allow_approximate",
    "local_heads",
    "boundary_tokens",
    "max_cache_bytes",
    "rope_theta",
    "rotary_dim",
    "supported_vllm_commit",
    "model_revision",
}


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _finite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("RedKnot configuration must be a JSON object")
    unknown = set(raw) - _ALLOWED
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = {
        "schema_version": 1,
        "engine_family": "mha",
        "enabled": True,
        "allow_approximate": False,
        "boundary_tokens": 128,
        "max_cache_bytes": 2 * 1024**3,
        "rope_theta": 10000.0,
        "rotary_dim": None,
        "supported_vllm_commit": VLLM_COMMIT,
        "local_heads": {},
        **raw,
    }
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    family = config["engine_family"]
    if not isinstance(family, str) or family not in ENGINE_FAMILIES:
        raise ValueError("engine_family must be mha or deepseek_v4_flash")
    for field in ("enabled", "allow_approximate"):
        if type(config[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    for field in ("boundary_tokens", "max_cache_bytes"):
        if type(config[field]) is not int or config[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if family == "mha":
        theta = config["rope_theta"]
        if isinstance(theta, bool) or not isinstance(theta, (float, int)):
            raise ValueError("rope_theta must be finite and greater than zero")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("rope_theta must be finite and greater than zero")
        config["rope_theta"] = float(theta)
        rotary_dim = config["rotary_dim"]
        if rotary_dim is not None and (
            type(rotary_dim) is not int or rotary_dim <= 0 or rotary_dim % 2
        ):
            raise ValueError("rotary_dim must be null or a positive even integer")
    else:
        # These are unused compatibility fields in RedKnotSettings. DSV4 uses
        # each native layer's real cos_sin_cache, including compressed YaRN.
        if config["rope_theta"] != 10000.0 or config["rotary_dim"] is not None:
            raise ValueError(
                "DSV4 uses native per-layer RoPE; rope_theta/rotary_dim overrides "
                "are unsupported (omit them, or retain unused defaults 10000/null)"
            )
        config["rope_theta"] = 10000.0
    if config["supported_vllm_commit"] != VLLM_COMMIT:
        raise ValueError(f"This adapter is pinned to vLLM {VLLM_COMMIT}")
    revision = config.get("model_revision")
    if config["enabled"] and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("Enabled reuse requires an explicit model_revision")
    if revision is not None and (not isinstance(revision, str) or len(revision) > 1024):
        raise ValueError("model_revision must be a string of at most 1024 characters")
    heads = config["local_heads"]
    head_kind = "KV" if family == "mha" else "logical query"
    if not isinstance(heads, dict):
        raise ValueError(
            f"local_heads must map layer indices/names to {head_kind} head IDs"
        )
    normalized = {}
    for layer, ids in heads.items():
        if not isinstance(layer, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", layer):
            raise ValueError("Layer keys must be numeric strings or dotted names")
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError(
                f"Local {head_kind} head IDs must be nonnegative integer lists"
            )
        if len(ids) != len(set(ids)):
            raise ValueError(f"Repeated {head_kind} head ID in layer {layer}")
        normalized[layer] = sorted(ids)
    if config["enabled"] and not any(normalized.values()):
        raise ValueError("Enabled RedKnot needs at least one explicit local-head layer")
    config["local_heads"] = normalized
    return config


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Load an explicit file, or the opt-in environment variable; unset is inert."""
    selected = path if path is not None else os.environ.get("VLLM_REDKNOT_CONFIG")
    if not selected:
        return None
    candidate = Path(selected)
    if candidate.stat().st_size > 1024**2:
        raise ValueError("Configuration exceeds the 1 MiB limit")
    raw = json.loads(
        candidate.read_text(encoding="utf-8"),
        object_pairs_hook=_object,
        parse_constant=_finite_json,
    )
    return validate_config(raw)


def policy_fingerprint(config: dict[str, Any]) -> str:
    normalized = validate_config(config)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def import_sglang_head_policy(
    data: dict[str, Any], *, model_revision: str, rope_theta: float
) -> dict[str, Any]:
    """Import only head identities, not unvalidated windows/FFN/MLA semantics."""
    matrix = data.get("kv_head_classification")
    layers, heads = data.get("num_layers"), data.get("num_kv_heads")
    if type(layers) is not int or type(heads) is not int or layers <= 0 or heads <= 0:
        raise ValueError("Expected an MHA/GQA num_layers and num_kv_heads policy")
    if not isinstance(matrix, list) or len(matrix) != layers:
        raise ValueError("Head-policy layer count mismatch")
    local = {}
    dense_prefix = data.get("dense_prefix_layers", 0)
    if type(dense_prefix) is not int or not 0 <= dense_prefix <= layers:
        raise ValueError("Invalid dense_prefix_layers")
    for index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != heads:
            raise ValueError(f"Head-policy width mismatch in layer {index}")
        if any(value not in {"local", "global", "retrieval", "dense"} for value in row):
            raise ValueError("Unrecognized head class; legacy aliases need review")
        ids = [head for head, value in enumerate(row) if value == "local"]
        if index >= dense_prefix and ids:
            local[str(index)] = ids
    return validate_config(
        {
            "model_revision": model_revision,
            "rope_theta": rope_theta,
            "local_heads": local,
            "allow_approximate": False,
        }
    )
