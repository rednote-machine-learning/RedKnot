from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from sglang.srt.layers.attention.redknot.v4.types import CacheCompatibilityKey


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_compatibility_key(
    *,
    model_path: str,
    model_revision: str,
    config: Any,
    weight_dtype: str,
    kv_dtype: str,
    expert_dtype: str,
    cache_format_version: int = 1,
) -> CacheCompatibilityKey:
    root = Path(model_path)
    tokenizer_path = root / "tokenizer.json"
    encoding_path = root / "encoding" / "encoding_dsv4.py"
    if not tokenizer_path.is_file() or not encoding_path.is_file():
        raise FileNotFoundError(
            "RedKnot V4 cache keys require tokenizer.json and encoding/encoding_dsv4.py"
        )

    rope_config = {
        "rope_theta": getattr(config, "rope_theta", None),
        "compress_rope_theta": getattr(config, "compress_rope_theta", None),
        "rope_scaling": getattr(config, "rope_scaling", None),
        "rope_parameters": getattr(config, "rope_parameters", None),
        "qk_rope_head_dim": getattr(config, "qk_rope_head_dim", None),
    }
    compress_config = {
        "compress_ratios": getattr(config, "compress_ratios", None),
        "layer_types": getattr(config, "layer_types", None),
        "compress_rates": getattr(config, "compress_rates", None),
    }
    return CacheCompatibilityKey(
        model_id=str(root.resolve()),
        model_revision=str(model_revision),
        tokenizer_hash=_file_hash(tokenizer_path),
        encoding_hash=_file_hash(encoding_path),
        weight_dtype=str(weight_dtype),
        kv_dtype=str(kv_dtype),
        expert_dtype=str(expert_dtype),
        rope_config_hash=_stable_hash(rope_config),
        compress_ratio_hash=_stable_hash(compress_config),
        cache_format_version=int(cache_format_version),
    )


def compute_chunk_id(
    token_ids: Iterable[int], compatibility: CacheCompatibilityKey
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            asdict(compatibility),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for token in token_ids:
        digest.update(struct.pack("<q", int(token)))
    return digest.hexdigest()


__all__ = ["build_cache_compatibility_key", "compute_chunk_id"]
