from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class LayerAttentionType(str, Enum):
    SWA = "swa"
    CSA = "csa"
    HCA = "hca"


class StateKind(str, Enum):
    WINDOW_KV = "window_kv"
    COMPRESSED_KV = "compressed_kv"
    INDEXER_KV = "indexer_kv"
    COMPRESSOR_TAIL_KV = "compressor_tail_kv"
    COMPRESSOR_TAIL_SCORE = "compressor_tail_score"


class StateSource(str, Enum):
    OFFLINE = "offline"
    ONLINE_PREFIX = "online_prefix"
    ONLINE_BOUNDARY = "online_boundary"
    ONLINE_REFRESH = "online_refresh"


class ReuseAction(str, Enum):
    FULL_REUSE = "full_reuse"
    BOUNDARY_REPLAY = "boundary_replay"
    SELECTIVE_REFRESH = "selective_refresh"
    FULL_RECOMPUTE = "full_recompute"


class FallbackReason(str, Enum):
    CACHE_MISS = "cache_miss"
    INCOMPATIBLE_MODEL = "incompatible_model"
    TOKEN_MISMATCH = "token_mismatch"
    PHASE_MISMATCH = "phase_mismatch"
    INVALID_STATE = "invalid_state"
    COST_NOT_BENEFICIAL = "cost_not_beneficial"
    QUALITY_GATE = "quality_gate"
    UNSUPPORTED_KERNEL = "unsupported_kernel"
    OOM = "oom"
    DSPARK_CONFLICT = "dspark_conflict"


@dataclass(frozen=True)
class CacheCompatibilityKey:
    model_id: str
    model_revision: str
    tokenizer_hash: str
    encoding_hash: str
    weight_dtype: str
    kv_dtype: str
    expert_dtype: str
    rope_config_hash: str
    compress_ratio_hash: str
    cache_format_version: int


@dataclass(frozen=True)
class RuntimePlanValidation:
    valid: bool
    fallback_reason: Optional[FallbackReason] = None
    detail: str = ""


@dataclass(frozen=True)
class BlockValidity:
    logical_block: int
    offline_present: bool
    offline_valid: bool
    online_present: bool
    source: StateSource
    reason: str


@dataclass(frozen=True)
class PositionTransform:
    source_anchor: int
    target_anchor: int
    rope_dim: int
    compress_ratio: int
    is_pre_rope_state: bool

    @property
    def phase_compatible(self) -> bool:
        return (
            self.source_anchor % self.compress_ratio
            == self.target_anchor % self.compress_ratio
        )


TokenRange = Tuple[int, int]


__all__ = [
    "BlockValidity",
    "CacheCompatibilityKey",
    "FallbackReason",
    "LayerAttentionType",
    "PositionTransform",
    "ReuseAction",
    "RuntimePlanValidation",
    "StateKind",
    "StateSource",
    "TokenRange",
]
