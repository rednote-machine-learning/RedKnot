"""Strict geometry contract for the official DeepSeek-V4-Pro-0813 model.

This module intentionally lives outside the Flash-0731 implementation.  The
Pro launcher must validate this contract before selecting any RedKnot kernel or
opening a reusable-state cache.  A future checkpoint with a different shape is
therefore rejected instead of silently inheriting assumptions certified for
Flash-0731 or Pro-0813.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple


PRO0813_VARIANT = "deepseek_v4_pro_0813"
PRO0813_OFFICIAL_CONFIG_SHA256 = (
    "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
)


def _read(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


_EXPECTED_SCALARS = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 1,
    "hidden_size": 7168,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1536,
    "o_lora_rank": 1024,
    "o_groups": 16,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 1024,
    "n_routed_experts": 384,
    "num_experts_per_tok": 6,
    "sliding_window": 128,
    "dspark_markov_rank": 512,
}

_EXPECTED_TARGET_COMPRESS_RATIOS = (128, 128) + tuple(
    4 if layer_id % 2 == 0 else 128 for layer_id in range(2, 61)
)
_EXPECTED_AUXILIARY_COMPRESS_RATIOS = (0, 0, 0)


@dataclass(frozen=True)
class DeepSeekV4Pro0813Geometry:
    """Validated Pro-0813 topology used by the isolated RedKnot path."""

    variant: str
    official_config_sha256: str
    num_target_layers: int
    num_attention_heads: int
    physical_kv_heads: int
    hidden_size: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_routed_experts: int
    num_experts_per_tok: int
    sliding_window: int
    dspark_markov_rank: int
    dense_prefix_layers: int
    dense_suffix_layers: int
    dense_layer_ids: Tuple[int, ...]
    reusable_layer_ids: Tuple[int, ...]
    dspark_target_layer_ids: Tuple[int, ...]
    target_compress_ratios: Tuple[int, ...]
    auxiliary_compress_ratios: Tuple[int, ...]
    tp_size: int
    heads_per_tp_rank: int
    o_groups_per_tp_rank: int

    @property
    def first_reusable_layer(self) -> int:
        return self.reusable_layer_ids[0]

    @property
    def last_reusable_layer(self) -> int:
        return self.reusable_layer_ids[-1]

    def audit_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "official_config_sha256": self.official_config_sha256,
            "num_target_layers": self.num_target_layers,
            "num_attention_heads": self.num_attention_heads,
            "physical_kv_heads": self.physical_kv_heads,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_lora_rank": self.o_lora_rank,
            "o_groups": self.o_groups,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "sliding_window": self.sliding_window,
            "dspark_markov_rank": self.dspark_markov_rank,
            "dense_prefix_layers": self.dense_prefix_layers,
            "dense_suffix_layers": self.dense_suffix_layers,
            "dense_layer_ids": self.dense_layer_ids,
            "reusable_layer_ids": self.reusable_layer_ids,
            "dspark_target_layer_ids": self.dspark_target_layer_ids,
            "target_compress_ratios": self.target_compress_ratios,
            "auxiliary_compress_ratios": self.auxiliary_compress_ratios,
            "tp_size": self.tp_size,
            "heads_per_tp_rank": self.heads_per_tp_rank,
            "o_groups_per_tp_rank": self.o_groups_per_tp_rank,
        }

    @property
    def geometry_digest(self) -> str:
        payload = json.dumps(
            self.audit_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _official_tp8_geometry() -> DeepSeekV4Pro0813Geometry:
    num_layers = int(_EXPECTED_SCALARS["num_hidden_layers"])
    dense_prefix_layers = 3
    dense_suffix_layers = 3
    reusable_layers = tuple(
        range(dense_prefix_layers, num_layers - dense_suffix_layers)
    )
    dense_layers = tuple(range(dense_prefix_layers)) + tuple(
        range(num_layers - dense_suffix_layers, num_layers)
    )
    return DeepSeekV4Pro0813Geometry(
        variant=PRO0813_VARIANT,
        official_config_sha256=PRO0813_OFFICIAL_CONFIG_SHA256,
        num_target_layers=num_layers,
        num_attention_heads=int(_EXPECTED_SCALARS["num_attention_heads"]),
        physical_kv_heads=int(_EXPECTED_SCALARS["num_key_value_heads"]),
        hidden_size=int(_EXPECTED_SCALARS["hidden_size"]),
        head_dim=int(_EXPECTED_SCALARS["head_dim"]),
        qk_rope_head_dim=int(_EXPECTED_SCALARS["qk_rope_head_dim"]),
        q_lora_rank=int(_EXPECTED_SCALARS["q_lora_rank"]),
        o_lora_rank=int(_EXPECTED_SCALARS["o_lora_rank"]),
        o_groups=int(_EXPECTED_SCALARS["o_groups"]),
        index_n_heads=int(_EXPECTED_SCALARS["index_n_heads"]),
        index_head_dim=int(_EXPECTED_SCALARS["index_head_dim"]),
        index_topk=int(_EXPECTED_SCALARS["index_topk"]),
        n_routed_experts=int(_EXPECTED_SCALARS["n_routed_experts"]),
        num_experts_per_tok=int(_EXPECTED_SCALARS["num_experts_per_tok"]),
        sliding_window=int(_EXPECTED_SCALARS["sliding_window"]),
        dspark_markov_rank=int(_EXPECTED_SCALARS["dspark_markov_rank"]),
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        dense_layer_ids=dense_layers,
        reusable_layer_ids=reusable_layers,
        dspark_target_layer_ids=(58, 59, 60),
        target_compress_ratios=_EXPECTED_TARGET_COMPRESS_RATIOS,
        auxiliary_compress_ratios=_EXPECTED_AUXILIARY_COMPRESS_RATIOS,
        tp_size=8,
        heads_per_tp_rank=16,
        o_groups_per_tp_rank=2,
    )


PRO0813_TP8_GEOMETRY = _official_tp8_geometry()
PRO0813_TP8_GEOMETRY_DIGEST = PRO0813_TP8_GEOMETRY.geometry_digest


def inspect_pro0813_config(
    config: Any,
    *,
    tp_size: int = 8,
    dense_prefix_layers: int = 3,
    dense_suffix_layers: int = 3,
) -> DeepSeekV4Pro0813Geometry:
    """Validate an official Pro-0813 config and derive RedKnot layer domains."""

    mismatches = []
    for name, expected in _EXPECTED_SCALARS.items():
        observed = _read(config, name)
        if observed != expected:
            mismatches.append(f"{name}={observed!r} (expected {expected!r})")
    if mismatches:
        raise ValueError(
            "checkpoint does not match DeepSeek-V4-Pro-0813: "
            + "; ".join(mismatches)
        )

    tp_size = int(tp_size)
    if tp_size != 8:
        raise ValueError(
            "Pro-0813 B300/SM103 certification requires TP=8, "
            f"got TP={tp_size}"
        )
    num_heads = int(_EXPECTED_SCALARS["num_attention_heads"])
    o_groups = int(_EXPECTED_SCALARS["o_groups"])
    if num_heads % tp_size:
        raise ValueError(
            f"Pro-0813 attention heads ({num_heads}) are not divisible by TP={tp_size}"
        )
    if o_groups % tp_size:
        raise ValueError(
            f"Pro-0813 output groups ({o_groups}) are not divisible by TP={tp_size}"
        )

    num_layers = int(_EXPECTED_SCALARS["num_hidden_layers"])
    dense_prefix_layers = int(dense_prefix_layers)
    dense_suffix_layers = int(dense_suffix_layers)
    if (dense_prefix_layers, dense_suffix_layers) != (3, 3):
        raise ValueError(
            "Pro-0813 certification requires dense prefix/suffix layers 3/3"
        )

    dspark_target_layers = tuple(
        int(value) for value in (_read(config, "dspark_target_layer_ids", ()) or ())
    )
    if dspark_target_layers != (58, 59, 60):
        raise ValueError(
            "Pro-0813 D-Spark target layers must be (58, 59, 60), got "
            f"{dspark_target_layers}"
        )

    raw_ratios = tuple(
        int(value) for value in (_read(config, "compress_ratios", ()) or ())
    )
    if not raw_ratios:
        layer_types = tuple(_read(config, "layer_types", ()) or ())
        compress_rates = dict(_read(config, "compress_rates", {}) or {})
        type_to_ratio = {
            "sliding_attention": 0,
            "compressed_sparse_attention": int(
                compress_rates.get("compressed_sparse_attention", 4)
            ),
            "heavily_compressed_attention": int(
                compress_rates.get("heavily_compressed_attention", 128)
            ),
        }
        unknown = sorted(set(layer_types) - set(type_to_ratio))
        if unknown:
            raise ValueError(f"unsupported Pro-0813 layer types: {unknown}")
        raw_ratios = tuple(type_to_ratio[value] for value in layer_types)
    if len(raw_ratios) < num_layers:
        raise ValueError(
            f"Pro-0813 needs at least {num_layers} compression ratios, got "
            f"{len(raw_ratios)}"
        )
    target_ratios = raw_ratios[:num_layers]
    if target_ratios != _EXPECTED_TARGET_COMPRESS_RATIOS:
        raise ValueError(
            "Pro-0813 target compression topology differs from the official "
            "0813 layer order"
        )
    auxiliary_ratios = raw_ratios[num_layers:]
    if auxiliary_ratios and auxiliary_ratios != _EXPECTED_AUXILIARY_COMPRESS_RATIOS:
        raise ValueError(
            "Pro-0813 auxiliary D-Spark compression ratios must be (0, 0, 0), "
            f"got {auxiliary_ratios}"
        )

    return PRO0813_TP8_GEOMETRY


__all__ = [
    "PRO0813_OFFICIAL_CONFIG_SHA256",
    "PRO0813_TP8_GEOMETRY",
    "PRO0813_TP8_GEOMETRY_DIGEST",
    "PRO0813_VARIANT",
    "DeepSeekV4Pro0813Geometry",
    "inspect_pro0813_config",
]
