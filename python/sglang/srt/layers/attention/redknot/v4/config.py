from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
    deepseek_v4_redknot_topology,
)


@dataclass(frozen=True)
class RedKnotV4Config:
    mode: str = "correctness"
    strict_alignment: bool = True
    alignment_tokens: int = 128
    min_cache_tokens: int = 512
    boundary_replay_tokens: int = 128
    abort_cost_ratio: float = 0.85
    materialize_union: bool = True
    reuse_csa: bool = True
    reuse_hca: bool = True
    indexer_state_pre_rope: bool = False
    reuse_window_kv: bool = False
    sparse_moe_enabled: bool = False
    dspark_enabled: bool = False
    cache_format_version: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"correctness", "balanced", "aggressive"}:
            raise ValueError(f"unsupported RedKnot V4 mode: {self.mode}")
        if self.alignment_tokens != 128:
            raise ValueError("RedKnot V4 correctness MVP requires 128-token alignment")
        if self.min_cache_tokens < self.alignment_tokens:
            raise ValueError("min_cache_tokens must be at least one alignment unit")
        if self.boundary_replay_tokens < 128:
            raise ValueError("boundary replay must cover the 128-token SWA window")
        if self.mode == "correctness" and (
            self.sparse_moe_enabled or self.dspark_enabled or self.reuse_window_kv
        ):
            raise ValueError(
                "correctness mode requires dense MoE, D-Spark off, and online Window KV"
            )


@dataclass(frozen=True)
class DeepSeekV4Structure:
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_dim: int
    sliding_window: int
    target_compress_ratios: Tuple[int, ...]
    num_csa_layers: int
    num_hca_layers: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    num_routed_experts: int
    num_experts_per_token: int
    hc_mult: int
    num_dspark_stages: int


def inspect_deepseek_v4_config(config: Any) -> DeepSeekV4Structure:
    topology = deepseek_v4_redknot_topology(config)
    num_layers = topology["num_target_layers"]
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    rope_dim = int(config.qk_rope_head_dim)
    sliding_window = int(
        getattr(config, "sliding_window", None) or getattr(config, "window_size", 0)
    )

    if num_attention_heads != 64 or num_key_value_heads != 1:
        raise ValueError(
            "RedKnot V4 requires 64 logical query heads and one physical KV head"
        )
    if head_dim <= rope_dim or rope_dim != 64:
        raise ValueError(
            f"invalid DeepSeek V4 head dimensions: {head_dim=}, {rope_dim=}"
        )
    if sliding_window != 128:
        raise ValueError(
            f"correctness MVP requires sliding_window=128, got {sliding_window}"
        )

    return DeepSeekV4Structure(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        sliding_window=sliding_window,
        target_compress_ratios=topology["target_compress_ratios"],
        num_csa_layers=topology["num_c4_layers"],
        num_hca_layers=topology["num_c128_layers"],
        index_n_heads=int(config.index_n_heads),
        index_head_dim=int(config.index_head_dim),
        index_topk=int(config.index_topk),
        num_routed_experts=int(config.n_routed_experts),
        num_experts_per_token=int(config.num_experts_per_tok),
        hc_mult=int(config.hc_mult),
        num_dspark_stages=topology["num_dspark_stages"],
    )


__all__ = ["DeepSeekV4Structure", "RedKnotV4Config", "inspect_deepseek_v4_config"]
