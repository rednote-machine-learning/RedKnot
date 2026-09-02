from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class DSparkConfig:
    block_size: int
    noise_token: int
    target_layer_ids: Tuple[int, ...]
    markov_rank: int
    num_target_layers: int
    hidden_size: int
    vocab_size: int

    @property
    def num_stages(self) -> int:
        return len(self.target_layer_ids)

    @property
    def num_verify_tokens(self) -> int:
        return self.block_size + 1

    @property
    def target_hidden_size(self) -> int:
        return self.hidden_size * self.num_stages


def parse_dspark_config(config: Any) -> DSparkConfig:
    """Parse and validate the D-Spark contract from a target model config."""

    if getattr(config, "model_type", None) != "deepseek_v4":
        raise ValueError("D-Spark requires a DeepSeek V4 target model")

    block_size = int(getattr(config, "dspark_block_size", 0) or 0)
    noise_token = int(getattr(config, "dspark_noise_token_id", -1))
    target_layer_ids = tuple(
        int(x) for x in (getattr(config, "dspark_target_layer_ids", ()) or ())
    )
    markov_rank = int(getattr(config, "dspark_markov_rank", 0) or 0)
    num_target_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)

    if block_size <= 0:
        raise ValueError(f"D-Spark block size must be positive, got {block_size}")
    if not target_layer_ids:
        raise ValueError("D-Spark requires at least one target hidden-state layer")
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError(f"D-Spark target layers must be unique: {target_layer_ids}")
    if tuple(sorted(target_layer_ids)) != target_layer_ids:
        raise ValueError(
            f"D-Spark target layers must be strictly increasing: {target_layer_ids}"
        )
    invalid_layers = [
        layer_id
        for layer_id in target_layer_ids
        if layer_id < 0 or layer_id >= num_target_layers
    ]
    if invalid_layers:
        raise ValueError(f"D-Spark target layers are out of range: {invalid_layers}")
    if noise_token < 0 or noise_token >= vocab_size:
        raise ValueError(
            f"D-Spark noise token {noise_token} is outside vocabulary size {vocab_size}"
        )
    if markov_rank <= 0:
        raise ValueError(f"D-Spark Markov rank must be positive, got {markov_rank}")
    if hidden_size <= 0:
        raise ValueError(f"D-Spark hidden size must be positive, got {hidden_size}")

    return DSparkConfig(
        block_size=block_size,
        noise_token=noise_token,
        target_layer_ids=target_layer_ids,
        markov_rank=markov_rank,
        num_target_layers=num_target_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )


__all__ = ["DSparkConfig", "parse_dspark_config"]
