# REDKNOT-MODEL: Qwen3.5 layer/head policy; not a registered vLLM runtime.
# Copyright 2024-2026 SGLang RedKnot Integration.
"""Extracted RedKnot Qwen3.5 full/linear layer and full-attention head policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_redknot.core.head_config import HeadClassConfig


def full_attention_layer_indices(config) -> list[int]:
    """Return the indices of ``full_attention`` layers in execution order."""
    tc = getattr(config, "text_config", config)
    layer_types = getattr(tc, "layer_types", None)
    if layer_types is None:
        # Fall back to full_attention_interval if layer_types absent.
        interval = getattr(tc, "full_attention_interval", 1)
        n = tc.num_hidden_layers
        return [i for i in range(n) if (i + 1) % interval == 0]
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


def linear_attention_layer_indices(config) -> list[int]:
    """Return the indices of ``linear_attention`` layers in execution order."""
    tc = getattr(config, "text_config", config)
    layer_types = getattr(tc, "layer_types", None) or []
    return [i for i, t in enumerate(layer_types) if t == "linear_attention"]


def build_full_attention_head_config(
    config,
    *,
    frac_global: float = 0.10,
    local_window: int = 4096,
    sink_size: int = 4,
    seed: int = 1234,
    global_assign: str = "random",
) -> HeadClassConfig:
    """Build a HeadClassConfig sized for the FULL-ATTENTION layers only.

    The returned config has ``num_layers == number of full_attention layers``;
    callers map their k-th full layer to head config row ``k``.

    ``global_assign`` controls WHICH heads become global (tests the RedKnot
    "shallow=local, deep=global" depth hypothesis on Qwen3.5's full layers):
      * "random" : deterministic random spread (default; matches Llama sweet
                   spot generator).
      * "deep"   : the DEEPEST full layers are made global first
                   (shallow full layers stay local) -> tests the hypothesis.
      * "shallow": inverse control (shallowest full layers global first).
    Here "full-layer depth" is the ordinal among full layers (row 0 = shallowest
    full layer, row n_full-1 = deepest).
    """
    import random

    from vllm_redknot.core.head_config import HeadClassConfig

    tc = getattr(config, "text_config", config)
    n_full = len(full_attention_layer_indices(config))
    H = tc.num_key_value_heads
    total = n_full * H
    n_global = max(1, round(frac_global * total))

    if global_assign == "deep":
        # Fill global heads starting from the deepest full layer downward.
        coords = [(li, h) for li in range(n_full - 1, -1, -1) for h in range(H)]
        global_set = set(coords[:n_global])
    elif global_assign == "shallow":
        coords = [(li, h) for li in range(n_full) for h in range(H)]
        global_set = set(coords[:n_global])
    else:  # "random"
        rng = random.Random(seed)
        coords = [(li, h) for li in range(n_full) for h in range(H)]
        rng.shuffle(coords)
        global_set = set(coords[:n_global])

    head_class: list[list[str]] = []
    head_max_distance: list[list[int]] = []
    for li in range(n_full):
        row_cls, row_dist = [], []
        for h in range(H):
            if (li, h) in global_set:
                row_cls.append("global")
                row_dist.append(-1)
            else:
                row_cls.append("local")
                row_dist.append(local_window)
        head_class.append(row_cls)
        head_max_distance.append(row_dist)

    return HeadClassConfig(
        head_class=head_class,
        head_max_distance=head_max_distance,
        num_layers=n_full,
        num_kv_heads=H,
        default_sink_size=sink_size,
        local_default_window=local_window,
    )
