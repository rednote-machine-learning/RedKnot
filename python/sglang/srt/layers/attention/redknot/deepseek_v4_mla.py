# Copyright 2024-2026 SGLang RedKnot Integration.
"""DeepSeek V4 MLA head classification helpers for RedKnot.

DeepSeek V4 keeps the physical KV cache in MLA form: one latent KV stream per
layer, packed as nope/rope/scale bytes for FlashMLA. RedKnot's original
``HeadClassConfig`` classifies ``num_key_value_heads``; for MLA this would be a
single head and would lose the useful per-attention-head distinction. This file
adds a lightweight logical-head config for DeepSeek V4 where classification is
over the decompressed attention heads (``num_attention_heads``), while metadata
explicitly records that the stored cache still has one latent KV head.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.attention.redknot.head_config import (
    DEFAULT_LOCAL_WINDOW,
    DEFAULT_SINK_SIZE,
    HEAD_DENSE,
    HEAD_GLOBAL,
    HEAD_LOCAL,
)


@dataclass(frozen=True)
class MLAHeadStrategy:
    """Per-(layer, logical attention head) DeepSeek V4 MLA strategy."""

    layer: int
    logical_head: int
    head_type: str
    window: int
    sink_size: int

    def is_local(self) -> bool:
        return self.head_type == HEAD_LOCAL

    def is_global(self) -> bool:
        return self.head_type == HEAD_GLOBAL

    def is_dense(self) -> bool:
        return self.head_type == HEAD_DENSE


class DeepSeekV4MLAHeadConfig:
    """Head policy for MLA logical attention heads.

    ``physical_kv_heads`` is expected to stay 1 for DeepSeek V4. The matrix
    dimensions are ``[num_layers][num_attention_heads]`` because sparsity is
    chosen after the MLA latent is projected into per-head Q/K logits.
    """

    def __init__(
        self,
        *,
        head_class: List[List[str]],
        head_max_distance: List[List[int]],
        num_layers: int,
        num_attention_heads: int,
        physical_kv_heads: int = 1,
        head_sink_size: Optional[List[List[int]]] = None,
        default_sink_size: int = DEFAULT_SINK_SIZE,
        local_default_window: int = DEFAULT_LOCAL_WINDOW,
        dense_prefix_layers: int = 0,
    ) -> None:
        if len(head_class) != num_layers:
            raise ValueError(
                f"head_class layers={len(head_class)} != num_layers={num_layers}"
            )
        if head_class and len(head_class[0]) != num_attention_heads:
            raise ValueError(
                "head_class heads="
                f"{len(head_class[0])} != num_attention_heads={num_attention_heads}"
            )
        if len(head_max_distance) != num_layers:
            raise ValueError(
                "head_max_distance layers="
                f"{len(head_max_distance)} != num_layers={num_layers}"
            )
        self.head_class = head_class
        self.head_max_distance = head_max_distance
        self.head_sink_size = head_sink_size
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.physical_kv_heads = int(physical_kv_heads)
        self.default_sink_size = int(default_sink_size)
        self.local_default_window = int(local_default_window)
        self.dense_prefix_layers = int(dense_prefix_layers)
        self._cached_tensors: Dict[torch.device, Dict[str, torch.Tensor]] = {}
        self._normalize()

    @classmethod
    def from_model_config(
        cls,
        config: Any,
        *,
        dense_prefix_layers: int = 2,
        local_window: int = 4096,
        sink_size: int = DEFAULT_SINK_SIZE,
        global_head_stride: int = 8,
        global_layer_stride: int = 0,
    ) -> "DeepSeekV4MLAHeadConfig":
        """Build a conservative default before offline head profiling exists.

        Every ``global_head_stride``-th logical head is global. Optionally, every
        ``global_layer_stride``-th layer can be all-global. The first
        ``dense_prefix_layers`` are dense/full to preserve early-layer signal.
        """

        if not is_deepseek_v4_mla_config(config):
            raise ValueError("config is not DeepSeek V4 MLA")
        n_layers = int(config.num_hidden_layers)
        n_heads = int(config.num_attention_heads)
        physical_kv_heads = int(getattr(config, "num_key_value_heads", 1))
        global_head_stride = max(1, int(global_head_stride))
        global_layer_stride = int(global_layer_stride)

        head_class: List[List[str]] = []
        head_distance: List[List[int]] = []
        sinks: List[List[int]] = []
        for layer in range(n_layers):
            force_full_layer = layer < dense_prefix_layers or (
                global_layer_stride > 0 and layer % global_layer_stride == 0
            )
            row, dist = [], []
            for head in range(n_heads):
                if force_full_layer:
                    htype = HEAD_DENSE if layer < dense_prefix_layers else HEAD_GLOBAL
                    row.append(htype)
                    dist.append(-1)
                elif head % global_head_stride == 0:
                    row.append(HEAD_GLOBAL)
                    dist.append(-1)
                else:
                    row.append(HEAD_LOCAL)
                    dist.append(int(local_window))
            head_class.append(row)
            head_distance.append(dist)
            sinks.append([int(sink_size)] * n_heads)

        return cls(
            head_class=head_class,
            head_max_distance=head_distance,
            head_sink_size=sinks,
            num_layers=n_layers,
            num_attention_heads=n_heads,
            physical_kv_heads=physical_kv_heads,
            default_sink_size=sink_size,
            local_default_window=local_window,
            dense_prefix_layers=dense_prefix_layers,
        )

    @classmethod
    def from_json(cls, json_path: str) -> "DeepSeekV4MLAHeadConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            head_class=data["mla_head_classification"],
            head_max_distance=data["mla_head_max_distance"],
            head_sink_size=data.get("mla_head_sink_size"),
            num_layers=data["num_layers"],
            num_attention_heads=data["num_attention_heads"],
            physical_kv_heads=data.get("physical_kv_heads", 1),
            default_sink_size=data.get("default_sink_size", DEFAULT_SINK_SIZE),
            local_default_window=data.get("local_default_window", DEFAULT_LOCAL_WINDOW),
            dense_prefix_layers=data.get("dense_prefix_layers", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "redknot_deepseek_v4_mla_head_config_v1",
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "physical_kv_heads": self.physical_kv_heads,
            "dense_prefix_layers": self.dense_prefix_layers,
            "default_sink_size": self.default_sink_size,
            "local_default_window": self.local_default_window,
            "mla_head_classification": self.head_class,
            "mla_head_max_distance": self.head_max_distance,
            "mla_head_sink_size": self.head_sink_size,
        }

    def to_json(self, json_path: str) -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def get_strategy(self, layer: int, logical_head: int) -> MLAHeadStrategy:
        if layer < self.dense_prefix_layers:
            return MLAHeadStrategy(
                layer=layer,
                logical_head=logical_head,
                head_type=HEAD_DENSE,
                window=-1,
                sink_size=self._sink_at(layer, logical_head),
            )
        htype = self.head_class[layer][logical_head]
        max_distance = self.head_max_distance[layer][logical_head]
        window = max_distance if htype == HEAD_LOCAL and max_distance > 0 else -1
        if htype == HEAD_LOCAL and window <= 0:
            window = self.local_default_window
        return MLAHeadStrategy(
            layer=layer,
            logical_head=logical_head,
            head_type=htype,
            window=window,
            sink_size=self._sink_at(layer, logical_head),
        )

    def set_local_window(self, new_window: int) -> int:
        changed = 0
        for layer in range(self.num_layers):
            for head in range(self.num_attention_heads):
                if self.head_class[layer][head] == HEAD_LOCAL:
                    self.head_max_distance[layer][head] = int(new_window)
                    changed += 1
        self.local_default_window = int(new_window)
        self._cached_tensors.clear()
        return changed

    def layer_tensors(
        self, layer: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        cached = self._cached_tensors.setdefault(device, {})
        key = f"layer:{layer}"
        if key in cached:
            return cached[key]
        type_ids = []
        windows = []
        sinks = []
        for head in range(self.num_attention_heads):
            strategy = self.get_strategy(layer, head)
            if strategy.head_type == HEAD_LOCAL:
                type_ids.append(1)
            elif strategy.head_type == HEAD_GLOBAL:
                type_ids.append(2)
            else:
                type_ids.append(3)
            windows.append(strategy.window)
            sinks.append(strategy.sink_size)
        out = {
            "type_ids": torch.tensor(type_ids, dtype=torch.int32, device=device),
            "windows": torch.tensor(windows, dtype=torch.int32, device=device),
            "sinks": torch.tensor(sinks, dtype=torch.int32, device=device),
            "is_local": torch.tensor(
                [x == 1 for x in type_ids], dtype=torch.bool, device=device
            ),
        }
        cached[key] = out
        return out

    def summary(self) -> Dict[str, int]:
        c = Counter()
        for layer in range(self.num_layers):
            for head in range(self.num_attention_heads):
                c[self.get_strategy(layer, head).head_type] += 1
        c["total"] = self.num_layers * self.num_attention_heads
        c["physical_kv_heads"] = self.physical_kv_heads
        return dict(c)

    def _sink_at(self, layer: int, logical_head: int) -> int:
        if self.head_sink_size is None:
            return self.default_sink_size
        value = self.head_sink_size[layer][logical_head]
        return self.default_sink_size if value <= 0 else int(value)

    def _normalize(self) -> None:
        allowed = {HEAD_LOCAL, HEAD_GLOBAL, HEAD_DENSE}
        for layer in range(self.num_layers):
            if len(self.head_class[layer]) != self.num_attention_heads:
                raise ValueError(
                    f"layer {layer} heads={len(self.head_class[layer])} "
                    f"!= {self.num_attention_heads}"
                )
            if len(self.head_max_distance[layer]) != self.num_attention_heads:
                raise ValueError(
                    f"layer {layer} distances={len(self.head_max_distance[layer])} "
                    f"!= {self.num_attention_heads}"
                )
            for head, htype in enumerate(self.head_class[layer]):
                if htype not in allowed:
                    raise ValueError(
                        f"invalid MLA head type {htype!r} at layer={layer}, head={head}"
                    )
                if htype == HEAD_LOCAL and self.head_max_distance[layer][head] <= 0:
                    self.head_max_distance[layer][head] = self.local_default_window
                elif htype != HEAD_LOCAL:
                    self.head_max_distance[layer][head] = -1


def is_deepseek_v4_mla_config(config: Any) -> bool:
    return (
        getattr(config, "model_type", "") == "deepseek_v4"
        and int(getattr(config, "num_key_value_heads", 0)) == 1
        and hasattr(config, "q_lora_rank")
        and hasattr(config, "o_lora_rank")
        and hasattr(config, "qk_rope_head_dim")
    )


def deepseek_v4_mla_cache_descriptor(config: Any) -> Dict[str, int]:
    """Return the logical-vs-physical cache dimensions used by SGLang DSV4."""

    if not is_deepseek_v4_mla_config(config):
        raise ValueError("config is not DeepSeek V4 MLA")
    head_dim = int(config.head_dim)
    rope_dim = int(config.qk_rope_head_dim)
    nope_dim = head_dim - rope_dim
    return {
        "num_layers": int(config.num_hidden_layers),
        "logical_attention_heads": int(config.num_attention_heads),
        "physical_kv_heads": int(config.num_key_value_heads),
        "head_dim": head_dim,
        "qk_nope_head_dim": nope_dim,
        "qk_rope_head_dim": rope_dim,
        "q_lora_rank": int(config.q_lora_rank),
        "o_lora_rank": int(config.o_lora_rank),
    }


__all__ = [
    "DeepSeekV4MLAHeadConfig",
    "MLAHeadStrategy",
    "deepseek_v4_mla_cache_descriptor",
    "is_deepseek_v4_mla_config",
]
