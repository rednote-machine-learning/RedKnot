# Copyright 2024-2026 SGLang RedKnot Integration.
"""Head classification for RedKnot.

Each ``(layer_idx, kv_head_idx)`` pair is assigned one of four strategies:

- ``local``     : Real sliding window attention -- query only attends to
                  ``[sink] + [recent window]`` tokens. O(L * W) instead of O(L^2).
- ``global``    : Full attention over all previous KV + self (no truncation).
- ``retrieval`` : PTR-style sparse attention -- physically concat all prev KV
                  but at attention time keep only top-p important tokens.
- ``dense``     : First-N layers safety net -- full prev KV with RoPE
                  realignment, used to preserve fine-grained signal early on.

This module mirrors ``__REDKNOT_V02__/redknot/v7_head.py`` but is dependency-free
so it can be imported anywhere inside sglang. JSON layout is identical so the
existing config files under ``__REDKNOT_V02__/configs/`` work unchanged.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

# ──────────────────────────────────────────────────────────────────────────
# Head type constants
# ──────────────────────────────────────────────────────────────────────────
HEAD_LOCAL = "local"
HEAD_GLOBAL = "global"
HEAD_RETRIEVAL = "retrieval"
HEAD_DENSE = "dense"

DEFAULT_LOCAL_WINDOW = 512
DEFAULT_SINK_SIZE = 4
DEFAULT_RETRIEVAL_TOP_P = 0.9

_ALLOWED_TYPES = {HEAD_LOCAL, HEAD_GLOBAL, HEAD_RETRIEVAL, HEAD_DENSE}
# Legacy aliases kept around so old JSON configs (v0.2 / v0.3) still load.
_LEGACY_MAP = {
    "local_full": HEAD_LOCAL,
    "limited": HEAD_LOCAL,
    "streaming": HEAD_LOCAL,
    "sink_dominant": HEAD_LOCAL,
}


@dataclass(frozen=True)
class HeadStrategy:
    """Per-(layer, kv_head) attention strategy.

    Attributes
    ----------
    layer: int
        Layer index in the model.
    kv_head: int
        KV head index inside the layer (after tensor-parallel sharding).
    head_type: str
        One of ``local`` / ``global`` / ``retrieval`` / ``dense``.
    window: int
        Sliding window size for ``local``. ``-1`` for all other types.
    sink_size: int
        Number of attention-sink tokens kept from the very first segment.
    """

    layer: int
    kv_head: int
    head_type: str
    window: int
    sink_size: int

    def is_local(self) -> bool:
        return self.head_type == HEAD_LOCAL

    def is_global(self) -> bool:
        return self.head_type == HEAD_GLOBAL

    def is_retrieval(self) -> bool:
        return self.head_type == HEAD_RETRIEVAL

    def is_dense(self) -> bool:
        return self.head_type == HEAD_DENSE


class HeadClassConfig:
    """Container for the full ``[num_layers x num_kv_heads]`` head plan.

    Parameters
    ----------
    head_class:
        ``[num_layers][num_kv_heads]`` head type strings.
    head_max_distance:
        ``[num_layers][num_kv_heads]`` window sizes; meaningful only for
        ``local`` heads (use ``-1`` elsewhere).
    head_sink_size:
        Optional ``[num_layers][num_kv_heads]`` sink sizes.
    num_layers / num_kv_heads:
        Model topology. Validated against the head matrix.
    default_sink_size / local_default_window:
        Fallback values used when the matrices do not provide them.
    retrieval_top_p:
        Cumulative attention mass to retain for ``retrieval`` heads.
    dense_prefix_layers:
        If > 0, the first N layers are forced to ``dense`` regardless of the
        configured types. This is a safety knob for tiny models.
    """

    def __init__(
        self,
        head_class: List[List[str]],
        head_max_distance: List[List[int]],
        num_layers: int,
        num_kv_heads: int,
        head_sink_size: Optional[List[List[int]]] = None,
        default_sink_size: int = DEFAULT_SINK_SIZE,
        local_default_window: int = DEFAULT_LOCAL_WINDOW,
        retrieval_top_p: float = DEFAULT_RETRIEVAL_TOP_P,
        dense_prefix_layers: int = 0,
    ) -> None:
        assert len(head_class) == num_layers, (
            f"head_class layers={len(head_class)} != num_layers={num_layers}"
        )
        assert len(head_class[0]) == num_kv_heads, (
            f"head_class kv_heads={len(head_class[0])} != num_kv_heads={num_kv_heads}"
        )
        self.head_class = head_class
        self.head_max_distance = head_max_distance
        self.head_sink_size = head_sink_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.default_sink_size = default_sink_size
        self.local_default_window = local_default_window
        self.retrieval_top_p = float(retrieval_top_p)
        self.dense_prefix_layers = int(dense_prefix_layers)

        self._normalize()

        # ── Precomputed parallel tensors ───────────────────────────────
        # Building per-head tensors once allows the backend to vectorize
        # mask/window operations across the whole layer instead of looping.
        self._cached_tensors: Dict[torch.device, Dict[str, torch.Tensor]] = {}

    # ────────────────────────────────────────────────────────────────
    # IO
    # ────────────────────────────────────────────────────────────────
    @classmethod
    def from_json(
        cls,
        json_path: str,
        default_sink_size: int = DEFAULT_SINK_SIZE,
        local_default_window: int = DEFAULT_LOCAL_WINDOW,
    ) -> "HeadClassConfig":
        with open(json_path) as f:
            data = json.load(f)
        return cls(
            head_class=data["kv_head_classification"],
            head_max_distance=data["kv_head_max_distance"],
            head_sink_size=data.get("kv_head_sink_size"),
            num_layers=data["num_layers"],
            num_kv_heads=data["num_kv_heads"],
            default_sink_size=default_sink_size,
            local_default_window=local_default_window,
            retrieval_top_p=float(data.get("retrieval_top_p", DEFAULT_RETRIEVAL_TOP_P)),
            dense_prefix_layers=int(data.get("dense_prefix_layers", 0)),
        )

    # ────────────────────────────────────────────────────────────────
    # Lookups
    # ────────────────────────────────────────────────────────────────
    def get_strategy(self, layer: int, kv_head: int) -> HeadStrategy:
        # First-N dense override
        if layer < self.dense_prefix_layers:
            ss = self._sink_at(layer, kv_head)
            return HeadStrategy(
                layer=layer,
                kv_head=kv_head,
                head_type=HEAD_DENSE,
                window=-1,
                sink_size=ss,
            )

        htype = self.head_class[layer][kv_head]
        max_d = self.head_max_distance[layer][kv_head]
        ss = self._sink_at(layer, kv_head)

        if htype == HEAD_GLOBAL:
            window = -1
        elif htype == HEAD_LOCAL:
            window = max_d if max_d > 0 else self.local_default_window
        elif htype == HEAD_RETRIEVAL:
            window = -1
        elif htype == HEAD_DENSE:
            window = -1
        else:
            raise ValueError(f"Unknown head_type {htype!r}")

        return HeadStrategy(
            layer=layer,
            kv_head=kv_head,
            head_type=htype,
            window=window,
            sink_size=ss,
        )

    def set_local_window(self, new_window: int) -> int:
        """Override the sliding window of every LOCAL head to ``new_window``.

        Used for adaptive windowing: the offline-profiled config stores a
        fixed window, but the sweet spot scales with the request's context
        length (e.g. ``window = ctx_len // 2``). Only positive (local) window
        entries are changed; ``-1`` (global/full) heads are left untouched.

        Returns the number of head windows changed.
        """
        changed = 0
        for li in range(self.num_layers):
            for kvh in range(self.num_kv_heads):
                if self.head_max_distance[li][kvh] > 0:
                    self.head_max_distance[li][kvh] = new_window
                    changed += 1
        self._cached_tensors.clear()
        return changed

    def merge_retrieval_to_global(self) -> int:
        """Replace all ``retrieval`` heads with ``global``.

        This simplifies the deep-layer logic: retrieval heads were already
        treated as "keep full KV" in the SegPaged path, so promoting them
        to ``global`` has no semantic effect but removes the retrieval
        top-p sparse attention code path entirely.

        Returns the number of heads changed.
        """
        changed = 0
        for li in range(self.num_layers):
            for kvh in range(self.num_kv_heads):
                if self.head_class[li][kvh] == HEAD_RETRIEVAL:
                    self.head_class[li][kvh] = HEAD_GLOBAL
                    changed += 1
        # Invalidate cached tensors since head types changed.
        self._cached_tensors.clear()
        return changed

    def summary(self) -> Dict[str, int]:
        flat = []
        for li in range(self.num_layers):
            for kvh in range(self.num_kv_heads):
                if li < self.dense_prefix_layers:
                    flat.append(HEAD_DENSE)
                else:
                    flat.append(self.head_class[li][kvh])
        c = dict(Counter(flat))
        c["total"] = self.num_layers * self.num_kv_heads
        c["dense_prefix_layers"] = self.dense_prefix_layers
        return c

    # ────────────────────────────────────────────────────────────────
    # Parallel-friendly tensor views (used by the GPU backend)
    # ────────────────────────────────────────────────────────────────
    # Integer encoding for head_type so we can put it on GPU.
    TYPE_LOCAL = 0
    TYPE_GLOBAL = 1
    TYPE_RETRIEVAL = 2
    TYPE_DENSE = 3
    _TYPE_TO_INT = {
        HEAD_LOCAL: TYPE_LOCAL,
        HEAD_GLOBAL: TYPE_GLOBAL,
        HEAD_RETRIEVAL: TYPE_RETRIEVAL,
        HEAD_DENSE: TYPE_DENSE,
    }

    def as_tensors(
        self, device: torch.device, dtype: torch.dtype = torch.int32
    ) -> Dict[str, torch.Tensor]:
        """Return ``[L, KVH]`` int tensors for type / window / sink.

        Cached per device. The backend uses these to dispatch attention
        per-head in parallel using ``torch.where`` / segment masks instead
        of a Python loop.
        """
        cache_key = device
        if cache_key in self._cached_tensors:
            return self._cached_tensors[cache_key]

        L, KVH = self.num_layers, self.num_kv_heads
        types = torch.empty(L, KVH, dtype=dtype, device=device)
        windows = torch.empty(L, KVH, dtype=dtype, device=device)
        sinks = torch.empty(L, KVH, dtype=dtype, device=device)

        for li in range(L):
            for kvh in range(KVH):
                strat = self.get_strategy(li, kvh)
                types[li, kvh] = self._TYPE_TO_INT[strat.head_type]
                windows[li, kvh] = strat.window
                sinks[li, kvh] = strat.sink_size

        bundle = {
            "head_type": types,
            "window": windows,
            "sink_size": sinks,
            "retrieval_top_p": torch.tensor(
                self.retrieval_top_p, dtype=torch.float32, device=device
            ),
        }
        self._cached_tensors[cache_key] = bundle
        return bundle

    # ────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────
    def _sink_at(self, layer: int, kv_head: int) -> int:
        if self.head_sink_size is not None:
            return int(self.head_sink_size[layer][kv_head])
        return self.default_sink_size

    def _normalize(self) -> None:
        for li in range(self.num_layers):
            for kvh in range(self.num_kv_heads):
                t = self.head_class[li][kvh]
                if t in _ALLOWED_TYPES:
                    continue
                if t in _LEGACY_MAP:
                    self.head_class[li][kvh] = _LEGACY_MAP[t]
                    continue
                raise ValueError(
                    f"Unknown head_type {t!r} at (layer={li}, kvh={kvh}); "
                    f"allowed: {sorted(_ALLOWED_TYPES)} (or legacy "
                    f"{sorted(_LEGACY_MAP)})."
                )
