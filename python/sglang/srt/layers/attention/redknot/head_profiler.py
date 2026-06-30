# Copyright 2024-2026 SGLang RedKnot Integration.
"""Offline head profiling -> Head Class Map (RedKnot §3.2, fig. 3).

RedKnot classifies every ``(layer, head)`` pair **offline** as either
``global`` (prefix-sensitive, re-prefilled on reuse; 12-15% of heads) or
``local`` (prefix-robust, reused within a sliding window; 85-88%). This is a
model-intrinsic property (paper §2.2 / §3.2), so it is profiled once and
reused across requests at zero per-request cost.

This module turns that observation into a runnable profiler that produces a
:class:`HeadClassConfig` compatible with the rest of the RedKnot stack.

Two signals drive the classification (paper §3.2 / §6 "edge-mass"):

1. **Prefix sensitivity** — how much a head's KV / attention output changes
   when the same chunk is placed behind a different prefix vs. at the start.
   Global heads change a lot; local heads barely move.
2. **Attention distance / edge-mass** — how concentrated a head's attention
   is on the recent window + sink. A head whose attention mass lives in the
   local window is ``local``; a head that spreads mass across the full
   context is ``global``.

A head is labelled ``global`` if **either** signal exceeds its threshold,
which is the conservative choice (false-``local`` hurts quality more than
false-``global`` hurts speed).

Usage
-----
- :func:`profile_model_heads` runs the profiling on a real HuggingFace model
  using a small calibration corpus, and is the production entry point.
- :func:`classify_from_stats` is the pure, model-free core that maps raw
  per-head statistics to a Head Class Map; it is unit-testable on CPU.
- :func:`build_head_config` packages the labels into a ``HeadClassConfig``
  (and :func:`save_head_config_json` writes the JSON consumed by
  ``--redknot-head-config-path``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.attention.redknot.head_config import (
    DEFAULT_LOCAL_WINDOW,
    DEFAULT_SINK_SIZE,
    HEAD_GLOBAL,
    HEAD_LOCAL,
    HeadClassConfig,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Per-head statistics
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class HeadStats:
    """Accumulated profiling statistics for one ``(layer, head)``.

    Attributes
    ----------
    prefix_shift:
        Mean relative change of the head's attention output when the chunk
        is moved behind a different prefix (0 = identical, 1 = fully
        changed). High => prefix-sensitive => global.
    edge_mass:
        Fraction of attention mass that falls **outside** the local window
        + sink (i.e. on distant tokens). High => long-range => global.
    samples:
        Number of calibration samples accumulated.
    """

    prefix_shift: float = 0.0
    edge_mass: float = 0.0
    samples: int = 0


@dataclass
class ProfileResult:
    """Output of profiling: per-head labels + the stats that produced them."""

    num_layers: int
    num_kv_heads: int
    labels: List[List[str]]
    stats: List[List[HeadStats]]
    global_ratio: float
    thresholds: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Pure classification core (unit-testable, no model)
# ──────────────────────────────────────────────────────────────────────────
def classify_from_stats(
    prefix_shift: torch.Tensor,
    edge_mass: torch.Tensor,
    *,
    target_global_ratio: Optional[float] = 0.15,
    prefix_thresh: Optional[float] = None,
    edge_thresh: Optional[float] = None,
) -> Tuple[List[List[str]], Dict[str, float]]:
    """Map ``[L, H]`` per-head stats to a ``[L][H]`` Head Class Map.

    A head is ``global`` if its combined sensitivity score is in the top
    ``target_global_ratio`` of all heads, or (when explicit thresholds are
    given) if either signal exceeds its threshold.

    Parameters
    ----------
    prefix_shift, edge_mass:
        ``[L, H]`` float tensors of per-head statistics in ``[0, 1]``.
    target_global_ratio:
        If set, pick the top fraction of heads (by combined score) as
        global. Matches the paper's ~10-15% global ratio. When ``None``,
        explicit thresholds are used instead.
    prefix_thresh, edge_thresh:
        Optional absolute thresholds. Used when ``target_global_ratio`` is
        ``None`` (or as an additional OR-condition when both are given).

    Returns
    -------
    ``(labels, info)`` where ``labels[l][h]`` is ``"global"`` or
    ``"local"`` and ``info`` records the thresholds used.
    """
    if prefix_shift.shape != edge_mass.shape:
        raise ValueError(
            f"classify_from_stats: shape mismatch {tuple(prefix_shift.shape)} "
            f"vs {tuple(edge_mass.shape)}"
        )
    if prefix_shift.dim() != 2:
        raise ValueError("classify_from_stats expects [L, H] tensors")
    L, H = prefix_shift.shape

    ps = prefix_shift.float()
    em = edge_mass.float()
    # Combined score: a head is "global-like" if it is both prefix-sensitive
    # and spreads attention mass. Use the max so either signal can flag it.
    combined = torch.maximum(ps, em)

    info: Dict[str, float] = {}
    is_global = torch.zeros(L, H, dtype=torch.bool)

    if target_global_ratio is not None:
        n_total = L * H
        n_global = max(1, int(round(n_total * target_global_ratio)))
        flat = combined.flatten()
        # Top-n by combined score.
        topk = torch.topk(flat, k=min(n_global, n_total)).indices
        mask = torch.zeros(n_total, dtype=torch.bool)
        mask[topk] = True
        is_global = mask.reshape(L, H)
        info["target_global_ratio"] = float(target_global_ratio)
        info["cutoff_score"] = float(flat[topk].min().item())

    if prefix_thresh is not None or edge_thresh is not None:
        pt = prefix_thresh if prefix_thresh is not None else float("inf")
        et = edge_thresh if edge_thresh is not None else float("inf")
        thr_global = (ps >= pt) | (em >= et)
        is_global = is_global | thr_global
        info["prefix_thresh"] = float(pt)
        info["edge_thresh"] = float(et)

    labels = [
        [HEAD_GLOBAL if bool(is_global[l, h]) else HEAD_LOCAL for h in range(H)]
        for l in range(L)
    ]
    info["realized_global_ratio"] = float(is_global.float().mean().item())
    return labels, info


def build_head_config(
    labels: List[List[str]],
    *,
    num_layers: int,
    num_kv_heads: int,
    local_window: int = DEFAULT_LOCAL_WINDOW,
    sink_size: int = DEFAULT_SINK_SIZE,
    dense_prefix_layers: int = 0,
) -> HeadClassConfig:
    """Package a Head Class Map into a :class:`HeadClassConfig`."""
    head_max_distance = [
        [
            local_window if labels[l][h] == HEAD_LOCAL else -1
            for h in range(num_kv_heads)
        ]
        for l in range(num_layers)
    ]
    head_sink_size = [[sink_size] * num_kv_heads for _ in range(num_layers)]
    return HeadClassConfig(
        head_class=labels,
        head_max_distance=head_max_distance,
        head_sink_size=head_sink_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        local_default_window=local_window,
        default_sink_size=sink_size,
        dense_prefix_layers=dense_prefix_layers,
    )


def save_head_config_json(
    cfg: HeadClassConfig,
    path: str,
    *,
    extra: Optional[Dict] = None,
) -> None:
    """Write a ``HeadClassConfig`` in the JSON layout the backend loads."""
    data = {
        "num_layers": cfg.num_layers,
        "num_kv_heads": cfg.num_kv_heads,
        "kv_head_classification": cfg.head_class,
        "kv_head_max_distance": cfg.head_max_distance,
        "kv_head_sink_size": cfg.head_sink_size,
        "dense_prefix_layers": cfg.dense_prefix_layers,
        "retrieval_top_p": cfg.retrieval_top_p,
    }
    if extra:
        data["profiling_meta"] = extra
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────
# Model-driven profiling (production entry point)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _attention_edge_mass(
    attn_weights: torch.Tensor,
    *,
    window: int,
    sink: int,
) -> torch.Tensor:
    """Fraction of attention mass outside ``[sink] ∪ [recent window]``.

    Parameters
    ----------
    attn_weights:
        ``[H, Lq, Lk]`` attention probabilities for one layer (already
        softmaxed). We average over query positions.
    window, sink:
        Local visible region per query position.

    Returns
    -------
    ``[H]`` edge-mass per head in ``[0, 1]``.
    """
    H, Lq, Lk = attn_weights.shape
    device = attn_weights.device
    # Visible mask per (query i, key j): j in [0, sink) or j in (i-window, i].
    qi = torch.arange(Lq, device=device).unsqueeze(1)  # [Lq, 1]
    kj = torch.arange(Lk, device=device).unsqueeze(0)  # [1, Lk]
    in_sink = kj < sink
    in_window = (kj <= qi) & (kj > (qi - window))
    visible = in_sink | in_window  # [Lq, Lk]
    visible = visible.unsqueeze(0)  # [1, Lq, Lk]
    edge = (attn_weights * (~visible)).sum(dim=-1)  # [H, Lq]
    total = attn_weights.sum(dim=-1).clamp_min(1e-9)
    return (edge / total).mean(dim=-1)  # [H]


@torch.no_grad()
def profile_model_heads(
    model,
    tokenizer,
    calibration_texts: Sequence[str],
    *,
    prefixes: Optional[Sequence[str]] = None,
    window: int = DEFAULT_LOCAL_WINDOW,
    sink: int = DEFAULT_SINK_SIZE,
    target_global_ratio: float = 0.15,
    max_tokens: int = 2048,
) -> ProfileResult:
    """Profile a HuggingFace model and produce a Head Class Map.

    For each calibration chunk we run it (a) at the start and (b) behind a
    different prefix, capturing per-head attention. We then compute:

    - ``prefix_shift``: 1 - cosine of the head's attention output between
      the two placements (high = prefix-sensitive = global).
    - ``edge_mass``: attention mass outside the local window + sink.

    Heads in the top ``target_global_ratio`` by combined score are labelled
    ``global``; the rest ``local``.

    Notes
    -----
    Requires the model to be loaded with ``attn_implementation="eager"`` so
    attention weights are returned. This is a calibration-time cost only.
    """
    if prefixes is None:
        prefixes = [
            "The following document is provided for reference.\n\n",
            "Background context unrelated to the question below.\n\n",
        ]

    config = model.config
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(config.num_key_value_heads)
    num_q_heads = int(config.num_attention_heads)
    q_per_kv = num_q_heads // num_kv_heads
    device = model.device

    shift_acc = torch.zeros(num_layers, num_q_heads)
    edge_acc = torch.zeros(num_layers, num_q_heads)
    n_samples = 0

    def _run_capture(text: str):
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][:, :max_tokens].to(device)
        out = model(input_ids=ids, output_attentions=True, use_cache=False)
        # tuple[num_layers] of [B, Hq, Lq, Lk]
        return ids, out.attentions

    for text in calibration_texts:
        ids0, attn0 = _run_capture(text)
        L0 = ids0.shape[1]
        for prefix in prefixes:
            ids1, attn1 = _run_capture(prefix + text)
            p_len = ids1.shape[1] - L0
            for li in range(num_layers):
                a0 = attn0[li][0]  # [Hq, L0, L0]
                a1 = attn1[li][0]  # [Hq, L1, L1]
                # Compare the chunk's own attention rows (drop the prefix
                # rows/cols in the prefixed run).
                a1_chunk = a1[:, p_len:, p_len:]
                Lq = min(a0.shape[1], a1_chunk.shape[1])
                v0 = a0[:, :Lq, :Lq].reshape(a0.shape[0], -1).float()
                v1 = a1_chunk[:, :Lq, :Lq].reshape(a1_chunk.shape[0], -1).float()
                cos = torch.nn.functional.cosine_similarity(v0, v1, dim=-1)
                shift_acc[li] += (1.0 - cos).clamp(0, 1).cpu()
                edge_acc[li] += _attention_edge_mass(a0, window=window, sink=sink).cpu()
            n_samples += 1

    if n_samples == 0:
        raise ValueError("profile_model_heads: no calibration samples produced")

    shift_q = shift_acc / n_samples  # [L, Hq]
    edge_q = edge_acc / n_samples

    # Reduce query heads -> kv heads (GQA): a KV head is global if any of its
    # query heads is global-like, so take the max over the fanout group.
    shift_kv = shift_q.reshape(num_layers, num_kv_heads, q_per_kv).amax(dim=-1)
    edge_kv = edge_q.reshape(num_layers, num_kv_heads, q_per_kv).amax(dim=-1)

    labels, info = classify_from_stats(
        shift_kv, edge_kv, target_global_ratio=target_global_ratio
    )
    stats = [
        [
            HeadStats(
                prefix_shift=float(shift_kv[l, h]),
                edge_mass=float(edge_kv[l, h]),
                samples=n_samples,
            )
            for h in range(num_kv_heads)
        ]
        for l in range(num_layers)
    ]
    return ProfileResult(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        labels=labels,
        stats=stats,
        global_ratio=info.get("realized_global_ratio", target_global_ratio),
        thresholds=info,
    )


__all__ = [
    "HeadStats",
    "ProfileResult",
    "classify_from_stats",
    "build_head_config",
    "save_head_config_json",
    "profile_model_heads",
]
