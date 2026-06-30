# Copyright 2024-2026 SGLang RedKnot Integration.
"""Online per-(layer, head) attention-locality profiler for DeepSeek V4 MLA.

DeepSeek V4 attention runs in MLA absorb form: every logical attention head
shares a single latent KV stream but applies its own query projection. The
per-head attention score for query token ``i`` over key token ``j`` is therefore

    score[h, i, j] = (q[i, h, :] . latent_k[j, :]) * softmax_scale

which can be recomputed cheaply for a *sampled* subset of query rows during
prefill -- no per-head materialized K/V is required.

This module accumulates, per (layer, logical head), the distribution of
attention *mass* as a function of the query->key relative distance
``d = i - j`` (causal, so ``d >= 0``). From the accumulated histogram we derive,
for each head, the smallest window ``w`` such that the average attention mass
within distance ``w`` reaches a target coverage ``p`` (e.g. 0.95). Heads whose
required window stays small are classified ``local`` (window = w); heads that
need (almost) the whole context are classified ``global``.

The profiler is intentionally backend-side and append-only: a single global
collector is filled by ``RedKnotMLAAttnBackend`` when analysis mode is on, then
exported to a ``DeepSeekV4MLAHeadConfig`` JSON.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.attention.redknot.head_config import (
    DEFAULT_SINK_SIZE,
    HEAD_DENSE,
    HEAD_GLOBAL,
    HEAD_LOCAL,
)

logger = logging.getLogger(__name__)


# Log-spaced distance bin edges (upper-bounds, inclusive). The last bin acts as
# an overflow that captures "needs (almost) the whole context" mass.
def _default_bin_edges() -> List[int]:
    edges = [
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        1 << 30,
    ]
    return edges


@dataclass
class MLAHeadProfileConfig:
    """Controls how the online profiler samples and classifies heads."""

    num_layers: int
    num_heads: int
    # Coverage target: window must capture this fraction of attention mass.
    coverage: float = 0.95
    # Number of query rows sampled per layer per forward (keeps cost O(T)).
    sample_queries: int = 256
    # A head is global if the coverage window exceeds this fraction of context.
    global_window_ratio: float = 0.5
    # Safety multiplier applied to the measured coverage window.
    window_safety: float = 1.5
    # Round local windows up to a multiple of this (FlashMLA friendliness).
    window_round_to: int = 64
    # Minimum local window floor.
    window_min: int = 64
    # Layers below this stay dense (full attention) regardless of measurement.
    dense_prefix_layers: int = 2
    sink_size: int = DEFAULT_SINK_SIZE
    bin_edges: List[int] = field(default_factory=_default_bin_edges)


class MLAHeadLocalityCollector:
    """Accumulates per-(layer, head) attention-mass-by-distance histograms."""

    def __init__(self, cfg: MLAHeadProfileConfig) -> None:
        self.cfg = cfg
        n_bins = len(cfg.bin_edges)
        # mass[layer, head, bin]: summed attention mass; counts[layer]: #queries.
        self._mass = torch.zeros(
            (cfg.num_layers, cfg.num_heads, n_bins), dtype=torch.float64
        )
        self._query_rows = torch.zeros(cfg.num_layers, dtype=torch.float64)
        self._max_ctx = 0
        self._lock = threading.Lock()
        self._edges = torch.tensor(cfg.bin_edges, dtype=torch.long)
        # Per-layer token-level attention-mass concentration accumulator. For
        # each sampled (query, head) we measure the fraction of *visible* causal
        # key tokens needed to cover ``coverage`` of the attention mass, then sum
        # those fractions per layer (and track the count) so we can report the
        # mean/min/max layer-wise concentration -- the "tokens for 99% attn"
        # curve. This reuses the same real softmax probs computed for the
        # distance histogram, so it costs nothing extra beyond a sort.
        self._conc_sum = torch.zeros(cfg.num_layers, dtype=torch.float64)
        self._conc_sq = torch.zeros(cfg.num_layers, dtype=torch.float64)
        self._conc_min = torch.full(
            (cfg.num_layers,), float("inf"), dtype=torch.float64
        )
        self._conc_max = torch.zeros(cfg.num_layers, dtype=torch.float64)
        self._conc_count = torch.zeros(cfg.num_layers, dtype=torch.float64)
        # GPU-resident concentration accumulators, lazily allocated on the first
        # observed layer. CRITICAL: at TP>1 the MLA forward runs collectives
        # (all-reduce) between layers, so the profiler MUST NOT trigger any
        # GPU->CPU sync (.item()/.cpu()/.tolist()) inside the forward, or it
        # desyncs ranks and deadlocks. We accumulate purely on-GPU here and only
        # copy to CPU at export time (outside the forward).
        self._g_sum: Optional[torch.Tensor] = None
        self._g_sq: Optional[torch.Tensor] = None
        self._g_min: Optional[torch.Tensor] = None
        self._g_max: Optional[torch.Tensor] = None
        self._g_count: Optional[torch.Tensor] = None
        self._g_device = None

    @property
    def max_context(self) -> int:
        return self._max_ctx

    @torch.no_grad()
    def observe_layer(
        self,
        layer_id: int,
        q: torch.Tensor,
        latent_k: torch.Tensor,
        softmax_scale: float,
    ) -> None:
        """Record one prefill layer.

        ``q``: ``[T, H, D]`` per-head queries (post norm + RoPE).
        ``latent_k``: ``[T, D]`` shared latent key stream (post norm + RoPE).
        Only a sampled set of query rows is scored to keep this O(T).
        """
        if layer_id >= self.cfg.num_layers:
            return
        if q.ndim != 3 or latent_k.ndim != 2:
            return
        T, H, D = q.shape
        if T < 2 or H != self.cfg.num_heads or latent_k.shape[0] != T:
            return

        device = q.device
        q = q.float()
        latent_k = latent_k.float()

        # Sample query rows. Spread samples across the sequence. Keep this on GPU
        # (no .tolist()) so nothing in this method forces a CUDA sync.
        n_samp = min(self.cfg.sample_queries, T)
        if n_samp <= 0:
            return
        rows = torch.linspace(0, T - 1, steps=n_samp, device=device).long().unique()
        S = rows.numel()

        coverage = float(self.cfg.coverage)
        min_visible = max(8, int(0.25 * T))

        # Lazily allocate GPU accumulators (one entry per layer).
        if self._g_sum is None or self._g_device != device:
            L = self.cfg.num_layers
            self._g_sum = torch.zeros(L, dtype=torch.float64, device=device)
            self._g_sq = torch.zeros(L, dtype=torch.float64, device=device)
            self._g_min = torch.full(
                (L,), float("inf"), dtype=torch.float64, device=device
            )
            self._g_max = torch.zeros(L, dtype=torch.float64, device=device)
            self._g_count = torch.zeros(L, dtype=torch.float64, device=device)
            self._g_device = device

        # Vectorized over sampled queries: build [S, H, T] logits and apply a
        # causal mask (key j visible to query i iff j <= i). Masked positions get
        # -inf so softmax ignores them. This avoids the per-query Python loop and
        # any GPU->CPU sync entirely.
        qsel = q[rows]  # [S, H, D]
        logits = torch.einsum("shd,td->sht", qsel, latent_k) * softmax_scale  # [S,H,T]
        key_pos = torch.arange(T, device=device)
        causal = key_pos[None, :] <= rows[:, None]  # [S, T] visible
        mask = causal[:, None, :]  # [S, 1, T]
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)  # [S, H, T]

        # n_visible per sampled query = i + 1.
        n_vis = (rows + 1).to(torch.float64)  # [S]
        keep = n_vis >= float(min_visible)  # [S] bool

        # Fraction of visible keys needed to cover ``coverage`` of the mass.
        sorted_p, _ = torch.sort(probs, dim=-1, descending=True)  # [S,H,T]
        cum = torch.cumsum(sorted_p, dim=-1)
        reached = cum >= coverage  # [S,H,T]
        n_need = reached.float().argmax(dim=-1) + 1  # [S,H]
        frac = (n_need / n_vis[:, None].to(n_need.dtype)).clamp_(0.0, 1.0)  # [S,H]
        fmean_per_q = frac.mean(dim=1).to(torch.float64)  # [S]

        keep_f = keep.to(torch.float64)
        kept_vals = torch.where(keep, fmean_per_q, torch.zeros_like(fmean_per_q))
        layer_sum = (kept_vals).sum()
        layer_sq = (kept_vals * kept_vals).sum()
        layer_cnt = keep_f.sum()
        # min/max only over kept queries (push inf/-inf for dropped ones).
        big = torch.full_like(fmean_per_q, float("inf"))
        small = torch.full_like(fmean_per_q, float("-inf"))
        layer_min = torch.where(keep, fmean_per_q, big).min()
        layer_max = torch.where(keep, fmean_per_q, small).max()

        # Accumulate on GPU (no sync). Use index_add for the scalar layer slot.
        idx = torch.tensor([layer_id], device=device)
        self._g_sum.index_add_(0, idx, layer_sum.reshape(1))
        self._g_sq.index_add_(0, idx, layer_sq.reshape(1))
        self._g_count.index_add_(0, idx, layer_cnt.reshape(1))
        self._g_min[layer_id] = torch.minimum(self._g_min[layer_id], layer_min)
        self._g_max[layer_id] = torch.maximum(self._g_max[layer_id], layer_max)
        # Track max context without sync (Python int from tensor shape is fine).
        if T > self._max_ctx:
            self._max_ctx = T

    @torch.no_grad()
    def _coverage_window(self, layer_id: int, head: int) -> float:
        """Smallest distance bin upper-bound covering ``coverage`` mass."""
        counts = self._query_rows[layer_id].item()
        if counts <= 0:
            return float("inf")
        mass = self._mass[layer_id, head] / counts  # avg mass per bin
        total = float(mass.sum().item())
        if total <= 0:
            return float("inf")
        cum = torch.cumsum(mass / total, dim=0)
        target = self.cfg.coverage
        for b in range(cum.numel()):
            if cum[b].item() >= target:
                return float(self.cfg.bin_edges[b])
        return float(self.cfg.bin_edges[-1])

    @torch.no_grad()
    def build_head_config(self):
        """Derive a ``DeepSeekV4MLAHeadConfig`` from accumulated statistics."""
        from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
            DeepSeekV4MLAHeadConfig,
        )

        cfg = self.cfg
        ctx = max(self._max_ctx, 1)
        global_thresh = cfg.global_window_ratio * ctx

        head_class: List[List[str]] = []
        head_distance: List[List[int]] = []
        sinks: List[List[int]] = []
        for layer in range(cfg.num_layers):
            row_class: List[str] = []
            row_dist: List[int] = []
            for head in range(cfg.num_heads):
                if layer < cfg.dense_prefix_layers:
                    row_class.append(HEAD_DENSE)
                    row_dist.append(-1)
                    continue
                w = self._coverage_window(layer, head)
                if not math.isfinite(w) or w >= global_thresh:
                    row_class.append(HEAD_GLOBAL)
                    row_dist.append(-1)
                else:
                    win = int(math.ceil(w * cfg.window_safety))
                    win = max(cfg.window_min, win)
                    rt = max(1, cfg.window_round_to)
                    win = int(math.ceil(win / rt) * rt)
                    row_class.append(HEAD_LOCAL)
                    row_dist.append(win)
            head_class.append(row_class)
            head_distance.append(row_dist)
            sinks.append([cfg.sink_size] * cfg.num_heads)

        return DeepSeekV4MLAHeadConfig(
            head_class=head_class,
            head_max_distance=head_distance,
            head_sink_size=sinks,
            num_layers=cfg.num_layers,
            num_attention_heads=cfg.num_heads,
            physical_kv_heads=1,
            default_sink_size=cfg.sink_size,
            local_default_window=cfg.window_min,
            dense_prefix_layers=cfg.dense_prefix_layers,
        )

    @torch.no_grad()
    def report(self) -> Dict[str, object]:
        """Human-readable per-layer summary of head classification + windows."""
        hc = self.build_head_config()
        ctx = max(self._max_ctx, 1)
        layers = []
        n_global = n_local = n_dense = 0
        local_windows: List[int] = []
        for layer in range(self.cfg.num_layers):
            g = l = d = 0
            wins: List[int] = []
            for head in range(self.cfg.num_heads):
                t = hc.head_class[layer][head]
                if t == HEAD_GLOBAL:
                    g += 1
                elif t == HEAD_LOCAL:
                    l += 1
                    wins.append(hc.head_max_distance[layer][head])
                else:
                    d += 1
            n_global += g
            n_local += l
            n_dense += d
            local_windows.extend(wins)
            layers.append(
                {
                    "layer": layer,
                    "global": g,
                    "local": l,
                    "dense": d,
                    "local_window_min": min(wins) if wins else 0,
                    "local_window_max": max(wins) if wins else 0,
                }
            )
        return {
            "max_context": ctx,
            "coverage": self.cfg.coverage,
            "global_window_ratio": self.cfg.global_window_ratio,
            "totals": {
                "global": n_global,
                "local": n_local,
                "dense": n_dense,
                "total": self.cfg.num_layers * self.cfg.num_heads,
            },
            "local_window_overall_min": min(local_windows) if local_windows else 0,
            "local_window_overall_max": max(local_windows) if local_windows else 0,
            "per_layer": layers,
        }

    @torch.no_grad()
    def concentration_curve(self) -> Dict[str, object]:
        """Per-layer token-level attention-mass concentration.

        For every layer reports the mean / min / max fraction of visible causal
        key tokens needed to cover ``coverage`` of the attention mass (averaged
        over heads, then over sampled queries). This is the real
        "tokens for N% attn" curve measured from sglang's own forward.
        """
        # Merge GPU-resident accumulators (filled sync-free during the forward)
        # into CPU once, here at export time (safe: outside the forward).
        if self._g_count is not None:
            g_sum = self._g_sum.cpu()
            g_sq = self._g_sq.cpu()
            g_min = self._g_min.cpu()
            g_max = self._g_max.cpu()
            g_count = self._g_count.cpu()
        else:
            g_sum = self._conc_sum
            g_sq = self._conc_sq
            g_min = self._conc_min
            g_max = self._conc_max
            g_count = self._conc_count

        layers: List[int] = []
        mean: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        for layer in range(self.cfg.num_layers):
            c = float(g_count[layer].item())
            if c <= 0:
                continue
            m = float(g_sum[layer].item()) / c
            layers.append(layer)
            mean.append(m)
            lo.append(float(g_min[layer].item()))
            hi.append(float(g_max[layer].item()))
        return {
            "metric": f"frac_tokens_for_{int(round(self.cfg.coverage * 100))}pct_mass",
            "method": "sglang_real_forward_mla_eager_softmax",
            "coverage": self.cfg.coverage,
            "sample_queries": self.cfg.sample_queries,
            "max_context": self.max_context,
            "layers": layers,
            "mean": mean,
            "min": lo,
            "max": hi,
        }

    def export_concentration_json(
        self, path: str, model_name: str = "DeepSeek-V4"
    ) -> None:
        """Export the per-layer concentration curve as a figure-ready JSON."""
        curve = self.concentration_curve()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({model_name: curve}, f, indent=2)
            f.write("\n")
        logger.info("RedKnot MLA concentration curve exported to %s", path)

    def export_json(self, path: str) -> None:
        self.build_head_config().to_json(path)
        logger.info("RedKnot MLA head profile exported to %s", path)


# ──────────────────────────────────────────────────────────────────────────
# Process-global collector wiring (filled by the backend, read by the bench).
# ──────────────────────────────────────────────────────────────────────────
_GLOBAL_COLLECTOR: Optional[MLAHeadLocalityCollector] = None


def enable_global_collector(cfg: MLAHeadProfileConfig) -> MLAHeadLocalityCollector:
    global _GLOBAL_COLLECTOR
    _GLOBAL_COLLECTOR = MLAHeadLocalityCollector(cfg)
    return _GLOBAL_COLLECTOR


def get_global_collector() -> Optional[MLAHeadLocalityCollector]:
    return _GLOBAL_COLLECTOR


def disable_global_collector() -> None:
    global _GLOBAL_COLLECTOR
    _GLOBAL_COLLECTOR = None


__all__ = [
    "MLAHeadProfileConfig",
    "MLAHeadLocalityCollector",
    "enable_global_collector",
    "get_global_collector",
    "disable_global_collector",
]
