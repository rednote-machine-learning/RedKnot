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


# Hybrid distance bin edges (upper-bounds, inclusive). The last bin acts as an
# overflow that captures "needs (almost) the whole context" mass.
def _default_bin_edges() -> List[int]:
    # Fine resolution around the 8K RedKnot block boundary is important: with
    # pure power-of-two bins, every true range in 4097..8192 was rounded to 8K
    # and could be misclassified as global in a 16K context.  The hybrid grid
    # stays compact enough to export while retaining useful range precision up
    # through the planned 256K experiments.
    edges = [0, 1, 2, 4, 8, 16, 32, 64]
    edges.extend(range(128, 8192 + 1, 128))
    edges.extend(range(8704, 32768 + 1, 512))
    edges.extend(range(34816, 131072 + 1, 2048))
    edges.extend(range(139264, 262144 + 1, 8192))
    edges.append(1 << 30)
    return edges


@dataclass
class MLAHeadProfileConfig:
    """Controls how the online profiler samples and classifies heads."""

    num_layers: int
    num_heads: int
    # Exact length of the one request that is allowed to contribute a profile.
    # A value of zero preserves the legacy one-forward behavior.  Offline
    # chunked-prefill profiling must set this to the real tokenized context
    # length so warmups and incomplete requests can never be exported.
    expected_context: int = 0
    # Coverage target: window must capture this fraction of attention mass.
    coverage: float = 0.95
    # Number of query rows sampled per layer per forward (keeps cost O(T)).
    sample_queries: int = 256
    # Protect query-dependent retrieval heads: classify from this quantile of
    # per-query coverage windows, rather than only the average mass histogram.
    query_window_quantile: float = 0.90
    # A head is global if the coverage window exceeds this fraction of context.
    global_window_ratio: float = 0.5
    # Safety multiplier applied to the measured coverage window.
    window_safety: float = 1.5
    # Round local windows up to a multiple of this (FlashMLA friendliness).
    window_round_to: int = 64
    # Minimum local window floor.
    window_min: int = 64
    # Boundary layers stay dense (full attention) regardless of measurement.
    dense_prefix_layers: int = 3
    dense_suffix_layers: int = 0
    sink_size: int = DEFAULT_SINK_SIZE
    bin_edges: List[int] = field(default_factory=_default_bin_edges)


@dataclass
class _LayerRequestCapture:
    """GPU-resident pieces of one request for one transformer layer."""

    generation: int = 0
    next_prefix: int = 0
    k_chunks: List[torch.Tensor] = field(default_factory=list)
    q_chunks: List[torch.Tensor] = field(default_factory=list)
    query_rows: List[torch.Tensor] = field(default_factory=list)
    complete: bool = False
    invalid: bool = False


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
        # Distance-mass statistics must also stay on GPU while a TP forward is
        # in flight.  The original implementation allocated CPU accumulators
        # above but never populated them, which made every profiled head look
        # global.  Keep the small [L, H, bins] reduction on device and copy it
        # only when the profile is exported after the forward completes.
        self._g_mass: Optional[torch.Tensor] = None
        self._g_query_rows: Optional[torch.Tensor] = None
        self._g_window_counts: Optional[torch.Tensor] = None
        self._g_last_sample_rows: Optional[torch.Tensor] = None
        self._g_device = None
        # Chunked prefill invokes every layer once per chunk.  Keep latent K for
        # every chunk, but retain Q only at the globally preselected query rows.
        # A layer is reduced only after its K stream reaches expected_context.
        self._captures = [
            _LayerRequestCapture() for _ in range(cfg.num_layers)
        ]
        self._sealed = False

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
        *,
        prefix_len: int = 0,
        extend_len: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> bool:
        """Record one prefill layer.

        ``q`` is ``[extend_len, H, D]`` and ``latent_k`` is
        ``[extend_len, D]`` for the current chunk.  ``prefix_len`` and
        ``seq_len`` are request boundaries read from ForwardBatch's CPU
        metadata.  The method returns true exactly once: on the last layer of a
        fully captured request whose length equals ``expected_context``.

        K is accumulated across chunks.  Q is retained only for the globally
        preselected sample positions; scoring happens in FP32 after the full K
        stream is available, with a full-context causal mask.
        """
        if self._sealed or layer_id < 0 or layer_id >= self.cfg.num_layers:
            return False
        if q.ndim != 3 or latent_k.ndim != 2:
            return False
        chunk_tokens, H, _ = q.shape
        actual_extend = chunk_tokens if extend_len is None else int(extend_len)
        prefix = int(prefix_len)
        chunk_end = prefix + actual_extend
        sequence_end = chunk_end if seq_len is None else int(seq_len)
        if (
            actual_extend <= 0
            or actual_extend > chunk_tokens
            or actual_extend > latent_k.shape[0]
            or H != self.cfg.num_heads
            or prefix < 0
            or sequence_end != chunk_end
        ):
            return False

        expected = int(self.cfg.expected_context)
        if expected <= 0:
            # Backward-compatible non-chunked mode used by focused unit tests
            # and older offline runs.
            expected = sequence_end
        if chunk_end > expected:
            self._captures[layer_id].invalid = True
            return False

        capture = self._captures[layer_id]
        if prefix == 0:
            self._reset_layer_capture(layer_id)
            capture = self._captures[layer_id]
            capture.generation += 1
        elif capture.invalid or capture.complete or prefix != capture.next_prefix:
            # A missing, duplicated, interleaved, or cached-prefix chunk cannot
            # produce a complete causal K stream.  Ignore it until the next
            # unambiguous request boundary (prefix == 0).
            capture.invalid = True
            return False

        device = q.device
        sample_rows = self._select_query_rows(expected, device)
        in_chunk = (sample_rows >= prefix) & (sample_rows < chunk_end)
        local_rows = sample_rows[in_chunk] - prefix
        if local_rows.numel() > 0:
            capture.q_chunks.append(q[:actual_extend][local_rows].detach())
            capture.query_rows.append(sample_rows[in_chunk])
        capture.k_chunks.append(latent_k[:actual_extend].detach())
        capture.next_prefix = chunk_end

        if chunk_end < expected:
            return False

        q_rows = torch.cat(capture.q_chunks, dim=0) if capture.q_chunks else None
        rows = (
            torch.cat(capture.query_rows, dim=0)
            if capture.query_rows
            else None
        )
        latent_full = torch.cat(capture.k_chunks, dim=0)
        if (
            q_rows is None
            or rows is None
            or q_rows.shape[0] != sample_rows.shape[0]
            or latent_full.shape[0] != expected
        ):
            capture.invalid = True
            return False

        self._score_complete_layer(
            layer_id=layer_id,
            qsel=q_rows,
            rows=rows,
            latent_k=latent_full,
            softmax_scale=softmax_scale,
        )
        capture.k_chunks.clear()
        capture.q_chunks.clear()
        capture.query_rows.clear()
        capture.complete = True

        completed_generation = capture.generation
        if completed_generation <= 0 or any(
            not item.complete or item.generation != completed_generation
            for item in self._captures
        ):
            return False
        self._sealed = True
        self._max_ctx = expected
        return True

    def _reset_layer_capture(self, layer_id: int) -> None:
        """Discard an incomplete/old request and its staged layer statistics."""
        old = self._captures[layer_id]
        generation = old.generation
        self._captures[layer_id] = _LayerRequestCapture(generation=generation)
        # A previous candidate may have reached expected_context for this layer
        # before another layer proved incomplete.  Zero its staged reduction so
        # the next request cannot inherit any of that evidence.
        if self._g_mass is not None:
            self._g_mass[layer_id].zero_()
            self._g_query_rows[layer_id].zero_()
            self._g_window_counts[layer_id].zero_()
            self._g_sum[layer_id].zero_()
            self._g_sq[layer_id].zero_()
            self._g_min[layer_id].fill_(float("inf"))
            self._g_max[layer_id].zero_()
            self._g_count[layer_id].zero_()

    def _select_query_rows(self, context: int, device: torch.device) -> torch.Tensor:
        """Choose deterministic request-global rows before any chunk arrives."""
        min_visible = max(8, int(0.25 * context))
        n_samp = min(self.cfg.sample_queries, context)
        first_meaningful = min(context - 1, min_visible - 1)
        if context <= 2048 or n_samp < 8:
            rows = torch.linspace(
                first_meaningful, context - 1, steps=n_samp, device=device
            ).long()
        else:
            n_tail = max(1, n_samp // 4)
            n_uniform = n_samp - n_tail
            tail_start = max(first_meaningful, context - 1024)
            uniform_end = max(first_meaningful, tail_start - 1)
            uniform = torch.linspace(
                first_meaningful,
                uniform_end,
                steps=n_uniform,
                device=device,
            ).long()
            tail = torch.linspace(
                tail_start, context - 1, steps=n_tail, device=device
            ).long()
            rows = torch.cat((uniform, tail))
        return rows.unique(sorted=True)

    def _ensure_gpu_accumulators(self, device: torch.device) -> None:
        """Lazily allocate sync-free statistics on the profiling device."""
        if self._g_sum is not None and self._g_device == device:
            return
        L = self.cfg.num_layers
        self._g_sum = torch.zeros(L, dtype=torch.float64, device=device)
        self._g_sq = torch.zeros(L, dtype=torch.float64, device=device)
        self._g_min = torch.full(
            (L,), float("inf"), dtype=torch.float64, device=device
        )
        self._g_max = torch.zeros(L, dtype=torch.float64, device=device)
        self._g_count = torch.zeros(L, dtype=torch.float64, device=device)
        self._g_mass = torch.zeros(
            (L, self.cfg.num_heads, len(self.cfg.bin_edges)),
            dtype=torch.float64,
            device=device,
        )
        self._g_query_rows = torch.zeros(L, dtype=torch.float64, device=device)
        self._g_window_counts = torch.zeros(
            (L, self.cfg.num_heads, len(self.cfg.bin_edges)),
            dtype=torch.float64,
            device=device,
        )
        self._g_device = device

    @torch.no_grad()
    def _score_complete_layer(
        self,
        layer_id: int,
        qsel: torch.Tensor,
        rows: torch.Tensor,
        latent_k: torch.Tensor,
        softmax_scale: float,
    ) -> None:
        """Reduce sampled Q against the request's complete latent K stream."""
        device = qsel.device
        qsel = qsel.float()
        latent_k = latent_k.float()
        T = latent_k.shape[0]
        S, H, _ = qsel.shape

        coverage = float(self.cfg.coverage)
        min_visible = max(8, int(0.25 * T))
        self._ensure_gpu_accumulators(device)

        # Vectorized over sampled queries: build [S, H, T] logits and apply a
        # causal mask (key j visible to query i iff j <= i). Masked positions get
        # -inf so softmax ignores them. This avoids the per-query Python loop and
        # any GPU->CPU sync entirely.
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

        # Attention mass by causal query->key distance.  ``bucketize`` maps a
        # distance to the first inclusive upper-bound in ``bin_edges``.  Future
        # keys have exactly zero probability after the causal softmax, while
        # the ``keep`` mask drops early queries whose visible context is too
        # short to provide a meaningful locality measurement.
        distance = (rows[:, None] - key_pos[None, :]).clamp_min_(0)
        edges = self._edges.to(device=device, non_blocking=True)
        distance_bin = torch.bucketize(distance, edges, right=False)
        kept_probs = probs * keep[:, None, None].to(probs.dtype)
        query_head_mass = torch.zeros(
            (S, H, len(self.cfg.bin_edges)), dtype=probs.dtype, device=device
        )
        query_head_mass.scatter_add_(
            2,
            distance_bin[:, None, :].expand(-1, H, -1),
            kept_probs,
        )
        head_mass = query_head_mass.sum(dim=0)
        self._g_mass[layer_id].add_(head_mass.to(torch.float64))
        self._g_query_rows.index_add_(0, idx, layer_cnt.reshape(1))

        # Distribution of per-query coverage windows. This prevents a head
        # that is global for a minority of important tail queries from being
        # washed out by the average mass histogram.
        query_cum = torch.cumsum(query_head_mass, dim=-1)
        query_window_bin = (query_cum >= coverage).to(torch.int32).argmax(dim=-1)
        window_counts = torch.zeros(
            (H, len(self.cfg.bin_edges)), dtype=torch.float64, device=device
        )
        window_counts.scatter_add_(
            1,
            query_window_bin.transpose(0, 1),
            keep[:, None].expand(-1, H).transpose(0, 1).to(torch.float64),
        )
        self._g_window_counts[layer_id].add_(window_counts)
        self._g_last_sample_rows = rows

    @torch.no_grad()
    def _distance_stats_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a coherent CPU snapshot of the distance accumulators."""
        if self._g_mass is not None and self._g_query_rows is not None:
            return self._g_mass.cpu(), self._g_query_rows.cpu()
        return self._mass, self._query_rows

    @torch.no_grad()
    def _coverage_window(
        self,
        layer_id: int,
        head: int,
        *,
        target: Optional[float] = None,
        mass_by_bin: Optional[torch.Tensor] = None,
        query_rows: Optional[torch.Tensor] = None,
    ) -> float:
        """Smallest distance bin upper-bound covering ``coverage`` mass."""
        if mass_by_bin is None or query_rows is None:
            mass_by_bin, query_rows = self._distance_stats_cpu()
        counts = query_rows[layer_id].item()
        if counts <= 0:
            return float("inf")
        mass = mass_by_bin[layer_id, head] / counts  # avg mass per bin
        total = float(mass.sum().item())
        if total <= 0:
            return float("inf")
        cum = torch.cumsum(mass / total, dim=0)
        target = self.cfg.coverage if target is None else float(target)
        for b in range(cum.numel()):
            if cum[b].item() >= target:
                return float(self.cfg.bin_edges[b])
        return float(self.cfg.bin_edges[-1])

    @torch.no_grad()
    def _query_window_quantile(
        self,
        layer_id: int,
        head: int,
        quantile: float,
        *,
        window_counts: Optional[torch.Tensor] = None,
    ) -> float:
        """Quantile of the per-query coverage-window distribution."""
        if window_counts is None:
            if self._g_window_counts is None:
                return float("inf")
            window_counts = self._g_window_counts.cpu()
        counts = window_counts[layer_id, head]
        total = float(counts.sum().item())
        if total <= 0:
            return float("inf")
        cumulative = torch.cumsum(counts / total, dim=0)
        target = min(1.0, max(0.0, float(quantile)))
        for b in range(cumulative.numel()):
            if cumulative[b].item() >= target:
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
        mass_by_bin, query_rows = self._distance_stats_cpu()
        window_counts = (
            self._g_window_counts.cpu()
            if self._g_window_counts is not None
            else None
        )

        head_class: List[List[str]] = []
        head_distance: List[List[int]] = []
        sinks: List[List[int]] = []
        for layer in range(cfg.num_layers):
            row_class: List[str] = []
            row_dist: List[int] = []
            for head in range(cfg.num_heads):
                if layer < cfg.dense_prefix_layers or layer >= (
                    cfg.num_layers - cfg.dense_suffix_layers
                ):
                    row_class.append(HEAD_DENSE)
                    row_dist.append(-1)
                    continue
                w = self._query_window_quantile(
                    layer,
                    head,
                    cfg.query_window_quantile,
                    window_counts=window_counts,
                )
                if not math.isfinite(w):
                    w = self._coverage_window(
                        layer,
                        head,
                        mass_by_bin=mass_by_bin,
                        query_rows=query_rows,
                    )
                # ``w`` is a maximum zero-based distance, whereas FlashMLA's
                # topk_length is a token count.  A measured d=128 therefore
                # requires 129 visible tokens, not 128.
                required_tokens = (
                    float("inf") if not math.isfinite(w) else int(math.ceil(w)) + 1
                )
                if (
                    not math.isfinite(required_tokens)
                    or required_tokens >= global_thresh
                ):
                    row_class.append(HEAD_GLOBAL)
                    row_dist.append(-1)
                else:
                    win = int(math.ceil(required_tokens * cfg.window_safety))
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
            dense_suffix_layers=cfg.dense_suffix_layers,
        )

    @torch.no_grad()
    def distance_profile(self) -> Dict[str, object]:
        """Export head-specific range evidence, not only the final class.

        The configured coverage controls the production head policy, while
        d50/d90/d95/d99 make the measured attention range inspectable and let
        callers re-evaluate the local/global threshold without rerunning the
        expensive model prefill.
        """
        mass_by_bin, query_rows = self._distance_stats_cpu()
        coverages = (0.50, 0.90, 0.95, 0.99)
        windows: Dict[str, List[List[int]]] = {}
        for coverage in coverages:
            rows: List[List[int]] = []
            for layer in range(self.cfg.num_layers):
                row = []
                for head in range(self.cfg.num_heads):
                    value = self._coverage_window(
                        layer,
                        head,
                        target=coverage,
                        mass_by_bin=mass_by_bin,
                        query_rows=query_rows,
                    )
                    row.append(-1 if not math.isfinite(value) else int(value))
                rows.append(row)
            windows[f"d{int(round(coverage * 100))}"] = rows

        normalized_mass: List[List[List[float]]] = []
        for layer in range(self.cfg.num_layers):
            layer_rows = []
            for head in range(self.cfg.num_heads):
                mass = mass_by_bin[layer, head]
                total = float(mass.sum().item())
                if total <= 0:
                    layer_rows.append([0.0] * len(self.cfg.bin_edges))
                else:
                    layer_rows.append((mass / total).tolist())
            normalized_mass.append(layer_rows)

        per_query_windows: Dict[str, List[List[int]]] = {}
        window_counts = (
            self._g_window_counts.cpu()
            if self._g_window_counts is not None
            else None
        )
        for quantile in (0.50, 0.90, 0.95, 1.0):
            rows = []
            for layer in range(self.cfg.num_layers):
                row = []
                for head in range(self.cfg.num_heads):
                    value = self._query_window_quantile(
                        layer, head, quantile, window_counts=window_counts
                    )
                    row.append(-1 if not math.isfinite(value) else int(value))
                rows.append(row)
            per_query_windows[f"p{int(round(quantile * 100))}"] = rows

        sampled_rows = []
        if self._g_last_sample_rows is not None:
            sampled_rows = self._g_last_sample_rows.cpu().tolist()

        return {
            "format": "redknot_deepseek_v4_mla_distance_profile_v2",
            "num_layers": self.cfg.num_layers,
            "num_attention_heads": self.cfg.num_heads,
            "dense_prefix_layers": self.cfg.dense_prefix_layers,
            "dense_suffix_layers": self.cfg.dense_suffix_layers,
            "max_context": self.max_context,
            "sample_queries": self.cfg.sample_queries,
            "query_window_quantile_for_policy": self.cfg.query_window_quantile,
            "sampled_query_positions": sampled_rows,
            "query_rows_per_layer": query_rows.tolist(),
            "bin_edges": list(self.cfg.bin_edges),
            "coverage_windows": windows,
            "per_query_coverage_window_quantiles": per_query_windows,
            "normalized_mass_by_distance_bin": normalized_mass,
        }

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

    def export_distance_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.distance_profile(), f, indent=2)
            f.write("\n")
        logger.info("RedKnot MLA distance profile exported to %s", path)


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
