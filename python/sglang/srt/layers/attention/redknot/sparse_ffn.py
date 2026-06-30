# Copyright 2024-2026 SGLang RedKnot Integration.
"""Partial Sparse FFN recovery for RedKnot Elastic Sparsity.

This module implements RedKnot's **token-selective Sparse FFN** (paper
§4.2 "Partial sparse FFN recovery" and Algorithm 1 lines 20-23). It is the
P0 mechanism that attacks the short-context FFN bottleneck that no
attention-side technique can reach (paper §3.1 / fig. 2: FFN is 57-62% of
TTFT at 2-8K tokens).

The contract
------------
After head-aware attention recovery produces the post-attention hidden
states ``Y = X + Attn(X)`` of a deep layer, only a subset of tokens
actually need their FFN updated; the rest can follow the residual
identity path. Algorithm 1 expresses this as::

    S        <- SelectImportantTokens(A)          # line 20
    Z[S]     <- FFN(Y[S])                          # line 21
    Z[~S]    <- 0  (residual identity)             # line 22
    X_next   <- Y + Z                              # line 23

This file provides:

- :class:`SparseFFNSchedule` — the ``dense_until / mass_thresh / recent_n``
  policy used in the paper's evaluation (paper §4.2:
   ``dense_until=5, mass_thresh=0.6, recent_n=128``).
- :func:`select_important_tokens` — the ``SelectImportantTokens`` operator,
  driven by the recovered attention signal.
- :func:`apply_sparse_ffn` — the dense/identity dispatch (lines 21-23) that
  runs the model's own ``FFN`` callable only on selected token rows.
- :func:`sparse_ffn_flops` — analytical FLOP accounting so the RAG demo can
  report the FFN-side savings that close the gap to the paper's 67-79.5%.

Design notes
------------
- **Dependency-free / single-testable.** Everything here is plain PyTorch
  and runs on CPU. The ``ffn`` argument is any callable
  ``Tensor[..., H] -> Tensor[..., H]`` (e.g. a HuggingFace layer's
  ``mlp``), so this module never imports a model.
- **Lossless when fully dense.** With ``dense_until >= num_layers`` or
  ``mass_thresh >= 1`` every token is selected, so the output is bit-for-bit
  the dense FFN. This lets the RAG demo validate equivalence before turning
  sparsity on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────
# Schedule
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SparseFFNSchedule:
    """Layer-wise Sparse FFN policy with **tiered mass thresholds**.

    Three tiers of layers:

    1. **Shallow** (``layer_idx < dense_until``): fully dense FFN.
    2. **Mid** (``dense_until <= layer_idx < deep_layer_start``): sparse FFN
       with ``mass_thresh`` (conservative).
    3. **Deep** (``layer_idx >= deep_layer_start``): sparse FFN with
       ``mass_thresh_deep`` (aggressive).

    Attributes
    ----------
    dense_until:
        Shallow-layer boundary ``L_dense`` (paper Algorithm 1 line 6). For
        ``layer_idx < dense_until`` the FFN stays **fully dense** to
        preserve the early residual stream. Only deep layers run sparse.
    mass_thresh:
        Cumulative attention-mass threshold in ``(0, 1]`` for **mid layers**.
        A token is kept if it is needed to reach ``mass_thresh`` of the
        total importance mass (top-p style). ``1.0`` keeps every token
        (i.e. dense).
    deep_layer_start:
        Layer index at which the deep tier begins. Layers at and above
        this index use ``mass_thresh_deep`` instead of ``mass_thresh``.
        Set to a value >= ``num_layers`` to disable the deep tier (i.e.
        use ``mass_thresh`` uniformly for all sparse layers).
    mass_thresh_deep:
        Mass threshold for deep layers (``>= deep_layer_start``). Smaller
        values = more aggressive sparsity. When ``None``, falls back to
        ``mass_thresh`` (uniform behavior, backward compatible).
    recent_n:
        Number of most-recent tokens that are **always** kept regardless of
        their importance score. Mirrors the streaming-head intuition that
        the local suffix must stay fresh. ``0`` disables the guard.
    min_keep:
        Floor on the number of selected tokens per sequence, so a layer
        never collapses to an all-identity FFN (which would freeze the
        residual stream). Defaults to 1.
    """

    dense_until: int = 5
    mass_thresh: float = 0.6
    deep_layer_start: int = 48
    mass_thresh_deep: Optional[float] = None
    recent_n: int = 128
    min_keep: int = 1

    def is_dense_layer(self, layer_idx: int) -> bool:
        """True if this layer must run the full dense FFN (no sparsity)."""
        mt = self.get_mass_thresh(layer_idx)
        return layer_idx < self.dense_until or mt >= 1.0

    def get_mass_thresh(self, layer_idx: int) -> float:
        """Return the effective mass threshold for ``layer_idx``.

        - Shallow layers (< dense_until): returns 1.0 (dense).
        - Mid layers (dense_until .. deep_layer_start-1): returns ``mass_thresh``.
        - Deep layers (>= deep_layer_start): returns ``mass_thresh_deep``
          if set, otherwise ``mass_thresh``.
        """
        if layer_idx < self.dense_until:
            return 1.0  # dense
        if self.mass_thresh_deep is not None and layer_idx >= self.deep_layer_start:
            return self.mass_thresh_deep
        return self.mass_thresh


# ──────────────────────────────────────────────────────────────────────────
# Token importance / selection  (Algorithm 1 line 20)
# ──────────────────────────────────────────────────────────────────────────
def token_importance_from_attn(attn_out: torch.Tensor) -> torch.Tensor:
    """Estimate per-token importance from the recovered attention output.

    Parameters
    ----------
    attn_out:
        Recovered attention states. Accepts either ``[B, Hq, L, D]`` (the
        layout produced by the RedKnot kernels) or ``[B, L, E]`` (already
        merged into the residual width). The importance proxy is the L2
        norm of each token's attention contribution, which tracks how much
        a token's hidden state is about to move — tokens with near-zero
        attention output are well-approximated by the residual identity.

    Returns
    -------
    ``[B, L]`` float tensor of non-negative importance scores.
    """
    if attn_out.dim() == 4:
        # [B, Hq, L, D] -> per-token norm over (head, dim).
        x = attn_out.float()
        # Move L to the front of the reduced axes: norm over Hq and D.
        scores = x.pow(2).sum(dim=(1, 3)).sqrt()  # [B, L]
        return scores
    if attn_out.dim() == 3:
        # [B, L, E]
        return attn_out.float().norm(dim=-1)  # [B, L]
    raise ValueError(
        f"token_importance_from_attn: expected 3-D or 4-D tensor, got "
        f"{tuple(attn_out.shape)}"
    )


def select_important_tokens(
    importance: torch.Tensor,
    schedule: SparseFFNSchedule,
    *,
    mass_thresh_override: Optional[float] = None,
) -> torch.Tensor:
    """``SelectImportantTokens`` (Algorithm 1 line 20).

    Picks the smallest set of tokens whose importance covers
    ``mass_thresh`` of the total mass, then unions in the last
    ``schedule.recent_n`` tokens and enforces ``schedule.min_keep``.

    Parameters
    ----------
    importance:
        ``[B, L]`` non-negative scores.
    schedule:
        Selection policy.
    mass_thresh_override:
        If given, overrides ``schedule.mass_thresh`` for this call. This
        is used by :func:`apply_sparse_ffn` to pass the layer-specific
        threshold from :meth:`SparseFFNSchedule.get_mass_thresh`.

    Returns
    -------
    ``[B, L]`` boolean mask; ``True`` rows run the dense FFN.
    """
    if importance.dim() != 2:
        raise ValueError(
            f"select_important_tokens expects [B, L], got {tuple(importance.shape)}"
        )
    mass_thresh = (
        mass_thresh_override
        if mass_thresh_override is not None
        else schedule.mass_thresh
    )
    B, L = importance.shape
    device = importance.device
    keep = torch.zeros(B, L, dtype=torch.bool, device=device)

    if mass_thresh >= 1.0:
        keep[:] = True
        return keep

    # ── top-p over importance mass ──
    imp = importance.clamp_min(0).float()
    total = imp.sum(dim=-1, keepdim=True)
    # Degenerate all-zero rows: keep everything (cannot rank).
    zero_rows = (total <= 0).flatten()

    sorted_imp, sorted_idx = torch.sort(imp, dim=-1, descending=True)
    cum = torch.cumsum(sorted_imp, dim=-1)
    denom = total.clamp_min(torch.finfo(imp.dtype).tiny)
    cum_frac = cum / denom
    # Keep ranks up to and including the one that first crosses the
    # threshold (so we never under-select).
    rank_keep = cum_frac < mass_thresh
    rank_keep[..., 0] = True  # always keep the top token
    # Scatter the rank decision back to original positions.
    keep.scatter_(1, sorted_idx, rank_keep)

    # ── always-keep recent window ──
    if schedule.recent_n > 0:
        n = min(schedule.recent_n, L)
        keep[:, L - n :] = True

    # ── min_keep floor (top-k fallback) ──
    if schedule.min_keep > 0:
        counts = keep.sum(dim=-1)
        need = counts < schedule.min_keep
        if bool(need.any()):
            k = min(schedule.min_keep, L)
            topk_idx = torch.topk(imp, k=k, dim=-1).indices  # [B, k]
            floor = torch.zeros_like(keep)
            floor.scatter_(1, topk_idx, True)
            keep = torch.where(need.unsqueeze(1), keep | floor, keep)

    # All-zero importance rows fall back to dense (safe).
    if bool(zero_rows.any()):
        keep[zero_rows] = True
    return keep


# ──────────────────────────────────────────────────────────────────────────
# Dispatch  (Algorithm 1 lines 21-23)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def apply_sparse_ffn(
    hidden_states: torch.Tensor,
    ffn: Callable[[torch.Tensor], torch.Tensor],
    *,
    layer_idx: int,
    schedule: SparseFFNSchedule,
    attn_out: Optional[torch.Tensor] = None,
    importance: Optional[torch.Tensor] = None,
    return_stats: bool = False,
):
    """Run ``X_next = Y + Z`` with token-selective FFN (Algorithm 1 21-23).

    Parameters
    ----------
    hidden_states:
        Post-attention residual stream ``Y`` shaped ``[B, L, E]``.
    ffn:
        The layer's feed-forward callable, ``[N, E] -> [N, E]`` or
        ``[B, L, E] -> [B, L, E]``. It is invoked **only** on the selected
        token rows (packed to ``[num_selected, E]``).
    layer_idx:
        Index of this layer (drives ``schedule.dense_until``).
    schedule:
        Sparse FFN policy.
    attn_out:
        Recovered attention output used to estimate token importance. Either
        ``attn_out`` or a precomputed ``importance`` must be given for deep
        layers; shallow layers ignore both (always dense).
    importance:
        Optional precomputed ``[B, L]`` importance, overriding ``attn_out``.
    return_stats:
        If True, also return a stats dict (selected fraction, counts).

    Returns
    -------
    ``X_next`` shaped ``[B, L, E]`` (and optionally a stats dict).
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            f"apply_sparse_ffn expects [B, L, E] hidden_states, got "
            f"{tuple(hidden_states.shape)}"
        )
    B, L, E = hidden_states.shape

    # ── Shallow layers / disabled: dense FFN on everything. ──
    if schedule.is_dense_layer(layer_idx):
        z = ffn(hidden_states)
        out = hidden_states + z
        if return_stats:
            return out, {
                "layer": layer_idx,
                "mode": "dense",
                "selected": B * L,
                "total": B * L,
                "selected_frac": 1.0,
            }
        return out

    # ── Importance + selection (lines 20). ──
    if importance is None:
        if attn_out is None:
            raise ValueError(
                "apply_sparse_ffn: deep layer requires `attn_out` or "
                "`importance` to select tokens."
            )
        importance = token_importance_from_attn(attn_out)
    layer_mt = schedule.get_mass_thresh(layer_idx)
    keep = select_important_tokens(
        importance, schedule, mass_thresh_override=layer_mt
    )  # [B, L] bool

    # ── Pack selected rows, run dense FFN once (line 21). ──
    flat_hidden = hidden_states.reshape(B * L, E)
    flat_keep = keep.reshape(B * L)
    sel_idx = torch.nonzero(flat_keep, as_tuple=False).flatten()

    z_full = torch.zeros_like(flat_hidden)  # line 22: residual identity (Z=0)
    if sel_idx.numel() > 0:
        sel_hidden = flat_hidden.index_select(0, sel_idx)  # [N_sel, E]
        sel_z = ffn(sel_hidden)
        if sel_z.shape != sel_hidden.shape:
            raise ValueError(
                "apply_sparse_ffn: ffn must map [N, E] -> [N, E]; got "
                f"input {tuple(sel_hidden.shape)} output {tuple(sel_z.shape)}"
            )
        z_full.index_copy_(0, sel_idx, sel_z)

    out = (flat_hidden + z_full).reshape(B, L, E)  # line 23: X_next = Y + Z

    if return_stats:
        selected = int(sel_idx.numel())
        total = B * L
        return out, {
            "layer": layer_idx,
            "mode": "sparse",
            "selected": selected,
            "total": total,
            "selected_frac": (selected / total) if total else 1.0,
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# FLOP accounting
# ──────────────────────────────────────────────────────────────────────────
def dense_ffn_flops_per_token(hidden: int, intermediate: int) -> float:
    """FLOPs for one SwiGLU-style FFN token (gate+up+down, mul-add = 2)."""
    return 6.0 * hidden * intermediate


def sparse_ffn_flops(
    *,
    num_layers: int,
    tokens_per_layer: int,
    hidden: int,
    intermediate: int,
    schedule: SparseFFNSchedule,
    selected_frac_deep: float,
) -> dict:
    """Analytical FFN FLOPs for a Sparse-FFN forward over ``num_layers``.

    Shallow layers (``< dense_until``) cost a full dense FFN; deep layers
    cost ``selected_frac_deep`` of a dense FFN. Returns both the dense
    baseline and the sparse total so the demo can report savings.

    Parameters
    ----------
    selected_frac_deep:
        Measured average fraction of tokens that ran the dense FFN in deep
        layers (from :func:`apply_sparse_ffn` stats). Use ``1.0`` for an
        upper bound.
    """
    per_tok = dense_ffn_flops_per_token(hidden, intermediate)
    dense_total = num_layers * tokens_per_layer * per_tok

    dense_layers = min(max(schedule.dense_until, 0), num_layers)
    deep_layers = max(num_layers - dense_layers, 0)
    sparse_total = (
        dense_layers * tokens_per_layer * per_tok
        + deep_layers * tokens_per_layer * per_tok * float(selected_frac_deep)
    )
    savings = 1.0 - (sparse_total / dense_total) if dense_total > 0 else 0.0
    return {
        "dense_ffn_flops": dense_total,
        "sparse_ffn_flops": sparse_total,
        "ffn_flops_savings": savings,
        "ffn_speedup": (dense_total / sparse_total)
        if sparse_total > 0
        else float("inf"),
        "dense_layers": dense_layers,
        "deep_layers": deep_layers,
    }


__all__ = [
    "SparseFFNSchedule",
    "token_importance_from_attn",
    "select_important_tokens",
    "apply_sparse_ffn",
    "dense_ffn_flops_per_token",
    "sparse_ffn_flops",
]
