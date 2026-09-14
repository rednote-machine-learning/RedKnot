# REDKNOT-MODEL: sparse-MoE selector and explicit request-local mask resolution.
"""RedKnot sparse-MoE tensor policy; native router/expert execution stays native.

Selection math is extracted unchanged. The resolver consumes explicit request
context, prefill mode and layout generation instead of an SGLang ForwardBatch.
A layout-generation mismatch refuses reuse even when row counts match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .sparse_moe_policy import RedKnotSparseMoEPolicy, RedKnotTokenPolicyContext

logger = logging.getLogger(__name__)


def validate_mask_alignment(
    hidden_states: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    strict: bool = False,
) -> bool:
    """Structural, GPU-only alignment check for a routed keep mask.

    Fast path (always): shape / dtype / device must match the hidden states.
    No ``.item()`` or host sync is performed here so it is safe on the hot path.

    ``strict`` adds a contiguity assertion for debug builds.
    """
    if keep_mask is None or hidden_states is None:
        return False
    if hidden_states.ndim != 2:
        return False
    if keep_mask.ndim != 1:
        return False
    if keep_mask.shape[0] != hidden_states.shape[0]:
        return False
    if keep_mask.dtype != torch.bool:
        return False
    if keep_mask.device != hidden_states.device:
        return False
    if strict and not keep_mask.is_contiguous():
        return False
    return True


def resolve_routed_keep_mask(
    context: RedKnotTokenPolicyContext | None,
    layer_id: int,
    post_prepare_mlp_hidden_states: torch.Tensor,
    *,
    policy: RedKnotSparseMoEPolicy | None = None,
    is_prefill: bool,
    layout_version: int,
) -> torch.BoolTensor | None:
    """Resolve the routed keep mask aligned to the post-``prepare_mlp()`` layout.

    Contract (design §7):
      * returns a ``[N]`` bool mask on the same device as
        ``post_prepare_mlp_hidden_states``, or
      * returns ``None`` to request a dense fallback (never guesses / broadcasts).

    Dense fallback is returned when:
      * policy is disabled or the layer is dense (``layer < dense_until_layer``);
      * prefill_only and this is not an extend/mixed forward;
      * no RedKnot context / mask is present for this layer;
      * the mask fails the structural alignment check;
      * the mask carries NaN/Inf-derived garbage (guarded upstream by selector).

    This function is intentionally *pure* w.r.t. the executor: it only reads
    already-computed context state. Score computation / selection is the
    selector's job (Phase 2); tests may inject a mask directly (Phase 0, §5.2).
    """
    if policy is None:
        return None
    if not policy.enabled:
        return None
    if layer_id < policy.dense_until_layer:
        return None

    if policy.prefill_only and not is_prefill:
        return None

    ctx = context
    if ctx is None:
        return None
    if type(layout_version) is not int or ctx.layout_version != layout_version:
        return None
    if not ctx.mask_valid_for_layer(layer_id):
        return None

    mask = ctx.routed_keep_mask
    # Align mask to *this* layer's hidden layout. If we cannot prove the layout
    # matches, we MUST fall back to dense (design rule #10).
    if not validate_mask_alignment(post_prepare_mlp_hidden_states, mask):
        logger.debug(
            "[RedKnot] layer %d: mask alignment failed; dense fallback", layer_id
        )
        return None
    return mask


def _per_request_mean_ratio_mask(
    scores: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor:
    """Keep token if ``score >= alpha * per-request-mean(score)`` (spec §6.2).

    ``cu_seqlens`` are cumulative request boundaries of length ``num_req + 1``
    over the ``[N]`` logical token axis. When it is ``None`` (or degenerate) the
    whole batch is treated as one request. The computation is GPU-only and does
    not force a host sync (no ``.item()`` on the hot path).

    Cross-request means are forbidden (spec §6.2): each request uses its own
    mean so a long request cannot starve a short one.
    """
    n = scores.shape[0]
    device = scores.device
    scores = scores.float().clamp_min(0)

    if cu_seqlens is None or cu_seqlens.numel() < 3:
        # Single request (or unknown boundaries): one global mean over this batch.
        mean = scores.mean().clamp_min(torch.finfo(torch.float32).tiny)
        return scores >= alpha * mean

    cu = cu_seqlens.to(device=device, dtype=torch.long)
    # Segment id per token via searchsorted on the right boundaries.
    # boundaries: cu[1:] are the exclusive ends of each request.
    seg_id = torch.searchsorted(cu[1:], torch.arange(n, device=device), right=True)
    num_req = cu.numel() - 1
    seg_id = seg_id.clamp_max(num_req - 1)

    sums = torch.zeros(num_req, device=device, dtype=torch.float32)
    sums.index_add_(0, seg_id, scores)
    counts = torch.zeros(num_req, device=device, dtype=torch.float32)
    counts.index_add_(0, seg_id, torch.ones_like(scores))
    means = sums / counts.clamp_min(1.0)
    per_tok_mean = means.index_select(0, seg_id)
    per_tok_mean = per_tok_mean.clamp_min(torch.finfo(torch.float32).tiny)
    return scores >= alpha * per_tok_mean


def _protect_recent_per_request(
    keep: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    recent_tokens: int,
) -> torch.Tensor:
    """Force-keep the most recent ``recent_tokens`` of each request (spec §6.3)."""
    if recent_tokens <= 0:
        return keep
    n = keep.shape[0]
    device = keep.device
    if cu_seqlens is None or cu_seqlens.numel() < 3:
        if recent_tokens >= n:
            keep[:] = True
        else:
            keep[n - recent_tokens :] = True
        return keep
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    ends = cu[1:]  # exclusive end of each request
    starts = cu[:-1]
    pos = torch.arange(n, device=device)
    seg_id = torch.searchsorted(ends, pos, right=True).clamp_max(cu.numel() - 2)
    seg_end = ends.index_select(0, seg_id)
    seg_start = starts.index_select(0, seg_id)
    # recent window start per token's request, clamped to the request start.
    recent_start = torch.clamp(seg_end - recent_tokens, min=0)
    recent_start = torch.maximum(recent_start, seg_start)
    keep = keep | (pos >= recent_start)
    return keep


def _enforce_min_keep_per_request(
    keep: torch.Tensor,
    scores: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    min_keep_tokens: int,
    min_keep_ratio: float,
) -> torch.Tensor:
    """Guarantee a per-request floor on kept tokens (spec §6.3 min_keep_*).

    For each request, if fewer than ``max(min_keep_tokens, ratio*len)`` tokens
    are kept, promote the highest-scoring tokens until the floor is met.
    """
    if min_keep_tokens <= 0 and min_keep_ratio <= 0:
        return keep
    device = keep.device  # noqa: F841 - retained source expression for provenance
    n = keep.shape[0]
    if cu_seqlens is None or cu_seqlens.numel() < 3:
        segments = [(0, n)]
    else:
        cu = cu_seqlens.to(device="cpu", dtype=torch.long).tolist()
        segments = list(zip(cu[:-1], cu[1:]))
    for s, e in segments:
        length = e - s
        if length <= 0:
            continue
        floor = max(int(min_keep_tokens), int(round(min_keep_ratio * length)))
        floor = min(floor, length)
        if floor <= 0:
            continue
        seg_keep = keep[s:e]
        cur = int(seg_keep.sum())
        if cur >= floor:
            continue
        need = floor - cur
        seg_scores = scores[s:e].clone()
        seg_len = seg_scores.shape[0]
        if seg_len == 0:
            continue
        seg_scores[seg_keep] = float("-inf")  # exclude already-kept
        k = min(int(need), seg_len)
        if k <= 0:
            continue
        _, top = torch.topk(seg_scores, k=k, largest=True, sorted=False)
        seg_keep[top] = True
        keep[s:e] = seg_keep
    return keep


@dataclass
class RedKnotSelectionResult:
    keep_mask: torch.Tensor
    keep_ratio: float
    dense_fallback: bool
    reason: str = ""


def build_routed_keep_mask(
    scores: torch.Tensor,
    *,
    policy: RedKnotSparseMoEPolicy,
    cu_seqlens: torch.Tensor | None = None,
    protected_mask: torch.Tensor | None = None,
) -> RedKnotSelectionResult:
    """Build the final routed keep mask from token scores (spec §6.2 / §6.3).

    Pipeline:
      1. per-request mean-ratio threshold,
      2. union with protection sets (recent, min-keep floor, explicit protected),
      3. if the resulting keep ratio exceeds ``dense_fallback_keep_ratio`` the
         request set is not worth compacting -> request dense fallback.

    Returns a :class:`RedKnotSelectionResult`. ``keep_mask`` is only meaningful
    when ``dense_fallback`` is ``False``.
    """
    n = scores.shape[0]
    if n == 0:
        return RedKnotSelectionResult(
            keep_mask=torch.zeros(0, dtype=torch.bool, device=scores.device),
            keep_ratio=1.0,
            dense_fallback=True,
            reason="empty",
        )

    if not torch.isfinite(scores).all():
        return RedKnotSelectionResult(
            keep_mask=torch.ones(n, dtype=torch.bool, device=scores.device),
            keep_ratio=1.0,
            dense_fallback=True,
            reason="nonfinite_scores",
        )

    keep = _per_request_mean_ratio_mask(scores, cu_seqlens, policy.alpha)
    keep = _protect_recent_per_request(keep, cu_seqlens, policy.recent_tokens)
    if protected_mask is not None and protected_mask.shape == keep.shape:
        keep = keep | protected_mask.to(keep.device, torch.bool)
    keep = _enforce_min_keep_per_request(
        keep,
        scores,
        cu_seqlens,
        policy.min_keep_tokens,
        policy.min_keep_ratio,
    )

    keep_ratio = float(keep.float().mean())
    if keep_ratio > policy.dense_fallback_keep_ratio:
        return RedKnotSelectionResult(
            keep_mask=keep,
            keep_ratio=keep_ratio,
            dense_fallback=True,
            reason="keep_ratio_above_dense_fallback",
        )
    return RedKnotSelectionResult(
        keep_mask=keep,
        keep_ratio=keep_ratio,
        dense_fallback=False,
        reason="sparse",
    )


def token_scores_from_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
    """Activation-norm importance proxy aligned to the current hidden layout.

    Used as the default/fallback importance signal. Because it is computed from
    the exact tensor the MoE will consume, it is guaranteed layout-aligned
    (spec §16.4). The full-attention query-conditioned mass (spec §6.1) may be
    substituted when it can be proven to share this layout.
    """
    return hidden_states.detach().float().norm(dim=-1)
