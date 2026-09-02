"""RedKnot Token-Sparse MoE runtime support for Qwen3.5-397B-A17B.

This module implements the *specification-grade* runtime pieces described in
``RedKnot_Qwen3_5_397B_Sparse_MoE_Technical_Design.md``:

* :class:`RedKnotSparseMoEPolicy` -- immutable, config-driven policy built from
  server args. No numbers such as 512 / 10 / 60 / 1024 are hard-coded here; the
  layer boundary and selector knobs are all configurable.
* :class:`RedKnotTokenPolicyContext` -- **request/batch-local** runtime state
  carrying token scores, the routed keep mask, request boundaries and layout
  metadata. It is meant to live on
  ``forward_batch.model_specific_states["redknot_sparse_moe"]`` and never in a
  module global or a Python closure (design rule #8 / §7).
* :func:`resolve_routed_keep_mask` -- the single entry point the decoder layer
  calls after ``prepare_mlp()``. It returns a ``[num_local_tokens]`` bool mask
  that is *aligned to the post-``prepare_mlp()`` row layout*, or ``None`` to
  request a dense fallback. It must never guess or silently broadcast (§7).
* :func:`validate_mask_alignment` -- a cheap, GPU-only structural check used on
  the hot path plus a stricter debug check.

The actual sparse execution (compact -> router -> experts -> scatter-add) lives
in :mod:`sglang.srt.models.qwen2_moe` on ``Qwen2MoeSparseMoeBlock`` so it can
reuse the native TopK / experts / collective code paths. This module only owns
policy + mask resolution so the *selector* stays decoupled from the *executor*
(§6.1: "executor only consumes a bool mask").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Key under which the per-batch RedKnot context is stored on ForwardBatch.
REDKNOT_SPARSE_MOE_STATE_KEY = "redknot_sparse_moe"

# Sentinel meaning "this mask is valid for every layer at or after source".
VALID_UNTIL_UNBOUNDED = 1 << 30


@dataclass(frozen=True)
class RedKnotSparseMoEPolicy:
    """Immutable, config-driven RedKnot sparse-MoE policy.

    Built once from server args (see ``from_server_args``) and shared read-only
    across layers. All layer-boundary / selector knobs are configurable; nothing
    model-specific is hard-coded.
    """

    enabled: bool = False
    prefill_only: bool = True
    dense_until_layer: int = 24
    selector_type: str = "mean_ratio"
    alpha: float = 0.3
    recent_tokens: int = 256
    min_keep_tokens: int = 128
    min_keep_ratio: float = 0.20
    dense_fallback_keep_ratio: float = 0.85
    fallback_mode: str = "shared_only"

    @classmethod
    def from_server_args(cls, server_args) -> "RedKnotSparseMoEPolicy":
        """Construct a policy from server args, tolerating missing attributes.

        Returns a disabled policy when the feature flag is absent or false so
        that callers can unconditionally build the policy and check ``enabled``.
        """
        if server_args is None:
            return cls(enabled=False)

        def _get(name, default):
            return getattr(server_args, name, default)

        enabled = bool(_get("enable_redknot_sparse_moe", False))
        if not enabled:
            return cls(enabled=False)

        return cls(
            enabled=True,
            prefill_only=bool(_get("redknot_moe_prefill_only", True)),
            dense_until_layer=int(_get("redknot_moe_dense_until_layer", 24)),
            selector_type=str(_get("redknot_moe_selector", "mean_ratio")),
            alpha=float(_get("redknot_moe_alpha", 0.3)),
            recent_tokens=int(_get("redknot_moe_recent_tokens", 256)),
            min_keep_tokens=int(_get("redknot_moe_min_keep_tokens", 128)),
            min_keep_ratio=float(_get("redknot_moe_min_keep_ratio", 0.20)),
            dense_fallback_keep_ratio=float(
                _get("redknot_moe_dense_fallback_ratio", 0.85)
            ),
            fallback_mode=str(_get("redknot_moe_fallback_mode", "shared_only")),
        )

    def layer_is_sparse_eligible(self, layer_id: int) -> bool:
        """Whether ``layer_id`` may run sparse (before runtime/mask checks)."""
        return self.enabled and layer_id >= self.dense_until_layer


@dataclass
class RedKnotTokenPolicyContext:
    """Request/batch-local RedKnot runtime state.

    Stored on ``forward_batch.model_specific_states[REDKNOT_SPARSE_MOE_STATE_KEY]``.
    Never a module global (design rule #8). All tensors are expressed in the
    logical token order produced right after ``prepare_mlp()`` for the current
    forward; if a later layout change cannot be proven consistent, callers must
    fall back to dense rather than reuse a stale mask.
    """

    # Data in logical token order (post-prepare_mlp layout).
    token_scores: Optional[torch.Tensor] = None
    routed_keep_mask: Optional[torch.Tensor] = None
    protected_mask: Optional[torch.Tensor] = None

    # Request boundaries.
    cu_seqlens: Optional[torch.Tensor] = None
    request_ids: Optional[torch.Tensor] = None

    # Lifecycle: the mask produced at ``source_layer_id`` is reusable by
    # linear-attention layers up to (and including) ``valid_until_layer_id``.
    source_layer_id: int = -1
    valid_until_layer_id: int = -1
    score_kind: str = "unknown"

    # Layout tracking.
    logical_token_ids: Optional[torch.Tensor] = None
    layout_version: int = 0

    # Number of rows the mask was built for; used for a cheap alignment guard.
    num_tokens: int = -1

    def mask_valid_for_layer(self, layer_id: int) -> bool:
        if self.routed_keep_mask is None:
            return False
        if self.source_layer_id < 0:
            return False
        return self.source_layer_id <= layer_id <= self.valid_until_layer_id

    def set_mask(
        self,
        *,
        routed_keep_mask: torch.Tensor,
        source_layer_id: int,
        valid_until_layer_id: int,
        score_kind: str,
        token_scores: Optional[torch.Tensor] = None,
        protected_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.routed_keep_mask = routed_keep_mask
        self.source_layer_id = source_layer_id
        self.valid_until_layer_id = valid_until_layer_id
        self.score_kind = score_kind
        self.token_scores = token_scores
        self.protected_mask = protected_mask
        self.num_tokens = int(routed_keep_mask.shape[0])


def get_context(forward_batch) -> Optional[RedKnotTokenPolicyContext]:
    """Read the RedKnot context off a ForwardBatch, or ``None``."""
    if forward_batch is None:
        return None
    states = getattr(forward_batch, "model_specific_states", None)
    if not states:
        return None
    return states.get(REDKNOT_SPARSE_MOE_STATE_KEY)


def ensure_context(forward_batch) -> Optional[RedKnotTokenPolicyContext]:
    """Get-or-create the RedKnot context on a ForwardBatch.

    Returns ``None`` if the ForwardBatch cannot host model-specific states.
    """
    if forward_batch is None:
        return None
    states = getattr(forward_batch, "model_specific_states", None)
    if states is None:
        states = {}
        try:
            forward_batch.model_specific_states = states
        except Exception:
            return None
    ctx = states.get(REDKNOT_SPARSE_MOE_STATE_KEY)
    if ctx is None:
        ctx = RedKnotTokenPolicyContext()
        states[REDKNOT_SPARSE_MOE_STATE_KEY] = ctx
    return ctx


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
    forward_batch,
    layer_id: int,
    post_prepare_mlp_hidden_states: torch.Tensor,
    *,
    policy: Optional[RedKnotSparseMoEPolicy] = None,
) -> Optional[torch.BoolTensor]:
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

    if policy.prefill_only:
        fmode = getattr(forward_batch, "forward_mode", None)
        if fmode is None:
            return None
        # decode / idle stay dense; only extend-style prefill is eligible.
        is_extend = getattr(
            fmode, "is_extend_or_draft_extend_or_mixed", None
        )
        if is_extend is None or not is_extend():
            return None

    ctx = get_context(forward_batch)
    if ctx is None:
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


# --------------------------------------------------------------------------- #
# Phase 2: per-request mean-ratio selector + protection rules (spec §6).       #
# --------------------------------------------------------------------------- #
def _per_request_mean_ratio_mask(
    scores: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
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
    cu_seqlens: Optional[torch.Tensor],
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
    cu_seqlens: Optional[torch.Tensor],
    min_keep_tokens: int,
    min_keep_ratio: float,
) -> torch.Tensor:
    """Guarantee a per-request floor on kept tokens (spec §6.3 min_keep_*).

    For each request, if fewer than ``max(min_keep_tokens, ratio*len)`` tokens
    are kept, promote the highest-scoring tokens until the floor is met.
    """
    if min_keep_tokens <= 0 and min_keep_ratio <= 0:
        return keep
    device = keep.device
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
    cu_seqlens: Optional[torch.Tensor] = None,
    protected_mask: Optional[torch.Tensor] = None,
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
