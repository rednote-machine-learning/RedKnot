# REDKNOT-MODEL: per-request sparse MoE policy, not a native expert executor.
"""Portable configuration and request-local RedKnot sparse-MoE state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REDKNOT_SPARSE_MOE_STATE_KEY = "redknot_sparse_moe"
VALID_UNTIL_UNBOUNDED = 1 << 30


@dataclass(frozen=True)
class RedKnotSparseMoEPolicy:
    """Immutable, config-driven RedKnot sparse-MoE policy.

    Built once from injected config (see ``from_config``) and shared read-only
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
    def from_config(cls, config) -> RedKnotSparseMoEPolicy:
        """Construct a policy from injected config, tolerating missing attributes.

        Returns a disabled policy when the feature flag is absent or false so
        that callers can unconditionally build the policy and check ``enabled``.
        """
        if config is None:
            return cls(enabled=False)

        def _get(name, default):
            if isinstance(config, Mapping):
                return config.get(name, default)
            return getattr(config, name, default)

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

    Passed explicitly by the native adapter; not stored in a global or engine batch.
    Never a module global (design rule #8). All tensors are expressed in the
    logical token order produced right after ``prepare_mlp()`` for the current
    forward; if a later layout change cannot be proven consistent, callers must
    fall back to dense rather than reuse a stale mask.
    """

    # Data in logical token order (post-prepare_mlp layout).
    token_scores: torch.Tensor | None = None
    routed_keep_mask: torch.Tensor | None = None
    protected_mask: torch.Tensor | None = None

    # Request boundaries.
    cu_seqlens: torch.Tensor | None = None
    request_ids: torch.Tensor | None = None

    # Lifecycle: the mask produced at ``source_layer_id`` is reusable by
    # linear-attention layers up to (and including) ``valid_until_layer_id``.
    source_layer_id: int = -1
    valid_until_layer_id: int = -1
    score_kind: str = "unknown"

    # Layout tracking.
    logical_token_ids: torch.Tensor | None = None
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
        token_scores: torch.Tensor | None = None,
        protected_mask: torch.Tensor | None = None,
    ) -> None:
        self.routed_keep_mask = routed_keep_mask
        self.source_layer_id = source_layer_id
        self.valid_until_layer_id = valid_until_layer_id
        self.score_kind = score_kind
        self.token_scores = token_scores
        self.protected_mask = protected_mask
        self.num_tokens = int(routed_keep_mask.shape[0])
