"""Training-free, token-level adaptive Top-K for DeepSeek-V4 MoE.

The native router still produces its full Top-K candidate set.  This module
removes tail assignments whose cumulative *mixing* weight is outside a
configured mass threshold.  Dynamic per-token K keeps the original tensor
shape and masks filtered routes with expert id ``-1``.  When the threshold
mathematically guarantees one uniform K for every token, a physical compact
path returns rank-2 id/weight tensors with that smaller K so Marlin also
shrinks its activation, SwiGLU, down-projection, and reduction workspaces.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch


_LOGGED_LAYERS: set[tuple[int, float, Tuple[int, ...]]] = set()
_STATS_LOGGED_LAYERS: set[int] = set()
_COMBINED_EXECUTION_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)


def _strict_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be exactly 0 or 1, got {value!r}")
    return value == "1"


def _mass_threshold() -> float:
    raw = os.environ.get("REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS", "0.90")
    value = float(raw)
    if not (0.0 < value <= 1.0):
        raise ValueError(
            "REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS must be in (0, 1], "
            f"got {raw!r}"
        )
    return value


def _allowed_k(native_topk: int) -> Tuple[int, ...]:
    raw = os.environ.get("REDKNOT_ADAPTIVE_TOPK_BUCKETS", "4,5,6")
    try:
        values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError(
            "REDKNOT_ADAPTIVE_TOPK_BUCKETS must be comma-separated integers"
        ) from exc
    if not values or values[-1] != native_topk:
        raise ValueError(
            "REDKNOT_ADAPTIVE_TOPK_BUCKETS must be nonempty and end at the "
            f"native Top-K={native_topk}, got {values!r}"
        )
    if values[0] < 1 or any(k > native_topk for k in values):
        raise ValueError(
            f"adaptive Top-K buckets must lie in [1, {native_topk}], got {values!r}"
        )
    return values


def _round_up_k(required_k: torch.Tensor, allowed: Tuple[int, ...]) -> torch.Tensor:
    rounded = torch.full_like(required_k, allowed[-1])
    for k in reversed(allowed):
        rounded = torch.where(required_k <= k, k, rounded)
    return rounded


def _uniform_physical_k(
    native_topk: int, mass: float, allowed: Tuple[int, ...]
) -> Optional[int]:
    """Return a statically guaranteed uniform K, otherwise ``None``.

    For non-negative weights sorted in descending order, the first K of N
    entries always contain at least K/N of the normalized mass.  Therefore a
    threshold no larger than ``min_allowed / native_topk`` can never select a
    bucket wider than ``min_allowed``.  Since bucket rounding cannot select a
    narrower K, every token is exactly ``min_allowed`` and can share one
    physically compact tensor shape without a device-to-host predicate.
    """

    if not _strict_flag("REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION", "1"):
        return None
    minimum = int(allowed[0])
    if minimum >= native_topk:
        return None
    # The tiny tolerance only absorbs decimal parsing error at exact rational
    # boundaries such as 0.5 == 3/6; it does not broaden the policy.
    if mass <= (minimum / native_topk) + 1e-12:
        return minimum
    return None


def _request_uses_combined_redknot(forward_batch) -> bool:
    """Return true only when every packed request is a combined RedKnot plan.

    Adaptive routing changes model numerics, so a native dense request must not
    inherit it merely because the server also hosts RedKnot requests.  Waves in
    the benchmark are mode-homogeneous; rejecting mixed/partial plan lists
    keeps that contract fail-closed if a scheduler ever combines modes.
    """

    if forward_batch is None:
        return False
    plans = getattr(forward_batch, "redknot_reuse_plan", None)
    if not plans:
        return False
    for plan in plans:
        if plan is None:
            return False
        if not isinstance(plan, dict):
            raise RuntimeError(
                "plan-scoped adaptive Top-K requires mapping request plans"
            )
        if plan.get("mode") not in {"snapshot", "restore"}:
            return False
        if plan.get("mla_off_execution_profile") != _COMBINED_EXECUTION_PROFILE:
            return False
    return True


def maybe_apply_dsv4_adaptive_topk(
    *,
    layer_id: int,
    num_hidden_layers: int,
    native_routed_topk: int,
    num_fused_shared_experts: int,
    is_hash_layer: bool,
    hidden_states: torch.Tensor,
    topk_output,
    forward_batch=None,
) -> Optional[object]:
    """Return a route-masked TopKOutput, or ``None`` when policy is inactive.

    No tensor value is copied to the host on the normal path.  The returned
    weights are the original mixing weights with omitted assignments set to
    zero; retained weights are deliberately *not* renormalized.
    """

    if not _strict_flag("REDKNOT_ADAPTIVE_TOPK", "0"):
        return None
    plan_scoped = _strict_flag("REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED", "0")
    if plan_scoped and not _request_uses_combined_redknot(forward_batch):
        return None

    dense_prefix = int(os.environ.get("REDKNOT_ADAPTIVE_TOPK_DENSE_PREFIX_LAYERS", "3"))
    dense_suffix = int(os.environ.get("REDKNOT_ADAPTIVE_TOPK_DENSE_SUFFIX_LAYERS", "3"))
    if dense_prefix < 0 or dense_suffix < 0:
        raise ValueError("adaptive Top-K dense layer fences must be non-negative")
    if is_hash_layer or layer_id < dense_prefix or layer_id >= num_hidden_layers - dense_suffix:
        return None

    min_tokens = int(os.environ.get("REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS", "512"))
    if min_tokens < 1:
        raise ValueError("REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS must be positive")
    if int(hidden_states.shape[0]) < min_tokens:
        return None

    if num_fused_shared_experts != 0:
        raise RuntimeError(
            "adaptive routed Top-K currently requires shared-expert fusion disabled"
        )

    weights = topk_output.topk_weights
    ids = topk_output.topk_ids
    if weights.ndim != 2 or ids.shape != weights.shape:
        raise RuntimeError("adaptive Top-K requires matching rank-2 id/weight tensors")
    if int(weights.shape[1]) != native_routed_topk:
        raise RuntimeError(
            "adaptive Top-K requires the native routed candidate width, got "
            f"{int(weights.shape[1])}, expected {native_routed_topk}"
        )

    mass = _mass_threshold()
    allowed = _allowed_k(native_routed_topk)

    # TopK may return candidate slots in kernel order.  Rank by the actual
    # mixing weights, never by the bias-adjusted expert-choice score.
    sorted_weights, sorted_slots = torch.sort(weights, dim=1, descending=True)
    compact_k = _uniform_physical_k(native_routed_topk, mass, allowed)
    if compact_k is not None:
        # Preserve the native candidate-slot order among selected experts.
        # That makes the final K reduction as close as possible to the masked
        # width-N path while still returning truly compact [tokens, K] tensors.
        compact_slots = torch.sort(sorted_slots[:, :compact_k], dim=1).values
        compact_ids = torch.gather(ids, 1, compact_slots).contiguous()
        compact_weights = torch.gather(weights, 1, compact_slots).contiguous()
        compact_ids._sglang_moe_adaptive_topk = True
        compact_ids._sglang_moe_physical_topk = compact_k

        key = (layer_id, mass, allowed)
        if key not in _LOGGED_LAYERS:
            _LOGGED_LAYERS.add(key)
            print(
                "[REDKNOT_ADAPTIVE_TOPK] "
                f"layer={layer_id}/{num_hidden_layers} native_k={native_routed_topk} "
                f"mass={mass:.6f} buckets={','.join(map(str, allowed))} "
                f"min_tokens={min_tokens} renormalize=false "
                f"plan_scoped={str(plan_scoped).lower()} "
                f"compact_marlin=physical_width_{compact_k}",
                file=sys.stderr,
                flush=True,
            )

        if _strict_flag("REDKNOT_ADAPTIVE_TOPK_LOG_FIRST_HISTOGRAM", "0"):
            if layer_id not in _STATS_LOGGED_LAYERS:
                _STATS_LOGGED_LAYERS.add(layer_id)
                print(
                    "[REDKNOT_ADAPTIVE_TOPK_HIST] "
                    f"layer={layer_id} tokens={weights.shape[0]} "
                    f"k{compact_k}:{weights.shape[0]}",
                    file=sys.stderr,
                    flush=True,
                )

        return topk_output._replace(
            topk_ids=compact_ids, topk_weights=compact_weights
        )

    positive = sorted_weights.clamp_min(0)
    denom = positive.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(positive.dtype).tiny
    )
    cumulative = torch.cumsum(positive / denom, dim=1)
    required_k = torch.argmax((cumulative >= mass).to(torch.int32), dim=1) + 1
    selected_k = _round_up_k(required_k, allowed)

    ranks = torch.empty_like(sorted_slots)
    rank_values = torch.arange(
        native_routed_topk, device=ids.device, dtype=sorted_slots.dtype
    ).view(1, -1)
    ranks.scatter_(1, sorted_slots, rank_values.expand_as(sorted_slots))
    keep = ranks < selected_k.view(-1, 1)

    masked_ids = ids.masked_fill(~keep, -1)
    masked_weights = weights * keep.to(weights.dtype)
    # The Marlin aligner maps expert id -1 to a filtered block and the Marlin
    # CUDA kernel skips it.  This marker additionally makes the final routed
    # output buffer zero-initialize omitted slots before its Top-K reduction.
    masked_ids._sglang_moe_route_mask = keep.reshape(-1)
    masked_ids._sglang_moe_adaptive_topk = True

    key = (layer_id, mass, allowed)
    if key not in _LOGGED_LAYERS:
        _LOGGED_LAYERS.add(key)
        print(
            "[REDKNOT_ADAPTIVE_TOPK] "
            f"layer={layer_id}/{num_hidden_layers} native_k={native_routed_topk} "
            f"mass={mass:.6f} buckets={','.join(map(str, allowed))} "
            f"min_tokens={min_tokens} renormalize=false "
            f"plan_scoped={str(plan_scoped).lower()} compact_marlin=true",
            file=sys.stderr,
            flush=True,
        )

    if _strict_flag("REDKNOT_ADAPTIVE_TOPK_LOG_FIRST_HISTOGRAM", "0"):
        if layer_id not in _STATS_LOGGED_LAYERS:
            _STATS_LOGGED_LAYERS.add(layer_id)
            # Explicitly diagnostic: this one D2H is disabled in performance
            # runs and only used to prove that multiple K values were selected.
            hist = torch.bincount(selected_k, minlength=native_routed_topk + 1).cpu()
            encoded = ",".join(
                f"k{k}:{int(hist[k])}" for k in allowed
            )
            print(
                f"[REDKNOT_ADAPTIVE_TOPK_HIST] layer={layer_id} tokens={weights.shape[0]} {encoded}",
                file=sys.stderr,
                flush=True,
            )

    return topk_output._replace(topk_ids=masked_ids, topk_weights=masked_weights)


__all__ = ["maybe_apply_dsv4_adaptive_topk"]
