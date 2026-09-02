"""Read-only router-distribution profiling for RedKnot adaptive MoE Top-K.

The profiler is deliberately separate from the routing implementation: it
observes a native Top-6 DSV4 forward and never mutates Top-K ids or weights.
It runs only for an explicit pure-MLA restore request and only on TP rank 0.

Output records are JSON payloads prefixed by ``REDKNOT_ADAPTIVE_TOPK_PROFILE``.
They contain mergeable histograms instead of raw token scores, keeping a 256K
profile auditable without writing hundreds of millions of values.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

PROFILE_ENV = "REDKNOT_ADAPTIVE_TOPK_PROFILE"
PROFILE_PREFIX = "REDKNOT_ADAPTIVE_TOPK_PROFILE "
MASS_THRESHOLDS = (0.80, 0.90, 0.95, 0.98)
HISTOGRAM_BINS = 100


def _single_pure_restore_plan(forward_batch: Any) -> Optional[Dict[str, Any]]:
    plans = getattr(forward_batch, "redknot_reuse_plan", None)
    if isinstance(plans, dict):
        plans = [plans]
    if not isinstance(plans, (list, tuple)) or len(plans) != 1:
        return None
    plan = plans[0]
    if not isinstance(plan, dict):
        return None
    if plan.get("mode") != "restore" or plan.get("reuse_mla_off") is not True:
        return None
    return plan


def _histogram01(torch, values):
    values = values.float().clamp_(0.0, 1.0)
    return torch.histc(values, bins=HISTOGRAM_BINS, min=0.0, max=1.0).to(
        dtype=torch.int64
    )


def maybe_profile_dsv4_router_distribution(
    *,
    layer_id: int,
    topk_output: Any,
    forward_batch: Any,
    correction_bias: Any,
    scoring_func: str,
    num_routed_experts: int,
    native_routed_topk: int,
    num_fused_shared_experts: int,
) -> None:
    """Log native DSV4 router concentration without changing routing.

    This profile intentionally requires the checkpoint's native routed K=6.
    A fixed progressive schedule would truncate the observations before this
    function sees them and therefore cannot be used to infer an adaptive K.
    """

    if os.environ.get(PROFILE_ENV, "0") != "1":
        return
    if not 3 <= int(layer_id) <= 39:
        return

    from sglang.srt.distributed import get_tensor_model_parallel_rank

    if int(get_tensor_model_parallel_rank()) != 0:
        return

    plan = _single_pure_restore_plan(forward_batch)
    if plan is None:
        return

    import torch
    import torch.nn.functional as F

    if scoring_func != "sqrtsoftplus":
        raise RuntimeError(
            "adaptive Top-K profile requires DSV4 sqrtsoftplus scoring, got "
            f"{scoring_func!r}"
        )
    if int(native_routed_topk) != 6 or int(num_routed_experts) != 256:
        raise RuntimeError(
            "adaptive Top-K profile is frozen to DSV4 native K6/E256, got "
            f"K={native_routed_topk} E={num_routed_experts}"
        )
    if int(num_fused_shared_experts) not in (0, 1):
        raise RuntimeError(
            "adaptive Top-K profile supports zero or one fused shared expert"
        )
    if not all(hasattr(topk_output, name) for name in ("topk_ids", "topk_weights", "router_logits")):
        raise RuntimeError("adaptive Top-K profile requires StandardTopKOutput")

    ids = topk_output.topk_ids
    weights = topk_output.topk_weights
    logits = topk_output.router_logits
    expected_width = native_routed_topk + num_fused_shared_experts
    if ids.ndim != 2 or weights.shape != ids.shape or ids.shape[1] != expected_width:
        raise RuntimeError(
            "unexpected DSV4 Top-K layout: "
            f"ids={tuple(ids.shape)} weights={tuple(weights.shape)} "
            f"expected_width={expected_width}"
        )
    if logits.ndim != 2 or logits.shape != (ids.shape[0], num_routed_experts):
        raise RuntimeError(
            "unexpected DSV4 router-logit layout: "
            f"got={tuple(logits.shape)} expected={(ids.shape[0], num_routed_experts)}"
        )

    positions = getattr(forward_batch, "positions", None)
    if positions is None or positions.ndim != 1 or positions.numel() != ids.shape[0]:
        raise RuntimeError(
            "adaptive Top-K profile requires one canonical position per token"
        )
    query_start = int(plan.get("query_start", -1))
    if query_start <= 0:
        raise RuntimeError("adaptive Top-K profile plan has no positive query_start")
    document_rows = positions.to(torch.int64) < query_start
    if not bool(document_rows.any().item()):
        return

    ids = ids[document_rows]
    weights = weights[document_rows]
    logits = logits[document_rows]
    positions = positions[document_rows]
    routed_ids = ids[:, :native_routed_topk].to(torch.int64)
    shared_ids = ids[:, native_routed_topk:]
    if bool(((routed_ids < 0) | (routed_ids >= num_routed_experts)).any().item()):
        raise RuntimeError("native routed Top-6 contains an invalid expert id")
    if num_fused_shared_experts and bool(
        (shared_ids < num_routed_experts).any().item()
    ):
        raise RuntimeError("fused shared expert is not the final Top-K column")

    # The model renormalizes routed K=6. Sort by actual routed contribution so
    # cumulative mass answers: how much of the current K6 update survives at K.
    routed_weights = weights[:, :native_routed_topk].float().clamp_min_(0.0)
    routed_weights = torch.sort(routed_weights, dim=-1, descending=True).values
    routed_prob = routed_weights / routed_weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    cumulative = routed_prob.cumsum(dim=-1)

    scores = F.softplus(logits.float()).sqrt()
    full_prob = scores / scores.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    entropy_all = -(full_prob * full_prob.clamp_min(1e-30).log()).sum(dim=-1)
    entropy_all = entropy_all / math.log(float(num_routed_experts))
    entropy_top6 = -(routed_prob * routed_prob.clamp_min(1e-30).log()).sum(dim=-1)
    entropy_top6 = entropy_top6 / math.log(float(native_routed_topk))

    if correction_bias is None or correction_bias.numel() != num_routed_experts:
        raise RuntimeError("DSV4 adaptive profile requires the E256 correction bias")
    choice = scores + correction_bias.float().reshape(1, -1)
    choice_top7 = torch.topk(choice, k=7, dim=-1, largest=True, sorted=True).values
    margins = (choice_top7[:, :6] - choice_top7[:, 1:7]) / choice_top7[
        :, :6
    ].abs().clamp_min(1e-8)

    selected_score_mass = scores.gather(1, routed_ids).sum(dim=-1)
    top6_mass_all = selected_score_mass / scores.sum(dim=-1).clamp_min(1e-30)

    needed_counts = {}
    for threshold in MASS_THRESHOLDS:
        # Count entries strictly below the threshold, then include the first
        # crossing expert. This is the deliberate fix for the old top-p bug.
        needed = (cumulative < threshold).sum(dim=-1) + 1
        needed = needed.clamp_(1, native_routed_topk)
        needed_counts[f"{threshold:.2f}"] = torch.bincount(
            needed.to(torch.int64), minlength=native_routed_topk + 1
        )[1:]

    features = {
        "cum2": cumulative[:, 1],
        "cum3": cumulative[:, 2],
        "cum4": cumulative[:, 3],
        "cum5": cumulative[:, 4],
        "top1_share_top6": routed_prob[:, 0],
        "entropy_top6": entropy_top6,
        "entropy_all256": entropy_all,
        "top6_mass_all256": top6_mass_all,
        "margin_after2": margins[:, 1],
        "margin_after3": margins[:, 2],
        "margin_after4": margins[:, 3],
        "margin_after5": margins[:, 4],
        "margin_after6": margins[:, 5],
    }
    feature_histograms = {name: _histogram01(torch, value) for name, value in features.items()}
    expert_counts = torch.bincount(
        routed_ids.reshape(-1), minlength=num_routed_experts
    ).to(torch.int64)

    # One D2H transfer per record keeps the diagnostic explicit and minimizes
    # rank-0 stalls. Timing from this run is never used as a performance claim.
    ordered_names = tuple(feature_histograms)
    packed = torch.cat(
        [
            *(needed_counts[key] for key in sorted(needed_counts)),
            *(feature_histograms[name] for name in ordered_names),
            expert_counts,
        ]
    ).detach().cpu()
    cursor = 0
    needed_json = {}
    for key in sorted(needed_counts):
        needed_json[key] = packed[cursor : cursor + native_routed_topk].tolist()
        cursor += native_routed_topk
    hist_json = {}
    for name in ordered_names:
        hist_json[name] = packed[cursor : cursor + HISTOGRAM_BINS].tolist()
        cursor += HISTOGRAM_BINS
    expert_json = packed[cursor : cursor + num_routed_experts].tolist()

    payload = {
        "format": "redknot_adaptive_topk_profile_v1",
        "request_id": str(plan.get("benchmark_request_id") or ""),
        "layer_id": int(layer_id),
        "position_start": int(positions.min().item()),
        "position_end": int(positions.max().item()) + 1,
        "document_tokens": int(positions.numel()),
        "native_routed_topk": int(native_routed_topk),
        "num_routed_experts": int(num_routed_experts),
        "num_fused_shared_experts": int(num_fused_shared_experts),
        "mass_threshold_needed_k_counts": needed_json,
        "histogram_bins": HISTOGRAM_BINS,
        "histograms_0_1": hist_json,
        "expert_top6_counts": expert_json,
    }
    logger.info(PROFILE_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))


__all__ = [
    "HISTOGRAM_BINS",
    "MASS_THRESHOLDS",
    "PROFILE_ENV",
    "PROFILE_PREFIX",
    "maybe_profile_dsv4_router_distribution",
]
