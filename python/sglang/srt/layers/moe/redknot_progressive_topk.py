"""RedKnot Progressive Assignment-Sparse MoE: per-layer routed Top-K schedule.

This is the *assignment-sparse* lever, as opposed to the *token-sparse* lever
implemented by ``layers/attention/redknot/sparse_ffn.py``:

    token-sparse  (sparse FFN)
        A subset of tokens is routed to experts; the rest only see the shared
        expert.  Reduces the *number of tokens* entering routed experts.
        Mutually exclusive with the RedKnot MLA head-split measurement path
        (``redknot_mla_backend`` raises when both are on) because the two
        savings cannot be attributed apart.

    assignment-sparse (this module)
        *Every* token still receives a routed update; only the router's top-K
        is overridden per layer.  Reduces the *number of experts per token*.
        Because it does not read ``redknot_sparse_ffn_enable`` it does not trip
        that mutual-exclusion assert, so it composes with MLA head-split reuse.

Routed grouped-GEMM cost and all-to-all dispatch traffic both scale linearly in
the effective top-K, so the average-K reduction translates almost directly into
a routed-MoE FLOPs/traffic reduction.  No token is dropped, which makes the
accuracy risk substantially lower than token-granularity dropping.

Schedule syntax
---------------
``"start-end:k,start-end:k,..."`` with **inclusive** layer ranges, e.g. for
DeepSeek-V4-Flash-0731 (43 layers, native ``num_experts_per_tok=6``)::

    0-11:6,12-27:5,28-42:4     # ~17.8% routed assignment saving
    0-7:6,8-23:5,24-42:3       # ~28%
    0-3:6,4-19:4,20-42:3       # ~38%

An empty schedule is the native path: every layer keeps its checkpoint top-K
and existing deployments stay bit-identical.  This module is strictly opt-in.

Resolution rules
----------------
* First matching band wins (bands are scanned left to right).
* ``k`` is clamped to ``[1, default_topk]``; a schedule can only ever *shrink*
  the routed fan-out, never grow it beyond what the checkpoint router expects.
* A malformed schedule raises ``ValueError`` instead of silently degrading to
  the native path.  Silent no-ops are the single most expensive failure mode
  here: a typo would produce a clean-looking run whose numbers are simply the
  dense baseline.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, NamedTuple, NoReturn, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEDULE_SYNTAX = "'start-end:k[,start-end:k ...]' with inclusive layer ranges"

__all__ = [
    "LayerTopKBand",
    "SCHEDULE_SYNTAX",
    "estimate_routed_saving",
    "format_schedule_summary",
    "parse_progressive_topk_schedule",
    "resolve_routed_topk",
    "schedule_from_server_args",
]


class LayerTopKBand(NamedTuple):
    """Inclusive layer range with an overridden routed top-K."""

    start: int
    end: int
    top_k: int

    def contains(self, layer_id: int) -> bool:
        return self.start <= layer_id <= self.end


def _fail(schedule: str, token: str, reason: str) -> NoReturn:
    raise ValueError(
        f"invalid --redknot-progressive-topk-schedule entry {token!r} "
        f"in {schedule!r}: {reason}. Expected {SCHEDULE_SYNTAX}, "
        f"for example '0-11:6,12-27:5,28-42:4'."
    )


@lru_cache(maxsize=16)
def parse_progressive_topk_schedule(schedule: str) -> Tuple[LayerTopKBand, ...]:
    """Parse a schedule string into bands, or ``()`` when disabled.

    Raises ``ValueError`` on any malformed entry so a typo fails the launch
    instead of silently reverting to native top-K.
    """

    if schedule is None:
        return ()
    text = str(schedule).strip()
    if not text:
        return ()

    bands: List[LayerTopKBand] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token.count(":") != 1:
            _fail(text, token, "expected exactly one ':' separating range and k")
        rng, k_text = token.split(":", 1)
        rng, k_text = rng.strip(), k_text.strip()
        if rng.count("-") != 1:
            _fail(text, token, "expected exactly one '-' inside the layer range")
        start_text, end_text = (part.strip() for part in rng.split("-", 1))
        try:
            start, end, top_k = int(start_text), int(end_text), int(k_text)
        except ValueError:
            _fail(text, token, "start, end and k must all be integers")
        if start < 0:
            _fail(text, token, f"start layer must be >= 0, got {start}")
        if end < start:
            _fail(text, token, f"end layer {end} precedes start layer {start}")
        if top_k < 1:
            _fail(text, token, f"k must be >= 1, got {top_k}")
        bands.append(LayerTopKBand(start, end, top_k))

    if not bands:
        _fail(text, text, "no usable band found")
    return tuple(bands)


def schedule_from_server_args() -> str:
    """Read the schedule off the global server args, tolerating absence.

    Returns ``""`` (native path) when server args are not initialized yet, e.g.
    when a unit test constructs a MoE layer directly.
    """

    try:
        from sglang.srt.server_args import get_global_server_args

        return getattr(
            get_global_server_args(), "redknot_progressive_topk_schedule", ""
        ) or ""
    except Exception:
        return ""


def resolve_routed_topk(
    layer_id: int,
    default_topk: int,
    schedule: Optional[str] = None,
) -> int:
    """Effective routed top-K for ``layer_id``.

    ``default_topk`` is the checkpoint's ``num_experts_per_tok``.  It is
    returned unchanged when the feature is off or no band matches.
    """

    if schedule is None:
        schedule = schedule_from_server_args()
    if not schedule:
        return int(default_topk)
    try:
        bands = parse_progressive_topk_schedule(schedule)
    except ValueError:
        # Server-arg validation already rejects malformed schedules at launch;
        # if we somehow get here, fail closed onto the native path rather than
        # crashing a worker mid-forward.
        logger.error(
            "RedKnot progressive top-K: ignoring malformed schedule %r", schedule
        )
        return int(default_topk)
    for band in bands:
        if band.contains(int(layer_id)):
            return max(1, min(int(default_topk), band.top_k))
    return int(default_topk)


def estimate_routed_saving(
    num_layers: int,
    default_topk: int,
    schedule: str,
) -> float:
    """Fraction of routed expert *assignments* removed across all layers.

    This is an assignment-count ratio, not an end-to-end FLOPs or wall-clock
    saving: attention, shared experts, norms and the LM head are untouched.
    """

    num_layers = int(num_layers)
    default_topk = int(default_topk)
    if num_layers <= 0 or default_topk <= 0 or not schedule:
        return 0.0
    total = num_layers * default_topk
    kept = sum(
        resolve_routed_topk(layer_id, default_topk, schedule)
        for layer_id in range(num_layers)
    )
    return 1.0 - (kept / total)


def format_schedule_summary(
    num_layers: int,
    default_topk: int,
    schedule: str,
) -> str:
    """One-line human-readable resolution of a schedule, for startup logs."""

    bands = parse_progressive_topk_schedule(schedule)
    if not bands:
        return "RedKnot progressive top-K: disabled (native top-K everywhere)"
    per_k: Dict[int, List[int]] = {}
    for layer_id in range(int(num_layers)):
        k = resolve_routed_topk(layer_id, default_topk, schedule)
        per_k.setdefault(k, []).append(layer_id)
    parts = []
    for k in sorted(per_k, reverse=True):
        layers = per_k[k]
        parts.append(f"k={k}:{len(layers)}L[{layers[0]}..{layers[-1]}]")
    saving = estimate_routed_saving(num_layers, default_topk, schedule)
    return (
        f"RedKnot progressive top-K: schedule={schedule!r} "
        f"native_top_k={int(default_topk)} layers={int(num_layers)} "
        f"resolved={' '.join(parts)} "
        f"routed_assignment_saving={saving:.1%} "
        f"(assignment count only; not total-model FLOPs)"
    )
