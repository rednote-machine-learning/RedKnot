"""Fail-closed, CPU-reference gate for independently cached document blocks.

This module does not certify unsampled rows or downstream model accuracy.
Calibration identifiers are declarations supplied by the caller, not evidence
that calibration occurred. No tolerances are supplied by default. Positions
are document-local; head/tail guards apply to the independent document, not
to every decision block. Gate decisions must precede cache-side effects.
"""

import hashlib
import json
import math
from dataclasses import dataclass

from .core import Binding, OnlineRows, ValidationError, normalize_rows, validate_online


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _finite_number(value, label, *, positive=False):
    _require(type(value) in (int, float), label + " must be a real number, not bool")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValidationError(label + " must be finite") from None
    _require(
        math.isfinite(number) and (number > 0 if positive else number >= 0),
        label + (" must be finite and positive" if positive else " must be finite and nonnegative"),
    )
    return number


def _positions(value, label):
    _require(isinstance(value, tuple), label + " must be an immutable tuple")
    _require(all(type(p) is int and p >= 0 for p in value), label + " must contain nonnegative integer positions")
    _require(value == tuple(sorted(set(value))), label + " must be sorted and unique")


@dataclass(frozen=True)
class GatePolicy:
    enabled: bool = False
    block_size: int = 512
    probe_count: int = 8
    head_guard: int = 128
    tail_guard: int = 128
    relative_tolerance: float | None = None
    absolute_tolerance: float | None = None
    norm_floor: float = 1e-8
    calibration_id: str | None = None
    seed: str = "kce-v1"

    def __post_init__(self):
        _require(type(self.enabled) is bool, "enabled must be an explicit bool")
        for name in ("block_size", "probe_count"):
            value = getattr(self, name)
            _require(type(value) is int and value > 0, name + " must be a positive integer")
        for name in ("head_guard", "tail_guard"):
            value = getattr(self, name)
            _require(type(value) is int and value >= 0, name + " must be a nonnegative integer")
        _finite_number(self.norm_floor, "norm_floor", positive=True)
        _require(isinstance(self.seed, str) and bool(self.seed.strip()), "nonempty seed required")
        for name in ("relative_tolerance", "absolute_tolerance"):
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name)
        if self.calibration_id is not None:
            _require(
                isinstance(self.calibration_id, str) and bool(self.calibration_id.strip()), "calibration_id must be a nonempty declaration"
            )
        if self.enabled:
            _require(
                self.relative_tolerance is not None and self.absolute_tolerance is not None and self.calibration_id is not None,
                "enabled reuse requires explicit relative/absolute tolerances and calibration_id",
            )


@dataclass(frozen=True)
class BlockPlan:
    binding_sha256: str
    positions: tuple[int, ...]
    probe_positions: tuple[int, ...]
    mandatory_positions: tuple[int, ...]

    def __post_init__(self):
        _require(
            isinstance(self.binding_sha256, str)
            and len(self.binding_sha256) == 64
            and all(c in "0123456789abcdef" for c in self.binding_sha256),
            "plan requires a canonical binding SHA256",
        )
        for name in ("positions", "probe_positions", "mandatory_positions"):
            _positions(getattr(self, name), name)
        _require(bool(self.positions), "block positions cannot be empty")
        full = set(self.positions)
        _require(
            set(self.probe_positions) <= full and set(self.mandatory_positions) <= full,
            "probe and mandatory positions must belong to the block",
        )
        _require(set(self.probe_positions).isdisjoint(self.mandatory_positions), "probe and mandatory positions cannot overlap")


@dataclass(frozen=True)
class GateDecision:
    allow_reuse: bool
    reason: str
    max_relative_l2: float | None
    max_absolute_error: float | None

    def __post_init__(self):
        _require(
            type(self.allow_reuse) is bool and isinstance(self.reason, str) and bool(self.reason),
            "decision requires an explicit bool and nonempty reason",
        )
        _require(
            (self.max_relative_l2 is None) == (self.max_absolute_error is None), "decision metrics must both be present or both absent"
        )
        if self.max_relative_l2 is not None:
            _finite_number(self.max_relative_l2, "max_relative_l2")
            _finite_number(self.max_absolute_error, "max_absolute_error")
        _require(not self.allow_reuse or self.max_relative_l2 is not None, "reuse decision requires observed finite metrics")


def make_plan(binding: Binding, policy: GatePolicy) -> BlockPlan:
    """Plan aligned blocks and stratified probes without examining row values.

    The same identity/block/seed gives the same probes across request IDs and
    phases. Binding SHA still ties the resulting plan to the exact request.
    Each stratum contributes one probe; short candidate regions use every row.
    """
    _require(isinstance(binding, Binding), "Binding required")
    _require(isinstance(policy, GatePolicy), "GatePolicy required")
    policy.__post_init__()
    start, end = binding.block_start, binding.block_end
    count = binding.identity.token_count
    _require(
        start % policy.block_size == 0 and end == min(start + policy.block_size, count),
        "block must be aligned and complete, except the final document tail block",
    )
    positions = binding.positions
    mandatory = tuple(p for p in positions if p < policy.head_guard or p >= count - policy.tail_guard)
    protected = set(mandatory)
    candidates = tuple(p for p in positions if p not in protected)
    probes = []
    sample_count = min(policy.probe_count, len(candidates))
    for stratum in range(sample_count):
        lo = stratum * len(candidates) // sample_count
        hi = (stratum + 1) * len(candidates) // sample_count
        material = json.dumps(
            ["kv-cache-engram-stratified-probes-v1", policy.seed, binding.identity.sha256, start, end, stratum, sample_count, lo, hi],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(material).digest(), "big") % (hi - lo)
        probes.append(candidates[lo + offset])
    return BlockPlan(binding.sha256, positions, tuple(probes), mandatory)


def _cached_rows(values, expected_count, width):
    return normalize_rows(values, width=width, count=expected_count)


def evaluate(plan: BlockPlan, policy: GatePolicy, cached_probe_rows, online_probes: OnlineRows, binding: Binding) -> GateDecision:
    """Check all probes against explicit thresholds; never certify other rows.

    Invalid configuration, plan, bindings, shapes or observations raise
    ValidationError. Valid observations exceeding a threshold return a deny
    decision. Finite observations that overflow metric arithmetic also deny.
    A disabled policy or all-mandatory block denies without reading probes.

    Relative error is max over probes of
        ||online-cached||_2 / max(||online||_2, policy.norm_floor).
    Absolute error is max over all probe components of |online-cached|.
    Both maxima must satisfy their limits, including equality. Probe and
    mandatory rows themselves always remain online in the runtime merge.
    The caller authenticates the cached snapshot and extracts rows in exactly
    probe_positions order; this value-only argument cannot authenticate their
    provenance independently of the cache/runtime contract.
    """
    _require(isinstance(plan, BlockPlan), "BlockPlan required")
    expected = make_plan(binding, policy)
    _require(plan == expected, "plan does not match binding and deterministic policy")
    if not policy.enabled:
        return GateDecision(False, "policy_disabled", None, None)
    if not plan.probe_positions:
        return GateDecision(False, "no_reusable_positions", None, None)
    online = validate_online(binding, online_probes, plan.probe_positions)
    cached = _cached_rows(cached_probe_rows, len(plan.probe_positions), binding.identity.width)
    max_relative = max_absolute = 0.0
    for current, stored in zip(online.rows, cached, strict=False):
        delta = tuple(a - b for a, b in zip(current, stored, strict=False))
        difference_norm = math.hypot(*delta)
        current_norm = math.hypot(*current)
        absolute = max(abs(value) for value in delta)
        if not all(math.isfinite(value) for value in (difference_norm, current_norm, absolute)):
            return GateDecision(False, "nonfinite_metric", None, None)
        relative = difference_norm / max(current_norm, policy.norm_floor)
        if not math.isfinite(relative):
            return GateDecision(False, "nonfinite_metric", None, None)
        max_relative = max(max_relative, relative)
        max_absolute = max(max_absolute, absolute)
    allowed = max_relative <= policy.relative_tolerance and max_absolute <= policy.absolute_tolerance
    return GateDecision(allowed, "within_probe_limits" if allowed else "threshold_exceeded", max_relative, max_absolute)
