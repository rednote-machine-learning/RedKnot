"""CPU-only, single-use block transactions; no native KV-ring integration.

``prepare`` can call the pure ``compute_online(binding, positions)`` callback,
but cannot call a submitter or write the cache. The callback must return current
request/layer/block-bound ``OnlineRows`` and must not mutate a KV ring, cache, or
other externally committed state. Invalid output and ordinary callback errors
abort the transaction. Only an explicit ``ProbeFailure`` permits a probe-only
availability failure to fall back to a fresh full-block computation.

``commit`` is the sole submission boundary. It consumes the prepared transaction
before invoking ``submit(binding, online_rows)``. Submission can have arbitrary
external side effects; any exception therefore poisons this transaction, even if
the caller believes nothing was written. There is no retry or rollback claim.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .cache import CacheArtifact
from .calibration import CalibrationRecord
from .core import Binding, OnlineRows, ValidationError, validate_online
from .gate import GateDecision, GatePolicy, evaluate, make_plan


class TransactionStateError(RuntimeError):
    """An operation is not permitted in this transaction's current state."""


class ProbeFailure(RuntimeError):
    """Explicit, side-effect-free probe unavailability; full recompute is safe.

    A backend must not use this for wrong identity, shape, nonfinite output, or
    an uncertain externally committed side effect. Those are abort conditions.
    """


@dataclass(frozen=True)
class PreparedBlock:
    """Immutable inspection result, not an independently committable token."""

    binding_sha256: str
    positions: tuple[int, ...]
    rows: tuple[tuple[float, ...], ...]
    mode: str
    fallback_reason: str | None
    online_positions: tuple[int, ...]
    cached_positions: tuple[int, ...]
    gate_decision: GateDecision | None


class BlockTransaction:
    """Own one preparation and at most one submit attempt for one binding.

    State path: NEW -> PREPARING -> PREPARED -> COMMITTING -> COMMITTED.
    Preparation or pre-submit identity failure gives ABORTED. A submit exception
    gives POISONED. ABORTED/POISONED/COMMITTED cannot prepare or commit again.
    A short state lock rejects competing or reentrant prepare/commit attempts.
    Callbacks run without the lock; this is not a scheduler or backend lock.
    """

    def __init__(
        self,
        binding: Binding,
        cache: CacheArtifact | None,
        policy: GatePolicy,
        compute_online: Callable[[Binding, tuple[int, ...]], OnlineRows],
        *,
        calibration: CalibrationRecord | None = None,
    ) -> None:
        if not isinstance(binding, Binding):
            raise ValidationError("Binding required")
        if not isinstance(policy, GatePolicy):
            raise ValidationError("GatePolicy required")
        if not callable(compute_online):
            raise ValidationError("compute_online must be callable")
        self._binding = binding
        self._cache = cache
        self._policy = policy
        self._compute_online = compute_online
        self._calibration = calibration
        self._state = "NEW"
        self._state_lock = threading.Lock()
        self._prepared = None  # type: Optional[PreparedBlock]
        self._submission = None  # type: Optional[OnlineRows]

    @property
    def state(self) -> str:
        return self._state

    @property
    def binding(self) -> Binding:
        return self._binding

    @property
    def prepared(self) -> PreparedBlock | None:
        return self._prepared

    def _require_state(self, expected: str) -> None:
        if self._state != expected:
            raise TransactionStateError(f"expected state {expected}, found {self._state}")

    def _begin(self, expected: str, following: str) -> None:
        with self._state_lock:
            self._require_state(expected)
            self._state = following

    def _online(self, positions: tuple[int, ...]) -> OnlineRows:
        result = self._compute_online(self._binding, positions)
        return validate_online(self._binding, result, positions)

    def _full(self, reason: str, decision: GateDecision | None = None) -> PreparedBlock:
        data = self._online(self._binding.positions)
        return PreparedBlock(
            self._binding.sha256,
            data.positions,
            data.rows,
            "full_recompute",
            reason,
            data.positions,
            (),
            decision,
        )

    def _prepare(self) -> PreparedBlock:
        self._policy.__post_init__()
        if not self._policy.enabled:
            return self._full("policy_disabled")
        if self._calibration is None:
            return self._full("calibration_missing")
        # A declaration string in GatePolicy is not sufficient. This record
        # binds external evidence/configuration, but does not itself certify
        # model accuracy or that external calibration actually passed.
        try:
            if not isinstance(self._calibration, CalibrationRecord):
                raise ValidationError("CalibrationRecord required")
            self._calibration.validate(self._binding.identity, self._policy)
        except Exception as exc:
            return self._full("calibration_invalid:" + type(exc).__name__)
        plan = make_plan(self._binding, self._policy)
        if self._cache is None:
            return self._full("cache_missing")
        # Cache faults never authorize reuse. This guarded section performs only
        # local reads/validation and never invokes the online backend.
        try:
            if not isinstance(self._cache, CacheArtifact):
                raise ValidationError("CacheArtifact required")
            self._cache.validate(self._binding.identity)
            cached_rows = self._cache.take(plan.positions)
            checked_cache = OnlineRows(self._binding.sha256, plan.positions, cached_rows)
            validate_online(self._binding, checked_cache, plan.positions)
        except Exception as exc:
            return self._full("cache_invalid:" + type(exc).__name__)
        if not plan.probe_positions:
            return self._full("no_reusable_positions")
        try:
            online_probes = self._online(plan.probe_positions)
        except ProbeFailure:
            return self._full("probe_unavailable")
        cache_by_position = dict(zip(plan.positions, cached_rows, strict=False))
        decision = evaluate(
            plan,
            self._policy,
            tuple(cache_by_position[p] for p in plan.probe_positions),
            online_probes,
            self._binding,
        )
        if not decision.allow_reuse:
            return self._full(decision.reason, decision)
        # Probe rows are fresh online values, never stale cache values. Mandatory
        # document guards are separately recomputed if not already probed.
        online = dict(zip(online_probes.positions, online_probes.rows, strict=False))
        remaining = tuple(p for p in plan.mandatory_positions if p not in online)
        if remaining:
            mandatory = self._online(remaining)
            online.update(zip(mandatory.positions, mandatory.rows, strict=False))
        online_positions = tuple(sorted(online))
        cached_positions = tuple(p for p in plan.positions if p not in online)
        rows = tuple(online[p] if p in online else cache_by_position[p] for p in plan.positions)
        data = validate_online(
            self._binding,
            OnlineRows(self._binding.sha256, plan.positions, rows),
            self._binding.positions,
        )
        return PreparedBlock(
            self._binding.sha256,
            data.positions,
            data.rows,
            "gated_reuse" if cached_positions else "full_recompute",
            None if cached_positions else "all_positions_online",
            online_positions,
            cached_positions,
            decision,
        )

    def prepare(self) -> PreparedBlock:
        """Perform pure computation only; any failure permanently aborts."""
        self._begin("NEW", "PREPARING")
        try:
            prepared = self._prepare()
            submission = validate_online(
                self._binding,
                OnlineRows(prepared.binding_sha256, prepared.positions, prepared.rows),
                self._binding.positions,
            )
            self._prepared = prepared
            self._submission = submission
            self._state = "PREPARED"
            return prepared
        except BaseException:
            self._state = "ABORTED"
            raise

    def commit(self, submit: Callable, expected_binding: Binding):
        """Submit internal prepared data exactly once and return submit's result.

        An incorrect expected binding aborts before submit. No external
        PreparedBlock argument is accepted, so snapshots cannot be swapped
        across transactions. Submit must bind its write to the supplied binding.
        """
        with self._state_lock:
            self._require_state("PREPARED")
            try:
                if (
                    not isinstance(expected_binding, Binding)
                    or expected_binding != self._binding
                    or expected_binding.sha256 != self._binding.sha256
                ):
                    raise ValidationError("commit binding differs from prepared binding")
                if not callable(submit):
                    raise ValidationError("submit must be callable")
                data = validate_online(self._binding, self._submission, self._binding.positions)
            except BaseException:
                self._state = "ABORTED"
                raise
            self._state = "COMMITTING"
        try:
            result = submit(self._binding, data)
        except BaseException:
            self._state = "POISONED"
            raise
        self._state = "COMMITTED"
        return result
