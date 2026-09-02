"""CPU control plane for ragged, continuously batched DSV4 reuse.

The serving hot path flattens every request's current rows into one Q input.
Reuse metadata, however, remains request-relative: two requests can both have
logical position zero, bind different artifact generations, and replay
different boundary rows.  This module makes that distinction explicit before
any sparse Q projection or cache mutation is allowed.

The safety rule is deliberately asymmetric:

* global logical heads run for every flattened row;
* local logical heads may be omitted only for certified clean rows;
* a request with absent or invalid reuse metadata becomes entirely dirty; and
* invalid batch geometry raises, because no request-relative fallback can make
  an untrustworthy flattened-row mapping safe.

No CUDA, torch, SGLang, or collective dependency is imported here.  Tensor-
like ForwardBatch fields are copied through ``detach().cpu().tolist()`` when
available, so this module can be unit-tested with ordinary Python values.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Optional, Sequence, Tuple

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - exercised by minimal CPU-only installs.
    _orjson = None


BATCH_REUSE_FORMAT_VERSION = 1
STABLE_DIGEST_SEMANTICS = "sha256_canonical_json_v1"


class BatchReuseValidationError(ValueError):
    """The flattened batch geometry cannot safely enter a sparse path."""


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_rows(
    values: Sequence[int],
    *,
    name: str,
    upper_bound: Optional[int] = None,
    allow_empty: bool = True,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of integers")
    result = tuple(_strict_int(value, f"{name} entry") for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be strictly increasing and unique")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must be non-negative")
    if upper_bound is not None and any(value >= upper_bound for value in result):
        raise ValueError(f"{name} contains an out-of-range row")
    return result


def _canonical_digest(payload: object) -> str:
    encoded = None
    if _orjson is not None:
        try:
            candidate = _orjson.dumps(payload, option=_orjson.OPT_SORT_KEYS)
        except (TypeError, ValueError, OverflowError):
            candidate = None
        # orjson and the legacy json.dumps contract are byte-identical for
        # the immutable ASCII/int/bool/None control payloads used here.  The
        # fast C-level isascii guard preserves the old ensure_ascii=True
        # digest for any future non-ASCII request metadata by falling back.
        if candidate is not None and candidate.isascii():
            encoded = candidate
    if encoded is None:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def _cpu_list(value: object, name: str) -> list[Any]:
    """Copy a one-dimensional tensor-like value to an ordinary list."""

    if value is None:
        raise BatchReuseValidationError(f"ForwardBatch.{name} is required")
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        current = tolist()
    if isinstance(current, tuple):
        current = list(current)
    if not isinstance(current, list):
        raise BatchReuseValidationError(
            f"ForwardBatch.{name} must expose a one-dimensional CPU list"
        )
    if any(isinstance(item, (list, tuple, Mapping)) for item in current):
        raise BatchReuseValidationError(
            f"ForwardBatch.{name} must be one-dimensional"
        )
    return current


def validate_disjoint_union(
    *,
    left: Sequence[int],
    right: Sequence[int],
    universe: Sequence[int],
    left_name: str = "left rows",
    right_name: str = "right rows",
) -> None:
    """Require two canonical row domains to partition ``universe`` exactly."""

    left_rows = _strict_rows(left, name=left_name)
    right_rows = _strict_rows(right, name=right_name)
    all_rows = _strict_rows(universe, name="universe")
    left_set = set(left_rows)
    right_set = set(right_rows)
    if left_set.intersection(right_set):
        raise ValueError(f"{left_name} and {right_name} must be disjoint")
    if left_set.union(right_set) != set(all_rows):
        raise ValueError(
            f"{left_name} and {right_name} must union to the complete universe"
        )


@dataclass(frozen=True, order=True)
class ArtifactGroupKey:
    """A generation-compatible persistent device artifact group."""

    artifact_token: str
    artifact_epoch: int
    policy_digest: str

    def __post_init__(self) -> None:
        _nonempty_string(self.artifact_token, "artifact_token")
        epoch = _strict_int(self.artifact_epoch, "artifact_epoch")
        if epoch <= 0:
            raise ValueError("artifact_epoch must be positive")
        _nonempty_string(self.policy_digest, "policy_digest")

    @property
    def digest(self) -> str:
        return _canonical_digest(
            {
                "artifact_epoch": self.artifact_epoch,
                "artifact_token": self.artifact_token,
                "policy_digest": self.policy_digest,
            }
        )


@dataclass(frozen=True)
class SegmentBinding:
    """One request placement within a position-independent artifact bundle."""

    segment_token: str
    logical_start: int
    length: int
    artifact_row_start: int = 0

    def __post_init__(self) -> None:
        _nonempty_string(self.segment_token, "segment_token")
        logical_start = _strict_int(self.logical_start, "logical_start")
        length = _strict_int(self.length, "segment length")
        artifact_row_start = _strict_int(
            self.artifact_row_start, "artifact_row_start"
        )
        if logical_start < 0 or artifact_row_start < 0:
            raise ValueError("segment starts must be non-negative")
        if length <= 0:
            raise ValueError("segment length must be positive")

    @property
    def logical_end(self) -> int:
        return self.logical_start + self.length

    def _payload(self) -> Mapping[str, object]:
        return {
            "artifact_row_start": self.artifact_row_start,
            "length": self.length,
            "logical_start": self.logical_start,
            "segment_token": self.segment_token,
        }


@dataclass(frozen=True)
class RequestReuseLayout:
    """Validated mapping and reuse decision for one request in a ragged batch."""

    request_index: int
    request_token: str
    flat_row_start: int
    logical_positions: Tuple[int, ...]
    query_start: int
    dirty_request_rows: Tuple[int, ...]
    segments: Tuple[SegmentBinding, ...] = ()
    artifact_token: Optional[str] = None
    artifact_epoch: int = 0
    policy_digest: str = ""
    fallback_reason: str = ""
    _clean_request_rows_cache: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _flat_rows_cache: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _dirty_flat_rows_cache: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _clean_flat_rows_cache: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _cached_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        request_index = _strict_int(self.request_index, "request_index")
        flat_row_start = _strict_int(self.flat_row_start, "flat_row_start")
        query_start = _strict_int(self.query_start, "query_start")
        if request_index < 0 or flat_row_start < 0 or query_start < 0:
            raise ValueError("request index, offsets, and query_start must be non-negative")
        _nonempty_string(self.request_token, "request_token")
        if type(self.logical_positions) is not tuple:
            raise TypeError("logical_positions must be an immutable tuple")
        if type(self.dirty_request_rows) is not tuple:
            raise TypeError("dirty_request_rows must be an immutable tuple")
        if type(self.segments) is not tuple:
            raise TypeError("segments must be an immutable tuple")
        if not self.logical_positions:
            raise ValueError("each request must contribute at least one flattened row")
        positions = _strict_rows(
            self.logical_positions,
            name="logical_positions",
            allow_empty=False,
        )
        dirty = _strict_rows(
            self.dirty_request_rows,
            name="dirty_request_rows",
            upper_bound=len(positions),
        )
        if any(not isinstance(segment, SegmentBinding) for segment in self.segments):
            raise TypeError("segments must contain SegmentBinding values")
        if not isinstance(self.fallback_reason, str):
            raise TypeError("fallback_reason must be a string")

        if self.reuse_enabled:
            if not self.segments:
                raise ValueError("reuse requires at least one segment")
            ordered = tuple(sorted(self.segments, key=lambda item: item.logical_start))
            if ordered != self.segments:
                raise ValueError("segments must be ordered by logical_start")
            expected_start = 0
            for segment in ordered:
                if segment.logical_start != expected_start:
                    raise ValueError(
                        "segments must be disjoint and tile [0, query_start)"
                    )
                expected_start = segment.logical_end
            if expected_start != query_start:
                raise ValueError("segments must union exactly to [0, query_start)")

            dirty_set = set(dirty)
            for row, position in enumerate(positions):
                if position >= query_start and row not in dirty_set:
                    raise ValueError("query/new rows must stay dirty for local heads")
                if row not in dirty_set and position >= query_start:
                    raise AssertionError("unreachable clean query row")
        else:
            if self.artifact_token is not None:
                raise ValueError("dense fallback cannot bind an artifact token")
            if self.artifact_epoch != 0 or self.policy_digest or self.segments:
                raise ValueError("dense fallback cannot bind artifact metadata")
            if not self.fallback_reason:
                raise ValueError("dense fallback must record a reason")
            if dirty != tuple(range(len(positions))):
                raise ValueError("dense fallback must mark every request row dirty")

        dirty_set = set(dirty)
        clean_request_rows = tuple(
            row for row in range(len(positions)) if row not in dirty_set
        )
        validate_disjoint_union(
            left=clean_request_rows,
            right=dirty,
            universe=tuple(range(len(positions))),
            left_name="clean request rows",
            right_name="dirty request rows",
        )
        flat_rows = tuple(range(flat_row_start, flat_row_start + len(positions)))
        dirty_flat_rows = tuple(flat_row_start + row for row in dirty)
        clean_flat_rows = tuple(
            flat_row_start + row for row in clean_request_rows
        )
        object.__setattr__(
            self, "_clean_request_rows_cache", clean_request_rows
        )
        object.__setattr__(self, "_flat_rows_cache", flat_rows)
        object.__setattr__(self, "_dirty_flat_rows_cache", dirty_flat_rows)
        object.__setattr__(self, "_clean_flat_rows_cache", clean_flat_rows)
        object.__setattr__(
            self, "_cached_digest", _canonical_digest(self._payload())
        )

    @property
    def row_count(self) -> int:
        return len(self.logical_positions)

    @property
    def flat_row_end(self) -> int:
        return self.flat_row_start + self.row_count

    @property
    def reuse_enabled(self) -> bool:
        if self.fallback_reason:
            return False
        if self.artifact_token is None:
            return False
        _nonempty_string(self.artifact_token, "artifact_token")
        epoch = _strict_int(self.artifact_epoch, "artifact_epoch")
        if epoch <= 0:
            raise ValueError("artifact_epoch must be positive")
        _nonempty_string(self.policy_digest, "policy_digest")
        return True

    @property
    def group_key(self) -> Optional[ArtifactGroupKey]:
        if not self.reuse_enabled:
            return None
        assert self.artifact_token is not None
        return ArtifactGroupKey(
            artifact_token=self.artifact_token,
            artifact_epoch=self.artifact_epoch,
            policy_digest=self.policy_digest,
        )

    @property
    def clean_request_rows(self) -> Tuple[int, ...]:
        return self._clean_request_rows_cache

    @property
    def flat_rows(self) -> Tuple[int, ...]:
        return self._flat_rows_cache

    @property
    def dirty_flat_rows(self) -> Tuple[int, ...]:
        return self._dirty_flat_rows_cache

    @property
    def clean_flat_rows(self) -> Tuple[int, ...]:
        return self._clean_flat_rows_cache

    def request_row_to_flat(self, request_row: int) -> int:
        request_row = _strict_int(request_row, "request_row")
        if request_row < 0 or request_row >= self.row_count:
            raise IndexError("request_row is outside this request")
        return self.flat_row_start + request_row

    def flat_row_to_request(self, flat_row: int) -> int:
        flat_row = _strict_int(flat_row, "flat_row")
        if flat_row < self.flat_row_start or flat_row >= self.flat_row_end:
            raise IndexError("flat_row is outside this request")
        return flat_row - self.flat_row_start

    def logical_position_to_flat(self, logical_position: int) -> int:
        logical_position = _strict_int(logical_position, "logical_position")
        row = bisect_right(self.logical_positions, logical_position) - 1
        if row < 0 or self.logical_positions[row] != logical_position:
            raise KeyError("logical position is absent from this forward")
        return self.flat_row_start + row

    def as_dense(self, reason: str) -> "RequestReuseLayout":
        return RequestReuseLayout(
            request_index=self.request_index,
            request_token=self.request_token,
            flat_row_start=self.flat_row_start,
            logical_positions=self.logical_positions,
            query_start=self.query_start,
            dirty_request_rows=tuple(range(self.row_count)),
            fallback_reason=_nonempty_string(reason, "fallback reason"),
        )

    def _rebind_certified_artifact(
        self,
        *,
        artifact_token: str,
        artifact_epoch: int,
        policy_digest: str,
    ) -> "RequestReuseLayout":
        """Clone only artifact metadata after a parent plan live-check.

        The caller must first validate the containing :class:`BatchedReusePlan`.
        All O(rows) geometry objects and their cached complements are then
        reused by exact identity; only the layer-specific artifact generation
        changes.  This private constructor cannot be reached from request JSON.
        """

        token = _nonempty_string(artifact_token, "artifact_token")
        epoch = _strict_int(artifact_epoch, "artifact_epoch")
        policy = _nonempty_string(policy_digest, "policy_digest")
        if epoch <= 0 or not self.reuse_enabled:
            raise ValueError("certified artifact rebind requires reusable geometry")
        rebound = object.__new__(RequestReuseLayout)
        for name in (
            "request_index",
            "request_token",
            "flat_row_start",
            "logical_positions",
            "query_start",
            "dirty_request_rows",
            "segments",
            "fallback_reason",
            "_clean_request_rows_cache",
            "_flat_rows_cache",
            "_dirty_flat_rows_cache",
            "_clean_flat_rows_cache",
        ):
            object.__setattr__(rebound, name, getattr(self, name))
        object.__setattr__(rebound, "artifact_token", token)
        object.__setattr__(rebound, "artifact_epoch", epoch)
        object.__setattr__(rebound, "policy_digest", policy)
        object.__setattr__(
            rebound, "_cached_digest", _canonical_digest(rebound._payload())
        )
        return rebound

    def _payload(self) -> Mapping[str, object]:
        return {
            "artifact_epoch": self.artifact_epoch,
            "artifact_token": self.artifact_token,
            "dirty_request_rows": self.dirty_request_rows,
            "fallback_reason": self.fallback_reason,
            "flat_row_start": self.flat_row_start,
            "logical_positions": self.logical_positions,
            "policy_digest": self.policy_digest,
            "query_start": self.query_start,
            "request_index": self.request_index,
            "request_token": self.request_token,
            "segments": tuple(segment._payload() for segment in self.segments),
        }

    @property
    def digest(self) -> str:
        return self._cached_digest


@dataclass(frozen=True)
class ArtifactGroup:
    """Requests that can share one persistent artifact generation/policy."""

    key: ArtifactGroupKey
    request_indices: Tuple[int, ...]
    flat_rows: Tuple[int, ...]
    restore_flat_rows: Tuple[int, ...]
    dirty_flat_rows: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, ArtifactGroupKey):
            raise TypeError("group key has an invalid type")
        _strict_rows(
            self.request_indices, name="group request_indices", allow_empty=False
        )
        rows = _strict_rows(self.flat_rows, name="group flat_rows", allow_empty=False)
        restore = _strict_rows(self.restore_flat_rows, name="group restore_flat_rows")
        dirty = _strict_rows(self.dirty_flat_rows, name="group dirty_flat_rows")
        validate_disjoint_union(
            left=restore,
            right=dirty,
            universe=rows,
            left_name="group restore rows",
            right_name="group dirty rows",
        )

    def _payload(self) -> Mapping[str, object]:
        return {
            "dirty_flat_rows": self.dirty_flat_rows,
            "flat_rows": self.flat_rows,
            "key": {
                "artifact_epoch": self.key.artifact_epoch,
                "artifact_token": self.key.artifact_token,
                "policy_digest": self.key.policy_digest,
            },
            "request_indices": self.request_indices,
            "restore_flat_rows": self.restore_flat_rows,
        }


def _request_live_identity(request: RequestReuseLayout) -> Tuple[object, ...]:
    """Compact ABA guard for one already validated immutable request layout."""

    return (
        id(request),
        int(request.request_index),
        str(request.request_token),
        int(request.flat_row_start),
        id(request.logical_positions),
        int(request.query_start),
        id(request.dirty_request_rows),
        tuple(
            (
                id(segment),
                str(segment.segment_token),
                int(segment.logical_start),
                int(segment.length),
                int(segment.artifact_row_start),
            )
            for segment in request.segments
        ),
        request.artifact_token,
        int(request.artifact_epoch),
        str(request.policy_digest),
        str(request.fallback_reason),
        id(request._clean_request_rows_cache),
        id(request._flat_rows_cache),
        id(request._dirty_flat_rows_cache),
        id(request._clean_flat_rows_cache),
        str(request._cached_digest),
    )


def _group_live_identity(group: ArtifactGroup) -> Tuple[object, ...]:
    key = group.key
    return (
        id(group),
        id(key),
        str(key.artifact_token),
        int(key.artifact_epoch),
        str(key.policy_digest),
        id(group.request_indices),
        id(group.flat_rows),
        id(group.restore_flat_rows),
        id(group.dirty_flat_rows),
    )


def _plan_live_identity(plan: "BatchedReusePlan") -> Tuple[object, ...]:
    """Recheck frozen object/tuple bindings without rescanning token rows."""

    return (
        str(plan.batch_token),
        id(plan.requests),
        tuple(_request_live_identity(request) for request in plan.requests),
        id(plan.global_rows),
        id(plan.local_clean_rows),
        id(plan.local_dirty_rows),
        id(plan.groups),
        tuple(_group_live_identity(group) for group in plan.groups),
        int(plan.format_version),
        str(plan.digest_semantics),
    )


@dataclass(frozen=True)
class BatchedReusePlan:
    """Immutable all-request certificate for one flattened forward."""

    batch_token: str
    requests: Tuple[RequestReuseLayout, ...]
    global_rows: Tuple[int, ...]
    local_clean_rows: Tuple[int, ...]
    local_dirty_rows: Tuple[int, ...]
    groups: Tuple[ArtifactGroup, ...]
    format_version: int = BATCH_REUSE_FORMAT_VERSION
    digest_semantics: str = STABLE_DIGEST_SEMANTICS
    _cached_digest: str = field(init=False, repr=False, compare=False)
    _live_identity: Tuple[object, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _nonempty_string(self.batch_token, "batch_token")
        if self.format_version != BATCH_REUSE_FORMAT_VERSION:
            raise ValueError("batched reuse format is incompatible")
        if self.digest_semantics != STABLE_DIGEST_SEMANTICS:
            raise ValueError("batched reuse digest semantics are incompatible")
        if type(self.requests) is not tuple or not self.requests:
            raise ValueError("requests must be a non-empty immutable tuple")
        if type(self.groups) is not tuple:
            raise TypeError("groups must be an immutable tuple")
        if any(not isinstance(item, RequestReuseLayout) for item in self.requests):
            raise TypeError("requests contain an invalid layout")
        if any(not isinstance(item, ArtifactGroup) for item in self.groups):
            raise TypeError("groups contain an invalid value")

        expected_start = 0
        for request_index, request in enumerate(self.requests):
            if request.request_index != request_index:
                raise ValueError("request indices must equal their batch order")
            if request.flat_row_start != expected_start:
                raise ValueError("request spans must be disjoint and tile the flat batch")
            expected_start = request.flat_row_end

        expected_global = tuple(range(expected_start))
        global_rows = _strict_rows(
            self.global_rows, name="global_rows", allow_empty=False
        )
        if global_rows != expected_global:
            raise ValueError("global heads must cover every flattened batch row")
        clean_rows = _strict_rows(self.local_clean_rows, name="local_clean_rows")
        dirty_rows = _strict_rows(self.local_dirty_rows, name="local_dirty_rows")
        validate_disjoint_union(
            left=clean_rows,
            right=dirty_rows,
            universe=global_rows,
            left_name="local clean rows",
            right_name="local dirty rows",
        )
        expected_clean = tuple(
            row for request in self.requests for row in request.clean_flat_rows
        )
        expected_dirty = tuple(
            row for request in self.requests for row in request.dirty_flat_rows
        )
        if clean_rows != expected_clean or dirty_rows != expected_dirty:
            raise ValueError("batch local row domains differ from request certificates")

        expected_group_members = {
            request.request_index
            for request in self.requests
            if request.reuse_enabled
        }
        seen_members: set[int] = set()
        previous_key: Optional[ArtifactGroupKey] = None
        for group in self.groups:
            if previous_key is not None and group.key <= previous_key:
                raise ValueError("artifact groups must be unique and key-sorted")
            previous_key = group.key
            member_set = set(group.request_indices)
            if seen_members.intersection(member_set):
                raise ValueError("artifact group request membership must be disjoint")
            seen_members.update(member_set)
            members = tuple(self.requests[index] for index in group.request_indices)
            if any(member.group_key != group.key for member in members):
                raise ValueError("artifact group key differs from a member request")
            expected_rows = tuple(row for member in members for row in member.flat_rows)
            expected_restore = tuple(
                row for member in members for row in member.clean_flat_rows
            )
            expected_group_dirty = tuple(
                row for member in members for row in member.dirty_flat_rows
            )
            if (
                group.flat_rows != expected_rows
                or group.restore_flat_rows != expected_restore
                or group.dirty_flat_rows != expected_group_dirty
            ):
                raise ValueError("artifact group rows differ from member requests")
        if seen_members != expected_group_members:
            raise ValueError("artifact groups must union to every reusable request")
        object.__setattr__(
            self, "_cached_digest", _canonical_digest(self._payload())
        )
        object.__setattr__(self, "_live_identity", _plan_live_identity(self))

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def q_rows(self) -> int:
        return len(self.global_rows)

    @property
    def reusable_request_indices(self) -> Tuple[int, ...]:
        return tuple(
            request.request_index
            for request in self.requests
            if request.reuse_enabled
        )

    def request_to_flat(self, request_index: int, request_row: int) -> int:
        request_index = _strict_int(request_index, "request_index")
        if request_index < 0 or request_index >= self.batch_size:
            raise IndexError("request_index is outside the batch")
        return self.requests[request_index].request_row_to_flat(request_row)

    def flat_to_request(self, flat_row: int) -> Tuple[int, int]:
        flat_row = _strict_int(flat_row, "flat_row")
        if flat_row < 0 or flat_row >= self.q_rows:
            raise IndexError("flat_row is outside the batch")
        starts = tuple(request.flat_row_start for request in self.requests)
        request_index = bisect_right(starts, flat_row) - 1
        request = self.requests[request_index]
        return request_index, request.flat_row_to_request(flat_row)

    def flat_to_logical(self, flat_row: int) -> Tuple[int, int]:
        request_index, request_row = self.flat_to_request(flat_row)
        return request_index, self.requests[request_index].logical_positions[request_row]

    def as_dense(self, reason: str) -> "BatchedReusePlan":
        reason = _nonempty_string(reason, "fallback reason")
        requests = tuple(request.as_dense(reason) for request in self.requests)
        return _assemble_plan(batch_token=self.batch_token, requests=requests)

    def _payload(self) -> Mapping[str, object]:
        return {
            "batch_token": self.batch_token,
            "digest_semantics": self.digest_semantics,
            "format_version": self.format_version,
            "global_rows": self.global_rows,
            "groups": tuple(group._payload() for group in self.groups),
            "local_clean_rows": self.local_clean_rows,
            "local_dirty_rows": self.local_dirty_rows,
            "requests": tuple(request._payload() for request in self.requests),
        }

    @property
    def digest(self) -> str:
        return self._cached_digest

    def validate(self, *, expected_digest: Optional[str] = None) -> None:
        """Recheck frozen bindings without rebuilding 8K row partitions.

        Construction still performs the complete semantic proof.  Tuples and
        nested dataclasses are immutable thereafter, so identity plus scalar
        ABA guards are sufficient for the many live checks performed by one
        forward-wide transaction.  Any replaced nested object, tuple, segment,
        artifact key, or cached digest fails closed before omission.
        """

        if _plan_live_identity(self) != self._live_identity:
            raise ValueError("batched reuse plan immutable binding changed")
        if expected_digest is not None and self.digest != expected_digest:
            raise ValueError("batched reuse plan digest changed")

    def _rebind_certified_artifacts(
        self,
        artifact_bindings: Sequence[Optional[Tuple[str, int, str]]],
    ) -> "BatchedReusePlan":
        """Rebind one layer without rebuilding immutable row partitions."""

        self.validate()
        bindings = tuple(artifact_bindings)
        if len(bindings) != len(self.requests):
            raise ValueError("artifact binding count differs from batch size")
        rebound_requests = []
        for request, binding in zip(self.requests, bindings):
            if request.reuse_enabled:
                if type(binding) is not tuple or len(binding) != 3:
                    raise ValueError("reusable request has no artifact binding")
                token, epoch, policy = binding
                rebound_requests.append(
                    request._rebind_certified_artifact(
                        artifact_token=token,
                        artifact_epoch=epoch,
                        policy_digest=policy,
                    )
                )
            else:
                if binding is not None:
                    raise ValueError("dense request received an artifact binding")
                rebound_requests.append(request)
        requests = tuple(rebound_requests)

        groups = []
        for old_group in self.groups:
            members = tuple(requests[index] for index in old_group.request_indices)
            if not members:
                raise ValueError("certified artifact group lost every member")
            key = members[0].group_key
            if key is None or any(member.group_key != key for member in members):
                raise ValueError("artifact rebind changed certified group membership")
            group = object.__new__(ArtifactGroup)
            object.__setattr__(group, "key", key)
            object.__setattr__(group, "request_indices", old_group.request_indices)
            object.__setattr__(group, "flat_rows", old_group.flat_rows)
            object.__setattr__(
                group, "restore_flat_rows", old_group.restore_flat_rows
            )
            object.__setattr__(group, "dirty_flat_rows", old_group.dirty_flat_rows)
            groups.append(group)

        rebound = object.__new__(BatchedReusePlan)
        object.__setattr__(rebound, "batch_token", self.batch_token)
        object.__setattr__(rebound, "requests", requests)
        object.__setattr__(rebound, "global_rows", self.global_rows)
        object.__setattr__(rebound, "local_clean_rows", self.local_clean_rows)
        object.__setattr__(rebound, "local_dirty_rows", self.local_dirty_rows)
        object.__setattr__(rebound, "groups", tuple(groups))
        object.__setattr__(rebound, "format_version", self.format_version)
        object.__setattr__(rebound, "digest_semantics", self.digest_semantics)
        object.__setattr__(
            rebound, "_cached_digest", _canonical_digest(rebound._payload())
        )
        object.__setattr__(rebound, "_live_identity", _plan_live_identity(rebound))
        rebound.validate()
        return rebound


def _dense_request(
    *,
    request_index: int,
    request_token: str,
    flat_row_start: int,
    logical_positions: Tuple[int, ...],
    reason: str,
) -> RequestReuseLayout:
    return RequestReuseLayout(
        request_index=request_index,
        request_token=request_token,
        flat_row_start=flat_row_start,
        logical_positions=logical_positions,
        query_start=0,
        dirty_request_rows=tuple(range(len(logical_positions))),
        fallback_reason=reason,
    )


def _parse_segment(value: object, index: int) -> SegmentBinding:
    if not isinstance(value, Mapping):
        raise TypeError(f"segment {index} must be a mapping")
    if "segment_token" in value:
        token = value["segment_token"]
    elif "seg_hash" in value:
        token = value["seg_hash"]
    else:
        raise KeyError(f"segment {index} is missing segment_token")
    if "logical_start" in value:
        logical_start = value["logical_start"]
    elif "global_offset" in value:
        logical_start = value["global_offset"]
    else:
        raise KeyError(f"segment {index} is missing logical_start")
    return SegmentBinding(
        segment_token=token,
        logical_start=logical_start,
        length=value["length"],
        artifact_row_start=value.get("artifact_row_start", 0),
    )


def _parse_request(
    *,
    request_index: int,
    flat_row_start: int,
    logical_positions: Tuple[int, ...],
    metadata: object,
) -> RequestReuseLayout:
    default_request_token = f"request-index:{request_index}"
    if metadata is None:
        return _dense_request(
            request_index=request_index,
            request_token=default_request_token,
            flat_row_start=flat_row_start,
            logical_positions=logical_positions,
            reason="no_reuse_plan",
        )
    if not isinstance(metadata, Mapping):
        raise TypeError("request reuse metadata must be a mapping")

    request_token = metadata.get(
        "request_token",
        metadata.get("benchmark_request_id", default_request_token),
    )
    request_token = _nonempty_string(request_token, "request_token")
    if metadata.get("mode", "restore") != "restore":
        return _dense_request(
            request_index=request_index,
            request_token=request_token,
            flat_row_start=flat_row_start,
            logical_positions=logical_positions,
            reason="request_not_in_restore_mode",
        )
    if metadata.get("reuse_enabled", True) is not True:
        return _dense_request(
            request_index=request_index,
            request_token=request_token,
            flat_row_start=flat_row_start,
            logical_positions=logical_positions,
            reason="reuse_not_enabled",
        )
    if "batch_offset" in metadata and _strict_int(
        metadata["batch_offset"], "batch_offset"
    ) != flat_row_start:
        raise ValueError("request batch_offset differs from ForwardBatch geometry")
    if "row_count" in metadata and _strict_int(
        metadata["row_count"], "row_count"
    ) != len(logical_positions):
        raise ValueError("request row_count differs from ForwardBatch geometry")
    if "logical_positions" in metadata:
        declared_positions = tuple(metadata["logical_positions"])
        if declared_positions != logical_positions:
            raise ValueError(
                "request logical_positions differ from ForwardBatch positions"
            )

    artifact_token = _nonempty_string(metadata["artifact_token"], "artifact_token")
    artifact_epoch = _strict_int(metadata["artifact_epoch"], "artifact_epoch")
    if artifact_epoch <= 0:
        raise ValueError("artifact_epoch must be positive")
    policy_value = metadata.get("policy_digest", metadata.get("policy_hash"))
    policy_digest = _nonempty_string(policy_value, "policy_digest")
    query_start = _strict_int(metadata["query_start"], "query_start")
    segments_value = metadata["segments"]
    if isinstance(segments_value, (str, bytes, bytearray, Mapping)):
        raise TypeError("segments must be a sequence")
    segments = tuple(
        _parse_segment(value, index) for index, value in enumerate(segments_value)
    )
    dirty_value = metadata.get(
        "dirty_request_rows", metadata.get("dirty_rows", metadata.get("dirty_output_rows"))
    )
    if dirty_value is None:
        raise KeyError("request reuse metadata is missing dirty_request_rows")
    dirty_rows = tuple(dirty_value)

    return RequestReuseLayout(
        request_index=request_index,
        request_token=request_token,
        flat_row_start=flat_row_start,
        logical_positions=logical_positions,
        query_start=query_start,
        dirty_request_rows=dirty_rows,
        segments=segments,
        artifact_token=artifact_token,
        artifact_epoch=artifact_epoch,
        policy_digest=policy_digest,
    )


def _assemble_plan(
    *, batch_token: str, requests: Tuple[RequestReuseLayout, ...]
) -> BatchedReusePlan:
    q_rows = sum(request.row_count for request in requests)
    global_rows = tuple(range(q_rows))
    local_clean_rows = tuple(
        row for request in requests for row in request.clean_flat_rows
    )
    local_dirty_rows = tuple(
        row for request in requests for row in request.dirty_flat_rows
    )

    by_key: dict[ArtifactGroupKey, list[RequestReuseLayout]] = {}
    for request in requests:
        key = request.group_key
        if key is not None:
            by_key.setdefault(key, []).append(request)
    groups = []
    for key in sorted(by_key):
        members = tuple(by_key[key])
        groups.append(
            ArtifactGroup(
                key=key,
                request_indices=tuple(member.request_index for member in members),
                flat_rows=tuple(row for member in members for row in member.flat_rows),
                restore_flat_rows=tuple(
                    row for member in members for row in member.clean_flat_rows
                ),
                dirty_flat_rows=tuple(
                    row for member in members for row in member.dirty_flat_rows
                ),
            )
        )
    return BatchedReusePlan(
        batch_token=batch_token,
        requests=requests,
        global_rows=global_rows,
        local_clean_rows=local_clean_rows,
        local_dirty_rows=local_dirty_rows,
        groups=tuple(groups),
    )


def assemble_validated_batched_reuse_plan(
    *, batch_token: str, requests: Sequence[RequestReuseLayout]
) -> BatchedReusePlan:
    """Assemble server-validated request layouts without reparsing client data.

    The serving backend uses this after it has resolved exact artifact epochs
    and token rows.  Client metadata is never trusted to declare those epochs.
    """

    normalized = tuple(requests)
    if not normalized:
        raise BatchReuseValidationError("a batched reuse plan needs requests")
    return _assemble_plan(
        batch_token=_nonempty_string(batch_token, "batch_token"),
        requests=normalized,
    )


def rebind_validated_batched_reuse_plan(
    *,
    template: BatchedReusePlan,
    artifact_bindings: Sequence[Optional[Tuple[str, int, str]]],
) -> BatchedReusePlan:
    """Rebind layer artifacts onto an already certified batch geometry.

    This server-only path preserves the exact v1 payload and digest while
    avoiding repeated construction of 8K global/clean/dirty row tuples.  The
    template performs an immutable live check before any rebound object is
    published.
    """

    if not isinstance(template, BatchedReusePlan):
        raise TypeError("artifact rebind template has an invalid type")
    return template._rebind_certified_artifacts(artifact_bindings)


def build_batched_reuse_plan(
    *,
    batch_token: str,
    request_positions: Sequence[Sequence[int]],
    request_metadata: Sequence[object],
    flat_row_starts: Optional[Sequence[int]] = None,
    fail_closed: bool = True,
) -> BatchedReusePlan:
    """Build a ragged multi-request reuse certificate.

    Invalid per-request metadata is converted to an all-dirty request when
    ``fail_closed`` is true.  Invalid row counts/offsets always raise because
    continuing could associate one request's artifact with another request's
    flattened rows.
    """

    batch_token = _nonempty_string(batch_token, "batch_token")
    if isinstance(request_positions, (str, bytes, bytearray, Mapping)):
        raise BatchReuseValidationError("request_positions must be a sequence")
    if isinstance(request_metadata, (str, bytes, bytearray, Mapping)):
        raise BatchReuseValidationError("request_metadata must be a sequence")
    position_groups = tuple(request_positions)
    metadata_groups = tuple(request_metadata)
    if not position_groups:
        raise BatchReuseValidationError("a batch must contain at least one request")
    if len(metadata_groups) != len(position_groups):
        raise BatchReuseValidationError(
            "request metadata count must equal the ForwardBatch batch size"
        )

    normalized_positions = []
    for index, values in enumerate(position_groups):
        try:
            positions = _strict_rows(
                tuple(values),
                name=f"request {index} logical_positions",
                allow_empty=False,
            )
        except (TypeError, ValueError) as error:
            raise BatchReuseValidationError(str(error)) from error
        normalized_positions.append(positions)

    expected_starts = []
    cursor = 0
    for positions in normalized_positions:
        expected_starts.append(cursor)
        cursor += len(positions)
    if flat_row_starts is None:
        starts = tuple(expected_starts)
    else:
        try:
            starts = tuple(
                _strict_int(value, "flat_row_start") for value in flat_row_starts
            )
        except (TypeError, ValueError) as error:
            raise BatchReuseValidationError(str(error)) from error
        if len(starts) != len(expected_starts) or starts != tuple(expected_starts):
            raise BatchReuseValidationError(
                "request flat-row starts must be disjoint and tile the batch"
            )

    requests = []
    for index, (positions, metadata, start) in enumerate(
        zip(normalized_positions, metadata_groups, starts)
    ):
        try:
            request = _parse_request(
                request_index=index,
                flat_row_start=start,
                logical_positions=positions,
                metadata=metadata,
            )
        except (KeyError, TypeError, ValueError) as error:
            if not fail_closed:
                raise
            request_token = f"request-index:{index}"
            if isinstance(metadata, Mapping):
                candidate = metadata.get(
                    "request_token", metadata.get("benchmark_request_id")
                )
                if isinstance(candidate, str) and candidate:
                    request_token = candidate
            request = _dense_request(
                request_index=index,
                request_token=request_token,
                flat_row_start=start,
                logical_positions=positions,
                reason=f"invalid_reuse_metadata:{type(error).__name__}:{error}",
            )
        requests.append(request)
    return _assemble_plan(batch_token=batch_token, requests=tuple(requests))


def build_forward_batch_reuse_plan(
    forward_batch: object,
    *,
    batch_token: Optional[str] = None,
    fail_closed: bool = True,
) -> BatchedReusePlan:
    """Adapt a duck-typed SGLang ``ForwardBatch`` without assuming BS=1."""

    batch_size = _strict_int(getattr(forward_batch, "batch_size", None), "batch_size")
    if batch_size <= 0:
        raise BatchReuseValidationError("ForwardBatch.batch_size must be positive")
    lengths_raw = _cpu_list(
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        "extend_seq_lens_cpu",
    )
    if len(lengths_raw) != batch_size:
        raise BatchReuseValidationError(
            "extend_seq_lens_cpu length must equal ForwardBatch.batch_size"
        )
    try:
        lengths = tuple(_strict_int(value, "extend sequence length") for value in lengths_raw)
    except (TypeError, ValueError) as error:
        raise BatchReuseValidationError(str(error)) from error
    if any(length <= 0 for length in lengths):
        raise BatchReuseValidationError("every request must contribute at least one row")

    positions_raw = _cpu_list(getattr(forward_batch, "positions", None), "positions")
    if len(positions_raw) != sum(lengths):
        raise BatchReuseValidationError(
            "flattened positions length must equal sum(extend_seq_lens_cpu)"
        )
    try:
        flat_positions = tuple(_strict_int(value, "position") for value in positions_raw)
    except (TypeError, ValueError) as error:
        raise BatchReuseValidationError(str(error)) from error

    expected_starts = []
    request_positions = []
    cursor = 0
    for length in lengths:
        expected_starts.append(cursor)
        request_positions.append(flat_positions[cursor : cursor + length])
        cursor += length

    starts_value = getattr(forward_batch, "extend_start_loc", None)
    if starts_value is None:
        starts = tuple(expected_starts)
    else:
        starts_raw = _cpu_list(starts_value, "extend_start_loc")
        try:
            starts = tuple(_strict_int(value, "extend_start_loc") for value in starts_raw)
        except (TypeError, ValueError) as error:
            raise BatchReuseValidationError(str(error)) from error
        if starts != tuple(expected_starts):
            raise BatchReuseValidationError(
                "extend_start_loc must be the exclusive prefix sum of request rows"
            )

    metadata = getattr(forward_batch, "redknot_reuse_plan", None)
    if metadata is None:
        metadata_values: Tuple[object, ...] = (None,) * batch_size
    else:
        if isinstance(metadata, (str, bytes, bytearray, Mapping)):
            raise BatchReuseValidationError(
                "redknot_reuse_plan must contain one entry per request"
            )
        metadata_values = tuple(metadata)
        if len(metadata_values) != batch_size:
            raise BatchReuseValidationError(
                "redknot_reuse_plan length must equal ForwardBatch.batch_size"
            )

    if batch_token is None:
        supplied = getattr(forward_batch, "redknot_batch_token", None)
        if isinstance(supplied, str) and supplied:
            batch_token = supplied
        else:
            # This identifies the immutable geometry, not a long-lived request.
            # The final plan digest additionally binds every artifact generation.
            batch_token = _canonical_digest(
                {
                    "batch_size": batch_size,
                    "lengths": lengths,
                    "positions": flat_positions,
                }
            )
    return build_batched_reuse_plan(
        batch_token=batch_token,
        request_positions=tuple(request_positions),
        request_metadata=metadata_values,
        flat_row_starts=starts,
        fail_closed=fail_closed,
    )


__all__ = [
    "ArtifactGroup",
    "ArtifactGroupKey",
    "BATCH_REUSE_FORMAT_VERSION",
    "BatchReuseValidationError",
    "BatchedReusePlan",
    "RequestReuseLayout",
    "STABLE_DIGEST_SEMANTICS",
    "SegmentBinding",
    "build_batched_reuse_plan",
    "build_forward_batch_reuse_plan",
    "assemble_validated_batched_reuse_plan",
    "rebind_validated_batched_reuse_plan",
    "validate_disjoint_union",
]
