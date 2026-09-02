"""Packed CUDA representation for sparse DeepSeek-V4 ``wq_b`` output.

The control-plane plan in :mod:`dsv4_sparse_q` describes which logical
head/row pairs must be projected.  This module is the matching data plane: it
stores exactly those pairs in one contiguous ``[projected_head_rows, D]``
tensor.  It never constructs either the native ``[T, 64, D]`` presentation or
the rank-local ``[T, H_owned, D]`` tensor with uninitialised clean-local slots.

The object is deliberately *not* a ``torch.Tensor``.  Consumers must select a
certified global or dirty-local scope explicitly; passing it to a native MLA
kernel therefore fails closed instead of making omitted values look like a
dense Q activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, Optional, Sequence, Tuple


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _tensor_identity(tensor: object) -> Tuple[object, ...]:
    try:
        version: object = int(getattr(tensor, "_version"))
    except RuntimeError:
        version = "inference-immutable"
    return (
        id(tensor),
        int(tensor.data_ptr()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        version,
    )


@dataclass(frozen=True)
class PackedQRun:
    """One contiguous head run in the packed projection arena."""

    scope: str
    start_axis: int
    end_axis: int
    source_rows: int
    value_begin: int
    value_end: int

    def __post_init__(self) -> None:
        if self.scope not in ("global", "local"):
            raise ValueError("packed Q scope must be global or local")
        for name in (
            "start_axis",
            "end_axis",
            "source_rows",
            "value_begin",
            "value_end",
        ):
            _strict_int(getattr(self, name), name)
        if self.start_axis < 0 or self.end_axis <= self.start_axis:
            raise ValueError("packed Q head run is empty")
        if self.source_rows < 0 or self.value_begin < 0:
            raise ValueError("packed Q row/offset is negative")
        expected = self.source_rows * self.head_count
        if self.value_end - self.value_begin != expected:
            raise ValueError("packed Q run size does not match its geometry")

    @property
    def head_count(self) -> int:
        return self.end_axis - self.start_axis

    @property
    def axes(self) -> Tuple[int, ...]:
        return tuple(range(self.start_axis, self.end_axis))


def build_packed_q_runs(plan: object) -> Tuple[PackedQRun, ...]:
    """Lay out global/full-row runs followed by local/dirty-row runs."""

    validate = getattr(plan, "validate", None)
    if not callable(validate):
        raise TypeError("packed Q requires a validated sparse-Q plan")
    validate()
    cursor = 0
    result = []
    for scope, source_rows, head_runs in (
        ("global", int(plan.q_rows), tuple(plan.global_runs)),
        ("local", len(tuple(plan.online_local_rows)), tuple(plan.local_runs)),
    ):
        for head_run in head_runs:
            count = source_rows * int(head_run.head_count)
            result.append(
                PackedQRun(
                    scope=scope,
                    start_axis=int(head_run.start_axis),
                    end_axis=int(head_run.end_axis),
                    source_rows=source_rows,
                    value_begin=cursor,
                    value_end=cursor + count,
                )
            )
            cursor += count
    if cursor != int(plan.projected_head_rows):
        raise ValueError("packed Q layout disagrees with projected head-row count")
    return tuple(result)


class PackedSparseQProjection:
    """Immutable handle to the exact projected head rows on one TP rank."""

    def __init__(
        self,
        *,
        plan: object,
        values: object,
        local_rows: object,
        runs: Sequence[PackedQRun],
        projection_token: str,
    ) -> None:
        self.plan = plan
        self.values = values
        self.local_rows = local_rows
        self.runs = tuple(runs)
        self.projection_token = str(projection_token)
        self._values_identity = _tensor_identity(values)
        self._rows_identity = _tensor_identity(local_rows)
        self._axis_runs: Dict[Tuple[str, int], PackedQRun] = {}
        for run in self.runs:
            for axis in run.axes:
                key = (run.scope, axis)
                if key in self._axis_runs:
                    raise ValueError("packed Q contains duplicate scope/head axes")
                self._axis_runs[key] = run
        self.validate()

    @property
    def q_rows(self) -> int:
        return int(self.plan.q_rows)

    @property
    def ndim(self) -> int:
        # Logical attention presentation.  No tensor with this shape exists.
        return 4

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.q_rows, 1, self.owned_head_count, self.head_dim)

    @property
    def owned_head_count(self) -> int:
        return int(self.plan.owned_head_count)

    @property
    def head_dim(self) -> int:
        return int(self.plan.head_dim)

    @property
    def device(self):
        return self.values.device

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def digest(self) -> str:
        payload = (
            str(self.plan.digest),
            self.projection_token,
            tuple(
                (
                    run.scope,
                    run.start_axis,
                    run.end_axis,
                    run.source_rows,
                    run.value_begin,
                    run.value_end,
                )
                for run in self.runs
            ),
        )
        return "sha256:" + sha256(repr(payload).encode("utf-8")).hexdigest()

    def validate(self) -> None:
        import torch

        self.plan.validate()
        if not self.projection_token:
            raise ValueError("packed Q projection token is empty")
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 2:
            raise ValueError("packed Q values must be [projected_head_rows, head_dim]")
        if self.values.device.type != "cuda" or not self.values.is_contiguous():
            raise ValueError("packed Q values must be contiguous CUDA storage")
        if tuple(int(value) for value in self.values.shape) != (
            int(self.plan.projected_head_rows),
            int(self.plan.head_dim),
        ):
            raise ValueError("packed Q values do not match the sparse plan")
        if (
            not isinstance(self.local_rows, torch.Tensor)
            or self.local_rows.ndim != 1
            or self.local_rows.dtype != torch.long
            or self.local_rows.device != self.values.device
            or int(self.local_rows.numel()) != len(tuple(self.plan.online_local_rows))
        ):
            raise ValueError("packed Q dirty-local rows are incompatible")
        if _tensor_identity(self.values) != self._values_identity:
            raise ValueError("packed Q values changed after construction")
        if _tensor_identity(self.local_rows) != self._rows_identity:
            raise ValueError("packed Q row certificate changed after construction")
        if self.runs != build_packed_q_runs(self.plan):
            raise ValueError("packed Q run layout changed")
        expected_keys = {
            ("global", int(axis)) for axis in tuple(self.plan.global_head_axes)
        } | {("local", int(axis)) for axis in tuple(self.plan.local_head_axes)}
        if set(self._axis_runs) != expected_keys:
            raise ValueError("packed Q scope/head coverage is incomplete")

    def _axis_view(self, scope: str, axis: int):
        run = self._axis_runs.get((scope, int(axis)))
        if run is None:
            raise ValueError(f"head axis {axis} is not projected in {scope} scope")
        relative = int(axis) - run.start_axis
        values = self.values.narrow(
            0, run.value_begin, run.value_end - run.value_begin
        ).view(run.source_rows, run.head_count, self.head_dim)
        return values[:, relative : relative + 1, :]

    def select(
        self,
        *,
        scope: str,
        head_axes: Sequence[int],
        row_indices: Optional[object] = None,
    ):
        """Return ``[rows, 1, heads, D]`` for one certified attention launch."""

        import torch

        self.validate()
        axes = tuple(int(axis) for axis in head_axes)
        if not axes or len(axes) != len(set(axes)):
            raise ValueError("packed Q selection requires unique head axes")
        if scope == "global":
            if row_indices is not None:
                raise ValueError("global packed Q already covers every row")
            expected_rows = self.q_rows
        elif scope == "local":
            if row_indices is not self.local_rows:
                raise ValueError("local packed Q requires the certified dirty-row tensor")
            expected_rows = int(self.local_rows.numel())
        else:
            raise ValueError("packed Q selection has an unknown scope")
        pieces = tuple(self._axis_view(scope, axis) for axis in axes)
        selected = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        if tuple(int(value) for value in selected.shape) != (
            expected_rows,
            len(axes),
            self.head_dim,
        ):
            raise RuntimeError("packed Q selection produced an invalid shape")
        return selected.unsqueeze(1)

    def new_attention_output(self, head_dim_v: int):
        head_dim_v = _strict_int(head_dim_v, "head_dim_v")
        if head_dim_v <= 0:
            raise ValueError("head_dim_v must be positive")
        return self.values.new_zeros(
            self.q_rows, 1, self.owned_head_count, head_dim_v
        )


class PackedSparseQBuilder:
    """Single-use writer for a preallocated packed CUDA projection arena."""

    def __init__(self, *, plan: object, values: object, local_rows: object) -> None:
        self.plan = plan
        self.values = values
        self.local_rows = local_rows
        self.runs = build_packed_q_runs(plan)
        self._written: set[Tuple[str, int, int]] = set()

    def write(self, *, scope: str, start_axis: int, end_axis: int, projected) -> None:
        run = next(
            (
                item
                for item in self.runs
                if item.scope == scope
                and item.start_axis == int(start_axis)
                and item.end_axis == int(end_axis)
            ),
            None,
        )
        if run is None:
            raise ValueError("packed Q write is absent from the certified layout")
        key = (scope, int(start_axis), int(end_axis))
        if key in self._written:
            raise ValueError("packed Q run was written twice")
        expected = (run.source_rows, run.head_count, int(self.plan.head_dim))
        if tuple(int(value) for value in projected.shape) != expected:
            raise ValueError(
                f"packed Q write shape {tuple(projected.shape)} != {expected}"
            )
        self.values.narrow(
            0, run.value_begin, run.value_end - run.value_begin
        ).copy_(
            projected.reshape(
                run.source_rows * run.head_count,
                int(self.plan.head_dim),
            )
        )
        self._written.add(key)

    def finish(self, *, projection_token: str) -> PackedSparseQProjection:
        expected = {
            (run.scope, run.start_axis, run.end_axis) for run in self.runs
        }
        if self._written != expected:
            raise ValueError("packed Q projection is incomplete")
        return PackedSparseQProjection(
            plan=self.plan,
            values=self.values,
            local_rows=self.local_rows,
            runs=self.runs,
            projection_token=projection_token,
        )


@dataclass(frozen=True)
class SequentialPackedQReservation:
    """Mutation-free ticket for one future layer write into a shared arena."""

    plan: object
    values: object
    arena_token: str
    arena_index: int
    write_ordinal: int
    write_count: int
    version_offset: int
    arena_capacity_nbytes: int
    arena_base_version: int
    projection_token: str

    @property
    def expected_postwrite_version(self) -> int:
        return int(self.arena_base_version) + int(self.version_offset)


class _SequentialPackedQBuilder:
    """One ordered writer; completion advances its owning arena exactly once."""

    def __init__(self, arena, reservation, local_rows) -> None:
        self._arena = arena
        self.reservation = reservation
        self._builder = PackedSparseQBuilder(
            plan=reservation.plan,
            values=reservation.values,
            local_rows=local_rows,
        )
        self._finished = False

    @property
    def plan(self):
        return self.reservation.plan

    @property
    def values(self):
        return self.reservation.values

    @property
    def local_rows(self):
        return self._builder.local_rows

    def write(self, **kwargs) -> None:
        if self._finished:
            raise RuntimeError("sequential packed-Q builder is already finished")
        self._builder.write(**kwargs)

    def finish(self) -> PackedSparseQProjection:
        if self._finished:
            raise RuntimeError("sequential packed-Q builder may finish once")
        projection = self._builder.finish(
            projection_token=self.reservation.projection_token
        )
        self._arena._finish_layer(self.reservation, projection)
        for name in (
            "arena_token",
            "arena_index",
            "write_ordinal",
            "write_count",
            "version_offset",
            "arena_capacity_nbytes",
            "arena_base_version",
        ):
            setattr(projection, name, getattr(self.reservation, name))
        self._finished = True
        return projection


class SequentialPackedQArena:
    """One persistent packed-Q tensor reused sequentially across layers.

    Reservations are cheap Python objects and narrow views.  There is exactly
    one real backing tensor, sized to the largest layer plan, so a 37-layer
    forward never retains 37 packed activations.
    """

    def __init__(
        self,
        *,
        plans: Sequence[object],
        backing: object,
        arena_token: str,
    ) -> None:
        plans = tuple(plans)
        if not plans:
            raise ValueError("sequential packed-Q arena requires layer plans")
        for plan in plans:
            validate = getattr(plan, "validate", None)
            if not callable(validate):
                raise TypeError("sequential packed-Q plan has no validator")
            validate()
        layers = tuple(int(plan.layer_id) for plan in plans)
        if layers != tuple(sorted(set(layers))):
            raise ValueError("sequential packed-Q plans must be layer-sorted")
        if not isinstance(arena_token, str) or not arena_token:
            raise ValueError("sequential packed-Q arena token must be non-empty")
        dims = {int(plan.head_dim) for plan in plans}
        if len(dims) != 1:
            raise ValueError("sequential packed-Q plans disagree on head_dim")
        max_rows = max(int(plan.projected_head_rows) for plan in plans)
        if tuple(int(value) for value in backing.shape) != (
            max_rows,
            next(iter(dims)),
        ):
            raise ValueError("sequential packed-Q backing has another geometry")
        try:
            base_version = int(backing._version)
        except RuntimeError as error:
            raise ValueError(
                "sequential packed-Q backing must track mutation versions"
            ) from error
        if base_version < 0:
            raise ValueError("sequential packed-Q base version is invalid")
        self.plans = plans
        self.backing = backing
        self.arena_token = arena_token
        self.arena_index = 0
        self.arena_base_version = base_version
        self.arena_capacity_nbytes = int(backing.numel()) * int(
            backing.element_size()
        )
        self._backing_identity = (
            id(backing),
            int(backing.data_ptr()),
            tuple(int(value) for value in backing.shape),
            tuple(int(value) for value in backing.stride()),
            str(backing.dtype),
            str(backing.device),
        )
        reservations = []
        version_offset = 0
        for ordinal, plan in enumerate(plans):
            write_count = len(build_packed_q_runs(plan))
            if write_count <= 0:
                raise ValueError("sequential packed-Q layer has no writes")
            version_offset += write_count
            token_payload = (
                arena_token,
                int(plan.layer_id),
                str(plan.digest),
                ordinal,
                write_count,
                version_offset,
            )
            projection_token = "sequential-packed-q:sha256:" + sha256(
                repr(token_payload).encode("utf-8")
            ).hexdigest()
            reservations.append(
                SequentialPackedQReservation(
                    plan=plan,
                    values=backing.narrow(
                        0, 0, int(plan.projected_head_rows)
                    ),
                    arena_token=arena_token,
                    arena_index=0,
                    write_ordinal=ordinal,
                    write_count=write_count,
                    version_offset=version_offset,
                    arena_capacity_nbytes=self.arena_capacity_nbytes,
                    arena_base_version=base_version,
                    projection_token=projection_token,
                )
            )
        self._reservations = tuple(reservations)
        self._by_layer = {
            int(item.plan.layer_id): item for item in self._reservations
        }
        self._next_ordinal = 0
        self._active_ordinal: Optional[int] = None
        self._completed_projection: Optional[PackedSparseQProjection] = None

    @classmethod
    def allocate(
        cls,
        plans: Sequence[object],
        *,
        device: object,
        dtype: object,
        arena_token: str,
    ) -> "SequentialPackedQArena":
        plans = tuple(plans)
        if not plans:
            raise ValueError("sequential packed-Q allocation needs plans")
        head_dims = {int(plan.head_dim) for plan in plans}
        if len(head_dims) != 1:
            raise ValueError("sequential packed-Q plans disagree on head_dim")
        max_rows = max(int(plan.projected_head_rows) for plan in plans)
        import torch

        # SGLang model execution may run under inference_mode.  Reservation
        # receipts need a real version counter, so create the persistent arena
        # explicitly outside inference tensor creation.
        with torch.inference_mode(False):
            backing = torch.empty(
                (max_rows, next(iter(head_dims))),
                device=device,
                dtype=dtype,
            )
        return cls(plans=plans, backing=backing, arena_token=arena_token)

    @property
    def layer_ids(self) -> Tuple[int, ...]:
        return tuple(int(plan.layer_id) for plan in self.plans)

    @property
    def device_nbytes(self) -> int:
        return self.arena_capacity_nbytes

    def _validate_backing(self) -> int:
        current = (
            id(self.backing),
            int(self.backing.data_ptr()),
            tuple(int(value) for value in self.backing.shape),
            tuple(int(value) for value in self.backing.stride()),
            str(self.backing.dtype),
            str(self.backing.device),
        )
        if current != self._backing_identity:
            raise RuntimeError("sequential packed-Q backing identity changed")
        try:
            return int(self.backing._version)
        except RuntimeError as error:
            raise RuntimeError(
                "sequential packed-Q backing lost its version counter"
            ) from error

    def reservation_for(self, layer_id: int) -> SequentialPackedQReservation:
        try:
            return self._by_layer[int(layer_id)]
        except KeyError as error:
            raise KeyError("layer is outside the sequential Q arena") from error

    def begin_layer(self, layer_id: int, local_rows: object):
        reservation = self.reservation_for(layer_id)
        if self._active_ordinal is not None:
            raise RuntimeError("another sequential packed-Q layer is unfinished")
        if reservation.write_ordinal != self._next_ordinal:
            raise RuntimeError("sequential packed-Q layer order changed")
        prior_offset = reservation.version_offset - reservation.write_count
        expected_before = reservation.arena_base_version + prior_offset
        if self._validate_backing() != expected_before:
            raise RuntimeError("sequential packed-Q pre-write version changed")
        self._active_ordinal = reservation.write_ordinal
        return _SequentialPackedQBuilder(self, reservation, local_rows)

    def _finish_layer(self, reservation, projection) -> None:
        if self._active_ordinal != reservation.write_ordinal:
            raise RuntimeError("sequential packed-Q finish has no active ticket")
        if projection.values is not reservation.values:
            raise ValueError("sequential packed-Q projection replaced its arena view")
        if self._validate_backing() != reservation.expected_postwrite_version:
            raise RuntimeError("sequential packed-Q post-write version mismatch")
        projection.validate()
        self._active_ordinal = None
        self._next_ordinal += 1
        self._completed_projection = projection

    def validate_completed_projection(
        self,
        *,
        layer_id: int,
        projection: PackedSparseQProjection,
        local_rows: object,
    ) -> SequentialPackedQReservation:
        """O(1) post-write fence for the just-finished sequential slot.

        ``_finish_layer`` already performed the complete sparse plan/run/tensor
        validation before returning the projection to model execution.  The
        composite receipt only needs to prove that the exact returned object
        still names the reserved arena view and expected post-write version;
        repeating the full plan validation and digest construction at every
        layer unnecessarily serializes the serving stream.
        """

        reservation = self.reservation_for(layer_id)
        if self._active_ordinal is not None:
            raise RuntimeError("sequential packed-Q layer is still active")
        if self._next_ordinal != reservation.write_ordinal + 1:
            raise RuntimeError("sequential packed-Q completion order changed")
        if self._completed_projection is not projection:
            raise RuntimeError("sequential packed-Q completion object changed")
        if (
            projection.plan is not reservation.plan
            or projection.values is not reservation.values
            or projection.local_rows is not local_rows
            or projection.projection_token != reservation.projection_token
        ):
            raise ValueError("sequential packed-Q completion binding changed")
        for name in (
            "arena_token",
            "arena_index",
            "write_ordinal",
            "write_count",
            "version_offset",
            "arena_capacity_nbytes",
            "arena_base_version",
        ):
            if getattr(projection, name, None) != getattr(reservation, name):
                raise ValueError(
                    f"sequential packed-Q completion {name} changed"
                )
        if self._validate_backing() != reservation.expected_postwrite_version:
            raise RuntimeError("sequential packed-Q completion version changed")
        self._completed_projection = None
        return reservation


__all__ = [
    "PackedQRun",
    "PackedSparseQBuilder",
    "PackedSparseQProjection",
    "SequentialPackedQArena",
    "SequentialPackedQReservation",
    "build_packed_q_runs",
]
