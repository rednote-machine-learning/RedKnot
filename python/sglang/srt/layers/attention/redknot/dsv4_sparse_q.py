"""Pure control-plane contracts for sparse DeepSeek-V4 Q projection.

DeepSeek-V4 MLA has one physical latent KV stream and 64 logical query
heads.  With attention tensor parallelism, ``wq_b`` is column-parallel: at
TP=8 each rank owns eight contiguous logical heads.  A restored RedKnot
prefill therefore needs two different Q projection domains on each rank:

* global heads are projected for every input row;
* offline-local heads are projected only for certified online rows (query
  rows and deterministic boundary repair rows).

This module describes that geometry without importing SGLang, DeepGEMM, or
CUDA.  It deliberately does not execute a collective and does not claim that
a projection succeeded.  :func:`issue_sparse_q_commit_certificate` merely
records an all-rank readiness result supplied by the serving integration.  A
caller must finish its local partial GEMMs, normalization, RoPE, and row
scatter *before* issuing the collective readiness vote.

The optional UE8M0 helper operates on the canonical two-dimensional FP32
block-scale tensor.  DeepGEMM's packed scale has an N-dependent TMA layout, so
the packed tensor must never be sliced directly.  The integration supplies
the existing inverse/forward layout transforms; this module validates and
slices only the canonical scale rows between those transforms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Optional, Sequence, Tuple


SPARSE_Q_PLAN_FORMAT_VERSION = 1
SPARSE_Q_COMMIT_FORMAT_VERSION = 1
DEFAULT_LOGICAL_HEADS = 64
DEFAULT_HEAD_DIM = 512
DEFAULT_TP_SIZE = 8
DEFAULT_UE8M0_BLOCK_SHAPE = (128, 128)


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_sorted_unique(
    values: Sequence[int],
    *,
    name: str,
    allow_empty: bool,
) -> Tuple[int, ...]:
    result = tuple(_strict_int(value, f"{name} entry") for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def rank_owned_logical_heads(
    *,
    total_logical_heads: int,
    tp_size: int,
    tp_rank: int,
) -> Tuple[int, ...]:
    """Return the contiguous logical-head shard owned by one attention rank."""

    total_logical_heads = _strict_int(total_logical_heads, "total_logical_heads")
    tp_size = _strict_int(tp_size, "tp_size")
    tp_rank = _strict_int(tp_rank, "tp_rank")
    if total_logical_heads <= 0 or tp_size <= 0:
        raise ValueError("logical head count and TP size must be positive")
    if total_logical_heads % tp_size:
        raise ValueError("logical heads must divide evenly across attention TP")
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError("tp_rank is outside the attention TP group")
    heads_per_rank = total_logical_heads // tp_size
    start = tp_rank * heads_per_rank
    return tuple(range(start, start + heads_per_rank))


@dataclass(frozen=True, order=True)
class HeadRun:
    """One contiguous rank-local head interval and its ``wq_b`` output rows."""

    start_axis: int
    end_axis: int
    head_dim: int = DEFAULT_HEAD_DIM

    def __post_init__(self) -> None:
        start = _strict_int(self.start_axis, "start_axis")
        end = _strict_int(self.end_axis, "end_axis")
        head_dim = _strict_int(self.head_dim, "head_dim")
        if start < 0 or end <= start:
            raise ValueError("head run must be a non-empty non-negative interval")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

    @property
    def axes(self) -> Tuple[int, ...]:
        return tuple(range(self.start_axis, self.end_axis))

    @property
    def head_count(self) -> int:
        return self.end_axis - self.start_axis

    @property
    def output_row_start(self) -> int:
        return self.start_axis * self.head_dim

    @property
    def output_row_end(self) -> int:
        return self.end_axis * self.head_dim

    @property
    def output_rows(self) -> int:
        return self.output_row_end - self.output_row_start


def contiguous_head_runs(
    head_axes: Sequence[int],
    *,
    owned_head_count: int,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> Tuple[HeadRun, ...]:
    """Coalesce strictly increasing rank-local axes into maximal intervals."""

    owned_head_count = _strict_int(owned_head_count, "owned_head_count")
    head_dim = _strict_int(head_dim, "head_dim")
    if owned_head_count <= 0 or head_dim <= 0:
        raise ValueError("owned_head_count and head_dim must be positive")
    axes = _strict_sorted_unique(
        head_axes, name="head axes", allow_empty=True
    )
    if any(axis < 0 or axis >= owned_head_count for axis in axes):
        raise ValueError("head axis is outside this TP rank's owned shard")
    if not axes:
        return ()

    runs = []
    start = previous = axes[0]
    for axis in axes[1:]:
        if axis != previous + 1:
            runs.append(HeadRun(start, previous + 1, head_dim))
            start = axis
        previous = axis
    runs.append(HeadRun(start, previous + 1, head_dim))
    return tuple(runs)


@dataclass(frozen=True)
class RankLocalSparseQPlan:
    """Immutable head/row geometry for one layer on one attention-TP rank."""

    layer_id: int
    tp_rank: int
    tp_size: int
    total_logical_heads: int
    head_dim: int
    q_rows: int
    local_head_axes: Tuple[int, ...]
    online_local_rows: Tuple[int, ...]
    format_version: int = SPARSE_Q_PLAN_FORMAT_VERSION

    def __post_init__(self) -> None:
        layer_id = _strict_int(self.layer_id, "layer_id")
        head_dim = _strict_int(self.head_dim, "head_dim")
        q_rows = _strict_int(self.q_rows, "q_rows")
        if layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if head_dim <= 0 or q_rows <= 0:
            raise ValueError("head_dim and q_rows must be positive")
        if self.format_version != SPARSE_Q_PLAN_FORMAT_VERSION:
            raise ValueError("sparse-Q plan format is incompatible")

        owned = rank_owned_logical_heads(
            total_logical_heads=self.total_logical_heads,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        local_axes = _strict_sorted_unique(
            self.local_head_axes,
            name="local head axes",
            allow_empty=False,
        )
        if any(axis < 0 or axis >= len(owned) for axis in local_axes):
            raise ValueError("local head axis is outside this TP rank's shard")
        online_rows = _strict_sorted_unique(
            self.online_local_rows,
            name="online local rows",
            allow_empty=True,
        )
        if any(row < 0 or row >= q_rows for row in online_rows):
            raise ValueError("online local row is outside the Q input")

    @property
    def owned_logical_heads(self) -> Tuple[int, ...]:
        return rank_owned_logical_heads(
            total_logical_heads=self.total_logical_heads,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )

    @property
    def owned_head_count(self) -> int:
        return len(self.owned_logical_heads)

    @property
    def global_head_axes(self) -> Tuple[int, ...]:
        local = set(self.local_head_axes)
        return tuple(axis for axis in range(self.owned_head_count) if axis not in local)

    @property
    def local_logical_heads(self) -> Tuple[int, ...]:
        owned = self.owned_logical_heads
        return tuple(owned[axis] for axis in self.local_head_axes)

    @property
    def global_logical_heads(self) -> Tuple[int, ...]:
        owned = self.owned_logical_heads
        return tuple(owned[axis] for axis in self.global_head_axes)

    @property
    def global_runs(self) -> Tuple[HeadRun, ...]:
        return contiguous_head_runs(
            self.global_head_axes,
            owned_head_count=self.owned_head_count,
            head_dim=self.head_dim,
        )

    @property
    def local_runs(self) -> Tuple[HeadRun, ...]:
        return contiguous_head_runs(
            self.local_head_axes,
            owned_head_count=self.owned_head_count,
            head_dim=self.head_dim,
        )

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        return self.q_rows, self.owned_head_count, self.head_dim

    @property
    def full_owned_head_rows(self) -> int:
        return self.q_rows * self.owned_head_count

    @property
    def projected_head_rows(self) -> int:
        return (
            self.q_rows * len(self.global_head_axes)
            + len(self.online_local_rows) * len(self.local_head_axes)
        )

    @property
    def omitted_head_rows(self) -> int:
        return self.full_owned_head_rows - self.projected_head_rows

    @property
    def head_row_saving(self) -> float:
        return self.omitted_head_rows / self.full_owned_head_rows

    def validate(self) -> None:
        """Revalidate all derived geometry at the serving integration boundary."""

        owned = rank_owned_logical_heads(
            total_logical_heads=self.total_logical_heads,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        local_axes = _strict_sorted_unique(
            self.local_head_axes,
            name="local head axes",
            allow_empty=False,
        )
        online_rows = _strict_sorted_unique(
            self.online_local_rows,
            name="online local rows",
            allow_empty=True,
        )
        if any(axis < 0 or axis >= len(owned) for axis in local_axes):
            raise ValueError("local head axis is outside this TP rank's shard")
        if any(row < 0 or row >= self.q_rows for row in online_rows):
            raise ValueError("online local row is outside the Q input")
        if set(self.global_head_axes).intersection(local_axes):
            raise ValueError("global and local sparse-Q axes overlap")
        if set(self.global_head_axes).union(local_axes) != set(range(len(owned))):
            raise ValueError("global and local sparse-Q axes do not cover the shard")
        run_axes = tuple(axis for run in self.global_runs for axis in run.axes)
        if run_axes != self.global_head_axes:
            raise ValueError("global sparse-Q runs do not match global head axes")
        run_axes = tuple(axis for run in self.local_runs for axis in run.axes)
        if run_axes != local_axes:
            raise ValueError("local sparse-Q runs do not match local head axes")
        if self.projected_head_rows + self.omitted_head_rows != self.full_owned_head_rows:
            raise ValueError("sparse-Q head-row accounting is inconsistent")

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "layer_id": self.layer_id,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "total_logical_heads": self.total_logical_heads,
            "owned_logical_heads": list(self.owned_logical_heads),
            "head_dim": self.head_dim,
            "q_rows": self.q_rows,
            "global_head_axes": list(self.global_head_axes),
            "local_head_axes": list(self.local_head_axes),
            "online_local_rows": list(self.online_local_rows),
            "output_shape": list(self.output_shape),
            "projected_head_rows": self.projected_head_rows,
            "omitted_head_rows": self.omitted_head_rows,
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.as_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + sha256(encoded).hexdigest()


def build_rank_local_sparse_q_plan(
    *,
    layer_id: int,
    tp_rank: int,
    tp_size: int,
    q_rows: int,
    offline_local_logical_heads: Sequence[int],
    online_local_rows: Sequence[int],
    total_logical_heads: int = DEFAULT_LOGICAL_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    owned_logical_heads: Optional[Sequence[int]] = None,
) -> RankLocalSparseQPlan:
    """Translate logical offline heads into a deterministic rank-local plan."""

    owned = rank_owned_logical_heads(
        total_logical_heads=total_logical_heads,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    if owned_logical_heads is not None:
        supplied_owned = _strict_sorted_unique(
            owned_logical_heads,
            name="owned logical heads",
            allow_empty=False,
        )
        if supplied_owned != owned:
            raise ValueError("owned logical heads do not match contiguous TP sharding")

    local_logical = _strict_sorted_unique(
        offline_local_logical_heads,
        name="offline local logical heads",
        allow_empty=False,
    )
    owned_to_axis = {head: axis for axis, head in enumerate(owned)}
    if any(head not in owned_to_axis for head in local_logical):
        raise ValueError("offline local logical head is not owned by this TP rank")
    local_axes = tuple(owned_to_axis[head] for head in local_logical)
    online_rows = _strict_sorted_unique(
        online_local_rows,
        name="online local rows",
        allow_empty=True,
    )
    return RankLocalSparseQPlan(
        layer_id=layer_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        total_logical_heads=total_logical_heads,
        head_dim=head_dim,
        q_rows=q_rows,
        local_head_axes=local_axes,
        online_local_rows=online_rows,
    )


def build_sparse_q_plan(
    tp_rank: int,
    tp_size: int,
    total_heads: int,
    local_axes: Sequence[int],
    total_rows: int,
    online_rows: Sequence[int],
    *,
    layer_id: int = 0,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> RankLocalSparseQPlan:
    """Build the minimal rank-local plan consumed by the production hot path.

    ``local_axes`` are relative to this TP rank's contiguous owned-head shard,
    not global logical head ids.  Every complementary axis is global and is
    therefore projected for all ``total_rows``.
    """

    plan = RankLocalSparseQPlan(
        layer_id=layer_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        total_logical_heads=total_heads,
        head_dim=head_dim,
        q_rows=total_rows,
        local_head_axes=tuple(local_axes),
        online_local_rows=tuple(online_rows),
    )
    plan.validate()
    return plan


@dataclass(frozen=True)
class SparseQCommitCertificate:
    """Opaque record issued only after an external all-rank success vote."""

    generation_id: str
    collective_token: str
    projection_token: str
    plan_digest: str
    layer_id: int
    tp_rank: int
    tp_size: int
    ready_rank_count: int
    output_shape: Tuple[int, int, int]
    projected_head_rows: int
    omitted_head_rows: int
    format_version: int = SPARSE_Q_COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty_string(self.generation_id, "generation_id")
        _nonempty_string(self.collective_token, "collective_token")
        _nonempty_string(self.projection_token, "projection_token")
        plan_digest = _nonempty_string(self.plan_digest, "plan_digest")
        if not plan_digest.startswith("sha256:"):
            raise ValueError("plan_digest must be a SHA-256 digest")
        if self.format_version != SPARSE_Q_COMMIT_FORMAT_VERSION:
            raise ValueError("sparse-Q commit format is incompatible")
        if _strict_int(self.layer_id, "layer_id") < 0:
            raise ValueError("layer_id must be non-negative")
        tp_size = _strict_int(self.tp_size, "tp_size")
        tp_rank = _strict_int(self.tp_rank, "tp_rank")
        ready = _strict_int(self.ready_rank_count, "ready_rank_count")
        if tp_size <= 0 or tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError("commit certificate has invalid TP geometry")
        if ready != tp_size:
            raise ValueError("commit certificate requires every TP rank to be ready")
        if (
            not isinstance(self.output_shape, tuple)
            or len(self.output_shape) != 3
            or any(type(value) is not int or value <= 0 for value in self.output_shape)
        ):
            raise ValueError("commit output_shape must contain three positive integers")
        projected = _strict_int(self.projected_head_rows, "projected_head_rows")
        omitted = _strict_int(self.omitted_head_rows, "omitted_head_rows")
        if projected < 0 or omitted < 0:
            raise ValueError("commit head-row counts must be non-negative")

    def validate(
        self,
        plan: RankLocalSparseQPlan,
        *,
        generation_id: str,
        collective_token: str,
        projection_token: str,
    ) -> None:
        """Fail closed if any plan, forward, collective, or tensor token changed."""

        expected = (
            plan.digest,
            plan.layer_id,
            plan.tp_rank,
            plan.tp_size,
            plan.output_shape,
            plan.projected_head_rows,
            plan.omitted_head_rows,
        )
        actual = (
            self.plan_digest,
            self.layer_id,
            self.tp_rank,
            self.tp_size,
            self.output_shape,
            self.projected_head_rows,
            self.omitted_head_rows,
        )
        if actual != expected:
            raise ValueError("sparse-Q commit no longer matches the execution plan")
        if self.generation_id != generation_id:
            raise ValueError("sparse-Q commit belongs to another forward generation")
        if self.collective_token != collective_token:
            raise ValueError("sparse-Q commit belongs to another collective vote")
        if self.projection_token != projection_token:
            raise ValueError("sparse-Q commit belongs to another projected tensor")


def issue_sparse_q_commit_certificate(
    plan: RankLocalSparseQPlan,
    *,
    generation_id: str,
    collective_token: str,
    projection_token: str,
    ready_rank_count: int,
) -> SparseQCommitCertificate:
    """Record a successful external TP vote; this function performs no vote."""

    return SparseQCommitCertificate(
        generation_id=generation_id,
        collective_token=collective_token,
        projection_token=projection_token,
        plan_digest=plan.digest,
        layer_id=plan.layer_id,
        tp_rank=plan.tp_rank,
        tp_size=plan.tp_size,
        ready_rank_count=ready_rank_count,
        output_shape=plan.output_shape,
        projected_head_rows=plan.projected_head_rows,
        omitted_head_rows=plan.omitted_head_rows,
    )


@dataclass(frozen=True)
class UE8M0ScaleSlice:
    """Canonical block-scale rows needed by one contiguous Q-head run."""

    full_output_rows: int
    input_size: int
    output_row_start: int
    output_row_end: int
    block_n: int = DEFAULT_UE8M0_BLOCK_SHAPE[0]
    block_k: int = DEFAULT_UE8M0_BLOCK_SHAPE[1]

    def __post_init__(self) -> None:
        values = {
            "full_output_rows": self.full_output_rows,
            "input_size": self.input_size,
            "output_row_start": self.output_row_start,
            "output_row_end": self.output_row_end,
            "block_n": self.block_n,
            "block_k": self.block_k,
        }
        for name, value in values.items():
            _strict_int(value, name)
        if self.full_output_rows <= 0 or self.input_size <= 0:
            raise ValueError("full output and input sizes must be positive")
        if self.block_n <= 0 or self.block_k <= 0:
            raise ValueError("UE8M0 block dimensions must be positive")
        if self.full_output_rows % self.block_n:
            raise ValueError("full output rows must align to the UE8M0 N block")
        if self.input_size % self.block_k:
            raise ValueError("input size must align to the UE8M0 K block")
        if (
            self.output_row_start < 0
            or self.output_row_end <= self.output_row_start
            or self.output_row_end > self.full_output_rows
        ):
            raise ValueError("scale slice output rows are outside the full weight")
        if (
            self.output_row_start % self.block_n
            or self.output_row_end % self.block_n
        ):
            raise ValueError("scale slice must align to UE8M0 N blocks")

    @property
    def block_row_start(self) -> int:
        return self.output_row_start // self.block_n

    @property
    def block_row_end(self) -> int:
        return self.output_row_end // self.block_n

    @property
    def block_row_count(self) -> int:
        return self.block_row_end - self.block_row_start

    @property
    def run_output_rows(self) -> int:
        return self.output_row_end - self.output_row_start

    @property
    def canonical_shape(self) -> Tuple[int, int]:
        return (
            self.full_output_rows // self.block_n,
            self.input_size // self.block_k,
        )

    @property
    def run_canonical_shape(self) -> Tuple[int, int]:
        return self.block_row_count, self.input_size // self.block_k

    @property
    def packed_shape(self) -> Tuple[int, int]:
        """DeepGEMM int32 TMA scale shape for the full weight.

        The canonical K-block scale row is padded to a multiple of four
        UE8M0 bytes and packed into int32 columns.  Its N-block scale is then
        replicated for every output row before the MN-major transpose.
        """

        canonical_k = self.input_size // self.block_k
        return self.full_output_rows, (canonical_k + 3) // 4

    @property
    def packed_stride(self) -> Tuple[int, int]:
        """DeepGEMM MN-major TMA stride for :attr:`packed_shape`."""

        # ``full_output_rows`` is block_n-aligned (128 here), which is also
        # sufficient for DeepGEMM's TMA byte alignment.  The packed tensor is
        # the transpose view of [packed_K, aligned_N].
        return 1, self.full_output_rows

    @property
    def run_packed_shape(self) -> Tuple[int, int]:
        canonical_k = self.input_size // self.block_k
        return self.run_output_rows, (canonical_k + 3) // 4

    @property
    def run_packed_stride(self) -> Tuple[int, int]:
        return 1, self.run_output_rows


def ue8m0_scale_slice_for_head_run(
    run: HeadRun,
    *,
    full_output_rows: int,
    input_size: int,
    block_shape: Tuple[int, int] = DEFAULT_UE8M0_BLOCK_SHAPE,
) -> UE8M0ScaleSlice:
    """Map a head run to canonical FP8 weight-scale block rows."""

    if (
        not isinstance(block_shape, tuple)
        or len(block_shape) != 2
        or any(type(value) is not int for value in block_shape)
    ):
        raise TypeError("block_shape must be a pair of integers")
    return UE8M0ScaleSlice(
        full_output_rows=full_output_rows,
        input_size=input_size,
        output_row_start=run.output_row_start,
        output_row_end=run.output_row_end,
        block_n=block_shape[0],
        block_k=block_shape[1],
    )


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without torch installs
        raise RuntimeError("this UE8M0 helper requires optional PyTorch") from error
    return torch


def slice_canonical_ue8m0_scale(
    canonical_scale: Any,
    geometry: UE8M0ScaleSlice,
):
    """Return a contiguous canonical scale slice without reading tensor values."""

    torch = _require_torch()
    if not isinstance(canonical_scale, torch.Tensor):
        raise TypeError("canonical UE8M0 scale must be a torch.Tensor")
    if canonical_scale.ndim != 2:
        raise ValueError("canonical UE8M0 scale must be two-dimensional")
    if tuple(canonical_scale.shape) != geometry.canonical_shape:
        raise ValueError(
            "canonical UE8M0 scale shape does not match the full weight geometry"
        )
    if canonical_scale.dtype != torch.float32:
        raise ValueError("canonical UE8M0 scale must use float32 block scales")
    result = canonical_scale.narrow(
        0, geometry.block_row_start, geometry.block_row_count
    ).contiguous()
    if tuple(result.shape) != geometry.run_canonical_shape:
        raise RuntimeError("canonical UE8M0 scale slice has an unexpected shape")
    return result


def slice_block_scale(
    weight_scale: Any,
    start_output: int,
    end_output: int,
    *,
    full_output: int = 4096,
    block_n: int = DEFAULT_UE8M0_BLOCK_SHAPE[0],
):
    """Slice canonical FP32 block scales for an output-row interval.

    This concise production helper intentionally accepts only the canonical
    ``[full_output / block_n, K / block_k]`` FP32 scale.  Passing DeepGEMM's
    N-dependent packed int32 TMA layout fails closed.  The caller must use the
    repository's ``inverse_transform_scale_ue8m0`` first and must repack the
    returned scale with ``transform_scale_ue8m0(..., mn=end_output-start_output)``.
    """

    torch = _require_torch()
    if not isinstance(weight_scale, torch.Tensor):
        raise TypeError("canonical block scale must be a torch.Tensor")
    if weight_scale.ndim != 2:
        raise ValueError("canonical block scale must be two-dimensional")
    if weight_scale.dtype != torch.float32:
        raise ValueError(
            "canonical block scale must be float32; packed UE8M0 scales "
            "cannot be sliced directly"
        )
    full_output = _strict_int(full_output, "full_output")
    block_n = _strict_int(block_n, "block_n")
    start_output = _strict_int(start_output, "start_output")
    end_output = _strict_int(end_output, "end_output")
    if full_output <= 0 or block_n <= 0 or full_output % block_n:
        raise ValueError("full_output must be positive and block-aligned")
    if (
        start_output < 0
        or end_output <= start_output
        or end_output > full_output
        or start_output % block_n
        or end_output % block_n
    ):
        raise ValueError("output scale slice must be an in-bounds block interval")
    if int(weight_scale.shape[0]) != full_output // block_n:
        raise ValueError("canonical block scale does not match full_output")
    return weight_scale.narrow(
        0,
        start_output // block_n,
        (end_output - start_output) // block_n,
    ).contiguous()


def repack_ue8m0_scale_for_head_run(
    full_packed_scale: Any,
    geometry: UE8M0ScaleSlice,
    *,
    inverse_transform: Callable[..., Any],
    transform: Callable[..., Any],
):
    """Inverse-layout, slice canonically, then repack for the run's new N.

    ``full_packed_scale.narrow(...)`` is intentionally never used.  Its TMA
    stride and padding were created for ``geometry.full_output_rows`` and are
    invalid for ``geometry.run_output_rows`` even when the underlying head
    interval is contiguous.
    """

    if not callable(inverse_transform) or not callable(transform):
        raise TypeError("UE8M0 inverse and forward transforms must be callable")
    canonical = inverse_transform(
        full_packed_scale, mn=geometry.full_output_rows
    )
    run_canonical = slice_canonical_ue8m0_scale(canonical, geometry)
    return transform(run_canonical, mn=geometry.run_output_rows)


def materialize_sparse_q_run_scale(
    weight_scale: Any,
    geometry: UE8M0ScaleSlice,
    *,
    inverse_transform: Callable[..., Any],
    transform: Callable[..., Any],
):
    """Return the backend-native 128x128 scale for one sparse ``wq_b`` run.

    DeepSeek-V4 FP8 checkpoints natively expose canonical FP32 block scales
    with shape ``[N / 128, K / 128]``.  Some DeepGEMM startup profiles requant
    those weights and replace the scale parameter with an N-dependent packed
    UE8M0 int32 TMA layout.  Sparse output-column projection must support the
    representation selected by the *dense* linear method on this worker:

    * canonical FP32 scales are sliced by output block row and passed through;
    * packed UE8M0 scales are inverse-laid-out, sliced, and repacked for the
      run's smaller N.

    No value is copied to CPU and no scale is sliced in packed layout space.
    """

    torch = _require_torch()
    if not isinstance(weight_scale, torch.Tensor):
        raise TypeError("sparse-Q weight scale must be a torch.Tensor")
    if weight_scale.ndim != 2:
        raise ValueError("sparse-Q weight scale must be two-dimensional")
    packed_ue8m0 = bool(getattr(weight_scale, "format_ue8m0", False))
    if weight_scale.dtype == torch.float32:
        # Some 0731 checkpoints label the *values* UE8M0 while retaining the
        # canonical FP32 block matrix.  Dtype plus exact geometry identify the
        # layout; ``format_ue8m0`` alone must not misclassify it as int32 TMA.
        return slice_canonical_ue8m0_scale(weight_scale.data, geometry)
    if weight_scale.dtype == torch.int32:
        if not packed_ue8m0:
            raise ValueError("int32 wq_b scale is missing its UE8M0 format marker")
        if (
            tuple(int(value) for value in weight_scale.shape)
            != geometry.packed_shape
        ):
            raise ValueError("packed UE8M0 wq_b scale has an invalid shape")
        if (
            tuple(int(value) for value in weight_scale.stride())
            != geometry.packed_stride
        ):
            raise ValueError("packed UE8M0 wq_b scale has an invalid TMA stride")
        run_scale = repack_ue8m0_scale_for_head_run(
            weight_scale.data,
            geometry,
            inverse_transform=inverse_transform,
            transform=transform,
        )
        if not isinstance(run_scale, torch.Tensor) or run_scale.dtype != torch.int32:
            raise ValueError("repacked UE8M0 run scale must be an int32 tensor")
        if (
            tuple(int(value) for value in run_scale.shape)
            != geometry.run_packed_shape
        ):
            raise ValueError("repacked UE8M0 run scale has an invalid shape")
        if (
            tuple(int(value) for value in run_scale.stride())
            != geometry.run_packed_stride
        ):
            raise ValueError("repacked UE8M0 run scale has an invalid TMA stride")
        return run_scale
    raise ValueError(
        "sparse-Q supports canonical FP32 or packed UE8M0 int32 wq_b scales"
    )


__all__ = [
    "DEFAULT_HEAD_DIM",
    "DEFAULT_LOGICAL_HEADS",
    "DEFAULT_TP_SIZE",
    "DEFAULT_UE8M0_BLOCK_SHAPE",
    "HeadRun",
    "RankLocalSparseQPlan",
    "SPARSE_Q_COMMIT_FORMAT_VERSION",
    "SPARSE_Q_PLAN_FORMAT_VERSION",
    "SparseQCommitCertificate",
    "UE8M0ScaleSlice",
    "build_rank_local_sparse_q_plan",
    "build_sparse_q_plan",
    "contiguous_head_runs",
    "issue_sparse_q_commit_certificate",
    "materialize_sparse_q_run_scale",
    "rank_owned_logical_heads",
    "repack_ue8m0_scale_for_head_run",
    "slice_canonical_ue8m0_scale",
    "slice_block_scale",
    "ue8m0_scale_slice_for_head_run",
]
