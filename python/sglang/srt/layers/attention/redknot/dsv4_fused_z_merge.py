"""Persistent-device ``z_off + z_online`` merge for DSV4 MLA reuse.

The old serving path rebuilt a dense ``[T, G, R]`` offline tensor for every
layer and every chunk.  This module keeps the committed per-segment projection
on device and binds the current forward to immutable row spans.  The common
4--16 segment path (including 8x8K) is consumed by one fixed-pointer Triton
kernel: dirty rows copy ``z_online`` and clean rows write
``z_online + z_off``.  No temporary offline tensor, gather, or scatter is
materialized.

An opt-in production fast path pre-packs contiguous global/local ``wo_a``
column runs and uses cuBLAS BF16 GEMMs with FP32 outputs.  It is disabled by
default and selected only by preflight; unsupported layouts retain the Triton
certificate, while committed execution remains fail-closed.

The control-plane geometry is intentionally torch-independent so it can be
validated in CPU-only environments.  CUDA/Triton is imported lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Optional, Sequence, Tuple


MAX_PERSISTENT_PROJECTION_VIEWS = 16
PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN = (
    "dsv4_headsplit_woa_persistent_merge:v1"
)
PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN = (
    "dsv4_headsplit_cublas_woa_fp32_persistent_merge:v1"
)
_CUBLAS_WOA_FASTPATH_ENV = "REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH"
_CUBLAS_WOA_PACK_CACHE_ATTRIBUTE = (
    "_redknot_mla_off_cublas_woa_fp32_pack_v1"
)


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _tensor_identity(tensor: object) -> Tuple[object, ...]:
    """Return a mutation-sensitive identity without synchronizing CUDA."""

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


def _is_contiguous_axis_run(axes: Tuple[int, ...]) -> bool:
    return bool(axes) and axes == tuple(range(axes[0], axes[0] + len(axes)))


def _torch_mm_supports_fp32_out_dtype(torch_module: object) -> bool:
    """Conservatively prove that ``torch.mm`` accepts ``out_dtype``.

    Calling ``mm`` as a capability probe would enqueue real GPU work during
    preflight.  Newer PyTorch builds expose the keyword through either the
    Python signature or the generated operator documentation; an ambiguous
    build stays on the already-certified Triton path.
    """

    mm = getattr(torch_module, "mm", None)
    if not callable(mm):
        return False
    try:
        import inspect

        if "out_dtype" in inspect.signature(mm).parameters:
            return True
    except (TypeError, ValueError):
        pass
    return any(
        "out_dtype" in str(text)
        for text in (
            getattr(mm, "__text_signature__", ""),
            getattr(mm, "__doc__", ""),
        )
    )


@dataclass(frozen=True, eq=False)
class _PackedWOAColumnCache:
    """One source-owned global/local pack; it never retains the source.

    The cache is attached to the exact ``wo_a`` tensor.  Replacing it for a
    different axis split bounds retention to one entry per model weight, while
    the object id plus mutation-sensitive identity prevents data-pointer reuse
    from authenticating another tensor.
    """

    source_object_id: int
    source_identity: Tuple[object, ...]
    local_head_axes: Tuple[int, ...]
    global_head_axes: Tuple[int, ...]
    head_dim: int
    o_lora_rank: int
    packed_global_weight: object
    packed_global_weight_identity: Tuple[object, ...]
    packed_local_weight: object
    packed_local_weight_identity: Tuple[object, ...]

    def matches(
        self,
        *,
        torch_module: object,
        source_weight: object,
        source_identity: Tuple[object, ...],
        local_head_axes: Tuple[int, ...],
        global_head_axes: Tuple[int, ...],
        head_dim: int,
        o_lora_rank: int,
    ) -> bool:
        tensor_type = getattr(torch_module, "Tensor", ())
        expected_global_shape = (
            o_lora_rank,
            len(global_head_axes) * head_dim,
        )
        expected_local_shape = (
            o_lora_rank,
            len(local_head_axes) * head_dim,
        )
        return (
            self.source_object_id == id(source_weight)
            and self.source_identity == source_identity
            and _tensor_identity(source_weight) == source_identity
            and self.local_head_axes == local_head_axes
            and self.global_head_axes == global_head_axes
            and self.head_dim == head_dim
            and self.o_lora_rank == o_lora_rank
            and isinstance(self.packed_global_weight, tensor_type)
            and isinstance(self.packed_local_weight, tensor_type)
            and tuple(int(value) for value in self.packed_global_weight.shape)
            == expected_global_shape
            and tuple(int(value) for value in self.packed_local_weight.shape)
            == expected_local_shape
            and self.packed_global_weight.device == source_weight.device
            and self.packed_local_weight.device == source_weight.device
            and self.packed_global_weight.dtype == source_weight.dtype
            and self.packed_local_weight.dtype == source_weight.dtype
            and self.packed_global_weight.is_contiguous()
            and self.packed_local_weight.is_contiguous()
            and _tensor_identity(self.packed_global_weight)
            == self.packed_global_weight_identity
            and _tensor_identity(self.packed_local_weight)
            == self.packed_local_weight_identity
        )


def _get_or_pack_woa_column_weights(
    *,
    torch_module: object,
    source_weight: object,
    local_head_axes: Tuple[int, ...],
    global_head_axes: Tuple[int, ...],
    head_dim: int,
    o_lora_rank: int,
) -> _PackedWOAColumnCache:
    """Return a validated cold-path pack, replacing any stale source entry."""

    source_identity = _tensor_identity(source_weight)
    cached = getattr(source_weight, _CUBLAS_WOA_PACK_CACHE_ATTRIBUTE, None)
    if isinstance(cached, _PackedWOAColumnCache) and cached.matches(
        torch_module=torch_module,
        source_weight=source_weight,
        source_identity=source_identity,
        local_head_axes=local_head_axes,
        global_head_axes=global_head_axes,
        head_dim=head_dim,
        o_lora_rank=o_lora_rank,
    ):
        return cached

    global_column_start = global_head_axes[0] * head_dim
    global_column_count = len(global_head_axes) * head_dim
    local_column_start = local_head_axes[0] * head_dim
    local_column_count = len(local_head_axes) * head_dim
    packed_global_weight = source_weight.narrow(
        1, global_column_start, global_column_count
    ).contiguous()
    packed_local_weight = source_weight.narrow(
        1, local_column_start, local_column_count
    ).contiguous()
    if _tensor_identity(source_weight) != source_identity:
        raise ValueError("wo_a source changed while packing cuBLAS columns")
    packed = _PackedWOAColumnCache(
        source_object_id=id(source_weight),
        source_identity=source_identity,
        local_head_axes=local_head_axes,
        global_head_axes=global_head_axes,
        head_dim=head_dim,
        o_lora_rank=o_lora_rank,
        packed_global_weight=packed_global_weight,
        packed_global_weight_identity=_tensor_identity(packed_global_weight),
        packed_local_weight=packed_local_weight,
        packed_local_weight_identity=_tensor_identity(packed_local_weight),
    )
    if not packed.matches(
        torch_module=torch_module,
        source_weight=source_weight,
        source_identity=source_identity,
        local_head_axes=local_head_axes,
        global_head_axes=global_head_axes,
        head_dim=head_dim,
        o_lora_rank=o_lora_rank,
    ):
        raise ValueError("cuBLAS wo_a packed columns have an incompatible layout")
    setattr(source_weight, _CUBLAS_WOA_PACK_CACHE_ATTRIBUTE, packed)
    if getattr(source_weight, _CUBLAS_WOA_PACK_CACHE_ATTRIBUTE, None) is not packed:
        raise ValueError("cuBLAS wo_a pack cache could not bind the source tensor")
    return packed


@dataclass(frozen=True)
class ProjectionSpanGeometry:
    """One immutable mapping from a committed segment into forward rows."""

    output_rows: Tuple[int, ...]
    local_rows: Tuple[int, ...]
    _is_unit_stride_cache: bool = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.output_rows) is not tuple or type(self.local_rows) is not tuple:
            raise TypeError("projection span rows must be immutable tuples")
        if len(self.output_rows) != len(self.local_rows):
            raise ValueError("projection output/local rows must have equal length")
        if not self.output_rows:
            raise ValueError("projection span cannot be empty")
        output = tuple(_strict_int(value, "output row") for value in self.output_rows)
        local = tuple(_strict_int(value, "local row") for value in self.local_rows)
        if any(value < 0 for value in output + local):
            raise ValueError("projection rows must be non-negative")
        if len(set(output)) != len(output) or len(set(local)) != len(local):
            raise ValueError("projection span rows must be unique")
        if any(right <= left for left, right in zip(output, output[1:])):
            raise ValueError("projection output rows must be strictly increasing")
        if any(right <= left for left, right in zip(local, local[1:])):
            raise ValueError("projection local rows must be strictly increasing")
        object.__setattr__(
            self,
            "_is_unit_stride_cache",
            all(
                right == left + 1
                for rows in (output, local)
                for left, right in zip(rows, rows[1:])
            ),
        )

    @property
    def length(self) -> int:
        return len(self.output_rows)

    @property
    def is_unit_stride(self) -> bool:
        return self._is_unit_stride_cache

    @property
    def output_start(self) -> int:
        return self.output_rows[0]

    @property
    def local_start(self) -> int:
        return self.local_rows[0]

    def validate(self, *, total_rows: int, segment_rows: int) -> None:
        total_rows = _strict_int(total_rows, "total rows")
        segment_rows = _strict_int(segment_rows, "segment rows")
        if total_rows <= 0 or segment_rows <= 0:
            raise ValueError("projection row bounds must be positive")
        if self.output_rows[-1] >= total_rows:
            raise ValueError("projection output row exceeds the forward")
        if self.local_rows[-1] >= segment_rows:
            raise ValueError("projection local row exceeds the segment")


@dataclass(frozen=True, eq=False)
class PersistentProjectionView:
    """Pinned device tensor plus the rows consumed by one forward."""

    seg_hash: str
    layer_id: int
    commit_epoch: int
    geometry: ProjectionSpanGeometry
    values: object
    values_identity: Tuple[object, ...]
    generation_token: str

    @classmethod
    def bind(
        cls,
        *,
        seg_hash: str,
        layer_id: int,
        commit_epoch: int,
        geometry: ProjectionSpanGeometry,
        values: object,
        generation_token: str,
    ) -> "PersistentProjectionView":
        if not isinstance(seg_hash, str) or not seg_hash:
            raise ValueError("projection segment hash must be non-empty")
        if not isinstance(generation_token, str) or not generation_token:
            raise ValueError("projection generation token must be non-empty")
        layer_id = _strict_int(layer_id, "layer id")
        commit_epoch = _strict_int(commit_epoch, "commit epoch")
        if layer_id < 0 or commit_epoch <= 0:
            raise ValueError("projection layer/epoch is invalid")
        if not hasattr(values, "shape") or len(values.shape) != 3:
            raise ValueError("persistent projection must be a [rows,groups,rank] tensor")
        if getattr(getattr(values, "device", None), "type", None) != "cuda":
            raise ValueError("persistent projection must remain device resident")
        geometry.validate(total_rows=max(geometry.output_rows) + 1, segment_rows=int(values.shape[0]))
        return cls(
            seg_hash=seg_hash,
            layer_id=layer_id,
            commit_epoch=commit_epoch,
            geometry=geometry,
            values=values,
            values_identity=_tensor_identity(values),
            generation_token=generation_token,
        )

    def validate(self, *, total_rows: int, tail_shape: Tuple[int, int]) -> None:
        if _tensor_identity(self.values) != self.values_identity:
            raise ValueError("persistent projection tensor changed after pin")
        if tuple(int(value) for value in self.values.shape[1:]) != tuple(tail_shape):
            raise ValueError("persistent projection tail shape changed")
        self.geometry.validate(
            total_rows=total_rows,
            segment_rows=int(self.values.shape[0]),
        )

    def validate_live(
        self,
        *,
        tail_shape: Tuple[int, int],
        expected_device: object,
        expected_dtype: object,
    ) -> None:
        """Revalidate only mutable tensor state after full geometry commit."""

        if _tensor_identity(self.values) != self.values_identity:
            raise ValueError("persistent projection tensor changed after pin")
        shape = tuple(int(value) for value in self.values.shape)
        if len(shape) != 3 or shape[1:] != tail_shape:
            raise ValueError("persistent projection live shape changed")
        if (
            self.values.device != expected_device
            or self.values.dtype != expected_dtype
        ):
            raise ValueError("persistent projection live device/dtype changed")


def _persistent_projection_live_identity(
    plan: "PersistentProjectionPlan",
) -> Tuple[object, ...]:
    return (
        int(plan.total_rows),
        tuple(plan.tail_shape),
        id(plan.views),
        tuple(
            (
                id(view),
                str(view.seg_hash),
                int(view.layer_id),
                int(view.commit_epoch),
                id(view.geometry),
                id(view.geometry.output_rows),
                id(view.geometry.local_rows),
                bool(view.geometry.is_unit_stride),
                _tensor_identity(view.values),
                view.values_identity,
                str(view.generation_token),
            )
            for view in plan.views
        ),
        str(plan.digest),
    )


@dataclass(frozen=True)
class PersistentProjectionPlan:
    total_rows: int
    tail_shape: Tuple[int, int]
    views: Tuple[PersistentProjectionView, ...]
    digest: str
    _live_identity: Tuple[object, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._validate_full()
        object.__setattr__(
            self, "_live_identity", _persistent_projection_live_identity(self)
        )

    @property
    def is_single_launch_eligible(self) -> bool:
        """Whether every view can be consumed by the fixed-pointer kernel."""

        return (
            0 < len(self.views) <= MAX_PERSISTENT_PROJECTION_VIEWS
            and all(view.geometry.is_unit_stride for view in self.views)
        )

    def _validate_full(self) -> None:
        if type(self.total_rows) is not int or self.total_rows <= 0:
            raise ValueError("persistent projection total_rows is invalid")
        if (
            type(self.tail_shape) is not tuple
            or len(self.tail_shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.tail_shape)
        ):
            raise ValueError("persistent projection tail_shape is invalid")
        if type(self.views) is not tuple or not self.views:
            raise ValueError("persistent projection plan requires pinned views")
        if len(self.views) > MAX_PERSISTENT_PROJECTION_VIEWS:
            raise ValueError(
                "persistent projection plan exceeds the 16-view GPU preflight"
            )
        seen = set()
        for view in self.views:
            if not isinstance(view, PersistentProjectionView):
                raise TypeError("persistent projection plan contains an invalid view")
            view.validate(total_rows=self.total_rows, tail_shape=self.tail_shape)
            overlap = seen.intersection(view.geometry.output_rows)
            if overlap:
                raise ValueError("persistent projection views overlap")
            seen.update(view.geometry.output_rows)
        if self.digest != _plan_digest(
            total_rows=self.total_rows,
            tail_shape=self.tail_shape,
            views=self.views,
        ):
            raise ValueError("persistent projection plan digest changed")

    def validate(self) -> None:
        """O(views) live check after the constructor's complete row proof."""

        if _persistent_projection_live_identity(self) != self._live_identity:
            raise ValueError("persistent projection immutable binding changed")

    def validate_live(
        self,
        *,
        expected_object_id: int,
        expected_digest: str,
        expected_device: object,
        expected_dtype: object,
    ) -> None:
        """O(views) live check for an already fully validated frozen plan."""

        if id(self) != expected_object_id or str(self.digest) != expected_digest:
            raise ValueError("persistent projection plan changed after commit")
        if (
            type(self.total_rows) is not int
            or self.total_rows <= 0
            or type(self.tail_shape) is not tuple
            or len(self.tail_shape) != 2
            or type(self.views) is not tuple
            or not self.views
            or len(self.views) > MAX_PERSISTENT_PROJECTION_VIEWS
        ):
            raise ValueError("persistent projection live metadata changed")
        for view in self.views:
            if not isinstance(view, PersistentProjectionView):
                raise TypeError("persistent projection live view is invalid")
            view.validate_live(
                tail_shape=self.tail_shape,
                expected_device=expected_device,
                expected_dtype=expected_dtype,
            )


def _plan_digest(
    *,
    total_rows: int,
    tail_shape: Tuple[int, int],
    views: Sequence[PersistentProjectionView],
) -> str:
    fields = [str(total_rows), repr(tuple(tail_shape))]
    for view in views:
        fields.append(
            repr(
                (
                    view.seg_hash,
                    view.layer_id,
                    view.commit_epoch,
                    view.geometry.output_rows,
                    view.geometry.local_rows,
                    view.generation_token,
                )
            )
        )
    return sha256("|".join(fields).encode("utf-8")).hexdigest()


def build_persistent_projection_plan(
    *,
    total_rows: int,
    tail_shape: Tuple[int, int],
    views: Sequence[PersistentProjectionView],
) -> PersistentProjectionPlan:
    total_rows = _strict_int(total_rows, "total rows")
    normalized_views = tuple(views)
    normalized_tail = tuple(_strict_int(value, "tail dimension") for value in tail_shape)
    plan = PersistentProjectionPlan(
        total_rows=total_rows,
        tail_shape=normalized_tail,
        views=normalized_views,
        digest=_plan_digest(
            total_rows=total_rows,
            tail_shape=normalized_tail,
            views=normalized_views,
        ),
    )
    plan.validate()
    return plan


def _projection_plan_identity(
    plan: PersistentProjectionPlan,
) -> Tuple[object, ...]:
    return (
        id(plan),
        str(plan.digest),
        tuple(
            (
                id(view),
                view.values_identity,
                view.geometry.output_rows,
                view.geometry.local_rows,
            )
            for view in plan.views
        ),
    )


def _cublas_woa_fp32_certificate_digest(
    *,
    source_weight_object_id: int,
    source_weight_identity: Tuple[object, ...],
    projection_plan_digest: str,
    dirty_row_values: Tuple[int, ...],
    local_head_axes: Tuple[int, ...],
    global_head_axes: Tuple[int, ...],
    head_dim: int,
    o_lora_rank: int,
    packed_global_weight_identity: Tuple[object, ...],
    packed_local_weight_identity: Tuple[object, ...],
) -> str:
    return sha256(
        repr(
            (
                source_weight_object_id,
                source_weight_identity,
                projection_plan_digest,
                dirty_row_values,
                local_head_axes,
                global_head_axes,
                head_dim,
                o_lora_rank,
                packed_global_weight_identity,
                packed_local_weight_identity,
                PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN,
            )
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, eq=False)
class CublasWOAFP32FastPathCertificate:
    """Immutable proof for the opt-in FP32-accumulating cuBLAS formula."""

    source_weight_object_id: int
    source_weight_identity: Tuple[object, ...]
    projection_plan_digest: str
    dirty_row_values: Tuple[int, ...]
    local_head_axes: Tuple[int, ...]
    global_head_axes: Tuple[int, ...]
    global_axis_start: int
    global_axis_count: int
    local_axis_start: int
    local_axis_count: int
    head_dim: int
    o_lora_rank: int
    packed_global_weight: object
    packed_global_weight_identity: Tuple[object, ...]
    packed_local_weight: object
    packed_local_weight_identity: Tuple[object, ...]
    kernel_token: str
    digest: str

    def validate(
        self,
        *,
        source_weight: object,
        projection_plan: PersistentProjectionPlan,
        dirty_row_values: Tuple[int, ...],
        local_head_axes: Tuple[int, ...],
        global_head_axes: Tuple[int, ...],
        owned_heads: int,
        head_dim: int,
        o_lora_rank: int,
    ) -> None:
        if (
            self.source_weight_object_id != id(source_weight)
            or self.source_weight_identity != _tensor_identity(source_weight)
        ):
            raise ValueError("cuBLAS wo_a source changed after preflight")
        if (
            self.projection_plan_digest != projection_plan.digest
            or self.dirty_row_values != dirty_row_values
            or self.local_head_axes != local_head_axes
            or self.global_head_axes != global_head_axes
            or self.head_dim != head_dim
            or self.o_lora_rank != o_lora_rank
        ):
            raise ValueError("cuBLAS wo_a certificate geometry changed")
        if (
            len(projection_plan.views) != 1
            or not projection_plan.views[0].geometry.is_unit_stride
            or dirty_row_values != tuple(range(len(dirty_row_values)))
            or not _is_contiguous_axis_run(global_head_axes)
            or not _is_contiguous_axis_run(local_head_axes)
            or tuple(sorted(global_head_axes + local_head_axes))
            != tuple(range(owned_heads))
        ):
            raise ValueError("cuBLAS wo_a certificate is not a contiguous split")
        view = projection_plan.views[0]
        dirty_count = len(dirty_row_values)
        if (
            view.geometry.output_start != dirty_count
            or view.geometry.length != projection_plan.total_rows - dirty_count
            or self.global_axis_start != global_head_axes[0]
            or self.global_axis_count != len(global_head_axes)
            or self.local_axis_start != local_head_axes[0]
            or self.local_axis_count != len(local_head_axes)
        ):
            raise ValueError("cuBLAS wo_a prefix/run certificate changed")
        expected_global_shape = (
            o_lora_rank,
            len(global_head_axes) * head_dim,
        )
        expected_local_shape = (
            o_lora_rank,
            len(local_head_axes) * head_dim,
        )
        for label, packed_weight, packed_identity, expected_shape in (
            (
                "global",
                self.packed_global_weight,
                self.packed_global_weight_identity,
                expected_global_shape,
            ),
            (
                "local",
                self.packed_local_weight,
                self.packed_local_weight_identity,
                expected_local_shape,
            ),
        ):
            if (
                _tensor_identity(packed_weight) != packed_identity
                or tuple(int(value) for value in packed_weight.shape)
                != expected_shape
                or packed_weight.device != source_weight.device
                or packed_weight.dtype != source_weight.dtype
                or not packed_weight.is_contiguous()
            ):
                raise ValueError(f"cuBLAS packed {label} weight changed")
        if self.kernel_token != PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN:
            raise ValueError("cuBLAS wo_a certificate kernel ABI changed")
        if self.digest != _cublas_woa_fp32_certificate_digest(
            source_weight_object_id=self.source_weight_object_id,
            source_weight_identity=self.source_weight_identity,
            projection_plan_digest=self.projection_plan_digest,
            dirty_row_values=self.dirty_row_values,
            local_head_axes=self.local_head_axes,
            global_head_axes=self.global_head_axes,
            head_dim=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            packed_global_weight_identity=self.packed_global_weight_identity,
            packed_local_weight_identity=self.packed_local_weight_identity,
        ):
            raise ValueError("cuBLAS wo_a certificate digest changed")

    def validate_live(
        self,
        *,
        source_weight: object,
        projection_plan: PersistentProjectionPlan,
    ) -> None:
        """Recheck live source/pack identities without geometry or digest work."""

        if (
            self.source_weight_object_id != id(source_weight)
            or self.source_weight_identity != _tensor_identity(source_weight)
        ):
            raise ValueError("cuBLAS wo_a source changed after preflight")
        if self.projection_plan_digest != str(projection_plan.digest):
            raise ValueError("cuBLAS wo_a projection digest changed")
        expected_global_shape = (
            int(self.o_lora_rank),
            int(self.global_axis_count) * int(self.head_dim),
        )
        expected_local_shape = (
            int(self.o_lora_rank),
            int(self.local_axis_count) * int(self.head_dim),
        )
        for label, packed_weight, packed_identity, expected_shape in (
            (
                "global",
                self.packed_global_weight,
                self.packed_global_weight_identity,
                expected_global_shape,
            ),
            (
                "local",
                self.packed_local_weight,
                self.packed_local_weight_identity,
                expected_local_shape,
            ),
        ):
            if (
                _tensor_identity(packed_weight) != packed_identity
                or tuple(int(value) for value in packed_weight.shape)
                != expected_shape
                or packed_weight.device != source_weight.device
                or packed_weight.dtype != source_weight.dtype
                or not packed_weight.is_contiguous()
            ):
                raise ValueError(f"cuBLAS packed {label} weight changed")
        if self.kernel_token != PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN:
            raise ValueError("cuBLAS wo_a live kernel ABI changed")


def _headsplit_plan_digest(
    *,
    projection_plan: PersistentProjectionPlan,
    dirty_row_values: Tuple[int, ...],
    local_head_axes: Tuple[int, ...],
    global_head_axes: Tuple[int, ...],
    owned_heads: int,
    groups: int,
    head_dim: int,
    o_lora_rank: int,
    wo_a_weight_identity: Tuple[object, ...],
    kernel_token: str,
    cublas_woa_fp32_certificate_digest: Optional[str],
) -> str:
    return sha256(
        repr(
            (
                projection_plan.digest,
                dirty_row_values,
                local_head_axes,
                global_head_axes,
                owned_heads,
                groups,
                head_dim,
                o_lora_rank,
                wo_a_weight_identity,
                kernel_token,
                cublas_woa_fp32_certificate_digest,
            )
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PersistentHeadSplitWOAMergePlan:
    """Immutable proof for a preselected pure-headsplit ``wo_a`` path.

    The safe default is the existing two-launch Triton implementation.  An
    explicitly enabled, separately certified layout may instead use two cuBLAS
    GEMMs with FP32 output and one final BF16 rounding.  The choice is frozen
    before composite omission; execution never catches a launch error and
    switches formulas after commit.
    """

    projection_plan: PersistentProjectionPlan
    projection_plan_identity: Tuple[object, ...]
    dirty_rows: object
    dirty_rows_identity: Tuple[object, ...]
    dirty_rows_cpu: object
    dirty_rows_cpu_identity: Tuple[object, ...]
    dirty_row_values: Tuple[int, ...]
    wo_a_weight: object
    wo_a_weight_identity: Tuple[object, ...]
    local_head_axes: Tuple[int, ...]
    global_head_axes: Tuple[int, ...]
    local_head_mask: int
    global_head_mask: int
    owned_heads: int
    groups: int
    head_dim: int
    o_lora_rank: int
    kernel_token: str
    digest: str
    cublas_woa_fp32_certificate: Optional[
        CublasWOAFP32FastPathCertificate
    ] = None

    @property
    def uses_cublas_woa_fp32_fastpath(self) -> bool:
        return (
            self.kernel_token
            == PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN
        )

    def validate(self) -> None:
        if not isinstance(self.projection_plan, PersistentProjectionPlan):
            raise TypeError("headsplit merge has no persistent projection plan")
        self.projection_plan.validate()
        if not self.projection_plan.is_single_launch_eligible:
            raise ValueError(
                "headsplit merge requires one to sixteen unit-stride z_off views"
            )
        if (
            _projection_plan_identity(self.projection_plan)
            != self.projection_plan_identity
        ):
            raise ValueError("headsplit persistent projection changed after preflight")
        if _tensor_identity(self.dirty_rows) != self.dirty_rows_identity:
            raise ValueError("headsplit dirty-row device tensor changed after preflight")
        if _tensor_identity(self.dirty_rows_cpu) != self.dirty_rows_cpu_identity:
            raise ValueError("headsplit dirty-row CPU tensor changed after preflight")
        if _tensor_identity(self.wo_a_weight) != self.wo_a_weight_identity:
            raise ValueError("headsplit wo_a weight changed after preflight")
        if self.kernel_token not in (
            PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN,
            PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN,
        ):
            raise ValueError("headsplit merge kernel ABI changed")
        if (
            type(self.owned_heads) is not int
            or not 1 <= self.owned_heads <= 16
            or self.groups != 1
            or type(self.head_dim) is not int
            or self.head_dim <= 0
            or type(self.o_lora_rank) is not int
            or self.o_lora_rank <= 0
        ):
            raise ValueError("headsplit projection dimensions are invalid")
        local_axes = tuple(self.local_head_axes)
        global_axes = tuple(self.global_head_axes)
        if (
            not local_axes
            or local_axes != tuple(sorted(set(local_axes)))
            or any(axis < 0 or axis >= self.owned_heads for axis in local_axes)
            or global_axes
            != tuple(axis for axis in range(self.owned_heads) if axis not in local_axes)
        ):
            raise ValueError("headsplit local/global head partition is invalid")
        expected_local_mask = sum(1 << axis for axis in local_axes)
        expected_global_mask = sum(1 << axis for axis in global_axes)
        if (
            self.local_head_mask != expected_local_mask
            or self.global_head_mask != expected_global_mask
            or self.local_head_mask & self.global_head_mask
            or (self.local_head_mask | self.global_head_mask)
            != (1 << self.owned_heads) - 1
        ):
            raise ValueError("headsplit compile-time head masks are invalid")
        certificate = self.cublas_woa_fp32_certificate
        if self.uses_cublas_woa_fp32_fastpath:
            if not isinstance(certificate, CublasWOAFP32FastPathCertificate):
                raise ValueError("cuBLAS wo_a plan has no fast-path certificate")
            certificate.validate(
                source_weight=self.wo_a_weight,
                projection_plan=self.projection_plan,
                dirty_row_values=self.dirty_row_values,
                local_head_axes=local_axes,
                global_head_axes=global_axes,
                owned_heads=self.owned_heads,
                head_dim=self.head_dim,
                o_lora_rank=self.o_lora_rank,
            )
        elif certificate is not None:
            raise ValueError("Triton wo_a plan carries a cuBLAS certificate")
        certificate_digest = certificate.digest if certificate is not None else None
        if self.digest != _headsplit_plan_digest(
            projection_plan=self.projection_plan,
            dirty_row_values=self.dirty_row_values,
            local_head_axes=local_axes,
            global_head_axes=global_axes,
            owned_heads=self.owned_heads,
            groups=self.groups,
            head_dim=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            wo_a_weight_identity=self.wo_a_weight_identity,
            kernel_token=self.kernel_token,
            cublas_woa_fp32_certificate_digest=certificate_digest,
        ):
            raise ValueError("headsplit merge plan digest changed")

    def validate_live(
        self,
        *,
        committed_plan_identity: Tuple[object, ...],
    ) -> None:
        """O(views) mutable-state check gated by a full commit receipt."""

        if (
            type(committed_plan_identity) is not tuple
            or len(committed_plan_identity) != 2
            or committed_plan_identity != (id(self), str(self.digest))
        ):
            raise ValueError("headsplit live validation requires its commit receipt")
        if not isinstance(self.projection_plan, PersistentProjectionPlan):
            raise TypeError("headsplit merge has no persistent projection plan")
        projection_identity = self.projection_plan_identity
        if type(projection_identity) is not tuple or len(projection_identity) < 2:
            raise ValueError("headsplit projection identity receipt changed")
        self.projection_plan.validate_live(
            expected_object_id=projection_identity[0],
            expected_digest=projection_identity[1],
            expected_device=self.wo_a_weight.device,
            expected_dtype=self.wo_a_weight.dtype,
        )
        if _tensor_identity(self.dirty_rows) != self.dirty_rows_identity:
            raise ValueError("headsplit dirty-row device tensor changed after preflight")
        if _tensor_identity(self.dirty_rows_cpu) != self.dirty_rows_cpu_identity:
            raise ValueError("headsplit dirty-row CPU tensor changed after preflight")
        if _tensor_identity(self.wo_a_weight) != self.wo_a_weight_identity:
            raise ValueError("headsplit wo_a weight changed after preflight")

        certificate = self.cublas_woa_fp32_certificate
        if self.kernel_token == PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN:
            if not isinstance(certificate, CublasWOAFP32FastPathCertificate):
                raise ValueError("cuBLAS wo_a plan has no fast-path certificate")
            if certificate.kernel_token != self.kernel_token:
                raise ValueError("cuBLAS wo_a plan/certificate token changed")
            certificate.validate_live(
                source_weight=self.wo_a_weight,
                projection_plan=self.projection_plan,
            )
        elif self.kernel_token == PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN:
            if certificate is not None:
                raise ValueError("Triton wo_a plan carries a cuBLAS certificate")
        else:
            raise ValueError("headsplit live kernel ABI changed")


def _preflight_cublas_woa_fp32_fastpath(
    *,
    torch_module: object,
    projection_plan: PersistentProjectionPlan,
    dirty_row_values: Tuple[int, ...],
    local_head_axes: Tuple[int, ...],
    global_head_axes: Tuple[int, ...],
    wo_a_weight: object,
    owned_heads: int,
    head_dim: int,
    o_lora_rank: int,
) -> Optional[CublasWOAFP32FastPathCertificate]:
    """Cold-path eligibility and packing; ``None`` selects legacy Triton."""

    if os.environ.get(_CUBLAS_WOA_FASTPATH_ENV) != "1":
        return None
    if (
        len(projection_plan.views) != 1
        or not projection_plan.views[0].geometry.is_unit_stride
        or dirty_row_values != tuple(range(len(dirty_row_values)))
        or not _is_contiguous_axis_run(global_head_axes)
        or not _is_contiguous_axis_run(local_head_axes)
        or tuple(sorted(global_head_axes + local_head_axes))
        != tuple(range(owned_heads))
        or not _torch_mm_supports_fp32_out_dtype(torch_module)
    ):
        return None
    view = projection_plan.views[0]
    dirty_count = len(dirty_row_values)
    if (
        view.geometry.output_start != dirty_count
        or view.geometry.length != projection_plan.total_rows - dirty_count
        or getattr(getattr(wo_a_weight, "device", None), "type", None) != "cuda"
        or str(getattr(wo_a_weight, "dtype", None)) != "torch.bfloat16"
        or not wo_a_weight.is_contiguous()
        or getattr(getattr(view.values, "device", None), "type", None) != "cuda"
        or str(getattr(view.values, "dtype", None)) != "torch.bfloat16"
        or not view.values.is_contiguous()
    ):
        return None
    try:
        packed = _get_or_pack_woa_column_weights(
            torch_module=torch_module,
            source_weight=wo_a_weight,
            local_head_axes=local_head_axes,
            global_head_axes=global_head_axes,
            head_dim=head_dim,
            o_lora_rank=o_lora_rank,
        )
        digest = _cublas_woa_fp32_certificate_digest(
            source_weight_object_id=id(wo_a_weight),
            source_weight_identity=packed.source_identity,
            projection_plan_digest=projection_plan.digest,
            dirty_row_values=dirty_row_values,
            local_head_axes=local_head_axes,
            global_head_axes=global_head_axes,
            head_dim=head_dim,
            o_lora_rank=o_lora_rank,
            packed_global_weight_identity=(
                packed.packed_global_weight_identity
            ),
            packed_local_weight_identity=packed.packed_local_weight_identity,
        )
        certificate = CublasWOAFP32FastPathCertificate(
            source_weight_object_id=id(wo_a_weight),
            source_weight_identity=packed.source_identity,
            projection_plan_digest=projection_plan.digest,
            dirty_row_values=dirty_row_values,
            local_head_axes=local_head_axes,
            global_head_axes=global_head_axes,
            global_axis_start=global_head_axes[0],
            global_axis_count=len(global_head_axes),
            local_axis_start=local_head_axes[0],
            local_axis_count=len(local_head_axes),
            head_dim=head_dim,
            o_lora_rank=o_lora_rank,
            packed_global_weight=packed.packed_global_weight,
            packed_global_weight_identity=(
                packed.packed_global_weight_identity
            ),
            packed_local_weight=packed.packed_local_weight,
            packed_local_weight_identity=packed.packed_local_weight_identity,
            kernel_token=PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN,
            digest=digest,
        )
        certificate.validate(
            source_weight=wo_a_weight,
            projection_plan=projection_plan,
            dirty_row_values=dirty_row_values,
            local_head_axes=local_head_axes,
            global_head_axes=global_head_axes,
            owned_heads=owned_heads,
            head_dim=head_dim,
            o_lora_rank=o_lora_rank,
        )
    except Exception:
        # Packing and capability proof happen before commit.  Any ambiguity
        # selects the pre-existing Triton certificate; execution never falls
        # back after omission has been authorized.
        return None
    return certificate


def preflight_persistent_headsplit_woa_merge(
    *,
    projection_plan: PersistentProjectionPlan,
    dirty_rows,
    dirty_rows_cpu,
    local_head_axes: Sequence[int],
    wo_a_weight,
    owned_heads: int,
    groups: int,
    head_dim: int,
    o_lora_rank: int,
) -> PersistentHeadSplitWOAMergePlan:
    """Certify every pointer/shape used after composite omission.

    The CUDA dirty-row tensor is already covered by the caller's sparse-Q and
    restore-layout certificates.  This preflight binds that exact pointer to
    the authoritative CPU row tuple and to the persistent projection views.
    It performs no CUDA read or synchronization.
    """

    import torch

    if not isinstance(projection_plan, PersistentProjectionPlan):
        raise TypeError("headsplit preflight requires a persistent projection plan")
    projection_plan.validate()
    if not projection_plan.is_single_launch_eligible:
        raise ValueError(
            "headsplit preflight requires one to sixteen unit-stride z_off views"
        )
    owned_heads = _strict_int(owned_heads, "owned heads")
    groups = _strict_int(groups, "output groups")
    head_dim = _strict_int(head_dim, "head dimension")
    o_lora_rank = _strict_int(o_lora_rank, "o_lora rank")
    if not 1 <= owned_heads <= 16:
        raise ValueError("headsplit kernel supports one to sixteen owned heads")
    if groups != 1:
        raise ValueError(
            "headsplit kernel v1 requires one rank-local wo_a output group"
        )
    if head_dim <= 0 or o_lora_rank <= 0:
        raise ValueError("headsplit projection dimensions must be positive")
    if tuple(projection_plan.tail_shape) != (groups, o_lora_rank):
        raise ValueError("persistent z_off tail differs from wo_a output")

    local_axes = tuple(_strict_int(axis, "local head axis") for axis in local_head_axes)
    if (
        not local_axes
        or local_axes != tuple(sorted(set(local_axes)))
        or any(axis < 0 or axis >= owned_heads for axis in local_axes)
    ):
        raise ValueError("headsplit local head axes must be sorted and unique")
    global_axes = tuple(
        axis for axis in range(owned_heads) if axis not in set(local_axes)
    )

    total_rows = int(projection_plan.total_rows)
    if (
        not isinstance(dirty_rows_cpu, torch.Tensor)
        or dirty_rows_cpu.ndim != 1
        or dirty_rows_cpu.dtype != torch.long
        or dirty_rows_cpu.device.type != "cpu"
        or not dirty_rows_cpu.is_contiguous()
    ):
        raise ValueError("headsplit CPU dirty rows must be contiguous int64")
    dirty_values = tuple(int(value) for value in dirty_rows_cpu.tolist())
    if (
        dirty_values != tuple(sorted(set(dirty_values)))
        or any(value < 0 or value >= total_rows for value in dirty_values)
    ):
        raise ValueError("headsplit dirty rows must be sorted, unique, and in range")
    prefix_dirty = dirty_values == tuple(range(len(dirty_values)))
    if prefix_dirty and all(
        view.geometry.is_unit_stride for view in projection_plan.views
    ):
        # Production context-bound RAG uses a dirty prefix followed by one or
        # more immutable clean spans.  Validate that exact partition using
        # span endpoints instead of rebuilding an 8K clean set per layer.
        cursor = len(dirty_values)
        for view in projection_plan.views:
            if view.geometry.output_start != cursor:
                raise ValueError(
                    "headsplit persistent clean spans must tile after dirty prefix"
                )
            cursor += view.geometry.length
        if cursor != total_rows:
            raise ValueError(
                "headsplit dirty prefix and clean spans do not cover the forward"
            )
    else:
        clean_values = tuple(
            int(row)
            for view in projection_plan.views
            for row in view.geometry.output_rows
        )
        if clean_values != tuple(sorted(clean_values)):
            raise ValueError(
                "headsplit persistent clean rows must be globally ordered"
            )
        clean_set = set(clean_values)
        expected_dirty = tuple(
            row for row in range(total_rows) if row not in clean_set
        )
        if dirty_values != expected_dirty:
            raise ValueError(
                "headsplit dirty rows are not the exact clean-row complement"
            )
    if (
        not isinstance(dirty_rows, torch.Tensor)
        or dirty_rows.ndim != 1
        or dirty_rows.dtype != torch.long
        or dirty_rows.device.type != "cuda"
        or not dirty_rows.is_contiguous()
        or int(dirty_rows.numel()) != len(dirty_values)
    ):
        raise ValueError("headsplit device dirty rows must be contiguous CUDA int64")
    if (
        not isinstance(wo_a_weight, torch.Tensor)
        or wo_a_weight.ndim != 2
        or wo_a_weight.device.type != "cuda"
        or wo_a_weight.dtype != torch.bfloat16
        or not wo_a_weight.is_contiguous()
        or tuple(int(value) for value in wo_a_weight.shape)
        != (groups * o_lora_rank, owned_heads * head_dim)
    ):
        raise ValueError("headsplit wo_a weight has an incompatible CUDA layout")
    if dirty_rows.device != wo_a_weight.device:
        raise ValueError("headsplit dirty rows and wo_a weight use different devices")
    for view in projection_plan.views:
        if (
            view.values.device != wo_a_weight.device
            or view.values.dtype != wo_a_weight.dtype
            or not view.values.is_contiguous()
        ):
            raise ValueError("headsplit persistent z_off view is incompatible")

    cublas_certificate = _preflight_cublas_woa_fp32_fastpath(
        torch_module=torch,
        projection_plan=projection_plan,
        dirty_row_values=dirty_values,
        local_head_axes=local_axes,
        global_head_axes=global_axes,
        wo_a_weight=wo_a_weight,
        owned_heads=owned_heads,
        head_dim=head_dim,
        o_lora_rank=o_lora_rank,
    )
    weight_identity = _tensor_identity(wo_a_weight)
    if (
        cublas_certificate is not None
        and cublas_certificate.source_weight_identity != weight_identity
    ):
        cublas_certificate = None
    kernel_token = (
        PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN
        if cublas_certificate is not None
        else PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN
    )
    certificate_digest = (
        cublas_certificate.digest if cublas_certificate is not None else None
    )
    plan = PersistentHeadSplitWOAMergePlan(
        projection_plan=projection_plan,
        projection_plan_identity=_projection_plan_identity(projection_plan),
        dirty_rows=dirty_rows,
        dirty_rows_identity=_tensor_identity(dirty_rows),
        dirty_rows_cpu=dirty_rows_cpu,
        dirty_rows_cpu_identity=_tensor_identity(dirty_rows_cpu),
        dirty_row_values=dirty_values,
        wo_a_weight=wo_a_weight,
        wo_a_weight_identity=weight_identity,
        local_head_axes=local_axes,
        global_head_axes=global_axes,
        local_head_mask=sum(1 << axis for axis in local_axes),
        global_head_mask=sum(1 << axis for axis in global_axes),
        owned_heads=owned_heads,
        groups=groups,
        head_dim=head_dim,
        o_lora_rank=o_lora_rank,
        cublas_woa_fp32_certificate=cublas_certificate,
        kernel_token=kernel_token,
        digest=_headsplit_plan_digest(
            projection_plan=projection_plan,
            dirty_row_values=dirty_values,
            local_head_axes=local_axes,
            global_head_axes=global_axes,
            owned_heads=owned_heads,
            groups=groups,
            head_dim=head_dim,
            o_lora_rank=o_lora_rank,
            wo_a_weight_identity=weight_identity,
            kernel_token=kernel_token,
            cublas_woa_fp32_certificate_digest=certificate_digest,
        ),
    )
    plan.validate()
    return plan


def _launch_contiguous_views_merge(
    online,
    views: Tuple[PersistentProjectionView, ...],
):
    """Copy online and merge up to 16 non-overlapping spans in one launch.

    The pointer ABI is deliberately fixed.  Missing views repeat the final
    valid pointer with a zero-length span, so the kernel never needs a device
    pointer table or a host-built concatenation.  Plan validation proves the
    live spans are non-overlapping; therefore at most one masked offline load
    contributes to each output row.
    """

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        online_ptr,
        output_ptr,
        offline_ptr_0,
        output_start_0,
        local_start_0,
        span_rows_0,
        offline_ptr_1,
        output_start_1,
        local_start_1,
        span_rows_1,
        offline_ptr_2,
        output_start_2,
        local_start_2,
        span_rows_2,
        offline_ptr_3,
        output_start_3,
        local_start_3,
        span_rows_3,
        offline_ptr_4,
        output_start_4,
        local_start_4,
        span_rows_4,
        offline_ptr_5,
        output_start_5,
        local_start_5,
        span_rows_5,
        offline_ptr_6,
        output_start_6,
        local_start_6,
        span_rows_6,
        offline_ptr_7,
        output_start_7,
        local_start_7,
        span_rows_7,
        offline_ptr_8,
        output_start_8,
        local_start_8,
        span_rows_8,
        offline_ptr_9,
        output_start_9,
        local_start_9,
        span_rows_9,
        offline_ptr_10,
        output_start_10,
        local_start_10,
        span_rows_10,
        offline_ptr_11,
        output_start_11,
        local_start_11,
        span_rows_11,
        offline_ptr_12,
        output_start_12,
        local_start_12,
        span_rows_12,
        offline_ptr_13,
        output_start_13,
        local_start_13,
        span_rows_13,
        offline_ptr_14,
        output_start_14,
        local_start_14,
        span_rows_14,
        offline_ptr_15,
        output_start_15,
        local_start_15,
        span_rows_15,
        total_values: tl.constexpr,
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_values
        row = offsets // width
        column = offsets - row * width
        online_value = tl.load(online_ptr + offsets, mask=mask)
        offline_value = tl.zeros(offsets.shape, dtype=tl.float32)

        in_span_0 = (row >= output_start_0) & (row < output_start_0 + span_rows_0)
        source_0 = (row - output_start_0 + local_start_0) * width + column
        offline_value += tl.load(
            offline_ptr_0 + source_0, mask=mask & in_span_0, other=0.0
        )
        in_span_1 = (row >= output_start_1) & (row < output_start_1 + span_rows_1)
        source_1 = (row - output_start_1 + local_start_1) * width + column
        offline_value += tl.load(
            offline_ptr_1 + source_1, mask=mask & in_span_1, other=0.0
        )
        in_span_2 = (row >= output_start_2) & (row < output_start_2 + span_rows_2)
        source_2 = (row - output_start_2 + local_start_2) * width + column
        offline_value += tl.load(
            offline_ptr_2 + source_2, mask=mask & in_span_2, other=0.0
        )
        in_span_3 = (row >= output_start_3) & (row < output_start_3 + span_rows_3)
        source_3 = (row - output_start_3 + local_start_3) * width + column
        offline_value += tl.load(
            offline_ptr_3 + source_3, mask=mask & in_span_3, other=0.0
        )
        in_span_4 = (row >= output_start_4) & (row < output_start_4 + span_rows_4)
        source_4 = (row - output_start_4 + local_start_4) * width + column
        offline_value += tl.load(
            offline_ptr_4 + source_4, mask=mask & in_span_4, other=0.0
        )
        in_span_5 = (row >= output_start_5) & (row < output_start_5 + span_rows_5)
        source_5 = (row - output_start_5 + local_start_5) * width + column
        offline_value += tl.load(
            offline_ptr_5 + source_5, mask=mask & in_span_5, other=0.0
        )
        in_span_6 = (row >= output_start_6) & (row < output_start_6 + span_rows_6)
        source_6 = (row - output_start_6 + local_start_6) * width + column
        offline_value += tl.load(
            offline_ptr_6 + source_6, mask=mask & in_span_6, other=0.0
        )
        in_span_7 = (row >= output_start_7) & (row < output_start_7 + span_rows_7)
        source_7 = (row - output_start_7 + local_start_7) * width + column
        offline_value += tl.load(
            offline_ptr_7 + source_7, mask=mask & in_span_7, other=0.0
        )
        in_span_8 = (row >= output_start_8) & (row < output_start_8 + span_rows_8)
        source_8 = (row - output_start_8 + local_start_8) * width + column
        offline_value += tl.load(
            offline_ptr_8 + source_8, mask=mask & in_span_8, other=0.0
        )
        in_span_9 = (row >= output_start_9) & (row < output_start_9 + span_rows_9)
        source_9 = (row - output_start_9 + local_start_9) * width + column
        offline_value += tl.load(
            offline_ptr_9 + source_9, mask=mask & in_span_9, other=0.0
        )
        in_span_10 = (row >= output_start_10) & (row < output_start_10 + span_rows_10)
        source_10 = (row - output_start_10 + local_start_10) * width + column
        offline_value += tl.load(
            offline_ptr_10 + source_10, mask=mask & in_span_10, other=0.0
        )
        in_span_11 = (row >= output_start_11) & (row < output_start_11 + span_rows_11)
        source_11 = (row - output_start_11 + local_start_11) * width + column
        offline_value += tl.load(
            offline_ptr_11 + source_11, mask=mask & in_span_11, other=0.0
        )
        in_span_12 = (row >= output_start_12) & (row < output_start_12 + span_rows_12)
        source_12 = (row - output_start_12 + local_start_12) * width + column
        offline_value += tl.load(
            offline_ptr_12 + source_12, mask=mask & in_span_12, other=0.0
        )
        in_span_13 = (row >= output_start_13) & (row < output_start_13 + span_rows_13)
        source_13 = (row - output_start_13 + local_start_13) * width + column
        offline_value += tl.load(
            offline_ptr_13 + source_13, mask=mask & in_span_13, other=0.0
        )
        in_span_14 = (row >= output_start_14) & (row < output_start_14 + span_rows_14)
        source_14 = (row - output_start_14 + local_start_14) * width + column
        offline_value += tl.load(
            offline_ptr_14 + source_14, mask=mask & in_span_14, other=0.0
        )
        in_span_15 = (row >= output_start_15) & (row < output_start_15 + span_rows_15)
        source_15 = (row - output_start_15 + local_start_15) * width + column
        offline_value += tl.load(
            offline_ptr_15 + source_15, mask=mask & in_span_15, other=0.0
        )
        tl.store(
            output_ptr + offsets,
            online_value + offline_value,
            mask=mask,
        )

    if not views or len(views) > MAX_PERSISTENT_PROJECTION_VIEWS:
        raise ValueError("fused persistent merge requires one to 16 views")
    if not online.is_contiguous() or any(
        not view.values.is_contiguous() for view in views
    ):
        raise ValueError("fused persistent merge requires contiguous tensors")
    output = torch.empty_like(online)
    width = int(online.shape[1]) * int(online.shape[2])
    total_values = int(online.numel())
    block = 256
    padded_views: Tuple[Optional[PersistentProjectionView], ...] = views + (
        (None,) * (MAX_PERSISTENT_PROJECTION_VIEWS - len(views))
    )
    kernel_view_args = []
    padding_pointer = views[-1].values
    for view in padded_views:
        if view is None:
            kernel_view_args.extend((padding_pointer, 0, 0, 0))
        else:
            kernel_view_args.extend(
                (
                    view.values,
                    int(view.geometry.output_start),
                    int(view.geometry.local_start),
                    int(view.geometry.length),
                )
            )
    _kernel[(triton.cdiv(total_values, block),)](
        online,
        output,
        *kernel_view_args,
        total_values=total_values,
        width=width,
        BLOCK=block,
    )
    return output


def _launch_single_contiguous_merge(online, view: PersistentProjectionView):
    """Keep the single-view ABI while using the same one-launch kernel."""

    return _launch_contiguous_views_merge(online, (view,))


def merge_persistent_projection(
    online,
    plan: PersistentProjectionPlan,
    *,
    ragged_fallback: Optional[
        Callable[[object, PersistentProjectionPlan], object]
    ] = None,
):
    """Merge without materializing a dense offline projection.

    One to sixteen contiguous views use one fixed-pointer CUDA kernel.  This is
    the production 8x8K/16x8K path and performs no gather, concatenation, or
    per-view launch.  Ragged row geometry is fail-closed by default; a caller
    that owns a separately audited fallback must pass it explicitly.
    """

    import torch

    if not isinstance(plan, PersistentProjectionPlan):
        raise TypeError("persistent projection merge requires a certified plan")
    plan.validate()
    if not isinstance(online, torch.Tensor) or online.ndim != 3:
        raise ValueError("online projection must be [rows,groups,rank]")
    if tuple(int(value) for value in online.shape) != (
        plan.total_rows,
        *plan.tail_shape,
    ):
        raise ValueError("online projection shape differs from the merge plan")
    if online.device.type != "cuda":
        raise ValueError("production persistent merge requires CUDA")
    for view in plan.views:
        if view.values.device != online.device or view.values.dtype != online.dtype:
            raise ValueError("online/offline persistent tensors are incompatible")

    if len(plan.views) == 1 and plan.is_single_launch_eligible:
        return _launch_single_contiguous_merge(online, plan.views[0])
    if plan.is_single_launch_eligible:
        return _launch_contiguous_views_merge(online, plan.views)
    if ragged_fallback is None:
        raise RuntimeError(
            "ragged persistent projection geometry has no implicit GPU fallback; "
            "pass an audited ragged_fallback explicitly"
        )
    merged = ragged_fallback(online, plan)
    if not isinstance(merged, torch.Tensor) or tuple(int(v) for v in merged.shape) != (
        plan.total_rows,
        *plan.tail_shape,
    ):
        raise ValueError("ragged fallback returned an incompatible projection")
    if merged.device != online.device or merged.dtype != online.dtype:
        raise ValueError("ragged fallback changed projection device or dtype")
    return merged


def _fixed_projection_view_kernel_args(
    views: Tuple[PersistentProjectionView, ...],
) -> Tuple[object, ...]:
    if not views or len(views) > MAX_PERSISTENT_PROJECTION_VIEWS:
        raise ValueError("headsplit merge requires one to sixteen z_off views")
    padded: Tuple[Optional[PersistentProjectionView], ...] = views + (
        (None,) * (MAX_PERSISTENT_PROJECTION_VIEWS - len(views))
    )
    padding_pointer = views[-1].values
    arguments = []
    for view in padded:
        if view is None:
            arguments.extend((padding_pointer, 0, 0, 0))
        else:
            arguments.extend(
                (
                    view.values,
                    int(view.geometry.output_start),
                    int(view.geometry.local_start),
                    int(view.geometry.length),
                )
            )
    return tuple(arguments)


def _launch_global_woa_zoff_kernel(
    rotated_attention_output,
    plan: PersistentHeadSplitWOAMergePlan,
):
    """Project global heads for all rows and fold z_off into the same store."""

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _global_woa_zoff_kernel(
        attention_ptr,
        weight_ptr,
        output_ptr,
        offline_ptr_0,
        output_start_0,
        local_start_0,
        span_rows_0,
        offline_ptr_1,
        output_start_1,
        local_start_1,
        span_rows_1,
        offline_ptr_2,
        output_start_2,
        local_start_2,
        span_rows_2,
        offline_ptr_3,
        output_start_3,
        local_start_3,
        span_rows_3,
        offline_ptr_4,
        output_start_4,
        local_start_4,
        span_rows_4,
        offline_ptr_5,
        output_start_5,
        local_start_5,
        span_rows_5,
        offline_ptr_6,
        output_start_6,
        local_start_6,
        span_rows_6,
        offline_ptr_7,
        output_start_7,
        local_start_7,
        span_rows_7,
        offline_ptr_8,
        output_start_8,
        local_start_8,
        span_rows_8,
        offline_ptr_9,
        output_start_9,
        local_start_9,
        span_rows_9,
        offline_ptr_10,
        output_start_10,
        local_start_10,
        span_rows_10,
        offline_ptr_11,
        output_start_11,
        local_start_11,
        span_rows_11,
        offline_ptr_12,
        output_start_12,
        local_start_12,
        span_rows_12,
        offline_ptr_13,
        output_start_13,
        local_start_13,
        span_rows_13,
        offline_ptr_14,
        output_start_14,
        local_start_14,
        span_rows_14,
        offline_ptr_15,
        output_start_15,
        local_start_15,
        span_rows_15,
        total_rows,
        OWNED_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        O_LORA_RANK: tl.constexpr,
        GLOBAL_HEAD_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        rank_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = row_offsets < total_rows
        rank_mask = rank_offsets < O_LORA_RANK
        input_width = OWNED_HEADS * HEAD_DIM
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for head_axis in tl.static_range(0, OWNED_HEADS):
            if ((GLOBAL_HEAD_MASK >> head_axis) & 1) == 1:
                for k_start in tl.static_range(0, HEAD_DIM, BLOCK_K):
                    k_offsets = k_start + tl.arange(0, BLOCK_K)
                    k_mask = k_offsets < HEAD_DIM
                    input_offsets = (
                        row_offsets[:, None] * input_width
                        + head_axis * HEAD_DIM
                        + k_offsets[None, :]
                    )
                    weight_offsets = (
                        rank_offsets[None, :] * input_width
                        + head_axis * HEAD_DIM
                        + k_offsets[:, None]
                    )
                    input_values = tl.load(
                        attention_ptr + input_offsets,
                        mask=row_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    weight_values = tl.load(
                        weight_ptr + weight_offsets,
                        mask=k_mask[:, None] & rank_mask[None, :],
                        other=0.0,
                    )
                    accumulator += tl.dot(input_values, weight_values)

        offline_value = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        in_span_0 = (row_offsets >= output_start_0) & (
            row_offsets < output_start_0 + span_rows_0
        )
        source_row_0 = row_offsets - output_start_0 + local_start_0
        offline_value += tl.load(
            offline_ptr_0
            + source_row_0[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_0[:, None],
            other=0.0,
        )
        in_span_1 = (row_offsets >= output_start_1) & (
            row_offsets < output_start_1 + span_rows_1
        )
        source_row_1 = row_offsets - output_start_1 + local_start_1
        offline_value += tl.load(
            offline_ptr_1
            + source_row_1[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_1[:, None],
            other=0.0,
        )
        in_span_2 = (row_offsets >= output_start_2) & (
            row_offsets < output_start_2 + span_rows_2
        )
        source_row_2 = row_offsets - output_start_2 + local_start_2
        offline_value += tl.load(
            offline_ptr_2
            + source_row_2[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_2[:, None],
            other=0.0,
        )
        in_span_3 = (row_offsets >= output_start_3) & (
            row_offsets < output_start_3 + span_rows_3
        )
        source_row_3 = row_offsets - output_start_3 + local_start_3
        offline_value += tl.load(
            offline_ptr_3
            + source_row_3[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_3[:, None],
            other=0.0,
        )
        in_span_4 = (row_offsets >= output_start_4) & (
            row_offsets < output_start_4 + span_rows_4
        )
        source_row_4 = row_offsets - output_start_4 + local_start_4
        offline_value += tl.load(
            offline_ptr_4
            + source_row_4[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_4[:, None],
            other=0.0,
        )
        in_span_5 = (row_offsets >= output_start_5) & (
            row_offsets < output_start_5 + span_rows_5
        )
        source_row_5 = row_offsets - output_start_5 + local_start_5
        offline_value += tl.load(
            offline_ptr_5
            + source_row_5[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_5[:, None],
            other=0.0,
        )
        in_span_6 = (row_offsets >= output_start_6) & (
            row_offsets < output_start_6 + span_rows_6
        )
        source_row_6 = row_offsets - output_start_6 + local_start_6
        offline_value += tl.load(
            offline_ptr_6
            + source_row_6[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_6[:, None],
            other=0.0,
        )
        in_span_7 = (row_offsets >= output_start_7) & (
            row_offsets < output_start_7 + span_rows_7
        )
        source_row_7 = row_offsets - output_start_7 + local_start_7
        offline_value += tl.load(
            offline_ptr_7
            + source_row_7[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_7[:, None],
            other=0.0,
        )
        in_span_8 = (row_offsets >= output_start_8) & (
            row_offsets < output_start_8 + span_rows_8
        )
        source_row_8 = row_offsets - output_start_8 + local_start_8
        offline_value += tl.load(
            offline_ptr_8
            + source_row_8[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_8[:, None],
            other=0.0,
        )
        in_span_9 = (row_offsets >= output_start_9) & (
            row_offsets < output_start_9 + span_rows_9
        )
        source_row_9 = row_offsets - output_start_9 + local_start_9
        offline_value += tl.load(
            offline_ptr_9
            + source_row_9[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_9[:, None],
            other=0.0,
        )
        in_span_10 = (row_offsets >= output_start_10) & (
            row_offsets < output_start_10 + span_rows_10
        )
        source_row_10 = row_offsets - output_start_10 + local_start_10
        offline_value += tl.load(
            offline_ptr_10
            + source_row_10[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_10[:, None],
            other=0.0,
        )
        in_span_11 = (row_offsets >= output_start_11) & (
            row_offsets < output_start_11 + span_rows_11
        )
        source_row_11 = row_offsets - output_start_11 + local_start_11
        offline_value += tl.load(
            offline_ptr_11
            + source_row_11[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_11[:, None],
            other=0.0,
        )
        in_span_12 = (row_offsets >= output_start_12) & (
            row_offsets < output_start_12 + span_rows_12
        )
        source_row_12 = row_offsets - output_start_12 + local_start_12
        offline_value += tl.load(
            offline_ptr_12
            + source_row_12[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_12[:, None],
            other=0.0,
        )
        in_span_13 = (row_offsets >= output_start_13) & (
            row_offsets < output_start_13 + span_rows_13
        )
        source_row_13 = row_offsets - output_start_13 + local_start_13
        offline_value += tl.load(
            offline_ptr_13
            + source_row_13[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_13[:, None],
            other=0.0,
        )
        in_span_14 = (row_offsets >= output_start_14) & (
            row_offsets < output_start_14 + span_rows_14
        )
        source_row_14 = row_offsets - output_start_14 + local_start_14
        offline_value += tl.load(
            offline_ptr_14
            + source_row_14[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_14[:, None],
            other=0.0,
        )
        in_span_15 = (row_offsets >= output_start_15) & (
            row_offsets < output_start_15 + span_rows_15
        )
        source_row_15 = row_offsets - output_start_15 + local_start_15
        offline_value += tl.load(
            offline_ptr_15
            + source_row_15[:, None] * O_LORA_RANK
            + rank_offsets[None, :],
            mask=row_mask[:, None] & rank_mask[None, :] & in_span_15[:, None],
            other=0.0,
        )

        output_offsets = (
            row_offsets[:, None] * O_LORA_RANK + rank_offsets[None, :]
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator + offline_value,
            mask=row_mask[:, None] & rank_mask[None, :],
        )

    output = torch.empty(
        (
            int(plan.projection_plan.total_rows),
            int(plan.groups),
            int(plan.o_lora_rank),
        ),
        dtype=rotated_attention_output.dtype,
        device=rotated_attention_output.device,
    )
    block_m = 16
    block_n = 32
    block_k = 32
    grid = (
        triton.cdiv(int(plan.projection_plan.total_rows), block_m),
        triton.cdiv(int(plan.o_lora_rank), block_n),
    )
    _global_woa_zoff_kernel[grid](
        rotated_attention_output,
        plan.wo_a_weight,
        output,
        *_fixed_projection_view_kernel_args(plan.projection_plan.views),
        int(plan.projection_plan.total_rows),
        OWNED_HEADS=int(plan.owned_heads),
        HEAD_DIM=int(plan.head_dim),
        O_LORA_RANK=int(plan.o_lora_rank),
        GLOBAL_HEAD_MASK=int(plan.global_head_mask),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


def _launch_dirty_local_woa_add_kernel(
    rotated_attention_output,
    output,
    plan: PersistentHeadSplitWOAMergePlan,
) -> None:
    """Project local heads only at dirty rows and add into the fused output."""

    import triton
    import triton.language as tl

    @triton.jit
    def _dirty_local_woa_add_kernel(
        attention_ptr,
        weight_ptr,
        dirty_rows_ptr,
        output_ptr,
        dirty_count,
        OWNED_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        O_LORA_RANK: tl.constexpr,
        LOCAL_HEAD_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        dirty_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        rank_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        dirty_mask = dirty_offsets < dirty_count
        rank_mask = rank_offsets < O_LORA_RANK
        logical_rows = tl.load(
            dirty_rows_ptr + dirty_offsets, mask=dirty_mask, other=0
        )
        input_width = OWNED_HEADS * HEAD_DIM
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for head_axis in tl.static_range(0, OWNED_HEADS):
            if ((LOCAL_HEAD_MASK >> head_axis) & 1) == 1:
                for k_start in tl.static_range(0, HEAD_DIM, BLOCK_K):
                    k_offsets = k_start + tl.arange(0, BLOCK_K)
                    k_mask = k_offsets < HEAD_DIM
                    input_offsets = (
                        logical_rows[:, None] * input_width
                        + head_axis * HEAD_DIM
                        + k_offsets[None, :]
                    )
                    weight_offsets = (
                        rank_offsets[None, :] * input_width
                        + head_axis * HEAD_DIM
                        + k_offsets[:, None]
                    )
                    input_values = tl.load(
                        attention_ptr + input_offsets,
                        mask=dirty_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    weight_values = tl.load(
                        weight_ptr + weight_offsets,
                        mask=k_mask[:, None] & rank_mask[None, :],
                        other=0.0,
                    )
                    accumulator += tl.dot(input_values, weight_values)

        output_offsets = (
            logical_rows[:, None] * O_LORA_RANK + rank_offsets[None, :]
        )
        output_mask = dirty_mask[:, None] & rank_mask[None, :]
        previous = tl.load(output_ptr + output_offsets, mask=output_mask, other=0.0)
        tl.store(
            output_ptr + output_offsets,
            previous + accumulator,
            mask=output_mask,
        )

    dirty_count = int(plan.dirty_rows.numel())
    if dirty_count == 0:
        return
    block_m = 16
    block_n = 32
    block_k = 32
    grid = (
        triton.cdiv(dirty_count, block_m),
        triton.cdiv(int(plan.o_lora_rank), block_n),
    )
    _dirty_local_woa_add_kernel[grid](
        rotated_attention_output,
        plan.wo_a_weight,
        plan.dirty_rows,
        output,
        dirty_count,
        OWNED_HEADS=int(plan.owned_heads),
        HEAD_DIM=int(plan.head_dim),
        O_LORA_RANK=int(plan.o_lora_rank),
        LOCAL_HEAD_MASK=int(plan.local_head_mask),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


def _project_merge_persistent_headsplit_cublas_fp32(
    rotated_attention_output,
    plan: PersistentHeadSplitWOAMergePlan,
):
    """Run the certified cuBLAS formula with exactly one final BF16 round."""

    import torch

    certificate = plan.cublas_woa_fp32_certificate
    if not isinstance(certificate, CublasWOAFP32FastPathCertificate):
        raise ValueError("cuBLAS wo_a execution has no certificate")
    total_rows = int(plan.projection_plan.total_rows)
    head_dim = int(plan.head_dim)
    o_lora_rank = int(plan.o_lora_rank)

    global_attention = rotated_attention_output.narrow(
        1,
        int(certificate.global_axis_start),
        int(certificate.global_axis_count),
    ).reshape(
        total_rows,
        int(certificate.global_axis_count) * head_dim,
    )
    output_fp32 = torch.mm(
        global_attention,
        certificate.packed_global_weight.T,
        out_dtype=torch.float32,
    )

    view = plan.projection_plan.views[0]
    output_start = int(view.geometry.output_start)
    local_start = int(view.geometry.local_start)
    span_rows = int(view.geometry.length)
    offline_projection = view.values.narrow(
        0,
        local_start,
        span_rows,
    ).reshape(span_rows, o_lora_rank)
    # The FP32 destination makes the BF16 offline input participate in an
    # FP32 add without materializing a second full-span FP32 tensor.
    output_fp32.narrow(0, output_start, span_rows).add_(offline_projection)

    dirty_count = len(plan.dirty_row_values)
    if dirty_count:
        local_attention = rotated_attention_output.narrow(
            0,
            0,
            dirty_count,
        ).narrow(
            1,
            int(certificate.local_axis_start),
            int(certificate.local_axis_count),
        ).reshape(
            dirty_count,
            int(certificate.local_axis_count) * head_dim,
        )
        dirty_local_fp32 = torch.mm(
            local_attention,
            certificate.packed_local_weight.T,
            out_dtype=torch.float32,
        )
        output_fp32.narrow(0, 0, dirty_count).add_(dirty_local_fp32)

    return output_fp32.reshape(
        total_rows,
        int(plan.groups),
        o_lora_rank,
    ).to(dtype=torch.bfloat16)


def project_merge_persistent_headsplit(
    rotated_attention_output,
    plan: PersistentHeadSplitWOAMergePlan,
    *,
    wo_a_weight=None,
    committed_plan_identity: Optional[Tuple[object, ...]] = None,
):
    """Run the preflight-selected ``wo_a + z_off`` pipeline without fallback."""

    import torch

    if not isinstance(plan, PersistentHeadSplitWOAMergePlan):
        raise TypeError("headsplit projection requires a certified merge plan")
    if committed_plan_identity is None:
        # Direct/noncommitted callers retain the complete geometry proof.
        plan.validate()
    else:
        plan.validate_live(
            committed_plan_identity=committed_plan_identity,
        )
    if wo_a_weight is not None and wo_a_weight is not plan.wo_a_weight:
        raise ValueError("headsplit projection received a different wo_a weight")
    if (
        not isinstance(rotated_attention_output, torch.Tensor)
        or rotated_attention_output.ndim != 3
        or rotated_attention_output.device.type != "cuda"
        or rotated_attention_output.dtype != torch.bfloat16
        or not rotated_attention_output.is_contiguous()
        or tuple(int(value) for value in rotated_attention_output.shape)
        != (
            int(plan.projection_plan.total_rows),
            int(plan.owned_heads),
            int(plan.head_dim),
        )
    ):
        raise ValueError(
            "headsplit attention output must be contiguous CUDA BF16 [T,H,D]"
        )
    if rotated_attention_output.device != plan.wo_a_weight.device:
        raise ValueError("headsplit attention output is on a different device")
    if plan.uses_cublas_woa_fp32_fastpath:
        return _project_merge_persistent_headsplit_cublas_fp32(
            rotated_attention_output,
            plan,
        )
    output = _launch_global_woa_zoff_kernel(rotated_attention_output, plan)
    _launch_dirty_local_woa_add_kernel(
        rotated_attention_output,
        output,
        plan,
    )
    return output


__all__ = [
    "CublasWOAFP32FastPathCertificate",
    "MAX_PERSISTENT_PROJECTION_VIEWS",
    "PERSISTENT_HEADSPLIT_CUBLAS_WOA_FP32_KERNEL_TOKEN",
    "PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN",
    "PersistentHeadSplitWOAMergePlan",
    "PersistentProjectionPlan",
    "PersistentProjectionView",
    "ProjectionSpanGeometry",
    "build_persistent_projection_plan",
    "merge_persistent_projection",
    "preflight_persistent_headsplit_woa_merge",
    "project_merge_persistent_headsplit",
]
