"""Composite restore preparation for RedKnot's production DSV4 backend.

This file keeps the v3 path separate from the legacy single-request restore
implementation.  It prepares one flattened ragged row certificate, direct
GPU-resident z_off views, and one shared-latent schedule/workspace per request.
Context preparation mutates no cache and executes no TP collective.  The
separate :class:`LayerCompositeCommitBuilder` added below owns the one-shot,
per-layer collective gate.  Until that gate returns an authorization, sparse
Q, restored cache slots, and persistent z_off rows are proposals only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - minimal CPU-only installs.
    _orjson = None

from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
    MLA_OFF_DIAGNOSTIC_ABLATION_FULL,
    MLA_OFF_DIAGNOSTIC_ABLATION_ZOFF_ONLY,
    MLA_OFF_DIAGNOSTIC_ABLATIONS,
    MLA_OFF_INDEPENDENT_POSITION_SEMANTICS,
    MLAOffLayerSpec,
    MLAOffRestoreRows,
    MLAOffRuntimeContext,
    build_restore_rows,
    get_dsv4_mla_off_controller,
    resolve_mla_off_diagnostic_ablation,
)
from sglang.srt.layers.attention.redknot.dsv4_reuse_batch import (
    RequestReuseLayout,
    SegmentBinding,
    assemble_validated_batched_reuse_plan,
    rebind_validated_batched_reuse_plan,
)
from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
    ATTENTION_COMPRESSOR_STATE,
    ArtifactGenerationBinding,
    C4,
    C128,
    CacheDomainBinding,
    CommitOutcome,
    COMMIT_SCOPE_FORWARD_FRAGMENT,
    COMMIT_SCOPE_FORWARD_RESERVED,
    CompositeForwardProposal,
    ForwardExecutionLedger,
    ForwardCommitSession,
    ForwardIdentity,
    ForwardPreparePreflight,
    GpuViewBinding,
    INDEXER,
    INDEXER_COMPRESSOR_STATE,
    LayerCompressionBinding,
    LayerExecutionReceipt,
    LayerReservationBinding,
    OMISSION_PROFILE_FULL,
    OMISSION_PROFILE_SHARED_ONLY,
    OMISSION_PROFILE_ZOFF_ONLY,
    OmissionAuthorization,
    RaggedBatchGeometry,
    RaggedRequestGeometry,
    SharedLatentBinding,
    SequentialQArenaBinding,
    SparseQBinding,
    SparseQInstallAuthorization,
    SWA,
    ZOffGpuViewBinding,
    build_cache_builders_preflight,
    build_forward_prepare_preflight,
    build_layer_reservation_binding,
    build_sequential_q_arena_binding,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_cache import (
    SegmentPlacement,
    SharedLatentRestorePlan,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
    DeviceRestoreWorkspace,
)
from sglang.srt.layers.attention.redknot.dsv4_timing import (
    timed as _redknot_runtime_timed,
)


_INDEPENDENT_RELOCATION_PROFILE = (
    "pure_headsplit_independent_rope_relocation_fullscope_"
    "boundary128_3_37_3_v1"
)
_COMBINED_ROW_SPARSE_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)


@dataclass(frozen=True)
class SharedRequestRestoreState:
    request_index: int
    flat_row_offset: int
    row_count: int
    cpu_plan: object
    pin: object
    schedule: object
    prepared: object
    reusable: bool = True

    @property
    def dirty_rows(self) -> Tuple[int, ...]:
        return tuple(
            self.flat_row_offset + int(row)
            for row in self.cpu_plan.dirty_output_rows
        )

    def validate(self) -> None:
        """Revalidate the exact request-scoped pin/schedule/workspace chain."""

        if type(self.request_index) is not int or self.request_index < 0:
            raise ValueError("shared request index is invalid")
        if type(self.flat_row_offset) is not int or self.flat_row_offset < 0:
            raise ValueError("shared request flat-row offset is invalid")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("shared request row count is invalid")
        if type(self.reusable) is not bool:
            raise TypeError("shared request reusable flag must be boolean")
        dirty = tuple(int(row) for row in self.cpu_plan.dirty_output_rows)
        if dirty != tuple(sorted(set(dirty))) or any(
            row < 0 or row >= self.row_count for row in dirty
        ):
            raise ValueError("shared request dirty rows changed")
        if not self.reusable:
            if self.pin is not None or self.schedule is not None or self.prepared is not None:
                raise ValueError("dense request cannot own shared artifact state")
            if dirty != tuple(range(self.row_count)):
                raise ValueError("dense request must keep every row dirty")
            return
        validate_open = getattr(self.pin, "validate_open", None)
        if not callable(validate_open):
            raise TypeError("shared request pin has no validation boundary")
        validate_open()
        if getattr(self.schedule, "pin_digest", None) != getattr(
            self.pin, "digest", None
        ):
            raise ValueError("shared request schedule belongs to another pin")
        if len(tuple(getattr(self.schedule, "positions", ()))) != self.row_count:
            raise ValueError("shared request schedule row geometry changed")
        if getattr(self.prepared, "pin", None) is not self.pin:
            raise ValueError("prepared shared restore belongs to another pin")
        if getattr(self.prepared, "schedule", None) is not self.schedule:
            raise ValueError("prepared shared restore schedule identity changed")
        workspace = getattr(self.prepared, "workspace", None)
        if getattr(workspace, "loaded_digest", None) != getattr(
            self.schedule, "digest", None
        ):
            raise ValueError("shared restore workspace no longer holds its schedule")

    def close(self) -> None:
        close = getattr(self.pin, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class _DenseLayerRestorePlan:
    layer_id: int
    compress_ratio: int


@dataclass(frozen=True)
class _DenseSharedRestorePlan:
    dirty_output_rows: Tuple[int, ...]
    layers: Mapping[int, _DenseLayerRestorePlan]
    spec: object = None


def _immutable_forward_generation(value: object) -> object:
    """Copy the scheduler generation into an immutable cache binding."""

    if isinstance(value, tuple):
        generation = tuple(value)
        if not generation:
            raise ValueError("composite forward generation must not be empty")
        return generation
    if isinstance(value, str) and value:
        return value
    raise ValueError("composite restore requires a scheduler forward generation")


def _plan_source_identity(plans: object) -> Tuple[object, ...]:
    try:
        count = len(plans)  # type: ignore[arg-type]
        members = tuple(id(plans[index]) for index in range(count))  # type: ignore[index]
    except Exception as exc:
        raise TypeError("composite reuse plans must be an indexable sequence") from exc
    return (id(plans), type(plans).__qualname__, count, members)


def _freeze_plan_payload(value: object) -> object:
    """Detach serving execution from the mutable request JSON tree.

    The request plan is fully hashed before the first reusable layer.  Keep an
    immutable recursive copy for all later layers so a cache hit only needs to
    revalidate the scheduler-owned root/member identities; it never has to
    serialize the same 64K plan another 36 times.  Unsupported objects fail at
    the first layer instead of leaking mutable state into the forward cache.
    """

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_plan_payload(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_plan_payload(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(
        "composite reuse plan contains a non-canonical mutable object"
    )


def _composite_plan_digest(plans: object) -> str:
    """Hash all request plans without consulting any tensor payload."""

    try:
        encoded = json.dumps(
            plans,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("composite reuse plans are not canonically serializable") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _geometry_template_plan_payload(value: object) -> object:
    """Remove request-observability fields from an immutable geometry key.

    ``benchmark_request_id`` changes for every HTTP request but cannot change
    row placement, artifact identity, or token contents.  Keeping it in the
    forward/TP identity is still required; excluding it only from this
    explicit template key lets a fixed RAG corpus reuse already validated CPU
    geometry.  Every hit is additionally guarded by exact scheduler token and
    position equality plus committed artifact generations.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _geometry_template_plan_payload(item)
            for key, item in value.items()
            if str(key) != "benchmark_request_id"
        }
    if isinstance(value, (tuple, list)):
        return [_geometry_template_plan_payload(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("geometry template plan contains a non-canonical value")


# One 8-document first-prefix RAG case uses twelve stable geometry keys:
# seed(2), per-measurement prefix refresh(2), and the eight online forwards.
# Eight entries caused a FIFO eviction cascade, so a repeated quality/QPS
# request rebuilt every immutable CPU restore plan.  Sixteen covers that exact
# single-request working set while keeping this rank-local CPU cache bounded.
_GEOMETRY_TEMPLATE_CACHE_MAX_ENTRIES = 16


def _optional_source_identity(value: object) -> Tuple[object, ...]:
    if value is None:
        return ("none",)
    if callable(getattr(value, "data_ptr", None)):
        return ("tensor", *_tensor_identity(value))
    if isinstance(value, (list, tuple)):
        # Scheduler CPU sequences are tiny.  Include their scalar values so an
        # in-place update cannot reuse geometry captured for an older chunk.
        return (
            "sequence",
            id(value),
            type(value).__qualname__,
            tuple(value),
        )
    return ("object", id(value), type(value).__qualname__)


@dataclass(frozen=True)
class CompositeGeometrySourceBinding:
    """Immutable proof that cached CPU geometry still belongs to this forward.

    Values from ``positions`` and ``input_ids`` are intentionally absent.  A
    cache hit is decided from their exact tensor/storage identity and mutation
    version, while the values copied during the first reusable layer remain
    independently bound by the cached input/reuse digests.
    """

    forward_generation: object
    plan_source_identity: Tuple[object, ...]
    plan_digest: str
    positions_identity: Tuple[object, ...]
    input_ids_identity: Tuple[object, ...]
    ragged_lengths: Tuple[int, ...]
    ragged_source_identity: Tuple[object, ...]
    scheduler_totals_source_identity: Tuple[object, ...]
    scheduler_extents_source_identity: Tuple[object, ...]
    batch_size: int
    q_rows: int

    @classmethod
    def capture(
        cls,
        *,
        forward_generation: object,
        raw_plans: object,
        positions: object,
        input_ids: object,
        ragged_lengths: Tuple[int, ...],
        ragged_source: object,
        scheduler_totals_source: object,
        scheduler_extents_source: object,
        batch_size: int,
        q_rows: int,
    ) -> "CompositeGeometrySourceBinding":
        return cls(
            forward_generation=_immutable_forward_generation(forward_generation),
            plan_source_identity=_plan_source_identity(raw_plans),
            plan_digest=_composite_plan_digest(raw_plans),
            positions_identity=_tensor_identity(positions),
            input_ids_identity=_tensor_identity(input_ids),
            ragged_lengths=tuple(int(value) for value in ragged_lengths),
            ragged_source_identity=_optional_source_identity(ragged_source),
            scheduler_totals_source_identity=_optional_source_identity(
                scheduler_totals_source
            ),
            scheduler_extents_source_identity=_optional_source_identity(
                scheduler_extents_source
            ),
            batch_size=int(batch_size),
            q_rows=int(q_rows),
        )

    def validate(
        self,
        *,
        forward_generation: object,
        raw_plans: object,
        positions: object,
        input_ids: object,
        ragged_lengths: Tuple[int, ...],
        ragged_source: object,
        scheduler_totals_source: object,
        scheduler_extents_source: object,
        batch_size: int,
        q_rows: int,
    ) -> None:
        current = self.capture(
            forward_generation=forward_generation,
            raw_plans=raw_plans,
            positions=positions,
            input_ids=input_ids,
            ragged_lengths=ragged_lengths,
            ragged_source=ragged_source,
            scheduler_totals_source=scheduler_totals_source,
            scheduler_extents_source=scheduler_extents_source,
            batch_size=batch_size,
            q_rows=q_rows,
        )
        if current.forward_generation != self.forward_generation:
            raise ValueError("composite geometry belongs to another forward generation")
        if current.plan_source_identity != self.plan_source_identity:
            raise ValueError("composite geometry plan object identity changed")
        if current.plan_digest != self.plan_digest:
            raise ValueError("composite geometry plan digest changed")
        if current.positions_identity != self.positions_identity:
            raise ValueError("composite geometry positions identity/version changed")
        if current.input_ids_identity != self.input_ids_identity:
            raise ValueError("composite geometry input_ids identity/version changed")
        if (
            current.ragged_lengths != self.ragged_lengths
            or current.ragged_source_identity != self.ragged_source_identity
            or current.batch_size != self.batch_size
            or current.q_rows != self.q_rows
        ):
            raise ValueError("composite geometry ragged batch binding changed")
        if (
            current.scheduler_totals_source_identity
            != self.scheduler_totals_source_identity
            or current.scheduler_extents_source_identity
            != self.scheduler_extents_source_identity
        ):
            raise ValueError("composite geometry scheduler-length source changed")

    def validate_live(
        self,
        *,
        forward_generation: object,
        raw_plans: object,
        positions: object,
        input_ids: object,
        ragged_lengths: Tuple[int, ...],
        ragged_source: object,
        scheduler_totals_source: object,
        scheduler_extents_source: object,
        batch_size: int,
        q_rows: int,
    ) -> None:
        """O(requests) identity check after the plan tree was frozen.

        ``capture`` already bound the canonical full JSON digest.  Later
        layers consume the recursively immutable copy owned by
        :class:`CompositeForwardGeometry`, so reserializing the external JSON
        cannot strengthen the execution contract.  Root/member identity,
        tensor version, forward generation, and scheduler sources remain live
        ABA guards on every layer.
        """

        if (
            _immutable_forward_generation(forward_generation)
            != self.forward_generation
        ):
            raise ValueError(
                "composite geometry belongs to another forward generation"
            )
        if _plan_source_identity(raw_plans) != self.plan_source_identity:
            raise ValueError("composite geometry plan object identity changed")
        if _tensor_identity(positions) != self.positions_identity:
            raise ValueError(
                "composite geometry positions identity/version changed"
            )
        if _tensor_identity(input_ids) != self.input_ids_identity:
            raise ValueError(
                "composite geometry input_ids identity/version changed"
            )
        if (
            tuple(int(value) for value in ragged_lengths)
            != self.ragged_lengths
            or _optional_source_identity(ragged_source)
            != self.ragged_source_identity
            or int(batch_size) != self.batch_size
            or int(q_rows) != self.q_rows
        ):
            raise ValueError("composite geometry ragged batch binding changed")
        if (
            _optional_source_identity(scheduler_totals_source)
            != self.scheduler_totals_source_identity
            or _optional_source_identity(scheduler_extents_source)
            != self.scheduler_extents_source_identity
        ):
            raise ValueError(
                "composite geometry scheduler-length source changed"
            )


@dataclass(frozen=True)
class _CachedRestoreBinding:
    rows: object
    token_ids_cpu: object


@dataclass(frozen=True)
class _CachedRequestGeometry:
    request_index: int
    flat_row_start: int
    row_count: int
    request_token: str
    logical_positions: Tuple[int, ...]
    input_token_ids: Tuple[int, ...]
    query_start: int
    dirty_request_rows: Tuple[int, ...]
    segments: Tuple[SegmentBinding, ...]
    segment_metadata: Tuple[Mapping[str, object], ...]
    restore_bindings: Tuple[_CachedRestoreBinding, ...]
    fallback_reason: str = ""

    @property
    def reusable(self) -> bool:
        return not self.fallback_reason


def _request_extent_matches_geometry(
    request: _CachedRequestGeometry,
    plan: Mapping[str, object],
    extent: int,
) -> bool:
    """Bind sparse active rows to the complete scheduler microforward."""

    positions = tuple(request.logical_positions)
    if not positions:
        return False
    if str(plan.get("mla_off_execution_profile", "")) != (
        _COMBINED_ROW_SPARSE_PROFILE
    ):
        return extent == positions[-1] + 1
    query_start = plan.get("query_start")
    if type(query_start) is not int or query_start < 0:
        return False
    if extent > query_start:
        # Query/sentinel rows are never selected away.
        return (
            positions[0] >= query_start
            and extent == positions[-1] + 1
            and all(
                right == left + 1
                for left, right in zip(positions, positions[1:])
            )
        )
    raw_merged_tokens = plan.get("merged_prefill_tokens", 0)
    if type(raw_merged_tokens) is not int or raw_merged_tokens < 0:
        return False
    merged_tokens = int(raw_merged_tokens)
    if merged_tokens:
        chunk_start = max(0, int(extent) - merged_tokens)
        if positions[0] < chunk_start or positions[-1] >= extent:
            return False
        segments = tuple(
            sorted(
                (
                    segment
                    for segment in tuple(plan.get("segments", ()))
                    if isinstance(segment, Mapping)
                    and type(segment.get("global_offset")) is int
                    and type(segment.get("length")) is int
                    and chunk_start <= int(segment["global_offset"])
                    and int(segment["global_offset"])
                    + int(segment["length"])
                    <= extent
                ),
                key=lambda segment: int(segment["global_offset"]),
            )
        )
        cursor = chunk_start
        for segment in segments:
            segment_start = int(segment["global_offset"])
            segment_length = int(segment["length"])
            if segment_length <= 0 or segment_start != cursor:
                return False
            cursor = segment_start + segment_length
        return bool(segments) and cursor == extent
    matches = tuple(
        segment
        for segment in tuple(plan.get("segments", ()))
        if isinstance(segment, Mapping)
        and type(segment.get("global_offset")) is int
        and type(segment.get("length")) is int
        and segment["global_offset"] + segment["length"] == extent
        and positions[0] >= segment["global_offset"]
        and positions[-1] < extent
    )
    return len(matches) == 1


_QUERY_SUFFIX_FULL_LOCAL_REASON = "query_suffix_only"


def _intentional_full_local_reason(
    requests: Sequence[_CachedRequestGeometry],
    *,
    q_row_count: int,
    reused_count: int,
) -> str:
    """Recognize the one safe zero-reuse geometry.

    A restore forward can legitimately contain no clean row when the current
    chunk is wholly the online query suffix.  Do not infer that state merely
    from an empty reuse bitmap: boundary-repair rows, uncovered document rows,
    malformed/dense request placeholders, and a damaged ragged tiling can all
    produce the same bitmap and must continue to fail closed.
    """

    if type(q_row_count) is not int or q_row_count <= 0 or reused_count != 0:
        return ""
    if not requests:
        return ""
    expected_offset = 0
    for request in requests:
        if not isinstance(request, _CachedRequestGeometry):
            return ""
        if (
            not request.reusable
            or request.flat_row_start != expected_offset
            or request.row_count <= 0
            or len(request.logical_positions) != request.row_count
            or len(request.input_token_ids) != request.row_count
            or request.query_start < 0
            or request.dirty_request_rows != tuple(range(request.row_count))
            or request.restore_bindings
            or not request.segment_metadata
            or any(
                int(position) < int(request.query_start)
                for position in request.logical_positions
            )
        ):
            return ""
        expected_offset += request.row_count
    if expected_offset != q_row_count:
        return ""
    return _QUERY_SUFFIX_FULL_LOCAL_REASON


class CompositeForwardGeometry:
    """Layer-independent, forward-scoped composite row geometry.

    The first reusable layer owns every CPU conversion and row construction.
    Later layers retain only their layer-specific persistent artifact views.
    Mutable certificates are installed exactly once and revalidated by tensor
    identity/version on every cache hit.
    """

    def __init__(
        self,
        *,
        source: CompositeGeometrySourceBinding,
        validated_plans: Tuple[object, ...],
        positions_cpu: object,
        input_ids_cpu: object,
        scheduler_totals: Tuple[Optional[int], ...],
        scheduler_extents: Tuple[Optional[int], ...],
        requests: Tuple[_CachedRequestGeometry, ...],
        restore_rows: Tuple[object, ...],
        reusable_cpu: object,
        dirty_rows_cpu: object,
        dirty_rows: Tuple[int, ...],
        reused_count: int,
        reuse_digest: object,
        input_layout_digest: object,
        batch_token: str,
        geometry_digest: str,
        request_token_digests: Tuple[str, ...],
        shared_generation_probe: Tuple[Tuple[object, ...], ...],
        forward_id: str,
        benchmark_request_id: str,
        benchmark_forward_id: str,
        benchmark_forward_mode: str,
        expected_layer_ids: Tuple[int, ...],
        intentional_full_local_reason: str = "",
        diagnostic_ablation: str = MLA_OFF_DIAGNOSTIC_ABLATION_FULL,
    ) -> None:
        self.source = source
        self.validated_plans = tuple(
            _freeze_plan_payload(plan) for plan in validated_plans
        )
        self.positions_cpu = positions_cpu
        self.input_ids_cpu = input_ids_cpu
        self.scheduler_totals = scheduler_totals
        self.scheduler_extents = scheduler_extents
        self.requests = requests
        self.restore_rows = restore_rows
        self.reusable_cpu = reusable_cpu
        self.dirty_rows_cpu = dirty_rows_cpu
        self.dirty_rows = tuple(int(value) for value in dirty_rows)
        self.reused_count = int(reused_count)
        self.reuse_digest = reuse_digest
        self.input_layout_digest = input_layout_digest
        self.batch_token = str(batch_token)
        self.geometry_digest = str(geometry_digest)
        self.request_token_digests = request_token_digests
        self.shared_generation_probe = shared_generation_probe
        self.forward_id = str(forward_id)
        self.benchmark_request_id = str(benchmark_request_id)
        self.benchmark_forward_id = str(benchmark_forward_id)
        self.benchmark_forward_mode = str(benchmark_forward_mode)
        self.expected_layer_ids = tuple(int(value) for value in expected_layer_ids)
        self.intentional_full_local_reason = str(intentional_full_local_reason)
        self.diagnostic_ablation = str(diagnostic_ablation)
        if self.diagnostic_ablation not in MLA_OFF_DIAGNOSTIC_ABLATIONS:
            raise ValueError("composite geometry has an invalid diagnostic mode")
        self.layout_certificate = None
        self.dirty_device_certificate = None
        self.dirty_device_indices = None
        self._token_rows_binding_identities = None
        self._batched_plan_template = None
        self._shared_restore_plan_templates = None
        self._shared_restore_plan_template_identities = None
        self._control_identities = self._capture_control_identities()
        self.validate_cached()

    @property
    def forward_key(self) -> str:
        return _canonical_digest(
            {
                "forward_id": self.forward_id,
                "geometry_digest": self.geometry_digest,
                "plan_digest": self.source.plan_digest,
                "request_token_digests": self.request_token_digests,
                "reuse_digest": self.reuse_digest,
                "shared_generations": self.shared_generation_probe,
            }
        )

    @property
    def is_intentional_full_local(self) -> bool:
        return (
            self.intentional_full_local_reason
            == _QUERY_SUFFIX_FULL_LOCAL_REASON
        )

    def _capture_control_identities(self) -> Tuple[object, ...]:
        row_tensors = []
        for request in self.requests:
            for binding in request.restore_bindings:
                rows = binding.rows
                row_tensors.extend(
                    (
                        _tensor_identity(rows.output_rows_cpu),
                        _tensor_identity(rows.local_positions_cpu),
                        _tensor_identity(binding.token_ids_cpu),
                    )
                )
        return (
            _tensor_identity(self.positions_cpu),
            _tensor_identity(self.input_ids_cpu),
            _tensor_identity(self.reusable_cpu),
            _tensor_identity(self.dirty_rows_cpu),
            tuple(row_tensors),
        )

    def validate_cached(self) -> None:
        if not isinstance(self.source, CompositeGeometrySourceBinding):
            raise TypeError("composite geometry has no source certificate")
        if (
            not self.expected_layer_ids
            or self.expected_layer_ids != tuple(sorted(set(self.expected_layer_ids)))
        ):
            raise ValueError("composite geometry layer domain is invalid")
        if len(self.requests) != self.source.batch_size:
            raise ValueError("composite geometry request count changed")
        if (
            len(self.scheduler_totals) != len(self.requests)
            or len(self.scheduler_extents) != len(self.requests)
        ):
            raise ValueError("composite scheduler lengths do not cover requests")
        for request, total, extent, plan in zip(
            self.requests,
            self.scheduler_totals,
            self.scheduler_extents,
            self.validated_plans,
        ):
            if not request.reusable:
                continue
            if (
                type(total) is not int
                or total <= 0
                or type(extent) is not int
                or extent <= 0
                or extent > total
                or not request.logical_positions
                or not isinstance(plan, Mapping)
                or not _request_extent_matches_geometry(
                    request, plan, extent
                )
            ):
                raise ValueError(
                    "composite active request scheduler extent changed"
                )
        if self.reused_count < 0:
            raise ValueError("composite geometry reusable row count is invalid")
        if self.reused_count == 0:
            reason = _intentional_full_local_reason(
                self.requests,
                q_row_count=int(self.source.q_rows),
                reused_count=self.reused_count,
            )
            if (
                reason != _QUERY_SUFFIX_FULL_LOCAL_REASON
                or self.intentional_full_local_reason != reason
                or self.restore_rows
                or self.dirty_rows != tuple(range(int(self.source.q_rows)))
            ):
                raise ValueError(
                    "zero-reuse composite geometry is not a trusted query suffix"
                )
        elif self.intentional_full_local_reason:
            raise ValueError(
                "reusable composite geometry cannot request a dense bypass"
            )
        if (
            self.dirty_rows != tuple(sorted(set(self.dirty_rows)))
            or self.reused_count + len(self.dirty_rows) != self.source.q_rows
        ):
            raise ValueError("composite geometry dirty rows are invalid")
        if self._control_identities != self._capture_control_identities():
            raise ValueError("composite cached CPU control tensor changed")
        if not _is_sha_digest(self.geometry_digest):
            raise ValueError("composite geometry digest is invalid")
        if any(not _is_sha_digest(value) for value in self.request_token_digests):
            raise ValueError("composite request token digest is invalid")
        if self.dirty_device_indices is not None:
            identity = getattr(self, "_dirty_device_identity", None)
            if identity != _tensor_identity(self.dirty_device_indices):
                raise ValueError("composite cached dirty device indices changed")
        if self._shared_restore_plan_templates is not None:
            identities = tuple(
                None if plan is None else id(plan)
                for plan in self._shared_restore_plan_templates
            )
            if identities != self._shared_restore_plan_template_identities:
                raise ValueError("cached shared restore plan identity changed")

    def install_layout_certificate(self, certificate: object) -> None:
        if self.layout_certificate is not None:
            raise RuntimeError("composite restore layout was installed twice")
        self.layout_certificate = certificate

    def install_dirty_device_indices(
        self, *, certificate: object, indices: object
    ) -> None:
        if self.dirty_device_certificate is not None or self.dirty_device_indices is not None:
            raise RuntimeError("composite dirty device indices were installed twice")
        self.dirty_device_certificate = certificate
        self.dirty_device_indices = indices
        self._dirty_device_identity = _tensor_identity(indices)

    def validate_or_install_token_rows_bindings(
        self,
        controller: object,
        bindings: Sequence[Tuple[object, object, object]],
    ) -> None:
        """Validate token values once, then recheck immutable identities.

        Each item is ``(restore_view, restore_rows, token_ids_cpu)``.  The
        certificate is layer-independent because every layer entry lives in
        the same committed segment generation and consumes the same token-row
        mapping.  A changed segment, epoch, tensor storage/version, or row
        object fails closed before a layer context is registered.
        """

        items = tuple(bindings)
        if not items:
            raise ValueError("token-row binding certificate cannot be empty")
        current = []
        validate_values = self._token_rows_binding_identities is None
        for view, rows, token_ids in items:
            if validate_values:
                controller.validate_view_token_rows(
                    view,
                    local_positions=rows.local_positions_cpu,
                    token_ids=token_ids,
                )
            current.append(
                controller.token_rows_binding_identity(
                    view,
                    local_positions=rows.local_positions_cpu,
                    token_ids=token_ids,
                )
            )
        current_tuple = tuple(current)
        if self._token_rows_binding_identities is None:
            self._token_rows_binding_identities = current_tuple
        elif current_tuple != self._token_rows_binding_identities:
            raise ValueError("cached MLA-off token-row artifact identity changed")

    def install_batched_plan_template(self, plan: object) -> None:
        """Seal the first fully validated layer plan as row geometry authority."""

        if self._batched_plan_template is not None:
            raise RuntimeError("batched plan template was installed twice")
        validate = getattr(plan, "validate", None)
        if not callable(validate):
            raise TypeError("batched plan template has no live validator")
        validate()
        if _batched_geometry_digest(plan) != self.geometry_digest:
            raise ValueError("batched plan template differs from forward geometry")
        if tuple(plan.local_dirty_rows) != self.dirty_rows:
            raise ValueError("batched plan template changed cached dirty rows")
        self._batched_plan_template = plan

    def install_shared_restore_plan_templates(
        self, plans: Tuple[Optional[SharedLatentRestorePlan], ...]
    ) -> None:
        """Seal content-bound CPU restore plans for identical RAG chunks.

        These plans contain no request id, GPU target, workspace, or mutable
        cache slot.  A template hit still obtains a fresh epoch pin, compiles a
        forward-id-bound device schedule, validates live artifact generations,
        and participates in the ordinary TP prepare/final protocol.
        """

        if self._shared_restore_plan_templates is not None:
            raise RuntimeError("shared restore plan templates were installed twice")
        normalized = tuple(plans)
        if len(normalized) != len(self.requests):
            raise ValueError("shared restore plan templates do not cover requests")
        for request, plan in zip(self.requests, normalized):
            if request.reusable:
                if not isinstance(plan, SharedLatentRestorePlan):
                    raise TypeError("reusable request lacks a shared restore plan")
                if (
                    plan.positions != request.logical_positions
                    or int(plan.query_start) != int(request.query_start)
                    or tuple(int(row) for row in plan.dirty_output_rows)
                    != tuple(int(row) for row in request.dirty_request_rows)
                ):
                    raise ValueError(
                        "shared restore plan differs from cached request geometry"
                    )
            elif plan is not None:
                raise ValueError("dense request cannot install a shared restore plan")
        self._shared_restore_plan_templates = normalized
        self._shared_restore_plan_template_identities = tuple(
            None if plan is None else id(plan) for plan in normalized
        )

    @property
    def shared_restore_plan_templates(
        self,
    ) -> Optional[Tuple[Optional[SharedLatentRestorePlan], ...]]:
        templates = self._shared_restore_plan_templates
        if templates is None:
            return None
        identities = tuple(
            None if plan is None else id(plan) for plan in templates
        )
        if identities != self._shared_restore_plan_template_identities:
            raise ValueError("shared restore plan template identity changed")
        return templates

    @property
    def batched_plan_template(self) -> object:
        template = self._batched_plan_template
        if template is None:
            return None
        validate = getattr(template, "validate", None)
        if not callable(validate):
            raise TypeError("batched plan template lost its live validator")
        validate()
        if tuple(template.local_dirty_rows) != self.dirty_rows:
            raise ValueError("batched plan template dirty rows changed")
        return template


def _close_shared_request_states(
    states: object,
) -> Optional[BaseException]:
    """Close every pin in a partially constructed forward lease.

    Construction failure must not strand an epoch pin.  This helper is
    deliberately tolerant of malformed constructor input and always attempts
    every close before returning the first close error to an explicit owner.
    """

    try:
        items = tuple(states)  # type: ignore[arg-type]
    except BaseException:
        return None
    first_error: Optional[BaseException] = None
    for state in items:
        close = getattr(state, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error


class CompositeForwardResources:
    """Forward-owned lease for continuously batched shared GPU artifacts.

    ``ForwardBatch`` instances may be recycled by the scheduler.  The lease's
    key therefore binds tokens, positions, ragged geometry, artifact epochs,
    and policy.  Replacing that key closes every old request pin before a new
    forward can observe the object.  Layer contexts and commit builders are
    registered exactly once and become invalid as soon as the lease closes.
    """

    def __init__(
        self,
        *,
        forward_key: str,
        forward_id: str,
        batch_digest: str,
        total_rows: int,
        request_token_digests: Tuple[str, ...],
        shared_states: Tuple[SharedRequestRestoreState, ...],
        expected_layer_ids: Tuple[int, ...],
        geometry: Optional[CompositeForwardGeometry] = None,
    ) -> None:
        # Take ownership before validating constructor arguments.  A caller
        # hands this object live artifact pins, so *any* constructor failure
        # must close them even when the failure precedes ``self.validate()``.
        self.shared_states = shared_states
        self.geometry = geometry
        self._contexts: Dict[int, object] = {}
        self._commit_builders: Dict[int, object] = {}
        self._forward_commit_coordinator: Optional[object] = None
        self._dirty_state_slot_certificates: Dict[int, object] = {}
        self._ragged_geometry_template: Optional[RaggedBatchGeometry] = None
        self._ragged_geometry_template_identity: Optional[Tuple[object, ...]] = None
        self._shared_latent_template: Optional[SharedLatentBinding] = None
        self._shared_latent_template_identity: Optional[Tuple[object, ...]] = None
        self._completed_layers: set[int] = set()
        self._closed = False
        try:
            if not _is_sha_digest(forward_key):
                raise ValueError("composite forward key must be a SHA-256 digest")
            if not forward_id:
                raise ValueError("composite forward id must be non-empty")
            if not _is_sha_digest(batch_digest):
                raise ValueError("ragged batch digest must be a SHA-256 digest")
            if type(total_rows) is not int or total_rows <= 0:
                raise ValueError("composite forward row count is invalid")
            if type(request_token_digests) is not tuple or any(
                not _is_sha_digest(value) for value in request_token_digests
            ):
                raise ValueError("request token digests are invalid")
            if type(shared_states) is not tuple or not shared_states:
                raise ValueError("composite forward needs request restore states")
            layers = tuple(int(value) for value in expected_layer_ids)
            if not layers or layers != tuple(sorted(set(layers))):
                raise ValueError(
                    "composite reusable layers must be sorted and unique"
                )
            self.forward_key = forward_key
            self.forward_id = str(forward_id)
            self.batch_digest = batch_digest
            self.total_rows = total_rows
            self.request_token_digests = request_token_digests
            self.expected_layer_ids = layers
            if geometry is not None:
                if not isinstance(geometry, CompositeForwardGeometry):
                    raise TypeError("composite forward geometry has a foreign type")
                geometry.validate_cached()
                if (
                    geometry.forward_key != forward_key
                    or geometry.forward_id != str(forward_id)
                    or geometry.geometry_digest != batch_digest
                    or geometry.source.q_rows != total_rows
                    or geometry.request_token_digests != request_token_digests
                    or geometry.expected_layer_ids != layers
                ):
                    raise ValueError("composite forward geometry differs from its lease")
            self.validate()
        except Exception:
            _close_shared_request_states(shared_states)
            self._contexts.clear()
            self._commit_builders.clear()
            self._forward_commit_coordinator = None
            self._dirty_state_slot_certificates.clear()
            self._ragged_geometry_template = None
            self._ragged_geometry_template_identity = None
            self._shared_latent_template = None
            self._shared_latent_template_identity = None
            self._closed = True
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_layer_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._contexts))

    @property
    def committed_builder_layer_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._commit_builders))

    @property
    def forward_commit_coordinator(self) -> Optional[object]:
        """Return the forward-wide coordinator, if the static plan installed it."""

        return self._forward_commit_coordinator

    def install_forward_commit_coordinator(self, coordinator: object) -> None:
        """Bind one all-layer prepare/final state machine to this lease."""

        self.validate()
        if getattr(coordinator, "resources", None) is not self:
            raise ValueError("forward commit coordinator owns another lease")
        proposal = getattr(coordinator, "proposal", None)
        if not isinstance(proposal, CompositeForwardProposal):
            raise TypeError("forward commit coordinator has no valid proposal")
        if proposal.reusable_layer_ids != self.expected_layer_ids:
            raise ValueError("forward commit coordinator has another layer domain")
        if proposal.ragged.total_rows != self.total_rows:
            raise ValueError("forward commit coordinator has another row domain")
        if self._forward_commit_coordinator is None:
            self._forward_commit_coordinator = coordinator
        elif self._forward_commit_coordinator is not coordinator:
            raise RuntimeError("forward commit coordinator was replaced")

    def dirty_state_slot_certificate(
        self, compress_ratio: int
    ) -> Optional[object]:
        """Return this forward's sole physical-slot proof for one ratio."""

        self.validate()
        ratio = int(compress_ratio)
        if ratio not in (4, 128):
            raise ValueError("dirty-state certificate ratio must be C4 or C128")
        return self._dirty_state_slot_certificates.get(ratio)

    def bind_dirty_state_slot_certificate(
        self, compress_ratio: int, certificate: object
    ) -> None:
        """Install exactly once; never replace a forward/ratio certificate."""

        self.validate()
        ratio = int(compress_ratio)
        if ratio not in (4, 128):
            raise ValueError("dirty-state certificate ratio must be C4 or C128")
        if (
            getattr(certificate, "compress_ratio", None) != ratio
            or getattr(certificate, "forward_token", None) != self.forward_id
            or getattr(certificate, "q_rows", None) != self.total_rows
        ):
            raise ValueError("dirty-state certificate differs from its forward lease")
        existing = self._dirty_state_slot_certificates.get(ratio)
        if existing is None:
            self._dirty_state_slot_certificates[ratio] = certificate
        elif existing is not certificate:
            raise RuntimeError("dirty-state certificate was replaced within a forward")

    def validate(
        self,
        *,
        forward_key: Optional[str] = None,
        batch_digest: Optional[str] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("composite forward resource lease is closed")
        if forward_key is not None and str(forward_key) != self.forward_key:
            raise ValueError("composite resource lease belongs to another forward")
        if batch_digest is not None and str(batch_digest) != self.batch_digest:
            raise ValueError("composite resource lease binds another ragged batch")
        if self.geometry is not None:
            self.geometry.validate_cached()
            if (
                self.geometry.forward_key != self.forward_key
                or self.geometry.forward_id != self.forward_id
                or self.geometry.geometry_digest != self.batch_digest
                or self.geometry.source.q_rows != self.total_rows
                or self.geometry.request_token_digests
                != self.request_token_digests
                or self.geometry.expected_layer_ids != self.expected_layer_ids
            ):
                raise ValueError("composite forward geometry lease binding changed")
        offset = 0
        dirty_rows = []
        for index, state in enumerate(self.shared_states):
            if not isinstance(state, SharedRequestRestoreState):
                raise TypeError("composite lease contains an invalid request state")
            state.validate()
            if state.request_index != index or state.flat_row_offset != offset:
                raise ValueError("shared request states no longer tile the batch")
            if state.reusable and str(
                getattr(state.schedule, "forward_id", "")
            ) != f"{self.forward_id}:request:{index}":
                raise ValueError("shared request schedule belongs to another forward")
            dirty_rows.extend(state.dirty_rows)
            offset += state.row_count
        if offset != self.total_rows:
            raise ValueError("shared request states do not span the forward")
        if len(self.request_token_digests) != len(self.shared_states):
            raise ValueError("request token digests do not cover the ragged batch")
        if tuple(dirty_rows) != tuple(sorted(set(dirty_rows))):
            raise ValueError("shared request dirty-row domains overlap")
        for ratio, certificate in self._dirty_state_slot_certificates.items():
            if (
                ratio not in (4, 128)
                or getattr(certificate, "compress_ratio", None) != ratio
                or getattr(certificate, "forward_token", None) != self.forward_id
                or getattr(certificate, "q_rows", None) != self.total_rows
            ):
                raise ValueError("composite dirty-state certificate binding changed")
        coordinator = self._forward_commit_coordinator
        if coordinator is not None:
            if (
                getattr(coordinator, "resources", None) is not self
                or getattr(getattr(coordinator, "proposal", None), "reusable_layer_ids", ())
                != self.expected_layer_ids
            ):
                raise ValueError("forward commit coordinator binding changed")

    def register_context(self, layer_id: int, context: object) -> None:
        self.validate()
        layer_id = int(layer_id)
        if layer_id not in self.expected_layer_ids:
            raise ValueError("context layer is outside the reusable layer set")
        if layer_id in self._completed_layers:
            raise RuntimeError("a completed layer context cannot be reopened")
        if layer_id in self._contexts:
            raise RuntimeError("composite restore context was prepared twice")
        self._contexts[layer_id] = context

    def register_commit_builder(self, layer_id: int, builder: object) -> None:
        self.validate()
        layer_id = int(layer_id)
        if self._contexts.get(layer_id) is not getattr(builder, "context", None):
            raise ValueError("commit builder does not own the active layer context")
        if layer_id in self._commit_builders:
            raise RuntimeError("composite TP commit builder exists for this layer")
        self._commit_builders[layer_id] = builder

    def complete_context(self, context: object) -> bool:
        if self._closed:
            raise RuntimeError("composite forward resource lease is closed")
        layer_id = int(getattr(context, "layer_id", -1))
        if self._contexts.get(layer_id) is not context:
            raise ValueError("cannot release a foreign composite context")
        del self._contexts[layer_id]
        self._commit_builders.pop(layer_id, None)
        self._completed_layers.add(layer_id)
        return tuple(sorted(self._completed_layers)) == self.expected_layer_ids

    def close(self) -> None:
        if self._closed:
            return
        coordinator = self._forward_commit_coordinator
        ledger = getattr(coordinator, "ledger", None)
        if (
            bool(getattr(coordinator, "committed", False))
            and ledger is not None
            and not bool(getattr(ledger, "final_attempted", False))
        ):
            raise RuntimeError(
                "committed forward resources cannot close before the fixed "
                "forward-final rendezvous"
            )
        first_error = _close_shared_request_states(self.shared_states)
        self._contexts.clear()
        self._commit_builders.clear()
        self._forward_commit_coordinator = None
        self._dirty_state_slot_certificates.clear()
        self._ragged_geometry_template = None
        self._ragged_geometry_template_identity = None
        self._shared_latent_template = None
        self._shared_latent_template_identity = None
        self._closed = True
        if first_error is not None:
            raise first_error


@dataclass(frozen=True)
class LayerCacheDomainPreflight:
    """One already-built cache domain, before any omission is consumed."""

    component: str
    total_units: int
    restored_units: int
    dirty_units: int
    artifact_digest: str
    gpu_view: object
    builder_preflight_token: str
    omitted_slot_consumed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("cache component must be non-empty")
        for name in ("total_units", "restored_units", "dirty_units"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.total_units <= 0 or min(self.restored_units, self.dirty_units) < 0:
            raise ValueError("cache preflight unit counts are invalid")
        if self.restored_units + self.dirty_units != self.total_units:
            raise ValueError("cache preflight units do not partition total units")
        if not isinstance(self.artifact_digest, str) or not self.artifact_digest:
            raise ValueError("cache artifact digest must be non-empty")
        if not isinstance(self.builder_preflight_token, str) or not self.builder_preflight_token:
            raise ValueError("cache builder preflight token must be non-empty")
        if type(self.omitted_slot_consumed) is not bool:
            raise TypeError("omitted_slot_consumed must be boolean")
        if self.omitted_slot_consumed:
            raise RuntimeError("cache omission was consumed before composite commit")


@dataclass(frozen=True)
class LayerCompositeCommitResult:
    """The result of one layer's single composite TP vote."""

    proposal: CompositeForwardProposal
    outcome: Optional[CommitOutcome]
    sparse_q_authorization: Optional[SparseQInstallAuthorization]
    omission_authorization: Optional[OmissionAuthorization]
    packed_sparse_q: object
    irreversible: bool = False
    postcommit_error: Optional[BaseException] = field(
        default=None, repr=False, compare=False
    )

    @property
    def committed(self) -> bool:
        return bool(
            self.irreversible
            or (self.outcome is not None and self.outcome.committed)
        )


def merge_layer_composite_proposals(
    resources: CompositeForwardResources,
    layer_proposals: Sequence[CompositeForwardProposal],
    *,
    forward_ordinal: int,
    omission_profile: str = OMISSION_PROFILE_FULL,
    restore_batch_common_digest: Optional[str] = None,
    restore_batch_local_digest: Optional[str] = None,
    failure_carrier_view: Optional[GpuViewBinding] = None,
) -> CompositeForwardProposal:
    """Seal preallocated one-layer reservations into one forward proposal.

    This function performs no tensor work and no collective.  Production may
    call it only after a forward arena has reserved every layer's post-write Q,
    cache and z_off views.  In particular, all z_off bindings must name one
    persistent forward arena; silently merging per-layer temporary arenas
    would make the resulting certificate meaningless.
    """

    if not isinstance(resources, CompositeForwardResources):
        raise TypeError("proposal merge requires composite forward resources")
    resources.validate()
    if type(forward_ordinal) is not int or forward_ordinal < 0:
        raise ValueError("forward_ordinal must be a non-negative integer")
    if omission_profile not in (
        OMISSION_PROFILE_FULL,
        OMISSION_PROFILE_ZOFF_ONLY,
        OMISSION_PROFILE_SHARED_ONLY,
    ):
        raise ValueError("forward omission profile is invalid")
    proposals = tuple(layer_proposals)
    if not proposals or any(
        not isinstance(item, CompositeForwardProposal) for item in proposals
    ):
        raise TypeError("layer proposals must contain composite proposals")
    if any(len(item.reusable_layer_ids) != 1 for item in proposals):
        raise ValueError("proposal merge accepts one reserved layer per item")
    if any(item.omission_profile != omission_profile for item in proposals):
        raise ValueError(
            "layer proposal omission profiles differ from the aggregate"
        )
    proposals = tuple(sorted(proposals, key=lambda item: item.reusable_layer_ids[0]))
    layers = tuple(item.reusable_layer_ids[0] for item in proposals)
    if layers != resources.expected_layer_ids:
        raise ValueError("layer proposals do not cover the forward exactly")
    first = proposals[0]
    identity_common = (
        first.identity.model_hash,
        first.identity.policy_hash,
        first.identity.tp_size,
        first.tp_rank,
    )
    shared_common = (
        first.ragged.digest,
        first.shared_latent.spec_digest,
        first.shared_latent.restore_plan_digest,
        first.shared_latent.artifacts,
        first.shared_latent.clean_rows,
        first.shared_latent.dirty_rows,
        first.shared_latent.clean_rows_digest,
        first.shared_latent.dirty_rows_digest,
        first.shared_latent.boundary_tokens,
        first.persistent_zoff_arena_token,
        first.fused_merge_kernel_token,
        first.restore_provider_token,
        first.restore_provider_local_token,
        first.restore_batch_common_digest,
        first.restore_batch_local_digest,
        first.failure_carrier_view,
    )
    for item in proposals[1:]:
        if (
            item.identity.model_hash,
            item.identity.policy_hash,
            item.identity.tp_size,
            item.tp_rank,
        ) != identity_common:
            raise ValueError("layer proposals disagree on model/policy/TP identity")
        if (
            item.ragged.digest,
            item.shared_latent.spec_digest,
            item.shared_latent.restore_plan_digest,
            item.shared_latent.artifacts,
            item.shared_latent.clean_rows,
            item.shared_latent.dirty_rows,
            item.shared_latent.clean_rows_digest,
            item.shared_latent.dirty_rows_digest,
            item.shared_latent.boundary_tokens,
            item.persistent_zoff_arena_token,
            item.fused_merge_kernel_token,
            item.restore_provider_token,
            item.restore_provider_local_token,
            item.restore_batch_common_digest,
            item.restore_batch_local_digest,
            item.failure_carrier_view,
        ) != shared_common:
            raise ValueError(
                "layer proposals disagree on ragged/shared/persistent arena identity"
            )
    shared = SharedLatentBinding(
        spec_digest=first.shared_latent.spec_digest,
        restore_plan_digest=first.shared_latent.restore_plan_digest,
        layer_compression=tuple(
            item.shared_latent.layer_compression[0] for item in proposals
        ),
        artifacts=first.shared_latent.artifacts,
        clean_rows=first.shared_latent.clean_rows,
        dirty_rows=first.shared_latent.dirty_rows,
        clean_rows_digest=first.shared_latent.clean_rows_digest,
        dirty_rows_digest=first.shared_latent.dirty_rows_digest,
        boundary_tokens=first.shared_latent.boundary_tokens,
    )
    layer_plan_digests = tuple(
        (item.reusable_layer_ids[0], item.rank_local_batch_plan_digest)
        for item in proposals
    )
    rank_local_batch_plan_digest = (
        layer_plan_digests[0][1]
        if len(layer_plan_digests) == 1
        else _canonical_digest(
            {
                "schema": "redknot-forward-layer-plan-set-v1",
                "layers": layer_plan_digests,
            }
        )
    )
    if omission_profile == OMISSION_PROFILE_SHARED_ONLY:
        if any(item.sparse_q or item.z_off_views for item in proposals):
            raise ValueError(
                "shared-only layer proposal retained Q/z_off reservations"
            )
        sparse_q = ()
        z_off_views = ()
        sequential_q_arena = None
    else:
        if any(
            len(item.sparse_q) != 1 or len(item.z_off_views) != 1
            for item in proposals
        ):
            raise ValueError(
                "head-split layer proposal lacks Q/z_off reservations"
            )
        sparse_q = tuple(item.sparse_q[0] for item in proposals)
        z_off_views = tuple(item.z_off_views[0] for item in proposals)
        sequential_q_arena = build_sequential_q_arena_binding(sparse_q)
    return CompositeForwardProposal(
        identity=ForwardIdentity(
            generation_id=resources.forward_id,
            forward_ordinal=forward_ordinal,
            model_hash=first.identity.model_hash,
            policy_hash=first.identity.policy_hash,
            tp_size=first.identity.tp_size,
        ),
        tp_rank=first.tp_rank,
        ragged=first.ragged,
        shared_latent=shared,
        sparse_q=sparse_q,
        cache_domains=tuple(
            sorted(
                (
                    domain
                    for item in proposals
                    for domain in item.cache_domains
                ),
                key=lambda domain: domain.key,
            )
        ),
        z_off_views=z_off_views,
        rank_local_batch_plan_digest=rank_local_batch_plan_digest,
        persistent_zoff_arena_token=first.persistent_zoff_arena_token,
        fused_merge_kernel_token=first.fused_merge_kernel_token,
        restore_provider_token=first.restore_provider_token,
        restore_provider_local_token=first.restore_provider_local_token,
        restore_batch_common_digest=_as_sha_digest(
            restore_batch_common_digest
            or first.restore_batch_common_digest
        ),
        restore_batch_local_digest=_as_sha_digest(
            restore_batch_local_digest
            or first.restore_batch_local_digest
        ),
        failure_carrier_view=(
            failure_carrier_view or first.failure_carrier_view
        ),
        sequential_q_arena=sequential_q_arena,
        omission_profile=omission_profile,
        commit_scope=COMMIT_SCOPE_FORWARD_RESERVED,
    )


@dataclass(frozen=True)
class _PreparedFullLayerReceiptAnchor:
    """Static objects sealed by the forward prepare for one live Q write."""

    context: object = field(repr=False, compare=False)
    q_arena: object = field(repr=False, compare=False)
    q_reservation: object = field(repr=False, compare=False)
    cache_domains: Tuple[LayerCacheDomainPreflight, ...] = field(
        repr=False, compare=False
    )
    cache_tensor_identities: Tuple[Tuple[object, ...], ...]
    persistent_projection_plan: object = field(repr=False, compare=False)
    persistent_projection_identity: object = field(repr=False, compare=False)
    observed_reservation: LayerReservationBinding
    execution_token: str


class ForwardCompositeCommitCoordinator:
    """One prepare certificate and one final rendezvous for all middle layers.

    This coordinator is deliberately separate from the compatibility
    :class:`LayerCompositeCommitBuilder`.  The caller first installs an
    all-layer :class:`CompositeForwardProposal` whose sparse-Q views are
    *reserved post-write slots*.  ``commit`` performs the forward's only
    prepare/readiness collective.  Each sequential layer then calls
    :meth:`record_layer_builder` after writing and revalidating its exact GPU
    view/version; this creates a local receipt without a TP collective.

    A post-certificate failure is sticky.  Callers must carry shape-only work
    through the remaining layers and invoke :meth:`finalize` exactly once so
    every TP rank reaches the fixed fail-stop rendezvous.
    """

    def __init__(
        self,
        *,
        resources: CompositeForwardResources,
        proposal: CompositeForwardProposal,
        builder_epoch_token: str,
    ) -> None:
        if not isinstance(resources, CompositeForwardResources):
            raise TypeError("forward coordinator requires composite resources")
        resources.validate()
        if not isinstance(proposal, CompositeForwardProposal):
            raise TypeError("forward coordinator proposal has an invalid type")
        if not isinstance(builder_epoch_token, str) or not builder_epoch_token:
            raise ValueError("forward builder_epoch_token must be non-empty")
        if proposal.commit_scope != COMMIT_SCOPE_FORWARD_RESERVED:
            raise ValueError("forward coordinator requires reserved commit scope")
        if proposal.reusable_layer_ids != resources.expected_layer_ids:
            raise ValueError("forward proposal does not cover every reusable layer")
        if proposal.ragged.total_rows != resources.total_rows:
            raise ValueError("forward proposal row count differs from its lease")
        if proposal.identity.generation_id != resources.forward_id:
            raise ValueError("forward proposal generation differs from its lease")
        geometry = resources.geometry
        if geometry is not None:
            geometry.validate_cached()
            if geometry.is_intentional_full_local:
                raise ValueError(
                    "query-suffix/full-local forwards cannot receive omission "
                    "certificates"
                )
            expected_profile = {
                "full": OMISSION_PROFILE_FULL,
                "zoff_only": OMISSION_PROFILE_ZOFF_ONLY,
                "shared_only": OMISSION_PROFILE_SHARED_ONLY,
            }.get(str(geometry.diagnostic_ablation))
            if expected_profile is None or proposal.omission_profile != expected_profile:
                raise ValueError(
                    "forward omission profile differs from diagnostic ablation"
                )
        self.resources = resources
        self.proposal = proposal
        self.builder_epoch_token = builder_epoch_token
        self.session = ForwardCommitSession(proposal)
        self.preflight: Optional[ForwardPreparePreflight] = None
        self.outcome: Optional[CommitOutcome] = None
        self.authorization: Optional[OmissionAuthorization] = None
        self.ledger: Optional[ForwardExecutionLedger] = None
        self.local_preflight_error = ""
        self._commit_attempted = False
        self._bound_contexts: Dict[int, object] = {}
        self._prepared_full_layer_anchors: Dict[
            int, _PreparedFullLayerReceiptAnchor
        ] = {}
        resources.install_forward_commit_coordinator(self)

    @property
    def committed(self) -> bool:
        return bool(self.outcome is not None and self.outcome.committed)

    @property
    def failed(self) -> bool:
        return bool(self.ledger is not None and self.ledger.failed)

    def commit(
        self,
        adapter: object,
        *,
        ready: bool = True,
        failure_code: str = "",
        local_preflight_error: str = "",
    ) -> CommitOutcome:
        """Perform the sole all-layer prepare vote, even on local failure."""

        if self._commit_attempted:
            raise RuntimeError("forward composite prepare may be attempted once")
        self._commit_attempted = True
        error = ""
        try:
            if type(ready) is not bool:
                raise TypeError("ready must be boolean")
            if not isinstance(local_preflight_error, str):
                raise TypeError("local_preflight_error must be a string")
            if ready and failure_code:
                raise ValueError("ready forward prepare cannot carry failure_code")
            if not ready and not failure_code:
                raise ValueError("rejected forward prepare requires failure_code")
            if local_preflight_error:
                raise RuntimeError(local_preflight_error)
            self.resources.validate()
            preflight = build_forward_prepare_preflight(
                self.proposal,
                builder_epoch_token=self.builder_epoch_token,
                ready=ready,
                failure_code=failure_code,
                omitted_slots_consumed=False,
            )
            self.session.record_forward_prepare_preflight(preflight)
            self.preflight = preflight
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.local_preflight_error = error
        outcome = self.session.commit(
            adapter,
            local_preexchange_error=error,
        )
        self.outcome = outcome
        if outcome.committed:
            try:
                certificate = outcome.certificate
                assert certificate is not None
                authorization = self.session.authorize_reserved_omissions(
                    certificate
                )
                self.authorization = authorization
                self.ledger = ForwardExecutionLedger(
                    self.session, authorization
                )
            except BaseException as exc:
                # The TP decision is already accepted.  There is no safe
                # dense fallback and, without a ledger, no final receipt
                # rendezvous can be constructed.  Invoke the adapter's real
                # out-of-band fail-stop hook immediately.
                self.session.fail_closed_after_commit(
                    adapter,
                    reason_code="forward_ledger_install_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
        return outcome

    def bind_context(self, context: MLAOffRuntimeContext) -> None:
        """Publish only the forward certificate; Q stays receipt-gated."""

        if not self.committed or self.ledger is None or self.authorization is None:
            raise RuntimeError("forward prepare has no committed certificate")
        layer_id = int(context.layer_id)
        self.resources.validate()
        if self.resources._contexts.get(layer_id) is not context:
            raise ValueError("forward coordinator received a foreign layer context")
        if layer_id in self._bound_contexts:
            if self._bound_contexts[layer_id] is context:
                if (
                    getattr(context, "composite_irreversible", False) is not True
                    or getattr(context, "composite_commit_session", None)
                    is not self.session
                    or getattr(context, "composite_certificate", None)
                    is not self.session.certificate
                    or getattr(context, "composite_omission_authorization", None)
                    is not self.authorization
                ):
                    raise RuntimeError(
                        "forward coordinator context binding changed"
                    )
                return
            raise RuntimeError("forward coordinator layer context was replaced")
        if (
            getattr(context, "composite_irreversible", False) is not False
            or getattr(context, "composite_commit_session", None) is not None
            or getattr(context, "composite_certificate", None) is not None
            or getattr(context, "composite_omission_authorization", None)
            is not None
            or getattr(context, "composite_layer_execution_receipt", None)
            is not None
            or getattr(context, "sparse_q_committed", False) is not False
        ):
            raise RuntimeError(
                "forward coordinator requires a pristine uncommitted context"
            )
        context.composite_irreversible = True
        context.composite_commit_session = self.session
        context.composite_certificate = self.session.certificate
        context.composite_omission_authorization = self.authorization
        self._bound_contexts[layer_id] = context

    def record_layer_builder(
        self,
        builder: "LayerCompositeCommitBuilder",
        *,
        execution_token: Optional[str] = None,
    ) -> Optional[LayerExecutionReceipt]:
        """Revalidate a compatibility builder and issue no-collective receipt."""

        if self.ledger is None:
            raise RuntimeError("forward prepare is not committed")
        if not isinstance(builder, LayerCompositeCommitBuilder):
            raise TypeError("layer builder has an invalid type")
        layer_id = builder.layer_id
        if builder.resources is not self.resources:
            raise ValueError("layer builder belongs to another forward lease")
        try:
            self.resources.validate()
            if self.resources.forward_commit_coordinator is not self:
                raise RuntimeError("forward commit coordinator binding changed")
            builder._validate_precommit(
                committed_forward_session=self.session,
                committed_forward_authorization=self.authorization,
            )
            observed = build_layer_reservation_binding(
                builder.proposal, layer_id
            )
            derived_execution_token = _canonical_digest(
                {
                    "forward_id": self.resources.forward_id,
                    "layer_id": layer_id,
                    "reservation_digest": observed.digest,
                    "builder_epoch_token": builder.builder_epoch_token,
                    "projection_token": str(
                        getattr(builder.packed_sparse_q, "projection_token", "")
                    ),
                    "cache_builder_tokens": tuple(
                        item.builder_preflight_token
                        for item in builder.proposal.cache_domains
                    ),
                    "zoff_artifact_digest": builder.proposal.z_off_views[
                        0
                    ].artifact_digest,
                }
            )
            if execution_token is not None and (
                str(execution_token) != derived_execution_token
            ):
                raise ValueError(
                    "caller execution token differs from the validated layer receipt"
                )
        except BaseException as exc:
            self.ledger.record_failure(
                layer_id=layer_id,
                stage="layer_live_preflight",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return None
        receipt = self.ledger.record_layer_execution(
            layer_id=layer_id,
            observed_reservation=observed,
            execution_token=derived_execution_token,
        )
        if receipt is not None:
            try:
                self.bind_context(builder.context)
                builder.context.composite_layer_execution_receipt = receipt
            except BaseException as exc:
                self.ledger.record_failure(
                    layer_id=layer_id,
                    stage="layer_receipt_publish",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return None
        return receipt

    def register_prepared_full_layers(
        self,
        *,
        contexts: Mapping[int, MLAOffRuntimeContext],
        q_arena: object,
        q_reservations: Mapping[int, object],
        prepared_layers: Mapping[int, object],
    ) -> None:
        """Seal the already-certified static inputs for O(1) live receipts."""

        if self.ledger is None or not self.committed:
            raise RuntimeError("forward prepare is not committed")
        if self.proposal.omission_profile not in (
            OMISSION_PROFILE_FULL,
            OMISSION_PROFILE_ZOFF_ONLY,
        ):
            raise RuntimeError(
                "prepared head-split anchors require full or zoff_only profile"
            )
        if self._prepared_full_layer_anchors:
            raise RuntimeError("prepared full-layer anchors were installed twice")
        self.resources.validate()
        layer_ids = self.proposal.reusable_layer_ids
        if (
            tuple(sorted(int(key) for key in contexts)) != layer_ids
            or tuple(sorted(int(key) for key in q_reservations)) != layer_ids
            or tuple(sorted(int(key) for key in prepared_layers)) != layer_ids
        ):
            raise ValueError("prepared full-layer anchors do not cover the forward")
        sparse_by_layer = {item.layer_id: item for item in self.proposal.sparse_q}
        zoff_by_layer = {item.layer_id: item for item in self.proposal.z_off_views}
        if tuple(sorted(sparse_by_layer)) != layer_ids or tuple(
            sorted(zoff_by_layer)
        ) != layer_ids:
            raise RuntimeError("forward proposal lacks full-layer Q/z_off anchors")

        anchors: Dict[int, _PreparedFullLayerReceiptAnchor] = {}
        for layer_id in layer_ids:
            context = contexts[layer_id]
            if (
                self.resources._contexts.get(layer_id) is not context
                or self._bound_contexts.get(layer_id) is not context
                or getattr(context, "composite_commit_session", None)
                is not self.session
                or getattr(context, "composite_certificate", None)
                is not self.session.certificate
                or getattr(context, "composite_omission_authorization", None)
                is not self.authorization
            ):
                raise RuntimeError("prepared full-layer context binding changed")
            reservation = q_reservations[layer_id]
            reservation_for = getattr(q_arena, "reservation_for", None)
            if not callable(reservation_for) or reservation_for(layer_id) is not reservation:
                raise RuntimeError("prepared full-layer Q reservation changed")
            sparse = sparse_by_layer[layer_id]
            if (
                getattr(reservation, "plan", None) is None
                or int(getattr(reservation.plan, "layer_id", -1)) != layer_id
                or str(getattr(reservation, "projection_token", ""))
                != str(sparse.projection_token)
            ):
                raise ValueError("prepared full-layer Q anchor differs from proposal")
            prepared = prepared_layers[layer_id]
            cache_domains = getattr(prepared, "cache_preflights", None)
            if type(cache_domains) is not tuple:
                raise TypeError("prepared head-split cache anchors are invalid")
            if (
                self.proposal.omission_profile == OMISSION_PROFILE_FULL
                and not cache_domains
            ):
                raise TypeError("prepared full-layer cache anchors are absent")
            if (
                self.proposal.omission_profile == OMISSION_PROFILE_ZOFF_ONLY
                and cache_domains
            ):
                raise TypeError(
                    "prepared zoff-only layer unexpectedly retained cache anchors"
                )
            persistent_plan = getattr(context, "persistent_projection_plan", None)
            persistent_identity = getattr(
                context, "_persistent_projection_plan_identity", None
            )
            if persistent_plan is None or persistent_identity is None:
                raise RuntimeError("prepared full-layer z_off anchor is absent")
            observed = self.ledger.expected_layer_reservation(layer_id)
            execution_token = _canonical_digest(
                {
                    "forward_id": self.resources.forward_id,
                    "layer_id": layer_id,
                    "reservation_digest": observed.digest,
                    "builder_epoch_token": str(prepared.builder_epoch_token),
                    "projection_token": str(reservation.projection_token),
                    "cache_builder_tokens": tuple(
                        item.builder_preflight_token for item in cache_domains
                    ),
                    "zoff_artifact_digest": zoff_by_layer[
                        layer_id
                    ].artifact_digest,
                }
            )
            anchors[layer_id] = _PreparedFullLayerReceiptAnchor(
                context=context,
                q_arena=q_arena,
                q_reservation=reservation,
                cache_domains=cache_domains,
                cache_tensor_identities=_cache_preflight_live_identities(
                    cache_domains
                ),
                persistent_projection_plan=persistent_plan,
                persistent_projection_identity=persistent_identity,
                observed_reservation=observed,
                execution_token=execution_token,
            )
        self._prepared_full_layer_anchors = anchors

    def record_sealed_full_layer(
        self,
        *,
        context: MLAOffRuntimeContext,
        packed_sparse_q: object,
        cache_domains: Sequence[LayerCacheDomainPreflight],
    ) -> Optional[LayerExecutionReceipt]:
        """Validate only the live sequential-Q write; reuse static anchors."""

        if self.ledger is None:
            raise RuntimeError("forward prepare is not committed")
        layer_id = int(context.layer_id)
        try:
            anchor = self._prepared_full_layer_anchors.get(layer_id)
            if anchor is None:
                raise RuntimeError("prepared full-layer anchor is absent")
            if (
                self.resources.closed
                or self.resources.forward_commit_coordinator is not self
                or self.resources._contexts.get(layer_id) is not context
                or anchor.context is not context
                or self._bound_contexts.get(layer_id) is not context
                or getattr(context, "composite_irreversible", False) is not True
                or getattr(context, "composite_commit_session", None)
                is not self.session
                or getattr(context, "composite_certificate", None)
                is not self.session.certificate
                or getattr(context, "composite_omission_authorization", None)
                is not self.authorization
            ):
                raise RuntimeError("sealed full-layer context binding changed")
            if bool(getattr(context, "sparse_q_committed", False)) or getattr(
                context, "composite_layer_execution_receipt", None
            ) is not None:
                raise RuntimeError("sealed full-layer omission was already installed")
            cache_domains = tuple(cache_domains)
            if cache_domains is not anchor.cache_domains or (
                _cache_preflight_live_identities(cache_domains)
                != anchor.cache_tensor_identities
            ):
                raise ValueError("sealed full-layer cache binding changed")
            if (
                getattr(context, "persistent_projection_plan", None)
                is not anchor.persistent_projection_plan
                or getattr(context, "_persistent_projection_plan_identity", None)
                is not anchor.persistent_projection_identity
            ):
                raise ValueError("sealed full-layer z_off binding changed")
            validate_projection = getattr(
                anchor.q_arena, "validate_completed_projection", None
            )
            if not callable(validate_projection):
                raise TypeError("sequential packed-Q arena has no completion fence")
            if validate_projection(
                layer_id=layer_id,
                projection=packed_sparse_q,
                local_rows=context.online_local_row_indices,
            ) is not anchor.q_reservation:
                raise ValueError("sealed full-layer Q reservation changed")
        except BaseException as exc:
            self.ledger.record_failure(
                layer_id=layer_id,
                stage="sealed_full_layer_live_preflight",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return None

        receipt = self.ledger.record_layer_execution(
            layer_id=layer_id,
            observed_reservation=anchor.observed_reservation,
            execution_token=anchor.execution_token,
        )
        if receipt is not None:
            try:
                # Registration already bound and sealed this exact context;
                # repeating bind_context() here would revalidate the complete
                # forward lease once per layer.
                context.composite_layer_execution_receipt = receipt
            except BaseException as exc:
                self.ledger.record_failure(
                    layer_id=layer_id,
                    stage="sealed_full_layer_receipt_publish",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return None
        return receipt

    def record_prepared_full_layer(
        self,
        *,
        context: MLAOffRuntimeContext,
        packed_sparse_q: object,
        cache_domains: Sequence[LayerCacheDomainPreflight],
        builder_epoch_token: str,
    ) -> Optional[LayerExecutionReceipt]:
        """Issue a full-profile receipt from the forward-wide reservation.

        The forward prepare proposal already binds the complete ragged batch,
        shared-latent schedule, persistent z_off views, and every cache target.
        Rebuilding a compatibility ``LayerCompositeCommitBuilder`` here used to
        revalidate and re-hash the same 8K immutable row geometry once per
        layer.  On TP=8 that host work serialized the CUDA stream and dominated
        TTFT even though the actual restore had already completed.

        This fast path does not trust the earlier proposal for the one value
        that did not exist at prepare time: the live packed sparse-Q write.  It
        reconstructs the live sparse-Q, cache, and z_off bindings from their
        current tensors, requires exact equality with the reserved bindings,
        and only then publishes the ordinary layer execution receipt.  Thus it
        removes duplicate semantic work without weakening the post-write
        identity/version checks or the fixed final rendezvous.
        """

        if self.ledger is None:
            raise RuntimeError("forward prepare is not committed")
        if self.proposal.omission_profile not in (
            OMISSION_PROFILE_FULL,
            OMISSION_PROFILE_ZOFF_ONLY,
        ):
            raise RuntimeError(
                "prepared head-split receipt requires full or zoff_only profile"
            )
        if not isinstance(builder_epoch_token, str) or not builder_epoch_token:
            raise ValueError("full-layer builder epoch token is empty")
        layer_id = int(context.layer_id)
        cache_domains = tuple(cache_domains)
        try:
            self.resources.validate()
            if self.resources.forward_commit_coordinator is not self:
                raise RuntimeError("forward commit coordinator binding changed")
            if self.resources._contexts.get(layer_id) is not context:
                raise ValueError("full-layer context belongs to another forward")
            if (
                getattr(self, "_bound_contexts", {}).get(layer_id) is not context
                or getattr(context, "composite_irreversible", False) is not True
                or getattr(context, "composite_commit_session", None)
                is not self.session
                or getattr(context, "composite_certificate", None)
                is not self.session.certificate
                or getattr(context, "composite_omission_authorization", None)
                is not self.authorization
            ):
                raise RuntimeError("full-layer context binding changed")
            if bool(getattr(context, "sparse_q_committed", False)) or getattr(
                context, "composite_layer_execution_receipt", None
            ) is not None:
                raise RuntimeError("full-layer omission was already installed")

            expected_sparse = tuple(
                item
                for item in self.proposal.sparse_q
                if int(item.layer_id) == layer_id
            )
            expected_caches = tuple(
                item
                for item in self.proposal.cache_domains
                if int(item.layer_id) == layer_id
            )
            expected_zoff = tuple(
                item
                for item in self.proposal.z_off_views
                if int(item.layer_id) == layer_id
            )
            expected_cache_presence = bool(
                self.proposal.omission_profile == OMISSION_PROFILE_FULL
            )
            if (
                len(expected_sparse) != 1
                or bool(expected_caches) != expected_cache_presence
                or len(expected_zoff) != 1
            ):
                raise RuntimeError(
                    "forward proposal lacks its profile-specific layer reservation"
                )

            live_sparse = _build_sparse_q_binding(context, packed_sparse_q)
            compression_ratio = _layer_compression_ratio(
                self.resources, layer_id
            )
            if self.proposal.omission_profile == OMISSION_PROFILE_ZOFF_ONLY:
                if cache_domains:
                    raise ValueError(
                        "zoff-only live receipt retained cache domains"
                    )
                live_caches = ()
            else:
                live_caches, _ = _build_cache_domain_bindings(
                    context=context,
                    preflights=cache_domains,
                    compression_ratio=compression_ratio,
                    restore_plan_digest=(
                        self.proposal.shared_latent.restore_plan_digest
                    ),
                    expected_device_index=(
                        live_sparse.packed_projection_view.device_index
                    ),
                )
            live_zoff, _ = _build_zoff_binding(
                context,
                persistent_arena_token=(
                    self.proposal.persistent_zoff_arena_token
                ),
                fused_merge_kernel_token=(
                    self.proposal.fused_merge_kernel_token
                ),
            )
            if (
                (live_sparse,) != expected_sparse
                or tuple(live_caches) != expected_caches
                or (live_zoff,) != expected_zoff
            ):
                raise ValueError("live full-layer bindings differ from reservation")

            observed = build_layer_reservation_binding(
                self.proposal, layer_id
            )
            execution_token = _canonical_digest(
                {
                    "forward_id": self.resources.forward_id,
                    "layer_id": layer_id,
                    "reservation_digest": observed.digest,
                    "builder_epoch_token": builder_epoch_token,
                    "projection_token": str(
                        getattr(packed_sparse_q, "projection_token", "")
                    ),
                    "cache_builder_tokens": tuple(
                        item.builder_preflight_token
                        for item in cache_domains
                    ),
                    "zoff_artifact_digest": live_zoff.artifact_digest,
                }
            )
        except BaseException as exc:
            self.ledger.record_failure(
                layer_id=layer_id,
                stage="prepared_full_layer_live_preflight",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return None

        receipt = self.ledger.record_layer_execution(
            layer_id=layer_id,
            observed_reservation=observed,
            execution_token=execution_token,
        )
        if receipt is not None:
            try:
                self.bind_context(context)
                context.composite_layer_execution_receipt = receipt
            except BaseException as exc:
                self.ledger.record_failure(
                    layer_id=layer_id,
                    stage="prepared_full_layer_receipt_publish",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return None
        return receipt

    def record_shared_only_layer(
        self,
        *,
        context: MLAOffRuntimeContext,
        cache_domains: Sequence[LayerCacheDomainPreflight],
        builder_epoch_token: str,
    ) -> Optional[LayerExecutionReceipt]:
        """Issue a cache-only receipt without fabricating sparse-Q work.

        ``shared_only`` deliberately evaluates complete Q/attention/wo_a and
        consumes only restored clean cache rows.  Reusing the compatibility
        builder here would require a live packed-Q projection and z_off view
        that this profile neither computes nor authorizes.  Rebuild the exact
        cache bindings from their live tensors instead, then compare them with
        the forward reservation before issuing the ordinary layer receipt.
        """

        if self.ledger is None:
            raise RuntimeError("forward prepare is not committed")
        if self.proposal.omission_profile != OMISSION_PROFILE_SHARED_ONLY:
            raise RuntimeError("cache-only receipt requires shared_only profile")
        if not isinstance(builder_epoch_token, str) or not builder_epoch_token:
            raise ValueError("cache-only builder epoch token is empty")
        layer_id = int(context.layer_id)
        try:
            self.resources.validate()
            if self.resources._contexts.get(layer_id) is not context:
                raise ValueError("cache-only context belongs to another forward")
            expected_caches = tuple(
                item
                for item in self.proposal.cache_domains
                if item.layer_id == layer_id
            )
            if not expected_caches:
                raise ValueError("cache-only layer has no reserved domains")
            compression_ratio = _layer_compression_ratio(
                self.resources, layer_id
            )
            observed_caches, _ = _build_cache_domain_bindings(
                context=context,
                preflights=cache_domains,
                compression_ratio=compression_ratio,
                restore_plan_digest=self.proposal.shared_latent.restore_plan_digest,
                expected_device_index=expected_caches[0].gpu_view.device_index,
            )
            if observed_caches != expected_caches:
                raise ValueError("cache-only live GPU bindings changed")
            expected = build_layer_reservation_binding(
                self.proposal, layer_id
            )
            observed = LayerReservationBinding(
                layer_id=layer_id,
                sparse_q_binding_digest=expected.sparse_q_binding_digest,
                cache_binding_digest=expected.cache_binding_digest,
                zoff_binding_digest=expected.zoff_binding_digest,
                gpu_view_versions=tuple(
                    sorted(
                        (
                            f"cache:{item.component}",
                            item.gpu_view.version,
                        )
                        for item in observed_caches
                    )
                ),
            )
            execution_token = _canonical_digest(
                {
                    "forward_id": self.resources.forward_id,
                    "layer_id": layer_id,
                    "reservation_digest": observed.digest,
                    "builder_epoch_token": builder_epoch_token,
                    "profile": OMISSION_PROFILE_SHARED_ONLY,
                }
            )
        except BaseException as exc:
            self.ledger.record_failure(
                layer_id=layer_id,
                stage="shared_only_live_preflight",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return None
        receipt = self.ledger.record_layer_execution(
            layer_id=layer_id,
            observed_reservation=observed,
            execution_token=execution_token,
        )
        if receipt is not None:
            try:
                self.bind_context(context)
                context.composite_layer_execution_receipt = receipt
            except BaseException as exc:
                self.ledger.record_failure(
                    layer_id=layer_id,
                    stage="shared_only_receipt_publish",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return None
        return receipt

    def consume_omitted_slot(
        self,
        *,
        context: MLAOffRuntimeContext,
        slot: str,
    ) -> bool:
        if self.ledger is None:
            raise RuntimeError("forward prepare is not committed")
        layer_id = int(context.layer_id)
        receipt = getattr(context, "composite_layer_execution_receipt", None)
        if not isinstance(receipt, LayerExecutionReceipt):
            self.ledger.record_failure(
                layer_id=layer_id,
                stage="missing_layer_receipt",
                detail="omission reached a layer without an execution receipt",
            )
            return False
        return self.ledger.consume_omitted_slot(
            layer_id=layer_id,
            receipt=receipt,
            slot=slot,
        )

    def consume_layer_omitted_slots(
        self,
        *,
        context: MLAOffRuntimeContext,
    ) -> bool:
        """Consume the complete prepared omission set with one O(1) fence."""

        if self.ledger is None:
            raise RuntimeError("forward prepare is not committed")
        layer_id = int(context.layer_id)
        receipt = getattr(context, "composite_layer_execution_receipt", None)
        if not isinstance(receipt, LayerExecutionReceipt):
            self.ledger.record_failure(
                layer_id=layer_id,
                stage="missing_layer_receipt",
                detail="omission reached a layer without an execution receipt",
            )
            return False
        return self.ledger.consume_layer_omitted_slots(
            layer_id=layer_id,
            receipt=receipt,
        )

    def record_pipeline_failure(
        self,
        *,
        layer_id: int,
        stage: str,
        error: BaseException,
    ) -> None:
        if self.ledger is None:
            raise RuntimeError("cannot carry failure before forward certification")
        self.ledger.record_failure(
            layer_id=int(layer_id),
            stage=str(stage),
            detail=f"{type(error).__name__}: {error}",
        )

    def finalize(self, adapter: object):
        if self.ledger is None:
            raise RuntimeError("cannot finalize an uncommitted forward")
        return self.ledger.finalize(adapter)


def _is_sha_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    if len(value) != 71:
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


def _as_sha_digest(value: object) -> str:
    """Normalize external stable identifiers into the commit digest domain."""

    if not isinstance(value, str) or not value:
        raise ValueError("stable digest source must be a non-empty string")
    text = value
    if _is_sha_digest(text):
        return text
    if len(text) == 64:
        try:
            int(text, 16)
        except ValueError:
            pass
        else:
            return "sha256:" + text.lower()
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = None
    if _orjson is not None:
        try:
            candidate = _orjson.dumps(payload, option=_orjson.OPT_SORT_KEYS)
        except (TypeError, ValueError, OverflowError):
            candidate = None
        # All hot-path control payloads are immutable ASCII/int/bool/None
        # trees.  Keep the historical ensure_ascii=True byte contract for any
        # future Unicode or unsupported value by falling back to stdlib JSON.
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
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _composite_serving_forward_id(
    *,
    plan_digest: str,
    geometry_digest: str,
    request_token_digests: Sequence[str],
    scheduler_totals: Sequence[Optional[int]],
    scheduler_extents: Sequence[Optional[int]],
    benchmark_forward_mode: str,
    q_row_count: int,
) -> str:
    """Return the TP-common semantic generation for one serving forward.

    ``CompositeGeometrySourceBinding.forward_generation`` intentionally does
    not participate: it contains rank-local Python object identity and exists
    only to reject stale reuse of a local ``ForwardBatch`` cache slot.
    """

    token_digests = tuple(str(value) for value in request_token_digests)
    if not _is_sha_digest(plan_digest) or not _is_sha_digest(geometry_digest):
        raise ValueError("serving forward identity requires semantic digests")
    if not token_digests or any(
        not _is_sha_digest(value) for value in token_digests
    ):
        raise ValueError("serving forward token digests are invalid")
    if type(q_row_count) is not int or q_row_count <= 0:
        raise ValueError("serving forward row count is invalid")
    return "composite-forward-v1:" + _canonical_digest(
        {
            "benchmark_forward_mode": str(benchmark_forward_mode),
            "geometry_digest": str(geometry_digest),
            "plan_digest": str(plan_digest),
            "q_rows": int(q_row_count),
            "request_token_digests": list(token_digests),
            "scheduler_totals": [
                None if value is None else int(value)
                for value in scheduler_totals
            ],
            "scheduler_extents": [
                None if value is None else int(value)
                for value in scheduler_extents
            ],
        }
    )


def _batched_geometry_digest(batch_plan: object) -> str:
    """Layer-independent identity of one flattened scheduler batch."""

    return _canonical_digest(
        {
            "q_rows": int(batch_plan.q_rows),
            "global_rows": list(int(row) for row in batch_plan.global_rows),
            "local_clean_rows": list(
                int(row) for row in batch_plan.local_clean_rows
            ),
            "local_dirty_rows": list(
                int(row) for row in batch_plan.local_dirty_rows
            ),
            "requests": [
                {
                    "request_index": int(request.request_index),
                    "request_token": str(request.request_token),
                    "flat_row_start": int(request.flat_row_start),
                    "logical_positions": list(
                        int(value) for value in request.logical_positions
                    ),
                    "query_start": int(request.query_start),
                    "dirty_request_rows": list(
                        int(value) for value in request.dirty_request_rows
                    ),
                }
                for request in tuple(batch_plan.requests)
            ],
        }
    )


def _tensor_identity(value: object) -> Tuple[object, ...]:
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    stride_method = getattr(value, "stride", None)
    strides = (
        tuple(int(item) for item in stride_method())
        if callable(stride_method)
        else tuple(int(item) for item in getattr(value, "strides", ()))
    )
    data_ptr_method = getattr(value, "data_ptr", None)
    data_ptr = int(data_ptr_method()) if callable(data_ptr_method) else 0
    storage_offset_method = getattr(value, "storage_offset", None)
    storage_offset = (
        int(storage_offset_method()) if callable(storage_offset_method) else 0
    )
    try:
        version: object = int(getattr(value, "_version", 0))
    except RuntimeError:
        version = "inference-immutable"
    return (
        id(value),
        data_ptr,
        shape,
        strides,
        str(getattr(value, "device", "")),
        str(getattr(value, "dtype", "")),
        storage_offset,
        version,
    )


def _gpu_view_from_tensor(
    value: object,
    *,
    storage_role: str,
    view_role: str,
) -> GpuViewBinding:
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    stride_method = getattr(value, "stride", None)
    strides = (
        tuple(int(item) for item in stride_method())
        if callable(stride_method)
        else tuple(int(item) for item in getattr(value, "strides", ()))
    )
    if not shape or len(shape) != len(strides) or any(item <= 0 for item in shape):
        raise ValueError("GPU commit view has invalid tensor geometry")
    device = getattr(value, "device", None)
    if getattr(device, "type", None) != "cuda":
        raise ValueError("composite commit views must remain CUDA resident")
    device_index = getattr(device, "index", None)
    if device_index is None:
        device_index = 0
    numel_method = getattr(value, "numel", None)
    element_size_method = getattr(value, "element_size", None)
    if not callable(numel_method) or not callable(element_size_method):
        raise TypeError("GPU commit view must expose tensor byte geometry")
    nbytes = int(numel_method()) * int(element_size_method())
    if nbytes <= 0:
        raise ValueError("GPU commit view cannot be empty")
    storage_offset_method = getattr(value, "storage_offset", None)
    byte_offset = (
        int(storage_offset_method()) * int(element_size_method())
        if callable(storage_offset_method)
        else 0
    )
    data_ptr_method = getattr(value, "data_ptr", None)
    data_ptr = int(data_ptr_method()) if callable(data_ptr_method) else id(value)
    identity_digest = _canonical_digest(
        [storage_role, view_role, list(_tensor_identity(value))]
    )
    try:
        version = int(getattr(value, "_version", 0))
    except RuntimeError:
        # Inference tensors intentionally hide their version counter.  Their
        # immutable identity is already captured by ``_tensor_identity``;
        # use the protocol's stable zero sentinel in the integer ABI.
        version = 0
    return GpuViewBinding(
        storage_token=f"{storage_role}:{data_ptr}:{identity_digest}",
        view_token=f"{view_role}:{identity_digest}",
        device_index=int(device_index),
        dtype=str(getattr(value, "dtype", "unknown")),
        shape=shape,
        strides=strides,
        byte_offset=byte_offset,
        nbytes=nbytes,
        version=version,
    )


def _cache_preflight_live_identities(
    preflights: Sequence[LayerCacheDomainPreflight],
) -> Tuple[Tuple[object, ...], ...]:
    """Capture only the mutable live tensor identity of cache preflights."""

    result = []
    for item in tuple(preflights):
        if not isinstance(item, LayerCacheDomainPreflight):
            raise TypeError("cache preflight anchor has an invalid type")
        view = item.gpu_view
        if isinstance(view, GpuViewBinding):
            result.append(("sealed-binding", id(view)))
        else:
            result.append(("live-tensor", *_tensor_identity(view)))
    return tuple(result)


def _packed_sparse_q_identity(projection: object) -> Tuple[object, ...]:
    return (
        id(projection),
        id(getattr(projection, "plan", None)),
        str(getattr(projection, "digest", "")),
        str(getattr(projection, "projection_token", "")),
        _tensor_identity(getattr(projection, "values", None)),
        _tensor_identity(getattr(projection, "local_rows", None)),
    )


def _persistent_projection_identity(plan: object) -> Tuple[object, ...]:
    return (
        id(plan),
        str(getattr(plan, "digest", "")),
        int(getattr(plan, "total_rows", -1)),
        tuple(int(item) for item in getattr(plan, "tail_shape", ())),
        tuple(
            (
                id(view),
                str(view.seg_hash),
                int(view.layer_id),
                int(view.commit_epoch),
                str(view.generation_token),
                tuple(int(row) for row in view.geometry.output_rows),
                tuple(int(row) for row in view.geometry.local_rows),
                _tensor_identity(view.values),
            )
            for view in tuple(getattr(plan, "views", ()))
        ),
    )


def _expected_cache_components(compression_ratio: int) -> Tuple[str, ...]:
    if int(compression_ratio) == 4:
        return tuple(
            sorted(
                (
                    SWA,
                    C4,
                    INDEXER,
                    ATTENTION_COMPRESSOR_STATE,
                    INDEXER_COMPRESSOR_STATE,
                )
            )
        )
    if int(compression_ratio) == 128:
        return tuple(sorted((SWA, C128, ATTENTION_COMPRESSOR_STATE)))
    raise ValueError("composite reusable layer must use C4 or C128")


def _layer_compression_ratio(
    resources: CompositeForwardResources, layer_id: int
) -> int:
    ratios = []
    for state in resources.shared_states:
        try:
            layer = state.cpu_plan.layers[int(layer_id)]
        except (KeyError, TypeError) as exc:
            raise ValueError("shared restore plan lacks this reusable layer") from exc
        ratios.append(int(layer.compress_ratio))
    if len(set(ratios)) != 1 or ratios[0] not in (4, 128):
        raise ValueError("request restore plans disagree on layer compression")
    return ratios[0]


def _ragged_geometry_identity(ragged: RaggedBatchGeometry) -> Tuple[object, ...]:
    return (
        id(ragged),
        id(ragged.requests),
        int(ragged.total_rows),
        str(ragged.scheduler_epoch),
        tuple(
            (
                id(request),
                str(request.request_id),
                int(request.row_offset),
                int(request.row_count),
                int(request.document_rows),
                int(request.query_rows),
                int(request.position_begin),
                str(request.token_digest),
            )
            for request in ragged.requests
        ),
    )


def _shared_latent_template_identity(
    binding: SharedLatentBinding,
) -> Tuple[object, ...]:
    return (
        id(binding),
        str(binding.spec_digest),
        str(binding.restore_plan_digest),
        id(binding.artifacts),
        tuple(
            (
                str(item.pin_digest),
                int(item.commit_epoch),
                str(item.artifact_kind),
                str(item.storage_generation),
            )
            for item in binding.artifacts
        ),
        int(binding.clean_rows),
        int(binding.dirty_rows),
        str(binding.clean_rows_digest),
        str(binding.dirty_rows_digest),
        tuple(
            (int(item.layer_id), int(item.ratio))
            for item in binding.layer_compression
        ),
    )


def _build_ragged_geometry(
    context: MLAOffRuntimeContext,
    resources: CompositeForwardResources,
) -> RaggedBatchGeometry:
    cached = getattr(resources, "_ragged_geometry_template", None)
    if cached is not None:
        if not isinstance(cached, RaggedBatchGeometry):
            raise TypeError("ragged forward template has a foreign type")
        identity = _ragged_geometry_identity(cached)
        if identity != getattr(
            resources, "_ragged_geometry_template_identity", None
        ):
            raise ValueError("ragged forward template changed within the forward")
        return cached
    batch_plan = context.batched_reuse_plan
    validate = getattr(batch_plan, "validate", None)
    if not callable(validate):
        raise TypeError("context has no validated ragged reuse plan")
    validate()
    if _batched_geometry_digest(batch_plan) != resources.batch_digest:
        raise ValueError("context ragged geometry differs from forward resources")
    requests = []
    for layout, token_digest in zip(
        tuple(batch_plan.requests), resources.request_token_digests
    ):
        document_rows = sum(
            int(position) < int(layout.query_start)
            for position in layout.logical_positions
        )
        requests.append(
            RaggedRequestGeometry(
                request_id=f"{int(layout.request_index)}:{layout.request_token}",
                row_offset=int(layout.flat_row_start),
                row_count=int(layout.row_count),
                document_rows=int(document_rows),
                query_rows=int(layout.row_count) - int(document_rows),
                position_begin=int(layout.logical_positions[0]),
                token_digest=token_digest,
            )
        )
    ragged = RaggedBatchGeometry(
        requests=tuple(requests),
        total_rows=int(batch_plan.q_rows),
        # Shared TP payloads may only bind the rank-independent row geometry.
        # The full plan also contains rank-local artifact epochs and is carried
        # separately in CompositeForwardProposal.rank_local_payload().
        scheduler_epoch=(
            f"{resources.forward_id}:ragged-batch:{resources.batch_digest}"
        ),
    )
    if int(ragged.total_rows) != int(resources.total_rows):
        raise ValueError("ragged commit rows differ from forward resources")
    resources._ragged_geometry_template = ragged
    resources._ragged_geometry_template_identity = _ragged_geometry_identity(
        ragged
    )
    return ragged


def _build_shared_latent_binding(
    context: MLAOffRuntimeContext,
    resources: CompositeForwardResources,
    *,
    compression_ratio: int,
) -> SharedLatentBinding:
    layer_id = int(context.layer_id)
    batch_plan = context.batched_reuse_plan
    clean_rows = tuple(int(row) for row in batch_plan.local_clean_rows)
    dirty_rows = tuple(int(row) for row in batch_plan.local_dirty_rows)
    layer_compression = (
        LayerCompressionBinding(
            layer_id=layer_id,
            ratio=int(compression_ratio),
        ),
    )
    cached = getattr(resources, "_shared_latent_template", None)
    if cached is not None:
        if not isinstance(cached, SharedLatentBinding):
            raise TypeError("shared-latent forward template has a foreign type")
        if (
            _shared_latent_template_identity(cached)
            != getattr(resources, "_shared_latent_template_identity", None)
        ):
            raise ValueError(
                "shared-latent forward template changed within the forward"
            )
        if (
            cached.clean_rows != len(clean_rows)
            or cached.dirty_rows != len(dirty_rows)
            or batch_plan.local_clean_rows
            is not getattr(resources, "_shared_latent_clean_rows", None)
            or batch_plan.local_dirty_rows
            is not getattr(resources, "_shared_latent_dirty_rows", None)
        ):
            raise ValueError("shared-latent row geometry changed within the forward")
        return replace(cached, layer_compression=layer_compression)
    artifacts = []
    spec_payload = []
    restore_payload = [resources.batch_digest]
    for state in resources.shared_states:
        state.validate()
        if not state.reusable:
            spec_payload.append(
                [int(state.request_index), "dense-request", int(state.row_count)]
            )
            restore_payload.append(
                [
                    int(state.request_index),
                    int(state.flat_row_offset),
                    int(state.row_count),
                    "dense-request",
                    list(state.dirty_rows),
                ]
            )
            continue
        spec_payload.append(
            [
                int(state.request_index),
                str(state.schedule.layout_fingerprint),
                str(getattr(state.cpu_plan.spec, "model_hash", "")),
                str(getattr(state.cpu_plan.spec, "policy_hash", "")),
                int(getattr(state.cpu_plan.spec, "length", 0)),
            ]
        )
        restore_payload.append(
            [
                int(state.request_index),
                int(state.flat_row_offset),
                int(state.row_count),
                str(state.schedule.layout_fingerprint),
                list(int(value) for value in state.schedule.positions),
                int(state.schedule.query_start),
                # CPU/GPU commit epochs are local monotonic ABA guards.  A
                # publish followed by a TP-wide rollback can advance them on
                # only the ranks that reached publication, so they must not
                # enter the TP-common restore digest.  Bind the semantic
                # segment/token/spec identity here; the exact local epochs
                # remain in ArtifactGenerationBinding below and therefore in
                # the proposal's rank-local payload.
                [
                    [
                        str(seg_hash),
                        str(getattr(artifact, "token_hash", "")),
                        str(getattr(getattr(artifact, "spec", None), "model_hash", "")),
                        str(getattr(getattr(artifact, "spec", None), "policy_hash", "")),
                        int(getattr(getattr(artifact, "spec", None), "length", 0)),
                    ]
                    for seg_hash, artifact in sorted(
                        getattr(state.cpu_plan, "artifacts", {}).items()
                    )
                ]
                or [
                    [str(seg_hash), "", "", "", 0]
                    for seg_hash in sorted(state.schedule.artifact_epochs)
                ],
                [
                    [
                        str(operation.domain),
                        int(operation.layer_id),
                        int(operation.count),
                        str(operation.position_semantics),
                        list(
                            int(value)
                            for value in state.schedule.index_arena[
                                operation.output_rows.begin : operation.output_rows.end
                            ]
                        ),
                    ]
                    for operation in state.schedule.operations
                ],
                list(state.dirty_rows),
            ]
        )
        for seg_hash, mirror in sorted(state.pin.mirrors.items()):
            mirror_pin = _canonical_digest(
                [
                    str(seg_hash),
                    int(mirror.commit_epoch),
                    str(mirror.layout_fingerprint),
                ]
            )
            artifacts.append(
                ArtifactGenerationBinding(
                    pin_digest=mirror_pin,
                    commit_epoch=int(mirror.commit_epoch),
                    artifact_kind="shared_latent_segment",
                    storage_generation=(
                        f"{mirror.layout_fingerprint}:segment:{seg_hash}:"
                        f"epoch:{mirror.commit_epoch}"
                    ),
                )
            )
    result = SharedLatentBinding(
        spec_digest=_canonical_digest(spec_payload),
        restore_plan_digest=_canonical_digest(restore_payload),
        layer_compression=layer_compression,
        artifacts=tuple(sorted(set(artifacts))),
        clean_rows=len(clean_rows),
        dirty_rows=len(dirty_rows),
        clean_rows_digest=_canonical_digest(list(clean_rows)),
        dirty_rows_digest=_canonical_digest(list(dirty_rows)),
    )
    resources._shared_latent_template = result
    # BatchedReusePlan's live certificate binds these exact immutable tuples.
    # Layer rebinds preserve their object identity, so later layers can check
    # the certificate in O(1) instead of serializing 57K row ids twice again.
    resources._shared_latent_clean_rows = batch_plan.local_clean_rows
    resources._shared_latent_dirty_rows = batch_plan.local_dirty_rows
    resources._shared_latent_template_identity = (
        _shared_latent_template_identity(result)
    )
    return result


def _build_sparse_q_binding(
    context: MLAOffRuntimeContext,
    projection: object,
) -> SparseQBinding:
    if bool(getattr(context, "sparse_q_committed", False)):
        raise RuntimeError("sparse-Q was installed before composite commit")
    validate = getattr(projection, "validate", None)
    if not callable(validate):
        raise TypeError("composite sparse-Q must be a packed projection")
    validate()
    plan = getattr(projection, "plan", None)
    if plan is None:
        raise TypeError("packed sparse-Q projection has no plan")
    plan.validate()
    if int(plan.layer_id) != int(context.layer_id):
        raise ValueError("packed sparse-Q belongs to another layer")
    if tuple(int(axis) for axis in plan.local_head_axes) != tuple(
        int(axis) for axis in context.local_head_axes
    ):
        raise ValueError("packed sparse-Q local heads differ from MLA restore")
    expected_rows = tuple(
        int(row) for row in context.batched_reuse_plan.local_dirty_rows
    )
    if tuple(int(row) for row in plan.online_local_rows) != expected_rows:
        raise ValueError("packed sparse-Q dirty rows differ from MLA restore")
    if getattr(projection, "local_rows", None) is not context.online_local_row_indices:
        raise ValueError("packed sparse-Q must use the context dirty-row tensor")
    values = projection.values
    view = _gpu_view_from_tensor(
        values,
        storage_role=f"packed-sparse-q-layer-{int(context.layer_id)}",
        view_role=str(projection.projection_token),
    )
    arena_token = getattr(projection, "arena_token", None)
    arena_index = int(getattr(projection, "arena_index", 0))
    write_ordinal = int(getattr(projection, "write_ordinal", 0))
    write_count = int(getattr(projection, "write_count", 1))
    version_offset = int(
        getattr(projection, "version_offset", max(1, view.version))
    )
    arena_capacity_nbytes = int(
        getattr(projection, "arena_capacity_nbytes", view.nbytes)
    )
    if arena_token is not None:
        if not isinstance(arena_token, str) or not arena_token:
            raise ValueError("packed sparse-Q arena token is malformed")
        arena_base_version = int(
            getattr(projection, "arena_base_version", -1)
        )
        if arena_base_version < 0:
            raise ValueError("packed sparse-Q arena base version is absent")
        if view.version != arena_base_version + version_offset:
            raise ValueError("packed sparse-Q arena post-write version changed")
        view = replace(
            view,
            storage_token=arena_token,
            view_token=str(projection.projection_token),
        )
    return SparseQBinding(
        layer_id=int(plan.layer_id),
        tp_rank=int(plan.tp_rank),
        tp_size=int(plan.tp_size),
        plan_digest=_as_sha_digest(plan.digest),
        projection_token=str(projection.projection_token),
        q_rows=int(plan.q_rows),
        owned_head_count=int(plan.owned_head_count),
        head_dim=int(plan.head_dim),
        projected_head_rows=int(plan.projected_head_rows),
        omitted_head_rows=int(plan.omitted_head_rows),
        packed_projection_view=view,
        arena_index=arena_index,
        write_ordinal=write_ordinal,
        write_count=write_count,
        version_offset=version_offset,
        arena_capacity_nbytes=arena_capacity_nbytes,
    )


def _build_sparse_q_reservation_binding(
    context: MLAOffRuntimeContext,
    reservation: object,
) -> SparseQBinding:
    """Bind a future sequential-arena write without claiming Q is computed."""

    plan = getattr(reservation, "plan", None)
    validate_plan = getattr(plan, "validate", None)
    if not callable(validate_plan):
        raise TypeError("sparse-Q reservation has no validated plan")
    validate_plan()
    if int(plan.layer_id) != int(context.layer_id):
        raise ValueError("sparse-Q reservation belongs to another layer")
    if tuple(int(axis) for axis in plan.local_head_axes) != tuple(
        int(axis) for axis in context.local_head_axes
    ):
        raise ValueError("sparse-Q reservation local heads differ from restore")
    if tuple(int(row) for row in plan.online_local_rows) != tuple(
        int(row) for row in context.batched_reuse_plan.local_dirty_rows
    ):
        raise ValueError("sparse-Q reservation dirty rows differ from restore")
    values = getattr(reservation, "values", None)
    view = _gpu_view_from_tensor(
        values,
        storage_role=f"packed-sparse-q-reservation-{int(context.layer_id)}",
        view_role=str(getattr(reservation, "projection_token", "")),
    )
    arena_token = str(getattr(reservation, "arena_token", ""))
    arena_base_version = int(getattr(reservation, "arena_base_version", -1))
    version_offset = int(getattr(reservation, "version_offset", -1))
    expected_version = arena_base_version + version_offset
    if not arena_token or arena_base_version < 0 or version_offset <= 0:
        raise ValueError("sparse-Q reservation lacks an arena/version schedule")
    if int(getattr(reservation, "write_count", 0)) <= 0:
        raise ValueError("sparse-Q reservation has no writes")
    if tuple(view.shape) != (
        int(plan.projected_head_rows),
        int(plan.head_dim),
    ):
        raise ValueError("sparse-Q reservation view has another packed geometry")
    # The tensor is intentionally still at its arena base version.  The
    # forward proposal binds the future post-write version; record_layer_builder
    # later compares the live projection against this exact reservation.
    if view.version != arena_base_version:
        raise ValueError("sparse-Q reservation arena was written before prepare")
    view = replace(
        view,
        storage_token=arena_token,
        view_token=str(getattr(reservation, "projection_token", "")),
        version=expected_version,
    )
    return SparseQBinding(
        layer_id=int(plan.layer_id),
        tp_rank=int(plan.tp_rank),
        tp_size=int(plan.tp_size),
        plan_digest=_as_sha_digest(plan.digest),
        projection_token=str(getattr(reservation, "projection_token", "")),
        q_rows=int(plan.q_rows),
        owned_head_count=int(plan.owned_head_count),
        head_dim=int(plan.head_dim),
        projected_head_rows=int(plan.projected_head_rows),
        omitted_head_rows=int(plan.omitted_head_rows),
        packed_projection_view=view,
        arena_index=int(getattr(reservation, "arena_index", -1)),
        write_ordinal=int(getattr(reservation, "write_ordinal", -1)),
        write_count=int(getattr(reservation, "write_count", 0)),
        version_offset=version_offset,
        arena_capacity_nbytes=int(
            getattr(reservation, "arena_capacity_nbytes", 0)
        ),
    )


def _build_zoff_binding(
    context: MLAOffRuntimeContext,
    *,
    persistent_arena_token: str,
    fused_merge_kernel_token: str,
) -> Tuple[ZOffGpuViewBinding, Tuple[object, ...]]:
    plan = context.validate_persistent_projection_commit()
    plan_identity = _persistent_projection_identity(plan)
    values = tuple(view.values for view in plan.views)
    if not values:
        raise ValueError("persistent z_off plan has no resident views")
    devices = {str(value.device) for value in values}
    dtypes = {str(value.dtype) for value in values}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("persistent z_off views disagree on device/dtype")
    first_device = values[0].device
    if getattr(first_device, "type", None) != "cuda":
        raise ValueError("persistent z_off views must remain CUDA resident")
    device_index = getattr(first_device, "index", None)
    if device_index is None:
        device_index = 0
    nbytes = 0
    versions = []
    for value in values:
        nbytes += int(value.numel()) * int(value.element_size())
        versions.append(int(getattr(value, "_version", 0)))
    groups, rank = (int(item) for item in plan.tail_shape)
    logical_view = GpuViewBinding(
        storage_token=(
            f"persistent-z-storage:{_canonical_digest([repr(plan_identity)])}"
        ),
        view_token=f"persistent-z-plan:{_as_sha_digest(plan.digest)}",
        device_index=int(device_index),
        dtype=str(values[0].dtype),
        shape=(int(plan.total_rows), groups, rank),
        strides=(groups * rank, rank, 1),
        byte_offset=0,
        nbytes=max(1, nbytes),
        version=max(versions, default=0),
    )
    binding = ZOffGpuViewBinding(
        layer_id=int(context.layer_id),
        artifact_digest=_as_sha_digest(plan.digest),
        clean_rows=int(context.reused_row_count),
        dirty_rows=int(context.online_local_row_count),
        local_head_count=len(tuple(context.local_head_axes)),
        gpu_view=logical_view,
        persistent_arena_token=persistent_arena_token,
        merge_kernel_token=fused_merge_kernel_token,
    )
    return binding, plan_identity


def _build_cache_domain_bindings(
    *,
    context: MLAOffRuntimeContext,
    preflights: Sequence[LayerCacheDomainPreflight],
    compression_ratio: int,
    restore_plan_digest: str,
    expected_device_index: int,
) -> Tuple[Tuple[CacheDomainBinding, ...], Tuple[Tuple[object, Tuple[object, ...]], ...]]:
    items = tuple(preflights)
    if not items or any(
        not isinstance(item, LayerCacheDomainPreflight) for item in items
    ):
        raise TypeError("cache_domains must contain layer cache preflights")
    expected_components = _expected_cache_components(compression_ratio)
    actual_components = tuple(sorted(item.component for item in items))
    if actual_components != expected_components:
        raise ValueError("cache preflights do not cover the layer domains exactly")
    if len(set(actual_components)) != len(actual_components):
        raise ValueError("cache preflight components must be unique")
    bindings = []
    live_tensors = []
    for item in items:
        if item.omitted_slot_consumed:
            raise RuntimeError("cache omission was consumed before composite commit")
        if isinstance(item.gpu_view, GpuViewBinding):
            view = item.gpu_view
        else:
            view = _gpu_view_from_tensor(
                item.gpu_view,
                storage_role=(
                    f"cache-layer-{int(context.layer_id)}-{item.component}"
                ),
                view_role=item.builder_preflight_token,
            )
            live_tensors.append((item.gpu_view, _tensor_identity(item.gpu_view)))
        if int(view.device_index) != int(expected_device_index):
            raise ValueError("cache, sparse-Q, and z_off views must share a device")
        bindings.append(
            CacheDomainBinding(
                layer_id=int(context.layer_id),
                compression_ratio=int(compression_ratio),
                component=item.component,
                total_units=int(item.total_units),
                restored_units=int(item.restored_units),
                dirty_units=int(item.dirty_units),
                artifact_digest=_as_sha_digest(item.artifact_digest),
                restore_plan_digest=restore_plan_digest,
                gpu_view=view,
                builder_preflight_token=item.builder_preflight_token,
            )
        )
    return (
        tuple(sorted(bindings, key=lambda item: item.key)),
        tuple(live_tensors),
    )


class LayerCompositeCommitBuilder:
    """One layer, one TP collective, and no pre-certificate omission.

    Construction registers immutable sparse-Q/cache/z_off proposals only.
    :meth:`commit` revalidates every live object immediately before delegating
    the sole collective to :class:`ForwardCommitSession`.  The returned
    authorizations are proofs for a later backend installation; this class
    deliberately never calls ``MLAOffRuntimeContext.install_sparse_q_commit``.
    """

    def __init__(
        self,
        *,
        context: MLAOffRuntimeContext,
        resources: CompositeForwardResources,
        proposal: CompositeForwardProposal,
        packed_sparse_q: object,
        packed_identity: Tuple[object, ...],
        persistent_identity: Tuple[object, ...],
        cache_live_tensors: Tuple[Tuple[object, Tuple[object, ...]], ...],
        builder_epoch_token: str,
        committed_forward_session: Optional[ForwardCommitSession] = None,
        committed_forward_authorization: Optional[OmissionAuthorization] = None,
    ) -> None:
        if not isinstance(builder_epoch_token, str) or not builder_epoch_token:
            raise ValueError("builder_epoch_token must be non-empty")
        if (committed_forward_session is None) != (
            committed_forward_authorization is None
        ):
            raise ValueError(
                "forward receipt session and authorization must be supplied together"
            )
        if committed_forward_session is not None and not isinstance(
            committed_forward_session, ForwardCommitSession
        ):
            raise TypeError("forward receipt session has an invalid type")
        if committed_forward_authorization is not None and not isinstance(
            committed_forward_authorization, OmissionAuthorization
        ):
            raise TypeError("forward receipt authorization has an invalid type")
        self.context = context
        self.resources = resources
        self.proposal = proposal
        self.packed_sparse_q = packed_sparse_q
        self._packed_identity = packed_identity
        self._persistent_identity = persistent_identity
        self._cache_live_tensors = cache_live_tensors
        self.builder_epoch_token = builder_epoch_token
        self.committed_forward_session = committed_forward_session
        self.committed_forward_authorization = committed_forward_authorization
        self.session = ForwardCommitSession(proposal)
        self.preflight = None
        self.result: Optional[LayerCompositeCommitResult] = None
        self.local_preflight_error = ""
        self._commit_attempted = False

    @property
    def layer_id(self) -> int:
        return int(self.context.layer_id)

    @property
    def committed(self) -> bool:
        return bool(self.result is not None and self.result.committed)

    def _validate_precommit(
        self,
        *,
        committed_forward_session: Optional[ForwardCommitSession] = None,
        committed_forward_authorization: Optional[OmissionAuthorization] = None,
    ) -> None:
        """Validate either a legacy pre-vote or a forward receipt boundary.

        The default remains the legacy per-layer protocol: no certificate or
        omission authorization may already be installed.  The forward-wide
        protocol must opt in with the exact committed session/authorization;
        that mode accepts only the coordinator binding published before layer
        execution and still rejects an already installed sparse-Q omission or
        layer receipt.
        """

        self.resources.validate(batch_digest=self.resources.batch_digest)
        if self.resources._contexts.get(self.layer_id) is not self.context:
            raise ValueError("composite commit builder lost its context lease")
        batch_plan = self.context.batched_reuse_plan
        batch_plan.validate()
        if _batched_geometry_digest(batch_plan) != self.resources.batch_digest:
            raise ValueError("context ragged geometry changed before commit")
        if bool(getattr(self.context, "sparse_q_committed", False)):
            raise RuntimeError("sparse-Q omission was installed before TP commit")
        if (committed_forward_session is None) != (
            committed_forward_authorization is None
        ):
            raise ValueError(
                "forward receipt session and authorization must be supplied together"
            )
        if committed_forward_session is None:
            if (
                self.committed_forward_session is not None
                or self.committed_forward_authorization is not None
            ):
                raise RuntimeError(
                    "forward receipt builder cannot enter the legacy TP commit"
                )
            if getattr(self.context, "composite_certificate", None) is not None or getattr(
                self.context, "composite_omission_authorization", None
            ) is not None:
                raise RuntimeError("composite omission was installed before TP commit")
        else:
            if not isinstance(committed_forward_session, ForwardCommitSession):
                raise TypeError("forward receipt session has an invalid type")
            if not isinstance(
                committed_forward_authorization, OmissionAuthorization
            ):
                raise TypeError("forward receipt authorization has an invalid type")
            if (
                self.committed_forward_session is not committed_forward_session
                or self.committed_forward_authorization
                is not committed_forward_authorization
            ):
                raise RuntimeError(
                    "layer builder was not registered for this forward certificate"
                )
            coordinator = self.resources.forward_commit_coordinator
            if (
                coordinator is None
                or getattr(coordinator, "session", None)
                is not committed_forward_session
                or getattr(coordinator, "authorization", None)
                is not committed_forward_authorization
                or not bool(getattr(coordinator, "committed", False))
            ):
                raise RuntimeError(
                    "forward receipt builder has a stale or foreign coordinator"
                )
            ledger = getattr(coordinator, "ledger", None)
            if (
                ledger is None
                or getattr(ledger, "session", None)
                is not committed_forward_session
                or getattr(ledger, "authorization", None)
                is not committed_forward_authorization
            ):
                raise RuntimeError("forward receipt ledger binding changed")
            certificate = committed_forward_session.certificate
            if certificate is None:
                raise RuntimeError("forward receipt session has no certificate")
            if (
                getattr(coordinator, "_bound_contexts", {}).get(self.layer_id)
                is not self.context
                or getattr(self.context, "composite_irreversible", False)
                is not True
                or getattr(self.context, "composite_commit_session", None)
                is not committed_forward_session
                or getattr(self.context, "composite_certificate", None)
                is not certificate
                or getattr(
                    self.context, "composite_omission_authorization", None
                )
                is not committed_forward_authorization
            ):
                raise RuntimeError("forward receipt context binding changed")
            if (
                getattr(
                    self.context, "composite_layer_execution_receipt", None
                )
                is not None
            ):
                raise RuntimeError("forward layer receipt was already installed")
        validate_projection = getattr(self.packed_sparse_q, "validate", None)
        if not callable(validate_projection):
            raise TypeError("registered sparse-Q projection lost validation")
        validate_projection()
        if _packed_sparse_q_identity(self.packed_sparse_q) != self._packed_identity:
            raise ValueError("registered packed sparse-Q identity changed")
        persistent = self.context.validate_persistent_projection_commit()
        if _persistent_projection_identity(persistent) != self._persistent_identity:
            raise ValueError("registered persistent z_off identity changed")
        for tensor, identity in self._cache_live_tensors:
            if _tensor_identity(tensor) != identity:
                raise ValueError("registered cache GPU view identity changed")
        if self.proposal.ragged.total_rows != self.resources.total_rows:
            raise ValueError("composite proposal ragged rows changed")
        if self.proposal.shared_latent.restore_plan_digest != (
            self.proposal.cache_domains[0].restore_plan_digest
        ):
            raise ValueError("cache domains lost the shared restore schedule")

    def commit(
        self,
        adapter: object,
        *,
        ready: bool = True,
        failure_code: str = "",
    ) -> LayerCompositeCommitResult:
        """Execute this layer's one composite TP vote and issue proofs."""

        if self.committed_forward_session is not None:
            raise RuntimeError(
                "forward receipt builder cannot execute a second TP commit"
            )
        if self._commit_attempted:
            raise RuntimeError("composite layer commit may be attempted once")
        self._commit_attempted = True
        preexchange_error = ""
        try:
            if type(ready) is not bool:
                raise TypeError("ready must be boolean")
            if ready and failure_code:
                raise ValueError("a ready layer commit cannot carry failure_code")
            if not ready and not failure_code:
                raise ValueError("a rejected layer commit needs failure_code")
            if ready:
                self._validate_precommit()
            preflight = build_cache_builders_preflight(
                self.proposal,
                builder_epoch_token=self.builder_epoch_token,
                ready=ready,
                failure_code=failure_code,
                omitted_slots_consumed=False,
            )
            self.session.record_cache_builders_preflight(preflight)
            self.preflight = preflight
        except BaseException as exc:
            # Preflight construction/recording is still pre-exchange.  Carry
            # every failure, including cancellation-like BaseException, into
            # the session's fixed-shape not-ready vote.
            preexchange_error = f"{type(exc).__name__}: {exc}"
            self.local_preflight_error = preexchange_error
        try:
            self.context.composite_collective_adapter = adapter
        except BaseException as exc:
            assignment_error = f"{type(exc).__name__}: {exc}"
            preexchange_error = preexchange_error or assignment_error
            self.local_preflight_error = preexchange_error
        outcome = None
        try:
            if preexchange_error:
                outcome = self.session.commit(
                    adapter,
                    local_preexchange_error=preexchange_error,
                )
            else:
                outcome = self.session.commit(adapter)
            sparse_authorization = None
            omission_authorization = None
            if outcome.committed:
                certificate = outcome.certificate
                assert certificate is not None
                # These are the first integration writes after the session's
                # explicit accepted decision.
                self.context.composite_irreversible = True
                self.context.composite_certificate = certificate
                sparse_authorization = self.session.install_registered_sparse_q(
                    certificate
                )
                omission_authorization = self.session.authorize_omissions(certificate)
                self.context.composite_omission_authorization = omission_authorization
                self.context.composite_sparse_q_authorization = sparse_authorization
            result = LayerCompositeCommitResult(
                proposal=self.proposal,
                outcome=outcome,
                sparse_q_authorization=sparse_authorization,
                omission_authorization=omission_authorization,
                packed_sparse_q=self.packed_sparse_q,
                irreversible=bool(outcome.committed),
            )
            self.result = result
            return result
        except BaseException as postcollective_error:
            decision = str(
                getattr(self.session, "collective_decision", "indeterminate")
            )
            if decision == "accepted":
                # ready-count and shared moments agreed.  A later local error
                # cannot make dense fallback safe, so preserve an irreversible
                # shape-only carrier even if certificate/authorization/result
                # construction was interrupted.
                try:
                    self.context.composite_irreversible = True
                    if (
                        outcome is not None
                        and getattr(outcome, "certificate", None) is not None
                    ):
                        self.context.composite_certificate = outcome.certificate
                    if getattr(self.context, "composite_pipeline_error", None) is None:
                        self.context.composite_pipeline_error = postcollective_error
                except BaseException:
                    pass
                result = LayerCompositeCommitResult(
                    proposal=self.proposal,
                    outcome=outcome,
                    sparse_q_authorization=None,
                    omission_authorization=None,
                    packed_sparse_q=self.packed_sparse_q,
                    irreversible=True,
                    postcommit_error=postcollective_error,
                )
                self.result = result
                return result
            if decision == "rejected":
                # The collective made a globally safe dense decision.  Never
                # relabel a wrapper/cancellation failure as committed merely
                # because exchange itself completed.
                result = LayerCompositeCommitResult(
                    proposal=self.proposal,
                    outcome=outcome,
                    sparse_q_authorization=None,
                    omission_authorization=None,
                    packed_sparse_q=self.packed_sparse_q,
                    irreversible=False,
                    postcommit_error=postcollective_error,
                )
                self.result = result
                return result
            # not_attempted/indeterminate has no safe local interpretation.
            # ForwardCommitSession already emitted its coordinated fail-stop
            # signal for exchange/malformed-result failures.
            raise

    def consume_omitted_slot(
        self,
        adapter: object,
        authorization: OmissionAuthorization,
        slot: str,
    ) -> None:
        """Fail-closed gate called immediately before an omitted consumer."""

        if self.result is None or not self.result.committed:
            raise RuntimeError("no committed authorization exists for omission")
        if authorization is not self.result.omission_authorization:
            raise ValueError("omission authorization is stale or foreign")
        self._validate_postcommit_live_objects(adapter)
        self.session.consume_omitted_slot(adapter, authorization, slot)

    def _validate_postcommit_live_objects(self, adapter: object) -> None:
        try:
            self.resources.validate()
            validate_projection = getattr(self.packed_sparse_q, "validate", None)
            if not callable(validate_projection):
                raise TypeError("packed sparse-Q validation disappeared")
            validate_projection()
            if _packed_sparse_q_identity(self.packed_sparse_q) != self._packed_identity:
                raise ValueError("packed sparse-Q identity changed after commit")
            persistent = self.context.validate_persistent_projection_commit()
            if _persistent_projection_identity(persistent) != self._persistent_identity:
                raise ValueError("persistent z_off identity changed after commit")
            for tensor, identity in self._cache_live_tensors:
                if _tensor_identity(tensor) != identity:
                    raise ValueError("cache GPU view changed after commit")
        except Exception as exc:
            self.session.fail_closed_after_commit(
                adapter,
                reason_code="live_binding_changed",
                detail=f"{type(exc).__name__}: {exc}",
            )


def _begin_layer_composite_commit_impl(
    context: MLAOffRuntimeContext,
    *,
    cache_domains: Sequence[LayerCacheDomainPreflight],
    packed_sparse_q: object,
    forward_ordinal: int,
    builder_epoch_token: str,
    generation_id: Optional[str] = None,
    model_hash: Optional[str] = None,
    policy_hash: Optional[str] = None,
    persistent_arena_token: Optional[str] = None,
    fused_merge_kernel_token: str = "dsv4_fused_z_merge:v1",
    restore_provider_token: Optional[str] = None,
    restore_provider_local_token: Optional[str] = None,
    restore_batch_common_digest: Optional[str] = None,
    restore_batch_local_digest: Optional[str] = None,
    committed_forward_session: Optional[ForwardCommitSession] = None,
    committed_forward_authorization: Optional[OmissionAuthorization] = None,
) -> LayerCompositeCommitBuilder:
    """Register one reusable layer for its single composite TP commit.

    This function binds the context's full ragged digest, every request's
    shared-latent pin/schedule, the exact packed sparse-Q object/tensors, the
    persistent z_off plan, and already materialized cache-domain views.  It
    does *not* install or consume an omission.
    """

    resources = getattr(context, "_redknot_composite_forward_resources", None)
    if not isinstance(resources, CompositeForwardResources):
        raise RuntimeError("context has no forward-scoped composite resources")
    resources.validate()
    layer_id = int(context.layer_id)
    if resources._contexts.get(layer_id) is not context:
        raise ValueError("context is not active in its composite resource lease")
    if type(forward_ordinal) is not int or forward_ordinal < 0:
        raise ValueError("forward_ordinal must be a non-negative integer")
    if not isinstance(builder_epoch_token, str) or not builder_epoch_token:
        raise ValueError("builder_epoch_token must be non-empty")
    if not isinstance(fused_merge_kernel_token, str) or not fused_merge_kernel_token:
        raise ValueError("fused_merge_kernel_token must be non-empty")
    if (committed_forward_session is None) != (
        committed_forward_authorization is None
    ):
        raise ValueError(
            "forward receipt session and authorization must be supplied together"
        )
    if committed_forward_session is not None:
        if not isinstance(committed_forward_session, ForwardCommitSession):
            raise TypeError("forward receipt session has an invalid type")
        if not isinstance(
            committed_forward_authorization, OmissionAuthorization
        ):
            raise TypeError("forward receipt authorization has an invalid type")
        coordinator = resources.forward_commit_coordinator
        certificate = committed_forward_session.certificate
        if (
            coordinator is None
            or getattr(coordinator, "session", None)
            is not committed_forward_session
            or getattr(coordinator, "authorization", None)
            is not committed_forward_authorization
            or not bool(getattr(coordinator, "committed", False))
            or certificate is None
        ):
            raise RuntimeError(
                "forward receipt construction requires the active committed coordinator"
            )
        if (
            getattr(coordinator, "_bound_contexts", {}).get(layer_id)
            is not context
            or getattr(context, "composite_irreversible", False) is not True
            or getattr(context, "composite_commit_session", None)
            is not committed_forward_session
            or getattr(context, "composite_certificate", None) is not certificate
            or getattr(context, "composite_omission_authorization", None)
            is not committed_forward_authorization
        ):
            raise RuntimeError("forward receipt context binding changed")
        if bool(getattr(context, "sparse_q_committed", False)) or getattr(
            context, "composite_layer_execution_receipt", None
        ) is not None:
            raise RuntimeError(
                "forward receipt construction arrived after layer omission"
            )

    compression_ratio = _layer_compression_ratio(resources, layer_id)
    ragged = _build_ragged_geometry(context, resources)
    shared = _build_shared_latent_binding(
        context,
        resources,
        compression_ratio=compression_ratio,
    )
    sparse = _build_sparse_q_binding(context, packed_sparse_q)
    plan = context.persistent_projection_plan
    arena_token = str(
        persistent_arena_token
        or f"persistent-z-arena:{_as_sha_digest(getattr(plan, 'digest', ''))}"
    )
    zoff, persistent_identity = _build_zoff_binding(
        context,
        persistent_arena_token=arena_token,
        fused_merge_kernel_token=fused_merge_kernel_token,
    )
    if sparse.packed_projection_view.device_index != zoff.gpu_view.device_index:
        raise ValueError("packed sparse-Q and persistent z_off use different devices")
    diagnostic_ablation = str(
        getattr(context, "diagnostic_ablation", "full") or "full"
    )
    if diagnostic_ablation == MLA_OFF_DIAGNOSTIC_ABLATION_ZOFF_ONLY:
        if tuple(cache_domains):
            raise ValueError(
                "zoff-only layer receipt cannot bind shared cache domains"
            )
        caches = ()
        cache_live_tensors = ()
    else:
        caches, cache_live_tensors = _build_cache_domain_bindings(
            context=context,
            preflights=cache_domains,
            compression_ratio=compression_ratio,
            restore_plan_digest=shared.restore_plan_digest,
            expected_device_index=sparse.packed_projection_view.device_index,
        )
    raw_generation = str(generation_id or resources.forward_id)
    layer_generation = f"{raw_generation}:composite-layer:{layer_id}"
    proposal = CompositeForwardProposal(
        identity=ForwardIdentity(
            generation_id=layer_generation,
            forward_ordinal=forward_ordinal,
            model_hash=_as_sha_digest(
                model_hash
                if model_hash is not None
                else context.spec.model_compat_hash
            ),
            policy_hash=_as_sha_digest(
                policy_hash
                if policy_hash is not None
                else context.spec.head_policy_hash
            ),
            tp_size=int(context.spec.tp_size),
        ),
        tp_rank=int(context.spec.tp_rank),
        ragged=ragged,
        shared_latent=shared,
        sparse_q=(sparse,),
        cache_domains=caches,
        z_off_views=(zoff,),
        rank_local_batch_plan_digest=_as_sha_digest(
            context.batched_reuse_plan.digest
        ),
        persistent_zoff_arena_token=arena_token,
        fused_merge_kernel_token=fused_merge_kernel_token,
        restore_provider_token=_as_sha_digest(
            restore_provider_token or "legacy-layer-restore-provider"
        ),
        restore_provider_local_token=_as_sha_digest(
            restore_provider_local_token or "legacy-layer-restore-provider-local"
        ),
        restore_batch_common_digest=_as_sha_digest(
            restore_batch_common_digest or "legacy-layer-restore-batch-common"
        ),
        restore_batch_local_digest=_as_sha_digest(
            restore_batch_local_digest or "legacy-layer-restore-batch-local"
        ),
    )
    builder = LayerCompositeCommitBuilder(
        context=context,
        resources=resources,
        proposal=proposal,
        packed_sparse_q=packed_sparse_q,
        packed_identity=_packed_sparse_q_identity(packed_sparse_q),
        persistent_identity=persistent_identity,
        cache_live_tensors=cache_live_tensors,
        builder_epoch_token=builder_epoch_token,
        committed_forward_session=committed_forward_session,
        committed_forward_authorization=committed_forward_authorization,
    )
    resources.register_commit_builder(layer_id, builder)
    if committed_forward_session is None:
        context.composite_commit_session = builder.session
    elif context.composite_commit_session is not committed_forward_session:
        raise RuntimeError("forward receipt construction replaced its TP session")
    return builder


def build_layer_composite_reservation(
    context: MLAOffRuntimeContext,
    *,
    cache_domains: Sequence[LayerCacheDomainPreflight],
    sparse_q_reservation: Optional[object],
    forward_ordinal: int,
    builder_epoch_token: str,
    generation_id: Optional[str] = None,
    model_hash: Optional[str] = None,
    policy_hash: Optional[str] = None,
    persistent_arena_token: Optional[str] = None,
    fused_merge_kernel_token: str = "dsv4_fused_z_merge:v1",
    omission_profile: str = OMISSION_PROFILE_FULL,
    restore_provider_token: Optional[str] = None,
    restore_provider_local_token: Optional[str] = None,
    restore_batch_common_digest: Optional[str] = None,
    restore_batch_local_digest: Optional[str] = None,
    failure_carrier: Optional[object] = None,
) -> CompositeForwardProposal:
    """Build one mutation-free layer proposal for a future Q arena write.

    Unlike :func:`begin_layer_composite_commit`, this function neither creates
    nor registers a compatibility builder.  It is safe before layer 3 because
    the sparse-Q binding carries the ticket's expected post-write version.
    """

    resources = getattr(context, "_redknot_composite_forward_resources", None)
    if not isinstance(resources, CompositeForwardResources):
        raise RuntimeError("context has no forward-scoped composite resources")
    resources.validate()
    layer_id = int(context.layer_id)
    if resources._contexts.get(layer_id) is not context:
        raise ValueError("context is not active in its composite resource lease")
    if type(forward_ordinal) is not int or forward_ordinal < 0:
        raise ValueError("forward_ordinal must be a non-negative integer")
    if not isinstance(builder_epoch_token, str) or not builder_epoch_token:
        raise ValueError("builder_epoch_token must be non-empty")
    if omission_profile not in (
        OMISSION_PROFILE_FULL,
        OMISSION_PROFILE_ZOFF_ONLY,
        OMISSION_PROFILE_SHARED_ONLY,
    ):
        raise ValueError("layer reservation omission profile is invalid")
    compression_ratio = _layer_compression_ratio(resources, layer_id)
    ragged = _build_ragged_geometry(context, resources)
    shared = _build_shared_latent_binding(
        context,
        resources,
        compression_ratio=compression_ratio,
    )
    headsplit_profile = omission_profile in (
        OMISSION_PROFILE_FULL,
        OMISSION_PROFILE_ZOFF_ONLY,
    )
    if headsplit_profile:
        if sparse_q_reservation is None:
            raise ValueError("head-split reservation requires a Q arena ticket")
        sparse = _build_sparse_q_reservation_binding(
            context, sparse_q_reservation
        )
        arena_token = str(
            persistent_arena_token
            or f"persistent-z-arena:{resources.forward_id}:tp:{int(context.spec.tp_rank)}"
        )
        zoff, _ = _build_zoff_binding(
            context,
            persistent_arena_token=arena_token,
            fused_merge_kernel_token=fused_merge_kernel_token,
        )
        expected_device_index = sparse.packed_projection_view.device_index
        sparse_bindings = (sparse,)
        zoff_bindings = (zoff,)
    else:
        if sparse_q_reservation is not None:
            raise ValueError("shared-only reservation cannot receive a Q ticket")
        arena_token = str(
            persistent_arena_token or "shared-only:no-persistent-zoff"
        )
        sparse_bindings = ()
        zoff_bindings = ()
        first_preflight = tuple(cache_domains)[0]
        first_view = first_preflight.gpu_view
        if isinstance(first_view, GpuViewBinding):
            expected_device_index = int(first_view.device_index)
        else:
            first_device = getattr(first_view, "device", None)
            if getattr(first_device, "type", None) != "cuda":
                raise ValueError("shared-only cache views must remain CUDA resident")
            expected_device_index = int(
                getattr(first_device, "index", None) or 0
            )
    if omission_profile == OMISSION_PROFILE_ZOFF_ONLY:
        if tuple(cache_domains):
            raise ValueError(
                "zoff-only reservation cannot bind shared cache domains"
            )
        caches = ()
    else:
        caches, _ = _build_cache_domain_bindings(
            context=context,
            preflights=cache_domains,
            compression_ratio=compression_ratio,
            restore_plan_digest=shared.restore_plan_digest,
            expected_device_index=expected_device_index,
        )
    if failure_carrier is None:
        raise ValueError("forward reservation requires a failure carrier arena")
    failure_carrier_view = _gpu_view_from_tensor(
        failure_carrier,
        storage_role=f"failure-carrier-forward-{resources.forward_id}",
        view_role=f"failure-carrier-tp-{int(context.spec.tp_rank)}",
    )
    if failure_carrier_view.device_index != expected_device_index:
        raise ValueError("failure carrier is on another GPU")
    raw_generation = str(generation_id or resources.forward_id)
    return CompositeForwardProposal(
        identity=ForwardIdentity(
            generation_id=f"{raw_generation}:composite-layer:{layer_id}",
            forward_ordinal=forward_ordinal,
            model_hash=_as_sha_digest(
                model_hash
                if model_hash is not None
                else context.spec.model_compat_hash
            ),
            policy_hash=_as_sha_digest(
                policy_hash
                if policy_hash is not None
                else context.spec.head_policy_hash
            ),
            tp_size=int(context.spec.tp_size),
        ),
        tp_rank=int(context.spec.tp_rank),
        ragged=ragged,
        shared_latent=shared,
        sparse_q=sparse_bindings,
        cache_domains=caches,
        z_off_views=zoff_bindings,
        rank_local_batch_plan_digest=_as_sha_digest(
            context.batched_reuse_plan.digest
        ),
        persistent_zoff_arena_token=arena_token,
        fused_merge_kernel_token=fused_merge_kernel_token,
        restore_provider_token=_as_sha_digest(
            restore_provider_token or "missing-forward-restore-provider"
        ),
        restore_provider_local_token=_as_sha_digest(
            restore_provider_local_token
            or "missing-forward-restore-provider-local"
        ),
        restore_batch_common_digest=_as_sha_digest(
            restore_batch_common_digest or "pending-forward-restore-batch-common"
        ),
        restore_batch_local_digest=_as_sha_digest(
            restore_batch_local_digest or "pending-forward-restore-batch-local"
        ),
        failure_carrier_view=failure_carrier_view,
        omission_profile=omission_profile,
        commit_scope=COMMIT_SCOPE_FORWARD_FRAGMENT,
    )


def begin_layer_composite_commit(
    context: MLAOffRuntimeContext,
    *,
    cache_domains: Sequence[LayerCacheDomainPreflight],
    packed_sparse_q: object,
    forward_ordinal: int,
    builder_epoch_token: str,
    generation_id: Optional[str] = None,
    model_hash: Optional[str] = None,
    policy_hash: Optional[str] = None,
    persistent_arena_token: Optional[str] = None,
    fused_merge_kernel_token: str = "dsv4_fused_z_merge:v1",
    restore_provider_token: Optional[str] = None,
    restore_provider_local_token: Optional[str] = None,
    restore_batch_common_digest: Optional[str] = None,
    restore_batch_local_digest: Optional[str] = None,
    committed_forward_session: Optional[ForwardCommitSession] = None,
    committed_forward_authorization: Optional[OmissionAuthorization] = None,
) -> LayerCompositeCommitBuilder:
    """Build/register a legacy commit or a committed-forward layer receipt.

    Legacy construction owns pre-commit cleanup.  An explicit forward receipt
    runs after the all-layer certificate is irreversible, so its transaction
    must retain the lease and carry any failure to the fixed final rendezvous.
    """

    resources = getattr(context, "_redknot_composite_forward_resources", None)
    forward_receipt_requested = bool(
        committed_forward_session is not None
        or committed_forward_authorization is not None
    )
    try:
        return _begin_layer_composite_commit_impl(
            context,
            cache_domains=cache_domains,
            packed_sparse_q=packed_sparse_q,
            forward_ordinal=forward_ordinal,
            builder_epoch_token=builder_epoch_token,
            generation_id=generation_id,
            model_hash=model_hash,
            policy_hash=policy_hash,
            persistent_arena_token=persistent_arena_token,
            fused_merge_kernel_token=fused_merge_kernel_token,
            restore_provider_token=restore_provider_token,
            restore_provider_local_token=restore_provider_local_token,
            restore_batch_common_digest=restore_batch_common_digest,
            restore_batch_local_digest=restore_batch_local_digest,
            committed_forward_session=committed_forward_session,
            committed_forward_authorization=committed_forward_authorization,
        )
    except Exception:
        if forward_receipt_requested:
            # The all-layer vote is already accepted.  Closing or detaching its
            # lease here would bypass the mandatory forward-final rendezvous;
            # the backend records this exception in the coordinator ledger.
            raise
        if isinstance(resources, CompositeForwardResources):
            try:
                resources.close()
            except Exception:
                # close() attempted every pin and marked the lease closed; keep
                # the proposal/registration exception as the primary failure.
                pass
            owner = getattr(context, "_redknot_forward_batch_owner", None)
            if owner is not None and getattr(
                owner, "_redknot_composite_forward_resources", None
            ) is resources:
                owner._redknot_composite_forward_resources = None
            if owner is not None:
                owner._redknot_mla_off_restore_layout = None
        raise


def _split_lengths(forward_batch, total_rows: int) -> Tuple[int, ...]:
    values = tuple(int(value) for value in forward_batch.extend_seq_lens_cpu)
    if len(values) != int(forward_batch.batch_size) or sum(values) != int(total_rows):
        raise ValueError("ragged extend lengths do not tile attention rows")
    if any(value <= 0 for value in values):
        raise ValueError("every continuously batched request needs rows")
    return values


def _split_request_positions_cpu(
    positions_cpu: torch.Tensor,
    *,
    lengths: Sequence[int],
    total_rows: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Materialize request-scoped positions once on a geometry-cache miss."""

    values = tuple(int(value) for value in positions_cpu.tolist())
    if len(values) != int(total_rows) or sum(int(value) for value in lengths) != int(
        total_rows
    ):
        raise ValueError("request positions do not tile the packed forward")
    requests = []
    offset = 0
    for raw_length in lengths:
        length = int(raw_length)
        if length <= 0:
            raise ValueError("every request position slice needs rows")
        requests.append(values[offset : offset + length])
        offset += length
    if offset != int(total_rows):
        raise ValueError("request positions do not tile the packed forward")
    return tuple(requests)


def _scheduler_request_total_tokens(
    forward_batch,
    *,
    expected_requests: int,
) -> Tuple[Optional[int], ...]:
    """Read all scheduler-owned unchunked lengths with one CPU transfer."""

    missing = (None,) * int(expected_requests)
    raw = getattr(forward_batch, "orig_seq_lens", None)
    if not isinstance(raw, torch.Tensor):
        return missing
    try:
        if raw.ndim != 1 or int(raw.numel()) != int(expected_requests):
            return missing
        integer_dtypes = tuple(
            value
            for value in (getattr(torch, "int32", None), torch.long)
            if value is not None
        )
        if integer_dtypes and raw.dtype not in integer_dtypes:
            return missing
        values = tuple(
            int(value)
            for value in raw.detach().to(device="cpu", dtype=torch.long).tolist()
        )
    except (RuntimeError, TypeError, ValueError):
        return missing
    if len(values) != int(expected_requests) or any(value <= 0 for value in values):
        return missing
    return values


def _scheduler_request_current_extents(
    forward_batch,
    *,
    expected_requests: int,
) -> Tuple[Optional[int], ...]:
    """Read the cumulative sequence extent of this prefill microforward."""

    missing = (None,) * int(expected_requests)
    raw = getattr(forward_batch, "seq_lens_cpu", None)
    if isinstance(raw, torch.Tensor):
        try:
            if raw.ndim != 1 or int(raw.numel()) != int(expected_requests):
                return missing
            integer_dtypes = tuple(
                value
                for value in (getattr(torch, "int32", None), torch.long)
                if value is not None
            )
            if integer_dtypes and raw.dtype not in integer_dtypes:
                return missing
            values = tuple(int(value) for value in raw.tolist())
        except (RuntimeError, TypeError, ValueError):
            return missing
    elif isinstance(raw, (list, tuple)):
        if len(raw) != int(expected_requests) or any(
            type(value) is not int for value in raw
        ):
            return missing
        values = tuple(raw)
    else:
        return missing
    if len(values) != int(expected_requests) or any(value <= 0 for value in values):
        return missing
    return values


def _active_restore_plan_fallback_reason(
    backend,
    *,
    plan: Mapping[str, object],
    scheduler_total_tokens: Optional[int],
    scheduler_current_extent: Optional[int],
    request_positions: Sequence[int],
    original_chunk_range: Optional[Tuple[int, int]] = None,
) -> str:
    """Validate one chunked request before any artifact pin/view is acquired.

    The production backend validator is imported lazily to avoid the module
    cycle.  A duck-typed validator is accepted for isolated CPU tests and for
    embedding the runtime in a backend-compatible implementation.  The
    ``orig_seq_lens`` authenticates the request's declared final length, while
    ``seq_lens_cpu`` supplies the cumulative extent of this microforward.
    Binding both scheduler-owned values keeps every chunk independently
    authenticated without weakening the full-plan segment/artifact contract.
    """

    validator = getattr(
        backend, "_validate_pure_headsplit_plan_contract", None
    )
    if not callable(validator):
        try:
            from sglang.srt.layers.attention.redknot_mla_backend import (
                _validate_pure_headsplit_plan_contract as validator,
            )
        except Exception:
            return "pure_headsplit_validator_unavailable"
    try:
        validator(plan)
    except Exception as exc:
        return f"pure_headsplit_contract:{type(exc).__name__}"
    if str(plan.get("model_compat_hash", "")) != str(
        getattr(backend, "_redknot_mla_off_model_hash", "")
    ):
        return "model_compat_hash_mismatch"
    if str(plan.get("head_policy_hash", "")) != str(
        getattr(backend, "_redknot_mla_off_policy_hash", "")
    ):
        return "head_policy_hash_mismatch"

    query_start = plan.get("query_start")
    if type(query_start) is not int or query_start < 0:
        return "invalid_query_start"

    cap_resolver = getattr(backend, "_mla_off_resolve_context_cap", None)
    if not callable(cap_resolver):
        return "context_cap_resolution_unavailable"
    try:
        context_cap, cap_reason = cap_resolver(plan)
    except Exception:
        return "context_cap_resolution_failed"
    if cap_reason:
        return str(cap_reason)
    if type(context_cap) is not int or context_cap <= 0:
        return "context_cap_resolution_failed"
    declared_total = plan.get("total_tokens")
    if type(declared_total) is not int or declared_total <= 0:
        return "missing_total_tokens"
    if type(scheduler_total_tokens) is not int or scheduler_total_tokens <= 0:
        return "missing_actual_total_tokens"
    if scheduler_total_tokens != declared_total:
        return "total_tokens_mismatch"
    if type(scheduler_current_extent) is not int or scheduler_current_extent <= 0:
        return "missing_scheduler_extent"
    if scheduler_current_extent > declared_total:
        return "scheduler_extent_exceeds_total_tokens"
    positions = tuple(request_positions)
    if not positions:
        return "missing_request_positions"
    if any(type(position) is not int for position in positions):
        return "invalid_request_positions"
    combined_row_sparse = str(plan.get("mla_off_execution_profile", "")) == (
        _COMBINED_ROW_SPARSE_PROFILE
    )
    if positions[0] < 0:
        return "invalid_request_positions"
    if combined_row_sparse:
        if any(right <= left for left, right in zip(positions, positions[1:])):
            return "invalid_request_positions"
        if (
            not isinstance(original_chunk_range, tuple)
            or len(original_chunk_range) != 2
            or any(type(value) is not int for value in original_chunk_range)
        ):
            return "missing_selected_row_chunk_range"
        chunk_start, chunk_end = original_chunk_range
        if (
            chunk_start < 0
            or chunk_start >= chunk_end
            or chunk_end != scheduler_current_extent
            or positions[0] < chunk_start
            or positions[-1] >= chunk_end
        ):
            return "selected_row_chunk_geometry_mismatch"
    elif any(
        right != left + 1 for left, right in zip(positions, positions[1:])
    ):
        return "invalid_request_positions"
    if positions[-1] >= declared_total:
        return "request_position_exceeds_total_tokens"
    if not combined_row_sparse and positions[-1] + 1 != scheduler_current_extent:
        return "scheduler_extent_position_mismatch"
    if query_start > declared_total:
        return "query_start_exceeds_total_tokens"
    if scheduler_total_tokens > context_cap:
        if plan.get("mla_off_qualification_only") is True:
            return "context_exceeds_qualification"
        return "context_exceeds_certification"
    return ""


def _validate_independent_restore_chunk(
    *,
    plan: Mapping[str, object],
    request_positions: Sequence[int],
    request_tokens: Sequence[int],
    trusted_cached_prefix_tokens: Sequence[int] = (),
) -> str:
    """Validate one independent-document restore microforward.

    Independent artifacts are deliberately captured with local positions
    ``[0, length)`` and therefore cannot use the context-bound prefix-chain
    certificate.  They are still fail-closed: every online offline-prefix row
    must tile complete frozen document segments, every segment's token hash is
    rechecked, and a radix consumer must prove the scheduler-owned cached first
    document.  Query/sentinel rows are online-only and need no artifact.
    """

    from sglang.srt.layers.attention.redknot.dsv4_context_identity import (
        token_ids_sha256,
    )

    if str(plan.get("mla_off_execution_profile", "")) != (
        "pure_headsplit_independent_rope_relocation_fullscope_"
        "boundary128_3_37_3_v1"
    ):
        raise ValueError("independent restore validator received a foreign profile")

    positions = tuple(request_positions)
    tokens = tuple(request_tokens)
    if not positions or len(positions) != len(tokens):
        raise ValueError(
            "independent restore needs one token for every non-empty row"
        )
    if positions[0] < 0 or any(
        right != left + 1 for left, right in zip(positions, positions[1:])
    ):
        raise ValueError("independent restore positions must be contiguous")
    if any(type(token_id) is not int for token_id in tokens):
        raise TypeError("independent restore token ids must be built-in integers")

    query_start = plan.get("query_start")
    declared_total = plan.get("total_tokens")
    if (
        type(query_start) is not int
        or type(declared_total) is not int
        or query_start < 0
        or declared_total < query_start
    ):
        raise ValueError("independent restore query/total geometry is invalid")
    start = positions[0]
    end = positions[-1] + 1
    if end > declared_total:
        raise ValueError("independent restore rows exceed total_tokens")

    raw_segments = plan.get("segments")
    if not isinstance(raw_segments, (tuple, list)) or not raw_segments:
        raise ValueError("independent restore needs frozen document segments")
    segments = tuple(raw_segments)

    radix_role = plan.get("radix_prefix_role")
    if radix_role == "consume":
        prefix_tokens = plan.get("radix_prefix_tokens")
        prefix_hash = plan.get("radix_prefix_input_hash")
        cached_prefix = tuple(trusted_cached_prefix_tokens)
        if (
            type(prefix_tokens) is not int
            or prefix_tokens <= 0
            or len(cached_prefix) != prefix_tokens
            or token_ids_sha256(cached_prefix) != prefix_hash
        ):
            raise ValueError(
                "independent radix consumer lacks its exact cached first document"
            )
        if start < prefix_tokens:
            raise ValueError(
                "independent radix consumer restarted inside its cached prefix"
            )
    elif trusted_cached_prefix_tokens:
        raise ValueError(
            "independent non-consumer unexpectedly received cached prefix tokens"
        )

    if start >= query_start:
        return "suffix_complete" if end == declared_total else "suffix"
    if end > query_start:
        raise ValueError(
            "independent restore microforward crosses the query boundary"
        )

    by_offset = {}
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise TypeError("independent restore segment must be a mapping")
        offset = segment.get("global_offset")
        length = segment.get("length")
        if (
            type(offset) is not int
            or type(length) is not int
            or offset < 0
            or length <= 0
            or offset in by_offset
        ):
            raise ValueError("independent restore segment geometry is invalid")
        by_offset[offset] = segment

    cursor = start
    while cursor < end:
        segment = by_offset.get(cursor)
        if segment is None:
            raise ValueError(
                "independent restore rows do not begin on a document boundary"
            )
        length = int(segment["length"])
        segment_end = cursor + length
        if segment_end > end:
            raise ValueError(
                "independent restore microforward contains a partial document"
            )
        token_begin = cursor - start
        token_end = segment_end - start
        if token_ids_sha256(tokens[token_begin:token_end]) != segment.get(
            "token_hash"
        ):
            raise ValueError("independent restore document token hash mismatch")
        cursor = segment_end
    if cursor != end:
        raise AssertionError("independent restore document tiling is incomplete")
    return "segment"


def _validate_combined_sparse_restore_chunk(
    *,
    plan: Mapping[str, object],
    request_positions: Sequence[int],
    request_tokens: Sequence[int],
    original_chunk_range: Optional[Tuple[int, int]],
    trusted_cached_prefix_tokens: Sequence[int] = (),
) -> str:
    """Validate the selected-row view against its scheduler-owned full chunk.

    The model runner has already validated and materialized the checkpoint
    closure before it removes inactive transformer rows.  MLA consumes that
    sparse view, but authenticates it against the original scheduler chunk so
    a non-contiguous selected-row vector is never mistaken for a shorter
    contiguous request.  Reusable z_off rows receive a second, row-exact token
    identity check when their committed artifact view is gathered.
    """

    if str(plan.get("mla_off_execution_profile", "")) != (
        _COMBINED_ROW_SPARSE_PROFILE
    ):
        raise ValueError("combined sparse validator received a foreign profile")
    positions = tuple(request_positions)
    tokens = tuple(request_tokens)
    if not positions or len(positions) != len(tokens):
        raise ValueError(
            "combined sparse restore needs one token for every selected row"
        )
    if any(type(position) is not int for position in positions) or any(
        type(token_id) is not int for token_id in tokens
    ):
        raise TypeError("combined sparse rows and token ids must be integers")
    if positions[0] < 0 or any(
        right <= left for left, right in zip(positions, positions[1:])
    ):
        raise ValueError("combined sparse restore positions must be sorted unique")
    if (
        not isinstance(original_chunk_range, tuple)
        or len(original_chunk_range) != 2
        or any(type(value) is not int for value in original_chunk_range)
    ):
        raise ValueError("combined sparse restore lacks its original chunk range")
    chunk_start, chunk_end = original_chunk_range
    query_start = plan.get("query_start")
    declared_total = plan.get("total_tokens")
    if (
        type(query_start) is not int
        or type(declared_total) is not int
        or query_start < 0
        or declared_total < query_start
        or chunk_start < 0
        or chunk_start >= chunk_end
        or chunk_end > declared_total
        or positions[0] < chunk_start
        or positions[-1] >= chunk_end
    ):
        raise ValueError("combined sparse restore chunk geometry is invalid")

    # The sole radix-enabled combined contract is an exact document-1 seed.
    # Authenticate scheduler-owned cached token ids here as well as the
    # backend's physical terminal-state receipt; neither proof is sufficient
    # on its own.  A non-consumer must never inherit an implicit cached prefix.
    radix_role = plan.get("radix_prefix_role")
    cached_prefix = tuple(trusted_cached_prefix_tokens)
    if radix_role == "consume":
        from sglang.srt.layers.attention.redknot.dsv4_context_identity import (
            token_ids_sha256,
        )

        prefix_tokens = plan.get("radix_prefix_tokens")
        prefix_hash = plan.get("radix_prefix_input_hash")
        if (
            type(prefix_tokens) is not int
            or prefix_tokens <= 0
            or len(cached_prefix) != prefix_tokens
            or token_ids_sha256(cached_prefix) != prefix_hash
            or chunk_start < prefix_tokens
        ):
            raise ValueError(
                "combined radix consumer lacks its exact cached first document"
            )
    elif cached_prefix:
        raise ValueError(
            "combined non-consumer unexpectedly received cached prefix tokens"
        )

    if chunk_start >= query_start:
        if positions != tuple(range(chunk_start, chunk_end)):
            raise ValueError("combined sparse query suffix must remain fully online")
        return "suffix_complete" if chunk_end == declared_total else "suffix"
    if chunk_end > query_start:
        raise ValueError("combined sparse chunk crosses the query boundary")

    # A scheduler-owned merged prefill may cover several complete frozen
    # documents.  It is safe only when the declared chunk is an exact,
    # gap-free tiling of whole segments; accepting a partial first/last
    # document would make an independently captured z_off artifact appear to
    # have the wrong local coordinate system.  Do not infer the tiling from
    # the sparse rows themselves because the canonical first document may be
    # represented solely by the already materialized prefix.
    all_segments = tuple(
        sorted(
            (
                segment
                for segment in tuple(plan.get("segments", ()))
                if isinstance(segment, Mapping)
            ),
            key=lambda segment: int(segment.get("global_offset", -1)),
        )
    )
    matching_segments = tuple(
        segment
        for segment in all_segments
        if type(segment.get("global_offset")) is int
        and type(segment.get("length")) is int
        and chunk_start <= int(segment["global_offset"])
        and int(segment["global_offset"]) + int(segment["length"]) <= chunk_end
    )
    if not matching_segments:
        raise ValueError(
            "combined sparse document chunk does not tile frozen segments"
        )
    cursor = chunk_start
    for segment in matching_segments:
        segment_start = int(segment["global_offset"])
        segment_length = int(segment["length"])
        segment_end = segment_start + segment_length
        if segment_length <= 0 or segment_start != cursor:
            raise ValueError(
                "combined sparse document chunk does not tile frozen segments"
            )
        if segment_start != 0:
            boundary_end = min(segment_start + 128, segment_end)
            required_boundary = set(range(segment_start, boundary_end))
            if not required_boundary.issubset(positions):
                raise ValueError(
                    "combined sparse document omitted its relocation boundary rows"
                )
        cursor = segment_end
    if cursor != chunk_end:
        raise ValueError(
            "combined sparse document chunk does not tile frozen segments"
        )
    return "segments" if len(matching_segments) > 1 else "segment"


def _dense_placeholder_for_invalid_restore(
    plan: Mapping[str, object],
    *,
    reason: str,
) -> Mapping[str, object]:
    """Retain request identity while removing every restore activation bit."""

    placeholder = {
        "mode": "dense",
        "reuse_mla_off": False,
        "_redknot_composite_fallback_reason": str(reason),
    }
    for name in ("benchmark_request_id", "request_id"):
        value = plan.get(name)
        if value is not None:
            placeholder[name] = str(value)
    return MappingProxyType(placeholder)


def _server_artifact_token(views: Sequence[object]) -> str:
    payload = tuple(
        sorted(
            (str(view.seg_hash), int(view.commit_epoch), int(view.layer_id))
            for view in views
        )
    )
    return "sha256:" + hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _spec_for_layer(
    backend,
    *,
    layer_id: int,
    n_local_heads: int,
    n_local_groups: int,
    head_dim: int,
    o_lora_rank: int,
) -> Tuple[MLAOffLayerSpec, Tuple[int, ...]]:
    logical_start = int(backend._redknot_tp_rank) * int(n_local_heads)
    owned = tuple(range(logical_start, logical_start + int(n_local_heads)))
    owned_set = set(owned)
    plan = backend._redknot_dual_layer_plans[int(layer_id)]
    local = tuple(
        head
        for _, heads in plan.local_groups
        for head in heads
        if head in owned_set
    )
    local_set = set(local)
    axes = tuple(axis for axis, head in enumerate(owned) if head in local_set)
    if not axes:
        raise ValueError("every TP rank must own reusable heads")
    spec = MLAOffLayerSpec(
        layer_id=int(layer_id),
        tp_rank=int(backend._redknot_tp_rank),
        tp_size=int(backend._redknot_tp_size),
        owned_logical_heads=owned,
        offline_local_heads=local,
        num_output_groups=int(n_local_groups),
        heads_per_group=int(n_local_heads // n_local_groups),
        head_dim=int(head_dim),
        o_lora_rank=int(o_lora_rank),
        model_compat_hash=backend._redknot_mla_off_model_hash,
        head_policy_hash=backend._redknot_mla_off_policy_hash,
        execution_profile=backend._redknot_mla_off_execution_profile,
        required_layer_ids=backend._redknot_mla_off_rank_local_layer_ids,
        position_semantics=(
            MLA_OFF_INDEPENDENT_POSITION_SEMANTICS
            if backend._redknot_mla_off_execution_profile
            in (
                "pure_headsplit_independent_rope_relocation_fullscope_"
                "boundary128_3_37_3_v1",
                "combined_headsplit_independent_rope_zoff_checkpoint_"
                "rowsparse_3_37_3_v1",
            )
            else "post_inverse_rope_offline_head_woa_"
            "context_exactpos_fullscope_v3"
        ),
    )
    return spec, axes


def _shared_selected_prefix_tokens(
    execution_profile: str,
    segments: Sequence[Mapping[str, object]],
) -> Tuple[Tuple[int, int], ...]:
    """Match shared-KV dirty rows to the effective z_off boundary domain."""

    combined = str(execution_profile) == _COMBINED_ROW_SPARSE_PROFILE
    return tuple(
        (
            int(index),
            0
            if combined and int(segment.get("global_offset", 0)) == 0
            else int(segment.get("skip_first", 0)),
        )
        for index, segment in enumerate(tuple(segments))
    )


def _prepare_shared_request_states(
    backend,
    *,
    geometry: CompositeForwardGeometry,
    plans: Tuple[object, ...],
    positions_cpu: torch.Tensor,
    input_ids_cpu: torch.Tensor,
    lengths: Tuple[int, ...],
    device: torch.device,
    forward_id: str,
) -> Tuple[SharedRequestRestoreState, ...]:
    controller = backend._redknot_shared_latent_controller
    stores = backend._redknot_shared_gpu_stores
    states = []
    dense_requests = []
    template_cache_enabled = bool(
        getattr(backend, "_redknot_geometry_template_cache_enabled", False)
    )
    cached_plan_templates = (
        geometry.shared_restore_plan_templates
        if template_cache_enabled
        else None
    )
    built_plan_templates = []
    offset = 0
    try:
        for request_index, (plan, row_count) in enumerate(zip(plans, lengths)):
            request_positions = positions_cpu.narrow(0, offset, row_count)
            request_tokens = input_ids_cpu.narrow(0, offset, row_count)
            if (
                not isinstance(plan, Mapping)
                or plan.get("mode") != "restore"
                or not bool(plan.get("reuse_mla_off", False))
            ):
                dense_requests.append((request_index, offset, row_count))
                states.append(None)
                built_plan_templates.append(None)
                offset += row_count
                continue
            cpu_plan = (
                cached_plan_templates[request_index]
                if cached_plan_templates is not None
                else None
            )
            if cpu_plan is None:
                segments = tuple(
                    SegmentPlacement(
                        seg_hash=str(segment["seg_hash"]),
                        global_offset=int(segment["global_offset"]),
                        length=int(segment["length"]),
                        token_hash=str(segment.get("token_hash", "")),
                    )
                    for segment in tuple(plan.get("segments", ()))
                )
                if not segments:
                    raise ValueError("shared-latent restore needs segment placements")
                first_artifact = controller.get_committed(segments[0].seg_hash)
                spec = first_artifact.spec
                with _redknot_runtime_timed("mla_shared_cpu_plan_build"):
                    execution_profile = str(plan["mla_off_execution_profile"])
                    independent_relocation = execution_profile in (
                        _INDEPENDENT_RELOCATION_PROFILE,
                        _COMBINED_ROW_SPARSE_PROFILE,
                    )
                    cpu_plan = controller.prepare_restore(
                        spec=spec,
                        placements=segments,
                        positions=tuple(
                            int(value) for value in request_positions.tolist()
                        ),
                        input_token_ids=tuple(
                            int(value) for value in request_tokens.tolist()
                        ),
                        query_start=int(plan["query_start"]),
                        boundary_tokens=128 if independent_relocation else 0,
                        execution_profile=execution_profile,
                        selected_prefix_tokens=_shared_selected_prefix_tokens(
                            execution_profile,
                            tuple(plan["segments"]),
                        ),
                        checkpoint_islands=tuple(
                            plan.get("checkpoint_islands", ())
                        ),
                        protected_ranges=tuple(
                            plan.get("query_protected_ranges", ())
                        ),
                    )
            elif not isinstance(cpu_plan, SharedLatentRestorePlan):
                raise TypeError("shared restore plan template has a foreign type")
            spec = cpu_plan.spec
            controller.validate_restore_plan(cpu_plan)
            built_plan_templates.append(cpu_plan)
            fingerprint = (
                str(spec.model_hash)
                + ":"
                + str(spec.policy_hash)
                + ":"
                + str(spec.length)
            )
            store = stores.get(fingerprint)
            if store is None:
                raise ValueError(
                    "shared-latent GPU store is absent for this artifact"
                )
            pin = store.atomic_pin(cpu_plan)
            try:
                schedule = store.preflight(
                    pin,
                    cpu_plan,
                    forward_id=f"{forward_id}:request:{request_index}",
                )
                workspace = DeviceRestoreWorkspace(
                    max_index_values=max(1, len(schedule.index_arena)),
                    max_restore_rows=max(
                        1,
                        max(
                            (operation.count for operation in schedule.operations),
                            default=0,
                        ),
                    ),
                    max_record_bytes=max(
                        domain.record_bytes for domain in store.layout.domains
                    ),
                    device=device,
                    # The production composite path consumes only the loaded
                    # source/output index arena when it builds one cross-layer
                    # pointer-table batch.  The legacy scratch cartesian
                    # product [max_rows, max_record_bytes] is never read and
                    # reaches 31.88 GiB for a 64K merged prefill.  Keep the
                    # declared capacities for validation, but allocate no
                    # legacy gather scratch; restore_clean() fails closed if
                    # an index-only workspace is ever routed to that API.
                    allocate_restore_scratch=False,
                )
                prepared = store.prepare(schedule, pin, workspace)
            except Exception:
                pin.close()
                raise
            state = SharedRequestRestoreState(
                request_index=request_index,
                flat_row_offset=offset,
                row_count=row_count,
                cpu_plan=cpu_plan,
                pin=pin,
                schedule=schedule,
                prepared=prepared,
            )
            states.append(state)
            state.validate()
            offset += row_count
        if (
            template_cache_enabled
            and cached_plan_templates is None
        ):
            geometry.install_shared_restore_plan_templates(
                tuple(built_plan_templates)
            )
            counter = getattr(backend, "_count", None)
            if callable(counter):
                counter("shared_restore_plan_template_cache_misses")
        elif cached_plan_templates is not None:
            counter = getattr(backend, "_count", None)
            if callable(counter):
                counter("shared_restore_plan_template_cache_hits")
    except Exception:
        _close_shared_request_states(states)
        raise
    try:
        reusable_states = tuple(state for state in states if state is not None)
        if not reusable_states:
            raise ValueError("composite shared restore has no reusable request")
        reference_layers = reusable_states[0].cpu_plan.layers
        dense_layers = MappingProxyType(
            {
                int(layer_id): _DenseLayerRestorePlan(
                    layer_id=int(layer_id),
                    compress_ratio=int(
                        reference_layers[int(layer_id)].compress_ratio
                    ),
                )
                for layer_id in backend._redknot_mla_off_rank_local_layer_ids
            }
        )
        for request_index, dense_offset, row_count in dense_requests:
            states[request_index] = SharedRequestRestoreState(
                request_index=int(request_index),
                flat_row_offset=int(dense_offset),
                row_count=int(row_count),
                cpu_plan=_DenseSharedRestorePlan(
                    dirty_output_rows=tuple(range(int(row_count))),
                    layers=dense_layers,
                ),
                pin=None,
                schedule=None,
                prepared=None,
                reusable=False,
            )
        result = tuple(states)
        for state in result:
            state.validate()
        return result
    except Exception:
        _close_shared_request_states(states)
        raise


def _query_suffix_dense_request_states(
    geometry: CompositeForwardGeometry,
) -> Tuple[SharedRequestRestoreState, ...]:
    """Build a pin-free lease for an intentional query-only dense pass."""

    if not geometry.is_intentional_full_local:
        raise ValueError("dense request states require a trusted query suffix")
    states = tuple(
        SharedRequestRestoreState(
            request_index=int(request.request_index),
            flat_row_offset=int(request.flat_row_start),
            row_count=int(request.row_count),
            cpu_plan=_DenseSharedRestorePlan(
                dirty_output_rows=tuple(range(int(request.row_count))),
                # No shared artifact is read or committed by this bypass, so
                # layer compression metadata is intentionally absent.
                layers=MappingProxyType({}),
            ),
            pin=None,
            schedule=None,
            prepared=None,
            reusable=False,
        )
        for request in geometry.requests
    )
    for state in states:
        state.validate()
    if tuple(row for state in states for row in state.dirty_rows) != tuple(
        range(int(geometry.source.q_rows))
    ):
        raise ValueError("query-suffix dense states do not span the forward")
    return states


def _geometry_template_request_token(
    plan: Mapping[str, object], request_index: int
) -> str:
    """Return a content-bound request token that is stable across HTTP calls."""

    return _canonical_digest(
        {
            "request_index": int(request_index),
            "plan": _geometry_template_plan_payload(plan),
        }
    )


def _geometry_template_generation_probe(
    backend, plans: Sequence[object]
) -> Tuple[Tuple[object, ...], ...]:
    controller = backend._redknot_shared_latent_controller
    segment_hashes = sorted(
        {
            str(segment["seg_hash"])
            for plan in plans
            if isinstance(plan, Mapping)
            and plan.get("mode") == "restore"
            and bool(plan.get("reuse_mla_off", False))
            for segment in tuple(plan.get("segments", ()))
            if isinstance(segment, Mapping) and segment.get("seg_hash")
        }
    )
    probe = []
    for seg_hash in segment_hashes:
        artifact = controller.get_committed(seg_hash)
        probe.append(
            (
                seg_hash,
                int(artifact.commit_epoch),
                str(artifact.token_hash),
                str(artifact.spec.model_hash),
                str(artifact.spec.policy_hash),
            )
        )
    return tuple(probe)


def _geometry_template_key(
    *,
    plans: Sequence[object],
    lengths: Sequence[int],
    scheduler_totals: Sequence[Optional[int]],
    scheduler_extents: Sequence[Optional[int]],
    q_row_count: int,
    input_layout_digest: Tuple[int, int],
    shared_generation_probe: Sequence[Tuple[object, ...]],
) -> str:
    return _canonical_digest(
        {
            "schema": "redknot_composite_geometry_template_v1",
            "plans": _geometry_template_plan_payload(tuple(plans)),
            "lengths": [int(value) for value in lengths],
            "scheduler_totals": [
                None if value is None else int(value)
                for value in scheduler_totals
            ],
            "scheduler_extents": [
                None if value is None else int(value)
                for value in scheduler_extents
            ],
            "q_rows": int(q_row_count),
            # Shape/plan alone is not a safe template identity: query suffixes
            # from different requests commonly have identical lengths and
            # positions but different token IDs.  Bind the scheduler-owned
            # positions and token values so those requests become independent
            # cache misses instead of tripping the fail-closed collision check.
            "input_layout_digest": [
                int(input_layout_digest[0]),
                int(input_layout_digest[1]),
            ],
            "shared_generations": [list(value) for value in shared_generation_probe],
        }
    )


def _current_benchmark_request_id(
    plans: Sequence[object], geometry_digest: str
) -> str:
    active_ids = tuple(
        str(plan.get("benchmark_request_id"))
        for plan in plans
        if isinstance(plan, Mapping) and plan.get("benchmark_request_id")
    )
    if len(active_ids) == 1:
        return active_ids[0]
    if active_ids:
        return "ragged-" + hashlib.sha256(
            repr(active_ids).encode("utf-8")
        ).hexdigest()[:16]
    return f"ragged-{geometry_digest[7:23]}"


def _clone_geometry_template_for_forward(
    *,
    backend,
    template: CompositeForwardGeometry,
    source: CompositeGeometrySourceBinding,
    plans: Tuple[object, ...],
    scheduler_totals: Tuple[Optional[int], ...],
    scheduler_extents: Tuple[Optional[int], ...],
    forward_batch,
) -> CompositeForwardGeometry:
    template.validate_cached()
    mode_name = getattr(backend, "_mla_off_forward_mode_name", None)
    if callable(mode_name):
        benchmark_forward_mode = str(mode_name(forward_batch))
    else:
        raw_mode = getattr(forward_batch, "forward_mode", None)
        benchmark_forward_mode = str(
            getattr(raw_mode, "name", "") or raw_mode or "unknown"
        ).lower()
    benchmark_request_id = _current_benchmark_request_id(
        plans, template.geometry_digest
    )
    benchmark_forward_id = "f" + hashlib.sha256(
        repr(
            (
                benchmark_request_id,
                benchmark_forward_mode,
                int(source.q_rows),
                template.request_token_digests,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    forward_id = _composite_serving_forward_id(
        plan_digest=source.plan_digest,
        geometry_digest=template.geometry_digest,
        request_token_digests=template.request_token_digests,
        scheduler_totals=scheduler_totals,
        scheduler_extents=scheduler_extents,
        benchmark_forward_mode=benchmark_forward_mode,
        q_row_count=int(source.q_rows),
    )
    result = CompositeForwardGeometry(
        source=source,
        validated_plans=plans,
        positions_cpu=template.positions_cpu,
        input_ids_cpu=template.input_ids_cpu,
        scheduler_totals=scheduler_totals,
        scheduler_extents=scheduler_extents,
        requests=template.requests,
        restore_rows=template.restore_rows,
        reusable_cpu=template.reusable_cpu,
        dirty_rows_cpu=template.dirty_rows_cpu,
        dirty_rows=template.dirty_rows,
        reused_count=template.reused_count,
        reuse_digest=template.reuse_digest,
        input_layout_digest=template.input_layout_digest,
        batch_token=template.batch_token,
        geometry_digest=template.geometry_digest,
        request_token_digests=template.request_token_digests,
        shared_generation_probe=template.shared_generation_probe,
        forward_id=forward_id,
        benchmark_request_id=benchmark_request_id,
        benchmark_forward_id=benchmark_forward_id,
        benchmark_forward_mode=benchmark_forward_mode,
        expected_layer_ids=template.expected_layer_ids,
        intentional_full_local_reason=template.intentional_full_local_reason,
        diagnostic_ablation=template.diagnostic_ablation,
    )
    prior_plan = template.batched_plan_template
    if prior_plan is not None:
        result.install_batched_plan_template(prior_plan)
    prior_token_rows = template._token_rows_binding_identities
    if prior_token_rows is not None:
        result._token_rows_binding_identities = prior_token_rows
    prior_shared_plans = template.shared_restore_plan_templates
    if prior_shared_plans is not None:
        result.install_shared_restore_plan_templates(prior_shared_plans)
    return result


def _build_forward_composite_geometry(
    backend,
    *,
    source: CompositeGeometrySourceBinding,
    plans: Tuple[object, ...],
    positions_cpu: torch.Tensor,
    input_ids_cpu: torch.Tensor,
    scheduler_totals: Tuple[Optional[int], ...],
    scheduler_extents: Tuple[Optional[int], ...],
    lengths: Tuple[int, ...],
    q_row_count: int,
    forward_batch,
) -> CompositeForwardGeometry:
    """Build the only CPU row geometry allocated by a composite forward."""

    resolved_diagnostics = tuple(
        (plan, resolve_mla_off_diagnostic_ablation(plan))
        for plan in plans
        if isinstance(plan, Mapping)
    )
    diagnostic_modes = tuple(
        diagnostic_mode
        for plan, diagnostic_mode in resolved_diagnostics
        if plan.get("mode") == "restore"
        and bool(plan.get("reuse_mla_off", False))
    )
    if not diagnostic_modes or len(set(diagnostic_modes)) != 1:
        raise ValueError(
            "composite restore requests must use one diagnostic ablation mode"
        )
    diagnostic_ablation = diagnostic_modes[0]
    cache_enabled = bool(
        getattr(backend, "_redknot_geometry_template_cache_enabled", False)
    )
    input_layout_digest = backend._mla_off_tensors_digest(
        positions_cpu, input_ids_cpu
    )
    shared_generation_probe = ()
    template_key = ""
    if cache_enabled:
        shared_generation_probe = _geometry_template_generation_probe(
            backend, plans
        )
        template_key = _geometry_template_key(
            plans=plans,
            lengths=lengths,
            scheduler_totals=scheduler_totals,
            scheduler_extents=scheduler_extents,
            q_row_count=int(q_row_count),
            input_layout_digest=input_layout_digest,
            shared_generation_probe=shared_generation_probe,
        )
        cache = getattr(backend, "_redknot_geometry_template_cache", None)
        if cache is None:
            cache = {}
            backend._redknot_geometry_template_cache = cache
        template = cache.get(template_key)
        if template is not None:
            if not isinstance(template, CompositeForwardGeometry):
                raise TypeError("geometry template cache contains a foreign value")
            if tuple(template.shared_generation_probe) != tuple(
                shared_generation_probe
            ):
                raise ValueError("geometry template artifact generation changed")
            if (
                tuple(int(value) for value in template.positions_cpu.shape)
                != tuple(int(value) for value in positions_cpu.shape)
                or not torch.equal(template.positions_cpu, positions_cpu)
                or tuple(int(value) for value in template.input_ids_cpu.shape)
                != tuple(int(value) for value in input_ids_cpu.shape)
                or not torch.equal(template.input_ids_cpu, input_ids_cpu)
            ):
                raise ValueError(
                    "geometry template key matched different scheduler tokens"
                )
            result = _clone_geometry_template_for_forward(
                backend=backend,
                template=template,
                source=source,
                plans=plans,
                scheduler_totals=scheduler_totals,
                scheduler_extents=scheduler_extents,
                forward_batch=forward_batch,
            )
            counter = getattr(backend, "_count", None)
            if callable(counter):
                counter("composite_geometry_template_cache_hits")
            return result
    reusable = torch.zeros(int(q_row_count), dtype=torch.bool)
    requests = []
    all_restore_rows = []
    offset = 0
    for request_index, (plan, row_count) in enumerate(zip(plans, lengths)):
        request_positions = positions_cpu.narrow(0, offset, row_count)
        request_tokens = input_ids_cpu.narrow(0, offset, row_count)
        logical_positions = tuple(int(value) for value in request_positions.tolist())
        input_token_ids = tuple(int(value) for value in request_tokens.tolist())
        active_restore = bool(
            isinstance(plan, Mapping)
            and plan.get("mode") == "restore"
            and bool(plan.get("reuse_mla_off", False))
        )
        if not active_restore:
            fallback_reason = (
                str(
                    plan.get(
                        "_redknot_composite_fallback_reason",
                        "request_not_in_composite_restore",
                    )
                )
                if isinstance(plan, Mapping)
                else "request_not_in_composite_restore"
            )
            requests.append(
                _CachedRequestGeometry(
                    request_index=int(request_index),
                    flat_row_start=int(offset),
                    row_count=int(row_count),
                    request_token=(
                        str(plan.get("benchmark_request_id"))
                        if isinstance(plan, Mapping)
                        and plan.get("benchmark_request_id")
                        else f"dense-request:{request_index}"
                    ),
                    logical_positions=logical_positions,
                    input_token_ids=input_token_ids,
                    query_start=0,
                    dirty_request_rows=tuple(range(int(row_count))),
                    segments=(),
                    segment_metadata=(),
                    restore_bindings=(),
                    fallback_reason=fallback_reason,
                )
            )
            offset += row_count
            continue

        assert isinstance(plan, Mapping)
        restore_rows, request_reusable = build_restore_rows(
            plan=plan,
            positions_cpu=request_positions,
            refresh_layer=False,
        )
        segment_metadata = tuple(
            MappingProxyType(dict(segment))
            for segment in tuple(plan.get("segments", ()))
        )
        segments_by_hash = {
            str(segment["seg_hash"]): segment for segment in segment_metadata
        }
        if not segments_by_hash:
            raise ValueError("MLA-off restore needs segment metadata per request")
        restore_bindings = []
        for rows in restore_rows:
            if rows.seg_hash not in segments_by_hash:
                raise ValueError("MLA-off restore row refers to an unknown segment")
            local_output = rows.output_rows_cpu
            shifted = MLAOffRestoreRows(
                seg_hash=rows.seg_hash,
                output_rows_cpu=local_output + int(offset),
                local_positions_cpu=rows.local_positions_cpu,
            )
            restore_bindings.append(
                _CachedRestoreBinding(
                    rows=shifted,
                    token_ids_cpu=request_tokens.index_select(0, local_output),
                )
            )
            all_restore_rows.append(shifted)
        reusable[offset : offset + row_count].copy_(request_reusable)
        dirty_local = torch.nonzero(~request_reusable, as_tuple=False).flatten()
        requests.append(
            _CachedRequestGeometry(
                request_index=int(request_index),
                flat_row_start=int(offset),
                row_count=int(row_count),
                request_token=(
                    _geometry_template_request_token(plan, int(request_index))
                    if cache_enabled
                    else str(
                        plan.get(
                            "benchmark_request_id", f"request:{request_index}"
                        )
                    )
                ),
                logical_positions=logical_positions,
                input_token_ids=input_token_ids,
                query_start=int(plan["query_start"]),
                dirty_request_rows=tuple(int(value) for value in dirty_local.tolist()),
                segments=tuple(
                    SegmentBinding(
                        segment_token=str(segment["seg_hash"]),
                        logical_start=int(segment["global_offset"]),
                        length=int(segment["length"]),
                    )
                    for segment in segment_metadata
                ),
                segment_metadata=segment_metadata,
                restore_bindings=tuple(restore_bindings),
            )
        )
        offset += row_count

    dirty_rows_cpu = torch.nonzero(~reusable, as_tuple=False).flatten()
    reused_count = int(reusable.sum().item())
    expected_dirty = tuple(int(value) for value in dirty_rows_cpu.tolist())
    request_tuple = tuple(requests)
    intentional_full_local_reason = ""
    if reused_count <= 0:
        intentional_full_local_reason = _intentional_full_local_reason(
            request_tuple,
            q_row_count=int(q_row_count),
            reused_count=reused_count,
        )
        if not intentional_full_local_reason:
            raise ValueError("composite forward geometry has no reusable row")
    reuse_digest = backend._mla_off_tensors_digest(
        positions_cpu, reusable, input_ids_cpu
    )
    batch_token = "sha256:" + hashlib.sha256(
        repr((tuple(lengths), reuse_digest)).encode("utf-8")
    ).hexdigest()
    global_rows = tuple(range(int(q_row_count)))
    dirty_set = set(expected_dirty)
    geometry_digest = _canonical_digest(
        {
            "q_rows": int(q_row_count),
            "global_rows": list(global_rows),
            "local_clean_rows": [row for row in global_rows if row not in dirty_set],
            "local_dirty_rows": list(expected_dirty),
            "requests": [
                {
                    "request_index": request.request_index,
                    "request_token": request.request_token,
                    "flat_row_start": request.flat_row_start,
                    "logical_positions": list(request.logical_positions),
                    "query_start": request.query_start,
                    "dirty_request_rows": list(request.dirty_request_rows),
                }
                for request in request_tuple
            ],
        }
    )
    request_token_digests = tuple(
        _canonical_digest(
            {
                "positions": list(request.logical_positions),
                "token_ids": list(request.input_token_ids),
            }
        )
        for request in request_tuple
    )
    if not cache_enabled:
        # Preserve the production control-flow and request identity contract
        # when the explicit optimization is disabled.  Cache mode probes the
        # same committed generations earlier because they are part of its key.
        shared_generation_probe = _geometry_template_generation_probe(
            backend, plans
        )
    mode_name = getattr(backend, "_mla_off_forward_mode_name", None)
    if callable(mode_name):
        benchmark_forward_mode = str(mode_name(forward_batch))
    else:
        raw_mode = getattr(forward_batch, "forward_mode", None)
        benchmark_forward_mode = str(
            getattr(raw_mode, "name", "") or raw_mode or "unknown"
        ).lower()
    benchmark_request_id = _current_benchmark_request_id(
        plans, geometry_digest
    )
    benchmark_forward_id = "f" + hashlib.sha256(
        repr(
            (
                benchmark_request_id,
                benchmark_forward_mode,
                int(q_row_count),
                request_token_digests,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    forward_id = _composite_serving_forward_id(
        plan_digest=source.plan_digest,
        geometry_digest=geometry_digest,
        request_token_digests=request_token_digests,
        scheduler_totals=scheduler_totals,
        scheduler_extents=scheduler_extents,
        benchmark_forward_mode=benchmark_forward_mode,
        q_row_count=int(q_row_count),
    )
    result = CompositeForwardGeometry(
        source=source,
        validated_plans=plans,
        positions_cpu=positions_cpu,
        input_ids_cpu=input_ids_cpu,
        scheduler_totals=scheduler_totals,
        scheduler_extents=scheduler_extents,
        requests=request_tuple,
        restore_rows=tuple(all_restore_rows),
        reusable_cpu=reusable,
        dirty_rows_cpu=dirty_rows_cpu,
        dirty_rows=expected_dirty,
        reused_count=reused_count,
        reuse_digest=reuse_digest,
        input_layout_digest=input_layout_digest,
        batch_token=batch_token,
        geometry_digest=geometry_digest,
        request_token_digests=request_token_digests,
        shared_generation_probe=tuple(shared_generation_probe),
        forward_id=forward_id,
        benchmark_request_id=benchmark_request_id,
        benchmark_forward_id=benchmark_forward_id,
        benchmark_forward_mode=benchmark_forward_mode,
        expected_layer_ids=tuple(
            int(value) for value in backend._redknot_mla_off_rank_local_layer_ids
        ),
        intentional_full_local_reason=intentional_full_local_reason,
        diagnostic_ablation=diagnostic_ablation,
    )
    if cache_enabled:
        cache = backend._redknot_geometry_template_cache
        cache[template_key] = result
        while len(cache) > _GEOMETRY_TEMPLATE_CACHE_MAX_ENTRIES:
            del cache[next(iter(cache))]
        counter = getattr(backend, "_count", None)
        if callable(counter):
            counter("composite_geometry_template_cache_misses")
    return result


def _bind_layer_composite_geometry(
    backend,
    *,
    geometry: CompositeForwardGeometry,
    z_controller: object,
    spec: object,
) -> Tuple[Tuple[object, ...], Tuple[object, ...], object]:
    """Attach one layer's persistent views to cached forward row geometry."""

    all_bindings = []
    token_row_bindings = []
    request_layouts = []
    artifact_bindings = []
    template = geometry.batched_plan_template
    build_template = template is None
    for request in geometry.requests:
        if not request.reusable:
            artifact_bindings.append(None)
            if build_template:
                request_layouts.append(
                    RequestReuseLayout(
                        request_index=request.request_index,
                        request_token=request.request_token,
                        flat_row_start=request.flat_row_start,
                        logical_positions=request.logical_positions,
                        query_start=0,
                        dirty_request_rows=request.dirty_request_rows,
                        fallback_reason=request.fallback_reason,
                    )
                )
            continue
        artifact_views = {}
        for segment in request.segment_metadata:
            seg_hash = str(segment["seg_hash"])
            artifact_views[seg_hash] = z_controller.prepare_restore_view(
                seg_hash=seg_hash,
                length=int(segment["length"]),
                spec=spec,
                token_hash=str(segment.get("token_hash", seg_hash)),
            )
        for binding in request.restore_bindings:
            rows = binding.rows
            view = artifact_views.get(rows.seg_hash)
            if view is None:
                raise ValueError("cached restore row refers to an unknown segment")
            token_row_bindings.append((view, rows, binding.token_ids_cpu))
            all_bindings.append((view, rows))
        artifact_binding = (
            _server_artifact_token(tuple(artifact_views.values())),
            max(int(view.commit_epoch) for view in artifact_views.values()),
            str(backend._redknot_mla_off_policy_hash),
        )
        artifact_bindings.append(artifact_binding)
        if build_template:
            request_layouts.append(
                RequestReuseLayout(
                    request_index=request.request_index,
                    request_token=request.request_token,
                    flat_row_start=request.flat_row_start,
                    logical_positions=request.logical_positions,
                    query_start=request.query_start,
                    dirty_request_rows=request.dirty_request_rows,
                    segments=request.segments,
                    artifact_token=artifact_binding[0],
                    artifact_epoch=artifact_binding[1],
                    policy_digest=artifact_binding[2],
                )
            )
    geometry.validate_or_install_token_rows_bindings(
        z_controller,
        tuple(token_row_bindings),
    )
    if build_template:
        batch_plan = assemble_validated_batched_reuse_plan(
            batch_token=geometry.batch_token,
            requests=tuple(request_layouts),
        )
        if _batched_geometry_digest(batch_plan) != geometry.geometry_digest:
            raise ValueError("layer artifact binding changed cached ragged geometry")
        if batch_plan.local_dirty_rows != geometry.dirty_rows:
            raise ValueError("layer artifact binding changed cached dirty rows")
        geometry.install_batched_plan_template(batch_plan)
    else:
        batch_plan = rebind_validated_batched_reuse_plan(
            template=template,
            artifact_bindings=tuple(artifact_bindings),
        )
        if (
            batch_plan.global_rows is not template.global_rows
            or batch_plan.local_clean_rows is not template.local_clean_rows
            or batch_plan.local_dirty_rows is not template.local_dirty_rows
            or batch_plan.local_dirty_rows != geometry.dirty_rows
        ):
            raise ValueError("artifact rebind changed cached ragged row geometry")
    return geometry.restore_rows, tuple(all_bindings), batch_plan


def _prepare_composite_restore_context_impl(
    backend,
    *,
    layer_id: int,
    positions: torch.Tensor,
    forward_batch,
    q_row_count: int,
    n_local_heads: int,
    n_local_groups: int,
    head_dim: int,
    o_lora_rank: int,
    device: torch.device,
    projection_dtype: torch.dtype,
):
    """Return a v3 context or ``None`` when this is not a restore forward."""

    raw_plans = getattr(forward_batch, "redknot_reuse_plan", None)
    if raw_plans is None or len(raw_plans) != int(forward_batch.batch_size):
        return None
    plans = tuple(raw_plans)
    if not any(
        isinstance(plan, Mapping)
        and plan.get("mode") == "restore"
        and bool(plan.get("reuse_mla_off", False))
        for plan in plans
    ):
        return None
    input_ids = getattr(forward_batch, "input_ids", None)
    if not isinstance(input_ids, torch.Tensor) or int(input_ids.numel()) != int(q_row_count):
        raise ValueError("composite restore requires one token id per packed row")
    lengths = _split_lengths(forward_batch, int(q_row_count))
    forward_generation = getattr(
        forward_batch, "_redknot_mla_off_forward_generation", None
    )
    resources = getattr(
        forward_batch, "_redknot_composite_forward_resources", None
    )
    if resources is not None and not isinstance(
        resources, CompositeForwardResources
    ):
        raise RuntimeError("ForwardBatch composite resource slot has a foreign type")

    geometry = None
    if resources is not None:
        if resources.closed:
            raise RuntimeError("closed composite resources cannot rebuild geometry")
        geometry = resources.geometry
        if not isinstance(geometry, CompositeForwardGeometry):
            raise RuntimeError("composite resource lease has no geometry cache")
        # A hit performs no D2H, tensor value digest, restore-row build, or
        # Python conversion of row/token payloads.  Any source drift aborts the
        # existing lease; it is never treated as a cache miss within a forward.
        geometry.source.validate_live(
            forward_generation=forward_generation,
            raw_plans=raw_plans,
            positions=positions,
            input_ids=input_ids,
            ragged_lengths=lengths,
            ragged_source=getattr(forward_batch, "extend_seq_lens_cpu", None),
            scheduler_totals_source=getattr(forward_batch, "orig_seq_lens", None),
            scheduler_extents_source=getattr(forward_batch, "seq_lens_cpu", None),
            batch_size=int(forward_batch.batch_size),
            q_rows=int(q_row_count),
        )
        geometry.validate_cached()
        resources.validate(
            forward_key=geometry.forward_key,
            batch_digest=geometry.geometry_digest,
        )
        plans = geometry.validated_plans
        positions_cpu = geometry.positions_cpu
        input_ids_cpu = geometry.input_ids_cpu
        scheduler_totals = geometry.scheduler_totals
    else:
        expected_layers = tuple(
            int(value) for value in backend._redknot_mla_off_rank_local_layer_ids
        )
        if not expected_layers or int(layer_id) != expected_layers[0]:
            raise RuntimeError(
                "composite geometry may only be created by the first reusable layer"
            )
        source = CompositeGeometrySourceBinding.capture(
            forward_generation=forward_generation,
            raw_plans=raw_plans,
            positions=positions,
            input_ids=input_ids,
            ragged_lengths=lengths,
            ragged_source=getattr(forward_batch, "extend_seq_lens_cpu", None),
            scheduler_totals_source=getattr(forward_batch, "orig_seq_lens", None),
            scheduler_extents_source=getattr(forward_batch, "seq_lens_cpu", None),
            batch_size=int(forward_batch.batch_size),
            q_rows=int(q_row_count),
        )
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long)
        input_ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.long)
        scheduler_totals = _scheduler_request_total_tokens(
            forward_batch,
            expected_requests=int(forward_batch.batch_size),
        )
        scheduler_extents = _scheduler_request_current_extents(
            forward_batch,
            expected_requests=int(forward_batch.batch_size),
        )
        request_positions_by_request = _split_request_positions_cpu(
            positions_cpu,
            lengths=lengths,
            total_rows=int(q_row_count),
        )
        validated_plans = []
        valid_restore_requests = 0
        preflight_fallbacks = []
        context_stream_phases = []
        request_token_offset = 0
        for request_index, plan in enumerate(plans):
            request_positions = request_positions_by_request[request_index]
            request_row_count = int(lengths[request_index])
            request_tokens = input_ids_cpu.narrow(
                0, request_token_offset, request_row_count
            )
            request_token_offset += request_row_count
            active_restore = bool(
                isinstance(plan, Mapping)
                and plan.get("mode") == "restore"
                and bool(plan.get("reuse_mla_off", False))
            )
            if not active_restore:
                context_stream_phases.append("dense")
                validated_plans.append(plan)
                continue
            assert isinstance(plan, Mapping)
            fallback_reason = _active_restore_plan_fallback_reason(
                backend,
                plan=plan,
                scheduler_total_tokens=scheduler_totals[request_index],
                scheduler_current_extent=scheduler_extents[request_index],
                request_positions=request_positions,
                original_chunk_range=getattr(
                    forward_batch, "redknot_original_chunk_token_range", None
                ),
            )
            if not fallback_reason:
                try:
                    execution_profile = str(
                        plan.get("mla_off_execution_profile", "")
                    )
                    independent_relocation = execution_profile == (
                        _INDEPENDENT_RELOCATION_PROFILE
                    )
                    combined_row_sparse = execution_profile == (
                        _COMBINED_ROW_SPARSE_PROFILE
                    )
                    cached_prefixes = getattr(
                        forward_batch,
                        "redknot_cached_prefix_input_ids",
                        None,
                    )
                    trusted_cached_prefix = ()
                    if plan.get("radix_prefix_role") == "consume":
                        if (
                            not isinstance(cached_prefixes, (tuple, list))
                            or len(cached_prefixes)
                            != int(forward_batch.batch_size)
                        ):
                            raise ValueError(
                                "radix-prefix restore lacks scheduler-owned prefix ids"
                            )
                        trusted_cached_prefix = cached_prefixes[request_index]
                        if not isinstance(trusted_cached_prefix, tuple):
                            raise ValueError(
                                "radix-prefix restore did not hit its complete prefix"
                            )
                    if combined_row_sparse:
                        context_phase = _validate_combined_sparse_restore_chunk(
                            plan=plan,
                            request_positions=tuple(request_positions),
                            request_tokens=tuple(
                                int(value) for value in request_tokens.tolist()
                            ),
                            original_chunk_range=getattr(
                                forward_batch,
                                "redknot_original_chunk_token_range",
                                None,
                            ),
                            trusted_cached_prefix_tokens=trusted_cached_prefix,
                        )
                    elif independent_relocation:
                        context_phase = _validate_independent_restore_chunk(
                            plan=plan,
                            request_positions=tuple(request_positions),
                            request_tokens=tuple(
                                int(value) for value in request_tokens.tolist()
                            ),
                            trusted_cached_prefix_tokens=trusted_cached_prefix,
                        )
                    else:
                        binding_builder = getattr(
                            backend, "_mla_off_context_request_binding", None
                        )
                        registry = getattr(
                            backend, "_redknot_context_token_streams", None
                        )
                        if not callable(binding_builder) or registry is None:
                            raise RuntimeError(
                                "context stream request binding/registry is unavailable"
                            )
                        request_binding = binding_builder(
                            forward_batch=forward_batch,
                            plan=plan,
                            request_index=int(request_index),
                        )
                        request_id = str(
                            plan.get(
                                "benchmark_request_id",
                                getattr(forward_batch, "rids", [""])[request_index],
                            )
                        )
                        context_phase = registry.observe_restore_chunk(
                            request_id=request_id,
                            request_binding=request_binding,
                            plan=plan,
                            positions=tuple(request_positions),
                            token_ids=tuple(
                                int(value) for value in request_tokens.tolist()
                            ),
                            scheduler_total=scheduler_totals[request_index],
                            scheduler_current_extent=scheduler_extents[request_index],
                            trusted_cached_prefix_tokens=trusted_cached_prefix,
                        )
                    if context_phase not in (
                        "segment",
                        "segments",
                        "suffix",
                        "suffix_complete",
                    ):
                        raise ValueError(
                            f"unexpected context restore phase {context_phase!r}"
                        )
                except Exception as exc:
                    fallback_reason = (
                        "context_input_certificate:"
                        + type(exc).__name__
                        + ":"
                        + str(exc)
                    )
                    context_phase = "invalid"
            else:
                context_phase = "invalid"
            context_stream_phases.append(context_phase)
            if fallback_reason:
                preflight_fallbacks.append(
                    (int(request_index), str(fallback_reason))
                )
                validated_plans.append(
                    _dense_placeholder_for_invalid_restore(
                        plan,
                        reason=fallback_reason,
                    )
                )
                continue
            valid_restore_requests += 1
            validated_plans.append(plan)
        if request_token_offset != int(q_row_count):
            raise ValueError("context stream tokens do not tile the packed forward")
        forward_batch._redknot_context_restore_phases = tuple(
            context_stream_phases
        )
        # Keep request-scoped, token-free evidence on the ForwardBatch.  TP
        # preflight logs can remain low-noise while a failed aggregate vote is
        # still diagnosable after the fact.
        forward_batch._redknot_composite_preflight_fallbacks = tuple(
            preflight_fallbacks
        )
        plans = tuple(validated_plans)
        if valid_restore_requests <= 0:
            return None
        geometry = _build_forward_composite_geometry(
            backend,
            source=source,
            plans=plans,
            positions_cpu=positions_cpu,
            input_ids_cpu=input_ids_cpu,
            scheduler_totals=scheduler_totals,
            scheduler_extents=scheduler_extents,
            lengths=lengths,
            q_row_count=int(q_row_count),
            forward_batch=forward_batch,
        )
    with _redknot_runtime_timed(
        "mla_context_layer_spec", layer_id=int(layer_id)
    ):
        spec, local_axes = _spec_for_layer(
            backend,
            layer_id=int(layer_id),
            n_local_heads=int(n_local_heads),
            n_local_groups=int(n_local_groups),
            head_dim=int(head_dim),
            o_lora_rank=int(o_lora_rank),
        )
    z_controller = get_dsv4_mla_off_controller()
    if geometry.is_intentional_full_local:
        # Query/new-token suffixes have no offline row to bind.  Keep a
        # forward-scoped, pin-free lease so every middle layer observes the
        # same certified geometry, but bypass all z_off/shared-KV artifact and
        # device-index preparation.  The model will execute its ordinary dense
        # Q/KV/Indexer/Compressor and native attention paths.
        try:
            if resources is None:
                shared_states = _query_suffix_dense_request_states(geometry)
                resources = CompositeForwardResources(
                    forward_key=geometry.forward_key,
                    forward_id=geometry.forward_id,
                    batch_digest=geometry.geometry_digest,
                    total_rows=int(q_row_count),
                    request_token_digests=geometry.request_token_digests,
                    shared_states=shared_states,
                    expected_layer_ids=geometry.expected_layer_ids,
                    geometry=geometry,
                )
                forward_batch._redknot_composite_forward_resources = resources
            else:
                shared_states = resources.shared_states
                if any(state.reusable for state in shared_states) or tuple(
                    row for state in shared_states for row in state.dirty_rows
                ) != tuple(range(int(q_row_count))):
                    raise ValueError(
                        "query-suffix lease contains reusable artifact state"
                    )
            context = MLAOffRuntimeContext(
                mode="full_local",
                layer_id=int(layer_id),
                spec=spec,
                local_head_axes=local_axes,
                controller=z_controller,
                input_layout_digest=geometry.input_layout_digest,
                benchmark_request_id=geometry.benchmark_request_id,
                benchmark_forward_id=geometry.benchmark_forward_id,
                benchmark_forward_mode=geometry.benchmark_forward_mode,
                diagnostic_ablation=geometry.diagnostic_ablation,
                reused_row_count=0,
                online_local_row_count=int(q_row_count),
                benchmark_q_rows=int(q_row_count),
                intentional_full_local_reason=(
                    geometry.intentional_full_local_reason
                ),
            )
            context._redknot_composite_forward_resources = resources
            context._redknot_forward_batch_owner = forward_batch
            resources.register_context(int(layer_id), context)
            return context
        except BaseException:
            if isinstance(resources, CompositeForwardResources):
                try:
                    resources.close()
                except Exception:
                    pass
                if getattr(
                    forward_batch, "_redknot_composite_forward_resources", None
                ) is resources:
                    forward_batch._redknot_composite_forward_resources = None
            raise
    with _redknot_runtime_timed(
        "mla_context_bind_artifacts", layer_id=int(layer_id)
    ):
        all_restore_rows, all_bindings, batch_plan = (
            _bind_layer_composite_geometry(
                backend,
                geometry=geometry,
                z_controller=z_controller,
                spec=spec,
            )
        )
    reusable = geometry.reusable_cpu
    dirty_rows_cpu = geometry.dirty_rows_cpu
    reused_count = geometry.reused_count
    reuse_digest = geometry.reuse_digest
    input_layout_digest = geometry.input_layout_digest
    expected_dirty = batch_plan.local_dirty_rows
    forward_id = geometry.forward_id
    benchmark_request_id = geometry.benchmark_request_id
    benchmark_forward_id = geometry.benchmark_forward_id
    benchmark_forward_mode = geometry.benchmark_forward_mode
    forward_key = geometry.forward_key
    geometry_digest = geometry.geometry_digest
    request_token_digests = geometry.request_token_digests
    if resources is None:
        shared_states = _prepare_shared_request_states(
            backend,
            geometry=geometry,
            plans=plans,
            positions_cpu=positions_cpu,
            input_ids_cpu=input_ids_cpu,
            lengths=lengths,
            device=torch.device(device),
            forward_id=forward_id,
        )
        shared_dirty = tuple(
            row for state in shared_states for row in state.dirty_rows
        )
        if shared_dirty != expected_dirty:
            _close_shared_request_states(shared_states)
            raise ValueError("shared KV and z_off dirty-row domains differ")
        new_resources = None
        try:
            new_resources = CompositeForwardResources(
                forward_key=forward_key,
                forward_id=forward_id,
                batch_digest=geometry_digest,
                total_rows=int(q_row_count),
                request_token_digests=tuple(request_token_digests),
                shared_states=tuple(shared_states),
                expected_layer_ids=geometry.expected_layer_ids,
                geometry=geometry,
            )
            forward_batch._redknot_composite_forward_resources = new_resources
        except Exception:
            if new_resources is not None:
                try:
                    new_resources.close()
                except Exception:
                    pass
            raise
        resources = new_resources
    else:
        shared_states = resources.shared_states
        shared_dirty = tuple(
            row for state in shared_states for row in state.dirty_rows
        )
        if shared_dirty != expected_dirty:
            try:
                resources.close()
            finally:
                forward_batch._redknot_composite_forward_resources = None
            raise ValueError("shared KV and z_off dirty-row domains differ")

    try:
        with _redknot_runtime_timed(
            "mla_context_bind_projection", layer_id=int(layer_id)
        ):
            persistent_plan = z_controller.bind_persistent_projection_plan(
                bindings=tuple(all_bindings),
                total_rows=int(q_row_count),
                spec=spec,
                device=torch.device(device),
                dtype=projection_dtype,
            )
        from sglang.srt.layers.attention.redknot_mla_backend import (
            _MLAOffRestoreLayoutCertificate,
            _mla_off_control_tensor_identity,
        )

        layout_key = (
            forward_id,
            geometry.source.plan_digest,
            geometry.geometry_digest,
            reuse_digest,
        )
        layout = geometry.layout_certificate
        if layout is None:
            layout = _MLAOffRestoreLayoutCertificate(
                layout_key=layout_key,
                certified_layer_ids=geometry.expected_layer_ids,
                restore_rows=geometry.restore_rows,
                reusable_cpu=reusable,
                reuse_mask_digest=reuse_digest,
                reused_count=reused_count,
                dirty_rows_cpu=dirty_rows_cpu,
                segments_by_hash=MappingProxyType(
                    {
                        str(segment["seg_hash"]): segment
                        for request in geometry.requests
                        if request.reusable
                        for segment in request.segment_metadata
                    }
                ),
                reusable_identity=_mla_off_control_tensor_identity(reusable),
                dirty_identity=_mla_off_control_tensor_identity(dirty_rows_cpu),
            )
            geometry.install_layout_certificate(layout)
        elif not isinstance(layout, _MLAOffRestoreLayoutCertificate):
            raise TypeError("cached composite restore layout has a foreign type")
        with _redknot_runtime_timed(
            "mla_context_validate_layout", layer_id=int(layer_id)
        ):
            layout.validate(
                layer_id=int(layer_id),
                layout_key=layout_key,
                reusable_cpu=reusable,
                dirty_rows_cpu=dirty_rows_cpu,
                reuse_mask_digest=reuse_digest,
                q_rows=int(q_row_count),
            )
        # Production attention checks object identity, not merely digest.
        forward_batch._redknot_mla_off_restore_layout = layout
        dirty_certificate = geometry.dirty_device_certificate
        dirty_device = geometry.dirty_device_indices
        if dirty_certificate is None and dirty_device is None:
            dirty_certificate = z_controller.prepare_device_indices(
                dirty_rows_cpu,
                device=torch.device(device),
                role="online_local_rows",
                semantic_digest=reuse_digest,
                upper_bound=int(q_row_count),
            )
            dirty_device = z_controller.device_indices_from_certificate(
                dirty_certificate,
                cpu_indices=dirty_rows_cpu,
                device=torch.device(device),
                role="online_local_rows",
                semantic_digest=reuse_digest,
                upper_bound=int(q_row_count),
            )
            geometry.install_dirty_device_indices(
                certificate=dirty_certificate,
                indices=dirty_device,
            )
        elif dirty_certificate is None or dirty_device is None:
            raise ValueError("cached dirty device index pair is incomplete")
        with _redknot_runtime_timed(
            "mla_context_validate_dirty_indices", layer_id=int(layer_id)
        ):
            dirty_device = z_controller.device_indices_from_certificate(
                dirty_certificate,
                cpu_indices=dirty_rows_cpu,
                device=torch.device(device),
                role="online_local_rows",
                semantic_digest=reuse_digest,
                upper_bound=int(q_row_count),
            )
        if dirty_device is not geometry.dirty_device_indices:
            raise ValueError("cached dirty device indices lost certificate identity")
        with _redknot_runtime_timed(
            "mla_context_publish", layer_id=int(layer_id)
        ):
            context = MLAOffRuntimeContext(
                mode="restore",
                layer_id=int(layer_id),
                spec=spec,
                local_head_axes=local_axes,
                controller=z_controller,
                reuse_mask=reusable,
                reuse_mask_digest=reuse_digest,
                online_local_row_indices=dirty_device,
                online_local_row_indices_cpu=dirty_rows_cpu,
                online_local_row_indices_certificate=dirty_certificate,
                restore_layout_certificate=layout,
                input_layout_digest=input_layout_digest,
                benchmark_request_id=benchmark_request_id,
                benchmark_forward_id=benchmark_forward_id,
                benchmark_forward_mode=benchmark_forward_mode,
                diagnostic_ablation=geometry.diagnostic_ablation,
                reused_row_count=reused_count,
                online_local_row_count=int(dirty_rows_cpu.numel()),
                benchmark_q_rows=int(q_row_count),
                batched_reuse_plan=batch_plan,
                # zoff_only deliberately retains the same persistent z_off lease
                # and row geometry while withholding shared cache states from the
                # model.  WKV/C4/C128/Indexer therefore execute all rows online.
                shared_restore_states=(
                    ()
                    if geometry.diagnostic_ablation
                    == MLA_OFF_DIAGNOSTIC_ABLATION_ZOFF_ONLY
                    else tuple(shared_states)
                ),
            )
            context.install_persistent_projection_plan(persistent_plan)
            context._redknot_composite_forward_resources = resources
            context._redknot_forward_batch_owner = forward_batch
            resources.register_context(int(layer_id), context)
        return context
    except Exception:
        try:
            resources.close()
        except Exception:
            # The lease attempts every pin before reporting a close error.  Do
            # not hide the context construction/registration failure.
            pass
        if getattr(
            forward_batch, "_redknot_composite_forward_resources", None
        ) is resources:
            forward_batch._redknot_composite_forward_resources = None
        if getattr(
            forward_batch, "_redknot_mla_off_restore_layout", None
        ) is locals().get("layout"):
            forward_batch._redknot_mla_off_restore_layout = None
        raise


def prepare_composite_restore_context(
    backend,
    *,
    layer_id: int,
    positions: torch.Tensor,
    forward_batch,
    q_row_count: int,
    n_local_heads: int,
    n_local_groups: int,
    head_dim: int,
    o_lora_rank: int,
    device: torch.device,
    projection_dtype: torch.dtype,
):
    """Prepare v3 and close the entire forward lease on any partial failure."""

    try:
        result = _prepare_composite_restore_context_impl(
            backend,
            layer_id=layer_id,
            positions=positions,
            forward_batch=forward_batch,
            q_row_count=q_row_count,
            n_local_heads=n_local_heads,
            n_local_groups=n_local_groups,
            head_dim=head_dim,
            o_lora_rank=o_lora_rank,
            device=device,
            projection_dtype=projection_dtype,
        )
        if result is not None:
            return result
    except Exception:
        resources = getattr(
            forward_batch, "_redknot_composite_forward_resources", None
        )
        if isinstance(resources, CompositeForwardResources):
            try:
                resources.close()
            except Exception:
                # Preserve the preparation error; close() already attempted
                # every request pin and marked the lease unusable.
                pass
            forward_batch._redknot_composite_forward_resources = None
        forward_batch._redknot_mla_off_restore_layout = None
        raise
    resources = getattr(
        forward_batch, "_redknot_composite_forward_resources", None
    )
    if isinstance(resources, CompositeForwardResources):
        try:
            resources.close()
        finally:
            forward_batch._redknot_composite_forward_resources = None
    return None


def release_composite_restore_context(
    context: MLAOffRuntimeContext,
    *,
    close_when_forward_complete: bool = True,
) -> bool:
    """Release one layer context; close pins after the last reusable layer."""

    resources = getattr(context, "_redknot_composite_forward_resources", None)
    if not isinstance(resources, CompositeForwardResources):
        raise ValueError("context has no composite forward resource lease")
    complete = resources.complete_context(context)
    if complete and bool(close_when_forward_complete):
        owner = getattr(context, "_redknot_forward_batch_owner", None)
        try:
            resources.close()
        finally:
            if owner is not None and getattr(
                owner, "_redknot_composite_forward_resources", None
            ) is resources:
                owner._redknot_composite_forward_resources = None
    return complete


def close_composite_forward_resources(
    forward_batch,
    *,
    expected_forward_key: Optional[str] = None,
) -> bool:
    """Explicit fail-closed cleanup for cancellation/error/end-of-forward."""

    resources = getattr(
        forward_batch, "_redknot_composite_forward_resources", None
    )
    if resources is None:
        return False
    if not isinstance(resources, CompositeForwardResources):
        raise RuntimeError("ForwardBatch composite resource slot has a foreign type")
    if expected_forward_key is not None and (
        str(expected_forward_key) != resources.forward_key
    ):
        raise ValueError("refusing to close another composite forward")
    try:
        resources.close()
    finally:
        forward_batch._redknot_composite_forward_resources = None
    return True


__all__ = [
    "CompositeForwardGeometry",
    "CompositeForwardResources",
    "CompositeGeometrySourceBinding",
    "ForwardCompositeCommitCoordinator",
    "LayerCacheDomainPreflight",
    "LayerCompositeCommitBuilder",
    "LayerCompositeCommitResult",
    "SharedRequestRestoreState",
    "begin_layer_composite_commit",
    "build_layer_composite_reservation",
    "close_composite_forward_resources",
    "merge_layer_composite_proposals",
    "prepare_composite_restore_context",
    "release_composite_restore_context",
]
