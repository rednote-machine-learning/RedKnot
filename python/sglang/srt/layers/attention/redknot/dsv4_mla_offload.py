# Copyright 2024-2026 SGLang RedKnot Integration.
"""Offline local-head output artifacts for DeepSeek-V4 MLA.

This module intentionally does not extend the packed SWA/C4/C128 KV snapshot
format.  DeepSeek-V4 has one physical latent KV stream but many logical query
heads.  The reusable artifact here is the local-head contribution after
inverse RoPE and the bias-free ``wo_a`` projection, and before the shared
``wo_b`` projection::

    z_off = wo_a(attention_output_from_local_heads)

At restore time layers 3..39 evaluate the online head partition over the native
DSV4 full candidate scope.  Every offline head row is restored only from an
artifact captured at the same absolute position after the exact same cumulative
token prefix. Query/new rows evaluate every head online. Their true sliced
``wo_a`` contributions are added to ``z_off`` and the existing ``wo_b`` is
evaluated once. Layers 0..2 and 40..42 never create or consume this artifact.

Artifacts are staged until all 37 reusable layers cover every segment row.
A partial or incompatible artifact is never exposed to restore.  CPU BF16 is
the authoritative context-bound v3 format.  An explicitly capacity-bounded, same-epoch BF16
device mirror may be captured at the same offline boundary so warm online
restore avoids copying the full projection through host memory.  FP8 or a
low-rank representation would require a separately versioned accuracy gate.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

try:
    from .dsv4_fused_z_merge import (
        PersistentHeadSplitWOAMergePlan,
        PersistentProjectionPlan,
        PersistentProjectionView,
        ProjectionSpanGeometry,
        build_persistent_projection_plan,
        merge_persistent_projection,
        project_merge_persistent_headsplit,
    )
except ImportError:  # pragma: no cover - direct-file CPU tests
    from dsv4_fused_z_merge import (
        PersistentHeadSplitWOAMergePlan,
        PersistentProjectionPlan,
        PersistentProjectionView,
        ProjectionSpanGeometry,
        build_persistent_projection_plan,
        merge_persistent_projection,
        project_merge_persistent_headsplit,
    )


logger = logging.getLogger(__name__)


MLA_OFF_FORMAT_VERSION = 3
MLA_OFF_POSITION_SEMANTICS = (
    "post_inverse_rope_offline_head_woa_context_exactpos_fullscope_v3"
)
MLA_OFF_EXECUTION_PROFILE = (
    "pure_headsplit_context_bound_fullscope_3_37_3_v1"
)
MLA_OFF_INDEPENDENT_RELOCATION_PROFILE = (
    "pure_headsplit_independent_rope_relocation_fullscope_"
    "boundary128_3_37_3_v1"
)
MLA_OFF_COMBINED_ROW_SPARSE_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)
MLA_OFF_INDEPENDENT_POSITION_SEMANTICS = (
    "post_inverse_rope_offline_head_woa_independent_doc_"
    "relocation_fullscope_v1"
)
_PURE_HEADSPLIT_BOUNDARY_REPAIR_TOKENS = 0
_INDEPENDENT_BOUNDARY_REPAIR_TOKENS = 128
MLA_OFF_REQUIRED_LAYER_IDS = tuple(range(3, 40))
MLA_OFF_DIAGNOSTIC_ABLATION_FIELD = "mla_off_diagnostic_ablation"
MLA_OFF_DIAGNOSTIC_ABLATION_FULL = "full"
MLA_OFF_DIAGNOSTIC_ABLATION_ZOFF_ONLY = "zoff_only"
MLA_OFF_DIAGNOSTIC_ABLATION_SHARED_ONLY = "shared_only"
MLA_OFF_DIAGNOSTIC_ABLATIONS = (
    MLA_OFF_DIAGNOSTIC_ABLATION_FULL,
    MLA_OFF_DIAGNOSTIC_ABLATION_ZOFF_ONLY,
    MLA_OFF_DIAGNOSTIC_ABLATION_SHARED_ONLY,
)
DEFAULT_MLA_OFF_MAX_BYTES = 8 * 1024**3
DEFAULT_MLA_OFF_DEVICE_MAX_BYTES = 0
MLA_OFF_TOKEN_BYTES_PER_ROW = 9  # int64 token id + bool validity
MLA_OFF_TRANSFER_AUDIT_SCHEMA = "redknot_mla_off_controller_stats_v1"
MLA_OFF_TRANSFER_BYTE_SEMANTICS = "logical_cpu_source_payload_v1"
MLA_OFF_COMPACT_WOA_AUDIT_SCHEMA = "redknot_mla_off_compact_woa_v1"
MLA_OFF_COMPACT_WOA_MEASUREMENT_SEMANTICS = (
    "successful_indexed_inverse_rope_wo_a_row_geometry_v1"
)
MLA_OFF_COMPACT_WOA_CLAIM_SCOPE = "activation_evidence_not_flops_or_energy_v1"


def resolve_mla_off_diagnostic_ablation(plan: Mapping[str, object]) -> str:
    """Return the canonical request-scoped accuracy-attribution mode.

    The field is deliberately an enum, not two independent booleans: a request
    can never accidentally enable both cache-only and z_off-only attribution.
    Absence is the production ``full`` path and therefore leaves existing plan
    digests and serving behavior unchanged.  Historical/ad-hoc boolean spellings
    are rejected rather than guessed so a typo cannot silently run the wrong
    accuracy experiment.
    """

    if not isinstance(plan, Mapping):
        raise TypeError("MLA-off diagnostic ablation plan must be a mapping")
    legacy_flags = (
        "mla_off_zoff_only",
        "mla_off_shared_only",
        "zoff_only",
        "shared_only",
    )
    contaminated = tuple(name for name in legacy_flags if name in plan)
    if contaminated:
        raise ValueError(
            "MLA-off diagnostic ablation requires the canonical enum field; "
            "conflicting flags=" + ",".join(contaminated)
        )
    raw = plan.get(
        MLA_OFF_DIAGNOSTIC_ABLATION_FIELD,
        MLA_OFF_DIAGNOSTIC_ABLATION_FULL,
    )
    if type(raw) is not str or raw not in MLA_OFF_DIAGNOSTIC_ABLATIONS:
        raise ValueError(
            "MLA-off diagnostic ablation must be exactly one of "
            f"{MLA_OFF_DIAGNOSTIC_ABLATIONS!r}"
        )
    if (
        raw != MLA_OFF_DIAGNOSTIC_ABLATION_FULL
        and str(plan.get("mode", "")) != "restore"
    ):
        raise ValueError(
            "non-full MLA-off diagnostic ablation is valid only for restore plans"
        )
    return raw

# Cumulative counters have a deliberately stable, zero-filled schema so a
# caller can take exact before/after snapshots around one forward.  The byte
# values describe the logical CPU source payload accepted by ``Tensor.to``;
# they are not claimed to be PCIe transaction or board-level measurements.
MLA_OFF_TRANSFER_COUNTER_FIELDS = (
    "device_restore_calls",
    "device_rows_restored",
    "rows_restored",
    "online_artifact_h2d_calls",
    "online_artifact_h2d_bytes",
    "online_device_gather_index_h2d_calls",
    "online_device_gather_index_h2d_rows",
    "online_device_gather_index_h2d_bytes",
    "online_device_scatter_index_h2d_calls",
    "online_device_scatter_index_h2d_rows",
    "online_device_scatter_index_h2d_bytes",
    "online_dirty_index_h2d_calls",
    "online_dirty_index_h2d_rows",
    "online_dirty_index_h2d_bytes",
    "snapshot_device_index_h2d_calls",
    "snapshot_device_index_h2d_rows",
    "snapshot_device_index_h2d_bytes",
    "online_index_h2d_bytes",
    "online_total_h2d_bytes",
)
MLA_OFF_TRANSFER_GAUGE_FIELDS = (
    "device_cache_enabled",
    "reserved_device_bytes",
    "allocated_device_bytes",
    "max_device_cache_bytes",
)

_MLA_OFF_H2D_KINDS = {
    "online_artifact": ("online_artifact_h2d", False),
    "online_device_gather_index": (
        "online_device_gather_index_h2d",
        True,
    ),
    "online_device_scatter_index": (
        "online_device_scatter_index_h2d",
        True,
    ),
    "online_dirty_index": ("online_dirty_index_h2d", True),
    "snapshot_device_index": ("snapshot_device_index_h2d", True),
}


def _mla_off_tensor_identity(tensor: torch.Tensor) -> Tuple[object, ...]:
    """Return a versioned identity without reading device tensor values.

    Ordinary tensors include their mutation version. Inference-mode tensors
    expose no version counter, so they are bound by storage/lifetime identity
    under SGLang's one-forward immutable-ownership contract; this helper does
    not claim to detect an illicit in-place mutation of such a tensor.
    """

    try:
        version: object = int(tensor._version)
    except RuntimeError:
        # Tensors allocated under torch.inference_mode intentionally have no
        # version counter. SGLang owns these one-forward tensors and treats
        # their storage as immutable.
        version = "inference-immutable"
    return (
        id(tensor),
        int(tensor.data_ptr()),
        version,
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _mla_off_persistent_projection_identity(
    plan: PersistentProjectionPlan,
) -> Tuple[object, ...]:
    """Bind one context to the exact immutable plan and pinned storages.

    ``PersistentProjectionPlan.validate`` checks values and its semantic
    digest.  This additional identity prevents a caller from swapping in an
    equal-looking plan after preflight/TP commit while retaining the old
    context certificate.
    """

    return (
        id(plan),
        str(plan.digest),
        int(plan.total_rows),
        tuple(int(value) for value in plan.tail_shape),
        tuple(
            (
                id(view),
                id(view.values),
                tuple(view.values_identity),
                str(view.seg_hash),
                int(view.layer_id),
                int(view.commit_epoch),
                tuple(view.geometry.output_rows),
                tuple(view.geometry.local_rows),
                str(view.generation_token),
            )
            for view in plan.views
        ),
    )


def _mla_off_packed_sparse_q_identity(
    projection: object,
) -> Tuple[object, ...]:
    """Bind a packed Q handle and both of its device tensor certificates."""

    values = getattr(projection, "values", None)
    local_rows = getattr(projection, "local_rows", None)
    if not isinstance(values, torch.Tensor) or not isinstance(
        local_rows, torch.Tensor
    ):
        raise TypeError(
            "packed sparse-Q projection lost values/local_rows tensors"
        )
    projection_token = getattr(projection, "projection_token", None)
    digest = getattr(projection, "digest", None)
    if not isinstance(projection_token, str) or not projection_token:
        raise ValueError("packed sparse-Q projection token is empty")
    if not isinstance(digest, str) or not digest:
        raise ValueError("packed sparse-Q projection digest is empty")
    return (
        "packed_sparse_q_v1",
        id(projection),
        projection_token,
        digest,
        _mla_off_tensor_identity(values),
        _mla_off_tensor_identity(local_rows),
    )


def mla_off_layer_bytes_per_row(
    num_output_groups: int, o_lora_rank: int
) -> int:
    """BF16 projection plus one structural-validity byte per layer row."""

    num_output_groups = int(num_output_groups)
    o_lora_rank = int(o_lora_rank)
    if num_output_groups <= 0 or o_lora_rank <= 0:
        raise ValueError("MLA-off artifact dimensions must be positive")
    return num_output_groups * o_lora_rank * 2 + 1


def mla_off_expected_bytes(
    *,
    length: int,
    local_layer_count: int,
    num_output_groups: int,
    o_lora_rank: int,
) -> int:
    """Exact v1 CPU reservation for one complete TP-rank segment."""

    length = int(length)
    local_layer_count = int(local_layer_count)
    if length <= 0 or local_layer_count <= 0:
        raise ValueError("MLA-off segment length/layer count must be positive")
    return length * (
        MLA_OFF_TOKEN_BYTES_PER_ROW
        + local_layer_count
        * mla_off_layer_bytes_per_row(num_output_groups, o_lora_rank)
    )


def mla_off_device_expected_bytes(
    *,
    length: int,
    local_layer_count: int,
    num_output_groups: int,
    o_lora_rank: int,
) -> int:
    """Exact BF16 GPU-mirror values for one complete TP-rank segment."""

    length = int(length)
    local_layer_count = int(local_layer_count)
    num_output_groups = int(num_output_groups)
    o_lora_rank = int(o_lora_rank)
    if min(length, local_layer_count, num_output_groups, o_lora_rank) <= 0:
        raise ValueError("MLA-off device artifact dimensions must be positive")
    return length * local_layer_count * num_output_groups * o_lora_rank * 2


def _max_cache_bytes_from_env() -> int:
    raw = os.environ.get("REDKNOT_MLA_OFF_MAX_BYTES", "")
    if not raw:
        return DEFAULT_MLA_OFF_MAX_BYTES
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"REDKNOT_MLA_OFF_MAX_BYTES must be an integer, got {raw!r}"
        ) from error
    if value <= 0:
        raise ValueError("REDKNOT_MLA_OFF_MAX_BYTES must be positive")
    return value


def _max_device_cache_bytes_from_env() -> int:
    raw = os.environ.get("REDKNOT_MLA_OFF_DEVICE_MAX_BYTES", "")
    if not raw:
        return DEFAULT_MLA_OFF_DEVICE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "REDKNOT_MLA_OFF_DEVICE_MAX_BYTES must be an integer, "
            f"got {raw!r}"
        ) from error
    if value < 0:
        raise ValueError("REDKNOT_MLA_OFF_DEVICE_MAX_BYTES must be non-negative")
    return value


@dataclass(frozen=True)
class MLAOffLayerSpec:
    """Compatibility and shape contract for one TP rank and model layer."""

    layer_id: int
    tp_rank: int
    tp_size: int
    owned_logical_heads: Tuple[int, ...]
    offline_local_heads: Tuple[int, ...]
    num_output_groups: int
    heads_per_group: int
    head_dim: int
    o_lora_rank: int
    model_compat_hash: str
    head_policy_hash: str
    # Empty values keep the storage/controller helpers independently testable;
    # the production backend always supplies and validates the pure profile.
    execution_profile: str = ""
    required_layer_ids: Tuple[int, ...] = ()
    storage_dtype: str = "bfloat16"
    position_semantics: str = MLA_OFF_POSITION_SEMANTICS
    format_version: int = MLA_OFF_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("MLA-off layer_id must be non-negative")
        if self.execution_profile:
            supported_profiles = (
                MLA_OFF_EXECUTION_PROFILE,
                MLA_OFF_INDEPENDENT_RELOCATION_PROFILE,
                MLA_OFF_COMBINED_ROW_SPARSE_PROFILE,
            )
            if self.execution_profile not in supported_profiles:
                raise ValueError("MLA-off execution profile is incompatible")
            if tuple(self.required_layer_ids) != MLA_OFF_REQUIRED_LAYER_IDS:
                raise ValueError(
                    "MLA-off required layer contract must be exactly 3..39"
                )
            if self.layer_id not in self.required_layer_ids:
                raise ValueError(
                    f"MLA-off layer {self.layer_id} is outside the reusable range 3..39"
                )
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("MLA-off TP rank/world size is invalid")
        if not self.owned_logical_heads:
            raise ValueError("MLA-off spec must own at least one logical head")
        if len(set(self.owned_logical_heads)) != len(self.owned_logical_heads):
            raise ValueError("MLA-off owned logical heads must be unique")
        owned = set(self.owned_logical_heads)
        if not self.offline_local_heads or not set(self.offline_local_heads) <= owned:
            raise ValueError("MLA-off local heads must be a non-empty owned subset")
        if self.num_output_groups <= 0 or self.heads_per_group <= 0:
            raise ValueError("MLA-off output grouping must be positive")
        if (
            self.num_output_groups * self.heads_per_group
            != len(self.owned_logical_heads)
        ):
            raise ValueError("MLA-off output groups do not cover the owned heads")
        if self.head_dim <= 0 or self.o_lora_rank <= 0:
            raise ValueError("MLA-off projection dimensions must be positive")
        if self.storage_dtype != "bfloat16":
            raise ValueError("MLA-off v1 supports CPU bfloat16 storage only")
        expected_position_semantics = (
            MLA_OFF_INDEPENDENT_POSITION_SEMANTICS
            if self.execution_profile
            in (
                MLA_OFF_INDEPENDENT_RELOCATION_PROFILE,
                MLA_OFF_COMBINED_ROW_SPARSE_PROFILE,
            )
            else MLA_OFF_POSITION_SEMANTICS
        )
        if self.position_semantics != expected_position_semantics:
            raise ValueError("MLA-off position semantics are incompatible")
        if self.format_version != MLA_OFF_FORMAT_VERSION:
            raise ValueError("MLA-off artifact format version is incompatible")
        if not self.model_compat_hash or not self.head_policy_hash:
            raise ValueError("MLA-off compatibility hashes must be non-empty")

    @property
    def value_shape_tail(self) -> Tuple[int, int]:
        return self.num_output_groups, self.o_lora_rank

    def bytes_for_segment(self, length: int) -> int:
        # BF16 values plus a structural-validity byte per token.
        return int(length) * mla_off_layer_bytes_per_row(
            self.num_output_groups, self.o_lora_rank
        )


@dataclass
class MLAOffLayerEntry:
    spec: MLAOffLayerSpec
    values: torch.Tensor  # CPU BF16 [length, groups, o_lora_rank]
    valid_rows: torch.Tensor  # CPU bool [length]
    device_values: Optional[torch.Tensor] = None

    @property
    def nbytes(self) -> int:
        return self.values.numel() * self.values.element_size() + (
            self.valid_rows.numel() * self.valid_rows.element_size()
        )

    @property
    def device_nbytes(self) -> int:
        if self.device_values is None:
            return 0
        return self.device_values.numel() * self.device_values.element_size()


@dataclass
class MLAOffSegment:
    seg_hash: str
    token_hash: str
    length: int
    canonical_start_pos: int
    model_compat_hash: str
    head_policy_hash: str
    required_local_layers: Tuple[int, ...]
    expected_bytes: int
    expected_device_bytes: int
    token_ids: torch.Tensor  # CPU int64 [length]
    valid_token_rows: torch.Tensor  # CPU bool [length]
    entries: Dict[int, MLAOffLayerEntry] = field(default_factory=dict)
    committed: bool = False
    last_access_tick: int = 0
    commit_epoch: int = 0
    device_ready_event: Optional[object] = None

    @property
    def allocated_bytes(self) -> int:
        token_bytes = self.token_ids.numel() * self.token_ids.element_size()
        token_bytes += (
            self.valid_token_rows.numel()
            * self.valid_token_rows.element_size()
        )
        return token_bytes + sum(entry.nbytes for entry in self.entries.values())

    @property
    def allocated_device_bytes(self) -> int:
        return sum(entry.device_nbytes for entry in self.entries.values())


@dataclass(frozen=True)
class MLAOffRestoreView:
    """Immutable certificate for one committed segment/layer lookup."""

    seg_hash: str
    commit_epoch: int
    layer_id: int


@dataclass(frozen=True, eq=False)
class MLAOffDeviceIndices:
    """Opaque certificate for indices copied from authoritative CPU rows.

    Device index tensors are control-plane data.  Reading their first/last
    value (or a monotonicity predicate) with ``.item()`` introduces a CUDA
    synchronization in every local-bearing layer.  The controller instead
    validates the CPU rows once, performs the copy itself, and records
    versioned tensor identities. Consumers can then revalidate this certificate
    without reading a device value. Inference-mode device indices additionally
    rely on the documented one-forward immutable-ownership contract.

    ``owner_token`` prevents a certificate made by another controller from
    being accepted accidentally.  ``semantic_digest`` binds the indices to
    the restore mask/layout certificate chosen by the attention backend.
    """

    owner_token: object
    role: str
    semantic_digest: Tuple[int, int]
    upper_bound: int
    cpu_indices: torch.Tensor
    device_indices: torch.Tensor
    cpu_identity: Tuple[object, ...]
    device_identity: Tuple[object, ...]
    ready_event: object


@dataclass(frozen=True, eq=False)
class _MLAOffCompactWOAPrevalidated:
    """Opaque proof that compact ``wo_a`` is safe before attention skips rows.

    This proof deliberately owns references to every input used by the
    all-local indexed path. Consumers compare object/storage identity and, when
    PyTorch exposes it, the mutation version without reading a CUDA value.
    Inference-mode tensors have no version counter and therefore rely on
    SGLang's one-forward immutable-ownership contract. Restore-layout and
    device-index certificates are also bound by exact object identity.
    """

    total_rows: int
    layer_id: int
    target_device: torch.device
    projection_dtype: torch.dtype
    spec: MLAOffLayerSpec
    controller: "DSV4MLAOffController"
    reuse_mask_digest: Tuple[int, int]
    reused_row_count: int
    online_row_count: int
    offline_projection: torch.Tensor
    offline_identity: Tuple[object, ...]
    reuse_mask: torch.Tensor
    reuse_mask_identity: Tuple[object, ...]
    dirty_rows_cpu: torch.Tensor
    dirty_rows_cpu_identity: Tuple[object, ...]
    dirty_rows_device: torch.Tensor
    dirty_rows_device_identity: Tuple[object, ...]
    device_indices_certificate: MLAOffDeviceIndices
    restore_layout_certificate: object
    restore_layout_key: object
    positions: torch.Tensor
    positions_identity: Tuple[object, ...]


@dataclass(frozen=True)
class _MLAOffPreparedCompactWOAAudit:
    """Fully validated and serialized marker awaiting the all-TP vote."""

    payload: Mapping[str, object]
    serialized_payload: str


@dataclass(frozen=True)
class MLAOffPublishReceipt:
    """Short-lived rollback handle for the distributed publish vote."""

    seg_hash: str
    generation_id: str
    commit_epoch: int
    segment: MLAOffSegment
    previous_segment: Optional[MLAOffSegment]


@dataclass(frozen=True, eq=False)
class MLAOffRestoreRows:
    """Rows gathered from one independently-prefilled segment."""

    seg_hash: str
    output_rows_cpu: torch.Tensor
    local_positions_cpu: torch.Tensor
    _output_rows: Tuple[int, ...] = field(init=False, repr=False)
    _local_positions: Tuple[int, ...] = field(init=False, repr=False)
    _projection_geometry: ProjectionSpanGeometry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        output = self.output_rows_cpu
        local = self.local_positions_cpu
        if (
            not isinstance(output, torch.Tensor)
            or not isinstance(local, torch.Tensor)
            or output.ndim != 1
            or local.ndim != 1
            or output.dtype != torch.long
            or local.dtype != torch.long
            or output.device.type != "cpu"
            or local.device.type != "cpu"
            or int(output.numel()) != int(local.numel())
            or int(output.numel()) <= 0
        ):
            raise ValueError(
                "MLA-off restore rows must be aligned non-empty CPU int64 vectors"
            )
        output_rows = tuple(int(value) for value in output.tolist())
        local_positions = tuple(int(value) for value in local.tolist())
        geometry = ProjectionSpanGeometry(
            output_rows=output_rows,
            local_rows=local_positions,
        )
        object.__setattr__(self, "_output_rows", output_rows)
        object.__setattr__(self, "_local_positions", local_positions)
        object.__setattr__(self, "_projection_geometry", geometry)

    @property
    def output_rows(self) -> Tuple[int, ...]:
        """Compatibility alias for callers predating the tensor hot path."""

        return self._output_rows

    @property
    def local_positions(self) -> Tuple[int, ...]:
        """Compatibility alias for callers predating the tensor hot path."""

        return self._local_positions

    @property
    def projection_geometry(self) -> ProjectionSpanGeometry:
        """Return the one fully validated, layer-independent row mapping."""

        return self._projection_geometry


def _unit_stride_projection_slices(
    output_rows: Tuple[int, ...],
    local_rows: Tuple[int, ...],
) -> Tuple[Tuple[int, int], ...]:
    """Partition one sparse mapping into maximal unit-stride spans.

    Selected-row execution packs every active island into consecutive forward
    rows, while its source positions in an independently captured document
    retain the gaps between islands.  The persistent merge kernel consumes a
    fixed number of contiguous views, so normalize that ragged mapping here,
    before the merge plan is committed, without gathering or copying z_off.
    """

    if (
        type(output_rows) is not tuple
        or type(local_rows) is not tuple
        or not output_rows
        or len(output_rows) != len(local_rows)
        or any(type(value) is not int for value in output_rows + local_rows)
        or any(value < 0 for value in output_rows + local_rows)
        or any(right <= left for left, right in zip(output_rows, output_rows[1:]))
        or any(right <= left for left, right in zip(local_rows, local_rows[1:]))
    ):
        raise ValueError(
            "persistent projection rows must be aligned strictly increasing tuples"
        )
    starts = [0]
    for index in range(1, len(output_rows)):
        if (
            output_rows[index] != output_rows[index - 1] + 1
            or local_rows[index] != local_rows[index - 1] + 1
        ):
            starts.append(index)
    return tuple(
        (start, end)
        for start, end in zip(starts, starts[1:] + [len(output_rows)])
    )


@dataclass
class MLAOffRuntimeContext:
    """Layer-local bridge between the attention backend and ``wo_a``."""

    mode: str
    layer_id: int
    spec: MLAOffLayerSpec
    local_head_axes: Tuple[int, ...]
    controller: "DSV4MLAOffController"
    seg_hash: Optional[str] = None
    generation_id: Optional[str] = None
    length: int = 0
    local_positions_cpu: Optional[torch.Tensor] = None
    offline_projection: Optional[torch.Tensor] = None
    # Device-resident alternative to ``offline_projection``.  The plan points
    # directly at committed ``MLAOffLayerEntry.device_values`` and therefore
    # never owns a gathered/full-size offline assembly for this forward.
    persistent_projection_plan: Optional[PersistentProjectionPlan] = None
    # Certified post-attention projection path.  This binds the persistent
    # z_off views, exact dirty-row pointer, local/global head partition and
    # wo_a weight before the composite protocol authorizes omission.
    headsplit_woa_merge_plan: Optional[
        PersistentHeadSplitWOAMergePlan
    ] = None
    reuse_mask: Optional[torch.Tensor] = None
    online_local_row_indices: Optional[torch.Tensor] = None
    reused_row_count: int = 0
    online_local_row_count: int = 0
    backend_applied: bool = False
    reuse_mask_digest: Tuple[int, int] = (0, 0)
    online_local_row_indices_cpu: Optional[torch.Tensor] = None
    online_local_row_indices_certificate: Optional[MLAOffDeviceIndices] = None
    restore_layout_certificate: Optional[object] = None
    input_layout_digest: Tuple[int, int] = (0, 0)
    benchmark_request_id: str = ""
    benchmark_forward_id: str = ""
    benchmark_forward_mode: str = "unknown"
    # Request-scoped, default-off accuracy attribution.  ``full`` is the
    # production composite path; the two non-full values are accepted only
    # after request-plan validation and are never inferred from cache state.
    diagnostic_ablation: str = MLA_OFF_DIAGNOSTIC_ABLATION_FULL
    benchmark_q_rows: int = 0
    # Non-empty only for an explicitly certified dense bypass.  This is not a
    # restore failure: every owned head and row executes online by design.
    intentional_full_local_reason: str = ""
    transfer_audit_state: Optional[object] = None
    # ``collective_compact_enabled`` is the all-rank permission to omit clean
    # local rows in attention.  The wo_a A/B flag is deliberately separate:
    # the control run still exercises the exact same restore preflight and
    # attention-row skip, then projects the resulting zero-filled full tensor.
    collective_compact_enabled: bool = False
    collective_compact_woa_enabled: bool = False
    # Sparse-Q is installed only after the rank-local partial projection,
    # Q-normalization, RoPE and row scatter have all completed and every
    # attention-TP rank has voted ready.  Until then the serving path must keep
    # a complete-Q fallback available.  Once committed, native attention
    # fallback is forbidden because clean local-head Q slots are intentionally
    # unmaterialized.
    sparse_q_plan: Optional[object] = None
    sparse_q_commit_certificate: Optional[object] = None
    # Production uses dsv4_sparse_q_runtime.PackedSparseQProjection.  A dense
    # rank-local Tensor remains accepted only for compatibility/control runs.
    sparse_q_projection: Optional[object] = None
    sparse_q_projection_identity: Optional[Tuple[object, ...]] = None
    sparse_q_generation_id: str = ""
    sparse_q_collective_token: str = ""
    sparse_q_projection_token: str = ""
    sparse_q_backend_preflight_complete: bool = False
    # Composite v3 binds ragged request geometry, shared physical latent KV,
    # persistent z_off views and packed sparse-Q under one TP certificate.
    batched_reuse_plan: Optional[object] = None
    shared_restore_states: Tuple[object, ...] = ()
    shared_dirty_workset: Optional[object] = None
    composite_commit_session: Optional[object] = None
    composite_certificate: Optional[object] = None
    composite_omission_authorization: Optional[object] = None
    shared_restore_applied: bool = False
    # ``zoff_only`` becomes irreversible as soon as every TP rank commits its
    # packed sparse-Q omission.  From that point onward a local producer or
    # backend failure must be carried to the fixed post-pipeline vote; dense
    # fallback on only one rank would consume missing clean-local Q rows.
    diagnostic_irreversible: bool = False
    composite_pipeline_error: Optional[BaseException] = None
    composite_dense_fallback: bool = False
    _persistent_projection_plan_identity: Optional[
        Tuple[object, ...]
    ] = field(default=None, init=False, repr=False)
    _headsplit_woa_merge_plan_identity: Optional[
        Tuple[object, ...]
    ] = field(default=None, init=False, repr=False)
    _indexed_merge_consumed: bool = field(
        default=False, init=False, repr=False
    )
    _compact_woa_prevalidated: Optional[_MLAOffCompactWOAPrevalidated] = field(
        default=None, init=False, repr=False
    )
    _indexed_merge_pending_audit: Optional[Tuple[int, int]] = field(
        default=None, init=False, repr=False
    )
    _indexed_merge_audit_confirmed: bool = field(
        default=False, init=False, repr=False
    )
    _indexed_merge_audit_prepared: bool = field(
        default=False, init=False, repr=False
    )
    _indexed_merge_prepared_audit: Optional[
        _MLAOffPreparedCompactWOAAudit
    ] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.diagnostic_ablation) is not str
            or self.diagnostic_ablation not in MLA_OFF_DIAGNOSTIC_ABLATIONS
        ):
            raise ValueError("MLA-off runtime context has an invalid diagnostic mode")
        if type(self.diagnostic_irreversible) is not bool:
            raise TypeError("diagnostic irreversible state must be boolean")
        if type(self.composite_dense_fallback) is not bool:
            raise TypeError("composite dense-fallback state must be boolean")
        if (
            self.composite_pipeline_error is not None
            and not isinstance(self.composite_pipeline_error, BaseException)
        ):
            raise TypeError("composite pipeline error must be an exception")
        # A constructor-supplied plan follows exactly the same validation and
        # identity commit as a plan installed after restore-layout preflight.
        plan = self.persistent_projection_plan
        headsplit_plan = self.headsplit_woa_merge_plan
        if plan is not None:
            self.persistent_projection_plan = None
            self.install_persistent_projection_plan(plan)
        if headsplit_plan is not None:
            self.headsplit_woa_merge_plan = None
            self.install_headsplit_woa_merge_plan(headsplit_plan)

    def record_pipeline_error(self, error: BaseException) -> BaseException:
        """Keep the first irreversible local failure as a shape-only carrier."""

        if not isinstance(error, BaseException):
            raise TypeError("pipeline error carrier requires BaseException")
        if self.composite_pipeline_error is None:
            self.composite_pipeline_error = error
        return self.composite_pipeline_error

    @property
    def has_persistent_projection(self) -> bool:
        return self.persistent_projection_plan is not None

    @staticmethod
    def _persistent_plan_output_rows(
        plan: PersistentProjectionPlan,
    ) -> Tuple[int, ...]:
        return tuple(
            int(row)
            for view in plan.views
            for row in view.geometry.output_rows
        )

    @staticmethod
    def _persistent_plan_compact_output_bounds(
        plan: PersistentProjectionPlan,
        *,
        require_tiled: bool,
    ) -> Optional[Tuple[int, int]]:
        """Validate unit-stride view order without expanding every row.

        ``ProjectionSpanGeometry`` has already certified the complete row
        tuples at construction.  Its immutable live identity prevents later
        mutation, so compact unit-stride views can be revalidated from their
        certified endpoints.  Ragged geometry deliberately returns ``None``
        and retains the historical full-row proof.
        """

        if (
            not isinstance(plan, PersistentProjectionPlan)
            or not plan.views
            or not all(view.geometry.is_unit_stride for view in plan.views)
        ):
            return None
        first = int(plan.views[0].geometry.output_start)
        cursor = first
        for view in plan.views:
            start = int(view.geometry.output_start)
            length = int(view.geometry.length)
            if length <= 0 or start < cursor:
                return None
            if require_tiled and start != cursor:
                return None
            cursor = start + length
        return first, cursor

    def _validate_persistent_projection_state(
        self,
        plan: PersistentProjectionPlan,
        *,
        target_device: torch.device,
        projection_dtype: torch.dtype,
        require_committed_identity: bool,
    ) -> None:
        """Validate plan/cache/mask identity without reading device values."""

        if not self.is_restore:
            raise ValueError(
                "persistent projection requires an MLA-off restore context"
            )
        if not isinstance(plan, PersistentProjectionPlan):
            raise TypeError("persistent projection plan has an invalid type")
        plan.validate()
        total_rows = int(plan.total_rows)
        if tuple(plan.tail_shape) != tuple(self.spec.value_shape_tail):
            raise ValueError(
                "persistent projection tail differs from the MLA-off layer"
            )
        reusable = self.reuse_mask
        if (
            not isinstance(reusable, torch.Tensor)
            or reusable.ndim != 1
            or reusable.dtype != torch.bool
            or reusable.device.type != "cpu"
            or int(reusable.numel()) != total_rows
        ):
            raise ValueError(
                "persistent projection requires the authoritative CPU reuse mask"
            )
        compact_bounds = self._persistent_plan_compact_output_bounds(
            plan, require_tiled=True
        )
        if compact_bounds is not None and compact_bounds[1] == total_rows:
            clean_start, clean_end = compact_bounds
            prefix_has_clean = bool(
                clean_start > 0
                and reusable.narrow(0, 0, clean_start).any().item()
            )
            suffix_is_clean = bool(
                clean_end > clean_start
                and reusable.narrow(
                    0, clean_start, clean_end - clean_start
                ).all().item()
            )
            if prefix_has_clean or not suffix_is_clean:
                raise ValueError(
                    "compact persistent projection differs from the reuse mask"
                )
            projected_count = clean_end - clean_start
        else:
            expected_clean_rows = tuple(
                int(value)
                for value in torch.nonzero(
                    reusable, as_tuple=False
                ).flatten().tolist()
            )
            projected_rows = self._persistent_plan_output_rows(plan)
            if projected_rows != expected_clean_rows:
                raise ValueError(
                    "persistent projection rows differ from the certified clean rows"
                )
            projected_count = len(projected_rows)
        reused_count = int(self.reused_row_count)
        online_count = int(self.online_local_row_count)
        if (
            reused_count <= 0
            or reused_count != projected_count
            or online_count < 0
            or reused_count + online_count != total_rows
        ):
            raise ValueError(
                "persistent projection clean/online row counts are inconsistent"
            )
        if int(self.benchmark_q_rows) not in (0, total_rows):
            raise ValueError(
                "persistent projection rows differ from the benchmark forward"
            )
        self.controller.validate_persistent_projection_plan(
            plan,
            spec=self.spec,
            total_rows=total_rows,
            device=target_device,
            dtype=projection_dtype,
        )
        if require_committed_identity:
            identity = self._persistent_projection_plan_identity
            if (
                self.persistent_projection_plan is not plan
                or identity is None
                or _mla_off_persistent_projection_identity(plan) != identity
            ):
                raise ValueError(
                    "persistent projection plan changed after context commit"
                )

    def install_persistent_projection_plan(
        self, plan: PersistentProjectionPlan
    ) -> None:
        """Commit one immutable device-resident merge plan to this context."""

        if self.persistent_projection_plan is not None:
            raise RuntimeError(
                "persistent projection plan was already installed"
            )
        if not isinstance(plan, PersistentProjectionPlan) or not plan.views:
            raise TypeError("persistent projection plan has an invalid type")
        first_values = plan.views[0].values
        self._validate_persistent_projection_state(
            plan,
            target_device=torch.device(first_values.device),
            projection_dtype=first_values.dtype,
            require_committed_identity=False,
        )
        self.persistent_projection_plan = plan
        self._persistent_projection_plan_identity = (
            _mla_off_persistent_projection_identity(plan)
        )

    def validate_persistent_projection_commit(
        self, online_projection: Optional[torch.Tensor] = None
    ) -> PersistentProjectionPlan:
        """Revalidate the exact artifact generations immediately before merge."""

        plan = self.persistent_projection_plan
        if not isinstance(plan, PersistentProjectionPlan):
            raise RuntimeError(
                "MLA-off context has no committed persistent projection"
            )
        if online_projection is None:
            first_values = plan.views[0].values
            target_device = torch.device(first_values.device)
            projection_dtype = first_values.dtype
        else:
            if (
                not isinstance(online_projection, torch.Tensor)
                or online_projection.ndim != 3
            ):
                raise ValueError(
                    "persistent online projection must be a 3D tensor"
                )
            target_device = online_projection.device
            projection_dtype = online_projection.dtype
            if tuple(int(value) for value in online_projection.shape) != (
                int(plan.total_rows),
                *tuple(int(value) for value in plan.tail_shape),
            ):
                raise ValueError(
                    "persistent online projection shape differs from its plan"
                )
        self._validate_persistent_projection_state(
            plan,
            target_device=target_device,
            projection_dtype=projection_dtype,
            require_committed_identity=True,
        )
        return plan

    def _validate_headsplit_woa_merge_state(
        self,
        plan: PersistentHeadSplitWOAMergePlan,
        *,
        require_committed_identity: bool,
    ) -> None:
        if not self.is_restore:
            raise ValueError("headsplit wo_a merge requires a restore context")
        if not isinstance(plan, PersistentHeadSplitWOAMergePlan):
            raise TypeError("headsplit wo_a merge plan has an invalid type")
        if require_committed_identity:
            identity = self._headsplit_woa_merge_plan_identity
            if (
                self.headsplit_woa_merge_plan is not plan
                or identity != (id(plan), str(plan.digest))
            ):
                raise ValueError("headsplit wo_a merge changed after commit")
            plan.validate_live(committed_plan_identity=identity)
            projection_plan = self.persistent_projection_plan
            projection_identity = self._persistent_projection_plan_identity
            if (
                not isinstance(projection_plan, PersistentProjectionPlan)
                or projection_identity is None
                or type(projection_identity) is not tuple
                or len(projection_identity) < 2
                or projection_identity[0] != id(projection_plan)
                or projection_identity[1] != str(projection_plan.digest)
            ):
                raise ValueError(
                    "persistent projection plan changed after context commit"
                )
        else:
            # Installation remains the one complete geometry/mask/cache proof.
            plan.validate()
            projection_plan = self.validate_persistent_projection_commit()
        if plan.projection_plan is not projection_plan:
            raise ValueError("headsplit wo_a merge uses another z_off plan")
        if (
            plan.dirty_rows is not self.online_local_row_indices
            or plan.dirty_rows_cpu is not self.online_local_row_indices_cpu
        ):
            raise ValueError("headsplit wo_a merge uses another dirty-row plan")
        if tuple(plan.local_head_axes) != tuple(self.local_head_axes):
            raise ValueError("headsplit wo_a merge local heads changed")
        if (
            int(plan.owned_heads) != len(self.spec.owned_logical_heads)
            or int(plan.groups) != int(self.spec.num_output_groups)
            or int(plan.head_dim) != int(self.spec.head_dim)
            or int(plan.o_lora_rank) != int(self.spec.o_lora_rank)
        ):
            raise ValueError("headsplit wo_a merge differs from the MLA spec")

    def install_headsplit_woa_merge_plan(
        self, plan: PersistentHeadSplitWOAMergePlan
    ) -> None:
        """Install the pre-omission two-kernel projection certificate."""

        if self.headsplit_woa_merge_plan is not None:
            raise RuntimeError("headsplit wo_a merge plan was already installed")
        self._validate_headsplit_woa_merge_state(
            plan,
            require_committed_identity=False,
        )
        self.headsplit_woa_merge_plan = plan
        self._headsplit_woa_merge_plan_identity = (id(plan), str(plan.digest))

    def project_merge_headsplit(
        self,
        rotated_attention_output: torch.Tensor,
        *,
        wo_a_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Consume rank-local heads without any GPU gather/scatter tensor."""

        if not self.backend_applied:
            raise RuntimeError(
                "headsplit wo_a merge requires certified headwise attention"
            )
        plan = self.headsplit_woa_merge_plan
        if not isinstance(plan, PersistentHeadSplitWOAMergePlan):
            raise RuntimeError("headsplit wo_a merge was not committed")
        self._validate_headsplit_woa_merge_state(
            plan,
            require_committed_identity=True,
        )
        return project_merge_persistent_headsplit(
            rotated_attention_output,
            plan,
            wo_a_weight=wo_a_weight,
            committed_plan_identity=self._headsplit_woa_merge_plan_identity,
        )

    @property
    def is_snapshot(self) -> bool:
        return self.mode == "snapshot"

    @property
    def is_restore(self) -> bool:
        return self.mode == "restore"

    @property
    def is_full_local(self) -> bool:
        """True when this layer intentionally ran every owned head online."""

        return self.mode == "full_local"

    @property
    def sparse_q_committed(self) -> bool:
        return bool(
            self.sparse_q_plan is not None
            and self.sparse_q_commit_certificate is not None
            and self.sparse_q_projection is not None
            and self.sparse_q_projection_identity is not None
            and self.sparse_q_generation_id
            and self.sparse_q_collective_token
            and self.sparse_q_projection_token
        )

    def clear_sparse_q_commit(self) -> None:
        """Discard a pre-attention sparse-Q proposal before it is consumed."""

        self.sparse_q_plan = None
        self.sparse_q_commit_certificate = None
        self.sparse_q_projection = None
        self.sparse_q_projection_identity = None
        self.sparse_q_generation_id = ""
        self.sparse_q_collective_token = ""
        self.sparse_q_projection_token = ""
        self.sparse_q_backend_preflight_complete = False

    def install_sparse_q_commit(
        self,
        *,
        plan: object,
        certificate: object,
        projection: object,
        generation_id: str,
        collective_token: str,
        projection_token: str,
    ) -> None:
        """Bind one packed (or legacy dense) rank-local Q projection."""

        if not self.is_restore or int(self.reused_row_count) <= 0:
            raise ValueError("sparse-Q commit requires an active MLA restore")
        if self.sparse_q_committed:
            raise RuntimeError("sparse-Q was already committed for this layer")
        validate_plan = getattr(plan, "validate", None)
        validate_certificate = getattr(certificate, "validate", None)
        if not callable(validate_plan) or not callable(validate_certificate):
            raise TypeError("sparse-Q plan or certificate is invalid")
        validate_plan()
        if int(getattr(plan, "layer_id", -1)) != int(self.layer_id):
            raise ValueError("sparse-Q plan belongs to another layer")
        if tuple(getattr(plan, "local_head_axes", ())) != tuple(
            self.local_head_axes
        ):
            raise ValueError("sparse-Q local heads differ from MLA restore")
        expected_rows = tuple(
            int(value)
            for value in self.online_local_row_indices_cpu.tolist()
        )
        if tuple(getattr(plan, "online_local_rows", ())) != expected_rows:
            raise ValueError("sparse-Q online rows differ from MLA restore")
        generation_id = str(generation_id)
        collective_token = str(collective_token)
        projection_token = str(projection_token)
        if not generation_id or not collective_token or not projection_token:
            raise ValueError("sparse-Q commit tokens must be non-empty")
        if isinstance(projection, torch.Tensor):
            if projection.ndim != 3:
                raise ValueError(
                    "dense sparse-Q compatibility projection must be 3D"
                )
            if tuple(int(value) for value in projection.shape) != tuple(
                int(value) for value in getattr(plan, "output_shape", ())
            ):
                raise ValueError("sparse-Q projection shape differs from its plan")
            projection_identity = _mla_off_tensor_identity(projection)
        else:
            validate_projection = getattr(projection, "validate", None)
            if not callable(validate_projection):
                raise TypeError(
                    "sparse-Q projection must be a dense tensor or packed handle"
                )
            validate_projection()
            if getattr(projection, "plan", None) is not plan:
                raise ValueError(
                    "packed sparse-Q projection belongs to another plan"
                )
            if getattr(projection, "projection_token", None) != projection_token:
                raise ValueError(
                    "packed sparse-Q projection token differs from commit"
                )
            values = getattr(projection, "values", None)
            local_rows = getattr(projection, "local_rows", None)
            if (
                not isinstance(values, torch.Tensor)
                or values.ndim != 2
                or tuple(int(value) for value in values.shape)
                != (
                    int(getattr(plan, "projected_head_rows", -1)),
                    int(getattr(plan, "head_dim", -1)),
                )
            ):
                raise ValueError(
                    "packed sparse-Q values differ from the sparse plan"
                )
            if (
                not isinstance(local_rows, torch.Tensor)
                or local_rows is not self.online_local_row_indices
                or local_rows.ndim != 1
                or local_rows.dtype != torch.long
                or int(local_rows.numel()) != len(expected_rows)
                or local_rows.device != values.device
            ):
                raise ValueError(
                    "packed sparse-Q local rows differ from MLA restore"
                )
            projection_identity = _mla_off_packed_sparse_q_identity(projection)
        validate_certificate(
            plan,
            generation_id=generation_id,
            collective_token=collective_token,
            projection_token=projection_token,
        )
        self.sparse_q_plan = plan
        self.sparse_q_commit_certificate = certificate
        self.sparse_q_projection = projection
        self.sparse_q_projection_identity = projection_identity
        self.sparse_q_generation_id = generation_id
        self.sparse_q_collective_token = collective_token
        self.sparse_q_projection_token = projection_token
        self.sparse_q_backend_preflight_complete = False

    def validate_sparse_q_commit(self, projection: object) -> None:
        """Revalidate the opaque sparse-Q proof immediately before attention."""

        if not self.sparse_q_committed:
            raise RuntimeError("sparse-Q has no all-rank commit certificate")
        if projection is not self.sparse_q_projection:
            raise ValueError("sparse-Q attention projection identity changed")
        plan = self.sparse_q_plan
        certificate = self.sparse_q_commit_certificate
        plan.validate()
        certificate.validate(
            plan,
            generation_id=self.sparse_q_generation_id,
            collective_token=self.sparse_q_collective_token,
            projection_token=self.sparse_q_projection_token,
        )
        if tuple(plan.local_head_axes) != tuple(self.local_head_axes):
            raise ValueError("sparse-Q head policy changed after commit")
        expected_rows = tuple(
            int(value)
            for value in self.online_local_row_indices_cpu.tolist()
        )
        if tuple(plan.online_local_rows) != expected_rows:
            raise ValueError("sparse-Q row policy changed after commit")
        if isinstance(projection, torch.Tensor):
            if (
                _mla_off_tensor_identity(projection)
                != self.sparse_q_projection_identity
            ):
                raise ValueError("sparse-Q attention tensor storage changed")
            if tuple(int(value) for value in projection.shape) != tuple(
                int(value) for value in plan.output_shape
            ):
                raise ValueError("sparse-Q projection shape changed after commit")
        else:
            validate_projection = getattr(projection, "validate", None)
            if not callable(validate_projection):
                raise TypeError("packed sparse-Q validation disappeared")
            validate_projection()
            if (
                getattr(projection, "plan", None) is not plan
                or getattr(projection, "projection_token", None)
                != self.sparse_q_projection_token
            ):
                raise ValueError(
                    "packed sparse-Q plan/token changed after commit"
                )
            values = getattr(projection, "values", None)
            local_rows = getattr(projection, "local_rows", None)
            if (
                not isinstance(values, torch.Tensor)
                or values.ndim != 2
                or tuple(int(value) for value in values.shape)
                != (int(plan.projected_head_rows), int(plan.head_dim))
                or not isinstance(local_rows, torch.Tensor)
                or local_rows is not self.online_local_row_indices
                or local_rows.ndim != 1
                or local_rows.dtype != torch.long
                or int(local_rows.numel()) != len(expected_rows)
                or local_rows.device != values.device
            ):
                raise ValueError(
                    "packed sparse-Q storage/rows changed after commit"
                )
            if (
                _mla_off_packed_sparse_q_identity(projection)
                != self.sparse_q_projection_identity
            ):
                raise ValueError(
                    "packed sparse-Q object or tensor identity changed"
                )

    @property
    def requires_compact_woa_preflight(self) -> bool:
        """Whether this restore has the all-local geometry for compact ``wo_a``."""

        owned_count = len(self.spec.owned_logical_heads)
        return bool(
            self.is_restore
            # The persistent plan consumes a full online projection in one
            # fused merge.  The legacy compact-wo_a path instead clones a full
            # ``offline_projection`` and is intentionally not mixed with it.
            and self.persistent_projection_plan is None
            and self.local_head_axes == tuple(range(owned_count))
            and len(self.spec.offline_local_heads) == owned_count
            and set(self.spec.offline_local_heads)
            == set(self.spec.owned_logical_heads)
        )

    @property
    def compact_woa_preflight_ready(self) -> bool:
        """Whether the backend installed the opaque pre-attention proof."""

        return self._compact_woa_prevalidated is not None

    @property
    def can_merge_online_indexed(self) -> bool:
        """Whether this restore may project only its dirty input rows.

        The compact path is deliberately narrower than ordinary MLA-off
        restore.  It is valid only when this TP rank owns no global head, so a
        clean row contains no online contribution at all. In addition to the
        geometry, the backend must have completed the full certificate
        preflight before its attention-TP readiness vote, and attention must
        have applied the certified skip policy.
        """

        return bool(
            self.requires_compact_woa_preflight
            and self.backend_applied
            and self.compact_woa_preflight_ready
            and bool(self.collective_compact_enabled)
            and bool(self.collective_compact_woa_enabled)
        )

    def prevalidate_compact_woa(
        self,
        positions: torch.Tensor,
        *,
        total_rows: int,
        device: torch.device,
        projection_dtype: torch.dtype,
    ) -> bool:
        """Install the compact proof before the TP restore-readiness vote.

        Mixed/global restores are valid but do not use compact ``wo_a`` and
        therefore return ``False``. An all-local restore either installs a
        complete proof or raises; the backend converts that local failure into
        its existing all-TP native fallback vote.
        """

        self._compact_woa_prevalidated = None
        self.collective_compact_enabled = False
        self.collective_compact_woa_enabled = False
        if not self.requires_compact_woa_preflight:
            return False
        total_rows = int(total_rows)
        target_device = torch.device(device)
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 1
            or positions.dtype != torch.long
            or positions.device != target_device
            or int(positions.numel()) != total_rows
        ):
            raise ValueError(
                "MLA-off compact preflight positions are incompatible"
            )
        certified_rows = self._validate_compact_woa_structure(
            total_rows=total_rows,
            device=target_device,
            projection_dtype=projection_dtype,
        )
        self._compact_woa_prevalidated = _MLAOffCompactWOAPrevalidated(
            total_rows=total_rows,
            layer_id=int(self.layer_id),
            target_device=target_device,
            projection_dtype=projection_dtype,
            spec=self.spec,
            controller=self.controller,
            reuse_mask_digest=tuple(self.reuse_mask_digest),
            reused_row_count=int(self.reused_row_count),
            online_row_count=int(self.online_local_row_count),
            offline_projection=self.offline_projection,
            offline_identity=_mla_off_tensor_identity(
                self.offline_projection
            ),
            reuse_mask=self.reuse_mask,
            reuse_mask_identity=_mla_off_tensor_identity(self.reuse_mask),
            dirty_rows_cpu=self.online_local_row_indices_cpu,
            dirty_rows_cpu_identity=_mla_off_tensor_identity(
                self.online_local_row_indices_cpu
            ),
            dirty_rows_device=certified_rows,
            dirty_rows_device_identity=_mla_off_tensor_identity(certified_rows),
            device_indices_certificate=(
                self.online_local_row_indices_certificate
            ),
            restore_layout_certificate=self.restore_layout_certificate,
            restore_layout_key=getattr(
                self.restore_layout_certificate, "layout_key", None
            ),
            positions=positions,
            positions_identity=_mla_off_tensor_identity(positions),
        )
        return True

    def _validate_compact_woa_structure(
        self,
        *,
        total_rows: int,
        device: torch.device,
        projection_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Perform the complete mutable/certificate validation once."""

        if not self.requires_compact_woa_preflight:
            raise RuntimeError(
                "MLA-off compact preflight requires an all-local restore"
            )
        total_rows = int(total_rows)
        target_device = torch.device(device)
        expected_tail = self.spec.value_shape_tail
        if total_rows <= 0:
            raise ValueError("MLA-off indexed merge requires a non-empty input")
        if (
            not isinstance(self.offline_projection, torch.Tensor)
            or self.offline_projection.ndim != 3
            or tuple(self.offline_projection.shape)
            != (total_rows, *expected_tail)
            or self.offline_projection.device != target_device
            or self.offline_projection.dtype != projection_dtype
        ):
            raise ValueError(
                "MLA-off indexed merge has an incompatible offline projection"
            )
        reusable = self.reuse_mask
        dirty_cpu = self.online_local_row_indices_cpu
        dirty_device = self.online_local_row_indices
        certificate = self.online_local_row_indices_certificate
        if (
            not isinstance(reusable, torch.Tensor)
            or reusable.ndim != 1
            or reusable.dtype != torch.bool
            or reusable.device.type != "cpu"
            or int(reusable.numel()) != total_rows
            or not isinstance(dirty_cpu, torch.Tensor)
            or dirty_cpu.ndim != 1
            or dirty_cpu.dtype != torch.long
            or dirty_cpu.device.type != "cpu"
            or not isinstance(dirty_device, torch.Tensor)
            or dirty_device.ndim != 1
            or dirty_device.dtype != torch.long
            or dirty_device.device != target_device
        ):
            raise ValueError("MLA-off indexed merge row metadata is incomplete")
        online_count = int(self.online_local_row_count)
        reused_count = int(self.reused_row_count)
        if (
            online_count < 0
            or int(dirty_cpu.numel()) != online_count
            or int(dirty_device.numel()) != online_count
            or reused_count + online_count != total_rows
            or int(reusable.sum().item()) != reused_count
        ):
            raise ValueError("MLA-off indexed merge row counts are inconsistent")
        expected_dirty = torch.nonzero(~reusable, as_tuple=False).flatten()
        if not torch.equal(expected_dirty, dirty_cpu):
            raise ValueError(
                "MLA-off indexed merge rows are not the reuse-mask complement"
            )

        layout = self.restore_layout_certificate
        validate_layout = getattr(layout, "validate", None)
        layout_key = getattr(layout, "layout_key", None)
        if not callable(validate_layout) or layout_key is None:
            raise ValueError(
                "MLA-off indexed merge has no restore-layout certificate"
            )
        validate_layout(
            layer_id=self.layer_id,
            layout_key=layout_key,
            reusable_cpu=reusable,
            dirty_rows_cpu=dirty_cpu,
            reuse_mask_digest=self.reuse_mask_digest,
            q_rows=total_rows,
            reused_count=reused_count,
            online_count=online_count,
        )
        certified_rows = self.controller.device_indices_from_certificate(
            certificate,
            cpu_indices=dirty_cpu,
            device=target_device,
            role="online_local_rows",
            semantic_digest=self.reuse_mask_digest,
            upper_bound=total_rows,
        )
        if certified_rows is not dirty_device:
            raise ValueError(
                "MLA-off indexed merge lost its certified dirty-row tensor"
            )
        return certified_rows

    def _prevalidated_compact_rows(
        self,
        *,
        total_rows: int,
        device: torch.device,
        projection_dtype: torch.dtype,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Recheck opaque identities without discovering a new certificate."""

        state = self._compact_woa_prevalidated
        if not isinstance(state, _MLAOffCompactWOAPrevalidated):
            raise RuntimeError(
                "MLA-off compact wo_a was not prevalidated before attention"
            )
        total_rows = int(total_rows)
        target_device = torch.device(device)
        if (
            state.total_rows != total_rows
            or state.layer_id != int(self.layer_id)
            or state.target_device != target_device
            or state.projection_dtype != projection_dtype
            or self.spec is not state.spec
            or self.controller is not state.controller
            or tuple(self.reuse_mask_digest) != state.reuse_mask_digest
            or int(self.reused_row_count) != state.reused_row_count
            or int(self.online_local_row_count) != state.online_row_count
            or self.offline_projection is not state.offline_projection
            or _mla_off_tensor_identity(state.offline_projection)
            != state.offline_identity
            or self.reuse_mask is not state.reuse_mask
            or _mla_off_tensor_identity(state.reuse_mask)
            != state.reuse_mask_identity
            or self.online_local_row_indices_cpu is not state.dirty_rows_cpu
            or _mla_off_tensor_identity(state.dirty_rows_cpu)
            != state.dirty_rows_cpu_identity
            or self.online_local_row_indices is not state.dirty_rows_device
            or _mla_off_tensor_identity(state.dirty_rows_device)
            != state.dirty_rows_device_identity
            or self.online_local_row_indices_certificate
            is not state.device_indices_certificate
            or self.restore_layout_certificate
            is not state.restore_layout_certificate
            or getattr(
                state.restore_layout_certificate, "layout_key", None
            )
            != state.restore_layout_key
            or _mla_off_tensor_identity(state.positions)
            != state.positions_identity
            or (positions is not None and positions is not state.positions)
        ):
            raise ValueError(
                "MLA-off compact wo_a prevalidated identity changed"
            )

        # These certificate checks are intentionally repeated at each consumer
        # boundary. The expensive mask/complement construction happened before
        # the TP vote; here we only ensure that the opaque proofs still bind the
        # exact objects about to be consumed.
        layout = state.restore_layout_certificate
        validate_layout = getattr(layout, "validate", None)
        layout_key = state.restore_layout_key
        if not callable(validate_layout) or layout_key is None:
            raise ValueError("MLA-off compact wo_a layout proof disappeared")
        validate_layout(
            layer_id=self.layer_id,
            layout_key=layout_key,
            reusable_cpu=state.reuse_mask,
            dirty_rows_cpu=state.dirty_rows_cpu,
            reuse_mask_digest=self.reuse_mask_digest,
            q_rows=total_rows,
            reused_count=int(self.reused_row_count),
            online_count=int(self.online_local_row_count),
        )
        certified_rows = self.controller.device_indices_from_certificate(
            state.device_indices_certificate,
            cpu_indices=state.dirty_rows_cpu,
            device=target_device,
            role="online_local_rows",
            semantic_digest=self.reuse_mask_digest,
            upper_bound=total_rows,
        )
        if certified_rows is not state.dirty_rows_device:
            raise ValueError("MLA-off compact wo_a device proof changed")
        return certified_rows

    def prevalidated_rows_for_attention(
        self,
        *,
        total_rows: int,
        device: torch.device,
        projection_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Validate the installed proof immediately before attention skips."""

        return self._prevalidated_compact_rows(
            total_rows=total_rows,
            device=device,
            projection_dtype=projection_dtype,
        )

    def _validated_indexed_online_rows(
        self,
        *,
        total_rows: int,
        device: torch.device,
        projection_dtype: torch.dtype,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return prevalidated dirty rows at a compact consumer boundary."""

        if not self.can_merge_online_indexed:
            raise RuntimeError(
                "MLA-off indexed merge requires a prevalidated, applied "
                "all-local restore"
            )
        if self._indexed_merge_consumed:
            raise RuntimeError("MLA-off indexed merge was already consumed")
        return self._prevalidated_compact_rows(
            total_rows=total_rows,
            device=device,
            projection_dtype=projection_dtype,
            positions=positions,
        )

    def indexed_online_rows(
        self,
        per_head_output: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the online rows before inverse RoPE and ``wo_a``.

        Both returned tensors preserve the certified row order.  No device
        value is read by the control plane; bounds and ordering were already
        checked on the authoritative CPU row vector.
        """

        if not isinstance(per_head_output, torch.Tensor) or not isinstance(
            positions, torch.Tensor
        ):
            raise ValueError(
                "MLA-off indexed merge inputs must be tensors"
            )
        total_rows = int(per_head_output.shape[0]) if per_head_output.ndim else 0
        expected_shape = (
            total_rows,
            len(self.spec.owned_logical_heads),
            self.spec.head_dim,
        )
        if (
            per_head_output.ndim != 3
            or tuple(per_head_output.shape) != expected_shape
            or positions.ndim != 1
            or positions.dtype != torch.long
            or int(positions.numel()) != total_rows
            or positions.device != per_head_output.device
        ):
            raise ValueError(
                "MLA-off indexed merge input rows or positions are incompatible"
            )
        rows = self._validated_indexed_online_rows(
            total_rows=total_rows,
            device=per_head_output.device,
            projection_dtype=per_head_output.dtype,
            positions=positions,
        )
        return (
            per_head_output.index_select(0, rows),
            positions.index_select(0, rows),
        )

    def local_only(self, per_head_output: torch.Tensor) -> torch.Tensor:
        """Zero global-head axes while preserving the model's TP-local layout."""

        if per_head_output.ndim != 3:
            raise ValueError(
                "MLA-off expects inverse-RoPE output shaped [tokens, heads, dim]"
            )
        if per_head_output.shape[1] != len(self.spec.owned_logical_heads):
            raise ValueError("MLA-off per-head output does not match the TP head shard")
        mask = per_head_output.new_zeros((per_head_output.shape[1],))
        mask[list(self.local_head_axes)] = 1
        return per_head_output * mask.view(1, -1, 1)

    def capture(self, local_projection: torch.Tensor) -> None:
        if not self.is_snapshot:
            raise RuntimeError("MLA-off capture called on a non-snapshot context")
        if (
            self.seg_hash is None
            or self.generation_id is None
            or self.local_positions_cpu is None
        ):
            raise RuntimeError("MLA-off snapshot context is incomplete")
        self.controller.capture_rows(
            seg_hash=self.seg_hash,
            generation_id=self.generation_id,
            layer_id=self.layer_id,
            spec=self.spec,
            local_positions=self.local_positions_cpu,
            local_projection=local_projection,
        )

    def snapshot_complete(self) -> bool:
        if self.seg_hash is None or self.generation_id is None:
            raise RuntimeError("MLA-off snapshot context is incomplete")
        return self.controller.staging_complete(
            self.seg_hash, self.generation_id
        )

    def publish_snapshot(self) -> MLAOffPublishReceipt:
        if self.seg_hash is None or self.generation_id is None:
            raise RuntimeError("MLA-off snapshot context is incomplete")
        return self.controller.publish_staging(
            self.seg_hash, self.generation_id
        )

    def rollback_snapshot(
        self, receipt: Optional[MLAOffPublishReceipt] = None
    ) -> None:
        if self.seg_hash is None or self.generation_id is None:
            raise RuntimeError("MLA-off snapshot context is incomplete")
        self.controller.rollback_publish(
            receipt,
            seg_hash=self.seg_hash,
            generation_id=self.generation_id,
        )

    def confirm_snapshot(self, receipt: MLAOffPublishReceipt) -> None:
        self.controller.confirm_publish(receipt)

    def validate_snapshot_confirmation(
        self, receipt: MLAOffPublishReceipt
    ) -> None:
        self.controller.validate_publish_confirmation(receipt)

    def abort_snapshot(self) -> None:
        if self.seg_hash is not None and self.generation_id is not None:
            self.controller.abort_staging(self.seg_hash, self.generation_id)

    def merge_online(self, online_projection: torch.Tensor) -> torch.Tensor:
        if not self.is_restore:
            raise RuntimeError("MLA-off merge called on a non-restore context")
        if not self.backend_applied:
            raise RuntimeError(
                "MLA-off restore cannot merge after the backend used a full fallback"
            )
        # Prefer committed per-segment device storage.  If this plan exists but
        # its epoch/storage/mapping certificate is stale, fail closed: falling
        # through to an older assembled tensor after attention skipped work
        # could combine different artifact generations.
        if self.persistent_projection_plan is not None:
            plan = self.validate_persistent_projection_commit(online_projection)
            return merge_persistent_projection(online_projection, plan)
        if self.offline_projection is None:
            raise RuntimeError("MLA-off restore was not preloaded before attention")
        if self.offline_projection.shape != online_projection.shape:
            raise ValueError(
                "MLA-off online/offline projection shapes do not match: "
                f"online={tuple(online_projection.shape)} "
                f"offline={tuple(self.offline_projection.shape)}"
            )
        # All possible cache/device failures happened while preparing the
        # context, before the backend skipped any local-head work.
        return online_projection + self.offline_projection.to(
            dtype=online_projection.dtype
        )

    def merge_online_indexed(
        self,
        online_projection: torch.Tensor,
        *,
        total_rows: int,
    ) -> torch.Tensor:
        """Replace certified dirty rows in a fresh full offline projection.

        ``online_projection`` is compact ``[dirty, groups, o_lora_rank]``.
        The full tensor is cloned before ``index_copy_`` so neither a committed
        device mirror nor this context's preloaded projection can be mutated.
        A successful indexed merge is single-use, preventing an accidental
        second application from silently accepting unrelated online values.
        """

        if not isinstance(online_projection, torch.Tensor):
            raise ValueError("MLA-off indexed online projection must be a tensor")
        rows = self._validated_indexed_online_rows(
            total_rows=total_rows,
            device=online_projection.device,
            projection_dtype=online_projection.dtype,
        )
        expected_shape = (int(rows.numel()), *self.spec.value_shape_tail)
        if (
            online_projection.ndim != 3
            or tuple(online_projection.shape) != expected_shape
        ):
            raise ValueError(
                "MLA-off compact online projection shape does not match dirty rows: "
                f"online={tuple(online_projection.shape)} "
                f"expected={expected_shape}"
            )
        merged = self.offline_projection.clone()
        # Mark the context consumed before launching the in-place scatter.  If
        # the asynchronous device operation fails, retrying with a potentially
        # partially written result is not a safe recovery mechanism.
        self._indexed_merge_consumed = True
        merged.index_copy_(0, rows, online_projection)
        # Publication is deferred until every TP rank votes that its merge
        # succeeded. A marker must never describe a rank-local result that the
        # distributed request subsequently aborts.
        self._indexed_merge_pending_audit = (
            int(total_rows),
            int(rows.numel()),
        )
        return merged

    def prepare_indexed_merge_success_audit(
        self,
    ) -> Optional[_MLAOffPreparedCompactWOAAudit]:
        """Prepare compact evidence without making it externally visible.

        Every validation and JSON serialization step happens here, before the
        consumer's ``audit_publish`` TP vote.  A rank that cannot prepare the
        exact marker therefore cannot leave a false-positive log line on a
        peer that has not yet learned about the failure.
        """

        if (
            not self._indexed_merge_consumed
            or self._indexed_merge_pending_audit is None
            or self._indexed_merge_audit_confirmed
            or self._indexed_merge_audit_prepared
        ):
            raise RuntimeError(
                "MLA-off indexed merge has no unconfirmed successful result"
            )
        full_rows, online_rows = self._indexed_merge_pending_audit
        payload = None
        # An all-dirty algebraic boundary is correct but performs no compact
        # work reduction, so it must not emit a marker whose schema certifies a
        # strict row subset.
        if online_rows < full_rows:
            payload = self.controller.prepare_compact_woa_audit(
                request_id=self.benchmark_request_id,
                forward_id=self.benchmark_forward_id,
                forward_mode=self.benchmark_forward_mode,
                layer_id=self.layer_id,
                full_rows=full_rows,
                online_rows=online_rows,
                tp_rank=self.spec.tp_rank,
            )
        self._indexed_merge_audit_prepared = True
        self._indexed_merge_prepared_audit = payload
        return payload

    def publish_indexed_merge_success_audit(
        self,
        prepared: Optional[_MLAOffPreparedCompactWOAAudit],
    ) -> Optional[Mapping[str, object]]:
        """Publish a prepared marker after the caller's all-TP success vote."""

        if (
            not self._indexed_merge_audit_prepared
            or self._indexed_merge_audit_confirmed
            or prepared is not self._indexed_merge_prepared_audit
        ):
            raise RuntimeError(
                "MLA-off indexed merge audit payload was not prepared"
            )
        payload = None
        if prepared is not None:
            payload = self.controller.publish_compact_woa_audit(prepared)
        self._indexed_merge_audit_confirmed = True
        self._indexed_merge_pending_audit = None
        self._indexed_merge_prepared_audit = None
        return payload

    def confirm_indexed_merge_success(self) -> Optional[Mapping[str, object]]:
        """Single-rank/test convenience wrapper around prepare then publish.

        Production TP consumers use the split methods and place their
        collective between them.  Keeping this wrapper avoids widening the
        internal API change for algebra-only unit tests.
        """

        prepared = self.prepare_indexed_merge_success_audit()
        return self.publish_indexed_merge_success_audit(prepared)


class DSV4MLAOffController:
    """Process-local transactional CPU artifacts with an optional GPU mirror.

    CPU BF16 remains the authoritative v1 representation.  When the explicit
    device cap is non-zero, snapshot capture also materializes a same-epoch
    device mirror.  Restore may then require that mirror and fail before
    attention skips any local-head work if it is unavailable.
    """

    def __init__(
        self,
        max_cache_bytes: Optional[int] = None,
        max_device_cache_bytes: Optional[int] = None,
    ):
        self.max_cache_bytes = int(
            _max_cache_bytes_from_env()
            if max_cache_bytes is None
            else max_cache_bytes
        )
        self.max_device_cache_bytes = int(
            _max_device_cache_bytes_from_env()
            if max_device_cache_bytes is None
            else max_device_cache_bytes
        )
        if self.max_cache_bytes <= 0:
            raise ValueError("MLA-off cache byte cap must be positive")
        if self.max_device_cache_bytes < 0:
            raise ValueError("MLA-off device cache byte cap must be non-negative")
        self._committed: Dict[str, MLAOffSegment] = {}
        self._staging: Dict[Tuple[str, str], MLAOffSegment] = {}
        self._pending_publishes: Dict[
            Tuple[str, str], MLAOffPublishReceipt
        ] = {}
        self._tick = 0
        self._commit_epoch = 0
        self._device_indices_owner_token = object()
        # Plans are immutable views over committed device-resident z_off
        # entries.  Repeated RAG queries over the same frozen document bundle
        # otherwise rebuild and re-hash tens of thousands of row ids for all
        # 37 layers.  The cache is generation-bound and is invalidated before
        # an artifact can be replaced or evicted, so it never prolongs a stale
        # GPU mirror lifetime.
        self._persistent_projection_plan_cache: Dict[
            Tuple[object, ...], PersistentProjectionPlan
        ] = {}
        self._persistent_projection_plan_cache_limit = 512
        self.stats = Counter()

    def _invalidate_persistent_projection_plan_cache(
        self, seg_hash: Optional[str] = None
    ) -> None:
        cache = self._persistent_projection_plan_cache
        if seg_hash is None:
            cache.clear()
            return
        target = str(seg_hash)
        for key in tuple(cache):
            bindings = key[-1]
            if any(binding[0] == target for binding in bindings):
                cache.pop(key, None)

    def prepare_compact_woa_audit(
        self,
        *,
        request_id: str,
        forward_id: str,
        forward_mode: str,
        layer_id: int,
        full_rows: int,
        online_rows: int,
        tp_rank: int,
    ) -> _MLAOffPreparedCompactWOAAudit:
        """Validate and serialize one indexed ``wo_a`` marker without logging.

        This marker certifies only which token rows reached inverse RoPE and
        ``wo_a`` on this layer/rank.  It deliberately does not estimate model
        FLOPs, GPU energy, or end-to-end compute savings.  Keeping that scope
        in the payload prevents downstream benchmark consumers from silently
        relabelling row geometry as a hardware-level measurement.
        """

        if not all(
            isinstance(value, str)
            for value in (request_id, forward_id, forward_mode)
        ):
            raise TypeError("compact wo_a audit identifiers must be strings")
        if any(
            type(value) is not int
            for value in (layer_id, full_rows, online_rows, tp_rank)
        ):
            raise TypeError("compact wo_a audit row geometry must use integers")
        if layer_id < 0 or tp_rank < 0 or full_rows <= 0:
            raise ValueError("compact wo_a audit layer/rank/rows are invalid")
        if not 0 <= online_rows < full_rows:
            raise ValueError(
                "compact wo_a audit online rows must be a strict subset"
            )
        payload = {
            "schema": MLA_OFF_COMPACT_WOA_AUDIT_SCHEMA,
            "measurement_semantics": (
                MLA_OFF_COMPACT_WOA_MEASUREMENT_SEMANTICS
            ),
            "claim_scope": MLA_OFF_COMPACT_WOA_CLAIM_SCOPE,
            "request_id": request_id,
            "forward_id": forward_id,
            "forward_mode": forward_mode,
            "layer": layer_id,
            "full_rows": full_rows,
            "online_rows": online_rows,
            "tp_rank": tp_rank,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return _MLAOffPreparedCompactWOAAudit(
            payload=payload,
            serialized_payload=serialized,
        )

    def publish_compact_woa_audit(
        self, prepared: _MLAOffPreparedCompactWOAAudit
    ) -> Mapping[str, object]:
        """Best-effort log of an already validated observational marker.

        Publication happens after the all-TP vote and must never perturb model
        execution or strand peers before ``wo_b``.  A logging failure is
        counted locally; the benchmark then fails closed because the required
        marker is absent.
        """

        if not isinstance(prepared, _MLAOffPreparedCompactWOAAudit):
            raise TypeError("compact wo_a audit payload was not prepared")
        if os.environ.get("REDKNOT_MLA_OFF_METRICS", "0") == "1":
            try:
                logger.info(
                    "REDKNOT_MLA_OFF_COMPACT_WOA %s",
                    prepared.serialized_payload,
                )
            except Exception:
                # Do not log the logging failure: the same broken handler may
                # raise again. This counter remains available to in-process
                # diagnostics, while missing runtime evidence is authoritative.
                self.stats["compact_woa_audit_log_failures"] += 1
        return prepared.payload

    def emit_compact_woa_audit(self, **kwargs) -> Mapping[str, object]:
        """Compatibility helper for non-TP callers and focused unit tests."""

        return self.publish_compact_woa_audit(
            self.prepare_compact_woa_audit(**kwargs)
        )

    def record_h2d_tensor(
        self,
        kind: str,
        source_tensor: torch.Tensor,
        target_device,
    ) -> int:
        """Record one successful logical CPU-to-device tensor transfer.

        Callers invoke this only *after* the corresponding copy completed.
        A fixed kind whitelist prevents unrelated tensors from being silently
        mixed into the artifact audit.  The returned byte count is convenient
        for CPU/mock tests and remains zero for CPU-to-CPU or device-to-device
        operations.
        """

        try:
            prefix, has_rows = _MLA_OFF_H2D_KINDS[str(kind)]
        except KeyError as error:
            raise ValueError(f"unknown MLA-off H2D transfer kind {kind!r}") from error
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError("MLA-off H2D transfer source must be a tensor")
        target_device = torch.device(target_device)
        if source_tensor.device.type != "cpu" or target_device.type == "cpu":
            return 0
        logical_bytes = int(source_tensor.numel()) * int(
            source_tensor.element_size()
        )
        self.stats[f"{prefix}_calls"] += 1
        self.stats[f"{prefix}_bytes"] += logical_bytes
        if has_rows:
            self.stats[f"{prefix}_rows"] += int(source_tensor.numel())
        return logical_bytes

    @property
    def reserved_bytes(self) -> int:
        return sum(segment.expected_bytes for segment in self._accounted_segments())

    @property
    def allocated_bytes(self) -> int:
        return sum(segment.allocated_bytes for segment in self._accounted_segments())

    @property
    def device_cache_enabled(self) -> bool:
        return self.max_device_cache_bytes > 0

    @property
    def reserved_device_bytes(self) -> int:
        return sum(
            segment.expected_device_bytes for segment in self._accounted_segments()
        )

    @property
    def allocated_device_bytes(self) -> int:
        return sum(
            segment.allocated_device_bytes for segment in self._accounted_segments()
        )

    def _accounted_segments(self) -> Tuple[MLAOffSegment, ...]:
        """Return unique live tensors, including both sides of pending commits."""

        candidates = [*self._committed.values(), *self._staging.values()]
        for receipt in self._pending_publishes.values():
            candidates.append(receipt.segment)
            if receipt.previous_segment is not None:
                candidates.append(receipt.previous_segment)
        seen = set()
        result = []
        for segment in candidates:
            identity = id(segment)
            if identity not in seen:
                seen.add(identity)
                result.append(segment)
        return tuple(result)

    def _touch(self, segment: MLAOffSegment) -> None:
        self._tick += 1
        segment.last_access_tick = self._tick

    def _drop_committed(self, seg_hash: str, *, evicted: bool = False) -> None:
        self._invalidate_persistent_projection_plan_cache(str(seg_hash))
        if self._committed.pop(str(seg_hash), None) is not None:
            self.stats[
                "segments_evicted" if evicted else "segments_replaced"
            ] += 1

    def _drop_staging(self, key: Tuple[str, str]) -> None:
        if self._staging.pop(key, None) is not None:
            self.stats["segments_aborted"] += 1

    def _reserve(
        self,
        expected_bytes: int,
        expected_device_bytes: int,
        keep_hash: str,
    ) -> None:
        if expected_bytes > self.max_cache_bytes:
            raise MemoryError(
                "one MLA-off segment exceeds REDKNOT_MLA_OFF_MAX_BYTES: "
                f"need={expected_bytes} cap={self.max_cache_bytes}"
            )
        if expected_device_bytes < 0:
            raise ValueError("MLA-off expected device bytes must be non-negative")
        if self.device_cache_enabled:
            if expected_device_bytes <= 0:
                raise ValueError(
                    "device-resident MLA-off requires a positive exact reservation"
                )
            if expected_device_bytes > self.max_device_cache_bytes:
                raise MemoryError(
                    "one MLA-off device segment exceeds "
                    "REDKNOT_MLA_OFF_DEVICE_MAX_BYTES: "
                    f"need={expected_device_bytes} "
                    f"cap={self.max_device_cache_bytes}"
                )
        elif expected_device_bytes:
            raise ValueError(
                "device bytes were reserved while the MLA-off device cache is disabled"
            )
        while (
            self.reserved_bytes + expected_bytes > self.max_cache_bytes
            or (
                self.device_cache_enabled
                and self.reserved_device_bytes + expected_device_bytes
                > self.max_device_cache_bytes
            )
        ):
            # Never evict an artifact with the same segment hash while a new
            # generation is staging; successful commit atomically replaces it.
            committed_candidates = [
                (segment.last_access_tick, "committed", key)
                for key, segment in self._committed.items()
                if key != keep_hash
            ]
            staging_candidates = [
                (segment.last_access_tick, "staging", key)
                for key, segment in self._staging.items()
                if key[0] != keep_hash
            ]
            candidates = committed_candidates + staging_candidates
            if not candidates:
                raise MemoryError(
                    "MLA-off cache has no evictable segment within its byte cap"
                )
            _, kind, key = min(candidates)
            if kind == "committed":
                self._drop_committed(key, evicted=True)
            else:
                self._drop_staging(key)

    def begin_staging(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        token_hash: str,
        length: int,
        canonical_start_pos: int,
        model_compat_hash: str,
        head_policy_hash: str,
        required_local_layers: Sequence[int],
        expected_bytes: int,
        expected_device_bytes: int = 0,
    ) -> None:
        seg_hash = str(seg_hash)
        generation_id = str(generation_id)
        length = int(length)
        required_layers = tuple(int(layer) for layer in required_local_layers)
        if not seg_hash or not token_hash or not generation_id:
            raise ValueError(
                "MLA-off segment/token hashes and generation id must be non-empty"
            )
        if length <= 0 or canonical_start_pos != 0:
            raise ValueError(
                "MLA-off v1 requires a positive segment captured canonically at 0"
            )
        if not required_layers or len(set(required_layers)) != len(required_layers):
            raise ValueError("MLA-off required local-bearing layers are invalid")
        if any(layer < 0 for layer in required_layers):
            raise ValueError("MLA-off required layer ids must be non-negative")
        if int(expected_bytes) <= 0:
            raise ValueError("MLA-off expected segment bytes must be positive")
        token_bytes = length * MLA_OFF_TOKEN_BYTES_PER_ROW
        if int(expected_bytes) < token_bytes:
            raise ValueError("MLA-off expected bytes do not cover token identity")
        staging_key = (seg_hash, generation_id)
        if any(key[0] == seg_hash for key in self._pending_publishes):
            raise ValueError(
                "cannot stage a new MLA-off generation during pending publish"
            )
        # Position zero starts a new generation. Abort any incomplete retry for
        # the same logical segment, while preserving the committed generation
        # until this one is complete and can atomically replace it.
        for old_key in tuple(self._staging):
            if old_key[0] == seg_hash:
                self._drop_staging(old_key)
        expected_device_bytes = int(expected_device_bytes)
        self._reserve(int(expected_bytes), expected_device_bytes, seg_hash)
        segment = MLAOffSegment(
            seg_hash=seg_hash,
            token_hash=str(token_hash),
            length=length,
            canonical_start_pos=canonical_start_pos,
            model_compat_hash=str(model_compat_hash),
            head_policy_hash=str(head_policy_hash),
            required_local_layers=required_layers,
            expected_bytes=int(expected_bytes),
            expected_device_bytes=expected_device_bytes,
            token_ids=torch.empty(length, dtype=torch.long, device="cpu"),
            valid_token_rows=torch.zeros(length, dtype=torch.bool, device="cpu"),
        )
        self._staging[staging_key] = segment
        self._touch(segment)
        self.stats["segments_staged"] += 1

    def abort_staging(self, seg_hash: str, generation_id: str) -> None:
        self._drop_staging((str(seg_hash), str(generation_id)))

    def staging_token_ids(
        self, *, seg_hash: str, generation_id: str
    ) -> torch.Tensor:
        """Return a complete immutable token certificate for bundle capture.

        Shared latent KV is published in the same snapshot transaction as the
        local-head projection.  It may consume token identity only after every
        chunk has populated this authoritative staging vector.
        """

        segment = self._staging.get((str(seg_hash), str(generation_id)))
        if segment is None:
            raise KeyError("MLA-off staging generation is absent")
        if (
            segment.valid_token_rows.ndim != 1
            or int(segment.valid_token_rows.numel()) != int(segment.length)
            or not bool(segment.valid_token_rows.all().item())
        ):
            raise ValueError("MLA-off staging token certificate is incomplete")
        return segment.token_ids.clone()

    def capture_token_rows(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        local_positions: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        segment = self._staging.get((str(seg_hash), str(generation_id)))
        if segment is None:
            raise KeyError(
                f"MLA-off staging generation {generation_id!r} is absent"
            )
        positions = local_positions.detach().to(device="cpu", dtype=torch.long)
        tokens = token_ids.detach().to(device="cpu", dtype=torch.long)
        if positions.ndim != 1 or tokens.ndim != 1 or positions.shape != tokens.shape:
            raise ValueError("MLA-off token ids do not match snapshot positions")
        in_segment = (positions >= 0) & (positions < segment.length)
        positions = positions[in_segment]
        tokens = tokens[in_segment]
        if positions.numel() == 0:
            return
        already_valid = segment.valid_token_rows.index_select(0, positions)
        if bool(already_valid.any().item()):
            previous = segment.token_ids.index_select(0, positions[already_valid])
            incoming = tokens[already_valid]
            if not bool(torch.equal(previous, incoming)):
                raise ValueError("MLA-off snapshot token ids changed within a segment")
        segment.token_ids.index_copy_(0, positions, tokens)
        segment.valid_token_rows[positions] = True
        self._touch(segment)
        self.stats["token_rows_captured"] += int(positions.numel())

    def _get_compatible_segment(
        self,
        seg_hash: str,
        *,
        length: int,
        spec: MLAOffLayerSpec,
        require_committed: bool,
        token_hash: Optional[str] = None,
    ) -> MLAOffSegment:
        if any(key[0] == str(seg_hash) for key in self._pending_publishes):
            raise ValueError(
                f"MLA-off segment {seg_hash!r} has an unconfirmed publish"
            )
        segment = self._committed.get(str(seg_hash))
        if segment is None:
            raise KeyError(f"MLA-off segment {seg_hash!r} is not cached")
        if require_committed and not segment.committed:
            raise ValueError(f"MLA-off segment {seg_hash!r} is not committed")
        if require_committed and segment.commit_epoch <= 0:
            raise ValueError(f"MLA-off segment {seg_hash!r} has no commit certificate")
        self._validate_segment_metadata(
            segment,
            length=length,
            spec=spec,
            token_hash=token_hash,
        )
        self._touch(segment)
        return segment

    @staticmethod
    def _validate_segment_metadata(
        segment: MLAOffSegment,
        *,
        length: int,
        spec: MLAOffLayerSpec,
        token_hash: Optional[str],
    ) -> None:
        if segment.length != int(length):
            raise ValueError("MLA-off segment length changed")
        if token_hash is not None and segment.token_hash != str(token_hash):
            raise ValueError("MLA-off token hash changed")
        if segment.canonical_start_pos != 0:
            raise ValueError("MLA-off segment has incompatible canonical position")
        if segment.model_compat_hash != spec.model_compat_hash:
            raise ValueError("MLA-off model compatibility hash changed")
        if segment.head_policy_hash != spec.head_policy_hash:
            raise ValueError("MLA-off effective head policy changed")

    def validate_staging(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        token_hash: str,
        length: int,
        spec: MLAOffLayerSpec,
    ) -> Tuple[bool, str]:
        try:
            segment = self._staging.get((str(seg_hash), str(generation_id)))
            if segment is None:
                raise KeyError(
                    f"MLA-off staging generation {generation_id!r} is absent"
                )
            self._validate_segment_metadata(
                segment,
                token_hash=token_hash,
                length=length,
                spec=spec,
            )
            if segment.committed:
                raise ValueError("MLA-off segment is already committed")
        except (KeyError, TypeError, ValueError) as error:
            return False, str(error)
        return True, ""

    @torch.no_grad()
    def capture_rows(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        layer_id: int,
        spec: MLAOffLayerSpec,
        local_positions: torch.Tensor,
        local_projection: torch.Tensor,
    ) -> None:
        segment = self._staging.get((str(seg_hash), str(generation_id)))
        if segment is None:
            raise KeyError(
                f"MLA-off staging generation {generation_id!r} is absent"
            )
        self._validate_segment_metadata(
            segment,
            length=segment.length,
            spec=spec,
            token_hash=None,
        )
        if segment.committed:
            raise ValueError("cannot append rows to a committed MLA-off segment")
        if layer_id != spec.layer_id or layer_id not in segment.required_local_layers:
            raise ValueError("MLA-off capture targets an unexpected layer")
        positions = local_positions.detach().to(device="cpu", dtype=torch.long)
        if positions.ndim != 1 or positions.numel() != local_projection.shape[0]:
            raise ValueError("MLA-off positions do not match projection rows")
        in_segment = (positions >= 0) & (positions < segment.length)
        positions = positions[in_segment]
        if positions.numel() == 0:
            return
        in_segment_device = in_segment.to(local_projection.device)
        self.record_h2d_tensor(
            "snapshot_device_index", in_segment, local_projection.device
        )
        projection = local_projection.detach()[in_segment_device]
        if projection.ndim != 3 or tuple(projection.shape[1:]) != spec.value_shape_tail:
            raise ValueError(
                "MLA-off projection has incompatible shape: "
                f"got={tuple(projection.shape)} expected_tail={spec.value_shape_tail}"
            )
        entry = segment.entries.get(layer_id)
        if entry is None:
            entry_bytes = spec.bytes_for_segment(segment.length)
            if segment.allocated_bytes + entry_bytes > segment.expected_bytes:
                raise MemoryError(
                    "MLA-off artifact would exceed its pre-reserved segment budget"
                )
            entry = MLAOffLayerEntry(
                spec=spec,
                values=torch.empty(
                    (segment.length, *spec.value_shape_tail),
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
                valid_rows=torch.zeros(segment.length, dtype=torch.bool),
            )
            if self.device_cache_enabled:
                device_entry_bytes = (
                    segment.length
                    * spec.num_output_groups
                    * spec.o_lora_rank
                    * 2
                )
                if (
                    segment.allocated_device_bytes + device_entry_bytes
                    > segment.expected_device_bytes
                ):
                    raise MemoryError(
                        "MLA-off device artifact would exceed its exact "
                        "pre-reserved segment budget"
                    )
                entry.device_values = torch.empty(
                    (segment.length, *spec.value_shape_tail),
                    dtype=torch.bfloat16,
                    device=local_projection.device,
                )
            segment.entries[layer_id] = entry
        elif entry.spec != spec:
            raise ValueError("MLA-off layer spec changed during snapshot")
        elif self.device_cache_enabled and (
            entry.device_values is None
            or entry.device_values.device != local_projection.device
        ):
            raise ValueError("MLA-off device artifact changed device during snapshot")
        projection_cpu = projection.to(device="cpu", dtype=torch.bfloat16)
        entry.values.index_copy_(0, positions, projection_cpu)
        if self.device_cache_enabled:
            if entry.device_values is None:
                raise RuntimeError("MLA-off device cache entry is absent")
            positions_device = positions.to(
                device=entry.device_values.device, dtype=torch.long
            )
            self.record_h2d_tensor(
                "snapshot_device_index", positions, entry.device_values.device
            )
            projection_device = projection.to(
                device=entry.device_values.device, dtype=torch.bfloat16
            )
            entry.device_values.index_copy_(
                0, positions_device, projection_device
            )
            self.stats["device_rows_captured"] += int(positions.numel())
        entry.valid_rows[positions] = True
        self._touch(segment)
        self.stats["rows_captured"] += int(positions.numel())

    def staging_complete(self, seg_hash: str, generation_id: str) -> bool:
        segment = self._staging.get((str(seg_hash), str(generation_id)))
        if segment is None:
            raise KeyError(
                f"MLA-off staging generation {generation_id!r} is absent"
            )
        if not bool(segment.valid_token_rows.all().item()):
            return False
        for layer_id in segment.required_local_layers:
            entry = segment.entries.get(layer_id)
            if entry is None or not bool(entry.valid_rows.all().item()):
                return False
            if self.device_cache_enabled and entry.device_values is None:
                return False
        if self.device_cache_enabled and (
            segment.allocated_device_bytes != segment.expected_device_bytes
        ):
            return False
        if self.device_cache_enabled and segment.device_ready_event is None:
            devices = {
                entry.device_values.device
                for entry in segment.entries.values()
                if entry.device_values is not None
            }
            if len(devices) != 1:
                raise ValueError("MLA-off device artifact spans multiple devices")
            artifact_device = next(iter(devices))
            if artifact_device.type == "cuda":
                with torch.cuda.device(artifact_device):
                    ready_event = torch.cuda.Event(blocking=False)
                    ready_event.record(torch.cuda.current_stream(artifact_device))
                segment.device_ready_event = ready_event
            else:
                # CPU-only unit tests exercise the transaction and mirror
                # selection semantics without requiring CUDA.
                segment.device_ready_event = "cpu-synchronous"
        return True

    def publish_staging(
        self, seg_hash: str, generation_id: str
    ) -> MLAOffPublishReceipt:
        staging_key = (str(seg_hash), str(generation_id))
        if staging_key in self._pending_publishes:
            raise ValueError(
                "MLA-off staging generation already has a pending publish"
            )
        segment = self._staging.get(staging_key)
        if segment is None:
            raise KeyError(
                f"MLA-off staging generation {generation_id!r} is absent"
            )
        if not self.staging_complete(seg_hash, generation_id):
            raise ValueError("cannot publish an incomplete MLA-off artifact")
        old = self._committed.get(segment.seg_hash)
        self._invalidate_persistent_projection_plan_cache(segment.seg_hash)
        self._commit_epoch += 1
        receipt = MLAOffPublishReceipt(
            seg_hash=segment.seg_hash,
            generation_id=str(generation_id),
            commit_epoch=self._commit_epoch,
            segment=segment,
            previous_segment=old,
        )
        committed_before = self.stats["segments_committed"]
        replaced_before = self.stats["segments_replaced"]
        self._pending_publishes[staging_key] = receipt
        try:
            segment.committed = True
            segment.commit_epoch = receipt.commit_epoch
            self._committed[segment.seg_hash] = segment
            self._staging.pop(staging_key)
            if old is not None:
                self.stats["segments_replaced"] += 1
            self.stats["segments_committed"] += 1
        except Exception:
            if old is None:
                self._committed.pop(segment.seg_hash, None)
            else:
                self._committed[segment.seg_hash] = old
            self._staging[staging_key] = segment
            segment.committed = False
            segment.commit_epoch = 0
            self.stats["segments_committed"] = committed_before
            self.stats["segments_replaced"] = replaced_before
            self._pending_publishes.pop(staging_key, None)
            raise
        return receipt

    def rollback_publish(
        self,
        receipt: Optional[MLAOffPublishReceipt] = None,
        *,
        seg_hash: Optional[str] = None,
        generation_id: Optional[str] = None,
    ) -> None:
        """Undo a locally successful publish after a failed TP commit vote."""

        if receipt is None:
            staging_key = (str(seg_hash), str(generation_id))
            receipt = self._pending_publishes.get(staging_key)
            if receipt is None:
                # No pending receipt means publish either never mutated local
                # state or already completed a rollback/confirmation.
                return
        staging_key = (receipt.seg_hash, receipt.generation_id)
        if self._pending_publishes.get(staging_key) is not receipt:
            raise ValueError("MLA-off publish receipt is not pending")
        current = self._committed.get(receipt.seg_hash)
        if (
            current is not receipt.segment
            or current.commit_epoch != receipt.commit_epoch
        ):
            raise ValueError("MLA-off publish receipt is no longer current")
        if receipt.previous_segment is None:
            self._committed.pop(receipt.seg_hash, None)
        else:
            self._committed[receipt.seg_hash] = receipt.previous_segment
        receipt.segment.committed = False
        receipt.segment.commit_epoch = 0
        self.stats["segments_committed"] -= 1
        if receipt.previous_segment is not None:
            self.stats["segments_replaced"] -= 1
        self.stats["publish_rollbacks"] += 1
        self._pending_publishes.pop(staging_key)

    def confirm_publish(self, receipt: MLAOffPublishReceipt) -> None:
        """Release the rollback generation after every TP rank published."""

        self.validate_publish_confirmation(receipt)
        staging_key = (receipt.seg_hash, receipt.generation_id)
        self._pending_publishes.pop(staging_key)

    def validate_publish_confirmation(
        self, receipt: MLAOffPublishReceipt
    ) -> None:
        """Check commit state without releasing the capacity-accounted old copy."""

        staging_key = (receipt.seg_hash, receipt.generation_id)
        if self._pending_publishes.get(staging_key) is not receipt:
            raise ValueError("MLA-off publish receipt is not pending")
        current = self._committed.get(receipt.seg_hash)
        if (
            current is not receipt.segment
            or current.commit_epoch != receipt.commit_epoch
        ):
            raise ValueError("MLA-off pending publish is no longer current")

    def prepare_restore_view(
        self,
        *,
        seg_hash: str,
        length: int,
        spec: MLAOffLayerSpec,
        token_hash: Optional[str] = None,
    ) -> MLAOffRestoreView:
        segment = self._get_compatible_segment(
            seg_hash,
            token_hash=token_hash,
            length=length,
            spec=spec,
            require_committed=True,
        )
        entry = segment.entries.get(spec.layer_id)
        if entry is None:
            raise ValueError("MLA-off layer is absent from the artifact")
        if entry.spec != spec:
            raise ValueError("MLA-off layer spec is incompatible")
        if self.device_cache_enabled and entry.device_values is None:
            raise ValueError("MLA-off device mirror is absent from the artifact")
        return MLAOffRestoreView(
            seg_hash=str(seg_hash),
            commit_epoch=segment.commit_epoch,
            layer_id=int(spec.layer_id),
        )

    def _validate_restore_view(
        self, view: MLAOffRestoreView
    ) -> Tuple[MLAOffSegment, MLAOffLayerEntry]:
        if any(key[0] == view.seg_hash for key in self._pending_publishes):
            raise ValueError("MLA-off restore view targets an unconfirmed publish")
        current = self._committed.get(view.seg_hash)
        if (
            current is None
            or not current.committed
            or current.commit_epoch != view.commit_epoch
        ):
            raise ValueError("MLA-off restore view was replaced or evicted")
        entry = current.entries.get(view.layer_id)
        if entry is None:
            raise ValueError("MLA-off restore-view layer is no longer available")
        return current, entry

    @staticmethod
    def _persistent_projection_generation_token(
        view: MLAOffRestoreView,
    ) -> str:
        """Stable semantic generation token; tensor identity is bound separately."""

        return (
            f"mla-off-v{MLA_OFF_FORMAT_VERSION}:"
            f"{view.seg_hash}:epoch:{int(view.commit_epoch)}:"
            f"layer:{int(view.layer_id)}"
        )

    @staticmethod
    def _persistent_restore_row_tuples(
        restore_rows: MLAOffRestoreRows,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if not isinstance(restore_rows, MLAOffRestoreRows):
            raise TypeError(
                "persistent projection rows require MLAOffRestoreRows"
            )
        output = restore_rows.output_rows_cpu
        local = restore_rows.local_positions_cpu
        if (
            not isinstance(output, torch.Tensor)
            or not isinstance(local, torch.Tensor)
            or output.ndim != 1
            or local.ndim != 1
            or output.dtype != torch.long
            or local.dtype != torch.long
            or output.device.type != "cpu"
            or local.device.type != "cpu"
            or int(output.numel()) != int(local.numel())
            or int(output.numel()) <= 0
        ):
            raise ValueError(
                "persistent projection row mapping must be non-empty CPU int64"
            )
        return restore_rows.output_rows, restore_rows.local_positions

    @staticmethod
    def _wait_persistent_device_ready(
        segment: MLAOffSegment, target_device: torch.device
    ) -> None:
        if segment.device_ready_event is None:
            raise ValueError(
                "persistent projection device mirror has no readiness certificate"
            )
        if target_device.type == "cuda":
            with torch.cuda.device(target_device):
                torch.cuda.current_stream(target_device).wait_event(
                    segment.device_ready_event
                )

    def _bind_persistent_projection_view(
        self,
        view: MLAOffRestoreView,
        *,
        restore_rows: MLAOffRestoreRows,
        total_rows: int,
        spec: MLAOffLayerSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PersistentProjectionView:
        if not isinstance(view, MLAOffRestoreView):
            raise TypeError(
                "persistent projection requires an MLAOffRestoreView"
            )
        if not isinstance(spec, MLAOffLayerSpec):
            raise TypeError("persistent projection layer spec is invalid")
        if type(total_rows) is not int or total_rows <= 0:
            raise ValueError("persistent projection total_rows must be positive")
        if view.layer_id != spec.layer_id:
            raise ValueError("persistent projection view belongs to another layer")
        if restore_rows.seg_hash != view.seg_hash:
            raise ValueError(
                "persistent projection segment row mapping changed"
            )
        segment, entry = self._validate_restore_view(view)
        if entry.spec != spec:
            raise ValueError("persistent projection layer spec changed")
        if not self.device_cache_enabled or entry.device_values is None:
            raise ValueError(
                "persistent projection requires a committed device mirror"
            )
        values = entry.device_values
        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise ValueError("persistent projection production path requires CUDA")
        if values.device != target_device:
            raise ValueError(
                "persistent projection device differs from the requested device"
            )
        if values.dtype != dtype:
            raise ValueError(
                "persistent projection dtype differs from the online projection"
            )
        expected_shape = (segment.length, *spec.value_shape_tail)
        if values.ndim != 3 or tuple(values.shape) != expected_shape:
            raise ValueError(
                "persistent projection device mirror shape changed"
            )
        self._persistent_restore_row_tuples(restore_rows)
        geometry = restore_rows.projection_geometry
        geometry.validate(
            total_rows=total_rows,
            segment_rows=segment.length,
        )
        self._wait_persistent_device_ready(segment, target_device)
        return PersistentProjectionView.bind(
            seg_hash=view.seg_hash,
            layer_id=view.layer_id,
            commit_epoch=view.commit_epoch,
            geometry=geometry,
            values=values,
            generation_token=self._persistent_projection_generation_token(view),
        )

    def bind_persistent_projection_view(
        self,
        view: MLAOffRestoreView,
        *,
        restore_rows: MLAOffRestoreRows,
        total_rows: int,
        spec: MLAOffLayerSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PersistentProjectionView:
        """Pin one committed segment without gather/index_select/assembly."""

        bound = self._bind_persistent_projection_view(
            view,
            restore_rows=restore_rows,
            total_rows=total_rows,
            spec=spec,
            device=device,
            dtype=dtype,
        )
        self.stats["persistent_projection_views_bound"] += 1
        self.stats["persistent_projection_rows_bound"] += bound.geometry.length
        return bound

    def bind_persistent_projection_plan(
        self,
        *,
        bindings: Sequence[Tuple[MLAOffRestoreView, MLAOffRestoreRows]],
        total_rows: int,
        spec: MLAOffLayerSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PersistentProjectionPlan:
        """Bind one or more current device entries into an immutable merge plan.

        This function consumes only CPU row descriptors and references to
        already-resident tensors.  It deliberately never invokes ``gather``,
        ``index_select``, ``Tensor.to``, or a full-size offline allocation.
        """

        normalized = tuple(bindings)
        if not normalized:
            raise ValueError(
                "persistent projection plan requires at least one binding"
            )
        bound_views = []
        for index, binding in enumerate(normalized):
            if type(binding) is not tuple or len(binding) != 2:
                raise TypeError(
                    f"persistent projection binding {index} must be a view/rows tuple"
                )
            view, restore_rows = binding
            bound = self._bind_persistent_projection_view(
                view,
                restore_rows=restore_rows,
                total_rows=total_rows,
                spec=spec,
                device=device,
                dtype=dtype,
            )
            geometry = bound.geometry
            for start, end in _unit_stride_projection_slices(
                geometry.output_rows, geometry.local_rows
            ):
                span = ProjectionSpanGeometry(
                    output_rows=geometry.output_rows[start:end],
                    local_rows=geometry.local_rows[start:end],
                )
                bound_views.append(
                    PersistentProjectionView.bind(
                        seg_hash=bound.seg_hash,
                        layer_id=bound.layer_id,
                        commit_epoch=bound.commit_epoch,
                        geometry=span,
                        values=bound.values,
                        generation_token=bound.generation_token,
                    )
                )
        # A canonical output order gives the plan one stable digest regardless
        # of which artifact lookup happened to finish first.
        bound_views.sort(key=lambda item: item.geometry.output_rows[0])
        cache_key = None
        if all(view.geometry.is_unit_stride for view in bound_views):
            cache_key = (
                "persistent_projection_plan_unit_stride_v1",
                int(total_rows),
                spec,
                str(torch.device(device)),
                str(dtype),
                tuple(
                    (
                        str(view.seg_hash),
                        int(view.commit_epoch),
                        int(view.layer_id),
                        int(view.geometry.output_start),
                        int(view.geometry.local_start),
                        int(view.geometry.length),
                    )
                    for view in bound_views
                ),
            )
            cached = self._persistent_projection_plan_cache.get(cache_key)
            if cached is not None:
                # The incoming views were independently rebound to the current
                # committed epochs/storage above.  Require compact geometry
                # and tensor identities to match before returning the old
                # immutable plan; then run its ordinary live validation.
                if len(cached.views) != len(bound_views) or any(
                    (
                        old.seg_hash,
                        old.commit_epoch,
                        old.layer_id,
                        old.geometry.output_start,
                        old.geometry.local_start,
                        old.geometry.length,
                        old.values_identity,
                    )
                    != (
                        new.seg_hash,
                        new.commit_epoch,
                        new.layer_id,
                        new.geometry.output_start,
                        new.geometry.local_start,
                        new.geometry.length,
                        new.values_identity,
                    )
                    for old, new in zip(cached.views, bound_views)
                ):
                    self._persistent_projection_plan_cache.pop(cache_key, None)
                    raise ValueError(
                        "cached persistent projection binding changed"
                    )
                # The plan was fully validated before it entered this
                # generation-bound cache.  Re-hashing every output-row tuple
                # here made a repeated RAG restore pay O(active rows) for all
                # 37 layers even though the incoming compact view identities
                # above already proved the same artifact epochs, spans, and
                # device tensors.  The controller invalidates this cache
                # before replacement/eviction, so only mutable tensor state
                # needs to be checked on a hit.
                cached.validate_live(
                    expected_object_id=id(cached),
                    expected_digest=str(cached.digest),
                    expected_device=torch.device(device),
                    expected_dtype=dtype,
                )
                self.stats["persistent_projection_plan_cache_hits"] += 1
                self.stats["persistent_projection_plans_bound"] += 1
                self.stats["persistent_projection_views_bound"] += len(
                    cached.views
                )
                self.stats["persistent_projection_rows_bound"] += sum(
                    item.geometry.length for item in cached.views
                )
                return cached
        plan = build_persistent_projection_plan(
            total_rows=total_rows,
            tail_shape=spec.value_shape_tail,
            views=tuple(bound_views),
        )
        self.validate_persistent_projection_plan(
            plan,
            spec=spec,
            total_rows=total_rows,
            device=device,
            dtype=dtype,
        )
        self.stats["persistent_projection_plans_bound"] += 1
        self.stats["persistent_projection_views_bound"] += len(bound_views)
        self.stats["persistent_projection_rows_bound"] += sum(
            item.geometry.length for item in bound_views
        )
        if cache_key is not None:
            cache = self._persistent_projection_plan_cache
            if len(cache) >= self._persistent_projection_plan_cache_limit:
                cache.pop(next(iter(cache)))
            cache[cache_key] = plan
            self.stats["persistent_projection_plan_cache_misses"] += 1
        return plan

    def validate_persistent_projection_plan(
        self,
        plan: PersistentProjectionPlan,
        *,
        spec: MLAOffLayerSpec,
        total_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Rebind every view to the current committed epoch/storage identity."""

        if not isinstance(plan, PersistentProjectionPlan):
            raise TypeError("persistent projection plan has an invalid type")
        if not isinstance(spec, MLAOffLayerSpec):
            raise TypeError("persistent projection layer spec is invalid")
        if type(total_rows) is not int or total_rows <= 0:
            raise ValueError("persistent projection total_rows must be positive")
        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise ValueError("persistent projection production path requires CUDA")
        plan.validate()
        if (
            plan.total_rows != total_rows
            or tuple(plan.tail_shape) != tuple(spec.value_shape_tail)
        ):
            raise ValueError(
                "persistent projection plan geometry differs from the layer"
            )
        compact_bounds = MLAOffRuntimeContext._persistent_plan_compact_output_bounds(
            plan, require_tiled=False
        )
        if compact_bounds is None:
            flattened_output_rows = tuple(
                row
                for view in plan.views
                for row in view.geometry.output_rows
            )
            if flattened_output_rows != tuple(sorted(flattened_output_rows)):
                raise ValueError(
                    "persistent projection views must be ordered by output row"
                )
        for persistent_view in plan.views:
            if persistent_view.layer_id != spec.layer_id:
                raise ValueError(
                    "persistent projection plan spans multiple layers"
                )
            restore_view = MLAOffRestoreView(
                seg_hash=persistent_view.seg_hash,
                commit_epoch=persistent_view.commit_epoch,
                layer_id=persistent_view.layer_id,
            )
            segment, entry = self._validate_restore_view(restore_view)
            if entry.spec != spec:
                raise ValueError(
                    "persistent projection committed layer spec changed"
                )
            if entry.device_values is None:
                raise ValueError(
                    "persistent projection committed device mirror disappeared"
                )
            if persistent_view.values is not entry.device_values:
                raise ValueError(
                    "persistent projection storage changed after binding"
                )
            if (
                entry.device_values.device != target_device
                or entry.device_values.dtype != dtype
                or tuple(entry.device_values.shape)
                != (segment.length, *spec.value_shape_tail)
            ):
                raise ValueError(
                    "persistent projection device/dtype/shape changed"
                )
            if (
                persistent_view.generation_token
                != self._persistent_projection_generation_token(restore_view)
            ):
                raise ValueError(
                    "persistent projection generation token changed"
                )
            self._wait_persistent_device_ready(segment, target_device)

    def validate_restore(
        self,
        *,
        seg_hash: str,
        length: int,
        spec: MLAOffLayerSpec,
        token_hash: Optional[str] = None,
    ) -> Tuple[bool, str]:
        try:
            self.prepare_restore_view(
                seg_hash=seg_hash,
                length=length,
                spec=spec,
                token_hash=token_hash,
            )
        except (KeyError, TypeError, ValueError) as error:
            self.stats["restore_rejected"] += 1
            return False, str(error)
        return True, ""

    def validate_token_rows(
        self,
        *,
        seg_hash: str,
        local_positions: Sequence[int],
        token_ids: torch.Tensor,
    ) -> None:
        if any(key[0] == str(seg_hash) for key in self._pending_publishes):
            raise ValueError("MLA-off token rows target an unconfirmed publish")
        segment = self._committed.get(str(seg_hash))
        if segment is None or not segment.committed:
            raise KeyError(f"MLA-off segment {seg_hash!r} is not committed")
        self._validate_token_rows_for_segment(
            segment,
            local_positions=local_positions,
            token_ids=token_ids,
        )

    def validate_view_token_rows(
        self,
        view: MLAOffRestoreView,
        *,
        local_positions: Sequence[int],
        token_ids: torch.Tensor,
    ) -> None:
        segment, _ = self._validate_restore_view(view)
        self._validate_token_rows_for_segment(
            segment,
            local_positions=local_positions,
            token_ids=token_ids,
        )

    def token_rows_binding_identity(
        self,
        view: MLAOffRestoreView,
        *,
        local_positions: Sequence[int],
        token_ids: torch.Tensor,
    ) -> Tuple[object, ...]:
        """Return an O(1)-recheckable token-row artifact identity.

        The first reusable layer still performs the complete row/value check.
        Later layers may reuse that result only while the committed segment,
        epoch, authoritative token/validity tensors, and request row tensors
        retain their exact object/storage/version identities.
        """

        segment, _ = self._validate_restore_view(view)
        positions = self._positions_cpu(local_positions)
        tokens = (
            token_ids
            if (
                isinstance(token_ids, torch.Tensor)
                and token_ids.device.type == "cpu"
                and token_ids.dtype == torch.long
            )
            else token_ids.detach().to(device="cpu", dtype=torch.long)
        )
        return (
            id(segment),
            str(segment.seg_hash),
            int(segment.commit_epoch),
            self._index_tensor_identity(segment.token_ids),
            self._index_tensor_identity(segment.valid_token_rows),
            self._index_tensor_identity(positions),
            self._index_tensor_identity(tokens),
        )

    @staticmethod
    def _positions_cpu(local_positions: Sequence[int]) -> torch.Tensor:
        if isinstance(local_positions, torch.Tensor):
            if (
                local_positions.device.type == "cpu"
                and local_positions.dtype == torch.long
            ):
                return local_positions
            return local_positions.detach().to(device="cpu", dtype=torch.long)
        return torch.tensor(tuple(local_positions), dtype=torch.long)

    @staticmethod
    def _index_tensor_identity(tensor: torch.Tensor) -> Tuple[object, ...]:
        """Return an identity that can be checked without reading tensor data."""

        try:
            version: object = int(tensor._version)
        except RuntimeError:
            # Tensors allocated under torch.inference_mode intentionally have
            # no version counter.  SGLang owns these one-forward index tensors
            # and treats them as immutable, so storage identity is the exact
            # same contract used for ForwardBatch positions/input ids.
            version = "inference-immutable"
        return (
            id(tensor),
            int(tensor.data_ptr()),
            version,
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )

    def prepare_device_indices(
        self,
        cpu_indices: torch.Tensor,
        *,
        device: torch.device,
        role: str,
        semantic_digest: Tuple[int, int],
        upper_bound: int,
    ) -> MLAOffDeviceIndices:
        """Validate CPU indices once and copy them under an opaque certificate."""

        if (
            not isinstance(cpu_indices, torch.Tensor)
            or cpu_indices.ndim != 1
            or cpu_indices.dtype != torch.long
            or cpu_indices.device.type != "cpu"
        ):
            raise ValueError("MLA-off certified indices require a CPU int64 vector")
        role = str(role)
        if not role:
            raise ValueError("MLA-off device-index certificate needs a role")
        upper_bound = int(upper_bound)
        if upper_bound < 0:
            raise ValueError("MLA-off device-index upper bound must be non-negative")
        if cpu_indices.numel() > 0:
            # These are CPU reads.  Keeping all value validation on the
            # authoritative representation is what removes device .item()
            # synchronization from each attention layer.
            if (
                int(cpu_indices[0].item()) < 0
                or int(cpu_indices[-1].item()) >= upper_bound
                or bool((cpu_indices[1:] <= cpu_indices[:-1]).any().item())
            ):
                raise ValueError(
                    "MLA-off certified indices must be sorted unique input rows"
                )
        digest = tuple(int(value) for value in semantic_digest)
        if len(digest) != 2:
            raise ValueError("MLA-off device-index semantic digest is malformed")
        target_device = torch.device(device)
        device_indices = cpu_indices.to(device=target_device, dtype=torch.long)
        if role.startswith("artifact_local_positions:"):
            transfer_kind = "online_device_gather_index"
        elif role.startswith("artifact_output_rows:"):
            transfer_kind = "online_device_scatter_index"
        elif role == "online_local_rows":
            transfer_kind = "online_dirty_index"
        else:
            transfer_kind = None
        if transfer_kind is not None:
            self.record_h2d_tensor(
                transfer_kind, cpu_indices, target_device
            )
        if target_device.type == "cuda":
            with torch.cuda.device(target_device):
                ready_event: object = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(target_device))
        else:
            ready_event = "cpu-synchronous"
        certificate = MLAOffDeviceIndices(
            owner_token=self._device_indices_owner_token,
            role=role,
            semantic_digest=(digest[0], digest[1]),
            upper_bound=upper_bound,
            cpu_indices=cpu_indices,
            device_indices=device_indices,
            cpu_identity=self._index_tensor_identity(cpu_indices),
            device_identity=self._index_tensor_identity(device_indices),
            ready_event=ready_event,
        )
        self.stats["device_index_copies"] += 1
        self.stats["device_index_rows"] += int(cpu_indices.numel())
        return certificate

    def device_indices_from_certificate(
        self,
        certificate: MLAOffDeviceIndices,
        *,
        cpu_indices: torch.Tensor,
        device: torch.device,
        role: str,
        semantic_digest: Tuple[int, int],
        upper_bound: int,
    ) -> torch.Tensor:
        """Return certified indices without synchronizing on device values."""

        digest = tuple(int(value) for value in semantic_digest)
        target_device = torch.device(device)
        if (
            not isinstance(certificate, MLAOffDeviceIndices)
            or certificate.owner_token is not self._device_indices_owner_token
            or certificate.role != str(role)
            or certificate.semantic_digest != digest
            or certificate.upper_bound != int(upper_bound)
            or certificate.cpu_indices is not cpu_indices
            or certificate.cpu_identity
            != self._index_tensor_identity(cpu_indices)
            or certificate.device_indices.device != target_device
            or certificate.ready_event is None
            or certificate.device_identity
            != self._index_tensor_identity(certificate.device_indices)
        ):
            raise ValueError(
                "MLA-off device-index certificate is stale or incompatible"
            )
        if target_device.type == "cuda":
            with torch.cuda.device(target_device):
                torch.cuda.current_stream(target_device).wait_event(
                    certificate.ready_event
                )
        return certificate.device_indices

    @classmethod
    def _validate_token_rows_for_segment(
        cls,
        segment: MLAOffSegment,
        *,
        local_positions: Sequence[int],
        token_ids: torch.Tensor,
    ) -> None:
        positions = cls._positions_cpu(local_positions)
        tokens = token_ids.detach().to(device="cpu", dtype=torch.long)
        if tokens.ndim != 1 or tokens.numel() != positions.numel():
            raise ValueError("MLA-off restore token ids do not match cache rows")
        if positions.numel() and (
            int(positions.min().item()) < 0
            or int(positions.max().item()) >= segment.length
        ):
            raise ValueError("MLA-off token position is outside its segment")
        valid = segment.valid_token_rows.index_select(0, positions)
        if not bool(valid.all().item()):
            raise ValueError("MLA-off token identity row is absent")
        expected = segment.token_ids.index_select(0, positions)
        if not bool(torch.equal(expected, tokens)):
            raise ValueError("MLA-off server-verified token ids changed")

    @torch.no_grad()
    def gather_from_view(
        self,
        view: MLAOffRestoreView,
        *,
        local_positions: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
        use_device_cache: bool = False,
        device_indices_certificate: Optional[MLAOffDeviceIndices] = None,
        index_semantic_digest: Tuple[int, int] = (0, 0),
        index_role: str = "artifact_local_positions",
    ) -> torch.Tensor:
        segment, entry = self._validate_restore_view(view)
        positions = self._positions_cpu(local_positions)
        if positions.numel() and (
            int(positions.min().item()) < 0
            or int(positions.max().item()) >= segment.length
        ):
            raise ValueError("MLA-off restore position is outside its segment")
        if use_device_cache:
            if not self.device_cache_enabled or entry.device_values is None:
                raise ValueError("MLA-off device-resident restore is unavailable")
            target_device = torch.device(device)
            if entry.device_values.device != target_device:
                raise ValueError(
                    "MLA-off device mirror is on a different device: "
                    f"mirror={entry.device_values.device} requested={target_device}"
                )
            if segment.device_ready_event is None:
                raise ValueError("MLA-off device mirror has no readiness certificate")
            if target_device.type == "cuda":
                with torch.cuda.device(target_device):
                    torch.cuda.current_stream(target_device).wait_event(
                        segment.device_ready_event
                    )
            if device_indices_certificate is None:
                positions_device = positions.to(
                    device=target_device, dtype=torch.long
                )
                self.record_h2d_tensor(
                    "online_device_gather_index", positions, target_device
                )
            else:
                positions_device = self.device_indices_from_certificate(
                    device_indices_certificate,
                    cpu_indices=positions,
                    device=target_device,
                    role=index_role,
                    semantic_digest=index_semantic_digest,
                    upper_bound=segment.length,
                )
            values = entry.device_values.index_select(0, positions_device)
            self.stats["device_rows_restored"] += int(positions.numel())
            self.stats["device_restore_calls"] += 1
            return values.to(dtype=dtype)
        values = entry.values.index_select(0, positions)
        self.stats["rows_restored"] += int(positions.numel())
        restored = values.to(device=device, dtype=dtype)
        self.record_h2d_tensor("online_artifact", values, device)
        return restored

    @torch.no_grad()
    def gather_rows(
        self,
        *,
        seg_hash: str,
        length: int,
        spec: MLAOffLayerSpec,
        local_positions: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
        token_hash: Optional[str] = None,
        use_device_cache: bool = False,
    ) -> torch.Tensor:
        view = self.prepare_restore_view(
            seg_hash=seg_hash,
            length=length,
            spec=spec,
            token_hash=token_hash,
        )
        return self.gather_from_view(
            view,
            local_positions=local_positions,
            device=device,
            dtype=dtype,
            use_device_cache=use_device_cache,
        )

    def clear(self) -> None:
        self._invalidate_persistent_projection_plan_cache()
        self._committed.clear()
        self._staging.clear()
        self._pending_publishes.clear()
        self.stats["cache_clears"] += 1

    def snapshot_stats(self) -> Mapping[str, int]:
        result = {
            field: int(self.stats.get(field, 0))
            for field in MLA_OFF_TRANSFER_COUNTER_FIELDS
            if field not in ("online_index_h2d_bytes", "online_total_h2d_bytes")
        }
        result.update({key: int(value) for key, value in self.stats.items()})
        result["online_index_h2d_bytes"] = sum(
            int(result[field])
            for field in (
                "online_device_gather_index_h2d_bytes",
                "online_device_scatter_index_h2d_bytes",
                "online_dirty_index_h2d_bytes",
            )
        )
        result["online_total_h2d_bytes"] = (
            int(result["online_artifact_h2d_bytes"])
            + int(result["online_index_h2d_bytes"])
        )
        result.update(
            {
                "segments": len(self._committed),
                "staging_segments": len(self._staging),
                "pending_publishes": len(self._pending_publishes),
                "reserved_bytes": self.reserved_bytes,
                "allocated_bytes": self.allocated_bytes,
                "max_cache_bytes": self.max_cache_bytes,
                "device_cache_enabled": int(self.device_cache_enabled),
                "reserved_device_bytes": self.reserved_device_bytes,
                "allocated_device_bytes": self.allocated_device_bytes,
                "max_device_cache_bytes": self.max_device_cache_bytes,
            }
        )
        return result


def build_restore_rows(
    *,
    plan: Mapping[str, object],
    positions_cpu: torch.Tensor,
    refresh_layer: bool,
    extra_dirty_positions: Sequence[int] = (),
    extra_dirty_ranges: Sequence[Tuple[int, int]] = (),
    extra_dirty_mask: Optional[torch.Tensor] = None,
) -> Tuple[Tuple[MLAOffRestoreRows, ...], torch.Tensor]:
    """Map current rows to cache rows and return the clean-row bitmap.

    In the context-bound pure profile every document row is restored because
    each artifact was captured after the exact same cumulative prefix at the
    exact same absolute source interval. Query/new rows (positions at or beyond
    ``query_start``) stay online. Selected rows, Indexer-hot masks, dirty
    intervals and periodic refresh remain separate algorithms and are rejected.
    """

    positions = positions_cpu.detach().to(device="cpu", dtype=torch.long)
    if positions.ndim != 1:
        raise ValueError("MLA-off positions must be one-dimensional")
    execution_profile = str(plan.get("mla_off_execution_profile", ""))
    context_bound = execution_profile == MLA_OFF_EXECUTION_PROFILE
    independent_relocation = (
        execution_profile == MLA_OFF_INDEPENDENT_RELOCATION_PROFILE
    )
    combined_row_sparse = (
        execution_profile == MLA_OFF_COMBINED_ROW_SPARSE_PROFILE
    )
    pure_headsplit = context_bound or independent_relocation or combined_row_sparse
    if execution_profile and not pure_headsplit:
        raise ValueError("MLA-off restore execution profile is unsupported")
    if pure_headsplit and not combined_row_sparse:
        selected_row_fields = tuple(
            name
            for name in (
                "selection_policy",
                "active_token_budget_ratio",
                "hot_max_per_segment_ratio",
                "hot_frac",
                "skip_prefix_recompute",
                "checkpoint_stride_tokens",
                "checkpoint_max_islands",
                "interior_stride",
            )
            if name in plan
        )
        if selected_row_fields:
            raise ValueError(
                "pure headsplit does not accept selected/Indexer-hot fields: "
                + ",".join(selected_row_fields)
            )
        if bool(plan.get("mla_off_use_indexer_hot", False)):
            raise ValueError("pure headsplit does not accept Indexer-hot rows")
        if int(plan.get("mla_off_hot_expand_tokens", 0) or 0) != 0:
            raise ValueError("pure headsplit does not accept hot expansion")
        if int(plan.get("mla_off_refresh_layer_stride", 0) or 0) != 0:
            raise ValueError("pure headsplit does not accept refresh layers")
        if tuple(plan.get("mla_off_refresh_layers", ()) or ()):
            raise ValueError("pure headsplit does not accept refresh layers")
        if (
            extra_dirty_positions
            or extra_dirty_ranges
            or extra_dirty_mask is not None
        ):
            raise ValueError("pure headsplit does not accept extra dirty rows")
        if tuple(plan.get("mla_off_dirty_ranges", ()) or ()):
            raise ValueError("pure headsplit does not accept dirty ranges")
    reusable = torch.zeros(positions.numel(), dtype=torch.bool)
    if refresh_layer or positions.numel() == 0:
        return (), reusable

    raw_query_start = plan.get("query_start")
    query_start = (1 << 62) if raw_query_start is None else int(raw_query_start)
    dirty_ranges = []
    if combined_row_sparse:
        protection_policy = str(plan.get("query_protection_policy", "none"))
        protected_index = plan.get("query_protected_segment_index", -1)
        protected_ranges = plan.get("query_protected_ranges", [])
        segments = tuple(plan.get("segments", ()) or ())
        if type(protected_index) is not int:
            raise ValueError("query-protected segment index is invalid")
        if protection_policy == "none":
            if protected_index != -1 or protected_ranges != []:
                raise ValueError(
                    "disabled query protection requires index=-1/ranges=[]"
                )
        elif protection_policy in {
            "lexical_top1_full_segment_v1",
            "lexical_top1_block_windows_v1",
            "lexical_topk_block_windows_v2",
        }:
            if not 0 <= protected_index < len(segments):
                raise ValueError(
                    "query-protected segment is outside the restore chain"
                )
            protected_segment = segments[protected_index]
            protected_begin = int(protected_segment["global_offset"])
            protected_end = protected_begin + int(protected_segment["length"])
            if protected_begin < 0 or protected_begin >= protected_end:
                raise ValueError("query-protected segment geometry is invalid")
            if not isinstance(protected_ranges, list) or not protected_ranges:
                raise ValueError("query-protected ranges are absent")
            cursor = 0
            normalized = []
            protected_segment_indices = set()
            for item in protected_ranges:
                if not isinstance(item, Mapping) or frozenset(item) != {
                    "start",
                    "end",
                }:
                    raise ValueError("query-protected range schema is invalid")
                begin, end = int(item["start"]), int(item["end"])
                containing = [
                    segment_index
                    for segment_index, candidate in enumerate(segments)
                    if begin >= int(candidate["global_offset"])
                    and end
                    <= int(candidate["global_offset"]) + int(candidate["length"])
                ]
                if (
                    len(containing) != 1
                    or begin < cursor
                    or begin >= end
                    or begin % 512 != 0
                    or end % 512 != 0
                ):
                    raise ValueError("query-protected range geometry is invalid")
                normalized.append((begin, end))
                protected_segment_indices.add(containing[0])
                cursor = end
            if (
                protection_policy == "lexical_top1_full_segment_v1"
                and normalized != [(protected_begin, protected_end)]
            ):
                raise ValueError("full-segment query protection is incomplete")
            if (
                protection_policy == "lexical_top1_block_windows_v1"
                and protected_segment_indices != {protected_index}
            ):
                raise ValueError(
                    "top1 query protection escaped its selected segment"
                )
            if protection_policy == "lexical_topk_block_windows_v2" and (
                protected_index not in protected_segment_indices
                or len(protected_segment_indices) != 2
            ):
                raise ValueError(
                    "topk query protection must cover exactly two segments"
                )
            # The request-scoped relevance decision closes both sparsity
            # dimensions only on the output-blind ranges: ModelRunner keeps
            # their token rows active and this layer recomputes their local
            # heads instead of consuming z_off. Other rows retain both forms
            # of reuse.
            dirty_ranges.extend(normalized)
        else:
            raise ValueError("query protection policy is unsupported")
    for item in plan.get("mla_off_dirty_ranges", ()) or ():
        if isinstance(item, Mapping):
            begin, end = int(item["start"]), int(item["end"])
        else:
            begin, end = int(item[0]), int(item[1])
        if begin >= end:
            raise ValueError("MLA-off dirty ranges must satisfy start < end")
        dirty_ranges.append((begin, end))
    dirty_ranges.extend((int(value), int(value) + 1) for value in extra_dirty_positions)
    for begin, end in extra_dirty_ranges:
        begin, end = int(begin), int(end)
        if begin >= end:
            raise ValueError("MLA-off extra dirty ranges must satisfy start < end")
        dirty_ranges.append((begin, end))
    merged_dirty_ranges = []
    for begin, end in sorted(dirty_ranges):
        if merged_dirty_ranges and begin <= merged_dirty_ranges[-1][1]:
            merged_dirty_ranges[-1] = (
                merged_dirty_ranges[-1][0],
                max(end, merged_dirty_ranges[-1][1]),
            )
        else:
            merged_dirty_ranges.append((begin, end))
    segment_data = []
    for segment in plan.get("segments", ()) or ():
        seg_hash = str(segment["seg_hash"])
        global_offset = int(segment.get("global_offset", 0))
        length = int(segment["length"])
        canonical_start = int(segment.get("canonical_start_pos", 0))
        source_start = segment.get("source_start")
        source_end = segment.get("source_end")
        if (
            context_bound
            and (
                type(source_start) is not int
                or type(source_end) is not int
                or source_start != global_offset
                or source_end != source_start + length
            )
        ):
            raise ValueError(
                "context-bound segment source/global interval is invalid"
            )
        expected_pure_boundary = (
            _INDEPENDENT_BOUNDARY_REPAIR_TOKENS
            if combined_row_sparse
            or (independent_relocation and global_offset != 0)
            else _PURE_HEADSPLIT_BOUNDARY_REPAIR_TOKENS
        )
        skip_first = int(
            segment.get(
                "skip_first",
                expected_pure_boundary if pure_headsplit else 128,
            )
        )
        if canonical_start != 0:
            raise ValueError(
                "MLA-off artifacts require canonical-position-zero storage"
            )
        if (
            global_offset < 0
            or length <= 0
            or skip_first < 0
            or skip_first > length
            or (pure_headsplit and skip_first != expected_pure_boundary)
        ):
            raise ValueError("MLA-off segment length/boundary is invalid")
        # Checkpoint-island selected-row replay uses a uniform 128-token
        # segment contract, including segment zero.  Segment zero is already
        # at its canonical position, however, so it does not need local-head
        # boundary recomputation.  Keep the public plan contract uniform while
        # preserving the intended "first document is a free prefix" MLA
        # behavior in the effective z_off row map.
        effective_skip_first = (
            0 if combined_row_sparse and global_offset == 0 else skip_first
        )
        segment_data.append(
            (
                global_offset,
                global_offset + length,
                effective_skip_first,
                seg_hash,
            )
        )
    if not segment_data:
        return (), reusable
    segment_data.sort(key=lambda item: item[0])
    previous_end = None
    for global_offset, segment_end, _, _ in segment_data:
        if previous_end is not None and global_offset < previous_end:
            raise ValueError("MLA-off restore segments overlap in global position")
        previous_end = segment_end

    offsets = torch.tensor(
        [item[0] for item in segment_data], dtype=torch.long
    )
    ends = torch.tensor([item[1] for item in segment_data], dtype=torch.long)
    boundaries = torch.tensor(
        [item[2] for item in segment_data], dtype=torch.long
    )
    segment_ids = torch.bucketize(positions, ends, right=True)
    clamped_ids = segment_ids.clamp(max=len(segment_data) - 1)
    row_offsets = offsets.index_select(0, clamped_ids)
    row_ends = ends.index_select(0, clamped_ids)
    row_boundaries = boundaries.index_select(0, clamped_ids)
    in_segment = segment_ids.lt(len(segment_data))
    in_segment &= positions.ge(row_offsets) & positions.lt(row_ends)
    reusable = in_segment & positions.lt(query_start)
    relocated_boundary = row_offsets.ne(0) & positions.lt(
        row_offsets + row_boundaries
    )
    reusable &= ~relocated_boundary

    if merged_dirty_ranges:
        dirty_starts = torch.tensor(
            [begin for begin, _ in merged_dirty_ranges], dtype=torch.long
        )
        dirty_ends = torch.tensor(
            [end for _, end in merged_dirty_ranges], dtype=torch.long
        )
        dirty_ids = torch.bucketize(positions, dirty_starts, right=True) - 1
        has_dirty_interval = dirty_ids.ge(0)
        dirty_ids = dirty_ids.clamp(min=0)
        explicitly_dirty = has_dirty_interval & positions.lt(
            dirty_ends.index_select(0, dirty_ids)
        )
        reusable &= ~explicitly_dirty
    if extra_dirty_mask is not None:
        dirty_mask = extra_dirty_mask.detach().to(
            device="cpu", dtype=torch.bool
        )
        if dirty_mask.ndim != 1 or dirty_mask.numel() != positions.numel():
            raise ValueError("MLA-off extra dirty mask does not match positions")
        reusable &= ~dirty_mask

    grouped: Dict[str, Tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for segment_index, (global_offset, _, _, seg_hash) in enumerate(segment_data):
        output_tensor = torch.nonzero(
            reusable & segment_ids.eq(segment_index), as_tuple=False
        ).flatten()
        if output_tensor.numel() == 0:
            continue
        output_chunks, local_chunks = grouped.setdefault(seg_hash, ([], []))
        output_chunks.append(output_tensor)
        local_chunks.append(
            positions.index_select(0, output_tensor) - global_offset
        )

    rows = tuple(
        MLAOffRestoreRows(
            seg_hash=seg_hash,
            output_rows_cpu=torch.cat(output_chunks),
            local_positions_cpu=torch.cat(local_chunks),
        )
        for seg_hash, (output_chunks, local_chunks) in grouped.items()
        if output_chunks
    )
    return rows, reusable


_CONTROLLER: Optional[DSV4MLAOffController] = None


def get_dsv4_mla_off_controller() -> DSV4MLAOffController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = DSV4MLAOffController()
    return _CONTROLLER


__all__ = [
    "DEFAULT_MLA_OFF_DEVICE_MAX_BYTES",
    "DEFAULT_MLA_OFF_MAX_BYTES",
    "DSV4MLAOffController",
    "MLA_OFF_FORMAT_VERSION",
    "MLA_OFF_DIAGNOSTIC_ABLATION_FIELD",
    "MLA_OFF_DIAGNOSTIC_ABLATION_FULL",
    "MLA_OFF_DIAGNOSTIC_ABLATION_SHARED_ONLY",
    "MLA_OFF_DIAGNOSTIC_ABLATION_ZOFF_ONLY",
    "MLA_OFF_DIAGNOSTIC_ABLATIONS",
    "MLA_OFF_EXECUTION_PROFILE",
    "MLA_OFF_COMBINED_ROW_SPARSE_PROFILE",
    "MLA_OFF_INDEPENDENT_POSITION_SEMANTICS",
    "MLA_OFF_INDEPENDENT_RELOCATION_PROFILE",
    "MLA_OFF_POSITION_SEMANTICS",
    "MLA_OFF_REQUIRED_LAYER_IDS",
    "MLA_OFF_TOKEN_BYTES_PER_ROW",
    "MLA_OFF_TRANSFER_AUDIT_SCHEMA",
    "MLA_OFF_TRANSFER_BYTE_SEMANTICS",
    "MLA_OFF_TRANSFER_COUNTER_FIELDS",
    "MLA_OFF_TRANSFER_GAUGE_FIELDS",
    "MLAOffDeviceIndices",
    "MLAOffLayerEntry",
    "MLAOffLayerSpec",
    "MLAOffPublishReceipt",
    "MLAOffRestoreRows",
    "MLAOffRestoreView",
    "MLAOffRuntimeContext",
    "MLAOffSegment",
    "PersistentProjectionPlan",
    "PersistentProjectionView",
    "ProjectionSpanGeometry",
    "build_restore_rows",
    "get_dsv4_mla_off_controller",
    "mla_off_expected_bytes",
    "mla_off_device_expected_bytes",
    "mla_off_layer_bytes_per_row",
    "resolve_mla_off_diagnostic_ablation",
]
