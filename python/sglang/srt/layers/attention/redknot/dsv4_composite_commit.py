"""Forward-scoped TP commit protocol for composite DSV4 reuse.

The serving integration may omit work only when *all* of the following agree
for one ragged prefill forward:

* rank-local sparse ``wq_b`` projections (packed, never ``[T, 64, D]``);
* the shared position-independent latent/SWA restore generation;
* C4/C128 cache rows and blocks;
* Indexer keys and attention/Indexer compressor restart state;
* persistent rank-local ``z_off`` GPU views and their fused merge kernel; and
* the scheduler-owned ragged batch geometry.

This module is a pure CPU control plane.  It imports neither torch nor a
distributed runtime.  :class:`CollectiveAdapter` is the only integration
boundary.  Its ``exchange_commit_once`` method carries one fixed int64 vector
per rank and must be invoked at most once for a forward.  Large plans and device
views are bound transitively by SHA-256 digests rather than serialized through
the collective.

Ordering is deliberately strict::

    build cache/Q/z_off preflight -> one TP vote -> certificate
        -> omission authorization -> consume omitted slots

Before a certificate, every rank can take the dense path.  After a
certificate, failure is fail-closed and requires a coordinated abort; a rank
must never silently run dense while peers consume restored/sparse slots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Mapping, Optional, Protocol, Sequence, Tuple


COMPOSITE_PROPOSAL_FORMAT_VERSION = 5
COMPOSITE_PREFLIGHT_FORMAT_VERSION = 1
COMPOSITE_COMMIT_FORMAT_VERSION = 1
FORWARD_FINAL_FORMAT_VERSION = 1
BOUNDARY_TOKENS = 128
MAX_TP_SIZE = 64
_DIGEST_LIMB_BITS = 15
_DIGEST_LIMB_COUNT = 16
SPARSE_Q_REPRESENTATION = "packed_rank_local_head_rows_v1"
ZOFF_RESIDENCY = "persistent_request_layer_major_gpu_v1"

OMISSION_PROFILE_FULL = "full"
OMISSION_PROFILE_ZOFF_ONLY = "zoff_only"
OMISSION_PROFILE_SHARED_ONLY = "shared_only"
OMISSION_PROFILES = frozenset(
    {
        OMISSION_PROFILE_FULL,
        OMISSION_PROFILE_ZOFF_ONLY,
        OMISSION_PROFILE_SHARED_ONLY,
    }
)
COMMIT_SCOPE_COMPLETED = "completed_layer_builders_v1"
COMMIT_SCOPE_FORWARD_FRAGMENT = "forward_reserved_layer_fragment_v1"
COMMIT_SCOPE_FORWARD_RESERVED = "forward_reserved_postwrite_slots_v1"
COMMIT_SCOPES = frozenset(
    {
        COMMIT_SCOPE_COMPLETED,
        COMMIT_SCOPE_FORWARD_FRAGMENT,
        COMMIT_SCOPE_FORWARD_RESERVED,
    }
)

SWA = "swa"
C4 = "c4"
C128 = "c128"
INDEXER = "indexer"
ATTENTION_COMPRESSOR_STATE = "attention_compressor_state"
INDEXER_COMPRESSOR_STATE = "indexer_compressor_state"

_CACHE_COMPONENTS = frozenset(
    {
        SWA,
        C4,
        C128,
        INDEXER,
        ATTENTION_COMPRESSOR_STATE,
        INDEXER_COMPRESSOR_STATE,
    }
)


class CommitProtocolError(RuntimeError):
    """The integration violated the forward commit state machine."""


class CoordinatedAbortRequired(RuntimeError):
    """The forward cannot safely continue on either sparse or dense paths."""

    def __init__(self, signal: "CoordinatedAbortSignal") -> None:
        super().__init__(
            f"coordinated TP abort required: {signal.reason_code}: "
            f"{signal.detail}"
        )
        self.signal = signal


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha_digest(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if not result.startswith("sha256:") or len(result) != 71:
        raise ValueError(f"{name} must be a sha256:<64 hex> digest")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256:<64 hex> digest") from exc
    return result


def _immutable_tuple(value: object, name: str) -> tuple:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    return value


def _sorted_unique_ints(
    values: Sequence[int], *, name: str, allow_empty: bool = False
) -> Tuple[int, ...]:
    result = tuple(_strict_int(item, f"{name} entry") for item in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _sorted_unique_strings(
    values: Sequence[str], *, name: str, allow_empty: bool = False
) -> Tuple[str, ...]:
    result = tuple(_nonempty(item, f"{name} entry") for item in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _json_value(value: object) -> object:
    if hasattr(value, "as_payload"):
        return _json_value(value.as_payload())
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{type(value).__name__} is not canonically serializable")


def _digest(payload: object) -> str:
    encoded = json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ForwardIdentity:
    generation_id: str
    forward_ordinal: int
    model_hash: str
    policy_hash: str
    tp_size: int

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        _sha_digest(self.model_hash, "model_hash")
        _sha_digest(self.policy_hash, "policy_hash")
        if not 0 < _strict_int(self.tp_size, "tp_size") <= MAX_TP_SIZE:
            raise ValueError(f"tp_size must be in [1, {MAX_TP_SIZE}]")

    def as_payload(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "forward_ordinal": self.forward_ordinal,
            "model_hash": self.model_hash,
            "policy_hash": self.policy_hash,
            "tp_size": self.tp_size,
        }


@dataclass(frozen=True)
class RaggedRequestGeometry:
    request_id: str
    row_offset: int
    row_count: int
    document_rows: int
    query_rows: int
    position_begin: int
    token_digest: str

    def __post_init__(self) -> None:
        _nonempty(self.request_id, "request_id")
        values = {
            "row_offset": self.row_offset,
            "row_count": self.row_count,
            "document_rows": self.document_rows,
            "query_rows": self.query_rows,
            "position_begin": self.position_begin,
        }
        for name, value in values.items():
            _strict_int(value, name)
        if self.row_offset < 0 or self.position_begin < 0 or self.row_count <= 0:
            raise ValueError("ragged offsets must be non-negative and rows positive")
        if self.document_rows < 0 or self.query_rows < 0:
            raise ValueError("document/query row counts must be non-negative")
        if self.document_rows + self.query_rows != self.row_count:
            raise ValueError("document_rows + query_rows must equal row_count")
        _sha_digest(self.token_digest, "token_digest")

    def as_payload(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "row_offset": self.row_offset,
            "row_count": self.row_count,
            "document_rows": self.document_rows,
            "query_rows": self.query_rows,
            "position_begin": self.position_begin,
            "token_digest": self.token_digest,
        }


@dataclass(frozen=True)
class RaggedBatchGeometry:
    requests: Tuple[RaggedRequestGeometry, ...]
    total_rows: int
    scheduler_epoch: str

    def __post_init__(self) -> None:
        requests = _immutable_tuple(self.requests, "requests")
        if not requests or any(not isinstance(item, RaggedRequestGeometry) for item in requests):
            raise ValueError("requests must contain ragged request geometry")
        if _strict_int(self.total_rows, "total_rows") <= 0:
            raise ValueError("total_rows must be positive")
        _nonempty(self.scheduler_epoch, "scheduler_epoch")
        expected_offset = 0
        seen = set()
        for request in requests:
            if request.request_id in seen:
                raise ValueError("ragged request ids must be unique")
            seen.add(request.request_id)
            if request.row_offset != expected_offset:
                raise ValueError("ragged rows must tile the packed batch")
            expected_offset += request.row_count
        if expected_offset != self.total_rows:
            raise ValueError("ragged requests do not cover total_rows")

    def as_payload(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "total_rows": self.total_rows,
            "scheduler_epoch": self.scheduler_epoch,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())


@dataclass(frozen=True, order=True)
class LayerCompressionBinding:
    layer_id: int
    ratio: int

    def __post_init__(self) -> None:
        if _strict_int(self.layer_id, "layer_id") < 0:
            raise ValueError("layer_id must be non-negative")
        if _strict_int(self.ratio, "ratio") not in (4, 128):
            raise ValueError("reusable middle layers must be C4 or C128")

    def as_payload(self) -> dict[str, int]:
        return {"layer_id": self.layer_id, "ratio": self.ratio}


@dataclass(frozen=True, order=True)
class ArtifactGenerationBinding:
    """Generic pin for an immutable external artifact generation.

    The controller intentionally does not require a
    ``SharedLatentArtifact`` (or any other concrete artifact class).  Serving
    code extracts the external pin's digest and epoch into this descriptor;
    identity is then stable even if the external Python object is replaced.
    """

    pin_digest: str
    commit_epoch: int
    artifact_kind: str
    storage_generation: str

    def __post_init__(self) -> None:
        _sha_digest(self.pin_digest, "pin_digest")
        if _strict_int(self.commit_epoch, "commit_epoch") <= 0:
            raise ValueError("commit_epoch must be positive")
        _nonempty(self.artifact_kind, "artifact_kind")
        _nonempty(self.storage_generation, "storage_generation")

    def as_payload(self) -> dict[str, object]:
        return {
            "pin_digest": self.pin_digest,
            "commit_epoch": self.commit_epoch,
            "artifact_kind": self.artifact_kind,
            "storage_generation": self.storage_generation,
        }


def bind_external_artifact_pin(
    pin: object,
    *,
    artifact_kind: str,
    storage_generation: str,
    digest_attribute: str = "digest",
    epoch_attribute: str = "commit_epoch",
) -> ArtifactGenerationBinding:
    """Bind an arbitrary external pin by digest/epoch without importing it.

    ``pin`` may be a mapping or any object exposing the configured attribute
    names.  Payload bytes and concrete controller types never enter this
    protocol.
    """

    if isinstance(pin, Mapping):
        try:
            pin_digest = pin[digest_attribute]
            commit_epoch = pin[epoch_attribute]
        except KeyError as exc:
            raise ValueError("external artifact pin lacks digest/epoch") from exc
    else:
        if not hasattr(pin, digest_attribute) or not hasattr(pin, epoch_attribute):
            raise TypeError("external artifact pin must expose digest and epoch")
        pin_digest = getattr(pin, digest_attribute)
        commit_epoch = getattr(pin, epoch_attribute)
    return ArtifactGenerationBinding(
        pin_digest=pin_digest,
        commit_epoch=commit_epoch,
        artifact_kind=artifact_kind,
        storage_generation=storage_generation,
    )


@dataclass(frozen=True)
class SharedLatentBinding:
    spec_digest: str
    restore_plan_digest: str
    layer_compression: Tuple[LayerCompressionBinding, ...]
    artifacts: Tuple[ArtifactGenerationBinding, ...]
    clean_rows: int
    dirty_rows: int
    clean_rows_digest: str
    dirty_rows_digest: str
    boundary_tokens: int = BOUNDARY_TOKENS

    def __post_init__(self) -> None:
        _sha_digest(self.spec_digest, "spec_digest")
        _sha_digest(self.restore_plan_digest, "restore_plan_digest")
        compression = _immutable_tuple(self.layer_compression, "layer_compression")
        artifacts = _immutable_tuple(self.artifacts, "artifacts")
        if not compression or any(
            not isinstance(item, LayerCompressionBinding) for item in compression
        ):
            raise ValueError("layer_compression must not be empty")
        if compression != tuple(sorted(set(compression))):
            raise ValueError("layer_compression must be sorted and unique")
        if not artifacts or any(
            not isinstance(item, ArtifactGenerationBinding) for item in artifacts
        ):
            raise ValueError("artifacts must not be empty")
        if artifacts != tuple(sorted(set(artifacts))):
            raise ValueError("artifacts must be sorted and unique")
        clean = _strict_int(self.clean_rows, "clean_rows")
        dirty = _strict_int(self.dirty_rows, "dirty_rows")
        if clean < 0 or dirty < 0 or clean + dirty <= 0:
            raise ValueError("shared-latent row counts are invalid")
        _sha_digest(self.clean_rows_digest, "clean_rows_digest")
        _sha_digest(self.dirty_rows_digest, "dirty_rows_digest")
        if _strict_int(self.boundary_tokens, "boundary_tokens") != BOUNDARY_TOKENS:
            raise ValueError("the composite profile requires boundary_tokens=128")

    @property
    def layer_ids(self) -> Tuple[int, ...]:
        return tuple(item.layer_id for item in self.layer_compression)

    @property
    def compression_by_layer(self) -> Mapping[int, int]:
        return {item.layer_id: item.ratio for item in self.layer_compression}

    def as_payload(self) -> dict[str, object]:
        """Return only TP-common semantic restore identity.

        ``commit_epoch`` and physical GPU mirror generations are deliberately
        absent.  They are rank-local ABA guards: a rollback can advance the
        monotonic epoch on only the ranks that reached publish, even when all
        ranks still pin semantically identical segment content.  Requiring
        those local counters to match would permanently disable valid future
        restores after a fully rolled-back publication.
        """

        return {
            "spec_digest": self.spec_digest,
            "restore_plan_digest": self.restore_plan_digest,
            "layer_compression": self.layer_compression,
            "clean_rows": self.clean_rows,
            "dirty_rows": self.dirty_rows,
            "clean_rows_digest": self.clean_rows_digest,
            "dirty_rows_digest": self.dirty_rows_digest,
            "boundary_tokens": self.boundary_tokens,
        }

    def rank_local_payload(self) -> dict[str, object]:
        """Return exact local epoch pins used for ABA-safe consumption."""

        return {"artifacts": self.artifacts}

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())


@dataclass(frozen=True)
class GpuViewBinding:
    """Opaque identity and geometry of an already materialized GPU view."""

    storage_token: str
    view_token: str
    device_index: int
    dtype: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    byte_offset: int
    nbytes: int
    version: int

    def __post_init__(self) -> None:
        _nonempty(self.storage_token, "storage_token")
        _nonempty(self.view_token, "view_token")
        if _strict_int(self.device_index, "device_index") < 0:
            raise ValueError("device_index must be non-negative")
        _nonempty(self.dtype, "dtype")
        shape = _immutable_tuple(self.shape, "shape")
        strides = _immutable_tuple(self.strides, "strides")
        if not shape or len(shape) != len(strides):
            raise ValueError("GPU view shape/strides must be non-empty and aligned")
        if any(_strict_int(value, "shape entry") <= 0 for value in shape):
            raise ValueError("GPU view dimensions must be positive")
        if any(_strict_int(value, "stride entry") < 0 for value in strides):
            raise ValueError("GPU view strides must be non-negative")
        if _strict_int(self.byte_offset, "byte_offset") < 0:
            raise ValueError("byte_offset must be non-negative")
        if _strict_int(self.nbytes, "nbytes") <= 0:
            raise ValueError("nbytes must be positive")
        if _strict_int(self.version, "version") < 0:
            raise ValueError("GPU view version must be non-negative")

    def as_payload(self) -> dict[str, object]:
        return {
            "storage_token": self.storage_token,
            "view_token": self.view_token,
            "device_index": self.device_index,
            "dtype": self.dtype,
            "shape": self.shape,
            "strides": self.strides,
            "byte_offset": self.byte_offset,
            "nbytes": self.nbytes,
            "version": self.version,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())


@dataclass(frozen=True, order=True)
class SparseQBinding:
    layer_id: int
    tp_rank: int
    tp_size: int
    plan_digest: str
    projection_token: str
    q_rows: int
    owned_head_count: int
    head_dim: int
    projected_head_rows: int
    omitted_head_rows: int
    packed_projection_view: GpuViewBinding
    representation: str = SPARSE_Q_REPRESENTATION
    arena_index: int = 0
    write_ordinal: int = 0
    write_count: int = 1
    version_offset: int = 1
    arena_capacity_nbytes: int = 0

    def __post_init__(self) -> None:
        if _strict_int(self.layer_id, "layer_id") < 0:
            raise ValueError("layer_id must be non-negative")
        rank = _strict_int(self.tp_rank, "tp_rank")
        size = _strict_int(self.tp_size, "tp_size")
        if size <= 0 or rank < 0 or rank >= size:
            raise ValueError("sparse-Q TP geometry is invalid")
        _sha_digest(self.plan_digest, "plan_digest")
        _nonempty(self.projection_token, "projection_token")
        rows = _strict_int(self.q_rows, "q_rows")
        heads = _strict_int(self.owned_head_count, "owned_head_count")
        dim = _strict_int(self.head_dim, "head_dim")
        projected = _strict_int(self.projected_head_rows, "projected_head_rows")
        omitted = _strict_int(self.omitted_head_rows, "omitted_head_rows")
        if rows <= 0 or heads <= 0 or dim <= 0 or projected <= 0 or omitted < 0:
            raise ValueError("sparse-Q dimensions/counts are invalid")
        if projected + omitted != rows * heads:
            raise ValueError("sparse-Q head-row accounting is inconsistent")
        if self.representation != SPARSE_Q_REPRESENTATION:
            raise ValueError("full/padded Q activation is forbidden by this protocol")
        if not isinstance(self.packed_projection_view, GpuViewBinding):
            raise TypeError("packed_projection_view must be a GPU view binding")
        if self.packed_projection_view.shape != (projected, dim):
            raise ValueError("sparse-Q view must contain only packed projected rows")
        if _strict_int(self.arena_index, "arena_index") < 0:
            raise ValueError("sparse-Q arena index must be non-negative")
        if _strict_int(self.write_ordinal, "write_ordinal") < 0:
            raise ValueError("sparse-Q write ordinal must be non-negative")
        if _strict_int(self.write_count, "write_count") <= 0:
            raise ValueError("sparse-Q write count must be positive")
        if _strict_int(self.version_offset, "version_offset") <= 0:
            raise ValueError("sparse-Q version offset must be positive")
        capacity = _strict_int(
            self.arena_capacity_nbytes, "arena_capacity_nbytes"
        )
        if capacity == 0:
            # Compatibility one-layer proposals predate a forward arena.  The
            # merge normalizes their exact view size into a one-slot arena.
            capacity = self.packed_projection_view.nbytes
        if capacity < self.packed_projection_view.nbytes:
            raise ValueError("sparse-Q view exceeds its sequential arena")

    def as_payload(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "plan_digest": self.plan_digest,
            "projection_token": self.projection_token,
            "q_rows": self.q_rows,
            "owned_head_count": self.owned_head_count,
            "head_dim": self.head_dim,
            "projected_head_rows": self.projected_head_rows,
            "omitted_head_rows": self.omitted_head_rows,
            "packed_projection_view": self.packed_projection_view,
            "representation": self.representation,
            "arena_index": self.arena_index,
            "write_ordinal": self.write_ordinal,
            "write_count": self.write_count,
            "version_offset": self.version_offset,
            "arena_capacity_nbytes": self.arena_capacity_nbytes,
        }


@dataclass(frozen=True, order=True)
class SequentialQArenaWriteBinding:
    """Full rank-local write schedule for one sequential packed-Q layer slot.

    The layer/arena/ordinal subset is TP-common; ``write_count`` and the
    cumulative ``version_offset`` follow each rank's sparse head-run layout.
    """

    layer_id: int
    arena_index: int
    write_ordinal: int
    write_count: int
    version_offset: int

    def __post_init__(self) -> None:
        for name in (
            "layer_id",
            "arena_index",
            "write_ordinal",
            "write_count",
            "version_offset",
        ):
            value = _strict_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.write_count <= 0 or self.version_offset <= 0:
            raise ValueError("sequential Q writes/version offsets must be positive")

    def as_payload(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "arena_index": self.arena_index,
            "write_ordinal": self.write_ordinal,
            "write_count": self.write_count,
            "version_offset": self.version_offset,
        }


@dataclass(frozen=True)
class SequentialQArenaBinding:
    """At most two physical Q arenas reused in strict layer order.

    Only the arena count and ordered layer-to-arena assignment are TP-common.
    Physical storage, capacity, absolute versions, sparse run counts and their
    cumulative version offsets are rank-local because TP head runs may differ.
    The full object is still validated strictly before either digest is built.
    """

    arena_tokens: Tuple[str, ...]
    arena_capacities_nbytes: Tuple[int, ...]
    arena_base_versions: Tuple[int, ...]
    layer_writes: Tuple[SequentialQArenaWriteBinding, ...]
    representation: str = "sequential_packed_q_arena_v1"

    def __post_init__(self) -> None:
        tokens = _immutable_tuple(self.arena_tokens, "arena_tokens")
        capacities = _immutable_tuple(
            self.arena_capacities_nbytes, "arena_capacities_nbytes"
        )
        base_versions = _immutable_tuple(
            self.arena_base_versions, "arena_base_versions"
        )
        if not tokens or len(tokens) > 2:
            raise ValueError("forward Q reservation requires one or two arenas")
        if len(tokens) != len(capacities) or len(tokens) != len(base_versions):
            raise ValueError("sequential Q arena metadata lengths differ")
        if len(set(tokens)) != len(tokens):
            raise ValueError("sequential Q arena storage tokens must be unique")
        for token in tokens:
            _nonempty(token, "sequential Q arena token")
        if any(
            _strict_int(value, "sequential Q arena capacity") <= 0
            for value in capacities
        ):
            raise ValueError("sequential Q arena capacities must be positive")
        if any(
            _strict_int(value, "sequential Q arena base version") < 0
            for value in base_versions
        ):
            raise ValueError("sequential Q arena base versions must be non-negative")
        writes = _immutable_tuple(self.layer_writes, "layer_writes")
        if not writes or any(
            not isinstance(item, SequentialQArenaWriteBinding) for item in writes
        ):
            raise ValueError("sequential Q arena requires layer writes")
        if writes != tuple(sorted(writes, key=lambda item: item.layer_id)):
            raise ValueError("sequential Q writes must be layer-sorted")
        if tuple(item.write_ordinal for item in writes) != tuple(range(len(writes))):
            raise ValueError("sequential Q write ordinals must be dense forward order")
        cumulative = [0] * len(tokens)
        for item in writes:
            if item.arena_index >= len(tokens):
                raise ValueError("sequential Q write names an absent arena")
            cumulative[item.arena_index] += item.write_count
            if item.version_offset != cumulative[item.arena_index]:
                raise ValueError(
                    "sequential Q version offset is not cumulative for its arena"
                )
        if self.representation != "sequential_packed_q_arena_v1":
            raise ValueError("sequential Q arena representation changed")

    @property
    def layer_ids(self) -> Tuple[int, ...]:
        return tuple(item.layer_id for item in self.layer_writes)

    def shared_payload(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "arena_count": len(self.arena_tokens),
            "layer_order": tuple(
                (item.layer_id, item.arena_index, item.write_ordinal)
                for item in self.layer_writes
            ),
        }

    def rank_local_payload(self) -> dict[str, object]:
        return {
            "arena_tokens": self.arena_tokens,
            "arena_capacities_nbytes": self.arena_capacities_nbytes,
            "arena_base_versions": self.arena_base_versions,
            "layer_writes": self.layer_writes,
        }

    def expected_version(self, layer_id: int) -> int:
        matches = tuple(
            item for item in self.layer_writes if item.layer_id == int(layer_id)
        )
        if len(matches) != 1:
            raise ValueError("layer has no unique sequential Q reservation")
        item = matches[0]
        return self.arena_base_versions[item.arena_index] + item.version_offset

    def validate_sparse_bindings(
        self, sparse_q: Sequence[SparseQBinding]
    ) -> None:
        bindings = tuple(sparse_q)
        if tuple(item.layer_id for item in bindings) != self.layer_ids:
            raise ValueError("sequential Q schedule differs from sparse-Q layers")
        writes = {item.layer_id: item for item in self.layer_writes}
        for binding in bindings:
            write = writes[binding.layer_id]
            view = binding.packed_projection_view
            actual_capacity = (
                binding.arena_capacity_nbytes
                if binding.arena_capacity_nbytes > 0
                else view.nbytes
            )
            if (
                binding.arena_index,
                binding.write_ordinal,
                binding.write_count,
                binding.version_offset,
            ) != (
                write.arena_index,
                write.write_ordinal,
                write.write_count,
                write.version_offset,
            ):
                raise ValueError("sparse-Q binding differs from arena write schedule")
            if view.storage_token != self.arena_tokens[write.arena_index]:
                raise ValueError("sparse-Q view belongs to another forward arena")
            if actual_capacity != self.arena_capacities_nbytes[write.arena_index]:
                raise ValueError("sparse-Q arena capacity changed")
            if view.version != self.expected_version(binding.layer_id):
                raise ValueError("sparse-Q view does not bind its post-write version")


def build_sequential_q_arena_binding(
    sparse_q: Sequence[SparseQBinding],
) -> SequentialQArenaBinding:
    """Normalize layer bindings into a bounded sequential arena schedule."""

    bindings = tuple(sparse_q)
    if not bindings or any(not isinstance(item, SparseQBinding) for item in bindings):
        raise TypeError("sequential Q arena requires sparse-Q bindings")
    if bindings != tuple(sorted(bindings, key=lambda item: item.layer_id)):
        raise ValueError("sequential Q bindings must be layer-sorted")
    by_index: dict[int, list[SparseQBinding]] = {}
    for binding in bindings:
        by_index.setdefault(binding.arena_index, []).append(binding)
    if tuple(sorted(by_index)) != tuple(range(len(by_index))) or len(by_index) > 2:
        raise ValueError("sequential Q arenas must be dense indices bounded by two")
    tokens = []
    capacities = []
    bases = []
    for arena_index in range(len(by_index)):
        group = tuple(by_index[arena_index])
        group_tokens = {item.packed_projection_view.storage_token for item in group}
        if len(group_tokens) != 1:
            raise ValueError("one Q arena index names multiple physical storages")
        group_capacities = {
            item.arena_capacity_nbytes
            if item.arena_capacity_nbytes > 0
            else item.packed_projection_view.nbytes
            for item in group
        }
        if len(group_capacities) != 1:
            raise ValueError("one Q arena index has inconsistent capacities")
        group_bases = {
            item.packed_projection_view.version - item.version_offset
            for item in group
        }
        if len(group_bases) != 1 or next(iter(group_bases)) < 0:
            raise ValueError("one Q arena index has inconsistent base versions")
        tokens.append(next(iter(group_tokens)))
        capacities.append(next(iter(group_capacities)))
        bases.append(next(iter(group_bases)))
    result = SequentialQArenaBinding(
        arena_tokens=tuple(tokens),
        arena_capacities_nbytes=tuple(capacities),
        arena_base_versions=tuple(bases),
        layer_writes=tuple(
            SequentialQArenaWriteBinding(
                layer_id=item.layer_id,
                arena_index=item.arena_index,
                write_ordinal=item.write_ordinal,
                write_count=item.write_count,
                version_offset=item.version_offset,
            )
            for item in bindings
        ),
    )
    result.validate_sparse_bindings(bindings)
    return result


@dataclass(frozen=True, order=True)
class CacheDomainBinding:
    layer_id: int
    compression_ratio: int
    component: str
    total_units: int
    restored_units: int
    dirty_units: int
    artifact_digest: str
    restore_plan_digest: str
    gpu_view: GpuViewBinding
    builder_preflight_token: str

    def __post_init__(self) -> None:
        if _strict_int(self.layer_id, "layer_id") < 0:
            raise ValueError("layer_id must be non-negative")
        if _strict_int(self.compression_ratio, "compression_ratio") not in (4, 128):
            raise ValueError("cache domain must belong to C4 or C128")
        if self.component not in _CACHE_COMPONENTS:
            raise ValueError("unknown cache component")
        total = _strict_int(self.total_units, "total_units")
        restored = _strict_int(self.restored_units, "restored_units")
        dirty = _strict_int(self.dirty_units, "dirty_units")
        if total <= 0 or restored < 0 or dirty < 0 or restored + dirty != total:
            raise ValueError("cache restored/dirty unit accounting is inconsistent")
        _sha_digest(self.artifact_digest, "artifact_digest")
        _sha_digest(self.restore_plan_digest, "restore_plan_digest")
        if not isinstance(self.gpu_view, GpuViewBinding):
            raise TypeError("gpu_view must be a GPU view binding")
        _nonempty(self.builder_preflight_token, "builder_preflight_token")
        if self.component == C4 and self.compression_ratio != 4:
            raise ValueError("C4 cache component requires ratio 4")
        if self.component == C128 and self.compression_ratio != 128:
            raise ValueError("C128 cache component requires ratio 128")
        if self.component in (INDEXER, INDEXER_COMPRESSOR_STATE) and self.compression_ratio != 4:
            raise ValueError("Indexer domains exist only on C4 layers")

    @property
    def key(self) -> str:
        return f"{self.layer_id:06d}:{self.component}"

    def as_payload(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "compression_ratio": self.compression_ratio,
            "component": self.component,
            "total_units": self.total_units,
            "restored_units": self.restored_units,
            "dirty_units": self.dirty_units,
            "artifact_digest": self.artifact_digest,
            "restore_plan_digest": self.restore_plan_digest,
            "gpu_view": self.gpu_view,
            "builder_preflight_token": self.builder_preflight_token,
        }


@dataclass(frozen=True, order=True)
class ZOffGpuViewBinding:
    layer_id: int
    artifact_digest: str
    clean_rows: int
    dirty_rows: int
    local_head_count: int
    gpu_view: GpuViewBinding
    persistent_arena_token: str
    merge_kernel_token: str
    residency: str = ZOFF_RESIDENCY

    def __post_init__(self) -> None:
        if _strict_int(self.layer_id, "layer_id") < 0:
            raise ValueError("layer_id must be non-negative")
        _sha_digest(self.artifact_digest, "artifact_digest")
        clean = _strict_int(self.clean_rows, "clean_rows")
        dirty = _strict_int(self.dirty_rows, "dirty_rows")
        if clean < 0 or dirty < 0 or clean + dirty <= 0:
            raise ValueError("z_off row counts are invalid")
        if _strict_int(self.local_head_count, "local_head_count") <= 0:
            raise ValueError("local_head_count must be positive")
        if not isinstance(self.gpu_view, GpuViewBinding):
            raise TypeError("gpu_view must be a GPU view binding")
        _nonempty(self.persistent_arena_token, "persistent_arena_token")
        _nonempty(self.merge_kernel_token, "merge_kernel_token")
        if self.residency != ZOFF_RESIDENCY:
            raise ValueError("z_off must be a persistent GPU-resident view")

    def as_payload(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "artifact_digest": self.artifact_digest,
            "clean_rows": self.clean_rows,
            "dirty_rows": self.dirty_rows,
            "local_head_count": self.local_head_count,
            "gpu_view": self.gpu_view,
            "persistent_arena_token": self.persistent_arena_token,
            "merge_kernel_token": self.merge_kernel_token,
            "residency": self.residency,
        }


def _expected_cache_keys(
    compression_by_layer: Mapping[int, int],
) -> Tuple[str, ...]:
    keys = []
    for layer_id, ratio in sorted(compression_by_layer.items()):
        components = [SWA, ATTENTION_COMPRESSOR_STATE]
        if ratio == 4:
            components.extend((C4, INDEXER, INDEXER_COMPRESSOR_STATE))
        else:
            components.append(C128)
        keys.extend(f"{layer_id:06d}:{component}" for component in components)
    return tuple(sorted(keys))


@dataclass(frozen=True)
class CompositeForwardProposal:
    identity: ForwardIdentity
    tp_rank: int
    ragged: RaggedBatchGeometry
    shared_latent: SharedLatentBinding
    sparse_q: Tuple[SparseQBinding, ...]
    cache_domains: Tuple[CacheDomainBinding, ...]
    z_off_views: Tuple[ZOffGpuViewBinding, ...]
    rank_local_batch_plan_digest: str
    persistent_zoff_arena_token: str
    fused_merge_kernel_token: str
    restore_provider_token: str
    restore_provider_local_token: str
    restore_batch_common_digest: str
    restore_batch_local_digest: str
    failure_carrier_view: Optional[GpuViewBinding] = None
    sequential_q_arena: Optional[SequentialQArenaBinding] = None
    omission_profile: str = OMISSION_PROFILE_FULL
    commit_scope: str = COMMIT_SCOPE_COMPLETED
    format_version: int = COMPOSITE_PROPOSAL_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ForwardIdentity):
            raise TypeError("identity must be a ForwardIdentity")
        rank = _strict_int(self.tp_rank, "tp_rank")
        if rank < 0 or rank >= self.identity.tp_size:
            raise ValueError("proposal tp_rank is outside the TP group")
        if not isinstance(self.ragged, RaggedBatchGeometry):
            raise TypeError("ragged must be RaggedBatchGeometry")
        if not isinstance(self.shared_latent, SharedLatentBinding):
            raise TypeError("shared_latent must be SharedLatentBinding")
        if self.format_version != COMPOSITE_PROPOSAL_FORMAT_VERSION:
            raise ValueError("composite proposal format is incompatible")
        _sha_digest(
            self.rank_local_batch_plan_digest,
            "rank_local_batch_plan_digest",
        )
        _nonempty(self.persistent_zoff_arena_token, "persistent_zoff_arena_token")
        _nonempty(self.fused_merge_kernel_token, "fused_merge_kernel_token")
        _sha_digest(self.restore_provider_token, "restore_provider_token")
        _sha_digest(
            self.restore_provider_local_token,
            "restore_provider_local_token",
        )
        _sha_digest(
            self.restore_batch_common_digest,
            "restore_batch_common_digest",
        )
        _sha_digest(
            self.restore_batch_local_digest,
            "restore_batch_local_digest",
        )
        if self.omission_profile not in OMISSION_PROFILES:
            raise ValueError("composite omission profile is invalid")
        if self.commit_scope not in COMMIT_SCOPES:
            raise ValueError("composite commit scope is invalid")
        if self.shared_latent.clean_rows + self.shared_latent.dirty_rows != self.ragged.total_rows:
            raise ValueError("shared-latent rows do not match ragged total_rows")

        sparse = _immutable_tuple(self.sparse_q, "sparse_q")
        caches = _immutable_tuple(self.cache_domains, "cache_domains")
        z_off = _immutable_tuple(self.z_off_views, "z_off_views")
        if any(not isinstance(item, SparseQBinding) for item in sparse):
            raise TypeError("sparse_q contains an invalid binding")
        if any(not isinstance(item, CacheDomainBinding) for item in caches):
            raise TypeError("cache_domains contains an invalid binding")
        if any(not isinstance(item, ZOffGpuViewBinding) for item in z_off):
            raise TypeError("z_off_views contains an invalid binding")
        if sparse != tuple(sorted(sparse, key=lambda item: item.layer_id)):
            raise ValueError("sparse_q must be layer-sorted")
        if caches != tuple(sorted(caches, key=lambda item: item.key)):
            raise ValueError("cache_domains must be key-sorted")
        if z_off != tuple(sorted(z_off, key=lambda item: item.layer_id)):
            raise ValueError("z_off_views must be layer-sorted")

        layers = self.shared_latent.layer_ids
        sparse_layers = tuple(item.layer_id for item in sparse)
        zoff_layers = tuple(item.layer_id for item in z_off)
        headsplit_profile = self.omission_profile in (
            OMISSION_PROFILE_FULL,
            OMISSION_PROFILE_ZOFF_ONLY,
        )
        if headsplit_profile:
            if sparse_layers != layers or zoff_layers != layers:
                raise ValueError(
                    "sparse-Q/z_off must cover every head-split layer exactly"
                )
        elif sparse or z_off:
            # ``shared_only`` is an attribution profile, not a head-split
            # execution.  Its certificate authorizes only clean shared-cache
            # rows; carrying unused Q/z_off reservations would both pin a
            # large arena and make the proposal overstate its consumers.
            raise ValueError(
                "shared-only proposal cannot bind sparse-Q or z_off views"
            )
        if len(set(item.key for item in caches)) != len(caches):
            raise ValueError("cache domain keys must be unique")
        expected_keys = _expected_cache_keys(self.shared_latent.compression_by_layer)
        observed_keys = tuple(item.key for item in caches)
        if self.omission_profile == OMISSION_PROFILE_ZOFF_ONLY:
            if observed_keys:
                raise ValueError(
                    "zoff-only proposal cannot bind shared cache domains"
                )
        elif observed_keys != expected_keys:
            raise ValueError(
                "cache domains do not cover SWA/C4/C128/Indexer/state exactly"
            )

        ratios = self.shared_latent.compression_by_layer
        for item in sparse:
            if item.tp_rank != rank or item.tp_size != self.identity.tp_size:
                raise ValueError("sparse-Q binding belongs to another TP rank/group")
            if item.q_rows != self.ragged.total_rows:
                raise ValueError("sparse-Q rows do not match ragged total_rows")
        for item in caches:
            if item.compression_ratio != ratios[item.layer_id]:
                raise ValueError("cache component compression ratio changed")
            if item.restore_plan_digest != self.shared_latent.restore_plan_digest:
                raise ValueError("cache component uses another restore plan")
        for item in z_off:
            if item.clean_rows != self.shared_latent.clean_rows or item.dirty_rows != self.shared_latent.dirty_rows:
                raise ValueError("z_off clean/dirty rows changed")
            if item.clean_rows + item.dirty_rows != self.ragged.total_rows:
                raise ValueError("z_off rows do not match ragged total_rows")
            if item.gpu_view.shape[0] != self.ragged.total_rows:
                raise ValueError("z_off GPU view does not span the packed ragged rows")
            if item.persistent_arena_token != self.persistent_zoff_arena_token:
                raise ValueError("z_off view belongs to another persistent arena")
            if item.merge_kernel_token != self.fused_merge_kernel_token:
                raise ValueError("z_off view belongs to another fused merge kernel")
        if self.commit_scope in (
            COMMIT_SCOPE_FORWARD_FRAGMENT,
            COMMIT_SCOPE_FORWARD_RESERVED,
        ):
            if not isinstance(self.failure_carrier_view, GpuViewBinding):
                raise ValueError(
                    "forward reservation requires a preallocated failure carrier"
                )
            carrier_device = self.failure_carrier_view.device_index
            bound_devices = {
                item.gpu_view.device_index for item in caches
            }
            bound_devices.update(
                item.packed_projection_view.device_index for item in sparse
            )
            bound_devices.update(item.gpu_view.device_index for item in z_off)
            if bound_devices != {carrier_device}:
                raise ValueError(
                    "failure carrier and reserved GPU views use different devices"
                )
            if self.commit_scope == COMMIT_SCOPE_FORWARD_FRAGMENT:
                if len(layers) != 1:
                    raise ValueError(
                        "forward reservation fragment must cover exactly one layer"
                    )
                if self.sequential_q_arena is not None:
                    raise ValueError(
                        "forward reservation fragment cannot claim an aggregate Q arena"
                    )
            elif headsplit_profile:
                if not isinstance(
                    self.sequential_q_arena, SequentialQArenaBinding
                ):
                    raise ValueError(
                        "head-split forward reservation requires a sequential Q arena"
                    )
                self.sequential_q_arena.validate_sparse_bindings(sparse)
            elif self.sequential_q_arena is not None:
                raise ValueError(
                    "shared-only forward cannot reserve a sequential Q arena"
                )
        elif self.sequential_q_arena is not None or self.failure_carrier_view is not None:
            raise ValueError(
                "completed layer proposal cannot claim forward-only arenas"
            )

    @property
    def reusable_layer_ids(self) -> Tuple[int, ...]:
        return self.shared_latent.layer_ids

    @property
    def cache_domain_keys(self) -> Tuple[str, ...]:
        return tuple(item.key for item in self.cache_domains)

    @property
    def omission_slots(self) -> Tuple[str, ...]:
        slots = []
        if self.omission_profile in (
            OMISSION_PROFILE_FULL,
            OMISSION_PROFILE_ZOFF_ONLY,
        ):
            slots.extend(
                f"sparse_q:{layer_id:06d}"
                for layer_id in self.reusable_layer_ids
            )
            slots.extend(
                f"z_off:{layer_id:06d}" for layer_id in self.reusable_layer_ids
            )
        if self.omission_profile in (
            OMISSION_PROFILE_FULL,
            OMISSION_PROFILE_SHARED_ONLY,
        ):
            slots.extend(f"cache:{key}" for key in self.cache_domain_keys)
        return tuple(sorted(slots))

    def shared_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "identity": self.identity,
            "ragged": self.ragged,
            "shared_latent": self.shared_latent,
            "reusable_layer_ids": self.reusable_layer_ids,
            "omission_profile": self.omission_profile,
            "commit_scope": self.commit_scope,
            "restore_provider_token": self.restore_provider_token,
            "restore_batch_common_digest": self.restore_batch_common_digest,
            "failure_carrier": (
                None
                if self.failure_carrier_view is None
                else {
                    "dtype": self.failure_carrier_view.dtype,
                    "shape": self.failure_carrier_view.shape,
                    "strides": self.failure_carrier_view.strides,
                    "nbytes": self.failure_carrier_view.nbytes,
                }
            ),
            "sequential_q_arena": (
                None
                if self.sequential_q_arena is None
                else self.sequential_q_arena.shared_payload()
            ),
        }

    def rank_local_payload(self) -> dict[str, object]:
        return {
            "tp_rank": self.tp_rank,
            "batch_plan_digest": self.rank_local_batch_plan_digest,
            "shared_latent_epoch_pins": self.shared_latent.rank_local_payload(),
            "sparse_q": self.sparse_q,
            "cache_domains": self.cache_domains,
            "z_off_views": self.z_off_views,
            "persistent_zoff_arena_token": self.persistent_zoff_arena_token,
            "fused_merge_kernel_token": self.fused_merge_kernel_token,
            "restore_provider_local_token": self.restore_provider_local_token,
            "restore_batch_local_digest": self.restore_batch_local_digest,
            "failure_carrier_view": self.failure_carrier_view,
            "sequential_q_arena": (
                None
                if self.sequential_q_arena is None
                else self.sequential_q_arena.rank_local_payload()
            ),
        }

    @property
    def shared_digest(self) -> str:
        return _digest(self.shared_payload())

    @property
    def rank_local_digest(self) -> str:
        return _digest(self.rank_local_payload())

    @property
    def digest(self) -> str:
        return _digest(
            {
                "shared_digest": self.shared_digest,
                "rank_local_digest": self.rank_local_digest,
            }
        )


@dataclass(frozen=True)
class CacheBuildersPreflight:
    generation_id: str
    forward_ordinal: int
    tp_rank: int
    tp_size: int
    proposal_digest: str
    shared_digest: str
    ragged_digest: str
    shared_latent_digest: str
    completed_cache_domains: Tuple[str, ...]
    completed_sparse_q_layers: Tuple[int, ...]
    completed_zoff_layers: Tuple[int, ...]
    builder_epoch_token: str
    omitted_slots_consumed: bool
    ready: bool
    failure_code: str = ""
    format_version: int = COMPOSITE_PREFLIGHT_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        rank = _strict_int(self.tp_rank, "tp_rank")
        size = _strict_int(self.tp_size, "tp_size")
        if size <= 0 or rank < 0 or rank >= size:
            raise ValueError("preflight TP geometry is invalid")
        _sha_digest(self.proposal_digest, "proposal_digest")
        _sha_digest(self.shared_digest, "shared_digest")
        _sha_digest(self.ragged_digest, "ragged_digest")
        _sha_digest(self.shared_latent_digest, "shared_latent_digest")
        _sorted_unique_strings(
            self.completed_cache_domains,
            name="completed_cache_domains",
            allow_empty=True,
        )
        _sorted_unique_ints(
            self.completed_sparse_q_layers,
            name="completed_sparse_q_layers",
            allow_empty=True,
        )
        _sorted_unique_ints(
            self.completed_zoff_layers,
            name="completed_zoff_layers",
            allow_empty=True,
        )
        _nonempty(self.builder_epoch_token, "builder_epoch_token")
        if type(self.omitted_slots_consumed) is not bool or type(self.ready) is not bool:
            raise TypeError("preflight ready/omitted flags must be booleans")
        if self.omitted_slots_consumed:
            raise CommitProtocolError(
                "an omitted slot was consumed before the composite commit"
            )
        if self.ready and self.failure_code:
            raise ValueError("a ready preflight cannot carry failure_code")
        if not self.ready and not self.failure_code:
            raise ValueError("a rejected preflight requires failure_code")
        if self.format_version != COMPOSITE_PREFLIGHT_FORMAT_VERSION:
            raise ValueError("composite preflight format is incompatible")

    def validate(self, proposal: CompositeForwardProposal) -> None:
        if proposal.commit_scope != COMMIT_SCOPE_COMPLETED:
            raise CommitProtocolError(
                "completed-builder preflight cannot certify reserved slots"
            )
        identity = proposal.identity
        actual = (
            self.generation_id,
            self.forward_ordinal,
            self.tp_rank,
            self.tp_size,
            self.proposal_digest,
            self.shared_digest,
            self.ragged_digest,
            self.shared_latent_digest,
        )
        expected = (
            identity.generation_id,
            identity.forward_ordinal,
            proposal.tp_rank,
            identity.tp_size,
            proposal.digest,
            proposal.shared_digest,
            proposal.ragged.digest,
            proposal.shared_latent.digest,
        )
        if actual != expected:
            raise CommitProtocolError("cache-builder preflight binds another proposal")
        if self.ready:
            if self.completed_cache_domains != proposal.cache_domain_keys:
                raise CommitProtocolError("ready preflight omitted a cache domain")
            if self.completed_sparse_q_layers != proposal.reusable_layer_ids:
                raise CommitProtocolError("ready preflight omitted a sparse-Q layer")
            if self.completed_zoff_layers != proposal.reusable_layer_ids:
                raise CommitProtocolError("ready preflight omitted a z_off layer")

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "generation_id": self.generation_id,
            "forward_ordinal": self.forward_ordinal,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "proposal_digest": self.proposal_digest,
            "shared_digest": self.shared_digest,
            "ragged_digest": self.ragged_digest,
            "shared_latent_digest": self.shared_latent_digest,
            "completed_cache_domains": self.completed_cache_domains,
            "completed_sparse_q_layers": self.completed_sparse_q_layers,
            "completed_zoff_layers": self.completed_zoff_layers,
            "builder_epoch_token": self.builder_epoch_token,
            "omitted_slots_consumed": self.omitted_slots_consumed,
            "ready": self.ready,
            "failure_code": self.failure_code,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())


def build_cache_builders_preflight(
    proposal: CompositeForwardProposal,
    *,
    builder_epoch_token: str,
    ready: bool = True,
    failure_code: str = "",
    completed_cache_domains: Optional[Sequence[str]] = None,
    completed_sparse_q_layers: Optional[Sequence[int]] = None,
    completed_zoff_layers: Optional[Sequence[int]] = None,
    omitted_slots_consumed: bool = False,
) -> CacheBuildersPreflight:
    """Build and validate the receipt issued after all local preflight work."""

    if not isinstance(proposal, CompositeForwardProposal):
        raise TypeError("proposal has an invalid type")
    caches = (
        proposal.cache_domain_keys
        if completed_cache_domains is None
        else tuple(completed_cache_domains)
    )
    sparse = (
        proposal.reusable_layer_ids
        if completed_sparse_q_layers is None
        else tuple(completed_sparse_q_layers)
    )
    zoff = (
        proposal.reusable_layer_ids
        if completed_zoff_layers is None
        else tuple(completed_zoff_layers)
    )
    report = CacheBuildersPreflight(
        generation_id=proposal.identity.generation_id,
        forward_ordinal=proposal.identity.forward_ordinal,
        tp_rank=proposal.tp_rank,
        tp_size=proposal.identity.tp_size,
        proposal_digest=proposal.digest,
        shared_digest=proposal.shared_digest,
        ragged_digest=proposal.ragged.digest,
        shared_latent_digest=proposal.shared_latent.digest,
        completed_cache_domains=tuple(caches),
        completed_sparse_q_layers=tuple(sparse),
        completed_zoff_layers=tuple(zoff),
        builder_epoch_token=builder_epoch_token,
        omitted_slots_consumed=omitted_slots_consumed,
        ready=ready,
        failure_code=failure_code,
    )
    report.validate(proposal)
    return report


@dataclass(frozen=True, order=True)
class LayerReservationBinding:
    """Immutable post-write slot contract for one reusable layer.

    A forward-scoped prepare certificate cannot claim that future sparse-Q
    values have already been computed: those values depend on the hidden state
    produced by preceding layers.  It can, however, bind the exact destination
    views, their expected post-write versions, every shared-cache target and
    the persistent z_off view.  A :class:`LayerExecutionReceipt` later proves
    that the layer reached this reserved state before an omission is consumed.
    """

    layer_id: int
    sparse_q_binding_digest: str
    cache_binding_digest: str
    zoff_binding_digest: str
    gpu_view_versions: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        if _strict_int(self.layer_id, "layer_id") < 0:
            raise ValueError("layer reservation id must be non-negative")
        _sha_digest(self.sparse_q_binding_digest, "sparse_q_binding_digest")
        _sha_digest(self.cache_binding_digest, "cache_binding_digest")
        _sha_digest(self.zoff_binding_digest, "zoff_binding_digest")
        versions = _immutable_tuple(self.gpu_view_versions, "gpu_view_versions")
        if not versions:
            raise ValueError("layer reservation must bind GPU view versions")
        normalized = []
        for entry in versions:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("GPU version entries must be immutable pairs")
            key, version = entry
            normalized.append(
                (_nonempty(key, "GPU version key"), _strict_int(version, "GPU version"))
            )
            if version < 0:
                raise ValueError("GPU view versions must be non-negative")
        if tuple(normalized) != tuple(sorted(set(normalized))):
            raise ValueError("GPU version bindings must be sorted and unique")

    def as_payload(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "sparse_q_binding_digest": self.sparse_q_binding_digest,
            "cache_binding_digest": self.cache_binding_digest,
            "zoff_binding_digest": self.zoff_binding_digest,
            "gpu_view_versions": self.gpu_view_versions,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())


def build_layer_reservation_binding(
    proposal: CompositeForwardProposal,
    layer_id: int,
) -> LayerReservationBinding:
    """Derive the exact layer reservation transitively bound by a proposal."""

    if not isinstance(proposal, CompositeForwardProposal):
        raise TypeError("proposal has an invalid type")
    layer_id = _strict_int(layer_id, "layer_id")
    if layer_id not in proposal.reusable_layer_ids:
        raise ValueError("layer is outside the proposal's reusable domain")
    sparse = tuple(item for item in proposal.sparse_q if item.layer_id == layer_id)
    zoff = tuple(item for item in proposal.z_off_views if item.layer_id == layer_id)
    caches = tuple(
        item for item in proposal.cache_domains if item.layer_id == layer_id
    )
    if proposal.omission_profile == OMISSION_PROFILE_ZOFF_ONLY:
        if caches:
            raise CommitProtocolError(
                "zoff-only proposal unexpectedly reserved cache domains"
            )
    elif not caches:
        raise CommitProtocolError("proposal has incomplete cache reservations")
    if proposal.omission_profile == OMISSION_PROFILE_SHARED_ONLY:
        if sparse or zoff:
            raise CommitProtocolError(
                "shared-only layer unexpectedly has Q/z_off reservations"
            )
        sparse_digest = _digest(())
        zoff_digest = _digest(())
        versions = []
    else:
        if len(sparse) != 1 or len(zoff) != 1:
            raise CommitProtocolError(
                "head-split proposal has incomplete layer reservations"
            )
        sparse_digest = _digest(sparse[0])
        zoff_digest = _digest(zoff[0])
        versions = [
            ("sparse_q", sparse[0].packed_projection_view.version),
            ("z_off", zoff[0].gpu_view.version),
        ]
    versions.extend(
        (f"cache:{item.component}", item.gpu_view.version) for item in caches
    )
    return LayerReservationBinding(
        layer_id=layer_id,
        sparse_q_binding_digest=sparse_digest,
        cache_binding_digest=_digest(caches),
        zoff_binding_digest=zoff_digest,
        gpu_view_versions=tuple(sorted(versions)),
    )


@dataclass(frozen=True)
class ForwardPreparePreflight:
    """Forward-wide reservation receipt used before layers 3..39 execute.

    Unlike :class:`CacheBuildersPreflight`, this receipt does not falsely say
    that all future sparse-Q values already exist.  It certifies exact reserved
    post-write GPU views for every reusable layer.  Per-layer execution
    receipts and the fixed final rendezvous close the temporal gap.
    """

    generation_id: str
    forward_ordinal: int
    tp_rank: int
    tp_size: int
    proposal_digest: str
    shared_digest: str
    ragged_digest: str
    shared_latent_digest: str
    layer_reservations: Tuple[LayerReservationBinding, ...]
    builder_epoch_token: str
    omitted_slots_consumed: bool
    ready: bool
    failure_code: str = ""
    format_version: int = COMPOSITE_PREFLIGHT_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        rank = _strict_int(self.tp_rank, "tp_rank")
        size = _strict_int(self.tp_size, "tp_size")
        if size <= 0 or rank < 0 or rank >= size:
            raise ValueError("forward prepare TP geometry is invalid")
        _sha_digest(self.proposal_digest, "proposal_digest")
        _sha_digest(self.shared_digest, "shared_digest")
        _sha_digest(self.ragged_digest, "ragged_digest")
        _sha_digest(self.shared_latent_digest, "shared_latent_digest")
        reservations = _immutable_tuple(
            self.layer_reservations, "layer_reservations"
        )
        if not reservations or any(
            not isinstance(item, LayerReservationBinding) for item in reservations
        ):
            raise ValueError("forward prepare must reserve every reusable layer")
        if reservations != tuple(sorted(set(reservations))):
            raise ValueError("layer reservations must be sorted and unique")
        _nonempty(self.builder_epoch_token, "builder_epoch_token")
        if type(self.omitted_slots_consumed) is not bool or type(self.ready) is not bool:
            raise TypeError("prepare ready/omitted flags must be booleans")
        if self.omitted_slots_consumed:
            raise CommitProtocolError(
                "an omitted slot was consumed before the forward prepare commit"
            )
        if self.ready and self.failure_code:
            raise ValueError("a ready forward prepare cannot carry failure_code")
        if not self.ready and not self.failure_code:
            raise ValueError("a rejected forward prepare requires failure_code")
        if self.format_version != COMPOSITE_PREFLIGHT_FORMAT_VERSION:
            raise ValueError("forward prepare format is incompatible")

    def validate(self, proposal: CompositeForwardProposal) -> None:
        if proposal.commit_scope != COMMIT_SCOPE_FORWARD_RESERVED:
            raise CommitProtocolError(
                "forward prepare requires reserved post-write slot scope"
            )
        expected_identity = (
            proposal.identity.generation_id,
            proposal.identity.forward_ordinal,
            proposal.tp_rank,
            proposal.identity.tp_size,
            proposal.digest,
            proposal.shared_digest,
            proposal.ragged.digest,
            proposal.shared_latent.digest,
        )
        actual_identity = (
            self.generation_id,
            self.forward_ordinal,
            self.tp_rank,
            self.tp_size,
            self.proposal_digest,
            self.shared_digest,
            self.ragged_digest,
            self.shared_latent_digest,
        )
        if actual_identity != expected_identity:
            raise CommitProtocolError("forward prepare binds another proposal")
        expected_reservations = tuple(
            build_layer_reservation_binding(proposal, layer_id)
            for layer_id in proposal.reusable_layer_ids
        )
        if self.layer_reservations != expected_reservations:
            raise CommitProtocolError("forward prepare layer reservation changed")

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "generation_id": self.generation_id,
            "forward_ordinal": self.forward_ordinal,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "proposal_digest": self.proposal_digest,
            "shared_digest": self.shared_digest,
            "ragged_digest": self.ragged_digest,
            "shared_latent_digest": self.shared_latent_digest,
            "layer_reservations": self.layer_reservations,
            "builder_epoch_token": self.builder_epoch_token,
            "omitted_slots_consumed": self.omitted_slots_consumed,
            "ready": self.ready,
            "failure_code": self.failure_code,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())


def build_forward_prepare_preflight(
    proposal: CompositeForwardProposal,
    *,
    builder_epoch_token: str,
    ready: bool = True,
    failure_code: str = "",
    omitted_slots_consumed: bool = False,
) -> ForwardPreparePreflight:
    """Build the reservation receipt for the forward's sole prepare vote."""

    if not isinstance(proposal, CompositeForwardProposal):
        raise TypeError("proposal has an invalid type")
    if proposal.commit_scope != COMMIT_SCOPE_FORWARD_RESERVED:
        raise CommitProtocolError(
            "forward prepare requires reserved post-write slot scope"
        )
    report = ForwardPreparePreflight(
        generation_id=proposal.identity.generation_id,
        forward_ordinal=proposal.identity.forward_ordinal,
        tp_rank=proposal.tp_rank,
        tp_size=proposal.identity.tp_size,
        proposal_digest=proposal.digest,
        shared_digest=proposal.shared_digest,
        ragged_digest=proposal.ragged.digest,
        shared_latent_digest=proposal.shared_latent.digest,
        layer_reservations=tuple(
            build_layer_reservation_binding(proposal, layer_id)
            for layer_id in proposal.reusable_layer_ids
        ),
        builder_epoch_token=builder_epoch_token,
        omitted_slots_consumed=omitted_slots_consumed,
        ready=ready,
        failure_code=failure_code,
    )
    report.validate(proposal)
    return report


@dataclass(frozen=True)
class CommitVotePayload:
    """The complete, compact payload for the forward's only collective."""

    generation_id: str
    forward_ordinal: int
    tp_rank: int
    tp_size: int
    execution_identity_digest: str
    proposal_digest: str
    shared_digest: str
    rank_local_digest: str
    preflight_digest: str
    ready: bool
    failure_code: str
    format_version: int = COMPOSITE_COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        rank = _strict_int(self.tp_rank, "tp_rank")
        size = _strict_int(self.tp_size, "tp_size")
        if size <= 0 or size > MAX_TP_SIZE or rank < 0 or rank >= size:
            raise ValueError("vote TP geometry is invalid")
        _sha_digest(self.execution_identity_digest, "execution_identity_digest")
        _sha_digest(self.proposal_digest, "proposal_digest")
        _sha_digest(self.shared_digest, "shared_digest")
        _sha_digest(self.rank_local_digest, "rank_local_digest")
        _sha_digest(self.preflight_digest, "preflight_digest")
        if type(self.ready) is not bool:
            raise TypeError("vote ready must be boolean")
        if self.ready and self.failure_code:
            raise ValueError("a ready vote cannot carry failure_code")
        if not self.ready and not self.failure_code:
            raise ValueError("a rejected vote requires failure_code")
        if self.format_version != COMPOSITE_COMMIT_FORMAT_VERSION:
            raise ValueError("composite commit payload format is incompatible")

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "generation_id": self.generation_id,
            "forward_ordinal": self.forward_ordinal,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "execution_identity_digest": self.execution_identity_digest,
            "proposal_digest": self.proposal_digest,
            "shared_digest": self.shared_digest,
            "rank_local_digest": self.rank_local_digest,
            "preflight_digest": self.preflight_digest,
            "ready": self.ready,
            "failure_code": self.failure_code,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())

    @property
    def forward_digest(self) -> str:
        return self.execution_identity_digest

    def to_int64_vector(self) -> Tuple[int, ...]:
        """Encode this vote as one fixed-shape SUM-all-reduce contribution.

        Shared digest limbs carry first and second moments.  A reduced vector
        matches a local shared digest only when every rank supplied the same
        value (zero integer variance).  Fifteen-bit limbs keep all sums and
        squared sums safely inside signed int64 for ``MAX_TP_SIZE`` ranks.
        Rank-local proposal/preflight/vote digests are accumulated as a
        manifest fingerprint; they are intentionally not required to agree.
        """

        return _encode_commit_int64_vector(
            format_version=self.format_version,
            ready=self.ready,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            forward_ordinal=self.forward_ordinal,
            forward_digest=self.forward_digest,
            shared_digest=self.shared_digest,
            local_digests=(
                self.proposal_digest,
                self.rank_local_digest,
                self.preflight_digest,
                self.digest,
            ),
        )


def build_commit_vote(
    proposal: CompositeForwardProposal,
    preflight: object,
) -> CommitVotePayload:
    if not isinstance(preflight, (CacheBuildersPreflight, ForwardPreparePreflight)):
        raise TypeError("preflight has an invalid type")
    preflight.validate(proposal)
    return CommitVotePayload(
        generation_id=proposal.identity.generation_id,
        forward_ordinal=proposal.identity.forward_ordinal,
        tp_rank=proposal.tp_rank,
        tp_size=proposal.identity.tp_size,
        execution_identity_digest=_digest(proposal.identity.as_payload()),
        proposal_digest=proposal.digest,
        shared_digest=proposal.shared_digest,
        rank_local_digest=proposal.rank_local_digest,
        preflight_digest=preflight.digest,
        ready=preflight.ready,
        failure_code=preflight.failure_code,
    )


def _digest_limbs(digest: str) -> Tuple[int, ...]:
    _sha_digest(digest, "digest")
    value = int(digest[7:], 16)
    mask = (1 << _DIGEST_LIMB_BITS) - 1
    return tuple(
        (value >> (index * _DIGEST_LIMB_BITS)) & mask
        for index in range(_DIGEST_LIMB_COUNT)
    )


_COMMIT_HEADER_LENGTH = 5
_COMMIT_COMMON_MOMENTS_LENGTH = 2 * 2 * _DIGEST_LIMB_COUNT
_COMMIT_LOCAL_MANIFEST_LENGTH = 4 * _DIGEST_LIMB_COUNT
COMMIT_INT64_VECTOR_LENGTH = (
    _COMMIT_HEADER_LENGTH
    + MAX_TP_SIZE
    + _COMMIT_COMMON_MOMENTS_LENGTH
    + _COMMIT_LOCAL_MANIFEST_LENGTH
)


def _encode_commit_int64_vector(
    *,
    format_version: int,
    ready: bool,
    tp_rank: int,
    tp_size: int,
    forward_ordinal: int,
    forward_digest: str,
    shared_digest: str,
    local_digests: Tuple[str, str, str, str],
) -> Tuple[int, ...]:
    """Encode the fixed collective shape independently of vote construction."""

    vector = [
        int(format_version),
        1,
        int(bool(ready)),
        int(tp_size),
        int(forward_ordinal),
    ]
    vector.extend(1 if index == int(tp_rank) else 0 for index in range(MAX_TP_SIZE))
    for common_digest in (forward_digest, shared_digest):
        for limb in _digest_limbs(common_digest):
            vector.extend((limb, limb * limb))
    for local_digest in local_digests:
        vector.extend(_digest_limbs(local_digest))
    if len(vector) != COMMIT_INT64_VECTOR_LENGTH:
        raise AssertionError("commit vector layout changed unexpectedly")
    return tuple(vector)


def _build_rejected_commit_vector(
    proposal: CompositeForwardProposal,
    *,
    execution_identity_digest: str,
) -> Tuple[int, ...]:
    """Build a no-throw-at-commit-time contribution for local failures.

    This is constructed when the session is created, before preflight fault
    injection is possible.  It preserves the common forward identity and rank
    slot, but its ready bit is permanently zero.  Local manifest limbs are an
    opaque constant because no omission can be authorized from this vote.
    """

    rejected = _digest(
        {
            "kind": "composite-local-preexchange-rejected-v1",
            "tp_rank": proposal.tp_rank,
        }
    )
    return _encode_commit_int64_vector(
        format_version=COMPOSITE_COMMIT_FORMAT_VERSION,
        ready=False,
        tp_rank=proposal.tp_rank,
        tp_size=proposal.identity.tp_size,
        forward_ordinal=proposal.identity.forward_ordinal,
        forward_digest=execution_identity_digest,
        shared_digest=proposal.shared_digest,
        local_digests=(rejected, rejected, rejected, rejected),
    )


@dataclass(frozen=True)
class CollectiveReduction:
    """Result of exactly one SUM all-reduce over ``to_int64_vector()``."""

    collective_token: str
    reduced_int64: Tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty(self.collective_token, "collective_token")
        values = _immutable_tuple(self.reduced_int64, "reduced_int64")
        if len(values) != COMMIT_INT64_VECTOR_LENGTH:
            raise ValueError("collective reduction has the wrong vector length")
        for value in values:
            value = _strict_int(value, "reduced int64 entry")
            if value < 0 or value > 0x7FFFFFFFFFFFFFFF:
                raise ValueError("collective reduction is outside signed int64")


class CollectiveAdapter(Protocol):
    """Serving-owned bridge; implementations perform the real coordination.

    ``exchange_commit_once`` is the sole TP collective for this protocol.  It
    performs SUM all-reduce over the provided fixed-shape signed-int64 vector.
    ``coordinated_abort`` is an out-of-band, fail-stop notification and must
    never be implemented as a second readiness vote that could let ranks
    continue independently.
    """

    def exchange_commit_once(self, int64_vector: Tuple[int, ...]) -> CollectiveReduction:
        ...

    def coordinated_abort(self, signal: "CoordinatedAbortSignal") -> None:
        ...


@dataclass(frozen=True)
class ForwardCommitCertificate:
    generation_id: str
    forward_ordinal: int
    tp_size: int
    collective_token: str
    shared_digest: str
    tp_rank: int
    local_proposal_digest: str
    local_preflight_digest: str
    reduced_int64: Tuple[int, ...]
    manifest_digest: str
    format_version: int = COMPOSITE_COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        size = _strict_int(self.tp_size, "tp_size")
        if size <= 0:
            raise ValueError("tp_size must be positive")
        rank = _strict_int(self.tp_rank, "tp_rank")
        if rank < 0 or rank >= size:
            raise ValueError("certificate tp_rank is outside the TP group")
        _nonempty(self.collective_token, "collective_token")
        _sha_digest(self.shared_digest, "shared_digest")
        _sha_digest(self.local_proposal_digest, "local_proposal_digest")
        _sha_digest(self.local_preflight_digest, "local_preflight_digest")
        reduced = _immutable_tuple(self.reduced_int64, "reduced_int64")
        if len(reduced) != COMMIT_INT64_VECTOR_LENGTH:
            raise ValueError("certificate reduction vector has the wrong length")
        if any(
            _strict_int(value, "certificate reduced int64 entry") < 0
            or value > 0x7FFFFFFFFFFFFFFF
            for value in reduced
        ):
            raise ValueError("certificate reduction is outside signed int64")
        _sha_digest(self.manifest_digest, "manifest_digest")
        expected_manifest = _digest(
            {
                "generation_id": self.generation_id,
                "forward_ordinal": self.forward_ordinal,
                "tp_size": self.tp_size,
                "collective_token": self.collective_token,
                "shared_digest": self.shared_digest,
                "reduced_int64": self.reduced_int64,
            }
        )
        if self.manifest_digest != expected_manifest:
            raise ValueError("certificate manifest digest is invalid")
        if self.format_version != COMPOSITE_COMMIT_FORMAT_VERSION:
            raise ValueError("composite certificate format is incompatible")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "format_version": self.format_version,
                "manifest_digest": self.manifest_digest,
            }
        )

    def validate(
        self,
        proposal: CompositeForwardProposal,
        preflight: object,
    ) -> None:
        if not isinstance(
            preflight, (CacheBuildersPreflight, ForwardPreparePreflight)
        ):
            raise TypeError("preflight has an invalid type")
        preflight.validate(proposal)
        expected_identity = (
            proposal.identity.generation_id,
            proposal.identity.forward_ordinal,
            proposal.identity.tp_size,
            proposal.shared_digest,
            proposal.tp_rank,
            proposal.digest,
            preflight.digest,
        )
        actual_identity = (
            self.generation_id,
            self.forward_ordinal,
            self.tp_size,
            self.shared_digest,
            self.tp_rank,
            self.local_proposal_digest,
            self.local_preflight_digest,
        )
        if actual_identity != expected_identity:
            raise CommitProtocolError("certificate belongs to another forward")
        reduced = self.reduced_int64
        if (
            reduced[0] != self.tp_size * COMPOSITE_COMMIT_FORMAT_VERSION
            or reduced[1] != self.tp_size
            or reduced[2] != self.tp_size
            or reduced[3] != self.tp_size * self.tp_size
            or reduced[4] != self.tp_size * self.forward_ordinal
        ):
            raise CommitProtocolError("certificate reduction header is invalid")
        rank_counts = reduced[
            _COMMIT_HEADER_LENGTH : _COMMIT_HEADER_LENGTH + MAX_TP_SIZE
        ]
        if rank_counts[: self.tp_size] != (1,) * self.tp_size or any(
            rank_counts[self.tp_size :]
        ):
            raise CommitProtocolError("certificate does not cover the TP group")
        if not _common_moments_match(
            reduced,
            common_index=0,
            expected_digest=_digest(proposal.identity.as_payload()),
            tp_size=self.tp_size,
        ):
            raise CommitProtocolError("certificate execution identity is invalid")
        if not _common_moments_match(
            reduced,
            common_index=1,
            expected_digest=proposal.shared_digest,
            tp_size=self.tp_size,
        ):
            raise CommitProtocolError("certificate shared proposal is invalid")


@dataclass(frozen=True)
class DenseFallbackDecision:
    generation_id: str
    forward_ordinal: int
    reason_code: str
    collective_token: str = ""
    vote_manifest_digest: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        _nonempty(self.reason_code, "reason_code")
        if self.vote_manifest_digest:
            _sha_digest(self.vote_manifest_digest, "vote_manifest_digest")


@dataclass(frozen=True)
class CommitOutcome:
    certificate: Optional[ForwardCommitCertificate] = None
    dense_fallback: Optional[DenseFallbackDecision] = None

    def __post_init__(self) -> None:
        if (self.certificate is None) == (self.dense_fallback is None):
            raise ValueError("commit outcome must be certificate xor dense fallback")

    @property
    def committed(self) -> bool:
        return self.certificate is not None


@dataclass(frozen=True)
class OmissionAuthorization:
    generation_id: str
    forward_ordinal: int
    tp_rank: int
    proposal_digest: str
    certificate_digest: str
    allowed_slots: Tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        if _strict_int(self.tp_rank, "tp_rank") < 0:
            raise ValueError("tp_rank must be non-negative")
        _sha_digest(self.proposal_digest, "proposal_digest")
        _sha_digest(self.certificate_digest, "certificate_digest")
        _sorted_unique_strings(self.allowed_slots, name="allowed_slots")

    def validate_slot(
        self,
        slot: str,
        proposal: CompositeForwardProposal,
        certificate: ForwardCommitCertificate,
    ) -> None:
        slot = _nonempty(slot, "slot")
        expected = (
            proposal.identity.generation_id,
            proposal.identity.forward_ordinal,
            proposal.tp_rank,
            proposal.digest,
            certificate.digest,
            proposal.omission_slots,
        )
        actual = (
            self.generation_id,
            self.forward_ordinal,
            self.tp_rank,
            self.proposal_digest,
            self.certificate_digest,
            self.allowed_slots,
        )
        if actual != expected:
            raise CommitProtocolError("omission authorization is stale or foreign")
        if slot not in self.allowed_slots:
            raise CommitProtocolError(f"slot {slot!r} is not authorized for omission")


@dataclass(frozen=True)
class CoordinatedAbortSignal:
    generation_id: str
    forward_ordinal: int
    tp_rank: int
    reason_code: str
    detail: str
    proposal_digest: str
    certificate_digest: str = ""
    collective_token: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        if _strict_int(self.tp_rank, "tp_rank") < 0:
            raise ValueError("tp_rank must be non-negative")
        _nonempty(self.reason_code, "reason_code")
        _nonempty(self.detail, "detail")
        _sha_digest(self.proposal_digest, "proposal_digest")
        if self.certificate_digest:
            _sha_digest(self.certificate_digest, "certificate_digest")


class CommitState(str, Enum):
    NEW = "new"
    PREFLIGHTED = "preflighted"
    DENSE_FALLBACK = "dense_fallback"
    COMMITTED = "committed"
    SPARSE_Q_INSTALLED = "sparse_q_installed"
    OMISSIONS_AUTHORIZED = "omissions_authorized"
    ABORTED = "aborted"


def _manifest_digest(
    *,
    generation_id: str,
    forward_ordinal: int,
    tp_size: int,
    collective_token: str,
    shared_digest: str,
    reduced_int64: Tuple[int, ...],
) -> str:
    return _digest(
        {
            "generation_id": generation_id,
            "forward_ordinal": forward_ordinal,
            "tp_size": tp_size,
            "collective_token": collective_token,
            "shared_digest": shared_digest,
            "reduced_int64": reduced_int64,
        }
    )


def sum_commit_int64_vectors(
    vectors: Sequence[Sequence[int]],
) -> Tuple[int, ...]:
    """Reference SUM reduction for CPU adapters/tests (not a collective)."""

    items = tuple(tuple(vector) for vector in vectors)
    if not items:
        raise ValueError("at least one commit vector is required")
    if any(len(item) != COMMIT_INT64_VECTOR_LENGTH for item in items):
        raise ValueError("commit vector has the wrong fixed length")
    reduced = tuple(sum(item[index] for item in items) for index in range(COMMIT_INT64_VECTOR_LENGTH))
    if any(value < 0 or value > 0x7FFFFFFFFFFFFFFF for value in reduced):
        raise OverflowError("SUM reduction does not fit signed int64")
    return reduced


def _common_moments_match(
    reduced: Tuple[int, ...],
    *,
    common_index: int,
    expected_digest: str,
    tp_size: int,
) -> bool:
    begin = _COMMIT_HEADER_LENGTH + MAX_TP_SIZE
    begin += common_index * 2 * _DIGEST_LIMB_COUNT
    expected = _digest_limbs(expected_digest)
    for index, limb in enumerate(expected):
        if reduced[begin + 2 * index] != tp_size * limb:
            return False
        if reduced[begin + 2 * index + 1] != tp_size * limb * limb:
            return False
    return True


@dataclass(frozen=True)
class SparseQInstallAuthorization:
    """Proof that a registered sparse-Q proposal may now replace dense Q."""

    generation_id: str
    forward_ordinal: int
    tp_rank: int
    proposal_digest: str
    certificate_digest: str
    sparse_q_layers: Tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        if _strict_int(self.tp_rank, "tp_rank") < 0:
            raise ValueError("tp_rank must be non-negative")
        _sha_digest(self.proposal_digest, "proposal_digest")
        _sha_digest(self.certificate_digest, "certificate_digest")
        _sorted_unique_ints(self.sparse_q_layers, name="sparse_q_layers")

    def validate(
        self,
        proposal: CompositeForwardProposal,
        certificate: ForwardCommitCertificate,
    ) -> None:
        expected = (
            proposal.identity.generation_id,
            proposal.identity.forward_ordinal,
            proposal.tp_rank,
            proposal.digest,
            certificate.digest,
            proposal.reusable_layer_ids,
        )
        actual = (
            self.generation_id,
            self.forward_ordinal,
            self.tp_rank,
            self.proposal_digest,
            self.certificate_digest,
            self.sparse_q_layers,
        )
        if actual != expected:
            raise CommitProtocolError("sparse-Q install authorization is stale")


class ForwardCommitSession:
    """One-shot state machine owned by one rank for one forward."""

    def __init__(self, proposal: CompositeForwardProposal) -> None:
        if not isinstance(proposal, CompositeForwardProposal):
            raise TypeError("proposal has an invalid type")
        self.proposal = proposal
        self.state = CommitState.NEW
        self.preflight: Optional[object] = None
        self.certificate: Optional[ForwardCommitCertificate] = None
        self.collective_attempted = False
        self.collective_completed = False
        self.collective_decision = "not_attempted"
        self.local_preexchange_error = ""
        self._proposal_digest = proposal.digest
        self._execution_identity_digest = _digest(
            proposal.identity.as_payload()
        )
        # Prebuild the rejected contribution.  commit() can therefore still
        # enter the one collective if vote/preflight/vector construction later
        # raises BaseException (including cancellation-like exceptions).
        self._rejected_contribution = _build_rejected_commit_vector(
            proposal,
            execution_identity_digest=self._execution_identity_digest,
        )
        # Sparse-Q is registered by the immutable proposal but deliberately
        # not installed in the model/cache builders until after certification.
        self.sparse_q_registered = True
        self.sparse_q_installed = False

    def record_cache_builders_preflight(
        self, preflight: CacheBuildersPreflight
    ) -> None:
        if self.state is not CommitState.NEW:
            raise CommitProtocolError("cache-builder preflight may be recorded once")
        if not isinstance(preflight, CacheBuildersPreflight):
            raise TypeError("preflight has an invalid type")
        preflight.validate(self.proposal)
        self.preflight = preflight
        self.state = CommitState.PREFLIGHTED

    def record_forward_prepare_preflight(
        self, preflight: ForwardPreparePreflight
    ) -> None:
        """Record all-layer reservations for one forward-scoped prepare vote."""

        if self.state is not CommitState.NEW:
            raise CommitProtocolError("forward prepare may be recorded once")
        if not isinstance(preflight, ForwardPreparePreflight):
            raise TypeError("forward prepare has an invalid type")
        preflight.validate(self.proposal)
        self.preflight = preflight
        self.state = CommitState.PREFLIGHTED

    def choose_dense_before_commit(self, reason_code: str) -> CommitOutcome:
        """Choose dense only while no commit collective has been attempted."""

        if self.state not in (CommitState.NEW, CommitState.PREFLIGHTED):
            raise CommitProtocolError("dense fallback is no longer safe")
        if self.collective_attempted:
            raise CommitProtocolError("dense fallback is unsafe after a vote attempt")
        self.state = CommitState.DENSE_FALLBACK
        identity = self.proposal.identity
        return CommitOutcome(
            dense_fallback=DenseFallbackDecision(
                generation_id=identity.generation_id,
                forward_ordinal=identity.forward_ordinal,
                reason_code=_nonempty(reason_code, "reason_code"),
            )
        )

    def _abort(
        self,
        adapter: CollectiveAdapter,
        *,
        reason_code: str,
        detail: str,
        certificate: Optional[ForwardCommitCertificate] = None,
        collective_token: str = "",
    ) -> None:
        identity = self.proposal.identity
        signal = CoordinatedAbortSignal(
            generation_id=identity.generation_id,
            forward_ordinal=identity.forward_ordinal,
            tp_rank=self.proposal.tp_rank,
            reason_code=reason_code,
            detail=detail,
            proposal_digest=self._proposal_digest,
            certificate_digest=(certificate.digest if certificate else ""),
            collective_token=collective_token,
        )
        self.state = CommitState.ABORTED
        try:
            adapter.coordinated_abort(signal)
        except Exception:
            # The local rank must fail even if the external fail-stop notifier
            # itself raises.  The original signal remains the actionable cause.
            pass
        raise CoordinatedAbortRequired(signal)

    def commit(
        self,
        adapter: CollectiveAdapter,
        *,
        local_preexchange_error: str = "",
    ) -> CommitOutcome:
        if self.collective_attempted:
            raise CommitProtocolError(
                "commit may run once; the forward collective was already used"
            )
        vote = None
        contribution = self._rejected_contribution
        try:
            if not isinstance(local_preexchange_error, str):
                raise TypeError("local_preexchange_error must be a string")
            if local_preexchange_error:
                raise CommitProtocolError(local_preexchange_error)
            if self.state is not CommitState.PREFLIGHTED or self.preflight is None:
                raise CommitProtocolError(
                    "commit requires cache-builder preflight and may run once"
                )
            vote = build_commit_vote(self.proposal, self.preflight)
            contribution = vote.to_int64_vector()
        except BaseException as exc:
            # Never strand peers because this rank failed before exchange.
            # The prebuilt vector is fixed-shape, rank-addressed and not-ready.
            self.local_preexchange_error = f"{type(exc).__name__}: {exc}"
        self.collective_attempted = True
        try:
            reduction = adapter.exchange_commit_once(contribution)
        except BaseException as exc:
            self.collective_decision = "indeterminate"
            self._abort(
                adapter,
                reason_code="collective_indeterminate",
                detail=f"commit collective raised {type(exc).__name__}",
            )
            raise AssertionError("unreachable")  # pragma: no cover

        # From this point dense fallback is never safe even if a cancellation
        # or local certificate-construction failure interrupts this rank.  The
        # integration can carry that failure to its fixed post-pipeline TP
        # rendezvous while peers finish the same committed layer.
        self.collective_completed = True
        self.collective_decision = "indeterminate"

        try:
            return self._resolve_completed_collective(adapter, reduction, vote)
        except CoordinatedAbortRequired:
            raise
        except BaseException as exc:
            if self.collective_decision == "indeterminate":
                self._abort(
                    adapter,
                    reason_code="postcollective_decision_indeterminate",
                    detail=(
                        "post-collective decision parsing raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    collective_token=str(
                        getattr(reduction, "collective_token", "")
                    ),
                )
            # accepted/rejected are already durable decisions.  The runtime
            # builder converts their local wrapper failure using that marker.
            raise

    def _resolve_completed_collective(
        self,
        adapter: CollectiveAdapter,
        reduction: object,
        vote: Optional[CommitVotePayload],
    ) -> CommitOutcome:
        if not isinstance(reduction, CollectiveReduction):
            self._abort(
                adapter,
                reason_code="malformed_collective_result",
                detail="adapter returned an invalid SUM reduction",
            )
        identity = self.proposal.identity
        reduced = reduction.reduced_int64
        if len(reduced) != COMMIT_INT64_VECTOR_LENGTH:
            self._abort(
                adapter,
                reason_code="malformed_collective_result",
                detail="SUM reduction has the wrong fixed length",
                collective_token=reduction.collective_token,
            )
        if (
            reduced[0] != identity.tp_size * COMPOSITE_COMMIT_FORMAT_VERSION
            or reduced[1] != identity.tp_size
            or reduced[3] != identity.tp_size * identity.tp_size
            or reduced[4] != identity.tp_size * identity.forward_ordinal
        ):
            self._abort(
                adapter,
                reason_code="malformed_collective_result",
                detail="SUM reduction header is inconsistent",
                collective_token=reduction.collective_token,
            )
        rank_begin = _COMMIT_HEADER_LENGTH
        rank_counts = reduced[rank_begin : rank_begin + MAX_TP_SIZE]
        if rank_counts[: identity.tp_size] != (1,) * identity.tp_size or any(
            rank_counts[identity.tp_size :]
        ):
            self._abort(
                adapter,
                reason_code="malformed_collective_result",
                detail="SUM reduction does not contain each TP rank once",
                collective_token=reduction.collective_token,
            )
        if not _common_moments_match(
            reduced,
            common_index=0,
            expected_digest=self._execution_identity_digest,
            tp_size=identity.tp_size,
        ):
            self._abort(
                adapter,
                reason_code="forward_identity_mismatch",
                detail="TP ranks voted on different forwards",
                collective_token=reduction.collective_token,
            )

        vote_manifest = _digest(reduced)
        if reduced[2] != identity.tp_size:
            self.collective_decision = "rejected"
            self.state = CommitState.DENSE_FALLBACK
            return CommitOutcome(
                dense_fallback=DenseFallbackDecision(
                    generation_id=identity.generation_id,
                    forward_ordinal=identity.forward_ordinal,
                    reason_code="cache_preflight_rejected",
                    collective_token=reduction.collective_token,
                    vote_manifest_digest=vote_manifest,
                )
            )
        if vote is None or self.preflight is None:
            self._abort(
                adapter,
                reason_code="malformed_collective_result",
                detail="rejected local vote was reduced as globally ready",
                collective_token=reduction.collective_token,
            )
        if not _common_moments_match(
            reduced,
            common_index=1,
            expected_digest=self.proposal.shared_digest,
            tp_size=identity.tp_size,
        ):
            self.collective_decision = "rejected"
            self.state = CommitState.DENSE_FALLBACK
            return CommitOutcome(
                dense_fallback=DenseFallbackDecision(
                    generation_id=identity.generation_id,
                    forward_ordinal=identity.forward_ordinal,
                    reason_code="shared_proposal_mismatch",
                    collective_token=reduction.collective_token,
                    vote_manifest_digest=vote_manifest,
                )
            )

        self.collective_decision = "accepted"
        certificate = ForwardCommitCertificate(
            generation_id=identity.generation_id,
            forward_ordinal=identity.forward_ordinal,
            tp_size=identity.tp_size,
            collective_token=reduction.collective_token,
            shared_digest=self.proposal.shared_digest,
            tp_rank=self.proposal.tp_rank,
            local_proposal_digest=self.proposal.digest,
            local_preflight_digest=self.preflight.digest,
            reduced_int64=reduced,
            manifest_digest=_manifest_digest(
                generation_id=identity.generation_id,
                forward_ordinal=identity.forward_ordinal,
                tp_size=identity.tp_size,
                collective_token=reduction.collective_token,
                shared_digest=self.proposal.shared_digest,
                reduced_int64=reduced,
            ),
        )
        certificate.validate(self.proposal, self.preflight)
        self.certificate = certificate
        self.state = CommitState.COMMITTED
        return CommitOutcome(certificate=certificate)

    def install_registered_sparse_q(
        self, certificate: ForwardCommitCertificate
    ) -> SparseQInstallAuthorization:
        """Install Q omission only after cache preflight and TP commit.

        Constructing the session merely registers the Q proposal.  The caller
        must present this authorization to the model integration before it
        swaps out full ``wq_b`` or consumes packed sparse-Q views.
        """

        if self.state is not CommitState.COMMITTED or self.preflight is None:
            raise CommitProtocolError(
                "sparse-Q installation requires a committed certificate"
            )
        if isinstance(self.preflight, ForwardPreparePreflight):
            raise CommitProtocolError(
                "forward reservations require a layer execution receipt "
                "before sparse-Q installation"
            )
        if certificate is not self.certificate:
            raise CommitProtocolError("session does not own this certificate")
        certificate.validate(self.proposal, self.preflight)
        authorization = SparseQInstallAuthorization(
            generation_id=self.proposal.identity.generation_id,
            forward_ordinal=self.proposal.identity.forward_ordinal,
            tp_rank=self.proposal.tp_rank,
            proposal_digest=self.proposal.digest,
            certificate_digest=certificate.digest,
            sparse_q_layers=self.proposal.reusable_layer_ids,
        )
        self.sparse_q_installed = True
        self.state = CommitState.SPARSE_Q_INSTALLED
        return authorization

    def authorize_omissions(
        self, certificate: ForwardCommitCertificate
    ) -> OmissionAuthorization:
        if self.state is not CommitState.SPARSE_Q_INSTALLED or self.preflight is None:
            raise CommitProtocolError("omissions require a committed certificate")
        if certificate is not self.certificate:
            raise CommitProtocolError("session does not own this certificate")
        certificate.validate(self.proposal, self.preflight)
        authorization = OmissionAuthorization(
            generation_id=self.proposal.identity.generation_id,
            forward_ordinal=self.proposal.identity.forward_ordinal,
            tp_rank=self.proposal.tp_rank,
            proposal_digest=self.proposal.digest,
            certificate_digest=certificate.digest,
            allowed_slots=self.proposal.omission_slots,
        )
        self.state = CommitState.OMISSIONS_AUTHORIZED
        return authorization

    def authorize_reserved_omissions(
        self, certificate: ForwardCommitCertificate
    ) -> OmissionAuthorization:
        """Authorize only receipt-gated consumers after a prepare certificate.

        The returned forward authorization is intentionally insufficient on
        its own.  :class:`ForwardExecutionLedger` additionally requires the
        exact layer execution receipt before any slot can be consumed.
        """

        if (
            self.state is not CommitState.COMMITTED
            or not isinstance(self.preflight, ForwardPreparePreflight)
        ):
            raise CommitProtocolError(
                "reserved omissions require a forward prepare certificate"
            )
        if certificate is not self.certificate:
            raise CommitProtocolError("session does not own this certificate")
        certificate.validate(self.proposal, self.preflight)
        authorization = OmissionAuthorization(
            generation_id=self.proposal.identity.generation_id,
            forward_ordinal=self.proposal.identity.forward_ordinal,
            tp_rank=self.proposal.tp_rank,
            proposal_digest=self.proposal.digest,
            certificate_digest=certificate.digest,
            allowed_slots=self.proposal.omission_slots,
        )
        self.state = CommitState.OMISSIONS_AUTHORIZED
        return authorization

    def consume_omitted_slot(
        self,
        adapter: CollectiveAdapter,
        authorization: OmissionAuthorization,
        slot: str,
    ) -> None:
        """Validate one omission immediately before its cache/Q consumer.

        This is the fail-closed serving entry point.  A stale view, foreign
        authorization, or unknown slot after commit is never converted to a
        rank-local dense fallback.
        """

        if (
            self.state is not CommitState.OMISSIONS_AUTHORIZED
            or self.certificate is None
        ):
            raise CommitProtocolError("omitted slots are not authorized")
        if isinstance(self.preflight, ForwardPreparePreflight):
            raise CommitProtocolError(
                "forward reservations must be consumed through a layer receipt"
            )
        try:
            authorization.validate_slot(slot, self.proposal, self.certificate)
        except Exception as exc:
            self._abort(
                adapter,
                reason_code="postcommit_omission_validation_failed",
                detail=f"{type(exc).__name__}: {exc}",
                certificate=self.certificate,
                collective_token=self.certificate.collective_token,
            )

    def fail_closed_after_commit(
        self,
        adapter: CollectiveAdapter,
        *,
        reason_code: str,
        detail: str,
    ) -> None:
        if self.state not in (
            CommitState.COMMITTED,
            CommitState.SPARSE_Q_INSTALLED,
            CommitState.OMISSIONS_AUTHORIZED,
        ) or self.certificate is None:
            raise CommitProtocolError(
                "post-commit fail-closed is valid only after certification"
            )
        self._abort(
            adapter,
            reason_code=_nonempty(reason_code, "reason_code"),
            detail=_nonempty(detail, "detail"),
            certificate=self.certificate,
            collective_token=self.certificate.collective_token,
        )


@dataclass(frozen=True)
class LayerExecutionReceipt:
    """Post-write proof for one layer under a forward prepare certificate."""

    generation_id: str
    forward_ordinal: int
    tp_rank: int
    layer_id: int
    proposal_digest: str
    certificate_digest: str
    reservation: LayerReservationBinding
    execution_token: str

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        if _strict_int(self.tp_rank, "tp_rank") < 0:
            raise ValueError("tp_rank must be non-negative")
        if _strict_int(self.layer_id, "layer_id") < 0:
            raise ValueError("layer_id must be non-negative")
        _sha_digest(self.proposal_digest, "proposal_digest")
        _sha_digest(self.certificate_digest, "certificate_digest")
        if not isinstance(self.reservation, LayerReservationBinding):
            raise TypeError("execution receipt has an invalid reservation")
        if self.reservation.layer_id != self.layer_id:
            raise ValueError("execution receipt layer/reservation mismatch")
        _nonempty(self.execution_token, "execution_token")

    def as_payload(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "forward_ordinal": self.forward_ordinal,
            "tp_rank": self.tp_rank,
            "layer_id": self.layer_id,
            "proposal_digest": self.proposal_digest,
            "certificate_digest": self.certificate_digest,
            "reservation": self.reservation,
            "execution_token": self.execution_token,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())

    def validate(
        self,
        proposal: CompositeForwardProposal,
        certificate: ForwardCommitCertificate,
        preflight: ForwardPreparePreflight,
    ) -> None:
        certificate.validate(proposal, preflight)
        expected_reservation = build_layer_reservation_binding(
            proposal, self.layer_id
        )
        expected = (
            proposal.identity.generation_id,
            proposal.identity.forward_ordinal,
            proposal.tp_rank,
            proposal.digest,
            certificate.digest,
            expected_reservation,
        )
        actual = (
            self.generation_id,
            self.forward_ordinal,
            self.tp_rank,
            self.proposal_digest,
            self.certificate_digest,
            self.reservation,
        )
        if actual != expected:
            raise CommitProtocolError("layer execution receipt is stale or foreign")


def build_layer_execution_receipt(
    proposal: CompositeForwardProposal,
    certificate: ForwardCommitCertificate,
    preflight: ForwardPreparePreflight,
    *,
    layer_id: int,
    observed_reservation: LayerReservationBinding,
    execution_token: str,
) -> LayerExecutionReceipt:
    """Issue a receipt only for the exact reserved post-write GPU state."""

    if not isinstance(preflight, ForwardPreparePreflight):
        raise TypeError("layer execution requires a forward prepare preflight")
    certificate.validate(proposal, preflight)
    expected = build_layer_reservation_binding(proposal, layer_id)
    if not isinstance(observed_reservation, LayerReservationBinding):
        raise TypeError("observed reservation has an invalid type")
    if observed_reservation != expected:
        raise CommitProtocolError(
            "layer GPU view/version differs from its forward reservation"
        )
    receipt = LayerExecutionReceipt(
        generation_id=proposal.identity.generation_id,
        forward_ordinal=proposal.identity.forward_ordinal,
        tp_rank=proposal.tp_rank,
        layer_id=int(layer_id),
        proposal_digest=proposal.digest,
        certificate_digest=certificate.digest,
        reservation=observed_reservation,
        execution_token=execution_token,
    )
    receipt.validate(proposal, certificate, preflight)
    return receipt


def _omission_slot_layer_id(slot: str) -> int:
    text = _nonempty(slot, "slot")
    parts = text.split(":")
    if len(parts) < 2 or parts[0] not in ("sparse_q", "cache", "z_off"):
        raise CommitProtocolError(f"slot {slot!r} has an invalid namespace")
    try:
        layer_id = int(parts[1], 10)
    except ValueError as exc:
        raise CommitProtocolError(f"slot {slot!r} has an invalid layer id") from exc
    if layer_id < 0 or f"{layer_id:06d}" != parts[1]:
        raise CommitProtocolError(f"slot {slot!r} is not canonical")
    return layer_id


_FINAL_HEADER_LENGTH = 5
_FINAL_COMMON_MOMENTS_LENGTH = 2 * 2 * _DIGEST_LIMB_COUNT
_FINAL_LOCAL_MANIFEST_LENGTH = _DIGEST_LIMB_COUNT
FORWARD_FINAL_INT64_VECTOR_LENGTH = (
    _FINAL_HEADER_LENGTH
    + MAX_TP_SIZE
    + _FINAL_COMMON_MOMENTS_LENGTH
    + _FINAL_LOCAL_MANIFEST_LENGTH
)


def _encode_forward_final_vector(
    *,
    ready: bool,
    proposal: CompositeForwardProposal,
    certificate: ForwardCommitCertificate,
    receipt_manifest_digest: str,
) -> Tuple[int, ...]:
    identity = proposal.identity
    vector = [
        FORWARD_FINAL_FORMAT_VERSION,
        1,
        int(bool(ready)),
        identity.tp_size,
        identity.forward_ordinal,
    ]
    vector.extend(
        1 if index == proposal.tp_rank else 0 for index in range(MAX_TP_SIZE)
    )
    for common_digest in (
        _digest(identity.as_payload()),
        certificate.digest,
    ):
        for limb in _digest_limbs(common_digest):
            vector.extend((limb, limb * limb))
    vector.extend(_digest_limbs(receipt_manifest_digest))
    if len(vector) != FORWARD_FINAL_INT64_VECTOR_LENGTH:
        raise AssertionError("forward-final vector layout changed unexpectedly")
    return tuple(vector)


def sum_forward_final_int64_vectors(
    vectors: Sequence[Sequence[int]],
) -> Tuple[int, ...]:
    """Reference SUM reduction for the fixed forward-final rendezvous."""

    items = tuple(tuple(vector) for vector in vectors)
    if not items:
        raise ValueError("at least one forward-final vector is required")
    if any(len(item) != FORWARD_FINAL_INT64_VECTOR_LENGTH for item in items):
        raise ValueError("forward-final vector has the wrong fixed length")
    reduced = tuple(
        sum(item[index] for item in items)
        for index in range(FORWARD_FINAL_INT64_VECTOR_LENGTH)
    )
    if any(value < 0 or value > 0x7FFFFFFFFFFFFFFF for value in reduced):
        raise OverflowError("forward-final SUM does not fit signed int64")
    return reduced


@dataclass(frozen=True)
class ForwardFinalReduction:
    collective_token: str
    reduced_int64: Tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty(self.collective_token, "collective_token")
        reduced = _immutable_tuple(self.reduced_int64, "reduced_int64")
        if len(reduced) != FORWARD_FINAL_INT64_VECTOR_LENGTH:
            raise ValueError("forward-final reduction has the wrong vector length")
        if any(
            _strict_int(value, "forward-final int64 entry") < 0
            or value > 0x7FFFFFFFFFFFFFFF
            for value in reduced
        ):
            raise ValueError("forward-final reduction is outside signed int64")


class ForwardFinalAdapter(Protocol):
    """Serving bridge for the one mandatory post-pipeline rendezvous."""

    def exchange_final_once(
        self, int64_vector: Tuple[int, ...]
    ) -> ForwardFinalReduction:
        ...

    def coordinated_abort(self, signal: CoordinatedAbortSignal) -> None:
        ...


def _final_common_moments_match(
    reduced: Tuple[int, ...],
    *,
    common_index: int,
    expected_digest: str,
    tp_size: int,
) -> bool:
    begin = _FINAL_HEADER_LENGTH + MAX_TP_SIZE
    begin += common_index * 2 * _DIGEST_LIMB_COUNT
    for index, limb in enumerate(_digest_limbs(expected_digest)):
        if reduced[begin + 2 * index] != tp_size * limb:
            return False
        if reduced[begin + 2 * index + 1] != tp_size * limb * limb:
            return False
    return True


@dataclass(frozen=True)
class ForwardExecutionCertificate:
    generation_id: str
    forward_ordinal: int
    tp_rank: int
    prepare_certificate_digest: str
    receipt_manifest_digest: str
    collective_token: str
    reduced_int64: Tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty(self.generation_id, "generation_id")
        if _strict_int(self.forward_ordinal, "forward_ordinal") < 0:
            raise ValueError("forward_ordinal must be non-negative")
        if _strict_int(self.tp_rank, "tp_rank") < 0:
            raise ValueError("tp_rank must be non-negative")
        _sha_digest(self.prepare_certificate_digest, "prepare_certificate_digest")
        _sha_digest(self.receipt_manifest_digest, "receipt_manifest_digest")
        _nonempty(self.collective_token, "collective_token")
        reduced = _immutable_tuple(self.reduced_int64, "reduced_int64")
        if len(reduced) != FORWARD_FINAL_INT64_VECTOR_LENGTH:
            raise ValueError("execution certificate has the wrong reduction")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "generation_id": self.generation_id,
                "forward_ordinal": self.forward_ordinal,
                "tp_rank": self.tp_rank,
                "prepare_certificate_digest": self.prepare_certificate_digest,
                "receipt_manifest_digest": self.receipt_manifest_digest,
                "collective_token": self.collective_token,
                "reduced_int64": self.reduced_int64,
            }
        )


class ForwardExecutionLedger:
    """Receipt gate and sticky fail carrier for one prepared forward.

    No post-commit error raises locally from ``record_layer_execution`` or
    ``consume_omitted_slot``.  The integration must keep executing shape-only
    carriers and call :meth:`finalize` exactly once.  That method always enters
    the fixed TP rendezvous, then either returns a success certificate or emits
    a coordinated fail-stop on every rank.
    """

    def __init__(
        self,
        session: ForwardCommitSession,
        authorization: OmissionAuthorization,
    ) -> None:
        if not isinstance(session, ForwardCommitSession):
            raise TypeError("execution ledger requires a forward commit session")
        if (
            session.state is not CommitState.OMISSIONS_AUTHORIZED
            or session.certificate is None
            or not isinstance(session.preflight, ForwardPreparePreflight)
        ):
            raise CommitProtocolError(
                "execution ledger requires authorized forward reservations"
            )
        if not isinstance(authorization, OmissionAuthorization):
            raise TypeError("execution ledger authorization has an invalid type")
        authorization.validate_slot(
            session.proposal.omission_slots[0],
            session.proposal,
            session.certificate,
        )
        self.session = session
        self.authorization = authorization
        # ForwardPreparePreflight already owns the exact immutable post-write
        # reservation for every reusable layer.  Seal those objects once so
        # the hot layer path uses O(1) identity checks instead of rebuilding
        # and hashing the complete 37-layer proposal repeatedly.
        proposal = session.proposal
        certificate = session.certificate
        preflight = session.preflight
        assert certificate is not None
        assert isinstance(preflight, ForwardPreparePreflight)
        reservations = tuple(preflight.layer_reservations)
        if tuple(item.layer_id for item in reservations) != (
            proposal.reusable_layer_ids
        ):
            raise CommitProtocolError(
                "forward ledger reservations do not cover the prepared layers"
            )
        self._proposal = proposal
        self._certificate = certificate
        self._preflight = preflight
        self._authorization = authorization
        self._proposal_digest = str(authorization.proposal_digest)
        self._certificate_digest = str(authorization.certificate_digest)
        self._allowed_slots = authorization.allowed_slots
        self._expected_reservations = {
            item.layer_id: item for item in reservations
        }
        slots_by_layer: dict[int, list[str]] = {
            layer_id: [] for layer_id in proposal.reusable_layer_ids
        }
        for slot in authorization.allowed_slots:
            slots_by_layer[_omission_slot_layer_id(slot)].append(slot)
        self._slots_by_layer = {
            layer_id: tuple(sorted(slots))
            for layer_id, slots in slots_by_layer.items()
        }
        if tuple(
            slot
            for layer_id in proposal.reusable_layer_ids
            for slot in self._slots_by_layer[layer_id]
        ) != tuple(
            sorted(
                authorization.allowed_slots,
                key=lambda slot: (_omission_slot_layer_id(slot), slot),
            )
        ):
            raise CommitProtocolError("forward ledger omission slots are incomplete")
        self.receipts: dict[int, LayerExecutionReceipt] = {}
        self.consumed_slots: set[str] = set()
        self.failures: list[Tuple[int, str, str]] = []
        self.final_attempted = False
        self.final_certificate: Optional[ForwardExecutionCertificate] = None
        self._rejected_final_manifest = _digest(
            {
                "generation_id": session.proposal.identity.generation_id,
                "forward_ordinal": session.proposal.identity.forward_ordinal,
                "tp_rank": session.proposal.tp_rank,
                "status": "local-final-construction-rejected",
            }
        )
        self._rejected_final_contribution = _encode_forward_final_vector(
            ready=False,
            proposal=session.proposal,
            certificate=session.certificate,
            receipt_manifest_digest=self._rejected_final_manifest,
        )

    @property
    def failed(self) -> bool:
        return bool(self.failures)

    def _validate_sealed_state(self) -> None:
        """O(1) identity fence for already-certified immutable state."""

        if (
            self.session.proposal is not self._proposal
            or self.session.certificate is not self._certificate
            or self.session.preflight is not self._preflight
            or self.session.state is not CommitState.OMISSIONS_AUTHORIZED
            or self.authorization is not self._authorization
        ):
            raise CommitProtocolError("forward ledger sealed state was replaced")
        if (
            self.authorization.proposal_digest != self._proposal_digest
            or self.authorization.certificate_digest != self._certificate_digest
            or self.authorization.allowed_slots is not self._allowed_slots
        ):
            raise CommitProtocolError("forward ledger authorization changed")

    def layer_omission_slots(self, layer_id: int) -> Tuple[str, ...]:
        layer_id = _strict_int(layer_id, "layer_id")
        try:
            return self._slots_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError("layer is outside the prepared forward") from exc

    def expected_layer_reservation(
        self, layer_id: int
    ) -> LayerReservationBinding:
        layer_id = _strict_int(layer_id, "layer_id")
        try:
            return self._expected_reservations[layer_id]
        except KeyError as exc:
            raise ValueError("layer is outside the prepared forward") from exc

    def record_failure(self, *, layer_id: int, stage: str, detail: str) -> None:
        layer_id = _strict_int(layer_id, "layer_id")
        if layer_id not in self.session.proposal.reusable_layer_ids:
            raise ValueError("failure layer is outside the prepared forward")
        self.failures.append(
            (
                layer_id,
                _nonempty(stage, "stage"),
                _nonempty(detail, "detail"),
            )
        )

    def record_layer_execution(
        self,
        *,
        layer_id: int,
        observed_reservation: LayerReservationBinding,
        execution_token: str,
    ) -> Optional[LayerExecutionReceipt]:
        """Return a receipt, or retain a sticky error for the final vote."""

        layer_id = int(layer_id)
        try:
            self._validate_sealed_state()
            if self.final_attempted:
                raise CommitProtocolError("layer execution arrived after final vote")
            if self.failed:
                raise CommitProtocolError("an earlier layer already failed")
            if layer_id in self.receipts:
                raise CommitProtocolError("layer execution receipt was issued twice")
            expected = self._expected_reservations.get(layer_id)
            if expected is None:
                raise CommitProtocolError(
                    "layer execution is outside the prepared forward"
                )
            if not isinstance(observed_reservation, LayerReservationBinding):
                raise TypeError("observed reservation has an invalid type")
            if observed_reservation != expected:
                raise CommitProtocolError(
                    "layer GPU view/version differs from its forward reservation"
                )
            receipt = LayerExecutionReceipt(
                generation_id=self._proposal.identity.generation_id,
                forward_ordinal=self._proposal.identity.forward_ordinal,
                tp_rank=self._proposal.tp_rank,
                layer_id=layer_id,
                proposal_digest=self._proposal_digest,
                certificate_digest=self._certificate_digest,
                reservation=expected,
                execution_token=execution_token,
            )
            self.receipts[layer_id] = receipt
            return receipt
        except BaseException as exc:
            if layer_id in self.session.proposal.reusable_layer_ids:
                self.record_failure(
                    layer_id=layer_id,
                    stage="layer_execution_receipt",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return None
            raise

    def consume_omitted_slot(
        self,
        *,
        layer_id: int,
        receipt: LayerExecutionReceipt,
        slot: str,
    ) -> bool:
        """Receipt-gated local check; failures are carried to ``finalize``."""

        layer_id = int(layer_id)
        try:
            self._validate_sealed_state()
            if self.final_attempted:
                raise CommitProtocolError("omission arrived after final vote")
            if self.failed:
                raise CommitProtocolError("an earlier committed stage failed")
            if receipt is not self.receipts.get(layer_id):
                raise CommitProtocolError("omission receipt is stale or foreign")
            if _omission_slot_layer_id(slot) != layer_id:
                raise CommitProtocolError("omission slot belongs to another layer")
            if slot not in self._slots_by_layer[layer_id]:
                raise CommitProtocolError(
                    f"slot {slot!r} is not authorized for omission"
                )
            if slot in self.consumed_slots:
                raise CommitProtocolError("omission slot was consumed twice")
            self.consumed_slots.add(slot)
            return True
        except BaseException as exc:
            if layer_id in self.session.proposal.reusable_layer_ids:
                self.record_failure(
                    layer_id=layer_id,
                    stage="omission_consumer",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return False
            raise

    def consume_layer_omitted_slots(
        self,
        *,
        layer_id: int,
        receipt: LayerExecutionReceipt,
    ) -> bool:
        """Atomically consume the complete prepared omission set for a layer.

        The forward-wide producer installs sparse-Q and z_off (and, for the
        full profile, cache) omissions together immediately after issuing the
        layer receipt.  Validate their shared authority once, then publish the
        complete layer set as one local state transition.
        """

        layer_id = int(layer_id)
        try:
            self._validate_sealed_state()
            if self.final_attempted:
                raise CommitProtocolError("omission arrived after final vote")
            if self.failed:
                raise CommitProtocolError("an earlier committed stage failed")
            if receipt is not self.receipts.get(layer_id):
                raise CommitProtocolError("omission receipt is stale or foreign")
            slots = self._slots_by_layer.get(layer_id)
            if not slots:
                raise CommitProtocolError("prepared layer has no omission slots")
            if any(slot in self.consumed_slots for slot in slots):
                raise CommitProtocolError("layer omission was consumed twice")
            self.consumed_slots.update(slots)
            return True
        except BaseException as exc:
            if layer_id in self.session.proposal.reusable_layer_ids:
                self.record_failure(
                    layer_id=layer_id,
                    stage="omission_consumer",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return False
            raise

    def _receipt_manifest_digest(self) -> str:
        return _digest(
            {
                "expected_layers": self.session.proposal.reusable_layer_ids,
                "receipts": tuple(
                    self.receipts[layer_id]
                    for layer_id in sorted(self.receipts)
                ),
                "consumed_slots": tuple(sorted(self.consumed_slots)),
                "failures": tuple(self.failures),
            }
        )

    def finalize(
        self, adapter: ForwardFinalAdapter
    ) -> ForwardExecutionCertificate:
        """Enter the forward's one fixed final TP rendezvous exactly once."""

        if self.final_attempted:
            raise CommitProtocolError("forward-final rendezvous may run once")
        self.final_attempted = True
        proposal = self.session.proposal
        certificate = self.session.certificate
        assert certificate is not None
        manifest = self._rejected_final_manifest
        contribution = self._rejected_final_contribution
        try:
            manifest = self._receipt_manifest_digest()
            ready = bool(
                not self.failed
                and tuple(sorted(self.receipts)) == proposal.reusable_layer_ids
                and tuple(sorted(self.consumed_slots)) == proposal.omission_slots
            )
            contribution = _encode_forward_final_vector(
                ready=ready,
                proposal=proposal,
                certificate=certificate,
                receipt_manifest_digest=manifest,
            )
        except BaseException as exc:
            # The rejected vector was built in __init__, before any layer may
            # consume a slot.  Even cancellation-like failures here therefore
            # cannot strand peers at the mandatory final rendezvous.
            self.failures.append(
                (
                    proposal.reusable_layer_ids[-1],
                    "forward_final_construction",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        try:
            reduction = adapter.exchange_final_once(contribution)
        except BaseException as exc:
            self.session._abort(
                adapter,
                reason_code="forward_final_indeterminate",
                detail=f"forward-final collective raised {type(exc).__name__}",
                certificate=certificate,
                collective_token=certificate.collective_token,
            )
            raise AssertionError("unreachable")  # pragma: no cover
        if not isinstance(reduction, ForwardFinalReduction):
            self.session._abort(
                adapter,
                reason_code="malformed_forward_final",
                detail="adapter returned an invalid forward-final reduction",
                certificate=certificate,
                collective_token=str(getattr(reduction, "collective_token", "")),
            )
        reduced = reduction.reduced_int64
        identity = proposal.identity
        if (
            reduced[0] != identity.tp_size * FORWARD_FINAL_FORMAT_VERSION
            or reduced[1] != identity.tp_size
            or reduced[3] != identity.tp_size * identity.tp_size
            or reduced[4] != identity.tp_size * identity.forward_ordinal
        ):
            self.session._abort(
                adapter,
                reason_code="malformed_forward_final",
                detail="forward-final reduction header is inconsistent",
                certificate=certificate,
                collective_token=reduction.collective_token,
            )
        ranks = reduced[
            _FINAL_HEADER_LENGTH : _FINAL_HEADER_LENGTH + MAX_TP_SIZE
        ]
        if ranks[: identity.tp_size] != (1,) * identity.tp_size or any(
            ranks[identity.tp_size :]
        ):
            self.session._abort(
                adapter,
                reason_code="malformed_forward_final",
                detail="forward-final reduction does not contain each TP rank once",
                certificate=certificate,
                collective_token=reduction.collective_token,
            )
        if not _final_common_moments_match(
            reduced,
            common_index=0,
            expected_digest=_digest(identity.as_payload()),
            tp_size=identity.tp_size,
        ) or not _final_common_moments_match(
            reduced,
            common_index=1,
            expected_digest=certificate.digest,
            tp_size=identity.tp_size,
        ):
            self.session._abort(
                adapter,
                reason_code="forward_final_identity_mismatch",
                detail="TP ranks finalized different prepared forwards",
                certificate=certificate,
                collective_token=reduction.collective_token,
            )
        if reduced[2] != identity.tp_size:
            local_detail = (
                "; ".join(
                    f"layer={layer} stage={stage} {detail}"
                    for layer, stage, detail in self.failures[:3]
                )
                or "another TP rank reported a missing/failed layer receipt"
            )
            self.session._abort(
                adapter,
                reason_code="forward_postcommit_failed",
                detail=local_detail,
                certificate=certificate,
                collective_token=reduction.collective_token,
            )
        result = ForwardExecutionCertificate(
            generation_id=identity.generation_id,
            forward_ordinal=identity.forward_ordinal,
            tp_rank=proposal.tp_rank,
            prepare_certificate_digest=certificate.digest,
            receipt_manifest_digest=manifest,
            collective_token=reduction.collective_token,
            reduced_int64=reduced,
        )
        self.final_certificate = result
        return result


__all__ = [
    "ATTENTION_COMPRESSOR_STATE",
    "ArtifactGenerationBinding",
    "BOUNDARY_TOKENS",
    "C4",
    "C128",
    "COMPOSITE_COMMIT_FORMAT_VERSION",
    "COMMIT_SCOPE_COMPLETED",
    "COMMIT_SCOPE_FORWARD_FRAGMENT",
    "COMMIT_SCOPE_FORWARD_RESERVED",
    "COMMIT_SCOPES",
    "COMMIT_INT64_VECTOR_LENGTH",
    "COMPOSITE_PREFLIGHT_FORMAT_VERSION",
    "COMPOSITE_PROPOSAL_FORMAT_VERSION",
    "CacheBuildersPreflight",
    "CacheDomainBinding",
    "CollectiveAdapter",
    "CollectiveReduction",
    "CommitOutcome",
    "CommitProtocolError",
    "CommitState",
    "CommitVotePayload",
    "CompositeForwardProposal",
    "CoordinatedAbortRequired",
    "CoordinatedAbortSignal",
    "DenseFallbackDecision",
    "ForwardCommitCertificate",
    "ForwardCommitSession",
    "ForwardExecutionCertificate",
    "ForwardExecutionLedger",
    "ForwardFinalAdapter",
    "ForwardFinalReduction",
    "ForwardIdentity",
    "ForwardPreparePreflight",
    "FORWARD_FINAL_FORMAT_VERSION",
    "FORWARD_FINAL_INT64_VECTOR_LENGTH",
    "GpuViewBinding",
    "INDEXER",
    "INDEXER_COMPRESSOR_STATE",
    "LayerCompressionBinding",
    "LayerExecutionReceipt",
    "LayerReservationBinding",
    "MAX_TP_SIZE",
    "OmissionAuthorization",
    "OMISSION_PROFILE_FULL",
    "OMISSION_PROFILE_SHARED_ONLY",
    "OMISSION_PROFILE_ZOFF_ONLY",
    "OMISSION_PROFILES",
    "RaggedBatchGeometry",
    "RaggedRequestGeometry",
    "SequentialQArenaBinding",
    "SequentialQArenaWriteBinding",
    "SPARSE_Q_REPRESENTATION",
    "SWA",
    "SharedLatentBinding",
    "SparseQBinding",
    "SparseQInstallAuthorization",
    "ZOFF_RESIDENCY",
    "ZOffGpuViewBinding",
    "build_cache_builders_preflight",
    "build_commit_vote",
    "build_forward_prepare_preflight",
    "build_layer_execution_receipt",
    "build_layer_reservation_binding",
    "build_sequential_q_arena_binding",
    "bind_external_artifact_pin",
    "sum_commit_int64_vectors",
    "sum_forward_final_int64_vectors",
]
