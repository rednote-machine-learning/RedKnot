"""Persistent device mirror for position-independent DeepSeek-V4 latent KV.

The CPU control plane in :mod:`dsv4_shared_latent_cache` owns artifact
identity, token validation, and boundary/checkpoint geometry.  This module is
the serving-side data plane.  It deliberately has no SGLang dependency and
keeps PyTorch optional so its geometry and fail-closed contracts remain
testable on controller-only hosts.

The important storage property is that a segment is *not* represented by 37
independent CUDA allocations.  A store allocates one fixed-capacity bank for
each physical domain with layout::

    [layer_lane, segment_epoch_slot, domain_rows, record_bytes]

SWA, C4, C128, and Indexer bytes therefore stay device resident.  Replacing a
segment fills a free slot and atomically changes the active epoch; an old slot
is never overwritten while a forward pins it.  For a multi-segment forward,
source indices include the persistent segment slot.  All segments of one
layer/domain can consequently be gathered by one fused restore launch rather
than one launch/allocation per segment.

Packed latent rows are position independent.  They must not be copied into a
positioned FlashMLA cache verbatim.  :meth:`SharedLatentGPUStore.restore_clean`
accepts a model-owned fused scatter callback which consumes the canonical
bytes, destination positions, and cache slots and applies destination RoPE (or
the Indexer transform) exactly once.  The module validates every target and
callback before the first cache mutation.

Typical model integration::

    with store.atomic_pin(cpu_restore_plan, stream=stream) as pin:
        schedule = store.preflight(pin, cpu_restore_plan, forward_id=forward_id)
        prepared = store.prepare(schedule, pin, workspace)
        validated = store.preflight_targets(
            prepared,
            targets=layer_domain_targets,
            kernels=domain_restore_kernels,
            target_slots=layer_domain_slot_vectors,
            positions=positions,
            layer_id=layer_id,
        )
        # Perform the single TP readiness vote here.  Only then mutate caches.
        receipt = store.restore_clean(validated)
        dirty = schedule.dirty_for_layer(layer_id)

``dirty`` is a compact online workset for ``wkv``, the attention compressor,
and the C4 Indexer compressor.  Query rows and every boundary/checkpoint replay
row remain dirty; clean document blocks are absent from it.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os as _os
import sys
import threading
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

try:  # The control-plane API must import on hosts without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - exercised on controller-only hosts.
    _torch = None


DEVICE_MIRROR_FORMAT_VERSION = 1
PACKED_LATENT_BYTES = 584
COMBINED_ROW_SPARSE_EXECUTION_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)

DOMAIN_SWA = "swa"
DOMAIN_C4 = "c4"
DOMAIN_C128 = "c128"
DOMAIN_INDEXER = "indexer"
DOMAIN_C4_ATTENTION_STATE = "c4_attention_state"
DOMAIN_C128_ATTENTION_STATE = "c128_attention_state"
DOMAIN_INDEXER_STATE = "indexer_state"

DATA_DOMAINS = (DOMAIN_SWA, DOMAIN_C4, DOMAIN_C128, DOMAIN_INDEXER)
STATE_DOMAINS = (
    DOMAIN_C4_ATTENTION_STATE,
    DOMAIN_C128_ATTENTION_STATE,
    DOMAIN_INDEXER_STATE,
)
ALL_DOMAINS = DATA_DOMAINS + STATE_DOMAINS

# A production restore does not need one launch for every physical domain.
# SWA/C4/C128 share the exact same 584-byte canonical record and destination
# RoPE operation, while every state record is an opaque byte copy into a
# certified physical group slot.  Keeping these families explicit lets a
# model-owned pointer-table kernel restore all middle layers with three
# launches without weakening the individual artifact/domain contracts.
RESTORE_FAMILY_PACKED = "packed_latent"
RESTORE_FAMILY_INDEXER = "indexer"
RESTORE_FAMILY_STATE = "opaque_state"
RESTORE_FAMILIES = (
    RESTORE_FAMILY_PACKED,
    RESTORE_FAMILY_INDEXER,
    RESTORE_FAMILY_STATE,
)

_RESTORE_FAMILY_BY_DOMAIN = {
    DOMAIN_SWA: RESTORE_FAMILY_PACKED,
    DOMAIN_C4: RESTORE_FAMILY_PACKED,
    DOMAIN_C128: RESTORE_FAMILY_PACKED,
    DOMAIN_INDEXER: RESTORE_FAMILY_INDEXER,
    DOMAIN_C4_ATTENTION_STATE: RESTORE_FAMILY_STATE,
    DOMAIN_C128_ATTENTION_STATE: RESTORE_FAMILY_STATE,
    DOMAIN_INDEXER_STATE: RESTORE_FAMILY_STATE,
}

_DOMAIN_ORDER = {domain: index for index, domain in enumerate(ALL_DOMAINS)}
_OPAQUE_ATTENTION_STATE_SEMANTICS = "opaque_attention_compressor_restart_state_v1"
_OPAQUE_INDEXER_STATE_SEMANTICS = "opaque_indexer_compressor_restart_state_v1"


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_tuple(values: Sequence[int], name: str, *, allow_empty: bool = False) -> Tuple[int, ...]:
    result = tuple(_strict_int(value, f"{name} entry") for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _mapping_proxy(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(mapping))


def restore_family_for_domain(domain: str) -> str:
    """Return the only legal cross-layer launch family for ``domain``."""

    try:
        return _RESTORE_FAMILY_BY_DOMAIN[str(domain)]
    except KeyError as error:
        raise ValueError(f"unknown shared-latent restore domain {domain!r}") from error


def _tensor_identity(tensor: Any) -> Tuple[object, ...]:
    """Bind a tensor without reading a device value or synchronizing CUDA."""

    try:
        version: object = int(tensor._version)
    except RuntimeError:
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


def _tensor_storage_identity(tensor: Any) -> Tuple[object, ...]:
    """Identity for controller-owned banks whose contents change by design."""

    return (
        id(tensor),
        int(tensor.data_ptr()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _require_torch() -> Any:
    if _torch is None:
        raise RuntimeError(
            "PyTorch is required for the shared-latent device store; "
            "geometry/preflight helpers remain available without it"
        )
    return _torch


@dataclass(frozen=True)
class DeviceDomainLayout:
    """One homogeneous persistent bank and its layer-to-lane mapping."""

    domain: str
    layer_ids: Tuple[int, ...]
    rows_per_layer: int
    record_bytes: int
    position_semantics: str

    def __post_init__(self) -> None:
        if self.domain not in ALL_DOMAINS:
            raise ValueError(f"unknown shared-latent domain {self.domain!r}")
        layer_ids = _strict_tuple(self.layer_ids, "domain layer ids")
        if type(self.layer_ids) is not tuple:
            raise TypeError("domain layer ids must be an immutable tuple")
        if layer_ids != self.layer_ids:
            raise ValueError("domain layer ids changed during validation")
        if type(self.rows_per_layer) is not int or self.rows_per_layer <= 0:
            raise ValueError("domain rows_per_layer must be positive")
        if type(self.record_bytes) is not int or self.record_bytes <= 0:
            raise ValueError("domain record_bytes must be positive")
        _nonempty_string(self.position_semantics, "domain position semantics")

    @property
    def payload_shape(self) -> Tuple[int, int, int]:
        return len(self.layer_ids), self.rows_per_layer, self.record_bytes

    @property
    def bytes_per_segment(self) -> int:
        return len(self.layer_ids) * self.rows_per_layer * self.record_bytes

    def lane(self, layer_id: int) -> int:
        layer_id = _strict_int(layer_id, "layer_id")
        try:
            return self.layer_ids.index(layer_id)
        except ValueError as error:
            raise KeyError(
                f"layer {layer_id} is absent from device domain {self.domain}"
            ) from error


@dataclass(frozen=True)
class SharedLatentDeviceLayout:
    """Immutable device-bank geometry derived from a CPU artifact spec."""

    model_hash: str
    policy_hash: str
    segment_length: int
    checkpoint_anchors: Tuple[int, ...]
    domains: Tuple[DeviceDomainLayout, ...]
    spec_fingerprint: str
    format_version: int = DEVICE_MIRROR_FORMAT_VERSION

    def __post_init__(self) -> None:
        _nonempty_string(self.model_hash, "model_hash")
        _nonempty_string(self.policy_hash, "policy_hash")
        if type(self.segment_length) is not int or self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if type(self.checkpoint_anchors) is not tuple:
            raise TypeError("checkpoint anchors must be an immutable tuple")
        _strict_tuple(
            self.checkpoint_anchors,
            "checkpoint anchors",
            allow_empty=True,
        )
        if type(self.domains) is not tuple or not self.domains:
            raise ValueError("device layout must contain immutable domains")
        names = tuple(domain.domain for domain in self.domains)
        if len(names) != len(set(names)):
            raise ValueError("device layout contains duplicate domains")
        if tuple(sorted(names, key=_DOMAIN_ORDER.__getitem__)) != names:
            raise ValueError("device domains must use canonical order")
        _nonempty_string(self.spec_fingerprint, "spec_fingerprint")
        if self.format_version != DEVICE_MIRROR_FORMAT_VERSION:
            raise ValueError("device mirror format is incompatible")

    @property
    def domains_by_name(self) -> Mapping[str, DeviceDomainLayout]:
        return MappingProxyType({domain.domain: domain for domain in self.domains})

    @property
    def bytes_per_segment(self) -> int:
        return sum(domain.bytes_per_segment for domain in self.domains)

    def domain(self, name: str) -> DeviceDomainLayout:
        try:
            return self.domains_by_name[name]
        except KeyError as error:
            raise KeyError(f"device layout has no {name!r} domain") from error

    def as_payload(self) -> Mapping[str, object]:
        return {
            "format_version": self.format_version,
            "model_hash": self.model_hash,
            "policy_hash": self.policy_hash,
            "segment_length": self.segment_length,
            "checkpoint_anchors": list(self.checkpoint_anchors),
            "domains": [
                {
                    "domain": item.domain,
                    "layer_ids": list(item.layer_ids),
                    "rows_per_layer": item.rows_per_layer,
                    "record_bytes": item.record_bytes,
                    "position_semantics": item.position_semantics,
                }
                for item in self.domains
            ],
            "spec_fingerprint": self.spec_fingerprint,
        }


def shared_latent_device_nbytes(
    layout: SharedLatentDeviceLayout, *, segment_epoch_capacity: int = 1
) -> int:
    """Exact persistent-bank bytes, excluding small index workspaces/events."""

    if not isinstance(layout, SharedLatentDeviceLayout):
        raise TypeError("layout must be SharedLatentDeviceLayout")
    capacity = _strict_int(segment_epoch_capacity, "segment_epoch_capacity")
    if capacity <= 0:
        raise ValueError("segment_epoch_capacity must be positive")
    return layout.bytes_per_segment * capacity


def _uniform_positive_width(layers: Sequence[object], attribute: str, domain: str) -> int:
    values = tuple(_strict_int(getattr(layer, attribute), attribute) for layer in layers)
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{domain} requires positive {attribute} values")
    if len(set(values)) != 1:
        raise ValueError(
            f"{domain} cannot share one persistent bank because {attribute} varies"
        )
    return values[0]


def _spec_fingerprint_payload(spec: object) -> Mapping[str, object]:
    layers = tuple(sorted(tuple(getattr(spec, "layers")), key=lambda item: item.layer_id))
    return {
        "model_hash": getattr(spec, "model_hash"),
        "policy_hash": getattr(spec, "policy_hash"),
        "length": getattr(spec, "length"),
        "required_layer_ids": list(getattr(spec, "required_layer_ids")),
        "packed_record_bytes": getattr(spec, "packed_record_bytes"),
        "packed_position_semantics": getattr(spec, "packed_position_semantics"),
        "indexer_position_semantics": getattr(spec, "indexer_position_semantics"),
        "token_hash_semantics": getattr(spec, "token_hash_semantics"),
        "checkpoint_stride_tokens": getattr(spec, "checkpoint_stride_tokens"),
        "format_version": getattr(spec, "format_version"),
        "layers": [
            {
                "layer_id": layer.layer_id,
                "compress_ratio": layer.compress_ratio,
                "indexer_record_bytes": layer.indexer_record_bytes,
                "attention_terminal_state_bytes": layer.attention_terminal_state_bytes,
                "indexer_terminal_state_bytes": layer.indexer_terminal_state_bytes,
            }
            for layer in layers
        ],
    }


def shared_latent_spec_fingerprint(spec: object) -> str:
    """Return the exact CPU/device compatibility fingerprint."""

    encoded = json.dumps(
        _spec_fingerprint_payload(spec),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def build_shared_latent_device_layout(spec: object) -> SharedLatentDeviceLayout:
    """Build fixed SWA/C4/C128/Indexer banks from ``SharedLatentSpec``.

    The function intentionally uses the public dataclass attributes instead of
    importing the CPU module.  It is therefore safe in the final SGLang package
    and in standalone controller tests.
    """

    length = _strict_int(getattr(spec, "length"), "segment length")
    if length <= 0:
        raise ValueError("segment length must be positive")
    packed_bytes = _strict_int(
        getattr(spec, "packed_record_bytes"), "packed record bytes"
    )
    if packed_bytes != PACKED_LATENT_BYTES:
        raise ValueError("DeepSeek-V4 packed latent rows must be exactly 584 bytes")
    layers = tuple(sorted(tuple(getattr(spec, "layers")), key=lambda item: item.layer_id))
    if not layers:
        raise ValueError("shared-latent spec has no reusable layers")
    layer_ids = tuple(_strict_int(layer.layer_id, "layer id") for layer in layers)
    required = tuple(getattr(spec, "required_layer_ids"))
    if layer_ids != required:
        raise ValueError("device layout must cover every required layer in order")
    checkpoint_anchors = tuple(
        _strict_int(anchor, "checkpoint anchor")
        for anchor in tuple(getattr(spec, "checkpoint_anchors"))
    )
    _strict_tuple(checkpoint_anchors, "checkpoint anchors", allow_empty=True)
    state_rows = len(checkpoint_anchors) + 1  # internal anchors plus terminal.

    c4_layers = tuple(layer for layer in layers if layer.compress_ratio == 4)
    c128_layers = tuple(layer for layer in layers if layer.compress_ratio == 128)
    invalid = tuple(
        layer.layer_id
        for layer in layers
        if layer.compress_ratio not in (0, 4, 128)
    )
    if invalid:
        raise ValueError(f"unsupported compressor ratios at layers {invalid}")

    packed_semantics = _nonempty_string(
        getattr(spec, "packed_position_semantics"),
        "packed position semantics",
    )
    indexer_semantics = _nonempty_string(
        getattr(spec, "indexer_position_semantics"),
        "Indexer position semantics",
    )
    domains = [
        DeviceDomainLayout(
            domain=DOMAIN_SWA,
            layer_ids=layer_ids,
            rows_per_layer=length,
            record_bytes=packed_bytes,
            position_semantics=packed_semantics,
        )
    ]
    if c4_layers:
        if length % 4:
            raise ValueError("C4 device bank requires a 4-token aligned segment")
        c4_ids = tuple(layer.layer_id for layer in c4_layers)
        domains.extend(
            (
                DeviceDomainLayout(
                    DOMAIN_C4,
                    c4_ids,
                    length // 4,
                    packed_bytes,
                    packed_semantics,
                ),
                DeviceDomainLayout(
                    DOMAIN_INDEXER,
                    c4_ids,
                    length // 4,
                    _uniform_positive_width(
                        c4_layers, "indexer_record_bytes", DOMAIN_INDEXER
                    ),
                    indexer_semantics,
                ),
            )
        )
    if c128_layers:
        if length % 128:
            raise ValueError("C128 device bank requires a 128-token aligned segment")
        domains.append(
            DeviceDomainLayout(
                DOMAIN_C128,
                tuple(layer.layer_id for layer in c128_layers),
                length // 128,
                packed_bytes,
                packed_semantics,
            )
        )
    if c4_layers:
        c4_ids = tuple(layer.layer_id for layer in c4_layers)
        domains.extend(
            (
                DeviceDomainLayout(
                    DOMAIN_C4_ATTENTION_STATE,
                    c4_ids,
                    state_rows,
                    _uniform_positive_width(
                        c4_layers,
                        "attention_terminal_state_bytes",
                        DOMAIN_C4_ATTENTION_STATE,
                    ),
                    _OPAQUE_ATTENTION_STATE_SEMANTICS,
                ),
                DeviceDomainLayout(
                    DOMAIN_INDEXER_STATE,
                    c4_ids,
                    state_rows,
                    _uniform_positive_width(
                        c4_layers,
                        "indexer_terminal_state_bytes",
                        DOMAIN_INDEXER_STATE,
                    ),
                    _OPAQUE_INDEXER_STATE_SEMANTICS,
                ),
            )
        )
    if c128_layers:
        domains.append(
            DeviceDomainLayout(
                DOMAIN_C128_ATTENTION_STATE,
                tuple(layer.layer_id for layer in c128_layers),
                state_rows,
                _uniform_positive_width(
                    c128_layers,
                    "attention_terminal_state_bytes",
                    DOMAIN_C128_ATTENTION_STATE,
                ),
                _OPAQUE_ATTENTION_STATE_SEMANTICS,
            )
        )
    domains.sort(key=lambda item: _DOMAIN_ORDER[item.domain])
    return SharedLatentDeviceLayout(
        model_hash=_nonempty_string(getattr(spec, "model_hash"), "model_hash"),
        policy_hash=_nonempty_string(getattr(spec, "policy_hash"), "policy_hash"),
        segment_length=length,
        checkpoint_anchors=checkpoint_anchors,
        domains=tuple(domains),
        spec_fingerprint=shared_latent_spec_fingerprint(spec),
    )


def artifact_domain_payloads(
    artifact: object, layout: SharedLatentDeviceLayout
) -> Mapping[str, bytes]:
    """Serialize one immutable CPU artifact into layer-major domain payloads."""

    if shared_latent_spec_fingerprint(getattr(artifact, "spec")) != layout.spec_fingerprint:
        raise ValueError("CPU artifact is incompatible with the device layout")
    layers = getattr(artifact, "layers")
    payloads: Dict[str, bytes] = {}
    for domain in layout.domains:
        parts = []
        for layer_id in domain.layer_ids:
            layer = layers[layer_id]
            if domain.domain == DOMAIN_SWA:
                value = layer.swa_positionless_packed
            elif domain.domain in (DOMAIN_C4, DOMAIN_C128):
                value = layer.compressed_positionless_packed
            elif domain.domain == DOMAIN_INDEXER:
                value = layer.indexer_positionless_keys
            elif domain.domain in (
                DOMAIN_C4_ATTENTION_STATE,
                DOMAIN_C128_ATTENTION_STATE,
            ):
                checkpoints = layer.attention_checkpoint_states
                value = b"".join(
                    bytes(checkpoints[anchor]) for anchor in layout.checkpoint_anchors
                ) + bytes(layer.attention_terminal_state)
            elif domain.domain == DOMAIN_INDEXER_STATE:
                checkpoints = layer.indexer_checkpoint_states
                value = b"".join(
                    bytes(checkpoints[anchor]) for anchor in layout.checkpoint_anchors
                ) + bytes(layer.indexer_terminal_state)
            else:  # pragma: no cover - guarded by DeviceDomainLayout.
                raise AssertionError("unreachable shared-latent domain")
            if value is None:
                raise ValueError(
                    f"artifact layer {layer_id} is missing {domain.domain}"
                )
            raw = bytes(value)
            expected = domain.rows_per_layer * domain.record_bytes
            if len(raw) != expected:
                raise ValueError(
                    f"artifact layer {layer_id} {domain.domain} has "
                    f"{len(raw)} bytes; expected {expected}"
                )
            parts.append(raw)
        payload = b"".join(parts)
        if len(payload) != domain.bytes_per_segment:
            raise AssertionError("domain payload byte accounting changed")
        payloads[domain.domain] = payload
    return MappingProxyType(payloads)


@dataclass(frozen=True)
class IndexSpan:
    begin: int
    end: int

    def __post_init__(self) -> None:
        if type(self.begin) is not int or type(self.end) is not int:
            raise TypeError("index span bounds must be integers")
        if self.begin < 0 or self.end < self.begin:
            raise ValueError("index span is invalid")

    @property
    def length(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class LayerRestoreOp:
    """One layer/domain restore over every segment in this forward."""

    domain: str
    layer_id: int
    source_indices: IndexSpan
    output_rows: IndexSpan
    position_semantics: str

    def __post_init__(self) -> None:
        if self.domain not in ALL_DOMAINS:
            raise ValueError("restore operation has an unknown domain")
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError("restore operation layer_id must be non-negative")
        if self.source_indices.length != self.output_rows.length:
            raise ValueError("restore source/output index counts differ")
        _nonempty_string(self.position_semantics, "restore position semantics")

    @property
    def count(self) -> int:
        return self.source_indices.length


@dataclass(frozen=True)
class DirtyBlockWorkset:
    """Compact compressor completions that must be generated online."""

    placement_indices: Tuple[int, ...]
    local_blocks: Tuple[int, ...]
    completion_output_rows: Tuple[int, ...]
    absolute_blocks: Tuple[int, ...]

    def __post_init__(self) -> None:
        sizes = {
            len(self.placement_indices),
            len(self.local_blocks),
            len(self.completion_output_rows),
            len(self.absolute_blocks),
        }
        if len(sizes) != 1:
            raise ValueError("dirty block workset columns differ in length")

    @property
    def count(self) -> int:
        return len(self.completion_output_rows)


@dataclass(frozen=True)
class DirtyLayerWorkset:
    """Online rows for shared KV, compressor, and C4 Indexer builders."""

    layer_id: int
    compress_ratio: int
    input_rows: Tuple[int, ...]
    document_rows: Tuple[int, ...]
    query_rows: Tuple[int, ...]
    compressed_blocks: DirtyBlockWorkset
    indexer_blocks: DirtyBlockWorkset

    def __post_init__(self) -> None:
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError("dirty workset layer_id must be non-negative")
        if self.compress_ratio not in (0, 4, 128):
            raise ValueError("dirty workset compressor ratio is invalid")
        for name, values in (
            ("input rows", self.input_rows),
            ("document rows", self.document_rows),
            ("query rows", self.query_rows),
        ):
            _strict_tuple(values, name, allow_empty=True)
        if tuple(sorted(self.document_rows + self.query_rows)) != self.input_rows:
            raise ValueError("dirty document/query rows do not partition input rows")
        if self.compress_ratio == 4:
            if self.indexer_blocks != self.compressed_blocks:
                raise ValueError("C4 Indexer/compressor dirty blocks must match")
        elif self.indexer_blocks.count:
            raise ValueError("only C4 layers can own dirty Indexer blocks")


@dataclass(frozen=True)
class DeviceRestoreSchedule:
    """Pure preflight result; one shared device index arena per forward."""

    forward_id: str
    layout_fingerprint: str
    pin_digest: str
    positions: Tuple[int, ...]
    query_start: int
    index_arena: Tuple[int, ...]
    operations: Tuple[LayerRestoreOp, ...]
    dirty_worksets: Mapping[int, DirtyLayerWorkset]
    artifact_epochs: Mapping[str, int]
    digest: str

    def __post_init__(self) -> None:
        _nonempty_string(self.forward_id, "forward_id")
        _nonempty_string(self.layout_fingerprint, "layout_fingerprint")
        _nonempty_string(self.pin_digest, "pin_digest")
        _nonempty_string(self.digest, "schedule digest")
        if type(self.positions) is not tuple or not self.positions:
            raise ValueError("schedule positions must be a non-empty tuple")
        if type(self.index_arena) is not tuple:
            raise TypeError("schedule index arena must be an immutable tuple")
        for operation in self.operations:
            if operation.source_indices.end > len(self.index_arena):
                raise ValueError("source index span exceeds the shared arena")
            if operation.output_rows.end > len(self.index_arena):
                raise ValueError("output index span exceeds the shared arena")

    @property
    def restored_value_count(self) -> int:
        return sum(operation.count for operation in self.operations)

    def operations_for_layer(self, layer_id: int) -> Tuple[LayerRestoreOp, ...]:
        return tuple(op for op in self.operations if op.layer_id == layer_id)

    def dirty_for_layer(self, layer_id: int) -> DirtyLayerWorkset:
        try:
            return self.dirty_worksets[layer_id]
        except KeyError as error:
            raise KeyError(f"layer {layer_id} has no dirty workset") from error


def _append_span(arena: list[int], values: Sequence[int]) -> IndexSpan:
    begin = len(arena)
    arena.extend(int(value) for value in values)
    return IndexSpan(begin, len(arena))


def _schedule_digest(
    *,
    forward_id: str,
    layout_fingerprint: str,
    pin_digest: str,
    query_start: int,
    positions: Sequence[int],
    artifact_epochs: Mapping[str, int],
    arena: Sequence[int],
    operations: Sequence[LayerRestoreOp],
) -> str:
    digest = sha256()
    header = {
        "forward_id": forward_id,
        "layout_fingerprint": layout_fingerprint,
        "pin_digest": pin_digest,
        "query_start": query_start,
        "positions": [positions[0], positions[-1], len(positions)],
        "artifact_epochs": sorted(artifact_epochs.items()),
        "operations": [
            (
                op.domain,
                op.layer_id,
                op.source_indices.begin,
                op.source_indices.end,
                op.output_rows.begin,
                op.output_rows.end,
                op.position_semantics,
            )
            for op in operations
        ],
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for value in arena:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return "sha256:" + digest.hexdigest()


def _pattern_for_clean_rows(
    groups: Sequence[object],
    *,
    rows_per_segment: int,
    slot_by_segment: Mapping[str, int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    source = []
    output = []
    for group in groups:
        seg_hash = str(group.seg_hash)
        if seg_hash not in slot_by_segment:
            raise ValueError(f"clean-row segment {seg_hash!r} is not pinned")
        local_rows = tuple(int(value) for value in group.local_rows)
        output_rows = tuple(int(value) for value in group.output_rows)
        if len(local_rows) != len(output_rows):
            raise ValueError("clean-row source/output counts differ")
        if any(row < 0 or row >= rows_per_segment for row in local_rows):
            raise ValueError("clean-row source is outside the device mirror")
        slot_base = int(slot_by_segment[seg_hash]) * rows_per_segment
        source.extend(slot_base + row for row in local_rows)
        output.extend(output_rows)
    return tuple(source), tuple(output)


def _pattern_for_clean_blocks(
    groups: Sequence[object],
    *,
    blocks_per_segment: int,
    slot_by_segment: Mapping[str, int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    source = []
    output = []
    for group in groups:
        seg_hash = str(group.seg_hash)
        if seg_hash not in slot_by_segment:
            raise ValueError(f"clean-block segment {seg_hash!r} is not pinned")
        local_blocks = tuple(int(value) for value in group.local_blocks)
        output_rows = tuple(int(value) for value in group.output_completion_rows)
        if len(local_blocks) != len(output_rows):
            raise ValueError("clean-block source/output counts differ")
        if any(block < 0 or block >= blocks_per_segment for block in local_blocks):
            raise ValueError("clean block is outside the device mirror")
        slot_base = int(slot_by_segment[seg_hash]) * blocks_per_segment
        source.extend(slot_base + block for block in local_blocks)
        output.extend(output_rows)
    return tuple(source), tuple(output)


def _state_pattern(
    *,
    layer_plan: object,
    restore_plan: object,
    state_rows: int,
    slot_by_segment: Mapping[str, int],
    anchors: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    source = []
    output = []
    anchor_to_row = {anchor: index for index, anchor in enumerate(anchors)}
    for checkpoint in layer_plan.checkpoint_restores:
        try:
            state_row = anchor_to_row[int(checkpoint.local_anchor)]
        except KeyError as error:
            raise ValueError("checkpoint state targets an unknown anchor") from error
        seg_hash = str(checkpoint.seg_hash)
        slot = slot_by_segment.get(seg_hash)
        if slot is None:
            raise ValueError("checkpoint state segment is not pinned")
        source.append(int(slot) * state_rows + state_row)
        output.append(int(checkpoint.output_begin_row))

    position_to_output = {
        int(position): index for index, position in enumerate(restore_plan.positions)
    }
    terminal_row = state_rows - 1
    segment_length = int(restore_plan.spec.length)
    # CPU restore currently requires equal-length placements tiled from zero.
    for placement_index in layer_plan.terminal_state_placements:
        placement_index = int(placement_index)
        terminal_position = (placement_index + 1) * segment_length - 1
        output_row = position_to_output.get(terminal_position)
        if output_row is None:
            raise ValueError("terminal state placement is absent from this forward")
        seg_hash = None
        for clean_group in restore_plan.clean_rows:
            if int(clean_group.placement_index) == placement_index:
                seg_hash = str(clean_group.seg_hash)
                break
        if seg_hash is None:
            # A completely-online placement has no clean cache and must not
            # advertise a reusable terminal state.
            raise ValueError("terminal state placement has no pinned clean segment")
        source.append(int(slot_by_segment[seg_hash]) * state_rows + terminal_row)
        # When two independently cached documents share one scheduler
        # microforward, the first document's terminal compressor state is the
        # restart state consumed by the dirty boundary row of the next
        # document.  The SGLang target-slot adapter deliberately overrides
        # that boundary row to the predecessor terminal physical group.  Aim
        # the restore at the consumer row when it exists in this forward;
        # otherwise retain the terminal row so a later microforward can bind
        # the usual continuation receipt.
        continuation_row = position_to_output.get(terminal_position + 1)
        output.append(
            int(continuation_row)
            if continuation_row is not None
            else int(output_row)
        )
    return tuple(source), tuple(output)


def _dirty_workset_for_layer(restore_plan: object, layer_plan: object) -> DirtyLayerWorkset:
    positions = tuple(int(value) for value in restore_plan.positions)
    query_start = int(restore_plan.query_start)
    dirty_rows = tuple(int(value) for value in restore_plan.dirty_output_rows)
    document_rows = tuple(row for row in dirty_rows if positions[row] < query_start)
    query_rows = tuple(row for row in dirty_rows if positions[row] >= query_start)
    ratio = int(layer_plan.compress_ratio)
    empty_blocks = DirtyBlockWorkset((), (), (), ())
    if ratio == 0:
        return DirtyLayerWorkset(
            int(layer_plan.layer_id),
            ratio,
            dirty_rows,
            document_rows,
            query_rows,
            empty_blocks,
            empty_blocks,
        )

    clean = {
        (int(group.placement_index), int(block))
        for group in layer_plan.compressed_blocks
        for block in group.local_blocks
    }
    position_to_output = {position: row for row, position in enumerate(positions)}
    segment_length = int(restore_plan.spec.length)
    placement_count = len(tuple(restore_plan.selected_prefix_tokens))
    placement_indices = []
    local_blocks = []
    completion_rows = []
    absolute_blocks = []
    for placement_index in range(placement_count):
        global_offset = placement_index * segment_length
        for local_block in range(segment_length // ratio):
            completion_position = global_offset + (local_block + 1) * ratio - 1
            output_row = position_to_output.get(completion_position)
            if output_row is None or (placement_index, local_block) in clean:
                continue
            placement_indices.append(placement_index)
            local_blocks.append(local_block)
            completion_rows.append(output_row)
            absolute_blocks.append(completion_position // ratio)
    for output_row, absolute in enumerate(positions):
        if absolute < query_start or (absolute + 1) % ratio:
            continue
        placement_indices.append(-1)
        local_blocks.append(-1)
        completion_rows.append(output_row)
        absolute_blocks.append(absolute // ratio)
    blocks = DirtyBlockWorkset(
        tuple(placement_indices),
        tuple(local_blocks),
        tuple(completion_rows),
        tuple(absolute_blocks),
    )
    return DirtyLayerWorkset(
        int(layer_plan.layer_id),
        ratio,
        dirty_rows,
        document_rows,
        query_rows,
        blocks,
        blocks if ratio == 4 else empty_blocks,
    )


def _restore_positions_match_profile(
    execution_profile: str, positions: Sequence[int]
) -> bool:
    """Validate the CPU-certified row projection consumed by the GPU plan."""

    pos = tuple(int(value) for value in positions)
    if not pos:
        return False
    if str(execution_profile) == COMBINED_ROW_SPARSE_EXECUTION_PROFILE:
        return all(right > left for left, right in zip(pos, pos[1:]))
    return all(right == left + 1 for left, right in zip(pos, pos[1:]))


def compile_device_restore_schedule(
    *,
    layout: SharedLatentDeviceLayout,
    restore_plan: object,
    slot_by_segment: Mapping[str, int],
    pin_digest: str,
    forward_id: str,
) -> DeviceRestoreSchedule:
    """Compile all segments into shared layer/domain index patterns.

    Identical patterns are interned.  In particular every SWA layer points to
    the same two arena spans, every C4/Indexer layer shares another pair, and
    every C128 layer shares another pair.  No per-layer or per-segment CUDA
    tensor is created by this function.
    """

    forward_id = _nonempty_string(forward_id, "forward_id")
    pin_digest = _nonempty_string(pin_digest, "pin_digest")
    if shared_latent_spec_fingerprint(restore_plan.spec) != layout.spec_fingerprint:
        raise ValueError("restore plan is incompatible with the device layout")
    positions = tuple(_strict_int(value, "position") for value in restore_plan.positions)
    if not _restore_positions_match_profile(
        getattr(restore_plan, "execution_profile", ""), positions
    ):
        raise ValueError("restore schedule positions must be contiguous")
    epochs = {str(key): int(value) for key, value in restore_plan.artifact_epochs.items()}
    if set(slot_by_segment) != set(epochs):
        raise ValueError("pinned segments differ from the CPU restore plan")
    slots = tuple(int(value) for value in slot_by_segment.values())
    if any(value < 0 for value in slots) or len(slots) != len(set(slots)):
        raise ValueError("pinned device slots must be non-negative and unique")

    arena: list[int] = []
    interned: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[IndexSpan, IndexSpan]] = {}
    operations = []

    def spans(source: Tuple[int, ...], output: Tuple[int, ...]) -> Tuple[IndexSpan, IndexSpan]:
        if len(source) != len(output):
            raise ValueError("restore source/output pattern sizes differ")
        if any(row < 0 or row >= len(positions) for row in output):
            raise ValueError("restore output row is outside this forward")
        key = (source, output)
        result = interned.get(key)
        if result is None:
            result = (_append_span(arena, source), _append_span(arena, output))
            interned[key] = result
        return result

    swa_layout = layout.domain(DOMAIN_SWA)
    swa_pattern = _pattern_for_clean_rows(
        restore_plan.clean_rows,
        rows_per_segment=swa_layout.rows_per_layer,
        slot_by_segment=slot_by_segment,
    )
    swa_spans = spans(*swa_pattern)
    for layer_id in swa_layout.layer_ids:
        operations.append(
            LayerRestoreOp(
                DOMAIN_SWA,
                layer_id,
                swa_spans[0],
                swa_spans[1],
                swa_layout.position_semantics,
            )
        )

    state_rows = len(layout.checkpoint_anchors) + 1
    for layer_id, layer_plan in sorted(restore_plan.layers.items()):
        ratio = int(layer_plan.compress_ratio)
        if ratio == 4:
            domain_name = DOMAIN_C4
        elif ratio == 128:
            domain_name = DOMAIN_C128
        else:
            domain_name = ""
        if domain_name:
            domain_layout = layout.domain(domain_name)
            pattern = _pattern_for_clean_blocks(
                layer_plan.compressed_blocks,
                blocks_per_segment=domain_layout.rows_per_layer,
                slot_by_segment=slot_by_segment,
            )
            block_spans = spans(*pattern)
            operations.append(
                LayerRestoreOp(
                    domain_name,
                    int(layer_id),
                    block_spans[0],
                    block_spans[1],
                    domain_layout.position_semantics,
                )
            )
            if ratio == 4:
                indexer_layout = layout.domain(DOMAIN_INDEXER)
                indexer_pattern = _pattern_for_clean_blocks(
                    layer_plan.indexer_blocks,
                    blocks_per_segment=indexer_layout.rows_per_layer,
                    slot_by_segment=slot_by_segment,
                )
                indexer_spans = spans(*indexer_pattern)
                operations.append(
                    LayerRestoreOp(
                        DOMAIN_INDEXER,
                        int(layer_id),
                        indexer_spans[0],
                        indexer_spans[1],
                        indexer_layout.position_semantics,
                    )
                )

            state_domain = (
                DOMAIN_C4_ATTENTION_STATE
                if ratio == 4
                else DOMAIN_C128_ATTENTION_STATE
            )
            state_layout = layout.domain(state_domain)
            state_pattern = _state_pattern(
                layer_plan=layer_plan,
                restore_plan=restore_plan,
                state_rows=state_rows,
                slot_by_segment=slot_by_segment,
                anchors=layout.checkpoint_anchors,
            )
            state_spans = spans(*state_pattern)
            operations.append(
                LayerRestoreOp(
                    state_domain,
                    int(layer_id),
                    state_spans[0],
                    state_spans[1],
                    state_layout.position_semantics,
                )
            )
            if ratio == 4:
                indexer_state_layout = layout.domain(DOMAIN_INDEXER_STATE)
                operations.append(
                    LayerRestoreOp(
                        DOMAIN_INDEXER_STATE,
                        int(layer_id),
                        state_spans[0],
                        state_spans[1],
                        indexer_state_layout.position_semantics,
                    )
                )

    operations.sort(key=lambda op: (_DOMAIN_ORDER[op.domain], op.layer_id))
    dirty = {
        int(layer_id): _dirty_workset_for_layer(restore_plan, layer_plan)
        for layer_id, layer_plan in sorted(restore_plan.layers.items())
    }
    arena_tuple = tuple(arena)
    digest = _schedule_digest(
        forward_id=forward_id,
        layout_fingerprint=layout.spec_fingerprint,
        pin_digest=pin_digest,
        query_start=int(restore_plan.query_start),
        positions=positions,
        artifact_epochs=epochs,
        arena=arena_tuple,
        operations=operations,
    )
    return DeviceRestoreSchedule(
        forward_id=forward_id,
        layout_fingerprint=layout.spec_fingerprint,
        pin_digest=pin_digest,
        positions=positions,
        query_start=int(restore_plan.query_start),
        index_arena=arena_tuple,
        operations=tuple(operations),
        dirty_worksets=MappingProxyType(dirty),
        artifact_epochs=MappingProxyType(epochs),
        digest=digest,
    )


@dataclass(frozen=True)
class DeviceEpochMirror:
    """Opaque handle for one sealed slot in all persistent domain banks."""

    store_token: object = field(repr=False, compare=False)
    seg_hash: str
    commit_epoch: int
    slot: int
    layout_fingerprint: str
    device: str
    seal_nonce: int


@dataclass(eq=False)
class _StagedEpoch:
    store_token: object
    seg_hash: str
    generation_id: str
    commit_epoch: int
    slot: int
    device_nbytes: int
    ready_events: list[object]
    host_references: list[object]
    captured_components: set[Tuple[str, int]]
    inflight_components: set[Tuple[str, int]]
    state: str = "staged"


@dataclass(frozen=True)
class PreparedDevicePublish:
    """Upload-complete marker which has not changed the active epoch."""

    store_token: object = field(repr=False, compare=False)
    stage: _StagedEpoch = field(repr=False, compare=False)
    seg_hash: str
    generation_id: str
    commit_epoch: int
    slot: int


@dataclass(frozen=True)
class DevicePublishReceipt:
    """Rollback handle retained until the cross-controller/TP commit vote."""

    store_token: object = field(repr=False, compare=False)
    seg_hash: str
    generation_id: str
    commit_epoch: int
    mirror: DeviceEpochMirror
    previous_mirror: Optional[DeviceEpochMirror]
    stage: _StagedEpoch = field(repr=False, compare=False)


class DeviceEpochPin:
    """Atomic one-forward lease over exact active segment epochs."""

    def __init__(
        self,
        *,
        store: "SharedLatentGPUStore",
        mirrors: Mapping[str, DeviceEpochMirror],
        digest: str,
    ) -> None:
        self._store = store
        self._mirrors = MappingProxyType(dict(mirrors))
        self.digest = digest
        self._closed = False

    @property
    def mirrors(self) -> Mapping[str, DeviceEpochMirror]:
        return self._mirrors

    @property
    def slot_by_segment(self) -> Mapping[str, int]:
        return MappingProxyType(
            {seg_hash: mirror.slot for seg_hash, mirror in self._mirrors.items()}
        )

    @property
    def epoch_by_segment(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                seg_hash: mirror.commit_epoch
                for seg_hash, mirror in self._mirrors.items()
            }
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def validate_open(self) -> None:
        if self._closed:
            raise ValueError("device epoch pin is closed")
        self._store._validate_pin(self)

    def close(self) -> None:
        if self._closed:
            return
        self._store._release_pin(self)
        self._closed = True

    def __enter__(self) -> "DeviceEpochPin":
        self.validate_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class DeviceRestoreWorkspace:
    """Reusable one-forward index/scratch allocation.

    A scheduler should allocate one workspace per concurrent model forward and
    reuse it across continuous batches.  Loading a schedule performs one host
    staging copy and one device copy, never one allocation per layer/segment.
    """

    def __init__(
        self,
        *,
        max_index_values: int,
        max_restore_rows: int,
        max_record_bytes: int,
        device: object,
        pin_host_memory: bool = True,
        allocate_restore_scratch: bool = True,
    ) -> None:
        torch = _require_torch()
        self.max_index_values = _strict_int(max_index_values, "max_index_values")
        self.max_restore_rows = _strict_int(max_restore_rows, "max_restore_rows")
        self.max_record_bytes = _strict_int(max_record_bytes, "max_record_bytes")
        if min(self.max_index_values, self.max_restore_rows, self.max_record_bytes) <= 0:
            raise ValueError("workspace capacities must be positive")
        if type(allocate_restore_scratch) is not bool:
            raise TypeError("allocate_restore_scratch must be boolean")
        self.restore_scratch_allocated = allocate_restore_scratch
        self.device = torch.device(device)
        pin = bool(pin_host_memory and self.device.type == "cuda")
        self.host_indices = torch.empty(
            self.max_index_values, dtype=torch.long, device="cpu", pin_memory=pin
        )
        self.device_indices = torch.empty(
            self.max_index_values, dtype=torch.long, device=self.device
        )
        scratch_rows = self.max_restore_rows if allocate_restore_scratch else 0
        self.scratch = torch.empty(
            (scratch_rows, self.max_record_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        self.slot_scratch = torch.empty(
            scratch_rows, dtype=torch.long, device=self.device
        )
        self._loaded_digest = ""
        self._loaded_count = 0

    @property
    def loaded_digest(self) -> str:
        return self._loaded_digest

    @property
    def device_nbytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in (
                self.device_indices,
                self.scratch,
                self.slot_scratch,
            )
        )

    def load(self, schedule: DeviceRestoreSchedule, *, non_blocking: bool = True) -> None:
        torch = _require_torch()
        count = len(schedule.index_arena)
        if count > self.max_index_values:
            raise MemoryError(
                f"restore schedule needs {count} indices; workspace has "
                f"{self.max_index_values}"
            )
        if schedule.restored_value_count and max(
            operation.count for operation in schedule.operations
        ) > self.max_restore_rows:
            raise MemoryError("restore operation exceeds workspace row capacity")
        if count:
            raw = array("q", schedule.index_arena)
            if sys.byteorder != "little":  # pragma: no cover - uncommon host.
                raw.byteswap()
            cpu_view = torch.frombuffer(raw, dtype=torch.long, count=count)
            self.host_indices[:count].copy_(cpu_view)
            self.device_indices[:count].copy_(
                self.host_indices[:count], non_blocking=bool(non_blocking)
            )
        self._loaded_count = count
        self._loaded_digest = schedule.digest

    def indices(self, span: IndexSpan, schedule: DeviceRestoreSchedule) -> Any:
        if self._loaded_digest != schedule.digest:
            raise ValueError("workspace does not hold this restore schedule")
        if span.end > self._loaded_count:
            raise ValueError("restore span exceeds the loaded workspace")
        return self.device_indices[span.begin : span.end]


@dataclass(frozen=True)
class PreparedDeviceRestore:
    store_token: object = field(repr=False, compare=False)
    pin: DeviceEpochPin = field(repr=False, compare=False)
    schedule: DeviceRestoreSchedule
    workspace: DeviceRestoreWorkspace = field(repr=False, compare=False)


@dataclass(frozen=True)
class ValidatedDeviceRestore:
    prepared: PreparedDeviceRestore
    operations: Tuple[LayerRestoreOp, ...]
    targets: Mapping[Tuple[str, int], object]
    kernels: Mapping[str, Callable[..., object]]
    target_slots: Mapping[Tuple[str, int], object] = field(
        repr=False, compare=False
    )
    positions: object = field(repr=False, compare=False)
    target_identities: Mapping[Tuple[str, int], Tuple[object, ...]] = field(
        repr=False, compare=False
    )
    slot_identities: Mapping[Tuple[str, int], Tuple[object, ...]] = field(
        repr=False, compare=False
    )
    positions_identity: Tuple[object, ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class DeviceRestoreReceipt:
    forward_id: str
    schedule_digest: str
    operation_count: int
    restored_value_count: int
    restored_by_domain: Mapping[str, int]


@dataclass(frozen=True)
class DeviceRestoreBatchInput:
    """One already-preflighted request/layer contribution to a batch.

    ``operation_metadata`` is owned by the model adapter.  Its values may bind
    page size, target row stride, RoPE frequencies, or state-group geometry,
    but may not replace the generic source/target tensor identities certified
    by :meth:`SharedLatentGPUStore.preflight_targets`.
    """

    store: object = field(repr=False, compare=False)
    validated: ValidatedDeviceRestore = field(repr=False, compare=False)
    operation_metadata: Mapping[Tuple[str, int], object] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    request_index: int = -1
    layer_id: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.validated, ValidatedDeviceRestore):
            raise TypeError("restore batch input has no validated restore")
        if not isinstance(self.operation_metadata, Mapping):
            raise TypeError("restore batch metadata must be a mapping")
        if type(self.request_index) is not int or self.request_index < -1:
            raise ValueError("restore batch request index is invalid")
        if type(self.layer_id) is not int or self.layer_id < -1:
            raise ValueError("restore batch layer binding is invalid")
        selected_layers = {
            int(operation.layer_id) for operation in self.validated.operations
        }
        if self.layer_id == -1 and len(selected_layers) == 1:
            object.__setattr__(self, "layer_id", next(iter(selected_layers)))
        elif self.layer_id != -1 and selected_layers != {self.layer_id}:
            raise ValueError("restore batch input mixes another layer binding")
        allowed = {
            (operation.domain, operation.layer_id)
            for operation in self.validated.operations
        }
        required = {
            (operation.domain, operation.layer_id)
            for operation in self.validated.operations
            if operation.count
        }
        provided = set(self.operation_metadata)
        if not required.issubset(provided) or not provided.issubset(allowed):
            raise ValueError(
                "restore batch metadata differs from selected operations "
                f"(missing={sorted(required-provided)}, "
                f"extra={sorted(provided-allowed)})"
            )
        object.__setattr__(
            self,
            "operation_metadata",
            MappingProxyType(dict(self.operation_metadata)),
        )


def _batch_metadata_identity(metadata: object) -> Tuple[object, ...]:
    identity = getattr(metadata, "batch_identity", None)
    if callable(identity):
        values = identity()
    elif identity is not None:
        values = identity
    else:
        values = (type(metadata).__module__, type(metadata).__qualname__, repr(metadata))
    if not isinstance(values, tuple):
        values = tuple(values)
    return tuple(values)


def _batch_metadata_descriptor_values(metadata: object) -> Tuple[int, ...]:
    encode = getattr(metadata, "batch_descriptor_values", None)
    if not callable(encode):
        raise TypeError(
            "restore batch metadata must define batch_descriptor_values()"
        )
    values = tuple(encode())
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("restore batch descriptor values must be non-negative ints")
    return values


@dataclass(frozen=True)
class DeviceRestoreBatchJob:
    """One physical layer/domain scatter encoded for a pointer-table kernel."""

    input_ordinal: int
    domain: str
    family: str
    layer_id: int
    count: int
    source_bank: object = field(repr=False, compare=False)
    source_indices: object = field(repr=False, compare=False)
    target_cache: object = field(repr=False, compare=False)
    target_slots: object = field(repr=False, compare=False)
    output_rows: object = field(repr=False, compare=False)
    positions: object = field(repr=False, compare=False)
    record_bytes: int
    position_semantics: str
    metadata: object = field(repr=False, compare=False)
    metadata_identity: Tuple[object, ...]

    def __post_init__(self) -> None:
        if type(self.input_ordinal) is not int or self.input_ordinal < 0:
            raise ValueError("restore batch input ordinal is invalid")
        if self.domain not in ALL_DOMAINS:
            raise ValueError("restore batch job has an unknown domain")
        if self.family != restore_family_for_domain(self.domain):
            raise ValueError("restore batch job crossed a semantic launch family")
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError("restore batch job layer is invalid")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("restore batch job must contain rows")
        if type(self.record_bytes) is not int or self.record_bytes <= 0:
            raise ValueError("restore batch record width is invalid")
        _nonempty_string(self.position_semantics, "batch position semantics")
        if self.metadata_identity != _batch_metadata_identity(self.metadata):
            raise ValueError("restore batch metadata identity changed while planning")

    @property
    def descriptor_values(self) -> Tuple[int, ...]:
        return _batch_metadata_descriptor_values(self.metadata)


@dataclass(frozen=True)
class DeviceRestoreBatchPlan:
    """Immutable cross-request, cross-layer launch schedule."""

    forward_id: str
    jobs: Tuple[DeviceRestoreBatchJob, ...]
    family_spans: Mapping[str, IndexSpan]
    input_schedule_digests: Tuple[str, ...]
    digest: str

    def __post_init__(self) -> None:
        _nonempty_string(self.forward_id, "batch forward_id")
        _nonempty_string(self.digest, "restore batch digest")
        if type(self.jobs) is not tuple or not self.jobs:
            raise ValueError("restore batch must contain immutable jobs")
        if tuple(self.family_spans) != tuple(
            family for family in RESTORE_FAMILIES if family in self.family_spans
        ):
            raise ValueError("restore batch families are not in canonical order")
        covered = []
        for family, span in self.family_spans.items():
            if span.end > len(self.jobs):
                raise ValueError("restore batch family span exceeds its jobs")
            family_jobs = self.jobs[span.begin : span.end]
            if not family_jobs or any(job.family != family for job in family_jobs):
                raise ValueError("restore batch family span contains another family")
            covered.extend(range(span.begin, span.end))
        if tuple(covered) != tuple(range(len(self.jobs))):
            raise ValueError("restore batch family spans do not partition jobs")

    @property
    def operation_count(self) -> int:
        return len(self.jobs)

    @property
    def restored_value_count(self) -> int:
        return sum(job.count for job in self.jobs)

    @property
    def launch_count(self) -> int:
        return len(self.family_spans)

    def jobs_for_family(self, family: str) -> Tuple[DeviceRestoreBatchJob, ...]:
        try:
            span = self.family_spans[family]
        except KeyError as error:
            raise KeyError(f"restore batch has no {family!r} family") from error
        return self.jobs[span.begin : span.end]


@dataclass(frozen=True)
class DeviceRestorePipelineGroup:
    """One immutable, layer-contiguous restore launch group.

    Spans use the original :class:`DeviceRestoreBatchPlan` job ordinals.  The
    pipeline therefore never repacks descriptors or weakens the aggregate
    preflight certificate; it only delays each already-certified family slice
    until its group is enqueued on the restore stream.
    """

    ordinal: int
    layer_ids: Tuple[int, ...]
    family_spans: Mapping[str, IndexSpan]
    operation_count: int
    restored_value_count: int

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("restore pipeline group ordinal is invalid")
        layers = _strict_tuple(self.layer_ids, "restore pipeline layers")
        if any(right != left + 1 for left, right in zip(layers, layers[1:])):
            raise ValueError("restore pipeline layers must be contiguous")
        if tuple(self.family_spans) != tuple(
            family for family in RESTORE_FAMILIES if family in self.family_spans
        ):
            raise ValueError("restore pipeline families are not canonical")
        if not self.family_spans:
            raise ValueError("restore pipeline group has no family spans")
        for span in self.family_spans.values():
            if not isinstance(span, IndexSpan) or span.length <= 0:
                raise ValueError("restore pipeline family span is empty")
        if type(self.operation_count) is not int or self.operation_count <= 0:
            raise ValueError("restore pipeline group has no operations")
        if (
            type(self.restored_value_count) is not int
            or self.restored_value_count <= 0
        ):
            raise ValueError("restore pipeline group has no restored values")


@dataclass(frozen=True)
class DeviceRestorePipelinePlan:
    """Exact layer-group partition of one aggregate restore plan."""

    batch_digest: str
    layers_per_group: int
    groups: Tuple[DeviceRestorePipelineGroup, ...]
    common_digest: str
    digest: str

    def __post_init__(self) -> None:
        _nonempty_string(self.batch_digest, "restore pipeline batch digest")
        _nonempty_string(self.common_digest, "restore pipeline common digest")
        _nonempty_string(self.digest, "restore pipeline digest")
        if type(self.layers_per_group) is not int or self.layers_per_group <= 0:
            raise ValueError("restore pipeline group size must be positive")
        if type(self.groups) is not tuple or not self.groups:
            raise ValueError("restore pipeline plan has no groups")
        if tuple(group.ordinal for group in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise ValueError("restore pipeline group ordinals are not canonical")
        flattened_layers = tuple(
            layer for group in self.groups for layer in group.layer_ids
        )
        if flattened_layers != tuple(sorted(set(flattened_layers))):
            raise ValueError("restore pipeline layers overlap or are out of order")

    @property
    def layer_ids(self) -> Tuple[int, ...]:
        return tuple(layer for group in self.groups for layer in group.layer_ids)

    @property
    def operation_count(self) -> int:
        return sum(group.operation_count for group in self.groups)

    @property
    def restored_value_count(self) -> int:
        return sum(group.restored_value_count for group in self.groups)


def compile_device_restore_pipeline(
    plan: DeviceRestoreBatchPlan, *, layers_per_group: int
) -> DeviceRestorePipelinePlan:
    """Partition ``plan`` without changing a job or descriptor ordinal.

    Jobs are already sorted by ``family, layer, request, domain``.  Every
    family slice for a contiguous layer group must consequently be contiguous
    in the original descriptor table.  Any violation fails closed rather than
    silently repacking a certificate-owned workspace.
    """

    if not isinstance(plan, DeviceRestoreBatchPlan):
        raise TypeError("restore pipeline requires a batch plan")
    layers_per_group = _strict_int(layers_per_group, "layers_per_group")
    if layers_per_group <= 0:
        raise ValueError("layers_per_group must be positive")
    layers = tuple(sorted({int(job.layer_id) for job in plan.jobs}))
    if not layers or any(
        right != left + 1 for left, right in zip(layers, layers[1:])
    ):
        raise ValueError("restore batch layers are not contiguous")

    groups = []
    covered_job_ordinals = []
    for ordinal, begin in enumerate(range(0, len(layers), layers_per_group)):
        group_layers = layers[begin : begin + layers_per_group]
        layer_set = set(group_layers)
        family_spans = {}
        group_ordinals = []
        operation_count = 0
        restored_value_count = 0
        for family in RESTORE_FAMILIES:
            selected = tuple(
                index
                for index, job in enumerate(plan.jobs)
                if job.family == family and int(job.layer_id) in layer_set
            )
            if not selected:
                continue
            expected = tuple(range(selected[0], selected[-1] + 1))
            if selected != expected:
                raise ValueError(
                    "restore pipeline family/layer jobs are not contiguous"
                )
            span = IndexSpan(selected[0], selected[-1] + 1)
            family_spans[family] = span
            group_ordinals.extend(selected)
            operation_count += span.length
            restored_value_count += sum(
                int(plan.jobs[index].count) for index in selected
            )
        groups.append(
            DeviceRestorePipelineGroup(
                ordinal=ordinal,
                layer_ids=tuple(group_layers),
                family_spans=MappingProxyType(family_spans),
                operation_count=operation_count,
                restored_value_count=restored_value_count,
            )
        )
        covered_job_ordinals.extend(group_ordinals)

    if tuple(sorted(covered_job_ordinals)) != tuple(range(len(plan.jobs))):
        raise ValueError("restore pipeline groups do not partition batch jobs")
    if len(covered_job_ordinals) != len(set(covered_job_ordinals)):
        raise ValueError("restore pipeline groups repeat batch jobs")
    # ``plan.digest`` intentionally binds rank-local tensor identities and GPU
    # addresses.  It must never enter a TP-common proposal: otherwise every
    # rank computes a different prepare digest and the omission vote safely
    # rejects the restore.  Bind the rank-neutral layer/family partition in a
    # separate digest, while retaining the local batch digest in ``digest``.
    common_payload = {
        "schema": "redknot-device-restore-pipeline-common-v1",
        "layers_per_group": layers_per_group,
        "groups": [
            {
                "ordinal": group.ordinal,
                "layer_ids": group.layer_ids,
                "family_spans": {
                    family: (span.begin, span.end)
                    for family, span in group.family_spans.items()
                },
                "operation_count": group.operation_count,
                "restored_value_count": group.restored_value_count,
            }
            for group in groups
        ],
    }
    common_digest = "sha256:" + sha256(
        json.dumps(
            common_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": "redknot-device-restore-pipeline-local-v1",
        "batch_digest": str(plan.digest),
        "common_digest": common_digest,
    }
    digest = "sha256:" + sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    pipeline = DeviceRestorePipelinePlan(
        batch_digest=str(plan.digest),
        layers_per_group=layers_per_group,
        groups=tuple(groups),
        common_digest=common_digest,
        digest=digest,
    )
    if (
        pipeline.operation_count != plan.operation_count
        or pipeline.restored_value_count != plan.restored_value_count
        or pipeline.layer_ids != layers
    ):
        raise ValueError("restore pipeline accounting differs from batch plan")
    return pipeline


# Common columns are intentionally sufficient for a kernel to chase every
# source/target pointer without allocating or materializing a padded tensor.
# Model-specific descriptor columns follow these in the persistent workspace.
DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS = (
    "input_ordinal",
    "domain_ordinal",
    "layer_id",
    "count",
    "source_bank_ptr",
    "source_indices_ptr",
    "target_cache_ptr",
    "target_slots_ptr",
    "output_rows_ptr",
    "positions_ptr",
    "record_bytes",
    "source_rows",
    "vector_rows",
)


class DeviceRestoreBatchWorkspace:
    """Persistent descriptor table shared by every middle layer in a forward.

    Loading a plan performs one host-to-device descriptor copy.  The pointer
    table references the existing persistent artifact banks, per-request index
    arenas, native cache tensors, slot vectors, and position vectors directly;
    it never allocates a padded ``[layers, rows, 584]`` activation.
    """

    def __init__(
        self,
        *,
        max_jobs: int,
        max_extra_descriptor_columns: int,
        max_validation_entries: Optional[int] = None,
        device: object,
        pin_host_memory: bool = True,
    ) -> None:
        torch = _require_torch()
        self.max_jobs = _strict_int(max_jobs, "max_jobs")
        self.max_extra_descriptor_columns = _strict_int(
            max_extra_descriptor_columns, "max_extra_descriptor_columns"
        )
        if self.max_jobs <= 0 or self.max_extra_descriptor_columns < 0:
            raise ValueError("restore batch workspace capacities are invalid")
        if max_validation_entries is None:
            max_validation_entries = self.max_jobs
        self.max_validation_entries = _strict_int(
            max_validation_entries, "max_validation_entries"
        )
        if self.max_validation_entries <= 0:
            raise ValueError("restore batch validation capacity must be positive")
        collision_table_size = 1
        while collision_table_size < 2 * self.max_validation_entries:
            collision_table_size *= 2
        self.collision_table_size = collision_table_size
        self.device = torch.device(device)
        self.descriptor_width = (
            len(DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS)
            + self.max_extra_descriptor_columns
        )
        pin = bool(pin_host_memory and self.device.type == "cuda")
        self.host_descriptors = torch.empty(
            (self.max_jobs, self.descriptor_width),
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin,
        )
        self.device_descriptors = torch.empty(
            (self.max_jobs, self.descriptor_width),
            dtype=torch.int64,
            device=self.device,
        )
        self.host_validation_status = torch.empty(
            (1,), dtype=torch.int32, device="cpu", pin_memory=pin
        )
        self.device_validation_status = torch.empty(
            (1,), dtype=torch.int32, device=self.device
        )
        self.device_collision_table = torch.empty(
            (self.collision_table_size,), dtype=torch.int32, device=self.device
        )
        self._loaded_digest = ""
        self._loaded_count = 0
        self._loaded_validation_entries = 0
        self._loaded_collision_table_size = 0

    @property
    def loaded_digest(self) -> str:
        return self._loaded_digest

    @property
    def device_nbytes(self) -> int:
        return (
            int(self.device_descriptors.numel())
            * int(self.device_descriptors.element_size())
            + int(self.device_validation_status.numel())
            * int(self.device_validation_status.element_size())
            + int(self.device_collision_table.numel())
            * int(self.device_collision_table.element_size())
        )

    @property
    def loaded_validation_entries(self) -> int:
        return self._loaded_validation_entries

    @property
    def loaded_collision_table_size(self) -> int:
        return self._loaded_collision_table_size

    def load(
        self, plan: DeviceRestoreBatchPlan, *, non_blocking: bool = True
    ) -> None:
        if len(plan.jobs) > self.max_jobs:
            raise MemoryError(
                f"restore batch needs {len(plan.jobs)} jobs; workspace has "
                f"{self.max_jobs}"
            )
        validation_entries = sum(
            int(job.count)
            * max(
                1,
                int(
                    getattr(
                        job.metadata, "state_required_groups", 0
                    )
                ),
            )
            for job in plan.jobs
        )
        if validation_entries > self.max_validation_entries:
            raise MemoryError(
                "restore batch collision proof exceeds workspace capacity"
            )
        collision_table_size = 1
        while collision_table_size < 2 * validation_entries:
            collision_table_size *= 2
        self.host_descriptors[: len(plan.jobs)].zero_()
        common_width = len(DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS)
        for row, job in enumerate(plan.jobs):
            extra = job.descriptor_values
            if len(extra) > self.max_extra_descriptor_columns:
                raise MemoryError(
                    f"restore batch job needs {len(extra)} model columns; "
                    f"workspace has {self.max_extra_descriptor_columns}"
                )
            common = (
                job.input_ordinal,
                _DOMAIN_ORDER[job.domain],
                job.layer_id,
                job.count,
                int(job.source_bank.data_ptr()),
                int(job.source_indices.data_ptr()),
                int(job.target_cache.data_ptr()),
                int(job.target_slots.data_ptr()),
                int(job.output_rows.data_ptr()),
                int(job.positions.data_ptr()),
                job.record_bytes,
                int(job.source_bank.shape[0]),
                int(job.positions.numel()),
            )
            values = common + extra
            for column, value in enumerate(values):
                self.host_descriptors[row, column] = int(value)
            if len(values) < common_width:
                raise AssertionError("common restore descriptor width changed")
        self.device_descriptors[: len(plan.jobs)].copy_(
            self.host_descriptors[: len(plan.jobs)],
            non_blocking=bool(non_blocking),
        )
        self.host_validation_status.zero_()
        self.device_validation_status.copy_(
            self.host_validation_status, non_blocking=bool(non_blocking)
        )
        # CUDA implements this as an async memset, not an H2D copy of a
        # multi-megabyte host mirror.  It is audited separately from kernels.
        self.device_collision_table[:collision_table_size].zero_()
        self._loaded_count = len(plan.jobs)
        self._loaded_digest = plan.digest
        self._loaded_validation_entries = validation_entries
        self._loaded_collision_table_size = collision_table_size

    def descriptors_for_family(
        self, family: str, plan: DeviceRestoreBatchPlan
    ) -> object:
        if self._loaded_digest != plan.digest:
            raise ValueError("restore batch workspace does not hold this plan")
        span = plan.family_spans.get(family)
        if span is None or span.end > self._loaded_count:
            raise ValueError("restore batch descriptor family is unavailable")
        return self.device_descriptors[span.begin : span.end]


@dataclass(frozen=True)
class DeviceRestoreBatchKernel:
    """Explicit launch-count contract for one semantic family callback."""

    family: str
    callback: Callable[..., object] = field(repr=False, compare=False)
    production_certified: bool = False
    max_launches_per_call: int = 1

    def __post_init__(self) -> None:
        if self.family not in RESTORE_FAMILIES:
            raise ValueError("restore batch kernel has an unknown family")
        if not callable(self.callback):
            raise TypeError("restore batch kernel callback must be callable")
        if type(self.production_certified) is not bool:
            raise TypeError("production_certified must be boolean")
        if type(self.max_launches_per_call) is not int or self.max_launches_per_call <= 0:
            raise ValueError("restore batch kernel launch bound is invalid")
        if self.production_certified and self.max_launches_per_call != 1:
            raise ValueError("a production restore family must be one launch")


@dataclass(frozen=True)
class DeviceRestoreBatchPreflightKernel:
    """One target-mutation-free aggregate descriptor validation contract."""

    callback: Callable[..., object] = field(repr=False, compare=False)
    production_certified: bool = False
    max_launches_per_call: int = 2

    def __post_init__(self) -> None:
        if not callable(self.callback):
            raise TypeError("restore batch preflight callback must be callable")
        if type(self.production_certified) is not bool:
            raise TypeError("preflight production certification must be boolean")
        if (
            type(self.max_launches_per_call) is not int
            or self.max_launches_per_call <= 0
            or self.max_launches_per_call > 2
        ):
            raise ValueError("restore batch preflight launch bound is invalid")


@dataclass(frozen=True)
class ValidatedDeviceRestoreBatch:
    inputs: Tuple[DeviceRestoreBatchInput, ...]
    plan: DeviceRestoreBatchPlan
    workspace: DeviceRestoreBatchWorkspace = field(repr=False, compare=False)
    kernels: Mapping[str, DeviceRestoreBatchKernel] = field(
        repr=False, compare=False
    )
    production_certified: bool
    workspace_identity: Tuple[object, ...] = field(repr=False, compare=False)
    validation_status_identity: Tuple[object, ...] = field(
        repr=False, compare=False
    )
    collision_table_identity: Tuple[object, ...] = field(
        repr=False, compare=False
    )
    preflight_launch_count: int = 0


@dataclass(frozen=True)
class DeviceRestoreBatchReceipt:
    forward_id: str
    batch_digest: str
    launch_count: int
    validation_launch_count: int
    restore_launch_count: int
    descriptor_h2d_bytes: int
    validation_control_h2d_bytes: int
    validation_status_d2h_bytes: int
    validation_memset_bytes: int
    operation_count: int
    restored_value_count: int
    restored_by_domain: Mapping[str, int]
    input_receipts: Tuple[DeviceRestoreReceipt, ...]
    input_bindings: Tuple[Tuple[int, int], ...]

    def for_input(self, input_ordinal: int) -> DeviceRestoreReceipt:
        input_ordinal = _strict_int(input_ordinal, "input_ordinal")
        if input_ordinal < 0 or input_ordinal >= len(self.input_receipts):
            raise IndexError("restore batch input ordinal is out of range")
        return self.input_receipts[input_ordinal]

    def for_layer(
        self, layer_id: int
    ) -> Tuple[Tuple[int, DeviceRestoreReceipt], ...]:
        """Return ``(request_index, receipt)`` entries in input order."""

        layer_id = _strict_int(layer_id, "layer_id")
        result = tuple(
            (request_index, receipt)
            for (request_index, bound_layer), receipt in zip(
                self.input_bindings, self.input_receipts
            )
            if bound_layer == layer_id
        )
        if not result:
            raise KeyError(f"restore batch has no layer {layer_id}")
        return result


class SharedLatentGPUStore:
    """Fixed-capacity immutable-epoch banks for one model/rank/device."""

    def __init__(
        self,
        *,
        layout: SharedLatentDeviceLayout,
        max_segment_epochs: int,
        device: object,
    ) -> None:
        torch = _require_torch()
        if not isinstance(layout, SharedLatentDeviceLayout):
            raise TypeError("layout must be SharedLatentDeviceLayout")
        self.layout = layout
        self.max_segment_epochs = _strict_int(
            max_segment_epochs, "max_segment_epochs"
        )
        if self.max_segment_epochs <= 0:
            raise ValueError("max_segment_epochs must be positive")
        self.device = torch.device(device)
        self._token = object()
        self._lock = threading.RLock()
        self._banks: Dict[str, object] = {
            domain.domain: torch.empty(
                (
                    len(domain.layer_ids),
                    self.max_segment_epochs,
                    domain.rows_per_layer,
                    domain.record_bytes,
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            for domain in layout.domains
        }
        self._bank_identities = {
            name: _tensor_storage_identity(tensor)
            for name, tensor in self._banks.items()
        }
        self._free_slots = list(reversed(range(self.max_segment_epochs)))
        self._staged: Dict[int, _StagedEpoch] = {}
        self._active: Dict[str, DeviceEpochMirror] = {}
        self._slot_mirrors: Dict[int, DeviceEpochMirror] = {}
        self._pin_counts: Dict[int, int] = {}
        self._retired_slots: set[int] = set()
        self._pending_publishes: Dict[
            Tuple[str, str], DevicePublishReceipt
        ] = {}
        self._seal_nonce = 0

    @property
    def bytes_allocated(self) -> int:
        return shared_latent_device_nbytes(
            self.layout, segment_epoch_capacity=self.max_segment_epochs
        )

    @property
    def device_nbytes(self) -> int:
        """Compatibility alias used by combined z_off/latent capacity audits."""

        return self.bytes_allocated

    @property
    def device_nbytes_per_segment_epoch(self) -> int:
        return self.layout.bytes_per_segment

    @property
    def active_epochs(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(
                {
                    seg_hash: mirror.commit_epoch
                    for seg_hash, mirror in self._active.items()
                }
            )

    @property
    def free_slot_count(self) -> int:
        with self._lock:
            return len(self._free_slots)

    def _validate_banks(self) -> None:
        for name, tensor in self._banks.items():
            if _tensor_storage_identity(tensor) != self._bank_identities[name]:
                raise RuntimeError(f"persistent device bank {name!r} changed identity")

    def _payload_tensor(self, payload: object, domain: DeviceDomainLayout) -> Tuple[object, ...]:
        torch = _require_torch()
        if isinstance(payload, (bytes, bytearray, memoryview)):
            raw = bytearray(payload)
            if len(raw) != domain.bytes_per_segment:
                raise ValueError(
                    f"{domain.domain} payload has {len(raw)} bytes; expected "
                    f"{domain.bytes_per_segment}"
                )
            tensor = torch.frombuffer(raw, dtype=torch.uint8).view(domain.payload_shape)
            return tensor, raw
        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                f"{domain.domain} payload must be bytes or a torch.Tensor"
            )
        if payload.dtype != torch.uint8:
            raise ValueError(f"{domain.domain} payload must use torch.uint8")
        if tuple(int(value) for value in payload.shape) != domain.payload_shape:
            raise ValueError(
                f"{domain.domain} payload shape changed: {tuple(payload.shape)} "
                f"!= {domain.payload_shape}"
            )
        return (payload,)

    def _component_tensor(
        self, payload: object, domain: DeviceDomainLayout
    ) -> Tuple[object, ...]:
        torch = _require_torch()
        shape = (domain.rows_per_layer, domain.record_bytes)
        expected = domain.rows_per_layer * domain.record_bytes
        if isinstance(payload, (bytes, bytearray, memoryview)):
            raw = bytearray(payload)
            if len(raw) != expected:
                raise ValueError(
                    f"{domain.domain} component has {len(raw)} bytes; expected {expected}"
                )
            tensor = torch.frombuffer(raw, dtype=torch.uint8).view(shape)
            return tensor, raw
        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                f"{domain.domain} component must be bytes or a torch.Tensor"
            )
        if payload.dtype != torch.uint8 or tuple(int(v) for v in payload.shape) != shape:
            raise ValueError(
                f"{domain.domain} component must be uint8 with shape {shape}"
            )
        return (payload,)

    @property
    def _required_components(self) -> set[Tuple[str, int]]:
        return {
            (domain.domain, layer_id)
            for domain in self.layout.domains
            for layer_id in domain.layer_ids
        }

    def begin_stage(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        commit_epoch: int,
    ) -> _StagedEpoch:
        """Reserve an unused immutable epoch slot for layer-wise capture."""

        seg_hash = _nonempty_string(seg_hash, "seg_hash")
        generation_id = _nonempty_string(generation_id, "generation_id")
        commit_epoch = _strict_int(commit_epoch, "commit_epoch")
        if commit_epoch <= 0:
            raise ValueError("commit_epoch must be positive")
        with self._lock:
            self._validate_banks()
            if any(
                stage.seg_hash == seg_hash
                and stage.state in ("staged", "prepared")
                for stage in self._staged.values()
            ):
                raise ValueError("another device generation of this segment is staging")
            if any(key[0] == seg_hash for key in self._pending_publishes):
                raise ValueError("this segment has a pending device publish")
            active = self._active.get(seg_hash)
            if active is not None and commit_epoch <= active.commit_epoch:
                raise ValueError("device generation epoch must increase monotonically")
            if not self._free_slots:
                raise MemoryError("persistent shared-latent device bank is full")
            slot = self._free_slots.pop()
            stage = _StagedEpoch(
                self._token,
                seg_hash,
                generation_id,
                commit_epoch,
                slot,
                self.layout.bytes_per_segment,
                [],
                [],
                set(),
                set(),
            )
            self._staged[slot] = stage
            return stage

    def capture_component(
        self,
        stage: _StagedEpoch,
        *,
        domain: str,
        layer_id: int,
        payload: object,
        stream: Optional[object] = None,
        non_blocking: bool = True,
    ) -> None:
        """Capture one layer/domain directly from bytes or a device Tensor."""

        torch = _require_torch()
        if stage.store_token is not self._token:
            raise ValueError("staged epoch belongs to another store")
        domain_layout = self.layout.domain(str(domain))
        layer_id = _strict_int(layer_id, "layer_id")
        lane = domain_layout.lane(layer_id)
        key = (domain_layout.domain, layer_id)
        values = self._component_tensor(payload, domain_layout)
        with self._lock:
            if self._staged.get(stage.slot) is not stage or stage.state != "staged":
                raise ValueError("staged epoch is stale")
            if key in stage.captured_components or key in stage.inflight_components:
                raise ValueError(f"device component {key!r} was already captured")
            # Reserve the component before releasing the lock so two capture
            # streams cannot race to write the same immutable lane/slot.
            stage.inflight_components.add(key)
        copy_stream = None
        try:
            if self.device.type == "cuda":
                copy_stream = stream if stream is not None else torch.cuda.current_stream(self.device)
                with torch.cuda.stream(copy_stream):
                    self._banks[domain_layout.domain][lane, stage.slot].copy_(
                        values[0], non_blocking=bool(non_blocking)
                    )
                    event = torch.cuda.Event()
                    event.record(copy_stream)
            else:
                self._banks[domain_layout.domain][lane, stage.slot].copy_(values[0])
        except Exception:
            if self.device.type == "cuda" and copy_stream is not None:
                # A failed asynchronous enqueue must finish before this slot
                # can be retried or aborted.
                copy_stream.synchronize()
            with self._lock:
                stage.inflight_components.discard(key)
            raise
        with self._lock:
            if self._staged.get(stage.slot) is not stage or stage.state != "staged":
                raise RuntimeError("staged epoch changed during component capture")
            stage.host_references.extend(values)
            stage.ready_events.extend([event] if self.device.type == "cuda" else [])
            stage.inflight_components.remove(key)
            stage.captured_components.add(key)

    def stage_payloads(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        commit_epoch: int,
        payloads: Mapping[str, object],
        stream: Optional[object] = None,
        non_blocking: bool = True,
    ) -> _StagedEpoch:
        """Copy a complete generation into a free slot, without publishing it."""

        torch = _require_torch()
        expected_domains = {domain.domain for domain in self.layout.domains}
        if set(payloads) != expected_domains:
            missing = sorted(expected_domains - set(payloads))
            extra = sorted(set(payloads) - expected_domains)
            raise ValueError(
                f"device payload domains are incomplete (missing={missing}, extra={extra})"
            )
        converted: Dict[str, object] = {}
        host_references = []
        for domain in self.layout.domains:
            values = self._payload_tensor(payloads[domain.domain], domain)
            converted[domain.domain] = values[0]
            host_references.extend(values)

        stage = self.begin_stage(
            seg_hash=seg_hash,
            generation_id=generation_id,
            commit_epoch=commit_epoch,
        )
        with self._lock:
            stage.inflight_components.update(self._required_components)

        copy_stream = None
        try:
            if self.device.type == "cuda":
                copy_stream = stream if stream is not None else torch.cuda.current_stream(self.device)
                with torch.cuda.stream(copy_stream):
                    for domain in self.layout.domains:
                        self._banks[domain.domain][:, stage.slot].copy_(
                            converted[domain.domain], non_blocking=bool(non_blocking)
                        )
                    event = torch.cuda.Event()
                    event.record(copy_stream)
                stage.ready_events.append(event)
            else:
                for domain in self.layout.domains:
                    self._banks[domain.domain][:, stage.slot].copy_(
                        converted[domain.domain]
                    )
        except Exception:
            if self.device.type == "cuda" and copy_stream is not None:
                copy_stream.synchronize()
            with self._lock:
                self._staged.pop(stage.slot, None)
                self._free_slots.append(stage.slot)
                stage.inflight_components.clear()
            raise
        with self._lock:
            stage.host_references.extend(host_references)
            stage.inflight_components.clear()
            stage.captured_components.update(self._required_components)
        return stage

    def stage_artifact(
        self,
        artifact: object,
        *,
        generation_id: Optional[str] = None,
        stream: Optional[object] = None,
        non_blocking: bool = True,
    ) -> _StagedEpoch:
        return self.stage_payloads(
            seg_hash=str(artifact.seg_hash),
            generation_id=(
                str(generation_id)
                if generation_id is not None
                else f"shared-latent-epoch:{int(artifact.commit_epoch)}"
            ),
            commit_epoch=int(artifact.commit_epoch),
            payloads=artifact_domain_payloads(artifact, self.layout),
            stream=stream,
            non_blocking=non_blocking,
        )

    def stage_bundle(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        commit_epoch: int,
        payloads: Mapping[str, object],
        stream: Optional[object] = None,
        non_blocking: bool = True,
    ) -> _StagedEpoch:
        """Named two-phase API for a complete bytes-or-Tensor domain bundle."""

        return self.stage_payloads(
            seg_hash=seg_hash,
            generation_id=generation_id,
            commit_epoch=commit_epoch,
            payloads=payloads,
            stream=stream,
            non_blocking=non_blocking,
        )

    def abort_staged(self, stage: _StagedEpoch) -> None:
        if stage.store_token is not self._token:
            raise ValueError("staged epoch belongs to another store")
        with self._lock:
            if self._staged.get(stage.slot) is not stage:
                raise ValueError("staged epoch is stale")
            if stage.inflight_components:
                raise RuntimeError("cannot abort while device components are in flight")
            ready_events = tuple(stage.ready_events)
        for ready_event in ready_events:
            # The slot cannot return to the free list while its upload stream
            # may still be writing it.
            ready_event.synchronize()
        with self._lock:
            current = self._staged.get(stage.slot)
            if current is not stage or stage.state not in ("staged", "prepared"):
                raise ValueError("staged epoch is stale")
            stage.state = "aborted"
            self._staged.pop(stage.slot)
            self._free_slots.append(stage.slot)

    def _collect_retired_locked(self) -> None:
        for slot in tuple(self._retired_slots):
            if self._pin_counts.get(slot, 0):
                continue
            self._retired_slots.remove(slot)
            self._slot_mirrors.pop(slot, None)
            self._pin_counts.pop(slot, None)
            self._free_slots.append(slot)

    def prepare_publish(
        self, stage: _StagedEpoch, *, synchronize: bool = True
    ) -> PreparedDevicePublish:
        """Prove upload completeness without changing the active generation."""

        if stage.store_token is not self._token:
            raise ValueError("staged epoch belongs to another store")
        for ready_event in stage.ready_events:
            if synchronize:
                ready_event.synchronize()
            elif not bool(ready_event.query()):
                raise RuntimeError("device generation upload is not complete")
        with self._lock:
            self._validate_banks()
            if self._staged.get(stage.slot) is not stage or stage.state != "staged":
                raise ValueError("staged epoch is stale")
            if stage.inflight_components:
                raise RuntimeError("device component capture is still in flight")
            missing = self._required_components - stage.captured_components
            extra = stage.captured_components - self._required_components
            if missing or extra:
                raise ValueError(
                    "cannot publish an incomplete device artifact "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
                )
            active = self._active.get(stage.seg_hash)
            if active is not None and stage.commit_epoch <= active.commit_epoch:
                raise ValueError("device generation epoch no longer increases")
            key = (stage.seg_hash, stage.generation_id)
            if key in self._pending_publishes:
                raise ValueError("device generation already has a pending publish")
            stage.state = "prepared"
            return PreparedDevicePublish(
                self._token,
                stage,
                stage.seg_hash,
                stage.generation_id,
                stage.commit_epoch,
                stage.slot,
            )

    def publish(self, prepared: PreparedDevicePublish) -> DevicePublishReceipt:
        """Locally publish, retaining both generations for rollback/confirm."""

        if prepared.store_token is not self._token:
            raise ValueError("prepared publish belongs to another store")
        stage = prepared.stage
        key = (prepared.seg_hash, prepared.generation_id)
        with self._lock:
            self._validate_banks()
            if (
                self._staged.get(prepared.slot) is not stage
                or stage.state != "prepared"
                or stage.seg_hash != prepared.seg_hash
                or stage.generation_id != prepared.generation_id
                or stage.commit_epoch != prepared.commit_epoch
            ):
                raise ValueError("prepared device publish is stale")
            if key in self._pending_publishes:
                raise ValueError("device generation already has a pending publish")
            previous = self._active.get(stage.seg_hash)
            if previous is not None and stage.commit_epoch <= previous.commit_epoch:
                raise ValueError("device generation epoch no longer increases")
            self._seal_nonce += 1
            mirror = DeviceEpochMirror(
                self._token,
                stage.seg_hash,
                stage.commit_epoch,
                stage.slot,
                self.layout.spec_fingerprint,
                str(self.device),
                self._seal_nonce,
            )
            receipt = DevicePublishReceipt(
                self._token,
                stage.seg_hash,
                stage.generation_id,
                stage.commit_epoch,
                mirror,
                previous,
                stage,
            )
            # This is the only active-map mutation.  The old slot remains
            # capacity-accounted until confirm, so rollback never needs to
            # reconstruct or recopy a device artifact.
            self._active[stage.seg_hash] = mirror
            self._slot_mirrors[stage.slot] = mirror
            self._pin_counts.setdefault(stage.slot, 0)
            self._staged.pop(stage.slot)
            stage.state = "pending_publish"
            self._pending_publishes[key] = receipt
            return receipt

    def publish_staged(
        self, stage: _StagedEpoch, *, synchronize: bool = True
    ) -> DevicePublishReceipt:
        """Compatibility wrapper for prepare-then-local-publish."""

        return self.publish(self.prepare_publish(stage, synchronize=synchronize))

    def validate_publish_confirmation(self, receipt: DevicePublishReceipt) -> None:
        if receipt.store_token is not self._token:
            raise ValueError("device publish receipt belongs to another store")
        key = (receipt.seg_hash, receipt.generation_id)
        with self._lock:
            if self._pending_publishes.get(key) is not receipt:
                raise ValueError("device publish receipt is not pending")
            if self._active.get(receipt.seg_hash) is not receipt.mirror:
                raise ValueError("device publish receipt is no longer active")
            if self._slot_mirrors.get(receipt.mirror.slot) is not receipt.mirror:
                raise ValueError("device publish receipt lost its sealed slot")

    def rollback_publish(self, receipt: DevicePublishReceipt) -> None:
        """Undo local publication after another controller/TP rank rejects it."""

        self.validate_publish_confirmation(receipt)
        key = (receipt.seg_hash, receipt.generation_id)
        with self._lock:
            if self._pin_counts.get(receipt.mirror.slot, 0):
                raise RuntimeError("cannot rollback a device epoch pinned by a forward")
            if receipt.previous_mirror is None:
                self._active.pop(receipt.seg_hash, None)
            else:
                self._active[receipt.seg_hash] = receipt.previous_mirror
                self._retired_slots.discard(receipt.previous_mirror.slot)
            self._pending_publishes.pop(key)
            receipt.stage.state = "rolled_back"
            self._retired_slots.add(receipt.mirror.slot)
            self._collect_retired_locked()

    def confirm_publish(self, receipt: DevicePublishReceipt) -> DeviceEpochMirror:
        """Finalize publication after z_off and every TP rank also committed."""

        self.validate_publish_confirmation(receipt)
        key = (receipt.seg_hash, receipt.generation_id)
        with self._lock:
            self._pending_publishes.pop(key)
            receipt.stage.state = "confirmed"
            if receipt.previous_mirror is not None:
                self._retired_slots.add(receipt.previous_mirror.slot)
            self._collect_retired_locked()
            return receipt.mirror

    def stage_and_publish(
        self,
        artifact: object,
        *,
        generation_id: Optional[str] = None,
        stream: Optional[object] = None,
        non_blocking: bool = True,
    ) -> DeviceEpochMirror:
        """Single-rank convenience wrapper; distributed code must not use it."""

        stage = self.stage_artifact(
            artifact,
            generation_id=generation_id,
            stream=stream,
            non_blocking=non_blocking,
        )
        receipt = self.publish_staged(stage)
        return self.confirm_publish(receipt)

    def _pin_digest(self, mirrors: Mapping[str, DeviceEpochMirror]) -> str:
        payload = [
            (seg_hash, mirror.commit_epoch, mirror.slot, mirror.seal_nonce)
            for seg_hash, mirror in sorted(mirrors.items())
        ]
        encoded = json.dumps(
            {
                "layout": self.layout.spec_fingerprint,
                "mirrors": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + sha256(encoded).hexdigest()

    def atomic_pin(
        self, restore_plan: object, *, stream: Optional[object] = None
    ) -> DeviceEpochPin:
        """Pin every CPU-plan epoch under one lock or pin nothing."""

        torch = _require_torch()
        if shared_latent_spec_fingerprint(restore_plan.spec) != self.layout.spec_fingerprint:
            raise ValueError("CPU restore plan is incompatible with this device store")
        epochs = {
            str(seg_hash): int(epoch)
            for seg_hash, epoch in restore_plan.artifact_epochs.items()
        }
        with self._lock:
            self._validate_banks()
            mirrors: Dict[str, DeviceEpochMirror] = {}
            for seg_hash, epoch in epochs.items():
                if any(key[0] == seg_hash for key in self._pending_publishes):
                    raise ValueError(
                        f"device mirror {seg_hash!r} has an unconfirmed publish"
                    )
                mirror = self._active.get(seg_hash)
                if mirror is None or mirror.commit_epoch != epoch:
                    raise ValueError(
                        f"device mirror {seg_hash!r} does not match CPU epoch {epoch}"
                    )
                if self._slot_mirrors.get(mirror.slot) is not mirror:
                    raise RuntimeError("active device mirror lost its sealed slot")
                mirrors[seg_hash] = mirror
            for mirror in mirrors.values():
                self._pin_counts[mirror.slot] += 1
            pin = DeviceEpochPin(
                store=self,
                mirrors=mirrors,
                digest=self._pin_digest(mirrors),
            )
        # Publication synchronizes uploads, but this wait keeps future
        # non-blocking publication implementations stream-correct.
        if self.device.type == "cuda" and stream is not None:
            _ = torch  # retain the explicit optional dependency boundary.
        return pin

    def _validate_pin(self, pin: DeviceEpochPin) -> None:
        with self._lock:
            self._validate_banks()
            if pin._store is not self:
                raise ValueError("device epoch pin belongs to another store")
            for mirror in pin.mirrors.values():
                if mirror.store_token is not self._token:
                    raise ValueError("device epoch mirror belongs to another store")
                if self._slot_mirrors.get(mirror.slot) is not mirror:
                    raise ValueError("device epoch pin lost its immutable slot")
                if self._pin_counts.get(mirror.slot, 0) <= 0:
                    raise ValueError("device epoch pin has no live slot lease")

    def _release_pin(self, pin: DeviceEpochPin) -> None:
        with self._lock:
            if pin._store is not self:
                raise ValueError("device epoch pin belongs to another store")
            for mirror in pin.mirrors.values():
                count = self._pin_counts.get(mirror.slot, 0)
                if count <= 0:
                    raise RuntimeError("device mirror pin reference count underflow")
                self._pin_counts[mirror.slot] = count - 1
            self._collect_retired_locked()

    def preflight(
        self,
        pin: DeviceEpochPin,
        restore_plan: object,
        *,
        forward_id: str,
    ) -> DeviceRestoreSchedule:
        pin.validate_open()
        epochs = {
            str(key): int(value) for key, value in restore_plan.artifact_epochs.items()
        }
        if dict(pin.epoch_by_segment) != epochs:
            raise ValueError("device epoch pin differs from the CPU restore plan")
        return compile_device_restore_schedule(
            layout=self.layout,
            restore_plan=restore_plan,
            slot_by_segment=pin.slot_by_segment,
            pin_digest=pin.digest,
            forward_id=forward_id,
        )

    def prepare(
        self,
        schedule: DeviceRestoreSchedule,
        pin: DeviceEpochPin,
        workspace: DeviceRestoreWorkspace,
        *,
        non_blocking: bool = True,
    ) -> PreparedDeviceRestore:
        pin.validate_open()
        if schedule.pin_digest != pin.digest:
            raise ValueError("restore schedule is not bound to this device pin")
        if schedule.layout_fingerprint != self.layout.spec_fingerprint:
            raise ValueError("restore schedule changed device layout")
        if workspace.device != self.device:
            raise ValueError("restore workspace is on another device")
        max_width = max(domain.record_bytes for domain in self.layout.domains)
        if workspace.max_record_bytes < max_width:
            raise MemoryError("restore workspace record width is too small")
        workspace.load(schedule, non_blocking=non_blocking)
        return PreparedDeviceRestore(self._token, pin, schedule, workspace)

    def _kernel_for_domain(
        self,
        kernels: Mapping[str, Callable[..., object]],
        domain: str,
    ) -> Callable[..., object]:
        kernel = kernels.get(domain)
        if not callable(kernel):
            raise ValueError(f"no fused restore kernel is registered for {domain}")
        return kernel

    def preflight_targets(
        self,
        prepared: PreparedDeviceRestore,
        *,
        targets: Mapping[Tuple[str, int], object],
        kernels: Mapping[str, Callable[..., object]],
        positions: object,
        forward_cache_slots: Optional[object] = None,
        target_slots: Optional[Mapping[Tuple[str, int], object]] = None,
        layer_id: Optional[int] = None,
    ) -> ValidatedDeviceRestore:
        """Validate every dependency before any target cache is mutated.

        ``layer_id`` is the normal model-forward mode: it selects only that
        layer's SWA plus C4/C128/Indexer/state operations.  Omitting it prepares
        all layers for a model-runner-owned eager restore.  Per-target slot
        vectors are required when SWA translation, compressor pages, and the
        Indexer use different pools; ``forward_cache_slots`` is only a concise
        fallback when every selected target intentionally shares one vector.
        """

        torch = _require_torch()
        if prepared.store_token is not self._token:
            raise ValueError("prepared restore belongs to another store")
        prepared.pin.validate_open()
        if prepared.workspace.loaded_digest != prepared.schedule.digest:
            raise ValueError("restore workspace lost its preflight schedule")
        row_count = len(prepared.schedule.positions)
        if layer_id is None:
            operations = prepared.schedule.operations
        else:
            layer_id = _strict_int(layer_id, "layer_id")
            operations = prepared.schedule.operations_for_layer(layer_id)
            if not operations:
                raise ValueError(f"restore schedule has no layer {layer_id}")
        if target_slots is None:
            if forward_cache_slots is None:
                raise ValueError(
                    "restore requires target_slots or a shared forward_cache_slots vector"
                )
            slots_by_key = {
                (operation.domain, operation.layer_id): forward_cache_slots
                for operation in operations
                if operation.count
            }
        else:
            slots_by_key = dict(target_slots)
        for name, tensor in (("positions", positions),):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.device != self.device:
                raise ValueError(f"{name} is on another device")
            if tensor.ndim != 1 or int(tensor.numel()) != row_count:
                raise ValueError(f"{name} must contain exactly {row_count} rows")
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")
        target_identities = {}
        slot_identities = {}
        for operation in operations:
            if operation.count == 0:
                continue
            key = (operation.domain, operation.layer_id)
            if key not in targets:
                raise ValueError(f"restore target {key!r} is missing")
            if key not in slots_by_key:
                raise ValueError(f"restore target slots {key!r} are missing")
            target = targets[key]
            if not isinstance(target, torch.Tensor):
                raise TypeError(f"restore target {key!r} must be a torch.Tensor")
            if target.device != self.device:
                raise ValueError(f"restore target {key!r} is on another device")
            slots = slots_by_key[key]
            if not isinstance(slots, torch.Tensor):
                raise TypeError(f"restore target slots {key!r} must be a torch.Tensor")
            if slots.device != self.device:
                raise ValueError(f"restore target slots {key!r} are on another device")
            if (
                slots.ndim != 1
                or int(slots.numel()) != row_count
                or slots.dtype != torch.long
            ):
                raise ValueError(
                    f"restore target slots {key!r} must be torch.long[{row_count}]"
                )
            self._kernel_for_domain(kernels, operation.domain)
            target_identities[key] = _tensor_identity(target)
            slot_identities[key] = _tensor_identity(slots)
        return ValidatedDeviceRestore(
            prepared,
            tuple(operations),
            MappingProxyType(dict(targets)),
            MappingProxyType(dict(kernels)),
            MappingProxyType(slots_by_key),
            positions,
            MappingProxyType(target_identities),
            MappingProxyType(slot_identities),
            _tensor_identity(positions),
        )

    def _validate_ready_restore(self, validated: ValidatedDeviceRestore) -> None:
        prepared = validated.prepared
        if prepared.store_token is not self._token:
            raise ValueError("validated restore belongs to another store")
        prepared.pin.validate_open()
        if prepared.workspace.loaded_digest != prepared.schedule.digest:
            raise ValueError("restore workspace changed after target preflight")
        if _tensor_identity(validated.positions) != validated.positions_identity:
            raise ValueError("positions changed after target preflight")
        for key, identity in validated.target_identities.items():
            if _tensor_identity(validated.targets[key]) != identity:
                raise ValueError(f"restore target {key!r} changed after preflight")
        for key, identity in validated.slot_identities.items():
            if _tensor_identity(validated.target_slots[key]) != identity:
                raise ValueError(
                    f"restore target slots {key!r} changed after preflight"
                )

    def prepare_batch_jobs(
        self,
        batch_input: DeviceRestoreBatchInput,
        *,
        input_ordinal: int,
    ) -> Tuple[DeviceRestoreBatchJob, ...]:
        """Bind one validated contribution without mutating a cache.

        This is the only store-aware part of cross-layer planning: it converts
        a domain lane into a flattened persistent source-bank pointer and
        retains the already-loaded source/output index views.  Model metadata
        receives one final geometry callback before it can enter a descriptor.
        """

        if batch_input.store is not self:
            raise ValueError("restore batch input belongs to another GPU store")
        validated = batch_input.validated
        self._validate_ready_restore(validated)
        prepared = validated.prepared
        schedule = prepared.schedule
        workspace = prepared.workspace
        jobs = []
        for operation in validated.operations:
            if operation.count == 0:
                continue
            key = (operation.domain, operation.layer_id)
            metadata = batch_input.operation_metadata[key]
            domain_layout = self.layout.domain(operation.domain)
            lane = domain_layout.lane(operation.layer_id)
            source_bank = self._banks[operation.domain][lane].view(
                self.max_segment_epochs * domain_layout.rows_per_layer,
                domain_layout.record_bytes,
            )
            source_indices = workspace.indices(operation.source_indices, schedule)
            output_rows = workspace.indices(operation.output_rows, schedule)
            validate = getattr(metadata, "validate_batch_geometry", None)
            if not callable(validate):
                raise TypeError(
                    "restore batch metadata must define validate_batch_geometry()"
                )
            validate(
                domain=operation.domain,
                layer_id=operation.layer_id,
                source_bank=source_bank,
                source_indices=source_indices,
                target_cache=validated.targets[key],
                target_slots=validated.target_slots[key],
                output_rows=output_rows,
                positions=validated.positions,
                record_bytes=domain_layout.record_bytes,
                position_semantics=operation.position_semantics,
            )
            jobs.append(
                DeviceRestoreBatchJob(
                    input_ordinal=int(input_ordinal),
                    domain=operation.domain,
                    family=restore_family_for_domain(operation.domain),
                    layer_id=operation.layer_id,
                    count=operation.count,
                    source_bank=source_bank,
                    source_indices=source_indices,
                    target_cache=validated.targets[key],
                    target_slots=validated.target_slots[key],
                    output_rows=output_rows,
                    positions=validated.positions,
                    record_bytes=domain_layout.record_bytes,
                    position_semantics=operation.position_semantics,
                    metadata=metadata,
                    metadata_identity=_batch_metadata_identity(metadata),
                )
            )
        return tuple(jobs)

    def restore_clean(self, validated: ValidatedDeviceRestore) -> DeviceRestoreReceipt:
        """Invoke one fused multi-segment scatter per layer/domain operation.

        The callback contract is keyword-only and allocation-free from this
        module's perspective.  A production callback should fuse source gather,
        destination RoPE/Indexer materialization, slot lookup, and scatter.
        State-domain callbacks restore opaque restart state at the supplied
        output anchors.
        """

        self._validate_ready_restore(validated)
        prepared = validated.prepared
        schedule = prepared.schedule
        workspace = prepared.workspace
        if not bool(workspace.restore_scratch_allocated):
            raise RuntimeError(
                "index-only restore workspace cannot execute the legacy "
                "per-operation restore path"
            )
        restored_by_domain = {domain: 0 for domain in ALL_DOMAINS}
        executed = 0
        for operation in validated.operations:
            if operation.count == 0:
                continue
            domain_layout = self.layout.domain(operation.domain)
            lane = domain_layout.lane(operation.layer_id)
            # [capacity, rows, bytes] is contiguous for a fixed lane, so its
            # first two dimensions can be flattened without allocation.
            source_bank = self._banks[operation.domain][lane].view(
                self.max_segment_epochs * domain_layout.rows_per_layer,
                domain_layout.record_bytes,
            )
            source_indices = workspace.indices(operation.source_indices, schedule)
            output_rows = workspace.indices(operation.output_rows, schedule)
            kernel = self._kernel_for_domain(validated.kernels, operation.domain)
            kernel(
                domain=operation.domain,
                layer_id=operation.layer_id,
                source_bank=source_bank,
                source_indices=source_indices,
                target_cache=validated.targets[(operation.domain, operation.layer_id)],
                target_slots=validated.target_slots[
                    (operation.domain, operation.layer_id)
                ],
                output_rows=output_rows,
                positions=validated.positions,
                scratch=workspace.scratch[
                    : operation.count, : domain_layout.record_bytes
                ],
                slot_scratch=workspace.slot_scratch[: operation.count],
                position_semantics=operation.position_semantics,
            )
            restored_by_domain[operation.domain] += operation.count
            executed += 1
        return DeviceRestoreReceipt(
            schedule.forward_id,
            schedule.digest,
            executed,
            sum(restored_by_domain.values()),
            MappingProxyType(
                {key: value for key, value in restored_by_domain.items() if value}
            ),
        )


def _compile_device_restore_batch_plan(
    *,
    inputs: Tuple[DeviceRestoreBatchInput, ...],
    forward_id: str,
) -> DeviceRestoreBatchPlan:
    forward_id = _nonempty_string(forward_id, "batch forward_id")
    jobs = []
    seen = set()
    input_digests = []
    for input_ordinal, batch_input in enumerate(inputs):
        store = batch_input.store
        prepare_jobs = getattr(store, "prepare_batch_jobs", None)
        if not callable(prepare_jobs):
            raise TypeError("restore batch input has no compatible GPU store")
        schedule = batch_input.validated.prepared.schedule
        input_digests.append(str(schedule.digest))
        for job in prepare_jobs(batch_input, input_ordinal=input_ordinal):
            # Repeating the same request/layer/domain would write the same
            # physical slots twice and usually means the forward coordinator
            # accidentally replayed a layer receipt.  Different requests have
            # different schedule digests and remain legal.
            key = (
                id(store),
                str(schedule.digest),
                job.domain,
                job.layer_id,
                int(job.target_cache.data_ptr()),
                int(job.target_slots.data_ptr()),
                int(job.positions.data_ptr()),
            )
            if key in seen:
                raise ValueError("restore batch contains a duplicate cache mutation")
            seen.add(key)
            jobs.append(job)
    if not jobs:
        raise ValueError("restore batch has no clean values")
    jobs.sort(
        key=lambda job: (
            RESTORE_FAMILIES.index(job.family),
            job.layer_id,
            job.input_ordinal,
            _DOMAIN_ORDER[job.domain],
        )
    )
    spans = {}
    cursor = 0
    for family in RESTORE_FAMILIES:
        count = sum(job.family == family for job in jobs)
        if count:
            spans[family] = IndexSpan(cursor, cursor + count)
            cursor += count
    digest_payload = {
        "forward_id": forward_id,
        "input_schedules": input_digests,
        "jobs": [
            (
                job.input_ordinal,
                job.domain,
                job.family,
                job.layer_id,
                job.count,
                job.record_bytes,
                job.position_semantics,
                _tensor_identity(job.source_bank),
                _tensor_identity(job.source_indices),
                _tensor_identity(job.target_cache),
                _tensor_identity(job.target_slots),
                _tensor_identity(job.output_rows),
                _tensor_identity(job.positions),
                job.metadata_identity,
                job.descriptor_values,
            )
            for job in jobs
        ],
    }
    encoded = json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return DeviceRestoreBatchPlan(
        forward_id=forward_id,
        jobs=tuple(jobs),
        family_spans=MappingProxyType(spans),
        input_schedule_digests=tuple(input_digests),
        digest="sha256:" + sha256(encoded).hexdigest(),
    )


def preflight_device_restore_batch(
    *,
    inputs: Sequence[DeviceRestoreBatchInput],
    forward_id: str,
    workspace: DeviceRestoreBatchWorkspace,
    kernels: Mapping[str, DeviceRestoreBatchKernel],
    preflight_kernel: Optional[DeviceRestoreBatchPreflightKernel] = None,
    require_production_certified: bool = True,
    non_blocking: bool = True,
) -> ValidatedDeviceRestoreBatch:
    """Validate and descriptor-pack an entire middle-layer restore.

    This function performs no target-cache mutation.  Distributed serving may
    therefore run its one readiness vote after this call and invoke
    :func:`restore_clean_batched` only when every rank accepted the identical
    forward transaction.
    """

    if type(require_production_certified) is not bool:
        raise TypeError("require_production_certified must be boolean")
    batch_inputs = tuple(inputs)
    if not batch_inputs or any(
        not isinstance(item, DeviceRestoreBatchInput) for item in batch_inputs
    ):
        raise ValueError("restore batch inputs must be non-empty and validated")
    if require_production_certified:
        bindings = tuple(
            (item.request_index, item.layer_id) for item in batch_inputs
        )
        if any(request < 0 or layer < 0 for request, layer in bindings):
            raise ValueError(
                "production restore batch inputs require request/layer bindings"
            )
        if len(set(bindings)) != len(bindings):
            raise ValueError("production restore batch repeated a request/layer input")
    if not isinstance(workspace, DeviceRestoreBatchWorkspace):
        raise TypeError("restore batch needs a persistent descriptor workspace")
    plan = _compile_device_restore_batch_plan(
        inputs=batch_inputs,
        forward_id=forward_id,
    )
    required_families = set(plan.family_spans)
    if set(kernels) != required_families:
        raise ValueError(
            "restore batch kernels differ from scheduled families "
            f"(missing={sorted(required_families-set(kernels))}, "
            f"extra={sorted(set(kernels)-required_families)})"
        )
    normalized = {}
    for family in RESTORE_FAMILIES:
        if family not in required_families:
            continue
        kernel = kernels[family]
        if not isinstance(kernel, DeviceRestoreBatchKernel):
            raise TypeError("restore batch kernels need explicit launch contracts")
        if kernel.family != family:
            raise ValueError("restore batch kernel is registered for another family")
        if require_production_certified and not kernel.production_certified:
            raise ValueError(
                f"restore batch family {family!r} is not production certified"
            )
        normalized[family] = kernel
    if preflight_kernel is not None and not isinstance(
        preflight_kernel, DeviceRestoreBatchPreflightKernel
    ):
        raise TypeError("restore batch preflight needs an explicit launch contract")
    if require_production_certified:
        if preflight_kernel is None or not preflight_kernel.production_certified:
            raise ValueError(
                "production restore batch requires a certified aggregate preflight"
            )
    devices = {str(job.source_bank.device) for job in plan.jobs}
    if len(devices) != 1 or str(workspace.device) not in devices:
        raise ValueError("restore batch jobs/workspace use different devices")
    workspace.load(plan, non_blocking=non_blocking)
    preflight_launch_count = 0
    if preflight_kernel is not None:
        preflight_kernel.callback(
            jobs=plan.jobs,
            descriptors=workspace.device_descriptors[: len(plan.jobs)],
            descriptor_columns=DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS,
            device_status=workspace.device_validation_status,
            host_status=workspace.host_validation_status,
            device_collision_table=workspace.device_collision_table[
                : workspace.loaded_collision_table_size
            ],
            max_validation_entries=workspace.loaded_validation_entries,
            plan_digest=plan.digest,
        )
        if int(workspace.host_validation_status[0]) != 0:
            raise ValueError(
                "aggregate restore batch descriptor validation failed with code "
                f"{int(workspace.host_validation_status[0])}"
            )
        preflight_launch_count = int(preflight_kernel.max_launches_per_call)
    return ValidatedDeviceRestoreBatch(
        inputs=batch_inputs,
        plan=plan,
        workspace=workspace,
        kernels=MappingProxyType(normalized),
        production_certified=bool(require_production_certified),
        workspace_identity=_tensor_identity(workspace.device_descriptors),
        validation_status_identity=_tensor_identity(
            workspace.device_validation_status
        ),
        collision_table_identity=_tensor_identity(
            workspace.device_collision_table
        ),
        preflight_launch_count=preflight_launch_count,
    )


def _validate_ready_restore_batch(batch: ValidatedDeviceRestoreBatch) -> None:
    if not isinstance(batch, ValidatedDeviceRestoreBatch):
        raise TypeError("restore batch was not preflighted")
    if batch.workspace.loaded_digest != batch.plan.digest:
        raise ValueError("restore batch workspace changed after preflight")
    if (
        _tensor_identity(batch.workspace.device_descriptors)
        != batch.workspace_identity
    ):
        raise ValueError("restore batch descriptors changed after preflight")
    if (
        _tensor_identity(batch.workspace.device_validation_status)
        != batch.validation_status_identity
        or _tensor_identity(batch.workspace.device_collision_table)
        != batch.collision_table_identity
        or int(batch.workspace.host_validation_status[0]) != 0
    ):
        raise ValueError("restore batch aggregate validation receipt changed")
    for batch_input in batch.inputs:
        validate = getattr(batch_input.store, "_validate_ready_restore", None)
        if not callable(validate):
            raise TypeError("restore batch store lost its validation hook")
        validate(batch_input.validated)
    for job in batch.plan.jobs:
        if _batch_metadata_identity(job.metadata) != job.metadata_identity:
            raise ValueError("restore batch metadata changed after preflight")


def _build_device_restore_batch_receipt(
    batch: ValidatedDeviceRestoreBatch,
    *,
    restored_by_domain: Mapping[str, int],
    restore_launch_count: int,
) -> DeviceRestoreBatchReceipt:
    """Build the common receipt after either monolithic or pipelined launch."""

    input_receipts = []
    for input_ordinal, batch_input in enumerate(batch.inputs):
        counts = {domain: 0 for domain in ALL_DOMAINS}
        operation_count = 0
        for job in batch.plan.jobs:
            if job.input_ordinal != input_ordinal:
                continue
            counts[job.domain] += job.count
            operation_count += 1
        schedule = batch_input.validated.prepared.schedule
        input_receipts.append(
            DeviceRestoreReceipt(
                forward_id=schedule.forward_id,
                schedule_digest=schedule.digest,
                operation_count=operation_count,
                restored_value_count=sum(counts.values()),
                restored_by_domain=MappingProxyType(
                    {domain: count for domain, count in counts.items() if count}
                ),
            )
        )
    return DeviceRestoreBatchReceipt(
        forward_id=batch.plan.forward_id,
        batch_digest=batch.plan.digest,
        launch_count=batch.preflight_launch_count + restore_launch_count,
        validation_launch_count=batch.preflight_launch_count,
        restore_launch_count=restore_launch_count,
        descriptor_h2d_bytes=(
            len(batch.plan.jobs)
            * int(batch.workspace.descriptor_width)
            * int(batch.workspace.device_descriptors.element_size())
        ),
        validation_control_h2d_bytes=int(
            batch.workspace.device_validation_status.element_size()
        ),
        validation_status_d2h_bytes=(
            int(batch.workspace.host_validation_status.element_size())
            if batch.preflight_launch_count
            else 0
        ),
        validation_memset_bytes=(
            int(batch.workspace.loaded_collision_table_size)
            * int(batch.workspace.device_collision_table.element_size())
        ),
        operation_count=batch.plan.operation_count,
        restored_value_count=batch.plan.restored_value_count,
        restored_by_domain=MappingProxyType(
            {
                domain: int(restored_by_domain.get(domain, 0))
                for domain in ALL_DOMAINS
                if int(restored_by_domain.get(domain, 0))
            }
        ),
        input_receipts=tuple(input_receipts),
        input_bindings=tuple(
            (item.request_index, item.layer_id) for item in batch.inputs
        ),
    )


def restore_clean_batched(
    batch: ValidatedDeviceRestoreBatch,
) -> DeviceRestoreBatchReceipt:
    """Invoke one certified pointer-table callback per semantic family."""

    _validate_ready_restore_batch(batch)
    restored_by_domain = {domain: 0 for domain in ALL_DOMAINS}
    restore_launch_count = 0
    if _os.environ.get("REDKNOT_RESTORE_SHAPE_DIAG", "0") == "1":
        # Opt-in work attribution for one batched restore: how many kernels are
        # launched and how many rows each domain rewrites.
        import logging as _logging

        _logging.getLogger(__name__).info(
            "REDKNOT_RESTORE_BATCH inputs=%s jobs=%s operations=%s restored_values=%s families=%s",
            len(batch.inputs),
            len(batch.plan.jobs),
            batch.plan.operation_count,
            batch.plan.restored_value_count,
            [f for f in RESTORE_FAMILIES if f in batch.plan.family_spans],
        )
    for family in RESTORE_FAMILIES:
        if family not in batch.plan.family_spans:
            continue
        kernel = batch.kernels[family]
        jobs = batch.plan.jobs_for_family(family)
        descriptors = batch.workspace.descriptors_for_family(family, batch.plan)
        kernel.callback(
            family=family,
            jobs=jobs,
            descriptors=descriptors,
            descriptor_columns=DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS,
            plan_digest=batch.plan.digest,
        )
        restore_launch_count += int(kernel.max_launches_per_call)
        for job in jobs:
            restored_by_domain[job.domain] += job.count

    return _build_device_restore_batch_receipt(
        batch,
        restored_by_domain=restored_by_domain,
        restore_launch_count=restore_launch_count,
    )


def restore_clean_batched_pipelined(
    batch: ValidatedDeviceRestoreBatch,
    *,
    pipeline: DeviceRestorePipelinePlan,
    group_enqueued: Optional[
        Callable[[DeviceRestorePipelineGroup], object]
    ] = None,
) -> DeviceRestoreBatchReceipt:
    """Launch exact family slices in layer order after one batch preflight.

    ``group_enqueued`` runs immediately after every family in the group was
    submitted.  A CUDA integration uses it to record one completion event on
    the same restore stream.  The callback may not mutate the plan/workspace;
    all identities are revalidated before the first launch and the pipeline is
    mechanically recompiled from the certificate-owned batch plan.
    """

    _validate_ready_restore_batch(batch)
    if not isinstance(pipeline, DeviceRestorePipelinePlan):
        raise TypeError("restore pipeline was not compiled")
    expected = compile_device_restore_pipeline(
        batch.plan, layers_per_group=pipeline.layers_per_group
    )
    if pipeline != expected or pipeline.batch_digest != batch.plan.digest:
        raise ValueError("restore pipeline differs from the validated batch")
    if group_enqueued is not None and not callable(group_enqueued):
        raise TypeError("restore pipeline completion hook is not callable")

    restored_by_domain = {domain: 0 for domain in ALL_DOMAINS}
    restore_launch_count = 0
    launched_ordinals = []
    for group in pipeline.groups:
        for family in RESTORE_FAMILIES:
            span = group.family_spans.get(family)
            if span is None:
                continue
            jobs = batch.plan.jobs[span.begin : span.end]
            if not jobs or any(
                job.family != family or job.layer_id not in group.layer_ids
                for job in jobs
            ):
                raise ValueError("restore pipeline family slice changed")
            descriptors = batch.workspace.device_descriptors[
                span.begin : span.end
            ]
            kernel = batch.kernels[family]
            kernel.callback(
                family=family,
                jobs=jobs,
                descriptors=descriptors,
                descriptor_columns=DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS,
                plan_digest=batch.plan.digest,
            )
            restore_launch_count += int(kernel.max_launches_per_call)
            for index, job in zip(range(span.begin, span.end), jobs):
                launched_ordinals.append(index)
                restored_by_domain[job.domain] += job.count
        if group_enqueued is not None:
            group_enqueued(group)

    if tuple(sorted(launched_ordinals)) != tuple(range(len(batch.plan.jobs))):
        raise RuntimeError("restore pipeline did not launch every batch job")
    if len(launched_ordinals) != len(set(launched_ordinals)):
        raise RuntimeError("restore pipeline launched a batch job more than once")
    if (
        sum(restored_by_domain.values()) != batch.plan.restored_value_count
        or pipeline.operation_count != batch.plan.operation_count
    ):
        raise RuntimeError("restore pipeline launch accounting changed")
    return _build_device_restore_batch_receipt(
        batch,
        restored_by_domain=restored_by_domain,
        restore_launch_count=restore_launch_count,
    )


def prepare_forward_batched_restore(
    *,
    inputs: Sequence[DeviceRestoreBatchInput],
    forward_id: str,
    workspace: DeviceRestoreBatchWorkspace,
    kernels: Mapping[str, DeviceRestoreBatchKernel],
    preflight_kernel: Optional[DeviceRestoreBatchPreflightKernel] = None,
    require_production_certified: bool = True,
    non_blocking: bool = True,
) -> ValidatedDeviceRestoreBatch:
    """Named forward-coordinator entry point; performs no cache mutation."""

    return preflight_device_restore_batch(
        inputs=inputs,
        forward_id=forward_id,
        workspace=workspace,
        kernels=kernels,
        preflight_kernel=preflight_kernel,
        require_production_certified=require_production_certified,
        non_blocking=non_blocking,
    )


def execute_forward_batched_restore(
    prepared: ValidatedDeviceRestoreBatch,
) -> DeviceRestoreBatchReceipt:
    """Execute a prepared transaction after the caller's TP readiness vote."""

    return restore_clean_batched(prepared)


def direct_positionless_scatter_kernel(
    *,
    domain: str,
    layer_id: int,
    source_bank: object,
    source_indices: object,
    target_cache: object,
    target_slots: object,
    output_rows: object,
    positions: object,
    scratch: object,
    slot_scratch: object,
    position_semantics: str,
) -> None:
    """Allocation-free reference scatter for *positionless* target caches.

    This helper is useful for CPU tests and for a future cache whose declared
    storage semantics exactly equal the artifact semantics.  FlashMLA's normal
    positioned cache must use a model-owned fused RoPE/Indexer callback instead.
    """

    _ = (domain, layer_id, positions, position_semantics)
    torch = _require_torch()
    count = int(source_indices.numel())
    selected = scratch[:count, : int(source_bank.shape[-1])]
    torch.index_select(source_bank, 0, source_indices, out=selected)
    torch.index_select(target_slots, 0, output_rows, out=slot_scratch)
    target_cache.index_copy_(0, slot_scratch, selected)


__all__ = [
    "ALL_DOMAINS",
    "DATA_DOMAINS",
    "DEVICE_MIRROR_FORMAT_VERSION",
    "DOMAIN_C4",
    "DOMAIN_C4_ATTENTION_STATE",
    "DOMAIN_C128",
    "DOMAIN_C128_ATTENTION_STATE",
    "DOMAIN_INDEXER",
    "DOMAIN_INDEXER_STATE",
    "DOMAIN_SWA",
    "DeviceDomainLayout",
    "DeviceEpochMirror",
    "DeviceEpochPin",
    "DevicePublishReceipt",
    "DEVICE_RESTORE_BATCH_COMMON_DESCRIPTOR_COLUMNS",
    "DeviceRestoreBatchInput",
    "DeviceRestoreBatchJob",
    "DeviceRestoreBatchKernel",
    "DeviceRestoreBatchPreflightKernel",
    "DeviceRestoreBatchPlan",
    "DeviceRestoreBatchReceipt",
    "DeviceRestoreBatchWorkspace",
    "DeviceRestorePipelineGroup",
    "DeviceRestorePipelinePlan",
    "DeviceRestoreReceipt",
    "DeviceRestoreSchedule",
    "DeviceRestoreWorkspace",
    "DirtyBlockWorkset",
    "DirtyLayerWorkset",
    "IndexSpan",
    "LayerRestoreOp",
    "PACKED_LATENT_BYTES",
    "PreparedDeviceRestore",
    "PreparedDevicePublish",
    "RESTORE_FAMILIES",
    "RESTORE_FAMILY_INDEXER",
    "RESTORE_FAMILY_PACKED",
    "RESTORE_FAMILY_STATE",
    "STATE_DOMAINS",
    "SharedLatentDeviceLayout",
    "SharedLatentGPUStore",
    "ValidatedDeviceRestore",
    "artifact_domain_payloads",
    "build_shared_latent_device_layout",
    "compile_device_restore_schedule",
    "compile_device_restore_pipeline",
    "direct_positionless_scatter_kernel",
    "execute_forward_batched_restore",
    "prepare_forward_batched_restore",
    "preflight_device_restore_batch",
    "restore_clean_batched",
    "restore_clean_batched_pipelined",
    "restore_family_for_domain",
    "shared_latent_device_nbytes",
    "shared_latent_spec_fingerprint",
]
