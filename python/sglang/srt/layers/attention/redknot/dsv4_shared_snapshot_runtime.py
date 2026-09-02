"""Atomic z_off + CPU latent + GPU-bank snapshot publication for DSV4-0731.

The three artifact stores already provide local staging/publication APIs, but
none of them alone can guarantee that a segment generation is visible in all
three stores.  This module supplies that missing transaction boundary without
importing torch, SGLang, or any of the concrete controller modules.

The exact topology is immutable: layers 0..2 and 40..42 are always online;
only layers 3..39 may be captured.  Publication is two phase::

    begin -> capture_layer(3..39) -> prepare_publish(TP certificate)
          -> publish (rollback receipts retained) -> confirm

Every error before confirmation invokes rollback/abort on z_off, CPU shared
latent, and the persistent GPU shared bank.  ``prepare_publish`` performs no
active-generation mutation.  The TP certificate is duck typed but must bind
the segment/token/model/policy/generation and exact TP rank/group.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union


SNAPSHOT_BUNDLE_FORMAT_VERSION = 1
DSV4_0731_TOTAL_LAYERS = 43
DENSE_PREFIX_LAYERS = 3
DENSE_SUFFIX_LAYERS = 3
REUSABLE_LAYER_IDS = tuple(range(3, 40))

if (
    DENSE_PREFIX_LAYERS
    + len(REUSABLE_LAYER_IDS)
    + DENSE_SUFFIX_LAYERS
    != DSV4_0731_TOTAL_LAYERS
):  # pragma: no cover - import-time source invariant
    raise RuntimeError("invalid DSV4-0731 3 + 37 + 3 snapshot topology")

DOMAIN_SWA = "swa"
DOMAIN_C4 = "c4"
DOMAIN_C128 = "c128"
DOMAIN_INDEXER = "indexer"
DOMAIN_C4_ATTENTION_STATE = "c4_attention_state"
DOMAIN_C128_ATTENTION_STATE = "c128_attention_state"
DOMAIN_INDEXER_STATE = "indexer_state"


class SnapshotTransactionError(RuntimeError):
    """A snapshot transaction failed and rollback was attempted."""


class SnapshotRollbackError(SnapshotTransactionError):
    """One or more participant rollbacks failed."""

    def __init__(self, failures: Sequence[Tuple[str, BaseException]]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(
            f"{participant}: {type(error).__name__}: {error}"
            for participant, error in self.failures
        )
        super().__init__(f"snapshot rollback was incomplete: {detail}")


class SnapshotConfirmError(SnapshotTransactionError):
    """An invariant failed after confirmation became irreversible."""


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
    except ValueError as error:
        raise ValueError(f"{name} must be a sha256:<64 hex> digest") from error
    return result


def _field(value: object, name: str, *aliases: str) -> object:
    candidates = (name,) + aliases
    if isinstance(value, Mapping):
        for candidate in candidates:
            if candidate in value:
                return value[candidate]
    else:
        for candidate in candidates:
            if hasattr(value, candidate):
                return getattr(value, candidate)
    raise ValueError(f"value is missing {name}")


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def _opaque_identity(value: object) -> Tuple[object, ...]:
    """Bind an opaque payload without reading device values."""

    identity = [id(value), type(value).__module__, type(value).__qualname__]
    for name in ("shape", "dtype", "device"):
        if hasattr(value, name):
            raw = getattr(value, name)
            if name == "shape":
                try:
                    raw = tuple(int(item) for item in raw)
                except (TypeError, ValueError):
                    raw = str(raw)
            else:
                raw = str(raw)
            identity.append((name, raw))
    try:
        version: object = int(getattr(value, "_version"))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        version = "opaque"
    identity.append(("version", version))
    return tuple(identity)


def _payload_identity(value: object) -> object:
    """Create a no-device-read binding for a staged payload object."""

    if isinstance(value, bytes):
        return ("bytes", len(value), "sha256:" + sha256(value).hexdigest())
    if isinstance(value, (str, int, bool)) or value is None:
        return (type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_payload_identity(item) for item in value))
    return _opaque_identity(value)


def _exact_indices(value: object, count: int, name: str) -> Tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    result = tuple(_strict_int(item, f"{name} entry") for item in value)
    if result != tuple(range(count)):
        raise ValueError(f"{name} must cover the complete canonical artifact")
    return result


@dataclass(frozen=True)
class SnapshotIdentity:
    seg_hash: str
    token_hash: str
    model_hash: str
    policy_hash: str
    generation_id: str
    token_count: int
    tp_rank: int
    tp_size: int
    format_version: int = SNAPSHOT_BUNDLE_FORMAT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "seg_hash",
            "token_hash",
            "model_hash",
            "policy_hash",
            "generation_id",
        ):
            _nonempty(getattr(self, name), name)
        if _strict_int(self.token_count, "token_count") <= 0:
            raise ValueError("token_count must be positive")
        rank = _strict_int(self.tp_rank, "tp_rank")
        size = _strict_int(self.tp_size, "tp_size")
        if size <= 0 or rank < 0 or rank >= size:
            raise ValueError("snapshot TP geometry is invalid")
        if self.format_version != SNAPSHOT_BUNDLE_FORMAT_VERSION:
            raise ValueError("snapshot bundle format is incompatible")

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "seg_hash": self.seg_hash,
            "token_hash": self.token_hash,
            "model_hash": self.model_hash,
            "policy_hash": self.policy_hash,
            "generation_id": self.generation_id,
            "token_count": self.token_count,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "reusable_layers": REUSABLE_LAYER_IDS,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_payload())


@dataclass(frozen=True)
class SnapshotTPCertificateBinding:
    certificate_digest: str
    seg_hash: str
    token_hash: str
    model_hash: str
    policy_hash: str
    generation_id: str
    tp_rank: int
    tp_size: int
    ready_rank_count: int
    certificate_object: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "seg_hash",
            "token_hash",
            "model_hash",
            "policy_hash",
            "generation_id",
        ):
            _nonempty(getattr(self, name), name)
        _sha_digest(self.certificate_digest, "certificate_digest")
        rank = _strict_int(self.tp_rank, "certificate tp_rank")
        size = _strict_int(self.tp_size, "certificate tp_size")
        ready = _strict_int(
            self.ready_rank_count, "certificate ready_rank_count"
        )
        if size <= 0 or rank < 0 or rank >= size or ready < 0 or ready > size:
            raise ValueError("TP certificate geometry is invalid")

    @classmethod
    def from_value(cls, value: object) -> "SnapshotTPCertificateBinding":
        if isinstance(value, cls):
            return value
        digest = _field(value, "certificate_digest", "digest")
        if callable(digest):
            digest = digest()
        return cls(
            certificate_digest=_sha_digest(digest, "certificate_digest"),
            seg_hash=_nonempty(_field(value, "seg_hash"), "certificate seg_hash"),
            token_hash=_nonempty(
                _field(value, "token_hash"), "certificate token_hash"
            ),
            model_hash=_nonempty(
                _field(value, "model_hash"), "certificate model_hash"
            ),
            policy_hash=_nonempty(
                _field(value, "policy_hash"), "certificate policy_hash"
            ),
            generation_id=_nonempty(
                _field(value, "generation_id"), "certificate generation_id"
            ),
            tp_rank=_strict_int(_field(value, "tp_rank"), "certificate tp_rank"),
            tp_size=_strict_int(_field(value, "tp_size"), "certificate tp_size"),
            ready_rank_count=_strict_int(
                _field(value, "ready_rank_count"),
                "certificate ready_rank_count",
            ),
            certificate_object=value,
        )

    def validate(self, identity: SnapshotIdentity) -> None:
        expected = (
            identity.seg_hash,
            identity.token_hash,
            identity.model_hash,
            identity.policy_hash,
            identity.generation_id,
            identity.tp_rank,
            identity.tp_size,
            identity.tp_size,
        )
        actual = (
            self.seg_hash,
            self.token_hash,
            self.model_hash,
            self.policy_hash,
            self.generation_id,
            self.tp_rank,
            self.tp_size,
            self.ready_rank_count,
        )
        if actual != expected:
            raise ValueError("TP certificate does not bind this snapshot generation")


@dataclass(frozen=True)
class SnapshotCheckpointCapture:
    anchor: int
    attention_state: object
    indexer_state: Optional[object] = None


@dataclass(frozen=True)
class SnapshotLayerCapture:
    """All z_off/CPU/GPU payloads for one reusable layer."""

    z_off_spec: object
    local_positions: object
    local_projection: object
    swa_rows: Tuple[int, ...]
    swa_positionless_packed: object
    compressed_blocks: Tuple[int, ...]
    compressed_positionless_packed: object
    attention_terminal_state: object
    checkpoints: Tuple[SnapshotCheckpointCapture, ...]
    gpu_components: Mapping[str, object]
    indexer_blocks: Tuple[int, ...] = ()
    indexer_positionless_keys: Optional[object] = None
    indexer_position_semantics: str = ""
    indexer_terminal_state: Optional[object] = None
    gpu_stream: Optional[object] = field(default=None, repr=False, compare=False)
    gpu_non_blocking: bool = True


@dataclass(frozen=True)
class LayerCaptureReceipt:
    identity_digest: str
    layer_id: int
    compression_ratio: int
    gpu_domains: Tuple[str, ...]
    capture_digest: str


@dataclass(eq=False)
class SnapshotBundle:
    runtime_token: object = field(repr=False)
    identity: SnapshotIdentity
    cpu_spec: object = field(repr=False)
    gpu_stage: object = field(repr=False)
    expected_gpu_components: Mapping[int, Tuple[str, ...]] = field(repr=False)
    captured_layers: Dict[int, LayerCaptureReceipt] = field(default_factory=dict)
    state: str = "capturing"


@dataclass(frozen=True, eq=False)
class PreparedSnapshotBundle:
    runtime_token: object = field(repr=False, compare=False)
    bundle: SnapshotBundle = field(repr=False, compare=False)
    tp_certificate: SnapshotTPCertificateBinding
    gpu_prepared: object = field(repr=False, compare=False)
    layer_manifest_digest: str
    prepare_digest: str


@dataclass(eq=False)
class PublishedSnapshotBundle:
    runtime_token: object = field(repr=False)
    prepared: PreparedSnapshotBundle = field(repr=False)
    cpu_receipt: object = field(repr=False)
    gpu_receipt: object = field(repr=False)
    z_off_receipt: object = field(repr=False)
    publish_digest: str
    state: str = "published"


def _layer_specs(cpu_spec: object) -> Mapping[int, object]:
    layers = tuple(_field(cpu_spec, "layers"))
    result = {_strict_int(_field(layer, "layer_id"), "layer_id"): layer for layer in layers}
    if tuple(sorted(result)) != REUSABLE_LAYER_IDS or len(result) != len(layers):
        raise ValueError("CPU shared-latent spec must cover exactly layers 3..39")
    return MappingProxyType(result)


def _ratio(layer_spec: object) -> int:
    ratio = _strict_int(_field(layer_spec, "compress_ratio", "ratio"), "ratio")
    if ratio not in (4, 128):
        raise ValueError("reusable DSV4-0731 layers must be C4 or C128")
    return ratio


def _expected_gpu_domains(ratio: int) -> Tuple[str, ...]:
    if ratio == 4:
        return tuple(
            sorted(
                (
                    DOMAIN_SWA,
                    DOMAIN_C4,
                    DOMAIN_INDEXER,
                    DOMAIN_C4_ATTENTION_STATE,
                    DOMAIN_INDEXER_STATE,
                )
            )
        )
    return tuple(
        sorted((DOMAIN_SWA, DOMAIN_C128, DOMAIN_C128_ATTENTION_STATE))
    )


def _gpu_layout_components(gpu_store: object) -> Mapping[int, Tuple[str, ...]]:
    layout = _field(gpu_store, "layout")
    domains = tuple(_field(layout, "domains"))
    by_layer: Dict[int, list[str]] = {layer_id: [] for layer_id in REUSABLE_LAYER_IDS}
    for domain in domains:
        name = _nonempty(_field(domain, "domain"), "GPU domain")
        for raw_layer in tuple(_field(domain, "layer_ids")):
            layer_id = _strict_int(raw_layer, "GPU domain layer_id")
            if layer_id not in by_layer:
                raise ValueError("GPU shared bank contains a dense boundary layer")
            by_layer[layer_id].append(name)
    return MappingProxyType(
        {layer_id: tuple(sorted(names)) for layer_id, names in by_layer.items()}
    )


class DSV4SharedSnapshotRuntime:
    """Backend-facing three-participant snapshot transaction coordinator."""

    def __init__(
        self,
        *,
        z_off_controller: object,
        cpu_shared_controller: object,
        gpu_shared_store: object,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        rank = _strict_int(tp_rank, "tp_rank")
        size = _strict_int(tp_size, "tp_size")
        if size <= 0 or rank < 0 or rank >= size:
            raise ValueError("snapshot runtime TP geometry is invalid")
        self.z_off_controller = z_off_controller
        self.cpu_shared_controller = cpu_shared_controller
        self.gpu_shared_store = gpu_shared_store
        self.tp_rank = rank
        self.tp_size = size
        self._token = object()
        self._lock = threading.RLock()
        self._active: Dict[Tuple[str, str], SnapshotBundle] = {}

    def _validate_bundle(self, bundle: SnapshotBundle, *states: str) -> None:
        if not isinstance(bundle, SnapshotBundle) or bundle.runtime_token is not self._token:
            raise ValueError("snapshot bundle belongs to another runtime")
        key = (bundle.identity.seg_hash, bundle.identity.generation_id)
        if self._active.get(key) is not bundle:
            raise ValueError("snapshot bundle is stale")
        if states and bundle.state not in states:
            raise ValueError(
                f"snapshot bundle state {bundle.state!r} is not one of {states!r}"
            )

    def _validate_topology(
        self,
        *,
        cpu_spec: object,
        model_hash: str,
        policy_hash: str,
        token_count: int,
    ) -> Mapping[int, Tuple[str, ...]]:
        required = tuple(_field(cpu_spec, "required_layer_ids"))
        if required != REUSABLE_LAYER_IDS:
            raise ValueError("CPU shared-latent required layers must be exactly 3..39")
        if _field(cpu_spec, "model_hash") != model_hash:
            raise ValueError("CPU shared-latent model hash changed")
        if _field(cpu_spec, "policy_hash") != policy_hash:
            raise ValueError("CPU shared-latent policy hash changed")
        if _strict_int(_field(cpu_spec, "length"), "CPU segment length") != token_count:
            raise ValueError("CPU shared-latent segment length changed")
        specs = _layer_specs(cpu_spec)
        expected = {
            layer_id: _expected_gpu_domains(_ratio(specs[layer_id]))
            for layer_id in REUSABLE_LAYER_IDS
        }
        layout = _field(self.gpu_shared_store, "layout")
        if _field(layout, "model_hash") != model_hash:
            raise ValueError("GPU shared-bank model hash changed")
        if _field(layout, "policy_hash") != policy_hash:
            raise ValueError("GPU shared-bank policy hash changed")
        if (
            _strict_int(_field(layout, "segment_length"), "GPU segment length")
            != token_count
        ):
            raise ValueError("GPU shared-bank segment length changed")
        if tuple(_field(layout, "checkpoint_anchors")) != tuple(
            _field(cpu_spec, "checkpoint_anchors")
        ):
            raise ValueError("CPU/GPU checkpoint topology differs")
        actual = _gpu_layout_components(self.gpu_shared_store)
        if dict(actual) != expected:
            raise ValueError("GPU shared-bank layer/domain topology is incomplete")
        return MappingProxyType(expected)

    def begin(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        token_hash: str,
        token_ids: Sequence[int],
        model_hash: str,
        policy_hash: str,
        cpu_spec: object,
        z_off_expected_bytes: int,
        z_off_token_positions: object,
        z_off_token_ids: object,
        z_off_expected_device_bytes: int = 0,
    ) -> SnapshotBundle:
        """Begin all three staging generations or leave none behind."""

        tokens = tuple(_strict_int(token, "token id") for token in token_ids)
        identity = SnapshotIdentity(
            seg_hash=_nonempty(seg_hash, "seg_hash"),
            token_hash=_nonempty(token_hash, "token_hash"),
            model_hash=_nonempty(model_hash, "model_hash"),
            policy_hash=_nonempty(policy_hash, "policy_hash"),
            generation_id=_nonempty(generation_id, "generation_id"),
            token_count=len(tokens),
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        expected_gpu = self._validate_topology(
            cpu_spec=cpu_spec,
            model_hash=identity.model_hash,
            policy_hash=identity.policy_hash,
            token_count=identity.token_count,
        )
        z_off_expected_bytes = _strict_int(
            z_off_expected_bytes, "z_off_expected_bytes"
        )
        z_off_expected_device_bytes = _strict_int(
            z_off_expected_device_bytes, "z_off_expected_device_bytes"
        )
        if z_off_expected_bytes <= 0 or z_off_expected_device_bytes < 0:
            raise ValueError("z_off snapshot byte reservations are invalid")
        for name, value in (
            ("z_off_token_positions", z_off_token_positions),
            ("z_off_token_ids", z_off_token_ids),
        ):
            numel = getattr(value, "numel", None)
            if not callable(numel) or int(numel()) != identity.token_count:
                raise ValueError(f"{name} must cover the complete segment")

        key = (identity.seg_hash, identity.generation_id)
        with self._lock:
            if key in self._active:
                raise ValueError("this snapshot generation is already active")

        zoff_started = False
        cpu_started = False
        gpu_stage = None
        try:
            self.z_off_controller.begin_staging(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                token_hash=identity.token_hash,
                length=identity.token_count,
                canonical_start_pos=0,
                model_compat_hash=identity.model_hash,
                head_policy_hash=identity.policy_hash,
                required_local_layers=REUSABLE_LAYER_IDS,
                expected_bytes=z_off_expected_bytes,
                expected_device_bytes=z_off_expected_device_bytes,
            )
            zoff_started = True
            self.z_off_controller.capture_token_rows(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                local_positions=z_off_token_positions,
                token_ids=z_off_token_ids,
            )
            self.cpu_shared_controller.begin_capture(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                token_ids=tokens,
                spec=cpu_spec,
                token_hash=identity.token_hash,
            )
            cpu_started = True
            commit_epoch = _strict_int(
                _field(self.cpu_shared_controller, "next_commit_epoch"),
                "CPU next commit epoch",
            )
            gpu_stage = self.gpu_shared_store.begin_stage(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                commit_epoch=commit_epoch,
            )
        except BaseException as begin_error:
            failures = []
            if gpu_stage is not None:
                try:
                    self.gpu_shared_store.abort_staged(gpu_stage)
                except BaseException as error:
                    failures.append(("gpu", error))
            if cpu_started:
                try:
                    self.cpu_shared_controller.abort_capture(
                        seg_hash=identity.seg_hash,
                        generation_id=identity.generation_id,
                    )
                except BaseException as error:
                    failures.append(("cpu", error))
            if zoff_started:
                try:
                    self.z_off_controller.abort_staging(
                        identity.seg_hash, identity.generation_id
                    )
                except BaseException as error:
                    failures.append(("z_off", error))
            if failures:
                raise SnapshotRollbackError(failures) from begin_error
            raise

        bundle = SnapshotBundle(
            runtime_token=self._token,
            identity=identity,
            cpu_spec=cpu_spec,
            gpu_stage=gpu_stage,
            expected_gpu_components=expected_gpu,
        )
        with self._lock:
            if key in self._active:
                # A concurrent begin won after the preflight check.  Abort this
                # complete staging set rather than replacing its handle.
                failures = self._rollback_staging(bundle)
                if failures:
                    raise SnapshotRollbackError(failures)
                raise ValueError("this snapshot generation became active concurrently")
            self._active[key] = bundle
        return bundle

    def _validate_layer_capture(
        self,
        bundle: SnapshotBundle,
        layer_id: int,
        capture: SnapshotLayerCapture,
    ) -> Tuple[object, int, Tuple[str, ...]]:
        if type(layer_id) is not int or layer_id not in REUSABLE_LAYER_IDS:
            raise ValueError("snapshot capture may target only layers 3..39")
        if layer_id in bundle.captured_layers:
            raise ValueError("snapshot layer was already captured")
        if not isinstance(capture, SnapshotLayerCapture):
            raise TypeError("capture must be SnapshotLayerCapture")
        specs = _layer_specs(bundle.cpu_spec)
        layer_spec = specs[layer_id]
        ratio = _ratio(layer_spec)
        zoff_layer = _strict_int(_field(capture.z_off_spec, "layer_id"), "z_off layer_id")
        if zoff_layer != layer_id:
            raise ValueError("z_off capture spec belongs to another layer")
        if (
            _field(capture.z_off_spec, "model_compat_hash", "model_hash")
            != bundle.identity.model_hash
        ):
            raise ValueError("z_off capture model hash changed")
        if (
            _field(capture.z_off_spec, "head_policy_hash", "policy_hash")
            != bundle.identity.policy_hash
        ):
            raise ValueError("z_off capture policy hash changed")
        _exact_indices(
            capture.swa_rows,
            bundle.identity.token_count,
            "SWA rows",
        )
        block_count = bundle.identity.token_count // ratio
        if block_count * ratio != bundle.identity.token_count:
            raise ValueError("segment length is not compressor-ratio aligned")
        _exact_indices(capture.compressed_blocks, block_count, "compressed blocks")
        if (
            capture.swa_positionless_packed is None
            or capture.compressed_positionless_packed is None
            or capture.attention_terminal_state is None
        ):
            raise ValueError("CPU shared-latent cache/state payload is incomplete")
        positions_numel = getattr(capture.local_positions, "numel", None)
        if not callable(positions_numel) or int(positions_numel()) != bundle.identity.token_count:
            raise ValueError("z_off positions must cover the complete segment")
        projection_shape = getattr(capture.local_projection, "shape", None)
        if (
            projection_shape is None
            or len(projection_shape) < 1
            or int(projection_shape[0]) != bundle.identity.token_count
        ):
            raise ValueError("z_off projection must cover the complete segment")
        checkpoints = capture.checkpoints
        if type(checkpoints) is not tuple or any(
            not isinstance(item, SnapshotCheckpointCapture) for item in checkpoints
        ):
            raise TypeError("checkpoint captures must be an immutable tuple")
        required_anchors = tuple(_field(bundle.cpu_spec, "checkpoint_anchors"))
        anchors = tuple(item.anchor for item in checkpoints)
        if anchors != required_anchors:
            raise ValueError("snapshot checkpoints do not cover every required anchor")
        if any(item.attention_state is None for item in checkpoints):
            raise ValueError("snapshot is missing attention checkpoint state")
        if ratio == 4:
            _exact_indices(capture.indexer_blocks, block_count, "Indexer blocks")
            if (
                capture.indexer_blocks != capture.compressed_blocks
                or capture.indexer_positionless_keys is None
                or capture.indexer_position_semantics
                != _field(bundle.cpu_spec, "indexer_position_semantics")
                or capture.indexer_terminal_state is None
                or any(item.indexer_state is None for item in checkpoints)
            ):
                raise ValueError("C4 snapshot is missing Indexer cache/state")
        elif (
            capture.indexer_blocks
            or capture.indexer_positionless_keys is not None
            or capture.indexer_terminal_state is not None
            or any(item.indexer_state is not None for item in checkpoints)
        ):
            raise ValueError("C128 snapshot cannot contain Indexer cache/state")
        expected_gpu = bundle.expected_gpu_components[layer_id]
        if tuple(sorted(capture.gpu_components)) != expected_gpu:
            raise ValueError("GPU snapshot layer domains are incomplete")
        if any(capture.gpu_components[domain] is None for domain in expected_gpu):
            raise ValueError("GPU snapshot contains an absent domain payload")
        if type(capture.gpu_non_blocking) is not bool:
            raise TypeError("gpu_non_blocking must be boolean")
        return layer_spec, ratio, expected_gpu

    def capture_layer(
        self,
        bundle: SnapshotBundle,
        *,
        layer_id: int,
        capture: SnapshotLayerCapture,
    ) -> LayerCaptureReceipt:
        """Capture one complete middle-layer triple or rollback the bundle."""

        with self._lock:
            self._validate_bundle(bundle, "capturing")
        try:
            layer_spec, ratio, gpu_domains = self._validate_layer_capture(
                bundle, layer_id, capture
            )
            identity = bundle.identity
            self.cpu_shared_controller.capture_swa_rows(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                layer_id=layer_id,
                local_rows=capture.swa_rows,
                positionless_packed=capture.swa_positionless_packed,
            )
            self.cpu_shared_controller.capture_compressed_blocks(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                layer_id=layer_id,
                local_blocks=capture.compressed_blocks,
                positionless_packed=capture.compressed_positionless_packed,
            )
            if ratio == 4:
                self.cpu_shared_controller.capture_indexer_blocks(
                    seg_hash=identity.seg_hash,
                    generation_id=identity.generation_id,
                    layer_id=layer_id,
                    local_blocks=capture.indexer_blocks,
                    positionless_keys=capture.indexer_positionless_keys,
                    position_semantics=capture.indexer_position_semantics,
                )
            self.cpu_shared_controller.capture_terminal_states(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                layer_id=layer_id,
                attention_state=capture.attention_terminal_state,
                indexer_state=(capture.indexer_terminal_state if ratio == 4 else None),
            )
            for checkpoint in capture.checkpoints:
                self.cpu_shared_controller.capture_checkpoint_states(
                    seg_hash=identity.seg_hash,
                    generation_id=identity.generation_id,
                    layer_id=layer_id,
                    anchor=checkpoint.anchor,
                    attention_state=checkpoint.attention_state,
                    indexer_state=(checkpoint.indexer_state if ratio == 4 else None),
                )
            for domain in gpu_domains:
                self.gpu_shared_store.capture_component(
                    bundle.gpu_stage,
                    domain=domain,
                    layer_id=layer_id,
                    payload=capture.gpu_components[domain],
                    stream=capture.gpu_stream,
                    non_blocking=capture.gpu_non_blocking,
                )
            self.z_off_controller.capture_rows(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
                layer_id=layer_id,
                spec=capture.z_off_spec,
                local_positions=capture.local_positions,
                local_projection=capture.local_projection,
            )
            capture_digest = _digest(
                {
                    "identity": identity.digest,
                    "layer_id": layer_id,
                    "ratio": ratio,
                    "swa_rows": capture.swa_rows,
                    "compressed_blocks": capture.compressed_blocks,
                    "indexer_blocks": capture.indexer_blocks,
                    "checkpoint_anchors": tuple(item.anchor for item in capture.checkpoints),
                    "swa_payload": _payload_identity(
                        capture.swa_positionless_packed
                    ),
                    "compressed_payload": _payload_identity(
                        capture.compressed_positionless_packed
                    ),
                    "indexer_payload": _payload_identity(
                        capture.indexer_positionless_keys
                    ),
                    "terminal_state": _payload_identity(
                        capture.attention_terminal_state
                    ),
                    "indexer_terminal_state": _payload_identity(
                        capture.indexer_terminal_state
                    ),
                    "checkpoint_states": tuple(
                        (
                            item.anchor,
                            _payload_identity(item.attention_state),
                            _payload_identity(item.indexer_state),
                        )
                        for item in capture.checkpoints
                    ),
                    "gpu_components": tuple(
                        (
                            domain,
                            _payload_identity(capture.gpu_components[domain]),
                        )
                        for domain in gpu_domains
                    ),
                    "z_off_positions": _opaque_identity(capture.local_positions),
                    "z_off_projection": _opaque_identity(capture.local_projection),
                }
            )
            receipt = LayerCaptureReceipt(
                identity_digest=identity.digest,
                layer_id=layer_id,
                compression_ratio=ratio,
                gpu_domains=gpu_domains,
                capture_digest=capture_digest,
            )
            bundle.captured_layers[layer_id] = receipt
            return receipt
        except Exception as error:
            try:
                self.rollback(bundle)
            except SnapshotRollbackError as rollback_error:
                raise rollback_error from error
            raise SnapshotTransactionError(
                f"snapshot layer {layer_id} capture failed and was rolled back"
            ) from error

    def _layer_manifest(self, bundle: SnapshotBundle) -> str:
        if tuple(sorted(bundle.captured_layers)) != REUSABLE_LAYER_IDS:
            missing = sorted(set(REUSABLE_LAYER_IDS) - set(bundle.captured_layers))
            extra = sorted(set(bundle.captured_layers) - set(REUSABLE_LAYER_IDS))
            raise ValueError(
                f"snapshot layer capture is incomplete (missing={missing}, extra={extra})"
            )
        return _digest(
            [
                (layer_id, bundle.captured_layers[layer_id].capture_digest)
                for layer_id in REUSABLE_LAYER_IDS
            ]
        )

    def prepare_publish(
        self,
        bundle: SnapshotBundle,
        *,
        tp_certificate: object,
        synchronize_gpu: bool = True,
    ) -> PreparedSnapshotBundle:
        """Seal completeness and TP identity without publishing active state."""

        with self._lock:
            self._validate_bundle(bundle, "capturing")
        try:
            manifest = self._layer_manifest(bundle)
            certificate = SnapshotTPCertificateBinding.from_value(tp_certificate)
            certificate.validate(bundle.identity)
            if not self.z_off_controller.staging_complete(
                bundle.identity.seg_hash, bundle.identity.generation_id
            ):
                raise ValueError("z_off staging is incomplete")
            gpu_prepared = self.gpu_shared_store.prepare_publish(
                bundle.gpu_stage, synchronize=bool(synchronize_gpu)
            )
            if (
                _field(gpu_prepared, "seg_hash") != bundle.identity.seg_hash
                or _field(gpu_prepared, "generation_id")
                != bundle.identity.generation_id
            ):
                raise ValueError("GPU prepared publish belongs to another snapshot")
            prepare_digest = _digest(
                {
                    "identity": bundle.identity.digest,
                    "layer_manifest": manifest,
                    "tp_certificate": certificate.certificate_digest,
                    "gpu_epoch": _field(gpu_prepared, "commit_epoch"),
                    "gpu_slot": _field(gpu_prepared, "slot"),
                }
            )
            prepared = PreparedSnapshotBundle(
                runtime_token=self._token,
                bundle=bundle,
                tp_certificate=certificate,
                gpu_prepared=gpu_prepared,
                layer_manifest_digest=manifest,
                prepare_digest=prepare_digest,
            )
            bundle.state = "prepared"
            return prepared
        except Exception as error:
            try:
                self.rollback(bundle)
            except SnapshotRollbackError as rollback_error:
                raise rollback_error from error
            raise SnapshotTransactionError(
                "snapshot prepare failed and all participants were rolled back"
            ) from error

    def _validate_prepared(self, prepared: PreparedSnapshotBundle) -> SnapshotBundle:
        if (
            not isinstance(prepared, PreparedSnapshotBundle)
            or prepared.runtime_token is not self._token
        ):
            raise ValueError("prepared snapshot belongs to another runtime")
        bundle = prepared.bundle
        self._validate_bundle(bundle, "prepared")
        prepared.tp_certificate.validate(bundle.identity)
        if prepared.layer_manifest_digest != self._layer_manifest(bundle):
            raise ValueError("prepared snapshot layer manifest changed")
        return bundle

    def publish(self, prepared: PreparedSnapshotBundle) -> PublishedSnapshotBundle:
        """Locally publish all three generations while retaining rollback."""

        if (
            not isinstance(prepared, PreparedSnapshotBundle)
            or prepared.runtime_token is not self._token
        ):
            raise ValueError("prepared snapshot belongs to another runtime")
        bundle = prepared.bundle
        with self._lock:
            self._validate_bundle(bundle, "prepared")
        cpu_receipt = gpu_receipt = zoff_receipt = None
        identity = bundle.identity
        try:
            # Manifest/certificate validation is inside the transaction: a
            # mutated prepared handle must abort every still-staged store.
            with self._lock:
                self._validate_prepared(prepared)
            cpu_receipt = self.cpu_shared_controller.publish_capture(
                seg_hash=identity.seg_hash,
                generation_id=identity.generation_id,
            )
            artifact = _field(cpu_receipt, "artifact")
            if (
                _field(artifact, "seg_hash") != identity.seg_hash
                or _field(artifact, "token_hash") != identity.token_hash
                or _field(artifact, "spec") != bundle.cpu_spec
                or _field(artifact, "commit_epoch")
                != _field(prepared.gpu_prepared, "commit_epoch")
            ):
                raise ValueError("CPU published artifact changed snapshot identity")
            gpu_receipt = self.gpu_shared_store.publish(prepared.gpu_prepared)
            if (
                _field(gpu_receipt, "seg_hash") != identity.seg_hash
                or _field(gpu_receipt, "generation_id") != identity.generation_id
                or _field(gpu_receipt, "commit_epoch")
                != _field(artifact, "commit_epoch")
            ):
                raise ValueError("GPU published epoch differs from CPU artifact")
            zoff_receipt = self.z_off_controller.publish_staging(
                identity.seg_hash, identity.generation_id
            )
            if (
                _field(zoff_receipt, "seg_hash") != identity.seg_hash
                or _field(zoff_receipt, "generation_id") != identity.generation_id
            ):
                raise ValueError("z_off publish receipt changed snapshot identity")
            publish_digest = _digest(
                {
                    "prepare": prepared.prepare_digest,
                    "cpu_epoch": _field(artifact, "commit_epoch"),
                    "gpu_epoch": _field(gpu_receipt, "commit_epoch"),
                    "z_off_epoch": _field(zoff_receipt, "commit_epoch"),
                }
            )
            published = PublishedSnapshotBundle(
                runtime_token=self._token,
                prepared=prepared,
                cpu_receipt=cpu_receipt,
                gpu_receipt=gpu_receipt,
                z_off_receipt=zoff_receipt,
                publish_digest=publish_digest,
            )
            bundle.state = "published"
            return published
        except Exception as error:
            failures = self._rollback_parts(
                bundle=bundle,
                cpu_receipt=cpu_receipt,
                gpu_receipt=gpu_receipt,
                zoff_receipt=zoff_receipt,
            )
            if failures:
                raise SnapshotRollbackError(failures) from error
            raise SnapshotTransactionError(
                "snapshot publish failed and all participants were rolled back"
            ) from error

    def _rollback_staging(self, bundle: SnapshotBundle) -> Tuple[Tuple[str, BaseException], ...]:
        failures = []
        identity = bundle.identity
        for participant, callback in (
            ("gpu", lambda: self.gpu_shared_store.abort_staged(bundle.gpu_stage)),
            (
                "cpu",
                lambda: self.cpu_shared_controller.abort_capture(
                    seg_hash=identity.seg_hash,
                    generation_id=identity.generation_id,
                ),
            ),
            (
                "z_off",
                lambda: self.z_off_controller.abort_staging(
                    identity.seg_hash, identity.generation_id
                ),
            ),
        ):
            try:
                callback()
            except BaseException as error:
                failures.append((participant, error))
        return tuple(failures)

    def _rollback_parts(
        self,
        *,
        bundle: SnapshotBundle,
        cpu_receipt: Optional[object],
        gpu_receipt: Optional[object],
        zoff_receipt: Optional[object],
    ) -> Tuple[Tuple[str, BaseException], ...]:
        failures = []
        identity = bundle.identity
        operations = []
        operations.append(
            (
                "z_off",
                (
                    (lambda: self.z_off_controller.rollback_publish(zoff_receipt))
                    if zoff_receipt is not None
                    else lambda: self.z_off_controller.abort_staging(
                        identity.seg_hash, identity.generation_id
                    )
                ),
            )
        )
        operations.append(
            (
                "gpu",
                (
                    (lambda: self.gpu_shared_store.rollback_publish(gpu_receipt))
                    if gpu_receipt is not None
                    else lambda: self.gpu_shared_store.abort_staged(bundle.gpu_stage)
                ),
            )
        )
        operations.append(
            (
                "cpu",
                (
                    (lambda: self.cpu_shared_controller.rollback_publish(cpu_receipt))
                    if cpu_receipt is not None
                    else lambda: self.cpu_shared_controller.abort_capture(
                        seg_hash=identity.seg_hash,
                        generation_id=identity.generation_id,
                    )
                ),
            )
        )
        for participant, callback in operations:
            try:
                callback()
            except BaseException as error:
                failures.append((participant, error))
        bundle.state = "rolled_back"
        with self._lock:
            self._active.pop((identity.seg_hash, identity.generation_id), None)
        return tuple(failures)

    def rollback(
        self,
        value: Union[
            SnapshotBundle,
            PreparedSnapshotBundle,
            PublishedSnapshotBundle,
        ],
    ) -> None:
        """Rollback a staging, prepared, or locally-published bundle."""

        if isinstance(value, PublishedSnapshotBundle):
            if value.runtime_token is not self._token:
                raise ValueError("published snapshot belongs to another runtime")
            if value.state == "rolled_back":
                return
            if value.state != "published":
                raise ValueError("confirmed snapshot cannot be rolled back")
            bundle = value.prepared.bundle
            failures = self._rollback_parts(
                bundle=bundle,
                cpu_receipt=value.cpu_receipt,
                gpu_receipt=value.gpu_receipt,
                zoff_receipt=value.z_off_receipt,
            )
            value.state = "rolled_back"
        else:
            if isinstance(value, PreparedSnapshotBundle):
                if value.runtime_token is not self._token:
                    raise ValueError("prepared snapshot belongs to another runtime")
                bundle = value.bundle
            else:
                bundle = value
            if (
                isinstance(bundle, SnapshotBundle)
                and bundle.runtime_token is self._token
                and bundle.state == "rolled_back"
            ):
                return
            with self._lock:
                self._validate_bundle(bundle, "capturing", "prepared")
            failures = self._rollback_staging(bundle)
            bundle.state = "rolled_back"
            with self._lock:
                self._active.pop(
                    (bundle.identity.seg_hash, bundle.identity.generation_id), None
                )
        if failures:
            raise SnapshotRollbackError(failures)

    def _prevalidate_confirmation(self, published: PublishedSnapshotBundle) -> None:
        if published.runtime_token is not self._token or published.state != "published":
            raise ValueError("published snapshot is stale")
        bundle = published.prepared.bundle
        self._validate_bundle(bundle, "published")
        published.prepared.tp_certificate.validate(bundle.identity)
        self.z_off_controller.validate_publish_confirmation(published.z_off_receipt)
        self.gpu_shared_store.validate_publish_confirmation(published.gpu_receipt)
        cpu_receipt = published.cpu_receipt
        if _field(cpu_receipt, "state") != "published":
            raise ValueError("CPU shared-latent publish receipt is not pending")
        artifact = _field(cpu_receipt, "artifact")
        if self.cpu_shared_controller.get_committed(bundle.identity.seg_hash) is not artifact:
            raise ValueError("CPU shared-latent artifact is no longer current")

    def confirm(self, published: PublishedSnapshotBundle) -> None:
        """Validate all receipts, then irreversibly confirm all participants."""

        try:
            with self._lock:
                self._prevalidate_confirmation(published)
        except Exception as error:
            try:
                self.rollback(published)
            except SnapshotRollbackError as rollback_error:
                raise rollback_error from error
            raise SnapshotTransactionError(
                "snapshot confirmation preflight failed and was rolled back"
            ) from error

        confirmed = []
        try:
            # GPU confirmation is deliberately last.  Its atomic-pin API
            # rejects pending publishes, so it remains the serving visibility
            # gate while CPU/z_off release their rollback generations.
            self.cpu_shared_controller.confirm_publish(published.cpu_receipt)
            confirmed.append("cpu")
            self.z_off_controller.confirm_publish(published.z_off_receipt)
            confirmed.append("z_off")
            self.gpu_shared_store.confirm_publish(published.gpu_receipt)
            confirmed.append("gpu")
        except Exception as error:
            # Every participant was prevalidated under this runtime's lock.
            # A failure here indicates external concurrent mutation or a broken
            # participant no-fail confirm contract; already-confirmed stores no
            # longer expose rollback handles.
            published.state = "confirm_failed"
            published.prepared.bundle.state = "confirm_failed"
            raise SnapshotConfirmError(
                f"snapshot confirm failed after {tuple(confirmed)!r} confirmed"
            ) from error

        published.state = "confirmed"
        bundle = published.prepared.bundle
        bundle.state = "confirmed"
        with self._lock:
            self._active.pop(
                (bundle.identity.seg_hash, bundle.identity.generation_id), None
            )


__all__ = [
    "DENSE_PREFIX_LAYERS",
    "DENSE_SUFFIX_LAYERS",
    "DOMAIN_C4",
    "DOMAIN_C4_ATTENTION_STATE",
    "DOMAIN_C128",
    "DOMAIN_C128_ATTENTION_STATE",
    "DOMAIN_INDEXER",
    "DOMAIN_INDEXER_STATE",
    "DOMAIN_SWA",
    "DSV4SharedSnapshotRuntime",
    "DSV4_0731_TOTAL_LAYERS",
    "LayerCaptureReceipt",
    "PreparedSnapshotBundle",
    "PublishedSnapshotBundle",
    "REUSABLE_LAYER_IDS",
    "SNAPSHOT_BUNDLE_FORMAT_VERSION",
    "SnapshotBundle",
    "SnapshotCheckpointCapture",
    "SnapshotConfirmError",
    "SnapshotIdentity",
    "SnapshotLayerCapture",
    "SnapshotRollbackError",
    "SnapshotTPCertificateBinding",
    "SnapshotTransactionError",
]
