"""SGLang chunk adapter for the atomic DSV4 shared snapshot transaction.

This module is a pure control plane: it imports neither torch nor concrete
SGLang modules.  The backend supplies the production
``dsv4_shared_latent_sglang`` module (or an equivalent duck-typed object), and
the adapter drives its overwrite-safe layer staging functions::

    session = adapter.begin_segment(...)
    adapter.capture_chunk(session, layer_id=..., ...)  # for layers 3..39
    local = adapter.prepare_local(session)  # bind local.digest in one TP vote
    prepared = adapter.prepare_publish(local, tp_certificate=certificate)
    published = adapter.publish(prepared)
    adapter.validate_confirmation(published)  # final all-TP confirmation vote
    adapter.confirm(published)

The persistent GPU bank remains owned by ``DSV4SharedSnapshotRuntime``.  A
finalized SGLang layer bundle contributes its device ``components`` mapping to
that runtime exactly once, while ``export_cpu_components`` supplies bytes from
the same generation to the CPU artifact controller.  State rows are split as
``internal checkpoint anchors + terminal``; no state is reconstructed.

Any missing chunk, domain, state row, TP binding, or duck-typed method fails
closed and rolls back the z_off, CPU shared-latent, and GPU-bank transaction.
"""

from __future__ import annotations

import inspect
import json
import threading
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

try:  # Installed production location.
    from sglang.srt.layers.attention.redknot.dsv4_shared_snapshot_runtime import (
        DOMAIN_C4,
        DOMAIN_C4_ATTENTION_STATE,
        DOMAIN_C128,
        DOMAIN_C128_ATTENTION_STATE,
        DOMAIN_INDEXER,
        DOMAIN_INDEXER_STATE,
        DOMAIN_SWA,
        REUSABLE_LAYER_IDS,
        SnapshotCheckpointCapture,
        SnapshotLayerCapture,
        SnapshotRollbackError as RuntimeSnapshotRollbackError,
    )
except ImportError:  # Standalone work/reuse_full_scope execution and CPU tests.
    from dsv4_shared_snapshot_runtime import (  # type: ignore[no-redef]
        DOMAIN_C4,
        DOMAIN_C4_ATTENTION_STATE,
        DOMAIN_C128,
        DOMAIN_C128_ATTENTION_STATE,
        DOMAIN_INDEXER,
        DOMAIN_INDEXER_STATE,
        DOMAIN_SWA,
        REUSABLE_LAYER_IDS,
        SnapshotCheckpointCapture,
        SnapshotLayerCapture,
        SnapshotRollbackError as RuntimeSnapshotRollbackError,
    )


SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION = 1


class SGLangSnapshotAdapterError(RuntimeError):
    """The adapter failed closed and attempted the complete rollback."""


class SGLangSnapshotRollbackError(SGLangSnapshotAdapterError):
    """One or more transaction/local-staging rollback actions failed."""

    def __init__(self, failures: Sequence[Tuple[str, BaseException]]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(
            f"{name}: {type(error).__name__}: {error}"
            for name, error in self.failures
        )
        super().__init__(f"SGLang snapshot rollback was incomplete: {detail}")


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha_digest(value: object, name: str) -> str:
    if callable(value):
        value = value()
    result = _nonempty(value, name)
    if not result.startswith("sha256:") or len(result) != 71:
        raise ValueError(f"{name} must be a sha256:<64 hex> digest")
    try:
        int(result[7:], 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a sha256:<64 hex> digest") from error
    return result


def _field(value: object, name: str, *aliases: str) -> object:
    names = (name,) + aliases
    if isinstance(value, Mapping):
        for candidate in names:
            if candidate in value:
                return value[candidate]
    else:
        for candidate in names:
            if hasattr(value, candidate):
                return getattr(value, candidate)
    raise ValueError(f"adapter value is missing {name!r}")


def _optional_field(value: object, name: str, *aliases: str) -> object:
    try:
        return _field(value, name, *aliases)
    except ValueError:
        return None


def _callable(value: object, *names: str):
    for name in names:
        candidate = _optional_field(value, name)
        if callable(candidate):
            return candidate
    raise ValueError(f"adapter API has no callable among {names!r}")


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def _ratio(layer_spec: object) -> int:
    ratio = _strict_int(
        _field(layer_spec, "compress_ratio", "ratio"), "compress ratio"
    )
    if ratio not in (4, 128):
        raise ValueError("reusable SGLang snapshot layers must be C4 or C128")
    return ratio


def _layer_specs(cpu_spec: object) -> Mapping[int, object]:
    layers = tuple(_field(cpu_spec, "layers"))
    result = {
        _strict_int(_field(layer, "layer_id"), "layer_id"): layer
        for layer in layers
    }
    if tuple(sorted(result)) != REUSABLE_LAYER_IDS or len(result) != len(layers):
        raise ValueError("snapshot CPU spec must cover exactly layers 3..39")
    if tuple(_field(cpu_spec, "required_layer_ids")) != REUSABLE_LAYER_IDS:
        raise ValueError("snapshot CPU required layers must be exactly 3..39")
    return MappingProxyType(result)


def _expected_domains(ratio: int) -> Tuple[str, ...]:
    if ratio == 4:
        return (
            DOMAIN_SWA,
            DOMAIN_C4,
            DOMAIN_INDEXER,
            DOMAIN_C4_ATTENTION_STATE,
            DOMAIN_INDEXER_STATE,
        )
    return (DOMAIN_SWA, DOMAIN_C128, DOMAIN_C128_ATTENTION_STATE)


def _bytes(value: object, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be an exported CPU byte buffer")
    return bytes(value)


def _shared_layer_payload_digest(
    *,
    layer_id: int,
    compress_ratio: int,
    cpu_payloads: Mapping[str, bytes],
) -> str:
    """Hash only TP-replicated shared-latent bytes for one layer.

    z_off is rank-local because each attention-TP rank owns different logical
    heads.  SWA/compressed/Indexer/state records are physical shared latent KV
    and must be byte-identical across ranks before a common snapshot is made
    visible.  Length prefixes keep the domain framing unambiguous.
    """

    hasher = sha256()
    hasher.update(b"dsv4-shared-latent-layer-v1\0")
    hasher.update(_strict_int(layer_id, "layer id").to_bytes(4, "little"))
    hasher.update(
        _strict_int(compress_ratio, "compress ratio").to_bytes(4, "little")
    )
    for domain in sorted(cpu_payloads):
        payload = _bytes(cpu_payloads[domain], f"CPU {domain}")
        encoded_domain = str(domain).encode("utf-8")
        hasher.update(len(encoded_domain).to_bytes(4, "little"))
        hasher.update(encoded_domain)
        hasher.update(len(payload).to_bytes(8, "little"))
        hasher.update(payload)
    return "sha256:" + hasher.hexdigest()


def _split_records(
    value: object,
    *,
    record_bytes: int,
    record_count: int,
    name: str,
) -> Tuple[bytes, ...]:
    width = _strict_int(record_bytes, f"{name} record bytes")
    count = _strict_int(record_count, f"{name} record count")
    if width <= 0 or count <= 0:
        raise ValueError(f"{name} record geometry must be positive")
    payload = _bytes(value, name)
    if len(payload) != width * count:
        raise ValueError(
            f"{name} has {len(payload)} bytes; expected {width * count}"
        )
    return tuple(
        payload[index * width : (index + 1) * width]
        for index in range(count)
    )


@dataclass
class _Coverage:
    row_count: int
    spans: list[Tuple[int, int]] = field(default_factory=list)

    def validate(self, begin: int, end: int) -> None:
        if not 0 <= begin < end <= self.row_count:
            raise ValueError("z_off chunk rows are outside the segment")
        for old_begin, old_end in self.spans:
            if begin < old_end and old_begin < end:
                raise ValueError("z_off chunk rows overlap an earlier chunk")

    def add(self, begin: int, end: int) -> None:
        self.validate(begin, end)
        self.spans.append((begin, end))
        self.spans.sort()

    @property
    def complete(self) -> bool:
        cursor = 0
        for begin, end in self.spans:
            if begin != cursor:
                return False
            cursor = end
        return cursor == self.row_count


def _shape(value: object, name: str) -> Tuple[int, ...]:
    raw = _field(value, "shape")
    try:
        result = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} has invalid shape metadata") from error
    return result


def _copy_rows(target: object, source: object, *, non_blocking: bool) -> None:
    copy = _callable(target, "copy_")
    try:
        copy(source, non_blocking=non_blocking)
    except TypeError:
        # Small fake/provisional tensor APIs may not yet expose the keyword.
        copy(source)


@dataclass
class _ZOffLayerStaging:
    layer_id: int
    segment_length: int
    model_hash: str
    policy_hash: str
    coverage: _Coverage
    spec: Optional[object] = None
    local_positions: Optional[object] = None
    local_projection: Optional[object] = None
    position_dtype: str = ""
    position_device: str = ""
    projection_dtype: str = ""
    projection_device: str = ""
    projection_tail: Tuple[int, ...] = ()

    def capture(
        self,
        *,
        row_begin: int,
        z_off_spec: object,
        local_positions: object,
        local_projection: object,
    ) -> Tuple[int, int]:
        position_shape = _shape(local_positions, "z_off local positions")
        projection_shape = _shape(local_projection, "z_off local projection")
        if len(position_shape) != 1 or not projection_shape:
            raise ValueError("z_off chunk tensors have invalid row geometry")
        span = position_shape[0]
        if span <= 0 or projection_shape[0] != span:
            raise ValueError("z_off chunk position/projection rows differ")
        begin = _strict_int(row_begin, "row_begin")
        end = begin + span
        self.coverage.validate(begin, end)

        if _strict_int(_field(z_off_spec, "layer_id"), "z_off layer_id") != self.layer_id:
            raise ValueError("z_off chunk spec belongs to another layer")
        if (
            _field(z_off_spec, "model_compat_hash", "model_hash")
            != self.model_hash
            or _field(z_off_spec, "head_policy_hash", "policy_hash")
            != self.policy_hash
        ):
            raise ValueError("z_off chunk model/head policy changed")

        position_dtype = str(_optional_field(local_positions, "dtype"))
        position_device = str(_optional_field(local_positions, "device"))
        projection_dtype = str(_optional_field(local_projection, "dtype"))
        projection_device = str(_optional_field(local_projection, "device"))
        projection_tail = projection_shape[1:]
        if self.spec is None:
            self.spec = z_off_spec
            self.position_dtype = position_dtype
            self.position_device = position_device
            self.projection_dtype = projection_dtype
            self.projection_device = projection_device
            self.projection_tail = projection_tail
            self.local_positions = _callable(local_positions, "new_empty")(
                (self.segment_length,)
            )
            self.local_projection = _callable(local_projection, "new_empty")(
                (self.segment_length,) + projection_tail
            )
        elif (
            position_dtype != self.position_dtype
            or position_device != self.position_device
            or projection_dtype != self.projection_dtype
            or projection_device != self.projection_device
            or projection_tail != self.projection_tail
        ):
            raise ValueError("z_off chunk tensor layout changed within a layer")

        assert self.local_positions is not None
        assert self.local_projection is not None
        position_target = _callable(self.local_positions, "narrow")(
            0, begin, span
        )
        projection_target = _callable(self.local_projection, "narrow")(
            0, begin, span
        )
        _copy_rows(position_target, local_positions, non_blocking=False)
        _copy_rows(projection_target, local_projection, non_blocking=True)
        self.coverage.add(begin, end)
        return begin, end

    def finalize(self) -> Tuple[object, object, object]:
        if (
            not self.coverage.complete
            or self.spec is None
            or self.local_positions is None
            or self.local_projection is None
        ):
            raise ValueError(f"z_off layer {self.layer_id} capture is incomplete")
        return self.spec, self.local_positions, self.local_projection


class SnapshotCPUControllerKeywordAdapter:
    """Translate runtime ``anchor`` to the production ``anchor_tokens`` API."""

    def __init__(self, controller: object) -> None:
        self.controller = controller
        method = _callable(controller, "capture_checkpoint_states")
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError) as error:
            raise TypeError(
                "CPU shared controller checkpoint signature is not inspectable"
            ) from error
        if "anchor_tokens" in parameters:
            self._anchor_keyword = "anchor_tokens"
        elif "anchor" in parameters:
            self._anchor_keyword = "anchor"
        elif any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            self._anchor_keyword = "anchor_tokens"
        else:
            raise TypeError("CPU shared controller accepts no checkpoint anchor")

    def __getattr__(self, name: str) -> object:
        return getattr(self.controller, name)

    def capture_checkpoint_states(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        layer_id: int,
        anchor: Optional[int] = None,
        anchor_tokens: Optional[int] = None,
        attention_state: object,
        indexer_state: Optional[object] = None,
    ) -> None:
        if anchor is None and anchor_tokens is None:
            raise ValueError("checkpoint anchor is missing")
        if (
            anchor is not None
            and anchor_tokens is not None
            and anchor != anchor_tokens
        ):
            raise ValueError("checkpoint anchor aliases disagree")
        value = anchor if anchor is not None else anchor_tokens
        kwargs = {
            "seg_hash": seg_hash,
            "generation_id": generation_id,
            "layer_id": layer_id,
            self._anchor_keyword: value,
            "attention_state": attention_state,
            "indexer_state": indexer_state,
        }
        _callable(self.controller, "capture_checkpoint_states")(**kwargs)


def ensure_snapshot_cpu_keyword_compat(snapshot_runtime: object) -> object:
    """Install the narrow checkpoint-keyword proxy once and return it."""

    controller = _field(snapshot_runtime, "cpu_shared_controller")
    if isinstance(controller, SnapshotCPUControllerKeywordAdapter):
        return controller
    proxy = SnapshotCPUControllerKeywordAdapter(controller)
    setattr(snapshot_runtime, "cpu_shared_controller", proxy)
    return proxy


@dataclass(frozen=True)
class SGLangSnapshotChunkReceipt:
    session_digest: str
    layer_id: int
    row_begin: int
    row_end: int
    latent_receipt: object = field(repr=False, compare=False)


@dataclass(eq=False)
class SGLangSnapshotSession:
    adapter_token: object = field(repr=False)
    forward_token: str
    session_digest: str
    runtime_bundle: object = field(repr=False)
    cpu_spec: object = field(repr=False)
    token_to_kv_pool: object = field(repr=False)
    layer_specs: Mapping[int, object] = field(repr=False)
    latent_stagings: Dict[int, object] = field(repr=False)
    z_off_stagings: Dict[int, _ZOffLayerStaging] = field(repr=False)
    finalized_bundles: Dict[int, object] = field(default_factory=dict, repr=False)
    local_capture_receipts: Dict[int, object] = field(
        default_factory=dict, repr=False
    )
    shared_latent_layer_digests: Dict[int, str] = field(
        default_factory=dict, repr=False
    )
    shared_latent_domain_digests: Dict[int, Mapping[str, str]] = field(
        default_factory=dict, repr=False
    )
    local_prepared: Optional[object] = field(default=None, repr=False)
    runtime_prepared: Optional[object] = field(default=None, repr=False)
    runtime_published: Optional[object] = field(default=None, repr=False)
    rollback_failures: Tuple[Tuple[str, BaseException], ...] = field(
        default=(), repr=False
    )
    state: str = "capturing"


@dataclass(frozen=True, eq=False)
class LocalPreparedSGLangSnapshot:
    """All 37 layers are staged locally and may enter the one TP vote."""

    adapter_token: object = field(repr=False, compare=False)
    session: SGLangSnapshotSession = field(repr=False, compare=False)
    runtime_identity_digest: str
    layer_capture_digests: Tuple[Tuple[int, str], ...]
    shared_latent_digest: str
    local_prepare_digest: str

    @property
    def digest(self) -> str:
        return self.local_prepare_digest


@dataclass(frozen=True, eq=False)
class PreparedSGLangSnapshot:
    adapter_token: object = field(repr=False, compare=False)
    session: SGLangSnapshotSession = field(repr=False, compare=False)
    local_prepared: LocalPreparedSGLangSnapshot = field(
        repr=False, compare=False
    )
    runtime_prepared: object = field(repr=False, compare=False)
    prepare_digest: str


@dataclass(eq=False)
class PublishedSGLangSnapshot:
    adapter_token: object = field(repr=False)
    prepared: PreparedSGLangSnapshot = field(repr=False)
    runtime_published: object = field(repr=False)
    state: str = "published"


class DSV4SharedSnapshotSGLangAdapter:
    """Backend-facing one-forward/one-segment chunk transaction adapter."""

    def __init__(
        self,
        *,
        snapshot_runtime: object,
        shared_latent_sglang_api: object,
        cpu_component_exporter: Optional[object] = None,
    ) -> None:
        self.snapshot_runtime = snapshot_runtime
        self.shared_latent_sglang_api = shared_latent_sglang_api
        if cpu_component_exporter is not None and not callable(
            cpu_component_exporter
        ):
            raise TypeError("cpu_component_exporter must be callable")
        self.cpu_component_exporter = cpu_component_exporter
        self._begin_layer = _callable(
            shared_latent_sglang_api,
            "begin_layer_capture_bundle",
            "begin_layer_chunk_staging",
        )
        self._capture_chunk = _callable(
            shared_latent_sglang_api,
            "capture_chunk_components",
            "capture_layer_chunk_components",
        )
        finalize = _optional_field(
            shared_latent_sglang_api,
            "finalize_layer_capture_bundle",
            "finalize_layer_chunk_staging",
        )
        self._finalize_layer = finalize if callable(finalize) else None
        ensure_snapshot_cpu_keyword_compat(snapshot_runtime)
        self._token = object()
        self._lock = threading.RLock()
        self._active: Dict[Tuple[str, str, str, str], SGLangSnapshotSession] = {}

    def _key(self, session: SGLangSnapshotSession) -> Tuple[str, str, str, str]:
        identity = _field(session.runtime_bundle, "identity")
        return (
            session.forward_token,
            str(_field(identity, "seg_hash")),
            str(_field(identity, "generation_id")),
            session.session_digest,
        )

    def _validate_session(
        self, session: SGLangSnapshotSession, *states: str
    ) -> None:
        if (
            not isinstance(session, SGLangSnapshotSession)
            or session.adapter_token is not self._token
        ):
            raise ValueError("SGLang snapshot session belongs to another adapter")
        if self._active.get(self._key(session)) is not session:
            raise ValueError("SGLang snapshot session is stale")
        if states and session.state not in states:
            raise ValueError(
                f"SGLang snapshot session state {session.state!r} "
                f"is not one of {states!r}"
            )

    def _abort_local_stagings(
        self, session: SGLangSnapshotSession
    ) -> Tuple[Tuple[str, BaseException], ...]:
        failures = []
        for layer_id, staging in sorted(session.latent_stagings.items()):
            state = _optional_field(staging, "_state", "state")
            if state in ("finalized", "aborted"):
                continue
            abort = _optional_field(staging, "abort")
            if not callable(abort):
                failures.append(
                    (
                        f"latent-layer-{layer_id}",
                        TypeError("chunk staging has no abort method"),
                    )
                )
                continue
            try:
                abort()
            except BaseException as error:
                failures.append((f"latent-layer-{layer_id}", error))
        return tuple(failures)

    def _rollback_session(
        self,
        session: SGLangSnapshotSession,
        *,
        runtime_value: Optional[object] = None,
    ) -> Tuple[Tuple[str, BaseException], ...]:
        if session.state == "rollback_failed":
            return tuple(session.rollback_failures)
        failures = []
        if session.state not in ("rolled_back", "confirmed"):
            value = runtime_value
            if value is None:
                value = (
                    session.runtime_published
                    or session.runtime_prepared
                    or session.runtime_bundle
                )
            try:
                self.snapshot_runtime.rollback(value)
            except BaseException as error:
                # Runtime rollback marks its bundle terminal before reporting
                # participant failures.  Never use that state bit to suppress
                # the exception: it may mean CPU/GPU/z_off rollback was only
                # partially successful.  A genuinely idempotent second call
                # returns normally and does not reach this branch.
                failures.append(("snapshot-runtime", error))
        failures.extend(self._abort_local_stagings(session))
        session.finalized_bundles.clear()
        session.latent_stagings.clear()
        session.z_off_stagings.clear()
        session.rollback_failures = tuple(failures)
        session.state = "rollback_failed" if failures else "rolled_back"
        with self._lock:
            self._active.pop(self._key(session), None)
        return tuple(failures)

    def _fail_session(
        self, session: SGLangSnapshotSession, error: BaseException
    ) -> None:
        failures = self._rollback_session(session)
        if isinstance(error, SGLangSnapshotRollbackError):
            failures = tuple(error.failures) + tuple(failures)
        elif isinstance(error, RuntimeSnapshotRollbackError):
            # The runtime can mark its bundle terminal and still report that a
            # CPU/GPU/z_off participant failed to roll back.  A second adapter
            # rollback is then idempotent; preserve the original incomplete
            # transaction signal instead of misreporting full cleanup.
            failures = (("snapshot-runtime", error),) + tuple(failures)
        if failures:
            # Persist an incomplete rollback as a terminal, repeatable signal.
            # Backend cleanup may call rollback a second time after a TP vote;
            # returning success there would falsely certify a healthy epoch.
            session.rollback_failures = tuple(failures)
            session.state = "rollback_failed"
            with self._lock:
                self._active.pop(self._key(session), None)
            raise SGLangSnapshotRollbackError(failures) from error
        raise SGLangSnapshotAdapterError(
            "SGLang snapshot failed closed and was rolled back"
        ) from error

    def begin_segment(
        self,
        *,
        forward_token: str,
        seg_hash: str,
        generation_id: str,
        token_hash: str,
        token_ids: Sequence[int],
        model_hash: str,
        policy_hash: str,
        cpu_spec: object,
        token_to_kv_pool: object,
        z_off_expected_bytes: int,
        z_off_token_positions: object,
        z_off_token_ids: object,
        z_off_expected_device_bytes: int = 0,
        latent_staging_device: Optional[object] = None,
    ) -> SGLangSnapshotSession:
        """Begin the unified runtime and all 37 overwrite-safe layer stages."""

        forward_token = _nonempty(forward_token, "forward_token")
        tokens = tuple(_strict_int(token, "token id") for token in token_ids)
        layer_specs = _layer_specs(cpu_spec)
        length = _strict_int(_field(cpu_spec, "length"), "segment length")
        if length != len(tokens) or length <= 0:
            raise ValueError("CPU spec/token ids do not cover one segment")
        if _field(cpu_spec, "model_hash") != model_hash:
            raise ValueError("CPU spec model hash differs from adapter begin")
        if _field(cpu_spec, "policy_hash") != policy_hash:
            raise ValueError("CPU spec policy hash differs from adapter begin")
        identity_payload = {
            "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
            "forward_token": forward_token,
            "seg_hash": seg_hash,
            "generation_id": generation_id,
            "token_hash": token_hash,
            "model_hash": model_hash,
            "policy_hash": policy_hash,
            "token_count": len(tokens),
            "layers": tuple(
                (layer_id, _ratio(layer_specs[layer_id]))
                for layer_id in REUSABLE_LAYER_IDS
            ),
        }
        session_digest = _digest(identity_payload)
        key = (forward_token, str(seg_hash), str(generation_id), session_digest)

        with self._lock:
            if key in self._active:
                raise ValueError("this forward/segment snapshot is already active")

            runtime_bundle = None
            latent_stagings: Dict[int, object] = {}
            try:
                runtime_bundle = self.snapshot_runtime.begin(
                    seg_hash=seg_hash,
                    generation_id=generation_id,
                    token_hash=token_hash,
                    token_ids=tokens,
                    model_hash=model_hash,
                    policy_hash=policy_hash,
                    cpu_spec=cpu_spec,
                    z_off_expected_bytes=z_off_expected_bytes,
                    z_off_token_positions=z_off_token_positions,
                    z_off_token_ids=z_off_token_ids,
                    z_off_expected_device_bytes=z_off_expected_device_bytes,
                )
                for layer_id in REUSABLE_LAYER_IDS:
                    kwargs = {
                        "token_to_kv_pool": token_to_kv_pool,
                        "layer_id": layer_id,
                        "compress_ratio": _ratio(layer_specs[layer_id]),
                        "segment_length": length,
                    }
                    if latent_staging_device is not None:
                        kwargs["device"] = latent_staging_device
                    latent_stagings[layer_id] = self._begin_layer(**kwargs)
            except BaseException as error:
                temporary = SGLangSnapshotSession(
                    adapter_token=self._token,
                    forward_token=forward_token,
                    session_digest=session_digest,
                    runtime_bundle=runtime_bundle,
                    cpu_spec=cpu_spec,
                    token_to_kv_pool=token_to_kv_pool,
                    layer_specs=layer_specs,
                    latent_stagings=latent_stagings,
                    z_off_stagings={},
                )
                failures = []
                if isinstance(error, RuntimeSnapshotRollbackError):
                    # begin() may already have attempted and incompletely
                    # rolled back its participants before it can return a
                    # bundle handle.  Preserve that terminal failure even
                    # though there is no runtime_bundle to roll back again.
                    failures.append(("snapshot-runtime", error))
                if runtime_bundle is not None:
                    try:
                        self.snapshot_runtime.rollback(runtime_bundle)
                    except BaseException as rollback_error:
                        failures.append(("snapshot-runtime", rollback_error))
                failures.extend(self._abort_local_stagings(temporary))
                if failures:
                    raise SGLangSnapshotRollbackError(failures) from error
                raise SGLangSnapshotAdapterError(
                    "SGLang snapshot begin failed and was rolled back"
                ) from error

            session = SGLangSnapshotSession(
                adapter_token=self._token,
                forward_token=forward_token,
                session_digest=session_digest,
                runtime_bundle=runtime_bundle,
                cpu_spec=cpu_spec,
                token_to_kv_pool=token_to_kv_pool,
                layer_specs=layer_specs,
                latent_stagings=latent_stagings,
                z_off_stagings={
                    layer_id: _ZOffLayerStaging(
                        layer_id=layer_id,
                        segment_length=length,
                        model_hash=str(model_hash),
                        policy_hash=str(policy_hash),
                        coverage=_Coverage(length),
                    )
                    for layer_id in REUSABLE_LAYER_IDS
                },
            )
            self._active[key] = session
            return session

    def capture_chunk(
        self,
        session: SGLangSnapshotSession,
        *,
        layer_id: int,
        row_begin: int,
        z_off_spec: object,
        local_positions: object,
        local_projection: object,
        latent_chunk: Mapping[str, object],
    ) -> SGLangSnapshotChunkReceipt:
        """Capture one layer's z_off and shared-latent rows for one chunk."""

        with self._lock:
            self._validate_session(session, "capturing")
        try:
            layer_id = _strict_int(layer_id, "layer_id")
            if layer_id not in REUSABLE_LAYER_IDS:
                raise ValueError("chunk capture may target only layers 3..39")
            if not isinstance(latent_chunk, Mapping):
                raise TypeError("latent_chunk must be a keyword mapping")
            forbidden = {"staging", "token_to_kv_pool", "row_begin"} & set(
                latent_chunk
            )
            if forbidden:
                raise ValueError(
                    "adapter-owned latent chunk fields were supplied: "
                    f"{sorted(forbidden)}"
                )
            begin, end = session.z_off_stagings[layer_id].capture(
                row_begin=row_begin,
                z_off_spec=z_off_spec,
                local_positions=local_positions,
                local_projection=local_projection,
            )
            latent_receipt = self._capture_chunk(
                session.latent_stagings[layer_id],
                token_to_kv_pool=session.token_to_kv_pool,
                row_begin=begin,
                **dict(latent_chunk),
            )
            if (
                _strict_int(_field(latent_receipt, "layer_id"), "receipt layer_id")
                != layer_id
                or _strict_int(
                    _field(latent_receipt, "swa_row_begin"),
                    "receipt swa_row_begin",
                )
                != begin
                or _strict_int(
                    _field(latent_receipt, "swa_row_end"),
                    "receipt swa_row_end",
                )
                != end
            ):
                raise ValueError("latent/z_off chunk receipts disagree")
            return SGLangSnapshotChunkReceipt(
                session_digest=session.session_digest,
                layer_id=layer_id,
                row_begin=begin,
                row_end=end,
                latent_receipt=latent_receipt,
            )
        except BaseException as error:
            self._fail_session(session, error)
        raise AssertionError("unreachable")

    def _finalize_latent(self, staging: object) -> object:
        if self._finalize_layer is not None:
            return self._finalize_layer(staging)
        return _callable(staging, "finalize")()

    def _export_cpu_components(self, bundle: object) -> Mapping[str, object]:
        if self.cpu_component_exporter is not None:
            exported = self.cpu_component_exporter(bundle)
        else:
            exported = _callable(bundle, "export_cpu_components")()
        if not isinstance(exported, Mapping):
            raise TypeError("CPU component exporter must return a Mapping")
        return exported

    def _make_layer_capture(
        self,
        session: SGLangSnapshotSession,
        *,
        layer_id: int,
        bundle: object,
        gpu_stream: Optional[object],
        gpu_non_blocking: bool,
    ) -> Tuple[SnapshotLayerCapture, str, Mapping[str, str]]:
        layer_spec = session.layer_specs[layer_id]
        ratio = _ratio(layer_spec)
        if (
            _strict_int(_field(bundle, "layer_id"), "bundle layer_id")
            != layer_id
            or _strict_int(
                _field(bundle, "compress_ratio", "ratio"), "bundle ratio"
            )
            != ratio
        ):
            raise ValueError("finalized latent bundle belongs to another layer")
        components = _field(bundle, "components")
        if not isinstance(components, Mapping):
            raise TypeError("finalized latent components must be a Mapping")
        expected_domains = _expected_domains(ratio)
        if tuple(sorted(components)) != tuple(sorted(expected_domains)):
            raise ValueError("finalized GPU latent domains are incomplete")
        if any(components[domain] is None for domain in expected_domains):
            raise ValueError("finalized GPU latent domain payload is absent")

        cpu_components = self._export_cpu_components(bundle)
        if tuple(sorted(cpu_components)) != tuple(sorted(expected_domains)):
            raise ValueError("exported CPU latent domains are incomplete")
        cpu_payloads = {
            domain: _bytes(cpu_components[domain], f"CPU {domain}")
            for domain in expected_domains
        }
        checkpoints = tuple(_field(session.cpu_spec, "checkpoint_anchors"))
        state_count = len(checkpoints) + 1
        attention_domain = (
            DOMAIN_C4_ATTENTION_STATE
            if ratio == 4
            else DOMAIN_C128_ATTENTION_STATE
        )
        attention_records = _split_records(
            cpu_payloads[attention_domain],
            record_bytes=_strict_int(
                _field(layer_spec, "attention_terminal_state_bytes"),
                "attention state bytes",
            ),
            record_count=state_count,
            name=f"layer {layer_id} attention states",
        )
        indexer_records: Tuple[bytes, ...] = ()
        if ratio == 4:
            indexer_records = _split_records(
                cpu_payloads[DOMAIN_INDEXER_STATE],
                record_bytes=_strict_int(
                    _field(layer_spec, "indexer_terminal_state_bytes"),
                    "Indexer state bytes",
                ),
                record_count=state_count,
                name=f"layer {layer_id} Indexer states",
            )
        checkpoint_captures = tuple(
            SnapshotCheckpointCapture(
                anchor=_strict_int(anchor, "checkpoint anchor"),
                attention_state=attention_records[index],
                indexer_state=(indexer_records[index] if ratio == 4 else None),
            )
            for index, anchor in enumerate(checkpoints)
        )
        z_off_spec, local_positions, local_projection = session.z_off_stagings[
            layer_id
        ].finalize()
        length = _strict_int(_field(session.cpu_spec, "length"), "segment length")
        compressed_domain = DOMAIN_C4 if ratio == 4 else DOMAIN_C128
        capture = SnapshotLayerCapture(
            z_off_spec=z_off_spec,
            local_positions=local_positions,
            local_projection=local_projection,
            swa_rows=tuple(range(length)),
            swa_positionless_packed=cpu_payloads[DOMAIN_SWA],
            compressed_blocks=tuple(range(length // ratio)),
            compressed_positionless_packed=cpu_payloads[compressed_domain],
            attention_terminal_state=attention_records[-1],
            checkpoints=checkpoint_captures,
            gpu_components=MappingProxyType(dict(components)),
            indexer_blocks=(tuple(range(length // ratio)) if ratio == 4 else ()),
            indexer_positionless_keys=(
                cpu_payloads[DOMAIN_INDEXER] if ratio == 4 else None
            ),
            indexer_position_semantics=(
                str(_field(session.cpu_spec, "indexer_position_semantics"))
                if ratio == 4
                else ""
            ),
            indexer_terminal_state=(
                indexer_records[-1] if ratio == 4 else None
            ),
            gpu_stream=gpu_stream,
            gpu_non_blocking=gpu_non_blocking,
        )
        shared_digest = _shared_layer_payload_digest(
            layer_id=layer_id,
            compress_ratio=ratio,
            cpu_payloads=cpu_payloads,
        )
        domain_digests = MappingProxyType(
            {
                domain: "sha256:" + sha256(cpu_payloads[domain]).hexdigest()
                for domain in sorted(cpu_payloads)
            }
        )
        return capture, shared_digest, domain_digests

    def prepare_local(
        self,
        session: SGLangSnapshotSession,
        *,
        gpu_stream: Optional[object] = None,
        gpu_non_blocking: bool = True,
    ) -> LocalPreparedSGLangSnapshot:
        """Stage all 37 layers locally and return the object to TP-vote on.

        This method performs no TP collective and does not call the runtime's
        ``prepare_publish``.  Production must bind ``local_prepare_digest`` in
        its single readiness vote, then pass the resulting certificate to
        :meth:`prepare_publish`.
        """

        with self._lock:
            self._validate_session(session, "capturing", "local_prepared")
            if session.state == "local_prepared":
                if not isinstance(
                    session.local_prepared, LocalPreparedSGLangSnapshot
                ):
                    raise ValueError("local-prepared session lost its handle")
                return session.local_prepared
        try:
            if type(gpu_non_blocking) is not bool:
                raise TypeError("gpu_non_blocking must be boolean")
            # Finalize/export every layer first.  Incomplete local staging can
            # therefore never partially populate the runtime participants.
            captures = {}
            shared_layer_digests = {}
            shared_domain_digests = {}
            for layer_id in REUSABLE_LAYER_IDS:
                session.z_off_stagings[layer_id].finalize()
                bundle = self._finalize_latent(session.latent_stagings[layer_id])
                session.finalized_bundles[layer_id] = bundle
                capture, shared_digest, domain_digests = self._make_layer_capture(
                    session,
                    layer_id=layer_id,
                    bundle=bundle,
                    gpu_stream=gpu_stream,
                    gpu_non_blocking=gpu_non_blocking,
                )
                captures[layer_id] = capture
                shared_layer_digests[layer_id] = _sha_digest(
                    shared_digest, "shared-latent layer digest"
                )
                shared_domain_digests[layer_id] = domain_digests
            capture_receipts = {}
            for layer_id in REUSABLE_LAYER_IDS:
                receipt = self.snapshot_runtime.capture_layer(
                    session.runtime_bundle,
                    layer_id=layer_id,
                    capture=captures[layer_id],
                )
                if (
                    _strict_int(
                        _field(receipt, "layer_id"), "capture receipt layer_id"
                    )
                    != layer_id
                ):
                    raise ValueError("runtime capture receipt changed layer identity")
                _sha_digest(
                    _field(receipt, "capture_digest"),
                    "runtime layer capture digest",
                )
                capture_receipts[layer_id] = receipt
            session.local_capture_receipts.update(capture_receipts)
            session.shared_latent_layer_digests.update(shared_layer_digests)
            session.shared_latent_domain_digests.update(shared_domain_digests)
            identity = _field(session.runtime_bundle, "identity")
            runtime_identity_digest = _sha_digest(
                _field(identity, "digest"), "runtime snapshot identity digest"
            )
            layer_capture_digests = tuple(
                (
                    layer_id,
                    _sha_digest(
                        _field(capture_receipts[layer_id], "capture_digest"),
                        "runtime layer capture digest",
                    ),
                )
                for layer_id in REUSABLE_LAYER_IDS
            )
            shared_latent_digest = _digest(
                {
                    "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
                    "session": session.session_digest,
                    "layers": tuple(
                        (
                            layer_id,
                            session.shared_latent_layer_digests[layer_id],
                        )
                        for layer_id in REUSABLE_LAYER_IDS
                    ),
                }
            )
            local_prepare_digest = _digest(
                {
                    "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
                    "session": session.session_digest,
                    "runtime_identity": runtime_identity_digest,
                    "layer_captures": layer_capture_digests,
                    "shared_latent": shared_latent_digest,
                }
            )
            local_prepared = LocalPreparedSGLangSnapshot(
                adapter_token=self._token,
                session=session,
                runtime_identity_digest=runtime_identity_digest,
                layer_capture_digests=layer_capture_digests,
                shared_latent_digest=shared_latent_digest,
                local_prepare_digest=local_prepare_digest,
            )
            session.local_prepared = local_prepared
            session.state = "local_prepared"
            # Runtime participants now own CPU bytes, z_off copies, and GPU
            # upload references.  Drop duplicate adapter buffers promptly.
            session.finalized_bundles.clear()
            session.latent_stagings.clear()
            session.z_off_stagings.clear()
            return local_prepared
        except BaseException as error:
            self._fail_session(session, error)
        raise AssertionError("unreachable")

    def _validate_local_prepared(
        self, local_prepared: LocalPreparedSGLangSnapshot
    ) -> SGLangSnapshotSession:
        if (
            not isinstance(local_prepared, LocalPreparedSGLangSnapshot)
            or local_prepared.adapter_token is not self._token
        ):
            raise ValueError("local-prepared snapshot belongs to another adapter")
        session = local_prepared.session
        self._validate_session(session, "local_prepared")
        if session.local_prepared is not local_prepared:
            raise ValueError("local-prepared snapshot handle changed")
        identity = _field(session.runtime_bundle, "identity")
        runtime_identity_digest = _sha_digest(
            _field(identity, "digest"), "runtime snapshot identity digest"
        )
        current_layer_digests = tuple(
            (
                layer_id,
                _sha_digest(
                    _field(
                        session.local_capture_receipts[layer_id],
                        "capture_digest",
                    ),
                    "runtime layer capture digest",
                ),
            )
            for layer_id in REUSABLE_LAYER_IDS
        )
        current_shared_latent_digest = _digest(
            {
                "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
                "session": session.session_digest,
                "layers": tuple(
                    (
                        layer_id,
                        _sha_digest(
                            session.shared_latent_layer_digests[layer_id],
                            "shared-latent layer digest",
                        ),
                    )
                    for layer_id in REUSABLE_LAYER_IDS
                ),
            }
        )
        expected = _digest(
            {
                "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
                "session": session.session_digest,
                "runtime_identity": runtime_identity_digest,
                "layer_captures": current_layer_digests,
                "shared_latent": current_shared_latent_digest,
            }
        )
        if (
            local_prepared.runtime_identity_digest != runtime_identity_digest
            or local_prepared.layer_capture_digests != current_layer_digests
            or local_prepared.shared_latent_digest
            != current_shared_latent_digest
            or local_prepared.local_prepare_digest != expected
        ):
            raise ValueError("local-prepared snapshot digest changed")
        return session

    def _validate_local_vote_certificate(
        self,
        local_prepared: LocalPreparedSGLangSnapshot,
        tp_certificate: object,
    ) -> None:
        validator = _optional_field(
            tp_certificate, "validate_snapshot_local_prepare"
        )
        if callable(validator):
            result = validator(local_prepared)
            if result is False:
                raise ValueError("TP certificate rejected local snapshot staging")
            return
        certified_digest = _sha_digest(
            _field(tp_certificate, "snapshot_local_prepare_digest"),
            "certificate snapshot_local_prepare_digest",
        )
        if certified_digest != local_prepared.local_prepare_digest:
            raise ValueError("TP certificate binds another local snapshot staging")

    def prepare_publish(
        self,
        value: Union[SGLangSnapshotSession, LocalPreparedSGLangSnapshot],
        *,
        tp_certificate: object,
        synchronize_gpu: bool = True,
        gpu_stream: Optional[object] = None,
        gpu_non_blocking: bool = True,
    ) -> PreparedSGLangSnapshot:
        """Seal an externally certified local staging generation.

        Passing a session remains a convenience wrapper for tests/legacy
        callers; production should call :meth:`prepare_local`, perform its one
        TP readiness vote over ``local_prepared.digest``, then pass that handle
        here.
        """

        if isinstance(value, SGLangSnapshotSession):
            local_prepared = self.prepare_local(
                value,
                gpu_stream=gpu_stream,
                gpu_non_blocking=gpu_non_blocking,
            )
        else:
            local_prepared = value
        if (
            not isinstance(local_prepared, LocalPreparedSGLangSnapshot)
            or local_prepared.adapter_token is not self._token
        ):
            raise ValueError("local-prepared snapshot belongs to another adapter")
        session = local_prepared.session
        with self._lock:
            self._validate_session(session, "local_prepared")
        try:
            with self._lock:
                self._validate_local_prepared(local_prepared)
            self._validate_local_vote_certificate(
                local_prepared, tp_certificate
            )
            runtime_prepared = self.snapshot_runtime.prepare_publish(
                session.runtime_bundle,
                tp_certificate=tp_certificate,
                synchronize_gpu=bool(synchronize_gpu),
            )
            session.runtime_prepared = runtime_prepared
            session.state = "prepared"
            prepare_digest = _digest(
                {
                    "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
                    "local_prepare": local_prepared.local_prepare_digest,
                    "runtime_prepare": str(
                        _field(runtime_prepared, "prepare_digest")
                    ),
                }
            )
            return PreparedSGLangSnapshot(
                adapter_token=self._token,
                session=session,
                local_prepared=local_prepared,
                runtime_prepared=runtime_prepared,
                prepare_digest=prepare_digest,
            )
        except Exception as error:
            self._fail_session(session, error)
        raise AssertionError("unreachable")

    def _validate_prepared(
        self, prepared: PreparedSGLangSnapshot
    ) -> SGLangSnapshotSession:
        if (
            not isinstance(prepared, PreparedSGLangSnapshot)
            or prepared.adapter_token is not self._token
        ):
            raise ValueError("prepared SGLang snapshot belongs to another adapter")
        session = prepared.session
        self._validate_session(session, "prepared")
        if session.runtime_prepared is not prepared.runtime_prepared:
            raise ValueError("prepared runtime snapshot handle changed")
        if session.local_prepared is not prepared.local_prepared:
            raise ValueError("prepared local snapshot handle changed")
        expected = _digest(
            {
                "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
                "local_prepare": prepared.local_prepared.local_prepare_digest,
                "runtime_prepare": str(
                    _field(prepared.runtime_prepared, "prepare_digest")
                ),
            }
        )
        if prepared.prepare_digest != expected:
            raise ValueError("prepared SGLang snapshot digest changed")
        return session

    def publish(
        self, prepared: PreparedSGLangSnapshot
    ) -> PublishedSGLangSnapshot:
        """Publish the runtime's rollback-capable three-store generation."""

        if (
            not isinstance(prepared, PreparedSGLangSnapshot)
            or prepared.adapter_token is not self._token
        ):
            raise ValueError("prepared SGLang snapshot belongs to another adapter")
        session = prepared.session
        with self._lock:
            self._validate_session(session, "prepared")
        try:
            with self._lock:
                self._validate_prepared(prepared)
            runtime_published = self.snapshot_runtime.publish(
                prepared.runtime_prepared
            )
            session.runtime_published = runtime_published
            session.state = "published"
            return PublishedSGLangSnapshot(
                adapter_token=self._token,
                prepared=prepared,
                runtime_published=runtime_published,
            )
        except Exception as error:
            self._fail_session(session, error)
        raise AssertionError("unreachable")

    def _validate_participant_confirmation(
        self,
        session: SGLangSnapshotSession,
        runtime_published: object,
    ) -> None:
        """Public-participant fallback when runtime has no public validator."""

        if _field(runtime_published, "state") != "published":
            raise ValueError("runtime publish handle is not pending")
        runtime_prepared = _field(runtime_published, "prepared")
        if _field(runtime_prepared, "bundle") is not session.runtime_bundle:
            raise ValueError("runtime publish belongs to another snapshot bundle")
        identity = _field(session.runtime_bundle, "identity")
        tp_binding = _field(runtime_prepared, "tp_certificate")
        _callable(tp_binding, "validate")(identity)

        z_off_receipt = _field(runtime_published, "z_off_receipt")
        gpu_receipt = _field(runtime_published, "gpu_receipt")
        cpu_receipt = _field(runtime_published, "cpu_receipt")
        _callable(
            _field(self.snapshot_runtime, "z_off_controller"),
            "validate_publish_confirmation",
        )(z_off_receipt)
        _callable(
            _field(self.snapshot_runtime, "gpu_shared_store"),
            "validate_publish_confirmation",
        )(gpu_receipt)
        if _field(cpu_receipt, "state") != "published":
            raise ValueError("CPU shared-latent publish receipt is not pending")
        artifact = _field(cpu_receipt, "artifact")
        seg_hash = str(_field(identity, "seg_hash"))
        committed = _callable(
            _field(self.snapshot_runtime, "cpu_shared_controller"),
            "get_committed",
        )(seg_hash)
        if committed is not artifact:
            raise ValueError("CPU shared-latent artifact is no longer current")

    def validate_confirmation(
        self, published: PublishedSGLangSnapshot
    ) -> str:
        """Validate all pending receipts without changing visible state.

        The returned digest may be bound into the backend's final all-TP vote.
        The caller must invoke :meth:`rollback` on every rank if that vote is
        not unanimous; only a unanimous vote may proceed to :meth:`confirm`.
        """

        if (
            not isinstance(published, PublishedSGLangSnapshot)
            or published.adapter_token is not self._token
            or published.state != "published"
        ):
            raise ValueError("published SGLang snapshot is stale or foreign")
        session = published.prepared.session
        with self._lock:
            self._validate_session(session, "published")
        if session.runtime_published is not published.runtime_published:
            raise ValueError("runtime published snapshot handle changed")
        validator = _optional_field(
            self.snapshot_runtime, "validate_confirmation"
        )
        if callable(validator):
            validator(published.runtime_published)
        else:
            self._validate_participant_confirmation(
                session, published.runtime_published
            )
        runtime_publish_digest = _sha_digest(
            _field(published.runtime_published, "publish_digest"),
            "runtime publish digest",
        )
        return _digest(
            {
                "format_version": SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION,
                "session": session.session_digest,
                "prepare": published.prepared.prepare_digest,
                "runtime_publish": runtime_publish_digest,
            }
        )

    def confirm(self, published: PublishedSGLangSnapshot) -> None:
        """Confirm CPU, z_off, and finally the GPU visibility gate."""

        if (
            not isinstance(published, PublishedSGLangSnapshot)
            or published.adapter_token is not self._token
            or published.state != "published"
        ):
            raise ValueError("published SGLang snapshot is stale or foreign")
        session = published.prepared.session
        with self._lock:
            self._validate_session(session, "published")
        try:
            self.validate_confirmation(published)
            self.snapshot_runtime.confirm(published.runtime_published)
        except BaseException as error:
            bundle_state = _optional_field(session.runtime_bundle, "state")
            if bundle_state in ("capturing", "prepared", "published"):
                failures = self._rollback_session(
                    session, runtime_value=published.runtime_published
                )
                if failures:
                    raise SGLangSnapshotRollbackError(failures) from error
            elif isinstance(error, RuntimeSnapshotRollbackError):
                failures = (("snapshot-runtime", error),)
                session.rollback_failures = failures
                session.state = "rollback_failed"
                with self._lock:
                    self._active.pop(self._key(session), None)
                raise SGLangSnapshotRollbackError(failures) from error
            else:
                session.state = (
                    "rolled_back" if bundle_state == "rolled_back" else "failed"
                )
                with self._lock:
                    self._active.pop(self._key(session), None)
            raise SGLangSnapshotAdapterError(
                "SGLang snapshot confirmation failed closed"
            ) from error
        published.state = "confirmed"
        session.state = "confirmed"
        with self._lock:
            self._active.pop(self._key(session), None)

    def rollback(
        self,
        value: Union[
            SGLangSnapshotSession,
            LocalPreparedSGLangSnapshot,
            PreparedSGLangSnapshot,
            PublishedSGLangSnapshot,
        ],
    ) -> None:
        """Idempotently rollback any pre-confirm adapter handle."""

        if isinstance(value, PublishedSGLangSnapshot):
            if value.adapter_token is not self._token:
                raise ValueError("published SGLang snapshot is foreign")
            session = value.prepared.session
            runtime_value = value.runtime_published
        elif isinstance(value, PreparedSGLangSnapshot):
            if value.adapter_token is not self._token:
                raise ValueError("prepared SGLang snapshot is foreign")
            session = value.session
            runtime_value = value.runtime_prepared
        elif isinstance(value, LocalPreparedSGLangSnapshot):
            if value.adapter_token is not self._token:
                raise ValueError("local-prepared SGLang snapshot is foreign")
            session = value.session
            runtime_value = value.session.runtime_bundle
        else:
            session = value
            runtime_value = None
        if (
            not isinstance(session, SGLangSnapshotSession)
            or session.adapter_token is not self._token
        ):
            raise ValueError("SGLang snapshot session is foreign")
        if session.state == "rolled_back":
            return
        if session.state == "rollback_failed":
            raise SGLangSnapshotRollbackError(session.rollback_failures)
        if session.state == "confirmed":
            raise ValueError("confirmed SGLang snapshot cannot be rolled back")
        failures = self._rollback_session(session, runtime_value=runtime_value)
        if isinstance(value, PublishedSGLangSnapshot):
            value.state = "rolled_back"
        if failures:
            raise SGLangSnapshotRollbackError(failures)


__all__ = [
    "DSV4SharedSnapshotSGLangAdapter",
    "LocalPreparedSGLangSnapshot",
    "PreparedSGLangSnapshot",
    "PublishedSGLangSnapshot",
    "SGLANG_SNAPSHOT_ADAPTER_FORMAT_VERSION",
    "SGLangSnapshotAdapterError",
    "SGLangSnapshotChunkReceipt",
    "SGLangSnapshotRollbackError",
    "SGLangSnapshotSession",
    "SnapshotCPUControllerKeywordAdapter",
    "ensure_snapshot_cpu_keyword_compat",
]
