"""Streaming offline KV snapshots, with an explicit full-prefill commit boundary.

This module neither imports a model runtime nor launches inference. A caller may
supply an isolated model and driver to ``capture_snapshot``. CPU fixtures are
labelled as references; capture provenance is not an accuracy certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from types import MappingProxyType

from .cache import CacheArtifact
from .core import (CacheIdentity, ValidationError, canonical_bytes, digest,
                   integer, normalize_rows, require, token_digest, validate_digest)


SCHEMA = "kv-cache-engram-streaming-snapshot-v1"
DEFAULT_MAX_PAYLOAD_BYTES = 2 * 1024 ** 3
MAX_MANIFEST_BYTES = 16 * 1024 ** 2
MAX_CHUNK_VALUES = 8_388_608
ENCODER_LAYERS = tuple(range(3, 8)) + tuple(range(9, 14)) + tuple(range(15, 19))
CAPTURE_POINT = "V4Attention._prepare_swa_kv:projected-input/native-norm-only"
_OPERATORS = {
    "python-float64-rmsnorm-reference": (False, False),
    "torch-cpu-float32-rmsnorm-reference": (False, False),
    "deepinfer-native-fused-rmsnorm-only": (True, True),
    "deepinfer-norm-dependency-fixture": (False, False),
}


@dataclass(frozen=True)
class CaptureSpec:
    request_id: str
    phase: str
    model_sha256: str
    tokenizer_sha256: str
    execution_sha256: str
    input_ids: tuple
    document_start: int
    document_end: int
    kind: str = "independent_document"
    layers: tuple = ENCODER_LAYERS
    width: int = 512

    def __post_init__(self):
        for name in ("request_id", "phase"):
            require(type(getattr(self, name)) is str and bool(getattr(self, name).strip()),
                    name + " required")
        for name in ("model_sha256", "tokenizer_sha256", "execution_sha256"):
            validate_digest(getattr(self, name), name)
        require(type(self.input_ids) in (tuple, list), "complete CPU input IDs required")
        ids = tuple(self.input_ids)
        require(0 < len(ids) <= 4_194_304, "input token count outside CPU capture bound")
        token_digest(ids)
        object.__setattr__(self, "input_ids", ids)
        integer(self.document_start, "document_start")
        integer(self.document_end, "document_end", 1)
        require(self.document_start < self.document_end <= len(ids), "document outside input")
        require(self.kind in ("independent_document", "full_context_reference"), "unknown capture kind")
        if self.kind == "independent_document":
            require(self.document_start == 0 and self.document_end == len(ids),
                    "independent document must be the entire position-zero input")
        require(type(self.layers) in (tuple, list), "explicit layer sequence required")
        layers = tuple(self.layers)
        require(bool(layers) and all(type(x) is int for x in layers)
                and layers == tuple(sorted(set(layers)))
                and set(layers) <= set(ENCODER_LAYERS), "unsupported or duplicate capture layers")
        object.__setattr__(self, "layers", layers)
        integer(self.width, "width", 1)
        require(self.width <= 65_536, "capture width exceeds CPU reference bound")

    @property
    def document_ids(self):
        return self.input_ids[self.document_start:self.document_end]

    @property
    def request_sha256(self):
        """Exact full-input digest; request ID and phase are bound separately."""
        return token_digest(self.input_ids)

    @property
    def identities(self):
        return MappingProxyType({layer: CacheIdentity(
            self.model_sha256, self.tokenizer_sha256, self.execution_sha256,
            token_digest(self.document_ids), layer, len(self.document_ids), self.width)
            for layer in self.layers})

    def metadata(self):
        fields = asdict(self)
        fields["input_ids"] = list(self.input_ids)
        fields["layers"] = list(self.layers)
        fields["input_ids_sha256"] = self.request_sha256
        fields["document_tokens_sha256"] = token_digest(self.document_ids)
        fields["identities"] = {str(k): asdict(v) for k, v in self.identities.items()}
        return fields


def _spec_from_metadata(value):
    require(type(value) is dict and set(value) == set(CaptureSpec.__dataclass_fields__) | {
        "input_ids_sha256", "document_tokens_sha256", "identities"}, "unexpected capture spec fields")
    spec = CaptureSpec(**{k: value[k] for k in CaptureSpec.__dataclass_fields__})
    require(type(value["identities"]) is dict, "identity mapping required")
    for identity in value["identities"].values():
        require(type(identity) is dict and set(identity) == set(CacheIdentity.__dataclass_fields__),
                "unexpected identity fields")
        CacheIdentity(**identity)
    require(value == spec.metadata(), "capture spec derived identities disagree")
    return spec


def _provenance(value, spec):
    fields = {"adapter", "capture_point", "request_id", "phase", "input_ids_sha256",
              "input_token_count", "document_start", "document_end", "operator",
              "native_operator", "accelerator_capture", "streaming", "chunks",
              "context_equivalence_certified", "native_numerical_alignment_measured",
              "identity_sha256_by_layer"}
    require(type(value) is dict and set(value) == fields, "unexpected capture provenance fields")
    require(value["adapter"] == "deepinfer-v4-swa-capture-v1"
            and value["capture_point"] == CAPTURE_POINT, "unsupported capture adapter or point")
    for name, expected in (("request_id", spec.request_id), ("phase", spec.phase),
                           ("input_ids_sha256", spec.request_sha256),
                           ("document_start", spec.document_start), ("document_end", spec.document_end),
                           ("input_token_count", len(spec.input_ids))):
        require(type(value[name]) is type(expected) and value[name] == expected,
                "foreign capture provenance: " + name)
    require(type(value["operator"]) is str and value["operator"] in _OPERATORS,
            "unknown capture operator")
    native, accelerator = _OPERATORS[value["operator"]]
    require(value["native_operator"] is native and value["accelerator_capture"] is accelerator,
            "capture operator origin flags disagree")
    require(value["streaming"] is True and value["context_equivalence_certified"] is False
            and value["native_numerical_alignment_measured"] is False,
            "unsupported capture qualification or nonstreaming origin")
    require(value["identity_sha256_by_layer"] == {str(k): v.sha256 for k, v in spec.identities.items()},
            "foreign capture layer identity")
    require(type(value["chunks"]) is list and bool(value["chunks"]), "capture chunks required")
    next_row = {layer: 0 for layer in spec.layers}
    for chunk in value["chunks"]:
        require(type(chunk) is dict and set(chunk) == {"layer", "start", "end", "document_rows"},
                "unexpected chunk fields")
        for name in chunk:
            integer(chunk[name], "chunk " + name)
        layer, start, end = chunk["layer"], chunk["start"], chunk["end"]
        require(layer in next_row and start == next_row[layer] and start < end <= len(spec.input_ids),
                "gapped, duplicate, foreign or reordered layer coverage")
        expected = max(0, min(end, spec.document_end) - max(start, spec.document_start))
        require(chunk["document_rows"] == expected, "document chunk overlap mismatch")
        next_row[layer] = end
    require(all(n == len(spec.input_ids) for n in next_row.values()), "incomplete full-input capture")
    # Return an ownership-independent, JSON-only copy; never retain caller dicts.
    return json.loads(canonical_bytes(value))


def _json(data):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj

    def bad_constant(value):
        raise ValidationError("nonfinite JSON value: " + value)

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=bad_constant)
    except ValidationError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("malformed snapshot manifest") from exc


def _open_directory(path):
    return os.open(os.fspath(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def _regular(fd, name):
    value = os.fstat(fd)
    require(stat.S_ISREG(value.st_mode), "snapshot file must be regular: " + name)
    return value


class SnapshotWriter:
    """Exclusive append-only generation. Only the final manifest publishes it.

    Failure preserves the partial directory and permanently aborts this writer.
    A pending manifest is never readable as a snapshot. Directory publication
    uses an exclusive hard link after payloads and pending manifest are fsynced.
    This is atomic readable publication, not a power-loss-durable transaction;
    the final directory-entry update has no post-publication durability promise.
    """

    def __init__(self, directory, spec, max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES):
        require(type(spec) is CaptureSpec, "CaptureSpec required")
        integer(max_payload_bytes, "max_payload_bytes", 1)
        total = len(spec.layers) * len(spec.document_ids) * spec.width * 8
        require(total <= max_payload_bytes, "snapshot payload exceeds byte budget")
        self.directory, self.spec = Path(directory), spec
        self.max_payload_bytes = max_payload_bytes
        self.state = "NEW"
        self._counts = {layer: 0 for layer in spec.layers}
        self._hashes = {layer: hashlib.sha256() for layer in spec.layers}
        self._streams = {}
        self._dir_fd = None
        # The parent must already exist; no broad or implicit directory creation.
        self.directory.mkdir()
        try:
            self._dir_fd = _open_directory(self.directory)
            self.state = "WRITING"
        except BaseException:
            self.state = "ABORTED"
            raise

    def _close(self):
        for stream in self._streams.values():
            try:
                stream.close()
            except OSError:
                pass
        self._streams.clear()
        if self._dir_fd is not None:
            os.close(self._dir_fd)
            self._dir_fd = None

    def abort(self):
        require(self.state != "COMPLETE", "completed snapshot cannot be aborted")
        self.state = "ABORTED"
        self._close()

    def append(self, layer, document_local_start, rows):
        require(self.state == "WRITING", "snapshot writer is not writable")
        try:
            integer(layer, "layer")
            integer(document_local_start, "document_local_start")
            require(layer in self._counts and document_local_start == self._counts[layer],
                    "foreign, gapped, duplicate or reordered payload rows")
            require(type(rows) in (list, tuple) and len(rows) * self.spec.width <= MAX_CHUNK_VALUES,
                    "payload chunk exceeds CPU value budget")
            data = normalize_rows(rows, width=self.spec.width)
            require(document_local_start + len(data) <= len(self.spec.document_ids), "too many payload rows")
            if layer not in self._streams:
                name = "layer%d.f64le" % layer
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self._dir_fd)
                self._streams[layer] = os.fdopen(fd, "wb")
            stream = self._streams[layer]
            pack = struct.Struct("<%dd" % self.spec.width)
            for row in data:
                encoded = pack.pack(*row)
                written = stream.write(encoded)
                require(written == len(encoded), "short snapshot write")
                self._hashes[layer].update(encoded)
            self._counts[layer] += len(data)
        except BaseException:
            self.abort()
            raise

    def finalize(self, provenance):
        require(self.state == "WRITING", "snapshot writer cannot finalize")
        try:
            provenance = _provenance(provenance, self.spec)
            require(all(n == len(self.spec.document_ids) for n in self._counts.values()),
                    "incomplete payload cannot be published")
            for stream in self._streams.values():
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            self._streams.clear()
            files = {str(layer): {"path": "layer%d.f64le" % layer,
                     "bytes": self._counts[layer] * self.spec.width * 8,
                     "sha256": self._hashes[layer].hexdigest()}
                     for layer in self.spec.layers}
            payload = {"schema": SCHEMA, "storage": "little-endian-float64",
                       "spec": self.spec.metadata(), "provenance": provenance, "files": files}
            sealed = dict(payload, seal=digest(payload))
            encoded = canonical_bytes(sealed) + b"\n"
            require(len(encoded) <= MAX_MANIFEST_BYTES, "manifest exceeds byte budget")
            fd = os.open(".manifest.pending", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=self._dir_fd)
            with os.fdopen(fd, "wb") as stream:
                require(stream.write(encoded) == len(encoded), "short manifest write")
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(self._dir_fd)
            # Atomic no-overwrite publication. The pending inode is retained;
            # loader never considers it. No retry or mutation after this point.
            os.link(".manifest.pending", "manifest.json", src_dir_fd=self._dir_fd,
                    dst_dir_fd=self._dir_fd, follow_symlinks=False)
            self.state = "COMPLETE"
            self._close()
            return sealed
        except BaseException:
            if self.state != "COMPLETE":
                self.abort()
            raise


class Snapshot:
    """Validated immutable-on-read view; every row read rechecks the whole file.

    Validation and row selection share one opened file handle, so replacement
    cannot switch a checked inode for an unchecked one. No native GPU tensor
    allocation or automatic gate authorization is provided.
    """

    def __init__(self, directory, expected_spec=None, max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES):
        integer(max_payload_bytes, "max_payload_bytes", 1)
        require(expected_spec is None or type(expected_spec) is CaptureSpec, "CaptureSpec required")
        self.directory = Path(directory)
        self.max_payload_bytes = max_payload_bytes
        self._dir_fd = _open_directory(self.directory)
        try:
            fd = os.open("manifest.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._dir_fd)
            with os.fdopen(fd, "rb") as stream:
                size = _regular(stream.fileno(), "manifest.json").st_size
                require(size <= MAX_MANIFEST_BYTES, "manifest exceeds byte budget")
                encoded = stream.read(MAX_MANIFEST_BYTES + 1)
            require(len(encoded) == size and len(encoded) <= MAX_MANIFEST_BYTES, "manifest changed or truncated")
            manifest = _json(encoded)
            require(type(manifest) is dict and set(manifest) == {
                "schema", "storage", "spec", "provenance", "files", "seal"}, "unexpected manifest fields")
            require(manifest["schema"] == SCHEMA and manifest["storage"] == "little-endian-float64",
                    "unsupported snapshot schema or storage")
            validate_digest(manifest["seal"], "manifest seal")
            require(manifest["seal"] == digest({k: v for k, v in manifest.items() if k != "seal"}),
                    "manifest seal mismatch")
            self.spec = _spec_from_metadata(manifest["spec"])
            if expected_spec is not None:
                require(self.spec == expected_spec, "snapshot differs from expected capture spec")
            _provenance(manifest["provenance"], self.spec)
            files = manifest["files"]
            require(type(files) is dict and set(files) == {str(k) for k in self.spec.layers},
                    "unexpected payload layer mapping")
            total = 0
            for layer in self.spec.layers:
                entry = files[str(layer)]
                require(type(entry) is dict and set(entry) == {"path", "bytes", "sha256"},
                        "unexpected payload entry")
                require(entry["path"] == "layer%d.f64le" % layer, "unsafe payload path")
                integer(entry["bytes"], "payload bytes", 1)
                require(entry["bytes"] == len(self.spec.document_ids) * self.spec.width * 8,
                        "payload dimensions disagree")
                validate_digest(entry["sha256"], "payload SHA256")
                total += entry["bytes"]
            require(total <= max_payload_bytes, "snapshot payload exceeds byte budget")
            self._manifest = manifest
            for layer in self.spec.layers:
                self._scan(layer, 0, 0)
        except BaseException:
            self.close()
            raise

    @property
    def manifest(self):
        return json.loads(canonical_bytes(self._manifest))

    @property
    def seal(self):
        return self._manifest["seal"]

    def close(self):
        if self._dir_fd is not None:
            os.close(self._dir_fd)
            self._dir_fd = None

    def __enter__(self):
        require(self._dir_fd is not None, "snapshot is closed")
        return self

    def __exit__(self, *unused):
        self.close()

    def _scan(self, layer, start, end):
        require(self._dir_fd is not None, "snapshot is closed")
        entry = self._manifest["files"][str(layer)]
        fd = os.open(entry["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._dir_fd)
        selected = []
        hasher = hashlib.sha256()
        width, row_size = self.spec.width, self.spec.width * 8
        block_rows = max(1, 65_536 // row_size)
        row_index = 0
        scanned_bytes = 0
        unpack = struct.Struct("<%dd" % width)
        with os.fdopen(fd, "rb") as stream:
            before = _regular(stream.fileno(), entry["path"])
            require(before.st_size == entry["bytes"], "payload byte size mismatch")
            while True:
                block = stream.read(row_size * block_rows)
                if not block:
                    break
                scanned_bytes += len(block)
                require(scanned_bytes <= entry["bytes"], "payload grew beyond sealed byte size")
                require(len(block) % row_size == 0, "partial binary row")
                hasher.update(block)
                for offset in range(0, len(block), row_size):
                    row = unpack.unpack_from(block, offset)
                    require(all(math.isfinite(v) for v in row), "nonfinite binary KV row")
                    if start <= row_index < end:
                        selected.append(row)
                    row_index += 1
            after = os.fstat(stream.fileno())
        require(row_index == len(self.spec.document_ids) and hasher.hexdigest() == entry["sha256"],
                "payload checksum or coverage mismatch")
        require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "payload changed during read")
        return tuple(selected)

    def read_rows(self, layer, start, end, max_values=1_048_576):
        integer(layer, "layer")
        integer(start, "start")
        integer(end, "end", 1)
        integer(max_values, "max_values", 1)
        require(layer in self.spec.layers and start < end <= len(self.spec.document_ids),
                "read outside snapshot layer/document")
        require((end - start) * self.spec.width <= max_values, "requested CPU rows exceed value budget")
        return self._scan(layer, start, end)

    def as_cache(self, layer, max_values=1_048_576):
        require(self.spec.kind == "independent_document", "context reference cannot become independent cache")
        rows = self.read_rows(layer, 0, len(self.spec.document_ids), max_values=max_values)
        return CacheArtifact.build(self.spec.identities[layer], rows)


def capture_snapshot(directory, spec, attentions, full_prefill_driver, *, ops=None,
                     max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES, max_chunk_values=MAX_CHUNK_VALUES):
    """Capture through caller-owned isolated attentions; never loads a model.

    The driver must actually execute the complete input and attach the session's
    per-chunk binding to native metadata. Python cannot prove driver honesty.
    A normal return after complete session coverage is mandatory for publication.
    """
    require(callable(full_prefill_driver), "full prefill driver required")
    from .deepinfer_adapter import DeepInferCaptureSession
    writer = SnapshotWriter(directory, spec, max_payload_bytes=max_payload_bytes)
    try:
        session = DeepInferCaptureSession(
            attentions, request_id=spec.request_id, phase=spec.phase, input_ids=spec.input_ids,
            identities=spec.identities, document_start=spec.document_start, ops=ops,
            sink=writer.append, max_chunk_values=max_chunk_values)
        with session:
            full_prefill_driver(spec, session)
        require(session.state == "COMPLETE", "driver did not complete full capture")
        writer.finalize(session.capture_provenance)
    except BaseException:
        if writer.state != "COMPLETE":
            writer.abort()
        raise
    return Snapshot(directory, expected_spec=spec, max_payload_bytes=max_payload_bytes)
