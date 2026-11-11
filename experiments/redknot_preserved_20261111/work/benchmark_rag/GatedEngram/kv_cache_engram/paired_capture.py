"""Independent/full-context KV pairs through explicit caller-owned fresh runs.

No model, accelerator runtime or gate is imported or launched here. A caller
supplies a context-manager factory which creates and disposes one isolated run
per side. Clean-state declarations and object-identity checks are contracts,
not proof that a dishonest driver used the claimed weights or empty caches.
The pair is published only after both runs, their cleanup, and a fresh snapshot
validation succeed. Partial captures are retained and are never overwritten.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType

from .core import canonical_bytes, digest, integer, require, validate_digest
from .offline import (CaptureSpec, Snapshot, capture_snapshot, DEFAULT_MAX_PAYLOAD_BYTES,
                      MAX_CHUNK_VALUES, MAX_MANIFEST_BYTES, _json, _open_directory,
                      _regular, _spec_from_metadata)


PAIR_SCHEMA = "kv-cache-engram-paired-capture-v1"
CLEAN_STATE_CONTRACT = "fresh-request-no-prefix-or-kv-reuse-v1"
MAX_PAIR_MANIFEST_BYTES = 2 * MAX_MANIFEST_BYTES + 65536
_SIDES = ("independent", "reference")


def _name(value, field):
    require(type(value) is str and bool(value.strip()) and len(value) <= 4096,
            field + " must be a nonempty bounded string")


@dataclass(frozen=True)
class PairCaptureSpec:
    pair_id: str
    source_document_id: str
    split: str
    independent: CaptureSpec
    reference: CaptureSpec

    def __post_init__(self):
        _name(self.pair_id, "pair_id")
        _name(self.source_document_id, "source_document_id")
        require(type(self.split) is str and self.split in ("calibration", "evaluation"), "unknown pair split")
        require(type(self.independent) is CaptureSpec and type(self.reference) is CaptureSpec,
                "two CaptureSpec values required")
        for spec in (self.independent, self.reference):
            spec.__post_init__()
        require(self.independent.kind == "independent_document"
                and self.reference.kind == "full_context_reference", "wrong pair capture kinds")
        require(self.independent.request_id != self.reference.request_id,
                "paired sides require different request IDs")
        require(self.independent.document_ids == self.reference.document_ids,
                "paired document token IDs differ")
        for name in ("model_sha256", "tokenizer_sha256", "execution_sha256", "layers", "width"):
            require(getattr(self.independent, name) == getattr(self.reference, name),
                    "pair configuration differs: " + name)

    def metadata(self):
        return {"pair_id": self.pair_id, "source_document_id": self.source_document_id,
                "split": self.split, "independent": self.independent.metadata(),
                "reference": self.reference.metadata()}


def _pair_spec(value):
    require(type(value) is dict and set(value) == set(PairCaptureSpec.__dataclass_fields__),
            "unexpected pair spec fields")
    spec = PairCaptureSpec(value["pair_id"], value["source_document_id"], value["split"],
                           _spec_from_metadata(value["independent"]),
                           _spec_from_metadata(value["reference"]))
    require(value == spec.metadata(), "pair spec metadata differs")
    return spec


@dataclass(frozen=True)
class FreshRun:
    """Explicit clean-request descriptor yielded by ``fresh_run_factory(spec)``.

    The factory owns model initialization and disposal; the orchestrator never
    performs either. ``full_prefill_driver(spec, session)`` must execute the
    complete specified input. Supplying this descriptor does not authorize GPU
    execution and does not certify actual model behavior or KV equivalence.
    This first isolation contract requires different attention wrappers and
    request states on the two sides. They may share immutable model weights;
    neither copying weights nor loading a second model is required here.
    """

    spec: CaptureSpec
    attentions: Mapping
    full_prefill_driver: object
    run_id: str
    clean_state_contract: str
    initial_kv_state_empty: bool
    state_reused: bool
    ops: object = None

    def __post_init__(self):
        require(type(self.spec) is CaptureSpec, "fresh run CaptureSpec required")
        self.spec.__post_init__()
        require(isinstance(self.attentions, Mapping), "fresh attention mapping required")
        attentions = dict(self.attentions)
        require(set(attentions) == set(self.spec.layers), "fresh run layers differ from spec")
        require(all(type(layer) is int for layer in attentions), "integer attention keys required")
        require(len({id(module) for module in attentions.values()}) == len(attentions),
                "attention objects must be distinct across layers")
        object.__setattr__(self, "attentions", MappingProxyType(attentions))
        require(callable(self.full_prefill_driver), "full prefill driver required")
        _name(self.run_id, "run_id")
        require(self.clean_state_contract == CLEAN_STATE_CONTRACT
                and self.initial_kv_state_empty is True and self.state_reused is False,
                "fresh empty request state must be explicitly declared")

    def metadata(self):
        return {"run_id": self.run_id, "capture_spec_sha256": digest(self.spec.metadata()),
                "request_id": self.spec.request_id, "phase": self.spec.phase,
                "clean_state_contract": self.clean_state_contract,
                "initial_kv_state_empty": self.initial_kv_state_empty,
                "state_reused": self.state_reused}


def _run_metadata(value, spec):
    require(type(value) is dict and set(value) == {"run_id", "capture_spec_sha256", "request_id",
                "phase", "clean_state_contract", "initial_kv_state_empty", "state_reused"},
            "unexpected fresh-run metadata")
    _name(value["run_id"], "run_id")
    validate_digest(value["capture_spec_sha256"], "capture spec digest")
    require(value["capture_spec_sha256"] == digest(spec.metadata())
            and value["request_id"] == spec.request_id and value["phase"] == spec.phase,
            "fresh run bound to another capture spec")
    require(value["clean_state_contract"] == CLEAN_STATE_CONTRACT
            and value["initial_kv_state_empty"] is True and value["state_reused"] is False,
            "missing clean-state contract")


def _same_directory(fd, directory):
    current = _open_directory(directory)
    try:
        expected, actual = os.fstat(fd), os.fstat(current)
        require((expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino),
                "paired directory identity changed")
    finally:
        os.close(current)


def _snapshot_at(parent_fd, directory, side, spec, max_payload_bytes):
    # Snapshot's own reads remain descriptor-relative. Check that its opened
    # child is the same inode reached from our retained parent descriptor.
    # This portable wrapper does not promise protection against a malicious
    # same-user process repeatedly moving the caller-owned directory mid-read.
    _same_directory(parent_fd, directory)
    child_fd = os.open(side, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    snapshot = None
    try:
        snapshot = Snapshot(Path(directory) / side, expected_spec=spec, max_payload_bytes=max_payload_bytes)
        expected, actual = os.fstat(child_fd), os.fstat(snapshot._dir_fd)
        require((expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino),
                "paired child directory identity changed")
        _same_directory(parent_fd, directory)
        return snapshot
    except BaseException:
        if snapshot is not None:
            snapshot.close()
        raise
    finally:
        os.close(child_fd)


def _entry(snapshot, side, run):
    provenance = snapshot.manifest["provenance"]
    return {"path": side, "snapshot_seal": snapshot.seal, "run": run,
            "operator": provenance["operator"], "native_operator": provenance["native_operator"],
            "accelerator_capture": provenance["accelerator_capture"], "provenance": provenance}


def _validate_entry(entry, snapshot, side, spec):
    require(type(entry) is dict and set(entry) == {"path", "snapshot_seal", "run", "operator",
                "native_operator", "accelerator_capture", "provenance"}, "unexpected pair capture entry")
    require(entry["path"] == side, "unsafe paired snapshot path")
    _run_metadata(entry["run"], spec)
    validate_digest(entry["snapshot_seal"], "snapshot seal")
    require(entry["snapshot_seal"] == snapshot.seal, "paired snapshot seal changed")
    provenance = snapshot.manifest["provenance"]
    require(entry["provenance"] == provenance, "paired capture provenance changed")
    for key in ("operator", "native_operator", "accelerator_capture"):
        require(type(entry[key]) is type(provenance[key]) and entry[key] == provenance[key],
                "paired operator metadata differs: " + key)


class PairedCapture:
    """Read-only pair with validated Snapshot sources; no gate authorization."""

    def __init__(self, directory, expected_spec=None, max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES):
        integer(max_payload_bytes, "max_payload_bytes", 1)
        require(expected_spec is None or type(expected_spec) is PairCaptureSpec,
                "expected PairCaptureSpec required")
        self.directory = Path(directory)
        self._dir_fd = _open_directory(self.directory)
        self._sources = {}
        try:
            fd = os.open("pair.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._dir_fd)
            with os.fdopen(fd, "rb") as stream:
                size = _regular(stream.fileno(), "pair.json").st_size
                require(size <= MAX_PAIR_MANIFEST_BYTES, "pair manifest exceeds byte budget")
                encoded = stream.read(MAX_PAIR_MANIFEST_BYTES + 1)
            require(len(encoded) == size and len(encoded) <= MAX_PAIR_MANIFEST_BYTES,
                    "pair manifest changed or truncated")
            manifest = _json(encoded)
            require(type(manifest) is dict and set(manifest) == {"schema", "spec", "captures", "seal",
                        "context_equivalence_certified", "gate_authorized"}, "unexpected pair manifest fields")
            require(manifest["schema"] == PAIR_SCHEMA, "unsupported paired capture schema")
            require(manifest["context_equivalence_certified"] is False and manifest["gate_authorized"] is False,
                    "pair is evidence, not authorization")
            validate_digest(manifest["seal"], "pair seal")
            require(manifest["seal"] == digest({k: v for k, v in manifest.items() if k != "seal"}),
                    "pair seal mismatch")
            self.spec = _pair_spec(manifest["spec"])
            if expected_spec is not None:
                require(self.spec == expected_spec, "pair differs from expected spec")
            captures = manifest["captures"]
            require(type(captures) is dict and set(captures) == set(_SIDES), "two paired captures required")
            for side in _SIDES:
                require(type(captures[side]) is dict and captures[side].get("path") == side,
                        "unsafe paired snapshot path")
                spec = getattr(self.spec, side)
                snapshot = _snapshot_at(self._dir_fd, self.directory, side, spec, max_payload_bytes)
                self._sources[side] = snapshot
                _validate_entry(captures[side], snapshot, side, spec)
            require(captures["independent"]["run"]["run_id"] != captures["reference"]["run"]["run_id"],
                    "paired captures share a run identity")
            require(captures["independent"]["operator"] == captures["reference"]["operator"],
                    "paired capture operators differ")
            self._manifest = manifest
        except BaseException:
            self.close()
            raise

    @property
    def independent(self):
        require(self._dir_fd is not None, "paired capture is closed")
        return self._sources["independent"]

    @property
    def reference(self):
        require(self._dir_fd is not None, "paired capture is closed")
        return self._sources["reference"]

    @property
    def manifest(self):
        return json.loads(canonical_bytes(self._manifest))

    @property
    def seal(self):
        return self._manifest["seal"]

    def validate(self):
        """Recheck bound identities, manifests and every sealed payload."""
        require(self._dir_fd is not None, "paired capture is closed")
        require(self._manifest["seal"] == digest({k: v for k, v in self._manifest.items() if k != "seal"}),
                "pair manifest changed in memory")
        require(_pair_spec(self._manifest["spec"]) == self.spec, "pair spec changed after loading")
        for side in _SIDES:
            spec, source = getattr(self.spec, side), self._sources[side]
            require(source.spec == spec, "snapshot spec changed after loading")
            _validate_entry(self._manifest["captures"][side], source, side, spec)
            for layer in spec.layers:
                source._scan(layer, 0, 0)

    def close(self):
        for source in self._sources.values():
            source.close()
        self._sources.clear()
        if self._dir_fd is not None:
            os.close(self._dir_fd)
            self._dir_fd = None

    def __enter__(self):
        require(self._dir_fd is not None, "paired capture is closed")
        return self

    def __exit__(self, *unused):
        self.close()


def load_pair(directory, expected_spec=None, max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES):
    return PairedCapture(directory, expected_spec=expected_spec, max_payload_bytes=max_payload_bytes)


def capture_pair(directory, spec, fresh_run_factory, *, max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES,
                 max_chunk_values=MAX_CHUNK_VALUES):
    """Sequential fresh runs, then exclusive pair publication and revalidation.

    Each side's snapshot may exist after failure, but no successful pair is
    published unless both factories exit normally and both snapshots validate.
    The pair directory must not exist. The atomic final link is no-overwrite;
    as with SnapshotWriter, this is not a power-loss-durable transaction.
    max_payload_bytes applies separately to each side's document KV payload.
    """
    require(type(spec) is PairCaptureSpec, "PairCaptureSpec required")
    spec.__post_init__()
    require(callable(fresh_run_factory), "fresh run factory required")
    integer(max_payload_bytes, "max_payload_bytes", 1)
    integer(max_chunk_values, "max_chunk_values", 1)
    require(len(canonical_bytes(spec.metadata())) <= MAX_PAIR_MANIFEST_BYTES,
            "pair spec alone exceeds manifest byte budget")
    for side in _SIDES:
        capture = getattr(spec, side)
        require(len(canonical_bytes(capture.metadata())) <= MAX_MANIFEST_BYTES,
                "capture spec alone exceeds snapshot manifest byte budget")
        require(len(capture.document_ids) * len(capture.layers) * capture.width * 8 <= max_payload_bytes,
                "snapshot payload exceeds byte budget")
    directory = Path(directory)
    directory.mkdir()
    directory_fd = _open_directory(directory)
    captures, owned_runs, owned_contexts = {}, [], []
    try:
        for side in _SIDES:
            capture = getattr(spec, side)
            manager = fresh_run_factory(capture)
            require(callable(getattr(manager, "__enter__", None)) and callable(getattr(manager, "__exit__", None)),
                    "factory must return a fresh context manager")
            require(all(manager is not previous for previous in owned_contexts), "factory reused a context manager")
            owned_contexts.append(manager)
            completed = False
            with manager as run:
                require(type(run) is FreshRun, "factory must yield FreshRun")
                run.__post_init__()
                require(run.spec == capture, "fresh run differs from requested capture")
                require(all(run is not previous and run.run_id != previous.run_id for previous in owned_runs),
                        "factory reused a run identity")
                previous_modules = tuple(module for previous in owned_runs for module in previous.attentions.values())
                require(all(module is not old for module in run.attentions.values() for old in previous_modules),
                        "factory reused attention objects across paired sides")
                owned_runs.append(run)  # Keep objects alive: id reuse cannot fake isolation.
                _same_directory(directory_fd, directory)
                child = directory / side
                with capture_snapshot(child, capture, run.attentions, run.full_prefill_driver,
                                      ops=run.ops, max_payload_bytes=max_payload_bytes,
                                      max_chunk_values=max_chunk_values) as snapshot:
                    captures[side] = _entry(snapshot, side, run.metadata())
                completed = True
            # A context manager may suppress driver exceptions; that is not a
            # successful capture. Its cleanup must also have returned normally.
            require(completed, "factory suppressed a failed capture")
            _same_directory(directory_fd, directory)
        require(captures["independent"]["operator"] == captures["reference"]["operator"],
                "paired capture operators differ")
        for side in _SIDES:
            with _snapshot_at(directory_fd, directory, side, getattr(spec, side), max_payload_bytes) as snapshot:
                _validate_entry(captures[side], snapshot, side, getattr(spec, side))
        payload = {"schema": PAIR_SCHEMA, "spec": spec.metadata(), "captures": captures,
                   "context_equivalence_certified": False, "gate_authorized": False}
        manifest = dict(payload, seal=digest(payload))
        encoded = canonical_bytes(manifest) + b"\n"
        require(len(encoded) <= MAX_PAIR_MANIFEST_BYTES, "pair manifest exceeds byte budget")
        fd = os.open(".pair.pending", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        with os.fdopen(fd, "wb") as stream:
            require(stream.write(encoded) == len(encoded), "short pair manifest write")
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory_fd)
        os.link(".pair.pending", "pair.json", src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False)
    finally:
        os.close(directory_fd)
    return PairedCapture(directory, expected_spec=spec, max_payload_bytes=max_payload_bytes)
