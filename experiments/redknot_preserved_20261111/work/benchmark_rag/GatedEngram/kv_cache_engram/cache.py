"""Sealed, non-executable CPU-reference cache files; no native GPU loading."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import CacheIdentity, Rows, ValidationError, canonical_bytes, integer, normalize_rows, require, validate_digest

SCHEMA = "kv-cache-engram-cpu-cache-v1"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def _payload(identity: CacheIdentity, rows: Rows) -> dict:
    return {"schema": SCHEMA, "identity": asdict(identity), "rows": rows}


def _checksum(identity: CacheIdentity, rows: Rows) -> str:
    return hashlib.sha256(canonical_bytes(_payload(identity, rows))).hexdigest()


@dataclass(frozen=True)
class CacheArtifact:
    identity: CacheIdentity
    rows: Rows
    checksum: str

    def __post_init__(self) -> None:
        require(type(self.identity) is CacheIdentity, "CacheIdentity required")
        validate_digest(self.checksum, "checksum")
        object.__setattr__(self, "rows", normalize_rows(self.rows, count=self.identity.token_count, width=self.identity.width))

    @classmethod
    def build(cls, identity: CacheIdentity, rows) -> CacheArtifact:
        require(type(identity) is CacheIdentity, "CacheIdentity required")
        immutable = normalize_rows(rows, count=identity.token_count, width=identity.width)
        return cls(identity, immutable, _checksum(identity, immutable))

    def validate(self, expected_identity: CacheIdentity | None = None) -> None:
        require(type(self.identity) is CacheIdentity, "invalid cache identity")
        # Reconstruct to catch accidental post-construction corruption as well.
        CacheIdentity(**asdict(self.identity))
        if expected_identity is not None:
            require(type(expected_identity) is CacheIdentity and self.identity == expected_identity, "cache identity mismatch")
        normalize_rows(self.rows, count=self.identity.token_count, width=self.identity.width)
        require(self.checksum == _checksum(self.identity, self.rows), "cache checksum mismatch")

    def take(self, positions) -> Rows:
        positions = tuple(positions)
        for index in positions:
            integer(index, "cache row index")
            require(index < self.identity.token_count, "cache row outside document")
        require(positions == tuple(sorted(set(positions))), "cache positions must be sorted and unique")
        return tuple(self.rows[i] for i in positions)

    def to_bytes(self) -> bytes:
        self.validate()
        return canonical_bytes(dict(_payload(self.identity, self.rows), checksum=self.checksum)) + b"\n"


def _unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValidationError("nonfinite JSON value: " + value)


def decode_cache(data: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES) -> CacheArtifact:
    integer(max_bytes, "max_bytes", 1)
    require(type(data) is bytes and len(data) <= max_bytes, "cache byte limit exceeded or invalid input")
    try:
        item = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        require(type(item) is dict and set(item) == {"schema", "identity", "rows", "checksum"}, "unexpected cache envelope")
        require(item["schema"] == SCHEMA, "unknown cache schema")
        require(
            type(item["identity"]) is dict and set(item["identity"]) == set(CacheIdentity.__dataclass_fields__),
            "unexpected cache identity fields",
        )
        artifact = CacheArtifact(CacheIdentity(**item["identity"]), item["rows"], item["checksum"])
        artifact.validate()
        return artifact
    except ValidationError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValidationError("malformed CPU cache") from exc


def load_cache(path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> CacheArtifact:
    integer(max_bytes, "max_bytes", 1)
    with Path(path).open("rb") as stream:
        data = stream.read(max_bytes + 1)
    return decode_cache(data, max_bytes=max_bytes)


def save_cache(path, artifact: CacheArtifact) -> None:
    """Create exclusively; never overwrite evidence or an existing cache.

    A failed write may leave a partial file. Loading such a file fails closed;
    no overwrite, retry-in-place or automatic deletion is attempted.
    """
    require(type(artifact) is CacheArtifact, "CacheArtifact required")
    data = artifact.to_bytes()
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
