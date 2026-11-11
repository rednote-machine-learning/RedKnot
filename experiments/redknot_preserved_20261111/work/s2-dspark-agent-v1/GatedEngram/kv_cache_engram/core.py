"""Immutable, CPU-only contracts for KV Cache Engram.

This reference stores finite Python floats, not native BF16/FP8 tensors.
Fingerprints detect accidental identity/content changes; they are not signatures.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

Rows = tuple[tuple[float, ...], ...]
REPRESENTATION = "swa-postnorm-prerope-v1"
STORAGE_DTYPE = "float64-cpu-reference"


class ValidationError(ValueError):
    """An identity, representation, coverage or numeric contract failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_digest(value: str, name: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, name + " must be a lowercase SHA256 hex digest")


def integer(value: int, name: str, minimum: int = 0) -> None:
    require(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def token_digest(token_ids) -> str:
    """Digest exact IDs, without normalization or access to answers."""
    ids = tuple(token_ids)
    require(bool(ids), "token sequence must not be empty")
    for item in ids:
        integer(item, "token ID")
    return digest(ids)


def normalize_rows(rows, *, width: int | None = None, count: int | None = None, allow_empty: bool = False) -> Rows:
    """Copy CPU list/tuple data into immutable finite rectangular rows.

    Reject tensor-like inputs instead of accidentally copying a GPU tensor or
    silently invoking device synchronization. A future backend needs its own
    independently qualified representation adapter.
    """
    require(type(rows) in (list, tuple), "rows must be CPU Python lists or tuples")
    require(allow_empty or bool(rows), "rows must not be empty")
    if width is not None:
        integer(width, "width", 1)
    if count is not None:
        integer(count, "row count")
        require(len(rows) == count, "row count mismatch")
    result = []
    expected_width = width
    for row in rows:
        require(type(row) in (list, tuple) and bool(row), "each row must be a nonempty CPU sequence")
        if expected_width is None:
            expected_width = len(row)
        require(len(row) == expected_width, "row width mismatch")
        converted = []
        for number in row:
            require(type(number) in (int, float), "KV components must be real CPU numbers, not booleans")
            try:
                value = float(number)
            except (OverflowError, ValueError) as exc:
                raise ValidationError("KV component is not representable") from exc
            require(math.isfinite(value), "nonfinite KV component")
            converted.append(value)
        result.append(tuple(converted))
    return tuple(result)


@dataclass(frozen=True)
class CacheIdentity:
    model_sha256: str
    tokenizer_sha256: str
    execution_sha256: str
    tokens_sha256: str
    layer: int
    token_count: int
    width: int
    representation: str = REPRESENTATION
    storage_dtype: str = STORAGE_DTYPE

    def __post_init__(self) -> None:
        for name in ("model_sha256", "tokenizer_sha256", "execution_sha256", "tokens_sha256"):
            validate_digest(getattr(self, name), name)
        integer(self.layer, "layer")
        integer(self.token_count, "token_count", 1)
        integer(self.width, "width", 1)
        require(self.representation == REPRESENTATION, "unsupported KV representation")
        require(self.storage_dtype == STORAGE_DTYPE, "only CPU-reference float storage is supported")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Binding:
    request_id: str
    phase: str
    identity: CacheIdentity
    document_start: int
    block_start: int
    block_end: int
    request_sha256: str

    def __post_init__(self) -> None:
        require(isinstance(self.request_id, str) and bool(self.request_id.strip()), "request_id required")
        require(isinstance(self.phase, str) and bool(self.phase.strip()), "phase required")
        require(type(self.identity) is CacheIdentity, "CacheIdentity required")
        validate_digest(self.request_sha256, "request_sha256")
        integer(self.document_start, "document_start")
        integer(self.block_start, "block_start")
        integer(self.block_end, "block_end", 1)
        require(self.block_start < self.block_end <= self.identity.token_count, "block outside document")

    @property
    def positions(self) -> tuple[int, ...]:
        """Document-local indices; absolute position = document_start + index."""
        return tuple(range(self.block_start, self.block_end))

    @property
    def sha256(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class OnlineRows:
    binding_sha256: str
    positions: tuple[int, ...]
    rows: Rows

    def __post_init__(self) -> None:
        validate_digest(self.binding_sha256, "binding_sha256")
        require(type(self.positions) in (tuple, list), "positions must be a CPU sequence")
        positions = tuple(self.positions)
        for index in positions:
            integer(index, "position")
        require(positions == tuple(sorted(set(positions))), "positions must be sorted and unique")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "rows", normalize_rows(self.rows, count=len(positions), allow_empty=True))


def validate_online(binding: Binding, data: OnlineRows, expected_positions) -> OnlineRows:
    require(type(binding) is Binding and type(data) is OnlineRows, "typed online binding and result required")
    expected = tuple(expected_positions)
    require(expected == tuple(sorted(set(expected))), "expected positions must be unique and sorted")
    require(
        all(type(i) is int and binding.block_start <= i < binding.block_end for i in expected), "expected positions outside bound block"
    )
    require(data.binding_sha256 == binding.sha256, "foreign or stale online binding")
    require(data.positions == expected, "online positions differ from requested exact coverage")
    normalize_rows(data.rows, width=binding.identity.width, count=len(expected), allow_empty=True)
    return data
