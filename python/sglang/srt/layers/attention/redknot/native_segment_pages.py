"""Fail-closed policy and certificates for native RedKnot MLA pages.

This module intentionally has no Torch dependency.  The serving integration
uses it to bind two independent decisions before any GPU cache is touched:

* immutable offline document-local packed-MLA banks; and
* a bounded per-document Indexer Top-K policy which never caps online suffix
  positions.

The GPU kernels mirror these values, while this module remains the small CPU
oracle used by tests and runtime manifest validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence


NATIVE_SEGMENT_PAGE_FORMAT = "redknot_native_segment_pages_v1"
NATIVE_INDEXER_BUCKET_FORMAT = "redknot_native_indexer_bucket_v1"


def _strict_int(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NativeIndexerBucketPolicy:
    """Exact geometry for bounded offline-document Indexer selection.

    ``document_compressed_rows`` is expressed in the C4 logical position
    domain.  Positions at or beyond ``documents * document_compressed_rows``
    are online/query rows and are deliberately not capped.
    """

    documents: int
    document_compressed_rows: int
    per_document_cap: int
    indexer_topk: int

    def __post_init__(self) -> None:
        _strict_int(self.documents, "documents", minimum=1)
        _strict_int(
            self.document_compressed_rows,
            "document compressed rows",
            minimum=1,
        )
        _strict_int(self.per_document_cap, "per-document cap", minimum=1)
        _strict_int(self.indexer_topk, "Indexer Top-K", minimum=1)
        if self.per_document_cap > self.indexer_topk:
            raise ValueError("per-document cap cannot exceed Indexer Top-K")

    @property
    def offline_compressed_rows(self) -> int:
        return self.documents * self.document_compressed_rows

    @property
    def maximum_offline_kept(self) -> int:
        return min(self.indexer_topk, self.documents * self.per_document_cap)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "format": NATIVE_INDEXER_BUCKET_FORMAT,
                "documents": self.documents,
                "document_compressed_rows": self.document_compressed_rows,
                "per_document_cap": self.per_document_cap,
                "indexer_topk": self.indexer_topk,
                "online_suffix_uncapped": True,
            }
        )

    def bucket_for_position(self, logical_position: int) -> int:
        logical_position = _strict_int(logical_position, "logical position")
        if logical_position >= self.offline_compressed_rows:
            return self.documents
        return logical_position // self.document_compressed_rows

    def retain_reference(self, raw_positions: Sequence[int]) -> tuple[int, ...]:
        """Stable CPU oracle; GPU order is allowed to differ as a set.

        Invalid ``-1`` padding is ignored.  Each offline document is capped,
        while online suffix positions consume only the original global Top-K
        capacity.  No retained position may be invented.
        """

        counts = [0] * self.documents
        kept: list[int] = []
        for position in raw_positions:
            if type(position) is not int:
                raise TypeError("raw Indexer positions must be exact ints")
            if position < 0:
                if position != -1:
                    raise ValueError("only -1 is valid Indexer padding")
                continue
            bucket = self.bucket_for_position(position)
            if bucket < self.documents:
                if counts[bucket] >= self.per_document_cap:
                    continue
                counts[bucket] += 1
            kept.append(position)
            if len(kept) == self.indexer_topk:
                break
        return tuple(kept)


@dataclass(frozen=True)
class NativeSegmentBank:
    """One immutable document-local packed-MLA bank binding."""

    segment_ordinal: int
    domain: str
    layer_id: int
    source_epoch: int
    rows: int
    record_bytes: int
    token_hash: str
    bank_identity: tuple[object, ...]

    def __post_init__(self) -> None:
        _strict_int(self.segment_ordinal, "segment ordinal")
        _nonempty(self.domain, "domain")
        _strict_int(self.layer_id, "layer id")
        _strict_int(self.source_epoch, "source epoch")
        _strict_int(self.rows, "rows", minimum=1)
        _strict_int(self.record_bytes, "record bytes", minimum=1)
        if not (
            isinstance(self.token_hash, str)
            and self.token_hash.startswith("sha256:")
            and len(self.token_hash) == 71
        ):
            raise ValueError("token hash must be canonical sha256")
        if type(self.bank_identity) is not tuple or not self.bank_identity:
            raise ValueError("bank identity must be a non-empty tuple")


@dataclass(frozen=True)
class NativeSegmentPageCertificate:
    """Immutable bank bundle consumed by direct segment-page attention."""

    forward_id: str
    domain: str
    layer_id: int
    banks: tuple[NativeSegmentBank, ...]
    digest: str

    def __post_init__(self) -> None:
        _nonempty(self.forward_id, "forward id")
        _nonempty(self.domain, "domain")
        _strict_int(self.layer_id, "layer id")
        if type(self.banks) is not tuple or not self.banks:
            raise ValueError("native page certificate has no banks")
        ordinals = tuple(bank.segment_ordinal for bank in self.banks)
        if ordinals != tuple(range(len(self.banks))):
            raise ValueError("native page banks are not in canonical segment order")
        if any(
            bank.domain != self.domain or bank.layer_id != self.layer_id
            for bank in self.banks
        ):
            raise ValueError("native page bank crossed domain/layer ownership")
        _nonempty(self.digest, "certificate digest")


def compile_native_segment_page_certificate(
    *,
    forward_id: str,
    domain: str,
    layer_id: int,
    banks: Iterable[NativeSegmentBank],
) -> NativeSegmentPageCertificate:
    forward_id = _nonempty(forward_id, "forward id")
    domain = _nonempty(domain, "domain")
    layer_id = _strict_int(layer_id, "layer id")
    frozen = tuple(banks)
    payload = {
        "format": NATIVE_SEGMENT_PAGE_FORMAT,
        "forward_id": forward_id,
        "domain": domain,
        "layer_id": layer_id,
        "banks": [
            {
                "segment_ordinal": bank.segment_ordinal,
                "source_epoch": bank.source_epoch,
                "rows": bank.rows,
                "record_bytes": bank.record_bytes,
                "token_hash": bank.token_hash,
                "bank_identity": list(bank.bank_identity),
            }
            for bank in frozen
        ],
    }
    return NativeSegmentPageCertificate(
        forward_id=forward_id,
        domain=domain,
        layer_id=layer_id,
        banks=frozen,
        digest=_digest(payload),
    )


def parse_native_indexer_bucket_policy(
    environment: Mapping[str, str], *, indexer_topk: int
) -> NativeIndexerBucketPolicy | None:
    """Parse the opt-in policy without silently accepting partial config."""

    cap_text = str(environment.get("REDKNOT_NATIVE_INDEXER_DOC_CAP", "0"))
    try:
        cap = int(cap_text)
    except ValueError as error:
        raise ValueError("REDKNOT_NATIVE_INDEXER_DOC_CAP must be an int") from error
    if cap == 0:
        return None
    if cap < 0:
        raise ValueError("REDKNOT_NATIVE_INDEXER_DOC_CAP cannot be negative")
    required = {
        "REDKNOT_NATIVE_INDEXER_DOCUMENTS": "documents",
        "REDKNOT_NATIVE_INDEXER_C4_ROWS_PER_DOCUMENT": "rows",
    }
    parsed: dict[str, int] = {}
    for key, label in required.items():
        value = environment.get(key)
        if value is None:
            raise ValueError(f"{key} is required when native bucketing is enabled")
        try:
            parsed[label] = int(value)
        except ValueError as error:
            raise ValueError(f"{key} must be an int") from error
    return NativeIndexerBucketPolicy(
        documents=parsed["documents"],
        document_compressed_rows=parsed["rows"],
        per_document_cap=cap,
        indexer_topk=_strict_int(indexer_topk, "Indexer Top-K", minimum=1),
    )
