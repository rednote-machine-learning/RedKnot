"""Fail-closed merger and selector for DeepSeek V4 drift captures.

The runtime profiler writes one ``.pt`` capture per tensor-parallel rank.  A
standalone snapshot request contains exactly one reusable segment, while an
oracle request may contain every segment in a composed prompt.  This module
validates those files, builds one rank-local :class:`MLAHeadDriftCollector` per
TP shard, merges the eight reports in embedded-rank order, and emits two
8/16/24/32/48/56-local-head families: conservative threshold-admitted strict
configurations and explicitly unsafe fixed-count exploratory ablations.

Capture files are untrusted inputs.  They are loaded only with
``torch.load(weights_only=True)`` and the resulting tree is restricted to
plain dict/list/tuple containers, tensors, and primitive scalar values.
Selection is deliberately only a screening stage.  Every emitted candidate
still requires both end-to-end accuracy gates documented in ``summary.json``;
the exploratory family additionally forces counts that strict admission may
reject and must never be presented as an accuracy or performance claim.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

try:  # Installed package layout.
    from .mla_head_drift_profiler import (
        HeadSelectionThresholds,
        MLAHeadDriftCollector,
        build_deepseek_v4_head_config_dict,
        merge_tp_drift_reports,
        select_tp_balanced_fixed_count_exploratory,
        select_tp_balanced_nested,
    )
except ImportError:  # Flat staging directory and direct CLI execution.
    from mla_head_drift_profiler import (
        HeadSelectionThresholds,
        MLAHeadDriftCollector,
        build_deepseek_v4_head_config_dict,
        merge_tp_drift_reports,
        select_tp_balanced_fixed_count_exploratory,
        select_tp_balanced_nested,
    )


RANK_CAPTURE_FORMAT = "redknot_deepseek_v4_mla_head_drift_rank_capture_v1"
ANALYSIS_SUMMARY_FORMAT = "redknot_deepseek_v4_mla_head_drift_analysis_v1"
CALIBRATION_MANIFEST_FORMAT = "redknot_dsv4_drift_calibration_manifest_v2"
CALIBRATION_PAYLOAD_SCHEMA = "redknot_dsv4_drift_calibration_payload_v1"
CALIBRATION_DIGEST_SCOPE = {
    "algorithm": "sha256",
    "payload_field": "canonical_payload",
    "canonicalization": "utf8_json_sort_keys_compact_no_nan_v1",
}
EXPECTED_LAYERS = 43
EXPECTED_TP_SIZE = 8
EXPECTED_GLOBAL_HEADS = 64
EXPECTED_HEADS_PER_RANK = 8
EXPECTED_LOCAL_GROUPS = 1
MIN_SEGMENTS = 4
MAX_SEGMENTS = 16
LOCAL_WINDOW = 128
TARGET_HEADS_PER_RANK = (1, 2, 3, 4, 6, 7)
STRICT_SELECTION_MODE = "strict_thresholded"
EXPLORATORY_SELECTION_MODE = "exploratory_fixed_count"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RANK_IN_NAME = "rank[-_]?{rank}"
_MAX_MANIFEST_BYTES = 1 << 20

# Conservative screening defaults.  They are CLI-visible and recorded in
# every output; none of them is an end-to-end quality claim.
DEFAULT_MAX_HEAD_RELATIVE_RMS = 0.02
DEFAULT_MAX_HEAD_ROW_P99 = 0.05
DEFAULT_MAX_HEAD_ROW_MAX = 0.15
DEFAULT_MIN_HEAD_COSINE = 0.999
DEFAULT_MAX_RANK_NO_CANCEL_ERROR = 0.05

STRICT_WARNING = (
    "Projected-head drift is only a screening signal. No strict candidate may "
    "be called accuracy-safe, TTFT-improving, or QPS-improving until both "
    "held-out end-to-end accuracy gates and real performance benchmarks pass."
)
EXPLORATORY_WARNING = (
    "Exploratory fixed-count candidates deliberately force the requested number "
    "of local heads even when strict drift admission rejects them. They are "
    "ablation inputs only: they are not accuracy-safe or performance-qualified "
    "and require both held-out end-to-end accuracy gates before any claim."
)


@dataclass(frozen=True)
class CaptureSegment:
    segment_id: str
    start: int
    length: int
    sample_rows: Tuple[int, ...]
    token_ids_sha256: str

    def content_identity(self) -> Tuple[Any, ...]:
        """Identity shared by standalone and repositioned composed captures."""

        return (
            self.segment_id,
            self.length,
            self.sample_rows,
            self.token_ids_sha256,
        )


@dataclass(frozen=True)
class RankCapture:
    path: Path
    role: str
    run_id: str
    calibration_id: str
    calibration_digest: str
    capture_digest: str
    tp_rank: int
    projection_rank: int
    source_dtype: str
    logical_seq_len: int
    segments: Tuple[CaptureSegment, ...]
    layers: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]]


@dataclass(frozen=True)
class ManifestOracleSegment:
    segment_id: str
    slot: int
    start: int
    length: int


@dataclass(frozen=True)
class ManifestOracle:
    oracle_id: str
    logical_seq_len: int
    segments: Tuple[ManifestOracleSegment, ...]


@dataclass(frozen=True)
class CalibrationManifest:
    path: Path
    run_id: str
    calibration_digest: str
    model_config_sha256: str
    o_lora_rank: int
    tp_size: int
    segments: Tuple[CaptureSegment, ...]
    oracles: Tuple[ManifestOracle, ...]
    estimated_capture_tensor_bytes: int
    max_capture_tensor_bytes: int
    canonical_payload: Mapping[str, Any]


def _strict_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _strict_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{name} must match {_SAFE_NAME.pattern!r}")
    return value


def _strict_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def token_ids_sha256(input_ids: Sequence[int]) -> str:
    """Hash token IDs exactly as the runtime's ``_hash_token_ids`` does."""

    if isinstance(input_ids, (str, bytes)) or not isinstance(input_ids, Sequence):
        raise TypeError("input_ids must be a sequence of non-negative integers")
    values = []
    for index, value in enumerate(input_ids):
        values.append(_strict_int(f"input_ids[{index}]", value))
    digest = hashlib.sha256()
    digest.update(f"{len(values)}:".encode("ascii"))
    digest.update(",".join(str(value) for value in values).encode("ascii"))
    return digest.hexdigest()


def canonical_manifest_digest(payload: Mapping[str, Any]) -> str:
    """Return the sole defined calibration digest over canonical payload JSON."""

    if type(payload) is not dict:
        raise TypeError("canonical manifest payload must be a plain dict")
    _validate_json_tree(payload, "canonical_payload")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_json_tree(value: Any, location: str = "root", depth: int = 0) -> None:
    if depth > 16:
        raise ValueError(f"manifest nesting is too deep at {location}")
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"manifest contains a non-finite number at {location}")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"manifest object key must be a string at {location}")
            _validate_json_tree(item, f"{location}[{key!r}]", depth + 1)
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{location}[{index}]", depth + 1)
        return
    raise TypeError(
        f"unsupported manifest JSON type at {location}: {type(value).__name__}"
    )


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"manifest contains duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"manifest contains forbidden JSON constant {value}")


def _load_safe_json(path: os.PathLike[str] | str) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest is not a regular file: {manifest_path}")
    size = manifest_path.stat().st_size
    if size <= 0 or size > _MAX_MANIFEST_BYTES:
        raise ValueError(f"manifest size must be in [1, {_MAX_MANIFEST_BYTES}] bytes")
    raw_text = manifest_path.read_text(encoding="utf-8")
    value = json.loads(
        raw_text,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )
    _validate_json_tree(value)
    if type(value) is not dict:
        raise TypeError("manifest root must be a JSON object")
    return manifest_path, value


def _expected_capture_tensor_bytes(
    *,
    segments: Sequence[CaptureSegment],
    oracles: Sequence[ManifestOracle],
    o_lora_rank: int,
) -> int:
    rows_per_rank = sum(len(segment.sample_rows) for segment in segments)
    rows_by_id = {segment.segment_id: len(segment.sample_rows) for segment in segments}
    rows_per_rank += sum(
        rows_by_id[item.segment_id] for oracle in oracles for item in oracle.segments
    )
    # Runtime artifacts materialize projected values as float32.
    return (
        EXPECTED_TP_SIZE
        * EXPECTED_LAYERS
        * rows_per_rank
        * EXPECTED_HEADS_PER_RANK
        * o_lora_rank
        * 4
    )


def _validate_safe_tree(value: Any, location: str = "root", depth: int = 0) -> None:
    """Reject every deserialized type outside the capture's data-only schema."""

    if depth > 16:
        raise ValueError(f"capture nesting is too deep at {location}")
    if torch.is_tensor(value):
        return
    if type(value) in (str, int, float, bool, type(None)):
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int):
                raise TypeError(
                    f"unsupported dictionary key type at {location}: "
                    f"{type(key).__name__}"
                )
            _validate_safe_tree(item, f"{location}[{key!r}]", depth + 1)
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_safe_tree(item, f"{location}[{index}]", depth + 1)
        return
    raise TypeError(
        f"unsupported deserialized type at {location}: {type(value).__name__}"
    )


def _load_weights_only_tree(path: os.PathLike[str] | str) -> Dict[str, Any]:
    capture_path = Path(path)
    if not capture_path.is_file():
        raise FileNotFoundError(f"capture is not a regular file: {capture_path}")
    # There is intentionally no compatibility fallback to unsafe pickle load.
    value = torch.load(capture_path, map_location="cpu", weights_only=True)
    _validate_safe_tree(value)
    if type(value) is not dict:
        raise TypeError(f"capture root must be a plain dict: {capture_path}")
    return value


def _capture_digest(segments: Sequence[CaptureSegment]) -> str:
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.segment_id.encode("utf-8"))
        digest.update(f":{segment.length}:".encode("ascii"))
        digest.update(",".join(str(row) for row in segment.sample_rows).encode("ascii"))
        digest.update(segment.token_ids_sha256.encode("ascii"))
    return digest.hexdigest()


def _plain_dict(name: str, value: Any) -> Dict[Any, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a plain dict")
    return value


def _plain_sequence(name: str, value: Any) -> Sequence[Any]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be a plain list or tuple")
    return value


def load_calibration_manifest(
    path: os.PathLike[str] | str,
) -> CalibrationManifest:
    """Safely load, re-hash, and semantically validate the calibration manifest."""

    manifest_path, raw = _load_safe_json(path)
    required_top = {
        "format",
        "digest_scope",
        "canonical_payload",
        "calibration_digest",
        "output_dir",
    }
    if set(raw) != required_top:
        raise ValueError(
            "manifest top-level keys must be exactly " + ", ".join(sorted(required_top))
        )
    if raw.get("format") != CALIBRATION_MANIFEST_FORMAT:
        raise ValueError("unsupported calibration manifest format")
    if raw.get("digest_scope") != CALIBRATION_DIGEST_SCOPE:
        raise ValueError("manifest digest_scope is not the canonical v1 definition")
    output_dir = _strict_string("output_dir", raw.get("output_dir"))
    if not Path(output_dir).is_absolute():
        raise ValueError("manifest output_dir must be absolute")
    payload = _plain_dict("canonical_payload", raw.get("canonical_payload"))
    declared_digest = _strict_digest(
        "calibration_digest", raw.get("calibration_digest")
    )
    actual_digest = canonical_manifest_digest(payload)
    if declared_digest != actual_digest:
        raise ValueError("calibration_digest is not the SHA-256 of canonical_payload")

    required_payload = {
        "schema",
        "run_id",
        "model",
        "topology",
        "segments",
        "oracles",
        "resource_guard",
    }
    if set(payload) != required_payload:
        raise ValueError(
            "canonical_payload keys must be exactly "
            + ", ".join(sorted(required_payload))
        )
    if payload.get("schema") != CALIBRATION_PAYLOAD_SCHEMA:
        raise ValueError("unsupported canonical calibration payload schema")
    run_id = _strict_name("canonical_payload.run_id", payload.get("run_id"))

    model = _plain_dict("canonical_payload.model", payload.get("model"))
    if set(model) != {
        "config_sha256",
        "num_layers",
        "num_attention_heads",
        "o_lora_rank",
    }:
        raise ValueError("manifest model geometry/hash fields are incomplete")
    model_hash = _strict_digest("model.config_sha256", model.get("config_sha256"))
    if (
        _strict_int("model.num_layers", model.get("num_layers"), minimum=1)
        != EXPECTED_LAYERS
    ):
        raise ValueError(f"manifest model must have exactly {EXPECTED_LAYERS} layers")
    if (
        _strict_int(
            "model.num_attention_heads",
            model.get("num_attention_heads"),
            minimum=1,
        )
        != EXPECTED_GLOBAL_HEADS
    ):
        raise ValueError(
            f"manifest model must have exactly {EXPECTED_GLOBAL_HEADS} attention heads"
        )
    o_lora_rank = _strict_int("model.o_lora_rank", model.get("o_lora_rank"), minimum=1)

    topology = _plain_dict("canonical_payload.topology", payload.get("topology"))
    expected_topology = {
        "tp_size": EXPECTED_TP_SIZE,
        "heads_per_rank": EXPECTED_HEADS_PER_RANK,
        "num_output_groups": EXPECTED_TP_SIZE,
    }
    if set(topology) != set(expected_topology):
        raise ValueError("manifest topology fields are incomplete")
    for name, expected in expected_topology.items():
        if _strict_int(f"topology.{name}", topology.get(name), minimum=1) != expected:
            raise ValueError(f"manifest topology.{name} must equal {expected}")

    raw_segments = _plain_sequence(
        "canonical_payload.segments", payload.get("segments")
    )
    if not MIN_SEGMENTS <= len(raw_segments) <= MAX_SEGMENTS:
        raise ValueError(
            f"manifest must define between {MIN_SEGMENTS} and {MAX_SEGMENTS} "
            "calibration segments"
        )
    num_segments = len(raw_segments)
    segments: List[CaptureSegment] = []
    seen_segment_ids = set()
    for index, raw_segment in enumerate(raw_segments):
        segment = _plain_dict(f"canonical_payload.segments[{index}]", raw_segment)
        if set(segment) != {"id", "length", "sample_rows", "token_ids_sha256"}:
            raise ValueError(f"manifest segment {index} has unexpected/missing fields")
        segment_id = _strict_name(f"segments[{index}].id", segment.get("id"))
        if segment_id in seen_segment_ids:
            raise ValueError(f"manifest contains duplicate segment {segment_id!r}")
        seen_segment_ids.add(segment_id)
        length = _strict_int(
            f"segments[{index}].length", segment.get("length"), minimum=1
        )
        raw_rows = _plain_sequence(
            f"segments[{index}].sample_rows", segment.get("sample_rows")
        )
        rows = tuple(
            _strict_int(f"segments[{index}].sample_rows[{row_index}]", row)
            for row_index, row in enumerate(raw_rows)
        )
        if not rows or tuple(sorted(set(rows))) != rows or rows[-1] >= length:
            raise ValueError(f"manifest segment {segment_id!r} has invalid sample_rows")
        token_digest = _strict_digest(
            f"segments[{index}].token_ids_sha256",
            segment.get("token_ids_sha256"),
        )
        segments.append(
            CaptureSegment(
                segment_id=segment_id,
                start=0,
                length=length,
                sample_rows=rows,
                token_ids_sha256=token_digest,
            )
        )

    raw_oracles = _plain_sequence("canonical_payload.oracles", payload.get("oracles"))
    if len(raw_oracles) != num_segments:
        raise ValueError(
            "manifest must define exactly one cyclic oracle rotation per segment"
        )
    base_ids = tuple(segment.segment_id for segment in segments)
    segment_by_id = {segment.segment_id: segment for segment in segments}
    oracles: List[ManifestOracle] = []
    seen_oracle_ids = set()
    slot_coverage = {segment_id: [] for segment_id in base_ids}
    for oracle_index, raw_oracle in enumerate(raw_oracles):
        oracle = _plain_dict(f"canonical_payload.oracles[{oracle_index}]", raw_oracle)
        if set(oracle) != {"id", "logical_seq_len", "segments"}:
            raise ValueError(
                f"manifest oracle {oracle_index} has unexpected/missing fields"
            )
        oracle_id = _strict_name(f"oracles[{oracle_index}].id", oracle.get("id"))
        if oracle_id in seen_oracle_ids:
            raise ValueError(f"manifest contains duplicate oracle id {oracle_id!r}")
        seen_oracle_ids.add(oracle_id)
        logical_seq_len = _strict_int(
            f"oracles[{oracle_index}].logical_seq_len",
            oracle.get("logical_seq_len"),
            minimum=1,
        )
        raw_layout = _plain_sequence(
            f"oracles[{oracle_index}].segments", oracle.get("segments")
        )
        if len(raw_layout) != num_segments:
            raise ValueError("every oracle must place every calibration segment")
        expected_order = tuple(
            base_ids[(oracle_index + slot) % num_segments]
            for slot in range(num_segments)
        )
        layout: List[ManifestOracleSegment] = []
        expected_start = 0
        for slot, raw_item in enumerate(raw_layout):
            item = _plain_dict(f"oracles[{oracle_index}].segments[{slot}]", raw_item)
            if set(item) != {"id", "slot", "start", "length"}:
                raise ValueError(
                    "manifest oracle segment has unexpected/missing fields"
                )
            segment_id = _strict_name(
                f"oracles[{oracle_index}].segments[{slot}].id", item.get("id")
            )
            if segment_id != expected_order[slot]:
                raise ValueError(
                    f"oracle {oracle_id!r} is not cyclic rotation {oracle_index}"
                )
            declared_slot = _strict_int("oracle segment slot", item.get("slot"))
            start = _strict_int("oracle segment start", item.get("start"))
            length = _strict_int("oracle segment length", item.get("length"), minimum=1)
            if declared_slot != slot:
                raise ValueError(f"oracle {oracle_id!r} slot numbering is not dense")
            if start != expected_start:
                raise ValueError(f"oracle {oracle_id!r} has an invalid segment start")
            if length != segment_by_id[segment_id].length:
                raise ValueError(f"oracle {oracle_id!r} segment length mismatch")
            expected_start += length
            slot_coverage[segment_id].append(slot)
            layout.append(
                ManifestOracleSegment(
                    segment_id=segment_id,
                    slot=slot,
                    start=start,
                    length=length,
                )
            )
        if logical_seq_len != expected_start:
            raise ValueError(f"oracle {oracle_id!r} logical_seq_len mismatch")
        oracles.append(
            ManifestOracle(
                oracle_id=oracle_id,
                logical_seq_len=logical_seq_len,
                segments=tuple(layout),
            )
        )
    expected_slots = list(range(num_segments))
    for segment_id, slots in slot_coverage.items():
        if sorted(slots) != expected_slots:
            raise ValueError(
                f"cyclic oracle coverage for {segment_id!r} is not exactly once per slot"
            )

    guard = _plain_dict(
        "canonical_payload.resource_guard", payload.get("resource_guard")
    )
    if set(guard) != {
        "estimated_capture_tensor_bytes",
        "max_capture_tensor_bytes",
    }:
        raise ValueError("manifest resource_guard fields are incomplete")
    estimated_bytes = _strict_int(
        "resource_guard.estimated_capture_tensor_bytes",
        guard.get("estimated_capture_tensor_bytes"),
        minimum=1,
    )
    max_bytes = _strict_int(
        "resource_guard.max_capture_tensor_bytes",
        guard.get("max_capture_tensor_bytes"),
        minimum=1,
    )
    recomputed_bytes = _expected_capture_tensor_bytes(
        segments=segments,
        oracles=oracles,
        o_lora_rank=o_lora_rank,
    )
    if estimated_bytes != recomputed_bytes:
        raise ValueError("manifest capture-memory estimate is not canonical")
    if estimated_bytes > max_bytes:
        raise ValueError("manifest capture-memory estimate exceeds its declared guard")

    return CalibrationManifest(
        path=manifest_path,
        run_id=run_id,
        calibration_digest=declared_digest,
        model_config_sha256=model_hash,
        o_lora_rank=o_lora_rank,
        tp_size=EXPECTED_TP_SIZE,
        segments=tuple(segments),
        oracles=tuple(oracles),
        estimated_capture_tensor_bytes=estimated_bytes,
        max_capture_tensor_bytes=max_bytes,
        canonical_payload=payload,
    )


def _parse_segments(raw: Mapping[str, Any], path: Path) -> Tuple[CaptureSegment, ...]:
    raw_segments = _plain_sequence("segments", raw.get("segments"))
    if not raw_segments:
        raise ValueError(f"capture has no segments: {path}")
    segments: List[CaptureSegment] = []
    seen = set()
    previous_end = 0
    for index, raw_segment in enumerate(raw_segments):
        segment = _plain_dict(f"segments[{index}]", raw_segment)
        segment_id = _strict_string(f"segments[{index}].id", segment.get("id"))
        if segment_id in seen:
            raise ValueError(f"duplicate segment id {segment_id!r}: {path}")
        seen.add(segment_id)
        start = _strict_int(f"segments[{index}].start", segment.get("start"))
        length = _strict_int(
            f"segments[{index}].length", segment.get("length"), minimum=1
        )
        if index and start < previous_end:
            raise ValueError(f"segments overlap or are out of order: {path}")
        previous_end = start + length
        rows_raw = _plain_sequence(
            f"segments[{index}].sample_rows", segment.get("sample_rows")
        )
        rows = tuple(
            _strict_int(f"segments[{index}].sample_rows[{row_index}]", row)
            for row_index, row in enumerate(rows_raw)
        )
        if not rows or tuple(sorted(set(rows))) != rows:
            raise ValueError(
                f"segment {segment_id!r} sample rows must be non-empty, "
                f"strictly increasing, and unique: {path}"
            )
        if rows[-1] >= length:
            raise ValueError(
                f"segment {segment_id!r} sample row exceeds its length: {path}"
            )
        token_digest = _strict_digest(
            f"segments[{index}].token_ids_sha256",
            segment.get("token_ids_sha256"),
        )
        segments.append(
            CaptureSegment(
                segment_id=segment_id,
                start=start,
                length=length,
                sample_rows=rows,
                token_ids_sha256=token_digest,
            )
        )
    return tuple(segments)


def _validate_tensor_entry(
    *,
    entry: Any,
    segment: CaptureSegment,
    projection_rank: int,
    path: Path,
    layer_id: int,
) -> Dict[str, torch.Tensor]:
    item = _plain_dict(f"layers[{layer_id}][{segment.segment_id!r}]", entry)
    if set(item) != {"row_ids", "projection"}:
        raise ValueError(
            f"layer entry keys must be exactly row_ids/projection at "
            f"layer={layer_id}, segment={segment.segment_id!r}: {path}"
        )
    rows = item["row_ids"]
    projection = item["projection"]
    if not torch.is_tensor(rows) or rows.device.type != "cpu":
        raise ValueError(f"row_ids must be a CPU tensor: {path}")
    if rows.layout != torch.strided or rows.dtype != torch.int64 or rows.ndim != 1:
        raise ValueError(f"row_ids must be a dense int64 vector: {path}")
    expected_rows = torch.tensor(segment.sample_rows, dtype=torch.int64)
    if not torch.equal(rows, expected_rows):
        raise ValueError(
            f"row_ids disagree with segment manifest at layer={layer_id}, "
            f"segment={segment.segment_id!r}: {path}"
        )
    if not torch.is_tensor(projection) or projection.device.type != "cpu":
        raise ValueError(f"projection must be a CPU tensor: {path}")
    if (
        projection.layout != torch.strided
        or projection.dtype != torch.float32
        or projection.ndim != 3
        or tuple(projection.shape)
        != (len(segment.sample_rows), EXPECTED_HEADS_PER_RANK, projection_rank)
    ):
        raise ValueError(
            "projection must be dense float32 with shape "
            f"[{len(segment.sample_rows)}, {EXPECTED_HEADS_PER_RANK}, "
            f"{projection_rank}] at layer={layer_id}, "
            f"segment={segment.segment_id!r}: {path}"
        )
    if not bool(torch.isfinite(projection).all().item()):
        raise ValueError(
            f"projection contains non-finite values at layer={layer_id}, "
            f"segment={segment.segment_id!r}: {path}"
        )
    return {"row_ids": rows, "projection": projection}


def load_rank_capture(
    path: os.PathLike[str] | str,
    *,
    expected_role: str,
) -> RankCapture:
    """Load and fully validate one runtime rank capture."""

    capture_path = Path(path).resolve()
    raw = _load_weights_only_tree(capture_path)
    if raw.get("format") != RANK_CAPTURE_FORMAT:
        raise ValueError(f"unsupported capture format: {capture_path}")
    role = raw.get("role")
    if role != expected_role or role not in ("snapshot", "oracle"):
        raise ValueError(
            f"capture role={role!r}, expected {expected_role!r}: {capture_path}"
        )
    run_id = _strict_string("run_id", raw.get("run_id"))
    calibration_id = _strict_string("calibration_id", raw.get("calibration_id"))
    if calibration_id != run_id:
        raise ValueError(f"run_id/calibration_id mismatch: {capture_path}")
    calibration_digest = _strict_digest(
        "calibration_digest", raw.get("calibration_digest")
    )
    declared_capture_digest = _strict_digest(
        "capture_digest", raw.get("capture_digest")
    )

    rank = _strict_int("tp_rank", raw.get("tp_rank"))
    if rank >= EXPECTED_TP_SIZE:
        raise ValueError(f"tp_rank is outside TP8: {capture_path}")
    exact_int_fields = {
        "tp_size": EXPECTED_TP_SIZE,
        "source_tp_rank": rank,
        "source_tp_world_size": EXPECTED_TP_SIZE,
        "num_layers": EXPECTED_LAYERS,
        "num_attention_heads": EXPECTED_GLOBAL_HEADS,
        "local_head_start": rank * EXPECTED_HEADS_PER_RANK,
        "local_head_end": (rank + 1) * EXPECTED_HEADS_PER_RANK,
        "heads_per_rank": EXPECTED_HEADS_PER_RANK,
        "num_local_heads": EXPECTED_HEADS_PER_RANK,
        "n_local_groups": EXPECTED_LOCAL_GROUPS,
        "num_output_groups": EXPECTED_TP_SIZE,
        "heads_per_output_group": EXPECTED_HEADS_PER_RANK,
    }
    for name, expected in exact_int_fields.items():
        observed = _strict_int(name, raw.get(name))
        if observed != expected:
            raise ValueError(f"{name}={observed}, expected {expected}: {capture_path}")
    if raw.get("represented_tp_ranks") != [rank]:
        raise ValueError(f"represented_tp_ranks must be [{rank}]: {capture_path}")
    expected_heads = list(
        range(rank * EXPECTED_HEADS_PER_RANK, (rank + 1) * EXPECTED_HEADS_PER_RANK)
    )
    if raw.get("global_head_ids") != expected_heads:
        raise ValueError(
            f"global_head_ids are not the contiguous rank-{rank} shard: {capture_path}"
        )
    if raw.get("local_output_group_ids") != [rank]:
        raise ValueError(
            f"local_output_group_ids must identify group {rank}: {capture_path}"
        )

    projection_rank = _strict_int(
        "projection_rank", raw.get("projection_rank"), minimum=1
    )
    source_dtype = _strict_string("source_dtype", raw.get("source_dtype"))
    logical_seq_len = _strict_int(
        "logical_seq_len", raw.get("logical_seq_len"), minimum=1
    )
    segments = _parse_segments(raw, capture_path)
    if segments[-1].start + segments[-1].length > logical_seq_len:
        raise ValueError(f"segment extends past logical_seq_len: {capture_path}")
    actual_capture_digest = _capture_digest(segments)
    if declared_capture_digest != actual_capture_digest:
        raise ValueError(f"capture_digest does not match its manifest: {capture_path}")

    raw_layers = _plain_dict("layers", raw.get("layers"))
    expected_layer_ids = set(range(EXPECTED_LAYERS))
    if set(raw_layers) != expected_layer_ids:
        raise ValueError(
            f"layers must be exactly 0..{EXPECTED_LAYERS - 1}: {capture_path}"
        )
    expected_segment_ids = {segment.segment_id for segment in segments}
    segment_by_id = {segment.segment_id: segment for segment in segments}
    layers: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    for layer_id in range(EXPECTED_LAYERS):
        raw_layer = _plain_dict(f"layers[{layer_id}]", raw_layers[layer_id])
        if set(raw_layer) != expected_segment_ids:
            raise ValueError(
                f"layer {layer_id} segment IDs disagree with manifest: {capture_path}"
            )
        layers[layer_id] = {
            segment_id: _validate_tensor_entry(
                entry=raw_layer[segment_id],
                segment=segment_by_id[segment_id],
                projection_rank=projection_rank,
                path=capture_path,
                layer_id=layer_id,
            )
            for segment_id in expected_segment_ids
        }
    return RankCapture(
        path=capture_path,
        role=role,
        run_id=run_id,
        calibration_id=calibration_id,
        calibration_digest=calibration_digest,
        capture_digest=declared_capture_digest,
        tp_rank=rank,
        projection_rank=projection_rank,
        source_dtype=source_dtype,
        logical_seq_len=logical_seq_len,
        segments=segments,
        layers=layers,
    )


def _rank_filename_group(path: Path, rank: int) -> str:
    """Return a rank-independent request key from a runtime output template.

    The runtime accepts an arbitrary ``{tp_rank}`` position, so the serialized
    artifact has no oracle identifier.  We consequently require the basename
    to retain an unambiguous ``rankN`` token (or one delimited numeric rank
    token, as in ``oracle_0.pt``).  This prevents list ordering from pairing
    two different oracle requests.
    """

    pattern = re.compile(
        rf"(?i)(?<![A-Za-z0-9]){_RANK_IN_NAME.format(rank=rank)}(?![0-9])"
    )
    matches = list(pattern.finditer(path.stem))
    if len(matches) == 1:
        match = matches[0]
    elif not matches:
        # Also accept a template such as ``oracle_0.pt``.  A delimited numeric
        # token is safe only when it occurs exactly once; ambiguous names fail
        # instead of relying on glob/list ordering.
        numeric = re.compile(rf"(?<![0-9]){rank}(?![0-9])")
        numeric_matches = list(numeric.finditer(path.stem))
        if len(numeric_matches) != 1:
            raise ValueError(
                "oracle capture basename must contain exactly one unambiguous "
                f"rank token for embedded rank {rank}: {path.name}"
            )
        match = numeric_matches[0]
    else:
        raise ValueError(
            "oracle capture basename must contain exactly one unambiguous "
            f"rank token for embedded rank {rank}: {path.name}"
        )
    normalized = path.stem[: match.start()] + "rank{tp_rank}" + path.stem[match.end() :]
    return str(path.parent.resolve() / (normalized + path.suffix))


def _require_complete_ranks(
    captures: Sequence[RankCapture], *, label: str
) -> Dict[int, RankCapture]:
    by_rank: Dict[int, RankCapture] = {}
    for capture in captures:
        if capture.tp_rank in by_rank:
            raise ValueError(f"duplicate {label} TP rank {capture.tp_rank}")
        by_rank[capture.tp_rank] = capture
    expected = set(range(EXPECTED_TP_SIZE))
    if set(by_rank) != expected:
        raise ValueError(
            f"{label} TP ranks are incomplete: expected={sorted(expected)} "
            f"observed={sorted(by_rank)}"
        )
    # The per-file validation plus the complete rank set proves global logical
    # head IDs are exactly contiguous 0..63; retain a defensive explicit check.
    head_ids = [
        head
        for rank in range(EXPECTED_TP_SIZE)
        for head in range(
            rank * EXPECTED_HEADS_PER_RANK,
            (rank + 1) * EXPECTED_HEADS_PER_RANK,
        )
    ]
    if head_ids != list(range(EXPECTED_GLOBAL_HEADS)):
        raise RuntimeError("internal TP head-contiguity invariant failed")
    return by_rank


def _same_request_across_ranks(
    captures: Mapping[int, RankCapture], *, label: str
) -> None:
    first = captures[0]
    for rank in range(1, EXPECTED_TP_SIZE):
        other = captures[rank]
        if (
            other.run_id != first.run_id
            or other.calibration_id != first.calibration_id
            or other.calibration_digest != first.calibration_digest
            or other.capture_digest != first.capture_digest
            or other.projection_rank != first.projection_rank
            or other.source_dtype != first.source_dtype
            or other.logical_seq_len != first.logical_seq_len
            or other.segments != first.segments
        ):
            raise ValueError(f"{label} metadata differs at TP rank {rank}")


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if type(value) is dict:
        return {str(key): _json_safe(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_global_identity(captures: Sequence[RankCapture]) -> Tuple[str, str]:
    if not captures:
        raise ValueError("no captures were supplied")
    first = captures[0]
    for capture in captures[1:]:
        if capture.run_id != first.run_id:
            raise ValueError(f"run_id mismatch between {first.path} and {capture.path}")
        if capture.calibration_id != first.calibration_id:
            raise ValueError(
                "calibration_id mismatch between captures: "
                f"{first.path} and {capture.path}"
            )
        if capture.calibration_digest != first.calibration_digest:
            raise ValueError(
                "global calibration_digest mismatch between captures: "
                f"{first.path} and {capture.path}"
            )
        if capture.projection_rank != first.projection_rank:
            raise ValueError("projection_rank differs between captures")
        if capture.source_dtype != first.source_dtype:
            raise ValueError("source_dtype differs between captures")
    return first.run_id, first.calibration_digest


def _capture_size_limit(tensor_bytes: int, file_count: int) -> int:
    # One percent covers serialization alignment; 1 MiB/file is a deliberately
    # generous fixed allowance for tensor metadata and zip container records.
    return tensor_bytes + max(1, tensor_bytes // 100) + file_count * (1 << 20)


def _preflight_capture_files(
    *,
    snapshot_paths: Sequence[Path],
    oracle_paths: Sequence[Path],
    manifest: CalibrationManifest,
) -> None:
    """Bound file count and bytes before any ``torch.load`` allocation."""

    expected_per_role = len(manifest.segments) * EXPECTED_TP_SIZE
    if len(snapshot_paths) != expected_per_role:
        raise ValueError(
            f"snapshot capture file count must be exactly {expected_per_role}, "
            f"got {len(snapshot_paths)}"
        )
    if len(oracle_paths) != expected_per_role:
        raise ValueError(
            f"oracle capture file count must be exactly {expected_per_role}, "
            f"got {len(oracle_paths)}"
        )
    all_paths = tuple(snapshot_paths) + tuple(oracle_paths)
    sizes: Dict[Path, int] = {}
    for path in all_paths:
        if not path.is_file():
            raise FileNotFoundError(f"capture is not a regular file: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"capture file is empty: {path}")
        sizes[path] = size

    rows_by_segment = [len(segment.sample_rows) for segment in manifest.segments]
    snapshot_tensor_max = (
        EXPECTED_LAYERS
        * max(rows_by_segment)
        * EXPECTED_HEADS_PER_RANK
        * manifest.o_lora_rank
        * 4
    )
    oracle_tensor_max = (
        EXPECTED_LAYERS
        * sum(rows_by_segment)
        * EXPECTED_HEADS_PER_RANK
        * manifest.o_lora_rank
        * 4
    )
    snapshot_file_limit = _capture_size_limit(snapshot_tensor_max, 1)
    oracle_file_limit = _capture_size_limit(oracle_tensor_max, 1)
    for path in snapshot_paths:
        if sizes[path] > snapshot_file_limit:
            raise ValueError(
                f"snapshot capture exceeds pre-load size bound: {path} "
                f"size={sizes[path]} limit={snapshot_file_limit}"
            )
    for path in oracle_paths:
        if sizes[path] > oracle_file_limit:
            raise ValueError(
                f"oracle capture exceeds pre-load size bound: {path} "
                f"size={sizes[path]} limit={oracle_file_limit}"
            )

    # Bind total allocation to both the canonical estimate and the explicitly
    # declared resource budget.  The tighter bound wins.
    file_count = len(all_paths)
    total_limit = min(
        _capture_size_limit(manifest.estimated_capture_tensor_bytes, file_count),
        _capture_size_limit(manifest.max_capture_tensor_bytes, file_count),
    )
    total_size = sum(sizes.values())
    if total_size > total_limit:
        raise ValueError(
            "capture set exceeds pre-load total size bound: "
            f"size={total_size} limit={total_limit}"
        )


def _prepare_capture_sets(
    snapshot_paths: Sequence[os.PathLike[str] | str],
    oracle_paths: Sequence[os.PathLike[str] | str],
    manifest: CalibrationManifest,
) -> Tuple[
    Dict[str, Dict[int, RankCapture]],
    Dict[str, Dict[int, RankCapture]],
    Tuple[str, ...],
    Tuple[str, ...],
]:
    snapshot_resolved = tuple(Path(path).resolve() for path in snapshot_paths)
    oracle_resolved = tuple(Path(path).resolve() for path in oracle_paths)
    if not snapshot_resolved or not oracle_resolved:
        raise ValueError("snapshot_paths and oracle_paths must both be non-empty")
    if len(set(snapshot_resolved)) != len(snapshot_resolved):
        raise ValueError("snapshot_paths contains duplicate files")
    if len(set(oracle_resolved)) != len(oracle_resolved):
        raise ValueError("oracle_paths contains duplicate files")
    overlap = set(snapshot_resolved).intersection(oracle_resolved)
    if overlap:
        raise ValueError(f"snapshot/oracle path sets overlap: {sorted(overlap)}")

    _preflight_capture_files(
        snapshot_paths=snapshot_resolved,
        oracle_paths=oracle_resolved,
        manifest=manifest,
    )

    snapshots = [
        load_rank_capture(path, expected_role="snapshot") for path in snapshot_resolved
    ]
    oracles = [
        load_rank_capture(path, expected_role="oracle") for path in oracle_resolved
    ]
    run_id, calibration_digest = _validate_global_identity(snapshots + oracles)
    if run_id != manifest.run_id:
        raise ValueError("capture run_id does not match the calibration manifest")
    if calibration_digest != manifest.calibration_digest:
        raise ValueError(
            "capture calibration_digest does not match the recomputed manifest digest"
        )
    if any(
        capture.projection_rank != manifest.o_lora_rank
        for capture in snapshots + oracles
    ):
        raise ValueError(
            "capture projection_rank does not match manifest model.o_lora_rank"
        )

    snapshots_by_segment: Dict[str, List[RankCapture]] = {}
    for capture in snapshots:
        if len(capture.segments) != 1:
            raise ValueError(
                "each snapshot capture must contain exactly one standalone "
                f"segment: {capture.path}"
            )
        segment_id = capture.segments[0].segment_id
        snapshots_by_segment.setdefault(segment_id, []).append(capture)
    snapshot_groups = {
        segment_id: _require_complete_ranks(items, label=f"snapshot {segment_id!r}")
        for segment_id, items in snapshots_by_segment.items()
    }
    for segment_id, group in snapshot_groups.items():
        _same_request_across_ranks(group, label=f"snapshot {segment_id!r}")

    manifest_segment_ids = tuple(segment.segment_id for segment in manifest.segments)
    if set(snapshot_groups) != set(manifest_segment_ids):
        raise ValueError("snapshot segment IDs do not exactly match the manifest")
    manifest_segments = {segment.segment_id: segment for segment in manifest.segments}
    for segment_id in manifest_segment_ids:
        capture = snapshot_groups[segment_id][0]
        observed = capture.segments[0]
        expected = manifest_segments[segment_id]
        if observed.content_identity() != expected.content_identity():
            raise ValueError(
                f"snapshot segment {segment_id!r} length/sample/token digest "
                "does not match the manifest"
            )
        if observed.start != 0 or capture.logical_seq_len != expected.length:
            raise ValueError(
                f"snapshot segment {segment_id!r} must be a standalone start=0 request"
            )

    oracle_by_filename: Dict[str, List[RankCapture]] = {}
    for capture in oracles:
        key = _rank_filename_group(capture.path, capture.tp_rank)
        oracle_by_filename.setdefault(key, []).append(capture)
    oracle_groups_by_key = {
        key: _require_complete_ranks(items, label=f"oracle request {key!r}")
        for key, items in oracle_by_filename.items()
    }
    if len(oracle_groups_by_key) != len(manifest.oracles):
        raise ValueError(
            "captures must contain exactly one cyclic oracle request per "
            f"manifest segment ({len(manifest.oracles)} required)"
        )
    for key, group in oracle_groups_by_key.items():
        _same_request_across_ranks(group, label=f"oracle request {key!r}")

    segment_ids = manifest_segment_ids
    expected_oracle_by_signature = {
        (
            oracle.logical_seq_len,
            tuple(
                (item.segment_id, item.start, item.length) for item in oracle.segments
            ),
        ): oracle
        for oracle in manifest.oracles
    }
    oracle_groups: Dict[str, Dict[int, RankCapture]] = {}
    seen_signatures = set()
    for key in sorted(oracle_groups_by_key):
        group = oracle_groups_by_key[key]
        first = group[0]
        signature = (
            first.logical_seq_len,
            tuple(
                (segment.segment_id, segment.start, segment.length)
                for segment in first.segments
            ),
        )
        if signature in seen_signatures:
            raise ValueError("duplicate cyclic oracle layout in rank captures")
        seen_signatures.add(signature)
        manifest_oracle = expected_oracle_by_signature.get(signature)
        if manifest_oracle is None:
            raise ValueError(
                "oracle layout does not match manifest order/start/length/"
                "logical_seq_len"
            )
        identities = {
            segment.segment_id: segment.content_identity() for segment in first.segments
        }
        if set(identities) != set(manifest_segments):
            raise ValueError(
                f"oracle request {key!r} segment set differs from snapshots"
            )
        for segment_id in segment_ids:
            if (
                identities[segment_id]
                != manifest_segments[segment_id].content_identity()
            ):
                raise ValueError(
                    f"oracle request {key!r} content/sample digest differs for "
                    f"segment {segment_id!r}"
                )
        if manifest_oracle.oracle_id in oracle_groups:
            raise ValueError("duplicate oracle layout matched one manifest oracle")
        oracle_groups[manifest_oracle.oracle_id] = group
    oracle_ids = tuple(oracle.oracle_id for oracle in manifest.oracles)
    if set(oracle_groups) != set(oracle_ids):
        raise ValueError("cyclic oracle captures do not cover every manifest oracle")
    return (
        snapshot_groups,
        oracle_groups,
        segment_ids,
        oracle_ids,
    )


def analyze_capture_files(
    snapshot_paths: Sequence[os.PathLike[str] | str],
    oracle_paths: Sequence[os.PathLike[str] | str],
    out_dir: os.PathLike[str] | str,
    *,
    manifest: os.PathLike[str] | str,
    max_head_relative_rms: float = DEFAULT_MAX_HEAD_RELATIVE_RMS,
    max_head_row_p99: float = DEFAULT_MAX_HEAD_ROW_P99,
    max_head_row_max: float = DEFAULT_MAX_HEAD_ROW_MAX,
    min_head_cosine: float = DEFAULT_MIN_HEAD_COSINE,
    max_rank_no_cancel_error: float = DEFAULT_MAX_RANK_NO_CANCEL_ERROR,
    dense_prefix_layers: int = 2,
    sink_size: int = 4,
) -> Dict[str, Any]:
    """Validate captures and write report, summary, strict, and exploratory configs.

    This is the public API intended for the benchmark harness.  It never loads
    a model and performs CPU-only tensor statistics.
    """

    calibration_manifest = load_calibration_manifest(manifest)
    dense_prefix = _strict_int("dense_prefix_layers", dense_prefix_layers, minimum=0)
    if dense_prefix > EXPECTED_LAYERS:
        raise ValueError("dense_prefix_layers exceeds the 43-layer model")
    sink = _strict_int("sink_size", sink_size, minimum=0)
    thresholds = HeadSelectionThresholds(
        max_head_relative_rms=max_head_relative_rms,
        max_head_row_p99=max_head_row_p99,
        max_head_row_max=max_head_row_max,
        min_head_cosine=min_head_cosine,
        max_rank_no_cancel_error=max_rank_no_cancel_error,
    )
    (
        snapshot_groups,
        oracle_groups,
        segment_ids,
        oracle_ids,
    ) = _prepare_capture_sets(
        snapshot_paths,
        oracle_paths,
        calibration_manifest,
    )
    run_id = calibration_manifest.run_id
    calibration_digest = calibration_manifest.calibration_digest

    rank_reports = []
    for rank in range(EXPECTED_TP_SIZE):
        collector = MLAHeadDriftCollector(
            num_layers=EXPECTED_LAYERS,
            num_heads=EXPECTED_HEADS_PER_RANK,
            num_output_groups=EXPECTED_LOCAL_GROUPS,
            tp_size=1,
            tp_world_size=EXPECTED_TP_SIZE,
            represented_tp_ranks=(rank,),
            calibration_id=run_id,
            dense_prefix_layers=dense_prefix,
            expected_segments=segment_ids,
            expected_oracles=oracle_ids,
        )
        for layer_id in range(EXPECTED_LAYERS):
            for segment_id in segment_ids:
                snapshot_entry = snapshot_groups[segment_id][rank].layers[layer_id][
                    segment_id
                ]
                collector.add_snapshot(
                    layer_id,
                    segment_id,
                    snapshot_entry["row_ids"],
                    snapshot_entry["projection"],
                )
                for oracle_id in oracle_ids:
                    oracle_entry = oracle_groups[oracle_id][rank].layers[layer_id][
                        segment_id
                    ]
                    collector.add_oracle(
                        layer_id,
                        segment_id,
                        oracle_id,
                        oracle_entry["row_ids"],
                        oracle_entry["projection"],
                    )
        rank_reports.append(collector.finalize())

    merged = merge_tp_drift_reports(tuple(reversed(rank_reports)))
    if (
        merged.num_layers != EXPECTED_LAYERS
        or merged.num_heads != EXPECTED_GLOBAL_HEADS
        or merged.tp_size != EXPECTED_TP_SIZE
        or merged.represented_tp_ranks != tuple(range(EXPECTED_TP_SIZE))
    ):
        raise RuntimeError("merged drift report has an invalid DSV4 TP8 topology")
    strict_selections = select_tp_balanced_nested(
        merged,
        thresholds=thresholds,
        target_local_heads_per_rank=TARGET_HEADS_PER_RANK,
    )
    exploratory_selections = select_tp_balanced_fixed_count_exploratory(
        merged,
        thresholds=thresholds,
        target_local_heads_per_rank=TARGET_HEADS_PER_RANK,
    )
    for family, expected_mode in (
        (strict_selections, STRICT_SELECTION_MODE),
        (exploratory_selections, EXPLORATORY_SELECTION_MODE),
    ):
        if tuple(
            selection.target_local_heads_per_rank for selection in family
        ) != TARGET_HEADS_PER_RANK:
            raise RuntimeError(
                f"{expected_mode} selector returned an invalid target family"
            )
        if any(selection.selection_mode != expected_mode for selection in family):
            raise RuntimeError(
                f"{expected_mode} selector returned inconsistent selection metadata"
            )

    output_dir = Path(out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.json"
    strict_candidate_paths = {
        target: output_dir
        / f"candidate_local_{target * EXPECTED_TP_SIZE:02d}_heads.json"
        for target in TARGET_HEADS_PER_RANK
    }
    exploratory_candidate_paths = {
        target: output_dir
        / f"exploratory_fixed_local_{target * EXPECTED_TP_SIZE:02d}_heads.json"
        for target in TARGET_HEADS_PER_RANK
    }
    all_paths = [
        report_path,
        summary_path,
        *strict_candidate_paths.values(),
        *exploratory_candidate_paths.values(),
    ]
    existing = [str(path) for path in all_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing analysis outputs: " + ", ".join(existing)
        )

    two_level_gates = [
        {
            "name": "policy_online_vs_all_global_oracle",
            "required": True,
            "meaning": (
                "Run the selected head policy fully online and compare it with "
                "the all-global online model on held-out prompts and generation."
            ),
        },
        {
            "name": "offline_reuse_vs_same_policy_online",
            "required": True,
            "meaning": (
                "Compare offline MLA reuse with a fully online run using the "
                "same selected head policy on held-out long generation."
            ),
        },
    ]

    def _selection_summary(selection, output_path: Path) -> Dict[str, Any]:
        target = selection.target_local_heads_per_rank
        target_global = target * EXPECTED_TP_SIZE
        non_dense_counts = selection.local_heads_per_rank_by_layer[dense_prefix:]
        return {
            "selection_mode": selection.selection_mode,
            "target_local_heads_per_rank": target,
            "target_local_heads_per_layer": target_global,
            "output_file": output_path.name,
            "realized_local_heads_across_model": selection.realized_local_heads,
            "realized_non_dense_ratio": selection.realized_local_ratio,
            "realized_whole_model_ratio": selection.whole_model_local_ratio,
            "minimum_realized_heads_per_rank_per_non_dense_layer": (
                min(non_dense_counts) if non_dense_counts else 0
            ),
            "maximum_realized_heads_per_rank_per_non_dense_layer": (
                max(non_dense_counts) if non_dense_counts else 0
            ),
            "layers_reaching_target": sum(
                count == target for count in non_dense_counts
            ),
            "accuracy_claim_qualified": False,
            "performance_claim_qualified": False,
        }

    strict_candidate_summaries = []
    strict_candidate_payloads: Dict[int, Dict[str, Any]] = {}
    strict_by_target = {
        selection.target_local_heads_per_rank: selection
        for selection in strict_selections
    }
    for selection in strict_selections:
        target = selection.target_local_heads_per_rank
        config = build_deepseek_v4_head_config_dict(
            selection,
            local_window=LOCAL_WINDOW,
            swa_capacity=LOCAL_WINDOW,
            sink_size=sink,
            profiling_meta={
                "calibration_id": run_id,
                "calibration_digest": calibration_digest,
                "calibration_manifest_format": CALIBRATION_MANIFEST_FORMAT,
                "model_config_sha256": calibration_manifest.model_config_sha256,
                "expected_segments": list(segment_ids),
                "expected_oracles": list(oracle_ids),
                "candidate_family": STRICT_SELECTION_MODE,
                "screening_only": True,
                "accuracy_claim_qualified": False,
                "performance_claim_qualified": False,
                "final_accuracy_gate_count": len(two_level_gates),
                "warning": STRICT_WARNING,
            },
        )
        if (
            config["profiling_meta"].get(
                "selection_mode", STRICT_SELECTION_MODE
            )
            != STRICT_SELECTION_MODE
        ):
            raise RuntimeError("strict candidate lost its selection_mode metadata")
        strict_candidate_payloads[target] = config
        strict_summary = _selection_summary(
            selection, strict_candidate_paths[target]
        )
        strict_summary["warning"] = STRICT_WARNING
        strict_candidate_summaries.append(strict_summary)

    exploratory_candidate_summaries = []
    exploratory_candidate_payloads: Dict[int, Dict[str, Any]] = {}
    for selection in exploratory_selections:
        target = selection.target_local_heads_per_rank
        strict_counterpart = strict_by_target[target]
        non_dense_counts = selection.local_heads_per_rank_by_layer[dense_prefix:]
        if any(count != target for count in non_dense_counts):
            raise RuntimeError(
                "exploratory fixed-count selector did not realize its requested "
                f"count for target={target}"
            )
        config = build_deepseek_v4_head_config_dict(
            selection,
            local_window=LOCAL_WINDOW,
            swa_capacity=LOCAL_WINDOW,
            sink_size=sink,
            profiling_meta={
                "calibration_id": run_id,
                "calibration_digest": calibration_digest,
                "calibration_manifest_format": CALIBRATION_MANIFEST_FORMAT,
                "model_config_sha256": calibration_manifest.model_config_sha256,
                "expected_segments": list(segment_ids),
                "expected_oracles": list(oracle_ids),
                "candidate_family": EXPLORATORY_SELECTION_MODE,
                "exploratory_only": True,
                "strict_threshold_admission_enforced": False,
                "strict_counterpart_file": strict_candidate_paths[target].name,
                "strict_counterpart_realized_local_heads": (
                    strict_counterpart.realized_local_heads
                ),
                "forced_local_heads_beyond_strict": (
                    selection.realized_local_heads
                    - strict_counterpart.realized_local_heads
                ),
                "final_accuracy_gate_count": len(two_level_gates),
                "warning": EXPLORATORY_WARNING,
            },
        )
        if (
            config["profiling_meta"].get("selection_mode")
            != EXPLORATORY_SELECTION_MODE
        ):
            raise RuntimeError(
                "exploratory candidate lost its selection_mode metadata"
            )
        exploratory_candidate_payloads[target] = config
        exploratory_summary = _selection_summary(
            selection, exploratory_candidate_paths[target]
        )
        exploratory_summary.update(
            {
                "strict_counterpart_file": strict_candidate_paths[target].name,
                "strict_counterpart_realized_local_heads_across_model": (
                    strict_counterpart.realized_local_heads
                ),
                "forced_local_heads_beyond_strict": (
                    selection.realized_local_heads
                    - strict_counterpart.realized_local_heads
                ),
                "warning": EXPLORATORY_WARNING,
            }
        )
        exploratory_candidate_summaries.append(exploratory_summary)

    report_payload = merged.to_dict()
    report_payload["calibration_digest"] = calibration_digest
    report_payload["calibration_manifest"] = {
        "path": str(calibration_manifest.path),
        "format": CALIBRATION_MANIFEST_FORMAT,
        "digest_scope": CALIBRATION_DIGEST_SCOPE,
        "model_config_sha256": calibration_manifest.model_config_sha256,
    }
    report_payload["capture_files"] = {
        "snapshot": sorted(str(Path(path).resolve()) for path in snapshot_paths),
        "oracle": sorted(str(Path(path).resolve()) for path in oracle_paths),
    }
    summary: Dict[str, Any] = {
        "format": ANALYSIS_SUMMARY_FORMAT,
        "status": "screening_candidates_only",
        "run_id": run_id,
        "calibration_id": run_id,
        "calibration_digest": calibration_digest,
        "manifest": {
            "path": str(calibration_manifest.path),
            "format": CALIBRATION_MANIFEST_FORMAT,
            "digest_scope": CALIBRATION_DIGEST_SCOPE,
            "model_config_sha256": calibration_manifest.model_config_sha256,
        },
        "resource_guard": {
            "estimated_capture_tensor_bytes": (
                calibration_manifest.estimated_capture_tensor_bytes
            ),
            "max_capture_tensor_bytes": (calibration_manifest.max_capture_tensor_bytes),
        },
        "topology": {
            "num_layers": EXPECTED_LAYERS,
            "tp_size": EXPECTED_TP_SIZE,
            "num_attention_heads": EXPECTED_GLOBAL_HEADS,
            "heads_per_rank": EXPECTED_HEADS_PER_RANK,
            "num_output_groups": EXPECTED_TP_SIZE,
        },
        "segments": list(segment_ids),
        "oracles": list(oracle_ids),
        "local_window": LOCAL_WINDOW,
        "target_local_heads_per_rank": list(TARGET_HEADS_PER_RANK),
        "target_local_heads_per_layer": [
            target * EXPECTED_TP_SIZE for target in TARGET_HEADS_PER_RANK
        ],
        "thresholds": thresholds.to_dict(),
        "threshold_policy": (
            "Conservative worst-pair per-head gates plus additive rank-local "
            "error norm with no cancellation credit. This policy applies only "
            "to strict_thresholded candidates."
        ),
        "exploratory_fixed_policy": (
            "Force the requested TP-balanced local-head count using the "
            "profiler ordering even when strict threshold admission rejects "
            "those heads."
        ),
        "report_file": report_path.name,
        # Backward-compatible alias: candidates remains the strict family.
        "candidates": strict_candidate_summaries,
        "strict_candidates": strict_candidate_summaries,
        "exploratory_fixed_candidates": exploratory_candidate_summaries,
        "strict_selection": {
            "selection_mode": STRICT_SELECTION_MODE,
            "candidates": strict_candidate_summaries,
            "accuracy_claim_qualified": False,
            "performance_claim_qualified": False,
            "warning": STRICT_WARNING,
        },
        "exploratory_fixed_selection": {
            "selection_mode": EXPLORATORY_SELECTION_MODE,
            "candidates": exploratory_candidate_summaries,
            "accuracy_claim_qualified": False,
            "performance_claim_qualified": False,
            "warning": EXPLORATORY_WARNING,
        },
        "required_final_accuracy_gates": two_level_gates,
        "accuracy_claim_qualified": False,
        "performance_claim_qualified": False,
        "warning": STRICT_WARNING,
        "exploratory_fixed_accuracy_claim_qualified": False,
        "exploratory_fixed_performance_claim_qualified": False,
        "exploratory_fixed_warning": EXPLORATORY_WARNING,
    }

    _write_json(report_path, report_payload)
    for target in TARGET_HEADS_PER_RANK:
        _write_json(
            strict_candidate_paths[target], strict_candidate_payloads[target]
        )
        _write_json(
            exploratory_candidate_paths[target],
            exploratory_candidate_payloads[target],
        )
    _write_json(summary_path, summary)
    return summary


def _expand_globs(patterns: Sequence[str], *, label: str) -> Tuple[Path, ...]:
    matches: List[Path] = []
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{label} glob must be a non-empty string")
        matches.extend(
            Path(path).resolve() for path in glob.glob(pattern, recursive=True)
        )
    files = tuple(sorted(path for path in set(matches) if path.is_file()))
    if not files:
        raise FileNotFoundError(f"{label} glob matched no regular files")
    return files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and merge TP8 DeepSeek V4 MLA head-drift captures."
    )
    parser.add_argument(
        "--snapshot-glob",
        action="append",
        required=True,
        help="Glob for standalone snapshot rank captures (repeatable).",
    )
    parser.add_argument(
        "--oracle-glob",
        action="append",
        required=True,
        help="Glob for composed oracle rank captures (repeatable).",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Canonical calibration manifest JSON used to bind every capture.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--max-head-relative-rms",
        type=float,
        default=DEFAULT_MAX_HEAD_RELATIVE_RMS,
    )
    parser.add_argument(
        "--max-head-row-p99", type=float, default=DEFAULT_MAX_HEAD_ROW_P99
    )
    parser.add_argument(
        "--max-head-row-max", type=float, default=DEFAULT_MAX_HEAD_ROW_MAX
    )
    parser.add_argument(
        "--min-head-cosine", type=float, default=DEFAULT_MIN_HEAD_COSINE
    )
    parser.add_argument(
        "--max-rank-no-cancel-error",
        type=float,
        default=DEFAULT_MAX_RANK_NO_CANCEL_ERROR,
    )
    parser.add_argument("--dense-prefix-layers", type=int, default=2)
    parser.add_argument("--sink-size", type=int, default=4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    snapshot_paths = _expand_globs(args.snapshot_glob, label="snapshot")
    oracle_paths = _expand_globs(args.oracle_glob, label="oracle")
    summary = analyze_capture_files(
        snapshot_paths,
        oracle_paths,
        args.out_dir,
        manifest=args.manifest,
        max_head_relative_rms=args.max_head_relative_rms,
        max_head_row_p99=args.max_head_row_p99,
        max_head_row_max=args.max_head_row_max,
        min_head_cosine=args.min_head_cosine,
        max_rank_no_cancel_error=args.max_rank_no_cancel_error,
        dense_prefix_layers=args.dense_prefix_layers,
        sink_size=args.sink_size,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(Path(args.out_dir).resolve() / "summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SUMMARY_FORMAT",
    "CALIBRATION_DIGEST_SCOPE",
    "CALIBRATION_MANIFEST_FORMAT",
    "CALIBRATION_PAYLOAD_SCHEMA",
    "EXPLORATORY_SELECTION_MODE",
    "EXPLORATORY_WARNING",
    "LOCAL_WINDOW",
    "STRICT_SELECTION_MODE",
    "STRICT_WARNING",
    "TARGET_HEADS_PER_RANK",
    "analyze_capture_files",
    "canonical_manifest_digest",
    "load_calibration_manifest",
    "load_rank_capture",
    "main",
    "token_ids_sha256",
]
