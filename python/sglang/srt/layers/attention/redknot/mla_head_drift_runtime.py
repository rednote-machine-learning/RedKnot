"""Fail-closed runtime capture for DeepSeek V4 MLA projected-head drift.

The accuracy profiler in :mod:`mla_head_drift_profiler` consumes paired
snapshot/oracle observations.  This module is the deliberately small serving
hook that records one observation on one tensor-parallel rank.  It is active
only for an explicit ``redknot_reuse_plan.mode == "drift_profile"`` request.

The supported calibration topology is intentionally narrow: one complete,
unchunked EXTEND request, PP=1, TP=8, 64 logical attention heads (eight
contiguous heads per rank), and the non-FP8 ``wo_a`` path.  Rejecting everything
else is important: a partial prefill or a CUDA-graph replay can silently turn a
supposed snapshot/oracle comparison into a comparison of different rows.

Each rank keeps sampled projections on device while the model runs and writes
one rank-specific ``.pt`` file only after every model layer has been observed.
The write uses a same-directory temporary file followed by ``os.replace``.
No per-layer files are produced.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

try:  # Production package layout.
    from .mla_head_drift_profiler import decompose_wo_a_per_head
except ImportError:  # Flat staging directory used by focused CPU tests.
    from mla_head_drift_profiler import decompose_wo_a_per_head


RANK_CAPTURE_FORMAT = "redknot_deepseek_v4_mla_head_drift_rank_capture_v1"
DRIFT_PROFILE_MODE = "drift_profile"
SUPPORTED_TP_SIZE = 8
SUPPORTED_GLOBAL_HEADS = 64
SUPPORTED_HEADS_PER_RANK = 8
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _strict_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _strict_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{name} must match {_SAFE_NAME.pattern!r}, got {value!r}")
    return value


def is_drift_profile_plan(plan: Any) -> bool:
    """Return true only for the explicit profiling mode.

    This predicate performs no mutation and is safe on every ordinary request.
    Full validation happens in :func:`begin_drift_profile_request`.
    """

    return isinstance(plan, Mapping) and plan.get("mode") == DRIFT_PROFILE_MODE


@dataclass(frozen=True)
class DriftSegment:
    segment_id: str
    start: int
    length: int
    sample_rows: Tuple[int, ...]
    token_ids_sha256: str

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class DriftProfilePlan:
    run_id: str
    calibration_digest: str
    role: str
    output_path: Path
    tp_rank: int
    tp_size: int
    num_layers: int
    logical_seq_len: int
    segments: Tuple[DriftSegment, ...]
    # Concatenated model-row indices and the slice belonging to each segment.
    sample_indices: Tuple[int, ...]
    segment_slices: Mapping[str, Tuple[int, int]]


def _cpu_int_vector(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor) or tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(f"{name} must use an integer dtype")
    return tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()


def _hash_token_ids(token_ids: torch.Tensor) -> str:
    values = token_ids.to(dtype=torch.int64, device="cpu").contiguous()
    # A textual encoding is architecture-independent and tiny relative to the
    # actual calibration forward.  Including the length removes concatenation
    # ambiguity and lets the offline merger verify token identity across roles.
    digest = hashlib.sha256()
    digest.update(f"{values.numel()}:".encode("ascii"))
    digest.update(",".join(str(int(x)) for x in values.tolist()).encode("ascii"))
    return digest.hexdigest()


def _resolve_output_path(template: Any, tp_rank: int) -> Path:
    if not isinstance(template, str) or template.count("{tp_rank}") != 1:
        raise ValueError(
            "out_path must be an absolute .pt template containing exactly one "
            "'{tp_rank}' placeholder"
        )
    rendered = template.replace("{tp_rank}", str(tp_rank))
    if "{" in rendered or "}" in rendered:
        raise ValueError("out_path contains an unsupported format placeholder")
    output_path = Path(rendered)
    if not output_path.is_absolute() or output_path.suffix != ".pt":
        raise ValueError("resolved out_path must be an absolute path ending in .pt")
    parent = output_path.parent
    if not parent.is_dir():
        raise ValueError(f"out_path parent does not exist: {parent}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite drift capture: {output_path}")
    return output_path


def parse_drift_profile_plan(
    plan: Mapping[str, Any],
    *,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    logical_seq_len: int,
    batch_size: int,
    is_extend: bool,
    tp_rank: int,
    tp_size: int,
    pp_size: int,
    num_layers: int,
    fp8_wo_a: bool,
    attention_backend: str,
    sparse_ffn: bool,
) -> DriftProfilePlan:
    """Validate a runtime plan and map segment-local rows to model rows.

    ``positions`` must describe the entire logical prompt as ``0..N-1``.  This
    intentionally rejects scheduler chunking and prefix-cache hits.
    """

    if not isinstance(plan, Mapping) or plan.get("mode") != DRIFT_PROFILE_MODE:
        raise ValueError("plan.mode must be 'drift_profile'")
    if _strict_int("batch_size", batch_size, minimum=1) != 1:
        raise ValueError("drift profiling supports exactly one request")
    if is_extend is not True:
        raise ValueError("drift profiling supports only ForwardMode.EXTEND")
    if fp8_wo_a:
        raise ValueError(
            "drift profiling does not support SGLANG_OPT_FP8_WO_A_GEMM; "
            "restart with the non-FP8 wo_a path"
        )
    if not isinstance(attention_backend, str) or attention_backend.lower() != "dsv4":
        raise ValueError(
            "projected-head drift requires the all-online DSV4 prefill backend; "
            f"got {attention_backend!r}"
        )
    if sparse_ffn is not False:
        raise ValueError(
            "projected-head drift requires the dense all-online FFN oracle; "
            "disable redknot_sparse_ffn_enable"
        )
    rank = _strict_int("tp_rank", tp_rank)
    size = _strict_int("tp_size", tp_size, minimum=1)
    if size != SUPPORTED_TP_SIZE or rank >= size:
        raise ValueError(
            f"drift profiling requires TP={SUPPORTED_TP_SIZE}, got rank={rank}, "
            f"tp_size={size}"
        )
    if _strict_int("pp_size", pp_size, minimum=1) != 1:
        raise ValueError("drift profiling requires PP=1")
    layers = _strict_int("num_layers", num_layers, minimum=1)
    seq_len = _strict_int("logical_seq_len", logical_seq_len, minimum=1)

    run_id = _strict_name("run_id", plan.get("run_id"))
    calibration_digest = plan.get("calibration_digest")
    if not isinstance(calibration_digest, str) or not _SHA256_HEX.fullmatch(
        calibration_digest
    ):
        raise ValueError(
            "calibration_digest must be a lowercase 64-hex SHA-256 shared by "
            "every snapshot/oracle request in this calibration"
        )
    role = plan.get("role")
    if role not in ("snapshot", "oracle"):
        raise ValueError("role must be exactly 'snapshot' or 'oracle'")
    output_path = _resolve_output_path(plan.get("out_path"), rank)

    positions_cpu = _cpu_int_vector("positions", positions)
    input_ids_cpu = _cpu_int_vector("input_ids", input_ids)
    if positions_cpu.numel() != seq_len or input_ids_cpu.numel() != seq_len:
        raise ValueError(
            "drift profiling requires a complete unchunked EXTEND: positions, "
            "input_ids, and logical_seq_len must have identical lengths"
        )
    expected_positions = torch.arange(seq_len, dtype=torch.int64)
    if not torch.equal(positions_cpu, expected_positions):
        raise ValueError(
            "drift profiling requires dense positions 0..logical_seq_len-1; "
            "disable chunked prefill and prefix/radix cache reuse"
        )

    raw_segments = plan.get("segments")
    if (
        isinstance(raw_segments, (str, bytes))
        or not isinstance(raw_segments, Sequence)
        or not raw_segments
    ):
        raise ValueError("segments must be a non-empty sequence")
    raw_sample_rows = plan.get("sample_rows")
    if not isinstance(raw_sample_rows, Mapping):
        raise ValueError("sample_rows must map every segment id to local rows")

    parsed = []
    seen_ids = set()
    previous_end = 0
    flattened_indices = []
    segment_slices: Dict[str, Tuple[int, int]] = {}
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ValueError(f"segments[{index}] must be a mapping")
        segment_id = _strict_name(f"segments[{index}].id", raw.get("id"))
        if segment_id in seen_ids:
            raise ValueError(f"duplicate segment id: {segment_id}")
        seen_ids.add(segment_id)
        start = _strict_int(f"segments[{index}].start", raw.get("start"))
        length = _strict_int(f"segments[{index}].length", raw.get("length"), minimum=1)
        end = start + length
        if end > seq_len:
            raise ValueError(
                f"segment {segment_id!r} ends at {end}, beyond sequence {seq_len}"
            )
        if index and start < previous_end:
            raise ValueError("segments must be ordered by start and non-overlapping")
        previous_end = end

        raw_rows = raw_sample_rows.get(segment_id)
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
            raise ValueError(
                f"sample_rows[{segment_id!r}] must be a non-empty sequence"
            )
        rows = tuple(
            _strict_int(f"sample_rows[{segment_id!r}][{row_index}]", value)
            for row_index, value in enumerate(raw_rows)
        )
        if not rows:
            raise ValueError(f"sample_rows[{segment_id!r}] cannot be empty")
        if tuple(sorted(set(rows))) != rows:
            raise ValueError(
                f"sample_rows[{segment_id!r}] must be strictly increasing and unique"
            )
        if rows[-1] >= length:
            raise ValueError(
                f"sample row {rows[-1]} is outside segment {segment_id!r} "
                f"length={length}"
            )
        declared_hash = raw.get("token_ids_sha256")
        actual_hash = _hash_token_ids(input_ids_cpu[start:end])
        if declared_hash is not None and declared_hash != actual_hash:
            raise ValueError(f"token_ids_sha256 mismatch for segment {segment_id!r}")

        slice_start = len(flattened_indices)
        flattened_indices.extend(start + row for row in rows)
        slice_end = len(flattened_indices)
        segment_slices[segment_id] = (slice_start, slice_end)
        parsed.append(
            DriftSegment(
                segment_id=segment_id,
                start=start,
                length=length,
                sample_rows=rows,
                token_ids_sha256=actual_hash,
            )
        )

    extra_sample_ids = set(raw_sample_rows) - seen_ids
    if extra_sample_ids:
        raise ValueError(
            f"sample_rows contains unknown segment ids: {sorted(extra_sample_ids)}"
        )

    return DriftProfilePlan(
        run_id=run_id,
        calibration_digest=calibration_digest,
        role=role,
        output_path=output_path,
        tp_rank=rank,
        tp_size=size,
        num_layers=layers,
        logical_seq_len=seq_len,
        segments=tuple(parsed),
        sample_indices=tuple(flattened_indices),
        segment_slices=segment_slices,
    )


class DriftProfileSession:
    """One active rank-local capture; not reusable after finish/abort."""

    def __init__(self, plan: DriftProfilePlan) -> None:
        self.plan = plan
        self._layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self._index_by_device: Dict[torch.device, torch.Tensor] = {}
        self._projection_rank: Optional[int] = None
        self._num_output_groups: Optional[int] = None
        self._source_dtype: Optional[str] = None
        self._sealed = False

    def _device_indices(self, device: torch.device) -> torch.Tensor:
        cached = self._index_by_device.get(device)
        if cached is None:
            cached = torch.tensor(
                self.plan.sample_indices, dtype=torch.long, device=device
            )
            self._index_by_device[device] = cached
        return cached

    def capture_layer(
        self,
        *,
        layer_id: int,
        per_head_output: torch.Tensor,
        wo_a_weight: torch.Tensor,
        num_local_groups: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        if self._sealed:
            raise RuntimeError("drift profile session is sealed")
        layer = _strict_int("layer_id", layer_id)
        if layer >= self.plan.num_layers:
            raise IndexError(f"layer_id={layer} is outside [0, {self.plan.num_layers})")
        if layer in self._layers:
            raise RuntimeError(f"layer {layer} was captured more than once")
        if tp_rank != self.plan.tp_rank or tp_size != self.plan.tp_size:
            raise RuntimeError(
                "attention TP topology differs from the ModelRunner topology: "
                f"attention=({tp_rank}, {tp_size}), runner="
                f"({self.plan.tp_rank}, {self.plan.tp_size})"
            )
        if not torch.is_tensor(per_head_output) or per_head_output.ndim != 3:
            raise ValueError("per_head_output must have shape [tokens, heads, dim]")
        if per_head_output.shape[0] != self.plan.logical_seq_len:
            raise ValueError(
                "layer output rows do not cover the complete logical sequence: "
                f"got {per_head_output.shape[0]}, expected {self.plan.logical_seq_len}"
            )
        local_heads = int(per_head_output.shape[1])
        if local_heads != SUPPORTED_HEADS_PER_RANK:
            raise ValueError(
                f"TP8 drift profiling requires {SUPPORTED_HEADS_PER_RANK} local "
                f"heads, got {local_heads}"
            )
        if local_heads * self.plan.tp_size != SUPPORTED_GLOBAL_HEADS:
            raise ValueError(
                f"drift profiling requires {SUPPORTED_GLOBAL_HEADS} global heads"
            )
        if per_head_output.device != wo_a_weight.device:
            raise ValueError("per_head_output and wo_a_weight must share a device")
        if per_head_output.dtype != wo_a_weight.dtype:
            raise ValueError("per_head_output and wo_a_weight must share a dtype")
        if not per_head_output.dtype.is_floating_point:
            raise ValueError("per_head_output must be floating point")
        if wo_a_weight.ndim != 3:
            raise ValueError(
                "wo_a_weight must have shape [local_groups, rank, group_input]"
            )
        local_groups = _strict_int("num_local_groups", num_local_groups, minimum=1)
        if int(wo_a_weight.shape[0]) != local_groups:
            raise ValueError(
                "MQALayer.n_local_groups disagrees with wo_a weight shape: "
                f"layer={local_groups}, weight={int(wo_a_weight.shape[0])}"
            )
        if local_groups <= 0 or local_heads % local_groups != 0:
            raise ValueError(
                "wo_a local output groups must be positive and divide the "
                f"eight local heads, got n_local_groups={local_groups}"
            )
        if self._num_output_groups is None:
            self._num_output_groups = local_groups
        elif local_groups != self._num_output_groups:
            raise RuntimeError("wo_a local output-group count changed between layers")

        sampled_output = per_head_output.index_select(
            0, self._device_indices(per_head_output.device)
        )
        with torch.no_grad():
            projected = decompose_wo_a_per_head(sampled_output, wo_a_weight)
        if projected.shape[1] != SUPPORTED_HEADS_PER_RANK:
            raise RuntimeError("per-head decomposition returned an invalid head count")
        projection_rank = int(projected.shape[2])
        if self._projection_rank is None:
            self._projection_rank = projection_rank
            self._source_dtype = str(projected.dtype)
        elif projection_rank != self._projection_rank:
            raise RuntimeError(
                f"layer {layer} projection rank={projection_rank} differs from "
                f"the first layer rank={self._projection_rank}"
            )

        per_segment: Dict[str, torch.Tensor] = {}
        for segment in self.plan.segments:
            begin, end = self.plan.segment_slices[segment.segment_id]
            # Keep the sampled tensor on device until request completion.  The
            # only host transfer is the final all-layer materialization.
            per_segment[segment.segment_id] = projected[begin:end].detach()
        self._layers[layer] = per_segment

    def _build_artifact(self) -> Dict[str, Any]:
        expected_layers = set(range(self.plan.num_layers))
        missing = expected_layers - set(self._layers)
        extra = set(self._layers) - expected_layers
        if missing or extra:
            raise RuntimeError(
                f"incomplete drift capture: missing_layers={sorted(missing)} "
                f"extra_layers={sorted(extra)}"
            )
        if (
            self._projection_rank is None
            or self._num_output_groups is None
            or self._source_dtype is None
        ):
            raise RuntimeError("drift capture contains no projections")

        layers: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        for layer_id in range(self.plan.num_layers):
            layer_segments: Dict[str, Dict[str, torch.Tensor]] = {}
            for segment in self.plan.segments:
                values = (
                    self._layers[layer_id][segment.segment_id]
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                if not bool(torch.isfinite(values).all().item()):
                    raise RuntimeError(
                        f"non-finite projection at layer={layer_id}, "
                        f"segment={segment.segment_id}"
                    )
                layer_segments[segment.segment_id] = {
                    "row_ids": torch.tensor(segment.sample_rows, dtype=torch.int64),
                    "projection": values,
                }
            layers[layer_id] = layer_segments

        head_start = self.plan.tp_rank * SUPPORTED_HEADS_PER_RANK
        group_start = self.plan.tp_rank * self._num_output_groups
        capture_digest = hashlib.sha256()
        for segment in self.plan.segments:
            # Deliberately exclude absolute ``start`` and role: a standalone
            # snapshot and its composed oracle can place identical segment
            # tokens at different global positions.  Everything that defines
            # the paired sampled content remains covered.
            capture_digest.update(segment.segment_id.encode("utf-8"))
            capture_digest.update(f":{segment.length}:".encode("ascii"))
            capture_digest.update(
                ",".join(str(row) for row in segment.sample_rows).encode("ascii")
            )
            capture_digest.update(segment.token_ids_sha256.encode("ascii"))
        return {
            "format": RANK_CAPTURE_FORMAT,
            "run_id": self.plan.run_id,
            # The paired profiler calls this calibration_id.  The runtime plan
            # uses run_id as the one shared identity across snapshot/oracle
            # roles, so publish both names without introducing two authorities.
            "calibration_id": self.plan.run_id,
            # calibration_digest is supplied by the harness after hashing the
            # complete cross-request calibration manifest.  capture_digest is
            # local evidence for exactly the segments in this one request.
            "calibration_digest": self.plan.calibration_digest,
            "capture_digest": capture_digest.hexdigest(),
            "role": self.plan.role,
            "tp_rank": self.plan.tp_rank,
            "tp_size": self.plan.tp_size,
            "source_tp_rank": self.plan.tp_rank,
            "source_tp_world_size": self.plan.tp_size,
            "represented_tp_ranks": [self.plan.tp_rank],
            "num_layers": self.plan.num_layers,
            "num_attention_heads": SUPPORTED_GLOBAL_HEADS,
            "local_head_start": head_start,
            "local_head_end": head_start + SUPPORTED_HEADS_PER_RANK,
            "global_head_ids": list(
                range(head_start, head_start + SUPPORTED_HEADS_PER_RANK)
            ),
            "heads_per_rank": SUPPORTED_HEADS_PER_RANK,
            "num_local_heads": SUPPORTED_HEADS_PER_RANK,
            "n_local_groups": self._num_output_groups,
            "num_output_groups": self._num_output_groups * self.plan.tp_size,
            "local_output_group_ids": list(
                range(group_start, group_start + self._num_output_groups)
            ),
            "heads_per_output_group": (
                SUPPORTED_HEADS_PER_RANK // self._num_output_groups
            ),
            "projection_rank": self._projection_rank,
            "source_dtype": self._source_dtype,
            "logical_seq_len": self.plan.logical_seq_len,
            "segments": [
                {
                    "id": item.segment_id,
                    "start": item.start,
                    "length": item.length,
                    "sample_rows": list(item.sample_rows),
                    "token_ids_sha256": item.token_ids_sha256,
                }
                for item in self.plan.segments
            ],
            "layers": layers,
        }

    def finish(self) -> Path:
        if self._sealed:
            raise RuntimeError("drift profile session is already sealed")
        self._sealed = True
        try:
            artifact = self._build_artifact()
            output_path = self.plan.output_path
            if output_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite drift capture: {output_path}"
                )
            temporary = output_path.with_name(
                f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            )
            try:
                with temporary.open("xb") as handle:
                    torch.save(artifact, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Atomic publication: readers see either no artifact or the
                # complete all-layer artifact.
                os.replace(temporary, output_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return output_path
        finally:
            self._layers.clear()
            self._index_by_device.clear()

    def abort(self) -> None:
        self._sealed = True
        self._layers.clear()
        self._index_by_device.clear()


_ACTIVE_LOCK = threading.Lock()
_ACTIVE_SESSION: Optional[DriftProfileSession] = None


def begin_drift_profile_request(
    plan: Mapping[str, Any],
    **runtime: Any,
) -> DriftProfileSession:
    """Validate and install the sole active request in this worker process."""

    parsed = parse_drift_profile_plan(plan, **runtime)
    session = DriftProfileSession(parsed)
    global _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        if _ACTIVE_SESSION is not None:
            raise RuntimeError("another drift profile request is already active")
        _ACTIVE_SESSION = session
    return session


def drift_profile_request_is_active() -> bool:
    with _ACTIVE_LOCK:
        return _ACTIVE_SESSION is not None


def capture_active_drift_layer(
    *,
    layer_id: int,
    per_head_output: torch.Tensor,
    wo_a_weight: torch.Tensor,
    num_local_groups: int,
    tp_rank: int,
    tp_size: int,
) -> None:
    """Capture one layer for the current request, rejecting absent state."""

    with _ACTIVE_LOCK:
        session = _ACTIVE_SESSION
    if session is None:
        raise RuntimeError("no active drift profile request")
    session.capture_layer(
        layer_id=layer_id,
        per_head_output=per_head_output,
        wo_a_weight=wo_a_weight,
        num_local_groups=num_local_groups,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def _take_active(expected: DriftProfileSession) -> DriftProfileSession:
    global _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        if _ACTIVE_SESSION is not expected:
            raise RuntimeError("active drift profile session identity mismatch")
        session = _ACTIVE_SESSION
        _ACTIVE_SESSION = None
    return session


def finish_drift_profile_request(session: DriftProfileSession) -> Path:
    """Clear global state and atomically publish this rank's one artifact."""

    return _take_active(session).finish()


def abort_drift_profile_request(session: DriftProfileSession) -> None:
    """Clear global state and discard every in-memory projection."""

    _take_active(session).abort()
