"""Low-intrusion CUDA-event timing for RedKnot DeepSeek-V4 restore forwards.

The profiler is deliberately dormant unless ``REDKNOT_V4_TIMING=1`` and a
restore forward explicitly opens a session.  Region boundaries only enqueue
CUDA events and collect host timestamps.  ``finish_forward`` performs the
single device synchronization for the whole forward and emits one compact JSON
record on global rank zero.

This module must remain dependency-light: it is imported from the model,
attention backend, and model runner.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch


logger = logging.getLogger(__name__)

_ENV_NAME = "REDKNOT_V4_TIMING"
_LOG_PREFIX = "REDKNOT_V4_TIMING_JSON"
_STATE = threading.local()


@dataclass
class _Region:
    phase: str
    layer_id: Optional[int]
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    cpu_start_ns: int
    cpu_end_ns: int = 0


@dataclass
class _Session:
    metadata: Dict[str, object]
    device: torch.device
    start_event: torch.cuda.Event
    cpu_start_ns: int
    regions: List[_Region] = field(default_factory=list)


def redknot_v4_timing_enabled() -> bool:
    """Return whether event timing was explicitly requested."""

    return os.environ.get(_ENV_NAME, "0") == "1"


def _active_session() -> Optional[_Session]:
    return getattr(_STATE, "session", None)


def _rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return 0


def begin_forward(
    *,
    device: torch.device,
    rows: int,
    forward_mode: str,
    batch_size: int,
) -> bool:
    """Open a timing session and enqueue its root event.

    No synchronization or tensor-to-host conversion is performed here.  A
    stale session is cleared before CUDA setup.  Once timing is explicitly
    requested, setup failures propagate so an unprofiled request cannot be
    mistaken for a valid calibration sample.
    """

    if not redknot_v4_timing_enabled():
        _STATE.session = None
        return False
    # Explicit timing is fail-closed.  Clear any stale session before touching
    # CUDA so a setup failure cannot leave later no-op regions recording into a
    # previous forward.
    _STATE.session = None
    if not torch.cuda.is_available():
        raise RuntimeError("REDKNOT_V4_TIMING=1 requires CUDA")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError(
            "REDKNOT_V4_TIMING=1 requires a CUDA forward device"
        )
    start_event = None
    try:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream(resolved_device))
        _STATE.session = _Session(
            metadata={
                "rows": int(rows),
                "forward_mode": str(forward_mode),
                "batch_size": int(batch_size),
                "rank": _rank(),
            },
            device=resolved_device,
            start_event=start_event,
            cpu_start_ns=time.perf_counter_ns(),
        )
    except BaseException:
        _STATE.session = None
        start_event = None
        raise
    return True


def abort_forward() -> None:
    """Drop pending diagnostic state without synchronizing."""

    _STATE.session = None


@contextlib.contextmanager
def timed(
    phase: str,
    *,
    layer_id: Optional[int] = None,
) -> Iterator[None]:
    """Enqueue a start/end event pair for one phase.

    Event allocation and recording happen only while a restore-forward session
    is active.  In the default path this is exactly a null context manager.
    """

    session = _active_session()
    if session is None:
        yield
        return

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(session.device)
    start_event.record(stream)
    region = _Region(
        phase=str(phase),
        layer_id=None if layer_id is None else int(layer_id),
        start_event=start_event,
        end_event=end_event,
        cpu_start_ns=time.perf_counter_ns(),
    )
    try:
        yield
    finally:
        region.cpu_end_ns = time.perf_counter_ns()
        # CUDA stream context managers restore the caller's stream before this
        # point.  Record on the current stream so the event brackets work that
        # is visible to the model's normal dependency chain.
        try:
            end_event.record(torch.cuda.current_stream(session.device))
            session.regions.append(region)
        finally:
            # Do not retain CUDA objects in a propagated exception traceback;
            # the session owns successful regions until finish/abort.
            start_event = None
            end_event = None
            region = None
            session = None
            stream = None


def _add_sample(
    bucket: Dict[str, Dict[str, float]],
    key: str,
    *,
    gpu_ms: float,
    cpu_ms: float,
) -> None:
    sample = bucket.setdefault(key, {"gpu_ms": 0.0, "cpu_ms": 0.0, "calls": 0})
    sample["gpu_ms"] += float(gpu_ms)
    sample["cpu_ms"] += float(cpu_ms)
    sample["calls"] += 1


def _rounded_bucket(
    bucket: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, object]]:
    return {
        key: {
            "gpu_ms": round(float(value["gpu_ms"]), 3),
            "cpu_ms": round(float(value["cpu_ms"]), 3),
            "calls": int(value["calls"]),
        }
        for key, value in sorted(bucket.items())
    }


def finish_forward() -> Optional[Dict[str, object]]:
    """Synchronize once, aggregate all event pairs, and emit one JSON line."""

    session = _active_session()
    _STATE.session = None
    if session is None:
        return None

    end_event = None
    region = None
    try:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(torch.cuda.current_stream(session.device))
        cpu_end_ns = time.perf_counter_ns()
        # The one intentional diagnostic synchronization.  Device-wide sync also
        # covers kernels launched on auxiliary streams before their normal waits.
        torch.cuda.synchronize(session.device)

        phases: Dict[str, Dict[str, float]] = {}
        layers: Dict[str, Dict[str, float]] = {}
        for region in session.regions:
            gpu_ms = float(region.start_event.elapsed_time(region.end_event))
            cpu_ms = (
                max(0, region.cpu_end_ns - region.cpu_start_ns) / 1_000_000.0
            )
            _add_sample(phases, region.phase, gpu_ms=gpu_ms, cpu_ms=cpu_ms)
            if region.layer_id is not None:
                _add_sample(
                    layers,
                    f"{region.layer_id}:{region.phase}",
                    gpu_ms=gpu_ms,
                    cpu_ms=cpu_ms,
                )

        payload: Dict[str, object] = {
            "schema": "redknot_v4_timing_v1",
            **session.metadata,
            "forward_gpu_ms": round(
                float(session.start_event.elapsed_time(end_event)), 3
            ),
            "forward_cpu_ms": round(
                max(0, cpu_end_ns - session.cpu_start_ns) / 1_000_000.0, 3
            ),
            "region_count": len(session.regions),
            "phases": _rounded_bucket(phases),
            "layers": _rounded_bucket(layers),
        }
        if int(session.metadata.get("rank", 0)) == 0:
            logger.info(
                "%s %s",
                _LOG_PREFIX,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
        return payload
    finally:
        # Session ownership is already detached from TLS.  Clear the remaining
        # event references even when Event.record, synchronize, elapsed_time, or
        # logging raises and the original traceback is propagated.
        session.regions.clear()
        session = None
        end_event = None
        region = None


__all__ = [
    "abort_forward",
    "begin_forward",
    "finish_forward",
    "redknot_v4_timing_enabled",
    "timed",
]
