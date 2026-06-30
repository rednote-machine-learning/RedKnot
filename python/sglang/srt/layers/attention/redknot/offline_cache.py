# Copyright 2024-2026 SGLang RedKnot Integration.
"""Offline KV cache storage for RedKnot segments.

The classic RedKnot pipeline pre-fills each document segment once and
reuses the resulting (K, V) tensors many times across requests. This module
provides a small, thread-safe store on top of sglang's runtime that is:

- **Segment-keyed**: identified by a hash over ``(model_id, segment_token_ids,
  prepend_bos)`` so we never accidentally cross-mix caches between models.
- **Multi-tier**: caches live on the device that produced them (typically
  ``cuda:0``); a CPU mirror is kept so we can offload under memory pressure
  and re-upload on demand without recomputation.
- **LRU bounded**: an optional ``max_bytes`` ceiling triggers eviction.

The store is intentionally decoupled from sglang's ``TokenToKVPool`` because
RedKnot segments do not live in the token-paged address space — they are
whole, position-zeroed segment KV blobs that need RoPE realignment when
they are spliced into a real request.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class OfflineSegment:
    """One pre-filled segment.

    Attributes
    ----------
    segment_id:
        Stable hash key returned by :meth:`OfflineKVCache.compute_segment_id`.
    token_ids:
        ``[L]`` int tensor on CPU (kept for debug + recomputation).
    doc_len:
        Number of tokens (== ``len(token_ids)``).
    kv:
        ``num_layers``-long list of ``(K, V)`` tuples. Each tensor is shaped
        ``[1, num_kv_heads, L, head_dim]`` and rotated under positions
        ``[0, L)``. May live on CPU or device depending on policy.
    device:
        Current physical location of ``kv`` tensors.
    nbytes:
        Total byte size across all (K, V) tensors. Used for LRU accounting.
    """

    segment_id: str
    token_ids: torch.Tensor
    doc_len: int
    kv: List[Tuple[torch.Tensor, torch.Tensor]]
    device: torch.device
    nbytes: int
    meta: Dict[str, object] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────────
class OfflineKVCache:
    """LRU store of pre-filled segments.

    Concurrency
    -----------
    All mutating ops take ``self._lock``. CUDA streams used to copy KV
    between host and device are *not* serialised — callers are expected to
    issue copies on the model stream.

    Eviction
    --------
    Eviction policy is strict LRU on device residency. Evicted entries fall
    back to CPU automatically; only when the CPU mirror also exceeds
    ``max_host_bytes`` (if set) are entries removed entirely.
    """

    def __init__(
        self,
        max_device_bytes: Optional[int] = None,
        max_host_bytes: Optional[int] = None,
    ) -> None:
        self._segments: "OrderedDict[str, OfflineSegment]" = OrderedDict()
        self._lock = threading.RLock()
        self.max_device_bytes = max_device_bytes
        self.max_host_bytes = max_host_bytes
        # Running totals to avoid scanning the dict on every put.
        self._device_bytes = 0
        self._host_bytes = 0

    # ────────────────────────────────────────────────────────────────
    # Key derivation
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_segment_id(
        model_id: str,
        token_ids: Sequence[int],
        *,
        prepend_bos: bool = False,
        extra: str = "",
    ) -> str:
        """Deterministic key for a segment.

        We hash the *token ids* rather than the source text so the cache
        keeps working under tokenizer-level changes that produce identical
        ids (e.g. whitespace normalisation).
        """
        h = hashlib.blake2b(digest_size=20)
        h.update(model_id.encode("utf-8"))
        h.update(b"|bos=" + (b"1" if prepend_bos else b"0"))
        if extra:
            h.update(b"|x=" + extra.encode("utf-8"))
        # Pack token ids as little-endian int32 for speed.
        tok_arr = torch.as_tensor(list(token_ids), dtype=torch.int32)
        h.update(tok_arr.numpy().tobytes())
        return h.hexdigest()

    # ────────────────────────────────────────────────────────────────
    # CRUD
    # ────────────────────────────────────────────────────────────────
    def put(self, segment: OfflineSegment) -> None:
        with self._lock:
            if segment.segment_id in self._segments:
                # Promote.
                self._segments.move_to_end(segment.segment_id)
                return
            self._segments[segment.segment_id] = segment
            if segment.device.type == "cuda":
                self._device_bytes += segment.nbytes
            else:
                self._host_bytes += segment.nbytes
            self._enforce_limits()

    def get(self, segment_id: str) -> Optional[OfflineSegment]:
        with self._lock:
            seg = self._segments.get(segment_id)
            if seg is not None:
                self._segments.move_to_end(seg.segment_id)
            return seg

    def has(self, segment_id: str) -> bool:
        with self._lock:
            return segment_id in self._segments

    def drop(self, segment_id: str) -> None:
        with self._lock:
            seg = self._segments.pop(segment_id, None)
            if seg is None:
                return
            if seg.device.type == "cuda":
                self._device_bytes -= seg.nbytes
            else:
                self._host_bytes -= seg.nbytes

    # ────────────────────────────────────────────────────────────────
    # Residency management
    # ────────────────────────────────────────────────────────────────
    def to_device(
        self, segment_id: str, device: torch.device
    ) -> Optional[OfflineSegment]:
        """Ensure the segment's KV lives on ``device``; copy if necessary."""
        with self._lock:
            seg = self._segments.get(segment_id)
            if seg is None:
                return None
            if seg.device == device:
                self._segments.move_to_end(segment_id)
                return seg
            new_kv = [
                (
                    (k.to(device, non_blocking=True), v.to(device, non_blocking=True))
                    if k is not None and v is not None
                    else (k, v)
                )
                for k, v in seg.kv
            ]
            if seg.device.type == "cuda" and device.type != "cuda":
                self._device_bytes -= seg.nbytes
                self._host_bytes += seg.nbytes
            elif seg.device.type != "cuda" and device.type == "cuda":
                self._host_bytes -= seg.nbytes
                self._device_bytes += seg.nbytes
            seg.kv = new_kv
            seg.device = device
            self._segments.move_to_end(segment_id)
            self._enforce_limits()
            return seg

    def offload(self, segment_id: str) -> None:
        """Move a segment from device to CPU without dropping it."""
        self.to_device(segment_id, torch.device("cpu"))

    # ────────────────────────────────────────────────────────────────
    # Stats
    # ────────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "n_segments": len(self._segments),
                "device_bytes": self._device_bytes,
                "host_bytes": self._host_bytes,
            }

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()
            self._device_bytes = 0
            self._host_bytes = 0

    # ────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────
    def _enforce_limits(self) -> None:
        """Evict least-recently-used segments to respect byte ceilings."""
        if self.max_device_bytes is not None:
            for sid in list(self._segments.keys()):
                if self._device_bytes <= self.max_device_bytes:
                    break
                seg = self._segments[sid]
                if seg.device.type == "cuda":
                    # Offload to CPU first (don't delete the entry).
                    self.to_device(sid, torch.device("cpu"))
        if self.max_host_bytes is not None:
            for sid in list(self._segments.keys()):
                if (
                    self._device_bytes + self._host_bytes
                    <= (self.max_device_bytes or 0) + self.max_host_bytes
                ):
                    break
                seg = self._segments.pop(sid)
                if seg.device.type == "cuda":
                    self._device_bytes -= seg.nbytes
                else:
                    self._host_bytes -= seg.nbytes
                logger.debug("RedKnot: evicted offline segment %s", sid[:12])


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def kv_nbytes(kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Total byte count for a per-layer KV list. Skips (None, None)
    placeholders used for layers the full-attention backend never reads."""
    n = 0
    for k, v in kv:
        if k is None or v is None:
            continue
        n += k.numel() * k.element_size()
        n += v.numel() * v.element_size()
    return n


def build_offline_segment(
    *,
    segment_id: str,
    token_ids: torch.Tensor,
    kv: List[Tuple[torch.Tensor, torch.Tensor]],
) -> OfflineSegment:
    """Construct an :class:`OfflineSegment` from raw materials, deriving
    derived fields (``nbytes``, ``device``)."""
    assert kv, "build_offline_segment: kv list is empty"
    # Some entries may be (None, None) placeholders (e.g. linear layers in a
    # hybrid model that the full-attention backend never reads). Derive device
    # and byte count from the first materialised entry.
    first = next((pair for pair in kv if pair[0] is not None), None)
    assert first is not None, "build_offline_segment: all kv entries are None"
    device = first[0].device
    nbytes = kv_nbytes(kv)
    return OfflineSegment(
        segment_id=segment_id,
        token_ids=token_ids.to("cpu"),
        doc_len=int(token_ids.shape[-1]),
        kv=kv,
        device=device,
        nbytes=nbytes,
    )


# Process-wide singleton for convenience. Tests can replace it.
_GLOBAL_CACHE: Optional[OfflineKVCache] = None
_GLOBAL_LOCK = threading.Lock()


def get_global_offline_cache() -> OfflineKVCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_CACHE is None:
                _GLOBAL_CACHE = OfflineKVCache()
    return _GLOBAL_CACHE


def set_global_offline_cache(cache: OfflineKVCache) -> None:
    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        _GLOBAL_CACHE = cache
