# Copyright 2024-2026 SGLang RedKnot Integration.
"""DeepSeek-V4 offline MLA KV reuse controller (in scheduler process).

Implements the RedKnot offline-reuse pipeline for DSV4:

  OFFLINE  per segment, prefill at LOCAL positions [0, L) and snapshot the
           per-layer SWA KV (the packed nope_fp8 + rope_bf16 records) for the
           segment's token slots. Stored keyed by a hash of the segment tokens.

  ONLINE   when a concatenated request reuses cached segments:
             * segment 1 KV is used verbatim (its local positions already match
               the global positions [0, L1) because it is first);
             * from segment 2 on, the cached KV's rope portion is RE-ROTATED
               from local positions to the segment's global offset, and the
               first ``boundary=128`` tokens of each segment are RECOMPUTED
               online (SWA window crossing the segment boundary).

Because the DSV4 SWA cache only keeps the most recent ``swa_window`` tokens per
query, recomputing the first 128 tokens of each segment is what lets the window
"see" across the segment join; the rest of each segment is reused.

This module lives in the scheduler process so it can touch the
``DeepSeekV4TokenToKVPool`` buffers directly.

The controller is intentionally stateless w.r.t. SGLang scheduling: it is driven
explicitly by the dsv4 backend at well-defined hook points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.redknot.dsv4_rope_reloc import (
    NOPE_DIM,
    ROPE_DIM,
    read_rope_bf16,
    reposition_rope,
    write_rope_bf16,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentSnapshot:
    """Per-segment snapshot of SWA KV (rope portion only; nope is reused as-is).

    We snapshot only the rope_bf16[L, 64] per layer because:
      * nope_fp8 + scale are position-independent -> reused verbatim from the
        radix-cached slots, no copy needed;
      * rope_bf16 is what we re-rotate. Keeping a CPU/GPU copy of the *local*
        rope lets us re-derive any global offset losslessly (we always rotate
        from the stored local positions, never compounding).
    """

    seg_hash: str
    length: int
    # rope_local[layer] : [L, 64] bf16, rotated at local positions [0, L)
    rope_local: Dict[int, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)


class DSV4OfflineReuseController:
    """Owns offline segment snapshots and applies online RoPE relocation."""

    def __init__(self, swa_window: int = 128, boundary: int = 128):
        self.swa_window = swa_window
        self.boundary = boundary
        self._segments: Dict[str, SegmentSnapshot] = {}
        self._enabled = False
        # Per-request plan, set right before a concatenated forward.
        # plan: list of (seg_hash, global_offset, local_len, slot_indices_per_layer)
        self._active_plan: Optional[List[dict]] = None
        # stats
        self.stats = {
            "segments_cached": 0,
            "reuse_hits": 0,
            "tokens_reused": 0,
            "tokens_recomputed": 0,
        }

    # ──────────────────────────────────────────────────────────────────
    # Enablement
    # ──────────────────────────────────────────────────────────────────
    def enable(self):
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ──────────────────────────────────────────────────────────────────
    # OFFLINE: snapshot a segment's rope KV after it is prefilled
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def snapshot_segment(
        self,
        seg_hash: str,
        length: int,
        layer_id: int,
        kv_buffer: torch.Tensor,
        slot_indices: torch.Tensor,
        page_size: int,
    ) -> None:
        """Store the rope_bf16[L,64] for one layer of a freshly-prefilled segment.

        slot_indices: [L] int — KV-pool slots holding this segment's tokens, in
        token order. kv_buffer: the layer's [num_pages, numel_per_page] uint8.
        """
        seg = self._segments.get(seg_hash)
        if seg is None:
            seg = SegmentSnapshot(seg_hash=seg_hash, length=length)
            self._segments[seg_hash] = seg
            self.stats["segments_cached"] += 1
        rope = read_rope_bf16(kv_buffer, slot_indices, page_size)  # [L,64]
        seg.rope_local[layer_id] = rope.detach().clone()

    def has_segment(self, seg_hash: str) -> bool:
        return seg_hash in self._segments

    def get_segment(self, seg_hash: str) -> Optional[SegmentSnapshot]:
        return self._segments.get(seg_hash)

    # ──────────────────────────────────────────────────────────────────
    # ONLINE: relocate a reused segment's rope into target slots
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def relocate_into_slots(
        self,
        seg_hash: str,
        layer_id: int,
        kv_buffer: torch.Tensor,
        dst_slots: torch.Tensor,
        global_offset: int,
        freqs_cis: torch.Tensor,
        page_size: int,
        skip_first: int = 0,
    ) -> int:
        """Write a reused segment's rope (relocated to global positions) into
        ``dst_slots``.

        The reused tokens are ``[skip_first, length)`` of the segment (the first
        ``skip_first`` tokens are recomputed online and must NOT be overwritten).
        Their local positions are ``[skip_first, length)`` and the target global
        positions are ``[global_offset+skip_first, global_offset+length)``.

        Returns the number of tokens written (reused).
        """
        seg = self._segments.get(seg_hash)
        if seg is None or layer_id not in seg.rope_local:
            return 0
        L = seg.length
        if skip_first >= L:
            return 0
        rope_local = seg.rope_local[layer_id][skip_first:L].to(kv_buffer.device)
        n = rope_local.shape[0]
        src_pos = torch.arange(skip_first, L, device=kv_buffer.device, dtype=torch.long)
        dst_pos = torch.arange(
            global_offset + skip_first,
            global_offset + L,
            device=kv_buffer.device,
            dtype=torch.long,
        )
        rope_reloc = reposition_rope(rope_local, src_pos, dst_pos, freqs_cis)
        # dst_slots are the target token slots for tokens [skip_first, L)
        write_rope_bf16(kv_buffer, dst_slots[:n], rope_reloc, page_size)
        return n

    def clear(self):
        self._segments.clear()
        self.stats = {k: 0 for k in self.stats}


# Process-wide singleton (scheduler process).
_CONTROLLER: Optional[DSV4OfflineReuseController] = None


def get_offline_reuse_controller() -> DSV4OfflineReuseController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = DSV4OfflineReuseController()
    return _CONTROLLER
