# Copyright 2024-2026 SGLang RedKnot Integration.
"""Segment page table + per-head paged KV cache for SegPaged v2.

This layer ties together the two lower abstractions:

- :mod:`visible_plan` decides *which* tokens a head keeps.
- :mod:`storage` decides *where* the bytes live (virtual page -> physical).

:class:`SegmentPageTable` records, for each ``(layer, head, segment)``, the
consecutive virtual pages it maps to plus the original token positions those
pages hold. :class:`PagedHeadKVCache` is the user-facing object: you feed it
dense per-head KV plus a :class:`HeadVisiblePlan`, and it stores only the
visible tokens into the storage backend and lets you gather a head's KV back.

The key property is that **storage and gather never branch on head class** —
they operate purely on positions and pages. That is what makes this a single
unified base usable by global / local / retrieval / custom heads alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .storage import KVStorageBackend, LocalPagedPool
from .visible_plan import HeadVisiblePlan


@dataclass
class HeadSegment:
    """One ``(layer, head, segment)`` and its virtual page span.

    Attributes
    ----------
    layer, head, segment:
        The ``(ℓ, h, s)`` index this segment belongs to.
    page_ids:
        Consecutive virtual page ids backing this segment.
    positions:
        ``[kept]`` original token positions stored across these pages, in
        storage order. Used by the attention layer to reconstruct causal
        ordering and for verification.
    """

    layer: int
    head: int
    segment: int
    page_ids: List[int]
    positions: torch.Tensor

    @property
    def kept(self) -> int:
        return int(self.positions.numel())


class SegmentPageTable:
    """``(layer, head, segment) -> virtual pages`` mapping.

    Also tracks the ordered segment list per ``(layer, head)`` so a head's
    full KV view can be reassembled segment-by-segment.
    """

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self._segments: Dict[Tuple[int, int, int], HeadSegment] = {}
        self._head_segments: Dict[Tuple[int, int], List[int]] = {}

    def add(self, seg: HeadSegment) -> None:
        key = (seg.layer, seg.head, seg.segment)
        self._segments[key] = seg
        self._head_segments.setdefault((seg.layer, seg.head), []).append(seg.segment)

    def get(self, layer: int, head: int, segment: int) -> HeadSegment:
        return self._segments[(layer, head, segment)]

    def segments_of(self, layer: int, head: int) -> List[HeadSegment]:
        seg_ids = self._head_segments.get((layer, head), [])
        return [self._segments[(layer, head, s)] for s in seg_ids]

    def all_segments(self) -> List[HeadSegment]:
        return list(self._segments.values())


class PagedHeadKVCache:
    """Per-(layer, head) paged KV store driven by visible-token plans.

    Parameters
    ----------
    num_kv_heads, head_dim:
        Topology (after TP sharding).
    page_size:
        Tokens per page.
    storage:
        Optional pre-built :class:`KVStorageBackend`. When omitted a
        :class:`LocalPagedPool` reference backend is created. Passing a
        custom backend is the seam for a future SGLang-pool integration.
    device, dtype:
        Used only when constructing the default backend.
    """

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 64,
        storage: Optional[KVStorageBackend] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.storage = storage or LocalPagedPool(
            head_dim=head_dim,
            page_size=page_size,
            device=self.device,
            dtype=dtype,
        )
        self.table = SegmentPageTable(page_size=page_size)

    # ────────────────────────────────────────────────────────────────
    # Write
    # ────────────────────────────────────────────────────────────────
    def store_head_segment(
        self,
        *,
        layer: int,
        head: int,
        segment: int,
        k_dense: torch.Tensor,  # [seq_len, head_dim] full dense KV for this head
        v_dense: torch.Tensor,
        plan: HeadVisiblePlan,
    ) -> HeadSegment:
        """Store only ``plan.positions`` of one head's dense KV into pages.

        The dense ``[seq_len, head_dim]`` tensor is gathered down to the
        visible positions, then scattered into freshly allocated pages.
        Heads with few visible tokens occupy few pages — the physical
        capacity win.
        """
        if k_dense.dim() != 2 or v_dense.dim() != 2:
            raise ValueError(
                f"store_head_segment expects [seq_len, head_dim] k/v, got "
                f"{tuple(k_dense.shape)} / {tuple(v_dense.shape)}"
            )
        pos = plan.positions.to(k_dense.device)
        k_vis = k_dense.index_select(0, pos)
        v_vis = v_dense.index_select(0, pos)
        n = k_vis.shape[0]

        n_pages = (n + self.page_size - 1) // self.page_size
        virt_ids = self.storage.alloc_pages(n_pages)
        for i, vid in enumerate(virt_ids):
            start = i * self.page_size
            end = min(start + self.page_size, n)
            if end > start:
                self.storage.write_page(vid, k_vis[start:end], v_vis[start:end])

        seg = HeadSegment(
            layer=layer,
            head=head,
            segment=segment,
            page_ids=virt_ids,
            positions=pos.cpu(),
        )
        self.table.add(seg)
        return seg

    # ────────────────────────────────────────────────────────────────
    # Read
    # ────────────────────────────────────────────────────────────────
    def gather_segment(self, seg: HeadSegment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialise one segment's stored ``[kept, head_dim]`` K/V."""
        remaining = seg.kept
        k_parts, v_parts = [], []
        for vid in seg.page_ids:
            take = min(self.page_size, remaining)
            if take <= 0:
                break
            k, v = self.storage.read_page(vid, take)
            k_parts.append(k)
            v_parts.append(v)
            remaining -= take
        if not k_parts:
            empty = torch.zeros(0, self.head_dim, device=self.device, dtype=self.dtype)
            return empty, empty.clone()
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def gather_head(self, layer: int, head: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all segments of a ``(layer, head)`` into one view."""
        segs = self.table.segments_of(layer, head)
        if not segs:
            empty = torch.zeros(0, self.head_dim, device=self.device, dtype=self.dtype)
            return empty, empty.clone()
        ks, vs = [], []
        for seg in segs:
            k, v = self.gather_segment(seg)
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    # ────────────────────────────────────────────────────────────────
    # Accounting
    # ────────────────────────────────────────────────────────────────
    def stored_token_count(self) -> int:
        return sum(seg.kept for seg in self.table.all_segments())

    def physical_bytes(self) -> int:
        return self.storage.physical_bytes()


__all__ = [
    "HeadSegment",
    "SegmentPageTable",
    "PagedHeadKVCache",
]
