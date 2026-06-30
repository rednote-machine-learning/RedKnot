# Copyright 2024-2026 SGLang RedKnot Integration.
"""Paged KV storage backend abstraction for SegPaged v2.

This is the "段页式存储" (segment-paged storage) substrate. It separates the
**logical address** a head uses to read/write KV from the **physical
storage** that actually holds the bytes:

    (req, layer, head, segment) --page table--> virtual page ids
    virtual page id            --backend--> physical block

The :class:`KVStorageBackend` abstract interface is the seam where a real
SGLang ``TokenToKVPool`` can be plugged in later: anything that can
``allocate`` physical blocks and expose per-block K/V views satisfies the
contract. :class:`LocalPagedPool` is a self-contained reference backend
(a single contiguous ``[num_blocks, page_size, head_dim]`` pool) so the
runtime is fully usable and testable today, with **zero coupling** to the
existing attention backends — current benchmarks are untouched.

Design goals
------------
- Virtual page space is per backend; the page table (see ``page_table.py``)
  owns the ``head -> [virtual pages]`` mapping.
- Physical allocation is contiguous-per-page and grows lazily, so a head
  that keeps few tokens occupies few physical blocks — the capacity win.
- The backend never knows about head classes or policies; it only moves
  bytes for virtual pages. That keeps the abstraction reusable as the
  unified base for any future multi-head storage scheme.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple

import torch


class KVStorageBackend(abc.ABC):
    """Abstract paged K/V byte store addressed by virtual page id.

    A backend manages a pool of fixed-size physical pages, each holding
    ``page_size`` token slots of ``head_dim`` features for K and V. The
    page table layer requests virtual pages and reads/writes token slices
    through them; the backend resolves virtual -> physical and returns
    tensor views.
    """

    page_size: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    @abc.abstractmethod
    def alloc_pages(self, n_pages: int) -> List[int]:
        """Reserve ``n_pages`` virtual pages, returning their virtual ids."""

    @abc.abstractmethod
    def write_page(
        self,
        virt_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write ``[n, head_dim]`` K/V (``n <= page_size``) into one page."""

    @abc.abstractmethod
    def read_page(self, virt_id: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the first ``n`` token slots of one page as ``[n, head_dim]``."""

    @abc.abstractmethod
    def num_physical_pages(self) -> int:
        """Number of physical pages currently backing virtual pages."""

    def physical_bytes(self) -> int:
        """Total bytes occupied by physical pages (K + V)."""
        n = self.num_physical_pages()
        if n == 0:
            return 0
        elt = torch.tensor([], dtype=self.dtype).element_size()
        per_page = self.page_size * self.head_dim * elt
        return 2 * n * per_page


class LocalPagedPool(KVStorageBackend):
    """Reference backend: one contiguous device pool, grown lazily.

    Physical K/V live in growable lists of ``[page_size, head_dim]`` tensors
    (one entry per physical page). A ``virt -> phys`` dict provides the
    indirection so a future backend can remap pages (e.g. onto a shared
    SGLang block pool) without touching callers.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        page_size: int = 64,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._k_pages: List[torch.Tensor] = []
        self._v_pages: List[torch.Tensor] = []
        self._virt_to_phys: Dict[int, int] = {}
        self._next_virt = 0
        self._next_phys = 0

    def alloc_pages(self, n_pages: int) -> List[int]:
        virt_ids: List[int] = []
        for _ in range(n_pages):
            vid = self._next_virt
            self._next_virt += 1
            phys = self._next_phys
            self._next_phys += 1
            self._virt_to_phys[vid] = phys
            self._k_pages.append(
                torch.zeros(
                    self.page_size, self.head_dim, device=self.device, dtype=self.dtype
                )
            )
            self._v_pages.append(
                torch.zeros(
                    self.page_size, self.head_dim, device=self.device, dtype=self.dtype
                )
            )
            virt_ids.append(vid)
        return virt_ids

    def write_page(self, virt_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.shape[0]
        if n > self.page_size:
            raise ValueError(
                f"write_page: {n} tokens exceed page_size={self.page_size}"
            )
        phys = self._virt_to_phys[virt_id]
        self._k_pages[phys][:n] = k.to(self.device, self.dtype)
        self._v_pages[phys][:n] = v.to(self.device, self.dtype)

    def read_page(self, virt_id: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        phys = self._virt_to_phys[virt_id]
        return self._k_pages[phys][:n], self._v_pages[phys][:n]

    def num_physical_pages(self) -> int:
        return self._next_phys


__all__ = [
    "KVStorageBackend",
    "LocalPagedPool",
]
