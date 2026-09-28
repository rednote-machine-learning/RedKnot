"""Bounded head-page slabs with generation checked ownership and event retirement.

All published pages are immutable. A ref is ownership, a pin is an in-flight
reader, and an event is a physical lifetime dependency; none replaces another.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


class CapacityError(RuntimeError):
    pass


class StaleReference(RuntimeError):
    pass


class ImmediateEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


def completion_event(device):
    if torch.device(device).type == "cuda":
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return event
    return ImmediateEvent()


@dataclass(frozen=True, order=True)
class PageRef:
    slot: int
    generation: int
    content_id: str


@dataclass
class _Page:
    ref: PageRef
    owners: int = 1
    pins: int = 0
    extent: int = 0
    sealed: bool = False
    events: list[Any] = field(default_factory=list)


class HeadPagePool:
    """One bounded K/V slab per layout, shared by all layers and KV groups.

    Views returned by read_page are borrowed, read-only by contract. Callers
    must hold ownership and a pin until every asynchronous consumer finishes.
    """

    def __init__(self, capacity_pages: int, page_size: int, head_dim: int,
                 *, dtype=torch.float32, device="cpu"):
        if min(capacity_pages, page_size, head_dim) <= 0:
            raise ValueError("capacity, page size and head dimension must be positive")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("initial head KV pool supports float16/bfloat16/float32")
        self.capacity_pages = int(capacity_pages)
        self.page_size, self.head_dim = int(page_size), int(head_dim)
        self.device, self.dtype = torch.device(device), dtype
        self.k = torch.empty((capacity_pages, page_size, head_dim), device=self.device, dtype=dtype)
        # Resolve an index-free "cuda" once. Later callers may switch the
        # current device; descriptors and completion events must follow slabs.
        self.device = self.k.device
        self.v = torch.empty_like(self.k)
        self._free = list(reversed(range(capacity_pages)))
        self._generation = [0] * capacity_pages
        self._pages: dict[int, _Page] = {}
        self._lock = threading.RLock()
        self._boot = uuid.uuid4().hex
        self._sequence = 0
        self._retained_buffers = []
        self.poisoned = False
        self.high_water_pages = 0
        self.copied_bytes = 0
        self.written_bytes = 0

    @property
    def page_bytes(self):
        return 2 * self.page_size * self.head_dim * self.k.element_size()

    def _get(self, ref):
        p = self._pages.get(ref.slot)
        if p is None or p.ref != ref:
            raise StaleReference(f"stale page slot/generation: {ref.slot}/{ref.generation}")
        return p

    def reserve(self, count: int, *, content_ids=None) -> tuple[PageRef, ...]:
        if count < 0:
            raise ValueError("negative reservation")
        if content_ids is not None and (len(content_ids) != count or any(not isinstance(x, str) or not x for x in content_ids)):
            raise ValueError("invalid imported content identities")
        with self._lock:
            if self.poisoned:
                raise RuntimeError("pool quarantined after device/event failure")
            self.collect()
            if count > len(self._free):
                raise CapacityError(f"need {count} pages, available {len(self._free)}")
            # No device writes occur during reserve. If host allocation or user
            # interruption fails halfway through creating page metadata, restore
            # every candidate slot. Generations/identities remain monotonic.
            free_before = self._free[:]
            candidate_slots = self._free[-count:] if count else []
            result = []
            try:
                for i in range(count):
                    slot = self._free.pop()
                    self._generation[slot] += 1
                    self._sequence += 1
                    content_id = content_ids[i] if content_ids is not None else f"{self._boot}:{self._sequence}"
                    ref = PageRef(slot, self._generation[slot], content_id)
                    self._pages[slot] = _Page(ref)
                    result.append(ref)
            except BaseException:
                for slot in candidate_slots:
                    self._pages.pop(slot, None)
                self._free = free_before
                raise
            self.high_water_pages = max(self.high_water_pages, len(self._pages))
            return tuple(result)

    def retain(self, refs: Iterable[PageRef]):
        with self._lock:
            pages = [self._get(r) for r in refs]
            for p in pages:
                p.owners += 1

    def release(self, refs: Iterable[PageRef], event=None):
        with self._lock:
            pages = [self._get(r) for r in refs]
            for p in pages:
                if p.owners <= 0:
                    raise RuntimeError("unbalanced page release")
                p.owners -= 1
                if event is not None:
                    p.events.append(event)
            self.collect()

    def pin(self, refs: Iterable[PageRef]):
        with self._lock:
            pages = [self._get(r) for r in refs]
            for p in pages:
                p.pins += 1

    def unpin(self, refs: Iterable[PageRef], event=None):
        with self._lock:
            pages = [self._get(r) for r in refs]
            for p in pages:
                if p.pins <= 0:
                    raise RuntimeError("unbalanced page unpin")
                p.pins -= 1
                if event is not None:
                    p.events.append(event)
            self.collect()

    def write_page(self, ref, k, v):
        with self._lock:
            p = self._get(ref)
            if p.sealed or p.owners != 1 or p.pins:
                raise RuntimeError("write requires an unpublished exclusively owned page")
            if k.ndim != 2 or k.shape != v.shape or k.shape[1] != self.head_dim:
                raise ValueError("invalid K/V page shape")
            n = len(k)
            if not 0 < n <= self.page_size:
                raise ValueError("invalid page extent")
            self.k[ref.slot, :n].copy_(k)
            self.v[ref.slot, :n].copy_(v)
            p.extent = n
            self.written_bytes += 2 * n * self.head_dim * self.k.element_size()

    def clone_patch(self, source, dest, extent, indices, k, v):
        """Copy retained rows only; a complete rewrite performs zero old KV copy."""
        with self._lock:
            src, dst = self._get(source), self._get(dest)
            if not src.sealed or dst.sealed or dst.owners != 1 or dst.pins:
                raise RuntimeError("invalid clone ownership")
            if extent > src.extent or len(set(indices)) != len(indices):
                raise ValueError("invalid repair indices/extent")
            if any(i < 0 or i >= extent for i in indices):
                raise ValueError("repair outside visible extent")
            if tuple(k.shape) != (len(indices), self.head_dim) or k.shape != v.shape:
                raise ValueError("invalid repair payload shape")
            changed = set(indices)
            kept = [i for i in range(extent) if i not in changed]
            if kept:
                idx = torch.tensor(kept, device=self.device, dtype=torch.long)
                self.k[dest.slot].index_copy_(0, idx, self.k[source.slot].index_select(0, idx))
                self.v[dest.slot].index_copy_(0, idx, self.v[source.slot].index_select(0, idx))
                self.copied_bytes += 2 * len(kept) * self.head_dim * self.k.element_size()
            if indices:
                idx = torch.tensor(indices, device=self.device, dtype=torch.long)
                self.k[dest.slot].index_copy_(0, idx, k.to(device=self.device, dtype=self.dtype))
                self.v[dest.slot].index_copy_(0, idx, v.to(device=self.device, dtype=self.dtype))
                self.written_bytes += 2 * len(indices) * self.head_dim * self.k.element_size()
            dst.extent = extent

    def clone_patch_batch(self, plans, payloads):
        """Clone/repair all touched pages with a constant number of GPU launches.

        Each plan is ``(source, dest, extent, local_indices, payload_index,
        payload_rows)``. ``payloads`` contains compact (K, V) repair tensors, one
        per input patch. Every repair row must occur exactly once in the plan.
        The caller owns every reserved destination and pins the immutable source
        version until its completion event. No full-history KV gather is used:
        only retained rows from touched source pages form temporary tensors.
        """
        with self._lock:
            if self.poisoned:
                raise RuntimeError("pool quarantined")
            offsets, total_rows = [], 0
            for k, v in payloads:
                if (k.ndim != 2 or k.shape != v.shape or k.shape[1] != self.head_dim or
                        k.device != v.device):
                    raise ValueError("invalid batch repair payload shape/device")
                offsets.append(total_rows)
                total_rows += len(k)
            # Global repair-row order lets us scatter concatenated payloads
            # directly, without per-page index_select operations.
            repair_dest = [-1] * total_rows
            kept_source, kept_dest, destinations = [], [], []
            seen_dest = set()
            for source, dest, extent, indices, payload_i, rows in plans:
                src, dst = self._get(source), self._get(dest)
                if not src.sealed or dst.sealed or dst.owners != 1 or dst.pins:
                    raise RuntimeError("invalid batch clone ownership")
                if dest in seen_dest:
                    raise ValueError("batch destinations must be unique")
                seen_dest.add(dest)
                if (type(extent) is not int or not 0 < extent <= src.extent or
                        type(payload_i) is not int or not 0 <= payload_i < len(payloads) or
                        len(indices) != len(rows) or len(set(indices)) != len(indices)):
                    raise ValueError("invalid batch repair plan")
                changed = set(indices)
                if any(type(i) is not int or not 0 <= i < extent for i in indices):
                    raise ValueError("batch repair outside visible extent")
                source_start, dest_start = source.slot * self.page_size, dest.slot * self.page_size
                for index in range(extent):
                    if index not in changed:
                        kept_source.append(source_start + index)
                        kept_dest.append(dest_start + index)
                for index, row in zip(indices, rows):
                    if type(row) is not int or not 0 <= row < len(payloads[payload_i][0]):
                        raise ValueError("batch repair payload row is out of range")
                    flat_row = offsets[payload_i] + row
                    if repair_dest[flat_row] != -1:
                        raise ValueError("batch repair payload row is used twice")
                    repair_dest[flat_row] = dest_start + index
                destinations.append((dst, extent))
            if any(index == -1 for index in repair_dest):
                raise ValueError("batch repair payload contains unused rows")
            if not destinations:
                return

            # Validate everything before issuing writes. Packing indices into
            # one tensor avoids synchronous H2D setup once per page/group.
            packed = torch.tensor(kept_source + kept_dest + repair_dest,
                                  device=self.device, dtype=torch.long)
            kept_count = len(kept_source)
            source_idx = packed[:kept_count]
            dest_idx = packed[kept_count:2 * kept_count]
            repair_idx = packed[2 * kept_count:]
            k_flat, v_flat = self.k.view(-1, self.head_dim), self.v.view(-1, self.head_dim)
            if total_rows:
                # Inputs are compact repair rows, never the historical KV.
                ks = [k.to(device=self.device, dtype=self.dtype) for k, _ in payloads if len(k)]
                vs = [v.to(device=self.device, dtype=self.dtype) for _, v in payloads if len(v)]
                new_k = ks[0] if len(ks) == 1 else torch.cat(ks)
                new_v = vs[0] if len(vs) == 1 else torch.cat(vs)
            if kept_count:
                # These temporary tensors have touched-retained-row extent only.
                k_flat.index_copy_(0, dest_idx, k_flat.index_select(0, source_idx))
                v_flat.index_copy_(0, dest_idx, v_flat.index_select(0, source_idx))
                self.copied_bytes += 2 * kept_count * self.head_dim * self.k.element_size()
            if total_rows:
                k_flat.index_copy_(0, repair_idx, new_k)
                v_flat.index_copy_(0, repair_idx, new_v)
                self.written_bytes += 2 * total_rows * self.head_dim * self.k.element_size()
            for dst, extent in destinations:
                dst.extent = extent

    def seal(self, refs):
        with self._lock:
            for ref in refs:
                p = self._get(ref)
                if p.extent <= 0:
                    raise ValueError("cannot publish an uninitialized page")
                p.sealed = True

    def read_page(self, ref, extent=None):
        with self._lock:
            if self.poisoned:
                raise RuntimeError("pool quarantined")
            p = self._get(ref)
            n = p.extent if extent is None else extent
            if not p.sealed or not 0 <= n <= p.extent:
                raise ValueError("unpublished page or invalid extent")
            return self.k[ref.slot, :n], self.v[ref.slot, :n]

    def collect(self):
        with self._lock:
            if self.poisoned:
                return 0
            freed = 0
            # One immutable completion event can protect thousands of pages and
            # descriptor buffers. Query it once per collection pass, not once
            # per reference. Keep the event itself alive alongside the result
            # so Python cannot reuse its id while the cache is active. A cached
            # False only delays retirement until the next pass; no result is
            # carried across collect() calls. Attached events must not be
            # re-recorded to represent later work after publication to a lease.
            queried = {}

            def complete(event):
                key = id(event)
                cached = queried.get(key)
                if cached is None:
                    cached = (event, bool(event.query()))
                    queried[key] = cached
                return cached[1]

            try:
                self._retained_buffers[:] = [(e, buffers) for e, buffers in self._retained_buffers if not complete(e)]
            except Exception:
                self.poisoned = True
                raise
            for slot, p in list(self._pages.items()):
                try:
                    p.events[:] = [e for e in p.events if not complete(e)]
                except Exception:
                    self.poisoned = True
                    raise
                if p.owners == 0 and p.pins == 0 and not p.events:
                    del self._pages[slot]
                    self._free.append(slot)
                    freed += 1
            return freed

    def defer_buffers(self, buffers, event):
        with self._lock:
            self._retained_buffers.append((event, buffers))

    def stats(self):
        with self._lock:
            self.collect()
            reserved = sum(not p.sealed and p.owners > 0 for p in self._pages.values())
            live = sum(p.sealed and p.owners > 0 for p in self._pages.values())
            retiring = sum(p.owners == 0 for p in self._pages.values())
            assert len(self._free) + reserved + live + retiring == self.capacity_pages
            return dict(capacity_pages=self.capacity_pages, free_pages=len(self._free),
                        reserved_pages=reserved, live_pages=live, retiring_pages=retiring,
                        pinned_pages=sum(p.pins > 0 or bool(p.events) for p in self._pages.values()),
                        allocated_bytes=len(self._pages) * self.page_bytes,
                        reserved_slab_bytes=self.capacity_pages * self.page_bytes,
                        high_water_pages=self.high_water_pages, copied_bytes=self.copied_bytes,
                        written_bytes=self.written_bytes, poisoned=self.poisoned)
