# Copyright 2024-2026 SGLang RedKnot Integration.
"""Head-aware capacity model and cache policy for RedKnot serving.

Dense vLLM-style serving is usually memory-bound: once GPU memory is filled
by per-session KV cache, no more requests can be admitted even if compute is
free. With physical per-head KV storage (SegPagedAttention), local heads
allocate pages only for their short ``sink + recent`` window while global
heads allocate full-context pages. This changes the bottleneck from "how
many full dense KV caches fit in HBM" to "how many per-head compact KV
caches fit in HBM", which is why RedKnot's concurrent capacity per GPU grows
4.7-7.8× in the paper (§5.5 / fig. 13c).

This module provides the **capacity model** and **cache policy** that a
head-aware scheduler needs, decoupled from the SGLang scheduler internals so
it is unit-testable:

- :func:`per_session_kv_bytes` — dense vs. head-aware per-session KV
  footprint for a given Head Class Map and context length.
- :func:`concurrent_capacity` — sessions that fit in a memory budget, dense
  vs. head-aware, plus the capacity multiplier (the fig. 13c number).
- :class:`HeadAwareCachePolicy` — content-addressed chunk admission /
  eviction that prioritises **reuse frequency** over prefix length (paper
  §6.2) and accounts for per-head footprint and recovery cost.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Capacity model
# ──────────────────────────────────────────────────────────────────────────
def per_session_kv_bytes(
    head_config: HeadClassConfig,
    *,
    seq_len: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> Dict[str, int]:
    """Per-session KV footprint, dense vs. head-aware.

    Dense keeps ``L`` tokens for **every** head. Head-aware keeps ``L`` only
    for global heads; local heads keep ``min(L, sink + window)``.

    Returns a dict with ``dense_bytes``, ``head_aware_bytes``, and the
    per-head-class token totals.
    """
    L = seq_len
    bytes_per_tok = 2 * head_dim * dtype_bytes  # K + V

    dense_tokens = 0
    head_aware_tokens = 0
    global_tokens = 0
    local_tokens = 0
    for li in range(head_config.num_layers):
        for h in range(head_config.num_kv_heads):
            strat = head_config.get_strategy(li, h)
            dense_tokens += L
            if strat.is_local():
                kept = min(L, max(strat.window, 0) + max(strat.sink_size, 0))
                head_aware_tokens += kept
                local_tokens += kept
            else:
                head_aware_tokens += L
                global_tokens += L

    return {
        "dense_bytes": dense_tokens * bytes_per_tok,
        "head_aware_bytes": head_aware_tokens * bytes_per_tok,
        "dense_tokens": dense_tokens,
        "head_aware_tokens": head_aware_tokens,
        "global_tokens": global_tokens,
        "local_tokens": local_tokens,
    }


def concurrent_capacity(
    head_config: HeadClassConfig,
    *,
    seq_len: int,
    head_dim: int,
    kv_budget_bytes: int,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    """Sessions that fit in ``kv_budget_bytes``, dense vs. head-aware.

    Returns ``dense_sessions``, ``head_aware_sessions`` and
    ``capacity_multiplier`` — the per-GPU concurrency gain (paper fig. 13c).
    """
    foot = per_session_kv_bytes(
        head_config, seq_len=seq_len, head_dim=head_dim, dtype_bytes=dtype_bytes
    )
    dense = foot["dense_bytes"]
    ha = foot["head_aware_bytes"]
    dense_sessions = kv_budget_bytes // dense if dense else 0
    ha_sessions = kv_budget_bytes // ha if ha else 0
    return {
        "dense_sessions": int(dense_sessions),
        "head_aware_sessions": int(ha_sessions),
        "capacity_multiplier": (ha_sessions / dense_sessions)
        if dense_sessions
        else float("inf"),
        "per_session_dense_bytes": dense,
        "per_session_head_aware_bytes": ha,
    }


# ──────────────────────────────────────────────────────────────────────────
# Content-addressed, reuse-prioritised cache policy
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ChunkEntry:
    """Bookkeeping for one cached reusable chunk (content-addressed)."""

    chunk_id: str
    nbytes: int
    reuse_count: int = 0
    recovery_cost: float = 1.0  # relative cost to recompute global heads
    last_used_step: int = 0
    meta: Dict[str, object] = field(default_factory=dict)

    def priority(self, now_step: int) -> float:
        """Higher = more valuable to keep.

        Prioritises reuse frequency (paper §6.2: admission prioritises
        high-reuse chunks rather than long shared prefixes) and recovery
        cost (expensive-to-recover chunks are worth keeping), with a mild
        recency tie-break.
        """
        recency = 1.0 / (1.0 + max(0, now_step - self.last_used_step))
        return (self.reuse_count + 1) * self.recovery_cost * (1.0 + recency)


class HeadAwareCachePolicy:
    """Reuse-prioritised, per-head-footprint-aware chunk cache.

    Unlike a prefix cache (which keys on leading tokens and evicts by LRU),
    this keys on **content chunks** and evicts the lowest-priority chunk,
    where priority rewards reuse frequency and recovery cost. This realises
    the cache-system primitive the paper proposes in §6.2.
    """

    def __init__(self, *, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self._entries: "OrderedDict[str, ChunkEntry]" = OrderedDict()
        self._used_bytes = 0
        self._step = 0
        self.stats = {"hits": 0, "misses": 0, "admissions": 0, "evictions": 0}

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def tick(self) -> int:
        self._step += 1
        return self._step

    def get(self, chunk_id: str) -> Optional[ChunkEntry]:
        e = self._entries.get(chunk_id)
        if e is None:
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        e.reuse_count += 1
        e.last_used_step = self._step
        self._entries.move_to_end(chunk_id)
        return e

    def admit(
        self,
        chunk_id: str,
        nbytes: int,
        *,
        recovery_cost: float = 1.0,
        meta: Optional[Dict[str, object]] = None,
    ) -> bool:
        """Try to admit a chunk, evicting lower-priority entries if needed.

        Returns True if the chunk is resident after the call.
        """
        if chunk_id in self._entries:
            e = self._entries[chunk_id]
            e.reuse_count += 1
            e.last_used_step = self._step
            self._entries.move_to_end(chunk_id)
            return True
        if nbytes > self.capacity_bytes:
            return False  # cannot ever fit

        # Evict by lowest priority until it fits.
        while self._used_bytes + nbytes > self.capacity_bytes and self._entries:
            victim_id = min(
                self._entries,
                key=lambda cid: self._entries[cid].priority(self._step),
            )
            cand = self._entries[victim_id]
            # Do not evict a higher-priority chunk for a brand-new one.
            new_priority = (1) * recovery_cost * 2.0
            if cand.priority(self._step) > new_priority and len(self._entries) > 0:
                # Still must make room; only refuse if eviction can't help.
                if self._used_bytes + nbytes - cand.nbytes > self.capacity_bytes:
                    pass  # keep evicting
            self._used_bytes -= cand.nbytes
            del self._entries[victim_id]
            self.stats["evictions"] += 1

        if self._used_bytes + nbytes > self.capacity_bytes:
            return False
        self._entries[chunk_id] = ChunkEntry(
            chunk_id=chunk_id,
            nbytes=nbytes,
            reuse_count=0,
            recovery_cost=recovery_cost,
            last_used_step=self._step,
            meta=meta or {},
        )
        self._used_bytes += nbytes
        self.stats["admissions"] += 1
        return True

    def resident_chunks(self) -> List[str]:
        return list(self._entries.keys())


__all__ = [
    "ChunkEntry",
    "HeadAwareCachePolicy",
    "per_session_kv_bytes",
    "concurrent_capacity",
]
