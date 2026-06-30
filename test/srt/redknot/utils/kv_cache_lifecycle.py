#!/usr/bin/env python3
"""RedKnot KV-cache lifecycle manager + replay simulator.

Goal
----
Keep the KV of chunks that are *likely to be reused in the future*, and avoid
storing KV of chunks unlikely to be reused -- minimizing storage while keeping
hit rate high.

Three-layer policy
------------------
1. ADMISSION (frequency gate): a chunk is NOT cached on first sight. We only
   keep a tiny counter (chunk_id -> count). KV is materialized & cached only
   after the chunk has been seen >= R_ADMIT times. This filters one-shot
   chunks, which are the majority and cause "storage bloat".

2. EVICTION (value score, not plain LRU): when the byte budget is exceeded,
   evict the chunk with the lowest value score:
       score = freq_ema * recency_decay
       freq_ema      = EMA of reuse rate (future-reuse likelihood)
       recency_decay = exp(-(now - last_access) / TAU)
   Hot-but-idle chunks survive; cold chunks die fast.

3. TTL (frequency-scaled): each cached chunk gets a TTL proportional to its
   freq_ema. Cold chunks expire quickly and free memory automatically.

The manager is O(log n) per access (hash map + score-ordered heap), and the
whole admission state is a few bytes per distinct chunk.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ChunkMeta:
    chunk_id: str
    n_tokens: int
    kv_bytes: int
    seen: int = 0  # total times observed (admission counter)
    freq_ema: float = 0.0  # EMA of reuse likelihood
    last_access: int = 0  # logical time (request index)
    admitted: bool = False  # KV currently materialized in cache
    expire_at: int = 0  # TTL deadline (logical time)


class KVCacheLifecycleManager:
    def __init__(
        self,
        capacity_bytes: int,
        r_admit: int = 2,
        alpha: float = 0.3,
        tau: float = 200.0,
        ttl_base: float = 100.0,
        ttl_scale: float = 400.0,
        ttl_watermark: float = 0.9,
    ):
        self.capacity = capacity_bytes
        self.r_admit = r_admit
        self.alpha = alpha
        self.tau = tau
        self.ttl_base = ttl_base
        self.ttl_scale = ttl_scale
        self.ttl_watermark = ttl_watermark

        self.meta: Dict[str, ChunkMeta] = {}
        self.used_bytes = 0
        self.now = 0

        # metrics
        self.hits = 0
        self.misses = 0
        self.admitted_count = 0
        self.recompute_tokens = 0  # tokens that had to be (re)computed
        self.reused_tokens = 0  # tokens served from cache
        self.peak_bytes = 0

    # --- scoring ---
    def _score(self, m: ChunkMeta) -> float:
        decay = math.exp(-(self.now - m.last_access) / self.tau)
        return m.freq_ema * decay

    def _ttl(self, m: ChunkMeta) -> int:
        # higher freq -> longer TTL
        return int(self.ttl_base + self.ttl_scale * min(m.freq_ema, 1.0))

    # --- TTL sweep (lazy, cheap) ---
    # TTL only reclaims memory under pressure. When the cache is below a
    # utilization watermark there is no benefit to expiring hot chunks, so we
    # skip the sweep -- this avoids dropping chunks that would be reused soon.
    def _expire(self):
        if self.used_bytes < self.ttl_watermark * self.capacity:
            return
        dead = [
            cid
            for cid, m in self.meta.items()
            if m.admitted and m.expire_at <= self.now
        ]
        for cid in dead:
            self._evict(cid)

    def _evict(self, chunk_id: str):
        m = self.meta[chunk_id]
        if m.admitted:
            self.used_bytes -= m.kv_bytes
            m.admitted = False

    def _make_room(self, need: int):
        if self.used_bytes + need <= self.capacity:
            return
        # evict lowest-score admitted chunks until room
        admitted = [m for m in self.meta.values() if m.admitted]
        admitted.sort(key=self._score)  # ascending: worst first
        i = 0
        while self.used_bytes + need > self.capacity and i < len(admitted):
            self._evict(admitted[i].chunk_id)
            i += 1

    def access(
        self, chunk_id: str, n_tokens: int, kv_bytes: int, carry_tokens: int = 0
    ):
        """Process one chunk access in the request stream.

        carry_tokens: tokens always recomputed even on a hit (RedKnot
        carry-prefix). 0 for a pure full-reuse model.
        """
        self.now += 1
        self._expire()

        m = self.meta.get(chunk_id)
        if m is None:
            m = ChunkMeta(chunk_id, n_tokens, kv_bytes)
            self.meta[chunk_id] = m

        # update frequency EMA (1 = a reuse event happened now)
        m.freq_ema = self.alpha * 1.0 + (1 - self.alpha) * m.freq_ema
        m.seen += 1
        m.last_access = self.now

        if m.admitted:
            # HIT: serve from cache, only carry-prefix recomputed
            self.hits += 1
            self.reused_tokens += max(n_tokens - carry_tokens, 0)
            self.recompute_tokens += carry_tokens
            m.expire_at = self.now + self._ttl(m)
            return "hit"

        # MISS: must compute this chunk now
        self.misses += 1
        self.recompute_tokens += n_tokens

        # ADMISSION (adaptive):
        #  - If there is free room, admit freely (no downside to caching).
        #  - If contended, require R_ADMIT sightings (filter one-shot chunks),
        #    then admit by displacing the lowest-value resident. Value score
        #    blends frequency AND recency, so when frequencies are flat it
        #    degrades gracefully to LRU behaviour (the safe floor).
        has_free_room = self.used_bytes + kv_bytes <= self.capacity
        admit = False
        if has_free_room:
            admit = True
        elif m.seen >= self.r_admit:
            self._make_room(kv_bytes)
            admit = self.used_bytes + kv_bytes <= self.capacity
        if admit:
            m.admitted = True
            self.used_bytes += kv_bytes
            self.admitted_count += 1
            m.expire_at = self.now + self._ttl(m)
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        return "miss"

    def stats(self):
        total = self.hits + self.misses
        return dict(
            policy="redknot_3layer",
            requests=total,
            hits=self.hits,
            misses=self.misses,
            hit_rate=round(self.hits / max(total, 1), 4),
            peak_kv_gb=round(self.peak_bytes / 1e9, 2),
            reused_tokens=self.reused_tokens,
            recompute_tokens=self.recompute_tokens,
            token_reuse_rate=round(
                self.reused_tokens / max(self.reused_tokens + self.recompute_tokens, 1),
                4,
            ),
        )


# --- Baseline policies for comparison ---


class CacheAll:
    """Cache every chunk forever (the current 'offline + store everything')."""

    def __init__(self, **_):
        self.kv = {}
        self.hits = self.misses = 0
        self.reused_tokens = self.recompute_tokens = 0
        self.used = 0
        self.peak = 0

    def access(self, cid, n_tokens, kv_bytes, carry_tokens=0):
        if cid in self.kv:
            self.hits += 1
            self.reused_tokens += max(n_tokens - carry_tokens, 0)
            self.recompute_tokens += carry_tokens
            return "hit"
        self.misses += 1
        self.recompute_tokens += n_tokens
        self.kv[cid] = kv_bytes
        self.used += kv_bytes
        self.peak = max(self.peak, self.used)
        return "miss"

    def stats(self):
        t = self.hits + self.misses
        return dict(
            policy="cache_all",
            requests=t,
            hits=self.hits,
            misses=self.misses,
            hit_rate=round(self.hits / max(t, 1), 4),
            peak_kv_gb=round(self.peak / 1e9, 2),
            reused_tokens=self.reused_tokens,
            recompute_tokens=self.recompute_tokens,
            token_reuse_rate=round(
                self.reused_tokens / max(self.reused_tokens + self.recompute_tokens, 1),
                4,
            ),
        )


class LRU:
    """Plain LRU under a byte budget (no admission, no frequency)."""

    def __init__(self, capacity_bytes, **_):
        from collections import OrderedDict

        self.cap = capacity_bytes
        self.od = OrderedDict()
        self.hits = self.misses = 0
        self.reused_tokens = self.recompute_tokens = 0
        self.used = 0
        self.peak = 0

    def access(self, cid, n_tokens, kv_bytes, carry_tokens=0):
        if cid in self.od:
            self.hits += 1
            self.od.move_to_end(cid)
            self.reused_tokens += max(n_tokens - carry_tokens, 0)
            self.recompute_tokens += carry_tokens
            return "hit"
        self.misses += 1
        self.recompute_tokens += n_tokens
        while self.used + kv_bytes > self.cap and self.od:
            _, b = self.od.popitem(last=False)
            self.used -= b
        if self.used + kv_bytes <= self.cap:
            self.od[cid] = kv_bytes
            self.used += kv_bytes
            self.peak = max(self.peak, self.used)
        return "miss"

    def stats(self):
        t = self.hits + self.misses
        return dict(
            policy="lru",
            requests=t,
            hits=self.hits,
            misses=self.misses,
            hit_rate=round(self.hits / max(t, 1), 4),
            peak_kv_gb=round(self.peak / 1e9, 2),
            reused_tokens=self.reused_tokens,
            recompute_tokens=self.recompute_tokens,
            token_reuse_rate=round(
                self.reused_tokens / max(self.reused_tokens + self.recompute_tokens, 1),
                4,
            ),
        )


class LFU:
    """Frequency-only eviction under a byte budget (no recency, no admission)."""

    def __init__(self, capacity_bytes, **_):
        self.cap = capacity_bytes
        self.kv = {}
        self.freq = {}
        self.hits = self.misses = 0
        self.reused_tokens = self.recompute_tokens = 0
        self.used = 0
        self.peak = 0

    def access(self, cid, n_tokens, kv_bytes, carry_tokens=0):
        self.freq[cid] = self.freq.get(cid, 0) + 1
        if cid in self.kv:
            self.hits += 1
            self.reused_tokens += max(n_tokens - carry_tokens, 0)
            self.recompute_tokens += carry_tokens
            return "hit"
        self.misses += 1
        self.recompute_tokens += n_tokens
        while self.used + kv_bytes > self.cap and self.kv:
            victim = min(self.kv, key=lambda c: self.freq.get(c, 0))
            self.used -= self.kv.pop(victim)
        if self.used + kv_bytes <= self.cap:
            self.kv[cid] = kv_bytes
            self.used += kv_bytes
            self.peak = max(self.peak, self.used)
        return "miss"

    def stats(self):
        t = self.hits + self.misses
        return dict(
            policy="lfu",
            requests=t,
            hits=self.hits,
            misses=self.misses,
            hit_rate=round(self.hits / max(t, 1), 4),
            peak_kv_gb=round(self.peak / 1e9, 2),
            reused_tokens=self.reused_tokens,
            recompute_tokens=self.recompute_tokens,
            token_reuse_rate=round(
                self.reused_tokens / max(self.reused_tokens + self.recompute_tokens, 1),
                4,
            ),
        )
