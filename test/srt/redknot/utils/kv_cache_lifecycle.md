# RedKnot KV-Cache Lifecycle Management

## Problem

The current approach generates KV offline and stores everything for online
reuse, causing **storage bloat**: in musique, 62% of chunks are never reused
yet "cache-all" would need **524 GB** of KV to hold them all (DeepSeek V4 MLA).

We want a policy that **keeps the KV of chunks likely to be reused soon, and
does not store KV of chunks unlikely to be reused** -- minimal storage, high
hit rate. Design must be simple and efficient.

## Three-Layer Policy

### Layer 1 -- Admission (frequency gate)
A chunk is NOT cached on first sight; we only keep a tiny counter
(`chunk_id -> seen`). KV is materialized and cached only after the chunk has
been seen `>= R_ADMIT` times (default 2). This filters one-shot chunks -- the
majority -- and directly stops storage bloat.

### Layer 2 -- Eviction (value score, not plain LRU)
When the byte budget is exceeded, evict the lowest value score:

```
score(chunk) = freq_ema * recency_decay
freq_ema      = EMA of reuse rate   (future-reuse likelihood)
recency_decay = exp(-(now - last_access) / TAU)
```

Hot-but-idle chunks survive; cold chunks die fast. When frequencies are flat,
the recency term makes it degrade gracefully toward LRU.

### Layer 3 -- TTL (frequency-scaled, watermark-gated)
Each cached chunk gets `TTL = ttl_base + ttl_scale * freq_ema`; cold chunks
expire and free memory. The TTL sweep only runs above a utilization watermark
(default 0.9), so hot chunks are never dropped while memory is plentiful.

### Complexity
One hash map (`chunk_id -> meta`, a few bytes per distinct chunk) plus
score-ordered eviction. O(log n) per access; admission state is negligible.

## Validation -- replay on real non-prefix reuse

We replay the real musique stream (2417 questions, 48315 chunk accesses,
17629 unique passages, 95% non-prefix reuse) through every policy under the
same byte budget, using DeepSeek V4 Flash MLA KV (49,536 B/token).

### Hit rate vs budget (DeepSeek V4 MLA)

| Budget | LRU | LFU | RedKnot 3-layer | RedKnot vs LRU |
|---:|---:|---:|---:|---:|
| 0.05 GB | 0.018 | -    | 0.023 | **+29%** |
| 0.10 GB | 0.043 | 0.024 | 0.063 | **+47%** |
| 0.25 GB | 0.088 | 0.038 | 0.122 | **+39%** |
| 0.50 GB | 0.171 | 0.054 | 0.208 | **+22%** |
| 1.00 GB | 0.311 | 0.077 | 0.306 | -1% |
| 2.00 GB | 0.409 | 0.109 | 0.370 | -10% |
| 4.00 GB | 0.457 | 0.155 | 0.420 | -8% |

cache-all reaches hit rate 0.635 but needs **524 GB** -- impractical.

## Findings (honest)

1. **Under memory scarcity the 3-layer policy wins decisively**: +22% to +47%
   hit rate over LRU at 0.05-0.5 GB. The admission gate stops one-shot chunks
   from evicting useful residents -- exactly the regime that matters for
   100B+ models where per-instance KV budget is a few GB.

2. **Under memory abundance LRU is slightly better** (-1% to -10% at 1-4 GB):
   musique has strong sequential locality that plain LRU exploits perfectly,
   and admission filtering becomes a mild handicap when there is room to cache
   everything anyway.

3. **LFU is consistently worst**: pure frequency without recency keeps stale
   chunks. This confirms the value score must blend both.

4. **The crossover (~1 GB here) defines when to use which policy.** A practical
   deployment can switch to LRU-only behaviour above the crossover by setting
   `R_ADMIT=1` (admit everything), and use the full 3-layer policy below it.

## Practical recommendation

- **Scarce KV budget (the common case for large models): use the 3-layer
  policy.** Admission alone removes the 62% one-shot bloat; score-based
  eviction + TTL keep the hot long-tail.
- **Abundant budget: set `R_ADMIT=1`** to recover LRU behaviour, since
  admission filtering no longer helps.

## Files
- `kv_cache_lifecycle.py`  -- the manager + LRU/LFU/cache-all baselines
- `replay_kv_lifecycle.py` -- single-budget replay
- `plot_kv_lifecycle_sweep.py` -- budget sweep + crossover plot
- figures: `kv_lifecycle_sweep.png/.pdf`, `replay_kv_lifecycle.json`

## Reproduce
```bash
python test/srt/redknot/replay_kv_lifecycle.py --budget-gb 0.25 --carry-tokens 0
python test/srt/redknot/plot_kv_lifecycle_sweep.py
```
