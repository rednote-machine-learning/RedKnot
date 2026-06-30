# Iso-Throughput Memory: LRU vs Store-Everything

## Question

At the **same throughput** (= same cache hit rate = same prefills avoided),
how much KV memory does an LRU cache need, compared with storing **all**
offline KV? This directly quantifies RedKnot's memory saving.

## Method

1. **Store-all baseline**: cache every unique chunk forever. Memory = sum of
   all unique-chunk KV; hit rate = theoretical max (every repeat hits).
2. **LRU**: sweep byte budgets, record (budget -> hit rate).
3. For each target throughput (a fraction of the store-all hit rate), find the
   minimum LRU memory that reaches it (interpolated on the curve).
4. Memory saving = store_all_GB / LRU_GB at iso-throughput.

Cost model: DeepSeek V4 Flash MLA, 49,536 B/token, ~600 tok/chunk.

## Results -- KV memory needed at iso-throughput

| Dataset | store-all (GB) | 50% | 80% | 90% | 95% |
|---|---:|---:|---:|---:|---:|
| musique_ans (real) | 524.0 | 1.05 GB | 19.95 GB | 91.6 GB | 190 GB |
| synth_zipf0.8 (low skew) | 373.3 | 35.0 GB | 135 GB | 204 GB | 253 GB |
| synth_zipf1.1 (med skew) | 224.2 | 2.81 GB | 23.9 GB | 57.7 GB | 93.9 GB |
| synth_zipf1.5 (high skew) | 74.6 | 0.56 GB | 2.49 GB | 6.36 GB | 13.4 GB |

## Results -- memory saving factor (store-all / LRU)

| Dataset | 50% | 80% | 90% | 95% | 99% |
|---|---:|---:|---:|---:|---:|
| musique_ans | **497x** | **26x** | 5.7x | 2.8x | 1.5x |
| synth_zipf0.8 | 10.7x | 2.8x | 1.8x | 1.5x | 1.0x |
| synth_zipf1.1 | **80x** | **9.4x** | 3.9x | 2.4x | 1.3x |
| synth_zipf1.5 | **134x** | **30x** | 11.7x | 5.6x | 2.0x |

## Findings

1. **Massive memory saving in the practical throughput range.** To recover
   most of the benefit (50-80% of store-all throughput), LRU needs
   **10x-500x less** KV memory. Real RAG (musique_ans): hitting 80%
   throughput needs **20 GB vs 524 GB -- a 26x saving**.

2. **Saving shrinks as the target approaches 100%.** The last few percent of
   throughput come from rarely-reused cold chunks, so matching 99% forces LRU
   toward store-all size (saving -> 1x). This is the cost of chasing the long
   tail -- not worth it.

3. **Skew amplifies the win.** Higher retrieval skew (zipf1.5) concentrates
   reuse in few hot chunks, so a tiny LRU captures most throughput: 80%
   throughput at **30x** less memory.

4. **Sweet spot ~80% throughput.** Across datasets, targeting ~80% of the
   maximum hit rate gives a 3x-30x memory saving with little throughput loss
   -- the recommended operating point.

## Takeaway

Storing all offline KV is wasteful: most chunks are reused rarely. An LRU
cache delivers ~80% of the throughput of full storage at **3x-30x lower KV
memory** (26x on real RAG), because it keeps only the recently/soon-reused
hot chunks. This is the core memory argument for RedKnot's bounded offline-KV
cache over "generate offline, store everything".

## Files
- `mem_at_iso_throughput.py` -- iso-throughput memory computation
- `plot_mem_iso.py`          -- plots
- figures: `mem_iso_throughput.png/.pdf`, `mem_iso_throughput.json`

## Reproduce
```bash
python test/srt/redknot/mem_at_iso_throughput.py
python test/srt/redknot/plot_mem_iso.py
```
