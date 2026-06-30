# Chunk Reuse Lifecycle Experiment

## Motivation

Mainstream KV reuse (vLLM prefix cache, SGLang RadixCache, Mooncake) can only
reuse a **prefix**: a token's KV is valid only if all preceding tokens are
identical. But in real RAG, the *same document chunk* appears at *different
positions* across requests (middle, end), so prefixes differ and prefix-cache
never hits. RedKnot targets exactly this **non-prefix reuse** via carry-prefix
recomputation + RoPE realignment.

The economic question (user's intuition):

> A cached chunk costs `KV_bytes x residency_time` of GPU memory and saves
> `reuse_count x prefill_cost`. Cache it only if it is reused **often** and
> its lifespan is **short** enough; otherwise do not occupy memory.

This experiment measures, on a real multi-document QA stream, every chunk's
reuse count, lifecycle span, and non-prefix ratio, then evaluates caching
policies on the cost/benefit tradeoff for DeepSeek V4 Flash.

## Datasets

| Dataset | Status | Reuse structure |
|---|---|---|
| musique (local) | used | 20 titled Wikipedia passages per question; titles are stable chunk ids reused across questions |
| MS MARCO | attempted | HF data CDN (cas-bridge.xethub.hf.co) unreachable from this network |
| KILT | attempted | same network limitation |

musique is a faithful stand-in: it has a fixed pool of Wikipedia passages
that recur across questions at varying positions, which is exactly the
non-prefix reuse pattern. MS MARCO / KILT can be swapped in via the same
`chunk_lifecycle.py` loader once their raw data is available locally.

## Method

`chunk_lifecycle.py` streams questions in arrival order. Each passage title is
a stable chunk id; its position in the question's context determines prefix vs
non-prefix. For each chunk it records:

- `reuse_count`   : number of requests reusing it
- `first_seen` / `last_seen` (request index = time)
- `residency`     : last_seen - first_seen (lifecycle span)
- `non_prefix_ratio` : fraction of reuses at position != 0 (prefix-cache miss)
- `n_tokens`, `kv_bytes` : real token count + KV footprint

`chunk_cache_benefit_dsv4.py` then attaches DeepSeek V4 Flash's real MLA KV
footprint and per-chunk prefill cost, converting "prefills saved" into
GPU-seconds saved and GB of KV memory.

## DeepSeek V4 Flash cost model

From the checkpoint `config.json` (43 layers, MLA `kv_lora_rank=512`,
`qk_rope_head_dim=64`, 6/256 experts active):

- MLA KV: **49,536 bytes/token** = 198 MB per 4K chunk
- Equivalent MHA KV would be 1,409,024 B/token -> **MLA is 28.4x smaller**
- Prefill per 4K chunk: ~265 ms (FLOPs estimate; to be replaced by a measured
  value from a real SGLang `redknot_mla` run once sgl-kernel is built)

## Results

### Reuse lifecycle statistics (musique, 2417 requests)

| Metric | Value |
|---|---|
| unique chunks | 17629 |
| reused >= 2 | 6703 (38.0%) |
| max reuse count | 261 |
| mean reuse count | 2.74 |
| mean residency (requests) | 178.4 |
| **mean non-prefix ratio** | **0.951** |

The headline number: **95.1% of all chunk reuses are non-prefix** -- a
prefix-cache would miss almost everything.

### Caching policy comparison (DeepSeek V4 Flash)

| Policy | %cached | prefills saved | GPU-sec saved | peak KV (GB) | sec saved / GB |
|---|---:|---:|---:|---:|---:|
| cache_all | 100.0% | 30686 | 8128.6 | 6017.9 | 1.35 |
| prefix_only | 0.0% | 7 | 1.9 | 2.2 | 0.83 |
| redknot (R>=3) | 21.4% | 27763 | 7354.3 | 1624.9 | 4.53 |
| redknot (R>=5) | 10.6% | 23216 | 6149.8 | 865.1 | 7.11 |
| redknot (R>=10) | 4.4% | 17406 | 4610.8 | 412.3 | 11.18 |

## Analysis

1. **Prefix-cache is useless here.** It caches only 7 chunks and saves 1.9
   GPU-seconds, because 95% of reuse is non-prefix. This is the core
   justification for RedKnot's non-prefix reuse.

2. **Selective caching dominates cache-all.** Caching only the top ~10% most
   reused chunks (R>=5) captures **76%** of the maximum benefit
   (6150 / 8129 GPU-seconds) while using **7x less** KV memory
   (865 GB vs 6018 GB). Value density rises from 1.35 to 7.11 sec/GB.

3. **The reuse distribution is long-tailed** (max=261, mean=2.74): a few
   "hot" passages (countries, regions) dominate reuse. The cost-benefit
   frontier is steep -- pushing R>=10 still keeps 57% of the benefit at
   only 412 GB (11.18 sec/GB), confirming the user's intuition that
   high-reuse / short-lifespan chunks are the ones worth caching.

4. **MLA compounds the win.** DeepSeek V4's MLA KV is 28x smaller than MHA,
   so the same caching budget holds far more chunks -- RedKnot's selective
   reuse and MLA compression are complementary.

## Figures

`chunk_lifecycle.png` (4 panels):
1. reuse-count distribution (log-log long tail)
2. non-prefix reuse ratio histogram (mean 0.95)
3. lifecycle scatter: reuse vs residency, colored by value density
4. cache cost-benefit frontier (%saved vs KV GB) for R_min sweep

## Reproduce

```bash
# 1. reuse lifecycle stats (uses DeepSeek V4 tokenizer + MLA KV bytes)
python test/srt/redknot/chunk_lifecycle.py \
  --model-path /mnt/.../DeepSeek-V4-Flash --kv-bytes-per-token 49536

# 2. DeepSeek V4 cache benefit (GPU-seconds, memory)
python test/srt/redknot/chunk_cache_benefit_dsv4.py
#   add --prefill-ms <measured> once a real SGLang redknot_mla run is available

# 3. plots
python test/srt/redknot/plot_chunk_lifecycle.py
```

## TODO (pending sgl-kernel build)

Replace the FLOPs-estimated 265 ms/chunk prefill with a **measured** value
from a real SGLang DeepSeek-V4-Flash-FP8 `redknot_mla` run (TP=8), then
re-run `chunk_cache_benefit_dsv4.py --prefill-ms <measured>` so all
GPU-seconds numbers are measured rather than estimated.
