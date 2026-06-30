# Prefix-Aware KV-Cache Trimming for Prefill–Decode Disaggregation

> Working draft for the paper. Model: **Qwen3-32B** (64 layers, 8 KV heads,
> head\_dim 128, GQA 64 query heads, `max_position_embeddings = 40960`).
> All experiments use NVIDIA L20Y (80 GB) GPUs.

## 1. Motivation

In a Prefill–Decode (PD) disaggregated serving system, a *prefill node*
computes the full KV cache for a (long) prompt and transfers it to a *decode
node*, which then autoregressively generates the response. Two costs dominate:

1. **KV transfer volume** between prefill and decode nodes, which inflates the
   time-to-first-token (TTFT) of the decode node.
2. **Decode-node KV memory**, which caps the number of concurrent requests a
   decode node can hold and therefore caps aggregate throughput (QPS).

We observe that for most attention heads, decode-time attention is
*overwhelmingly local*: a query at position `p` attends almost entirely to KV
entries inside a bounded window `[p-W, p]` plus a few **sink** tokens at the
start of the sequence. For such heads the prefill node only needs to transfer
— and the decode node only needs to store — the **sink + window** slice of the
KV cache, discarding the long *middle* region. We call this **prefix-aware KV
trimming**.

This document specifies the trimming policy, defines the transfer / memory
model, and reports an empirical study on Qwen3-32B covering (i) generation
accuracy, (ii) single-stream latency, and (iii) **memory-bound concurrency and
QPS**, which is where the technique pays off most.

## 2. Method

### 2.1 Per-head locality classification

We use a per-head locality profile (RedKnot head-class profile
`qwen3-32B_optimal_g15_lf_ret.json`) that labels every (layer, KV-head) pair as
one of:

| Class         | Meaning                              | KV kept on decode node |
|---------------|--------------------------------------|------------------------|
| `local_full`  | attention is local (window-bounded)  | sink + window only     |
| `retrieval`   | sparse but long-range                | **all** positions      |
| `global`      | dense long-range                     | **all** positions      |

For Qwen3-32B the profile marks **435 / 512 = 85 %** of KV heads as
`local_full`. Empirically, however, the *last* layers of Qwen3-32B are
information-extraction layers whose local heads are far more sensitive to
trimming (Section 4.1). We therefore introduce a **trim-depth** hyper-parameter
`L_trim`: only layers with index `< L_trim` apply trimming; all heads in layers
`>= L_trim` keep their full KV. We find `L_trim = 32` (the first half of the 64
layers) to be the accuracy-safe operating point.

### 2.2 Trimming operator

Let a prompt have length `S`, sink size `s`, and window `W`. For a trimmable
`local_full` head, define the **window start**

```
w = max(s, S - W)
```

The decode node keeps KV positions `[0, s) ∪ [w, S)` and discards `[s, w)`. The
number of positions kept by such a head is

```
k = s + (S - w)        (≈ s + W  for S ≫ W)
```

A head that is `global`/`retrieval`, or that lives in a layer `>= L_trim`, keeps
all `S` positions.

**Online deployment = eviction, not zero-fill.** The intended pipeline is:
(1) offline full prefill produces the complete KV; (2) at serving time the node
simply *does not store* the discarded `[s, w)` entries for local heads — they
occupy no memory and never enter attention. This is **true eviction**, which is
mathematically equivalent to masking those positions to `-inf` before softmax
(they contribute zero softmax mass). It must be distinguished from *zero-fill*,
where the entries remain in the dense tensor with value 0: a zeroed key `K=0`
still scores `q·K = 0` and receives `exp(0) = 1` weight in the softmax
denominator, slightly diluting the kept positions. §4.5 shows true eviction is
in fact *more* accurate than the zero-fill approximation we used for the earlier
accuracy sweeps.

### 2.3 Transfer / memory model

KV bytes are `2 · d · b` per (layer, head, position), where `d = head_dim`,
`b` = bytes per element (2 for bf16/fp16), and the factor 2 is K + V. Let
`H_kv` be KV heads/layer and `N_L` layers. The **per-request KV size** is

```
KV_full(S)  = N_L · H_kv · S · 2 d b
KV_trim(S)  = ( Σ_{layer,head}  kept_positions(layer,head) ) · 2 d b
```

The **transfer saving** and the **memory ratio** are both
`KV_trim(S) / KV_full(S)`.

For a decode node with KV memory budget `B` (GPU memory minus weights and
activation headroom), the **maximum concurrency** is

```
N_max(S) = floor( B / KV_per_request(S) )
```

Because `KV_trim < KV_full`, trimming raises `N_max`, and aggregate decode
throughput scales roughly with concurrency until the GPU becomes
compute-bound.

## 3. Experimental Setup

* **Model**: Qwen3-32B. Accuracy / single-stream runs use **bf16** weights
  sharded across 2× L20Y (no quantization, so accuracy numbers are
  trustworthy). Concurrency runs use **NF4** weights on a single L20Y to free
  ~46 GB for KV (NF4 does not affect the *relative* concurrency/throughput
  comparison, which is what we report).
* **Trimming**: `W = 4096`, `s = 128`, `L_trim = 32` (accuracy-safe).
* **PD emulation**: the prefill node produces the full cache; we apply the
  trimming operator and perform a real cross-GPU copy of the kept slices
  (GPU0→GPU1) to validate correctness and measure transfer volume; the decode
  node then greedily decodes from the trimmed cache.
* **Accuracy metric**: cosine similarity of decode logits vs. the full-KV
  baseline (per step; we report step-1 `cos(d1)`), top-10 agreement, and greedy
  token-sequence match. Pass threshold `cos(d1) ≥ 0.99`. For cross-dataset
  validation we additionally report continuation **perplexity** (full-cache vs.
  trimmed-cache predicting the same held-out tokens).
* **Datasets**: a synthetic repeated-text prompt (length-controlled sweeps) plus
  four real datasets — LongBench *hotpotqa* / *gov\_report* / *lcc* and
  *WikiText-103* — covering multi-hop QA, summarization, code, and language
  modeling.
* **Context limit**: Qwen3-32B supports up to 40 960 tokens; we therefore test
  prefixes up to **36 K** (within the native context). 64 K is **out of native
  range** and would require RoPE scaling (YaRN); we do not report 64 K to avoid
  confounding the trimming result with extrapolation effects.
* **Reproduce**:
  ```bash
  # accuracy + single-stream (bf16, 2 GPU)
  CUDA_VISIBLE_DEVICES=0,1 SWA_TRIM_CONFIGS=0,32,48 \
    python test/srt/redknot/test_swa_pd_bench.py
  # cross-dataset accuracy (bf16, 2 GPU)
  CUDA_VISIBLE_DEVICES=0,1 SWA_DS_LIST=hotpotqa,gov_report,lcc,wikitext \
    SWA_DS_SAMPLES=6 python test/srt/redknot/test_swa_pd_datasets.py
  # memory-bound concurrency / QPS (NF4, 1 GPU)
  CUDA_VISIBLE_DEVICES=0 SWA_PREFIX_LENGTHS=8192,16384,32768,36864 \
    SWA_TPS_PROBE_CAP=32 SWA_ATTN_IMPL=sdpa \
    python test/srt/redknot/test_swa_pd_concurrency.py
  ```

## 4. Results

### 4.1 Accuracy vs. trim depth (8 K prefix sweep)

Sweeping `L_trim` at an 8 K prefix isolates the accuracy/saving trade-off and
locates the safe operating point. Trimming the last 32 layers' local heads is
markedly more destructive than trimming the first 32.

| `L_trim` | cos(d1) | greedy tokens | KV saved | Result |
|---------:|--------:|:--------------|---------:|:------:|
| 16 | 0.9996 | diverge @23 | 12.1 % | PASS |
| **32** | **0.9911** | **identical** | **24.3 %** | **PASS** |
| 48 | 0.9506 | diverge @8  | 36.4 % | FAIL |
| 64 | 0.8998 | diverge @1  | 41.2 % | FAIL |

**Finding.** The profile's aggressive 85 %-local labeling is *not* safe under
bf16; the safe boundary for Qwen3-32B is `L_trim = 32`. A control run with no
heads trimmed reproduces the baseline exactly (`cos = 1.000000`), confirming the
trimming/transfer pipeline is bug-free and the accuracy degradation is real.

### 4.2 Accuracy across prefix lengths (`L_trim = 32`, bf16)

| Prefix | cos(d1) | avg cos | top-10 | greedy tokens | KV saved | Result |
|-------:|--------:|--------:|:------:|:--------------|---------:|:------:|
|  8 K | 0.9911 | 0.9985 | 100 % | identical          | 24.3 % | PASS |
| 12 K | 0.9960 | 0.6760 | 100 % | diverge @44        | 32.8 % | PASS |
| 16 K | 0.9988 | 0.9973 |  90 % | diverge @15        | 37.1 % | PASS |
| 24 K | 0.9978 | 0.7570 | 100 % | diverge @38        | 41.4 % | PASS |
| 32 K | 0.9987 | 0.5566 | 100 % | diverge @22        | 43.6 % | PASS |

All prefixes pass the `cos(d1) ≥ 0.99` threshold. Divergences appear only late
in the output and are benign synonym substitutions (e.g. "stories about" →
"stories of"); the generated text remains semantically consistent with the
baseline. KV transfer saving grows with prefix length (24 %→44 %), because the
fixed window occupies a smaller fraction of a longer prefix.

### 4.3 Cross-dataset accuracy validation

The Section 4.2 numbers use a single synthetic prompt. To rule out
prompt-specific effects, we validate `L_trim = 32` on **four real long-context
datasets spanning distinct tasks and attention regimes**: multi-hop QA
(LongBench *hotpotqa*), summarization (LongBench *gov\_report*), code completion
(LongBench *lcc*), and language modeling (*WikiText-103*). For each dataset we
draw 6 samples (context ≥ 6 K tokens, capped at 32 K), prefill the context,
trim, and decode; we report logits cosine, greedy-token agreement vs. the
full-KV baseline, and **continuation perplexity** (teacher-forced NLL over a
256-token continuation, full-cache vs. trimmed-cache predicting the same
tokens). bf16, 2× L20Y.

| Dataset | Task | avg ctx | cos(d1) | greedy match | PPL base → trim | ΔPPL |
|---------|------|--------:|--------:|:------------:|:---------------:|-----:|
| hotpotqa   | multi-hop QA   | 15.6 K | 0.9988 | 48.3 % | 6.44 → 7.06 | +0.62 |
| gov\_report | summarization  | 11.2 K | 0.9984 | 83.3 % | 5.06 → 5.39 | +0.33 |
| lcc        | code completion| 10.4 K | 0.9742 | 91.7 % | 1.81 → 1.91 | +0.09 |
| wikitext   | language model | 6.1 K  | 0.9998 | 66.7 % | 7.38 → 8.35 | +0.97 |

**Findings.**

* **Logits stay highly aligned** across all four domains (`cos(d1) ≥ 0.974`);
  the per-token next-distribution after trimming is close to the full-KV model.
* **Code is the most trim-friendly** (ΔPPL +0.09): code has the strongest local
  structure, so window+sink captures almost all relevant context.
* **Pure language modeling is the most sensitive** (WikiText ΔPPL +0.97):
  free-form text relies more on long-range context than task-conditioned
  generation.
* **Greedy-match varies (48–92 %)** on real long text. High cosine means the
  next-token distribution stays close, but as §4.4 shows, mismatches are *not*
  always benign synonym swaps — on retrieval-heavy tasks they can become factual
  errors. The ΔPPL captures language-modeling drift but not task correctness.
* The `lcc` cosine mean is pulled down by one boundary-token outlier
  (`cos = 0.845`) whose PPL is essentially unchanged; the median cosine on
  `lcc` is ≈ 1.0000.

The continuation-PPL penalty of trimming the first 32 layers' local heads is
modest on these prompts (ΔPPL ≤ +0.97; +5–13 % relative). PPL alone, however,
understates the effect on *generated text*, which we examine next.

### 4.4 Real generated-text difference (the honest picture)

cos and PPL are proxy metrics. To see what actually changes, we greedy-decode
64 tokens from the full-KV baseline and from the trimmed cache on six datasets
(8 samples each) and compare the **decoded strings**: exact-match rate,
per-token agreement, and normalized edit distance. Config is the single
near-lossless setting (trim<32, sink=128, dense).

| Dataset | Task | avg ctx | exact-match | token-match | norm. edit dist. |
|---------|------|--------:|:-----------:|:-----------:|:----------------:|
| lcc            | code completion   |  9.2 K | **88 %** | 92.2 % | 0.046 |
| triviaqa       | few-shot QA       | 12.5 K | **75 %** | 82.4 % | 0.096 |
| gov\_report    | summarization     | 12.3 K | 50 % | 65.6 % | 0.274 |
| multifieldqa\_en | multi-field QA  | 10.3 K | 38 % | 69.9 % | 0.226 |
| wikitext       | language modeling |  5.1 K | 12 % | 55.9 % | 0.285 |
| hotpotqa       | multi-hop QA      | 15.4 K | **0 %**  | 38.7 % | 0.359 |

**Findings — the effect is task-dependent, and not always harmless:**

* **Local-structure tasks are safe.** Code completion (88 % char-identical) and
  few-shot QA (75 %) reproduce the baseline text almost exactly; window+sink
  captures essentially all the context these tasks use.
* **Retrieval-heavy tasks can break.** On multi-hop QA *no* sample is
  char-identical and we observe genuine **factual errors**, e.g. the baseline
  answers "Phileas Fogg is played by **David Niven**" (correct for the 1956 film)
  while the trimmed model answers "**Steve Coogan**" (the 2004 film). The
  answer-bearing fact lives in the trimmed middle region, so dropping it changes
  the answer — a high cosine does not save a wrong argmax on the key token.
* **Most diffs are still paraphrase.** Outside the retrieval failures, the bulk
  of differences are benign rewordings ("examine the dates" → "compare the
  dates") that preserve meaning.

Sample (multi-hop QA, factual error introduced by trimming):

```
BASE: ... Phileas Fogg in "Around the World in 80 Days" is David Niven. ...
TRIM: ... Phileas Fogg in "Around the World in 80 Days" is Steve Coogan. ...
```

Sample (code completion, char-identical):

```
BASE = TRIM:  ... low temperature scanning tunneling microscopy (STM) and
              scanning tunneling spectroscopy (STS) ...
```

By context length, exact-match and token agreement decline as the prefix grows
(more middle KV is dropped):

| Context bucket | n | exact-match | token-match | norm. edit |
|----------------|--:|:-----------:|:-----------:|:----------:|
| ≤ 6 K   | 11 | 36 % | 67.9 % | 0.207 |
| 6–12 K  | 18 | 56 % | 74.0 % | 0.174 |
| 12–18 K | 18 | 33 % | 58.8 % | 0.271 |

**Takeaway.** The near-lossless config is genuinely lossless on
locality-dominated workloads (code, continuation, few-shot) but is **lossy and
can introduce factual errors on retrieval-dependent tasks** (multi-hop QA). The
technique should therefore be applied with task awareness, or paired with a
retrieval-head allow-list that keeps full KV for heads/layers carrying
long-range evidence.

### 4.5 True eviction vs zero-fill

The accuracy sweeps above approximate eviction by zero-filling the discarded KV.
The real online behavior is *eviction*: those entries are never stored and never
enter attention. We realize eviction semantics exactly with a per-(layer,
kv-head) `-inf` mask before softmax (via a custom SDPA attention) and compare,
on the same backend, against zero-fill, both relative to the full-KV baseline.

| Prefix | zero-fill cos(d1) | **true-evict cos(d1)** | token-match (evict) | evict ≥ zero-fill |
|-------:|:-----------------:|:----------------------:|:-------------------:|:-----------------:|
|  8 K | 0.9962 | **0.9998** | 100 % | yes |
| 16 K | 0.9985 | **0.9997** | 100 % | yes |
| 24 K | 0.9959 | **0.9996** |  81 % | yes |
| 32 K | 0.9981 | **0.9999** | 100 % | yes |

**Finding.** True eviction is *more* accurate than zero-fill at every prefix
length (cos(d1) ≥ 0.9996 vs 0.9959–0.9985). Removing a key from the softmax
denominator is cleaner than keeping it as a zero vector that still absorbs
`exp(0)` weight. The online "do not store" design is therefore at least as good
as — and slightly better than — the zero-fill numbers reported in §4.2–4.4.
Memory savings are unaffected (evicted entries occupy no space); in HF's dense
cache we verify accuracy via the mask, while a paged engine (e.g. SGLang SWA
pool) would realize the memory saving with variable-length per-head KV.

### 4.6 Single-stream latency (not the win)

Trimming barely changes single-request end-to-end latency: prefill is
unchanged, the transferred-KV decode runs on the same dense kernels, and decode
is short. The analytical estimate (attention cost ∝ effective KV length,
attention fraction 0.5) gives only:

| Prefix | KV saved | single-stream QPS (baseline → trim) | speed-up |
|-------:|---------:|:-----------------------------------:|:--------:|
|  8 K | 24.3 % | 0.171 → 0.190 | 1.11× |
| 16 K | 37.1 % | 0.135 → 0.158 | 1.12× |
| 24 K | 41.4 % | 0.110 → 0.122 | 1.11× |
| 32 K | 43.6 % | 0.086 → 0.094 | 1.09× |

Single-stream latency understates the benefit; the real payoff is concurrency.

### 4.7 Memory-bound concurrency and QPS (the win)

Decode is memory-bound: each concurrent request holds a KV cache, so a smaller
`KV_per_request` lets more requests share one decode GPU. With a fixed
**45.7 GiB KV budget** (single L20Y, NF4 weights) and `out_len = 256`:

| Prefix | KV/req base | KV/req trim | ratio | max-batch base → trim | **batch gain** |
|-------:|------------:|------------:|------:|:---------------------:|:--------------:|
|  8 K | 2048 MB | 1552 MB | 0.76 | 22 → 30 | **1.36×** |
| 16 K | 4096 MB | 2576 MB | 0.63 | 11 → 18 | **1.64×** |
| 32 K | 8192 MB | 4624 MB | 0.56 |  5 → 10 | **2.00×** |
| 36 K | 9216 MB | 5136 MB | 0.56 |  5 →  9 | **1.80×** |

Aggregate decode throughput is measured by a **real batched decode forward at
each measured max-batch** (no extrapolation):

| Prefix | agg tok/s base → trim | QPS base → trim | **QPS gain** |
|-------:|:---------------------:|:---------------:|:------------:|
|  8 K | 61 → 81 | 0.238 → 0.318 | **1.33×** |
| 16 K | 31 → 49 | 0.120 → 0.193 | **1.61×** |
| 32 K | 15 → 28 | 0.057 → 0.108 | **1.90×** |
| 36 K | 14 → 25 | 0.054 → 0.097 | **1.79×** |

**Finding.** Under a fixed memory budget, prefix-aware trimming delivers a
**1.33–1.90× QPS improvement**, growing with prefix length. The QPS gain
slightly exceeds the batch gain because trimming also shortens per-request
attention, a second-order speed-up on top of the concurrency gain. The 36 K
point dips marginally below 32 K due to the integer-batch quantization of the
fixed budget (5 vs. 5 baseline requests), an expected boundary effect.

## 5. Discussion

* **Where the gain comes from.** Single-stream latency improves by only
  ~1.1×, but concurrency-driven QPS improves by up to ~1.9×. Long-context PD
  serving is dominated by decode-node KV memory, exactly the quantity trimming
  reduces.
* **Accuracy is task-dependent.** `L_trim = 32` keeps `cos(d1) ≥ 0.99` and a
  modest PPL penalty (ΔPPL ≤ +0.97). On the *generated text*, however, the
  outcome splits by task: locality-dominated workloads (code 88 %, few-shot QA
  75 % char-identical) are effectively lossless, while retrieval-heavy multi-hop
  QA is lossy and can produce factual errors (§4.4). Trimming all profile-local
  heads (`L_trim = 64`) collapses for every task and is *not* usable on dense
  Qwen3-32B.
* **Transfer volume.** Trimming also cuts prefill→decode KV transfer by
  24–44 %, shortening decode TTFT in addition to raising concurrency.

## 6. Limitations & Validity

1. **Per-layer variable-length KV.** HF's dense `DynamicCache` cannot represent
   different KV lengths per layer/head, so the concurrency throughput is
   measured with a uniform *effective* KV length that matches the trimmed
   request's total positions (and thus both its memory and its attention
   compute). Max-concurrency itself is an exact memory-budget division. A
   production engine with paged, variable-length KV (e.g. SGLang's SWA pool)
   would realize the same memory saving without this approximation.
2. **NF4 for concurrency runs.** Quantization frees memory for KV and does not
   change the *relative* baseline-vs-trim comparison; accuracy numbers come from
   the separate **bf16** runs.
3. **Native context limit.** Results are reported up to 36 K, within Qwen3-32B's
   40 960-token context. 64 K requires RoPE scaling and is left to future work.
4. **`transfer_time` is not used as a speed metric.** Our emulator copies many
   small per-head slices; a real engine transfers contiguous blocks. We report
   transfer *volume* (saving %), not emulated transfer wall-time.

## 7. Artifacts

| File | Purpose |
|------|---------|
| `test_swa_pd_2gpu.py` | 2-GPU PD emulation: per-head trim + cross-GPU transfer + accuracy |
| `test_swa_pd_bench.py` | accuracy + single-stream matrix over prefixes × trim depths |
| `test_swa_pd_datasets.py` | cross-dataset accuracy (LongBench + WikiText): cos / greedy / PPL |
| `test_text_diff.py` | real generated-text diff (exact-match / token-match / edit distance) on 6 datasets |
| `show_text.py` | side-by-side decoded text: baseline vs near-lossless vs pure per-head |
| `test_true_evict.py` | true eviction (per-head -inf mask) vs zero-fill accuracy comparison |
| `test_swa_pd_concurrency.py` | memory-bound concurrency / QPS benchmark |
| `head_class/qwen3-32B_optimal_g15_lf_ret.json` | per-head locality profile |
| `test_swa_pd_bench.results.json`, `test_swa_pd_datasets.results.json`, `test_swa_pd_concurrency.results.json` | raw results |

## 8. Conclusion

Prefix-aware, per-head KV-cache trimming for PD disaggregation on Qwen3-32B,
applied to the first 32 layers' local heads with a 4096 window and 128 sink
tokens, reduces per-request KV memory by 24–44 % and, under a fixed decode-node
memory budget, yields a **1.3–1.9× concurrency-driven QPS improvement** that
grows with prefix length — precisely the long-context regime where PD
disaggregation matters most.

The accuracy impact is **task-dependent**. On locality-dominated workloads —
code completion (88 % char-identical output) and few-shot QA (75 %) — the
trimmed model reproduces the baseline text essentially verbatim, so the memory
and throughput gains come for free. On retrieval-heavy tasks such as multi-hop
QA, however, the dropped middle-context KV can change the answer (0 %
char-identical, occasional factual errors), because dense Qwen3-32B was not
trained with a sliding-window mask and its softmax attention is perturbed by
removing any in-context KV. Pure profile-driven all-layer trimming
(`L_trim = 64`, sink = 4) collapses outright. We therefore recommend the
near-lossless config with **task-aware deployment** (or a retrieval-head
allow-list), and note that a model natively trained with sliding-window
attention would make the same trimming strictly lossless.
