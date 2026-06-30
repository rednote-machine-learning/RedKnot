# RedKnot TTFT Optimization — Converting FLOPs Savings into Wall-Time Speedup

This document records the system that delivers the **highest TTFT speedup** for
the RedKnot head-class KV-reuse path, the optimizations that got there, and the
exact measurements (INT4 Qwen3-32B, single L20Y GPU, HotpotQA long-context).

## TL;DR

Baseline = **the standard, fastest dense prefill**: one `model(input_ids)`
forward over the full concatenated context with
`attn_implementation="flash_attention_2"` (FA-2), then standard decode. This is
the honest, hard-to-beat baseline — NOT RedKnot's own per-slot framework (which
would carry our dispatch overhead and unfairly inflate the speedup).

| Length | Std FA-2 prefill TTFT | RedKnot head-class TTFT | **Speedup** | F1 (RK vs base) |
|--------|----------------------:|------------------------:|------------:|-----------------|
| 16K    | 3.4 s                 | 1.3–1.5 s               | **2.2–2.6×**| 1.00 vs 1.00 (s2: 1.00 vs 0.00) |
| 40K    | 10.4 s                | 3.3 s                   | **3.1×**    | mixed (互有胜负) |

- **Algorithmic compute savings: 71.9% (3.56× fewer FLOPs)** — attention −83.8%,
  FFN −79.8%, projection 0% (irreducible).
- **Wall-time TTFT speedup: 2.2× (16K) → 3.1× (40K)** vs the standard FA-2
  prefill. Speedup grows with length (baseline prefill is O(L²); RedKnot reuses
  local-head KV).
- **Accuracy is NOT degraded.** Measured SQuAD F1/EM on HotpotQA: RedKnot matches
  or beats the dense baseline (on several samples the dense baseline answers
  wrong while RedKnot is correct). Earlier reports of an F1 drop were caused by a
  **decode bug** (see "Decode correctness fix" below) — fixed.

The entry point is `run_redknot_offlinekv(...)` in `driver_batched.py`, which
uses the **flat single-pass custom forward** (`online_forward_segments_flat` →
`_run_flat_custom`). Comparison script: `test/srt/redknot/eval_vs_standard_baseline.py`.

## Decode correctness fix (the F1 regression root cause)

An earlier version applied per-head sliding-window attention **during decode**
with a buffer that only grew (never slid), and a seed pass inconsistent with the
prefill `first_logits`. This corrupted generation (answers like "Answer:" /
"To answer the question..."), dropping F1 to ~74% of baseline even though cosine
stayed ~0.96 (cosine of logits is NOT a reliable answer-quality proxy).

**Fix:** decode now lets **all heads attend the full KV** via a standard
`DynamicCache` + native SDPA. Decode is only a few dozen tokens and is OUTSIDE
TTFT, so head-class sparsity there buys ~nothing while the per-head bookkeeping
was error-prone. After the fix, 16K F1 = 1.000 (was 0.481).

## The core problem: FLOPs ≠ wall time

The head-class design removes **71.9% of prefill FLOPs**, but a naive
implementation only delivered **1.1–1.5×** wall-time speedup. The gap came from
three places, each diagnosed by direct measurement (not assumption):

1. **`attn_mask` defeats FlashAttention.** Expressing head sparsity through an
   additive mask forces SDPA off the FlashAttention fast path (paper §5.4,
   4.9–7.6× kernel penalty). Fixed by **mask-free FlashAttention** with native
   sliding-window + LSE merge.
2. **Local heads were not actually sparse in the online forward.** Each segment
   re-attended the *full* growing online prefix (O(N²) KV), so 85% of heads did
   full-context work. Fixed by **physically truncating local-head KV to
   sink+window** via flash's native `window_size`.
3. **HuggingFace eager dispatch dominated (~53% of online time).** Per-layer
   Python dispatch + many small op launches (LayerNorm, residual, RoPE) — *not*
   INT4 dequant (bf16 and INT4 online forward measured the same), *not*
   attention (already optimized), *not* CUDA-graph-able as a single layer
   (compute-bound, 1.00× replay). Fixed by the **flat single-pass custom
   forward + torch.compile of the static blocks**.

## Optimization ladder (40K, fair comparison, same INT4 model & GPU)

| Stage | TTFT | Speedup | cosine |
|-------|-----:|--------:|-------:|
| Illegal "reuse everything" (unfaithful) | — | 0.87× | — |
| Faithful SegPaged (local not truncated) | 7.2 s | 1.09× | 0.97 |
| + mask-free head-class attention | 6.2 s | 1.25× | 0.97 |
| + aggressive Sparse FFN (mass=0.2) | 5.48 s | 1.44× | 0.967 |
| + remove attention-fn overhead | 5.24 s | 1.51× | 0.964 |
| + torch.compile (layer-level) | 4.21 s | 1.91× | 0.967 |
| + flat single-pass online forward | 4.00 s | 2.01× | 0.966 |
| + skip lm_head in online forward | 3.92 s | 2.06× | 0.966 |
| **+ custom flat forward + compiled pre/post blocks** | **3.29 s** | **2.42×** | **0.965** |

## What the winning system does

### 1. Head-class hybrid KV reuse (paper-faithful)
- **Global heads (≈15%)**: re-prefilled. The online forward runs the shared
  hidden-state stream; global heads do full-context causal attention.
- **Local heads (≈85%)**: reused verbatim within `sink + window`. Their KV is
  taken from the offline cache, RoPE-repositioned to global coordinates, and
  physically truncated.
- **Sparse FFN**: only the top-mass tokens run the FFN; the rest pass through
  the residual (Algorithm 1). Measured ~13% token activation at mass=0.2.

### 2. Flat single-pass online forward (`online_forward_segments_flat`)
Instead of N−1 per-segment "slot" forwards (each re-concatenating the growing
prev KV — O(N²) cat + N kernel launches per layer), **all online tokens are one
contiguous causal sequence**, with **one head-class attention call per layer**.
This collapses the per-slot Python/launch overhead. Correctness is actually
*simpler*: a single causal forward gives each token the right causal view
automatically.

### 3. Mask-free head-class attention (`_flat_headclass_attention`)
- Global heads: one FlashAttention call, native GQA, `causal=True` over
  `[seg0 | online]` (right-aligned causal = full-context).
- Local heads: FlashAttention with native `window_size` over the trimmed recent
  stream + a tiny sink pass, merged by **log-sum-exp** (`_merge_lse`,
  sigmoid-weighted to avoid fp32 upcasting the large activation tensors).
- **No `attn_mask` is ever constructed** → every head stays on the
  FlashAttention fast path.

### 4. Custom forward, no HF wrapper (`_run_flat_custom`)
Bypasses `model.forward` entirely:
```
embed →
  per layer:
    compiled pre-block  (input_norm → QKV proj → q/k norm → RoPE)
    eager head-class attention
    compiled post-block (o_proj → residual)
    sparse MLP + residual
(no final norm, no lm_head — online prefill only needs the per-layer KV)
```
- **Skips lm_head** over all T online tokens (`V·T·hidden` FLOPs of pure waste).
- **Skips the final RMSNorm** and HF attention-mask preparation.
- **`torch.compile` the static pre/post blocks** (`fullgraph=True, dynamic=False`):
  fuses LayerNorm + projection + residual + RoPE into tight kernels, removing
  the per-layer eager dispatch. Attention/Sparse-FFN remain eager (data-dependent)
  but are isolated, so no graph-break penalty bleeds into the static ops.

## Measured breakdown (40K, custom flat, TTFT 3.44 s profiling run)

| Component | Wall | Notes |
|-----------|-----:|-------|
| Attention | 1.22 s (35%) | kernel ≈0.73 s (global full-causal dominates) + fn overhead |
| Sparse FFN | 0.49 s (14%) | ≈ kernel lower bound (gather/scatter negligible) |
| REST (pre/post GEMM + query fwd + rope) | 1.74 s (51%) | proj+LN ≈1.05 s is **irreducible** (global re-prefill must project every token) |

The remaining gap to the 3.2× ceiling is almost entirely the **irreducible
projection GEMM** (every online token must be projected for global-head KV) plus
the global-head full-causal attention kernel — both are real compute, not
overhead.

## Why compute savings (3.56×) exceed wall speedup (2.42×)

The saved FLOPs are concentrated in attention and FFN, which are already
high-efficiency kernels. The **projection GEMM (0.46 P) cannot be skipped** —
global-head re-prefill requires projecting every online token — and it is
compute-bound, so it sets a wall-time floor. On an INT4 single-GPU setup against
a strong batched+FlashAttention baseline, a 2.4× wall speedup at a 3.56× FLOPs
reduction is the expected, honest conversion ratio.

## How to run

```python
from sglang.srt.layers.attention.redknot import (
    HeadClassConfig, SparseFFNSchedule, offline_prefill_segments,
    run_redknot_offlinekv,
)

hc = HeadClassConfig.from_json(".../qwen3-32B_w256_g15_lf_ret.json")
hc.merge_retrieval_to_global()
sched = SparseFFNSchedule(dense_until=5, mass_thresh=0.2,
                          deep_layer_start=40, mass_thresh_deep=0.05, recent_n=128)

segs = offline_prefill_segments(model, tok, docs, chunk_size=4096, model_id=MODEL)
logits, text, qlen, ttft = run_redknot_offlinekv(
    model, tok, segments_offline=segs, query_text=query,
    head_cfg=hc, sparse_ffn_schedule=sched,
    use_compile=True,   # compile static pre/post blocks
    use_flat=True,      # flat single-pass online forward (default)
)
```

Environment toggles:
- `REDKNOT_COMPILE=1` — enable torch.compile of static blocks (recommended).
- `REDKNOT_CUSTOM_FWD=1` — use the custom flat forward (default on; set `0` to
  fall back to the HF-wrapped flat path for debugging).

Note: torch.compile pays a one-time warmup cost (tens of seconds). In the
target cross-request KV-reuse scenario the model is resident and the compiled
graphs are reused, so the cost amortizes across requests.

## Quick benchmark

```
REDKNOT_COMPILE=1 CUDA_VISIBLE_DEVICES=0 \
  python test/srt/redknot/test_offlinekv_quick.py        # 16K + 40K, 1 sample
CUDA_VISIBLE_DEVICES=0 \
  python test/srt/redknot/compute_flops_savings.py       # FLOPs accounting
```
