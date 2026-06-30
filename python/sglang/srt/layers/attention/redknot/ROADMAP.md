# RedKnot Implementation Roadmap

This document tracks the staged plan to bring the RedKnot integration in this
repository to full paper parity, and pins down the immediate Stage A / Stage B
work.

Paper: RedKnot — head-classified KV reuse + Elastic Sparsity (head-aware
attention recovery + Sparse FFN) on a SegPagedAttention runtime.

Legend: `[ ]` todo, `[~]` partial / prototype exists, `[x]` done.

---

## Current Status Snapshot

- [x] Head class config (`global / local / retrieval / dense`) + loader
- [x] Offline segment KV cache + RoPE realignment (RoPE scaling bug fixed)
- [x] Head-aware attention recovery (FA-2 / FA-3 buckets)
- [x] Sparse FFN recovery (token-selective FFN, Algorithm 1 lines 20-23)
- [x] Parallel / batched online prefill (`run_redknot_batched`)
- [x] SegPagedAttention primitives: per-head page table + KV store
- [x] SegPagedAttention synthetic / DuoAttention test (cosine, KV saving)
- [x] SegPaged decode path + standalone `--attention-backend segpaged`
- [x] Head profiling, PD transfer model, head-aware scheduler/capacity model
- [x] Naming migration `redcache -> redknot`
- [~] SegPaged in the **prefill/extend** path (decode only today)
- [ ] Sparse FFN fused with SegPaged prefill metadata
- [ ] Production SGLang backend (token_to_kv_pool / RadixCache page-aware)
- [ ] PD disaggregation runtime KV transfer (only modeled, not wired)
- [ ] Baseline comparisons (CacheBlend / ProphetKV)
- [ ] Concurrency / throughput serving benchmark

---

## Stage A — SegPaged prefill prototype (target: ~3 days)

Goal: replace the `gather + cat -> FlashAttention` online prefill with a
per-head page-table SegPagedAttention prefill in the standalone driver, and
prove it is numerically equivalent and at least as fast.

- [ ] A1. Add `run_redknot_batched_segpaged(...)` in `driver_batched.py`
      (new path; keep the existing one intact for comparison).
- [ ] A2. Per layer, build a `SegPagedKVCache`:
      - global heads -> full-context pages
      - local heads  -> sink + recent-window pages only
      - retrieval heads -> top-p selected pages
- [ ] A3. Run attention via `segpaged_attention(...)` instead of building a
      dense `prev_k`/`prev_v` via `torch.cat`.
- [ ] A4. Preserve GQA fanout without copying KV per q-head
      (one KV head segment serves `q_per_kv` query heads).
- [ ] A5. Wire RoPE realignment into the paged KV before attention.
- [ ] A6. Numerical check: cosine(dense DuoAttention, SegPaged prefill)
      > 0.99 on synthetic; > 0.95 on Qwen3 16K/32K.
- [ ] A7. Report TTFT and KV-bytes/FLOPs savings vs the current batched path.
- [ ] A8. New runnable script:
      `test/srt/redknot/test_segpaged_prefill_prototype.py`
      printing baseline vs SegPaged-prefill answer / cosine / TTFT / KV saving.

Stage A acceptance:
- [ ] Qwen3 16K and 32K run end-to-end through SegPaged prefill.
- [ ] cosine vs dense DuoAttention > 0.95.
- [ ] TTFT not worse than current batched path (ideally 10-30% better at 32K).

---

## Stage B — Sparse FFN + SegPaged fusion & accuracy/speed sweet spot (target: ~1 week)

Goal: combine Sparse FFN with the SegPaged prefill prototype and find a
config that preserves accuracy while delivering > 2x speedup at long context.

- [ ] B1. Feed SegPagedAttention output into Sparse FFN token-importance
      selection (replace the current attn-norm proxy if needed).
- [ ] B2. Keep batched/microbatched multi-segment alignment correct under
      SegPaged (importance shape `[B, L]`, residual identity for unselected).
- [ ] B3. Report per-layer deep_frac, FFN FLOPs savings, total FLOPs savings.
- [ ] B4. Accuracy/speed sweep over:
      - head config: window size, global ratio, retrieval top-p
      - Sparse FFN: `dense_until`, `mass_thresh`, `recent_n`
- [ ] B5. Select sweet spots:
      - 16K: F1 within baseline, wall speedup > 1.5x
      - 35K / 64K: wall speedup > 2x, accuracy preserved
- [ ] B6. Multi-model validation: Qwen3-32B and Llama-3.3-70B
      (verify RoPE-scaling correctness on llama3).
- [ ] B7. Persist chosen presets and document them in
      `examples/redknot/README.md`.

Stage B acceptance:
- [ ] One documented config reaching > 2x wall speedup at 35K-64K with
      accuracy on par with full-recompute baseline.
- [ ] FLOPs savings reported and consistent with attention + Sparse FFN.

---

## Stage C — SGLang serving backend (target: ~2-4 weeks)

Goal: make `--attention-backend segpaged` apply SegPaged in the real serving
forward path (prefill + decode), not just the standalone driver.

- [ ] C1. Make `redknot_backend.forward_extend` build the SegPaged prefill
      view from `token_to_kv_pool` instead of dense `[H, L, D]`.
- [ ] C2. Per-(layer, head) page table backed by the real KV pool /
      RadixCache, with local heads bounded to sink + recent.
- [ ] C3. Decode reads the same per-head paged layout (already prototyped).
- [ ] C4. CUDA graph compatibility for the SegPaged decode path.
- [ ] C5. Correctness parity vs the standalone driver outputs.
- [ ] C6. Server smoke: launch with `--attention-backend segpaged` and run
      a RAG request end-to-end.

---

## Stage D — PD disaggregation runtime (target: ~1-2 weeks)

Goal: turn the modeled head-class KV transfer into a real transfer path.

- [ ] D1. Integrate `pd_transfer` payload build/restore into the disagg
      transfer backend (Mooncake / NIXL).
- [ ] D2. Send only each head's required KV (global full, local sink+recent).
- [ ] D3. Measure KV transfer bytes reduction (target 4.3x-6.3x).
- [ ] D4. End-to-end PD TTFT and throughput.

---

## Stage E — Head-aware scheduling & capacity (target: ~1-2 weeks)

Goal: realize the per-head capacity advantage in the live scheduler.

- [ ] E1. Bind `scheduler.HeadAwareCachePolicy` to the real cache admission.
- [ ] E2. Per-head KV footprint accounting in the scheduler.
- [ ] E3. Concurrency / capacity benchmark (target 4.7x-7.8x capacity).
- [ ] E4. Throughput and CoV (latency stability) measurement.

---

## Stage F — Full evaluation & baselines (target: ~1-2 weeks)

Goal: reproduce the paper's evaluation matrix.

- [ ] F1. Datasets: HotpotQA, MuSiQue, 2WikiMQA, TriviaQA, MultiFieldQA, Qasper.
- [ ] F2. Models: Qwen3-32B, Llama-3.3-70B, Mistral-7B.
- [ ] F3. Baselines: Dense, SGLang/PagedAttention, CacheBlend, ProphetKV, DuoAttention.
- [ ] F4. Metrics: F1/EM/RougeL, cosine/top-1/top-10, TTFT, FLOPs, KV bytes,
      throughput, concurrency, CoV.
- [ ] F5. Length scaling: 8K / 16K / 32K / 64K / 128K.
- [ ] F6. Ablations: no Sparse FFN, token-level vs head-aware, global ratio,
      window size, dense prefix layers, RoPE realignment, SegPaged vs dense mask,
      PD transfer on/off.

---

## Estimated Timeline

| Stage | Scope | Estimate |
|---|---|---|
| A | SegPaged prefill prototype | ~3 days |
| B | Sparse FFN + SegPaged fusion, sweet spot | ~1 week |
| C | SGLang serving backend | ~2-4 weeks |
| D | PD disaggregation runtime | ~1-2 weeks |
| E | Scheduling & capacity | ~1-2 weeks |
| F | Full evaluation & baselines | ~1-2 weeks |

Prototype-level (A+B): ~1.5-2 weeks.
Full paper parity (A-F): ~6-10 weeks depending on serving integration depth.
