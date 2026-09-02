# RedKnot on Ascend NPU — Adaptation Notes

> **Status: Work in progress.** This document describes RedKnot's ongoing
> adaptation to Huawei Ascend NPU platforms. It complements — and does not
> replace — the upstream SGLang Ascend documentation, which now provides the
> baseline NPU runtime, container images and CI. See
> [SGLang Ascend NPU Quickstart](https://github.com/sgl-project/sglang/blob/main/docs/platforms/ascend/ascend_npu_quick_start.md)
> and the surrounding
> [`docs/platforms/ascend/`](https://github.com/sgl-project/sglang/tree/main/docs/platforms/ascend)
> tree for the underlying platform contract.

RedKnot is a long-context inference framework layered on top of SGLang. Its
core mechanisms — head-classified KV reuse, SegPagedAttention, sparse FFN /
adaptive expert Top-K, and offline KV + RoPE relocation — are implemented in a
model-aware but platform-neutral way. Bringing them up on Ascend therefore
splits cleanly into (1) reusing the upstream SGLang NPU runtime and (2)
porting the RedKnot-specific kernels and policy contracts.

Huawei Cloud is actively driving the Ascend port and will co-publish
qualification profiles for the frozen release suites once they clear our
compute-ledger contract. Until then, treat the numbers on Ascend as
preliminary and always re-run against the frozen manifests before quoting
them.

---

## 1. Upstream platform baseline (landed)

The following upstream SGLang assets are already available and cover the
platform layer that RedKnot builds on:

| Area | Upstream asset |
|---|---|
| Supported devices | Atlas 800I A2 / A3 inference series |
| Container image | `quay.io/ascend/sglang:main-cann8.5.0-a3` (A3) · `main-cann8.5.0-910b` (A2) |
| Dockerfile | [`docker/npu.Dockerfile`](https://github.com/sgl-project/sglang/blob/main/docker/npu.Dockerfile) |
| Quickstart | [`docs/platforms/ascend/ascend_npu_quick_start.md`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/ascend/ascend_npu_quick_start.md) |
| Best practice | [`ascend_npu_best_practice.md`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/ascend/ascend_npu_best_practice.md) |
| Environment variables | [`ascend_npu_environment_variables.md`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/ascend/ascend_npu_environment_variables.md) |
| Supported models & features | [`ascend_npu_support_models.md`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/ascend/ascend_npu_support_models.md), [`ascend_npu_support_features.md`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/ascend/ascend_npu_support_features.md) |
| CI | `.github/workflows/{pr-test-npu,full-test-npu,nightly-test-npu,release-docker-npu*}.yml` |

If you only need SGLang on Ascend, follow the upstream quickstart directly.
The remainder of this document is about the additional work required to run
RedKnot on top of that baseline.

---

## 2. RedKnot Ascend adaptation status

RedKnot's runtime is split into three layers; the porting effort tracks each
one independently.

| RedKnot layer | CUDA path | Ascend port status |
|---|---|---|
| Head classification & policy loader (`redknot/head_config.py`, `head_profiler.py`) | Pure Python / JSON | ✅ Platform-neutral; expected to work as-is |
| Offline KV cache & RoPE relocation (`redknot/offline_cache.py`, `rope_helper.py`) | Pure Python + torch ops | ✅ Platform-neutral tensor path; validated on CPU / CUDA, Ascend validation in progress |
| Head-aware attention recovery — FA2 / FA3 buckets (`redknot/ops_flash.py`, `ops_flash3.py`) | FlashAttention-2 / FA-3 | 🚧 Ascend requires a native kernel; wiring to CANN attention operators in progress |
| SegPagedAttention runtime (`layers/attention/redknot/*`) | Custom CUDA + FlashInfer | 🚧 Ascend port depends on the FA replacement above and on the upstream NPU paged-attention operator |
| Sparse FFN / adaptive expert Top-K (`redknot/sparse_ffn.py`) | Torch + Triton | 🚧 Torch fallback is expected to run; Triton fast path has no Ascend equivalent yet |
| DeepSeek-V4 MLA integration (`layers/attention/redknot/dsv4_*`) | FlashMLA `1.0.0+9241ae3` + custom kernels | 🚧 Depends on an Ascend MLA operator; blocked until the CANN operator publish |

Legend: ✅ works on Ascend today, 🚧 in progress, ❌ not planned.

### 2.1 Known gaps

- **FlashAttention-2 / FA-3 buckets.** The reference implementation calls into
  FA-2 / FA-3 kernels that are CUDA-only. On Ascend they need to be replaced
  with a CANN paged-attention operator with equivalent per-head masking and
  bucketed batch layout. This is the largest single item on the port.
- **FlashMLA.** The DeepSeek-V4 path pins FlashMLA `1.0.0+9241ae3`, which has
  no Ascend build. An Ascend MLA operator with the same compressor / indexer /
  sparse-Q contract is required before the DeepSeek-V4-Flash release path can
  reproduce on NPU.
- **Triton sparse-FFN kernel.** Ascend does not run Triton. A Torch fallback
  is acceptable for functional runs but will not hit the published TTFT
  numbers.
- **Numerical alignment.** The compute-ledger contract in
  [`test/srt/redknot/`](../test/srt/redknot) treats the online Recomputed path
  as the numerical reference. On Ascend, the Recomputed path itself must
  first match the CUDA reference within the per-suite tolerance before any
  RedKnot-on-Ascend delta is meaningful.

---

## 3. Recommended workflow (until qualification lands)

1. **Bring up SGLang on Ascend first.** Follow the upstream quickstart, pick
   the container that matches your A2 / A3 device, and validate a dense
   forward pass on a small model.
2. **Enable RedKnot in dry-run / Recomputed mode.** Point RedKnot at the same
   model, but disable the reuse path (run the online Recomputed reference
   only). This exercises the RedKnot scheduling and result-contract plumbing
   without depending on the CUDA-only kernels.
3. **Reproduce a frozen suite in Recomputed mode.** Use the release manifests
   under
   [`test/srt/redknot/datasets/LongBench/suites/`](../test/srt/redknot/datasets/LongBench/suites)
   and confirm that the Recomputed answers and per-rank logs match the CUDA
   reference within the suite's tolerance.
4. **Enable RedKnot reuse only after the Ascend attention operator is in
   place.** Turning on the reuse path before the FA replacement lands will
   produce numerically invalid results and should not be quoted.

---

## 4. How to contribute

- File issues that reproduce on Ascend under the `platform:ascend` label so
  they can be triaged jointly with the Huawei Cloud team.
- If you are porting a specific RedKnot kernel to CANN, please align the
  operator signature with the CUDA reference in
  `python/sglang/srt/layers/attention/redknot/` and open a draft PR early;
  the head-policy contract and the Recomputed reference make it possible to
  review correctness before performance is tuned.
- Do not modify the frozen release manifests
  (`test/srt/redknot/datasets/LongBench/suites/*.sha256`,
  `head_class/*.json`, `qualification_profiles/*`) as part of an Ascend
  bring-up patch. Ascend-side qualification profiles will be published as
  additional files, not as edits to the CUDA-side release.

---

## 5. Roadmap

- **Short term.** Reach functional parity with the upstream SGLang Ascend
  baseline in RedKnot's Recomputed reference path.
- **Medium term.** Publish Ascend-side qualification profiles for the frozen
  64K / 128K / 256K / 440K LongBench-derived suites once the attention
  operator gap is closed.
- **Long term.** Co-publish an Ascend companion to the DeepSeek-V4-Flash
  release, including a hardware-specific compute-ledger and TTFT contract.

Updates will be posted here and reflected in the top-level README's *News*
section.
