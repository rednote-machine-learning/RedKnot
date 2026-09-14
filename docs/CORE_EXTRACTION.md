# RedKnot portable core extraction

`vllm_redknot/core/` contains RedKnot-specific source extracted from the
SGLang-hosted RedKnot implementation. It is **not** a copy of SGLang, a SGLang
compatibility shim, or a claim that every extracted feature is active in vLLM.

The source repository remains intact. The native vLLM implementations in
`dsv4_backend.py`, `dsv4_runtime.py`, `dsv4_runner.py`, `vllm_backend.py`,
`runtime.py`, and `runner.py` keep their existing execution semantics. The
extraction adds no model hooks, registrations, cache-layout overrides, or
automatic sparse-FFN/TP8 activation.

## Source identity and code markers

The source is RedKnot revision
`55ee4e8401603f8d2612877e4053e18b37b1c1bd`, specifically
`python/sglang/srt/layers/attention/redknot/`. This extraction contains
**43 implementation modules plus 4 lightweight package initializers**.
The 18 omitted implementation files are enumerated individually in
[core_provenance.json](core_provenance.json).

Each implementation begins with `REDKNOT-CORE`. Its original copyright and
algorithm implementation are retained. Changes are limited to:

- the provenance marker;
- replacing the RedKnot-specific `sglang.srt.layers.attention.redknot`
  namespace with `vllm_redknot.core`, including import examples;
- qualifying the two direct-file sibling import fallbacks under the new
  package namespace; and
- replacing the original eager package initializers with documentation-only
  initializers. Import submodules explicitly rather than relying on the old
  package-level reexports.

The JSON records source and destination paths, both SHA256 digests, modification
types, and the runtime integration status for every extracted file. Every
extracted file currently has `runtime_integrated: false`: this flag describes
that exact target module, not the separate existing vLLM RedKnot adapter.
Original documentation that describes a SGLang serving path or historical
measurements remains source documentation, not a vLLM validation result.

## Implementation map

| RedKnot responsibility | Portable entry points | vLLM integration boundary |
|---|---|---|
| Logical-head classification and planning | `head_config.py`, `mask_plan.py`, `deepseek_v4_mla.py`, `scheduler.py` | Policy assets; existing native vLLM `config.py` remains active |
| Whole-segment and per-head cache storage | `offline_cache.py`, `per_head_storage.py`, `segpaged_v2/{storage,page_table,visible_plan}.py` | No replacement of vLLM's allocator or page table |
| Positionless shared-latent artifacts | `dsv4_shared_latent_cache.py`, `dsv4_shared_latent_gpu.py` | Capture, RoPE relocation, cache-target and compressor hooks still need native vLLM wiring |
| Atomic snapshot and all-rank commit contracts | `dsv4_shared_snapshot_runtime.py`, `dsv4_composite_commit.py` | Runtime must supply its own participants, collectives, and cache ownership |
| Context identity and batch plans | `dsv4_context_identity.py`, `dsv4_reuse_batch.py`, `v4/` | Contracts do not authorize reuse on a cache hit without compatibility and runtime checks |
| Offline MLA output and online aggregation | `dsv4_mla_offload.py`, `dsv4_fused_z_merge.py` | Extracted implementations are separate from the active native `dsv4_runtime.py` and `dsv4_projection.py` path |
| Sparse query geometry and projection certificates | `dsv4_sparse_q.py`, `dsv4_sparse_q_runtime.py` | No TP8 sparse-projection activation or kernel performance claim |
| FFN policy, drift and head profiling | `sparse_ffn.py`, `head_profiler.py`, `mla_head*_profiler.py`, `mla_head_drift_*.py` | No automatic MoE/top-k changes to native model execution |
| Native segment-page policy and index compaction | `native_segment_pages.py`, `native_segment_page_kernels.py` | RedKnot-specific policy/kernel assets; no replacement of native attention kernels |
| Pro geometry and component sizing | `pro0813/profile.py`, `pro0813/scale_policy.py` | A geometry/sizing contract is not a working Pro model adapter |
| Evaluation aggregation and timings | `eval_harness.py`, `dsv4_timing.py` | Existing benchmark execution remains separate; projected throughput is not measured QPS |

Some policies encode the exact Flash-0731 geometry or historical Pro-0813
profile. Merely importing them does not make their dimensions correct for an
arbitrary model, checkpoint revision, tensor-parallel layout, or GPU.

## Deliberately not copied

No native SGLang model implementation, model runner, scheduler, token/KV pool,
distributed implementation, JIT kernel tree, or `sgl_kernel` is vendored.
Dependencies on those modules are a hard extraction boundary.

In particular, `dsv4_rope_reloc.py`, `dsv4_offline_reuse_v2.py`,
`dsv4_shared_latent_sglang.py`, the SGLang batch/snapshot glue, and
`dsv4_reuse_backend_runtime.py` remain only in the source. Their cache layout,
RoPE, compressor, and attention integration must be implemented against native
vLLM interfaces rather than satisfying the imports with a fake SGLang module.

The FA3/SegPaged attention callers using `sgl_kernel`, the Transformers model
drivers and offline prefill loader, and the Qwen3.5 SGLang glue are also not
copied. `ops_flash.py` is included as a RedKnot-specific standalone algorithm
asset with optional FlashAttention use; it is not the active native vLLM
attention backend. The exact exclusion reason for each file is in the JSON.

## CPU validation and import behavior

Importing `vllm_redknot.core`, `core.v4`, `core.pro0813`, or `core.segpaged_v2`
does not import Torch, Triton, SGLang, vLLM, or initialize CUDA. Pure control
contracts can also be imported without those tensor runtimes. Explicit tensor
modules require Torch; explicit Triton kernels require Triton. The active vLLM
plugin does not eagerly import the extracted core.

Run the extraction tests from the project root:

```bash
python -B -m unittest discover -s tests -p test_portable_redknot_core.py -v
```

Coverage includes:

- exact destination hashes and complete provenance coverage;
- source hashes and exact mechanical transformations when the source snapshot
  is available locally;
- all Python syntax/import dependencies, including a check against dynamic
  SGLang import strings and missing extracted sibling modules;
- controller-only imports with Torch/Triton/SGLang/vLLM explicitly blocked;
- optional full module imports on a Torch/Triton host, asserting no CUDA
  context is initialized;
- incomplete-publication rejection, immutable artifacts and rollback;
- prefix-sensitive identity, head-row conservation, boundary replay,
  sparse-position compressor coverage, projection row bounds, and uncapped
  online suffixes; and
- exact persistent-bank bytes and component-specific Flash/Pro `z_off` sizing.

Local Python 3.12 verification: **14 tests, 13 passed and 1 skipped** because
Torch/Triton are absent on the controller host. These tests certify selected
portable contracts and extraction integrity, not all tensor operations,
GPU kernels, TP8 integration, model accuracy, TTFT, or throughput.

The extracted source keeps its original formatting and type-annotation style.
Do not run bulk autofix on this directory: that would obscure source lineage.
Any scoped lint exceptions apply to this source-preservation boundary, not to
the new native adapters or tests. The guarded cross-loop `prev_row_pos` in
`v4/segmented_compressor.py` is covered by a sparse-position CPU test; its
existing static F821 warning is not permission to ignore undefined names
throughout the project.
