# vLLM-RedKnot

**RedKnot non-prefix chunk reuse for vLLM V1.** This plugin directory contains
RedKnot algorithms, cache management, vLLM integration code, and evaluation tools.
It does not vendor the native SGLang execution engine or copy vLLM's models,
scheduler, GPU memory pools, or third-party kernel implementations. The
`RedKnot-vLLM` branch contains this standalone project at the repository root,
replacing the previous SGLang-based file tree on this branch. The SGLang version
remains on `main` and in Git history; this migration does not modify `main`.

The RedKnot-specific implementation originates from `/workspace/RedKnot`. The
original SGLang reproduction directory remains unchanged. The independent server
delivery directory is `/workspace/vllm-RedKnot`, while the Python package name is
`vllm_redknot`. On Linux, this differs from the legacy lowercase directory
`/workspace/vllm-redknot`, which this migration does not overwrite. vLLM is an
external dependency, not a vendored subdirectory.

## Code Organization and Migration Boundaries

This delivery prioritizes **code and experiment-asset migration**. It does not
require every model to be downloaded, end-to-end inference to pass, or a speedup
to be demonstrated. There are three distinct implementation states:

- **Integrated with vLLM:** `runtime.py`, `cache.py`, `runner.py`,
  `vllm_backend.py`, and the top-level `dsv4_*.py` modules are invoked by the
  explicitly enabled plugin.
- **Extracted, integration pending:** `core/` contains engine-independent
  algorithm assets extracted from RedKnot. Internal imports use the standalone
  package namespace. These modules are not enabled simply by being present, nor
  do they establish support for the corresponding GPU data paths, concurrent
  scheduling, or models. See the [extraction guide](docs/CORE_EXTRACTION.md) and
  [per-file provenance](docs/core_provenance.json).
- **Model code and experiment protocols migrated:** `model_backends/` and the
  four model benchmarks preserve RedKnot-specific policies, offline reuse
  algorithms, data preparation, and source configurations. Explicit parameters
  and callbacks define engine boundaries; SGLang and Transformers model executors
  are not included. Benchmark execution is rejected where required native vLLM
  integration is missing. A successful CPU preflight is not a successful model
  experiment.

```text
vllm-RedKnot/
├── vllm_redknot/
│   ├── core/                 # RedKnot algorithms; extraction != runtime integration
│   ├── model_backends/       # Model reuse/state/policy algorithms; native hooks pending
│   ├── plugin.py             # vLLM general-plugin registration
│   ├── config.py, compat.py  # Policy parsing, pinned interface fingerprints, guards
│   ├── runtime.py, cache.py  # Request plans, content identity, leases, budgets, commits
│   ├── runner.py            # MHA/GQA integration with the vLLM V1 runner
│   ├── vllm_backend.py      # MHA/GQA attention and physical KV-page consistency
│   ├── ops.py, mla.py       # RoPE/head operations and reference MLA decomposition
│   ├── dsv4_runner.py       # Flash model-geometry and execution-mode checks
│   ├── dsv4_runtime.py      # Flash chunk/z_off transactions and fallback
│   ├── dsv4_backend.py      # Native FlashMLA attention/projection integration
│   ├── dsv4_sparse.py       # Sparse MLA kernels selected by query head and row
│   ├── dsv4_projection.py   # Local z_off capture and aggregation through one wo_b
│   └── implementation_map.json  # Machine-readable implementation index
├── benchmarks/              # Flash + four model entries, paired evaluation, preflight
├── examples/                # Explicitly enabled model/head-classification policies
├── tests/                   # Cache, contract, numerical, migration, and entrypoint tests
└── docs/                    # Provenance, boundaries, usage, and validation records
```

SGLang's `ForwardBatch`, `ServerArgs`, `ScheduleBatch`, native memory pools,
radix cache, and model executors are not copied here. RedKnot behavior embedded
in those files requires integration through vLLM's own interfaces, not copying
entire native files or creating a `sglang` compatibility shim. See the
[migration boundaries](docs/MIGRATION_BOUNDARY.md) and
[integration design](docs/PORTING.md).

## Mistral, Llama, Qwen3, and Qwen3.5 Implementation Map

The four entrypoints retain their original filenames under `benchmarks/`.
They are not aliases that launch the SGLang scripts. Data preparation and
migration plans can be inspected without a GPU or model weights. Source model
configurations, prompts, chunking, metrics, and provenance are documented in the
[multi-model benchmark guide](docs/MULTIMODEL_BENCHMARKS.md).

| Model | Benchmark entrypoint | Migrated RedKnot-specific implementation | Native vLLM integration still required |
| --- | --- | --- | --- |
| Mistral | [benchmark_RedKnot_Mistral_RAG.py](benchmarks/benchmark_RedKnot_Mistral_RAG.py) | Native-SWA document-boundary replay planning, offline K relocation, prefix replacement, and suffix reuse | Mistral/SWA physical pages and query/decode integration; not an alias for full-causal MHA |
| Llama3.3 | [benchmark_RedKnot_Llama3.3_RAG.py](benchmarks/benchmark_RedKnot_Llama3.3_RAG.py) | Global/local head classification, sink/window policies, offline KV reuse, and Sparse FFN policy assets | Llama3 scaled RoPE, source quantization settings, per-head window/sink behavior, and Sparse FFN execution |
| Qwen3 | [benchmark_RedKnot_Qwen3_RAG.py](benchmarks/benchmark_RedKnot_Qwen3_RAG.py) | Source head classes including `local_full`, window/retention settings, Sparse FFN configuration, and RAG protocol | Complete source-policy integration with native attention/MLP; generic Dense support does not implement the source configuration |
| Qwen3.5-397B | [benchmark_RedKnot_Qwen35_397B_RAG.py](benchmarks/benchmark_RedKnot_Qwen35_397B_RAG.py) | Full/linear head policies, linear recurrence/window operations, offline KV/conv/recurrent-state contracts, and sparse MoE policies | vLLM hybrid-attention state pools, Q-gating, request-state restoration, and the MoE executor |

Backend algorithms are described in the
[MHA/SWA migration guide](docs/MHA_BACKEND_MIGRATION.md) and
[Qwen3.5/sparse MoE migration guide](docs/MULTIMODEL_BACKEND.md).

The source Mistral benchmark uses SWA=4096 and 20% boundary recomputation for
subsequent documents. Historical head/FFN profiles are retained separately and
must not be presented as the same experiment. Historical gains described in
source configurations are not measurements of this vLLM implementation.

The migrated Qwen3.5 state-restoration contract uses **one ordered document
bundle** as its reuse unit. Independent chunks' final linear recurrent states
cannot simply be concatenated. This does not implement arbitrary non-prefix
hybrid-state reuse or establish feature equivalence with Flash MLA.

Inspect the entrypoints without starting a model:

```bash
cd /workspace/vllm-RedKnot
python benchmarks/benchmark_RedKnot_Mistral_RAG.py --help
python benchmarks/benchmark_RedKnot_Llama3.3_RAG.py --help
python benchmarks/benchmark_RedKnot_Qwen3_RAG.py --help
python benchmarks/benchmark_RedKnot_Qwen35_397B_RAG.py --help
```

This migration step does not launch GPU experiments, require completed model
downloads, or alter the GPU keeper monitor.

## RedKnot Data Flow in vLLM

```text
SamplingParams.extra_args["redknot"]
  → plugin + pinned source/model/execution-mode checks
  → vLLM V1 runner / ForwardContext
  → RedKnot request plans, complete-chunk caching, and leases
  ├─ capture: preserve native outputs; collect all selected layers → atomic chunk commit
  ├─ reuse: content/model/policy match → reuse clean local rows; compute global/boundary/query rows
  └─ miss/unsupported: fall back to native computation before skipping work
  → remaining native vLLM layers and decode
```

This is **non-prefix content reuse**: the same chunk may appear in the middle of
a new request or at a different position. Matching uses actual tokens and
model/policy identity, not merely a shared prefix. An independently captured
chunk has not seen its new preceding context, so cross-context reuse is an
approximation that requires explicit consent and validation against real outputs.

### Flash MLA: Offline Local + Online Global/Dirty + Aggregation

Offline capture applies native inverse RoPE and `wo_a` projection to local
query-head attention, caching its low-rank contribution `z_off`. Online
execution computes all global rows and the local boundary/new-query rows, then
adds the corresponding `z_off` to clean rows. **Only one final `wo_b` is
executed.** Cached z is not rotated again.

Shared latent KV, SWA/C4/C128 state, the compressor, indexer, and FFN/MoE remain
native online vLLM computations. The current masked `wo_a` still performs a
full-width operation. Saved attention head-rows therefore cannot be equated
with whole-model compute savings or used to claim a 2–5× TTFT speedup.

## Core Implementation Markers

Key entrypoints contain `REDKNOT:` source markers. These are navigation aids;
they do not modify upstream code or bypass version checks. Paths in the JSON
index are relative to this plugin's source root.

| Marker ID | Implementation file and symbol | Responsibility |
| --- | --- | --- |
| `RK-PLUGIN` | [plugin.py](vllm_redknot/plugin.py) / `register` | Explicit vLLM registration; no runner/backend changes while disabled |
| `RK-REQUEST` | [runtime.py](vllm_redknot/runtime.py) / `RequestPlan` | Non-prefix token spans and capture/reuse/recomputed protocol |
| `RK-CACHE` | [cache.py](vllm_redknot/cache.py) / `CacheManager` | CPU payload byte budget, LRU, and protection against eviction while leased |
| `RK-TRANSACTION` | [runtime.py](vllm_redknot/runtime.py) / `RedKnotRuntime` | Content addressing, all-layer commits, request-wide leases, and exception cleanup |
| `RK-MHA-RUNNER` | [runner.py](vllm_redknot/runner.py) / `install_runner_hooks` | Request context entering native V1 execution |
| `RK-MHA-ATTENTION` | [vllm_backend.py](vllm_redknot/vllm_backend.py) / `RedKnotImpl` | Head-aware attention and local K/V scatter into native physical pages |
| `RK-FLASH-RUNNER` | [dsv4_runner.py](vllm_redknot/dsv4_runner.py) / `install_dsv4_runner` | Flash-specific native runner integration |
| `RK-FLASH-TRANSACTION` | [dsv4_runtime.py](vllm_redknot/dsv4_runtime.py) / `DSV4Runtime` | z_off cache transactions and safe fallback on missing/incompatible data |
| `RK-FLASH-ATTENTION` | [dsv4_backend.py](vllm_redknot/dsv4_backend.py) / `install_dsv4_attention` | Preserve native state updates; integrate sparse prefill and output projection |
| `RK-MLA-CAPTURE` | [dsv4_projection.py](vllm_redknot/dsv4_projection.py) / `capture_local_z` | Capture offline local-head z_off contributions |
| `RK-MLA-MERGE` | [dsv4_projection.py](vllm_redknot/dsv4_projection.py) / `merge_cached_z_and_project` | Aggregate clean cached rows, recompute dirty rows, and execute one wo_b |

```bash
cd /workspace/vllm-RedKnot
rg -n 'REDKNOT: RK-' vllm_redknot
/workspace/vllm/.venv/bin/python -B -m vllm_redknot code-map
```

`code-map` reports responsibilities, paths, and integration status without
loading vLLM, Torch, or models. Its `migrated_model_backends` section lists the
four-model assets separately with `runtime_integrated: false`; those paths are
not automatically registered by the plugin.

## Current Capabilities and Limitations

| Scope | Code status | Validation status |
| --- | --- | --- |
| Qwen2/Qwen3/Llama static-RoPE MHA/GQA | Native backend/runner integration exists | Full GPU model validation pending |
| Flash-0731 local/global MLA | Dedicated backend, head-selection kernels, and z_off aggregation exist | CPU and GPU micro-kernel checks passed; real-model validation pending |
| Cache management | CPU budgets, LRU, leases, and atomic commits are integrated | Does not establish concurrent GPU requests or complete GPU-memory management |
| Generic paired benchmark | Warm-state TTFT, reference F1, raw outputs, and measured reuse counts | No real Flash TTFT/F1/QPS results |
| Original Flash release suites | vLLM-only entrypoint and migration preflight | Does not reproduce the four SGLang long-context suites by itself |
| Advanced core policies/layouts | Independently extracted assets with traceable provenance | Unconnected assets must not be advertised as supported runtime features |
| Source Mistral/Llama3.3/Qwen3/Qwen3.5 protocols | Named entrypoints, data/configuration assets, and backend algorithms migrated; some native hooks pending | CPU/static checks only for this migration; no model experiments |
| Pro | Profile/scale-policy assets retained in core; dedicated runtime incomplete | No Pro serving or performance qualification |

The native adapters remain limited to **TP/PP/DP=1, one request, V1 eager
execution, and full prefill**, with APC, chunked prefill, and KV Connector
disabled. Multi-request execution, TP8, Sparse FFN/MoE, and SegPaged vLLM data
paths remain incomplete. Adding `core/` assets does not remove these limits.

## Usage and Validation

1. See [USAGE](docs/USAGE.md) for installation and the request protocol. Nothing
   automatically installs/upgrades vLLM, Torch, or CUDA, downloads models, or
   changes a shared inference environment. Run named benchmarks, configuration
   assets, and documentation examples from this source tree; the wheel installs
   only the plugin package.
2. The Flash entrypoint is `benchmarks/benchmark_RedKnot_DeepSeekV4Flash.py`;
   its one-command script is
   `benchmarks/run_deepseek_v4_flash_reproduction.sh`. Start with `--help`
   and CPU-only preflight. Frozen-suite token export, cache capacity, and
   execution requirements are in the
   [benchmark migration guide](docs/BENCHMARK_MIGRATION.md).
3. Real inference requires a complete, independently verified checkpoint.
   Downloading `.part` files are not loadable weights. CPU contract tests and
   small-tensor GPU tests do not qualify the complete model.
4. Warm-state TTFT excludes startup, offline capture, and compilation warmup.
   Quality is evaluated against reference answers. Serial measurements do not
   establish QPS or parallel speedup. Actual execution evidence is recorded in
   [VALIDATION](docs/VALIDATION.md).

## Provenance and Versions

- RedKnot-specific algorithms originate from
  `rednote-machine-learning/RedKnot`, pinned to revision
  `55ee4e8401603f8d2612877e4053e18b37b1c1bd`.
- The vLLM interface is pinned to revision
  `e52be1a62d3879b1202f4f355d3c3472b560c6f2`.
- Python >=3.12; Apache-2.0. Original copyright,
  [LICENSE](LICENSE), and [NOTICE](NOTICE) are retained.
- The source `/workspace/RedKnot` is not modified. Original SGLang H200/B300
  results are not relabeled as results of this vLLM version. This is an
  independent research implementation, not an official upstream release.
