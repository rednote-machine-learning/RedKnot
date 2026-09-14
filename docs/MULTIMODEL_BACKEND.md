# Multi-model RedKnot implementation boundary

The RedKnot-specific backend assets are now separated from their source engine
under `vllm_redknot/model_backends/`, using the shared algorithms in `core/`.
They do not vendor native SGLang model implementations, a SGLang compatibility
package, or another engine's runner. The original `/workspace/RedKnot` source
remains unchanged.

**Extraction is not runtime integration.** The Qwen3.5 helpers below have
`runtime_integrated: false`. They are callable implementation code, not empty
stubs, but the native vLLM model, cache lifecycle, and expert-execution hooks
must still invoke them with validated metadata. Existing Flash runtime behavior
is not changed by importing this package. No new model/benchmark run is claimed.

## Mistral, Llama, and Qwen3

The common GQA/MHA RedKnot implementation remains in `core/head_config.py`,
`core/mask_plan.py`, `core/offline_cache.py`, `core/per_head_storage.py`,
`core/rope_helper.py`, `core/ops_flash.py`, and `core/sparse_ffn.py`.
`model_backends/mha_reuse.py` supplies the separate engine-independent
multi-segment reuse boundary. It uses injected RoPE/attention operations rather
than importing native SGLang model classes. See
[mha_migration_provenance.json](mha_migration_provenance.json) for its lineage.
Architecture-specific Mistral sliding-window behavior and actual vLLM model
activation must not be inferred from a generic MHA entry point alone.

## Qwen3.5 and sparse MoE

| Target file | Source RedKnot implementation | Preserved responsibility |
|---|---|---|
| `qwen35_policy.py` | Selected functions from `attention/redknot/driver_qwen35.py` | Full/linear layer identification and random/deep/shallow full-attention head-class allocation |
| `qwen35_full_attention.py` | `_qwen35_full_attn_headclass` | Q/K normalization, Q-gating, local/global full-attention policy; native RoPE and head-class attention supplied as explicit callables |
| `qwen35_linear.py` | `_linear_local_recurrence`, `_linear_local_token_window` | Original GDN recurrence/prefix relay and research per-head token-window operations |
| `qwen35_reuse_contract.py` | Control path from `qwen35_offline_reuse.py` | CPU preflight, one ordered bundle, once-only restore, mode guards and logical-length accounting |
| `qwen35_recurrent.py` | Capture, restore and position-offset path from `qwen35_offline_reuse.py` | Actual conv/GDN snapshots, shape-validated copies and packed RoPE offsets |
| `sparse_moe_policy.py` | Policy/context definitions from `models/redknot_sparse_moe.py` | Explicit configuration and caller-owned request-local token policy context |
| `sparse_moe.py` | Selector/mask functions from `models/redknot_sparse_moe.py` | Per-request mean-ratio selection, recent/protected/minimum keep rules, dense fallback and aligned masks |

All paths in the first column are relative to `vllm_redknot/model_backends/`.
[qwen35_migration_provenance.json](qwen35_migration_provenance.json) records
source revision `55ee4e8401603f8d2612877e4053e18b37b1c1bd`, source paths,
symbol ranges, source/target SHA256 digests, modifications and integration flags.
Markers use `REDKNOT-MODEL`. Modern type annotations and formatting do not
change selector or recurrence arithmetic.

### Explicit native-engine inputs

The original SGLang global server flags, DP-attention accessor and request-pool
lookups are not copied. The native adapter must provide:

- `Qwen35ReuseConfig`, with explicit graph, prefix-cache, multimodal, parallel,
  speculative-decode and model-length settings;
- resolved live recurrent slot IDs and `Qwen35RecurrentBuffers` with
  `[layer, slot, ...]` conv/GDN tensors;
- a cache lookup callback and per-slot loaded-bundle receipts;
- immutable ordered-bundle IDs and lengths, current/prefix lengths, and
  prefill/decode mode; and
- the native RoPE/head-class callbacks and correctly shaped full-attention
  projection view, plus request context and layout generation for sparse MoE.

This interface is not a claim that vLLM uses the same physical tensor layout as
SGLang. The adapter must prove the layout mapping. The helper never creates a
fake SGLang pool or silently imports a native model implementation to fill gaps.

### Hybrid state correctness

A Qwen3.5 offline bundle contains both full-attention KV and causal conv/GDN
state. Arbitrary independent final GDN states cannot be concatenated. The strict
ported path accepts one immutable, ordered document bundle as a **logical
prefix**, restores its recurrent state once, and adds bundle length to online
RoPE positions. It is not an arbitrary non-prefix linear-state splice.

The adapter must clear the loaded-slot receipt whenever a native slot is reused,
even when a new request uses the same bundle ID. Fresh restore cannot be combined
with a live prefix-cache hit; decode without a matching loaded receipt is
rejected. Graph execution, prefix caching, multimodal, PP > 1, DP-attention and
speculative decoding remain unsupported in this strict helper.

Preflight checks the whole batch's state shapes and context limits before the
first recurrent write. If a device copy itself fails, receipts are invalidated
and the error is raised. This does not promise rollback of asynchronous CUDA
writes; the native owner must abort/reinitialize the affected request.

The exact `_linear_local_recurrence` preserves supplied prefix state. The
separate `_linear_local_token_window` is retained as source research code and
is not certified to equal full-history recurrence. Historical source comments
and speed claims are not new vLLM measurements.

### Sparse MoE integration

`RedKnotSparseMoEPolicy.from_config(...)` accepts an explicit mapping or
attribute object; a missing enable flag leaves the policy disabled.
`resolve_routed_keep_mask(...)` receives `RedKnotTokenPolicyContext`,
`is_prefill`, and `layout_version` explicitly. A stale layout generation,
ineligible layer/mode, missing mask or wrong tensor shape returns dense fallback.
The selector preserves per-request statistics and protection floors rather than
mixing unrelated requests into a global importance threshold.

The native expert executor is deliberately not copied. Compacting kept rows,
running the router/experts, preserving shared-expert behavior, scatter-add,
collectives and output-layout ownership still require a native vLLM hook.
An extracted selector does not mean token-sparse MoE is enabled.

## CPU checks

```bash
python -B -m unittest discover -s tests -p test_qwen35_model_backends.py -v
```

Local controller verification: **11 tests, 7 passed, 4 skipped** because Torch
is not installed. The four optional tests use CPU tensors only: conv/GDN
snapshot/restore and slot clearing, all-request shape preflight without partial
writes, exact recurrence prefix relay, and per-request sparse selection/layout
generation. No model weights, GPU experiment, precision, TTFT or QPS test is
required or claimed in this migration step.

The source import/SHA checks cover the migrated Qwen3.5/MoE modules. Package
initialization and the three pure policy/control modules also pass a subprocess
test with Torch, Triton, SGLang, vLLM and Transformers imports explicitly blocked.
