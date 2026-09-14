# Multi-model RedKnot benchmark migration

The four requested entrypoints now share real CPU-safe data preparation,
source-policy inspection, and metric code. **Code migration is not runtime
qualification.** No model experiment was run for this migration, and none of
the original SGLang benchmark scores is presented as a vLLM result.

| Entrypoint in `benchmarks/` | Migrated RedKnot logic | Native vLLM execution state |
| --- | --- | --- |
| `benchmark_RedKnot_Mistral_RAG.py` | Standard RAG, middle context cap, even document split, optional instruction wrapping; native SWA 4096 and 20% document-boundary policy recorded | Refused: native-SWA offline reuse/offset repair runner not connected |
| `benchmark_RedKnot_Llama3.3_RAG.py` | Source-order LongBench selection, leading cap, fixed chunks and short-tail rule; g10 head matrix, fixed window 4096, optional ratio fallback 0.5, three-tier FFN profile | Native Llama-3.3 refused: scaled Llama3 RoPE is unsupported; source INT4 is also unsupported |
| `benchmark_RedKnot_Qwen3_RAG.py` | Longest-context-first selection, leading cap, fixed document chunks; actual local_full/global/retrieval matrix and SparseFFN profile | Source policy refused; an explicitly different, reviewed static-RoPE/unquantized MHA variant may use the existing vLLM runner |
| `benchmark_RedKnot_Qwen35_397B_RAG.py` | Seeded selection, round-robin distractors, fixed target; 397B full/linear/MoE settings, short-answer prompt | Refused: native hybrid full-attention/GatedDeltaNet state runner not connected |

The tiny four entrypoints are dispatchers, not empty placeholders: the shared
implementation is `benchmarks/multimodel_rag.py`. Its functions are:

- `prepare_rows`: four source-specific selection, document assembly, token
  preparation and audit paths; accepts an injected CPU tokenizer for tests.
- `source_policy_plan`: actual head-class counts, source windows/ratios and
  separate FFN/linear profiles. It does not pretend they are vLLM configs.
- `checkpoint_preflight`: invokes the existing worker's CPU compatibility guard
  before GPU import, including static RoPE and unquantized-model constraints.
- `data_preflight`: preserves tokens and checks the one-to-eight chunk contract,
  model-length/output reserve and all-unique-capture payload budget.
- `short_answer`, `score_answer`, `migrated_metrics`: symmetric answer extraction,
  English reference F1/EM and explicit percentage-point F1 decrease.
- `run_gpu`: opt-in dispatch to existing `benchmark_redknot.run_benchmark`,
  retaining warmup/capture separation, alternating pairs and raw outputs.

Code markers `RK-RAG-DATA`, `RK-RAG-GUARD` and `RK-RAG-METRICS` identify those
RedKnot-specific responsibilities. There is no native SGLang loader, allocator,
scheduler or kernel in these entrypoints.

## Safe default: inspect only

From the repository directory, no arguments or `--dry-run` prints a CPU
migration plan. `--help` lists options. Neither action imports torch, vLLM or
SGLang, starts a model, downloads data, changes GPU keepers or edits source
repositories.

```bash
python3 benchmarks/benchmark_RedKnot_Mistral_RAG.py
python3 benchmarks/benchmark_RedKnot_Llama3.3_RAG.py --help
python3 benchmarks/benchmark_RedKnot_Qwen3_RAG.py --dry-run
python3 benchmarks/benchmark_RedKnot_Qwen35_397B_RAG.py --dry-run
```

CPU plan exit 0 means the plan was printed, not that runtime blockers passed.
The plan records `gpu_initialized=false`, `runtime_assessed=false` and
`source_reproduction_equivalent=false`.

## Prepare an existing local corpus without model execution

Only the optional CPU `tokenizers` library is required for raw-text preparation;
it is not installed automatically. `tokenizer.json` is loaded locally, with
padding, truncation and special-token insertion disabled. No remote tokenizer
code or automatic chat template is executed.

```bash
python3 benchmarks/benchmark_RedKnot_Qwen3_RAG.py \
  --prepare-only \
  --data-dir /workspace/RedKnot/test/srt/redknot/datasets/LongBench/data \
  --dataset hotpotqa --samples 3 \
  --tokenizer /workspace/Models/Qwen3-32B/tokenizer.json \
  --output /workspace/new-qwen3-cases.json
```

`--dataset` may be repeated; otherwise the model-specific source defaults are
used. `--rag-file /path/requests.jsonl` alternatively accepts explicit
`question`, `documents` and `answers`, keeping document order for non-prefix
reuse. JSON arrays are supported too. Input and tokenizer hashes, source row
IDs, query templates, document fragments, exact token counts, cap/tail removal,
distractor row IDs and full token hashes are written in the new result.
Existing outputs are never overwritten. `--cases` can consume previously
prepared exact-token cases without importing a tokenizer.

The migration preserves source preprocessing rather than silently pretending
all source contexts were complete: Qwen3 uses a leading token cap; Llama uses a
leading cap and drops tails shorter than 64 tokens; Mistral's standard path
retains the beginning/end when over its cap; Qwen3.5 fills its exact target with
other rows without inserting gold answers. Omitted tokens are counted. The
original source's decode/re-encode document construction is retained, then BOTH
methods receive the same concatenation of encoded document fragments and query.
The preparation audit reports whether this equals a single full-text encoding;
it is not silently assumed.

The original Mistral entry defaulted to a separate `longbench_rag.jsonl` override;
use `--rag-file` to migrate that path explicitly. Its standard LongBench mode is
also implemented. `[INST] ... [/INST]` wrapping is available only through the
explicit `--mistral-instruction-wrap` flag; no tokenizer/chat-format guess is
made. Original oracle-answer-injection and evidence-selection experiments are
not enabled by this entrypoint. Llama's optional padded length-sweep mode is not
implemented here; its default source-order corpus path is.

Source-default chunking can produce more than eight documents (for example,
Llama's 40K cap with 4K chunks). CPU preparation can describe this faithfully;
`data_plan.blockers` then reports that the current vLLM request cannot run it.
It never drops extra documents to make the request appear supported.

## Profiles remain source profiles

`benchmarks/data/multimodel_profiles.json` preserves the source head matrices
and FFN/linear policy JSON, with original paths and SHA256 checksums. The source
is RedKnot commit `55ee4e8401603f8d2612877e4053e18b37b1c1bd`; all four benchmark
files and relevant profile files appear in the provenance inventory. Their
RedKnot code/profile license is Apache-2.0; upstream dataset terms still apply.
No raw LongBench corpus or model weights were copied here.

Important distinctions retained in the code:

- Mistral's actual benchmark uses SWA 4096 and recompute ratio 0.20, **not** the
  companion `mistral-7B_optimal_g15_lf_ret.json` head matrix. The asset is retained
  as a companion profile and explicitly marked unused by that source benchmark.
- Llama's actual selected matrix has 576 local and 64 global KV-head positions
  across 80 layers. The source main function applies the FFN profile's fixed
  window 4096, overriding the earlier ratio setting. Only explicitly disabling
  `REDKNOT_WINDOW_FIXED` permits the `context_tokens * 0.5` ratio fallback.
  Both precedence and fallback are recorded, alongside FFN thresholds 0.2/0.05.
  These windows and FFN thresholds are not active merely because the asset
  was copied.
- Qwen3's actual matrix has **435 local_full, 48 global and 29 retrieval**
  positions. Some source prose/summary fields describe older ratios; the plan
  counts the matrix itself. `local_full` is not silently renamed `local`, nor
  is retrieval sparsity silently discarded or declared implemented.
- Qwen3.5 uses the **397B** asset explicitly, not the source script's implicit
  35B model default: 60 layers, 15 full-attention layers, first 9 full layers
  dense, deep 6 selected with source global fraction 0.4/window 2048. Its linear
  profile retains safety 2.0, minimum window 256, dense prefix 5, and MoE start
  24/mass threshold 0.7. Runtime calibration and application remain separate.

Historical accuracy, speedup and analytical FLOPs prose in copied assets are
source metadata only; the plan marks them unvalidated on vLLM. This migration
does not derive measured compute savings from those old component ratios.

## Distinct compatible variant: explicit, not source-equivalent

Plain `--run` fails for all four source benchmark profiles. Mistral and Qwen3.5
remain blocked even with an override because their native adapters are not
connected. Qwen3/Llama may only enter the generic vLLM path with **both** `--run`
and `--run-compatible-variant`, a reviewed explicit vLLM `--config`, local
`--model`, trusted local `--model-manifest`, prepared cases, a new output path
and `--gpu-confirmed-idle`. That flag acknowledges a different protocol/policy,
not implementation of missing source window/sink/FFN behavior.

The actual checkpoint is then checked against the existing worker guards.
Native Llama-3.3 scaling still fails; the script never removes `rope_scaling`
from the checkpoint. NF4/INT4 or other quantized MHA checkpoints also fail.
Models must match the named layer/head geometry. One visible GPU, TP/PP=1,
eager V1, one serial request, no APC and no chunked prefill are required.

A trusted model manifest is a local JSON object containing `model_revision`
matching the policy and `files: [{path, size, sha256}, ...]`. It must cover
`config.json`, the safetensors index and every referenced shard (or the single
`model.safetensors`). Files are independently size/SHA-checked before loading;
missing/partial shards are not loadable. The manifest's trust is supplied by
the operator, not established by this migration. No model/benchmark was run to
create such a manifest in this task.

## Metrics and validation

The compatible vLLM path reuses the existing serial paired benchmark unchanged:
at least three untimed warmup pairs, ten measured pairs, alternating execution,
identical greedy fixed output budgets (default 128, minimum 50), actual worker
reuse evidence and direct `first_token_latency`. Raw dense/reuse outputs remain
in the JSON. Additional source-style extraction is applied symmetrically and
stored separately from the raw generic F1/TTFT metrics. Missing references stay
unscored. English token-overlap F1/EM is not a Chinese semantic-quality metric.

These output budgets differ from the old source defaults (Mistral 16, Llama 48,
Qwen3 32 and Qwen3.5 24 tokens). Consequently the report explicitly sets
`source_reproduction_equivalent=false`. No throughput/QPS or FLOPs savings are
claimed; source theoretical compute proxies are not relabeled wall-clock gains.

Twenty CPU tests cover all four default entries without torch/vLLM/SGLang,
source matrices and 397B-specific ratios, selection/capping/distractor logic,
explicit documents, non-mutation, unsupported-runtime refusal, scaled-RoPE and
quantization refusal, chunk/model length guards and symmetric metrics. These
tests validate migrated code contracts, not model speed or answer quality.
