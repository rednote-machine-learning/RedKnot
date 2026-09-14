# Flash benchmark migration: vLLM only

This is an executable, fail-closed **benchmark entrypoint migration**, not a claim
that the original SGLang performance results have been reproduced on vLLM.
No SGLang engine, scheduler, kernels, installation scripts or environment presets
are copied into these benchmark entrypoints. `/workspace/RedKnot` is unchanged.

## Implemented code map

| RedKnot-specific responsibility | vLLM implementation |
| --- | --- |
| Flash entrypoint and explicit GPU opt-in | `benchmarks/benchmark_RedKnot_DeepSeekV4Flash.py` |
| One-command wrapper, safe CPU default | `benchmarks/run_deepseek_v4_flash_reproduction.sh` |
| Frozen-case token/hash conversion and cache preflight | `benchmarks/flash_reproduction.py` |
| Capture, paired hot timing, raw output and reference F1 | existing `benchmarks/benchmark_redknot.py`, reused unchanged |
| Original 4 × 15 case identities, geometry and hashes | `benchmarks/data/flash_release_catalog.json` |
| Independent complete local model-file SHA inventory | `benchmarks/data/flash_model_files.json` |
| CPU regression coverage | `tests/test_flash_reproduction.py` |

The wrapper does not install anything, download a model or dataset, stop GPU
processes, manipulate SSH mappings, or change the GPU keeper. It uses
`REDKNOT_PYTHON` if supplied, otherwise `python3`.

## CPU-only use

From this repository directory:

```bash
./benchmarks/run_deepseek_v4_flash_reproduction.sh --help
./benchmarks/run_deepseek_v4_flash_reproduction.sh --dry-run
./benchmarks/run_deepseek_v4_flash_reproduction.sh --suite 64K --dry-run
```

Default execution is a CPU plan of all 60 source cases. `status=cpu_plan_only`
and exit code 0 mean the plan was generated, **not** that its blockers passed.
The result explicitly reports `gpu_initialized=false`, `model_verified=false`,
the required payload bytes and all blockers. CPU import does not load torch,
vLLM or SGLang.

For your own exact-token corpus:

```bash
./benchmarks/run_deepseek_v4_flash_reproduction.sh \
  --suite custom --cases /data/exact-token-cases.json \
  --prepare-only --output /data/new-prepared-cases.json
```

Input format (illustrative token IDs, not a useful accuracy dataset):

```json
{"cases":[{"id":"example","chunks":[[1,2],[3,4]],"query":[5],"references":["answer"]}]}
```

One to eight nonempty chunks are accepted. Preparation preserves all IDs and
references and writes a **new** file; it never truncates, re-tokenizes, pads or
overwrites inputs. Missing references are unscored, never interpreted as F1=1.
The miniature example is intentionally blocked for GPU performance measurement
because it has no clean rows beyond the default 128-token dirty boundary.

## Frozen suite identity and remaining token-export step

The source is RedKnot commit
`55ee4e8401603f8d2612877e4053e18b37b1c1bd`, specifically
`test/srt/redknot/datasets/LongBench/suites/RELEASE_SUITES.json`, its four release
JSONL files, and their 60 referenced profile records. The derived catalog embeds
each source path and SHA256. It retains 64K/128K/256K/440K geometry, case ordering,
question/gold answers, full-prompt hash, per-chunk hashes, query hash and original
accuracy eligibility. It does not copy the upstream LongBench raw corpus.
Dataset attribution/terms remain those of the underlying datasets; RedKnot's
metadata and source are Apache-2.0.

**Those source records contain hashes, not token IDs.** The original prompt
builder is embedded in a large SGLang-specific benchmark and has not been
imported as a hidden dependency. To prepare a frozen suite, supply an independent
exact-token export in the above input schema, with IDs formatted as
`64K:short_00_hotpotqa_row0` (all 15 IDs for a selected suite; all 60 for `all`).
Export the already constructed `prompt_chunks` and each query's online token
suffix before the original engine call; do not re-tokenize raw text with a
different chat template. This migration does **not** yet automate that source
export. The converter requires exact suite membership and matches every length
and the original SHA256 serialization (unsigned uint32, little endian).
Missing cases, reordering within a chunk, changed boundaries, suffix changes and
different supplied reference answers are rejected.

Original long-output cases were excluded from the short-span F1 aggregate.
The converter retains their original gold answers in provenance but emits
`references=[]`, so those outputs are not scored against an inappropriate short
span. Their complete model outputs are still recorded. The generic runner marks
missing-reference quality qualification as unavailable; that is an explicit
limitation, not a failed token conversion or fabricated zero-error result.

## Actual GPU execution is opt-in

After independently ensuring the selected GPU is free and its owned keeper has
yielded, an example for a suitably sized custom corpus is:

```bash
CUDA_VISIBLE_DEVICES=0 ./benchmarks/run_deepseek_v4_flash_reproduction.sh \
  --run --gpu-confirmed-idle --suite custom \
  --cases /data/new-prepared-cases.json \
  --model /workspace/Models/DeepSeek-V4-Flash-0731 \
  --config examples/deepseek_v4_flash_policy.json \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --output /data/new-flash-vllm-result.json
```

The caller must already have the pinned vLLM and plugin installed. One visible
GPU, TP/PP=1, one serial request, eager V1, native FlashMLA sparse attention,
prefix caching off and chunked prefill off are required. There is no fake TP8
flag or implicit distributed launch. H200/B300 SGLang environment settings are
not transplanted into vLLM, and passing a CPU preflight does not certify that a
GPU has sufficient memory or the correct kernels.

Before loading vLLM, all 74 local files of Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062` must exist with their exact sizes and
SHA256. The helper checks the inventory first, then hashes every full file and
records a separate `<output>.model-verification.json`. Missing shards, partial
files, symlinks, SHA mismatches and mutation during a file read fail closed. It
does not trust a download progress counter or an operator checkbox as proof of
model completeness. Keep the checkpoint immutable while benchmarking. The
inventory derives from the already pinned download manifest with SHA256
`780edbcd5eeb25acdc08a8d743843675130e385c3bb2b8f5b51cbe229b42b38c`;
no network source is used at run time.

The owned vLLM engine is shut down in `finally`. No other process is killed.
Restoring the selected GPU's keeper is the caller/keeper monitor's responsibility.

## Cache budget is not silently raised

The current Flash backend caches per-layer BF16 `z_off[T, 8*1024]` and int64
source positions. With the supplied 37 selected layers the payload is:

```text
37 * (2 * 8 * 1024 + 8) = 606,504 bytes per cached token
8 GiB / 606,504 = 14,163 complete cached tokens
```

Even one 64K frozen case needs approximately 37.02 GiB of payload, before Python
objects, staging, native GPU KV and model memory. The generic runner captures
all unique chunks before all pairs; therefore the preflight also checks the
union across the entire selected suite, not just the largest individual case.
Shared chunks are counted once. Staging bytes are reported separately. Merely
changing `max_model_len` does not make an oversized suite supported. No policy
ratio or MLA aggregation algorithm is changed by this benchmark migration.

## Measurement contract and intentionally missing features

Each case uses at least 3 untimed warmup pairs and 10 measured pairs, alternating
Recomputed/RedKnot order with the same greedy fixed output budget (128 by default,
minimum 50). Startup, offline capture and warmup are excluded from hot TTFT;
offline capture wall time is separately retained. TTFT comes directly from
`output.metrics.first_token_latency`; it does not subtract incompatible clocks.
The JSON records all measured text/token outputs, per-case F1, mean F1 decrease
in **percentage points**, TTFT p50/p95, ratio of mean TTFT, and the separately
named mean paired TTFT ratio. One complete first measured Recomputed/RedKnot
output pair per case is also printed for human inspection. These are serial hot
latencies, **not QPS** or throughput claims.

This differs from the original SGLang release's short-answer stopping rules,
30-token supplemental outputs and client-streaming clock. Reports set
`source_reproduction_equivalent=false`; do not compare the resulting speedup as
if every protocol/engine setting were identical. English token-overlap F1 is
not a semantic metric for Chinese answers or an automatic validation of long
explanations. Actual reusable-head/capture/native-state evidence is checked by
the existing runner, not inferred from a desired speedup.

Not migrated here: automated original-token export, TP8, true concurrent/QPS
tests, the original SGLang long-context memory optimizations, adaptive Sparse
FFN/MoE execution, six-model release parity, Pro, Mistral and Qwen3.5 serving
entrypoints. They are not represented by placeholder scripts claiming to run.

CPU validation for this entrypoint: 20 new tests pass, including source catalog
pinning, exact token hash preservation, F1 eligibility, cache rejection,
partial-model/SHA rejection, default no-GPU behavior and the explicit run guard.
No full-model GPU benchmark was run as part of this code migration.
