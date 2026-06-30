# RedKnot Experiments 2 and 3

This document designs and operationalizes two additional RedKnot system experiments beyond accuracy, TTFT, and QPS:

- Experiment 2: end-to-end latency breakdown
- Experiment 3: throughput under load

Both experiments compare **Recompute** against **RedKnot**. CacheBlend and ProphetKV can be added later if corresponding serving implementations are available, but the critical system comparison here is lossless RedKnot versus full recomputation.

## Experiment 2: Latency Breakdown

### Goal

Show where RedKnot saves time in a single long-context request. TTFT alone hides whether the improvement comes from prefill, decode, patch/setup overhead, or end-to-end latency. This experiment decomposes latency into prefill and decode phases.

### Metrics

| Metric | Meaning | Why it matters |
|---|---|---|
| Prefill latency | Time for processing input context and building KV/cache state | Main bottleneck for long-context serving |
| Decode latency | Time for generating output tokens after prefill | Checks RedKnot does not hurt generation |
| Total latency | Prefill + decode | User-visible single-request latency |
| Prefill tokens/s | Input tokens / prefill latency | Normalized prefill efficiency |
| Decode tokens/s | Output tokens / decode latency | Normalized decode efficiency |
| Prefill speedup | Recompute prefill latency / RedKnot prefill latency | Directly quantifies prefill savings |

### Workload

Recommended default workload:

| Model | Context lengths | Output length | Samples | Dataset |
|---|---:|---:|---:|---|
| Qwen3.5-397B-A17B | 16K, 32K, 64K | 64 | 3 | LongBench triviaqa |

Use the same prompt construction for Recompute and RedKnot. The context is concatenated/truncated to the target token length. Each context length is warmed up before timed measurement.

### Method

Recompute path:

1. Tokenize the full prompt.
2. Run one full prefill with `use_cache=True`.
3. Decode `MAX_NEW` tokens autoregressively.
4. Record prefill time, decode time, and token throughput.

RedKnot path:

1. Build RedKnot attention/head configuration.
2. Collect attention mass for sparse MoE/FFN selection.
3. Build linear-attention window configuration.
4. Install RedKnot patches.
5. Run chunked/carry-prefix prefill.
6. Decode with the resulting cache.
7. Restore original model patches.
8. Record the same metrics as Recompute.

### Command

```bash
PYTHONPATH=python \
REDKNOT_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-FP8 \
REDKNOT_LONGBENCH_DIR=/path/to/LongBench/data \
REDKNOT_DATASETS=triviaqa \
REDKNOT_CTXS=16000,32000,64000 \
REDKNOT_N_SAMPLES=3 \
REDKNOT_MAX_NEW=64 \
REDKNOT_LATENCY_OUT=test/srt/redknot/figures/latency_breakdown_397b.json \
python test/srt/redknot/latency_multi_ctx_397b.py
```

Plot and print the summary table:

```bash
python test/srt/redknot/plot_latency_breakdown.py \
  --in test/srt/redknot/figures/latency_breakdown_397b.json \
  --out test/srt/redknot/figures/latency_breakdown_397b.png \
  --title "Qwen3.5-397B-A17B latency breakdown"
```

### Output

The latency script now saves:

- `/tmp/multi_ctx_results.pkl`
- `test/srt/redknot/figures/latency_breakdown_397b.json`

The plotting script saves:

- `test/srt/redknot/figures/latency_breakdown_397b.png`

### Expected Analysis

The main claim should be based on prefill latency and total latency:

- RedKnot should reduce prefill latency more strongly at longer contexts.
- Decode latency should remain similar or only slightly changed.
- Total latency improvement should be dominated by prefill savings.
- If RedKnot setup overhead is visible, report it as part of prefill cost to avoid hiding overhead.

## Experiment 3: Throughput Under Load

### Goal

Show RedKnot behavior under real serving pressure, not just isolated single-request latency. This experiment measures how QPS, output token throughput, and tail latency change as concurrency increases.

### Metrics

| Metric | Meaning | Why it matters |
|---|---|---|
| Request throughput | Completed requests per second | Primary serving throughput metric |
| Output token throughput | Generated tokens per second | Throughput normalized by output length |
| Median latency | p50 end-to-end request latency | Typical user experience |
| P99 latency | p99 end-to-end request latency | Tail latency / SLA risk |
| Saturation point | Concurrency where QPS stops scaling or p99 explodes | Practical serving capacity |

### Workload

Recommended default workload:

| Model | Input length | Output length | Concurrency sweep | Number of prompts |
|---|---:|---:|---:|---:|
| Qwen3.5-397B-A17B | 8K | 128 | 1, 2, 4, 8, 16, 32, 64 | 256 |

Use `request-rate=inf` and `max-concurrency=C` so that the concurrency value is the actual in-flight request cap. Use the same fixed request pool for Recompute and RedKnot.

### Method

For each backend:

1. Start a fresh SGLang server.
2. Wait for health check to pass.
3. For each concurrency level, run `sglang.bench_serving` with the same model and fixed workload.
4. Append one JSONL metric row per concurrency point.
5. Shut down the server and free GPU memory.
6. Repeat for the next backend.

Backends:

- `baseline`: default SGLang attention backend, labeled as Recompute in plots.
- `redknot`: SGLang RedKnot backend with RedKnot head configuration.

### Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=python \
python test/srt/redknot/benchmark_RedKnot_QPS_sweep.py \
  --model-path /path/to/Qwen3.5-397B-A17B-FP8 \
  --tp-size 8 \
  --quantization fp8 \
  --redknot-head-config test/srt/redknot/head_class/qwen3.5-397B-A17B_redknot_server.json \
  --context-length 40000 \
  --mem-fraction-static 0.85 \
  --concurrency 1,2,4,8,16,32,64 \
  --num-prompts 256 \
  --random-input-len 8000 \
  --random-output-len 128 \
  --out test/srt/redknot/figures/qps_sweep_qwen35_397b.jsonl
```

Plot and print the summary table:

```bash
python test/srt/redknot/plot_qps_sweep.py \
  --in test/srt/redknot/figures/qps_sweep_qwen35_397b.jsonl \
  --out test/srt/redknot/figures/qps_sweep_qwen35_397b.png \
  --title "Qwen3.5-397B-A17B throughput under load"
```

### Output

The sweep script saves:

- `test/srt/redknot/figures/qps_sweep_qwen35_397b.jsonl`

The plotting script saves:

- `test/srt/redknot/figures/qps_sweep_qwen35_397b.png`

The plot contains:

- QPS vs concurrency
- Output token throughput vs concurrency
- P99 latency vs concurrency

### Expected Analysis

The throughput-under-load claim should focus on three points:

- At low concurrency, RedKnot should reduce request latency because each request spends less time in long-context prefill.
- At medium concurrency, RedKnot should support higher QPS before saturation.
- At high concurrency, RedKnot should delay the point where p99 latency rises sharply.

The most convincing result is not only higher peak QPS, but a better QPS-versus-p99 tradeoff under the same latency SLA.

## Reporting Template

Use this short form in the paper or experiment section:

```text
We evaluate RedKnot under two additional serving metrics. First, we decompose single-request latency into prefill and decode time across long-context lengths. This verifies that RedKnot's gain primarily comes from reducing long-context prefill while preserving decode performance. Second, we run a real SGLang continuous-batching server and sweep request concurrency. We report QPS, output tokens/s, median latency, and p99 latency, which captures serving capacity and tail-latency behavior under load.
```
