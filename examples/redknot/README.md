# RedKnot Demos

This directory contains two runnable demos for the SGLang RedKnot integration:

| Demo | File | What It Measures |
|---|---|---|
| RAG end-to-end | `rag_redknot_demo.py` | HotpotQA-style RAG quality, TTFT, and analytical FLOPs for dense baseline vs RedKnot online KV reuse |
| SegPagedAttention micro-benchmark | `segpaged_redknot_demo.py` | Dense KV + mask vs ragged per-head KV + FA-3 varlen attention latency and numerical equivalence |

## HuggingFace Assets

Default RAG demo assets:

| Type | HuggingFace Address | Notes |
|---|---|---|
| Model | `Qwen/Qwen3-32B` | Default model used by `rag_redknot_demo.py` |
| Dataset | `hotpotqa/hotpot_qa` | Use config `distractor`, split `validation` |

Other model addresses used by the paper-style experiments:

| Model | HuggingFace Address | Notes |
|---|---|---|
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | Requires Meta license access on HuggingFace |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | Useful for smaller smoke tests |
| Small Qwen smoke test | `Qwen/Qwen3-0.6B` | Not paper-scale, but practical for validating script wiring |

## RAG Demo

The RAG demo builds HotpotQA distractor samples, formats each sample as multiple reusable document segments, and compares:

| Path | Meaning |
|---|---|
| Dense baseline | Full chunked prefill over the concatenated RAG prompt |
| RedKnot | Offline prefill per document segment, then online position-independent KV reuse with head-classified attention |

Metrics written to JSON:

| Metric Group | Fields |
|---|---|
| Answer quality | `baseline_f1`, `redknot_f1`, `baseline_em`, `redknot_em` |
| Logit fidelity | `logits_cosine`, `top1_match_rate`, `top10_overlap` |
| Speed | `baseline_ttft_s`, `redknot_online_ttft_s`, `wall_speedup`, `offline_prefill_s` |
| Compute | `baseline_flops`, `redknot_online_flops`, `flops_speedup`, `flops_savings` |

Run a paper-style Qwen3-32B RAG sample:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0,1 python examples/redknot/rag_redknot_demo.py \
  --model-path Qwen/Qwen3-32B \
  --dataset hotpotqa/hotpot_qa \
  --dataset-config distractor \
  --split validation \
  --n-samples 1 \
  --n-segments 6 \
  --tokens-per-segment 5000 \
  --output /tmp/redknot_rag_qwen3_32b.json
```

Run a smaller wiring smoke test:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0 python examples/redknot/rag_redknot_demo.py \
  --model-path Qwen/Qwen3-0.6B \
  --dataset hotpotqa/hotpot_qa \
  --dataset-config distractor \
  --split validation \
  --n-samples 1 \
  --n-segments 4 \
  --tokens-per-segment 512 \
  --max-new-tokens 16 \
  --output /tmp/redknot_rag_smoke.json
```

Use a pre-profiled head config JSON if you have one:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0,1 python examples/redknot/rag_redknot_demo.py \
  --model-path Qwen/Qwen3-32B \
  --head-config /path/to/qwen3-32b-redknot-head-config.json \
  --n-samples 1 \
  --output /tmp/redknot_rag_with_profiled_heads.json
```

If `--head-config` is omitted, the demo generates a simple policy from model metadata: roughly 15% KV heads are global and the rest are local with `--local-window 256`. This is intended for a runnable demo. Paper-quality results should use profiled head configs.

## SegPagedAttention Demo

`segpaged_redknot_demo.py` isolates the engine/layout question behind the paper's SegPagedAttention results:

| Path | Storage | Kernel | Meaning |
|---|---|---|---|
| `A1_manual` | Dense `[B,Hkv,L,D]` | Manual `matmul + mask + softmax` | Slow reference path |
| `A2_sdpa_mask` | Dense `[B,Hkv,L,D]` | PyTorch SDPA with `attn_mask` | Demonstrates the mask fallback penalty |
| `B_segpaged_fused` | Ragged per-head KV | One FA-3 varlen call per layer | Intended SegPagedAttention physical layout |

Run a fast smoke test:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0 python examples/redknot/segpaged_redknot_demo.py \
  --mode smoke \
  --output /tmp/redknot_segpaged_smoke.json
```

Run the paper-mode sweep:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0 python examples/redknot/segpaged_redknot_demo.py \
  --mode paper \
  --output /tmp/redknot_segpaged_paper.json
```

`paper` mode includes 8K/32K/128K decode, 8K/32K/128K prefill, and batch=4 prefill. It can take several minutes because the dense+mask 128K prefill path is intentionally expensive.

## SegPagedAttention Backend

SegPagedAttention is also exposed as a standalone SGLang attention backend:

```bash
python -m sglang.launch_server \
  --attention-backend segpaged \
  --segpaged-head-config-path /path/to/head_config.json \
  --segpaged-page-size 64
```

Backend behavior:

| Component | Behavior |
|---|---|
| Head config | Per `(layer, kv_head)` global/local/retrieval policy loaded from JSON |
| Global heads | Read full-context KV pages |
| Local heads | Read only `sink + recent window` KV pages |
| Kernel | Mask-free SegPagedAttention varlen path when available; exact fallback otherwise |

The backend is registered as `segpaged`, separate from `redknot`. The existing
`redknot` backend can still enable SegPaged decode with `--redknot-segpaged-decode`,
but `--attention-backend segpaged` is the clean paper-style backend entry point.

### DuoAttention Test

The paper-style DuoAttention/SegPagedAttention synthetic test lives at:

`test/srt/redknot/test_segpaged_duo_attention.py`

It compares dense DuoAttention (`global` heads full-context, `local` heads
masked to `sink + window`) against SegPagedAttention (`global` heads full pages,
`local` heads physically store only `sink + window`). It prints cosine,
latency, speedup, and KV token savings.

Run a CPU smoke test:

```bash
cd /workspace/096/RedKnot
PYTHONPATH=python python test/srt/redknot/test_segpaged_duo_attention.py \
  --cpu --seq-len 1024 --q-len 2 --repeat 2
```

Run a GPU 32K decode-style test:

```bash
cd /workspace/096/RedKnot
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 \
python test/srt/redknot/test_segpaged_duo_attention.py \
  --seq-len 32768 --q-len 1 --repeat 10
```

## Requirements

Common requirements:

```bash
pip install transformers datasets accelerate safetensors
```

RAG demo requirements:

| Requirement | Why |
|---|---|
| CUDA GPU | Large model inference and TTFT measurement |
| `transformers` | Model and tokenizer loading |
| `datasets` | Loading `hotpotqa/hotpot_qa` from HuggingFace |
| `flash_attn` | Required when `--kernel fa2` is used for RedKnot attention |

SegPagedAttention demo requirements:

| Requirement | Why |
|---|---|
| Hopper-class or compatible FA-3 GPU | `B_segpaged_fused` uses `sgl_kernel.flash_attn.flash_attn_varlen_func` |
| Built `sgl_kernel.flash_attn` | Provides FA-3 varlen kernel |

Quick FA-3 check:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from sgl_kernel.flash_attn import is_fa3_supported
print(torch.cuda.get_device_name(0))
print("FA3:", is_fa3_supported())
PY
```

## Output Files

Both demos write structured JSON. `segpaged_redknot_demo.py` also writes a Markdown summary next to the JSON output.

Example RAG summary shape:

```json
{
  "model": "Qwen/Qwen3-32B",
  "dataset": "hotpotqa/hotpot_qa",
  "summary": {
    "baseline_f1": 0.5,
    "redknot_f1": 0.5,
    "baseline_ttft_s": 12.3,
    "redknot_online_ttft_s": 5.6,
    "flops_speedup": 2.1,
    "flops_savings": 0.52
  },
  "results": []
}
```

## Test Scripts

The repository also includes two one-command test/demo scripts under
`test/srt/redknot/`. These are intended for quick reproduction with local
or HuggingFace assets.

### Qwen3-32B + HotpotQA 16K

Script:

`test/srt/redknot/test_redknot_hotpot_16k_demo.py`

Assets:

| Type | HuggingFace Address | Local Override |
|---|---|---|
| Model | `Qwen/Qwen3-32B` | `REDKNOT_MODEL_PATH` |
| Dataset | `hotpotqa/hotpot_qa` (`distractor`, `validation`) | `REDKNOT_HOTPOT_PARQUET` |

Configuration:

| Setting | Value |
|---|---|
| Context layout | `4 x 4K` (~16K) |
| Online prefill | parallel/batched RedKnot |
| Sparse FFN | `dense_until=3`, `mass_thresh=0.3`, `recent_n=64` |
| Default samples | `10` (`REDKNOT_N_SAMPLES` to override) |

Run with HuggingFace assets:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python test/srt/redknot/test_redknot_hotpot_16k_demo.py
```

Run with local assets:

```bash
cd /workspace/096/RedKnot
REDKNOT_MODEL_PATH=/workspace/096/models/Qwen3-32B \
REDKNOT_HOTPOT_PARQUET=/workspace/096/__REDKNOT_V02__/datasets/HotpotQA/distractor/validation-00000-of-00001.parquet \
REDKNOT_N_SAMPLES=10 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python test/srt/redknot/test_redknot_hotpot_16k_demo.py
```

Output includes only RedKnot answers, TTFT, and analytical compute.

### Llama-3.3-70B + MuSiQue 35K

Script:

`test/srt/redknot/test_redknot_musique_35k_llama70b.py`

Assets:

| Type | HuggingFace Address | Local Override |
|---|---|---|
| Model | `meta-llama/Llama-3.3-70B-Instruct` | `REDKNOT_MODEL_PATH` |
| Dataset | `dgslibisey/MuSiQue` | `REDKNOT_MUSIQUE_JSONL` |

Configuration:

| Setting | Default |
|---|---|
| Context layout | `5 x 7K` (~35K) |
| Online prefill | parallel/batched RedKnot |
| Baseline | full dense recompute over the concatenated `5 x 7K` prompt |
| Head preset | `optimal` (`REDKNOT_HEAD_PRESET`) |
| FFN preset | `balanced` (`REDKNOT_FFN_PRESET`) |
| Default samples | `5` (`REDKNOT_N_SAMPLES` to override) |

Head presets:

| Preset | Config |
|---|---|
| `optimal` | `configs/llama-70B_optimal_g15_lf_ret.json` |
| `w256` | `configs/llama-70B_w256_g15_lf_ret.json` |
| `w1024` | `configs/llama-70B_w1024_g15_lf_ret.json` |
| `all_global` | `configs/llama-70B_all_global.json` |

FFN presets:

| Preset | `dense_until` | `mass_thresh` | `recent_n` |
|---|---:|---:|---:|
| `speed` | 3 | 0.3 | 64 |
| `balanced` | 20 | 0.5 | 128 |
| `quality` | 32 | 0.7 | 128 |
| `dense` | 80 | 1.0 | 128 |

Run default comparison:

```bash
cd /workspace/096/RedKnot
REDKNOT_HEAD_PRESET=optimal \
REDKNOT_FFN_PRESET=balanced \
REDKNOT_N_SAMPLES=5 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python test/srt/redknot/test_redknot_musique_35k_llama70b.py
```

Run with local assets:

```bash
cd /workspace/096/RedKnot
REDKNOT_MODEL_PATH=/workspace/096/models/Llama-3.3-70B-Instruct \
REDKNOT_MUSIQUE_JSONL=/workspace/096/__REDKNOT_V02__/datasets/musique_ans_v1.0_dev.jsonl \
REDKNOT_HEAD_PRESET=optimal \
REDKNOT_FFN_PRESET=balanced \
REDKNOT_N_SAMPLES=5 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python test/srt/redknot/test_redknot_musique_35k_llama70b.py
```

Output includes baseline vs RedKnot answers, TTFT speedup, and compute
speedup/savings. Set `REDKNOT_PRINT_METRICS=1` to also print F1/EM against
MuSiQue aliases.

## P0 Components: Sparse FFN and SegPagedAttention Runtime

Two paper-critical mechanisms are implemented in
`python/sglang/srt/layers/attention/redknot/`:

| Component | File | Paper Reference |
|---|---|---|
| Partial Sparse FFN recovery | `sparse_ffn.py` | §4.2, Algorithm 1 lines 20-23 |
| SegPagedAttention runtime | `segpaged.py` | §4.3, Algorithm 2, fig. 6 |

### Sparse FFN recovery (`sparse_ffn.py`)

Token-selective FFN that attacks the short-context FFN bottleneck (57-62% of
TTFT at 2-8K tokens). Deep layers run the dense FFN only on important tokens
(selected from the recovered attention signal) and route the rest through the
residual identity path.

| API | Role |
|---|---|
| `SparseFFNSchedule(dense_until, mass_thresh, recent_n)` | Layer-wise policy (paper Table 4: `dense_until=20, mass_thresh=0.5, recent_n=128`) |
| `select_important_tokens` | `SelectImportantTokens` operator (Algorithm 1 line 20) |
| `apply_sparse_ffn` | Dense/identity dispatch (lines 21-23) |
| `sparse_ffn_flops` | Analytical FFN FLOP savings |

Enable it in the RAG demo:

```bash
cd /workspace/096/RedKnot
CUDA_VISIBLE_DEVICES=0,1 python examples/redknot/rag_redknot_demo.py \
  --model-path Qwen/Qwen3-32B \
  --dataset hotpotqa/hotpot_qa --dataset-config distractor --split validation \
  --n-samples 1 --n-segments 6 --tokens-per-segment 5000 \
  --sparse-ffn --ffn-dense-until 20 --ffn-mass-thresh 0.5 --ffn-recent-n 128 \
  --output /tmp/redknot_rag_sparseffn.json
```

The demo reports deep-layer selected fraction, FFN FLOPs savings, and FFN
speedup. With `--ffn-mass-thresh 1.0` the FFN is bit-for-bit dense, which is
useful to confirm equivalence before turning sparsity on.

### SegPagedAttention runtime (`segpaged.py`)

A real per-`(layer, head, segment)` paged KV store plus a mask-free fused
varlen attention path. Local heads physically store only `sink + recent`
tokens; global heads store the full context. This is the layout that turns
algorithmic head sparsity into physical memory and bandwidth savings.

| API | Role |
|---|---|
| `SegmentPageTable` | `(layer, head, segment) -> consecutive virtual pages` (Algorithm 2 line 3) |
| `SegPagedKVCache` | Physical per-head paged KV pool with virtual page table |
| `build_segpaged_cache` | Build a layer's store from dense per-head KV with head-class policy |
| `segpaged_attention` | Algorithm 2 execution (GLOBAL = full pages; LOCAL = sink + recent pages); fused FA-3 varlen with exact PyTorch fallback |
| `verify_against_dense` | Numerical-equivalence harness (cos vs dense+mask) + KV-token saving |

Quick numerical check (CPU-friendly, no model needed):

```bash
cd /workspace/096/RedKnot
PYTHONPATH=python python - <<'PY'
from sglang.srt.layers.attention.redknot import verify_against_dense
import torch
rep = verify_against_dense(
    num_kv_heads=8, num_q_per_kv=4, head_dim=128,
    seq_len=2048, sink=4, recent=256, global_ratio=0.5,
    page_size=64, q_len=8, dtype=torch.float32, seed=1,
)
print(rep)
PY
```

Expected: `cosine` near 1.0 (paper requires > 0.99998) and a positive
`kv_token_saving` that grows with context length.

## P1 / P2 Components

Beyond the two P0 mechanisms, the following components are implemented in
`python/sglang/srt/layers/attention/redknot/`:

| Tier | Component | File / Integration | Paper Reference |
|---|---|---|---|
| P1 | Automatic head profiling | `head_profiler.py` | §3.2, fig. 3 |
| P1 | SegPaged decode in backend | `redknot_backend.py` (`--redknot-segpaged-decode`) | §5.4 / §5.5 |
| P2 | PD head-class KV transfer | `pd_transfer.py` | fig. 13a |
| P2 | Head-aware scheduler / capacity | `scheduler.py` | fig. 13c, §6 |
| P2 | Evaluation harness | `eval_harness.py` | §5, Table 3/4 |

### P1: Head profiling (`head_profiler.py`)

Classifies every `(layer, head)` as `global` (prefix-sensitive) or `local`
(prefix-robust) offline, producing a `HeadClassConfig` JSON for
`--redknot-head-config-path`.

| API | Role |
|---|---|
| `profile_model_heads(model, tokenizer, texts)` | Run profiling on a real HF model (needs `attn_implementation="eager"`) |
| `classify_from_stats(prefix_shift, edge_mass, target_global_ratio=0.15)` | Pure, model-free classifier (unit-testable) |
| `build_head_config` / `save_head_config_json` | Package + write the JSON the backend loads |

### P1: SegPaged decode in the backend

`--redknot-segpaged-decode` makes the RedKnot backend run decode through
the per-head paged KV view + fused varlen path (local heads read only
sink+recent; global heads read full context), instead of dense SDPA:

```bash
python -m sglang.launch_server --attention-backend redknot \
  --redknot-head-config-path /path/to/head_config.json \
  --redknot-kernel fa3 \
  --redknot-segpaged-decode --redknot-page-size 64 ...
```

### P2: PD head-class KV transfer (`pd_transfer.py`)

Ships only the KV each head can consume across the prefill->decode boundary.

| API | Role |
|---|---|
| `build_transfer_payload(kv_per_layer, head_config)` | Slice KV to head-class visible region |
| `HeadClassKVPayload.saving()` | Report `byte_reduction_x` (paper: 4.3-6.3×) |
| `serialize` / `deserialize` | Transport-agnostic packing |
| `restore_payload` | Rebuild dense `[H, L, D]` KV on the decode side |

### P2: Head-aware scheduler / capacity (`scheduler.py`)

| API | Role |
|---|---|
| `per_session_kv_bytes` / `concurrent_capacity` | Dense vs. head-aware footprint and the per-GPU `capacity_multiplier` (paper: 4.7-7.8×) |
| `HeadAwareCachePolicy` | Content-addressed chunk cache; admission/eviction prioritise reuse frequency over prefix length (§6.2) |

### P2: Evaluation harness (`eval_harness.py`)

| API | Role |
|---|---|
| `DATASET_SPECS` / `list_datasets` | The six QA datasets with HuggingFace ids and RAG layouts (Table 3) |
| `aggregate_quality` / `aggregate_efficiency` | Reduce per-sample records into the paper's summary rows |
| `coefficient_of_variation` | TTFT stability (CoV, Table 4) |
| `pd_throughput_projection` | Project capacity-bound throughput from a capacity multiplier (fig. 13c) |

Paper-scale sanity check (CPU, no model):

```bash
cd /workspace/096/RedKnot
PYTHONPATH=python python - <<'PY'
import torch
from sglang.srt.layers.attention.redknot import (
    build_head_config, build_transfer_payload, concurrent_capacity,
)
NL, KVH, L, D = 80, 8, 16000, 128            # Llama-70B-like, ~10% global
labels = [["global"] + ["local"] * 7 for _ in range(NL)]
cfg = build_head_config(labels, num_layers=NL, num_kv_heads=KVH,
                        local_window=256, sink_size=4)
kv = [(torch.zeros(KVH, L, D, dtype=torch.bfloat16),
       torch.zeros(KVH, L, D, dtype=torch.bfloat16)) for _ in range(NL)]
print("PD reduction:", build_transfer_payload(kv, cfg).saving()["byte_reduction_x"])
print("capacity x:", concurrent_capacity(cfg, seq_len=L, head_dim=D,
      kv_budget_bytes=40 * 1024**3)["capacity_multiplier"])
PY
```

Expected (paper-scale config): PD reduction ~7× and capacity ~7×.

## Output Files (continued)

When `--sparse-ffn` is on, the RAG summary additionally exposes the
deep-layer selected fraction and FFN savings through the per-sample log.

## Caveats

- The RAG demo's FLOPs are analytical estimates for the online path. Offline document prefill is reported separately as `offline_prefill_s` because it is reusable across requests.
- All P0/P1/P2 mechanisms are implemented as unit-testable modules with clear SGLang integration points. The CPU self-tests validate correctness and reproduce the paper's magnitude on paper-scale configs.
- `segpaged_attention` runs the fused FA-3 varlen kernel on Hopper GPUs and falls back to an exact PyTorch reference elsewhere; both paths are numerically equivalent. SegPaged decode is wired into the backend via `--redknot-segpaged-decode`; the prefill path remains a further integration step.
- `pd_transfer` and `scheduler` provide the payload/policy/capacity logic; binding them to a concrete transport (Mooncake/NIXL/RDMA) and the live SGLang scheduler is the remaining production wiring.
- Large HuggingFace models may require license approval, local cache space, and multi-GPU `device_map=auto` execution.
