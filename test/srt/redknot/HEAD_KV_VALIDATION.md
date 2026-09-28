# Head KV implementation qualification — 2026-09-28

This is an experimental MHA/GQA KV manager, not a production serving rollout.
See the package README for API usage and integration boundaries.

## Measured qualification

- 91 tests passed on two visible CUDA GPUs, with no skipped tests. Tests cover COW, atomic admission, event retirement, device changes, real HTTP transfers, durable decisions and fault handling.
- Three independent owner/source/destination processes transferred 98,304 bytes between two GPUs on one host. Non-prefix occurrence reordering, two-page repair, unchanged parent hashes, exact target KV and same-policy attention passed. READY checkpoint data remained resident through authority COMMIT; stale owners were fenced. All page/store references and process CUDA allocations were reclaimed.
- Real pretrained Qwen3-8B FP32 (36 layers, 32 Q/8 KV heads): 40/40 next-token argmax agreement; maximum logit error 3.7193298e-05. Shared partial tails detached correctly and all 3168 pool pages were reclaimed. The model experiment uses a batch-one instance adapter and short teacher-forced tokens, not a full SGLang scheduler.
- BF16 strict model qualification **fails**: original HF eager 37/40 argmax, max logit difference 0.2578125; separately declared dense FP32 accumulation oracle 38/40, max difference 0.25. Thresholds and inputs were not changed. These failures remain separate from successful storage-isolation tests.

## Capacity and performance

One-layer fixture: BF16, 8 KV heads, 32 Q heads, head dimension 128, page size 16, 4096 tokens. Repair 2 heads, half their pages, one quarter of each touched page. At least 3 warmups and 10 measured repetitions; synchronized wall time includes host scheduling and validation.

| Measurement | Head-page COW | Full independent clone |
| --- | --- | --- |
| Identical fixed KV slab budget | 32 MiB | 32 MiB |
| Observed admitted children | 8 | 1 |
| Old payload copied per repair | 1.5 MiB | 16 MiB |
| Median repair latency | 8.143 ms | 0.0802 ms |

Batching touched-row copies and querying shared completion events once per collection improved COW latency from 30.275 ms to 8.143 ms. COW remains slower than the contiguous-copy baseline in this fixture. Admission capacity is not throughput; fixed slab bytes exclude input/reference tensors and scratch workspace. Active page bytes do not equal CUDA reserved memory. The attention comparison is a Torch mathematical reference, not an optimized FlashAttention serving benchmark.

Hardware was reported by the driver as NVIDIA L20X (143771 MiB per GPU); Python 3.11.13, PyTorch 2.9.1+cu128, Triton 3.5.1, Transformers 4.57.1. No physical H200 identity is inferred from the driver label.

## Limits

This qualifies single-host, multi-process HTTP host staging, not cross-physical-host networking, RDMA, replicated authority failover or full TP/CP/PP serving migration. Synthetic non-prefix KV tests do not establish an LLM reuse policy's quality. Certificates are trusted adapter attestations. MLA/quantized/native state adapters and full scheduler integration remain outside this implementation. Snapshot budgets are deliberately finite; large whole-model checkpoints need a chunked manifest/bundle extension.

## Reproduce

Run the four `test_head_kv_*.py` suites under `test/srt/redknot` with pytest.
For a two-device component experiment:

```bash
python test/srt/redknot/validate_head_kv_distributed_gpu.py \
  --source-device cuda:0 --destination-device cuda:1 \
  --dtype bfloat16 --output distributed.json
python test/srt/redknot/benchmark_head_kv_manager.py \
  --device cuda:0 --dtype bfloat16 --heads 8 --pages 256 \
  --dirty-head-ratios .25 --admission-heads 8 \
  --admission-pages-per-head 256 --admission-capacity-pages 4096 \
  --output capacity.json
python test/srt/redknot/validate_head_kv_qwen3.py \
  --model-path /path/to/Qwen3-8B --device cuda:0 --dtype float32 \
  --output model_fp32.json
```

Running the same model command with BF16 retains strict all-position token-agreement gating and can report the documented failure. `--dense-backend dense_fp32_accum --attention-probe` records a separately labeled mathematical comparison and identical-input layer diagnostics; it does not override the original eager result.
