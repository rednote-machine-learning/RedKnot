<p align="center">
  <img src="RedKnot_Logo.png" alt="RedKnot logo" width="600" />
</p>

# RedKnot + DeepInfer

The DeepInfer-based development track of **RedKnot**, an open-source project for efficient long-context LLM serving.

[Website](https://rednote-machine-learning.github.io/RedKnot/) · [RedKnot paper](https://arxiv.org/abs/2606.06256) · [Upstream DeepInfer](https://github.com/deepseek-ai/DeepInfer) · [Apache-2.0 license](LICENSE)

**Status:** This branch starts from the complete [DeepInfer baseline at `5537415`](https://github.com/deepseek-ai/DeepInfer/commit/553741557a57df44a46fb3ab72b6aafeaf945337), including its pinned submodules. RedKnot integration is planned; the features, usage instructions, and benchmarks below describe the upstream DeepInfer baseline.

## Framework branches

| Branch | Purpose |
| --- | --- |
| **[RedKnot+DeepInfer](https://github.com/rednote-machine-learning/RedKnot/tree/RedKnot%2BDeepInfer)** | **Default branch** · DeepInfer baseline for RedKnot development |
| [RedKnot-SGLang](https://github.com/rednote-machine-learning/RedKnot/tree/RedKnot-SGLang) | SGLang-based RedKnot implementation; formerly `main` |
| [RedKnot-vLLM](https://github.com/rednote-machine-learning/RedKnot/tree/RedKnot-vLLM) | vLLM adaptation track |

RedKnot belongs to its contributors. We welcome issues, bug reports, suggestions, and pull requests across all framework branches.

## DeepInfer baseline

DeepInfer is the high-performance inference framework that powers DeepSeek's large-scale production services. This open-source release provides offline inference with the core features needed for serving at scale.

Supported models:

- DeepSeek-V4.1-Flash

Supported features:

- Optimized kernels, model and sampler runtime
- Parallelism: Attention DP / balanced CP, MoE EP + EPLB
- DSpark and confidence-scheduled verification
- SWA bounded replay
- Asynchronous scheduling (zero-overhead batch scheduler)
- Batch-invariant inference
- Deterministic sampling
- Structured outputs

<details>
<summary>Feature details and configuration</summary>

- **Optimized kernels:** [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP), [TileKernels](https://github.com/deepseek-ai/TileKernels), [DeepSelect](https://github.com/deepseek-ai/DeepSelect), and custom [TileLang kernels](deepinfer/model_executor/layers/tilelang_ops).
- **DSpark and confidence-scheduled verification:** accelerate decoding with draft tokens and use confidence estimates to schedule verification. Set `DEEPINFER_ENABLE_MTP_VARLEN=1`; `DEEPINFER_USE_NEXTN_DECODE` controls the maximum draft length (default: 5).
- **Decoder SWA bounded replay:** reduce decoder prefill work by replaying only a bounded suffix. Set `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` and `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` for the default V4.1 settings; the retained suffix must cover the SWA window plus the maximum backtracking length.
- **Asynchronous scheduling:** overlap CPU batch scheduling with device execution. Enabled by default.
- **Batch-invariant inference:** keep model results identical across different batch compositions. Enable with `DEEPINFER_BATCH_INVARIANT=1` (NVIDIA only).
- **Deterministic sampling:** produce identical outputs for identical input tokens and sampling parameters. Set `DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0` and pass a fixed `seq_seed` to `SamplingParams` (NVIDIA only).
- **Structured outputs:** constrain generation to a regex, JSON schema, or EBNF grammar. Pass `so_args={"schema_type": ..., "schema": ...}` to `SamplingParams`; see [examples](examples/_structured_output.py).

</details>

Supported hardware:

- NVIDIA SM100 GPUs
- Huawei Ascend 950 NPUs

Required for a complete inference service (not included in this release):

- A KV cache system
- VE/Prefill/Decode disaggregation (most required engine functionality is already implemented)
- Load balancing across data-parallel replicas for both prefill and decode

## Getting started

Clone this branch together with its upstream dependencies:

```bash
git clone --branch 'RedKnot+DeepInfer' --recurse-submodules https://github.com/rednote-machine-learning/RedKnot.git
cd RedKnot
```

### Prerequisites

- Linux
- GCC >= 13.0
- Python 3.12
- CUDA 13 (NVIDIA only)
- CANN 9.2 (Ascend only)

### Build from source

Before building DeepInfer, please create and activate an empty Python 3.12 environment.

``` bash
# Use `ascend` instead for the Ascend platform
# Keep it exported in the environment where you run DeepInfer
export DEEPINFER_ACCELERATOR=nvidia
./build.sh
```

### Offline inference

The default checkpoint, [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), is
downloaded into the Hugging Face cache on the first run; set `DEEPINFER_HF_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017` to
run from a local copy instead.

``` bash
python examples/offline_inference.py
```

See [examples](examples/README.md) for details.

## Upstream performance benchmarks

| Prefill | Decode |
| :---: | :---: |
| [![Prefill: non-cached input throughput vs. latency](doc/images/prefill_pareto.png)](doc/images/prefill_pareto.png) | [![Decode: output throughput vs. per-user speed](doc/images/decode_pareto.png)](doc/images/decode_pareto.png) |

Measured on 4 SM100 GPUs for prefill and 64 for decode — each within one scale-up domain. To reproduce the benchmarks, see [run_sweep.sh](examples/sweep/run_sweep.sh).

## Evaluation benchmarks

See [eval/README.md](eval/README.md).

## License

DeepInfer is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

This RedKnot development track builds on [DeepInfer](https://github.com/deepseek-ai/DeepInfer) by DeepSeek-AI. We preserve the upstream source, license, and attribution.

DeepInfer was originally built primarily on [vLLM](https://github.com/vllm-project/vllm), with additional inspiration from [SGLang](https://github.com/sgl-project/sglang) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). We thank their developers.

## Citation

```bibtex
@misc{deepinfer2026,
  title = {DeepInfer: High-performance, large-scale inference framework for DeepSeek models},
  author = {DeepSeek-AI},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/deepseek-ai/DeepInfer}},
}
```
