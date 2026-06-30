# DeepSeek-V4-Flash (FP8) RedKnot — Setup & Run Notes

Companion to `requirements_dsv4_flash_fp8.txt`. Covers the install steps that
are not plain `pip install <pkg>`, the environment variables the run needs, and
the current known-good vs. blocked state.

- **Model**: `checkpoints/opensource/DeepSeek-V4-Flash` (FP8 e4m3, block
  `128x128`, scale `ue8m0`; `model_type: deepseek_v4`, 43 layers, 64 heads,
  256 routed experts, `moe_intermediate_size=2048`).
- **Hardware verified**: 8x NVIDIA SM90 (Hopper; reported as "L20Y" but
  `torch.cuda.get_device_properties` returns `sm_90`, 132 SMs), CUDA 12.9.
- **Python env**: use `.venv_tf5` (transformers 5.12). It is a
  `--system-site-packages` venv, so it sees all system packages.

## 1. Python environment

```bash
cd /mnt/tidal-alsh01/dataset/redone/RedKnotV0.3
PY=.venv_tf5/bin/python          # transformers 5.12 + system site-packages
```

## 2. Pip installs

```bash
# safetensors that understands the F8_E8M0 (ue8m0) dtype
pip install "safetensors>=0.7.0"

# MHC / hash kernels (tilelang) — pinned, plus a compatible tvm-ffi
pip install "tilelang==0.1.8" "apache-tvm-ffi==0.1.8.post2"
```

## 3. FlashMLA (build from source, SM90)

The PyPI sdist (`flash_mla 1.0.0+748b13d`) is incomplete (missing headers /
cutlass submodule) and will not compile. Build from the public repo instead;
its API matches what `deepseek_v4_backend.py` calls
(`get_mla_metadata`, `flash_mla_with_kvcache`, `FlashMLASchedMeta`).

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git /root/.cache/flash-mla
cd /root/.cache/flash-mla
git submodule update --init --recursive          # pulls cutlass
FLASH_MLA_DISABLE_SM100=1 MAX_JOBS=32 \
  pip install --no-build-isolation -v .          # SM90 only
```

## 4. DeepGEMM (sgl-project fork) — BUILD FROM SOURCE

SGLang needs `import deep_gemm` with the `recipe_a/recipe_b` API. The prebuilt
PyPI/GitHub wheels (`sgl-deep-gemm==0.1.0`, incl. the `+cu129` one) **segfault**
in their `_C.so` on this host (verified with an isolated grouped-FP8-GEMM smoke
test). Build from source instead so the kernels match the local CUDA/driver:

```bash
git clone --recursive https://github.com/sgl-project/DeepGEMM.git /tmp/sgl_deepgemm_src
cd /tmp/sgl_deepgemm_src
git checkout v0.1.0
git submodule update --init --recursive
pip install --no-build-isolation -v .     # -> deep_gemm 2.5.0+<sha>, links libcudart.so.12
```

Note: building DeepGEMM bumps `apache-tvm-ffi` to 0.1.9; tilelang 0.1.8 still
works with it, so no need to re-pin tvm-ffi afterwards.

## 5. Required environment variables

```bash
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1  # engine checks wrong pkg name
                                               # ("sglang-kernel" vs real "sgl-kernel")
export SGLANG_BARE_SUBPROCESS_LAUNCH=1         # skip numa/memory-saver wrappers
                                               # ("setting membind: Invalid argument"
                                               #  kills spawned schedulers otherwise)
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0        # default uses
                                               # deep_gemm.tf32_hc_prenorm_gemm,
                                               # which the stock deep_gemm lacks;
                                               # 0 selects the built-in fallback
export SGLANG_OPT_USE_TOPK_V2=0                 # the topk_v2 indexer kernel asserts
                                               # "score_stride must be a multiple of 4
                                               #  (TMA 16-byte alignment)" on SM90;
                                               # 0 uses the v1 kernel which has no such
                                               # constraint
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Debug helper: per-rank scheduler logs (crashes before configure_logger):
export SGLANG_RANK_LOG_DIR=/tmp/dsv4_ranklog
```

### MoE backend: FP4 experts on SM90

DeepSeek-V4-Flash has `expert_dtype: fp4` (the MoE experts are FP4-packed, even
though attention/dense are FP8). Native FP4 MoE GEMM in `deep_gemm` requires
**SM100 (Blackwell)** — on SM90 (Hopper) it asserts
`ab.scalar_type()==kPackedFP4 and arch_major==10`. So on SM90 you must NOT use
the deep_gemm MoE runner. Working options on SM90:
- `moe_runner_backend='marlin'`  — verified working (used below).
- `moe_runner_backend='flashinfer_mxfp4'` — needs FlashInfer >= 0.6.11
  (current image has 0.5.3, so it errors out; upgrade if you prefer this path).

## 6. Launch — verified working recipe (Engine API, TP=8)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
REDKNOT_TP_SIZE=8 REDKNOT_N_SAMPLES=1 REDKNOT_LENGTHS=8K REDKNOT_DATASETS=hotpotqa \
REDKNOT_MOE_RUNNER_BACKEND=marlin REDKNOT_DISABLE_CUDA_GRAPH=1 \
SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
SGLANG_BARE_SUBPROCESS_LAUNCH=1 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_tf5/bin/python test/srt/redknot/benchmark_RedKnot_DeepSeekV4_RAG.py
```

A minimal raw smoke (no benchmark harness):

```python
import sglang as sgl
eng = sgl.Engine(
    model_path=".../DeepSeek-V4-Flash", attention_backend="dsv4",
    tp_size=8, disable_cuda_graph=True, mem_fraction_static=0.85,
    moe_runner_backend="marlin",
)
print(eng.generate("Paris is the capital of France. " * 128 +
                   "\nQuestion: What is the capital of France? Answer:",
                   {"temperature": 0.0, "max_new_tokens": 16}))
```

## 7. Status — WORKING

End-to-end run succeeds with the §6 recipe:
- all 46 FP8 shards load; kernel warmup passes; Engine prints `READY`;
- `attention_backend='dsv4'` baseline produces correct output, e.g. HotpotQA@8K
  `Q: who portrayed Steve Urkel -> "Jaleel White"` with **F1=0.80**;
- the benchmark script runs both engines (dsv4 baseline + redknot_mla) to a
  clean SUMMARY and exits 0.

How the blockers were resolved:
- FP8 linear/attention segfault — fixed by building `deep_gemm` from source
  (§4); the prebuilt wheel's `_C.so` was the culprit.
- MoE — DeepSeek-V4's experts are FP4; FP4 MoE GEMM needs SM100, so on SM90 use
  `moe_runner_backend='marlin'` (§5) instead of deep_gemm.
- Indexer topk — `SGLANG_OPT_USE_TOPK_V2=0` to avoid the SM90 TMA-alignment
  assertion (§5).

Known remaining issue (NOT a setup/dependency problem):
- The RedKnot MLA path (`attention_backend='redknot_mla'`) runs but its decode
  output degenerates into repetition (e.g. `"J.AL, the answer is J.AL, ..."`,
  F1=0.00), while the dsv4 baseline is correct. This is the same class of
  RedKnot decode-quality regression already noted for Llama-3.3 in
  `RAG_BENCHMARK_RESULTS.md` and needs separate algorithm-side investigation in
  the RedKnot MLA backend / head-config, not in the environment.
