#!/bin/bash
# 启动 DeepSeek-V4-Flash (149GB) SGLang server —— mxfp4 experts 路径验证
# experts: mxfp4 (weight=I8, scale=F8_E8M0)
# MoE runner: flashinfer_mxfp4 -> sm_90 走 FlashInfer cutlass mixed-input MoE (PR #3084)
#   (绕开默认 deep_gemm runner 的 recipe_a/recipe_b 版本不兼容问题)
# 设备: 8x GPU (sm_90), 149GB < 648GB, 无需 CPU offload
set -euo pipefail

ulimit -l unlimited

REDKNOT_ROOT=/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2
MODEL_PATH=/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash

export PYTHONPATH=$REDKNOT_ROOT/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BARE_SUBPROCESS_LAUNCH=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_RANK_LOG_DIR=/tmp/ranklogs_flash
mkdir -p "$SGLANG_RANK_LOG_DIR"

# MHC / fp8 优化路径回退 (tilelang 未装)
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FP8_WO_A_GEMM=0

# DeepGEMM JIT 按需 (attention 等仍可能用到 deep_gemm)
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS=16

# 注意: 不设 SGLANG_DSV4_FP4_EXPERTS, 让其自动探测为 mxfp4(is_fp4_experts=True),
# 才会走 mxfp4 expert 路径并由 flashinfer_mxfp4 runner 处理.

cd "$REDKNOT_ROOT"
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --max-total-tokens 32768 \
  --trust-remote-code \
  --port 31999
