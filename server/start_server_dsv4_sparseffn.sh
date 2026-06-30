#!/bin/bash
# 启动 DeepSeek-V4-Flash-Base (275GB) SGLang server
# experts: 真 fp8 (F8_E4M3) —— 满足 "用 fp8 不用 fp4" 的要求 (Pro/Flash 的 experts 是 mxfp4)
# 设备: 8x GPU (PyTorch 检测 sm_90 / Hopper), 8x81GB=648GB > 275GB, 无需 CPU offload
# attention backend: dsv4 (原生 MLA + indexer 稀疏, 依赖 flash_mla sm_90a)
set -euo pipefail

ulimit -l unlimited

REDKNOT_ROOT=/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2
MODEL_PATH=/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash-Base

export PYTHONPATH=$REDKNOT_ROOT/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BARE_SUBPROCESS_LAUNCH=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_RANK_LOG_DIR=/tmp/ranklogs_dsv4_sff
mkdir -p "$SGLANG_RANK_LOG_DIR"

# MHC / fp8 优化路径回退 (tilelang 未装, 走 mhc_fallback 纯 torch)
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FP8_WO_A_GEMM=0

# DeepGEMM JIT: 关闭全量 precompile, 改按需 JIT (避免首请求长时间卡 warmup)
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS=16

# Flash/Flash-Base/Pro 的 routed-expert 物理布局自动探测; Flash-Base 是 F8_E4M3,
# 显式声明 fp8 (=0) 走 fp8 expert layout, 避免任何 mxfp4 误判.
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_REDKNOT_FFN_DEBUG=1

cd "$REDKNOT_ROOT"
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend dsv4 \
  --redknot-sparse-ffn-enable \
  --redknot-sparse-ffn-importance ${REDKNOT_FFN_IMPORTANCE:-indexer} \
  --redknot-sparse-ffn-dense-until 4 \
  --redknot-sparse-ffn-mass-thresh 0.6 \
  --redknot-sparse-ffn-recent-n 256 \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --max-total-tokens 32768 \
  --trust-remote-code \
  --port ${PORT:-31998}
