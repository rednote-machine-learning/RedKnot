#!/bin/bash
# 启动 DeepSeek-V4-Pro (806GB, fp8 e4m3 blockwise 128x128) SGLang server
# 设备: 8x GPU (PyTorch 检测 compute_cap=sm_90 / Hopper)
# attention backend: dsv4 (原生 MLA + indexer 稀疏, 依赖 flash_mla sm_90a)
# 量化: fp8 (从 config.json 的 quant_method=fp8 自动读取, 不使用 fp4)
set -euo pipefail

ulimit -l unlimited

REDKNOT_ROOT=/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2
# 用 zhongming 那份完整的 Pro (opensource 那份 shard 被误删过)
MODEL_PATH=/mnt/tidal-alsh01/dataset/redone/zhongming/checkpoints/DeepSeek-V4-Pro

export PYTHONPATH=$REDKNOT_ROOT/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BARE_SUBPROCESS_LAUNCH=1          # 绕过 numa/memsaver 子进程 wrapper (关键)
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_RANK_LOG_DIR=/tmp/ranklogs_pro    # per-rank 日志, 便于抓子进程崩溃
mkdir -p "$SGLANG_RANK_LOG_DIR"

# --- MHC / fp8 优化路径回退 (tilelang/deep_gemm 未装时走 fallback) ---
# DeepSeek-V4 的 MHC kernel 默认走 tilelang, kernel_warmup 会强制 import tilelang.
# 当前环境未装 tilelang, 故禁用 tilelang/deepgemm 路径, 走 mhc_fallback (纯 torch).
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FP8_WO_A_GEMM=0

# --- DeepGEMM JIT 预编译: 关闭全量 precompile, 改为推理时按需 JIT ---
# 默认 PRECOMPILE=1 会对 16384 个 m-shape x 多种 kernel type 逐个 JIT 编译,
# 在本环境极慢(数十分钟且反复触发), 导致首请求长时间无响应.
# 关闭后只编译实际用到的 shape, 首请求快很多.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS=16

cd "$REDKNOT_ROOT"
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend dsv4 \
  --moe-runner-backend deep_gemm \
  --tp-size 8 \
  --cpu-offload-gb 250 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --max-total-tokens 32768 \
  --trust-remote-code \
  --port 31999
