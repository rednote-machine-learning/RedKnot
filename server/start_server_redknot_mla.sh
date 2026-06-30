#!/bin/bash
# 启动 DeepSeek-V4-Flash-Base + RedKnot MLA 集成路径 SGLang server
#
# 与 start_server_flashbase.sh 的区别:
#   --attention-backend redknot_mla   : RedKnot 逻辑 head 分化 (local/global) 的 MLA backend
#   --redknot-sparse-ffn-enable       : 开启 token-selective sparse FFN
#   --redknot-sparse-ffn-importance indexer
#                                     : 用 DeepSeek-V4 自带 indexer 信号
#                                       (c4_topk_lengths_raw) 驱动 sparse-FFN token 选择,
#                                       而不是事后 activation L2 范数 (本次集成的核心改动)
#
# 即"三刀稀疏同源于 indexer": token(attention 选块) / head(local 跳远程块) / MoE(sparse FFN)。
set -euo pipefail

ulimit -l unlimited

REDKNOT_ROOT=/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2
MODEL_PATH=/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash-Base

export PYTHONPATH=$REDKNOT_ROOT/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BARE_SUBPROCESS_LAUNCH=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_RANK_LOG_DIR=/tmp/ranklogs_redknot_mla
mkdir -p "$SGLANG_RANK_LOG_DIR"

# MHC / fp8 优化路径回退 (tilelang 未装, 走 mhc_fallback 纯 torch)
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FP8_WO_A_GEMM=0

# DeepGEMM JIT: 按需 JIT
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS=16

# Flash-Base 是 F8_E4M3, 显式声明 fp8 expert layout
export SGLANG_DSV4_FP4_EXPERTS=0

PORT="${PORT:-31998}"

cd "$REDKNOT_ROOT"
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend redknot_mla \
  --redknot-sparse-ffn-enable \
  --redknot-sparse-ffn-importance "${REDKNOT_FFN_IMPORTANCE:-indexer}" \
  --redknot-sparse-ffn-dense-until 4 \
  --redknot-sparse-ffn-mass-thresh 0.6 \
  --redknot-sparse-ffn-recent-n 256 \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --max-total-tokens 32768 \
  --trust-remote-code \
  --port "$PORT"
