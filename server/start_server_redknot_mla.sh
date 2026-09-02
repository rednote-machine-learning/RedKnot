#!/bin/bash
# 启动 DeepSeek-V4-Flash-0731 + RedKnot MLA 集成路径 SGLang server
#
# 与 start_server_flashbase.sh 的区别:
#   --attention-backend redknot_mla : RedKnot 逐层、逐逻辑 head 的 MLA backend
#   --redknot-mla-pass-mode headwise: 每个 TP rank 只计算其拥有的 Q heads；默认
#                                     精度优先模式下 local 表示可离线复用，在线
#                                     脏行仍读完整 SWA + CSA/HCA。所有 heads 始终
#                                     共享一份位置无关 packed KV。
#
# correctness 模式默认保持 dense MoE；sparse FFN 是独立实验开关，不能把它的
# 节省计入本脚本的 MLA 多头解耦结果。
set -euo pipefail

ulimit -l unlimited 2>/dev/null || echo "warning: memlock ulimit unchanged" >&2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REDKNOT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
MODEL_PATH=${REDKNOT_MODEL_PATH:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash-0731}
if [[ "$MODEL_PATH" == "~/"* ]]; then
  MODEL_PATH="$HOME/${MODEL_PATH:2}"
fi
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$PWD/$MODEL_PATH"
fi
if [[ -n "${REDKNOT_HEAD_CFG:-}" ]]; then
  if [[ "$REDKNOT_HEAD_CFG" == "~/"* ]]; then
    export REDKNOT_HEAD_CFG="$HOME/${REDKNOT_HEAD_CFG:2}"
  elif [[ "$REDKNOT_HEAD_CFG" != /* ]]; then
    export REDKNOT_HEAD_CFG="$PWD/$REDKNOT_HEAD_CFG"
  fi
fi
if [[ -n "${REDKNOT_SERVER_POLICY_MANIFEST_OUT:-}" ]]; then
  if [[ "$REDKNOT_SERVER_POLICY_MANIFEST_OUT" == "~/"* ]]; then
    export REDKNOT_SERVER_POLICY_MANIFEST_OUT="$HOME/${REDKNOT_SERVER_POLICY_MANIFEST_OUT:2}"
  elif [[ "$REDKNOT_SERVER_POLICY_MANIFEST_OUT" != /* ]]; then
    export REDKNOT_SERVER_POLICY_MANIFEST_OUT="$PWD/$REDKNOT_SERVER_POLICY_MANIFEST_OUT"
  fi
fi
PYTHON_BIN=${REDKNOT_PYTHON_BIN:-${REDKNOT_PYTHON:-$REDKNOT_ROOT/.venv_tf5/bin/python}}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3 || command -v python)
fi

test -r "$REDKNOT_ROOT/python/sglang/srt/layers/attention/redknot_mla_backend.py"
test -r "$MODEL_PATH/config.json"
test -r "$MODEL_PATH/model.safetensors.index.json"
test -r "$MODEL_PATH/tokenizer.json"
if [[ -n "${REDKNOT_HEAD_CFG:-}" ]]; then
  test -r "$REDKNOT_HEAD_CFG"
fi

export PYTHONPATH="$REDKNOT_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
echo "REDKNOT_RELEASE_SERVER root=$REDKNOT_ROOT model=$MODEL_PATH python=$PYTHON_BIN head_cfg=${REDKNOT_HEAD_CFG:-none}" >&2
# Preserve the certified H200 launch path byte-for-byte.  B300 needs two
# architecture-specific compatibility switches: Triton's bundled CUDA 12.8
# ptxas cannot assemble sm_103a, and sgl-kernel 0.3.20 has no sm_103 image for
# RMSNorm.  Both fallbacks are JIT implementations already shipped in this
# tree, and are enabled only after an explicit/automatic B300 classification.
REDKNOT_DETECTED_GPU_NAME="$(
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
    | sed -n '1p' \
    | tr -d '\r' || true
)"
REDKNOT_EFFECTIVE_HARDWARE_PROFILE="${REDKNOT_HARDWARE_PROFILE:-auto}"
if [[ "$REDKNOT_EFFECTIVE_HARDWARE_PROFILE" == "auto" ]]; then
  if [[ "${REDKNOT_DETECTED_GPU_NAME^^}" == *B300* ]]; then
    REDKNOT_EFFECTIVE_HARDWARE_PROFILE=b300
  else
    REDKNOT_EFFECTIVE_HARDWARE_PROFILE=h200
  fi
fi
case "$REDKNOT_EFFECTIVE_HARDWARE_PROFILE" in
  h200) ;;
  b300)
    if [[ -z "${TRITON_PTXAS_PATH:-}" && -x /usr/local/cuda/bin/ptxas ]]; then
      export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    fi
    export SGLANG_USE_JIT_RMSNORM="${SGLANG_USE_JIT_RMSNORM:-1}"
    # Torch's SM103 advanced-index read used by full->SWA translation has no
    # runnable image in the shared wheel.  Use the equivalent in-tree Triton
    # gather on B300; H200 keeps the original ATen indexing path.
    export SGLANG_USE_JIT_SWA_TRANSLATION="${SGLANG_USE_JIT_SWA_TRANSLATION:-1}"
    # Snapshot/restore reads and writes the packed 584-byte DeepSeek-V4 KV
    # records.  Its SM103-safe Triton gather/scatter is already implemented in
    # dsv4_rope_reloc; select it only for B300.
    export SGLANG_USE_JIT_PACKED_OFFSETS="${SGLANG_USE_JIT_PACKED_OFFSETS:-1}"
    # The shared PyTorch wheel does not ship every elementwise/broadcast kernel
    # variant used by the MHC post-combine for SM103.  The in-tree Triton MHC
    # implementation covers both pre and post paths without changing H200.
    export SGLANG_USE_JIT_MHC="${SGLANG_USE_JIT_MHC:-1}"
    # sgl-kernel's packaged per-token FP8 group-quant image does not include
    # SM103.  DeepSeek V4's activation quantization here uses neither fused
    # SiLU nor masked-M, so the repository's active-arch Triton path applies.
    export SGLANG_USE_JIT_GROUP_QUANT="${SGLANG_USE_JIT_GROUP_QUANT:-1}"
    # topk_v1's PDL/radix CUDA kernel performs an illegal access on SM103.
    # Select the repository's vectorized, numerically equivalent Top-K path on
    # B300.  H200 continues to use the original optimized CUDA implementation.
    export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-1}"
    # MXFP4 Marlin's packaged kernels are Hopper-specific. FlashInfer 0.5.3
    # ships a native SM103 FP4 fused-MoE generator, so make that the B300
    # default. Ignore a stale generic Marlin setting inherited from the H200
    # launcher; callers can select another verified B300 backend explicitly via
    # REDKNOT_B300_MOE_RUNNER_BACKEND.
    export REDKNOT_MOE_RUNNER_BACKEND="${REDKNOT_B300_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
    # FlashInfer 0.5.3 exposes autotune(tune_mode) but this SGLang checkout
    # calls the newer autotune(..., cache=...) API. Use the deterministic
    # default SM103 tactic until a cache-aware FlashInfer build is installed.
    export REDKNOT_DISABLE_FLASHINFER_AUTOTUNE="${REDKNOT_DISABLE_FLASHINFER_AUTOTUNE:-1}"
    # Triton 3.5's dual-scope split-K DSV4 attention specialization requires
    # 544 tensor-memory columns on SM103, whose architectural limit is 512.
    # The numerically equivalent non-split kernel is already SM103-capable.
    export REDKNOT_DSV4_DISABLE_DUAL_SCOPE_SPLITK="${REDKNOT_DSV4_DISABLE_DUAL_SCOPE_SPLITK:-1}"
    ;;
  *)
    echo "unsupported REDKNOT_HARDWARE_PROFILE=$REDKNOT_EFFECTIVE_HARDWARE_PROFILE (expected auto, h200 or b300)" >&2
    exit 64
    ;;
esac
export REDKNOT_HARDWARE_PROFILE="$REDKNOT_EFFECTIVE_HARDWARE_PROFILE"
echo "REDKNOT_HARDWARE profile=$REDKNOT_HARDWARE_PROFILE gpu=${REDKNOT_DETECTED_GPU_NAME:-unknown} moe_backend=${REDKNOT_MOE_RUNNER_BACKEND:-marlin} ptxas=${TRITON_PTXAS_PATH:-bundled} jit_rmsnorm=${SGLANG_USE_JIT_RMSNORM:-0} jit_swa_translation=${SGLANG_USE_JIT_SWA_TRANSLATION:-0} jit_packed_offsets=${SGLANG_USE_JIT_PACKED_OFFSETS:-0} jit_mhc=${SGLANG_USE_JIT_MHC:-0} jit_group_quant=${SGLANG_USE_JIT_GROUP_QUANT:-0} torch_topk512=${SGLANG_TOPK_TRANSFORM_512_TORCH:-0} disable_dual_scope_splitk=${REDKNOT_DSV4_DISABLE_DUAL_SCOPE_SPLITK:-0}" >&2
unset REDKNOT_DETECTED_GPU_NAME REDKNOT_EFFECTIVE_HARDWARE_PROFILE
# The repository and virtualenv live on a FUSE mount. Eight spawned TP workers
# writing the same import-time __pycache__ entries can serialize for minutes on
# rename locks; bytecode files are not needed by this long-lived server.
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BARE_SUBPROCESS_LAUNCH=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_RANK_LOG_DIR="${SGLANG_RANK_LOG_DIR:-/tmp/ranklogs_redknot_mla}"
mkdir -p "$SGLANG_RANK_LOG_DIR"

# MHC / fp8 优化路径。原注释称 "tilelang 未装, 走 mhc_fallback 纯 torch"，但
# 2026-08-21 核实 tilelang 与 deep_gemm 均已安装，该注释已过时。默认仍保持 0 以
# 保证与历史测量可比，但改为可被调用方覆盖，便于评估纯 torch 回退的成本。
export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-0}"
export SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST:-0}"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-0}"
export SGLANG_OPT_FP8_WO_A_GEMM="${SGLANG_OPT_FP8_WO_A_GEMM:-0}"
# Shared latent-KV restore needs independently skippable q_a and kv_a work.
# The fused wqkv_a projection materializes KV for every row even if the later
# cache writer is bypassed, which would turn the reported KV saving into fake
# accounting.  Keep the production MLA-off profile de-fused; native-oracle
# launches may override this back to 1 in their separate server process.
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" ]]; then
  export SGLANG_OPT_FUSE_WQA_WKV=0
  export SGLANG_OPT_USE_FUSED_QK_NORM_ROPE=0
fi

# DeepGEMM JIT: 按需 JIT
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS=16
export REDKNOT_V4_MODE="${REDKNOT_V4_MODE:-correctness}"
export REDKNOT_V4_SEGMENTED_COMPRESSOR="${REDKNOT_V4_SEGMENTED_COMPRESSOR:-1}"
# Experimental local-head output reuse. This remains off unless both this
# server switch and per-request capture_mla_off/reuse_mla_off are supplied.
# Restore is approximate and is rejected in REDKNOT_V4_MODE=correctness.
export REDKNOT_MLA_OFFLOAD="${REDKNOT_MLA_OFFLOAD:-0}"
export REDKNOT_MLA_OFF_EXECUTION_PROFILE="${REDKNOT_MLA_OFF_EXECUTION_PROFILE:-pure_headsplit_boundary128_3_37_3_v2}"
# The multi-head decomposition treats a profiled local head as SWA-local and a
# global head as native SWA+CSA/HCA.  The shared physical latent KV is never
# split by head.  Full-scope local heads are a different accuracy experiment
# and are intentionally rejected by the pure 3+37+3 artifact profile.
export REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE="${REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE:-0}"
# Persist positionless SWA/C4/C128/Indexer/state artifacts on each TP rank.
# Capacity is measured in complete segment generations; eviction/pinning is
# generation-atomic, so no request can see a partially replaced bank slot.
export REDKNOT_SHARED_LATENT_GPU="${REDKNOT_SHARED_LATENT_GPU:-1}"
export REDKNOT_SHARED_LATENT_MAX_SEGMENT_EPOCHS="${REDKNOT_SHARED_LATENT_MAX_SEGMENT_EPOCHS:-16}"
# Fail closed unless the caller supplies a context length that has passed the
# paired dense/reuse quality gate.  Zero means no request is certified for
# offline MLA-output reuse; snapshots and ordinary native attention still work.
export REDKNOT_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS="${REDKNOT_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS:-0}"
# Per TP worker. BF16 v1 stores [tokens, local_output_groups, o_lora_rank]
# for every local-bearing layer and evicts complete segments under this cap.
export REDKNOT_MLA_OFF_MAX_BYTES="${REDKNOT_MLA_OFF_MAX_BYTES:-8589934592}"
# Optional benchmark handshake. Rank 0 writes the manifest only after the
# effective backend policy has initialized and passed TP startup consensus.
export REDKNOT_SERVER_INSTANCE_NONCE="${REDKNOT_SERVER_INSTANCE_NONCE:-${REDKNOT_IH_SERVER_INSTANCE_NONCE:-}}"
# The selected-row segmented path has not yet been ported to compressor_v2.
# Refuse silent use of its default-on implementation in this experimental server.
export SGLANG_OPT_USE_COMPRESSOR_V2="${SGLANG_OPT_USE_COMPRESSOR_V2:-0}"
# Checkpoint-island restore needs one independently addressable C128 state per
# checkpoint. The online-C128 layout intentionally collapses those state slots;
# model_runner also fails closed if a caller explicitly overrides this to 1.
export SGLANG_OPT_USE_ONLINE_COMPRESS="${SGLANG_OPT_USE_ONLINE_COMPRESS:-0}"

PORT="${PORT:-31998}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
WATCHDOG_TIMEOUT="${REDKNOT_WATCHDOG_TIMEOUT:-1800}"
RANDOM_SEED="${REDKNOT_RANDOM_SEED:-2026}"
MOE_RUNNER_BACKEND="${REDKNOT_MOE_RUNNER_BACKEND:-marlin}"
MEM_FRACTION_STATIC="${REDKNOT_MEM_FRACTION_STATIC:-0.80}"
MLA_PASS_MODE="${REDKNOT_MLA_PASS_MODE:-headwise}"
TP_SIZE="${REDKNOT_TP_SIZE:-8}"
MLA_DENSE_PREFIX="${REDKNOT_MLA_DENSE_PREFIX_LAYERS:-3}"
MLA_DENSE_SUFFIX="${REDKNOT_MLA_DENSE_SUFFIX_LAYERS:-3}"
MLA_LOCAL_WINDOW="${REDKNOT_MLA_LOCAL_WINDOW:-128}"
MLA_GLOBAL_HEAD_STRIDE="${REDKNOT_MLA_GLOBAL_HEAD_STRIDE:-8}"
MLA_GLOBAL_LAYER_STRIDE="${REDKNOT_MLA_GLOBAL_LAYER_STRIDE:-0}"
RADIX_EVICTION_POLICY="${REDKNOT_RADIX_EVICTION_POLICY:-lru}"
case "$RADIX_EVICTION_POLICY" in
  lru|lfu|slru|priority) ;;
  *)
    echo "unsupported radix eviction policy: $RADIX_EVICTION_POLICY" >&2
    exit 2
    ;;
esac
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" && "$MLA_PASS_MODE" != "headwise" ]]; then
  echo "MLA-off requires REDKNOT_MLA_PASS_MODE=headwise" >&2
  exit 2
fi
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" && ( "$MLA_DENSE_PREFIX" != "3" || "$MLA_DENSE_SUFFIX" != "3" ) ]]; then
  echo "MLA-off profile requires exactly 3 dense prefix and 3 dense suffix layers" >&2
  exit 2
fi
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" && "$MLA_GLOBAL_LAYER_STRIDE" != "0" ]]; then
  echo "MLA-off requires every middle layer 3..39 to remain head-split" >&2
  exit 2
fi
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" ]]; then
  case "$REDKNOT_MLA_OFF_EXECUTION_PROFILE" in
    pure_headsplit_boundary128_3_37_3_v2)
      if [[ "$REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE" != "0" ]]; then
        echo "legacy v2 MLA-off requires profiled local-window heads" >&2
        exit 2
      fi
      ;;
    pure_headsplit_context_bound_fullscope_3_37_3_v1|\
    pure_headsplit_independent_rope_relocation_fullscope_boundary128_3_37_3_v1|\
    combined_headsplit_independent_rope_zoff_checkpoint_rowsparse_3_37_3_v1)
      if [[ "$REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE" != "1" ]]; then
        echo "full-scope MLA-off requires native full-scope reusable heads" >&2
        exit 2
      fi
      ;;
    *)
      echo "unsupported MLA-off execution profile: $REDKNOT_MLA_OFF_EXECUTION_PROFILE" >&2
      exit 2
      ;;
  esac
fi
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" && "${REDKNOT_MLA_OFF_COMPACT_WOA:-0}" != "0" ]]; then
  echo "MLA-off forbids the legacy selected-row compact wo_a path" >&2
  exit 2
fi
EXTRA_SERVER_ARGS=()
EXTRA_SERVER_ARGS+=(--radix-eviction-policy "$RADIX_EVICTION_POLICY")
if [[ -n "${REDKNOT_SWA_FULL_TOKENS_RATIO:-}" ]]; then
  case "$REDKNOT_SWA_FULL_TOKENS_RATIO" in
    0.1|0.10|0.125|0.13|0.20|0.25|0.30|0.40|0.5|0.50) ;;
    *)
      echo "unsupported REDKNOT_SWA_FULL_TOKENS_RATIO: $REDKNOT_SWA_FULL_TOKENS_RATIO" >&2
      exit 2
      ;;
  esac
  EXTRA_SERVER_ARGS+=(
    --swa-full-tokens-ratio "$REDKNOT_SWA_FULL_TOKENS_RATIO"
  )
fi
if [[ "${REDKNOT_DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--disable-radix-cache)
fi
if [[ "${REDKNOT_DETERMINISTIC_INFERENCE:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--enable-deterministic-inference)
fi
if [[ "${REDKNOT_DISABLE_OVERLAP_SCHEDULE:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--disable-overlap-schedule)
fi
if [[ "${REDKNOT_ENABLE_METRICS:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--enable-metrics)
fi
if [[ -n "${REDKNOT_HEAD_CFG:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--redknot-head-config-path "$REDKNOT_HEAD_CFG")
fi
# Progressive Assignment-Sparse MoE (per-layer routed top-K). Independent of
# REDKNOT_SPARSE_FFN: no token is dropped, so it does not trip the MLA-off vs
# token-sparse-FFN mutual exclusion in redknot_mla_backend and can be measured
# together with the MLA head-split path.
if [[ -n "${REDKNOT_PROGRESSIVE_TOPK_SCHEDULE:-}" ]]; then
  EXTRA_SERVER_ARGS+=(
    --redknot-progressive-topk-schedule "$REDKNOT_PROGRESSIVE_TOPK_SCHEDULE"
  )
fi
if [[ "${REDKNOT_DISABLE_FLASHINFER_AUTOTUNE:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--disable-flashinfer-autotune)
fi
if [[ "${REDKNOT_SPARSE_FFN:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(
    --redknot-sparse-ffn-enable
    --redknot-sparse-ffn-dense-until "${REDKNOT_FFN_DENSE_UNTIL:-4}"
    --redknot-sparse-ffn-deep-start "${REDKNOT_FFN_DEEP_START:-24}"
    --redknot-sparse-ffn-mass-thresh "${REDKNOT_FFN_MASS:-0.60}"
    --redknot-sparse-ffn-mass-thresh-deep "${REDKNOT_FFN_MASS_DEEP:-0.30}"
    --redknot-sparse-ffn-recent-n "${REDKNOT_FFN_RECENT_N:-256}"
    --redknot-sparse-ffn-min-seq-len "${REDKNOT_FFN_MIN_SEQ_LEN:-16384}"
    --redknot-sparse-ffn-importance "${REDKNOT_FFN_IMPORTANCE:-activation}"
    --redknot-sparse-ffn-min-full-ratio "${REDKNOT_FFN_MIN_FULL_RATIO:-0.20}"
    --redknot-sparse-ffn-max-full-ratio "${REDKNOT_FFN_MAX_FULL_RATIO:-0.80}"
  )
  # HARD REQUIREMENT: _select_redknot_sparse_ffn_tokens returns None when
  # mlp.num_fused_shared_experts != 0 (deepseek_v4.py:3172). DSV4 has
  # n_shared_experts=1 and fuses it by default, so without this flag the whole
  # sparse-FFN path is a silent no-op.
  EXTRA_SERVER_ARGS+=(--disable-shared-experts-fusion)
fi

cd "$REDKNOT_ROOT"
exec "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend redknot_mla \
  --redknot-mla-pass-mode "$MLA_PASS_MODE" \
  --redknot-mla-dense-prefix-layers "$MLA_DENSE_PREFIX" \
  --redknot-mla-dense-suffix-layers "$MLA_DENSE_SUFFIX" \
  --redknot-mla-local-window "$MLA_LOCAL_WINDOW" \
  --redknot-mla-global-head-stride "$MLA_GLOBAL_HEAD_STRIDE" \
  --redknot-mla-global-layer-stride "$MLA_GLOBAL_LAYER_STRIDE" \
  --tp-size "$TP_SIZE" \
  --moe-runner-backend "$MOE_RUNNER_BACKEND" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --disable-cuda-graph \
  --skip-server-warmup \
  --random-seed "$RANDOM_SEED" \
  --watchdog-timeout "$WATCHDOG_TIMEOUT" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --trust-remote-code \
  "${EXTRA_SERVER_ARGS[@]}" \
  --port "$PORT"
