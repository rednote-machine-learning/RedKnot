#!/bin/bash
# 启动 DeepSeek-V4-Pro-0813 + RedKnot MLA 独立适配路径
# 认证目标: 8x NVIDIA B300 / Blackwell SM103 / TP8
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
MODEL_PATH=${REDKNOT_MODEL_PATH:-/workspace/Models/DeepSeek-V4-Pro-0813}
if [[ "$MODEL_PATH" == "~/"* ]]; then
  MODEL_PATH="$HOME/${MODEL_PATH:2}"
fi
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$PWD/$MODEL_PATH"
fi
if [[ "$MODEL_PATH" != "/workspace/Models/DeepSeek-V4-Pro-0813" ]]; then
  echo "Pro-0813 certified launcher requires the pinned official model path: $MODEL_PATH" >&2
  exit 2
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
PYTHON_BIN=/workspace/RedKnot/.venv_sm103/bin/python
if [[ -n "${REDKNOT_PYTHON_BIN:-}" \
      && "$REDKNOT_PYTHON_BIN" != "$PYTHON_BIN" ]]; then
  echo "Pro-0813 forbids a non-certified REDKNOT_PYTHON_BIN: $REDKNOT_PYTHON_BIN" >&2
  exit 2
fi
if [[ -n "${REDKNOT_PYTHON:-}" \
      && "$REDKNOT_PYTHON" != "$PYTHON_BIN" ]]; then
  echo "Pro-0813 forbids a non-certified REDKNOT_PYTHON: $REDKNOT_PYTHON" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "certified Pro-0813 runtime Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

# Never extend an ambient PYTHONPATH here.  In particular, the certified
# virtualenv is shared with the Flash reproduction and callers commonly have
# /workspace/RedKnot/python in their login environment.  Appending that path
# lets missing Pro modules fall through to an unaudited Flash checkout.  The
# Pro server is allowed exactly two source roots: this physical Pro checkout
# first, and the separately certified SM103 FlashMLA build second.  Pin these
# before the first Python subprocess, including the CPU-only model gate.
FLASHMLA_SM103_ROOT="${REDKNOT_FLASHMLA_SM103_ROOT:-/data/temp/FlashMLA-sm103-src}"
if [[ "$FLASHMLA_SM103_ROOT" != /* || "$FLASHMLA_SM103_ROOT" == *:* ]]; then
  echo "REDKNOT_FLASHMLA_SM103_ROOT must be an absolute colon-free path: $FLASHMLA_SM103_ROOT" >&2
  exit 2
fi
if ! FLASHMLA_SM103_ROOT=$(cd "$FLASHMLA_SM103_ROOT" 2>/dev/null && pwd -P); then
  echo "required SM103 FlashMLA source root is unavailable: $FLASHMLA_SM103_ROOT" >&2
  exit 2
fi
test -d "$FLASHMLA_SM103_ROOT/flash_mla"
export REDKNOT_FLASHMLA_SM103_ROOT="$FLASHMLA_SM103_ROOT"
export PYTHONPATH="$REDKNOT_ROOT/python:$FLASHMLA_SM103_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

test -r "$REDKNOT_ROOT/python/sglang/srt/layers/attention/redknot_mla_backend.py"
test -r "$REDKNOT_ROOT/test/srt/redknot/utils/verify_pro0813_official_model.py"
test -r "$REDKNOT_ROOT/test/srt/redknot/utils/probe_pro0813_jit_rmsnorm_sm103.py"
test -r "$REDKNOT_ROOT/test/srt/redknot/utils/probe_pro0813_triton_h1_sm103.py"
if ! "$PYTHON_BIN" \
  "$REDKNOT_ROOT/test/srt/redknot/utils/verify_pro0813_official_model.py"; then
  echo "official Pro-0813 model is incomplete or does not match the immutable manifest" >&2
  exit 2
fi
test -r "$MODEL_PATH/config.json"
test -r "$MODEL_PATH/model.safetensors.index.json"
test -r "$MODEL_PATH/tokenizer.json"
if [[ -n "${REDKNOT_HEAD_CFG:-}" ]]; then
  test -r "$REDKNOT_HEAD_CFG"
fi

echo "REDKNOT_PRO0813_SERVER root=$REDKNOT_ROOT model=$MODEL_PATH python=$PYTHON_BIN head_cfg=${REDKNOT_HEAD_CFG:-none}" >&2
# The repository and virtualenv live on a FUSE mount. Eight spawned TP workers
# writing the same import-time __pycache__ entries can serialize for minutes on
# rename locks; bytecode files are not needed by this long-lived server.
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BARE_SUBPROCESS_LAUNCH=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ -n "${SGLANG_USE_JIT_RMSNORM:-}" \
      && "$SGLANG_USE_JIT_RMSNORM" != "1" ]]; then
  echo "B300/SM103 Pro-0813 requires SGLANG_USE_JIT_RMSNORM=1" >&2
  exit 2
fi
# sgl_kernel 0.3.20 in the certified environment has no SM103 cubin/PTX for
# rmsnorm.  The SGLang JIT implementation is numerically certified below for
# Pro's 512/1536/7168 widths and is mandatory for both ordinary and residual
# fused RMSNorm.
export SGLANG_USE_JIT_RMSNORM=1
# FlashInfer 0.5.3 only adds the Blackwell feature suffix during automatic
# detection. An explicit bare 10.3 generates sm_103 and fails on the MXFP8
# E2M1 conversion instruction; the B300 target must be sm_103a.
if [[ -n "${FLASHINFER_CUDA_ARCH_LIST:-}" \
      && "$FLASHINFER_CUDA_ARCH_LIST" != "10.3a" ]]; then
  echo "B300/SM103 Pro-0813 requires FLASHINFER_CUDA_ARCH_LIST=10.3a" >&2
  exit 2
fi
export FLASHINFER_CUDA_ARCH_LIST=10.3a
export SGLANG_RANK_LOG_DIR="${SGLANG_RANK_LOG_DIR:-/tmp/ranklogs_redknot_pro0813}"
mkdir -p "$SGLANG_RANK_LOG_DIR"
# The shared environment contains older pip NVRTC/nvJitLink libraries (12.8),
# while this server uses CUDA/PyTorch 12.9.  PyTorch's DT_RPATH otherwise wins
# over LD_LIBRARY_PATH and makes its Jiterator pass compute_103 to NVRTC 12.8,
# which rejects that B300 architecture.  Preload the matching system libraries
# (NVRTC first for symbol interposition) and expose the installed NVSHMEM host
# library before importing torch in any TP worker.
NVRTC_LIB=/usr/local/cuda/lib64/libnvrtc.so.12
NVJITLINK_LIB=/usr/local/cuda/lib64/libnvJitLink.so.12
NVSHMEM_LIB_DIR=/root/miniconda3/lib/python3.11/site-packages/nvidia/nvshmem/lib
if [[ ! -r "$NVRTC_LIB" ]]; then
  echo "required CUDA-12.9 NVRTC is not readable: $NVRTC_LIB" >&2
  exit 2
fi
if [[ ! -r "$NVJITLINK_LIB" ]]; then
  echo "required CUDA-12.9 nvJitLink is not readable: $NVJITLINK_LIB" >&2
  exit 2
fi
if [[ ! -d "$NVSHMEM_LIB_DIR" ]] \
   || ! compgen -G "$NVSHMEM_LIB_DIR/libnvshmem*.so*" >/dev/null; then
  echo "required NVSHMEM library directory is invalid: $NVSHMEM_LIB_DIR" >&2
  exit 2
fi
export LD_PRELOAD="$NVRTC_LIB:$NVJITLINK_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$NVSHMEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Triton 3.x in the shared environment bundles CUDA 12.8 ptxas, which cannot
# assemble the sm_103a target selected for B300.  Use the system CUDA 12.9
# assembler; validate the target before any model worker is launched.
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
test -x "$TRITON_PTXAS_PATH"
if ! "$TRITON_PTXAS_PATH" --help 2>&1 | awk '
  /sm_103a/ { found = 1 }
  END { exit(found ? 0 : 1) }
'; then
  echo "configured Triton ptxas does not support B300 sm_103a" >&2
  exit 2
fi
# Keep the Flash baseline venv untouched.  Pro uses the separately built
# CUDA-12.9 SM100-family binary: NVIDIA's family code object covers both
# Blackwell CC 10.0 and B300 CC 10.3, whereas the venv default is SM90a-only.
if ! PRO0813_SOURCE_AUDIT=$(CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -c '
import importlib.util
import pathlib
import sys

for package, root_arg in (("sglang", sys.argv[1]), ("flash_mla", sys.argv[2])):
    root = pathlib.Path(root_arg).resolve()
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise SystemExit(f"{package} has no importable regular package origin")
    paths = [pathlib.Path(spec.origin).resolve()]
    paths.extend(
        pathlib.Path(item).resolve()
        for item in (spec.submodule_search_locations or ())
    )
    if not paths or any(not path.is_relative_to(root) for path in paths):
        rendered = ",".join(str(path) for path in paths)
        raise SystemExit(
            f"{package} escaped certified source root {root}: {rendered}"
        )
    print(f"{package}={paths[0]}")
' "$REDKNOT_ROOT/python/sglang" "$FLASHMLA_SM103_ROOT/flash_mla"); then
  echo "Pro-0813 Python source provenance audit failed" >&2
  exit 2
fi
printf 'REDKNOT_PRO0813_SOURCE_AUDIT %s\n' \
  "$(printf '%s' "$PRO0813_SOURCE_AUDIT" | tr '\n' ' ')" >&2
FLASHMLA_CUDA_SO=$(CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -c 'import importlib.util; spec=importlib.util.find_spec("flash_mla.cuda"); print(spec.origin if spec else "")')
case "$FLASHMLA_CUDA_SO" in
  "$FLASHMLA_SM103_ROOT"/*) ;;
  *)
    echo "Pro-0813 resolved an unisolated FlashMLA binary: $FLASHMLA_CUDA_SO" >&2
    exit 2
    ;;
esac
CUOBJDUMP_BIN=${REDKNOT_CUOBJDUMP_BIN:-/usr/local/cuda/bin/cuobjdump}
if [[ "$CUOBJDUMP_BIN" != /* || ! -x "$CUOBJDUMP_BIN" ]]; then
  echo "REDKNOT_CUOBJDUMP_BIN must be an executable absolute path: $CUOBJDUMP_BIN" >&2
  exit 2
fi
if ! "$CUOBJDUMP_BIN" --dump-elf "$FLASHMLA_CUDA_SO" 2>/dev/null | awk '
  /arch = sm_100f/ { found = 1 }
  END { exit(found ? 0 : 1) }
'; then
  echo "Pro-0813 FlashMLA binary has no SM100-family code object for B300" >&2
  exit 2
fi

if ! accelerator_rows=$(nvidia-smi \
  --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null); then
  echo "cannot query the B300 accelerator contract" >&2
  exit 2
fi
if ! awk -F',' '
  BEGIN { count = 0; bad = 0 }
  {
    name = $1
    capability = $2
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", capability)
    count += 1
    if (name !~ /B300/ || capability != "10.3") bad = 1
  }
  END { exit(count == 8 && !bad ? 0 : 1) }
' <<<"$accelerator_rows"; then
  echo "Pro-0813 requires exactly 8x NVIDIA B300 compute_cap=10.3" >&2
  exit 2
fi
if ! CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  "$REDKNOT_ROOT/test/srt/redknot/utils/probe_pro0813_jit_rmsnorm_sm103.py"; then
  echo "B300/SM103 JIT RMSNorm numerical oracle failed" >&2
  exit 2
fi
if ! CUDA_VISIBLE_DEVICES=0 REDKNOT_PRO0813_B300_TRITON_H1_PROBE=1 \
  "$PYTHON_BIN" \
  "$REDKNOT_ROOT/test/srt/redknot/utils/probe_pro0813_triton_h1_sm103.py" \
  --expected-source-root "$REDKNOT_ROOT/python"; then
  echo "B300/SM103 Triton H1 numerical oracle failed" >&2
  exit 2
fi

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
export REDKNOT_DSV4_VARIANT=pro0813
export REDKNOT_V4_SEGMENTED_COMPRESSOR="${REDKNOT_V4_SEGMENTED_COMPRESSOR:-1}"
# Experimental local-head output reuse. This remains off unless both this
# server switch and per-request capture_mla_off/reuse_mla_off are supplied.
# Restore is approximate and is rejected in REDKNOT_V4_MODE=correctness.
export REDKNOT_MLA_OFFLOAD="${REDKNOT_MLA_OFFLOAD:-0}"
export REDKNOT_MLA_OFF_EXECUTION_PROFILE="${REDKNOT_MLA_OFF_EXECUTION_PROFILE:-pure_headsplit_pro0813_context_bound_fullscope_3_55_3_v1}"
# The multi-head decomposition treats a profiled local head as SWA-local and a
# global head as native SWA+CSA/HCA.  The shared physical latent KV is never
# split by head.  Full-scope local heads are a different accuracy experiment
# and are intentionally rejected by the Pro-0813 pure 3+55+3 artifact profile.
export REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE="${REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE:-1}"
# B300 is SM103. The padded H64 provider is certified only for SM90/Hopper;
# Pro-0813 must use the arbitrary-head Triton provider until a separate SM103
# certificate says otherwise.
export REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL="${REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL:-triton_h1}"
export REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH=0
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
export REDKNOT_MLA_OFF_MAX_BYTES="${REDKNOT_MLA_OFF_MAX_BYTES:-17179869184}"
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

if [[ -n "${PORT:-}" && -n "${REDKNOT_PRO0813_SERVER_PORT:-}" \
      && "$PORT" != "$REDKNOT_PRO0813_SERVER_PORT" ]]; then
  echo "PORT and REDKNOT_PRO0813_SERVER_PORT disagree" >&2
  exit 2
fi
PORT="${PORT:-${REDKNOT_PRO0813_SERVER_PORT:-31998}}"
if ! [[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( 10#$PORT > 65535 )); then
  echo "Pro-0813 server port must be in [1, 65535]: $PORT" >&2
  exit 2
fi
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
WATCHDOG_TIMEOUT="${REDKNOT_WATCHDOG_TIMEOUT:-1800}"
RANDOM_SEED="${REDKNOT_RANDOM_SEED:-2026}"
MOE_RUNNER_BACKEND="${REDKNOT_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
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
if [[ "$TP_SIZE" != "8" ]]; then
  echo "DeepSeek-V4-Pro-0813 RedKnot certification currently requires TP=8" >&2
  exit 2
fi
if [[ "$REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL" != "triton_h1" ]]; then
  echo "B300/SM103 Pro-0813 requires REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL=triton_h1" >&2
  exit 2
fi
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" && ( "$MLA_DENSE_PREFIX" != "3" || "$MLA_DENSE_SUFFIX" != "3" ) ]]; then
  echo "MLA-off profile requires exactly 3 dense prefix and 3 dense suffix layers" >&2
  exit 2
fi
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" && "$MLA_GLOBAL_LAYER_STRIDE" != "0" ]]; then
  echo "MLA-off requires every middle layer 3..57 to remain head-split" >&2
  exit 2
fi
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" ]]; then
  case "$REDKNOT_MLA_OFF_EXECUTION_PROFILE" in
    pure_headsplit_pro0813_context_bound_fullscope_3_55_3_v1|\
    pure_headsplit_pro0813_independent_rope_relocation_fullscope_boundary128_3_55_3_v1|\
    combined_headsplit_pro0813_independent_rope_zoff_checkpoint_rowsparse_3_55_3_v1|\
    combined_headsplit_pro0813_independent_rope_full_checkpoint_rowsparse_3_55_3_v1)
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
"$PYTHON_BIN" -c 'import json,sys; from sglang.srt.layers.attention.redknot.pro0813.profile import inspect_pro0813_config; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(inspect_pro0813_config(cfg, tp_size=int(sys.argv[2])).audit_dict())' "$MODEL_PATH/config.json" "$TP_SIZE"
if [[ "${REDKNOT_MLA_OFFLOAD:-0}" == "1" && "${REDKNOT_MLA_OFF_COMPACT_WOA:-0}" != "0" ]]; then
  echo "MLA-off forbids the legacy selected-row compact wo_a path" >&2
  exit 2
fi
EXTRA_SERVER_ARGS=()
EXTRA_SERVER_ARGS+=(--radix-eviction-policy "$RADIX_EVICTION_POLICY")
if [[ -n "${REDKNOT_SWA_FULL_TOKENS_RATIO:-}" ]]; then
  case "$REDKNOT_SWA_FULL_TOKENS_RATIO" in
    0.1|0.10|0.125|0.13|0.20|0.25|0.30|0.40|0.50) ;;
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
    --redknot-sparse-ffn-deep-start "${REDKNOT_FFN_DEEP_START:-34}"
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
