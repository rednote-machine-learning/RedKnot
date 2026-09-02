#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$script_dir/../../.." && pwd -P)
utils_dir="$script_dir/utils"
lock_file=${REDKNOT_RUN_LOCK:-/tmp/redknot_deepseek_v4_flash_release.lock}

hardware_profile=${REDKNOT_HARDWARE_PROFILE:-auto}
if [[ "$hardware_profile" == "auto" ]]; then
  gpu_name="$(
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
      | sed -n '1p' \
      | tr -d '\r' || true
  )"
  if [[ "${gpu_name^^}" == *B300* ]]; then
    hardware_profile=b300
  else
    hardware_profile=h200
  fi
fi
case "$hardware_profile" in
  h200) default_venv="$repo/.venv_tf5" ;;
  b300) default_venv="$repo/.venv_sm103" ;;
  *)
    printf 'ERROR: unsupported REDKNOT_HARDWARE_PROFILE=%s\n' "$hardware_profile" >&2
    exit 64
    ;;
esac
venv=${REDKNOT_VENV:-$default_venv}
export REDKNOT_HARDWARE_PROFILE="$hardware_profile"
export REDKNOT_VENV="$venv"

exec 9>"$lock_file"
if ! flock -n 9; then
  printf 'ERROR: another RedKnot release wrapper already owns %s\n' \
    "$lock_file" >&2
  printf '%s\n' \
    'Wait for it to finish; the eight GPUs cannot run two TP8 instances.' >&2
  exit 75
fi

active_release_pids=$(pgrep -f '[b]enchmark_RedKnot_DeepSeekV4Flash.py' || true)
active_worker_pids=$(pgrep -f '[b]enchmark_dsv4_redknot_http.py' || true)
if [[ -n "$active_release_pids" || -n "$active_worker_pids" ]]; then
  printf '%s\n' \
    'ERROR: another RedKnot benchmark is already using this TP8 server.' \
    "release_pids=${active_release_pids:-none}" \
    "worker_pids=${active_worker_pids:-none}" \
    'Wait for its result and automatic holder restoration before retrying.' >&2
  exit 75
fi

if [[ "$hardware_profile" == "b300" ]]; then
  # The existing B300 runtime is isolated from the H200 CUDA-12.8 venv.  CUDA
  # 12.9 supplies SM103-capable ATen/Jiterator kernels and the separately built
  # FlashMLA tree supplies its SM100-family code object.
  nvrtc_lib=/usr/local/cuda/lib64/libnvrtc.so.12
  nvjitlink_lib=/usr/local/cuda/lib64/libnvJitLink.so.12
  nvshmem_lib_dir=/root/miniconda3/lib/python3.11/site-packages/nvidia/nvshmem/lib
  flashmla_sm103_root=${REDKNOT_FLASHMLA_SM103_ROOT:-/data/temp/FlashMLA-sm103-src}
  test -r "$nvrtc_lib"
  test -r "$nvjitlink_lib"
  test -d "$nvshmem_lib_dir"
  test -d "$flashmla_sm103_root/flash_mla"
  export LD_PRELOAD="$nvrtc_lib:$nvjitlink_lib${LD_PRELOAD:+:$LD_PRELOAD}"
  export LD_LIBRARY_PATH="$nvshmem_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
  export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.3a}"
  export SGLANG_USE_JIT_RMSNORM="${SGLANG_USE_JIT_RMSNORM:-1}"
  export SGLANG_USE_JIT_MHC="${SGLANG_USE_JIT_MHC:-1}"
  export SGLANG_USE_JIT_GROUP_QUANT="${SGLANG_USE_JIT_GROUP_QUANT:-1}"
  export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-1}"
  export REDKNOT_MOE_RUNNER_BACKEND="${REDKNOT_B300_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
  export REDKNOT_DISABLE_FLASHINFER_AUTOTUNE="${REDKNOT_DISABLE_FLASHINFER_AUTOTUNE:-1}"
  export REDKNOT_DSV4_DISABLE_DUAL_SCOPE_SPLITK="${REDKNOT_DSV4_DISABLE_DUAL_SCOPE_SPLITK:-1}"
  export REDKNOT_FLASHMLA_SM103_ROOT="$flashmla_sm103_root"
  export PYTHONPATH="$repo/python:$flashmla_sm103_root${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$script_dir"
if [[ -x "$venv/bin/python" ]]; then
  printf '[setup] validating existing environment: %s\n' "$venv"
  "$utils_dir/setup_deepseek_v4_flash_env.sh" --check-only
else
  printf '[setup] creating pinned environment: %s\n' "$venv"
  "$utils_dir/setup_deepseek_v4_flash_env.sh"
fi

source "$utils_dir/environment-deepseek-v4-flash.env"
printf '[run] python=%s\n' "$(command -v python)"
if [[ $# -eq 0 ]]; then
  printf '%s\n' \
    '[run] default suites=64K + 128K + 256K + 440K, 15 cases each (10 short + 5 long)' \
    '[run] hot-state TTFT=3 warmups + 10 measured pairs per case' \
    '[run] output=one complete Recomputed vs RedKnot text pair per case' \
    '[run] Recomputed=same DeepSeek-V4-Flash model with full online recomputation'
fi
exec python "$script_dir/benchmark_RedKnot_DeepSeekV4Flash.py" "$@"
