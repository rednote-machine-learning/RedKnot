#!/bin/bash
# Briefly release all eight B300 GPUs, run the isolated Pro kernel oracle on
# GPU 0, and restore the exact gpu_hold.py workload even when the probe fails.
set -uo pipefail

HOLDER_DIR=/workspace/RedKnot/test/srt/redknot/utils
HOLDER_PYTHON=/root/miniconda3/bin/python
HOLDER_PATTERN='^/root/miniconda3/bin/python gpu_hold.py$'
PROBE_ROOT=/workspace/RedKnot
PROBE_PYTHON=/workspace/RedKnot/.venv_sm103/bin/python
STATUS_FILE=/tmp/pro0813_groups2_sm103_probe.status
NVRTC_LIB=/usr/local/cuda/lib64/libnvrtc.so.12
NVJITLINK_LIB=/usr/local/cuda/lib64/libnvJitLink.so.12
NVSHMEM_LIB_DIR=/root/miniconda3/lib/python3.11/site-packages/nvidia/nvshmem/lib

# PyTorch's DT_RPATH resolves the shared pip NVRTC 12.8 build by default.  It
# cannot compile Jiterator operations for B300 compute_103.  Fail before
# releasing the holder if the CUDA 12.9 interposition libraries are absent.
if [[ ! -r "$NVRTC_LIB" || ! -r "$NVJITLINK_LIB" ]]; then
  echo "CUDA-12.9 NVRTC/nvJitLink libraries required for B300 are unreadable" >&2
  exit 69
fi
if [[ ! -d "$NVSHMEM_LIB_DIR" ]] \
   || ! compgen -G "$NVSHMEM_LIB_DIR/libnvshmem*.so*" >/dev/null; then
  echo "required NVSHMEM library directory is invalid: $NVSHMEM_LIB_DIR" >&2
  exit 69
fi

gpu_holder_healthy() {
  local health_rows=""
  if ! health_rows=$(nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null); then
    return 1
  fi
  awk -F',' '
    BEGIN { count = 0; bad = 0 }
    {
      for (field = 1; field <= 3; field += 1) {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
      }
      count += 1
      if ($1 != count - 1 || $2 + 0 < 95 || $3 + 0 < 200000) bad = 1
    }
    END { exit(count == 8 && !bad ? 0 : 1) }
  ' <<<"$health_rows"
}

gpu_compute_empty() {
  local compute_rows=""
  if ! compute_rows=$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null); then
    return 1
  fi
  [[ -z "$(sed '/^[[:space:]]*$/d' <<<"$compute_rows")" ]]
}

restore_holder() {
  rc=$?
  restore_rc=$rc
  trap - EXIT
  holder_pid=$(pgrep -f "$HOLDER_PATTERN" | head -1 || true)
  if [[ -z "$holder_pid" ]]; then
    if gpu_compute_empty; then
      (
        cd "$HOLDER_DIR" || exit 1
        setsid nohup "$HOLDER_PYTHON" gpu_hold.py \
          >/tmp/gpu_hold_b300.log 2>&1 </dev/null &
        echo $! >/tmp/gpu_hold_b300.pid
      )
      for _ in $(seq 1 120); do
        holder_pid=$(pgrep -f "$HOLDER_PATTERN" | head -1 || true)
        if [[ -n "$holder_pid" ]]; then
          worker_count=$(ps --ppid "$holder_pid" -o cmd= 2>/dev/null | grep -c 'multiprocessing.spawn' || true)
          if [[ "$worker_count" -eq 8 ]]; then
            break
          fi
        fi
        sleep 1
      done
    fi
  fi
  holder_pid=$(pgrep -f "$HOLDER_PATTERN" | head -1 || true)
  worker_count=0
  holder_pgid=missing
  holder_health=failed
  if [[ -n "$holder_pid" ]]; then
    for _ in $(seq 1 120); do
      worker_count=$(ps --ppid "$holder_pid" -o cmd= 2>/dev/null | grep -c 'multiprocessing.spawn' || true)
      holder_pgid=$(ps -o pgid= -p "$holder_pid" 2>/dev/null | tr -d ' ')
      if [[ "$worker_count" -eq 8 && "$holder_pgid" == "$holder_pid" ]] \
         && gpu_holder_healthy; then
        holder_health=pass
        break
      fi
      sleep 1
    done
  fi
  if [[ "$holder_health" != pass && "$restore_rc" -eq 0 ]]; then
    restore_rc=73
  fi
  printf 'probe_rc=%s final_rc=%s holder_pid=%s holder_pgid=%s holder_workers=%s holder_health=%s restored_at=%s\n' \
    "$rc" "$restore_rc" "${holder_pid:-missing}" "$holder_pgid" "$worker_count" \
    "$holder_health" "$(date -Is)" >"$STATUS_FILE"
  exit "$restore_rc"
}
trap restore_holder EXIT

holder_pid=$(pgrep -f "$HOLDER_PATTERN" | head -1 || true)
if [[ -n "$holder_pid" ]]; then
  holder_worker_count=$(ps --ppid "$holder_pid" -o cmd= 2>/dev/null | grep -c 'multiprocessing.spawn' || true)
  holder_pgid=$(ps -o pgid= -p "$holder_pid" 2>/dev/null | tr -d ' ')
  # The legacy holder may predate the dedicated-session contract. Its exact
  # parent/children are still safe to stop; restore_holder normalizes the new
  # holder to PID=PGID via setsid for the formal supervisor.
  if [[ "$holder_worker_count" -ne 8 ]] || ! gpu_holder_healthy; then
    echo "refusing an unauthenticated or unhealthy B300 holder pid=$holder_pid workers=$holder_worker_count pgid=${holder_pgid:-missing}" >&2
    exit 70
  fi
  echo "Stopping B300 holder pid=$holder_pid for the SM103 oracle"
  holder_children=$(ps --ppid "$holder_pid" -o pid= | tr '\n' ' ')
  kill -INT "$holder_pid"
  for _ in $(seq 1 15); do
    if ! kill -0 "$holder_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$holder_pid" 2>/dev/null; then
    # A holder started as a background job may inherit SIGINT=ignored.  The
    # targets below were captured from this exact parent before termination.
    echo "Holder ignored SIGINT; terminating its exact process tree"
    kill -TERM $holder_children "$holder_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      remaining=0
      for pid in $holder_children "$holder_pid"; do
        kill -0 "$pid" 2>/dev/null && remaining=$((remaining + 1))
      done
      [[ "$remaining" -eq 0 ]] && break
      sleep 1
    done
    if [[ "$remaining" -ne 0 ]]; then
      kill -KILL $holder_children "$holder_pid" 2>/dev/null || true
    fi
  fi
fi

compute_pids=-1
gpu_query_ok=0
for _ in $(seq 1 120); do
  if ! compute_rows=$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null); then
    sleep 1
    continue
  fi
  gpu_query_ok=1
  compute_pids=$(sed '/^[[:space:]]*$/d' <<<"$compute_rows" | wc -l)
  if [[ "$compute_pids" -eq 0 ]]; then
    break
  fi
  sleep 1
done
if [[ "$gpu_query_ok" -ne 1 || "$compute_pids" -ne 0 ]]; then
  echo "GPU state could not be proven empty after holder shutdown" >&2
  exit 71
fi

cd "$PROBE_ROOT"
export LD_PRELOAD="$NVRTC_LIB:$NVJITLINK_LIB"
export LD_LIBRARY_PATH="$NVSHMEM_LIB_DIR"
export PYTHONPATH=/data/temp/FlashMLA-sm103-src:/workspace/RedKnot/python
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH=0
export SGLANG_USE_JIT_RMSNORM=1
export FLASHINFER_CUDA_ARCH_LIST=10.3a

CUDA_VISIBLE_DEVICES=0 timeout 1800 "$PROBE_PYTHON" \
  test/srt/redknot/utils/probe_pro0813_jit_rmsnorm_sm103.py
jit_rmsnorm_rc=$?
CUDA_VISIBLE_DEVICES=0 REDKNOT_PRO0813_B300_TRITON_H1_PROBE=1 \
  timeout 1800 "$PROBE_PYTHON" \
  test/srt/redknot/utils/probe_pro0813_triton_h1_sm103.py \
  --expected-source-root /workspace/RedKnot/python
triton_h1_rc=$?
CUDA_VISIBLE_DEVICES=0 timeout 1800 "$PROBE_PYTHON" \
  test/srt/redknot/utils/probe_pro0813_groups2_sm103.py
groups2_rc=$?
flashmla_rc=0
if [[ "${REDKNOT_SKIP_FLASHMLA_PROBE:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 timeout 1800 "$PROBE_PYTHON" \
    test/srt/redknot/utils/probe_flashmla_pro0813_sm103.py
  flashmla_rc=$?
fi
shared_latent_rc=0
if [[ "${REDKNOT_SKIP_SHARED_LATENT_PROBE:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 timeout 1800 "$PROBE_PYTHON" \
    test/srt/redknot/utils/probe_pro0813_shared_latent_batch_sm103.py
  shared_latent_rc=$?
fi
rope_reloc_rc=0
if [[ "${REDKNOT_SKIP_ROPE_RELOC_PROBE:-0}" != "1" ]]; then
  SGLANG_USE_JIT_PACKED_OFFSETS=1 \
    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 timeout 1800 "$PROBE_PYTHON" \
    test/srt/redknot/utils/probe_pro0813_rope_reloc_sm103.py
  rope_reloc_rc=$?
fi
if [[ "$jit_rmsnorm_rc" -ne 0 || "$triton_h1_rc" -ne 0 || \
      "$groups2_rc" -ne 0 || "$flashmla_rc" -ne 0 || \
      "$shared_latent_rc" -ne 0 || "$rope_reloc_rc" -ne 0 ]]; then
  echo "SM103 probes failed: jit_rmsnorm_rc=$jit_rmsnorm_rc triton_h1_rc=$triton_h1_rc groups2_rc=$groups2_rc flashmla_rc=$flashmla_rc shared_latent_rc=$shared_latent_rc rope_reloc_rc=$rope_reloc_rc" >&2
  exit 72
fi
