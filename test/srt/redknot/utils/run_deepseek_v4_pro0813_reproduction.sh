#!/usr/bin/env bash
# One-click isolated Pro-0813 reproduction.  The paired supervisor owns the
# exact holder handoff and restores 8-card utilization on every exit path.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$script_dir/../../../.." && pwd -P)
# Discard ambient source overlays before even the profile/model CPU gates.
# The Flash venv remains the certified interpreter, not a source checkout.
flashmla_sm103_root=${REDKNOT_FLASHMLA_SM103_ROOT:-/data/temp/FlashMLA-sm103-src}
if [[ "$flashmla_sm103_root" != /* || "$flashmla_sm103_root" == *:* ]]; then
  printf 'ERROR: REDKNOT_FLASHMLA_SM103_ROOT must be an absolute colon-free path: %q\n' \
    "$flashmla_sm103_root" >&2
  exit 66
fi
export PYTHONPATH="$repo/python:$flashmla_sm103_root"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
usage='usage: run_deepseek_v4_pro0813_reproduction.sh [TARGET_TOKENS [HOLDER_PID]] | [RUN_DIR [TARGET_TOKENS [HOLDER_PID]]]'

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' "$usage"
  printf '%s\n' \
    'Official model: /workspace/Models/DeepSeek-V4-Pro-0813' \
    'Geometry: TP8, 61 layers, 128 heads, dense={0,1,2,58,59,60}, reuse=3..57, Indexer Top-K=1024.' \
    'Hardware: 8x NVIDIA B300 / Blackwell SM103. Default target: 65536 tokens (minimum expected benefit point).' \
    'Omitting RUN_DIR names it from target tokens: pro0813-{64k,128k,256k,440k,512k}-TIMESTAMP.' \
    'Formal defaults: TTFT warmup/iters=3/10, QPS concurrency=1 with warmup/measurement waves=3/10, strict-performance.' \
    'Explicit performance opt-out: REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1 keeps the full combined algorithm but marks results claim-ineligible.' \
    'Optional zoff-only algorithm ablation additionally requires REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY=1.' \
    'Other overrides include REDKNOT_FIRST_DOCUMENT_PREFIX, REDKNOT_MEASURE_QPS, or REDKNOT_PRO0813_SERVER_PORT.'
  exit 0
fi

run_dir=""
target_tokens=65536
holder_pid=""
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  if (( $# > 2 )); then
    printf 'ERROR: target-first form expects at most 2 arguments, got %s\n%s\n' \
      "$#" "$usage" >&2
    exit 64
  fi
  target_tokens=$1
  holder_pid=${2:-}
else
  if (( $# > 3 )); then
    printf 'ERROR: path-first form expects at most 3 arguments, got %s\n%s\n' \
      "$#" "$usage" >&2
    exit 64
  fi
  run_dir=${1:-}
  target_tokens=${2:-65536}
  holder_pid=${3:-}
fi
runtime_python=/workspace/RedKnot/.venv_sm103/bin/python
lock_file=${REDKNOT_PRO0813_RUN_LOCK:-/tmp/redknot_deepseek_v4_pro0813_release.lock}

if [[ -n "$holder_pid" ]] && ! [[ "$holder_pid" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: HOLDER_PID must be a positive integer, got %q\n' \
    "$holder_pid" >&2
  exit 64
fi

case "$target_tokens" in
  65536) target_label=64k; target_mem_fraction_static=0.80 ;;
  131072) target_label=128k; target_mem_fraction_static=0.77 ;;
  262144) target_label=256k; target_mem_fraction_static=0.67 ;;
  450560) target_label=440k; target_mem_fraction_static=0.80 ;;
  524288) target_label=512k; target_mem_fraction_static=0.80 ;;
  *)
    printf 'ERROR: unsupported Pro-0813 target tokens: %q\n' "$target_tokens" >&2
    exit 64
    ;;
esac
if [[ -z "$run_dir" ]]; then
  run_dir="/workspace/RedKnot/results/pro0813-${target_label}-$(date +%Y%m%d-%H%M%S)"
fi
if [[ "$run_dir" != /* || "$run_dir" == "/" ]]; then
  printf 'ERROR: RUN_DIR must be an absolute non-root path, got %q\n' \
    "$run_dir" >&2
  exit 64
fi
qualification_profile=${REDKNOT_QUALIFICATION_PROFILE:-}
expected_qualification_profile_sha256=${REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256:-}
if [[ ( "$target_tokens" == 450560 || "$target_tokens" == 524288 ) \
      && -z "$qualification_profile" ]]; then
  printf '%s\n' \
    'ERROR: 440K/512K requires REDKNOT_QUALIFICATION_PROFILE with a frozen prompt/data contract.' >&2
  exit 64
fi
if [[ -n "$qualification_profile" \
      && ( "$qualification_profile" != /* \
           || -L "$qualification_profile" \
           || ! -f "$qualification_profile" \
           || ! -r "$qualification_profile" ) ]]; then
  printf 'ERROR: REDKNOT_QUALIFICATION_PROFILE must be a readable absolute regular file, got %q\n' \
    "$qualification_profile" >&2
  exit 66
fi
if [[ -n "$expected_qualification_profile_sha256" \
      && ! "$expected_qualification_profile_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'ERROR: REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256 must be lowercase SHA-256, got %q\n' \
    "$expected_qualification_profile_sha256" >&2
  exit 66
fi

exec 9>"$lock_file"
if ! flock -n 9; then
  printf 'ERROR: another Pro-0813 reproduction owns %s\n' "$lock_file" >&2
  exit 75
fi

for required in \
  "$script_dir/verify_pro0813_official_model.py" \
  "$script_dir/verify_pro0813_formal_assets.py" \
  "$script_dir/verify_pro0813_qualification_profile.py" \
  "$repo/server/start_server_redknot_pro0813.sh" \
  "$script_dir/benchmark_dsv4_pro0813_redknot_http.py" \
  "$script_dir/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py"; do
  if [[ ! -r "$required" ]]; then
    printf 'ERROR: required Pro-0813 input is absent: %s\n' "$required" >&2
    exit 66
  fi
done
if [[ ! -x "$runtime_python" ]]; then
  printf 'ERROR: certified Pro-0813 runtime Python is not executable: %s\n' \
    "$runtime_python" >&2
  exit 66
fi
printf '%s\n' \
  'validating frozen Pro-0813 target assets before model gate and holder lookup'
if ! formal_asset_record=$(
  "$runtime_python" "$script_dir/verify_pro0813_formal_assets.py" \
    --target-tokens "$target_tokens"
); then
  printf '%s\n' \
    'ERROR: Pro-0813 target manifest/dataset/profile assets failed their immutable CPU contract' >&2
  exit 66
fi
printf 'formal_assets_verified %s\n' "$formal_asset_record"
printf '%s\n' 'validating complete official Pro-0813 model before holder lookup'
if ! "$runtime_python" "$script_dir/verify_pro0813_official_model.py"; then
  printf '%s\n' \
    'ERROR: official Pro-0813 model is incomplete or does not match the immutable manifest' >&2
  exit 66
fi

if [[ -z "$holder_pid" ]]; then
  holder_candidates=()
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    pgid=$(ps -o pgid= -p "$candidate" 2>/dev/null | tr -d ' ')
    [[ "$pgid" == "$candidate" ]] && holder_candidates+=("$candidate")
  done < <(pgrep -f '[g]pu_hold.py' || true)
  if [[ ${#holder_candidates[@]} -ne 1 ]]; then
    printf 'ERROR: expected one gpu_hold.py process-group leader, found: %s\n' \
      "${holder_candidates[*]:-none}" >&2
    printf '%s\n' 'Pass HOLDER_PID explicitly after verifying the holder identity.' >&2
    exit 67
  fi
  holder_pid=${holder_candidates[0]}
fi

mkdir -p "$run_dir"
cd "$repo"
printf 'pro0813_reproduction run_dir=%s target_tokens=%s holder_pid=%s\n' \
  "$run_dir" "$target_tokens" "$holder_pid"

exec "$script_dir/run_pro0813_combined_supervisor.sh" \
  "$run_dir" \
  "$holder_pid" \
  "${REDKNOT_TIMING:-1}" \
  "${REDKNOT_TTFT_WARMUP:-3}" \
  "${REDKNOT_TTFT_ITERS:-10}" \
  "${REDKNOT_MERGED_PREFILL_TOKENS:-0}" \
  "$target_tokens" \
  "${REDKNOT_FIRST_DOCUMENT_PREFIX:-1}" \
  "${REDKNOT_GEOMETRY_TEMPLATE_CACHE:-0}" \
  "${REDKNOT_MEM_FRACTION_STATIC:-$target_mem_fraction_static}" \
  "${REDKNOT_QPS_CONCURRENCIES:-1}" \
  "${REDKNOT_MEASURE_QPS:-1}" \
  "${REDKNOT_QPS_WARMUP_WAVES:-3}" \
  "${REDKNOT_QPS_WAVES:-10}" \
  "${REDKNOT_ROW_SPARSE_ACTIVE_RATIO:-0.20}" \
  "$qualification_profile" \
  "${REDKNOT_QUERY_PROTECTION_TOKENS:-8192}" \
  "${REDKNOT_GENERALIZED_ADAPTIVE_CONTROLLER:-0}" \
  "${REDKNOT_QUALITY_REPEATS:-3}" \
  "${REDKNOT_GENERALIZED_STRONG_RATIO:-0.15}" \
  "${REDKNOT_GENERALIZED_MEDIUM_RATIO:-0.20}" \
  "${REDKNOT_GENERALIZED_DIFFUSE_RATIO:-0.25}"
