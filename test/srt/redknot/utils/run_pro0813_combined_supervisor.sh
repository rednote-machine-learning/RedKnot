#!/usr/bin/env bash
# Isolated DeepSeek-V4-Pro-0813 reproduction supervisor.
# Accelerator contract: 8x NVIDIA B300 / Blackwell SM103 / TP8.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$script_dir/../../../.." && pwd -P)
# The supervisor invokes several Python programs before the server launcher.
# Pin their import graph too: never propagate a login-shell PYTHONPATH (most
# importantly /workspace/RedKnot/python) into the isolated Pro process tree.
flashmla_sm103_root=${REDKNOT_FLASHMLA_SM103_ROOT:-/data/temp/FlashMLA-sm103-src}
if [[ "$flashmla_sm103_root" != /* || "$flashmla_sm103_root" == *:* ]]; then
  printf 'ERROR: REDKNOT_FLASHMLA_SM103_ROOT must be an absolute colon-free path: %q\n' \
    "$flashmla_sm103_root" >&2
  exit 87
fi
export PYTHONPATH="$repo/python:$flashmla_sm103_root"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
holder_cwd=${REDKNOT_HOLDER_CWD:-/workspace/RedKnot/test/srt/redknot/utils}
holder_python=${REDKNOT_HOLDER_PYTHON:-/root/miniconda3/bin/python}
usage='usage: run_pro0813_combined_supervisor.sh RUN_DIR HOLDER_PID [TIMING] [TTFT_WARMUP] [TTFT_ITERS] [MERGED_PREFILL] [TARGET_TOKENS] [FIRST_DOCUMENT_PREFIX] [GEOMETRY_TEMPLATE_CACHE] [MEM_FRACTION_STATIC] [QPS_CONCURRENCIES] [MEASURE_QPS] [QPS_WARMUP_WAVES] [QPS_WAVES] [ROW_SPARSE_ACTIVE_RATIO] [QUALIFICATION_PROFILE] [QUERY_PROTECTION_TOKENS] [GENERALIZED_ADAPTIVE_CONTROLLER] [QUALITY_REPEATS] [STRONG_RATIO] [MEDIUM_RATIO] [DIFFUSE_RATIO]'
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' "$usage"
  printf '%s\n' \
    'Runs the isolated Pro-0813 benchmark on 8x B300/SM103 and attempts an authenticated full-holder restore on every exit; cleanup fails closed if GPU idleness cannot be proved.' \
    'Default target is 65536 tokens; 64K is the minimum expected RedKnot benefit point.' \
    'Formal defaults: TTFT=3/10, QPS concurrency=1 and waves=3/10, MEASURE_QPS=1, strict-performance.' \
    'Explicit performance opt-out: REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1 keeps the full combined algorithm but marks results claim-ineligible.' \
    'Algorithm ablation requires both REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1 and REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY=1.' \
    'Other defaults: FIRST_DOCUMENT_PREFIX=1, REDKNOT_PRO0813_SERVER_PORT=31998.'
  exit 0
fi
if (( $# < 2 || $# > 22 )); then
  printf 'ERROR: expected 2..22 arguments, got %s\n%s\n' "$#" "$usage" >&2
  exit 64
fi
run_dir=$1
holder_pid=$2
timing_mode=${3:-1}
ttft_warmup=${4:-3}
ttft_iters=${5:-10}
merged_prefill_tokens=${6:-0}
target_tokens=${7:-65536}
first_document_prefix=${8:-1}
geometry_template_cache=${9:-0}
mem_fraction_static=${10:-}
qps_concurrencies=${11:-1}
measure_qps=${12:-1}
qps_warmup_waves=${13:-3}
qps_waves=${14:-10}
row_sparse_active_ratio=${15:-0.20}
qualification_profile=${16:-}
query_protection_tokens=${17:-8192}
generalized_adaptive_controller=${18:-0}
quality_repeats=${19:-3}
generalized_strong_ratio=${20:-0.15}
generalized_medium_ratio=${21:-0.20}
generalized_diffuse_ratio=${22:-0.25}
adaptive_topk_mass=${REDKNOT_RELEASE_ADAPTIVE_TOPK_MASS:-0.50}
adaptive_topk_buckets=${REDKNOT_RELEASE_ADAPTIVE_TOPK_BUCKETS:-3,4,5,6}
combined_headsplit_row_sparse=${REDKNOT_COMBINED_HEADSPLIT_ROW_SPARSE:-1}
diagnostic_performance=${REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE:-0}
diagnostic_zoff_only=${REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY:-0}
adaptive_topk_enabled=${REDKNOT_PRO0813_ADAPTIVE_TOPK_ENABLED:-0}
expected_qualification_profile_sha256=${REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256:-}
server_port=${REDKNOT_PRO0813_SERVER_PORT:-31998}
if [[ "$run_dir" != /* || "$run_dir" == "/" ]]; then
  printf 'ERROR: RUN_DIR must be an absolute non-root path, got %q\n' \
    "$run_dir" >&2
  exit 64
fi
if ! [[ "$holder_pid" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: HOLDER_PID must be a positive integer, got %q\n' \
    "$holder_pid" >&2
  exit 64
fi
if ! [[ "$server_port" =~ ^[1-9][0-9]{0,4}$ ]] \
   || (( 10#$server_port > 65535 )); then
  printf 'ERROR: REDKNOT_PRO0813_SERVER_PORT must be in [1, 65535], got %q\n' \
    "$server_port" >&2
  exit 64
fi
if [[ "$timing_mode" != 0 && "$timing_mode" != 1 ]]; then
  printf 'ERROR: TIMING must be 0 or 1, got %q\n' "$timing_mode"
  exit 89
fi
if ! [[ "$ttft_warmup" =~ ^[0-9]+$ && "$ttft_iters" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: invalid TTFT sampling warmup=%q iters=%q\n' \
    "$ttft_warmup" "$ttft_iters"
  exit 89
fi
if [[ "$merged_prefill_tokens" != 0 \
      && "$merged_prefill_tokens" != 32768 \
      && "$merged_prefill_tokens" != 57344 \
      && "$merged_prefill_tokens" != 65536 ]]; then
  printf 'ERROR: MERGED_PREFILL must be 0, 32768, 57344 or 65536, got %q\n' \
    "$merged_prefill_tokens"
  exit 89
fi
if [[ "$target_tokens" != 65536 \
      && "$target_tokens" != 131072 \
      && "$target_tokens" != 262144 \
      && "$target_tokens" != 450560 \
      && "$target_tokens" != 524288 ]]; then
  printf 'ERROR: TARGET_TOKENS must be 65536, 131072, 262144, 450560 or 524288, got %q\n' \
    "$target_tokens"
  exit 89
fi
case "$target_tokens" in
  65536|131072) target_checkpoint_max_islands=64 ;;
  262144) target_checkpoint_max_islands=128 ;;
  450560|524288) target_checkpoint_max_islands=256 ;;
esac
checkpoint_max_islands=${REDKNOT_ROW_SPARSE_CHECKPOINT_MAX_ISLANDS:-$target_checkpoint_max_islands}
if ! [[ "$checkpoint_max_islands" =~ ^[1-9][0-9]*$ ]] \
   || (( 10#$checkpoint_max_islands > 256 )); then
  printf 'ERROR: REDKNOT_ROW_SPARSE_CHECKPOINT_MAX_ISLANDS must be in [1, 256], got %q\n' \
    "$checkpoint_max_islands"
  exit 89
fi
if [[ -z "$mem_fraction_static" ]]; then
  case "$target_tokens" in
    65536) mem_fraction_static=0.80 ;;
    131072) mem_fraction_static=0.77 ;;
    262144) mem_fraction_static=0.67 ;;
    450560|524288) mem_fraction_static=0.80 ;;
  esac
fi
if [[ "$adaptive_topk_enabled" != 0 && "$adaptive_topk_enabled" != 1 ]]; then
  printf 'ERROR: REDKNOT_PRO0813_ADAPTIVE_TOPK_ENABLED must be 0 or 1, got %q\n' \
    "$adaptive_topk_enabled"
  exit 89
fi
if [[ "$first_document_prefix" != 0 && "$first_document_prefix" != 1 ]]; then
  printf 'ERROR: FIRST_DOCUMENT_PREFIX must be 0 or 1, got %q\n' \
    "$first_document_prefix"
  exit 89
fi
if [[ "$geometry_template_cache" != 0 && "$geometry_template_cache" != 1 ]]; then
  printf 'ERROR: GEOMETRY_TEMPLATE_CACHE must be 0 or 1, got %q\n' \
    "$geometry_template_cache"
  exit 89
fi
if ! [[ "$mem_fraction_static" =~ ^0\.[0-9]+$|^1\.0+$ ]]; then
  printf 'ERROR: MEM_FRACTION_STATIC is invalid: %q\n' "$mem_fraction_static"
  exit 89
fi
if ! awk -v value="$mem_fraction_static" \
  'BEGIN { exit !(value > 0.0 && value <= 1.0) }'; then
  printf 'ERROR: MEM_FRACTION_STATIC must be in (0, 1], got %q\n' \
    "$mem_fraction_static"
  exit 89
fi
if ! [[ "$qps_concurrencies" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  printf 'ERROR: QPS_CONCURRENCIES is invalid: %q\n' "$qps_concurrencies"
  exit 89
fi
if [[ "$measure_qps" != 0 && "$measure_qps" != 1 ]]; then
  printf 'ERROR: MEASURE_QPS must be 0 or 1, got %q\n' "$measure_qps"
  exit 89
fi
if [[ "$diagnostic_performance" != 0 \
      && "$diagnostic_performance" != 1 ]]; then
  printf 'ERROR: REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE must be 0 or 1, got %q\n' \
    "$diagnostic_performance"
  exit 89
fi
if [[ "$diagnostic_zoff_only" != 0 \
      && "$diagnostic_zoff_only" != 1 ]]; then
  printf 'ERROR: REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY must be 0 or 1, got %q\n' \
    "$diagnostic_zoff_only"
  exit 89
fi
if ! [[ "$qps_warmup_waves" =~ ^[0-9]+$ \
        && "$qps_waves" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: invalid QPS sampling warmup=%q waves=%q\n' \
    "$qps_warmup_waves" "$qps_waves"
  exit 89
fi
if [[ "$diagnostic_performance" == 0 ]] \
   && (( ttft_warmup < 3 \
         || ttft_iters < 10 \
         || measure_qps != 1 \
         || qps_warmup_waves < 3 \
         || qps_waves < 10 )); then
  printf '%s\n' \
    'ERROR: formal Pro-0813 reproduction requires TTFT warmup/iters >=3/10, QPS measurement enabled, and QPS warmup/measurement waves >=3/10; set REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1 only for claim-ineligible diagnostics'
  exit 89
fi
if ! [[ "$row_sparse_active_ratio" =~ ^0\.[0-9]+$|^1\.0+$ ]]; then
  printf 'ERROR: ROW_SPARSE_ACTIVE_RATIO is invalid: %q\n' \
    "$row_sparse_active_ratio"
  exit 89
fi
if ! awk -v value="$row_sparse_active_ratio" \
  'BEGIN { exit !(value > 0.0 && value < 0.85) }'; then
  printf 'ERROR: ROW_SPARSE_ACTIVE_RATIO must be in (0, 0.85), got %q\n' \
    "$row_sparse_active_ratio"
  exit 89
fi
if ! [[ "$query_protection_tokens" =~ ^[1-9][0-9]*$ ]] \
   || (( query_protection_tokens < 512 \
         || query_protection_tokens % 512 != 0 )); then
  printf 'ERROR: QUERY_PROTECTION_TOKENS must be a positive 512 multiple: %q\n' \
    "$query_protection_tokens"
  exit 89
fi
if [[ "$generalized_adaptive_controller" != 0 \
      && "$generalized_adaptive_controller" != 1 ]]; then
  printf 'ERROR: GENERALIZED_ADAPTIVE_CONTROLLER must be 0 or 1, got %q\n' \
    "$generalized_adaptive_controller"
  exit 89
fi
if [[ "$combined_headsplit_row_sparse" != 0 \
      && "$combined_headsplit_row_sparse" != 1 ]]; then
  printf 'ERROR: REDKNOT_COMBINED_HEADSPLIT_ROW_SPARSE must be 0 or 1, got %q\n' \
    "$combined_headsplit_row_sparse"
  exit 89
fi
if [[ "$combined_headsplit_row_sparse" == 1 \
      && "$first_document_prefix" != 1 ]]; then
  printf '%s\n' \
    'ERROR: combined headsplit/row-sparse requires FIRST_DOCUMENT_PREFIX=1'
  exit 89
fi
if [[ "$diagnostic_zoff_only" == 1 \
      && ( "$diagnostic_performance" != 1 \
           || "$combined_headsplit_row_sparse" != 1 ) ]]; then
  printf '%s\n' \
    'ERROR: REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY=1 requires REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1 and REDKNOT_COMBINED_HEADSPLIT_ROW_SPARSE=1'
  exit 89
fi
if [[ "$generalized_adaptive_controller" == 1 \
      && "$combined_headsplit_row_sparse" != 1 ]]; then
  printf '%s\n' \
    'ERROR: generalized adaptive controller requires the combined path'
  exit 89
fi
if ! [[ "$quality_repeats" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: QUALITY_REPEATS must be positive, got %q\n' "$quality_repeats"
  exit 89
fi
if [[ "$diagnostic_performance" != 1 && "$quality_repeats" -lt 3 ]]; then
  printf 'ERROR: formal performance requires QUALITY_REPEATS >= 3, got %q\n' \
    "$quality_repeats"
  exit 89
fi
for generalized_ratio in \
  "$generalized_strong_ratio" \
  "$generalized_medium_ratio" \
  "$generalized_diffuse_ratio"; do
  if ! [[ "$generalized_ratio" =~ ^0\.[0-9]+$ ]]; then
    printf 'ERROR: generalized active ratio is invalid: %q\n' "$generalized_ratio"
    exit 84
  fi
done
if ! awk \
  -v strong="$generalized_strong_ratio" \
  -v medium="$generalized_medium_ratio" \
  -v diffuse="$generalized_diffuse_ratio" \
  'BEGIN {
    valid = strong > 0.0 && diffuse < 0.85 && strong <= medium && medium <= diffuse
    exit(valid ? 0 : 1)
  }'; then
  printf 'ERROR: generalized ratios must be in (0, 0.85) and strong <= medium <= diffuse: %q,%q,%q\n' \
    "$generalized_strong_ratio" "$generalized_medium_ratio" \
    "$generalized_diffuse_ratio"
  exit 84
fi
if ! [[ "$adaptive_topk_mass" =~ ^0\.[0-9]+$|^1\.0+$ ]]; then
  printf 'ERROR: adaptive Top-K mass is invalid: %q\n' "$adaptive_topk_mass"
  exit 84
fi
if ! awk -v value="$adaptive_topk_mass" \
  'BEGIN { exit !(value > 0.0 && value <= 1.0) }'; then
  printf 'ERROR: adaptive Top-K mass must be in (0, 1], got %q\n' \
    "$adaptive_topk_mass"
  exit 84
fi
if ! [[ "$adaptive_topk_buckets" =~ ^[1-6](,[1-6])*$ ]]; then
  printf 'ERROR: adaptive Top-K buckets are invalid: %q\n' \
    "$adaptive_topk_buckets"
  exit 84
fi
case "$target_tokens" in
  65536|131072) max_query_protection_tokens=8192 ;;
  262144) max_query_protection_tokens=32768 ;;
  450560) max_query_protection_tokens=56320 ;;
  524288) max_query_protection_tokens=65536 ;;
esac
if (( query_protection_tokens > max_query_protection_tokens )); then
  printf 'ERROR: QUERY_PROTECTION_TOKENS=%s exceeds target document size=%s\n' \
    "$query_protection_tokens" "$max_query_protection_tokens"
  exit 89
fi
if [[ ( "$target_tokens" == 450560 || "$target_tokens" == 524288 ) \
      && -z "$qualification_profile" ]]; then
  printf '%s\n' \
    'ERROR: 440K/512K requires a frozen QUALIFICATION_PROFILE'
  exit 89
fi
if [[ -n "$qualification_profile" \
      && ( "$qualification_profile" != /* \
           || -L "$qualification_profile" \
           || ! -f "$qualification_profile" \
           || ! -r "$qualification_profile" ) ]]; then
  printf 'ERROR: QUALIFICATION_PROFILE must be a readable absolute regular file: %q\n' \
    "$qualification_profile"
  exit 89
fi
if [[ -n "$expected_qualification_profile_sha256" \
      && ! "$expected_qualification_profile_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'ERROR: REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256 must be lowercase SHA-256: %q\n' \
    "$expected_qualification_profile_sha256"
  exit 89
fi
if [[ "$generalized_adaptive_controller" == 1 \
      && "$max_query_protection_tokens" -lt 32768 ]]; then
  printf '%s\n' \
    'ERROR: generalized adaptive controller requires 32K documents'
  exit 89
fi
benchmark_extra_args=()
if [[ "$diagnostic_performance" == 1 ]]; then
  benchmark_extra_args+=(--diagnostic-performance)
else
  # Keep the production intent explicit even though the benchmark CLI itself
  # also defaults to fail-closed formal qualification.
  benchmark_extra_args+=(--strict-performance)
fi
if [[ "$combined_headsplit_row_sparse" == 1 ]]; then
  if [[ "$diagnostic_zoff_only" == 1 ]]; then
    benchmark_extra_args+=(
      --combined-headsplit-row-sparse-diagnostic-zoff-only
    )
  else
    benchmark_extra_args+=(--combined-headsplit-row-sparse)
  fi
fi
if [[ "$first_document_prefix" == 1 ]]; then
  benchmark_extra_args+=(--first-document-prefix)
fi
if [[ "$geometry_template_cache" == 1 ]]; then
  benchmark_extra_args+=(--geometry-template-cache)
fi
if [[ "$generalized_adaptive_controller" == 1 ]]; then
  benchmark_extra_args+=(--generalized-adaptive-controller)
fi
if [[ "$adaptive_topk_enabled" == 1 ]]; then
  # This is a Pro calibration arm only until a router-histogram/quality
  # profile is pinned to the official config and geometry digests.
  benchmark_extra_args+=(
    --adaptive-topk
    --adaptive-topk-mass "$adaptive_topk_mass"
    --adaptive-topk-buckets "$adaptive_topk_buckets"
  )
fi
if [[ "$measure_qps" == 1 ]]; then
  benchmark_extra_args+=(
    --measure-qps
    --qps-warmup-waves "$qps_warmup_waves"
    --qps-waves "$qps_waves"
  )
else
  benchmark_extra_args+=(--no-measure-qps)
fi
if [[ -n "$qualification_profile" ]]; then
  benchmark_extra_args+=(--qualification-profile "$qualification_profile")
fi

# Validate and export the CUDA-12.9 loader contract before the first Python
# process is allowed to import anything.  In particular, this prevents the
# CPU contract subprocess from inheriting the incompatible pip nvJitLink 12.8.
nvjitlink_lib=/usr/local/cuda/lib64/libnvJitLink.so.12
nvshmem_lib_dir=/root/miniconda3/lib/python3.11/site-packages/nvidia/nvshmem/lib
if [[ ! -r "$nvjitlink_lib" ]]; then
  printf 'ERROR: required CUDA-12.9 nvJitLink is not readable: %s\n' \
    "$nvjitlink_lib" >&2
  exit 87
fi
if [[ ! -d "$nvshmem_lib_dir" ]] \
   || ! compgen -G "$nvshmem_lib_dir/libnvshmem*.so*" >/dev/null; then
  printf 'ERROR: required NVSHMEM library directory is invalid: %s\n' \
    "$nvshmem_lib_dir" >&2
  exit 87
fi
export LD_PRELOAD="$nvjitlink_lib${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$nvshmem_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
runtime_python=/workspace/RedKnot/.venv_sm103/bin/python
if [[ -n "${REDKNOT_PYTHON:-}" \
      && "$REDKNOT_PYTHON" != "$runtime_python" ]]; then
  printf 'ERROR: Pro-0813 forbids a non-certified REDKNOT_PYTHON: %q\n' \
    "$REDKNOT_PYTHON" >&2
  exit 87
fi
if [[ -n "${REDKNOT_PYTHON_BIN:-}" \
      && "$REDKNOT_PYTHON_BIN" != "$runtime_python" ]]; then
  printf 'ERROR: Pro-0813 forbids a non-certified REDKNOT_PYTHON_BIN: %q\n' \
    "$REDKNOT_PYTHON_BIN" >&2
  exit 87
fi
if [[ ! -x "$runtime_python" ]]; then
  printf 'ERROR: certified Pro-0813 runtime Python is not executable: %q\n' \
    "$runtime_python" >&2
  exit 87
fi
export REDKNOT_PYTHON="$runtime_python"
export REDKNOT_PYTHON_BIN="$runtime_python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/redknot-pro0813-pycache}"
profile_verifier="$script_dir/verify_pro0813_qualification_profile.py"
formal_asset_verifier="$script_dir/verify_pro0813_formal_assets.py"
if [[ ! -r "$formal_asset_verifier" ]]; then
  printf 'ERROR: formal asset verifier is not readable: %s\n' \
    "$formal_asset_verifier" >&2
  exit 88
fi
printf '%s\n' \
  'validating frozen Pro-0813 target assets before model gate and holder handoff'
if ! formal_asset_record=$(
  "$runtime_python" "$formal_asset_verifier" \
    --target-tokens "$target_tokens"
); then
  printf '%s\n' \
    'ERROR: Pro-0813 target manifest/dataset/profile assets failed their immutable CPU contract' >&2
  exit 88
fi
printf 'formal_assets_verified %s\n' "$formal_asset_record"
if [[ "$target_tokens" == 450560 || "$target_tokens" == 524288 ]]; then
  if [[ ! -r "$profile_verifier" ]]; then
    printf 'ERROR: frozen profile verifier is not readable: %s\n' \
      "$profile_verifier" >&2
    exit 88
  fi
  profile_verify_args=(
    "$qualification_profile"
    --expected-target-tokens "$target_tokens"
  )
  if [[ -n "$expected_qualification_profile_sha256" ]]; then
    profile_verify_args+=(
      --expected-profile-sha256 "$expected_qualification_profile_sha256"
    )
  fi
  printf '%s\n' \
    'validating frozen Pro-0813 qualification profile before model gate and holder handoff'
  if ! qualification_profile_record=$(
    "$runtime_python" "$profile_verifier" "${profile_verify_args[@]}"
  ); then
    printf '%s\n' \
      'ERROR: Pro-0813 qualification profile failed its immutable CPU contract' >&2
    exit 88
  fi
  printf 'qualification_profile_verified %s\n' "$qualification_profile_record"
fi
model_gate="$script_dir/verify_pro0813_official_model.py"
if [[ ! -r "$model_gate" ]]; then
  printf 'ERROR: official Pro-0813 model gate is not readable: %s\n' \
    "$model_gate" >&2
  exit 88
fi
printf '%s\n' 'validating complete official Pro-0813 model before holder handoff'
if ! "$runtime_python" "$model_gate"; then
  printf '%s\n' \
    'ERROR: official Pro-0813 model is incomplete or does not match the immutable manifest' >&2
  exit 88
fi
if ! server_identity_nonce=$("$runtime_python" -c \
  'import secrets; print(secrets.token_hex(32))'); then
  printf '%s\n' 'ERROR: cannot generate the owned-server identity nonce' >&2
  exit 86
fi
if ! [[ "$server_identity_nonce" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'ERROR: generated an invalid owned-server identity nonce' >&2
  exit 86
fi
server_identity_path="$run_dir/.pro0813-server-identity.$server_identity_nonce.json"
worker_ready_dir="$run_dir/worker_gpu_ready"
worker_go_file="$run_dir/worker_gpu_release.go"
for stale_artifact in \
  "$server_identity_path" \
  "$worker_ready_dir" \
  "$worker_go_file" \
  "$run_dir/result.json"; do
  if [[ -e "$stale_artifact" || -L "$stale_artifact" ]]; then
    printf 'ERROR: refusing stale Pro-0813 run artifact: %s\n' \
      "$stale_artifact" >&2
    exit 86
  fi
done
# Check the listening state without importing Python.  The launcher receives
# the same validated port below, so no implicit default can diverge.
server_port_hex=$(printf '%04X' "$server_port")
for socket_table in /proc/net/tcp /proc/net/tcp6; do
  [[ -r "$socket_table" ]] || continue
  if awk -v suffix=":$server_port_hex" \
    '$2 ~ suffix "$" && $4 == "0A" { found = 1 }
     END { exit(found ? 0 : 1) }' "$socket_table"; then
    printf 'ERROR: Pro-0813 server port is already listening: %s\n' \
      "$server_port" >&2
    exit 86
  fi
done
holder_pgid=""
benchmark_pid=""
benchmark_pgid=""
server_pid=""
server_pgid=""
server_sid=""
server_starttime=""
server_identity_state=""
server_identity_verified=0
heartbeat_pid=""
holder_restarted=0
holder_stopped=0
bootstrap_holder_active=0

if ! mkdir -p "$run_dir"; then
  printf 'ERROR: cannot create Pro-0813 run directory: %s\n' "$run_dir" >&2
  exit 86
fi
exec > >(tee -a "$run_dir/supervisor.log") 2>&1

printf '%s\n' 'validating official Pro-0813 CPU contract before holder handoff'
"$runtime_python" "$repo/test/srt/redknot/utils/benchmark_dsv4_pro0813_redknot_http.py" \
  --contract-only \
  --contract-config /workspace/Models/DeepSeek-V4-Pro-0813/config.json \
  | tee "$run_dir/pro0813_contract.json"
contract_pipeline_status=("${PIPESTATUS[@]}")
if (( contract_pipeline_status[0] != 0 \
      || contract_pipeline_status[1] != 0 )); then
  printf 'ERROR: Pro-0813 CPU contract pipeline failed python=%s tee=%s\n' \
    "${contract_pipeline_status[0]}" "${contract_pipeline_status[1]}"
  exit 88
fi

if ! gpu_contract=$(nvidia-smi \
  --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null); then
  printf '%s\n' 'ERROR: cannot query the B300 accelerator contract'
  exit 90
fi
gpu_contract_count=$(printf '%s\n' "$gpu_contract" \
  | sed '/^$/d' | wc -l | tr -d ' ')
if [[ "$gpu_contract_count" != 8 ]] \
   || ! printf '%s\n' "$gpu_contract" | awk -F ',' '
     NF != 2 { bad = 1; next }
     {
       gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
       gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
       if ($1 !~ /B300/ || $2 != "10.3") bad = 1
     }
     END { exit(bad ? 1 : 0) }
   '; then
  printf 'ERROR: Pro-0813 requires exactly 8x B300 compute_cap=10.3, observed=%q\n' \
    "$gpu_contract"
  exit 90
fi
printf '%s\n' \
  'accelerator_contract_pass=8x NVIDIA B300 / Blackwell SM103 compute_cap=10.3 / TP8'

GPU_QUERY_ATTEMPTS=3
GPU_QUERY_RETRY_DELAY_S=1
HOLDER_UTIL_MIN_PERCENT=90
HOLDER_UTIL_REQUIRED_SAMPLES=3
HOLDER_UTIL_MAX_SAMPLES=15
HOLDER_UTIL_SAMPLE_DELAY_S=1
declare -A EXPECTED_GPU_UUID_BY_INDEX=()
declare -A EXPECTED_GPU_INDEX_BY_UUID=()
GPU_VERIFY_REASON=""
GPU_VERIFY_MAPPING=""
HOLDER_VERIFY_REASON=""

trim_field() {
  local variable_name=$1
  local value=${!variable_name}
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf -v "$variable_name" '%s' "$value"
}

gpu_query_with_retries() {
  local query_flag=$1
  local attempt=0
  local output=""
  for ((attempt = 1; attempt <= GPU_QUERY_ATTEMPTS; attempt += 1)); do
    if output=$(nvidia-smi "$query_flag" \
      --format=csv,noheader,nounits 2>/dev/null); then
      printf '%s' "$output"
      return 0
    fi
    if (( attempt < GPU_QUERY_ATTEMPTS )); then
      sleep "$GPU_QUERY_RETRY_DELAY_S"
    fi
  done
  return 1
}

gpu_compute_snapshot() {
  gpu_query_with_retries '--query-compute-apps=gpu_uuid,pid'
}

gpu_inventory_snapshot() {
  gpu_query_with_retries '--query-gpu=index,uuid,utilization.gpu'
}

# Print unique compute PIDs only after a successful, schema-checked NVML query.
# An empty successful snapshot means idle GPUs; query failure is always nonzero.
gpu_pids() {
  local snapshot=""
  if ! snapshot=$(gpu_compute_snapshot); then
    return 2
  fi
  if [[ -z "$snapshot" ]]; then
    return 0
  fi
  printf '%s\n' "$snapshot" | awk -F ',' '
    NF != 2 { exit 2 }
    {
      uuid = $1
      pid = $2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
      if (uuid == "" || pid !~ /^[1-9][0-9]*$/) exit 2
      print pid
    }
  ' | sort -n -u
}

initialize_expected_gpu_inventory() {
  local snapshot=""
  local index=""
  local uuid=""
  local utilization=""
  local extra=""
  local count=0
  if ! snapshot=$(gpu_inventory_snapshot); then
    printf '%s\n' \
      'ERROR: cannot obtain the authenticated B300 index/UUID inventory' >&2
    return 2
  fi
  while IFS=',' read -r index uuid utilization extra; do
    [[ -z "$index$uuid$utilization$extra" ]] && continue
    trim_field index
    trim_field uuid
    trim_field utilization
    trim_field extra
    if [[ -n "$extra" || ! "$index" =~ ^[0-7]$ \
          || -z "$uuid" || ! "$utilization" =~ ^[0-9]+$ ]]; then
      printf 'ERROR: malformed or duplicate B300 inventory row: %q\n' \
        "$index,$uuid,$utilization${extra:+,$extra}" >&2
      return 2
    fi
    if [[ ${EXPECTED_GPU_UUID_BY_INDEX[$index]+present} ]]; then
      printf 'ERROR: duplicate B300 index in inventory: %s\n' "$index" >&2
      return 2
    fi
    if [[ ${EXPECTED_GPU_INDEX_BY_UUID[$uuid]+present} ]]; then
      printf 'ERROR: duplicate B300 UUID in inventory: %s\n' "$uuid" >&2
      return 2
    fi
    EXPECTED_GPU_UUID_BY_INDEX[$index]=$uuid
    EXPECTED_GPU_INDEX_BY_UUID[$uuid]=$index
    count=$((count + 1))
  done <<< "$snapshot"
  if [[ "$count" != 8 ]]; then
    printf 'ERROR: expected eight B300 index/UUID rows, observed=%s\n' \
      "$count" >&2
    return 2
  fi
  for index in $(seq 0 7); do
    if [[ ! ${EXPECTED_GPU_UUID_BY_INDEX[$index]+present} ]]; then
      printf 'ERROR: B300 inventory is missing logical index=%s\n' \
        "$index" >&2
      return 2
    fi
  done
  printf 'accelerator_uuid_inventory_pass indices=0..7 uuids=%s\n' \
    "$(for index in $(seq 0 7); do printf '%s:%s ' \
      "$index" "${EXPECTED_GPU_UUID_BY_INDEX[$index]}"; done)"
}

verify_holder_process_identity() {
  local candidate_pid=$1
  local holder_mode=$2
  local candidate_exe=""
  local expected_exe=""
  local candidate_cwd=""
  local candidate_pgid=""
  local candidate_sid=""
  local candidate_args=""
  HOLDER_VERIFY_REASON=""
  if [[ ! -r "/proc/$candidate_pid/cmdline" ]]; then
    HOLDER_VERIFY_REASON="holder pid is absent"
    return 1
  fi
  if ! candidate_exe=$(readlink -f "/proc/$candidate_pid/exe") \
     || ! expected_exe=$(readlink -f "$holder_python") \
     || ! candidate_cwd=$(readlink -f "/proc/$candidate_pid/cwd") \
     || ! candidate_pgid=$(ps -o pgid= -p "$candidate_pid" 2>/dev/null) \
     || ! candidate_sid=$(ps -o sid= -p "$candidate_pid" 2>/dev/null); then
    HOLDER_VERIFY_REASON="cannot read holder process identity"
    return 1
  fi
  candidate_pgid=${candidate_pgid//[[:space:]]/}
  candidate_sid=${candidate_sid//[[:space:]]/}
  if ! candidate_args=$(tr '\0' '\n' < "/proc/$candidate_pid/cmdline" \
    | sed '1d'); then
    HOLDER_VERIFY_REASON="cannot read holder argv"
    return 1
  fi
  if [[ "$candidate_exe" != "$expected_exe" \
        || "$candidate_cwd" != "$holder_cwd" \
        || "$candidate_pgid" != "$candidate_pid" \
        || "$candidate_sid" != "$candidate_pid" ]]; then
    HOLDER_VERIFY_REASON="holder exe/cwd/pgid/sid differs"
    return 1
  fi
  case "$holder_mode" in
    full)
      if [[ "$candidate_args" != "gpu_hold.py" ]]; then
        HOLDER_VERIFY_REASON="full holder argv is not the canonical no-argument form"
        return 1
      fi
      ;;
    bootstrap)
      if [[ "$candidate_args" != $'gpu_hold.py\n--mem-frac\n0.08\n--size\n8192' ]]; then
        HOLDER_VERIFY_REASON="bootstrap holder argv differs"
        return 1
      fi
      ;;
    *)
      HOLDER_VERIFY_REASON="unknown holder mode"
      return 1
      ;;
  esac
  return 0
}

# Verify one canonical holder worker PID on every expected GPU UUID/index.
# allow_extra=1 is used only while authenticated server probes may coexist with
# the bootstrap holder; the holder's own eight-PID mapping remains mandatory.
verify_gpu_group_coverage_once() {
  local expected_pgid=$1
  local allow_extra=${2:-0}
  local snapshot=""
  local uuid=""
  local pid=""
  local extra=""
  local observed_pgid=""
  local index=""
  declare -A holder_pid_by_uuid=()
  declare -A holder_uuid_by_pid=()
  GPU_VERIFY_REASON=""
  GPU_VERIFY_MAPPING=""
  if ! snapshot=$(gpu_compute_snapshot); then
    GPU_VERIFY_REASON="compute-app telemetry failed after bounded retries"
    return 2
  fi
  while IFS=',' read -r uuid pid extra; do
    [[ -z "$uuid$pid$extra" ]] && continue
    trim_field uuid
    trim_field pid
    trim_field extra
    if [[ -n "$extra" || -z "$uuid" \
          || ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
      GPU_VERIFY_REASON="malformed or unknown compute-app row"
      return 2
    fi
    if [[ ! ${EXPECTED_GPU_INDEX_BY_UUID[$uuid]+present} ]]; then
      GPU_VERIFY_REASON="compute-app row names an unknown GPU UUID"
      return 2
    fi
    if ! observed_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null); then
      GPU_VERIFY_REASON="compute-app PID vanished during coverage snapshot"
      return 1
    fi
    observed_pgid=${observed_pgid//[[:space:]]/}
    if [[ "$observed_pgid" == "$expected_pgid" ]]; then
      if [[ ${holder_pid_by_uuid[$uuid]+present} \
            && "${holder_pid_by_uuid[$uuid]}" != "$pid" ]]; then
        GPU_VERIFY_REASON="multiple holder PIDs map to one GPU UUID"
        return 1
      fi
      if [[ ${holder_uuid_by_pid[$pid]+present} \
            && "${holder_uuid_by_pid[$pid]}" != "$uuid" ]]; then
        GPU_VERIFY_REASON="one holder PID maps to multiple GPU UUIDs"
        return 1
      fi
      holder_pid_by_uuid[$uuid]=$pid
      holder_uuid_by_pid[$pid]=$uuid
    elif [[ "$allow_extra" != 1 ]]; then
      GPU_VERIFY_REASON="unexpected GPU process group overlaps the holder"
      return 1
    fi
  done <<< "$snapshot"
  if [[ ${#holder_uuid_by_pid[@]} -ne 8 ]]; then
    GPU_VERIFY_REASON="holder does not expose exactly eight unique GPU worker PIDs"
    return 1
  fi
  for index in $(seq 0 7); do
    uuid=${EXPECTED_GPU_UUID_BY_INDEX[$index]}
    if [[ ! ${holder_pid_by_uuid[$uuid]+present} ]]; then
      GPU_VERIFY_REASON="holder is missing a physical GPU UUID"
      return 1
    fi
    GPU_VERIFY_MAPPING+="${index}:${uuid}:${holder_pid_by_uuid[$uuid]} "
  done
  GPU_VERIFY_MAPPING=${GPU_VERIFY_MAPPING% }
  return 0
}

verify_holder_utilization() {
  local label=$1
  local snapshot=""
  local sample=0
  local good_samples=0
  local index=""
  local uuid=""
  local utilization=""
  local extra=""
  local count=0
  local all_high=0
  declare -A seen_indices=()
  for ((sample = 1; sample <= HOLDER_UTIL_MAX_SAMPLES; sample += 1)); do
    if ! snapshot=$(gpu_inventory_snapshot); then
      GPU_VERIFY_REASON="utilization telemetry failed after bounded retries"
      return 2
    fi
    seen_indices=()
    count=0
    all_high=1
    while IFS=',' read -r index uuid utilization extra; do
      [[ -z "$index$uuid$utilization$extra" ]] && continue
      trim_field index
      trim_field uuid
      trim_field utilization
      trim_field extra
      if [[ -n "$extra" || ! "$index" =~ ^[0-7]$ \
            || -z "$uuid" || ! "$utilization" =~ ^[0-9]+$ ]]; then
        GPU_VERIFY_REASON="malformed utilization inventory"
        return 2
      fi
      if [[ ! ${EXPECTED_GPU_UUID_BY_INDEX[$index]+present} \
            || "${EXPECTED_GPU_UUID_BY_INDEX[$index]}" != "$uuid" \
            || ${seen_indices[$index]+present} ]]; then
        GPU_VERIFY_REASON="utilization inventory identity differs"
        return 2
      fi
      seen_indices[$index]=1
      count=$((count + 1))
      if (( utilization < HOLDER_UTIL_MIN_PERCENT || utilization > 100 )); then
        all_high=0
      fi
    done <<< "$snapshot"
    if [[ "$count" != 8 ]]; then
      GPU_VERIFY_REASON="utilization inventory does not cover eight GPUs"
      return 2
    fi
    if [[ "$all_high" == 1 ]]; then
      good_samples=$((good_samples + 1))
      if (( good_samples >= HOLDER_UTIL_REQUIRED_SAMPLES )); then
        printf 'holder_utilization_verified label=%s threshold_pct=%s good_samples=%s max_samples=%s\n' \
          "$label" "$HOLDER_UTIL_MIN_PERCENT" "$good_samples" \
          "$HOLDER_UTIL_MAX_SAMPLES"
        return 0
      fi
    fi
    if (( sample < HOLDER_UTIL_MAX_SAMPLES )); then
      sleep "$HOLDER_UTIL_SAMPLE_DELAY_S"
    fi
  done
  GPU_VERIFY_REASON="holder utilization did not reach the stable threshold"
  return 1
}

wait_for_holder_ready() {
  local candidate_pid=$1
  local holder_mode=$2
  local allow_extra=$3
  local label=$4
  local max_attempts=$5
  local attempt=0
  local coverage_status=0
  for ((attempt = 1; attempt <= max_attempts; attempt += 1)); do
    if verify_holder_process_identity "$candidate_pid" "$holder_mode"; then
      verify_gpu_group_coverage_once "$candidate_pid" "$allow_extra"
      coverage_status=$?
      if (( coverage_status == 0 )); then
        printf 'holder_gpu_coverage_verified label=%s mapping=%s\n' \
          "$label" "$GPU_VERIFY_MAPPING"
        verify_holder_utilization "$label"
        coverage_status=$?
        if (( coverage_status == 0 )); then
          return 0
        fi
        return "$coverage_status"
      fi
      if (( coverage_status == 2 )); then
        return 2
      fi
    fi
    if (( attempt < max_attempts )); then
      sleep 1
    fi
  done
  GPU_VERIFY_REASON=${GPU_VERIFY_REASON:-${HOLDER_VERIFY_REASON:-holder readiness timed out}}
  return 1
}

verify_worker_gpu_coverage_once() {
  local expected_pids=$1
  local snapshot=""
  local uuid=""
  local pid=""
  local extra=""
  local index=""
  declare -A expected_pid=()
  declare -A worker_pid_by_uuid=()
  declare -A worker_uuid_by_pid=()
  GPU_VERIFY_REASON=""
  GPU_VERIFY_MAPPING=""
  for pid in $expected_pids; do
    expected_pid[$pid]=1
  done
  if [[ ${#expected_pid[@]} -ne 8 ]]; then
    GPU_VERIFY_REASON="worker identity set does not contain eight PIDs"
    return 2
  fi
  if ! snapshot=$(gpu_compute_snapshot); then
    GPU_VERIFY_REASON="worker compute-app telemetry failed after bounded retries"
    return 2
  fi
  while IFS=',' read -r uuid pid extra; do
    [[ -z "$uuid$pid$extra" ]] && continue
    trim_field uuid
    trim_field pid
    trim_field extra
    if [[ -n "$extra" || -z "$uuid" \
          || ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
      GPU_VERIFY_REASON="malformed worker compute-app row"
      return 2
    fi
    if [[ ! ${EXPECTED_GPU_INDEX_BY_UUID[$uuid]+present} ]]; then
      GPU_VERIFY_REASON="worker compute-app row names an unknown GPU UUID"
      return 2
    fi
    if [[ ! ${expected_pid[$pid]+present} ]]; then
      GPU_VERIFY_REASON="unexpected GPU process remains after holder release"
      return 1
    fi
    if [[ ${worker_pid_by_uuid[$uuid]+present} \
          && "${worker_pid_by_uuid[$uuid]}" != "$pid" ]]; then
      GPU_VERIFY_REASON="multiple TP workers map to one GPU UUID"
      return 1
    fi
    if [[ ${worker_uuid_by_pid[$pid]+present} \
          && "${worker_uuid_by_pid[$pid]}" != "$uuid" ]]; then
      GPU_VERIFY_REASON="one TP worker maps to multiple GPU UUIDs"
      return 1
    fi
    worker_pid_by_uuid[$uuid]=$pid
    worker_uuid_by_pid[$pid]=$uuid
  done <<< "$snapshot"
  if [[ ${#worker_uuid_by_pid[@]} -ne 8 ]]; then
    GPU_VERIFY_REASON="not all eight TP workers have entered CUDA"
    return 1
  fi
  for index in $(seq 0 7); do
    uuid=${EXPECTED_GPU_UUID_BY_INDEX[$index]}
    if [[ ! ${worker_pid_by_uuid[$uuid]+present} ]]; then
      GPU_VERIFY_REASON="TP workers do not cover every physical GPU UUID"
      return 1
    fi
    GPU_VERIFY_MAPPING+="${index}:${uuid}:${worker_pid_by_uuid[$uuid]} "
  done
  GPU_VERIFY_MAPPING=${GPU_VERIFY_MAPPING% }
  return 0
}

if ! initialize_expected_gpu_inventory; then
  exit 90
fi

verify_server_identity() {
  local require_live=${1:-1}
  local identity_record=""
  local identity_rc=0
  local extra_field=""
  identity_record=$(
    "$runtime_python" - \
      "$server_identity_path" \
      "$server_identity_nonce" \
      "$repo/server/start_server_redknot_pro0813.sh" \
      "$runtime_python" \
      "/workspace/Models/DeepSeek-V4-Pro-0813" \
      "$server_port" \
      "$require_live" <<'PY'
import json
import os
import re
import stat
import sys
from pathlib import Path

(
    identity_name,
    expected_nonce,
    expected_launcher,
    expected_runtime,
    expected_model,
    expected_port_raw,
    require_live_raw,
) = sys.argv[1:]
expected_port = int(expected_port_raw)
require_live = require_live_raw == "1"
schema = "redknot_pro0813_owned_server_identity_v1"
keys = {
    "schema",
    "nonce",
    "pid",
    "pgid",
    "sid",
    "starttime_ticks",
    "launcher",
    "model_path",
    "port",
}


def fail(message, status=20):
    print(f"ERROR: unsafe Pro-0813 server identity: {message}", file=sys.stderr)
    raise SystemExit(status)


def proc_stat(pid):
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        fail(f"malformed /proc/{pid}/stat")
    fields = raw[close + 1 :].split()
    if len(fields) <= 19:
        fail(f"truncated /proc/{pid}/stat")
    try:
        return fields[0], int(fields[2]), int(fields[3]), int(fields[19])
    except (TypeError, ValueError):
        fail(f"invalid /proc/{pid}/stat identity")


def group_has_live_members(pgid):
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as error:
        fail(f"cannot inspect /proc: {error}")
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            state, process_group, _, _ = proc_stat(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if process_group == pgid and state not in {"Z", "X", "x"}:
            return True
    return False


def has_exact_pair(argv, flag, value):
    values = tuple(
        argv[index + 1]
        for index in range(len(argv) - 1)
        if argv[index] == flag
    )
    return values == (value,) and not any(
        item.startswith(f"{flag}=") for item in argv
    )


def is_certified_server_cmdline(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = tuple(
            item.decode("utf-8", errors="strict")
            for item in raw.rstrip(b"\0").split(b"\0")
            if item
        )
        executable = os.path.realpath(f"/proc/{pid}/exe")
    except (OSError, UnicodeDecodeError) as error:
        fail(f"cannot read current server command: {error}")
    if len(argv) == 2 and Path(argv[0]).name == "bash":
        return bool(
            executable == os.path.realpath("/usr/bin/bash")
            and os.path.realpath(argv[1]) == os.path.realpath(expected_launcher)
        )
    return (
        executable == os.path.realpath(expected_runtime)
        and has_exact_pair(argv, "-m", "sglang.launch_server")
        and has_exact_pair(argv, "--model-path", expected_model)
        and has_exact_pair(argv, "--attention-backend", "redknot_mla")
        and has_exact_pair(argv, "--tp-size", "8")
        and has_exact_pair(argv, "--port", str(expected_port))
    )


path = Path(identity_name)
if not path.is_absolute():
    fail("identity path is not absolute")
try:
    before = os.lstat(path)
except FileNotFoundError:
    raise SystemExit(10)
except OSError as error:
    fail(f"cannot lstat identity: {error}")
if not stat.S_ISREG(before.st_mode):
    fail("identity is a symlink or non-regular file")
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(path, flags)
except OSError as error:
    fail(f"cannot open identity without following links: {error}")
try:
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or after.st_nlink != 1
        or after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) != 0o600
        or not 0 < after.st_size <= 4096
    ):
        fail("identity file metadata is unsafe")
    raw = b""
    while len(raw) <= 4096:
        chunk = os.read(descriptor, min(4097 - len(raw), 4096))
        if not chunk:
            break
        raw += chunk
    if len(raw) != after.st_size or len(raw) > 4096:
        fail("identity changed while being read")
finally:
    os.close(descriptor)
def no_duplicate_keys(pairs):
    parsed = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON key {key!r}")
        parsed[key] = value
    return parsed


try:
    identity = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=no_duplicate_keys,
    )
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    fail(f"malformed identity JSON: {error}")
if not isinstance(identity, dict) or set(identity) != keys:
    fail("identity JSON schema differs")
expected = {
    "schema": schema,
    "nonce": expected_nonce,
    "launcher": expected_launcher,
    "model_path": expected_model,
    "port": expected_port,
}
for key, value in expected.items():
    if type(identity.get(key)) is not type(value) or identity.get(key) != value:
        fail(f"identity {key} differs")
for key in ("pid", "pgid", "sid", "starttime_ticks"):
    if type(identity.get(key)) is not int or identity[key] <= 0:
        fail(f"identity {key} is invalid")
pid = identity["pid"]
pgid = identity["pgid"]
sid = identity["sid"]
starttime = identity["starttime_ticks"]
if pid <= 1 or not pid == pgid == sid:
    fail("identity does not describe a dedicated process group and session")
if not re.fullmatch(r"[0-9a-f]{64}", expected_nonce):
    fail("supervisor nonce is malformed")
try:
    state, current_pgid, current_sid, current_starttime = proc_stat(pid)
except FileNotFoundError:
    if group_has_live_members(pgid):
        if require_live:
            fail("session leader vanished while its process group remains live")
        print("ORPHAN", pid, pgid, sid, starttime, after.st_dev, after.st_ino)
        raise SystemExit(0)
    if require_live:
        fail("server is no longer live")
    print("DEAD", pid, pgid, sid, starttime, after.st_dev, after.st_ino)
    raise SystemExit(0)
if state in {"Z", "X", "x"}:
    if group_has_live_members(pgid):
        if require_live:
            fail("dead session leader still has live group members")
        print("ORPHAN", pid, pgid, sid, starttime, after.st_dev, after.st_ino)
        raise SystemExit(0)
    if require_live:
        fail("server leader is dead")
    print("DEAD", pid, pgid, sid, starttime, after.st_dev, after.st_ino)
    raise SystemExit(0)
if (
    current_pgid != pgid
    or current_sid != sid
    or current_starttime != starttime
):
    fail("live PID/session/starttime differs (possible PID reuse)")
if not is_certified_server_cmdline(pid):
    fail("current process command is not the certified Pro-0813 server")
print("LIVE", pid, pgid, sid, starttime, after.st_dev, after.st_ino)
PY
  )
  identity_rc=$?
  if (( identity_rc == 10 )); then
    return 10
  fi
  if (( identity_rc != 0 )); then
    return 20
  fi
  IFS=' ' read -r \
    server_identity_state \
    server_pid \
    server_pgid \
    server_sid \
    server_starttime \
    server_identity_dev \
    server_identity_ino \
    extra_field <<< "$identity_record"
  if [[ -n "$extra_field" \
        || ( "$server_identity_state" != LIVE \
             && "$server_identity_state" != ORPHAN \
             && "$server_identity_state" != DEAD ) \
        || ! "$server_pid" =~ ^[1-9][0-9]*$ \
        || "$server_pid" != "$server_pgid" \
        || "$server_pid" != "$server_sid" \
        || ! "$server_starttime" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: malformed server identity verifier output: %q\n' \
      "$identity_record" >&2
    return 20
  fi
  if [[ "$require_live" == 1 && "$server_identity_state" != LIVE ]]; then
    printf '%s\n' 'ERROR: expected a live authenticated Pro-0813 server' >&2
    return 20
  fi
  server_identity_verified=1
  return 0
}

server_process_group_alive() {
  "$runtime_python" - "$server_pgid" <<'PY'
import sys
from pathlib import Path

pgid = int(sys.argv[1])
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        raw = (entry / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 1 :].split()
        state = fields[0]
        process_group = int(fields[2])
    except (IndexError, OSError, ValueError):
        continue
    if process_group == pgid and state not in {"Z", "X", "x"}:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

stop_authenticated_server() {
  local verify_status=0
  local deadline=0
  if [[ ! -e "$server_identity_path" && ! -L "$server_identity_path" ]]; then
    if [[ "$server_identity_verified" == 1 ]]; then
      printf '%s\n' \
        'ERROR: authenticated Pro-0813 server identity disappeared' >&2
      return 100
    fi
    return 0
  fi
  verify_server_identity 0 || verify_status=$?
  if (( verify_status != 0 )); then
    printf 'ERROR: refusing unauthenticated server cleanup status=%s\n' \
      "$verify_status" >&2
    return 100
  fi
  if [[ "$server_identity_state" == DEAD ]]; then
    return 0
  fi
  # verify_server_identity checked nonce, inode, PID=start-of-session PGID/SID,
  # starttime and current Pro launcher command immediately before this signal.
  if ! kill -TERM -- "-$server_pgid" 2>/dev/null; then
    if server_process_group_alive; then
      printf 'ERROR: failed to terminate authenticated server pgid=%s\n' \
        "$server_pgid" >&2
      return 100
    fi
    return 0
  fi
  deadline=$((SECONDS + 30))
  while server_process_group_alive && (( SECONDS < deadline )); do
    sleep 1
  done
  if server_process_group_alive; then
    # The PGID cannot be recycled while a member of this exact group remains.
    # It was authenticated immediately before SIGTERM, so SIGKILL is still
    # restricted to that same extant process group.
    kill -KILL -- "-$server_pgid" 2>/dev/null || true
    deadline=$((SECONDS + 10))
    while server_process_group_alive && (( SECONDS < deadline )); do
      sleep 1
    done
  fi
  if server_process_group_alive; then
    printf 'ERROR: authenticated server group survived SIGKILL pgid=%s\n' \
      "$server_pgid" >&2
    return 100
  fi
  return 0
}

restart_holder() {
  local server_cleanup_status=0
  if [[ "$holder_restarted" == 1 ]]; then
    return 0
  fi
  if [[ -n "$heartbeat_pid" ]]; then
    kill -TERM "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=""
  fi
  stop_authenticated_server || server_cleanup_status=$?
  if [[ -n "$benchmark_pgid" ]]; then
    kill -TERM -- "-$benchmark_pgid" 2>/dev/null || true
  fi
  if [[ "$bootstrap_holder_active" == 1 && -n "$holder_pgid" ]]; then
    kill -TERM -- "-$holder_pgid" 2>/dev/null || true
    bootstrap_holder_active=0
  fi
  if [[ "$holder_stopped" != 1 ]]; then
    # A failure before the handoff still needs an authenticated final-state
    # receipt.  This lets the five-target driver distinguish "holder was
    # retained and re-proved" from an unobserved early exit before attempting
    # the next target.
    if ! wait_for_holder_ready "$holder_pid" full 0 full_retained 15; then
      printf 'ERROR: retained holder final verification failed reason=%s holder_reason=%s\n' \
        "${GPU_VERIFY_REASON:-unknown}" "${HOLDER_VERIFY_REASON:-unknown}"
      return 100
    fi
    if ! printf 'holder_retained pid=%s workers=8 util_threshold_pct=%s util_good_samples=%s/%s mapping=%s\n' \
      "$holder_pid" "$HOLDER_UTIL_MIN_PERCENT" \
      "$HOLDER_UTIL_REQUIRED_SAMPLES" "$HOLDER_UTIL_MAX_SAMPLES" \
      "$GPU_VERIFY_MAPPING" \
      | tee "$run_dir/holder_restore_status"; then
      printf '%s\n' 'ERROR: retained holder verified but status write failed'
      return 100
    fi
    holder_restarted=1
    return "$server_cleanup_status"
  fi
  local active_gpu_pids=""
  local new_holder=""
  for _ in $(seq 1 120); do
    if ! active_gpu_pids=$(gpu_pids); then
      printf '%s\n' \
        'ERROR: cannot prove GPUs are idle during holder restore; refusing overlap'
      return 100
    fi
    [[ -z "$active_gpu_pids" ]] && break
    sleep 1
  done
  if [[ -n "$active_gpu_pids" ]]; then
    printf 'ERROR: refusing to overlap holder with residual GPU pids: %s\n' \
      "$active_gpu_pids"
    return 100
  fi
  if ! cd "$holder_cwd"; then
    printf 'ERROR: cannot enter holder cwd during restore: %s\n' \
      "$holder_cwd"
    return 100
  fi
  setsid nohup "$holder_python" gpu_hold.py \
    > "$run_dir/gpu_holder.log" 2>&1 < /dev/null &
  new_holder=$!
  if wait_for_holder_ready "$new_holder" full 0 full_restore 90; then
    if ! printf 'holder_restarted pid=%s workers=8 util_threshold_pct=%s util_good_samples=%s/%s mapping=%s\n' \
      "$new_holder" "$HOLDER_UTIL_MIN_PERCENT" \
      "$HOLDER_UTIL_REQUIRED_SAMPLES" "$HOLDER_UTIL_MAX_SAMPLES" \
      "$GPU_VERIFY_MAPPING" \
      | tee "$run_dir/holder_restore_status"; then
      printf '%s\n' 'ERROR: holder restored but status write failed'
      return 100
    fi
    holder_restarted=1
    if (( server_cleanup_status != 0 )); then
      return "$server_cleanup_status"
    fi
    return 0
  fi
  printf 'holder_restore_failed pid=%s\n' "$new_holder" \
    | tee "$run_dir/holder_restore_status"
  printf 'ERROR: holder restore verification failed reason=%s holder_reason=%s\n' \
    "${GPU_VERIFY_REASON:-unknown}" "${HOLDER_VERIFY_REASON:-unknown}"
  return 100
}

finalize_supervisor() {
  local original_status=$?
  local restore_status=0
  local final_status=$original_status
  trap - EXIT
  trap '' INT TERM
  restart_holder || restore_status=$?
  if (( restore_status != 0 )); then
    printf 'ERROR: holder cleanup failed restore_status=%s original_status=%s\n' \
      "$restore_status" "$original_status"
    # Holder safety outranks the benchmark's original error.  In particular,
    # callers must never mistake an arbitrary benchmark failure for a
    # successful cleanup and continue a multi-target sweep on unowned GPUs.
    final_status=$restore_status
  fi
  exit "$final_status"
}

exit_for_signal() {
  local signal_name=$1
  local signal_status=1
  case "$signal_name" in
    INT) signal_status=130 ;;
    TERM) signal_status=143 ;;
  esac
  printf 'received_%s exiting_status=%s\n' "$signal_name" "$signal_status"
  exit "$signal_status"
}

# Bash dispatches traps only between commands.  Defer signals across the few
# spawn/kill bookkeeping sequences so EXIT cleanup never observes a live
# process whose PID/PGID ownership fields have not yet been recorded.
pending_transition_signal=""
defer_transition_signal() {
  if [[ -z "$pending_transition_signal" ]]; then
    pending_transition_signal=$1
  fi
}
begin_signal_transition() {
  pending_transition_signal=""
  trap 'defer_transition_signal INT' INT
  trap 'defer_transition_signal TERM' TERM
}
end_signal_transition() {
  local deferred_signal=$pending_transition_signal
  pending_transition_signal=""
  trap 'exit_for_signal INT' INT
  trap 'exit_for_signal TERM' TERM
  if [[ -n "$deferred_signal" ]]; then
    exit_for_signal "$deferred_signal"
  fi
}

trap finalize_supervisor EXIT
trap 'exit_for_signal INT' INT
trap 'exit_for_signal TERM' TERM

if [[ ! -r "/proc/$holder_pid/cmdline" ]]; then
  printf 'ERROR: holder pid is absent: %s\n' "$holder_pid"
  exit 90
fi
holder_cmd=$(tr '\0' ' ' < "/proc/$holder_pid/cmdline")
if ! verify_holder_process_identity "$holder_pid" full; then
  printf 'ERROR: holder identity mismatch pid=%s reason=%s cmd=%q\n' \
    "$holder_pid" "$HOLDER_VERIFY_REASON" "$holder_cmd"
  exit 91
fi
holder_pgid=$holder_pid
if ! wait_for_holder_ready "$holder_pid" full 0 original_full_holder 15; then
  printf 'ERROR: original full holder failed 8-GPU coverage/utilization gate reason=%s holder_reason=%s\n' \
    "${GPU_VERIFY_REASON:-unknown}" "${HOLDER_VERIFY_REASON:-unknown}"
  exit 91
fi
printf 'holder_retained_during_cpu_prelaunch pid=%s pgid=%s mapping=%s\n' \
  "$holder_pid" "$holder_pgid" "$GPU_VERIFY_MAPPING"

# A full-memory holder can block CUDA work performed indirectly by plugin
# imports before the per-worker release hook is reached.  Replace it with a
# compute-active, low-memory bootstrap holder before spawning the server.  GPU
# utilization stays high, while each model worker still has enough memory for
# incidental early CUDA initialization.  EXIT cleanup restores the canonical
# full holder only after it can prove no residual GPU process would overlap it.
printf 'switching_to_light_bootstrap_holder old_pid=%s old_pgid=%s\n' \
  "$holder_pid" "$holder_pgid"
begin_signal_transition
holder_stopped=1
if ! kill -TERM -- "-$holder_pgid"; then
  holder_stopped=0
  end_signal_transition
  printf 'ERROR: failed to stop full holder pgid=%s\n' "$holder_pgid"
  exit 98
fi
end_signal_transition
active_gpu_pids=""
for _ in $(seq 1 90); do
  if ! active_gpu_pids=$(gpu_pids); then
    printf '%s\n' \
      'ERROR: cannot prove GPUs are idle before bootstrap holder; refusing overlap'
    exit 98
  fi
  [[ -z "$active_gpu_pids" ]] && break
  sleep 1
done
if [[ -n "$active_gpu_pids" ]]; then
  printf 'ERROR: GPUs did not become free before bootstrap holder: %s\n' \
    "$active_gpu_pids"
  exit 98
fi
cd "$holder_cwd" || exit 92
begin_signal_transition
setsid nohup "$holder_python" gpu_hold.py --mem-frac 0.08 --size 8192 \
  > "$run_dir/gpu_bootstrap_holder.log" 2>&1 < /dev/null &
holder_pid=$!
holder_pgid=$holder_pid
bootstrap_holder_active=1
end_signal_transition
if ! wait_for_holder_ready "$holder_pid" bootstrap 0 light_bootstrap_holder 90; then
  printf 'ERROR: light bootstrap holder verification failed pid=%s reason=%s holder_reason=%s\n' \
    "$holder_pid" "${GPU_VERIFY_REASON:-unknown}" \
    "${HOLDER_VERIFY_REASON:-unknown}"
  exit 99
fi
printf 'light_bootstrap_holder_ready pid=%s workers=8 mem_frac=0.08 util_threshold_pct=%s util_good_samples=%s/%s mapping=%s\n' \
  "$holder_pid" "$HOLDER_UTIL_MIN_PERCENT" "$HOLDER_UTIL_REQUIRED_SAMPLES" \
  "$HOLDER_UTIL_MAX_SAMPLES" "$GPU_VERIFY_MAPPING"

cd "$repo" || exit 92
set +e
printf '%s\n' \
  'benchmark_starting: live output is mirrored to driver.log' \
  'cold_start_note: the first exact long-context snapshot may spend several minutes in host-side preparation and CUDA/Triton compilation; this offline snapshot time is not counted as online TTFT'
# Histogram profiling launches reductions and emits a large JSON object for
# every MoE layer.  It is an offline calibration tool, not part of the
# physical adaptive-TopK serving path, so keep it out of TTFT/QPS runs.
begin_signal_transition
setsid env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$repo/python:$flashmla_sm103_root" \
  PYTHONNOUSERSITE=1 \
  PYTHONSAFEPATH=1 \
  REDKNOT_DSV4_VARIANT=pro0813 \
  REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL=triton_h1 \
  REDKNOT_ADAPTIVE_TOPK_PROFILE=0 \
  REDKNOT_V4_TIMING="$timing_mode" \
  REDKNOT_GPU_HOLDER_WORKER_READY_DIR="$worker_ready_dir" \
  REDKNOT_GPU_HOLDER_WORKER_GO_FILE="$worker_go_file" \
  REDKNOT_GPU_HOLDER_EXPECTED_WORKERS=8 \
  REDKNOT_GPU_HOLDER_RELEASE_TIMEOUT_S=300 \
  REDKNOT_IH_SERVER_IDENTITY_PATH="$server_identity_path" \
  REDKNOT_IH_SERVER_IDENTITY_NONCE="$server_identity_nonce" \
  REDKNOT_IH_SERVER_READY_TIMEOUT_S=1800 \
  REDKNOT_PRO0813_SERVER_PORT="$server_port" \
  "$runtime_python" test/srt/redknot/utils/benchmark_dsv4_pro0813_redknot_http.py \
    --mode redknot \
    --port "$server_port" \
    --row-sparse-active-ratio "$row_sparse_active_ratio" \
    --row-sparse-checkpoint-max-islands "$checkpoint_max_islands" \
    --generalized-strong-active-ratio "$generalized_strong_ratio" \
    --generalized-medium-active-ratio "$generalized_medium_ratio" \
    --generalized-diffuse-active-ratio "$generalized_diffuse_ratio" \
    --query-protection-tokens "$query_protection_tokens" \
    --target-tokens "$target_tokens" \
    --max-new 64 \
    --progressive-topk-schedule '' \
    --quality-repeats "$quality_repeats" \
    --ttft-warmup "$ttft_warmup" \
    --ttft-iters "$ttft_iters" \
    --merged-prefill-tokens "$merged_prefill_tokens" \
    --mem-fraction-static "$mem_fraction_static" \
    --qps-concurrencies "$qps_concurrencies" \
    "${benchmark_extra_args[@]}" \
    --out "$run_dir/result.json" \
    --log "$run_dir/server.log" \
    > >(tee -a "$run_dir/driver.log") 2>&1 &
benchmark_pid=$!
benchmark_pgid=$benchmark_pid
end_signal_transition

barrier_ready=0
identity_failure=0
holder_failure=0
for _ in $(seq 1 12000); do
  if (( (_ - 1) % 10 == 0 )) \
     && ! verify_holder_process_identity "$holder_pid" bootstrap; then
    printf 'ERROR: bootstrap holder identity failed during worker barrier pid=%s reason=%s\n' \
      "$holder_pid" "$HOLDER_VERIFY_REASON"
    holder_failure=1
    break
  fi
  if (( (_ - 1) % 100 == 0 )); then
    verify_gpu_group_coverage_once "$holder_pgid" 1
    holder_coverage_status=$?
    if (( holder_coverage_status != 0 )); then
      printf 'ERROR: bootstrap holder GPU coverage failed during worker barrier status=%s reason=%s\n' \
        "$holder_coverage_status" "$GPU_VERIFY_REASON"
      holder_failure=1
      break
    fi
  fi
  if [[ "$server_identity_verified" != 1 \
        && ( -e "$server_identity_path" || -L "$server_identity_path" ) ]]; then
    if ! verify_server_identity 1; then
      identity_failure=1
      break
    fi
    printf 'server_identity_authenticated pid=%s pgid=%s sid=%s starttime=%s\n' \
      "$server_pid" "$server_pgid" "$server_sid" "$server_starttime"
  fi
  ready_count=$(find "$worker_ready_dir" -maxdepth 1 -type f \
    -name 'rank[0-9][0-9][0-9].ready' 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$ready_count" == 8 ]]; then
    if [[ "$server_identity_verified" == 1 ]]; then
      barrier_ready=1
    else
      printf '%s\n' \
        'ERROR: TP workers appeared before an authenticated server identity'
      identity_failure=1
    fi
    break
  fi
  if ! kill -0 "$benchmark_pid" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if [[ "$barrier_ready" != 1 ]]; then
  if [[ "$holder_failure" == 1 ]]; then
    printf '%s\n' 97 > "$run_dir/exit_code"
    exit 97
  fi
  if [[ "$identity_failure" == 1 ]]; then
    printf '%s\n' 94 > "$run_dir/exit_code"
    exit 94
  fi
  wait "$benchmark_pid"
  benchmark_exit=$?
  benchmark_pid=""
  benchmark_pgid=""
  printf '%s\n' "$benchmark_exit" > "$run_dir/exit_code"
  printf 'benchmark_exited_before_gpu_release exit=%s\n' "$benchmark_exit"
  exit "$benchmark_exit"
fi
worker_pids=""
for rank in $(seq 0 7); do
  ready_path=$(printf '%s/rank%03d.ready' "$worker_ready_dir" "$rank")
  if [[ ! -f "$ready_path" ]]; then
    printf 'ERROR: missing worker ready file rank=%s\n' "$rank"
    exit 93
  fi
  ready_line=$(tr -d '\r\n' < "$ready_path")
  worker_pid=$(printf '%s\n' "$ready_line" | sed -n 's/^pid=\([0-9][0-9]*\) .*/\1/p')
  # Each TP subprocess receives a rank-local CUDA_VISIBLE_DEVICES and therefore
  # reports local gpu_id=0.  Global ownership is certified by unique tp_rank,
  # unique PID, a common server PGID, and the later 8-device nvidia-smi check.
  expected_tail=$(printf 'gpu_id=0 tp_rank=%s tp_size=8' "$rank")
  if [[ -z "$worker_pid" || "$ready_line" != "pid=$worker_pid $expected_tail" \
        || ! -r "/proc/$worker_pid/stat" ]]; then
    printf 'ERROR: invalid worker ready record rank=%s record=%q\n' \
      "$rank" "$ready_line"
    exit 94
  fi
  this_pgid=$(ps -o pgid= -p "$worker_pid" 2>/dev/null | tr -d ' ')
  if [[ "$this_pgid" != "$server_pgid" ]]; then
    printf 'ERROR: worker process groups differ rank=%s pgid=%s expected=%s\n' \
      "$rank" "$this_pgid" "$server_pgid"
    exit 95
  fi
  worker_pids+="$worker_pid "
done
unique_worker_count=$(printf '%s\n' $worker_pids | sed '/^$/d' | sort -n -u | wc -l | tr -d ' ')
if [[ "$unique_worker_count" != 8 || "$server_pgid" == "$holder_pgid" ]]; then
  printf 'ERROR: worker identity set is invalid workers=%s server_pgid=%s\n' \
    "$worker_pids" "$server_pgid"
  exit 96
fi
if ! verify_server_identity 1; then
  printf '%s\n' \
    'ERROR: Pro-0813 server identity changed before GPU release'
  exit 96
fi
if ! verify_holder_process_identity "$holder_pid" bootstrap; then
  printf 'ERROR: holder identity changed before GPU release pid=%s reason=%s\n' \
    "$holder_pid" "$HOLDER_VERIFY_REASON"
  exit 97
fi
verify_gpu_group_coverage_once "$holder_pgid" 1
holder_coverage_status=$?
if (( holder_coverage_status != 0 )); then
  printf 'ERROR: holder GPU coverage changed before release status=%s reason=%s\n' \
    "$holder_coverage_status" "$GPU_VERIFY_REASON"
  exit 97
fi
printf 'all_tp_workers_ready_before_cuda pids=%s server_pgid=%s\n' \
  "$worker_pids" "$server_pgid"
printf 'stopping_holder_after_tp_worker_import pid=%s pgid=%s\n' \
  "$holder_pid" "$holder_pgid"
begin_signal_transition
if ! kill -TERM -- "-$holder_pgid"; then
  end_signal_transition
  printf 'ERROR: failed to stop bootstrap holder pgid=%s\n' "$holder_pgid"
  exit 98
fi
bootstrap_holder_active=0
holder_stopped=1
end_signal_transition
unexpected_gpu_pids=""
for _ in $(seq 1 90); do
  if ! active_gpu_pids=$(gpu_pids); then
    printf '%s\n' \
      'ERROR: cannot prove bootstrap holder released the GPUs; refusing TP release'
    exit 98
  fi
  unexpected_gpu_pids=""
  while IFS= read -r gpu_pid; do
    [[ -z "$gpu_pid" ]] && continue
    case " $worker_pids " in
      *" $gpu_pid "*) ;;
      *) unexpected_gpu_pids+="$gpu_pid " ;;
    esac
  done <<< "$active_gpu_pids"
  [[ -z "$unexpected_gpu_pids" ]] && break
  sleep 1
done
if [[ -n "$unexpected_gpu_pids" ]]; then
  printf 'ERROR: unexpected GPU processes remain after stopping holder: %s\n' \
    "$unexpected_gpu_pids"
  exit 98
fi
if [[ -e "$worker_go_file" || -L "$worker_go_file" ]]; then
  printf 'ERROR: TP worker release path appeared before publication: %s\n' \
    "$worker_go_file"
  exit 98
fi
worker_go_tmp="$worker_go_file.tmp.$$"
worker_go_tmp_identity=""
worker_go_identity=""
worker_go_record=""
# A same-directory hard link is an atomic, no-replace publication: unlike
# `mv`, it cannot overwrite a release path created between the stale check and
# publication.  Sync the private 0600 inode first, then prove the public path
# is the same regular inode with the exact release record before proceeding.
if ! (
  umask 077 &&
    set -o noclobber &&
    printf 'release_workers=8\n' > "$worker_go_tmp"
) \
   || ! sync "$worker_go_tmp" \
   || ! worker_go_tmp_identity=$(stat -Lc '%d:%i:%u:%a' "$worker_go_tmp") \
   || ! ln "$worker_go_tmp" "$worker_go_file"; then
  printf 'ERROR: atomic TP worker release publication failed path=%s\n' \
    "$worker_go_file"
  [[ ! -e "$worker_go_tmp" && ! -L "$worker_go_tmp" ]] \
    || rm -f -- "$worker_go_tmp"
  exit 98
fi
if ! worker_go_identity=$(stat -Lc '%d:%i:%u:%a' "$worker_go_file") \
   || ! worker_go_record=$(tr -d '\r\n' < "$worker_go_file") \
   || [[ -L "$worker_go_file" || ! -f "$worker_go_file" \
         || "$worker_go_identity" != "$worker_go_tmp_identity" \
         || "$worker_go_record" != release_workers=8 ]]; then
  printf 'ERROR: atomic TP worker release verification failed path=%s identity=%q expected=%q record=%q\n' \
    "$worker_go_file" "$worker_go_identity" "$worker_go_tmp_identity" \
    "$worker_go_record"
  rm -f -- "$worker_go_tmp"
  exit 98
fi
if ! rm -f -- "$worker_go_tmp"; then
  printf 'ERROR: cannot remove private TP release link after publication: %s\n' \
    "$worker_go_tmp"
  exit 98
fi
printf 'gpu_release_acknowledged workers=8\n'

worker_gpu_ready=0
worker_gpu_status=1
for _ in $(seq 1 180); do
  verify_worker_gpu_coverage_once "$worker_pids"
  worker_gpu_status=$?
  if (( worker_gpu_status == 0 )); then
    worker_gpu_ready=1
    break
  fi
  if (( worker_gpu_status == 2 )); then
    break
  fi
  if ! kill -0 "$benchmark_pid" 2>/dev/null; then
    GPU_VERIFY_REASON="benchmark exited before TP GPU coverage was authenticated"
    break
  fi
  if (( _ % 10 == 0 )) && ! verify_server_identity 1; then
    GPU_VERIFY_REASON="server identity changed during TP GPU coverage wait"
    break
  fi
  sleep 1
done
if [[ "$worker_gpu_ready" != 1 ]]; then
  printf 'ERROR: TP workers failed physical 8-GPU UUID coverage status=%s reason=%s\n' \
    "$worker_gpu_status" "$GPU_VERIFY_REASON"
  exit 96
fi
printf 'tp_worker_gpu_coverage_verified workers=8 mapping=%s\n' \
  "$GPU_VERIFY_MAPPING"

benchmark_started_epoch=$(date +%s)
(
  while kill -0 "$benchmark_pid" 2>/dev/null; do
    sleep 30
    if kill -0 "$benchmark_pid" 2>/dev/null; then
      now_epoch=$(date +%s)
      elapsed_s=$((now_epoch - benchmark_started_epoch))
      last_driver_line=$(tail -n 1 "$run_dir/driver.log" 2>/dev/null \
        | tr '\r\n' ' ' | cut -c1-240)
      printf '[progress] benchmark_alive elapsed_s=%s result=%s last=%s\n' \
        "$elapsed_s" "$([[ -f "$run_dir/result.json" ]] && printf present || printf pending)" \
        "$last_driver_line"
    fi
  done
) &
heartbeat_pid=$!

wait "$benchmark_pid"
benchmark_exit=$?
benchmark_pid=""
benchmark_pgid=""
kill -TERM "$heartbeat_pid" 2>/dev/null || true
wait "$heartbeat_pid" 2>/dev/null || true
heartbeat_pid=""
printf '%s\n' "$benchmark_exit" > "$run_dir/exit_code"
printf 'benchmark_exit=%s\n' "$benchmark_exit"
exit "$benchmark_exit"
