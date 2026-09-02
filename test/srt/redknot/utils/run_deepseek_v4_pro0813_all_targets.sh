#!/usr/bin/env bash
# Run one fail-closed formal Pro-0813 reproduction at each supported length.
# Release suites remain a separate, explicit expansion; this entry point runs
# exactly one frozen profile per target in ascending token order.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
redknot_dir=$(cd "$script_dir/.." && pwd -P)
repo=$(cd "$script_dir/../../../.." && pwd -P)
# The two profile preflights below run before per-target ``env`` calls, so the
# sweep itself must discard ambient Python overlays at entry as well.
export PYTHONPATH="$repo/python:/data/temp/FlashMLA-sm103-src"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
runner="$script_dir/run_deepseek_v4_pro0813_reproduction.sh"
runtime_python=/workspace/RedKnot/.venv_sm103/bin/python
profile_verifier="$script_dir/verify_pro0813_qualification_profile.py"
formal_asset_verifier="$script_dir/verify_pro0813_formal_assets.py"
summary_writer="$script_dir/write_pro0813_all_targets_summary.py"
usage='usage: run_deepseek_v4_pro0813_all_targets.sh [ROOT_RUN_DIR]'

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' "$usage"
  printf '%s\n' \
    'Runs exactly: 64K, 128K, 256K, 440K, 512K.' \
    '64K/128K/256K use their built-in frozen inputs.' \
    '440K/512K require the canonical qualification_profiles/pro0813_{440k,512k}_hotpotqa_10q/profile.json assets and fixed SHA-256 values.' \
    'Profile, sampling, algorithm, and kernel-backend overrides are rejected or replaced by the canonical formal environment.' \
    'This is the five-target formal sweep, not the separate 15-case release-suite expansion.'
  exit 0
fi
if (( $# > 1 )); then
  printf 'ERROR: expected at most one argument, got %s\n%s\n' \
    "$#" "$usage" >&2
  exit 64
fi
if [[ ! -x "$runner" ]]; then
  printf 'ERROR: Pro-0813 target runner is not executable: %s\n' \
    "$runner" >&2
  exit 66
fi
if [[ ! -x "$runtime_python" \
      || ! -r "$profile_verifier" \
      || ! -r "$formal_asset_verifier" \
      || ! -r "$summary_writer" ]]; then
  printf 'ERROR: certified Python or formal asset tooling is unavailable: python=%s profile_verifier=%s asset_verifier=%s summary_writer=%s\n' \
    "$runtime_python" "$profile_verifier" "$formal_asset_verifier" \
    "$summary_writer" >&2
  exit 66
fi
if [[ "${REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE:-0}" != 0 ]]; then
  printf '%s\n' \
    'ERROR: the all-target entry point is formal-only; unset REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE or set it to 0' >&2
  exit 64
fi
if [[ "${REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY:-0}" != 0 ]]; then
  printf '%s\n' \
    'ERROR: the all-target entry point cannot run a zoff-only diagnostic ablation; unset REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY or set it to 0' >&2
  exit 64
fi

root_run_dir=${1:-/workspace/RedKnot/results/pro0813-formal-all-$(date +%Y%m%d-%H%M%S)}
if [[ "$root_run_dir" != /* || "$root_run_dir" == "/" ]]; then
  printf 'ERROR: ROOT_RUN_DIR must be an absolute non-root path, got %q\n' \
    "$root_run_dir" >&2
  exit 64
fi
if [[ -e "$root_run_dir" || -L "$root_run_dir" ]]; then
  printf 'ERROR: refusing an existing all-target run directory: %s\n' \
    "$root_run_dir" >&2
  exit 73
fi

qualification_profile_440k=$redknot_dir/qualification_profiles/pro0813_440k_hotpotqa_10q/profile.json
qualification_profile_512k=$redknot_dir/qualification_profiles/pro0813_512k_hotpotqa_10q/profile.json
canonical_profile_440k_sha256=52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b
canonical_profile_512k_sha256=e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d
if [[ -n "${REDKNOT_QUALIFICATION_PROFILE_440K:-}" \
      && "$REDKNOT_QUALIFICATION_PROFILE_440K" != "$qualification_profile_440k" ]]; then
  printf '%s\n' \
    'ERROR: canonical all-target formal sweep forbids REDKNOT_QUALIFICATION_PROFILE_440K overrides' >&2
  exit 64
fi
if [[ -n "${REDKNOT_QUALIFICATION_PROFILE_512K:-}" \
      && "$REDKNOT_QUALIFICATION_PROFILE_512K" != "$qualification_profile_512k" ]]; then
  printf '%s\n' \
    'ERROR: canonical all-target formal sweep forbids REDKNOT_QUALIFICATION_PROFILE_512K overrides' >&2
  exit 64
fi
for profile_name in qualification_profile_440k qualification_profile_512k; do
  profile_path=${!profile_name}
  if [[ "$profile_path" != /* \
        || -L "$profile_path" \
        || ! -f "$profile_path" \
        || ! -r "$profile_path" ]]; then
    printf 'ERROR: %s must name a readable absolute frozen single-profile file, got %q\n' \
      "$profile_name" "$profile_path" >&2
    exit 66
  fi
done

printf '%s\n' \
  'prevalidating all five selection/dataset/profile asset identities before any target, model, or holder action'
if ! formal_asset_record=$(
  "$runtime_python" "$formal_asset_verifier"
); then
  printf '%s\n' \
    'ERROR: formal Pro-0813 asset preflight failed before holder lookup' >&2
  exit 66
fi
profile_440k_record=$(printf '%s\n' "$formal_asset_record" \
  | "$runtime_python" -c \
    'import json,sys; d=json.load(sys.stdin); print(json.dumps(next(x for x in d["assets"] if x["target_tokens"]==450560),sort_keys=True))')
profile_512k_record=$(printf '%s\n' "$formal_asset_record" \
  | "$runtime_python" -c \
    'import json,sys; d=json.load(sys.stdin); print(json.dumps(next(x for x in d["assets"] if x["target_tokens"]==524288),sort_keys=True))')
profile_440k_sha256=$(printf '%s\n' "$profile_440k_record" \
  | "$runtime_python" -c \
    'import json, sys; print(json.load(sys.stdin)["profile_sha256"])')
profile_512k_sha256=$(printf '%s\n' "$profile_512k_record" \
  | "$runtime_python" -c \
    'import json, sys; print(json.load(sys.stdin)["profile_sha256"])')
if ! [[ "$profile_440k_sha256" =~ ^[0-9a-f]{64}$ \
        && "$profile_512k_sha256" =~ ^[0-9a-f]{64}$ \
        && "$profile_440k_sha256" == "$canonical_profile_440k_sha256" \
        && "$profile_512k_sha256" == "$canonical_profile_512k_sha256" ]]; then
  printf '%s\n' \
    'ERROR: qualification profile verifier did not return the canonical SHA-256 records' >&2
  exit 66
fi

# The all-target command is the canonical formal sweep, not a parameterized
# experiment entry point. Pin every ambient knob consumed by the one-target
# wrapper, supervisor, launcher, or HTTP driver that can change the algorithm,
# sampling plan, kernel backend, or claim evidence. Operational paths/port/lock
# remain explicit override points.
formal_unset_args=(
  -u REDKNOT_HEAD_CFG
  -u REDKNOT_SWA_FULL_TOKENS_RATIO
  -u REDKNOT_SERVER_POLICY_MANIFEST_OUT
  -u REDKNOT_SERVER_INSTANCE_NONCE
)
formal_env=(
  PYTHONPATH=$repo/python:/data/temp/FlashMLA-sm103-src
  PYTHONNOUSERSITE=1
  PYTHONSAFEPATH=1
  REDKNOT_TIMING=1
  REDKNOT_TTFT_WARMUP=3
  REDKNOT_TTFT_ITERS=10
  REDKNOT_MERGED_PREFILL_TOKENS=0
  REDKNOT_FIRST_DOCUMENT_PREFIX=1
  REDKNOT_GEOMETRY_TEMPLATE_CACHE=0
  REDKNOT_QPS_CONCURRENCIES=1
  REDKNOT_MEASURE_QPS=1
  REDKNOT_QPS_WARMUP_WAVES=3
  REDKNOT_QPS_WAVES=10
  REDKNOT_ROW_SPARSE_ACTIVE_RATIO=0.20
  REDKNOT_QUERY_PROTECTION_TOKENS=8192
  REDKNOT_GENERALIZED_ADAPTIVE_CONTROLLER=0
  REDKNOT_QUALITY_REPEATS=3
  REDKNOT_GENERALIZED_STRONG_RATIO=0.15
  REDKNOT_GENERALIZED_MEDIUM_RATIO=0.20
  REDKNOT_GENERALIZED_DIFFUSE_RATIO=0.25
  REDKNOT_RELEASE_ADAPTIVE_TOPK_MASS=0.50
  REDKNOT_RELEASE_ADAPTIVE_TOPK_BUCKETS=3,4,5,6
  REDKNOT_PRO0813_ADAPTIVE_TOPK_ENABLED=0
  REDKNOT_COMBINED_HEADSPLIT_ROW_SPARSE=1
  REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=0
  REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY=0
  REDKNOT_MOE_RUNNER_BACKEND=flashinfer_mxfp4
  SGLANG_OPT_USE_TILELANG_MHC_PRE=0
  SGLANG_OPT_USE_TILELANG_MHC_POST=0
  SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
  REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH=0
  SGLANG_OPT_FP8_WO_A_GEMM=0
  SGLANG_OPT_USE_ONLINE_COMPRESS=0
  REDKNOT_DISABLE_FLASHINFER_AUTOTUNE=0
  REDKNOT_DETERMINISTIC_INFERENCE=0
  REDKNOT_DISABLE_OVERLAP_SCHEDULE=0
  REDKNOT_SHARED_LATENT_MAX_SEGMENT_EPOCHS=16
  REDKNOT_SPARSE_FFN=0
  REDKNOT_PROGRESSIVE_TOPK_SCHEDULE=
  REDKNOT_RANDOM_SEED=2026
  REDKNOT_PYTHON=/workspace/RedKnot/.venv_sm103/bin/python
  REDKNOT_PYTHON_BIN=/workspace/RedKnot/.venv_sm103/bin/python
  SGLANG_USE_JIT_RMSNORM=1
  FLASHINFER_CUDA_ARCH_LIST=10.3a
  TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  REDKNOT_FLASHMLA_SM103_ROOT=/data/temp/FlashMLA-sm103-src
  REDKNOT_CUOBJDUMP_BIN=/usr/local/cuda/bin/cuobjdump
)

mkdir -p "$root_run_dir"
printf '%s\n' "$formal_asset_record" \
  > "$root_run_dir/formal_assets.preflight.json"
printf '%s\n' "$profile_440k_record" \
  > "$root_run_dir/qualification_profile_440k.preflight.json"
printf '%s\n' "$profile_512k_record" \
  > "$root_run_dir/qualification_profile_512k.preflight.json"
printf '%s\n' "${formal_env[@]}" > "$root_run_dir/formal_environment.env"
plan_file="$root_run_dir/sequence.plan.tsv"
printf 'ordinal\ttarget_tokens\tlabel\trun_dir\tqualification_profile\tprofile_sha256\tmem_fraction_static\tcheckpoint_max_islands\n' \
  > "$plan_file"
targets=(65536 131072 262144 450560 524288)
labels=(64k 128k 256k 440k 512k)
mem_fractions=(0.80 0.77 0.67 0.80 0.80)
checkpoint_capacities=(64 64 128 256 256)
target_exit_codes=()
for index in "${!targets[@]}"; do
  ordinal=$((index + 1))
  target_tokens=${targets[$index]}
  target_label=${labels[$index]}
  target_mem_fraction_static=${mem_fractions[$index]}
  target_checkpoint_max_islands=${checkpoint_capacities[$index]}
  target_run_dir="$root_run_dir/$target_label"
  target_profile=builtin
  target_profile_sha256=builtin
  if [[ "$target_tokens" == 450560 ]]; then
    target_profile=$qualification_profile_440k
    target_profile_sha256=$profile_440k_sha256
  elif [[ "$target_tokens" == 524288 ]]; then
    target_profile=$qualification_profile_512k
    target_profile_sha256=$profile_512k_sha256
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ordinal" "$target_tokens" "$target_label" \
    "$target_run_dir" "$target_profile" "$target_profile_sha256" \
    "$target_mem_fraction_static" "$target_checkpoint_max_islands" \
    >> "$plan_file"
done

for index in "${!targets[@]}"; do
  target_tokens=${targets[$index]}
  target_label=${labels[$index]}
  target_mem_fraction_static=${mem_fractions[$index]}
  target_checkpoint_max_islands=${checkpoint_capacities[$index]}
  target_run_dir="$root_run_dir/$target_label"
  printf 'formal_target_start ordinal=%s/5 label=%s target_tokens=%s run_dir=%s\n' \
    "$((index + 1))" "$target_label" "$target_tokens" "$target_run_dir"
  target_exit_code=0
  case "$target_tokens" in
    450560)
      if env "${formal_unset_args[@]}" "${formal_env[@]}" \
        REDKNOT_MEM_FRACTION_STATIC="$target_mem_fraction_static" \
        REDKNOT_ROW_SPARSE_CHECKPOINT_MAX_ISLANDS="$target_checkpoint_max_islands" \
        REDKNOT_QUALIFICATION_PROFILE="$qualification_profile_440k" \
        REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256="$profile_440k_sha256" \
        "$runner" "$target_run_dir" "$target_tokens"; then
        target_exit_code=0
      else
        target_exit_code=$?
      fi
      ;;
    524288)
      if env "${formal_unset_args[@]}" "${formal_env[@]}" \
        REDKNOT_MEM_FRACTION_STATIC="$target_mem_fraction_static" \
        REDKNOT_ROW_SPARSE_CHECKPOINT_MAX_ISLANDS="$target_checkpoint_max_islands" \
        REDKNOT_QUALIFICATION_PROFILE="$qualification_profile_512k" \
        REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256="$profile_512k_sha256" \
        "$runner" "$target_run_dir" "$target_tokens"; then
        target_exit_code=0
      else
        target_exit_code=$?
      fi
      ;;
    *)
      if env "${formal_unset_args[@]}" \
        -u REDKNOT_QUALIFICATION_PROFILE \
        -u REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256 \
        "${formal_env[@]}" \
        REDKNOT_MEM_FRACTION_STATIC="$target_mem_fraction_static" \
        REDKNOT_ROW_SPARSE_CHECKPOINT_MAX_ISLANDS="$target_checkpoint_max_islands" \
        "$runner" "$target_run_dir" "$target_tokens"; then
        target_exit_code=0
      else
        target_exit_code=$?
      fi
      ;;
  esac
  target_exit_codes+=("$target_exit_code")
  printf 'formal_target_complete ordinal=%s/5 label=%s target_tokens=%s exit_code=%s continuing=true\n' \
    "$((index + 1))" "$target_label" "$target_tokens" \
    "$target_exit_code"
done

summary_args=(--root-run-dir "$root_run_dir")
for target_exit_code in "${target_exit_codes[@]}"; do
  summary_args+=(--exit-code "$target_exit_code")
done
if ! summary_record=$(
  "$runtime_python" "$summary_writer" "${summary_args[@]}"
); then
  printf '%s\n' \
    'ERROR: could not publish the machine-readable all-target summary' >&2
  exit 66
fi
if ! overall_exit_code=$(printf '%s\n' "$summary_record" \
  | "$runtime_python" -c \
    'import json,sys; value=json.load(sys.stdin)["overall_exit_code"]; assert type(value) is int and 0 <= value <= 255; print(value)'); then
  printf '%s\n' 'ERROR: all-target summary has no valid overall exit code' >&2
  exit 66
fi
printf 'formal_all_targets_complete root_run_dir=%s summary=%s overall_exit_code=%s\n' \
  "$root_run_dir" "$root_run_dir/all_targets_summary.json" \
  "$overall_exit_code"
exit "$overall_exit_code"
