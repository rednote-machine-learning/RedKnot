#!/usr/bin/env bash
# Sequential reproduction; prepare mode only tokenizes data and never loads a model.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_redknot_benchmarks.sh --model-path DIR --output-root NEW_DIR [options]

  --prepare-only          Freeze inputs without starting GPU inference
  --suites "64k 128k"     Space-separated suites (default: all four release lengths)
  --runtime-root DIR      Explicit external DeepInfer backend checkout
  --ep N                  Expert parallel size (default: 8)
  --cp N                  Context parallel size (default: 1)
  --iterations N          Measured iterations per suite (default: 10)
  --warmup-iterations N   Independent warmup iterations (default: 3)
  --trust-remote-code     Permit the selected tokenizer's custom Python code

Run from a prepared DeepInfer environment. Set PYTHON_BIN to select its interpreter.
EOF
}

model_path=""
output_root=""
runtime_root=""
suites="64k 128k 256k 440k"
prepare_only=0
ep=8
cp=1
iterations=10
warmup_iterations=3
tokenizer_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path|--output-root|--runtime-root|--suites|--ep|--cp|--iterations|--warmup-iterations)
      [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; } ;;
  esac
  case "$1" in
    --model-path) model_path="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --runtime-root) runtime_root="$2"; shift 2 ;;
    --suites) suites="$2"; shift 2 ;;
    --ep) ep="$2"; shift 2 ;;
    --cp) cp="$2"; shift 2 ;;
    --iterations) iterations="$2"; shift 2 ;;
    --warmup-iterations) warmup_iterations="$2"; shift 2 ;;
    --prepare-only) prepare_only=1; shift ;;
    --trust-remote-code) tokenizer_args+=(--trust-remote-code); shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$model_path" && -n "$output_root" ]] || { usage >&2; exit 2; }
suite_count=0
for suite in $suites; do
  case "$suite" in
    64k|128k|256k|440k) suite_count=$((suite_count + 1)) ;;
    *) printf 'Invalid suite: %s\n' "$suite" >&2; exit 2 ;;
  esac
done
[[ "$suite_count" -gt 0 ]] || { printf 'At least one suite is required\n' >&2; exit 2; }
[[ ! -e "$output_root" ]] || { printf 'Output already exists: %s\n' "$output_root" >&2; exit 2; }
model_path="$(cd "$model_path" && pwd)"
mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"
if [[ -n "$runtime_root" ]]; then runtime_root="$(cd "$runtime_root" && pwd)"; fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
runtime_args=()
if [[ -n "$runtime_root" ]]; then runtime_args+=(--runtime-root "$runtime_root"); fi

for suite in $suites; do
  "$python_bin" -m benchmarks.redknot prepare \
    --suite "$suite" --model-profile deepseek-v4.1-flash \
    --tokenizer "$model_path" ${tokenizer_args[@]+"${tokenizer_args[@]}"} \
    --output "$output_root/$suite.manifest.json"
  "$python_bin" -m benchmarks.redknot validate "$output_root/$suite.manifest.json"
  if [[ "$prepare_only" -eq 0 ]]; then
    "$python_bin" -m benchmarks.redknot run \
      --manifest "$output_root/$suite.manifest.json" --model-path "$model_path" \
      --ep "$ep" --cp "$cp" --iterations "$iterations" --warmup-iterations "$warmup_iterations" \
      ${runtime_args[@]+"${runtime_args[@]}"} --output-dir "$output_root/$suite-results"
  fi
done
