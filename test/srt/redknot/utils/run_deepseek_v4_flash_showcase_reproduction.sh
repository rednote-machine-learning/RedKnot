#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
showcase_dir=$script_dir/datasets/LongBench/showcase

for length in 256k 440k; do
  showcase=$showcase_dir/long_output_${length}.json
  if [[ ! -s "$showcase" ]]; then
    printf 'ERROR: frozen showcase dataset is absent: %s\n' "$showcase" >&2
    exit 2
  fi
done

# Re-run the full output-blind source pool, not only the five post-hoc examples.
# This keeps every failure visible while guaranteeing that each selected case
# is present in the generated comparison report.
exec "$script_dir/run_deepseek_v4_flash_reproduction.sh" \
  --datasets hotpotqa,musique,multifieldqa_en \
  --lengths 256K,440K \
  --cases-per-dataset 3 \
  --long-output-tokens 30 \
  "$@"
