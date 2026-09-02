#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
root=/mnt/tidal-alsh01/dataset/redone/redknot_runs/long_output_30_3datasets_256k_440k
matrix_pid=$(<"$root/matrix.pid")
while kill -0 "$matrix_pid" 2>/dev/null; do
  sleep 60
done

completed=$(find "$root" -name result.json -type f | wc -l)
if [[ "$completed" -ne 6 ]]; then
  printf 'showcase finalization refused: source matrix has %s/6 results\n' \
    "$completed" >&2
  exit 2
fi

showcase_dir=$script_dir/datasets/LongBench/showcase
mkdir -p "$showcase_dir"
python_bin=${REDKNOT_PYTHON:-$script_dir/../../../.venv_tf5/bin/python}

"$python_bin" "$script_dir/build_long_output_showcase.py" \
  --run-root "$root" \
  --target-output-tokens 30 \
  --offline-context-tokens 262144 \
  --limit 5 \
  --output "$showcase_dir/long_output_256k.json"

"$python_bin" "$script_dir/build_long_output_showcase.py" \
  --run-root "$root" \
  --target-output-tokens 30 \
  --offline-context-tokens 450560 \
  --limit 5 \
  --output "$showcase_dir/long_output_440k.json"

sha256sum \
  "$showcase_dir/long_output_256k.json" \
  "$showcase_dir/long_output_440k.json" \
  >"$showcase_dir/SHA256SUMS"
