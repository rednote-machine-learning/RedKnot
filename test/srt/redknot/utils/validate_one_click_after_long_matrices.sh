#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
run_root=/mnt/tidal-alsh01/dataset/redone/redknot_runs
root50=$run_root/long_output_50_3datasets_256k_440k
queue_pid=$(<"$root50/queue.pid")
while kill -0 "$queue_pid" 2>/dev/null; do
  sleep 60
done

if [[ ! -s "$root50/queued_run_status" ]] || \
   ! grep -qx 'exit=0' "$root50/queued_run_status"; then
  printf 'one-click validation refused: 50-token matrix did not finish cleanly\n' >&2
  exit 2
fi

validation=$run_root/deepseek_v4_flash_showcase_oneclick_validation
if [[ -e "$validation" ]]; then
  printf 'one-click validation output already exists: %s\n' "$validation" >&2
  exit 2
fi

exec "$script_dir/run_deepseek_v4_flash_showcase_reproduction.sh" \
  --prepare-only \
  --output-dir "$validation"
