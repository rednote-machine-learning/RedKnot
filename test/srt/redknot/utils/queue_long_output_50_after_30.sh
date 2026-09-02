#!/usr/bin/env bash
set -u

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
run_root=/mnt/tidal-alsh01/dataset/redone/redknot_runs
root30=$run_root/long_output_30_3datasets_256k_440k
root50=$run_root/long_output_50_3datasets_256k_440k
pid_file=$root30/matrix.pid
status_file=$root50/queued_run_status

if [[ ! -s "$pid_file" ]]; then
  printf '30-token matrix pid file is absent: %s\n' "$pid_file" >&2
  exit 2
fi
matrix_pid=$(<"$pid_file")
while kill -0 "$matrix_pid" 2>/dev/null; do
  sleep 60
done

completed=$(find "$root30" -name result.json -type f | wc -l)
if [[ "$completed" -ne 6 ]]; then
  printf 'not_started: 30-token matrix produced %s/6 results\n' "$completed" \
    >"$status_file"
  exit 2
fi

cd "$script_dir"
./run_deepseek_v4_flash_reproduction.sh \
  --datasets hotpotqa,musique,multifieldqa_en \
  --lengths 256K,440K \
  --cases-per-dataset 3 \
  --long-output-tokens 50 \
  --ttft-warmup 3 \
  --ttft-iters 5 \
  --quality-repeats 2 \
  --resume \
  --output-dir "$root50"
status=$?
printf 'exit=%s\n' "$status" >"$status_file"
exit "$status"
