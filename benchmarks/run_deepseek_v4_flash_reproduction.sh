#!/usr/bin/env bash
# RedKnot vLLM-only entrypoint. No installer, model downloader or GPU process control.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
python_bin=${REDKNOT_PYTHON:-python3}
exec "$python_bin" "$script_dir/benchmark_RedKnot_DeepSeekV4Flash.py" "$@"
