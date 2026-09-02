#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# run_all_rag.sh — one-click driver for ALL RedKnot RAG benchmarks.
#
# Runs the per-model `benchmark_RedKnot_<Model>_RAG.py` scripts one after the
# other, saving each full log under ./rag_logs/ and finally printing every
# script's SUMMARY block so you get speed (TTFT speedup), accuracy (F1/EM) and
# compute (FLOPs saving) for each model in one place.
#
# Each benchmark already compares the honest dense baseline vs RedKnot, so the
# "background overhead" you asked about is visible as:
#   * the per-headclass timing line  [headclass] online=.. rope=.. kv_build=..
#   * TTFT baseline vs RedKnot (and the speedup)
#   * decode tok/s baseline vs RedKnot
#
# Usage:
#   # default: the 4 small/medium models that fit on the free GPU memory
#   bash test/srt/redknot/run_all_rag.sh
#
#   # pick models + run size:
#   RK_MODELS="mistral qwen3" RK_SAMPLES=4 RK_LENGTHS=16K,32K \
#     bash test/srt/redknot/run_all_rag.sh
#
#   # include the big MoE models (needs lots of free VRAM):
#   RK_MODELS="mistral qwen3 llama qwen35 deepseek" \
#     bash test/srt/redknot/run_all_rag.sh
#
# Env knobs (with defaults):
#   RK_MODELS   "mistral qwen3 llama qwen35"   space-separated model keys
#   RK_SAMPLES  4                              REDKNOT_N_SAMPLES per model
#   RK_LENGTHS  16K,24K,32K                    REDKNOT_LENGTHS (HotpotQA models)
#   RK_MAX_NEW  48                             REDKNOT_MAX_NEW
#   RK_GPUS     0                              CUDA_VISIBLE_DEVICES
#   RK_COMPILE  0                              REDKNOT_COMPILE (1 = torch.compile)
# ───────────────────────────────────────────────────────────────────────────
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${HERE}/rag_logs"
mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

RK_MODELS="${RK_MODELS:-mistral qwen3 llama qwen35}"
RK_SAMPLES="${RK_SAMPLES:-4}"
RK_LENGTHS="${RK_LENGTHS:-16K,24K,32K}"
RK_MAX_NEW="${RK_MAX_NEW:-48}"
RK_GPUS="${RK_GPUS:-0}"
RK_COMPILE="${RK_COMPILE:-0}"

# model key -> benchmark script (post-rename, no spaces)
declare -A SCRIPT=(
  [mistral]="benchmark_RedKnot_Mistral_RAG.py"
  [qwen3]="benchmark_RedKnot_Qwen3_RAG.py"
  [llama]="benchmark_RedKnot_Llama3.3_RAG.py"
  [qwen35]="benchmark_RedKnot_Qwen35_397B_RAG.py"
  [deepseek]="benchmark_RedKnot_DeepSeekV4_RAG.py"
)

# Qwen3.5 (qwen3_5_moe) and DeepSeek-V4 need a newer transformers than the
# system one (4.57). The repo ships a venv (.venv_tf5, transformers 5.x) that
# supports them; use it for those keys, system python for the rest.
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
VENV_TF5_PY="${REPO_ROOT}/.venv_tf5/bin/python"
pyfor() {  # echo the python interpreter to use for a given model key
  case "$1" in
    qwen35|deepseek)
      if [[ -x "${VENV_TF5_PY}" ]]; then echo "${VENV_TF5_PY}"; else echo "python"; fi ;;
    *) echo "python" ;;
  esac
}

echo "============================================================================"
echo " RedKnot RAG benchmark sweep   (stamp=${STAMP})"
echo " models   : ${RK_MODELS}"
echo " samples  : ${RK_SAMPLES}   lengths: ${RK_LENGTHS}   max_new: ${RK_MAX_NEW}"
echo " gpus     : ${RK_GPUS}      compile: ${RK_COMPILE}"
echo " logs     : ${LOGDIR}"
echo "============================================================================"

declare -A LOGFILE
for m in ${RK_MODELS}; do
  scr="${SCRIPT[$m]:-}"
  if [[ -z "${scr}" ]]; then
    echo ">> [skip] unknown model key '${m}'"
    continue
  fi
  if [[ ! -f "${HERE}/${scr}" ]]; then
    echo ">> [skip] ${m}: script not found (${scr})"
    continue
  fi
  log="${LOGDIR}/${m}_${STAMP}.log"
  LOGFILE[$m]="${log}"
  PYBIN="$(pyfor "${m}")"
  echo ""
  echo ">> [run] ${m}  ->  ${scr}   (py: ${PYBIN})"
  echo "         log: ${log}"
  CUDA_VISIBLE_DEVICES="${RK_GPUS}" \
  REDKNOT_COMPILE="${RK_COMPILE}" \
  REDKNOT_N_SAMPLES="${RK_SAMPLES}" \
  REDKNOT_LENGTHS="${RK_LENGTHS}" \
  REDKNOT_MAX_NEW="${RK_MAX_NEW}" \
    "${PYBIN}" "${HERE}/${scr}" >"${log}" 2>&1
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "   [FAILED] rc=${rc} (see ${log})"
  else
    echo "   [done]"
  fi
done

# ── aggregate: print each model's SUMMARY block ──
echo ""
echo "============================================================================"
echo " AGGREGATE SUMMARY (per model)"
echo "============================================================================"
for m in ${RK_MODELS}; do
  log="${LOGFILE[$m]:-}"
  [[ -z "${log}" || ! -f "${log}" ]] && continue
  echo ""
  echo "######## ${m} ########"
  # Print from the last 'SUMMARY' / 'Summary' line to end of file.
  awk 'BEGIN{p=0} /SUMMARY|^Summary/{p=1} p{print}' "${log}" | tail -n 40
  # Also surface PLOTROW lines (Qwen3.5 emits machine-readable rows).
  grep -h "^PLOTROW" "${log}" 2>/dev/null
done
echo ""
echo "Full logs in: ${LOGDIR}"
