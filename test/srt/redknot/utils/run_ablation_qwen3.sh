#!/usr/bin/env bash
# Cumulative component ablation on Qwen3-32B (quality + analytic FLOPs).
# Hardware-independent metrics (EM/F1/FLOPs) — valid regardless of GPU model.
set -uo pipefail
cd /mnt/tidal-alsh01/dataset/redone/RedKnotV0.3
RK=test/srt/redknot
OUT=$RK/figures/runs
MODEL=/mnt/tidal-alsh01/dataset/redone/096/models/Qwen3-32B
REAL=$RK/head_class/qwen3-32B_optimal_g15_lf_ret.json
ALLG=$RK/head_class/qwen3-32B_ablation_allglobal.json
COMMON="REDKNOT_N_SAMPLES=3 REDKNOT_MAX_NEW=16 REDKNOT_LENGTHS=24K REDKNOT_COMPILE=0 REDKNOT_MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1"

echo "##### [ABLATION 1/3] DENSE (all-global heads, dense FFN) #####"
env $COMMON REDKNOT_HEAD_CFG=$ALLG REDKNOT_FFN_DENSE_UNTIL=64 \
    python3 $RK/benchmark_RedKnot_Qwen3_RAG.py > $OUT/abl_dense.log 2>&1
echo "  -> $(grep -E '^ 24K ' $OUT/abl_dense.log | tail -1)"

echo "##### [ABLATION 2/3] +HEAD-CLASS (real heads, dense FFN) #####"
env $COMMON REDKNOT_HEAD_CFG=$REAL REDKNOT_FFN_DENSE_UNTIL=64 \
    python3 $RK/benchmark_RedKnot_Qwen3_RAG.py > $OUT/abl_headclass.log 2>&1
echo "  -> $(grep -E '^ 24K ' $OUT/abl_headclass.log | tail -1)"

echo "##### [ABLATION 3/3] +SPARSE-FFN (real heads, sparse FFN = full RedKnot) #####"
env $COMMON REDKNOT_HEAD_CFG=$REAL \
    python3 $RK/benchmark_RedKnot_Qwen3_RAG.py > $OUT/abl_full.log 2>&1
echo "  -> $(grep -E '^ 24K ' $OUT/abl_full.log | tail -1)"

echo "##### ABLATION DONE #####"
