# Copyright 2024-2026 SGLang RedKnot Integration.
"""Evaluation harness utilities for RedKnot (paper §5, Table 3/4).

The paper evaluates across six public long-context QA datasets and reports
both quality (F1, EM, cos, top-1/top-10) and efficiency (TTFT, prefill
FLOPs, KV bytes, throughput, CoV). This module centralises the parts that
are model-free and therefore unit-testable:

- :data:`DATASET_SPECS` — the six datasets with their HuggingFace ids and
  RAG layout (paper Table 3).
- :func:`coefficient_of_variation` — TTFT stability metric (Table 4 CoV).
- :func:`aggregate_quality` / :func:`aggregate_efficiency` — reduce per-sample
  records into the paper's summary rows.
- :func:`pd_throughput_projection` — project capacity-bound throughput from a
  concurrent-capacity multiplier (paper fig. 13c: 4.7-7.8× capacity ->
  3.4-3.9× capacity-bound throughput).

The full model-driven evaluation is driven by
``examples/redknot/rag_redknot_demo.py`` (and an eval runner can loop it
over :data:`DATASET_SPECS`); this module is its reusable, testable core.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────────
# Dataset registry (paper Table 3)
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetSpec:
    """A long-context QA dataset and its RAG layout.

    Attributes
    ----------
    key:
        Short label used in panels (e.g. ``"HQA"``).
    hf_id:
        HuggingFace dataset id.
    hf_config:
        HuggingFace config name (or ``None``).
    split:
        Evaluation split.
    qa_type:
        Description (multi-hop / single-hop / long-context / academic).
    layouts:
        List of ``(n_segments, tokens_per_segment)`` RAG layouts to sweep.
    """

    key: str
    hf_id: str
    hf_config: Optional[str]
    split: str
    qa_type: str
    layouts: Sequence[tuple]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec(
        key="HQA",
        hf_id="hotpotqa/hotpot_qa",
        hf_config="distractor",
        split="validation",
        qa_type="multi-hop",
        layouts=[(8, 8000), (4, 2000), (4, 4000)],
    ),
    "musique": DatasetSpec(
        key="MSQ",
        hf_id="dgslibisey/MuSiQue",
        hf_config=None,
        split="validation",
        qa_type="multi-hop",
        layouts=[(10, 2000), (5, 4000), (5, 6000)],
    ),
    "2wikimqa": DatasetSpec(
        key="2WMQA",
        hf_id="voidful/2WikiMultihopQA",
        hf_config=None,
        split="validation",
        qa_type="multi-hop",
        layouts=[(4, 2000), (4, 4000), (4, 8000)],
    ),
    "triviaqa": DatasetSpec(
        key="TQA",
        hf_id="mandarjoshi/trivia_qa",
        hf_config="rc",
        split="validation",
        qa_type="single-hop",
        layouts=[(4, 2000), (4, 4000), (4, 8000)],
    ),
    "multifieldqa": DatasetSpec(
        key="MFQA",
        hf_id="THUDM/LongBench",
        hf_config="multifieldqa_en",
        split="test",
        qa_type="long-context",
        layouts=[(4, 2000), (4, 4000), (4, 8000)],
    ),
    "qasper": DatasetSpec(
        key="QASP",
        hf_id="allenai/qasper",
        hf_config=None,
        split="validation",
        qa_type="academic-paper",
        layouts=[(4, 2000)],
    ),
}


def list_datasets() -> List[str]:
    return list(DATASET_SPECS.keys())


# ──────────────────────────────────────────────────────────────────────────
# Metric reducers
# ──────────────────────────────────────────────────────────────────────────
def coefficient_of_variation(samples: Sequence[float]) -> float:
    """CoV = std / mean (paper Table 4); 0 for <2 samples or zero mean."""
    vals = [float(s) for s in samples]
    if len(vals) < 2:
        return 0.0
    mu = statistics.mean(vals)
    if mu == 0:
        return 0.0
    sigma = statistics.pstdev(vals)
    return sigma / mu


def _mean(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals]
    return statistics.mean(vals) if vals else 0.0


def aggregate_quality(records: Sequence[Dict]) -> Dict[str, float]:
    """Aggregate per-sample quality records into the paper's quality row.

    Each record may carry ``baseline_f1, redknot_f1, baseline_em,
    redknot_em, logits_cosine, top1_match, top10_overlap``.
    """
    if not records:
        return {}
    return {
        "n": len(records),
        "baseline_f1": _mean([r.get("baseline_f1", 0.0) for r in records]),
        "redknot_f1": _mean([r.get("redknot_f1", 0.0) for r in records]),
        "baseline_em": _mean([r.get("baseline_em", 0.0) for r in records]),
        "redknot_em": _mean([r.get("redknot_em", 0.0) for r in records]),
        "logits_cosine": _mean([r.get("logits_cosine", 0.0) for r in records]),
        "top1_match_rate": _mean([float(r.get("top1_match", 0.0)) for r in records]),
        "top10_overlap": _mean([r.get("top10_overlap", 0.0) for r in records]),
    }


def aggregate_efficiency(records: Sequence[Dict]) -> Dict[str, float]:
    """Aggregate per-sample efficiency records into the paper's efficiency row.

    Includes TTFT means, wall/FLOPs speedups, FLOPs savings, and the TTFT
    coefficient of variation (latency stability).
    """
    if not records:
        return {}
    bl_ttft = [r.get("baseline_ttft_s", 0.0) for r in records]
    rc_ttft = [
        r.get("redknot_online_ttft_s", r.get("redknot_ttft_s", 0.0)) for r in records
    ]
    return {
        "n": len(records),
        "baseline_ttft_s": _mean(bl_ttft),
        "redknot_ttft_s": _mean(rc_ttft),
        "wall_speedup": _mean([r.get("wall_speedup", 0.0) for r in records]),
        "flops_speedup": _mean([r.get("flops_speedup", 0.0) for r in records]),
        "flops_savings": _mean([r.get("flops_savings", 0.0) for r in records]),
        "baseline_ttft_cov": coefficient_of_variation(bl_ttft),
        "redknot_ttft_cov": coefficient_of_variation(rc_ttft),
    }


# ──────────────────────────────────────────────────────────────────────────
# PD / throughput projection (paper fig. 13)
# ──────────────────────────────────────────────────────────────────────────
def pd_throughput_projection(
    *,
    capacity_multiplier: float,
    pipeline_efficiency: float = 0.5,
) -> float:
    """Project capacity-bound throughput gain from a capacity multiplier.

    The paper observes a 4.7-7.8× capacity gain projecting to 3.4-3.9×
    capacity-bound throughput under a conservative pipelined serving model.
    We model that conservatism with ``pipeline_efficiency`` in ``(0, 1]``:
    ``throughput_gain = 1 + (capacity_multiplier - 1) * pipeline_efficiency``.
    """
    if capacity_multiplier <= 1:
        return max(capacity_multiplier, 0.0)
    return 1.0 + (capacity_multiplier - 1.0) * pipeline_efficiency


def burst_throughput_gain(
    dense_req_per_s: float, redknot_req_per_s: float
) -> Dict[str, float]:
    """Burst-mode throughput delta (paper fig. 13b)."""
    gain = (
        (redknot_req_per_s - dense_req_per_s) / dense_req_per_s
        if dense_req_per_s
        else 0.0
    )
    return {
        "dense_req_per_s": dense_req_per_s,
        "redknot_req_per_s": redknot_req_per_s,
        "throughput_gain_frac": gain,
    }


__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "list_datasets",
    "coefficient_of_variation",
    "aggregate_quality",
    "aggregate_efficiency",
    "pd_throughput_projection",
    "burst_throughput_gain",
]
