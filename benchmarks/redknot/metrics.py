"""Standard-library metrics for English QA and native request measurements.

Scores are fractions in [0, 1]. Repeated observations are reported separately
from unique samples; a larger repetition count is not a larger quality set.
"""

from __future__ import annotations

import math
import re
import statistics
import string
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

PERCENTILE_METHOD = "linear interpolation at (n - 1) * q; Hyndman-Fan type 7"


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return float(value)


def normalize_english_answer(text: str) -> str:
    """SQuAD-style English normalization: case, punctuation, articles, spaces."""
    if not isinstance(text, str):
        raise ValueError("answer must be text")
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_english_qa(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """Take the best EM and token F1 across reference answers independently."""
    if not isinstance(references, (list, tuple)) or not references or any(not isinstance(ref, str) for ref in references):
        raise ValueError("reference answers must be a nonempty list of strings")
    normalized = normalize_english_answer(prediction)
    tokens = normalized.split()
    exact, f1_values = [], []
    for reference in references:
        gold = normalize_english_answer(reference)
        gold_tokens = gold.split()
        overlap = sum((Counter(tokens) & Counter(gold_tokens)).values())
        exact.append(float(normalized == gold))
        f1_values.append(2.0 * overlap / (len(tokens) + len(gold_tokens)) if tokens or gold_tokens else 1.0)
    return {"answer_em": max(exact), "answer_token_f1": max(f1_values)}


def percentile(values: Sequence[float], quantile: float) -> float:
    q = finite_number(quantile, "quantile")
    if not 0 <= q <= 1 or not values:
        raise ValueError("percentile requires observations and a quantile in [0, 1]")
    ordered = sorted(finite_number(value, "percentile observation") for value in values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def _quality(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record["scored"]]
    scores = [score_english_qa(record["text"], record["references"]) for record in scored]
    return {
        "scored_observations": len(scored),
        "unique_scored_samples": len({record["sample_id"] for record in scored}),
        "unscored_observations": len(records) - len(scored),
        "answer_em": statistics.fmean(score["answer_em"] for score in scores) if scores else None,
        "answer_token_f1": statistics.fmean(score["answer_token_f1"] for score in scores) if scores else None,
    }


def summarize_requests(requests: Sequence[Mapping[str, Any]], *, elapsed_s: float) -> dict[str, Any]:
    """Summarize a complete, flattened multi-iteration / multi-DP cohort.

    ``elapsed_s`` is the sum of synchronized global phase durations. Never pass
    a sum of rank durations or sum rank-local QPS. Long fixed-length showcase
    outputs must have ``scored=False`` and do not enter answer quality.
    """
    duration = finite_number(elapsed_s, "global elapsed_s", positive=True)
    if not isinstance(requests, (list, tuple)) or not requests:
        raise ValueError("a completed run requires request observations")
    seen, by_dataset = set(), defaultdict(list)
    ttfts, latencies = [], []
    for record in requests:
        if not isinstance(record, Mapping):
            raise ValueError("request must be an object")
        sample_id, dataset = record.get("sample_id"), record.get("dataset")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(dataset, str) or not dataset:
            raise ValueError("request requires sample_id and dataset")
        iteration, rank = record.get("iteration"), record.get("dp_rank")
        if type(iteration) is not int or iteration < 0 or type(rank) is not int or rank < 0:
            raise ValueError("request requires nonnegative iteration and dp_rank")
        identity = (iteration, rank, sample_id)
        if identity in seen:
            raise ValueError("duplicate request observation")
        seen.add(identity)
        if record.get("finished") is not True or record.get("error"):
            raise ValueError("unfinished or failed request cannot enter a successful summary")
        if "finish_reason" in record and record["finish_reason"] not in ("stop", "length"):
            raise ValueError("failed or unknown finish reason cannot enter a successful summary")
        scored, ignore_eos, budget = record.get("scored"), record.get("ignore_eos"), record.get("max_new_tokens")
        if type(scored) is not bool or type(ignore_eos) is not bool or (scored and ignore_eos):
            raise ValueError("invalid QA / fixed-output scoring policy")
        if type(budget) is not int or budget <= 0:
            raise ValueError("invalid request output budget")
        tokens = record.get("output_token_ids")
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= budget
            or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens)
            or (ignore_eos and len(tokens) != budget)
        ):
            raise ValueError("invalid or incomplete output token sequence")
        if ignore_eos and "finish_reason" in record and record["finish_reason"] != "length":
            raise ValueError("fixed-output request did not terminate at its length budget")
        if not isinstance(record.get("text"), str):
            raise ValueError("decoded output text is required")
        if scored:
            score_english_qa(record["text"], record.get("references"))
        ttft = finite_number(record.get("ttft_s"), "request TTFT", positive=True)
        latency = finite_number(record.get("latency_s"), "request latency", positive=True)
        if ttft > latency or latency > duration + max(1e-8, duration * 1e-8):
            raise ValueError("request timing is outside the global measurement window")
        ttfts.append(ttft)
        latencies.append(latency)
        by_dataset[dataset].append(record)
    output_tokens = sum(len(record["output_token_ids"]) for record in requests)
    return {
        "completed_requests": len(requests),
        "unique_samples": len({record["sample_id"] for record in requests}),
        "elapsed_s": duration,
        "completed_requests_per_s": len(requests) / duration,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / duration,
        "throughput_semantics": "all completed requests divided by synchronized global elapsed time",
        "fixed_output_qps_eligible": all(record["ignore_eos"] and not record["scored"] for record in requests),
        "ttft_s": _distribution(ttfts),
        "latency_s": _distribution(latencies),
        "percentile_method": PERCENTILE_METHOD,
        "quality_scale": "fraction_0_to_1",
        "quality_aggregation": "mean over scored request observations; repetitions are not independent samples",
        "quality": _quality(requests),
        "datasets": {dataset: _quality(records) for dataset, records in sorted(by_dataset.items())},
    }
