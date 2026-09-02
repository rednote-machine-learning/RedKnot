#!/usr/bin/env python3
"""Aggregate paired 256K LongBench dense/RedKnot quality results."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def _bootstrap_mean_ci(values, *, seed=2026, draws=10000):
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(draws)
    )
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    suite = Path(args.suite_dir).resolve()
    rows = []
    datasets = []
    seen_datasets = set()
    for result_path in sorted(suite.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        dataset = result["config"]["dataset"]
        if dataset in seen_datasets:
            raise RuntimeError(f"duplicate dataset result: {dataset}")
        seen_datasets.add(dataset)
        config = result["config"]
        expected_profile = (
            "combined_headsplit_independent_rope_zoff_checkpoint_"
            "rowsparse_3_37_3_v1"
        )
        if (
            config.get("mla_off_execution_profile") != expected_profile
            or config.get("prefix_materialization_scope") != "first_document"
            or config.get("full_input_tokens", 0) < 262144
            or config.get("adaptive_topk") is not True
            or config.get("adaptive_topk_physical_compaction") is not True
            or str(config.get("adaptive_topk_cumulative_mass")) != "0.5"
            or str(config.get("adaptive_topk_buckets")) != "3,4,5,6"
        ):
            raise RuntimeError(
                f"dataset {dataset} is not the frozen 256K combined K3/K4/K5/K6 arm"
            )
        query_rows = result["config"]["query_row_ids"]
        if len(result["queries"]) != len(query_rows):
            raise RuntimeError(
                f"dataset {dataset} query/result cardinality drifted"
            )
        per_dataset_rows = []
        for query, row_id in zip(result["queries"], query_rows):
            row = {
                "dataset": dataset,
                "row_id": int(row_id),
                "question": query["question"],
                "golds": query["golds"],
                "dense_text": query["dense_text"],
                "reuse_text": query["reuse_text"],
                "dense_f1": float(query["dense_f1"]),
                "reuse_f1": float(query["reuse_f1"]),
                "dense_em": float(query["dense_em"]),
                "reuse_em": float(query["reuse_em"]),
                "f1_delta": float(query["reuse_f1"] - query["dense_f1"]),
                "top1_agreement": bool(query["top1_agreement"]),
                "top10_probability_cosine": float(
                    query["min_top10_probability_cosine"]
                ),
                "generation_token_agreement": float(
                    query["min_generation_token_agreement"]
                ),
                "generation_exact_match_rate": float(
                    query["generation_exact_match_rate"]
                ),
                "full_input_ids_sha256": query["full_input_ids_sha256"],
            }
            rows.append(row)
            per_dataset_rows.append(row)
        datasets.append(
            {
                "dataset": dataset,
                "num_queries": len(per_dataset_rows),
                "dense_f1": statistics.fmean(
                    row["dense_f1"] for row in per_dataset_rows
                ),
                "reuse_f1": statistics.fmean(
                    row["reuse_f1"] for row in per_dataset_rows
                ),
                "dense_em": statistics.fmean(
                    row["dense_em"] for row in per_dataset_rows
                ),
                "reuse_em": statistics.fmean(
                    row["reuse_em"] for row in per_dataset_rows
                ),
                "exact_generation_pairs": sum(
                    row["generation_exact_match_rate"] == 1.0
                    for row in per_dataset_rows
                ),
                "min_top10_probability_cosine": min(
                    row["top10_probability_cosine"]
                    for row in per_dataset_rows
                ),
                "min_generation_token_agreement": min(
                    row["generation_token_agreement"]
                    for row in per_dataset_rows
                ),
                "runtime_evidence_pass": bool(
                    result["runtime"]["runtime_evidence_pass"]
                ),
                "fallbacks": int(result["runtime"]["fallbacks"]),
                "online_row_saving": float(
                    result["runtime"]["online_row_saving"]
                ),
                "scoped_mla_head_row_saving": float(
                    result["runtime"]["measured_scoped_mla_head_row_saving"]
                ),
                "full_model_mla_head_row_saving": float(
                    result["runtime"]["measured_full_model_mla_head_row_saving"]
                ),
                "client_ttft_dense_p50_seconds": float(
                    result["latency"]["dense_p50"]
                ),
                "client_ttft_reuse_p50_seconds": float(
                    result["latency"]["reuse_p50"]
                ),
                "client_ttft_speedup": float(
                    result["latency"]["speedup"]
                ),
                "model_ttft_dense_p50_seconds": float(
                    result["latency"]["model_internal"]["dense_p50"]
                ),
                "model_ttft_reuse_p50_seconds": float(
                    result["latency"]["model_internal"]["reuse_p50"]
                ),
                "model_ttft_speedup": float(
                    result["latency"]["model_internal"]["speedup"]
                ),
                "dense_requests_per_second": float(
                    result["sequential_service_rate"][
                        "dense_requests_per_second"
                    ]
                ),
                "reuse_requests_per_second": float(
                    result["sequential_service_rate"][
                        "reuse_requests_per_second"
                    ]
                ),
                "adaptive_topk": {
                    "enabled": bool(result["config"]["adaptive_topk"]),
                    "cumulative_mass": float(
                        result["config"]["adaptive_topk_cumulative_mass"]
                    ),
                    "buckets": str(result["config"]["adaptive_topk_buckets"]),
                    "physical_compaction": bool(
                        result["config"]["adaptive_topk_physical_compaction"]
                    ),
                },
            }
        )
    if not rows:
        raise RuntimeError("suite contains no result.json files")
    deltas = [row["f1_delta"] for row in rows]
    aggregate = {
        "format": "redknot_multidataset_256k_accuracy_v1",
        "suite_dir": str(suite),
        "num_datasets": len(datasets),
        "num_queries": len(rows),
        "datasets": datasets,
        "overall": {
            "dense_f1": statistics.fmean(row["dense_f1"] for row in rows),
            "reuse_f1": statistics.fmean(row["reuse_f1"] for row in rows),
            "dense_em": statistics.fmean(row["dense_em"] for row in rows),
            "reuse_em": statistics.fmean(row["reuse_em"] for row in rows),
            "paired_f1_delta_mean": statistics.fmean(deltas),
            "paired_f1_delta_bootstrap_95ci": _bootstrap_mean_ci(deltas),
            "top1_agreement_rate": statistics.fmean(
                row["top1_agreement"] for row in rows
            ),
            "exact_generation_pair_rate": statistics.fmean(
                row["generation_exact_match_rate"] for row in rows
            ),
            "min_top10_probability_cosine": min(
                row["top10_probability_cosine"] for row in rows
            ),
            "min_generation_token_agreement": min(
                row["generation_token_agreement"] for row in rows
            ),
            "dense_answerable_count": sum(row["dense_f1"] > 0 for row in rows),
            "reuse_answerable_count": sum(row["reuse_f1"] > 0 for row in rows),
            "all_runtime_evidence_pass": all(
                dataset["runtime_evidence_pass"] for dataset in datasets
            ),
            "total_fallbacks": sum(dataset["fallbacks"] for dataset in datasets),
            "min_online_row_saving": min(
                dataset["online_row_saving"] for dataset in datasets
            ),
            "max_online_row_saving": max(
                dataset["online_row_saving"] for dataset in datasets
            ),
            "min_scoped_mla_head_row_saving": min(
                dataset["scoped_mla_head_row_saving"] for dataset in datasets
            ),
            "max_scoped_mla_head_row_saving": max(
                dataset["scoped_mla_head_row_saving"] for dataset in datasets
            ),
            "min_client_ttft_speedup": min(
                dataset["client_ttft_speedup"] for dataset in datasets
            ),
            "median_client_ttft_speedup": statistics.median(
                dataset["client_ttft_speedup"] for dataset in datasets
            ),
            "min_model_ttft_speedup": min(
                dataset["model_ttft_speedup"] for dataset in datasets
            ),
            "median_model_ttft_speedup": statistics.median(
                dataset["model_ttft_speedup"] for dataset in datasets
            ),
        },
        "queries": rows,
    }
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
