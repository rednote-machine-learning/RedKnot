from __future__ import annotations

import copy
import unittest

from benchmarks.redknot.metrics import percentile, score_english_qa, summarize_requests


def observation(sample_id: str = "short", **changes: object) -> dict:
    return {
        "iteration": 0,
        "dp_rank": 0,
        "sample_id": sample_id,
        "dataset": "hotpotqa",
        "references": ["The Amazon River", "Amazon"],
        "scored": True,
        "ignore_eos": False,
        "max_new_tokens": 16,
        "output_token_ids": [11, 12],
        "text": "Amazon!",
        "ttft_s": 1.0,
        "latency_s": 2.0,
        "finished": True,
        **changes,
    }


class MetricsTests(unittest.TestCase):
    def test_english_normalization_and_best_reference(self) -> None:
        self.assertEqual(score_english_qa("  THE Amazon! ", ["Amazon river", "an Amazon"]), {"answer_em": 1.0, "answer_token_f1": 1.0})
        self.assertAlmostEqual(score_english_qa("blue blue red", ["blue red red"])["answer_token_f1"], 2 / 3)

    def test_empty_prediction_has_zero_f1_against_nonempty_gold(self) -> None:
        self.assertEqual(score_english_qa("", ["Mercury"]), {"answer_em": 0.0, "answer_token_f1": 0.0})
        with self.assertRaises(ValueError):
            score_english_qa("Mercury", [])

    def test_percentiles_are_interpolated_and_declared(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.95), 3.85)
        self.assertEqual(percentile([4], 0.95), 4)
        for values, quantile in (([], 0.5), ([1], 1.5), ([float("nan")], 0.5)):
            with self.subTest(values=values, quantile=quantile), self.assertRaises(ValueError):
                percentile(values, quantile)

    def test_showcase_does_not_contaminate_quality(self) -> None:
        rows = [observation(), observation("long", scored=False, ignore_eos=True, max_new_tokens=2, text="wrong")]
        result = summarize_requests(rows, elapsed_s=4)
        self.assertEqual(result["quality"]["answer_em"], 1.0)
        self.assertEqual(result["quality"]["scored_observations"], 1)
        self.assertEqual(result["quality"]["unscored_observations"], 1)
        self.assertEqual(result["datasets"]["hotpotqa"]["answer_em"], 1.0)
        self.assertFalse(result["fixed_output_qps_eligible"])

    def test_global_qps_uses_global_duration_and_exposes_repetition(self) -> None:
        rows = [observation(iteration=iteration, dp_rank=rank) for iteration in range(2) for rank in range(2)]
        result = summarize_requests(rows, elapsed_s=10)
        self.assertEqual(result["completed_requests_per_s"], 0.4)
        self.assertEqual(result["quality"]["unique_scored_samples"], 1)
        self.assertEqual(result["quality"]["scored_observations"], 4)
        self.assertIn("Hyndman-Fan type 7", result["percentile_method"])

    def test_unscored_only_suite_has_no_quality_value(self) -> None:
        result = summarize_requests([observation(scored=False, ignore_eos=True, max_new_tokens=2)], elapsed_s=4)
        self.assertIsNone(result["quality"]["answer_em"])
        self.assertTrue(result["fixed_output_qps_eligible"])

    def test_invalid_observation_never_looks_successful(self) -> None:
        changes = [
            {"finished": False},
            {"error": "device failed"},
            {"finish_reason": "abort"},
            {"ttft_s": float("inf")},
            {"latency_s": float("nan")},
            {"ttft_s": -1},
            {"ttft_s": 3},
            {"latency_s": 9},
            {"output_token_ids": []},
            {"output_token_ids": [True]},
            {"ignore_eos": True},
            {"scored": "yes"},
            {"scored": False, "ignore_eos": True, "max_new_tokens": 3},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_requests([observation(**change)], elapsed_s=4)
        row = observation()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_requests([row, copy.deepcopy(row)], elapsed_s=4)


if __name__ == "__main__":
    unittest.main()
