import itertools
import unittest
from fractions import Fraction

from kv_cache_engram.core import ValidationError
from kv_cache_engram.coverage import (DETERMINISTIC, UNIFORM, miss_probability,
                                     plan_coverage, required_probes)


class CoverageTests(unittest.TestCase):
    def test_exact_probability_against_exhaustive_small_populations(self):
        for n in range(1, 9):
            for k in range(1, n + 1):
                for m in range(n + 1):
                    samples = list(itertools.combinations(range(n), m))
                    misses = sum(not set(sample).intersection(range(k)) for sample in samples)
                    self.assertEqual(miss_probability(n, k, m), Fraction(misses, len(samples)))

    def test_single_spike_needs_almost_all_rows(self):
        self.assertEqual(miss_probability(512, 1, 8), Fraction(504, 512))
        self.assertEqual(required_probes(512, max_miss_probability=0.01), 507)
        self.assertEqual(required_probes(32768, max_miss_probability=0.01), 32441)

    def test_zero_risk_is_exact_not_underflow(self):
        self.assertEqual(required_probes(512, min_bad_rows=32, max_miss_probability=0), 481)
        self.assertGreater(miss_probability(512, 32, 480), 0)
        self.assertEqual(miss_probability(512, 32, 481), 0)

    def test_required_is_minimal(self):
        for n in (4, 16, 512):
            for k in (1, n // 2, n):
                for risk in (0, 0.001, 0.1, 0.9):
                    m = required_probes(n, min_bad_rows=k, max_miss_probability=risk)
                    self.assertLessEqual(miss_probability(n, k, m), Fraction(risk))
                    if m:
                        self.assertGreater(miss_probability(n, k, m - 1), Fraction(risk))

    def test_default_deterministic_never_claims_uniform_bound(self):
        result = plan_coverage(512, available_probes=511, max_miss_probability=0.01)
        self.assertEqual(result.sampling_scheme, DETERMINISTIC)
        self.assertEqual(result.recommendation, "full_recompute")
        self.assertFalse(result.to_dict()["runtime_admission_created"])

    def test_insufficient_budget_falls_back(self):
        result = plan_coverage(512, available_probes=8, max_miss_probability=0.01,
                               sampling_scheme=UNIFORM)
        self.assertEqual(result.reason, "probe_budget_insufficient")
        self.assertEqual(result.recommendation, "full_recompute")

    def test_sufficient_budget_is_not_admission(self):
        result = plan_coverage(512, available_probes=507, max_miss_probability=0.01,
                               sampling_scheme=UNIFORM)
        self.assertEqual(result.recommendation, "collect_uniform_probes")
        self.assertFalse(result.to_dict()["runtime_admission_created"])

    def test_full_probe_does_not_claim_reuse(self):
        result = plan_coverage(512, available_probes=512, max_miss_probability=0,
                               sampling_scheme=UNIFORM)
        self.assertEqual(result.reason, "no_unprobed_rows_would_remain")

    def test_invalid_parameters(self):
        for args in ((True, 1, 0), (0, 1, 0), (5, 0, 1), (5, 6, 1),
                     (5, 1, 6), (5, 1, -1), (65537, 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                miss_probability(*args)
        for risk in (float("nan"), float("inf"), -0.1, 1, True, "0.01", 10 ** 1000):
            with self.subTest(risk=risk), self.assertRaises(ValidationError):
                required_probes(512, max_miss_probability=risk)


if __name__ == "__main__":
    unittest.main()
