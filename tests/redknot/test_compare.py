from __future__ import annotations

import copy
import unittest

from benchmarks.redknot.compare import RUN_SCHEMA, compare_runs, validate_run
from benchmarks.redknot.contract import BOUNDARY, SCHEMA, digest, seal
from benchmarks.redknot.metrics import summarize_requests


def make_case(sample_id: str, tokens: list[int], *, scored: bool = True) -> dict:
    references = ["Mercury"]
    return {
        "sample_id": sample_id,
        "dataset": "hotpotqa",
        "input_ids": tokens,
        "input_ids_sha256": digest(tokens),
        "references": references,
        "references_sha256": digest(references),
        "max_new_tokens": 2,
        "scored": scored,
        "ignore_eos": not scored,
    }


def manifest_fixture(*, fixed_output: bool = False) -> dict:
    return seal(
        {
            "schema": SCHEMA,
            "model": {"profile": "deepseek-v4.1-flash", "model_id": "test/DeepSeek-V4.1-Flash", "tokenizer": {"fingerprint": "a" * 64}},
            "suite": {"name": "unit-fixture"},
            "warmup": [make_case("warmup", [1, 2])],
            "cases": [make_case("short", [3, 4], scored=not fixed_output), make_case("long", [5, 6], scored=False)],
        }
    )


def run_fixture(manifest: dict, mode: str = "baseline") -> dict:
    scale = 1.0 if mode == "baseline" else 0.5
    records = []
    for iteration in range(2):
        for rank in range(2):
            for case in manifest["cases"]:
                records.append(
                    {
                        **{
                            key: case[key]
                            for key in ("sample_id", "dataset", "input_ids_sha256", "references", "max_new_tokens", "scored", "ignore_eos")
                        },
                        "iteration": iteration,
                        "dp_rank": rank,
                        "output_token_ids": [31, 32],
                        "text": "Mercury",
                        "ttft_s": scale,
                        "latency_s": 2 * scale,
                        "finished": True,
                    }
                )
    report = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "mode": mode,
        "manifest_sha256": manifest["sha256"],
        "metadata": {
            "model": manifest["model"],
            "tokenizer": manifest["model"]["tokenizer"],
            "runtime": {"engine": "deepinfer", "git_commit": "a" * 40},
            "settings": {"iterations": 2, "dp_size": 2, "replication": "full_cohort_per_dp_rank"},
            "hardware": {"gpu": "test-fixture", "count": 2},
            "sampling": {"temperature": 0, "seed": 42},
            "warmup": {"iterations": 1, "cases_per_rank": 1, "complete": True},
            "measurement_boundary": BOUNDARY,
        },
        "activation": {"redknot_enabled": mode == "redknot", "verified": True, "implementation": "unit-test-fixture"},
        "phases": [
            {"iteration": i, "elapsed_s": 5 * scale, "rank_elapsed_s": [4 * scale, 5 * scale], "completed_requests": 4} for i in range(2)
        ],
        "requests": records,
        "elapsed_s": 10 * scale,
        "completed_requests": 8,
    }
    if mode == "redknot":
        report["activation"]["configuration_sha256"] = "b" * 64
    report["summary"] = summarize_requests(records, elapsed_s=report["elapsed_s"])
    return report


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = manifest_fixture()
        self.baseline = run_fixture(self.manifest)
        self.redknot = run_fixture(self.manifest, "redknot")

    def test_pairs_by_identity_and_suppresses_uncontrolled_qps(self) -> None:
        self.redknot["requests"].reverse()
        result = compare_runs(self.baseline, self.redknot, self.manifest)
        self.assertEqual(result["ttft_median_paired_speedup"], 2.0)
        self.assertEqual(result["baseline"]["completed_requests_per_s"], 0.8)
        self.assertIsNone(result["fixed_output_qps_speedup"])
        self.assertEqual(result["dataset_quality_deltas"]["hotpotqa"]["unique_scored_samples"], 1)
        self.assertIn("not independent proof", result["limits"][0])

    def test_fixed_output_can_compare_controlled_qps(self) -> None:
        manifest = manifest_fixture(fixed_output=True)
        result = compare_runs(run_fixture(manifest), run_fixture(manifest, "redknot"), manifest)
        self.assertEqual(result["fixed_output_qps_speedup"], 2.0)
        self.assertIsNone(result["baseline"]["quality"]["answer_em"])

    def test_mode_relabel_is_not_an_activation(self) -> None:
        renamed = copy.deepcopy(self.baseline)
        renamed["mode"] = "redknot"
        with self.assertRaisesRegex(ValueError, "activation"):
            compare_runs(self.baseline, renamed, self.manifest)
        self.redknot["activation"].pop("configuration_sha256")
        with self.assertRaisesRegex(ValueError, "attestation"):
            compare_runs(self.baseline, self.redknot, self.manifest)

    def test_matching_omissions_in_both_reports_still_fail(self) -> None:
        for report in (self.baseline, self.redknot):
            report["requests"] = [row for row in report["requests"] if row["sample_id"] != "long"]
            report["completed_requests"] = len(report["requests"])
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            compare_runs(self.baseline, self.redknot, self.manifest)

    def test_duplicate_can_not_replace_a_missing_case(self) -> None:
        self.redknot["requests"][1] = copy.deepcopy(self.redknot["requests"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_runs(self.baseline, self.redknot, self.manifest)

    def test_mismatched_controls_fail(self) -> None:
        for field in ("runtime", "hardware", "sampling"):
            candidate = copy.deepcopy(self.redknot)
            candidate["metadata"][field]["changed"] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "comparison"):
                compare_runs(self.baseline, candidate, self.manifest)

    def test_global_duration_must_not_sum_rank_durations(self) -> None:
        self.redknot["phases"][0]["elapsed_s"] = sum(self.redknot["phases"][0]["rank_elapsed_s"])
        with self.assertRaisesRegex(ValueError, "global phase duration"):
            compare_runs(self.baseline, self.redknot, self.manifest)

    def test_nonfinite_nested_metadata_and_request_fail(self) -> None:
        for mutate in (
            lambda run: run["metadata"]["runtime"].update(extra=float("nan")),
            lambda run: run["requests"][0].update(ttft_s=float("inf")),
        ):
            candidate = copy.deepcopy(self.redknot)
            mutate(candidate)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare_runs(self.baseline, candidate, self.manifest)

    def test_partial_failed_and_cold_reports_fail(self) -> None:
        mutations = [
            lambda run: run.update(status="failed"),
            lambda run: run["requests"][0].update(finished=False),
            lambda run: run["metadata"]["warmup"].update(complete=False),
            lambda run: run["metadata"]["warmup"].update(iterations=0),
            lambda run: run["metadata"].update(measurement_boundary="model_init_to_finish"),
            lambda run: run["requests"][0].update(input_ids_sha256="f" * 64),
        ]
        for mutate in mutations:
            candidate = copy.deepcopy(self.redknot)
            mutate(candidate)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                compare_runs(self.baseline, candidate, self.manifest)

    def test_summary_is_recomputed_and_seals_checked(self) -> None:
        self.redknot["summary"]["completed_requests_per_s"] = 1000
        with self.assertRaisesRegex(ValueError, "derived summary"):
            compare_runs(self.baseline, self.redknot, self.manifest)
        sealed = seal(self.baseline)
        self.assertEqual(validate_run(sealed, self.manifest, mode="baseline")["completed_requests"], 8)
        sealed["elapsed_s"] = 1
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_run(sealed, self.manifest, mode="baseline")

    def test_cold_smoke_can_be_recorded_but_not_compared(self) -> None:
        cold = copy.deepcopy(self.baseline)
        cold["metadata"]["warmup"].update(iterations=0, complete=False, state="cold")
        cold["comparison_eligible"] = False
        self.assertEqual(validate_run(cold, self.manifest, mode="baseline", require_warmup=False)["completed_requests"], 8)
        with self.assertRaisesRegex(ValueError, "warmup"):
            compare_runs(cold, self.redknot, self.manifest)


if __name__ == "__main__":
    unittest.main()
