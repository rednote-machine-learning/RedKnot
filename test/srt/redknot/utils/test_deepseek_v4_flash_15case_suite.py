#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SPEC = importlib.util.spec_from_file_location(
    "redknot_release_entry", HERE.parent / "benchmark_RedKnot_DeepSeekV4Flash.py"
)
assert SPEC is not None and SPEC.loader is not None
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


@unittest.skipUnless(
    all(path.is_file() for path in ENTRY.DEFAULT_SUITES)
    and all(
        (ENTRY.DATA_DIR / f"{dataset}.jsonl").is_file()
        for dataset in ("hotpotqa", "musique", "multifieldqa_en")
    ),
    "packaged suite data is absent",
)
class TestRelease15CaseSuites(unittest.TestCase):
    def test_real_suite_and_profiles_are_exact(self) -> None:
        for length, suite_path in zip(
            ("64K", "128K", "256K", "440K"),
            ENTRY.DEFAULT_SUITES,
            strict=True,
        ):
            cases, groups = ENTRY._load_suite_jsonl(suite_path)
            self.assertEqual(len(cases), 15)
            self.assertEqual({case["length"] for case in cases}, {length})
            expected_group_sizes = (
                [1] * 15 if length in ("64K", "128K") else [10, 2, 2, 1]
            )
            self.assertEqual(
                [len(group["cases"]) for group in groups], expected_group_sizes
            )
            self.assertEqual(
                [case["target_output_tokens"] for case in cases],
                [0] * 10 + [30] * 5,
            )
            self.assertTrue(
                all(case["eligible_for_accuracy_aggregate"] for case in cases[:10])
            )
            self.assertTrue(
                all(
                    not case["eligible_for_accuracy_aggregate"] for case in cases[10:]
                )
            )

    def test_report_contains_exactly_fifteen_ordered_pairs(self) -> None:
        for length, suite_path in zip(
            ("64K", "128K", "256K", "440K"),
            ENTRY.DEFAULT_SUITES,
            strict=True,
        ):
            cases, groups = ENTRY._load_suite_jsonl(suite_path)
            runs = []
            for group in groups:
                pairs = []
                for case in group["cases"]:
                    pairs.append(
                        {
                            "query_index": case["query_index"],
                            "repeat": 0,
                            "suite_case_index": case["suite_case_index"],
                            "case_id": case["case_id"],
                            "selection_origin": case["selection_origin"],
                            "eligible_for_accuracy_aggregate": case[
                                "eligible_for_accuracy_aggregate"
                            ],
                            "question": case["question"],
                            "recomputed_output": f"dense-{case['suite_case_index']}",
                            "redknot_output": f"redknot-{case['suite_case_index']}",
                            "recomputed_output_tokens": 1,
                            "redknot_output_tokens": 1,
                        }
                    )
                runs.append(
                    {
                        "length": length,
                        "dataset": group["dataset"],
                        "run_id": group["group_id"],
                        "long_output_target_tokens": group["long_output_tokens"],
                        "metrics": {},
                        "output_comparisons": pairs,
                    }
                )
            release = {
                "run_mode": f"release_{length.lower()}_15case_suite",
                "lengths": [length],
                "runs": runs,
            }
            with tempfile.TemporaryDirectory() as tmp:
                report = Path(tmp) / "comparison.md"
                ENTRY._write_comparison_report(report, release)
                text = report.read_text(encoding="utf-8")
            self.assertIn(f"# Ordered {length} suite outputs", text)
            self.assertEqual(text.count("## Case "), 15)
            offsets = [
                text.index(f"## Case {index + 1:02d}:") for index in range(15)
            ]
            self.assertEqual(offsets, sorted(offsets))
            self.assertNotIn("Long-output showcase (post-hoc, non-scoring)", text)

    def test_default_dispatches_all_four_lengths_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            ENTRY, "HERE", Path(tmp)
        ), mock.patch.object(ENTRY.subprocess, "run") as run:
            run.return_value.returncode = 0
            status = ENTRY._run_default_release_suites()
            summaries = list(
                Path(tmp).glob(
                    "results/deepseek_v4_flash/*/four_length_summary.json"
                )
            )
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(
            summary["ordered_lengths"], ["64K", "128K", "256K", "440K"]
        )
        self.assertEqual(run.call_count, 4)
        commands = [call.args[0] for call in run.call_args_list]
        for command, length in zip(
            commands, ("64k", "128k", "256k", "440k"), strict=True
        ):
            self.assertIn(f"release_{length}_15case.jsonl", " ".join(command))


if __name__ == "__main__":
    unittest.main()
