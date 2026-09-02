#!/usr/bin/env python3
"""CPU-only contract tests for the isolated 30/50-token supplement."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
CORE = HERE / "benchmark_RedKnot_DeepSeekV4_Flash_RAG.py"
ENTRY = HERE.parent / "benchmark_RedKnot_DeepSeekV4Flash.py"


def _load_query_text(target: int):
    tree = ast.parse(CORE.read_text(encoding="utf-8"), filename=str(CORE))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_query_text"
    )
    namespace = {"IH_LONG_OUTPUT_TOKENS": target}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(CORE), "exec"), namespace)
    return namespace["_query_text"]


def _load_entry():
    sys.path.insert(0, str(HERE.parent))
    try:
        spec = importlib.util.spec_from_file_location("redknot_release_entry", ENTRY)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class TestLongOutputSupplement(unittest.TestCase):
    def test_default_prompt_is_byte_exact_legacy_prompt(self) -> None:
        question = "Which label released the album?"
        self.assertEqual(
            _load_query_text(0)(question),
            "\n\nAnswer the question using only the documents above. "
            "Return the shortest exact answer span only, with no explanation.\n"
            "Question: Which label released the album?\nAnswer:",
        )

    def test_long_prompts_bind_distinct_ranges_and_evidence_style(self) -> None:
        prompt30 = _load_query_text(30)("Q")
        prompt50 = _load_query_text(50)("Q")
        self.assertIn("between 25 and 35 tokens", prompt30)
        self.assertIn("between 45 and 55 tokens", prompt50)
        self.assertIn("direct answer alone on the first line", prompt30)
        self.assertIn("supporting evidence", prompt50)
        self.assertNotEqual(prompt30, prompt50)

    def test_profile_paths_do_not_overlap_standard_profiles(self) -> None:
        entry = _load_entry()
        standard = entry._profile_path("256K", "musique", 3, 0)
        long30 = entry._profile_path("256K", "musique", 3, 30)
        long50 = entry._profile_path("256K", "musique", 3, 50)
        self.assertEqual(standard.parent.parent.name, "256k_3q")
        self.assertEqual(long30.parent.parent.name, "256k_3q_long30")
        self.assertEqual(long50.parent.parent.name, "256k_3q_long50")
        self.assertEqual(len({standard, long30, long50}), 3)

    def test_full_text_comparison_keeps_generated_token_counts(self) -> None:
        entry = _load_entry()
        pairs = entry._extract_output_pairs(
            {
                "queries": [
                    {
                        "query_index": 2,
                        "question": "Q",
                        "repeats": [
                            {
                                "repeat": 0,
                                "dense_text": "dense complete text",
                                "reuse_text": "reuse complete text",
                                "dense_output_tokens": 31,
                                "reuse_output_tokens": 29,
                            }
                        ],
                    }
                ]
            }
        )
        self.assertEqual(pairs[0]["recomputed_output"], "dense complete text")
        self.assertEqual(pairs[0]["redknot_output"], "reuse complete text")
        self.assertEqual(pairs[0]["recomputed_output_tokens"], 31)
        self.assertEqual(pairs[0]["redknot_output_tokens"], 29)

    def test_five_posthoc_examples_are_appended_and_labelled_non_scoring(self) -> None:
        entry = _load_entry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = [
                {
                    "dataset": "musique",
                    "query_row_id": index,
                    "full_input_ids_sha256": f"sha256:{index}",
                    "question": f"Question {index}",
                    "dense_output": f"Answer {index}\nDense evidence.",
                    "redknot_output": f"Answer {index}\nReuse evidence.",
                    "dense_output_tokens": 30,
                    "redknot_output_tokens": 31,
                }
                for index in range(5)
            ]
            (root / "long_output_256k.json").write_text(
                json.dumps(
                    {
                        "format": "redknot_long_output_posthoc_showcase_v1",
                        "selection_is_posthoc": True,
                        "eligible_for_accuracy_aggregate": False,
                        "showcase_sha256": "showcase",
                        "candidate_count": 9,
                        "selected": selected,
                    }
                ),
                encoding="utf-8",
            )
            entry.SHOWCASE_ROOT = root
            report = root / "comparison.md"
            entry._write_comparison_report(
                report,
                {
                    "runs": [
                        {
                            "length": "256K",
                            "dataset": "hotpotqa",
                            "metrics": {},
                            "output_comparisons": [],
                        }
                    ]
                },
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("Long-output showcase (post-hoc, non-scoring)", text)
            self.assertEqual(text.count("### 256K long example"), 5)
            self.assertGreater(
                text.index("Long-output showcase"), text.index("256K / hotpotqa")
            )


if __name__ == "__main__":
    unittest.main()
