#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_long_output_showcase as showcase


class TestLongOutputShowcase(unittest.TestCase):
    def test_candidates_require_matching_direct_answer_not_full_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "config": {
                            "dataset": "musique",
                            "full_input_tokens": 262200,
                            "chunk_tokens": 32768,
                            "num_chunks": 8,
                            "long_output_target_tokens": 30,
                            "query_row_ids": [7, 8],
                            "prompt_manifest_sha256": "prompt",
                        },
                        "queries": [
                            {
                                "query_index": 0,
                                "question": "Q0",
                                "dense_text": "Kanine Records\nDense evidence.",
                                "reuse_text": "Kanine Records\nDifferent evidence.",
                                "dense_output_tokens": 30,
                                "reuse_output_tokens": 31,
                                "min_top10_probability_cosine": 0.99,
                                "min_generation_token_agreement": 0.5,
                            },
                            {
                                "query_index": 1,
                                "question": "Q1",
                                "dense_text": "Sun\nEvidence.",
                                "reuse_text": "Jupiter\nEvidence.",
                                "dense_output_tokens": 30,
                                "reuse_output_tokens": 30,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows = showcase._candidates(path, 30)
            self.assertTrue(rows[0]["direct_answer_match"])
            self.assertTrue(rows[0]["length_qualified"])
            self.assertEqual(rows[0]["query_row_id"], 7)
            self.assertFalse(rows[1]["direct_answer_match"])
            self.assertEqual(
                showcase._candidates(path, 30, 450560),
                [],
            )

    def test_wrong_target_is_not_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "config": {"long_output_target_tokens": 50},
                        "queries": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(showcase._candidates(path, 30), [])


if __name__ == "__main__":
    unittest.main()
