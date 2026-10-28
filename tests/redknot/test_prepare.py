"""Offline preparation invariants, using a reversible fake tokenizer."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from benchmarks.redknot.catalog import ASSETS_ROOT, DATASETS, MODEL_PROFILES, SOURCE_COMMIT, SUITES, get_suite
from benchmarks.redknot.contract import digest, file_digest, seal, validate_manifest
from benchmarks.redknot.prepare import _load_release_cases, _Rows, prepare_manifest, validate_source_assets


class FakeTokenizer:
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) + self.offset for character in value]

    def decode(self, values: list[int], **kwargs: Any) -> str:
        assert kwargs.get("skip_special_tokens") is False
        return "".join(chr(value - self.offset) for value in values)


def fixture_assets(root: Path) -> str:
    def write(name: str, payload: bytes) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def write_json(name: str, value: dict) -> None:
        write(name, json.dumps(value, sort_keys=True).encode())

    data_rows = [
        {"input": f"What is answer {index}?", "context": f"Document {index}: answer is answer-{index}.", "answers": [f"answer-{index}"]}
        for index in range(40)
    ]
    raw_rows = [json.dumps(row, sort_keys=True).encode() for row in data_rows]
    provenance = {"datasets": {}}
    for dataset in DATASETS:
        payload = b"\n".join(raw_rows) + b"\n"
        write(f"datasets/LongBench/data/{dataset}.jsonl", payload)
        provenance["datasets"][dataset] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    write_json("datasets/LongBench/PROVENANCE.json", provenance)
    chunks = [
        {
            "chunk_index": index,
            "rows": [
                {"row_id": row_id, "row_sha256": hashlib.sha256(raw_rows[row_id]).hexdigest()} for row_id in (2 * index, 2 * index + 1)
            ],
        }
        for index in range(4)
    ]
    queries = [
        {
            "row_id": row_id,
            "expected_chunk": row_id // 2,
            "row_sha256": hashlib.sha256(raw_rows[row_id]).hexdigest(),
            "question_sha256": hashlib.sha256(data_rows[row_id]["input"].encode()).hexdigest(),
            "answers_sha256": digest(data_rows[row_id]["answers"]),
        }
        for row_id in (0, 2)
    ]
    selection = {"dataset": {"name": "hotpotqa"}, "selection": {"chunks": chunks, "queries": queries}}
    selection["selection_sha256"] = digest(selection)
    cohort = "datasets/LongBench/cohort-fixture"
    write_json(f"{cohort}/data_selection.json", selection)
    write_json(
        f"{cohort}/profile.json",
        {"dataset": "hotpotqa", "data_selection_sha256": selection["selection_sha256"], "data_selection": "/unavailable/old/machine.json"},
    )
    write_json(f"{cohort}/prompt_manifest.json", {"protocol": {"name": "source-0731-fixture"}})
    cases = [
        {
            "case_id": f"source-row{row_id}",
            "length": "64K",
            "dataset": "hotpotqa",
            "profile": "../cohort-fixture/profile.json",
            "query_index": index,
            "query_row_id": row_id,
            "question": data_rows[row_id]["input"],
            "answers": data_rows[row_id]["answers"],
            "full_input_ids_sha256": "0" * 64,
            "selection_origin": "legacy-curated-fixture",
            "eligible_for_accuracy_aggregate": index == 0,
            "target_output_tokens": 0 if index == 0 else 30,
        }
        for index, row_id in enumerate((0, 2))
    ]
    suite_name = "release_64k_15case.jsonl"
    suite_path = write("datasets/LongBench/suites/" + suite_name, ("\n".join(json.dumps(case) for case in cases) + "\n").encode())
    suite_sha = file_digest(suite_path)
    write("datasets/LongBench/suites/" + suite_name + ".sha256", (suite_sha + "  " + suite_name + "\n").encode())
    write_json(
        "datasets/LongBench/suites/RELEASE_SUITES.json",
        {
            "suites": {"64K": {"path": suite_name, "sha256": suite_sha}},
            "selection_policy": {"cases_per_length": 2},
            "execution": {"pairs": 10},
        },
    )
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            name = str(path.relative_to(root))
            files[name] = {"source_path": "test/srt/redknot/" + name, "bytes": path.stat().st_size, "sha256": file_digest(path)}
    inventory = seal(
        {
            "schema": "redknot-sglang-source-assets-v1",
            "repository": "https://github.com/rednote-machine-learning/RedKnot",
            "commit": SOURCE_COMMIT,
            "files": files,
        }
    )
    write_json("SOURCE.json", inventory)
    return inventory["sha256"]


class PrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_sha = fixture_assets(self.root)

    def prepare(self, **kwargs: Any) -> dict:
        options = {
            "tokenizer_identity": {"fingerprint": "fake-char-v1"},
            "target_tokens": 512,
            "assets_root": self.root,
            "expected_source_digest": self.source_sha,
            "prompt_renderer": lambda content: "<user>" + content + "</user><assistant>",
            "prompt_protocol": {"name": "cpu-fixture-chat-v1"},
        }
        options.update(kwargs)
        tokenizer = options.pop("tokenizer", FakeTokenizer())
        return prepare_manifest("64k", "deepseek-v4.1-flash", tokenizer, **options)

    def test_target_freeze_preserves_source_and_answers(self) -> None:
        result = self.prepare()
        self.assertIs(validate_manifest(result), result)
        self.assertEqual(result["suite"]["case_count"], 2)
        self.assertFalse(result["suite"]["source_token_identical"])
        short, long = result["cases"]
        self.assertEqual(short["references"], ["answer-0"])
        self.assertEqual(short["source"]["query_row_id"], 0)
        self.assertEqual(short["source"]["selection_origin"], "legacy-curated-fixture")
        self.assertEqual(short["source"]["original_input_ids_sha256"], "0" * 64)
        self.assertNotEqual(short["input_ids_sha256"], "0" * 64)
        self.assertTrue(short["source"]["evidence"]["complete_source_context_retained"])
        self.assertIn("Document 0: answer is answer-0.", FakeTokenizer().decode(short["input_ids"], skip_special_tokens=False))
        self.assertTrue(short["scored"])
        self.assertFalse(long["scored"])
        self.assertTrue(long["ignore_eos"])
        self.assertEqual(long["max_new_tokens"], 30)

    def test_identical_assets_and_tokenizer_are_deterministic(self) -> None:
        self.assertEqual(self.prepare(), self.prepare())

    def test_warmup_query_rows_and_hashes_are_disjoint(self) -> None:
        result = self.prepare(warmup_count=2)
        case_rows = {case["source"]["query_row_id"] for case in result["cases"]}
        warmup_rows = {case["source"]["query_row_id"] for case in result["warmup"]}
        self.assertFalse(case_rows & warmup_rows)
        self.assertFalse({case["input_ids_sha256"] for case in result["cases"]} & {case["input_ids_sha256"] for case in result["warmup"]})
        self.assertTrue(all(not case["scored"] for case in result["warmup"]))

    def test_subset_keeps_all_release_queries_excluded_from_warmup(self) -> None:
        result = self.prepare(max_cases=1, warmup_count=38)
        self.assertEqual(len(result["cases"]), 1)
        self.assertFalse({0, 2} & {case["source"]["query_row_id"] for case in result["warmup"]})

    def test_document_budget_and_complete_evidence_mapping(self) -> None:
        result = self.prepare()
        for case in result["cases"]:
            for chunk in case["source"]["target_chunks"]:
                self.assertEqual(chunk["rows"][-1]["end"], 128)
                for row in chunk["rows"]:
                    if row["protected_query_context"]:
                        self.assertTrue(row["complete_context"])
                        self.assertEqual(row["end"] - row["start"], row["target_context_tokens"])
            self.assertGreater(case["input_tokens"], 512)

    def test_budget_that_truncates_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "truncate evidence"):
            self.prepare(target_tokens=64)

    def test_target_input_tampering_is_rejected(self) -> None:
        result = self.prepare()
        result["cases"][0]["input_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            validate_manifest(result)
        with self.assertRaisesRegex(ValueError, "input token hash"):
            validate_manifest(seal(result))

    def test_original_asset_tampering_is_rejected(self) -> None:
        path = self.root / "datasets/LongBench/data/hotpotqa.jsonl"
        path.write_bytes(path.read_bytes().replace(b"Document 0", b"Document X", 1))
        with self.assertRaisesRegex(ValueError, "asset hash mismatch"):
            self.prepare()

    def test_resealed_inventory_cannot_change_pinned_source(self) -> None:
        path = self.root / "SOURCE.json"
        source = json.loads(path.read_text())
        source["repository"] = "https://example.invalid/changed"
        path.write_text(json.dumps(seal(source)))
        with self.assertRaisesRegex(ValueError, "pinned snapshot"):
            self.prepare()

    def test_new_tokenizer_regenerates_ids_without_changing_evidence(self) -> None:
        source = self.prepare()
        target = self.prepare(tokenizer=FakeTokenizer(offset=1000), tokenizer_identity={"fingerprint": "fake-offset-v2"})
        self.assertNotEqual(source["cases"][0]["input_ids_sha256"], target["cases"][0]["input_ids_sha256"])
        self.assertEqual(
            source["cases"][0]["source"]["target_prompt_text_sha256"], target["cases"][0]["source"]["target_prompt_text_sha256"]
        )

    def test_five_dataset_matrix_is_explicit_and_output_blind(self) -> None:
        result = self.prepare(datasets=DATASETS, cases_per_dataset=2)
        self.assertEqual(result["suite"]["kind"], "output_blind_dataset_matrix")
        self.assertEqual(len(result["cases"]), 10)
        self.assertEqual({case["dataset"] for case in result["cases"]}, set(DATASETS))
        for dataset in DATASETS:
            cases = [case for case in result["cases"] if case["dataset"] == dataset]
            self.assertEqual([case["source"]["query_row_id"] for case in cases], [0, 1])
            self.assertTrue(all(case["source"]["selection_origin"] == "physical-row-order-before-inference-v1" for case in cases))

    def test_renderer_is_required_unless_plain_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt_renderer is required"):
            self.prepare(prompt_renderer=None)
        result = self.prepare(prompt_renderer=None, allow_plain_prompt=True)
        self.assertEqual(result["model"]["prompt_protocol"]["name"], "experimental-plain-rag-v1")

    def test_renderer_cannot_drop_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "retain the entire user content"):
            self.prepare(prompt_renderer=lambda _: "only question")

    def test_invalid_budget_and_counts(self) -> None:
        for options in ({"target_tokens": 513}, {"warmup_count": -1}, {"max_cases": 0}, {"cases_per_dataset": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)

    def test_six_profiles_only_current_v41_is_native(self) -> None:
        self.assertEqual(len(MODEL_PROFILES), 6)
        self.assertEqual([name for name, profile in MODEL_PROFILES.items() if profile["native_supported"]], ["deepseek-v4.1-flash"])


class CommittedSourceTests(unittest.TestCase):
    def test_committed_snapshot_integrity(self) -> None:
        source = validate_source_assets()
        self.assertEqual(source["commit"], SOURCE_COMMIT)
        self.assertEqual(len(source["files"]), 152)

    def test_all_sixty_frozen_case_source_mappings(self) -> None:
        rows = _Rows(ASSETS_ROOT)
        for name in SUITES:
            with self.subTest(suite=name):
                cases, _ = _load_release_cases(ASSETS_ROOT.resolve(), get_suite(name), rows)
                self.assertEqual(len(cases), 15)
                self.assertEqual(sum(case["case"]["eligible_for_accuracy_aggregate"] for case in cases), 10)


if __name__ == "__main__":
    unittest.main()
