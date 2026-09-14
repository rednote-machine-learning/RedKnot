"""CPU contracts for the vLLM-only Flash release entrypoint.

No tokenization, installed model, torch, vLLM, network or GPU is needed. Test
hash/selection preservation, all-capture budget, new-file semantics and GPU
opt-in boundaries rather than inventing model accuracy or throughput results.
"""

import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import benchmark_RedKnot_DeepSeekV4Flash as entry
from benchmarks import flash_reproduction as prep
from benchmarks.benchmark_redknot import parse_cases
from vllm_redknot.config import load_config


def fixture():
    raw = {
        "cases": [
            {
                "id": "64K:test",
                "chunks": [[1, 2], [3, 4]],
                "query": [5],
                "references": ["correct answer"],
            }
        ]
    }
    row = {
        "id": "64K:test",
        "num_chunks": 2,
        "chunk_tokens": 2,
        "total_tokens": 5,
        "query_tokens": 1,
        "offline_chunk_hashes": [prep.token_hash([1, 2]), prep.token_hash([3, 4])],
        "full_input_ids_sha256": prep.token_hash([1, 2, 3, 4, 5]),
        "query_hash": prep.token_hash([5]),
        "answers": ["correct answer"],
        "eligible_for_accuracy_aggregate": True,
    }
    return raw, [row]


class FlashReproductionTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_config(
            prep.HERE.parent / "examples/deepseek_v4_flash_policy.json"
        )

    def test_catalog_has_frozen_four_by_fifteen(self):
        all_rows = prep.load_catalog("all")
        self.assertEqual(len(all_rows), 60)
        self.assertEqual(len({row["id"] for row in all_rows}), 60)
        for length in prep.LENGTHS:
            rows = prep.load_catalog(length)
            self.assertEqual(len(rows), 15)
            self.assertEqual(
                sum(row["eligible_for_accuracy_aggregate"] for row in rows), 10
            )
            self.assertTrue(all(row["total_tokens"] > 65536 - 1 for row in rows))

    def test_catalog_is_integrity_pinned(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "changed.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prep._pinned_json(path, prep.CATALOG_SHA256)

    def test_model_inventory_contains_all_74_pinned_files(self):
        manifest = prep.model_file_inventory()
        self.assertEqual(sum(row["size"] for row in manifest["files"]), 166898661074)
        self.assertEqual(
            sum(row["path"].endswith(".safetensors") for row in manifest["files"]), 48
        )

    def test_original_uint32_hash_encoding_and_types(self):
        expected = hashlib.sha256(b"\x01\x00\x00\x00\xff\xff\xff\xff").hexdigest()
        self.assertEqual(prep.token_hash([1, 2**32 - 1]), "sha256:" + expected)
        for bad in (True, -1, 2**32, 0.5):
            with self.assertRaises(ValueError):
                prep.token_hash([bad])

    def test_exact_conversion_and_references(self):
        raw, catalog = fixture()
        before = copy.deepcopy(raw)
        converted = prep.convert_cases(raw, catalog)
        self.assertEqual(converted["cases"], raw["cases"])
        self.assertEqual(raw, before)
        self.assertFalse(converted["provenance"]["retokenized"])

    def test_missing_and_extra_frozen_cases_fail(self):
        raw, catalog = fixture()
        raw["cases"][0]["id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            prep.convert_cases(raw, catalog)

    def test_reordering_one_token_or_query_fails(self):
        for field in ("chunks", "query"):
            raw, catalog = fixture()
            if field == "chunks":
                raw["cases"][0][field][0] = [2, 1]
            else:
                raw["cases"][0][field] = [9]
            with self.assertRaisesRegex(ValueError, "hash"):
                prep.convert_cases(raw, catalog)

    def test_short_reference_is_not_scored_as_long_answer_gold(self):
        raw, catalog = fixture()
        catalog[0]["eligible_for_accuracy_aggregate"] = False
        result = prep.convert_cases(raw, catalog)
        self.assertEqual(result["cases"][0]["references"], [])
        self.assertEqual(
            result["provenance"]["cases"][0]["answers"], ["correct answer"]
        )

    def test_wrong_supplied_gold_is_rejected(self):
        raw, catalog = fixture()
        raw["cases"][0]["references"] = ["made-up"]
        with self.assertRaisesRegex(ValueError, "references differ"):
            prep.convert_cases(raw, catalog)

    def test_real_64k_budget_is_blocked_not_truncated(self):
        plan = prep.cache_plan(self.policy, catalog=prep.load_catalog("64K"))
        self.assertEqual(plan["payload_bytes_per_cached_token"], 606504)
        self.assertEqual(plan["budget_token_capacity"], 14163)
        self.assertTrue(any("exceeds CPU cache budget" in b for b in plan["blockers"]))

    def test_all_unique_capture_budget_counts_shared_chunks_once(self):
        raw = {
            "cases": [
                {"id": "a", "chunks": [list(range(256))], "query": [300]},
                {"id": "b", "chunks": [list(range(256))], "query": [301]},
            ]
        }
        plan = prep.cache_plan(self.policy, cases=parse_cases(raw))
        self.assertEqual(plan["unique_chunks"], 1)
        self.assertEqual(plan["required_all_capture_payload_bytes"], 256 * 606504)
        self.assertEqual(plan["blockers"], [])

    def test_no_clean_rows_are_blocked(self):
        raw, _ = fixture()
        plan = prep.cache_plan(self.policy, cases=parse_cases(raw))
        self.assertTrue(any("no clean rows" in b for b in plan["blockers"]))

    def test_model_file_checks_pass_sha_and_fail_partial_missing_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "weight"
            path.write_bytes(b"123")
            spec = [
                {
                    "path": "weight",
                    "size": 3,
                    "sha256": hashlib.sha256(b"123").hexdigest(),
                }
            ]
            self.assertEqual(prep.verify_local_files(root, spec), spec)
            path.write_bytes(b"456")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                prep.verify_local_files(root, spec)
            path.write_bytes(b"1")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                prep.verify_local_files(root, spec)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                prep.verify_local_files(root, spec)
            path.symlink_to(root / "other")
            with self.assertRaisesRegex(ValueError, "symlink"):
                prep.verify_local_files(root, spec)

    def test_model_manifest_path_escape_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unsafe"):
                prep.verify_local_files(Path(temp), [{"path": "../escape"}])

    def test_gpu_environment_is_single_device_and_single_rank(self):
        prep.validate_gpu_environment({"CUDA_VISIBLE_DEVICES": "0"})
        for env in (
            {},
            {"CUDA_VISIBLE_DEVICES": "0,1"},
            {"CUDA_VISIBLE_DEVICES": "0", "WORLD_SIZE": "8"},
        ):
            with self.assertRaises(ValueError):
                prep.validate_gpu_environment(env)

    def test_dry_run_uses_no_vllm_or_torch(self):
        code = (
            "import sys; "
            "from benchmarks.benchmark_RedKnot_DeepSeekV4Flash import main; "
            "main(['--dry-run']); assert 'torch' not in sys.modules; "
            "assert 'vllm' not in sys.modules; assert 'sglang' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=prep.HERE.parent,
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["case_count"], 60)
        self.assertFalse(report["gpu_initialized"])
        self.assertTrue(report["blockers"])

    def test_prepare_writes_new_exact_custom_cases_without_run(self):
        with tempfile.TemporaryDirectory() as temp:
            input_path, output = Path(temp) / "input.json", Path(temp) / "out.json"
            raw, _ = fixture()
            input_path.write_text(json.dumps(raw))
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(entry, "run_gpu") as run,
            ):
                result = entry.main(
                    [
                        "--suite",
                        "custom",
                        "--cases",
                        str(input_path),
                        "--prepare-only",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            run.assert_not_called()
            self.assertEqual(json.loads(output.read_text())["cases"], raw["cases"])
            with self.assertRaises(FileExistsError):
                entry.new_path(output)

    def test_run_refuses_without_idle_optin(self):
        args = entry.build_parser().parse_args(["--run"])
        with self.assertRaisesRegex(ValueError, "gpu-confirmed-idle"):
            entry.run_gpu(args, {}, {})

    def test_fixed_output_and_warmup_protocol_cannot_be_weakened(self):
        for option in ("--warmup-pairs", "--measured-pairs", "--output-tokens"):
            args = entry.build_parser().parse_args([option, "1"])
            with self.assertRaises(ValueError):
                entry.prepare(args)

    def test_ratio_of_means_is_separate_from_mean_paired_ratio(self):
        result = entry.aggregate_means(
            {
                "measurements": [
                    {
                        "case_id": "a",
                        "pair_index": 0,
                        "mode": "dense",
                        "ttft_seconds": 2,
                    },
                    {
                        "case_id": "a",
                        "pair_index": 0,
                        "mode": "reuse",
                        "ttft_seconds": 1,
                    },
                    {
                        "case_id": "a",
                        "pair_index": 1,
                        "mode": "dense",
                        "ttft_seconds": 12,
                    },
                    {
                        "case_id": "a",
                        "pair_index": 1,
                        "mode": "reuse",
                        "ttft_seconds": 3,
                    },
                ],
                "summary": {"f1_drop_percentage_points": 0.5},
            }
        )
        self.assertEqual(result["ratio_of_mean_ttft_dense_over_reuse"], 3.5)
        self.assertEqual(result["mean_paired_ttft_ratio_dense_over_reuse"], 3)


if __name__ == "__main__":
    unittest.main()
