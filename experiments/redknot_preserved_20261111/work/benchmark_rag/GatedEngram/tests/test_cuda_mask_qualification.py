"""CPU validation of the additional strict CUDA-test contract, not GPU proof."""
import ast
import copy
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_pilot import cuda_mask_qualification as check
from kv_cache_engram.core import digest, ValidationError


def fixture():
    return dict(schema="ffn-only-cuda-mask-qualification-v1", passed=True,
        device_type="cuda", device_index=0, quality_eligible=False, timing_valid=False,
        checks=[dict(name=name, rows=rows, selected_rows=len(ids), selected_ids_sha256=digest(ids),
            all_selected=all_selected, sync_debug_mode=2, exact_mask=True,
            device_preserved=True, dtype_bool=True) for name, rows, ids, all_selected in check.mask_cases()])


class CudaMaskContractTests(unittest.TestCase):
    def test_cases_cover_empty_partial_full_and_original_127(self):
        cases = check.mask_cases()
        self.assertEqual([len(ids) for _, _, ids, _ in cases], [0, 0, 127, 1229, 8192, 8192, 1])
        for _, rows, ids, _ in cases:
            self.assertEqual(ids, sorted(set(ids)))
            self.assertTrue(all(0 <= i < rows for i in ids))

    def test_exact_contract_accepts_only_strict_cuda_and_all_cases(self):
        self.assertTrue(check.validate_report(fixture())["passed"])
        for key, value in (("device_type", "cpu"), ("passed", False), ("quality_eligible", True), ("timing_valid", True)):
            altered = fixture(); altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                check.validate_report(altered)
        for field, value in (("sync_debug_mode", 0), ("exact_mask", False), ("device_preserved", False),
                             ("dtype_bool", False), ("selected_rows", 15), ("selected_ids_sha256", "0"*64)):
            altered = fixture(); altered["checks"][2][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                check.validate_report(altered)

    def test_missing_duplicate_or_reordered_cases_rejected(self):
        for mode in ("missing", "duplicate", "reordered"):
            altered = fixture()
            if mode == "missing": altered["checks"].pop()
            elif mode == "duplicate": altered["checks"][1] = copy.deepcopy(altered["checks"][0])
            else: altered["checks"].reverse()
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                check.validate_report(altered)

    def test_same_production_helper_inside_strict_region_not_allow_sync(self):
        tree = ast.parse(inspect.getsource(check.run_cuda_mask_qualification))
        strict = next(node for node in ast.walk(tree) if isinstance(node, ast.Try))
        calls = [node for node in ast.walk(strict) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(c.func, ast.Name) and c.func.id == "build_ffn_mask" for c in calls))
        self.assertFalse(any(isinstance(c.func, ast.Name) and c.func.id == "allow_sync" for c in calls))
        self.assertTrue(any(isinstance(c.func, ast.Attribute) and c.func.attr == "set_sync_debug_mode"
                            and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "error" for c in calls))
        self.assertTrue(strict.finalbody, "restore previous CUDA sync-debug mode after success or failure")

    def test_reader_requires_eight_ranks_and_current_execution(self):
        from gpu_pilot import read_ffn_only_results as reader
        value = dict(schema="ffn-only-cuda-mask-all-ranks-v1", passed=True,
            execution_configuration_sha256="e"*64, quality_eligible=False, timing_valid=False,
            reports=[dict(rank=i, report=fixture()) for i in range(8)])
        preflight = dict(identity=dict(execution_configuration_sha256="e"*64))
        with patch.object(reader, "read", return_value=value):
            self.assertTrue(reader.cuda_mask_qualification(Path("unused"), preflight, {})["passed"])
        for mode in ("missing", "duplicate", "wrong_execution", "wrong_mode"):
            altered = copy.deepcopy(value)
            if mode == "missing": altered["reports"].pop()
            elif mode == "duplicate": altered["reports"][7]["rank"] = 0
            elif mode == "wrong_execution": altered["execution_configuration_sha256"] = "f"*64
            else: altered["reports"][0]["report"]["checks"][0]["sync_debug_mode"] = 0
            with self.subTest(mode=mode), patch.object(reader, "read", return_value=altered), self.assertRaises(ValidationError):
                reader.cuda_mask_qualification(Path("unused"), preflight, {})
