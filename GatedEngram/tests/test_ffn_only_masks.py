"""Pure CPU geometry tests; none are evidence of actual model accuracy."""

import json
import math
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from kv_cache_engram.core import ValidationError
from kv_cache_engram.ffn_only_masks import plan_layer_masks


class IndependentMaskTests(unittest.TestCase):
    def plan(self, **changes):
        values = dict(
            input_tokens=1000,
            layer_index=6,
            document_spans=[(10, 410), (430, 930)],
            sensitive_indices=[200, 700],
            mandatory_question_indices=[950],
        )
        values.update(changes)
        return plan_layer_masks(**values)

    def test_first_six_layers_are_full_for_both_paths(self):
        for layer in range(6):
            with self.subTest(layer=layer):
                plan = self.plan(layer_index=layer)
                self.assertEqual(plan.swa_online_rows, tuple(range(1000)))
                self.assertEqual(plan.ffn_selected_rows, tuple(range(1000)))
                self.assertEqual(plan.swa_reuse_rows, ())
                self.assertEqual(plan.ffn_skipped_rows, ())
                self.assertTrue(plan.first_six_native)

    def test_later_swa_exact_prefix127_and_non_document_only(self):
        plan = self.plan()
        expected = set(range(10)) | set(range(410, 430)) | set(range(930, 1000))
        expected |= set(range(10, 137)) | set(range(430, 557))
        self.assertEqual(plan.swa_online_rows, tuple(sorted(expected)))
        for position in (137, 409, 557, 929):
            self.assertIn(position, plan.swa_reuse_rows)
        self.assertIn(136, plan.swa_online_rows)
        self.assertIn(556, plan.swa_online_rows)

    def test_sensitive_ffn_rows_do_not_imply_swa_recomputation(self):
        plan = self.plan()
        self.assertEqual(plan.ffn_selected_rows, (200, 700, 950))
        for position in (200, 700):
            self.assertIn(position, plan.swa_reuse_rows)
        self.assertIn(10, plan.swa_online_rows)
        self.assertIn(10, plan.ffn_skipped_rows)

    def test_no_8k_or_document_tail_guard(self):
        plan = self.plan(input_tokens=32768, document_spans=[(0, 32768)], sensitive_indices=[8192, 32767], mandatory_question_indices=[])
        self.assertEqual(plan.swa_online_rows, tuple(range(127)))
        for position in (8191, 8192, 8193, 16384, 24576, 32640, 32767):
            self.assertIn(position, plan.swa_reuse_rows)
        self.assertEqual(plan.ffn_selected_rows, (8192, 32767))

    def test_each_adjacent_actual_document_has_its_own_prefix(self):
        plan = self.plan(input_tokens=600, document_spans=[(0, 300), (300, 600)], sensitive_indices=[], mandatory_question_indices=[])
        self.assertEqual(plan.swa_online_rows, tuple(range(127)) + tuple(range(300, 427)))
        self.assertIn(299, plan.swa_reuse_rows)
        self.assertIn(427, plan.swa_reuse_rows)

    def test_short_document_prefix_is_clamped_without_spilling(self):
        plan = self.plan(input_tokens=300, document_spans=[(0, 30), (30, 300)], sensitive_indices=[], mandatory_question_indices=[])
        self.assertEqual(plan.swa_online_rows, tuple(range(157)))
        self.assertIn(157, plan.swa_reuse_rows)

    def test_explicit_native_required_override_for_both_paths(self):
        plan = self.plan(layer_index=20, native_required_layer=True)
        self.assertEqual(plan.swa_online_rows, tuple(range(1000)))
        self.assertEqual(plan.ffn_selected_rows, tuple(range(1000)))
        self.assertFalse(plan.first_six_native)
        self.assertTrue(plan.native_required_layer)
        self.assertTrue(plan.full_native)

    def test_no_automatic_yoco_or_global_layer_assumption(self):
        for layer in (6, 8, 14, 20, 39, 100):
            with self.subTest(layer=layer):
                plan = self.plan(layer_index=layer)
                self.assertFalse(plan.full_native)
                self.assertEqual(plan.ffn_selected_rows, (200, 700, 950))
                self.assertFalse(plan.as_dict()["model_layer_eligibility_inferred"])

    def test_duplicates_and_input_order_do_not_affect_deterministic_unique_output(self):
        a = self.plan(sensitive_indices=[700, 200, 700, 200], mandatory_question_indices=[950, 950])
        b = self.plan(sensitive_indices={200, 700}, mandatory_question_indices=(950,))
        self.assertEqual(a, b)
        self.assertEqual(a.sensitive_indices, (200, 700))

    def test_question_mandatory_is_explicit_and_must_be_outside_documents(self):
        plan = self.plan(mandatory_question_indices=[])
        self.assertIn(950, plan.swa_online_rows)
        self.assertIn(950, plan.ffn_skipped_rows)
        with self.assertRaises(ValidationError):
            self.plan(mandatory_question_indices=[200])

    def test_empty_documents_and_no_selected_ffn_are_valid_geometry_only(self):
        plan = self.plan(document_spans=[], sensitive_indices=[], mandatory_question_indices=[])
        self.assertEqual(plan.swa_online_rows, tuple(range(1000)))
        self.assertEqual(plan.ffn_selected_rows, ())
        self.assertFalse(plan.full_native)

    def test_both_paths_partition_full_input(self):
        plan = self.plan()
        for left, right in ((plan.swa_online_rows, plan.swa_reuse_rows), (plan.ffn_selected_rows, plan.ffn_skipped_rows)):
            self.assertFalse(set(left) & set(right))
            self.assertEqual(set(left) | set(right), set(range(1000)))
            self.assertEqual(left, tuple(sorted(set(left))))
            self.assertEqual(right, tuple(sorted(set(right))))

    def test_invalid_input_or_layer_or_override_rejected(self):
        for key, values in (
            ("input_tokens", (0, -1, True, 3.0, math.inf, math.nan)),
            ("layer_index", (-1, True, 6.0, math.inf, math.nan)),
            ("native_required_layer", (0, 1, None, "yes")),
        ):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                    self.plan(**{key: value})

    def test_bad_spans_rejected_without_sorting_clipping_or_merging(self):
        for spans in (
            None,
            "0:10",
            [(0, 0)],
            [(-1, 20)],
            [(0, 1001)],
            [(20, 40), (10, 15)],
            [(0, 30), (29, 50)],
            [(0, 30), (0, 30)],
            [(0, 10, 20)],
            [(False, 10)],
            [(0, 10.0)],
            [(0, math.inf)],
            [(math.nan, 10)],
        ):
            with self.subTest(spans=spans), self.assertRaises(ValidationError):
                self.plan(document_spans=spans)

    def test_bad_indices_rejected_including_bools_nonfinite_and_fractional(self):
        for key in ("sensitive_indices", "mandatory_question_indices"):
            for value in ([-1], [1000], [True], [False], [1.0], [1.5], [math.nan], [math.inf], ["20"], None, "20"):
                with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                    self.plan(**{key: value})

    def test_full_native_does_not_bypass_argument_validation(self):
        for changes in (
            dict(layer_index=0, sensitive_indices=[math.nan]),
            dict(native_required_layer=True, document_spans=[(10, 0)]),
            dict(layer_index=1, mandatory_question_indices=[10]),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.plan(**changes)

    def test_plan_immutable_json_roundtrip_and_no_effect_claim(self):
        plan = self.plan()
        with self.assertRaises(FrozenInstanceError):
            plan.layer_index = 99
        encoded = plan.as_dict()
        self.assertEqual(json.loads(json.dumps(encoded, allow_nan=False)), encoded)
        self.assertFalse(encoded["accuracy_or_speedup_proven"])
        self.assertFalse(encoded["runtime_adapter_installed"])
        self.assertFalse(encoded["score_policy_selected"])

    def test_import_and_geometry_do_not_import_gpu_runtime(self):
        command = (
            "import sys; from kv_cache_engram.ffn_only_masks import plan_layer_masks; "
            "plan_layer_masks(input_tokens=10,layer_index=6,document_spans=[(0,10)],sensitive_indices=[]); "
            "assert 'torch' not in sys.modules; "
            "assert not any(n == 'deepinfer' or n.startswith('deepinfer.') for n in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", command], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
