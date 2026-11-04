"""CPU-only adjacent score/rank references, not model quality measurements."""

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

from kv_cache_engram.adjacent_kv_selector import (
    FEATURE_KIND,
    LAYER_INDICES,
    REPRESENTATION,
    SWAKVFeatures,
    adjacent_relative_scores,
    plan_adjacent_chunk,
    select_chunk_by_scores,
)
from kv_cache_engram.core import ValidationError


def features(values=None, positions=(0, 1), **changes):
    values = values or {2: [[1.0], [2.0]], 3: [[2.0], [2.0]], 4: [[4.0], [4.0]], 5: [[8.0], [4.0]]}
    result = {}
    for layer in LAYER_INDICES:
        fields = dict(
            layer_index=layer,
            input_ids_sha256="a" * 64,
            positions=positions,
            rows=values[layer],
            feature_kind=FEATURE_KIND,
            representation=REPRESENTATION,
            source_layer_index=layer,
        )
        fields.update(changes)
        result[layer] = SWAKVFeatures(**fields)
    return result


class AdjacentScoreTests(unittest.TestCase):
    def test_import_and_scoring_remain_cpu_standard_library_only(self):
        code = (
            "import sys; from kv_cache_engram.adjacent_kv_selector import select_chunk_by_scores; "
            "select_chunk_by_scores([1.],input_tokens=1,chunk_start=0,chunk_end=1,document_spans=[(0,1)]); "
            "assert 'torch' not in sys.modules; assert not any(n == 'deepinfer' or n.startswith('deepinfer.') for n in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_previous_norm_and_per_token_mean_of_three_transitions(self):
        scored = adjacent_relative_scores(features())
        self.assertEqual(scored.transition_scores, ((1.0, 0.0), (1.0, 1.0), (1.0, 0.0)))
        self.assertEqual(scored.scores, (1.0, 1 / 3))
        self.assertEqual(scored.as_dict()["human_layer_numbers"], [3, 4, 5, 6])
        self.assertFalse(scored.as_dict()["cohort_aggregation"])

    def test_vector_l2_not_elementwise_relative_mean(self):
        values = {2: [[3.0, 4.0]], 3: [[0.0, 0.0]], 4: [[0.0, 0.0]], 5: [[0.0, 0.0]]}
        scored = adjacent_relative_scores(features(values, positions=(10,)))
        self.assertEqual(scored.transition_scores, ((1.0,), (0.0,), (0.0,)))
        self.assertEqual(scored.scores, (1 / 3,))

    def test_zero_previous_norm_uses_exact_floor(self):
        values = {2: [[0.0]], 3: [[1e-8]], 4: [[0.0]], 5: [[0.0]]}
        scored = adjacent_relative_scores(features(values, positions=(0,)))
        self.assertEqual(scored.transition_scores, ((1.0,), (1.0,), (0.0,)))
        self.assertEqual(scored.scores, (2 / 3,))

    def test_inputs_are_copied_not_mutably_aliased(self):
        rows, positions = [[1.0], [2.0]], [0, 1]
        value = SWAKVFeatures(2, "a" * 64, positions, rows, FEATURE_KIND, REPRESENTATION, 2)
        rows[0][0], positions[0] = 999.0, 999
        self.assertEqual(value.rows, ((1.0,), (2.0,)))
        self.assertEqual(value.positions, (0, 1))

    def test_missing_extra_bool_and_mislabelled_layers_rejected(self):
        original = features()
        for bad in (
            {k: v for k, v in original.items() if k != 5},
            dict(original, extra=original[2]),
            {True: original[2], 3: original[3], 4: original[4], 5: original[5]},
            dict(original, **{}),
        ):
            if bad == original:
                bad[2] = original[3]
            with self.subTest(keys=list(bad)), self.assertRaises(ValidationError):
                adjacent_relative_scores(bad)

    def test_shared_global_or_postrope_or_wrong_producer_rejected(self):
        for changes in (
            dict(feature_kind="global_kv"),
            dict(feature_kind="shared_global_kv"),
            dict(representation="swa-postrope"),
            dict(source_layer_index=2),
            dict(source_layer_index=True),
            dict(layer_index=6),
            dict(layer_index=True),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                features(**changes)

    def test_same_input_exact_positions_and_width_required(self):
        for changes in (dict(input_ids_sha256="b" * 64), dict(positions=(1, 2)), dict(rows=[[1.0, 2.0], [3.0, 4.0]])):
            values = features()
            fields = dict(vars(values[5]))
            fields.update(changes)
            values[5] = SWAKVFeatures(**fields)
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                adjacent_relative_scores(values)

    def test_invalid_position_and_component_values_rejected(self):
        for positions in ((0, 0), (1, 0), (-1, 0), (True, 1), (0.0, 1), (math.nan, 1), ()):
            with self.subTest(positions=positions), self.assertRaises(ValidationError):
                features(positions=positions)
        for value in (math.nan, math.inf, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                features({layer: [[value], [1.0]] for layer in LAYER_INDICES})

    def test_overflow_in_derived_norm_or_change_rejected(self):
        values = {2: [[-1e308]], 3: [[1e308]], 4: [[1.0]], 5: [[1.0]]}
        with self.assertRaises(ValidationError):
            adjacent_relative_scores(features(values, positions=(0,)))

    def test_score_evidence_json_roundtrip_without_effect_claim(self):
        value = adjacent_relative_scores(features()).as_dict()
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
        self.assertFalse(value["accuracy_admission"])


class ChunkSelectorTests(unittest.TestCase):
    def select(self, scores, spans=None, **changes):
        args = dict(
            input_tokens=len(scores), chunk_start=0, chunk_end=len(scores), document_spans=[(0, len(scores))] if spans is None else spans
        )
        args.update(changes)
        return select_chunk_by_scores(scores, **args)

    def test_exact_ceil15pct_not_of_whole_chunk_with_non_document_mandatory(self):
        plan = self.select(list(range(20)), spans=[(2, 12)])
        self.assertEqual(plan["document_eligible_count"], 10)
        self.assertEqual(plan["sensitive_count"], 2)
        self.assertEqual(plan["sensitive_indices"], [10, 11])
        self.assertEqual(plan["mandatory_ffn_indices"], [0, 1] + list(range(12, 20)))
        self.assertEqual(plan["ffn_selected_rows"], [0, 1] + list(range(10, 20)))

    def test_ties_ascending_absolute_position_within_second_native_chunk(self):
        plan = self.select([1.0] * 20, input_tokens=8212, chunk_start=8192, chunk_end=8212, spans=[(8192, 8212)])
        self.assertEqual(plan["sensitive_indices"], [8192, 8193, 8194])
        self.assertEqual(plan["ranked_document_positions"], list(range(8192, 8212)))

    def test_each_native_chunk_ranks_independently(self):
        a = self.select([1.0] * 8192, input_tokens=16384, chunk_start=0, chunk_end=8192, spans=[(0, 16384)])
        b = self.select([0.0] * 8192, input_tokens=16384, chunk_start=8192, chunk_end=16384, spans=[(0, 16384)])
        self.assertEqual(a["sensitive_count"], 1229)
        self.assertEqual(b["sensitive_count"], 1229)
        self.assertEqual(b["sensitive_indices"][0], 8192)

    def test_prefix127_not_added_to_ffn_mask_and_no_tail_guard(self):
        plan = self.select(list(range(1000)))
        self.assertEqual(plan["sensitive_count"], 150)
        self.assertEqual(plan["sensitive_indices"], list(range(850, 1000)))
        self.assertFalse(set(range(127)) & set(plan["ffn_selected_rows"]))

    def test_empty_document_set_all_non_document_ffn_mandatory(self):
        plan = self.select([1.0] * 10, spans=[])
        self.assertEqual(plan["sensitive_count"], 0)
        self.assertEqual(plan["ffn_selected_rows"], list(range(10)))

    def test_doc_fragments_and_gaps_use_actual_spans(self):
        plan = self.select(list(range(20)), spans=[(0, 8195), (8200, 8212)], input_tokens=8212, chunk_start=8192, chunk_end=8212)
        self.assertEqual(plan["document_eligible_count"], 15)
        self.assertEqual(plan["sensitive_count"], 3)
        self.assertEqual(plan["mandatory_ffn_indices"], list(range(8195, 8200)))

    def test_invalid_native_chunk_geometry_rejected(self):
        for changes in (
            dict(input_tokens=True),
            dict(chunk_start=True),
            dict(chunk_end=2.0),
            dict(chunk_start=1),
            dict(input_tokens=100, chunk_end=20),
            dict(input_tokens=9000, chunk_end=9000),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.select([1.0] * 20, **changes)

    def test_bad_scores_spans_and_fraction_substitutes_rejected(self):
        for value in (math.nan, math.inf, -1.0, True, "1", 10**10000):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.select([value] * 20)
        for spans in ([(1, 0)], [(0, 21)], [(0, 10), (9, 20)], [(10, 20), (0, 5)], [(False, 20)], [(0, math.inf)]):
            with self.subTest(spans=spans), self.assertRaises(ValidationError):
                self.select([1.0] * 20, spans=spans)
        with self.assertRaises(ValidationError):
            self.select([1.0] * 20, chunk_end=19, input_tokens=19)

    def test_combined_plan_checks_exact_chunk_positions(self):
        plan = plan_adjacent_chunk(features(), input_tokens=2, chunk_start=0, chunk_end=2, document_spans=[(0, 2)])
        self.assertEqual(plan["sensitive_indices"], [0])
        self.assertEqual(plan["feature_provenance"]["scores"], [1.0, 1 / 3])
        with self.assertRaises(ValidationError):
            plan_adjacent_chunk(features(positions=(2, 3)), input_tokens=2, chunk_start=0, chunk_end=2, document_spans=[(0, 2)])


if __name__ == "__main__":
    unittest.main()
