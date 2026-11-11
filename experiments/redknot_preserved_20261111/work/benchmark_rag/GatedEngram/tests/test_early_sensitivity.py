"""CPU-only selector contract tests. These do not demonstrate model accuracy."""
import inspect
import math
import unittest

from kv_cache_engram.core import ValidationError
from kv_cache_engram.early_sensitivity import (
    EARLY_LAYERS, METRIC, REPRESENTATION, EarlySensitivityConfig, LayerRoles,
    aggregate_layer_scores, eligible_sparse_layers, plan_chunk, plan_from_scores,
    reference_layer_scores, score_layer_pairs,
)


def config(**changes):
    values = dict(metric=METRIC, optional_fractions=(.2, .5, 1.), representation=REPRESENTATION)
    values.update(changes)
    return EarlySensitivityConfig(**values)


class ScoreTests(unittest.TestCase):
    def test_exact_same_layer_relative_l2_and_online_denominator(self):
        self.assertEqual(reference_layer_scores([[3., 4.], [2., 0.]],
                         [[0., 0.], [1., 0.]], config=config()), (1., .5))

    def test_multidimensional_rows_flatten_all_features(self):
        values = reference_layer_scores([[[3., 4.], [0., 0.]]],
                    [[[0., 0.], [0., 0.]]], config=config())
        self.assertEqual(values, (1.,))

    def test_zero_norm_uses_declared_floor(self):
        self.assertEqual(reference_layer_scores([[0.], [0.]], [[0.], [1e-8]],
                         config=config()), (0., 1.))

    def test_six_layer_mean_and_pair_reference_agree(self):
        pairs = {layer: ([[2., 0.]], [[2.-layer, 0.]]) for layer in EARLY_LAYERS}
        self.assertEqual(score_layer_pairs(pairs, config=config()), (1.25,))
        self.assertEqual(aggregate_layer_scores({layer: [layer/2] for layer in EARLY_LAYERS},
                         config=config()), (1.25,))

    def test_missing_extra_or_bool_layer_fails_closed(self):
        for layers in ({i: [1.] for i in range(5)}, {i: [1.] for i in range(7)},
                       {False: [1.], 1: [1.], 2: [1.], 3: [1.], 4: [1.], 5: [1.]}):
            with self.subTest(layers=layers), self.assertRaises(ValidationError):
                aggregate_layer_scores(layers, config=config())

    def test_score_nonfinite_negative_shape_and_overflow_fail(self):
        for score in (math.nan, math.inf, -1., True):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                aggregate_layer_scores({i: [score] for i in EARLY_LAYERS}, config=config())
        for online, offline in (([], []), ([[1.]], [[1.], [2.]]),
                ([[1., 2.]], [[1.]]), ([[1.], [1., 2.]], [[1.], [1., 2.]]),
                ([[[1.], [1., 2.]]], [[[1.], [1., 2.]]]),
                ([[math.nan]], [[0.]]), ([[1e308]], [[-1e308]]),
                ([[1e308]*4], [[1e308]*4]), ([1.], [1.])):
            with self.subTest(online=online), self.assertRaises(ValidationError):
                reference_layer_scores(online, offline, config=config())

    def test_explicit_config_required_and_invalid_protocol_rejected(self):
        with self.assertRaises(TypeError):
            EarlySensitivityConfig()
        for changes in (dict(metric="adjacent-layer"), dict(representation="postrope"),
                dict(optional_fractions=(.5, .2, 1.)), dict(optional_fractions=(.2,)),
                dict(optional_fractions=(.2, .2, 1.)), dict(optional_fractions=(math.nan, 1.)),
                dict(optional_fractions=(True,)), dict(early_layers=(0, 1)),
                dict(norm_floor=0), dict(guard_tokens=127), dict(native_chunk_tokens=4096),
                dict(document_tokens=16384)):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                config(**changes)


class PlanTests(unittest.TestCase):
    def plan(self, scores=None, **changes):
        values = dict(start=0, end=8192, input_tokens=32768,
                      document_spans=[(0, 32768)], config=config())
        values.update(changes)
        return plan_from_scores(scores if scores is not None else [1.]*8192, **values)

    def test_ties_rank_by_absolute_position_nested_and_full_identity(self):
        result = self.plan()
        self.assertEqual(result["optional_ranked_positions"], list(range(128, 8064)))
        plans = result["selections"]
        self.assertEqual([p["optional_selected_rows"] for p in plans], [1588, 3968, 7936])
        self.assertTrue(set(plans[0]["retained_positions"]) <= set(plans[1]["retained_positions"]))
        self.assertEqual(plans[2]["retained_positions"], list(range(8192)))
        self.assertEqual(plans[2]["bypass_positions"], [])
        self.assertGreater(plans[0]["actual_retained_fraction"], .2)

    def test_high_scores_retained_not_bypassed(self):
        scores = [0.]*8192
        scores[8000] = 100.
        result = self.plan(scores)
        self.assertEqual(result["optional_ranked_positions"][0], 8000)
        self.assertIn(8000, result["selections"][0]["retained_positions"])

    def test_document_nativechunk_template_guards_union(self):
        result = self.plan(start=32768, end=40960, input_tokens=65539,
                           document_spans=[(3, 32771), (32771, 65539)])
        protected = set(result["mandatory_positions"])
        self.assertTrue(set(range(32768, 32899)) <= protected)
        self.assertTrue(set(range(40832, 40960)) <= protected)
        self.assertNotIn(32899, protected)
        first = self.plan(input_tokens=32771, document_spans=[(3, 32771)])
        self.assertTrue(set(range(131)) <= set(first["mandatory_positions"]))

    def test_last_document_guard_and_query_tail_compulsory(self):
        result = self.plan(scores=[0.]*8, start=32768, end=32776, input_tokens=32776)
        self.assertEqual(result["mandatory_positions"], list(range(32768, 32776)))
        self.assertEqual(result["optional_positions"], [])
        self.assertTrue(all(p["retained_rows"] == 8 for p in result["selections"]))

    def test_explicit_eligibility_and_mandatory_are_additive(self):
        result = self.plan(eligible_positions=[1000, 2000, 3000], mandatory_positions=[2000])
        self.assertEqual(result["optional_positions"], [1000, 3000])
        self.assertIn(2000, result["mandatory_positions"])
        self.assertEqual(result["selections"][0]["optional_selected_positions"], [1000])

    def test_plan_from_six_layer_scores_matches_public_reference(self):
        values = dict(start=0, end=8192, input_tokens=32768,
                      document_spans=[(0, 32768)], config=config())
        result = plan_chunk({i: [1.]*8192 for i in EARLY_LAYERS}, **values)
        self.assertEqual(result, self.plan())

    def test_invalid_shapes_missing_scores_and_boundaries_fail_closed(self):
        for changes in (dict(start=1), dict(end=8000), dict(input_tokens=True),
                dict(document_spans=[]), dict(document_spans=[(0, 32767)]),
                dict(document_spans=[(0, 32768), (0, 32768)]),
                dict(eligible_positions=[128, 128]), dict(eligible_positions=[8192]),
                dict(mandatory_positions=[True]), dict(mandatory_positions=[-1])):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.plan(**changes)
        for scores in ([1.]*8191, [math.nan]*8192, [-1.]*8192):
            with self.subTest(score=scores[0]), self.assertRaises(ValidationError):
                self.plan(scores)

    def test_selection_does_not_grant_accuracy_or_cache_admission(self):
        result = self.plan()
        self.assertFalse(result["accuracy_admission"])
        self.assertFalse(result["cache_reuse_permission_granted"])
        self.assertTrue(result["global_index_and_cache_state_updates_must_be_preserved"])


class LayerRoleTests(unittest.TestCase):
    def roles(self):
        return [LayerRoles(i, i in (2, 8, 14), False, False, i >= 19) for i in range(40)]

    def test_all_early_and_all_source_roles_excluded(self):
        roles = self.roles()
        self.assertEqual(eligible_sparse_layers(roles, layer_count=40),
                         (6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18))
        roles[6] = LayerRoles(6, False, True, False, False)
        roles[7] = LayerRoles(7, False, False, True, False)
        self.assertNotIn(6, eligible_sparse_layers(roles, layer_count=40))
        self.assertNotIn(7, eligible_sparse_layers(roles, layer_count=40))

    def test_unknown_missing_duplicate_geometry_fails_closed(self):
        with self.assertRaises(ValidationError):
            LayerRoles(6, None, False, False, False)
        with self.assertRaises(TypeError):
            LayerRoles(6)
        for roles in (self.roles()[:-1], self.roles()[:-1]+[self.roles()[0]], [None]*40):
            with self.subTest(roles=roles), self.assertRaises(ValidationError):
                eligible_sparse_layers(roles, layer_count=40)

    def test_module_has_no_torch_or_device_imports(self):
        import kv_cache_engram.early_sensitivity as module
        source = inspect.getsource(module)
        self.assertNotIn("import torch", source)
        self.assertNotIn("cuda.", source)


if __name__ == "__main__":
    unittest.main()
