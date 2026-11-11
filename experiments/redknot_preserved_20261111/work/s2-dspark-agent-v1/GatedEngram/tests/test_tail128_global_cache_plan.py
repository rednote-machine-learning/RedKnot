"""CPU geometry/provenance tests; these do not qualify GPU numerical hooks."""
import copy
import math
import random
import unittest

from kv_cache_engram.tail128_global_cache_plan import (
    canonical_bank_layout, gated_compress_reference, source_update_plan,
    topology_for_layer,
)


def doc(start, end, key="a"):
    return dict(start=start, end=end, key=key * 64)


class GlobalCacheTopologyTests(unittest.TestCase):
    def test_index_query_refresh_does_not_create_new_key_source(self):
        for layer in (24, 28, 32, 36):
            t = topology_for_layer(layer)
            self.assertEqual(t["main_kv_source_layer"], 20)
            self.assertEqual(t["index_k_source_layer"], 20)
            self.assertTrue(t["recomputes_index_queries"])
            self.assertFalse(t["writes_main_kv"])
            self.assertFalse(t["writes_index_k"])

    def test_native_source2_is_retained_through_layer7(self):
        self.assertEqual(topology_for_layer(1)["compression_ratio"], 0)
        for layer in range(2, 8):
            self.assertTrue(topology_for_layer(layer)["main_cache_from_native_prepass"])
        for layer in (2, 3, 7, 9, 24):
            with self.assertRaises(ValueError):
                source_update_plan([doc(0, 4)], 4, [3], layer=layer, phase_generation=1)

    def test_layout_requires_raw_gates_and_source_weights_binding(self):
        layout = canonical_bank_layout([doc(1, 4), doc(5, 8)], 9, scope_sha256="f" * 64)
        self.assertEqual(layout["entry_count"], 3)  # equal documents share immutable raw bank
        self.assertEqual(layout["raw_bank_bytes"], 3 * (2 * 4096 + 1024))
        self.assertEqual(layout["entries"][0]["shape"], [3, 1024])
        self.assertEqual(layout["entries"][-1]["shape"], [3, 512])
        self.assertFalse(layout["source2_offline_override_allowed"])
        with self.assertRaises(ValueError):
            canonical_bank_layout([doc(0, 3), doc(4, 8)], 8, scope_sha256="f" * 64)
        with self.assertRaises(ValueError):
            canonical_bank_layout([doc(0, 3)], 3, scope_sha256="unbound")


class GlobalCacheGroupPlanTests(unittest.TestCase):
    def plan(self, docs, total, selected, **kwargs):
        return source_update_plan(docs, total, selected, layer=kwargs.pop("layer", 8),
                                  phase_generation=kwargs.pop("phase_generation", 1), **kwargs)

    def test_odd_document_offset_and_nondoc_pair_mix(self):
        p = self.plan([doc(1, 6)], 7, [0, 4, 5, 6])
        self.assertEqual(p["new_complete_compressed_ids"], [0, 1, 2])
        self.assertEqual(p["affected_compressed_ids"], [0, 2])
        b = p["blocks"][0]
        self.assertEqual(b["member_positions"], [0, 1])
        self.assertEqual([m["origin"] for m in b["members"]],
                         ["current_online_projection", "immutable_offline_projection"])
        self.assertEqual(b["members"][1]["document_local_row"], 0)
        self.assertEqual(p["pending_incomplete_members"][0]["position"], 6)

    def test_adjacent_document_boundary_does_not_reset_compression(self):
        p = self.plan([doc(0, 3), doc(3, 8, "b")], 8, [6, 7])
        b = p["blocks"][1]
        self.assertEqual(b["member_positions"], [2, 3])
        self.assertEqual([m["document_occurrence"] for m in b["members"]], [0, 1])
        self.assertEqual(b["action"], "reuse_phase_offline_base")
        self.assertEqual(b["rope_position"], 2)
        self.assertEqual(b["causal_visible_from_query"], 3)

    def test_selected_c2_carry_must_be_same_source_phase_projection(self):
        first = self.plan([doc(0, 8)], 8, [2, 3, 7], chunk_start=0, chunk_end=3)
        self.assertEqual(first["pending_incomplete_members"][0]["position"], 2)
        with self.assertRaisesRegex(ValueError, "carry is absent"):
            self.plan([doc(0, 8)], 8, [2, 3, 7], chunk_start=3, chunk_end=8)
        second = self.plan([doc(0, 8)], 8, [2, 3, 7], chunk_start=3, chunk_end=8,
                           previous_projection_positions=[2], phase_generation=9,
                           previous_projection_scope=dict(source_layer=8, phase_generation=9))
        b = second["blocks"][0]
        self.assertEqual(b["member_positions"], [2, 3])
        self.assertEqual(b["members"][0]["origin"], "same_phase_source_previous_projection")
        self.assertEqual(b["members"][0]["phase_generation"], 9)
        self.assertEqual(b["members"][0]["source_layer"], 8)

    def test_carry_from_another_source_or_phase_is_rejected(self):
        for scope in (None, dict(source_layer=14, phase_generation=1),
                      dict(source_layer=8, phase_generation=2)):
            with self.assertRaisesRegex(ValueError, "scope"):
                self.plan([doc(0, 8)], 8, [2, 3], chunk_start=3,
                          previous_projection_positions=[2], previous_projection_scope=scope)

    def test_current_projection_cannot_omit_selected_or_restore_unselected(self):
        for actual in ([4], [1, 4, 5]):
            with self.assertRaises(ValueError):
                self.plan([doc(0, 6)], 6, [4, 5], current_projection_positions=actual)

    def test_no_offline_nondocument_fallback(self):
        with self.assertRaisesRegex(ValueError, "nondocument"):
            self.plan([doc(1, 6)], 7, [4, 5])

    def test_c1_source20_has_per_token_groups_and_no_c2_carry(self):
        p = self.plan([doc(0, 7)], 7, [1, 6], layer=20)
        self.assertEqual(p["affected_compressed_ids"], [1, 6])
        self.assertEqual(p["visible_compressed_rows_after_chunk"], 7)
        self.assertEqual(p["pending_incomplete_members"], [])
        self.assertEqual(p["topology"]["main_hidden_source"], "source20_boundary_attn_normalized_bf16")

    def test_source_and_phase_are_in_seal_and_inputs_are_immutable(self):
        docs, selected = [doc(0, 8)], [1, 7]
        before = copy.deepcopy((docs, selected))
        a = self.plan(docs, 8, selected)
        b = self.plan(docs, 8, selected, layer=14)
        c = self.plan(docs, 8, selected, phase_generation=2)
        self.assertEqual(len({a["sha256"], b["sha256"], c["sha256"]}), 3)
        self.assertEqual((docs, selected), before)

    def test_random_chunk_boundaries_match_absolute_complete_groups(self):
        rng = random.Random(431)
        for _ in range(120):
            n = rng.randrange(4, 70)
            split = rng.randrange(1, n)
            selected = sorted(rng.sample(range(n), rng.randrange(n + 1)))
            for layer, ratio in ((8, 2), (20, 1)):
                a = self.plan([doc(0, n)], n, selected, layer=layer, chunk_end=split)
                b = self.plan([doc(0, n)], n, selected, layer=layer, chunk_start=split,
                              previous_projection_positions=[p for p in selected if p < split],
                              previous_projection_scope=dict(source_layer=layer, phase_generation=1))
                blocks = a["blocks"] + b["blocks"]
                self.assertEqual([x["compressed_id"] for x in blocks], list(range(n // ratio)))
                self.assertEqual([x["compressed_id"] for x in blocks if x["affected"]],
                                 [j for j in range(n // ratio)
                                  if any(p in selected for p in range(j * ratio, (j + 1) * ratio))])


class GlobalCacheCompressionReferenceTests(unittest.TestCase):
    def test_featurewise_gate_is_not_uniform_token_average(self):
        values = [dict(gate=[0, math.log(3)], latent=[2, 10]),
                  dict(gate=[math.log(3), 0], latent=[6, 30])]
        for actual, expected in zip(gated_compress_reference(values, ratio=2), [5, 15]):
            self.assertAlmostEqual(actual, expected, places=13)

    def test_mixed_online_offline_members_are_combined_before_compression(self):
        offline = dict(gate=[0], latent=[2])
        current = dict(gate=[math.log(3)], latent=[10])
        before = copy.deepcopy((offline, current))
        self.assertEqual(gated_compress_reference([offline, current], ratio=2), [8])
        self.assertEqual((offline, current), before)

    def test_extreme_gates_are_stable_and_c1_has_no_gate(self):
        self.assertEqual(gated_compress_reference([
            dict(gate=[10000], latent=[3]), dict(gate=[-10000], latent=[9])], ratio=2), [3])
        self.assertEqual(gated_compress_reference([dict(latent=[3, 4])], ratio=1), [3, 4])
        with self.assertRaises(ValueError):
            gated_compress_reference([dict(gate=[0], latent=[3])], ratio=1)

    def test_incomplete_and_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            gated_compress_reference([dict(gate=[0], latent=[3])], ratio=2)
        with self.assertRaises(ValueError):
            gated_compress_reference([dict(latent=[math.nan])], ratio=1)
        with self.assertRaises(ValueError):
            topology_for_layer(True)


if __name__ == "__main__":
    unittest.main()
