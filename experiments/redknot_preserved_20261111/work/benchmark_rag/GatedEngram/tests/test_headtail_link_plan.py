import copy
import unittest

from kv_cache_engram.headtail_link_plan import (
    plan, native_positions, query_key_positions, materialize_indices, rebind_layer,
)


class HeadTailPlanTests(unittest.TestCase):
    def fixture(self):
        return plan([dict(start=2, end=12), dict(start=12, end=22)], list(range(24)),
                    0, 24, 24, layer=3, head_tokens=2, tail_tokens=2, window_size=4)

    def test_head_tail_nondocument_union_has_no_duplicates(self):
        e = self.fixture()
        self.assertEqual(e['selected_positions'], [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])
        self.assertEqual(e['inactive_local_rows'], [4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19])

    def test_completed_document_rule_never_adds_own_future_tail(self):
        e = self.fixture()
        self.assertEqual(query_key_positions(e, 3), [0, 1, 2, 3])
        self.assertEqual(e['completed_document_counts'][11], 0)
        self.assertEqual(e['completed_document_counts'][12], 1)
        for q in e['current_positions']:
            self.assertTrue(all(k <= q for k in query_key_positions(e, q)))

    def test_native_window_is_preserved_and_overlap_is_deduplicated(self):
        e = self.fixture()
        self.assertEqual(query_key_positions(e, 12), [9, 10, 11, 12])
        self.assertEqual(query_key_positions(e, 14), [11, 12, 13, 14, 10])
        self.assertEqual(query_key_positions(e, 22), [19, 20, 21, 22, 10, 11])
        for q in e['current_positions']:
            keys = query_key_positions(e, q)
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(keys), e['union_lengths'][q])

    def test_index_mapping_preserves_native_prefix_and_offsets_global(self):
        e = self.fixture()
        matrix = materialize_indices(e, [14])
        self.assertEqual(matrix[0][:4], [15, 16, 17, 18])
        self.assertEqual(matrix[0][4:8], [28, -1, -1, -1])
        self.assertEqual(e['global_index_offset'], 32)
        self.assertTrue(all(i < e['global_index_offset'] for i in matrix[0] if i >= 0))

    def test_previous_chunk_native_keys_map_to_ring_and_extra_overlap_dedups(self):
        docs = [dict(start=0, end=16), dict(start=16, end=32)]
        e = plan(docs, list(range(16, 24)), 16, 24, 34, layer=4,
                 head_tokens=2, tail_tokens=2, window_size=4)
        self.assertEqual(materialize_indices(e, [16])[0][:4], [1, 2, 3, 4])
        self.assertEqual(query_key_positions(e, 16), [13, 14, 15, 16])
        self.assertEqual(query_key_positions(e, 20), [17, 18, 19, 20, 14, 15])

    def test_yoco_placeholder_has_empty_native_window_but_prior_tail_links(self):
        docs = [dict(start=0, end=400), dict(start=400, end=800)]
        positions = native_positions(400, 500, 1000, 20, 10)
        e = plan(docs, positions, 400, 500, 1000, layer=20, yoco_keep_tokens=10)
        self.assertEqual(positions, [499])
        self.assertEqual(e['native_window_lengths'], [0])
        self.assertEqual(query_key_positions(e, 499), list(range(272, 400)))
        self.assertEqual(materialize_indices(e)[0][:128], [-1] * 128)
        self.assertTrue(e['extra_links_cross_yoco_rectangle'])

    def test_deleted_yoco_queries_are_not_revived(self):
        docs = [dict(start=0, end=400), dict(start=400, end=800)]
        e = plan(docs, [599], 500, 600, 1000, layer=39, yoco_keep_tokens=10)
        self.assertEqual(e['selected_local_rows'], [])
        self.assertEqual(e['full_rows'], 1)
        with self.assertRaises(ValueError):
            plan(docs, list(range(500, 600)), 500, 600, 1000, layer=39, yoco_keep_tokens=10)

    def test_final_yoco_local_rectangle_and_cross_rectangle_extras(self):
        e = plan([dict(start=0, end=400), dict(start=400, end=800)], list(range(990, 1000)),
                 984, 1000, 1000, layer=24, yoco_keep_tokens=10)
        self.assertEqual(query_key_positions(e, 990)[:1], [990])
        self.assertEqual(e['native_window_lengths'], list(range(1, 11)))
        self.assertEqual(e['extra_tail_lengths'], [256] * 10)
        self.assertEqual(len(query_key_positions(e, 999)), 266)

    def test_final_question_can_link_all_eight_tails_width_is_1152(self):
        docs = [dict(start=i * 32768, end=(i + 1) * 32768) for i in range(8)]
        e = plan(docs, [262144], 262144, 262145, 262145, layer=3)
        self.assertEqual(e['completed_document_counts'], [8])
        self.assertEqual(e['index_width'], 1152)
        self.assertEqual(e['extra_tail_lengths'], [897])  # 127 old-tail keys already local

    def test_short_document_head_tail_overlap_is_selected_once(self):
        e = plan([dict(start=0, end=3)], list(range(4)), 0, 4, 4,
                 layer=3, head_tokens=2, tail_tokens=2, window_size=4)
        self.assertEqual(e['selected_local_rows'], [0, 1, 2, 3])

    def test_layer_rebinding_shares_layout_but_not_layer_seal(self):
        e = self.fixture()
        other = rebind_layer(e, 19)
        self.assertIs(other['current_positions'], e['current_positions'])
        self.assertEqual(other['layout_sha256'], e['layout_sha256'])
        self.assertNotEqual(other['plan_sha256'], e['plan_sha256'])
        with self.assertRaises(ValueError): rebind_layer(e, 20)

    def test_geometry_and_valid_start_are_fail_closed(self):
        docs = [dict(start=0, end=10)]
        for positions in ([0, 2], [0, 1, 1], [False, 1]):
            with self.assertRaises(ValueError): plan(docs, positions, 0, 2, 10, layer=3)
        with self.assertRaises(ValueError):
            plan(docs, [0, 1], 0, 2, 10, layer=3, native_valid_start=1)

    def test_real_producer_matches_independent_reader_schema_and_hashes(self):
        from gpu_pilot.read_headtail_link_results import expected_plan
        docs = [dict(start=5+i*32768,end=5+(i+1)*32768,key=str(i)*64) for i in range(8)]
        spans = [(d['start'],d['end'],d['key']) for d in docs]
        total = docs[-1]['end']+7
        for layer,start,end in ((3,0,8192),(19,32768,40960),(20,0,8192),
                                (24,32768,40960),(39,262144,total)):
            positions = native_positions(start,end,total,layer)
            emitted = plan(docs,positions,start,end,total,layer=layer)
            self.assertEqual(emitted,expected_plan(positions,start,end,total,layer,spans))


if __name__ == '__main__':
    unittest.main()
