"""Explicit ten-block geometry; no Torch/GPU imports or numerical policy changes."""
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from kv_cache_engram.core import token_digest
from kv_cache_engram.native_reuse import document_plan
from kv_cache_engram.s1_runtime import S1Controller, s1_documents
from kv_cache_engram.s1_plan import all_chunks
from kv_cache_engram.s1_attention import local_memory_plan


def inputs(count=10):
    spans = []
    cursor = 13
    for i in range(count):
        length = (4097 if i == 2 else 256 + 71 * i)
        spans.append((cursor, cursor + length))
        cursor += length + i % 3
    ids = list(range(cursor + 21))
    docs = [dict(input_start=a, input_end=b, document_id=token_digest(ids[a:b]),
                 token_ids_sha256=token_digest(ids[a:b]), offline_position=0) for a, b in spans]
    return ids, docs


def controller(count=8):
    layers = [NS(layer_idx=i) for i in range(40)]
    model = NS(layers=layers, yoco_boundary_layer_idx=20)
    value = S1Controller(layers, native_model=model, model_sha256='a'*64,
        execution_sha256='b'*64, tokenizer_sha256='c'*64, expected_document_count=count)
    value.state = 'IDLE'
    value.original_boundary = 20
    value.cache = NS(bank={}, prepare_case=Mock(), begin_phase=Mock())
    value.attention = NS(begin_phase=Mock())
    return value


class MuSiQueS1GeometryTests(unittest.TestCase):
    def test_explicit_ten_preserves_exact_variable_spans_and_gaps(self):
        ids, docs = inputs()
        parsed = document_plan(ids, docs, natural_boundary_inputs=True, expected_document_count=10)
        self.assertEqual(len(parsed), 10)
        self.assertEqual([(d['start'], d['end']) for d in parsed],
                         [(d['input_start'], d['input_end']) for d in docs])
        self.assertEqual(s1_documents(ids, docs, expected_document_count=10), parsed)
        self.assertTrue(any(a['end'] < b['start'] for a, b in zip(parsed, parsed[1:])))

    def test_legacy_natural_default_stays_eight_not_inferred_from_input(self):
        ids8, docs8 = inputs(8)
        self.assertEqual(len(document_plan(ids8, docs8, natural_boundary_inputs=True)), 8)
        self.assertEqual(len(s1_documents(ids8, docs8)), 8)
        ids10, docs10 = inputs(10)
        with self.assertRaises(ValueError):
            document_plan(ids10, docs10, natural_boundary_inputs=True)
        with self.assertRaises(ValueError):
            s1_documents(ids10, docs10)
        with self.assertRaises(ValueError):
            s1_documents(ids8, docs8, expected_document_count=10)

    def test_legacy_fixed32k_cannot_opt_into_ten(self):
        ids, docs = inputs()
        with self.assertRaisesRegex(ValueError, 'legacy count remains eight'):
            document_plan(ids, docs, expected_document_count=10)

    def test_count_is_explicit_integer_not_bool_or_document_length(self):
        ids, docs = inputs()
        for count in (True, False, 0, -1, 10.0, '10', None):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    document_plan(ids, docs, natural_boundary_inputs=True, expected_document_count=count)
                with self.assertRaises(ValueError):
                    s1_documents(ids, docs, expected_document_count=count)
                with self.assertRaises(ValueError):
                    controller(count)

    def test_ten_does_not_relax_hash_order_or_overlap_checks(self):
        ids, docs = inputs()
        for change in ('hash', 'order', 'overlap'):
            bad = copy.deepcopy(docs)
            if change == 'hash':
                bad[4]['token_ids_sha256'] = '0'*64
            elif change == 'order':
                bad[1], bad[2] = bad[2], bad[1]
            else:
                bad[2]['input_start'] = bad[1]['input_end'] - 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                document_plan(ids, bad, natural_boundary_inputs=True, expected_document_count=10)

    def test_independent_offline_capture_remains_one_untemplated_block(self):
        ids = list(range(317))
        self.assertEqual(document_plan(ids, None, independent=True, natural_boundary_inputs=True,
            expected_document_count=10), (dict(start=0, end=317, key=token_digest(ids)),))

    def test_single_full16k_qualification_survives_explicit_ten_contract(self):
        ids = list(range(16384))
        doc = dict(input_start=0, input_end=len(ids), token_ids_sha256=token_digest(ids))
        for count in (8, 10):
            self.assertEqual(s1_documents(ids, [doc], expected_document_count=count),
                             (dict(start=0, end=len(ids), key=token_digest(ids)),))
        for n in (16383, 16385):
            other = list(range(n))
            d = dict(input_start=0, input_end=n, token_ids_sha256=token_digest(other))
            with self.assertRaises(ValueError):
                s1_documents(other, [d], expected_document_count=10)

    def test_controller_prepare_and_native_reset_bind_declared_ten(self):
        owner = controller(10)
        ids, docs = inputs()
        owner.prepare_case(input_ids=ids, documents=docs, index_device='cpu-fixture')
        received = owner.cache.prepare_case.call_args.kwargs
        self.assertEqual(len(received['documents']), 10)
        self.assertEqual(received['input_tokens'], len(ids))
        owner.reset('r', 'native', arm='recomputed', input_ids=ids, documents=docs, phase='case2')
        self.assertEqual(len(owner.docs), 10)
        self.assertEqual(owner.expected_document_count, 10)
        self.assertEqual(owner.state, 'RUNNING')
        self.assertEqual(owner.native_model.yoco_boundary_layer_idx, 20)

    def test_default_controller_rejects_unannounced_ten_before_bank_prepare(self):
        owner = controller()
        ids, docs = inputs()
        with self.assertRaises(ValueError):
            owner.prepare_case(input_ids=ids, documents=docs, index_device='cpu-fixture')
        owner.cache.prepare_case.assert_not_called()

    def test_ten_block_tail_and_halo_window_closure_across_native_chunks(self):
        ids, supplied = inputs()
        docs = s1_documents(ids, supplied, expected_document_count=10)
        plans = all_chunks(docs, len(ids))
        self.assertGreater(len(plans), 1)
        doc_positions = {p for d in docs for p in range(d['start'], d['end'])}
        tails = {p for d in docs for p in range(d['end']-128, d['end'])}
        nondoc = set(range(len(ids))) - doc_positions
        selected = {p for c in plans for p in c['selected_positions']}
        self.assertEqual(selected, tails | nondoc)
        self.assertEqual(len(tails), 10*128)
        halos = {p for d in docs for p in range(d['end']-256, d['end']-128)}
        previous = []
        for chunk in plans:
            result = local_memory_plan(docs, len(ids), chunk['chunk_start'], chunk['chunk_end'],
                                       chunk['selected_positions'], previous)
            for q in chunk['selected_positions']:
                self.assertLessEqual(set(range(max(0,q-127),q+1)) - selected, halos)
            self.assertFalse(chunk['extra_cross_document_links'])
            self.assertFalse(chunk['native_yoco140_query_intersection'])
            self.assertIsInstance(result, dict)
            previous = [p for p in previous+chunk['selected_positions'] if p >= chunk['chunk_end']-127]


if __name__ == '__main__':
    unittest.main()
