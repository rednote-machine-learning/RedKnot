"""Pure stdlib contract/oracle tests; no model or CUDA import."""

import copy
import hashlib
import importlib.util
import json
import pathlib
import random
import subprocess
import sys
import unittest


PATH = pathlib.Path(__file__).resolve().parents[1] / "kv_cache_engram" / "tail_link_plan.py"
SPEC = importlib.util.spec_from_file_location("standalone_tail_link_plan", PATH)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def fixture(n=2, length=32768, prefix=3, suffix=4):
    docs = []
    cursor = prefix
    for index in range(n):
        docs.append(dict(start=cursor, end=cursor + length, key=f"doc{index}"))
        cursor += length
    return docs, cursor + suffix


class TailLinkPlanTests(unittest.TestCase):
    def check_plan(self, *args, **kwargs):
        result = P.plan(*args, **kwargs)
        audit = P.audit_plan_result(result, *args, **kwargs)
        self.assertTrue(audit["passed"], audit)
        indices = P.materialize_indices(result["selected_memory_lengths"], result["memory_count"])
        self.assertTrue(P.audit_observed_indices(result["memory_positions"], result["selected_positions"], indices)["passed"])
        return result

    def test_cpu_only(self):
        # The full validation harness deliberately imports CPU Torch before
        # discovery. Check this module's import isolation in a fresh process,
        # not the unrelated modules already loaded by the test harness.
        code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('tail_plan',"+repr(str(PATH))+");"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "assert 'torch' not in sys.modules;"
                "assert not any(n=='deepinfer' or n.startswith('deepinfer.') for n in sys.modules)")
        result = subprocess.run([sys.executable,"-I","-B","-c",code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_one_doc_all_tail_and_template(self):
        docs, full = fixture(1, 256)
        r = self.check_plan(docs, list(range(full)), 0, full, full)
        self.assertEqual(r["selected_rows"], 3 + 128 + 4)
        self.assertEqual(r["selected_positions"][3], docs[0]["end"] - 128)
        self.assertEqual(r["memory_count"], 135)

    def test_two_docs_nonlocal_prior_tail(self):
        docs, full = fixture(2, 256)
        q = docs[1]["end"] - 128
        r = self.check_plan(docs, [q], 256, full, full)
        self.assertEqual(r["selected_memory_lengths"], [3 + 128 + 1])
        self.assertIn(docs[0]["end"] - 128, r["memory_positions"])
        self.assertNotIn(q + 1, r["memory_positions"])

    def test_eight_docs_real_32k_geometry(self):
        docs, full = fixture(8)
        r = self.check_plan(docs, list(range(full - 140, full)), full - 8192, full, full)
        self.assertEqual(len(r["current_positions"]), 140)
        self.assertEqual(r["selected_rows"], 132)
        self.assertEqual(r["memory_count"], 8 * 128 + 7)

    def test_global_offset_is_compact_not_absolute(self):
        docs, full = fixture(2)
        r = self.check_plan(docs, [full - 1], full - 10, full, full)
        self.assertEqual(r["global_index_offset"], 263)
        self.assertNotEqual(r["global_index_offset"], full)

    def test_empty_query_no_memory_no_forced_rows(self):
        docs, full = fixture()
        r = self.check_plan(docs, [], 0, 8192, full)
        self.assertEqual(r["selected_local_rows"], [])
        self.assertEqual(r["memory_positions"], [])

    def test_yoco_nonfinal_placeholder_not_forced(self):
        docs, full = fixture()
        r = self.check_plan(docs, [8191], 0, 8192, full, available_nondoc_positions=[])
        self.assertEqual(r["selected_rows"], 0)
        self.assertEqual(r["reference_memory_lengths"], [0])

    def test_yoco_final_140_actual_positions_preserved(self):
        docs, full = fixture()
        positions = list(range(full - 140, full))
        r = self.check_plan(docs, positions, full - 8192, full, full,
                            available_nondoc_positions=list(range(full - 4, full)))
        self.assertEqual(r["current_positions"], positions)
        self.assertEqual(r["memory_count"], 260)
        self.assertEqual(r["selected_rows"], 132)

    def test_explicit_nondoc_availability_does_not_fabricate(self):
        docs, full = fixture(1, 256)
        r = self.check_plan(docs, [full - 1], 0, full, full,
                            available_nondoc_positions=[0, full - 1])
        self.assertEqual(r["memory_positions"][0], 0)
        self.assertNotIn(1, r["memory_positions"])
        self.assertEqual(r["memory_count"], 130)

    def test_nondoc_queries_are_mandatory_even_before_kv_written(self):
        docs, full = fixture(1, 256)
        r = self.check_plan(docs, [0, 1, 2], 0, 3, full, available_nondoc_positions=[])
        self.assertEqual(r["selected_local_rows"], [0, 1, 2])
        self.assertEqual(r["selected_memory_lengths"], [0, 0, 0])

    def test_non_doc_between_documents(self):
        docs = [dict(start=2, end=258), dict(start=263, end=519)]
        r = self.check_plan(docs, [258, 259, 263], 258, 264, 524,
                            available_nondoc_positions=[0, 1, 258, 259])
        self.assertEqual(r["selected_local_rows"], [0, 1])
        self.assertEqual(r["selected_memory_lengths"], [131, 132])

    def test_same_content_disjoint_occurrences_are_distinct_positions(self):
        docs, full = fixture(2, 256)
        docs[1]["key"] = docs[0]["key"]
        r = self.check_plan(docs, [full - 1], 0, full, full)
        self.assertEqual(set(r["memory_doc_indices"]), {-1, 0, 1})
        rows = P.tail_bank_rows(docs, full)
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows[0]["document_key"], rows[128]["document_key"])
        self.assertNotEqual(rows[0]["position"], rows[128]["position"])

    def test_tail_local_offsets_separate_from_rope_positions(self):
        docs, full = fixture(1)
        q = docs[0]["end"] - 128
        r = self.check_plan(docs, [q], q, q + 1, full, available_nondoc_positions=[])
        self.assertEqual(r["memory_positions"], [q])
        self.assertEqual(r["memory_tail_bank_rows"], [0])
        self.assertEqual(r["memory_doc_token_offsets"], [32640])

    def test_current_vs_historical_memory_mapping(self):
        docs, full = fixture(2, 256)
        q = docs[1]["end"] - 1
        r = self.check_plan(docs, [q], q, q + 1, full)
        self.assertEqual(r["memory_current_local_rows"][-1], 0)
        self.assertTrue(all(x == -1 for x in r["memory_current_local_rows"][:-1]))

    def test_future_document_tails_never_visible(self):
        docs, full = fixture(8, 256)
        q = docs[0]["end"] - 1
        r = self.check_plan(docs, [q], 0, q + 1, full)
        self.assertEqual(set(r["memory_doc_indices"]), {-1, 0})

    def test_dense_reference_query_uses_same_causal_memory(self):
        docs, full = fixture(2, 256)
        queries = [docs[0]["end"] - 1, docs[1]["start"], docs[1]["end"] - 128]
        r = self.check_plan(docs, queries, 0, full, full)
        self.assertEqual(r["selected_local_rows"], [0, 2])
        self.assertEqual(r["reference_memory_lengths"], [131, 131, 132])

    def test_short_document_tail_is_entire_document(self):
        docs, full = fixture(1, 37)
        r = self.check_plan(docs, list(range(full)), 0, full, full)
        self.assertEqual(r["documents"][0]["tail_rows"], 37)
        self.assertEqual(r["selected_rows"], full)

    def test_byte_geometry_40_and_37_layers(self):
        docs, full = fixture(8)
        for layers in (40, 37):
            result = P.canonical_bank_bytes(docs, full, layers=layers)
            self.assertEqual(result["tail_bytes_per_layer"], 1024 * 1024)
            self.assertEqual(result["tail_bank_bytes"], layers * 1024 * 1024)
            self.assertEqual(result["full_bank_bytes"] // result["tail_bank_bytes"], 256)

    def test_final_128_logical_decode_memory_covered(self):
        for suffix in (1, 4, 50, 128, 300):
            docs, full = fixture(8, suffix=suffix)
            positions = list(range(full - 140, full))
            available = [p for p in positions if p >= docs[-1]["end"]]
            r = self.check_plan(docs, positions, full - 8192, full, full,
                                available_nondoc_positions=available)
            result = P.audit_decode_memory_coverage(r["memory_positions"], full)
            self.assertTrue(result["passed"])
            self.assertFalse(result["physical_ring_values_verified"])
            self.assertFalse(result["backtracking_safety_verified"])

    def test_missing_online_decode_memory_fails_not_zero_filled(self):
        docs, full = fixture(1, 256)
        r = self.check_plan(docs, [full - 1], full - 1, full, full,
                            available_nondoc_positions=[full - 1])
        result = P.audit_decode_memory_coverage(r["memory_positions"], full)
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_positions"], list(range(full - 4, full - 1)))

    def test_padding_and_zero_length_rows(self):
        matrix = P.materialize_indices([0, 1, 2], 2, padded_width=4)
        self.assertEqual(matrix, [[-1] * 4, [0, -1, -1, -1], [0, 1, -1, -1]])
        self.assertTrue(P.audit_observed_indices([5, 10], [1, 5, 10], matrix)["passed"])

    def test_bad_causal_mask_future_missing_duplicate_reordered(self):
        for row in ([0, 1], [1, -1], [0, 0], [-1, 0], [2, -1], [-2, 0], [True, -1]):
            self.assertFalse(P.audit_observed_indices([5, 10], [5], [row])["passed"], row)

    def test_bad_index_row_count_and_ragged(self):
        self.assertFalse(P.audit_observed_indices([5], [5], [])["passed"])
        self.assertFalse(P.audit_observed_indices([5], [5, 6], [[0], [0, -1]])["passed"])

    def test_mutated_report_rejected_by_independent_oracle(self):
        docs, full = fixture(1, 256)
        args = (docs, [full - 1], 0, full, full)
        original = P.plan(*args)
        for key in ("selected_local_rows", "memory_positions", "memory_tail_bank_rows",
                    "reference_memory_lengths", "global_index_offset", "plan_sha256"):
            changed = copy.deepcopy(original)
            changed[key] = "tampered"
            self.assertFalse(P.audit_plan_result(changed, *args)["passed"], key)

    def test_invalid_document_intervals(self):
        for docs in ([], [dict(start=2, end=2)], [dict(start=4, end=2)],
                     [dict(start=0, end=11)], [dict(start=0, end=5), dict(start=4, end=8)],
                     [dict(start=5, end=8), dict(start=0, end=4)]):
            with self.assertRaises(ValueError):
                P.plan(docs, [], 0, 10, 10)

    def test_resealed_boolean_integer_substitution_rejected(self):
        docs, full = fixture(1, 256)
        args = (docs, [full - 1], 0, full, full)
        r = P.plan(*args)
        r["selected_local_rows"] = [False]
        body = {key: value for key, value in r.items() if key != "plan_sha256"}
        r["plan_sha256"] = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        audit = P.audit_plan_result(r, *args)
        self.assertFalse(audit["passed"])
        self.assertIn("selected_local_rows: independent oracle mismatch", audit["errors"])

    def test_invalid_current_positions(self):
        docs, full = fixture(1, 256)
        for positions in ([1, 1], [2, 1], [-1], [True], [1.0], [full], [0]):
            with self.assertRaises(ValueError):
                P.plan(docs, positions, 1, full, full)

    def test_invalid_availability(self):
        docs, full = fixture(1, 256)
        for available in ([3], [0, 0], [2, 1], [full], [-1], [True]):
            with self.assertRaises(ValueError):
                P.plan(docs, [], 0, full, full, available_nondoc_positions=available)

    def test_invalid_geometry(self):
        docs, full = fixture(1, 256)
        for kwargs in (dict(tail_tokens=0), dict(tail_tokens=True)):
            with self.assertRaises(ValueError):
                P.plan(docs, [], 0, full, full, **kwargs)
        with self.assertRaises(ValueError):
            P.materialize_indices([2], 1)
        with self.assertRaises(ValueError):
            P.materialize_indices([0], 2, padded_width=1)
        with self.assertRaises(ValueError):
            P.canonical_bank_bytes(docs, full, layers=0)

    def test_randomized_scalar_oracle(self):
        rng = random.Random(98761)
        for _ in range(70):
            n = rng.choice((1, 2, 8))
            docs, full = fixture(n, rng.randint(10, 180), prefix=rng.randint(0, 5))
            start = rng.randrange(full)
            end = rng.randrange(start, full + 1)
            positions = sorted(rng.sample(range(start, end), min(end - start, rng.randrange(20))))
            available = sorted(position for position in range(end)
                               if not any(doc["start"] <= position < doc["end"] for doc in docs)
                               and rng.randrange(2))
            self.check_plan(docs, positions, start, end, full,
                            tail_tokens=rng.randint(1, 128), available_nondoc_positions=available)


if __name__ == "__main__":
    unittest.main()
