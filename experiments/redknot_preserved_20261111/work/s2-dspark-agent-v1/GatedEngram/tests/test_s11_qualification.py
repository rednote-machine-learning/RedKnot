"""Admission/state tests; these do not claim native GPU numerical parity."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from kv_cache_engram.core import token_digest
from kv_cache_engram.s1_runtime import S1Controller
from kv_cache_engram.s1_system_runtime import S1SystemController
from kv_cache_engram.s11_qualification import S11QualificationController, ADAPTER_SCHEMA


class Backend:
    def __init__(self):
        self.enabled = None
        self.phases = []
        self.bank = {}
    def configure(self, *, enabled): self.enabled = enabled
    def begin_phase(self, *args, **kwargs): self.phases.append((self.enabled, args, kwargs))


class S11QualificationTests(unittest.TestCase):
    def make(self, n=16384):
        c = object.__new__(S11QualificationController)
        c.state = "IDLE"; c.ctx = c.capture_frame = None
        c.native_model = NS(yoco_boundary_layer_idx=20); c.original_boundary = 20
        c._configured_system = True; c._configured_group = 1
        c.system_qualified = {1: False, 2: False, 4: False}
        c._empty_group_case = None; c.expected_document_count = 10
        c.sparse_reference = {}; c.sparse_reference_input = None; c.sparse_qualified = False
        c.qualified = True; c.cache = Backend(); c.attention = Backend()
        c.case_binding = (); ids = [1] * n; key = token_digest(ids)
        c.case_input = key
        c.t = NS(long="long", int32="int32", tensor=lambda v, **kw: tuple(v))
        docs = [dict(input_start=0, input_end=n, token_ids_sha256=key, document_id=key)]
        args = dict(arm="s1", input_ids=ids, documents=docs, phase="reference", index_device="cpu")
        return c, args

    def test_actual_base_reset_uses_enabled_backends_without_qualifying(self):
        c, args = self.make()
        c.reset("r", "reference_sparse", **args)
        self.assertEqual(c.state, "RUNNING")
        self.assertEqual(len(c.prepared), 2)
        self.assertTrue(c.cache.phases[0][0]); self.assertTrue(c.attention.phases[0][0])
        self.assertTrue(c.cache.phases[0][2]["build_global_base"])
        self.assertEqual(c.system_qualified, {1: False, 2: False, 4: False})
        self.assertEqual(c.sparse_reference_input, c.case_input)
        self.assertFalse(c.sparse_reference); self.assertIsNone(c.batch_schedule)

    def test_retired_original_policy_is_rejected(self):
        for mode in ("reference_sparse", "sensitive"):
            c, args = self.make(); c.configure(system_enabled=False)
            with self.assertRaises(ValueError): c.reset("r", mode, **args)

    def test_reference_rejects_batched_verified_and_prequalified_states(self):
        for change in ("batch", "verify", "qualified"):
            c, args = self.make()
            if change == "batch": c.configure(system_enabled=True, batch_group_chunks=2)
            if change == "verify": args["verify_sparse"] = True
            if change == "qualified": c.system_qualified[4] = True
            with self.assertRaises(ValueError): c.reset("r", "reference_sparse", **args)

    def test_original_native_identity_and_fresh_bank_guards_remain(self):
        for change in ("native", "bank", "reference", "active"):
            c, args = self.make()
            if change == "native": c.qualified = False
            if change == "bank": c.case_input = "a" * 64
            if change == "reference": c.sparse_reference = {"existing": 1}
            if change == "active": c.state = "RUNNING"
            with self.assertRaises(ValueError): c.reset("r", "reference_sparse", **args)

    def test_real_32769_empty_group_validator_is_isolated(self):
        c, args = self.make(32769); source = "a" * 64
        c._empty_group_case = (c.case_input, source); args["phase"] = "qualification.empty_group.reference"
        c.reset("r", "reference_sparse", system_empty_group_qualification=True,
                source_full_input_ids_sha256=source, **args)
        self.assertEqual(len(c.prepared), 5)
        self.assertEqual(c.prepared[0, 8192]["plan"]["selected_rows"], 0)
        self.assertEqual(c.prepared[24576, 32768]["plan"]["selected_rows"], 127)
        # The original function/global validator remains intact.
        c, args = self.make(32769)
        with self.assertRaisesRegex(ValueError, "complete16K"):
            c.reset("r", "reference_sparse", **args)

    def test_empty_group_source_mismatch_rejected(self):
        c, args = self.make(32769); c._empty_group_case = (c.case_input, "b" * 64)
        args["phase"] = "qualification.empty_group.reference"
        with self.assertRaisesRegex(ValueError, "binding differs"):
            c.reset("r", "reference_sparse", system_empty_group_qualification=True,
                    source_full_input_ids_sha256="a" * 64, **args)

    def test_formal_requires_its_own_actual_group_qualification(self):
        c, args = self.make(); c.configure(system_enabled=True, batch_group_chunks=4)
        c.system_qualified[1] = True
        with self.assertRaisesRegex(ValueError, "lacks actual"):
            c.reset("r", "sensitive", **args)

    def test_reference_finish_stays_unqualified_and_emits_contract(self):
        c, args = self.make(); c.reset("r", "reference_sparse", **args)
        with patch.object(S1Controller, "finish", return_value={"sparse_identity_exact": False}):
            report = c.finish()
        self.assertIs(report["system_numerically_qualified"], False)
        self.assertEqual(report["s11_qualification_adapter_schema"], ADAPTER_SCHEMA)
        self.assertFalse(report["s11_original_s1_benchmark_executed"])

    def test_candidate_finish_only_promotes_verified_group(self):
        c, args = self.make(); c.reset("r", "reference_sparse", **args)
        c.mode = "sensitive"; c.verify_sparse = True; c.batch_group_chunks = 4
        with patch.object(S1Controller, "finish", return_value={"sparse_identity_exact": False}):
            with self.assertRaisesRegex(ValueError, "byte parity"): c.finish()
        self.assertFalse(any(c.system_qualified.values()))
        with patch.object(S1Controller, "finish", return_value={"sparse_identity_exact": True}):
            report = c.finish()
        self.assertEqual(c.system_qualified, {1: False, 2: False, 4: True})
        self.assertTrue(report["system_numerically_qualified"])

    def test_original_system_reference_restriction_is_unchanged(self):
        c, args = self.make()
        with self.assertRaisesRegex(ValueError, "reference remains original"):
            S1SystemController.reset(c, "r", "reference_sparse", **args)


if __name__ == "__main__": unittest.main()
