"""Regressions for the failed v1 pilot; CPU fixtures, never model qualification."""
import importlib.util
import json
from types import SimpleNamespace
import unittest

from kv_cache_engram.core import ValidationError, digest
from kv_cache_engram.native_reuse import CHUNK, DIM, DOC, LAYERS, KVCacheEngramHook, NativeReuseOps
from test_native_reuse import FakeOps, Module, Tensor, inputs


class ReportRoundTripTests(unittest.TestCase):
    def complete_report(self, mode):
        ops = FakeOps()
        modules = [Module(layer, ops) for layer in LAYERS]
        hook = KVCacheEngramHook(modules, model_sha256="a"*64,
            tokenizer_sha256="b"*64, execution_sha256="c"*64,
            ops=ops, enabled=True).install()
        self.addCleanup(hook.close)
        ids, docs = ([1]*DOC, None) if mode == "capture_independent" else inputs()
        hook.reset("roundtrip-fixture", mode, input_ids=ids, documents=docs, phase="test")
        for start in range(0, len(ids), CHUNK):
            end = min(start+CHUNK, len(ids))
            metadata = SimpleNamespace(canonical_request_binding=hook.chunk_marker(start, end),
                total_q=end-start, num_reqs=1, is_prompt=True, normal_full_view=True)
            hidden = Tensor(ids[start:end])
            cross = SimpleNamespace(consumer_full_metadata=None)
            for module in modules:
                module.forward(hidden, None, metadata, hidden, cross)
        hook.finish()
        return hook, hook.report()

    def assert_roundtrip_seal(self, report):
        phase = {"schema": "fixture-only-phase", "hook": report}
        sealed = dict(phase, sha256=digest(phase))
        decoded = json.loads(json.dumps(sealed, sort_keys=True, allow_nan=False))
        self.assertEqual(decoded, sealed)
        self.assertEqual(decoded["sha256"], digest({k: v for k, v in decoded.items() if k != "sha256"}))
        self.assertEqual(set(report["coverage"]), {str(layer) for layer in LAYERS})
        self.assertEqual(set(report["counts"]), {str(layer) for layer in LAYERS})
        self.assertFalse(report["native_execution_declared"])

    def test_independent_capture_report_survives_json_before_verification(self):
        hook, report = self.complete_report("capture_independent")
        self.assert_roundtrip_seal(report)
        self.assertEqual(report["coverage"], {str(layer): DOC for layer in LAYERS})
        # Published statistics must not alias the live internal layer counters.
        report["counts"]["3"]["q_rows"] = -1
        self.assertEqual(hook.counts[3]["q_rows"], DOC)

    def test_split_identity_report_survives_json_before_verification(self):
        hook, report = self.complete_report("split_identity")
        self.assertTrue(hook.identity_qualified)
        self.assertTrue(report["identity_exact"])
        self.assertTrue(report["qualification"])
        self.assert_roundtrip_seal(report)

    def test_regression_fixture_exposes_numeric_key_sorting_bug(self):
        old = {"coverage": {layer: DOC for layer in LAYERS}}
        self.assertNotEqual(digest(old), digest(json.loads(json.dumps(old))))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional CPU torch unavailable")
class InferenceAllocationTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.ops = NativeReuseOps()
        self.ops.torch = torch
        # Direct CPU dependency injection only: never from_deepinfer().

    def test_outer_inference_mode_allocation_has_version_without_autograd(self):
        t = self.torch
        with t.inference_mode():
            source = t.ones((4, DIM), dtype=t.bfloat16, device="cpu")
            self.assertTrue(t.is_inference(source))
            result = self.ops.allocate(4, source)
            self.assertTrue(t.is_inference_mode_enabled(), "caller inference scope must be restored")
            self.assertFalse(t.is_inference(result))
            self.assertFalse(result.requires_grad)
            self.assertEqual(self.ops.version(result), 0)
            self.ops.copy(result, source)
            self.assertGreater(self.ops.version(result), 0)
            self.assertTrue(t.equal(result, source))

    def test_chunk_copy_and_alias_mutation_keep_shared_version_guard(self):
        t = self.torch
        with t.inference_mode():
            source = t.arange(8*DIM, device="cpu", dtype=t.float32).reshape(8, DIM).to(t.bfloat16)
            resident = self.ops.allocate(8, source)
            before = self.ops.version(resident)
            self.ops.copy(self.ops.slice(resident, 0, 3), source[:3])
            first = self.ops.version(resident)
            self.ops.copy(self.ops.slice(resident, 3, 8), source[3:])
            complete = self.ops.version(resident)
            self.assertGreater(first, before)
            self.assertGreater(complete, first)
            self.assertTrue(t.equal(resident, source))
            alias = self.ops.slice(resident, 1, 2)
            alias.add_(1)
            self.assertGreater(self.ops.version(resident), complete,
                               "mutation through a view must invalidate captured version")
            self.assertEqual(self.ops.version(alias), self.ops.version(resident))

    def test_scatter_temporary_is_versioned_under_inference_mode(self):
        t = self.torch
        with t.inference_mode():
            source = t.ones((2, DIM), dtype=t.bfloat16, device="cpu")
            result = self.ops.allocate(4, source)
            before = self.ops.version(result)
            self.ops.scatter(result, (1, 3), source)
            self.assertGreater(self.ops.version(result), before)
            self.assertTrue(t.equal(self.ops.take(result, (1, 3)), source))

    def test_capture_finish_records_version_and_export_rejects_later_alias_write(self):
        t = self.torch
        # Exercise the actual finish/export control flow with one shared CPU
        # buffer as a fixture for all layers. This is not a native capture and
        # makes no numerical identity or model/geometry qualification claim.
        hook = object.__new__(KVCacheEngramHook)
        hook.state, hook.mode, hook.ids = "RUNNING", "capture_independent", (1,)*DOC
        hook.coverage = {layer: DOC for layer in LAYERS}
        hook.documents = ({"key": "d"*64},)
        hook.ops = self.ops
        self.ops.devices = set()  # synchronize() must never query/init CUDA.
        with t.inference_mode():
            source = t.ones((CHUNK, DIM), dtype=t.bfloat16, device="cpu")
            resident = self.ops.allocate(DOC, source)
            for start in range(0, DOC, CHUNK):
                self.ops.copy(self.ops.slice(resident, start, start+CHUNK), source)
            hook.bank = {("d"*64, layer): {"tensor": resident, "cursor": DOC, "complete": False}
                         for layer in LAYERS}
            hook.finish()
            self.assertEqual(hook.state, "COMPLETE")
            self.assertTrue(all(entry["complete"] for entry in hook.bank.values()))
            self.assertEqual(len(list(hook.iter_cache_entries())), len(LAYERS))
            self.ops.slice(resident, 0, 1).add_(1)
            with self.assertRaisesRegex(ValidationError, "invalid resident cache"):
                list(hook.iter_cache_entries())

    def test_foreign_inference_tensor_still_fails_closed(self):
        t = self.torch
        with t.inference_mode():
            foreign = t.ones((1, DIM), dtype=t.bfloat16, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "Inference tensors do not track version counter"):
            self.ops.version(foreign)

    def test_allocation_preserves_noninference_grad_scope(self):
        t = self.torch
        with t.enable_grad():
            source = t.ones((1, DIM), dtype=t.bfloat16, device="cpu", requires_grad=True)
            result = self.ops.allocate(1, source)
            self.assertTrue(t.is_grad_enabled())
            self.assertFalse(result.requires_grad)
            self.assertFalse(t.is_inference(result))
            self.assertEqual(self.ops.version(result), 0)


if __name__ == "__main__":
    unittest.main()
