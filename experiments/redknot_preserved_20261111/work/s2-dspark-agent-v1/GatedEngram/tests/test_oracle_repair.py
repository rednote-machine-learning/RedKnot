"""CPU causal operator fixtures, never evidence of real-model quality or speed."""
from contextlib import nullcontext
import importlib.util
from types import SimpleNamespace
import unittest

from kv_cache_engram.core import ValidationError, digest
from kv_cache_engram.native_reuse import DOC
from kv_cache_engram.oracle_repair import (
    ORACLE_LAYERS, PROTECTED_LAYERS, OracleRepairConfig, OracleRepairController, oracle_spans,
)


class OracleSpanTests(unittest.TestCase):
    def test_document_and_native_carry_guards_are_intersected(self):
        spans = oracle_spans([dict(start=0, end=DOC, key="a"*64)], 0, 1024)
        self.assertEqual([(s["input_start"], s["input_end"]) for s in spans], [(128, 512), (512, 896)])
        self.assertEqual(spans[-1]["block_end"], 1024)

    def test_cross_chunk_fragments_and_short_tail_never_injected(self):
        spans = oracle_spans([dict(start=7, end=DOC+7, key="a"*64)], 0, 1024)
        self.assertEqual([(s["input_start"], s["input_end"]) for s in spans], [(135, 519)])
        self.assertEqual(oracle_spans([dict(start=0, end=DOC, key="a"*64)], 0, 128), [])

    def test_fixed_layers_and_positive_resource_limits(self):
        self.assertEqual(ORACLE_LAYERS, (6, 7, 9, 10, 11, 12, 13, 16, 17, 18))
        self.assertTrue(set(PROTECTED_LAYERS).isdisjoint(ORACLE_LAYERS))
        for kwargs in (dict(max_cpu_reference_bytes=0), dict(max_extra_bytes=-1)):
            with self.assertRaises(ValidationError):
                OracleRepairConfig(**kwargs)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional Torch CPU fixture dependency absent")
class OracleCausalFixtureTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch

    def fixture(self, *, rows=512, cpu_limit=256 << 20, documents=True):
        t = self.t
        class Module:
            use_fused_norm_rope_attn_rope_cast = False
            def __init__(self, layer):
                self.layer_idx, self.calls = layer, 0
                self.seen_values = None
            def _prefill_attention(self, q, kv, ca, swa, cached, current, metadata):
                self.calls += 1
                self.seen_values = kv.detach().clone()
                return q + .25*kv[cached:cached+current].expand(-1, q.shape[1], -1)
        modules = {layer: Module(layer) for layer in range(19)}
        class Ops:
            native_execution = False
            backend = "torch-cpu-causal-fixture-not-native"
            torch = t
            def __init__(self):
                self.postprocess_calls = 0
            @staticmethod
            def version(value):
                return value._version
            def postprocess(self, module, canonical, metadata):
                self.postprocess_calls += 1
                # Explicitly position-dependent fake operator catches double RoPE.
                return (canonical+metadata.positions[:, None].float()*.001).unsqueeze(1), None
        ops = Ops()
        canonical = t.zeros(1, 512).expand(DOC, -1)
        frame_slot = [None]
        hook = SimpleNamespace(modules=modules, state="RUNNING", mode="native", identity_qualified=True,
            request_id="capture", phase="capture", input_sha256="a"*64, ids=range(rows), ops=ops,
            allow_sync=nullcontext, documents=[dict(start=0, end=DOC, key="b"*64)] if documents else [],
            bank={("b"*64, layer): dict(tensor=canonical, complete=True, version=canonical._version)
                  for layer in ORACLE_LAYERS}, _frame=lambda module: frame_slot[0])
        controller = OracleRepairController(hook, OracleRepairConfig(max_cpu_reference_bytes=cpu_limit)).install()
        self.addCleanup(controller.close)
        return hook, controller, frame_slot

    def run_arm(self, hook, controller, slot, mode, *, bias=2.):
        t, rows = self.t, len(hook.ids)
        hook.state, hook.request_id, hook.phase = "RUNNING", mode, mode
        controller.reset(mode)
        hidden = t.full((rows, 1, 512), bias)
        snapshots = []
        for layer in ORACLE_LAYERS:
            module = hook.modules[layer]
            marker = dict(request_id=mode, phase=mode, start=0, end=rows)
            metadata = SimpleNamespace(total_q=rows, positions=t.arange(rows), canonical_request_binding=marker)
            slot[0] = SimpleNamespace(marker=marker, module=module, metadata=metadata)
            # All rows are projected from current (possibly hybrid) hidden.
            kv = t.cat((t.full((128, 1, 512), 9.), hidden+.001*metadata.positions[:, None, None]))
            before = kv.clone()
            context = module._prefill_attention(hidden, kv, None, None, 128, rows, metadata)
            self.assertTrue(t.equal(kv, before), "controller must not change native stored/carry KV")
            snapshots.append(module.seen_values)
            # A simple downstream nonlinear update makes propagation observable.
            hidden = context + .01*context.square()
        slot[0] = None
        hook.state = "COMPLETE"
        return hidden, controller.finish(), snapshots

    def test_c_repairs_downstream_output_b_really_propagates(self):
        hook, controller, slot = self.fixture()
        a, ra, native = self.run_arm(hook, controller, slot, "capture")
        plan = controller.export_plan()
        b, rb, cached = self.run_arm(hook, controller, slot, "reuse")
        c, rc, repaired = self.run_arm(hook, controller, slot, "repair")
        self.assertGreater((a-b).abs().max().item(), 0)
        self.assertTrue(self.t.equal(a, c), "all injected fixture rows are in the single restored source block")
        self.assertNotEqual(rb["mode"], ra["mode"])
        self.assertEqual(ra["plan_sha256"], plan["sha256"])
        for report in (ra, rb, rc):
            self.assertEqual(report["sha256"], digest({k:v for k,v in report.items() if k != "sha256"}))
            self.assertFalse(report["native_execution"])
            self.assertTrue(report["no_ttft_claim"])
        self.assertTrue(rb["propagates_to_later_layers"])
        self.assertTrue(rc["forced_unqualified_reuse"])
        self.assertTrue(all(r["reused_rows"] == 256 for r in rb["records"]))
        self.assertTrue(all(r["oracle_restored_rows"] == 256 and r["reused_rows"] == 0 for r in rc["records"]))
        # Native carry and document head remain original even under forced reuse.
        self.assertTrue(self.t.equal(cached[0][:256], native[0][:256]))
        self.assertTrue(self.t.equal(cached[0][-128:], native[0][-128:]))

    def test_reference_rows_are_positioned_once_and_c_does_not_postprocess_them_again(self):
        hook, controller, slot = self.fixture()
        _, _, full = self.run_arm(hook, controller, slot, "capture")
        plan = controller.export_plan()
        reference = controller.references[(6, 0, 512)]["tensor"]
        self.assertTrue(self.t.equal(reference, full[0][256:512]))
        before = hook.ops.postprocess_calls
        self.run_arm(hook, controller, slot, "reuse")
        reuse_calls = hook.ops.postprocess_calls-before
        before = hook.ops.postprocess_calls
        self.run_arm(hook, controller, slot, "repair")
        self.assertEqual(hook.ops.postprocess_calls-before, reuse_calls)
        self.assertEqual(plan["records"][0]["oracle_reference_shape"], [256, 1, 512])

    def test_repair_can_worsen_a_target_and_is_not_clipped_or_replaced_by_b(self):
        hook, controller, slot = self.fixture()
        self.run_arm(hook, controller, slot, "capture")
        b, _, _ = self.run_arm(hook, controller, slot, "reuse")
        c, _, _ = self.run_arm(hook, controller, slot, "repair")
        # This fixture target is B itself; restorative closeness to A is not a
        # guarantee of target quality. The controller must preserve that fact.
        error_b, error_c = (b-b).abs().mean().item(), (c-b).abs().mean().item()
        self.assertLess(error_b-error_c, 0)

    def test_protected_layers_have_no_controller_hook(self):
        hook, controller, slot = self.fixture()
        for layer in PROTECTED_LAYERS:
            self.assertNotIn("_prefill_attention", hook.modules[layer].__dict__)
        self.assertEqual(set(m.layer_idx for m, _ in controller.handles), set(ORACLE_LAYERS))

    def test_b_and_c_rejected_before_complete_a(self):
        hook, controller, slot = self.fixture()
        for mode in ("reuse", "repair"):
            with self.assertRaises(ValidationError):
                controller.reset(mode)
        controller.reset("capture")
        hook.state = "COMPLETE"
        with self.assertRaises(ValidationError):
            controller.finish()

    def test_reference_and_plan_mutation_fail_closed(self):
        hook, controller, slot = self.fixture()
        self.run_arm(hook, controller, slot, "capture")
        entry = controller.references[(6, 0, 512)]
        entry["tensor"].add_(1)
        hook.state = "RUNNING"
        with self.assertRaises(ValidationError):
            controller.reset("repair")

    def test_exported_plan_is_not_a_mutable_alias(self):
        hook, controller, slot = self.fixture()
        self.run_arm(hook, controller, slot, "capture")
        exported = controller.export_plan()
        exported["records"][0]["selected_span"]["input_start"] = -1
        self.assertEqual(controller.export_plan()["records"][0]["selected_span"]["input_start"], 128)

    def test_mismatched_input_and_gated_native_hook_are_rejected(self):
        hook, controller, slot = self.fixture()
        self.run_arm(hook, controller, slot, "capture")
        hook.state, hook.input_sha256 = "RUNNING", "c"*64
        with self.assertRaises(ValidationError):
            controller.reset("reuse")
        hook.input_sha256, hook.mode = "a"*64, "gated"
        with self.assertRaises(ValidationError):
            controller.reset("repair")

    def test_small_cpu_budget_aborts_without_frozen_plan(self):
        hook, controller, slot = self.fixture(cpu_limit=1)
        with self.assertRaises(ValidationError):
            self.run_arm(hook, controller, slot, "capture")
        self.assertTrue(controller.failed)
        with self.assertRaises(ValidationError):
            controller.export_plan()

    def test_empty_candidates_stay_zero_in_all_three_arms(self):
        hook, controller, slot = self.fixture(rows=128, documents=False)
        outputs = []
        for mode in ("capture", "reuse", "repair"):
            value, report, _ = self.run_arm(hook, controller, slot, mode)
            outputs.append(value)
            self.assertTrue(all(r["source_rows"] == r["injected_rows"] == r["oracle_restored_rows"] == 0
                                for r in report["records"]))
        self.assertTrue(all(self.t.equal(outputs[0], x) for x in outputs))
        self.assertEqual(controller.export_plan()["cpu_reference_bytes"], 0)

    def test_decode_and_inactive_calls_are_passthrough(self):
        hook, controller, slot = self.fixture()
        module = hook.modules[6]
        q = self.t.zeros(1, 1, 512)
        kv = self.t.ones(129, 1, 512)
        module._prefill_attention(q, kv, None, None, 128, 1, SimpleNamespace())
        self.assertEqual(module.calls, 1)
        controller.reset("capture")
        module._prefill_attention(q, kv, None, None, 128, 1, SimpleNamespace())
        self.assertEqual(module.calls, 2)
        self.assertFalse(controller.records)

    def test_foreign_hook_is_not_silently_overwritten(self):
        hook, controller, slot = self.fixture()
        foreign = lambda *args: None
        hook.modules[6]._prefill_attention = foreign
        with self.assertRaises(ValidationError):
            controller.close()
        self.assertIs(hook.modules[6]._prefill_attention, foreign)


if __name__ == "__main__":
    unittest.main()
