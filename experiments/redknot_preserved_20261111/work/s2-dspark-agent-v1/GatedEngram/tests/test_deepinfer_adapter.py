"""Synthetic native-signature CPU fixtures, not a real model benchmark."""
from dataclasses import replace
from types import SimpleNamespace
import importlib.abc
import math
import sys
import threading
import unittest

from kv_cache_engram.core import CacheIdentity, ValidationError, token_digest
from kv_cache_engram.deepinfer_adapter import (
    CPUReferenceCaptureOps, DeepInferCaptureSession, NativeCaptureOps,
)


class FixtureAttention:
    def __init__(self, layer=3, width=2):
        self.layer_idx = layer
        self.qk_head_dim = width
        self.qk_rope_head_dim = 1
        self.window_size = 128
        self.use_swa = True
        self.is_kv_source = self.is_index_source = self.is_index_k_source = False
        self.is_yoco_hidden_consumer_layer = self.yoco_swa_share_hidden = False
        self.kv_swa_norm = SimpleNamespace(weight=(2.0,) * width, variance_epsilon=1e-6, out_scale=1.0)
        self.calls = []
        self.fail = False

    def _prepare_swa_kv(self, raw, cache, metadata, use_cp):
        self.calls.append((raw, cache, metadata, use_cp))
        if self.fail:
            raise RuntimeError("fixture native failure")
        self.last_result = ("native-post-rope-cast-result", metadata.total_q)
        return self.last_result


def fixture(*, layers=(3,), start=2, count=4, input_count=8, **kwargs):
    ids = tuple(range(input_count))
    identities = {layer: CacheIdentity("a" * 64, "b" * 64, "c" * 64,
                   token_digest(ids[start:start + count]), layer, count, 2) for layer in layers}
    modules = {layer: FixtureAttention(layer) for layer in layers}
    session = DeepInferCaptureSession(modules, request_id="capture-r", phase="offline-full",
                  input_ids=ids, identities=identities, document_start=start, **kwargs)
    return session, modules


def metadata(session, start, end, **changes):
    # Independent expansion from absolute visible keys into [history | chunk].
    # The real native metadata already has both forms at the capture point.
    pivots, expanded = [], []
    for pos in range(start, end):
        keys = tuple(range(max(0, pos - 127), pos + 1))
        history = tuple(128 - start + key for key in keys if key < start)
        current = tuple(128 - start + key for key in keys if key >= start)
        pivots.append((history[0] if history else 128, len(history), current[0], len(current)))
        expanded.append(history + current + (-1,) * (128 - len(keys)))
    values = dict(is_prompt=True, num_reqs=1, total_q=end - start,
                  positions=tuple(range(start, end)), cu_seqlens_q=(0, end - start),
                  canonical_request_binding=session.chunk_marker(start, end),
                  total_k_swa=128, kv_swa_full_index_pivots=tuple(pivots),
                  kv_swa_full_index=tuple(expanded))
    values.update(changes)
    return SimpleNamespace(**values)


def call(session, module, start, end, *, meta=None):
    rows = tuple((float(i + 1), float(i + 2)) for i in range(start, end))
    return module._prepare_swa_kv(rows, None, metadata(session, start, end) if meta is None else meta, False)


class CaptureSessionTests(unittest.TestCase):
    def test_native_signature_pass_through_and_document_slice(self):
        session, modules = fixture(layers=(3, 4))
        originals = {layer: module._prepare_swa_kv.__func__ for layer, module in modules.items()}
        with session:
            for start, end in ((0, 3), (3, 8)):
                for module in modules.values():
                    result = call(session, module, start, end)
                    self.assertEqual(result, ("native-post-rope-cast-result", end - start))
                    self.assertIs(result, module.last_result)
        self.assertEqual(session.state, "COMPLETE")
        for layer, module in modules.items():
            self.assertNotIn("_prepare_swa_kv", vars(module))
            self.assertIs(module._prepare_swa_kv.__func__, originals[layer])
            self.assertEqual(len(module.calls), 2)
            expected = CPUReferenceCaptureOps().normalize(module, tuple((float(i + 1), float(i + 2)) for i in range(2, 6)))
            self.assertEqual(session.rows_by_layer[layer], expected)
        self.assertFalse(session.capture_provenance["native_operator"])
        self.assertFalse(session.capture_provenance["context_equivalence_certified"])

    def test_cpu_norm_uses_weight_and_epsilon(self):
        module = FixtureAttention()
        module.kv_swa_norm.weight = (2.0, 3.0)
        module.kv_swa_norm.variance_epsilon = 0.5
        row = CPUReferenceCaptureOps().normalize(module, ((3.0, 4.0),))[0]
        self.assertAlmostEqual(row[0], 6.0 / math.sqrt(13.0))
        self.assertAlmostEqual(row[1], 12.0 / math.sqrt(13.0))

    def test_full_encoder_can_carry_unused_yoco_plan(self):
        session, modules = fixture()
        with session:
            m = metadata(session, 0, 8, yoco_truncation_enabled=True,
                         yoco_consumer_query_indices=(6, 7),
                         yoco_mask_swa_index_pivots=((128, 0, 128, 0),) * 8)
            call(session, modules[3], 0, 8, meta=m)
        self.assertEqual(session.state, "COMPLETE")

    def test_incomplete_capture_fails_and_restores(self):
        session, modules = fixture()
        with self.assertRaisesRegex(ValidationError, "incomplete"):
            with session:
                call(session, modules[3], 0, 3)
        self.assertEqual(session.state, "ABORTED")
        self.assertNotIn("_prepare_swa_kv", vars(modules[3]))
        with self.assertRaises(ValidationError):
            _ = session.rows_by_layer

    def test_native_exception_leaves_no_export_and_restores(self):
        session, modules = fixture()
        modules[3].fail = True
        with self.assertRaises(RuntimeError):
            with session:
                call(session, modules[3], 0, 8)
        self.assertEqual(session.state, "ABORTED")
        self.assertEqual(session._rows[3], [])
        self.assertNotIn("_prepare_swa_kv", vars(modules[3]))

    def test_wrong_context_phase_positions_or_shape_abort_before_native(self):
        for mutation in ("request", "phase", "sha", "positions", "rows", "segments", "cp", "decode", "batch", "masked", "yoco_masked"):
            with self.subTest(mutation=mutation):
                session, modules = fixture()
                m = metadata(session, 0, 8)
                if mutation == "request": m.canonical_request_binding["request_id"] = "foreign"
                if mutation == "phase": m.canonical_request_binding["phase"] = "foreign"
                if mutation == "sha": m.canonical_request_binding["input_ids_sha256"] = "f" * 64
                if mutation == "positions": m.positions = (0, 1, 2, 3, 4, 5, 7, 8)
                if mutation == "rows": m.total_q = 4
                if mutation == "segments": m.cu_seqlens_q = (0, 4, 8)
                if mutation == "decode": m.is_prompt = False
                if mutation == "batch": m.num_reqs = 2
                if mutation == "masked": m.kv_swa_full_index_pivots = ((0, 1),)
                if mutation == "yoco_masked":
                    # get_mask_view installs the candidate into the actual view.
                    m.yoco_mask_swa_index_pivots = ((128, 0, 128, 0),) * 8
                    m.kv_swa_full_index_pivots = m.yoco_mask_swa_index_pivots
                with self.assertRaises(ValidationError):
                    with session:
                        modules[3]._prepare_swa_kv(((1., 2.),) * 8, None, m, mutation == "cp")
                self.assertEqual(modules[3].calls, [])
                self.assertEqual(session.state, "ABORTED")

    def test_normal_full_swa_across_window_and_metadata_copy_boundaries(self):
        class BoundedOps(CPUReferenceCaptureOps):
            def integer_matrix_rows(self, values, start, end):
                self.slices.append(end - start)
                return super().integer_matrix_rows(values, start, end)
        ops = BoundedOps()
        ops.slices = []
        session, modules = fixture(start=0, count=1024, input_count=1024, ops=ops)
        with session:
            for start, end in ((0, 127), (127, 128), (128, 256), (256, 1024)):
                call(session, modules[3], start, end)
        self.assertEqual(session.state, "COMPLETE")
        self.assertTrue(ops.slices and max(ops.slices) == 256)

    def test_actual_swa_view_must_be_complete_and_consistent(self):
        for mutation in ("missing_pivots", "missing_index", "index_only_masked", "zero_state",
                         "wrong_history", "wrong_shape", "boolean_index"):
            session, modules = fixture(input_count=140)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                with session:
                    call(session, modules[3], 0, 130)
                    m = metadata(session, 130, 140)
                    if mutation == "missing_pivots": m.kv_swa_full_index_pivots = None
                    if mutation == "missing_index": m.kv_swa_full_index = None
                    if mutation == "wrong_history": m.total_k_swa = 256
                    if mutation == "wrong_shape": m.kv_swa_full_index = ((0,),) * 10
                    if mutation in ("index_only_masked", "boolean_index"):
                        rows = [list(row) for row in m.kv_swa_full_index]
                        rows[0][0] = -1 if mutation == "index_only_masked" else True
                        m.kv_swa_full_index = rows
                    if mutation == "zero_state":
                        rows = list(m.kv_swa_full_index_pivots)
                        first = rows[0]
                        rows[0] = (first[0] + 1, first[1] - 1, first[2], first[3])
                        m.kv_swa_full_index_pivots = rows
                    call(session, modules[3], 130, 140, meta=m)
            self.assertEqual(len(modules[3].calls), 1)
            self.assertEqual(session.state, "ABORTED")

    def test_duplicate_or_gapped_chunks_fail(self):
        for start, end in ((0, 2), (4, 8)):
            session, modules = fixture()
            with self.assertRaises(ValidationError):
                with session:
                    call(session, modules[3], 0, 3)
                    call(session, modules[3], start, end)
            self.assertEqual(len(modules[3].calls), 1)

    def test_existing_instance_hook_is_not_overwritten(self):
        session, modules = fixture()
        hook = lambda *args: None
        modules[3]._prepare_swa_kv = hook
        with self.assertRaises(ValidationError):
            with session:
                pass
        self.assertIs(modules[3]._prepare_swa_kv, hook)

    def test_foreign_metadata_hook_is_rejected(self):
        session, modules = fixture()
        with self.assertRaises(ValidationError):
            with session:
                call(session, modules[3], 0, 8, meta=metadata(session, 0, 8, redknot_binding=object()))
        self.assertEqual(modules[3].calls, [])

    def test_layer_source_and_yoco_flags_rejected(self):
        for flag in ("is_kv_source", "is_index_source", "is_index_k_source", "is_yoco_hidden_consumer_layer", "yoco_swa_share_hidden"):
            session, modules = fixture()
            setattr(modules[3], flag, True)
            with self.assertRaises(ValidationError):
                with session: pass
            self.assertNotIn("_prepare_swa_kv", vars(modules[3]))
        with self.assertRaises(ValidationError):
            fixture(layers=(2,))

    def test_duplicate_attention_object_rejected(self):
        session, modules = fixture(layers=(3, 4))
        with self.assertRaises(ValidationError):
            DeepInferCaptureSession({3: modules[3], 4: modules[3]}, request_id="r", phase="p",
                 input_ids=session.input_ids, identities=session.identities, document_start=2)

    def test_document_tokens_and_identity_are_checked(self):
        session, modules = fixture()
        for identity in (replace(session.identities[3], tokens_sha256="e" * 64),
                         replace(session.identities[3], layer=4)):
            with self.assertRaises(ValidationError):
                DeepInferCaptureSession(modules, request_id="r", phase="p", input_ids=session.input_ids,
                                         identities={3: identity}, document_start=2)

    def test_readonly_export_and_single_use(self):
        session, modules = fixture()
        with session: call(session, modules[3], 0, 8)
        with self.assertRaises(TypeError): session.rows_by_layer[3] = ()
        with self.assertRaises(TypeError): session.identities[3] = None
        with self.assertRaises(TypeError): session.attentions[3] = None
        metadata_copy = session.capture_provenance
        metadata_copy["chunks"][0]["start"] = 999
        self.assertEqual(session.capture_provenance["chunks"][0]["start"], 0)
        with self.assertRaises(ValidationError):
            with session: pass

    def test_nested_session_cannot_steal_hooks(self):
        session, modules = fixture()
        other = DeepInferCaptureSession(modules, request_id="r2", phase="p", input_ids=session.input_ids,
                                       identities=session.identities, document_start=2)
        with session:
            wrapper = modules[3]._prepare_swa_kv
            with self.assertRaises(ValidationError):
                with other: pass
            self.assertIs(modules[3]._prepare_swa_kv, wrapper)
            call(session, modules[3], 0, 8)

    def test_wrong_thread_abort(self):
        session, modules = fixture()
        errors = []
        def foreign():
            try: call(session, modules[3], 0, 8)
            except Exception as exc: errors.append(exc)
        with self.assertRaises(ValidationError):
            with session:
                worker = threading.Thread(target=foreign)
                worker.start(); worker.join()
        self.assertIsInstance(errors[0], ValidationError)
        self.assertEqual(modules[3].calls, [])

    def test_restore_does_not_overwrite_foreign_replacement(self):
        session, modules = fixture()
        foreign = lambda *args: None
        with self.assertRaisesRegex(ValidationError, "ownership"):
            with session:
                call(session, modules[3], 0, 8)
                modules[3]._prepare_swa_kv = foreign
        self.assertIs(modules[3]._prepare_swa_kv, foreign)
        self.assertEqual(session.state, "ABORTED")

    def test_nonfinite_cpu_capture_aborts(self):
        session, modules = fixture()
        with self.assertRaises(ValidationError):
            with session:
                modules[3]._prepare_swa_kv(((float("nan"), 2.),) * 8, None, metadata(session, 0, 8), False)
        self.assertEqual(modules[3].calls, [])

    def test_retained_budget_requires_streaming(self):
        with self.assertRaisesRegex(ValidationError, "budget"):
            fixture(max_retained_values=1)

    def test_streaming_sink_receives_only_document_rows_after_native_success(self):
        chunks = []
        session, modules = fixture(sink=lambda layer, start, rows: chunks.append((layer, start, rows)),
                                   max_retained_values=1)
        with session:
            call(session, modules[3], 0, 3)
            call(session, modules[3], 3, 8)
        self.assertEqual([(l, s, len(rows)) for l, s, rows in chunks], [(3, 0, 1), (3, 1, 3)])
        self.assertEqual(session._rows[3], [])
        self.assertTrue(session.capture_provenance["streaming"])
        with self.assertRaisesRegex(ValidationError, "streaming"):
            _ = session.rows_by_layer

    def test_sink_error_aborts_without_success_provenance(self):
        def broken(*args): raise OSError("fixture full disk")
        session, modules = fixture(sink=broken)
        with self.assertRaises(OSError):
            with session: call(session, modules[3], 0, 8)
        self.assertEqual(len(modules[3].calls), 1)
        with self.assertRaises(ValidationError): _ = session.capture_provenance

    def test_completion_barrier_runs_after_driver_not_after_each_layer(self):
        class Ops(CPUReferenceCaptureOps):
            def finish(self): events.append("finish")
        events = []
        session, modules = fixture(layers=(3, 4), ops=Ops())
        with session:
            for module in modules.values(): call(session, module, 0, 8)
            events.append("driver-tail")
            self.assertEqual(events, ["driver-tail"])
        self.assertEqual(events, ["driver-tail", "finish"])
        self.assertEqual(session.state, "COMPLETE")

    def test_completion_barrier_failure_aborts_and_restores(self):
        class Ops(CPUReferenceCaptureOps):
            def finish(self): raise RuntimeError("asynchronous driver failure")
        session, modules = fixture(ops=Ops())
        with self.assertRaisesRegex(RuntimeError, "asynchronous"):
            with session: call(session, modules[3], 0, 8)
        self.assertEqual(session.state, "ABORTED")
        self.assertNotIn("_prepare_swa_kv", vars(modules[3]))
        with self.assertRaises(ValidationError): _ = session.capture_provenance

    def test_native_completion_barrier_visits_captured_devices_without_GPU(self):
        calls = []
        ops = NativeCaptureOps(torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=calls.append)),
                               fused_transform=lambda *a, **k: None, norm_args=lambda *a: None)
        ops.finish()
        self.assertEqual(calls, [])
        ops._native_factory = True
        ops._devices.update(("cuda:1", "cuda:0"))
        ops.finish()
        self.assertEqual(calls, ["cuda:0", "cuda:1"])

    def test_chunk_budget_checked_before_normalization_or_native(self):
        session, modules = fixture(max_chunk_values=1)
        with self.assertRaises(ValidationError):
            with session: call(session, modules[3], 0, 8)
        self.assertEqual(modules[3].calls, [])

    def test_raw_shape_checked_before_normalization(self):
        class Ops(CPUReferenceCaptureOps):
            def normalize(self, *args): raise AssertionError("must check raw shape first")
        session, modules = fixture(ops=Ops())
        with self.assertRaisesRegex(ValidationError, "raw projection shape"):
            with session:
                modules[3]._prepare_swa_kv(((1., 2.),) * 16, None, metadata(session, 0, 8), False)
        self.assertEqual(modules[3].calls, [])

    def test_missing_norm_weight_rejected(self):
        session, modules = fixture()
        modules[3].kv_swa_norm.weight = None
        with self.assertRaises(ValidationError):
            with session: call(session, modules[3], 0, 8)
        self.assertEqual(modules[3].calls, [])

    def test_native_factory_requires_optin_before_heavy_import(self):
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"torch", "deepinfer"}:
                    raise AssertionError("unexpected heavy import")
        blocker = Block()
        sys.meta_path.insert(0, blocker)
        try:
            with self.assertRaises(ValidationError): NativeCaptureOps.from_deepinfer()
        finally:
            sys.meta_path.remove(blocker)


if __name__ == "__main__":
    unittest.main()
