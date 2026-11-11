"""Independent CPU contract checks, not native/GPU numerical qualification."""

import ast
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kv_cache_engram.core import ValidationError, token_digest
from kv_cache_engram.ffn_only import CAPTURE_LAYERS, FFN_LAYERS, FFNOnlyController, build_ffn_mask, fingerprint
from kv_cache_engram.native_reuse import NativeReuseOps


class Adapter:
    def __init__(self, *args, **kwargs):
        self.poisoned = self.in_call = self.closed = False
        self.events, self.counts = [], {"full_rows": 0}

    def install(self):
        return self

    def report(self):
        if self.poisoned or self.in_call:
            raise ValidationError("dirty CPU adapter")
        return dict(events=self.events, counts=self.counts)

    def close(self, **kwargs):
        self.closed = True


def controller():
    layers = tuple(SimpleNamespace(layer_idx=i) for i in range(40))
    model = SimpleNamespace(layers=layers, yoco_boundary_layer_idx=20)
    c = FFNOnlyController(layers, model_sha256="a" * 64, execution_sha256="b" * 64, tokenizer_sha256="c" * 64, native_model=model)
    c.original_yoco_boundary, c.adapter, c.state = 20, Adapter(), "IDLE"
    c.t = SimpleNamespace(Tensor=type("UnusedTensor", (), {}))
    return c


def marker(start=0, end=8192):
    return dict(request_id="r", phase="p", start=start, end=end)


def qualification(c, mode="qualified_native"):
    c.reset("r", mode, input_ids=[1] * 16384, phase="p", **(dict(index_device="cpu") if mode == "all_selected" else {}))


class LifecycleTests(unittest.TestCase):
    def test_import_is_cpu_only_even_when_suite_already_imported_torch(self):
        script = "import sys; import kv_cache_engram.ffn_only; assert 'torch' not in sys.modules; assert not any(x.startswith('deepinfer') for x in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_ordered_forty_layers_and_identity_required(self):
        for layers, model_sha in (
            ([SimpleNamespace(layer_idx=0)] * 40, "a" * 64),
            ([SimpleNamespace(layer_idx=i) for i in range(40)], "bad"),
        ):
            with self.assertRaises(ValidationError):
                FFNOnlyController(layers, model_sha256=model_sha, execution_sha256="b" * 64, tokenizer_sha256="c" * 64)

    def test_install_rejects_unowned_native_model_before_native_import(self):
        c = controller()
        c.state, c.native_model = "NEW", None
        with (
            patch.object(NativeReuseOps, "from_deepinfer", side_effect=AssertionError("native import")),
            self.assertRaises(ValidationError),
        ):
            c.install()

    def test_reset_rejects_poisoned_or_reentrant_adapter_without_clearing(self):
        for field in ("poisoned", "in_call"):
            c = controller()
            c.adapter.events.append("keep failure evidence")
            setattr(c.adapter, field, True)
            with self.assertRaises(ValidationError):
                qualification(c)
            self.assertEqual(c.adapter.events, ["keep failure evidence"])

    def test_sensitive_requires_qualification_and_exact_fraction(self):
        for mode, fraction in (("sensitive", 0.15), ("sensitive", 0.2), ("native", 0.15)):
            with self.assertRaises(ValidationError):
                controller().reset("r", mode, input_ids=[1] * 16384, phase="p", fraction=fraction)

    def test_marker_binds_request_phase_and_chunk(self):
        c = controller()
        qualification(c)
        c._marker(marker())
        for changes in (
            dict(start=True),
            dict(request_id="other"),
            dict(phase="other"),
            dict(end=8193),
            dict(end=0),
            dict(start=-1),
            dict(extra=1),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                c._marker(dict(marker(), **changes))

    def test_qualification_cannot_finish_with_missing_layer(self):
        c = controller()
        qualification(c)
        c.coverage = 16384
        c.observed = {(0, 8192, i): {} for i in range(40)}
        with self.assertRaises(ValidationError):
            c.finish()
        self.assertFalse(c.qualified)

    def test_reference_pass_does_not_qualify_sparse_and_roundtrips_json(self):
        c = controller()
        qualification(c)
        c.coverage = 16384
        c.observed = {(s, s + 8192, i): {"output": None, "lazy": {}} for s in (0, 8192) for i in range(40)}
        c.reference = dict(c.observed)
        report = c.finish()
        self.assertEqual(report, json.loads(json.dumps(report)))
        self.assertEqual(report["identity_check_count"], 80)
        self.assertFalse(c.qualified)
        self.assertFalse(report["identity_exact"])

    def test_failed_controller_cannot_reset_into_native_fallback(self):
        c = controller()
        c.state = "FAILED"
        with self.assertRaises(ValidationError):
            qualification(c)

    def test_capture_scope_changes_only_native_boundary_and_close_restores(self):
        c = controller()
        c.native_model.other_config = "unchanged"
        with tempfile.TemporaryDirectory() as directory:
            c.set_cache_output_dir(Path(directory) / "bank")
            c.reset("r", "capture_independent", input_ids=[1] * 32768, phase="p")
            self.assertTrue(c.offline_full_layout)
            self.assertIsNone(c.native_model.yoco_boundary_layer_idx)
            self.assertEqual(c.native_model.other_config, "unchanged")
            self.assertEqual(set(c.capture_coverage), set(range(40)))
            c.close()
            self.assertEqual(c.native_model.yoco_boundary_layer_idx, 20)
            self.assertFalse(c.offline_full_layout)
            c.close()

    def test_capture_finish_failure_restores_before_validation_or_export(self):
        c = controller()
        c.cache_output_dir = Path("/unused-cpu-fixture")
        c.reset("r", "capture_independent", input_ids=[1] * 32768, phase="p")
        c.coverage = 32768
        with self.assertRaisesRegex(ValidationError, "capture incomplete"):
            c.finish()
        self.assertEqual(c.native_model.yoco_boundary_layer_idx, 20)
        self.assertFalse(c.offline_full_layout)

    def test_upper_fused_attention_captures_actual_rows_and_error_restores_boundary(self):
        class Attention:
            layer_idx, fuse_kv_swa, yoco_swa_share_hidden = 39, True, False
            fail = False

            def forward(self, hidden, cache, meta, bf16, cross_layer_runtime=None):
                if self.fail:
                    raise RuntimeError("native CPU fixture failure")
                return self._prepare_swa_kv(hidden, cache, meta, False)

            def _prepare_swa_kv(self, raw, cache, meta, use_cp):
                return "native-prepared", None

            def _project_q_and_swa(self, hidden, shared):
                return hidden, hidden

        c = controller()
        c.cache_output_dir = Path("/unused-cpu-fixture")
        c.reset("r", "capture_independent", input_ids=[1] * 32768, phase="p")
        c.t = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        c.ops = SimpleNamespace(
            PG=SimpleNamespace(context_parallel_group=lambda: SimpleNamespace(size=lambda: 1)), normalize=lambda mod, raw: raw
        )
        c._capture = Mock()
        attn = Attention()
        c._install_attention(attn)
        meta = SimpleNamespace(is_prompt=True, total_q=8192, canonical_request_binding=marker())
        raw = object()
        self.assertEqual(attn.forward(raw, None, meta, raw), ("native-prepared", None))
        c._capture.assert_called_once_with(39, marker(), raw)
        self.assertIsNone(c.native_model.yoco_boundary_layer_idx)
        attn.fail = True
        with self.assertRaisesRegex(RuntimeError, "native CPU fixture failure"):
            attn.forward(raw, None, meta, raw)
        self.assertEqual(c.state, "FAILED")
        self.assertEqual(c.native_model.yoco_boundary_layer_idx, 20)
        self.assertFalse(c.offline_full_layout)
        self.assertIsNone(c.frame)
        c.close()

    def test_all_forty_layer_bank_entries_have_full_independent_coverage(self):
        class Storage:
            shape, dtype = (32768, 512), "bfloat16"
            _version = 0

            def __getitem__(self, key):
                return self

            def copy_(self, value):
                self._version += 1

        c = controller()
        c.t = SimpleNamespace(bfloat16="bfloat16")
        c.ops = SimpleNamespace(allocate=lambda rows, like: Storage())
        c.ids = (1,) * 32768
        key = token_digest(c.ids)
        c.docs = ({"key": key, "start": 0, "end": 32768},)
        c.capture_coverage = {layer_index: 0 for layer_index in CAPTURE_LAYERS}
        for start in range(0, 32768, 8192):
            for layer in CAPTURE_LAYERS:
                c._capture(layer, marker(start, start + 8192), SimpleNamespace(shape=(8192, 512)))
        self.assertEqual(len(c.bank), 40)
        self.assertEqual(set(c.capture_coverage.values()), {32768})
        for layer in range(40):
            self.assertEqual(c._entry(key, layer)["scope"]["layer"], layer)
        c.bank[(key, 39)]["scope"]["layer"] = 38
        with self.assertRaises(ValidationError):
            c._entry(key, 39)

    def test_foreign_boundary_mutation_not_overwritten(self):
        c = controller()
        c.offline_full_layout = True
        c.native_model.yoco_boundary_layer_idx = 19
        with self.assertRaises(ValidationError):
            c.close()
        self.assertEqual(c.native_model.yoco_boundary_layer_idx, 19)

    def test_capture_destination_is_exclusive_and_running_clear_forbidden(self):
        c = controller()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bank"
            c.set_cache_output_dir(target)
            with self.assertRaises(FileExistsError):
                c.set_cache_output_dir(target)
            c.state = "RUNNING"
            with self.assertRaises(ValidationError):
                c.clear_cache()

    def test_forward_planning_has_no_host_tensor_constructor_or_transfer(self):
        for name in ("_layout", "_chunk_plan", "get_ffn_plan", "_selected_metadata"):
            tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(FFNOnlyController, name))))
            forbidden = {"tensor", "as_tensor", "from_numpy", "cpu", "numpy", "tolist"}
            calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertFalse(calls & forbidden, (name, calls & forbidden))

    def test_mask_builder_and_forward_plans_have_no_subscript_scalar_store(self):
        # The old selected[ids] = True was not a tensor() call, so the factory
        # guard alone missed its implicit CUDA scalar/index_put path.
        for function in (build_ffn_mask, FFNOnlyController._chunk_plan):
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            scalar_stores = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (bool, int, float, complex))
                and any(isinstance(target, ast.Subscript) for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
            ]
            self.assertFalse(scalar_stores, function.__name__)
            calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertFalse(calls & {"index_put_", "__setitem__", "allow_sync", "set_sync_debug_mode"}, function.__name__)


TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "Torch CPU runtime unavailable")
class TorchControllerTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.t = torch
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        self.t.set_num_threads(self.threads)

    def small(self, n=20, start=0, docs=((4, 14),)):
        t, c = self.t, controller()
        c.t, c.state, c.mode = t, "RUNNING", "sensitive"
        c.rid, c.phase, c.ids, c.input_sha = "r", "p", (1,) * (start + n), "d" * 64
        c.docs = tuple(dict(start=a, end=b, key="e" * 64) for a, b in docs)
        c.scores, c.plans, c.layouts, c.ffn_layouts = {}, {}, {}, []
        optional = [p - start for p in range(start, start + n) if any(a <= p < b for a, b in docs)]
        mandatory = [p - start for p in range(start, start + n) if not any(a <= p < b for a, b in docs)]
        online = [
            p - start
            for p in range(start, start + n)
            if not any(a <= p < b for a, b in docs) or any(a <= p < min(b, a + 127) for a, b in docs)
        ]
        c.prepared = {
            (start, start + n): dict(
                optional=t.tensor(optional, dtype=t.long),
                mandatory=t.tensor(mandatory, dtype=t.long),
                online=t.tensor(online, dtype=t.long),
                online_mask=t.tensor([i in online for i in range(n)], dtype=t.bool),
            )
        }
        m = marker(start, start + n)
        for layer, values in ((2, t.ones(n, 2)), (3, t.ones(n, 2)), (4, t.ones(n, 2)), (5, t.ones(n, 2))):
            c._score(layer, m, values)
        meta = SimpleNamespace(is_prompt=True, num_reqs=1, total_q=n, positions=t.arange(start, start + n), canonical_request_binding=m)
        return c, meta

    def no_advanced_index_put(self):
        # CPU dispatch evidence, deliberately not a claim of CUDA qualification.
        from torch.utils._python_dispatch import TorchDispatchMode

        class NoAdvancedIndexPut(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if "index_put" in str(func):
                    raise AssertionError("advanced index_put is forbidden in mask construction")
                return func(*args, **(kwargs or {}))

        return NoAdvancedIndexPut()

    def test_mask_dispatch_guard_detects_original_scalar_assignment(self):
        mask = self.t.zeros(5, dtype=self.t.bool)
        ids = self.t.tensor([1, 3], dtype=self.t.long)
        with self.no_advanced_index_put(), self.assertRaisesRegex(AssertionError, "index_put"):
            mask[ids] = True

    def test_device_mask_fill_preserves_empty_sparse_full_and_identity_inputs(self):
        for rows, chosen, all_selected in ((0, [], True), (5, [], False), (5, [1, 3], False), (5, list(range(5)), True)):
            with self.subTest(rows=rows, chosen=chosen, all_selected=all_selected):
                ids = self.t.tensor(chosen, dtype=self.t.long)
                original = ids.clone()
                with self.no_advanced_index_put(), self.t.inference_mode():
                    mask = build_ffn_mask(self.t, rows, ids, all_selected=all_selected)
                self.assertEqual(mask.tolist(), [i in chosen for i in range(rows)])
                self.assertEqual(mask.dtype, self.t.bool)
                self.assertEqual(mask.device, ids.device)
                self.assertEqual(mask.shape, (rows,))
                self.assertEqual(ids.tolist(), original.tolist())

    def test_both_chunk_modes_avoid_scalar_index_put_and_keep_exact_selection(self):
        for mode in ("sensitive", "all_selected"):
            with self.subTest(mode=mode):
                c, meta = self.small()
                c.mode = mode
                with self.no_advanced_index_put(), patch.object(c, "allow_sync", side_effect=AssertionError("mask sync waiver")):
                    plan = c._chunk_plan(meta.canonical_request_binding)
                expected = list(range(20)) if mode == "all_selected" else [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19]
                self.assertEqual(plan["ids"].tolist(), expected)
                self.assertEqual(plan["mask"].tolist(), [i in expected for i in range(20)])
                self.assertEqual(plan["mask"].device, plan["ids"].device)
                self.assertIs(c._chunk_plan(meta.canonical_request_binding), plan)

    def test_adjacent_score_is_per_token_previous_norm_and_mean_three(self):
        c, meta = self.small()
        n = meta.total_q
        c.scores[(0, n)] = {2: self.t.ones(n, 2), 3: self.t.full((n, 2), 2.0), 4: self.t.full((n, 2), 6.0), 5: self.t.full((n, 2), 3.0)}
        p = c._chunk_plan(meta.canonical_request_binding)
        self.assertTrue(self.t.allclose(p["score"], self.t.full((n,), (1 + 2 + 0.5) / 3, dtype=self.t.double)))
        self.assertEqual(p["ids"].tolist(), [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19])
        self.assertNotIn((0, n), c.scores)

    def test_ranking_is_per_chunk_ceil15_and_ties_absolute_ascending(self):
        c, meta = self.small(8192, 8192, ((8192, 16384),))
        plan = c._chunk_plan(meta.canonical_request_binding)
        self.assertEqual(plan["ids"].tolist(), list(range(1229)))
        self.assertEqual(plan["optional_rows"], 8192)

    def test_first_six_ffn_native_and_compact_intersection_not_forced(self):
        c, meta = self.small()
        self.assertIsNone(c.get_ffn_plan(SimpleNamespace(layer_idx=5), meta, None))
        full = c.get_ffn_plan(SimpleNamespace(layer_idx=6), meta, None)
        selected_absolute = set(meta.positions[full.selected_rows].tolist())
        compact = SimpleNamespace(**vars(meta))
        compact.positions = self.t.tensor([2, 6, 15, 18])
        compact.total_q = 4
        plan = c.get_ffn_plan(SimpleNamespace(layer_idx=20), compact, None)
        self.assertEqual(plan.selected_rows.tolist(), [0, 2, 3])
        self.assertEqual(set(compact.positions[plan.selected_rows].tolist()), selected_absolute & {2, 6, 15, 18})
        self.assertIs(plan.current_positions, compact.positions)

    def test_compact_empty_intersection_remains_empty_no_forced_token(self):
        c, meta = self.small()
        meta.positions, meta.total_q = self.t.tensor([7, 8, 9]), 3
        plan = c.get_ffn_plan(SimpleNamespace(layer_idx=39), meta, None)
        self.assertEqual(plan.selected_count, 0)
        self.assertEqual(plan.selected_rows.tolist(), [])

    def test_swa_rows_only_document_prefix127_and_no_chunk_or_tail_guards(self):
        c, meta = self.small(8192, 8192, ((3, 32771),))
        self.assertEqual(c._layout(meta)["online_ids"].tolist(), [])
        c, meta = self.small(8192, 32768, ((3, 32771), (32771, 65539)))
        self.assertEqual(c._layout(meta)["online_ids"].tolist(), list(range(3, 130)))
        plan = c.get_ffn_plan(SimpleNamespace(layer_idx=6), meta, None)
        self.assertIn(0, plan.selected_rows.tolist())
        self.assertNotIn(0, c._layout(meta)["online_ids"].tolist())

    def test_forward_plans_and_metadata_do_not_call_host_tensor_factory(self):
        c, meta = self.small()
        meta.valid_token_mask = self.t.ones(20, dtype=self.t.bool)
        meta.gate_force_random = self.t.arange(20)
        meta.images_seq_mask = None
        with patch.object(self.t, "tensor", side_effect=AssertionError("host construction in forward")):
            plan = c.get_ffn_plan(SimpleNamespace(layer_idx=6), meta, None)
            selected = c._selected_metadata(None, meta, plan)
        self.assertIsNot(selected, meta)
        self.assertEqual(selected.total_q, plan.selected_count)
        self.assertEqual(meta.total_q, 20)
        self.assertEqual(selected.positions.tolist(), meta.positions[plan.selected_rows].tolist())

    def test_score_missing_duplicate_nonfinite_and_layout_drift_fail_closed(self):
        c, meta = self.small()
        with self.assertRaises(ValidationError):
            c._score(5, meta.canonical_request_binding, self.t.ones(20, 2))
        del c.scores[(0, 20)][3]
        with self.assertRaises(ValidationError):
            c._chunk_plan(meta.canonical_request_binding)
        c, meta = self.small()
        c.scores[(0, 20)][5][0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            c._chunk_plan(meta.canonical_request_binding)
        c, meta = self.small()
        c._layout(meta)
        meta.positions = meta.positions.flip(0)
        with self.assertRaises(RuntimeError):
            c._layout(meta)

    def test_inference_bank_is_mutation_tracked_and_alias_write_rejected(self):
        t, c = self.t, controller()
        c.t = t
        c.ops = SimpleNamespace(allocate=lambda rows, like: NativeReuseOps.allocate(SimpleNamespace(torch=t), rows, like))
        c.ids = (1,) * 32768
        key = token_digest(c.ids)
        c.docs = ({"key": key, "start": 0, "end": 32768},)
        c.capture_coverage = {layer_index: 0 for layer_index in CAPTURE_LAYERS}
        with t.inference_mode():
            value = t.ones((8192, 512), dtype=t.bfloat16)
            for start in range(0, 32768, 8192):
                c._capture(39, marker(start, start + 8192), value)
            entry = c._entry(key, 39)
            self.assertFalse(entry["tensor"].is_inference())
            entry["tensor"][0:1].add_(1)
            with self.assertRaises(ValidationError):
                c._entry(key, 39)

    def test_qualification_fingerprint_is_strict_bytes_and_finite(self):
        t = self.t
        with t.inference_mode():
            a = t.tensor([1.0, 2.0], dtype=t.bfloat16)
            fp = fingerprint(a, t)
            self.assertEqual(fp, fingerprint(a.clone(), t))
            self.assertNotEqual(fp, fingerprint(a + 1, t))
            with self.assertRaises(ValidationError):
                fingerprint(t.tensor([float("nan")]), t)

    def test_all_selected_needs_complete_projection_checks_before_admission(self):
        c = controller()
        c.t = self.t
        c.reference = {(s, s + 8192, layer_index): {} for s in (0, 8192) for layer_index in range(40)}
        c.reference_input = token_digest([1] * 16384)
        qualification(c, "all_selected")
        c.coverage = 16384
        c.observed = dict(c.reference)
        events = [
            dict(chunk_start=s, chunk_end=s + 8192, layer=layer_index, full_rows=8192) for s in (0, 8192) for layer_index in FFN_LAYERS
        ]
        c.swa_events, c.ffn_layouts = list(events), list(events)
        c.projection_checks = 3
        c.projection_events = [
            dict(events[0], checks=3, max_absolute_error=0.0, checks_named=["split_q", "split_swa", "packed_gather_swa127"])
        ]
        with self.assertRaisesRegex(ValidationError, "every nonempty"):
            c.finish()
        self.assertFalse(c.qualified)
        c.projection_events = [
            dict(e, checks=3, max_absolute_error=0.0, checks_named=["split_q", "split_swa", "packed_gather_swa127"]) for e in events
        ]
        c.projection_checks = 3 * len(events)
        report = c.finish()
        self.assertTrue(report["identity_exact"])
        self.assertTrue(c.qualified)
        self.assertEqual(report, json.loads(json.dumps(report)))


if __name__ == "__main__":
    unittest.main()
