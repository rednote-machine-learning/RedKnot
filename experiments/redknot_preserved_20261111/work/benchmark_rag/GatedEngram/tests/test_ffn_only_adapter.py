"""CPU contracts only: no model-quality, GPU-kernel or EP qualification claim."""

import copy
import subprocess
import sys
import unittest
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

from kv_cache_engram.ffn_only_adapter import (
    FFNOnlyAdapter,
    TorchFFNOnlyOps,
    _snapshot,
    _unchanged,
    make_selected_mlp_metadata,
)

try:
    import torch
except ImportError:
    torch = None


class Rows:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values), 1)


class Ops:
    native_execution = False

    def check_runtime(self):
        pass

    @staticmethod
    def rows(x):
        return (x[0] if isinstance(x, tuple) else x).shape[0]

    def validate_full(self, norm, bf16, rows):
        if self.rows(norm) != rows or self.rows(bf16) != rows:
            raise ValueError("wrong input row count")
        if isinstance(norm, tuple) and self.rows(norm[1]) != rows:
            raise ValueError("wrong SF row count")

    def validate_positions(self, positions, reference, rows):
        if not isinstance(positions, list) or len(positions) != rows:
            raise ValueError("wrong explicit current positions")

    def validate_plan(self, plan, reference):
        ids = plan.selected_rows
        if (
            len(ids) != plan.selected_count
            or plan.full_rows != self.rows(reference)
            or ids != sorted(set(ids))
            or any(i < 0 or i >= plan.full_rows for i in ids)
        ):
            raise ValueError("invalid FFN row IDs")

    def validate_selected_metadata(self, full, active, plan):
        for name in ("positions", "valid_token_mask", "gate_force_random", "images_seq_mask"):
            value = getattr(full, name, None)
            expected = None if value is None else [value[i] for i in plan.selected_rows]
            if getattr(active, name, None) != expected:
                raise ValueError("wrong selected " + name)

    def gather(self, value, ids):
        if isinstance(value, tuple):
            return tuple(self.gather(x, ids) for x in value)
        return Rows(value.values[i] for i in ids)

    def scatter_zero(self, selected, reference, plan):
        if self.rows(selected) != plan.selected_count:
            raise ValueError("invalid selected output")
        result = [0.0] * plan.full_rows
        for i, v in zip(plan.selected_rows, selected.values, strict=False):
            result[i] = v
        return Rows(result)


class MLP:
    use_mega_moe = False

    def __init__(self):
        self.calls, self.fail = [], False

    def forward(self, norm, bf16, metadata):
        self.calls.append((norm, bf16, metadata))
        if self.fail:
            raise RuntimeError("native MLP error")
        return Rows(5 * x for x in bf16.values)


class Layer:
    def __init__(self, index):
        self.layer_idx, self.mlp = index, MLP()
        self.self_attn = NS(forward=lambda *a, **k: "native-attention")

    def forward(self):
        return "unchanged-decoder"


def selected_metadata(layer, metadata, plan):
    result = copy.copy(metadata)
    result.total_q = plan.selected_count
    for name in ("positions", "valid_token_mask", "gate_force_random", "images_seq_mask"):
        value = getattr(metadata, name, None)
        if value is not None:
            setattr(result, name, [value[i] for i in plan.selected_rows])
    return result


def fixture(index=6, positions=None, selected=(0, 2), *, empty_callback=None):
    positions = [8, 9, 10, 11] if positions is None else positions
    rows = len(positions)
    layer = Layer(index)
    metadata = NS(
        is_prompt=True,
        num_reqs=1,
        total_q=rows,
        positions=positions,
        canonical_request_binding=dict(request_id="rid", phase="p", start=8, end=8200),
        valid_token_mask=[True] * rows,
        gate_force_random=[i % 2 == 0 for i in range(rows)],
        images_seq_mask=[False] * rows,
    )
    plan = NS(
        full_rows=rows,
        selected_rows=list(selected),
        selected_count=len(selected),
        current_positions=positions,
        request_id="rid",
        phase="p",
        chunk_start=8,
        chunk_end=8200,
        plan_sha256="a" * 64,
    )
    adapter = FFNOnlyAdapter(
        [layer], lambda layer_index, m, x: plan, selected_metadata, ops=Ops(), allow_empty_native=empty_callback
    ).install()
    return NS(layer=layer, metadata=metadata, plan=plan, adapter=adapter)


class FFNOnlyAdapterTests(unittest.TestCase):
    def run_mlp(self, f, norm=None, bf16=None):
        n = f.metadata.total_q
        bf16 = Rows(range(1, n + 1)) if bf16 is None else bf16
        norm = bf16 if norm is None else norm
        return f.layer.mlp.forward(norm, bf16, f.metadata)

    def test_partial_calls_real_mlp_and_zeros_inactive_outputs(self):
        f = fixture()
        out = self.run_mlp(f)
        self.assertEqual(out.values, [5, 0, 15, 0])
        norm, bf16, active = f.layer.mlp.calls[0]
        self.assertEqual(bf16.values, [1, 3])
        self.assertEqual(active.positions, [8, 10])
        self.assertEqual(active.gate_force_random, [True, True])
        self.assertIsNot(active, f.metadata)
        self.assertEqual(f.metadata.total_q, 4)
        self.assertEqual(f.metadata.positions, [8, 9, 10, 11])
        self.assertEqual(f.adapter.report()["counts"]["inactive_ffn_zero_rows"], 2)
        f.adapter.close()

    def test_all_selected_preserves_exact_native_aliases_and_metadata(self):
        f = fixture(selected=(0, 1, 2, 3))
        norm, bf16 = (Rows([1, 2, 3, 4]), Rows([101, 102, 103, 104])), Rows([1, 2, 3, 4])
        f.adapter.make_selected_metadata = lambda *a: self.fail("must not build active metadata")
        out = self.run_mlp(f, norm, bf16)
        self.assertEqual(out.values, [5, 10, 15, 20])
        self.assertIs(f.layer.mlp.calls[0][0], norm)
        self.assertIs(f.layer.mlp.calls[0][1], bf16)
        self.assertIs(f.layer.mlp.calls[0][2], f.metadata)
        self.assertTrue(f.adapter.report()["events"][0]["all_selected_native"])
        f.adapter.close()

    def test_fp8_data_and_sf_gather_same_ids(self):
        f = fixture()
        self.run_mlp(f, (Rows([1, 2, 3, 4]), Rows([100, 101, 102, 103])))
        norm = f.layer.mlp.calls[0][0]
        self.assertEqual(norm[0].values, [1, 3])
        self.assertEqual(norm[1].values, [100, 102])
        f.adapter.close()

    def test_source_boundary_and_last_layers_have_no_old_eligibility_filter(self):
        for index in (6, 8, 14, 19, 20, 39):
            with self.subTest(layer=index):
                f = fixture(index=index)
                self.assertEqual(self.run_mlp(f).values, [5, 0, 15, 0])
                f.adapter.close()

    def test_compact_positions_not_full_native_chunk_row_numbers(self):
        f = fixture(index=39, positions=[8, 42, 8199], selected=(1,))
        out = self.run_mlp(f)
        self.assertEqual(out.values, [0, 10, 0])
        self.assertEqual(f.layer.mlp.calls[0][2].positions, [42])
        event = f.adapter.report()["events"][0]
        self.assertEqual(event["full_rows"], 3)
        self.assertEqual(event["chunk_end"] - event["chunk_start"], 8192)
        f.adapter.close()

    def test_equal_but_not_identical_current_positions_rejected(self):
        f = fixture()
        f.plan.current_positions = list(f.metadata.positions)
        with self.assertRaisesRegex(ValueError, "actual current"):
            self.run_mlp(f)
        self.assertEqual(f.layer.mlp.calls, [])
        f.adapter.close(allow_incomplete=True)

    def test_compact_wrong_geometry_does_not_silently_fallback(self):
        f = fixture(index=20, positions=[8, 8199], selected=(0,))
        f.plan.full_rows = 8192
        with self.assertRaisesRegex(ValueError, "current row count"):
            self.run_mlp(f)
        self.assertEqual(f.layer.mlp.calls, [])
        f.adapter.close(allow_incomplete=True)

    def test_first_six_plan_rejected_but_none_native_allowed(self):
        f = fixture(index=5)
        with self.assertRaisesRegex(ValueError, "first six"):
            self.run_mlp(f)
        f.adapter.close(allow_incomplete=True)
        f = fixture(index=5)
        f.adapter.get_plan = lambda *a: None
        self.assertEqual(self.run_mlp(f).values, [5, 10, 15, 20])
        f.adapter.close()

    def test_no_plan_native_path_does_not_validate_or_compact(self):
        f = fixture()
        f.adapter.get_plan = lambda *a: None
        f.metadata.canonical_request_binding = None
        self.assertEqual(self.run_mlp(f).values, [5, 10, 15, 20])
        self.assertEqual(f.adapter.report()["counts"]["native_mlp_calls"], 1)
        f.adapter.close()

    def test_decode_never_calls_plan_provider(self):
        f = fixture()
        f.metadata.is_prompt = False
        f.adapter.get_plan = lambda *a: self.fail("decode must remain native")
        self.assertEqual(self.run_mlp(f).values, [5, 10, 15, 20])
        f.adapter.close()

    def test_native_zero_current_rows_still_call_original(self):
        f = fixture(positions=[], selected=())
        f.adapter.get_plan = lambda *a: None
        self.assertEqual(self.run_mlp(f).values, [])
        self.assertEqual(len(f.layer.mlp.calls), 1)
        f.adapter.close()

    def test_zero_selection_default_rejects_not_skips_collective(self):
        f = fixture(selected=())
        with self.assertRaisesRegex(ValueError, "collective cannot be skipped"):
            self.run_mlp(f)
        self.assertEqual(f.layer.mlp.calls, [])
        f.adapter.close(allow_incomplete=True)

    def test_qualified_zero_selection_enters_original_empty_mlp(self):
        calls = []
        f = fixture(selected=(), empty_callback=lambda *a: calls.append(a) or True)
        self.assertEqual(self.run_mlp(f).values, [0, 0, 0, 0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(f.layer.mlp.calls), 1)
        self.assertEqual(f.layer.mlp.calls[0][1].values, [])
        self.assertEqual(f.layer.mlp.calls[0][2].total_q, 0)
        self.assertTrue(f.adapter.report()["events"][0]["empty_native_collective"])
        f.adapter.close()

    def test_truthy_empty_qualification_is_not_sufficient(self):
        f = fixture(selected=(), empty_callback=lambda *a: "yes")
        with self.assertRaises(ValueError):
            self.run_mlp(f)
        f.adapter.close(allow_incomplete=True)

    def test_wrong_routing_mask_rejected_before_native_call(self):
        f = fixture()

        def wrong(layer, meta, plan):
            active = selected_metadata(layer, meta, plan)
            active.gate_force_random = [False, False]
            return active

        f.adapter.make_selected_metadata = wrong
        with self.assertRaisesRegex(ValueError, "gate_force_random"):
            self.run_mlp(f)
        self.assertEqual(f.layer.mlp.calls, [])
        f.adapter.close(allow_incomplete=True)

    def test_metadata_provider_cannot_replace_original_position_field(self):
        f = fixture()

        def wrong(layer, meta, plan):
            active = selected_metadata(layer, meta, plan)
            meta.positions = list(meta.positions)
            return active

        f.adapter.make_selected_metadata = wrong
        with self.assertRaisesRegex(ValueError, "mutated native"):
            self.run_mlp(f)
        self.assertEqual(f.layer.mlp.calls, [])
        f.adapter.close(allow_incomplete=True)

    def test_metadata_provider_cannot_mutate_original_list_in_place(self):
        f = fixture()

        def wrong(layer, meta, plan):
            active = selected_metadata(layer, meta, plan)
            meta.positions[0] = 999
            return active

        f.adapter.make_selected_metadata = wrong
        with self.assertRaisesRegex(ValueError, "mutated native"):
            self.run_mlp(f)
        self.assertEqual(f.layer.mlp.calls, [])
        f.adapter.close(allow_incomplete=True)

    def test_provider_same_metadata_and_foreign_scope_rejected(self):
        for kind in ("same", "scope"):
            with self.subTest(kind=kind):
                f = fixture()

                def wrong(layer, meta, plan):
                    if kind == "same":
                        return meta
                    active = selected_metadata(layer, meta, plan)
                    active.canonical_request_binding = dict(meta.canonical_request_binding, phase="foreign")
                    return active

                f.adapter.make_selected_metadata = wrong
                with self.assertRaises(ValueError):
                    self.run_mlp(f)
                self.assertEqual(f.layer.mlp.calls, [])
                f.adapter.close(allow_incomplete=True)

    def test_duplicate_unsorted_and_out_of_range_ids_rejected(self):
        for selected in ((0, 0), (2, 0), (0, 4), (-1, 2)):
            with self.subTest(selected=selected):
                f = fixture(selected=selected)
                with self.assertRaises(ValueError):
                    self.run_mlp(f)
                self.assertEqual(f.layer.mlp.calls, [])
                f.adapter.close(allow_incomplete=True)

    def test_native_exception_poisons_without_retry_and_restores_methods(self):
        f = fixture()
        f.layer.mlp.fail = True
        with self.assertRaisesRegex(RuntimeError, "native MLP"):
            self.run_mlp(f)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.run_mlp(f)
        self.assertEqual(len(f.layer.mlp.calls), 1)
        with self.assertRaises(ValueError):
            f.adapter.report()
        with self.assertRaises(ValueError):
            f.adapter.close()
        f.adapter.close(allow_incomplete=True)
        self.assertNotIn("forward", f.layer.mlp.__dict__)

    def test_attention_decoder_and_mhc_not_owned_or_changed(self):
        f = fixture()
        attn = f.layer.self_attn.forward
        self.run_mlp(f)
        self.assertIs(f.layer.self_attn.forward, attn)
        self.assertIs(f.layer.forward.__func__, Layer.forward)
        report = f.adapter.report()
        self.assertTrue(report["attention_forward_unchanged"])
        self.assertTrue(report["native_mhc_unchanged"])
        self.assertFalse(report["accuracy_guarantee"])
        self.assertFalse(report["speedup_claim"])
        f.adapter.close()
        f.adapter.close()

    def test_owner_conflict_does_not_replace_existing_wrapper(self):
        f = fixture()
        original = f.layer.mlp.forward
        other = FFNOnlyAdapter([f.layer], lambda *a: None, selected_metadata, ops=Ops())
        with self.assertRaisesRegex(ValueError, "conflicting"):
            other.install()
        self.assertIs(f.layer.mlp.forward, original)
        f.adapter.close()

    def test_restore_existing_instance_forward(self):
        layer = Layer(8)
        custom = MethodType(lambda self, n, b, m: b, layer.mlp)
        layer.mlp.forward = custom
        a = FFNOnlyAdapter([layer], lambda *a: None, selected_metadata, ops=Ops()).install()
        a.close()
        self.assertIs(layer.mlp.forward, custom)

    def test_install_failure_rolls_back_already_installed_layers(self):
        layers = [Layer(6), Layer(7)]

        class Broken(FFNOnlyAdapter):
            def _install_layer(self, layer):
                if layer.layer_idx == 7:
                    raise RuntimeError("install failure")
                super()._install_layer(layer)

        a = Broken(layers, lambda *a: None, selected_metadata, ops=Ops())
        with self.assertRaisesRegex(RuntimeError, "install failure"):
            a.install()
        self.assertTrue(a.closed)
        self.assertNotIn("forward", layers[0].mlp.__dict__)
        self.assertFalse(hasattr(layers[0].mlp, "_ffn_only_owner"))

    def test_close_checks_all_ownership_before_restoring(self):
        layers = [Layer(6), Layer(7)]
        a = FFNOnlyAdapter(layers, lambda *a: None, selected_metadata, ops=Ops()).install()
        owned = [x.mlp.forward for x in layers]
        layers[1].mlp.forward = lambda *args: None
        with self.assertRaisesRegex(ValueError, "ownership"):
            a.close()
        self.assertIs(layers[0].mlp.forward, owned[0])
        layers[1].mlp.forward = owned[1]
        a.close()

    def test_mega_moe_rejected_before_selected_input_call(self):
        f = fixture()
        f.layer.mlp.use_mega_moe = True
        with self.assertRaisesRegex(ValueError, "mega-MoE"):
            self.run_mlp(f)
        self.assertEqual(f.layer.mlp.calls, [])
        f.adapter.close(allow_incomplete=True)

    def test_import_has_no_torch_or_deepinfer_dependency(self):
        root = str(Path(__file__).resolve().parents[1])
        script = f"import sys; sys.path.insert(0, {root!r}); import kv_cache_engram.ffn_only_adapter; assert 'torch' not in sys.modules; assert 'deepinfer' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@unittest.skipIf(torch is None, "Torch not installed; native server CPU suite required")
class TorchFFNOnlyOpsTests(unittest.TestCase):
    def setUp(self):
        self.ops = TorchFFNOnlyOps()

    def test_empty_fp8_gather_has_explicit_padded_sf_stride(self):
        data = torch.zeros((4, 256), dtype=torch.uint8)
        sf = torch.arange(8, dtype=torch.int32).reshape(4, 2)
        ids = torch.empty((0,), dtype=torch.int64)
        x, scale = self.ops.gather((data, sf), ids)
        self.assertEqual(x.shape, (0, 256))
        self.assertEqual(scale.shape, (0, 2))
        self.assertEqual(scale.stride(), (1, 4))
        self.assertEqual(scale.untyped_storage().nbytes(), 2 * 4 * 4)
        self.assertTrue(torch.equal(sf, torch.arange(8, dtype=torch.int32).reshape(4, 2)))

    def test_selected_metadata_helper_gathers_all_gate_fields(self):
        positions = torch.tensor([8, 42, 8199], dtype=torch.int64)
        full = NS(
            total_q=3,
            positions=positions,
            valid_token_mask=torch.tensor([True, False, True]),
            gate_force_random=torch.tensor([False, True, False]),
            images_seq_mask=None,
        )
        plan = NS(full_rows=3, selected_count=2, selected_rows=torch.tensor([0, 2]))
        active = make_selected_mlp_metadata(None, full, plan)
        self.ops.validate_selected_metadata(full, active, plan)
        self.assertTrue(torch.equal(active.positions, torch.tensor([8, 8199])))
        self.assertEqual(active.total_q, 2)
        self.assertEqual(full.total_q, 3)
        self.assertIs(full.positions, positions)

    def test_non_multiple_of_four_packed_gather(self):
        for n in (1, 3, 33, 129, 1845):
            with self.subTest(n=n):
                ids = torch.arange(n, dtype=torch.int64) * 2
                data = torch.zeros((2 * n, 256), dtype=torch.uint8)
                sf = torch.arange(4 * n, dtype=torch.int32).reshape(2 * n, 2)
                x, scale = self.ops.gather((data, sf), ids)
                self.assertEqual(scale.stride(), (1, (n + 3) // 4 * 4))
                self.assertTrue(torch.equal(scale, sf.index_select(0, ids)))
                self.assertNotEqual(scale.data_ptr(), sf.data_ptr())

    def test_wrong_selected_metadata_values_fail(self):
        full = NS(positions=torch.tensor([1, 2]), valid_token_mask=None, gate_force_random=None, images_seq_mask=None)
        plan = NS(full_rows=2, selected_count=1, selected_rows=torch.tensor([1]))
        active = NS(positions=torch.tensor([1]), valid_token_mask=None, gate_force_random=None, images_seq_mask=None)
        with self.assertRaises(RuntimeError):
            self.ops.validate_selected_metadata(full, active, plan)

    def test_inference_mode_metadata_snapshot_does_not_access_version(self):
        with torch.inference_mode():
            metadata = NS(
                is_prompt=True,
                num_reqs=1,
                total_q=3,
                positions=torch.tensor([8, 42, 8199]),
                valid_token_mask=None,
                gate_force_random=torch.tensor([False, True, False]),
                images_seq_mask=None,
                canonical_request_binding=dict(request_id="rid", phase="p", start=8, end=8200),
            )
            before = _snapshot(metadata)
            plan = NS(full_rows=3, selected_count=1, selected_rows=torch.tensor([1]))
            active = make_selected_mlp_metadata(None, metadata, plan)
            _unchanged(metadata, before, self.ops)
            self.ops.validate_selected_metadata(metadata, active, plan)
            self.assertTrue(torch.equal(active.positions, torch.tensor([42])))

    def test_inference_mode_metadata_snapshot_detects_in_place_mutation(self):
        with torch.inference_mode():
            metadata = NS(
                is_prompt=True,
                num_reqs=1,
                total_q=2,
                positions=torch.tensor([8, 42]),
                valid_token_mask=None,
                gate_force_random=None,
                images_seq_mask=None,
                canonical_request_binding=dict(request_id="rid", phase="p", start=8, end=8200),
            )
            before = _snapshot(metadata)
            metadata.positions[0] = 9
            with self.assertRaises(RuntimeError):
                _unchanged(metadata, before, self.ops)


if __name__ == "__main__":
    unittest.main()
