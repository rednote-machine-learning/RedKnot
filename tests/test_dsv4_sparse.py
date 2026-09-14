"""Selection/padding contracts and optional CPU Torch sparse-attention oracle."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

from vllm_redknot.dsv4_sparse import (
    _selection,
    selected_sparse_mla,
    selected_sparse_mla_reference,
    sparse_workload,
)

if importlib.util.find_spec("torch") is not None:
    import torch
else:
    torch = None


class SelectionTests(unittest.TestCase):
    def test_native_oracle_capacity_must_be_128_aligned_before_gpu_import(self):
        script = Path(__file__).parents[1] / "benchmarks/check_dsv4_sparse_kernel.py"
        for capacity in (64, 192):
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run",
                    "--device",
                    "cuda:0",
                    "--candidates",
                    str(capacity),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("128", result.stderr)
            self.assertNotIn("No module named", result.stderr)

    def test_benchmark_requires_explicit_device_and_run_before_importing_torch(self):
        script = Path(__file__).parents[1] / "benchmarks/check_dsv4_sparse_kernel.py"
        result = subprocess.run(
            [sys.executable, str(script), "--device", "cuda:0"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --run", result.stderr)
        self.assertNotIn("No module named", result.stderr)

    def test_import_has_no_torch_triton_or_cuda_side_effect(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import vllm_redknot.dsv4_sparse; "
                    "assert not {'torch', 'triton', 'vllm'} & sys.modules.keys()"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_arbitrary_original_head_order_is_not_sorted(self):
        self.assertEqual(_selection([63, 0, 17], 64, "heads"), (63, 0, 17))

    def test_invalid_selections_fail_before_kernel(self):
        for selection in ([1, 1], [64], [-1], [True], [1.0]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                _selection(selection, 64, "heads")
        with self.assertRaises(TypeError):
            _selection("01", 64, "heads")

    def test_true_head_padding_is_sixteen_not_native_sixty_four(self):
        report = sparse_workload(3, 17, 65)
        self.assertEqual(report["logical_query_head_rows"], 51)
        self.assertEqual(report["padded_query_head_rows"], 96)
        self.assertEqual(report["padded_heads_per_row"], 32)
        self.assertEqual(report["programs"], 6)
        self.assertEqual(report["padded_candidate_capacity"], 128)
        self.assertEqual(sparse_workload(3, 1, 64)["padded_heads_per_row"], 16)

    def test_empty_selection_launches_no_programs(self):
        self.assertEqual(sparse_workload(0, 63, 512)["programs"], 0)
        self.assertEqual(sparse_workload(4, 0, 512)["programs"], 0)

    def test_workload_rejects_invalid_counts(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                sparse_workload(1, value, 64)


@unittest.skipUnless(torch is not None, "CPU Torch is required for numerical tests")
class ReferenceTests(unittest.TestCase):
    def test_native_oracle_normalization_is_readonly_and_keeps_valid_duplicates(self):
        from benchmarks.check_dsv4_sparse_kernel import _native_oracle_indices

        indices = torch.tensor([[-2, -1, 0, 2, 2, 3, 99]])
        before = indices.clone()
        actual = _native_oracle_indices(indices, 3)
        torch.testing.assert_close(actual, torch.tensor([[-1, -1, 0, 2, 2, -1, -1]]))
        torch.testing.assert_close(indices, before)

    def tensors(self):
        q = torch.zeros(3, 4, 512, dtype=torch.float64)
        kv = torch.zeros(3, 1, 512, dtype=torch.float64)
        kv[:, 0, 0] = torch.tensor([2, 4, 8])
        indices = torch.tensor([[0, 1, 2, -1], [2, 0, 0, 1], [9, -1, 1, 2]])
        lens = torch.tensor([2, 3, 4])
        sink = torch.tensor([0, -float("inf"), float("inf"), 0], dtype=torch.float64)
        return q, kv, indices, lens, sink

    def test_sink_and_duplicate_candidates_keep_native_semantics(self):
        args = self.tensors()
        out = selected_sparse_mla_reference(*args, [1, 0, 2], [2, 1, 0])
        expected = torch.tensor([[0, 4, 3], [0, 3, 2], [0, 6, 4]], dtype=out.dtype)
        torch.testing.assert_close(out[:, :, 0], expected)
        self.assertEqual(out[:, :, 1:].count_nonzero().item(), 0)

    def test_all_invalid_candidates_and_infinite_sinks_return_zero(self):
        q, kv, indices, lens, sink = self.tensors()
        indices.fill_(-1)
        out = selected_sparse_mla_reference(q, kv, indices, lens, sink, [0], [0, 1, 2])
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.count_nonzero().item(), 0)

    def test_lengths_mask_trailing_candidates(self):
        q, kv, indices, lens, sink = self.tensors()
        lens[:] = torch.tensor([0, 0, 3])
        out = selected_sparse_mla_reference(q, kv, indices, lens, sink, [0, 1, 2], [1])
        torch.testing.assert_close(out[:, 0, 0], torch.tensor([0.0, 0.0, 4.0]).double())

    def test_invalid_selected_lengths_fail_closed(self):
        q, kv, indices, lens, sink = self.tensors()
        for bad_length in (-1, 5):
            lens[0] = bad_length
            with self.assertRaisesRegex(ValueError, "lens must be in"):
                selected_sparse_mla_reference(q, kv, indices, lens, sink, [0], [1])
        # Other rows are genuinely not processed and do not constrain this call.
        selected_sparse_mla_reference(q, kv, indices, lens, sink, [1], [1])

    def test_strided_inputs_and_three_dimensional_candidates(self):
        args = self.tensors()
        q, kv, indices, lens, sink = args
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        kv = kv.expand(-1, 2, -1)[:, :1]
        indices = indices.T.contiguous().T[:, None, :]
        out = selected_sparse_mla_reference(q, kv, indices, lens, sink, [2, 0], [3, 1])
        expected = selected_sparse_mla_reference(*args, [2, 0], [3, 1])
        torch.testing.assert_close(out, expected)

    def test_no_causal_mask_is_invented_from_gathered_indices(self):
        q, kv, indices, lens, sink = self.tensors()
        indices[0] = 2
        lens[0] = 1
        out = selected_sparse_mla_reference(q, kv, indices, lens, sink, [0], [1])
        self.assertEqual(out[0, 0, 0].item(), 8)

    def test_empty_selection_shapes_and_input_readonly(self):
        args = self.tensors()
        copies = [value.clone() for value in args]
        out = selected_sparse_mla_reference(*args, [], [1])
        self.assertEqual(out.shape, (0, 1, 512))
        out = selected_sparse_mla_reference(*args, [2], [])
        self.assertEqual(out.shape, (1, 0, 512))
        for original, copied in zip(args, copies, strict=True):
            torch.testing.assert_close(original, copied)

    def test_extreme_finite_sink_is_stable(self):
        q, kv, indices, lens, sink = self.tensors()
        sink[0] = 1e30
        out = selected_sparse_mla_reference(q, kv, indices, lens, sink, [0], [0])
        self.assertEqual(out.count_nonzero().item(), 0)

    def test_selected_dot_products_match_manual_softmax(self):
        args = self.tensors()
        args[0][1, 3, 0] = 2
        out = selected_sparse_mla_reference(*args, [1], [3], scale=0.5)
        logits = torch.tensor([8.0, 2.0, 2.0, 0.0], dtype=torch.float64)
        expected = (logits.softmax(0) * torch.tensor([8.0, 2.0, 2.0, 0.0])).sum()
        torch.testing.assert_close(out[0, 0, 0], expected)

    def test_gpu_entrypoint_does_not_silently_run_cpu_reference(self):
        with self.assertRaisesRegex(ValueError, "CUDA BF16/FP16"):
            selected_sparse_mla(*self.tensors(), [0], [1])

    def test_input_geometry_dtype_and_scale_fail_closed(self):
        valid = self.tensors()
        for index, replacement in (
            (0, torch.zeros(3, 4, 64)),
            (1, torch.zeros(3, 2, 512, dtype=torch.float64)),
            (2, valid[2].float()),
            (3, valid[3][:1]),
            (4, valid[4][:1]),
        ):
            args = list(valid)
            args[index] = replacement
            with self.subTest(index=index), self.assertRaises(ValueError):
                selected_sparse_mla_reference(*args, [0], [1])
        for scale in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                selected_sparse_mla_reference(*valid, [0], [1], scale)


if __name__ == "__main__":
    unittest.main()
