"""CPU tensor tests; no native/GPU kernel execution or model-quality claims."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kv_cache_engram.core import ValidationError
from kv_cache_engram.deepinfer_adapter import NativeCaptureOps, TorchCPUCaptureOps
from test_deepinfer_adapter import FixtureAttention, fixture, metadata


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch CPU runtime not installed")
class TorchCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def module(self):
        module = FixtureAttention()
        module.kv_swa_norm.weight = self.torch.tensor([2., 3.], dtype=self.torch.bfloat16, device="cpu")
        return module

    def test_torch_cpu_normalization_matches_cpu_formula(self):
        torch = self.torch
        ops = TorchCPUCaptureOps()
        raw = torch.tensor([[3., 4.], [-2., 1.]], dtype=torch.bfloat16, device="cpu")
        module = self.module()
        before = raw.clone()
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA query")), \
             patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA init")):
            rows = ops.normalize(module, raw)
        expected = raw.float() / torch.sqrt(raw.float().square().mean(-1, keepdim=True) + 1e-6) * module.kv_swa_norm.weight.float()
        torch.testing.assert_close(torch.tensor(rows), expected)
        self.assertTrue(torch.equal(raw, before))

    def test_cpu_tensor_full_capture(self):
        torch = self.torch
        session, modules = fixture(ops=TorchCPUCaptureOps())
        modules[3].kv_swa_norm.weight = torch.ones(2, device="cpu")
        with session:
            m = metadata(session, 0, 8)
            m.positions = torch.arange(8, device="cpu")
            m.cu_seqlens_q = torch.tensor([0, 8], device="cpu")
            m.kv_swa_full_index_pivots = torch.tensor(m.kv_swa_full_index_pivots, device="cpu", dtype=torch.int32)
            m.kv_swa_full_index = torch.tensor(m.kv_swa_full_index, device="cpu", dtype=torch.int32)
            modules[3]._prepare_swa_kv(torch.arange(16, device="cpu", dtype=torch.float32).reshape(8, 2), None, m, False)
        self.assertEqual(len(session.rows_by_layer[3]), 4)
        self.assertEqual(session.capture_provenance["operator"], "torch-cpu-float32-rmsnorm-reference")
        self.assertFalse(session.capture_provenance["native_operator"])

    def test_rejects_nonfinite_shape_and_dtype(self):
        torch = self.torch
        ops, module = TorchCPUCaptureOps(), self.module()
        for raw in (torch.tensor([[float("inf"), 2.]], device="cpu"),
                    torch.ones(2, device="cpu"), torch.ones((1, 3), device="cpu"),
                    torch.ones((1, 2), device="cpu", dtype=torch.int64)):
            with self.assertRaises(ValidationError): ops.normalize(module, raw)
        with self.assertRaises(ValidationError): ops.integers(torch.tensor([0., 1.], device="cpu"))
        with self.assertRaises(ValidationError): ops.integers(torch.tensor([True, False], device="cpu"))
        with self.assertRaises(ValidationError): ops.integer_matrix_shape(torch.zeros((2, 4), device="cpu"))
        with self.assertRaises(ValidationError): ops.integer_matrix_shape(torch.zeros((2, 4), device="cpu", dtype=torch.bool))

    def test_cpu_ops_reject_meta_device_without_copy(self):
        # A meta tensor has no real backing data and creates no accelerator.
        with self.assertRaisesRegex(ValidationError, "reject accelerator"):
            TorchCPUCaptureOps().normalize(self.module(), self.torch.empty((2, 2), device="meta"))

    def test_native_operation_dependency_fixture_uses_norm_only(self):
        """A fake native function checks plumbing only, NOT the native kernel."""
        torch, calls = self.torch, []
        def fake_transform(raw, **kwargs):
            calls.append(kwargs)
            args = kwargs["norm_args"]
            return (raw.float() * torch.rsqrt(raw.float().square().mean(-1, keepdim=True) + args.eps)
                    * args.weight.float()).to(torch.bfloat16)
        def norm_args(weight, eps): return SimpleNamespace(weight=weight, eps=eps)
        ops = NativeCaptureOps(torch_module=torch, fused_transform=fake_transform, norm_args=norm_args)
        self.assertEqual(ops.label, "deepinfer-norm-dependency-fixture")
        self.assertFalse(ops.native_operator)
        self.assertFalse(ops.accelerator_capture)
        rows = ops.normalize(self.module(), torch.ones((3, 2), device="cpu", dtype=torch.bfloat16))
        self.assertEqual(len(rows), 3)
        self.assertIsNone(calls[0]["rope_args"])
        self.assertIsNone(calls[0]["cast_args"])
        self.assertIsNone(calls[0]["out_args"])
        self.assertEqual(ops.integers(torch.tensor([0, 1], device="cpu")), (0, 1))
        session, modules = fixture(ops=ops)
        modules[3].kv_swa_norm.weight = torch.ones(2, device="cpu", dtype=torch.bfloat16)
        with session:
            m = metadata(session, 0, 8)
            m.positions = torch.arange(8, device="cpu")
            m.cu_seqlens_q = torch.tensor([0, 8], device="cpu")
            m.kv_swa_full_index_pivots = torch.tensor(m.kv_swa_full_index_pivots, device="cpu", dtype=torch.int32)
            m.kv_swa_full_index = torch.tensor(m.kv_swa_full_index, device="cpu", dtype=torch.int32)
            result = modules[3]._prepare_swa_kv(torch.ones((8, 2), device="cpu", dtype=torch.bfloat16), None, m, False)
            self.assertIs(result, modules[3].last_result)
        self.assertEqual(len(session.rows_by_layer[3]), 4)
        self.assertEqual(session.capture_provenance["operator"], "deepinfer-norm-dependency-fixture")
        self.assertFalse(session.capture_provenance["native_operator"])
        self.assertFalse(session.capture_provenance["accelerator_capture"])

    def test_native_factory_mode_rejects_cpu_tensors_before_operator(self):
        ops = NativeCaptureOps(torch_module=self.torch,
                               fused_transform=lambda *a, **kw: self.fail("must reject before operator"),
                               norm_args=lambda *a: None)
        # Exercise the factory's device contract without importing DeepInfer or
        # creating a CUDA context. This is not a real native-kernel invocation.
        ops._native_factory = True
        with self.assertRaisesRegex(ValidationError, "requires CUDA"):
            ops.normalize(self.module(), self.torch.ones((2, 2), dtype=self.torch.bfloat16, device="cpu"))

    def test_dependency_fixture_rejects_non_CPU_metadata_before_copy(self):
        torch = self.torch
        ops = NativeCaptureOps(torch_module=torch, fused_transform=lambda *a, **kw: None,
                               norm_args=lambda *a: None)
        for action in (
            lambda: ops.integers(torch.empty(2, dtype=torch.int64, device="meta")),
            lambda: ops.integer_matrix_shape(torch.empty((2, 4), dtype=torch.int32, device="meta")),
            lambda: ops.integer_matrix_rows(torch.empty((2, 4), dtype=torch.int32, device="meta"), 0, 2),
            lambda: ops.row_shape(torch.empty((2, 2), dtype=torch.bfloat16, device="meta")),
        ):
            with self.assertRaisesRegex(ValidationError, "reject non-CPU"):
                action()

    def test_native_output_contract_mismatch_rejected(self):
        torch = self.torch
        def norm_args(weight, eps): return (weight, eps)
        for output in (None, torch.ones((1, 2), device="cpu"),
                       torch.ones((1, 3), device="cpu", dtype=torch.bfloat16),
                       torch.full((1, 2), float("nan"), device="cpu", dtype=torch.bfloat16)):
            ops = NativeCaptureOps(torch_module=torch, fused_transform=lambda *a, **kw: output, norm_args=norm_args)
            with self.assertRaises(ValidationError):
                ops.normalize(self.module(), torch.ones((1, 2), device="cpu", dtype=torch.bfloat16))

    def test_norm_overflow_rejected(self):
        with self.assertRaisesRegex(ValidationError, "overflow"):
            TorchCPUCaptureOps().normalize(self.module(), self.torch.tensor([[1e30, 1e30]], device="cpu"))


if __name__ == "__main__":
    unittest.main()
