"""CPU oracles for native-callback, masked-input DSV4 z_off reuse."""

import unittest
from dataclasses import replace

try:
    import torch
except ModuleNotFoundError:
    torch = None

from vllm_redknot.dsv4_projection import (
    CachedContribution,
    capture_local_z,
    merge_cached_z_and_project,
)


@unittest.skipIf(torch is None, "Torch is required for numerical CPU oracles")
class DSV4ProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)
        self.policy = "test:model:rope:tp0:woa-recipe:heads-6-0-3"
        self.heads = (6, 0, 3)

    def _projector(self, *, dtype=None, inverse_rope=False, block_quantize=False):
        dtype = torch.float64 if dtype is None else dtype
        weight = torch.randn(2, 5, 4 * 512, dtype=torch.float64) / 32
        calls = []

        def project(o, positions):
            calls.append((o.detach().clone(), positions.detach().clone()))
            values = o.double().clone()
            if inverse_rope:
                # Position-sensitive GPT-J inverse RoPE on LAST 64 dimensions.
                angle = positions.double()[:, None, None] / 17
                even = values[..., -64::2].clone()
                odd = values[..., -63::2].clone()
                values[..., -64::2] = even * angle.cos() + odd * angle.sin()
                values[..., -63::2] = odd * angle.cos() - even * angle.sin()
            if block_quantize:
                values = self._block_quantize(values)
            return (
                torch.einsum("tgd,grd->tgr", values.reshape(o.shape[0], 2, -1), weight)
                .flatten(1)
                .to(dtype)
            )

        return project, calls

    @staticmethod
    def _block_quantize(values):
        # CPU math oracle for native per-token, per-head 128-dimension blocks.
        # This is not a CUDA kernel or a substitute for a real native FP8 test.
        blocks = values.reshape(values.shape[0], values.shape[1], 4, 128)
        maximum = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        scales = torch.exp2(torch.ceil(torch.log2(maximum / 448)))
        quantized = (blocks / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
        return (quantized.double() * scales).reshape_as(values)

    def test_capture_noncontiguous_heads_owns_cpu_copy_without_mutation(self):
        o = torch.randn(3, 8, 512, dtype=torch.float64)
        positions = torch.tensor([0, 4, 10])
        original = o.clone()
        project, calls = self._projector(inverse_rope=True)
        cached = capture_local_z(
            o, positions, self.heads, project, policy_key=self.policy
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(cached.z_off.device.type, "cpu")
        self.assertEqual(cached.z_off.shape, (3, 10))
        self.assertEqual(cached.nbytes, 3 * 10 * 8 + 3 * 8)
        self.assertEqual(cached.local_head_ids, self.heads)
        self.assertTrue(torch.equal(calls[0][0][:, self.heads], o[:, self.heads]))
        self.assertEqual(calls[0][0][:, [1, 2, 4, 5, 7]].count_nonzero(), 0)
        self.assertTrue(torch.equal(o, original))
        positions.fill_(99)
        self.assertTrue(torch.equal(cached.source_positions, torch.tensor([0, 4, 10])))

    def test_dirty_query_rows_receive_no_cache_and_wo_b_is_called_once(self):
        offline = torch.randn(3, 8, 512, dtype=torch.float64)
        online = torch.randn(6, 8, 512, dtype=torch.float64)
        source_positions = torch.tensor([0, 1, 2])
        positions = torch.arange(40, 46)
        clean = torch.tensor([3, 0, 2])
        cache_rows = torch.tensor([2, 0, 1])
        project, calls = self._projector(inverse_rope=True)
        cached = capture_local_z(
            offline, source_positions, self.heads, project, policy_key=self.policy
        )
        cached_before = cached.z_off.clone()
        # Rows 4 and 5 are query/new, row 1 is a dirty document boundary.
        for row in clean.tolist():
            online[row, self.heads] = 0
        expected = project(online, positions)
        expected[clean] += cached.z_off[cache_rows]
        calls.clear()
        final_calls = []

        def wo_b(z):
            final_calls.append(z.clone())
            return z * 2

        actual = merge_cached_z_and_project(
            online,
            positions,
            [CachedContribution(cached, clean, cache_rows)],
            project,
            wo_b,
            local_head_ids=self.heads,
            policy_key=self.policy,
        )
        torch.testing.assert_close(actual, expected * 2, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0][1], positions))
        self.assertEqual(len(final_calls), 1)
        self.assertTrue(torch.equal(cached.z_off, cached_before))
        # Only current positions reached the online callback. Applying RoPE to
        # z_off again, or passing capture positions online, violates this oracle.
        self.assertFalse(torch.equal(calls[0][1][:3], source_positions))

    def test_same_context_bf16_partial_rounding_is_bounded_not_bitwise_claimed(self):
        o = torch.randn(9, 8, 512, dtype=torch.bfloat16)
        positions = torch.arange(9)
        project, _ = self._projector(dtype=torch.bfloat16, inverse_rope=True)
        cached = capture_local_z(
            o, positions, self.heads, project, policy_key=self.policy
        )
        online = o.clone()
        online[:, self.heads] = 0
        actual = merge_cached_z_and_project(
            online,
            positions,
            [CachedContribution(cached, positions, positions)],
            project,
            lambda z: z,
            local_head_ids=self.heads,
            policy_key=self.policy,
        )
        native = project(o, positions)
        # Each partial rounds to BF16 before addition; one full native GEMM
        # rounds once. An absolute norm bound handles cancellation near zero.
        error = (actual.float() - native.float()).abs().max()
        scale = native.float().abs().max().clamp_min(1)
        self.assertLessEqual(error.item(), (0.025 * scale).item())

    def test_fp8_activation_blocks_preserve_selected_heads_and_partial_sum(self):
        o = torch.randn(5, 8, 512, dtype=torch.bfloat16)
        # Large adjacent-head magnitudes would expose an incorrectly shared
        # cross-head scale, while each 512-wide head owns four whole blocks.
        o[:, 1] *= 100
        o[:, 4] *= 0.001
        masked = torch.zeros_like(o)
        masked[:, self.heads] = o[:, self.heads]
        torch.testing.assert_close(
            self._block_quantize(masked)[:, self.heads],
            self._block_quantize(o)[:, self.heads],
            rtol=0,
            atol=0,
        )
        positions = torch.arange(5)
        project, _ = self._projector(
            dtype=torch.bfloat16, inverse_rope=True, block_quantize=True
        )
        cached = capture_local_z(
            o, positions, self.heads, project, policy_key=self.policy
        )
        online = o.clone()
        online[:, self.heads] = 0
        actual = merge_cached_z_and_project(
            online,
            positions,
            [CachedContribution(cached, positions, positions)],
            project,
            lambda z: z,
            local_head_ids=self.heads,
            policy_key=self.policy,
        )
        native = project(o, positions)
        error = (actual.float() - native.float()).abs().max()
        scale = native.float().abs().max().clamp_min(1)
        self.assertLessEqual(error.item(), (0.025 * scale).item())

    def test_all_dirty_and_all_local_policy(self):
        o = torch.randn(3, 8, 512, dtype=torch.float64)
        positions = torch.arange(3)
        project, _ = self._projector()
        heads = tuple(range(8))
        cached = capture_local_z(o, positions, heads, project, policy_key=self.policy)
        empty = torch.empty(0, dtype=torch.long)
        actual = merge_cached_z_and_project(
            o,
            positions,
            [CachedContribution(cached, empty, empty)],
            project,
            lambda z: z,
            local_head_ids=heads,
            policy_key=self.policy,
        )
        torch.testing.assert_close(actual, project(o, positions))
        actual = merge_cached_z_and_project(
            torch.zeros_like(o),
            positions,
            [CachedContribution(cached, positions, positions)],
            project,
            lambda z: z,
            local_head_ids=heads,
            policy_key=self.policy,
        )
        torch.testing.assert_close(actual, cached.z_off)
        actual = merge_cached_z_and_project(
            o,
            positions,
            [],
            project,
            lambda z: z,
            local_head_ids=heads,
            policy_key=self.policy,
        )
        torch.testing.assert_close(actual, project(o, positions))

    def test_multiple_chunks_use_one_online_projection_and_disjoint_rows(self):
        o = torch.randn(6, 8, 512, dtype=torch.float64)
        positions = torch.arange(6)
        project, calls = self._projector(inverse_rope=True)
        first = capture_local_z(
            o[:2], positions[:2], self.heads, project, policy_key=self.policy
        )
        second = capture_local_z(
            o[2:4], positions[:2], self.heads, project, policy_key=self.policy
        )
        online = o.clone()
        online[:4, self.heads] = 0
        expected = project(online, positions)
        expected[:2] += first.z_off
        expected[2:4] += second.z_off
        contributions = [
            CachedContribution(first, torch.tensor([0, 1]), torch.tensor([0, 1])),
            CachedContribution(second, torch.tensor([2, 3]), torch.tensor([0, 1])),
        ]
        calls.clear()
        wo_b_calls = []

        def wo_b(z):
            wo_b_calls.append(z.shape)
            return z

        actual = merge_cached_z_and_project(
            online,
            positions,
            contributions,
            project,
            wo_b,
            local_head_ids=self.heads,
            policy_key=self.policy,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(wo_b_calls, [torch.Size([6, 10])])
        calls.clear()
        overlapping = replace(contributions[1], clean_rows=torch.tensor([1, 3]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            merge_cached_z_and_project(
                online,
                positions,
                [contributions[0], overlapping],
                project,
                wo_b,
                local_head_ids=self.heads,
                policy_key=self.policy,
            )
        self.assertEqual(calls, [])
        self.assertEqual(len(wo_b_calls), 1)

    def test_policy_head_and_row_contracts_fail_before_final_projection(self):
        o = torch.randn(3, 8, 512, dtype=torch.float64)
        positions = torch.arange(3)
        project, calls = self._projector()
        cached = capture_local_z(
            o, positions, self.heads, project, policy_key=self.policy
        )
        online = torch.zeros_like(o)
        final_calls = []

        def run(**overrides):
            contribution = CachedContribution(
                overrides.pop("cached", cached),
                overrides.pop("clean_rows", torch.tensor([0, 2])),
                overrides.pop("cache_rows", torch.tensor([1, 0])),
            )
            args = dict(
                o=online,
                positions=positions,
                contributions=[contribution],
                native_project_z=project,
                wo_b=lambda z: final_calls.append(z),
                local_head_ids=self.heads,
                policy_key=self.policy,
            )
            args.update(overrides)
            return merge_cached_z_and_project(**args)

        invalid = (
            dict(policy_key="another policy"),
            dict(local_head_ids=(0, 1, 3)),
            dict(local_head_ids=(0, 0, 3)),
            dict(clean_rows=torch.tensor([0, 0])),
            dict(clean_rows=torch.tensor([0, 3])),
            dict(cache_rows=torch.tensor([0, 3])),
            dict(cache_rows=torch.tensor([1])),
            dict(clean_rows=torch.tensor([0.0, 1.0])),
            dict(positions=torch.tensor([0, -1, 2])),
            dict(cached=replace(cached, source_positions=torch.tensor([0]))),
            dict(cached=replace(cached, z_off=cached.z_off[:, :3])),
            dict(native_project_z=lambda o, p: o),
            dict(o=o),  # Clean local slots are nonzero: double addition.
        )
        for override in invalid:
            with self.subTest(override=list(override)), self.assertRaises(ValueError):
                run(**override)
        self.assertEqual(final_calls, [])
        self.assertGreaterEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
