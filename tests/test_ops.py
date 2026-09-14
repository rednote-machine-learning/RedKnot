"""CPU mathematical oracles for head routing, static RoPE and blocked GQA."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from vllm_redknot.ops import (
    causal_attention_reference,
    head_partition,
    merge_heads,
    relocate_rope,
)


class HeadPartitionTests(unittest.TestCase):
    def test_noncontiguous_gqa_partition_preserves_group_mapping(self):
        self.assertEqual(
            head_partition(12, 4, [3, 1]),
            ((9, 10, 11, 3, 4, 5), (0, 1, 2, 6, 7, 8), (0, 2)),
        )

    def test_invalid_groups_and_duplicate_ids_are_rejected(self):
        for args in ((7, 2, [0]), (4, 2, [1, 1]), (4, 2, [-1]), (4, 2, [True])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                head_partition(*args)


@unittest.skipIf(torch is None, "Torch is required for numerical CPU oracles")
class TorchOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(918)

    def assertClose(self, actual, expected, atol=1e-11, rtol=1e-11):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_head_scatter_restores_order_including_empty_partition(self):
        original = torch.randn(5, 6, 4, dtype=torch.float64)
        for local_ids in ([4, 1, 5], [], list(range(6))):
            global_ids = [head for head in range(6) if head not in local_ids]
            result = merge_heads(
                original[:, local_ids],
                original[:, global_ids],
                local_ids,
                global_ids,
                6,
            )
            self.assertClose(result, original)
        with self.assertRaises(ValueError):
            merge_heads(original[:, [0]], original[:, [0]], [0], [0], 6)
        with self.assertRaises(ValueError):
            merge_heads(original[:, [0]], original[:, [1]], [0], [1], 6)

    def _rotate_oracle(self, keys, positions, rotary_dim, theta, interleaved, scale):
        frequencies = theta ** (
            -torch.arange(0, rotary_dim, 2, dtype=keys.dtype) / rotary_dim
        )
        phase = positions.to(keys.dtype)[:, None, None] / scale * frequencies
        result = keys.clone()
        for pair in range(rotary_dim // 2):
            a = 2 * pair if interleaved else pair
            b = a + 1 if interleaved else pair + rotary_dim // 2
            cos, sin = phase[..., pair].cos(), phase[..., pair].sin()
            result[..., a] = keys[..., a] * cos - keys[..., b] * sin
            result[..., b] = keys[..., b] * cos + keys[..., a] * sin
        return result

    def test_rope_relocation_matches_fresh_rotation_and_preserves_magnitude(self):
        raw = torch.randn(7, 3, 12, dtype=torch.float64) * 1.37
        source = torch.arange(7, dtype=torch.int64) + 31
        target = torch.arange(7, dtype=torch.int64) + 109
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                cached = self._rotate_oracle(
                    raw, source, 8, 500_000.0, interleaved, 2.0
                )
                original = cached.clone()
                moved = relocate_rope(
                    cached, source, target, 8, 500_000.0, interleaved, 2.0
                )
                expected = self._rotate_oracle(
                    raw, target, 8, 500_000.0, interleaved, 2.0
                )
                self.assertClose(moved, expected)
                self.assertClose(moved.norm(dim=-1), cached.norm(dim=-1))
                self.assertTrue(torch.equal(moved[..., 8:], raw[..., 8:]))
                self.assertTrue(torch.equal(cached, original))
                self.assertNotEqual(moved.data_ptr(), cached.data_ptr())
                self.assertClose(
                    relocate_rope(
                        moved, target, source, 8, 500_000.0, interleaved, 2.0
                    ),
                    cached,
                )

    def test_rope_rejects_invalid_position_and_frequency_contract(self):
        keys = torch.zeros(2, 1, 8)
        positions = torch.arange(2)
        for kwargs in (
            {"rotary_dim": 3, "theta": 10000.0},
            {"rotary_dim": 10, "theta": 10000.0},
            {"rotary_dim": 8, "theta": float("nan")},
            {"rotary_dim": 8, "theta": 10000.0, "position_scale": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                relocate_rope(keys, positions, positions, **kwargs)
        with self.assertRaises(ValueError):
            relocate_rope(keys, positions.float(), positions, 8, 10000.0)

    def _attention_oracle(self, q, k, v, query_positions, key_positions, scale):
        keys = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        values = v.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        logits = torch.einsum("qhd,khd->hqk", q, keys) * scale
        visible = key_positions[None, :] <= query_positions[:, None]
        weights = torch.softmax(logits.masked_fill(~visible[None], -torch.inf), -1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.einsum("hqk,khd->qhd", weights, values)

    def test_streaming_scaled_gqa_matches_dense_for_unordered_positions(self):
        q = torch.randn(9, 6, 8, dtype=torch.float64)
        k = torch.randn(11, 2, 8, dtype=torch.float64)
        v = torch.randn(11, 2, 5, dtype=torch.float64)
        qp = torch.tensor([0, 3, 9, 14, 2, 11, 1, 5, 7])
        kp = torch.tensor([8, 2, 6, 1, 10, 13, 4, 9, 3, 7, 12])
        expected = self._attention_oracle(q, k, v, qp, kp, 0.27)
        for query_block, key_block in ((1, 1), (4, 3), (64, 256)):
            result = causal_attention_reference(
                q, k, v, qp, kp, 0.27, query_block, key_block
            )
            self.assertClose(result, expected)
            self.assertTrue(torch.equal(result[0], torch.zeros_like(result[0])))

    def test_two_key_partitions_equal_joint_softmax_with_extreme_logits(self):
        q = torch.randn(3, 4, 4, dtype=torch.float64) * 80
        k = torch.randn(8, 2, 4, dtype=torch.float64) * 80
        v = torch.randn(8, 2, 3, dtype=torch.float64)
        qp, kp = torch.arange(3) + 8, torch.arange(8)
        expected = self._attention_oracle(q, k, v, qp, kp, 0.5)
        result = causal_attention_reference(q, k, v, qp, kp, key_block_size=4)
        self.assertTrue(torch.isfinite(result).all())
        self.assertClose(result, expected)

    def test_empty_kv_and_empty_query_have_defined_output_shapes(self):
        q = torch.zeros(2, 4, 8)
        k, v = torch.zeros(0, 2, 8), torch.zeros(0, 2, 6)
        out = causal_attention_reference(q, k, v, torch.arange(2), torch.arange(0))
        self.assertEqual(out.shape, (2, 4, 6))
        self.assertEqual(out.count_nonzero().item(), 0)
        out = causal_attention_reference(q[:0], k, v, torch.arange(0), torch.arange(0))
        self.assertEqual(out.shape, (0, 4, 6))


if __name__ == "__main__":
    unittest.main()
