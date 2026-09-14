"""CPU grouped-MLA oracle: head projection and dirty-row replacement."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from vllm_redknot.mla import (
    GroupedMLAProjector,
    merge_mla_contributions,
    project_head_contributions,
)


@unittest.skipIf(torch is None, "Torch is required for numerical CPU oracles")
class MLAProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def _full_projection(self, attn, wo_a):
        return torch.einsum(
            "tgd,grd->tgr", attn.reshape(attn.shape[0], wo_a.shape[0], -1), wo_a
        )

    def test_noncontiguous_heads_sum_to_full_grouped_projection(self):
        attn = torch.randn(7, 8, 4, dtype=torch.float64)
        wo_a = torch.randn(2, 5, 16, dtype=torch.float64)
        local, global_ = [6, 0, 3], [1, 2, 4, 5, 7]
        z_local = project_head_contributions(attn[:, local], wo_a, local, 8)
        z_global = project_head_contributions(attn[:, global_], wo_a, global_, 8)
        self.assertClose(z_local + z_global, self._full_projection(attn, wo_a))
        z_empty = project_head_contributions(attn[:, []], wo_a, [], 8)
        self.assertEqual(z_empty.count_nonzero().item(), 0)

    def test_dirty_and_query_rows_replace_stale_local_before_one_wo_b(self):
        offline = torch.randn(7, 8, 4, dtype=torch.float64)
        online = torch.randn_like(offline)
        wo_a = torch.randn(2, 5, 16, dtype=torch.float64)
        weight_b = torch.randn(11, 10, dtype=torch.float64)
        local, global_ = [6, 0, 3], [1, 2, 4, 5, 7]
        dirty = torch.tensor([6, 2, 5])  # Rows 5 and 6 are new query tokens.
        calls = []

        def wo_b(values):
            calls.append(values.shape)
            return values @ weight_b.T

        projector = GroupedMLAProjector(wo_a, wo_b, 8)
        z_off = projector.project(offline[:, local], local)
        z_global = projector.project(online[:, global_], global_)
        z_dirty = projector.project(online[dirty][:, local], local)
        off_original = z_off.clone()
        merged = merge_mla_contributions(z_off, z_global, z_dirty, dirty)
        expected_heads = online.clone()
        expected_heads[:, local] = offline[:, local]
        expected_heads[dirty] = online[dirty]
        expected = self._full_projection(expected_heads, wo_a)
        self.assertClose(merged, expected)
        self.assertTrue(torch.equal(z_off, off_original))
        final = projector.merge_and_project(z_off, z_global, z_dirty, dirty)
        self.assertEqual(calls, [torch.Size([7, 10])])
        self.assertClose(final, expected.flatten(1) @ weight_b.T)

    def test_all_clean_and_all_dirty_rows(self):
        z_off = torch.randn(4, 2, 3, dtype=torch.float64)
        z_global = torch.randn_like(z_off)
        fresh = torch.randn_like(z_off)
        empty = torch.empty(0, dtype=torch.int64)
        self.assertClose(
            merge_mla_contributions(z_off, z_global, fresh[:0], empty),
            z_off + z_global,
        )
        self.assertClose(
            merge_mla_contributions(z_off, z_global, fresh, torch.arange(4)),
            fresh + z_global,
        )

    def test_invalid_head_mapping_and_dirty_row_geometry_fail_closed(self):
        attn = torch.zeros(2, 2, 4)
        wo_a = torch.zeros(2, 3, 8)
        with self.assertRaises(ValueError):
            project_head_contributions(attn, wo_a, [0, 0], 4)
        with self.assertRaises(ValueError):
            project_head_contributions(attn, wo_a, [0, 1], 6)
        z = torch.zeros(4, 2, 3)
        for rows in (
            torch.tensor([1, 1]),
            torch.tensor([0, 4]),
            torch.tensor([0.0, 1.0]),
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                merge_mla_contributions(z, z, z[:2], rows)
        with self.assertRaises(ValueError):
            merge_mla_contributions(z, z, z[:1], torch.tensor([0, 1]))


if __name__ == "__main__":
    unittest.main()
