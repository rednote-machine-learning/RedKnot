from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

import torch


MODULE = Path(__file__).with_name("redknot_adaptive_topk.py")
if not MODULE.is_file():
    MODULE = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/layers/moe/redknot_adaptive_topk.py"
    )
spec = importlib.util.spec_from_file_location("redknot_adaptive_topk", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class Output(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


def apply(layer: int, weights: torch.Tensor, *, forward_batch=None, **env):
    ids = torch.arange(6, dtype=torch.int32).view(1, 6).expand(weights.shape[0], -1)
    output = Output(weights, ids, torch.zeros(weights.shape[0], 256))
    base = {
        "REDKNOT_ADAPTIVE_TOPK": "1",
        "REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS": "0.90",
        "REDKNOT_ADAPTIVE_TOPK_BUCKETS": "4,5,6",
        "REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS": "1",
        "REDKNOT_ADAPTIVE_TOPK_LOG_FIRST_HISTOGRAM": "0",
        "REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION": "1",
    }
    base.update({k: str(v) for k, v in env.items()})
    with mock.patch.dict(os.environ, base, clear=False):
        return mod.maybe_apply_dsv4_adaptive_topk(
            layer_id=layer,
            num_hidden_layers=43,
            native_routed_topk=6,
            num_fused_shared_experts=0,
            is_hash_layer=layer < 3,
            hidden_states=torch.zeros(weights.shape[0], 8),
            topk_output=output,
            forward_batch=forward_batch,
        )


class AdaptiveTopKTest(unittest.TestCase):
    def setUp(self):
        mod._LOGGED_LAYERS.clear()
        mod._STATS_LOGGED_LAYERS.clear()

    def test_middle_layer_selects_real_k4_k5_k6_without_renormalizing(self):
        weights = torch.tensor(
            [
                [.40, .30, .15, .10, .04, .01],  # mass4=.95 -> K4
                [.30, .25, .20, .10, .08, .07],  # mass4=.85, mass5=.93 -> K5
                [.22, .20, .18, .16, .13, .11],  # mass5=.89 -> K6
            ],
            dtype=torch.float32,
        )
        result = apply(24, weights)
        self.assertIsNotNone(result)
        kept = (result.topk_ids >= 0).sum(dim=1)
        self.assertTrue(torch.equal(kept, torch.tensor([4, 5, 6])))
        self.assertTrue(torch.equal(result.topk_weights, weights * (result.topk_ids >= 0)))
        self.assertTrue(hasattr(result.topk_ids, "_sglang_moe_route_mask"))

    def test_selection_uses_weight_rank_not_candidate_slot_order(self):
        weights = torch.tensor([[.01, .40, .04, .30, .15, .10]])
        result = apply(10, weights)
        kept_ids = result.topk_ids[result.topk_ids >= 0].tolist()
        self.assertEqual(set(kept_ids), {1, 3, 4, 5})

    def test_hash_prefix_and_suffix_layers_stay_native(self):
        weights = torch.tensor([[.91, .04, .02, .01, .01, .01]])
        self.assertIsNone(apply(0, weights))
        self.assertIsNone(apply(40, weights))

    def test_small_batch_stays_native(self):
        weights = torch.tensor([[.91, .04, .02, .01, .01, .01]])
        result = apply(
            20,
            weights,
            REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS=2,
        )
        self.assertIsNone(result)

    def test_mass_half_physically_compacts_to_k3_in_native_slot_order(self):
        weights = torch.tensor(
            [
                [.01, .40, .04, .30, .15, .10],
                [.18, .16, .14, .12, .20, .20],
            ],
            dtype=torch.float32,
        )
        result = apply(
            20,
            weights,
            REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS="0.50",
            REDKNOT_ADAPTIVE_TOPK_BUCKETS="3,4,5,6",
        )
        self.assertEqual(tuple(result.topk_ids.shape), (2, 3))
        self.assertEqual(tuple(result.topk_weights.shape), (2, 3))
        self.assertTrue(
            torch.equal(
                result.topk_ids,
                torch.tensor([[1, 3, 4], [0, 4, 5]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                result.topk_weights,
                torch.tensor([[.40, .30, .15], [.18, .20, .20]]),
            )
        )
        self.assertFalse(hasattr(result.topk_ids, "_sglang_moe_route_mask"))
        self.assertEqual(result.topk_ids._sglang_moe_physical_topk, 3)

    def test_physical_compaction_can_be_disabled_for_ab_comparison(self):
        weights = torch.tensor([[.40, .30, .15, .10, .04, .01]])
        result = apply(
            20,
            weights,
            REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS="0.50",
            REDKNOT_ADAPTIVE_TOPK_BUCKETS="3,4,5,6",
            REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION="0",
        )
        self.assertEqual(tuple(result.topk_ids.shape), (1, 6))
        self.assertEqual(int((result.topk_ids >= 0).sum()), 3)
        self.assertTrue(hasattr(result.topk_ids, "_sglang_moe_route_mask"))

    def test_physical_k3_matches_masked_k3_routes_and_weights(self):
        generator = torch.Generator().manual_seed(20260824)
        weights = torch.softmax(torch.randn(257, 6, generator=generator), dim=1)
        compact = apply(
            20,
            weights,
            REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS="0.50",
            REDKNOT_ADAPTIVE_TOPK_BUCKETS="3,4,5,6",
            REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION="1",
        )
        masked = apply(
            20,
            weights,
            REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS="0.50",
            REDKNOT_ADAPTIVE_TOPK_BUCKETS="3,4,5,6",
            REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION="0",
        )
        masked_keep = masked.topk_ids >= 0
        self.assertTrue(
            torch.equal(compact.topk_ids, masked.topk_ids[masked_keep].view(-1, 3))
        )
        self.assertTrue(
            torch.equal(
                compact.topk_weights,
                masked.topk_weights[masked_keep].view(-1, 3),
            )
        )

    def test_plan_scoped_policy_keeps_dense_request_native_top6(self):
        weights = torch.tensor([[.40, .30, .15, .10, .04, .01]])
        result = apply(
            20,
            weights,
            forward_batch=SimpleNamespace(redknot_reuse_plan=[None]),
            REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED="1",
            REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS="0.50",
            REDKNOT_ADAPTIVE_TOPK_BUCKETS="3,4,5,6",
        )
        self.assertIsNone(result)

    def test_plan_scoped_policy_compacts_combined_restore_only(self):
        weights = torch.tensor([[.40, .30, .15, .10, .04, .01]])
        plan = {
            "mode": "restore",
            "mla_off_execution_profile": mod._COMBINED_EXECUTION_PROFILE,
        }
        result = apply(
            20,
            weights,
            forward_batch=SimpleNamespace(redknot_reuse_plan=[plan]),
            REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED="1",
            REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS="0.50",
            REDKNOT_ADAPTIVE_TOPK_BUCKETS="3,4,5,6",
        )
        self.assertEqual(tuple(result.topk_ids.shape), (1, 3))

    def test_plan_scoped_policy_rejects_mixed_dense_and_redknot_batch(self):
        weights = torch.tensor([[.40, .30, .15, .10, .04, .01]])
        plan = {
            "mode": "restore",
            "mla_off_execution_profile": mod._COMBINED_EXECUTION_PROFILE,
        }
        result = apply(
            20,
            weights,
            forward_batch=SimpleNamespace(redknot_reuse_plan=[plan, None]),
            REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED="1",
            REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS="0.50",
            REDKNOT_ADAPTIVE_TOPK_BUCKETS="3,4,5,6",
        )
        self.assertIsNone(result)

    def test_invalid_policy_fails_closed(self):
        weights = torch.tensor([[.5, .2, .1, .1, .05, .05]])
        with self.assertRaises(ValueError):
            apply(
                20,
                weights,
                REDKNOT_ADAPTIVE_TOPK_BUCKETS="4,5",
            )

    def test_marlin_patch_zeroes_filtered_down_projection_slots(self):
        path = Path(__file__).with_name("fused_marlin_moe.py")
        if not path.is_file():
            path = (
                Path(__file__).resolve().parents[4]
                / "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"
            )
        source = path.read_text()
        self.assertIn('getattr(topk_ids, "_sglang_moe_route_mask", None)', source)
        self.assertIn("has_filtered_routes = expert_map is not None or has_route_mask", source)
        self.assertEqual(source.count("is_ep=has_filtered_routes"), 2)
        self.assertIn("if has_filtered_routes", source)
        self.assertIn("intermediate_cache3.zero_()", source)


if __name__ == "__main__":
    unittest.main()
