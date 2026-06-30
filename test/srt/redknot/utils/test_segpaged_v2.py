#!/usr/bin/env python3
"""Unit tests for the SegPaged v2 segment-paged storage substrate.

These tests are CPU-only (no FA-3 / CUDA required) and exercise the new,
standalone ``redknot.segpaged_v2`` package:

  1. Visible-token plans (global / local / custom) produce correct positions.
  2. The paged storage stores only visible tokens and gathers them back
     losslessly (segment-paged virtual->physical mapping is correct).
  3. ``paged_attention`` (reference path) is numerically equivalent to a
     dense + masked baseline (cosine ~ 1).
  4. KV token savings match the analytical expectation.

Run::

    python -m pytest test/srt/redknot/test_segpaged_v2.py -q
    # or
    python test/srt/redknot/test_segpaged_v2.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.layers.attention.redknot.segpaged_v2 import (  # noqa: E402
    POLICY_CUSTOM,
    POLICY_GLOBAL,
    POLICY_LOCAL,
    LocalPagedPool,
    PagedHeadKVCache,
    build_paged_cache,
    custom_plan,
    dense_reference_attention,
    global_plan,
    local_plan,
    paged_attention,
    plans_from_policies,
    verify_against_dense,
)


class TestVisiblePlan(unittest.TestCase):
    def test_global_plan_keeps_all(self):
        p = global_plan(10)
        self.assertEqual(p.policy, POLICY_GLOBAL)
        self.assertEqual(p.kept, 10)
        self.assertTrue(torch.equal(p.positions, torch.arange(10)))

    def test_local_plan_sink_and_recent(self):
        p = local_plan(100, sink=4, recent=8)
        self.assertEqual(p.policy, POLICY_LOCAL)
        expected = torch.cat([torch.arange(0, 4), torch.arange(92, 100)])
        self.assertTrue(torch.equal(p.positions, expected))

    def test_local_plan_overlap_degrades_to_full(self):
        # sink + recent >= seq_len -> no duplicate positions, sorted unique
        p = local_plan(10, sink=6, recent=8)
        self.assertEqual(p.positions.tolist(), list(range(10)))

    def test_custom_plan_dedup_sort_clamp(self):
        p = custom_plan(10, [9, 9, 3, 1, 100, -5, 3])
        self.assertEqual(p.policy, POLICY_CUSTOM)
        self.assertEqual(p.positions.tolist(), [1, 3, 9])


class TestPagedStorage(unittest.TestCase):
    def test_store_and_gather_lossless_global(self):
        L, D = 130, 16  # spans 3 pages at page_size=64
        k = torch.randn(L, D)
        v = torch.randn(L, D)
        cache = PagedHeadKVCache(
            num_kv_heads=1, head_dim=D, page_size=64, dtype=torch.float32
        )
        cache.store_head_segment(
            layer=0, head=0, segment=0, k_dense=k, v_dense=v, plan=global_plan(L)
        )
        gk, gv = cache.gather_head(0, 0)
        self.assertTrue(torch.allclose(gk, k))
        self.assertTrue(torch.allclose(gv, v))

    def test_store_only_visible_tokens_local(self):
        L, D = 200, 8
        sink, recent = 4, 16
        k = torch.randn(L, D)
        v = torch.randn(L, D)
        cache = PagedHeadKVCache(
            num_kv_heads=1, head_dim=D, page_size=64, dtype=torch.float32
        )
        plan = local_plan(L, sink=sink, recent=recent)
        cache.store_head_segment(
            layer=0, head=0, segment=0, k_dense=k, v_dense=v, plan=plan
        )
        # Only sink + recent tokens are physically stored.
        self.assertEqual(cache.stored_token_count(), sink + recent)
        gk, gv = cache.gather_head(0, 0)
        expected_k = k.index_select(0, plan.positions)
        self.assertTrue(torch.allclose(gk, expected_k))
        self.assertTrue(torch.allclose(gv, v.index_select(0, plan.positions)))

    def test_custom_backend_injection(self):
        # A custom storage backend must satisfy the same contract.
        D = 8
        backend = LocalPagedPool(head_dim=D, page_size=32, dtype=torch.float32)
        cache = PagedHeadKVCache(
            num_kv_heads=1,
            head_dim=D,
            page_size=32,
            storage=backend,
            dtype=torch.float32,
        )
        k = torch.randn(70, D)
        v = torch.randn(70, D)
        cache.store_head_segment(
            layer=0, head=0, segment=0, k_dense=k, v_dense=v, plan=global_plan(70)
        )
        self.assertIs(cache.storage, backend)
        # 70 tokens at page_size=32 -> 3 physical pages.
        self.assertEqual(backend.num_physical_pages(), 3)
        gk, _ = cache.gather_head(0, 0)
        self.assertTrue(torch.allclose(gk, k))


class TestPagedAttentionEquivalence(unittest.TestCase):
    def _run_equiv(self, policies, sink, recent):
        torch.manual_seed(0)
        Hkv, gqa, D, L, Lq = len(policies), 2, 16, 128, 1
        Hq = Hkv * gqa
        k = torch.randn(Hkv, L, D)
        v = torch.randn(Hkv, L, D)
        q = torch.randn(Hq, Lq, D)
        sm = 1.0 / (D**0.5)
        plans = plans_from_policies(
            head_policy=policies, seq_len=L, sink=sink, recent=recent
        )
        cache = build_paged_cache(k_dense=k, v_dense=v, plans=plans)
        seg = paged_attention(
            q, cache, layer=0, num_q_per_kv=gqa, sm_scale=sm, use_fused=False
        )
        ref = dense_reference_attention(q, k, v, plans, num_q_per_kv=gqa, sm_scale=sm)
        cos = torch.nn.functional.cosine_similarity(
            seg.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        return cos

    def test_mixed_global_local_equivalence(self):
        policies = [POLICY_GLOBAL, POLICY_GLOBAL, POLICY_LOCAL, POLICY_LOCAL]
        cos = self._run_equiv(policies, sink=4, recent=32)
        self.assertGreater(cos, 0.999999)

    def test_all_local_equivalence(self):
        policies = [POLICY_LOCAL] * 4
        cos = self._run_equiv(policies, sink=2, recent=16)
        self.assertGreater(cos, 0.999999)

    def test_custom_positions_equivalence(self):
        torch.manual_seed(1)
        Hkv, gqa, D, L = 2, 2, 16, 64
        Hq = Hkv * gqa
        k = torch.randn(Hkv, L, D)
        v = torch.randn(Hkv, L, D)
        q = torch.randn(Hq, 1, D)
        sm = 1.0 / (D**0.5)
        plans = [
            global_plan(L),
            custom_plan(L, [0, 5, 9, 17, 33, 60]),
        ]
        cache = build_paged_cache(k_dense=k, v_dense=v, plans=plans)
        seg = paged_attention(
            q, cache, layer=0, num_q_per_kv=gqa, sm_scale=sm, use_fused=False
        )
        ref = dense_reference_attention(q, k, v, plans, num_q_per_kv=gqa, sm_scale=sm)
        cos = torch.nn.functional.cosine_similarity(
            seg.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.999999)


class TestVerifyHarness(unittest.TestCase):
    def test_verify_against_dense_reports_savings(self):
        res = verify_against_dense(
            num_kv_heads=8,
            num_q_per_kv=4,
            head_dim=32,
            seq_len=512,
            sink=4,
            recent=64,
            global_ratio=0.25,
            dtype=torch.float32,
        )
        self.assertGreater(res["cosine"], 0.999999)
        self.assertGreater(res["kv_token_saving"], 0.0)
        self.assertLess(res["segpaged_kv_tokens"], res["dense_kv_tokens"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
