"""CPU contract tests; no vLLM/model launch or CUDA experiment."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from vllm_redknot.model_backends.mha_reuse import (
    RUNTIME_INTEGRATED,
    build_head_class_policy,
    build_sparse_ffn_policy,
    plan_swa_reuse,
    replay_swa_documents,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class SWAPlanTests(unittest.TestCase):
    def test_default_twenty_percent_and_first_document_unchanged(self):
        plan = plan_swa_reuse([7500, 7500, 7500, 7500], sliding_window=4096)
        self.assertEqual(plan.offsets, (0, 7500, 15000, 22500))
        self.assertEqual(plan.boundary_lengths, (0, 1500, 1500, 1500))
        self.assertEqual(plan.query_position, 30000)
        self.assertEqual(plan.replay_tokens, 4500)
        self.assertFalse(RUNTIME_INTEGRATED)

    def test_ratio_floor_keeps_at_least_one_boundary_token(self):
        plan = plan_swa_reuse([3, 1, 9], sliding_window=4)
        self.assertEqual(plan.boundary_lengths, (0, 1, 1))

    def test_explicit_prefix_is_capped_at_document_length(self):
        plan = plan_swa_reuse(
            [3, 2, 8], sliding_window=4, recompute_ratio=None, recompute_prefix=5
        )
        self.assertEqual(plan.boundary_lengths, (0, 2, 5))

    def test_all_replay_still_retains_first_document(self):
        self.assertEqual(
            plan_swa_reuse(
                [7, 11], sliding_window=4, recompute_ratio=1
            ).boundary_lengths,
            (0, 11),
        )

    def test_single_document_requires_no_replay(self):
        self.assertEqual(plan_swa_reuse([12], sliding_window=4).boundary_lengths, (0,))

    def test_invalid_inputs_fail_closed(self):
        for lengths in ([], [0], [-1], [True], [1.2]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                plan_swa_reuse(lengths, sliding_window=4)
        for ratio in (0, -0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                plan_swa_reuse([4, 4], sliding_window=4, recompute_ratio=ratio)
        for window in (0, -1, True):
            with self.subTest(window=window), self.assertRaises(ValueError):
                plan_swa_reuse([4, 4], sliding_window=window)

    def test_import_does_not_import_tensor_or_engine_dependencies(self):
        code = """
import importlib.abc, sys
blocked = {'torch', 'triton', 'vllm', 'sglang', 'transformers'}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('heavy dependency imported')
sys.meta_path.insert(0, Deny())
from vllm_redknot.model_backends.mha_reuse import plan_swa_reuse
assert plan_swa_reuse([10, 10], sliding_window=4).replay_tokens == 2
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_target_provenance_hash(self):
        document = json.loads((ROOT / "docs/mha_migration_provenance.json").read_text())
        self.assertFalse(document["runtime_integrated"])
        self.assertEqual(
            hashlib.sha256((ROOT / document["target_path"]).read_bytes()).hexdigest(),
            document["target_sha256"],
        )
        self.assertEqual(len(document["sources"]), 4)

    def test_source_hashes_when_explicit_source_available(self):
        location = os.environ.get("REDKNOT_SOURCE_ROOT")
        if not location:
            self.skipTest("source hash check requires explicit REDKNOT_SOURCE_ROOT")
        source = Path(location)
        document = json.loads((ROOT / "docs/mha_migration_provenance.json").read_text())
        for item in document["sources"]:
            self.assertEqual(
                hashlib.sha256((source / item["path"]).read_bytes()).hexdigest(),
                item["sha256"],
                item["path"],
            )


@unittest.skipUnless(HAS_TORCH, "CPU tensor callback oracle requires Torch")
class SWATensorTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.ids = ((10, 11, 12, 13), (20, 21, 22, 23), (30, 31, 32, 33))
        self.documents = tuple(
            tuple(
                (
                    torch.full((1, 2, 4, 3), float(doc * 10 + layer), device="cpu"),
                    torch.full((1, 2, 4, 3), float(doc * 10 + layer + 1), device="cpu"),
                )
                for layer in range(2)
            )
            for doc in range(3)
        )
        self.plan = plan_swa_reuse([4, 4, 4], sliding_window=4, recompute_ratio=0.5)

    @staticmethod
    def reposition(key, *, dst_start, src_start, **_):
        # Deliberately a simple offset oracle, not a real RoPE formula.
        return key + dst_start - src_start

    def test_relocate_then_prefix_replace_suffix_reuse(self):
        calls = []

        def forward(request):
            calls.append(request)
            self.assertEqual(request.token_ids, self.ids[request.document_index][:2])
            self.assertEqual(
                request.prior_layer_kv[0][0].shape[2], request.start_position
            )
            if request.document_index == 2:
                self.assertTrue(
                    self.torch.all(request.prior_layer_kv[0][0][:, :, 4:6] == 101)
                )
            return tuple(
                (
                    self.torch.full(
                        (1, 2, 2, 3), 100.0 + request.document_index + layer
                    ),
                    self.torch.full(
                        (1, 2, 2, 3), 200.0 + request.document_index + layer
                    ),
                )
                for layer in range(2)
            )

        before = deepcopy(self.documents)
        result = replay_swa_documents(
            self.plan,
            self.ids,
            self.documents,
            reposition_key=self.reposition,
            forward_boundary=forward,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.query_position, 12)
        self.assertEqual(result.sliding_window, 4)
        self.assertEqual(result.query_layer_kv[0][0].shape[2], 12)
        for doc in range(3):
            for layer in range(2):
                for side in (0, 1):
                    self.assertTrue(
                        self.torch.equal(
                            self.documents[doc][layer][side], before[doc][layer][side]
                        )
                    )
        self.assertTrue(
            self.torch.equal(result.document_layer_kv[0][0][0], before[0][0][0])
        )
        self.assertTrue(
            self.torch.all(result.document_layer_kv[1][0][0][:, :, :2] == 101)
        )
        self.assertTrue(
            self.torch.all(result.document_layer_kv[1][0][0][:, :, 2:] == 14)
        )
        self.assertTrue(
            self.torch.all(result.document_layer_kv[1][0][1][:, :, 2:] == 11)
        )

    def test_single_document_never_calls_callbacks(self):
        def unexpected(*_, **__):
            self.fail("single document should not relocate or replay")

        result = replay_swa_documents(
            plan_swa_reuse([4], sliding_window=4),
            self.ids[:1],
            self.documents[:1],
            reposition_key=unexpected,
            forward_boundary=unexpected,
        )
        self.assertTrue(
            self.torch.equal(result.query_layer_kv[0][0], self.documents[0][0][0])
        )

    def test_callback_failure_cannot_mutate_input(self):
        def mutate_then_fail(request):
            request.prior_layer_kv[0][0].fill_(999)
            raise RuntimeError("native callback failed")

        before = deepcopy(self.documents)
        with self.assertRaises(RuntimeError):
            replay_swa_documents(
                self.plan,
                self.ids,
                self.documents,
                reposition_key=self.reposition,
                forward_boundary=mutate_then_fail,
            )
        for doc in range(3):
            for layer in range(2):
                for side in (0, 1):
                    self.assertTrue(
                        self.torch.equal(
                            self.documents[doc][layer][side], before[doc][layer][side]
                        )
                    )

    def test_bad_boundary_return_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_swa_documents(
                self.plan,
                self.ids,
                self.documents,
                reposition_key=self.reposition,
                forward_boundary=lambda request: request.prior_layer_kv,
            )

    def test_forged_plan_and_token_count_are_rejected(self):
        for plan, ids in (
            (replace(self.plan, offsets=(0, 99, 8)), self.ids),
            (replace(self.plan, boundary_lengths=(0, 5, 2)), self.ids),
            (self.plan, self.ids[:1]),
        ):
            with self.assertRaises(ValueError):
                replay_swa_documents(
                    plan,
                    ids,
                    self.documents,
                    reposition_key=self.reposition,
                    forward_boundary=lambda _: (),
                )

    def test_core_head_policy_alias_sink_window_and_no_source_mutation(self):
        raw = {
            "num_layers": 1,
            "num_kv_heads": 3,
            "kv_head_classification": [["local_full", "global", "retrieval"]],
            "kv_head_max_distance": [[4096, -1, -1]],
            "kv_head_sink_size": [[4, 4, 8]],
        }
        saved = deepcopy(raw)
        config = build_head_class_policy(
            raw, total_context_tokens=10000, window_ratio=0.5
        )
        self.assertEqual(config.get_strategy(0, 0).window, 5000)
        self.assertEqual(config.get_strategy(0, 0).sink_size, 4)
        self.assertEqual(config.get_strategy(0, 2).head_type, "global")
        self.assertEqual(raw, saved)
        fixed = build_head_class_policy(
            raw,
            total_context_tokens=10000,
            window_ratio=0.5,
            fixed_window=4096,
            merge_retrieval_to_global=False,
        )
        self.assertEqual(fixed.get_strategy(0, 0).window, 4096)
        self.assertEqual(fixed.get_strategy(0, 2).head_type, "retrieval")

    def test_sparse_ffn_contract_uses_core_schedule(self):
        raw = {
            "dense_until": 20,
            "deep_layer_start": 60,
            "mass_thresh": 0.2,
            "mass_thresh_deep": 0.05,
            "recent_n": 512,
            "local_window": 4096,
        }
        policy = build_sparse_ffn_policy(raw)
        self.assertEqual(policy.__class__.__module__, "vllm_redknot.core.sparse_ffn")
        self.assertEqual(
            [policy.get_mass_thresh(i) for i in (0, 20, 60)], [1.0, 0.2, 0.05]
        )
        self.assertEqual(raw["local_window"], 4096)
        with self.assertRaises(ValueError):
            build_sparse_ffn_policy({"typo": 1})


if __name__ == "__main__":
    unittest.main()
