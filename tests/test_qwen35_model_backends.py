"""CPU contracts for extracted Qwen3.5/MoE helpers, not model validation."""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from vllm_redknot.model_backends.qwen35_policy import (
    full_attention_layer_indices,
    linear_attention_layer_indices,
)
from vllm_redknot.model_backends.qwen35_reuse_contract import (
    Qwen35ReuseConfig,
    plan_qwen35_prefix_reuse,
)
from vllm_redknot.model_backends.sparse_moe_policy import RedKnotSparseMoEPolicy

PROJECT = Path(__file__).resolve().parents[1]
PROVENANCE = PROJECT / "docs" / "qwen35_migration_provenance.json"


def safe_config():
    return Qwen35ReuseConfig(False, False, False, False, 1, False, False, 1024)


def plan_kwargs():
    return dict(
        segments=("doc",),
        state_slots=(0,),
        seq_lens=(4,),
        prefix_lens=(0,),
        document_lengths={"doc": 128},
        loaded_slots={},
        is_prefill=True,
        config=safe_config(),
    )


class Qwen35PortableContractTest(unittest.TestCase):
    def test_layer_policy_preserves_explicit_types_and_interval_fallback(self):
        config = SimpleNamespace(layer_types=["linear_attention", "full_attention"] * 3)
        self.assertEqual(full_attention_layer_indices(config), [1, 3, 5])
        self.assertEqual(linear_attention_layer_indices(config), [0, 2, 4])
        fallback = SimpleNamespace(num_hidden_layers=8, full_attention_interval=4)
        self.assertEqual(full_attention_layer_indices(fallback), [3, 7])
        self.assertEqual(linear_attention_layer_indices(fallback), [])

    def test_plan_retains_ordered_bundle_offset_and_once_only_restore(self):
        kwargs = plan_kwargs()
        result = plan_qwen35_prefix_reuse(**kwargs)
        self.assertEqual(result.position_offsets, (128,))
        self.assertEqual(result.logical_seq_lens, (132,))
        self.assertEqual(result.restore_rows, (0,))
        kwargs.update(loaded_slots={0: "doc"}, prefix_lens=(3,))
        self.assertEqual(plan_qwen35_prefix_reuse(**kwargs).restore_rows, ())
        kwargs["is_prefill"] = False
        self.assertEqual(plan_qwen35_prefix_reuse(**kwargs).restore_rows, ())

    def test_incompatible_state_plans_fail_before_mutating_receipts(self):
        variants = (
            {"segments": (("doc", "other"),)},
            {"is_prefill": False},
            {"prefix_lens": (1,)},
            {"state_slots": (True,)},
            {"document_lengths": {}},
            {"config": replace(safe_config(), max_model_len=131)},
        )
        for delta in variants:
            kwargs = plan_kwargs()
            kwargs.update(delta)
            before = dict(kwargs["loaded_slots"])
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                plan_qwen35_prefix_reuse(**kwargs)
            self.assertEqual(kwargs["loaded_slots"], before)

    def test_unsupported_native_modes_are_explicitly_rejected(self):
        for name in (
            "cuda_graph_enabled",
            "piecewise_cuda_graph_enabled",
            "prefix_caching_enabled",
            "multimodal_enabled",
            "data_parallel_attention",
            "speculative_decoding",
        ):
            with self.subTest(mode=name), self.assertRaises(ValueError):
                replace(safe_config(), **{name: True}).validate()
        with self.assertRaises(ValueError):
            replace(safe_config(), pipeline_parallel_size=2).validate()

    def test_sparse_moe_config_is_explicit_and_disabled_by_default(self):
        self.assertFalse(RedKnotSparseMoEPolicy.from_config(None).enabled)
        policy = RedKnotSparseMoEPolicy.from_config(
            {
                "enable_redknot_sparse_moe": True,
                "redknot_moe_dense_until_layer": 7,
                "redknot_moe_min_keep_ratio": 0.3,
            }
        )
        self.assertFalse(policy.layer_is_sparse_eligible(6))
        self.assertTrue(policy.layer_is_sparse_eligible(7))
        self.assertEqual(policy.min_keep_ratio, 0.3)

    def test_control_modules_import_without_tensor_or_native_engines(self):
        code = """
import importlib, importlib.abc, sys
blocked = {'torch', 'triton', 'sglang', 'sgl_kernel', 'vllm', 'transformers'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
for name in ('qwen35_policy', 'qwen35_reuse_contract', 'sparse_moe_policy'):
    importlib.import_module('vllm_redknot.model_backends.' + name)
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_provenance_and_native_import_boundary(self):
        provenance = json.loads(PROVENANCE.read_text())
        source_root = PROJECT / provenance["source_snapshot_relative_to_project"]
        for entry in provenance["files"]:
            target = PROJECT / entry["target_path"]
            self.assertFalse(entry["runtime_integrated"])
            self.assertEqual(
                hashlib.sha256(target.read_bytes()).hexdigest(),
                entry["target_sha256"],
            )
            if source_root.is_dir() and entry["source_path"]:
                self.assertEqual(
                    hashlib.sha256(
                        (source_root / entry["source_path"]).read_bytes()
                    ).hexdigest(),
                    entry["source_sha256"],
                )
            for node in ast.walk(ast.parse(target.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                self.assertFalse(
                    any(
                        name.split(".")[0] in {"sglang", "sgl_kernel", "transformers"}
                        for name in names
                    )
                )


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch is optional")
class Qwen35TensorContractTest(unittest.TestCase):
    def test_recurrent_capture_restore_once_and_cleared_slot_reload(self):
        import torch

        from vllm_redknot.model_backends.qwen35_recurrent import (
            Qwen35RecurrentBuffers,
            capture_qwen35_recurrent_state,
            prepare_qwen35_offline_reuse,
        )

        buffers = Qwen35RecurrentBuffers(
            (torch.ones(1, 2, 3),), torch.full((1, 2, 2, 2), 2.0)
        )
        snapshot = capture_qwen35_recurrent_state(buffers, 0)
        buffers.conv[0].zero_()
        buffers.temporal.zero_()
        segment = SimpleNamespace(doc_len=128, recurrent_state=snapshot)
        kwargs = plan_kwargs()
        kwargs.pop("document_lengths")
        kwargs.update(buffers=buffers, get_segment=lambda sid: segment)
        result = prepare_qwen35_offline_reuse(**kwargs)
        self.assertEqual(result.restore_rows, (0,))
        self.assertTrue(torch.equal(buffers.conv[0][:, 0], snapshot.conv[0]))
        buffers.conv[0][:, 0].fill_(7)
        self.assertEqual(prepare_qwen35_offline_reuse(**kwargs).restore_rows, ())
        self.assertTrue(torch.all(buffers.conv[0][:, 0] == 7))
        prepare_qwen35_offline_reuse(**kwargs, cleared_slots=(0,))
        self.assertTrue(torch.all(buffers.conv[0][:, 0] == 1))

    def test_bad_later_snapshot_does_not_partially_restore_earlier_request(self):
        import torch

        from vllm_redknot.core.offline_cache import OfflineRecurrentState
        from vllm_redknot.model_backends.qwen35_recurrent import (
            Qwen35RecurrentBuffers,
            prepare_qwen35_offline_reuse,
        )

        buffers = Qwen35RecurrentBuffers((torch.zeros(1, 2, 3),), torch.zeros(1, 2, 2))
        segments = {
            "doc": SimpleNamespace(
                doc_len=128,
                recurrent_state=OfflineRecurrentState(
                    [torch.ones(1, 3)], torch.ones(1, 2)
                ),
            ),
            "bad": SimpleNamespace(
                doc_len=128,
                recurrent_state=OfflineRecurrentState(
                    [torch.ones(1, 3)], torch.ones(1, 7)
                ),
            ),
        }
        loaded = {}
        with self.assertRaisesRegex(ValueError, "GDN state shapes"):
            prepare_qwen35_offline_reuse(
                buffers=buffers,
                segments=("doc", "bad"),
                state_slots=(0, 1),
                seq_lens=(4, 4),
                prefix_lens=(0, 0),
                loaded_slots=loaded,
                get_segment=segments.get,
                is_prefill=True,
                config=safe_config(),
            )
        self.assertFalse(bool(buffers.conv[0].any()))
        self.assertFalse(bool(buffers.temporal.any()))
        self.assertEqual(loaded, {})

    def test_linear_prefix_state_relay_matches_single_pass(self):
        import torch

        from vllm_redknot.model_backends.qwen35_linear import _linear_local_recurrence

        gen = torch.Generator(device="cpu").manual_seed(1)
        q = torch.randn(1, 2, 5, 3, generator=gen, dtype=torch.float64)
        k = torch.randn(1, 2, 5, 3, generator=gen, dtype=torch.float64)
        v = torch.randn(1, 2, 5, 2, generator=gen, dtype=torch.float64)
        g = torch.full((1, 2, 5), -0.1, dtype=torch.float64)
        beta = torch.full((1, 2, 5), 0.3, dtype=torch.float64)
        full, state = _linear_local_recurrence(
            None, q, k, v, g, beta, None, return_state=True
        )
        first, carried = _linear_local_recurrence(
            None,
            q[:, :, :2],
            k[:, :, :2],
            v[:, :, :2],
            g[:, :, :2],
            beta[:, :, :2],
            None,
            return_state=True,
        )
        last, final = _linear_local_recurrence(
            None,
            q[:, :, 2:],
            k[:, :, 2:],
            v[:, :, 2:],
            g[:, :, 2:],
            beta[:, :, 2:],
            None,
            initial_state=carried,
            return_state=True,
        )
        torch.testing.assert_close(torch.cat((first, last), dim=2), full)
        torch.testing.assert_close(final, state)

    def test_sparse_moe_request_threshold_and_layout_generation(self):
        import torch

        from vllm_redknot.model_backends.sparse_moe import (
            build_routed_keep_mask,
            resolve_routed_keep_mask,
        )
        from vllm_redknot.model_backends.sparse_moe_policy import (
            RedKnotTokenPolicyContext,
        )

        policy = RedKnotSparseMoEPolicy(
            enabled=True,
            dense_until_layer=0,
            alpha=1,
            recent_tokens=0,
            min_keep_tokens=0,
            min_keep_ratio=0,
            dense_fallback_keep_ratio=1,
        )
        result = build_routed_keep_mask(
            torch.tensor([10.0, 0.0, 1.0, 0.0]),
            policy=policy,
            cu_seqlens=torch.tensor([0, 2, 4]),
        )
        self.assertEqual(result.keep_mask.tolist(), [True, False, True, False])
        context = RedKnotTokenPolicyContext(layout_version=4)
        context.set_mask(
            routed_keep_mask=result.keep_mask,
            source_layer_id=0,
            valid_until_layer_id=8,
            score_kind="test",
        )
        hidden = torch.zeros(4, 3)
        self.assertIsNone(
            resolve_routed_keep_mask(
                context, 1, hidden, policy=policy, is_prefill=True, layout_version=5
            )
        )
        self.assertIs(
            resolve_routed_keep_mask(
                context, 1, hidden, policy=policy, is_prefill=True, layout_version=4
            ),
            result.keep_mask,
        )


if __name__ == "__main__":
    unittest.main()
