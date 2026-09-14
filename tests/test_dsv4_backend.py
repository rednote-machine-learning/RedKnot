"""Exercise real DSV4 hooks with CPU oracles and fake pinned vLLM imports.

Only native GPU operations/platform imports are replaced. The actual installer,
request transaction, sparse interception and z_off projection helpers execute.
This does not certify CUDA kernels, real compressor numerics or model quality.
"""

import importlib.util
import sys
import types
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from vllm_redknot import dsv4_sparse
from vllm_redknot.cache import CacheManager
from vllm_redknot.dsv4_runner import CONTEXT_KEY
from vllm_redknot.dsv4_runtime import DSV4Chunk, DSV4LayerSpec, DSV4Runtime
from vllm_redknot.runtime import RedKnotSettings


@unittest.skipUnless(torch is not None, "CPU Torch is required for DSV4 hook tests")
class DSV4BackendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)
        self.context = NS(additional_kwargs={})
        self.native_calls = Counter()
        self.selected_calls = []
        self.project_calls = []
        self.wo_b_inputs = []
        self.active_layers = []
        self.raise_mqa = False
        self.name = "model.layers.3.attn"
        self.heads = (0, 3, 6, 17, 45, 63)
        # TP1 Flash has 64 real heads: padded and real dimensions coincide.
        self.spec = DSV4LayerSpec(self.heads, 64, 8, 2, 4)
        self.weight_a = torch.randn(8, 2, 8 * 512) / 32
        self.weight_b = torch.randn(16, 7) / 4
        self.settings = RedKnotSettings.from_mapping(
            {
                "local_heads": {"3": list(self.heads)},
                "model_revision": "cpu-fixture",
                "allow_approximate": True,
                "boundary_tokens": 1,
            }
        )
        self.runtime = DSV4Runtime(self.settings, CacheManager(1 << 20), "fixture")
        owner = self
        names = (
            "vllm",
            "vllm.forward_context",
            "vllm.models",
            "vllm.models.deepseek_v4",
            "vllm.models.deepseek_v4.nvidia",
            "vllm.models.deepseek_v4.nvidia.flashmla",
        )
        modules = {name: types.ModuleType(name) for name in names}
        modules["vllm.forward_context"].get_forward_context = lambda: self.context
        native = modules["vllm.models.deepseek_v4.nvidia.flashmla"]

        def original_sparse(**kwargs):
            owner.native_calls["sparse"] += 1
            q = kwargs["q"]
            result = dsv4_sparse.selected_sparse_mla_reference(
                q,
                kwargs["kv"],
                kwargs["indices"],
                kwargs["topk_length"],
                kwargs["attn_sink"],
                tuple(range(q.shape[0])),
                tuple(range(q.shape[1])),
                scale=kwargs["sm_scale"],
            )
            kwargs["out"].copy_(result)
            return kwargs["out"], None, None

        class NativeAttention:
            def __init__(self):
                self.prefix = owner.name
                self.n_local_heads, self.head_dim = 64, 512
                self.n_local_groups, self.o_lora_rank = 8, 2
                self.attn_sink = torch.linspace(-2, 0, 64)
                self.scale = 512**-0.5

            def wo_b(self, z):
                owner.wo_b_inputs.append(z.clone())
                return (z.float() @ owner.weight_b).to(z.dtype)

            def forward(self, q, kv, positions):
                # This fake native preparation mirrors the producer ordering.
                # Hooks must not replace it or prune rows from any producer.
                for producer in (
                    "swa_kv",
                    "attention_compressor",
                    "indexer_compressor",
                    "indexer",
                ):
                    owner.native_calls[producer] += 1
                    self.produced[producer] = kv.detach().clone()
                out = torch.empty_like(q)
                self.forward_mqa(q, kv, positions, out)
                self.last_attention = out.detach().clone()
                return self._o_proj(out[:, : self.n_local_heads], positions)

            def forward_mqa(self, q, kv, positions, output):
                owner.native_calls["mqa"] += 1
                entry = owner.context.additional_kwargs.get(CONTEXT_KEY)
                owner.active_layers.append(entry[1].active_layer if entry else None)
                if owner.raise_mqa:
                    raise RuntimeError("native MQA failed")
                indices, lengths = owner.candidates(q.shape[0], kv.shape[0])
                if q.shape[0] == 1 and int(positions[0]) > 0:
                    owner.native_calls["decode"] += 1
                    output.copy_(
                        dsv4_sparse.selected_sparse_mla_reference(
                            q,
                            kv,
                            indices,
                            lengths,
                            self.attn_sink,
                            (0,),
                            tuple(range(64)),
                            scale=self.scale,
                        )
                    )
                    return
                native.flash_mla_sparse_fwd(
                    q=q,
                    kv=kv,
                    indices=indices,
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=lengths,
                    out=output,
                )

            def _o_proj(self, o, positions):
                owner.native_calls["o_proj"] += 1
                return self.wo_b(owner.project_z(self, o, positions))

        native.DeepseekV4FlashMLAAttention = NativeAttention
        native.flash_mla_sparse_fwd = original_sparse
        source = Path(__file__).parents[1] / "vllm_redknot" / "dsv4_backend.py"
        spec = importlib.util.spec_from_file_location(
            "vllm_redknot._dsv4_backend_contract_test", source
        )
        self.module = importlib.util.module_from_spec(spec)

        def selected(*args, **kwargs):
            self.selected_calls.append((tuple(args[5]), tuple(args[6])))
            return dsv4_sparse.selected_sparse_mla_reference(*args, **kwargs)

        with (
            patch.dict(sys.modules, modules),
            patch.object(dsv4_sparse, "selected_sparse_mla", selected),
        ):
            spec.loader.exec_module(self.module)
            self.module.native_project_z = self.project_z
            self.module.install_dsv4_attention()
            installed = native.flash_mla_sparse_fwd
            self.module.install_dsv4_attention()
            self.assertIs(native.flash_mla_sparse_fwd, installed)
        self.layer = NativeAttention()
        self.layer.produced = {}

    @staticmethod
    def candidates(rows, kv_rows):
        # Repeated last candidates verify the hooks do not deduplicate them.
        indices = torch.arange(kv_rows).repeat(rows, 1)
        lengths = torch.arange(1, rows + 1).clamp_max(kv_rows).int()
        return indices[:, None, :].int(), lengths

    def z_oracle(self, o, positions):
        values = o.float().clone()
        angle = positions.float()[:, None, None] / 100
        even, odd = values[..., -64::2].clone(), values[..., -63::2].clone()
        values[..., -64::2] = even * angle.cos() + odd * angle.sin()
        values[..., -63::2] = odd * angle.cos() - even * angle.sin()
        return (
            torch.einsum(
                "tgd,grd->tgr", values.reshape(o.shape[0], 8, -1), self.weight_a
            )
            .flatten(1)
            .to(o.dtype)
        )

    def project_z(self, layer, o, positions):
        self.project_calls.append((o.clone(), positions.clone()))
        return self.z_oracle(o, positions)

    def tensors(self, length):
        return (
            (torch.randn(length, 64, 512) / 4).bfloat16(),
            (torch.randn(length, 1, 512) / 4).bfloat16(),
        )

    def args(self, mode, spans):
        return {
            "redknot": {
                "mode": mode,
                "namespace": "fixture",
                "allow_approximate": True,
                "chunks": [{"start": start, "end": end} for start, end in spans],
            }
        }

    def execute(self, mode, tokens, spans, q, kv, *, reason=None, positions=None):
        positions = torch.arange(q.shape[0]) if positions is None else positions
        with self.runtime.step(
            extra_args=self.args(mode, spans),
            token_ids=tokens,
            specs={self.name: self.spec},
            unsupported_reason=reason,
        ) as state:
            self.context.additional_kwargs[CONTEXT_KEY] = (self.runtime, state)
            try:
                result = self.layer.forward(q, kv, positions)
            finally:
                self.context.additional_kwargs.clear()
        return result, state

    def full_attention(self, q, kv):
        indices, lens = self.candidates(q.shape[0], kv.shape[0])
        return dsv4_sparse.selected_sparse_mla_reference(
            q,
            kv,
            indices,
            lens,
            self.layer.attn_sink,
            tuple(range(q.shape[0])),
            tuple(range(64)),
            scale=self.layer.scale,
        )

    def test_capture_keeps_native_output_and_stores_real_masked_z(self):
        q, kv = self.tensors(3)
        expected_attention = self.full_attention(q, kv)
        expected_z = self.z_oracle(expected_attention, torch.arange(3))
        expected = (expected_z.float() @ self.weight_b).bfloat16()
        result, state = self.execute("capture", [11, 12, 13], [(0, 3)], q, kv)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.layer.last_attention, expected_attention)
        self.assertEqual(self.selected_calls, [])
        self.assertEqual(len(self.wo_b_inputs), 1)
        self.assertEqual(len(self.project_calls), 2)
        masked = torch.zeros_like(expected_attention)
        masked[:, self.heads] = expected_attention[:, self.heads]
        torch.testing.assert_close(
            state.staged[self.name].z_off, self.z_oracle(masked, torch.arange(3))
        )
        self.assertEqual(self.runtime.counters["capture_committed"], 1)
        self.assertEqual(self.active_layers, [self.name])
        self.assertIsNone(state.active_layer)

    def test_nonprefix_multi_chunk_reuse_changes_attention_and_only_clean_z(self):
        captured = []
        for tokens in ([11, 12, 13], [21, 22, 23]):
            q, kv = self.tensors(3)
            _, state = self.execute("capture", tokens, [(0, 3)], q, kv)
            captured.append(state.staged[self.name].z_off.clone())
        tokens = [90, 91, 11, 12, 13, 92, 21, 22, 23, 99, 100]
        q, kv = self.tensors(len(tokens))
        full = self.full_attention(q, kv)
        selected_attention = full.clone()
        clean = torch.tensor([3, 4, 7, 8])
        selected_attention[clean[:, None], torch.tensor(self.heads)[None, :]] = 0
        expected_z = self.z_oracle(selected_attention, torch.arange(len(tokens)))
        expected_z[3:5] += captured[0][1:3]
        expected_z[7:9] += captured[1][1:3]
        counts_before = self.native_calls.copy()
        self.project_calls.clear()
        self.wo_b_inputs.clear()
        result, state = self.execute("reuse", tokens, [(2, 5), (6, 9)], q, kv)
        torch.testing.assert_close(self.layer.last_attention, selected_attention)
        torch.testing.assert_close(self.wo_b_inputs[0], expected_z, rtol=0, atol=0)
        torch.testing.assert_close(
            result, (expected_z.float() @ self.weight_b).bfloat16(), rtol=0, atol=0
        )
        dirty = (0, 1, 2, 5, 6, 9, 10)
        global_heads = tuple(head for head in range(64) if head not in self.heads)
        self.assertEqual(
            self.selected_calls,
            [
                (tuple(range(11)), global_heads),
                (dirty, self.heads),
            ],
        )
        self.assertEqual(len(self.project_calls), 1)
        self.assertEqual(len(self.wo_b_inputs), 1)
        self.assertEqual(self.native_calls["sparse"], counts_before["sparse"])
        for producer in (
            "swa_kv",
            "attention_compressor",
            "indexer_compressor",
            "indexer",
            "mqa",
        ):
            self.assertEqual(self.native_calls[producer], counts_before[producer] + 1)
            if producer != "mqa":
                torch.testing.assert_close(self.layer.produced[producer], kv)
        # This is not a fake cache-hit counter: projected values must differ.
        native_z = self.z_oracle(full, torch.arange(11))
        self.assertGreater(
            (expected_z[clean] - native_z[clean]).abs().max().item(), 0.01
        )
        torch.testing.assert_close(expected_z[list(dirty)], native_z[list(dirty)])
        self.assertEqual(state.projected_layers, {self.name})
        self.assertEqual(self.runtime.cache.stats()["pinned_entries"], 0)

    def test_cache_miss_retains_full_native_attention(self):
        q, kv = self.tensors(5)
        result, state = self.execute("reuse", [90, 11, 12, 13, 99], [(1, 4)], q, kv)
        self.assertEqual(state.reason, "cache_miss")
        expected = self.z_oracle(self.full_attention(q, kv), torch.arange(5))
        torch.testing.assert_close(
            result, (expected.float() @ self.weight_b).bfloat16()
        )
        self.assertEqual(self.selected_calls, [])
        self.assertEqual(self.native_calls["sparse"], 1)

    def test_decode_fallback_calls_original_methods_without_cache_merge(self):
        q, kv = self.tensors(3)
        self.execute("capture", [11, 12, 13], [(0, 3)], q, kv)
        self.project_calls.clear()
        self.wo_b_inputs.clear()
        decode_q, decode_kv = self.tensors(1)
        _, state = self.execute(
            "reuse",
            [11, 12, 13],
            [(0, 3)],
            decode_q,
            decode_kv,
            reason="not_full_prefill",
            positions=torch.tensor([3]),
        )
        self.assertEqual(state.mode, "native")
        self.assertEqual(self.native_calls["decode"], 1)
        self.assertEqual(self.native_calls["sparse"], 1)  # Capture only.
        self.assertEqual(self.selected_calls, [])
        self.assertEqual(len(self.project_calls), 1)
        self.assertEqual(len(self.wo_b_inputs), 1)
        self.assertEqual(self.active_layers[-1], None)

    def test_active_layer_restored_and_capture_not_published_on_exception(self):
        q, kv = self.tensors(3)
        self.raise_mqa = True
        with self.assertRaisesRegex(RuntimeError, "native MQA failed"):
            with self.runtime.step(
                extra_args=self.args("capture", [(0, 3)]),
                token_ids=[11, 12, 13],
                specs={self.name: self.spec},
            ) as state:
                state.active_layer = "outer-layer"
                self.context.additional_kwargs[CONTEXT_KEY] = self.runtime, state
                try:
                    self.layer.forward(q, kv, torch.arange(3))
                finally:
                    self.assertEqual(state.active_layer, "outer-layer")
                    self.context.additional_kwargs.clear()
        self.assertEqual(self.runtime.counters["capture_aborted"], 1)
        self.assertEqual(self.runtime.cache.stats()["entries"], 0)
        self.assertEqual(self.wo_b_inputs, [])

    def test_invalid_artifact_falls_back_before_skipping_sparse_attention(self):
        q, kv = self.tensors(3)
        _, captured = self.execute("capture", [11, 12, 13], [(0, 3)], q, kv)
        key = self.runtime.chunk_key("fixture", [11, 12, 13])
        original = captured.staged[self.name]
        invalid = (
            replace(original, policy_key="different policy"),
            replace(original, z_off=original.z_off.float()),
            replace(original, source_positions=original.source_positions.float()),
        )
        for item in invalid:
            with self.subTest(source_dtype=item.source_positions.dtype):
                self.runtime.cache.put(
                    key, DSV4Chunk(3, {self.name: item}), item.nbytes
                )
                online_q, online_kv = self.tensors(5)
                _, state = self.execute(
                    "reuse", [90, 11, 12, 13, 99], [(1, 4)], online_q, online_kv
                )
                self.assertEqual(state.reason, "cache_contract")
                self.assertEqual(self.selected_calls, [])
        self.assertEqual(self.runtime.cache.stats()["pinned_entries"], 0)


if __name__ == "__main__":
    unittest.main()
