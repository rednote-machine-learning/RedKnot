"""Run the actual adapter with CPU tensors and a deterministic attention oracle.

Only vLLM's platform imports and native FlashAttention call are substituted.
These tests do not certify CUDA kernels or end-to-end model quality.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from vllm_redknot.cache import CacheManager
from vllm_redknot.ops import relocate_rope
from vllm_redknot.runner import CONTEXT_KEY
from vllm_redknot.runtime import LayerSpec, RedKnotRuntime, RedKnotSettings


def dense(q, k, v):
    group = q.shape[1] // k.shape[1]
    keys = k.repeat_interleave(group, dim=1)
    values = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), keys.float()) * q.shape[-1] ** -0.5
    qpos = torch.arange(k.shape[0] - q.shape[0], k.shape[0])
    mask = torch.arange(k.shape[0])[None, :] <= qpos[:, None]
    scores.masked_fill_(~mask, -float("inf"))
    result = torch.einsum("hts,shd->thd", scores.softmax(-1), values.float())
    return result.to(q.dtype)


@unittest.skipUnless(
    torch is not None, "CPU torch is required for numerical backend contract tests"
)
class BackendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.context = NS(additional_kwargs={}, slot_mapping={})
        self.calls = []
        owner = self

        class Native:
            native_calls = 0

            def forward(
                self, layer, query, key, value, kv_cache, metadata, output, *unused
            ):
                self.native_calls += 1
                output.copy_(dense(query, key, value))
                return output

        def attention(**kwargs):
            owner.calls.append((kwargs["q"].shape, kwargs["k"].shape))
            kwargs["out"].copy_(dense(kwargs["q"], kwargs["k"], kwargs["v"]))
            return kwargs["out"]

        names = [
            "vllm",
            "vllm.v1",
            "vllm.v1.attention",
            "vllm.v1.attention.backends",
            "vllm.forward_context",
            "vllm.v1.attention.backend",
            "vllm.v1.attention.backends.flash_attn",
        ]
        modules = {name: types.ModuleType(name) for name in names}
        modules["vllm.forward_context"].get_forward_context = lambda: self.context
        modules["vllm.v1.attention.backend"].AttentionCGSupport = NS(NEVER="never")
        flash = modules["vllm.v1.attention.backends.flash_attn"]
        flash.FlashAttentionBackend = type("FlashAttentionBackend", (), {})
        flash.FlashAttentionMetadataBuilder = type(
            "FlashAttentionMetadataBuilder", (), {}
        )
        flash.FlashAttentionImpl = Native
        flash.flash_attn_varlen_func = attention
        source = Path(__file__).parents[1] / "vllm_redknot" / "vllm_backend.py"
        module_spec = importlib.util.spec_from_file_location(
            "vllm_redknot._backend_contract_test", source
        )
        module = importlib.util.module_from_spec(module_spec)
        with patch.dict(sys.modules, modules):
            module_spec.loader.exec_module(module)
        self.impl = module.RedKnotImpl()
        self.impl.scale = 8**-0.5
        self.impl.vllm_flash_attn_version = 2
        self.layer = NS(layer_name="model.layers.0.self_attn.attn")
        self.spec = LayerSpec((1,), 4, 2, 8, "torch.float16")
        self.settings = RedKnotSettings.from_mapping(
            {
                "local_heads": {"0": [1]},
                "model_revision": "fixture",
                "allow_approximate": True,
                "boundary_tokens": 1,
                "rotary_dim": 8,
            }
        )
        self.runtime = RedKnotRuntime(self.settings, CacheManager(1 << 20), "fixture")

    def args(self, mode, start, end):
        return {
            "redknot": {
                "mode": mode,
                "namespace": "fixture",
                "chunks": [{"start": start, "end": end}],
                "allow_approximate": True,
            }
        }

    def tensors(self, length):
        return (
            torch.randn(length, 4, 8).half(),
            torch.randn(length, 2, 8).half(),
            torch.randn(length, 2, 8).half(),
        )

    def kv_cache(self, k, v, slots):
        cache = torch.full((4, 2, 4, 16), -123, dtype=k.dtype)
        for row, slot in enumerate(slots.tolist()):
            cache[slot // 4, :, slot % 4, :8] = k[row]
            cache[slot // 4, :, slot % 4, 8:] = v[row]
        return cache

    def call(self, mode, tokens, span, q, k, v, slots):
        out = torch.empty_like(q)
        cache = self.kv_cache(k, v, slots)
        self.context.slot_mapping[self.layer.layer_name] = slots
        with self.runtime.step(
            extra_args=self.args(mode, *span),
            token_ids=tokens,
            specs={self.layer.layer_name: self.spec},
        ) as state:
            self.context.additional_kwargs[CONTEXT_KEY] = self.runtime, state
            try:
                self.impl.forward(self.layer, q, k, v, cache, None, out)
            finally:
                self.context.additional_kwargs.clear()
        return out, cache

    def test_cache_miss_executes_native_attention_without_kv_overwrite(self):
        q, k, v = self.tensors(6)
        slots = torch.tensor([8, 9, 10, 11, 0, 1])
        original = self.kv_cache(k, v, slots)
        out, cache = self.call(
            "reuse", [90, 91, 11, 12, 13, 99], (2, 5), q, k, v, slots
        )
        self.assertEqual(self.impl.native_calls, 1)
        self.assertFalse(self.calls)
        torch.testing.assert_close(out, dense(q, k, v))
        torch.testing.assert_close(cache, original)

    def test_reuse_skips_clean_queries_and_writes_relocated_local_kv(self):
        offline_q, offline_k, offline_v = self.tensors(3)
        offline_out, _ = self.call(
            "capture",
            [11, 12, 13],
            (0, 3),
            offline_q,
            offline_k,
            offline_v,
            torch.tensor([4, 5, 6]),
        )
        q, k, v = self.tensors(6)
        slots = torch.tensor([8, 9, 10, 11, 0, 1])
        original = self.kv_cache(k, v, slots)
        out, cache = self.call(
            "reuse", [90, 91, 11, 12, 13, 99], (2, 5), q, k, v, slots
        )
        self.assertEqual(
            self.impl.native_calls, 1, "warm reuse must not call full native attention"
        )
        self.assertEqual(
            [(tuple(qs), tuple(ks)) for qs, ks in self.calls],
            [((6, 2, 8), (6, 1, 8)), ((3, 2, 8), (3, 1, 8)), ((1, 2, 8), (6, 1, 8))],
        )
        mixed_k = k[:, 1:2].clone()
        mixed_v = v[:, 1:2].clone()
        relocated = relocate_rope(
            offline_k[1:3, 1:2], torch.arange(1, 3), torch.arange(3, 5), 8, 10000
        )
        mixed_k[3:5] = relocated
        mixed_v[3:5] = offline_v[1:3, 1:2]
        expected = dense(q, k, v)
        local_expected = dense(q[:, 2:4], mixed_k, mixed_v)
        local_expected[3:5] = offline_out[1:3, 2:4]
        expected[:, 2:4] = local_expected
        torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
        for row, slot in enumerate(slots.tolist()):
            block, offset = slot // 4, slot % 4
            torch.testing.assert_close(
                cache[block, 0, offset], original[block, 0, offset]
            )
            if row in (3, 4):
                torch.testing.assert_close(
                    cache[block, 1, offset, :8], relocated[row - 3, 0]
                )
                torch.testing.assert_close(
                    cache[block, 1, offset, 8:], offline_v[row - 2, 1]
                )
            else:
                torch.testing.assert_close(
                    cache[block, 1, offset], original[block, 1, offset]
                )
        counts = self.runtime.stats()["runtime"]
        self.assertEqual(counts["reused_local_query_rows"], 4)
        self.assertEqual(counts["computed_global_query_rows"], 12)
        self.assertEqual(counts["computed_local_query_rows"], 8)

    def test_capture_owns_cpu_copies_after_online_buffers_are_mutated(self):
        q, k, v = self.tensors(3)
        out, _ = self.call(
            "capture", [11, 12, 13], (0, 3), q, k, v, torch.tensor([4, 5, 6])
        )
        original_k, original_v, original_out = k.clone(), v.clone(), out.clone()
        k.zero_()
        v.zero_()
        out.zero_()
        key = self.runtime.chunk_key("fixture", [11, 12, 13])
        with self.runtime.cache.lease([key]) as cached:
            payload = cached[key].layers[self.layer.layer_name]
            torch.testing.assert_close(payload.keys, original_k[:, 1:2])
            torch.testing.assert_close(payload.values, original_v[:, 1:2])
            torch.testing.assert_close(payload.outputs, original_out[:, 2:4])


if __name__ == "__main__":
    unittest.main()
