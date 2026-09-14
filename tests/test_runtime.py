"""CPU contract tests for cache admission and request-wide fallback."""

import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from vllm_redknot.cache import CacheManager
from vllm_redknot.runner import (
    CONTEXT_KEY,
    _step_reason,
    install_runner_hooks,
    resolve_layer_specs,
    validate_engine_config,
)
from vllm_redknot.runtime import (
    ChunkPayload,
    ChunkSpan,
    LayerPayload,
    LayerSpec,
    RedKnotRuntime,
    RedKnotSettings,
    RequestPlan,
    clean_and_dirty_runs,
)


class ShapeTensor:
    def __init__(self, shape, dtype="torch.float16"):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)


class Array:
    def __init__(self, values):
        self.values = values
        self.ndim = 2 if values and isinstance(values[0], list) else 1

    def numel(self):
        return len(self.values)

    def tolist(self):
        return self.values

    def __getitem__(self, index):
        item = self.values[index]
        return Array(item) if isinstance(item, list) else item


def settings(**kwargs):
    return RedKnotSettings.from_mapping(
        {
            "model_revision": "immutable-checkpoint-hash",
            "local_heads": {"0": [1]},
            "boundary_tokens": 1,
            "allow_approximate": True,
            **kwargs,
        }
    )


def request(mode, chunks, **kwargs):
    return {
        "redknot": {
            "mode": mode,
            "namespace": "test",
            "chunks": [{"start": a, "end": b} for a, b in chunks],
            "allow_approximate": True,
            **kwargs,
        }
    }


SPEC = LayerSpec((1,), 4, 2, 8, "torch.float16")
NAME = "model.layers.0.self_attn.attn"


def layer_payload(length=3, spec=SPEC):
    return LayerPayload(
        spec,
        ShapeTensor((length, len(spec.local_heads), 8)),
        ShapeTensor((length, len(spec.local_heads), 8)),
        ShapeTensor((length, len(spec.local_query_heads), 8)),
        100,
    )


def engine_config(**hf_updates):
    hf = NS(
        architectures=["Qwen3ForCausalLM"],
        rope_theta=10000,
        head_dim=8,
        num_attention_heads=4,
        hidden_size=32,
        **hf_updates,
    )
    return NS(
        model_config=NS(hf_config=hf, enforce_eager=True, dtype="torch.float16"),
        parallel_config=NS(),
        use_v2_model_runner=False,
        scheduler_config=NS(max_num_seqs=1, enable_chunked_prefill=False),
        cache_config=NS(enable_prefix_caching=False, cache_dtype="auto"),
    )


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = RedKnotRuntime(
            settings(), CacheManager(1000), {"model": "fixture"}
        )
        self.specs = {NAME: SPEC}

    def capture(self, tokens=(11, 12, 13)):
        with self.runtime.step(
            extra_args=request("capture", [(0, len(tokens))]),
            token_ids=tokens,
            specs=self.specs,
        ) as state:
            state.staged[NAME] = layer_payload(len(tokens))

    def test_nested_transport_and_capture_span_validation(self):
        raw = request("capture", [(0, 3)])
        plan = RequestPlan.parse({"kv_transfer_params": raw}, 3)
        self.assertEqual(plan.chunks, (ChunkSpan(0, 3),))
        with self.assertRaises(ValueError):
            RequestPlan.parse(request("capture", [(1, 3)]), 3)

    def test_overlap_boolean_indices_and_unknown_modes_are_rejected(self):
        for mode, chunks in [
            ("reuse", [(0, 3), (2, 4)]),
            ("reuse", [(True, 3)]),
            ("mystery", [(0, 3)]),
        ]:
            with self.subTest(mode=mode, chunks=chunks), self.assertRaises(ValueError):
                RequestPlan.parse(request(mode, chunks), 5)

    def test_request_caps_bound_chunk_work_and_namespace_size(self):
        with self.assertRaisesRegex(ValueError, "at most 8 chunks"):
            RequestPlan.parse(request("reuse", [(i, i + 1) for i in range(9)]), 9)
        with self.assertRaisesRegex(ValueError, "256 characters"):
            RequestPlan.parse(request("reuse", [(0, 1)], namespace="x" * 257), 1)
        allowed = RequestPlan.parse(
            request("reuse", [(i, i + 1) for i in range(8)], namespace="x" * 256), 8
        )
        self.assertEqual(len(allowed.chunks), 8)

    def test_nonprefix_span_restores_only_clean_rows_and_keeps_gaps_dirty(self):
        clean, dirty = clean_and_dirty_runs([ChunkSpan(2, 6), ChunkSpan(8, 12)], 14, 1)
        self.assertEqual(clean, ((3, 6, 0), (9, 12, 1)))
        self.assertEqual(dirty, ((0, 3), (6, 9), (12, 14)))
        rows = [r for a, b, _ in clean for r in range(a, b)] + [
            r for a, b in dirty for r in range(a, b)
        ]
        self.assertEqual(sorted(rows), list(range(14)))

    def test_short_chunks_do_not_create_clean_rows(self):
        self.assertEqual(
            clean_and_dirty_runs([ChunkSpan(4, 6)], 8, 128), ((), ((0, 8),))
        )

    def test_capture_is_not_published_until_all_layers_succeed(self):
        tokens = [11, 12, 13]
        key = self.runtime.chunk_key("test", tokens)
        with self.runtime.step(
            extra_args=request("capture", [(0, 3)]), token_ids=tokens, specs=self.specs
        ) as state:
            state.staged[NAME] = layer_payload()
            with self.runtime.cache.lease([key]) as visible:
                self.assertIsNone(visible)
        with self.runtime.cache.lease([key]) as visible:
            self.assertEqual(visible[key].length, 3)

    def test_oversized_capture_falls_back_before_allocating_snapshots(self):
        runtime = RedKnotRuntime(
            settings(max_cache_bytes=100), CacheManager(100), "fixture"
        )
        with runtime.step(
            extra_args=request("capture", [(0, 3)]),
            token_ids=[11, 12, 13],
            specs=self.specs,
        ) as state:
            self.assertEqual((state.mode, state.reason), ("native", "capture_capacity"))
            self.assertFalse(state.staged)
        self.assertEqual(runtime.cache.stats()["bytes"], 0)

    def test_incomplete_or_failed_capture_never_publishes(self):
        with self.runtime.step(
            extra_args=request("capture", [(0, 3)]),
            token_ids=[11, 12, 13],
            specs=self.specs,
        ):
            pass
        self.assertEqual(self.runtime.cache.stats()["entries"], 0)
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with self.runtime.step(
                extra_args=request("capture", [(0, 3)]),
                token_ids=[11, 12, 13],
                specs=self.specs,
            ) as state:
                state.staged[NAME] = layer_payload()
                raise RuntimeError("forward failed")
        self.assertEqual(self.runtime.cache.stats()["entries"], 0)

    def test_content_match_is_reusable_at_a_different_nonprefix_position(self):
        self.capture()
        with self.runtime.step(
            extra_args=request("reuse", [(2, 5)]),
            token_ids=[90, 91, 11, 12, 13, 99],
            specs=self.specs,
        ) as state:
            self.assertEqual(state.mode, "reuse")
            self.assertEqual(state.clean_runs, ((3, 5, 0),))
            self.assertEqual(state.dirty_runs, ((0, 3), (5, 6)))
            self.assertEqual(self.runtime.cache.stats()["pinned_entries"], 1)
        self.assertEqual(self.runtime.cache.stats()["pinned_entries"], 0)

    def test_one_missing_chunk_falls_back_for_entire_request(self):
        self.capture()
        with self.runtime.step(
            extra_args=request("reuse", [(0, 3), (3, 6)]),
            token_ids=[11, 12, 13, 21, 22, 23],
            specs=self.specs,
        ) as state:
            self.assertEqual((state.mode, state.reason), ("native", "cache_miss"))
            self.assertFalse(state.cached)
            self.assertEqual(state.modified_layers, 0)

    def test_wrong_payload_geometry_falls_back_before_a_layer_runs(self):
        key = self.runtime.chunk_key("test", [11, 12, 13])
        bad = layer_payload()
        bad.keys.shape = (2, 1, 8)
        self.runtime.cache.put(key, ChunkPayload(3, {NAME: bad}), 100)
        with self.runtime.step(
            extra_args=request("reuse", [(0, 3)]),
            token_ids=[11, 12, 13],
            specs=self.specs,
        ) as state:
            self.assertEqual(state.reason, "cache_contract")

    def test_cache_identity_separates_namespace_model_and_policy(self):
        key = self.runtime.chunk_key("test", [11])
        self.assertNotEqual(key, self.runtime.chunk_key("other", [11]))
        other = RedKnotRuntime(settings(), CacheManager(1000), {"model": "other"})
        self.assertNotEqual(key, other.chunk_key("test", [11]))
        other = RedKnotRuntime(
            settings(boundary_tokens=2), CacheManager(1000), {"model": "fixture"}
        )
        self.assertNotEqual(key, other.chunk_key("test", [11]))

    def test_dual_approximation_consent_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "server and request"):
            with self.runtime.step(
                extra_args=request("reuse", [(0, 3)], allow_approximate=False),
                token_ids=[11, 12, 13],
                specs=self.specs,
            ):
                pass

    def test_decode_and_recomputed_use_native_without_pinning(self):
        self.capture()
        with self.runtime.step(
            extra_args=request("reuse", [(0, 3)]),
            token_ids=[11, 12, 13],
            specs=self.specs,
            unsupported_reason="not_full_prefill",
        ) as state:
            self.assertEqual(state.mode, "native")
            self.assertEqual(self.runtime.cache.stats()["pinned_entries"], 0)
        with self.runtime.step(
            extra_args=request("recomputed", []),
            token_ids=[11, 12, 13],
            specs=self.specs,
        ) as state:
            self.assertEqual(state.mode, "recomputed")


class RunnerGuardTests(unittest.TestCase):
    def test_supported_model_resolves_partial_rotary_dimension(self):
        resolved = validate_engine_config(
            engine_config(partial_rotary_factor=0.5), settings()
        )
        self.assertEqual(resolved.rotary_dim, 4)

    def test_explicit_rope_dim_has_precedence_over_partial_factor(self):
        config = engine_config(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "rope_dim": 2,
                "partial_rotary_factor": 1.0,
            }
        )
        self.assertEqual(validate_engine_config(config, settings()).rotary_dim, 2)

    def test_fourier_rope_is_not_mistaken_for_default_static_rope(self):
        config = engine_config(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "use_fope": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "RoPE"):
            validate_engine_config(config, settings())

    def test_deepseek_mla_and_non_eager_startup_are_explicitly_rejected(self):
        config = engine_config()
        config.model_config.hf_config.architectures = ["DeepseekV4ForCausalLM"]
        with self.assertRaisesRegex(ValueError, "DeepSeek V4"):
            validate_engine_config(config, settings())
        config = engine_config()
        config.model_config.enforce_eager = False
        with self.assertRaisesRegex(ValueError, "enforce-eager"):
            validate_engine_config(config, settings())

    def test_prefix_caching_and_theta_mismatch_are_rejected(self):
        config = engine_config()
        config.cache_config.enable_prefix_caching = True
        with self.assertRaisesRegex(ValueError, "prefix-caching"):
            validate_engine_config(config, settings())
        config = engine_config()
        config.model_config.hf_config.rope_theta = 1000000
        with self.assertRaisesRegex(ValueError, "rope_theta"):
            validate_engine_config(config, settings())

    def test_exact_layer_resolution_preserves_global_only_layers(self):
        impl = NS(
            redknot_implementation=True,
            num_kv_heads=2,
            num_heads=4,
            head_size=8,
            sliding_window=(-1, -1),
            alibi_slopes=None,
            logits_soft_cap=0,
            sinks=None,
            kv_sharing_target_layer_name=None,
        )
        layers = {
            NAME: NS(impl=impl, dtype="torch.float16"),
            "model.layers.1.self_attn.attn": NS(impl=impl, dtype="torch.float16"),
        }
        self.assertEqual(resolve_layer_specs(settings(), layers), {NAME: SPEC})

    def test_physical_slot_contract_checked_before_reuse(self):
        metadata = NS(
            redknot_query_start_loc_cpu=(0, 3),
            num_actual_tokens=3,
            seq_lens=Array([3]),
            use_cascade=False,
            causal=True,
            block_table=Array([[1, 2]]),
        )
        context = NS(
            attn_metadata={NAME: metadata},
            slot_mapping={NAME: Array([2, 3, 4])},
            no_compile_layers={NAME: NS(kv_cache=ShapeTensor((4, 2, 2, 16)))},
        )
        runner = NS(input_batch=NS(req_ids=["request"]))
        self.assertIsNone(
            _step_reason(runner, context, Array([0, 1, 2]), 3, {NAME: SPEC})
        )
        context.slot_mapping[NAME] = Array([2, 3, 3])
        self.assertEqual(
            _step_reason(runner, context, Array([0, 1, 2]), 3, {NAME: SPEC}),
            "invalid_kv_slots",
        )

    def test_runner_context_is_restored_when_model_forward_raises(self):
        config = engine_config()
        config.model_config.model = "fixture"
        config.model_config.hf_config.to_dict = lambda: {"model_type": "qwen3"}
        impl = NS(
            redknot_implementation=True,
            num_kv_heads=2,
            num_heads=4,
            head_size=8,
            sliding_window=(-1, -1),
            alibi_slopes=None,
            logits_soft_cap=0,
            sinks=None,
            kv_sharing_target_layer_name=None,
        )
        previous = object()
        context = NS(
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={CONTEXT_KEY: previous, "unrelated": 12},
            no_compile_layers={NAME: NS(impl=impl, dtype="torch.float16")},
        )

        class FakeRunner:
            def __init__(self, config, device):
                self.input_batch = NS(req_ids=["one"])
                self.requests = {
                    "one": NS(
                        prompt_token_ids=[11, 12, 13],
                        sampling_params=NS(extra_args=None),
                    )
                }

            def get_model(self):
                rope_cls = type(
                    "RotaryEmbedding",
                    (),
                    {
                        "__module__": (
                            "vllm.model_executor.layers.rotary_embedding.base"
                        ),
                        "base": 10000,
                        "rotary_dim": 8,
                        "is_neox_style": True,
                    },
                )
                return NS(
                    named_modules=lambda: [
                        ("model.layers.0.self_attn", NS(rotary_emb=rope_cls()))
                    ]
                )

            def _model_forward(self, **kwargs):
                assert context.additional_kwargs[CONTEXT_KEY] is not previous
                raise RuntimeError("model failed")

        ctx_module = types.ModuleType("vllm.forward_context")
        ctx_module.get_forward_context = lambda: context
        runner_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")
        runner_module.GPUModelRunner = FakeRunner
        with patch.dict(
            sys.modules,
            {
                "vllm.forward_context": ctx_module,
                "vllm.v1.worker.gpu_model_runner": runner_module,
            },
        ):
            install_runner_hooks(settings())
            runner = FakeRunner(config, "cpu")
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                runner._model_forward()
        self.assertIs(context.additional_kwargs[CONTEXT_KEY], previous)
        self.assertEqual(context.additional_kwargs["unrelated"], 12)


if __name__ == "__main__":
    unittest.main()
