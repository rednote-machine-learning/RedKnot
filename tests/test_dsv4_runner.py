"""No-GPU tests for the actual DSV4 runner hooks and full-prefill guards."""

import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from vllm_redknot.dsv4_runner import (
    CONTEXT_KEY,
    FLASH_REVISION,
    install_dsv4_runner,
    prefill_reason,
    resolve_dsv4_layers,
    validate_dsv4_config,
)
from vllm_redknot.dsv4_runtime import DSV4LayerSpec
from vllm_redknot.runtime import RedKnotSettings


class Vector:
    def __init__(self, values):
        self.values = values
        self.ndim = 1

    def tolist(self):
        return self.values

    def numel(self):
        return len(self.values)


def fixture_config():
    hf = NS(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=43,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=8,
        # Native YaRN must not be rejected by the MHA static-RoPE guard.
        rope_scaling={"rope_type": "yarn", "factor": 16},
    )
    hf.to_dict = lambda: {k: v for k, v in vars(hf).items() if not callable(v)}
    return NS(
        model_config=NS(
            hf_config=hf, dtype="torch.bfloat16", enforce_eager=True, model="fixture"
        ),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        use_v2_model_runner=False,
        scheduler_config=NS(max_num_seqs=1, enable_chunked_prefill=False),
        cache_config=NS(enable_prefix_caching=False, cache_dtype="auto"),
        attention_config=NS(backend=NS(name="FLASHMLA_SPARSE_DSV4")),
    )


def settings():
    return RedKnotSettings.from_mapping(
        {
            "local_heads": {"3": [1, 2, 3]},
            "model_revision": FLASH_REVISION,
            "allow_approximate": True,
        }
    )


def layer_fixture():
    cls = type("DeepseekV4FlashMLAAttention", (), {})
    cls.__module__ = "vllm.models.deepseek_v4.nvidia.flashmla"
    layer = cls()
    for key, value in dict(
        n_local_heads=64,
        head_dim=512,
        n_local_groups=8,
        o_lora_rank=1024,
        compress_ratio=4,
        max_image_tokens=0,
        nope_head_dim=448,
        rope_head_dim=64,
        _einsum_recipe=(1, 1, 128),
        swa_cache_layer=NS(prefix="swa"),
    ).items():
        setattr(layer, key, value)
    return layer


def context_fixture():
    layer = layer_fixture()
    swa = NS(
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=3,
        query_start_loc_cpu=Vector([0, 3]),
        prefill_seq_lens_cpu=Vector([3]),
        prefill_query_lens_cpu=Vector([3]),
        prefill_left_visible=None,
        prefill_right_visible=None,
    )
    return NS(
        attn_metadata={"swa": swa, "model.layers.3.attn": NS(num_actual_tokens=3)},
        no_compile_layers={"model.layers.3.attn": layer},
        additional_kwargs={},
    )


class DSV4RunnerTests(unittest.TestCase):
    def test_native_yarn_configuration_accepted(self):
        validate_dsv4_config(fixture_config(), settings())

    def test_unsupported_global_modes_rejected(self):
        changes = (
            ("parallel_config", "tensor_parallel_size", 8),
            ("parallel_config", "enable_dbo", True),
            ("scheduler_config", "max_num_seqs", 2),
            ("scheduler_config", "enable_chunked_prefill", True),
            ("cache_config", "enable_prefix_caching", True),
            ("cache_config", "cache_dtype", "bfloat16"),
            ("model_config", "enforce_eager", False),
            ("model_config", "dtype", "torch.float16"),
            ("attention_config", "backend", NS(name="CUSTOM")),
        )
        for group, field, value in changes:
            with self.subTest(field=field):
                config = fixture_config()
                setattr(getattr(config, group), field, value)
                with self.assertRaises(ValueError):
                    validate_dsv4_config(config, settings())

    def test_non_flash_geometry_and_revision_rejected(self):
        config = fixture_config()
        config.model_config.hf_config.num_hidden_layers = 61
        with self.assertRaisesRegex(ValueError, "geometry"):
            validate_dsv4_config(config, settings())
        bad = RedKnotSettings.from_mapping(
            {"local_heads": {"3": [1]}, "model_revision": "unknown"}
        )
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_dsv4_config(fixture_config(), bad)

    def test_resolve_only_native_attention_not_compressor(self):
        context = context_fixture()
        context.no_compile_layers["model.layers.3.attn.compressor"] = NS()
        specs = resolve_dsv4_layers(settings(), context.no_compile_layers)
        self.assertEqual(
            specs, {"model.layers.3.attn": DSV4LayerSpec((1, 2, 3), 64, 8, 1024, 4)}
        )

    def test_missing_or_changed_projection_layout_rejected(self):
        with self.assertRaises(ValueError):
            resolve_dsv4_layers(settings(), {})
        layer = layer_fixture()
        layer._einsum_recipe = (1, 1, 4096)
        with self.assertRaisesRegex(ValueError, "128-dim"):
            resolve_dsv4_layers(settings(), {"model.layers.3.attn": layer})

    def test_full_prefill_guard(self):
        runner = NS(input_batch=NS(req_ids=["request"]))
        context = context_fixture()
        specs = resolve_dsv4_layers(settings(), context.no_compile_layers)
        self.assertIsNone(prefill_reason(runner, context, Vector([0, 1, 2]), 3, specs))
        self.assertEqual(
            prefill_reason(runner, context, Vector([3]), 3, specs), "not_full_prefill"
        )
        context.attn_metadata["swa"].prefill_left_visible = Vector([0])
        self.assertEqual(
            prefill_reason(runner, context, Vector([0, 1, 2]), 3, specs),
            "multimodal_visibility",
        )

    def test_hook_idempotence_context_cleanup_native_decode_and_exception(self):
        context = context_fixture()
        calls = []

        class Runner:
            def __init__(self, config, device):
                self.input_batch = NS(req_ids=["r"])
                self.requests = {
                    "r": NS(
                        prompt_token_ids=[11, 12, 13],
                        sampling_params=NS(extra_args=None),
                    )
                }
                self.fail = False

            def _model_forward(self, **kwargs):
                calls.append(context.additional_kwargs[CONTEXT_KEY][1].mode)
                if self.fail:
                    raise RuntimeError("native failure")
                return "native-result"

        names = [
            "vllm",
            "vllm.forward_context",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu_model_runner",
        ]
        modules = {name: types.ModuleType(name) for name in names}
        modules["vllm.forward_context"].get_forward_context = lambda: context
        modules[names[-1]].GPUModelRunner = Runner
        with patch.dict(sys.modules, modules):
            install_dsv4_runner(settings())
            wrapped = Runner._model_forward
            install_dsv4_runner(settings())
            self.assertIs(Runner._model_forward, wrapped)
            runner = Runner(fixture_config(), "cpu")
            self.assertEqual(
                runner._model_forward(positions=Vector([0, 1, 2])), "native-result"
            )
            self.assertNotIn(CONTEXT_KEY, context.additional_kwargs)
            sentinel = object()
            context.additional_kwargs[CONTEXT_KEY] = sentinel
            runner.fail = True
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                runner._model_forward(positions=Vector([3]))
            self.assertIs(context.additional_kwargs[CONTEXT_KEY], sentinel)
            self.assertEqual(calls, ["native", "native"])
            self.assertEqual(
                runner.redknot_runtime.counters["fallback:not_full_prefill"], 1
            )


if __name__ == "__main__":
    unittest.main()
