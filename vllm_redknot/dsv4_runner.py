"""Opt-in single-request Flash-0731 runner hooks; native model state stays online."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import wraps
from typing import Any

from .cache import CacheManager
from .dsv4_runtime import DSV4LayerSpec, DSV4Runtime
from .runtime import RedKnotSettings

CONTEXT_KEY = "vllm_redknot_dsv4"
FLASH_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
ATTENTION_CLASS = "DeepseekV4FlashMLAAttention"


def validate_dsv4_config(config: Any, settings: RedKnotSettings) -> None:
    """Reject unsupported global modes before a worker allocates model memory."""
    model, parallel = config.model_config, config.parallel_config
    hf = model.hf_config
    if getattr(hf, "architectures", None) != ["DeepseekV4ForCausalLM"]:
        raise ValueError("DSV4 mode requires DeepseekV4ForCausalLM")
    expected = {
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
    }
    if any(getattr(hf, key, None) != value for key, value in expected.items()):
        raise ValueError("Only the Flash-0731 geometry has been adapted")
    if settings.model_revision != FLASH_REVISION:
        raise ValueError("DSV4 mode requires the pinned Flash-0731 checkpoint revision")
    if not model.enforce_eager or config.use_v2_model_runner:
        raise ValueError("DSV4 reuse requires eager V1 model runner")
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"DSV4 first integration requires {name}=1")
    if getattr(parallel, "enable_dbo", False):
        raise ValueError("DSV4 reuse does not support DBO")
    if config.scheduler_config.max_num_seqs != 1:
        raise ValueError("DSV4 first integration requires max_num_seqs=1")
    if config.scheduler_config.enable_chunked_prefill:
        raise ValueError("Disable chunked prefill for DSV4 reuse")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Disable prefix caching for DSV4 reuse")
    for name in ("speculative_config", "lora_config", "kv_transfer_config"):
        if getattr(config, name, None) is not None:
            raise ValueError(f"DSV4 reuse does not support {name}")
    if str(model.dtype) != "torch.bfloat16":
        raise ValueError(
            "DSV4 requires bfloat16 activations with native quantized weights"
        )
    if getattr(hf, "vision_n_layers", 0):
        raise ValueError("Only text-only Flash is adapted")
    if config.cache_config.cache_dtype not in {"auto", "fp8", "fp8_ds_mla"}:
        raise ValueError("DSV4 FlashMLA requires native fp8_ds_mla KV layout")
    backend = getattr(config.attention_config.backend, "name", "")
    if backend not in {"FLASHMLA_SPARSE_DSV4", "FLASHMLA_SPARSE"}:
        raise ValueError("DSV4 reuse requires FLASHMLA_SPARSE_DSV4 attention")


def resolve_dsv4_layers(
    settings: RedKnotSettings, layers: Mapping[str, Any]
) -> dict[str, DSV4LayerSpec]:
    selected = {}
    candidates = {
        name: layer
        for name, layer in layers.items()
        if type(layer).__name__ == ATTENTION_CLASS
        and type(layer).__module__ == "vllm.models.deepseek_v4.nvidia.flashmla"
    }
    for selector, local in settings.local_heads.items():
        if selector in candidates:
            names = [selector]
        elif selector.isdigit():
            names = [
                name
                for name in candidates
                if re.search(rf"(?:^|\.)layers\.{int(selector)}\.", name)
            ]
        else:
            names = []
        if len(names) != 1 or names[0] in selected:
            raise ValueError(
                f"DSV4 selector {selector!r} must match one distinct layer"
            )
        name = names[0]
        layer = candidates[name]
        if layer.n_local_heads != 64 or layer.head_dim != 512:
            raise ValueError("Unexpected Flash attention geometry")
        if any(head >= layer.n_local_heads for head in local):
            raise ValueError("DSV4 local logical query-head index out of range")
        if layer.compress_ratio not in {1, 4, 128} or layer.max_image_tokens:
            raise ValueError("Unsupported compressed or multimodal attention")
        if layer.nope_head_dim != 448 or layer.rope_head_dim != 64:
            raise ValueError("DSV4 inverse-RoPE tail geometry changed")
        if layer._einsum_recipe[2] != 128:
            raise ValueError("DSV4 FP8 projection must use independent 128-dim blocks")
        selected[name] = DSV4LayerSpec(
            local,
            layer.n_local_heads,
            layer.n_local_groups,
            layer.o_lora_rank,
            layer.compress_ratio,
        )
    return selected


def prefill_reason(
    runner: Any, context: Any, positions: Any, n: int, specs
) -> str | None:
    if len(runner.input_batch.req_ids) != 1:
        return "batch_size"
    if not isinstance(context.attn_metadata, dict):
        return "metadata_unavailable"
    if positions is None or positions.ndim != 1 or positions.numel() != n or n < 2:
        return "not_full_prefill"
    if positions.tolist() != list(range(n)):
        return "not_full_prefill"
    for name, spec in specs.items():
        layer = context.no_compile_layers[name]
        swa = context.attn_metadata.get(layer.swa_cache_layer.prefix)
        if swa is None or (
            swa.num_prefills != 1
            or swa.num_decodes != 0
            or swa.num_decode_tokens != 0
            or swa.num_prefill_tokens != n
            or swa.query_start_loc_cpu is None
            or swa.query_start_loc_cpu.tolist() != [0, n]
            or swa.prefill_seq_lens_cpu is None
            or swa.prefill_seq_lens_cpu.tolist() != [n]
            or swa.prefill_query_lens_cpu is None
            or swa.prefill_query_lens_cpu.tolist() != [n]
        ):
            return "not_full_prefill"
        if (
            swa.prefill_left_visible is not None
            or swa.prefill_right_visible is not None
        ):
            return "multimodal_visibility"
        if spec.compress_ratio > 1:
            metadata = context.attn_metadata.get(name)
            if metadata is None or metadata.num_actual_tokens != n:
                return "missing_compressed_metadata"
    return None


# REDKNOT: RK-FLASH-RUNNER — preserve native V1 lifecycle and explicit guards.
def install_dsv4_runner(settings: RedKnotSettings) -> None:
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_redknot_dsv4_installed", False):
        return
    if getattr(GPUModelRunner, "_redknot_hooks_installed", False):
        raise RuntimeError("MHA and DSV4 runner hooks cannot share one process")
    original_init = GPUModelRunner.__init__
    original_forward = GPUModelRunner._model_forward

    @wraps(original_init)
    def initialize(self, vllm_config, device, *args, **kwargs):
        validate_dsv4_config(vllm_config, settings)
        original_init(self, vllm_config, device, *args, **kwargs)
        model = vllm_config.model_config
        self.redknot_runtime = DSV4Runtime(
            settings,
            CacheManager(settings.max_cache_bytes),
            {
                "family": "deepseek_v4_flash",
                "model": model.model,
                "config": model.hf_config.to_dict(),
                "dtype": str(model.dtype),
                "tp": 1,
            },
        )
        self._redknot_dsv4_specs = None

    @wraps(original_forward)
    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        def call_native():
            return original_forward(
                self,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        context = get_forward_context()
        req_ids = self.input_batch.req_ids
        if not req_ids or context.attn_metadata is None:
            return call_native()
        request = self.requests[req_ids[0]]
        runtime = self.redknot_runtime
        if self._redknot_dsv4_specs is None:
            self._redknot_dsv4_specs = resolve_dsv4_layers(
                runtime.settings, context.no_compile_layers
            )
        specs = self._redknot_dsv4_specs
        tokens = request.prompt_token_ids or []
        reason = prefill_reason(self, context, positions, len(tokens), specs)
        if inputs_embeds is not None or getattr(request, "mm_features", None):
            reason = "unsupported_input"
        with runtime.step(
            extra_args=getattr(request.sampling_params, "extra_args", None),
            token_ids=tokens,
            specs=specs,
            unsupported_reason=reason,
        ) as state:
            sentinel = object()
            previous = context.additional_kwargs.get(CONTEXT_KEY, sentinel)
            context.additional_kwargs[CONTEXT_KEY] = (runtime, state)
            try:
                return call_native()
            finally:
                if previous is sentinel:
                    context.additional_kwargs.pop(CONTEXT_KEY, None)
                else:
                    context.additional_kwargs[CONTEXT_KEY] = previous

    GPUModelRunner.__init__ = initialize
    GPUModelRunner._model_forward = forward
    GPUModelRunner._redknot_dsv4_installed = True
