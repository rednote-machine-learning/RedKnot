"""Opt-in hooks for the pinned vLLM V1 model runner."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any

from .cache import CacheManager
from .runtime import LayerSpec, RedKnotRuntime, RedKnotSettings

CONTEXT_KEY = "vllm_redknot"
_ARCHITECTURES = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"}


def validate_engine_config(config: Any, settings: RedKnotSettings) -> RedKnotSettings:
    """Reject unsupported execution modes before the runner allocates caches."""
    model = config.model_config
    hf = model.hf_config
    architectures = getattr(hf, "architectures", None)
    if not architectures or any(a not in _ARCHITECTURES for a in architectures):
        raise ValueError(
            "vllm-redknot supports only Llama/Qwen2/Qwen3 causal LM; "
            "MLA, DeepSeek V4 and hybrid models are unsupported"
        )
    parallel = config.parallel_config
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"vllm-redknot requires {name}=1")
    if getattr(parallel, "enable_dbo", False):
        raise ValueError("vllm-redknot does not support DBO")
    if not model.enforce_eager:
        raise ValueError("vllm-redknot requires --enforce-eager")
    if config.use_v2_model_runner:
        raise ValueError("vllm-redknot requires VLLM_USE_V2_MODEL_RUNNER=0")
    if config.scheduler_config.max_num_seqs != 1:
        raise ValueError("vllm-redknot requires --max-num-seqs 1")
    if config.scheduler_config.enable_chunked_prefill:
        raise ValueError("vllm-redknot requires --no-enable-chunked-prefill")
    if config.cache_config.enable_prefix_caching:
        raise ValueError(
            "vllm-redknot requires --no-enable-prefix-caching "
            "to isolate attention policies"
        )
    if getattr(config, "speculative_config", None) is not None:
        raise ValueError("vllm-redknot does not support speculative decoding")
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("vllm-redknot does not support LoRA")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("vllm-redknot uses its own bounded cache, not a KV connector")
    if (
        getattr(model, "quantization", None) is not None
        or getattr(config, "quant_config", None) is not None
    ):
        raise ValueError("vllm-redknot first version requires unquantized weights")
    if str(model.dtype) not in {"torch.float16", "torch.bfloat16"}:
        raise ValueError("vllm-redknot requires float16 or bfloat16")
    cache_dtype = config.cache_config.cache_dtype
    if cache_dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("vllm-redknot requires unquantized KV cache")
    if cache_dtype != "auto" and str(model.dtype) != f"torch.{cache_dtype}":
        raise ValueError("KV cache dtype must match model dtype")
    if getattr(hf, "sliding_window", None) is not None and getattr(
        hf, "use_sliding_window", True
    ):
        raise ValueError("vllm-redknot first version requires full causal attention")
    if getattr(hf, "dual_chunk_attention_config", None):
        raise ValueError("dual chunk attention is unsupported")
    if getattr(hf, "rope_scaling", None):
        raise ValueError("vllm-redknot supports only default static RoPE")
    rope = getattr(hf, "rope_parameters", None) or {}
    if not isinstance(rope, Mapping) or rope.get("rope_type", "default") != "default":
        raise ValueError("vllm-redknot supports only default static RoPE")
    if (
        set(rope)
        - {"rope_type", "rope_theta", "partial_rotary_factor", "rope_dim", "type"}
        or rope.get("type", "default") != "default"
    ):
        raise ValueError("scaled or multidimensional RoPE is unsupported")
    theta = rope.get("rope_theta", getattr(hf, "rope_theta", None))
    if theta is None or float(theta) != settings.rope_theta:
        raise ValueError(
            "config rope_theta must match the model's actual static RoPE theta"
        )
    heads = getattr(hf, "num_attention_heads", None)
    hidden_size = getattr(hf, "hidden_size", None)
    head_dim = getattr(hf, "head_dim", None)
    if head_dim is None and heads and hidden_size and hidden_size % heads == 0:
        head_dim = hidden_size // heads
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("cannot determine model attention head dimension")
    partial = rope.get(
        "partial_rotary_factor", getattr(hf, "partial_rotary_factor", 1.0)
    )
    rotary_dim = rope.get("rope_dim") or int(head_dim * partial)
    if (
        type(rotary_dim) is not int
        or rotary_dim <= 0
        or rotary_dim > head_dim
        or rotary_dim % 2
    ):
        raise ValueError("invalid model rotary dimension")
    if settings.rotary_dim is not None and settings.rotary_dim != rotary_dim:
        raise ValueError("config rotary_dim does not match the model")
    return replace(settings, rotary_dim=rotary_dim)


def validate_loaded_rope(
    model: Any, specs: Mapping[str, LayerSpec], settings: RedKnotSettings
) -> None:
    """Verify the instantiated operator, not only HF configuration fields."""
    modules = dict(model.named_modules())
    for layer_name in specs:
        parent_name = layer_name.rsplit(".", 1)[0]
        parent = modules.get(parent_name)
        rope = getattr(parent, "rotary_emb", None)
        if (
            rope is None
            or type(rope).__name__ != "RotaryEmbedding"
            or type(rope).__module__
            != "vllm.model_executor.layers.rotary_embedding.base"
        ):
            raise ValueError(
                f"{parent_name} must use the original static RotaryEmbedding"
            )
        if (
            rope.base != settings.rope_theta
            or rope.rotary_dim != settings.rotary_dim
            or not rope.is_neox_style
        ):
            raise ValueError(
                f"instantiated RoPE on {parent_name} differs from cache policy"
            )


def resolve_layer_specs(
    settings: RedKnotSettings, layers: Mapping[str, Any]
) -> dict[str, LayerSpec]:
    selected = {}
    for selector, local_heads in settings.local_heads.items():
        names = [selector] if selector in layers else []
        if not names and selector.isdigit():
            names = [
                name
                for name in layers
                if re.search(rf"(?:^|\.)layers\.{int(selector)}\.", name)
            ]
        if len(names) != 1:
            raise ValueError(
                f"local_heads selector {selector!r} must match "
                "exactly one attention layer"
            )
        name = names[0]
        if name in selected:
            raise ValueError(f"duplicate local_heads selectors for {name}")
        layer = layers[name]
        impl = getattr(layer, "impl", None)
        if impl is None or not getattr(impl, "redknot_implementation", False):
            raise ValueError(f"{name} is not using RedKnot CUSTOM attention")
        if any(h >= impl.num_kv_heads for h in local_heads):
            raise ValueError(f"local KV head index out of range for {name}")
        if impl.num_heads % impl.num_kv_heads:
            raise ValueError("invalid GQA head geometry")
        if (
            impl.sliding_window != (-1, -1)
            or impl.alibi_slopes is not None
            or impl.logits_soft_cap
            or impl.sinks is not None
            or impl.kv_sharing_target_layer_name
        ):
            raise ValueError(f"unsupported attention feature on {name}")
        selected[name] = LayerSpec(
            local_heads,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
            str(layer.dtype),
        )
    return selected


def _step_reason(
    runner: Any,
    context: Any,
    positions: Any,
    prompt_length: int,
    specs: Mapping[str, LayerSpec],
) -> str | None:
    if len(runner.input_batch.req_ids) != 1:
        return "batch_size"
    if not isinstance(context.attn_metadata, dict):
        return "missing_or_microbatch_metadata"
    if not isinstance(context.slot_mapping, dict):
        return "unsupported_slot_mapping"
    if positions is None or positions.ndim != 1 or positions.numel() != prompt_length:
        return "not_full_prefill"
    if positions.tolist() != list(range(prompt_length)):
        return "not_full_prefill"
    for metadata in context.attn_metadata.values():
        boundaries = getattr(metadata, "redknot_query_start_loc_cpu", None)
        if (
            boundaries != (0, prompt_length)
            or metadata.num_actual_tokens != prompt_length
        ):
            return "not_full_prefill"
        if metadata.use_cascade or metadata.causal is not True:
            return "unsupported_attention_metadata"
        if metadata.seq_lens.numel() != 1 or int(metadata.seq_lens[0]) != prompt_length:
            return "not_full_prefill"
        if any(
            getattr(metadata, attr, None) is not None
            for attr in ("mm_prefix_query_range_tensor", "rswa_prefix_lens")
        ):
            return "unsupported_attention_metadata"
    for name, spec in specs.items():
        if name not in context.attn_metadata:
            return "missing_layer_metadata"
        layer = context.no_compile_layers[name]
        cache = layer.kv_cache
        slots = context.slot_mapping.get(name)
        if (
            cache.ndim != 4
            or cache.shape[1] != spec.num_kv_heads
            or cache.shape[-1] != 2 * spec.head_size
            or str(cache.dtype) != spec.dtype
        ):
            return "unsupported_cache_layout"
        if slots is None or slots.ndim != 1 or slots.numel() != prompt_length:
            return "invalid_kv_slots"
        row_slots = slots.tolist()
        if len(set(row_slots)) != len(row_slots) or any(
            i < 0 or i >= cache.shape[0] * cache.shape[2] for i in row_slots
        ):
            return "invalid_kv_slots"
        block_table = context.attn_metadata[name].block_table[0].tolist()
        block_size = cache.shape[2]
        if len(block_table) * block_size < prompt_length:
            return "invalid_block_table"
        if row_slots != [
            block_table[i // block_size] * block_size + i % block_size
            for i in range(prompt_length)
        ]:
            return "slot_block_table_mismatch"
    return None


# REDKNOT: RK-MHA-RUNNER — guarded V1 hook; no copied SGLang runner.
def install_runner_hooks(settings: RedKnotSettings) -> None:
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_redknot_hooks_installed", False):
        return
    original_init = GPUModelRunner.__init__
    original_forward = GPUModelRunner._model_forward

    @wraps(original_init)
    def initialize(self: Any, vllm_config: Any, device: Any, *args: Any, **kwargs: Any):
        resolved = validate_engine_config(vllm_config, settings)
        original_init(self, vllm_config, device, *args, **kwargs)
        model = vllm_config.model_config
        identity = {
            "model": model.model,
            "revision": getattr(model, "revision", None),
            "hf_config": model.hf_config.to_dict(),
            "dtype": str(model.dtype),
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        }
        self.redknot_runtime = RedKnotRuntime(
            resolved, CacheManager(resolved.max_cache_bytes), identity
        )

    @wraps(original_forward)
    def forward(
        self: Any,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        context = get_forward_context()
        runtime = self.redknot_runtime
        req_ids = self.input_batch.req_ids
        if not req_ids or context.attn_metadata is None:
            return original_forward(
                self,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        request = self.requests[req_ids[0]]
        params = request.sampling_params
        extra = getattr(params, "extra_args", None)
        tokens = request.prompt_token_ids or []
        specs = resolve_layer_specs(runtime.settings, context.no_compile_layers)
        if not getattr(self, "_redknot_rope_validated", False):
            validate_loaded_rope(self.get_model(), specs, runtime.settings)
            self._redknot_rope_validated = True
        reason = _step_reason(self, context, positions, len(tokens), specs)
        if (
            inputs_embeds is not None
            or request.prompt_token_ids is None
            or getattr(request, "mm_features", None)
        ):
            reason = "unsupported_input"
        with runtime.step(
            extra_args=extra, token_ids=tokens, specs=specs, unsupported_reason=reason
        ) as state:
            previous = context.additional_kwargs.get(CONTEXT_KEY)
            context.additional_kwargs[CONTEXT_KEY] = (runtime, state)
            try:
                return original_forward(
                    self,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
            finally:
                if previous is None:
                    context.additional_kwargs.pop(CONTEXT_KEY, None)
                else:
                    context.additional_kwargs[CONTEXT_KEY] = previous

    GPUModelRunner.__init__ = initialize
    GPUModelRunner._model_forward = forward
    GPUModelRunner._redknot_hooks_installed = True
