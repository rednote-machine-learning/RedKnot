"""Head-output reuse on vLLM's real FlashAttention KV cache and layer path."""

from __future__ import annotations

import torch
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
    flash_attn_varlen_func,
)

from .ops import head_partition, relocate_rope
from .runner import CONTEXT_KEY
from .runtime import LayerPayload


class RedKnotMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.redknot_query_start_loc_cpu = tuple(
            common_attn_metadata.query_start_loc_cpu.tolist()
        )
        return metadata

    def use_cascade_attention(self, *args, **kwargs):
        return False


class RedKnotBackend(FlashAttentionBackend):
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return RedKnotImpl

    @staticmethod
    def get_builder_cls():
        return RedKnotMetadataBuilder

    @classmethod
    def supports_sink(cls):
        return False

    @classmethod
    def supports_mm_prefix(cls):
        return False

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False

    @classmethod
    def supports_kv_connector(cls):
        return False


# REDKNOT: RK-MHA-ATTENTION — local/global reuse on native vLLM KV pages.
class RedKnotImpl(FlashAttentionImpl):
    redknot_implementation = True

    def _attention(self, q, k, v):
        """One contiguous dirty run, with bottom-right aligned causal masking."""
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=q.device)
        out = torch.empty_like(q)
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=k.shape[0],
            causal=True,
            softmax_scale=self.scale,
            fa_version=self.vllm_flash_attn_version,
            num_splits=1
            if getattr(self, "batch_invariant_enabled", False)
            or getattr(self, "fa4_hd256", False)
            else 0,
        )
        return out

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        entry = get_forward_context().additional_kwargs.get(CONTEXT_KEY)
        name = layer.layer_name
        if entry is None:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        runtime, state = entry
        spec = state.specs.get(name)
        if state.mode not in {"capture", "reuse"} or spec is None:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        if output_scale is not None or output_block_scale is not None:
            raise RuntimeError("RedKnot does not support output quantization")
        n = state.prompt_length
        if (
            query.shape != (n, spec.num_query_heads, spec.head_size)
            or key.shape != (n, spec.num_kv_heads, spec.head_size)
            or value.shape != key.shape
        ):
            raise RuntimeError(
                "RedKnot tensor geometry changed after request preflight"
            )
        local_q, global_q, global_kv = head_partition(
            spec.num_query_heads, spec.num_kv_heads, spec.local_heads
        )
        lq = torch.tensor(local_q, device=query.device, dtype=torch.long)
        lkv = torch.tensor(spec.local_heads, device=query.device, dtype=torch.long)
        if state.mode == "capture":
            result = super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
            tensors = (
                key.index_select(1, lkv).detach().to("cpu", copy=True),
                value.index_select(1, lkv).detach().to("cpu", copy=True),
                output.index_select(1, lq).detach().to("cpu", copy=True),
            )
            nbytes = sum(t.numel() * t.element_size() for t in tensors)
            state.staged[name] = LayerPayload(spec, *tensors, nbytes)
            runtime.counters["capture_layers"] += 1
            return result

        # Native do_kv_cache_update has already written the online K/V. Build
        # the mixed local-head view before touching those persistent slots.
        mixed_k = key.index_select(1, lkv).clone()
        mixed_v = value.index_select(1, lkv).clone()
        output_view = output.view(n, spec.num_query_heads, spec.head_size)
        clean_payloads = []
        for start, end, chunk_index in state.clean_runs:
            span = state.plan.chunks[chunk_index]
            item = state.cached[state.keys[chunk_index]].layers[name]
            source_start = start - span.start
            source_end = end - span.start
            saved_k = item.keys[source_start:source_end].to(
                device=query.device, dtype=query.dtype
            )
            saved_v = item.values[source_start:source_end].to(
                device=query.device, dtype=query.dtype
            )
            saved_o = item.outputs[source_start:source_end].to(
                device=query.device, dtype=query.dtype
            )
            if span.start:
                source_positions = torch.arange(
                    source_start, source_end, device=query.device, dtype=torch.int64
                )
                target_positions = torch.arange(
                    start, end, device=query.device, dtype=torch.int64
                )
                saved_k = relocate_rope(
                    saved_k,
                    source_positions,
                    target_positions,
                    rotary_dim=runtime.settings.rotary_dim,
                    theta=runtime.settings.rope_theta,
                )
            mixed_k[start:end] = saved_k
            mixed_v[start:end] = saved_v
            output_view[start:end, lq] = saved_o
            clean_payloads.append((start, end, saved_k, saved_v))

        # Global heads execute all rows. Local heads execute only dirty rows;
        # no attention call includes a clean local-head query row.
        if global_q:
            gq = torch.tensor(global_q, device=query.device, dtype=torch.long)
            gkv = torch.tensor(global_kv, device=query.device, dtype=torch.long)
            output_view[:, gq] = self._attention(
                query.index_select(1, gq),
                key.index_select(1, gkv),
                value.index_select(1, gkv),
            )
        for start, end in state.dirty_runs:
            output_view[start:end, lq] = self._attention(
                query[start:end].index_select(1, lq), mixed_k[:end], mixed_v[:end]
            )

        slots = get_forward_context().slot_mapping[name]
        if slots.ndim != 1 or slots.numel() != n or bool((slots < 0).any()):
            raise RuntimeError("RedKnot requires valid full-prefill physical KV slots")
        if (
            kv_cache.ndim != 4
            or kv_cache.shape[1] != spec.num_kv_heads
            or kv_cache.shape[-1] != 2 * spec.head_size
        ):
            raise RuntimeError("unexpected pinned FlashAttention KV cache layout")
        block_size = kv_cache.shape[2]
        for start, end, saved_k, saved_v in clean_payloads:
            row_slots = slots[start:end].long()
            blocks = (row_slots // block_size)[:, None]
            offsets = (row_slots % block_size)[:, None]
            heads = lkv[None, :]
            kv_cache[blocks, heads, offsets, : spec.head_size] = saved_k
            kv_cache[blocks, heads, offsets, spec.head_size :] = saved_v
        state.modified_layers += 1
        clean_rows = sum(end - start for start, end, _ in state.clean_runs)
        runtime.counters["reused_local_query_rows"] += clean_rows * len(local_q)
        runtime.counters["computed_global_query_rows"] += n * len(global_q)
        runtime.counters["computed_local_query_rows"] += (n - clean_rows) * len(local_q)
        runtime.counters["restored_local_kv_rows"] += clean_rows * len(spec.local_heads)
        return output
