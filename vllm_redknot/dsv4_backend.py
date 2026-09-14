"""DSV4 FlashMLA hooks that retain native KV/compressor/indexer production.

Only the final prefill sparse-attention call and the output projection are
intercepted. Decode, latent cache writes, candidate gathering and all FFNs use
the unmodified vLLM path. Registration alone does not allocate GPU memory.
"""

from __future__ import annotations

from functools import wraps

from .dsv4_projection import (
    CachedContribution,
    capture_local_z,
    merge_cached_z_and_project,
)
from .dsv4_runner import CONTEXT_KEY


def native_project_z(layer, o, positions):
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    return deep_gemm_fp8_o_proj(
        o,
        positions,
        layer.rotary_emb.cos_sin_cache,
        layer.wo_a,
        lambda z: z,
        n_groups=layer.n_local_groups,
        heads_per_group=layer.n_local_heads // layer.n_local_groups,
        nope_dim=layer.nope_head_dim,
        rope_dim=layer.rope_head_dim,
        o_lora_rank=layer.o_lora_rank,
        einsum_recipe=layer._einsum_recipe,
        tma_aligned_scales=layer._tma_aligned_scales,
    )


# REDKNOT: RK-FLASH-ATTENTION — intercept sparse prefill/projection, not KV state.
def install_dsv4_attention() -> None:
    import torch
    from vllm.forward_context import get_forward_context
    from vllm.models.deepseek_v4.nvidia import flashmla as native

    from .dsv4_sparse import selected_sparse_mla

    cls = native.DeepseekV4FlashMLAAttention
    if getattr(cls, "_redknot_dsv4_attention_installed", False):
        return
    original_mqa = cls.forward_mqa
    original_projection = cls._o_proj
    original_sparse = native.flash_mla_sparse_fwd

    def current_entry():
        return get_forward_context().additional_kwargs.get(CONTEXT_KEY)

    @wraps(original_mqa)
    def forward_mqa(self, q, kv, positions, output):
        entry = current_entry()
        if entry is None or self.prefix not in entry[1].specs:
            return original_mqa(self, q, kv, positions, output)
        state = entry[1]
        if state.mode not in {"capture", "reuse"}:
            return original_mqa(self, q, kv, positions, output)
        previous = state.active_layer
        state.active_layer = self.prefix
        try:
            return original_mqa(self, q, kv, positions, output)
        finally:
            state.active_layer = previous

    @wraps(original_sparse)
    def sparse_forward(*args, **kwargs):
        entry = current_entry()
        if entry is None:
            return original_sparse(*args, **kwargs)
        runtime, state = entry
        if state.mode != "reuse" or state.active_layer not in state.specs:
            return original_sparse(*args, **kwargs)
        # This exact pinned callsite uses keywords and ignores the return value.
        # Reject interface drift instead of guessing argument positions.
        expected = {"q", "kv", "indices", "sm_scale", "attn_sink", "topk_length", "out"}
        if args or set(kwargs) != expected:
            raise RuntimeError("DSV4 native sparse call contract changed")
        spec = state.specs[state.active_layer]
        q, out = kwargs["q"], kwargs["out"]
        if q.shape != (state.prompt_length, spec.num_heads, spec.head_dim):
            raise RuntimeError(
                "DSV4 selected-head kernel requires one complete prefill"
            )
        if out.shape != q.shape or out.dtype != q.dtype:
            raise RuntimeError("DSV4 output geometry mismatch")
        local = spec.local_heads
        global_heads = tuple(
            head for head in range(spec.num_heads) if head not in local
        )
        dirty = tuple(
            row for start, end in state.dirty_runs for row in range(start, end)
        )
        out.zero_()

        def compute(rows, heads):
            return selected_sparse_mla(
                q,
                kwargs["kv"],
                kwargs["indices"],
                kwargs["topk_length"],
                kwargs["attn_sink"],
                rows,
                heads,
                scale=kwargs["sm_scale"],
            )

        if global_heads:
            global_out = compute(tuple(range(state.prompt_length)), global_heads)
            head_index = torch.tensor(global_heads, device=q.device, dtype=torch.long)
            out.index_copy_(1, head_index, global_out)
        if dirty:
            local_out = compute(dirty, local)
            rows = torch.tensor(dirty, device=q.device, dtype=torch.long)
            heads = torch.tensor(local, device=q.device, dtype=torch.long)
            out[rows[:, None], heads[None, :], :] = local_out
        clean_rows = state.prompt_length - len(dirty)
        runtime.counters["reused_local_query_rows"] += clean_rows * len(local)
        runtime.counters["computed_global_query_rows"] += state.prompt_length * len(
            global_heads
        )
        runtime.counters["computed_local_query_rows"] += len(dirty) * len(local)
        runtime.counters["native_state_token_rows"] += state.prompt_length
        runtime.counters["launched_sparse_head_rows"] += state.prompt_length * (
            (len(global_heads) + 15) // 16 * 16
        ) + len(dirty) * ((len(local) + 15) // 16 * 16)
        state.sparse_layers.add(state.active_layer)
        # Native _forward_prefill consumes only the mutated out buffer. No LSE
        # placeholder is consumed or exposed by this guarded private callsite.
        return out, None, None

    @wraps(original_projection)
    def output_projection(self, o, positions):
        entry = current_entry()
        if entry is None or self.prefix not in entry[1].specs:
            return original_projection(self, o, positions)
        runtime, state = entry
        spec = state.specs[self.prefix]
        policy_key = runtime.policy_key(self.prefix, spec)

        def project_z(attention, at_positions):
            return native_project_z(self, attention, at_positions)

        if state.mode == "capture":
            result = original_projection(self, o, positions)
            artifact = capture_local_z(
                o, positions, spec.local_heads, project_z, policy_key=policy_key
            )
            if tuple(artifact.z_off.shape) != (
                state.prompt_length,
                spec.groups * spec.rank,
            ):
                raise RuntimeError("DSV4 capture projection width changed")
            state.staged[self.prefix] = artifact
            runtime.counters["capture_layers"] += 1
            return result
        if state.mode != "reuse":
            return original_projection(self, o, positions)
        if self.prefix not in state.sparse_layers:
            raise RuntimeError(
                "DSV4 z_off cannot merge without selected-head attention"
            )
        contributions = []
        for start, end, chunk_index in state.clean_runs:
            span = state.plan.chunks[chunk_index]
            cached = state.cached[state.keys[chunk_index]].layers[self.prefix]
            contributions.append(
                CachedContribution(
                    cached,
                    torch.arange(start, end, dtype=torch.long),
                    torch.arange(
                        start - span.start, end - span.start, dtype=torch.long
                    ),
                )
            )
        result = merge_cached_z_and_project(
            o,
            positions,
            contributions,
            project_z,
            self.wo_b,
            local_head_ids=spec.local_heads,
            policy_key=policy_key,
        )
        state.projected_layers.add(self.prefix)
        runtime.counters["reused_projected_token_rows"] += sum(
            end - start for start, end, _ in state.clean_runs
        )
        return result

    cls.forward_mqa = forward_mqa
    cls._o_proj = output_projection
    native.flash_mla_sparse_fwd = sparse_forward
    cls._redknot_dsv4_attention_installed = True
