# REDKNOT-MODEL: Q-gating/head-class glue with explicitly injected native operations.
# Copyright 2024-2026 SGLang RedKnot Integration.
"""Qwen3.5 RedKnot full-attention helper with no native model/kernel imports.

The adapter supplies the projected module view, native RoPE operation, and
head-class attention callable. Tensor shapes follow the source helper, not a
claim of direct compatibility with vLLM packed modules or TP layouts.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm_redknot.core.head_config import HeadClassConfig


@torch.no_grad()
def _qwen35_full_attn_headclass(
    attn_module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    head_cfg: HeadClassConfig,
    hc_layer_idx: int,
    seg0_k: torch.Tensor | None,
    seg0_v: torch.Tensor | None,
    seg0_len: int,
    rotary_apply: Callable,
    headclass_attention: Callable,
):
    """Drop-in replacement for one Qwen3.5 full-attention forward.

    Mirrors ``Qwen3_5MoeAttention.forward`` but routes the attention through
    RedKnot's head-class kernel. Handles Q-gating (q_proj -> [Q|gate], output
    *= sigmoid(gate)) and q/k norm.
    """
    input_shape = hidden_states.shape[:-1]
    hd = attn_module.head_dim
    hidden_shape = (*input_shape, -1, hd)

    # q_proj emits [Q | gate]; split on the head_dim*2 axis.
    q_raw = attn_module.q_proj(hidden_states).view(*input_shape, -1, hd * 2)
    query_states, gate = torch.chunk(q_raw, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = attn_module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = attn_module.k_norm(
        attn_module.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if cos.device != query_states.device:
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
    query_states, key_states = rotary_apply(query_states, key_states, cos, sin)

    # Head-class metadata for this full layer. as_tensors gives int-encoded
    # head_type + per-head window; derive the bool local mask the kernel wants.
    strat = head_cfg.as_tensors(query_states.device)
    is_local = strat["head_type"][hc_layer_idx] == HeadClassConfig.TYPE_LOCAL  # [Hkv]
    win_row = strat["window"][hc_layer_idx]
    win_pos = win_row[win_row > 0]
    window = (
        int(win_pos.max().item())
        if win_pos.numel() > 0
        else head_cfg.local_default_window
    )
    sink = head_cfg.default_sink_size
    num_q_per_kv = attn_module.num_key_value_groups

    if seg0_k is None:
        # No offline prefix: fabricate empty seg0 so the kernel sees [online].
        B, Hkv, _, D = key_states.shape
        seg0_k = key_states.new_zeros(B, Hkv, 0, D)
        seg0_v = value_states.new_zeros(B, Hkv, 0, D)
        seg0_len = 0

    attn_out = headclass_attention(
        query_states,
        key_states,
        value_states,
        seg0_k,
        seg0_v,
        is_local,
        sink_size=sink,
        window=window,
        seg0_len=seg0_len,
        num_q_per_kv=num_q_per_kv,
        sm_scale=attn_module.scaling,
    )

    attn_output = attn_out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    # Q-gating (Qwen3.5-specific).
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = attn_module.o_proj(attn_output)
    return attn_output, key_states, value_states
