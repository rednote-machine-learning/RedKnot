# Copyright 2024-2026 SGLang RedKnot Integration.
"""Standalone RedKnot forward driver.

The full sglang serving stack (Scheduler + RadixCache + CUDA graph) is a
heavy harness to bring up for offline benchmarks. This driver is a thin
wrapper that lets us run *exactly the same RedKnot kernels* the sglang
backend uses, but on top of a vanilla HuggingFace ``AutoModelForCausalLM``.

It is the recommended way to:

- Validate that the per-head vectorised attention matches the original
  RedKnot numerics.
- Run head-to-head quality / TTFT comparisons against an HF baseline.
- Smoke-test new head configurations before going through sglang.

The driver reuses every internal helper that the sglang backend uses, so a
positive result here means the backend's hot path is correct.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.redknot.head_config import (
    HEAD_DENSE,
    HeadClassConfig,
)
from sglang.srt.layers.attention.redknot.mask_plan import (
    build_layer_mask_plan,
    pad_per_head_sinks,
)
from sglang.srt.layers.attention.redknot.offline_cache import OfflineSegment
from sglang.srt.layers.attention.redknot.ops_flash import (
    DEFAULT_Q_CHUNK,
    is_flash_attn_available,
    segment_attention_flash,
)
from sglang.srt.layers.attention.redknot.ops_flash3 import (
    is_fa3_available,
    segment_attention_flash3,
)
from sglang.srt.layers.attention.redknot.ops_flash3_parallel import (
    is_fa3_parallel_available,
    segment_attention_flash3_parallel,
)
from sglang.srt.layers.attention.redknot.rope_helper import RoPEHelper
from sglang.srt.layers.attention.redknot.sparse_ffn import (
    SparseFFNSchedule,
    apply_sparse_ffn,
    token_importance_from_attn,
)

logger = logging.getLogger(__name__)


def _resolve_kernel(kernel: str):
    """Map ``kernel`` name to the attention function.

    Supported values:
      - ``"fa2"``           : FlashAttention-2 (``flash_attn`` package).
      - ``"fa3"``           : FlashAttention-3 (``sgl_kernel.flash_attn``,
                              requires Hopper SM 9.0+); per-head-bucket
                              serial dispatch.
      - ``"fa3_parallel"``  : FA-3 packed across all KV heads of the layer
                              into 2-3 large kernel calls (one packed call
                              for non-local heads, two packed calls + LSE
                              merge for local heads). Lowest launch
                              overhead; numerics identical to ``fa3``.

    Aliases: ``"flash"`` -> ``"fa2"``.
    """
    k = kernel.lower()
    if k in ("flash", "fa2"):
        if not is_flash_attn_available():
            raise RuntimeError("kernel='fa2' requested but flash_attn missing.")
        return segment_attention_flash
    if k == "fa3":
        if not is_fa3_available():
            raise RuntimeError(
                "kernel='fa3' requested but FA-3 not usable on this GPU "
                "(needs sgl_kernel.flash_attn AND SM 9.0+)."
            )
        return segment_attention_flash3
    if k == "fa3_parallel":
        if not is_fa3_parallel_available():
            raise RuntimeError(
                "kernel='fa3_parallel' requested but FA-3 not usable here."
            )
        return segment_attention_flash3_parallel
    raise ValueError(
        f"Unknown RedKnot kernel {kernel!r}; expected 'fa2', 'fa3', or 'fa3_parallel'."
    )


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _get_apply_rotary(model_type: str):
    if model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    elif model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    elif model_type == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    return apply_rotary_pos_emb


# ──────────────────────────────────────────────────────────────────────────
# Online segment forward (== backend's _redknot_single_extend, but using
# the model's own Q/K/V projections via attention forward patching)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def online_forward_segment(
    model,
    *,
    segment_input_ids: torch.Tensor,
    position_offset: int,
    head_cfg: HeadClassConfig,
    rope_helper: RoPEHelper,
    prev_online_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    prev_offline_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    prev_doc_offsets: List[int],
    q_chunk_size: int = DEFAULT_Q_CHUNK,
    kernel: str = "fa2",
    sparse_ffn_schedule: Optional[SparseFFNSchedule] = None,
    sparse_ffn_stats: Optional[List[dict]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Run one online segment forward through ``model``, returning per-layer KV.

    The function temporarily patches every ``self_attn.forward`` with a
    closure that:
      1. Projects Q/K/V at the true positions ``[position_offset, ...)``.
      2. Builds prev-K/V from ``prev_online_kvs`` (concatenated) and the
         **sink** from ``prev_offline_kvs[0]``.
      3. For ``dense`` layers, RoPE-realigns ``prev_offline_kvs`` to their
         true positions and uses those instead of online KV.
      4. Invokes the kernel selected by ``kernel`` (``"fa2"`` |  ``"fa3"``).

    Sparse FFN
    ----------
    When ``sparse_ffn_schedule`` is provided, RedKnot's token-selective
    Sparse FFN (paper Algorithm 1 lines 20-23) is also activated: each
    decoder layer's ``mlp`` is patched so that deep layers run the dense
    FFN only on important tokens (selected from the recovered attention
    signal) and route the rest through the residual identity. Per-layer
    selection stats are appended to ``sparse_ffn_stats`` when supplied.

    Returns
    -------
    ``[num_layers]`` list of ``(K, V)`` tuples for this segment (already at
    their true positions; ready to be concatenated for query forward).
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    base_model = model.model if hasattr(model, "model") else model

    device = model.device
    L = int(segment_input_ids.shape[1])
    full_position_ids = torch.arange(
        position_offset, position_offset + L, device=device
    ).unsqueeze(0)

    captured_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers
    # Per-layer token importance captured during attention, consumed by the
    # patched MLP for Sparse FFN selection.
    captured_importance: List[Optional[torch.Tensor]] = [None] * num_layers
    orig_forwards: dict = {}
    orig_mlp_forwards: dict = {}
    apply_rotary = _get_apply_rotary(config.model_type)

    def make_patched(layer_idx: int):
        attn_module = base_model.layers[layer_idx].self_attn

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)
            num_kv_groups = attn_module.num_key_value_groups

            # ── Q/K/V projections at true positions ──
            q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k_self = (
                attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            v_self = (
                attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            if getattr(attn_module, "q_norm", None) is not None:
                q = attn_module.q_norm(q)
            if getattr(attn_module, "k_norm", None) is not None:
                k_self = attn_module.k_norm(k_self)
            cos, sin = position_embeddings
            if cos.device != q.device:
                cos = cos.to(q.device)
                sin = sin.to(q.device)
            q, k_self = apply_rotary(q, k_self, cos, sin)

            # ── Build prev K/V across all prev segments ──
            # For dense layers we use offline KV (RoPE realigned).
            # For others we use the online KV produced by previous segments.
            plan = build_layer_mask_plan(head_cfg, layer_idx, q.device)

            # Determine whether this whole layer is "dense" (we treat dense_prefix_layers
            # as a layer-level switch, matching v0.4 semantics).
            layer_is_dense = layer_idx < head_cfg.dense_prefix_layers

            if layer_is_dense and prev_offline_kvs:
                prev_k_parts = []
                prev_v_parts = []
                for d_idx, seg_kv in enumerate(prev_offline_kvs):
                    pk, pv = seg_kv[layer_idx]
                    pk = pk.to(q.device)
                    pv = pv.to(q.device)
                    L_i = pk.shape[2]
                    pk = rope_helper.reposition_offset(
                        pk, src_start=0, dst_start=prev_doc_offsets[d_idx], length=L_i
                    )
                    prev_k_parts.append(pk)
                    prev_v_parts.append(pv)
                prev_k = torch.cat(prev_k_parts, dim=2)
                prev_v = torch.cat(prev_v_parts, dim=2)
            elif prev_online_kvs:
                prev_k_parts = []
                prev_v_parts = []
                for seg_kv in prev_online_kvs:
                    pk, pv = seg_kv[layer_idx]
                    prev_k_parts.append(pk.to(q.device))
                    prev_v_parts.append(pv.to(q.device))
                prev_k = torch.cat(prev_k_parts, dim=2)
                prev_v = torch.cat(prev_v_parts, dim=2)
            else:
                prev_k = None
                prev_v = None

            # ── Sink: from segment-0 offline KV, padded to plan.sink_max ──
            if prev_offline_kvs:
                seg0_k = prev_offline_kvs[0][layer_idx][0].to(q.device)
                seg0_v = prev_offline_kvs[0][layer_idx][1].to(q.device)
            else:
                seg0_k = k_self
                seg0_v = v_self
            k_sink, v_sink = pad_per_head_sinks(seg0_k, seg0_v, plan)

            # ── Run FlashAttention per-head-type attention ──
            # ``kernel`` switches between FA-2 (ops_flash) and FA-3
            # (ops_flash3). Each head-type bucket is dispatched to the
            # appropriate flash variant: local -> two-pass (sink+prev
            # full visible + self windowed-causal) merged via LSE;
            # global/dense -> two-pass causal/non-causal + LSE merge;
            # retrieval -> top-p selection -> global path.
            kfn = _resolve_kernel(kernel)
            attn_out = kfn(
                q=q,
                k_self=k_self,
                v_self=v_self,
                k_prev=prev_k,
                v_prev=prev_v,
                k_sink_padded=k_sink,
                v_sink_padded=v_sink,
                plan=plan,
                num_q_per_kv=num_kv_groups,
                sm_scale=attn_module.scaling,
                retrieval_top_p=head_cfg.retrieval_top_p,
                q_chunk_size=q_chunk_size,
            )  # [B, Hq, L_q, D]
            # ── Capture token importance for Sparse FFN selection ──
            # Done before o_proj so the signal reflects the recovered
            # attention contribution (paper §4.2: token importance is
            # estimated from the recovered attention signal).
            if sparse_ffn_schedule is not None:
                captured_importance[layer_idx] = token_importance_from_attn(
                    attn_out
                ).detach()

            attn_output = attn_out.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)

            captured_kv[layer_idx] = (
                k_self.detach().clone(),
                v_self.detach().clone(),
            )
            return attn_output, None

        return patched_forward

    def make_patched_mlp(layer_idx: int):
        """Patch the layer MLP to run RedKnot's partial Sparse FFN.

        The HF decoder layer calls ``hidden = residual + mlp(hidden)`` after
        attention, where the input ``hidden`` to ``mlp`` is the normalised
        post-attention residual stream. We intercept ``mlp(Y)`` and instead
        return ``Z`` such that only important tokens carry the real FFN
        update and the rest get ``Z = 0`` (residual identity). Because the
        decoder still adds ``residual + Z``, this realises Algorithm 1
        lines 21-23 without touching the decoder layer code.
        """
        mlp_module = base_model.layers[layer_idx].mlp
        orig_mlp = mlp_module.forward

        def patched_mlp(hidden_states, *args, **kwargs):
            # Fall back to dense if no importance was captured (e.g. the
            # very first prefilled segment that skips RedKnot attention).
            importance = captured_importance[layer_idx]
            if importance is None:
                return orig_mlp(hidden_states, *args, **kwargs)

            # apply_sparse_ffn expects [B, L, E]; the HF mlp input is the
            # same shape. It returns X_next = Y + Z, but the decoder will
            # add `residual + mlp_out`. To keep the decoder's own residual
            # add correct, we must return Z (not Y + Z). We compute Z here.
            if hidden_states.dim() != 3:
                return orig_mlp(hidden_states, *args, **kwargs)

            sched = sparse_ffn_schedule
            if sched.is_dense_layer(layer_idx):
                z = orig_mlp(hidden_states, *args, **kwargs)
                if sparse_ffn_stats is not None:
                    B, Lq, _ = hidden_states.shape
                    sparse_ffn_stats.append(
                        {
                            "layer": layer_idx,
                            "mode": "dense",
                            "selected": B * Lq,
                            "total": B * Lq,
                            "selected_frac": 1.0,
                        }
                    )
                return z

            # Deep layer: Z=0 except for selected rows where Z = mlp(Y).
            out_xnext, stats = apply_sparse_ffn(
                hidden_states,
                lambda rows: orig_mlp(rows, *args, **kwargs),
                layer_idx=layer_idx,
                schedule=sched,
                importance=importance,
                return_stats=True,
            )
            if sparse_ffn_stats is not None:
                sparse_ffn_stats.append(stats)
            # out_xnext = Y + Z  =>  Z = out_xnext - Y, so the decoder's
            # `residual + Z` reproduces `residual + (out_xnext - Y)`. Since
            # Y == hidden_states here, return the delta.
            return out_xnext - hidden_states

        return patched_mlp

    for layer_idx in range(num_layers):
        m = base_model.layers[layer_idx].self_attn
        orig_forwards[layer_idx] = m.forward
        m.forward = make_patched(layer_idx)
        if sparse_ffn_schedule is not None:
            mlp_mod = getattr(base_model.layers[layer_idx], "mlp", None)
            if mlp_mod is not None:
                orig_mlp_forwards[layer_idx] = mlp_mod.forward
                mlp_mod.forward = make_patched_mlp(layer_idx)

    try:
        model(
            input_ids=segment_input_ids.to(device),
            position_ids=full_position_ids,
            use_cache=False,
        )
    finally:
        for layer_idx in range(num_layers):
            base_model.layers[layer_idx].self_attn.forward = orig_forwards[layer_idx]
        for layer_idx, orig in orig_mlp_forwards.items():
            base_model.layers[layer_idx].mlp.forward = orig

    return captured_kv


# ──────────────────────────────────────────────────────────────────────────
# End-to-end RedKnot run
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_redknot(
    model,
    tokenizer,
    *,
    segments_offline: List[OfflineSegment],
    query_text: str,
    head_cfg: HeadClassConfig,
    rope_helper: Optional[RoPEHelper] = None,
    max_new_tokens: int = 100,
    q_chunk_size: int = DEFAULT_Q_CHUNK,
    kernel: str = "fa2",
    sparse_ffn_schedule: Optional[SparseFFNSchedule] = None,
    sparse_ffn_stats: Optional[List[dict]] = None,
) -> Tuple[torch.Tensor, str, int, float]:
    """End-to-end forward with RedKnot compression.

    ``kernel`` is ``"fa2"`` (default) for FlashAttention-2 or ``"fa3"`` for
    FlashAttention-3 (requires Hopper).

    When ``sparse_ffn_schedule`` is given, RedKnot's token-selective Sparse
    FFN is applied to the online segment forwards; per-layer selection stats
    are appended to ``sparse_ffn_stats`` if provided.

    Returns
    -------
    first_logits : ``[V]`` tensor — logits at the last query token.
    text         : decoded greedy generation.
    query_len    : token count of ``query_text``.
    ttft_seconds : wall-clock from start of segment forwards to first logits.
    """
    from transformers import DynamicCache

    if rope_helper is None:
        base_model = model.model if hasattr(model, "model") else model
        rope_helper = RoPEHelper(base_model.rotary_emb)

    # Compute offsets.
    doc_lens = [seg.doc_len for seg in segments_offline]
    offsets, p = [], 0
    for dl in doc_lens:
        offsets.append(p)
        p += dl
    query_offset = p

    n_layers = model.config.num_hidden_layers

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    # Segment 1: reuse offline KV directly (it is already at positions [0, L_1)).
    doc_online_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    doc_online_kvs.append(
        [(k.to(model.device), v.to(model.device)) for k, v in segments_offline[0].kv]
    )

    # Segments 2..N: online RedKnot forward.
    for d_idx in range(1, len(segments_offline)):
        seg = segments_offline[d_idx]
        prev_offline = [s.kv for s in segments_offline[:d_idx]]
        kv = online_forward_segment(
            model,
            segment_input_ids=seg.token_ids.unsqueeze(0),
            position_offset=offsets[d_idx],
            head_cfg=head_cfg,
            rope_helper=rope_helper,
            prev_online_kvs=doc_online_kvs[:d_idx],
            prev_offline_kvs=prev_offline,
            prev_doc_offsets=offsets[:d_idx],
            q_chunk_size=q_chunk_size,
            kernel=kernel,
            sparse_ffn_schedule=sparse_ffn_schedule,
            sparse_ffn_stats=sparse_ffn_stats,
        )
        doc_online_kvs.append(kv)

    # Build a DynamicCache by concatenating all per-layer KV.
    past = DynamicCache()
    base_model = model.model if hasattr(model, "model") else model
    for layer_idx in range(n_layers):
        layer_device = next(base_model.layers[layer_idx].self_attn.parameters()).device
        k_parts, v_parts = [], []
        for d_idx in range(len(segments_offline)):
            dk, dv = doc_online_kvs[d_idx][layer_idx]
            k_parts.append(dk.to(layer_device))
            v_parts.append(dv.to(layer_device))
        past.update(
            torch.cat(k_parts, dim=2),
            torch.cat(v_parts, dim=2),
            layer_idx,
        )

    # Query forward. Use SDPA so the [L_query x L_total] attention matrix
    # is not materialised at 31k+ contexts.
    query_ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    query_len = int(query_ids.shape[1])
    total_kv = sum(doc_lens)
    query_position_ids = torch.arange(
        query_offset, query_offset + query_len, device=model.device
    ).unsqueeze(0)
    cache_position = torch.arange(total_kv, total_kv + query_len, device=model.device)

    _q_orig = _switch_attn_impl(model, "sdpa")
    try:
        out = model(
            input_ids=query_ids,
            position_ids=query_position_ids,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
        )
    finally:
        _restore_attn_impl(model, _q_orig)
    first_logits = out.logits[0, -1, :].clone()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t_start

    # Greedy decode.
    generated = []
    past_for_gen = out.past_key_values
    next_id = first_logits.argmax().unsqueeze(0).unsqueeze(0)
    generated.append(int(next_id[0, 0].item()))
    total_seen = total_kv + query_len
    for step in range(max_new_tokens - 1):
        cur_pos = total_seen + len(generated) - 1
        out_g = model(
            input_ids=next_id,
            position_ids=torch.tensor([[cur_pos]], device=model.device),
            past_key_values=past_for_gen,
            cache_position=torch.tensor([cur_pos], device=model.device),
            use_cache=True,
        )
        past_for_gen = out_g.past_key_values
        next_id = out_g.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        tid = int(next_id[0, 0].item())
        generated.append(tid)
        if tid == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    del past, past_for_gen, out, doc_online_kvs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return first_logits, text, query_len, ttft


# ──────────────────────────────────────────────────────────────────────────
# Baseline (chunked full-recompute) for comparison
# ──────────────────────────────────────────────────────────────────────────
def _switch_attn_impl(model, impl: str):
    """Temporarily flip the attention implementation on every layer.

    Mirrors the helper in ``__REDKNOT_V02__/main.py`` so the 31k-context
    baseline can use SDPA (no materialised attention matrix) instead of
    eager attention which OOMs at that length.
    """
    base = model.model if hasattr(model, "model") else model
    orig = {}
    for li, layer in enumerate(base.layers):
        attn = layer.self_attn
        if hasattr(attn, "config"):
            orig[li] = getattr(attn.config, "_attn_implementation", "eager")
            attn.config._attn_implementation = impl
    if hasattr(model, "config"):
        orig["_model"] = getattr(model.config, "_attn_implementation", None)
        model.config._attn_implementation = impl
    return orig


def _restore_attn_impl(model, orig):
    base = model.model if hasattr(model, "model") else model
    for li, layer in enumerate(base.layers):
        attn = layer.self_attn
        if hasattr(attn, "config") and li in orig:
            attn.config._attn_implementation = orig[li]
    if "_model" in orig and hasattr(model, "config"):
        model.config._attn_implementation = orig["_model"]


@torch.no_grad()
def run_baseline(
    model,
    tokenizer,
    *,
    prompt: str,
    max_new_tokens: int = 100,
    chunk_size: int = 8192,
    attn_impl: str = "sdpa",
) -> Tuple[torch.Tensor, str, int, float]:
    """Standard HF baseline: chunked prefill + KV-cached greedy decode.

    ``attn_impl`` is the value passed to each layer's
    ``self_attn.config._attn_implementation``. Supported values mirror
    what HuggingFace recognises:

      - ``"sdpa"``              : PyTorch SDPA (auto-picks FA-2 / mem-eff).
      - ``"flash_attention_2"`` : Direct call to the ``flash_attn`` v2 kernel.
      - ``"eager"``             : Pure-PyTorch matmul + softmax (OOMs on >32k).

    HuggingFace does *not* expose FA-3 to user-defined attn_impl strings
    yet (as of transformers 4.57); to compare RedKnot+FA-3 against a
    FA-3 baseline you would need a custom wrapper. The closest fair
    baseline today is ``flash_attention_2``.

    Defaults to ``"sdpa"`` for long-context safety.
    """
    from transformers import DynamicCache

    orig_impl = _switch_attn_impl(model, attn_impl)
    try:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        L = int(ids.shape[1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        past = DynamicCache()
        first_logits = None
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk = ids[:, start:end]
            out = model(input_ids=chunk, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if end == L:
                first_logits = out.logits[0, -1, :].clone()
            del out
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t_start

        generated = [int(first_logits.argmax().item())]
        cur_tok = torch.tensor([[generated[0]]], device=model.device)
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=cur_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur_tok = out.logits[0, -1, :].argmax().view(1, 1)
            tid = int(cur_tok[0, 0].item())
            generated.append(tid)
            if tid == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated, skip_special_tokens=True)
        del past, out
    finally:
        _restore_attn_impl(model, orig_impl)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return first_logits, text, L, ttft
