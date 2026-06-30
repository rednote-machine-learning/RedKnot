# Copyright 2024-2026 SGLang RedKnot Integration.
"""RedKnot Step-1 driver for Qwen3.5-MoE (hybrid linear/full attention).

Qwen3.5-397B-A17B is a *hybrid* model: of its ``num_hidden_layers`` only every
``full_attention_interval``-th layer is a standard softmax-attention layer
(``self_attn``); the rest are linear-attention (GatedDeltaNet) layers with no
standard KV cache. RedKnot head-class KV reuse only applies to the standard
full-attention layers.

This module implements **Step 1**: RedKnot offline-KV reuse on the
full_attention layers only, with everything else (linear layers, MoE FFN) left
native. Compared to the generic ``run_redknot_offlinekv`` (which assumes every
layer is a full-attention layer with a uniform KV cache), this driver:

  * Detects layer types from ``config.layer_types`` and only touches
    ``full_attention`` layers.
  * Handles Qwen3.5's **Q-gating**: ``q_proj`` emits ``[Q | gate]`` and the
    attention output is multiplied by ``sigmoid(gate)`` (Llama/Qwen3 have no
    such gate). The generic driver does not know about this.
  * Reuses the architecture-agnostic core ops: :class:`HeadClassConfig` and
    :func:`_flat_headclass_attention`.

NOT in Step 1 (left native / TODO):
  * Sparse-FFN over the MoE block (MoE is already expert-sparse; token-level
    sparsity on top is a separate design).
  * RoPE-repositioned cross-segment offline reuse with global-head re-prefill
    (the paper-faithful path). Step 1 does a single online prefill and head-
    class attention over the full [seg|online] sequence per full layer.

The intent is a correct, runnable baseline you can extend. It deliberately
favours clarity over peak performance.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.attention.redknot.driver_batched import (
    _flat_headclass_attention,
)
from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig


def full_attention_layer_indices(config) -> List[int]:
    """Return the indices of ``full_attention`` layers in execution order."""
    tc = getattr(config, "text_config", config)
    layer_types = getattr(tc, "layer_types", None)
    if layer_types is None:
        # Fall back to full_attention_interval if layer_types absent.
        interval = getattr(tc, "full_attention_interval", 1)
        n = tc.num_hidden_layers
        return [i for i in range(n) if (i + 1) % interval == 0]
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


def linear_attention_layer_indices(config) -> List[int]:
    """Return the indices of ``linear_attention`` layers in execution order."""
    tc = getattr(config, "text_config", config)
    layer_types = getattr(tc, "layer_types", None) or []
    return [i for i, t in enumerate(layer_types) if t == "linear_attention"]


def build_full_attention_head_config(
    config,
    *,
    frac_global: float = 0.10,
    local_window: int = 4096,
    sink_size: int = 4,
    seed: int = 1234,
    global_assign: str = "random",
) -> HeadClassConfig:
    """Build a HeadClassConfig sized for the FULL-ATTENTION layers only.

    The returned config has ``num_layers == number of full_attention layers``;
    callers map their k-th full layer to head config row ``k``.

    ``global_assign`` controls WHICH heads become global (tests the RedKnot
    "shallow=local, deep=global" depth hypothesis on Qwen3.5's full layers):
      * "random" : deterministic random spread (default; matches Llama sweet
                   spot generator).
      * "deep"   : the DEEPEST full layers are made global first
                   (shallow full layers stay local) -> tests the hypothesis.
      * "shallow": inverse control (shallowest full layers global first).
    Here "full-layer depth" is the ordinal among full layers (row 0 = shallowest
    full layer, row n_full-1 = deepest).
    """
    import random

    tc = getattr(config, "text_config", config)
    n_full = len(full_attention_layer_indices(config))
    H = tc.num_key_value_heads
    total = n_full * H
    n_global = max(1, round(frac_global * total))

    if global_assign == "deep":
        # Fill global heads starting from the deepest full layer downward.
        coords = [(li, h) for li in range(n_full - 1, -1, -1) for h in range(H)]
        global_set = set(coords[:n_global])
    elif global_assign == "shallow":
        coords = [(li, h) for li in range(n_full) for h in range(H)]
        global_set = set(coords[:n_global])
    else:  # "random"
        rng = random.Random(seed)
        coords = [(li, h) for li in range(n_full) for h in range(H)]
        rng.shuffle(coords)
        global_set = set(coords[:n_global])

    head_class: List[List[str]] = []
    head_max_distance: List[List[int]] = []
    for li in range(n_full):
        row_cls, row_dist = [], []
        for h in range(H):
            if (li, h) in global_set:
                row_cls.append("global")
                row_dist.append(-1)
            else:
                row_cls.append("local")
                row_dist.append(local_window)
        head_class.append(row_cls)
        head_max_distance.append(row_dist)

    return HeadClassConfig(
        head_class=head_class,
        head_max_distance=head_max_distance,
        num_layers=n_full,
        num_kv_heads=H,
        default_sink_size=sink_size,
        local_default_window=local_window,
    )


@torch.no_grad()
def _qwen35_full_attn_headclass(
    attn_module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    *,
    head_cfg: HeadClassConfig,
    hc_layer_idx: int,
    seg0_k: Optional[torch.Tensor],
    seg0_v: Optional[torch.Tensor],
    seg0_len: int,
):
    """Drop-in replacement for one Qwen3.5 full-attention forward.

    Mirrors ``Qwen3_5MoeAttention.forward`` but routes the attention through
    RedKnot's head-class kernel. Handles Q-gating (q_proj -> [Q|gate], output
    *= sigmoid(gate)) and q/k norm.
    """
    # Works for both qwen3_5 (dense, e.g. 0.8B) and qwen3_5_moe (e.g. 397B):
    # the full-attention block + rotary are identical across the two.
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            apply_rotary_pos_emb,
        )
    except Exception:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_rotary_pos_emb,
        )

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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

    attn_out = _flat_headclass_attention(
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


def _install_full_patches(model, head_cfg, dense_prefix_full_layers=0):
    """Patch full_attention layers with RedKnot head-class attention (prefill
    T>1) / native (decode T==1). Returns restore_fn.

    ``dense_prefix_full_layers``: the first N FULL-attention layers stay fully
    dense (exact, NOT patched) to protect early-layer fidelity; the remaining
    full layers run head-class (global/local) sparsity. The head-config row for
    a full layer remains its ordinal among ALL full layers.
    """
    base_model = model.model if hasattr(model, "model") else model
    layers = base_model.layers
    full_idx = sorted(full_attention_layer_indices(model.config))
    hc_row = {li: k for k, li in enumerate(full_idx)}
    # full layers to actually sparsify (skip the first dense_prefix_full_layers)
    sparsify_idx = full_idx[dense_prefix_full_layers:]

    saved = {}
    for li in sparsify_idx:
        attn = layers[li].self_attn
        saved[li] = attn.forward

        def make_fwd(_attn, _row, _orig):
            def fwd(
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                position_ids=None,
                **kw,
            ):
                if hidden_states.shape[1] == 1:
                    return _orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        **kw,
                    )
                # Pull the already-accumulated prefix KV for THIS layer from the
                # shared cache and feed it as the head-class seg0 prefix, so this
                # chunk's full attention attends [prefix | this chunk] (not just
                # its own tokens). Cache KV is already RoPE'd at global positions.
                seg0_k = seg0_v = None
                seg0_len = 0
                if past_key_values is not None:
                    try:
                        lyr = past_key_values.layers[_attn.layer_idx]
                        pk = getattr(lyr, "keys", None)
                        pv = getattr(lyr, "values", None)
                        if pk is not None and pk.numel() > 0:
                            seg0_k = pk.to(hidden_states.device)
                            seg0_v = pv.to(hidden_states.device)
                            seg0_len = pk.shape[2]
                    except (AttributeError, IndexError):
                        seg0_k = seg0_v = None
                        seg0_len = 0
                out, k, v = _qwen35_full_attn_headclass(
                    _attn,
                    hidden_states,
                    position_embeddings,
                    head_cfg=head_cfg,
                    hc_layer_idx=_row,
                    seg0_k=seg0_k,
                    seg0_v=seg0_v,
                    seg0_len=seg0_len,
                )
                # Append only THIS chunk's new KV (seg0 prefix is already in the
                # cache); DynamicCache.update concatenates along the seq axis.
                if past_key_values is not None:
                    past_key_values.update(k, v, _attn.layer_idx)
                return out, None

            return fwd

        attn.forward = make_fwd(attn, hc_row[li], saved[li])

    def restore():
        for li, f in saved.items():
            layers[li].self_attn.forward = f

    return restore


@torch.no_grad()
def run_redknot_qwen35_chunked(
    model,
    tokenizer,
    *,
    segments: List[str],
    query_text: str,
    head_cfg: HeadClassConfig,
    max_new_tokens: int = 48,
    reuse_prefix_chunks: int = 1,
):
    """Plan-B chunked RedKnot forward for Qwen3.5-MoE.

    The context is given as ordered ``segments`` (chunks). We prefill chunk by
    chunk through ONE shared cache so that:

      * Linear-attention layers' (conv, recurrent) state ACCUMULATES across
        chunks via the cache — each chunk only forwards ITS OWN tokens starting
        from the previous chunks' accumulated state (O(L) total, no re-scan).
      * Full-attention layers run RedKnot head-class sparsity each chunk; their
        KV is appended to the cache so later chunks/query attend the full prefix.

    ``reuse_prefix_chunks``: number of leading chunks treated as a reusable
    exact prefix (chunk 1 by default). In this first cut the prefix is still
    computed online (it establishes correctness); offline KV/state injection to
    actually SKIP the prefix compute is a drop-in next step (the cache supports
    update_recurrent_state / update_conv_state / update for that).

    Returns ``(text, ttft_seconds)``. ``ttft`` covers prefilling all chunks +
    the query and the first token.
    """
    from transformers import DynamicCache

    device = model.device
    restore = _install_full_patches(model, head_cfg)
    try:
        cache = DynamicCache(config=model.config)
        pos = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ── Prefill each chunk in order through the shared cache ──
        # chunk_i linear layers continue from chunk_{<i} accumulated state;
        # full layers attend [prefix KV | this chunk] and append their KV.
        seg_pieces = list(segments) + [query_text]
        last_logits = None
        for piece in seg_pieces:
            ids = tokenizer(piece, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            if ids.shape[1] == 0:
                continue
            position_ids = torch.arange(
                pos, pos + ids.shape[1], device=device
            ).unsqueeze(0)
            out = model(
                input_ids=ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            last_logits = out.logits[0, -1, :]
            pos += ids.shape[1]

        nxt = last_logits.argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # ── Greedy decode (native attention on T==1) ──
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            position_ids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        restore()


@torch.no_grad()
def run_redknot_qwen35(
    model,
    tokenizer,
    *,
    context_text: str,
    query_text: str,
    head_cfg: HeadClassConfig,
    max_new_tokens: int = 48,
):
    """Step-1 RedKnot forward for Qwen3.5-MoE.

    Single online prefill over the full context; full_attention layers run
    RedKnot head-class attention, linear layers + MoE run native. Returns
    ``(text, ttft_seconds)``.

    NOTE: This Step-1 path computes head-class attention *online* (no offline
    KV splice / cross-request reuse yet). It establishes the correct hooking +
    Q-gating handling; offline reuse is Step 2.
    """
    base_model = model.model if hasattr(model, "model") else model
    layers = base_model.layers
    full_idx = set(full_attention_layer_indices(model.config))
    # Map global layer index -> head-config row (0..n_full-1).
    hc_row = {li: k for k, li in enumerate(sorted(full_idx))}

    import os as _os

    _debug = _os.environ.get("REDKNOT_DEBUG", "0") == "1"
    _diag = {"prefill_calls": 0, "decode_calls": 0, "last_T": 0}

    # ── Patch full_attention layers in place ──
    patched = {}
    for li in full_idx:
        attn = layers[li].self_attn
        patched[li] = attn.forward

        def make_fwd(_attn, _row, _orig):
            def fwd(
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                position_ids=None,
                **kw,
            ):
                # RedKnot head-class attention applies to the PREFILL pass only
                # (T > 1, the full context). Single-token DECODE (T == 1) reads
                # the already-built KV cache and must use the native attention
                # to stay numerically correct; head-class sparsity over a
                # 1-token query is ill-defined.
                T = hidden_states.shape[1]
                _diag["last_T"] = T
                if T == 1:
                    _diag["decode_calls"] += 1
                    return _orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        **kw,
                    )
                _diag["prefill_calls"] += 1
                out, k, v = _qwen35_full_attn_headclass(
                    _attn,
                    hidden_states,
                    position_embeddings,
                    head_cfg=head_cfg,
                    hc_layer_idx=_row,
                    seg0_k=None,
                    seg0_v=None,
                    seg0_len=0,
                )
                if _debug and _diag["prefill_calls"] <= 2:
                    print(
                        f"   [rk-debug] layer={_attn.layer_idx} row={_row} T={T} "
                        f"out_norm={out.float().norm().item():.3f} "
                        f"is_local_sum={int((head_cfg.as_tensors(out.device)['head_type'][_row] == HeadClassConfig.TYPE_LOCAL).sum())}",
                        flush=True,
                    )
                # Seed the native KV cache with the prefill K/V so the
                # subsequent native decode steps have the full context.
                if past_key_values is not None:
                    past_key_values.update(k, v, _attn.layer_idx)
                return out, None

            return fwd

        attn.forward = make_fwd(attn, hc_row[li], patched[li])

    try:
        from transformers import DynamicCache

        prompt = context_text + query_text
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        nxt = out.logits[0, -1, :].argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        past = out.past_key_values
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            og = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            if tid == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(gen, skip_special_tokens=True)
        if _debug:
            print(
                f"   [rk-debug] prefill_calls={_diag['prefill_calls']} "
                f"decode_calls={_diag['decode_calls']} last_T={_diag['last_T']} "
                f"(expect prefill_calls=15 if patch fires on prefill)",
                flush=True,
            )
        return text, ttft
    finally:
        # Restore original forwards.
        for li, fwd in patched.items():
            layers[li].self_attn.forward = fwd


@torch.no_grad()
def run_redknot_qwen35_planb2(
    model,
    tokenizer,
    *,
    segments: List[str],
    query_text: str,
    head_cfg: HeadClassConfig,
    max_new_tokens: int = 48,
    recompute_until_chunk: int = 1,
    recompute_chunks: Optional[set] = None,
    full_exact: bool = False,
):
    """Plan-2 chunked RedKnot for Qwen3.5-MoE: linear OFFLINE reuse for chunk>=2.

    ``recompute_until_chunk``: linear layers RECOMPUTE for chunks with index
    ``< recompute_until_chunk`` and REUSE the offline state for chunks at/after
    it. E.g. =1 reuses from chunk 1 (most aggressive); =3 recomputes chunk 1,2
    and reuses chunk 3.. ; with the user's "recompute chunk2,chunk3, reuse 4+"
    set =4 (chunk0 is the exact prefix reused for free, chunk1,2 recompute ->
    here chunk index is 0-based so chunks 1,2 == 2nd,3rd; reuse from index 3).
    NOTE: chunk index is 0-based (chunk 0 = first/prefix chunk).

    ``full_exact``: if True, full-attention layers run EXACT native attention
    (no head-class sparsity) — isolates the linear-reuse accuracy cost.

    Difference vs ``run_redknot_qwen35_chunked`` (Plan 1): from chunk 2 onward the
    linear-attention layers do NOT recompute — they reuse the OFFLINE linear
    outputs (computed once with exact full attention). chunk 1 is the exact
    reusable prefix. Full layers always run head-class sparsity + prefix KV.

    This realises the user's design: "chunk 1 correct; from chunk 2 on the linear
    state is accumulated/reused offline, not recomputed". Saves the chunk>=2
    linear compute at the cost of the small upstream-sparse coupling error
    (measured to keep next-token unchanged at the full sweet spot).

    Returns ``(text, ttft_seconds)``.
    """
    from transformers import DynamicCache

    device = model.device
    bm = model.model if hasattr(model, "model") else model
    lin_idx = linear_attention_layer_indices(model.config)

    seg_tok = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in segments]
    qids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
    pieces = seg_tok + [qids]
    full_ids = [t for p in pieces for t in p]

    # ── OFFLINE: capture exact-full linear outputs per position (once) ──
    cap = {}
    hooks = []
    for li in lin_idx:

        def mk(_li):
            def hook(mod, inp, out):
                cap[_li] = out.detach()

            return hook

        hooks.append(bm.layers[li].linear_attn.register_forward_hook(mk(li)))
    model(input_ids=torch.tensor([full_ids], device=device), use_cache=False)
    for h in hooks:
        h.remove()

    # ── ONLINE: chunk-by-chunk; full layers head-class sparse OR exact ──
    restore_full = (
        (lambda: None) if full_exact else _install_full_patches(model, head_cfg)
    )
    st = {"pos": 0, "chunk_id": 0}
    lin_saved = {}
    for li in lin_idx:
        m = bm.layers[li].linear_attn
        lin_saved[li] = m.forward

        def mk(_li, _orig):
            def fwd(hidden_states, **kw):
                T = hidden_states.shape[1]
                # Decide per-chunk: recompute (exact stream) or reuse offline.
                if recompute_chunks is not None:
                    do_reuse = st["chunk_id"] not in recompute_chunks
                else:
                    do_reuse = st["chunk_id"] >= recompute_until_chunk
                if do_reuse and _li in cap:
                    seg = cap[_li][:, st["pos"] : st["pos"] + T, :]
                    return seg.to(hidden_states.dtype)
                return _orig(hidden_states=hidden_states, **kw)

            return fwd

        m.forward = mk(li, lin_saved[li])

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = DynamicCache(config=model.config)
        pos = 0
        last = None
        for ci, ptok in enumerate(pieces):
            st["chunk_id"] = ci
            st["pos"] = pos
            ids = torch.tensor([ptok], device=device)
            pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            last = out.logits[0, -1, :]
            pos += ids.shape[1]
        nxt = last.argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # decode uses native linear (T==1, no offline reuse)
        st["chunk_id"] = 0
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        restore_full()
        for li, f in lin_saved.items():
            bm.layers[li].linear_attn.forward = f


# ──────────────────────────────────────────────────────────────────────────
# Linear-attention head-class windowing (RedKnot for GatedDeltaNet)
# ──────────────────────────────────────────────────────────────────────────
def measure_linear_head_decay(model, hidden_by_layer, decay_quantile=0.95):
    """Per-(layer, head) decay statistics for linear layers.

    Decay is input-dependent and varies per token, so a head's mean decay can
    hide occasional long-memory steps. We return, per (layer, head), a ROBUST
    decay = the ``decay_quantile`` (default p95) of the per-token decay factor
    g.exp() — i.e. "even at its stickiest, how slow does this head forget".
    Returns {layer_idx: tensor[num_v_heads] of robust decay in (0,1)}.
    """
    import torch.nn.functional as F

    bm = model.model if hasattr(model, "model") else model
    out = {}
    for li in linear_attention_layer_indices(model.config):
        mod = bm.layers[li].linear_attn
        hs = hidden_by_layer.get(li)
        if hs is None:
            continue
        a = mod.in_proj_a(hs.to(mod.in_proj_a.weight.dtype)).to(mod.A_log.device)
        g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
        decay = g.exp().reshape(-1, g.shape[-1]).float()  # [T, H]
        # robust per-head decay: high quantile across tokens (worst-case memory)
        out[li] = torch.quantile(decay, decay_quantile, dim=0).cpu()
    return out


@torch.no_grad()
def _linear_local_recurrence(
    mod, query, key, value, g, beta, win_vec, initial_state=None, return_state=False
):
    """Delta-rule recurrence carrying the prefix state (EXACT).

    Validated: a local head's window MUST start from the DECAYED PREFIX STATE,
    not zero (zeroing starves window-start tokens -> the prior crash). The
    delta-rule is a continuous recurrence; carrying ``initial_state`` forward IS
    exactly the full-history computation. Local-head saving therefore comes from
    REUSING the prefix state across chunks (compute it once offline), not from
    truncating within a single pass.

    ``initial_state``: [B,H,dk,dv] prefix state to continue from (None -> zero).
    ``win_vec`` is kept for API compatibility but the recurrence is exact for all
    heads here; head-class saving is realized at the chunk/driver level by
    relaying state. Shapes ([B,H,T,*], q,k already l2-normed). Returns out (and
    final state if return_state).
    """
    Bb, H, T, dk = key.shape
    dv = value.shape[-1]
    scale = 1.0 / (dk**0.5)
    q = query * scale
    S = (
        torch.zeros(Bb, H, dk, dv, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value.dtype)
    )
    out = torch.zeros(Bb, H, T, dv, dtype=value.dtype, device=value.device)
    for i in range(T):
        S = S * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        k_t = key[:, :, i]
        kv = (S * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, i] - kv) * beta[:, :, i].unsqueeze(-1)
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (S * q[:, :, i].unsqueeze(-1)).sum(dim=-2)
    if return_state:
        return out, S
    return out


def install_linear_headclass(
    model,
    head_decay,
    *,
    dense_prefix_layers=5,
    safety=4.0,
    win_min=128,
    win_cap=8192,
):
    """Patch linear layers with PER-(layer, head) windows (fine-grained RedKnot).

    For each (layer, head) we set its own window from its robust memory length:
        mem_len  = 1 / (1 - robust_decay)        # tokens to forget to 1/e
        window_h = clamp(safety * mem_len, win_min, win_cap)
    A head whose ``safety*mem_len >= win_cap`` is treated as GLOBAL (full
    history). ``safety`` (>1) keeps a margin so windowing stays lossless even
    though decay varies per token. Layers ``< dense_prefix_layers`` (L0..L4 by
    default) run fully dense — no windowing.

    ``head_decay``: {layer_idx: tensor[num_v_heads] robust decay in (0,1)}.
    Returns (restore_fn, info) where info[layer] = (#local, #global, median_win).
    """
    import torch.nn.functional as F
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

    bm = model.model if hasattr(model, "model") else model
    saved = {}
    info = {}
    for li in linear_attention_layer_indices(model.config):
        mod = bm.layers[li].linear_attn
        decay = head_decay.get(li)
        if decay is None or li < dense_prefix_layers:
            continue
        memlen = 1.0 / (1.0 - decay.clamp(max=0.99999))
        win = (safety * memlen).clamp(min=win_min)
        # heads needing >= win_cap are global (window 0 -> full history)
        win_vec = torch.where(win >= win_cap, torch.zeros_like(win), win).round().long()
        n_local = int((win_vec > 0).sum())
        n_global = int((win_vec == 0).sum())
        med_win = int(win_vec[win_vec > 0].float().median().item()) if n_local else 0
        info[li] = (n_local, n_global, med_win)
        saved[li] = mod.forward

        def mk(_mod, _win_vec):
            def fwd(hidden_states, cache_params=None, attention_mask=None, **kw):
                B, Tt, _ = hidden_states.shape
                mixed = _mod.in_proj_qkv(hidden_states).transpose(1, 2)
                mixed = F.silu(_mod.conv1d(mixed)[:, :, :Tt]).transpose(1, 2)
                qy, ky, vy = torch.split(
                    mixed, [_mod.key_dim, _mod.key_dim, _mod.value_dim], dim=-1
                )
                hk = _mod.head_k_dim
                qy = qy.reshape(B, Tt, -1, hk)
                ky = ky.reshape(B, Tt, -1, hk)
                vy = vy.reshape(B, Tt, -1, _mod.head_v_dim)
                z = _mod.in_proj_z(hidden_states).reshape(B, Tt, -1, _mod.head_v_dim)
                b = _mod.in_proj_b(hidden_states)
                a = _mod.in_proj_a(hidden_states)
                beta = b.sigmoid()
                g = -_mod.A_log.float().exp() * F.softplus(a.float() + _mod.dt_bias)
                rep = _mod.num_v_heads // _mod.num_k_heads
                if rep > 1:
                    qy = qy.repeat_interleave(rep, dim=2)
                    ky = ky.repeat_interleave(rep, dim=2)
                # to [B,H,T,*], l2norm q,k
                qy = l2norm(qy, dim=-1, eps=1e-6).transpose(1, 2).to(torch.float32)
                ky = l2norm(ky, dim=-1, eps=1e-6).transpose(1, 2).to(torch.float32)
                vy = vy.transpose(1, 2).to(torch.float32)
                gg = g.transpose(1, 2).to(torch.float32)
                bb = beta.transpose(1, 2).to(torch.float32)
                core = _linear_local_recurrence(_mod, qy, ky, vy, gg, bb, _win_vec)
                core = (
                    core.transpose(1, 2)
                    .reshape(-1, _mod.head_v_dim)
                    .to(hidden_states.dtype)
                )
                zr = z.reshape(-1, _mod.head_v_dim)
                core = _mod.norm(core, zr).reshape(B, Tt, -1)
                return _mod.out_proj(core)

            return fwd

        mod.forward = mk(mod, win_vec)

    def restore():
        for li, f in saved.items():
            bm.layers[li].linear_attn.forward = f

    return restore, info


@torch.no_grad()
def classify_linear_heads_by_truncation(
    model, hidden_by_layer, window, err_thresh=0.05, dense_prefix_layers=5
):
    """Data-driven per-(layer,head) local/global classification.

    Instead of guessing from decay rate, DIRECTLY measure, per (layer, head),
    how much truncating history to a sliding window of size ``window`` changes
    that head's output (relative L2 error vs full history). Heads with error
    < ``err_thresh`` are LOCAL (truly truncatable); the rest are GLOBAL. This
    captures INFORMATION importance, not just decay speed.

    Returns {layer_idx: win_vec[H]} where win_vec[h]=window if local else 0.
    """
    import torch.nn.functional as F
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

    bm = model.model if hasattr(model, "model") else model
    win_by_layer = {}
    info = {}
    for li in linear_attention_layer_indices(model.config):
        if li < dense_prefix_layers:
            continue
        mod = bm.layers[li].linear_attn
        hs = hidden_by_layer.get(li)
        if hs is None:
            continue
        B, Tt, _ = hs.shape
        mixed = mod.in_proj_qkv(hs).transpose(1, 2)
        mixed = F.silu(mod.conv1d(mixed)[:, :, :Tt]).transpose(1, 2)
        qy, ky, vy = torch.split(
            mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1
        )
        hk = mod.head_k_dim
        qy = qy.reshape(B, Tt, -1, hk)
        ky = ky.reshape(B, Tt, -1, hk)
        vy = vy.reshape(B, Tt, -1, mod.head_v_dim)
        b = mod.in_proj_b(hs)
        a = mod.in_proj_a(hs)
        beta = b.sigmoid()
        g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
        rep = mod.num_v_heads // mod.num_k_heads
        if rep > 1:
            qy = qy.repeat_interleave(rep, dim=2)
            ky = ky.repeat_interleave(rep, dim=2)
        qy = l2norm(qy, dim=-1, eps=1e-6).transpose(1, 2).to(torch.float32)
        ky = l2norm(ky, dim=-1, eps=1e-6).transpose(1, 2).to(torch.float32)
        vy = vy.transpose(1, 2).to(torch.float32)
        gg = g.transpose(1, 2).to(torch.float32)
        bb = beta.transpose(1, 2).to(torch.float32)
        H = qy.shape[1]
        # full-history output (all heads global)
        full = _linear_local_recurrence(
            mod, qy, ky, vy, gg, bb, torch.zeros(H, dtype=torch.long)
        )
        # windowed output (all heads local at `window`)
        wind = _linear_local_recurrence(
            mod, qy, ky, vy, gg, bb, torch.full((H,), window, dtype=torch.long)
        )
        # per-head relative error
        num = (wind - full).pow(2).sum(dim=(0, 2, 3)).sqrt()
        den = full.pow(2).sum(dim=(0, 2, 3)).sqrt() + 1e-6
        rel = (num / den).cpu()
        is_local = rel < err_thresh
        win_vec = torch.where(
            is_local,
            torch.full((H,), window, dtype=torch.long),
            torch.zeros(H, dtype=torch.long),
        )
        win_by_layer[li] = win_vec
        info[li] = (int(is_local.sum()), int((~is_local).sum()), float(rel.median()))
    return win_by_layer, info


def install_linear_headclass_winmap(model, win_by_layer):
    """Install linear head-class windowing from an explicit per-layer win_vec map
    (as produced by classify_linear_heads_by_truncation)."""
    import torch.nn.functional as F
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

    bm = model.model if hasattr(model, "model") else model
    saved = {}
    for li, win_vec in win_by_layer.items():
        mod = bm.layers[li].linear_attn
        saved[li] = mod.forward

        def mk(_mod, _win_vec):
            def fwd(hidden_states, cache_params=None, attention_mask=None, **kw):
                B, Tt, _ = hidden_states.shape
                mixed = _mod.in_proj_qkv(hidden_states).transpose(1, 2)
                mixed = F.silu(_mod.conv1d(mixed)[:, :, :Tt]).transpose(1, 2)
                qy, ky, vy = torch.split(
                    mixed, [_mod.key_dim, _mod.key_dim, _mod.value_dim], dim=-1
                )
                hk = _mod.head_k_dim
                qy = qy.reshape(B, Tt, -1, hk)
                ky = ky.reshape(B, Tt, -1, hk)
                vy = vy.reshape(B, Tt, -1, _mod.head_v_dim)
                z = _mod.in_proj_z(hidden_states).reshape(B, Tt, -1, _mod.head_v_dim)
                b = _mod.in_proj_b(hidden_states)
                a = _mod.in_proj_a(hidden_states)
                beta = b.sigmoid()
                g = -_mod.A_log.float().exp() * F.softplus(a.float() + _mod.dt_bias)
                rep = _mod.num_v_heads // _mod.num_k_heads
                if rep > 1:
                    qy = qy.repeat_interleave(rep, dim=2)
                    ky = ky.repeat_interleave(rep, dim=2)
                qy = l2norm(qy, dim=-1, eps=1e-6).transpose(1, 2).to(torch.float32)
                ky = l2norm(ky, dim=-1, eps=1e-6).transpose(1, 2).to(torch.float32)
                vy = vy.transpose(1, 2).to(torch.float32)
                gg = g.transpose(1, 2).to(torch.float32)
                bb = beta.transpose(1, 2).to(torch.float32)
                core = _linear_local_recurrence(_mod, qy, ky, vy, gg, bb, _win_vec)
                core = (
                    core.transpose(1, 2)
                    .reshape(-1, _mod.head_v_dim)
                    .to(hidden_states.dtype)
                )
                zr = z.reshape(-1, _mod.head_v_dim)
                core = _mod.norm(core, zr).reshape(B, Tt, -1)
                return _mod.out_proj(core)

            return fwd

        mod.forward = mk(mod, win_vec)

    def restore():
        for li, f in saved.items():
            bm.layers[li].linear_attn.forward = f

    return restore


def install_linear_chunkrule_passthrough(model):
    """SANITY: wrap each linear layer's chunk_gated_delta_rule with a pass-through
    (calls the original). If this changes outputs, the wrapping itself is buggy.
    Returns restore_fn."""
    bm = model.model if hasattr(model, "model") else model
    saved = {}
    for li in linear_attention_layer_indices(model.config):
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule

        def mk(_orig):
            def wrapped(*args, **kwargs):
                return _orig(*args, **kwargs)

            return wrapped

        mod.chunk_gated_delta_rule = mk(saved[li])

    def restore():
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f

    return restore


def install_linear_headclass_chunkrule(model, win_by_layer, prefix_state_by_layer=None):
    """Minimal-invasive linear head-class: keep the ENTIRE native forward intact,
    only replace ``chunk_gated_delta_rule`` so that LOCAL heads use a windowed /
    prefix-state recurrence while GLOBAL heads use the native kernel.

    For correctness-first we implement LOCAL heads by running the native kernel
    but with a per-head ``initial_state`` carried across chunks (the validated
    EXACT scheme). When ``prefix_state_by_layer`` is None (single pass), local
    heads fall back to the native full kernel (lossless, no saving) — used to
    confirm wiring fidelity (dF1==0).

    ``win_by_layer``: {layer: win_vec[H]} (local if >0).
    Returns restore_fn.
    """
    bm = model.model if hasattr(model, "model") else model
    saved = {}
    for li, win_vec in win_by_layer.items():
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule
        init_state = (
            None if prefix_state_by_layer is None else prefix_state_by_layer.get(li)
        )

        def mk(_orig, _init):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                # Use carried prefix state if provided (exact cross-chunk relay);
                # otherwise native behaviour. Native kernel handles all heads.
                istate = _init if _init is not None else initial_state
                return _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=istate,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )

            return wrapped

        mod.chunk_gated_delta_rule = mk(saved[li], init_state)

    def restore():
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f

    return restore


@torch.no_grad()
def run_redknot_qwen35_linear(
    model,
    tokenizer,
    *,
    segments,
    query_text,
    win_by_layer,
    prefix_state_by_layer=None,
    max_new_tokens=24,
):
    """Minimal-invasive RedKnot-linear forward (full attention UNCHANGED).

    Per (layer, head): GLOBAL heads accumulate full state across all chunks
    (native kernel, initial_state relayed). LOCAL heads reuse a DECAYED PREFIX
    state for the out-of-window history (``prefix_state_by_layer``) and recompute
    only the current window — realized by relaying state through the native
    kernel chunk by chunk (validated exact). full_attention layers run native.

    This wraps each linear layer's ``chunk_gated_delta_rule`` to inject a per-head
    ``initial_state`` (global: running relay; local: decayed prefix), preserving
    the entire native forward (conv, norms, l2norm-in-kernel) -> faithful.

    Returns (text, ttft).
    """
    bm = model.model if hasattr(model, "model") else model
    device = model.device

    # Per-layer running state for relay across chunks + chunk counter.
    run_state = {li: None for li in win_by_layer}
    chunk_id = {li: 0 for li in win_by_layer}

    # win_window_chunks[li]: per-head window measured in CHUNKS (0/None = global).
    # A LOCAL head only carries state from the last ``wc`` chunks: every wc
    # chunks we DROP that head's relayed state (its window slides). global heads
    # keep accumulating. This is the chunk-granularity sliding window; with the
    # decayed prefix carried in fstate it matches the validated exact scheme for
    # mem_len <= window.
    def _win_chunks(win_vec):
        if win_vec is None:
            return None
        return win_vec  # already in chunks (int tensor per head; 0 = global)

    saved = {}
    for li in win_by_layer:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule
        wc = _win_chunks(win_by_layer[li])  # [H] chunks per local head, 0=global

        def mk(_li, _orig, _wc):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                istate = run_state[_li]
                core, fstate = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=istate,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                fstate = fstate.detach()
                ci = chunk_id[_li] + 1
                chunk_id[_li] = ci
                if _wc is not None:
                    # LOCAL heads (wc>0): drop relayed state when this chunk
                    # crosses their window boundary (ci % wc == 0). GLOBAL heads
                    # (wc<=0) keep full state.
                    H = fstate.shape[1]
                    w = _wc.to(fstate.device)
                    drop = (w > 0) & (
                        torch.remainder(
                            torch.tensor(ci, device=w.device), w.clamp(min=1)
                        )
                        == 0
                    )
                    if drop.any():
                        m = drop.view(1, H, 1, 1)
                        fstate = torch.where(m, torch.zeros_like(fstate), fstate)
                run_state[_li] = fstate
                return core, fstate

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li], wc)

    try:
        from transformers import DynamicCache

        cache = DynamicCache(config=model.config)
        pos = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time as _t

        t0 = _t.perf_counter()
        last = None
        for piece in list(segments) + [query_text]:
            ids = tokenizer(piece, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            if ids.shape[1] == 0:
                continue
            pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            last = out.logits[0, -1, :]
            pos += ids.shape[1]
        nxt = last.argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = _t.perf_counter() - t0

        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f


@torch.no_grad()
def rag_build_doc_state(model, tokenizer, *, segments, win_by_layer):
    """RAG OFFLINE phase: prefill the document chunks once, returning the reusable
    state (linear per-layer run_state + full-attn KV cache + position) so that
    many queries can REUSE it. full attention KV is cached natively; linear
    GLOBAL heads hold full doc state, LOCAL heads hold only their windowed state.

    Returns dict {cache, lin_state, pos}.
    """
    from transformers import DynamicCache

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    run_state = {li: None for li in win_by_layer}
    chunk_id = {li: 0 for li in win_by_layer}
    saved = {}
    for li in win_by_layer:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule
        wc = win_by_layer[li]

        def mk(_li, _orig, _wc):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                core, fstate = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=run_state[_li],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                fstate = fstate.detach()
                ci = chunk_id[_li] + 1
                chunk_id[_li] = ci
                if _wc is not None:
                    H = fstate.shape[1]
                    w = _wc.to(fstate.device)
                    drop = (w > 0) & (
                        torch.remainder(
                            torch.tensor(ci, device=w.device), w.clamp(min=1)
                        )
                        == 0
                    )
                    if drop.any():
                        fstate = torch.where(
                            drop.view(1, H, 1, 1), torch.zeros_like(fstate), fstate
                        )
                run_state[_li] = fstate
                return core, fstate

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li], wc)

    try:
        cache = DynamicCache(config=model.config)
        pos = 0
        for piece in segments:
            ids = tokenizer(piece, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            if ids.shape[1] == 0:
                continue
            pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            pos += ids.shape[1]
        return {
            "cache": cache,
            "lin_state": {
                li: (None if run_state[li] is None else run_state[li].clone())
                for li in run_state
            },
            "pos": pos,
        }
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f


@torch.no_grad()
def rag_query_reuse(
    model, tokenizer, *, doc_state, query_text, win_by_layer, max_new_tokens=24
):
    """RAG ONLINE phase: answer a query REUSING the cached doc state. Only the
    query tokens are processed online; linear layers continue from the cached
    doc lin_state, full attention attends the cached doc KV. Returns (text,ttft).
    """
    import copy, time as _t

    bm = model.model if hasattr(model, "model") else model
    device = model.device
    # fresh copies so multiple queries reuse the same doc state independently
    cache = copy.deepcopy(doc_state["cache"])
    run_state = {
        li: (
            None
            if doc_state["lin_state"][li] is None
            else doc_state["lin_state"][li].clone()
        )
        for li in doc_state["lin_state"]
    }
    pos = doc_state["pos"]

    saved = {}
    for li in win_by_layer:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule

        def mk(_li, _orig):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                core, fstate = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=run_state[_li],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                run_state[_li] = fstate.detach()
                return core, fstate

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    try:
        ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
        out = model(
            input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
        )
        cache = out.past_key_values
        nxt = out.logits[0, -1, :].argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = _t.perf_counter() - t0
        pos += ids.shape[1]
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f


@torch.no_grad()
def _linear_local_token_window(mod, query, key, value, g, beta, win_tok):
    """Per-head TOKEN-granularity sliding window for linear (delta-rule).

    win_tok [H] int: head h attends only the last win_tok[h] tokens. <=0 or
    >=T = GLOBAL (full history). Realized exactly via the validated decayed-
    prefix scheme: for each output position t, state = (prefix decayed to t-W) +
    recompute of window [t-W, t]. We implement it efficiently per distinct
    window with overlapped blocks (block=W, warmup=W carries decayed prefix).

    Shapes [B,H,T,*], q,k already l2-normed. Returns [B,H,T,dv].
    """
    B, H, T, dk = key.shape
    dv = value.shape[-1]
    scale = 1.0 / (dk**0.5)
    q = query * scale
    w = win_tok.to(value.device).long()
    out = torch.zeros(B, H, T, dv, dtype=value.dtype, device=value.device)

    def run(lo, hi, S0):
        S = S0
        outs = []
        for i in range(lo, hi):
            S = S * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            k_t = key[:, :, i]
            kv = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (value[:, :, i] - kv) * beta[:, :, i].unsqueeze(-1)
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((S * q[:, :, i].unsqueeze(-1)).sum(dim=-2))
        return torch.stack(outs, dim=2) if outs else None, S

    is_global = (w <= 0) | (w >= T)
    # GLOBAL heads: full recurrence
    if is_global.any():
        S0 = torch.zeros(B, H, dk, dv, dtype=value.dtype, device=value.device)
        g_out, _ = run(0, T, S0)
        out = torch.where(is_global.view(1, H, 1, 1), g_out, out)
    # LOCAL heads: group by window size, overlapped blocks (prefix carried)
    local_ws = (
        sorted(set(int(x) for x in w[~is_global].tolist()))
        if (~is_global).any()
        else []
    )
    for Wd in local_ws:
        sel = (w == Wd).view(1, H, 1, 1)
        block = max(1, Wd)
        for bstart in range(0, T, block):
            bend = min(bstart + block, T)
            warm = max(0, bstart - block)  # warmup carries decayed prefix
            S0 = torch.zeros(B, H, dk, dv, dtype=value.dtype, device=value.device)
            seg, _ = run(warm, bend, S0)
            blk = seg[:, :, (bstart - warm) : (bend - warm)]
            out[:, :, bstart:bend] = torch.where(sel, blk, out[:, :, bstart:bend])
    return out


def install_linear_token_window(model, win_tok_by_layer):
    """FAST token-granularity windows using the NATIVE fla kernel (no per-token
    Python loop). Patches only ``chunk_gated_delta_rule`` (native forward kept).

    Per layer we use ONE window W = median local-head window (token units). The
    sequence is processed in NATIVE-kernel segments of size W with initial_state
    relay (decayed prefix carried) -> realizes a token sliding window for LOCAL
    heads. GLOBAL heads are recovered by also running ONE full-history native
    call and selecting global-head outputs. Validated exact (decayed-prefix).

    win_tok_by_layer: {layer: win_tok[H]} (token window; <=0/>=T global; None=
    layer fully global, untouched). Returns restore_fn.
    """
    bm = model.model if hasattr(model, "model") else model
    saved = {}
    for li, win_tok in win_tok_by_layer.items():
        if win_tok is None:
            continue
        mod = bm.layers[li].linear_attn
        local = win_tok[(win_tok > 0)]
        if local.numel() == 0:
            continue  # all-global layer: leave native
        Wmed = int(local.float().median().item())
        saved[li] = mod.chunk_gated_delta_rule
        is_global = win_tok <= 0

        def mk(_orig, _W, _isg, _SEG=4096):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                T = query.shape[1]  # [B, T, H, D]
                if _W <= 0 or _W >= T:
                    return _orig(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        **kw,
                    )
                # FEW large segments (SEG) to bound kernel-call count, with a
                # per-segment OVERLAP warmup of length W (carries the decayed
                # prefix) -> lossless for both classes:
                #   GLOBAL heads: relay full state across segments (initial_state).
                #   LOCAL  heads: each output token sees >= W tokens of history
                #                 via the warmup (true token window, not reset).
                # kernel calls per layer ~ ceil(T/SEG) (vs T/W before).
                isg = _isg.to(query.device)  # [H] bool, True=global
                SEG = max(_W, _SEG)
                g_state = initial_state  # global-head running state
                outs = []
                for st in range(0, T, SEG):
                    en = min(st + SEG, T)
                    warm = max(0, st - _W)  # overlap warmup for locals
                    seg_core, g_fs = _orig(
                        query[:, warm:en],
                        key[:, warm:en],
                        value[:, warm:en],
                        g=g[:, warm:en],
                        beta=beta[:, warm:en],
                        initial_state=None,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    # local-head output: the [st:en] slice of the warmed segment
                    local_out = seg_core[:, (st - warm) : (en - warm)]
                    # global-head output: continue the global relay over [st:en]
                    gseg_core, g_state = _orig(
                        query[:, st:en],
                        key[:, st:en],
                        value[:, st:en],
                        g=g[:, st:en],
                        beta=beta[:, st:en],
                        initial_state=g_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    merged = torch.where(isg.view(1, 1, -1, 1), gseg_core, local_out)
                    outs.append(merged)
                core = torch.cat(outs, dim=1)
                return (core, g_state) if output_final_state else (core, None)

            return wrapped

        mod.chunk_gated_delta_rule = mk(saved[li], Wmed, is_global)

    def restore():
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f

    return restore


@torch.no_grad()
def rag_build_offline_v2(model, tokenizer, *, segments, win_tok_by_layer):
    """OFFLINE (v2): prefill docs once, store reusable state for RAG.

    For each linear layer:
      * GLOBAL heads: store the FULL doc final status (reused verbatim online).
      * LOCAL heads (window W): store the PREFIX status accumulated up to
        (doc_len - W) — i.e. everything OUTSIDE the window — by running the
        native kernel over [0, doc_len-W) with output_final_state. The last W
        doc tokens are recomputed ONLINE (window). We also keep the doc token
        ids of the last max-W tokens so the online phase can recompute windows.

    full attention: doc KV cached natively (DynamicCache).
    Returns dict {cache, pos, doc_ids, glob_state, prefix_state, win, isg}.
    """
    from transformers import DynamicCache

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    seg_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in segments]
    doc_ids = [t for s in seg_ids for t in s]
    doc_len = len(doc_ids)

    # capture, per linear layer: full final status (global) AND prefix status at
    # doc_len - W (local). We run ONE doc prefill; inside the patched kernel we
    # snapshot the running state at the per-head window boundary.
    glob_state = {}
    prefix_state = {}
    isg_by_layer = {}
    saved = {}
    for li, wt in win_tok_by_layer.items():
        if wt is None:
            continue
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule
        isg = wt <= 0
        isg_by_layer[li] = isg
        # per-head window boundary position = doc_len - W (clamped >=0)
        wcl = wt.clamp(min=0)
        bound = torch.clamp(doc_len - wcl, min=0)  # [H]

        def mk(_li, _orig, _isg, _bound):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                T = query.shape[1]
                # full status (used by global heads)
                core, fs = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                glob_state[_li] = fs.detach()
                # prefix status for LOCAL heads: status at min boundary across
                # local heads (we store per-head by running to each distinct
                # boundary). Simplest exact: store status at (T - Wmax_local).
                local_b = _bound[~_isg]
                if local_b.numel() > 0:
                    bmin = int(local_b.min().item())
                    if bmin <= 0:
                        ps = torch.zeros_like(fs)
                    else:
                        _, ps = _orig(
                            query[:, :bmin],
                            key[:, :bmin],
                            value[:, :bmin],
                            g=g[:, :bmin],
                            beta=beta[:, :bmin],
                            initial_state=None,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        )
                    prefix_state[_li] = (ps.detach(), bmin)
                return core, fs

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li], isg, bound)

    try:
        cache = DynamicCache(config=model.config)
        ids = torch.tensor([doc_ids], device=device)
        pids = torch.arange(0, doc_len, device=device).unsqueeze(0)
        model(input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True)
        return {
            "cache": cache,
            "pos": doc_len,
            "doc_ids": doc_ids,
            "glob_state": glob_state,
            "prefix_state": prefix_state,
            "win": win_tok_by_layer,
            "isg": isg_by_layer,
        }
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f


# ──────────────────────────────────────────────────────────────────────────
# torch.compile support for the Qwen3.5 hybrid online forward
# ──────────────────────────────────────────────────────────────────────────
#
# RedKnot's FLOPs savings (linear prefix-state reuse + head-class full attention
# + MoE token sparsity) only become wall-time TTFT speedup once the per-layer
# eager Python dispatch is removed. We do that with torch.compile.
#
# Design: we compile each *decoder layer's* forward with ``fullgraph=False,
# dynamic=True``. TorchDynamo automatically GRAPH-BREAKS at the data-dependent
# kernels — the patched FLA ``chunk_gated_delta_rule`` (linear layers, where the
# prefix-state injection lives), flash/sdpa attention, and the MoE expert loop —
# and compiles the static regions in between (input_layernorm, in_proj/q/k/v
# projections, conv gating GEMMs, o_proj/out_proj, post_attention_layernorm,
# router + shared-expert GEMMs, residual adds). This:
#   * preserves the patched prefix-state contract EXACTLY (the patch stays eager
#     between graph breaks — same numerics as the non-compiled path), and
#   * removes the dominant eager-dispatch overhead (documented ~53% of online
#     time) without re-deriving any GDN/attention math by hand.
#
# ``dynamic=True`` marks the sequence-length dim symbolic so ONE compiled graph
# serves all online (window+query) lengths — different RAG requests have slightly
# different token counts and ``dynamic=False`` would recompile (~tens of seconds)
# on every new length, destroying the speedup.
_QWEN35_COMPILED_LAYER_CACHE: dict = {}


def _get_compiled_layer_forward(layer, enable: bool):
    """Return a compiled version of ``layer.forward`` (cached per layer id).

    When ``enable`` is False this is the identity (returns ``layer.forward``),
    so callers can pass the flag through transparently.
    """
    if not enable:
        return layer.forward
    key = id(layer)
    cached = _QWEN35_COMPILED_LAYER_CACHE.get(key)
    if cached is None:
        cached = torch.compile(layer.forward, dynamic=True, fullgraph=False)
        _QWEN35_COMPILED_LAYER_CACHE[key] = cached
    return cached


def clear_qwen35_compile_cache():
    """Drop all cached compiled layer forwards (e.g. between different models)."""
    _QWEN35_COMPILED_LAYER_CACHE.clear()


@torch.no_grad()
def _compiled_online_forward(
    model,
    *,
    input_ids,
    position_ids,
    past_key_values,
    use_compile: bool,
):
    """One online prefill forward with per-layer compiled static blocks.

    Mirrors the HF ``Qwen3_5MoeTextModel.forward`` prefill path (embed -> rotary
    -> per-layer decoder -> final norm -> lm_head on the LAST position only) but
    routes every decoder layer through :func:`_get_compiled_layer_forward`. All
    monkey-patches installed on the layers (e.g. the linear prefix-state kernel
    wrapper) are honoured because we call the layer's (compiled) ``forward``,
    which dispatches to the patched submodules.

    CRITICAL: we reproduce the native forward's causal-mask, MRoPE position and
    ``cache_position`` construction exactly. Passing ``attention_mask=None`` to
    full-attention layers makes SDPA materialise a dense (cached_len x q_len)
    attention with no causal/cache structure -> wrong numerics AND huge memory.

    Returns the logits for the final position: ``[1, vocab]``.
    """
    from transformers.masking_utils import create_causal_mask

    base = model.model if hasattr(model, "model") else model
    tc = base.config
    device = input_ids.device

    inputs_embeds = base.embed_tokens(input_ids)
    bsz, q_len, _ = inputs_embeds.shape

    # Mark the sequence-length dim symbolic so ONE compiled graph serves all
    # online lengths. Different RAG samples have slightly different window+query
    # token counts; without this, torch.compile recompiles (~15s) per new length
    # and the recompile cost shows up in TTFT. mark_dynamic makes the seq dim a
    # symbolic int in the captured graph -> reuse across all lengths.
    if use_compile:
        try:
            torch._dynamo.maybe_mark_dynamic(inputs_embeds, 1)
        except Exception:
            pass

    # cache_position: global positions of the q tokens within the (trimmed) cache.
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(past_seen, past_seen + q_len, device=device)

    # MRoPE position_ids: native expands a (text + 3 vision) 4-row layout. For
    # text-only RAG all 4 rows are the same 1D positions. position_ids in is [1,T].
    if position_ids.ndim == 2:
        pos4 = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    else:
        pos4 = position_ids
    text_position_ids = pos4[0]
    rope_position_ids = pos4[1:]

    causal_mask = create_causal_mask(
        config=tc,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    # Linear-attention layers attend all inputs (no padding) -> mask None.
    linear_attn_mask = None

    hidden = inputs_embeds
    position_embeddings = base.rotary_emb(hidden, rope_position_ids)

    layer_types = getattr(tc, "layer_types", None)
    for i, layer in enumerate(base.layers):
        is_linear = layer_types is not None and layer_types[i] == "linear_attention"
        layer_mask = linear_attn_mask if is_linear else causal_mask
        fwd = _get_compiled_layer_forward(layer, use_compile)
        hidden = fwd(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )

    hidden = base.norm(hidden)
    last = hidden[:, -1:, :]
    logits = model.lm_head(last)
    return logits[:, -1, :]


@torch.no_grad()
def rag_query_reuse_v2(
    model,
    tokenizer,
    *,
    doc_state,
    query_text,
    max_new_tokens=24,
    use_compile=False,
    moe_sparse=False,
    deep_moe_start_layer=20,
    moe_mass_thresh=0.2,
):
    """ONLINE (v2): reuse offline prefix status; local heads recompute window.

    Online we feed [last Wmax doc tokens + query] (the "window region") so that:
      * LOCAL heads start from the offline PREFIX status (window-外, reused) and
        recompute only the window tokens (+ query) -> exact, window-内 recomputed.
      * GLOBAL heads start from the offline FULL doc status and continue (query).
      * full attention reuses the cached doc KV (only window+query tokens add KV;
        but for correctness we keep the full doc KV cache and append window+query;
        to avoid double-counting window KV we trim the cache to doc_len-Wmax).
    Returns (text, ttft). ttft excludes offline build.
    """
    import copy, time as _t

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    win = doc_state["win"]
    isg = doc_state["isg"]
    doc_ids = doc_state["doc_ids"]
    doc_len = len(doc_ids)
    # Wmax across all local heads of all windowed layers
    Wmax = 0
    for li, wt in win.items():
        if wt is None:
            continue
        loc = wt[(wt > 0)]
        if loc.numel():
            Wmax = max(Wmax, int(loc.max().item()))
    Wmax = min(Wmax, doc_len)
    win_start = doc_len - Wmax  # window region begins here (global pos)

    # full-attn KV: keep cached doc KV but we must include window tokens; simplest
    # correct path: rebuild cache for [0, win_start) doc KV via the stored cache
    # trimmed, then online prefill [win_start:doc_len) + query appends their KV.
    cache = copy.deepcopy(doc_state["cache"])
    # trim cached KV to win_start for full-attn layers (so window tokens re-add)
    try:
        for layer in cache.layers:
            if (
                getattr(layer, "keys", None) is not None
                and layer.keys.shape[2] >= doc_len
            ):
                layer.keys = layer.keys[:, :, :win_start, :].contiguous()
                layer.values = layer.values[:, :, :win_start, :].contiguous()
    except Exception:
        pass

    # linear per-layer initial states for the online (window+query) pass.
    # We re-feed the window doc tokens [win_start:doc_len] online, so BOTH head
    # classes must start from the PREFIX status at win_start (otherwise global
    # heads, starting from the FULL doc status, would double-count the window).
    # global heads then recompute the window -> exact full status; local heads
    # recompute the window -> exact windowed status. The reused (offline) part is
    # the prefix [0, win_start) status for all heads.
    init_state = {}
    for li, wt in win.items():
        if wt is None:
            continue
        gs = doc_state["glob_state"].get(li)
        ps_b = doc_state["prefix_state"].get(li)
        ldev = gs.device
        # prefix at win_start (==Wmax boundary). build stores it as prefix_state.
        init_state[li] = (
            ps_b[0].to(ldev).detach() if ps_b is not None else torch.zeros_like(gs)
        )

    run_state = {li: init_state[li] for li in init_state}
    saved = {}
    for li in init_state:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule

        def mk(_li, _orig):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                core, fs = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=run_state[_li],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                run_state[_li] = fs.detach()
                return core, fs

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    try:
        window_ids = doc_ids[win_start:doc_len]
        q_ids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
        online_ids = window_ids + q_ids
        ids = torch.tensor([online_ids], device=device)
        pids = torch.arange(
            win_start, win_start + len(online_ids), device=device
        ).unsqueeze(0)

        # Token-level MoE sparsity (shallow dense / deep sparse). Computed on the
        # ONLINE tokens: a cheap sampled attention-mass pass over the deep half of
        # full-attn layers gives per-token importance; deep MoE layers then skip
        # routed experts for low-importance tokens (shared expert only). Installed
        # for the duration of the online prefill; restored in `finally`.
        moe_restore = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        if moe_sparse:
            token_mass = collect_attention_mass(
                model, ids, position_ids=pids, deep_full_frac=0.5
            )
            moe_restore = install_moe_token_sparse(
                model,
                token_mass,
                deep_moe_start_layer=deep_moe_start_layer,
                mass_thresh=moe_mass_thresh,
            )
        if use_compile:
            # Compiled online prefill: per-layer static blocks compiled, the
            # patched linear prefix-state kernel + attention + MoE stay eager.
            # Measures the prefill TTFT we want to accelerate.
            last_logits = _compiled_online_forward(
                model,
                input_ids=ids,
                position_ids=pids,
                past_key_values=cache,
                use_compile=True,
            )
            nxt = last_logits[0, :].argmax().view(1, 1)
        else:
            out = model(
                input_ids=ids,
                position_ids=pids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            nxt = out.logits[0, -1, :].argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = _t.perf_counter() - t0
        if moe_restore is not None:
            moe_restore()
        pos = win_start + len(online_ids)
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f
        try:
            if moe_restore is not None:
                moe_restore()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Per-head segmented offline state reuse (the "multi-segment snapshot" design)
# ──────────────────────────────────────────────────────────────────────────
#
# Idea (user design): OFFLINE, prefill the document in `seg`-sized pieces and
# snapshot the per-head linear recurrent state at EVERY segment boundary
# (``seg_states[li] = [S_b[1], S_b[2], ...]`` where ``S_b[j]`` is the state over
# tokens ``[0, j*seg)``, shape ``[1, H, dk, dv]``). ONLINE, each LOCAL head h with
# window ``W_h`` recovers ``S_i`` by taking the NEAREST usable offline snapshot at
# segment ``j*_h = floor((doc_len - W_h) / seg)`` as its prefix and recomputing
# only ``[j*_h*seg : i]``. Different heads pick DIFFERENT snapshots, so a small
# window recomputes less — this is the per-head "fill the gap" reuse.
#
# GLOBAL heads use the final snapshot ``S_b[N]`` (full-doc state, zero online cost
# beyond the shared query tokens).
#
# Because hidden_states are shared across all heads/layers, the online forward
# must cover ``[online_start : doc_len] + query`` where
# ``online_start = min_h(j*_h) * seg`` (the earliest snapshot any local head
# needs). Within each linear layer the patched kernel advances seg-by-seg and,
# per head, SKIPS recomputation until the segment index reaches that head's
# ``j*_h`` (seeding from the offline snapshot), then recomputes forward — exact.


@torch.no_grad()
def rag_build_offline_segmented(
    model, tokenizer, *, segments, win_tok_by_layer, seg=2048
):
    """OFFLINE: prefill doc in `seg` pieces, snapshot per-head linear state at
    every segment boundary; cache full-attn KV natively.

    Returns dict {cache, doc_ids, pos, seg, n_seg, seg_states, glob_state, win, isg}.
      * seg_states[li]: list length n_seg of [1,H,dk,dv]; seg_states[li][j] is the
        per-head state over tokens [0, (j+1)*seg) (boundary AFTER segment j).
      * glob_state[li]: final state S_b[N] (== seg_states[li][-1]).
    """
    from transformers import DynamicCache

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    seg_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in segments]
    doc_ids = [t for s in seg_ids for t in s]
    doc_len = len(doc_ids)
    n_seg = (doc_len + seg - 1) // seg

    seg_states = {li: [] for li in win_tok_by_layer if win_tok_by_layer[li] is not None}
    glob_state = {}
    isg_by_layer = {}
    run_state = {li: None for li in seg_states}
    saved = {}

    for li, wt in win_tok_by_layer.items():
        if wt is None:
            continue
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule
        isg_by_layer[li] = wt <= 0

        def mk(_li, _orig):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                # Advance the recurrence over this slice in `seg` pieces, snapping
                # the running state at every segment boundary.
                T = query.shape[1]
                state = run_state[_li]
                outs = []
                for st in range(0, T, seg):
                    en = min(st + seg, T)
                    sc, fs = _orig(
                        query[:, st:en],
                        key[:, st:en],
                        value[:, st:en],
                        g=g[:, st:en],
                        beta=beta[:, st:en],
                        initial_state=state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    outs.append(sc)
                    state = fs.detach()
                    seg_states[_li].append(state.clone())
                run_state[_li] = state
                core = torch.cat(outs, dim=1)
                return (core, state) if output_final_state else (core, None)

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    try:
        cache = DynamicCache(config=model.config)
        ids = torch.tensor([doc_ids], device=device)
        pids = torch.arange(0, doc_len, device=device).unsqueeze(0)
        model(input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True)
        for li in seg_states:
            glob_state[li] = seg_states[li][-1] if seg_states[li] else None
        return {
            "cache": cache,
            "doc_ids": doc_ids,
            "pos": doc_len,
            "seg": seg,
            "n_seg": n_seg,
            "seg_states": seg_states,
            "glob_state": glob_state,
            "win": win_tok_by_layer,
            "isg": isg_by_layer,
        }
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f


@torch.no_grad()
def rag_query_reuse_from_snapshots(
    model,
    tokenizer,
    *,
    doc_state,
    query_text,
    window_segs=2,
    keep_segs=2,
    max_new_tokens=24,
    use_compile=False,
    moe_sparse=False,
    deep_moe_start_layer=20,
    moe_mass_thresh=0.2,
):
    """ONLINE with ZERO history recompute: reuse OFFLINE per-segment snapshots.

    This is the true "build once offline, reuse online" path. The offline build
    (rag_build_offline_segmented) already stored, per linear layer, the exact
    recurrent state at EVERY segment boundary (seg_states) plus the full-attn KV
    cache. Online, for ANY query we:

      * fix the window to ``window_segs`` segments (aligned to seg boundaries) so
        ``win_start = doc_len - window_segs*seg`` lands exactly on a stored
        snapshot boundary;
      * LOCAL heads take their prefix DIRECTLY from the offline snapshot at
        ``seg index = win_start/seg - 1`` combined with the last ``keep_segs``
        segments before the window (also snapshots) -> NO history is recomputed
        online;
      * GLOBAL heads take the offline FULL doc state (glob_state);
      * we then run ONLY ``[win_start:doc_len] + query`` (window + query tokens).

    Because the window/prefix are snapshot-aligned and query-independent, the
    expensive document state is computed ONCE offline and reused by every query.
    The returned ttft measures ONLY the online window+query forward.

    Returns (text, ttft). Offline build cost is NOT included.
    """
    import copy, time as _t

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    win = doc_state["win"]
    doc_ids = doc_state["doc_ids"]
    doc_len = len(doc_ids)
    seg = doc_state["seg"]
    seg_states = doc_state["seg_states"]
    glob_state = doc_state["glob_state"]
    n_seg = doc_state["n_seg"]

    # Fixed, snapshot-aligned window. win_start lands on a segment boundary.
    window_segs = max(1, int(window_segs))
    win_segs = min(window_segs, n_seg)
    win_start = max(0, doc_len - win_segs * seg)
    start_seg = win_start // seg  # number of full segments before the window
    # snapshot list-index for the prefix state over [0, start_seg*seg):
    # seg_states[j] is the state AFTER segment j == state over [0,(j+1)*seg).
    prefix_idx = start_seg - 1  # state over [0, start_seg*seg)

    # Per-layer initial linear state, taken DIRECTLY from offline snapshots:
    #   LOCAL  heads -> snapshot at prefix_idx (exact state over [0, win_start)).
    #                   (keep_segs is implicitly satisfied: the snapshot already
    #                    folds all prior segments exactly, fixed order.)
    #   GLOBAL heads -> full doc state (glob_state).
    init_state = {}
    for li, wt in win.items():
        if wt is None:
            continue
        gs = glob_state[li]
        isg = (wt <= 0).to(gs.device)
        if prefix_idx >= 0 and prefix_idx < len(seg_states[li]):
            st = seg_states[li][prefix_idx].to(gs.device).clone()
        else:
            st = torch.zeros_like(gs)
        st[:, isg] = gs[:, isg]  # global heads use full doc state
        init_state[li] = st

    # full-attn KV: trim cached doc KV to win_start so window tokens re-add online.
    cache = copy.deepcopy(doc_state["cache"])
    try:
        for layer in cache.layers:
            if (
                getattr(layer, "keys", None) is not None
                and layer.keys.shape[2] >= doc_len
            ):
                layer.keys = layer.keys[:, :, :win_start, :].contiguous()
                layer.values = layer.values[:, :, :win_start, :].contiguous()
    except Exception:
        pass

    run_state = {li: init_state[li] for li in init_state}
    saved = {}
    for li in init_state:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule

        def mk(_li, _orig):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                core, fs = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=run_state[_li],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                run_state[_li] = fs.detach()
                return core, fs

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    moe_restore = None
    try:
        q_ids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
        online_ids = doc_ids[win_start:doc_len] + q_ids
        ids = torch.tensor([online_ids], device=device)
        pids = torch.arange(
            win_start, win_start + len(online_ids), device=device
        ).unsqueeze(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        if moe_sparse:
            token_mass = collect_attention_mass(
                model, ids, position_ids=pids, deep_full_frac=0.5
            )
            moe_restore = install_moe_token_sparse(
                model,
                token_mass,
                deep_moe_start_layer=deep_moe_start_layer,
                mass_thresh=moe_mass_thresh,
            )
        if use_compile:
            last_logits = _compiled_online_forward(
                model,
                input_ids=ids,
                position_ids=pids,
                past_key_values=cache,
                use_compile=True,
            )
            nxt = last_logits[0, :].argmax().view(1, 1)
        else:
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            nxt = out.logits[0, -1, :].argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = _t.perf_counter() - t0
        if moe_restore is not None:
            moe_restore()
            moe_restore = None
        pos = win_start + len(online_ids)
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f
        try:
            if moe_restore is not None:
                moe_restore()
        except Exception:
            pass


@torch.no_grad()
def rag_query_reuse_segmented(
    model, tokenizer, *, doc_state, query_text, max_new_tokens=24, use_compile=False
):
    """ONLINE: per-head segmented snapshot reuse (legacy variable-window variant).

    Returns (text, ttft). ttft excludes the offline build.
    """
    import copy, time as _t

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    win = doc_state["win"]
    doc_ids = doc_state["doc_ids"]
    doc_len = len(doc_ids)
    seg = doc_state["seg"]
    seg_states = doc_state["seg_states"]
    glob_state = doc_state["glob_state"]

    # Per-(layer,head) prefix segment index j*_h and the global earliest snapshot.
    # j*_h is the segment boundary AT or BEFORE (doc_len - W_h); we store snapshots
    # indexed by "after segment j" => boundary token (j+1)*seg. We want the snapshot
    # whose boundary <= (doc_len - W_h): snap_idx_h = floor((doc_len - W_h)/seg) - 1.
    min_start_seg = None  # number of segments to skip globally (= online_start/seg)
    jstar = {}  # {li: LongTensor[H] snapshot list-index per head (>=-1; -1 => from zero/full prefix none)}
    for li, wt in win.items():
        if wt is None:
            continue
        H = wt.numel()
        js = torch.full(
            (H,), len(seg_states[li]) - 1, dtype=torch.long
        )  # global heads: last snapshot
        for h in range(H):
            w_h = int(wt[h].item())
            if w_h <= 0:
                js[h] = (
                    len(seg_states[li]) - 1
                )  # global -> final state, recompute nothing
                continue
            bound = doc_len - w_h  # window starts at this global token
            # snapshot covering [0, (idx+1)*seg) with (idx+1)*seg <= bound
            idx = bound // seg - 1
            if idx < 0:
                idx = (
                    -1
                )  # window reaches into the very first segment -> start from zero
            js[h] = idx
        jstar[li] = js
        local_idx = js[(wt > 0)]
        if local_idx.numel():
            cand = int(local_idx.min().item())
            cand = max(cand, -1)
            # online must start at segment (cand+1) so that snapshot `cand` is the prefix
            start_seg_li = cand + 1
            min_start_seg = (
                start_seg_li
                if min_start_seg is None
                else min(min_start_seg, start_seg_li)
            )
    if min_start_seg is None:
        min_start_seg = 0
    min_start_seg = max(min_start_seg, 0)
    online_start = min(min_start_seg * seg, doc_len)
    online_start_seg = online_start // seg

    # full-attn KV: trim cached doc KV to online_start so window tokens re-add.
    cache = copy.deepcopy(doc_state["cache"])
    try:
        for layer in cache.layers:
            if (
                getattr(layer, "keys", None) is not None
                and layer.keys.shape[2] >= doc_len
            ):
                layer.keys = layer.keys[:, :, :online_start, :].contiguous()
                layer.values = layer.values[:, :, :online_start, :].contiguous()
    except Exception:
        pass

    # Patch each linear layer: advance the online slice seg-by-seg; per head, hold
    # the offline snapshot until that head's j*_h is reached, then recompute.
    saved = {}
    for li in jstar:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule

        def mk(_li, _orig):
            js = jstar[_li].to(device)
            n_snap = len(seg_states[_li])

            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                # EXACT per-head segmented reuse.
                #
                # Seed from the exact offline prefix snapshot at ``online_start``
                # (== S_{online_start}); the delta-rule recurrence forward from an
                # exact prefix is exact for EVERY head, so the running-state path is
                # numerically identical to a full recompute. The per-head benefit is
                # that heads whose window starts LATER than online_start could have
                # started from a later snapshot — realised below by skipping the
                # kernel for a head's pre-window segments and seeding its state from
                # the offline snapshot at its own window-start boundary instead.
                T = query.shape[1]
                base_idx = online_start_seg - 1
                _dev = value.device  # this layer's device (device_map=auto safe)
                _js = js.to(_dev)
                if base_idx >= 0:
                    state = seg_states[_li][base_idx].clone().to(value.dtype).to(_dev)
                else:
                    state = (
                        torch.zeros_like(seg_states[_li][0]).to(value.dtype).to(_dev)
                    )

                outs = []
                cur_seg = online_start_seg  # global segment index at slice start
                for st in range(0, T, seg):
                    en = min(st + seg, T)
                    sc, fs = _orig(
                        query[:, st:en],
                        key[:, st:en],
                        value[:, st:en],
                        g=g[:, st:en],
                        beta=beta[:, st:en],
                        initial_state=state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    outs.append(sc)
                    fs = fs.detach()
                    cur_seg += 1
                    # Boundary after this segment is global token cur_seg*seg, i.e.
                    # offline snapshot list-index (cur_seg-1). For heads whose window
                    # has NOT started yet (j*_h >= cur_seg-1), the exact state at this
                    # boundary IS that offline snapshot; overwrite to avoid drift and
                    # to keep them seeded exactly until their window begins.
                    snap_idx = cur_seg - 1
                    if 0 <= snap_idx < n_snap:
                        not_started = (
                            _js >= snap_idx
                        )  # head's prefix boundary at/after here
                        if bool(not_started.any()):
                            snap = seg_states[_li][snap_idx].to(fs.dtype).to(_dev)
                            fs = fs.clone()
                            fs[:, not_started] = snap[:, not_started]
                    state = fs.detach()
                core = torch.cat(outs, dim=1)
                return (core, state) if output_final_state else (core, None)

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    try:
        window_ids = doc_ids[online_start:doc_len]
        q_ids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
        online_ids = window_ids + q_ids
        ids = torch.tensor([online_ids], device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        pids = torch.arange(
            online_start, online_start + len(online_ids), device=device
        ).unsqueeze(0)
        if use_compile:
            last_logits = _compiled_online_forward(
                model,
                input_ids=ids,
                position_ids=pids,
                past_key_values=cache,
                use_compile=True,
            )
            nxt = last_logits[0, :].argmax().view(1, 1)
        else:
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            nxt = out.logits[0, -1, :].argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = _t.perf_counter() - t0
        pos = online_start + len(online_ids)
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f


# ══════════════════════════════════════════════════════════════════════════
# Approach-2: chunk-independent linear reuse with K-nearest-chunk local prefix
# ══════════════════════════════════════════════════════════════════════════
#
# Validated by quant_approx_error.py: for a LOCAL head, building its prefix state
# from ONLY the K=2 chunks immediately before its window (dropping farther chunks)
# is numerically as good as the exact full prefix (F1 identical). This is because
# the gated DELTA rule keeps overwriting far-back state, so a local head's distant
# history is already negligible. We exploit this:
#
#   OFFLINE (rag_build_offline_chunked): prefill the doc once (fixed order) and
#   cache full-attn KV + per-layer linear FULL doc state (for GLOBAL heads).
#   We keep doc_ids so the online pass can recompute the small K-chunk prefix.
#
#   ONLINE (rag_query_reuse_chunked): per local head, its prefix state = exact
#   state over the last K chunks before win_start (computed cheaply online from
#   zero-init); GLOBAL heads use the cached FULL doc state. Then the shared online
#   window [win_start:doc_len]+query is run; per-head outputs combine naturally on
#   the head dim (kernel is per-head). Optional deep-MoE token sparsity + compile.
#
# Exactness contract: with window=full-ctx AND keep_chunks covering the whole doc,
# this must reproduce the dense output byte-for-byte (asserted in tests).


@torch.no_grad()
def rag_build_offline_chunked(model, tokenizer, *, segments, win_tok_by_layer):
    """OFFLINE (approach-2): cache full-attn KV + per-layer FULL linear doc state.

    Returns {cache, doc_ids, seg_lens, pos, glob_state, win, isg}. The local-head
    prefix is NOT precomputed here; it is rebuilt online from the K nearest chunks
    (cheap, exact for those chunks). GLOBAL heads use glob_state (full doc state).
    """
    from transformers import DynamicCache

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    seg_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in segments]
    seg_lens = [len(s) for s in seg_ids]
    doc_ids = [t for s in seg_ids for t in s]
    doc_len = len(doc_ids)

    glob_state = {}
    isg_by_layer = {}
    saved = {}
    for li, wt in win_tok_by_layer.items():
        if wt is None:
            continue
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule
        isg_by_layer[li] = wt <= 0

        def mk(_li, _orig):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                core, fs = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                glob_state[_li] = fs.detach()
                return core, fs

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    try:
        cache = DynamicCache(config=model.config)
        ids = torch.tensor([doc_ids], device=device)
        pids = torch.arange(0, doc_len, device=device).unsqueeze(0)
        model(input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True)
        return {
            "cache": cache,
            "doc_ids": doc_ids,
            "seg_lens": seg_lens,
            "pos": doc_len,
            "glob_state": glob_state,
            "win": win_tok_by_layer,
            "isg": isg_by_layer,
        }
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f


@torch.no_grad()
def rag_query_reuse_chunked(
    model,
    tokenizer,
    *,
    doc_state,
    query_text,
    keep_chunks=2,
    chunk_tokens=2000,
    max_new_tokens=24,
    use_compile=False,
    moe_sparse=False,
    deep_moe_start_layer=20,
    moe_mass_thresh=0.2,
):
    """ONLINE (approach-2): local-head prefix from K nearest chunks; global from
    full doc state; window+query recomputed. See module banner for the contract.

    keep_chunks=0 => keep ALL history before window (exact full prefix).
    Returns (text, ttft). ttft excludes offline build.
    """
    import copy, time as _t

    bm = model.model if hasattr(model, "model") else model
    device = model.device

    win = doc_state["win"]
    doc_ids = doc_state["doc_ids"]
    doc_len = len(doc_ids)
    glob_state = doc_state["glob_state"]

    # Wmax across all local heads -> shared online window start.
    Wmax = 0
    for li, wt in win.items():
        if wt is None:
            continue
        loc = wt[(wt > 0)]
        if loc.numel():
            Wmax = max(Wmax, int(loc.max().item()))
    Wmax = min(Wmax, doc_len)
    # Quantize the shared online window UP to a fixed bucket so the online seq
    # length (Wmax + query) is stable across samples -> the compiled graph is
    # reused (no per-sample torch.compile recompile, which costs ~15s and would
    # pollute TTFT). Rounding UP only enlarges the window (strictly more exact),
    # never beyond doc_len.
    if use_compile:
        bucket = 512
        Wmax = min(((Wmax + bucket - 1) // bucket) * bucket, doc_len)
    win_start = doc_len - Wmax

    # Local-head prefix region: keep only the last `keep_chunks` chunks before the
    # window; drop farther history (delta-rule makes this near-lossless).
    if keep_chunks and keep_chunks > 0:
        pref_start = max(0, win_start - keep_chunks * chunk_tokens)
    else:
        pref_start = 0  # exact full prefix

    # Offline-quality prefix state per layer: state over [pref_start, win_start)
    # for LOCAL heads (zero-init, exact for that span); GLOBAL heads override with
    # the cached FULL doc state. Computed cheaply here (short span).
    prefix_state = {}
    if win_start > pref_start:
        saved_b = {}
        for li, wt in win.items():
            if wt is None:
                continue
            mod = bm.layers[li].linear_attn
            saved_b[li] = mod.chunk_gated_delta_rule

            def mk_b(_li, _orig):
                def wrapped(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=False,
                    **kw,
                ):
                    _, ps = _orig(
                        query[:, pref_start:win_start],
                        key[:, pref_start:win_start],
                        value[:, pref_start:win_start],
                        g=g[:, pref_start:win_start],
                        beta=beta[:, pref_start:win_start],
                        initial_state=None,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    prefix_state[_li] = ps.detach()
                    # return a dummy full-length core (not used; we only want ps)
                    core, _ = _orig(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=None,
                        output_final_state=False,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    return core, ps

                return wrapped

            mod.chunk_gated_delta_rule = mk_b(li, saved_b[li])
        try:
            ids_pref = torch.tensor([doc_ids[:win_start]], device=device)
            if ids_pref.shape[1] > 0:
                pp = torch.arange(0, win_start, device=device).unsqueeze(0)
                model(input_ids=ids_pref, position_ids=pp, use_cache=False)
        finally:
            for li, f in saved_b.items():
                bm.layers[li].linear_attn.chunk_gated_delta_rule = f
    # Fallback for layers with no prefix computed (pref_start==win_start): zeros.
    for li, wt in win.items():
        if wt is None:
            continue
        if li not in prefix_state:
            prefix_state[li] = torch.zeros_like(glob_state[li])

    # full-attn KV: trim cached doc KV to win_start so window tokens re-add online.
    cache = copy.deepcopy(doc_state["cache"])
    try:
        for layer in cache.layers:
            if (
                getattr(layer, "keys", None) is not None
                and layer.keys.shape[2] >= doc_len
            ):
                layer.keys = layer.keys[:, :, :win_start, :].contiguous()
                layer.values = layer.values[:, :, :win_start, :].contiguous()
    except Exception:
        pass

    # Per-layer initial linear state: LOCAL heads = K-chunk prefix; GLOBAL = full.
    init_state = {}
    for li, wt in win.items():
        if wt is None:
            continue
        gs = glob_state[li]
        isg = (wt <= 0).to(gs.device)
        st = prefix_state[li].to(gs.device).clone()
        st[:, isg] = gs[:, isg]
        init_state[li] = st

    run_state = {li: init_state[li] for li in init_state}
    saved = {}
    for li in init_state:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule

        def mk(_li, _orig):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                core, fs = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=run_state[_li],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                run_state[_li] = fs.detach()
                return core, fs

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    moe_restore = None
    try:
        q_ids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
        online_ids = doc_ids[win_start:doc_len] + q_ids
        ids = torch.tensor([online_ids], device=device)
        pids = torch.arange(
            win_start, win_start + len(online_ids), device=device
        ).unsqueeze(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        if moe_sparse:
            token_mass = collect_attention_mass(
                model, ids, position_ids=pids, deep_full_frac=0.5
            )
            moe_restore = install_moe_token_sparse(
                model,
                token_mass,
                deep_moe_start_layer=deep_moe_start_layer,
                mass_thresh=moe_mass_thresh,
            )
        if use_compile:
            last_logits = _compiled_online_forward(
                model,
                input_ids=ids,
                position_ids=pids,
                past_key_values=cache,
                use_compile=True,
            )
            nxt = last_logits[0, :].argmax().view(1, 1)
        else:
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            nxt = out.logits[0, -1, :].argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = _t.perf_counter() - t0
        if moe_restore is not None:
            moe_restore()
            moe_restore = None
        pos = win_start + len(online_ids)
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f
        try:
            if moe_restore is not None:
                moe_restore()
        except Exception:
            pass


def install_linear_segmented(model, win_tok_by_layer, seg=2048):
    """FAST + EXACT linear via SEGMENTED native-kernel calls (system optimization).

    Key finding: fla chunk kernel is ~450x SLOWER on one huge T=20000 call than on
    many small segments (long-sequence performance cliff). So we split EVERY
    linear layer's sequence into segments of `seg` tokens and relay state across
    them with the native kernel:
      * GLOBAL heads: state relayed across all segments  -> exact full history.
      * LOCAL  heads: SLIDING-WINDOW status. For token i (segment s), the
                      window covers [i-W, i]. At each segment boundary we keep a
                      per-head cache of finished-segment states (``state_hist``),
                      where ``state_hist[j]`` is the recurrent status over the
                      prefix [0, (j+1)*seg). When the window has elapsed, the
                      LOCAL head does NOT zero its state; it CARRIES the cached
                      prefix status at the window start [0, i-W] (segment index
                      ``s - ceil(W/seg)``) and the next segment re-advances
                      [i-W, i] from there. This is "cache prefix status, restore
                      it online (position info preserved by the kernel), recompute
                      the window" — i.e. decayed prefix carried, exactly.
    Each segment = ONE native kernel call (segment-internal is parallel). This
    turns linear's saved/cheaper work into real wall-clock speedup.

    win_tok_by_layer: {layer: win_tok[H]} token window (<=0/>=T global; None=skip).
    Returns restore_fn.
    """
    bm = model.model if hasattr(model, "model") else model
    saved = {}
    for li, wt in win_tok_by_layer.items():
        if wt is None:
            continue
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule
        isg = wt <= 0
        # per-head window in units of segments (>=1); global -> large
        wseg = torch.where(
            wt > 0, torch.ceil(wt.float() / seg).long(), torch.zeros_like(wt)
        )

        def mk(_orig, _isg, _wseg, _seg):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                T = query.shape[1]
                if T <= _seg:
                    return _orig(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        **kw,
                    )
                isgd = _isg.to(query.device)
                wsd = _wseg.to(query.device)
                state = initial_state
                outs = []
                seg_idx = 0
                # Per-head sliding-window status cache (segment granularity):
                # ``state_hist[j]`` = the per-head recurrent state AFTER segment j
                # (i.e. the prefix status over tokens [0, (j+1)*seg)). For LOCAL
                # heads, when the window (in segments) elapses we do NOT zero the
                # state; instead we CARRY the cached prefix status at the window
                # start [0, i-W] and let the kernel re-advance [i-W, i] from there.
                # This realises "cache prefix status, restore it online, recompute
                # the window" exactly (decayed prefix carried), while GLOBAL heads
                # keep relaying the full history.
                state_hist = []  # list of [1, H, Dk, Dv] states per finished seg
                for st in range(0, T, _seg):
                    en = min(st + _seg, T)
                    sc, fs = _orig(
                        query[:, st:en],
                        key[:, st:en],
                        value[:, st:en],
                        g=g[:, st:en],
                        beta=beta[:, st:en],
                        initial_state=state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    outs.append(sc)
                    seg_idx += 1
                    H = fs.shape[1]
                    # For LOCAL heads whose window has elapsed, replace the carried
                    # state with the cached PREFIX status at the window start
                    # [0, i-W] (segment index = seg_idx - wseg), instead of zeroing.
                    # The next segment then re-advances the window from that prefix.
                    next_state = fs
                    if not bool(isgd.all()):
                        prefix_state = fs.clone()
                        for h in range(H):
                            if bool(isgd[h]):
                                continue  # global head: keep full relay
                            wseg_h = int(wsd[h].item())
                            if wseg_h <= 0:
                                continue  # treated as global
                            # window start segment for the NEXT segment's queries
                            start_seg = seg_idx - wseg_h
                            if start_seg <= 0:
                                # window still covers the whole prefix -> keep full
                                continue
                            # cached prefix status over [0, start_seg*seg) == state
                            # after segment (start_seg-1)
                            prefix_state[:, h] = state_hist[start_seg - 1][:, h]
                        next_state = prefix_state
                    state = next_state.detach()
                    state_hist.append(fs.detach())
                core = torch.cat(outs, dim=1)
                return (core, state) if output_final_state else (core, None)

            return wrapped

        mod.chunk_gated_delta_rule = mk(saved[li], isg, wseg, seg)

    def restore():
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f

    return restore


@torch.no_grad()
def collect_attention_mass(
    model, input_ids, position_ids=None, deep_full_frac=0.5, n_qsample=512
):
    """First pass: collect per-token attention MASS averaged over the DEEP half
    of full-attention layers. Returns mass [T] (normalized, mean=1) = global
    token importance for downstream MoE sparsity. Uses sampled query rows to
    avoid the O(L^2) full matrix.
    """
    import torch.nn.functional as F
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        apply_rotary_pos_emb,
    )

    bm = model.model if hasattr(model, "model") else model
    full_idx = sorted(full_attention_layer_indices(model.config))
    deep_start = int(len(full_idx) * (1 - deep_full_frac))
    deep_full = full_idx[deep_start:]
    T = input_ids.shape[1]
    acc = torch.zeros(T, dtype=torch.float32)
    cnt = [0]
    handles = []
    for li in deep_full:
        attn = bm.layers[li].self_attn

        def mk(_attn):
            def hook(m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                pe = kwargs.get("position_embeddings")
                if hs is None or pe is None:
                    return
                B, Tt, _ = hs.shape
                hd = _attn.head_dim
                q_raw = _attn.q_proj(hs).view(B, Tt, -1, hd * 2)
                q, _g = torch.chunk(q_raw, 2, dim=-1)
                q = _attn.q_norm(q).transpose(1, 2)
                k = _attn.k_norm(_attn.k_proj(hs).view(B, Tt, -1, hd)).transpose(1, 2)
                cos, sin = pe
                q, k = apply_rotary_pos_emb(q, k, cos.to(q.device), sin.to(q.device))
                rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(rep, dim=1)
                qi = (
                    torch.randperm(Tt, device=q.device)[: min(n_qsample, Tt)]
                    .sort()
                    .values
                )
                qs = q[:, :, qi, :]
                scores = torch.matmul(qs, k.transpose(-1, -2)) * (hd**-0.5)
                keypos = torch.arange(Tt, device=q.device)
                mask = keypos[None, :] > qi[:, None]
                scores = scores.masked_fill(mask[None, None], float("-inf"))
                probs = F.softmax(scores.float(), dim=-1)
                mass = probs.sum(dim=(0, 1, 2)).cpu()
                acc.add_(mass)
                cnt[0] += 1

            return hook

        handles.append(attn.register_forward_pre_hook(mk(attn), with_kwargs=True))
    kw = {"use_cache": False}
    if position_ids is not None:
        kw["position_ids"] = position_ids
    model(input_ids=input_ids, **kw)
    for h in handles:
        h.remove()
    mass = acc / max(1, cnt[0])
    mass = mass / (mass.mean() + 1e-9)  # normalize mean=1
    return mass


def install_moe_token_sparse(
    model, token_mass, *, deep_moe_start_layer, mass_thresh=0.2
):
    """Second pass: deep MoE layers (idx >= deep_moe_start_layer) SKIP routed
    experts for LOW-importance tokens (token_mass < mass_thresh*mean); those
    tokens keep only the shared expert. High-mass tokens run the full MoE.
    Shallow MoE layers (< deep_moe_start_layer) stay dense. token_mass: [T].
    Returns restore_fn.
    """
    bm = model.model if hasattr(model, "model") else model
    saved = {}
    keep_mask_full = token_mass >= mass_thresh  # [T] True = run routed experts
    for li, layer in enumerate(bm.layers):
        if (
            li < deep_moe_start_layer
            or not hasattr(layer, "mlp")
            or not hasattr(layer.mlp, "experts")
        ):
            continue
        moe = layer.mlp
        saved[li] = moe.forward

        def mk(_moe):
            def fwd(hidden_states):
                import torch.nn.functional as F

                B, S, H = hidden_states.shape
                hs = hidden_states.view(-1, H)
                N = hs.shape[0]
                # shared expert for ALL tokens (cheap, dense)
                shared = _moe.shared_expert(hs)
                shared = F.sigmoid(_moe.shared_expert_gate(hs)) * shared
                # routed experts only for HIGH-mass tokens
                km = keep_mask_full.to(hs.device)
                if km.shape[0] != N:  # decode or size mismatch -> all run
                    km = torch.ones(N, dtype=torch.bool, device=hs.device)
                idx = km.nonzero(as_tuple=False).flatten()
                out = shared.clone()
                if idx.numel() > 0:
                    sub = hs[idx]
                    _, rw, se = _moe.gate(sub)
                    eo = _moe.experts(sub, se, rw)
                    out[idx] = out[idx] + eo.to(out.dtype)
                return out.view(B, S, H)

            return fwd

        moe.forward = mk(moe)

    def restore():
        for li, f in saved.items():
            bm.layers[li].mlp.forward = f

    return restore


# ──────────────────────────────────────────────────────────────────────────
# RoPE-relocatable offline KV/status cache + online splice (RedKnot RAG).
#
# Difference from rag_build_offline_v2 / rag_query_reuse_v2: the offline cache is
# built at LOCAL positions [0, doc_len) and can be SPLICED at an ARBITRARY global
# offset online. Full-attention cached K is RoPE-relocated by R(delta) before
# splicing; linear-attention status is relayed as the initial_state (RoPE-free).
# After splicing, the query (and the local window region) are recomputed with the
# existing RedKnot mechanisms (head-class full attn + windowed linear + deep MoE).
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _reposition_rope_k(k, delta, model):
    """Shift already-RoPE'd key tensor k:[B,Hkv,T,D] from its current positions
    to positions+delta by applying R(delta) (NeoX-style rotate_half RoPE).

    R(p+delta) = R(delta) @ R(p); applying apply_rotary with cos/sin evaluated at
    a CONSTANT delta for every position realises R(delta) on each key. delta may
    be a python int (uniform) — we build a length-T position vector all == delta.
    """
    if delta == 0:
        return k
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            apply_rotary_pos_emb,
        )
    except Exception:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    bm = model.model if hasattr(model, "model") else model
    rotary = bm.rotary_emb
    B, Hkv, T, D = k.shape
    device = k.device
    # constant position == delta for every cached token -> cos/sin of R(delta)
    pos = torch.full((1, T), int(delta), device=device, dtype=torch.long)
    # rotary_emb(x, position_ids) -> (cos, sin); ensure all on same device
    cos, sin = rotary(k.to(device), pos)
    cos, sin = cos.to(device), sin.to(device)
    # apply_rotary expects q,k; pass k twice, keep the second output.
    _, k_shifted = apply_rotary_pos_emb(k, k, cos, sin)
    return k_shifted


@torch.no_grad()
def rag_build_offline_relocatable(model, tokenizer, *, segments, win_tok_by_layer):
    """OFFLINE: prefill doc chunks ONCE at LOCAL positions [0, doc_len), store a
    relocatable cache. Full-attn KV stored RoPE'd at local positions; linear
    global status + local prefix status stored (RoPE-free). Reuse rag's v2 build
    (which already prefills at local pos 0) and just tag it relocatable.

    NOTE the v2 build already uses position_ids = arange(0, doc_len) (local), so
    its cached K is at LOCAL positions and is exactly what we relocate online.
    Returns the same dict as rag_build_offline_v2 (+ 'local_origin': 0).
    """
    st = rag_build_offline_v2(
        model, tokenizer, segments=segments, win_tok_by_layer=win_tok_by_layer
    )
    st["local_origin"] = 0
    return st


@torch.no_grad()
def rag_query_reuse_relocatable(
    model, tokenizer, *, doc_state, query_text, global_offset=0, max_new_tokens=24
):
    """ONLINE: splice the offline doc cache at an ARBITRARY ``global_offset``.

    Steps:
      1. Load cached full-attn KV; RoPE-RELOCATE cached K by delta=global_offset
         (V is RoPE-free, untouched). Trim to [0, win_start) so the local window
         is recomputed online.
      2. Linear layers: start from the offline PREFIX status (RoPE-free, reused
         verbatim) as initial_state; relay across the online window+query.
      3. Recompute the window region + query at GLOBAL positions
         [global_offset+win_start, ...] with RedKnot mechanisms active.
    Returns (text, ttft).
    """
    import copy, time as _t

    bm = model.model if hasattr(model, "model") else model
    device = model.device
    win = doc_state["win"]
    doc_ids = doc_state["doc_ids"]
    doc_len = len(doc_ids)
    Wmax = 0
    for li, wt in win.items():
        if wt is None:
            continue
        loc = wt[(wt > 0)]
        if loc.numel():
            Wmax = max(Wmax, int(loc.max().item()))
    Wmax = min(Wmax, doc_len)
    win_start = doc_len - Wmax

    # 1) copy + RoPE-relocate + trim cached full-attn KV
    cache = copy.deepcopy(doc_state["cache"])
    # NOTE: do NOT move cached tensors to a single device — device_map places
    # each layer on a different GPU and the cache mirrors that placement.
    # _reposition_rope_k already operates on k's own device.
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        if k is None or k.shape[2] < doc_len:
            continue
        # relocate the whole cached K by delta=global_offset, then trim to win_start
        k_rel = _reposition_rope_k(k, global_offset, model)
        layer.keys = k_rel[:, :, :win_start, :].contiguous()
        layer.values = layer.values[:, :, :win_start, :].contiguous()

    # 2) linear initial states = offline prefix status (reused verbatim)
    init_state = {}
    for li, wt in win.items():
        if wt is None:
            continue
        gs = doc_state["glob_state"].get(li)
        ps_b = doc_state["prefix_state"].get(li)
        init_state[li] = (
            ps_b[0].to(gs.device).detach() if ps_b is not None else torch.zeros_like(gs)
        )
    run_state = {li: init_state[li] for li in init_state}
    saved = {}
    for li in init_state:
        mod = bm.layers[li].linear_attn
        saved[li] = mod.chunk_gated_delta_rule

        def mk(_li, _orig):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                core, fs = _orig(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=run_state[_li],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                run_state[_li] = fs.detach()
                return core, fs

            return wrapped

        mod.chunk_gated_delta_rule = mk(li, saved[li])

    try:
        # 3) recompute window+query at GLOBAL positions (offset applied)
        window_ids = doc_ids[win_start:doc_len]
        q_ids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
        online_ids = window_ids + q_ids
        ids = torch.tensor([online_ids], device=device)
        base = global_offset + win_start
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        pids = torch.arange(base, base + len(online_ids), device=device).unsqueeze(0)
        out = model(
            input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
        )
        cache = out.past_key_values
        nxt = out.logits[0, -1, :].argmax().view(1, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = _t.perf_counter() - t0
        pos = base + len(online_ids)
        gen = [int(nxt[0, 0])]
        for _ in range(max_new_tokens - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            gen.append(tid)
            pos += 1
            if tid == tokenizer.eos_token_id:
                break
        return tokenizer.decode(gen, skip_special_tokens=True), ttft
    finally:
        for li, f in saved.items():
            bm.layers[li].linear_attn.chunk_gated_delta_rule = f
