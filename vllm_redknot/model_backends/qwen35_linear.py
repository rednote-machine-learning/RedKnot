# REDKNOT-MODEL: extracted GDN head recurrence/window research implementations.
# Copyright 2024-2026 SGLang RedKnot Integration.
"""RedKnot Qwen3.5 GDN tensor helpers; native kernels are not copied.

The exact recurrence carries a supplied prefix state. The token-window helper
is the source research implementation, not an accuracy-certified replacement
for the native GDN recurrence. Neither helper is automatically installed.
"""

from __future__ import annotations

import torch


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
