#!/usr/bin/env python3
"""Verify the CORRECT linear local scheme: window start carries the DECAYED
PREFIX STATE (cross-chunk relay), not zero.

User's insight: a local head's window [t-W, t] must START from the prefix state
decayed to the window start, NOT from zero. Dropping the prefix (S0=0) starves
window-start tokens -> the bug behind all prior crashes.

Claim to test: for a head, output computed as
    S_win_start = S_prefix   (the FULL state at window start, carried over)
    then recompute the window normally
... is EXACT vs full history (trivially, since it's just continuing the
recurrence). The SAVING comes when S_prefix is REUSED across chunks (computed
once offline) instead of recomputed. So the real question:

  Does "decayed prefix state + window recompute" equal full history? YES by
  construction (it IS the full recurrence). The lossy part is only if we DROP
  the prefix. So we compare:
    REF        : full recurrence (S0=0 from token 0).
    PREFIX_RELAY: split at C; run [0,C) -> S_C; run [C,T) from S_C.  (== REF)
    ZERO_WINDOW : run [C,T) from ZERO (what the buggy impl did).      (!= REF)

Confirms that carrying the prefix state is what makes it lossless, and that the
saving is "compute prefix once, reuse S_C".
"""

from __future__ import annotations

import torch


def _l2(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def run(q, k, v, g, beta, lo, hi, S0):
    dk, dv = k.shape[-1], v.shape[-1]
    scale = 1.0 / (dk**0.5)
    S = S0.clone()
    outs = []
    for i in range(lo, hi):
        S = S * g[i].exp()
        kv = (S * k[i].unsqueeze(-1)).sum(dim=0)
        delta = (v[i] - kv) * beta[i]
        S = S + k[i].unsqueeze(-1) * delta.unsqueeze(0)
        outs.append((S * (q[i] * scale).unsqueeze(-1)).sum(dim=0))
    return torch.stack(outs) if outs else torch.empty(0, dv), S


def main():
    torch.manual_seed(0)
    import math

    dk = dv = 32
    T = 600
    C = 400  # prefix length (offline chunk); window region = [C, T)
    q = _l2(torch.randn(T, dk))
    k = _l2(torch.randn(T, dk))
    v = torch.randn(T, dv)
    beta = torch.rand(T)
    g = (math.log(0.97) + 0.05 * torch.randn(T)).clamp(max=-1e-4)

    # REF full
    ref, _ = run(q, k, v, g, beta, 0, T, torch.zeros(dk, dv))

    # PREFIX RELAY: prefix [0,C) -> S_C (computed once, reusable); then [C,T) from S_C
    _, S_C = run(q, k, v, g, beta, 0, C, torch.zeros(dk, dv))
    relay_out, _ = run(q, k, v, g, beta, C, T, S_C)

    # ZERO WINDOW (buggy): [C,T) from zero
    zero_out, _ = run(q, k, v, g, beta, C, T, torch.zeros(dk, dv))

    ref_tail = ref[C:]

    def rel(a):
        return (a - ref_tail).abs().max().item() / (ref_tail.abs().max().item() + 1e-6)

    print("=" * 66)
    print(f" Decayed-prefix-state relay (T={T}, prefix C={C})")
    print(
        f"   PREFIX_RELAY (carry S_C) vs full : rel_err={rel(relay_out):.6f}  {'EXACT' if rel(relay_out) < 1e-4 else 'DIFF'}"
    )
    print(
        f"   ZERO_WINDOW  (drop prefix) vs full: rel_err={rel(zero_out):.6f}  {'(the bug)' if rel(zero_out) > 1e-2 else ''}"
    )
    print("=" * 66)
    print(" => Carrying the decayed prefix state is EXACT. Saving = compute S_C")
    print("    once (offline) and reuse it; window region recompute is cheap.")
    print("=" * 66)


if __name__ == "__main__":
    main()
