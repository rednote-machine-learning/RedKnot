#!/usr/bin/env python3
"""Verify a CORRECT sliding-window delta-rule (no sink) vs the broken block-reset.

linear attention has NO attention sink (that's a softmax concept). A local head
should see a true SLIDING window: position t uses tokens [t-W+1, t]. The earlier
"block reset" (zero state every W steps) is WRONG: tokens just after a reset see
almost no history.

Correct & efficient realization = OVERLAPPED CHUNKS: process in blocks of size
W; for each block, WARM UP the state over the previous W tokens (then discard
warmup outputs), so every output token has >= W tokens of true history. This
equals the per-position sliding window.

Tests:
  REF_SLIDE : per-position exact sliding window (out[t] from [t-W+1, t]). O(T*W).
  OVERLAP   : overlapped-chunk implementation. Should EQUAL REF_SLIDE.
  BLOCK     : block-reset (the broken one). Should DIFFER near block starts.
"""

from __future__ import annotations

import torch


def _l2(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _step(S, k_t, v_t, g_t, beta_t):
    S = S * g_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=0)
    delta = (v_t - kv_mem) * beta_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(0)
    return S


def out_at(S, q_t):
    return (S * q_t.unsqueeze(-1)).sum(dim=0)


def run_segment(q, k, v, g, beta, lo, hi, S0=None):
    """Run delta-rule over tokens [lo, hi); return outputs for those positions
    and final state. q,k l2-normed, q scaled."""
    dk = k.shape[-1]
    dv = v.shape[-1]
    scale = 1.0 / (dk**0.5)
    S = torch.zeros(dk, dv) if S0 is None else S0.clone()
    outs = []
    for i in range(lo, hi):
        S = _step(S, k[i], v[i], g[i].exp(), beta[i])
        outs.append(out_at(S, q[i] * scale))
    return torch.stack(outs) if outs else torch.empty(0, dv), S


def ref_sliding(q, k, v, g, beta, W):
    """Exact per-position sliding window: out[t] from tokens [max(0,t-W+1), t]."""
    T, dv = v.shape[0], v.shape[-1]
    out = torch.zeros(T, dv)
    for t in range(T):
        lo = max(0, t - W + 1)
        o, _ = run_segment(q, k, v, g, beta, lo, t + 1)
        out[t] = o[-1]
    return out


def overlap_chunks(q, k, v, g, beta, W, warm_mult=1):
    """Overlapped-chunk sliding window: block size B, warm up over previous
    warm_mult*W tokens. For a block ending at bend-1, the last token needs
    [bend-W, bend-1]; with block size B and warmup warm_mult*W from bstart, the
    block-start token needs [bstart-W+1, bstart] -> warmup>=W. But the running
    state in run_segment is FULL-history within the segment, so the block-start
    token actually sees warmup tokens of history (>=W) which is correct, while
    the block-END token sees warmup+B history. To make EVERY token see exactly
    a sliding window we use block size B=W and warmup=W (each token sees between
    W and 2W history; since mem_len<<W the extra is harmless). Larger warm_mult
    reduces residual from block-start tokens with longer memory."""
    B = W
    T, dv = v.shape[0], v.shape[-1]
    out = torch.zeros(T, dv)
    for bstart in range(0, T, B):
        bend = min(bstart + B, T)
        warm = max(0, bstart - warm_mult * W)
        o, _ = run_segment(q, k, v, g, beta, warm, bend)
        out[bstart:bend] = o[(bstart - warm) : (bend - warm)]
    return out


def dual_state(q, k, v, g, beta, W):
    """Dual-state streaming sliding window (matches driver implementation)."""
    T, dk, dv = v.shape[0], k.shape[-1], v.shape[-1]
    scale = 1.0 / (dk**0.5)
    out = torch.zeros(T, dv)
    S = torch.zeros(dk, dv)
    Sh = torch.zeros(dk, dv)
    half = max(1, W // 2)
    for t in range(T):
        if t > 0 and t % half == 0:
            S = Sh.clone()
            Sh = torch.zeros(dk, dv)
        gt = g[t].exp()
        S = _step(S, k[t], v[t], gt, beta[t])
        Sh = _step(Sh, k[t], v[t], gt, beta[t])
        out[t] = out_at(S, q[t] * scale)
    return out


def block_reset(q, k, v, g, beta, W):
    """Broken: zero state every W steps."""
    T, dk, dv = v.shape[0], k.shape[-1], v.shape[-1]
    scale = 1.0 / (dk**0.5)
    out = torch.zeros(T, dv)
    S = torch.zeros(dk, dv)
    for t in range(T):
        if t % W == 0 and t > 0:
            S = torch.zeros(dk, dv)
        S = _step(S, k[t], v[t], g[t].exp(), beta[t])
        out[t] = out_at(S, q[t] * scale)
    return out


def main():
    torch.manual_seed(0)
    import math

    dk = dv = 32
    T = 600
    W = 128
    q = _l2(torch.randn(T, dk))
    k = _l2(torch.randn(T, dk))
    v = torch.randn(T, dv)
    beta = torch.rand(T)
    g = (math.log(0.97) + 0.05 * torch.randn(T)).clamp(max=-1e-4)  # mem_len~33

    ref = ref_sliding(q, k, v, g, beta, W)
    bl = block_reset(q, k, v, g, beta, W)

    def rel(a):
        return (a - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)

    print("=" * 64)
    print(f" Sliding-window delta-rule (T={T}, W={W}, mem_len~33, no sink)")
    ds = dual_state(q, k, v, g, beta, W)
    print(
        f"   DUAL-STATE vs exact sliding: rel_err={rel(ds):.6f}  {'PASS' if rel(ds) < 1e-3 else 'CHECK'}"
    )
    ov = overlap_chunks(q, k, v, g, beta, W, warm_mult=1)
    print(f"   OVERLAP    vs exact sliding: rel_err={rel(ov):.6f}")
    print(f"   BLOCK      vs exact sliding: rel_err={rel(bl):.6f}  (broken)")


if __name__ == "__main__":
    main()
