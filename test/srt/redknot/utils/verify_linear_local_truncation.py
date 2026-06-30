#!/usr/bin/env python3
"""Verify: can a FAST-decay GatedDeltaNet head be truncated to a LOCAL window
(only the last W tokens) without changing its output?

Mechanism: in the recurrence S_t = g_t*S_{t-1} + k_t(outer)delta_t, contributions
from tokens older than ~1/(1-g) are multiplied by a product of decays ->
exponentially forgotten. So for a head whose effective memory mem_len << W, the
output computed from only the last W tokens should equal the full-history output.

PART A (synthetic, exact): build heads at various decay rates; compare full vs
windowed output at query position. Reports max output diff vs mem_len/window.

PART B (head-class rule): given per-head mem_len (decay), classify heads as LOCAL
(mem_len < W -> truncate) vs GLOBAL (keep full). Report the output error when
applying this rule across a realistic decay distribution.

This validates the "linear state adapts to local" claim before wiring it into
the model.
"""

from __future__ import annotations

import torch


def _l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def delta_rule_windowed(q_last, k, v, g, beta, window):
    """Compute the LAST token's output using only the last `window` tokens
    (state reset to zero at the window start). q_last is the final query.
    k,v,g,beta: full [T,*]; we slice the tail. Matches the HF kernel which
    L2-normalizes q,k and scales q by 1/sqrt(dk)."""
    T, dk = k.shape
    dv = v.shape[-1]
    scale = 1.0 / (dk**0.5)
    start = max(0, T - window)
    S = torch.zeros(dk, dv)
    kln = _l2norm(k)
    for i in range(start, T):
        g_t = g[i].exp()
        S = S * g_t
        kv_mem = (S * kln[i].unsqueeze(-1)).sum(dim=0)
        delta = (v[i] - kv_mem) * beta[i]
        S = S + kln[i].unsqueeze(-1) * delta.unsqueeze(0)
    q_n = _l2norm(q_last) * scale
    out = (S * q_n.unsqueeze(-1)).sum(dim=0)
    return out


def main():
    torch.manual_seed(0)
    dk = dv = 32
    T = 4000

    def make_head(decay_mean):
        # choose A so that mean g.exp ~ decay_mean
        q = torch.randn(T, dk)
        k = torch.randn(T, dk)
        v = torch.randn(T, dv)
        beta = torch.rand(T)
        # g negative; g.exp ~ decay_mean -> g ~ ln(decay_mean)
        import math

        base = math.log(max(decay_mean, 1e-4))
        g = base + 0.05 * torch.randn(T)  # jitter around target
        g = g.clamp(max=-1e-4)
        return q, k, v, g, beta

    print("=" * 74)
    print(" PART A: full-history vs windowed output for heads of varying decay")
    print("=" * 74)
    print(
        f" {'decay':>7} {'mem_len':>8} {'win=256':>10} {'win=512':>10} {'win=2048':>10}"
    )
    for dm in [0.5, 0.9, 0.97, 0.99, 0.999, 0.9999]:
        q, k, v, g, beta = make_head(dm)
        mem = 1.0 / (1.0 - min(dm, 0.99999))
        # full
        out_full = delta_rule_windowed(q[-1], k, v, g, beta, window=T)
        diffs = []
        for w in [256, 512, 2048]:
            out_w = delta_rule_windowed(q[-1], k, v, g, beta, window=w)
            rel = (out_w - out_full).abs().max().item() / (
                out_full.abs().max().item() + 1e-6
            )
            diffs.append(rel)
        print(
            f" {dm:>7.4f} {mem:>8.0f} {diffs[0]:>10.4f} {diffs[1]:>10.4f} {diffs[2]:>10.4f}"
        )
    print("-" * 74)
    print(" (relative max output error; ~0 means windowing is lossless for that head)")

    print("\n" + "=" * 74)
    print(" PART B: head-class rule — LOCAL if mem_len<W (truncate), else GLOBAL")
    print("=" * 74)
    # realistic mix matching the measured distribution (~69% short, ~31% long)
    decays = (
        [0.5] * 20
        + [0.9] * 25
        + [0.97] * 24
        + [0.99] * 15
        + [0.999] * 10
        + [0.9999] * 6
    )
    for W in [256, 512, 1024]:
        errs_local = []
        n_local = n_global = 0
        for dm in decays:
            mem = 1.0 / (1.0 - min(dm, 0.99999))
            q, k, v, g, beta = make_head(dm)
            out_full = delta_rule_windowed(q[-1], k, v, g, beta, window=T)
            if mem < W:  # LOCAL: truncate to window
                n_local += 1
                out_w = delta_rule_windowed(q[-1], k, v, g, beta, window=W)
                errs_local.append(
                    (out_w - out_full).abs().max().item()
                    / (out_full.abs().max().item() + 1e-6)
                )
            else:  # GLOBAL: keep full (zero error)
                n_global += 1
        frac_local = n_local / len(decays)
        mean_err = sum(errs_local) / len(errs_local) if errs_local else 0.0
        max_err = max(errs_local) if errs_local else 0.0
        print(
            f" W={W:5d}: local={n_local} ({frac_local * 100:.0f}%) global={n_global} | "
            f"local-head rel err mean={mean_err:.4f} max={max_err:.4f}"
        )
    print("=" * 74)
    print(" If local-head error ~0, truncating fast heads to a window is lossless,")
    print(" and ~frac_local of linear heads can be windowed -> compute saving.")
    print("=" * 74)


if __name__ == "__main__":
    main()
