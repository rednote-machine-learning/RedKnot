#!/usr/bin/env python3
"""Verify whether GatedDeltaNet state can be MERGED from independently-computed
segments (offline) vs only SEQUENTIALLY accumulated.

The recurrence (modeling_qwen3_5_moe.torch_recurrent_gated_delta_rule):
    S_t = S_{t-1} * g_t
    kv_mem = (S_t * k_t).sum(-2)
    delta = (v_t - kv_mem) * beta_t        # <-- depends on current S (history!)
    S_t = S_t + k_t (outer) delta

Because `delta` subtracts the running prediction `kv_mem`, each token's
contribution depends on the accumulated state. So a segment computed from ZERO
state differs from the same segment computed from a PREFIX state. We test:

  REF      : run seg1+seg2 sequentially from zero (true state).
  RELAY    : run seg1 from zero -> S1; run seg2 from S1 (sequential, online).
  MERGE    : run seg1 from zero -> S1; run seg2 from ZERO -> S2 + decay; then
             try S_total = decay_seg2 * S1 + S2  (offline-independent + merge).

If RELAY==REF but MERGE!=REF, then delta-rule segments CANNOT be merged from
independent offline computation (only sequential accumulation works -> no
offline saving). Pure synthetic tensors, fast.
"""

from __future__ import annotations

import torch


def delta_rule(q, k, v, g, beta, S0=None):
    """One head batch. q,k,v,g,beta: [T, d] (g,beta scalars per token).
    Returns (out [T,dv], S_final [dk,dv]). Mirrors the HF torch recurrence."""
    T, dk = k.shape
    dv = v.shape[-1]
    S = torch.zeros(dk, dv) if S0 is None else S0.clone()
    out = torch.zeros(T, dv)
    decay_prod = torch.ones(())  # product of g over this segment
    for i in range(T):
        g_t = g[i].exp()
        S = S * g_t
        decay_prod = decay_prod * g_t
        kv_mem = (S * k[i].unsqueeze(-1)).sum(dim=0)  # [dv]
        delta = (v[i] - kv_mem) * beta[i]
        S = S + k[i].unsqueeze(-1) * delta.unsqueeze(0)
        out[i] = (S * q[i].unsqueeze(-1)).sum(dim=0)
    return out, S, decay_prod


def main():
    torch.manual_seed(0)
    dk, dv = 16, 16
    T1, T2 = 30, 30

    def rnd(T):
        q = torch.randn(T, dk)
        k = torch.randn(T, dk)
        v = torch.randn(T, dv)
        g = -torch.rand(T) * 0.1  # small negative -> g.exp() in (0,1)
        beta = torch.rand(T)
        return q, k, v, g, beta

    q1, k1, v1, g1, b1 = rnd(T1)
    q2, k2, v2, g2, b2 = rnd(T2)

    # REF: sequential from zero over concat
    q = torch.cat([q1, q2])
    k = torch.cat([k1, k2])
    v = torch.cat([v1, v2])
    g = torch.cat([g1, g2])
    b = torch.cat([b1, b2])
    _, S_ref, _ = delta_rule(q, k, v, g, b)

    # RELAY: seg1 from zero -> S1; seg2 from S1
    _, S1, _ = delta_rule(q1, k1, v1, g1, b1)
    _, S_relay, _ = delta_rule(q2, k2, v2, g2, b2, S0=S1)

    # MERGE (offline-independent): seg2 from ZERO -> S2, decay2; merge
    _, S2, decay2 = delta_rule(q2, k2, v2, g2, b2, S0=None)
    S_merge = decay2 * S1 + S2  # naive linear-attn-style merge with decay

    d_relay = (S_relay - S_ref).abs().max().item()
    d_merge = (S_merge - S_ref).abs().max().item()
    ref_scale = S_ref.abs().max().item()
    print("=" * 64)
    print(" GatedDeltaNet segment merge test")
    print(f"   S_ref scale            : {ref_scale:.4f}")
    print(
        f"   RELAY (seq accumulate) : max|diff|={d_relay:.6f}  -> {'EXACT' if d_relay < 1e-4 else 'DIFF'}"
    )
    print(
        f"   MERGE (offline+decay)  : max|diff|={d_merge:.6f}  -> {'EXACT' if d_merge < 1e-4 else 'DIFF'}"
    )
    print("=" * 64)
    if d_relay < 1e-4 and d_merge > 1e-2:
        print(" CONCLUSION: delta-rule needs SEQUENTIAL accumulation; independent")
        print(" offline segments CANNOT be merged (delta depends on history).")
        print(" -> No offline-compute saving for linear via segment merge.")
    elif d_merge < 1e-4:
        print(" CONCLUSION: MERGE works! offline-independent segments can be")
        print(" merged with decay -> offline linear reuse is viable.")
    print("=" * 64)


if __name__ == "__main__":
    main()
