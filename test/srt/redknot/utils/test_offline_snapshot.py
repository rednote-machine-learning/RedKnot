#!/usr/bin/env python3
"""Unit test for RedKnot offline-segment KV snapshot logic (Phase 2).

Verifies the numerics-critical parts WITHOUT loading a real model:
  * snapshot reads the correct token rows from a paged KV pool
  * per-layer shape becomes [1, KVH_per_rank, L, head_dim]
  * (None, None) placeholders for non-captured layers survive
    build_offline_segment / kv_nbytes / to_device
  * round-trip through OfflineKVCache.get returns identical tensors
  * the captured KV matches the consumer's online-read convention
    (k_cache[tok_locs] -> [L, KVH, D] -> movedim -> [1, KVH, L, D])

Run:
  PYTHONPATH=python python test/srt/redknot/test_offline_snapshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

import torch

from sglang.srt.layers.attention.redknot.offline_cache import (
    OfflineKVCache,
    build_offline_segment,
    kv_nbytes,
)


def _ref_consumer_view(k_buf, v_buf, tok_locs):
    """Mirror redknot_backend._redknot_single_extend online-read:
    cache shape [N, KVH, D]; gather rows -> [L, KVH, D] -> [1, KVH, L, D]."""
    k = k_buf[tok_locs].unsqueeze(0).movedim(2, 1)  # [1, KVH, L, D]
    v = v_buf[tok_locs].unsqueeze(0).movedim(2, 1)
    return k, v


def _snapshot_like_backend(k_buf, v_buf, tok_locs):
    """Exactly what RedKnotAttnBackend.snapshot_offline_segment does per layer."""
    k = k_buf[tok_locs].unsqueeze(0).movedim(2, 1).contiguous().clone()
    v = v_buf[tok_locs].unsqueeze(0).movedim(2, 1).contiguous().clone()
    return k, v


def main():
    torch.manual_seed(0)
    N = 256  # pool capacity (tokens)
    KVH = 2  # kv heads per rank
    D = 128  # head dim
    L = 40  # segment length
    n_layers = 8
    full_attn_ids = [3, 7]  # hybrid: only these layers captured

    # Build mock paged KV pool: per-layer [N, KVH, D]
    k_pool = [torch.randn(N, KVH, D) for _ in range(n_layers)]
    v_pool = [torch.randn(N, KVH, D) for _ in range(n_layers)]

    # The request occupies a scattered set of token slots (paged!).
    perm = torch.randperm(N)
    tok_locs = perm[:L]

    # ── Build the offline segment kv list (global-layer-indexed, placeholders
    #    for non-full-attn layers) exactly as snapshot_offline_segment does ──
    max_lid = max(full_attn_ids)
    kv = [(None, None)] * (max_lid + 1)
    for lid in full_attn_ids:
        kv[lid] = _snapshot_like_backend(k_pool[lid], v_pool[lid], tok_locs)

    token_ids = torch.arange(L, dtype=torch.int32)
    seg = build_offline_segment(segment_id="seg0", token_ids=token_ids, kv=kv)

    # ── Assertions on shape & placeholders ──
    assert seg.doc_len == L, seg.doc_len
    assert len(seg.kv) == max_lid + 1
    for lid in range(max_lid + 1):
        if lid in full_attn_ids:
            k, v = seg.kv[lid]
            assert k.shape == (1, KVH, L, D), (lid, k.shape)
            assert v.shape == (1, KVH, L, D), (lid, v.shape)
        else:
            assert seg.kv[lid] == (None, None), lid
    print(f"[ok] shapes & placeholders: doc_len={L}, captured={full_attn_ids}")

    # ── kv_nbytes skips placeholders ──
    expected_bytes = len(full_attn_ids) * 2 * (1 * KVH * L * D) * 4  # float32
    assert kv_nbytes(kv) == expected_bytes, (kv_nbytes(kv), expected_bytes)
    print(f"[ok] kv_nbytes skips None placeholders: {kv_nbytes(kv)} bytes")

    # ── Numeric parity vs the consumer's online-read convention ──
    for lid in full_attn_ids:
        k_ref, v_ref = _ref_consumer_view(k_pool[lid], v_pool[lid], tok_locs)
        k_got, v_got = seg.kv[lid]
        assert torch.equal(k_ref, k_got), f"K mismatch layer {lid}"
        assert torch.equal(v_ref, v_got), f"V mismatch layer {lid}"
    print("[ok] snapshot matches consumer online-read convention exactly")

    # ── Round-trip through OfflineKVCache (put -> get) ──
    cache = OfflineKVCache()
    cache.put(seg)
    got = cache.get("seg0")
    assert got is not None
    for lid in full_attn_ids:
        assert torch.equal(got.kv[lid][0], seg.kv[lid][0]), lid
        assert torch.equal(got.kv[lid][1], seg.kv[lid][1]), lid
    print("[ok] OfflineKVCache put/get round-trip preserves tensors")

    # ── to_device with placeholders does not crash (cpu->cpu no-op) ──
    moved = cache.to_device("seg0", torch.device("cpu"))
    assert moved is not None
    print("[ok] to_device handles None placeholders")

    # ── Independence from pool mutation (snapshot must be a clone) ──
    k_pool[full_attn_ids[0]].zero_()
    k_after, _ = seg.kv[full_attn_ids[0]]
    assert k_after.abs().sum() > 0, "snapshot aliased the pool (not cloned)!"
    print("[ok] snapshot is an independent clone (survives pool mutation)")

    print("\nALL SNAPSHOT UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
