# Copyright 2024-2026 SGLang RedKnot Integration.
"""Head-class KV transfer for prefill-decode (PD) disaggregation.

Under PD disaggregation the prefill pool produces the KV cache and ships it
to the decode pool before first-token generation. A dense engine must ship
the full ``[B, H, L, D]`` tensor even for heads that will never read most
positions. RedKnot ships only the KV that each head can actually consume
(paper §5.5 / fig. 13a): global heads keep the full context; local heads
keep only the sink + recent window. This reduces transferred KV bytes by
4.3-6.3× in the paper.

This module implements the **payload construction / serialization / restore**
contract, decoupled from any concrete transport (Mooncake / NIXL / RDMA),
plus a byte-accounting helper so the saving can be measured directly.

Pipeline
--------
1. Prefill side: :func:`build_transfer_payload` slices each ``(layer, head)``
   KV down to its head-class visible region and records the kept positions.
2. Transport: :meth:`HeadClassKVPayload.serialize` packs everything into a
   single contiguous buffer + a small metadata header. Any byte transport
   can move ``payload.serialize()``.
3. Decode side: :func:`restore_payload` rebuilds the dense ``[H, L, D]``
   per-layer KV (zero-padding masked positions), which is semantically safe
   for the supported head classes (local masks never read the padded
   positions; global heads are unpadded).

Everything is plain PyTorch + tensors, runs on CPU, and is unit-testable.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
from sglang.srt.layers.attention.redknot.segpaged import (
    POLICY_GLOBAL,
    POLICY_LOCAL,
    local_visible_indices,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Payload
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class HeadKVSlice:
    """One ``(layer, head)`` transferred KV slice."""

    layer: int
    head: int
    policy: str
    positions: torch.Tensor  # [kept] long, intrinsic token positions
    k: torch.Tensor  # [kept, D]
    v: torch.Tensor  # [kept, D]


@dataclass
class HeadClassKVPayload:
    """The head-class KV payload to cross the PD boundary.

    Attributes
    ----------
    num_layers, num_kv_heads, head_dim, seq_len:
        Topology needed to restore the dense layout on the decode side.
    slices:
        Flat list of per-(layer, head) kept slices.
    dtype:
        KV dtype (for restore + byte accounting).
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    slices: List[HeadKVSlice]
    dtype: torch.dtype

    # ── Byte accounting ──
    def transferred_bytes(self) -> int:
        total = 0
        for s in self.slices:
            total += s.k.numel() * s.k.element_size()
            total += s.v.numel() * s.v.element_size()
            total += s.positions.numel() * s.positions.element_size()
        return total

    def dense_bytes(self) -> int:
        """Bytes a dense ``[H, L, D]`` (K+V) transfer would have cost."""
        elt = torch.tensor([], dtype=self.dtype).element_size()
        per_kv = self.num_kv_heads * self.seq_len * self.head_dim * elt
        return 2 * self.num_layers * per_kv  # K + V

    def saving(self) -> Dict[str, float]:
        dense = self.dense_bytes()
        sent = self.transferred_bytes()
        return {
            "dense_bytes": dense,
            "transferred_bytes": sent,
            "byte_saving_frac": 1.0 - (sent / dense) if dense else 0.0,
            "byte_reduction_x": (dense / sent) if sent else float("inf"),
        }

    # ── Serialization (single contiguous buffer) ──
    def serialize(self) -> bytes:
        """Pack the payload into one ``bytes`` blob for any byte transport."""
        buf = io.BytesIO()
        meta = {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
            "dtype": str(self.dtype).replace("torch.", ""),
            "slices": [
                {
                    "layer": s.layer,
                    "head": s.head,
                    "policy": s.policy,
                    "kept": int(s.positions.numel()),
                }
                for s in self.slices
            ],
        }
        tensors = {}
        for i, s in enumerate(self.slices):
            tensors[f"pos_{i}"] = s.positions.cpu()
            tensors[f"k_{i}"] = s.k.cpu()
            tensors[f"v_{i}"] = s.v.cpu()
        torch.save({"meta": meta, "tensors": tensors}, buf)
        return buf.getvalue()

    @classmethod
    def deserialize(cls, blob: bytes) -> "HeadClassKVPayload":
        obj = torch.load(io.BytesIO(blob), weights_only=False)
        meta = obj["meta"]
        tensors = obj["tensors"]
        dtype = getattr(torch, meta["dtype"])
        slices: List[HeadKVSlice] = []
        for i, sm in enumerate(meta["slices"]):
            slices.append(
                HeadKVSlice(
                    layer=sm["layer"],
                    head=sm["head"],
                    policy=sm["policy"],
                    positions=tensors[f"pos_{i}"],
                    k=tensors[f"k_{i}"],
                    v=tensors[f"v_{i}"],
                )
            )
        return cls(
            num_layers=meta["num_layers"],
            num_kv_heads=meta["num_kv_heads"],
            head_dim=meta["head_dim"],
            seq_len=meta["seq_len"],
            slices=slices,
            dtype=dtype,
        )


# ──────────────────────────────────────────────────────────────────────────
# Build (prefill side)
# ──────────────────────────────────────────────────────────────────────────
def _policy_for(head_config: HeadClassConfig, layer: int, head: int) -> str:
    strat = head_config.get_strategy(layer, head)
    return POLICY_LOCAL if strat.is_local() else POLICY_GLOBAL


def build_transfer_payload(
    kv_per_layer: List[Tuple[torch.Tensor, torch.Tensor]],
    head_config: HeadClassConfig,
) -> HeadClassKVPayload:
    """Slice full KV down to the head-class visible region for PD transfer.

    Parameters
    ----------
    kv_per_layer:
        ``num_layers``-long list of ``(K, V)`` tensors, each ``[H, L, D]``
        (the produced prefill KV for one request).
    head_config:
        The Head Class Map governing which positions each head keeps.

    Returns
    -------
    A :class:`HeadClassKVPayload` carrying only the kept KV per head.
    """
    if not kv_per_layer:
        raise ValueError("build_transfer_payload: empty kv_per_layer")
    sample_k = kv_per_layer[0][0]
    if sample_k.dim() != 3:
        raise ValueError(
            f"build_transfer_payload expects [H, L, D] per-layer K, got "
            f"{tuple(sample_k.shape)}"
        )
    H, L, D = sample_k.shape
    num_layers = len(kv_per_layer)
    slices: List[HeadKVSlice] = []

    for li in range(num_layers):
        K, V = kv_per_layer[li]
        for h in range(H):
            strat = head_config.get_strategy(li, h)
            policy = POLICY_LOCAL if strat.is_local() else POLICY_GLOBAL
            if policy == POLICY_GLOBAL:
                pos = torch.arange(L, dtype=torch.long)
            else:
                window = strat.window if strat.window > 0 else 0
                pos = local_visible_indices(
                    L, max(strat.sink_size, 0), window, device=K.device
                ).cpu()
            k_h = K[h].index_select(0, pos.to(K.device)).contiguous()
            v_h = V[h].index_select(0, pos.to(V.device)).contiguous()
            slices.append(
                HeadKVSlice(
                    layer=li,
                    head=h,
                    policy=policy,
                    positions=pos,
                    k=k_h,
                    v=v_h,
                )
            )

    return HeadClassKVPayload(
        num_layers=num_layers,
        num_kv_heads=H,
        head_dim=D,
        seq_len=L,
        slices=slices,
        dtype=sample_k.dtype,
    )


# ──────────────────────────────────────────────────────────────────────────
# Restore (decode side)
# ──────────────────────────────────────────────────────────────────────────
def restore_payload(
    payload: HeadClassKVPayload,
    *,
    device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Rebuild dense ``[H, L, D]`` per-layer KV from a head-class payload.

    Masked (non-transferred) positions of local heads are zero-padded; the
    decode-side attention never reads them because the same Head Class Map
    masks them out (paper note in §4.3 / per_head_storage docstring).

    Returns
    -------
    ``num_layers``-long list of ``(K, V)`` tensors, each ``[H, L, D]``.
    """
    device = device or torch.device("cpu")
    H = payload.num_kv_heads
    L = payload.seq_len
    D = payload.head_dim
    out: List[Tuple[torch.Tensor, torch.Tensor]] = [
        (
            torch.zeros(H, L, D, dtype=payload.dtype, device=device),
            torch.zeros(H, L, D, dtype=payload.dtype, device=device),
        )
        for _ in range(payload.num_layers)
    ]
    for s in payload.slices:
        K, V = out[s.layer]
        pos = s.positions.to(device)
        K[s.head].index_copy_(0, pos, s.k.to(device, payload.dtype))
        V[s.head].index_copy_(0, pos, s.v.to(device, payload.dtype))
    return out


__all__ = [
    "HeadKVSlice",
    "HeadClassKVPayload",
    "build_transfer_payload",
    "restore_payload",
]
