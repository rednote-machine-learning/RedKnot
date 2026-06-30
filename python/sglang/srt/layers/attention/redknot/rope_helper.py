# Copyright 2024-2026 SGLang RedKnot Integration.
"""RoPE repositioning utility for offline KV reuse.

When a document segment is prefilled offline, its keys are rotated under
positions ``[0, L)``. To plug that KV into a longer concatenated cache,
we need to rotate it again so the rotation phase matches the *target*
positions ``[offset, offset + L)``.

Math:    R(dst) @ R(-src) @ k  =  R(dst - src) @ k

We avoid going through the model's ``rotary_emb`` module directly because
sglang's ``get_rope`` (``sglang.srt.layers.rotary_embedding``) wraps several
RoPE flavours behind different APIs. Instead we re-derive cos/sin from the
``inv_freq`` buffer (universal across Llama / Mistral / Qwen2 / Qwen3 /
Mistral-Nemo / DeepSeek style RoPE) on the target device, which is also
multi-GPU-safe.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class RoPEHelper:
    """Re-rotates KV from a source position range to a destination one.

    Parameters
    ----------
    rotary_emb: nn.Module
        The model's rotary embedding module. Must expose ``inv_freq``;
        optionally ``attention_scaling`` (Llama-3 long-context RoPE).
    """

    def __init__(self, rotary_emb: nn.Module) -> None:
        if not hasattr(rotary_emb, "inv_freq"):
            raise AttributeError(
                "rotary_emb is missing 'inv_freq' buffer; "
                "this RedKnot helper only supports standard RoPE variants."
            )
        self.rotary_emb = rotary_emb

    # ────────────────────────────────────────────────────────────────
    # Low-level rotation primitives
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply(
        self, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if k.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return (k * cos) + (self.rotate_half(k) * sin)

    def _unapply(
        self, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return self._apply(k, cos, -sin)

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def reposition(
        self,
        k_rotated: torch.Tensor,
        src_positions: torch.Tensor,
        dst_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Realign ``k_rotated`` from ``src_positions`` to ``dst_positions``.

        Parameters
        ----------
        k_rotated: ``[B, H, T, D]`` or ``[B, T, D]`` (single-head)
        src_positions: ``[T]`` or ``[B, T]``, int64.
        dst_positions: same shape as ``src_positions``.
        """
        device = k_rotated.device
        dtype = k_rotated.dtype
        src = src_positions.to(device=device, dtype=torch.long)
        dst = dst_positions.to(device=device, dtype=torch.long)
        if src.dim() == 1:
            src = src.unsqueeze(0)
        if dst.dim() == 1:
            dst = dst.unsqueeze(0)

        inv_freq = self.rotary_emb.inv_freq.to(device=device, dtype=torch.float32)
        # NOTE: ``attention_scaling`` (Llama-3 long-context RoPE) is a scalar
        # *magnitude* factor applied to q/k. Repositioning only changes the
        # rotation *phase*; it must NOT touch magnitude, otherwise the
        # unapply/apply pair no longer cancels and K's norm is corrupted
        # (which made Llama-3.3 generations degenerate into repeated tokens).
        # The scaling is identical at src and dst and cancels in
        # R(dst) R(src)^-1, so we omit it entirely here.

        def _cos_sin(pos_long: torch.Tensor):
            pos_f = pos_long.to(torch.float32)
            freqs = torch.einsum("bt,d->btd", pos_f, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
            return cos, sin

        cos_src, sin_src = _cos_sin(src)
        k_no_rope = self._unapply(k_rotated, cos_src, sin_src)

        cos_dst, sin_dst = _cos_sin(dst)
        k_realigned = self._apply(k_no_rope, cos_dst, sin_dst)
        return k_realigned

    @torch.no_grad()
    def reposition_offset(
        self,
        k_rotated: torch.Tensor,
        src_start: int,
        dst_start: int,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """Convenience: shift a contiguous ``[src_start, src_start+L)`` block.

        Useful when the entire segment is moved by a single offset, which is
        the common case for offline KV reuse.
        """
        if length is None:
            length = k_rotated.shape[-2]
        device = k_rotated.device
        src = torch.arange(src_start, src_start + length, device=device)
        dst = torch.arange(dst_start, dst_start + length, device=device)
        return self.reposition(k_rotated, src, dst)
