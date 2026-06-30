# Copyright 2024-2026 SGLang RedKnot Integration.
"""Per-(layer, head) trim for offline KV segments.

Storage-only optimisation: the dense ``OfflineSegment.kv`` tensor of shape
``[1, num_kv_heads, L, head_dim]`` is replaced *in storage* by a per-head
list whose length per head follows the ``HeadClassConfig`` policy
(local heads keep only ``sink + window`` tokens; global heads keep all
``L``; retrieval heads keep ``min(L, budget)``). At read time the segment
is re-expanded into the dense ``[1, num_kv_heads, L, head_dim]`` layout
expected by ``redknot.core.online_redknot_forward`` and HF
``DynamicCache``, with missing positions zero-padded.

Why "storage-only":
    The downstream consumer (``core.online_redknot_forward`` line 592)
    calls ``torch.cat`` on per-segment KV tensors and feeds them into a
    HF ``DynamicCache``, which expects a dense ``[B, H, T, D]`` tensor.
    Changing that contract is out of scope here. The per-head trim
    therefore lives entirely on the *storage* side, where it directly
    increases how many segments the host/device caches can hold.

The exact saving on Qwen3-32B with ``qwen3-32B_w256_deepglobal_nolocret``
(435 local_full heads with window=256, sink=4; 77 global heads; 0
retrieval) at B=19K is ~84%: a 4.7 GB raw segment compresses to
~770 MB on disk/host before re-expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .offline_cache import (
    OfflineSegment,
    build_offline_segment,
    kv_nbytes,
)


# ──────────────────────────────────────────────────────────────────────────
# Trim policy
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class HeadTrimPolicy:
    """Per-(layer, kv_head) keep-length policy.

    Attributes
    ----------
    keep_lens:
        ``[num_layers][num_kv_heads]`` int list. Each entry is the number
        of *tail* tokens to retain for that head's K/V at the segment's
        intrinsic position window ``[L - keep_len, L)``. A value of ``-1``
        means "keep all L tokens" (i.e. global heads). A value greater
        than ``L`` is silently clamped to ``L``.
    sink_lens:
        ``[num_layers][num_kv_heads]`` int list. Number of *head* tokens
        to retain at positions ``[0, sink_len)`` in addition to the tail
        window. Defaults to all zeros if omitted. The local head policy
        is encoded as ``sink_len = sink_size, keep_len = window``.
    """

    keep_lens: List[List[int]]
    sink_lens: List[List[int]]

    def num_layers(self) -> int:
        return len(self.keep_lens)

    def num_kv_heads(self) -> int:
        return len(self.keep_lens[0]) if self.keep_lens else 0

    @classmethod
    def from_head_config(
        cls,
        head_cfg,
        *,
        retrieval_budget: int = 512,
    ) -> "HeadTrimPolicy":
        """Build a trim policy from a ``redknot.HeadClassConfig``-like
        object.

        Parameters
        ----------
        head_cfg:
            Must expose ``head_class[L][H]``, ``head_max_distance[L][H]``,
            ``head_sink_size[L][H]``. We match ``redknot.v7_head``'s
            ``HeadClassConfig`` shape exactly.
        retrieval_budget:
            Static budget for "retrieval" heads in the absence of a
            dynamic score-driven selector. Conservative default keeps
            ``retrieval_budget`` tail tokens so the dynamic top-p path
            still has room to operate on the read side.
        """
        n_layers = len(head_cfg.head_class)
        n_heads = len(head_cfg.head_class[0])
        keep_lens: List[List[int]] = []
        sink_lens: List[List[int]] = []
        for li in range(n_layers):
            row_keep: List[int] = []
            row_sink: List[int] = []
            for hi in range(n_heads):
                cls_ = head_cfg.head_class[li][hi]
                w = head_cfg.head_max_distance[li][hi]
                s = head_cfg.head_sink_size[li][hi]
                if cls_ in ("global", "dense"):
                    row_keep.append(-1)
                    row_sink.append(0)
                elif cls_ in ("local", "local_full"):
                    # window=w tail tokens + sink head tokens
                    row_keep.append(int(w) if w > 0 else 0)
                    row_sink.append(int(s) if s > 0 else 0)
                elif cls_ in ("retrieval",):
                    row_keep.append(int(retrieval_budget))
                    row_sink.append(int(s) if s > 0 else 0)
                else:
                    # Unknown: be safe, keep all.
                    row_keep.append(-1)
                    row_sink.append(0)
            keep_lens.append(row_keep)
            sink_lens.append(row_sink)
        return cls(keep_lens=keep_lens, sink_lens=sink_lens)


# ──────────────────────────────────────────────────────────────────────────
# Trimmed segment
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TrimmedSegment:
    """Per-(layer, head) trimmed KV storage for one segment.

    Layout
    ------
    ``per_head_kv[layer][head]`` is a ``(K, V)`` tuple where each tensor
    has shape ``[1, kept_len, head_dim]`` for that head, or shape
    ``[1, 0, head_dim]`` if ``kept_len == 0``. The intrinsic positions
    are recorded in ``positions[layer][head]`` as a 1-D LongTensor of
    length ``kept_len`` (values in ``[0, doc_len)``).

    The dense ``OfflineSegment.kv`` can always be reconstructed via
    :func:`expand_to_dense` (zero-padding the missing positions).
    """

    segment_id: str
    doc_len: int
    head_dim: int
    num_kv_heads: int
    num_layers: int
    per_head_kv: List[List[Tuple[torch.Tensor, torch.Tensor]]]
    positions: List[List[torch.Tensor]]
    nbytes: int
    device: torch.device
    meta: Dict[str, object] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Trim
# ──────────────────────────────────────────────────────────────────────────
def _select_indices(L: int, keep_len: int, sink_len: int) -> torch.Tensor:
    """Compute the kept token indices for one head at intrinsic length L.

    keep_len = -1 means "keep all". Sink and tail windows are merged
    (deduplicated) so overlapping ranges are handled cleanly.
    """
    if keep_len < 0 or keep_len >= L:
        # Keep all.
        return torch.arange(L, dtype=torch.long)
    sink_len = max(0, min(sink_len, L))
    tail_len = max(0, min(keep_len, L))
    # Sink head: [0, sink_len). Tail: [L - tail_len, L).
    tail_start = max(sink_len, L - tail_len)
    sink_idx = torch.arange(0, sink_len, dtype=torch.long)
    tail_idx = torch.arange(tail_start, L, dtype=torch.long)
    if sink_idx.numel() == 0:
        return tail_idx
    if tail_idx.numel() == 0:
        return sink_idx
    return torch.cat([sink_idx, tail_idx])


def trim_segment_per_head(
    segment: OfflineSegment, policy: HeadTrimPolicy
) -> TrimmedSegment:
    """Materialise a per-head trimmed view of ``segment``.

    The returned :class:`TrimmedSegment` references freshly-allocated
    sub-tensors (contiguous slices) so the original ``segment.kv`` can be
    freed by the caller. Total storage is the sum of per-head slice
    sizes; for the Qwen3-32B "85% local + 15% global" recipe at L=19K
    this is ~16% of ``segment.nbytes``.
    """
    if not segment.kv:
        raise ValueError("trim_segment_per_head: segment.kv is empty")
    n_layers = len(segment.kv)
    sample_k = segment.kv[0][0]
    if sample_k.dim() != 4:
        raise ValueError(
            f"trim_segment_per_head: expected 4-D [B, H, L, D] K tensor, "
            f"got shape {tuple(sample_k.shape)}"
        )
    B = sample_k.shape[0]
    if B != 1:
        raise ValueError(f"trim_segment_per_head: only batch=1 supported (got B={B})")
    n_heads = sample_k.shape[1]
    L = sample_k.shape[2]
    head_dim = sample_k.shape[3]
    if L != segment.doc_len:
        raise ValueError(
            f"trim_segment_per_head: K len {L} != segment.doc_len {segment.doc_len}"
        )
    if policy.num_layers() != n_layers:
        raise ValueError(
            f"trim_segment_per_head: policy n_layers "
            f"{policy.num_layers()} != segment n_layers {n_layers}"
        )
    if policy.num_kv_heads() != n_heads:
        raise ValueError(
            f"trim_segment_per_head: policy n_kv_heads "
            f"{policy.num_kv_heads()} != segment n_kv_heads {n_heads}"
        )

    per_head_kv: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    positions: List[List[torch.Tensor]] = []
    total_bytes = 0
    device = sample_k.device

    for li in range(n_layers):
        K_full, V_full = segment.kv[li]
        # Sanity: same H, L per layer (we trust this in canonical RC).
        layer_heads_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        layer_positions: List[torch.Tensor] = []
        for hi in range(n_heads):
            keep = policy.keep_lens[li][hi]
            sink = policy.sink_lens[li][hi]
            idx = _select_indices(L, keep, sink).to(device)
            # K_full[:, hi, :, :] is [1, L, D]; gather along seq dim.
            K_h = K_full[:, hi, :, :].index_select(1, idx).contiguous()
            V_h = V_full[:, hi, :, :].index_select(1, idx).contiguous()
            layer_heads_kv.append((K_h, V_h))
            layer_positions.append(idx.to("cpu"))
            total_bytes += K_h.numel() * K_h.element_size()
            total_bytes += V_h.numel() * V_h.element_size()
        per_head_kv.append(layer_heads_kv)
        positions.append(layer_positions)

    return TrimmedSegment(
        segment_id=segment.segment_id,
        doc_len=L,
        head_dim=head_dim,
        num_kv_heads=n_heads,
        num_layers=n_layers,
        per_head_kv=per_head_kv,
        positions=positions,
        nbytes=total_bytes,
        device=device,
        meta=dict(segment.meta),
    )


# ──────────────────────────────────────────────────────────────────────────
# Expand back to dense
# ──────────────────────────────────────────────────────────────────────────
def expand_to_dense(
    trimmed: TrimmedSegment,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> OfflineSegment:
    """Reconstruct a dense ``OfflineSegment`` from a ``TrimmedSegment``.

    Missing positions are zero-padded. This is the canonical read path
    consumed by ``redknot.core.online_redknot_forward``: the dense
    tensor it returns is interface-compatible with the original
    ``OfflineSegment.kv`` and can be fed verbatim into ``torch.cat`` and
    HF ``DynamicCache``.

    Note
    ----
    The zero-padding is *semantically safe* for the head classes we
    support:
      - ``local_full``: positions outside the ``[sink ∪ tail-window]``
        view are not read by the local strategy mask, so zeros there
        are a no-op.
      - ``global``: ``keep_len = -1`` means we kept everything, so
        there is no padding for global heads.
      - ``retrieval``: positions outside the ``retrieval_budget``
        slice are masked out by the top-p mask before any scores
        contribute, so the zeros never reach the softmax numerator.
    The padding is purely a layout-restoration trick to keep the
    downstream ``[B, H, L, D]`` contract intact.
    """
    target_device = device if device is not None else trimmed.device
    K0_sample = trimmed.per_head_kv[0][0][0]
    target_dtype = dtype if dtype is not None else K0_sample.dtype
    L = trimmed.doc_len
    H = trimmed.num_kv_heads
    D = trimmed.head_dim

    dense_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for li in range(trimmed.num_layers):
        K_dense = torch.zeros((1, H, L, D), dtype=target_dtype, device=target_device)
        V_dense = torch.zeros((1, H, L, D), dtype=target_dtype, device=target_device)
        for hi in range(H):
            K_h, V_h = trimmed.per_head_kv[li][hi]
            if K_h.shape[1] == 0:
                continue
            idx = trimmed.positions[li][hi].to(target_device)
            # K_dense[0, hi].index_copy_ on dim=0 (seq dim within the
            # [L, D] sub-view).
            K_dense[0, hi].index_copy_(
                0, idx, K_h[0].to(target_device, dtype=target_dtype)
            )
            V_dense[0, hi].index_copy_(
                0, idx, V_h[0].to(target_device, dtype=target_dtype)
            )
        dense_kv.append((K_dense, V_dense))

    return OfflineSegment(
        segment_id=trimmed.segment_id,
        token_ids=torch.empty(
            L, dtype=torch.int32
        ),  # placeholder; caller may overwrite
        doc_len=L,
        kv=dense_kv,
        device=target_device,
        nbytes=kv_nbytes(dense_kv),
        meta=dict(trimmed.meta),
    )


# ──────────────────────────────────────────────────────────────────────────
# Stats helpers (for benches)
# ──────────────────────────────────────────────────────────────────────────
def estimate_savings(
    segment: OfflineSegment, policy: HeadTrimPolicy
) -> Dict[str, float]:
    """Predict bytes saved by a trim without materialising tensors.

    Returns a dict with ``original_bytes``, ``trimmed_bytes``,
    ``saving_frac`` and ``per_class_breakdown`` (kept head-tokens per
    head class). Cheap; intended for sweep planning.
    """
    if not segment.kv:
        return {
            "original_bytes": 0,
            "trimmed_bytes": 0,
            "saving_frac": 0.0,
            "per_class_breakdown": {},
        }
    sample_k = segment.kv[0][0]
    L = sample_k.shape[2]
    head_dim = sample_k.shape[3]
    elt = sample_k.element_size()
    bytes_per_token_per_head_kv = 2 * head_dim * elt  # K + V

    orig = policy.num_layers() * policy.num_kv_heads() * L * bytes_per_token_per_head_kv
    trimmed = 0
    per_class_tokens: Dict[str, int] = {}
    for li in range(policy.num_layers()):
        for hi in range(policy.num_kv_heads()):
            keep = policy.keep_lens[li][hi]
            sink = policy.sink_lens[li][hi]
            n = _select_indices(L, keep, sink).numel()
            trimmed += n * bytes_per_token_per_head_kv
            # Classify for the breakdown.
            if keep < 0:
                cls = "global"
            elif keep + sink >= L:
                cls = "full_via_overflow"
            else:
                cls = "local_or_retrieval"
            per_class_tokens[cls] = per_class_tokens.get(cls, 0) + n
    saving = 1.0 - (trimmed / orig if orig else 0.0)
    return {
        "original_bytes": orig,
        "trimmed_bytes": trimmed,
        "saving_frac": saving,
        "per_class_breakdown": per_class_tokens,
    }
