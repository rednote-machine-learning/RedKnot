# Copyright 2024-2026 SGLang RedKnot Integration.
"""SegPagedAttention: per-(layer, head) paged KV runtime for RedKnot.

This is the P0 storage-and-execution substrate from the paper (§4.3,
Algorithm 2, fig. 6). It replaces the dense ``[B, H, L, D]`` KV layout with
a **head-segmented** paged store so each KV head physically retains only the
tokens it needs, and runs attention through a **mask-free fused varlen**
kernel — the contract that turns algorithmic head sparsity into real GPU
work reduction (paper §5.4: 2.5-21× decode, 6.3-23.3× prefill speedup over
dense+mask, at cos > 0.99998 equivalence).

What this module provides
-------------------------
- :class:`HeadSegment` / :class:`SegmentPageTable` — the ``(ℓ, h, s) ->
  consecutive virtual pages`` mapping (paper fig. 6). Pages are a virtual
  address space over a per-(layer, head) page pool.
- :class:`SegPagedKVCache` — the physical per-head paged KV store. Local
  heads allocate pages only for ``sink + recent window``; global heads
  allocate full-context pages. This is the layout that makes the byte
  savings physical (paper §5.5: 4.7-7.8× concurrent capacity).
- :func:`segpaged_attention` — Algorithm 2: iterate head segments, query the
  page table, apply the per-head execution policy (GLOBAL = full pages;
  LOCAL = sink pages ∪ recent pages), and merge head outputs. Uses the
  fused varlen FA-3 kernel when available, with an exact PyTorch reference
  fallback so the path is CPU-testable and numerically checkable.
- :func:`dense_reference_attention` and :func:`verify_against_dense` — the
  numerical-equivalence harness (cos vs the dense masked baseline) and a
  memory-footprint accounting that quantifies the per-head capacity win.

Notes
-----
- The runtime is **decoupled from sglang's TokenToKVPool**: SegPagedAttention
  manages its own virtual page space, exactly as the paper frames it as a
  separate KV abstraction. A backend integration can map these pages onto a
  real pool later; here the focus is a correct, verifiable runtime.
- Everything is plain PyTorch and runs on CPU; the FA-3 fused path is used
  only when ``sgl_kernel.flash_attn`` + Hopper are present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Head execution policies (Algorithm 2 line 4).
POLICY_GLOBAL = "global"
POLICY_LOCAL = "local"

# Optional fused varlen FA-3 kernel.
try:
    from sgl_kernel.flash_attn import (
        flash_attn_varlen_func as _fa3_varlen,
        is_fa3_supported as _is_fa3_supported_hw,
    )

    _HAS_FA3 = True
except Exception as exc:  # pragma: no cover - environment specific
    _fa3_varlen = None
    _is_fa3_supported_hw = None
    _HAS_FA3 = False
    logger.info("SegPagedAttention: FA-3 fused varlen unavailable (%s).", exc)


def is_fused_varlen_available() -> bool:
    """True iff the fused FA-3 varlen kernel is importable and supported."""
    if not _HAS_FA3:
        return False
    try:
        return bool(_is_fa3_supported_hw())
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Page table  (paper fig. 6: head segment -> consecutive virtual pages)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class HeadSegment:
    """One head segment ``g = (layer, head, segment)`` and its page span.

    Attributes
    ----------
    layer, head, segment:
        The ``(ℓ, h, s)`` index.
    page_ids:
        Consecutive virtual page ids this segment maps to (paper: a segment
        ``(L0, h0, s0)`` maps to pages 3-4, etc.).
    seq_len:
        Number of valid KV tokens stored across this segment's pages (the
        last page may be partially filled).
    """

    layer: int
    head: int
    segment: int
    page_ids: List[int]
    seq_len: int


@dataclass
class SegmentPageTable:
    """``T[(ℓ, h, s)] -> consecutive virtual pages`` (Algorithm 2 line 3).

    Also records each ``(ℓ, h)`` head's execution policy (Algorithm 2 line 4)
    and the per-head ordered list of its segments.
    """

    page_size: int
    segments: Dict[Tuple[int, int, int], HeadSegment] = field(default_factory=dict)
    head_policy: Dict[Tuple[int, int], str] = field(default_factory=dict)
    head_segments: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)

    def lookup(self, layer: int, head: int, segment: int) -> HeadSegment:
        return self.segments[(layer, head, segment)]

    def policy(self, layer: int, head: int) -> str:
        return self.head_policy[(layer, head)]

    def segments_of(self, layer: int, head: int) -> List[HeadSegment]:
        seg_ids = self.head_segments.get((layer, head), [])
        return [self.segments[(layer, head, s)] for s in seg_ids]


# ──────────────────────────────────────────────────────────────────────────
# Physical paged KV store
# ──────────────────────────────────────────────────────────────────────────
class SegPagedKVCache:
    """Per-(layer, head) paged KV store with a virtual page table.

    Physical pages live in a single pool of shape
    ``[num_pages, page_size, head_dim]`` for K and V separately. Each head
    segment owns a contiguous run of virtual pages, which are translated to
    physical pages via ``_page_phys``. Because local heads only ever
    *append* the tokens inside their visible window, their physical
    footprint stays bounded regardless of context length — this is the
    capacity win in the paper.

    Parameters
    ----------
    num_layers, num_kv_heads, head_dim:
        Model topology (after TP sharding).
    page_size:
        Tokens per page.
    device, dtype:
        Pool placement.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 64,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.table = SegmentPageTable(page_size=page_size)
        # Physical page pools (grown lazily).
        self._k_pages: List[torch.Tensor] = []
        self._v_pages: List[torch.Tensor] = []
        self._next_phys = 0
        # virtual page id -> physical page id
        self._virt_to_phys: Dict[int, int] = {}
        self._next_virt = 0

    # ────────────────────────────────────────────────────────────────
    # Allocation
    # ────────────────────────────────────────────────────────────────
    def _alloc_pages(self, n_tokens: int) -> List[int]:
        n_pages = (n_tokens + self.page_size - 1) // self.page_size
        virt_ids: List[int] = []
        for _ in range(n_pages):
            vid = self._next_virt
            self._next_virt += 1
            phys = self._next_phys
            self._next_phys += 1
            self._virt_to_phys[vid] = phys
            self._k_pages.append(
                torch.zeros(
                    self.page_size, self.head_dim, device=self.device, dtype=self.dtype
                )
            )
            self._v_pages.append(
                torch.zeros(
                    self.page_size, self.head_dim, device=self.device, dtype=self.dtype
                )
            )
            virt_ids.append(vid)
        return virt_ids

    def add_head_segment(
        self,
        *,
        layer: int,
        head: int,
        segment: int,
        policy: str,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> HeadSegment:
        """Store one head's KV for one segment into freshly allocated pages.

        Parameters
        ----------
        k, v:
            ``[seq_len, head_dim]`` tensors for this ``(layer, head, segment)``.
            For LOCAL heads the caller passes only the visible tokens
            (sink ∪ recent window); for GLOBAL heads the full segment.
        policy:
            ``POLICY_GLOBAL`` or ``POLICY_LOCAL``.
        """
        if k.dim() != 2 or v.dim() != 2:
            raise ValueError(
                f"add_head_segment expects [seq_len, head_dim] k/v, got "
                f"{tuple(k.shape)} / {tuple(v.shape)}"
            )
        seq_len = k.shape[0]
        k = k.to(self.device, self.dtype)
        v = v.to(self.device, self.dtype)
        virt_ids = self._alloc_pages(seq_len)
        # Scatter tokens into pages.
        for i, vid in enumerate(virt_ids):
            phys = self._virt_to_phys[vid]
            start = i * self.page_size
            end = min(start + self.page_size, seq_len)
            n = end - start
            self._k_pages[phys][:n] = k[start:end]
            self._v_pages[phys][:n] = v[start:end]

        seg = HeadSegment(
            layer=layer,
            head=head,
            segment=segment,
            page_ids=virt_ids,
            seq_len=seq_len,
        )
        self.table.segments[(layer, head, segment)] = seg
        self.table.head_policy[(layer, head)] = policy
        self.table.head_segments.setdefault((layer, head), []).append(segment)
        return seg

    # ────────────────────────────────────────────────────────────────
    # Read
    # ────────────────────────────────────────────────────────────────
    def gather_segment(self, seg: HeadSegment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialise one head segment's ``[seq_len, head_dim]`` K/V."""
        k_parts, v_parts = [], []
        remaining = seg.seq_len
        for vid in seg.page_ids:
            phys = self._virt_to_phys[vid]
            take = min(self.page_size, remaining)
            k_parts.append(self._k_pages[phys][:take])
            v_parts.append(self._v_pages[phys][:take])
            remaining -= take
        if not k_parts:
            return (
                torch.zeros(0, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.head_dim, device=self.device, dtype=self.dtype),
            )
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def gather_head(self, layer: int, head: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all segments of a ``(layer, head)`` into one K/V view."""
        segs = self.table.segments_of(layer, head)
        if not segs:
            return (
                torch.zeros(0, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.head_dim, device=self.device, dtype=self.dtype),
            )
        ks, vs = [], []
        for seg in segs:
            k, v = self.gather_segment(seg)
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    # ────────────────────────────────────────────────────────────────
    # Accounting
    # ────────────────────────────────────────────────────────────────
    def physical_bytes(self) -> int:
        """Total bytes of allocated physical pages."""
        if not self._k_pages:
            return 0
        per_page = self._k_pages[0].numel() * self._k_pages[0].element_size()
        return 2 * len(self._k_pages) * per_page  # K + V

    def stored_token_count(self) -> int:
        """Sum of stored KV tokens across all head segments."""
        return sum(seg.seq_len for seg in self.table.segments.values())


# ──────────────────────────────────────────────────────────────────────────
# Builder: dense per-head KV -> SegPaged store with head-class policy
# ──────────────────────────────────────────────────────────────────────────
def build_segpaged_cache(
    *,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    head_policy: List[str],
    sink: int,
    recent: int,
    page_size: int = 64,
    layer: int = 0,
) -> SegPagedKVCache:
    """Build a SegPaged store for one layer from dense per-head KV.

    Parameters
    ----------
    k_dense, v_dense:
        ``[num_kv_heads, L, head_dim]`` dense KV for one layer.
    head_policy:
        Length ``num_kv_heads`` list of ``POLICY_GLOBAL`` / ``POLICY_LOCAL``.
    sink, recent:
        Local-head visible window: keep ``[0, sink)`` ∪ ``[L - recent, L)``.
    page_size:
        Tokens per page.

    Returns
    -------
    A :class:`SegPagedKVCache` holding one segment per head, where local
    heads physically store only ``sink + recent`` tokens.
    """
    if k_dense.dim() != 3:
        raise ValueError(
            f"build_segpaged_cache expects [H, L, D] k_dense, got "
            f"{tuple(k_dense.shape)}"
        )
    H, L, D = k_dense.shape
    cache = SegPagedKVCache(
        num_layers=layer + 1,
        num_kv_heads=H,
        head_dim=D,
        page_size=page_size,
        device=k_dense.device,
        dtype=k_dense.dtype,
    )
    for h in range(H):
        policy = head_policy[h]
        if policy == POLICY_GLOBAL:
            k_h = k_dense[h]
            v_h = v_dense[h]
        elif policy == POLICY_LOCAL:
            idx = local_visible_indices(L, sink, recent, device=k_dense.device)
            k_h = k_dense[h].index_select(0, idx)
            v_h = v_dense[h].index_select(0, idx)
        else:
            raise ValueError(f"Unknown head policy {policy!r}")
        cache.add_head_segment(
            layer=layer, head=h, segment=0, policy=policy, k=k_h, v=v_h
        )
    return cache


def local_visible_indices(
    L: int, sink: int, recent: int, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return ``[0, sink) ∪ [L - recent, L)`` indices (Algorithm 2 line 9-10)."""
    sink = max(0, min(sink, L))
    recent = max(0, min(recent, L))
    tail_start = max(sink, L - recent)
    sink_idx = torch.arange(0, sink, dtype=torch.long, device=device)
    tail_idx = torch.arange(tail_start, L, dtype=torch.long, device=device)
    if sink_idx.numel() == 0:
        return tail_idx
    if tail_idx.numel() == 0:
        return sink_idx
    return torch.cat([sink_idx, tail_idx])


# ──────────────────────────────────────────────────────────────────────────
# Attention execution  (Algorithm 2)
# ──────────────────────────────────────────────────────────────────────────
def _sdpa_one_head(
    q: torch.Tensor,  # [Lq, D]
    k: torch.Tensor,  # [Lk, D]
    v: torch.Tensor,  # [Lk, D]
    sm_scale: float,
) -> torch.Tensor:
    """Exact non-causal attention for one head (reference path)."""
    if k.shape[0] == 0:
        return torch.zeros_like(q)
    scores = (q.float() @ k.float().transpose(0, 1)) * sm_scale  # [Lq, Lk]
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(q.dtype)


def _fused_varlen_attention(
    q_packed: torch.Tensor,  # [sum_Lq, Hq, D]
    k_packed: torch.Tensor,  # [sum_Lk, Hkv, D]
    v_packed: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    sm_scale: float,
) -> torch.Tensor:
    """Single mask-free fused FA-3 varlen call (paper §5.4 fused path)."""
    out = _fa3_varlen(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=sm_scale,
        causal=False,
        ver=3,
    )
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def segpaged_attention(
    query: torch.Tensor,  # [Hq, Lq, D]
    cache: SegPagedKVCache,
    *,
    layer: int,
    num_q_per_kv: int,
    sm_scale: float,
    use_fused: bool = True,
) -> torch.Tensor:
    """SegPagedAttention forward for one layer (Algorithm 2).

    For each KV head: look up its segments in the page table, apply the
    head execution policy (GLOBAL reads all mapped pages; LOCAL reads only
    the visible sink/recent pages it physically stores), run mask-free
    attention, and merge head outputs.

    Parameters
    ----------
    query:
        ``[Hq, Lq, D]`` query for this layer (already RoPE-applied).
    cache:
        The SegPaged KV store, with this layer's head segments populated.
    layer:
        Layer index to read from the page table.
    num_q_per_kv:
        GQA fanout (``Hq / Hkv``).
    sm_scale:
        Attention softmax scale.
    use_fused:
        Use the fused FA-3 varlen kernel when available; otherwise (or on
        CPU) fall back to the exact per-head reference. Both paths are
        numerically equivalent up to fp tolerance.

    Returns
    -------
    ``[Hq, Lq, D]`` attention output.
    """
    Hq, Lq, D = query.shape
    Hkv = cache.num_kv_heads
    if Hq != Hkv * num_q_per_kv:
        raise ValueError(
            f"segpaged_attention: Hq={Hq} != Hkv({Hkv}) * num_q_per_kv({num_q_per_kv})"
        )

    fused_ok = use_fused and is_fused_varlen_available() and query.is_cuda

    out = torch.zeros_like(query)

    if fused_ok:
        # ── Fused path: pack all (q-head, kv-head) pairs into one varlen
        # call. Each query head attends to its KV head's gathered view.
        q_parts: List[torch.Tensor] = []
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for kv_h in range(Hkv):
            k_view, v_view = cache.gather_head(layer, kv_h)  # [Lk, D]
            Lk = k_view.shape[0]
            for g in range(num_q_per_kv):
                qh = kv_h * num_q_per_kv + g
                q_parts.append(query[qh])  # [Lq, D]
                k_parts.append(k_view)
                v_parts.append(v_view)
                cu_q.append(cu_q[-1] + Lq)
                cu_k.append(cu_k[-1] + Lk)
                max_q = max(max_q, Lq)
                max_k = max(max_k, Lk)
        q_packed = torch.cat(q_parts, dim=0).unsqueeze(1)  # [sumLq, 1, D]
        k_packed = torch.cat(k_parts, dim=0).unsqueeze(1)
        v_packed = torch.cat(v_parts, dim=0).unsqueeze(1)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=query.device)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=query.device)
        packed_out = _fused_varlen_attention(
            q_packed, k_packed, v_packed, cu_q_t, cu_k_t, max_q, max_k, sm_scale
        )  # [sumLq, 1, D]
        packed_out = packed_out.squeeze(1)
        # Unpack back to [Hq, Lq, D].
        slot = 0
        for kv_h in range(Hkv):
            for g in range(num_q_per_kv):
                qh = kv_h * num_q_per_kv + g
                start = cu_q[slot]
                end = cu_q[slot + 1]
                out[qh] = packed_out[start:end]
                slot += 1
        return out

    # ── Reference path: Algorithm 2 head-by-head. ──
    for kv_h in range(Hkv):
        k_view, v_view = cache.gather_head(layer, kv_h)  # [Lk, D]
        for g in range(num_q_per_kv):
            qh = kv_h * num_q_per_kv + g
            out[qh] = _sdpa_one_head(query[qh], k_view, v_view, sm_scale)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Numerical-equivalence harness  (paper §5.4: cos > 0.99998)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def dense_reference_attention(
    query: torch.Tensor,  # [Hq, Lq, D]
    k_dense: torch.Tensor,  # [Hkv, L, D]
    v_dense: torch.Tensor,
    head_policy: List[str],
    *,
    sink: int,
    recent: int,
    num_q_per_kv: int,
    sm_scale: float,
) -> torch.Tensor:
    """Dense + additive-mask reference (the slow path SegPaged replaces).

    Encodes the same head-class sparsity as an ``attn_mask`` over the full
    dense ``[Hkv, L, D]`` KV, exactly like the baseline the paper measures.
    Used only to validate that SegPagedAttention is numerically equivalent.
    """
    Hkv, L, D = k_dense.shape
    Hq = query.shape[0]
    out = torch.zeros_like(query)
    for kv_h in range(Hkv):
        if head_policy[kv_h] == POLICY_GLOBAL:
            visible = torch.ones(L, dtype=torch.bool, device=query.device)
        else:
            idx = local_visible_indices(L, sink, recent, device=query.device)
            visible = torch.zeros(L, dtype=torch.bool, device=query.device)
            visible[idx] = True
        k_h = k_dense[kv_h].float()
        v_h = v_dense[kv_h].float()
        for g in range(num_q_per_kv):
            qh = kv_h * num_q_per_kv + g
            scores = (query[qh].float() @ k_h.transpose(0, 1)) * sm_scale  # [Lq, L]
            scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[qh] = (probs @ v_h).to(query.dtype)
    return out


@torch.no_grad()
def verify_against_dense(
    *,
    num_kv_heads: int,
    num_q_per_kv: int,
    head_dim: int,
    seq_len: int,
    sink: int,
    recent: int,
    global_ratio: float = 0.5,
    page_size: int = 64,
    q_len: int = 1,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Build random KV, run both paths, and report cosine + memory savings.

    Returns a dict with ``cosine`` (SegPaged vs dense reference),
    ``max_abs_err``, ``dense_kv_tokens``, ``segpaged_kv_tokens``, and
    ``kv_token_saving`` — the physical per-head capacity win.
    """
    device = device or torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    Hkv = num_kv_heads
    Hq = Hkv * num_q_per_kv
    L = seq_len

    k_dense = torch.randn(Hkv, L, head_dim, generator=gen, device=device, dtype=dtype)
    v_dense = torch.randn(Hkv, L, head_dim, generator=gen, device=device, dtype=dtype)
    query = torch.randn(Hq, q_len, head_dim, generator=gen, device=device, dtype=dtype)
    sm_scale = 1.0 / (head_dim**0.5)

    n_global = max(1, int(round(Hkv * global_ratio)))
    head_policy = [POLICY_GLOBAL if h < n_global else POLICY_LOCAL for h in range(Hkv)]

    cache = build_segpaged_cache(
        k_dense=k_dense,
        v_dense=v_dense,
        head_policy=head_policy,
        sink=sink,
        recent=recent,
        page_size=page_size,
        layer=0,
    )
    seg_out = segpaged_attention(
        query,
        cache,
        layer=0,
        num_q_per_kv=num_q_per_kv,
        sm_scale=sm_scale,
        use_fused=False,
    )
    ref_out = dense_reference_attention(
        query,
        k_dense,
        v_dense,
        head_policy,
        sink=sink,
        recent=recent,
        num_q_per_kv=num_q_per_kv,
        sm_scale=sm_scale,
    )

    cos = F.cosine_similarity(
        seg_out.float().flatten(), ref_out.float().flatten(), dim=0
    ).item()
    max_err = (seg_out.float() - ref_out.float()).abs().max().item()

    dense_tokens = Hkv * L
    segpaged_tokens = cache.stored_token_count()
    saving = 1.0 - (segpaged_tokens / dense_tokens) if dense_tokens else 0.0
    return {
        "cosine": cos,
        "max_abs_err": max_err,
        "dense_kv_tokens": dense_tokens,
        "segpaged_kv_tokens": segpaged_tokens,
        "kv_token_saving": saving,
        "n_global_heads": n_global,
        "n_local_heads": Hkv - n_global,
    }


__all__ = [
    "POLICY_GLOBAL",
    "POLICY_LOCAL",
    "HeadSegment",
    "SegmentPageTable",
    "SegPagedKVCache",
    "build_segpaged_cache",
    "local_visible_indices",
    "segpaged_attention",
    "dense_reference_attention",
    "verify_against_dense",
    "is_fused_varlen_available",
]
