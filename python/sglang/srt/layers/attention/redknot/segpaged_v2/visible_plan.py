# Copyright 2024-2026 SGLang RedKnot Integration.
"""Unified per-head visible-token plan for SegPaged v2.

The paper's central idea is that each KV head physically retains only the
tokens it actually needs. Different head classes express "what it needs"
differently:

- **global / dense / retrieval-keep-all**: the full ``[0, L)`` context.
- **local (streaming)**: a ``sink`` prefix plus a ``recent`` suffix window,
  i.e. ``[0, sink) ∪ [L - recent, L)``.
- **retrieval / custom**: an arbitrary set of token positions (e.g. a
  top-k / top-p selection, or any externally provided index set).

Rather than hard-coding ``global`` vs ``local`` everywhere (as the original
``segpaged.py`` does), SegPaged v2 funnels *every* head class through one
object: :class:`HeadVisiblePlan`. A plan is just "the ordered token
positions this head keeps". Storage, page tables, and the attention kernel
all consume positions and never branch on the policy that produced them.

This is the unified base substrate for future multi-head processing: adding
a new head class only means adding a new *constructor* of positions, not a
new code path in the storage or attention layers.

Everything here is plain PyTorch and runs on CPU; it is independent of the
existing ``segpaged.py`` module so current benchmarks are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

# Head visibility policies. ``CUSTOM`` covers retrieval / top-k / any
# externally supplied index set — the open extension point.
POLICY_GLOBAL = "global"
POLICY_LOCAL = "local"
POLICY_CUSTOM = "custom"


@dataclass
class HeadVisiblePlan:
    """The ordered token positions one ``(layer, head)`` physically keeps.

    Attributes
    ----------
    policy:
        Provenance tag (``global`` / ``local`` / ``custom``). Informational
        only — downstream code consumes :attr:`positions`, never the tag.
    positions:
        ``[kept]`` ``long`` tensor of token indices into the original
        ``[0, seq_len)`` sequence, in the order they are stored. Must be
        sorted ascending for causal correctness in the attention kernel.
    seq_len:
        The original (dense) sequence length the positions index into.
    """

    policy: str
    positions: torch.Tensor
    seq_len: int

    @property
    def kept(self) -> int:
        """Number of tokens this head keeps."""
        return int(self.positions.numel())

    def to(self, device: torch.device) -> "HeadVisiblePlan":
        if self.positions.device == device:
            return self
        return HeadVisiblePlan(
            policy=self.policy,
            positions=self.positions.to(device),
            seq_len=self.seq_len,
        )


# ──────────────────────────────────────────────────────────────────────────
# Constructors (one per head class). Adding a head class == adding one here.
# ──────────────────────────────────────────────────────────────────────────
def global_plan(
    seq_len: int, *, device: Optional[torch.device] = None
) -> HeadVisiblePlan:
    """Full-context plan: keep every token ``[0, seq_len)``."""
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    return HeadVisiblePlan(policy=POLICY_GLOBAL, positions=pos, seq_len=seq_len)


def local_plan(
    seq_len: int,
    sink: int,
    recent: int,
    *,
    device: Optional[torch.device] = None,
) -> HeadVisiblePlan:
    """Streaming plan: keep ``[0, sink) ∪ [seq_len - recent, seq_len)``.

    Overlapping sink/recent regions are merged so positions stay unique and
    sorted. When ``sink + recent >= seq_len`` this degrades gracefully to a
    full-context plan.
    """
    sink = max(0, min(sink, seq_len))
    recent = max(0, min(recent, seq_len))
    tail_start = max(sink, seq_len - recent)
    sink_idx = torch.arange(0, sink, dtype=torch.long, device=device)
    tail_idx = torch.arange(tail_start, seq_len, dtype=torch.long, device=device)
    if sink_idx.numel() == 0:
        pos = tail_idx
    elif tail_idx.numel() == 0:
        pos = sink_idx
    else:
        pos = torch.cat([sink_idx, tail_idx])
    return HeadVisiblePlan(policy=POLICY_LOCAL, positions=pos, seq_len=seq_len)


def custom_plan(
    seq_len: int,
    positions: Sequence[int] | torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> HeadVisiblePlan:
    """Arbitrary-position plan (retrieval / top-k / externally provided).

    Positions are de-duplicated, clamped to ``[0, seq_len)`` and sorted
    ascending so the stored KV stays in causal order.
    """
    if isinstance(positions, torch.Tensor):
        pos = positions.to(dtype=torch.long, device=device).flatten()
    else:
        pos = torch.tensor(list(positions), dtype=torch.long, device=device)
    if pos.numel() > 0:
        pos = pos[(pos >= 0) & (pos < seq_len)]
        pos = torch.unique(pos, sorted=True)
    return HeadVisiblePlan(policy=POLICY_CUSTOM, positions=pos, seq_len=seq_len)


__all__ = [
    "POLICY_GLOBAL",
    "POLICY_LOCAL",
    "POLICY_CUSTOM",
    "HeadVisiblePlan",
    "global_plan",
    "local_plan",
    "custom_plan",
]
