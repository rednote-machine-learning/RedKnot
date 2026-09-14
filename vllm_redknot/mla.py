"""DSV4 grouped output-projection math, without a serving integration claim.

MLA uses one shared latent KV stream. Its offline artifact can instead contain
the reusable heads' bias-free wo_a contribution *after inverse RoPE*. Ordinary
KV-head partitioning does not represent that contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ops import _floating_tensor, _head_ids, _positive_int

if TYPE_CHECKING:
    import torch


def project_head_contributions(
    attn: torch.Tensor,
    wo_a: torch.Tensor,
    head_ids: Sequence[int],
    total_heads: int,
) -> torch.Tensor:
    """Apply only the wo_a columns belonging to the supplied logical heads.

    Args:
        attn: Inverse-RoPE attention output [T, len(head_ids), D], in head_ids order.
        wo_a: Bias-free grouped projection [G, R, (total_heads / G) * D].
        head_ids: Head IDs within this projection's owned head domain [0, H).
        total_heads: Total heads owned by this wo_a tensor, not a TP-global count.

    Returns:
        Contributions [T, G, R]. Summing a disjoint complete head partition gives
        the full grouped projection, up to floating-point accumulation order.
        Quantized wo_a, its scales and inverse RoPE are deliberately not inferred.
    """
    import torch

    _positive_int(total_heads, "total_heads")
    _floating_tensor(attn, 3, "attn")
    _floating_tensor(wo_a, 3, "wo_a")
    heads = _head_ids(head_ids, total_heads, "head_ids")
    if attn.shape[1] != len(heads) or attn.shape[2] <= 0:
        raise ValueError("attention axes must match head_ids and a positive head_dim")
    groups, rank, width = wo_a.shape
    if groups <= 0 or rank <= 0 or total_heads % groups:
        raise ValueError(
            "wo_a groups must evenly divide total_heads and rank must be positive"
        )
    heads_per_group = total_heads // groups
    if width != heads_per_group * attn.shape[2]:
        raise ValueError("wo_a width must equal heads_per_group * head_dim")
    if attn.device != wo_a.device or attn.dtype != wo_a.dtype:
        raise ValueError("attention and wo_a must share dtype and device")
    output = attn.new_zeros((attn.shape[0], groups, rank))
    for group in range(groups):
        selected = [
            (axis, head % heads_per_group)
            for axis, head in enumerate(heads)
            if head // heads_per_group == group
        ]
        if not selected:
            continue
        axes = torch.tensor(
            [axis for axis, _ in selected], device=attn.device, dtype=torch.long
        )
        columns = torch.tensor(
            [column for _, column in selected], device=attn.device, dtype=torch.long
        )
        values = attn.index_select(1, axes).reshape(
            attn.shape[0], len(selected) * attn.shape[2]
        )
        weight = wo_a[group].reshape(rank, heads_per_group, attn.shape[2])
        weight = weight.index_select(1, columns).reshape(
            rank, len(selected) * attn.shape[2]
        )
        output[:, group] = values @ weight.transpose(0, 1)
    return output


def merge_mla_contributions(
    z_off: torch.Tensor,
    z_global: torch.Tensor,
    z_local_dirty: torch.Tensor,
    dirty_rows: torch.Tensor,
) -> torch.Tensor:
    """Replace offline contributions on dirty/query rows, then add global ones.

    z_off and z_global have shape [T, G, R]. z_local_dirty is compact [N_dirty,
    G, R], ordered exactly like unique dirty_rows. Every query/new row must be
    dirty. Clean rows use z_off + z_global; dirty rows use z_local_dirty +
    z_global. This function does not establish artifact/context compatibility.
    """
    import torch

    for value, name in (
        (z_off, "z_off"),
        (z_global, "z_global"),
        (z_local_dirty, "z_local_dirty"),
    ):
        _floating_tensor(value, 3, name)
    if z_off.shape != z_global.shape or z_local_dirty.shape[1:] != z_off.shape[1:]:
        raise ValueError("MLA contribution shapes must share [T, G, R] geometry")
    if any(
        value.device != z_off.device or value.dtype != z_off.dtype
        for value in (z_global, z_local_dirty)
    ):
        raise ValueError("MLA contributions must share dtype and device")
    if (
        not isinstance(dirty_rows, torch.Tensor)
        or dirty_rows.ndim != 1
        or dirty_rows.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("dirty_rows must be a rank-1 integer tensor")
    if dirty_rows.numel() != z_local_dirty.shape[0]:
        raise ValueError("dirty row count must match z_local_dirty")
    if bool(((dirty_rows < 0) | (dirty_rows >= z_off.shape[0])).any()):
        raise ValueError("dirty_rows must be within the output token axis")
    if torch.unique(dirty_rows).numel() != dirty_rows.numel():
        raise ValueError("dirty_rows must not contain duplicates")
    rows = dirty_rows.to(device=z_off.device, dtype=torch.long)
    local = z_off.clone()
    local.index_copy_(0, rows, z_local_dirty)
    return local + z_global


@dataclass(frozen=True)
class GroupedMLAProjector:
    """Bias-free grouped wo_a and a final wo_b callable, evaluated once per merge.

    The caller owns inverse RoPE, valid cache provenance, dirty-row dependency
    closure and TP collectives. This helper is not a vLLM DSV4 model backend.
    """

    wo_a: torch.Tensor
    wo_b: Callable
    total_heads: int

    def __post_init__(self) -> None:
        _positive_int(self.total_heads, "total_heads")
        _floating_tensor(self.wo_a, 3, "wo_a")
        if not callable(self.wo_b):
            raise TypeError("wo_b must be callable")
        groups, rank, width = self.wo_a.shape
        if groups <= 0 or rank <= 0 or width <= 0 or self.total_heads % groups:
            raise ValueError("wo_a geometry must be positive and divide total_heads")
        if width % (self.total_heads // groups):
            raise ValueError("wo_a width must contain complete head dimensions")

    def project(self, attn: torch.Tensor, head_ids: Sequence[int]) -> torch.Tensor:
        return project_head_contributions(attn, self.wo_a, head_ids, self.total_heads)

    def merge_and_project(
        self,
        z_off: torch.Tensor,
        z_global: torch.Tensor,
        z_local_dirty: torch.Tensor,
        dirty_rows: torch.Tensor,
    ):
        """Merge wo_a contributions and invoke wo_b once on [T, G * R]."""
        merged = merge_mla_contributions(z_off, z_global, z_local_dirty, dirty_rows)
        if merged.shape[1:] != self.wo_a.shape[:2]:
            raise ValueError("contributions do not match this projector's groups/rank")
        return self.wo_b(merged.flatten(1))
