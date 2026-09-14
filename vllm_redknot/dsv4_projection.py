"""DSV4 z_off reuse through the unmodified native output-projection callback.

The callback applies inverse RoPE, the native activation quantization and the
complete grouped wo_a, but replaces wo_b with identity. It returns the native
flattened [tokens, groups * rank] intermediate. We never slice/dequantize FP8
weights, change their scales or change the quantization recipe.

DSV4's 512-dimensional logical heads contain four independent 128-dimensional
activation-quantization blocks. Masking complete heads therefore preserves the
quantization of the remaining heads. Both masked projections still execute a
FULL-WIDTH wo_a GEMM: this helper claims no wo_a compute savings. Storing two
partial BF16 results and adding them introduces additional rounding versus one
native GEMM; exact bitwise native equivalence is not promised.

Cross-context/position document reuse remains an explicit model approximation,
not an implication of this linear decomposition. The caller must authenticate
model/weights/RoPE/quantization/TP provenance, mark every query/new row dirty,
and publish complete artifacts transactionally before skipping attention.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ops import _floating_tensor, _head_ids, _positions, _positive_int

if TYPE_CHECKING:
    import torch

NativeProjectZ = Callable[["torch.Tensor", "torch.Tensor"], "torch.Tensor"]
FinalProjection = Callable[["torch.Tensor"], "torch.Tensor"]


@dataclass(frozen=True)
class CachedLocalZ:
    """CPU local contribution and its TP-local projection/policy binding.

    policy_key must bind the caller's model, weights, RoPE, quantization recipe,
    TP shard and head policy. source_positions are provenance only: restore does
    not rotate or project z_off again. The tensors are owned copies, but callers
    must still treat their contents as immutable.
    """

    z_off: torch.Tensor
    source_positions: torch.Tensor
    local_head_ids: tuple[int, ...]
    num_heads: int
    head_dim: int
    policy_key: str

    @property
    def nbytes(self) -> int:
        """Logical artifact bytes, including source-position provenance."""
        return (
            self.z_off.numel() * self.z_off.element_size()
            + self.source_positions.numel() * self.source_positions.element_size()
        )


@dataclass(frozen=True)
class CachedContribution:
    """Map one chunk's cached rows to unique clean output rows.

    Both maps are rank-1 int32/int64 tensors on CPU or GPU. Output rows must not
    overlap another contribution; cache rows may repeat for an intentional
    repeated document. Row maps are indices, not RoPE positions.
    """

    cached: CachedLocalZ
    clean_rows: torch.Tensor
    cache_rows: torch.Tensor


def _policy_key(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("policy_key must be a nonempty compatibility identifier")


def _attention_geometry(o: torch.Tensor) -> None:
    _floating_tensor(o, 3, "attention output")
    if o.shape[0] <= 0 or o.shape[1] <= 0 or o.shape[2] != 512:
        raise ValueError("DSV4 attention output must have shape [T>0, H>0, 512]")


def _row_ids(rows: torch.Tensor, limit: int, name: str, *, unique: bool):
    import torch

    if (
        not isinstance(rows, torch.Tensor)
        or rows.ndim != 1
        or rows.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"{name} must be a rank-1 integer tensor")
    if bool(((rows < 0) | (rows >= limit)).any()):
        raise ValueError(f"{name} must index rows in [0, {limit})")
    if unique and torch.unique(rows).numel() != rows.numel():
        raise ValueError(f"{name} must not repeat output rows")
    return rows


def _project_z(
    o: torch.Tensor, positions: torch.Tensor, native_project_z: NativeProjectZ
) -> torch.Tensor:
    if not callable(native_project_z):
        raise TypeError("native_project_z must be callable")
    z = native_project_z(o, positions)
    _floating_tensor(z, 2, "native projected z")
    if z.shape[0] != o.shape[0] or z.shape[1] <= 0 or z.device != o.device:
        raise ValueError("native_project_z must return [T, G*R] on the input device")
    return z


# REDKNOT: RK-MLA-CAPTURE — cache the local-head low-rank contribution only.
def capture_local_z(
    o: torch.Tensor,
    source_positions: torch.Tensor,
    local_head_ids: Sequence[int],
    native_project_z: NativeProjectZ,
    *,
    policy_key: str,
) -> CachedLocalZ:
    """Mask global heads, project at source positions, and own a CPU artifact.

    o is [T, H, 512] BEFORE inverse RoPE, with only real TP-local heads (not
    FlashMLA padding). local_head_ids index this local head axis, including
    noncontiguous IDs. native_project_z performs inverse RoPE exactly once on
    this masked copy, then unmodified native grouped wo_a with wo_b=identity.
    Neither o nor source_positions is mutated, and the real wo_b is not called.
    """
    import torch

    _policy_key(policy_key)
    _attention_geometry(o)
    heads = _head_ids(local_head_ids, o.shape[1], "local_head_ids")
    if not heads:
        raise ValueError("capture requires at least one reusable local head")
    positions = _positions(
        source_positions, o.shape[0], o.device, "source_positions"
    ).to(dtype=torch.long)
    index = torch.tensor(heads, device=o.device, dtype=torch.long)
    masked = torch.zeros_like(o)
    masked.index_copy_(1, index, o.index_select(1, index))
    with torch.no_grad():
        z = _project_z(masked, positions, native_project_z)
    return CachedLocalZ(
        z_off=z.detach().to(device="cpu").clone(),
        source_positions=positions.detach().to(device="cpu").clone(),
        local_head_ids=heads,
        num_heads=o.shape[1],
        head_dim=o.shape[2],
        policy_key=policy_key,
    )


# REDKNOT: RK-MLA-MERGE — clean z_off + online contribution, then one wo_b.
def merge_cached_z_and_project(
    o: torch.Tensor,
    positions: torch.Tensor,
    contributions: Sequence[CachedContribution],
    native_project_z: NativeProjectZ,
    wo_b: FinalProjection,
    *,
    local_head_ids: Sequence[int],
    policy_key: str,
) -> torch.Tensor:
    """Project online heads, add z_off ONLY to clean rows, then call wo_b once.

    o is [T,H,512] before inverse RoPE. Clean local-head slots must be exactly
    zero because their attention was skipped; global slots and every dirty/query
    row contain freshly computed attention. Each contribution has equally sized
    clean_rows and cache_rows maps. Output rows must be disjoint across chunks;
    repeating cache rows is permitted for intentional repeated documents.

    All rows absent from every clean_rows map receive NO cached addition. The
    caller must include every query/new row in that complement. The native
    callback processes the ENTIRE o once, even with many chunk contributions.
    Cached source_positions never enter the online callback. No RoPE is applied
    to the cached projection. An empty contributions sequence runs all-online.

    Validation is correctness-first and can synchronize a GPU. This helper does
    not establish model/context compatibility from policy_key text alone.
    """
    import torch

    _policy_key(policy_key)
    _attention_geometry(o)
    heads = _head_ids(local_head_ids, o.shape[1], "local_head_ids")
    if not heads:
        raise ValueError("reuse policy requires at least one local head")
    current = _positions(positions, o.shape[0], o.device, "positions").to(
        dtype=torch.long
    )
    if not callable(native_project_z) or not callable(wo_b):
        raise TypeError("native_project_z and wo_b must be callable")
    head_index = torch.tensor(heads, device=o.device, dtype=torch.long)
    prepared = []
    seen_rows: set[int] = set()
    for contribution in contributions:
        if not isinstance(contribution, CachedContribution):
            raise TypeError("contributions must contain CachedContribution entries")
        cached = contribution.cached
        if not isinstance(cached, CachedLocalZ):
            raise TypeError("cached must be a CachedLocalZ artifact")
        _positive_int(cached.num_heads, "cached.num_heads")
        if (
            heads != cached.local_head_ids
            or o.shape[1:] != (cached.num_heads, cached.head_dim)
            or cached.policy_key != policy_key
        ):
            raise ValueError("cached projection head geometry or policy does not match")
        _floating_tensor(cached.z_off, 2, "cached.z_off")
        if cached.z_off.device.type != "cpu" or cached.z_off.shape[1] <= 0:
            raise ValueError("cached.z_off must be a CPU [T, G*R] artifact")
        source = _positions(
            cached.source_positions,
            cached.z_off.shape[0],
            "cpu",
            "cached.source_positions",
        )
        if cached.source_positions.device.type != "cpu" or source.numel() <= 0:
            raise ValueError(
                "cached source positions must be nonempty and CPU resident"
            )
        clean_rows = _row_ids(
            contribution.clean_rows, o.shape[0], "clean_rows", unique=True
        )
        cache_rows = _row_ids(
            contribution.cache_rows, cached.z_off.shape[0], "cache_rows", unique=False
        )
        if clean_rows.numel() != cache_rows.numel():
            raise ValueError("clean_rows and cache_rows must have the same row count")
        row_set = set(clean_rows.to(device="cpu").tolist())
        if seen_rows.intersection(row_set):
            raise ValueError("clean output rows overlap across cached contributions")
        seen_rows.update(row_set)
        clean = clean_rows.to(device=o.device, dtype=torch.long)
        if bool(o.index_select(0, clean).index_select(1, head_index).ne(0).any()):
            raise ValueError(
                "clean local-head output must be zero before cached addition"
            )
        prepared.append((cached, clean, cache_rows))
    z_online = _project_z(o, current, native_project_z)
    merged = z_online.clone()
    for cached, clean, cache_rows in prepared:
        if (
            cached.z_off.shape[1] != z_online.shape[1]
            or cached.z_off.dtype != z_online.dtype
        ):
            raise ValueError("cached and native z must share projected width and dtype")
        if not clean.numel():
            continue
        cached_rows = cached.z_off.index_select(
            0, cache_rows.to(device="cpu", dtype=torch.long)
        ).to(device=z_online.device)
        merged.index_copy_(0, clean, z_online.index_select(0, clean) + cached_rows)
    return wo_b(merged)
