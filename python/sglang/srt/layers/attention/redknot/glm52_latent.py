from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_GLM52_H0_SNAPSHOTS: dict[str, "GLM52H0Snapshot"] = {}


@dataclass(frozen=True)
class GLM52LatentLayout:
    num_layers: int
    ckv_dim: int
    kpe_dim: int
    page_tokens: int
    index_topk: int
    full_indexer_layer_ids: tuple[int, ...]
    shared_indexer_layer_ids: tuple[int, ...]
    dense_layer_ids: tuple[int, ...]

    @property
    def latent_dim(self) -> int:
        return self.ckv_dim + self.kpe_dim

    @classmethod
    def from_config(cls, config, page_tokens: int = 64) -> "GLM52LatentLayout":
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if "GlmMoeDsaForCausalLM" not in architectures:
            raise ValueError(f"Unsupported GLM architecture: {architectures}")
        if page_tokens != 64:
            raise ValueError("GLM-5.2 DSA currently requires 64-token pages")

        num_layers = int(config.num_hidden_layers)
        indexer_types = tuple(config.indexer_types)
        mlp_layer_types = tuple(config.mlp_layer_types)
        if len(indexer_types) != num_layers or len(mlp_layer_types) != num_layers:
            raise ValueError("GLM layer metadata must match num_hidden_layers")

        return cls(
            num_layers=num_layers,
            ckv_dim=int(config.kv_lora_rank),
            kpe_dim=int(config.qk_rope_head_dim),
            page_tokens=page_tokens,
            index_topk=int(config.index_topk),
            full_indexer_layer_ids=tuple(
                index for index, kind in enumerate(indexer_types) if kind == "full"
            ),
            shared_indexer_layer_ids=tuple(
                index for index, kind in enumerate(indexer_types) if kind == "shared"
            ),
            dense_layer_ids=tuple(
                index for index, kind in enumerate(mlp_layer_types) if kind == "dense"
            ),
        )

    def split_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {latent.shape[-1]}"
            )
        return latent[..., : self.ckv_dim], latent[..., self.ckv_dim :]


@dataclass(frozen=True)
class GLM52H0Snapshot:
    length: int
    canonical_start_pos: int
    page_tokens: int
    start_layer: int
    end_layer: int
    payload: dict[str, Any]


def _validate_page_slots(slot_indices: torch.Tensor, page_tokens: int) -> None:
    if slot_indices.ndim != 1 or slot_indices.numel() == 0:
        raise ValueError("slot_indices must be a non-empty 1-D tensor")
    slots = slot_indices.detach().to(device="cpu", dtype=torch.int64)
    for start in range(0, slots.numel(), page_tokens):
        page = slots[start : start + page_tokens]
        if int(page[0]) % page_tokens != 0:
            raise ValueError(
                "Each H0 cache span must start at a physical page boundary"
            )
        expected = torch.arange(int(page[0]), int(page[0]) + page.numel())
        if not torch.equal(page, expected):
            raise ValueError(
                "H0 cache slots must be contiguous within each physical page"
            )


def capture_glm52_h0(
    pool,
    slot_indices: torch.Tensor,
    *,
    canonical_start_pos: int = 0,
) -> GLM52H0Snapshot:
    """Capture same-position MLA latent and DSA index pages from a live pool."""
    page_tokens = int(pool.page_size)
    if page_tokens != 64:
        raise ValueError(f"GLM-5.2 H0 requires page_size=64, got {page_tokens}")
    if canonical_start_pos != 0:
        raise ValueError("H0 snapshots currently require canonical_start_pos=0")
    _validate_page_slots(slot_indices, page_tokens)
    payload = pool.get_cpu_copy(slot_indices)
    if not isinstance(payload, dict) or set(payload) != {"kv", "index_k"}:
        raise TypeError("Expected DSATokenToKVPool payload with kv and index_k")
    start_layer = int(pool.start_layer or 0)
    end_layer = int(pool.end_layer or (start_layer + pool.layer_num))
    return GLM52H0Snapshot(
        length=int(slot_indices.numel()),
        canonical_start_pos=canonical_start_pos,
        page_tokens=page_tokens,
        start_layer=start_layer,
        end_layer=end_layer,
        payload=payload,
    )


def restore_glm52_h0(
    pool,
    slot_indices: torch.Tensor,
    snapshot: GLM52H0Snapshot,
    *,
    destination_start_pos: int = 0,
) -> None:
    """Restore an H0 snapshot into new physical slots at identical positions."""
    if destination_start_pos != snapshot.canonical_start_pos:
        raise ValueError("H0 restore cannot change logical token positions")
    if int(pool.page_size) != snapshot.page_tokens:
        raise ValueError("Snapshot and destination pool page sizes differ")
    if int(slot_indices.numel()) != snapshot.length:
        raise ValueError("Snapshot and destination lengths differ")
    start_layer = int(pool.start_layer or 0)
    end_layer = int(pool.end_layer or (start_layer + pool.layer_num))
    if (start_layer, end_layer) != (snapshot.start_layer, snapshot.end_layer):
        raise ValueError("Snapshot and destination layer ranges differ")
    _validate_page_slots(slot_indices, snapshot.page_tokens)
    pool.load_cpu_copy(snapshot.payload, slot_indices)


def clear_glm52_h0_snapshots() -> None:
    _GLM52_H0_SNAPSHOTS.clear()


def _payload_byte_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("kv", "index_k"):
        if len(left[key]) != len(right[key]):
            return False
        for left_layer, right_layer in zip(left[key], right[key]):
            if len(left_layer) != len(right_layer):
                return False
            if any(
                not torch.equal(left_chunk, right_chunk)
                for left_chunk, right_chunk in zip(left_layer, right_layer)
            ):
                return False
    return True


def process_glm52_h0_before_forward(forward_batch, pool, req_to_token_pool) -> None:
    plans = getattr(forward_batch, "redknot_reuse_plan", None)
    if not plans:
        return
    req_to_token = req_to_token_pool.req_to_token
    for req_index, plan in enumerate(plans):
        if not isinstance(plan, dict) or plan.get("mode") != "glm52_h0_restore":
            continue
        snapshot_id = str(plan["snapshot_id"])
        snapshot = _GLM52_H0_SNAPSHOTS.get(snapshot_id)
        if snapshot is None:
            raise RuntimeError(f"GLM H0 snapshot not found: {snapshot_id}")
        prefix_len = int(plan["prefix_len"])
        if prefix_len != snapshot.length:
            raise ValueError("GLM H0 restore prefix length does not match snapshot")
        req_pool_index = int(forward_batch.req_pool_indices[req_index])
        slots = req_to_token[req_pool_index, :prefix_len].to(torch.int64)
        restore_glm52_h0(pool, slots, snapshot)
        if plan.get("verify_byte_parity"):
            restored_payload = pool.get_cpu_copy(slots)
            if not _payload_byte_equal(snapshot.payload, restored_payload):
                raise RuntimeError("GLM H0 live cache differs after restore")


def process_glm52_h0_after_forward(forward_batch, pool, req_to_token_pool) -> None:
    plans = getattr(forward_batch, "redknot_reuse_plan", None)
    if not plans or not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
        return
    req_to_token = req_to_token_pool.req_to_token
    for req_index, plan in enumerate(plans):
        if not isinstance(plan, dict) or plan.get("mode") != "glm52_h0_snapshot":
            continue
        snapshot_id = str(plan["snapshot_id"])
        prefix_len = int(plan["prefix_len"])
        req_pool_index = int(forward_batch.req_pool_indices[req_index])
        slots = req_to_token[req_pool_index, :prefix_len].to(torch.int64)
        _GLM52_H0_SNAPSHOTS[snapshot_id] = capture_glm52_h0(pool, slots)


@torch.no_grad()
def reposition_interleaved_kpe(
    kpe: torch.Tensor,
    src_positions: torch.Tensor,
    dst_positions: torch.Tensor,
    rope_theta: float,
) -> torch.Tensor:
    """Move already-rotated GLM kPE vectors between absolute positions."""
    if kpe.shape[-1] % 2:
        raise ValueError("Interleaved RoPE requires an even kPE dimension")
    if (
        kpe.shape[-2] != src_positions.numel()
        or src_positions.shape != dst_positions.shape
    ):
        raise ValueError("Position tensors must match the kPE token dimension")

    device = kpe.device
    rotary_dim = kpe.shape[-1]
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    delta = (dst_positions - src_positions).to(device=device, dtype=torch.float32)
    phase = delta[:, None] * inv_freq[None, :]
    cos = phase.cos()
    sin = phase.sin()

    work = kpe.float().reshape(*kpe.shape[:-1], rotary_dim // 2, 2)
    even, odd = work.unbind(dim=-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(kpe.dtype)
