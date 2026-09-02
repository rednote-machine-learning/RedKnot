# Copyright 2024-2026 SGLang RedKnot Integration.
"""DeepSeek-V4 offline three-layer KV reuse controller (v2).

DSV4 has three KV layers:
  SWA  – sliding window (last 128 tokens), ring buffer, nope_fp8+rope_bf16+scale
  C4   – 4x compressed full-sequence cache, same packed format, page_size=64
  C128 – 128x compressed full-sequence cache, same packed format, page_size=2
  Indexer – C4 top-k index (uint8 packed K + FP32 scales)

For a segment of L tokens prefilled at local positions [0, L):
  SWA:     ring buffer only keeps last min(L,128) tokens → snapshot those
  C4:      holds L//4 compressed tokens → snapshot all
  C128:    holds L//128 compressed tokens → snapshot all
  Indexer: holds L//4 index entries → snapshot all

On restore (segment at global offset G):
  SWA:     only last 128 tokens matter; boundary recompute handles this
  C4/C128: write packed KV back, then RoPE-relocate from local to global
           positions (compressed positions = original // ratio)
  Indexer: undo its post-RoPE Hadamard mixing, relocate RoPE, then reapply the
           transform and requantize the complete 128-dim key

Key insight: C4/C128 store nope_fp8+rope_bf16+scale in the SAME 584-byte
packed format as SWA, just with different page_size. So read_packed_kv /
write_packed_kv / reposition_rope work unchanged — only page_size differs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.attention.redknot.dsv4_rope_reloc import (
    read_packed_kv,
    read_rope_bf16,
    reposition_rope,
    write_packed_kv,
    write_rope_bf16,
)

logger = logging.getLogger(__name__)


@dataclass
class LayerCacheEntry:
    """Per-layer snapshot of packed KV for one pool (SWA, C4, or C128)."""

    packed: torch.Tensor  # [N, 584] uint8, CPU
    # We keep rope_local separately for efficient re-rotation (avoids
    # read-after-write from the packed buffer).
    rope_local: torch.Tensor  # [N, 64] bf16, CPU


@dataclass
class SegmentSnapshot:
    """Complete three-layer snapshot of a prefilled segment.

    After a segment of L tokens is prefilled at local positions [0, L),
    we capture:
      swa:     last min(L, swa_window) tokens from SWA pool
      c4:      L//4 compressed tokens from C4 pool (only for C4 layers)
      c128:    L//128 compressed tokens from C128 pool (only for C128 layers)
      indexer: L//4 index entries from indexer pool (only for C4 layers)
      compress_state: compressor ring-buffer state per layer
    """

    seg_hash: str
    length: int  # original segment length in tokens
    canonical_start_pos: int = 0

    # Per-layer caches, keyed by absolute model layer_id.
    # Only layers with the corresponding compress_ratio have entries.
    swa: Dict[int, LayerCacheEntry] = field(default_factory=dict)  # all layers
    c4: Dict[int, LayerCacheEntry] = field(default_factory=dict)  # ratio=4 layers
    c128: Dict[int, LayerCacheEntry] = field(default_factory=dict)  # ratio=128 layers

    # Indexer KV (C4 layers only): [N_c4, index_bytes] uint8
    indexer: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Compressor ring-buffer state: kv_score_buffer slice
    compress_state: Dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_compress_state: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Indexer hot-page selection captured during offline snapshot (the segment
    # was fully computed offline, so its indexer already knows which C4 pages are
    # high-value). Used at restore time to drive aggressive active-row selection:
    # tokens covered by these hot pages are recomputed online, the rest reuse
    # offline state. 1-D int32 tensor of *local* C4 page indices within the seg.
    indexer_hot_pages: Optional[torch.Tensor] = None
    # Per-unit selection frequency for request-level frac filtering: parallel
    # int32 tensors of (local C4 unit ordinal, #queries that selected it) plus
    # the total query count. Restore filters units with freq >= hot_frac * n_q.
    indexer_unit_ordinals: Optional[torch.Tensor] = None
    indexer_unit_freqs: Optional[torch.Tensor] = None
    indexer_num_queries: int = 0

    # Optional restart checkpoints for sparse, non-prefix replay.  Anchors are
    # local token positions inside the independently-prefilled segment.  A
    # checkpoint at ``anchor`` represents the carry state immediately before
    # token ``anchor``:
    #   * SWA keeps [anchor-128, anchor) for every target layer;
    #   * C4 attention/Indexer compressors keep their two terminal entries;
    #   * C128 keeps its one terminal entry.
    # The full C4/C128/Indexer caches above remain the source of truth for
    # skipped blocks.  These small artifacts only make it safe to restart an
    # online repair island without replaying the whole segment prefix.
    checkpoint_stride_tokens: int = 0
    swa_checkpoints: Dict[int, Dict[int, LayerCacheEntry]] = field(
        default_factory=dict
    )
    compress_state_checkpoints: Dict[int, Dict[int, torch.Tensor]] = field(
        default_factory=dict
    )
    indexer_compress_state_checkpoints: Dict[
        int, Dict[int, torch.Tensor]
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class RestoreReadiness:
    """Result of validating all offline artifacts before row pruning."""

    ready: bool
    reason: str = ""


def _compress_state_group_view(
    state_buffer: torch.Tensor, state_group_width: int
) -> torch.Tensor:
    """View the raw compressor ring as the groups addressed by metadata.

    ``compute_compressed_slots`` returns the first-dimension index consumed by
    the CUDA kernel *after* it views the raw pool as ``[-1, ratio, D]``.  Using
    that index directly on the original two-dimensional tensor copies only one
    raw row.  C4 needs four rows per group (and two adjacent groups for its
    overlap); offline C128 needs 128 rows per group.  Online-C128 uses width 1.
    """

    width = int(state_group_width)
    if width <= 0 or state_buffer.ndim != 2:
        raise ValueError("invalid compressor state group view")
    if int(state_buffer.shape[0]) % width != 0:
        raise ValueError(
            "compressor state rows are not divisible by the group width"
        )
    return state_buffer.view(-1, width, state_buffer.shape[-1])


def _jit_gather_slices_1d(
    values: torch.Tensor, ranges: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    """Concatenate 1-D CUDA slices through the active-architecture gather."""
    source_positions = torch.tensor(
        [position for begin, end in ranges for position in range(begin, end)],
        dtype=torch.int64,
    ).to(values.device)
    from sglang.jit_kernel.dsv4.attn import triton_index_select_rows

    return triton_index_select_rows(values, source_positions)


class DSV4OfflineReuseControllerV2:
    """Three-layer offline segment KV reuse for DSV4."""

    def __init__(self, swa_window: int = 128):
        self.swa_window = swa_window
        self._segments: Dict[str, SegmentSnapshot] = {}
        self._enabled = False
        self.stats = {
            "segments_cached": 0,
            "reuse_hits": 0,
            "tokens_reused_swa": 0,
            "tokens_reused_c4": 0,
            "tokens_reused_c128": 0,
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    @staticmethod
    def _checkpoint_anchors(length: int, stride_tokens: int) -> Tuple[int, ...]:
        length = int(length)
        stride_tokens = int(stride_tokens)
        if stride_tokens == 0:
            return ()
        if stride_tokens < 512 or stride_tokens % 512 != 0:
            raise ValueError(
                "checkpoint stride must be a positive multiple of 512 tokens"
            )
        if length <= 0 or length % 128 != 0:
            raise ValueError("checkpointed segments must be positive/128-aligned")
        return tuple(range(stride_tokens, length, stride_tokens))

    def _configure_checkpoint_stride(
        self, seg: SegmentSnapshot, stride_tokens: int
    ) -> Tuple[int, ...]:
        stride_tokens = int(stride_tokens)
        anchors = self._checkpoint_anchors(seg.length, stride_tokens)
        if seg.checkpoint_stride_tokens not in (0, stride_tokens):
            raise ValueError(
                f"segment {seg.seg_hash} checkpoint stride changed from "
                f"{seg.checkpoint_stride_tokens} to {stride_tokens}"
            )
        seg.checkpoint_stride_tokens = stride_tokens
        return anchors

    # ──────────────────────────────────────────────────────────────────
    # SNAPSHOT: capture all three layers after a segment prefill
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def snapshot_swa_layer(
        self,
        seg_hash: str,
        length: int,
        layer_id: int,
        kv_buffer: torch.Tensor,
        slot_indices: torch.Tensor,
        page_size: int,
    ) -> None:
        """Snapshot SWA packed KV for the tail of a freshly-prefilled segment."""
        seg = self._get_or_create(seg_hash, length)
        # SWA ring only keeps last swa_window tokens
        tail = min(length, self.swa_window)
        slots = slot_indices[-tail:]
        packed = read_packed_kv(kv_buffer, slots, page_size).detach().cpu()
        rope = read_rope_bf16(kv_buffer, slots, page_size).detach().cpu()
        seg.swa[layer_id] = LayerCacheEntry(packed=packed, rope_local=rope)

    @torch.no_grad()
    def snapshot_swa_checkpoints(
        self,
        seg_hash: str,
        length: int,
        layer_id: int,
        kv_buffer: torch.Tensor,
        slot_indices: torch.Tensor,
        page_size: int,
        checkpoint_stride_tokens: int,
    ) -> None:
        """Snapshot the 128-token SWA carry before every restart anchor."""

        seg = self._get_or_create(seg_hash, length)
        anchors = self._configure_checkpoint_stride(
            seg, checkpoint_stride_tokens
        )
        if not anchors:
            seg.swa_checkpoints[layer_id] = {}
            return
        if int(slot_indices.numel()) < length:
            raise ValueError("SWA checkpoint slots do not cover the segment")

        if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
            checkpoint_slots = _jit_gather_slices_1d(
                slot_indices,
                [
                    (anchor - self.swa_window, anchor)
                    for anchor in anchors
                ],
            )
        else:
            checkpoint_slots = torch.cat(
                [
                    slot_indices[anchor - self.swa_window : anchor]
                    for anchor in anchors
                ]
            )
        packed = read_packed_kv(
            kv_buffer, checkpoint_slots, page_size
        ).detach().cpu()
        rope = read_rope_bf16(
            kv_buffer, checkpoint_slots, page_size
        ).detach().cpu()
        entries: Dict[int, LayerCacheEntry] = {}
        cursor = 0
        for anchor in anchors:
            width = min(self.swa_window, anchor)
            entries[anchor] = LayerCacheEntry(
                packed=packed[cursor : cursor + width],
                rope_local=rope[cursor : cursor + width],
            )
            cursor += width
        seg.swa_checkpoints[layer_id] = entries

    @torch.no_grad()
    def snapshot_compress_checkpoints(
        self,
        seg_hash: str,
        length: int,
        layer_id: int,
        state_buffer: torch.Tensor,
        state_slots: torch.Tensor,
        compress_ratio: int,
        checkpoint_stride_tokens: int,
        is_indexer: bool = False,
        state_group_width: int = 1,
    ) -> None:
        """Snapshot compressor carry state at every 128-aligned anchor."""

        if is_indexer and compress_ratio != 4:
            raise ValueError("only the C4 path has an Indexer compressor")
        seg = self._get_or_create(seg_hash, length)
        anchors = self._configure_checkpoint_stride(
            seg, checkpoint_stride_tokens
        )
        expected_slots = length // compress_ratio
        if int(state_slots.numel()) < expected_slots:
            raise ValueError("compressor checkpoint slots do not cover the segment")

        anchor_slots = []
        widths = []
        for anchor in anchors:
            prefix_slots = state_slots[: anchor // compress_ratio]
            terminal = select_terminal_compress_state_slots(
                prefix_slots, compress_ratio
            )
            expected_width = 2 if compress_ratio == 4 else 1
            if int(terminal.numel()) != expected_width:
                raise ValueError(
                    f"checkpoint {anchor} has incomplete compressor history"
                )
            anchor_slots.append(terminal)
            widths.append(expected_width)

        destination = (
            seg.indexer_compress_state_checkpoints
            if is_indexer
            else seg.compress_state_checkpoints
        )
        if not anchor_slots:
            destination[layer_id] = {}
            return
        state_groups = _compress_state_group_view(
            state_buffer, state_group_width
        )
        if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
            from sglang.jit_kernel.dsv4.attn import triton_index_select_rows

            selected_slots = _jit_gather_slices_1d(
                state_slots,
                [
                    (
                        anchor // compress_ratio - width,
                        anchor // compress_ratio,
                    )
                    for anchor, width in zip(anchors, widths)
                ],
            )
            data = triton_index_select_rows(
                state_groups, selected_slots
            ).detach().cpu()
        else:
            selected_slots = torch.cat(anchor_slots)
            data = state_groups.index_select(
                0, selected_slots.to(torch.long)
            ).detach().cpu()
        checkpoints: Dict[int, torch.Tensor] = {}
        cursor = 0
        for anchor, width in zip(anchors, widths):
            checkpoints[anchor] = data[cursor : cursor + width]
            cursor += width
        destination[layer_id] = checkpoints

    @torch.no_grad()
    def store_indexer_hot_pages(
        self, seg_hash: str, length: int, hot_pages: torch.Tensor
    ) -> None:
        """Store the offline indexer hot-page selection for a segment.

        Called during snapshot: the segment is fully computed at local positions
        so its C4 indexer already knows the high-value pages. Stored as local C4
        page indices; restore maps them to global token rows.
        """
        seg = self._get_or_create(seg_hash, length)
        if seg.indexer_hot_pages is None and hot_pages is not None:
            seg.indexer_hot_pages = hot_pages.detach().cpu()

    @torch.no_grad()
    def store_indexer_unit_freqs(
        self,
        seg_hash: str,
        length: int,
        unit_ordinals: torch.Tensor,
        unit_freqs: torch.Tensor,
        num_queries: int,
    ) -> None:
        """Store per-C4-unit selection frequencies for request-level frac filter.

        Restore chooses hot units on the fly via ``freq >= hot_frac * n_q``,
        letting a single server sweep multiple frac values without re-snapshot.
        """
        seg = self._get_or_create(seg_hash, length)
        if seg.indexer_unit_ordinals is None:
            seg.indexer_unit_ordinals = unit_ordinals.detach().cpu()
            seg.indexer_unit_freqs = unit_freqs.detach().cpu()
            seg.indexer_num_queries = int(num_queries)

    def get_indexer_hot_pages(self, seg_hash: str) -> Optional[torch.Tensor]:
        seg = self._segments.get(seg_hash)
        return seg.indexer_hot_pages if seg is not None else None

    def get_indexer_hot_units_by_frac(
        self, seg_hash: str, hot_frac: float
    ) -> Optional[torch.Tensor]:
        """Return local C4 unit ordinals with freq >= hot_frac * num_queries.

        Falls back to the pre-filtered ``indexer_hot_pages`` if frequency data
        was not stored.
        """
        seg = self._segments.get(seg_hash)
        if seg is None:
            return None
        if seg.indexer_unit_ordinals is None or seg.indexer_num_queries <= 0:
            return seg.indexer_hot_pages
        kmin = max(1, int(seg.indexer_num_queries * float(hot_frac)))
        mask = seg.indexer_unit_freqs >= kmin
        return seg.indexer_unit_ordinals[mask]

    def get_indexer_unit_scores(
        self, seg_hash: str, *, min_exposure: int = 128
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return local C4 units and causally-normalized offline salience.

        Raw selection counts favor early units because more later tokens can
        attend to them.  Dividing by the number of causally-visible queries
        makes units from different positions and segments comparable for one
        request-level global budget.  The floor avoids amplifying the noisy tail.
        """

        seg = self._segments.get(seg_hash)
        if (
            seg is None
            or seg.indexer_unit_ordinals is None
            or seg.indexer_unit_freqs is None
            or seg.indexer_num_queries <= 0
        ):
            return None
        ordinals = seg.indexer_unit_ordinals.to(torch.int64)
        exposure = seg.indexer_num_queries - 4 * (ordinals + 1)
        exposure = torch.clamp(exposure, min=max(1, int(min_exposure)))
        scores = seg.indexer_unit_freqs.to(torch.float32) / exposure.to(
            torch.float32
        )
        return ordinals, scores

    def validate_restore_segments(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        expected_c4_layers: Optional[int] = None,
        expected_c128_layers: Optional[int] = None,
        expected_swa_layer_ids: Optional[Sequence[int]] = None,
        expected_c4_layer_ids: Optional[Sequence[int]] = None,
        expected_c128_layer_ids: Optional[Sequence[int]] = None,
        expected_state_group_widths: Optional[Mapping[int, int]] = None,
        require_indexer_units: bool = False,
        required_checkpoint_stride: Optional[int] = None,
        require_swa_checkpoints: bool = False,
    ) -> RestoreReadiness:
        """Validate every cache hit before the model runner mutates its batch.

        A partial hit is unsafe for selected-row execution: skipped token slots
        would remain uninitialized and a layer-local "dense fallback" could not
        reconstruct them.  Validation is deliberately strict and all-or-nothing.
        """

        if not segments:
            return RestoreReadiness(False, "restore plan has no segments")

        def tensor_signature(tensor: torch.Tensor) -> tuple:
            return (tuple(tensor.shape), tensor.dtype, tensor.device.type)

        def validate_layer_entry(
            entry: LayerCacheEntry,
            *,
            expected_tokens: int,
            label: str,
        ) -> Optional[str]:
            if not isinstance(entry, LayerCacheEntry):
                return f"{label} is not a LayerCacheEntry"
            if not isinstance(entry.packed, torch.Tensor) or not isinstance(
                entry.rope_local, torch.Tensor
            ):
                return f"{label} does not contain tensors"
            if (
                entry.packed.shape != (expected_tokens, 584)
                or entry.packed.dtype != torch.uint8
                or entry.packed.device.type != "cpu"
            ):
                return (
                    f"{label} packed artifact is invalid: "
                    f"shape={tuple(entry.packed.shape)} dtype={entry.packed.dtype} "
                    f"device={entry.packed.device.type}"
                )
            if (
                entry.rope_local.shape != (expected_tokens, 64)
                or entry.rope_local.dtype != torch.bfloat16
                or entry.rope_local.device.type != "cpu"
            ):
                return (
                    f"{label} RoPE artifact is invalid: "
                    f"shape={tuple(entry.rope_local.shape)} "
                    f"dtype={entry.rope_local.dtype} "
                    f"device={entry.rope_local.device.type}"
                )
            return None

        def validate_terminal_state(
            tensor: object,
            *,
            label: str,
            expected_entries: int,
            expected_group_widths: Tuple[int, ...],
        ) -> Optional[str]:
            if not isinstance(tensor, torch.Tensor):
                return f"{label} is not a tensor"
            if (
                tensor.ndim != 3
                or tensor.shape[0] != expected_entries
                or int(tensor.shape[1]) not in expected_group_widths
                or tensor.numel() == 0
            ):
                return (
                    f"{label} has invalid shape {tuple(tensor.shape)}; "
                    f"expected {expected_entries} terminal entries with group "
                    f"width in {expected_group_widths}"
                )
            if tensor.device.type != "cpu":
                return f"{label} must be stored on CPU"
            if tensor.dtype != torch.float32:
                return f"{label} must use float32 compressor state"
            return None

        reference_signature = None
        for index, metadata in enumerate(segments):
            try:
                seg_hash = str(metadata["seg_hash"])
                expected_length = int(metadata["length"])
                expected_canonical_start = int(
                    metadata.get("canonical_start_pos", 0)
                )
            except (KeyError, TypeError, ValueError) as error:
                return RestoreReadiness(
                    False, f"segment {index} has invalid cache metadata: {error}"
                )
            if expected_canonical_start != 0:
                return RestoreReadiness(
                    False,
                    f"segment {index} canonical position must be 0 for relocation",
                )
            seg = self._segments.get(seg_hash)
            if seg is None:
                return RestoreReadiness(False, f"segment {index} cache miss: {seg_hash}")
            if seg.length != expected_length:
                return RestoreReadiness(
                    False,
                    f"segment {index} length mismatch: cached={seg.length} "
                    f"requested={expected_length}",
                )
            if seg.canonical_start_pos != expected_canonical_start:
                return RestoreReadiness(
                    False,
                    f"segment {index} canonical position mismatch: "
                    f"cached={seg.canonical_start_pos} "
                    f"requested={expected_canonical_start}",
                )
            checkpoint_anchors: Tuple[int, ...] = ()
            if required_checkpoint_stride is not None:
                try:
                    checkpoint_anchors = self._checkpoint_anchors(
                        seg.length, int(required_checkpoint_stride)
                    )
                except ValueError as error:
                    return RestoreReadiness(False, str(error))
                if seg.checkpoint_stride_tokens != int(
                    required_checkpoint_stride
                ):
                    return RestoreReadiness(
                        False,
                        f"segment {index} checkpoint stride mismatch: "
                        f"cached={seg.checkpoint_stride_tokens} "
                        f"requested={required_checkpoint_stride}",
                    )
            if not seg.c4 or not seg.c128:
                return RestoreReadiness(
                    False, f"segment {index} has incomplete C4/C128 state"
                )
            if expected_c4_layers is not None and len(seg.c4) != expected_c4_layers:
                return RestoreReadiness(
                    False,
                    f"segment {index} C4 layer count mismatch: "
                    f"cached={len(seg.c4)} expected={expected_c4_layers}",
                )
            if expected_swa_layer_ids is not None and set(seg.swa) != set(
                expected_swa_layer_ids
            ):
                return RestoreReadiness(
                    False, f"segment {index} SWA layer set mismatch"
                )
            if expected_c4_layer_ids is not None and set(seg.c4) != set(
                expected_c4_layer_ids
            ):
                return RestoreReadiness(
                    False, f"segment {index} C4 layer set mismatch"
                )
            if expected_c128_layer_ids is not None and set(seg.c128) != set(
                expected_c128_layer_ids
            ):
                return RestoreReadiness(
                    False, f"segment {index} C128 layer set mismatch"
                )
            if (
                expected_c128_layers is not None
                and len(seg.c128) != expected_c128_layers
            ):
                return RestoreReadiness(
                    False,
                    f"segment {index} C128 layer count mismatch: "
                    f"cached={len(seg.c128)} expected={expected_c128_layers}",
                )
            if set(seg.indexer) != set(seg.c4):
                return RestoreReadiness(
                    False, f"segment {index} indexer/C4 layer set mismatch"
                )
            compressed_layers = set(seg.c4) | set(seg.c128)
            if expected_state_group_widths is not None and set(
                expected_state_group_widths
            ) != compressed_layers:
                return RestoreReadiness(
                    False,
                    f"segment {index} runtime compressor layout layer set mismatch",
                )
            if set(seg.compress_state) != compressed_layers:
                return RestoreReadiness(
                    False,
                    f"segment {index} compressor state layer set is incomplete "
                    "or contains unexpected entries",
                )
            if set(seg.indexer_compress_state) != set(seg.c4):
                return RestoreReadiness(
                    False,
                    f"segment {index} indexer compressor state layer set is "
                    "incomplete or contains unexpected entries",
                )
            expected_swa_tokens = min(seg.length, self.swa_window)
            for layer_id, entry in seg.swa.items():
                artifact_error = validate_layer_entry(
                    entry,
                    expected_tokens=expected_swa_tokens,
                    label=f"segment {index} SWA layer {layer_id}",
                )
                if artifact_error is not None:
                    return RestoreReadiness(
                        False,
                        artifact_error,
                    )
            expected_c4_tokens = seg.length // 4
            for layer_id, entry in seg.c4.items():
                indexer = seg.indexer.get(layer_id)
                artifact_error = validate_layer_entry(
                    entry,
                    expected_tokens=expected_c4_tokens,
                    label=f"segment {index} C4 layer {layer_id}",
                )
                if artifact_error is not None:
                    return RestoreReadiness(
                        False,
                        artifact_error,
                    )
                if (
                    not isinstance(indexer, torch.Tensor)
                    or indexer.shape != (expected_c4_tokens, 132)
                    or indexer.dtype != torch.uint8
                    or indexer.device.type != "cpu"
                ):
                    return RestoreReadiness(
                        False,
                        f"segment {index} indexer layer {layer_id} artifact is invalid",
                    )
            expected_c128_tokens = seg.length // 128
            for layer_id, entry in seg.c128.items():
                artifact_error = validate_layer_entry(
                    entry,
                    expected_tokens=expected_c128_tokens,
                    label=f"segment {index} C128 layer {layer_id}",
                )
                if artifact_error is not None:
                    return RestoreReadiness(
                        False,
                        artifact_error,
                    )
            for layer_id, state in seg.compress_state.items():
                # C4 is an overlapping eight-token window stored in two
                # four-token pages.  Continuing at the next segment reads both
                # the terminal page and its predecessor.  C128 is non-overlap
                # and reads only one terminal state entry.
                expected_entries = (
                    min(2, expected_c4_tokens)
                    if layer_id in seg.c4
                    else min(1, expected_c128_tokens)
                )
                artifact_error = validate_terminal_state(
                    state,
                    label=f"segment {index} compressor state layer {layer_id}",
                    expected_entries=expected_entries,
                    expected_group_widths=(
                        (int(expected_state_group_widths[layer_id]),)
                        if expected_state_group_widths is not None
                        else ((4,) if layer_id in seg.c4 else (1, 128))
                    ),
                )
                if artifact_error is not None:
                    return RestoreReadiness(False, artifact_error)
            for layer_id, state in seg.indexer_compress_state.items():
                artifact_error = validate_terminal_state(
                    state,
                    label=(
                        f"segment {index} indexer compressor state layer {layer_id}"
                    ),
                    expected_entries=min(2, expected_c4_tokens),
                    expected_group_widths=(4,),
                )
                if artifact_error is not None:
                    return RestoreReadiness(False, artifact_error)
            if checkpoint_anchors:
                expected_anchor_set = set(checkpoint_anchors)
                compressed_layers = set(seg.c4) | set(seg.c128)
                if set(seg.compress_state_checkpoints) != compressed_layers:
                    return RestoreReadiness(
                        False,
                        f"segment {index} checkpoint compressor layer set is "
                        "incomplete or contains unexpected entries",
                    )
                if set(seg.indexer_compress_state_checkpoints) != set(seg.c4):
                    return RestoreReadiness(
                        False,
                        f"segment {index} checkpoint Indexer layer set is "
                        "incomplete or contains unexpected entries",
                    )
                for layer_id, checkpoints in seg.compress_state_checkpoints.items():
                    if set(checkpoints) != expected_anchor_set:
                        return RestoreReadiness(
                            False,
                            f"segment {index} compressor checkpoint anchors are "
                            f"incomplete for layer {layer_id}",
                        )
                    expected_entries = 2 if layer_id in seg.c4 else 1
                    for anchor, state in checkpoints.items():
                        artifact_error = validate_terminal_state(
                            state,
                            label=(
                                f"segment {index} compressor checkpoint "
                                f"layer {layer_id} anchor {anchor}"
                            ),
                            expected_entries=expected_entries,
                            expected_group_widths=(
                                (int(expected_state_group_widths[layer_id]),)
                                if expected_state_group_widths is not None
                                else ((4,) if layer_id in seg.c4 else (1, 128))
                            ),
                        )
                        if artifact_error is not None:
                            return RestoreReadiness(False, artifact_error)
                for (
                    layer_id,
                    checkpoints,
                ) in seg.indexer_compress_state_checkpoints.items():
                    if set(checkpoints) != expected_anchor_set:
                        return RestoreReadiness(
                            False,
                            f"segment {index} Indexer checkpoint anchors are "
                            f"incomplete for layer {layer_id}",
                        )
                    for anchor, state in checkpoints.items():
                        artifact_error = validate_terminal_state(
                            state,
                            label=(
                                f"segment {index} Indexer checkpoint "
                                f"layer {layer_id} anchor {anchor}"
                            ),
                            expected_entries=2,
                            expected_group_widths=(4,),
                        )
                        if artifact_error is not None:
                            return RestoreReadiness(False, artifact_error)
                if require_swa_checkpoints:
                    expected_swa_layers = set(
                        expected_swa_layer_ids
                        if expected_swa_layer_ids is not None
                        else seg.swa
                    )
                    if set(seg.swa_checkpoints) != expected_swa_layers:
                        return RestoreReadiness(
                            False,
                            f"segment {index} SWA checkpoint layer set mismatch",
                        )
                    for layer_id, checkpoints in seg.swa_checkpoints.items():
                        if set(checkpoints) != expected_anchor_set:
                            return RestoreReadiness(
                                False,
                                f"segment {index} SWA checkpoint anchors are "
                                f"incomplete for layer {layer_id}",
                            )
                        for anchor, entry in checkpoints.items():
                            artifact_error = validate_layer_entry(
                                entry,
                                expected_tokens=self.swa_window,
                                label=(
                                    f"segment {index} SWA checkpoint "
                                    f"layer {layer_id} anchor {anchor}"
                                ),
                            )
                            if artifact_error is not None:
                                return RestoreReadiness(False, artifact_error)
            if require_indexer_units:
                ordinals = seg.indexer_unit_ordinals
                frequencies = seg.indexer_unit_freqs
                integer_dtypes = {
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                }
                if (
                    not isinstance(ordinals, torch.Tensor)
                    or not isinstance(frequencies, torch.Tensor)
                    or ordinals.ndim != 1
                    or frequencies.ndim != 1
                    or ordinals.shape != frequencies.shape
                    or ordinals.numel() == 0
                    or ordinals.dtype not in integer_dtypes
                    or frequencies.dtype not in integer_dtypes
                    or ordinals.device.type != "cpu"
                    or frequencies.device.type != "cpu"
                    or seg.indexer_num_queries <= 0
                ):
                    return RestoreReadiness(
                        False, f"segment {index} has invalid indexer frequency artifact"
                    )
                ordinals_i64 = ordinals.to(torch.int64)
                frequencies_i64 = frequencies.to(torch.int64)
                if (
                    bool((ordinals_i64 < 0).any().item())
                    or bool((ordinals_i64 >= expected_c4_tokens).any().item())
                    or torch.unique(ordinals_i64).numel() != ordinals_i64.numel()
                    or bool((frequencies_i64 <= 0).any().item())
                    or bool(
                        (frequencies_i64 > seg.indexer_num_queries).any().item()
                    )
                ):
                    return RestoreReadiness(
                        False, f"segment {index} has invalid indexer frequencies"
                    )
            signature = (
                frozenset(seg.c4),
                frozenset(seg.c128),
                frozenset(seg.indexer),
                seg.checkpoint_stride_tokens,
                tuple(
                    sorted(
                        (layer_id, tensor_signature(state))
                        for layer_id, state in seg.compress_state.items()
                    )
                ),
                tuple(
                    sorted(
                        (layer_id, tensor_signature(state))
                        for layer_id, state in seg.indexer_compress_state.items()
                    )
                ),
                tuple(
                    sorted(
                        (
                            layer_id,
                            tuple(
                                sorted(
                                    (anchor, tensor_signature(state))
                                    for anchor, state in checkpoints.items()
                                )
                            ),
                        )
                        for layer_id, checkpoints in (
                            seg.compress_state_checkpoints.items()
                        )
                    )
                ),
                tuple(
                    sorted(
                        (
                            layer_id,
                            tuple(
                                sorted(
                                    (anchor, tensor_signature(state))
                                    for anchor, state in checkpoints.items()
                                )
                            ),
                        )
                        for layer_id, checkpoints in (
                            seg.indexer_compress_state_checkpoints.items()
                        )
                    )
                ),
            )
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                return RestoreReadiness(
                    False, f"segment {index} layer signature differs from segment 0"
                )
        return RestoreReadiness(True)

    @torch.no_grad()
    def snapshot_c4_layer(
        self,
        seg_hash: str,
        length: int,
        layer_id: int,
        c4_buffer: torch.Tensor,
        c4_slots: torch.Tensor,
        c4_page_size: int,
        indexer_buffer: Optional[torch.Tensor] = None,
        indexer_slots: Optional[torch.Tensor] = None,
        indexer_page_size: Optional[int] = None,
    ) -> None:
        """Snapshot C4 compressed KV and optionally indexer state."""
        seg = self._get_or_create(seg_hash, length)
        n_c4 = length // 4
        if n_c4 == 0:
            return
        slots = c4_slots[:n_c4]
        packed = read_packed_kv(c4_buffer, slots, c4_page_size).detach().cpu()
        rope = read_rope_bf16(c4_buffer, slots, c4_page_size).detach().cpu()
        seg.c4[layer_id] = LayerCacheEntry(packed=packed, rope_local=rope)

        # Indexer buffer has a different format; save the raw paged bytes.
        if indexer_buffer is not None and indexer_slots is not None:
            # indexer buffer: [num_pages, page_bytes] uint8
            # We save a flat copy of the relevant slots' data.
            seg.indexer[layer_id] = _read_indexer_packed(
                indexer_buffer, indexer_slots[:n_c4], indexer_page_size or c4_page_size
            ).detach().cpu()

    @torch.no_grad()
    def snapshot_c128_layer(
        self,
        seg_hash: str,
        length: int,
        layer_id: int,
        c128_buffer: torch.Tensor,
        c128_slots: torch.Tensor,
        c128_page_size: int,
    ) -> None:
        """Snapshot C128 compressed KV."""
        seg = self._get_or_create(seg_hash, length)
        n_c128 = length // 128
        if n_c128 == 0:
            return
        slots = c128_slots[:n_c128]
        packed = read_packed_kv(c128_buffer, slots, c128_page_size).detach().cpu()
        rope = read_rope_bf16(c128_buffer, slots, c128_page_size).detach().cpu()
        seg.c128[layer_id] = LayerCacheEntry(packed=packed, rope_local=rope)

    @torch.no_grad()
    def snapshot_compress_state(
        self,
        seg_hash: str,
        length: int,
        layer_id: int,
        state_buffer: torch.Tensor,
        state_slots: torch.Tensor,
        is_indexer: bool = False,
        state_group_width: int = 1,
    ) -> None:
        """Snapshot compressor ring-buffer state (kv_score_buffer) for one layer."""
        seg = self._get_or_create(seg_hash, length)
        state_groups = _compress_state_group_view(
            state_buffer, state_group_width
        )
        if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
            from sglang.jit_kernel.dsv4.attn import triton_index_select_rows

            data = triton_index_select_rows(
                state_groups, state_slots
            ).detach().cpu()
        else:
            data = state_groups.index_select(
                0, state_slots.to(torch.long)
            ).detach().cpu()
        if is_indexer:
            seg.indexer_compress_state[layer_id] = data
        else:
            seg.compress_state[layer_id] = data

    # ──────────────────────────────────────────────────────────────────
    # RESTORE: write cached KV back with RoPE relocation
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def restore_c4_layer(
        self,
        seg_hash: str,
        layer_id: int,
        c4_buffer: torch.Tensor,
        dst_slots: torch.Tensor,
        global_offset: int,
        freqs_cis: torch.Tensor,
        c4_page_size: int,
        indexer_buffer: Optional[torch.Tensor] = None,
        indexer_slots: Optional[torch.Tensor] = None,
        indexer_page_size: Optional[int] = None,
        skip_tokens: int = 0,
        restore_begin_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        restore_c4: bool = True,
        restore_indexer: bool = True,
    ) -> int:
        """Restore C4 interior blocks while preserving online boundary blocks."""
        seg = self._segments.get(seg_hash)
        if seg is None or layer_id not in seg.c4:
            return 0
        entry = seg.c4[layer_id]
        n = entry.packed.shape[0]
        if max_tokens is not None:
            n = min(n, max(0, int(max_tokens)) // 4)
        begin_tokens = max(0, int(skip_tokens))
        if restore_begin_tokens is not None:
            begin_tokens = max(begin_tokens, int(restore_begin_tokens))
        start = min(n, (begin_tokens + 3) // 4)
        if start >= n:
            return 0
        slots = dst_slots[start:n]

        if os.environ.get("REDKNOT_V4_COMPARE_STATE", "0") == "1" and layer_id == 2:
            online_packed = read_packed_kv(c4_buffer, slots, c4_page_size)
            offline_packed = entry.packed[start:n].to(c4_buffer.device)
            token_equal = (online_packed == offline_packed).all(dim=1).float().mean()
            byte_equal = (online_packed == offline_packed).float().mean()
            logger.warning(
                "RedKnot V4 C4 compare layer=%d blocks=%d token_equal=%.6f byte_equal=%.6f",
                layer_id,
                n - start,
                token_equal.item(),
                byte_equal.item(),
            )
            if (
                indexer_buffer is not None
                and indexer_slots is not None
                and layer_id in seg.indexer
            ):
                online_indexer = _read_indexer_packed(
                    indexer_buffer,
                    indexer_slots[start:n],
                    indexer_page_size or c4_page_size,
                )
                offline_indexer = seg.indexer[layer_id][start:n].to(
                    indexer_buffer.device
                )
                index_token_equal = (
                    (online_indexer == offline_indexer).all(dim=1).float().mean()
                )
                index_byte_equal = (
                    (online_indexer == offline_indexer).float().mean()
                )
                logger.warning(
                    "RedKnot V4 indexer compare layer=%d blocks=%d "
                    "token_equal=%.6f byte_equal=%.6f",
                    layer_id,
                    n - start,
                    index_token_equal.item(),
                    index_byte_equal.item(),
                )

        device = c4_buffer.device
        src_pos = torch.arange(start, n, device=device, dtype=torch.long) * 4
        dst_pos = src_pos + global_offset
        if restore_c4:
            # 1. Write nope+scale (position-independent) via full packed write.
            write_packed_kv(c4_buffer, slots, entry.packed[start:n], c4_page_size)

            # 2. Relocate the compressed C4 RoPE phase from local to global.
            rope_local = entry.rope_local[start:n].to(device)
            rope_global = reposition_rope(rope_local, src_pos, dst_pos, freqs_cis)
            write_rope_bf16(c4_buffer, slots, rope_global, c4_page_size)

        # 3. Restore Indexer K with RoPE relocation. The stored 128-dim FP8 key
        #    is post-Hadamard, so its RoPE64 values are mixed across every byte;
        #    ``_relocate_indexer_rope`` reverses/reapplies that transform around
        #    the phase change and computes a fresh finite quantization scale.
        if (
            restore_indexer
            and indexer_buffer is not None
            and indexer_slots is not None
            and layer_id in seg.indexer
        ):
            packed_idx = seg.indexer[layer_id][start:n]
            if global_offset != 0:
                packed_idx = _relocate_indexer_rope(
                    packed_idx, src_pos, dst_pos, freqs_cis
                )
            _write_indexer_packed(
                indexer_buffer,
                indexer_slots[start:n],
                packed_idx,
                indexer_page_size or c4_page_size,
            )

        return n - start

    @torch.no_grad()
    def restore_c128_layer(
        self,
        seg_hash: str,
        layer_id: int,
        c128_buffer: torch.Tensor,
        dst_slots: torch.Tensor,
        global_offset: int,
        freqs_cis: torch.Tensor,
        c128_page_size: int,
        skip_tokens: int = 0,
        restore_begin_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> int:
        """Restore HCA interior blocks while preserving online boundary blocks."""
        seg = self._segments.get(seg_hash)
        if seg is None or layer_id not in seg.c128:
            return 0
        entry = seg.c128[layer_id]
        n = entry.packed.shape[0]
        if max_tokens is not None:
            n = min(n, max(0, int(max_tokens)) // 128)
        begin_tokens = max(0, int(skip_tokens))
        if restore_begin_tokens is not None:
            begin_tokens = max(begin_tokens, int(restore_begin_tokens))
        start = min(n, (begin_tokens + 127) // 128)
        if start >= n:
            return 0
        slots = dst_slots[start:n]

        write_packed_kv(c128_buffer, slots, entry.packed[start:n], c128_page_size)

        device = c128_buffer.device
        src_pos = torch.arange(start, n, device=device, dtype=torch.long) * 128
        dst_pos = src_pos + global_offset
        rope_local = entry.rope_local[start:n].to(device)
        rope_global = reposition_rope(rope_local, src_pos, dst_pos, freqs_cis)
        write_rope_bf16(c128_buffer, slots, rope_global, c128_page_size)
        return n - start

    @torch.no_grad()
    def restore_swa_tail(
        self,
        seg_hash: str,
        layer_id: int,
        swa_buffer: torch.Tensor,
        dst_slots: torch.Tensor,
        global_offset: int,
        freqs_cis: torch.Tensor,
        swa_page_size: int,
    ) -> int:
        """Restore a segment's final SWA window at its online positions."""

        seg = self._segments.get(seg_hash)
        if seg is None or layer_id not in seg.swa:
            return 0
        entry = seg.swa[layer_id]
        tail = int(entry.packed.shape[0])
        if tail == 0 or dst_slots.shape[0] < tail:
            return 0
        slots = dst_slots[-tail:]
        write_packed_kv(swa_buffer, slots, entry.packed, swa_page_size)
        device = swa_buffer.device
        src_pos = torch.arange(
            seg.length - tail, seg.length, device=device, dtype=torch.long
        )
        dst_pos = src_pos + global_offset
        rope_global = reposition_rope(
            entry.rope_local.to(device), src_pos, dst_pos, freqs_cis
        )
        write_rope_bf16(swa_buffer, slots, rope_global, swa_page_size)
        return tail

    @torch.no_grad()
    def restore_swa_checkpoint(
        self,
        seg_hash: str,
        layer_id: int,
        checkpoint_anchor: int,
        swa_buffer: torch.Tensor,
        dst_segment_slots: torch.Tensor,
        global_offset: int,
        freqs_cis: torch.Tensor,
        swa_page_size: int,
    ) -> int:
        """Restore the SWA carry immediately preceding one repair island."""

        seg = self._segments.get(seg_hash)
        if seg is None:
            return 0
        anchor = int(checkpoint_anchor)
        entry = seg.swa_checkpoints.get(layer_id, {}).get(anchor)
        if entry is None:
            return 0
        tail = int(entry.packed.shape[0])
        if (
            tail <= 0
            or anchor < tail
            or int(dst_segment_slots.numel()) < anchor
        ):
            raise ValueError(
                f"invalid SWA checkpoint anchor={anchor} tail={tail}"
            )
        slots = dst_segment_slots[anchor - tail : anchor]
        write_packed_kv(swa_buffer, slots, entry.packed, swa_page_size)
        device = swa_buffer.device
        src_pos = torch.arange(
            anchor - tail, anchor, device=device, dtype=torch.long
        )
        dst_pos = src_pos + int(global_offset)
        rope_global = reposition_rope(
            entry.rope_local.to(device), src_pos, dst_pos, freqs_cis
        )
        write_rope_bf16(swa_buffer, slots, rope_global, swa_page_size)
        return tail

    @torch.no_grad()
    def restore_indexer_layer(
        self,
        seg_hash: str,
        layer_id: int,
        indexer_buffer: torch.Tensor,
        dst_slots: torch.Tensor,
        indexer_page_size: int,
        skip_tokens: int = 0,
        restore_begin_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> int:
        """Materialize CSA Indexer interior blocks before global Top-K scoring."""

        seg = self._segments.get(seg_hash)
        if seg is None or layer_id not in seg.indexer:
            return 0
        packed = seg.indexer[layer_id]
        n = packed.shape[0]
        if max_tokens is not None:
            n = min(n, max(0, int(max_tokens)) // 4)
        begin_tokens = max(0, int(skip_tokens))
        if restore_begin_tokens is not None:
            begin_tokens = max(begin_tokens, int(restore_begin_tokens))
        start = min(n, (begin_tokens + 3) // 4)
        if start >= n:
            return 0
        _write_indexer_packed(
            indexer_buffer,
            dst_slots[start:n],
            packed[start:n],
            indexer_page_size,
        )
        return n - start

    @torch.no_grad()
    def restore_compress_state(
        self,
        seg_hash: str,
        layer_id: int,
        state_buffer: torch.Tensor,
        dst_slots: torch.Tensor,
        is_indexer: bool = False,
        state_group_width: int = 1,
    ) -> int:
        """Restore compressor ring-buffer state for one layer."""
        seg = self._segments.get(seg_hash)
        if seg is None:
            return 0
        src = seg.indexer_compress_state if is_indexer else seg.compress_state
        if layer_id not in src:
            return 0
        data = src[layer_id].to(state_buffer.device)
        if data.shape[0] != dst_slots.shape[0]:
            raise ValueError(
                "compressor terminal-state depth mismatch: "
                f"cached={data.shape[0]} destination={dst_slots.shape[0]}"
            )
        state_groups = _compress_state_group_view(
            state_buffer, state_group_width
        )
        if tuple(data.shape[1:]) != tuple(state_groups.shape[1:]):
            raise ValueError(
                "compressor state group shape mismatch: "
                f"cached={tuple(data.shape[1:])} "
                f"destination={tuple(state_groups.shape[1:])}"
            )
        if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
            from sglang.jit_kernel.dsv4.attn import triton_index_copy_rows

            triton_index_copy_rows(state_groups, dst_slots, data)
        else:
            state_groups.index_copy_(0, dst_slots.to(torch.long), data)
        return int(data.shape[0])

    @torch.no_grad()
    def restore_compress_checkpoint(
        self,
        seg_hash: str,
        layer_id: int,
        checkpoint_anchor: int,
        state_buffer: torch.Tensor,
        dst_slots: torch.Tensor,
        is_indexer: bool = False,
        state_group_width: int = 1,
    ) -> int:
        """Restore compressor carry state before a disconnected online island."""

        seg = self._segments.get(seg_hash)
        if seg is None:
            return 0
        source = (
            seg.indexer_compress_state_checkpoints
            if is_indexer
            else seg.compress_state_checkpoints
        )
        data = source.get(layer_id, {}).get(int(checkpoint_anchor))
        if data is None:
            return 0
        data = data.to(state_buffer.device)
        if data.shape[0] != dst_slots.shape[0]:
            raise ValueError(
                "compressor checkpoint depth mismatch: "
                f"cached={data.shape[0]} destination={dst_slots.shape[0]}"
            )
        state_groups = _compress_state_group_view(
            state_buffer, state_group_width
        )
        if tuple(data.shape[1:]) != tuple(state_groups.shape[1:]):
            raise ValueError(
                "compressor checkpoint group shape mismatch: "
                f"cached={tuple(data.shape[1:])} "
                f"destination={tuple(state_groups.shape[1:])}"
            )
        if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
            from sglang.jit_kernel.dsv4.attn import triton_index_copy_rows

            triton_index_copy_rows(state_groups, dst_slots, data)
        else:
            state_groups.index_copy_(0, dst_slots.to(torch.long), data)
        return int(data.shape[0])

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    def has_segment(self, seg_hash: str) -> bool:
        return seg_hash in self._segments

    def get_segment(self, seg_hash: str) -> Optional[SegmentSnapshot]:
        return self._segments.get(seg_hash)

    def begin_snapshot(
        self,
        seg_hash: str,
        length: int,
        canonical_start_pos: int = 0,
        checkpoint_stride_tokens: int = 0,
    ) -> SegmentSnapshot:
        """Atomically replace stale artifacts before a new snapshot pass."""

        if int(length) <= 0:
            raise ValueError("snapshot segment length must be positive")
        if int(canonical_start_pos) != 0:
            raise ValueError(
                "DSV4 offline snapshots must use canonical_start_pos=0"
            )
        self._checkpoint_anchors(length, checkpoint_stride_tokens)
        existed = seg_hash in self._segments
        seg = SegmentSnapshot(
            seg_hash=seg_hash,
            length=length,
            canonical_start_pos=canonical_start_pos,
            checkpoint_stride_tokens=int(checkpoint_stride_tokens),
        )
        self._segments[seg_hash] = seg
        if not existed:
            self.stats["segments_cached"] += 1
        return seg

    def _get_or_create(
        self, seg_hash: str, length: int, canonical_start_pos: int = 0
    ) -> SegmentSnapshot:
        if int(length) <= 0:
            raise ValueError("snapshot segment length must be positive")
        if int(canonical_start_pos) != 0:
            raise ValueError(
                "DSV4 offline snapshots must use canonical_start_pos=0"
            )
        seg = self._segments.get(seg_hash)
        if seg is None:
            seg = SegmentSnapshot(
                seg_hash=seg_hash,
                length=length,
                canonical_start_pos=canonical_start_pos,
            )
            self._segments[seg_hash] = seg
            self.stats["segments_cached"] += 1
        elif (
            seg.length != length
            or seg.canonical_start_pos != canonical_start_pos
        ):
            raise ValueError(
                f"segment key collision for {seg_hash}: "
                f"cached=(length={seg.length}, canonical={seg.canonical_start_pos}), "
                f"new=(length={length}, canonical={canonical_start_pos})"
            )
        return seg

    def clear(self):
        self._segments.clear()
        self.stats = {k: 0 for k in self.stats}


# ──────────────────────────────────────────────────────────────────────
# C4/C128 slot computation from full token slots
# ──────────────────────────────────────────────────────────────────────
# Two methods:
#   1) compute_compressed_slots: from full_slots via SWA page→ring→state mapping
#      (mirrors compressor write path — used for compress_state_pool slots)
#   2) compute_paged_compressed_slots: from page_table via standard paging
#      (mirrors attention read path — used for C4/C128 KV pool slots)
#
# For snapshot/restore of C4/C128 *KV data*, use method 2 (paged).
# For compress_state_pool data, use method 1 (state_loc).


@torch.no_grad()
def compute_compressed_slots(
    full_slots: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_page_size: int,
    ring_size: int,
    compress_ratio: int,
    seq_len: int,
    state_group_width: Optional[int] = None,
) -> torch.Tensor:
    """Compute compress-state-pool slot indices from full token pool slots.

    This matches the compressor's `get_raw_loc` path:
        full → swa → swa_page → state_loc → state_loc // compress_ratio

    Used for compress_state_pool (kv_score_buffer), NOT for C4/C128 KV pool.
    Returns: int32 tensor of shape ``[seq_len // compress_ratio]``.
    """
    group_width = int(compress_ratio) if state_group_width is None else int(
        state_group_width
    )
    if group_width <= 0 or int(ring_size) % group_width != 0:
        raise ValueError("invalid compressor state group width")
    n = seq_len // compress_ratio
    if n == 0:
        return torch.empty(0, dtype=torch.int32, device=full_slots.device)
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_compute_compressed_slots

        return triton_compute_compressed_slots(
            full_slots,
            full_to_swa,
            seq_len,
            compress_ratio,
            swa_page_size,
            ring_size,
        )
    boundary_indices = (
        torch.arange(1, n + 1, device=full_slots.device, dtype=torch.long)
        * compress_ratio
        - 1
    )  # [cr-1, 2*cr-1, ..., n*cr-1]
    boundary_full = full_slots[boundary_indices]
    swa_loc = full_to_swa[boundary_full.to(torch.int64)].to(torch.int32)
    swa_pages = swa_loc // swa_page_size
    state_loc = swa_pages * ring_size + swa_loc % ring_size
    # This divisor must mirror create_paged_compressor_data/get_raw_loc and its
    # Triton prefill helper.  In particular, online-C128 still divides by 128
    # even though its physical state group width is one.  That layout collapses
    # multiple page-boundary states and is therefore rejected by checkpoint-
    # island restore before row pruning.
    return (state_loc // compress_ratio).to(torch.int32)


def select_terminal_compress_state_slots(
    state_slots: torch.Tensor, compress_ratio: int
) -> torch.Tensor:
    """Select all history entries required to continue a compressor.

    C4 uses an overlapping eight-token window backed by two four-token state
    pages, so both terminal entries must survive an offline boundary.  C128 is
    non-overlapping and needs only its final entry.  Short segments naturally
    return the history that exists.
    """

    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compress ratio {compress_ratio}")
    required = 2 if compress_ratio == 4 else 1
    return state_slots[-min(required, state_slots.numel()) :]


@torch.no_grad()
def compute_paged_compressed_slots(
    page_table: torch.Tensor,
    req_idx: int,
    seq_len: int,
    compress_ratio: int,
    compressed_page_size: int,
    token_offset: int = 0,
) -> torch.Tensor:
    """Compute C4/C128 KV pool slot indices using the page table.

    Mirrors the metadata kernel's paged index computation:
        slot = page_table[req, page_i] * compressed_page_size + offset_in_page

    Parameters
    ----------
    page_table : [max_reqs, max_pages] int32
        The request-to-page mapping table from core_attn_metadata.
    req_idx : int
        Index of the request in the batch.
    seq_len : int
        Number of tokens in the *segment* (not the full sequence).
    compress_ratio : int
        4 for C4, 128 for C128.
    compressed_page_size : int
        Page size of the compressed pool (64 for C4, 2 for C128).
    token_offset : int
        Global token offset of this segment within the full sequence.
        The compressed index start = token_offset // compress_ratio.

    Returns
    -------
    int32 tensor of shape ``[seq_len // compress_ratio]``
    """
    n = seq_len // compress_ratio
    if n == 0:
        return torch.empty(0, dtype=torch.int32, device=page_table.device)
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import (
            triton_compute_paged_compressed_slots,
        )

        return triton_compute_paged_compressed_slots(
            page_table,
            req_idx,
            seq_len,
            compress_ratio,
            compressed_page_size,
            token_offset,
        )
    c_start = token_offset // compress_ratio
    offsets = c_start + torch.arange(n, device=page_table.device, dtype=torch.long)
    page_idx = offsets // compressed_page_size
    offset_in_page = offsets % compressed_page_size
    page_vals = page_table[req_idx, page_idx.to(torch.long)]
    return (page_vals * compressed_page_size + offset_in_page).to(torch.int32)


# ──────────────────────────────────────────────────────────────────────
# Indexer buffer read/write helpers
# ──────────────────────────────────────────────────────────────────────
# The indexer buffer has a different layout from the KV pools:
#   [num_pages, page_bytes] uint8
# where page_bytes = page_size * index_head_dim + page_size * scales_per_token * 4
# We save and restore the full page-resident data per-slot.
# For simplicity, we read/write entire contiguous page slices.

@torch.no_grad()
def _read_indexer_packed(
    buf: torch.Tensor, loc: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Read indexer entries for given slots as flat bytes per slot."""
    # Ensure uint8 view — buffer may be returned as float8 view
    if buf.dtype != torch.uint8:
        buf = buf.view(torch.uint8)
    num_pages, page_bytes = buf.shape
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_read_packed_kv_u8

        return triton_read_packed_kv_u8(
            buf, loc, page_size, page_bytes, 128, 4
        )
    # index_head_dim = 128 typically, scales = 4 bytes, total = 132 bytes/slot
    # But the actual per-slot layout within a page is more complex.
    # Safe approach: save entire pages that contain our slots.
    flat = buf.flatten()
    n = loc.shape[0]
    loc = loc.to(torch.int64)
    page_idx = loc // page_size
    token_off = loc % page_size
    # K part: page_idx * page_bytes + token_off * 128
    # Scale part: page_idx * page_bytes + page_size * 128 + token_off * 4
    # Total per slot = 132 bytes
    INDEX_HEAD_DIM = 128
    SCALE_BYTES = 4
    SLOT_BYTES = INDEX_HEAD_DIM + SCALE_BYTES
    k_base = page_idx * page_bytes + token_off * INDEX_HEAD_DIM
    s_base = page_idx * page_bytes + page_size * INDEX_HEAD_DIM + token_off * SCALE_BYTES
    k_idx = k_base[:, None] + torch.arange(INDEX_HEAD_DIM, device=buf.device)
    s_idx = s_base[:, None] + torch.arange(SCALE_BYTES, device=buf.device)
    return torch.cat((flat[k_idx], flat[s_idx]), dim=1)  # [N, 132]


@torch.no_grad()
def _write_indexer_packed(
    buf: torch.Tensor,
    loc: torch.Tensor,
    packed: torch.Tensor,
    page_size: int,
) -> None:
    """Write indexer entries back to their slots."""
    # Ensure uint8 view — buffer may be returned as float8 view
    if buf.dtype != torch.uint8:
        buf = buf.view(torch.uint8)
    num_pages, page_bytes = buf.shape
    if os.environ.get("SGLANG_USE_JIT_PACKED_OFFSETS", "0") == "1":
        from sglang.jit_kernel.dsv4.attn import triton_write_packed_kv_u8

        packed = packed.to(device=buf.device, dtype=torch.uint8).contiguous()
        triton_write_packed_kv_u8(
            buf, loc, packed, page_size, page_bytes, 128, 4
        )
        return
    flat = buf.flatten()
    n = loc.shape[0]
    loc = loc.to(torch.int64)
    page_idx = loc // page_size
    token_off = loc % page_size
    INDEX_HEAD_DIM = 128
    SCALE_BYTES = 4
    k_base = page_idx * page_bytes + token_off * INDEX_HEAD_DIM
    s_base = page_idx * page_bytes + page_size * INDEX_HEAD_DIM + token_off * SCALE_BYTES
    k_idx = k_base[:, None] + torch.arange(INDEX_HEAD_DIM, device=buf.device)
    s_idx = s_base[:, None] + torch.arange(SCALE_BYTES, device=buf.device)
    packed = packed.to(device=buf.device, dtype=torch.uint8)
    flat[k_idx] = packed[:, :INDEX_HEAD_DIM]
    flat[s_idx] = packed[:, INDEX_HEAD_DIM:]


def _indexer_rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply DSV4's normalized Hadamard transform to an Indexer activation."""

    # Keep the import lazy: dsa_indexer pulls in the GPU Indexer kernels, while
    # this controller is also imported by CPU-only planning/unit-test paths.
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

    return rotate_activation(x)


def _quantize_indexer_activation(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize one 128-dim Indexer row with its own FP32 scale."""

    # This is the same quantizer used by the normal Indexer cache-write path.
    # In particular, it recomputes amax and clamps to the finite e4m3 range.
    from sglang.srt.layers.attention.dsa.triton_kernel import act_quant

    return act_quant(x)


@torch.no_grad()
def _relocate_indexer_rope(
    packed: torch.Tensor,
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Re-locate a post-Hadamard, FP8 Indexer K to its global position.

    ``packed`` is ``[N, 132]`` uint8: bytes ``[:128]`` are the 128-dim FP8
    e4m3 Indexer K and bytes ``[128:132]`` are its FP32 per-token scale.

    The Indexer compressor applies RoPE to the final 64 dimensions *before* a
    normalized 128-dim Hadamard transform.  Consequently, every stored FP8
    dimension mixes RoPE and non-RoPE values; treating stored bytes 64:128 as
    the RoPE subspace is incorrect.  The normalized Hadamard transform is its
    own inverse, so relocation must do:

      dequant(all 128) -> H^-1 -> relocate(last 64) -> H -> requant(all 128)

    Requantization deliberately computes a new scale.  A pairwise RoPE
    rotation can increase the largest component even though it preserves the
    vector norm; retaining the old scale can exceed e4m3's finite range and
    encode NaNs.
    """
    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed Indexer artifact must be a tensor")
    if packed.ndim != 2 or packed.shape[1] != 132:
        raise ValueError(
            f"packed Indexer artifact must have shape [N, 132], got {tuple(packed.shape)}"
        )
    if packed.dtype != torch.uint8:
        raise TypeError(
            f"packed Indexer artifact must use uint8 storage, got {packed.dtype}"
        )
    if freqs_cis is None:
        raise ValueError("Indexer relocation requires a RoPE frequency table")

    n = packed.shape[0]
    if src_pos.numel() != n or dst_pos.numel() != n:
        raise ValueError(
            "Indexer relocation positions must match packed rows: "
            f"rows={n} src={src_pos.numel()} dst={dst_pos.numel()}"
        )

    device = freqs_cis.device
    if n == 0:
        return packed.to(device=device)

    # Snapshot artifacts normally live on CPU. Validate their cheap scalar
    # metadata before transferring the 128-byte key payload to the GPU.
    source_scale = (
        packed[:, 128:132].contiguous().view(torch.float32).reshape(n)
    )
    if not bool(
        (torch.isfinite(source_scale) & (source_scale > 0)).all().item()
    ):
        raise ValueError("packed Indexer artifact contains a non-finite scale")

    work = packed.to(device=device)
    k_u8 = work[:, :128].contiguous()
    scale = source_scale.to(device=device, dtype=torch.float32)

    # The cache contains H([nope64, RoPE(local, rope64)]), not a directly
    # splittable [nope64, rope64] vector. Undo H over all 128 dimensions first.
    indexer_fp8 = k_u8.view(torch.float8_e4m3fn)
    indexer_dequant = indexer_fp8.to(torch.float32) * scale[:, None]
    pre_hadamard = _indexer_rotate_activation(indexer_dequant.contiguous())

    rope_reloc = reposition_rope(
        pre_hadamard[:, 64:128].contiguous().to(torch.bfloat16),
        src_pos.to(device=device, dtype=torch.long),
        dst_pos.to(device=device, dtype=torch.long),
        freqs_cis,
    ).to(dtype=pre_hadamard.dtype)
    relocated_pre_hadamard = torch.cat(
        (pre_hadamard[:, :64], rope_reloc), dim=-1
    ).contiguous()
    relocated_indexer = _indexer_rotate_activation(relocated_pre_hadamard)

    # Quantize the entire mixed vector with a fresh scale. This mirrors the
    # normal online Indexer store and guarantees clamping before the FP8 cast.
    relocated_fp8, relocated_scale = _quantize_indexer_activation(
        relocated_indexer.contiguous()
    )
    if relocated_fp8.shape != (n, 128) or relocated_scale.numel() != n:
        raise ValueError(
            "Indexer quantizer returned an invalid shape: "
            f"key={tuple(relocated_fp8.shape)} scale={tuple(relocated_scale.shape)}"
        )
    if os.environ.get("REDKNOT_V4_VALIDATE_INDEXER_RELOCATION", "0") == "1":
        finite = torch.isfinite(relocated_fp8.to(torch.float32)).all()
        finite &= torch.isfinite(relocated_scale).all()
        finite &= (relocated_scale > 0).all()
        if not bool(finite.item()):
            raise ValueError("Indexer relocation produced a non-finite artifact")

    # Keep the result on the destination/cache device. _write_indexer_packed
    # accepts a device-local packed tensor, avoiding a GPU->CPU->GPU round trip.
    out = torch.empty((n, 132), dtype=torch.uint8, device=device)
    out[:, :128] = relocated_fp8.contiguous().view(torch.uint8)
    out[:, 128:132] = (
        relocated_scale.to(torch.float32).reshape(n, 1).contiguous().view(torch.uint8)
    )
    return out


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ──────────────────────────────────────────────────────────────────────
_CONTROLLER_V2: Optional[DSV4OfflineReuseControllerV2] = None


def get_offline_reuse_controller_v2() -> DSV4OfflineReuseControllerV2:
    global _CONTROLLER_V2
    if _CONTROLLER_V2 is None:
        _CONTROLLER_V2 = DSV4OfflineReuseControllerV2()
    return _CONTROLLER_V2
