"""Transactional CPU control plane for position-independent DSV4 latent reuse.

This is deliberately independent from the serving hot path.  It defines the
artifact and row/block contracts that the GPU implementation must satisfy
before ``deepseek_v4.MQALayer._forward_prepare`` is allowed to omit work.

DeepSeek-V4 has 64 logical query heads but one physical latent KV stream.  A
FlashMLA SWA record is 584 bytes::

    448 bytes no-PE FP8 | 128 bytes inverse-RoPE BF16 | 8 bytes FP8 scales

The 64 BF16 values in the middle are stored after *removing* the capture
position's RoPE.  Consequently an artifact row has no source-position
semantics: restore applies RoPE exactly once at the destination position.  C4
and C128 records use the same packed representation at block granularity.

The controller stores every SWA row of a segment, never just its 128-token
tail.  A tail-only artifact is sufficient to seed a later query, but it cannot
serve online global-head attention for all document rows during prefill.

An artifact that claims compressor/indexer skipping is complete only when it
also contains:

* every C4/C128 packed block;
* every C4 Indexer key in a separately versioned position-independent format;
* every internal 512-token attention/Indexer compressor restart state;
* the terminal attention-compressor state; and
* for C4, the terminal Indexer-compressor state.

The SWA carry for an internal restart anchor is derived from the preceding 128
rows of the complete SWA artifact; it is therefore covered by the same atomic
commit rather than stored as a second, potentially inconsistent copy.

This module owns no CUDA tensors and performs no RoPE math.  GPU capture must
canonicalize packed records before calling :meth:`capture_swa_rows` or
:meth:`capture_compressed_blocks`; GPU restore must materialize destination
RoPE after consuming the immutable row descriptors returned here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple


SHARED_LATENT_FORMAT_VERSION = 1
PACKED_LATENT_BYTES = 584
PACKED_LATENT_POSITION_SEMANTICS = (
    "fp8_nope_plus_inverse_rope_bf16_plus_scales_v1"
)
INDEXER_POSITION_SEMANTICS = "pre_hadamard_positionless_indexer_key_v1"
TOKEN_HASH_SEMANTICS = "sha256_unsigned_u32_le_v1"
BOUNDARY128_REPLAY_TOKENS = 128
CONTEXT_BOUND_EXECUTION_PROFILE = (
    "pure_headsplit_context_bound_fullscope_3_37_3_v1"
)
INDEPENDENT_RELOCATION_EXECUTION_PROFILE = (
    "pure_headsplit_independent_rope_relocation_fullscope_"
    "boundary128_3_37_3_v1"
)
COMBINED_ROW_SPARSE_EXECUTION_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)
CHECKPOINT_REPLAY_STRIDE = 512
DSV4_0731_NUM_TARGET_LAYERS = 43
DSV4_0731_DENSE_PREFIX_LAYERS = 3
DSV4_0731_DENSE_SUFFIX_LAYERS = 3
DEFAULT_REQUIRED_LAYER_IDS = tuple(range(3, 40))
DSV4_0731_TARGET_COMPRESS_RATIOS = (
    (0, 0) + tuple(value for _ in range(20) for value in (4, 128)) + (4,)
)

if len(DSV4_0731_TARGET_COMPRESS_RATIOS) != DSV4_0731_NUM_TARGET_LAYERS:
    raise AssertionError("the DSV4-0731 target topology must contain 43 layers")
if (
    DSV4_0731_DENSE_PREFIX_LAYERS
    + len(DEFAULT_REQUIRED_LAYER_IDS)
    + DSV4_0731_DENSE_SUFFIX_LAYERS
    != DSV4_0731_NUM_TARGET_LAYERS
):
    raise AssertionError("the DSV4-0731 execution profile must be 3 + 37 + 3")


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _as_bytes(value: object, *, expected: int, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must expose a byte buffer")
    result = bytes(value)
    if len(result) != expected:
        raise ValueError(f"{name} has {len(result)} bytes; expected {expected}")
    return result


def _token_digest(token_ids: Sequence[int]) -> str:
    """Return the benchmark/runtime token identity digest.

    ``benchmark_RedKnot_DeepSeekV4_Flash_RAG.py::_ih_chunk_hash`` serializes
    every tokenizer id as one unsigned little-endian 32-bit value.  Using
    Python's more convenient 64-bit encoding here would produce a different
    digest for identical text and make every real restore fail preflight.
    """

    payload = bytearray()
    for token in token_ids:
        token = _strict_int(token, "token id")
        if token < 0 or token > 0xFFFFFFFF:
            raise ValueError("token ids must fit an unsigned 32-bit value")
        payload.extend(token.to_bytes(4, "little", signed=False))
    return "sha256:" + sha256(payload).hexdigest()


@dataclass(frozen=True)
class LayerComponentSpec:
    """Exact component geometry required to skip one layer's cache builders."""

    layer_id: int
    compress_ratio: int
    indexer_record_bytes: int = 0
    attention_terminal_state_bytes: int = 0
    indexer_terminal_state_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if type(self.compress_ratio) is not int or self.compress_ratio not in (
            0,
            4,
            128,
        ):
            raise ValueError("compress_ratio must be 0, 4, or 128")
        sizes = (
            self.indexer_record_bytes,
            self.attention_terminal_state_bytes,
            self.indexer_terminal_state_bytes,
        )
        if any(type(value) is not int or value < 0 for value in sizes):
            raise ValueError("component byte sizes must be non-negative integers")
        if self.compress_ratio == 0 and any(sizes):
            raise ValueError("SWA-only layers cannot declare compressed state")
        if self.compress_ratio == 128 and (
            self.indexer_record_bytes or self.indexer_terminal_state_bytes
        ):
            raise ValueError("C128 layers do not own a C4 Indexer")
        if self.compress_ratio == 4 and (
            self.indexer_record_bytes <= 0
            or self.attention_terminal_state_bytes <= 0
            or self.indexer_terminal_state_bytes <= 0
        ):
            raise ValueError(
                "C4 compressor skipping requires Indexer records and both "
                "terminal states"
            )
        if self.compress_ratio == 128 and self.attention_terminal_state_bytes <= 0:
            raise ValueError(
                "C128 compressor skipping requires terminal attention state"
            )


@dataclass(frozen=True)
class SharedLatentSpec:
    """Compatibility contract shared by capture and restore."""

    model_hash: str
    policy_hash: str
    length: int
    layers: Tuple[LayerComponentSpec, ...]
    required_layer_ids: Tuple[int, ...] = DEFAULT_REQUIRED_LAYER_IDS
    packed_record_bytes: int = PACKED_LATENT_BYTES
    packed_position_semantics: str = PACKED_LATENT_POSITION_SEMANTICS
    indexer_position_semantics: str = INDEXER_POSITION_SEMANTICS
    token_hash_semantics: str = TOKEN_HASH_SEMANTICS
    checkpoint_stride_tokens: int = CHECKPOINT_REPLAY_STRIDE
    format_version: int = SHARED_LATENT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_hash, str)
            or not self.model_hash
            or not isinstance(self.policy_hash, str)
            or not self.policy_hash
        ):
            raise ValueError("model and policy hashes must be non-empty")
        if (
            type(self.layers) is not tuple
            or type(self.required_layer_ids) is not tuple
        ):
            raise TypeError("layers and required_layer_ids must be immutable tuples")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("segment length must be a positive integer")
        if self.packed_record_bytes != PACKED_LATENT_BYTES:
            raise ValueError("DSV4 shared latent records must be exactly 584 bytes")
        if self.packed_position_semantics != PACKED_LATENT_POSITION_SEMANTICS:
            raise ValueError("packed latent position semantics are incompatible")
        if self.indexer_position_semantics != INDEXER_POSITION_SEMANTICS:
            raise ValueError("Indexer position semantics are incompatible")
        if self.token_hash_semantics != TOKEN_HASH_SEMANTICS:
            raise ValueError("token hash semantics are incompatible")
        if self.checkpoint_stride_tokens != CHECKPOINT_REPLAY_STRIDE:
            raise ValueError("boundary128-v2 checkpoints require a 512-token stride")
        if self.format_version != SHARED_LATENT_FORMAT_VERSION:
            raise ValueError("shared latent format version is incompatible")
        layer_ids = tuple(layer.layer_id for layer in self.layers)
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("layer component specs must have unique layer ids")
        required_layer_ids = tuple(
            _strict_int(layer_id, "required layer id")
            for layer_id in self.required_layer_ids
        )
        if required_layer_ids != tuple(sorted(set(required_layer_ids))):
            raise ValueError("required layer ids must be unique and sorted")
        if tuple(sorted(layer_ids)) != required_layer_ids:
            raise ValueError("component specs must cover every required layer exactly")
        for layer in self.layers:
            if layer.compress_ratio and self.length % layer.compress_ratio:
                raise ValueError(
                    f"segment length {self.length} is not aligned to layer "
                    f"{layer.layer_id}'s ratio {layer.compress_ratio}"
                )

    @property
    def layers_by_id(self) -> Mapping[int, LayerComponentSpec]:
        return MappingProxyType({layer.layer_id: layer for layer in self.layers})

    @property
    def checkpoint_anchors(self) -> Tuple[int, ...]:
        return tuple(
            range(
                self.checkpoint_stride_tokens,
                self.length,
                self.checkpoint_stride_tokens,
            )
        )


def build_dsv4_0731_shared_latent_spec(
    *,
    model_hash: str,
    policy_hash: str,
    length: int,
    c4_indexer_record_bytes: int,
    c4_attention_terminal_state_bytes: int,
    c4_indexer_terminal_state_bytes: int,
    c128_attention_terminal_state_bytes: int,
) -> SharedLatentSpec:
    """Build the exact 3 + 37 + 3 DSV4-0731 reuse contract.

    Layers 0..2 and 40..42 are deliberately absent: they run completely
    online.  The reusable middle is layers 3..39.  In the published 0731
    topology that middle contains 18 C4 and 19 C128 layers; deriving ratios
    from the complete 43-layer tuple avoids an off-by-one alternation bug.
    """

    widths = {
        "c4_indexer_record_bytes": c4_indexer_record_bytes,
        "c4_attention_terminal_state_bytes": (
            c4_attention_terminal_state_bytes
        ),
        "c4_indexer_terminal_state_bytes": c4_indexer_terminal_state_bytes,
        "c128_attention_terminal_state_bytes": (
            c128_attention_terminal_state_bytes
        ),
    }
    for name, value in widths.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    layers = []
    for layer_id in DEFAULT_REQUIRED_LAYER_IDS:
        ratio = DSV4_0731_TARGET_COMPRESS_RATIOS[layer_id]
        if ratio == 4:
            layers.append(
                LayerComponentSpec(
                    layer_id=layer_id,
                    compress_ratio=ratio,
                    indexer_record_bytes=c4_indexer_record_bytes,
                    attention_terminal_state_bytes=(
                        c4_attention_terminal_state_bytes
                    ),
                    indexer_terminal_state_bytes=(
                        c4_indexer_terminal_state_bytes
                    ),
                )
            )
        elif ratio == 128:
            layers.append(
                LayerComponentSpec(
                    layer_id=layer_id,
                    compress_ratio=ratio,
                    attention_terminal_state_bytes=(
                        c128_attention_terminal_state_bytes
                    ),
                )
            )
        else:  # pragma: no cover - guarded by the immutable topology above.
            raise AssertionError("the reusable middle cannot contain an SWA-only layer")
    return SharedLatentSpec(
        model_hash=model_hash,
        policy_hash=policy_hash,
        length=length,
        layers=tuple(layers),
        required_layer_ids=DEFAULT_REQUIRED_LAYER_IDS,
    )


class _RowStore:
    """Fixed-width transactional rows with duplicate-write consistency checks."""

    __slots__ = ("count", "width", "_data", "_valid", "_valid_count")

    def __init__(self, count: int, width: int) -> None:
        if count <= 0 or width <= 0:
            raise ValueError("row-store count and width must be positive")
        self.count = int(count)
        self.width = int(width)
        self._data = bytearray(count * width)
        self._valid = bytearray(count)
        self._valid_count = 0

    @property
    def complete(self) -> bool:
        return self._valid_count == self.count

    def capture(self, rows: Sequence[int], payload: object, *, name: str) -> None:
        row_ids = tuple(_strict_int(row, f"{name} row") for row in rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError(f"{name} row ids must be unique within one capture")
        if any(row < 0 or row >= self.count for row in row_ids):
            raise ValueError(f"{name} row is outside the artifact")
        incoming = _as_bytes(
            payload,
            expected=len(row_ids) * self.width,
            name=f"{name} payload",
        )
        # Check every conflicting duplicate before publishing any new row from
        # this call.  Otherwise a late mismatch could leave an undocumented
        # prefix of the rejected batch staged.
        for payload_row, row in enumerate(row_ids):
            source_begin = payload_row * self.width
            source_end = source_begin + self.width
            target_begin = row * self.width
            target_end = target_begin + self.width
            new_value = incoming[source_begin:source_end]
            if self._valid[row]:
                if self._data[target_begin:target_end] != new_value:
                    raise ValueError(f"{name} row {row} changed within a generation")
        for payload_row, row in enumerate(row_ids):
            if self._valid[row]:
                continue
            source_begin = payload_row * self.width
            source_end = source_begin + self.width
            target_begin = row * self.width
            target_end = target_begin + self.width
            new_value = incoming[source_begin:source_end]
            self._data[target_begin:target_end] = new_value
            self._valid[row] = 1
            self._valid_count += 1

    def freeze(self, *, name: str) -> bytes:
        if not self.complete:
            missing = self.count - self._valid_count
            raise ValueError(f"{name} is incomplete: {missing} rows are missing")
        return bytes(self._data)


@dataclass
class _LayerStage:
    spec: LayerComponentSpec
    length: int
    swa: _RowStore = field(init=False)
    compressed: Optional[_RowStore] = field(init=False, default=None)
    indexer: Optional[_RowStore] = field(init=False, default=None)
    attention_terminal_state: Optional[bytes] = None
    indexer_terminal_state: Optional[bytes] = None
    attention_checkpoint_states: Dict[int, bytes] = field(default_factory=dict)
    indexer_checkpoint_states: Dict[int, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.swa = _RowStore(self.length, PACKED_LATENT_BYTES)
        ratio = self.spec.compress_ratio
        if ratio:
            self.compressed = _RowStore(
                self.length // ratio, PACKED_LATENT_BYTES
            )
        if ratio == 4:
            self.indexer = _RowStore(
                self.length // ratio, self.spec.indexer_record_bytes
            )


@dataclass
class _SegmentStage:
    seg_hash: str
    generation_id: str
    token_ids: Tuple[int, ...]
    token_hash: str
    spec: SharedLatentSpec
    layers: Dict[int, _LayerStage]


@dataclass(frozen=True)
class LayerArtifact:
    spec: LayerComponentSpec
    swa_positionless_packed: bytes
    compressed_positionless_packed: Optional[bytes]
    indexer_positionless_keys: Optional[bytes]
    attention_terminal_state: Optional[bytes]
    indexer_terminal_state: Optional[bytes]
    attention_checkpoint_states: Mapping[int, bytes]
    indexer_checkpoint_states: Mapping[int, bytes]


@dataclass(frozen=True)
class SharedLatentArtifact:
    seg_hash: str
    token_hash: str
    token_ids: Tuple[int, ...]
    spec: SharedLatentSpec
    layers: Mapping[int, LayerArtifact]
    commit_epoch: int


@dataclass(eq=False)
class SharedLatentPublishReceipt:
    """Rollback-capable publication of one complete CPU artifact epoch."""

    controller_token: object
    artifact: SharedLatentArtifact
    previous_artifact: Optional[SharedLatentArtifact]
    state: str = "published"


@dataclass(frozen=True)
class SegmentPlacement:
    seg_hash: str
    global_offset: int
    length: int
    token_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.seg_hash, str) or not self.seg_hash:
            raise ValueError("segment hash must be non-empty")
        if not isinstance(self.token_hash, str):
            raise TypeError("segment token hash must be a string")
        if type(self.global_offset) is not int or self.global_offset < 0:
            raise ValueError("segment global_offset must be non-negative")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("segment length must be positive")


@dataclass(frozen=True)
class RestoreRows:
    placement_index: int
    seg_hash: str
    output_rows: Tuple[int, ...]
    local_rows: Tuple[int, ...]


@dataclass(frozen=True)
class CheckpointReplayIsland:
    """Selector-compatible segment-local online replay interval."""

    segment_index: int
    token_begin: int
    token_end: int


@dataclass(frozen=True)
class Boundary128RowLayout:
    """Immutable row layout for legacy boundary128 or context-bound skip0.

    ``selected_prefix_tokens`` has the exact shape emitted by
    ``CheckpointReplayLayout.selected_prefix_tokens``.  ``checkpoint_islands``
    is the normalized form of that layout's ``restore_islands``.  This keeps
    the full-latent prototype independent of SGLang imports while making the
    interface mechanically testable.
    """

    placements: Tuple[SegmentPlacement, ...]
    positions: Tuple[int, ...]
    query_start: int
    clean_mask: Tuple[bool, ...]
    clean_rows: Tuple[RestoreRows, ...]
    dirty_output_rows: Tuple[int, ...]
    selected_prefix_tokens: Tuple[Tuple[int, int], ...]
    checkpoint_islands: Tuple[CheckpointReplayIsland, ...]
    protected_ranges: Tuple[Tuple[int, int], ...]
    online_local_ranges: Mapping[int, Tuple[Tuple[int, int], ...]]
    boundary_tokens: int = BOUNDARY128_REPLAY_TOKENS

    @property
    def clean_count(self) -> int:
        return sum(self.clean_mask)

    @property
    def dirty_document_count(self) -> int:
        return sum(
            not reusable
            for position, reusable in zip(self.positions, self.clean_mask)
            if position < self.query_start
        )


@dataclass(frozen=True)
class RestoreBlocks:
    placement_index: int
    seg_hash: str
    output_completion_rows: Tuple[int, ...]
    local_blocks: Tuple[int, ...]


@dataclass(frozen=True)
class CheckpointRestore:
    """State/SWA carry needed immediately before one online replay island."""

    placement_index: int
    seg_hash: str
    local_anchor: int
    global_anchor: int
    output_begin_row: int
    swa_carry_local_rows: Tuple[int, ...]
    needs_attention_state: bool
    needs_indexer_state: bool


@dataclass(frozen=True)
class LayerRestorePlan:
    layer_id: int
    compress_ratio: int
    compressed_blocks: Tuple[RestoreBlocks, ...]
    indexer_blocks: Tuple[RestoreBlocks, ...]
    checkpoint_restores: Tuple[CheckpointRestore, ...]
    terminal_state_placements: Tuple[int, ...]
    may_skip_compressor_for_clean_blocks: bool
    must_run_indexer_query_path: bool


@dataclass(frozen=True)
class SharedLatentRestorePlan:
    """Immutable CPU certificate consumed before any cache mutation.

    ``artifacts`` pins the exact immutable generations selected at preflight.
    A later controller replacement cannot make a consumer accidentally gather
    from the new generation while following row descriptors for the old one.
    """

    spec: SharedLatentSpec
    positions: Tuple[int, ...]
    query_start: int
    clean_mask: Tuple[bool, ...]
    clean_rows: Tuple[RestoreRows, ...]
    dirty_output_rows: Tuple[int, ...]
    layers: Mapping[int, LayerRestorePlan]
    artifacts: Mapping[str, SharedLatentArtifact]
    artifact_epochs: Mapping[str, int]
    selected_prefix_tokens: Tuple[Tuple[int, int], ...]
    checkpoint_islands: Tuple[CheckpointReplayIsland, ...]
    protected_ranges: Tuple[Tuple[int, int], ...]
    boundary_tokens: int
    execution_profile: str = ""

    @property
    def clean_count(self) -> int:
        return sum(self.clean_mask)

    @property
    def dirty_document_count(self) -> int:
        return sum(
            not reusable
            for position, reusable in zip(self.positions, self.clean_mask)
            if position < self.query_start
        )


def _ordered_tiled_placements(
    placements: Sequence[SegmentPlacement], query_start: int
) -> Tuple[SegmentPlacement, ...]:
    raw = tuple(placements)
    if any(not isinstance(item, SegmentPlacement) for item in raw):
        raise TypeError("restore placements must be SegmentPlacement values")
    ordered = tuple(sorted(raw, key=lambda item: item.global_offset))
    if not ordered:
        raise ValueError("restore requires at least one segment placement")
    previous_end = 0
    for placement in ordered:
        if placement.global_offset != previous_end:
            raise ValueError("segment placements must tile the document prefix")
        previous_end = placement.global_offset + placement.length
    if previous_end != query_start:
        raise ValueError("segment placements do not end at query_start")
    return ordered


def _selector_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"checkpoint island is missing {name}")
        return value[name]
    if not hasattr(value, name):
        raise TypeError(
            "checkpoint islands must expose segment_index/token_begin/token_end"
        )
    return getattr(value, name)


def _overlaps(
    begin: int, end: int, ranges: Sequence[Tuple[int, int]]
) -> bool:
    return any(
        begin < range_end and range_begin < end
        for range_begin, range_end in ranges
    )


def _build_row_layout(
    *,
    placements: Sequence[SegmentPlacement],
    positions: Sequence[int],
    query_start: int,
    selected_prefix_tokens: Sequence[Tuple[int, int]] = (),
    checkpoint_islands: Sequence[object] = (),
    protected_ranges: Sequence[object] = (),
    boundary_tokens: int,
    allow_sparse_positions: bool = False,
) -> Boundary128RowLayout:
    """Materialize one explicitly selected legacy or context-bound geometry."""

    query_start = _strict_int(query_start, "query_start")
    if query_start < 0:
        raise ValueError("query_start must be non-negative")
    pos = tuple(_strict_int(value, "position") for value in positions)
    if not pos:
        raise ValueError("restore positions must be non-empty")
    if any(value < 0 for value in pos):
        raise ValueError("restore positions must be non-negative")
    if type(allow_sparse_positions) is not bool:
        raise TypeError("allow_sparse_positions must be boolean")
    if allow_sparse_positions:
        if any(right <= left for left, right in zip(pos, pos[1:])):
            raise ValueError(
                "combined restore positions must be strictly increasing"
            )
    elif any(right != left + 1 for left, right in zip(pos, pos[1:])):
        raise ValueError("chunked restore positions must be strictly contiguous")

    ordered = _ordered_tiled_placements(placements, query_start)
    if boundary_tokens not in (0, BOUNDARY128_REPLAY_TOKENS):
        raise ValueError("row layout boundary profile is unsupported")
    base_prefixes = {
        index: (
            0
            if boundary_tokens == 0 or placement.global_offset == 0
            else min(boundary_tokens, placement.length)
        )
        for index, placement in enumerate(ordered)
    }
    prefixes = dict(base_prefixes)
    seen_prefixes = set()
    for raw_segment_index, raw_prefix in selected_prefix_tokens:
        segment_index = _strict_int(raw_segment_index, "selected-prefix segment")
        prefix = _strict_int(raw_prefix, "selected-prefix length")
        if segment_index in seen_prefixes:
            raise ValueError("selected prefix contains a duplicate segment")
        if segment_index not in prefixes:
            raise ValueError("selected prefix targets an unknown segment")
        seen_prefixes.add(segment_index)
        length = ordered[segment_index].length
        if boundary_tokens == 0:
            if prefix != 0:
                raise ValueError(
                    "context-bound exact-position geometry requires skip_first=0"
                )
        elif (
            prefix < base_prefixes[segment_index]
            or prefix > length
            or prefix % boundary_tokens != 0
        ):
            raise ValueError("selected prefix violates boundary128 geometry")
        prefixes[segment_index] = prefix

    if boundary_tokens == 0 and checkpoint_islands:
        raise ValueError("context-bound geometry forbids checkpoint islands")
    normalized_islands = []
    for raw_island in checkpoint_islands:
        segment_index = _strict_int(
            _selector_field(raw_island, "segment_index"),
            "checkpoint-island segment",
        )
        token_begin = _strict_int(
            _selector_field(raw_island, "token_begin"),
            "checkpoint-island begin",
        )
        token_end = _strict_int(
            _selector_field(raw_island, "token_end"),
            "checkpoint-island end",
        )
        if segment_index not in prefixes:
            raise ValueError("checkpoint island targets an unknown segment")
        if (
            token_begin <= 0
            or token_begin % CHECKPOINT_REPLAY_STRIDE != 0
            or token_end <= token_begin
            or token_end % boundary_tokens != 0
            or token_end > ordered[segment_index].length
        ):
            raise ValueError("checkpoint island violates boundary128 geometry")
        normalized_islands.append(
            CheckpointReplayIsland(segment_index, token_begin, token_end)
        )
    normalized_islands.sort(key=lambda item: (item.segment_index, item.token_begin))

    ranges_by_segment: Dict[int, list[Tuple[int, int]]] = {
        index: ([(0, prefix)] if prefix else [])
        for index, prefix in prefixes.items()
    }
    previous_end_by_segment = dict(prefixes)
    for island in normalized_islands:
        previous_end = previous_end_by_segment[island.segment_index]
        if island.token_begin - previous_end < boundary_tokens:
            raise ValueError(
                "checkpoint islands overlap or are adjacent to an online range"
            )
        ranges_by_segment[island.segment_index].append(
            (island.token_begin, island.token_end)
        )
        previous_end_by_segment[island.segment_index] = island.token_end

    if protected_ranges and not allow_sparse_positions:
        raise ValueError(
            "query-protected ranges require combined sparse-position geometry"
        )
    normalized_protected_ranges = []
    protected_cursor = 0
    for raw_range in protected_ranges:
        begin = _strict_int(
            _selector_field(raw_range, "start"), "query-protected range begin"
        )
        end = _strict_int(
            _selector_field(raw_range, "end"), "query-protected range end"
        )
        if (
            begin < protected_cursor
            or begin < 0
            or begin >= end
            or end > query_start
            or begin % CHECKPOINT_REPLAY_STRIDE != 0
            or end % CHECKPOINT_REPLAY_STRIDE != 0
        ):
            raise ValueError("query-protected range geometry is invalid")
        placement_index = next(
            (
                index
                for index, placement in enumerate(ordered)
                if placement.global_offset <= begin
                and end <= placement.global_offset + placement.length
            ),
            None,
        )
        if placement_index is None:
            raise ValueError(
                "query-protected range must stay inside one segment"
            )
        placement = ordered[placement_index]
        ranges_by_segment[placement_index].append(
            (
                begin - placement.global_offset,
                end - placement.global_offset,
            )
        )
        normalized_protected_ranges.append((begin, end))
        protected_cursor = end

    clean = [False] * len(pos)
    grouped_rows: Dict[int, Tuple[list[int], list[int]]] = {}
    placement_index = 0
    for output_row, absolute in enumerate(pos):
        if absolute >= query_start:
            continue
        while (
            placement_index + 1 < len(ordered)
            and absolute
            >= ordered[placement_index].global_offset
            + ordered[placement_index].length
        ):
            placement_index += 1
        placement = ordered[placement_index]
        if not (
            placement.global_offset
            <= absolute
            < placement.global_offset + placement.length
        ):
            raise ValueError("document position is not covered by a segment")
        local = absolute - placement.global_offset
        if _overlaps(local, local + 1, ranges_by_segment[placement_index]):
            continue
        clean[output_row] = True
        rows, local_rows = grouped_rows.setdefault(placement_index, ([], []))
        rows.append(output_row)
        local_rows.append(local)

    clean_rows = tuple(
        RestoreRows(
            placement_index=index,
            seg_hash=ordered[index].seg_hash,
            output_rows=tuple(rows),
            local_rows=tuple(local_rows),
        )
        for index, (rows, local_rows) in sorted(grouped_rows.items())
    )
    return Boundary128RowLayout(
        placements=ordered,
        positions=pos,
        query_start=query_start,
        clean_mask=tuple(clean),
        clean_rows=clean_rows,
        dirty_output_rows=tuple(
            index for index, reusable in enumerate(clean) if not reusable
        ),
        selected_prefix_tokens=tuple(sorted(prefixes.items())),
        checkpoint_islands=tuple(normalized_islands),
        protected_ranges=tuple(normalized_protected_ranges),
        online_local_ranges=MappingProxyType(
            {
                index: tuple(ranges)
                for index, ranges in sorted(ranges_by_segment.items())
            }
        ),
        boundary_tokens=boundary_tokens,
    )


def build_boundary128_row_layout(
    *,
    placements: Sequence[SegmentPlacement],
    positions: Sequence[int],
    query_start: int,
    selected_prefix_tokens: Sequence[Tuple[int, int]] = (),
    checkpoint_islands: Sequence[object] = (),
) -> Boundary128RowLayout:
    """Materialize the unchanged legacy boundary128-v2 geometry."""

    return _build_row_layout(
        placements=placements,
        positions=positions,
        query_start=query_start,
        selected_prefix_tokens=selected_prefix_tokens,
        checkpoint_islands=checkpoint_islands,
        protected_ranges=(),
        boundary_tokens=BOUNDARY128_REPLAY_TOKENS,
    )


def build_combined_row_sparse_layout(
    *,
    placements: Sequence[SegmentPlacement],
    positions: Sequence[int],
    query_start: int,
    selected_prefix_tokens: Sequence[Tuple[int, int]] = (),
    checkpoint_islands: Sequence[object] = (),
    protected_ranges: Sequence[object] = (),
) -> Boundary128RowLayout:
    """Materialize boundary128 reuse over authenticated selected rows."""

    return _build_row_layout(
        placements=placements,
        positions=positions,
        query_start=query_start,
        selected_prefix_tokens=selected_prefix_tokens,
        checkpoint_islands=checkpoint_islands,
        protected_ranges=protected_ranges,
        boundary_tokens=BOUNDARY128_REPLAY_TOKENS,
        allow_sparse_positions=True,
    )


def build_context_bound_row_layout(
    *,
    placements: Sequence[SegmentPlacement],
    positions: Sequence[int],
    query_start: int,
    selected_prefix_tokens: Sequence[Tuple[int, int]] = (),
    checkpoint_islands: Sequence[object] = (),
) -> Boundary128RowLayout:
    """Restore every document row at its authenticated source position."""

    return _build_row_layout(
        placements=placements,
        positions=positions,
        query_start=query_start,
        selected_prefix_tokens=selected_prefix_tokens,
        checkpoint_islands=checkpoint_islands,
        protected_ranges=(),
        boundary_tokens=0,
    )


def build_layer_restore_plans(
    *, spec: SharedLatentSpec, row_layout: Boundary128RowLayout
) -> Mapping[int, LayerRestorePlan]:
    """Return block/state descriptors for exactly the clean row geometry."""

    if any(placement.length != spec.length for placement in row_layout.placements):
        raise ValueError("segment placement length changed")
    position_to_output = {
        position: output_row
        for output_row, position in enumerate(row_layout.positions)
    }
    layer_plans: Dict[int, LayerRestorePlan] = {}
    for layer_spec in spec.layers:
        ratio = layer_spec.compress_ratio
        checkpoint_restores = []
        for island in row_layout.checkpoint_islands:
            placement = row_layout.placements[island.segment_index]
            global_anchor = placement.global_offset + island.token_begin
            output_begin_row = position_to_output.get(global_anchor)
            if output_begin_row is None:
                continue
            checkpoint_restores.append(
                CheckpointRestore(
                    placement_index=island.segment_index,
                    seg_hash=placement.seg_hash,
                    local_anchor=island.token_begin,
                    global_anchor=global_anchor,
                    output_begin_row=output_begin_row,
                    swa_carry_local_rows=tuple(
                        range(
                            island.token_begin - BOUNDARY128_REPLAY_TOKENS,
                            island.token_begin,
                        )
                    ),
                    needs_attention_state=bool(ratio),
                    needs_indexer_state=(ratio == 4),
                )
            )
        compressed_groups = []
        if ratio:
            for placement_index, placement in enumerate(row_layout.placements):
                online_ranges = row_layout.online_local_ranges[placement_index]
                output_completion_rows = []
                local_blocks = []
                for block in range(placement.length // ratio):
                    local_begin = block * ratio
                    local_end = local_begin + ratio
                    if _overlaps(local_begin, local_end, online_ranges):
                        continue
                    completion_position = placement.global_offset + local_end - 1
                    output_row = position_to_output.get(completion_position)
                    if output_row is not None:
                        output_completion_rows.append(output_row)
                        local_blocks.append(block)
                if local_blocks:
                    compressed_groups.append(
                        RestoreBlocks(
                            placement_index=placement_index,
                            seg_hash=placement.seg_hash,
                            output_completion_rows=tuple(output_completion_rows),
                            local_blocks=tuple(local_blocks),
                        )
                    )
        terminal_placements = tuple(
            placement_index
            for placement_index, placement in enumerate(row_layout.placements)
            if placement.global_offset + placement.length - 1 in position_to_output
            and not _overlaps(
                placement.length - 1,
                placement.length,
                row_layout.online_local_ranges[placement_index],
            )
        )
        groups_tuple = tuple(compressed_groups)
        layer_plans[layer_spec.layer_id] = LayerRestorePlan(
            layer_id=layer_spec.layer_id,
            compress_ratio=ratio,
            compressed_blocks=groups_tuple,
            indexer_blocks=groups_tuple if ratio == 4 else (),
            checkpoint_restores=tuple(checkpoint_restores),
            terminal_state_placements=terminal_placements if ratio else (),
            may_skip_compressor_for_clean_blocks=bool(groups_tuple),
            # Reusing positionless Indexer K never replaces query-dependent Q/K
            # scoring and top-k selection.
            must_run_indexer_query_path=(ratio == 4),
        )
    return MappingProxyType(layer_plans)


class DSV4SharedLatentController:
    """Process-local, fail-closed controller for complete segment artifacts."""

    def __init__(self) -> None:
        self._owner_token = object()
        self._staging: Dict[Tuple[str, str], _SegmentStage] = {}
        self._committed: Dict[str, SharedLatentArtifact] = {}
        self._pending_publishes: Dict[str, SharedLatentPublishReceipt] = {}
        self._commit_epoch = 0

    @property
    def next_commit_epoch(self) -> int:
        return self._commit_epoch + 1

    def begin_capture(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        token_ids: Sequence[int],
        spec: SharedLatentSpec,
        token_hash: str = "",
    ) -> None:
        if (
            not isinstance(seg_hash, str)
            or not seg_hash
            or not isinstance(generation_id, str)
            or not generation_id
        ):
            raise ValueError("segment hash and generation id must be non-empty")
        tokens = tuple(_strict_int(token, "token id") for token in token_ids)
        if len(tokens) != spec.length:
            raise ValueError("token ids must cover the complete segment")
        actual_token_hash = _token_digest(tokens)
        if token_hash and token_hash != actual_token_hash:
            raise ValueError("provided token hash does not match token ids")
        key = (seg_hash, generation_id)
        if any(existing[0] == seg_hash for existing in self._staging):
            raise ValueError("another generation of this segment is staging")
        self._staging[key] = _SegmentStage(
            seg_hash=seg_hash,
            generation_id=generation_id,
            token_ids=tokens,
            token_hash=actual_token_hash,
            spec=spec,
            layers={
                layer.layer_id: _LayerStage(layer, spec.length)
                for layer in spec.layers
            },
        )

    def _stage(
        self, seg_hash: str, generation_id: str, layer_id: int
    ) -> Tuple[_SegmentStage, _LayerStage]:
        stage = self._staging.get((str(seg_hash), str(generation_id)))
        if stage is None:
            raise KeyError("shared-latent capture generation is absent")
        layer = stage.layers.get(int(layer_id))
        if layer is None:
            raise ValueError("capture targets a non-required layer")
        return stage, layer

    def capture_swa_rows(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        layer_id: int,
        local_rows: Sequence[int],
        positionless_packed: object,
    ) -> None:
        _, layer = self._stage(seg_hash, generation_id, layer_id)
        layer.swa.capture(
            local_rows,
            positionless_packed,
            name=f"layer {layer_id} full SWA",
        )

    def capture_compressed_blocks(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        layer_id: int,
        local_blocks: Sequence[int],
        positionless_packed: object,
    ) -> None:
        _, layer = self._stage(seg_hash, generation_id, layer_id)
        if layer.compressed is None:
            raise ValueError("SWA-only layer has no compressed cache")
        layer.compressed.capture(
            local_blocks,
            positionless_packed,
            name=f"layer {layer_id} compressed cache",
        )

    def capture_indexer_blocks(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        layer_id: int,
        local_blocks: Sequence[int],
        positionless_keys: object,
        position_semantics: str,
    ) -> None:
        _, layer = self._stage(seg_hash, generation_id, layer_id)
        if layer.indexer is None:
            raise ValueError("only C4 layers own Indexer keys")
        if position_semantics != INDEXER_POSITION_SEMANTICS:
            raise ValueError(
                "post-Hadamard canonical-position Indexer bytes are not a "
                "position-independent artifact"
            )
        layer.indexer.capture(
            local_blocks,
            positionless_keys,
            name=f"layer {layer_id} Indexer cache",
        )

    def capture_terminal_states(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        layer_id: int,
        attention_state: object,
        indexer_state: Optional[object] = None,
    ) -> None:
        _, layer = self._stage(seg_hash, generation_id, layer_id)
        spec = layer.spec
        if spec.compress_ratio == 0:
            raise ValueError("SWA-only layers have no compressor state")
        attention = _as_bytes(
            attention_state,
            expected=spec.attention_terminal_state_bytes,
            name=f"layer {layer_id} terminal attention state",
        )
        if (
            layer.attention_terminal_state is not None
            and layer.attention_terminal_state != attention
        ):
            raise ValueError("terminal attention state changed within a generation")
        indexer = None
        if spec.compress_ratio == 4:
            indexer = _as_bytes(
                indexer_state,
                expected=spec.indexer_terminal_state_bytes,
                name=f"layer {layer_id} terminal Indexer state",
            )
            if (
                layer.indexer_terminal_state is not None
                and layer.indexer_terminal_state != indexer
            ):
                raise ValueError("terminal Indexer state changed within a generation")
        elif indexer_state is not None:
            raise ValueError("C128 layers do not accept Indexer state")
        # Publish both components to staging only after every payload and
        # duplicate-write check above succeeds.
        layer.attention_terminal_state = attention
        if indexer is not None:
            layer.indexer_terminal_state = indexer

    def capture_checkpoint_states(
        self,
        *,
        seg_hash: str,
        generation_id: str,
        layer_id: int,
        anchor_tokens: int,
        attention_state: object,
        indexer_state: Optional[object] = None,
    ) -> None:
        """Capture one internal 512-token restart anchor atomically.

        The preceding SWA carry is derived from this artifact's complete SWA
        rows.  Compressor state cannot be reconstructed from packed output
        blocks, so it is an explicit required component at every internal
        anchor.  C4 also requires its independent Indexer-compressor state.
        """

        stage, layer = self._stage(seg_hash, generation_id, layer_id)
        spec = layer.spec
        anchor = _strict_int(anchor_tokens, "checkpoint anchor")
        if anchor not in stage.spec.checkpoint_anchors:
            raise ValueError("checkpoint anchor is not required by this artifact")
        if spec.compress_ratio == 0:
            raise ValueError("SWA-only layers have no compressor checkpoints")
        attention = _as_bytes(
            attention_state,
            expected=spec.attention_terminal_state_bytes,
            name=f"layer {layer_id} checkpoint {anchor} attention state",
        )
        previous_attention = layer.attention_checkpoint_states.get(anchor)
        if previous_attention is not None and previous_attention != attention:
            raise ValueError(
                "checkpoint attention state changed within a generation"
            )
        indexer = None
        if spec.compress_ratio == 4:
            indexer = _as_bytes(
                indexer_state,
                expected=spec.indexer_terminal_state_bytes,
                name=f"layer {layer_id} checkpoint {anchor} Indexer state",
            )
            previous_indexer = layer.indexer_checkpoint_states.get(anchor)
            if previous_indexer is not None and previous_indexer != indexer:
                raise ValueError(
                    "checkpoint Indexer state changed within a generation"
                )
        elif indexer_state is not None:
            raise ValueError("C128 layers do not accept Indexer checkpoints")
        # Keep the checkpoint pair transactional within the staging generation.
        layer.attention_checkpoint_states[anchor] = attention
        if indexer is not None:
            layer.indexer_checkpoint_states[anchor] = indexer

    def abort_capture(self, *, seg_hash: str, generation_id: str) -> None:
        self._staging.pop((str(seg_hash), str(generation_id)), None)

    def commit_capture(
        self, *, seg_hash: str, generation_id: str
    ) -> SharedLatentArtifact:
        key = (str(seg_hash), str(generation_id))
        stage = self._staging.get(key)
        if stage is None:
            raise KeyError("shared-latent capture generation is absent")
        frozen: Dict[int, LayerArtifact] = {}
        required_checkpoint_anchors = set(stage.spec.checkpoint_anchors)
        for layer_id, layer in sorted(stage.layers.items()):
            compressed = (
                None
                if layer.compressed is None
                else layer.compressed.freeze(
                    name=f"layer {layer_id} compressed cache"
                )
            )
            indexer = (
                None
                if layer.indexer is None
                else layer.indexer.freeze(name=f"layer {layer_id} Indexer cache")
            )
            if layer.spec.compress_ratio and layer.attention_terminal_state is None:
                raise ValueError(
                    f"layer {layer_id} terminal attention state is missing"
                )
            if layer.spec.compress_ratio == 4 and layer.indexer_terminal_state is None:
                raise ValueError(f"layer {layer_id} terminal Indexer state is missing")
            if layer.spec.compress_ratio and (
                set(layer.attention_checkpoint_states)
                != required_checkpoint_anchors
            ):
                missing = sorted(
                    required_checkpoint_anchors
                    - set(layer.attention_checkpoint_states)
                )
                extra = sorted(
                    set(layer.attention_checkpoint_states)
                    - required_checkpoint_anchors
                )
                raise ValueError(
                    f"layer {layer_id} attention checkpoints are incomplete "
                    f"(missing={missing}, extra={extra})"
                )
            if layer.spec.compress_ratio == 4 and (
                set(layer.indexer_checkpoint_states)
                != required_checkpoint_anchors
            ):
                missing = sorted(
                    required_checkpoint_anchors
                    - set(layer.indexer_checkpoint_states)
                )
                extra = sorted(
                    set(layer.indexer_checkpoint_states)
                    - required_checkpoint_anchors
                )
                raise ValueError(
                    f"layer {layer_id} Indexer checkpoints are incomplete "
                    f"(missing={missing}, extra={extra})"
                )
            frozen[layer_id] = LayerArtifact(
                spec=layer.spec,
                swa_positionless_packed=layer.swa.freeze(
                    name=f"layer {layer_id} full SWA"
                ),
                compressed_positionless_packed=compressed,
                indexer_positionless_keys=indexer,
                attention_terminal_state=layer.attention_terminal_state,
                indexer_terminal_state=layer.indexer_terminal_state,
                attention_checkpoint_states=MappingProxyType(
                    dict(sorted(layer.attention_checkpoint_states.items()))
                ),
                indexer_checkpoint_states=MappingProxyType(
                    dict(sorted(layer.indexer_checkpoint_states.items()))
                ),
            )
        self._commit_epoch += 1
        artifact = SharedLatentArtifact(
            seg_hash=stage.seg_hash,
            token_hash=stage.token_hash,
            token_ids=stage.token_ids,
            spec=stage.spec,
            layers=MappingProxyType(frozen),
            commit_epoch=self._commit_epoch,
        )
        # Atomic replacement: the old committed generation remains visible
        # until every required component above has frozen successfully.
        self._committed[stage.seg_hash] = artifact
        self._staging.pop(key)
        return artifact

    def publish_capture(
        self, *, seg_hash: str, generation_id: str
    ) -> SharedLatentPublishReceipt:
        """Publish a frozen artifact but retain an exact rollback receipt."""

        seg_hash = str(seg_hash)
        if seg_hash in self._pending_publishes:
            raise ValueError("this shared-latent segment already has a pending publish")
        previous = self._committed.get(seg_hash)
        artifact = self.commit_capture(
            seg_hash=seg_hash,
            generation_id=str(generation_id),
        )
        receipt = SharedLatentPublishReceipt(
            controller_token=self._owner_token,
            artifact=artifact,
            previous_artifact=previous,
        )
        self._pending_publishes[seg_hash] = receipt
        return receipt

    def rollback_publish(self, receipt: SharedLatentPublishReceipt) -> None:
        if (
            not isinstance(receipt, SharedLatentPublishReceipt)
            or receipt.controller_token is not self._owner_token
            or receipt.state != "published"
            or self._pending_publishes.get(receipt.artifact.seg_hash) is not receipt
            or self._committed.get(receipt.artifact.seg_hash) is not receipt.artifact
        ):
            raise ValueError("shared-latent publish receipt is stale")
        if receipt.previous_artifact is None:
            self._committed.pop(receipt.artifact.seg_hash, None)
        else:
            self._committed[receipt.artifact.seg_hash] = receipt.previous_artifact
        self._pending_publishes.pop(receipt.artifact.seg_hash)
        receipt.state = "rolled_back"

    def confirm_publish(self, receipt: SharedLatentPublishReceipt) -> SharedLatentArtifact:
        if (
            not isinstance(receipt, SharedLatentPublishReceipt)
            or receipt.controller_token is not self._owner_token
            or receipt.state != "published"
            or self._pending_publishes.get(receipt.artifact.seg_hash) is not receipt
            or self._committed.get(receipt.artifact.seg_hash) is not receipt.artifact
        ):
            raise ValueError("shared-latent publish receipt is stale")
        self._pending_publishes.pop(receipt.artifact.seg_hash)
        receipt.state = "confirmed"
        return receipt.artifact

    def get_committed(self, seg_hash: str) -> SharedLatentArtifact:
        artifact = self._committed.get(str(seg_hash))
        if artifact is None:
            raise KeyError(f"shared-latent segment {seg_hash!r} is not committed")
        return artifact

    def validate_restore_plan(self, plan: SharedLatentRestorePlan) -> None:
        """Reject a plan whose artifact was replaced after preflight.

        Serving must call this immediately before the all-TP readiness vote and
        cache mutation.  It is the CPU equivalent of the commit-epoch/storage
        identity certificate needed by the eventual device mirror.
        """

        if not isinstance(plan, SharedLatentRestorePlan):
            raise TypeError("shared-latent restore plan has an invalid type")
        for seg_hash, epoch in plan.artifact_epochs.items():
            bound_artifact = plan.artifacts.get(seg_hash)
            if bound_artifact is None or bound_artifact.commit_epoch != epoch:
                raise ValueError("restore plan lost its bound artifact generation")
            artifact = self.get_committed(seg_hash)
            if artifact is not bound_artifact or artifact.commit_epoch != epoch:
                raise ValueError(
                    f"shared-latent segment {seg_hash!r} was replaced after preflight"
                )
            if artifact.spec != plan.spec:
                raise ValueError("shared-latent compatibility changed after preflight")

    def prepare_restore(
        self,
        *,
        spec: SharedLatentSpec,
        placements: Sequence[SegmentPlacement],
        positions: Sequence[int],
        input_token_ids: Sequence[int],
        query_start: int,
        boundary_tokens: int = BOUNDARY128_REPLAY_TOKENS,
        execution_profile: str = "",
        selected_prefix_tokens: Sequence[Tuple[int, int]] = (),
        checkpoint_islands: Sequence[object] = (),
        protected_ranges: Sequence[object] = (),
    ) -> SharedLatentRestorePlan:
        """Build a complete row/block certificate for one chunked prefill call.

        ``positions`` are the scheduler-owned active absolute rows in this
        forward, not necessarily the complete request.  Legacy/context-bound
        profiles require a contiguous microforward.  The combined profile
        accepts a strictly increasing selected-row projection while retaining
        the same boundary128/checkpoint-island provenance. Query rows remain
        contiguous and fully online by the enclosing runtime contract.
        """

        query_start = _strict_int(query_start, "query_start")
        boundary_tokens = _strict_int(boundary_tokens, "boundary_tokens")
        if execution_profile == CONTEXT_BOUND_EXECUTION_PROFILE:
            if boundary_tokens != 0:
                raise ValueError(
                    "context-bound artifact profile requires boundary_tokens=0"
                )
            layout_builder = build_context_bound_row_layout
        elif execution_profile in (
            "",
            "boundary128_v2",
            INDEPENDENT_RELOCATION_EXECUTION_PROFILE,
        ):
            if boundary_tokens != BOUNDARY128_REPLAY_TOKENS:
                raise ValueError(
                    "legacy artifact profile requires boundary_tokens=128"
                )
            layout_builder = build_boundary128_row_layout
        elif execution_profile == COMBINED_ROW_SPARSE_EXECUTION_PROFILE:
            if boundary_tokens != BOUNDARY128_REPLAY_TOKENS:
                raise ValueError(
                    "combined artifact profile requires boundary_tokens=128"
                )
            layout_builder = build_combined_row_sparse_layout
        else:
            raise ValueError("shared-latent restore execution profile is unsupported")
        pos = tuple(_strict_int(value, "position") for value in positions)
        tokens = tuple(_strict_int(value, "input token id") for value in input_token_ids)
        if not pos or len(pos) != len(tokens):
            raise ValueError("restore positions and token ids must be non-empty/equal")
        if any(token < 0 or token > 0xFFFFFFFF for token in tokens):
            raise ValueError("input token ids must fit an unsigned 32-bit value")
        row_layout = layout_builder(
            placements=placements,
            positions=pos,
            query_start=query_start,
            selected_prefix_tokens=selected_prefix_tokens,
            checkpoint_islands=checkpoint_islands,
            **(
                {"protected_ranges": protected_ranges}
                if execution_profile == COMBINED_ROW_SPARSE_EXECUTION_PROFILE
                else {}
            ),
        )

        artifacts: Dict[str, SharedLatentArtifact] = {}
        for placement in row_layout.placements:
            artifact = self.get_committed(placement.seg_hash)
            if artifact.spec != spec:
                raise ValueError("shared-latent model/policy/geometry changed")
            if placement.length != spec.length:
                raise ValueError("segment placement length changed")
            if placement.token_hash and placement.token_hash != artifact.token_hash:
                raise ValueError("segment placement token hash changed")
            artifacts[placement.seg_hash] = artifact

        placement_index = 0
        for output_row, absolute in enumerate(pos):
            if absolute >= query_start:
                continue
            while (
                placement_index + 1 < len(row_layout.placements)
                and absolute
                >= row_layout.placements[placement_index].global_offset
                + row_layout.placements[placement_index].length
            ):
                placement_index += 1
            placement = row_layout.placements[placement_index]
            local = absolute - placement.global_offset
            expected_token = artifacts[placement.seg_hash].token_ids[local]
            if tokens[output_row] != expected_token:
                raise ValueError("server token identity differs from the artifact")

        return SharedLatentRestorePlan(
            spec=spec,
            positions=pos,
            query_start=query_start,
            clean_mask=row_layout.clean_mask,
            clean_rows=row_layout.clean_rows,
            dirty_output_rows=row_layout.dirty_output_rows,
            layers=build_layer_restore_plans(spec=spec, row_layout=row_layout),
            artifacts=MappingProxyType(dict(artifacts)),
            artifact_epochs=MappingProxyType(
                {
                    seg_hash: artifact.commit_epoch
                    for seg_hash, artifact in artifacts.items()
                }
            ),
            selected_prefix_tokens=row_layout.selected_prefix_tokens,
            checkpoint_islands=row_layout.checkpoint_islands,
            protected_ranges=row_layout.protected_ranges,
            boundary_tokens=boundary_tokens,
            execution_profile=str(execution_profile),
        )


__all__ = [
    "BOUNDARY128_REPLAY_TOKENS",
    "Boundary128RowLayout",
    "CHECKPOINT_REPLAY_STRIDE",
    "CheckpointReplayIsland",
    "CheckpointRestore",
    "COMBINED_ROW_SPARSE_EXECUTION_PROFILE",
    "CONTEXT_BOUND_EXECUTION_PROFILE",
    "DEFAULT_REQUIRED_LAYER_IDS",
    "DSV4_0731_DENSE_PREFIX_LAYERS",
    "DSV4_0731_DENSE_SUFFIX_LAYERS",
    "DSV4_0731_NUM_TARGET_LAYERS",
    "DSV4_0731_TARGET_COMPRESS_RATIOS",
    "DSV4SharedLatentController",
    "INDEXER_POSITION_SEMANTICS",
    "LayerArtifact",
    "LayerComponentSpec",
    "LayerRestorePlan",
    "PACKED_LATENT_BYTES",
    "PACKED_LATENT_POSITION_SEMANTICS",
    "RestoreBlocks",
    "RestoreRows",
    "SHARED_LATENT_FORMAT_VERSION",
    "SegmentPlacement",
    "SharedLatentArtifact",
    "SharedLatentPublishReceipt",
    "SharedLatentRestorePlan",
    "SharedLatentSpec",
    "TOKEN_HASH_SEMANTICS",
    "build_boundary128_row_layout",
    "build_combined_row_sparse_layout",
    "build_context_bound_row_layout",
    "build_dsv4_0731_shared_latent_spec",
    "build_layer_restore_plans",
]
