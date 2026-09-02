from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple


# Flash-0731 keeps its legacy/default 64-island policy.  Pro-0813 may request
# a larger explicit cap for 256K+ contexts; the model runner guards that path
# with REDKNOT_DSV4_VARIANT=pro0813 so this limit expansion does not alter the
# Flash execution profile.
MAX_CHECKPOINT_ISLANDS = 256


@dataclass(frozen=True)
class SegmentPrefixCandidates:
    """Offline Indexer salience for one independently cached segment."""

    segment_index: int
    length: int
    base_prefix_tokens: int
    query_weight: float
    unit_ordinals: Sequence[int]
    unit_scores: Sequence[float]


@dataclass(frozen=True)
class SelectedSegmentPrefix:
    """The contiguous online prefix selected for one cached segment."""

    segment_index: int
    prefix_tokens: int
    accumulated_score: float


@dataclass(frozen=True)
class CheckpointCellCandidates:
    """Prefix utilities for one independently restorable checkpoint cell.

    ``cell_index`` addresses a checkpoint-sized cell relative to its segment.
    ``block_scores[i]`` is the non-negative utility of recomputing the i-th
    block in that cell.  A legal choice always takes a prefix of these blocks;
    callers must compose any query weight into the scores before allocation.
    """

    segment_index: int
    cell_index: int
    block_scores: Sequence[float]


@dataclass(frozen=True)
class CheckpointBridgeCandidates:
    """Prefix utilities between the mandatory boundary and first checkpoint.

    A migrated segment always recomputes its first ``block_tokens`` rows.  The
    remaining rows before the first checkpoint are optional, but they must be
    selected as one contiguous prefix so the live SWA/compressor state can be
    advanced safely.  Unlike a checkpoint cell, this bridge never needs a
    restore descriptor.  Filling the bridge makes the first checkpoint cell a
    continuation of the same live prefix.
    """

    segment_index: int
    block_scores: Sequence[float]


@dataclass(frozen=True)
class SelectedCheckpointIsland:
    """One merged segment-relative interval selected for online recomputation."""

    segment_index: int
    token_begin: int
    token_end: int
    accumulated_score: float


@dataclass(frozen=True)
class CheckpointReplayLayout:
    """Descriptor-free live prefixes and independently restored intervals."""

    selected_prefix_tokens: Tuple[Tuple[int, int], ...]
    restore_islands: Tuple[SelectedCheckpointIsland, ...]


def checkpoint_mandatory_prefix_tokens(
    *,
    segment_global_offset: int,
    segment_length: int,
    block_tokens: int = 128,
) -> int:
    """Return the mandatory online prefix for one cached segment.

    Segment zero remains at its canonical position and needs no migration
    boundary.  Every migrated segment replays exactly one online block (or the
    whole segment when it is shorter).  Optional rows up to the first 512-token
    checkpoint belong to :class:`CheckpointBridgeCandidates` and therefore
    consume the same global budget as checkpoint cells.
    """

    offset = int(segment_global_offset)
    length = int(segment_length)
    if offset < 0:
        raise ValueError("segment_global_offset must be non-negative")
    if block_tokens <= 0:
        raise ValueError("block size must be positive")
    if length <= 0 or length % block_tokens != 0:
        raise ValueError("segment_length must be positive and block-aligned")
    return 0 if offset == 0 else min(block_tokens, length)


def checkpoint_effective_segment_cap_tokens(
    *,
    segment_length: int,
    cap_ratio: float,
    mandatory_prefix_tokens: int,
    block_tokens: int = 128,
) -> int:
    """Return an aligned replay cap that never rejects mandatory state.

    The ratio cap constrains optional replay.  It cannot make a migrated
    segment's mandatory first block illegal, including for a one-block segment
    where flooring ``length * cap_ratio`` would otherwise produce zero.
    """

    length = int(segment_length)
    mandatory = int(mandatory_prefix_tokens)
    ratio = float(cap_ratio)
    if block_tokens <= 0:
        raise ValueError("block size must be positive")
    if length <= 0 or length % block_tokens != 0:
        raise ValueError("segment_length must be positive and block-aligned")
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("cap_ratio must be finite and in (0, 1]")
    if mandatory < 0 or mandatory > length or mandatory % block_tokens != 0:
        raise ValueError(
            "mandatory_prefix_tokens must be aligned and within the segment"
        )
    aligned_ratio_cap = math.floor(length * ratio / block_tokens) * block_tokens
    return min(length, max(mandatory, aligned_ratio_cap))


def allocate_request_global_prefixes(
    segments: Iterable[SegmentPrefixCandidates],
    *,
    hot_budget_tokens: int,
    block_tokens: int = 128,
    indexer_unit_tokens: int = 4,
    per_segment_cap_ratio: float = 0.75,
    min_query_weight: float = 0.05,
) -> Tuple[SelectedSegmentPrefix, ...]:
    """Allocate a deterministic request-level budget to contiguous prefixes.

    Arbitrary four-token C4 islands are unsafe: the overlap compressor and SWA
    path need state from the skipped gap.  This allocator therefore exposes only
    the *next* 128-token block of each segment.  Selecting a later block first
    requires selecting every earlier block, so all online work is a prefix whose
    C4, C128 and SWA state can be rebuilt sequentially.

    ``hot_budget_tokens`` is the additional budget after mandatory boundary and
    query rows.  A segment cap limits its total prefix (including the mandatory
    base prefix) to a fraction of its length.
    """

    if block_tokens <= 0 or indexer_unit_tokens <= 0:
        raise ValueError("block and indexer unit sizes must be positive")
    if block_tokens % indexer_unit_tokens != 0:
        raise ValueError("block_tokens must be divisible by indexer_unit_tokens")
    if not 0.0 < per_segment_cap_ratio <= 1.0:
        raise ValueError("per_segment_cap_ratio must be in (0, 1]")
    if not math.isfinite(min_query_weight) or min_query_weight < 0.0:
        raise ValueError("min_query_weight must be finite and non-negative")
    if hot_budget_tokens < 0:
        raise ValueError("hot_budget_tokens must be non-negative")

    block_budget = hot_budget_tokens // block_tokens
    if block_budget <= 0:
        return ()

    state: Dict[int, dict] = {}
    for segment in segments:
        index = int(segment.segment_index)
        if index in state:
            raise ValueError(f"duplicate segment index: {index}")
        length = int(segment.length)
        base = int(segment.base_prefix_tokens)
        if length <= 0 or base < 0 or base > length:
            raise ValueError(f"invalid segment length/base prefix for {index}")
        if base % block_tokens != 0 or length % block_tokens != 0:
            raise ValueError(f"segment {index} is not {block_tokens}-token aligned")
        weight = float(segment.query_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"segment {index} has invalid query weight")
        if len(segment.unit_ordinals) != len(segment.unit_scores):
            raise ValueError(f"segment {index} has mismatched units/scores")

        unit_scores: Dict[int, float] = {}
        max_units = length // indexer_unit_tokens
        for raw_ordinal, raw_score in zip(
            segment.unit_ordinals, segment.unit_scores
        ):
            ordinal = int(raw_ordinal)
            score = float(raw_score)
            if ordinal < 0 or ordinal >= max_units:
                raise ValueError(f"segment {index} has out-of-range Indexer unit")
            if ordinal in unit_scores:
                raise ValueError(f"segment {index} has duplicate Indexer unit")
            if not math.isfinite(score) or score < 0.0:
                raise ValueError(f"segment {index} has invalid Indexer score")
            unit_scores[ordinal] = score

        cap = math.floor(length * per_segment_cap_ratio / block_tokens) * block_tokens
        cap = max(base, min(length, cap))
        query_weight = max(min_query_weight, weight)
        units_per_block = block_tokens // indexer_unit_tokens
        block_scores = []
        for token_begin in range(base, cap, block_tokens):
            first_unit = token_begin // indexer_unit_tokens
            offline_score = sum(
                unit_scores.get(unit, 0.0)
                for unit in range(first_unit, first_unit + units_per_block)
            )
            block_scores.append(offline_score * query_weight)

        state[index] = {
            "base": base,
            "block_scores": block_scores,
        }

    # Multiple-choice knapsack over prefix lengths. A greedy next-block heap is
    # not optimal when a weak block gates a later strong block (for example
    # A=[0,100], B=[1,1], budget=2). The real problem is tiny: eight segments,
    # roughly 64 blocks each, and about 50 budget blocks.
    ordered = sorted(state)
    # used_blocks -> (score, choices_by_ordered_segment)
    dynamic = {0: (0.0, ())}
    for index in ordered:
        scores = state[index]["block_scores"]
        cumulative = [0.0]
        for score in scores:
            cumulative.append(cumulative[-1] + score)
        next_dynamic = {}
        for used, (total_score, choices) in dynamic.items():
            max_take = min(len(scores), block_budget - used)
            for take in range(max_take + 1):
                candidate = (total_score + cumulative[take], choices + (take,))
                new_used = used + take
                incumbent = next_dynamic.get(new_used)
                if incumbent is None or candidate[0] > incumbent[0] + 1e-12:
                    next_dynamic[new_used] = candidate
                elif (
                    incumbent is not None
                    and abs(candidate[0] - incumbent[0]) <= 1e-12
                    and candidate[1] < incumbent[1]
                ):
                    next_dynamic[new_used] = candidate
        dynamic = next_dynamic

    best_used, (best_score, best_choices) = min(
        dynamic.items(),
        key=lambda item: (-item[1][0], item[0], item[1][1]),
    )
    del best_used, best_score
    selected = []
    for index, blocks in zip(ordered, best_choices):
        if blocks <= 0:
            continue
        item = state[index]
        selected.append(
            SelectedSegmentPrefix(
                segment_index=index,
                prefix_tokens=item["base"] + blocks * block_tokens,
                accumulated_score=sum(item["block_scores"][:blocks]),
            )
        )
    return tuple(selected)


def allocate_checkpoint_cell_islands(
    cells: Iterable[CheckpointCellCandidates],
    *,
    token_budget_tokens: int,
    block_tokens: int = 128,
    checkpoint_stride: int = 512,
    max_islands: int | None = None,
    max_tokens_by_segment: Mapping[int, int] | None = None,
    bridges: Iterable[CheckpointBridgeCandidates] = (),
) -> Tuple[SelectedCheckpointIsland, ...]:
    """Allocate a request-global budget to checkpoint-safe sparse islands.

    A bridge contributes prefix choices for the rows after a migrated segment's
    mandatory first block and before its first checkpoint.  Bridge blocks and
    independently restorable cells compete in this same exact dynamic program.
    A bridge is contiguous with already-live state, so selecting it consumes
    tokens and a per-segment cap but never consumes a restore-island slot.  A
    full bridge also lets a selected cell at the first checkpoint continue the
    live prefix without a descriptor.

    Every checkpoint cell contributes multiple-choice prefix options: select
    zero blocks, its first block, its first two blocks, and so on.  Selecting a
    later block without the earlier blocks in the same cell is forbidden.  A
    checkpoint at every cell boundary makes choices in different cells
    independent, unlike :func:`allocate_request_global_prefixes`, which must
    rebuild a whole segment prefix.

    The exact dynamic program maximizes summed utility under one global token
    budget.  Token cost is the number of selected 128-token blocks; any budget
    remainder smaller than a block is intentionally unusable.  ``max_islands``
    limits the number of restore descriptors after adjacent cell selections are
    merged; the live mandatory/bridge prefix does not consume this limit.  It is
    also a simple proxy for fixed checkpoint-restore and launch overhead, which
    are not otherwise represented by the token cost model.
    ``max_tokens_by_segment`` applies aligned per-segment replay caps *inside*
    the dynamic program.  Enforcing those caps only after allocation can reject
    a request even when a lower-scoring, valid cross-segment allocation exists.
    Mandatory query/boundary rows are outside this allocator; callers should
    subtract them before passing the remaining request-global budget.  The
    caller must also ensure that a compatible state checkpoint exists at every
    supplied cell boundary.

    Ties are deterministic: prefer fewer selected tokens, then fewer restore
    islands, then selections in earlier segment-relative candidates.  Returned
    token offsets are relative to the start of each segment.
    """

    if block_tokens <= 0:
        raise ValueError("block size must be positive")
    if checkpoint_stride < 512 or checkpoint_stride % 512 != 0:
        raise ValueError("checkpoint_stride must be a positive multiple of 512")
    if checkpoint_stride % block_tokens != 0:
        raise ValueError("checkpoint_stride must be divisible by block_tokens")
    if token_budget_tokens < 0:
        raise ValueError("token_budget_tokens must be non-negative")
    if max_islands is not None and not 1 <= max_islands <= MAX_CHECKPOINT_ISLANDS:
        raise ValueError("max_islands must be in [1, 256] or None")

    segment_cap_blocks: Dict[int, int] = {}
    if max_tokens_by_segment is not None:
        for raw_segment_index, raw_cap in max_tokens_by_segment.items():
            segment_index = int(raw_segment_index)
            cap_tokens = int(raw_cap)
            if segment_index < 0 or cap_tokens < 0:
                raise ValueError("per-segment token caps must be non-negative")
            if cap_tokens % block_tokens != 0:
                raise ValueError("per-segment token caps must be block-aligned")
            segment_cap_blocks[segment_index] = cap_tokens // block_tokens

    blocks_per_cell = checkpoint_stride // block_tokens
    normalized = []
    bridge_segments = set()
    for bridge in bridges:
        segment_index = int(bridge.segment_index)
        if segment_index < 0:
            raise ValueError("bridge segment index must be non-negative")
        if segment_index in bridge_segments:
            raise ValueError(f"duplicate checkpoint bridge: {segment_index}")
        bridge_segments.add(segment_index)

        scores = tuple(float(raw_score) for raw_score in bridge.block_scores)
        max_bridge_blocks = blocks_per_cell - 1
        if not scores or len(scores) > max_bridge_blocks:
            raise ValueError(
                f"checkpoint bridge {segment_index} must contain between 1 and "
                f"{max_bridge_blocks} block scores"
            )
        if any(not math.isfinite(score) or score < 0.0 for score in scores):
            raise ValueError(
                f"checkpoint bridge {segment_index} has invalid block score"
            )
        cumulative = [0.0]
        for score in scores:
            cumulative.append(cumulative[-1] + score)
        token_begin = block_tokens
        token_limit = token_begin + len(scores) * block_tokens
        normalized.append(
            (
                segment_index,
                token_begin,
                token_limit,
                scores,
                tuple(cumulative),
                True,
            )
        )

    seen = set()
    for cell in cells:
        segment_index = int(cell.segment_index)
        cell_index = int(cell.cell_index)
        identity = (segment_index, cell_index)
        if segment_index < 0 or cell_index < 0:
            raise ValueError(
                "segment and checkpoint cell indices must be non-negative"
            )
        if identity in seen:
            raise ValueError(f"duplicate checkpoint cell: {identity}")
        seen.add(identity)
        if cell_index == 0 and segment_index in bridge_segments:
            raise ValueError(
                f"checkpoint cell {identity} overlaps the mandatory/bridge prefix"
            )

        scores = tuple(float(raw_score) for raw_score in cell.block_scores)
        if not scores or len(scores) > blocks_per_cell:
            raise ValueError(
                f"checkpoint cell {identity} must contain between 1 and "
                f"{blocks_per_cell} block scores"
            )
        if any(not math.isfinite(score) or score < 0.0 for score in scores):
            raise ValueError(f"checkpoint cell {identity} has invalid block score")
        cumulative = [0.0]
        for score in scores:
            cumulative.append(cumulative[-1] + score)
        token_begin = cell_index * checkpoint_stride
        token_limit = token_begin + len(scores) * block_tokens
        normalized.append(
            (
                segment_index,
                token_begin,
                token_limit,
                scores,
                tuple(cumulative),
                False,
            )
        )

    normalized.sort(key=lambda item: (item[0], item[1]))
    if not normalized:
        return ()
    previous_span = None
    for segment_index, token_begin, token_limit, _, _, _ in normalized:
        if (
            previous_span is not None
            and previous_span[0] == segment_index
            and token_begin < previous_span[1]
        ):
            raise ValueError(
                f"overlapping checkpoint candidates in segment {segment_index}"
            )
        previous_span = (segment_index, token_limit)

    block_budget = token_budget_tokens // block_tokens
    island_limit = len(normalized) if max_islands is None else int(max_islands)
    if block_budget <= 0:
        return ()

    # State is (used_blocks, merged_islands, reaches_next_checkpoint,
    # current_segment_blocks).  The final component resets at segment changes
    # and makes a per-segment cap exact without carrying a vector of every
    # segment's usage through the global DP.
    # record stores (score, base-N choice code).  Encoding choices as one integer
    # avoids copying a growing tuple through every DP transition while retaining
    # a deterministic lexicographic tie-break in favour of earlier cells.
    choice_base = blocks_per_cell + 1
    states: Dict[Tuple[int, int, bool, int], Tuple[float, int]] = {
        (0, 0, False, 0): (0.0, 0)
    }
    previous_identity = None
    for (
        segment_index,
        token_begin,
        token_limit,
        scores,
        cumulative,
        is_bridge,
    ) in normalized:
        starts_new_segment = (
            previous_identity is None or previous_identity[0] != segment_index
        )
        adjacent_to_previous = (
            previous_identity is not None
            and previous_identity[0] == segment_index
            and previous_identity[1] == token_begin
        )
        next_states: Dict[Tuple[int, int, bool, int], Tuple[float, int]] = {}
        for (used, islands, open_to_next, prior_segment_used), (
            total_score,
            choice_code,
        ) in states.items():
            segment_used = 0 if starts_new_segment else prior_segment_used
            merges_previous = open_to_next and adjacent_to_previous
            max_take = min(len(scores), block_budget - used)
            if segment_index in segment_cap_blocks:
                max_take = min(
                    max_take,
                    segment_cap_blocks[segment_index] - segment_used,
                )
            for take in range(max_take + 1):
                selected = take > 0
                next_used = used + take
                if is_bridge:
                    # The bridge starts at the mandatory live prefix and never
                    # needs a checkpoint restore descriptor of its own.
                    next_islands = islands
                else:
                    next_islands = islands + (
                        0 if selected and merges_previous else int(selected)
                    )
                if next_islands > island_limit:
                    continue
                # A candidate can touch the next cell only when the selected
                # prefix reaches an actual checkpoint boundary.  This covers a
                # complete cell and a complete [128, checkpoint_stride) bridge.
                selected_end = token_begin + take * block_tokens
                next_open = (
                    selected
                    and take == len(scores)
                    and selected_end == token_limit
                    and selected_end % checkpoint_stride == 0
                )
                key = (
                    next_used,
                    next_islands,
                    next_open,
                    segment_used + take,
                )
                candidate = (
                    total_score + cumulative[take],
                    choice_code * choice_base + take,
                )
                incumbent = next_states.get(key)
                if (
                    incumbent is None
                    or candidate[0] > incumbent[0] + 1e-12
                    or (
                        abs(candidate[0] - incumbent[0]) <= 1e-12
                        and candidate[1] > incumbent[1]
                    )
                ):
                    next_states[key] = candidate
        states = next_states
        previous_identity = (segment_index, token_limit)

    best_state = None
    best_record = None
    for state, record in states.items():
        if best_record is None:
            best_state, best_record = state, record
            continue
        used, islands, _, _ = state
        best_used, best_islands, _, _ = best_state
        if (
            record[0] > best_record[0] + 1e-12
            or (
                abs(record[0] - best_record[0]) <= 1e-12
                and (
                    used < best_used
                    or (
                        used == best_used
                        and (
                            islands < best_islands
                            or (
                                islands == best_islands
                                and record[1] > best_record[1]
                            )
                        )
                    )
                )
            )
        ):
            best_state, best_record = state, record

    assert best_record is not None
    choices = [0] * len(normalized)
    choice_code = best_record[1]
    for choice_index in range(len(choices) - 1, -1, -1):
        choices[choice_index] = choice_code % choice_base
        choice_code //= choice_base

    selected = []
    for (
        segment_index,
        token_begin,
        _,
        _,
        cumulative,
        _,
    ), take in zip(normalized, choices):
        if take <= 0:
            continue
        token_end = token_begin + take * block_tokens
        score = cumulative[take]
        if (
            selected
            and selected[-1].segment_index == segment_index
            and selected[-1].token_end == token_begin
        ):
            previous = selected[-1]
            selected[-1] = SelectedCheckpointIsland(
                segment_index=segment_index,
                token_begin=previous.token_begin,
                token_end=token_end,
                accumulated_score=previous.accumulated_score + score,
            )
        else:
            selected.append(
                SelectedCheckpointIsland(
                    segment_index=segment_index,
                    token_begin=token_begin,
                    token_end=token_end,
                    accumulated_score=score,
                )
            )
    return tuple(selected)


def allocate_checkpoint_cell_islands_fast(
    cells: Iterable[CheckpointCellCandidates],
    *,
    token_budget_tokens: int,
    block_tokens: int = 128,
    checkpoint_stride: int = 512,
    max_islands: int | None = None,
    max_tokens_by_segment: Mapping[int, int] | None = None,
    bridges: Iterable[CheckpointBridgeCandidates] = (),
) -> Tuple[SelectedCheckpointIsland, ...]:
    """Bounded-latency allocator for the explicitly approximate online arm.

    The exact allocator above carries budget, island count, continuation state,
    per-segment use and a base-N choice code through every candidate.  At 256K
    that state space is unsuitable for a TTFT hot path.  This allocator keeps
    the same safety constraints but uses deterministic marginal prefix
    allocation.  Each checkpoint cell is conservatively charged as one island;
    adjacent choices are merged afterwards, so the returned descriptor count
    can only be smaller than the enforced limit.

    It is intentionally a separate API: formal/exact callers continue to use
    :func:`allocate_checkpoint_cell_islands`.  The model runner selects this
    path only for a plan carrying the explicit ``row_sparse_closure`` marker.
    """

    if block_tokens <= 0:
        raise ValueError("block size must be positive")
    if checkpoint_stride < 512 or checkpoint_stride % 512 != 0:
        raise ValueError("checkpoint_stride must be a positive multiple of 512")
    if checkpoint_stride % block_tokens != 0:
        raise ValueError("checkpoint_stride must be divisible by block_tokens")
    if token_budget_tokens < 0:
        raise ValueError("token_budget_tokens must be non-negative")
    if max_islands is not None and not 1 <= max_islands <= MAX_CHECKPOINT_ISLANDS:
        raise ValueError("max_islands must be in [1, 256] or None")

    blocks_per_cell = checkpoint_stride // block_tokens
    segment_cap_blocks: Dict[int, int] = {}
    if max_tokens_by_segment is not None:
        for raw_segment_index, raw_cap in max_tokens_by_segment.items():
            segment_index = int(raw_segment_index)
            cap_tokens = int(raw_cap)
            if segment_index < 0 or cap_tokens < 0:
                raise ValueError("per-segment token caps must be non-negative")
            if cap_tokens % block_tokens != 0:
                raise ValueError("per-segment token caps must be block-aligned")
            segment_cap_blocks[segment_index] = cap_tokens // block_tokens

    # (segment, begin, limit, scores, cumulative, is_bridge)
    normalized = []
    bridge_segments = set()
    for bridge in bridges:
        segment_index = int(bridge.segment_index)
        if segment_index < 0 or segment_index in bridge_segments:
            raise ValueError("invalid or duplicate checkpoint bridge")
        bridge_segments.add(segment_index)
        scores = tuple(float(value) for value in bridge.block_scores)
        if not scores or len(scores) >= blocks_per_cell:
            raise ValueError("checkpoint bridge has an invalid block count")
        if any(not math.isfinite(score) or score < 0.0 for score in scores):
            raise ValueError("checkpoint bridge has an invalid block score")
        cumulative = [0.0]
        for score in scores:
            cumulative.append(cumulative[-1] + score)
        begin = block_tokens
        normalized.append(
            (
                segment_index,
                begin,
                begin + len(scores) * block_tokens,
                scores,
                tuple(cumulative),
                True,
            )
        )

    seen = set()
    for cell in cells:
        segment_index = int(cell.segment_index)
        cell_index = int(cell.cell_index)
        identity = (segment_index, cell_index)
        if segment_index < 0 or cell_index < 0 or identity in seen:
            raise ValueError("invalid or duplicate checkpoint cell")
        seen.add(identity)
        if cell_index == 0 and segment_index in bridge_segments:
            raise ValueError("checkpoint cell overlaps a bridge")
        scores = tuple(float(value) for value in cell.block_scores)
        if not scores or len(scores) > blocks_per_cell:
            raise ValueError("checkpoint cell has an invalid block count")
        if any(not math.isfinite(score) or score < 0.0 for score in scores):
            raise ValueError("checkpoint cell has an invalid block score")
        cumulative = [0.0]
        for score in scores:
            cumulative.append(cumulative[-1] + score)
        begin = cell_index * checkpoint_stride
        normalized.append(
            (
                segment_index,
                begin,
                begin + len(scores) * block_tokens,
                scores,
                tuple(cumulative),
                False,
            )
        )

    normalized.sort(key=lambda item: (item[0], item[1]))
    previous = None
    for segment_index, begin, limit, _, _, _ in normalized:
        if previous is not None and previous[0] == segment_index and begin < previous[1]:
            raise ValueError("overlapping checkpoint candidates")
        previous = (segment_index, limit)
    if not normalized:
        return ()

    remaining = token_budget_tokens // block_tokens
    if remaining <= 0:
        return ()
    island_limit = len(normalized) if max_islands is None else int(max_islands)
    takes = [0] * len(normalized)
    segment_used: Dict[int, int] = {}
    opened_cells = 0
    opened_cells_by_segment: Dict[int, int] = {}

    while remaining > 0:
        best = None
        for candidate_index, item in enumerate(normalized):
            segment_index, begin, _, scores, cumulative, is_bridge = item
            current = takes[candidate_index]
            if current >= len(scores):
                continue
            if current == 0 and not is_bridge and opened_cells >= island_limit:
                continue
            segment_remaining = remaining
            if segment_index in segment_cap_blocks:
                segment_remaining = min(
                    segment_remaining,
                    segment_cap_blocks[segment_index]
                    - segment_used.get(segment_index, 0),
                )
            maximum = min(len(scores), current + segment_remaining)
            for target in range(current + 1, maximum + 1):
                added = target - current
                gain = cumulative[target] - cumulative[current]
                # Prefer utility density, then absolute utility, fewer added
                # rows, and finally the earlier canonical candidate.
                priority = (
                    gain / added,
                    gain,
                    -opened_cells_by_segment.get(segment_index, 0),
                    -added,
                    -segment_index,
                    -begin,
                    -target,
                )
                if best is None or priority > best[0]:
                    best = (priority, candidate_index, target, gain)
        if best is None or best[3] <= 0.0:
            break
        _, candidate_index, target, _ = best
        segment_index, _, _, _, _, is_bridge = normalized[candidate_index]
        current = takes[candidate_index]
        added = target - current
        if current == 0 and not is_bridge:
            opened_cells += 1
            opened_cells_by_segment[segment_index] = (
                opened_cells_by_segment.get(segment_index, 0) + 1
            )
        takes[candidate_index] = target
        segment_used[segment_index] = segment_used.get(segment_index, 0) + added
        remaining -= added

    selected = []
    for item, take in zip(normalized, takes):
        if take <= 0:
            continue
        segment_index, begin, _, _, cumulative, _ = item
        end = begin + take * block_tokens
        score = cumulative[take]
        if (
            selected
            and selected[-1].segment_index == segment_index
            and selected[-1].token_end == begin
        ):
            prior = selected[-1]
            selected[-1] = SelectedCheckpointIsland(
                segment_index=segment_index,
                token_begin=prior.token_begin,
                token_end=end,
                accumulated_score=prior.accumulated_score + score,
            )
        else:
            selected.append(
                SelectedCheckpointIsland(
                    segment_index=segment_index,
                    token_begin=begin,
                    token_end=end,
                    accumulated_score=score,
                )
            )
    return tuple(selected)


def materialize_checkpoint_replay_layout(
    selected: Iterable[SelectedCheckpointIsland],
    *,
    base_prefix_tokens_by_segment: Mapping[int, int],
    block_tokens: int = 128,
    checkpoint_stride: int = 512,
) -> CheckpointReplayLayout:
    """Partition selected ranges into live prefixes and restore descriptors.

    Any selected interval that starts exactly where a segment's live prefix
    ends extends that prefix and needs no descriptor.  Every non-prefix interval
    must begin on a checkpoint anchor.  Adjacent or overlapping intervals are
    rejected because the allocator must merge them before this materialization
    step.  These checks make descriptor coverage exactly match the selected rows
    without duplicated or silently missing ranges.
    """

    if block_tokens <= 0:
        raise ValueError("block size must be positive")
    if checkpoint_stride < 512 or checkpoint_stride % 512 != 0:
        raise ValueError("checkpoint_stride must be a positive multiple of 512")
    if checkpoint_stride % block_tokens != 0:
        raise ValueError("checkpoint_stride must be divisible by block_tokens")

    prefixes: Dict[int, int] = {}
    for raw_segment_index, raw_prefix in base_prefix_tokens_by_segment.items():
        segment_index = int(raw_segment_index)
        prefix_tokens = int(raw_prefix)
        if segment_index < 0 or prefix_tokens < 0:
            raise ValueError("base prefixes must be non-negative")
        if prefix_tokens % block_tokens != 0:
            raise ValueError("base prefixes must be block-aligned")
        prefixes[segment_index] = prefix_tokens

    ordered = sorted(
        selected,
        key=lambda item: (int(item.segment_index), int(item.token_begin)),
    )
    restore_islands = []
    previous_end_by_segment = dict(prefixes)
    seen_begins = set()
    for item in ordered:
        segment_index = int(item.segment_index)
        token_begin = int(item.token_begin)
        token_end = int(item.token_end)
        identity = (segment_index, token_begin)
        if segment_index not in prefixes:
            raise ValueError(f"selected range has unknown segment {segment_index}")
        if identity in seen_begins:
            raise ValueError(f"duplicate selected range: {identity}")
        seen_begins.add(identity)
        if (
            token_begin < 0
            or token_end <= token_begin
            or token_begin % block_tokens != 0
            or token_end % block_tokens != 0
        ):
            raise ValueError("selected ranges must be positive and block-aligned")

        online_prefix = prefixes[segment_index]
        previous_end = previous_end_by_segment[segment_index]
        if token_begin == online_prefix:
            if previous_end != online_prefix:
                raise ValueError("live-prefix extension appears after a restore island")
            prefixes[segment_index] = token_end
            previous_end_by_segment[segment_index] = token_end
            continue
        if token_begin < previous_end:
            raise ValueError("selected checkpoint ranges overlap")
        if token_begin - previous_end < block_tokens:
            raise ValueError("adjacent checkpoint ranges were not merged")
        if token_begin <= 0 or token_begin % checkpoint_stride != 0:
            raise ValueError("restore island does not begin on a checkpoint anchor")
        restore_islands.append(item)
        previous_end_by_segment[segment_index] = token_end

    return CheckpointReplayLayout(
        selected_prefix_tokens=tuple(sorted(prefixes.items())),
        restore_islands=tuple(restore_islands),
    )


__all__ = [
    "CheckpointBridgeCandidates",
    "CheckpointCellCandidates",
    "CheckpointReplayLayout",
    "MAX_CHECKPOINT_ISLANDS",
    "SegmentPrefixCandidates",
    "SelectedCheckpointIsland",
    "SelectedSegmentPrefix",
    "allocate_checkpoint_cell_islands",
    "allocate_checkpoint_cell_islands_fast",
    "allocate_request_global_prefixes",
    "checkpoint_effective_segment_cap_tokens",
    "checkpoint_mandatory_prefix_tokens",
    "materialize_checkpoint_replay_layout",
]
