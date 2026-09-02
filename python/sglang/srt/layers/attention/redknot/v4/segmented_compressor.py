from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple

from sglang.srt.layers.attention.redknot.v4.boundary_replay import (
    RequestBoundaryReplay,
)


class CompressorEventKind(str, Enum):
    ONLINE_RANGE = "online_range"
    OFFLINE_TAIL = "offline_tail"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass(frozen=True)
class SegmentedCompressorEvent:
    kind: CompressorEventKind
    token_begin: int
    token_end: int
    row_begin: int
    row_end: int
    segment_index: int = -1
    checkpoint_anchor: int = -1


@dataclass(frozen=True)
class SegmentedCompressorSchedule:
    compress_ratio: int
    chunk_token_range: Tuple[int, int]
    events: Tuple[SegmentedCompressorEvent, ...]

    @property
    def online_rows(self) -> int:
        return sum(
            event.row_end - event.row_begin
            for event in self.events
            if event.kind == CompressorEventKind.ONLINE_RANGE
        )


def select_segment_output_locations(
    output_locations: Any,
    *,
    row_begin: int,
    row_end: int,
    total_rows: int,
):
    """Return the per-token cache locations for one online compressor event.

    ``FusedCompressMetadata.write_loc`` addresses the compressor's recurrent
    state and therefore has one entry per request.  The C4/C128 locations on
    the core attention metadata instead have one entry per online token row.
    Keeping this distinction explicit prevents a single state location from
    being (incorrectly) used to scatter a multi-row compressed output.
    """

    if output_locations is None:
        raise ValueError("compressed cache output locations are missing")
    if getattr(output_locations, "ndim", None) != 1:
        raise ValueError("compressed cache output locations must be one-dimensional")
    location_rows = int(output_locations.numel())
    if location_rows != total_rows:
        raise ValueError(
            "compressed cache output locations do not match online rows: "
            f"locations={location_rows} rows={total_rows}"
        )
    if not 0 <= row_begin < row_end <= total_rows:
        raise ValueError(
            f"invalid compressor row range [{row_begin}, {row_end}) for {total_rows} rows"
        )
    selected = output_locations[row_begin:row_end]
    if int(selected.numel()) != row_end - row_begin:
        raise ValueError("compressed cache output location slice has the wrong length")
    return selected


def validate_complete_online_row_coverage(
    schedule: SegmentedCompressorSchedule, *, total_rows: int
) -> None:
    """Require selected-row schedules to cover every retained row exactly once."""

    cursor = 0
    for event in schedule.events:
        if event.kind != CompressorEventKind.ONLINE_RANGE:
            continue
        if event.row_begin != cursor or event.row_end <= event.row_begin:
            raise ValueError(
                "selected-row compressor schedule has a gap or overlap: "
                f"expected row {cursor}, got [{event.row_begin}, {event.row_end})"
            )
        if event.token_end - event.token_begin != event.row_end - event.row_begin:
            raise ValueError(
                "online compressor event must map contiguous token and row ranges"
            )
        cursor = event.row_end
    if cursor != total_rows:
        raise ValueError(
            "selected-row compressor schedule does not cover all retained rows: "
            f"covered={cursor} rows={total_rows}"
        )


def _intersect(left: Tuple[int, int], right: Tuple[int, int]) -> Tuple[int, int] | None:
    begin = max(left[0], right[0])
    end = min(left[1], right[1])
    return (begin, end) if begin < end else None


def build_segmented_compressor_schedule(
    *,
    replay: RequestBoundaryReplay,
    positions: Sequence[int],
    compress_ratio: int,
    include_all_present_rows: bool = False,
    logical_chunk_range: Tuple[int, int] | None = None,
    checkpoint_islands: Sequence[Mapping[str, int]] = (),
) -> SegmentedCompressorSchedule:
    """Build ordered online-range and offline-tail events for one prefill chunk.

    ``positions`` are the *global token positions* of the rows actually present
    in this forward pass, in row order. For dense/contiguous prefill they form a
    contiguous run and row = position - positions[0]. For indexer-hot aggressive
    replay the retained rows are non-contiguous in position (the forward tensor
    was index_select'd), so row is simply the index within ``positions`` and the
    position->row map is built explicitly. An online range intersected with the
    present positions may split into several contiguous-row spans; each becomes
    its own ONLINE_RANGE event (compressor still slices x[row_begin:row_end]).

    ``logical_chunk_range`` is the scheduler chunk before selected-row pruning.
    A partially online segment owns its offline terminal state in the logical
    chunk that completes the segment, even when the last retained online row is
    far before that boundary.  Without this range the restore is delayed until a
    later retained row happens to cross the segment end, and can be missed
    entirely by different chunking/query layouts.
    """

    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compressor ratio: {compress_ratio}")
    if not positions:
        return SegmentedCompressorSchedule(compress_ratio, (0, 0), ())
    normalized_positions = [int(position) for position in positions]
    if normalized_positions[0] < 0 or any(
        right <= left
        for left, right in zip(normalized_positions, normalized_positions[1:])
    ):
        raise ValueError("positions must be non-negative and strictly increasing")
    chunk_range = (normalized_positions[0], normalized_positions[-1] + 1)
    tail_ownership_range = chunk_range
    if logical_chunk_range is not None:
        logical_begin, logical_end = map(int, logical_chunk_range)
        if (
            logical_begin < 0
            or logical_begin >= logical_end
            or normalized_positions[0] < logical_begin
            or normalized_positions[-1] >= logical_end
        ):
            raise ValueError(
                "logical chunk range must be non-empty and contain all positions"
            )
        tail_ownership_range = (logical_begin, logical_end)

    # position -> row (row is the index within the present rows).
    pos_to_row = {pos: row for row, pos in enumerate(normalized_positions)}
    present = set(normalized_positions)

    events = []
    if include_all_present_rows:
        # After selected-row pruning every present row is intentional online
        # work: mandatory SWA boundary/query rows plus Indexer-selected C4 units.
        # Cover the whole chunk range and let ``present`` split it into the exact
        # contiguous repair islands.  Without this branch interior hot rows flow
        # through the transformer but never update compressed KV/Indexer state.
        online_ranges = [chunk_range]
    else:
        online_ranges = [replay.online_prefix_range]
        online_ranges.extend(segment.online_token_range for segment in replay.segments)
        online_ranges.append(replay.online_query_range)
    for token_range in online_ranges:
        intersection = _intersect(token_range, chunk_range)
        if intersection is None:
            continue
        begin, end = intersection
        # Walk the intersection and emit maximal contiguous-row spans over the
        # positions that are actually present. Contiguous in *row* means the
        # present positions are consecutive rows (rows are dense by construction,
        # so any present position advances the row by exactly 1).
        run_start_pos = None
        prev_row = None
        for pos in range(begin, end):
            if pos not in present:
                # gap: flush current run
                if run_start_pos is not None:
                    events.append(
                        SegmentedCompressorEvent(
                            kind=CompressorEventKind.ONLINE_RANGE,
                            token_begin=run_start_pos,
                            token_end=prev_row_pos + 1,
                            row_begin=pos_to_row[run_start_pos],
                            row_end=pos_to_row[prev_row_pos] + 1,
                        )
                    )
                    run_start_pos = None
                continue
            row = pos_to_row[pos]
            if run_start_pos is None:
                run_start_pos = pos
                prev_row = row
                prev_row_pos = pos
            elif row == prev_row + 1:
                prev_row = row
                prev_row_pos = pos
            else:
                # non-adjacent rows: flush and restart
                events.append(
                    SegmentedCompressorEvent(
                        kind=CompressorEventKind.ONLINE_RANGE,
                        token_begin=run_start_pos,
                        token_end=prev_row_pos + 1,
                        row_begin=pos_to_row[run_start_pos],
                        row_end=pos_to_row[prev_row_pos] + 1,
                    )
                )
                run_start_pos = pos
                prev_row = row
                prev_row_pos = pos
        if run_start_pos is not None:
            events.append(
                SegmentedCompressorEvent(
                    kind=CompressorEventKind.ONLINE_RANGE,
                    token_begin=run_start_pos,
                    token_end=prev_row_pos + 1,
                    row_begin=pos_to_row[run_start_pos],
                    row_end=pos_to_row[prev_row_pos] + 1,
                )
            )

    for segment_index, segment in enumerate(replay.segments):
        tail_position = segment.offline_token_range[1]
        owns_tail = (
            tail_ownership_range[0] < tail_position <= tail_ownership_range[1]
            if logical_chunk_range is not None
            else chunk_range[0] <= tail_position < chunk_range[1]
        )
        if owns_tail:
            # The offline tail marks the segment's compressed-state restore point.
            # Its row is only meaningful if that position is present; otherwise the
            # tail row is the nearest present row <= tail_position (state restore
            # is position-addressed, not row-addressed downstream).
            row = pos_to_row.get(tail_position)
            if row is None:
                # nearest present row below tail (state restore uses token pos)
                row = 0
            events.append(
                SegmentedCompressorEvent(
                    kind=CompressorEventKind.OFFLINE_TAIL,
                    token_begin=tail_position,
                    token_end=tail_position,
                    row_begin=row,
                    row_end=row,
                    segment_index=segment_index,
                )
            )

    for island in checkpoint_islands:
        try:
            segment_index = int(island["segment_index"])
            global_begin = int(island["global_begin"])
            global_end = int(island["global_end"])
            checkpoint_anchor = int(island["checkpoint_anchor"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid checkpoint island metadata: {error}") from error
        if not 0 <= segment_index < len(replay.segments):
            raise ValueError("checkpoint island has an invalid segment index")
        if (
            checkpoint_anchor <= 0
            or checkpoint_anchor % 128 != 0
            or global_begin % 128 != 0
            or global_end % 128 != 0
            or global_begin >= global_end
        ):
            raise ValueError("checkpoint island is not positive/128-aligned")
        segment = replay.segments[segment_index]
        segment_begin = segment.online_token_range[0]
        segment_end = segment.offline_token_range[1]
        if (
            global_begin != segment_begin + checkpoint_anchor
            or global_end > segment_end
        ):
            raise ValueError("checkpoint island is outside its source segment")
        # An island split by scheduler chunking restores its carry exactly once,
        # in the logical chunk that owns the island's first token.  Later chunks
        # continue from the state written by the preceding chunk.
        if not (
            tail_ownership_range[0]
            <= global_begin
            < tail_ownership_range[1]
        ):
            continue
        if global_begin not in present:
            raise ValueError("checkpoint island start is absent from its owner chunk")
        containing_online = [
            event
            for event in events
            if event.kind == CompressorEventKind.ONLINE_RANGE
            and event.token_begin <= global_begin < event.token_end
        ]
        if (
            len(containing_online) != 1
            or containing_online[0].token_begin != global_begin
        ):
            raise ValueError(
                "checkpoint restore must begin its corresponding online range"
            )
        row = pos_to_row[global_begin]
        events.append(
            SegmentedCompressorEvent(
                kind=CompressorEventKind.CHECKPOINT_RESTORE,
                token_begin=global_begin,
                token_end=global_begin,
                row_begin=row,
                row_end=row,
                segment_index=segment_index,
                checkpoint_anchor=checkpoint_anchor,
            )
        )

    events.sort(
        key=lambda event: (
            event.token_begin,
            {
                CompressorEventKind.OFFLINE_TAIL: 0,
                CompressorEventKind.CHECKPOINT_RESTORE: 1,
                CompressorEventKind.ONLINE_RANGE: 2,
            }[event.kind],
        )
    )
    return SegmentedCompressorSchedule(
        compress_ratio=compress_ratio,
        chunk_token_range=chunk_range,
        events=tuple(events),
    )


__all__ = [
    "CompressorEventKind",
    "SegmentedCompressorEvent",
    "SegmentedCompressorSchedule",
    "build_segmented_compressor_schedule",
    "select_segment_output_locations",
    "validate_complete_online_row_coverage",
]
