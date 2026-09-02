from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple


@dataclass(frozen=True)
class SegmentReplayMask:
    online_token_range: Tuple[int, int]
    offline_token_range: Tuple[int, int]
    online_c4_blocks: Tuple[int, int]
    offline_c4_blocks: Tuple[int, int]
    online_c128_blocks: Tuple[int, int]
    offline_c128_blocks: Tuple[int, int]


@dataclass(frozen=True)
class RequestBoundaryReplay:
    segments: Tuple[SegmentReplayMask, ...]
    online_prefix_range: Tuple[int, int]
    online_query_range: Tuple[int, int]
    total_tokens: int

    @property
    def online_token_count(self) -> int:
        ranges = [self.online_prefix_range, self.online_query_range]
        ranges.extend(segment.online_token_range for segment in self.segments)
        return sum(max(0, end - begin) for begin, end in ranges)


def _block_range(token_begin: int, token_end: int, ratio: int) -> Tuple[int, int]:
    if token_begin % ratio != 0 or token_end % ratio != 0:
        raise ValueError(
            f"token range [{token_begin}, {token_end}) is not aligned to ratio {ratio}"
        )
    return token_begin // ratio, token_end // ratio


def build_boundary_replay(
    *,
    segments: Iterable[Mapping[str, int]],
    total_tokens: int,
    boundary_tokens: int = 128,
) -> RequestBoundaryReplay:
    """Build block-aware online/offline ranges for strict Level-A chunks."""

    normalized = sorted(
        (
            int(segment["global_offset"]),
            int(segment["global_offset"]) + int(segment["length"]),
        )
        for segment in segments
    )
    previous_end = 0
    replay_segments = []
    for begin, end in normalized:
        if begin < previous_end:
            raise ValueError("RedKnot V4 segments overlap")
        if begin % 128 != 0 or end % 128 != 0:
            raise ValueError("RedKnot V4 boundary replay requires Level-A segments")
        boundary_end = min(begin + boundary_tokens, end)
        replay_segments.append(
            SegmentReplayMask(
                online_token_range=(begin, boundary_end),
                offline_token_range=(boundary_end, end),
                online_c4_blocks=_block_range(begin, boundary_end, 4),
                offline_c4_blocks=_block_range(boundary_end, end, 4),
                online_c128_blocks=_block_range(begin, boundary_end, 128),
                offline_c128_blocks=_block_range(boundary_end, end, 128),
            )
        )
        previous_end = end

    prefix_end = normalized[0][0] if normalized else total_tokens
    query_begin = normalized[-1][1] if normalized else total_tokens
    if total_tokens < query_begin:
        raise ValueError("total_tokens ends before the last RedKnot segment")
    return RequestBoundaryReplay(
        segments=tuple(replay_segments),
        online_prefix_range=(0, prefix_end),
        online_query_range=(query_begin, int(total_tokens)),
        total_tokens=int(total_tokens),
    )


__all__ = ["RequestBoundaryReplay", "SegmentReplayMask", "build_boundary_replay"]
