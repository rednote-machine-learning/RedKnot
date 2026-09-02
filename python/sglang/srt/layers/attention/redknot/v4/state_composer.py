from __future__ import annotations

from typing import Iterable, Mapping, Set

from sglang.srt.layers.attention.redknot.v4.types import (
    BlockValidity,
    StateSource,
)


def compose_block_validity(
    *,
    offline_blocks: Iterable[int],
    invalid_offline_blocks: Iterable[int],
    online_blocks: Mapping[int, StateSource],
) -> list[BlockValidity]:
    """Compose one effective version per logical compressed block."""

    offline: Set[int] = {int(x) for x in offline_blocks}
    invalid: Set[int] = {int(x) for x in invalid_offline_blocks}
    online = {int(block): source for block, source in online_blocks.items()}
    all_blocks = sorted(offline | set(online))
    result = []
    for block in all_blocks:
        if block in online:
            source = online[block]
            result.append(
                BlockValidity(
                    logical_block=block,
                    offline_present=block in offline,
                    offline_valid=block in offline and block not in invalid,
                    online_present=True,
                    source=source,
                    reason="online override",
                )
            )
        elif block in invalid:
            result.append(
                BlockValidity(
                    logical_block=block,
                    offline_present=True,
                    offline_valid=False,
                    online_present=False,
                    source=StateSource.OFFLINE,
                    reason="invalid offline block has no online replacement",
                )
            )
        else:
            result.append(
                BlockValidity(
                    logical_block=block,
                    offline_present=True,
                    offline_valid=True,
                    online_present=False,
                    source=StateSource.OFFLINE,
                    reason="valid offline block",
                )
            )

    effective = [x.logical_block for x in result if x.online_present or x.offline_valid]
    if len(effective) != len(set(effective)):
        raise RuntimeError("effective state contains duplicate logical blocks")
    return result


__all__ = ["compose_block_validity"]
