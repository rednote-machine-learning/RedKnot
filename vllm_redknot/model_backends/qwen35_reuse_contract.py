# REDKNOT-MODEL: explicit engine-independent Qwen3.5 hybrid reuse preflight.
"""CPU control contract extracted from Qwen3.5 RedKnot prefix-state reuse.

No native pool, ForwardBatch, global server configuration, or tensor runtime
is imported. The adapter provides already-resolved recurrent slot ids and
immutable document-bundle metadata. Independent final GDN states cannot be
concatenated: one ordered bundle is the only supported reuse unit here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

RKBUILD_PREFIX = "__RKBUILD__:"


@dataclass(frozen=True)
class Qwen35ReuseConfig:
    cuda_graph_enabled: bool
    piecewise_cuda_graph_enabled: bool
    prefix_caching_enabled: bool
    multimodal_enabled: bool
    pipeline_parallel_size: int
    data_parallel_attention: bool
    speculative_decoding: bool
    max_model_len: int

    def validate(self) -> None:
        flags = (
            self.cuda_graph_enabled,
            self.piecewise_cuda_graph_enabled,
            self.prefix_caching_enabled,
            self.multimodal_enabled,
            self.data_parallel_attention,
            self.speculative_decoding,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("runtime feature states must be explicit booleans")
        if any(flags):
            raise ValueError(
                "Qwen3.5 prefix-state reuse requires eager text-only execution, "
                "no prefix caching, DP-attention, or speculative decoding"
            )
        if (
            type(self.pipeline_parallel_size) is not int
            or self.pipeline_parallel_size != 1
        ):
            raise ValueError(
                "Qwen3.5 prefix-state reuse requires pipeline parallel size 1"
            )
        if type(self.max_model_len) is not int or self.max_model_len <= 0:
            raise ValueError("max_model_len must be a positive integer")


@dataclass(frozen=True)
class Qwen35PrefixReusePlan:
    segment_ids: tuple[str | None, ...]
    state_slots: tuple[int, ...]
    position_offsets: tuple[int, ...]
    logical_seq_lens: tuple[int, ...]
    restore_rows: tuple[int, ...]


def _nonnegative_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative built-in integers")
    return result


def plan_qwen35_prefix_reuse(
    *,
    segments: Sequence[Sequence[str] | str | None],
    state_slots: Sequence[int],
    seq_lens: Sequence[int],
    prefix_lens: Sequence[int],
    document_lengths: Mapping[str, int],
    loaded_slots: Mapping[int, str],
    is_prefill: bool,
    config: Qwen35ReuseConfig,
) -> Qwen35PrefixReusePlan:
    """Validate all requests before any recurrent-state buffer may be mutated.

    Slot ids must be native-engine-resolved, unique live request slots. A reused
    slot must be removed from ``loaded_slots`` by the owning adapter, including
    slots reassigned to a request that happens to use the same document bundle.
    """
    if not isinstance(config, Qwen35ReuseConfig):
        raise TypeError("an explicit Qwen35ReuseConfig is required")
    if type(is_prefill) is not bool:
        raise TypeError("is_prefill must be a boolean")
    slots = _nonnegative_ints(state_slots, "state_slots")
    lengths = _nonnegative_ints(seq_lens, "seq_lens")
    prefixes = _nonnegative_ints(prefix_lens, "prefix_lens")
    if not (len(segments) == len(slots) == len(lengths) == len(prefixes)):
        raise ValueError("plan, slots, sequence lengths and prefix lengths must align")
    if len(slots) != len(set(slots)):
        raise ValueError("one recurrent slot cannot belong to multiple live requests")
    ids = []
    offsets = []
    restores = []
    for row, raw in enumerate(segments):
        bundle = [] if raw is None else ([raw] if isinstance(raw, str) else list(raw))
        bundle = [sid for sid in bundle if sid is not None]
        if any(not isinstance(sid, str) or not sid for sid in bundle):
            raise ValueError("segment ids must be non-empty strings")
        if not bundle or bundle[0].startswith(RKBUILD_PREFIX):
            ids.append(None)
            offsets.append(0)
            continue
        config.validate()
        if len(bundle) != 1:
            raise ValueError(
                "exactly one ordered document bundle is supported; independent "
                "GDN final states cannot be composed"
            )
        sid = bundle[0]
        doc_len = document_lengths.get(sid)
        if type(doc_len) is not int or doc_len <= 0:
            raise ValueError(f"missing or invalid offline document length: {sid}")
        already_loaded = loaded_slots.get(slots[row]) == sid
        if not already_loaded:
            if not is_prefill:
                raise ValueError(
                    "decode reached without the matching offline GDN state"
                )
            if prefixes[row] != 0:
                raise ValueError(
                    "fresh offline GDN restore cannot combine a live prefix hit"
                )
            restores.append(row)
        ids.append(sid)
        offsets.append(doc_len)
    logical = tuple(length + offset for length, offset in zip(lengths, offsets))
    if any(ids) and any(length > config.max_model_len for length in logical):
        raise ValueError("logical sequence exceeds max_model_len")
    return Qwen35PrefixReusePlan(
        segment_ids=tuple(ids),
        state_slots=slots,
        position_offsets=tuple(offsets),
        logical_seq_lens=logical,
        restore_rows=tuple(restores),
    )
