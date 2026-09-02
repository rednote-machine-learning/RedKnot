"""Pure validation helpers for opt-in RedKnot merged prefill."""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, Tuple


MERGED_PREFILL_ALIGNMENT = 128
# Keep the physical FlashMLA sparse-decode shape below the largest shape that
# has passed the GPU gate.  A single 128K logical pass reached 13,056 selected
# query rows and failed inside get_decoding_sched_meta; 64K (6,528 rows) passed.
# Longer prefixes remain supported as repeated, boundary-aligned 64K passes.
MERGED_PREFILL_MAX_TOKENS = 65_536
MERGED_PREFILL_MAX_LOGICAL_TOKENS = 262_144
MERGED_PREFILL_MIN_SEGMENTS = 4
MERGED_PREFILL_MAX_SEGMENTS = 16
# Seven is the exact online document count for the certified first-document
# prefix workload: document 1 is already resident in radix, while documents
# 2..8 must still execute pure context-bound restore.  Keeping this as an
# explicit count (rather than accepting arbitrary groups) preserves the
# fail-closed scheduler contract.
# Two 32K document segments form the largest already-certified 64K physical
# prefill.  This is the row-sparse 256K execution shape: four large scheduler
# passes replace eight 32K passes without changing the selected-row mask.
MERGED_PREFILL_SEGMENT_COUNTS = (2, 4, 7, 8, 16)
PURE_CONTEXT_PROFILE = "pure_headsplit_context_bound_fullscope_3_37_3_v1"
PURE_INDEPENDENT_RELOCATION_PROFILE = (
    "pure_headsplit_independent_rope_relocation_fullscope_"
    "boundary128_3_37_3_v1"
)
COMBINED_ROW_SPARSE_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)
PURE_CONTEXT_HEAD_SCOPE = "native_dsv4_full_candidate_scope_v1"


def _effective_max_logical_tokens() -> int:
    """Keep H200's certified limit while allowing B300's 16x32K geometry."""

    if os.environ.get("REDKNOT_HARDWARE_PROFILE", "").strip().lower() == "b300":
        return 524_288
    return MERGED_PREFILL_MAX_LOGICAL_TOKENS


def validate_merged_prefill_request(
    requested_tokens: int,
    *,
    segment_tokens: int,
    selection_policy: str,
    num_segments: Optional[int] = None,
    pure_context_bound: bool = False,
    resident_prefix_segments: int = 0,
) -> int:
    """Validate the benchmark/user-facing merged-prefill knob."""

    requested = int(requested_tokens)
    if requested < 0:
        raise ValueError("merged prefill tokens must be non-negative")
    if not requested:
        return 0
    if not pure_context_bound and selection_policy != "checkpoint_islands":
        raise ValueError("merged prefill is only supported by checkpoint_islands")
    if requested > MERGED_PREFILL_MAX_TOKENS:
        raise ValueError("physical merged prefill tokens must not exceed 65536")
    if requested % MERGED_PREFILL_ALIGNMENT != 0:
        raise ValueError("merged prefill tokens must be a multiple of 128")
    segment_tokens = int(segment_tokens)
    if (
        segment_tokens <= 0
        or segment_tokens % MERGED_PREFILL_ALIGNMENT != 0
        or requested % segment_tokens != 0
    ):
        raise ValueError("merged prefill must contain whole segments")
    merged_segments = requested // segment_tokens
    if merged_segments not in MERGED_PREFILL_SEGMENT_COUNTS:
        raise ValueError(
            "merged prefill must equal two, four, seven, eight, or sixteen "
            "segments"
        )
    if num_segments is not None:
        total_segments = int(num_segments)
        resident_segments = int(resident_prefix_segments)
        if (
            not MERGED_PREFILL_MIN_SEGMENTS
            <= total_segments
            <= MERGED_PREFILL_MAX_SEGMENTS
        ):
            raise ValueError("the request segment count must be between 4 and 16")
        if resident_segments not in (0, 1):
            raise ValueError("resident prefix segments must be zero or one")
        if resident_segments and not pure_context_bound:
            raise ValueError(
                "a resident prefix segment is valid only for pure context-bound MLA"
            )
        online_segments = total_segments - resident_segments
        # A resident first document leaves seven online 32K documents.  The
        # certified 64K physical group therefore executes three exact two-doc
        # groups followed by one ordinary 32K tail.  Requiring divisibility
        # would reject that safe tail and force all seven documents back to
        # separate scheduler forwards.
        if online_segments < merged_segments:
            raise ValueError(
                "the online request must contain at least one complete "
                "merged group"
            )
        if total_segments * segment_tokens > _effective_max_logical_tokens():
            raise ValueError("logical merged-prefill prefix must not exceed 262144")
    return requested


def _validated_plan_request(plan: object) -> Optional[int]:
    """Validate all request metadata needed before enlarging a scheduler batch."""

    if not isinstance(plan, Mapping):
        return None
    if plan.get("mode") != "restore":
        return None
    execution_profile = plan.get("mla_off_execution_profile")
    combined_row_sparse = (
        plan.get("reuse_mla_off") is True
        and plan.get("skip_forward") is True
        and plan.get("selection_policy") == "checkpoint_islands"
        and execution_profile == COMBINED_ROW_SPARSE_PROFILE
        and plan.get("mla_off_head_scope_policy") == PURE_CONTEXT_HEAD_SCOPE
        and plan.get("allow_approximate") is True
        and plan.get("mla_off_diagnostic_ablation") == "zoff_only"
        and plan.get("inject_full_blocks") is True
        and plan.get("refresh_selected_c4_rows") is True
        and plan.get("reuse_window_kv") is True
        and plan.get("skip_prefix_recompute") is True
        and plan.get("mla_off_use_indexer_hot") is True
        and plan.get("row_sparse_closure") is True
    )
    legacy_selected_rows = (
        plan.get("skip_forward") is True
        and plan.get("selection_policy") == "checkpoint_islands"
        and execution_profile != COMBINED_ROW_SPARSE_PROFILE
    )
    pure_context_bound = (
        plan.get("reuse_mla_off") is True
        and plan.get("mla_off_qualification_only") is True
        and plan.get("skip_forward") is not True
        and plan.get("mla_off_execution_profile") == PURE_CONTEXT_PROFILE
        and plan.get("mla_off_head_scope_policy") == PURE_CONTEXT_HEAD_SCOPE
        and plan.get("allow_approximate") is False
    )
    pure_independent_relocation = (
        plan.get("reuse_mla_off") is True
        and plan.get("mla_off_qualification_only") is True
        and plan.get("skip_forward") is not True
        and plan.get("mla_off_execution_profile")
        == PURE_INDEPENDENT_RELOCATION_PROFILE
        and plan.get("mla_off_head_scope_policy") == PURE_CONTEXT_HEAD_SCOPE
        and plan.get("allow_approximate") is True
    )
    if (
        not legacy_selected_rows
        and not pure_context_bound
        and not pure_independent_relocation
        and not combined_row_sparse
    ):
        return None
    raw_requested = plan.get("merged_prefill_tokens", 0)
    if isinstance(raw_requested, bool) or not isinstance(raw_requested, int):
        return None
    requested = int(raw_requested)
    if (
        requested <= 0
        or requested > MERGED_PREFILL_MAX_TOKENS
        or requested % MERGED_PREFILL_ALIGNMENT != 0
    ):
        return None

    segments = plan.get("segments")
    if (
        not isinstance(segments, Sequence)
        or isinstance(segments, (str, bytes))
        or not segments
    ):
        return None
    expected_offset = 0
    segment_length = None
    for segment in segments:
        if not isinstance(segment, Mapping):
            return None
        raw_offset = segment.get("global_offset")
        raw_length = segment.get("length")
        if (
            isinstance(raw_offset, bool)
            or not isinstance(raw_offset, int)
            or isinstance(raw_length, bool)
            or not isinstance(raw_length, int)
        ):
            return None
        offset = int(raw_offset)
        length = int(raw_length)
        if (
            offset != expected_offset
            or length <= 0
            or length % MERGED_PREFILL_ALIGNMENT != 0
        ):
            return None
        if segment_length is None:
            segment_length = length
        elif length != segment_length:
            return None
        if pure_context_bound:
            raw_source_start = segment.get("source_start")
            raw_source_end = segment.get("source_end")
            raw_skip_first = segment.get("skip_first")
            if (
                isinstance(raw_source_start, bool)
                or not isinstance(raw_source_start, int)
                or isinstance(raw_source_end, bool)
                or not isinstance(raw_source_end, int)
                or isinstance(raw_skip_first, bool)
                or not isinstance(raw_skip_first, int)
                or int(raw_source_start) != offset
                or int(raw_source_end) != offset + length
                or int(raw_skip_first) != 0
            ):
                return None
        elif pure_independent_relocation or combined_row_sparse:
            raw_canonical_start = segment.get("canonical_start_pos")
            raw_skip_first = segment.get("skip_first")
            expected_skip_first = (
                128 if combined_row_sparse else (0 if offset == 0 else 128)
            )
            if (
                isinstance(raw_canonical_start, bool)
                or not isinstance(raw_canonical_start, int)
                or int(raw_canonical_start) != 0
                or isinstance(raw_skip_first, bool)
                or not isinstance(raw_skip_first, int)
                or int(raw_skip_first) != expected_skip_first
            ):
                return None
        expected_offset += length
    raw_query_start = plan.get("query_start")
    raw_total_tokens = plan.get("total_tokens")
    radix_prefix_role = plan.get("radix_prefix_role")
    if radix_prefix_role not in (None, "consume"):
        return None
    merged_segments = (
        requested // segment_length
        if segment_length and requested % segment_length == 0
        else 0
    )
    alignment_origin = _merged_prefill_alignment_origin(plan)
    origin_segments = (
        alignment_origin // segment_length
        if segment_length
        and alignment_origin >= 0
        and alignment_origin % segment_length == 0
        else -1
    )
    online_segments = len(segments) - origin_segments
    if (
        isinstance(raw_query_start, bool)
        or not isinstance(raw_query_start, int)
        or int(raw_query_start) != expected_offset
        or isinstance(raw_total_tokens, bool)
        or not isinstance(raw_total_tokens, int)
        or int(raw_total_tokens) < int(raw_query_start)
        or segment_length is None
        or merged_segments not in MERGED_PREFILL_SEGMENT_COUNTS
        or len(segments) > MERGED_PREFILL_MAX_SEGMENTS
        or origin_segments < 0
        or (radix_prefix_role == "consume" and alignment_origin <= 0)
        or online_segments < merged_segments
        or expected_offset > _effective_max_logical_tokens()
    ):
        return None
    return requested


def _merged_prefill_alignment_origin(plan: object) -> int:
    """Return the only certified non-zero scheduler alignment origin.

    Ordinary merged-prefill requests remain aligned to logical position zero.
    A first-document consumer may instead align to its scheduler-owned radix
    prefix, but only when all four prefix receipt fields are structurally
    present and the prefix ends on the first certified segment boundary.  The
    backend performs the stronger physical receipt/hash validation before it
    permits omission; this helper merely prevents the scheduler from splitting
    the seven-document physical group.
    """

    if not isinstance(plan, Mapping) or plan.get("radix_prefix_role") != "consume":
        return 0
    prefix_tokens = plan.get("radix_prefix_tokens")
    prefix_hash = plan.get("radix_prefix_input_hash")
    receipt_key = plan.get("radix_prefix_receipt_key")
    segments = plan.get("segments")
    if (
        isinstance(prefix_tokens, bool)
        or not isinstance(prefix_tokens, int)
        or prefix_tokens <= 0
        or prefix_tokens % MERGED_PREFILL_ALIGNMENT != 0
        or not isinstance(prefix_hash, str)
        or not prefix_hash.startswith("sha256:")
        or len(prefix_hash) != 71
        or not isinstance(receipt_key, str)
        or not receipt_key.startswith("sha256:")
        or len(receipt_key) != 71
        or not isinstance(segments, Sequence)
        or isinstance(segments, (str, bytes))
        or not segments
        or not isinstance(segments[0], Mapping)
    ):
        return 0
    first_segment = segments[0]
    execution_profile = plan.get("mla_off_execution_profile")
    if execution_profile == PURE_CONTEXT_PROFILE:
        prefix_binding_valid = (
            first_segment.get("source_end") == prefix_tokens
            and first_segment.get("full_input_hash") == prefix_hash
        )
    elif execution_profile in (
        PURE_INDEPENDENT_RELOCATION_PROFILE,
        COMBINED_ROW_SPARSE_PROFILE,
    ):
        expected_skip_first = (
            128 if execution_profile == COMBINED_ROW_SPARSE_PROFILE else 0
        )
        prefix_binding_valid = (
            first_segment.get("global_offset") == 0
            and first_segment.get("length") == prefix_tokens
            and first_segment.get("token_hash") == prefix_hash
            and first_segment.get("canonical_start_pos") == 0
            and first_segment.get("skip_first") == expected_skip_first
        )
    else:
        prefix_binding_valid = False
    if not prefix_binding_valid:
        return 0
    return int(prefix_tokens)


def merged_prefill_tokens_from_plan(
    plan: object,
    global_chunk_tokens: Optional[int],
    *,
    server_max_prefill_tokens: Optional[int],
    server_opt_in_cap: Optional[int],
) -> Optional[int]:
    """Return a safe opt-in override, or ``None`` for the normal scheduler path."""

    requested = _validated_plan_request(plan)
    if requested is None:
        return None
    if (
        global_chunk_tokens is None
        or int(global_chunk_tokens) <= 0
        or requested < int(global_chunk_tokens)
        or server_max_prefill_tokens is None
        or requested > int(server_max_prefill_tokens)
        or server_opt_in_cap is None
        or int(server_opt_in_cap) <= 0
        or requested > int(server_opt_in_cap)
    ):
        return None
    return requested


def choose_merged_prefill_tokens(
    *,
    global_chunk_tokens: Optional[int],
    server_max_prefill_tokens: Optional[int],
    server_opt_in_cap: Optional[int],
    running_batch_size: int,
    has_active_chunked_req: bool,
    waiting_queue_size: int,
    active_chunked_logical_offset: Optional[int] = None,
    active_chunked_plan: object = None,
    first_waiting_plan: object = None,
) -> Optional[int]:
    """Choose an override only for a batch that can remain single-request."""

    if int(running_batch_size) != 0:
        return None
    # An active chunked request is always admitted first. If it is not eligible,
    # do not inspect the waiting queue and accidentally enlarge a mixed batch.
    if has_active_chunked_req:
        candidate = active_chunked_plan
    else:
        # A later waiting request can be selected if the first one is skipped
        # for LoRA, cache, or memory reasons. Only a singleton queue makes the
        # proposed plan identical to the request that will actually be admitted.
        if int(waiting_queue_size) != 1:
            return None
        candidate = first_waiting_plan
    requested = merged_prefill_tokens_from_plan(
        candidate,
        global_chunk_tokens,
        server_max_prefill_tokens=server_max_prefill_tokens,
        server_opt_in_cap=server_opt_in_cap,
    )
    if requested is None:
        return None
    if has_active_chunked_req:
        raw_offset = active_chunked_logical_offset
        alignment_origin = _merged_prefill_alignment_origin(candidate)
        raw_query_start = (
            candidate.get("query_start")
            if isinstance(candidate, Mapping)
            else None
        )
        if (
            isinstance(raw_offset, bool)
            or not isinstance(raw_offset, int)
            or int(raw_offset) < alignment_origin
            or (int(raw_offset) - alignment_origin) % requested != 0
            or isinstance(raw_query_start, bool)
            or not isinstance(raw_query_start, int)
            # Never enlarge the final one-document tail into the online query
            # suffix.  Returning None restores the ordinary 32K scheduler
            # chunk for that tail, and the suffix remains a separate forward.
            or int(raw_offset) + requested > int(raw_query_start)
        ):
            return None
    return requested


def merged_prefill_capacity_preflight(
    requested_tokens: int,
    *,
    candidate_extend_tokens: int,
    locked_prefix_tokens: int,
    page_size: int,
    full_available_tokens: int,
    full_evictable_tokens: int,
    swa_available_tokens: int,
    swa_evictable_tokens: int,
    sliding_window_tokens: int,
    max_new_tokens: int,
) -> Tuple[bool, str]:
    """Mirror ``PrefillAdder`` capacity checks before enlarging a chunk.

    The adder rejects equality (``needed >= remaining``), so this preflight
    deliberately requires one token of headroom in both physical pools.  It is
    conservative about generation by reserving the clipped remaining output
    budget even when the proposed logical chunk is not the final chunk.
    """

    values = {
        "requested_tokens": requested_tokens,
        "candidate_extend_tokens": candidate_extend_tokens,
        "locked_prefix_tokens": locked_prefix_tokens,
        "page_size": page_size,
        "full_available_tokens": full_available_tokens,
        "full_evictable_tokens": full_evictable_tokens,
        "swa_available_tokens": swa_available_tokens,
        "swa_evictable_tokens": swa_evictable_tokens,
        "sliding_window_tokens": sliding_window_tokens,
        "max_new_tokens": max_new_tokens,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f"{name} must be an integer"
    requested = int(requested_tokens)
    candidate_extend = int(candidate_extend_tokens)
    locked_prefix = int(locked_prefix_tokens)
    page = int(page_size)
    if requested <= 0 or requested > MERGED_PREFILL_MAX_TOKENS:
        return False, "requested tokens are outside the merged-prefill limit"
    if page <= 0:
        return False, "page size must be positive"
    if requested % page != 0:
        return False, "requested tokens must be page-aligned"
    if any(
        value < 0
        for value in (
            candidate_extend,
            locked_prefix,
            full_available_tokens,
            full_evictable_tokens,
            swa_available_tokens,
            swa_evictable_tokens,
            sliding_window_tokens,
            max_new_tokens,
        )
    ):
        return False, "capacity inputs must be non-negative"

    paged_requested = -(-requested // page) * page
    paged_extend = -(-candidate_extend // page) * page
    # ``PrefillAdder`` repeats both capacity checks after locking the matched
    # radix node.  Before that lock, the materialized first document is still
    # counted as evictable in both hybrid pools.  Mirror the post-lock state;
    # otherwise a 32K prefix + 64K merged chunk passes this preflight and then
    # spins forever on AddReqResult.NO_TOKEN.
    full_capacity = (
        int(full_available_tokens)
        + int(full_evictable_tokens)
        - locked_prefix
    )
    full_needed = paged_extend + int(max_new_tokens) + page
    if full_needed >= full_capacity:
        return (
            False,
            f"full KV capacity={full_capacity} must exceed budget={full_needed} "
            f"(extend={paged_extend}, max_new={max_new_tokens}, page={page})",
        )

    swa_capacity = (
        int(swa_available_tokens)
        + int(swa_evictable_tokens)
        - locked_prefix
    )
    swa_alloc = min(paged_extend, paged_requested)
    swa_needed = max(swa_alloc, int(sliding_window_tokens)) + page
    if swa_needed >= swa_capacity:
        return (
            False,
            f"SWA capacity={swa_capacity} must exceed budget={swa_needed} "
            f"(chunk={swa_alloc}, page={page})",
        )
    return True, ""


def allow_cross_segment_merged_prefill(
    plan: object,
    *,
    batch_size: int,
    logical_chunk_start: int,
    logical_chunk_tokens: int,
    server_max_prefill_tokens: Optional[int],
    server_opt_in_cap: Optional[int],
) -> bool:
    """Authorize a cross-segment model pass only inside its explicit token cap."""

    if int(batch_size) != 1:
        return False
    requested = _validated_plan_request(plan)
    if requested is None:
        return False
    if (
        server_max_prefill_tokens is None
        or requested > int(server_max_prefill_tokens)
        or server_opt_in_cap is None
        or int(server_opt_in_cap) <= 0
        or requested > int(server_opt_in_cap)
    ):
        return False
    logical_start = int(logical_chunk_start)
    logical_tokens = int(logical_chunk_tokens)
    alignment_origin = _merged_prefill_alignment_origin(plan)
    return (
        logical_start >= alignment_origin
        and (logical_start - alignment_origin) % requested == 0
        and logical_tokens == requested
    )


__all__ = [
    "MERGED_PREFILL_ALIGNMENT",
    "MERGED_PREFILL_MAX_LOGICAL_TOKENS",
    "MERGED_PREFILL_MAX_SEGMENTS",
    "MERGED_PREFILL_MAX_TOKENS",
    "MERGED_PREFILL_MIN_SEGMENTS",
    "MERGED_PREFILL_SEGMENT_COUNTS",
    "COMBINED_ROW_SPARSE_PROFILE",
    "PURE_INDEPENDENT_RELOCATION_PROFILE",
    "_validated_plan_request",
    "_merged_prefill_alignment_origin",
    "allow_cross_segment_merged_prefill",
    "choose_merged_prefill_tokens",
    "merged_prefill_capacity_preflight",
    "merged_prefill_tokens_from_plan",
    "validate_merged_prefill_request",
]
