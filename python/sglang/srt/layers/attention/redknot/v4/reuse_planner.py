from __future__ import annotations

from typing import Any, Mapping

from sglang.srt.layers.attention.redknot.v4.config import RedKnotV4Config
from sglang.srt.layers.attention.redknot.v4.types import (
    FallbackReason,
    RuntimePlanValidation,
)


# This is the wire-level execution contract emitted by the DSV4 MLA-off
# producer.  Keep the validator fail-closed: a missing or older profile must
# never be interpreted as the current pure 3 + 37 + 3 head-split semantics.
MLA_OFF_EXECUTION_PROFILE = (
    "pure_headsplit_context_bound_fullscope_3_37_3_v1"
)
MLA_OFF_INDEPENDENT_RELOCATION_PROFILE = (
    "pure_headsplit_independent_rope_relocation_fullscope_"
    "boundary128_3_37_3_v1"
)
MLA_OFF_COMBINED_ROW_SPARSE_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)
MLA_OFF_HEAD_SCOPE_POLICY = "native_dsv4_full_candidate_scope_v1"

_SNAPSHOT_REQUIRED_KEYS = frozenset(
    {
        "mode",
        "capture_mla_off",
        "mla_off_execution_profile",
        "mla_off_head_scope_policy",
        "allow_approximate",
        "seg_hash",
        "token_hash",
        "prefix_input_hash",
        "full_input_hash",
        "source_start",
        "source_end",
        "length",
        "canonical_start_pos",
        "model_compat_hash",
        "head_policy_hash",
    }
)
_RESTORE_REQUIRED_KEYS = frozenset(
    {
        "mode",
        "reuse_mla_off",
        "mla_off_execution_profile",
        "mla_off_head_scope_policy",
        "allow_approximate",
        "query_start",
        "total_tokens",
        "offline_prefix_hash",
        "request_input_hash",
        "segments",
        "mla_off_qualification_only",
        "model_compat_hash",
        "head_policy_hash",
    }
)
_RESTORE_OPTIONAL_KEYS = frozenset({"merged_prefill_tokens"})
_RUNTIME_IDENTITY_KEYS = frozenset(
    {"benchmark_request_id", "request_id", "snapshot_generation_id"}
)
_RESTORE_SEGMENT_KEYS = frozenset(
    {
        "seg_hash",
        "token_hash",
        "prefix_input_hash",
        "full_input_hash",
        "source_start",
        "source_end",
        "global_offset",
        "length",
        "canonical_start_pos",
        "skip_first",
    }
)
_INDEPENDENT_SNAPSHOT_REQUIRED_KEYS = frozenset(
    {
        "mode",
        "capture_mla_off",
        "mla_off_execution_profile",
        "mla_off_head_scope_policy",
        "allow_approximate",
        "seg_hash",
        "token_hash",
        "length",
        "canonical_start_pos",
        "model_compat_hash",
        "head_policy_hash",
    }
)
_INDEPENDENT_RESTORE_REQUIRED_KEYS = frozenset(
    {
        "mode",
        "reuse_mla_off",
        "mla_off_execution_profile",
        "mla_off_head_scope_policy",
        "allow_approximate",
        "query_start",
        "total_tokens",
        "segments",
        "mla_off_qualification_only",
        "model_compat_hash",
        "head_policy_hash",
    }
)
_INDEPENDENT_RESTORE_SEGMENT_KEYS = frozenset(
    {
        "seg_hash",
        "token_hash",
        "global_offset",
        "length",
        "canonical_start_pos",
        "skip_first",
    }
)
_COMBINED_SNAPSHOT_REQUIRED_KEYS = _INDEPENDENT_SNAPSHOT_REQUIRED_KEYS | {
    "reuse_window_kv",
    "checkpoint_stride_tokens",
}
_COMBINED_RESTORE_REQUIRED_KEYS = (
    _INDEPENDENT_RESTORE_REQUIRED_KEYS - {"mla_off_qualification_only"}
) | {
    "mla_off_diagnostic_ablation",
    "skip_forward",
    "inject_full_blocks",
    "refresh_selected_c4_rows",
    "reuse_window_kv",
    "skip_prefix_recompute",
    "selection_policy",
    "active_token_budget_ratio",
    "hot_max_per_segment_ratio",
    "checkpoint_stride_tokens",
    "checkpoint_max_islands",
    "interior_stride",
    "hot_frac",
    "mla_off_use_indexer_hot",
    "row_sparse_closure",
    "query_protection_policy",
    "query_protected_segment_index",
    "query_protected_ranges",
}


def _invalid(reason: FallbackReason, detail: str) -> RuntimePlanValidation:
    return RuntimePlanValidation(False, reason, detail)


def validate_runtime_reuse_plan(
    plan: Mapping[str, Any],
    *,
    config: RedKnotV4Config,
    dspark_active: bool = False,
) -> RuntimePlanValidation:
    try:
        checkpoint_stride = int(plan.get("checkpoint_stride_tokens", 0) or 0)
    except (TypeError, ValueError) as error:
        return _invalid(
            FallbackReason.INVALID_STATE,
            f"invalid checkpoint stride: {error}",
        )
    if checkpoint_stride and (
        checkpoint_stride < 512 or checkpoint_stride % 512 != 0
    ):
        return _invalid(
            FallbackReason.PHASE_MISMATCH,
            "checkpoint stride must be a positive multiple of 512 tokens",
        )
    if (
        plan.get("selection_policy") == "checkpoint_islands"
        and checkpoint_stride == 0
    ):
        return _invalid(
            FallbackReason.INVALID_STATE,
            "checkpoint-island selection requires checkpoint_stride_tokens",
        )
    if dspark_active:
        return _invalid(
            FallbackReason.DSPARK_CONFLICT,
            "correctness MVP does not compose reuse with D-Spark",
        )
    context_bound_profile = (
        str(plan.get("mla_off_execution_profile", ""))
        == MLA_OFF_EXECUTION_PROFILE
    )
    independent_relocation_profile = (
        str(plan.get("mla_off_execution_profile", ""))
        == MLA_OFF_INDEPENDENT_RELOCATION_PROFILE
    )
    combined_row_sparse_profile = (
        str(plan.get("mla_off_execution_profile", ""))
        == MLA_OFF_COMBINED_ROW_SPARSE_PROFILE
    )
    if context_bound_profile and plan.get("allow_approximate") is not False:
        return _invalid(
            FallbackReason.QUALITY_GATE,
            "context-bound pure MLA requires allow_approximate=false",
        )
    if (
        (independent_relocation_profile or combined_row_sparse_profile)
        and config.mode == "aggressive"
        and not bool(plan.get("allow_approximate", False))
    ):
        return _invalid(
            FallbackReason.UNSUPPORTED_KERNEL,
            "aggressive mode requires allow_approximate=true",
        )

    mode = plan.get("mode")
    if mode == "snapshot":
        segments = [
            {
                "length": plan.get("length"),
                "global_offset": plan.get("canonical_start_pos", 0),
                "canonical_start_pos": plan.get("canonical_start_pos", 0),
                "skip_first": (
            0 if context_bound_profile else config.boundary_replay_tokens
                ),
            }
        ]
    elif mode == "restore":
        segments = plan.get("segments") or []
        if not segments:
            return _invalid(
                FallbackReason.INVALID_STATE, "restore plan has no segments"
            )
    else:
        return _invalid(FallbackReason.INVALID_STATE, f"unsupported plan mode: {mode}")

    for index, segment in enumerate(segments):
        try:
            length = int(segment["length"])
            online_start = int(segment.get("global_offset", 0))
            canonical_start = int(segment.get("canonical_start_pos", 0))
            skip_first = int(segment.get("skip_first", config.boundary_replay_tokens))
        except (KeyError, TypeError, ValueError) as error:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"segment {index} has invalid metadata: {error}",
            )

        if length < config.min_cache_tokens or length % config.alignment_tokens != 0:
            return _invalid(
                FallbackReason.PHASE_MISMATCH,
                f"segment {index} length {length} is not a cacheable Level-A body",
            )
        pure_context_bound = (
            str(plan.get("mla_off_execution_profile", ""))
            == MLA_OFF_EXECUTION_PROFILE
        )
        if (
            (independent_relocation_profile or combined_row_sparse_profile)
            and online_start % 128 != canonical_start % 128
        ):
            return _invalid(
                FallbackReason.PHASE_MISMATCH,
                f"segment {index} has incompatible 128-token phase",
            )
        # Post-RoPE Indexer snapshots are relocatable.  Restore undoes the
        # normalized 128-dim Hadamard mixing, repositions the true RoPE64
        # subspace, reapplies the transform, and requantizes the complete key.
        # Keep the 128-token phase check above, but do not require the old
        # ``allow_post_rope_indexer_approx`` escape hatch.
        if pure_context_bound and skip_first != 0:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"segment {index} context-bound restore requires skip_first=0",
            )
        if (
            (independent_relocation_profile or combined_row_sparse_profile)
            and online_start != 0
            and skip_first < config.boundary_replay_tokens
        ):
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"segment {index} boundary replay is shorter than 128 tokens",
            )

    return RuntimePlanValidation(True)


def validate_mla_off_plan(
    plan: Mapping[str, Any],
    *,
    config: RedKnotV4Config,
) -> RuntimePlanValidation:
    """Validate only the optional MLA-off overlay.

    This validator is deliberately separate from ``validate_runtime_reuse_plan``.
    Rejecting an output artifact must not disable otherwise-valid packed-KV
    snapshot/restore for the same request.
    """

    mode = plan.get("mode")
    capture = bool(plan.get("capture_mla_off", False))
    restore = bool(plan.get("reuse_mla_off", False))
    if not capture and not restore:
        return RuntimePlanValidation(True)
    execution_profile = str(plan.get("mla_off_execution_profile", ""))
    head_scope_policy = str(plan.get("mla_off_head_scope_policy", ""))
    context_bound_profile = execution_profile == MLA_OFF_EXECUTION_PROFILE
    independent_relocation_profile = (
        execution_profile == MLA_OFF_INDEPENDENT_RELOCATION_PROFILE
    )
    combined_row_sparse_profile = (
        execution_profile == MLA_OFF_COMBINED_ROW_SPARSE_PROFILE
    )
    pure_boundary_repair_tokens = (
        128
        if independent_relocation_profile or combined_row_sparse_profile
        else 0
    )
    pure_headsplit = (
        context_bound_profile
        or independent_relocation_profile
        or combined_row_sparse_profile
    )
    if not pure_headsplit:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "unsupported MLA-off execution profile "
            f"{execution_profile!r}; expected one of "
            f"{(MLA_OFF_EXECUTION_PROFILE, MLA_OFF_INDEPENDENT_RELOCATION_PROFILE, MLA_OFF_COMBINED_ROW_SPARSE_PROFILE)!r}",
        )
    if head_scope_policy != MLA_OFF_HEAD_SCOPE_POLICY:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "pure MLA requires native DSV4 full candidate scope",
        )
    if context_bound_profile and plan.get("allow_approximate") is not False:
        return _invalid(
            FallbackReason.QUALITY_GATE,
            "context-bound pure MLA requires allow_approximate=false",
        )
    if (
        (independent_relocation_profile or combined_row_sparse_profile)
        and plan.get("allow_approximate") is not True
    ):
        return _invalid(
            FallbackReason.QUALITY_GATE,
            "independent-document RoPE relocation requires explicit "
            "allow_approximate=true",
        )
    if capture and plan.get("capture_mla_off") is not True:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "context-bound snapshot requires capture_mla_off=true",
        )
    if restore and plan.get("reuse_mla_off") is not True:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "context-bound restore requires reuse_mla_off=true",
        )
    if restore:
        qualification_marker = plan.get("mla_off_qualification_only")
        if combined_row_sparse_profile:
            if "mla_off_qualification_only" in plan:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "combined row-sparse formal restore forbids the qualification marker",
                )
        elif qualification_marker is not True:
            return _invalid(
                FallbackReason.INVALID_STATE,
                "context-bound qualification restore marker must be built-in true",
            )
    if combined_row_sparse_profile:
        required_keys = (
            _COMBINED_SNAPSHOT_REQUIRED_KEYS
            if capture
            else _COMBINED_RESTORE_REQUIRED_KEYS
        )
    elif independent_relocation_profile:
        required_keys = (
            _INDEPENDENT_SNAPSHOT_REQUIRED_KEYS
            if capture
            else _INDEPENDENT_RESTORE_REQUIRED_KEYS
        )
    else:
        required_keys = _SNAPSHOT_REQUIRED_KEYS if capture else _RESTORE_REQUIRED_KEYS
    observed_keys = frozenset(plan)
    missing_keys = required_keys - observed_keys
    optional_keys = _RESTORE_OPTIONAL_KEYS if restore else frozenset()
    unexpected_keys = (
        observed_keys - required_keys - optional_keys - _RUNTIME_IDENTITY_KEYS
    )
    if missing_keys or unexpected_keys:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "pure MLA plan schema mismatch: "
            f"missing={sorted(missing_keys)} unexpected={sorted(unexpected_keys)}",
        )
    if combined_row_sparse_profile and restore:
        protection_policy = str(plan.get("query_protection_policy", ""))
        protected_index = plan.get("query_protected_segment_index")
        segments = tuple(plan.get("segments", ()) or ())
        segment_count = len(segments)
        protected_ranges = plan.get("query_protected_ranges")
        if type(protected_index) is not int:
            return _invalid(
                FallbackReason.INVALID_STATE,
                "combined query-protected segment index is invalid",
            )
        if protection_policy == "none":
            if protected_index != -1 or protected_ranges != []:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "disabled query protection requires index=-1/ranges=[]",
                )
        elif protection_policy in {
            "lexical_top1_full_segment_v1",
            "lexical_top1_block_windows_v1",
            "lexical_topk_block_windows_v2",
        }:
            if not 0 <= protected_index < segment_count:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "query-protected segment is outside the restore chain",
                )
            if not isinstance(protected_ranges, list) or not protected_ranges:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "query-protected ranges are absent",
                )
            segment = segments[protected_index]
            segment_begin = int(segment["global_offset"])
            segment_end = segment_begin + int(segment["length"])
            cursor = 0
            normalized = []
            protected_segment_indices = set()
            for item in protected_ranges:
                if not isinstance(item, Mapping) or frozenset(item) != {
                    "start",
                    "end",
                }:
                    return _invalid(
                        FallbackReason.INVALID_STATE,
                        "query-protected range schema is invalid",
                    )
                begin, end = int(item["start"]), int(item["end"])
                containing = [
                    segment_index
                    for segment_index, candidate in enumerate(segments)
                    if begin >= int(candidate["global_offset"])
                    and end
                    <= int(candidate["global_offset"]) + int(candidate["length"])
                ]
                if (
                    len(containing) != 1
                    or begin < cursor
                    or begin >= end
                    or begin % 512 != 0
                    or end % 512 != 0
                ):
                    return _invalid(
                        FallbackReason.INVALID_STATE,
                        "query-protected range geometry is invalid",
                    )
                normalized.append((begin, end))
                protected_segment_indices.add(containing[0])
                cursor = end
            if protection_policy == "lexical_top1_full_segment_v1" and normalized != [
                (segment_begin, segment_end)
            ]:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "full-segment query protection is incomplete",
                )
            if (
                protection_policy == "lexical_top1_block_windows_v1"
                and protected_segment_indices != {protected_index}
            ):
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "top1 query protection escaped its selected segment",
                )
            if protection_policy == "lexical_topk_block_windows_v2" and (
                protected_index not in protected_segment_indices
                or len(protected_segment_indices) != 2
            ):
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "topk query protection must cover exactly two segments",
                )
        else:
            return _invalid(
                FallbackReason.INVALID_STATE,
                "combined query protection policy is unsupported",
            )
    if capture and restore:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "a plan cannot capture and restore MLA-off simultaneously",
        )
    if capture and mode != "snapshot":
        return _invalid(
            FallbackReason.INVALID_STATE,
            "capture_mla_off is valid only for snapshot plans",
        )
    if restore and mode != "restore":
        return _invalid(
            FallbackReason.INVALID_STATE,
            "reuse_mla_off is valid only for restore plans",
        )
    if restore and not combined_row_sparse_profile:
        legacy_kv_flags = tuple(
            name
            for name in (
                "reuse_window_kv",
                "reuse_csa",
                "reuse_hca",
                "inject_full_blocks",
                "refresh_selected_c4_rows",
            )
            if bool(plan.get(name, False))
        )
        if legacy_kv_flags:
            return _invalid(
                FallbackReason.INVALID_STATE,
                "MLA-off restore recomputes global-head KV online and cannot "
                "also enable legacy KV restore: " + ",".join(legacy_kv_flags),
            )
    if (
        (capture or restore)
        and not combined_row_sparse_profile
        and bool(plan.get("skip_forward", False))
    ):
        return _invalid(
            FallbackReason.UNSUPPORTED_KERNEL,
            "MLA-off requires all rows for global heads; set skip_forward=false",
        )
    if capture and "merged_prefill_tokens" in plan:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "snapshot plans cannot request merged prefill",
        )
    if restore and "merged_prefill_tokens" in plan:
        from sglang.srt.layers.attention.redknot.v4.merged_prefill import (
            _validated_plan_request,
        )

        if _validated_plan_request(plan) is None:
            return _invalid(
                FallbackReason.INVALID_STATE,
                "context restore has an invalid pure merged-prefill contract",
            )

    forbidden_selected_fields = tuple(
        name
        for name in (
            "selection_policy",
            "active_token_budget_ratio",
            "hot_max_per_segment_ratio",
            "hot_frac",
            "skip_prefix_recompute",
            "checkpoint_stride_tokens",
            "checkpoint_max_islands",
            "interior_stride",
            "mla_off_dirty_ranges",
        )
        if name in plan
    )
    if pure_headsplit and not combined_row_sparse_profile and forbidden_selected_fields:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "pure MLA headsplit cannot include selected-row fields: "
            + ",".join(forbidden_selected_fields),
        )
    if (
        pure_headsplit
        and not combined_row_sparse_profile
        and bool(plan.get("mla_off_use_indexer_hot", False))
    ):
        return _invalid(
            FallbackReason.INVALID_STATE,
            "pure MLA headsplit cannot use Indexer-hot rows",
        )

    try:
        refresh_stride = int(plan.get("mla_off_refresh_layer_stride", 0) or 0)
        refresh_layers = tuple(
            int(layer) for layer in plan.get("mla_off_refresh_layers", ()) or ()
        )
        hot_expand = int(plan.get("mla_off_hot_expand_tokens", 0) or 0)
        hot_frac = float(plan.get("hot_frac", 0.5))
    except (TypeError, ValueError) as error:
        return _invalid(
            FallbackReason.INVALID_STATE,
            f"invalid MLA-off refresh policy: {error}",
        )
    if pure_headsplit and (refresh_stride != 0 or hot_expand != 0 or refresh_layers):
        return _invalid(
            FallbackReason.INVALID_STATE,
            "pure MLA headsplit forbids refresh layers and hot expansion",
        )
    if not pure_headsplit and (
        refresh_stride < 0
        or hot_expand < 0
        or any(layer < 0 for layer in refresh_layers)
    ):
        return _invalid(
            FallbackReason.INVALID_STATE,
            "MLA-off refresh stride/layers/hot expansion must be non-negative",
        )
    if (
        not pure_headsplit
        and bool(plan.get("mla_off_use_indexer_hot", True))
        and not 0.0 < hot_frac <= 1.0
    ):
        return _invalid(
            FallbackReason.INVALID_STATE,
            "MLA-off Indexer hot_frac must be in (0, 1]",
        )
    # Bound interval growth and the amount of local replay from one bad request.
    if hot_expand > 4096:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "MLA-off Indexer hot expansion cannot exceed 4096 tokens",
        )

    for index, dirty_range in enumerate(plan.get("mla_off_dirty_ranges", ()) or ()):
        try:
            if isinstance(dirty_range, Mapping):
                begin = int(dirty_range["start"])
                end = int(dirty_range["end"])
            else:
                begin = int(dirty_range[0])
                end = int(dirty_range[1])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"MLA-off dirty range {index} is invalid: {error}",
            )
        if begin < 0 or begin >= end:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"MLA-off dirty range {index} must satisfy 0 <= start < end",
            )

    if combined_row_sparse_profile:
        if capture:
            if plan.get("reuse_window_kv") is not True:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "combined snapshot must capture legacy window/indexer state",
                )
            if int(plan.get("checkpoint_stride_tokens", 0) or 0) <= 0:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "combined snapshot requires a checkpoint stride",
                )
        else:
            exact_true = (
                "skip_forward",
                "inject_full_blocks",
                "refresh_selected_c4_rows",
                "reuse_window_kv",
                "skip_prefix_recompute",
                "mla_off_use_indexer_hot",
                "row_sparse_closure",
            )
            disabled = tuple(name for name in exact_true if plan.get(name) is not True)
            if disabled:
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "combined restore requires selected-row closure flags: "
                    + ",".join(disabled),
                )
            if plan.get("selection_policy") != "checkpoint_islands":
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "combined restore requires checkpoint_islands selection",
                )
            if plan.get("mla_off_diagnostic_ablation") != "zoff_only":
                return _invalid(
                    FallbackReason.INVALID_STATE,
                    "combined restore requires diagnostic_ablation=zoff_only",
                )

    if capture and (independent_relocation_profile or combined_row_sparse_profile):
        try:
            length = int(plan["length"])
            canonical_start = int(plan["canonical_start_pos"])
            seg_hash = str(plan["seg_hash"])
            token_hash = str(plan["token_hash"])
        except (KeyError, TypeError, ValueError) as error:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"invalid independent snapshot metadata: {error}",
            )
        if length <= 0 or canonical_start != 0 or not seg_hash or not token_hash:
            return _invalid(
                FallbackReason.INVALID_STATE,
                "independent snapshot requires length>0, canonical_start=0 and hashes",
            )
        return RuntimePlanValidation(True)

    if capture:
        try:
            from sglang.srt.layers.attention.redknot.dsv4_context_identity import (
                ContextSegmentContract,
            )

            contract = ContextSegmentContract.from_mapping(
                plan,
                execution_profile=execution_profile,
                head_scope_policy=head_scope_policy,
                model_compat_hash=str(plan["model_compat_hash"]),
                head_policy_hash=str(plan["head_policy_hash"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"invalid MLA-off snapshot metadata: {error}",
            )
        if contract.source_end <= 0:
            return _invalid(
                FallbackReason.INVALID_STATE,
                "context-bound MLA snapshot source interval is empty",
            )
        return RuntimePlanValidation(True)

    segments = plan.get("segments") or []
    if not segments:
        return _invalid(FallbackReason.INVALID_STATE, "MLA-off restore has no segments")
    expected_segment_keys = (
        _INDEPENDENT_RESTORE_SEGMENT_KEYS
        if independent_relocation_profile or combined_row_sparse_profile
        else _RESTORE_SEGMENT_KEYS
    )
    if any(
        not isinstance(segment, Mapping)
        or frozenset(segment) != expected_segment_keys
        for segment in segments
    ):
        return _invalid(
            FallbackReason.INVALID_STATE,
            "pure MLA restore segment schema mismatch",
        )
    if context_bound_profile:
        try:
            from sglang.srt.layers.attention.redknot.dsv4_context_identity import (
                validate_context_segment_chain,
            )

            validate_context_segment_chain(
                segments,
                execution_profile=execution_profile,
                head_scope_policy=head_scope_policy,
                model_compat_hash=str(plan["model_compat_hash"]),
                head_policy_hash=str(plan["head_policy_hash"]),
                offline_prefix_hash=str(plan["offline_prefix_hash"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"invalid context-bound MLA segment chain: {error}",
            )
    cursor = 0
    seen_hashes = {}
    try:
        ordered_segments = sorted(
            segments, key=lambda item: int(item.get("global_offset", 0))
        )
    except (AttributeError, TypeError, ValueError) as error:
        return _invalid(
            FallbackReason.INVALID_STATE,
            f"MLA-off segment ordering is invalid: {error}",
        )
    for index, segment in enumerate(ordered_segments):
        try:
            seg_hash = str(segment["seg_hash"])
            token_hash = str(segment.get("token_hash", seg_hash))
            length = int(segment["length"])
            offset = int(segment.get("global_offset", 0))
            canonical_start = int(segment.get("canonical_start_pos", 0))
            expected_pure_boundary = (
                pure_boundary_repair_tokens
                if combined_row_sparse_profile
                else (0 if offset == 0 else pure_boundary_repair_tokens)
            )
            skip_first = int(
                segment.get(
                    "skip_first",
                    expected_pure_boundary
                    if pure_headsplit
                    else config.boundary_replay_tokens,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"MLA-off segment {index} has invalid metadata: {error}",
            )
        if not seg_hash or not token_hash or length <= 0:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"MLA-off segment {index} has empty hashes or non-positive length",
            )
        if offset != cursor:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"MLA-off segments must be contiguous from zero; expected {cursor}, got {offset}",
            )
        if canonical_start != 0:
            return _invalid(
                FallbackReason.PHASE_MISMATCH,
                f"MLA-off segment {index} must have canonical_start_pos=0",
            )
        if offset < 0:
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"MLA-off segment {index} global offset must be non-negative",
            )
        if pure_headsplit and (
            skip_first != expected_pure_boundary or skip_first > length
        ):
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"pure MLA segment {index} requires skip_first="
                f"{expected_pure_boundary}",
            )
        if not pure_headsplit and not (
            config.boundary_replay_tokens <= skip_first <= length
        ):
            return _invalid(
                FallbackReason.INVALID_STATE,
                f"MLA-off segment {index} boundary must be within "
                f"[{config.boundary_replay_tokens}, length]",
            )
        identity = (token_hash, length)
        if seg_hash in seen_hashes and seen_hashes[seg_hash] != identity:
            return _invalid(
                FallbackReason.TOKEN_MISMATCH,
                f"MLA-off duplicate segment hash {seg_hash!r} has conflicting metadata",
            )
        seen_hashes[seg_hash] = identity
        cursor += length

    raw_query_start = plan.get("query_start")
    raw_total = plan.get("total_tokens")
    try:
        query_start = None if raw_query_start is None else int(raw_query_start)
        total_tokens = None if raw_total is None else int(raw_total)
    except (TypeError, ValueError) as error:
        return _invalid(
            FallbackReason.INVALID_STATE,
            f"MLA-off query/total token metadata is invalid: {error}",
        )
    if query_start is not None and query_start != cursor:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "MLA-off query_start must equal the end of contiguous segments",
        )
    if total_tokens is not None and total_tokens < cursor:
        return _invalid(
            FallbackReason.INVALID_STATE,
            "MLA-off total_tokens cannot end inside an offline segment",
        )
    request_input_hash = plan.get("request_input_hash")
    if context_bound_profile and not (
        isinstance(request_input_hash, str)
        and request_input_hash.startswith("sha256:")
        and len(request_input_hash) == 71
        and all(
            character in "0123456789abcdef"
            for character in request_input_hash[7:]
        )
    ):
        return _invalid(
            FallbackReason.INVALID_STATE,
            "context-bound MLA restore request_input_hash is invalid",
        )
    return RuntimePlanValidation(True)


__all__ = [
    "MLA_OFF_EXECUTION_PROFILE",
    "MLA_OFF_INDEPENDENT_RELOCATION_PROFILE",
    "MLA_OFF_COMBINED_ROW_SPARSE_PROFILE",
    "MLA_OFF_HEAD_SCOPE_POLICY",
    "validate_mla_off_plan",
    "validate_runtime_reuse_plan",
]
