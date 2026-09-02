from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Dict, Literal, Mapping, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4AttnMetadata,
    FLASHMLA_MAX_BATCH_ROWS,
    SWA_WINDOW,
    _create_flashmla_metadata,
    _pad_tensor_to_size,
)
from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
    DeepSeekV4MLAHeadConfig,
    deepseek_v4_redknot_topology,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_size,
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)

_REDKNOT_MLA_TIMING_NOOP = nullcontext()
if os.environ.get("REDKNOT_V4_TIMING", "0") == "1":
    from sglang.srt.layers.attention.redknot.dsv4_timing import (
        timed as _redknot_mla_timed,
    )
else:

    def _redknot_mla_timed(*_args, **_kwargs):
        return _REDKNOT_MLA_TIMING_NOOP

_MLA_OFF_PLAN_DIGEST_UNSET = object()
_PURE_HEADSPLIT_PROFILE = (
    "pure_headsplit_context_bound_fullscope_3_37_3_v1"
)
_INDEPENDENT_RELOCATION_PROFILE = (
    "pure_headsplit_independent_rope_relocation_fullscope_"
    "boundary128_3_37_3_v1"
)
_COMBINED_ROW_SPARSE_PROFILE = (
    "combined_headsplit_independent_rope_zoff_checkpoint_"
    "rowsparse_3_37_3_v1"
)
_PURE_HEADSPLIT_PROFILES = (
    _PURE_HEADSPLIT_PROFILE,
    _INDEPENDENT_RELOCATION_PROFILE,
    _COMBINED_ROW_SPARSE_PROFILE,
)
_PURE_HEADSPLIT_HEAD_SCOPE_POLICY = "native_dsv4_full_candidate_scope_v1"
_PURE_HEADSPLIT_BOUNDARY_REPAIR_TOKENS = 0
_PURE_HEADSPLIT_NUM_LAYERS = 43
_PURE_HEADSPLIT_DENSE_PREFIX = 3
_PURE_HEADSPLIT_DENSE_SUFFIX = 3
_PURE_HEADSPLIT_OFFLINE_LAYER_IDS = tuple(range(3, 40))
_PURE_HEADSPLIT_DENSE_LAYER_IDS = (0, 1, 2, 40, 41, 42)
_MLA_OFF_QUALIFICATION_PLAN_FIELD = "mla_off_qualification_only"
_SHARED_SNAPSHOT_AUDIT_SCHEMA = "redknot_shared_snapshot_publication_v1"
_SHARED_RESTORE_AUDIT_SCHEMA = "redknot_shared_device_restore_stats_v1"
_SHARED_RESTORE_COUNTER_KEYS = {
    "shared_device_restore_calls": "mla_off.shared_device_restore_calls",
    "shared_device_restore_operations": (
        "mla_off.shared_device_restore_operations"
    ),
    "shared_device_values_restored": (
        "mla_off.shared_device_values_restored"
    ),
}
_MLA_OFF_GLOBAL_ATTN_IMPLS = (
    "triton_h1",
    "padded_flashmla_h64",
)
_MLA_OFF_PADDED_FLASHMLA_MIN_ROWS = 2048


def _mla_off_snapshot_generation_id(
    *,
    explicit_generation_id,
    seg_hash: str,
    token_hash: str,
    request_id: str,
    benchmark_forward_id: str,
    input_layout_digest,
    length: int,
    canonical_start_pos: int,
    source_start: int,
    source_end: int,
    prefix_input_hash: str,
    full_input_hash: str,
    head_scope_policy: str,
    model_hash: str,
    policy_hash: str,
) -> str:
    """Return a TP-common snapshot generation identity.

    An explicit producer generation remains authoritative.  The fallback is
    derived only from semantic inputs that are identical on every attention-TP
    rank.  In particular, it must never depend on the rank-local ModelRunner
    object identity or its local forward-pass counter.
    """

    if explicit_generation_id:
        return str(explicit_generation_id)
    canonical = {
        "benchmark_forward_id": str(benchmark_forward_id),
        "canonical_start_pos": int(canonical_start_pos),
        "input_layout_digest": [
            int(value) for value in tuple(input_layout_digest)
        ],
        "length": int(length),
        "source_start": int(source_start),
        "source_end": int(source_end),
        "prefix_input_hash": str(prefix_input_hash),
        "full_input_hash": str(full_input_hash),
        "head_scope_policy": str(head_scope_policy),
        "model_hash": str(model_hash),
        "policy_hash": str(policy_hash),
        "request_id": str(request_id),
        "seg_hash": str(seg_hash),
        "token_hash": str(token_hash),
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return "snapshot-v1:sha256:" + digest


def _validate_pure_headsplit_plan_contract(plan: Mapping[str, object]) -> None:
    """Reject row-selection overlays before they can affect head execution."""

    from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
        resolve_mla_off_diagnostic_ablation,
    )

    if _MLA_OFF_QUALIFICATION_PLAN_FIELD in plan and type(
        plan[_MLA_OFF_QUALIFICATION_PLAN_FIELD]
    ) is not bool:
        raise ValueError(
            "pure headsplit plan qualification marker must be a boolean"
        )
    # Validation is intentionally request-plan based.  Cache availability must
    # never infer an ablation mode, and malformed/conflicting flags fail before
    # any artifact pin, Q omission, or TP data-path collective.
    diagnostic_ablation = resolve_mla_off_diagnostic_ablation(plan)
    if (
        plan.get(_MLA_OFF_QUALIFICATION_PLAN_FIELD) is True
        and diagnostic_ablation != "full"
    ):
        raise ValueError(
            "MLA-off qualification-only plan requires diagnostic_ablation=full"
        )
    profile = str(plan.get("mla_off_execution_profile", ""))
    if profile not in _PURE_HEADSPLIT_PROFILES:
        raise ValueError(
            "pure headsplit plan requires a supported execution profile; "
            f"observed={profile!r} supported={_PURE_HEADSPLIT_PROFILES!r}"
        )
    if str(plan.get("mla_off_head_scope_policy", "")) != (
        _PURE_HEADSPLIT_HEAD_SCOPE_POLICY
    ):
        raise ValueError(
            "context-bound pure headsplit requires native DSV4 full candidate scope"
        )
    context_bound_profile = profile == _PURE_HEADSPLIT_PROFILE
    independent_relocation_profile = profile == _INDEPENDENT_RELOCATION_PROFILE
    combined_row_sparse_profile = profile == _COMBINED_ROW_SPARSE_PROFILE
    if context_bound_profile and plan.get("allow_approximate") is not False:
        raise ValueError(
            "context-bound pure headsplit requires allow_approximate=false"
        )
    if (
        (independent_relocation_profile or combined_row_sparse_profile)
        and plan.get("allow_approximate") is not True
    ):
        raise ValueError(
            "independent-document RoPE relocation requires "
            "allow_approximate=true"
        )
    mode = str(plan.get("mode", ""))
    if mode == "snapshot" and plan.get("capture_mla_off") is not True:
        raise ValueError("context snapshot marker must be built-in true")
    if mode == "restore" and plan.get("reuse_mla_off") is not True:
        raise ValueError("context restore marker must be built-in true")
    radix_fields = (
        "radix_prefix_role",
        "radix_prefix_tokens",
        "radix_prefix_input_hash",
        "radix_prefix_receipt_key",
    )
    radix_present = tuple(name for name in radix_fields if name in plan)
    if radix_present:
        if mode != "restore" or radix_present != radix_fields:
            raise ValueError(
                "radix-prefix restore requires its complete four-field contract"
            )
        role = plan.get("radix_prefix_role")
        prefix_tokens = plan.get("radix_prefix_tokens")
        prefix_hash = plan.get("radix_prefix_input_hash")
        receipt_key = plan.get("radix_prefix_receipt_key")
        if role not in ("seed", "consume"):
            raise ValueError("radix-prefix role must be seed or consume")
        if type(prefix_tokens) is not int or prefix_tokens <= 0:
            raise ValueError("radix-prefix token extent must be positive")
        for value, label in (
            (prefix_hash, "input hash"),
            (receipt_key, "receipt key"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 71
                or not value.startswith("sha256:")
                or any(char not in "0123456789abcdef" for char in value[7:])
            ):
                raise ValueError(f"radix-prefix {label} is not canonical SHA-256")
        segments = plan.get("segments")
        if not isinstance(segments, (tuple, list)) or not segments:
            raise ValueError("radix-prefix restore has no segment chain")
        first = segments[0]
        first_hash = (
            first.get("full_input_hash")
            if context_bound_profile and isinstance(first, Mapping)
            else first.get("token_hash") if isinstance(first, Mapping) else None
        )
        if (
            not isinstance(first, Mapping)
            or int(first.get("global_offset", -1)) != 0
            or int(first.get("length", -1)) != prefix_tokens
            or first_hash != prefix_hash
        ):
            raise ValueError(
                "radix-prefix extent/hash differs from the first offline segment"
            )
        if role == "seed" and (
            len(segments) != 1 or plan.get("query_start") != prefix_tokens
        ):
            raise ValueError(
                "radix-prefix seed must contain exactly the first offline segment"
            )
        if role == "consume" and (
            len(segments) <= 1 or int(plan.get("query_start", -1)) <= prefix_tokens
        ):
            raise ValueError(
                "radix-prefix consumer must restore later offline segments"
            )

    forbidden_present = (
        "selection_policy",
        "active_token_budget_ratio",
        "hot_max_per_segment_ratio",
        "hot_frac",
        "skip_prefix_recompute",
        "checkpoint_stride_tokens",
        "checkpoint_max_islands",
        "checkpoint_islands",
        "interior_stride",
        "mla_off_dirty_ranges",
    )
    contaminated = tuple(name for name in forbidden_present if name in plan)
    if combined_row_sparse_profile:
        contaminated = ()
    if contaminated:
        raise ValueError(
            "pure headsplit plan contains selected-row fields: "
            + ",".join(contaminated)
        )
    if mode == "snapshot" and "merged_prefill_tokens" in plan:
        raise ValueError(
            "context snapshot must keep the certified 8192-token producer boundary"
        )
    if mode == "restore" and "merged_prefill_tokens" in plan:
        from sglang.srt.layers.attention.redknot.v4.merged_prefill import (
            _validated_plan_request,
        )

        if _validated_plan_request(plan) is None:
            raise ValueError(
                "context restore has an invalid pure merged-prefill contract"
            )

    forbidden_truthy = (
        "skip_forward",
        "inject_full_blocks",
        "refresh_selected_c4_rows",
        "reuse_window_kv",
        "reuse_csa",
        "reuse_hca",
        "mla_off_use_indexer_hot",
    )
    enabled = tuple(name for name in forbidden_truthy if bool(plan.get(name, False)))
    if combined_row_sparse_profile:
        enabled = ()
    if enabled:
        raise ValueError(
            "pure headsplit plan enables incompatible row/KV paths: "
            + ",".join(enabled)
        )
    if int(plan.get("mla_off_refresh_layer_stride", 0) or 0) != 0:
        raise ValueError("pure headsplit plan forbids layer refresh stride")
    if tuple(plan.get("mla_off_refresh_layers", ()) or ()):
        raise ValueError("pure headsplit plan forbids refresh layers")
    if int(plan.get("mla_off_hot_expand_tokens", 0) or 0) != 0:
        raise ValueError("pure headsplit plan forbids Indexer hot expansion")
    if str(plan.get("mode", "")) == "restore":
        for segment in plan.get("segments", ()) or ():
            global_offset = int(segment.get("global_offset", 0))
            if global_offset < 0:
                raise ValueError(
                    "pure headsplit segment global_offset must be non-negative"
                )
            length = segment.get("length")
            if type(length) is not int:
                raise ValueError("pure headsplit segment length is invalid")
            if context_bound_profile:
                source_start = segment.get("source_start")
                source_end = segment.get("source_end")
                if (
                    type(source_start) is not int
                    or type(source_end) is not int
                    or source_start != global_offset
                    or source_end != source_start + length
                ):
                    raise ValueError(
                        "context-bound segment source/global interval is inconsistent"
                    )
            expected_boundary = (
                128
                if combined_row_sparse_profile
                or (independent_relocation_profile and global_offset != 0)
                else _PURE_HEADSPLIT_BOUNDARY_REPAIR_TOKENS
            )
            if int(segment.get("skip_first", expected_boundary)) != expected_boundary:
                if context_bound_profile:
                    raise ValueError(
                        "context-bound pure headsplit requires skip_first=0 "
                        "for every segment"
                    )
                raise ValueError(
                    "independent-document pure headsplit requires "
                    "skip_first=128 for every relocated segment"
                )

    if combined_row_sparse_profile:
        if mode == "snapshot":
            if plan.get("reuse_window_kv") is not True:
                raise ValueError("combined snapshot must capture window/indexer state")
            if int(plan.get("checkpoint_stride_tokens", 0) or 0) <= 0:
                raise ValueError("combined snapshot requires checkpoint stride")
        elif mode == "restore":
            required_true = (
                "skip_forward",
                "inject_full_blocks",
                "refresh_selected_c4_rows",
                "reuse_window_kv",
                "skip_prefix_recompute",
                "mla_off_use_indexer_hot",
                "row_sparse_closure",
            )
            missing = tuple(name for name in required_true if plan.get(name) is not True)
            if missing:
                raise ValueError(
                    "combined restore is missing selected-row closure flags: "
                    + ",".join(missing)
                )
            if plan.get("selection_policy") != "checkpoint_islands":
                raise ValueError("combined restore requires checkpoint_islands")
            if diagnostic_ablation != "zoff_only":
                raise ValueError("combined restore requires zoff_only head merge")
            if _MLA_OFF_QUALIFICATION_PLAN_FIELD in plan:
                raise ValueError(
                    "combined row-sparse formal restore forbids the "
                    "qualification marker"
                )
            protection_policy = str(plan.get("query_protection_policy", ""))
            protected_index = plan.get("query_protected_segment_index")
            protected_ranges = plan.get("query_protected_ranges")
            segments = tuple(plan.get("segments", ()) or ())
            segment_count = len(segments)
            if type(protected_index) is not int:
                raise ValueError(
                    "combined query-protected segment index is invalid"
                )
            if protection_policy == "none":
                if protected_index != -1 or protected_ranges != []:
                    raise ValueError(
                        "disabled query protection requires index=-1/ranges=[]"
                    )
            elif protection_policy in {
                "lexical_top1_full_segment_v1",
                "lexical_top1_block_windows_v1",
                "lexical_topk_block_windows_v2",
            }:
                if not 0 <= protected_index < segment_count:
                    raise ValueError(
                        "query-protected segment is outside the restore chain"
                    )
                if not isinstance(protected_ranges, list) or not protected_ranges:
                    raise ValueError("query-protected ranges are absent")
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
                        raise ValueError("query-protected range schema is invalid")
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
                        raise ValueError("query-protected range geometry is invalid")
                    normalized.append((begin, end))
                    protected_segment_indices.add(containing[0])
                    cursor = end
                if (
                    protection_policy == "lexical_top1_full_segment_v1"
                    and normalized != [(segment_begin, segment_end)]
                ):
                    raise ValueError(
                        "full-segment query protection is incomplete"
                    )
                if (
                    protection_policy == "lexical_top1_block_windows_v1"
                    and protected_segment_indices != {protected_index}
                ):
                    raise ValueError(
                        "top1 query protection escaped its selected segment"
                    )
                if protection_policy == "lexical_topk_block_windows_v2" and (
                    protected_index not in protected_segment_indices
                    or len(protected_segment_indices) != 2
                ):
                    raise ValueError(
                        "topk query protection must cover exactly two segments"
                    )
            else:
                raise ValueError(
                    "combined query protection policy is unsupported"
                )


def _linux_process_identity(pid: int) -> Tuple[str, int]:
    """Return Linux process state and PID-reuse-resistant start-time ticks."""

    pid = int(pid)
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing_paren = raw.rfind(")")
    if closing_paren < 0:
        raise ValueError(f"malformed /proc/{pid}/stat")
    fields = raw[closing_paren + 1 :].split()
    # fields[0] is state (proc field 3); fields[19] is starttime (field 22).
    if len(fields) <= 19:
        raise ValueError(f"truncated /proc/{pid}/stat")
    state = fields[0]
    start_time_ticks = int(fields[19])
    if len(state) != 1 or start_time_ticks <= 0:
        raise ValueError(f"invalid /proc/{pid}/stat identity")
    return state, start_time_ticks


@dataclass(frozen=True)
class _DualLayerPassPlan:
    """Executable logical-head policy for one shared physical KV stream."""

    local_groups: Tuple[Tuple[int, Tuple[int, ...]], ...]
    global_heads: Tuple[int, ...]
    promoted_heads: Tuple[Tuple[int, int], ...]

    @property
    def effective_local_heads(self) -> int:
        return sum(len(heads) for _, heads in self.local_groups)


class _RedKnotCompositeCollectiveAdapter:
    """One-vector bridge from the pure protocol to attention TP.

    The adapter intentionally exposes no readiness helper: a layer gets one
    fixed-shape int64 SUM collective, whose reduction token is derived from the
    identical reduced vector on every rank.  Post-commit errors are fail-stop
    signals rather than another sequence of latency-sensitive votes.
    """

    def __init__(self, backend, device) -> None:
        self._backend = backend
        self._device = torch.device(device)

    def exchange_commit_once(self, int64_vector):
        from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
            CollectiveReduction,
            COMMIT_INT64_VECTOR_LENGTH,
        )

        values = tuple(int(value) for value in int64_vector)
        if len(values) != int(COMMIT_INT64_VECTOR_LENGTH) or any(
            value < 0 for value in values
        ):
            raise ValueError("composite commit contribution is malformed")
        signal = torch.tensor(values, dtype=torch.int64, device=self._device)
        if int(self._backend._redknot_tp_size) > 1:
            signal = self._backend._mla_off_control_all_reduce(signal)
        reduced = tuple(int(value) for value in signal.tolist())
        token = "sha256:" + hashlib.sha256(
            json.dumps(reduced, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        return CollectiveReduction(
            collective_token=token,
            reduced_int64=reduced,
        )

    def exchange_final_once(self, int64_vector):
        from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
            FORWARD_FINAL_INT64_VECTOR_LENGTH,
            ForwardFinalReduction,
        )

        values = tuple(int(value) for value in int64_vector)
        if len(values) != int(FORWARD_FINAL_INT64_VECTOR_LENGTH) or any(
            value < 0 for value in values
        ):
            raise ValueError("composite forward-final contribution is malformed")
        signal = torch.tensor(values, dtype=torch.int64, device=self._device)
        if int(self._backend._redknot_tp_size) > 1:
            signal = self._backend._mla_off_control_all_reduce(signal)
        reduced = tuple(int(value) for value in signal.tolist())
        token = "sha256:" + hashlib.sha256(
            json.dumps(reduced, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        return ForwardFinalReduction(
            collective_token=token,
            reduced_int64=reduced,
        )

    def coordinated_abort(self, signal) -> None:
        # This hook must not become a second success vote.  Recording the
        # fail-stop signal and raising on the originating rank prevents a
        # rank-local dense fallback after omitted slots have been consumed.
        self._backend._count("mla_off.composite_coordinated_aborts")
        logger.error(
            "RedKnot composite forward aborted: generation=%s layer_ordinal=%s "
            "rank=%s reason=%s detail=%s",
            getattr(signal, "generation_id", ""),
            getattr(signal, "forward_ordinal", -1),
            getattr(signal, "tp_rank", -1),
            getattr(signal, "reason_code", "unknown"),
            getattr(signal, "detail", ""),
        )
        self._backend._redknot_force_composite_fail_stop(signal)


@dataclass(frozen=True)
class _RedKnotPreparedRequestRestore:
    """One request/layer batch input with a request-scoped receipt validator."""

    request_index: int
    store: object = field(repr=False, compare=False)
    validated: object = field(repr=False, compare=False)
    receipt_adapter: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class _RedKnotPreparedSharedLayer:
    """All cache/dirty-builder inputs validated before the TP commit."""

    validated_restores: Tuple[_RedKnotPreparedRequestRestore, ...]
    cache_preflights: Tuple[object, ...]
    dirty_worksets: Tuple[Mapping[str, object], ...]
    attention_state_receipt: Mapping[str, object]
    indexer_state_receipt: Optional[Mapping[str, object]]
    compressed_target_loc: torch.Tensor
    indexer_target_loc: Optional[torch.Tensor]
    forward_token: str
    builder_epoch_token: str
    restore_adapter: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class _RedKnotPreparedZOffLayer:
    """Forward-wide z_off reservation with no shared-cache restoration."""

    cache_preflights: Tuple[object, ...]
    builder_epoch_token: str

    def __post_init__(self) -> None:
        if self.cache_preflights:
            raise ValueError("zoff-only reservation cannot retain cache preflights")
        if not isinstance(self.builder_epoch_token, str) or not self.builder_epoch_token:
            raise ValueError("zoff-only builder epoch token is absent")


@dataclass(frozen=True)
class _RedKnotLivePrefixContinuationReceipt:
    """Qualification-only proof of one completed chunked microforward."""

    request_token: str
    request_pool_index: int
    total_tokens: int
    completed_extent: int
    plan_digest: str
    forward_token: str
    terminal_state_digest: str
    terminal_state_slots: Tuple[Tuple[int, str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_token, str) or not self.request_token:
            raise ValueError("prefix continuation request token is absent")
        if type(self.request_pool_index) is not int or self.request_pool_index < 0:
            raise ValueError("prefix continuation request-pool index is invalid")
        if (
            type(self.total_tokens) is not int
            or type(self.completed_extent) is not int
            or self.total_tokens <= 0
            or self.completed_extent <= 0
            or self.completed_extent > self.total_tokens
        ):
            raise ValueError("prefix continuation token extent is invalid")
        if not isinstance(self.plan_digest, str) or not self.plan_digest.startswith(
            "sha256:"
        ):
            raise ValueError("prefix continuation plan digest is invalid")
        if not isinstance(self.forward_token, str) or not self.forward_token:
            raise ValueError("prefix continuation forward receipt is absent")
        if (
            not isinstance(self.terminal_state_digest, str)
            or not self.terminal_state_digest.startswith("sha256:")
        ):
            raise ValueError("prefix continuation terminal-state proof is absent")
        if (
            type(self.terminal_state_slots) is not tuple
            or not self.terminal_state_slots
            or tuple(sorted(set(self.terminal_state_slots)))
            != self.terminal_state_slots
        ):
            raise ValueError("prefix continuation terminal slots are invalid")


@dataclass
class _RedKnotForwardCompositeTransaction:
    """Forward-owned layer3..39 reservation/receipt lifecycle."""

    forward_id: str
    contexts: Mapping[int, object]
    q_arena: object
    q_reservations: Mapping[int, object]
    prepared_layers: Mapping[int, _RedKnotPreparedSharedLayer]
    merge_plans: Mapping[int, object]
    coordinator: object
    collective_adapter: object
    omission_profile: str
    restore_batch_receipt: object
    restore_pipeline_plan: object = field(default=None, repr=False, compare=False)
    failure_carrier: object = field(default=None, repr=False, compare=False)
    validated_restore_batch: object = field(
        default=None, repr=False, compare=False
    )
    slot_bounds_certificate: object = field(
        default=None, repr=False, compare=False
    )
    restore_stream: object = field(default=None, repr=False, compare=False)
    restore_completion_event: object = field(
        default=None, repr=False, compare=False
    )
    restore_stream_identity: Tuple[object, ...] = ()
    restore_group_events: Mapping[int, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    restore_layer_groups: Mapping[int, int] = field(
        default_factory=dict, repr=False, compare=False
    )
    restore_group_recorded: set = field(
        default_factory=set, repr=False, compare=False
    )
    restore_group_waited: set = field(
        default_factory=set, repr=False, compare=False
    )
    restore_event_recorded: bool = False
    restore_event_waited: bool = False
    all_layers_released: bool = False
    finalized: bool = False
    closed: bool = False

    def context_for(self, layer_id: int):
        if self.closed:
            raise RuntimeError("forward composite transaction is closed")
        try:
            return self.contexts[int(layer_id)]
        except KeyError as error:
            raise KeyError("forward transaction has no layer context") from error

    def wait_for_restore(self, device, *, layer_id: int) -> None:
        """Wait only for the restore group that owns ``layer_id``."""

        if self.closed:
            raise RuntimeError("forward composite transaction is closed")
        layer_id = int(layer_id)
        if self.restore_group_events:
            try:
                group = int(self.restore_layer_groups[layer_id])
                event = self.restore_group_events[group]
            except KeyError as error:
                raise RuntimeError(
                    "restore pipeline has no event for the requested layer"
                ) from error
            if group in self.restore_group_waited:
                return
            if group not in self.restore_group_recorded:
                raise RuntimeError("restore pipeline group event was never recorded")
        else:
            group = None
            event = self.restore_completion_event
            if event is None or self.restore_event_waited:
                return
            if not self.restore_event_recorded:
                raise RuntimeError("restore completion event was never recorded")
        if len(self.restore_stream_identity) != 2:
            raise RuntimeError("restore stream identity is absent")
        device = torch.device(device)
        expected_device = self.restore_stream_identity[0]
        if str(device) != str(expected_device):
            raise RuntimeError("restore event belongs to another CUDA device")
        torch.cuda.current_stream(device).wait_event(event)
        if group is not None:
            self.restore_group_waited.add(group)
            if event is self.restore_completion_event:
                self.restore_event_waited = True
        else:
            self.restore_event_waited = True

    def close(self) -> None:
        """Break tensor-owning context cycles after final/fail-stop teardown."""

        if self.closed:
            return
        if (
            self.coordinator is not None
            and bool(getattr(self.coordinator, "committed", False))
            and not self.finalized
        ):
            raise RuntimeError(
                "committed transaction cannot release GPU state before final"
            )
        if (
            self.restore_completion_event is not None
            and self.coordinator is not None
            and bool(getattr(self.coordinator, "committed", False))
            and not self.restore_event_waited
        ):
            raise RuntimeError(
                "restore workspace cannot release before its completion dependency"
            )
        for context in tuple(self.contexts.values()):
            if getattr(
                context, "_redknot_forward_composite_transaction", None
            ) is self:
                context._redknot_forward_composite_transaction = None
            context.sequential_q_reservation = None
            context.shared_restore_receipts = ()
            context.composite_layer_execution_receipt = None
            context.composite_omission_authorization = None
            context.composite_certificate = None
            context.composite_commit_session = None
            context.composite_collective_adapter = None
        self.contexts = {}
        self.q_reservations = {}
        self.prepared_layers = {}
        self.merge_plans = {}
        self.q_arena = None
        self.failure_carrier = None
        self.restore_batch_receipt = None
        self.restore_pipeline_plan = None
        self.validated_restore_batch = None
        self.slot_bounds_certificate = None
        self.restore_stream = None
        self.restore_completion_event = None
        self.restore_group_events = {}
        self.restore_layer_groups = {}
        self.restore_group_recorded.clear()
        self.restore_group_waited.clear()
        self.restore_event_recorded = False
        self.closed = True


@dataclass(frozen=True)
class _RedKnotSnapshotTPCertificate:
    """Rank-local snapshot proof plus one common all-rank digest."""

    certificate_digest: str
    snapshot_local_prepare_digest: str
    seg_hash: str
    token_hash: str
    model_hash: str
    policy_hash: str
    generation_id: str
    tp_rank: int
    tp_size: int
    ready_rank_count: int


def _mla_off_control_tensor_identity(tensor: torch.Tensor) -> Tuple[object, ...]:
    """Identify an immutable control tensor without reading device values."""

    try:
        version: object = int(tensor._version)
    except RuntimeError:
        version = "inference-immutable"
    return (
        id(tensor),
        int(tensor.data_ptr()),
        version,
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


@dataclass(frozen=True, eq=False)
class _MLAOffRestoreLayoutCertificate:
    """ForwardBatch-scoped proof for one immutable restore row layout.

    The plan digest and input tensor identities in ``layout_key`` are checked
    in every layer.  The structural validation below is performed once, then
    this certificate binds its exact mask/complement to every local-bearing
    layer.  Artifact commit epochs remain layer-specific and are intentionally
    revalidated by ``MLAOffRestoreView``; this cache never weakens that check.
    """

    layout_key: Tuple[object, ...]
    certified_layer_ids: Tuple[int, ...]
    restore_rows: Tuple[object, ...]
    reusable_cpu: torch.Tensor
    reuse_mask_digest: Tuple[int, int]
    reused_count: int
    dirty_rows_cpu: torch.Tensor
    segments_by_hash: Mapping[str, Mapping[str, object]]
    reusable_identity: Tuple[object, ...]
    dirty_identity: Tuple[object, ...]
    device_index_certificates: Dict[Tuple[object, ...], object] = field(
        default_factory=dict
    )

    def validate(
        self,
        *,
        layer_id: int,
        layout_key: Tuple[object, ...],
        reusable_cpu: torch.Tensor,
        dirty_rows_cpu: torch.Tensor,
        reuse_mask_digest: Tuple[int, int],
        q_rows: int,
        reused_count: Optional[int] = None,
        online_count: Optional[int] = None,
    ) -> None:
        """Fail closed if a cached mask/layout is reused outside its proof."""

        if int(layer_id) not in self.certified_layer_ids:
            raise ValueError("MLA-off restore layout is not certified for this layer")
        if self.layout_key != layout_key:
            raise ValueError("MLA-off restore layout plan/input certificate changed")
        if (
            self.reusable_cpu is not reusable_cpu
            or self.dirty_rows_cpu is not dirty_rows_cpu
            or self.reuse_mask_digest != tuple(reuse_mask_digest)
            or self.reusable_identity
            != _mla_off_control_tensor_identity(reusable_cpu)
            or self.dirty_identity
            != _mla_off_control_tensor_identity(dirty_rows_cpu)
            or reusable_cpu.ndim != 1
            or reusable_cpu.dtype != torch.bool
            or reusable_cpu.device.type != "cpu"
            or int(reusable_cpu.numel()) != int(q_rows)
            or dirty_rows_cpu.ndim != 1
            or dirty_rows_cpu.dtype != torch.long
            or dirty_rows_cpu.device.type != "cpu"
            or self.reused_count + int(dirty_rows_cpu.numel()) != int(q_rows)
            or (
                reused_count is not None
                and self.reused_count != int(reused_count)
            )
            or (
                online_count is not None
                and int(dirty_rows_cpu.numel()) != int(online_count)
            )
        ):
            raise ValueError("MLA-off restore mask certificate is stale")


def _validate_forced_local_window(window: int, swa_capacity: int) -> int:
    """Validate a window used when every logical head is forced local."""

    window = int(window)
    swa_capacity = int(swa_capacity)
    if window <= 0:
        raise ValueError(f"local window must be positive, got {window}")
    if window > swa_capacity:
        raise ValueError(
            f"local window {window} exceeds the physical SWA cache capacity "
            f"{swa_capacity}; the position-independent extra KV cache cannot "
            "silently be treated as an SWA window"
        )
    return window


def _build_dual_layer_pass_plan(
    head_cfg: DeepSeekV4MLAHeadConfig,
    *,
    layer_id: int,
    swa_capacity: int,
) -> _DualLayerPassPlan:
    """Group local heads by exact window and promote unexecutable policies.

    DeepSeek V4 stores one shared latent KV stream.  This plan therefore only
    controls which logical Q heads consume the SWA and compressed cache scopes;
    it never creates per-head copies of KV.  A local policy whose requested
    range exceeds the physical SWA capacity is explicitly promoted to global.
    """

    local_by_window: Dict[int, list[int]] = defaultdict(list)
    global_heads = []
    promoted_heads = []
    for logical_head in range(head_cfg.num_attention_heads):
        strategy = head_cfg.get_strategy(layer_id, logical_head)
        if not strategy.is_local():
            global_heads.append(logical_head)
            continue
        window = int(strategy.window)
        if window > int(swa_capacity):
            global_heads.append(logical_head)
            promoted_heads.append((logical_head, window))
            continue
        # Config normalization guarantees a positive local window, but keep the
        # execution helper independently safe for hand-constructed configs.
        window = _validate_forced_local_window(window, swa_capacity)
        local_by_window[window].append(logical_head)

    local_groups = tuple(
        (window, tuple(local_by_window[window])) for window in sorted(local_by_window)
    )
    return _DualLayerPassPlan(
        local_groups=local_groups,
        global_heads=tuple(global_heads),
        promoted_heads=tuple(promoted_heads),
    )


class RedKnotMLAAttnBackend(DeepseekV4AttnBackend):
    """RedKnot logical-head decoupling for DeepSeek V4 MLA.

    DeepSeek V4 has 64 logical query heads but one physical packed latent KV
    stream per layer.  The 448 no-PE dimensions and quantization scale are
    position independent; the 64 RoPE dimensions are relocated by the existing
    offline-reuse path.  This backend deliberately leaves that physical cache
    and its snapshot/restore format unchanged.

    Execution modes:

    * ``global`` delegates exactly to native DSV4;
    * ``dual`` is a FlashMLA correctness oracle (full-head pass per scope);
    * ``headwise`` computes only each TP rank's owned Q heads with arbitrary-head
      Triton kernels, reusing the same SWA/CSA/HCA cache tensors for every group;
    * ``local`` forces every head to the configured physical SWA window.

    The DeepSeek-V4-Flash-0731 production profile adds a non-overridable layer
    fence: layers 0..2 and 40..42 call native DSV4, while only layers 3..39 may
    construct pure offline-local/online-global MLA contexts.
    """

    def __init__(self, model_runner, *args, **kwargs):
        super().__init__(model_runner, *args, **kwargs)
        server_args = model_runner.server_args
        self.redknot_mla_pass_mode = server_args.redknot_mla_pass_mode
        self.redknot_v4_mode = os.environ.get("REDKNOT_V4_MODE", "correctness")
        if self.redknot_v4_mode == "correctness" and getattr(
            server_args, "redknot_sparse_ffn_enable", False
        ):
            raise ValueError(
                "RedKnot V4 correctness mode requires dense MoE; "
                "disable --redknot-sparse-ffn-enable"
            )
        hf_config = model_runner.model_config.hf_config
        self.redknot_dsv4_topology = deepseek_v4_redknot_topology(hf_config)
        cfg_path = getattr(server_args, "redknot_head_config_path", None)
        dense_prefix_layers = int(
            getattr(server_args, "redknot_mla_dense_prefix_layers", 3)
        )
        dense_suffix_layers = int(
            getattr(server_args, "redknot_mla_dense_suffix_layers", 3)
        )
        if cfg_path:
            self.redknot_mla_head_cfg = DeepSeekV4MLAHeadConfig.from_json(
                cfg_path,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
            )
        else:
            self.redknot_mla_head_cfg = DeepSeekV4MLAHeadConfig.from_model_config(
                hf_config,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                local_window=getattr(server_args, "redknot_mla_local_window", 128),
                global_head_stride=getattr(
                    server_args, "redknot_mla_global_head_stride", 8
                ),
                global_layer_stride=getattr(
                    server_args, "redknot_mla_global_layer_stride", 0
                ),
            )
        if (
            self.redknot_mla_head_cfg.num_layers
            != self.redknot_dsv4_topology["num_target_layers"]
        ):
            raise ValueError(
                "RedKnot MLA head config targets "
                f"{self.redknot_mla_head_cfg.num_layers} layers, but the model has "
                f"{self.redknot_dsv4_topology['num_target_layers']} target layers"
            )
        if self.redknot_mla_head_cfg.num_attention_heads != int(
            hf_config.num_attention_heads
        ):
            raise ValueError(
                "RedKnot MLA head config has "
                f"{self.redknot_mla_head_cfg.num_attention_heads} heads, but the "
                f"model has {hf_config.num_attention_heads} attention heads"
            )
        if self.redknot_mla_head_cfg.physical_kv_heads != 1:
            raise ValueError(
                "DeepSeek V4 RedKnot requires exactly one shared physical latent "
                f"KV head, got {self.redknot_mla_head_cfg.physical_kv_heads}"
            )
        if self.redknot_dsv4_topology["num_target_layers"] != (
            _PURE_HEADSPLIT_NUM_LAYERS
        ):
            raise ValueError(
                "pure DeepSeek-V4-Flash MLA split requires exactly 43 target "
                f"layers, got {self.redknot_dsv4_topology['num_target_layers']}"
            )
        dspark_target_layers = tuple(
            int(value)
            for value in getattr(hf_config, "dspark_target_layer_ids", ())
        )
        if dspark_target_layers and dspark_target_layers != (40, 41, 42):
            raise ValueError(
                "DeepSeek-V4-Flash pure headsplit expects D-Spark target layers "
                f"40,41,42; got {dspark_target_layers}"
            )
        if (
            self.redknot_mla_head_cfg.dense_prefix_layers
            != _PURE_HEADSPLIT_DENSE_PREFIX
            or self.redknot_mla_head_cfg.dense_suffix_layers
            != _PURE_HEADSPLIT_DENSE_SUFFIX
        ):
            raise ValueError(
                "pure DeepSeek-V4-Flash MLA split requires dense prefix/suffix "
                "3/3 so only layers 3..39 are reusable"
            )

        # The pool may reserve a wider physical row for alignment.  DSV4's
        # executable SWA contract remains SWA_WINDOW, so larger requested local
        # windows must be promoted rather than silently reading padding.
        self._redknot_swa_capacity = min(
            int(self.token_to_kv_pool.swa_window_size), SWA_WINDOW
        )
        self._redknot_dual_layer_plans = tuple(
            _build_dual_layer_pass_plan(
                self.redknot_mla_head_cfg,
                layer_id=layer_id,
                swa_capacity=self._redknot_swa_capacity,
            )
            for layer_id in range(self.redknot_mla_head_cfg.num_layers)
        )
        for layer_id, plan in enumerate(self._redknot_dual_layer_plans):
            if layer_id in _PURE_HEADSPLIT_DENSE_LAYER_IDS:
                if plan.local_groups:
                    raise ValueError(
                        f"dense boundary layer {layer_id} contains local heads"
                    )
                continue
            if layer_id not in _PURE_HEADSPLIT_OFFLINE_LAYER_IDS:
                raise ValueError(f"unexpected target layer {layer_id}")
            if not plan.local_groups or not plan.global_heads:
                raise ValueError(
                    "every middle layer 3..39 must contain both offline-local "
                    f"and online-global logical heads; layer={layer_id}"
                )
            if plan.promoted_heads:
                raise ValueError(
                    "pure head split cannot promote local heads at runtime; "
                    f"layer={layer_id} promoted={plan.promoted_heads}"
                )
        self._redknot_tp_rank = get_attention_tp_rank()
        self._redknot_tp_size = get_attention_tp_size()
        self._redknot_runtime_counters = Counter()
        self._redknot_logged_paths = set()
        self._redknot_mla_off_logged_failures = set()
        self._redknot_trace_actual_passes = (
            os.environ.get("REDKNOT_MLA_TRACE_PASSES", "0") == "1"
        )
        # Accuracy-first reuse mode: a head's offline-reuse eligibility must
        # not silently change its normal DSV4 attention scope.  When enabled,
        # heads classified as local/reusable still consume the same SWA +
        # compressed/indexer-selected candidates as global heads whenever they
        # are evaluated online.  The policy then controls only which head rows
        # may be restored from MLA-off, not which KV scopes define the head.
        self._redknot_reuse_heads_full_scope = (
            os.environ.get("REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE", "1") == "1"
        )
        if self._redknot_reuse_heads_full_scope and os.environ.get(
            "REDKNOT_C4_TOPK_CLAMP", ""
        ).strip():
            raise ValueError(
                "REDKNOT_C4_TOPK_CLAMP is unsafe in accuracy-first mode: "
                "DSV4 Top-512 indices have no score-sorted-prefix contract"
            )
        self._redknot_mla_off_enabled = (
            os.environ.get("REDKNOT_MLA_OFFLOAD", "0") == "1"
        )
        self._redknot_mla_off_global_attention_impl = os.environ.get(
            "REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL", "triton_h1"
        ).strip()
        if self._redknot_mla_off_global_attention_impl not in (
            _MLA_OFF_GLOBAL_ATTN_IMPLS
        ):
            raise ValueError(
                "REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL must be exactly one of "
                f"{_MLA_OFF_GLOBAL_ATTN_IMPLS}"
            )
        if (
            self._redknot_mla_off_global_attention_impl
            == "padded_flashmla_h64"
            and not self._redknot_mla_off_enabled
        ):
            raise ValueError(
                "padded FlashMLA H64 is valid only for MLA-off restore"
            )
        self._redknot_mla_off_global_q_workspace = None
        geometry_template_cache_raw = os.environ.get(
            "REDKNOT_MLA_OFF_GEOMETRY_TEMPLATE_CACHE", "0"
        )
        if geometry_template_cache_raw not in ("0", "1"):
            raise ValueError(
                "REDKNOT_MLA_OFF_GEOMETRY_TEMPLATE_CACHE must be exactly 0 or 1"
            )
        self._redknot_geometry_template_cache_enabled = (
            geometry_template_cache_raw == "1"
        )
        self._redknot_geometry_template_cache = {}
        # A repeated RAG query over one frozen document bundle has identical
        # projection/artifact and dirty-row identities.  The full merge
        # preflight walks every clean/dirty row and re-hashes the complete
        # plan, so rebuilding it for all 37 layers on every request turns
        # saved attention work back into Python control latency.  Entries are
        # live-validated on every hit and the bounded key contains tensor
        # object/storage/version identities, artifact-plan identity/digest,
        # and the exact head/weight geometry.
        self._redknot_zoff_merge_plan_cache = {}
        self._redknot_zoff_merge_plan_cache_limit = 512
        self._redknot_shared_latent_enabled = bool(
            self._redknot_mla_off_enabled
            and os.environ.get("REDKNOT_SHARED_LATENT_GPU", "1") == "1"
        )
        if self._redknot_mla_off_enabled and not self._redknot_shared_latent_enabled:
            raise ValueError(
                "context-bound pure MLA requires the atomic shared-latent snapshot path"
            )
        # A failed three-part snapshot confirmation can leave CPU, z_off, and
        # GPU participants at different irreversible epochs.  Such a worker
        # must never attempt another restore until it is restarted; unlike an
        # ordinary pre-publish failure, clearing a handle cannot make the
        # already-confirmed participant rollback-capable again.
        self._redknot_shared_latent_poisoned = False
        self._redknot_shared_latent_poison_reason = ""
        if self._redknot_shared_latent_enabled:
            from sglang.srt.layers.attention.redknot.dsv4_shared_latent_cache import (
                DSV4SharedLatentController,
            )

            self._redknot_shared_latent_controller = DSV4SharedLatentController()
            self._redknot_shared_gpu_stores = {}
            self._redknot_shared_snapshot_services = {}
            self._redknot_shared_snapshot_stages = {}
            self._redknot_shared_freqs_by_layer = {}
            # Production installs three certified pointer-table family
            # callbacks plus one persistent descriptor workspace.  Until that
            # provider exists, forward-wide reuse fails closed before commit;
            # a Python loop over 37 layers is intentionally not a fallback.
            self._redknot_restore_batch_workspace = None
            self._redknot_restore_batch_kernels = None
            self._redknot_restore_batch_preflight_kernel = None
            self._redknot_restore_batch_provider = None
            self._redknot_restore_batch_provider_common_token = None
            self._redknot_restore_batch_provider_local_token = None
            self._redknot_restore_batch_workspace_event = None
            self._redknot_restore_batch_retired_workspaces = []
            self._redknot_restore_batch_streams = {}
        self._redknot_three_way_closure = (
            os.environ.get("REDKNOT_THREE_WAY_CLOSURE", "0") == "1"
        )
        token_sparse_ffn = bool(
            getattr(server_args, "redknot_sparse_ffn_enable", False)
        )
        self._redknot_token_sparse_dense_suffix_layers = int(
            os.environ.get("REDKNOT_FFN_DENSE_SUFFIX_LAYERS", "0")
        )
        self._redknot_token_sparse_boundary_tokens = int(
            os.environ.get("REDKNOT_FFN_BOUNDARY_TOKENS", "128")
        )
        if not 0 <= self._redknot_token_sparse_dense_suffix_layers <= int(
            getattr(model_runner.model_config.hf_config, "num_hidden_layers", 0)
        ):
            raise ValueError("token-sparse FFN dense suffix layer count is invalid")
        if self._redknot_token_sparse_boundary_tokens < 0:
            raise ValueError("token-sparse FFN boundary tokens must be non-negative")
        if self._redknot_mla_off_enabled and token_sparse_ffn:
            if not self._redknot_three_way_closure:
                raise ValueError(
                    "pure MLA and token-sparse FFN require the explicit "
                    "REDKNOT_THREE_WAY_CLOSURE=1 qualification contract"
                )
            if os.environ.get("REDKNOT_MLA_OFF_QUALIFICATION_ONLY", "0") != "1":
                raise ValueError("the three-way closure is qualification-only")
        elif self._redknot_three_way_closure:
            raise ValueError(
                "REDKNOT_THREE_WAY_CLOSURE requires MLA-off and token-sparse FFN"
            )
        if self._redknot_mla_off_enabled and not self._redknot_reuse_heads_full_scope:
            raise ValueError(
                "context-bound pure MLA labels heads only for offline/online "
                "partitioning; REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE must be 1"
            )
        self._redknot_mla_off_execution_profile = os.environ.get(
            "REDKNOT_MLA_OFF_EXECUTION_PROFILE",
            _PURE_HEADSPLIT_PROFILE,
        ).strip()
        if self._redknot_mla_off_enabled and (
            self._redknot_mla_off_execution_profile
            not in _PURE_HEADSPLIT_PROFILES
        ):
            raise ValueError(
                "MLA-off requires one of the pure execution profiles "
                f"{_PURE_HEADSPLIT_PROFILES!r}; selected-row/indexer profiles "
                "cannot be mixed into the pure multi-head result"
            )
        self._redknot_server_instance_nonce = os.environ.get(
            "REDKNOT_SERVER_INSTANCE_NONCE", ""
        ).strip()
        if self._redknot_mla_off_enabled:
            if not self._redknot_server_instance_nonce:
                raise ValueError(
                    "context-bound MLA requires REDKNOT_SERVER_INSTANCE_NONCE"
                )
            if self._redknot_mla_off_execution_profile == _PURE_HEADSPLIT_PROFILE:
                from sglang.srt.layers.attention.redknot.dsv4_context_identity import (
                    ContextTokenStreamRegistry,
                )

                self._redknot_context_token_streams = ContextTokenStreamRegistry()
            else:
                # Independent-document artifacts are authenticated by their
                # exact token hash and local position-0 snapshot.  They do not
                # consume the cumulative-prefix registry used by the exact
                # context-bound profile.
                self._redknot_context_token_streams = None
        if self._redknot_mla_off_enabled and self.redknot_mla_pass_mode != "headwise":
            raise ValueError("pure MLA-off requires redknot_mla pass mode=headwise")
        compact_woa_raw = os.environ.get(
            "REDKNOT_MLA_OFF_COMPACT_WOA", "0"
        )
        if compact_woa_raw not in ("0", "1"):
            raise ValueError(
                "REDKNOT_MLA_OFF_COMPACT_WOA must be exactly 0 or 1"
            )
        self._redknot_mla_off_compact_woa = compact_woa_raw == "1"
        if self._redknot_mla_off_enabled and self._redknot_mla_off_compact_woa:
            raise ValueError(
                "pure MLA headsplit uses explicit W_L/W_G column slices and "
                "forbids the legacy selected-row compact wo_a path"
            )
        try:
            self._redknot_mla_off_certified_max_context_tokens = int(
                os.environ.get(
                    "REDKNOT_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS", "0"
                )
            )
        except ValueError as error:
            raise ValueError(
                "REDKNOT_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS must be an integer"
            ) from error
        if self._redknot_mla_off_certified_max_context_tokens < 0:
            raise ValueError(
                "REDKNOT_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS must be non-negative"
            )
        qualification_only_raw = os.environ.get(
            "REDKNOT_MLA_OFF_QUALIFICATION_ONLY", "0"
        )
        if qualification_only_raw not in ("0", "1"):
            raise ValueError(
                "REDKNOT_MLA_OFF_QUALIFICATION_ONLY must be exactly 0 or 1"
            )
        self._redknot_mla_off_qualification_only = (
            qualification_only_raw == "1"
        )
        try:
            self._redknot_mla_off_qualification_max_context_tokens = int(
                os.environ.get(
                    "REDKNOT_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS", "0"
                )
            )
        except ValueError as error:
            raise ValueError(
                "REDKNOT_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS must be an "
                "integer"
            ) from error
        if self._redknot_mla_off_qualification_max_context_tokens < 0:
            raise ValueError(
                "REDKNOT_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS must be "
                "non-negative"
            )
        if self._redknot_mla_off_qualification_only:
            if not self._redknot_mla_off_enabled:
                raise ValueError(
                    "MLA-off qualification-only mode requires MLA offload"
                )
            if self._redknot_mla_off_certified_max_context_tokens != 0:
                raise ValueError(
                    "MLA-off qualification-only mode requires the formal "
                    "certified max context to remain 0"
                )
            if self._redknot_mla_off_qualification_max_context_tokens <= 0:
                raise ValueError(
                    "MLA-off qualification-only mode requires a positive "
                    "qualification max context"
                )
        elif self._redknot_mla_off_qualification_max_context_tokens != 0:
            raise ValueError(
                "MLA-off qualification max context is valid only when "
                "qualification-only mode is enabled"
            )
        try:
            self._redknot_restore_pipeline_group_layers = int(
                os.environ.get(
                    "REDKNOT_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS", "0"
                )
            )
        except ValueError as error:
            raise ValueError(
                "REDKNOT_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS must be an integer"
            ) from error
        if not 0 <= self._redknot_restore_pipeline_group_layers <= len(
            _PURE_HEADSPLIT_OFFLINE_LAYER_IDS
        ):
            raise ValueError(
                "restore pipeline group size must be in [0, 37]"
            )
        if self._redknot_restore_pipeline_group_layers:
            if not self._redknot_mla_off_qualification_only:
                raise ValueError(
                    "restore pipeline is qualification-only until its grouped "
                    "launch/event oracle is certified"
                )
            if not self._redknot_shared_latent_enabled:
                raise ValueError(
                    "restore pipeline requires the persistent shared-latent store"
                )
        self._redknot_disable_radix_cache = bool(
            getattr(server_args, "disable_radix_cache", False)
        )
        self._redknot_radix_eviction_policy = str(
            getattr(server_args, "radix_eviction_policy", "lru")
        ).strip().lower()
        self._redknot_mla_prefix_materialization = (
            os.environ.get("REDKNOT_MLA_PREFIX_MATERIALIZATION", "0") == "1"
        )
        if (
            self._redknot_mla_prefix_materialization
            and not self._redknot_mla_off_enabled
        ):
            raise ValueError(
                "RedKnot prefix materialization requires MLA-offload"
            )
        if (
            self._redknot_mla_off_enabled
            and not self._redknot_disable_radix_cache
            and not self._redknot_mla_prefix_materialization
        ):
            raise ValueError(
                "context-bound pure MLA requires disable_radix_cache=true "
                "unless explicit prefix materialization is enabled"
            )
        if (
            self._redknot_mla_prefix_materialization
            and self._redknot_disable_radix_cache
        ):
            raise ValueError(
                "RedKnot prefix materialization requires the device radix "
                "cache to be enabled"
            )
        if (
            self._redknot_mla_prefix_materialization
            and self._redknot_radix_eviction_policy != "lfu"
        ):
            raise ValueError(
                "RedKnot prefix materialization requires radix eviction "
                "policy=lfu so repeated exact-prefix hits remain resident "
                "under concurrent one-shot dense traffic"
            )
        # This receipt table is deliberately qualification-only.  Formal
        # production restores must obtain a separately certified persistent
        # prefix-state provenance mechanism rather than inheriting this
        # single-request/no-radix diagnostic authorization.
        self._redknot_qualification_prefix_receipts = {}
        self._redknot_radix_prefix_receipts = {}
        self._redknot_mla_off_strict_row_verify = (
            os.environ.get("REDKNOT_MLA_OFF_STRICT_ROW_VERIFY", "0") == "1"
        )
        self._redknot_mla_off_dp_attention = bool(
            getattr(server_args, "enable_dp_attention", False)
        )
        self._redknot_mla_off_dp_size = int(
            getattr(server_args, "dp_size", 1)
        )
        self._redknot_mla_off_cp_size = int(get_attention_cp_size())
        self._redknot_mla_off_pp_size = int(
            getattr(model_runner, "pp_size", getattr(server_args, "pp_size", 1))
        )
        unsupported_parallel = []
        if self._redknot_mla_off_dp_size != 1:
            unsupported_parallel.append(
                f"request data parallelism size {self._redknot_mla_off_dp_size}"
            )
        if self._redknot_mla_off_dp_attention:
            unsupported_parallel.append("DP attention")
        if self._redknot_mla_off_cp_size != 1:
            unsupported_parallel.append(
                f"context parallel size {self._redknot_mla_off_cp_size}"
            )
        if self._redknot_mla_off_pp_size != 1:
            unsupported_parallel.append(
                f"pipeline parallel size {self._redknot_mla_off_pp_size}"
            )
        self._redknot_mla_off_unsupported_parallel = tuple(unsupported_parallel)
        if self._redknot_mla_off_enabled and unsupported_parallel:
            self._redknot_mla_off_enabled = False
            logger.warning(
                "RedKnot MLA-off disabled before request dispatch: v1 does not "
                "support %s",
                ", ".join(unsupported_parallel),
            )
        self._redknot_mla_off_local_layer_ids = tuple(
            layer_id
            for layer_id, plan in enumerate(self._redknot_dual_layer_plans)
            if plan.local_groups
        )
        if self._redknot_mla_off_local_layer_ids != (
            _PURE_HEADSPLIT_OFFLINE_LAYER_IDS
        ):
            raise ValueError(
                "effective MLA-off layer ids must be exactly 3..39, got "
                f"{self._redknot_mla_off_local_layer_ids}"
            )
        rank_head_start = self._redknot_tp_rank * (
            self.redknot_mla_head_cfg.num_attention_heads // self._redknot_tp_size
        )
        rank_head_end = rank_head_start + (
            self.redknot_mla_head_cfg.num_attention_heads // self._redknot_tp_size
        )
        rank_head_set = set(range(rank_head_start, rank_head_end))
        self._redknot_mla_off_rank_local_layer_ids = tuple(
            layer_id
            for layer_id, plan in enumerate(self._redknot_dual_layer_plans)
            if any(
                head in rank_head_set
                for _, heads in plan.local_groups
                for head in heads
            )
        )
        if self._redknot_mla_off_rank_local_layer_ids != (
            _PURE_HEADSPLIT_OFFLINE_LAYER_IDS
        ):
            raise ValueError(
                "every TP rank must own offline-local heads on every layer "
                "3..39"
            )
        heads_per_rank = (
            self.redknot_mla_head_cfg.num_attention_heads // self._redknot_tp_size
        )
        asymmetric_layers = []
        rank_scope_invalid_layers = []
        for layer_id in self._redknot_mla_off_local_layer_ids:
            local_heads = {
                head
                for _, heads in self._redknot_dual_layer_plans[layer_id].local_groups
                for head in heads
            }
            if any(
                not local_heads.intersection(
                    range(rank * heads_per_rank, (rank + 1) * heads_per_rank)
                )
                for rank in range(self._redknot_tp_size)
            ):
                asymmetric_layers.append(layer_id)
            # The packed restore kernel intentionally executes global heads on
            # every row and local heads only on dirty rows.  Until an all-local
            # persistent output kernel exists, every TP shard must therefore
            # own both scopes.  Accepting an all-local shard would only fail
            # after the composite certificate had authorized omitted Q slots.
            for rank in range(self._redknot_tp_size):
                rank_heads = set(
                    range(rank * heads_per_rank, (rank + 1) * heads_per_rank)
                )
                local_count = len(local_heads.intersection(rank_heads))
                if not 0 < local_count < heads_per_rank:
                    rank_scope_invalid_layers.append((layer_id, rank, local_count))
        if asymmetric_layers and self._redknot_mla_off_enabled:
            self._redknot_mla_off_enabled = False
            logger.warning(
                "RedKnot MLA-off disabled: every local-bearing layer must own "
                "at least one local head on every TP rank; asymmetric_layers=%s",
                tuple(asymmetric_layers),
            )
        if rank_scope_invalid_layers and self._redknot_mla_off_enabled:
            self._redknot_mla_off_enabled = False
            logger.warning(
                "RedKnot MLA-off disabled: each TP shard must own at least one "
                "local and one global logical head; invalid=%s",
                tuple(rank_scope_invalid_layers),
            )
        model_payload = {
            "model_path": str(getattr(model_runner.model_config, "model_path", "")),
            "name_or_path": str(getattr(hf_config, "_name_or_path", "")),
            "model_revision": str(
                getattr(model_runner.model_config, "revision", "")
                or getattr(hf_config, "_commit_hash", "")
            ),
            "hidden_size": int(getattr(hf_config, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(hf_config, "num_hidden_layers", 0)),
            "num_attention_heads": int(getattr(hf_config, "num_attention_heads", 0)),
            "num_key_value_heads": int(
                getattr(hf_config, "num_key_value_heads", 0)
            ),
            "o_groups": int(getattr(hf_config, "o_groups", 0)),
            "o_lora_rank": int(getattr(hf_config, "o_lora_rank", 0)),
            "head_dim": int(getattr(hf_config, "head_dim", 0)),
            "qk_rope_head_dim": int(
                getattr(hf_config, "qk_rope_head_dim", 0)
            ),
            "rope_theta": str(getattr(hf_config, "rope_theta", "")),
            "rope_scaling": getattr(hf_config, "rope_scaling", None),
            "compress_ratios": tuple(
                int(value)
                for value in (getattr(hf_config, "compress_ratios", ()) or ())
            ),
            "kv_cache_dtype": str(getattr(server_args, "kv_cache_dtype", "")),
            "dtype": str(getattr(model_runner.model_config, "dtype", "")),
        }
        self._redknot_mla_off_model_hash = hashlib.sha256(
            json.dumps(model_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        performance_claim_status = (
            "qualification_only_claim_ineligible"
            if self._redknot_mla_off_qualification_only
            else "unverified"
        )
        policy_payload = {
            "execution_profile": self._redknot_mla_off_execution_profile,
            "dense_prefix_layers": _PURE_HEADSPLIT_DENSE_PREFIX,
            "dense_suffix_layers": _PURE_HEADSPLIT_DENSE_SUFFIX,
            "dense_layer_ids": _PURE_HEADSPLIT_DENSE_LAYER_IDS,
            "offline_online_layer_ids": _PURE_HEADSPLIT_OFFLINE_LAYER_IDS,
            "selected_row_enabled": (
                self._redknot_mla_off_execution_profile
                == _COMBINED_ROW_SPARSE_PROFILE
            ),
            "indexer_hot_enabled": (
                self._redknot_mla_off_execution_profile
                == _COMBINED_ROW_SPARSE_PROFILE
            ),
            "disable_radix_cache": bool(self._redknot_disable_radix_cache),
            "prefix_materialization": bool(
                self._redknot_mla_prefix_materialization
            ),
            "radix_eviction_policy": self._redknot_radix_eviction_policy,
            "q_projection_scope": (
                "q_a_checkpoint_selected_rows_headsplit_v1"
                if self._redknot_mla_off_execution_profile
                == _COMBINED_ROW_SPARSE_PROFILE
                else "q_a_full_rows_native_dsv4_fullscope_skip0_v1"
            ),
            "head_scope_policy": _PURE_HEADSPLIT_HEAD_SCOPE_POLICY,
            "online_global_attention_impl": (
                self._redknot_mla_off_global_attention_impl
            ),
            "geometry_template_cache": bool(
                self._redknot_geometry_template_cache_enabled
            ),
            "kv_projection_scope": (
                "shared_clean_rows_gpu_restore_dirty_rows_wkv_v1"
            ),
            "compressor_projection_scope": (
                "shared_clean_blocks_gpu_restore_dirty_islands_online_v1"
            ),
            "shared_latent_restore_scope": (
                "persistent_gpu_layer_group_pipeline_v1"
                if self._redknot_restore_pipeline_group_layers
                else "persistent_gpu_ragged_fused_scatter_v1"
            ),
            "restore_pipeline_group_layers": int(
                self._redknot_restore_pipeline_group_layers
            ),
            "tp_commit_scope": (
                "composite_forward_prepare_and_full_layer_final_v6"
            ),
            "wo_a_projection_scope": "true_head_column_slices_v1",
            "performance_claim_status": performance_claim_status,
            "three_way_closure": bool(self._redknot_three_way_closure),
            "token_sparse_ffn_enabled": token_sparse_ffn,
            "token_sparse_ffn_importance": str(
                getattr(server_args, "redknot_sparse_ffn_importance", "activation")
            ),
            "token_sparse_ffn_mass_thresh": float(
                getattr(server_args, "redknot_sparse_ffn_mass_thresh", 1.0)
            ),
            "token_sparse_ffn_mass_thresh_deep": float(
                getattr(server_args, "redknot_sparse_ffn_mass_thresh_deep", 1.0)
            ),
            "token_sparse_ffn_min_full_ratio": float(
                getattr(server_args, "redknot_sparse_ffn_min_full_ratio", 0.0)
            ),
            "token_sparse_ffn_max_full_ratio": float(
                getattr(server_args, "redknot_sparse_ffn_max_full_ratio", 1.0)
            ),
            "token_sparse_ffn_dense_suffix_layers": int(
                self._redknot_token_sparse_dense_suffix_layers
            ),
            "token_sparse_ffn_boundary_tokens": int(
                self._redknot_token_sparse_boundary_tokens
            ),
            "reuse_heads_full_scope": bool(
                self._redknot_reuse_heads_full_scope
            ),
            "certified_max_context_tokens": int(
                self._redknot_mla_off_certified_max_context_tokens
            ),
            "qualification_only": bool(
                self._redknot_mla_off_qualification_only
            ),
            "qualification_max_context_tokens": int(
                self._redknot_mla_off_qualification_max_context_tokens
            ),
            "layers": [
                {
                    "local_groups": plan.local_groups,
                    "global_heads": plan.global_heads,
                    "promoted_heads": plan.promoted_heads,
                }
                for plan in self._redknot_dual_layer_plans
            ],
        }
        self._redknot_mla_off_policy_hash = hashlib.sha256(
            json.dumps(policy_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if self._redknot_shared_latent_enabled:
            self._initialize_mla_off_restore_batch_certificate()
        self._mla_off_initialize_tp_consensus()
        self._mla_off_write_runtime_manifest(
            model_runner=model_runner,
            server_args=server_args,
        )
        promoted = sum(
            len(plan.promoted_heads) for plan in self._redknot_dual_layer_plans
        )
        logger.info(
            "RedKnot MLA policy loaded: %s; DSV4 topology: %s; "
            "physical_kv_streams=1 swa_capacity=%d pass_mode=%s tp_head_shard=%d/%d "
            "promoted_heads=%d reuse_heads_full_scope=%s "
            "certified_max_context_tokens=%d qualification_only=%s "
            "qualification_max_context_tokens=%d",
            self.redknot_mla_head_cfg.summary(),
            self.redknot_dsv4_topology,
            self._redknot_swa_capacity,
            self.redknot_mla_pass_mode,
            self._redknot_tp_rank,
            self._redknot_tp_size,
            promoted,
            self._redknot_reuse_heads_full_scope,
            self._redknot_mla_off_certified_max_context_tokens,
            self._redknot_mla_off_qualification_only,
            self._redknot_mla_off_qualification_max_context_tokens,
        )
        if self._redknot_mla_off_enabled:
            logger.warning(
                "RedKnot context-bound MLA-off qualification is enabled; "
                "numerical equivalence remains gated by the real benchmark: "
                "local_layers=%s model_hash=%s policy_hash=%s",
                self._redknot_mla_off_local_layer_ids,
                self._redknot_mla_off_model_hash[:12],
                self._redknot_mla_off_policy_hash[:12],
            )

    def _mla_off_initialize_tp_consensus(self) -> None:
        """Disable the overlay uniformly if TP startup contracts drift."""

        if self._redknot_tp_size <= 1:
            if (
                self._redknot_shared_latent_enabled
                and getattr(
                    self, "_redknot_restore_batch_provider", None
                )
                is None
            ):
                self._redknot_mla_off_enabled = False
            return
        layer_payload = json.dumps(
            {
                "local_layers": self._redknot_mla_off_local_layer_ids,
            },
            sort_keys=True,
        )
        layer_hash = hashlib.sha256(layer_payload.encode("utf-8")).hexdigest()
        provider_token = str(
            getattr(
                self,
                "_redknot_restore_batch_provider_common_token",
                "",
            )
            or hashlib.sha256(b"restore-provider-unavailable").hexdigest()
        )
        if provider_token.startswith("sha256:"):
            provider_token = provider_token[7:]
        provider_ready = bool(
            not self._redknot_shared_latent_enabled
            or getattr(self, "_redknot_restore_batch_provider", None)
            is not None
        )
        values = [int(self._redknot_mla_off_enabled and provider_ready)]
        contract_digests = (
            self._redknot_mla_off_model_hash,
            self._redknot_mla_off_policy_hash,
            layer_hash,
            provider_token,
        )
        for digest in contract_digests:
            first = int(digest[:5], 16)
            second = int(digest[5:10], 16)
            values.extend((first, first * first, second, second * second))
        signal = torch.tensor(values, dtype=torch.int64, device=self.device)
        totals = tuple(
            int(value)
            for value in self._mla_off_control_all_reduce(signal).tolist()
        )
        world = int(self._redknot_tp_size)
        enabled_count = totals[0]
        contract_consistent = all(
            world * totals[offset + 1] == totals[offset] * totals[offset]
            and world * totals[offset + 3]
            == totals[offset + 2] * totals[offset + 2]
            for offset in range(1, 1 + 4 * len(contract_digests), 4)
        )
        if not contract_consistent:
            self._redknot_mla_off_enabled = False
            raise RuntimeError(
                "RedKnot MLA TP startup contract drift: model, effective head "
                "policy, local-layer layout, and certified restore provider "
                "must be identical on every rank"
            )
        if enabled_count == 0:
            self._redknot_mla_off_enabled = False
            return
        if enabled_count != world:
            self._redknot_mla_off_enabled = False
            logger.warning(
                "RedKnot MLA-off disabled on every TP rank: startup enable/model/"
                "head-policy/local-layer contracts are not identical"
            )
        elif not provider_ready:
            self._redknot_mla_off_enabled = False

    def _mla_off_write_runtime_manifest(self, *, model_runner, server_args) -> None:
        """Atomically publish rank-0 effective state after backend initialization."""

        output_path = os.environ.get("REDKNOT_SERVER_POLICY_MANIFEST_OUT", "")
        if not output_path or self._redknot_tp_rank != 0:
            return
        model_path = Path(
            str(getattr(model_runner.model_config, "model_path", ""))
        ).expanduser().resolve()
        head_cfg_raw = str(
            getattr(server_args, "redknot_head_config_path", "") or ""
        )
        head_cfg_path = (
            Path(head_cfg_raw).expanduser().resolve() if head_cfg_raw else None
        )

        def sha256_file(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
            MLA_OFF_FORMAT_VERSION,
            MLA_OFF_TOKEN_BYTES_PER_ROW,
            MLA_OFF_TRANSFER_AUDIT_SCHEMA,
            MLA_OFF_TRANSFER_BYTE_SEMANTICS,
            mla_off_layer_bytes_per_row,
        )

        hf_config = model_runner.model_config.hf_config
        o_groups = int(getattr(hf_config, "o_groups", 0))
        o_lora_rank = int(getattr(hf_config, "o_lora_rank", 0))
        output_groups_per_rank = (
            o_groups // int(self._redknot_tp_size)
            if o_groups > 0 and o_groups % int(self._redknot_tp_size) == 0
            else 0
        )
        layer_bytes_per_row = (
            mla_off_layer_bytes_per_row(output_groups_per_rank, o_lora_rank)
            if output_groups_per_rank > 0 and o_lora_rank > 0
            else 0
        )

        server_pid = os.getpid()
        _, pid_start_time_ticks = _linux_process_identity(server_pid)
        manifest = {
            "format": "redknot_mla_server_policy_v4",
            "backend_ready": True,
            "server_instance_nonce": os.environ.get(
                "REDKNOT_SERVER_INSTANCE_NONCE", ""
            ),
            "pid": server_pid,
            "pid_start_time_ticks": pid_start_time_ticks,
            "port": int(server_args.port),
            "chunked_prefill_size": int(server_args.chunked_prefill_size),
            "max_prefill_tokens": int(server_args.max_prefill_tokens),
            "model_path": str(model_path),
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "head_config_path": str(head_cfg_path) if head_cfg_path else "",
            "head_config_sha256": (
                sha256_file(head_cfg_path) if head_cfg_path else ""
            ),
            "tp_size": int(self._redknot_tp_size),
            "mla_pass_mode": str(self.redknot_mla_pass_mode),
            "reuse_heads_full_scope": bool(
                self._redknot_reuse_heads_full_scope
            ),
            "mla_off_certified_max_context_tokens": int(
                self._redknot_mla_off_certified_max_context_tokens
            ),
            "mla_off_qualification_only": bool(
                self._redknot_mla_off_qualification_only
            ),
            "mla_off_qualification_max_context_tokens": int(
                self._redknot_mla_off_qualification_max_context_tokens
            ),
            "mla_off_effective_restore_max_context_tokens": int(
                self._redknot_mla_off_qualification_max_context_tokens
                if self._redknot_mla_off_qualification_only
                else self._redknot_mla_off_certified_max_context_tokens
            ),
            "dense_prefix_layers": int(
                getattr(server_args, "redknot_mla_dense_prefix_layers", 3)
            ),
            "dense_suffix_layers": int(
                getattr(server_args, "redknot_mla_dense_suffix_layers", 3)
            ),
            "execution_profile": self._redknot_mla_off_execution_profile,
            "dense_layer_ids": list(_PURE_HEADSPLIT_DENSE_LAYER_IDS),
            "offline_online_layer_ids": list(
                _PURE_HEADSPLIT_OFFLINE_LAYER_IDS
            ),
            "selected_row_enabled": (
                self._redknot_mla_off_execution_profile
                == _COMBINED_ROW_SPARSE_PROFILE
            ),
            "indexer_hot_enabled": (
                self._redknot_mla_off_execution_profile
                == _COMBINED_ROW_SPARSE_PROFILE
            ),
            "disable_radix_cache": bool(self._redknot_disable_radix_cache),
            "prefix_materialization": bool(
                self._redknot_mla_prefix_materialization
            ),
            "radix_eviction_policy": self._redknot_radix_eviction_policy,
            "local_window": int(
                getattr(server_args, "redknot_mla_local_window", 128)
            ),
            "global_head_stride": int(
                getattr(server_args, "redknot_mla_global_head_stride", 8)
            ),
            "global_layer_stride": int(
                getattr(server_args, "redknot_mla_global_layer_stride", 0)
            ),
            "mla_off_max_bytes": int(
                os.environ.get("REDKNOT_MLA_OFF_MAX_BYTES", str(8 * 1024**3))
            ),
            "mla_off_device_max_bytes": int(
                os.environ.get("REDKNOT_MLA_OFF_DEVICE_MAX_BYTES", "0")
            ),
            "mla_off_device_cache_enabled": (
                int(os.environ.get("REDKNOT_MLA_OFF_DEVICE_MAX_BYTES", "0")) > 0
            ),
            "mla_off_device_representation": "bfloat16_same_epoch_mirror_v1",
            "mla_off_transfer_audit_format": MLA_OFF_TRANSFER_AUDIT_SCHEMA,
            "mla_off_transfer_byte_semantics": (
                MLA_OFF_TRANSFER_BYTE_SEMANTICS
            ),
            "mla_offload_enabled": bool(self._redknot_mla_off_enabled),
            "mla_off_compact_woa_enabled": bool(
                getattr(self, "_redknot_mla_off_compact_woa", False)
            ),
            "redknot_v4_mode": str(self.redknot_v4_mode),
            "three_way_closure": bool(self._redknot_three_way_closure),
            "token_sparse_ffn_enabled": bool(
                getattr(server_args, "redknot_sparse_ffn_enable", False)
            ),
            "token_sparse_ffn_importance": str(
                getattr(server_args, "redknot_sparse_ffn_importance", "activation")
            ),
            "token_sparse_ffn_mass_thresh": float(
                getattr(server_args, "redknot_sparse_ffn_mass_thresh", 1.0)
            ),
            "token_sparse_ffn_mass_thresh_deep": float(
                getattr(server_args, "redknot_sparse_ffn_mass_thresh_deep", 1.0)
            ),
            "token_sparse_ffn_min_full_ratio": float(
                getattr(server_args, "redknot_sparse_ffn_min_full_ratio", 0.0)
            ),
            "token_sparse_ffn_max_full_ratio": float(
                getattr(server_args, "redknot_sparse_ffn_max_full_ratio", 1.0)
            ),
            "token_sparse_ffn_dense_suffix_layers": int(
                self._redknot_token_sparse_dense_suffix_layers
            ),
            "token_sparse_ffn_boundary_tokens": int(
                self._redknot_token_sparse_boundary_tokens
            ),
            "attention_backend": "redknot_mla",
            "dp_size": int(self._redknot_mla_off_dp_size),
            "dp_attention": bool(self._redknot_mla_off_dp_attention),
            "cp_size": int(self._redknot_mla_off_cp_size),
            "pp_size": int(self._redknot_mla_off_pp_size),
            "swa_capacity": int(self._redknot_swa_capacity),
            "o_groups": o_groups,
            "o_lora_rank": o_lora_rank,
            "output_groups_per_rank": output_groups_per_rank,
            "mla_off_format_version": int(MLA_OFF_FORMAT_VERSION),
            "mla_off_storage_dtype": "bfloat16",
            "mla_off_token_bytes_per_row": int(MLA_OFF_TOKEN_BYTES_PER_ROW),
            "mla_off_layer_bytes_per_row": layer_bytes_per_row,
            "effective_head_policy_hash": self._redknot_mla_off_policy_hash,
            "model_compat_hash": self._redknot_mla_off_model_hash,
            "q_projection_scope": (
                "q_a_checkpoint_selected_rows_headsplit_v1"
                if self._redknot_mla_off_execution_profile
                == _COMBINED_ROW_SPARSE_PROFILE
                else "q_a_full_rows_native_dsv4_fullscope_skip0_v1"
            ),
            "head_scope_policy": _PURE_HEADSPLIT_HEAD_SCOPE_POLICY,
            "online_global_attention_impl": str(
                getattr(
                    self,
                    "_redknot_mla_off_global_attention_impl",
                    "triton_h1",
                )
            ),
            "geometry_template_cache": bool(
                getattr(
                    self,
                    "_redknot_geometry_template_cache_enabled",
                    False,
                )
            ),
            "kv_projection_scope": (
                "shared_clean_rows_gpu_restore_dirty_rows_wkv_v1"
            ),
            "compressor_projection_scope": (
                "shared_clean_blocks_gpu_restore_dirty_islands_online_v1"
            ),
            "shared_latent_restore_scope": (
                "persistent_gpu_layer_group_pipeline_v1"
                if self._redknot_restore_pipeline_group_layers
                else "persistent_gpu_ragged_fused_scatter_v1"
            ),
            "restore_pipeline_group_layers": int(
                self._redknot_restore_pipeline_group_layers
            ),
            "tp_commit_scope": (
                "composite_forward_prepare_and_full_layer_final_v6"
            ),
            "wo_a_projection_scope": "true_head_column_slices_v1",
            "performance_claim_status": (
                "qualification_only_claim_ineligible"
                if self._redknot_mla_off_qualification_only
                else "unverified"
            ),
            "runtime_local_layer_ids": list(
                self._redknot_mla_off_rank_local_layer_ids
            ),
            "batch_restore_provider_ready": bool(
                getattr(self, "_redknot_restore_batch_provider", None)
                is not None
            ),
            "batch_restore_provider_common_token": str(
                getattr(
                    self,
                    "_redknot_restore_batch_provider_common_token",
                    "",
                )
                or ""
            ),
            "batch_restore_provider_local_token": str(
                getattr(
                    self,
                    "_redknot_restore_batch_provider_local_token",
                    "",
                )
                or ""
            ),
            "batch_restore_provider_error": str(
                getattr(
                    self, "_redknot_restore_batch_provider_error", ""
                )
                or ""
            ),
        }
        restore_provider = getattr(
            self, "_redknot_restore_batch_provider", None
        )
        if restore_provider is not None:
            certificate = restore_provider.certificate
            manifest["batch_restore_oracle_evidence"] = {
                "kernel_source_sha256": str(
                    certificate.kernel_source_sha256
                ),
                "oracle_source_sha256": str(
                    certificate.oracle_source_sha256
                ),
                "torch_version": str(certificate.torch_version),
                "triton_version": str(certificate.triton_version),
                "cuda_runtime_version": str(
                    certificate.cuda_runtime_version
                ),
                "device_name": str(certificate.device_name),
                "device_capability": list(
                    certificate.device_capability
                ),
                "strict_pass": bool(
                    certificate.report.get("strict_pass", False)
                ),
            }
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def runtime_counters(self) -> Dict[str, int]:
        return dict(self._redknot_runtime_counters)

    def _count(self, key: str, value: int = 1) -> None:
        self._redknot_runtime_counters[key] += int(value)

    def _record_path(
        self,
        path: str,
        *,
        layer_id: int,
        plan: _DualLayerPassPlan,
        logical_heads: Tuple[int, ...],
    ) -> None:
        self._count(f"path.{path}")
        if not self._redknot_trace_actual_passes:
            return
        key = (path, layer_id, logical_heads)
        if key in self._redknot_logged_paths:
            return
        self._redknot_logged_paths.add(key)
        logger.info(
            "RedKnot MLA execution: path=%s layer=%d tp_rank=%s/%s "
            "logical_heads=%s local_groups=%s global_heads=%s promoted=%s "
            "physical_kv_streams=1",
            path,
            layer_id,
            getattr(self, "_redknot_tp_rank", 0),
            getattr(self, "_redknot_tp_size", 1),
            logical_heads,
            plan.local_groups,
            plan.global_heads,
            plan.promoted_heads,
        )

    def _native_forward(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool,
        attn_sink: Optional[torch.Tensor],
        kwargs,
    ) -> torch.Tensor:
        return super().forward(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            compress_ratio=compress_ratio,
            save_kv_cache=save_kv_cache,
            attn_sink=attn_sink,
            **kwargs,
        )

    def _headwise_owned_view(
        self, q_heads: int, layer_tp_heads: int
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Return ``(logical_head_ids, q_axis_indices)`` owned by this TP rank."""

        total_heads = self.redknot_mla_head_cfg.num_attention_heads
        layer_tp_heads = int(layer_tp_heads)
        if layer_tp_heads <= 0 or total_heads % layer_tp_heads != 0:
            return None
        if layer_tp_heads == total_heads:
            # DSA context-parallel prefill deliberately presents all 64 heads on
            # each CP worker and resets the model-side TP head rank to zero.
            logical_heads = tuple(range(total_heads))
        else:
            tp_rank = getattr(self, "_redknot_tp_rank", None)
            if tp_rank is None:
                tp_rank = get_attention_tp_rank()
            tp_rank = int(tp_rank)
            logical_start = tp_rank * layer_tp_heads
            if logical_start + layer_tp_heads > total_heads:
                return None
            logical_heads = tuple(range(logical_start, logical_start + layer_tp_heads))
        if q_heads == total_heads:
            # DSV4 pads Q back to 64 heads for FlashMLA.  Only this rank's slice
            # is initialized; selecting it before Triton avoids both undefined Q
            # reads and the native full-64-head replication on every TP rank.
            return logical_heads, logical_heads
        if q_heads == layer_tp_heads:
            return logical_heads, tuple(range(layer_tp_heads))
        return None

    def accepts_rank_local_q(self, *, layer_id: int, forward_batch=None) -> bool:
        """Whether the model may omit the native 64-head Q presentation.

        Only middle target layers use the arbitrary-head backend.  Dense
        prefix/suffix and draft layers retain native FlashMLA's full-head
        contract.
        """

        if forward_batch is None:
            # The execution mode is part of this authorization.  Refuse an
            # incomplete query rather than allowing an auxiliary draft path
            # to receive rank-local Q under native FlashMLA's 64-head ABI.
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is None or forward_mode.is_draft_extend(include_v2=True):
            return False
        authorizations = getattr(
            forward_batch, "_redknot_rank_local_q_authorizations", None
        )
        return bool(
            self.redknot_mla_pass_mode == "headwise"
            and isinstance(authorizations, Mapping)
            and authorizations.get(int(layer_id)) is True
        )

    @staticmethod
    def _set_rank_local_q_authorization(
        forward_batch: ForwardBatch,
        *,
        layer_id: int,
        context,
    ) -> None:
        authorizations = getattr(
            forward_batch, "_redknot_rank_local_q_authorizations", None
        )
        if not isinstance(authorizations, dict):
            authorizations = {}
            forward_batch._redknot_rank_local_q_authorizations = authorizations
        authorizations[int(layer_id)] = bool(
            context is not None
            and getattr(context, "is_restore", False)
            and int(getattr(context, "reused_row_count", 0)) > 0
        )

    @staticmethod
    def _mla_off_composite_committed(context) -> bool:
        return bool(
            context is not None
            and getattr(context, "composite_certificate", None) is not None
            and getattr(context, "composite_omission_authorization", None)
            is not None
        )

    def _abort_mla_off_composite(
        self, context, *, reason_code: str, detail: str, device
    ) -> None:
        session = getattr(context, "composite_commit_session", None)
        adapter = getattr(context, "composite_collective_adapter", None)
        fail_closed = getattr(session, "fail_closed_after_commit", None)
        if not callable(fail_closed) or adapter is None:
            raise RuntimeError(
                f"composite restore failed without abort controller: {detail}"
            )
        fail_closed(
            adapter,
            reason_code=str(reason_code),
            detail=str(detail),
        )

    def _redknot_force_composite_fail_stop(self, abort_signal) -> None:
        """Make an indeterminate/postcommit TP failure process-fatal.

        Raising a Python exception is not sufficient: peers may be executing
        a different NCCL collective and the scheduler can otherwise retain
        committed cache pins forever.  Tests may install an explicit hook;
        production first aborts/destroys the attention process group when the
        runtime exposes such an API, then terminates this worker so the SGLang
        supervisor tears down the remaining rank group.
        """

        test_hook = getattr(self, "_redknot_composite_fail_stop_hook", None)
        if callable(test_hook):
            test_hook(abort_signal)
            return
        try:
            tp_group = get_attention_tp_group()
            device_group = getattr(tp_group, "device_group", None)
            abort_group = getattr(device_group, "abort", None)
            if callable(abort_group):
                abort_group()
            elif device_group is not None and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group(device_group)
        except BaseException:
            # Process termination below is the authoritative fail-stop.  A
            # communicator teardown error must not convert it to a local
            # recoverable exception.
            logger.exception("RedKnot attention-TP group abort failed")
        os.kill(os.getpid(), signal.SIGABRT)
        raise RuntimeError("RedKnot fail-stop signal unexpectedly returned")

    def _fail_stop_mla_off_transaction(
        self,
        transaction: _RedKnotForwardCompositeTransaction,
        *,
        reason_code: str,
        detail: str,
    ) -> None:
        """Terminate a rank group without relying on commit-session state."""

        from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
            CoordinatedAbortSignal,
        )

        coordinator = transaction.coordinator
        proposal = getattr(coordinator, "proposal", None)
        identity = getattr(proposal, "identity", None)
        session = getattr(coordinator, "session", None)
        certificate = getattr(session, "certificate", None)
        try:
            proposal_digest = str(getattr(proposal, "digest", "") or "")
            if not (
                proposal_digest.startswith("sha256:")
                and len(proposal_digest) == 71
            ):
                proposal_digest = "sha256:" + hashlib.sha256(
                    f"{transaction.forward_id}\0{reason_code}".encode("utf-8")
                ).hexdigest()
            abort_signal = CoordinatedAbortSignal(
                generation_id=str(
                    getattr(identity, "generation_id", transaction.forward_id)
                ),
                forward_ordinal=int(getattr(identity, "forward_ordinal", 0)),
                tp_rank=int(self._redknot_tp_rank),
                reason_code=str(reason_code),
                detail=str(detail),
                proposal_digest=proposal_digest,
                certificate_digest=str(
                    getattr(certificate, "digest", "") or ""
                ),
                collective_token=str(
                    getattr(certificate, "collective_token", "") or ""
                ),
            )
        except BaseException:
            # Do not let diagnostic materialization obstruct process-group
            # termination after a CUDA/OOM/control-plane fault.
            self._redknot_force_composite_fail_stop(transaction)
            raise RuntimeError("emergency composite fail-stop returned")
        try:
            transaction.collective_adapter.coordinated_abort(abort_signal)
        except BaseException:
            self._redknot_force_composite_fail_stop(abort_signal)
            raise
        # A test hook is permitted to observe the signal without killing the
        # process.  It must never make production code continue locally.
        raise RuntimeError("composite fail-stop hook unexpectedly returned")

    def _fail_stop_mla_off_without_transaction(
        self, *, forward_id: str, reason_code: str, detail: str
    ) -> None:
        """Process-fatal cleanup path before a coordinator can be published."""

        from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
            CoordinatedAbortSignal,
        )

        signal_payload = f"{forward_id}\0{reason_code}\0{detail}"
        try:
            abort_signal = CoordinatedAbortSignal(
                generation_id=str(forward_id or "redknot-unpublished-forward"),
                forward_ordinal=0,
                tp_rank=int(self._redknot_tp_rank),
                reason_code=str(reason_code),
                detail=str(detail),
                proposal_digest="sha256:"
                + hashlib.sha256(signal_payload.encode("utf-8")).hexdigest(),
            )
        except BaseException:
            self._redknot_force_composite_fail_stop(self)
            raise RuntimeError("emergency precommit fail-stop returned")
        self._redknot_force_composite_fail_stop(abort_signal)
        raise RuntimeError("composite fail-stop hook unexpectedly returned")

    def _cleanup_uncommitted_mla_off_forward_collectively(
        self, *, forward_batch: ForwardBatch, device, stage: str
    ) -> None:
        """Close every local pin, then prove cleanup success on every TP rank."""

        transaction = getattr(
            forward_batch, "_redknot_mla_off_forward_transaction", None
        )
        resources = getattr(
            forward_batch, "_redknot_composite_forward_resources", None
        )
        forward_id = str(
            getattr(resources, "forward_id", "redknot-unpublished-forward")
        )
        local_error = None
        try:
            if isinstance(transaction, _RedKnotForwardCompositeTransaction):
                coordinator = transaction.coordinator
                if bool(getattr(coordinator, "committed", False)):
                    raise RuntimeError(
                        "cleanup helper received a committed transaction"
                    )
                transaction.close()
            if resources is not None:
                resources.close()
        except BaseException as error:
            local_error = error
        try:
            cleanup_signal = torch.tensor(
                [int(local_error is None)],
                dtype=torch.int64,
                device=torch.device(device),
            )
            if int(self._redknot_tp_size) > 1:
                cleanup_signal = self._mla_off_control_all_reduce(
                    cleanup_signal
                )
            ready_count = int(cleanup_signal.tolist()[0])
        except BaseException as vote_error:
            self._fail_stop_mla_off_without_transaction(
                forward_id=forward_id,
                reason_code="uncommitted_cleanup_vote_indeterminate",
                detail=f"stage={stage} {type(vote_error).__name__}: {vote_error}",
            )
        if ready_count != int(self._redknot_tp_size):
            self._fail_stop_mla_off_without_transaction(
                forward_id=forward_id,
                reason_code="uncommitted_cleanup_failed",
                detail=(
                    f"stage={stage} ready={ready_count}/{int(self._redknot_tp_size)} "
                    f"local={type(local_error).__name__ if local_error else 'ok'}"
                ),
            )
        forward_batch._redknot_composite_forward_resources = None
        forward_batch._redknot_mla_off_forward_transaction = None

    def finish_mla_off_forward_resources(
        self, *, mla_off_context, forward_batch: ForwardBatch
    ) -> None:
        """Release a layer lease; close persistent epochs after layer 39."""

        if mla_off_context is None:
            return
        resources = getattr(
            mla_off_context, "_redknot_composite_forward_resources", None
        )
        if resources is not None:
            from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                release_composite_restore_context,
            )
            transaction = getattr(
                mla_off_context,
                "_redknot_forward_composite_transaction",
                None,
            )
            if (
                isinstance(transaction, _RedKnotForwardCompositeTransaction)
                and transaction.coordinator is not None
            ):
                # The fixed final rendezvous belongs to the complete decoder
                # layer boundary, not self-attention: layer-39 wo_b and MLP
                # must return before the model owner finalizes.  Keep pins
                # reachable after the last attention context is released.
                complete = release_composite_restore_context(
                    mla_off_context,
                    close_when_forward_complete=False,
                )
                transaction.all_layers_released = bool(complete)
                return
            complete = release_composite_restore_context(mla_off_context)
            if complete:
                if isinstance(
                    transaction, _RedKnotForwardCompositeTransaction
                ):
                    transaction.close()
                forward_batch._redknot_mla_off_forward_transaction = None
            return
        if int(getattr(mla_off_context, "layer_id", -1)) == int(
            _PURE_HEADSPLIT_OFFLINE_LAYER_IDS[-1]
        ):
            self._release_mla_off_shared_forward_pins(forward_batch)

    def finalize_mla_off_forward_transaction(
        self, *, forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        """Finalize exactly once after the complete decoder layer 39 returns."""

        transaction = getattr(
            forward_batch, "_redknot_mla_off_forward_transaction", None
        )
        if not isinstance(transaction, _RedKnotForwardCompositeTransaction):
            resources = getattr(
                forward_batch, "_redknot_composite_forward_resources", None
            )
            if resources is None:
                return False
            resources.close()
            forward_batch._redknot_composite_forward_resources = None
            return True
        coordinator = transaction.coordinator
        if coordinator is None:
            return False
        if int(layer_id) != int(_PURE_HEADSPLIT_OFFLINE_LAYER_IDS[-1]):
            raise ValueError("forward finalization is only valid after layer 39")
        if transaction.finalized:
            raise RuntimeError("forward transaction was finalized twice")
        resources = getattr(
            forward_batch, "_redknot_composite_forward_resources", None
        )
        if resources is None or not transaction.all_layers_released:
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="forward_final_resource_domain_incomplete",
                detail="not all layer3..39 contexts reached their full-layer boundary",
            )
        prefix_receipt_proof = None
        try:
            prefix_receipt_proof = (
                self._redknot_prepare_qualification_prefix_receipt(
                    resources=resources,
                    forward_batch=forward_batch,
                    transaction=transaction,
                )
            )
        except BaseException as prefix_error:
            coordinator.record_pipeline_failure(
                layer_id=int(resources.expected_layer_ids[-1]),
                stage="prefix_continuation_receipt",
                error=prefix_error,
            )
        try:
            final_certificate = coordinator.finalize(
                transaction.collective_adapter
            )
            # Only a completed all-layer final rendezvous may authorize the
            # resident compressor state consumed by the next chunked
            # microforward.  The helper is a no-op outside the explicitly
            # ineligible qualification-only/no-radix/single-request mode.
            self._redknot_record_qualification_prefix_receipt(
                prepared_proof=prefix_receipt_proof,
                final_certificate=final_certificate,
            )
        except BaseException as final_error:
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="forward_final_failed",
                detail=f"{type(final_error).__name__}: {final_error}",
            )
        # Mark success only after a real ForwardExecutionCertificate returns.
        transaction.finalized = True
        try:
            resources.close()
            forward_batch._redknot_composite_forward_resources = None
            transaction.close()
            forward_batch._redknot_mla_off_forward_transaction = None
        except BaseException as close_error:
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="forward_postfinal_cleanup_failed",
                detail=f"{type(close_error).__name__}: {close_error}",
            )
        return True

    def abort_mla_off_forward_transaction(
        self,
        *,
        forward_batch: ForwardBatch,
        error: BaseException,
        layer_id: int = -1,
    ) -> bool:
        """Fail-stop an unexpected model escape after forward certification."""

        transaction = getattr(
            forward_batch, "_redknot_mla_off_forward_transaction", None
        )
        if not isinstance(transaction, _RedKnotForwardCompositeTransaction):
            return False
        coordinator = transaction.coordinator
        indeterminate_prepare = bool(
            coordinator is not None
            and bool(getattr(coordinator.session, "collective_attempted", False))
            and str(
                getattr(coordinator.session, "collective_decision", "")
            )
            != "rejected"
            and not transaction.finalized
        )
        if indeterminate_prepare:
            failure_layer = int(layer_id)
            if failure_layer not in tuple(coordinator.proposal.reusable_layer_ids):
                failure_layer = int(coordinator.proposal.reusable_layer_ids[0])
            if coordinator.ledger is not None:
                coordinator.record_pipeline_failure(
                    layer_id=failure_layer,
                    stage=f"model_escape_layer_{int(layer_id)}",
                    error=error,
                )
            # An unexpected escape cannot preserve the intervening TP
            # collective order, so entering forward-final early would itself
            # mismatch peers.  Use the session's out-of-band production
            # fail-stop path; ordinary close is forbidden until then.
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="model_pipeline_escape",
                detail=f"layer={int(layer_id)} {type(error).__name__}: {error}",
            )

        # No certificate exists (or query-suffix full-local was selected), so
        # local dense cleanup remains safe.
        resources = getattr(
            forward_batch, "_redknot_composite_forward_resources", None
        )
        if resources is not None:
            resources.close()
            forward_batch._redknot_composite_forward_resources = None
        transaction.close()
        forward_batch._redknot_mla_off_forward_transaction = None
        return True

    @staticmethod
    def _mla_off_postcommit_error_is_fatal(error: BaseException) -> bool:
        """Allow carriers only for ordinary, clearly local logic failures."""

        oom_type = getattr(torch.cuda, "OutOfMemoryError", MemoryError)
        if not isinstance(error, Exception) or isinstance(
            error, (MemoryError, oom_type)
        ):
            return True
        if not isinstance(
            error,
            (AssertionError, IndexError, KeyError, RuntimeError, TypeError, ValueError),
        ):
            return True
        type_name = type(error).__name__.lower()
        if any(
            marker in type_name
            for marker in ("cancel", "cuda", "nccl", "distributed", "timeout")
        ):
            return True
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "out of memory",
                "device-side assert",
                "illegal memory access",
                "cuda error",
                "nccl",
                "cublas",
                "launch failure",
                "misaligned address",
            )
        )

    def make_mla_off_failure_carrier(
        self,
        *,
        mla_off_context,
        reference: torch.Tensor,
        shape: Tuple[int, ...],
        stage: str,
        error: BaseException,
    ) -> torch.Tensor:
        """Return a pre-certified shape carrier, or fail-stop immediately."""

        transaction = getattr(
            mla_off_context,
            "_redknot_forward_composite_transaction",
            None,
        )
        if not isinstance(transaction, _RedKnotForwardCompositeTransaction):
            return reference.new_zeros(shape)
        coordinator = transaction.coordinator
        if self._mla_off_postcommit_error_is_fatal(error):
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="committed_pipeline_resource_failure",
                detail=f"stage={stage} {type(error).__name__}: {error}",
            )
        try:
            carrier = transaction.failure_carrier
            if not isinstance(carrier, torch.Tensor):
                raise RuntimeError("preallocated failure carrier is absent")
            required = 1
            for dimension in tuple(int(value) for value in shape):
                if dimension < 0:
                    raise ValueError("failure carrier shape is invalid")
                required *= dimension
            binding = coordinator.proposal.failure_carrier_view
            try:
                live_version = int(carrier._version)
            except RuntimeError:
                live_version = 0
            if (
                required > int(carrier.numel())
                or str(carrier.device) != str(reference.device)
                or carrier.dtype != reference.dtype
                or binding is None
                or binding.nbytes
                != int(carrier.numel()) * int(carrier.element_size())
                or binding.dtype != str(carrier.dtype)
                or binding.version != live_version
                or f":{int(carrier.data_ptr())}:" not in binding.storage_token
            ):
                raise RuntimeError("preallocated failure carrier binding changed")
            return carrier.narrow(0, 0, required).view(shape)
        except BaseException as carrier_error:
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="failure_carrier_unavailable",
                detail=(
                    f"stage={stage} original={type(error).__name__} "
                    f"carrier={type(carrier_error).__name__}: {carrier_error}"
                ),
            )
            raise AssertionError("unreachable")

    def install_mla_off_restore_batch_provider(
        self, *, provider, workspace
    ) -> None:
        """Install one live target-GPU-oracle-certified restore provider.

        A boolean or environment flag is deliberately insufficient.  The
        provider must retain the process-local OracleCertificate that created
        its callbacks, while its TP-common and rank-local digests are bound by
        the forward proposal separately.
        """

        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            DeviceRestoreBatchKernel,
            DeviceRestoreBatchPreflightKernel,
            DeviceRestoreBatchWorkspace,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_batch_kernels import (
            TritonBatchRestoreProvider,
            get_cached_target_gpu_batch_restore_oracle,
        )

        if not isinstance(provider, TritonBatchRestoreProvider):
            raise TypeError("restore batch provider lacks an oracle certificate")
        if not isinstance(workspace, DeviceRestoreBatchWorkspace):
            raise TypeError("restore batch provider has no persistent workspace")
        cached = get_cached_target_gpu_batch_restore_oracle(workspace.device)
        if cached is not provider.certificate:
            raise ValueError("restore provider certificate is not live/cached")
        kernels = provider.kernels
        if not isinstance(kernels, Mapping) or not kernels:
            raise ValueError("restore batch provider has no family kernels")
        normalized = dict(kernels)
        for family, kernel in normalized.items():
            if (
                not isinstance(kernel, DeviceRestoreBatchKernel)
                or kernel.family != family
                or not kernel.production_certified
                or kernel.max_launches_per_call != 1
            ):
                raise ValueError(
                    "restore batch provider must certify one launch per family"
                )
        preflight_kernel = provider.preflight_kernel
        if not (
            isinstance(preflight_kernel, DeviceRestoreBatchPreflightKernel)
            and preflight_kernel.production_certified
            and preflight_kernel.max_launches_per_call <= 2
        ):
            raise ValueError("restore batch aggregate preflight is uncertified")
        if str(workspace.device) != str(
            torch.device("cuda", provider.certificate.device_index)
        ):
            raise ValueError("restore provider/workspace device changed")
        self._redknot_restore_batch_workspace = workspace
        self._redknot_restore_batch_kernels = normalized
        self._redknot_restore_batch_preflight_kernel = preflight_kernel
        self._redknot_restore_batch_provider = provider
        self._redknot_restore_batch_provider_common_token = str(
            provider.common_provider_token
        )
        self._redknot_restore_batch_provider_local_token = str(
            provider.rank_local_provider_token
        )

    def _initialize_mla_off_restore_batch_certificate(self) -> None:
        """Run/cache the exact target-GPU oracle before advertising readiness."""

        try:
            from sglang.srt.layers.attention.redknot import (
                dsv4_shared_latent_batch_kernels as batch_kernels,
            )

            device = torch.device(self.device)
            get_cached = getattr(
                batch_kernels,
                "get_cached_target_gpu_batch_restore_oracle",
                batch_kernels.get_cached_hopper_batch_restore_oracle,
            )
            run_oracle = getattr(
                batch_kernels,
                "run_target_gpu_batch_restore_oracle",
                batch_kernels.run_hopper_batch_restore_oracle,
            )
            certificate = get_cached(device)
            if certificate is None:
                certificate = run_oracle(device)
            provider = batch_kernels.build_triton_batch_restore_provider(
                certificate=certificate,
                device=device,
            )
            self._redknot_restore_batch_provider = provider
            self._redknot_restore_batch_kernels = dict(provider.kernels)
            self._redknot_restore_batch_preflight_kernel = (
                provider.preflight_kernel
            )
            self._redknot_restore_batch_provider_common_token = str(
                provider.common_provider_token
            )
            self._redknot_restore_batch_provider_local_token = str(
                provider.rank_local_provider_token
            )
            self._redknot_restore_batch_provider_error = ""
        except BaseException as error:
            # No uncertified callback is installed.  Startup consensus below
            # disables the overlay uniformly; ordinary dense DSV4 remains
            # available and no cache omission certificate can be produced.
            self._redknot_restore_batch_provider = None
            self._redknot_restore_batch_kernels = None
            self._redknot_restore_batch_preflight_kernel = None
            self._redknot_restore_batch_provider_common_token = None
            self._redknot_restore_batch_provider_local_token = None
            self._redknot_restore_batch_provider_error = (
                f"{type(error).__name__}: {error}"
            )
            logger.exception(
                "RedKnot target-GPU batch restore oracle failed closed"
            )

    def _ensure_mla_off_restore_batch_provider(
        self, *, device, max_requests: int, max_batch_rows: int
    ):
        """Lazily certify target-GPU kernels and size persistent workspace."""

        from sglang.srt.layers.attention.redknot import (
            dsv4_shared_latent_batch_kernels as batch_kernels,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            DeviceRestoreBatchWorkspace,
        )

        device = torch.device(device)
        get_cached = getattr(
            batch_kernels,
            "get_cached_target_gpu_batch_restore_oracle",
            batch_kernels.get_cached_hopper_batch_restore_oracle,
        )
        run_oracle = getattr(
            batch_kernels,
            "run_target_gpu_batch_restore_oracle",
            batch_kernels.run_hopper_batch_restore_oracle,
        )
        certificate = get_cached(device)
        if certificate is None:
            certificate = run_oracle(device)
        provider = getattr(self, "_redknot_restore_batch_provider", None)
        if (
            provider is None
            or provider.certificate is not certificate
        ):
            provider = batch_kernels.build_triton_batch_restore_provider(
                certificate=certificate,
                device=device,
            )
        requirements = dict(
            provider.workspace_requirements(
                max_requests=int(max_requests),
                max_batch_rows=int(max_batch_rows),
            )
        )
        workspace = getattr(self, "_redknot_restore_batch_workspace", None)
        retired = list(
            getattr(self, "_redknot_restore_batch_retired_workspaces", ())
        )
        still_live = []
        for retired_workspace, retired_event in retired:
            if not bool(retired_event.query()):
                still_live.append((retired_workspace, retired_event))
        self._redknot_restore_batch_retired_workspaces = still_live
        needs_workspace = bool(
            not isinstance(workspace, DeviceRestoreBatchWorkspace)
            or str(workspace.device) != str(device)
            or int(workspace.max_jobs) < int(requirements["max_jobs"])
            or int(workspace.max_extra_descriptor_columns)
            < int(requirements["max_extra_descriptor_columns"])
            or int(workspace.max_validation_entries)
            < int(requirements["max_validation_entries"])
        )
        prior_event = getattr(
            self, "_redknot_restore_batch_workspace_event", None
        )
        if (
            isinstance(workspace, DeviceRestoreBatchWorkspace)
            and prior_event is not None
            and not bool(prior_event.query())
        ):
            # Rewriting the pinned host descriptor table can race its prior
            # asynchronous H2D DMA even if a future current-stream wait_event
            # is enqueued.  Retire the whole workspace/event pair and use a
            # fresh persistent slot until the event reports completion.
            self._redknot_restore_batch_retired_workspaces.append(
                (workspace, prior_event)
            )
            needs_workspace = True
        if needs_workspace:
            workspace = DeviceRestoreBatchWorkspace(
                device=device,
                **requirements,
            )
            self._redknot_restore_batch_workspace_event = None
        self.install_mla_off_restore_batch_provider(
            provider=provider,
            workspace=workspace,
        )
        return provider

    def _preflight_or_reuse_zoff_merge_plan(
        self,
        *,
        projection_plan,
        dirty_rows: torch.Tensor,
        dirty_rows_cpu: torch.Tensor,
        local_head_axes: Tuple[int, ...],
        wo_a_weight: torch.Tensor,
        owned_heads: int,
        groups: int,
        head_dim: int,
        o_lora_rank: int,
    ):
        """Return a fully certified merge plan, avoiding repeated row walks.

        A cache hit is allowed only for the exact immutable projection-plan
        object/digest and the exact live tensor storage/version identities.
        The cached plan then performs its ordinary O(views) live validation.
        A miss runs the original complete preflight; there is no fallback from
        a failed hit to silently recertifying changed state.
        """

        from sglang.srt.layers.attention.redknot.dsv4_fused_z_merge import (
            preflight_persistent_headsplit_woa_merge,
        )

        local_axes = tuple(int(axis) for axis in local_head_axes)
        key = (
            "redknot_zoff_merge_plan_cache_v1",
            id(projection_plan),
            str(getattr(projection_plan, "digest", "")),
            _mla_off_control_tensor_identity(dirty_rows),
            _mla_off_control_tensor_identity(dirty_rows_cpu),
            local_axes,
            _mla_off_control_tensor_identity(wo_a_weight),
            int(owned_heads),
            int(groups),
            int(head_dim),
            int(o_lora_rank),
            os.environ.get("REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH", "0"),
        )
        cache = self._redknot_zoff_merge_plan_cache
        cached = cache.get(key)
        if cached is not None:
            validate_live = getattr(cached, "validate_live", None)
            if not callable(validate_live):
                raise TypeError("cached z_off merge plan has no live validator")
            validate_live(
                committed_plan_identity=(id(cached), str(cached.digest))
            )
            self._count("zoff_merge_plan_cache_hits")
            return cached

        plan = preflight_persistent_headsplit_woa_merge(
            projection_plan=projection_plan,
            dirty_rows=dirty_rows,
            dirty_rows_cpu=dirty_rows_cpu,
            local_head_axes=local_axes,
            wo_a_weight=wo_a_weight,
            owned_heads=int(owned_heads),
            groups=int(groups),
            head_dim=int(head_dim),
            o_lora_rank=int(o_lora_rank),
        )
        if len(cache) >= int(self._redknot_zoff_merge_plan_cache_limit):
            cache.pop(next(iter(cache)))
        cache[key] = plan
        self._count("zoff_merge_plan_cache_misses")
        return plan

    def _begin_mla_off_zoff_forward_transaction(
        self,
        *,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layers: Tuple[object, ...],
        q_row_count: int,
        device,
        projection_dtype: torch.dtype,
        active_plans: Tuple[Mapping[str, object], ...],
    ):
        """Commit all z_off/head-split omissions with one forward TP vote.

        ``zoff_only`` deliberately keeps WKV, compressor and Indexer work
        online.  It therefore needs no shared-cache restore batch, but its
        packed-Q tickets, persistent z_off views and fused merge plans can be
        reserved before layer 3 exactly like the full composite profile.  A
        single prepare certificate replaces the legacy per-layer geometry,
        sparse-Q install and merge-consumer votes; one final rendezvous after
        layer 39 remains fail-closed.
        """

        collective_adapter = _RedKnotCompositeCollectiveAdapter(self, device)
        local_error = None
        local_mode = "invalid"
        setup_stage = "initialize_zoff_forward"
        contexts = {}
        q_arena = None
        q_reservations = {}
        prepared_layers = {}
        merge_plans = {}
        resources = None
        coordinator = None
        transaction = None
        failure_carrier = None
        forward_transfer_audit_state = None
        expected_layers = tuple(self._redknot_mla_off_rank_local_layer_ids)
        try:
            layer_map = {int(layer.layer_id): layer for layer in tuple(layers)}
            if tuple(sorted(layer_map)) != expected_layers:
                raise ValueError(
                    "model did not expose exactly the 37 reusable layers"
                )
            forward_transfer_audit_state = (
                self._mla_off_begin_composite_transfer_audit(
                    forward_batch=forward_batch,
                    layer_id=int(expected_layers[0]),
                )
            )
            from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                ForwardCompositeCommitCoordinator,
                build_layer_composite_reservation,
                merge_layer_composite_proposals,
                prepare_composite_restore_context,
            )

            for layer_id in expected_layers:
                setup_stage = f"prepare_zoff_context_layer_{int(layer_id)}"
                module = layer_map[layer_id]
                with _redknot_mla_timed(
                    "mla_prepare_context", layer_id=layer_id
                ):
                    context = prepare_composite_restore_context(
                        self,
                        layer_id=layer_id,
                        positions=positions,
                        forward_batch=forward_batch,
                        q_row_count=int(q_row_count),
                        n_local_heads=int(module.n_local_heads),
                        n_local_groups=int(module.n_local_groups),
                        head_dim=int(module.head_dim),
                        o_lora_rank=int(module.o_lora_rank),
                        device=torch.device(device),
                        projection_dtype=projection_dtype,
                    )
                if context is None:
                    raise ValueError(
                        "zoff forward reservation produced an empty context"
                    )
                if str(getattr(context, "diagnostic_ablation", "")) != "zoff_only":
                    raise ValueError("zoff forward received another diagnostic profile")
                if tuple(getattr(context, "shared_restore_states", ()) or ()):
                    raise ValueError("zoff forward unexpectedly retained shared restore")
                contexts[layer_id] = context
            for context in contexts.values():
                self._mla_off_bind_composite_transfer_audit(
                    forward_transfer_audit_state, context
                )

            full_local = tuple(
                bool(getattr(context, "is_full_local", False))
                for context in contexts.values()
            )
            if any(full_local):
                if not all(full_local):
                    raise ValueError("middle layers disagree on query-suffix mode")
                resources = getattr(
                    forward_batch, "_redknot_composite_forward_resources", None
                )
                if resources is None:
                    raise RuntimeError("full-local zoff forward lost its resource lease")
                transaction = _RedKnotForwardCompositeTransaction(
                    forward_id=str(resources.forward_id),
                    contexts=dict(contexts),
                    q_arena=None,
                    q_reservations={},
                    prepared_layers={},
                    merge_plans={},
                    coordinator=None,
                    collective_adapter=collective_adapter,
                    omission_profile="full_local",
                    restore_batch_receipt=None,
                    failure_carrier=None,
                )
                forward_batch._redknot_mla_off_forward_transaction = transaction
                for context in contexts.values():
                    context._redknot_forward_composite_transaction = transaction
                local_mode = "full_local"
            else:
                resources = getattr(
                    forward_batch, "_redknot_composite_forward_resources", None
                )
                if resources is None:
                    raise RuntimeError("zoff forward lost its resource lease")
                from sglang.srt.layers.attention.redknot.dsv4_sparse_q import (
                    build_sparse_q_plan,
                )
                from sglang.srt.layers.attention.redknot.dsv4_sparse_q_runtime import (
                    SequentialPackedQArena,
                )
                sparse_plans = []
                for layer_id in expected_layers:
                    module = layer_map[layer_id]
                    context = contexts[layer_id]
                    sparse_plans.append(
                        build_sparse_q_plan(
                            int(self._redknot_tp_rank),
                            int(self._redknot_tp_size),
                            int(module.n_heads),
                            tuple(int(axis) for axis in context.local_head_axes),
                            int(q_row_count),
                            tuple(
                                int(row)
                                for row in context.online_local_row_indices_cpu.tolist()
                            ),
                            layer_id=layer_id,
                            head_dim=int(module.head_dim),
                        )
                    )
                q_arena = SequentialPackedQArena.allocate(
                    tuple(sparse_plans),
                    device=device,
                    dtype=projection_dtype,
                    arena_token=(
                        f"sequential-q:{resources.forward_id}:"
                        f"tp:{int(self._redknot_tp_rank)}:zoff-arena:0"
                    ),
                )
                persistent_z_arena_token = (
                    f"persistent-z:{resources.forward_id}:"
                    f"tp:{int(self._redknot_tp_rank)}"
                )
                carrier_width = max(
                    max(
                        int(layer_map[layer_id].n_local_heads)
                        * int(layer_map[layer_id].head_dim),
                        int(layer_map[layer_id].n_local_heads)
                        * int(layer_map[layer_id].attn_mqa.v_head_dim),
                        int(layer_map[layer_id].n_local_groups)
                        * int(layer_map[layer_id].o_lora_rank),
                    )
                    for layer_id in expected_layers
                )
                # Values are never consumed as model data: this arena only
                # preserves tensor shapes and collective ordering after a
                # post-commit peer failure.  Initializing ~100 MiB per 64K
                # request needlessly adds an exposed memset to TTFT.
                failure_carrier = torch.empty(
                    (int(q_row_count) * int(carrier_width),),
                    dtype=projection_dtype,
                    device=torch.device(device),
                )
                layer_proposals = []
                for ordinal, layer_id in enumerate(expected_layers):
                    module = layer_map[layer_id]
                    context = contexts[layer_id]
                    reservation = q_arena.reservation_for(layer_id)
                    q_reservations[layer_id] = reservation
                    merge_plan = self._preflight_or_reuse_zoff_merge_plan(
                        projection_plan=(
                            context.validate_persistent_projection_commit()
                        ),
                        dirty_rows=context.online_local_row_indices,
                        dirty_rows_cpu=context.online_local_row_indices_cpu,
                        local_head_axes=context.local_head_axes,
                        wo_a_weight=module.wo_a.weight,
                        owned_heads=int(module.n_local_heads),
                        groups=int(module.n_local_groups),
                        head_dim=int(module.head_dim),
                        o_lora_rank=int(module.o_lora_rank),
                    )
                    merge_plans[layer_id] = merge_plan
                    builder_epoch_token = "sha256:" + hashlib.sha256(
                        (
                            f"zoff-forward:{resources.forward_id}:"
                            f"layer:{int(layer_id)}:{merge_plan.kernel_token}"
                        ).encode("utf-8")
                    ).hexdigest()
                    prepared = _RedKnotPreparedZOffLayer(
                        cache_preflights=(),
                        builder_epoch_token=builder_epoch_token,
                    )
                    prepared_layers[layer_id] = prepared
                    layer_proposals.append(
                        build_layer_composite_reservation(
                            context,
                            cache_domains=(),
                            sparse_q_reservation=reservation,
                            forward_ordinal=ordinal,
                            builder_epoch_token=builder_epoch_token,
                            generation_id=resources.forward_id,
                            model_hash=self._redknot_mla_off_model_hash,
                            policy_hash=self._redknot_mla_off_policy_hash,
                            persistent_arena_token=persistent_z_arena_token,
                            fused_merge_kernel_token=merge_plan.kernel_token,
                            omission_profile="zoff_only",
                            restore_provider_token="zoff-only:no-shared-provider",
                            restore_provider_local_token=(
                                f"zoff-only:no-shared-provider:"
                                f"tp:{int(self._redknot_tp_rank)}"
                            ),
                            restore_batch_common_digest=(
                                "zoff-only:no-device-restore-batch"
                            ),
                            restore_batch_local_digest=(
                                f"zoff-only:no-device-restore-batch:"
                                f"tp:{int(self._redknot_tp_rank)}"
                            ),
                            failure_carrier=failure_carrier,
                        )
                    )
                forward_ordinal = int(
                    hashlib.sha256(resources.forward_id.encode("utf-8")).hexdigest()[:8],
                    16,
                )
                proposal = merge_layer_composite_proposals(
                    resources,
                    tuple(layer_proposals),
                    forward_ordinal=forward_ordinal,
                    omission_profile="zoff_only",
                    restore_batch_common_digest=(
                        "zoff-only:no-device-restore-batch"
                    ),
                    restore_batch_local_digest=(
                        f"zoff-only:no-device-restore-batch:"
                        f"tp:{int(self._redknot_tp_rank)}"
                    ),
                )
                builder_epoch_token = "forward-zoff-builders:" + hashlib.sha256(
                    repr(
                        tuple(
                            prepared_layers[layer].builder_epoch_token
                            for layer in expected_layers
                        )
                    ).encode("utf-8")
                ).hexdigest()
                coordinator = ForwardCompositeCommitCoordinator(
                    resources=resources,
                    proposal=proposal,
                    builder_epoch_token=builder_epoch_token,
                )
                transaction = _RedKnotForwardCompositeTransaction(
                    forward_id=resources.forward_id,
                    contexts=dict(contexts),
                    q_arena=q_arena,
                    q_reservations=dict(q_reservations),
                    prepared_layers=dict(prepared_layers),
                    merge_plans=dict(merge_plans),
                    coordinator=coordinator,
                    collective_adapter=collective_adapter,
                    omission_profile="zoff_only",
                    restore_batch_receipt=None,
                    failure_carrier=failure_carrier,
                )
                forward_batch._redknot_mla_off_forward_transaction = transaction
                for layer_id, context in contexts.items():
                    context._redknot_forward_composite_transaction = transaction
                    context.sequential_q_reservation = q_reservations[layer_id]
                local_mode = "coordinator"
        except BaseException as error:
            local_error = error
            local_mode = "invalid"

        forward_batch._redknot_mla_off_forward_transaction_attempted = True
        setup_modes = ("full_local", "coordinator", "invalid")
        try:
            setup_signal = [0] * len(setup_modes)
            setup_signal[setup_modes.index(local_mode)] = 1
            setup_signal.append(int(local_error is None))
            reduced_setup = torch.tensor(
                setup_signal, dtype=torch.int64, device=device
            )
            if int(self._redknot_tp_size) > 1:
                with _redknot_mla_timed("mla_prepare_setup_vote"):
                    reduced_setup = self._mla_off_control_all_reduce(reduced_setup)
            totals = tuple(int(value) for value in reduced_setup.tolist())
        except BaseException as vote_error:
            if isinstance(transaction, _RedKnotForwardCompositeTransaction):
                self._fail_stop_mla_off_transaction(
                    transaction,
                    reason_code="zoff_forward_setup_vote_indeterminate",
                    detail=f"{type(vote_error).__name__}: {vote_error}",
                )
            self._fail_stop_mla_off_without_transaction(
                forward_id=str(
                    getattr(resources, "forward_id", "redknot-zoff-forward-setup")
                ),
                reason_code="zoff_forward_setup_vote_indeterminate",
                detail=f"{type(vote_error).__name__}: {vote_error}",
            )
        world = int(self._redknot_tp_size)
        agreed = tuple(
            setup_modes[index]
            for index, count in enumerate(totals[: len(setup_modes)])
            if count == world
        )
        if (
            totals[-1] != world
            or len(agreed) != 1
            or sum(totals[: len(setup_modes)]) != world
            or agreed[0] == "invalid"
        ):
            forward_batch._redknot_mla_off_disabled = True
            self._cleanup_uncommitted_mla_off_forward_collectively(
                forward_batch=forward_batch,
                device=device,
                stage="zoff_forward_setup_reject",
            )
            self._mla_off_log_failure(
                "zoff_forward_setup_failed",
                f"stage={setup_stage}; {type(local_error).__name__}: {local_error}"
                if local_error is not None
                else f"TP setup modes disagreed: {totals!r}",
            )
            return None
        if agreed[0] == "full_local":
            if not isinstance(transaction, _RedKnotForwardCompositeTransaction):
                raise AssertionError("full-local zoff setup lost its transaction")
            self._mla_off_log_composite_forward_manifest(
                transaction.context_for(expected_layers[0])
            )
            self._mla_off_log_forward_transaction_full_local_statuses(
                transaction=transaction,
                active_plans=active_plans,
                expected_layers=expected_layers,
            )
            return transaction

        if not (
            coordinator is not None
            and isinstance(transaction, _RedKnotForwardCompositeTransaction)
        ):
            raise AssertionError("zoff coordinator setup lost its transaction")
        try:
            with _redknot_mla_timed("mla_prepare_commit_vote"):
                outcome = coordinator.commit(collective_adapter, ready=True)
        except BaseException as commit_error:
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="zoff_forward_prepare_exception",
                detail=f"{type(commit_error).__name__}: {commit_error}",
            )
        if not outcome.committed:
            for context in contexts.values():
                context.composite_dense_fallback = True
            forward_batch._redknot_mla_off_disabled = True
            self._cleanup_uncommitted_mla_off_forward_collectively(
                forward_batch=forward_batch,
                device=device,
                stage="zoff_forward_commit_reject",
            )
            return None
        try:
            for context in contexts.values():
                coordinator.bind_context(context)
            coordinator.register_prepared_full_layers(
                contexts=contexts,
                q_arena=q_arena,
                q_reservations=q_reservations,
                prepared_layers=prepared_layers,
            )
        except BaseException as bind_error:
            coordinator.session.fail_closed_after_commit(
                collective_adapter,
                reason_code="zoff_forward_context_bind_failed",
                detail=f"{type(bind_error).__name__}: {bind_error}",
            )
        self._mla_off_log_composite_forward_manifest(
            transaction.context_for(expected_layers[0])
        )
        return transaction

    def begin_mla_off_forward_transaction(
        self,
        *,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layers: Tuple[object, ...],
        q_row_count: int,
        device,
        projection_dtype: torch.dtype,
    ):
        """Reserve/commit layers 3..39 before layer 3 starts executing.

        The Python loop below is control-plane preflight only.  Clean cache
        mutation is one ``restore_clean_batched`` call whose implementation is
        bounded to one certified GPU launch per semantic family.
        """

        # Establish the ForwardBatch generation before inspecting plans or
        # publishing any intent decision.  In particular, an invalid/disagreed
        # intent below latches ``disabled`` + ``transaction_attempted`` for this
        # generation; a later per-layer prepare must observe those markers
        # instead of clearing them and re-entering the legacy protocol.
        self._mla_off_prepare_forward_generation(forward_batch)
        existing = getattr(
            forward_batch, "_redknot_mla_off_forward_transaction", None
        )
        if existing is not None:
            return existing
        if not self._redknot_mla_off_enabled or not self._redknot_shared_latent_enabled:
            return None
        if forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
            return None
        from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
            resolve_mla_off_diagnostic_ablation,
        )

        raw_plans = getattr(forward_batch, "redknot_reuse_plan", None)
        active = ()
        diagnostic = None
        intent_error = None
        intent_mode = "bypass"
        try:
            if raw_plans:
                if len(raw_plans) != int(forward_batch.batch_size):
                    raise ValueError("reuse plans do not span the forward batch")
                active = tuple(
                    item
                    for item in raw_plans
                    if isinstance(item, Mapping)
                    and item.get("mode") == "restore"
                    and bool(item.get("reuse_mla_off", False))
                )
                has_snapshot = any(
                    isinstance(item, Mapping)
                    and item.get("mode") == "snapshot"
                    for item in raw_plans
                )
                if active and has_snapshot:
                    raise ValueError(
                        "one continuous batch cannot mix restore and snapshot plans"
                    )
                if active:
                    diagnostics = tuple(
                        resolve_mla_off_diagnostic_ablation(item)
                        for item in active
                    )
                    if len(set(diagnostics)) != 1:
                        raise ValueError(
                            "restore requests disagree on diagnostic ablation"
                        )
                    diagnostic = diagnostics[0]
                    if diagnostic not in (
                        "full",
                        "shared_only",
                        "zoff_only",
                    ):
                        raise ValueError(
                            "restore diagnostic ablation is unsupported"
                        )
                    intent_mode = (
                        "zoff_only"
                        if diagnostic == "zoff_only"
                        else "forward"
                    )
        except BaseException as error:
            intent_error = error
            intent_mode = "invalid"

        # Plan metadata is rank-local input.  No rank may return before this
        # fixed execution-class vote while another starts the all-layer
        # reservation.  Snapshot-only and no-restore batches unanimously keep
        # their legacy/dense path; mixed restore+snapshot is explicitly
        # rejected instead of silently dropping snapshot publication.
        intent_modes = ("bypass", "zoff_only", "forward", "invalid")
        try:
            intent_signal = [0] * len(intent_modes)
            intent_signal[intent_modes.index(intent_mode)] = 1
            intent_signal.append(int(intent_error is None))
            reduced_intent = torch.tensor(
                intent_signal, dtype=torch.int64, device=device
            )
            if int(self._redknot_tp_size) > 1:
                with _redknot_mla_timed("mla_prepare_intent_vote"):
                    reduced_intent = self._mla_off_control_all_reduce(
                        reduced_intent
                    )
            intent_totals = tuple(
                int(value) for value in reduced_intent.tolist()
            )
        except BaseException as vote_error:
            self._fail_stop_mla_off_without_transaction(
                forward_id="redknot-forward-intent",
                reason_code="forward_intent_vote_indeterminate",
                detail=f"{type(vote_error).__name__}: {vote_error}",
            )
        intent_world = int(self._redknot_tp_size)
        agreed_intent = tuple(
            intent_modes[index]
            for index, count in enumerate(intent_totals[:-1])
            if count == intent_world
        )
        if (
            intent_totals[-1] != intent_world
            or len(agreed_intent) != 1
            or sum(intent_totals[:-1]) != intent_world
            or agreed_intent[0] == "invalid"
        ):
            forward_batch._redknot_mla_off_disabled = True
            forward_batch._redknot_mla_off_forward_transaction_attempted = True
            self._mla_off_log_failure(
                "forward_intent_failed",
                str(intent_error)
                if intent_error is not None
                else f"TP execution classes disagreed: {intent_totals!r}",
            )
            return None
        if agreed_intent[0] == "bypass":
            return None
        if diagnostic == "zoff_only":
            forward_batch._redknot_mla_off_forward_transaction_attempted = True
            return self._begin_mla_off_zoff_forward_transaction(
                positions=positions,
                forward_batch=forward_batch,
                layers=layers,
                q_row_count=int(q_row_count),
                device=device,
                projection_dtype=projection_dtype,
                active_plans=tuple(active),
            )
        if diagnostic not in ("full", "shared_only"):
            raise AssertionError("forward intent accepted an invalid profile")
        forward_batch._redknot_mla_off_forward_transaction_attempted = True

        local_error = None
        # Diagnostic-only stage marker for rank-local setup failures.  Keep it
        # as a plain Python string so reporting cannot allocate CUDA memory or
        # alter the transaction/collective contract.
        setup_stage = "initialize"
        local_mode = "invalid"
        contexts = {}
        q_arena = None
        failure_carrier = None
        q_reservations = {}
        prepared_layers = {}
        merge_plans = {}
        proposal = None
        validated_batch = None
        restore_pipeline_plan = None
        slot_bounds_batch = None
        slot_bounds_certificate = None
        restore_provider = None
        resources = None
        coordinator = None
        transaction = None
        batch_receipt_bindings = []
        forward_transfer_audit_state = None
        collective_adapter = _RedKnotCompositeCollectiveAdapter(self, device)
        try:
            setup_stage = "validate_layer_geometry"
            forward_batch._redknot_mla_off_forward_transaction_attempted = True
            layer_map = {int(layer.layer_id): layer for layer in tuple(layers)}
            expected_layers = tuple(self._redknot_mla_off_rank_local_layer_ids)
            if tuple(sorted(layer_map)) != expected_layers:
                raise ValueError(
                    "model did not expose exactly the 37 reusable layers"
                )
            forward_transfer_audit_state = (
                self._mla_off_begin_composite_transfer_audit(
                    forward_batch=forward_batch,
                    layer_id=int(expected_layers[0]),
                )
            )
            setup_stage = "ensure_restore_provider"
            with _redknot_mla_timed("mla_prepare_provider"):
                restore_provider = self._ensure_mla_off_restore_batch_provider(
                    device=device,
                    max_requests=int(forward_batch.batch_size),
                    max_batch_rows=int(q_row_count),
                )
            from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                ForwardCompositeCommitCoordinator,
                build_layer_composite_reservation,
                merge_layer_composite_proposals,
                prepare_composite_restore_context,
            )

            for layer_id in expected_layers:
                setup_stage = f"prepare_context_layer_{int(layer_id)}"
                module = layer_map[layer_id]
                with _redknot_mla_timed(
                    "mla_prepare_context", layer_id=layer_id
                ):
                    context = prepare_composite_restore_context(
                        self,
                        layer_id=layer_id,
                        positions=positions,
                        forward_batch=forward_batch,
                        q_row_count=int(q_row_count),
                        n_local_heads=int(module.n_local_heads),
                        n_local_groups=int(module.n_local_groups),
                        head_dim=int(module.head_dim),
                        o_lora_rank=int(module.o_lora_rank),
                        device=torch.device(device),
                        projection_dtype=projection_dtype,
                    )
                if context is None:
                    preflight_fallbacks = tuple(
                        getattr(
                            forward_batch,
                            "_redknot_composite_preflight_fallbacks",
                            (),
                        )
                        or ()
                    )
                    raise ValueError(
                        "forward reservation produced an empty context; "
                        f"preflight_fallbacks={preflight_fallbacks!r}"
                    )
                contexts[layer_id] = context
            for context in contexts.values():
                setup_stage = "bind_transfer_audit"
                self._mla_off_bind_composite_transfer_audit(
                    forward_transfer_audit_state,
                    context,
                )
            full_local = tuple(
                bool(getattr(context, "is_full_local", False))
                for context in contexts.values()
            )
            if any(full_local):
                setup_stage = "construct_full_local_transaction"
                if not all(full_local):
                    raise ValueError("middle layers disagree on query-suffix mode")
                resources = getattr(
                    forward_batch, "_redknot_composite_forward_resources", None
                )
                if resources is None:
                    raise RuntimeError("full-local forward lost its resource lease")
                transaction = _RedKnotForwardCompositeTransaction(
                    forward_id=str(resources.forward_id),
                    contexts=dict(contexts),
                    q_arena=None,
                    q_reservations={},
                    prepared_layers={},
                    merge_plans={},
                    coordinator=None,
                    collective_adapter=collective_adapter,
                    omission_profile="full_local",
                    restore_batch_receipt=None,
                    failure_carrier=None,
                )
                forward_batch._redknot_mla_off_forward_transaction = transaction
                for context in contexts.values():
                    context._redknot_forward_composite_transaction = transaction
                local_mode = "full_local"
            else:
                omission_profile = {
                    "full": "full",
                    "shared_only": "shared_only",
                }[diagnostic]
                resources = getattr(
                    forward_batch, "_redknot_composite_forward_resources", None
                )
                if resources is None:
                    raise RuntimeError("forward transaction lost its resource lease")
                if omission_profile == "full":
                    setup_stage = "build_sparse_q_plans"
                    from sglang.srt.layers.attention.redknot.dsv4_sparse_q import (
                        build_sparse_q_plan,
                    )
                    from sglang.srt.layers.attention.redknot.dsv4_sparse_q_runtime import (
                        SequentialPackedQArena,
                    )

                    sparse_plans = []
                    for layer_id in expected_layers:
                        module = layer_map[layer_id]
                        context = contexts[layer_id]
                        with _redknot_mla_timed(
                            "mla_prepare_sparse_q_plan", layer_id=layer_id
                        ):
                            sparse_plans.append(
                                build_sparse_q_plan(
                                    int(self._redknot_tp_rank),
                                    int(self._redknot_tp_size),
                                    int(module.n_heads),
                                    tuple(
                                        int(axis)
                                        for axis in context.local_head_axes
                                    ),
                                    int(q_row_count),
                                    tuple(
                                        int(row)
                                        for row in context.online_local_row_indices_cpu.tolist()
                                    ),
                                    layer_id=layer_id,
                                    head_dim=int(module.head_dim),
                                )
                            )
                    with _redknot_mla_timed("mla_prepare_q_arena"):
                        setup_stage = "allocate_sparse_q_arena"
                        q_arena = SequentialPackedQArena.allocate(
                            tuple(sparse_plans),
                            device=device,
                            dtype=projection_dtype,
                            arena_token=(
                                f"sequential-q:{resources.forward_id}:"
                                f"tp:{int(self._redknot_tp_rank)}:arena:0"
                            ),
                        )
                    persistent_z_arena_token = (
                        f"persistent-z:{resources.forward_id}:"
                        f"tp:{int(self._redknot_tp_rank)}"
                    )
                else:
                    # Attribution-B computes complete Q/attention/wo_a.  It
                    # must not allocate a packed-Q arena or claim a z_off
                    # consumer merely to certify shared-cache restoration.
                    persistent_z_arena_token = "shared-only:no-persistent-zoff"
                # Reserve one rank-local, zero-filled shape carrier before
                # the omission certificate.  It is sequentially reused only
                # after a carrier-safe local logic error; no 37-layer or
                # 64-head padded activation is materialized.
                carrier_width = max(
                    max(
                        int(layer_map[layer_id].n_local_heads)
                        * int(layer_map[layer_id].head_dim),
                        int(layer_map[layer_id].n_local_heads)
                        * int(layer_map[layer_id].attn_mqa.v_head_dim),
                        int(layer_map[layer_id].n_local_groups)
                        * int(layer_map[layer_id].o_lora_rank),
                    )
                    for layer_id in expected_layers
                )
                with _redknot_mla_timed("mla_prepare_failure_carrier"):
                    setup_stage = "allocate_failure_carrier"
                    failure_carrier = torch.zeros(
                        (int(q_row_count) * int(carrier_width),),
                        dtype=projection_dtype,
                        device=torch.device(device),
                    )
                layer_proposals = []
                from sglang.srt.layers.attention.redknot.dsv4_shared_latent_sglang import (
                    RestoreSlotBoundsBatch,
                )

                slot_bounds_batch = RestoreSlotBoundsBatch()
                for ordinal, layer_id in enumerate(expected_layers):
                    setup_stage = f"prepare_shared_layer_{int(layer_id)}"
                    module = layer_map[layer_id]
                    context = contexts[layer_id]
                    reservation = (
                        q_arena.reservation_for(layer_id)
                        if q_arena is not None
                        else None
                    )
                    if reservation is not None:
                        q_reservations[layer_id] = reservation
                    with _redknot_mla_timed(
                        "mla_prepare_shared_layer", layer_id=layer_id
                    ):
                        prepared = self._prepare_mla_off_shared_layer(
                            mla_off_context=context,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=layer_id,
                            compress_ratio=int(module.compress_ratio),
                            freqs_cis=module.freqs_cis,
                            slot_bounds_batch=slot_bounds_batch,
                        )
                    prepared_layers[layer_id] = prepared
                    if omission_profile == "full":
                        from sglang.srt.layers.attention.redknot.dsv4_fused_z_merge import (
                            preflight_persistent_headsplit_woa_merge,
                        )

                        with _redknot_mla_timed(
                            "mla_prepare_merge_plan", layer_id=layer_id
                        ):
                            setup_stage = f"prepare_merge_plan_layer_{int(layer_id)}"
                            merge_plans[layer_id] = (
                                preflight_persistent_headsplit_woa_merge(
                                    projection_plan=context.validate_persistent_projection_commit(),
                                    dirty_rows=context.online_local_row_indices,
                                    dirty_rows_cpu=context.online_local_row_indices_cpu,
                                    local_head_axes=context.local_head_axes,
                                    wo_a_weight=module.wo_a.weight,
                                    owned_heads=int(module.n_local_heads),
                                    groups=int(module.n_local_groups),
                                    head_dim=int(module.head_dim),
                                    o_lora_rank=int(module.o_lora_rank),
                                )
                            )
                    with _redknot_mla_timed(
                        "mla_prepare_layer_reservation", layer_id=layer_id
                    ):
                        setup_stage = f"prepare_reservation_layer_{int(layer_id)}"
                        layer_proposals.append(
                            build_layer_composite_reservation(
                                context,
                                cache_domains=prepared.cache_preflights,
                                sparse_q_reservation=reservation,
                                forward_ordinal=ordinal,
                                builder_epoch_token=prepared.builder_epoch_token,
                                generation_id=resources.forward_id,
                                model_hash=self._redknot_mla_off_model_hash,
                                policy_hash=self._redknot_mla_off_policy_hash,
                                persistent_arena_token=persistent_z_arena_token,
                                fused_merge_kernel_token=(
                                    merge_plans[layer_id].kernel_token
                                    if omission_profile == "full"
                                    else "shared-only:no-fused-z-merge"
                                ),
                                omission_profile=omission_profile,
                                restore_provider_token=str(
                                    restore_provider.common_provider_token
                                ),
                                restore_provider_local_token=str(
                                    restore_provider.rank_local_provider_token
                                ),
                                failure_carrier=failure_carrier,
                            )
                        )
                # Layer target construction above is mutation-free.  Fence all
                # 147 DeepSeek-V4 full-vector predicates once, before aggregate
                # descriptor preflight, TP prepare commit, or cache mutation.
                with _redknot_mla_timed("mla_prepare_slot_bounds"):
                    setup_stage = "finalize_slot_bounds"
                    slot_bounds_certificate = slot_bounds_batch.finalize(
                        expected_layer_ids=expected_layers
                    )
                expected_slot_vectors = sum(
                    len(prepared.restore_adapter.target_slots)
                    for prepared in prepared_layers.values()
                )
                if (
                    int(slot_bounds_certificate.vector_count)
                    != int(expected_slot_vectors)
                    or int(slot_bounds_certificate.predicate_count)
                    != 2 * int(expected_slot_vectors)
                ):
                    raise RuntimeError(
                        "forward-wide slot bounds certificate lost a target vector"
                    )
                forward_ordinal = int(
                    hashlib.sha256(resources.forward_id.encode("utf-8")).hexdigest()[:8],
                    16,
                )
                from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
                    DeviceRestoreBatchInput,
                    compile_device_restore_pipeline,
                    preflight_device_restore_batch,
                )

                batch_inputs = []
                setup_stage = "build_device_restore_inputs"
                for layer_id in expected_layers:
                    prepared = prepared_layers[layer_id]
                    for request_restore in prepared.validated_restores:
                        request_index = int(request_restore.request_index)
                        store = request_restore.store
                        validated = request_restore.validated
                        batch_inputs.append(
                            DeviceRestoreBatchInput(
                                store=store,
                                validated=validated,
                                operation_metadata=prepared.restore_adapter.batch_metadata,
                                request_index=int(request_index),
                                layer_id=int(layer_id),
                            )
                        )
                        batch_receipt_bindings.append(
                            (
                                layer_id,
                                request_index,
                                request_restore.receipt_adapter,
                                validated.prepared.schedule,
                            )
                        )
                workspace = getattr(
                    self, "_redknot_restore_batch_workspace", None
                )
                kernels = getattr(self, "_redknot_restore_batch_kernels", None)
                preflight_kernel = getattr(
                    self, "_redknot_restore_batch_preflight_kernel", None
                )
                if workspace is None or not kernels or preflight_kernel is None:
                    raise RuntimeError(
                        "production pointer-table restore provider is not installed"
                    )
                with _redknot_mla_timed("mla_prepare_device_batch_preflight"):
                    setup_stage = "device_batch_preflight"
                    validated_batch = preflight_device_restore_batch(
                        inputs=tuple(batch_inputs),
                        forward_id=resources.forward_id,
                        workspace=workspace,
                        kernels=kernels,
                        preflight_kernel=preflight_kernel,
                        require_production_certified=True,
                        non_blocking=True,
                    )
                if self._redknot_restore_pipeline_group_layers:
                    setup_stage = "compile_restore_pipeline"
                    restore_pipeline_plan = compile_device_restore_pipeline(
                        validated_batch.plan,
                        layers_per_group=int(
                            self._redknot_restore_pipeline_group_layers
                        ),
                    )
                restore_batch_common_digest = "sha256:" + hashlib.sha256(
                    json.dumps(
                        {
                            "schema": "redknot-restore-batch-common-v1",
                            "forward_id": str(validated_batch.plan.forward_id),
                            "input_bindings": tuple(
                                (
                                    int(item.request_index),
                                    int(item.layer_id),
                                )
                                for item in validated_batch.inputs
                            ),
                            "jobs": tuple(
                                (
                                    int(job.input_ordinal),
                                    str(job.domain),
                                    str(job.family),
                                    int(job.layer_id),
                                    int(job.count),
                                    int(job.record_bytes),
                                    str(job.position_semantics),
                                )
                                for job in validated_batch.plan.jobs
                            ),
                            "families": tuple(
                                (
                                    str(family),
                                    int(span.end - span.begin),
                                )
                                for family, span in validated_batch.plan.family_spans.items()
                            ),
                            "preflight_launches": int(
                                validated_batch.preflight_launch_count
                            ),
                            # The local pipeline digest includes this rank's
                            # pointer-bound batch digest.  Only the explicitly
                            # rank-neutral topology digest belongs in the TP
                            # common proposal; the local batch plan remains
                            # bound by restore_batch_local_digest below.
                            "restore_pipeline_common_digest": str(
                                restore_pipeline_plan.common_digest
                                if restore_pipeline_plan is not None
                                else ""
                            ),
                            "restore_pipeline_group_layers": int(
                                self._redknot_restore_pipeline_group_layers
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                with _redknot_mla_timed("mla_prepare_merge_proposals"):
                    setup_stage = "merge_layer_proposals"
                    proposal = merge_layer_composite_proposals(
                        resources,
                        tuple(layer_proposals),
                        forward_ordinal=forward_ordinal,
                        omission_profile=omission_profile,
                        restore_batch_common_digest=restore_batch_common_digest,
                        restore_batch_local_digest=str(
                            validated_batch.plan.digest
                        ),
                    )
                builder_epoch_token = "forward-builders:" + hashlib.sha256(
                    repr(
                        tuple(
                            prepared_layers[layer].builder_epoch_token
                            for layer in resources.expected_layer_ids
                        )
                    ).encode("utf-8")
                ).hexdigest()
                coordinator = ForwardCompositeCommitCoordinator(
                    resources=resources,
                    proposal=proposal,
                    builder_epoch_token=builder_epoch_token,
                )
                transaction = _RedKnotForwardCompositeTransaction(
                    forward_id=resources.forward_id,
                    contexts=dict(contexts),
                    q_arena=q_arena,
                    q_reservations=dict(q_reservations),
                    prepared_layers=dict(prepared_layers),
                    merge_plans=dict(merge_plans),
                    coordinator=coordinator,
                    collective_adapter=collective_adapter,
                    omission_profile=str(proposal.omission_profile),
                    restore_batch_receipt=None,
                    restore_pipeline_plan=restore_pipeline_plan,
                    failure_carrier=failure_carrier,
                    validated_restore_batch=validated_batch,
                    slot_bounds_certificate=slot_bounds_certificate,
                )
                forward_batch._redknot_mla_off_forward_transaction = transaction
                for layer_id, context in contexts.items():
                    setup_stage = f"bind_transaction_layer_{int(layer_id)}"
                    context._redknot_forward_composite_transaction = transaction
                    context.sequential_q_reservation = q_reservations.get(layer_id)
                    context.shared_dirty_workset = prepared_layers[layer_id]
                local_mode = "coordinator"
        except BaseException as error:
            local_error = error
            local_mode = "invalid"
        # Even a rejected setup must fence the legacy per-layer collective;
        # every rank will either use this forward-wide transaction or dense.
        forward_batch._redknot_mla_off_forward_transaction_attempted = True

        # Proposal construction itself can fail rank-locally.  This one fixed
        # setup rendezvous keeps every rank out of the coordinator collective
        # unless all ranks reached the same execution class.  It replaces 37
        # per-layer v3 readiness votes; it is not an omission certificate.
        setup_modes = ("full_local", "coordinator", "invalid")
        try:
            setup_signal = [0] * len(setup_modes)
            setup_signal[setup_modes.index(local_mode)] = 1
            setup_signal.append(int(local_error is None))
            reduced_setup = torch.tensor(
                setup_signal, dtype=torch.int64, device=device
            )
            if int(self._redknot_tp_size) > 1:
                with _redknot_mla_timed("mla_prepare_setup_vote"):
                    reduced_setup = self._mla_off_control_all_reduce(
                        reduced_setup
                    )
            totals = tuple(int(value) for value in reduced_setup.tolist())
        except BaseException as vote_error:
            if isinstance(transaction, _RedKnotForwardCompositeTransaction):
                self._fail_stop_mla_off_transaction(
                    transaction,
                    reason_code="forward_setup_vote_indeterminate",
                    detail=f"{type(vote_error).__name__}: {vote_error}",
                )
            self._fail_stop_mla_off_without_transaction(
                forward_id=str(
                    getattr(resources, "forward_id", "redknot-forward-setup")
                ),
                reason_code="forward_setup_vote_indeterminate",
                detail=f"{type(vote_error).__name__}: {vote_error}",
            )
        world = int(self._redknot_tp_size)
        agreed = tuple(
            setup_modes[index]
            for index, count in enumerate(totals[: len(setup_modes)])
            if count == world
        )
        if (
            totals[-1] != world
            or len(agreed) != 1
            or sum(totals[: len(setup_modes)]) != world
            or agreed[0] == "invalid"
        ):
            forward_batch._redknot_mla_off_disabled = True
            self._cleanup_uncommitted_mla_off_forward_collectively(
                forward_batch=forward_batch,
                device=device,
                stage="forward_setup_reject",
            )
            self._mla_off_log_failure(
                "forward_composite_setup_failed",
                f"stage={setup_stage}; {type(local_error).__name__}: {local_error}"
                if local_error is not None
                else f"TP setup modes disagreed: {totals!r}",
            )
            return None
        if agreed[0] == "full_local":
            if not isinstance(transaction, _RedKnotForwardCompositeTransaction):
                raise AssertionError("full-local setup lost its transaction owner")
            self._mla_off_log_composite_forward_manifest(
                transaction.context_for(expected_layers[0])
            )
            self._mla_off_log_forward_transaction_full_local_statuses(
                transaction=transaction,
                active_plans=active,
                expected_layers=expected_layers,
            )
            return transaction

        assert (
            proposal is not None
            and validated_batch is not None
            and slot_bounds_certificate is not None
            and coordinator is not None
            and isinstance(transaction, _RedKnotForwardCompositeTransaction)
        )
        if proposal.omission_profile == "full" and q_arena is None:
            raise AssertionError("full forward lost its sequential Q arena")
        if proposal.omission_profile == "shared_only" and q_arena is not None:
            raise AssertionError("shared-only forward unexpectedly owns a Q arena")
        try:
            with _redknot_mla_timed("mla_prepare_commit_vote"):
                outcome = coordinator.commit(collective_adapter, ready=True)
        except BaseException as commit_error:
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="forward_prepare_exception",
                detail=f"{type(commit_error).__name__}: {commit_error}",
            )
        if not outcome.committed:
            for context in contexts.values():
                context.composite_dense_fallback = True
            forward_batch._redknot_mla_off_disabled = True
            self._cleanup_uncommitted_mla_off_forward_collectively(
                forward_batch=forward_batch,
                device=device,
                stage="forward_commit_reject",
            )
            return None

        try:
            with _redknot_mla_timed("mla_prepare_bind_contexts"):
                for context in contexts.values():
                    coordinator.bind_context(context)
                if proposal.omission_profile == "full":
                    coordinator.register_prepared_full_layers(
                        contexts=contexts,
                        q_arena=q_arena,
                        q_reservations=q_reservations,
                        prepared_layers=prepared_layers,
                    )
        except BaseException as bind_error:
            coordinator.session.fail_closed_after_commit(
                collective_adapter,
                reason_code="forward_context_bind_failed",
                detail=f"{type(bind_error).__name__}: {bind_error}",
            )

        restore_batch_receipt = None
        restore_error = None
        restore_stream = None
        restore_completion_event = None
        restore_stream_identity = ()
        restore_group_events = {}
        restore_layer_groups = {}
        receipts_by_layer = {layer_id: [] for layer_id in contexts}
        try:
            from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
                restore_clean_batched,
                restore_clean_batched_pipelined,
            )

            restore_device = torch.device(device)
            if restore_device.type == "cuda":
                stream_key = int(
                    restore_device.index
                    if restore_device.index is not None
                    else torch.cuda.current_device()
                )
                restore_stream = self._redknot_restore_batch_streams.get(
                    stream_key
                )
                if restore_stream is None:
                    restore_stream = torch.cuda.Stream(device=restore_device)
                    self._redknot_restore_batch_streams[stream_key] = (
                        restore_stream
                    )
                caller_stream = torch.cuda.current_stream(restore_device)
                restore_stream.wait_stream(caller_stream)
                if transaction.restore_pipeline_plan is not None:
                    restore_group_events = {
                        int(group.ordinal): torch.cuda.Event(
                            blocking=False,
                            interprocess=False,
                        )
                        for group in transaction.restore_pipeline_plan.groups
                    }
                    restore_layer_groups = {
                        int(layer_id): int(group.ordinal)
                        for group in transaction.restore_pipeline_plan.groups
                        for layer_id in group.layer_ids
                    }
                    restore_completion_event = restore_group_events[
                        int(transaction.restore_pipeline_plan.groups[-1].ordinal)
                    ]
                else:
                    restore_completion_event = torch.cuda.Event(
                        blocking=False,
                        interprocess=False,
                    )
                restore_stream_identity = (
                    restore_device,
                    int(restore_stream.cuda_stream),
                )
                # Publish ownership before the first family launch.  If a
                # later callback raises, model-level teardown can still find
                # the stream/event and the validated inputs/workspace.
                transaction.restore_stream = restore_stream
                transaction.restore_completion_event = (
                    restore_completion_event
                )
                transaction.restore_stream_identity = restore_stream_identity
                transaction.restore_group_events = dict(restore_group_events)
                transaction.restore_layer_groups = dict(restore_layer_groups)
                with torch.cuda.stream(restore_stream):
                    try:
                        with _redknot_mla_timed("mla_prepare_restore_enqueue"):
                            if transaction.restore_pipeline_plan is None:
                                restore_batch_receipt = restore_clean_batched(
                                    validated_batch
                                )
                            else:
                                def _record_restore_group(group):
                                    ordinal = int(group.ordinal)
                                    event = restore_group_events[ordinal]
                                    event.record(restore_stream)
                                    transaction.restore_group_recorded.add(
                                        ordinal
                                    )

                                restore_batch_receipt = (
                                    restore_clean_batched_pipelined(
                                        validated_batch,
                                        pipeline=transaction.restore_pipeline_plan,
                                        group_enqueued=_record_restore_group,
                                    )
                                )
                    finally:
                        # Some family launches may already be enqueued when a
                        # later callback raises.  Always fence the stream so
                        # workspace/input/pin lifetime remains knowable.
                        try:
                            final_group = (
                                int(
                                    transaction.restore_pipeline_plan.groups[
                                        -1
                                    ].ordinal
                                )
                                if transaction.restore_pipeline_plan is not None
                                else None
                            )
                            if (
                                final_group is None
                                or final_group
                                not in transaction.restore_group_recorded
                            ):
                                restore_completion_event.record(restore_stream)
                        except BaseException as event_error:
                            coordinator.session.fail_closed_after_commit(
                                collective_adapter,
                                reason_code="restore_event_record_failed",
                                detail=(
                                    f"{type(event_error).__name__}: "
                                    f"{event_error}"
                                ),
                            )
                            raise RuntimeError(
                                "restore event fail-stop hook unexpectedly returned"
                            ) from event_error
                        # Workspace descriptors and every source/input remain
                        # fenced even when restore_clean_batched raised after
                        # enqueuing only a subset of its three families.
                        transaction.restore_event_recorded = True
                        self._redknot_restore_batch_workspace_event = (
                            restore_completion_event
                        )
            else:
                if transaction.restore_pipeline_plan is None:
                    restore_batch_receipt = restore_clean_batched(validated_batch)
                else:
                    restore_batch_receipt = restore_clean_batched_pipelined(
                        validated_batch,
                        pipeline=transaction.restore_pipeline_plan,
                    )
            expected_restore_launches = (
                3
                if transaction.restore_pipeline_plan is None
                else sum(
                    len(group.family_spans)
                    for group in transaction.restore_pipeline_plan.groups
                )
            )
            if (
                int(restore_batch_receipt.launch_count)
                != 2 + int(expected_restore_launches)
                or int(restore_batch_receipt.validation_launch_count) != 2
                or int(restore_batch_receipt.restore_launch_count)
                != int(expected_restore_launches)
            ):
                raise RuntimeError(
                    "restore batch did not preserve the certified validation + "
                    "family-slice launch contract"
                )
            if (
                str(restore_batch_receipt.forward_id)
                != str(validated_batch.plan.forward_id)
                or str(restore_batch_receipt.batch_digest)
                != str(validated_batch.plan.digest)
                or int(restore_batch_receipt.operation_count)
                != int(validated_batch.plan.operation_count)
                or int(restore_batch_receipt.restored_value_count)
                != int(validated_batch.plan.restored_value_count)
                or int(restore_batch_receipt.descriptor_h2d_bytes) <= 0
                or int(restore_batch_receipt.validation_control_h2d_bytes)
                <= 0
                or int(restore_batch_receipt.validation_status_d2h_bytes)
                <= 0
                or int(restore_batch_receipt.validation_memset_bytes) <= 0
            ):
                raise RuntimeError("restore batch receipt audit changed")
            if len(restore_batch_receipt.input_receipts) != len(
                batch_receipt_bindings
            ):
                raise RuntimeError("restore batch receipt ordinal count changed")
            for input_ordinal, binding in enumerate(batch_receipt_bindings):
                layer_id, request_index, layer_adapter, schedule = binding
                expected_binding = (
                    int(validated_batch.inputs[input_ordinal].request_index),
                    int(layer_id),
                )
                if (
                    tuple(restore_batch_receipt.input_bindings[input_ordinal])
                    != expected_binding
                ):
                    raise RuntimeError(
                        "restore batch receipt request/layer binding changed"
                    )
                device_receipt = restore_batch_receipt.for_input(input_ordinal)
                # This validates the request/layer receipt against the exact
                # adapter schedule; dirty builder metadata remains the already
                # preflighted aggregate layer object below.
                with _redknot_mla_timed(
                    "mla_prepare_bind_receipt", layer_id=layer_id
                ):
                    layer_adapter.bind_batch_receipt(
                        device_receipt=device_receipt,
                        schedule=schedule,
                        request_index=int(request_index),
                    )
                receipts_by_layer[layer_id].append(device_receipt)
        except BaseException as error:
            restore_error = error
            coordinator.record_pipeline_failure(
                layer_id=int(resources.expected_layer_ids[0]),
                stage="forward_batched_restore",
                error=error,
            )
            if self._mla_off_postcommit_error_is_fatal(error):
                self._fail_stop_mla_off_transaction(
                    transaction,
                    reason_code="forward_batched_restore_resource_failure",
                    detail=f"{type(error).__name__}: {error}",
                )
        try:
            transaction.restore_batch_receipt = restore_batch_receipt
            transaction.restore_stream = restore_stream
            transaction.restore_completion_event = restore_completion_event
            transaction.restore_stream_identity = restore_stream_identity
            transaction.restore_group_events = dict(restore_group_events)
            transaction.restore_layer_groups = dict(restore_layer_groups)
            for layer_id, context in contexts.items():
                # This is deliberately a tuple for all request-scoped inputs
                # of one layer; downstream code must never treat it as a
                # single-request receipt.
                context.shared_restore_receipts = tuple(
                    receipts_by_layer[layer_id]
                )
                context.shared_restore_applied = restore_error is None
                if restore_error is not None:
                    context.record_pipeline_error(restore_error)
        except BaseException as publish_error:
            self._fail_stop_mla_off_transaction(
                transaction,
                reason_code="forward_restore_receipt_publish_failed",
                detail=f"{type(publish_error).__name__}: {publish_error}",
            )
        self._count("mla_off.forward_composite_prepare_commits")
        if restore_error is None:
            # One forward-wide device batch replaces the legacy per-layer
            # restore calls.  Publish the exact already-certified aggregate
            # receipt counts so CONTROLLER_STATS can prove real device work.
            self._count("mla_off.shared_device_restore_calls", 1)
            self._count(
                "mla_off.shared_device_restore_operations",
                int(restore_batch_receipt.operation_count),
            )
            self._count(
                "mla_off.shared_clean_restore_operations",
                int(restore_batch_receipt.operation_count),
            )
            self._count(
                "mla_off.shared_device_values_restored",
                int(restore_batch_receipt.restored_value_count),
            )
        self._mla_off_log_composite_forward_manifest(
            transaction.context_for(expected_layers[0])
        )
        return transaction

    def _mla_off_log_failure(self, key: str, detail: str) -> None:
        self._count(f"mla_off.fallback.{key}")
        log_key = (key, detail)
        if log_key in self._redknot_mla_off_logged_failures:
            return
        self._redknot_mla_off_logged_failures.add(log_key)
        logger.warning("RedKnot MLA-off disabled for this forward (%s): %s", key, detail)

    def _mla_off_quarantine_shared_latent(self, reason: str) -> None:
        """Permanently fence a worker whose shared transaction is unknown.

        This is reserved for an incomplete rollback or an irreversible
        confirmation failure.  Ordinary preflight/cache misses must continue
        to use request-local dense fallback and must not call this method.
        """

        reason = str(reason) or "unknown shared-latent transaction failure"
        self._redknot_shared_latent_poisoned = True
        self._redknot_shared_latent_poison_reason = reason
        self._redknot_shared_latent_enabled = False
        self._redknot_mla_off_enabled = False
        self._mla_off_log_failure("shared_latent_quarantined", reason)

    @staticmethod
    def _mla_off_request_id(plan) -> str:
        if not isinstance(plan, Mapping):
            return ""
        return str(plan.get("benchmark_request_id", "") or "")

    @staticmethod
    def _mla_off_forward_mode_name(forward_batch: ForwardBatch) -> str:
        mode = getattr(forward_batch, "forward_mode", None)
        name = str(getattr(mode, "name", "") or "")
        if not name:
            name = str(mode).rsplit(".", 1)[-1]
        return "_".join(name.lower().split()) or "unknown"

    def _mla_off_forward_evidence(
        self,
        *,
        forward_batch: ForwardBatch,
        plan,
        positions: torch.Tensor,
        positions_cpu: Optional[torch.Tensor] = None,
        q_rows: int,
    ) -> Tuple[str, str, str, int, int, int]:
        """Return a stable identity shared by every local layer in one forward."""

        request_id = self._mla_off_request_id(plan)
        forward_mode = self._mla_off_forward_mode_name(forward_batch)
        q_rows = int(q_rows)
        if not request_id:
            return "", "", forward_mode, q_rows, -1, -1
        position_cache_key = self._mla_off_forward_tensor_cache_key(
            forward_batch,
            positions,
        )
        cache_key = (request_id, forward_mode, q_rows, position_cache_key)
        cached = getattr(
            forward_batch, "_redknot_mla_off_forward_evidence", None
        )
        if (
            isinstance(cached, tuple)
            and len(cached) == 3
            and position_cache_key is not None
            and cached[0] == cache_key
            and cached[1] is positions
        ):
            return cached[2]
        authoritative_positions = positions_cpu
        if authoritative_positions is None and positions.device.type == "cpu":
            authoritative_positions = positions
        try:
            if authoritative_positions is None:
                raise ValueError("CPU position authority is unavailable")
            position_start = int(authoritative_positions[0].item())
            position_end = int(authoritative_positions[-1].item()) + 1
        except Exception:
            position_start = position_end = -1
        original_chunk_range = getattr(
            forward_batch, "redknot_original_chunk_token_range", None
        )
        if (
            isinstance(original_chunk_range, tuple)
            and len(original_chunk_range) == 2
            and all(type(value) is int for value in original_chunk_range)
            and 0 <= original_chunk_range[0] < original_chunk_range[1]
            and position_start >= original_chunk_range[0]
            and position_end <= original_chunk_range[1]
        ):
            # Selected-row execution is intentionally non-contiguous.  Its
            # q_rows count is the physical packed work, while evidence coverage
            # must retain the scheduler-authenticated logical chunk span.  This
            # lets the formal audit prove 8x document coverage without falsely
            # pretending every token was recomputed.
            position_start, position_end = original_chunk_range
        identity = (
            f"{request_id}\0{forward_mode}\0{q_rows}\0"
            f"{position_start}\0{position_end}"
        ).encode("utf-8")
        forward_id = "f" + hashlib.sha256(identity).hexdigest()[:16]
        evidence = (
            request_id,
            forward_id,
            forward_mode,
            q_rows,
            position_start,
            position_end,
        )
        # ForwardBatch is normally stable across a layer traversal, but some
        # scheduler paths can reuse the object for another prefill invocation.
        # Key the cache by the current positions tensor identity instead of the
        # request alone so a reused object cannot inherit the previous span.
        forward_batch._redknot_mla_off_forward_evidence = (
            cache_key,
            positions,
            evidence,
        )
        return evidence

    def _mla_off_log_forward_start(
        self,
        *,
        request_id: str,
        forward_id: str,
        forward_mode: str,
        q_rows: int,
        position_start: int,
        position_end: int,
        position_contiguous: bool,
        plan_mode: str,
        diagnostic_ablation: str = "full",
    ) -> None:
        """Declare each measured prefill forward before per-layer evidence."""

        if (
            os.environ.get("REDKNOT_MLA_OFF_METRICS", "0") != "1"
            or not request_id
            or not forward_id
        ):
            return
        logger.info(
            "REDKNOT_MLA_OFF_FORWARD request_id=%s forward_id=%s "
            "forward_mode=%s q_rows=%d position_start=%d position_end=%d "
            "position_contiguous=%d plan_mode=%s diagnostic_ablation=%s",
            request_id,
            forward_id,
            forward_mode,
            int(q_rows),
            int(position_start),
            int(position_end),
            int(bool(position_contiguous)),
            str(plan_mode).replace(" ", "_"),
            str(diagnostic_ablation).replace(" ", "_"),
        )

    def _mla_off_log_composite_forward_manifest(self, context) -> None:
        """Publish one single-request manifest for a composite restore forward."""

        resources = getattr(
            context,
            "_redknot_composite_forward_resources",
            None,
        )
        geometry = getattr(resources, "geometry", None)
        requests = tuple(getattr(geometry, "requests", ()) or ())
        if len(requests) != 1:
            # The current formal benchmark certifies one synchronous request.
            # Never collapse a continuously batched geometry into a fake span.
            return
        logical_positions = tuple(
            int(value) for value in requests[0].logical_positions
        )
        position_start = logical_positions[0] if logical_positions else -1
        position_end = logical_positions[-1] + 1 if logical_positions else -1
        owner = getattr(context, "_redknot_forward_batch_owner", None)
        original_chunk_range = getattr(
            owner, "redknot_original_chunk_token_range", None
        )
        if (
            isinstance(original_chunk_range, tuple)
            and len(original_chunk_range) == 2
            and all(type(value) is int for value in original_chunk_range)
            and 0 <= original_chunk_range[0] < original_chunk_range[1]
            and position_start >= original_chunk_range[0]
            and position_end <= original_chunk_range[1]
        ):
            # Combined selected-row execution packs only physical online rows.
            # The manifest's position span is logical coverage, certified by
            # the scheduler before selection, while q_rows remains the exact
            # amount of physical work.  Keep position_contiguous below tied to
            # the actual selected rows so evidence never pretends they were a
            # dense token interval.
            position_start, position_end = original_chunk_range
        self._mla_off_log_forward_start(
            request_id=str(getattr(context, "benchmark_request_id", "")),
            forward_id=str(getattr(context, "benchmark_forward_id", "")),
            forward_mode=str(
                getattr(context, "benchmark_forward_mode", "unknown")
            ),
            q_rows=int(getattr(context, "benchmark_q_rows", 0)),
            position_start=position_start,
            position_end=position_end,
            position_contiguous=bool(
                logical_positions
                and all(
                    right == left + 1
                    for left, right in zip(
                        logical_positions,
                        logical_positions[1:],
                    )
                )
            ),
            plan_mode="restore",
            diagnostic_ablation=str(
                getattr(context, "diagnostic_ablation", "full") or "full"
            ),
        )

    def _mla_off_log_forward_transaction_full_local_statuses(
        self,
        *,
        transaction: _RedKnotForwardCompositeTransaction,
        active_plans: Tuple[Mapping[str, object], ...],
        expected_layers: Tuple[int, ...],
    ) -> None:
        """Emit the per-layer proof skipped by the transaction fast path."""

        if not active_plans:
            raise RuntimeError("full-local transaction has no active restore plan")
        for layer_id in expected_layers:
            context = transaction.context_for(int(layer_id))
            if not bool(getattr(context, "is_full_local", False)):
                raise RuntimeError("full-local transaction context changed mode")
            reason = str(
                getattr(context, "intentional_full_local_reason", "")
                or "query_suffix_only"
            )
            self._count("mla_off.intentional_full_local_layers")
            for plan in active_plans:
                self._mla_off_log_request_status(
                    plan=plan,
                    layer_id=int(layer_id),
                    status="full_local",
                    reason=reason,
                    forward_id=str(
                        getattr(context, "benchmark_forward_id", "")
                    ),
                    forward_mode=str(
                        getattr(context, "benchmark_forward_mode", "unknown")
                    ),
                    q_rows=int(getattr(context, "benchmark_q_rows", 0)),
                )

    def _mla_off_log_request_status(
        self,
        *,
        plan,
        layer_id: int,
        status: str,
        reason: str,
        forward_id: str = "",
        forward_mode: str = "unknown",
        q_rows: int = 0,
    ) -> None:
        """Emit non-deduplicated request evidence for benchmark fail-closed gates."""

        if os.environ.get("REDKNOT_MLA_OFF_METRICS", "0") != "1":
            return
        request_id = self._mla_off_request_id(plan)
        if not request_id:
            return
        diagnostic_ablation = (
            str(plan.get("mla_off_diagnostic_ablation", "full"))
            if isinstance(plan, Mapping)
            else "invalid"
        )
        logger.info(
            "REDKNOT_MLA_OFF_REQUEST request_id=%s forward_id=%s "
            "forward_mode=%s q_rows=%d layer=%d status=%s reason=%s "
            "diagnostic_ablation=%s",
            request_id,
            str(forward_id or "unattributed"),
            str(forward_mode or "unknown"),
            int(q_rows),
            int(layer_id),
            str(status),
            str(reason).replace(" ", "_"),
            diagnostic_ablation.replace(" ", "_"),
        )

    def _mla_off_shared_restore_counter_snapshot(self) -> Dict[str, int]:
        """Return stable observational counters for shared-GPU restores."""

        return {
            field: int(self._redknot_runtime_counters.get(counter_key, 0))
            for field, counter_key in _SHARED_RESTORE_COUNTER_KEYS.items()
        }

    def _mla_off_begin_transfer_audit(
        self,
        *,
        forward_batch: ForwardBatch,
        controller,
        layer_id: int,
        request_id: str,
        forward_id: str,
        forward_mode: str,
        q_rows: int,
    ):
        """Bind one controller baseline to a single restore forward.

        MLA-off v1 rejects mixed batches, so process-local controller deltas
        are attributable to this one request.  The first rank-local
        local-bearing layer snapshots before any artifact/index restore copy;
        every later layer must recover the same state object.
        """

        if (
            os.environ.get("REDKNOT_MLA_OFF_METRICS", "0") != "1"
            or not request_id
            or not forward_id
        ):
            return None
        local_layers = tuple(self._redknot_mla_off_rank_local_layer_ids)
        if not local_layers:
            raise RuntimeError("MLA-off transfer audit has no local-bearing layers")
        key = (
            str(request_id),
            str(forward_id),
            str(forward_mode or "unknown"),
            int(q_rows),
        )
        state = getattr(
            forward_batch, "_redknot_mla_off_transfer_audit_state", None
        )
        if int(layer_id) == int(local_layers[0]):
            if (
                isinstance(state, dict)
                and state.get("key") == key
                and not bool(state.get("emitted", False))
            ):
                raise RuntimeError(
                    "MLA-off transfer audit baseline was initialized twice"
                )
            state = {
                "key": key,
                "controller": controller,
                "baseline": dict(controller.snapshot_stats()),
                "shared_restore_baseline": (
                    self._mla_off_shared_restore_counter_snapshot()
                ),
                "emitted": False,
            }
            forward_batch._redknot_mla_off_transfer_audit_state = state
        elif (
            not isinstance(state, dict)
            or state.get("key") != key
            or state.get("controller") is not controller
            or bool(state.get("emitted", False))
        ):
            raise RuntimeError(
                "MLA-off transfer audit baseline is absent or belongs to another "
                "forward"
            )
        return state

    def _mla_off_begin_composite_transfer_audit(
        self,
        *,
        forward_batch: ForwardBatch,
        layer_id: int,
    ):
        """Snapshot v3 transfer counters before composite preparation.

        The composite runtime prepares persistent z_off views and device index
        schedules before it returns an ``MLAOffRuntimeContext``.  Taking this
        baseline afterwards would silently omit those online transfers.  The
        context owns the authoritative benchmark identity, so the first layer
        binds the key only after preparation succeeds.
        """

        if os.environ.get("REDKNOT_MLA_OFF_METRICS", "0") != "1":
            return None
        local_layers = tuple(self._redknot_mla_off_rank_local_layer_ids)
        if not local_layers:
            raise RuntimeError("MLA-off transfer audit has no local-bearing layers")
        from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
            get_dsv4_mla_off_controller,
        )

        controller = get_dsv4_mla_off_controller()
        state = getattr(
            forward_batch, "_redknot_mla_off_transfer_audit_state", None
        )
        if int(layer_id) == int(local_layers[0]):
            if isinstance(state, dict) and not bool(state.get("emitted", False)):
                raise RuntimeError(
                    "composite transfer audit baseline was initialized twice"
                )
            state = {
                # Composite geometry, not a rank-local plan, is authoritative
                # for request/forward identity.  Bind this after preparation.
                "key": None,
                "controller": controller,
                "baseline": dict(controller.snapshot_stats()),
                "shared_restore_baseline": (
                    self._mla_off_shared_restore_counter_snapshot()
                ),
                "emitted": False,
            }
            forward_batch._redknot_mla_off_transfer_audit_state = state
        elif (
            not isinstance(state, dict)
            or state.get("controller") is not controller
            or state.get("key") is None
            or bool(state.get("emitted", False))
        ):
            raise RuntimeError(
                "composite transfer audit baseline is absent or stale"
            )
        return state

    @staticmethod
    def _mla_off_bind_composite_transfer_audit(state, context) -> None:
        """Bind a pre-composite baseline to its authoritative context id."""

        if state is None:
            return
        if not isinstance(state, dict):
            raise TypeError("composite transfer audit state is malformed")
        controller = getattr(context, "controller", None)
        if state.get("controller") is not controller:
            raise RuntimeError("composite transfer audit controller changed")
        key = (
            str(getattr(context, "benchmark_request_id", "") or ""),
            str(getattr(context, "benchmark_forward_id", "") or ""),
            str(getattr(context, "benchmark_forward_mode", "unknown") or "unknown"),
            int(getattr(context, "benchmark_q_rows", 0)),
        )
        if not key[0] or not key[1] or key[3] <= 0:
            raise RuntimeError("composite transfer audit identity is incomplete")
        if state.get("key") is None:
            state["key"] = key
        elif state.get("key") != key:
            raise RuntimeError("composite transfer audit forward identity changed")
        context.transfer_audit_state = state

    def _mla_off_maybe_emit_transfer_audit(
        self, *, layer_id: int, context
    ) -> None:
        """Best-effort transfer evidence that never changes model execution."""

        try:
            self._mla_off_emit_transfer_audit(
                layer_id=layer_id, context=context
            )
        except Exception:
            # This runs after attention_application consensus. A rank-local
            # JSON/print/state failure must not strand peers at the model's
            # next TP collective. Missing evidence makes the benchmark fail.
            try:
                self._count("mla_off.transfer_audit_publish_failures")
            except Exception:
                pass

    def _mla_off_emit_transfer_audit(
        self, *, layer_id: int, context
    ) -> None:
        """Validate and emit one all-rank-parsable transfer event."""

        state = getattr(context, "transfer_audit_state", None)
        if state is None:
            return
        local_layers = tuple(self._redknot_mla_off_rank_local_layer_ids)
        if not local_layers or int(layer_id) != int(local_layers[-1]):
            return
        if not isinstance(state, dict) or bool(state.get("emitted", False)):
            raise RuntimeError("MLA-off transfer audit would emit more than once")
        controller = getattr(context, "controller", None)
        if state.get("controller") is not controller:
            raise RuntimeError("MLA-off transfer audit controller changed")
        expected_key = (
            str(getattr(context, "benchmark_request_id", "") or ""),
            str(getattr(context, "benchmark_forward_id", "") or ""),
            str(
                getattr(context, "benchmark_forward_mode", "unknown")
                or "unknown"
            ),
            int(getattr(context, "benchmark_q_rows", 0)),
        )
        if state.get("key") != expected_key:
            raise RuntimeError("MLA-off transfer audit forward identity changed")
        from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
            MLA_OFF_TRANSFER_AUDIT_SCHEMA,
            MLA_OFF_TRANSFER_BYTE_SEMANTICS,
            MLA_OFF_TRANSFER_COUNTER_FIELDS,
            MLA_OFF_TRANSFER_GAUGE_FIELDS,
        )

        baseline = state.get("baseline")
        if not isinstance(baseline, Mapping):
            raise RuntimeError("MLA-off transfer audit baseline is malformed")
        after = dict(controller.snapshot_stats())
        counter_start = {}
        counter_end = {}
        counter_delta = {}
        for field in MLA_OFF_TRANSFER_COUNTER_FIELDS:
            start_value = int(baseline.get(field, 0))
            end_value = int(after.get(field, 0))
            if start_value < 0 or end_value < start_value:
                raise RuntimeError(
                    f"MLA-off transfer counter regressed: {field} "
                    f"start={start_value} end={end_value}"
                )
            counter_start[field] = start_value
            counter_end[field] = end_value
            counter_delta[field] = end_value - start_value
        gauges = {
            field: int(after.get(field, 0))
            for field in MLA_OFF_TRANSFER_GAUGE_FIELDS
        }
        shared_start = state.get("shared_restore_baseline")
        if not isinstance(shared_start, Mapping):
            raise RuntimeError("shared restore audit baseline is malformed")
        shared_end = self._mla_off_shared_restore_counter_snapshot()
        shared_delta = {}
        for field in _SHARED_RESTORE_COUNTER_KEYS:
            start_value = int(shared_start.get(field, 0))
            end_value = int(shared_end.get(field, 0))
            if start_value < 0 or end_value < start_value:
                raise RuntimeError(
                    f"shared restore counter regressed: {field} "
                    f"start={start_value} end={end_value}"
                )
            shared_delta[field] = end_value - start_value
        payload = {
            "schema": MLA_OFF_TRANSFER_AUDIT_SCHEMA,
            "byte_semantics": MLA_OFF_TRANSFER_BYTE_SEMANTICS,
            "request_id": expected_key[0],
            "forward_id": expected_key[1],
            "forward_mode": expected_key[2],
            "q_rows": expected_key[3],
            "tp_rank": int(self._redknot_tp_rank),
            "tp_size": int(self._redknot_tp_size),
            "diagnostic_ablation": str(
                getattr(context, "diagnostic_ablation", "full") or "full"
            ),
            "counter_start": counter_start,
            "counter_end": counter_end,
            "counter_delta": counter_delta,
            "gauge_snapshot": gauges,
            "shared_restore": {
                "schema": _SHARED_RESTORE_AUDIT_SCHEMA,
                "counter_start": dict(shared_start),
                "counter_end": shared_end,
                "counter_delta": shared_delta,
            },
        }
        # Rank INFO logging can be filtered.  Scheduler stdout is independently
        # redirected to rank{N}.log, so write and flush the compact JSON there.
        print(
            "REDKNOT_MLA_OFF_CONTROLLER_STATS "
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        state["emitted"] = True

    def _mla_off_record_runtime_rows(
        self,
        *,
        request_id: str,
        forward_id: str,
        forward_mode: str,
        q_rows: int,
        layer_id: int,
        reused_local_head_rows: int,
        online_local_head_rows: int,
        online_global_head_rows: int,
        mla_off_context=None,
    ) -> None:
        """Count every measured prefill path, including intentional full-online."""

        reused_local_head_rows = int(reused_local_head_rows)
        online_local_head_rows = int(online_local_head_rows)
        online_global_head_rows = int(online_global_head_rows)
        self._count("mla_off.reused_local_head_rows", reused_local_head_rows)
        self._count("mla_off.online_local_head_rows", online_local_head_rows)
        self._count("mla_off.online_global_head_rows", online_global_head_rows)
        if os.environ.get("REDKNOT_MLA_OFF_METRICS", "0") == "1":
            try:
                logger.info(
                    "REDKNOT_MLA_OFF_METRIC request_id=%s forward_id=%s "
                    "forward_mode=%s q_rows=%d layer=%d "
                    "reused_local_head_rows=%d online_local_head_rows=%d "
                    "online_global_head_rows=%d diagnostic_ablation=%s",
                    str(request_id or "unattributed"),
                    str(forward_id or "unattributed"),
                    str(forward_mode or "unknown"),
                    int(q_rows),
                    int(layer_id),
                    reused_local_head_rows,
                    online_local_head_rows,
                    online_global_head_rows,
                    str(
                        getattr(mla_off_context, "diagnostic_ablation", "full")
                        or "full"
                    ),
                )
            except Exception:
                try:
                    self._count("mla_off.runtime_metric_publish_failures")
                except Exception:
                    pass
        if mla_off_context is not None:
            self._mla_off_maybe_emit_transfer_audit(
                layer_id=layer_id, context=mla_off_context
            )

    def _mla_off_vote_count(self, value: bool, device) -> int:
        if self._redknot_tp_size <= 1:
            return int(bool(value))
        signal = torch.tensor(
            [1 if value else 0], dtype=torch.int32, device=device
        )
        signal = self._mla_off_control_all_reduce(signal)
        return int(signal.item())

    def _mla_off_control_all_reduce(self, signal: torch.Tensor) -> torch.Tensor:
        """Reduce exact control-plane values without data-plane custom kernels.

        MLA-off consensus carries int64 digest moments.  SGLang's custom
        all-reduce v2 kernels are specialized for floating-point activation
        tensors and cannot compile an int64 instantiation.  Use the native
        process group for these tiny control collectives so their integer
        values remain exact and no dtype-specific JIT path is selected.

        Lightweight test groups do not expose ``device_group``; retaining the
        coordinator fallback keeps the consensus helpers unit-testable.
        """

        if self._redknot_tp_size <= 1:
            return signal
        tp_group = get_attention_tp_group()
        device_group = getattr(tp_group, "device_group", None)
        if device_group is None:
            return tp_group.all_reduce(signal)
        torch.distributed.all_reduce(signal, group=device_group)
        return signal

    def _mla_off_vote_restore_ready(self, ready: bool, device) -> bool:
        """Require every attention-TP rank to preload before any head is skipped."""

        return self._mla_off_vote_count(ready, device) == int(
            self._redknot_tp_size
        )

    def _mla_off_snapshot_digest_vote(
        self,
        *,
        local_ready: bool,
        local_digest: str,
        common_digest: str,
        device,
    ) -> Tuple[bool, Tuple[str, ...], str]:
        """Collect one different SHA-256 digest from each attention-TP rank.

        Snapshot payloads are rank-local because z_off owns different logical
        heads.  An equality/moment vote would therefore reject a correct
        snapshot.  Each rank writes 18 base-2^15 limbs into its own fixed row;
        a single SUM all-reduce is an all-gather with an explicit one-hot rank
        participation proof.  The returned aggregate digest is identical on
        every rank and the local digest remains independently certificate-bound.
        """

        limb_bits = 15
        limb_count = 18
        row_width = 1 + 2 * limb_count
        world = int(self._redknot_tp_size)
        rank = int(self._redknot_tp_rank)
        digest_ok = bool(
            local_ready
            and isinstance(local_digest, str)
            and local_digest.startswith("sha256:")
            and len(local_digest) == 71
            and isinstance(common_digest, str)
            and common_digest.startswith("sha256:")
            and len(common_digest) == 71
        )
        digest_value = 0
        common_value = 0
        if digest_ok:
            try:
                digest_value = int(local_digest[7:], 16)
                common_value = int(common_digest[7:], 16)
            except ValueError:
                digest_ok = False
                digest_value = 0
                common_value = 0
        values = [0] * (world * row_width)
        base = rank * row_width
        values[base] = int(digest_ok)
        for index in range(limb_count):
            values[base + 1 + index] = (
                digest_value >> (index * limb_bits)
            ) & ((1 << limb_bits) - 1)
            values[base + 1 + limb_count + index] = (
                common_value >> (index * limb_bits)
            ) & ((1 << limb_bits) - 1)
        signal = torch.tensor(values, dtype=torch.int64, device=device)
        if world > 1:
            signal = self._mla_off_control_all_reduce(signal)
        reduced = tuple(int(value) for value in signal.tolist())
        digests = []
        common_digests = []
        all_ready = True
        for owner in range(world):
            start = owner * row_width
            if reduced[start] != 1:
                all_ready = False
            value = 0
            for index, limb in enumerate(
                reduced[start + 1 : start + 1 + limb_count]
            ):
                if not 0 <= limb < (1 << limb_bits):
                    all_ready = False
                    limb = 0
                value |= int(limb) << (index * limb_bits)
            digests.append("sha256:" + f"{value:064x}")
            common = 0
            for index, limb in enumerate(
                reduced[start + 1 + limb_count : start + row_width]
            ):
                if not 0 <= limb < (1 << limb_bits):
                    all_ready = False
                    limb = 0
                common |= int(limb) << (index * limb_bits)
            common_digests.append("sha256:" + f"{common:064x}")
        if len(set(common_digests)) != 1:
            all_ready = False
        aggregate = "sha256:" + hashlib.sha256(
            json.dumps(
                (tuple(common_digests), tuple(digests)),
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        return all_ready, tuple(digests), aggregate

    def resolve_mla_off_attention_application(
        self, *, local_success: bool, device
    ) -> Tuple[bool, str]:
        """Fixed post-``_forward_headwise`` vote reached on success or error."""

        success_count = self._mla_off_vote_count(local_success, device)
        if success_count != int(self._redknot_tp_size):
            return False, "attention_application_failed"
        return True, ""

    def commit_mla_off_sparse_q(
        self,
        *,
        mla_off_context,
        projection: object,
        local_success: bool,
        device,
    ) -> bool:
        """Commit rank-local sparse Q only after every TP rank finished it.

        A false vote is still safe: no attention head has been omitted yet, so
        the model can discard the proposal and recompute its ordinary complete
        Q tensor.  Once this method returns ``True``, native fallback is no
        longer legal because clean local-head slots are intentionally absent.
        """

        local_commit_error = ""
        from sglang.srt.layers.attention.redknot.dsv4_sparse_q_runtime import (
            PackedSparseQProjection,
        )

        packed_projection = isinstance(projection, PackedSparseQProjection)
        tensor_projection = isinstance(projection, torch.Tensor)
        eligible = bool(
            local_success
            and mla_off_context is not None
            and getattr(mla_off_context, "is_restore", False)
            and int(getattr(mla_off_context, "reused_row_count", 0)) > 0
            and (tensor_projection or packed_projection)
        )
        plan = getattr(mla_off_context, "sparse_q_plan", None)
        if eligible and packed_projection:
            try:
                projection.validate()
                if projection.plan is not plan:
                    raise ValueError("packed sparse-Q uses another context plan")
            except BaseException:
                eligible = False
        # Vote not only readiness but the cross-rank invariants. Logical head
        # ids differ by rank, so the consensus intentionally binds the common
        # row geometry/counts rather than the full rank-local plan digest.
        if eligible:
            row_digest = tuple(
                int(value)
                for value in getattr(
                    mla_off_context, "reuse_mask_digest", (0, 0)
                )
            )
            invariants = (
                int(getattr(plan, "layer_id", -1)),
                int(getattr(plan, "q_rows", -1)),
                len(getattr(plan, "local_head_axes", ())),
                len(getattr(plan, "global_head_axes", ())),
                len(getattr(plan, "online_local_rows", ())),
                int(row_digest[0]),
                int(row_digest[1]),
            )
        else:
            invariants = (-1, -1, -1, -1, -1, 0, 0)
        signal_values = [int(eligible)]
        for value in invariants:
            signal_values.extend((int(value), int(value) * int(value)))
        signal = torch.tensor(signal_values, dtype=torch.int64, device=device)
        if self._redknot_tp_size > 1:
            signal = self._mla_off_control_all_reduce(signal)
        totals = tuple(int(value) for value in signal.tolist())
        ready_count = totals[0]
        world = int(self._redknot_tp_size)
        invariant_consensus = all(
            world * totals[index + 1] == totals[index] * totals[index]
            for index in range(1, len(totals), 2)
        )
        if ready_count != world or not invariant_consensus:
            if mla_off_context is not None:
                clear_commit = getattr(
                    mla_off_context, "clear_sparse_q_commit", None
                )
                if callable(clear_commit):
                    clear_commit()
            self._count(
                "mla_off.sparse_q_fallback"
                if invariant_consensus
                else "mla_off.sparse_q_geometry_mismatch"
            )
            return False

        try:
            from sglang.srt.layers.attention.redknot.dsv4_sparse_q import (
                issue_sparse_q_commit_certificate,
            )

            plan = mla_off_context.sparse_q_plan
            generation = getattr(
                mla_off_context, "benchmark_forward_id", ""
            ) or str(
                getattr(
                    getattr(
                        mla_off_context, "restore_layout_certificate", None
                    ),
                    "layout_key",
                    "",
                )
            )
            collective_token = (
                f"sparse-q:layer={int(mla_off_context.layer_id)}:"
                f"digest={getattr(plan, 'digest', '')}"
            )
            if packed_projection:
                projection_token = str(projection.projection_token)
            else:
                try:
                    tensor_version: object = int(projection._version)
                except RuntimeError:
                    tensor_version = "inference-immutable"
                projection_token = (
                    f"tensor={int(projection.data_ptr())}:version={tensor_version}:"
                    f"shape={tuple(int(v) for v in projection.shape)}"
                )
            certificate = issue_sparse_q_commit_certificate(
                plan,
                generation_id=generation,
                collective_token=collective_token,
                projection_token=projection_token,
                ready_rank_count=ready_count,
            )
            mla_off_context.install_sparse_q_commit(
                plan=plan,
                certificate=certificate,
                projection=projection,
                generation_id=generation,
                collective_token=collective_token,
                projection_token=projection_token,
            )
            local_commit_ok = True
        except BaseException as error:
            local_commit_ok = False
            local_commit_error = str(error)

        committed_count = self._mla_off_vote_count(local_commit_ok, device)
        if committed_count != world:
            if mla_off_context is not None:
                clear_commit = getattr(
                    mla_off_context, "clear_sparse_q_commit", None
                )
                if callable(clear_commit):
                    clear_commit()
            self._mla_off_log_failure(
                "sparse_q_commit_failed",
                (
                    local_commit_error
                    if not local_commit_ok
                    else "another attention-TP rank rejected sparse-Q commit"
                ),
            )
            return False
        self._count("mla_off.sparse_q_committed")
        try:
            logger.info(
                "REDKNOT_SPARSE_Q_METRIC request_id=%s forward_id=%s "
                "layer=%d q_rows=%d projected_head_rows=%d "
                "omitted_head_rows=%d status=committed "
                "diagnostic_ablation=%s",
                str(
                    getattr(
                        mla_off_context, "benchmark_request_id", ""
                    )
                    or "unattributed"
                ),
                str(
                    getattr(
                        mla_off_context, "benchmark_forward_id", ""
                    )
                    or "unattributed"
                ),
                int(plan.layer_id),
                int(plan.q_rows),
                int(plan.projected_head_rows),
                int(plan.omitted_head_rows),
                str(
                    getattr(mla_off_context, "diagnostic_ablation", "full")
                    or "full"
                ),
            )
        except BaseException:
            self._count("mla_off.sparse_q_metric_publish_failures")
        return True

    def commit_mla_off_zoff_only_layer(
        self,
        *,
        mla_off_context,
        projection: object,
        local_success: bool,
        forward_batch: ForwardBatch,
        wo_a_weight: torch.Tensor,
        owned_heads: int,
        groups: int,
        head_dim: int,
        o_lora_rank: int,
        device,
    ) -> bool:
        """Commit packed sparse-Q + persistent z_off without shared KV restore.

        This is an explicitly request-scoped diagnostic path.  It installs the
        same two-kernel head-split merge proof as the composite path, but leaves
        ``shared_restore_states`` empty so model-side WKV, C4/C128 and Indexer
        producers execute every row online.  A precommit failure drops the whole
        context on every rank; it never degrades into an unproved z_off merge.
        """

        transaction = getattr(
            mla_off_context, "_redknot_forward_composite_transaction", None
        )
        if isinstance(transaction, _RedKnotForwardCompositeTransaction) and (
            transaction.coordinator is not None
        ):
            if transaction.omission_profile != "zoff_only":
                raise RuntimeError(
                    "zoff-only commit reached another forward omission profile"
                )
            return self._commit_mla_off_forward_layer(
                transaction=transaction,
                mla_off_context=mla_off_context,
                projection=projection,
                local_success=local_success,
                layer_id=int(mla_off_context.layer_id),
                wo_a_weight=wo_a_weight,
                owned_heads=owned_heads,
                groups=groups,
                head_dim=head_dim,
                o_lora_rank=o_lora_rank,
            )

        from sglang.srt.layers.attention.redknot.dsv4_fused_z_merge import (
            preflight_persistent_headsplit_woa_merge,
        )

        preflight_ok = bool(
            local_success
            and mla_off_context is not None
            and getattr(mla_off_context, "diagnostic_ablation", "full")
            == "zoff_only"
            and not tuple(
                getattr(mla_off_context, "shared_restore_states", ()) or ()
            )
        )
        if preflight_ok:
            try:
                merge_plan = preflight_persistent_headsplit_woa_merge(
                    projection_plan=(
                        mla_off_context.validate_persistent_projection_commit()
                    ),
                    dirty_rows=mla_off_context.online_local_row_indices,
                    dirty_rows_cpu=mla_off_context.online_local_row_indices_cpu,
                    local_head_axes=mla_off_context.local_head_axes,
                    wo_a_weight=wo_a_weight,
                    owned_heads=int(owned_heads),
                    groups=int(groups),
                    head_dim=int(head_dim),
                    o_lora_rank=int(o_lora_rank),
                )
                # Installing a merge proof authorizes no omission by itself.  It
                # is safe to do before the sparse-Q vote and lets a failed rank
                # participate in that fixed collective as not-ready.
                mla_off_context.install_headsplit_woa_merge_plan(merge_plan)
            except BaseException as error:
                self._mla_off_log_failure(
                    "zoff_only_merge_preflight_failed",
                    f"{type(error).__name__}: {error}",
                )
                preflight_ok = False
        committed = self.commit_mla_off_sparse_q(
            mla_off_context=mla_off_context,
            projection=projection,
            local_success=preflight_ok,
            device=device,
        )
        if committed:
            # Sparse-Q omission is now a TP-wide decision.  Unlike an ordinary
            # legacy proposal, zoff_only may no longer escape locally or fall
            # back dense: every later failure is carried to the one fixed
            # post-projection consumer vote.
            mla_off_context.diagnostic_irreversible = True
            return True

        # commit_mla_off_sparse_q returned the same safe reject on every TP
        # rank.  Disable this forward before releasing its persistent z_off and
        # shared-artifact lease, then collectively verify that cleanup itself
        # did not diverge.  The current layer will rebuild full Q/KV and every
        # later layer observes the disabled flag and stays dense.
        cleanup_error = None
        if mla_off_context is not None:
            mla_off_context.composite_dense_fallback = True
        forward_batch._redknot_mla_off_disabled = True
        try:
            from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                close_composite_forward_resources,
            )

            closed = close_composite_forward_resources(forward_batch)
            if not closed:
                raise RuntimeError(
                    "zoff_only reject had no composite resource lease to close"
                )
        except BaseException as error:
            cleanup_error = error
        finally:
            forward_batch._redknot_mla_off_restore_layout = None
        cleanup_ready = self._mla_off_vote_restore_ready(
            cleanup_error is None, device
        )
        if not cleanup_ready:
            raise RuntimeError(
                "zoff_only collective reject could not close every TP resource lease"
            ) from cleanup_error
        self._count("mla_off.zoff_only_collective_rejects")
        return committed

    def preflight_mla_off_sparse_q_backend(
        self,
        *,
        mla_off_context,
        q,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        device,
    ) -> bool:
        """Finish backend validation collectively before sparse Q is consumed.

        This closes the gap between Q commit and ``forward``: no rank may enter
        an early native-fallback branch or raise on mutable restore metadata
        while a peer is already launching RedKnot attention.
        """

        local_ok = True
        local_error = ""
        try:
            if not bool(
                mla_off_context is not None
                and getattr(mla_off_context, "sparse_q_committed", False)
            ):
                raise ValueError("sparse-Q backend preflight has no commit")
            mla_off_context.validate_sparse_q_commit(q)
            if self.redknot_mla_pass_mode != "headwise":
                raise ValueError("sparse-Q requires headwise attention")
            self._maybe_upgrade_forward_metadata()
            if not isinstance(
                self.forward_metadata.core_attn_metadata, DSV4AttnMetadata
            ):
                raise ValueError("sparse-Q requires materialized DSV4 metadata")
            layer_id = int(layer.layer_id)
            if layer_id != int(mla_off_context.layer_id):
                raise ValueError("sparse-Q context belongs to another layer")
            if layer_id not in _PURE_HEADSPLIT_OFFLINE_LAYER_IDS:
                raise ValueError("sparse-Q is outside layers 3..39")
            if (
                forward_batch.forward_mode.is_draft_extend(include_v2=True)
                or layer_id >= self.redknot_mla_head_cfg.num_layers
            ):
                raise ValueError("sparse-Q cannot execute a draft layer")
            layer_plan = self._redknot_dual_layer_plans[layer_id]
            if not layer_plan.local_groups:
                raise ValueError("sparse-Q layer has no offline-local heads")
            q_heads = int(q.shape[2]) if int(q.ndim) == 4 else int(q.shape[1])
            if self._headwise_owned_view(q_heads, layer.tp_q_head_num) is None:
                raise ValueError("sparse-Q owned-head view is invalid")
            token_to_kv_pool = self.token_to_kv_pool
            if not isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool):
                raise ValueError("sparse-Q requires the DSV4 cache pool")

            reusable = mla_off_context.reuse_mask
            online_rows = mla_off_context.online_local_row_indices
            online_rows_cpu = mla_off_context.online_local_row_indices_cpu
            self._mla_off_validate_restore_row_metadata(
                q=q,
                layer_id=layer_id,
                reuse_mask=reusable,
                online_rows=online_rows,
                online_rows_cpu=online_rows_cpu,
                online_rows_certificate=(
                    mla_off_context.online_local_row_indices_certificate
                ),
                restore_layout_certificate=(
                    mla_off_context.restore_layout_certificate
                ),
                controller=mla_off_context.controller,
                reuse_mask_digest=tuple(mla_off_context.reuse_mask_digest),
                reused_row_count=int(mla_off_context.reused_row_count),
                online_row_count=int(mla_off_context.online_local_row_count),
            )
        except Exception as error:
            local_ok = False
            local_error = str(error)
        transaction = getattr(
            mla_off_context, "_redknot_forward_composite_transaction", None
        )
        if isinstance(transaction, _RedKnotForwardCompositeTransaction) and (
            transaction.coordinator is not None
        ):
            if not local_ok:
                error = RuntimeError(
                    local_error or "forward sparse-Q backend preflight failed"
                )
                mla_off_context.record_pipeline_error(error)
                transaction.coordinator.record_pipeline_failure(
                    layer_id=int(mla_off_context.layer_id),
                    stage="sparse_q_backend_preflight",
                    error=error,
                )
            else:
                mla_off_context.sparse_q_backend_preflight_complete = True
            # The forward certificate is already irreversible.  Never add a
            # per-layer TP vote or attempt dense fallback here; a local error
            # is carried through shape-safe execution to the fixed final vote.
            return True
        ready = self._mla_off_vote_restore_ready(local_ok, device)
        if not ready:
            self._mla_off_log_failure(
                "sparse_q_backend_preflight_failed",
                local_error or "another attention-TP rank rejected backend metadata",
            )
            return False
        mla_off_context.sparse_q_backend_preflight_complete = True
        return True

    @staticmethod
    def _redknot_contiguous_row_runs(
        rows: Tuple[int, ...]
    ) -> Tuple[Tuple[int, int], ...]:
        if not rows:
            return ()
        if rows != tuple(sorted(set(rows))) or rows[0] < 0:
            raise ValueError("dirty compressor rows must be sorted and unique")
        runs = []
        begin = previous = int(rows[0])
        for row in rows[1:]:
            row = int(row)
            if row != previous + 1:
                runs.append((begin, previous + 1))
                begin = row
            previous = row
        runs.append((begin, previous + 1))
        return tuple(runs)

    @staticmethod
    def _redknot_single_cpu_int_sequence(value, *, name: str) -> Tuple[int, ...]:
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.device.type != "cpu":
                raise ValueError(f"{name} must be a rank-1 CPU tensor")
            values = tuple(int(item) for item in value.tolist())
        elif isinstance(value, (tuple, list)):
            if any(type(item) is not int for item in value):
                raise TypeError(f"{name} contains a non-integer value")
            values = tuple(value)
        else:
            raise TypeError(f"{name} is not a scheduler CPU sequence")
        return values

    def _redknot_qualification_continuation_geometry(
        self, *, resources, forward_batch: ForwardBatch
    ):
        """Return one exact qualification continuation or ``None``.

        Ordinary qualification remains no-radix.  The sole exception is the
        explicit first-document-prefix contract: a seed request publishes a
        physical terminal-state receipt, and a consumer must hit and bind that
        exact 8K radix prefix before any later document may restore.
        """

        if not bool(getattr(self, "_redknot_mla_off_qualification_only", False)):
            return None
        if int(getattr(forward_batch, "batch_size", 0)) != 1:
            return None
        geometry = getattr(resources, "geometry", None)
        if geometry is None:
            raise ValueError("qualification continuation lost composite geometry")
        geometry.validate_cached()
        if len(geometry.requests) != 1 or len(geometry.validated_plans) != 1:
            raise ValueError("qualification continuation is not single-request")
        request = geometry.requests[0]
        plan = geometry.validated_plans[0]
        radix_role = (
            plan.get("radix_prefix_role") if isinstance(plan, Mapping) else None
        )
        radix_enabled = not bool(
            getattr(self, "_redknot_disable_radix_cache", False)
        )
        if radix_enabled and (
            not bool(getattr(self, "_redknot_mla_prefix_materialization", False))
            or radix_role not in ("seed", "consume")
        ):
            return None
        if not radix_enabled and radix_role is not None:
            raise ValueError("radix-prefix plan reached a no-radix server")
        if (
            not isinstance(plan, Mapping)
            or plan.get("mode") != "restore"
            or plan.get("reuse_mla_off") is not True
            or plan.get(_MLA_OFF_QUALIFICATION_PLAN_FIELD) is not True
            or str(geometry.diagnostic_ablation) != "full"
            or not bool(getattr(request, "reusable", False))
        ):
            raise ValueError("qualification continuation plan is not pure full MLA")
        positions = tuple(int(value) for value in request.logical_positions)
        if not positions or any(
            right != left + 1 for left, right in zip(positions, positions[1:])
        ):
            raise ValueError("qualification continuation positions are not contiguous")
        start = int(positions[0])
        end = int(positions[-1]) + 1
        total = geometry.scheduler_totals[0]
        extent = geometry.scheduler_extents[0]
        if (
            type(total) is not int
            or type(extent) is not int
            or int(extent) != end
            or int(total) != int(plan.get("total_tokens", -1))
            or end > int(total)
        ):
            raise ValueError("qualification continuation scheduler extent changed")
        extend_lens = self._redknot_single_cpu_int_sequence(
            getattr(forward_batch, "extend_seq_lens_cpu", None),
            name="extend_seq_lens_cpu",
        )
        seq_lens = self._redknot_single_cpu_int_sequence(
            getattr(forward_batch, "seq_lens_cpu", None),
            name="seq_lens_cpu",
        )
        if extend_lens != (len(positions),) or seq_lens != (end,):
            raise ValueError("qualification continuation ragged extent changed")
        derived_prefix = end - len(positions)
        prefix_source = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if prefix_source is not None:
            prefix_lens = self._redknot_single_cpu_int_sequence(
                prefix_source, name="extend_prefix_lens_cpu"
            )
            if prefix_lens != (derived_prefix,):
                raise ValueError("qualification continuation prefix extent changed")
        if start != derived_prefix:
            raise ValueError("qualification continuation position/prefix mismatch")
        raw_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if isinstance(raw_pool_indices, torch.Tensor):
            if raw_pool_indices.ndim != 1:
                raise ValueError("req_pool_indices must be rank-1")
            pool_indices = tuple(
                int(value)
                for value in raw_pool_indices.detach()
                .to(device="cpu", dtype=torch.long)
                .tolist()
            )
        elif isinstance(raw_pool_indices, (tuple, list)):
            if any(type(value) is not int for value in raw_pool_indices):
                raise TypeError("req_pool_indices contains a non-integer value")
            pool_indices = tuple(raw_pool_indices)
        else:
            raise TypeError("req_pool_indices is not an index sequence")
        if len(pool_indices) != 1 or pool_indices[0] < 0:
            raise ValueError("qualification continuation request-pool binding changed")
        key = (
            str(request.request_token),
            int(pool_indices[0]),
            int(total),
            str(geometry.source.plan_digest),
        )
        return geometry, request, start, end, key

    def _redknot_live_prefix_authorizations(
        self, *, resources, forward_batch: ForwardBatch
    ) -> Tuple[object, ...]:
        cached = getattr(resources, "_qualification_prefix_authorizations", None)
        if cached is not None:
            return tuple(cached)
        continuation = self._redknot_qualification_continuation_geometry(
            resources=resources, forward_batch=forward_batch
        )
        authorizations = ()
        if continuation is not None:
            geometry, request, start, _end, key = continuation
            plan = geometry.validated_plans[0]
            if start > 0:
                receipt = self._redknot_qualification_prefix_receipts.get(key)
                if (
                    not isinstance(
                        receipt, _RedKnotLivePrefixContinuationReceipt
                    )
                    and isinstance(plan, Mapping)
                    and plan.get("radix_prefix_role") == "consume"
                    and int(plan.get("radix_prefix_tokens", -1)) == int(start)
                ):
                    radix_key = (
                        str(plan.get("radix_prefix_receipt_key", "")),
                        str(plan.get("radix_prefix_input_hash", "")),
                        int(start),
                    )
                    receipt = self._redknot_radix_prefix_receipts.get(radix_key)
                if (
                    not isinstance(
                        receipt, _RedKnotLivePrefixContinuationReceipt
                    )
                    or int(receipt.completed_extent) != int(start)
                ):
                    raise ValueError(
                        "qualification live-prefix state has no completed prior "
                        "microforward receipt"
                    )
                from sglang.srt.layers.attention.redknot.dsv4_shared_latent_sglang import (
                    LivePrefixStateContinuationAuthorization,
                )

                authorization_token = "sha256:" + hashlib.sha256(
                    json.dumps(
                        {
                            "forward_certificate": receipt.forward_token,
                            "terminal_state_digest": (
                                receipt.terminal_state_digest
                            ),
                            "terminal_state_slots": (
                                receipt.terminal_state_slots
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                authorizations = (
                    LivePrefixStateContinuationAuthorization(
                        request_index=int(request.request_index),
                        flat_row_offset=int(request.flat_row_start),
                        seq_len_before=int(start),
                        row_count=int(request.row_count),
                        prior_forward_token=authorization_token,
                        terminal_state_slots=receipt.terminal_state_slots,
                    ),
                )
            resources._qualification_prefix_authorizations = authorizations
            resources._qualification_prefix_geometry = (
                str(geometry.benchmark_request_id),
                int(start),
                int(_end),
                bool(authorizations),
            )
        return tuple(authorizations)

    def _redknot_prepare_qualification_prefix_receipt(
        self,
        *,
        resources,
        forward_batch: ForwardBatch,
        transaction: _RedKnotForwardCompositeTransaction,
    ):
        continuation = self._redknot_qualification_continuation_geometry(
            resources=resources, forward_batch=forward_batch
        )
        if continuation is None:
            return None
        geometry, request, _start, end, key = continuation
        # No later prefill microforward can consume a final-request carry.
        # In particular, an all-dirty query suffix need not (and generally
        # does not) have an artifact terminal-state scatter.
        if int(end) == int(key[2]):
            return None
        if transaction.restore_batch_receipt is None:
            raise RuntimeError(
                "qualification prefix receipt has no completed restore proof"
            )
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            DOMAIN_C128_ATTENTION_STATE,
            DOMAIN_C4_ATTENTION_STATE,
            DOMAIN_INDEXER_STATE,
        )

        terminal_row = int(request.row_count) - 1
        expected_layers = tuple(
            int(value) for value in resources.expected_layer_ids
        )
        if tuple(sorted(transaction.prepared_layers)) != expected_layers:
            raise RuntimeError(
                "qualification prefix receipt lost a prepared reusable layer"
            )
        pending_materialization = []
        terminal_slot_views = []
        for layer_id in expected_layers:
            prepared_layer = transaction.prepared_layers[layer_id]
            adapter = prepared_layer.restore_adapter
            ratio = int(adapter.compress_ratio)
            required_domains = (
                (DOMAIN_C4_ATTENTION_STATE, DOMAIN_INDEXER_STATE)
                if ratio == 4
                else (DOMAIN_C128_ATTENTION_STATE,)
            )
            request_restores = tuple(prepared_layer.validated_restores)
            if (
                len(request_restores) != 1
                or int(request_restores[0].request_index) != 0
            ):
                raise RuntimeError(
                    "qualification prefix receipt is not single-request"
                )
            schedule = request_restores[0].validated.prepared.schedule
            schedule_positions = tuple(int(value) for value in schedule.positions)
            if (
                schedule_positions != tuple(request.logical_positions)
                or schedule_positions[-1] + 1 != int(end)
            ):
                raise RuntimeError(
                    "qualification prefix terminal schedule changed positions"
                )
            arena = tuple(int(value) for value in schedule.index_arena)
            restored_by_domain = {}
            for operation in schedule.operations_for_layer(layer_id):
                restored_by_domain.setdefault(str(operation.domain), set()).update(
                    arena[
                        operation.output_rows.begin : operation.output_rows.end
                    ]
                )
            for domain in required_domains:
                if terminal_row not in restored_by_domain.get(domain, set()):
                    raise RuntimeError(
                        "qualification prefix terminal compressor state was not "
                        f"materialized: layer={layer_id} domain={domain} "
                        f"terminal_row={terminal_row}"
                    )
            domain_proofs = []
            for domain in required_domains:
                slot_vector = adapter.target_slots.get((domain, layer_id))
                if (
                    not isinstance(slot_vector, torch.Tensor)
                    or slot_vector.ndim != 1
                    or int(slot_vector.numel()) != int(request.row_count)
                ):
                    raise RuntimeError(
                        "qualification prefix terminal state slot vector changed"
                    )
                slot_ordinal = len(terminal_slot_views)
                terminal_slot_views.append(slot_vector[terminal_row])
                domain_proofs.append(
                    (
                        str(domain),
                        tuple(sorted(restored_by_domain[str(domain)])),
                        slot_ordinal,
                    )
                )
            pending_materialization.append(
                (
                    int(layer_id),
                    ratio,
                    str(schedule.digest),
                    tuple(domain_proofs),
                    terminal_row,
                )
            )
        if not terminal_slot_views:
            raise RuntimeError(
                "qualification prefix terminal state proof has no slots"
            )
        # All layer/domain slot tensors are already bound by the prepared
        # restore certificate.  Copy their terminal scalars in one D2H instead
        # of serializing the CUDA stream once per domain (55 times for the
        # 19xC128 + 18xC4 DeepSeek-V4 reusable layers).
        terminal_slots = tuple(
            int(value)
            for value in torch.stack(tuple(terminal_slot_views), dim=0)
            .detach()
            .to(device="cpu", dtype=torch.long)
            .tolist()
        )
        if len(terminal_slots) != len(terminal_slot_views) or any(
            slot < 0 for slot in terminal_slots
        ):
            raise RuntimeError(
                "qualification prefix terminal state slot is negative or incomplete"
            )
        terminal_materialization = []
        for (
            layer_id,
            ratio,
            schedule_digest,
            pending_domain_proofs,
            terminal_row,
        ) in pending_materialization:
            domain_proofs = tuple(
                (domain, restored_rows, terminal_slots[slot_ordinal])
                for domain, restored_rows, slot_ordinal in pending_domain_proofs
            )
            terminal_materialization.append(
                (
                    layer_id,
                    ratio,
                    schedule_digest,
                    domain_proofs,
                    terminal_row,
                )
            )
        return geometry, request, int(end), key, tuple(terminal_materialization)

    def _redknot_publish_qualification_prefix_receipt(
        self, *, prepared_proof, final_certificate
    ) -> None:
        if prepared_proof is None:
            return
        if final_certificate is None or not isinstance(
            getattr(final_certificate, "digest", None), str
        ):
            raise RuntimeError(
                "qualification prefix receipt has no final execution certificate"
            )
        geometry, request, end, key, terminal_materialization = prepared_proof
        terminal_state_slots = tuple(
            sorted(
                (
                    int(layer_id),
                    str(domain),
                    int(slot),
                )
                for (
                    layer_id,
                    _ratio,
                    _schedule_digest,
                    domain_proofs,
                    _terminal_row,
                ) in terminal_materialization
                for domain, _restored_rows, slot in domain_proofs
            )
        )
        terminal_state_digest = "sha256:" + hashlib.sha256(
            json.dumps(
                {
                    "final_certificate": str(final_certificate.digest),
                    "request_token": str(request.request_token),
                    "request_pool_index": int(key[1]),
                    "completed_extent": int(end),
                    "layers": terminal_materialization,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        receipt = _RedKnotLivePrefixContinuationReceipt(
            request_token=str(request.request_token),
            request_pool_index=int(key[1]),
            total_tokens=int(key[2]),
            completed_extent=int(end),
            plan_digest=str(key[3]),
            forward_token=str(final_certificate.digest),
            terminal_state_digest=terminal_state_digest,
            terminal_state_slots=terminal_state_slots,
        )
        receipts = self._redknot_qualification_prefix_receipts
        receipts[key] = receipt
        while len(receipts) > 64:
            del receipts[next(iter(receipts))]
        plan = geometry.validated_plans[0]
        if (
            isinstance(plan, Mapping)
            and plan.get("radix_prefix_role") == "seed"
            and int(plan.get("radix_prefix_tokens", -1)) == int(end)
        ):
            radix_key = (
                str(plan.get("radix_prefix_receipt_key", "")),
                str(plan.get("radix_prefix_input_hash", "")),
                int(end),
            )
            radix_receipts = self._redknot_radix_prefix_receipts
            radix_receipts[radix_key] = receipt
            while len(radix_receipts) > 8:
                del radix_receipts[next(iter(radix_receipts))]
            logger.info(
                "REDKNOT_MLA_OFF_RADIX_PREFIX_RECEIPT request_id=%s "
                "position_end=%d receipt_key=%s forward_id=%s",
                str(geometry.benchmark_request_id),
                int(end),
                str(radix_key[0]),
                str(receipt.forward_token),
            )
        logger.info(
            "REDKNOT_MLA_OFF_PREFIX_RECEIPT request_id=%s position_end=%d "
            "total_tokens=%d req_pool=%d forward_id=%s",
            str(geometry.benchmark_request_id),
            int(end),
            int(receipt.total_tokens),
            int(receipt.request_pool_index),
            str(receipt.forward_token),
        )

    def _redknot_record_qualification_prefix_receipt(
        self, *, prepared_proof, final_certificate
    ) -> None:
        """Publish after final without invalidating an already completed forward."""

        if prepared_proof is None:
            return
        key = None
        try:
            key = prepared_proof[3]
            self._redknot_publish_qualification_prefix_receipt(
                prepared_proof=prepared_proof,
                final_certificate=final_certificate,
            )
        except BaseException as publish_error:
            # A missing rank-local continuation receipt makes the next
            # microforward fail closed before its TP prepare commit.  It must
            # not retroactively relabel a real ForwardExecutionCertificate as
            # a failed final rendezvous.
            try:
                if key is not None:
                    self._redknot_qualification_prefix_receipts.pop(key, None)
            except BaseException:
                pass
            try:
                logger.error(
                    "RedKnot qualification prefix receipt unavailable after final: "
                    "%s: %s",
                    type(publish_error).__name__,
                    publish_error,
                )
            except BaseException:
                pass

    def _prepare_mla_off_shared_layer(
        self,
        *,
        mla_off_context,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compress_ratio: int,
        freqs_cis: torch.Tensor,
        slot_bounds_batch=None,
    ) -> _RedKnotPreparedSharedLayer:
        """Preflight every clean target and dirty island without mutation."""

        from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
            ATTENTION_COMPRESSOR_STATE,
            C4,
            C128,
            INDEXER,
            INDEXER_COMPRESSOR_STATE,
            SWA,
        )
        from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
            LayerCacheDomainPreflight,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            DOMAIN_C128,
            DOMAIN_C128_ATTENTION_STATE,
            DOMAIN_C4,
            DOMAIN_C4_ATTENTION_STATE,
            DOMAIN_INDEXER,
            DOMAIN_INDEXER_STATE,
            DOMAIN_SWA,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_sglang import (
            build_layer_restore_adapter,
        )

        layer_id = int(layer_id)
        ratio = int(compress_ratio)
        if ratio not in (4, 128):
            raise ValueError("shared-latent middle layer must be C4 or C128")
        if int(getattr(mla_off_context, "layer_id", -1)) != layer_id:
            raise ValueError("shared restore context belongs to another layer")
        if positions.ndim != 1 or positions.dtype != torch.long:
            raise ValueError("shared restore positions must be a flat int64 tensor")
        q_rows = int(positions.numel())
        full_loc = forward_batch.out_cache_loc
        if (
            not isinstance(full_loc, torch.Tensor)
            or full_loc.ndim != 1
            or int(full_loc.numel()) != q_rows
            or full_loc.device != positions.device
        ):
            raise ValueError("shared restore full-cache locations are incomplete")
        self._maybe_upgrade_forward_metadata()
        core = self.forward_metadata.core_metadata
        if not isinstance(core, DSV4AttnMetadata):
            raise ValueError("shared restore requires materialized DSV4 metadata")
        compressed_loc = core.c4_out_loc if ratio == 4 else core.c128_out_loc
        if (
            not isinstance(compressed_loc, torch.Tensor)
            or compressed_loc.ndim != 1
            or int(compressed_loc.numel()) != q_rows
            or compressed_loc.device != positions.device
        ):
            raise ValueError("compressed output locations do not span the forward")

        resources = getattr(
            mla_off_context, "_redknot_composite_forward_resources", None
        )
        if resources is None:
            raise ValueError("shared restore has no forward-scoped resource lease")
        resources.validate()
        forward_token = str(resources.forward_id)
        seq_lens = tuple(int(value) for value in forward_batch.seq_lens_cpu)
        extend_lens = tuple(
            int(value) for value in forward_batch.extend_seq_lens_cpu
        )
        if (
            len(seq_lens) != int(forward_batch.batch_size)
            or len(extend_lens) != int(forward_batch.batch_size)
            or sum(extend_lens) != q_rows
        ):
            raise ValueError("shared dirty builders lost ragged batch geometry")

        validated_restores = []
        raw_dirty_worksets = []
        restored_counts = defaultdict(int)
        dirty_counts = defaultdict(int)
        target_by_component = {}
        schedule_tokens = []
        request_geometry = []
        domain_to_component = {
            DOMAIN_SWA: SWA,
            DOMAIN_C4: C4,
            DOMAIN_C128: C128,
            DOMAIN_INDEXER: INDEXER,
            DOMAIN_C4_ATTENTION_STATE: ATTENTION_COMPRESSOR_STATE,
            DOMAIN_C128_ATTENTION_STATE: ATTENTION_COMPRESSOR_STATE,
            DOMAIN_INDEXER_STATE: INDEXER_COMPRESSOR_STATE,
        }

        # First build one flat, request-ragged dirty geometry.  The state slot
        # field is deliberately empty here: logical block numbers are not
        # physical compressor-state addresses and must never be guessed by the
        # backend.  The SGLang adapter resolves them from full-cache slots (or
        # req_to_token for a prefix boundary) in one canonical pass below.
        for state in tuple(resources.shared_states):
            state.validate()
            request_index = int(state.request_index)
            offset = int(state.flat_row_offset)
            row_count = int(state.row_count)
            if row_count != extend_lens[request_index]:
                raise ValueError("shared request state differs from extend length")
            if bool(getattr(state, "reusable", True)):
                schedule_tokens.append(str(state.schedule.digest))
                dirty = state.schedule.dirty_for_layer(layer_id)
                dirty_rows = tuple(int(value) for value in dirty.input_rows)
                scheduled_compressed_blocks = int(dirty.compressed_blocks.count)
                scheduled_indexer_blocks = int(dirty.indexer_blocks.count)
            else:
                # Continuous batching may mix restore requests with ordinary
                # requests.  A dense request owns no shared artifact or pin;
                # all of its rows are deliberately part of the dirty domain
                # and therefore flow through the same packed builders without
                # serializing or splitting the GPU batch.
                dirty_rows = tuple(range(row_count))
                schedule_tokens.append(
                    f"dense-request:{request_index}:rows:{row_count}"
                )
                scheduled_compressed_blocks = None
                scheduled_indexer_blocks = None
            runs = self._redknot_contiguous_row_runs(dirty_rows)
            islands = []
            seq_before = seq_lens[request_index] - row_count
            for row_begin, row_end in runs:
                token_begin = seq_before + row_begin
                token_end = seq_before + row_end
                completion_rows = tuple(
                    offset + relative
                    for relative, token in enumerate(
                        range(token_begin, token_end), start=row_begin
                    )
                    if (token + 1) % ratio == 0
                )
                island = {
                    "flat_begin": offset + row_begin,
                    "flat_end": offset + row_end,
                    "request_row_begin": row_begin,
                    "request_row_end": row_end,
                    "token_begin": token_begin,
                    "token_end": token_end,
                    # The adapter treats this field as untrusted and replaces
                    # it with full->SWA->state physical group slots.
                    "state_slot_indices": (),
                    "completion_output_rows": completion_rows,
                }
                islands.append(island)
            raw_dirty_worksets.append(
                {
                    "request_index": request_index,
                    "flat_row_offset": offset,
                    "row_count": row_count,
                    "seq_len_before": seq_before,
                    "islands": tuple(islands),
                }
            )
            exact_dirty_blocks = sum(
                len(island["completion_output_rows"]) for island in islands
            )
            if scheduled_compressed_blocks is not None and (
                exact_dirty_blocks != scheduled_compressed_blocks
                or (
                    ratio == 4
                    and exact_dirty_blocks != scheduled_indexer_blocks
                )
            ):
                raise ValueError(
                    "shared dirty schedule and physical compressor completions differ"
                )
            dirty_counts[SWA] += len(dirty_rows)
            dirty_counts[C4 if ratio == 4 else C128] += int(
                exact_dirty_blocks
            )
            if ratio == 4:
                dirty_counts[INDEXER] += int(exact_dirty_blocks)
            dirty_counts[ATTENTION_COMPRESSOR_STATE] += len(islands)
            if ratio == 4:
                dirty_counts[INDEXER_COMPRESSOR_STATE] += len(islands)
            request_geometry.append((state, offset, row_count))

        restore_token = "sha256:" + hashlib.sha256(
            repr((forward_token, layer_id, tuple(schedule_tokens))).encode("utf-8")
        ).hexdigest()
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None:
            raise ValueError("shared restore requires request-pool indices")
        dirty_state_slot_certificate = resources.dirty_state_slot_certificate(
            ratio
        )
        live_prefix_state_authorizations = (
            self._redknot_live_prefix_authorizations(
                resources=resources, forward_batch=forward_batch
            )
        )
        adapter = build_layer_restore_adapter(
            token_to_kv_pool=self.token_to_kv_pool,
            layer_id=layer_id,
            compress_ratio=ratio,
            full_cache_slots=full_loc,
            compressed_slots_by_output_row=compressed_loc,
            freqs_cis=freqs_cis,
            dirty_worksets=tuple(raw_dirty_worksets),
            forward_token=forward_token,
            restore_token=restore_token,
            req_to_token=self.req_to_token,
            request_pool_indices=req_pool_indices,
            dirty_state_slot_certificate=dirty_state_slot_certificate,
            enable_dirty_state_slot_certificate=True,
            live_prefix_state_authorizations=(
                live_prefix_state_authorizations
            ),
            slot_bounds_batch=slot_bounds_batch,
        )
        resolved_slot_certificate = (
            adapter.restore_targets.dirty_state_slot_certificate
        )
        if resolved_slot_certificate is None:
            raise ValueError("shared restore did not certify dirty state slots")
        resources.bind_dirty_state_slot_certificate(
            ratio, resolved_slot_certificate
        )
        dirty_worksets = tuple(adapter.dirty_worksets)
        if layer_id == int(_PURE_HEADSPLIT_OFFLINE_LAYER_IDS[0]):
            for state, offset, row_count in request_geometry:
                if not bool(getattr(state, "reusable", True)):
                    continue
                request_workset = next(
                    workset
                    for workset in dirty_worksets
                    if int(workset.request_index) == int(state.request_index)
                )
                row_zero = next(
                    (
                        island
                        for island in request_workset.islands
                        if int(island.request_row_begin) == 0
                    ),
                    None,
                )
                scheduled_state_rows = tuple(
                    sorted(
                        {
                            int(value)
                            for operation in state.schedule.operations_for_layer(
                                layer_id
                            )
                            if operation.domain
                            in (
                                DOMAIN_C4_ATTENTION_STATE,
                                DOMAIN_C128_ATTENTION_STATE,
                                DOMAIN_INDEXER_STATE,
                            )
                            for value in state.schedule.index_arena[
                                operation.output_rows.begin : operation.output_rows.end
                            ]
                        }
                    )
                )
                logger.info(
                    "REDKNOT_MLA_OFF_STATE_PREFLIGHT request_index=%d "
                    "position_start=%d position_end=%d seq_len=%d extend_len=%d "
                    "seq_before=%d row0_token_begin=%d row0_state_slots=%d "
                    "scheduled_state_rows=%s live_prefix_authorized=%s",
                    int(state.request_index),
                    int(state.schedule.positions[0]),
                    int(state.schedule.positions[-1]) + 1,
                    int(seq_lens[int(state.request_index)]),
                    int(extend_lens[int(state.request_index)]),
                    int(request_workset.seq_len_before),
                    int(row_zero.token_begin) if row_zero is not None else -1,
                    len(row_zero.state_slot_indices) if row_zero is not None else 0,
                    scheduled_state_rows,
                    any(
                        int(item.request_index) == int(state.request_index)
                        for item in live_prefix_state_authorizations
                    ),
                )
        attention_receipt = adapter.restore_targets.restored_state_receipt(
            is_indexer=False,
            restore_token=restore_token,
            forward_token=forward_token,
        )
        indexer_receipt = (
            adapter.restore_targets.restored_state_receipt(
                is_indexer=True,
                restore_token=restore_token,
                forward_token=forward_token,
            )
            if ratio == 4
            else None
        )

        # Clean schedules remain request-local, while the target arena is one
        # persistent flattened batch.  Slice only the slot vectors; every
        # target cache/storage identity remains the global one certified below.
        for state, offset, row_count in request_geometry:
            if not bool(getattr(state, "reusable", True)):
                continue
            store = getattr(state.pin, "_store", None)
            if store is None:
                raise ValueError("shared request pin lost its GPU store")
            request_slots = {
                key: value.narrow(0, offset, row_count)
                for key, value in adapter.target_slots.items()
            }
            request_positions = positions.narrow(0, offset, row_count)
            # Run the same state-receipt provenance check before the forward
            # TP commit that bind_batch_receipt repeats after device restore.
            # Keep the full batch certificate/workset on the adapter; only the
            # target slot vectors and schedule validation are request-scoped.
            # Projecting LayerRestoreTargets would destroy its canonical batch
            # tiling and dirty-state certificate identity.
            validated = adapter.preflight(
                store,
                state.prepared,
                positions=request_positions,
                request_index=int(state.request_index),
                target_slots=MappingProxyType(request_slots),
            )
            validated_restores.append(
                _RedKnotPreparedRequestRestore(
                    request_index=int(state.request_index),
                    store=store,
                    validated=validated,
                    receipt_adapter=adapter,
                )
            )
            for operation in validated.operations:
                component = domain_to_component[operation.domain]
                restored_counts[component] += int(operation.count)

        # Bind every cache view even when this particular forward has no clean
        # operation for one domain (for example, an all-query dense request).
        for domain, component in domain_to_component.items():
            key = (domain, layer_id)
            if key in adapter.targets:
                target_by_component.setdefault(component, adapter.targets[key])

        # Validate the exact duck-typed compressor contracts now, before the
        # composite vote and before a clean cache slot can be overwritten.
        from sglang.srt.layers.attention.dsv4.compressor import (
            preflight_dirty_compressor_geometry,
        )

        common_preflight = dict(
            layer_id=layer_id,
            compress_ratio=ratio,
            total_rows=q_rows,
            batch_size=int(forward_batch.batch_size),
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            dirty_worksets=tuple(dirty_worksets),
            target_loc_rows=q_rows,
            forward_token=forward_token,
        )
        preflight_dirty_compressor_geometry(
            **common_preflight,
            is_indexer=False,
            restored_state=attention_receipt,
        )
        if ratio == 4:
            preflight_dirty_compressor_geometry(
                **common_preflight,
                is_indexer=True,
                restored_state=indexer_receipt,
            )

        components = (
            (SWA, C4, INDEXER, ATTENTION_COMPRESSOR_STATE, INDEXER_COMPRESSOR_STATE)
            if ratio == 4
            else (SWA, C128, ATTENTION_COMPRESSOR_STATE)
        )
        artifact_digest = "sha256:" + hashlib.sha256(
            repr((forward_token, layer_id, tuple(schedule_tokens))).encode("utf-8")
        ).hexdigest()
        cache_preflights = []
        for component in components:
            restored = int(restored_counts[component])
            dirty = int(dirty_counts[component])
            if restored + dirty <= 0:
                # Recurrent state may be carried from the preceding chunk with
                # no state scatter in this exact chunk. It is still a live,
                # preflighted cache dependency and must be in the certificate.
                if component not in (
                    ATTENTION_COMPRESSOR_STATE,
                    INDEXER_COMPRESSOR_STATE,
                ):
                    raise ValueError(
                        f"shared cache component {component} has no coverage"
                    )
                restored = 1
            builder_token = (
                f"shared-layer:{layer_id}:{component}:restore:{restore_token}"
            )
            cache_preflights.append(
                LayerCacheDomainPreflight(
                    component=component,
                    total_units=restored + dirty,
                    restored_units=restored,
                    dirty_units=dirty,
                    artifact_digest=artifact_digest,
                    gpu_view=target_by_component[component],
                    builder_preflight_token=builder_token,
                )
            )
        builder_epoch_token = "sha256:" + hashlib.sha256(
            repr(
                (
                    restore_token,
                    tuple(
                        (item.component, item.total_units)
                        for item in cache_preflights
                    ),
                )
            ).encode("utf-8")
        ).hexdigest()
        return _RedKnotPreparedSharedLayer(
            validated_restores=tuple(validated_restores),
            cache_preflights=tuple(cache_preflights),
            dirty_worksets=tuple(dirty_worksets),
            attention_state_receipt=attention_receipt,
            indexer_state_receipt=indexer_receipt,
            compressed_target_loc=adapter.full_compressed_target_loc,
            indexer_target_loc=adapter.full_indexer_target_loc,
            forward_token=forward_token,
            builder_epoch_token=builder_epoch_token,
            restore_adapter=adapter,
        )

    def _commit_mla_off_forward_layer(
        self,
        *,
        transaction: _RedKnotForwardCompositeTransaction,
        mla_off_context,
        projection,
        local_success: bool,
        layer: Optional[RadixAttention] = None,
        layer_id: Optional[int] = None,
        wo_a_weight: torch.Tensor,
        owned_heads: int,
        groups: int,
        head_dim: int,
        o_lora_rank: int,
    ) -> bool:
        """Issue a local layer receipt under the one forward certificate."""

        coordinator = transaction.coordinator
        if layer_id is None:
            if layer is None:
                raise ValueError("forward layer receipt has no layer identity")
            layer_id = int(layer.layer_id)
        else:
            layer_id = int(layer_id)
        context_error = getattr(mla_off_context, "composite_pipeline_error", None)
        if context_error is not None:
            coordinator.record_pipeline_failure(
                layer_id=layer_id,
                stage="preexisting_forward_failure",
                error=context_error,
            )
        prepared = transaction.prepared_layers[layer_id]
        if transaction.omission_profile == "shared_only":
            if projection is not None or not local_success:
                error = RuntimeError(
                    "shared-only layer received a packed-Q execution"
                )
                mla_off_context.record_pipeline_error(error)
                coordinator.record_pipeline_failure(
                    layer_id=layer_id,
                    stage="shared_only_q_contract",
                    error=error,
                )
                return True
            receipt = coordinator.record_shared_only_layer(
                context=mla_off_context,
                cache_domains=prepared.cache_preflights,
                builder_epoch_token=prepared.builder_epoch_token,
            )
            try:
                if receipt is None:
                    raise RuntimeError(
                        "forward coordinator rejected cache-only layer receipt"
                    )
                if not bool(
                    getattr(mla_off_context, "shared_restore_applied", False)
                ):
                    raise RuntimeError(
                        "batched clean restore has no cache-only receipt"
                    )
                if not coordinator.consume_layer_omitted_slots(
                    context=mla_off_context,
                ):
                    raise RuntimeError(
                        "forward coordinator rejected cache-only layer omissions"
                    )
                mla_off_context.shared_dirty_workset = prepared
                mla_off_context.sparse_q_backend_preflight_complete = False
            except BaseException as error:
                mla_off_context.record_pipeline_error(error)
                coordinator.record_pipeline_failure(
                    layer_id=layer_id,
                    stage="shared_only_receipt_install",
                    error=error,
                )
            self._count("mla_off.forward_shared_only_layer_receipts")
            return True
        if not local_success or projection is None:
            error = RuntimeError("sequential packed-Q layer write failed")
            mla_off_context.record_pipeline_error(error)
            coordinator.record_pipeline_failure(
                layer_id=layer_id,
                stage="sequential_q_write",
                error=error,
            )
            return True
        merge_plan = transaction.merge_plans[layer_id]
        receipt = None
        try:
            if str(merge_plan.kernel_token) != str(
                coordinator.proposal.fused_merge_kernel_token
            ):
                raise RuntimeError(
                    "live headsplit merge kernel differs from forward reservation"
                )
            receipt = coordinator.record_sealed_full_layer(
                context=mla_off_context,
                packed_sparse_q=projection,
                cache_domains=prepared.cache_preflights,
            )
            if receipt is None:
                raise RuntimeError("forward coordinator rejected the live layer receipt")
            if (
                transaction.omission_profile != "zoff_only"
                and not bool(
                    getattr(mla_off_context, "shared_restore_applied", False)
                )
            ):
                raise RuntimeError("batched clean restore has no layer receipt")
            if not coordinator.consume_layer_omitted_slots(
                context=mla_off_context,
            ):
                raise RuntimeError(
                    "forward coordinator rejected full layer omissions"
                )
            mla_off_context.install_headsplit_woa_merge_plan(merge_plan)
            from sglang.srt.layers.attention.redknot.dsv4_sparse_q import (
                issue_sparse_q_commit_certificate,
            )

            certificate = coordinator.session.certificate
            if certificate is None:
                raise RuntimeError("forward coordinator lost its certificate")
            q_certificate = issue_sparse_q_commit_certificate(
                projection.plan,
                generation_id=str(certificate.generation_id),
                collective_token=str(certificate.collective_token),
                projection_token=str(projection.projection_token),
                ready_rank_count=int(self._redknot_tp_size),
            )
            mla_off_context.install_sparse_q_commit(
                plan=projection.plan,
                certificate=q_certificate,
                projection=projection,
                generation_id=str(certificate.generation_id),
                collective_token=str(certificate.collective_token),
                projection_token=str(projection.projection_token),
            )
            mla_off_context.composite_commit_session = coordinator.session
            mla_off_context.composite_certificate = certificate
            mla_off_context.composite_omission_authorization = (
                coordinator.authorization
            )
            mla_off_context.composite_collective_adapter = (
                transaction.collective_adapter
            )
            mla_off_context.shared_dirty_workset = (
                None
                if transaction.omission_profile == "zoff_only"
                else prepared
            )
            mla_off_context.sparse_q_backend_preflight_complete = True
        except BaseException as error:
            mla_off_context.record_pipeline_error(error)
            coordinator.record_pipeline_failure(
                layer_id=layer_id,
                stage="layer_receipt_install",
                error=error,
            )
            mla_off_context.sparse_q_backend_preflight_complete = False
        if receipt is not None:
            self._count("mla_off.sparse_q_committed")
            try:
                logger.info(
                    "REDKNOT_SPARSE_Q_METRIC request_id=%s forward_id=%s "
                    "layer=%d q_rows=%d projected_head_rows=%d "
                    "omitted_head_rows=%d status=committed "
                    "diagnostic_ablation=%s",
                    str(
                        getattr(mla_off_context, "benchmark_request_id", "")
                        or "unattributed"
                    ),
                    str(
                        getattr(mla_off_context, "benchmark_forward_id", "")
                        or "unattributed"
                    ),
                    int(projection.plan.layer_id),
                    int(projection.plan.q_rows),
                    int(projection.plan.projected_head_rows),
                    int(projection.plan.omitted_head_rows),
                    str(
                        getattr(mla_off_context, "diagnostic_ablation", "full")
                        or "full"
                    ),
                )
            except BaseException:
                self._count("mla_off.sparse_q_metric_publish_failures")
        self._count("mla_off.forward_layer_receipts")
        return True

    def commit_mla_off_reuse_layer(
        self,
        *,
        mla_off_context,
        projection,
        local_success: bool,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
        compress_ratio: int,
        freqs_cis: torch.Tensor,
        compressor,
        indexer,
        wo_a_weight: torch.Tensor,
        owned_heads: int,
        groups: int,
        head_dim: int,
        o_lora_rank: int,
        device,
    ) -> bool:
        """Commit packed-Q + all shared cache domains in one TP collective."""

        transaction = getattr(
            mla_off_context, "_redknot_forward_composite_transaction", None
        )
        if isinstance(transaction, _RedKnotForwardCompositeTransaction) and (
            transaction.coordinator is not None
        ):
            return self._commit_mla_off_forward_layer(
                transaction=transaction,
                mla_off_context=mla_off_context,
                projection=projection,
                local_success=local_success,
                layer=layer,
                wo_a_weight=wo_a_weight,
                owned_heads=owned_heads,
                groups=groups,
                head_dim=head_dim,
                o_lora_rank=o_lora_rank,
            )

        del compressor, indexer  # module identity is checked by dirty execution.
        from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
            begin_layer_composite_commit,
            close_composite_forward_resources,
        )
        from sglang.srt.layers.attention.redknot.dsv4_fused_z_merge import (
            preflight_persistent_headsplit_woa_merge,
        )

        adapter = _RedKnotCompositeCollectiveAdapter(self, device)
        layer_id = int(layer.layer_id)
        prepared = None
        builder = None
        headsplit_merge_plan = None
        restore_receipts = ()
        local_precommit_error = None
        try:
            if not local_success or projection is None:
                raise RuntimeError("packed sparse-Q proposal failed before commit")
            if not bool(
                mla_off_context is not None
                and getattr(mla_off_context, "is_restore", False)
                and int(getattr(mla_off_context, "reused_row_count", 0)) > 0
            ):
                raise ValueError(
                    "composite commit requires an active restore context"
                )
            if layer_id != int(mla_off_context.layer_id):
                raise ValueError("composite commit layer/context mismatch")
            prepared = self._prepare_mla_off_shared_layer(
                mla_off_context=mla_off_context,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                compress_ratio=int(compress_ratio),
                freqs_cis=freqs_cis,
            )
            # This proof is created before begin_layer_composite_commit can
            # authorize omission.  It binds every post-attention pointer and
            # rejects layouts that cannot execute the two fixed Triton kernels.
            headsplit_merge_plan = preflight_persistent_headsplit_woa_merge(
                projection_plan=(
                    mla_off_context.validate_persistent_projection_commit()
                ),
                dirty_rows=mla_off_context.online_local_row_indices,
                dirty_rows_cpu=mla_off_context.online_local_row_indices_cpu,
                local_head_axes=mla_off_context.local_head_axes,
                wo_a_weight=wo_a_weight,
                owned_heads=int(owned_heads),
                groups=int(groups),
                head_dim=int(head_dim),
                o_lora_rank=int(o_lora_rank),
            )
            # Restore before freezing the composite proposal: the fused
            # scatter legitimately advances mutable cache tensor versions, so
            # the proposal and its live-view guard must bind the post-restore
            # identities.  This is still before the TP readiness vote and
            # certificate; any target/kernel failure therefore makes every
            # rank choose the ordinary all-row KV/compressor fallback, which
            # overwrites these provisional slots.
            restore_receipts = tuple(
                store.restore_clean(validated)
                for item in prepared.validated_restores
                for store, validated in ((item.store, item.validated),)
            )
            builder = begin_layer_composite_commit(
                mla_off_context,
                cache_domains=prepared.cache_preflights,
                packed_sparse_q=projection,
                forward_ordinal=layer_id,
                builder_epoch_token=prepared.builder_epoch_token,
                generation_id=prepared.forward_token,
                model_hash=self._redknot_mla_off_model_hash,
                policy_hash=self._redknot_mla_off_policy_hash,
                fused_merge_kernel_token=headsplit_merge_plan.kernel_token,
            )
        except BaseException as error:
            local_precommit_error = error

        if not self._mla_off_vote_restore_ready(
            local_precommit_error is None, device
        ):
            if mla_off_context is not None:
                mla_off_context.composite_dense_fallback = True
            forward_batch._redknot_mla_off_disabled = True
            close_composite_forward_resources(forward_batch)
            self._count("mla_off.composite_precommit_fallback")
            self._mla_off_log_failure(
                "composite_precommit_failed",
                str(local_precommit_error)
                if local_precommit_error is not None
                else "another attention-TP rank rejected packed Q/cache restore",
            )
            return False
        if (
            local_precommit_error is not None
            or prepared is None
            or builder is None
            or headsplit_merge_plan is None
        ):
            raise AssertionError("precommit consensus accepted an invalid proposal")

        result = builder.commit(adapter, ready=True)
        if not result.committed:
            dense_fallback = getattr(result.outcome, "dense_fallback", None)
            reason_code = str(
                getattr(
                    dense_fallback,
                    "reason_code",
                    "composite_commit_not_committed",
                )
            )
            reason_code = "".join(
                char if char.isalnum() or char in "_-" else "_"
                for char in reason_code
            )[:64] or "composite_commit_not_committed"
            local_preflight_error = str(
                getattr(builder, "local_preflight_error", "") or ""
            )
            local_preflight_error_type = (
                local_preflight_error.partition(":")[0].strip() or "none"
            )
            if not local_preflight_error_type.replace("_", "").isalnum():
                local_preflight_error_type = "unknown"
            diagnostic = {
                "schema": "redknot_composite_commit_reject_v1",
                "reason_code": reason_code,
                "layer": layer_id,
                "tp_rank": int(self._redknot_tp_rank),
                "shared_digest": str(result.proposal.shared_digest),
                "execution_identity_digest": str(
                    builder.session._execution_identity_digest
                ),
                "local_preflight_error_type": local_preflight_error_type,
            }
            print(
                "REDKNOT_COMPOSITE_COMMIT_REJECT "
                + json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            mla_off_context.composite_dense_fallback = True
            forward_batch._redknot_mla_off_disabled = True
            close_composite_forward_resources(forward_batch)
            self._count("mla_off.composite_dense_fallback")
            return False

        local_install_error = None
        try:
            certificate = result.outcome.certificate
            authorization = result.omission_authorization
            if certificate is None or authorization is None:
                raise RuntimeError(
                    "composite commit returned no omission certificate"
                )
            # Validate all omission slots while every pinned/tensor identity is
            # still exactly the one included in the commit proposal.  These
            # calls are local checks and do not add TP collectives.
            for slot in result.proposal.omission_slots:
                builder.consume_omitted_slot(adapter, authorization, slot)

            mla_off_context.install_headsplit_woa_merge_plan(
                headsplit_merge_plan
            )

            from sglang.srt.layers.attention.redknot.dsv4_sparse_q import (
                issue_sparse_q_commit_certificate,
            )

            plan = projection.plan
            q_certificate = issue_sparse_q_commit_certificate(
                plan,
                generation_id=str(certificate.generation_id),
                collective_token=str(certificate.collective_token),
                projection_token=str(projection.projection_token),
                ready_rank_count=int(self._redknot_tp_size),
            )
            mla_off_context.install_sparse_q_commit(
                plan=plan,
                certificate=q_certificate,
                projection=projection,
                generation_id=str(certificate.generation_id),
                collective_token=str(certificate.collective_token),
                projection_token=str(projection.projection_token),
            )
            mla_off_context.composite_commit_session = builder.session
            mla_off_context.composite_certificate = certificate
            mla_off_context.composite_omission_authorization = authorization
            mla_off_context.composite_collective_adapter = adapter
            mla_off_context.shared_dirty_workset = prepared
            mla_off_context.sparse_q_backend_preflight_complete = True

            mla_off_context.shared_restore_receipts = tuple(restore_receipts)
            mla_off_context.shared_restore_applied = True
        except BaseException as error:
            local_install_error = error

        # Do not insert another fine-grained TP barrier here.  The composite
        # certificate already puts every rank on the irreversible path.  A
        # local installation/identity failure is carried through the same
        # shape-only pipeline used by dirty-builder/attention failures and is
        # rejected by the single fixed rendezvous immediately before wo_b.
        if local_install_error is not None:
            mla_off_context.composite_pipeline_error = local_install_error
            mla_off_context.sparse_q_backend_preflight_complete = False
            self._count("mla_off.composite_install_local_failures")

        self._count("mla_off.composite_layer_commits")
        installed_restore_receipts = tuple(
            getattr(mla_off_context, "shared_restore_receipts", ())
        )
        self._count(
            "mla_off.shared_device_restore_calls",
            len(installed_restore_receipts),
        )
        shared_restore_operations = sum(
            int(receipt.operation_count)
            for receipt in installed_restore_receipts
        )
        self._count(
            "mla_off.shared_device_restore_operations",
            shared_restore_operations,
        )
        # Retain the pre-audit counter name for downstream diagnostics.
        self._count(
            "mla_off.shared_clean_restore_operations",
            shared_restore_operations,
        )
        self._count(
            "mla_off.shared_device_values_restored",
            sum(
                int(receipt.restored_value_count)
                for receipt in installed_restore_receipts
            ),
        )
        return True

    def forward_mla_off_dirty_cache_builders(
        self,
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        mla_off_context,
        layer_id: int,
        compressor,
        indexer,
    ) -> None:
        """Run only dirty KV-compressor islands, then online Indexer scoring."""

        if not self._mla_off_composite_committed(mla_off_context) or not bool(
            getattr(mla_off_context, "shared_restore_applied", False)
        ):
            raise RuntimeError("dirty builders require committed clean restore")
        prepared = getattr(mla_off_context, "shared_dirty_workset", None)
        if not isinstance(prepared, _RedKnotPreparedSharedLayer):
            raise RuntimeError("dirty builder preflight receipt is absent")
        if int(layer_id) != int(mla_off_context.layer_id):
            raise ValueError("dirty builders received another layer context")
        local_builder_error = None
        try:
            if indexer is not None:
                if prepared.indexer_state_receipt is None:
                    raise RuntimeError("C4 dirty Indexer state receipt is absent")
                if prepared.indexer_target_loc is None:
                    raise RuntimeError("C4 dirty Indexer target locations are absent")
                self.forward_indexer_compressor_dirty(
                    x=x,
                    forward_batch=forward_batch,
                    layer_id=int(layer_id),
                    compressor=indexer.compressor,
                    dirty_worksets=prepared.dirty_worksets,
                    restored_state=prepared.indexer_state_receipt,
                    target_loc=prepared.indexer_target_loc,
                    forward_token=prepared.forward_token,
                )
                # Indexer K/state can be restored offline, but query Q,
                # scoring, and Top-512 selection remain query-dependent.
                indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    attn_backend=self,
                    skip_compressor=True,
                )
            if compressor is not None:
                self.forward_core_compressor_dirty(
                    x=x,
                    forward_batch=forward_batch,
                    layer_id=int(layer_id),
                    compressor=compressor,
                    dirty_worksets=prepared.dirty_worksets,
                    restored_state=prepared.attention_state_receipt,
                    target_loc=prepared.compressed_target_loc,
                    forward_token=prepared.forward_token,
                )
        except BaseException as error:
            local_builder_error = error
        # Do not synchronize between dirty cache construction, headwise
        # attention, and the fused z merge.  They contain no attention-TP
        # collective and form one continuous local GPU pipeline.  A failed rank
        # carries its error to the single post-pipeline rendezvous immediately
        # before wo_b; healthy peers can finish their local kernels and join the
        # same vote instead of blocking at three fine-grained barriers.
        if local_builder_error is not None:
            mla_off_context.composite_pipeline_error = local_builder_error
            self._count("mla_off.composite_dirty_builder_local_failures")
            return
        self._count("mla_off.dirty_only_cache_builder_layers")

    def resolve_mla_off_consumer_stage(
        self,
        *,
        stage: str,
        local_success: bool,
        device,
        mla_off_context=None,
    ) -> Tuple[bool, str]:
        """Coordinate post-attention restore consumers before ``wo_b``.

        A rank-local indexed select or projection merge failure cannot safely
        fall back after attention has already omitted clean local-head rows.
        Every rank therefore enters this one-hot stage vote. A stage mismatch
        or one failed consumer makes every rank abort before the shared
        tensor-parallel ``wo_b`` boundary.
        """

        transaction = getattr(
            mla_off_context, "_redknot_forward_composite_transaction", None
        )
        if isinstance(transaction, _RedKnotForwardCompositeTransaction) and (
            transaction.coordinator is not None
        ):
            if not local_success:
                error = RuntimeError(
                    f"forward composite consumer failed at {stage}"
                )
                mla_off_context.record_pipeline_error(error)
                transaction.coordinator.record_pipeline_failure(
                    layer_id=int(mla_off_context.layer_id),
                    stage=str(stage),
                    error=error,
                )
                return True, "carried_to_forward_final"
            return True, ""

        stages = (
            "indexed_pipeline",
            "projection_compute",
            "projection_merge",
            "pure_headsplit_projection_merge",
            "audit_publish",
            "invalid",
        )
        vote_stage = str(stage)
        if vote_stage not in stages[:-1]:
            vote_stage = "invalid"
        values = [0] * len(stages)
        values[stages.index(vote_stage)] = 1
        values.append(int(bool(local_success)))
        signal = torch.tensor(values, dtype=torch.int32, device=device)
        if self._redknot_tp_size > 1:
            signal = self._mla_off_control_all_reduce(signal)
        totals = tuple(int(value) for value in signal.tolist())
        world = int(self._redknot_tp_size)
        agreed = [
            index for index, count in enumerate(totals[: len(stages)])
            if count == world
        ]
        if len(agreed) != 1 or sum(totals[: len(stages)]) != world:
            return False, "consumer_stage_mismatch"
        if stages[agreed[0]] == "invalid":
            return False, "consumer_stage_invalid"
        if totals[-1] != world:
            return False, "consumer_stage_failed"
        return True, ""

    def _mla_off_resolve_pre_attention_restore(
        self,
        *,
        local_ready: bool,
        local_compact_eligible: bool,
        local_compact_requested: bool = True,
        device,
    ) -> Tuple[bool, bool, bool, str]:
        """Agree on restore readiness, row skipping, and compact ``wo_a``.

        Compact ``wo_a`` is an independently voted A/B choice. Disabling it
        does not disable the certified attention-row skip; it only chooses the
        legacy full-row inverse-RoPE/wo_a consumer on every TP rank.
        """

        world = int(self._redknot_tp_size)
        ready_count = self._mla_off_vote_count(local_ready, device)
        if ready_count != world:
            return False, False, False, "pre_attention_recheck_failed"
        compact_count = self._mla_off_vote_count(
            local_compact_eligible, device
        )
        if compact_count not in (0, world):
            return False, False, False, "compact_eligibility_mismatch"
        requested_count = self._mla_off_vote_count(
            local_compact_requested, device
        )
        if requested_count not in (0, world):
            return False, False, False, "compact_mode_mismatch"
        row_skip_enabled = compact_count == world
        compact_woa_enabled = bool(
            row_skip_enabled and requested_count == world
        )
        return True, row_skip_enabled, compact_woa_enabled, ""

    @staticmethod
    def _mla_off_plan_digest(plan) -> Tuple[int, int]:
        payload = json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:5], 16), int(digest[5:10], 16)

    def _mla_off_preflight_mode(
        self,
        local_mode: str,
        plan,
        device,
        *,
        local_plan_digest=_MLA_OFF_PLAN_DIGEST_UNSET,
    ):
        """Reach consensus before any rank may enter mode-specific collectives.

        The two 20-bit digest moments make plan agreement independently
        observable on every rank: ``n*sum(x^2) == sum(x)^2`` iff all integer
        fingerprints are equal. Two moments keep accidental aliasing small
        without risking int64 overflow at practical TP sizes.
        """

        modes = ("none", "snapshot", "restore", "invalid")
        if local_mode not in modes:
            raise ValueError(f"unknown MLA-off preflight mode {local_mode!r}")
        vote_mode = local_mode
        if local_mode == "none":
            digest_a = digest_b = 0
        else:
            try:
                if local_plan_digest is None:
                    raise ValueError("request plan digest is unavailable")
                if local_plan_digest is _MLA_OFF_PLAN_DIGEST_UNSET:
                    digest_a, digest_b = self._mla_off_plan_digest(plan)
                else:
                    digest_a, digest_b = local_plan_digest
            except Exception:
                # Serialization must not raise before the collective: one bad
                # rank would otherwise strand its peers inside all_reduce.
                vote_mode = "invalid"
                digest_a = digest_b = 0
        values = [0] * len(modes)
        values[modes.index(vote_mode)] = 1
        values.extend(
            (digest_a, digest_a * digest_a, digest_b, digest_b * digest_b)
        )
        signal = torch.tensor(values, dtype=torch.int64, device=device)
        if self._redknot_tp_size > 1:
            signal = self._mla_off_control_all_reduce(signal)
        totals = tuple(int(value) for value in signal.tolist())
        world = int(self._redknot_tp_size)
        agreed_indices = [
            index for index, count in enumerate(totals[: len(modes)])
            if count == world
        ]
        if len(agreed_indices) != 1 or sum(totals[: len(modes)]) != world:
            return None, "request mode differs across attention-TP ranks"
        agreed_mode = modes[agreed_indices[0]]
        digest_consistent = (
            world * totals[5] == totals[4] * totals[4]
            and world * totals[7] == totals[6] * totals[6]
        )
        if agreed_mode != "none" and not digest_consistent:
            return None, "request plan differs across attention-TP ranks"
        if agreed_mode == "invalid":
            return None, "request contains an invalid MLA-off mode/flag combination"
        return agreed_mode, ""

    def _mla_off_resolve_restore_context(
        self,
        local_context,
        *,
        intentional_full_local: bool,
        device,
    ):
        """Resolve readiness, context presence and row policy in one TP vote."""

        ready = local_context is not None or intentional_full_local
        has_context = local_context is not None
        digest_a, digest_b = (
            getattr(local_context, "reuse_mask_digest", (0, 0))
            if has_context
            else (0, 0)
        )
        values = (
            int(ready),
            int(has_context),
            int(digest_a),
            int(digest_a) * int(digest_a),
            int(digest_b),
            int(digest_b) * int(digest_b),
        )
        signal = torch.tensor(values, dtype=torch.int64, device=device)
        if self._redknot_tp_size > 1:
            signal = self._mla_off_control_all_reduce(signal)
        totals = tuple(int(value) for value in signal.tolist())
        ready_count, context_count = totals[:2]
        if ready_count != int(self._redknot_tp_size):
            return None, "restore_not_ready"
        if context_count == 0:
            return None, "intentional_full_local"
        if context_count != int(self._redknot_tp_size):
            return None, "restore_context_mismatch"
        world = int(self._redknot_tp_size)
        digest_consistent = (
            world * totals[3] == totals[2] * totals[2]
            and world * totals[5] == totals[4] * totals[4]
        )
        if not digest_consistent:
            return None, "restore_mask_mismatch"
        return local_context, ""

    def _mla_off_resolve_snapshot_context(self, local_context, *, device):
        """Vote chunk readiness and replicated token/position identity once."""

        ready = local_context is not None
        digest_a, digest_b = (
            getattr(local_context, "input_layout_digest", (0, 0))
            if ready
            else (0, 0)
        )
        values = (
            int(ready),
            int(digest_a),
            int(digest_a) * int(digest_a),
            int(digest_b),
            int(digest_b) * int(digest_b),
        )
        signal = torch.tensor(values, dtype=torch.int64, device=device)
        if self._redknot_tp_size > 1:
            signal = self._mla_off_control_all_reduce(signal)
        totals = tuple(int(value) for value in signal.tolist())
        world = int(self._redknot_tp_size)
        if totals[0] != world:
            return None, "snapshot_not_ready"
        digest_consistent = (
            world * totals[2] == totals[1] * totals[1]
            and world * totals[4] == totals[3] * totals[3]
        )
        if not digest_consistent:
            return None, "snapshot_layout_mismatch"
        return local_context, ""

    def _mla_off_shared_snapshot_service(self, *, length: int, device):
        """Return one persistent GPU-bank + atomic snapshot service."""

        if self._redknot_shared_latent_poisoned:
            raise RuntimeError(
                "shared-latent service is quarantined after an irreversible "
                "snapshot confirmation failure; restart this worker: "
                f"{self._redknot_shared_latent_poison_reason}"
            )
        if not self._redknot_shared_latent_enabled:
            raise RuntimeError("shared-latent snapshot service is disabled")
        from sglang.srt.layers.attention.redknot import (
            dsv4_shared_latent_sglang as shared_sglang,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            SharedLatentGPUStore,
            build_shared_latent_device_layout,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_snapshot_runtime import (
            DSV4SharedSnapshotRuntime,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_snapshot_sglang import (
            DSV4SharedSnapshotSGLangAdapter,
        )
        from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
            get_dsv4_mla_off_controller,
        )

        cpu_spec = shared_sglang.build_runtime_shared_latent_spec(
            token_to_kv_pool=self.token_to_kv_pool,
            model_hash=self._redknot_mla_off_model_hash,
            policy_hash=self._redknot_mla_off_policy_hash,
            segment_length=int(length),
            c4_layer_id=4,
            c128_layer_id=3,
        )
        store_key = (
            f"{cpu_spec.model_hash}:{cpu_spec.policy_hash}:{cpu_spec.length}"
        )
        layout = build_shared_latent_device_layout(cpu_spec)
        store = self._redknot_shared_gpu_stores.get(store_key)
        if store is None:
            max_epochs = int(
                os.environ.get("REDKNOT_SHARED_LATENT_MAX_SEGMENT_EPOCHS", "16")
            )
            if max_epochs <= 0:
                raise ValueError(
                    "REDKNOT_SHARED_LATENT_MAX_SEGMENT_EPOCHS must be positive"
                )
            store = SharedLatentGPUStore(
                layout=layout,
                max_segment_epochs=max_epochs,
                device=torch.device(device),
            )
            self._redknot_shared_gpu_stores[store_key] = store
        elif (
            store.layout.spec_fingerprint != layout.spec_fingerprint
            or store.device != torch.device(device)
        ):
            raise ValueError("shared-latent GPU store compatibility changed")

        service_key = (store_key, str(torch.device(device)))
        service = self._redknot_shared_snapshot_services.get(service_key)
        if service is None:
            runtime = DSV4SharedSnapshotRuntime(
                z_off_controller=get_dsv4_mla_off_controller(),
                cpu_shared_controller=self._redknot_shared_latent_controller,
                gpu_shared_store=store,
                tp_rank=int(self._redknot_tp_rank),
                tp_size=int(self._redknot_tp_size),
            )
            adapter = DSV4SharedSnapshotSGLangAdapter(
                snapshot_runtime=runtime,
                shared_latent_sglang_api=shared_sglang,
            )
            service = {
                "cpu_spec": cpu_spec,
                "store": store,
                "runtime": runtime,
                "adapter": adapter,
                "store_key": store_key,
            }
            self._redknot_shared_snapshot_services[service_key] = service
        elif service["cpu_spec"] != cpu_spec or service["store"] is not store:
            raise ValueError("shared snapshot service compatibility changed")
        return service

    def _mla_off_begin_shared_snapshot(
        self,
        *,
        context,
        plan: Mapping[str, object],
        positions_cpu: torch.Tensor,
        input_ids_cpu: torch.Tensor,
        device,
    ) -> None:
        """Begin/rebind one complete 8K-style atomic segment snapshot."""

        length = int(plan.get("length", 0))
        canonical_start = int(plan.get("canonical_start_pos", 0))
        if canonical_start != 0:
            raise ValueError("shared snapshot requires canonical_start_pos=0")
        if (
            int(positions_cpu.numel()) != length
            or int(input_ids_cpu.numel()) != length
            or not torch.equal(
                positions_cpu,
                torch.arange(length, dtype=torch.long, device="cpu"),
            )
        ):
            # The benchmark's offline producer is exactly one complete 8K
            # segment.  Partial producer chunks need a scheduler-owned full
            # token manifest before begin_segment and are rejected for now.
            raise ValueError(
                "shared snapshot requires one complete contiguous segment forward"
            )
        seg_hash = str(plan.get("seg_hash", ""))
        token_hash = str(plan.get("token_hash", seg_hash))
        generation_id = str(getattr(context, "generation_id", "") or "")
        if not seg_hash or not token_hash or not generation_id:
            raise ValueError("shared snapshot identity is incomplete")
        stage_key = (seg_hash, generation_id)
        service = self._mla_off_shared_snapshot_service(
            length=length, device=device
        )
        entry = self._redknot_shared_snapshot_stages.get(stage_key)
        if entry is None:
            from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
                mla_off_device_expected_bytes,
                mla_off_expected_bytes,
            )

            spec = context.spec
            expected_bytes = mla_off_expected_bytes(
                length=length,
                local_layer_count=len(
                    self._redknot_mla_off_rank_local_layer_ids
                ),
                num_output_groups=spec.num_output_groups,
                o_lora_rank=spec.o_lora_rank,
            )
            # Bind byte accounting to the exact controller already installed
            # in this cached atomic service.  A process-global lookup here
            # would both require another local import and could silently
            # diverge after a controller reset/reconfiguration.
            controller = service["runtime"].z_off_controller
            expected_device_bytes = (
                mla_off_device_expected_bytes(
                    length=length,
                    local_layer_count=len(
                        self._redknot_mla_off_rank_local_layer_ids
                    ),
                    num_output_groups=spec.num_output_groups,
                    o_lora_rank=spec.o_lora_rank,
                )
                if controller.device_cache_enabled
                else 0
            )
            session = service["adapter"].begin_segment(
                forward_token=f"snapshot:{seg_hash}:{generation_id}",
                seg_hash=seg_hash,
                generation_id=generation_id,
                token_hash=token_hash,
                token_ids=tuple(int(value) for value in input_ids_cpu.tolist()),
                model_hash=self._redknot_mla_off_model_hash,
                policy_hash=self._redknot_mla_off_policy_hash,
                cpu_spec=service["cpu_spec"],
                token_to_kv_pool=self.token_to_kv_pool,
                z_off_expected_bytes=expected_bytes,
                z_off_token_positions=positions_cpu,
                z_off_token_ids=input_ids_cpu,
                z_off_expected_device_bytes=expected_device_bytes,
                latent_staging_device=torch.device(device),
            )
            entry = {
                "service": service,
                "session": session,
                "seg_hash": seg_hash,
                "token_hash": token_hash,
                "generation_id": generation_id,
                "length": length,
            }
            self._redknot_shared_snapshot_stages[stage_key] = entry
        elif (
            entry["service"] is not service
            or int(entry["length"]) != length
            or entry["token_hash"] != token_hash
        ):
            raise ValueError("active shared snapshot identity changed")
        context.shared_snapshot_enabled = True
        context.shared_snapshot_stage_key = stage_key
        context.shared_snapshot_adapter = service["adapter"]
        context.shared_snapshot_session = entry["session"]
        context.shared_snapshot_cpu_spec = service["cpu_spec"]

    def capture_mla_off_shared_snapshot_chunk(
        self,
        *,
        mla_off_context,
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
        local_projection: torch.Tensor,
        layer_id: int,
        compress_ratio: int,
        freqs_cis: torch.Tensor,
    ) -> bool:
        """Capture the real z/SWA/compressed/Indexer/state layer generation."""

        if not bool(
            mla_off_context is not None
            and getattr(mla_off_context, "shared_snapshot_enabled", False)
        ):
            return False
        layer_id = int(layer_id)
        ratio = int(compress_ratio)
        if layer_id != int(mla_off_context.layer_id) or ratio not in (4, 128):
            raise ValueError("shared snapshot layer topology changed")
        canonical_swa = getattr(
            mla_off_context, "shared_snapshot_canonical_swa", None
        )
        if not isinstance(canonical_swa, torch.Tensor):
            raise ValueError("shared snapshot canonical SWA rows are absent")
        local_positions = mla_off_context.local_positions_cpu
        length = int(getattr(mla_off_context, "length", 0))
        if (
            not isinstance(local_positions, torch.Tensor)
            or local_positions.device.type != "cpu"
            or local_positions.dtype != torch.long
            or int(local_positions.numel()) != length
            or int(positions.numel()) != length
            or int(local_projection.shape[0]) != length
        ):
            raise ValueError("shared snapshot chunk is not one complete segment")
        row_begin = int(local_positions[0].item())
        if row_begin != 0 or not torch.equal(
            local_positions, torch.arange(length, dtype=torch.long)
        ):
            raise ValueError("shared snapshot rows must be canonical 0..length-1")
        full_loc = forward_batch.out_cache_loc
        if (
            not isinstance(full_loc, torch.Tensor)
            or full_loc.ndim != 1
            or int(full_loc.numel()) != length
            or full_loc.device != positions.device
        ):
            raise ValueError("shared snapshot full-cache slots are incomplete")

        self._maybe_upgrade_forward_metadata()
        core = self.forward_metadata.core_metadata
        if not isinstance(core, DSV4AttnMetadata):
            raise ValueError("shared snapshot requires materialized DSV4 metadata")
        compressed_loc = core.c4_out_loc if ratio == 4 else core.c128_out_loc
        if (
            not isinstance(compressed_loc, torch.Tensor)
            or compressed_loc.ndim != 1
            or int(compressed_loc.numel()) != length
            or compressed_loc.device != positions.device
        ):
            raise ValueError("shared snapshot compressed slots are incomplete")

        completion_rows = torch.arange(
            ratio - 1, length, ratio, device=positions.device, dtype=torch.long
        )
        compressed_slots = compressed_loc.index_select(0, completion_rows)
        compressed_positions = positions.index_select(0, completion_rows)

        # Reuse the production target mapper to derive physical compressor
        # state groups.  Never derive them from logical token/block numbers.
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            DOMAIN_C4_ATTENTION_STATE,
            DOMAIN_C128_ATTENTION_STATE,
            DOMAIN_INDEXER_STATE,
        )
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_sglang import (
            build_layer_restore_targets,
        )

        targets = build_layer_restore_targets(
            token_to_kv_pool=self.token_to_kv_pool,
            layer_id=layer_id,
            compress_ratio=ratio,
            full_cache_slots=full_loc,
            compressed_slots_by_output_row=compressed_loc,
        )
        cpu_spec = mla_off_context.shared_snapshot_cpu_spec
        if int(cpu_spec.layers_by_id[layer_id].compress_ratio) != ratio:
            raise ValueError("live DSV4 compressor topology differs from 0731 spec")
        anchors = tuple(int(value) for value in cpu_spec.checkpoint_anchors) + (
            length,
        )
        anchor_rows = torch.tensor(
            [anchor - 1 for anchor in anchors],
            dtype=torch.long,
            device=positions.device,
        )
        attention_domain = (
            DOMAIN_C4_ATTENTION_STATE
            if ratio == 4
            else DOMAIN_C128_ATTENTION_STATE
        )
        attention_state_slots = targets.target_slots[
            (attention_domain, layer_id)
        ].index_select(0, anchor_rows)
        if int(torch.unique(attention_state_slots).numel()) != len(anchors):
            raise ValueError(
                "shared snapshot attention checkpoint state slots alias"
            )
        indexer_state_slots = None
        if ratio == 4:
            indexer_state_slots = targets.target_slots[
                (DOMAIN_INDEXER_STATE, layer_id)
            ].index_select(0, anchor_rows)
            if (
                int(torch.unique(indexer_state_slots).numel()) != len(anchors)
                or not torch.equal(indexer_state_slots, attention_state_slots)
            ):
                raise ValueError(
                    "shared snapshot Indexer checkpoint state slots alias/differ"
                )

        adapter = mla_off_context.shared_snapshot_adapter
        receipt = adapter.capture_chunk(
            mla_off_context.shared_snapshot_session,
            layer_id=layer_id,
            row_begin=row_begin,
            z_off_spec=mla_off_context.spec,
            local_positions=local_positions,
            local_projection=local_projection,
            latent_chunk={
                "freqs_cis": freqs_cis,
                "full_cache_slots": full_loc,
                "source_positions": positions,
                "canonical_swa_rows": canonical_swa,
                "compressed_row_begin": 0,
                "compressed_slots": compressed_slots,
                "compressed_source_positions": compressed_positions,
                "state_row_begin": 0,
                "attention_state_terminal_slots": attention_state_slots,
                "indexer_state_terminal_slots": indexer_state_slots,
            },
        )
        mla_off_context.shared_snapshot_chunk_receipt = receipt
        return True

    def _mla_off_poison_context_snapshot(
        self, mla_off_context=None, *, request_binding=None
    ) -> None:
        """Make a rolled-back input certificate safely evictable, never reusable."""

        if request_binding is None and mla_off_context is not None:
            request_binding = getattr(
                mla_off_context, "context_snapshot_request_binding", None
            )
        registry = getattr(self, "_redknot_context_token_streams", None)
        if isinstance(request_binding, tuple) and registry is not None:
            registry.poison_snapshot_publication(
                request_binding=request_binding
            )

    def abort_mla_off_snapshot_context(self, mla_off_context) -> None:
        """Rollback the unified transaction or the legacy z-only staging."""

        self._mla_off_poison_context_snapshot(mla_off_context)
        if bool(
            mla_off_context is not None
            and getattr(mla_off_context, "shared_snapshot_enabled", False)
        ):
            adapter = mla_off_context.shared_snapshot_adapter
            session = mla_off_context.shared_snapshot_session
            stage_key = mla_off_context.shared_snapshot_stage_key
            try:
                adapter.rollback(session)
            except BaseException as error:
                reason = (
                    "shared snapshot rollback was incomplete during abort: "
                    f"{type(error).__name__}: {error}"
                )
                self._mla_off_quarantine_shared_latent(reason)
                raise RuntimeError(
                    "RedKnot shared-latent state is unknown; this worker must "
                    "be restarted"
                ) from error
            finally:
                self._redknot_shared_snapshot_stages.pop(stage_key, None)
            return
        if mla_off_context is not None:
            mla_off_context.abort_snapshot()

    def _mla_off_rollback_snapshot_staging(self, snapshot_staging) -> str:
        """Best-effort local rollback that never skips the paired TP vote.

        Snapshot preparation failures are still inside a rank-symmetric
        preflight protocol.  Rollback is an external transaction and can
        itself fail; callers record the returned detail but must continue to
        their fixed collective so a cleanup exception cannot strand peers.
        """

        if snapshot_staging is None:
            return ""
        shared = bool(snapshot_staging[0] == "shared")
        try:
            if shared:
                _, adapter, session, _ = snapshot_staging
                adapter.rollback(session)
            else:
                _, controller, seg_hash, generation_id = snapshot_staging
                controller.abort_staging(seg_hash, generation_id)
        except BaseException as error:
            return f"snapshot rollback failed: {type(error).__name__}: {error}"
        finally:
            if shared:
                self._redknot_shared_snapshot_stages.pop(
                    snapshot_staging[3], None
                )
        return ""

    @staticmethod
    def _mla_off_incomplete_snapshot_rollback_detail(error) -> str:
        """Recover a terminal rollback failure through adapter wrapping."""

        seen = set()
        current = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            name = type(current).__name__
            failures = getattr(current, "failures", None)
            if name.endswith("SnapshotRollbackError") or failures:
                return (
                    "snapshot transaction reported incomplete rollback: "
                    f"{name}: {current}"
                )
            current = getattr(current, "__cause__", None) or getattr(
                current, "__context__", None
            )
        return ""

    def _mla_off_require_snapshot_rollback_consensus(
        self,
        *,
        cleanup_failure: str,
        snapshot_staging,
        device,
        shared_transaction: bool = False,
    ) -> None:
        """Certify rollback health after an already rank-common failure."""

        try:
            cleanup_all = self._mla_off_vote_restore_ready(
                not cleanup_failure, device
            )
        except BaseException as vote_error:
            reason = (
                "snapshot rollback health vote failed: "
                f"{type(vote_error).__name__}: {vote_error}; "
                f"local={cleanup_failure or 'ok'}"
            )
            if shared_transaction or (
                snapshot_staging is not None
                and snapshot_staging[0] == "shared"
            ):
                self._mla_off_quarantine_shared_latent(reason)
            raise RuntimeError(
                "RedKnot cannot certify snapshot rollback; worker restart is "
                "required"
            ) from vote_error
        if cleanup_all:
            return
        reason = cleanup_failure or (
            "another attention-TP rank reported incomplete snapshot rollback"
        )
        if shared_transaction or (
            snapshot_staging is not None and snapshot_staging[0] == "shared"
        ):
            self._mla_off_quarantine_shared_latent(reason)
        else:
            self._redknot_mla_off_enabled = False
            self._mla_off_log_failure("snapshot_rollback_incomplete", reason)
        raise RuntimeError(
            "RedKnot snapshot rollback diverged across TP; worker restart is "
            "required: " + reason
        )

    def _mla_off_maybe_emit_shared_snapshot_audit(
        self, *, adapter, published
    ) -> None:
        """Best-effort post-confirm evidence; never changes serving state."""

        if os.environ.get("REDKNOT_MLA_OFF_METRICS", "0") != "1":
            return
        try:
            self._mla_off_emit_shared_snapshot_audit(
                adapter=adapter, published=published
            )
        except Exception:
            # Confirmation is already irreversible.  Observability must not
            # turn a healthy three-store publication into a serving failure;
            # the benchmark fails closed when this marker is absent.
            try:
                self._count("mla_off.shared_snapshot_audit_publish_failures")
            except BaseException:
                pass

    def _mla_off_emit_shared_snapshot_audit(
        self, *, adapter, published
    ) -> None:
        """Emit one rank-local proof after CPU, z_off and GPU confirmation."""

        runtime = adapter.snapshot_runtime
        runtime_published = published.runtime_published
        identity = runtime_published.prepared.bundle.identity
        seg_hash = str(identity.seg_hash)
        cpu_receipt = runtime_published.cpu_receipt
        gpu_receipt = runtime_published.gpu_receipt
        z_off_receipt = runtime_published.z_off_receipt

        cpu_active = (
            runtime.cpu_shared_controller.get_committed(seg_hash)
            is cpu_receipt.artifact
        )
        cpu_ready = str(cpu_receipt.state) == "confirmed"
        cpu_pending = str(cpu_receipt.state) == "published"

        gpu_epoch = runtime.gpu_shared_store.active_epochs.get(seg_hash)
        gpu_active = gpu_epoch == int(gpu_receipt.commit_epoch)
        gpu_state = str(gpu_receipt.stage.state)
        gpu_ready = gpu_state == "confirmed"
        gpu_pending = gpu_state == "pending_publish"

        z_off_stats = dict(runtime.z_off_controller.snapshot_stats())
        z_off_active = bool(
            z_off_receipt.segment.committed
            and int(z_off_receipt.segment.commit_epoch)
            == int(z_off_receipt.commit_epoch)
        )
        z_off_pending = int(z_off_stats.get("pending_publishes", 0)) != 0
        z_off_ready = z_off_active and not z_off_pending
        payload = {
            "schema": _SHARED_SNAPSHOT_AUDIT_SCHEMA,
            "tp_rank": int(self._redknot_tp_rank),
            "tp_size": int(self._redknot_tp_size),
            "segment_hash": seg_hash,
            "published_total": int(
                self._redknot_runtime_counters.get(
                    "mla_off.shared_snapshot_published", 0
                )
            ),
            "stores": {
                "cpu": {
                    "active": bool(cpu_active),
                    "ready": bool(cpu_ready),
                    "pending": bool(cpu_pending),
                },
                "gpu": {
                    "active": bool(gpu_active),
                    "ready": bool(gpu_ready),
                    "pending": bool(gpu_pending),
                },
                "z_off": {
                    "active": bool(z_off_active),
                    "ready": bool(z_off_ready),
                    "pending": bool(z_off_pending),
                },
            },
        }
        print(
            "REDKNOT_SHARED_SNAPSHOT_AUDIT "
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def finalize_mla_off_snapshot_context(
        self,
        mla_off_context,
        *,
        capture_succeeded: bool,
        device,
    ) -> bool:
        """Publish a snapshot generation only after every TP rank captured it."""

        if mla_off_context is None or not mla_off_context.is_snapshot:
            raise ValueError("MLA-off snapshot finalization needs a snapshot context")
        if bool(getattr(mla_off_context, "shared_snapshot_enabled", False)):
            adapter = mla_off_context.shared_snapshot_adapter
            session = mla_off_context.shared_snapshot_session
            stage_key = mla_off_context.shared_snapshot_stage_key

            def rollback_shared(value=None) -> str:
                self._mla_off_poison_context_snapshot(mla_off_context)
                try:
                    adapter.rollback(session if value is None else value)
                except BaseException as error:
                    detail = (
                        "shared snapshot rollback was incomplete: "
                        f"{type(error).__name__}: {error}"
                    )
                    self._mla_off_log_failure(
                        "shared_snapshot_rollback_failed", detail
                    )
                    return detail
                finally:
                    self._redknot_shared_snapshot_stages.pop(stage_key, None)
                return ""

            def rollback_shared_collectively(value, phase: str) -> None:
                """Resolve cleanup health on a branch already shared by TP."""

                rollback_failure = rollback_shared(value)
                try:
                    rollback_all = self._mla_off_vote_restore_ready(
                        not rollback_failure, device
                    )
                except BaseException as vote_error:
                    reason = (
                        f"{phase} rollback health vote failed: "
                        f"{type(vote_error).__name__}: {vote_error}; "
                        f"local={rollback_failure or 'ok'}"
                    )
                    self._mla_off_quarantine_shared_latent(reason)
                    raise RuntimeError(
                        "RedKnot cannot certify shared snapshot rollback; "
                        "this worker must be restarted"
                    ) from vote_error
                if not rollback_all:
                    reason = rollback_failure or (
                        f"another TP rank reported incomplete {phase} rollback"
                    )
                    self._mla_off_quarantine_shared_latent(reason)
                    raise RuntimeError(
                        "RedKnot shared snapshot rollback diverged across TP; "
                        "this worker must be restarted: " + reason
                    )

            # Offline capture is not on online TTFT/QPS.  Keep a single small
            # readiness barrier per layer so one rank that loses a physical
            # checkpoint slot cannot disappear while peers reach layer 39.
            try:
                layer_ready = self._mla_off_vote_restore_ready(
                    bool(capture_succeeded), device
                )
            except BaseException as vote_error:
                rollback_failure = rollback_shared()
                reason = (
                    "layer-capture TP collective became indeterminate: "
                    f"{type(vote_error).__name__}: {vote_error}; "
                    f"local_rollback={rollback_failure or 'ok'}"
                )
                self._mla_off_quarantine_shared_latent(reason)
                raise RuntimeError(
                    "RedKnot cannot certify the TP snapshot layer state; "
                    "this worker must be restarted"
                ) from vote_error
            if not layer_ready:
                rollback_shared_collectively(
                    None, "layer-capture"
                )
                self._mla_off_log_failure(
                    "shared_snapshot_layer_capture_failed",
                    f"layer {int(mla_off_context.layer_id)} failed on a TP rank",
                )
                return False
            if int(mla_off_context.layer_id) != int(
                _PURE_HEADSPLIT_OFFLINE_LAYER_IDS[-1]
            ):
                return True

            local_prepared = None
            local_prepare_error = None
            try:
                local_prepared = adapter.prepare_local(
                    session,
                    gpu_stream=torch.cuda.current_stream(device=device),
                    gpu_non_blocking=True,
                )
            except BaseException as error:
                local_prepare_error = error
            entry = self._redknot_shared_snapshot_stages.get(stage_key)
            if entry is None:
                local_prepare_error = local_prepare_error or RuntimeError(
                    "shared snapshot stage disappeared before prepare"
                )
                identity = {
                    "seg_hash": stage_key[0],
                    "generation_id": stage_key[1],
                    "token_hash": "missing",
                }
            else:
                identity = entry
            identity_common_digest = "sha256:" + hashlib.sha256(
                json.dumps(
                    {
                        "seg_hash": str(identity["seg_hash"]),
                        "token_hash": str(identity["token_hash"]),
                        "generation_id": str(identity["generation_id"]),
                        "model_hash": self._redknot_mla_off_model_hash,
                        "policy_hash": self._redknot_mla_off_policy_hash,
                        "length": int(getattr(mla_off_context, "length", 0)),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            shared_common_digest = (
                str(local_prepared.shared_latent_digest)
                if local_prepared is not None
                and isinstance(
                    getattr(local_prepared, "shared_latent_digest", None), str
                )
                else identity_common_digest
            )
            try:
                prepared_all, rank_digests, certificate_digest = (
                    self._mla_off_snapshot_digest_vote(
                        local_ready=local_prepare_error is None,
                        local_digest=(
                            str(local_prepared.digest)
                            if local_prepared is not None
                            else ""
                        ),
                        common_digest=shared_common_digest,
                        device=device,
                    )
                )
            except BaseException as vote_error:
                rollback_failure = rollback_shared(local_prepared)
                reason = (
                    "local-prepare TP collective became indeterminate: "
                    f"{type(vote_error).__name__}: {vote_error}; "
                    f"local_rollback={rollback_failure or 'ok'}"
                )
                self._mla_off_quarantine_shared_latent(reason)
                raise RuntimeError(
                    "RedKnot cannot certify the TP snapshot prepare state; "
                    "this worker must be restarted"
                ) from vote_error
            if not prepared_all or local_prepared is None:
                if not prepared_all:
                    layer_digests = getattr(
                        session, "shared_latent_layer_digests", {}
                    )
                    if not isinstance(layer_digests, Mapping):
                        layer_digests = {}
                    domain_digests = getattr(
                        session, "shared_latent_domain_digests", {}
                    )
                    if not isinstance(domain_digests, Mapping):
                        domain_digests = {}
                    shared_aggregate_digest = str(
                        getattr(
                            local_prepared,
                            "shared_latent_digest",
                            "missing",
                        )
                    )
                    layer_prefixes = ",".join(
                        f"{layer_id}:"
                        + str(layer_digests.get(layer_id, "missing"))
                        .removeprefix("sha256:")[:12]
                        for layer_id in _PURE_HEADSPLIT_OFFLINE_LAYER_IDS
                    )
                    domain_prefixes = ";".join(
                        f"{layer_id}:"
                        + ",".join(
                            f"{domain}="
                            + str(digest).removeprefix("sha256:")[:12]
                            for domain, digest in sorted(
                                dict(domain_digests.get(layer_id, {})).items()
                            )
                        )
                        for layer_id in _PURE_HEADSPLIT_OFFLINE_LAYER_IDS
                        if domain_digests.get(layer_id)
                    )
                    self._mla_off_log_failure(
                        "shared_snapshot_prepare_identity",
                        "tp_rank="
                        f"{int(self._redknot_tp_rank)} "
                        "session_digest="
                        f"{str(getattr(session, 'session_digest', 'missing'))} "
                        "shared_aggregate_digest="
                        f"{shared_aggregate_digest} "
                        "shared_layer_digest_prefixes="
                        f"[{layer_prefixes}] "
                        "shared_c4_domain_digest_prefixes="
                        f"[{domain_prefixes}]",
                    )
                rollback_shared_collectively(
                    local_prepared, "local-prepare"
                )
                self._mla_off_log_failure(
                    "shared_snapshot_prepare_failed",
                    str(local_prepare_error)
                    if local_prepare_error is not None
                    else "TP snapshot prepare identity differed",
                )
                return False
            if rank_digests[int(self._redknot_tp_rank)] != str(
                local_prepared.digest
            ):
                rollback_failure = rollback_shared(local_prepared)
                reason = rollback_failure or "snapshot rank-slot digest changed"
                self._mla_off_quarantine_shared_latent(reason)
                raise RuntimeError("snapshot rank-slot digest changed")

            certificate = _RedKnotSnapshotTPCertificate(
                certificate_digest=certificate_digest,
                snapshot_local_prepare_digest=str(local_prepared.digest),
                seg_hash=str(identity["seg_hash"]),
                token_hash=str(identity["token_hash"]),
                model_hash=self._redknot_mla_off_model_hash,
                policy_hash=self._redknot_mla_off_policy_hash,
                generation_id=str(identity["generation_id"]),
                tp_rank=int(self._redknot_tp_rank),
                tp_size=int(self._redknot_tp_size),
                ready_rank_count=int(self._redknot_tp_size),
            )
            prepared_publish = None
            published = None
            confirmation_digest = ""
            publish_error = None
            try:
                prepared_publish = adapter.prepare_publish(
                    local_prepared,
                    tp_certificate=certificate,
                    synchronize_gpu=True,
                )
                published = adapter.publish(prepared_publish)
                confirmation_digest = adapter.validate_confirmation(published)
            except BaseException as error:
                publish_error = error
            try:
                publish_all, _, _ = self._mla_off_snapshot_digest_vote(
                    local_ready=publish_error is None,
                    local_digest=confirmation_digest,
                    common_digest=str(local_prepared.shared_latent_digest),
                    device=device,
                )
            except BaseException as vote_error:
                rollback_failure = rollback_shared(
                    published or prepared_publish or local_prepared
                )
                reason = (
                    "publish TP collective became indeterminate: "
                    f"{type(vote_error).__name__}: {vote_error}; "
                    f"local_rollback={rollback_failure or 'ok'}"
                )
                self._mla_off_quarantine_shared_latent(reason)
                raise RuntimeError(
                    "RedKnot cannot certify the TP snapshot publish state; "
                    "this worker must be restarted"
                ) from vote_error
            if not publish_all or published is None:
                rollback_shared_collectively(
                    published or prepared_publish or local_prepared,
                    "publish",
                )
                self._mla_off_log_failure(
                    "shared_snapshot_publish_failed",
                    str(publish_error)
                    if publish_error is not None
                    else "TP publish/confirmation validation differed",
                )
                return False

            if self._redknot_mla_off_execution_profile in (
                _INDEPENDENT_RELOCATION_PROFILE,
                _COMBINED_ROW_SPARSE_PROFILE,
            ):
                # Independent position-0 documents have no cumulative-prefix
                # publication receipt.  Their shared/cache and z_off payloads
                # are already bound by token hash, generation, model/policy
                # hash and the TP snapshot certificate above.  Confirm the
                # atomic artifact directly, retaining the same irreversible
                # visibility/fail-stop discipline as the context-bound path.
                confirm_error = None
                try:
                    adapter.confirm(published)
                except BaseException as error:
                    confirm_error = error
                try:
                    confirm_all = self._mla_off_vote_restore_ready(
                        confirm_error is None, device
                    )
                except BaseException as vote_error:
                    self._redknot_shared_snapshot_stages.pop(stage_key, None)
                    reason = (
                        "independent snapshot final visibility vote failed: "
                        f"{type(vote_error).__name__}: {vote_error}; "
                        f"local_confirm={confirm_error or 'ok'}"
                    )
                    self._mla_off_quarantine_shared_latent(reason)
                    raise RuntimeError(
                        "RedKnot cannot certify independent snapshot visibility; "
                        "this worker must be restarted"
                    ) from vote_error
                self._redknot_shared_snapshot_stages.pop(stage_key, None)
                if not confirm_all or confirm_error is not None:
                    reason = str(confirm_error) if confirm_error is not None else (
                        "another TP rank failed independent snapshot confirmation"
                    )
                    self._mla_off_quarantine_shared_latent(reason)
                    raise RuntimeError(
                        "RedKnot independent snapshot confirmation diverged; "
                        "this worker must be restarted: " + reason
                    )
                try:
                    self._count("mla_off.shared_snapshot_published")
                except BaseException:
                    pass
                try:
                    self._mla_off_maybe_emit_shared_snapshot_audit(
                        adapter=adapter, published=published
                    )
                except BaseException:
                    pass
                return True

            prepared_context_publication = None
            context_receipt_error = None
            try:
                prepared_context_publication = (
                    self._redknot_context_token_streams.prepare_snapshot_publication(
                        request_binding=tuple(
                            mla_off_context.context_snapshot_request_binding
                        ),
                        confirmation_digest=str(confirmation_digest),
                        seg_hash=str(identity["seg_hash"]),
                        model_compat_hash=self._redknot_mla_off_model_hash,
                        head_policy_hash=self._redknot_mla_off_policy_hash,
                        generation_id=str(identity["generation_id"]),
                        published_layer_ids=tuple(
                            self._redknot_mla_off_rank_local_layer_ids
                        ),
                    )
                )
            except BaseException as error:
                context_receipt_error = error
            try:
                context_receipt_all, _, _ = self._mla_off_snapshot_digest_vote(
                    local_ready=context_receipt_error is None,
                    local_digest=(
                        prepared_context_publication.receipt
                        if prepared_context_publication is not None
                        else ""
                    ),
                    common_digest=identity_common_digest,
                    device=device,
                )
            except BaseException as vote_error:
                rollback_failure = rollback_shared(published)
                reason = (
                    "context receipt TP vote failed before visibility: "
                    f"{type(vote_error).__name__}: {vote_error}; "
                    f"local_rollback={rollback_failure or 'ok'}"
                )
                self._mla_off_quarantine_shared_latent(reason)
                raise RuntimeError(
                    "RedKnot cannot certify the context snapshot receipt vote; "
                    "this worker must be restarted"
                ) from vote_error
            if not context_receipt_all or prepared_context_publication is None:
                rollback_shared_collectively(published, "context-receipt")
                self._mla_off_log_failure(
                    "context_snapshot_receipt_prepare_failed",
                    str(context_receipt_error)
                    if context_receipt_error is not None
                    else "TP context receipt preparation differed",
                )
                return False

            confirm_error = None
            try:
                adapter.confirm(published)
            except BaseException as error:
                confirm_error = error
            try:
                confirm_all = self._mla_off_vote_restore_ready(
                    confirm_error is None, device
                )
            except BaseException as vote_error:
                prepared_context_publication.poison_noexcept()
                self._redknot_shared_snapshot_stages.pop(stage_key, None)
                reason = (
                    "final shared snapshot confirmation vote failed after "
                    "an irreversible visibility transition: "
                    f"{type(vote_error).__name__}: {vote_error}; "
                    f"local_confirm={confirm_error or 'ok'}"
                )
                self._mla_off_quarantine_shared_latent(reason)
                raise RuntimeError(
                    "RedKnot cannot certify shared snapshot visibility; this "
                    "worker must be restarted"
                ) from vote_error
            self._redknot_shared_snapshot_stages.pop(stage_key, None)
            if not confirm_all:
                prepared_context_publication.poison_noexcept()
                poison_reason = (
                    str(confirm_error)
                    if confirm_error is not None
                    else "another TP rank failed final visibility confirmation"
                )
                # Every rank reaches this branch through the same final vote.
                # Confirmation may already be irreversible on a subset of
                # participants/ranks, so rollback or a normal dense fallback
                # would falsely certify a coherent epoch.  Quarantine both
                # shared-latent and MLA-off on every rank and fail the request;
                # a process restart is the only operation that reconstructs a
                # provably common serving state.
                self._mla_off_quarantine_shared_latent(poison_reason)
                self._mla_off_log_failure(
                    "shared_snapshot_confirm_failed", poison_reason
                )
                raise RuntimeError(
                    "RedKnot shared-latent snapshot confirmation diverged "
                    "after the serving visibility gate; MLA reuse is "
                    "quarantined and this worker must be restarted: "
                    f"{poison_reason}"
                )
            if confirm_error is not None:
                prepared_context_publication.poison_noexcept()
                raise AssertionError("snapshot confirmation vote accepted failure")
            # All validation/allocation and the TP receipt vote occurred before
            # adapter.confirm.  The receipt commit only installs prebuilt
            # references.  Observability below is best-effort and must never
            # let one rank escape before the following wo_b collective.
            prepared_context_publication.commit_noexcept()
            try:
                self._count("mla_off.shared_snapshot_published")
            except BaseException:
                # Metrics are downstream of an irreversible visibility gate.
                # Even process-control exceptions must not split TP ranks
                # before the following wo_b collective.
                pass
            try:
                self._mla_off_maybe_emit_shared_snapshot_audit(
                    adapter=adapter, published=published
                )
            except BaseException:
                # A missing metric/audit marker makes the benchmark fail
                # closed, but cannot change an already-confirmed artifact or
                # split the TP execution path.
                pass
            return True

        complete_failure = ""
        locally_complete = False
        local_stage_ok = bool(capture_succeeded)
        try:
            if local_stage_ok:
                locally_complete = bool(mla_off_context.snapshot_complete())
        except Exception as error:
            local_stage_ok = False
            locally_complete = False
            complete_failure = str(error)
        try:
            capture_ready = self._mla_off_vote_restore_ready(
                local_stage_ok, device
            )
        except Exception:
            # A failed collective cannot certify the staging generation.
            mla_off_context.abort_snapshot()
            raise
        if not capture_ready:
            mla_off_context.abort_snapshot()
            self._mla_off_log_failure(
                "snapshot_capture_failed",
                complete_failure
                or "at least one TP rank failed to capture snapshot rows",
            )
            return False
        try:
            complete_count = self._mla_off_vote_count(
                locally_complete, device
            )
        except Exception:
            mla_off_context.abort_snapshot()
            raise
        if complete_count not in (0, int(self._redknot_tp_size)):
            mla_off_context.abort_snapshot()
            self._mla_off_log_failure(
                "snapshot_completion_mismatch",
                complete_failure
                or "snapshot coverage differed across attention-TP ranks",
            )
            return False
        if complete_count == int(self._redknot_tp_size):
            receipt = None
            publish_ok = True
            publish_failure = ""
            try:
                receipt = mla_off_context.publish_snapshot()
            except Exception as error:
                publish_ok = False
                publish_failure = str(error)
            try:
                published_count = self._mla_off_vote_count(publish_ok, device)
            except Exception:
                # Some ranks may already have installed the new generation.
                # Without a publish certificate, clear this CPU-only cache so
                # no pending or ambiguously committed artifact can be reused.
                mla_off_context.controller.clear()
                raise
            if published_count != int(self._redknot_tp_size):
                rollback_ok = True
                try:
                    mla_off_context.rollback_snapshot(receipt)
                except Exception as error:
                    rollback_ok = False
                    publish_failure = publish_failure or str(error)
                mla_off_context.abort_snapshot()
                try:
                    rollback_count = self._mla_off_vote_count(
                        rollback_ok, device
                    )
                except Exception:
                    mla_off_context.controller.clear()
                    raise
                receipt = None
                if rollback_count != int(self._redknot_tp_size):
                    # Unknown local publish state must never remain eligible for
                    # reuse. Clearing this experiment's CPU cache is fail-safe
                    # and does not touch the packed KV cache.
                    mla_off_context.controller.clear()
                self._mla_off_log_failure(
                    "snapshot_publish_failed",
                    publish_failure
                    or "at least one TP rank failed the artifact publish step",
                )
                return False
            confirm_ok = True
            try:
                mla_off_context.validate_snapshot_confirmation(receipt)
            except Exception as error:
                confirm_ok = False
                publish_failure = str(error)
            try:
                confirmed_count = self._mla_off_vote_count(confirm_ok, device)
            except Exception:
                mla_off_context.controller.clear()
                raise
            if confirmed_count != int(self._redknot_tp_size):
                mla_off_context.controller.clear()
                self._mla_off_log_failure(
                    "snapshot_confirm_failed",
                    publish_failure
                    or "at least one TP rank failed to confirm artifact publish",
                )
                return False
            try:
                mla_off_context.confirm_snapshot(receipt)
            except Exception as error:
                mla_off_context.controller.clear()
                self._mla_off_log_failure(
                    "snapshot_confirm_release_failed", str(error)
                )
                return False
            receipt = None
        return True

    @staticmethod
    def _mla_off_unsupported_auxiliary_inputs(
        forward_batch: ForwardBatch,
    ) -> Tuple[str, ...]:
        """Return request state that is not part of the v1 artifact identity.

        Token ids alone do not identify the attention output when adapters,
        caller-provided embeddings, multimodal payloads, or alternate position
        semantics are active.  Until those revisions are included in the
        compatibility hash, fail closed instead of reusing a numerically
        unrelated offline projection.
        """

        unsupported = []
        if getattr(forward_batch, "input_embeds", None) is not None:
            unsupported.append("input_embeds")
        if getattr(forward_batch, "replace_embeds", None) is not None:
            unsupported.append("replace_embeds")
        if any(
            item is not None
            for item in (getattr(forward_batch, "mm_inputs", None) or ())
        ):
            unsupported.append("mm_inputs")
        if any(getattr(forward_batch, "lora_ids", None) or ()):
            unsupported.append("lora_ids")
        if getattr(forward_batch, "token_type_ids", None) is not None:
            unsupported.append("token_type_ids")
        if getattr(forward_batch, "mrope_positions", None) is not None:
            unsupported.append("mrope_positions")
        if getattr(forward_batch, "ngram_embedding_info", None) is not None:
            unsupported.append("ngram_embedding_info")
        if (
            getattr(forward_batch, "multi_item_delimiter_indices", None)
            is not None
        ):
            unsupported.append("multi_item_delimiter_indices")
        if (
            getattr(forward_batch, "tbo_split_seq_index", None) is not None
            or bool(getattr(forward_batch, "tbo_children", None))
        ):
            unsupported.append("two_batch_overlap")
        if getattr(forward_batch, "attn_cp_metadata", None) is not None:
            unsupported.append("attn_cp_metadata")
        return tuple(unsupported)

    @staticmethod
    def _mla_off_tensors_digest(*tensors: torch.Tensor) -> Tuple[int, int]:
        """Fingerprint ordered CPU tensor values and lengths."""

        hasher = hashlib.sha256()
        for tensor in tensors:
            tensor = tensor.detach().to(device="cpu").contiguous()
            hasher.update(int(tensor.numel()).to_bytes(8, "little", signed=False))
            hasher.update(str(tensor.dtype).encode("ascii"))
            hasher.update(tensor.numpy().tobytes())
        digest = hasher.hexdigest()
        return int(digest[:5], 16), int(digest[5:10], 16)

    @staticmethod
    def _mla_off_forward_generation_is_current(forward_batch: ForwardBatch) -> bool:
        requested = getattr(
            forward_batch, "_redknot_forward_generation_id", None
        )
        prepared = getattr(
            forward_batch, "_redknot_mla_off_forward_generation", None
        )
        return (
            isinstance(requested, tuple)
            and len(requested) == 2
            and all(type(value) is int and value > 0 for value in requested)
            and prepared == requested
        )

    def _mla_off_prepare_forward_generation(
        self, forward_batch: ForwardBatch
    ) -> bool:
        """Invalidate every ForwardBatch-local certificate once per forward."""

        requested = getattr(
            forward_batch, "_redknot_forward_generation_id", None
        )
        if not (
            isinstance(requested, tuple)
            and len(requested) == 2
            and all(type(value) is int and value > 0 for value in requested)
        ):
            # Older/non-ModelRunner callers remain correct by taking the strict
            # content-verification path instead of trusting inference storage
            # identity as a cross-forward generation.
            return False
        if self._mla_off_forward_generation_is_current(forward_batch):
            return True
        # A persistent shared-latent epoch is leased for the whole 3..39
        # traversal, not independently by every layer.  Release an unfinished
        # previous traversal before replacing any ForwardBatch-local
        # certificates; otherwise retired GPU-bank slots could remain pinned
        # forever after a cancelled request.
        composite_resources = getattr(
            forward_batch, "_redknot_composite_forward_resources", None
        )
        if composite_resources is not None:
            prior_transaction = getattr(
                forward_batch,
                "_redknot_mla_off_forward_transaction",
                None,
            )
            if (
                isinstance(
                    prior_transaction,
                    _RedKnotForwardCompositeTransaction,
                )
                and prior_transaction.coordinator is not None
                and bool(
                    getattr(prior_transaction.coordinator, "committed", False)
                )
                and not prior_transaction.finalized
            ):
                self._fail_stop_mla_off_transaction(
                    prior_transaction,
                    reason_code="unfinished_forward_generation_replaced",
                    detail="scheduler advanced generation before fixed final",
                )
            from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                close_composite_forward_resources,
            )

            close_composite_forward_resources(forward_batch)
        self._release_mla_off_shared_forward_pins(forward_batch)
        for attribute in (
            "_redknot_mla_off_disabled",
            "_redknot_mla_off_forward_evidence",
            "_redknot_mla_off_transfer_audit_state",
            "_redknot_mla_off_plan_validation",
            "_redknot_mla_off_positions_cpu",
            "_redknot_mla_off_input_ids_cpu",
            "_redknot_mla_off_input_layout_digest",
            "_redknot_mla_off_context_stream_phase",
            "_redknot_context_restore_phases",
            "_redknot_rank_local_q_authorizations",
            "_redknot_mla_off_restore_layout",
            "_redknot_mla_off_verified_tokens",
            "_redknot_mla_off_scheduler_total_tokens_cache",
            "_redknot_shared_restore_states",
            "_redknot_shared_restore_forward_token",
            "_redknot_shared_restore_released",
            "_redknot_composite_layer_executions",
            "_redknot_composite_forward_resources",
            "_redknot_mla_off_forward_transaction",
            "_redknot_mla_off_forward_transaction_attempted",
        ):
            if hasattr(forward_batch, attribute):
                delattr(forward_batch, attribute)
        forward_batch._redknot_mla_off_forward_generation = requested
        return True

    @staticmethod
    def _release_mla_off_shared_forward_pins(forward_batch: ForwardBatch) -> None:
        """Idempotently release the per-request persistent GPU epoch leases."""

        if bool(
            getattr(forward_batch, "_redknot_shared_restore_released", False)
        ):
            return
        states = getattr(forward_batch, "_redknot_shared_restore_states", ())
        first_error = None
        for state in tuple(states or ()):
            pin = getattr(state, "pin", None)
            close = getattr(pin, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as error:  # release every remaining request pin.
                if first_error is None:
                    first_error = error
        forward_batch._redknot_shared_restore_released = True
        if first_error is not None:
            raise RuntimeError(
                "failed to release a shared-latent forward epoch pin"
            ) from first_error

    def _mla_off_cache_requires_strict(
        self, forward_batch: ForwardBatch
    ) -> bool:
        return bool(
            getattr(self, "_redknot_mla_off_strict_row_verify", False)
        ) or not self._mla_off_forward_generation_is_current(forward_batch)

    def _mla_off_forward_tensor_cache_key(
        self, forward_batch: ForwardBatch, tensor: torch.Tensor
    ):
        tensor_key = self._mla_off_tensor_cache_key(
            tensor,
            strict=self._mla_off_cache_requires_strict(forward_batch),
        )
        if tensor_key is None:
            return None
        return (
            getattr(
                forward_batch, "_redknot_mla_off_forward_generation", None
            ),
            tensor_key,
        )

    @staticmethod
    def _mla_off_tensor_cache_key(
        tensor: torch.Tensor, *, strict: bool = False
    ):
        """Return a mutation-sensitive or ForwardBatch-lifetime identity."""

        try:
            version = int(tensor._version)
        except RuntimeError:
            if strict:
                return None
            # Inference tensors have no version counter. SGLang treats
            # positions/input_ids as immutable across a ForwardBatch's layer
            # traversal, so storage identity is the production cache contract.
            version = "inference-immutable"
        return (
            id(tensor),
            int(tensor.data_ptr()),
            version,
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )

    def _mla_off_positions_cpu_for_forward(
        self,
        *,
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
    ) -> Tuple[object, torch.Tensor, bool]:
        """Materialize the CPU position authority at most once per forward."""

        position_key = self._mla_off_forward_tensor_cache_key(
            forward_batch, positions
        )
        cached_positions = getattr(
            forward_batch, "_redknot_mla_off_positions_cpu", None
        )
        if (
            position_key is not None
            and isinstance(cached_positions, tuple)
            and len(cached_positions) == 4
            and cached_positions[0] == position_key
            and cached_positions[1] is positions
        ):
            return position_key, cached_positions[2], bool(cached_positions[3])
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long)
        position_contiguous = bool(
            positions_cpu.numel() <= 1
            or torch.equal(positions_cpu[1:], positions_cpu[:-1] + 1)
        )
        forward_batch._redknot_mla_off_positions_cpu = (
            position_key,
            positions,
            positions_cpu,
            position_contiguous,
        )
        return position_key, positions_cpu, position_contiguous

    @staticmethod
    def _mla_off_layout_device_indices(
        *,
        layout_certificate: _MLAOffRestoreLayoutCertificate,
        controller,
        cpu_indices: torch.Tensor,
        device: torch.device,
        role: str,
        semantic_digest: Tuple[int, int],
        upper_bound: int,
    ):
        """Reuse a controller-certified CPU-to-device index copy per forward."""

        target_device = torch.device(device)
        cache_key = (
            id(controller),
            str(role),
            tuple(int(value) for value in semantic_digest),
            int(upper_bound),
            str(target_device),
            _mla_off_control_tensor_identity(cpu_indices),
        )
        certificate = layout_certificate.device_index_certificates.get(
            cache_key
        )
        if certificate is None:
            certificate = controller.prepare_device_indices(
                cpu_indices,
                device=target_device,
                role=role,
                semantic_digest=semantic_digest,
                upper_bound=upper_bound,
            )
            layout_certificate.device_index_certificates[cache_key] = certificate
        else:
            # Do not silently replace a corrupted same-forward certificate.
            # A failed validation must fall back before attention skips work.
            controller.device_indices_from_certificate(
                certificate,
                cpu_indices=cpu_indices,
                device=target_device,
                role=role,
                semantic_digest=semantic_digest,
                upper_bound=upper_bound,
            )
        return certificate

    def _mla_off_indexer_dirty_mask(
        self, plan, positions_cpu: torch.Tensor
    ) -> torch.Tensor:
        dirty_mask = torch.zeros(positions_cpu.numel(), dtype=torch.bool)
        if not bool(plan.get("mla_off_use_indexer_hot", True)):
            return dirty_mask
        try:
            from sglang.srt.layers.attention.redknot.dsv4_offline_reuse_v2 import (
                get_offline_reuse_controller_v2,
            )

            controller = get_offline_reuse_controller_v2()
            hot_frac = float(plan.get("hot_frac", 0.5))
            expansion = max(0, int(plan.get("mla_off_hot_expand_tokens", 0)))
            for segment in plan.get("segments", ()) or ():
                units = controller.get_indexer_hot_units_by_frac(
                    str(segment["seg_hash"]), hot_frac
                )
                if units is None:
                    raise ValueError(
                        "Indexer-hot artifact is missing for segment "
                        f"{segment['seg_hash']!r}"
                )
                offset = int(segment.get("global_offset", 0))
                length = int(segment["length"])
                units = units.detach().to(device="cpu", dtype=torch.long).flatten()
                if bool((units < 0).any().item()) or bool(
                    (units > (length - 1) // 4).any().item()
                ):
                    raise ValueError(
                        f"Indexer-hot units are outside segment {segment['seg_hash']!r}"
                    )
                row_indices = torch.nonzero(
                    positions_cpu.ge(offset)
                    & positions_cpu.lt(offset + length),
                    as_tuple=False,
                ).flatten()
                if row_indices.numel() == 0 or units.numel() == 0:
                    continue
                units = torch.unique(units, sorted=True)
                starts = (units * 4 - expansion).clamp(min=0, max=length)
                ends = ((units + 1) * 4 + expansion).clamp(
                    min=0, max=length
                )
                local_positions = positions_cpu.index_select(0, row_indices) - offset
                interval_ids = torch.bucketize(
                    local_positions, starts, right=True
                ) - 1
                has_interval = interval_ids.ge(0)
                interval_ids = interval_ids.clamp(min=0)
                selected = has_interval & local_positions.lt(
                    ends.index_select(0, interval_ids)
                )
                dirty_mask[row_indices[selected]] = True
            return dirty_mask
        except Exception as error:
            self._count("mla_off.indexer_signal_unavailable")
            raise ValueError(
                "MLA-off requested Indexer-hot dirty rows, but the signal is "
                f"unavailable: {error}"
            ) from error

    def _mla_off_scheduler_total_tokens(self, forward_batch) -> Optional[int]:
        """Read the scheduler-owned, unchunked request length fail-closed."""

        orig_seq_lens = getattr(forward_batch, "orig_seq_lens", None)
        if not isinstance(orig_seq_lens, torch.Tensor):
            return None
        if orig_seq_lens.ndim != 1 or int(orig_seq_lens.numel()) != 1:
            return None
        if orig_seq_lens.dtype not in (torch.int32, torch.int64):
            return None
        cached = getattr(
            forward_batch,
            "_redknot_mla_off_scheduler_total_tokens_cache",
            None,
        )
        if (
            self._mla_off_forward_generation_is_current(forward_batch)
            and isinstance(cached, tuple)
            and len(cached) == 3
            and cached[0]
            == getattr(
                forward_batch, "_redknot_mla_off_forward_generation", None
            )
            and cached[1] is orig_seq_lens
            and type(cached[2]) is int
            and cached[2] > 0
        ):
            return cached[2]
        try:
            total_tokens = int(orig_seq_lens.detach()[0].item())
        except (RuntimeError, TypeError, ValueError):
            return None
        if total_tokens <= 0:
            return None
        if self._mla_off_forward_generation_is_current(forward_batch):
            forward_batch._redknot_mla_off_scheduler_total_tokens_cache = (
                getattr(
                    forward_batch,
                    "_redknot_mla_off_forward_generation",
                    None,
                ),
                orig_seq_lens,
                total_tokens,
            )
        return total_tokens

    def _mla_off_context_request_binding(
        self,
        *,
        forward_batch: ForwardBatch,
        plan: Mapping[str, object],
        request_index: int = 0,
    ) -> Tuple[object, ...]:
        """Bind a token stream to one scheduler request/pool lifecycle.

        ``id(plan)`` is not a wire identity: ForwardBatch.init_new takes the
        mapping directly from the live scheduler request, so its object
        identity acts as a process-local request-generation token.  Combining
        it with the scheduler rid, request-pool index and server nonce prevents
        a cancelled request from being resumed by a foreign request that only
        repeats user-controlled hashes/ids.
        """

        rids = getattr(forward_batch, "rids", None)
        if type(request_index) is not int or request_index < 0:
            raise ValueError("context-bound MLA request index is invalid")
        if (
            not isinstance(rids, (tuple, list))
            or request_index >= len(rids)
            or not isinstance(rids[request_index], str)
            or not rids[request_index]
        ):
            raise ValueError("context-bound MLA needs a scheduler request id")
        raw_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if isinstance(raw_pool_indices, torch.Tensor):
            if (
                raw_pool_indices.ndim != 1
                or request_index >= int(raw_pool_indices.numel())
            ):
                raise ValueError("context-bound MLA needs a request-pool index")
            pool_index = int(
                raw_pool_indices.detach()
                .to(device="cpu", dtype=torch.long)[request_index]
                .item()
            )
        elif (
            isinstance(raw_pool_indices, (tuple, list))
            and request_index < len(raw_pool_indices)
            and type(raw_pool_indices[request_index]) is int
        ):
            pool_index = raw_pool_indices[request_index]
        else:
            raise ValueError("context-bound MLA request-pool binding is absent")
        if pool_index < 0:
            raise ValueError("context-bound MLA request-pool index is negative")
        return (
            self._redknot_server_instance_nonce,
            rids[request_index],
            pool_index,
            id(plan),
        )

    def _mla_off_context_scheduler_extent(
        self,
        *,
        forward_batch: ForwardBatch,
        positions_cpu: torch.Tensor,
    ) -> int:
        seq_lens = self._redknot_single_cpu_int_sequence(
            getattr(forward_batch, "seq_lens_cpu", None),
            name="seq_lens_cpu",
        )
        extend_lens = self._redknot_single_cpu_int_sequence(
            getattr(forward_batch, "extend_seq_lens_cpu", None),
            name="extend_seq_lens_cpu",
        )
        if len(seq_lens) != 1 or len(extend_lens) != 1:
            raise ValueError("context-bound MLA requires one scheduler extent")
        if extend_lens[0] != int(positions_cpu.numel()):
            raise ValueError("scheduler extend length differs from attention rows")
        start = int(positions_cpu[0].item())
        end = int(positions_cpu[-1].item()) + 1
        if seq_lens[0] != end:
            raise ValueError("scheduler cumulative extent differs from positions")
        prefix_source = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if prefix_source is not None:
            prefix_lens = self._redknot_single_cpu_int_sequence(
                prefix_source, name="extend_prefix_lens_cpu"
            )
            if prefix_lens != (start,):
                raise ValueError("scheduler prefix extent differs from positions")
        return end

    def _mla_off_resolve_context_cap(self, plan) -> Tuple[int, str]:
        """Resolve the only context cap this server may use for ``plan``.

        Qualification capacity is deliberately disjoint from the formal
        certified capacity.  A qualification server accepts only explicitly
        marked pure restore plans, while a formal server rejects that marker.
        This prevents a qualification run from becoming claim-eligible by
        changing only request metadata or a result post-processor.
        """

        marker_present = _MLA_OFF_QUALIFICATION_PLAN_FIELD in plan
        marker = plan.get(_MLA_OFF_QUALIFICATION_PLAN_FIELD, False)
        if marker_present and type(marker) is not bool:
            return 0, "invalid_qualification_plan_marker"
        qualification_only = bool(
            getattr(self, "_redknot_mla_off_qualification_only", False)
        )
        if qualification_only:
            if marker is not True:
                return 0, "qualification_plan_marker_required"
            try:
                from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
                    resolve_mla_off_diagnostic_ablation,
                )

                diagnostic_ablation = resolve_mla_off_diagnostic_ablation(plan)
            except Exception:
                return 0, "invalid_diagnostic_ablation"
            if diagnostic_ablation != "full":
                return 0, "qualification_requires_full_diagnostic_ablation"
            cap = getattr(
                self,
                "_redknot_mla_off_qualification_max_context_tokens",
                0,
            )
            if type(cap) is not int or cap <= 0:
                return 0, "qualification_context_unconfigured"
            return cap, ""
        if marker_present:
            return 0, "qualification_plan_marker_forbidden"
        cap = getattr(
            self, "_redknot_mla_off_certified_max_context_tokens", 0
        )
        if type(cap) is not int or cap <= 0:
            return 0, "uncertified_context"
        return cap, ""

    def _mla_off_context_safety_reason(self, plan, forward_batch) -> str:
        """Return why this request is outside its server-bound context range.

        ``plan`` is request-controlled metadata.  Its length is therefore only
        accepted when it is a strict integer equal to the scheduler-owned
        unchunked request length.  This prevents stale or deliberately low
        metadata from bypassing either the formal certified cap or the
        explicitly claim-ineligible qualification cap.
        """

        context_cap, cap_reason = self._mla_off_resolve_context_cap(plan)
        if cap_reason:
            return cap_reason
        total_tokens = plan.get("total_tokens")
        if type(total_tokens) is not int:
            return "missing_total_tokens"
        if total_tokens <= 0:
            return "missing_total_tokens"
        scheduler_total_tokens = self._mla_off_scheduler_total_tokens(
            forward_batch
        )
        if scheduler_total_tokens is None:
            return "missing_actual_total_tokens"
        if total_tokens != scheduler_total_tokens:
            return "total_tokens_mismatch"
        if scheduler_total_tokens > context_cap:
            if plan.get(_MLA_OFF_QUALIFICATION_PLAN_FIELD) is True:
                return "context_exceeds_qualification"
            return "context_exceeds_certification"
        return ""

    def _mla_off_restore_safety(
        self, *, plan, layer_id: int, forward_batch
    ) -> Tuple[bool, str]:
        """Resolve explicit refresh before applying request certification."""

        refresh_layers = {
            int(value) for value in plan.get("mla_off_refresh_layers", ()) or ()
        }
        refresh_stride = int(plan.get("mla_off_refresh_layer_stride", 0) or 0)
        explicit_refresh_layer = layer_id in refresh_layers or (
            refresh_stride > 0 and layer_id % refresh_stride == 0
        )
        context_safety_reason = (
            ""
            if explicit_refresh_layer
            else self._mla_off_context_safety_reason(plan, forward_batch)
        )
        return explicit_refresh_layer, context_safety_reason

    def prepare_mla_off_context(
        self,
        *,
        layer_id: int,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        q_head_count: int,
        q_row_count: int,
        n_local_heads: int,
        n_local_groups: int,
        head_dim: int,
        o_lora_rank: int,
        fp8_wo_a: bool,
        device,
        projection_dtype: torch.dtype,
    ):
        """Prepare a fully validated context before attention may skip work.

        A restore context includes a device-local offline projection. Therefore
        a later merge is only an addition and cannot discover a cache miss after
        clean local-head rows have already been omitted.
        """

        layer_id = int(layer_id)
        if layer_id < 0:
            raise ValueError(f"MLA layer id must be non-negative, got {layer_id}")
        if layer_id >= _PURE_HEADSPLIT_NUM_LAYERS:
            # D-Spark/speculative auxiliary layers are outside the 43 target
            # layers and always stay on their native online path.
            return None
        if layer_id in _PURE_HEADSPLIT_DENSE_LAYER_IDS:
            # Hard fence: boundary layers cannot even construct an MLA-off
            # context, so they necessarily take native full recomputation.
            return None
        if layer_id not in _PURE_HEADSPLIT_OFFLINE_LAYER_IDS:
            raise ValueError(
                f"target MLA layer {layer_id} is outside the 3..39 split range"
            )
        if not self._redknot_mla_off_enabled:
            return None
        if forward_batch.forward_mode not in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
        ):
            # MLA-off is a prefill-only optimization. Decode/speculative steps
            # keep the ordinary online attention path without recording a
            # restore failure or invalidating a committed prefill artifact.
            self._count("mla_off.non_prefill_bypass")
            return None
        if layer_id not in self._redknot_mla_off_local_layer_ids:
            return None
        self._set_rank_local_q_authorization(
            forward_batch, layer_id=layer_id, context=None
        )
        self._mla_off_prepare_forward_generation(forward_batch)
        if bool(
            getattr(
                forward_batch,
                "_redknot_mla_off_forward_transaction_attempted",
                False,
            )
        ):
            transaction = getattr(
                forward_batch, "_redknot_mla_off_forward_transaction", None
            )
            if transaction is None:
                # The one forward setup rendezvous rejected the reservation;
                # never re-enter the legacy per-layer v3 readiness protocol.
                return None
            context = transaction.context_for(layer_id)
            if (
                transaction.coordinator is not None
            ):
                try:
                    transaction.wait_for_restore(
                        device, layer_id=int(layer_id)
                    )
                except BaseException as error:
                    self._fail_stop_mla_off_transaction(
                        transaction,
                        reason_code="restore_completion_dependency_failed",
                        detail=f"{type(error).__name__}: {error}",
                    )
            self._set_rank_local_q_authorization(
                forward_batch, layer_id=layer_id, context=context
            )
            return context

        # Production v3: resolve every request in the continuously batched
        # forward, pin z_off/shared-latent GPU epochs, and build persistent
        # schedules before wq_b.  Snapshot publication still uses the legacy
        # staging entry below until the combined bundle is finalized.
        plans_for_v3 = getattr(forward_batch, "redknot_reuse_plan", None)
        locally_disabled_v3 = bool(
            getattr(forward_batch, "_redknot_mla_off_disabled", False)
        )
        v3_restore_requested = bool(
            self._redknot_shared_latent_enabled
            and not locally_disabled_v3
            and plans_for_v3
            and len(plans_for_v3) == int(forward_batch.batch_size)
            and any(
                isinstance(item, Mapping)
                and item.get("mode") == "restore"
                and bool(item.get("reuse_mla_off", False))
                for item in plans_for_v3
            )
        )
        v3_context = None
        v3_error = None
        v3_transfer_audit_state = None
        if v3_restore_requested:
            try:
                from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                    close_composite_forward_resources,
                    prepare_composite_restore_context,
                )

                # Capture before composite preparation: that preparation may
                # create persistent z_off views and upload device indices.
                v3_transfer_audit_state = (
                    self._mla_off_begin_composite_transfer_audit(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                    )
                )
                v3_context = prepare_composite_restore_context(
                    self,
                    layer_id=layer_id,
                    positions=positions,
                    forward_batch=forward_batch,
                    q_row_count=q_row_count,
                    n_local_heads=n_local_heads,
                    n_local_groups=n_local_groups,
                    head_dim=head_dim,
                    o_lora_rank=o_lora_rank,
                    device=torch.device(device),
                    projection_dtype=projection_dtype,
                )
                if v3_context is None:
                    preflight_fallbacks = tuple(
                        getattr(
                            forward_batch,
                            "_redknot_composite_preflight_fallbacks",
                            (),
                        )
                        or ()
                    )
                    raise ValueError(
                        "composite restore produced no reusable row context; "
                        f"preflight_fallbacks={preflight_fallbacks!r}"
                    )
                self._mla_off_bind_composite_transfer_audit(
                    v3_transfer_audit_state, v3_context
                )
            except Exception as error:
                # Do not return locally.  Every rank below enters the same
                # readiness rendezvous, including the rank whose artifact pin,
                # ragged geometry, or persistent-view preparation failed.
                v3_error = error
                v3_context = None

        if self._redknot_shared_latent_enabled:
            diagnostic_modes = (
                "none",
                "full",
                "zoff_only",
                "shared_only",
                "invalid",
            )
            diagnostic_mode = "none"
            plan_digest_a = plan_digest_b = 0
            if v3_restore_requested:
                try:
                    from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
                        resolve_mla_off_diagnostic_ablation,
                    )

                    requested_diagnostics = tuple(
                        resolve_mla_off_diagnostic_ablation(item)
                        for item in plans_for_v3
                        if isinstance(item, Mapping)
                        and item.get("mode") == "restore"
                        and bool(item.get("reuse_mla_off", False))
                    )
                    if (
                        not requested_diagnostics
                        or len(set(requested_diagnostics)) != 1
                    ):
                        raise ValueError(
                            "ragged restore mixes diagnostic ablation modes"
                        )
                    diagnostic_mode = requested_diagnostics[0]
                    plan_digest_a, plan_digest_b = self._mla_off_plan_digest(
                        plans_for_v3
                    )
                except Exception as error:
                    diagnostic_mode = "invalid"
                    if v3_error is None:
                        v3_error = error
                    v3_context = None
            # One fixed vector replaces rank-local v3 branching.  It is reached
            # on every middle-layer prefill rank, so a malformed local plan can
            # never leave healthy peers waiting in the later composite commit.
            diagnostic_vote = [0] * len(diagnostic_modes)
            diagnostic_vote[diagnostic_modes.index(diagnostic_mode)] = 1
            v3_signal = torch.tensor(
                [
                    int(v3_restore_requested),
                    int(v3_restore_requested and v3_context is not None),
                    int(locally_disabled_v3),
                    int(
                        v3_restore_requested
                        and v3_context is not None
                        and getattr(v3_context, "is_full_local", False)
                    ),
                    *diagnostic_vote,
                    int(plan_digest_a),
                    int(plan_digest_a) * int(plan_digest_a),
                    int(plan_digest_b),
                    int(plan_digest_b) * int(plan_digest_b),
                ],
                dtype=torch.int64,
                device=device,
            )
            if self._redknot_tp_size > 1:
                v3_signal = self._mla_off_control_all_reduce(v3_signal)
            reduced_v3 = tuple(int(value) for value in v3_signal.tolist())
            requested_count, ready_count, disabled_count, full_local_count = (
                reduced_v3[:4]
            )
            world = int(self._redknot_tp_size)
            v3_consensus = requested_count in (0, world)
            full_local_consensus = full_local_count in (0, world)
            diagnostic_totals = reduced_v3[4 : 4 + len(diagnostic_modes)]
            agreed_diagnostics = tuple(
                diagnostic_modes[index]
                for index, count in enumerate(diagnostic_totals)
                if count == world
            )
            diagnostic_consensus = bool(
                len(agreed_diagnostics) == 1
                and sum(diagnostic_totals) == world
                and agreed_diagnostics[0] != "invalid"
                and (
                    (requested_count == 0 and agreed_diagnostics[0] == "none")
                    or (
                        requested_count == world
                        and agreed_diagnostics[0] != "none"
                    )
                )
            )
            digest_offset = 4 + len(diagnostic_modes)
            plan_digest_consensus = bool(
                requested_count == 0
                or (
                    world * reduced_v3[digest_offset + 1]
                    == reduced_v3[digest_offset] * reduced_v3[digest_offset]
                    and world * reduced_v3[digest_offset + 3]
                    == reduced_v3[digest_offset + 2]
                    * reduced_v3[digest_offset + 2]
                )
            )
            if disabled_count or not v3_consensus or (
                requested_count == world and ready_count != world
            ) or not full_local_consensus or not diagnostic_consensus or not (
                plan_digest_consensus
            ):
                try:
                    from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                        close_composite_forward_resources,
                    )

                    close_composite_forward_resources(forward_batch)
                except Exception:
                    pass
                forward_batch._redknot_mla_off_disabled = True
                self._mla_off_log_failure(
                    "composite_v3_preflight_failed",
                    str(v3_error)
                    if v3_error is not None
                    else (
                        "attention-TP ranks disagreed on v3 mode/readiness: "
                        f"requested={requested_count} ready={ready_count} "
                        f"disabled={disabled_count} "
                        f"full_local={full_local_count} world={world}"
                        f" diagnostic={agreed_diagnostics!r} "
                        f"plan_digest_consensus={plan_digest_consensus}"
                    ),
                )
                return None
            if requested_count == world:
                if v3_context is None:
                    raise AssertionError("v3 consensus accepted an empty context")
                self._count("mla_off.composite_v3_contexts")
                # Every formal single-request composite forward needs a
                # manifest before its per-layer metrics.  This cannot live in
                # the full-local branch below: reusable document chunks take
                # this same early-return path and would otherwise leave only
                # their metrics, causing the benchmark to reject them as
                # missing-manifest evidence.
                if (
                    self._redknot_mla_off_rank_local_layer_ids
                    and layer_id
                    == self._redknot_mla_off_rank_local_layer_ids[0]
                ):
                    self._mla_off_log_composite_forward_manifest(
                        v3_context
                    )
                if getattr(v3_context, "is_full_local", False):
                    reason = str(
                        getattr(
                            v3_context,
                            "intentional_full_local_reason",
                            "",
                        )
                        or "query_suffix_only"
                    )
                    self._count("mla_off.intentional_full_local_layers")
                    for v3_plan in tuple(plans_for_v3):
                        if not (
                            isinstance(v3_plan, Mapping)
                            and v3_plan.get("mode") == "restore"
                            and bool(v3_plan.get("reuse_mla_off", False))
                        ):
                            continue
                        self._mla_off_log_request_status(
                            plan=v3_plan,
                            layer_id=layer_id,
                            status="full_local",
                            reason=reason,
                            forward_id=str(
                                getattr(
                                    v3_context,
                                    "benchmark_forward_id",
                                    "",
                                )
                            ),
                            forward_mode=str(
                                getattr(
                                    v3_context,
                                    "benchmark_forward_mode",
                                    "unknown",
                                )
                            ),
                            q_rows=int(
                                getattr(v3_context, "benchmark_q_rows", 0)
                            ),
                        )
                self._set_rank_local_q_authorization(
                    forward_batch, layer_id=layer_id, context=v3_context
                )
                return v3_context

        plans = getattr(forward_batch, "redknot_reuse_plan", None)
        plan_parse_invalid = False
        try:
            plan = plans[0] if plans and len(plans) == 1 else None
            if plan is not None and not isinstance(plan, Mapping):
                plan_parse_invalid = True
                capture_requested = restore_flag = False
                plan_mode = ""
            else:
                capture_requested = bool(
                    plan and plan.get("capture_mla_off", False)
                )
                restore_flag = bool(plan and plan.get("reuse_mla_off", False))
                plan_mode = str(plan.get("mode", "")) if plan else ""
                if capture_requested or restore_flag:
                    _validate_pure_headsplit_plan_contract(plan)
                    if (
                        restore_flag
                        and str(
                            plan.get("mla_off_diagnostic_ablation", "full")
                        )
                        != "full"
                    ):
                        raise ValueError(
                            "diagnostic ablation requires composite v3 restore"
                        )
        except Exception:
            # Keep malformed rank-local input inside the preflight protocol;
            # raising here could strand healthy peers in the collective.
            plan_parse_invalid = True
            plan = None
            capture_requested = restore_flag = False
            plan_mode = ""
        locally_disabled = bool(
            getattr(forward_batch, "_redknot_mla_off_disabled", False)
        )
        if plan_parse_invalid:
            local_mode = "invalid"
        elif locally_disabled or (not capture_requested and not restore_flag):
            local_mode = "none"
        elif (
            capture_requested
            and not restore_flag
            and plan_mode == "snapshot"
        ):
            local_mode = "snapshot"
        elif restore_flag and not capture_requested and plan_mode == "restore":
            local_mode = "restore"
        else:
            local_mode = "invalid"
        local_plan_digest = _MLA_OFF_PLAN_DIGEST_UNSET
        if local_mode != "none":
            try:
                local_plan_digest = self._mla_off_plan_digest(plan)
            except Exception:
                # Preserve the collective fail-closed path: peers must vote
                # before any rank returns for a malformed local plan.
                local_plan_digest = None
        prefetched_positions = None
        if local_mode in ("snapshot", "restore"):
            try:
                # This CPU tensor is required later for row planning anyway.
                # Materializing it before evidence avoids two device .item()
                # synchronizations while malformed input still proceeds to the
                # collective preflight (the exception is handled below).
                prefetched_positions = self._mla_off_positions_cpu_for_forward(
                    forward_batch=forward_batch,
                    positions=positions,
                )
            except Exception:
                prefetched_positions = None
        (
            benchmark_request_id,
            benchmark_forward_id,
            benchmark_forward_mode,
            benchmark_q_rows,
            benchmark_position_start,
            benchmark_position_end,
        ) = self._mla_off_forward_evidence(
            forward_batch=forward_batch,
            plan=plan,
            positions=positions,
            positions_cpu=(
                prefetched_positions[1]
                if prefetched_positions is not None
                else None
            ),
            q_rows=q_row_count,
        )
        mode, preflight_failure = self._mla_off_preflight_mode(
            local_mode,
            plan,
            device,
            local_plan_digest=local_plan_digest,
        )
        if preflight_failure:
            forward_batch._redknot_mla_off_disabled = True
            self._mla_off_log_request_status(
                plan=plan,
                layer_id=layer_id,
                status="fallback",
                reason="preflight_mismatch",
                forward_id=benchmark_forward_id,
                forward_mode=benchmark_forward_mode,
                q_rows=benchmark_q_rows,
            )
            self._mla_off_log_failure("preflight_mismatch", preflight_failure)
            return None
        if mode == "none":
            return None
        if plan is None:
            # Consensus makes this unreachable, but keep the local contract
            # explicit in case the preflight implementation changes.
            forward_batch._redknot_mla_off_disabled = True
            self._mla_off_log_failure(
                "preflight_mismatch", "agreed MLA-off mode has no local plan"
            )
            return None

        restore_requested = mode == "restore"
        local_context = None
        failure = ""
        failure_error = None
        intentional_full_local = False
        pending_full_local_context = None
        snapshot_staging = None
        snapshot_resolution = ""
        transfer_audit_state = None
        context_snapshot_request_binding = None
        try:
            if mode not in ("snapshot", "restore"):
                raise ValueError(f"unsupported MLA-off mode {mode!r}")
            from sglang.srt.layers.attention.redknot.v4.config import RedKnotV4Config
            from sglang.srt.layers.attention.redknot.v4.reuse_planner import (
                validate_mla_off_plan,
            )

            if not isinstance(local_plan_digest, tuple):
                raise ValueError("MLA-off plan has no certified digest")
            validation_key = (
                getattr(
                    forward_batch,
                    "_redknot_mla_off_forward_generation",
                    None,
                ),
                id(plan),
                local_plan_digest,
                str(self.redknot_v4_mode),
            )
            cached_validation = getattr(
                forward_batch, "_redknot_mla_off_plan_validation", None
            )
            if (
                isinstance(cached_validation, tuple)
                and len(cached_validation) == 3
                and cached_validation[0] == validation_key
            ):
                plan_valid = bool(cached_validation[1])
                validation_detail = str(cached_validation[2])
            else:
                overlay_validation = validate_mla_off_plan(
                    plan,
                    config=RedKnotV4Config(mode=self.redknot_v4_mode),
                )
                plan_valid = bool(overlay_validation.valid)
                validation_detail = str(
                    getattr(
                        overlay_validation,
                        "detail",
                        "MLA-off plan validation failed",
                    )
                )
                forward_batch._redknot_mla_off_plan_validation = (
                    validation_key,
                    plan_valid,
                    validation_detail,
                )
            if not plan_valid:
                raise ValueError(validation_detail)
            if str(plan.get("model_compat_hash", "")) != (
                self._redknot_mla_off_model_hash
            ):
                raise ValueError(
                    "context-bound MLA plan model compatibility hash changed"
                )
            if str(plan.get("head_policy_hash", "")) != (
                self._redknot_mla_off_policy_hash
            ):
                raise ValueError(
                    "context-bound MLA plan head policy hash changed"
                )
            if self.redknot_mla_pass_mode != "headwise":
                raise ValueError("MLA-off requires redknot_mla pass mode=headwise")
            unsupported_aux = self._mla_off_unsupported_auxiliary_inputs(
                forward_batch
            )
            if unsupported_aux:
                raise ValueError(
                    "MLA-off v1 does not support request-dependent auxiliary "
                    f"inputs: {','.join(unsupported_aux)}"
                )
            self._maybe_upgrade_forward_metadata()
            if not isinstance(
                self.forward_metadata.core_attn_metadata, DSV4AttnMetadata
            ):
                raise ValueError("MLA-off requires materialized DSV4 metadata")
            if fp8_wo_a:
                raise ValueError("MLA-off v1 does not support the FP8 wo_a path")
            if int(forward_batch.batch_size) != 1:
                raise ValueError("MLA-off v1 requires batch_size=1")
            combined_row_sparse = (
                self._redknot_mla_off_execution_profile
                == _COMBINED_ROW_SPARSE_PROFILE
            )
            if (
                getattr(forward_batch, "redknot_active_row_indices", None)
                is not None
                and not combined_row_sparse
            ):
                raise ValueError("MLA-off cannot consume an already row-pruned batch")
            if bool(plan.get("skip_forward", False)) and not combined_row_sparse:
                raise ValueError("MLA-off requires skip_forward=false")
            if restore_requested:
                expected_approximate = True if combined_row_sparse else False
                if plan.get("allow_approximate") is not expected_approximate:
                    raise ValueError(
                        "MLA-off restore approximation marker differs from profile"
                    )
            if (
                self._redknot_tp_size * int(n_local_heads)
                != self.redknot_mla_head_cfg.num_attention_heads
            ):
                raise ValueError("MLA-off v1 does not support this CP/TP head layout")
            if int(q_head_count) not in (
                int(n_local_heads),
                self.redknot_mla_head_cfg.num_attention_heads,
            ):
                raise ValueError("MLA-off Q presentation is incompatible")
            if self._headwise_owned_view(q_head_count, n_local_heads) is None:
                raise ValueError("MLA-off cannot execute this owned-head view")
            if n_local_heads <= 0 or n_local_groups <= 0:
                raise ValueError("MLA-off local output grouping is invalid")
            if n_local_heads % n_local_groups != 0:
                raise ValueError("MLA-off heads are not divisible by output groups")
            if positions.ndim != 1 or int(positions.numel()) == 0:
                raise ValueError("MLA-off positions are empty or not one-dimensional")
            if int(positions.numel()) != int(q_row_count):
                raise ValueError("MLA-off positions do not match attention rows")

            logical_start = self._redknot_tp_rank * int(n_local_heads)
            owned_logical = tuple(
                range(logical_start, logical_start + int(n_local_heads))
            )
            owned_set = set(owned_logical)
            layer_plan = self._redknot_dual_layer_plans[layer_id]
            local_logical = tuple(
                head
                for _, heads in layer_plan.local_groups
                for head in heads
                if head in owned_set
            )
            local_set = set(local_logical)
            local_axes = tuple(
                axis for axis, head in enumerate(owned_logical) if head in local_set
            )
            if not local_axes:
                raise ValueError(
                    "MLA-off requires a local-head contribution on every TP rank"
                )

            from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
                MLA_OFF_INDEPENDENT_POSITION_SEMANTICS,
                MLAOffLayerSpec,
                MLAOffRuntimeContext,
                build_restore_rows,
                get_dsv4_mla_off_controller,
                mla_off_device_expected_bytes,
                mla_off_expected_bytes,
            )

            spec = MLAOffLayerSpec(
                layer_id=int(layer_id),
                tp_rank=int(self._redknot_tp_rank),
                tp_size=int(self._redknot_tp_size),
                owned_logical_heads=owned_logical,
                offline_local_heads=local_logical,
                num_output_groups=int(n_local_groups),
                heads_per_group=int(n_local_heads // n_local_groups),
                head_dim=int(head_dim),
                o_lora_rank=int(o_lora_rank),
                model_compat_hash=self._redknot_mla_off_model_hash,
                head_policy_hash=self._redknot_mla_off_policy_hash,
                execution_profile=self._redknot_mla_off_execution_profile,
                required_layer_ids=self._redknot_mla_off_rank_local_layer_ids,
                position_semantics=(
                    MLA_OFF_INDEPENDENT_POSITION_SEMANTICS
                    if self._redknot_mla_off_execution_profile
                    in (
                        _INDEPENDENT_RELOCATION_PROFILE,
                        _COMBINED_ROW_SPARSE_PROFILE,
                    )
                    else "post_inverse_rope_offline_head_woa_"
                    "context_exactpos_fullscope_v3"
                ),
            )
            controller = get_dsv4_mla_off_controller()
            if prefetched_positions is None:
                prefetched_positions = self._mla_off_positions_cpu_for_forward(
                    forward_batch=forward_batch,
                    positions=positions,
                )
            position_key, positions_cpu, position_contiguous = (
                prefetched_positions
            )
            if not position_contiguous and not combined_row_sparse:
                raise ValueError(
                    "MLA-off requires strictly contiguous prefill positions"
                )
            input_ids = getattr(forward_batch, "input_ids", None)
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.ndim != 1
                or int(input_ids.numel()) != int(q_row_count)
            ):
                raise ValueError(
                    "MLA-off requires one materialized token id per attention row"
                )
            input_key = self._mla_off_forward_tensor_cache_key(
                forward_batch, input_ids
            )
            cached_input_ids = getattr(
                forward_batch, "_redknot_mla_off_input_ids_cpu", None
            )
            if (
                input_key is None
                or cached_input_ids is None
                or cached_input_ids[0] != input_key
                or cached_input_ids[1] is not input_ids
            ):
                input_ids_cpu = input_ids.detach().to(
                    device="cpu", dtype=torch.long
                )
                forward_batch._redknot_mla_off_input_ids_cpu = (
                    input_key,
                    input_ids,
                    input_ids_cpu,
                )
            else:
                input_ids_cpu = cached_input_ids[2]
            input_layout_cache_key = (position_key, input_key)
            cached_input_layout = getattr(
                forward_batch, "_redknot_mla_off_input_layout_digest", None
            )
            if (
                position_key is None
                or input_key is None
                or not isinstance(cached_input_layout, tuple)
                or len(cached_input_layout) != 4
                or cached_input_layout[0] != input_layout_cache_key
                or cached_input_layout[1] is not positions
                or cached_input_layout[2] is not input_ids
            ):
                input_layout_digest = self._mla_off_tensors_digest(
                    positions_cpu, input_ids_cpu
                )
                forward_batch._redknot_mla_off_input_layout_digest = (
                    input_layout_cache_key,
                    positions,
                    input_ids,
                    input_layout_digest,
                )
            else:
                input_layout_digest = cached_input_layout[3]
            if (
                self._redknot_mla_off_rank_local_layer_ids
                and layer_id == self._redknot_mla_off_rank_local_layer_ids[0]
            ):
                self._mla_off_log_forward_start(
                    request_id=benchmark_request_id,
                    forward_id=benchmark_forward_id,
                    forward_mode=benchmark_forward_mode,
                    q_rows=benchmark_q_rows,
                    position_start=benchmark_position_start,
                    position_end=benchmark_position_end,
                    position_contiguous=position_contiguous,
                    plan_mode=mode,
                    diagnostic_ablation=str(
                        plan.get("mla_off_diagnostic_ablation", "full")
                    ),
                )
            if mode == "restore":
                transfer_audit_state = self._mla_off_begin_transfer_audit(
                    forward_batch=forward_batch,
                    controller=controller,
                    layer_id=layer_id,
                    request_id=benchmark_request_id,
                    forward_id=benchmark_forward_id,
                    forward_mode=benchmark_forward_mode,
                    q_rows=benchmark_q_rows,
                )

            if mode == "snapshot":
                seg_hash = str(plan.get("seg_hash", ""))
                token_hash = str(plan.get("token_hash", seg_hash))
                request_id = (
                    str(forward_batch.rids[0])
                    if getattr(forward_batch, "rids", None)
                    else "missing-rid"
                )
                length = int(plan.get("length", 0))
                canonical_start = int(plan.get("canonical_start_pos", 0))
                independent_relocation = (
                    self._redknot_mla_off_execution_profile
                    == _INDEPENDENT_RELOCATION_PROFILE
                )
                if independent_relocation or (
                    self._redknot_mla_off_execution_profile
                    == _COMBINED_ROW_SPARSE_PROFILE
                ):
                    source_start = 0
                    source_end = length
                    if canonical_start != 0 or length <= 0:
                        raise ValueError(
                            "independent snapshot requires position-0 artifact"
                        )
                    if not str(plan.get("snapshot_generation_id", "")):
                        raise ValueError(
                            "independent chunked snapshot requires one stable "
                            "snapshot_generation_id"
                        )
                    request_binding = None
                else:
                    source_start = plan.get("source_start")
                    source_end = plan.get("source_end")
                    if (
                        type(source_start) is not int
                        or type(source_end) is not int
                        or source_start < 0
                        or source_end != source_start + length
                    ):
                        raise ValueError(
                            "context snapshot source interval is inconsistent"
                        )
                    request_binding = self._mla_off_context_request_binding(
                        forward_batch=forward_batch,
                        plan=plan,
                    )
                context_snapshot_request_binding = request_binding
                phase_cache_key = (
                    getattr(
                        forward_batch,
                        "_redknot_mla_off_forward_generation",
                        None,
                    ),
                    id(plan),
                    input_layout_digest,
                    request_binding,
                )
                cached_phase = getattr(
                    forward_batch,
                    "_redknot_mla_off_context_stream_phase",
                    None,
                )
                phase_error = None
                if independent_relocation or (
                    self._redknot_mla_off_execution_profile
                    == _COMBINED_ROW_SPARSE_PROFILE
                ):
                    if (
                        positions_cpu.numel() == 0
                        or int(positions_cpu[0].item()) < 0
                        or int(positions_cpu[-1].item()) >= length
                        or not position_contiguous
                    ):
                        raise ValueError(
                            "independent snapshot rows must be a contiguous "
                            "subset of local positions [0,length)"
                        )
                    snapshot_phase = "capture"
                    phase_digest = tuple(input_layout_digest)
                    phase_error = None
                elif (
                    isinstance(cached_phase, tuple)
                    and len(cached_phase) == 3
                    and cached_phase[0] == phase_cache_key
                ):
                    snapshot_phase = str(cached_phase[1])
                    phase_digest = tuple(cached_phase[2])
                elif layer_id == self._redknot_mla_off_rank_local_layer_ids[0]:
                    try:
                        scheduler_total = self._mla_off_scheduler_total_tokens(
                            forward_batch
                        )
                        if scheduler_total is None:
                            raise ValueError(
                                "context snapshot lacks scheduler-owned total tokens"
                            )
                        scheduler_extent = self._mla_off_context_scheduler_extent(
                            forward_batch=forward_batch,
                            positions_cpu=positions_cpu,
                        )
                        snapshot_phase = (
                            self._redknot_context_token_streams.observe_snapshot_chunk(
                                request_id=benchmark_request_id,
                                request_binding=request_binding,
                                plan=plan,
                                positions=tuple(
                                    int(value) for value in positions_cpu.tolist()
                                ),
                                token_ids=tuple(
                                    int(value) for value in input_ids_cpu.tolist()
                                ),
                                scheduler_total=int(scheduler_total),
                                scheduler_current_extent=int(scheduler_extent),
                            )
                        )
                        phase_code = {"prefix": 1, "capture": 2}.get(
                            snapshot_phase, 0
                        )
                        if phase_code == 0:
                            raise ValueError(
                                f"unsupported context snapshot phase {snapshot_phase!r}"
                            )
                        phase_digest = (
                            int(input_layout_digest[0]) ^ phase_code,
                            int(input_layout_digest[1]) ^ (phase_code << 4),
                        )
                        forward_batch._redknot_mla_off_context_stream_phase = (
                            phase_cache_key,
                            snapshot_phase,
                            phase_digest,
                        )
                    except BaseException as error:
                        snapshot_phase = "invalid"
                        phase_digest = (0, 0)
                        phase_error = error
                else:
                    snapshot_phase = "invalid"
                    phase_digest = (0, 0)
                    phase_error = RuntimeError(
                        "context snapshot phase was not certified by first reusable layer"
                    )
                phase_probe, phase_resolution = (
                    self._mla_off_resolve_snapshot_context(
                        (
                            SimpleNamespace(input_layout_digest=phase_digest)
                            if phase_error is None
                            else None
                        ),
                        device=device,
                    )
                )
                if phase_probe is None:
                    raise ValueError(
                        phase_resolution
                        or str(phase_error)
                        or "context snapshot phase differed across TP ranks"
                    ) from phase_error
                if snapshot_phase == "prefix":
                    self._count("mla_off.context_snapshot_dense_prefix_layers")
                    return None
                if snapshot_phase != "capture":
                    raise ValueError("context snapshot did not authorize capture")
                generation_id = _mla_off_snapshot_generation_id(
                    explicit_generation_id=plan.get(
                        "snapshot_generation_id"
                    ),
                    seg_hash=seg_hash,
                    token_hash=token_hash,
                    request_id=request_id,
                    benchmark_forward_id=benchmark_forward_id,
                    input_layout_digest=input_layout_digest,
                    length=length,
                    canonical_start_pos=canonical_start,
                    source_start=source_start,
                    source_end=source_end,
                    prefix_input_hash=str(plan.get("prefix_input_hash", "")),
                    full_input_hash=str(plan.get("full_input_hash", "")),
                    head_scope_policy=str(
                        plan.get("mla_off_head_scope_policy", "")
                    ),
                    model_hash=self._redknot_mla_off_model_hash,
                    policy_hash=self._redknot_mla_off_policy_hash,
                )
                first_local_layer = self._redknot_mla_off_rank_local_layer_ids[0]
                if not self._redknot_shared_latent_enabled:
                    snapshot_staging = (
                        "legacy",
                        controller,
                        seg_hash,
                        generation_id,
                    )
                    if layer_id == first_local_layer and bool(
                        (positions_cpu == canonical_start).any().item()
                    ):
                        expected_bytes = mla_off_expected_bytes(
                            length=length,
                            local_layer_count=len(
                                self._redknot_mla_off_rank_local_layer_ids
                            ),
                            num_output_groups=spec.num_output_groups,
                            o_lora_rank=spec.o_lora_rank,
                        )
                        expected_device_bytes = (
                            mla_off_device_expected_bytes(
                                length=length,
                                local_layer_count=len(
                                    self._redknot_mla_off_rank_local_layer_ids
                                ),
                                num_output_groups=spec.num_output_groups,
                                o_lora_rank=spec.o_lora_rank,
                            )
                            if controller.device_cache_enabled
                            else 0
                        )
                        controller.begin_staging(
                            seg_hash=seg_hash,
                            generation_id=generation_id,
                            token_hash=token_hash,
                            length=length,
                            canonical_start_pos=canonical_start,
                            model_compat_hash=spec.model_compat_hash,
                            head_policy_hash=spec.head_policy_hash,
                            required_local_layers=(
                                self._redknot_mla_off_rank_local_layer_ids
                            ),
                            expected_bytes=expected_bytes,
                            expected_device_bytes=expected_device_bytes,
                        )
                    if layer_id == first_local_layer:
                        controller.capture_token_rows(
                            seg_hash=seg_hash,
                            generation_id=generation_id,
                            local_positions=positions_cpu - canonical_start,
                            token_ids=input_ids_cpu,
                        )
                    staging_ready, staging_reason = controller.validate_staging(
                        seg_hash=seg_hash,
                        generation_id=generation_id,
                        token_hash=token_hash,
                        length=length,
                        spec=spec,
                    )
                    if not staging_ready:
                        raise ValueError(staging_reason)
                local_context = MLAOffRuntimeContext(
                    mode="snapshot",
                    layer_id=int(layer_id),
                    spec=spec,
                    local_head_axes=local_axes,
                    controller=controller,
                    seg_hash=seg_hash,
                    generation_id=generation_id,
                    length=length,
                    local_positions_cpu=positions_cpu - source_start,
                    input_layout_digest=input_layout_digest,
                    benchmark_request_id=self._mla_off_request_id(plan),
                )
                if request_binding is not None:
                    local_context.context_snapshot_request_binding = request_binding
                    local_context.context_snapshot_source_start = source_start
                    local_context.context_snapshot_source_end = source_end
                if self._redknot_shared_latent_enabled:
                    self._mla_off_begin_shared_snapshot(
                        context=local_context,
                        plan=plan,
                        positions_cpu=positions_cpu - source_start,
                        input_ids_cpu=input_ids_cpu,
                        device=device,
                    )
                    snapshot_staging = (
                        "shared",
                        local_context.shared_snapshot_adapter,
                        local_context.shared_snapshot_session,
                        local_context.shared_snapshot_stage_key,
                    )
                self._count("mla_off.snapshot_contexts")
                local_context, snapshot_resolution = (
                    self._mla_off_resolve_snapshot_context(
                        local_context, device=device
                    )
                )
                if local_context is None:
                    self._mla_off_poison_context_snapshot(
                        request_binding=request_binding
                    )
                    cleanup_failure = self._mla_off_rollback_snapshot_staging(
                        snapshot_staging
                    )
                    self._mla_off_require_snapshot_rollback_consensus(
                        cleanup_failure=cleanup_failure,
                        snapshot_staging=snapshot_staging,
                        device=device,
                        shared_transaction=bool(
                            snapshot_staging is not None
                            and snapshot_staging[0] == "shared"
                        ),
                    )
                    forward_batch._redknot_mla_off_disabled = True
                    self._mla_off_log_failure(
                        snapshot_resolution,
                        (
                            "snapshot rows or input identity differed across TP ranks"
                            + (f"; {cleanup_failure}" if cleanup_failure else "")
                        ),
                    )
                    return None
                return local_context

            explicit_refresh_layer, context_safety_reason = (
                self._mla_off_restore_safety(
                    plan=plan,
                    layer_id=layer_id,
                    forward_batch=forward_batch,
                )
            )
            refresh_layer = explicit_refresh_layer or bool(context_safety_reason)
            if context_safety_reason:
                self._count(f"mla_off.{context_safety_reason}")
            if not isinstance(local_plan_digest, tuple):
                raise ValueError("MLA-off plan has no certified digest")
            plan_digest = local_plan_digest
            content_cache_key = (
                self._mla_off_tensors_digest(positions_cpu, input_ids_cpu)
                if position_key is None or input_key is None
                else None
            )
            layout_key = (
                getattr(
                    forward_batch,
                    "_redknot_mla_off_forward_generation",
                    None,
                ),
                id(plan),
                plan_digest,
                position_key,
                input_key,
                content_cache_key,
            )
            cached_layout = getattr(
                forward_batch, "_redknot_mla_off_restore_layout", None
            )
            layout_certificate = None
            if refresh_layer:
                restore_rows, reusable_cpu = build_restore_rows(
                    plan=plan,
                    positions_cpu=positions_cpu,
                    refresh_layer=True,
                )
                reuse_mask_digest = (0, 0)
                reused_count = 0
                dirty_rows_cpu = torch.arange(
                    positions_cpu.numel(), dtype=torch.long
                )
                segments_by_hash = {}
            elif (
                isinstance(cached_layout, _MLAOffRestoreLayoutCertificate)
                and cached_layout.layout_key == layout_key
            ):
                layout_certificate = cached_layout
                restore_rows = layout_certificate.restore_rows
                reusable_cpu = layout_certificate.reusable_cpu
                reuse_mask_digest = layout_certificate.reuse_mask_digest
                reused_count = layout_certificate.reused_count
                dirty_rows_cpu = layout_certificate.dirty_rows_cpu
                segments_by_hash = layout_certificate.segments_by_hash
                layout_certificate.validate(
                    layer_id=layer_id,
                    layout_key=layout_key,
                    reusable_cpu=reusable_cpu,
                    dirty_rows_cpu=dirty_rows_cpu,
                    reuse_mask_digest=reuse_mask_digest,
                    q_rows=positions_cpu.numel(),
                )
            else:
                restore_rows, reusable_cpu = build_restore_rows(
                    plan=plan,
                    positions_cpu=positions_cpu,
                    refresh_layer=False,
                )
                reuse_mask_digest = self._mla_off_tensors_digest(
                    positions_cpu,
                    reusable_cpu,
                    input_ids_cpu,
                )
                if (
                    reusable_cpu.ndim != 1
                    or reusable_cpu.dtype != torch.bool
                    or reusable_cpu.device.type != "cpu"
                    or int(reusable_cpu.numel()) != int(positions_cpu.numel())
                ):
                    raise ValueError("MLA-off reusable-row bitmap is inconsistent")
                reused_count = int(reusable_cpu.sum().item())
                dirty_rows_cpu = torch.nonzero(
                    ~reusable_cpu, as_tuple=False
                ).flatten()
                if reused_count + int(dirty_rows_cpu.numel()) != int(
                    positions_cpu.numel()
                ):
                    raise ValueError(
                        "MLA-off clean/dirty rows do not partition input"
                    )
                descriptor_coverage = torch.zeros_like(reusable_cpu)
                for rows in restore_rows:
                    output_rows_cpu = rows.output_rows_cpu
                    local_positions_cpu = rows.local_positions_cpu
                    if (
                        output_rows_cpu.ndim != 1
                        or output_rows_cpu.dtype != torch.long
                        or output_rows_cpu.device.type != "cpu"
                        or local_positions_cpu.ndim != 1
                        or local_positions_cpu.dtype != torch.long
                        or local_positions_cpu.device.type != "cpu"
                        or output_rows_cpu.numel()
                        != local_positions_cpu.numel()
                    ):
                        raise ValueError("MLA-off row descriptor is malformed")
                    if output_rows_cpu.numel() and (
                        int(output_rows_cpu.min().item()) < 0
                        or int(output_rows_cpu.max().item())
                        >= int(reusable_cpu.numel())
                        or int(torch.unique(output_rows_cpu).numel())
                        != int(output_rows_cpu.numel())
                        or bool(
                            descriptor_coverage.index_select(
                                0, output_rows_cpu
                            ).any().item()
                        )
                    ):
                        raise ValueError(
                            "MLA-off row descriptors overlap or leave input bounds"
                        )
                    descriptor_coverage[output_rows_cpu] = True
                if not torch.equal(descriptor_coverage, reusable_cpu):
                    raise ValueError(
                        "MLA-off row descriptors do not equal the reusable bitmap"
                    )
                segments_by_hash = {
                    str(segment["seg_hash"]): segment
                    for segment in plan.get("segments", ()) or ()
                }
                missing_segments = tuple(
                    rows.seg_hash
                    for rows in restore_rows
                    if rows.seg_hash not in segments_by_hash
                )
                if missing_segments:
                    raise ValueError(
                        "MLA-off segment metadata is missing for restore rows: "
                        f"{missing_segments}"
                    )
                layout_certificate = _MLAOffRestoreLayoutCertificate(
                    layout_key=layout_key,
                    certified_layer_ids=tuple(
                        self._redknot_mla_off_rank_local_layer_ids
                    ),
                    restore_rows=restore_rows,
                    reusable_cpu=reusable_cpu,
                    reuse_mask_digest=reuse_mask_digest,
                    reused_count=reused_count,
                    dirty_rows_cpu=dirty_rows_cpu,
                    segments_by_hash=segments_by_hash,
                    reusable_identity=_mla_off_control_tensor_identity(
                        reusable_cpu
                    ),
                    dirty_identity=_mla_off_control_tensor_identity(
                        dirty_rows_cpu
                    ),
                )
                layout_certificate.validate(
                    layer_id=layer_id,
                    layout_key=layout_key,
                    reusable_cpu=reusable_cpu,
                    dirty_rows_cpu=dirty_rows_cpu,
                    reuse_mask_digest=reuse_mask_digest,
                    q_rows=positions_cpu.numel(),
                )
                forward_batch._redknot_mla_off_restore_layout = (
                    layout_certificate
                )
            if reused_count == 0:
                intentional_full_local = True
                pending_full_local_context = MLAOffRuntimeContext(
                    mode="full_local",
                    layer_id=int(layer_id),
                    spec=spec,
                    local_head_axes=local_axes,
                    controller=controller,
                    benchmark_request_id=benchmark_request_id,
                    benchmark_forward_id=benchmark_forward_id,
                    benchmark_forward_mode=benchmark_forward_mode,
                    benchmark_q_rows=benchmark_q_rows,
                    input_layout_digest=input_layout_digest,
                    transfer_audit_state=transfer_audit_state,
                )
            else:
                if layout_certificate is None:
                    raise ValueError("MLA-off restore layout certificate is absent")
                use_device_artifact = bool(controller.device_cache_enabled)
                artifact_device = (
                    torch.device(device)
                    if use_device_artifact
                    else torch.device("cpu")
                )
                artifact_dtype = (
                    projection_dtype if use_device_artifact else torch.bfloat16
                )
                offline_artifact = torch.zeros(
                    (positions_cpu.numel(), n_local_groups, o_lora_rank),
                    device=artifact_device,
                    dtype=artifact_dtype,
                )
                for rows in restore_rows:
                    segment = segments_by_hash.get(rows.seg_hash)
                    if segment is None:
                        raise ValueError(
                            f"MLA-off segment metadata missing for {rows.seg_hash!r}"
                        )
                    length = int(segment["length"])
                    token_hash = str(segment.get("token_hash", rows.seg_hash))
                    restore_view = controller.prepare_restore_view(
                        seg_hash=rows.seg_hash,
                        length=length,
                        spec=spec,
                        token_hash=token_hash,
                    )
                    output_rows_cpu = rows.output_rows_cpu
                    verified_tokens = getattr(
                        forward_batch,
                        "_redknot_mla_off_verified_tokens",
                        None,
                    )
                    if verified_tokens is None:
                        verified_tokens = set()
                        forward_batch._redknot_mla_off_verified_tokens = (
                            verified_tokens
                        )
                    token_validation_key = (
                        rows.seg_hash,
                        int(restore_view.commit_epoch),
                        layout_key,
                        input_key,
                    )
                    if token_validation_key not in verified_tokens:
                        controller.validate_view_token_rows(
                            restore_view,
                            local_positions=rows.local_positions_cpu,
                            token_ids=input_ids_cpu.index_select(
                                0, output_rows_cpu
                            ),
                        )
                        verified_tokens.add(token_validation_key)
                    values = controller.gather_from_view(
                        restore_view,
                        local_positions=rows.local_positions_cpu,
                        device=artifact_device,
                        dtype=artifact_dtype,
                        use_device_cache=use_device_artifact,
                        device_indices_certificate=(
                            self._mla_off_layout_device_indices(
                                layout_certificate=layout_certificate,
                                controller=controller,
                                cpu_indices=rows.local_positions_cpu,
                                device=artifact_device,
                                role=(
                                    "artifact_local_positions:"
                                    f"{rows.seg_hash}"
                                ),
                                semantic_digest=reuse_mask_digest,
                                upper_bound=length,
                            )
                            if use_device_artifact
                            else None
                        ),
                        index_semantic_digest=reuse_mask_digest,
                        index_role=(
                            "artifact_local_positions:"
                            f"{rows.seg_hash}"
                        ),
                    )
                    if use_device_artifact:
                        output_certificate = self._mla_off_layout_device_indices(
                            layout_certificate=layout_certificate,
                            controller=controller,
                            cpu_indices=output_rows_cpu,
                            device=artifact_device,
                            role=f"artifact_output_rows:{rows.seg_hash}",
                            semantic_digest=reuse_mask_digest,
                            upper_bound=positions_cpu.numel(),
                        )
                        output_rows = controller.device_indices_from_certificate(
                            output_certificate,
                            cpu_indices=output_rows_cpu,
                            device=artifact_device,
                            role=f"artifact_output_rows:{rows.seg_hash}",
                            semantic_digest=reuse_mask_digest,
                            upper_bound=positions_cpu.numel(),
                        )
                    else:
                        output_rows = output_rows_cpu
                    offline_artifact.index_copy_(0, output_rows, values)
                offline = offline_artifact.to(
                    device=device, dtype=projection_dtype
                )
                # The production CPU-authoritative path assembles all segment
                # rows on CPU and performs this one bulk upload.  Counting the
                # earlier CPU gathers would miss the real transfer entirely.
                controller.record_h2d_tensor(
                    "online_artifact", offline_artifact, torch.device(device)
                )
                online_rows_certificate = self._mla_off_layout_device_indices(
                    layout_certificate=layout_certificate,
                    controller=controller,
                    cpu_indices=dirty_rows_cpu,
                    device=torch.device(device),
                    role="online_local_rows",
                    semantic_digest=reuse_mask_digest,
                    upper_bound=positions_cpu.numel(),
                )
                online_rows_device = controller.device_indices_from_certificate(
                    online_rows_certificate,
                    cpu_indices=dirty_rows_cpu,
                    device=torch.device(device),
                    role="online_local_rows",
                    semantic_digest=reuse_mask_digest,
                    upper_bound=positions_cpu.numel(),
                )
                local_context = MLAOffRuntimeContext(
                    mode="restore",
                    layer_id=int(layer_id),
                    spec=spec,
                    local_head_axes=local_axes,
                    controller=controller,
                    offline_projection=offline,
                    reuse_mask=reusable_cpu,
                    reuse_mask_digest=reuse_mask_digest,
                    online_local_row_indices=online_rows_device,
                    online_local_row_indices_cpu=dirty_rows_cpu,
                    online_local_row_indices_certificate=(
                        online_rows_certificate
                    ),
                    restore_layout_certificate=layout_certificate,
                    reused_row_count=reused_count,
                    online_local_row_count=int(dirty_rows_cpu.numel()),
                    benchmark_request_id=benchmark_request_id,
                    benchmark_forward_id=benchmark_forward_id,
                    benchmark_forward_mode=benchmark_forward_mode,
                    benchmark_q_rows=benchmark_q_rows,
                    transfer_audit_state=transfer_audit_state,
                )
                # All-local compact wo_a is allowed to influence attention
                # row execution only after every mutable tensor and opaque
                # certificate has been checked. Keep this inside the existing
                # local try: a failure becomes ``local_context=None`` and the
                # all-TP restore-resolution vote selects native fallback
                # before any rank enters attention.
                local_context.prevalidate_compact_woa(
                    positions,
                    total_rows=int(q_row_count),
                    device=torch.device(device),
                    projection_dtype=projection_dtype,
                )
        except BaseException as error:
            failure = str(error)
            failure_error = error
            local_context = None

        if restore_requested:
            local_context, restore_resolution = (
                self._mla_off_resolve_restore_context(
                    local_context,
                    intentional_full_local=intentional_full_local,
                    device=device,
                )
            )
            if restore_resolution == "restore_not_ready":
                forward_batch._redknot_mla_off_disabled = True
                self._mla_off_log_request_status(
                    plan=plan,
                    layer_id=layer_id,
                    status="fallback",
                    reason="restore_not_ready",
                    forward_id=benchmark_forward_id,
                    forward_mode=benchmark_forward_mode,
                    q_rows=benchmark_q_rows,
                )
                self._mla_off_log_failure(
                    "restore_not_ready", failure or "another TP rank rejected restore"
                )
                return None
            if restore_resolution == "restore_context_mismatch":
                forward_batch._redknot_mla_off_disabled = True
                self._mla_off_log_request_status(
                    plan=plan,
                    layer_id=layer_id,
                    status="fallback",
                    reason="restore_context_mismatch",
                    forward_id=benchmark_forward_id,
                    forward_mode=benchmark_forward_mode,
                    q_rows=benchmark_q_rows,
                )
                self._mla_off_log_failure(
                    "restore_context_mismatch",
                    "reusable-row availability differed across attention-TP ranks",
                )
                return None
            if restore_resolution == "restore_mask_mismatch":
                forward_batch._redknot_mla_off_disabled = True
                self._mla_off_log_request_status(
                    plan=plan,
                    layer_id=layer_id,
                    status="fallback",
                    reason="restore_mask_mismatch",
                    forward_id=benchmark_forward_id,
                    forward_mode=benchmark_forward_mode,
                    q_rows=benchmark_q_rows,
                )
                self._mla_off_log_failure(
                    "restore_mask_mismatch",
                    "reusable-row policy differed across attention-TP ranks",
                )
                return None
            if restore_resolution == "intentional_full_local":
                self._count("mla_off.intentional_full_local_layers")
                (
                    pending_full_local_context,
                    full_local_resolution,
                ) = self._mla_off_resolve_snapshot_context(
                    pending_full_local_context, device=device
                )
                if pending_full_local_context is None:
                    forward_batch._redknot_mla_off_disabled = True
                    self._mla_off_log_request_status(
                        plan=plan,
                        layer_id=layer_id,
                        status="fallback",
                        reason="full_local_layout_mismatch",
                        forward_id=benchmark_forward_id,
                        forward_mode=benchmark_forward_mode,
                        q_rows=benchmark_q_rows,
                    )
                    self._mla_off_log_failure(
                        "full_local_layout_mismatch",
                        full_local_resolution
                        or "intentional full-local input layout differed",
                    )
                    return None
                self._mla_off_log_request_status(
                    plan=plan,
                    layer_id=layer_id,
                    status="full_local",
                    reason=(
                        "refresh_layer"
                        if explicit_refresh_layer
                        else context_safety_reason or "no_reusable_rows"
                    ),
                    forward_id=benchmark_forward_id,
                    forward_mode=benchmark_forward_mode,
                    q_rows=benchmark_q_rows,
                )
                return pending_full_local_context
            self._count("mla_off.restore_contexts")
            self._count("mla_off.reused_rows", local_context.reused_row_count)
            self._set_rank_local_q_authorization(
                forward_batch, layer_id=layer_id, context=local_context
            )
            return local_context

        if mode == "snapshot" and failure:
            self._mla_off_poison_context_snapshot(
                request_binding=context_snapshot_request_binding
            )
            prior_rollback_failure = (
                self._mla_off_incomplete_snapshot_rollback_detail(
                    failure_error
                )
            )
            staging_rollback_failure = self._mla_off_rollback_snapshot_staging(
                snapshot_staging
            )
            cleanup_failure = "; ".join(
                detail
                for detail in (
                    prior_rollback_failure,
                    staging_rollback_failure,
                )
                if detail
            )
            if cleanup_failure:
                failure = f"{failure}; {cleanup_failure}"
            _, snapshot_resolution = self._mla_off_resolve_snapshot_context(
                None, device=device
            )
            self._mla_off_require_snapshot_rollback_consensus(
                cleanup_failure=cleanup_failure,
                snapshot_staging=snapshot_staging,
                device=device,
                shared_transaction=bool(
                    prior_rollback_failure
                    or (
                        snapshot_staging is not None
                        and snapshot_staging[0] == "shared"
                    )
                ),
            )
        if failure:
            forward_batch._redknot_mla_off_disabled = True
            self._mla_off_log_failure(
                snapshot_resolution or "snapshot_not_ready", failure
            )
        return None

    def _mla_off_validate_restore_row_metadata(
        self,
        *,
        q: torch.Tensor,
        layer_id: int,
        reuse_mask,
        online_rows,
        online_rows_cpu,
        online_rows_certificate,
        restore_layout_certificate,
        controller,
        reuse_mask_digest: Tuple[int, int],
        reused_row_count: int,
        online_row_count: int,
    ) -> torch.Tensor:
        """Validate restore row metadata before the coordinated ready vote."""

        if (
            reuse_mask is None
            or online_rows is None
            or online_rows_cpu is None
            or online_rows_certificate is None
            or not isinstance(
                restore_layout_certificate,
                _MLAOffRestoreLayoutCertificate,
            )
            or controller is None
            or reuse_mask.ndim != 1
            or reuse_mask.dtype != torch.bool
            or reuse_mask.device.type != "cpu"
            or int(reuse_mask.numel()) != int(q.shape[0])
            or online_rows.ndim != 1
            or online_rows.dtype != torch.long
            or online_rows.device != q.device
            or online_rows_cpu.ndim != 1
            or online_rows_cpu.dtype != torch.long
            or online_rows_cpu.device.type != "cpu"
        ):
            raise ValueError("MLA-off backend row metadata is incomplete")
        if int(online_rows.numel()) != int(online_row_count):
            raise ValueError("MLA-off online-local row count is inconsistent")
        restore_layout_certificate.validate(
            layer_id=layer_id,
            layout_key=restore_layout_certificate.layout_key,
            reusable_cpu=reuse_mask,
            dirty_rows_cpu=online_rows_cpu,
            reuse_mask_digest=reuse_mask_digest,
            q_rows=q.shape[0],
            reused_count=reused_row_count,
            online_count=online_row_count,
        )
        certified_device_rows = controller.device_indices_from_certificate(
            online_rows_certificate,
            cpu_indices=online_rows_cpu,
            device=q.device,
            role="online_local_rows",
            semantic_digest=reuse_mask_digest,
            upper_bound=q.shape[0],
        )
        if certified_device_rows is not online_rows:
            raise ValueError(
                "MLA-off online-local device rows lost their certificate"
            )
        return certified_device_rows

    def forward(
        self,
        q,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        mla_off_context = kwargs.pop("mla_off_context", None)
        diagnostic_ablation = str(
            getattr(mla_off_context, "diagnostic_ablation", "full") or "full"
        )
        diagnostic_shared_only = diagnostic_ablation == "shared_only"

        from sglang.srt.layers.attention.redknot.dsv4_sparse_q_runtime import (
            PackedSparseQProjection,
        )

        packed_sparse_q = isinstance(q, PackedSparseQProjection)
        forward_transaction = getattr(
            mla_off_context,
            "_redknot_forward_composite_transaction",
            None,
        )
        forward_transaction_committed = bool(
            isinstance(
                forward_transaction, _RedKnotForwardCompositeTransaction
            )
            and forward_transaction.coordinator is not None
            and forward_transaction.coordinator.committed
        )

        def sparse_q_is_committed() -> bool:
            return bool(
                mla_off_context is not None
                and getattr(mla_off_context, "sparse_q_committed", False)
                and not diagnostic_shared_only
            )

        if packed_sparse_q and not sparse_q_is_committed():
            raise RuntimeError(
                "packed sparse-Q cannot enter attention before composite commit"
            )

        if sparse_q_is_committed() and not bool(
            getattr(
                mla_off_context,
                "sparse_q_backend_preflight_complete",
                False,
            )
        ):
            raise RuntimeError(
                "sparse-Q reached attention without collective backend preflight"
            )

        def run_native(
            *, save_kv_cache_override: Optional[bool] = None
        ) -> torch.Tensor:
            return self._native_forward(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=compress_ratio,
                save_kv_cache=(
                    save_kv_cache
                    if save_kv_cache_override is None
                    else bool(save_kv_cache_override)
                ),
                attn_sink=attn_sink,
                kwargs=kwargs,
            )

        def native_fallback() -> torch.Tensor:
            if sparse_q_is_committed() or forward_transaction_committed:
                raise RuntimeError(
                    "native attention fallback is forbidden after the "
                    "forward composite cache/Q certificate"
                )
            output = run_native()
            if mla_off_context is not None:
                applied_count = self._mla_off_vote_count(False, q.device)
                if applied_count != 0:
                    raise RuntimeError(
                        "MLA-off backend policy application differed across TP ranks"
                    )
            return output

        # ``global`` is intentionally bit-for-bit the native DSV4 path.
        if self.redknot_mla_pass_mode == "global":
            return native_fallback()

        self._count("forwards")
        self._maybe_upgrade_forward_metadata()
        core_attn_metadata = self.forward_metadata.core_attn_metadata
        if not isinstance(core_attn_metadata, DSV4AttnMetadata):
            return native_fallback()

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        layer_id = layer.layer_id
        if (
            forward_batch.forward_mode.is_draft_extend(include_v2=True)
            or layer_id >= self.redknot_mla_head_cfg.num_layers
        ):
            # D-Spark owns auxiliary ratio-0 draft layers, not target policies.
            return native_fallback()

        # Hard execution fence for DeepSeek-V4-Flash-0731. These six layers
        # bypass all logical-head splitting and MLA-off artifact handling.
        if layer_id in _PURE_HEADSPLIT_DENSE_LAYER_IDS:
            self._count("path.pure_mla_dense_native")
            return native_fallback()

        if diagnostic_shared_only:
            shared_only_full_local = bool(
                mla_off_context is not None
                and getattr(mla_off_context, "is_full_local", False)
            )
            if shared_only_full_local:
                if not (
                    str(
                        getattr(
                            mla_off_context,
                            "intentional_full_local_reason",
                            "",
                        )
                    )
                    == "query_suffix_only"
                    and int(
                        getattr(mla_off_context, "reused_row_count", -1)
                    )
                    == 0
                    and not tuple(
                        getattr(
                            mla_off_context, "shared_restore_states", ()
                        )
                        or ()
                    )
                    and not packed_sparse_q
                ):
                    raise RuntimeError(
                        "shared_only full-local diagnostic is not a certified "
                        "query suffix"
                    )
            elif not (
                self._mla_off_composite_committed(mla_off_context)
                and bool(
                    getattr(mla_off_context, "shared_restore_applied", False)
                )
                and not packed_sparse_q
            ):
                raise RuntimeError(
                    "shared_only diagnostic requires committed shared restore "
                    "and a complete online Q tensor"
                )

        assert k is v, "DeepseekV4 shares k and v"
        assert attn_sink is not None
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if sparse_q_is_committed():
            validate_sparse_q = getattr(
                mla_off_context, "validate_sparse_q_commit", None
            )
            if not callable(validate_sparse_q):
                raise RuntimeError("sparse-Q context has no validation hook")
            if not packed_sparse_q:
                raise ValueError(
                    "committed sparse-Q must enter attention as a packed "
                    "rank-local projection"
                )
            validate_sparse_q(q)

        committed_backend_preflight = bool(
            sparse_q_is_committed()
            and getattr(
                mla_off_context,
                "sparse_q_backend_preflight_complete",
                False,
            )
        )

        if not packed_sparse_q and q.ndim == 3:
            q = q.unsqueeze(1)

        total_heads = self.redknot_mla_head_cfg.num_attention_heads
        plan = self._redknot_dual_layer_plans[layer_id]
        if self.redknot_mla_pass_mode == "headwise" and not plan.local_groups:
            # There is no head decoupling to execute.  Native FlashMLA is both
            # exact and substantially faster than the arbitrary-head Triton
            # path for an all-global layer (notably the dense prefix layers).
            self._count("path.headwise_all_global_native")
            return native_fallback()
        owned_view = None
        if self.redknot_mla_pass_mode == "headwise":
            owned_view = self._headwise_owned_view(q.shape[2], layer.tp_q_head_num)
            if owned_view is None:
                self._count("path.headwise_native_fallback")
                return native_fallback()
        elif q.shape[2] != total_heads:
            # FlashMLA oracle requires the full logical-head view.
            return native_fallback()

        if (
            self.redknot_mla_pass_mode == "headwise"
            and bool(getattr(self, "_redknot_reuse_heads_full_scope", False))
            and int(q.shape[2]) == int(total_heads)
            and (
                mla_off_context is None
                or not getattr(mla_off_context, "is_restore", False)
                or getattr(mla_off_context, "is_full_local", False)
            )
        ):
            # Reuse eligibility alone must not replace native attention with an
            # experimental kernel.  Ordinary requests and snapshot requests
            # therefore use native DSV4 FlashMLA exactly.  The arbitrary-head
            # path is needed only for an actual restore, where clean reusable
            # head rows can be omitted and merged with the offline projection.
            assert owned_view is not None
            self._record_path(
                "headwise_reuse_full_scope_native",
                layer_id=layer_id,
                plan=plan,
                logical_heads=owned_view[0],
            )
            output = run_native()
            if mla_off_context is not None:
                applied_count = self._mla_off_vote_count(True, q.device)
                if applied_count != int(self._redknot_tp_size):
                    raise RuntimeError(
                        "MLA-off backend policy application differed across TP ranks"
                    )
                mla_off_context.backend_applied = True
                if getattr(mla_off_context, "is_full_local", False):
                    local_head_count = len(mla_off_context.local_head_axes)
                    owned_head_count = len(
                        mla_off_context.spec.owned_logical_heads
                    )
                    q_rows = int(
                        getattr(mla_off_context, "benchmark_q_rows", q.shape[0])
                    )
                    self._mla_off_record_runtime_rows(
                        request_id=str(
                            getattr(
                                mla_off_context, "benchmark_request_id", ""
                            )
                            or ""
                        ),
                        forward_id=str(
                            getattr(
                                mla_off_context, "benchmark_forward_id", ""
                            )
                            or ""
                        ),
                        forward_mode=str(
                            getattr(
                                mla_off_context,
                                "benchmark_forward_mode",
                                "unknown",
                            )
                            or "unknown"
                        ),
                        q_rows=q_rows,
                        layer_id=layer_id,
                        reused_local_head_rows=0,
                        online_local_head_rows=q_rows * local_head_count,
                        online_global_head_rows=(
                            q_rows * (owned_head_count - local_head_count)
                        ),
                        mla_off_context=mla_off_context,
                    )
            return output

        # The shared packed KV is written/restored once.  Every logical-head
        # pass below receives the same cache tensors; no [layer, head] KV copies
        # are materialized.
        if save_kv_cache:
            self.store_cache(layer_id, k, forward_batch)
        if envs.SGLANG_REDKNOT_OFFLINE_REUSE.get():
            self._maybe_redknot_reuse_hook(layer_id, forward_batch, token_to_kv_pool)

        swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
        swa_window_size = token_to_kv_pool.swa_window_size
        k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
        swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
            swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
        )

        extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
        if compress_ratio == 4:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_attn_metadata.c4_sparse_page_indices
            extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
        elif compress_ratio == 128:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_attn_metadata.c128_page_indices
            extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

        # Legacy locality ablation knob.  The current indexer returns an
        # unordered Top-512 set, so shortening its length is not a valid Top-K
        # refinement and is rejected by the accuracy-first path below.
        c4_clamp = os.environ.get("REDKNOT_C4_TOPK_CLAMP", "")
        if c4_clamp and extra_topk_lengths is not None:
            if bool(getattr(self, "_redknot_reuse_heads_full_scope", False)):
                raise ValueError(
                    "REDKNOT_C4_TOPK_CLAMP is unsafe in accuracy-first mode: "
                    "DSV4 Top-512 indices have no score-sorted-prefix contract"
                )
            try:
                clamp_topk = int(c4_clamp)
                if clamp_topk > 0:
                    extra_topk_lengths = torch.clamp(extra_topk_lengths, max=clamp_topk)
            except ValueError:
                pass

        if extra_k_cache is not None:
            page_sizes = {
                4: token_to_kv_pool.page_size // 4,
                128: token_to_kv_pool.page_size // 128,
            }
            extra_k_cache = extra_k_cache[
                :, : page_sizes[compress_ratio] * k_cache_total_dim
            ].view(
                extra_k_cache.shape[0],
                page_sizes[compress_ratio],
                1,
                k_cache_total_dim,
            )

        swa_page_indices = core_attn_metadata.swa_page_indices
        swa_topk_lengths = core_attn_metadata.swa_topk_lengths
        if self.mtp_enabled:
            if swa_page_indices.shape[0] != q.shape[0]:
                swa_page_indices = _pad_tensor_to_size(
                    swa_page_indices, q.shape[0], value=0
                )
            if swa_topk_lengths.shape[0] != q.shape[0]:
                swa_topk_lengths = _pad_tensor_to_size(
                    swa_topk_lengths, q.shape[0], value=1
                )

        if swa_page_indices.ndim == 2:
            swa_page_indices = swa_page_indices.unsqueeze(1)
        if extra_indices is not None and extra_indices.ndim == 2:
            extra_indices = extra_indices.unsqueeze(1)
        assert swa_page_indices.shape[-1] % 64 == 0
        if extra_indices is not None:
            assert extra_indices.shape[-1] % 64 == 0

        if self.redknot_mla_pass_mode == "headwise":
            assert owned_view is not None
            mla_off_reuse_mask = None
            mla_off_online_local_rows = None
            mla_off_online_local_rows_cpu = None
            mla_off_online_local_rows_certificate = None
            mla_off_restore_layout_certificate = None
            mla_off_controller = None
            mla_off_reuse_mask_digest = (0, 0)
            mla_off_reused_row_count = 0
            mla_off_online_local_row_count = q.shape[0]
            mla_off_benchmark_request_id = ""
            mla_off_benchmark_forward_id = ""
            mla_off_benchmark_forward_mode = "unknown"
            mla_off_benchmark_q_rows = int(q.shape[0])
            mla_off_runtime_rows = {}
            mla_off_compact_woa_prevalidated = False
            mla_off_restore_metadata_prevalidated = False
            restore_precheck_required = bool(
                mla_off_context is not None
                and getattr(mla_off_context, "is_restore", False)
            )
            restore_precheck_ok = True
            restore_precheck_failure = ""
            if mla_off_context is not None and getattr(
                mla_off_context, "is_restore", False
            ):
                try:
                    if int(getattr(mla_off_context, "layer_id", -1)) != int(
                        layer_id
                    ):
                        raise ValueError(
                            "MLA-off context was prepared for another layer"
                        )
                    mla_off_reuse_mask = mla_off_context.reuse_mask
                    mla_off_online_local_rows = (
                        mla_off_context.online_local_row_indices
                    )
                    mla_off_online_local_rows_cpu = (
                        mla_off_context.online_local_row_indices_cpu
                    )
                    mla_off_online_local_rows_certificate = (
                        mla_off_context.online_local_row_indices_certificate
                    )
                    mla_off_restore_layout_certificate = (
                        mla_off_context.restore_layout_certificate
                    )
                    if (
                        getattr(
                            forward_batch,
                            "_redknot_mla_off_restore_layout",
                            None,
                        )
                        is not mla_off_restore_layout_certificate
                    ):
                        raise ValueError(
                            "MLA-off restore context lost its ForwardBatch "
                            "layout certificate"
                        )
                    mla_off_controller = mla_off_context.controller
                    mla_off_reuse_mask_digest = tuple(
                        mla_off_context.reuse_mask_digest
                    )
                    mla_off_reused_row_count = int(
                        mla_off_context.reused_row_count
                    )
                    mla_off_online_local_row_count = int(
                        mla_off_context.online_local_row_count
                    )
                    mla_off_benchmark_request_id = str(
                        getattr(
                            mla_off_context, "benchmark_request_id", ""
                        )
                        or ""
                    )
                    mla_off_benchmark_forward_id = str(
                        getattr(
                            mla_off_context, "benchmark_forward_id", ""
                        )
                        or ""
                    )
                    mla_off_benchmark_forward_mode = str(
                        getattr(
                            mla_off_context,
                            "benchmark_forward_mode",
                            "unknown",
                        )
                        or "unknown"
                    )
                    mla_off_benchmark_q_rows = int(
                        getattr(
                            mla_off_context,
                            "benchmark_q_rows",
                            q.shape[0],
                        )
                    )
                    if mla_off_reused_row_count > 0:
                        certified_rows = mla_off_online_local_rows
                        if not committed_backend_preflight:
                            certified_rows = (
                                self._mla_off_validate_restore_row_metadata(
                                    q=q,
                                    layer_id=layer_id,
                                    reuse_mask=mla_off_reuse_mask,
                                    online_rows=mla_off_online_local_rows,
                                    online_rows_cpu=(
                                        mla_off_online_local_rows_cpu
                                    ),
                                    online_rows_certificate=(
                                        mla_off_online_local_rows_certificate
                                    ),
                                    restore_layout_certificate=(
                                        mla_off_restore_layout_certificate
                                    ),
                                    controller=mla_off_controller,
                                    reuse_mask_digest=(
                                        mla_off_reuse_mask_digest
                                    ),
                                    reused_row_count=(
                                        mla_off_reused_row_count
                                    ),
                                    online_row_count=(
                                        mla_off_online_local_row_count
                                    ),
                                )
                            )
                        if certified_rows is not mla_off_online_local_rows:
                            raise ValueError(
                                "MLA-off restore metadata returned different "
                                "online rows"
                            )
                        mla_off_restore_metadata_prevalidated = True
                    if bool(
                        getattr(
                            mla_off_context,
                            "requires_compact_woa_preflight",
                            False,
                        )
                    ):
                        validate_compact = getattr(
                            mla_off_context,
                            "prevalidated_rows_for_attention",
                            None,
                        )
                        if not callable(validate_compact):
                            raise ValueError(
                                "MLA-off all-local restore has no compact "
                                "preflight"
                            )
                        certified_rows = validate_compact(
                            total_rows=int(q.shape[0]),
                            device=q.device,
                            projection_dtype=q.dtype,
                        )
                        if certified_rows is not mla_off_online_local_rows:
                            raise ValueError(
                                "MLA-off compact preflight returned different "
                                "rows"
                            )
                        mla_off_compact_woa_prevalidated = True
                except Exception as error:
                    restore_precheck_ok = False
                    restore_precheck_failure = str(error)
            if diagnostic_shared_only:
                # The composite transaction remains authoritative for shared
                # cache restoration, but this attribution arm intentionally
                # consumes neither its sparse-Q nor its local-row omission.
                # Present an all-online head geometry to _forward_headwise.
                mla_off_reuse_mask = None
                mla_off_online_local_rows = None
                mla_off_online_local_rows_cpu = None
                mla_off_online_local_rows_certificate = None
                mla_off_restore_layout_certificate = None
                mla_off_controller = None
                mla_off_reuse_mask_digest = (0, 0)
                mla_off_reused_row_count = 0
                mla_off_online_local_row_count = int(q.shape[0])
                mla_off_compact_woa_prevalidated = False
                mla_off_restore_metadata_prevalidated = False
                restore_precheck_required = False
            if restore_precheck_required:
                if committed_backend_preflight:
                    restore_collectively_ready = restore_precheck_ok
                    collective_row_skip_enabled = bool(
                        mla_off_compact_woa_prevalidated
                    )
                    collective_compact_woa_enabled = bool(
                        collective_row_skip_enabled
                        and getattr(
                            self, "_redknot_mla_off_compact_woa", False
                        )
                    )
                    restore_collective_reason = (
                        "" if restore_precheck_ok else "committed_precheck_changed"
                    )
                else:
                    (
                        restore_collectively_ready,
                        collective_row_skip_enabled,
                        collective_compact_woa_enabled,
                        restore_collective_reason,
                    ) = self._mla_off_resolve_pre_attention_restore(
                        local_ready=restore_precheck_ok,
                        local_compact_eligible=(
                            mla_off_compact_woa_prevalidated
                        ),
                        local_compact_requested=bool(
                            getattr(
                                self, "_redknot_mla_off_compact_woa", False
                            )
                        ),
                        device=q.device,
                    )
                if not restore_collectively_ready:
                    if bool(
                        getattr(
                            mla_off_context, "sparse_q_committed", False
                        )
                    ):
                        raise RuntimeError(
                            "MLA-off restore precheck failed after sparse-Q "
                            "commit; native fallback is no longer safe"
                        )
                    forward_batch._redknot_mla_off_disabled = True
                    mla_off_context.backend_applied = False
                    self._mla_off_log_failure(
                        restore_collective_reason,
                        restore_precheck_failure
                        or (
                            "compact eligibility differed across TP ranks"
                            if restore_collective_reason
                            == "compact_eligibility_mismatch"
                            else "another TP rank rejected restore metadata"
                        ),
                    )
                    # ``store_cache`` already ran above. Native attention must
                    # consume that cache without writing the same KV a second
                    # time. The coordinated vote guarantees every rank takes
                    # this branch before any RedKnot attention launch.
                    return run_native(save_kv_cache_override=False)
                mla_off_context.collective_compact_enabled = (
                    collective_row_skip_enabled
                )
                mla_off_context.collective_compact_woa_enabled = (
                    collective_compact_woa_enabled
                )
                mla_off_compact_woa_prevalidated = (
                    collective_row_skip_enabled
                )
            # Every rank with an agreed MLA-off context must reach the fixed
            # attention_application vote, including a rank whose local Triton
            # launch, scatter, or runtime-row validation raised. Otherwise a
            # peer can block forever in the old post-call applied-count vote.
            irreversible_pipeline = bool(
                self._mla_off_composite_committed(mla_off_context)
                or getattr(
                    mla_off_context, "diagnostic_irreversible", False
                )
            )
            attention_application_error = (
                getattr(mla_off_context, "composite_pipeline_error", None)
                if irreversible_pipeline
                else None
            )
            output = None
            try:
                if attention_application_error is not None:
                    raise attention_application_error
                output = self._forward_headwise(
                    q=q,
                    attn_sink=attn_sink,
                    plan=plan,
                    layer_id=layer_id,
                    owned_view=owned_view,
                    swa_k_cache=swa_k_cache,
                    swa_page_indices=swa_page_indices,
                    swa_topk_lengths=swa_topk_lengths,
                    extra_k_cache=extra_k_cache,
                    extra_indices=extra_indices,
                    extra_topk_lengths=extra_topk_lengths,
                    mla_off_reuse_mask=mla_off_reuse_mask,
                    mla_off_online_local_rows=mla_off_online_local_rows,
                    mla_off_online_local_rows_cpu=mla_off_online_local_rows_cpu,
                    mla_off_online_local_rows_certificate=(
                        mla_off_online_local_rows_certificate
                    ),
                    mla_off_restore_layout_certificate=(
                        mla_off_restore_layout_certificate
                    ),
                    mla_off_controller=mla_off_controller,
                    mla_off_reuse_mask_digest=mla_off_reuse_mask_digest,
                    mla_off_reused_row_count=mla_off_reused_row_count,
                    mla_off_online_local_row_count=(
                        mla_off_online_local_row_count
                    ),
                    mla_off_benchmark_request_id=mla_off_benchmark_request_id,
                    mla_off_benchmark_forward_id=mla_off_benchmark_forward_id,
                    mla_off_benchmark_forward_mode=(
                        mla_off_benchmark_forward_mode
                    ),
                    mla_off_benchmark_q_rows=mla_off_benchmark_q_rows,
                    mla_off_runtime_rows_out=mla_off_runtime_rows,
                    mla_off_compact_woa_prevalidated=(
                        mla_off_compact_woa_prevalidated
                    ),
                    mla_off_restore_metadata_prevalidated=(
                        mla_off_restore_metadata_prevalidated
                    ),
                )
                if mla_off_context is not None and getattr(
                    mla_off_context, "is_restore", False
                ):
                    if diagnostic_shared_only:
                        local_head_count = len(mla_off_context.local_head_axes)
                        owned_head_count = len(
                            mla_off_context.spec.owned_logical_heads
                        )
                        mla_off_runtime_rows.update(
                            {
                                "reused_local_head_rows": 0,
                                "online_local_head_rows": (
                                    int(q.shape[0]) * local_head_count
                                ),
                                "online_global_head_rows": (
                                    int(q.shape[0])
                                    * (owned_head_count - local_head_count)
                                ),
                            }
                        )
                    required_row_keys = {
                        "reused_local_head_rows",
                        "online_local_head_rows",
                        "online_global_head_rows",
                    }
                    if set(mla_off_runtime_rows) != required_row_keys:
                        raise RuntimeError(
                            "restore completed without exact runtime row "
                            "accounting"
                        )
                if output is None:
                    raise RuntimeError(
                        "RedKnot attention_application returned no output"
                    )
            except BaseException as error:
                attention_application_error = error
            if irreversible_pipeline and attention_application_error is not None:
                # The composite data path has no TP collective between dirty
                # cache builders and wo_b.  Preserve this rank's first failure,
                # return a shape-correct value that is never consumed after the
                # final vote rejects the pipeline, and let healthy ranks finish
                # attention + merge without an intermediate host sync.
                mla_off_context.record_pipeline_error(
                    attention_application_error
                )
                output = (
                    q.new_attention_output(self.head_dim_v).squeeze(1)
                    if packed_sparse_q
                    else q.new_zeros((*q.shape[:-1], self.head_dim_v)).squeeze(1)
                )
                mla_off_context.backend_applied = True
                self._count("mla_off.irreversible_attention_local_failures")
                return output
            if mla_off_context is not None:
                if not irreversible_pipeline:
                    # Legacy z-only restore retains its post-attention vote.
                    # Composite shared-latent restore joins the dirty builder,
                    # attention, and merge outcome in one later rendezvous.
                    application_ok, application_reason = (
                        self.resolve_mla_off_attention_application(
                            local_success=attention_application_error is None,
                            device=q.device,
                        )
                    )
                    if not application_ok:
                        raise RuntimeError(
                            "MLA-off attention_application failed on at least one "
                            f"TP rank: {application_reason}"
                        ) from attention_application_error
                    if attention_application_error is not None:
                        raise RuntimeError(
                            "MLA-off attention_application vote accepted a local "
                            "failure"
                        ) from attention_application_error
                mla_off_context.backend_applied = True
                if getattr(mla_off_context, "is_restore", False):
                    self._mla_off_record_runtime_rows(
                        request_id=str(
                            getattr(
                                mla_off_context, "benchmark_request_id", ""
                            )
                            or ""
                        ),
                        forward_id=str(
                            getattr(
                                mla_off_context, "benchmark_forward_id", ""
                            )
                            or ""
                        ),
                        forward_mode=str(
                            getattr(
                                mla_off_context,
                                "benchmark_forward_mode",
                                "unknown",
                            )
                            or "unknown"
                        ),
                        q_rows=int(
                            getattr(
                                mla_off_context,
                                "benchmark_q_rows",
                                q.shape[0],
                            )
                        ),
                        layer_id=layer_id,
                        mla_off_context=mla_off_context,
                        **mla_off_runtime_rows,
                    )
                elif getattr(mla_off_context, "is_full_local", False):
                    local_head_count = len(mla_off_context.local_head_axes)
                    owned_head_count = len(
                        mla_off_context.spec.owned_logical_heads
                    )
                    q_rows = int(
                        getattr(mla_off_context, "benchmark_q_rows", q.shape[0])
                    )
                    self._mla_off_record_runtime_rows(
                        request_id=str(
                            getattr(
                                mla_off_context, "benchmark_request_id", ""
                            )
                            or ""
                        ),
                        forward_id=str(
                            getattr(
                                mla_off_context, "benchmark_forward_id", ""
                            )
                            or ""
                        ),
                        forward_mode=str(
                            getattr(
                                mla_off_context,
                                "benchmark_forward_mode",
                                "unknown",
                            )
                            or "unknown"
                        ),
                        q_rows=q_rows,
                        layer_id=layer_id,
                        reused_local_head_rows=0,
                        online_local_head_rows=q_rows * local_head_count,
                        online_global_head_rows=(
                            q_rows * (owned_head_count - local_head_count)
                        ),
                        mla_off_context=mla_off_context,
                    )
            elif attention_application_error is not None:
                raise attention_application_error
            return output

        return self._forward_flashmla_oracle(
            q=q,
            attn_sink=attn_sink,
            plan=plan,
            layer_id=layer_id,
            compress_ratio=compress_ratio,
            core_attn_metadata=core_attn_metadata,
            swa_k_cache=swa_k_cache,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            extra_k_cache=extra_k_cache,
            extra_indices=extra_indices,
            extra_topk_lengths=extra_topk_lengths,
        )

    def _forward_flashmla_oracle(
        self,
        *,
        q: torch.Tensor,
        attn_sink: torch.Tensor,
        plan: _DualLayerPassPlan,
        layer_id: int,
        compress_ratio: Literal[0, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        swa_k_cache: torch.Tensor,
        swa_page_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        extra_k_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        import flash_mla

        if self.redknot_mla_pass_mode == "local":
            forced_window = _validate_forced_local_window(
                self.redknot_mla_head_cfg.local_default_window,
                self._redknot_swa_capacity,
            )
            effective_plan = _DualLayerPassPlan(
                local_groups=((forced_window, tuple(range(q.shape[2]))),),
                global_heads=(),
                promoted_heads=(),
            )
            path = "forced_local"
        else:
            effective_plan = plan
            if bool(getattr(self, "_redknot_reuse_heads_full_scope", False)):
                path = "dual_reuse_full_scope"
            elif effective_plan.local_groups and effective_plan.global_heads:
                path = "dual_mixed"
            elif effective_plan.local_groups:
                path = "dual_all_local"
            else:
                path = "dual_all_global"
        self._record_path(
            path,
            layer_id=layer_id,
            plan=effective_plan,
            logical_heads=tuple(range(q.shape[2])),
        )

        out = q.new_zeros((*q.shape[:-1], self.head_dim_v))

        def run_flashmla(
            *,
            topk_lengths: torch.Tensor,
            metadata,
            use_global_scope: bool,
        ) -> torch.Tensor:
            return flash_mla.flash_mla_with_kvcache(
                q=q,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=metadata,
                softmax_scale=self.softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_page_indices,
                topk_length=topk_lengths,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache if use_global_scope else None,
                extra_indices_in_kvcache=(extra_indices if use_global_scope else None),
                extra_topk_length=(extra_topk_lengths if use_global_scope else None),
            )[0]

        if (
            bool(getattr(self, "_redknot_reuse_heads_full_scope", False))
            and self.redknot_mla_pass_mode != "local"
        ):
            # Dual mode remains a true full-scope FlashMLA oracle when the head
            # policy describes reuse eligibility rather than attention scope.
            out = run_flashmla(
                topk_lengths=swa_topk_lengths,
                metadata=core_attn_metadata.get_flashmla_metadata(compress_ratio),
                use_global_scope=True,
            )
            self._count("global_flashmla_calls")
            local_count = effective_plan.effective_local_heads
            global_count = len(effective_plan.global_heads)
            if local_count:
                self._count("policy_local_head_outputs", local_count)
            if global_count:
                self._count("policy_global_head_outputs", global_count)
            if effective_plan.promoted_heads:
                self._count(
                    "policy_promoted_head_outputs",
                    len(effective_plan.promoted_heads),
                )
            return out.squeeze(1)

        if effective_plan.global_heads:
            global_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)
            o_global = run_flashmla(
                topk_lengths=swa_topk_lengths,
                metadata=global_metadata,
                use_global_scope=True,
            )
            global_idx = torch.tensor(
                effective_plan.global_heads, dtype=torch.long, device=q.device
            )
            out.index_copy_(2, global_idx, o_global.index_select(2, global_idx))
            self._count("global_flashmla_calls")

        for window, local_heads in effective_plan.local_groups:
            local_metadata = _create_flashmla_metadata()
            local_lengths = torch.clamp(swa_topk_lengths, max=window)
            o_local = run_flashmla(
                topk_lengths=local_lengths,
                metadata=local_metadata,
                use_global_scope=False,
            )
            local_idx = torch.tensor(local_heads, dtype=torch.long, device=q.device)
            out.index_copy_(2, local_idx, o_local.index_select(2, local_idx))
            self._count("local_flashmla_calls")

        local_count = effective_plan.effective_local_heads
        global_count = len(effective_plan.global_heads)
        if local_count:
            self._count("policy_local_head_outputs", local_count)
        if global_count:
            self._count("policy_global_head_outputs", global_count)
        if effective_plan.promoted_heads:
            self._count(
                "policy_promoted_head_outputs", len(effective_plan.promoted_heads)
            )
        return out.squeeze(1)

    def _mla_off_forward_global_padded_flashmla_h64(
        self,
        *,
        layer_id: int,
        q_part: torch.Tensor,
        sink_part: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_page_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        extra_k_cache: torch.Tensor,
        extra_indices: torch.Tensor,
        extra_topk_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run one logical global head through the certified H64 provider.

        FlashMLA's Hopper binary is instantiated for 64 physical query heads.
        The extra 63 heads are exact zero padding and are discarded.  One
        persistent Q workspace is reused sequentially across decoder layers;
        creating a 512 MiB zero tensor per layer would erase the measured
        kernel gain and fragment the allocator.  All tensors execute on the
        current model stream, so the next layer's copy cannot overtake the
        preceding FlashMLA read.
        """

        import flash_mla

        if q_part.ndim != 4 or tuple(q_part.shape[1:3]) != (1, 1):
            raise ValueError(
                "padded FlashMLA H64 requires q=[rows,1,1,head_dim]"
            )
        if int(q_part.shape[-1]) != 512 or int(self.head_dim_v) != 512:
            raise ValueError("padded FlashMLA H64 requires DQK=DV=512")
        if sink_part.ndim != 1 or int(sink_part.numel()) != 1:
            raise ValueError("padded FlashMLA H64 requires one attention sink")
        if extra_k_cache is None or extra_indices is None or extra_topk_lengths is None:
            raise ValueError("padded FlashMLA H64 requires the full extra scope")
        capability = tuple(torch.cuda.get_device_capability(q_part.device))
        if not capability or int(capability[0]) != 9:
            raise RuntimeError(
                "padded FlashMLA H64 is certified only for Hopper/SM90, got "
                f"capability={capability}"
            )

        q_rows = int(q_part.shape[0])
        workspace_rows = min(q_rows, int(FLASHMLA_MAX_BATCH_ROWS))
        workspace = getattr(
            self, "_redknot_mla_off_global_q_workspace", None
        )
        workspace_compatible = bool(
            isinstance(workspace, torch.Tensor)
            and workspace.device == q_part.device
            and workspace.dtype == q_part.dtype
            and workspace.ndim == 4
            and tuple(int(value) for value in workspace.shape[1:])
            == (1, 64, 512)
            and int(workspace.shape[0]) >= workspace_rows
        )
        if not workspace_compatible:
            workspace = q_part.new_zeros((workspace_rows, 1, 64, 512))
            self._redknot_mla_off_global_q_workspace = workspace

        sink_padded = sink_part.new_zeros((64,))
        sink_padded[0].copy_(sink_part[0])
        # Do not retain a ``:1`` view of every padded H64 result.  Such a view
        # keeps the complete [rows, 1, 64, 512] allocation alive; a 57K
        # restore split into seven 8K chunks therefore pins roughly 3.5 GiB
        # until the final ``cat``.  Materialize the one logical head directly
        # into its compact destination so each padded result can be released
        # before the next chunk is launched.
        logical_output = q_part.new_empty(
            (q_rows, 1, 1, int(self.head_dim_v))
        )
        row_chunks = 0
        for row_start in range(0, q_rows, int(FLASHMLA_MAX_BATCH_ROWS)):
            row_end = min(q_rows, row_start + int(FLASHMLA_MAX_BATCH_ROWS))
            chunk_rows = row_end - row_start
            q_padded = workspace.narrow(0, 0, chunk_rows)
            q_padded[:, :, :1, :].copy_(q_part[row_start:row_end])

            # FlashMLA binds scheduler metadata to each exact row slice and
            # its top-k values.  A fresh object is required for every slice;
            # only the zero-padded Q workspace is safe to reuse.
            metadata = _create_flashmla_metadata()
            padded_output = flash_mla.flash_mla_with_kvcache(
                q=q_padded,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=metadata,
                softmax_scale=self.softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_page_indices[row_start:row_end],
                topk_length=swa_topk_lengths[row_start:row_end],
                attn_sink=sink_padded,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices[row_start:row_end],
                extra_topk_length=extra_topk_lengths[row_start:row_end],
            )[0]
            expected_shape = (chunk_rows, 1, 64, int(self.head_dim_v))
            if tuple(int(value) for value in padded_output.shape) != expected_shape:
                raise RuntimeError(
                    "padded FlashMLA H64 returned an invalid output shape: "
                    f"{tuple(padded_output.shape)} != {expected_shape}"
                )
            logical_output[row_start:row_end].copy_(
                padded_output[:, :, :1, :]
            )
            row_chunks += 1
        self._count("global_padded_flashmla_h64_calls")
        self._count(
            "global_padded_flashmla_h64_row_chunks", row_chunks
        )
        self._count("global_padded_flashmla_physical_head_rows", q_rows * 64)
        return logical_output

    def _forward_headwise(
        self,
        *,
        q,
        attn_sink: torch.Tensor,
        plan: _DualLayerPassPlan,
        layer_id: int,
        owned_view: Tuple[Tuple[int, ...], Tuple[int, ...]],
        swa_k_cache: torch.Tensor,
        swa_page_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        extra_k_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
        mla_off_reuse_mask: Optional[torch.Tensor] = None,
        mla_off_online_local_rows: Optional[torch.Tensor] = None,
        mla_off_online_local_rows_cpu: Optional[torch.Tensor] = None,
        mla_off_online_local_rows_certificate=None,
        mla_off_restore_layout_certificate: Optional[
            _MLAOffRestoreLayoutCertificate
        ] = None,
        mla_off_controller=None,
        mla_off_reuse_mask_digest: Tuple[int, int] = (0, 0),
        mla_off_reused_row_count: int = 0,
        mla_off_online_local_row_count: int = 0,
        mla_off_benchmark_request_id: str = "",
        mla_off_benchmark_forward_id: str = "",
        mla_off_benchmark_forward_mode: str = "unknown",
        mla_off_benchmark_q_rows: int = 0,
        mla_off_runtime_rows_out: Optional[dict] = None,
        mla_off_compact_woa_prevalidated: bool = False,
        mla_off_restore_metadata_prevalidated: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.triton_decode import (
            triton_fp8_attention_fwd,
        )
        from sglang.srt.layers.attention.redknot.dsv4_sparse_q_runtime import (
            PackedSparseQProjection,
        )

        packed_sparse_q = isinstance(q, PackedSparseQProjection)

        owned_logical, owned_q_axis = owned_view
        logical_to_q_axis = dict(zip(owned_logical, owned_q_axis))
        owned_set = set(owned_logical)
        global_logical = tuple(h for h in plan.global_heads if h in owned_set)
        local_groups = tuple(
            (window, tuple(h for h in heads if h in owned_set))
            for window, heads in plan.local_groups
        )
        local_groups = tuple((w, hs) for w, hs in local_groups if hs)
        owned_plan = _DualLayerPassPlan(
            local_groups=local_groups,
            global_heads=global_logical,
            promoted_heads=tuple(
                item for item in plan.promoted_heads if item[0] in owned_set
            ),
        )
        if owned_plan.local_groups and owned_plan.global_heads:
            path = "headwise_mixed"
        elif owned_plan.local_groups:
            path = "headwise_all_local"
        else:
            path = "headwise_all_global"
        self._record_path(
            path,
            layer_id=layer_id,
            plan=owned_plan,
            logical_heads=owned_logical,
        )

        if attn_sink.numel() == self.redknot_mla_head_cfg.num_attention_heads:
            sink_uses_logical_axis = True
        elif attn_sink.numel() == q.shape[2]:
            sink_uses_logical_axis = False
        else:
            raise ValueError(
                "RedKnot headwise attention sink has incompatible head count: "
                f"sink={attn_sink.numel()} q={q.shape[2]} "
                f"logical={self.redknot_mla_head_cfg.num_attention_heads}"
            )

        out = (
            q.new_attention_output(self.head_dim_v)
            if packed_sparse_q
            else q.new_zeros((*q.shape[:-1], self.head_dim_v))
        )
        mla_off_active = int(mla_off_reused_row_count) > 0
        all_owned_heads_local = bool(
            not owned_plan.global_heads
            and owned_plan.effective_local_heads == len(owned_logical)
        )
        if (
            mla_off_active
            and all_owned_heads_local
            and not bool(mla_off_compact_woa_prevalidated)
        ):
            raise ValueError(
                "MLA-off all-local attention skip requires compact preflight"
            )
        if mla_off_active:
            if not bool(mla_off_restore_metadata_prevalidated):
                self._mla_off_validate_restore_row_metadata(
                    q=q,
                    layer_id=layer_id,
                    reuse_mask=mla_off_reuse_mask,
                    online_rows=mla_off_online_local_rows,
                    online_rows_cpu=mla_off_online_local_rows_cpu,
                    online_rows_certificate=(
                        mla_off_online_local_rows_certificate
                    ),
                    restore_layout_certificate=(
                        mla_off_restore_layout_certificate
                    ),
                    controller=mla_off_controller,
                    reuse_mask_digest=mla_off_reuse_mask_digest,
                    reused_row_count=mla_off_reused_row_count,
                    online_row_count=mla_off_online_local_row_count,
                )

        # Mixed local/global heads share one Triton launch.  MAIN remains one
        # physical SWA stream: every local head supplies its exact window while
        # global heads consume the full valid row and alone enable EXTRA.  The
        # grouped path below remains the fallback when EXTRA is unavailable and
        # the scopes therefore cannot be represented by a dual-scope launch.
        # The dual-scope kernel is an optimized path, but it is not covered by
        # SGLang's batch-invariant/deterministic contract.  In deterministic
        # mode use the existing grouped global/local launches so repeated
        # requests have a correctness oracle before enabling the fusion.
        reuse_heads_full_scope = bool(
            getattr(self, "_redknot_reuse_heads_full_scope", False)
        )
        can_fuse_scope_heads = (
            not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
            and bool(owned_plan.global_heads)
            and extra_k_cache is not None
            and extra_indices is not None
            and not mla_off_active
            and not reuse_heads_full_scope
        )
        used_fused_scope_heads = False
        if can_fuse_scope_heads:
            q_axis = tuple(logical_to_q_axis[h] for h in owned_logical)
            q_idx = torch.tensor(q_axis, dtype=torch.long, device=q.device)
            sink_axis = owned_logical if sink_uses_logical_axis else q_axis
            sink_idx = torch.tensor(sink_axis, dtype=torch.long, device=q.device)
            global_set = set(owned_plan.global_heads)
            local_window_by_head = {
                head: window
                for window, heads in owned_plan.local_groups
                for head in heads
            }
            extra_head_mask = torch.tensor(
                [head in global_set for head in owned_logical],
                dtype=torch.uint8,
                device=q.device,
            )
            main_topk = int(swa_page_indices.shape[-1])
            main_head_length_values = [
                main_topk if head in global_set else local_window_by_head[head]
                for head in owned_logical
            ]
            invalid_main_lengths = [
                (head, length)
                for head, length in zip(owned_logical, main_head_length_values)
                if length <= 0 or length > main_topk
            ]
            if invalid_main_lengths:
                raise ValueError(
                    "RedKnot per-head MAIN lengths must be within the available "
                    f"index row [1, {main_topk}], got {invalid_main_lengths}"
                )
            main_head_lengths = torch.tensor(
                main_head_length_values,
                dtype=torch.int32,
                device=q.device,
            )
            o_part = triton_fp8_attention_fwd(
                q=q.index_select(2, q_idx),
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                softmax_scale=self.softmax_scale,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths,
                attn_sink=attn_sink.index_select(0, sink_idx),
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                extra_topk_length=extra_topk_lengths,
                extra_head_mask=extra_head_mask,
                main_head_lengths=main_head_lengths,
                force_fused_headwise=True,
            )[0]
            out.index_copy_(2, q_idx, o_part)
            used_fused_scope_heads = True
            self._count("fused_scope_triton_calls")

        def run_group(
            logical_heads: Tuple[int, ...],
            *,
            local_window: Optional[int],
            row_indices: Optional[torch.Tensor] = None,
        ) -> None:
            if not logical_heads:
                return
            if row_indices is not None and int(row_indices.numel()) == 0:
                return
            q_axis = tuple(logical_to_q_axis[h] for h in logical_heads)
            q_idx = torch.tensor(q_axis, dtype=torch.long, device=q.device)
            sink_axis = logical_heads if sink_uses_logical_axis else q_axis
            sink_idx = torch.tensor(sink_axis, dtype=torch.long, device=q.device)
            if packed_sparse_q:
                q_part = q.select(
                    scope="global" if row_indices is None else "local",
                    head_axes=q_axis,
                    row_indices=row_indices,
                )
            else:
                q_rows = q if row_indices is None else q.index_select(0, row_indices)
                q_part = q_rows.index_select(2, q_idx)
            sink_part = attn_sink.index_select(0, sink_idx)
            # In accuracy-first reuse mode, ``local`` means eligible for
            # offline output reuse only.  Dirty/recomputed rows retain the
            # complete native DSV4 candidate scope instead of being truncated
            # to SWA.  This keeps attention semantics independent from the
            # reuse policy.
            use_global_scope = local_window is None or reuse_heads_full_scope
            selected_swa_lengths = (
                swa_topk_lengths
                if row_indices is None
                else swa_topk_lengths.index_select(0, row_indices)
            )
            topk_lengths = (
                selected_swa_lengths
                if use_global_scope
                else torch.clamp(selected_swa_lengths, max=local_window)
            )
            selected_swa_indices = (
                swa_page_indices
                if row_indices is None
                else swa_page_indices.index_select(0, row_indices)
            )
            selected_extra_indices = (
                extra_indices
                if row_indices is None or extra_indices is None
                else extra_indices.index_select(0, row_indices)
            )
            selected_extra_lengths = (
                extra_topk_lengths
                if row_indices is None or extra_topk_lengths is None
                else extra_topk_lengths.index_select(0, row_indices)
            )
            use_padded_global_flashmla = bool(
                getattr(
                    self,
                    "_redknot_mla_off_global_attention_impl",
                    "triton_h1",
                )
                == "padded_flashmla_h64"
                and mla_off_active
                and row_indices is None
                and local_window is None
                and len(logical_heads) == 1
                and int(q_part.shape[2]) == 1
                and int(q_part.shape[0]) >= _MLA_OFF_PADDED_FLASHMLA_MIN_ROWS
            )
            if use_padded_global_flashmla:
                o_part = self._mla_off_forward_global_padded_flashmla_h64(
                    layer_id=layer_id,
                    q_part=q_part,
                    sink_part=sink_part,
                    swa_k_cache=swa_k_cache,
                    swa_page_indices=selected_swa_indices,
                    swa_topk_lengths=topk_lengths,
                    extra_k_cache=extra_k_cache,
                    extra_indices=selected_extra_indices,
                    extra_topk_lengths=selected_extra_lengths,
                )
            else:
                o_part = triton_fp8_attention_fwd(
                    q=q_part,
                    k_cache=swa_k_cache,
                    head_dim_v=self.head_dim_v,
                    softmax_scale=self.softmax_scale,
                    indices=selected_swa_indices,
                    topk_length=topk_lengths,
                    attn_sink=sink_part,
                    extra_k_cache=extra_k_cache if use_global_scope else None,
                    extra_indices_in_kvcache=(
                        selected_extra_indices if use_global_scope else None
                    ),
                    extra_topk_length=(
                        selected_extra_lengths if use_global_scope else None
                    ),
                    force_fused_headwise=True,
                )[0]
            if row_indices is None:
                out.index_copy_(2, q_idx, o_part)
            else:
                # Preserve any global-head values already written for these
                # rows while scattering the dirty local-head result.
                out_rows = out.index_select(0, row_indices)
                out_rows.index_copy_(2, q_idx, o_part)
                out.index_copy_(0, row_indices, out_rows)

        # Run the shared global candidate scope once for all owned global heads,
        # then one SWA-only call per distinct local window.  Across calls the Q
        # head counts sum to this TP rank's owned heads; KV cache storage remains
        # exactly one physical latent stream.
        if not used_fused_scope_heads:
            if reuse_heads_full_scope and not mla_off_active:
                # With no restored rows, all owned heads have identical native
                # DSV4 scope.  Keep them in one launch so a mixed reuse policy
                # is numerically equivalent to the all-global headwise oracle.
                run_group(owned_logical, local_window=None)
                self._count("reuse_full_scope_triton_calls")
            else:
                if owned_plan.global_heads:
                    run_group(owned_plan.global_heads, local_window=None)
                    self._count("global_triton_calls")
                local_execution_groups = owned_plan.local_groups
                if reuse_heads_full_scope and mla_off_active:
                    # Window labels describe reuse eligibility in this mode,
                    # not attention scope.  All dirty reusable heads therefore
                    # share one full-scope call instead of producing different
                    # numerics merely because their offline policy labels use
                    # different historical windows.
                    all_local_heads = tuple(
                        head
                        for _, local_heads in owned_plan.local_groups
                        for head in local_heads
                    )
                    local_execution_groups = (
                        ((0, all_local_heads),) if all_local_heads else ()
                    )
                for window, local_heads in local_execution_groups:
                    run_group(
                        local_heads,
                        local_window=window,
                        row_indices=(
                            mla_off_online_local_rows if mla_off_active else None
                        ),
                    )
                    if not mla_off_active or mla_off_online_local_row_count > 0:
                        self._count("local_triton_calls")

        local_count = owned_plan.effective_local_heads
        global_count = len(owned_plan.global_heads)
        if local_count:
            self._count("policy_local_head_outputs", local_count)
        if global_count:
            self._count("policy_global_head_outputs", global_count)
        if owned_plan.promoted_heads:
            self._count("policy_promoted_head_outputs", len(owned_plan.promoted_heads))
        if mla_off_active:
            reused_local_head_rows = int(mla_off_reused_row_count) * local_count
            online_local_head_rows = (
                int(mla_off_online_local_row_count) * local_count
            )
            online_global_head_rows = int(q.shape[0]) * global_count
            runtime_rows = {
                "reused_local_head_rows": reused_local_head_rows,
                "online_local_head_rows": online_local_head_rows,
                "online_global_head_rows": online_global_head_rows,
            }
            if mla_off_runtime_rows_out is None:
                self._mla_off_record_runtime_rows(
                    request_id=mla_off_benchmark_request_id,
                    forward_id=mla_off_benchmark_forward_id,
                    forward_mode=mla_off_benchmark_forward_mode,
                    q_rows=(
                        int(mla_off_benchmark_q_rows)
                        if int(mla_off_benchmark_q_rows) > 0
                        else int(q.shape[0])
                    ),
                    layer_id=layer_id,
                    **runtime_rows,
                )
            else:
                if mla_off_runtime_rows_out:
                    raise RuntimeError(
                        "MLA-off runtime row accounting was already populated"
                    )
                mla_off_runtime_rows_out.update(runtime_rows)
        self._count("headwise_owned_q_heads", len(owned_logical))
        return out.squeeze(1)


__all__ = [
    "RedKnotMLAAttnBackend",
    "_DualLayerPassPlan",
    "_build_dual_layer_pass_plan",
    "_validate_pure_headsplit_plan_contract",
    "_validate_forced_local_window",
]
