# Copyright 2024-2026 SGLang RedKnot Integration.
"""RedKnot integration package.

Provides head-classified KV cache compression for long-context inference,
ported from the standalone __REDKNOT_V02__ project into sglang's attention stack.

Sub-modules
-----------
- ``head_config``  : Per-(layer, kv_head) strategy classification and config IO.
- ``mask_plan``    : Per-layer dispatch table (head types, sliding windows, sinks).
- ``rope_helper``  : RoPE repositioning utility for offline KV reuse.
- ``offline_cache``: Storage manager for segment-level offline KV cache.
- ``ops_flash``    : FlashAttention-2 backed per-head-type attention kernel.
- ``ops_flash3``   : FlashAttention-3 variant (Hopper SM 9.0+).
- ``prefill``      : Offline prefill helper (HF runtime).
- ``driver``       : Standalone forward driver for benchmarking / validation.
"""

from sglang.srt.layers.attention.redknot.driver import (
    online_forward_segment,
    run_baseline,
    run_redknot,
)
from sglang.srt.layers.attention.redknot.driver_batched import (
    online_forward_segments_batched,
    run_redknot_batched,
    run_redknot_offlinekv,
)
from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
    DeepSeekV4MLAHeadConfig,
    MLAHeadStrategy,
    deepseek_v4_mla_cache_descriptor,
    is_deepseek_v4_mla_config,
)
from sglang.srt.layers.attention.redknot.head_config import (
    DEFAULT_LOCAL_WINDOW,
    DEFAULT_RETRIEVAL_TOP_P,
    DEFAULT_SINK_SIZE,
    HEAD_DENSE,
    HEAD_GLOBAL,
    HEAD_LOCAL,
    HEAD_RETRIEVAL,
    HeadClassConfig,
    HeadStrategy,
)
from sglang.srt.layers.attention.redknot.mask_plan import (
    LayerMaskPlan,
    build_layer_mask_plan,
    group_heads_by_type,
    pad_per_head_sinks,
    q_head_indices,
)
from sglang.srt.layers.attention.redknot.offline_cache import (
    OfflineKVCache,
    OfflineSegment,
    get_global_offline_cache,
    set_global_offline_cache,
)
from sglang.srt.layers.attention.redknot.ops_flash import (
    DEFAULT_Q_CHUNK,
    is_flash_attn_available,
    segment_attention_flash,
)
from sglang.srt.layers.attention.redknot.ops_flash3 import (
    is_fa3_available,
    segment_attention_flash3,
)
from sglang.srt.layers.attention.redknot.ops_flash3_parallel import (
    is_fa3_parallel_available,
    segment_attention_flash3_parallel,
)
from sglang.srt.layers.attention.redknot.prefill import (
    offline_prefill_segment,
    offline_prefill_segments,
)
from sglang.srt.layers.attention.redknot.rope_helper import RoPEHelper
from sglang.srt.layers.attention.redknot.segpaged import (
    POLICY_GLOBAL,
    POLICY_LOCAL,
    HeadSegment,
    SegmentPageTable,
    SegPagedKVCache,
    build_segpaged_cache,
    dense_reference_attention,
    is_fused_varlen_available,
    local_visible_indices,
    segpaged_attention,
    verify_against_dense,
)
from sglang.srt.layers.attention.redknot.sparse_ffn import (
    SparseFFNSchedule,
    apply_sparse_ffn,
    select_important_tokens,
    sparse_ffn_flops,
    token_importance_from_attn,
)
from sglang.srt.layers.attention.redknot.head_profiler import (
    HeadStats,
    ProfileResult,
    build_head_config,
    classify_from_stats,
    profile_model_heads,
    save_head_config_json,
)
from sglang.srt.layers.attention.redknot.pd_transfer import (
    HeadClassKVPayload,
    HeadKVSlice,
    build_transfer_payload,
    restore_payload,
)
from sglang.srt.layers.attention.redknot.scheduler import (
    ChunkEntry,
    HeadAwareCachePolicy,
    concurrent_capacity,
    per_session_kv_bytes,
)
from sglang.srt.layers.attention.redknot.eval_harness import (
    DATASET_SPECS,
    DatasetSpec,
    aggregate_efficiency,
    aggregate_quality,
    coefficient_of_variation,
    list_datasets,
    pd_throughput_projection,
)

__all__ = [
    "HEAD_DENSE",
    "HEAD_GLOBAL",
    "HEAD_LOCAL",
    "HEAD_RETRIEVAL",
    "DEFAULT_LOCAL_WINDOW",
    "DEFAULT_RETRIEVAL_TOP_P",
    "DEFAULT_SINK_SIZE",
    "DEFAULT_Q_CHUNK",
    "DeepSeekV4MLAHeadConfig",
    "HeadClassConfig",
    "HeadStrategy",
    "MLAHeadStrategy",
    "LayerMaskPlan",
    "OfflineKVCache",
    "OfflineSegment",
    "RoPEHelper",
    "build_layer_mask_plan",
    "group_heads_by_type",
    "get_global_offline_cache",
    "deepseek_v4_mla_cache_descriptor",
    "is_deepseek_v4_mla_config",
    "is_fa3_available",
    "is_fa3_parallel_available",
    "is_flash_attn_available",
    "offline_prefill_segment",
    "offline_prefill_segments",
    "online_forward_segment",
    "online_forward_segments_batched",
    "pad_per_head_sinks",
    "q_head_indices",
    "run_baseline",
    "run_redknot",
    "run_redknot_batched",
    "run_redknot_offlinekv",
    "segment_attention_flash",
    "segment_attention_flash3",
    "segment_attention_flash3_parallel",
    "set_global_offline_cache",
    # Sparse FFN (RedKnot Elastic Sparsity, paper §4.2)
    "SparseFFNSchedule",
    "apply_sparse_ffn",
    "select_important_tokens",
    "sparse_ffn_flops",
    "token_importance_from_attn",
    # SegPagedAttention runtime (paper §4.3)
    "POLICY_GLOBAL",
    "POLICY_LOCAL",
    "HeadSegment",
    "SegmentPageTable",
    "SegPagedKVCache",
    "build_segpaged_cache",
    "dense_reference_attention",
    "is_fused_varlen_available",
    "local_visible_indices",
    "segpaged_attention",
    "verify_against_dense",
    # Head profiling (P1, paper §3.2)
    "HeadStats",
    "ProfileResult",
    "build_head_config",
    "classify_from_stats",
    "profile_model_heads",
    "save_head_config_json",
    # PD head-class KV transfer (P2, paper fig. 13a)
    "HeadClassKVPayload",
    "HeadKVSlice",
    "build_transfer_payload",
    "restore_payload",
    # Head-aware scheduler / capacity (P2, paper fig. 13c)
    "ChunkEntry",
    "HeadAwareCachePolicy",
    "concurrent_capacity",
    "per_session_kv_bytes",
    # Evaluation harness (P2, paper §5)
    "DATASET_SPECS",
    "DatasetSpec",
    "aggregate_efficiency",
    "aggregate_quality",
    "coefficient_of_variation",
    "list_datasets",
    "pd_throughput_projection",
]
