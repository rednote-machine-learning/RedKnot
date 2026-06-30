# Copyright 2024-2026 SGLang RedKnot Integration.
"""SegPaged v2: a unified segment-paged KV storage substrate.

This is a standalone, additive package that provides the "段页式存储"
(segment-paged storage) base paradigm for future unified multi-head KV
processing. It is intentionally decoupled from the existing
``redknot.segpaged`` module and the live attention backends, so it does
**not** affect any current benchmark or test.

Layered design
--------------
1. :mod:`visible_plan` — :class:`HeadVisiblePlan`: the single, unified way
   to declare which tokens a head keeps (global / local / custom positions).
2. :mod:`storage` — :class:`KVStorageBackend` abstraction + the reference
   :class:`LocalPagedPool` (virtual page -> physical block). The seam where
   a real SGLang ``TokenToKVPool`` can be plugged in later.
3. :mod:`page_table` — :class:`SegmentPageTable` (head -> virtual pages) and
   :class:`PagedHeadKVCache` (store/gather visible KV through the backend).
4. :mod:`attention` — :func:`paged_attention` (FA-3 varlen fused path +
   exact PyTorch reference) and a numerical-equivalence harness.

Typical use::

    from sglang.srt.layers.attention.redknot.segpaged_v2 import (
        global_plan, local_plan, custom_plan,
        build_paged_cache, paged_attention,
    )

    plans = [global_plan(L) if h < n_g else local_plan(L, sink, recent)
             for h in range(Hkv)]
    cache = build_paged_cache(k_dense=k, v_dense=v, plans=plans)
    out = paged_attention(q, cache, layer=0, num_q_per_kv=g, sm_scale=s)
"""

from .attention import (
    build_paged_cache,
    dense_reference_attention,
    is_fused_varlen_available,
    paged_attention,
    plans_from_policies,
    verify_against_dense,
)
from .page_table import (
    HeadSegment,
    PagedHeadKVCache,
    SegmentPageTable,
)
from .storage import (
    KVStorageBackend,
    LocalPagedPool,
)
from .visible_plan import (
    POLICY_CUSTOM,
    POLICY_GLOBAL,
    POLICY_LOCAL,
    HeadVisiblePlan,
    custom_plan,
    global_plan,
    local_plan,
)

__all__ = [
    # visible_plan
    "POLICY_GLOBAL",
    "POLICY_LOCAL",
    "POLICY_CUSTOM",
    "HeadVisiblePlan",
    "global_plan",
    "local_plan",
    "custom_plan",
    # storage
    "KVStorageBackend",
    "LocalPagedPool",
    # page_table
    "HeadSegment",
    "SegmentPageTable",
    "PagedHeadKVCache",
    # attention
    "build_paged_cache",
    "plans_from_policies",
    "paged_attention",
    "dense_reference_attention",
    "verify_against_dense",
    "is_fused_varlen_available",
]
