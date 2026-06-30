# Copyright 2024-2026 SGLang RedKnot Integration.
"""RedKnot attention backend for sglang.

This backend plugs the RedKnot head-classified attention strategy into
sglang's :class:`AttentionBackend` interface. Compared to the original
``__REDKNOT_V02__/redknot/core.py`` reference which serialises KV heads in a
Python loop, this backend processes all heads of a layer **in parallel**
through :func:`sglang.srt.layers.attention.redknot.ops_flash.segment_attention_flash`
which dispatches each head-type bucket to FlashAttention-2.

What this implementation covers
-------------------------------
- ``forward_extend``: prefill / extend a sequence. Each request can be
  associated with a list of *offline segment ids* (see
  :class:`sglang.srt.layers.attention.redknot.OfflineKVCache`). When such
  segments exist we splice them — with RoPE realignment — into the prev-KV
  view before computing the head-classified attention.
- ``forward_decode``: standard single-token decode, identical semantics to
  ``TorchNativeAttnBackend`` because once the prefill has produced a
  contiguous KV cache, decode just reads from it.

Backend selection
-----------------
Registered as ``"redknot"`` via
``sglang.srt.layers.attention.attention_registry``. Activate with:

    python -m sglang.launch_server --attention-backend redknot ...

Per-request control
-------------------
The backend reads three optional fields off ``ForwardBatch`` (set by the
scheduler / a custom processor):

- ``forward_batch.redknot_offline_segments``:
    ``Optional[List[Optional[List[str]]]]``
    For each request in the batch, a list of offline segment ids to splice
    in. ``None`` skips RedKnot for that request (falls back to plain SDPA).
- ``forward_batch.redknot_head_config``:
    Optional override for the layer-wide ``HeadClassConfig`` (otherwise the
    one passed at backend construction is used).

The backend lazily attaches them at first ``init_forward_metadata`` and
ignores them on subsequent layers in the same step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
from sglang.srt.layers.attention.redknot.offline_cache import (
    OfflineKVCache,
    OfflineSegment,
    get_global_offline_cache,
)
from sglang.srt.layers.attention.redknot.mask_plan import (
    build_layer_mask_plan,
    pad_per_head_sinks,
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
from sglang.srt.layers.attention.redknot.rope_helper import RoPEHelper
from sglang.srt.layers.attention.redknot.segpaged import (
    POLICY_GLOBAL,
    POLICY_LOCAL,
    SegPagedKVCache,
    is_fused_varlen_available,
    segpaged_attention,
)


def _resolve_kernel_fn(kernel: str):
    """Map ``kernel`` -> attention function (FA-2 or FA-3)."""
    k = (kernel or "fa2").lower()
    if k in ("flash", "fa2"):
        if not is_flash_attn_available():
            raise RuntimeError("RedKnot kernel='fa2' requires the flash_attn package.")
        return segment_attention_flash
    if k == "fa3":
        if not is_fa3_available():
            raise RuntimeError(
                "RedKnot kernel='fa3' requires sgl_kernel.flash_attn with FA-3 "
                "support (Hopper / SM 9.0)."
            )
        return segment_attention_flash3
    raise ValueError(f"Unknown RedKnot kernel {kernel!r}")


from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────────────────
class RedKnotAttnBackend(AttentionBackend):
    """sglang attention backend implementing head-classified KV compression."""

    def __init__(
        self,
        model_runner: "ModelRunner",
        head_config: Optional[HeadClassConfig] = None,
        offline_cache: Optional[OfflineKVCache] = None,
        q_chunk_size: int = DEFAULT_Q_CHUNK,
        kernel: str = "fa2",
        use_segpaged_decode: bool = False,
        segpaged_page_size: int = 64,
    ):
        super().__init__()
        self.device = model_runner.device
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        # Resolve kernel up-front so server start fails loudly if the
        # requested implementation isn't available.
        self.kernel_name = (kernel or "fa2").lower()
        self._kernel_fn = _resolve_kernel_fn(self.kernel_name)

        # SegPagedAttention decode: when enabled, decode reads a per-head
        # paged KV view (local heads -> sink+recent only, global heads ->
        # full context) and runs the mask-free fused varlen path, instead of
        # a dense SDPA over the full [H, L, D] KV. This is the P1 integration
        # point that turns head-class sparsity into real decode-time work
        # reduction (paper §5.4 / §5.5).
        self.use_segpaged_decode = bool(use_segpaged_decode)
        self.segpaged_page_size = int(segpaged_page_size)
        if self.use_segpaged_decode and not is_fused_varlen_available():
            logger.warning(
                "RedKnot: use_segpaged_decode requested but fused varlen "
                "FA-3 is unavailable; SegPaged decode will use the exact "
                "PyTorch reference path (correct but not kernel-accelerated)."
            )

        self.head_config = head_config or _maybe_load_default_head_config(model_runner)
        if self.head_config is None:
            logger.warning(
                "RedKnotAttnBackend: no head_config provided and no default "
                "could be inferred; the backend will route every head as "
                "global until a config is attached."
            )

        # Hybrid-model support: for models that interleave linear-attention and
        # full-attention layers (e.g. Qwen3.5 GDN: 45 linear + 15 full), the
        # RedKnot backend is wired as the *full-attention* sub-backend of
        # HybridLinearAttnBackend and only ever sees the full-attention layers.
        # Those layers keep their GLOBAL ``layer.layer_id`` (0..N_total-1) for
        # KV-cache addressing, but ``head_config`` is indexed by FULL-ATTENTION
        # POSITION (0..N_full-1). Build a global->relative map so head-config
        # lookups use the right row. For non-hybrid models the map is identity.
        self._full_attn_layer_index = _build_full_attn_index_map(model_runner)

        self.offline_cache = offline_cache or get_global_offline_cache()
        self.q_chunk_size = q_chunk_size

        # Tensor-parallel KV-head slice. The head_config is GLOBAL (all KV
        # heads); under TP each rank owns a contiguous slice. We record this
        # rank's index so per-layer mask plans are sliced to the local heads
        # (matching layer.tp_k_head_num) — without this the sparse kernel would
        # index global head ids into local KV tensors and crash under TP.
        try:
            from sglang.srt.layers.dp_attention import get_attention_tp_rank

            self._attn_tp_rank = int(get_attention_tp_rank())
        except Exception:  # pragma: no cover - single-rank / no dist
            self._attn_tp_rank = 0

        # Lazily constructed when we first see a rotary_emb (needed for
        # RoPE realignment on dense / offline-spliced segments).
        self._rope_helper: Optional[RoPEHelper] = None
        self._rope_owner = None  # The rotary_emb module we wrapped.

        # Forward-pass scratch.
        self.forward_metadata = None

    # ────────────────────────────────────────────────────────────────
    # Indexing helpers
    # ────────────────────────────────────────────────────────────────
    def _head_cfg_layer_index(self, layer_id: int) -> int:
        """Map a model-global ``layer_id`` to its row in ``head_config``.

        For hybrid models head_config only has rows for full-attention layers,
        so the global id (used for KV-cache addressing) must be translated to
        the full-attention position. For dense models this is the identity.
        """
        if self._full_attn_layer_index is None:
            return layer_id
        return self._full_attn_layer_index.get(layer_id, layer_id)

    # ────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Pick up per-batch RedKnot overrides without disturbing other backends."""
        # If the scheduler attached a per-request offline-segment plan, hold
        # onto a normalised view of it.
        offline_plan = getattr(forward_batch, "redknot_offline_segments", None)
        head_override = getattr(forward_batch, "redknot_head_config", None)

        bsz = (
            int(forward_batch.req_pool_indices.shape[0])
            if hasattr(forward_batch, "req_pool_indices")
            and forward_batch.req_pool_indices is not None
            else 0
        )
        if offline_plan is not None and len(offline_plan) != bsz:
            logger.warning(
                "RedKnot: offline_segments len(%d) != batch size(%d); ignoring.",
                len(offline_plan),
                bsz,
            )
            offline_plan = None

        self.forward_metadata = _ForwardMeta(
            offline_segments=offline_plan,
            head_config=head_override or self.head_config,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def support_triton(self) -> bool:
        # Mask construction is done in PyTorch; the actual SDPA call below
        # can run on any cuda backend. We report ``False`` because the
        # backend does not provide its own Triton kernels at this stage.
        return False

    # ────────────────────────────────────────────────────────────────
    # forward_extend / forward_decode
    # ────────────────────────────────────────────────────────────────
    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Prefill / extend with RedKnot head-classified attention.

        Behaviour mirrors :class:`TorchNativeAttnBackend.forward_extend` in
        the no-RedKnot case, but when ``forward_batch`` carries offline
        segment ids we route attention per-head according to ``head_config``.
        """
        # Write incoming K/V to sglang's KV cache exactly like other backends.
        if save_kv_cache and k is not None and v is not None:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        meta = self.forward_metadata or _ForwardMeta(None, self.head_config)
        head_config = meta.head_config

        if head_config is None or meta.offline_segments is None:
            # Nothing to compress / no plan — graceful fallback.
            return self._sdpa_fallback_extend(q, layer, forward_batch)

        # Allocate output.
        if layer.qk_head_dim != layer.v_head_dim:
            out = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            out = torch.empty_like(q)

        # Per-layer mask plan (cached across heads of this layer). head_config
        # is indexed by full-attention position, not global layer id.
        hc_idx = self._head_cfg_layer_index(layer.layer_id)
        plan = build_layer_mask_plan(
            head_config,
            hc_idx,
            self.device,
            kv_head_start=self._attn_tp_rank * layer.tp_k_head_num,
            kv_head_count=layer.tp_k_head_num,
        )

        # Walk requests and run RedKnot attention where a plan exists.
        start_q = 0
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token

        for seq_idx in range(forward_batch.seq_lens.shape[0]):
            ext_q = int(forward_batch.extend_seq_lens[seq_idx])
            prefix_len = int(forward_batch.extend_prefix_lens[seq_idx])
            end_q = start_q + ext_q
            offline_ids = (
                meta.offline_segments[seq_idx]
                if meta.offline_segments is not None
                else None
            )

            # ── Offline-segment BUILD mode ──
            # A request whose plan is a single sentinel "__RKBUILD__:<sid>" is
            # not a query: it is a document chunk being prefilled so we can
            # capture its KV as an offline segment. We run a plain dense
            # prefill (so KV is written normally by all layers) and, on the
            # LAST full-attention layer, snapshot the whole request's KV.
            build_sid = _parse_build_sentinel(offline_ids)
            if build_sid is not None:
                self._sdpa_single_extend(
                    q[start_q:end_q],
                    out[start_q:end_q],
                    layer,
                    seq_idx,
                    forward_batch,
                    k_cache,
                    v_cache,
                    req_to_token,
                    prefix_len,
                )
                if self._is_last_offline_layer(layer.layer_id):
                    seq_len_kv = int(forward_batch.seq_lens[seq_idx])
                    req_pool_idx = int(forward_batch.req_pool_indices[seq_idx])
                    tok_ids = req_to_token[req_pool_idx, :seq_len_kv].to("cpu")
                    try:
                        self.snapshot_offline_segment(
                            segment_id=build_sid,
                            token_ids=tok_ids,
                            req_pool_idx=req_pool_idx,
                            seq_len=seq_len_kv,
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.error(
                            "RedKnot: offline segment build failed for %s: %s",
                            build_sid,
                            exc,
                        )
                start_q = end_q
                continue

            if not offline_ids:
                # Standard prefill for this request via SDPA over its
                # current KV slice.
                self._sdpa_single_extend(
                    q[start_q:end_q],
                    out[start_q:end_q],
                    layer,
                    seq_idx,
                    forward_batch,
                    k_cache,
                    v_cache,
                    req_to_token,
                    prefix_len,
                )
                start_q = end_q
                continue

            # ── RedKnot path ──
            self._redknot_single_extend(
                q[start_q:end_q],
                out[start_q:end_q],
                layer=layer,
                seq_idx=seq_idx,
                forward_batch=forward_batch,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
                prefix_len=prefix_len,
                offline_segment_ids=offline_ids,
                plan=plan,
                head_config=head_config,
            )
            start_q = end_q

        return out

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Decode = read concatenated KV cache. RedKnot compression is
        applied during prefill; decode is a plain windowed SDPA over the
        materialised KV — identical to TorchNativeAttnBackend.
        """
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            out = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            out = torch.empty_like(q)
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if save_kv_cache and k is not None and v is not None:
            self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = out.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token

        # SegPaged decode path: only when enabled and a head plan exists.
        meta = self.forward_metadata
        head_config = meta.head_config if meta is not None else self.head_config
        if self.use_segpaged_decode and head_config is not None:
            return self._segpaged_decode(
                q_,
                out,
                layer,
                forward_batch,
                k_cache,
                v_cache,
                req_to_token,
                head_config,
            )

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        qm = q_.movedim(0, q_.dim() - 2)
        start_q = 0
        for seq_idx in range(forward_batch.seq_lens.shape[0]):
            seq_len_kv = int(forward_batch.seq_lens[seq_idx])
            end_q = start_q + 1
            req_pool_idx = forward_batch.req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, qm.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, qm.dim() - 2)

            per_req_query = qm[:, start_q:end_q, :]
            if per_req_query.dtype != per_req_key.dtype:
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=use_gqa,
                    scale=layer.scaling,
                    is_causal=False,
                )
                .squeeze(0)
                .movedim(qm.dim() - 2, 0)
            )
            o_[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return out

    # ────────────────────────────────────────────────────────────────
    # SegPaged decode (P1: fused varlen per-head decode)
    # ────────────────────────────────────────────────────────────────
    def _segpaged_decode(
        self,
        q_: torch.Tensor,
        out: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        head_config: HeadClassConfig,
    ) -> torch.Tensor:
        """Decode using a per-head SegPaged KV view + fused varlen attention.

        For each request, build a :class:`SegPagedKVCache` for this layer
        where local heads physically retain only their ``sink + recent``
        window and global heads retain the full context, then run
        :func:`segpaged_attention` (Algorithm 2). This avoids materialising
        the dense ``[H, L, D]`` KV and the SDPA mask penalty for local heads.
        """
        o_ = out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        num_q_per_kv = layer.tp_q_head_num // layer.tp_k_head_num
        # head_config is indexed by full-attention position, not global layer id.
        hc_idx = self._head_cfg_layer_index(layer.layer_id)
        _kvh_start = self._attn_tp_rank * layer.tp_k_head_num
        _kvh_end = _kvh_start + layer.tp_k_head_num
        plan = build_layer_mask_plan(
            head_config,
            hc_idx,
            self.device,
            kv_head_start=_kvh_start,
            kv_head_count=layer.tp_k_head_num,
        )
        type_codes = plan.type_codes  # [KVH_local]
        windows = plan.window
        sinks = head_config.as_tensors(self.device)["sink_size"][hc_idx][
            _kvh_start:_kvh_end
        ]

        # Map RedKnot head types -> SegPaged policy: global/dense/retrieval
        # read full context; local reads sink+recent.
        policies = []
        for kvh in range(layer.tp_k_head_num):
            code = int(type_codes[kvh].item())
            policies.append(
                POLICY_LOCAL if code == HeadClassConfig.TYPE_LOCAL else POLICY_GLOBAL
            )

        start_q = 0
        for seq_idx in range(forward_batch.seq_lens.shape[0]):
            seq_len_kv = int(forward_batch.seq_lens[seq_idx])
            end_q = start_q + 1
            req_pool_idx = forward_batch.req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            # [L, KVH, D] -> [KVH, L, D]
            k_seq = k_cache[per_req_tokens].movedim(0, 1)
            v_seq = v_cache[per_req_tokens].movedim(0, 1)

            cache = SegPagedKVCache(
                num_layers=layer.layer_id + 1,
                num_kv_heads=layer.tp_k_head_num,
                head_dim=layer.qk_head_dim,
                page_size=self.segpaged_page_size,
                device=k_seq.device,
                dtype=k_seq.dtype,
            )
            from sglang.srt.layers.attention.redknot.segpaged import (
                local_visible_indices,
            )

            for kvh in range(layer.tp_k_head_num):
                if policies[kvh] == POLICY_LOCAL:
                    w = int(windows[kvh].item())
                    s = int(sinks[kvh].item())
                    idx = local_visible_indices(
                        seq_len_kv, max(s, 0), max(w, 0), device=k_seq.device
                    )
                    k_h = k_seq[kvh].index_select(0, idx)
                    v_h = v_seq[kvh].index_select(0, idx)
                else:
                    k_h = k_seq[kvh]
                    v_h = v_seq[kvh]
                cache.add_head_segment(
                    layer=layer.layer_id,
                    head=kvh,
                    segment=0,
                    policy=policies[kvh],
                    k=k_h,
                    v=v_h,
                )

            # query for this token: [Hq, 1, D]
            q_tok = q_[start_q:end_q].movedim(0, 1)  # [Hq, 1, D]
            attn = segpaged_attention(
                q_tok,
                cache,
                layer=layer.layer_id,
                num_q_per_kv=num_q_per_kv,
                sm_scale=layer.scaling,
                use_fused=True,
            )  # [Hq, 1, D]
            o_[start_q:end_q] = attn.movedim(1, 0)
            start_q = end_q

        return out

    # ────────────────────────────────────────────────────────────────
    # Public helpers used by application code (e.g. an offline-prefill server)
    # ────────────────────────────────────────────────────────────────
    def attach_rope_helper(self, rotary_emb) -> RoPEHelper:
        """Register a model rotary embedding so dense / spliced segments
        can be RoPE-realigned. Safe to call repeatedly; subsequent calls
        with the same module are no-ops."""
        if self._rope_helper is not None and self._rope_owner is rotary_emb:
            return self._rope_helper
        self._rope_helper = RoPEHelper(rotary_emb)
        self._rope_owner = rotary_emb
        return self._rope_helper

    def _is_last_offline_layer(self, layer_id: int) -> bool:
        """True if ``layer_id`` is the last full-attention layer captured by an
        offline snapshot — i.e. the point in a forward where every layer the
        snapshot needs has already written its KV this step."""
        ids = self._offline_layer_ids()
        return bool(ids) and layer_id == max(ids)

    def _offline_layer_ids(self) -> List[int]:
        """Global layer ids whose KV must be captured for an offline segment.

        For hybrid models only the full-attention layers carry head-class
        sparsity (and are the only ones the backend ever serves), so the
        offline segment only needs their KV. For dense models every layer is
        a full-attention layer; we capture the full contiguous range owned by
        this pool.
        """
        if self._full_attn_layer_index:
            return sorted(self._full_attn_layer_index.keys())
        # Dense model: every layer this KV pool owns.
        start = int(getattr(self.token_to_kv_pool, "start_layer", 0))
        n = int(
            getattr(self.token_to_kv_pool, "layer_num", None)
            or getattr(self.token_to_kv_pool, "size", 0)
            and 0
        )
        if not n:
            # Fall back to attribute commonly present on KV pools.
            n = int(getattr(self.token_to_kv_pool, "layer_num", 0))
        return list(range(start, start + n)) if n else []

    @torch.no_grad()
    def snapshot_offline_segment(
        self,
        *,
        segment_id: str,
        token_ids: torch.Tensor,
        req_pool_idx: int,
        seq_len: int,
    ) -> "OfflineSegment":
        """Capture the just-prefilled KV of a request as an offline segment.

        Reads this rank's KV shard for ``seq_len`` tokens of the request at
        ``req_pool_idx`` out of the live ``token_to_kv_pool`` and registers a
        per-rank :class:`OfflineSegment` (same ``segment_id`` on every TP rank).

        The KV is stored in the per-layer shape the consumer expects:
        ``[1, KVH_per_rank, L, head_dim]``. Caller must ensure the request was
        prefilled with prefix_len == 0 so RoPE positions are ``[0, seq_len)``.
        """
        req_to_token = self.req_to_token_pool.req_to_token
        tok_locs = req_to_token[req_pool_idx, :seq_len]
        kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        layer_ids = self._offline_layer_ids()
        if not layer_ids:
            raise RuntimeError(
                "RedKnot.snapshot_offline_segment: could not determine which "
                "layers to capture (no full-attn map and no layer_num)."
            )
        # Build a dense kv list indexed by GLOBAL layer_id so the consumer's
        # ``seg.kv[layer.layer_id]`` lookup works. Non-captured layers (linear
        # layers in hybrid models) get a (None, None) placeholder that is never
        # read by the full-attention backend.
        max_lid = max(layer_ids)
        placeholder: Tuple[torch.Tensor, torch.Tensor] = (None, None)  # type: ignore
        kv = [placeholder] * (max_lid + 1)
        for lid in layer_ids:
            k_buf = self.token_to_kv_pool.get_key_buffer(lid)  # [N, KVH, D]
            v_buf = self.token_to_kv_pool.get_value_buffer(lid)
            k = k_buf[tok_locs].unsqueeze(0).movedim(2, 1).contiguous().clone()
            v = v_buf[tok_locs].unsqueeze(0).movedim(2, 1).contiguous().clone()
            # k,v: [1, KVH_per_rank, L, D]
            kv[lid] = (k, v)

        from sglang.srt.layers.attention.redknot.offline_cache import (
            build_offline_segment,
        )

        seg = build_offline_segment(segment_id=segment_id, token_ids=token_ids, kv=kv)
        self.offline_cache.put(seg)
        return seg

    # ────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────
    def _sdpa_fallback_extend(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Plain SDPA prefill (mirrors TorchNativeAttnBackend.forward_extend)."""
        if layer.qk_head_dim != layer.v_head_dim:
            out = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            out = torch.empty_like(q)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token

        start_q = 0
        for seq_idx in range(forward_batch.seq_lens.shape[0]):
            ext_q = int(forward_batch.extend_seq_lens[seq_idx])
            prefix_len = int(forward_batch.extend_prefix_lens[seq_idx])
            end_q = start_q + ext_q
            self._sdpa_single_extend(
                q[start_q:end_q],
                out[start_q:end_q],
                layer,
                seq_idx,
                forward_batch,
                k_cache,
                v_cache,
                req_to_token,
                prefix_len,
            )
            start_q = end_q
        return out

    def _sdpa_single_extend(
        self,
        q_slice: torch.Tensor,
        out_slice: torch.Tensor,
        layer: "RadixAttention",
        seq_idx: int,
        forward_batch: ForwardBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_len: int,
    ) -> None:
        seq_len_kv = int(forward_batch.seq_lens[seq_idx])
        ext_q = q_slice.shape[0]
        req_pool_idx = forward_batch.req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        # Reshape to [num_heads, ...] for SDPA.
        per_req_key = k_cache[per_req_tokens]  # [seq_len, KVH, D]
        per_req_value = v_cache[per_req_tokens]
        q_ = q_slice.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = out_slice.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
        causal = (
            not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
        )

        # SDPA expects [B, H, T, D].
        # Pad query to full seq length with empty rows so causal mask works
        # over the prefix as well (matches TorchNativeAttnBackend's redundant
        # query approach).
        per_req_q = torch.empty(
            (layer.tp_q_head_num, seq_len_kv, layer.qk_head_dim),
            dtype=q_.dtype,
            device=q_.device,
        )
        # Movedim from [T, H, D] -> [H, T, D]
        per_req_q[:, prefix_len:, :] = q_.movedim(0, 1)

        per_req_k = per_req_key.movedim(0, 1)
        per_req_v = per_req_value.movedim(0, 1)
        if per_req_q.dtype != per_req_k.dtype:
            per_req_k = per_req_k.to(per_req_q.dtype)
            per_req_v = per_req_v.to(per_req_q.dtype)

        full = scaled_dot_product_attention(
            per_req_q.unsqueeze(0),
            per_req_k.unsqueeze(0),
            per_req_v.unsqueeze(0),
            enable_gqa=use_gqa,
            scale=layer.scaling,
            is_causal=causal,
        ).squeeze(0)  # [H, T, D]
        o_[:] = full[:, prefix_len:, :].movedim(0, 1)

    def _redknot_single_extend(
        self,
        q_slice: torch.Tensor,
        out_slice: torch.Tensor,
        *,
        layer: "RadixAttention",
        seq_idx: int,
        forward_batch: ForwardBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_len: int,
        offline_segment_ids: List[Optional[str]],
        plan,
        head_config: HeadClassConfig,
    ) -> None:
        """Single-request RedKnot attention.

        Constructs prev-KV by concatenating:
          1) Offline segments listed in ``offline_segment_ids`` (with RoPE
             realignment when a RoPE helper is attached).
          2) Any KV already materialised in sglang's cache for this request
             (treated as online prev KV).
        Then runs the unified per-head attention kernel.
        """
        device = q_slice.device
        dtype = q_slice.dtype
        ext_q = q_slice.shape[0]
        seq_len_kv = int(forward_batch.seq_lens[seq_idx])

        # ── Resolve offline segments ──
        offline_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        offline_positions: List[Tuple[int, int]] = []
        cursor = 0
        for sid in offline_segment_ids:
            if sid is None:
                continue
            seg = self.offline_cache.to_device(sid, device)
            if seg is None:
                logger.warning("RedKnot: missing offline segment %s; skipping.", sid)
                continue
            # KV for this layer.
            k_off, v_off = seg.kv[layer.layer_id]
            # Optionally realign RoPE from [0, L) to [cursor, cursor + L).
            if self._rope_helper is not None and cursor != 0:
                k_off = self._rope_helper.reposition_offset(
                    k_off, src_start=0, dst_start=cursor, length=seg.doc_len
                )
            offline_kvs.append((k_off, v_off))
            offline_positions.append((cursor, cursor + seg.doc_len))
            cursor += seg.doc_len

        # ── Stack into per-layer (B, KVH, P, D) tensor ──
        if offline_kvs:
            k_prev = torch.cat([kv[0] for kv in offline_kvs], dim=2)
            v_prev = torch.cat([kv[1] for kv in offline_kvs], dim=2)
        else:
            k_prev = None
            v_prev = None

        # ── Self (current segment) KV: pulled from sglang cache slice ──
        req_pool_idx = forward_batch.req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        # cache shape: [num_tokens, KVH, D]
        k_full = k_cache[per_req_tokens].unsqueeze(0)  # [1, T, KVH, D]
        v_full = v_cache[per_req_tokens].unsqueeze(0)
        k_full = k_full.movedim(2, 1)  # [1, KVH, T, D]
        v_full = v_full.movedim(2, 1)

        # In RedKnot semantics, the current segment is the EXTENDED part.
        # Older prefix tokens (if any) live in the cache too; we treat them
        # as additional online prev for safety.
        if prefix_len > 0:
            k_prefix = k_full[:, :, :prefix_len, :]
            v_prefix = v_full[:, :, :prefix_len, :]
            if k_prev is None:
                k_prev = k_prefix
                v_prev = v_prefix
            else:
                k_prev = torch.cat([k_prev, k_prefix], dim=2)
                v_prev = torch.cat([v_prev, v_prefix], dim=2)
        k_self = k_full[:, :, prefix_len : prefix_len + ext_q, :]
        v_self = v_full[:, :, prefix_len : prefix_len + ext_q, :]

        # ── Sink tokens: from the very first offline segment (if any) ──
        if offline_kvs:
            first_k = offline_kvs[0][0]
            first_v = offline_kvs[0][1]
            k_sink, v_sink = pad_per_head_sinks(first_k, first_v, plan)
        elif prefix_len > 0:
            # No offline data; reuse the very first slice of the prefix.
            first_k = k_full[:, :, :1, :]
            first_v = v_full[:, :, :1, :]
            k_sink, v_sink = pad_per_head_sinks(first_k, first_v, plan)
        else:
            k_sink, v_sink = None, None

        # ── Build Q tensor in [B, Hq, L_q, D] ──
        q = q_slice.view(ext_q, layer.tp_q_head_num, layer.qk_head_dim)
        q = q.movedim(0, 1).unsqueeze(0)  # [1, Hq, L_q, D]

        # ── FlashAttention per-head-type attention ──
        # Dispatched through ``self._kernel_fn`` (FA-2 or FA-3). Each
        # (layer, kv_head) bucket maps to: local -> two-pass (sink/prev
        # full + self windowed) with LSE merge; global/dense -> two-pass +
        # LSE merge; retrieval -> top-p selection -> global path.
        num_q_per_kv = layer.tp_q_head_num // layer.tp_k_head_num
        attn_out = self._kernel_fn(
            q=q,
            k_self=k_self,
            v_self=v_self,
            k_prev=k_prev,
            v_prev=v_prev,
            k_sink_padded=k_sink,
            v_sink_padded=v_sink,
            plan=plan,
            num_q_per_kv=num_q_per_kv,
            sm_scale=layer.scaling,
            retrieval_top_p=head_config.retrieval_top_p,
            q_chunk_size=self.q_chunk_size,
        )  # [1, Hq, L_q, D]

        attn_out = attn_out.squeeze(0).movedim(0, 1).contiguous()  # [L_q, Hq, D]
        out_slice.view(ext_q, layer.tp_q_head_num, layer.v_head_dim).copy_(attn_out)


# ──────────────────────────────────────────────────────────────────────────
# Small POD for per-step metadata
# ──────────────────────────────────────────────────────────────────────────
class _ForwardMeta:
    __slots__ = ("offline_segments", "head_config")

    def __init__(
        self,
        offline_segments: Optional[List[Optional[List[str]]]],
        head_config: Optional[HeadClassConfig],
    ):
        self.offline_segments = offline_segments
        self.head_config = head_config


# ──────────────────────────────────────────────────────────────────────────
# Utility: try to load a head config from a CLI / server arg
# ──────────────────────────────────────────────────────────────────────────
RKBUILD_PREFIX = "__RKBUILD__:"


def _parse_build_sentinel(offline_ids) -> Optional[str]:
    """If a request's offline plan is a single build sentinel
    ``"__RKBUILD__:<segment_id>"``, return ``<segment_id>``; else None.

    This lets a document chunk be prefilled through the normal generate path
    purely to capture its KV as an offline segment, with no scheduler changes.
    """
    if not offline_ids:
        return None
    if isinstance(offline_ids, str):
        ids = [offline_ids]
    else:
        ids = list(offline_ids)
    if len(ids) == 1 and isinstance(ids[0], str) and ids[0].startswith(RKBUILD_PREFIX):
        return ids[0][len(RKBUILD_PREFIX) :]
    return None


def _build_full_attn_index_map(model_runner) -> Optional[Dict[int, int]]:
    """Map global ``layer_id`` -> full-attention position for hybrid models.

    Returns ``None`` for dense models (where the mapping is the identity and
    no translation is needed). For hybrid linear/full models (e.g. Qwen3.5
    GDN), returns ``{global_layer_id: full_attn_position}`` so that the RedKnot
    backend — which is wired as the full-attention sub-backend of
    HybridLinearAttnBackend and only sees full-attention layers — can index a
    head_config that was built for the full-attention layers only.
    """
    cfg = getattr(model_runner, "mambaish_config", None)
    if cfg is None:
        return None
    full_ids = getattr(cfg, "full_attention_layer_ids", None)
    if not full_ids:
        return None
    full_ids = sorted(int(i) for i in full_ids)
    index_map = {gid: pos for pos, gid in enumerate(full_ids)}
    logger.info(
        "RedKnot: hybrid model detected — %d full-attention layers; "
        "head_config indexed by full-attn position (ids=%s).",
        len(full_ids),
        full_ids if len(full_ids) <= 20 else f"{full_ids[:8]}...{full_ids[-4:]}",
    )
    return index_map


def _maybe_load_default_head_config(model_runner) -> Optional[HeadClassConfig]:
    """Look for ``model_runner.server_args.redknot_head_config_path``."""
    server_args = getattr(model_runner, "server_args", None)
    if server_args is None:
        return None
    path = getattr(server_args, "redknot_head_config_path", None)
    if not path:
        return None
    try:
        cfg = HeadClassConfig.from_json(path)
        logger.info(
            "RedKnot: loaded head config %s (L=%d, KVH=%d, summary=%s)",
            path,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.summary(),
        )
        return cfg
    except Exception as exc:  # pragma: no cover — graceful degradation
        logger.warning("RedKnot: failed to load head config %s: %s", path, exc)
        return None
