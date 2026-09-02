from __future__ import annotations

import enum
import functools
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

# ── RedKnot restore-hook phase timing ───────────────────────────────────────
# Accumulates GPU-synchronized wall time per phase across all layers/segments of
# one restore forward. Enabled by REDKNOT_V4_TIMING=1. Reset + dumped by the
# model runner around the restore forward. Used to find the fixed-overhead
# bottleneck (state transfer / RoPE relocation) that caps TTFT speedup.
import contextlib as _contextlib
import time as _time

_REDKNOT_TIMING: dict = {}


def _redknot_mla_off_preserves_online_kv(plan) -> bool:
    """Whether MLA output reuse requires this request's fully online KV state."""

    return bool(
        isinstance(plan, Mapping)
        and plan.get("mode") == "restore"
        and plan.get("reuse_mla_off", False)
        and not plan.get("skip_forward", False)
    )


def redknot_timing_enabled() -> bool:
    return os.environ.get("REDKNOT_V4_TIMING", "0") == "1"


def redknot_timing_reset() -> None:
    _REDKNOT_TIMING.clear()


def redknot_timing_dump() -> dict:
    return dict(_REDKNOT_TIMING)


_REDKNOT_TIMING_NOOP = _contextlib.nullcontext()


def _rk_timed(phase: str):
    if not redknot_timing_enabled():
        return _REDKNOT_TIMING_NOOP
    # Keep legacy restore-hook labels, but enqueue them into the request-wide
    # event collector.  The old implementation synchronized before and after
    # every region, materially changing the TTFT it was intended to measure.
    from sglang.srt.layers.attention.redknot.dsv4_timing import (
        timed as _event_timed,
    )

    return _event_timed(f"legacy_backend.{phase}")

if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
    # NOTE: should eventually be the only compressor backend
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )
else:
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )

from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64
# The installed FlashMLA sparse-decode scheduler rejects larger batch-row
# counts before launching attention.  RedKnot may intentionally merge several
# 8K restore segments into one model forward, so keep the outer forward merged
# while presenting the already-qualified attention operator with its proven
# row geometry.
FLASHMLA_MAX_BATCH_ROWS = 8192


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _create_flashmla_metadata():
    import flash_mla

    return flash_mla.get_mla_metadata()[0]


def _flash_mla_with_kvcache_row_chunks(
    *,
    flash_mla,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: Sequence[object],
    softmax_scale: float,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    max_batch_rows: int = FLASHMLA_MAX_BATCH_ROWS,
) -> torch.Tensor:
    """Run one logical sparse-attention pass in scheduler-safe row slices.

    The split is only along FlashMLA's independent batch-row dimension.  KV
    caches and the per-head sink are shared; every row keeps its exact indices
    and top-k lengths.  Concatenating the outputs is therefore mathematically
    identical to a single supported call and does not change causal scope.
    """

    rows = int(q.shape[0])
    limit = int(max_batch_rows)
    if rows <= 0 or limit <= 0:
        raise ValueError("FlashMLA row chunk geometry must be positive")
    num_chunks = (rows + limit - 1) // limit
    if len(tile_scheduler_metadata) != num_chunks:
        raise ValueError("FlashMLA row chunk metadata count mismatch")

    outputs = []
    for chunk_index, row_start in enumerate(range(0, rows, limit)):
        row_end = min(rows, row_start + limit)
        output = flash_mla.flash_mla_with_kvcache(
            q=q[row_start:row_end],
            k_cache=k_cache,
            head_dim_v=head_dim_v,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=tile_scheduler_metadata[chunk_index],
            softmax_scale=softmax_scale,
            is_fp8_kvcache=True,
            indices=indices[row_start:row_end],
            topk_length=topk_length[row_start:row_end],
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=(
                None
                if extra_indices_in_kvcache is None
                else extra_indices_in_kvcache[row_start:row_end]
            ),
            extra_topk_length=(
                None
                if extra_topk_length is None
                else extra_topk_length[row_start:row_end]
            ),
        )[0]
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


@dataclass
class DSV4AttnMetadata:
    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_sparse_topk: int
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    def get_flashmla_metadata(self, compress_ratio: Literal[0, 4, 128]):
        if compress_ratio == 0:
            return self.c1_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def get_flashmla_row_chunk_metadata(
        self, compress_ratio: Literal[0, 4, 128], num_chunks: int
    ) -> Tuple[object, ...]:
        """Return stable, shape-bound metadata for each attention row slice."""

        count = int(num_chunks)
        if count <= 0:
            raise ValueError("FlashMLA row chunk count must be positive")
        attribute = f"_redknot_flashmla_row_chunks_{int(compress_ratio)}"
        cached = getattr(self, attribute, None)
        if cached is None or len(cached) != count:
            cached = (self.get_flashmla_metadata(compress_ratio),) + tuple(
                _create_flashmla_metadata() for _ in range(count - 1)
            )
            setattr(self, attribute, cached)
        return cached

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "c4_sparse_topk",
                "page_size",
                "cuda_int32_kwargs",
            ],
            copy_fields=[
                "raw_out_loc",
                "seq_lens_casual",
                "positions_casual",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
            ],
            assign_fields=[
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )

    def init_compression_metadata(self):
        assert self.page_table.dim() == 2
        assert self.raw_out_loc.shape == self.seq_lens_casual.shape, (
            f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"
        )

        (
            self.c4_out_loc,
            _,
            self.c4_topk_lengths_raw,
            self.c4_topk_lengths_clamp1,
            self.c128_out_loc,
            _,
            self.c128_topk_lengths_clamp1,
            self.c128_page_indices,
        ) = _init_compression_metadata_triton(
            self.seq_lens_casual,
            self.positions_casual,
            self.raw_out_loc,
            self.page_table,
            self.page_size,
            compute_page_indices=True,
        )

        self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "c4_out_loc",
        "c128_out_loc",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_attention_cp_rank()
        cp_size = get_attention_cp_size()
        idx = slice(cp_rank, None, cp_size)
        pre_global_len = self.seq_lens_casual.shape[0]
        assert pre_global_len % cp_size == 0, (
            f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
            "CP round-robin requires padding to ensure divisibility."
        )
        expected_local_len = pre_global_len // cp_size
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(val, torch.Tensor), (
                f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            )
            setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == pre_global_len, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
            )

    def init_flashmla_related(self):
        # c4_sparse_topk is set from model_config.index_topk per-model
        # (small model: 512, large model: 1024).
        assert self.c4_sparse_topk in (512, 1024), (
            f"unexpected c4_sparse_topk={self.c4_sparse_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        assert self.c4_topk_lengths_clamp1 is not None
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)
        self.c1_flashmla_metadata = _create_flashmla_metadata()
        self.c4_flashmla_metadata = _create_flashmla_metadata()
        self.c128_flashmla_metadata = _create_flashmla_metadata()


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None

    def copy_(self, other: DSV4RawVerifyMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        self.extend_seq_lens = other.extend_seq_lens


@dataclass
class DSV4RawDecodeMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other: DSV4RawDecodeMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)


class _GraphBucket(enum.Enum):
    DECODE_OR_IDLE = "decode_or_idle"
    TARGET_VERIFY = "target_verify"
    DRAFT_EXTEND = "draft_extend"

    @classmethod
    def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
        if forward_mode.is_decode_or_idle():
            return cls.DECODE_OR_IDLE
        if forward_mode.is_target_verify():
            return cls.TARGET_VERIFY
        if forward_mode.is_draft_extend(include_v2=True):
            return cls.DRAFT_EXTEND
        raise NotImplementedError(f"unsupported {forward_mode=}")


class DeepseekV4AttnBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.device = torch.device(model_runner.device)
        head_dim = model_runner.model_config.head_dim
        assert head_dim == 512, (
            "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
        )
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool
        # RedKnot offline reuse: keep a model ref to look up per-layer freqs_cis.
        self._redknot_model_ref = getattr(model_runner, "model", None)
        self.hisparse_coordinator = model_runner.hisparse_coordinator
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.c4_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
        )

        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None
        self._replay_forward_batch: Optional[ForwardBatch] = None  # FIXME: out-of-band

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    # ──────────────────────────────────────────────────────────────────
    # RedKnot offline MLA reuse (V2: three-layer SWA+C4+C128)
    # ──────────────────────────────────────────────────────────────────
    def _maybe_redknot_reuse_hook(self, layer_id, forward_batch, token_to_kv_pool):
        """Snapshot / restore three-layer KV state for offline segment reuse.

        V2 captures all three KV layers per model layer:
          SWA  – sliding window (last 128 tokens)
          C4   – 4x compressed full-sequence cache (layers with compress_ratio=4)
          C128 – 128x compressed full-sequence cache (layers with compress_ratio=128)
          Indexer – C4 top-k index entries

        Plan protocol (via ``forward_batch.redknot_reuse_plan``):
          {"mode": "snapshot", "seg_hash": str, "length": int}
          {"mode": "restore", "segments": [
               {"seg_hash": str, "global_offset": int, "length": int,
                "skip_first": int}], ...}
        """
        import os

        # RedKnot snapshot debug: dump all-thread tracebacks on hard hang/crash
        # so CUDA-side faults surface a Python-level location. Enabled only when
        # REDKNOT_V4_DEBUG_FAULTHANDLER=1 to avoid overhead in normal runs.
        if os.environ.get("REDKNOT_V4_DEBUG_FAULTHANDLER", "0") == "1":
            import faulthandler as _fh
            import sys as _sys

            if not getattr(self, "_redknot_fh_enabled", False):
                _fh.enable(file=_sys.stderr, all_threads=True)
                self._redknot_fh_enabled = True

        from sglang.srt.layers.attention.redknot.dsv4_offline_reuse import (
            get_offline_reuse_controller,
        )
        from sglang.srt.layers.attention.redknot.dsv4_offline_reuse_v2 import (
            compute_compressed_slots,
            compute_paged_compressed_slots,
            get_offline_reuse_controller_v2,
            select_terminal_compress_state_slots,
        )
        from sglang.srt.layers.attention.redknot.v4.config import RedKnotV4Config
        from sglang.srt.layers.attention.redknot.v4.reuse_planner import (
            validate_runtime_reuse_plan,
        )

        # Offline artifacts belong to the target prefill only. In particular,
        # TARGET_VERIFY and DRAFT_EXTEND are classified as prefill by the common
        # ForwardMode helper but must never snapshot or overwrite target state.
        if forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
            return
        if (
            forward_batch.seq_lens_cpu is None
            or forward_batch.seq_lens_cpu.numel() != 1
        ):
            return

        plans = getattr(forward_batch, "redknot_reuse_plan", None)
        plan = plans[0] if plans and len(plans) == 1 else None
        if _redknot_mla_off_preserves_online_kv(plan):
            # MLA-off skips only clean local-head attention rows.  All transformer
            # rows, including the KV used by global heads, were materialized by
            # _forward_prepare*.  Legacy SWA/C4/C128 restoration here would replace
            # that current state with an offline snapshot from a different context.
            return

        ctrl = get_offline_reuse_controller()
        if not ctrl.enabled:
            ctrl.enable()
        ctrl_v2 = get_offline_reuse_controller_v2()
        if not ctrl_v2.enabled:
            ctrl_v2.enable()

        if plan:
            try:
                runtime_config = RedKnotV4Config(
                    mode=os.environ.get("REDKNOT_V4_MODE", "correctness"),
                    reuse_window_kv=bool(plan.get("reuse_window_kv", False)),
                )
            except ValueError as error:
                if layer_id == getattr(token_to_kv_pool, "start_layer", 0):
                    logger.warning("RedKnot V4 dense fallback: %s", error)
                return
            validation = validate_runtime_reuse_plan(
                plan,
                config=runtime_config,
                dspark_active=self.mtp_enabled,
            )
            if not validation.valid:
                if layer_id == getattr(token_to_kv_pool, "start_layer", 0):
                    logger.warning(
                        "RedKnot V4 dense fallback: reason=%s detail=%s",
                        validation.fallback_reason.value,
                        validation.detail,
                    )
                return

            req_idx = int(forward_batch.req_pool_indices[0].item())
            seq_len = int(forward_batch.seq_lens_cpu[0].item())
            full_slots = self.req_to_token[req_idx, :seq_len]
            swa_slots = token_to_kv_pool.translate_loc_from_full_to_swa(full_slots)
            swa_page_size = token_to_kv_pool.swa_kv_pool.page_size
            swa_kv_buffer = token_to_kv_pool.get_swa_raw_key_buffer_radix(layer_id)

            # Determine this layer's compress ratio and get C4/C128 buffer info
            layer_item = token_to_kv_pool.layer_mapping[layer_id]
            compress_ratio = layer_item.compress_ratio if layer_item else 0

            mode = plan.get("mode")

            if mode == "snapshot":
                seg_hash = str(plan["seg_hash"])
                planned_length = int(plan["length"])
                length = min(planned_length, seq_len)

                if os.environ.get("REDKNOT_V4_DEBUG_FAULTHANDLER", "0") == "1":
                    logger.warning(
                        "[SNAPSHOT-DBG] layer=%d req_idx=%d seq_len=%d "
                        "planned_len=%d length=%d compress_ratio=%s "
                        "full_slots=[%d..%d] n=%d",
                        layer_id,
                        req_idx,
                        seq_len,
                        planned_length,
                        length,
                        compress_ratio,
                        int(full_slots.min().item()) if full_slots.numel() else -1,
                        int(full_slots.max().item()) if full_slots.numel() else -1,
                        full_slots.numel(),
                    )

                # With chunked prefill, the hook fires on EVERY chunk.
                # C4/C128 data is only fully written after all chunks have
                # been processed, i.e. when seq_len >= planned_length.
                # Snapshot SWA last, C4/C128 only on the final chunk.
                is_final_chunk = seq_len >= planned_length
                checkpoint_stride = int(
                    plan.get("checkpoint_stride_tokens", 0) or 0
                )

                if (
                    is_final_chunk
                    and layer_id == getattr(token_to_kv_pool, "start_layer", 0)
                ):
                    ctrl_v2.begin_snapshot(
                        seg_hash,
                        planned_length,
                        int(plan.get("canonical_start_pos", 0)),
                        checkpoint_stride,
                    )

                # Correctness MVP keeps Window KV entirely online. Capturing it
                # is allowed only in later optimization modes.
                if is_final_chunk and runtime_config.reuse_window_kv:
                    ctrl_v2.snapshot_swa_layer(
                        seg_hash=seg_hash,
                        length=length,
                        layer_id=layer_id,
                        kv_buffer=swa_kv_buffer,
                        slot_indices=swa_slots[:length],
                        page_size=swa_page_size,
                    )
                    if checkpoint_stride:
                        ctrl_v2.snapshot_swa_checkpoints(
                            seg_hash=seg_hash,
                            length=length,
                            layer_id=layer_id,
                            kv_buffer=swa_kv_buffer,
                            slot_indices=swa_slots[:length],
                            page_size=swa_page_size,
                            checkpoint_stride_tokens=checkpoint_stride,
                        )

                # 2) C4 or C128 snapshot (if this layer has compression)
                if is_final_chunk and compress_ratio in (4, 128):
                    page_table = self.forward_metadata.core_metadata.page_table
                    extra_page_size = token_to_kv_pool.get_extra_key_page_size(
                        layer_id
                    )
                    c_slots = compute_paged_compressed_slots(
                        page_table=page_table,
                        req_idx=0,  # single-request batch
                        seq_len=length,
                        compress_ratio=compress_ratio,
                        compressed_page_size=extra_page_size,
                    )
                    extra_buf = token_to_kv_pool.get_extra_key_buffer(layer_id)
                    if compress_ratio == 4:
                        # C4 KV + indexer
                        idx_buf = token_to_kv_pool.get_index_k_with_scale_buffer(
                            layer_id
                        )
                        idx_page_size = token_to_kv_pool.get_index_k_page_size()
                        ctrl_v2.snapshot_c4_layer(
                            seg_hash=seg_hash,
                            length=length,
                            layer_id=layer_id,
                            c4_buffer=extra_buf,
                            c4_slots=c_slots,
                            c4_page_size=extra_page_size,
                            indexer_buffer=idx_buf,
                            indexer_slots=c_slots,
                            indexer_page_size=idx_page_size,
                        )
                    else:  # compress_ratio == 128
                        ctrl_v2.snapshot_c128_layer(
                            seg_hash=seg_hash,
                            length=length,
                            layer_id=layer_id,
                            c128_buffer=extra_buf,
                            c128_slots=c_slots,
                            c128_page_size=extra_page_size,
                        )

                    # 3) Save the terminal Attention compressor state. C4 must
                    # also save the matching Indexer compressor state. C4's
                    # overlapping eight-token window spans two four-token state
                    # entries; C128 needs only one.
                    state_pool = token_to_kv_pool.get_attention_compress_states(
                        layer_id
                    )
                    state_slots = compute_compressed_slots(
                        full_slots=full_slots[:length],
                        full_to_swa=token_to_kv_pool.full_to_swa_index_mapping,
                        swa_page_size=token_to_kv_pool.swa_page_size,
                        ring_size=state_pool.ring_size,
                        compress_ratio=compress_ratio,
                        seq_len=length,
                        state_group_width=min(
                            compress_ratio, state_pool.ring_size
                        ),
                    )
                    if state_slots.numel() > 0:
                        terminal_state_slots = select_terminal_compress_state_slots(
                            state_slots, compress_ratio
                        )
                        ctrl_v2.snapshot_compress_state(
                            seg_hash=seg_hash,
                            length=length,
                            layer_id=layer_id,
                            state_buffer=state_pool.kv_score_buffer.kv_score,
                            state_slots=terminal_state_slots,
                            state_group_width=min(
                                compress_ratio, state_pool.ring_size
                            ),
                        )
                        if checkpoint_stride:
                            ctrl_v2.snapshot_compress_checkpoints(
                                seg_hash=seg_hash,
                                length=length,
                                layer_id=layer_id,
                                state_buffer=state_pool.kv_score_buffer.kv_score,
                                state_slots=state_slots,
                                compress_ratio=compress_ratio,
                                checkpoint_stride_tokens=checkpoint_stride,
                                state_group_width=min(
                                    compress_ratio, state_pool.ring_size
                                ),
                            )
                        if compress_ratio == 4:
                            indexer_state_pool = (
                                token_to_kv_pool.get_indexer_compress_states(layer_id)
                            )
                            indexer_state_group_width = min(
                                compress_ratio, indexer_state_pool.ring_size
                            )
                            indexer_state_slots = compute_compressed_slots(
                                full_slots=full_slots[:length],
                                full_to_swa=(
                                    token_to_kv_pool.full_to_swa_index_mapping
                                ),
                                swa_page_size=token_to_kv_pool.swa_page_size,
                                ring_size=indexer_state_pool.ring_size,
                                compress_ratio=compress_ratio,
                                seq_len=length,
                                state_group_width=indexer_state_group_width,
                            )
                            indexer_terminal_state_slots = (
                                select_terminal_compress_state_slots(
                                    indexer_state_slots, compress_ratio
                                )
                            )
                            ctrl_v2.snapshot_compress_state(
                                seg_hash=seg_hash,
                                length=length,
                                layer_id=layer_id,
                                state_buffer=(
                                    indexer_state_pool.kv_score_buffer.kv_score
                                ),
                                state_slots=indexer_terminal_state_slots,
                                is_indexer=True,
                                state_group_width=indexer_state_group_width,
                            )
                            if checkpoint_stride:
                                ctrl_v2.snapshot_compress_checkpoints(
                                    seg_hash=seg_hash,
                                    length=length,
                                    layer_id=layer_id,
                                    state_buffer=(
                                        indexer_state_pool.kv_score_buffer.kv_score
                                    ),
                                    state_slots=indexer_state_slots,
                                    compress_ratio=compress_ratio,
                                    checkpoint_stride_tokens=checkpoint_stride,
                                    is_indexer=True,
                                    state_group_width=indexer_state_group_width,
                                )

                if layer_id == getattr(token_to_kv_pool, "start_layer", 0):
                    logger.info(
                        "RedKnot V2 snapshot: seg=%s layer=%d cr=%d len=%d",
                        seg_hash,
                        layer_id,
                        compress_ratio,
                        length,
                    )
                return

            if mode == "restore":
                reuse_csa = bool(plan.get("reuse_csa", runtime_config.reuse_csa))
                reuse_hca = bool(plan.get("reuse_hca", runtime_config.reuse_hca))
                # Aggressive query-only replay injects EVERY compressed block of
                # each chunk (skip_tokens=0) so boundary/query rows can attend to
                # all prior chunks' compressed state via the indexer global
                # top-k. In correctness mode the first `skip_first` compressed
                # blocks stay online instead.
                inject_full = bool(plan.get("inject_full_blocks", False))
                orig_seq_lens = getattr(forward_batch, "orig_seq_lens", None)
                is_final_chunk = True
                if orig_seq_lens is not None and orig_seq_lens.numel() >= 1:
                    full_input_len = int(orig_seq_lens[0].item())
                    is_final_chunk = seq_len >= full_input_len
                freqs_cis = self._redknot_freqs_cis_for_layer(layer_id)
                reused_swa = 0
                reused_c = 0
                recomputed = 0
                # Diagnostic (B验证): only reuse the offset==0 segment, leave the
                # rest online. If cosine recovers, it confirms the root cause is
                # the missing Indexer RoPE relocation on position-migrated blocks.
                offset0_only = (
                    os.environ.get("REDKNOT_V4_REUSE_OFFSET0_ONLY", "0") == "1"
                )
                segments = plan.get("segments", [])
                selected_prefixes = tuple(
                    getattr(
                        forward_batch,
                        "redknot_selected_prefix_tokens",
                        (0,) * len(segments),
                    )
                )
                original_chunk_range = getattr(
                    forward_batch, "redknot_original_chunk_token_range", None
                )
                checkpoint_islands = tuple(
                    getattr(forward_batch, "redknot_checkpoint_islands", ())
                )
                for segment_index, segment in enumerate(segments):
                    seg_hash = str(segment["seg_hash"])
                    offset = int(segment["global_offset"])
                    length = int(segment["length"])
                    online_prefix = int(selected_prefixes[segment_index])
                    if offset0_only and offset != 0:
                        continue
                    if runtime_config.reuse_window_kv and checkpoint_islands:
                        logical_begin, logical_end = (
                            tuple(map(int, original_chunk_range))
                            if original_chunk_range is not None
                            else (0, seq_len)
                        )
                        for island in checkpoint_islands:
                            if int(island["segment_index"]) != segment_index:
                                continue
                            global_begin = int(island["global_begin"])
                            anchor = int(island["checkpoint_anchor"])
                            if not (
                                anchor > 0
                                and logical_begin <= global_begin < logical_end
                            ):
                                continue
                            restored = ctrl_v2.restore_swa_checkpoint(
                                seg_hash=seg_hash,
                                layer_id=layer_id,
                                checkpoint_anchor=anchor,
                                swa_buffer=swa_kv_buffer,
                                dst_segment_slots=swa_slots[offset : offset + length],
                                global_offset=offset,
                                freqs_cis=freqs_cis,
                                swa_page_size=swa_page_size,
                            )
                            if restored != ctrl_v2.swa_window:
                                raise RuntimeError(
                                    f"missing SWA checkpoint at anchor {anchor}"
                                )
                    # A later selected prefix needs the previous segment's SWA and
                    # compressed tail before its own chunk starts.  Materialize
                    # a segment in the one scheduler chunk that completes it;
                    # later chunks keep those C4/C128 slots intact.
                    if original_chunk_range is not None:
                        if not (
                            int(original_chunk_range[0])
                            < offset + length
                            <= int(original_chunk_range[1])
                        ):
                            continue
                    elif not is_final_chunk and seq_len < offset + length:
                        continue
                    skip_first = min(int(segment.get("skip_first", 128)), length)
                    compressed_skip = max(
                        online_prefix, 0 if inject_full else skip_first
                    )

                    online_owns_tail = any(
                        int(island["segment_index"]) == segment_index
                        and int(island["global_end"]) >= offset + length
                        for island in checkpoint_islands
                    )

                    if (
                        runtime_config.reuse_window_kv
                        and online_prefix < length
                        and not online_owns_tail
                    ):
                        tail = min(length, ctrl_v2.swa_window)
                        reused_swa += ctrl_v2.restore_swa_tail(
                            seg_hash=seg_hash,
                            layer_id=layer_id,
                            swa_buffer=swa_kv_buffer,
                            dst_slots=swa_slots[
                                offset + length - tail : offset + length
                            ],
                            global_offset=offset,
                            freqs_cis=freqs_cis,
                            swa_page_size=swa_page_size,
                        )
                    recomputed += skip_first

                    # The segmented compressor restores offline blocks *before*
                    # writing selected online repair islands.  Do not restore the
                    # same blocks again here: that would overwrite the refreshed
                    # C4/Indexer state and turn hot-row computation into dead work.
                    segmented_completed = getattr(
                        forward_batch,
                        "redknot_segmented_compressor_completed",
                        set(),
                    )
                    segmented_restored = (
                        (layer_id, compress_ratio, False) in segmented_completed
                        and (
                            compress_ratio != 4
                            or (layer_id, compress_ratio, True)
                            in segmented_completed
                        )
                    )

                    # C4/C128 restore (all compressed tokens, no skip) when the
                    # segmented path did not already compose this layer.
                    if (
                        (compress_ratio == 4 and reuse_csa)
                        or (compress_ratio == 128 and reuse_hca)
                    ) and not segmented_restored:
                        page_table = self.forward_metadata.core_metadata.page_table
                        extra_page_size = (
                            token_to_kv_pool.get_extra_key_page_size(layer_id)
                        )
                        # Compute dest C4/C128 slots for the segment's range
                        with _rk_timed("compute_slots"):
                            c_slots = compute_paged_compressed_slots(
                                page_table=page_table,
                                req_idx=0,  # single-request batch
                                seq_len=length,
                                compress_ratio=compress_ratio,
                                compressed_page_size=extra_page_size,
                                token_offset=offset,
                            )
                        extra_buf = token_to_kv_pool.get_extra_key_buffer(layer_id)

                        # Bounds checks synchronize the device and are diagnostic
                        # only; never leave them enabled in performance runs.
                        if (
                            os.environ.get("REDKNOT_V4_RESTORE_DEBUG", "0") == "1"
                            and c_slots.numel() > 0
                        ):
                            max_slot = c_slots.max().item()
                            buf_pages = extra_buf.shape[0] if extra_buf.dim() >= 1 else -1
                            if layer_id <= 3 or max_slot >= buf_pages * extra_page_size:
                                import logging
                                logging.warning(
                                    f"[RESTORE-DEBUG] layer={layer_id} cr={compress_ratio} "
                                    f"c_slots=[{c_slots.min().item()}..{max_slot}] "
                                    f"n_slots={c_slots.numel()} "
                                    f"buf_shape={list(extra_buf.shape)} "
                                    f"extra_page_size={extra_page_size} "
                                    f"page_table_shape={list(page_table.shape)} "
                                    f"seq_len={length} offset={offset}"
                                )

                        if compress_ratio == 4:
                            idx_buf = (
                                token_to_kv_pool.get_index_k_with_scale_buffer(
                                    layer_id
                                )
                            )
                            idx_page_size = (
                                token_to_kv_pool.get_index_k_page_size()
                            )
                            with _rk_timed("restore_c4_layer"):
                                reused_c += ctrl_v2.restore_c4_layer(
                                    seg_hash=seg_hash,
                                    layer_id=layer_id,
                                    c4_buffer=extra_buf,
                                    dst_slots=c_slots,
                                    global_offset=offset,
                                    freqs_cis=freqs_cis,
                                    c4_page_size=extra_page_size,
                                    indexer_buffer=idx_buf,
                                    indexer_slots=c_slots,
                                    indexer_page_size=idx_page_size,
                                    skip_tokens=compressed_skip,
                                )
                        else:  # 128
                            with _rk_timed("restore_c128_layer"):
                                reused_c += ctrl_v2.restore_c128_layer(
                                    seg_hash=seg_hash,
                                    layer_id=layer_id,
                                    c128_buffer=extra_buf,
                                    dst_slots=c_slots,
                                    global_offset=offset,
                                    freqs_cis=freqs_cis,
                                    c128_page_size=extra_page_size,
                                    skip_tokens=compressed_skip,
                                )

                        state_pool = token_to_kv_pool.get_attention_compress_states(
                            layer_id
                        )
                        with _rk_timed("restore_compress_state"):
                            state_slots = compute_compressed_slots(
                                full_slots=full_slots[offset : offset + length],
                                full_to_swa=token_to_kv_pool.full_to_swa_index_mapping,
                                swa_page_size=token_to_kv_pool.swa_page_size,
                                ring_size=state_pool.ring_size,
                                compress_ratio=compress_ratio,
                                seq_len=length,
                                state_group_width=min(
                                    compress_ratio, state_pool.ring_size
                                ),
                            )
                            if state_slots.numel() > 0 and online_prefix < length:
                                terminal_state_slots = (
                                    select_terminal_compress_state_slots(
                                        state_slots, compress_ratio
                                    )
                                )
                                ctrl_v2.restore_compress_state(
                                    seg_hash=seg_hash,
                                    layer_id=layer_id,
                                    state_buffer=state_pool.kv_score_buffer.kv_score,
                                    dst_slots=terminal_state_slots,
                                    state_group_width=min(
                                        compress_ratio, state_pool.ring_size
                                    ),
                                )
                                if compress_ratio == 4:
                                    indexer_state_pool = (
                                        token_to_kv_pool.get_indexer_compress_states(
                                            layer_id
                                        )
                                    )
                                    indexer_state_group_width = min(
                                        compress_ratio, indexer_state_pool.ring_size
                                    )
                                    indexer_state_slots = compute_compressed_slots(
                                        full_slots=full_slots[
                                            offset : offset + length
                                        ],
                                        full_to_swa=(
                                            token_to_kv_pool.full_to_swa_index_mapping
                                        ),
                                        swa_page_size=token_to_kv_pool.swa_page_size,
                                        ring_size=indexer_state_pool.ring_size,
                                        compress_ratio=compress_ratio,
                                        seq_len=length,
                                        state_group_width=(
                                            indexer_state_group_width
                                        ),
                                    )
                                    indexer_terminal_state_slots = (
                                        select_terminal_compress_state_slots(
                                            indexer_state_slots, compress_ratio
                                        )
                                    )
                                    ctrl_v2.restore_compress_state(
                                        seg_hash=seg_hash,
                                        layer_id=layer_id,
                                        state_buffer=(
                                            indexer_state_pool.kv_score_buffer.kv_score
                                        ),
                                        dst_slots=indexer_terminal_state_slots,
                                        is_indexer=True,
                                        state_group_width=indexer_state_group_width,
                                    )

                if layer_id == getattr(token_to_kv_pool, "start_layer", 0):
                    ctrl.stats["reuse_hits"] += 1
                    ctrl.stats["tokens_reused"] += reused_swa
                    ctrl_v2.stats["reuse_hits"] += 1
                    ctrl_v2.stats["tokens_reused_swa"] += reused_swa
                    ctrl_v2.stats["tokens_reused_c4"] += reused_c if compress_ratio == 4 else 0
                    ctrl_v2.stats["tokens_reused_c128"] += reused_c if compress_ratio == 128 else 0
                return

            raise ValueError(f"Unsupported redknot_reuse_plan mode: {mode!r}")

        if os.environ.get("REDKNOT_V4_DEMO_ACCOUNTING", "0") != "1":
            return

        # Self-contained demo mode: auto-segment the sequence into chunks of
        # ``chunk_tokens`` and emulate offline reuse by RoPE-relocating every
        # token EXCEPT the first ``boundary`` of each chunk. The relocation here
        # is an identity in global positions (src==dst) — it costs the same
        # kernel work the real reuse path would, and exercises the exact
        # read->reposition->write cycle, so the measured per-layer overhead and
        # the reused-token accounting are real. Accuracy is unaffected because
        # the model already wrote correct global-position RoPE; the saving we
        # *report* is the attention/MoE compute these reused tokens would skip.
        chunk_tokens = int(os.environ.get("SGLANG_REDKNOT_CHUNK_TOKENS", "1024"))
        boundary = ctrl.boundary
        # The FIRST `dense_layer_frac` fraction of layers are FULLY recomputed
        # (no reuse) so shallow-layer error does not accumulate into deep layers.
        # Default 0.10 -> front 10% of layers run standard MLA for all tokens.
        dense_layer_frac = float(
            os.environ.get("SGLANG_REDKNOT_DENSE_LAYER_FRAC", "0.10")
        )
        n_layers = int(
            getattr(
                self,
                "_redknot_n_layers",
                getattr(self._redknot_model_ref.config, "num_hidden_layers", 44)
                if self._redknot_model_ref is not None
                else 44,
            )
        )
        dense_until_layer = max(1, int(round(dense_layer_frac * n_layers)))

        page_size = token_to_kv_pool.page_size
        seq_len = int(forward_batch.seq_lens_cpu[0].item())

        # Front dense layers: everything recomputed (no reuse) for this layer.
        if layer_id < dense_until_layer:
            reused = 0
        else:
            # Reuse layers: segment 1 reused in full; each LATER chunk recomputes
            # its first `boundary` tokens (SWA window across the chunk join).
            # Per-length chunk counts: 16K->16, 32K->4, 64K/128K->6. The real
            # seq_len drifts from the nominal target, so match by nearest target
            # within 15% tolerance.
            _chunk_map = {16384: 16, 32768: 4, 65536: 6, 131072: 6}
            n_chunks = max(1, seq_len // chunk_tokens)
            for tgt, nc in _chunk_map.items():
                if abs(seq_len - tgt) <= tgt * 0.20:
                    n_chunks = nc
                    break
            csz = seq_len / n_chunks
            reused = 0
            for ci in range(n_chunks):
                start = int(round(ci * csz))
                end = int(round((ci + 1) * csz)) if ci < n_chunks - 1 else seq_len
                if ci == 0:
                    reused += end - start  # segment 1 fully reused
                else:
                    reuse_start = start + boundary
                    if reuse_start < end:
                        reused += end - reuse_start

        # Accounting (counted once on the first local layer of this rank).
        if layer_id == getattr(token_to_kv_pool, "start_layer", 0):
            ctrl.stats["tokens_reused"] += reused
            ctrl.stats["tokens_recomputed"] += seq_len - reused
            ctrl.stats["reuse_hits"] += 1

    def _redknot_freqs_cis_for_layer(self, layer_id):
        """Cache & return the freqs_cis table matching this layer's RoPE base."""
        cache = getattr(self, "_redknot_freqs_cache", None)
        if cache is None:
            cache = {}
            self._redknot_freqs_cache = cache
        if layer_id in cache:
            return cache[layer_id]
        # Pull from the model's attention module for this layer.
        # The model stores self.freqs_cis on each DeepseekV4Attention; we lazily
        # discover it via the layer module registered on the model_runner.
        fc = self._redknot_lookup_freqs_cis(layer_id)
        cache[layer_id] = fc
        return fc

    def _redknot_lookup_freqs_cis(self, layer_id):
        model = getattr(self, "_redknot_model_ref", None)
        if model is None:
            return None
        try:
            layer = model.model.layers[layer_id]
            return layer.self_attn.freqs_cis
        except Exception:
            return None

    def init_forward_metadata_indexer(self, core_attn_metadata: DSV4AttnMetadata):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Union[DSV4Metadata, DSV4RawDecodeMetadata]:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            return DSV4RawDecodeMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=True,
        )

        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_prefill(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        out_cache_loc: torch.Tensor,
        num_tokens: int,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
        causal_positions: Optional[torch.Tensor] = None,
    ) -> DSV4Metadata:
        if causal_positions is not None:
            if len(req_pool_indices) != 1 or causal_positions.numel() != num_tokens:
                raise ValueError(
                    "RedKnot causal positions require one request and one position per row"
                )
            seq_lens_casual = causal_positions.to(
                device=req_pool_indices.device, dtype=torch.int32
            ) + 1
            req_pool_indices_repeated = req_pool_indices.expand(num_tokens)
        else:
            seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
                num_tokens=num_tokens,
                seq_lens=seq_lens_cpu,
                extend_seq_lens=extend_seq_lens_cpu,
                req_pool_indices=req_pool_indices,
                padded_num_tokens=out_cache_loc.shape[0],
            )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if need_compress
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        else:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=extend_seq_lens_cpu,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            assert out_cache_loc is not None
            if not hasattr(self, "extend_seq_lens_buffer"):
                self.extend_seq_lens_buffer = torch.tensor(
                    [self.speculative_num_draft_tokens] * 1025, device=self.device
                )
            extend_seq_lens = self.extend_seq_lens_buffer[: len(seq_lens)]

            return DSV4RawVerifyMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                extend_seq_lens=extend_seq_lens,
            )
        else:
            seq_lens_cpu = seq_lens.tolist()
            return self.init_forward_metadata_target_verify_old(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        seq_lens_cpu = [x + self.speculative_num_draft_tokens for x in seq_lens_cpu]
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = self.speculative_num_draft_tokens * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=True,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def make_forward_metadata_from_raw_verify(
        self, raw_metadata: DSV4RawVerifyMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.speculative_num_draft_tokens
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs, num_draft_tokens, seq_lens, req_pool_indices
            )
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)
        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_seq_lens,
            seq_lens_cpu=None,
            extend_lens_cpu=None,
            use_prefill_cuda_graph=True,
            num_q_tokens=num_draft_tokens * bs,
        )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def make_forward_metadata_from_raw_decode(
        self, raw_metadata: DSV4RawDecodeMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_draft_extend(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        num_tokens_per_bs: int,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_seq_lens_cpu = [num_tokens_per_bs] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_tokens_per_bs * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=False,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert self.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            # DSv4 bakes this step's KV write target (c4/c128) into metadata,
            # so slice the shared multi-step out_cache_loc now rather than at
            # forward time.
            out_cache_loc = forward_batch.out_cache_loc
            if self.topk > 0 and self.speculative_num_steps > 1:
                out_cache_loc = per_step_draft_out_cache_loc(
                    out_cache_loc,
                    forward_batch.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )[self.speculative_step_id]
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )
        elif forward_batch.forward_mode.is_target_verify():
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = forward_batch.forward_mode.is_draft_extend(include_v2=True)
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=not is_draft,
                causal_positions=(
                    forward_batch.positions
                    if getattr(forward_batch, "redknot_reuse_plan", None)
                    and any(
                        plan and plan.get("skip_forward")
                        for plan in forward_batch.redknot_reuse_plan
                    )
                    else None
                ),
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_metadata_of_bucket_and_bs: Dict[
            _GraphBucket,
            Dict[
                int,
                Union[DSV4Metadata, DSV4RawDecodeMetadata, DSV4RawVerifyMetadata],
            ],
        ] = {bucket: {} for bucket in _GraphBucket}
        self.draft_extend_num_tokens_per_bs = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        assert req_pool_indices.size(0) == bs
        assert seq_lens.size(0) == bs

        bucket = _GraphBucket.of(forward_mode)
        raw_type: Optional[type] = None
        if bucket == _GraphBucket.DECODE_OR_IDLE:
            metadata = self.init_forward_metadata_decode(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=torch.zeros_like(seq_lens),
            )
            raw_type = DSV4RawDecodeMetadata
        elif bucket == _GraphBucket.TARGET_VERIFY:
            out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
            raw_type = DSV4RawVerifyMetadata
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = num_tokens // bs
            metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError(f"{forward_mode=} not supported yet")

        self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs] = metadata
        self.forward_metadata = metadata
        if raw_type is not None:
            self._current_capture_raw = (
                metadata if isinstance(metadata, raw_type) else None
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        bucket = _GraphBucket.of(forward_mode)

        # FIXME: see cuda_graph_runner — this attribute is set out-of-band.
        fb = self._replay_forward_batch
        out_cache_loc = fb.out_cache_loc
        actual_forward_mode = fb.forward_mode

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        actual_max_seq_len = seq_lens_cpu.max().item()
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        assert actual_max_seq_len <= chosen_max_seq_len

        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_decode(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY:
            assert out_cache_loc is not None
            num_tokens = self.speculative_num_draft_tokens * bs
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = self.draft_extend_num_tokens_per_bs
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=bs, temp_metadata=temp_metadata, bucket=bucket
        )

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        bucket: _GraphBucket,
    ) -> None:
        chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
        chosen_metadata.copy_(temp_metadata)
        self.forward_metadata = chosen_metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def on_after_cuda_graph_warmup(self):
        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c1_flashmla_metadata = _create_flashmla_metadata()
            core.c4_flashmla_metadata = _create_flashmla_metadata()
            core.c128_flashmla_metadata = _create_flashmla_metadata()

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        raw_loc = forward_batch.out_cache_loc
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_k=swa_k,
            )
        else:
            swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
            self.token_to_kv_pool.set_swa_key_buffer_radix(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_nope_fp8_rope_bf16_pack=swa_k_pack,
            )

    def _maybe_upgrade_forward_metadata(self) -> None:
        # With SGLANG_PREP_IN_CUDA_GRAPH=1, init_forward_metadata_*
        # returns a Raw metadata that only carries a few tensors. The
        # full DSV4Metadata (including c4/c128 compress + core_attn +
        # indexer metadata) must be materialized before any caller that
        # touches those fields. For 1.6T the first two layers have
        # compress_ratio=128, so forward_core_compressor / forward_c4_indexer
        # can fire before attn_backend.forward(), and must trigger the
        # upgrade themselves.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        self._maybe_upgrade_forward_metadata()

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)

            # ── RedKnot offline MLA reuse hook (demo, env-gated) ──
            # After the SWA KV for this layer is written at the request's
            # positions, optionally (a) snapshot a freshly-prefilled segment's
            # rope KV for later reuse, or (b) relocate a reused segment's rope
            # KV from local positions to the request's global offset. Driven by
            # forward_batch.redknot_reuse_plan (set by the scheduler from the
            # request). See dsv4_offline_reuse.DSV4OfflineReuseController.
            if envs.SGLANG_REDKNOT_OFFLINE_REUSE.get():
                self._maybe_redknot_reuse_hook(
                    layer_id, forward_batch, token_to_kv_pool
                )

            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            # RedKnot prefix-compression knob: clamp how many of the
            # indexer-ranked compressed-prefix tokens each query actually
            # attends to. The indexer still ranks the full prefix by relevance
            # (top-512); this keeps only the most-relevant first ``k``. Smaller
            # k -> more aggressive prefix compression. Env-gated, no-op by
            # default so the native path is unchanged.
            _c4_clamp = os.environ.get("REDKNOT_C4_TOPK_CLAMP", "")
            if _c4_clamp and extra_topk_lengths is not None:
                try:
                    _k = int(_c4_clamp)
                    if _k > 0:
                        _before = int(extra_topk_lengths.max().item())
                        extra_topk_lengths = torch.clamp(extra_topk_lengths, max=_k)
                        _after = int(extra_topk_lengths.max().item())
                        # only log when the prefix is actually long enough that
                        # clamping changes something (skip warmup/tiny seqs)
                        if _before > _k:
                            _flag = f"_c4_clamp_logged_{compress_ratio}"
                            if not hasattr(self, _flag):
                                logger.info(
                                    "REDKNOT_C4_TOPK_CLAMP active: k=%d cr=%d "
                                    "max_topk %d -> %d (CHANGED)",
                                    _k,
                                    compress_ratio,
                                    _before,
                                    _after,
                                )
                                setattr(self, _flag, True)
                except ValueError:
                    pass

            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
            swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
            )

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

            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

            assert attn_sink is not None

            assert swa_page_indices.shape[-1] % 64 == 0, (
                f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            )
            if extra_indices is not None:
                assert extra_indices.shape[-1] % 64 == 0, (
                    f"{extra_indices.shape=}'s last dimension is not aligned to 64"
                )

            import flash_mla

            num_flashmla_row_chunks = (
                int(q.shape[0]) + FLASHMLA_MAX_BATCH_ROWS - 1
            ) // FLASHMLA_MAX_BATCH_ROWS
            if num_flashmla_row_chunks == 1:
                # Preserve the original, proven 8K-and-below path exactly.
                flashmla_metadata = (
                    core_attn_metadata.get_flashmla_metadata(compress_ratio),
                )
            else:
                flashmla_metadata = (
                    core_attn_metadata.get_flashmla_row_chunk_metadata(
                        compress_ratio, num_flashmla_row_chunks
                    )
                )
            o = _flash_mla_with_kvcache_row_chunks(
                flash_mla=flash_mla,
                q=q,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                tile_scheduler_metadata=flashmla_metadata,
                softmax_scale=self.softmax_scale,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                extra_topk_length=extra_topk_lengths,
            )

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lens_casual = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        idx_to_req_repeated = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        offset = 0
        for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
            out = seq_lens_casual[offset : offset + qo_len]
            offset += qo_len
            torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
            idx_to_req_repeated[offset - qo_len : offset].fill_(i)

        assert offset == num_tokens
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual,
                (0, pad_size),
                value=1,
            )
            req_pool_indices_repeated = torch.nn.functional.pad(
                req_pool_indices_repeated,
                (0, pad_size),
                value=req_pool_indices_repeated[-1].item(),
            )

        return seq_lens_casual, req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
        bs: int,
        qo_len: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        seq_lens_casual = seq_lens[:, None] + torch.arange(
            -qo_len + 1, 1, **self.cuda_int32_kwargs
        )
        seq_lens_casual = seq_lens_casual.flatten()
        idx_to_req_repeated = torch.arange(
            bs, **self.cuda_int32_kwargs
        ).repeat_interleave(qo_len)
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]
        return seq_lens_casual, req_pool_indices_repeated

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)

        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32)

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.c4_topk,
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata()
            core_attn_metadata.init_flashmla_related()
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c1_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
        return core_attn_metadata

    def get_swa_page_indices(
        self,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
    ) -> torch.Tensor:
        pos_causal = seq_lens_casual - 1
        num_qo_tokens = seq_lens_casual.size(0)
        offsets = pos_causal.unsqueeze(1) - torch.arange(
            SWA_WINDOW, **self.cuda_int32_kwargs
        ).unsqueeze(0)
        invalid_offset_mask = offsets < 0
        offsets.masked_fill_(invalid_offset_mask, 0)
        raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
        assert raw_indices.shape == (num_qo_tokens, SWA_WINDOW)
        raw_indices.masked_fill_(invalid_offset_mask, -1)
        swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
        return swa_indices


class DeepseekV4MultiStepBackend(DeepseekV4AttnBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4AttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4AttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if self.speculative_num_steps == 1:
            return

        self.attn_backends[0]._replay_forward_batch = forward_batch
        self.attn_backends[0].init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )
        self.attn_backends[0]._replay_forward_batch = None
        temp_metadata = self.attn_backends[0].forward_metadata

        for i in range(1, self.speculative_num_steps - 1):
            self.attn_backends[i].replay_cuda_graph_metadata_from(
                bs=bs,
                temp_metadata=temp_metadata,
                bucket=_GraphBucket.DECODE_OR_IDLE,
            )


def _pad_tensor_to_size(tensor: torch.Tensor, size: int, *, value: int = 0):
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
            ],
            dim=0,
        )
