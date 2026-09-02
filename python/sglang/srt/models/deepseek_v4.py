from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.dsv4 import (
    fused_norm_rope_inplace,
    fused_q_norm_rope,
    fused_rope_inplace,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    attn_tp_all_gather,
    dp_gather_partial,
    dp_scatter,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import (
    compile_in_capture_mode,
    get_is_capture_mode,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)

from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu

if not _is_hip:
    from sglang.srt.layers.utils.cp_utils import (
        prepare_context_parallel_metadata,
    )

from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    get_bool_env_var,
    is_gfx95_supported,
    log_info_on_rank0,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

# Keep the production path independent of the optional timing module.  The
# environment is fixed before worker startup; when timing is disabled every
# callsite reuses one stateless no-op context without importing timing state.
_REDKNOT_V4_TIMING_NOOP = nullcontext()
if os.environ.get("REDKNOT_V4_TIMING", "0") == "1":
    from sglang.srt.layers.attention.redknot.dsv4_timing import (
        timed as _redknot_v4_timed,
    )
else:

    def _redknot_v4_timed(*_args, **_kwargs):
        return _REDKNOT_V4_TIMING_NOOP


def _pack_positionless_swa_record(pack: object) -> torch.Tensor:
    """Materialize the native DSV4 pack as canonical 584-byte rows.

    ``quant_to_nope_fp8_rope_bf16_pack_triton`` returns a structured pack,
    while the shared-latent artifact deliberately owns the byte-exact storage
    representation consumed by FlashMLA.  Preserve every native bit here; in
    particular, casting FP8/UE8M0 values to uint8 would change their values
    instead of exposing their storage bytes.
    """

    field_names = (
        "k_nope_fp8",
        "k_rope_bf16",
        "scale_k_nope_ue8m0",
    )
    try:
        k_nope_fp8, k_rope_bf16, scale_k_nope_ue8m0 = (
            getattr(pack, field_name) for field_name in field_names
        )
    except AttributeError as error:
        raise TypeError("native DSV4 SWA pack is missing a tensor field") from error

    components = (k_nope_fp8, k_rope_bf16, scale_k_nope_ue8m0)
    if not all(isinstance(component, torch.Tensor) for component in components):
        raise TypeError("native DSV4 SWA pack fields must be tensors")
    if k_nope_fp8.ndim != 2 or int(k_nope_fp8.shape[1]) != 448:
        raise ValueError("native DSV4 no-PE FP8 pack must have shape [rows, 448]")
    rows = int(k_nope_fp8.shape[0])
    if k_rope_bf16.ndim != 2 or tuple(k_rope_bf16.shape) != (rows, 64):
        raise ValueError("native DSV4 RoPE pack must have shape [rows, 64]")
    if (
        scale_k_nope_ue8m0.ndim != 2
        or tuple(scale_k_nope_ue8m0.shape) != (rows, 7)
    ):
        raise ValueError("native DSV4 no-PE scale pack must have shape [rows, 7]")
    if int(k_nope_fp8.element_size()) != 1:
        raise TypeError("native DSV4 no-PE FP8 values must occupy one byte")
    if k_rope_bf16.dtype != torch.bfloat16:
        raise TypeError("native DSV4 RoPE values must be bfloat16")
    if int(scale_k_nope_ue8m0.element_size()) != 1:
        raise TypeError("native DSV4 UE8M0 scales must occupy one byte")
    if any(component.device != k_nope_fp8.device for component in components[1:]):
        raise ValueError("native DSV4 SWA pack fields must share one device")

    nope_bytes = k_nope_fp8.contiguous().view(torch.uint8).reshape(rows, 448)
    rope_bytes = k_rope_bf16.contiguous().view(torch.uint8).reshape(rows, 128)
    scale_bytes = (
        scale_k_nope_ue8m0.contiguous().view(torch.uint8).reshape(rows, 7)
    )
    scale_pad = torch.zeros(
        (rows, 1), dtype=torch.uint8, device=k_nope_fp8.device
    )
    canonical = torch.cat(
        (nope_bytes, rope_bytes, scale_bytes, scale_pad), dim=1
    ).contiguous()
    if (
        canonical.dtype != torch.uint8
        or canonical.ndim != 2
        or tuple(canonical.shape) != (rows, 584)
        or not canonical.is_contiguous()
    ):
        raise RuntimeError("canonical DSV4 SWA pack geometry changed")
    return canonical


# Process-local accumulator for RedKnot sparse-FFN statistics. Tracks the number
# of tokens that actually ran the MoE vs the total tokens seen on sparse-eligible
# layers, so a benchmark can report the *measured* MoE-FLOPs saving (keep ratio)
# rather than a configured threshold. Read/reset via the helpers below; a summary
# line is emitted to the log when SGLANG_REDKNOT_FFN_DEBUG=1.
_REDKNOT_FFN_STATS = {"kept": 0, "total": 0, "sparse_layer_calls": 0}


def redknot_ffn_stats_snapshot() -> dict:
    s = _REDKNOT_FFN_STATS
    kept, total = s["kept"], s["total"]
    full_ratio = (kept / total) if total > 0 else 1.0
    # DSV4 activates one shared plus six routed experts for FULL tokens;
    # SHARED_ONLY tokens retain one of those seven expert FFNs.
    expert_compute_ratio = (
        (kept * 7 + (total - kept)) / (total * 7) if total > 0 else 1.0
    )
    return {
        "kept_tokens": kept,
        "shared_only_tokens": total - kept,
        "total_tokens": total,
        "sparse_layer_calls": s["sparse_layer_calls"],
        "keep_ratio": full_ratio,
        "full_ratio": full_ratio,
        "expert_compute_ratio": expert_compute_ratio,
        "expert_compute_saving": 1.0 - expert_compute_ratio,
    }


def redknot_ffn_stats_reset() -> None:
    _REDKNOT_FFN_STATS.update(kept=0, total=0, sparse_layer_calls=0)


def _redknot_ffn_stats_record(kept: int, total: int) -> None:
    _REDKNOT_FFN_STATS["kept"] += int(kept)
    _REDKNOT_FFN_STATS["total"] += int(total)
    _REDKNOT_FFN_STATS["sparse_layer_calls"] += 1
    # Emit a cumulative summary periodically so a benchmark can grep the keep
    # ratio (= measured MoE-FLOPs fraction on sparse layers) from the rank log.
    if (
        os.environ.get("SGLANG_REDKNOT_FFN_DEBUG") == "1"
        and _REDKNOT_FFN_STATS["sparse_layer_calls"] % 10 == 0
    ):
        snap = redknot_ffn_stats_snapshot()
        logger.info(
            "[RedKnot sparse-FFN STATS] sparse_layer_calls=%d full=%d "
            "shared_only=%d total=%d full_ratio=%.4f "
            "expert_compute_ratio=%.4f",
            snap["sparse_layer_calls"],
            snap["kept_tokens"],
            snap["shared_only_tokens"],
            snap["total_tokens"],
            snap["full_ratio"],
            snap["expert_compute_ratio"],
        )


_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_gfx95_supported = is_gfx95_supported()

if _use_aiter:
    if _is_gfx95_supported:
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant


def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):
    x_quant, x_bf16, _, _ = fused_rms_fp8_group_quant(
        hidden_states,
        weight,
        eps,
        inp2=None,
        inp2_weight=None,
        inp2_epsilon=None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        output_unquantized_inp1=True,
    )
    return x_quant, x_bf16


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight

    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)

    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class MQALayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self._mla_profile_last_layer = int(config.num_hidden_layers) - 1
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else config.compress_ratios[layer_id]
        )
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio

        assert self.head_dim == config.head_dim
        assert config.num_key_value_heads == 1

        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta

        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=get_global_server_args().device,
        )

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        assert self.compress_ratio in {0, 4, 128}
        if self.compress_ratio:
            original_seq_len = rope_scaling["original_max_position_embeddings"]
        else:
            original_seq_len = 0

        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if _is_hip:
            cos_cache = freqs_cis.real.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            sin_cache = freqs_cis.imag.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            self.register_buffer("cos_cache", cos_cache, persistent=False)
            self.register_buffer("sin_cache", sin_cache, persistent=False)

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
                rotary_emb=getattr(self, "rotary_emb", None),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                    rotary_emb=getattr(self, "rotary_emb", None),
                )

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.fuse_wqa_wkv = not _is_hip and envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        if self.fuse_wqa_wkv:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            assert hasattr(self.wo_a, "weight_scale_inv"), (
                "FP8 quant_config must create weight_scale_inv"
            )
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.use_fused_qk_norm_rope = (
            _is_hip and envs.SGLANG_OPT_USE_FUSED_QK_NORM_ROPE.get()
        )

        # KV cache write is always fused into the K kernel
        # (`_compute_kv_to_cache`), so the legacy "overlap store cache" flag
        # has no effect here -- the fused path is on by default.

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        return self.q_norm(q)

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if q_out is None:
            q_out = torch.empty_like(q)
        # Fused warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        return q_out

    def _compute_q_b_sparse_restore(
        self,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        mla_off_context,
    ):
        """Project rank-local global/dirty-local Q domains exactly once.

        ``wq_b`` is already column-parallel, but the native implementation
        still evaluates all eight owned heads on every row.  For a certified
        MLA restore, global heads remain full-row while offline-local heads
        need Q only on the boundary/query rows that attention will execute.
        Canonical checkpoint block scales are sliced on output-block rows.
        When startup has instead converted them to DeepGEMM's N-dependent
        packed UE8M0 layout, each contiguous head run is inverse-laid-out,
        sliced in canonical block space, and repacked for its new output width.
        Direct packed-scale slicing is forbidden.
        """

        if not (
            mla_off_context is not None
            and getattr(mla_off_context, "is_restore", False)
            and int(getattr(mla_off_context, "reused_row_count", 0)) > 0
        ):
            raise ValueError("sparse-Q requires a reusable MLA restore context")
        if self.use_fused_qk_norm_rope:
            raise ValueError("sparse-Q does not support the fused HIP Q path")
        if q_lora.ndim != 2 or positions.ndim != 1:
            raise ValueError("sparse-Q requires flat prefill tensors")
        if int(q_lora.shape[0]) != int(positions.numel()):
            raise ValueError("sparse-Q rows do not match positions")

        from sglang.srt.layers.attention.redknot.dsv4_sparse_q import (
            build_sparse_q_plan,
            materialize_sparse_q_run_scale,
            ue8m0_scale_slice_for_head_run,
        )

        online_rows = mla_off_context.online_local_row_indices
        online_rows_cpu = mla_off_context.online_local_row_indices_cpu
        if (
            not isinstance(online_rows, torch.Tensor)
            or online_rows.ndim != 1
            or online_rows.dtype != torch.long
            or online_rows.device != q_lora.device
            or not isinstance(online_rows_cpu, torch.Tensor)
            or online_rows_cpu.ndim != 1
            or online_rows_cpu.dtype != torch.long
            or online_rows_cpu.device.type != "cpu"
        ):
            raise ValueError("sparse-Q online row certificate is incomplete")
        plan = build_sparse_q_plan(
            self.tp_rank,
            self.tp_size,
            self.n_heads,
            tuple(int(axis) for axis in mla_off_context.local_head_axes),
            int(q_lora.shape[0]),
            tuple(int(value) for value in online_rows_cpu.tolist()),
            layer_id=self.layer_id,
            head_dim=self.head_dim,
        )
        if plan.owned_head_count != self.n_local_heads:
            raise ValueError("sparse-Q plan does not match wq_b TP sharding")
        mla_off_context.sparse_q_plan = plan

        method = getattr(self.wq_b, "quant_method", None)
        quant_config = getattr(method, "quant_config", None)
        block_size = getattr(quant_config, "weight_block_size", None)
        partial_linear = getattr(method, "w8a8_block_fp8_linear", None)
        if method is None:
            raise ValueError("sparse-Q requires an FP8 wq_b quantization method")
        if bool(getattr(method, "use_marlin", False)):
            raise ValueError("sparse-Q does not support Marlin-repacked wq_b weights")
        if (
            not bool(getattr(method, "block_quant", False))
            or bool(getattr(method, "use_mxfp8", False))
            or not callable(partial_linear)
            or tuple(block_size or ()) != (128, 128)
            or not hasattr(self.wq_b, "weight_scale_inv")
        ):
            raise ValueError("sparse-Q requires block-FP8 wq_b with 128x128 scales")

        from sglang.srt.layers.quantization.fp8_utils import (
            _use_aiter_bpreshuffle_gfx95,
            aiter_w8a8_block_fp8_linear,
            inverse_transform_scale_ue8m0,
            transform_scale_ue8m0,
        )

        # AITER's gfx95 bpreshuffle mutates the full weight in place, and the
        # kernel decision is N-dependent.  A sparse head run can therefore
        # select a different layout than the original full-N linear.  Do not
        # reinterpret or partially slice that backend-specific representation.
        if (
            _use_aiter_bpreshuffle_gfx95
            and partial_linear is aiter_w8a8_block_fp8_linear
        ):
            raise ValueError("sparse-Q does not support AITER-bpreshuffled wq_b weights")

        weight = self.wq_b.weight
        weight_scale = self.wq_b.weight_scale_inv
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError("sparse-Q requires a two-dimensional wq_b weight")
        if not isinstance(weight_scale, torch.Tensor) or weight_scale.ndim != 2:
            raise ValueError("sparse-Q requires a two-dimensional wq_b scale")
        full_output_rows = int(weight.shape[0])
        input_size = int(weight.shape[1])
        fp8_weight_dtypes = (torch.float8_e4m3fn,)
        fp8_fnuz_dtype = getattr(torch, "float8_e4m3fnuz", None)
        if fp8_fnuz_dtype is not None:
            fp8_weight_dtypes += (fp8_fnuz_dtype,)
        if (
            full_output_rows != self.n_local_heads * self.head_dim
            or input_size != self.q_lora_rank
            or full_output_rows % 128
            or input_size % 128
        ):
            raise ValueError("sparse-Q wq_b weight geometry changed")
        if (
            weight.dtype not in fp8_weight_dtypes
            or not weight.is_contiguous()
            or tuple(int(value) for value in weight.stride()) != (input_size, 1)
            or bool(getattr(weight, "is_shuffled", False))
        ):
            raise ValueError(
                "sparse-Q requires an ordinary contiguous E4M3 wq_b weight layout"
            )
        canonical_scale = bool(
            weight_scale.dtype == torch.float32
            and weight_scale.ndim == 2
            and tuple(int(value) for value in weight_scale.shape)
            == (full_output_rows // 128, input_size // 128)
        )
        packed_scale_shape = (
            full_output_rows,
            ((input_size // 128) + 3) // 4,
        )
        packed_scale_stride = (1, full_output_rows)
        packed_scale = bool(
            weight_scale.dtype == torch.int32
            and bool(getattr(weight_scale, "format_ue8m0", False))
            and weight_scale.ndim == 2
            and tuple(int(value) for value in weight_scale.shape)
            == packed_scale_shape
            and tuple(int(value) for value in weight_scale.stride())
            == packed_scale_stride
        )
        if not (canonical_scale or packed_scale):
            raise ValueError(
                "sparse-Q requires canonical FP32 or packed UE8M0 wq_b scales"
            )

        scale_cache = getattr(self, "_redknot_sparse_q_scale_cache", None)
        cache_identity = (
            int(weight.data_ptr()),
            int(weight_scale.data_ptr()),
            tuple(int(value) for value in weight.shape),
            tuple(int(value) for value in weight_scale.shape),
            tuple(int(value) for value in weight_scale.stride()),
            str(weight_scale.dtype),
            bool(getattr(weight_scale, "format_ue8m0", False)),
        )
        if not isinstance(scale_cache, dict) or scale_cache.get(
            "identity"
        ) != cache_identity:
            scale_cache = {"identity": cache_identity, "runs": {}}
            self._redknot_sparse_q_scale_cache = scale_cache

        def project_run(
            source: torch.Tensor,
            run,
            run_positions: torch.Tensor,
        ) -> torch.Tensor:
            run_key = (int(run.start_axis), int(run.end_axis))
            run_scales = scale_cache["runs"]
            run_scale = run_scales.get(run_key)
            if run_scale is None:
                geometry = ue8m0_scale_slice_for_head_run(
                    run,
                    full_output_rows=full_output_rows,
                    input_size=input_size,
                    block_shape=(128, 128),
                )
                run_scale = materialize_sparse_q_run_scale(
                    weight_scale,
                    geometry,
                    inverse_transform=inverse_transform_scale_ue8m0,
                    transform=transform_scale_ue8m0,
                )
                run_scales[run_key] = run_scale
            run_weight = weight.narrow(
                0, int(run.output_row_start), int(run.output_rows)
            )
            projected = partial_linear(
                input=source,
                weight=run_weight,
                block_size=block_size,
                weight_scale=run_scale,
                input_scale=None,
                bias=None,
            ).view(-1, int(run.head_count), self.head_dim)
            normalized = torch.empty_like(projected)
            fused_q_norm_rope(
                projected,
                normalized,
                self.eps,
                self.freqs_cis,
                run_positions,
            )
            return normalized

        # Materialize only projected head rows.  In particular, do not allocate
        # either [T,64,D] or a holey rank-local [T,H_owned,D] activation whose
        # clean-local slots are never consumed.
        from sglang.srt.layers.attention.redknot.dsv4_sparse_q_runtime import (
            PackedSparseQBuilder,
        )

        transaction = getattr(
            mla_off_context, "_redknot_forward_composite_transaction", None
        )
        sequential_arena = getattr(transaction, "q_arena", None)
        if sequential_arena is not None:
            reservation = sequential_arena.reservation_for(self.layer_id)
            if str(reservation.plan.digest) != str(plan.digest):
                raise ValueError(
                    "sequential Q reservation differs from the live sparse plan"
                )
            plan = reservation.plan
            mla_off_context.sparse_q_plan = plan
            sparse_q = sequential_arena.begin_layer(
                self.layer_id, online_rows
            )
            packed_values = reservation.values
        else:
            packed_values = torch.empty(
                (int(plan.projected_head_rows), self.head_dim),
                device=q_lora.device,
                dtype=q_lora.dtype,
            )
            sparse_q = PackedSparseQBuilder(
                plan=plan,
                values=packed_values,
                local_rows=online_rows,
            )
        for run in plan.global_runs:
            run_q = project_run(q_lora, run, positions)
            sparse_q.write(
                scope="global",
                start_axis=int(run.start_axis),
                end_axis=int(run.end_axis),
                projected=run_q,
            )
        if int(online_rows.numel()) > 0:
            local_source = q_lora.index_select(0, online_rows)
            local_positions = positions.index_select(0, online_rows)
            for run in plan.local_runs:
                run_q = project_run(local_source, run, local_positions)
                sparse_q.write(
                    scope="local",
                    start_axis=int(run.start_axis),
                    end_axis=int(run.end_axis),
                    projected=run_q,
                )
        elif plan.local_runs:
            # Zero-row local runs occupy no arena space, but still belong to
            # the immutable layout and must be marked complete.  Construct
            # their exact logical shape directly: a zero-sized dimension
            # cannot be expanded from 0 to the run's nonzero head count.
            for run in plan.local_runs:
                empty = q_lora.new_empty(
                    (0, int(run.head_count), self.head_dim)
                )
                sparse_q.write(
                    scope="local",
                    start_axis=int(run.start_axis),
                    end_axis=int(run.end_axis),
                    projected=empty,
                )
        if sequential_arena is not None:
            return sparse_q.finish()
        return sparse_q.finish(
            projection_token=(
                f"layer:{self.layer_id}:plan:{plan.digest}:"
                f"storage:{int(packed_values.data_ptr())}"
            )
        )

    def _compute_kv_to_cache(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        qkv_a: Optional[torch.Tensor] = None,
        row_indices: Optional[torch.Tensor] = None,
        capture_positionless: bool = False,
    ) -> Optional[torch.Tensor]:
        """Fused: rmsnorm + RoPE + write directly to FlashMLA paged cache.

        Replaces the bf16-kv-intermediate path. Used everywhere except the DSA
        prefill-CP case (which needs bf16 kv for the cross-rank all-gather).
        """
        raw_loc = forward_batch.out_cache_loc
        if row_indices is not None:
            if (
                row_indices.ndim != 1
                or row_indices.dtype != torch.long
                or row_indices.device != x.device
            ):
                raise ValueError("dirty KV rows must be a device int64 vector")
            if qkv_a is not None:
                raise ValueError(
                    "dirty-only KV is incompatible with fused wqkv_a; "
                    "the shared-latent profile must de-fuse wq_a/wkv"
                )
            if int(row_indices.numel()) == 0:
                return None
            x = x.index_select(0, row_indices)
            positions = positions.index_select(0, row_indices)
            raw_loc = raw_loc.index_select(0, row_indices)
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        positionless_packed = None
        if bool(capture_positionless):
            if row_indices is not None:
                raise ValueError(
                    "snapshot canonical KV capture requires the complete row set"
                )
            # Snapshot is an offline producer, so it may materialize one
            # canonical packed copy while the serving restore path remains
            # dirty-only.  Applying the exact native norm/RoPE kernel at
            # position zero yields the position-independent 584-byte record;
            # this avoids rereading an SWA ring after an 8K fused writer may
            # already have recycled physical slots.
            canonical = kv.contiguous().clone()
            zero_positions = torch.zeros_like(positions)
            fused_norm_rope_inplace(
                canonical,
                self.kv_norm.weight.data,
                self.eps,
                self.freqs_cis,
                zero_positions,
            )
            from sglang.srt.layers.attention.dsv4.quant_k_cache import (
                quant_to_nope_fp8_rope_bf16_pack_triton,
            )

            native_positionless_pack = quant_to_nope_fp8_rope_bf16_pack_triton(
                canonical.bfloat16()
            )
            positionless_packed = _pack_positionless_swa_record(
                native_positionless_pack
            )
            if (
                positionless_packed.dtype != torch.uint8
                or positionless_packed.ndim != 2
                or int(positionless_packed.shape[0]) != int(x.shape[0])
                or int(positionless_packed.shape[1]) != 584
            ):
                raise RuntimeError(
                    "canonical shared-latent SWA pack geometry changed"
                )
        token_to_kv_pool = get_token_to_kv_pool()
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            raw_loc=raw_loc,
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
            cache_translation=row_indices is None,
        )
        return positionless_packed

    def _compute_kv_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bf16-kv path used by the DSA prefill-CP case (needs all-gather)."""
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        kv = kv.contiguous()
        fused_norm_rope_inplace(
            kv,
            self.kv_norm.weight.data,
            self.eps,
            self.freqs_cis,
            positions,
        )
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        x_linear = x_quant if x_quant is not None else x
        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            qkv_a_ready = current_stream.record_event()

        q_lora = self._compute_q_a(x_linear, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            # Fused norm + rope + cache write -- no bf16 KV intermediate.
            self._compute_kv_to_cache(x_linear, positions, forward_batch, qkv_a=qkv_a)

        del qkv_a

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_lora, positions, q_out)
        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q

    def _forward_prepare_multi_stream_hip(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        """ATOM-style ROCm path: overlap compressors, keep Q/KV on main stream."""
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 1

        current_stream = torch.cuda.current_stream()
        stream_compressor = self.alt_streams[0]
        stream_indexer_compressor = (
            self.alt_streams[1] if len(self.alt_streams) > 1 else None
        )

        if self.compressor is not None:
            stream_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        if self.indexer is not None and stream_indexer_compressor is not None:
            stream_indexer_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_indexer_compressor):
                attn_backend.forward_indexer_compressor(
                    x=x,
                    forward_batch=forward_batch,
                    layer_id=self.indexer.layer_id,
                    compressor=self.indexer.compressor,
                )

        x_linear = x_quant if x_quant is not None else x
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        if self.use_fused_qk_norm_rope:
            if _is_gfx95_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                )
                q, _ = self.wq_b(q_for_wqb)
            else:
                q_lora = self.q_norm(q_lora)
                q, _ = self.wq_b(q_lora)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            from sglang.srt.layers.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            token_to_kv_pool = get_token_to_kv_pool()
            swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )
            swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]
            swa_page_size = token_to_kv_pool.swa_kv_pool.page_size

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
            )
        else:
            q_lora = self.q_norm(q_lora)
            q = self._compute_q_b(q_lora, positions, q_out)
            self._compute_kv_to_cache(x_linear, positions, forward_batch, qkv_a=qkv_a)

        if self.indexer is not None:
            current_stream.wait_stream(stream_compressor)
            if stream_indexer_compressor is not None:
                current_stream.wait_stream(stream_indexer_compressor)
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                skip_compressor=True,
            )
        elif self.compressor is not None:
            current_stream.wait_stream(stream_compressor)

        return q

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
        mla_off_context=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_linear = x_quant if x_quant is not None else x
        diagnostic_ablation = str(
            getattr(mla_off_context, "diagnostic_ablation", "full") or "full"
        )
        diagnostic_zoff_only = diagnostic_ablation == "zoff_only"
        diagnostic_shared_only = diagnostic_ablation == "shared_only"
        shared_restore_requested = bool(
            mla_off_context is not None
            and getattr(mla_off_context, "is_restore", False)
            and tuple(getattr(mla_off_context, "shared_restore_states", ()) or ())
        )
        shared_snapshot_requested = bool(
            mla_off_context is not None
            and getattr(mla_off_context, "is_snapshot", False)
            and getattr(mla_off_context, "shared_snapshot_enabled", False)
            and callable(
                getattr(
                    attn_backend,
                    "capture_mla_off_shared_snapshot_chunk",
                    None,
                )
            )
        )
        if shared_snapshot_requested:
            # A context must never carry a prior layer/attempt's activation
            # into this KV producer.  The current layer installs a fresh value
            # only after its canonical position-zero write succeeds.
            self._clear_mla_off_ephemeral_snapshot_state(mla_off_context)
        if shared_restore_requested and self.fuse_wqa_wkv:
            # A fused wqkv_a evaluates the 512 KV output columns for every
            # clean row before sparse-Q/shared-cache preflight can omit them.
            # Production shared-latent mode therefore requires the launcher to
            # expose independent wq_a and wkv modules.  Silently using the
            # fused tensor would preserve correctness while destroying the
            # claimed KV saving.
            raise RuntimeError(
                "shared-latent restore requires SGLANG_OPT_FUSE_WQA_WKV=0"
            )
        if (
            (shared_restore_requested or shared_snapshot_requested)
            and self.use_fused_qk_norm_rope
        ):
            # The HIP fused helper owns a single all-row Q+KV invocation and
            # cannot represent packed global/all-row plus local/dirty-row Q,
            # nor a dirty-only KV destination.  Falling through would be
            # numerically correct but would silently pay the full projection
            # and overwrite the clean shared-latent rows.
            raise RuntimeError(
                "shared-latent restore is incompatible with the fused HIP "
                "QK-normalize/RoPE path"
            )
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        if shared_restore_requested and use_cp:
            # CP all-gathers the complete bf16 latent before writing cache.
            # A dirty-row collective and matching destination permutation are
            # not implemented, so advertising clean-row KV omission here
            # would be false.  Keep this a hard serving contract.
            raise RuntimeError(
                "shared-latent restore does not support DSA prefill context "
                "parallelism"
            )
        kv: Optional[torch.Tensor]
        shared_restore_committed = False
        zoff_restore_committed = False

        if self.use_fused_qk_norm_rope:
            if _is_gfx95_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                )
                q, _ = self.wq_b(q_for_wqb)
            else:
                q_lora = self.q_norm(q_lora)
                q, _ = self.wq_b(q_lora)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            from sglang.srt.layers.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            token_to_kv_pool = get_token_to_kv_pool()
            swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )
            swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]
            swa_page_size = token_to_kv_pool.swa_kv_pool.page_size

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
            )

            if use_cp:
                # DSA CP: keep bf16 kv around for the cross-rank all-gather, then
                # write to the FlashMLA cache after gather.
                kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
                kv = cp_all_gather_rerange_output(
                    kv.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
        else:
            q_lora = self.q_norm(q_lora)
            sparse_q_requested = bool(
                mla_off_context is not None
                and getattr(mla_off_context, "is_restore", False)
                and int(getattr(mla_off_context, "reused_row_count", 0)) > 0
                and not diagnostic_shared_only
            )
            commit_reuse_layer = getattr(
                attn_backend, "commit_mla_off_reuse_layer", None
            )
            if diagnostic_shared_only and shared_restore_requested:
                if not callable(commit_reuse_layer):
                    raise RuntimeError(
                        "shared-only backend has no cache receipt path"
                    )
                with _redknot_v4_timed(
                    "shared_restore_commit", layer_id=self.layer_id
                ):
                    shared_only_committed = bool(
                        commit_reuse_layer(
                            mla_off_context=mla_off_context,
                            projection=None,
                            local_success=True,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer=self.attn_mqa,
                            compress_ratio=self.compress_ratio,
                            freqs_cis=self.freqs_cis,
                            compressor=self.compressor,
                            indexer=self.indexer,
                            wo_a_weight=self.wo_a.weight,
                            owned_heads=self.n_local_heads,
                            groups=self.n_local_groups,
                            head_dim=self.head_dim,
                            o_lora_rank=self.o_lora_rank,
                            device=x.device,
                        )
                    )
                if not shared_only_committed:
                    raise RuntimeError(
                        "shared-only forward rejected its cache receipt"
                    )
            if sparse_q_requested:
                sparse_q = None
                sparse_q_error = None
                try:
                    with _redknot_v4_timed(
                        "q_sparse_projection", layer_id=self.layer_id
                    ):
                        sparse_q = self._compute_q_b_sparse_restore(
                            q_lora, positions, mla_off_context
                        )
                except BaseException as error:
                    sparse_q_error = error
                # v3 binds packed Q, every shared cache/state target, the
                # persistent z_off views and the ragged batch in one TP vote.
                # It also restores clean cache rows only after that vote.  The
                # legacy Q-only protocol remains available solely for an old
                # z_off artifact that has no shared-latent state.
                if diagnostic_zoff_only:
                    commit_zoff_only = getattr(
                        attn_backend, "commit_mla_off_zoff_only_layer", None
                    )
                    if not callable(commit_zoff_only):
                        raise RuntimeError(
                            "zoff_only diagnostic has no certified commit path"
                        )
                    sparse_q_committed = bool(
                        commit_zoff_only(
                            mla_off_context=mla_off_context,
                            projection=sparse_q,
                            local_success=sparse_q_error is None,
                            forward_batch=forward_batch,
                            wo_a_weight=self.wo_a.weight,
                            owned_heads=self.n_local_heads,
                            groups=self.n_local_groups,
                            head_dim=self.head_dim,
                            o_lora_rank=self.o_lora_rank,
                            device=x.device,
                        )
                    )
                elif shared_restore_requested:
                    if not callable(commit_reuse_layer):
                        raise RuntimeError(
                            "shared-latent backend has no composite commit path"
                        )
                    with _redknot_v4_timed(
                        "shared_restore_commit", layer_id=self.layer_id
                    ):
                        sparse_q_committed = bool(
                            commit_reuse_layer(
                                mla_off_context=mla_off_context,
                                projection=sparse_q,
                                local_success=sparse_q_error is None,
                                positions=positions,
                                forward_batch=forward_batch,
                                layer=self.attn_mqa,
                                compress_ratio=self.compress_ratio,
                                freqs_cis=self.freqs_cis,
                                compressor=self.compressor,
                                indexer=self.indexer,
                                wo_a_weight=self.wo_a.weight,
                                owned_heads=self.n_local_heads,
                                groups=self.n_local_groups,
                                head_dim=self.head_dim,
                                o_lora_rank=self.o_lora_rank,
                                device=x.device,
                            )
                        )
                else:
                    commit_sparse_q = getattr(
                        attn_backend, "commit_mla_off_sparse_q", None
                    )
                    with _redknot_v4_timed(
                        "sparse_q_commit", layer_id=self.layer_id
                    ):
                        sparse_q_committed = bool(
                            callable(commit_sparse_q)
                            and commit_sparse_q(
                                mla_off_context=mla_off_context,
                                projection=sparse_q,
                                local_success=sparse_q_error is None,
                                device=x.device,
                            )
                        )
                if sparse_q_committed:
                    try:
                        if shared_restore_requested:
                            # The composite preflight already covered attention
                            # metadata and every cache builder before its only TP
                            # collective; a second readiness vote here would put
                            # the old per-layer synchronization tax back.
                            backend_ready = bool(
                                getattr(
                                    mla_off_context,
                                    "sparse_q_backend_preflight_complete",
                                    False,
                                )
                            )
                        else:
                            preflight_sparse_q = getattr(
                                attn_backend,
                                "preflight_mla_off_sparse_q_backend",
                                None,
                            )
                            backend_ready = bool(
                                callable(preflight_sparse_q)
                                and preflight_sparse_q(
                                    mla_off_context=mla_off_context,
                                    q=sparse_q,
                                    layer=self.attn_mqa,
                                    forward_batch=forward_batch,
                                    device=x.device,
                                )
                            )
                        if not backend_ready:
                            raise RuntimeError(
                                "sparse-Q backend preflight failed after commit"
                            )
                        q = sparse_q
                    except BaseException as postcommit_q_error:
                        if not (
                            shared_restore_requested or diagnostic_zoff_only
                        ):
                            raise
                        # Omitted slots are already committed.  Preserve a
                        # shape/identity carrier and defer this failure to the
                        # single post-pipeline TP rendezvous; no rank may leave
                        # while peers enter attention or wo_b.
                        q = sparse_q
                        mla_off_context.record_pipeline_error(
                            postcommit_q_error
                        )
                else:
                    # Every TP rank reaches this branch after the collective
                    # rejected the partial proposal. Recompute a complete
                    # rank-local Q before the ordinary padded/native fallback.
                    with _redknot_v4_timed(
                        "q_dense_fallback", layer_id=self.layer_id
                    ):
                        q = self._compute_q_b(q_lora, positions, q_out)
                    if sparse_q_error is not None:
                        logger.warning(
                            "RedKnot sparse-Q fell back at layer %d: %s",
                            self.layer_id,
                            sparse_q_error,
                        )
            else:
                with _redknot_v4_timed(
                    "q_dense_projection", layer_id=self.layer_id
                ):
                    if diagnostic_shared_only and shared_restore_requested:
                        try:
                            prior_pipeline_error = getattr(
                                mla_off_context,
                                "composite_pipeline_error",
                                None,
                            )
                            if prior_pipeline_error is not None:
                                raise prior_pipeline_error
                            q = self._compute_q_b(q_lora, positions, q_out)
                        except BaseException as full_q_error:
                            # Cache omissions are already certified.  Carry
                            # only the ordinary rank-local Q shape; attention
                            # observes the sticky context error before reading
                            # any value and every TP rank reaches finalization.
                            make_carrier = getattr(
                                attn_backend,
                                "make_mla_off_failure_carrier",
                                None,
                            )
                            if not callable(make_carrier):
                                raise
                            q = make_carrier(
                                mla_off_context=mla_off_context,
                                reference=x,
                                shape=(
                                    int(x.shape[0]),
                                    int(self.n_local_heads),
                                    int(self.head_dim),
                                ),
                                stage="shared_only_full_q",
                                error=full_q_error,
                            )
                            mla_off_context.record_pipeline_error(full_q_error)
                    else:
                        q = self._compute_q_b(q_lora, positions, q_out)
            shared_restore_committed = bool(
                shared_restore_requested
                and bool(
                    getattr(
                        mla_off_context, "composite_irreversible", False
                    )
                    or getattr(
                        mla_off_context, "composite_certificate", None
                    )
                    is not None
                )
            )
            zoff_restore_committed = bool(
                diagnostic_zoff_only
                and mla_off_context is not None
                and getattr(
                    mla_off_context, "diagnostic_irreversible", False
                )
                and getattr(mla_off_context, "sparse_q_committed", False)
            )
            postcommit_prepare_failed = bool(
                (shared_restore_committed or zoff_restore_committed)
                and getattr(mla_off_context, "composite_pipeline_error", None)
                is not None
            )
            if postcommit_prepare_failed:
                # Clean rows are already restored and dirty rows are missing;
                # do not mutate any cache after a local committed-path error.
                # Attention is skipped below and the fixed final vote aborts
                # all TP ranks together.
                kv = None
            elif use_cp:
                # NSA CP: keep bf16 kv around for the cross-rank all-gather, then
                # write to the FlashMLA cache after gather.
                try:
                    kv = self._compute_kv_bf16(
                        x_linear, positions, qkv_a=qkv_a
                    )
                    kv = cp_all_gather_rerange_output(
                        kv.contiguous(),
                        self.cp_size,
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
                    attn_backend.store_cache(
                        layer_id=self.layer_id,
                        swa_k=kv,
                        forward_batch=forward_batch,
                    )
                except BaseException as kv_store_error:
                    if not zoff_restore_committed:
                        raise
                    kv = None
                    mla_off_context.record_pipeline_error(kv_store_error)
            else:
                try:
                    canonical_snapshot_kv = self._compute_kv_to_cache(
                        x_linear,
                        positions,
                        forward_batch,
                        qkv_a=qkv_a,
                        row_indices=(
                            mla_off_context.online_local_row_indices
                            if shared_restore_requested
                            and bool(
                                getattr(
                                    mla_off_context,
                                    "composite_irreversible",
                                    False,
                                )
                                or getattr(
                                    mla_off_context,
                                    "composite_certificate",
                                    None,
                                )
                                is not None
                            )
                            else None
                        ),
                        capture_positionless=shared_snapshot_requested,
                    )
                except BaseException as dirty_kv_error:
                    if not (
                        shared_restore_committed or zoff_restore_committed
                    ):
                        raise
                    canonical_snapshot_kv = None
                    mla_off_context.record_pipeline_error(dirty_kv_error)
                if shared_snapshot_requested:
                    if canonical_snapshot_kv is None:
                        raise RuntimeError(
                            "shared snapshot produced no canonical SWA rows"
                        )
                    mla_off_context.shared_snapshot_canonical_swa = (
                        canonical_snapshot_kv
                    )
                kv = None

        if shared_restore_committed:
            dirty_builders = getattr(
                attn_backend, "forward_mla_off_dirty_cache_builders", None
            )
            try:
                if getattr(
                    mla_off_context, "composite_pipeline_error", None
                ) is None:
                    if not callable(dirty_builders):
                        raise RuntimeError(
                            "shared-latent backend has no dirty-only cache builders"
                        )
                    with _redknot_v4_timed(
                        "dirty_kv_indexer_compressor",
                        layer_id=self.layer_id,
                    ):
                        dirty_builders(
                            x=x,
                            q_lora=q_lora,
                            forward_batch=forward_batch,
                            mla_off_context=mla_off_context,
                            layer_id=self.layer_id,
                            compressor=self.compressor,
                            indexer=self.indexer,
                        )
            except BaseException as dirty_builder_error:
                # The composite certificate has already authorized omission.
                # Carry every local failure, including validation before the
                # backend's inner kernel try-block, to the one post-merge TP
                # rendezvous instead of abandoning peers mid-pipeline.
                if getattr(
                    mla_off_context, "composite_pipeline_error", None
                ) is None:
                    mla_off_context.record_pipeline_error(dirty_builder_error)
        else:
            if zoff_restore_committed:
                # These producers have no attention-TP collective.  After the
                # sparse-Q commit, run them in order until the first local
                # BaseException, then skip all subsequent cache/state writes
                # and carry that exact error to the fixed final consumer vote.
                if (
                    getattr(
                        mla_off_context, "composite_pipeline_error", None
                    )
                    is None
                    and self.indexer is not None
                ):
                    try:
                        self.indexer(
                            x=x,
                            q_lora=q_lora,
                            forward_batch=forward_batch,
                            attn_backend=attn_backend,
                        )
                    except BaseException as indexer_error:
                        mla_off_context.record_pipeline_error(indexer_error)
                if (
                    getattr(
                        mla_off_context, "composite_pipeline_error", None
                    )
                    is None
                    and self.compressor is not None
                ):
                    try:
                        attn_backend.forward_core_compressor(
                            x,
                            forward_batch,
                            self.layer_id,
                            self.compressor,
                        )
                    except BaseException as compressor_error:
                        mla_off_context.record_pipeline_error(
                            compressor_error
                        )
            else:
                if self.indexer is not None:
                    self.indexer(
                        x=x,
                        q_lora=q_lora,
                        forward_batch=forward_batch,
                        attn_backend=attn_backend,
                    )
                if self.compressor is not None:
                    attn_backend.forward_core_compressor(
                        x,
                        forward_batch,
                        self.layer_id,
                        self.compressor,
                    )

        # Keep the fused qkv_a projection alive through the profiling hook.  In
        # the normal NVIDIA path ``fuse_wqa_wkv`` is true and the cache-writing
        # kernel does not return a bf16 latent KV tensor; dropping qkv_a here
        # made the profiler fall back to ``self.wkv``, which does not exist for
        # the fused module layout.
        if not bool(
            mla_off_context is not None
            and getattr(mla_off_context, "sparse_q_committed", False)
        ):
            self._maybe_profile_mla_heads(
                x, positions, forward_batch, q, kv, qkv_a=qkv_a
            )
        del qkv_a

        return q, kv

    def _maybe_profile_mla_heads(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        kv: Optional[torch.Tensor],
        qkv_a: Optional[torch.Tensor] = None,
    ) -> None:
        """RedKnot offline analysis hook (no-op unless the collector is on).

        Captures per-head queries and the shared latent key stream *before* MLA
        compression so the locality profiler can classify heads (global vs
        local) and size local windows. Only runs for single-sequence prefill to
        keep the recompute O(T) and the relative-distance stats well defined.
        """
        from sglang.srt.layers.attention.redknot.mla_head_profiler import (
            get_global_collector,
        )

        collector = get_global_collector()
        if collector is None:
            return
        if not forward_batch.forward_mode.is_extend():
            return
        # Restrict to a single sequence so query->key distances are unambiguous.
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and len(seq_lens_cpu) != 1:
            return

        def _single_cpu_int(name: str) -> Optional[int]:
            values = getattr(forward_batch, name, None)
            if values is None or len(values) != 1:
                return None
            if torch.is_tensor(values):
                # These fields are explicitly the scheduler's CPU mirrors.
                # Refuse a mislabeled CUDA tensor rather than introducing an
                # in-forward device synchronization via .item()/.cpu().
                if values.device.type != "cpu":
                    return None
                return int(values.tolist()[0])
            return int(values[0])

        actual_extend = _single_cpu_int("extend_seq_lens_cpu")
        sequence_end = _single_cpu_int("seq_lens_cpu")
        prefix_len = _single_cpu_int("extend_prefix_lens_cpu")
        if actual_extend is None or sequence_end is None:
            return
        if prefix_len is None:
            prefix_len = sequence_end - actual_extend
        if prefix_len < 0 or prefix_len + actual_extend != sequence_end:
            return
        # At TP>1 each rank holds only its local head shard. The full head-class
        # JSON requires all heads on one rank, so it is only exported at TP=1.
        # The token-mass concentration curve is a per-head average and works at
        # any TP: each rank collects its local heads and exports a rank-tagged
        # JSON to be averaged offline.
        tp_single = self.tp_size <= 1

        try:
            # Prefer the fused projection from this exact forward.  Slicing its
            # KV portion and applying the same norm+RoPE transformation gives
            # the real shared MLA latent without rerunning a linear layer.  A
            # passed ``kv`` is already transformed only on the non-fused bf16
            # path; in the fused-qk-normalization path it may still be raw.
            if qkv_a is not None:
                latent_kv = self._compute_kv_bf16(
                    x, positions, qkv_a=qkv_a
                )
            elif kv is not None and not self.use_fused_qk_norm_rope:
                latent_kv = kv
            else:
                latent_kv = self._compute_kv_bf16(x, positions)
            q_heads = q.view(q.shape[0], self.n_local_heads, self.head_dim)
            profile_complete = collector.observe_layer(
                layer_id=self.layer_id,
                q=q_heads.detach(),
                latent_k=latent_kv.detach(),
                softmax_scale=self.softmax_scale,
                prefix_len=prefix_len,
                extend_len=actual_extend,
                seq_len=sequence_end,
            )
            # Export only when every layer has captured exactly the configured
            # request length.  Short warmups and partial chunks never produce
            # files that the TP merge could mistake for a valid profile.
            out_path = getattr(
                get_global_server_args(), "redknot_mla_profile_out", None
            )
            if (
                profile_complete
                and out_path
                and self.layer_id == self._mla_profile_last_layer
            ):
                rank = get_attention_tp_rank()
                stem, ext = os.path.splitext(out_path)
                ext = ext or ".json"
                # At TP>1 every rank owns a contiguous shard of logical query
                # heads.  Export each shard separately; the benchmark parent
                # merges them after Engine shutdown.  Previously only TP=1
                # exported a head config, which made profiling this 8-way model
                # impossible in practice.
                head_path = out_path if tp_single else f"{stem}_rank{rank}{ext}"
                collector.export_json(head_path)
                distance_path = (
                    f"{stem}_distance{ext}"
                    if tp_single
                    else f"{stem}_distance_rank{rank}{ext}"
                )
                collector.export_distance_json(distance_path)
                # Always export the per-layer token-mass concentration curve
                # (the real "tokens for N% attn" signal). Tag with TP rank so
                # multi-rank runs can be averaged offline.
                try:
                    conc_path = out_path
                    if conc_path.endswith(".json"):
                        conc_path = conc_path[: -len(".json")]
                    suffix = (
                        "_concentration.json"
                        if tp_single
                        else (f"_concentration_rank{rank}.json")
                    )
                    collector.export_concentration_json(conc_path + suffix)
                except Exception as conc_exc:
                    logger.warning(
                        "RedKnot MLA concentration export skipped: %s", conc_exc
                    )
        except Exception as exc:  # serving remains fail-open by default
            logger.warning(
                "RedKnot MLA head profiling skipped at layer %d: %s",
                self.layer_id,
                exc,
            )
            # Offline benchmark runs opt into strict mode so a broken probe
            # fails on the first layer instead of spending a full long-context
            # forward and only discovering that no JSON shards were exported.
            if os.environ.get("REDKNOT_MLA_PROFILE_STRICT", "0") == "1":
                raise

    @staticmethod
    def _clear_mla_off_ephemeral_snapshot_state(mla_off_context) -> None:
        """Drop layer-local snapshot tensors as soon as their hook is done.

        ``shared_snapshot_canonical_swa`` aliases a potentially large device
        tensor produced by this layer's real KV projection.  It is a hand-off
        value for ``capture_mla_off_shared_snapshot_chunk`` only; retaining it
        on the context until a later layer (or after a cancelled forward) pins
        an otherwise dead activation and can also expose the wrong layer's
        rows to a retried scheduler forward.
        """

        if mla_off_context is None:
            return
        try:
            delattr(mla_off_context, "shared_snapshot_canonical_swa")
        except AttributeError:
            pass

    @staticmethod
    def _close_mla_off_composite_resources_on_error(
        forward_batch, *, error: BaseException, layer_id: int = -1
    ) -> None:
        """Fail closed only when this ForwardBatch owns a composite lease."""

        if (
            getattr(
                forward_batch, "_redknot_composite_forward_resources", None
            )
            is None
        ):
            # Dense/native forwards never import or invoke the RedKnot cleanup
            # runtime.  This keeps their exception path byte-for-byte neutral
            # apart from this inexpensive attribute read.
            return
        try:
            backend = get_attn_backend()
            abort_forward = getattr(
                backend, "abort_mla_off_forward_transaction", None
            )
            if callable(abort_forward) and abort_forward(
                forward_batch=forward_batch,
                error=error,
                layer_id=int(layer_id),
            ):
                return
            from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
                close_composite_forward_resources,
            )

            close_composite_forward_resources(forward_batch)
        except Exception as cleanup_error:
            # Preserve the original model/cancellation exception.  The close
            # primitive itself attempts every pin before reporting a failure;
            # logging here records that the fail-closed cleanup was incomplete
            # without replacing the error that caused the unwind.
            logger.exception(
                "RedKnot composite forward cleanup failed at layer teardown: %s",
                cleanup_error,
            )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        x_quant=None,
    ) -> torch.Tensor:
        """Run one layer with request-scoped fail-closed resource teardown."""

        lifecycle = {"mla_off_context": None}
        try:
            output = self._forward_impl(
                x,
                positions,
                forward_batch,
                x_quant=x_quant,
                _redknot_lifecycle=lifecycle,
            )
        except BaseException as layer_error:
            # This covers normal execution errors as well as scheduler
            # cancellation (which need not inherit from Exception).  A
            # committed composite omission may own persistent GPU-bank pins
            # spanning layers 3..39, so a per-context release is insufficient
            # once the layer aborts before its normal completion hook.
            failed_context = lifecycle.get("mla_off_context")
            self._clear_mla_off_ephemeral_snapshot_state(failed_context)
            if bool(
                failed_context is not None
                and getattr(failed_context, "is_snapshot", False)
            ):
                try:
                    failed_backend = get_attn_backend()
                    abort_snapshot = getattr(
                        failed_backend,
                        "abort_mla_off_snapshot_context",
                        None,
                    )
                    if callable(abort_snapshot):
                        # Unified shared snapshots own z_off, CPU latent, GPU
                        # bank staging, and the backend stage registry as one
                        # transaction.  Context.abort_snapshot() alone only
                        # covers the legacy z_off controller.
                        abort_snapshot(failed_context)
                    else:
                        failed_context.abort_snapshot()
                except Exception as snapshot_cleanup_error:
                    # Preserve the model/cancellation exception.  A duplicate
                    # rollback after a concurrent finalization may itself be
                    # stale, while the original failure remains actionable.
                    logger.exception(
                        "RedKnot snapshot rollback failed during layer teardown: %s",
                        snapshot_cleanup_error,
                    )
            self._close_mla_off_composite_resources_on_error(
                forward_batch,
                error=layer_error,
                layer_id=int(self.layer_id),
            )
            raise
        self._clear_mla_off_ephemeral_snapshot_state(
            lifecycle.get("mla_off_context")
        )
        return output

    def _forward_impl(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        x_quant=None,
        *,
        _redknot_lifecycle=None,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert not self.wo_b.reduce_results, (
                "short-circuiting allreduce will lead to hangs"
            )
            return x

        attn_backend = get_attn_backend()
        if TYPE_CHECKING:
            assert isinstance(
                attn_backend,
                (DeepseekV4AttnBackend, DeepseekV4HipRadixBackend),
            )

        # MLA restore geometry must be certified before wq_b runs; otherwise
        # the model has already paid for a complete 65Kx64 Q projection before
        # learning which local head/rows attention will omit.
        mla_off_context = None
        prepare_mla_off = getattr(attn_backend, "prepare_mla_off_context", None)
        if callable(prepare_mla_off):
            with _redknot_v4_timed(
                "prepare_context", layer_id=self.layer_id
            ):
                mla_off_context = prepare_mla_off(
                    layer_id=self.layer_id,
                    positions=positions,
                    forward_batch=forward_batch,
                    q_head_count=self.n_local_heads,
                    q_row_count=int(x.shape[0]),
                    n_local_heads=self.n_local_heads,
                    n_local_groups=self.n_local_groups,
                    head_dim=self.head_dim,
                    o_lora_rank=self.o_lora_rank,
                    fp8_wo_a=_FP8_WO_A_GEMM,
                    device=x.device,
                    projection_dtype=x.dtype,
                )
        if _redknot_lifecycle is not None:
            # Keep the original context even if a collectively rejected
            # proposal later switches the local execution variable to None.
            # The public wrapper needs this reference solely to clear ephemeral
            # snapshot tensors while the ForwardBatch owns resource lifetime.
            _redknot_lifecycle["mla_off_context"] = mla_off_context
        diagnostic_ablation = str(
            getattr(mla_off_context, "diagnostic_ablation", "full") or "full"
        )
        diagnostic_shared_only = diagnostic_ablation == "shared_only"
        diagnostic_shared_full_local = bool(
            diagnostic_shared_only
            and mla_off_context is not None
            and getattr(mla_off_context, "is_full_local", False)
        )
        sparse_q_candidate = bool(
            mla_off_context is not None
            and getattr(mla_off_context, "is_restore", False)
            and int(getattr(mla_off_context, "reused_row_count", 0)) > 0
            and not diagnostic_shared_only
        )
        accepts_rank_local_q = getattr(attn_backend, "accepts_rank_local_q", None)
        rank_local_q_enabled = bool(
            not diagnostic_shared_full_local
            and callable(accepts_rank_local_q)
            and accepts_rank_local_q(
                layer_id=self.layer_id, forward_batch=forward_batch
            )
        )

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch))
            and mla_off_context is None
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if (
            self.tp_size > 1
            and not sparse_q_candidate
            and not rank_local_q_enabled
        ):
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            # Multi-stream path always fuses cache write into the K kernel,
            # so the bf16 KV intermediate is gone.
            if _is_hip:
                q = self._forward_prepare_multi_stream_hip(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            else:
                q = self._forward_prepare_multi_stream(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            kv = None
        else:
            with _redknot_v4_timed(
                "q_kv_prepare_total", layer_id=self.layer_id
            ):
                q, kv = self._forward_prepare(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                    mla_off_context=mla_off_context,
                )

        if bool(
            mla_off_context is not None
            and getattr(mla_off_context, "composite_dense_fallback", False)
        ):
            # The one-shot TP vote rejected the proposal before any omission.
            # `_forward_prepare` has already recomputed full rank-local Q and
            # executed full-row KV/Indexer/Compressor builders.  Do not pass a
            # stale restore context into headwise attention: doing so could
            # omit clean local-head work despite the collective rejection.
            mla_off_context = None
            sparse_q_candidate = False

        if self.tp_size > 1 and sparse_q_candidate:
            if bool(getattr(mla_off_context, "sparse_q_committed", False)):
                # Backend headwise mode consumes the packed global/full and
                # local/dirty runs directly; there is no dense Q presentation.
                tp_slice = slice(None)
            elif not rank_local_q_enabled:
                # Sparse proposal was collectively rejected before attention.
                # A backend without rank-local support needs native padding;
                # the production middle-layer RedKnot backend never enters
                # this compatibility branch.
                q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
                rank = self.tp_rank
                tp_slice = slice(
                    rank * self.n_local_heads,
                    (rank + 1) * self.n_local_heads,
                )
                q_padded[:, tp_slice, :].copy_(q)
            else:
                # Safe pre-commit fallback is complete rank-local Q consumed
                # by the arbitrary-head path, still without [T,64,D].
                tp_slice = slice(None)

        # The cache write is always fused / already done by _forward_prepare* --
        # tell the backend to skip its own store_cache. When `kv is None`
        # (no DSA-CP), pass `q` as a sentinel for the `k is v` assert; the
        # attention path doesn't read it once `save_kv_cache=False`.
        attn_k = kv if kv is not None else q
        irreversible_postcommit = bool(
            mla_off_context is not None
            and bool(
                getattr(mla_off_context, "composite_irreversible", False)
                or getattr(mla_off_context, "composite_certificate", None)
                is not None
                or getattr(
                    mla_off_context, "diagnostic_irreversible", False
                )
            )
        )
        carried_postcommit_error = (
            getattr(mla_off_context, "composite_pipeline_error", None)
            if irreversible_postcommit
            else None
        )
        if carried_postcommit_error is None:
            try:
                with _redknot_v4_timed(
                    "attention", layer_id=self.layer_id
                ):
                    o = attn_backend.forward(
                        q=q_padded if q_padded is not None else q,
                        k=attn_k,
                        v=attn_k,
                        layer=self.attn_mqa,
                        forward_batch=forward_batch,
                        compress_ratio=self.compress_ratio,
                        attn_sink=self.attn_sink,
                        save_kv_cache=False,
                        mla_off_context=mla_off_context,
                    )
            except BaseException as attention_error:
                if not irreversible_postcommit:
                    raise
                carried_postcommit_error = (
                    mla_off_context.record_pipeline_error(attention_error)
                )
        if carried_postcommit_error is not None:
            # A committed rank may not escape before peers reach the one final
            # post-pipeline vote.  This tensor is shape-only: projection/merge
            # sees the carried error and skips all reads before the vote rejects
            # the layer on every rank.
            carrier_heads = (
                self.n_heads if q_padded is not None else self.n_local_heads
            )
            make_carrier = getattr(
                attn_backend, "make_mla_off_failure_carrier", None
            )
            if not callable(make_carrier):
                raise carried_postcommit_error
            o = make_carrier(
                mla_off_context=mla_off_context,
                reference=x,
                shape=(
                    int(x.shape[0]),
                    int(carrier_heads),
                    int(self.attn_mqa.v_head_dim),
                ),
                stage="attention_output",
                error=carried_postcommit_error,
            )
            mla_off_context.backend_applied = True
        o = o[:, tp_slice, :]
        drift_profile_active = bool(
            getattr(forward_batch, "_redknot_mla_head_drift_active", False)
        )
        pure_mla_off_restore = bool(
            not _FP8_WO_A_GEMM
            and not drift_profile_active
            and mla_off_context is not None
            and getattr(mla_off_context, "is_restore", False)
            and getattr(mla_off_context, "backend_applied", False)
            and not diagnostic_shared_only
            and getattr(
                getattr(mla_off_context, "spec", None),
                "execution_profile",
                "",
            )
            in (
                "pure_headsplit_context_bound_fullscope_3_37_3_v1",
                "pure_headsplit_independent_rope_relocation_fullscope_"
                "boundary128_3_37_3_v1",
                "combined_headsplit_independent_rope_zoff_checkpoint_"
                "rowsparse_3_37_3_v1",
            )
        )
        indexed_mla_off_restore = bool(
            not pure_mla_off_restore
            and not diagnostic_shared_only
            and not _FP8_WO_A_GEMM
            and not drift_profile_active
            and mla_off_context is not None
            and bool(
                getattr(mla_off_context, "can_merge_online_indexed", False)
            )
        )
        coordinated_mla_off_restore = bool(
            mla_off_context is not None
            and getattr(mla_off_context, "is_restore", False)
            and getattr(mla_off_context, "backend_applied", False)
        )
        forward_composite_deferred = bool(
            mla_off_context is not None
            and getattr(
                getattr(
                    mla_off_context,
                    "_redknot_forward_composite_transaction",
                    None,
                ),
                "coordinator",
                None,
            )
            is not None
        )

        def forward_projection_carrier(
            stage: str, local_error: BaseException
        ) -> torch.Tensor:
            make_carrier = getattr(
                attn_backend, "make_mla_off_failure_carrier", None
            )
            if not callable(make_carrier):
                raise local_error
            return make_carrier(
                mla_off_context=mla_off_context,
                reference=x,
                shape=(
                    int(x.shape[0]),
                    int(self.n_local_groups),
                    int(self.o_lora_rank),
                ),
                stage=stage,
                error=local_error,
            )

        def coordinate_mla_off_consumer(
            stage: str, local_error: Optional[BaseException]
        ) -> bool:
            """Make every attention-TP rank cross a restore stage together."""

            resolve_stage = getattr(
                attn_backend, "resolve_mla_off_consumer_stage", None
            )
            if not callable(resolve_stage):
                raise RuntimeError(
                    "MLA-off restore backend has no TP consumer coordinator"
                ) from local_error
            stage_ok, stage_reason = resolve_stage(
                stage=stage,
                local_success=local_error is None,
                device=o.device,
                mla_off_context=mla_off_context,
            )
            if not stage_ok:
                raise RuntimeError(
                    "MLA-off post-attention consumer aborted before wo_b: "
                    f"stage={stage} reason={stage_reason}"
                ) from local_error
            return bool(forward_composite_deferred and local_error is not None)

        def inverse_rope_and_project_wo_a(
            per_head_output: torch.Tensor,
            rope_positions: torch.Tensor,
            *,
            compact_empty_shape: bool,
            source_head_axes: Optional[Tuple[int, ...]] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Run post-attention projection, optionally slicing logical heads.

            ``source_head_axes`` are TP-rank-local axes in the original
            ``wo_a`` input.  Supplying them computes only the corresponding
            weight columns; it does not zero-fill a full head tensor and call a
            full GEMM.  This is the online half of pure MLA head splitting.
            """

            if int(per_head_output.shape[0]) > 0:
                fused_rope_inplace(
                    per_head_output[..., -self.qk_rope_head_dim :],
                    None,
                    self.freqs_cis,
                    positions=rope_positions,
                    inverse=True,
                )

            # Accuracy calibration hook. ModelRunner sets this request-local
            # flag only after fail-closed drift-profile validation.
            if drift_profile_active:
                if _FP8_WO_A_GEMM:
                    raise RuntimeError(
                        "MLA head-drift profiling does not support FP8 wo_a"
                    )
                from sglang.srt.layers.attention.redknot.mla_head_drift_runtime import (
                    capture_active_drift_layer,
                )

                capture_active_drift_layer(
                    layer_id=self.layer_id,
                    per_head_output=per_head_output,
                    wo_a_weight=self.wo_a.weight.view(
                        self.n_local_groups, self.o_lora_rank, -1
                    ),
                    num_local_groups=self.n_local_groups,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )

            if source_head_axes is not None:
                if _FP8_WO_A_GEMM:
                    raise RuntimeError(
                        "pure MLA head-sliced wo_a requires the BF16 projection path"
                    )
                source_head_axes = tuple(int(axis) for axis in source_head_axes)
                if (
                    len(source_head_axes) != int(per_head_output.shape[1])
                    or len(set(source_head_axes)) != len(source_head_axes)
                    or any(
                        axis < 0 or axis >= self.n_local_heads
                        for axis in source_head_axes
                    )
                ):
                    raise ValueError("MLA head-sliced wo_a axes are inconsistent")
                wo_a_weight = self.wo_a.weight.view(
                    self.n_local_groups, self.o_lora_rank, -1
                )
                heads_per_group = self.n_local_heads // self.n_local_groups
                projected = per_head_output.new_zeros(
                    (
                        per_head_output.shape[0],
                        self.n_local_groups,
                        self.o_lora_rank,
                    )
                )
                for group_id in range(self.n_local_groups):
                    group_start = group_id * heads_per_group
                    group_end = group_start + heads_per_group
                    selected_positions = tuple(
                        position
                        for position, axis in enumerate(source_head_axes)
                        if group_start <= axis < group_end
                    )
                    if not selected_positions:
                        continue
                    selected_axes = tuple(
                        source_head_axes[position] - group_start
                        for position in selected_positions
                    )
                    output_index = torch.tensor(
                        selected_positions,
                        dtype=torch.long,
                        device=per_head_output.device,
                    )
                    weight_index = torch.tensor(
                        selected_axes,
                        dtype=torch.long,
                        device=wo_a_weight.device,
                    )
                    selected_output = per_head_output.index_select(
                        1, output_index
                    ).reshape(per_head_output.shape[0], -1)
                    selected_weight = (
                        wo_a_weight[group_id]
                        .view(self.o_lora_rank, heads_per_group, self.head_dim)
                        .index_select(1, weight_index)
                        .reshape(self.o_lora_rank, -1)
                    )
                    projected[:, group_id, :] = torch.einsum(
                        "td,rd->tr", selected_output, selected_weight
                    )
                return projected, wo_a_weight

            if compact_empty_shape and int(per_head_output.shape[0]) == 0:
                grouped_head_dim = (
                    int(per_head_output.shape[1])
                    * int(per_head_output.shape[2])
                ) // int(self.n_local_groups)
                grouped = per_head_output.reshape(
                    0, self.n_local_groups, grouped_head_dim
                )
            else:
                grouped = per_head_output.view(
                    per_head_output.shape[0], self.n_local_groups, -1
                )

            if _FP8_WO_A_GEMM:
                import deep_gemm

                T, G, D = grouped.shape
                R = self.o_lora_rank
                o_fp8, o_s = sglang_per_token_group_quant_fp8(
                    grouped.reshape(T * G, D).contiguous(),
                    group_size=128,
                )
                o_s = deep_gemm.ceil_to_ue8m0(o_s)
                output = torch.empty(
                    T, G, R, device=grouped.device, dtype=torch.bfloat16
                )
                deep_gemm.fp8_einsum(
                    "bhr,hdr->bhd",
                    (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                    (
                        self.wo_a.weight.view(G, R, D),
                        self.wo_a.weight_scale_inv.data,
                    ),
                    output,
                    recipe=(1, 1, 128),
                )
                return output, None

            wo_a_weight = self.wo_a.weight.view(
                self.n_local_groups, self.o_lora_rank, -1
            )
            return (
                torch.einsum("tgd,grd->tgr", grouped, wo_a_weight),
                wo_a_weight,
            )

        if pure_mla_off_restore:
            # Document rows contain online global-head attention outputs and
            # zero local slots. Query/new rows additionally contain all local
            # heads. One full-tensor inverse RoPE preserves those zero slots;
            # the committed two-kernel plan then reads head/row coordinates
            # directly and folds persistent z_off into the global stores.
            # Dirty KV/Indexer/Compressor and headwise attention deliberately
            # run without intermediate TP barriers in the composite path.  A
            # rank that failed either stage carries the first error here, skips
            # projection/merge, and still joins the one fixed rendezvous below
            # before any rank enters the tensor-parallel wo_b GEMM.
            pure_pipeline_error = getattr(
                mla_off_context, "composite_pipeline_error", None
            )
            merged_projection = None
            try:
                with _redknot_v4_timed(
                    "persistent_woa_merge", layer_id=self.layer_id
                ):
                    if pure_pipeline_error is not None:
                        raise pure_pipeline_error
                    if int(o.shape[0]) > 0:
                        fused_rope_inplace(
                            o[..., -self.qk_rope_head_dim :],
                            None,
                            self.freqs_cis,
                            positions=positions,
                            inverse=True,
                        )
                    merged_projection = mla_off_context.project_merge_headsplit(
                        o,
                        wo_a_weight=self.wo_a.weight,
                    )
                    if merged_projection is None:
                        raise RuntimeError(
                            "pure MLA offline/online merge returned no projection"
                        )
            except BaseException as error:
                pure_pipeline_error = error
            deferred_failure = coordinate_mla_off_consumer(
                "pure_headsplit_projection_merge", pure_pipeline_error
            )
            if pure_pipeline_error is not None:
                if deferred_failure:
                    merged_projection = forward_projection_carrier(
                        "pure_headsplit_projection_merge",
                        pure_pipeline_error,
                    )
                else:
                    raise RuntimeError(
                        "pure MLA headsplit projection/merge failed"
                    ) from pure_pipeline_error
            o = merged_projection
        elif indexed_mla_off_restore:
            # Collective eligibility was committed by the backend before
            # attention, so every TP rank takes this exact branch. Capture the
            # entire select→inverse-RoPE→reshape→wo_a→merge pipeline: one rank
            # cannot escape while peers wait at a later wo_b collective.
            compact_pipeline_error = None
            compact_projection = None
            try:
                o_per_head, rope_positions = (
                    mla_off_context.indexed_online_rows(o, positions)
                )
                online_projection, _ = inverse_rope_and_project_wo_a(
                    o_per_head,
                    rope_positions,
                    compact_empty_shape=True,
                )
                compact_projection = mla_off_context.merge_online_indexed(
                    online_projection,
                    total_rows=int(positions.shape[0]),
                )
                if compact_projection is None:
                    raise RuntimeError(
                        "MLA-off compact pipeline returned no local projection"
                    )
            except BaseException as error:
                compact_pipeline_error = error
            deferred_compact_failure = coordinate_mla_off_consumer(
                "indexed_pipeline", compact_pipeline_error
            )
            if compact_pipeline_error is not None:
                if deferred_compact_failure:
                    compact_projection = forward_projection_carrier(
                        "indexed_pipeline",
                        compact_pipeline_error,
                    )
                else:
                    raise RuntimeError(
                        "MLA-off compact vote accepted a local pipeline failure"
                    ) from compact_pipeline_error
            o = compact_projection

            audit_publish_error = None
            prepared_audit = None
            try:
                prepared_audit = (
                    mla_off_context.prepare_indexed_merge_success_audit()
                )
            except BaseException as error:
                audit_publish_error = error
            deferred_audit_failure = coordinate_mla_off_consumer(
                "audit_publish", audit_publish_error
            )
            if audit_publish_error is not None:
                if not deferred_audit_failure:
                    raise RuntimeError(
                        "MLA-off audit vote accepted a local preparation failure"
                    ) from audit_publish_error
            # The marker becomes externally visible only after every rank has
            # prepared the same stage successfully. All validation and JSON
            # serialization happened above, so publication cannot strand a
            # peer at a later consumer collective.
            if audit_publish_error is None:
                mla_off_context.publish_indexed_merge_success_audit(
                    prepared_audit
                )
        else:
            # Mixed/global, snapshot, native fallback, drift profiling and FP8
            # retain the existing all-row projection path.
            o_per_head = o
            projection_error = None
            projected_online = None
            wo_a = None
            try:
                if diagnostic_shared_only:
                    shared_pipeline_error = getattr(
                        mla_off_context, "composite_pipeline_error", None
                    )
                    if shared_pipeline_error is not None:
                        raise shared_pipeline_error
                projected_online, wo_a = inverse_rope_and_project_wo_a(
                    o_per_head,
                    positions,
                    compact_empty_shape=False,
                )
                if projected_online is None:
                    raise RuntimeError(
                        "MLA-off projection returned no local result"
                    )
            except BaseException as error:
                projection_error = error
            if coordinated_mla_off_restore:
                deferred_projection_failure = coordinate_mla_off_consumer(
                    "projection_compute", projection_error
                )
            elif projection_error is not None:
                raise projection_error
            if projection_error is not None:
                if forward_composite_deferred and deferred_projection_failure:
                    projected_online = forward_projection_carrier(
                        "projection_compute",
                        projection_error,
                    )
                else:
                    raise RuntimeError(
                        "MLA-off projection vote accepted a local failure"
                    ) from projection_error
            o = projected_online

            if mla_off_context is not None:
                if not mla_off_context.backend_applied:
                    # A native/backend fallback returned a complete all-head
                    # output. Do not add an offline contribution twice.
                    if mla_off_context.is_snapshot:
                        forward_batch._redknot_mla_off_disabled = True
                        abort_snapshot = getattr(
                            attn_backend,
                            "abort_mla_off_snapshot_context",
                            None,
                        )
                        if callable(abort_snapshot):
                            abort_snapshot(mla_off_context)
                        else:
                            mla_off_context.abort_snapshot()
                    logger.warning(
                        "RedKnot MLA-off policy was not applied at layer %d; "
                        "keeping the full online projection",
                        self.layer_id,
                    )
                elif mla_off_context.is_snapshot:
                    if wo_a is None:
                        raise RuntimeError(
                            "MLA-off snapshot requires non-FP8 wo_a"
                        )
                    capture_succeeded = False
                    capture_error = ""
                    try:
                        local_per_head = mla_off_context.local_only(o_per_head)
                        local_grouped = local_per_head.view(
                            local_per_head.shape[0], self.n_local_groups, -1
                        )
                        local_projection = torch.einsum(
                            "tgd,grd->tgr", local_grouped, wo_a
                        )
                        capture_shared_snapshot = getattr(
                            attn_backend,
                            "capture_mla_off_shared_snapshot_chunk",
                            None,
                        )
                        if bool(
                            getattr(
                                mla_off_context,
                                "shared_snapshot_enabled",
                                False,
                            )
                        ):
                            if not callable(capture_shared_snapshot):
                                raise RuntimeError(
                                    "shared snapshot context has no capture hook"
                                )
                            # Capture immediately after this layer's native
                            # KV/Indexer/Compressor producers and z_off output.
                            # The hook canonicalizes their exact cache/state
                            # records into device staging before a later
                            # scheduler forward can reuse physical ring slots.
                            capture_shared_snapshot(
                                mla_off_context=mla_off_context,
                                forward_batch=forward_batch,
                                positions=positions,
                                local_projection=local_projection,
                                layer_id=self.layer_id,
                                compress_ratio=self.compress_ratio,
                                freqs_cis=self.freqs_cis,
                            )
                        else:
                            # Legacy z_off-only snapshots keep their original
                            # controller.  The unified shared adapter already
                            # owns that controller generation and must be the
                            # sole writer when enabled.
                            mla_off_context.capture(local_projection)
                        capture_succeeded = True
                    except BaseException as error:
                        capture_error = str(error)
                    finally:
                        # The canonical packed SWA rows are an ephemeral
                        # producer-to-capture hand-off.  Drop the device tensor
                        # before snapshot publication/TP coordination and do so
                        # even when capture failed locally.
                        self._clear_mla_off_ephemeral_snapshot_state(
                            mla_off_context
                        )
                    finalize_snapshot = getattr(
                        attn_backend,
                        "finalize_mla_off_snapshot_context",
                        None,
                    )
                    if not callable(finalize_snapshot):
                        abort_snapshot = getattr(
                            attn_backend,
                            "abort_mla_off_snapshot_context",
                            None,
                        )
                        if callable(abort_snapshot):
                            abort_snapshot(mla_off_context)
                        else:
                            mla_off_context.abort_snapshot()
                        snapshot_ready = False
                    else:
                        snapshot_ready = finalize_snapshot(
                            mla_off_context,
                            capture_succeeded=capture_succeeded,
                            device=x.device,
                        )
                    if not snapshot_ready:
                        forward_batch._redknot_mla_off_disabled = True
                        logger.warning(
                            "RedKnot MLA-off snapshot aborted at layer %d: %s",
                            self.layer_id,
                            capture_error or "attention-TP capture vote failed",
                        )
                elif mla_off_context.is_restore and not diagnostic_shared_only:
                    merged_projection = None
                    merge_error = None
                    try:
                        merged_projection = mla_off_context.merge_online(o)
                        if merged_projection is None:
                            raise RuntimeError(
                                "MLA-off merge returned no local projection"
                            )
                    except BaseException as error:
                        merge_error = error
                    deferred_merge_failure = coordinate_mla_off_consumer(
                        "projection_merge", merge_error
                    )
                    if merge_error is not None:
                        if deferred_merge_failure:
                            merged_projection = forward_projection_carrier(
                                "projection_merge",
                                merge_error,
                            )
                        else:
                            raise RuntimeError(
                                "MLA-off merge vote accepted a local failure"
                            ) from merge_error
                    o = merged_projection

        with _redknot_v4_timed("wo_b", layer_id=self.layer_id):
            o, _ = self.wo_b(o.flatten(1))

        finish_restore = getattr(
            attn_backend, "finish_mla_off_forward_resources", None
        )
        if callable(finish_restore):
            finish_restore(
                mla_off_context=mla_off_context,
                forward_batch=forward_batch,
            )

        return o


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        moe_alt_stream = (
            alt_streams[0]
            if (
                alt_streams is not None
                and (_is_cuda or envs.SGLANG_ROCM_USE_MULTI_STREAM.get())
            )
            else None
        )
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=moe_alt_stream,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()

    def prewarm_mhc_token_counts(
        self, token_counts: Tuple[int, ...], device: torch.device
    ) -> None:
        paths = (
            (
                "attn",
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm,
            ),
            (
                "ffn",
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.post_attention_layernorm,
            ),
        )

        with torch.inference_mode():
            for num_tokens in token_counts:
                for path_name, hc_fn, hc_scale, hc_base, norm in paths:
                    tic = time.perf_counter()
                    residual = torch.empty(
                        (num_tokens, self.hc_mult, self.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    y, post, comb, _ = self.hc_pre(
                        residual,
                        hc_fn,
                        hc_scale,
                        hc_base,
                        norm=norm,
                    )
                    del residual, y, post, comb
                    torch.cuda.synchronize()
                    logger.info(
                        "DeepSeek V4 MHC prewarm path=%s num_tokens=%s completed in %.3fs",
                        path_name,
                        num_tokens,
                        time.perf_counter() - tic,
                    )

    def prewarm_mhc_token_count_buckets(
        self, max_num_tokens: int, device: torch.device
    ) -> Tuple[int, ...]:
        from sglang.srt.layers.mhc import get_mhc_pre_token_count_representatives

        token_counts = get_mhc_pre_token_count_representatives(
            max_num_tokens, self.hc_mult * self.hidden_size
        )
        if not token_counts:
            return token_counts

        logger.info(
            "DeepSeek V4 MHC prewarm max_num_tokens=%s representative token counts: %s",
            max_num_tokens,
            token_counts,
        )
        self.prewarm_mhc_token_counts(token_counts, device)
        return token_counts

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
    ):
        """If *norm* is given and the TileLang path is active, the returned
        hidden_states are already post-norm (the norm is fused into the kernel)."""

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb, False

        if os.environ.get("SGLANG_USE_JIT_MHC", "0") == "1":
            from sglang.jit_kernel.dsv4.attn import triton_hc_pre

            y, post, comb = triton_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.rms_norm_eps,
                self.hc_sinkhorn_iters,
                self.hc_eps,
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            return y, post.squeeze(-1), comb, norm is not None

        if _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_PRE.get():
            from aiter.ops.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            return y, post.squeeze(-1), comb, False

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            import deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc_fallback import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if os.environ.get("SGLANG_USE_JIT_MHC", "0") == "1":
            from sglang.jit_kernel.dsv4.attn import triton_hc_post

            return triton_hc_post(x, residual, post, comb)

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        elif _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_POST.get():
            from aiter.ops.mhc import mhc_post

            result = torch.empty_like(residual)
            mhc_post(result, x, residual, post, comb)
            return result

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        return hc_post_torch_impl(x, residual, post, comb)

    def _redknot_indexer_token_importance(
        self, num_tokens: int, device: torch.device, mode: str = "out_degree"
    ) -> Optional[torch.Tensor]:
        """Per-token importance reused from DeepSeek V4's native indexer.

        Two interpretations of "indexer importance" are supported:

        * ``out_degree`` (``c4_topk_lengths_raw``): for each query token, how many
          compressed historical tokens its indexer selected. A token whose own
          query retrieves a lot of global context is doing long-range work.

        * ``in_degree`` (page popularity of ``c4_sparse_page_indices``): for each
          current token, how many queries in the batch selected the c4 page that
          token lives in. This matches the design intent "tokens frequently
          selected by others are important". In-degree needs the indexer's
          per-query top-k page indices and the current tokens' c4 write
          locations; if either is unavailable it falls back to out-degree, then
          to the activation heuristic.

        Returns ``None`` (caller falls back to activation-norm) whenever the
        signal is unavailable for this layer / forward mode.
        """

        def _dbg_bail(reason: str):
            if os.environ.get("SGLANG_REDKNOT_FFN_DEBUG") == "1" and not getattr(
                self, "_redknot_ffn_bail_logged", False
            ):
                logger.info(
                    "[RedKnot sparse-FFN] layer %d indexer signal unavailable "
                    "(%s) -> activation fallback",
                    self.layer_id,
                    reason,
                )
                self._redknot_ffn_bail_logged = True
            return None

        # ``compress_ratio`` / ``indexer`` live on the attention module
        # (``MQALayer``), not on the decoder layer. Only c4 (compress_ratio==4)
        # layers run the indexer and thus expose the indexer metadata.
        self_attn = getattr(self, "self_attn", None)
        if self_attn is None:
            return _dbg_bail("no self_attn")
        if getattr(self_attn, "compress_ratio", 0) != 4:
            return _dbg_bail(
                f"compress_ratio={getattr(self_attn, 'compress_ratio', 0)}"
            )
        if getattr(self_attn, "indexer", None) is None:
            return _dbg_bail("no indexer module")
        try:
            attn_backend = get_attn_backend()
            metadata = getattr(attn_backend, "forward_metadata", None)
            core_meta = getattr(metadata, "core_attn_metadata", None)
        except Exception as e:  # noqa: BLE001
            return _dbg_bail(f"metadata access error: {e!r}")
        if core_meta is None:
            return _dbg_bail("no core_attn_metadata")

        importance = None
        used = "out_degree"
        if mode == "in_degree":
            importance = self._redknot_indexer_indegree(core_meta, num_tokens, device)
            if importance is not None:
                used = "in_degree"
        if importance is None:
            # out-degree (also the fallback for in-degree)
            lengths = getattr(core_meta, "c4_topk_lengths_raw", None)
            if lengths is None:
                return _dbg_bail("c4_topk_lengths_raw is None")
            lengths = lengths.reshape(-1)
            if lengths.numel() != num_tokens:
                return _dbg_bail(
                    f"len mismatch lengths={lengths.numel()} num_tokens={num_tokens}"
                )
            importance = lengths.to(device=device, dtype=torch.float32).clamp_min(0)

        if importance is None or importance.numel() == 0:
            return None
        if os.environ.get("SGLANG_REDKNOT_FFN_DEBUG") == "1" and not getattr(
            self, "_redknot_ffn_idx_logged", False
        ):
            logger.info(
                "[RedKnot sparse-FFN] layer %d using INDEXER importance "
                "(%s): tokens=%d",
                self.layer_id,
                used,
                num_tokens,
            )
            self._redknot_ffn_idx_logged = True
        return importance

    def _redknot_indexer_indegree(
        self, core_meta, num_tokens: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """In-degree importance = how many queries selected each current token's
        c4 page. Uses ``c4_sparse_page_indices`` (per-query selected pages,
        encoded as ``(physical_page << page_bits) | offset``) and ``c4_out_loc``
        (each current token's write location in the c4 KV pool). Returns ``None``
        on any shape/availability mismatch so the caller can fall back."""
        try:
            page_idx = getattr(core_meta, "c4_sparse_page_indices", None)
            c4_out_loc = getattr(core_meta, "c4_out_loc", None)
            page_size = int(getattr(core_meta, "page_size", 0))
            if page_idx is None or c4_out_loc is None or page_size <= 0:
                return None
            # c4 page granularity is page_size // 4 (compress ratio 4).
            c4_page_size = max(1, page_size // 4)
            page_bits = int(c4_page_size).bit_length() - 1
            # Only exact powers of two give a clean shift decode; otherwise bail.
            if (1 << page_bits) != c4_page_size:
                return None

            flat = page_idx.reshape(-1)
            valid = flat[flat >= 0]
            if valid.numel() == 0:
                return None
            # physical page id of each selected entry
            sel_pages = valid.to(torch.int64) >> page_bits
            n_pages = int(sel_pages.max().item()) + 1
            popularity = torch.bincount(sel_pages, minlength=n_pages).to(torch.float32)

            # current tokens' c4 physical page id
            loc = c4_out_loc.reshape(-1).to(torch.int64)
            if loc.numel() != num_tokens:
                return None
            tok_pages = loc // c4_page_size
            tok_pages = tok_pages.clamp_(0, n_pages - 1)
            importance = popularity.index_select(0, tok_pages).to(device=device)
            return importance
        except Exception:  # noqa: BLE001
            return None

    def _select_redknot_sparse_ffn_tokens(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        server_args = get_global_server_args()
        if (
            not getattr(server_args, "redknot_sparse_ffn_enable", False)
            or self.layer_id < server_args.redknot_sparse_ffn_dense_until
            or self.layer_id
            >= int(self.config.num_hidden_layers)
            - int(os.environ.get("REDKNOT_FFN_DENSE_SUFFIX_LAYERS", "0"))
            or hidden_states.shape[0] <= 1
        ):
            return None
        if getattr(self.mlp, "num_fused_shared_experts", 0) != 0 or not hasattr(
            self.mlp, "shared_experts"
        ):
            return None
        raw_plans = getattr(forward_batch, "redknot_reuse_plan", None)
        restore_plans = tuple(
            plan
            for plan in (raw_plans or ())
            if isinstance(plan, Mapping)
            and plan.get("mode") == "restore"
            and plan.get("reuse_mla_off") is True
        )
        if (
            int(getattr(forward_batch, "batch_size", 0)) != 1
            or len(restore_plans) != 1
        ):
            return None
        restore_plan = restore_plans[0]
        from sglang.srt.layers.attention.redknot.v4.reuse_planner import (
            MLA_OFF_EXECUTION_PROFILE,
            MLA_OFF_INDEPENDENT_RELOCATION_PROFILE,
        )

        execution_profile = restore_plan.get("mla_off_execution_profile")
        context_bound_restore = bool(
            execution_profile == MLA_OFF_EXECUTION_PROFILE
            and restore_plan.get("allow_approximate") is False
        )
        independent_relocation_restore = bool(
            execution_profile == MLA_OFF_INDEPENDENT_RELOCATION_PROFILE
            and restore_plan.get("allow_approximate") is True
        )
        if (
            restore_plan.get("mla_off_qualification_only") is not True
            or not (context_bound_restore or independent_relocation_restore)
        ):
            return None

        min_seq_len = int(
            getattr(server_args, "redknot_sparse_ffn_min_seq_len", 0) or 0
        )
        if (
            min_seq_len > 0
            and forward_batch.seq_lens_cpu is not None
            and int(forward_batch.seq_lens_cpu.max().item()) < min_seq_len
        ):
            return None

        mass_thresh = server_args.redknot_sparse_ffn_mass_thresh
        if self.layer_id >= server_args.redknot_sparse_ffn_deep_start:
            mass_thresh = server_args.redknot_sparse_ffn_mass_thresh_deep
        if mass_thresh >= 1.0:
            return None

        num_tokens = hidden_states.shape[0]
        block_tokens = int(os.environ.get("REDKNOT_FFN_BLOCK_TOKENS", "0") or 0)
        freeze_blocks = os.environ.get(
            "REDKNOT_FFN_FREEZE_BLOCK_SELECTION", "0"
        ) == "1"
        if block_tokens < 0 or (block_tokens and block_tokens % 128 != 0):
            return None
        selection_generation = getattr(
            forward_batch, "_redknot_forward_generation_id", None
        )
        raw_query_start = restore_plan.get("query_start")
        raw_total_tokens = restore_plan.get("total_tokens")
        block_selection_key = (
            selection_generation,
            int(positions.data_ptr()),
            int(getattr(positions, "_version", -1)),
            int(num_tokens),
            int(block_tokens),
            float(
                getattr(
                    server_args, "redknot_sparse_ffn_max_full_ratio", 0.80
                )
            ),
            int(os.environ.get("REDKNOT_FFN_BOUNDARY_TOKENS", "128")),
            int(server_args.redknot_sparse_ffn_recent_n),
            raw_query_start if type(raw_query_start) is int else -1,
            raw_total_tokens if type(raw_total_tokens) is int else -1,
        )
        cached_block_selection = getattr(
            forward_batch, "_redknot_sparse_ffn_block_selection", None
        )
        if (
            block_tokens
            and freeze_blocks
            and isinstance(cached_block_selection, tuple)
            and len(cached_block_selection) == 2
            and cached_block_selection[0] == block_selection_key
        ):
            selected_idx = cached_block_selection[1]
            if (
                isinstance(selected_idx, torch.Tensor)
                and selected_idx.dtype == torch.long
                and selected_idx.device == hidden_states.device
            ):
                return selected_idx
            return None
        importance_mode = getattr(
            server_args, "redknot_sparse_ffn_importance", "activation"
        )

        # Activation magnitude is always available and is the safe fallback.
        act_importance = hidden_states.float().norm(dim=-1).clamp_min(0)

        importance = act_importance
        if importance_mode in ("indexer", "indexer_indegree", "blend"):
            idx_mode = (
                "in_degree" if importance_mode == "indexer_indegree" else "out_degree"
            )
            idx_importance = self._redknot_indexer_token_importance(
                num_tokens, hidden_states.device, mode=idx_mode
            )
            if idx_importance is not None:
                if importance_mode in ("indexer", "indexer_indegree"):
                    importance = idx_importance
                else:  # blend: indexer-mass-weighted activation norm
                    # Normalize the indexer mass to [eps, 1] so it reweights
                    # rather than dominates the activation magnitude.
                    idx_norm = idx_importance / idx_importance.max().clamp_min(1.0)
                    importance = act_importance * idx_norm.clamp_min(1e-3)
            elif importance_mode in ("indexer", "indexer_indegree"):
                # Strict indexer-guided mode stays dense when this layer has no
                # native Indexer signal (e.g. SWA/HCA layers).
                return None

        max_full_ratio = float(
            getattr(server_args, "redknot_sparse_ffn_max_full_ratio", 0.80)
        )
        use_fixed_budget = max_full_ratio < 1.0
        if importance.numel() == 0:
            return None
        if not use_fixed_budget and importance.sum() <= 0:
            return None
        keep = torch.zeros_like(importance, dtype=torch.bool)
        if not use_fixed_budget:
            sorted_imp, sorted_idx = torch.sort(importance, descending=True)
            cum_frac = torch.cumsum(sorted_imp, dim=0) / importance.sum().clamp_min(
                torch.finfo(sorted_imp.dtype).tiny
            )
            rank_keep = cum_frac < mass_thresh
            crossing = torch.nonzero(cum_frac >= mass_thresh, as_tuple=False)
            if crossing.numel() > 0:
                rank_keep[crossing[0, 0]] = True
            rank_keep[0] = True
            keep.scatter_(0, sorted_idx, rank_keep)
        protected = torch.zeros_like(keep)
        token_positions = positions[:num_tokens].to(torch.int64)
        query_start = restore_plan.get("query_start")
        total_tokens = restore_plan.get("total_tokens")
        segments = restore_plan.get("segments")
        if (
            type(query_start) is not int
            or type(total_tokens) is not int
            or query_start < 0
            or total_tokens < query_start
            or not isinstance(segments, (list, tuple))
        ):
            return None
        protected |= token_positions >= query_start
        seq_len = (
            int(forward_batch.seq_lens_cpu.max().item())
            if forward_batch.seq_lens_cpu is not None
            else total_tokens
        )
        request_begin = seq_len - num_tokens
        request_end = seq_len
        if request_begin < 0 or request_end > total_tokens:
            return None
        protected_intervals = []

        def _record_protected_interval(begin: int, end: int) -> None:
            clipped_begin = max(request_begin, int(begin))
            clipped_end = min(request_end, int(end))
            if clipped_begin < clipped_end:
                protected_intervals.append((clipped_begin, clipped_end))

        _record_protected_interval(query_start, total_tokens)
        boundary_tokens = int(os.environ.get("REDKNOT_FFN_BOUNDARY_TOKENS", "128"))
        if boundary_tokens < 0:
            return None
        for segment in segments:
            if not isinstance(segment, Mapping):
                return None
            begin = segment.get("global_offset")
            length = segment.get("length")
            if type(begin) is not int or type(length) is not int or length <= 0:
                return None
            end = begin + length
            if boundary_tokens:
                protected |= (token_positions >= begin) & (
                    token_positions < min(end, begin + boundary_tokens)
                )
                protected |= (token_positions >= max(begin, end - boundary_tokens)) & (
                    token_positions < end
                )
                _record_protected_interval(begin, min(end, begin + boundary_tokens))
                _record_protected_interval(max(begin, end - boundary_tokens), end)

        recent_n = int(server_args.redknot_sparse_ffn_recent_n)
        if recent_n > 0:
            protected |= token_positions >= max(0, seq_len - recent_n)
            _record_protected_interval(max(0, seq_len - recent_n), seq_len)

        if use_fixed_budget:
            protected_intervals.sort()
            protected_count = 0
            merged_end = request_begin
            for begin, end in protected_intervals:
                if end <= merged_end:
                    continue
                if begin > merged_end:
                    protected_count += end - begin
                    merged_end = end
                else:
                    protected_count += end - merged_end
                    merged_end = end
            if block_tokens:
                from sglang.srt.layers.attention.redknot.sparse_ffn import (
                    select_fixed_budget_blocks,
                )

                selected_idx = select_fixed_budget_blocks(
                    importance,
                    protected,
                    token_positions,
                    max_full_ratio=max_full_ratio,
                    protected_count=protected_count,
                    block_tokens=block_tokens,
                )
                if freeze_blocks:
                    forward_batch._redknot_sparse_ffn_block_selection = (
                        block_selection_key,
                        selected_idx,
                    )
                return selected_idx

            from sglang.srt.layers.attention.redknot.sparse_ffn import (
                select_fixed_budget_tokens,
            )

            return select_fixed_budget_tokens(
                importance,
                protected,
                max_full_ratio=max_full_ratio,
                protected_count=protected_count,
            )

        from sglang.srt.layers.attention.redknot.sparse_ffn import (
            enforce_token_selection_bounds,
        )

        return enforce_token_selection_bounds(
            importance,
            keep,
            protected,
            min_full_ratio=float(
                getattr(server_args, "redknot_sparse_ffn_min_full_ratio", 0.20)
            ),
            max_full_ratio=max_full_ratio,
        )

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        with _redknot_v4_timed("attn_hc_pre_norm", layer_id=self.layer_id):
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm=self.input_layernorm,
            )
            if not norm_fused:
                if _use_aiter and _is_gfx95_supported:
                    x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
                        hidden_states,
                        self.input_layernorm.weight,
                        self.rms_norm_eps,
                    )
                else:
                    hidden_states = self.input_layernorm(hidden_states)
                    x_quant = None
            else:
                x_quant = None

        with _redknot_v4_timed("self_attn_total", layer_id=self.layer_id):
            hidden_states = self.self_attn(
                x=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                x_quant=x_quant,
            )

        with _redknot_v4_timed("attn_hc_post", layer_id=self.layer_id):
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
        with _redknot_v4_timed("ffn_hc_pre_norm", layer_id=self.layer_id):
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                norm=self.post_attention_layernorm,
            )
            if not norm_fused:
                hidden_states = self.post_attention_layernorm(hidden_states)

        _use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()
            and get_attention_tp_size() > 1
            and not get_moe_a2a_backend().is_none()
        )
        if _use_cp:
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            cp_rank = get_attention_cp_rank()
            cp_size = get_attention_cp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            input_ids_global = input_ids
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_attention_tp_size(), get_attention_tp_rank()
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            input_ids = input_ids.tensor_split(s)[r].contiguous()
            input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        with _redknot_v4_timed("ffn_select", layer_id=self.layer_id):
            sparse_ffn_keep = None
            if not (_use_cp or _use_tp_moe_gather or _use_tp_attn_a2a_scatter):
                sparse_ffn_keep = self._select_redknot_sparse_ffn_tokens(
                    hidden_states, positions, forward_batch
                )

        with _redknot_v4_timed("moe", layer_id=self.layer_id):
            if sparse_ffn_keep is None:
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                    input_ids=input_ids,
                    input_ids_global=input_ids_global,
                )
            elif sparse_ffn_keep.dtype != torch.bool:
                selected_idx = sparse_ffn_keep
                _n = int(selected_idx.numel())
                _total = int(hidden_states.shape[0])
                _redknot_ffn_stats_record(_n, _total)
                if _n == _total:
                    hidden_states = self.mlp(
                        hidden_states,
                        forward_batch,
                        input_ids=input_ids,
                        input_ids_global=input_ids_global,
                    )
                else:
                    hidden_states = self.mlp.forward_redknot_sparse(
                        hidden_states,
                        selected_idx,
                        input_ids_global=input_ids_global,
                    )
            elif bool(sparse_ffn_keep.all()):
                if sparse_ffn_keep is not None:
                    # All tokens kept on a sparse-eligible layer: keep_ratio=1 row.
                    _n = int(sparse_ffn_keep.numel())
                    _redknot_ffn_stats_record(_n, _n)
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                    input_ids=input_ids,
                    input_ids_global=input_ids_global,
                )
            else:
                selected_idx = torch.nonzero(
                    sparse_ffn_keep, as_tuple=False
                ).flatten()
                _redknot_ffn_stats_record(
                    int(selected_idx.numel()), int(sparse_ffn_keep.numel())
                )
                hidden_states = self.mlp.forward_redknot_sparse(
                    hidden_states,
                    selected_idx,
                    input_ids_global=input_ids_global,
                )

        with _redknot_v4_timed("ffn_collective", layer_id=self.layer_id):
            if _use_tp_moe_gather:
                hidden_states, global_hidden_states = (
                    get_local_dp_buffer(get_tp_group()),
                    hidden_states,
                )
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
            if _use_tp_attn_a2a_scatter:
                assert _a2a_scatter_chunks is not None
                gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
                attn_tp_all_gather(gathered, hidden_states.contiguous())
                hidden_states = torch.cat(gathered)

        with _redknot_v4_timed("ffn_hc_post", layer_id=self.layer_id):
            hidden_states = self.hc_post(hidden_states, residual, post, comb)

        if (
            self.layer_id == int(self.config.num_hidden_layers) - 4
            and os.environ.get("SGLANG_REDKNOT_FFN_DEBUG") == "1"
            and getattr(get_global_server_args(), "redknot_sparse_ffn_enable", False)
        ):
            snap = redknot_ffn_stats_snapshot()
            request_ids = tuple(
                str(plan.get("benchmark_request_id", ""))
                for plan in (getattr(forward_batch, "redknot_reuse_plan", None) or ())
                if isinstance(plan, Mapping)
                and plan.get("mode") == "restore"
                and plan.get("reuse_mla_off") is True
            )
            logger.info(
                "[RedKnot sparse-FFN FORWARD] requests=%s full=%d "
                "shared_only=%d total=%d full_ratio=%.9f "
                "expert_compute_ratio=%.9f",
                request_ids,
                snap["kept_tokens"],
                snap["shared_only_tokens"],
                snap["total_tokens"],
                snap["full_ratio"],
                snap["expert_compute_ratio"],
            )

        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.rms_norm_eps = config.rms_norm_eps
        use_stream_pool = _is_cuda or (
            _is_hip
            and (
                envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                or envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            )
        )
        num_alt_streams = 5 if _is_cuda else 2
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(num_alt_streams)]
            if use_stream_pool
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.gemm_output_zero_allocator_size = 0
        self.layers_to_capture: List[int] = []
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        if self.pp_group.is_last_rank:
            hc_dim = hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()

    def prewarm_mhc_token_count_buckets(
        self, max_num_tokens: int, device: torch.device
    ) -> Tuple[int, ...]:
        tic = time.perf_counter()
        logger.info(
            "Running DeepSeek V4 MHC prewarm for max_num_tokens=%s",
            max_num_tokens,
        )
        token_counts = self.layers[self.start_layer].prewarm_mhc_token_count_buckets(
            max_num_tokens, device
        )
        logger.info(
            "DeepSeek V4 MHC prewarm finished in %.3fs for representative token counts: %s",
            time.perf_counter() - tic,
            token_counts,
        )
        return token_counts

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        if x.numel() > 0:
            from sglang.srt.layers.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            # Unflatten 2D PP IPC tensor back to 3D mHC shape.
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.view(
                    hidden_states.shape[0], self.hc_mult, self.hidden_size
                )

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if dsa_use_prefill_cp(forward_batch):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        # Reset Compressor's per-step freqs_cis cache from any previous step.
        for _attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, _attr):
                delattr(forward_batch, _attr)
        # Upgrade lazy raw metadata on the main stream once before any layer
        # forks alt-streams; later per-layer calls become no-ops.
        attn_backend = get_attn_backend()
        attn_backend._maybe_upgrade_forward_metadata()
        begin_mla_off_forward = getattr(
            attn_backend, "begin_mla_off_forward_transaction", None
        )
        finalize_mla_off_forward = getattr(
            attn_backend, "finalize_mla_off_forward_transaction", None
        )
        aux_hidden_states = []
        active_layer_id = -1
        try:
            for i in range(self.start_layer, self.end_layer):
                active_layer_id = int(i)
                # Dense layers 0..2 complete before the omission certificate.
                # This removes a postcommit failure window while keeping the
                # exact 3..39 reusable domain covered by one prepare/final
                # pair.  Pipeline partitions that do not own layer 3 never
                # attempt a partial certificate.
                if int(i) == 3 and callable(begin_mla_off_forward):
                    reusable_layers = tuple(
                        self.layers[layer_id].self_attn
                        for layer_id in range(3, 40)
                        if self.start_layer <= layer_id < self.end_layer
                    )
                    with _redknot_v4_timed("mla_forward_prepare", layer_id=3):
                        begin_mla_off_forward(
                            positions=positions,
                            forward_batch=forward_batch,
                            layers=reusable_layers,
                            q_row_count=int(hidden_states.shape[0]),
                            device=hidden_states.device,
                            projection_dtype=hidden_states.dtype,
                        )
                layer = self.layers[i]
                ctx = (
                    nullcontext()
                    if not get_global_server_args().disable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    with _redknot_v4_timed("decoder_layer", layer_id=int(i)):
                        hidden_states = layer(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            input_ids=input_ids,
                            input_ids_global=input_ids_global,
                        )
                if int(i) == 39 and callable(finalize_mla_off_forward):
                    # Finalize only after the complete decoder layer returns,
                    # including wo_b and its MLP/MoE path.
                    with _redknot_v4_timed(
                        "mla_forward_finalize", layer_id=39
                    ):
                        finalize_mla_off_forward(
                            forward_batch=forward_batch,
                            layer_id=int(i),
                        )
                if i in self.layers_to_capture:
                    # D-Spark consumes the mean over mHC lanes from each selected
                    # target layer, matching the official reference implementation.
                    aux_hidden_states.append(hidden_states.mean(dim=1))
        except BaseException as model_error:
            abort_forward = getattr(
                attn_backend, "abort_mla_off_forward_transaction", None
            )
            if callable(abort_forward):
                abort_forward(
                    forward_batch=forward_batch,
                    error=model_error,
                    layer_id=active_layer_id,
                )
            raise

        # CP all-gather only on the last PP rank; PP IPC carries CP-split tensors.
        if self.pp_group.is_last_rank and dsa_use_prefill_cp(forward_batch):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        if not self.pp_group.is_last_rank:
            # Flatten 3D mHC tensor for PP IPC.
            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})

        pre_hc_head = hidden_states.flatten(1)

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        output = (hidden_states, pre_hc_head)
        if self.layers_to_capture:
            if len(aux_hidden_states) != len(self.layers_to_capture):
                raise RuntimeError(
                    "DeepSeek V4 captured "
                    f"{len(aux_hidden_states)} of {len(self.layers_to_capture)} "
                    "requested D-Spark hidden states"
                )
            return output, aux_hidden_states
        return output


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.model.start_layer, self.model.end_layer)
                if isinstance(
                    self.model.layers[layer_id].mlp, deepseek_v2.DeepseekV2MoE
                )
            }
        )

        # Expose start_layer/end_layer for model_runner PP support
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_rank = get_attention_cp_rank()
            self.cp_size = get_attention_cp_size()

        self._maybe_enable_mla_head_profiler(config)

    @staticmethod
    def _maybe_enable_mla_head_profiler(config: DeepSeekV4Config) -> None:
        """Turn on the RedKnot MLA head-locality collector if requested."""
        server_args = get_global_server_args()
        if not getattr(server_args, "redknot_mla_profile_enable", False):
            return
        from sglang.srt.layers.attention.redknot.mla_head_profiler import (
            MLAHeadProfileConfig,
            enable_global_collector,
        )

        # Each TP rank runs its own collector over its *local* head shard. The
        # MLA latent KV is shared across heads (so every rank has the full key
        # stream), and the token-mass concentration metric is a per-head average,
        # so per-rank local-head collection + offline cross-rank averaging gives
        # the full-head curve. (The distance-histogram head classification is
        # only complete at TP=1; the concentration curve works at any TP.)
        try:
            tp_size = get_attention_tp_size()
        except Exception:
            tp_size = 1
        total_heads = int(config.num_attention_heads)
        local_heads = max(1, total_heads // max(1, tp_size))
        cfg = MLAHeadProfileConfig(
            num_layers=int(config.num_hidden_layers),
            num_heads=local_heads,
            expected_context=int(
                os.environ.get("REDKNOT_MLA_PROFILE_EXPECTED_CONTEXT", "0")
            ),
            coverage=float(server_args.redknot_mla_profile_coverage),
            sample_queries=int(server_args.redknot_mla_profile_sample_queries),
            query_window_quantile=float(
                os.environ.get("REDKNOT_MLA_PROFILE_QUERY_QUANTILE", "0.90")
            ),
            global_window_ratio=float(
                server_args.redknot_mla_profile_global_window_ratio
            ),
            window_safety=float(server_args.redknot_mla_profile_window_safety),
            dense_prefix_layers=int(server_args.redknot_mla_dense_prefix_layers),
            dense_suffix_layers=int(server_args.redknot_mla_dense_suffix_layers),
        )
        enable_global_collector(cfg)
        logger.info(
            "RedKnot MLA head profiler enabled: layers=%d local_heads=%d "
            "(total=%d, tp=%d) coverage=%.3f expected_context=%d",
            cfg.num_layers,
            cfg.num_heads,
            total_heads,
            tp_size,
            cfg.coverage,
            cfg.expected_context,
        )

    def prewarm_mhc_token_count_buckets(
        self, max_num_tokens: int, device: torch.device
    ) -> Tuple[int, ...]:
        return self.model.prewarm_mhc_token_count_buckets(max_num_tokens, device)

    def kernel_warmup(self, model_runner) -> None:
        if not model_runner.is_hybrid_swa:
            return
        if not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            return
        if not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            return

        max_num_tokens = model_runner.server_args.chunked_prefill_size
        if max_num_tokens is None or max_num_tokens <= 0:
            max_num_tokens = 8192

        token_counts = self.prewarm_mhc_token_count_buckets(
            max_num_tokens, model_runner.device
        )
        model_runner.tp_group.barrier()

        logger.info(
            "DeepSeek V4 MHC prewarm completed for representative token-count shapes: %s",
            token_counts,
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Waterfill needs shared-experts fusion so it can dispatch shared
        # expert tokens to least-loaded EP ranks.
        if get_global_server_args().enable_deepep_waterfill:
            if self.config.n_shared_experts != 1:
                raise ValueError(
                    "DeepEP Waterfill for DeepSeek V4 expects exactly one shared "
                    f"expert, but got n_shared_experts={self.config.n_shared_experts}."
                )
            self.num_fused_shared_experts = self.config.n_shared_experts
            log_info_on_rank0(
                logger,
                "DeepSeek V4: --enable-deepep-waterfill set; KEEP shared-experts "
                "fusion enabled so waterfill can rebalance shared expert dispatch.",
            )
            return

        get_global_server_args().disable_shared_experts_fusion = True
        log_info_on_rank0(
            logger,
            "DeepSeek V4 requires different clamping for shared and routed experts. "
            "Shared experts fusion optimization is disabled.",
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_lens=forward_batch.extend_seq_lens_cpu,
                )
                if is_dsa_prefill_cp_round_robin_split():
                    attn_backend = get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related()
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=pre_hc_head,
        )

    def set_dspark_layers_to_capture(self, layer_ids: List[int]) -> None:
        """Capture post-layer lane-mean states used by the D-Spark proposer."""

        if self.pp_group.world_size != 1:
            raise ValueError("D-Spark hidden-state capture currently requires pp_size=1")
        if not layer_ids:
            raise ValueError("D-Spark requires explicit target layer IDs")

        normalized = [int(layer_id) for layer_id in layer_ids]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"D-Spark target layers must be unique: {normalized}")
        if sorted(normalized) != normalized:
            raise ValueError(
                f"D-Spark target layers must be strictly increasing: {normalized}"
            )
        invalid = [
            layer_id
            for layer_id in normalized
            if layer_id < self.model.start_layer or layer_id >= self.model.end_layer
        ]
        if invalid:
            raise ValueError(f"D-Spark target layers are not local: {invalid}")

        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = normalized

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        if is_nextn:
            layers = [self.model.decoder]
        else:
            layers = [
                self.model.layers[layer_id]
                for layer_id in range(self.model.start_layer, self.model.end_layer)
            ]
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        if is_nextn:
            return
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str, is_nextn: bool = False, num_hidden_layers: Optional[int] = None
    ) -> str:
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if "self_attn" in name:
            name = name.replace(".scale", ".weight_scale_inv")

        name = name.replace(".gate.tid2eid", ".topk.tid2eid")
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if "mlp" in name:
            name = name.replace(".scale", ".weight_scale_inv")

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        if not envs.SGLANG_OPT_FP8_WO_A_GEMM.get():
            weights = list(weights)
            exists_wo_a_scale = any(n.endswith(".wo_a.scale") for n, t in weights)
            if exists_wo_a_scale:
                logger.info("Execute dequant fp8 wo_a")
                weights = _dequant_fp8_wo_a(weights)
            else:
                logger.info("Skip dequant fp8 wo_a")

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = not _is_hip and envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if (
                                name == "model.norm.weight"
                                and not self.pp_group.is_last_rank
                            ):
                                continue
                            if (
                                name.startswith("model.hc_head_")
                                or name == "lm_head.weight"
                            ) and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                bucket = cache_wqkv_a_weight.setdefault(param_name, {})
                                shard_key = "q" if is_q else "kv"
                                assert shard_key not in bucket, (
                                    f"duplicate shard {shard_key} for {param_name}"
                                )
                                bucket[shard_key] = loaded_weight
                                if len(bucket) == 2:
                                    fused_weight = torch.cat(
                                        [bucket["q"], bucket["kv"]], dim=0
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(param_name)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if not self.pp_group.is_first_rank:
            skipped_checking_patterns.append("embed_tokens")
        if not self.pp_group.is_last_rank:
            skipped_checking_patterns.append("model.norm.")
            skipped_checking_patterns.extend(["lm_head", "hc_head_"])
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from einops import rearrange

    assert weight.dtype == torch.float8_e4m3fn, (
        f"expected fp8_e4m3fn, got {weight.dtype}"
    )
    assert scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float32,
    ), f"expected fp8_e8m0fnu or float32, got {scale.dtype}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    return result.to(torch.bfloat16)


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()
