from __future__ import annotations

import logging
from typing import Literal, Optional

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4AttnMetadata,
    DeepseekV4AttnBackend,
    _create_flashmla_metadata,
    _pad_last_dim,
    _pad_tensor_to_size,
)
from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
    DeepSeekV4MLAHeadConfig,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class RedKnotMLAAttnBackend(DeepseekV4AttnBackend):
    """Experimental RedKnot backend for DeepSeek V4 MLA.

    The physical cache remains DeepSeek V4's packed FlashMLA cache. RedKnot is
    applied at the logical attention-head level by splitting Q heads into
    local/global groups:

    - local heads: attend only the native SWA cache window;
    - global/dense heads: use the normal DSV4 SWA + compressed extra cache.

    This is the first runnable step toward RedKnot+MLA prefill. It deliberately
    does not unpack MLA latent KV into traditional per-head K/V tensors.
    """

    def __init__(self, model_runner, *args, **kwargs):
        super().__init__(model_runner, *args, **kwargs)
        server_args = model_runner.server_args
        hf_config = model_runner.model_config.hf_config
        cfg_path = getattr(server_args, "redknot_head_config_path", None)
        if cfg_path:
            self.redknot_mla_head_cfg = DeepSeekV4MLAHeadConfig.from_json(cfg_path)
        else:
            self.redknot_mla_head_cfg = DeepSeekV4MLAHeadConfig.from_model_config(
                hf_config,
                dense_prefix_layers=getattr(
                    server_args, "redknot_mla_dense_prefix_layers", 2
                ),
                local_window=getattr(server_args, "redknot_mla_local_window", 128),
                global_head_stride=getattr(
                    server_args, "redknot_mla_global_head_stride", 8
                ),
                global_layer_stride=getattr(
                    server_args, "redknot_mla_global_layer_stride", 0
                ),
            )
        logger.info(
            "RedKnot MLA policy loaded: %s", self.redknot_mla_head_cfg.summary()
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
        **kwargs,
    ) -> torch.Tensor:
        self._maybe_upgrade_forward_metadata()

        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        if not isinstance(core_attn_metadata, DSV4AttnMetadata):
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

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        assert attn_sink is not None
        layer_id = layer.layer_id
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        if save_kv_cache:
            self.store_cache(layer_id, k, forward_batch)

        if q.ndim == 3:
            q = q.unsqueeze(1)

        layer_plan = self.redknot_mla_head_cfg.layer_tensors(layer_id, q.device)
        is_local = layer_plan["is_local"]
        if is_local.numel() != q.shape[2]:
            # Tensor parallel or speculative variants can present a different
            # head view. Fall back to the native path rather than silently
            # applying the wrong logical-head policy.
            return super().forward(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=compress_ratio,
                save_kv_cache=False,
                attn_sink=attn_sink,
                **kwargs,
            )

        local_idx = torch.nonzero(is_local, as_tuple=False).flatten()
        global_idx = torch.nonzero(~is_local, as_tuple=False).flatten()
        # Local window size: local heads only attend their most-recent
        # ``local_window`` SWA tokens. Use the min window across local heads
        # (conservative: smaller window = more saving). Falls back to the
        # configured default when windows are unset.
        local_window = None
        windows = layer_plan.get("windows")
        if windows is not None and local_idx.numel() > 0:
            lw = windows.index_select(0, local_idx)
            lw = lw[lw > 0]
            if lw.numel() > 0:
                local_window = int(lw.min().item())
        if local_idx.numel() == 0 or global_idx.numel() == 0:
            return super().forward(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=compress_ratio,
                save_kv_cache=False,
                attn_sink=attn_sink,
                **kwargs,
            )

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

        # The global (with-extra) and local (no-extra) passes must use SEPARATE
        # FlashMLA sched_meta objects: the sched_meta caches config (batch size
        # ``b``, ``extra_page_block_size`` ...) from its first call and asserts
        # every later call on the same object matches. Sharing one object across
        # a with-extra and a no-extra call (or across batches) trips that
        # assertion. dsv4 rebuilds the global sched_meta per forward batch (it
        # lives on core_attn_metadata); we mirror that for the local one by
        # creating a fresh empty sched_meta each forward (cheap: it is filled
        # lazily on first kernel call).
        flashmla_metadata_global = core_attn_metadata.get_flashmla_metadata(
            compress_ratio
        )
        flashmla_metadata_local = _create_flashmla_metadata()

        import flash_mla

        # For the LOCAL pass, restrict each query to its most-recent
        # ``local_window`` SWA tokens (true local-window sparsity) by clamping
        # the per-query topk_length. The SWA indices are ordered so clamping the
        # length keeps the most-recent window. The global pass keeps full SWA +
        # compressed extra cache. This makes local heads genuinely cheaper
        # rather than just "SWA without extra".
        if local_window is not None:
            swa_topk_lengths_local = torch.clamp(swa_topk_lengths, max=local_window)
        else:
            swa_topk_lengths_local = swa_topk_lengths

        def run_flashmla(q_part, sink_part, use_extra: bool):
            return flash_mla.flash_mla_with_kvcache(
                q=q_part,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=(
                    flashmla_metadata_global if use_extra else flashmla_metadata_local
                ),
                softmax_scale=self.softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths if use_extra else swa_topk_lengths_local,
                attn_sink=sink_part,
                extra_k_cache=extra_k_cache if use_extra else None,
                extra_indices_in_kvcache=extra_indices if use_extra else None,
                extra_topk_length=extra_topk_lengths if use_extra else None,
            )[0]

        # This FlashMLA build only supports h_q == 64 (see FlashMLA
        # sparse_fwd.h / sparse_decode.h: "Unsupported h_q" for any other head
        # count). Physically splitting Q into local/global head groups would
        # produce invalid h_q (e.g. 8), so instead we run the kernel twice on
        # the FULL head set (h_q stays valid) -- once WITH the compressed extra
        # cache (the "global" view) and once WITHOUT it (the "local",
        # SWA-window-only view) -- then select per head: global/dense heads keep
        # the extra-cache output, local heads keep the window-only output.
        #
        # Cost: 2x attention compute. This is the correctness-first path that
        # unblocks RedKnot head sparsity + MoE sparse-FFN running together; a
        # fused single-pass per-head-mask kernel is future work.
        o_global = run_flashmla(q, attn_sink, True)
        o_local = run_flashmla(q, attn_sink, False)

        # Per-head select: out[:, :, h] = o_local[h] if local else o_global[h].
        # is_local shape [H] -> broadcast over [s, 1, H, d_v].
        local_mask = is_local.view(1, 1, -1, 1)
        out = torch.where(local_mask, o_local, o_global)
        return out.squeeze(1)


__all__ = ["RedKnotMLAAttnBackend"]
