# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

from __future__ import annotations

import contextlib
import datetime
import gc
import hashlib
import inspect
import logging
import math
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
    get_num_indexer_layers,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_default_distributed_backend,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    join_process_groups,
    try_recover_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.breakable_cuda_graph_runner import (
    BreakableCudaGraphRunner,
)
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    DecodeInputBuffers,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    cpu_has_amx_support,
    dynamic_import,
    empty_context,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.common import ceil_align, require_mlp_sync
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "ascend",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "intel_xpu",
]

CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
]

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


def add_chunked_prefix_cache_attention_backend(backend_name):
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)

_UNSET: Any = object()


def resolve_language_model(model: nn.Module) -> nn.Module:
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        return model.thinker.model
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "language_model"):
        return model.language_model
    return model.model


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = getattr(
            model_config, "is_hybrid_swa_compress", False
        )
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        rope_scaling = getattr(
            model_config.hf_text_config, "rope_parameters", None
        ) or getattr(model_config.hf_text_config, "rope_scaling", {})
        self.model_is_mrope = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None

        self.msprobe_debugger = None
        if server_args.msprobe_dump_config is not None:
            self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.eagle_use_aux_hidden_state = False
        self.dflash_use_aux_hidden_state = False
        self.dspark_use_aux_hidden_state = False
        self.dflash_target_layer_ids = None
        self.dflash_draft_num_layers = None
        self.dspark_target_layer_ids = None
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # load draft config
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            self.eagle_use_aux_hidden_state = True

            try:
                # get the aux layer from draft model config
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                self.eagle_use_aux_hidden_state = eagle_config.get(
                    "use_aux_hidden_state", True
                )
                self.eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                self.eagle_aux_hidden_state_layer_ids = None

        if self.spec_algorithm.is_dflash() and not self.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                parse_dflash_draft_config,
            )

            # Select target layers to capture for building DFlash context features.
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            dflash_draft_config = parse_dflash_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            draft_num_layers = dflash_draft_config.require_num_layers()
            trained_target_layers = dflash_draft_config.num_target_layers

            target_num_layers = getattr(
                self.model_config.hf_text_config, "num_hidden_layers", None
            )
            if target_num_layers is None:
                raise ValueError(
                    "DFLASH requires target num_hidden_layers in config. "
                    f"Got target={target_num_layers}."
                )
            target_num_layers = int(target_num_layers)

            if (
                trained_target_layers is not None
                and trained_target_layers != target_num_layers
            ):
                logger.warning(
                    "DFLASH draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                    "selecting capture layers based on the runtime target model.",
                    trained_target_layers,
                    target_num_layers,
                )

            self.dflash_use_aux_hidden_state = True
            self.dflash_draft_num_layers = int(draft_num_layers)
            self.dflash_target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
                target_num_layers=int(target_num_layers),
                draft_num_layers=int(draft_num_layers),
            )

        if server_args.speculative_algorithm == "DSPARK" and not self.is_draft_worker:
            from sglang.srt.speculative.dspark_utils import parse_dspark_config

            dspark_config = parse_dspark_config(self.model_config.hf_text_config)
            self.dspark_use_aux_hidden_state = True
            self.dspark_target_layer_ids = list(dspark_config.target_layer_ids)

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Get available memory before model loading
        pre_model_load_memory = self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # CPU offload
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        self.hisparse_coordinator = None

        self._linear_attn_registry_cache: Any = _UNSET

        # Initialize the model runner
        self.initialize(pre_model_load_memory)
        self.check_quantized_moe_compatibility()

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            )
            ElasticEPStateManager.instance().reset()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            assert self.support_pp, (
                "Pipeline Parallel is not compatible with this model."
            )

        # For weight updates
        self._model_update_group = {}
        self._weights_send_group = {}

    def init_msprobe(self):
        # Init the msprobe
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError:
            logger.warning(
                "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
                "see https://gitcode.com/Ascend/msprobe for details."
            )
            return
        seed_all(mode=True)
        self.msprobe_debugger = PrecisionDebugger(
            config_path=self.server_args.msprobe_dump_config
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self, pre_model_load_memory: float):
        server_args = self.server_args

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        if self.server_args.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)
        # Load the model
        self.sampler = create_sampler()
        self.load_model()
        self._prepare_moe_topk()

        # Load the expert backup client
        self.expert_backup_client = (
            ExpertBackupClient(self.server_args, self)
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and self.server_args.remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )
            self._register_to_engine_info_bootstrap()

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        model_has_mtp_layers = self.model_config.num_nextn_predict_layers is not None
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        self.adjust_hybrid_swa_layers_for_pp()

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()
            if not server_args.disable_cuda_graph:
                # Phase 1 of LoRA CUDA graph init: pre-allocate large MoE
                # intermediate buffers before init_memory_pool() so memory
                # profiling accounts for them.  Phase 2 (dense LoRA batch
                # metadata) is handled in CudaGraphRunner.__init__() via
                # lora_manager.init_cuda_graph_batch_info().
                self._init_lora_cuda_graph_moe_buffers()

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        self.configure_kv_cache_dtype()

        # Init memory pool and attention backends
        self.init_memory_pool(pre_model_load_memory)

        # Init ngram embedding token table
        self.maybe_init_ngram_embedding()

        # Init routed experts capturer
        self.init_routed_experts_capturer()

        self.init_indexer_capturer()

        # TODO: Refactor device-specific init branches into platform interface (separate PR).
        # Must be called BEFORE init_device_graphs() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        self.init_aux_hidden_state_capture()

        if self.device == "cuda" or self.device == "musa":
            self.init_cublas()
            if self.enable_hisparse:
                from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                hisparse_cfg = parse_hisparse_config(self.server_args)
                hisparse_top_k = getattr(
                    self.model_config.hf_text_config, "index_topk", hisparse_cfg.top_k
                )
                self.hisparse_coordinator = HiSparseCoordinator(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    top_k=hisparse_top_k,
                    device_buffer_size=hisparse_cfg.device_buffer_size,
                    device=self.device,
                    tp_group=(
                        self.attention_tp_group.cpu_group
                        if self.server_args.enable_dp_attention
                        else self.tp_group.cpu_group
                    ),
                    host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                )
            self.init_attention_backend()
            self.kernel_warmup()
            self._pre_initialize_flashinfer_allreduce_workspace()
            self.init_device_graphs()
        elif self.device == "cpu":
            self.init_attention_backend()
            self.init_device_graphs()
        elif self.device == "npu":
            self.init_attention_backend()
            # lazy init for zbal with mix mode(before graph capture when enable_cuda_graph)
            if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 and not self.is_draft_worker:
                from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

                lazy_init_zbal_gva_mem(
                    self.device,
                    self.gpu_id,
                    get_world_group().rank_in_group,
                    get_world_group().world_size,
                    get_world_group().cpu_group,
                )
            self.init_device_graphs()
        elif current_platform.is_out_of_tree():
            self.init_attention_backend()
            if current_platform.support_cuda_graph():
                self.init_device_graphs()
            else:
                self.graph_runner = None
                self.graph_mem_usage = 0
        else:
            self.graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        if server_args.forward_hooks:
            register_forward_hooks(self.model, server_args.forward_hooks)

        # Initialize piecewise CUDA graph
        self.init_piecewise_cuda_graphs()

        self.prealloc_symmetric_memory_pool()

    def adjust_hybrid_swa_layers_for_pp(self):
        if not self.is_hybrid_swa:
            return

        if self.model_config.is_deepseek_v4_arch:
            return

        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

    def init_routed_experts_capturer(self):
        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_global_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        enable = get_global_server_args().enable_return_indexer_topk
        # Producer wiring is CUDA-only (Indexer.forward_cuda + MLA skip_topk
        # path); other backends would create a capturer but never feed it.
        if enable and self.device != "cuda":
            logger.warning(
                "indexer-topk capture is CUDA-only; %s backend not yet wired. "
                "Disabling capturer.",
                self.device,
            )
            set_global_indexer_capturer(None)
            return

        hf_text_config = self.model_config.hf_text_config
        num_indexer_layers = get_num_indexer_layers(hf_text_config)
        index_topk = getattr(hf_text_config, "index_topk", 0)
        set_global_indexer_capturer(
            create_indexer_capturer(
                enable=enable,
                num_indexer_layers=num_indexer_layers,
                index_topk=index_topk,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_aux_hidden_state_capture(self):
        """Configure auxiliary hidden state capture for speculative decoding.

        Must be called before CUDA graph capture so the captured graphs
        include aux hidden state output paths.
        """
        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )
        if self.dflash_use_aux_hidden_state:
            if not hasattr(self.model, "set_dflash_layers_to_capture"):
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not implement "
                    "set_dflash_layers_to_capture, which is required for DFLASH."
                )
            self.model.set_dflash_layers_to_capture(self.dflash_target_layer_ids)
        if self.dspark_use_aux_hidden_state:
            if not hasattr(self.model, "set_dspark_layers_to_capture"):
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not implement "
                    "set_dspark_layers_to_capture, which is required for D-Spark."
                )
            self.model.set_dspark_layers_to_capture(self.dspark_target_layer_ids)

    def remote_instance_init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip, "P2PHANDSHAKE", "rdma", envs.MOONCAKE_DEVICE.get()
        )
        self.remote_instance_transfer_engine_session_id = NetworkAddress(
            local_ip, self.remote_instance_transfer_engine.get_rpc_port()
        ).to_host_port_str()

    def _register_to_engine_info_bootstrap(self):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        import requests as http_requests

        if self.server_args.dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = self.server_args.engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        payload = {
            "tp_rank": self.tp_rank,
            "transfer_engine_info": {
                "session_id": self.remote_instance_transfer_engine_session_id,
                "weights_info_dict": self.remote_instance_transfer_engine_weight_info,
            },
        }

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )

    def model_specific_adjustment(self):
        server_args = self.server_args

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    def check_quantized_moe_compatibility(self):
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size // self.moe_dp_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (
                not envs.SGLANG_SHARED_EXPERT_TP1.get()
                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
                and not _use_aiter
            ):
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        backend = get_default_distributed_backend(self.device)
        if self.device == "cuda" and self.server_args.elastic_ep_backend == "mooncake":
            backend = "mooncake"
            if self.server_args.mooncake_ib_device:
                mooncake_ib_device = self.server_args.mooncake_ib_device.split(",")
                try:
                    from mooncake import ep as mooncake_ep

                    mooncake_ep.set_device_filter(mooncake_ib_device)
                except:
                    pass  # A warning will be raised in `init_distributed_environment`

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            na = NetworkAddress.parse(self.server_args.dist_init_addr)
            dist_init_method = na.to_tcp()
        else:
            dist_init_method = NetworkAddress(
                self.server_args.host or "127.0.0.1", self.dist_port
            ).to_tcp()
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                    @torch.library.register_fake("sgl_kernel::shm_allgather")
                    def _(data, dim):
                        return torch.cat([data] * self.tp_size, dim=dim)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
                moe_a2a_backend=self.server_args.moe_a2a_backend,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                attention_data_parallel_size=self.dp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                attention_context_model_parallel_size=self.attn_cp_size,
                moe_data_model_parallel_size=self.moe_dp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                enable_symm_mem=self.server_args.enable_symm_mem,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

            # Pre-warm NCCL/RCCL to eliminate cold-start latency in first request
            # Controlled by --pre-warm-nccl flag (default: enabled on AMD GPUs)
            if self.server_args.pre_warm_nccl and (
                self.tp_size > 1 or self.pp_size > 1 or self.moe_ep_size > 1
            ):
                warmup_start = time.perf_counter()
                tp_group_handle = get_tp_group().device_group

                # Single warmup all_reduce to initialize NCCL/RCCL communicator
                warmup_tensor = torch.zeros(1, device=torch.cuda.current_device())
                dist.all_reduce(warmup_tensor, group=tp_group_handle)
                current_platform.synchronize()

                warmup_elapsed = time.perf_counter() - warmup_start
                logger.info(
                    f"NCCL/RCCL warmup completed in {warmup_elapsed:.3f}s "
                    f"(tp_size={self.tp_size}, pp_size={self.pp_size}, ep_size={self.moe_ep_size})"
                )

        pre_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if pre_model_load_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{pre_model_load_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return pre_model_load_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
                and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_transfer_engine_session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                instance_ip = NetworkAddress.resolve_host(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                self.remote_instance_transfer_engine_weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        # Cache needs to be cleared after loading model weights (in the self.loader.load_model function).
        # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
        if _is_npu:
            torch.npu.empty_cache()
        monkey_patch_vllm_parallel_state(reverse=True)

        if not self.is_draft_worker:
            get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )
        if self.server_args.debug_tensor_dump_output_folder is not None:
            dump_folder = self.server_args.debug_tensor_dump_output_folder
            if self.spec_algorithm.is_eagle():
                role = "draft" if self.is_draft_worker else "target"
                dump_folder = os.path.join(dump_folder, role)
            register_forward_hook_for_model(
                self.model,
                dump_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def _prepare_moe_topk(self):
        balancer_cls = None
        num_prepared = 0
        num_routed_experts = None
        for module in self.model.modules():
            if not isinstance(module, (TopK, HashTopK)):
                continue
            if (
                not module.enable_deepep_waterfill
                or module.deepep_waterfill_balancer is not None
            ):
                continue
            if num_routed_experts is None:
                num_routed_experts = getattr(
                    self.model_config.hf_config, "n_routed_experts", None
                )
                if num_routed_experts is None:
                    raise ValueError(
                        "DeepEP waterfill requires model config n_routed_experts."
                    )
            if balancer_cls is None:
                from sglang.srt.layers.moe.deepep_waterfill import (
                    DeepEPWaterfillBalancer,
                )

                balancer_cls = DeepEPWaterfillBalancer
            # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
            # Redundant experts therefore need to be included in the per-rank
            # expert count used for Waterfill's shared-expert slot remapping.
            num_physical_routed_experts = (
                num_routed_experts + self.server_args.ep_num_redundant_experts
            )
            if isinstance(module, TopK):
                routed_scaling_factor = module.topk_config.routed_scaling_factor
            else:
                routed_scaling_factor = module.routed_scaling_factor
            module.deepep_waterfill_balancer = balancer_cls(
                num_routed_experts=num_physical_routed_experts,
                world_size=self.moe_ep_size,
                rank=self.moe_ep_rank,
                layer_id=module.layer_id,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
            )
            num_prepared += 1
        if num_prepared:
            log_info_on_rank0(
                logger, f"Prepared {num_prepared} DeepEP waterfill TopK modules."
            )

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        p2p_missing_logical_experts = self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

        if len(p2p_missing_logical_experts) > 0:
            # Load the missing expert weights from disk
            if callable(getattr(self.model, "generate_weight_name_filter", None)):
                # Filter and load only missing expert weights
                weight_name_filter = self.model.generate_weight_name_filter(
                    p2p_missing_logical_experts
                )
            else:
                # Do a full reload from disk/DRAM
                logger.info(
                    "[Elastic EP] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            if (
                self.expert_backup_client is not None
                and self.expert_backup_client.use_backup
            ):
                # Load the missing weights from the DRAM backup
                self.expert_backup_client.update_weights(weight_name_filter)
            else:
                # Load the missing weights from disk
                self.update_weights_from_disk(
                    get_global_server_args().model_path,
                    get_global_server_args().load_format,
                    weight_name_filter=weight_name_filter,
                )

    def maybe_recover_ep_ranks(self):
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0
            self.eplb_manager.reset_generator()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.init_device_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert torch.distributed.is_initialized(), (
            "Default torch process group must be initialized"
        )
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert len(ports_list) == self.tp_size, (
            f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        )
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        current_platform.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        current_platform.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert torch.distributed.is_initialized(), (
            "Default torch process group must be initialized"
        )
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert len(ports_list) == self.tp_size, (
            f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        )
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        current_platform.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        current_platform.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert torch.distributed.is_initialized(), (
            "Default torch process group must be initialized"
        )
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def _init_lora_cuda_graph_moe_buffers(self):
        """Phase 1 of LoRA CUDA graph init: pre-allocate MoE intermediate buffers.

        Must be called before init_memory_pool() so that memory profiling
        sees the reduced available memory and sizes KV cache correctly.
        All MoE LoRA layers share one set of buffers (managed by the
        lora_backend) since they execute sequentially during forward.

        Phase 2 (dense LoRA batch metadata) is handled later in
        CudaGraphRunner.__init__() via lora_manager.init_cuda_graph_batch_info(),
        because it needs capture-time parameters (max_bs, num_tokens_per_bs)
        that are only available at that stage.
        """
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        max_bs = self.server_args.cuda_graph_max_bs
        max_loras = self.server_args.max_loras_per_batch
        for module in self.model.modules():
            if isinstance(module, FusedMoEWithLoRA):
                self.lora_manager.init_cuda_graph_moe_buffers(
                    max_bs, max_loras, self.dtype, module
                )
                logger.info(
                    f"Pre-allocated shared MoE LoRA CUDA graph buffers "
                    f"(max_bs={max_bs}, max_loras={max_loras})"
                )
                break

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    def qwen3_next_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    def hybrid_lightning_config(self):
        config = self.model_config.hf_config
        if isinstance(config, BailingHybridConfig):
            return config
        return None

    @property
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | InternS2PreviewConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config,
            FalconH1Config
            | NemotronHConfig
            | Lfm2Config
            | Lfm2MoeConfig
            | Lfm2VlConfig,
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config

        if isinstance(config, GraniteMoeHybridConfig):
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(config, "layer_types", [])
            )
            if not has_mamba:
                return None
            else:
                return config

        return None

    @property
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            return self.full_max_total_num_tokens
        else:
            return self.max_total_num_tokens

    @property
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    def _get_linear_attn_registry_result(self):
        if self._linear_attn_registry_cache is _UNSET:
            self._linear_attn_registry_cache = get_linear_attn_config(
                self.model_config.hf_config
            )
        return self._linear_attn_registry_cache

    @property
    def linear_attn_model_spec(self):
        result = self._get_linear_attn_registry_result()
        return result[0] if result else None

    @property
    def mambaish_config(self):
        existing = (
            self.mamba2_config
            or self.hybrid_gdn_config
            or self.kimi_linear_config
            or self.hybrid_lightning_config
        )
        if existing:
            return existing
        result = self._get_linear_attn_registry_result()
        return result[1] if result else None

    def configure_kv_cache_dtype(self):
        if self.server_args.kv_cache_dtype == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    self.kv_cache_dtype = fp8_dtype
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
                else:
                    self.kv_cache_dtype = torch.float8_e4m3fn
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

        # ── RedKnot: bind rotary_emb so it can RoPE-realign offline KV.
        # Best-effort, optional. We probe attribute names common across
        # Llama / Mistral / Qwen2 / Qwen3 model implementations in sglang.
        try:
            from sglang.srt.layers.attention.redknot_backend import (
                RedKnotAttnBackend,
            )

            backend = getattr(self.attn_backend, "full_attn_backend", self.attn_backend)
            if (
                isinstance(backend, RedKnotAttnBackend)
                and getattr(self, "model", None) is not None
            ):
                base = getattr(self.model, "model", self.model)
                layers = getattr(base, "layers", None)
                if layers:
                    rotary_emb = next(
                        (
                            rotary
                            for layer in layers
                            for rotary in [
                                getattr(layer, "rotary_emb", None),
                                getattr(
                                    getattr(layer, "self_attn", None),
                                    "rotary_emb",
                                    None,
                                ),
                            ]
                            if rotary is not None
                        ),
                        None,
                    )
                    if rotary_emb is not None:
                        backend.attach_rope_helper(rotary_emb)
                        logger.info(
                            "RedKnot: bound rotary_emb (%s) to attention backend.",
                            type(rotary_emb).__name__,
                        )
        except Exception as _redknot_bind_exc:  # pragma: no cover
            logger.warning(
                "RedKnot: rotary_emb auto-bind failed: %s",
                _redknot_bind_exc,
            )

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def kernel_warmup(self):
        """
        Warmup and tune kernels before cuda graph capture.
        Covers framework-level warmups and optional model-specific warmups.
        """
        if self.device != "cuda":
            return

        if self._should_run_flashinfer_autotune():
            self._flashinfer_autotune()

        # Models may need their own warmup for model-specific kernels or JIT paths.
        # Register those hooks on the model class so ModelRunner can keep this
        # warmup entry point generic.
        model_kernel_warmup = getattr(self.model, "kernel_warmup", None)
        if model_kernel_warmup is not None:
            model_kernel_warmup(self)

    def _pre_initialize_flashinfer_allreduce_workspace(self):
        """Pre-initialize flashinfer allreduce fusion workspaces.

        Must run before CUDA graph capture to avoid collective operations
        (broadcasts, barriers) inside the graph capture context, which can
        deadlock with custom_all_reduce.register_graph_buffers.
        """
        if not self.server_args.enable_flashinfer_allreduce_fusion:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import (
            pre_initialize_workspaces,
        )

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=self.model_config.hidden_size,
            dtype=self.dtype,
        )

    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run."""
        if self.server_args.disable_flashinfer_autotune:
            return False

        # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
        # be autotuned -- its _dummy_run would dispatch more tokens per rank than
        # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
        # Read server_args directly to avoid depending on initialize_moe_config()
        # having already populated the MoE backend globals.
        if (
            self.server_args.moe_runner_backend == "flashinfer_cutedsl"
            and self.server_args.moe_a2a_backend == "deepep"
        ):
            return False

        backend_str = self.server_args.moe_runner_backend

        # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

        if backend_str not in [
            "flashinfer_trtllm",
            # TODO: Enable for flashinfer_trtllm_routed once https://github.com/flashinfer-ai/flashinfer/issues/2749 is fixed.
            # "flashinfer_trtllm_routed",
            "flashinfer_mxfp4",
            "flashinfer_cutedsl",
            # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
            # "flashinfer_cutlass",
        ]:
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False

        if self.spec_algorithm.is_speculative():
            return not self.is_draft_worker

        return True

    def _flashinfer_autotune(self):
        """Run flashinfer autotune."""
        from flashinfer.autotuner import autotune

        cache_path = self._flashinfer_autotune_cache_path()
        logger.info("Running FlashInfer autotune with cache: %s", cache_path)

        # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
        # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.get_device_module(self.device).stream(self.forward_stream):
            with torch.inference_mode(), autotune(True, cache=str(cache_path)):
                self._dummy_run(batch_size=self.req_to_token_pool.size)
        torch.cuda.current_stream().wait_stream(self.forward_stream)
        logger.info("FlashInfer autotune completed.")

    def _flashinfer_autotune_cache_path(self) -> Path:
        import flashinfer

        major, minor = torch.cuda.get_device_capability(self.device)
        arch = f"sm{major}{minor}"
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

        server_args = self.server_args
        model_key = "|".join(
            [
                str(server_args.model_path),
                str(self.dtype),
                str(server_args.quantization),
                str(server_args.moe_runner_backend),
                str(self.tp_size),
                str(self.pp_size),
                str(self.dp_size),
                str(self.moe_ep_size),
                str(self.model_config.hf_config.__class__.__name__),
            ]
        )
        cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
        cache_dir = (
            Path(envs.SGLANG_CACHE_DIR.get())
            / "flashinfer"
            / "autotune"
            / flashinfer_version
            / arch
            / cache_key
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir
            / f"rank_tp{self.tp_rank}_pp{self.pp_rank}_dp{self.dp_rank or 0}.json"
        )

    def _dummy_run(self, batch_size: int, run_ctx=None):
        """Run a dummy forward pass for warmup/profiling."""
        if self.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if self.spec_algorithm.is_speculative():
            if self.is_draft_worker:
                if not self.spec_algorithm.supports_target_verify_for_draft():
                    raise RuntimeError("This should not happen")
            capture_forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens_per_bs = (
                self.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                    self.server_args.speculative_num_draft_tokens, self.is_draft_worker
                )
            )

        if self.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        # Keep warmup aligned with scheduler MLP-sync padding.
        if require_mlp_sync(self.server_args):
            attn_tp_size = get_attention_tp_size()
            if attn_tp_size > 1 and num_tokens % attn_tp_size != 0:
                num_tokens = ceil_align(num_tokens, attn_tp_size)
                batch_size = num_tokens // num_tokens_per_bs

        seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        if self.server_args.enable_torch_compile:
            set_torch_compile_config()
            should_disable_torch_compile = not getattr(
                self.model, "_can_torch_compile", True
            )
            if should_disable_torch_compile:
                log_info_on_rank0(
                    logger,
                    "Transformers backend model reports it is not torch.compile "
                    "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
                )
                self.server_args.enable_torch_compile = False

        # NOTE: aux hidden state capture (eagle3/dflash) is already
        # configured by init_aux_hidden_state_capture() in initialize().

        require_mlp_tp_gather_ = require_mlp_tp_gather(self.server_args)
        if require_gathered_buffer(self.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(self.server_args)

        buffers: DecodeInputBuffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=batch_size,
            max_num_token=num_tokens,
            hidden_size=self.model_config.hidden_size,
            vocab_size=self.model_config.vocab_size,
            dtype=self.model_config.dtype,
            dp_size=self.server_args.dp_size,
            pp_size=self.server_args.pp_size,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather_,
            seq_len_fill_value=seq_len_fill_value,
            encoder_len_fill_value=(
                getattr(self.model_config.hf_config, "max_source_positions", 0)
                if self.model_config.is_encoder_decoder
                else 0
            ),
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
        )
        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        if not self.is_generation:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=self.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=self.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=self.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if self.server_args.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.server_args.dp_size
        elif require_attn_tp_gather(self.server_args):
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        def get_spec_info():
            spec_info = None
            if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
                from sglang.srt.speculative.eagle_info import EagleVerifyInput

                if self.is_draft_worker:
                    raise RuntimeError("This should not happen.")
                else:
                    spec_info = EagleVerifyInput(
                        draft_token=None,
                        custom_mask=buffers.custom_mask,
                        positions=None,
                        retrieve_index=None,
                        retrieve_next_token=None,
                        retrieve_next_sibling=None,
                        retrieve_cum_len=None,
                        spec_steps=self.server_args.speculative_num_steps,
                        topk=self.server_args.speculative_eagle_topk,
                        draft_token_num=self.server_args.speculative_num_draft_tokens,
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        seq_lens_sum=None,
                        seq_lens_cpu=None,
                    )
            elif self.spec_algorithm.is_dflash():
                from sglang.srt.speculative.dflash_info import DFlashVerifyInput

                # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
                spec_info = DFlashVerifyInput(
                    draft_token=None,
                    positions=None,
                    draft_token_num=self.server_args.speculative_num_draft_tokens,
                    custom_mask=None,
                    capture_hidden_mode=(
                        CaptureHiddenMode.NULL
                        if self.is_draft_worker
                        else CaptureHiddenMode.FULL
                    ),
                )

            elif self.spec_algorithm.is_ngram():
                from sglang.srt.speculative.ngram_info import NgramVerifyInput

                spec_info = NgramVerifyInput(
                    draft_token=None,
                    tree_mask=buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    draft_token_num=num_tokens_per_bs,
                )
                spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

            return spec_info

        spec_info = get_spec_info()
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            next_token_logits_buffer=buffers.next_token_logits_buffer,
            orig_seq_lens=buffers.seq_lens,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_sum=buffers.seq_lens.sum().item(),
            encoder_lens=buffers.encoder_lens,
            return_logprob=False,
            positions=buffers.positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=buffers.mrope_positions,
            spec_algorithm=self.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            self.lora_manager.prepare_lora_batch(forward_batch)

        self.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.server_args.pp_size > 1
                and "pp_proxy_tensors"
                in inspect.signature(self.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not self.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = self.model.forward(
                buffers.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        with forward_context(ForwardContext(attn_backend=self.attn_backend)):
            with torch.inference_mode(), run_ctx or empty_context():
                run_once()

    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            self.token_table = torch.empty(
                self.req_to_token_pool.req_to_token.shape[0],
                self.model_config.context_len,
                dtype=torch.int32,
                device=self.device,
            )
            chunked_prefill_size = self.server_args.chunked_prefill_size
            assert chunked_prefill_size is not None and chunked_prefill_size > 0, (
                "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            )
            for module in self.model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        self.max_running_requests, chunked_prefill_size, self.device
                    )

    def maybe_update_ngram_token_table(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """Update the ngram embedding token table after sampling."""
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        ngram_embedding_info.out_column_starts[: forward_batch.batch_size] = (
            forward_batch.seq_lens
        )
        ngram_embedding_info.out_req_lens[: forward_batch.batch_size] = 1
        update_token_table(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids.to(torch.int32),
            row_indices=forward_batch.req_pool_indices,
            column_starts=ngram_embedding_info.out_column_starts,
            req_lens=torch.ones_like(ngram_embedding_info.out_column_starts),
            ignore_tokens=None,
        )

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and self.server_args.disable_cuda_graph:
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: f"{current_platform.device_name} graph",
            {
                "cuda": "cuda graph",
                "musa": "cuda graph",
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        logger.info(
            f"Capture {graph_backend[self.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        if current_platform.is_out_of_tree():
            GraphRunnerCls = current_platform.get_graph_runner_cls()
            self.graph_runner = GraphRunnerCls(self)
        else:
            graph_runners = defaultdict(
                lambda: CudaGraphRunner,
                {
                    "cpu": CPUGraphRunner,
                    "npu": NPUGraphRunner,
                },
            )
            self.graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {graph_backend[self.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_piecewise_cuda_graphs(self, force_for_draft_worker: bool = False):
        """Initialize piecewise CUDA graph runner."""
        self.piecewise_cuda_graph_runner = None

        if self.server_args.disable_piecewise_cuda_graph:
            logger.info(
                "Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set"
            )
            return

        # Draft models skip here during __init__; the eagle worker calls
        # this method explicitly (force_for_draft_worker=True) after
        # init_lm_head so graphs capture the final embedding weights.
        if self.is_draft_worker and not force_for_draft_worker:
            return

        # Disable piecewise CUDA graph for non-language models
        if not hasattr(self.model, "model"):
            logger.warning(
                "Disable piecewise CUDA graph because the model is not a language model"
            )
            return

        # Disable piecewise CUDA graph for non capture size
        if not self.server_args.piecewise_cuda_graph_tokens:
            logger.warning(
                "Disable piecewise CUDA graph because the capture size is not set"
            )
            return

        # Collect attention layers and moe layers from the model
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)

        # Resolve model with layers: handle CausalLM wrapper (.model.layers) and direct TextModel (.layers)
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            layer_model = language_model.model
        elif hasattr(language_model, "layers"):
            layer_model = language_model
        else:
            logger.warning(
                "Disable piecewise CUDA graph because the model does not have a 'layers' attribute"
            )
            return

        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        self.dsa_indexers = []
        for layer in layer_model.layers:
            attn_layer = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_layer = layer.self_attn.attn
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    attn_layer = layer.self_attn.attn_mqa
            # For hybrid model
            elif hasattr(layer, "attn"):
                attn_layer = layer.attn
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    attn_layer = layer.linear_attn.attn
                else:
                    attn_layer = layer.linear_attn
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attn_layer = layer.attention.attn
            # For NemotronH and similar hybrid models using 'mixer' attribute
            elif hasattr(layer, "mixer"):
                if hasattr(layer.mixer, "attn"):
                    attn_layer = layer.mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    # Mamba layer with split op support - store the layer itself
                    attn_layer = layer

            if attn_layer is not None:
                self.attention_layers.append(attn_layer)
            elif hasattr(layer, "mixer"):
                self.attention_layers.append(None)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            # For NemotronH MoE layers using 'mixer' attribute
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
                moe_block = layer.mixer.experts
                moe_fusion = layer.mixer
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)
            # NSA indexers (None for layers without NSA)
            dsa_indexer = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "indexer"):
                dsa_indexer = layer.self_attn.indexer
            self.dsa_indexers.append(dsa_indexer)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        if self.server_args.enable_breakable_cuda_graph:
            # Experimental feature
            self.piecewise_cuda_graph_runner = BreakableCudaGraphRunner(self)
        else:
            self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # Set extra arguments
        pdmux_override = False
        if not skip_attn_backend_init:
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            if self.server_args.enable_pdmux:
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                # PDmux selects a per-stream backend; publish it to model-layer
                # readers via the active ForwardContext so RadixAttention etc.
                # dispatch against the right backend for this forward.
                pdmux_override = True
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors

        # Launch forward
        ctx = (
            self.device_timer.wrap(metadata={"category": "decode"})
            if self.device_timer
            else contextlib.nullcontext()
        )

        def _do_forward():
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        with ctx:
            if pdmux_override:
                with forward_context(
                    ForwardContext(attn_backend=self.decode_attn_backend)
                ):
                    return _do_forward()
            return _do_forward()

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        self._apply_redknot_selected_rows(forward_batch)
        self._prepare_redknot_v4_boundary_replay(forward_batch)

        # REDKNOT_V4_TIMING is a restore-only CUDA-event diagnostic.  A session
        # is opened below only after graph replay has been ruled out; region
        # boundaries enqueue events without synchronizing the host.
        _rk_plans = getattr(forward_batch, "redknot_reuse_plan", None)
        _rk_p0 = _rk_plans[0] if _rk_plans and len(_rk_plans) == 1 else None
        _rk_drift_requested = bool(
            _rk_plans
            and any(
                isinstance(plan, Mapping) and plan.get("mode") == "drift_profile"
                for plan in _rk_plans
            )
        )
        if _rk_drift_requested and (
            len(_rk_plans) != 1
            or not isinstance(_rk_p0, Mapping)
            or _rk_p0.get("mode") != "drift_profile"
        ):
            raise RuntimeError(
                "MLA head-drift profiling requires one explicit request plan"
            )
        _rk_timing_requested = bool(
            os.environ.get("REDKNOT_V4_TIMING", "0") == "1"
            and _rk_p0
            and _rk_p0.get("mode") == "restore"
        )
        _rk_timing = False
        _rk_timing_abort = None
        _rk_timing_finish = None
        _rk_timing_region = None

        # Setup extra arguments
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True

        # Check piecewies cuda graph
        can_run_graph = (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        )
        if can_run_graph:
            if _rk_drift_requested:
                raise RuntimeError(
                    "MLA head-drift profiling cannot run through a piecewise "
                    "CUDA graph; disable graph replay for calibration"
                )
            # TODO: device_timer.wrap is too broad here — it also includes
            # replay_prepare time. Move timing into the piecewise cuda graph
            # runner to capture only the model.forward part.
            ctx = (
                self.device_timer.wrap(metadata={"category": "extend"})
                if self.device_timer
                else contextlib.nullcontext()
            )
            with ctx:
                ret = self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs)
            return (ret, can_run_graph)

        if _rk_timing_requested:
            try:
                from sglang.srt.layers.attention.redknot.dsv4_timing import (
                    abort_forward as _rk_timing_abort,
                    begin_forward as _rk_timing_begin,
                    finish_forward as _rk_timing_finish,
                    timed as _rk_timing_region,
                )

                _rk_timing = bool(
                    _rk_timing_begin(
                        device=forward_batch.input_ids.device,
                        rows=int(forward_batch.input_ids.shape[0]),
                        forward_mode=str(forward_batch.forward_mode),
                        batch_size=int(forward_batch.batch_size),
                    )
                )
                if not _rk_timing:
                    raise RuntimeError(
                        "REDKNOT_V4_TIMING=1 failed to open a CUDA timing session"
                    )
            except BaseException:
                if _rk_timing_abort is not None:
                    try:
                        _rk_timing_abort()
                    except BaseException:
                        pass
                raise

        # Install the projected-head capture only for the explicit calibration
        # request.  Validation rejects chunked/cache-hit prefills, FP8 wo_a,
        # non-TP8 topologies, and any batch other than one full EXTEND before a
        # model layer can observe active state.
        _rk_drift_session = None
        if _rk_drift_requested:
            from sglang.srt.layers.attention.redknot.mla_head_drift_runtime import (
                begin_drift_profile_request,
            )

            _rk_seq_lens = getattr(forward_batch, "orig_seq_lens", None)
            if not torch.is_tensor(_rk_seq_lens) or _rk_seq_lens.numel() != 1:
                _rk_seq_lens = forward_batch.seq_lens_cpu
            if not torch.is_tensor(_rk_seq_lens) or _rk_seq_lens.numel() != 1:
                raise RuntimeError(
                    "MLA head-drift profiling requires one logical sequence length"
                )
            _rk_drift_session = begin_drift_profile_request(
                _rk_p0,
                positions=forward_batch.positions,
                input_ids=forward_batch.input_ids,
                logical_seq_len=int(_rk_seq_lens[0].item()),
                batch_size=forward_batch.batch_size,
                is_extend=forward_batch.forward_mode == ForwardMode.EXTEND,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                num_layers=self.model_config.num_hidden_layers,
                fp8_wo_a=bool(envs.SGLANG_OPT_FP8_WO_A_GEMM.get()),
                attention_backend=self.prefill_attention_backend_str,
                sparse_ffn=bool(
                    getattr(self.server_args, "redknot_sparse_ffn_enable", False)
                ),
            )
            forward_batch._redknot_mla_head_drift_active = True

        try:
            # Launch model forward
            if not skip_attn_backend_init:
                if hasattr(self.model, "prepare_forward_batch"):
                    # Prepare model-specific attention metadata before planning,
                    # e.g. Moss-VL's prefill cross-attention custom mask.
                    self.model.prepare_forward_batch(forward_batch)
                if _rk_timing:
                    with _rk_timing_region("init_forward_metadata"):
                        self.attn_backend.init_forward_metadata(forward_batch)
                else:
                    self.attn_backend.init_forward_metadata(forward_batch)

            ctx = (
                self.device_timer.wrap(metadata={"category": "extend"})
                if self.device_timer
                else contextlib.nullcontext()
            )
            if _rk_timing:
                with ctx, _rk_timing_region("model_forward"):
                    ret = self.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
            else:
                with ctx:
                    ret = self.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
            if _rk_timing:
                _rk_timing_finish()
        except BaseException:
            if _rk_timing and _rk_timing_abort is not None:
                try:
                    _rk_timing_abort()
                except BaseException:
                    pass
            if _rk_drift_session is not None:
                from sglang.srt.layers.attention.redknot.mla_head_drift_runtime import (
                    abort_drift_profile_request,
                )

                abort_drift_profile_request(_rk_drift_session)
            raise
        else:
            if _rk_drift_session is not None:
                from sglang.srt.layers.attention.redknot.mla_head_drift_runtime import (
                    finish_drift_profile_request,
                )

                finish_drift_profile_request(_rk_drift_session)
        finally:
            if _rk_drift_session is not None:
                forward_batch._redknot_mla_head_drift_active = False
        return (ret, can_run_graph)

    def _apply_redknot_selected_rows(self, forward_batch: ForwardBatch) -> None:
        """Keep only boundary/query rows while preserving full logical KV slots.

        This experimental path is intentionally limited to a single prefill
        request. The scheduler has already allocated slots for the complete
        logical sequence; skipped rows are restored per layer by the DSV4
        offline-reuse hook.
        """
        plans = getattr(forward_batch, "redknot_reuse_plan", None)
        if not plans or len(plans) != 1:
            return
        plan = plans[0]
        if not plan or plan.get("mode") != "restore" or not plan.get("skip_forward"):
            return
        if forward_batch.batch_size != 1 or forward_batch.forward_mode not in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
        ):
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: only one EXTEND/MIXED "
                    "request is supported"
                )
            forward_batch.redknot_reuse_plan = [None] * forward_batch.batch_size
            return

        # The immutable document-1 radix seed is produced outside all timed
        # observations.  It must materialize a complete transformer/KV prefix,
        # not a selected-row shell whose scheduler length happens to be 8K.
        # Run this one producer microforward through the ordinary full-row
        # model path; consumers still carry the combined plan and physically
        # prune documents 2..N.  The exact cached token ids are reauthenticated
        # by the combined restore validator before any consumer omission.
        if plan.get("radix_prefix_role") == "seed":
            combined_profiles = (
                "combined_headsplit_independent_rope_zoff_checkpoint_"
                "rowsparse_3_37_3_v1",
                "combined_headsplit_pro0813_independent_rope_zoff_checkpoint_"
                "rowsparse_3_55_3_v1",
                "combined_headsplit_pro0813_independent_rope_full_checkpoint_"
                "rowsparse_3_55_3_v1",
            )
            if str(plan.get("mla_off_execution_profile", "")) not in combined_profiles:
                raise ValueError(
                    "selected-row radix seed received a foreign execution profile"
                )
            from sglang.srt.layers.attention.redknot.dsv4_context_identity import (
                token_ids_sha256,
            )

            prefix_tokens = plan.get("radix_prefix_tokens")
            prefix_hash = plan.get("radix_prefix_input_hash")
            declared_total = plan.get("total_tokens")
            positions = tuple(
                int(value)
                for value in forward_batch.positions.detach()
                .to(device="cpu", dtype=torch.long)
                .tolist()
            )
            input_ids = tuple(
                int(value)
                for value in forward_batch.input_ids.detach()
                .to(device="cpu", dtype=torch.long)
                .tolist()
            )
            if (
                type(prefix_tokens) is not int
                or prefix_tokens <= 0
                or type(prefix_hash) is not str
                or type(declared_total) is not int
                or declared_total <= prefix_tokens
                or not positions
                or len(positions) != len(input_ids)
                or any(
                    right != left + 1
                    for left, right in zip(positions, positions[1:])
                )
            ):
                raise ValueError("selected-row radix seed geometry is invalid")
            start = positions[0]
            end = positions[-1] + 1
            if start == 0:
                if (
                    end != prefix_tokens
                    or token_ids_sha256(input_ids) != prefix_hash
                ):
                    raise ValueError(
                        "selected-row radix seed does not contain the exact "
                        "complete first document"
                    )
            elif start != prefix_tokens or end > declared_total:
                raise ValueError(
                    "selected-row radix seed suffix has an invalid extent"
                )
            forward_batch._redknot_combined_radix_dense_seed = True
            forward_batch.redknot_reuse_plan = [None]
            return

        from sglang.srt.layers.attention.redknot.dsv4_offline_reuse_v2 import (
            get_offline_reuse_controller_v2,
        )
        from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
            deepseek_v4_redknot_topology,
        )
        from sglang.srt.layers.attention.redknot.v4.config import RedKnotV4Config
        from sglang.srt.layers.attention.redknot.v4.request_selector import (
            MAX_CHECKPOINT_ISLANDS,
            checkpoint_effective_segment_cap_tokens,
            checkpoint_mandatory_prefix_tokens,
        )
        from sglang.srt.layers.attention.redknot.v4.reuse_planner import (
            validate_runtime_reuse_plan,
        )

        try:
            runtime_config = RedKnotV4Config(
                mode=os.environ.get("REDKNOT_V4_MODE", "correctness"),
                reuse_window_kv=bool(plan.get("reuse_window_kv", False)),
            )
        except ValueError as error:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: invalid runtime config: %s",
                    error,
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        validation = validate_runtime_reuse_plan(
            plan,
            config=runtime_config,
            dspark_active=self.spec_algorithm.is_speculative(),
        )
        if not validation.valid:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: reason=%s detail=%s",
                    validation.fallback_reason.value,
                    validation.detail,
                )
            # Disabling the whole plan is essential: merely returning here would
            # leave the backend restore hook active after rows were deemed unsafe.
            forward_batch.redknot_reuse_plan = [None]
            return

        try:
            query_start = int(plan["query_start"])
            positions = forward_batch.positions.to(torch.int64)
            if positions.numel() == 0 or not bool(
                torch.all(positions[1:] > positions[:-1]).item()
            ):
                raise ValueError("positions must be non-empty and strictly increasing")
            logical_total = int(
                plan.get(
                    "total_tokens",
                    max(query_start, int(forward_batch.seq_lens_cpu[0].item())),
                )
            )
            if query_start < 0 or query_start > logical_total:
                raise ValueError(
                    f"query_start={query_start} is outside total_tokens={logical_total}"
                )
            orig_seq_lens = getattr(forward_batch, "orig_seq_lens", None)
            if orig_seq_lens is not None and orig_seq_lens.numel() == 1:
                actual_total = int(orig_seq_lens[0].item())
                if logical_total != actual_total:
                    raise ValueError(
                        f"total_tokens={logical_total} differs from request={actual_total}"
                    )
            segments = list(plan.get("segments", ()))
            if not segments:
                raise ValueError("restore plan has no segments")
            selection_policy = str(plan.get("selection_policy", "legacy"))
            cap_ratio = float(plan.get("hot_max_per_segment_ratio", 0.75))
            active_budget_ratio = float(
                plan.get("active_token_budget_ratio", 0.10)
            )
            checkpoint_stride = int(
                plan.get("checkpoint_stride_tokens", 0) or 0
            )
            checkpoint_max_islands = int(plan.get("checkpoint_max_islands", 8))
            query_protection_policy = str(
                plan.get("query_protection_policy", "none")
            )
            query_protected_segment_index = plan.get(
                "query_protected_segment_index", -1
            )
            raw_query_protected_ranges = plan.get("query_protected_ranges", [])
            if not math.isfinite(cap_ratio) or not 0.0 < cap_ratio <= 1.0:
                raise ValueError("hot_max_per_segment_ratio must be in (0, 1]")
            if checkpoint_stride and (
                checkpoint_stride < 512 or checkpoint_stride % 512 != 0
            ):
                raise ValueError(
                    "checkpoint_stride_tokens must be a positive multiple of 512"
                )
            if not 0 < checkpoint_max_islands <= MAX_CHECKPOINT_ISLANDS:
                raise ValueError("checkpoint_max_islands must be in [1, 256]")
            if (
                checkpoint_max_islands > 64
                and os.environ.get("REDKNOT_DSV4_VARIANT", "").strip().lower()
                != "pro0813"
            ):
                raise ValueError(
                    "checkpoint_max_islands above the legacy 64-island cap "
                    "requires REDKNOT_DSV4_VARIANT=pro0813"
                )
            expected_offset = 0
            for segment_index, segment in enumerate(segments):
                offset = int(segment["global_offset"])
                length = int(segment["length"])
                boundary = int(segment.get("skip_first", 128))
                canonical = int(segment.get("canonical_start_pos", 0))
                query_score = float(segment.get("query_score", 1.0))
                if offset != expected_offset:
                    raise ValueError(
                        f"segment {segment_index} begins at {offset}, "
                        f"expected contiguous offset {expected_offset}"
                    )
                if length <= 0 or length % 128 != 0:
                    raise ValueError(
                        f"segment {segment_index} length must be positive/128-aligned"
                    )
                if not 0 <= boundary <= length or boundary % 128 != 0:
                    raise ValueError(
                        f"segment {segment_index} boundary is invalid"
                    )
                if selection_policy == "checkpoint_islands" and boundary != 128:
                    raise ValueError(
                        "checkpoint-island replay currently requires skip_first=128"
                    )
                if canonical != 0:
                    raise ValueError(
                        f"segment {segment_index} was not snapshotted at local position 0"
                    )
                if not math.isfinite(query_score) or query_score < 0.0:
                    raise ValueError(
                        f"segment {segment_index} query_score is invalid"
                    )
                expected_offset = offset + length
            if expected_offset != query_start:
                raise ValueError(
                    f"segments cover [0,{expected_offset}), query starts at {query_start}"
                )

            def protected_segment_index_for_range(begin, end):
                containing = [
                    segment_index
                    for segment_index, segment in enumerate(segments)
                    if begin >= int(segment["global_offset"])
                    and end
                    <= int(segment["global_offset"]) + int(segment["length"])
                ]
                return containing[0] if len(containing) == 1 else None

            if type(query_protected_segment_index) is not int:
                raise ValueError("query-protected segment index is invalid")
            if query_protection_policy == "none":
                if (
                    query_protected_segment_index != -1
                    or raw_query_protected_ranges != []
                ):
                    raise ValueError(
                        "disabled query protection requires index=-1/ranges=[]"
                    )
                query_protected_ranges = []
            elif query_protection_policy in {
                "lexical_top1_full_segment_v1",
                "lexical_top1_block_windows_v1",
                "lexical_topk_block_windows_v2",
            }:
                if not 0 <= query_protected_segment_index < len(segments):
                    raise ValueError(
                        "query-protected segment is outside the restore chain"
                    )
                if (
                    not isinstance(raw_query_protected_ranges, list)
                    or not raw_query_protected_ranges
                ):
                    raise ValueError("query-protected ranges are absent")
                protected_segment = segments[query_protected_segment_index]
                segment_begin = int(protected_segment["global_offset"])
                segment_end = segment_begin + int(protected_segment["length"])
                cursor = 0
                query_protected_ranges = []
                protected_segment_indices = set()
                for item in raw_query_protected_ranges:
                    if not isinstance(item, dict) or frozenset(item) != {
                        "start",
                        "end",
                    }:
                        raise ValueError("query-protected range schema is invalid")
                    begin, end = int(item["start"]), int(item["end"])
                    containing = protected_segment_index_for_range(begin, end)
                    if (
                        containing is None
                        or begin < cursor
                        or begin >= end
                        or begin % 512 != 0
                        or end % 512 != 0
                    ):
                        raise ValueError("query-protected range geometry is invalid")
                    query_protected_ranges.append((begin, end))
                    protected_segment_indices.add(containing)
                    cursor = end
                if (
                    query_protection_policy == "lexical_top1_full_segment_v1"
                    and query_protected_ranges != [(segment_begin, segment_end)]
                ):
                    raise ValueError("full-segment query protection is incomplete")
                if (
                    query_protection_policy == "lexical_top1_block_windows_v1"
                    and protected_segment_indices
                    != {query_protected_segment_index}
                ):
                    raise ValueError(
                        "top1 query protection escaped its selected segment"
                    )
                if query_protection_policy == "lexical_topk_block_windows_v2" and (
                    query_protected_segment_index not in protected_segment_indices
                    or len(protected_segment_indices) != 2
                ):
                    raise ValueError(
                        "topk query protection must cover exactly two segments"
                    )
            else:
                raise ValueError("query protection policy is unsupported")
            if str(plan.get("radix_prefix_role", "")) == "consume":
                query_protected_ranges_online = [
                    (begin, end)
                    for begin, end in query_protected_ranges
                    if protected_segment_index_for_range(begin, end) != 0
                ]
            else:
                query_protected_ranges_online = query_protected_ranges
            protected_segment_online = (
                query_protection_policy == "lexical_top1_full_segment_v1"
                and bool(query_protected_ranges_online)
            )
            if positions[-1].item() >= logical_total:
                raise ValueError("forward positions exceed total_tokens")
            chunk_token_range = (
                int(positions[0].item()),
                int(positions[-1].item()) + 1,
            )
            if (
                chunk_token_range[0] % 128 != 0
                or (
                    chunk_token_range[1] != logical_total
                    and chunk_token_range[1] % 128 != 0
                )
            ):
                raise ValueError(
                    "selected-row scheduler chunk boundaries must be 128-aligned"
                )
            crosses_segment_boundary = (
                runtime_config.mode != "correctness"
                and any(
                    chunk_token_range[0]
                    < int(segment["global_offset"]) + int(segment["length"])
                    < chunk_token_range[1]
                    for segment in segments
                )
            )
            if crosses_segment_boundary:
                from sglang.srt.layers.attention.redknot.v4.merged_prefill import (
                    allow_cross_segment_merged_prefill,
                )

                merged_prefill_allowed = allow_cross_segment_merged_prefill(
                    plan,
                    batch_size=forward_batch.batch_size,
                    logical_chunk_start=chunk_token_range[0],
                    logical_chunk_tokens=(
                        chunk_token_range[1] - chunk_token_range[0]
                    ),
                    server_max_prefill_tokens=self.server_args.max_prefill_tokens,
                    server_opt_in_cap=int(
                        os.environ.get(
                            "REDKNOT_V4_MERGED_PREFILL_MAX_TOKENS", "0"
                        )
                    ),
                )
            else:
                merged_prefill_allowed = False
            if crosses_segment_boundary and not merged_prefill_allowed:
                # SWA for a completed offline segment is restored by the layer
                # hook before attention. Cross-segment execution is nevertheless
                # restricted to the explicit, bounded merged-prefill opt-in.
                raise RuntimeError(
                    "selected-row scheduler chunks must not cross segment boundaries"
                )
            forward_batch.redknot_original_chunk_token_range = chunk_token_range
            if "hot_budget_tokens" in plan:
                explicit_budget = int(plan["hot_budget_tokens"])
                if explicit_budget < 0:
                    raise ValueError("hot_budget_tokens must be non-negative")
                mandatory = max(0, logical_total - query_start)
                if selection_policy == "checkpoint_islands":
                    mandatory += sum(
                        checkpoint_mandatory_prefix_tokens(
                            segment_global_offset=int(segment["global_offset"]),
                            segment_length=int(segment["length"]),
                        )
                        for segment in segments
                    )
                else:
                    skip_prefix = bool(plan.get("skip_prefix_recompute", True))
                    mandatory += sum(
                        0
                        if (int(segment["global_offset"]) == 0)
                        else int(segment.get("skip_first", 128))
                        for segment in segments
                        if not (
                            skip_prefix and int(segment["global_offset"]) == 0
                        )
                    )
                if query_protected_ranges_online:
                    for protected_begin, protected_end in query_protected_ranges_online:
                        segment_index = protected_segment_index_for_range(
                            protected_begin, protected_end
                        )
                        protected_segment = segments[segment_index]
                        base_end = int(protected_segment["global_offset"]) + (
                            checkpoint_mandatory_prefix_tokens(
                                segment_global_offset=int(
                                    protected_segment["global_offset"]
                                ),
                                segment_length=int(protected_segment["length"]),
                            )
                            if selection_policy == "checkpoint_islands"
                            else int(protected_segment.get("skip_first", 128))
                        )
                        mandatory += (protected_end - protected_begin) - max(
                            0,
                            min(protected_end, base_end) - protected_begin,
                        )
                max_extra = max(
                    0,
                    math.floor(runtime_config.abort_cost_ratio * logical_total)
                    - mandatory,
                )
                if explicit_budget > max_extra:
                    raise ValueError(
                        f"hot_budget_tokens={explicit_budget} exceeds safe "
                        f"request budget {max_extra}"
                    )
        except (KeyError, TypeError, ValueError) as error:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: invalid runtime metadata: %s",
                    error,
                )
            forward_batch.redknot_reuse_plan = [None]
            return

        if selection_policy not in {
            "legacy",
            "request_global",
            "checkpoint_islands",
        }:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: unsupported selection "
                    "policy %s",
                    selection_policy,
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        if selection_policy in {"request_global", "checkpoint_islands"}:
            requested_ratio = active_budget_ratio
            if not 0.0 < requested_ratio < runtime_config.abort_cost_ratio:
                if self.tp_rank == 0:
                    logger.warning(
                        "RedKnot selected-row dense fallback: active budget ratio "
                        "%.4f is outside (0, %.4f)",
                        requested_ratio,
                        runtime_config.abort_cost_ratio,
                    )
                forward_batch.redknot_reuse_plan = [None]
                return
            if int(plan.get("interior_stride", 0) or 0) != 0:
                if self.tp_rank == 0:
                    logger.warning(
                        "RedKnot selected-row dense fallback: selected replay "
                        "does not allow interior_stride"
                    )
                forward_batch.redknot_reuse_plan = [None]
                return
            if selection_policy == "checkpoint_islands" and checkpoint_stride == 0:
                if self.tp_rank == 0:
                    logger.warning(
                        "RedKnot selected-row dense fallback: checkpoint-island "
                        "replay requires checkpoint_stride_tokens"
                    )
                forward_batch.redknot_reuse_plan = [None]
                return
        if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: segmented replay "
                    "currently requires SGLANG_OPT_USE_COMPRESSOR_V2=0"
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        if (
            selection_policy == "checkpoint_islands"
            and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        ):
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: checkpoint islands "
                    "require SGLANG_OPT_USE_ONLINE_COMPRESS=0 because the "
                    "online-C128 state layout has no independent checkpoint slots"
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        if (
            self.server_args.enable_dp_attention
            or self.attn_cp_size != 1
            or self.moe_ep_size != 1
        ):
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: DP/CP attention and "
                    "expert parallelism are unsupported"
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        # Text-only scheduler batches normally carry ``mm_inputs=[None]``.
        # Testing the list itself treats that placeholder as real multimodal
        # state and silently disables selected-row execution for every request.
        # Keep the conservative fallback, but base it on actual payloads and log
        # the precise field so a future model-specific extension is diagnosable.
        unsupported_aux = []
        if forward_batch.input_embeds is not None:
            unsupported_aux.append("input_embeds")
        if forward_batch.replace_embeds is not None:
            unsupported_aux.append("replace_embeds")
        if any(item is not None for item in (forward_batch.mm_inputs or ())):
            unsupported_aux.append("mm_inputs")
        if any(forward_batch.lora_ids or ()):
            unsupported_aux.append("lora_ids")
        if forward_batch.token_type_ids is not None:
            unsupported_aux.append("token_type_ids")
        if forward_batch.mrope_positions is not None:
            unsupported_aux.append("mrope_positions")
        if forward_batch.ngram_embedding_info is not None:
            unsupported_aux.append("ngram_embedding_info")
        if forward_batch.multi_item_delimiter_indices is not None:
            unsupported_aux.append("multi_item_delimiter_indices")
        if forward_batch.tbo_split_seq_index is not None or forward_batch.tbo_children:
            unsupported_aux.append("two_batch_overlap")
        if unsupported_aux:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: per-token auxiliary "
                    "inputs are unsupported: %s",
                    ",".join(unsupported_aux),
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        use_indexer_hot = (
            os.environ.get("REDKNOT_V4_INDEXER_KV_REUSE", "0") == "1"
        )
        if selection_policy in {"request_global", "checkpoint_islands"} and (
            not use_indexer_hot
            or not runtime_config.reuse_window_kv
            or not bool(plan.get("refresh_selected_c4_rows", False))
            or os.environ.get("REDKNOT_V4_SEGMENTED_COMPRESSOR", "0") != "1"
        ):
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: selected-row "
                    "replay requires Indexer reuse, SWA reuse, prefix refresh, "
                    "and segmented compressor"
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        ctrl_hot = get_offline_reuse_controller_v2()
        topology = deepseek_v4_redknot_topology(self.model_config.hf_config)
        target_ratios = tuple(topology["target_compress_ratios"])
        c128_state_group_width = (
            1 if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get() else 128
        )
        expected_state_group_widths = {
            index: (4 if ratio == 4 else c128_state_group_width)
            for index, ratio in enumerate(target_ratios)
            if ratio in (4, 128)
        }
        readiness = ctrl_hot.validate_restore_segments(
            segments,
            expected_c4_layers=int(topology["num_c4_layers"]),
            expected_c128_layers=int(topology["num_c128_layers"]),
            expected_swa_layer_ids=(
                tuple(range(int(topology["num_target_layers"])))
                if runtime_config.reuse_window_kv
                else None
            ),
            expected_c4_layer_ids=tuple(
                index for index, ratio in enumerate(target_ratios) if ratio == 4
            ),
            expected_c128_layer_ids=tuple(
                index for index, ratio in enumerate(target_ratios) if ratio == 128
            ),
            expected_state_group_widths=expected_state_group_widths,
            require_indexer_units=(
                use_indexer_hot
                and selection_policy in {"request_global", "checkpoint_islands"}
            ),
            required_checkpoint_stride=(
                checkpoint_stride
                if selection_policy == "checkpoint_islands"
                else None
            ),
            require_swa_checkpoints=(selection_policy == "checkpoint_islands"),
        )
        ready_vote = torch.tensor(
            [1.0 if readiness.ready else 0.0],
            dtype=torch.float32,
            device=positions.device,
        )
        ready_votes = self.tp_group.all_reduce(ready_vote)
        globally_ready = (
            int(round(float(ready_votes.item()))) == self.tp_group.world_size
        )
        if not globally_ready:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: cache readiness failed "
                    "on at least one TP rank (local=%s)",
                    readiness.reason or "ready",
                )
            forward_batch.redknot_reuse_plan = [None]
            return

        if runtime_config.mode == "correctness":
            # Correctness mode may still reuse compressed state, but it must keep
            # the complete online hidden stream.
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot V4 correctness mode ignores unsafe skip_forward request"
                )
            forward_batch.redknot_rows_pruned = False
            return

        # Diagnostic: report this forward's token span so we can see whether the
        # scheduler chunked the request (each chunk is a separate forward_extend).
        _dbg_pos = forward_batch.positions
        logger.info(
            "RedKnot selected-rows ENTER: n_tokens=%d pos=[%d..%d] query_start=%s",
            int(_dbg_pos.numel()),
            int(_dbg_pos.min().item()) if _dbg_pos.numel() else -1,
            int(_dbg_pos.max().item()) if _dbg_pos.numel() else -1,
            plan.get("query_start"),
        )

        active_mask = positions >= query_start
        skip_prefix_seg = bool(plan.get("skip_prefix_recompute", True))
        base_prefixes = {}
        for segment_index, segment in enumerate(segments):
            offset = int(segment["global_offset"])
            length = int(segment["length"])
            if selection_policy == "checkpoint_islands":
                # Segment zero remains canonical.  A migrated segment pays only
                # one mandatory 128-token boundary block; [128, 512) is an
                # optional contiguous bridge that competes with checkpoint cells
                # in the request-global budget below.
                boundary = checkpoint_mandatory_prefix_tokens(
                    segment_global_offset=offset,
                    segment_length=length,
                )
            else:
                if skip_prefix_seg and offset == 0:
                    continue
                # Segment zero has no cross-segment SWA boundary. Migrated
                # segments keep a 128-token prefix after the previous segment's
                # offline SWA tail is materialized by its earlier chunk.
                boundary = min(int(segment.get("skip_first", 128)), length)
                if offset == 0:
                    boundary = 0
            base_prefixes[segment_index] = boundary
            active_mask |= (positions >= offset) & (positions < offset + boundary)
        for protected_begin, protected_end in query_protected_ranges_online:
            active_mask |= (positions >= protected_begin) & (positions < protected_end)

        # TP0 composes the query-weighted Indexer signal on CPU, then broadcasts
        # one compact mask.  Prefix-only allocation is deliberate: arbitrary C4
        # islands cannot reconstruct the skipped SWA/compressor history.
        hot_global_cpu = (
            torch.zeros(logical_total, dtype=torch.uint8)
            if self.tp_rank == 0
            else None
        )
        prefix_lengths_cpu = (
            torch.zeros(len(segments), dtype=torch.int32)
            if self.tp_rank == 0
            else None
        )
        if prefix_lengths_cpu is not None:
            for segment_index, prefix_tokens in base_prefixes.items():
                prefix_lengths_cpu[segment_index] = prefix_tokens
        hot_budget_tokens = 0
        selected_hot_tokens = 0
        checkpoint_descriptors_cpu = (
            torch.full(
                (checkpoint_max_islands, 4),
                -1,
                dtype=torch.int64,
            )
            if self.tp_rank == 0
            else None
        )
        checkpoint_island_count_cpu = 0
        selection_ok = torch.ones(1, dtype=torch.int32, device=positions.device)
        if use_indexer_hot and self.tp_rank == 0:
            try:
                if selection_policy not in {
                    "request_global",
                    "checkpoint_islands",
                }:
                    raise ValueError(
                        "Indexer selected-row replay requires a supported policy"
                    )
                from sglang.srt.layers.attention.redknot.v4.request_selector import (
                    CheckpointBridgeCandidates,
                    CheckpointCellCandidates,
                    SegmentPrefixCandidates,
                    allocate_checkpoint_cell_islands,
                    allocate_checkpoint_cell_islands_fast,
                    allocate_request_global_prefixes,
                    materialize_checkpoint_replay_layout,
                )

                mandatory_tokens = max(0, logical_total - query_start)
                mandatory_tokens += sum(base_prefixes.values())
                if query_protected_ranges_online:
                    for protected_begin, protected_end in query_protected_ranges_online:
                        segment_index = protected_segment_index_for_range(
                            protected_begin, protected_end
                        )
                        protected_segment = segments[segment_index]
                        base_end = int(protected_segment["global_offset"]) + (
                            base_prefixes.get(segment_index, 0)
                        )
                        mandatory_tokens += (protected_end - protected_begin) - max(
                            0,
                            min(protected_end, base_end) - protected_begin,
                        )
                if "hot_budget_tokens" in plan:
                    hot_budget_tokens = max(0, int(plan["hot_budget_tokens"]))
                else:
                    target_active = math.ceil(logical_total * active_budget_ratio)
                    hot_budget_tokens = max(0, target_active - mandatory_tokens)
                if selection_policy == "request_global":
                    candidates = []
                    for segment_index, segment in enumerate(segments):
                        offset = int(segment["global_offset"])
                        length = int(segment["length"])
                        if skip_prefix_seg and offset == 0:
                            continue
                        if (
                            protected_segment_online
                            and segment_index == query_protected_segment_index
                        ):
                            continue
                        base_prefix = base_prefixes[segment_index]
                        scored = ctrl_hot.get_indexer_unit_scores(
                            str(segment["seg_hash"])
                        )
                        if scored is None:
                            raise ValueError(
                                f"segment {segment_index} has no Indexer score artifact"
                            )
                        ordinals, scores = scored
                        candidates.append(
                            SegmentPrefixCandidates(
                                segment_index=segment_index,
                                length=length,
                                base_prefix_tokens=base_prefix,
                                query_weight=float(segment.get("query_score", 1.0)),
                                unit_ordinals=ordinals.tolist(),
                                unit_scores=scores.tolist(),
                            )
                        )
                    selected = allocate_request_global_prefixes(
                        candidates,
                        hot_budget_tokens=hot_budget_tokens,
                        per_segment_cap_ratio=cap_ratio,
                    )
                    for item in selected:
                        segment = segments[item.segment_index]
                        offset = int(segment["global_offset"])
                        begin = offset + base_prefixes[item.segment_index]
                        end = offset + item.prefix_tokens
                        hot_global_cpu[begin:end] = 1
                        prefix_lengths_cpu[item.segment_index] = item.prefix_tokens
                else:
                    bridges = []
                    cells = []
                    for segment_index, segment in enumerate(segments):
                        offset = int(segment["global_offset"])
                        length = int(segment["length"])
                        if skip_prefix_seg and offset == 0:
                            # Document one is the immutable offline prefix in
                            # the RAG closure experiment.  Do not let its
                            # internal checkpoint cells compete for online
                            # budget after its mandatory prefix was skipped.
                            continue
                        if (
                            protected_segment_online
                            and segment_index == query_protected_segment_index
                        ):
                            continue
                        scored = ctrl_hot.get_indexer_unit_scores(
                            str(segment["seg_hash"])
                        )
                        if scored is None:
                            raise ValueError(
                                f"segment {segment_index} has no Indexer score artifact"
                            )
                        ordinals, scores = scored
                        unit_scores = {
                            int(ordinal): float(score)
                            for ordinal, score in zip(
                                ordinals.tolist(), scores.tolist()
                            )
                        }
                        query_weight = max(
                            0.05, float(segment.get("query_score", 1.0))
                        )

                        def score_block(block_begin: int) -> float:
                            first_unit = block_begin // 4
                            return query_weight * sum(
                                unit_scores.get(unit, 0.0)
                                for unit in range(first_unit, first_unit + 32)
                            )

                        # The first 128 rows of a migrated segment are mandatory.
                        # The rest of its first checkpoint cell is one contiguous
                        # prefix choice, scored in the same DP as all 512+ cells.
                        bridge_begin = base_prefixes[segment_index]
                        bridge_end = min(checkpoint_stride, length)
                        bridge_block_count = (
                            (bridge_end - bridge_begin) // 128
                            if offset != 0
                            else 0
                        )
                        if any(
                            begin < offset + bridge_end
                            and end > offset + bridge_begin
                            for begin, end in query_protected_ranges_online
                        ):
                            bridge_block_count = 0
                        if bridge_block_count > 0:
                            bridges.append(
                                CheckpointBridgeCandidates(
                                    segment_index=segment_index,
                                    block_scores=[
                                        score_block(
                                            bridge_begin + block_index * 128
                                        )
                                        for block_index in range(bridge_block_count)
                                    ],
                                )
                            )

                        # Sparse restart artifacts begin at the first 512-token
                        # checkpoint anchor; cell zero is represented by the
                        # mandatory boundary plus the optional bridge above.
                        for cell_index in range(
                            1,
                            (length + checkpoint_stride - 1)
                            // checkpoint_stride,
                        ):
                            cell_begin = cell_index * checkpoint_stride
                            cell_global_begin = offset + cell_begin
                            cell_global_end = min(
                                offset + length,
                                cell_global_begin + checkpoint_stride,
                            )
                            # A checkpoint restore may only start a fresh
                            # online range.  Query-protected rows immediately
                            # before/after the cell would make the two ranges
                            # contiguous and either overwrite a freshly
                            # computed compressor state or violate the
                            # segmented-compressor restore contract.  Keep one
                            # boundary window on both sides; all endpoints are
                            # 512-aligned, so the next eligible cell remains a
                            # cheap, deterministic alternative.
                            checkpoint_guard_tokens = int(
                                runtime_config.boundary_replay_tokens
                            )
                            if any(
                                begin
                                < cell_global_end + checkpoint_guard_tokens
                                and end
                                > cell_global_begin - checkpoint_guard_tokens
                                for begin, end in query_protected_ranges_online
                            ):
                                continue
                            block_count = min(
                                checkpoint_stride // 128,
                                (length - cell_begin) // 128,
                            )
                            if block_count <= 0:
                                continue
                            cells.append(
                                CheckpointCellCandidates(
                                    segment_index=segment_index,
                                    cell_index=cell_index,
                                    block_scores=[
                                        score_block(
                                            cell_begin + block_index * 128
                                        )
                                        for block_index in range(block_count)
                                    ],
                                )
                            )
                    max_replay_by_segment = {}
                    effective_cap_by_segment = {}
                    for segment_index, segment in enumerate(segments):
                        effective_cap = checkpoint_effective_segment_cap_tokens(
                            segment_length=int(segment["length"]),
                            cap_ratio=cap_ratio,
                            mandatory_prefix_tokens=base_prefixes.get(
                                segment_index, 0
                            ),
                        )
                        effective_cap_by_segment[segment_index] = effective_cap
                        base_end = int(segment["global_offset"]) + (
                            base_prefixes.get(segment_index, 0)
                        )
                        protected_extra = sum(
                            (end - begin)
                            - max(0, min(end, base_end) - begin)
                            for begin, end in query_protected_ranges_online
                            if protected_segment_index_for_range(begin, end)
                            == segment_index
                        )
                        max_replay_by_segment[segment_index] = max(
                            0,
                            effective_cap
                            - base_prefixes.get(segment_index, 0)
                            - protected_extra,
                        )
                    checkpoint_allocator = (
                        allocate_checkpoint_cell_islands_fast
                        if plan.get("row_sparse_closure") is True
                        else allocate_checkpoint_cell_islands
                    )
                    selected_islands = checkpoint_allocator(
                        cells,
                        token_budget_tokens=hot_budget_tokens,
                        checkpoint_stride=checkpoint_stride,
                        max_islands=checkpoint_max_islands,
                        max_tokens_by_segment=max_replay_by_segment,
                        bridges=bridges,
                    )
                    has_eligible_positive_score = any(
                        max_replay_by_segment.get(candidate.segment_index, 0) >= 128
                        and any(score > 0.0 for score in candidate.block_scores)
                        for candidate in (*bridges, *cells)
                    )
                    if (
                        hot_budget_tokens >= 128
                        and has_eligible_positive_score
                        and not selected_islands
                    ):
                        raise AssertionError(
                            "checkpoint allocator dropped all positive candidates"
                        )
                    layout = materialize_checkpoint_replay_layout(
                        selected_islands,
                        base_prefix_tokens_by_segment=base_prefixes,
                        checkpoint_stride=checkpoint_stride,
                    )
                    per_segment_tokens = {}
                    for item in selected_islands:
                        segment = segments[item.segment_index]
                        offset = int(segment["global_offset"])
                        begin = offset + item.token_begin
                        end = offset + item.token_end
                        hot_global_cpu[begin:end] = 1
                        per_segment_tokens[item.segment_index] = (
                            per_segment_tokens.get(item.segment_index, 0)
                            + item.token_end
                            - item.token_begin
                        )
                    for segment_index, prefix_tokens in (
                        layout.selected_prefix_tokens
                    ):
                        prefix_lengths_cpu[segment_index] = prefix_tokens
                    if len(layout.restore_islands) > checkpoint_max_islands:
                        raise AssertionError(
                            "checkpoint allocator violated the restore-island limit"
                        )
                    for descriptor_index, item in enumerate(
                        layout.restore_islands
                    ):
                        checkpoint_descriptors_cpu[descriptor_index] = torch.tensor(
                            [
                                item.segment_index,
                                item.token_begin,
                                item.token_end,
                                item.token_begin,
                            ],
                            dtype=torch.int64,
                        )
                    checkpoint_island_count_cpu = len(layout.restore_islands)
                    if any(
                        per_segment_tokens.get(segment_index, 0)
                        + base_prefixes.get(segment_index, 0)
                        > effective_cap_by_segment[segment_index]
                        for segment_index in range(len(segments))
                    ):
                        raise AssertionError(
                            "checkpoint allocator violated a segment replay cap"
                        )
                selected_hot_tokens = int(hot_global_cpu.sum().item())
                if mandatory_tokens + selected_hot_tokens > math.floor(
                    runtime_config.abort_cost_ratio * logical_total
                ):
                    raise ValueError("selected replay mask exceeds abort cost ratio")
            except Exception as error:
                selection_ok.zero_()
                logger.warning(
                    "RedKnot selected-row selection failed on TP0: %s", error
                )
        self.tp_group.broadcast(selection_ok, src=0)
        if not bool(selection_ok.item()):
            forward_batch.redknot_reuse_plan = [None]
            return
        hot_global = torch.zeros(
            logical_total, dtype=torch.uint8, device=positions.device
        )
        if self.tp_rank == 0 and hot_global_cpu is not None:
            hot_global.copy_(hot_global_cpu.to(device=positions.device))
        self.tp_group.broadcast(hot_global, src=0)
        prefix_lengths = torch.zeros(
            len(segments), dtype=torch.int32, device=positions.device
        )
        if self.tp_rank == 0 and prefix_lengths_cpu is not None:
            prefix_lengths.copy_(prefix_lengths_cpu.to(device=positions.device))
        self.tp_group.broadcast(prefix_lengths, src=0)
        forward_batch.redknot_selected_prefix_tokens = tuple(
            int(value) for value in prefix_lengths.to(device="cpu").tolist()
        )
        checkpoint_island_count = torch.zeros(
            1, dtype=torch.int32, device=positions.device
        )
        checkpoint_descriptors = torch.full(
            (checkpoint_max_islands, 4),
            -1,
            dtype=torch.int64,
            device=positions.device,
        )
        if self.tp_rank == 0:
            checkpoint_island_count.fill_(checkpoint_island_count_cpu)
            if checkpoint_descriptors_cpu is not None:
                checkpoint_descriptors.copy_(
                    checkpoint_descriptors_cpu.to(device=positions.device)
                )
        self.tp_group.broadcast(checkpoint_island_count, src=0)
        self.tp_group.broadcast(checkpoint_descriptors, src=0)
        checkpoint_islands = []
        descriptor_error = ""
        try:
            descriptor_count = int(checkpoint_island_count.item())
            if not 0 <= descriptor_count <= checkpoint_max_islands:
                raise ValueError("broadcast checkpoint island count is invalid")
            descriptor_rows = checkpoint_descriptors[:descriptor_count].to(
                device="cpu"
            )
            if selection_policy != "checkpoint_islands" and descriptor_count:
                raise ValueError("non-checkpoint policy broadcast checkpoint islands")
            if selection_policy == "checkpoint_islands":
                expected_hot = torch.zeros_like(hot_global)
                previous_key = None
                previous_end_by_segment = {}
                for segment_index, segment in enumerate(segments):
                    offset = int(segment["global_offset"])
                    length = int(segment["length"])
                    base_prefix = int(base_prefixes.get(segment_index, 0))
                    online_prefix = int(prefix_lengths[segment_index].item())
                    if (
                        online_prefix < base_prefix
                        or online_prefix > length
                        or online_prefix % 128 != 0
                    ):
                        raise ValueError("broadcast online prefix is invalid")
                    if online_prefix > base_prefix:
                        expected_hot[
                            offset + base_prefix : offset + online_prefix
                        ] = 1
                    previous_end_by_segment[segment_index] = online_prefix

                for raw_row in descriptor_rows.tolist():
                    segment_index, local_begin, local_end, anchor = map(
                        int, raw_row
                    )
                    if not 0 <= segment_index < len(segments):
                        raise ValueError(
                            "broadcast checkpoint segment index is invalid"
                        )
                    segment = segments[segment_index]
                    offset = int(segment["global_offset"])
                    length = int(segment["length"])
                    key = (segment_index, local_begin)
                    if previous_key is not None and key <= previous_key:
                        raise ValueError(
                            "broadcast checkpoint islands are not strictly ordered"
                        )
                    previous_key = key
                    if (
                        anchor != local_begin
                        or anchor <= 0
                        or anchor % checkpoint_stride != 0
                        or local_begin % 128 != 0
                        or local_end % 128 != 0
                        or local_end <= local_begin
                        or local_end > length
                    ):
                        raise ValueError("broadcast checkpoint island is invalid")
                    previous_end = previous_end_by_segment[segment_index]
                    if local_begin - previous_end < 128:
                        raise ValueError(
                            "checkpoint islands with overlapping SWA carry "
                            "were not merged"
                        )
                    previous_end_by_segment[segment_index] = local_end
                    expected_hot[
                        offset + local_begin : offset + local_end
                    ] = 1
                    checkpoint_islands.append(
                        {
                            "segment_index": segment_index,
                            "checkpoint_anchor": anchor,
                            "global_begin": offset + local_begin,
                            "global_end": offset + local_end,
                        }
                    )
                if not torch.equal(expected_hot, hot_global):
                    raise ValueError(
                        "checkpoint descriptors do not match the selected hot mask"
                    )
        except Exception as error:
            descriptor_error = str(error)

        descriptor_vote = torch.tensor(
            [0.0 if descriptor_error else 1.0],
            dtype=torch.float32,
            device=positions.device,
        )
        descriptor_votes = self.tp_group.all_reduce(descriptor_vote)
        descriptors_globally_valid = (
            int(round(float(descriptor_votes.item()))) == self.tp_group.world_size
        )
        if not descriptors_globally_valid:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: invalid checkpoint "
                    "descriptor on at least one TP rank (local=%s)",
                    descriptor_error or "valid",
                )
            forward_batch.redknot_reuse_plan = [None]
            return
        forward_batch.redknot_checkpoint_islands = tuple(checkpoint_islands)
        if use_indexer_hot:
            valid = (positions >= 0) & (positions < logical_total)
            hot_local = torch.zeros_like(active_mask)
            if bool(valid.any().item()):
                hot_local[valid] = hot_global.index_select(
                    0, positions[valid]
                ).to(torch.bool)
            active_mask |= hot_local

        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        full_tokens = int(forward_batch.input_ids.shape[0])
        all_rows_active = active_indices.numel() == full_tokens
        if active_indices.numel() == 0:
            # Prefix-only chunk (segment 0 skipped, no query/hot rows): keep a
            # single placeholder row so the forward pipeline stays valid. Its
            # output is discarded — the chunk's KV is fully served from offline
            # reuse. This lets the pure-prefix chunk cost ~1 row instead of 8192.
            active_indices = active_indices.new_zeros(1)
            forward_batch.redknot_placeholder_only = True

        active_n = int(active_indices.numel())
        active_ratio = active_n / max(1, full_tokens)
        if self.tp_rank == 0:
            logger.info(
                "REDKNOT_SELECTION_LAYOUT policy=%s prefixes=%s islands=%s",
                selection_policy,
                tuple(forward_batch.redknot_selected_prefix_tokens),
                tuple(
                    (
                        int(item["segment_index"]),
                        int(item["checkpoint_anchor"]),
                        int(item["global_begin"]),
                        int(item["global_end"]),
                    )
                    for item in checkpoint_islands
                ),
            )
            logger.info(
                "REDKNOT_METRIC active_rows policy=%s active=%d full=%d "
                "active_ratio=%.6f online_row_saving=%.6f "
                "hot_budget_tokens=%d selected_hot_tokens=%d",
                selection_policy,
                active_n,
                full_tokens,
                active_ratio,
                1.0 - active_ratio,
                hot_budget_tokens,
                selected_hot_tokens,
            )

        # Build and validate both compressor schedules before deleting any rows.
        # A schedule/descriptor error is still recoverable here as a true dense
        # fallback; after the index_select below, skipped hidden rows no longer
        # exist and a layer-local fallback would be unsound.
        forward_batch.redknot_active_row_indices = active_indices
        planned_positions = positions.index_select(0, active_indices)
        schedule_ready = self._prepare_redknot_v4_boundary_replay(
            forward_batch,
            positions_override=planned_positions,
        )
        schedule_vote = torch.tensor(
            [1.0 if schedule_ready else 0.0],
            dtype=torch.float32,
            device=positions.device,
        )
        schedule_votes = self.tp_group.all_reduce(schedule_vote)
        schedules_globally_ready = (
            int(round(float(schedule_votes.item()))) == self.tp_group.world_size
        )
        if not schedules_globally_ready:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: compressor schedule "
                    "was invalid on at least one TP rank"
                )
            forward_batch.redknot_reuse_plan = [None]
            for attr_name in (
                "redknot_v4_boundary_replay",
                "redknot_v4_compressor_schedules",
                "redknot_active_row_indices",
                "redknot_placeholder_only",
            ):
                try:
                    delattr(forward_batch, attr_name)
                except AttributeError:
                    # ForwardBatch dataclass defaults are visible through
                    # hasattr even when no instance value was installed.
                    pass
            return

        if all_rows_active:
            # The selection is still included in accounting, but no tensor or
            # extend metadata needs mutation when every row remains online.
            forward_batch.redknot_active_row_count = full_tokens
            forward_batch.redknot_active_global_positions = positions
            forward_batch.redknot_rows_pruned = False
            return

        forward_batch.input_ids = forward_batch.input_ids.index_select(0, active_indices)
        forward_batch.positions = forward_batch.positions.index_select(0, active_indices)
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.index_select(
            0, active_indices
        )
        active_count = int(active_indices.numel())
        forward_batch.extend_num_tokens = active_count
        forward_batch.extend_seq_lens = forward_batch.extend_seq_lens.new_tensor(
            [active_count]
        )
        forward_batch.extend_seq_lens_cpu = [active_count]
        forward_batch.extend_start_loc = forward_batch.extend_start_loc.new_zeros((1,))
        if forward_batch.extend_logprob_start_lens_cpu is not None:
            # Selected-row execution cannot return prompt-token logprobs. Setting
            # the start to extend_len keeps only the final sampling row, which is
            # sufficient for generated-token top-logprobs used by the benchmark.
            forward_batch.extend_logprob_start_lens_cpu = [active_count]
        forward_batch.num_token_non_padded_cpu = active_count
        if forward_batch.num_token_non_padded is not None:
            forward_batch.num_token_non_padded.fill_(active_count)
        # Record which logical rows stayed online so the per-layer reuse hook
        # only injects offline compressed state for the skipped interior rows.
        forward_batch.redknot_active_row_indices = active_indices
        forward_batch.redknot_active_row_count = active_count
        forward_batch.redknot_active_global_positions = forward_batch.positions
        forward_batch.redknot_rows_pruned = True

    def _prepare_redknot_v4_boundary_replay(
        self,
        forward_batch: ForwardBatch,
        *,
        positions_override: Optional[torch.Tensor] = None,
    ) -> bool:
        if (
            getattr(forward_batch, "redknot_v4_boundary_replay", None) is not None
            and getattr(
                forward_batch, "redknot_v4_compressor_schedules", None
            )
            is not None
        ):
            return True
        plans = getattr(forward_batch, "redknot_reuse_plan", None)
        if not plans or len(plans) != 1:
            return False
        plan = plans[0]
        if not plan or plan.get("mode") != "restore":
            return False
        from sglang.srt.layers.attention.redknot.v4.reuse_planner import (
            MLA_OFF_INDEPENDENT_RELOCATION_PROFILE,
            MLA_OFF_EXECUTION_PROFILE,
        )

        if (
            positions_override is None
            and plan.get("mla_off_execution_profile")
            in {
                MLA_OFF_EXECUTION_PROFILE,
                MLA_OFF_INDEPENDENT_RELOCATION_PROFILE,
            }
            and bool(plan.get("reuse_mla_off", False))
            and not bool(plan.get("capture_mla_off", False))
            and not bool(plan.get("skip_forward", False))
        ):
            # Pure logical-head MLA reuse neither prunes transformer rows nor
            # restores the legacy C4/C128/Indexer state.  Preserve its plan for
            # RedKnotMLAAttnBackend and do not subject it to the unrelated
            # boundary-replay/compressor validator below.  This method's return
            # value is ignored by the ordinary full-row forward call.  Return
            # True nevertheless: the helper's contract is "schedule
            # requirement satisfied", and this is an intentional no-op.
            return True

        from sglang.srt.layers.attention.redknot.v4.boundary_replay import (
            build_boundary_replay,
        )
        from sglang.srt.layers.attention.redknot.v4.config import RedKnotV4Config
        from sglang.srt.layers.attention.redknot.v4.reuse_planner import (
            validate_runtime_reuse_plan,
        )

        try:
            config = RedKnotV4Config(
                mode=os.environ.get("REDKNOT_V4_MODE", "correctness"),
                reuse_window_kv=bool(plan.get("reuse_window_kv", False)),
            )
        except ValueError as error:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot boundary replay disabled by runtime config: %s", error
                )
            # Keep the backend and compressor on the same dense fallback path.
            forward_batch.redknot_reuse_plan = [None]
            return False
        validation = validate_runtime_reuse_plan(
            plan,
            config=config,
            dspark_active=self.spec_algorithm.is_speculative(),
        )
        if not validation.valid:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot boundary replay dense fallback: reason=%s detail=%s",
                    validation.fallback_reason.value,
                    validation.detail,
                )
            forward_batch.redknot_reuse_plan = [None]
            return False
        orig_seq_lens = getattr(forward_batch, "orig_seq_lens", None)
        total_tokens = (
            int(orig_seq_lens[0].item())
            if orig_seq_lens is not None and orig_seq_lens.numel() == 1
            else int(forward_batch.seq_lens_cpu[0].item())
        )
        from sglang.srt.layers.attention.redknot.v4.segmented_compressor import (
            build_segmented_compressor_schedule,
        )

        try:
            replay = build_boundary_replay(
                segments=plan.get("segments", ()),
                total_tokens=total_tokens,
                boundary_tokens=config.boundary_replay_tokens,
            )
            schedule_positions = (
                positions_override
                if positions_override is not None
                else forward_batch.positions
            )
            positions = schedule_positions.detach().to(device="cpu").tolist()
            schedules = {
                ratio: build_segmented_compressor_schedule(
                    replay=replay,
                    positions=positions,
                    compress_ratio=ratio,
                    # Every retained selected-row token is intentional online
                    # work: boundary/prefix, checkpoint replay, or query.
                    include_all_present_rows=(
                        bool(plan.get("refresh_selected_c4_rows", False))
                        and plan.get("selection_policy")
                        in {"request_global", "checkpoint_islands"}
                        and hasattr(forward_batch, "redknot_active_row_indices")
                    ),
                    logical_chunk_range=getattr(
                        forward_batch,
                        "redknot_original_chunk_token_range",
                        None,
                    ),
                    checkpoint_islands=getattr(
                        forward_batch, "redknot_checkpoint_islands", ()
                    ),
                )
                for ratio in (4, 128)
            }
        except Exception as error:
            if self.tp_rank == 0:
                logger.warning(
                    "RedKnot selected-row dense fallback: compressor schedule "
                    "validation failed: %s",
                    error,
                )
            forward_batch.redknot_reuse_plan = [None]
            return False
        forward_batch.redknot_v4_boundary_replay = replay
        forward_batch.redknot_v4_compressor_schedules = schedules
        return True

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches may be padded (batch_size > 0) for MLP
        # sync. Reinit metadata for the padded case so attention kernels see
        # the right batch_size (e.g. DSA Indexer). For the unpadded case
        # (batch_size == 0) explicitly drop any stale forward_metadata left
        # over from the previous forward — without this, attention layers
        # called from the idle path can re-read a prior batch's req_pool
        # indices and trigger SWA mapping use-after-free.
        if forward_batch.batch_size > 0:
            self.attn_backend.init_forward_metadata(forward_batch)
        else:
            self.attn_backend.forward_metadata = None

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        ctx = (
            self.device_timer.wrap(metadata={"category": "idle"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        self.forward_pass_id += 1
        # ForwardBatch objects and inference tensors may be recycled by the
        # scheduler.  Give attention backends a monotonic, runner-scoped
        # generation so per-forward CPU/layout certificates cannot survive an
        # in-place refill of the same tensor storage.  Split-prefill slices are
        # one logical forward and retain the generation installed at split 0.
        split_prefill = forward_batch.forward_mode.is_split_prefill()
        current_generation = getattr(
            forward_batch, "_redknot_forward_generation_id", None
        )
        starts_redknot_forward = (
            not split_prefill
            or int(getattr(forward_batch, "split_index", 0)) == 0
            or not (
                isinstance(current_generation, tuple)
                and len(current_generation) == 2
            )
        )
        if starts_redknot_forward:
            forward_batch._redknot_forward_generation_id = (
                id(self),
                int(self.forward_pass_id),
            )
            for attribute in (
                "redknot_v4_boundary_replay",
                "redknot_v4_compressor_schedules",
                "redknot_active_row_indices",
                "redknot_active_row_count",
                "redknot_active_global_positions",
                "redknot_original_chunk_token_range",
                "redknot_selected_prefix_tokens",
                "redknot_checkpoint_islands",
                "redknot_placeholder_only",
                "redknot_rows_pruned",
                "redknot_indexer_selected_tokens",
            ):
                try:
                    delattr(forward_batch, attribute)
                except AttributeError:
                    # Clearing a recycled ForwardBatch must be idempotent;
                    # class-level dataclass defaults are not instance state.
                    pass

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = (
            torch.profiler.record_function(_build_step_span_name(forward_batch))
            if torch.autograd._profiler_enabled()
            else contextlib.nullcontext()
        )
        with (
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            if forward_batch.redknot_reuse_plan:
                from sglang.srt.layers.attention.redknot.glm52_latent import (
                    process_glm52_h0_before_forward,
                )

                process_glm52_h0_before_forward(
                    forward_batch, self.token_to_kv_pool, self.req_to_token_pool
                )
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if forward_batch.redknot_reuse_plan:
                from sglang.srt.layers.attention.redknot.glm52_latent import (
                    process_glm52_h0_after_forward,
                )

                process_glm52_h0_after_forward(
                    forward_batch, self.token_to_kv_pool, self.req_to_token_pool
                )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    skip_attn_backend_init,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (experts_capturer := get_global_experts_capturer()) is not None:
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_recover_ep_ranks()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Honor an outer-published context (spec workers wrap each per-step
        # draft forward with the i-th child backend); otherwise publish this
        # runner's own attn_backend for the forward.
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        with ctx_mgr:
            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.graph_runner
                and self.graph_runner.can_run(forward_batch)
            )

            # Hisparse coordinator — backends now read it from self.model_runner.
            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            if self.is_hybrid_swa:
                self.token_to_kv_pool.invalidate_loc_cache()

            # Replay cuda graph if applicable
            if can_run_graph:
                ret = self.graph_runner.replay(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # For MLP sync
            if forward_batch.global_num_tokens_cpu is not None:
                forward_batch.prepare_mlp_sync_batch(self)
            else:
                forward_batch.prepare_attn_tp_scatter_input(self)

            # Normalize num_token_non_padded to be local to this attention TP rank if needed.
            # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
            # flavors already feed a zigzag-split rank-local layout whose token count
            # should not be further divided by attn_tp_size. MHA-arch prefill CP
            # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
            # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
            if (
                forward_batch.num_token_non_padded is not None
                and forward_batch.global_num_tokens_gpu is not None
                and require_gathered_buffer(self.server_args)
                and not is_dsa_enable_prefill_cp()
                and not is_mla_prefill_cp_enabled()
            ):
                forward_batch.adjust_num_token_non_padded_for_attn_tp(
                    server_args=self.server_args,
                )

            # Hisparse coordinator — backends now read it from self.model_runner.
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Forward without cuda graph
            if forward_batch.forward_mode.is_decode():
                ret = self.forward_decode(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_split_prefill():
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                ret, can_run_graph = self.forward_extend(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str):
        return self._weight_checker.handle(action=action)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )

    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and not elastic_ep_state.is_active_equal_last():
            elastic_ep_state.snapshot_active_to_last()
            elastic_ep_state.sync_active_to_cpu()
            logging.info("EPLB due to rank faults")
            gen = self.eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


def _build_step_span_name(forward_batch: ForwardBatch) -> str:
    """Build a profile-trace span name for one forward step."""
    mode = forward_batch.forward_mode
    bs = forward_batch.batch_size
    if mode == ForwardMode.EXTEND:
        ext_toks = forward_batch.extend_num_tokens or 0
        return f"step[EXTEND bs={bs} toks={ext_toks}]"
    return f"step[{mode.name} bs={bs}]"


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
