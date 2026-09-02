from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn

from sglang.jit_kernel.dsv4 import linear_bf16_fp32, triton_create_paged_compress_data
from sglang.jit_kernel.dsv4.compress_old import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_fused_norm_rope_inplace,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.triton_kernel import act_quant
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.dp_attention import get_attention_cp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.models.deepseek_v2 import _is_hip
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class FusedCompressMetadata(NamedTuple):
    write_loc: torch.Tensor
    extra_data: Optional[torch.Tensor]
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan]

    def copy_(self, other: FusedCompressMetadata) -> None:
        from .metadata import maybe_copy_inplace

        self.write_loc.copy_(other.write_loc)
        maybe_copy_inplace(self.extra_data, src=other.extra_data)
        self.plan.copy_(other.plan)


def _dirty_field(value: object, name: str, *aliases: str) -> object:
    """Read one dirty-workset field from either a Mapping or an object."""

    names = (name,) + aliases
    if isinstance(value, Mapping):
        for candidate in names:
            if candidate in value:
                return value[candidate]
    else:
        for candidate in names:
            if hasattr(value, candidate):
                return getattr(value, candidate)
    raise ValueError(f"dirty compressor value is missing {name}")


def _dirty_int(value: object, name: str) -> int:
    """Normalize a CPU scalar without accepting booleans or device tensors."""

    if type(value) is int:
        return value
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer CPU scalar")
    return value


def _dirty_int_tuple(
    values: object,
    name: str,
    *,
    allow_empty: bool,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    try:
        result = tuple(_dirty_int(value, f"{name} entry") for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer sequence") from error
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


@dataclass(frozen=True)
class DirtyCompressorIsland:
    """One contiguous online compressor interval in a packed ragged batch.

    ``request_row_*`` are relative to the request's extension rows while
    ``flat_*`` and ``completion_output_rows`` address the packed ``x``/loc
    tensors.  ``state_slot_indices`` name state slots already restored by the
    caller; this class never restores or overwrites clean state itself.
    """

    flat_begin: int
    flat_end: int
    request_row_begin: int
    request_row_end: int
    token_begin: int
    token_end: int
    state_slot_indices: Tuple[int, ...]
    completion_output_rows: Tuple[int, ...]


@dataclass(frozen=True)
class DirtyCompressorRequestWorkset:
    """Dirty islands for one request in the packed forward."""

    request_index: int
    flat_row_offset: int
    row_count: int
    seq_len_before: int
    islands: Tuple[DirtyCompressorIsland, ...]


@dataclass(frozen=True)
class DirtyCompressorStateBinding:
    """Opaque proof that one island's recurrent state is already present."""

    request_index: int
    token_begin: int
    state_slot_indices: Tuple[int, ...]


@dataclass(frozen=True)
class DirtyCompressorStateReceipt:
    """Duck-typed contract accepted from the shared-latent restore path."""

    layer_id: int
    compress_ratio: int
    is_indexer: bool
    bindings: Tuple[DirtyCompressorStateBinding, ...]
    restore_token: str
    forward_token: str = ""


@dataclass(frozen=True)
class DirtyCompressorPreflight:
    """Canonical geometry produced before metadata or cache mutation."""

    layer_id: int
    compress_ratio: int
    is_indexer: bool
    total_rows: int
    worksets: Tuple[DirtyCompressorRequestWorkset, ...]
    state_receipt: DirtyCompressorStateReceipt

    @property
    def islands(self) -> Tuple[Tuple[int, DirtyCompressorIsland], ...]:
        return tuple(
            (workset.request_index, island)
            for workset in self.worksets
            for island in workset.islands
        )


def _normalize_dirty_island(value: object) -> DirtyCompressorIsland:
    state_slots = _dirty_int_tuple(
        _dirty_field(value, "state_slot_indices", "state_slots"),
        "state_slot_indices",
        allow_empty=True,
    )
    completions = _dirty_int_tuple(
        _dirty_field(value, "completion_output_rows", "completion_rows"),
        "completion_output_rows",
        allow_empty=True,
    )
    return DirtyCompressorIsland(
        flat_begin=_dirty_int(_dirty_field(value, "flat_begin"), "flat_begin"),
        flat_end=_dirty_int(_dirty_field(value, "flat_end"), "flat_end"),
        request_row_begin=_dirty_int(
            _dirty_field(value, "request_row_begin", "row_begin"),
            "request_row_begin",
        ),
        request_row_end=_dirty_int(
            _dirty_field(value, "request_row_end", "row_end"),
            "request_row_end",
        ),
        token_begin=_dirty_int(
            _dirty_field(value, "token_begin"), "token_begin"
        ),
        token_end=_dirty_int(_dirty_field(value, "token_end"), "token_end"),
        state_slot_indices=state_slots,
        completion_output_rows=completions,
    )


def _normalize_dirty_workset(value: object) -> DirtyCompressorRequestWorkset:
    raw_islands = _dirty_field(value, "islands")
    try:
        islands = tuple(_normalize_dirty_island(item) for item in raw_islands)
    except TypeError as error:
        raise TypeError("dirty compressor islands must be a sequence") from error
    islands = tuple(
        sorted(
            islands,
            key=lambda item: (
                item.request_row_begin,
                item.request_row_end,
                item.flat_begin,
            ),
        )
    )
    return DirtyCompressorRequestWorkset(
        request_index=_dirty_int(
            _dirty_field(value, "request_index"), "request_index"
        ),
        flat_row_offset=_dirty_int(
            _dirty_field(value, "flat_row_offset"), "flat_row_offset"
        ),
        row_count=_dirty_int(_dirty_field(value, "row_count"), "row_count"),
        seq_len_before=_dirty_int(
            _dirty_field(value, "seq_len_before"), "seq_len_before"
        ),
        islands=islands,
    )


def _normalize_state_binding(value: object) -> DirtyCompressorStateBinding:
    return DirtyCompressorStateBinding(
        request_index=_dirty_int(
            _dirty_field(value, "request_index"), "state request_index"
        ),
        token_begin=_dirty_int(
            _dirty_field(value, "token_begin"), "state token_begin"
        ),
        state_slot_indices=_dirty_int_tuple(
            _dirty_field(value, "state_slot_indices", "state_slots"),
            "state_slot_indices",
            allow_empty=True,
        ),
    )


def _normalize_state_receipt(value: object) -> DirtyCompressorStateReceipt:
    raw_bindings = _dirty_field(value, "bindings", "entries", "state_bindings")
    try:
        bindings = tuple(_normalize_state_binding(item) for item in raw_bindings)
    except TypeError as error:
        raise TypeError("compressor state bindings must be a sequence") from error
    bindings = tuple(
        sorted(bindings, key=lambda item: (item.request_index, item.token_begin))
    )
    raw_is_indexer = _dirty_field(value, "is_indexer")
    if type(raw_is_indexer) is not bool:
        raise TypeError("state receipt is_indexer must be boolean")
    forward_token = ""
    try:
        forward_token = _dirty_field(value, "forward_token", "forward_id")
    except ValueError:
        pass
    if not isinstance(forward_token, str):
        raise TypeError("state receipt forward_token must be a string")
    restore_token = _dirty_field(
        value, "restore_token", "receipt_token", "schedule_digest"
    )
    if not isinstance(restore_token, str) or not restore_token:
        raise ValueError("state receipt restore_token must be a non-empty string")
    return DirtyCompressorStateReceipt(
        layer_id=_dirty_int(_dirty_field(value, "layer_id"), "state layer_id"),
        compress_ratio=_dirty_int(
            _dirty_field(value, "compress_ratio", "ratio"),
            "state compress_ratio",
        ),
        is_indexer=raw_is_indexer,
        bindings=bindings,
        restore_token=restore_token,
        forward_token=forward_token,
    )


def preflight_dirty_compressor_geometry(
    *,
    layer_id: int,
    compress_ratio: int,
    is_indexer: bool,
    total_rows: int,
    batch_size: int,
    seq_lens: Sequence[int],
    extend_lens: Sequence[int],
    dirty_worksets: Sequence[object],
    restored_state: object,
    target_loc_rows: int,
    forward_token: Optional[str] = None,
) -> DirtyCompressorPreflight:
    """Validate the entire ragged dirty transaction before its first write.

    This helper is torch-free and intentionally public so schedulers can
    preflight on CPU.  Mapping and attribute-based worksets/receipts are both
    accepted; the returned immutable representation is canonical.
    """

    layer_id = _dirty_int(layer_id, "layer_id")
    ratio = _dirty_int(compress_ratio, "compress_ratio")
    total_rows = _dirty_int(total_rows, "total_rows")
    batch_size = _dirty_int(batch_size, "batch_size")
    target_loc_rows = _dirty_int(target_loc_rows, "target_loc_rows")
    if layer_id < 0 or ratio not in (4, 128):
        raise ValueError("dirty compressor layer/ratio is invalid")
    if type(is_indexer) is not bool:
        raise TypeError("is_indexer must be boolean")
    if is_indexer and ratio != 4:
        raise ValueError("the Indexer compressor exists only for ratio 4")
    if total_rows < 0 or batch_size < 0 or target_loc_rows != total_rows:
        raise ValueError("dirty compressor packed row geometry is invalid")
    if forward_token is not None and (
        not isinstance(forward_token, str) or not forward_token
    ):
        raise ValueError("forward_token must be a non-empty string when supplied")

    seq = tuple(_dirty_int(value, "seq_len") for value in seq_lens)
    extend = tuple(_dirty_int(value, "extend_len") for value in extend_lens)
    if len(seq) != batch_size or len(extend) != batch_size:
        raise ValueError("ragged sequence lengths do not match batch_size")
    if any(value < 0 for value in seq) or any(value < 0 for value in extend):
        raise ValueError("ragged sequence lengths must be non-negative")
    if any(ext > length for ext, length in zip(extend, seq)):
        raise ValueError("extend length exceeds request sequence length")
    if sum(extend) != total_rows:
        raise ValueError("ragged extend lengths do not tile packed x")

    try:
        worksets = tuple(_normalize_dirty_workset(item) for item in dirty_worksets)
    except TypeError as error:
        raise TypeError("dirty_worksets must be a sequence") from error
    worksets = tuple(sorted(worksets, key=lambda item: item.request_index))
    if len(worksets) != batch_size or tuple(
        item.request_index for item in worksets
    ) != tuple(range(batch_size)):
        raise ValueError("dirty worksets must cover every request exactly once")

    expected_flat_offset = 0
    expected_state_bindings = []
    for request_index, workset in enumerate(worksets):
        expected_before = seq[request_index] - extend[request_index]
        if (
            workset.flat_row_offset != expected_flat_offset
            or workset.row_count != extend[request_index]
            or workset.seq_len_before != expected_before
        ):
            raise ValueError("dirty request workset disagrees with ragged geometry")
        previous_end = 0
        for island in workset.islands:
            if not (
                0 <= island.request_row_begin < island.request_row_end
                <= workset.row_count
            ):
                raise ValueError("dirty island request-row range is invalid")
            if island.request_row_begin < previous_end:
                raise ValueError("dirty islands overlap within one request")
            if (
                island.flat_begin
                != workset.flat_row_offset + island.request_row_begin
                or island.flat_end
                != workset.flat_row_offset + island.request_row_end
            ):
                raise ValueError("dirty island flat/request row ranges disagree")
            if (
                island.token_begin
                != workset.seq_len_before + island.request_row_begin
                or island.token_end
                != workset.seq_len_before + island.request_row_end
            ):
                raise ValueError("dirty island token/request row ranges disagree")
            span = island.request_row_end - island.request_row_begin
            if island.flat_end - island.flat_begin != span:
                raise ValueError("dirty island row spans disagree")
            expected_completions = tuple(
                island.flat_begin + offset
                for offset, token in enumerate(
                    range(island.token_begin, island.token_end)
                )
                if (token + 1) % ratio == 0
            )
            if island.completion_output_rows != expected_completions:
                raise ValueError("dirty island compressor completions are invalid")
            if island.token_begin > 0 and not island.state_slot_indices:
                raise ValueError("non-initial dirty island lacks restored state slots")
            if any(slot < 0 for slot in island.state_slot_indices):
                raise ValueError("dirty island state slots must be non-negative")
            expected_state_bindings.append(
                DirtyCompressorStateBinding(
                    request_index=request_index,
                    token_begin=island.token_begin,
                    state_slot_indices=island.state_slot_indices,
                )
            )
            previous_end = island.request_row_end
        expected_flat_offset += workset.row_count
    if expected_flat_offset != total_rows:
        raise ValueError("dirty request worksets do not tile packed x")

    receipt = _normalize_state_receipt(restored_state)
    if (
        receipt.layer_id != layer_id
        or receipt.compress_ratio != ratio
        or receipt.is_indexer != is_indexer
    ):
        raise ValueError("restored compressor state belongs to another domain")
    if not receipt.restore_token:
        raise ValueError("restored compressor state lacks a restore token")
    if forward_token is not None and receipt.forward_token != forward_token:
        raise ValueError("restored compressor state belongs to another forward")
    expected_bindings = tuple(
        sorted(
            expected_state_bindings,
            key=lambda item: (item.request_index, item.token_begin),
        )
    )
    if receipt.bindings != expected_bindings:
        raise ValueError("restored compressor state does not cover dirty islands")

    return DirtyCompressorPreflight(
        layer_id=layer_id,
        compress_ratio=ratio,
        is_indexer=is_indexer,
        total_rows=total_rows,
        worksets=worksets,
        state_receipt=receipt,
    )


class _PreparedDirtyCompressorIsland(NamedTuple):
    flat_begin: int
    flat_end: int
    metadata: FusedCompressMetadata
    target_loc: torch.Tensor


class CompressorBackendMixin:
    def get_paged_compress_metadata(self, compress_ratio: int) -> FusedCompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, FusedCompressMetadata)
        return metadata

    def _maybe_upgrade_forward_metadata(self) -> None:
        pass

    def _store_dirty_compressor_output(
        self,
        *,
        compressed: torch.Tensor,
        target_loc: torch.Tensor,
        layer_id: int,
        is_indexer: bool,
    ) -> None:
        """Store one already-validated dirty slice without touching clean rows."""

        token_to_kv_pool = self.token_to_kv_pool
        if is_indexer:
            if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
                token_to_kv_pool.set_index_k_fused(
                    layer_id=layer_id,
                    loc=target_loc,
                    cache_k=compressed,
                )
            else:
                compressed_fp8, compressed_scale = act_quant(compressed)
                token_to_kv_pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=target_loc,
                    index_k=compressed_fp8,
                    index_k_scale=compressed_scale,
                )
        elif envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=target_loc,
                cache_k=compressed,
            )
        else:
            pack = quant_to_nope_fp8_rope_bf16_pack_triton(
                compressed.bfloat16()
            )
            token_to_kv_pool.set_extra_key_buffer(layer_id, target_loc, pack)

    def _execute_dirty_compressor_islands(
        self,
        *,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
        is_indexer: bool,
        metadata_attr: str,
        islands: Sequence[_PreparedDirtyCompressorIsland],
    ) -> None:
        """Common online-island executor used by legacy and production paths.

        Every prepared slice is validated before the first metadata assignment.
        Compressor outputs are staged before cache stores, so an invalid output
        shape cannot partially overwrite C4/C128/Indexer cache rows.
        """

        prepared = tuple(islands)
        total_rows = int(x.shape[0])
        for item in prepared:
            if not isinstance(item, _PreparedDirtyCompressorIsland):
                raise TypeError("dirty compressor executor received an invalid island")
            if not 0 <= item.flat_begin < item.flat_end <= total_rows:
                raise ValueError("prepared dirty compressor slice is outside x")
            span = item.flat_end - item.flat_begin
            if item.target_loc.ndim != 1 or int(item.target_loc.numel()) != span:
                raise ValueError("prepared dirty target locations have wrong geometry")
            if not isinstance(item.metadata, FusedCompressMetadata):
                raise TypeError("prepared dirty compressor metadata is invalid")

        staged = []
        for item in prepared:
            setattr(self.forward_metadata, metadata_attr, item.metadata)
            compressed = compressor(
                x[item.flat_begin : item.flat_end],
                forward_batch,
                attn_backend=self,
            )
            span = item.flat_end - item.flat_begin
            if compressed.ndim < 1 or int(compressed.shape[0]) != span:
                raise RuntimeError(
                    "dirty-only compressor output does not match its island: "
                    f"output={tuple(compressed.shape)} rows={span}"
                )
            staged.append((compressed, item.target_loc))

        for compressed, target_loc in staged:
            self._store_dirty_compressor_output(
                compressed=compressed,
                target_loc=target_loc,
                layer_id=layer_id,
                is_indexer=is_indexer,
            )

    def _forward_compressor_dirty(
        self,
        *,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
        is_indexer: bool,
        dirty_worksets: Sequence[object],
        restored_state: object,
        target_loc: torch.Tensor,
        forward_token: Optional[str],
    ) -> None:
        """Production dirty-only path; clean cache/state is caller-owned."""

        if forward_batch.forward_mode.is_idle():
            if int(x.shape[0]) or int(target_loc.numel()):
                raise ValueError("idle dirty compressor received non-empty rows")
            return
        if not forward_batch.forward_mode.is_prefill():
            raise ValueError("dirty-only compressor supports prefill forwards only")
        if _dirty_int(compressor.layer_id, "compressor layer_id") != _dirty_int(
            layer_id, "layer_id"
        ):
            raise ValueError("dirty compressor module belongs to another layer")
        if bool(compressor.is_in_indexer) != is_indexer:
            raise ValueError("dirty compressor module belongs to another domain")
        if is_indexer and not is_overlap_compress(compressor.ratio):
            raise ValueError("Indexer dirty compressor requires C4")
        if target_loc.ndim != 1:
            raise ValueError("dirty compressor target_loc must be one-dimensional")
        if target_loc.device != x.device:
            raise ValueError("dirty compressor target_loc must be on x.device")
        if str(target_loc.dtype) not in ("torch.int32", "torch.int64"):
            raise TypeError("dirty compressor target_loc must use int32 or int64")
        if x.ndim < 1:
            raise ValueError("dirty compressor x must have a row dimension")

        batch_size = _dirty_int(forward_batch.batch_size, "batch_size")
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if seq_lens_cpu is None or extend_lens_cpu is None:
            raise ValueError("dirty compressor requires CPU ragged sequence lengths")
        req_pool_indices = forward_batch.req_pool_indices
        if req_pool_indices.ndim != 1 or int(req_pool_indices.numel()) != batch_size:
            raise ValueError("request-pool indices do not match dirty batch geometry")

        # Full geometry/state/target validation happens before lazy metadata
        # upgrade, metadata replacement, compressor execution, or cache store.
        preflight = preflight_dirty_compressor_geometry(
            layer_id=layer_id,
            compress_ratio=compressor.ratio,
            is_indexer=is_indexer,
            total_rows=int(x.shape[0]),
            batch_size=batch_size,
            seq_lens=seq_lens_cpu,
            extend_lens=extend_lens_cpu,
            dirty_worksets=dirty_worksets,
            restored_state=restored_state,
            target_loc_rows=int(target_loc.numel()),
            forward_token=forward_token,
        )
        if not preflight.islands:
            return

        token_to_kv_pool = self.token_to_kv_pool
        metadata_attr = f"c{compressor.ratio}_compress_metadata"
        prepared = []
        # Build *all* paged metadata before the first temporary metadata
        # assignment or cache mutation.  A later invalid request cannot leave
        # an earlier request partially committed.
        for request_index, island in preflight.islands:
            span_len = island.flat_end - island.flat_begin
            request_pool_index = req_pool_indices[
                request_index : request_index + 1
            ]
            if int(request_pool_index.numel()) != 1:
                raise ValueError("dirty island request-pool selection is invalid")
            seq_len_tensor = torch.tensor(
                [island.token_end], dtype=torch.int32, device=x.device
            )
            extend_len_tensor = torch.tensor(
                [span_len], dtype=torch.int32, device=x.device
            )
            metadata = create_paged_compressor_data(
                compress_ratio=compressor.ratio,
                is_prefill=True,
                token_to_kv_pool=token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=request_pool_index,
                seq_lens=seq_len_tensor,
                extend_lens=extend_len_tensor,
                seq_lens_cpu=[island.token_end],
                extend_lens_cpu=[span_len],
                num_q_tokens=span_len,
            )
            selected_target = target_loc[island.flat_begin : island.flat_end]
            if int(selected_target.numel()) != span_len:
                raise ValueError("dirty island target location slice is invalid")
            prepared.append(
                _PreparedDirtyCompressorIsland(
                    island.flat_begin,
                    island.flat_end,
                    metadata,
                    selected_target,
                )
            )

        # Only after every paged-data builder and target slice succeeded may
        # the backend replace/upgrade forward metadata.
        self._maybe_upgrade_forward_metadata()
        original_metadata = getattr(self.forward_metadata, metadata_attr)
        try:
            self._execute_dirty_compressor_islands(
                x=x,
                forward_batch=forward_batch,
                layer_id=layer_id,
                compressor=compressor,
                is_indexer=is_indexer,
                metadata_attr=metadata_attr,
                islands=prepared,
            )
        except Exception as error:
            # Clean cache/state has already been restored and may be consumed by
            # peer ranks.  Post-preflight failure is not a legal dense fallback.
            raise RuntimeError(
                f"dirty-only compressor failed closed at layer {layer_id}"
            ) from error
        finally:
            setattr(self.forward_metadata, metadata_attr, original_metadata)

    def forward_core_compressor_dirty(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
        *,
        dirty_worksets: Sequence[object],
        restored_state: object,
        target_loc: torch.Tensor,
        forward_token: Optional[str] = None,
    ) -> None:
        """Run the attention C4/C128 compressor only on dirty islands."""

        self._forward_compressor_dirty(
            x=x,
            forward_batch=forward_batch,
            layer_id=layer_id,
            compressor=compressor,
            is_indexer=False,
            dirty_worksets=dirty_worksets,
            restored_state=restored_state,
            target_loc=target_loc,
            forward_token=forward_token,
        )

    def forward_indexer_compressor_dirty(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
        *,
        dirty_worksets: Sequence[object],
        restored_state: object,
        target_loc: torch.Tensor,
        forward_token: Optional[str] = None,
    ) -> None:
        """Run the C4 Indexer compressor only on dirty islands."""

        self._forward_compressor_dirty(
            x=x,
            forward_batch=forward_batch,
            layer_id=layer_id,
            compressor=compressor,
            is_indexer=True,
            dirty_worksets=dirty_worksets,
            restored_state=restored_state,
            target_loc=target_loc,
            forward_token=forward_token,
        )

    def _forward_redknot_segmented_compressor(
        self,
        *,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
        is_indexer: bool,
    ) -> bool:
        if os.environ.get("REDKNOT_V4_SEGMENTED_COMPRESSOR", "0") != "1":
            return False
        placeholder_only = bool(
            getattr(forward_batch, "redknot_placeholder_only", False)
        )
        schedules = getattr(
            forward_batch, "redknot_v4_compressor_schedules", None
        )
        plans = getattr(forward_batch, "redknot_reuse_plan", None)
        plan = plans[0] if plans and len(plans) == 1 else None
        if not schedules or not plan or plan.get("mode") != "restore":
            return False
        if forward_batch.batch_size != 1 or not plan.get("segments"):
            return False

        from sglang.srt.layers.attention.redknot.dsv4_offline_reuse_v2 import (
            compute_compressed_slots,
            compute_paged_compressed_slots,
            get_offline_reuse_controller_v2,
            select_terminal_compress_state_slots,
        )
        from sglang.srt.layers.attention.redknot.v4.segmented_compressor import (
            CompressorEventKind,
            select_segment_output_locations,
            validate_complete_online_row_coverage,
        )

        ratio = compressor.ratio
        schedule = schedules[ratio]
        if layer_id == 2 and is_indexer:
            total_rows = int(x.shape[0])
            online_rows = int(schedule.online_rows)
            logger.info(
                "RedKnot segmented compressor: chunk=[%d,%d) online_rows=%d "
                "skipped_rows=%d ratio=%d",
                schedule.chunk_token_range[0],
                schedule.chunk_token_range[1],
                online_rows,
                max(0, total_rows - online_rows),
                ratio,
            )
        segments = plan["segments"]
        token_to_kv_pool = self.token_to_kv_pool
        metadata_attr = f"c{ratio}_compress_metadata"
        original_metadata = getattr(self.forward_metadata, metadata_attr)
        ctrl = get_offline_reuse_controller_v2()

        try:
            if bool(
                getattr(forward_batch, "redknot_rows_pruned", False)
            ) and not placeholder_only:
                validate_complete_online_row_coverage(
                    schedule, total_rows=int(x.shape[0])
                )

            # Materialize offline state first.  ONLINE_RANGE events below then
            # overwrite the selected boundary/hot blocks, giving online refresh
            # deterministic precedence.  The previous order compressed online
            # rows and subsequently restored full offline blocks on top of them,
            # making interior "hot" work invisible to the final Query.
            inject_full = bool(plan.get("inject_full_blocks", False))
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
                offset = int(segment["global_offset"])
                length = int(segment["length"])
                logical_begin, logical_end = (
                    tuple(map(int, original_chunk_range))
                    if original_chunk_range is not None
                    else schedule.chunk_token_range
                )
                local_restore_begin = max(0, logical_begin - offset)
                visible_tokens = max(0, min(length, logical_end - offset))
                if visible_tokens <= local_restore_begin:
                    continue
                online_prefix = int(selected_prefixes[segment_index])
                skip_first = max(
                    online_prefix,
                    0 if inject_full else int(segment.get("skip_first", 128)),
                )
                if visible_tokens <= skip_first:
                    continue
                page_table = self.forward_metadata.core_metadata.page_table
                extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
                slots = compute_paged_compressed_slots(
                    page_table=page_table,
                    req_idx=0,
                    seq_len=visible_tokens,
                    compress_ratio=ratio,
                    compressed_page_size=extra_page_size,
                    token_offset=offset,
                )
                extra_buffer = token_to_kv_pool.get_extra_key_buffer(layer_id)
                if ratio == 4:
                    ctrl.restore_c4_layer(
                        seg_hash=str(segment["seg_hash"]),
                        layer_id=layer_id,
                        c4_buffer=extra_buffer,
                        dst_slots=slots,
                        global_offset=offset,
                        freqs_cis=self._redknot_freqs_cis_for_layer(layer_id),
                        c4_page_size=extra_page_size,
                        indexer_buffer=(
                            token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
                            if is_indexer
                            else None
                        ),
                        indexer_slots=slots if is_indexer else None,
                        indexer_page_size=token_to_kv_pool.get_index_k_page_size(),
                        skip_tokens=skip_first,
                        restore_begin_tokens=local_restore_begin,
                        max_tokens=visible_tokens,
                        restore_c4=not is_indexer,
                        restore_indexer=is_indexer,
                    )
                elif not is_indexer:
                    ctrl.restore_c128_layer(
                        seg_hash=str(segment["seg_hash"]),
                        layer_id=layer_id,
                        c128_buffer=extra_buffer,
                        dst_slots=slots,
                        global_offset=offset,
                        freqs_cis=self._redknot_freqs_cis_for_layer(layer_id),
                        c128_page_size=extra_page_size,
                        skip_tokens=skip_first,
                        restore_begin_tokens=local_restore_begin,
                        max_tokens=visible_tokens,
                    )

            for event in schedule.events:
                if event.kind == CompressorEventKind.CHECKPOINT_RESTORE:
                    segment = segments[event.segment_index]
                    offset = int(segment["global_offset"])
                    anchor = int(event.checkpoint_anchor)
                    state_pool = compressor.get_state_pool(self)
                    full_slots = self.req_to_token[
                        int(forward_batch.req_pool_indices[0].item()),
                        offset : offset + anchor,
                    ]
                    state_slots = compute_compressed_slots(
                        full_slots=full_slots,
                        full_to_swa=token_to_kv_pool.full_to_swa_index_mapping,
                        swa_page_size=token_to_kv_pool.swa_page_size,
                        ring_size=state_pool.ring_size,
                        compress_ratio=ratio,
                        seq_len=anchor,
                        state_group_width=min(ratio, state_pool.ring_size),
                    )
                    destination_slots = select_terminal_compress_state_slots(
                        state_slots, ratio
                    )
                    restored = ctrl.restore_compress_checkpoint(
                        seg_hash=str(segment["seg_hash"]),
                        layer_id=layer_id,
                        checkpoint_anchor=anchor,
                        state_buffer=state_pool.kv_score_buffer.kv_score,
                        dst_slots=destination_slots,
                        is_indexer=is_indexer,
                        state_group_width=min(ratio, state_pool.ring_size),
                    )
                    if restored != int(destination_slots.numel()):
                        raise ValueError(
                            f"missing compressor checkpoint at anchor {anchor}"
                        )
                    continue
                if event.kind == CompressorEventKind.OFFLINE_TAIL:
                    segment = segments[event.segment_index]
                    offset = int(segment["global_offset"])
                    length = int(segment["length"])
                    online_owns_tail = any(
                        int(island["segment_index"]) == event.segment_index
                        and int(island["global_end"]) >= offset + length
                        for island in checkpoint_islands
                    )
                    if (
                        int(selected_prefixes[event.segment_index]) >= length
                        or online_owns_tail
                    ):
                        # A fully online segment owns its terminal compressor state.
                        continue
                    state_pool = compressor.get_state_pool(self)
                    full_slots = self.req_to_token[
                        int(forward_batch.req_pool_indices[0].item()),
                        offset : offset + length,
                    ]
                    state_slots = compute_compressed_slots(
                        full_slots=full_slots,
                        full_to_swa=token_to_kv_pool.full_to_swa_index_mapping,
                        swa_page_size=token_to_kv_pool.swa_page_size,
                        ring_size=state_pool.ring_size,
                        compress_ratio=ratio,
                        seq_len=length,
                        state_group_width=min(ratio, state_pool.ring_size),
                    )
                    if state_slots.numel() > 0:
                        terminal_state_slots = select_terminal_compress_state_slots(
                            state_slots, ratio
                        )
                        ctrl.restore_compress_state(
                            seg_hash=str(segment["seg_hash"]),
                            layer_id=layer_id,
                            state_buffer=state_pool.kv_score_buffer.kv_score,
                            dst_slots=terminal_state_slots,
                            is_indexer=is_indexer,
                            state_group_width=min(ratio, state_pool.ring_size),
                        )
                    continue

                if placeholder_only:
                    # The row only keeps the transformer pipeline alive.  Its
                    # compressor output must not mutate any logical cache/state;
                    # offline materialization above and OFFLINE_TAIL events are
                    # the complete work for this chunk.
                    continue

                row_begin, row_end = event.row_begin, event.row_end
                if row_begin >= row_end:
                    continue
                span_len = row_end - row_begin
                seq_end = event.token_end
                core_metadata = self.forward_metadata.core_metadata
                all_output_locations = (
                    core_metadata.c4_out_loc
                    if ratio == 4
                    else core_metadata.c128_out_loc
                )
                segment_output_locations = select_segment_output_locations(
                    all_output_locations,
                    row_begin=row_begin,
                    row_end=row_end,
                    total_rows=int(x.shape[0]),
                )
                segment_metadata = create_paged_compressor_data(
                    compress_ratio=ratio,
                    is_prefill=True,
                    token_to_kv_pool=token_to_kv_pool,
                    req_to_token=self.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=torch.tensor(
                        [seq_end], dtype=torch.int32, device=x.device
                    ),
                    extend_lens=torch.tensor(
                        [span_len], dtype=torch.int32, device=x.device
                    ),
                    seq_lens_cpu=[seq_end],
                    extend_lens_cpu=[span_len],
                    num_q_tokens=span_len,
                )
                self._execute_dirty_compressor_islands(
                    x=x,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    compressor=compressor,
                    is_indexer=is_indexer,
                    metadata_attr=metadata_attr,
                    islands=(
                        _PreparedDirtyCompressorIsland(
                            row_begin,
                            row_end,
                            segment_metadata,
                            segment_output_locations,
                        ),
                    ),
                )

            completed = getattr(
                forward_batch, "redknot_segmented_compressor_completed", None
            )
            if completed is None:
                completed = set()
                forward_batch.redknot_segmented_compressor_completed = completed
            completed.add((int(layer_id), int(ratio), bool(is_indexer)))
            return True
        except Exception as error:
            logger.warning(
                "RedKnot segmented compressor fell back to dense at layer %d: %s",
                layer_id,
                error,
            )
            if bool(getattr(forward_batch, "redknot_rows_pruned", False)):
                # Rows have already been removed from the batch.  Continuing with
                # a layer-local dense compressor cannot reconstruct those rows and
                # would silently consume uninitialized cache state.
                raise RuntimeError(
                    f"RedKnot selected-row compressor failed at layer {layer_id}"
                ) from error
            return False
        finally:
            setattr(self.forward_metadata, metadata_attr, original_metadata)

    def forward_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        rotate: bool,
        forward_batch: ForwardBatch,
        compress_ratio: int,
        is_paged: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

        assert compress_ratio in (
            4,
            128,
        ), f"DSV4 supports CSA(4x) and HCA(128x) only, got {compress_ratio=}"
        if is_paged:
            metadata = self.get_paged_compress_metadata(compress_ratio)
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
            else:
                last_dim = 2 * head_dim * coff
                assert kv_score_buffer.shape[-1] == last_dim
                kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
        else:
            plan = make_compressor_plan(compress_ratio, forward_batch)
            metadata = (forward_batch.req_pool_indices.to(torch.int32), None, plan)
        indices, extra_data, plan = metadata

        if _is_hip:
            if not is_paged:
                raise NotImplementedError("HIP fused compressor expects paged metadata")

            from sglang.srt.layers.attention.dsv4.fused_compress_triton import (
                hip_compress_forward,
                hip_compress_fused_norm_rope_inplace,
            )

            kv_compressed = hip_compress_forward(
                kv_score_buffer=kv_score_buffer,
                kv_score_input=kv_score_input,
                ape=ape,
                indices=indices,
                plan=plan,
                compress_ratio=compress_ratio,
                head_dim=head_dim,
                extra_data=extra_data,
            )
            norm_eps = (
                norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps
            )
            hip_compress_fused_norm_rope_inplace(
                kv_compressed,
                norm.weight,
                norm_eps,
                freqs_cis_cache,
                plan,
            )
            return rotate_activation(kv_compressed) if rotate else kv_compressed

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape,
            indices=indices,
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            extra_data=extra_data,
        )
        compress_fused_norm_rope_inplace(
            kv_compressed,
            norm.weight,
            norm.variance_epsilon,
            freqs_cis_cache,
            plan,
        )
        return rotate_activation(kv_compressed) if rotate else kv_compressed

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # Upgrade lazy metadata before the segmented path reads or replaces it.
        self._maybe_upgrade_forward_metadata()
        if self._forward_redknot_segmented_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=layer_id,
            compressor=compressor,
            is_indexer=False,
        ):
            return
        # PREP_IN_CG lazy upgrade: the concrete backend (DeepseekV4AttnBackend)
        # owns this helper. MQALayer._forward_prepare calls us before
        # attn_backend.forward(), so Raw -> DSV4Metadata must happen here too
        # (e.g. 1.6T layer 0 has compress_ratio=128 and needs cX_compress_metadata).
        token_to_kv_pool = self.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch, attn_backend=self)
        core_metadata = self.forward_metadata.core_metadata
        out_loc = (
            core_metadata.c4_out_loc
            if compressor.ratio == 4
            else core_metadata.c128_out_loc
        )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            pack = quant_to_nope_fp8_rope_bf16_pack_triton(new_compressed_kv.bfloat16())
            token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

    def forward_indexer_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        assert is_overlap_compress(compressor.ratio)
        # PREP_IN_CG lazy upgrade (see forward_core_compressor for rationale).
        self._maybe_upgrade_forward_metadata()
        if self._forward_redknot_segmented_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=layer_id,
            compressor=compressor,
            is_indexer=True,
        ):
            return
        token_to_kv_pool = self.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch, attn_backend=self)
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            new_compressed_kv_fp8, new_compressed_kv_scale = act_quant(
                new_compressed_kv
            )
            token_to_kv_pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                index_k=new_compressed_kv_fp8,
                index_k_scale=new_compressed_kv_scale,
            )


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def make_compressor_plan(
    compress_ratio: Literal[4, 128],
    forward_batch: ForwardBatch,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if forward_batch.forward_mode.is_decode():
        seq_lens_32 = forward_batch.seq_lens.to(torch.int32)
        return CompressorDecodePlan(compress_ratio, seq_lens_32)
    if forward_batch.forward_mode.is_prefill():
        assert not forward_batch.forward_mode.is_target_verify()
        extend_lens_list = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert extend_lens_list is not None and seq_lens_cpu is not None
        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            num_q_tokens=sum(extend_lens_list),
            seq_lens=seq_lens_cpu,
            extend_lens=torch.tensor(extend_lens_list),
            device=forward_batch.seq_lens.device,
        )
    elif forward_batch.forward_mode.is_target_verify():
        raise NotImplementedError("target verify mode to be implemented")
    else:
        raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> FusedCompressMetadata:
    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # assert ring_size % compress_ratio == 0

    def clip_down(positions: torch.Tensor) -> torch.Tensor:
        return positions // compress_ratio * compress_ratio

    def get_raw_loc(positions: torch.Tensor) -> torch.Tensor:
        positions = positions.masked_fill(positions < 0, 0)
        loc = req_to_token[req_pool_indices, positions]
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(loc)
        swa_pages = swa_loc // swa_page_size
        state_loc = swa_pages * ring_size + swa_loc % ring_size
        return (state_loc // compress_ratio).to(torch.int32)

    is_overlap = is_overlap_compress(compress_ratio)

    if is_prefill:
        assert extend_lens is not None
        write_loc, extra_data = triton_create_paged_compress_data(
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
        )

        plan_kwargs: dict
        if seq_lens_cpu is None:
            assert num_q_tokens is not None
            plan_kwargs = dict(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
            )
        else:
            assert extend_lens_cpu is not None
            plan_kwargs = dict(
                num_q_tokens=sum(extend_lens_cpu),
                seq_lens=torch.tensor(seq_lens_cpu),
                extend_lens=torch.tensor(extend_lens_cpu),
            )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            device=seq_lens.device,
            use_cuda_graph=use_prefill_cuda_graph,
            **plan_kwargs,
        )
    else:
        write_positions = clip_down(seq_lens - 1)
        write_loc = get_raw_loc(write_positions)
        if is_overlap:
            write_overlap_loc = get_raw_loc(write_positions - compress_ratio)
            extra_data = write_overlap_loc.view(-1, 1)
        elif _is_hip:
            extra_data = get_raw_loc(write_positions - compress_ratio)
        else:
            extra_data = None
        plan = CompressorDecodePlan(compress_ratio, seq_lens.to(torch.int32))

    return FusedCompressMetadata(write_loc=write_loc, extra_data=extra_data, plan=plan)


class Compressor(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        freqs_cis: torch.Tensor,
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        self.coff = coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        self.norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis

        self.ape_converted = False

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        if self.overlap:
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            ape = torch.cat([ape[0], ape[1]], dim=0)
            self.ape.data.copy_(ape.view(self.ratio, -1))

    def get_state_pool(self, attn_backend: AttentionBackend) -> CompressStatePool:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)
        assert isinstance(ret, CompressStatePool)
        return ret

    def compute_kv_score(self, x: torch.Tensor, forward_batch: ForwardBatch):
        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)

        # CUDA path: delegate to backend
        if dsa_use_prefill_cp(forward_batch):
            kv_score = cp_all_gather_rerange_output(
                kv_score,
                get_attention_cp_size(),
                forward_batch,
                torch.cuda.current_stream(),
            )
        return kv_score

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        kv_score = self.compute_kv_score(x, forward_batch)

        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4AttnBackend)
        kv_score_buffer = self.get_state_pool(attn_backend).kv_score_buffer.kv_score
        return attn_backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )


if _is_hip and not envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
    from sglang.srt.layers.attention.dsv4.compress_hip import (  # noqa: F811
        CompressorHip as Compressor,
    )
