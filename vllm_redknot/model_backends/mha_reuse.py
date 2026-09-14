"""REDKNOT-MIGRATION: Mistral SWA replay and MHA policy assets, not runner hooks.

Source: RedKnot 55ee4e8401603f8d2612877e4053e18b37b1c1bd,
attention/redknot/driver_batched.py::run_redknot_swa_offlinekv.
The source algorithm is expressed through explicit callbacks, not a copied
Transformers model/generation driver. See docs/mha_migration_provenance.json.
No engine registration, model loading, CUDA initialization or dependency import
occurs here. RUNTIME_INTEGRATED is deliberately false.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

RUNTIME_INTEGRATED = False
SOURCE_REVISION = "55ee4e8401603f8d2612877e4053e18b37b1c1bd"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class SWAReusePlan:
    document_lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    boundary_lengths: tuple[int, ...]
    sliding_window: int

    @property
    def query_position(self) -> int:
        return sum(self.document_lengths)

    @property
    def replay_tokens(self) -> int:
        return sum(self.boundary_lengths)


def plan_swa_reuse(
    document_lengths: Sequence[int],
    *,
    sliding_window: int,
    recompute_ratio: float | None = 0.20,
    recompute_prefix: int = 3000,
) -> SWAReusePlan:
    """Plan native-SWA boundary replay; document zero is never recomputed.

    The 20% default is the source Mistral benchmark policy, not an exactness
    proof. Its boundary can be shorter than the model's effective receptive
    field. A native sliding window must come from the actual checkpoint.
    """
    _integer(sliding_window, "sliding_window", 1)
    lengths = tuple(document_lengths)
    if not lengths:
        raise ValueError("at least one document is required")
    for length in lengths:
        _integer(length, "document length", 1)
    if recompute_ratio is None:
        _integer(recompute_prefix, "recompute_prefix", 1)
    elif (
        type(recompute_ratio) not in (float, int)
        or not math.isfinite(recompute_ratio)
        or not 0 < recompute_ratio <= 1
    ):
        raise ValueError("recompute_ratio must be finite and in (0, 1]")
    offsets, boundaries, offset = [], [], 0
    for index, length in enumerate(lengths):
        offsets.append(offset)
        boundaries.append(
            0
            if index == 0
            else (
                min(recompute_prefix, length)
                if recompute_ratio is None
                else max(1, int(length * recompute_ratio))
            )
        )
        offset += length
    return SWAReusePlan(lengths, tuple(offsets), tuple(boundaries), sliding_window)


@dataclass(frozen=True)
class BoundaryReplay:
    """Inputs to a caller-owned native forward; prior KV is an owned copy.

    Return one (K, V) pair per layer containing ONLY these boundary tokens.
    The callback must enforce native SWA and return post-RoPE keys at positions
    start_position .. start_position + len(token_ids) - 1. It must not generate
    tokens, reinterpret a source position, or return the entire prefix cache.
    """

    document_index: int
    start_position: int
    token_ids: tuple[int, ...]
    prior_layer_kv: tuple[tuple[Any, Any], ...]
    sliding_window: int


@dataclass(frozen=True)
class SWAReuseResult:
    """Full logical assembled KV; native query/decode MUST enforce SWA.

    This migration does not install pages or a native sliding cache. The caller
    owns slot mapping, bounded decode storage and model/precision compatibility.
    Complete retained documents here are not a GPU memory-saving claim.
    """

    document_layer_kv: tuple[tuple[tuple[Any, Any], ...], ...]
    query_layer_kv: tuple[tuple[Any, Any], ...]
    query_position: int
    sliding_window: int


def _check_pair(pair: Any, token_count: int, reference: Any = None) -> None:
    import torch

    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("a layer must contain one (K, V) pair")
    key, value = pair
    if not all(isinstance(tensor, torch.Tensor) for tensor in pair):
        raise ValueError("KV payload must contain tensors")
    if (
        key.ndim != 4
        or key.shape[0] != 1
        or key.shape[2] != token_count
        or min(key.shape) <= 0
        or value.shape != key.shape
        or value.dtype != key.dtype
        or value.device != key.device
        or not key.is_floating_point()
    ):
        raise ValueError(
            "KV requires matching floating [1, heads, tokens, dim] tensors"
        )
    if reference is not None:
        expected = reference[0]
        if (
            (key.shape[0], key.shape[1], key.shape[3])
            != (expected.shape[0], expected.shape[1], expected.shape[3])
            or key.dtype != expected.dtype
            or key.device != expected.device
        ):
            raise ValueError("layer KV shape, dtype or device changed")


def replay_swa_documents(
    plan: SWAReusePlan,
    chunk_token_ids: Sequence[Sequence[int]],
    document_layer_kv: Sequence[Sequence[tuple[Any, Any]]],
    *,
    reposition_key: Callable[..., Any],
    forward_boundary: Callable[[BoundaryReplay], Sequence[tuple[Any, Any]]],
) -> SWAReuseResult:
    """Relocate offline K, replay later boundaries and splice prefix/suffix KV.

    reposition_key(key, *, layer_index, document_index, src_start, dst_start)
    must perform the CHECKPOINT'S RoPE relocation and preserve tensor layout.
    No assumed static-RoPE formula is applied here. All callbacks receive owned
    tensor copies; callback failure never publishes a partial result or changes
    caller input. This is a tensor reference/data-movement contract, not a fast
    path, trusted cache-identity check, or native vLLM adapter.
    """
    import torch

    if not isinstance(plan, SWAReusePlan):
        raise ValueError("an explicit SWAReusePlan is required")
    expected = plan_swa_reuse(
        plan.document_lengths, sliding_window=plan.sliding_window, recompute_ratio=1
    )
    if (
        plan.offsets != expected.offsets
        or len(plan.boundary_lengths) != len(plan.document_lengths)
        or plan.boundary_lengths[0] != 0
    ):
        raise ValueError("invalid plan offsets or boundary layout")
    for offset in plan.offsets:
        _integer(offset, "document offset")
    for index, (length, count) in enumerate(
        zip(plan.document_lengths, plan.boundary_lengths, strict=True)
    ):
        _integer(count, "boundary length", 0 if index == 0 else 1)
        if count > length:
            raise ValueError("boundary exceeds document length")
    ids = tuple(tuple(tokens) for tokens in chunk_token_ids)
    documents = tuple(tuple(layers) for layers in document_layer_kv)
    if len(ids) != len(plan.document_lengths) or len(documents) != len(ids):
        raise ValueError("document count differs from plan")
    layer_count = len(documents[0])
    if not layer_count:
        raise ValueError("at least one KV layer is required")
    for index, (tokens, layers) in enumerate(zip(ids, documents, strict=True)):
        if len(tokens) != plan.document_lengths[index] or len(layers) != layer_count:
            raise ValueError("token or layer count differs from plan")
        for token in tokens:
            _integer(token, "token ID")
        for layer_index, pair in enumerate(layers):
            _check_pair(pair, len(tokens), documents[0][layer_index])

    def concatenate(previous):
        return tuple(
            tuple(
                torch.cat([doc[layer][side] for doc in previous], dim=2)
                for side in (0, 1)
            )
            for layer in range(layer_count)
        )

    with torch.no_grad():
        assembled = []
        for index, layers in enumerate(documents):
            moved = []
            for layer_index, (key, value) in enumerate(layers):
                key, value = key.detach().clone(), value.detach().clone()
                if plan.offsets[index]:
                    key = reposition_key(
                        key,
                        layer_index=layer_index,
                        document_index=index,
                        src_start=0,
                        dst_start=plan.offsets[index],
                    )
                pair = (key, value)
                _check_pair(pair, plan.document_lengths[index], layers[layer_index])
                moved.append(tuple(tensor.detach().clone() for tensor in pair))
            assembled.append(tuple(moved))

        for index in range(1, len(assembled)):
            count = plan.boundary_lengths[index]
            request = BoundaryReplay(
                index,
                plan.offsets[index],
                ids[index][:count],
                concatenate(assembled[:index]),
                plan.sliding_window,
            )
            corrected = tuple(forward_boundary(request))
            if len(corrected) != layer_count:
                raise ValueError("boundary callback returned wrong layer count")
            replaced = []
            for layer_index, pair in enumerate(corrected):
                _check_pair(pair, count, assembled[index][layer_index])
                replaced.append(
                    tuple(
                        torch.cat(
                            [
                                pair[side],
                                assembled[index][layer_index][side][:, :, count:],
                            ],
                            dim=2,
                        ).detach()
                        for side in (0, 1)
                    )
                )
            assembled[index] = tuple(replaced)
        return SWAReuseResult(
            tuple(assembled),
            concatenate(assembled),
            plan.query_position,
            plan.sliding_window,
        )


def build_head_class_policy(
    source: Mapping[str, Any],
    *,
    total_context_tokens: int,
    window_ratio: float | None = None,
    window_min: int = 256,
    fixed_window: int | None = None,
    merge_retrieval_to_global: bool = True,
) -> Any:
    """Reuse portable HeadClassConfig, including legacy local_full semantics.

    Source JSON matrices, not historical prose/summary fields, are authoritative.
    This does NOT use/import the active vLLM head-only config or enable a model.
    """
    from ..core.head_config import HeadClassConfig

    _integer(total_context_tokens, "total_context_tokens", 1)
    _integer(window_min, "window_min", 1)
    if type(merge_retrieval_to_global) is not bool:
        raise ValueError("merge_retrieval_to_global must be boolean")
    data = deepcopy(dict(source))
    config = HeadClassConfig(
        head_class=data["kv_head_classification"],
        head_max_distance=data["kv_head_max_distance"],
        head_sink_size=data.get("kv_head_sink_size"),
        num_layers=data["num_layers"],
        num_kv_heads=data["num_kv_heads"],
        retrieval_top_p=float(data.get("retrieval_top_p", 0.9)),
        dense_prefix_layers=int(data.get("dense_prefix_layers", 0)),
    )
    if merge_retrieval_to_global:
        config.merge_retrieval_to_global()
    if fixed_window is not None:
        config.set_local_window(_integer(fixed_window, "fixed_window", 1))
    elif window_ratio is not None:
        if (
            type(window_ratio) not in (int, float)
            or not math.isfinite(window_ratio)
            or window_ratio <= 0
        ):
            raise ValueError("window_ratio must be finite and positive")
        config.set_local_window(
            max(window_min, int(total_context_tokens * window_ratio))
        )
    return config


def build_sparse_ffn_policy(source: Mapping[str, Any]) -> Any:
    """Preserve the portable SparseFFNSchedule without adding an MLP hook.

    Llama's local_window is head-policy metadata, not an FFN schedule field.
    apply_sparse_ffn and its selector remain in core; native vLLM FFN still runs.
    """
    from ..core.sparse_ffn import SparseFFNSchedule

    data = deepcopy(dict(source))
    data.pop("local_window", None)
    allowed = {field.name for field in fields(SparseFFNSchedule)}
    if set(data) - allowed:
        raise ValueError("unknown sparse FFN policy fields")
    return SparseFFNSchedule(**data)
