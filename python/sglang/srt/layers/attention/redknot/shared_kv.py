"""Persistent shared KV ownership for RedKnot's ordinary backend entry points.

An integration supplies one stable handle per request, creates/forks/releases
those handles with its request lifecycle, and supplies projected, position-
encoded Q/K/V. No dense SGLang KV pool or request-slot identity is consulted.

This is an eager MHA/GQA adapter. Each request/layer append is atomic across its
physical KV heads. A whole batch or model forward is not a transaction: after a
runtime failure, the caller must abort affected requests (or restore a checkpoint)
before retrying. Preflight validation covers the entire batch before any append.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch

from sglang.srt.mem_cache.head_kv.manager import SegmentWrite
from sglang.srt.mem_cache.head_kv.segpaged import ManagedSegPagedKVCache


LIVE_SEGMENT = "__redknot_live__"


@dataclass(frozen=True)
class SharedKVRequestHandle:
    """Scheduler-owned request identity, fenced against request-ID reuse."""

    request_id: str
    generation: str


class SharedKVBackend:
    """Explicit request lifecycle and direct paged attention over a bounded pool.

    The caller's contract must identify model weights, adapters, KV format and
    position semantics; its namespace must identify the isolation domain. A fork
    promises that the child has exactly the parent's already-computed context.
    Arbitrary-context reuse requires the manager's explicit validity-proof API.
    """

    def __init__(self, manager):
        self.manager = manager
        self._views = {}

    def _version(self, handle):
        if not isinstance(handle, SharedKVRequestHandle):
            raise TypeError("explicit SharedKVRequestHandle required")
        try:
            version = self.manager.version(handle.request_id)
        except KeyError as exc:
            raise RuntimeError("stale shared KV request handle") from exc
        if version.generation != handle.generation:
            raise RuntimeError("stale shared KV request generation")
        return version

    @staticmethod
    def _handle(version):
        return SharedKVRequestHandle(version.request_id, version.generation)

    def create_request(self, request_id, *, context_id, contract, namespace):
        """Create an empty request; no implicit default model/isolation domain."""
        with self.manager._lock:
            return self._handle(self.manager.create_request(
                request_id, context_id=context_id, contract=contract,
                namespace=namespace,
            ))

    def fork_request(self, source, target_id):
        """Share sealed pages; future append/repair detaches changed pages."""
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("nonempty target request ID required")
        with self.manager._lock:
            self._version(source)
            return self._handle(self.manager.fork(source.request_id, target_id))

    def release_request(self, handle):
        with self.manager._lock:
            self._version(handle)
            self.manager.release_request(handle.request_id)
            self._views = {key: value for key, value in self._views.items()
                           if key[0] != handle}

    def repair_request(self, handle, patches):
        """Atomically repair heads; caller owns upstream invalidation closure."""
        with self.manager._lock:
            version = self._version(handle)
            return self.manager.update(handle.request_id, patches=tuple(patches),
                                       expected_epoch=version.epoch)

    @staticmethod
    def _integers(value, name):
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be a one-dimensional integer tensor")
            value = value.detach().cpu().tolist()
        if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in value
        ):
            raise ValueError(f"{name} must be an explicit integer sequence")
        return tuple(int(item) for item in value)

    @classmethod
    def _lengths(cls, batch, name):
        value = getattr(batch, name, None)
        if value is None:
            value = getattr(batch, name + "_cpu", None)
        return cls._integers(value, name)

    @staticmethod
    def _heads(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _tensor(value, name, heads, dim, pool):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be provided; cross-layer KV aliases are unsupported")
        if value.layout != torch.strided or value.ndim not in (2, 3):
            raise ValueError(f"{name} must be [tokens,heads*dim] or [tokens,heads,dim]")
        if (value.ndim == 2 and value.shape[1] != heads * dim) or (
            value.ndim == 3 and value.shape[1:] != (heads, dim)
        ):
            raise ValueError(f"{name} head shape does not match the layer")
        if value.device != pool.device or value.dtype != pool.dtype:
            raise ValueError(f"{name} must match the shared KV pool device and dtype")
        if value.requires_grad:
            raise ValueError("shared KV attention is inference-only")
        return value.reshape(value.shape[0], heads, dim)

    def _preflight(self, q, k, v, layer, batch, decode, save_kv_cache, windows, sinks):
        if not isinstance(decode, bool) or save_kv_cache is not True:
            raise ValueError("shared KV forward requires an explicit mode and cache writes")
        if getattr(layer, "is_cross_attention", False):
            raise ValueError("shared KV backend does not support cross-attention")
        attn_type = getattr(layer, "attn_type", "decoder")
        if getattr(attn_type, "value", attn_type) != "decoder":
            raise ValueError("shared KV backend requires causal decoder attention")
        if getattr(layer, "pos_encoding_mode", "NONE") != "NONE":
            raise ValueError("Q/K position encoding must be applied before shared KV attention")
        if getattr(layer, "use_irope", False):
            raise ValueError("iRoPE attention is unsupported by shared KV")
        if getattr(layer, "logit_cap", 0) != 0:
            raise ValueError("shared KV backend does not support logit capping")
        if getattr(layer, "xai_temperature_len", -1) not in (None, -1):
            raise ValueError("shared KV backend does not support position-dependent temperature")
        if any(getattr(layer, field, None) is not None for field in (
            "k_scale", "v_scale", "k_scale_float", "v_scale_float", "quant_method",
        )):
            raise ValueError("quantized/scaled KV is unsupported by shared KV")
        if getattr(batch, "encoder_lens", None) is not None:
            raise ValueError("encoder batches are unsupported by shared KV")
        if getattr(batch, "spec_info", None) is not None:
            raise ValueError("speculative/tree batches are unsupported by shared KV")
        if getattr(batch, "mrope_positions", None) is not None:
            raise ValueError("multiaxis positions are unsupported by shared KV")
        mode = getattr(getattr(batch, "forward_mode", None), "name", None)
        if mode is not None and mode not in (("DECODE",) if decode else ("EXTEND", "MIXED")):
            raise ValueError("unsupported shared KV forward mode")
        if any(getattr(batch, field, False) is True for field in (
            "is_cuda_graph_capture", "is_cuda_graph_replay", "is_cuda_graph",
        )) or (self.manager.pool.device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            raise ValueError("shared KV backend requires eager execution, not CUDA graphs")

        heads = self._heads(layer.tp_q_head_num, "query heads")
        kv_heads = self._heads(layer.tp_k_head_num, "KV heads")
        dim = self._heads(layer.qk_head_dim, "head dimension")
        if heads % kv_heads or getattr(layer, "tp_v_head_num", kv_heads) != kv_heads:
            raise ValueError("shared KV backend requires uniform MHA/GQA head grouping")
        if layer.v_head_dim != dim or self.manager.pool.head_dim != dim:
            raise ValueError("shared KV backend requires equal Q/K/V head dimensions")
        layer_id = getattr(layer, "layer_id", None)
        if isinstance(layer_id, bool) or not isinstance(layer_id, Integral) or layer_id < 0:
            raise ValueError("nonnegative layer ID required")
        scale = getattr(layer, "scaling", None)
        if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
            raise ValueError("finite attention scaling required")
        q = self._tensor(q, "Q", heads, dim, self.manager.pool)
        k = self._tensor(k, "K", kv_heads, dim, self.manager.pool)
        v = self._tensor(v, "V", kv_heads, dim, self.manager.pool)
        if len(q) != len(k) or len(q) != len(v):
            raise ValueError("Q/K/V token counts must match")

        handles = getattr(batch, "redknot_shared_kv_handles", None)
        if not isinstance(handles, (tuple, list)):
            raise ValueError("one explicit shared KV handle per request is required")
        versions = [self._version(handle) for handle in handles]
        if len(set(handle.request_id for handle in handles)) != len(handles):
            raise ValueError("duplicate request handles in one batch")
        if getattr(batch, "batch_size", len(handles)) != len(handles):
            raise ValueError("handle count must match batch size")
        seq_lens = self._lengths(batch, "seq_lens")
        if len(seq_lens) != len(handles) or any(length <= 0 for length in seq_lens):
            raise ValueError("seq_lens must contain one positive length per request")
        if decode:
            lengths = (1,) * len(handles)
            prefixes = tuple(length - 1 for length in seq_lens)
        else:
            lengths = self._lengths(batch, "extend_seq_lens")
            prefixes = self._lengths(batch, "extend_prefix_lens")
            if len(lengths) != len(handles) or len(prefixes) != len(handles):
                raise ValueError("extend metadata must match the handle count")
            if any(length <= 0 or prefix < 0 for length, prefix in zip(lengths, prefixes)):
                raise ValueError("positive extend lengths and nonnegative prefixes required")
            if tuple(a + b for a, b in zip(lengths, prefixes)) != seq_lens:
                raise ValueError("extend/prefix lengths must equal seq_lens")
        if sum(lengths) != len(q):
            raise ValueError("batch token counts do not match Q/K/V")
        positions_tensor = getattr(batch, "positions", None)
        if not isinstance(positions_tensor, torch.Tensor):
            raise ValueError("explicit per-token positions tensor required")
        positions = self._integers(positions_tensor, "positions")
        if len(positions) != len(q):
            raise ValueError("positions must contain one entry per input token")
        offset = 0
        for version, length, prefix in zip(versions, lengths, prefixes):
            if positions[offset:offset + length] != tuple(range(prefix, prefix + length)):
                raise ValueError("positions must exactly continue each request's causal prefix")
            expected_keys = {(int(layer_id), h, LIVE_SEGMENT) for h in range(kv_heads)}
            stored_keys = {key for key in version.segments if key[0] == layer_id}
            if stored_keys != (expected_keys if prefix else set()):
                raise ValueError("shared KV layer prefix is absent, partial, or uses unsupported segments")
            if any(version.segments[key].positions != tuple(range(prefix)) for key in stored_keys):
                raise ValueError("shared KV layer does not cover the complete causal prefix")
            if any(version.segments[key].position_basis != "model_encoded" for key in stored_keys):
                raise ValueError("shared KV layer position basis does not match the model adapter")
            offset += length

        windows = (0,) * kv_heads if windows is None else self._integers(windows, "windows")
        sinks = (0,) * kv_heads if sinks is None else self._integers(sinks, "sinks")
        if len(windows) != kv_heads or len(sinks) != kv_heads or min((*windows, *sinks), default=0) < 0:
            raise ValueError("nonnegative window/sink policy required per physical KV head")
        if any(window == 0 and sink for window, sink in zip(windows, sinks)):
            raise ValueError("global heads cannot specify sink tokens")
        sliding = getattr(layer, "sliding_window_size", -1)
        if sliding is None:
            sliding = -1
        if isinstance(sliding, bool) or not isinstance(sliding, Integral) or sliding < -1:
            raise ValueError("invalid layer sliding window")
        if sliding >= 0:
            if any(sinks):
                raise ValueError("sink policy cannot expand a model sliding window")
            # SGLang's layer value is the number of previous visible tokens;
            # the paged kernel's width includes the current token as well.
            width = int(sliding) + 1
            windows = tuple(min(window, width) if window else width for window in windows)
        return q, k, v, tuple(handles), versions, lengths, positions, int(layer_id), kv_heads, float(scale), windows, sinks

    def forward(self, q, k, v, layer, forward_batch, *, decode, save_kv_cache,
                windows=None, sinks=None):
        """Append all physical KV heads, then attend directly through page tables.

        ``windows`` includes the current token (zero denotes global attention).
        K/V must already contain the model's position transform. All request
        metadata is validated before the first append; OOM/runtime errors after
        earlier requests complete do not roll back the entire batch.
        """
        with self.manager._lock:
            (q, k, v, handles, versions, lengths, positions, layer_id, kv_heads,
             scale, windows, sinks) = self._preflight(
                q, k, v, layer, forward_batch, decode, save_kv_cache, windows, sinks,
            )
            outputs = []
            offset = 0
            for handle, version, length in zip(handles, versions, lengths):
                end = offset + length
                chunk_positions = positions[offset:end]
                writes = [SegmentWrite(
                    (layer_id, head, LIVE_SEGMENT), k[offset:end, head],
                    v[offset:end, head], chunk_positions,
                    f"backend:{handle.generation}:{version.epoch + 1}:layer:{layer_id}",
                    position_basis="model_encoded",
                ) for head in range(kv_heads)]
                self.manager.append_many(handle.request_id, appends=writes,
                                         expected_epoch=version.epoch)
                view_key = (handle, layer_id, kv_heads)
                view = self._views.get(view_key)
                if view is None:
                    view = ManagedSegPagedKVCache(
                        self.manager, handle.request_id, num_layers=layer_id + 1,
                        num_kv_heads=kv_heads,
                    )
                    self._views[view_key] = view
                for head, (window, sink) in enumerate(zip(windows, sinks)):
                    view.set_head_policy(layer_id, head, "local" if window else "global",
                                         window=window if window else None, sink=sink)
                query_positions = torch.tensor(chunk_positions, dtype=torch.int64,
                                               device=q.device)
                output = view.attention(
                    q[offset:end].transpose(0, 1), layer=layer_id,
                    query_positions=query_positions, num_q_per_kv=q.shape[1] // kv_heads,
                    sm_scale=scale, causal=True,
                )
                outputs.append(output.transpose(0, 1).reshape(length, -1))
                offset = end
            return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.shape[1] * q.shape[2]))
