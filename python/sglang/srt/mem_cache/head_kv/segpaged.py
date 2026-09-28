"""Managed SegPagedAttention adapter with explicit positions and ownership.

This adapter connects the existing RedKnot SegPagedAttention entry point to the
bounded manager/direct paged kernel. It does not replace SGLang's scheduler or
its model-specific KV production and reuse-validity decisions.
"""

from __future__ import annotations

from numbers import Integral
import threading

import torch

from .attention import paged_attention
from .manager import SegmentPatch, SegmentWrite
from .pool import completion_event


class ManagedSegPagedKVCache:
    """A persistent request view over a caller-owned ``HeadKVManager``.

    The caller creates/releases the request and determines the model/namespace
    contract. This object does not rebuild KV pages per attention call. Only
    descriptor metadata is rebuilt when a request version changes.

    ``position_basis`` is an adapter declaration, not a transform: K values
    passed to writes/repairs must already be in their declared target basis.
    """

    _managed_segpaged_cache = True

    def __init__(self, manager, request_id, *, num_layers: int, num_kv_heads: int):
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0
               for x in (num_layers, num_kv_heads)):
            raise ValueError("positive layer and KV-head counts required")
        manager.version(request_id)
        self.manager, self.request_id = manager, request_id
        self.num_layers, self.num_kv_heads = int(num_layers), int(num_kv_heads)
        self.head_dim, self.page_size = manager.pool.head_dim, manager.pool.page_size
        self.device, self.dtype = manager.pool.device, manager.pool.dtype
        self._policies = {}
        self._descriptor_cache = {}
        self._lock = threading.RLock()

    def _key(self, layer, head, segment):
        if (isinstance(layer, bool) or not isinstance(layer, Integral)
                or not 0 <= layer < self.num_layers
                or isinstance(head, bool) or not isinstance(head, Integral)
                or not 0 <= head < self.num_kv_heads):
            raise ValueError("layer/KV head outside topology")
        if isinstance(segment, bool) or not isinstance(segment, (str, Integral)):
            raise TypeError("segment occurrence must be a string or integer")
        return int(layer), int(head), str(segment)

    @staticmethod
    def _policy(policy, window, sink):
        if policy not in ("global", "local"):
            raise ValueError("policy must be 'global' or 'local'")
        if isinstance(sink, bool) or not isinstance(sink, Integral) or sink < 0:
            raise ValueError("sink must be a nonnegative integer")
        if policy == "global":
            if window not in (None, 0):
                raise ValueError("global heads cannot specify a local window")
            return ("global", 0, 0)
        if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
            raise ValueError("local policy requires an explicit positive window")
        return ("local", int(window), int(sink))

    def set_head_policy(self, layer, head, policy, *, window=None, sink=0):
        self._key(layer, head, "validation")
        value = self._policy(policy, window, sink)
        with self._lock:
            self._policies[(layer, head)] = value

    def add_head_segment(
        self, *, layer, head, segment, policy, k, v, positions, provenance,
        window=None, sink=0, position_basis="none", reuse_kind="exact_context",
        validity_certificate="",
    ):
        """Atomically write/replace a segment through manager admission/COW.

        Unlike the legacy cache, target positions and provenance are mandatory;
        dropping old local-window tokens remains an explicit manager operation.
        Non-exact reuse additionally requires the adapter's validity certificate;
        this bridge forwards it to the manager without deriving a proof itself.
        """
        key = self._key(layer, head, segment)
        config = self._policy(policy, window, sink)
        with self._lock:
            old_policy = self._policies.get((layer, head))
            if old_policy is not None and old_policy != config:
                raise ValueError("segments of one KV head must use the same policy")
            write = SegmentWrite(key, k, v, tuple(positions), provenance,
                                 position_basis=position_basis, reuse_kind=reuse_kind,
                                 validity_certificate=validity_certificate)
            version = self.manager.update(self.request_id, writes=(write,))
            self._policies[(layer, head)] = config
            return version.segments[key]

    def repair_head_segment(self, *, layer, head, segment, indices, k, v, provenance):
        """COW only affected pages; payload must use the segment's existing basis.

        The model adapter is responsible for cross-layer invalidation and repair
        inputs. Use one manager transaction for a repair spanning several heads.
        """
        key = self._key(layer, head, segment)
        version = self.manager.update(
            self.request_id,
            patches=(SegmentPatch(key, tuple(indices), k, v, provenance),),
        )
        return version.segments[key]

    def fork(self, target_request_id):
        """Create a shared immutable request root; caller owns its later release."""
        with self._lock:
            self.manager.fork(self.request_id, target_request_id)
            child = type(self)(self.manager, target_request_id, num_layers=self.num_layers,
                               num_kv_heads=self.num_kv_heads)
            child._policies = dict(self._policies)
            return child

    def _descriptor(self, lease, layer):
        identity = (lease.version.generation, lease.version.epoch)
        with self._lock:
            cached = self._descriptor_cache.get(layer)
            if cached is None or cached[0] != identity:
                descriptor = lease.descriptor(layer, self.num_kv_heads)
                ready = completion_event(self.device)
                cached = (identity, descriptor, ready)
                self._descriptor_cache[layer] = cached
            _, descriptor, ready = cached
            if self.device.type == "cuda":
                stream = torch.cuda.current_stream(self.device)
                stream.wait_event(ready)
                # Cached buffers may have been created on a different stream;
                # retain their allocation through this consumer even if replaced.
                for name in ("page_slots", "page_lengths", "key_positions"):
                    descriptor[name].record_stream(stream)
            return descriptor

    def attention(self, query, *, layer, query_positions, num_q_per_kv, sm_scale,
                  use_fused=True, causal=True):
        """Read one immutable version; retire its pin on the current CUDA stream."""
        self._key(layer, 0, "validation")
        if query_positions is None:
            raise ValueError("managed SegPagedAttention requires explicit query_positions")
        with self._lock:
            policies = [self._policies.get((layer, h), ("global", 0, 0))
                        for h in range(self.num_kv_heads)]
        with self.manager.bind(self.request_id) as lease:
            return paged_attention(
                query, **self._descriptor(lease, layer), query_positions=query_positions,
                num_q_per_kv=num_q_per_kv, scale=sm_scale, causal=causal,
                windows=[p[1] for p in policies], sinks=[p[2] for p in policies],
                backend="auto" if use_fused else "torch",
            )

    def physical_bytes(self):
        """Unique reachable payload-page capacity, excluding other requests."""
        version = self.manager.version(self.request_id)
        refs = {p for seg in version.segments.values() for p in seg.pages}
        return len(refs) * self.manager.pool.page_bytes

    def stored_token_count(self):
        """Logical entries; repeated shared occurrences count separately."""
        return sum(s.length for s in self.manager.version(self.request_id).segments.values())
