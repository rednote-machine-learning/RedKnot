"""Atomic chunk transactions for DSV4 projected local-head contributions.

The latent KV, compressor and indexer state are never stored here: vLLM updates
them online on every request. This cache stores only source-position z_off.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from .runtime import (
    RedKnotRuntime,
    RequestPlan,
    clean_and_dirty_runs,
    stable_digest,
)


@dataclass(frozen=True)
class DSV4LayerSpec:
    local_heads: tuple[int, ...]
    num_heads: int
    groups: int
    rank: int
    compress_ratio: int
    head_dim: int = 512


@dataclass(frozen=True)
class DSV4Chunk:
    length: int
    layers: Mapping[str, Any]


@dataclass
class DSV4Step:
    mode: str = "native"
    reason: str | None = None
    plan: RequestPlan | None = None
    keys: tuple[str, ...] = ()
    specs: Mapping[str, DSV4LayerSpec] = field(default_factory=dict)
    cached: Mapping[str, DSV4Chunk] = field(default_factory=dict)
    staged: dict[str, Any] = field(default_factory=dict)
    projected_layers: set[str] = field(default_factory=set)
    sparse_layers: set[str] = field(default_factory=set)
    clean_runs: tuple[tuple[int, int, int], ...] = ()
    dirty_runs: tuple[tuple[int, int], ...] = ()
    prompt_length: int = 0
    active_layer: str | None = None


# REDKNOT: RK-FLASH-TRANSACTION — z_off leases/commit; native state stays online.
class DSV4Runtime(RedKnotRuntime):
    def policy_key(self, name: str, spec: DSV4LayerSpec) -> str:
        return stable_digest(
            {
                "family": "dsv4-zoff-native-fp8-v1",
                "identity": self.identity,
                "layer": name,
                "spec": asdict(spec),
            }
        )

    def fallback(self, reason: str) -> DSV4Step:
        self.counters[f"fallback:{reason}"] += 1
        return DSV4Step(reason=reason)

    @contextmanager
    def step(
        self,
        *,
        extra_args: Mapping[str, Any] | None,
        token_ids: Sequence[int],
        specs: Mapping[str, DSV4LayerSpec],
        unsupported_reason: str | None = None,
    ):
        from .dsv4_projection import CachedLocalZ

        plan = RequestPlan.parse(extra_args, len(token_ids))
        if plan is None or unsupported_reason:
            yield self.fallback(unsupported_reason or "no_plan")
            return
        if plan.mode == "recomputed":
            self.counters["recomputed_steps"] += 1
            yield DSV4Step(mode="recomputed", plan=plan)
            return
        if plan.mode == "reuse" and not (
            plan.allow_approximate and self.settings.allow_approximate
        ):
            raise ValueError("DSV4 reuse requires both approximation opt-ins")
        keys = tuple(
            self.chunk_key(plan.namespace, token_ids[span.start : span.end])
            for span in plan.chunks
        )
        state = DSV4Step(
            mode=plan.mode,
            plan=plan,
            keys=keys,
            specs=specs,
            prompt_length=len(token_ids),
        )
        if plan.mode == "capture":
            projected = sum(
                len(token_ids) * (2 * spec.groups * spec.rank + 8)
                for spec in specs.values()
            )
            if projected > self.settings.max_cache_bytes:
                yield self.fallback("capture_capacity")
                return
            try:
                yield state
            except BaseException:
                self.counters["capture_aborted"] += 1
                raise
            if not specs or set(state.staged) != set(specs):
                self.counters["capture_incomplete"] += 1
                return
            payload = DSV4Chunk(len(token_ids), dict(state.staged))
            stored = self.cache.put(
                keys[0], payload, sum(item.nbytes for item in state.staged.values())
            )
            self.counters[
                "capture_committed" if stored else "capture_capacity_rejected"
            ] += 1
            return
        state.clean_runs, state.dirty_runs = clean_and_dirty_runs(
            plan.chunks, len(token_ids), self.settings.boundary_tokens
        )
        if not state.clean_runs:
            yield self.fallback("no_clean_rows")
            return
        with self.cache.lease(keys) as cached:
            if cached is None:
                yield self.fallback("cache_miss")
                return
            for span, key in zip(plan.chunks, keys, strict=True):
                payload = cached[key]
                if not isinstance(payload, DSV4Chunk) or (
                    payload.length != span.length or set(payload.layers) != set(specs)
                ):
                    yield self.fallback("cache_contract")
                    return
                for name, spec in specs.items():
                    item = payload.layers[name]
                    if not isinstance(item, CachedLocalZ) or (
                        item.policy_key != self.policy_key(name, spec)
                        or item.local_head_ids != spec.local_heads
                        or item.num_heads != spec.num_heads
                        or item.head_dim != spec.head_dim
                        or tuple(item.z_off.shape)
                        != (span.length, spec.groups * spec.rank)
                        or str(item.z_off.dtype) != "torch.bfloat16"
                        or str(item.z_off.device) != "cpu"
                        or tuple(item.source_positions.shape) != (span.length,)
                        or str(item.source_positions.dtype) != "torch.int64"
                        or str(item.source_positions.device) != "cpu"
                        or item.source_positions.tolist() != list(range(span.length))
                    ):
                        yield self.fallback("cache_contract")
                        return
            state.cached = cached
            yield state
            if state.projected_layers != set(specs) or state.sparse_layers != set(
                specs
            ):
                raise RuntimeError("DSV4 reuse did not visit every selected layer")
            self.counters["reuse_steps"] += 1
