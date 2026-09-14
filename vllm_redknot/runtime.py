"""Request plans and atomic chunk-cache transactions, independent of vLLM."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

MAX_CHUNKS = 8
MAX_NAMESPACE_LENGTH = 256


def stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class RedKnotSettings:
    local_heads: Mapping[str, tuple[int, ...]]
    model_revision: str
    allow_approximate: bool = False
    boundary_tokens: int = 128
    max_cache_bytes: int = 1 << 30
    rope_theta: float = 10000.0
    rotary_dim: int | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RedKnotSettings:
        heads = values.get("local_heads", {})
        if not isinstance(heads, Mapping) or not heads:
            raise ValueError("local_heads must contain at least one layer")
        normalized = {}
        for layer, indices in heads.items():
            if not isinstance(layer, str) or not layer:
                raise ValueError("local_heads layer keys must be nonempty strings")
            if not isinstance(indices, (list, tuple)):
                raise ValueError(f"local_heads[{layer!r}] must be an array")
            if not indices:
                continue
            checked = tuple(_integer(i, "local KV head") for i in indices)
            if len(set(checked)) != len(checked):
                raise ValueError("local KV heads must not contain duplicates")
            normalized[layer] = tuple(sorted(checked))
        if not normalized:
            raise ValueError("at least one layer must have local heads")
        revision = values.get("model_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("model_revision is required for cache isolation")
        allow = values.get("allow_approximate", False)
        if not isinstance(allow, bool):
            raise ValueError("allow_approximate must be a boolean")
        rotary_dim = values.get("rotary_dim")
        if rotary_dim is not None:
            _integer(rotary_dim, "rotary_dim", 2)
            if rotary_dim % 2:
                raise ValueError("rotary_dim must be even")
        return cls(
            local_heads=normalized,
            model_revision=revision,
            allow_approximate=allow,
            boundary_tokens=_integer(
                values.get("boundary_tokens", 128), "boundary_tokens"
            ),
            max_cache_bytes=_integer(
                values.get("max_cache_bytes", 1 << 30), "max_cache_bytes"
            ),
            rope_theta=float(values.get("rope_theta", 10000.0)),
            rotary_dim=rotary_dim,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "format": 1,
            "model_revision": self.model_revision,
            "local_heads": dict(self.local_heads),
            "boundary_tokens": self.boundary_tokens,
            "rope_theta": self.rope_theta,
            "rotary_dim": self.rotary_dim,
        }


@dataclass(frozen=True)
class ChunkSpan:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


# REDKNOT: RK-REQUEST — content spans, not a prefix-cache or physical-page plan.
@dataclass(frozen=True)
class RequestPlan:
    mode: str
    namespace: str
    chunks: tuple[ChunkSpan, ...]
    allow_approximate: bool

    @classmethod
    def parse(
        cls, extra_args: Mapping[str, Any] | None, prompt_length: int
    ) -> RequestPlan | None:
        if not extra_args:
            return None
        raw = extra_args.get("redknot")
        transfer = extra_args.get("kv_transfer_params")
        if raw is None and isinstance(transfer, Mapping):
            raw = transfer.get("redknot")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("redknot must be an object")
        mode = raw.get("mode")
        if mode not in {"capture", "reuse", "recomputed"}:
            raise ValueError("redknot.mode must be capture, reuse or recomputed")
        namespace = raw.get("namespace")
        if (
            not isinstance(namespace, str)
            or not namespace.strip()
            or len(namespace) > MAX_NAMESPACE_LENGTH
        ):
            raise ValueError(
                f"redknot.namespace must contain 1 to {MAX_NAMESPACE_LENGTH} characters"
            )
        allow = raw.get("allow_approximate", False)
        if not isinstance(allow, bool):
            raise ValueError("redknot.allow_approximate must be a boolean")
        raw_chunks = raw.get("chunks", [])
        if not isinstance(raw_chunks, list):
            raise ValueError("redknot.chunks must be an array")
        if len(raw_chunks) > MAX_CHUNKS:
            raise ValueError(
                f"redknot supports at most {MAX_CHUNKS} chunks per request"
            )
        chunks = []
        previous_end = 0
        for entry in raw_chunks:
            if not isinstance(entry, Mapping):
                raise ValueError("each chunk must be an object")
            start = _integer(entry.get("start"), "chunk.start")
            end = _integer(entry.get("end"), "chunk.end", 1)
            if start >= end or end > prompt_length or start < previous_end:
                raise ValueError("chunks must be ordered, disjoint prompt token spans")
            chunks.append(ChunkSpan(start, end))
            previous_end = end
        if mode in {"capture", "reuse"} and not chunks:
            raise ValueError(f"{mode} requires chunks")
        if mode == "capture" and chunks != [ChunkSpan(0, prompt_length)]:
            raise ValueError(
                "capture requires one isolated chunk spanning the full prompt"
            )
        return cls(mode, namespace, tuple(chunks), allow)


@dataclass(frozen=True)
class LayerSpec:
    local_heads: tuple[int, ...]
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    dtype: str

    @property
    def local_query_heads(self) -> tuple[int, ...]:
        group = self.num_query_heads // self.num_kv_heads
        return tuple(
            q for h in self.local_heads for q in range(h * group, (h + 1) * group)
        )


@dataclass(frozen=True)
class LayerPayload:
    spec: LayerSpec
    keys: Any
    values: Any
    outputs: Any
    nbytes: int


@dataclass(frozen=True)
class ChunkPayload:
    length: int
    layers: Mapping[str, LayerPayload]


@dataclass
class StepState:
    mode: str = "native"
    reason: str | None = None
    plan: RequestPlan | None = None
    keys: tuple[str, ...] = ()
    cached: Mapping[str, ChunkPayload] = field(default_factory=dict)
    staged: dict[str, LayerPayload] = field(default_factory=dict)
    specs: Mapping[str, LayerSpec] = field(default_factory=dict)
    clean_runs: tuple[tuple[int, int, int], ...] = ()
    dirty_runs: tuple[tuple[int, int], ...] = ()
    prompt_length: int = 0
    modified_layers: int = 0


def clean_and_dirty_runs(
    chunks: Sequence[ChunkSpan], prompt_length: int, boundary_tokens: int
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int], ...]]:
    """Return (clean start/end/chunk-index runs, complementary dirty runs)."""
    clean = []
    dirty = []
    cursor = 0
    for index, span in enumerate(chunks):
        start = min(span.end, span.start + (boundary_tokens if span.start else 0))
        if start >= span.end:
            continue
        if cursor < start:
            dirty.append((cursor, start))
        clean.append((start, span.end, index))
        cursor = span.end
    if cursor < prompt_length:
        dirty.append((cursor, prompt_length))
    return tuple(clean), tuple(dirty)


# REDKNOT: RK-TRANSACTION — whole-chunk publication and whole-request leases.
class RedKnotRuntime:
    def __init__(self, settings: RedKnotSettings, cache: Any, model_identity: Any):
        self.settings = settings
        self.cache = cache
        self.identity = stable_digest(
            {"model": model_identity, "policy": settings.identity()}
        )
        self.counters: Counter[str] = Counter()

    def chunk_key(self, namespace: str, tokens: Sequence[int]) -> str:
        return stable_digest(
            {"identity": self.identity, "namespace": namespace, "tokens": list(tokens)}
        )

    def fallback(self, reason: str) -> StepState:
        self.counters[f"fallback:{reason}"] += 1
        return StepState(reason=reason)

    @contextmanager
    def step(
        self,
        *,
        extra_args: Mapping[str, Any] | None,
        token_ids: Sequence[int],
        specs: Mapping[str, LayerSpec],
        unsupported_reason: str | None = None,
    ):
        plan = RequestPlan.parse(extra_args, len(token_ids))
        if plan is None:
            yield self.fallback("no_plan")
            return
        if unsupported_reason:
            yield self.fallback(unsupported_reason)
            return
        if plan.mode == "recomputed":
            self.counters["recomputed_steps"] += 1
            yield StepState(mode="recomputed", plan=plan)
            return
        if plan.mode == "reuse" and not (
            plan.allow_approximate and self.settings.allow_approximate
        ):
            raise ValueError("reuse requires server and request allow_approximate=true")
        keys = tuple(
            self.chunk_key(plan.namespace, token_ids[c.start : c.end])
            for c in plan.chunks
        )
        state = StepState(
            mode=plan.mode,
            plan=plan,
            keys=keys,
            specs=specs,
            prompt_length=len(token_ids),
        )
        if plan.mode == "capture":
            projected_bytes = sum(
                len(token_ids)
                * (2 * len(spec.local_heads) + len(spec.local_query_heads))
                * spec.head_size
                * 2
                for spec in specs.values()
            )
            if projected_bytes > self.settings.max_cache_bytes:
                yield self.fallback("capture_capacity")
                return
            try:
                yield state
            except BaseException:
                self.counters["capture_aborted"] += 1
                raise
            if set(state.staged) != set(specs) or not specs:
                self.counters["capture_incomplete"] += 1
                return
            payload = ChunkPayload(len(token_ids), dict(state.staged))
            stored = self.cache.put(
                keys[0], payload, sum(layer.nbytes for layer in state.staged.values())
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
            for span, key in zip(plan.chunks, keys):
                payload = cached.get(key)
                if (
                    not isinstance(payload, ChunkPayload)
                    or payload.length != span.length
                    or set(payload.layers) != set(specs)
                ):
                    yield self.fallback("cache_contract")
                    return
                for name, spec in specs.items():
                    item = payload.layers[name]
                    if not isinstance(item, LayerPayload) or item.spec != spec:
                        yield self.fallback("cache_contract")
                        return
                    expected = (
                        (
                            item.keys,
                            (span.length, len(spec.local_heads), spec.head_size),
                        ),
                        (
                            item.values,
                            (span.length, len(spec.local_heads), spec.head_size),
                        ),
                        (
                            item.outputs,
                            (span.length, len(spec.local_query_heads), spec.head_size),
                        ),
                    )
                    if any(
                        tuple(t.shape) != shape or str(t.dtype) != spec.dtype
                        for t, shape in expected
                    ):
                        yield self.fallback("cache_contract")
                        return
            state.cached = cached
            yield state
            self.counters["reuse_steps"] += 1

    def stats(self) -> dict[str, Any]:
        return {"runtime": dict(self.counters), "cache": self.cache.stats()}
