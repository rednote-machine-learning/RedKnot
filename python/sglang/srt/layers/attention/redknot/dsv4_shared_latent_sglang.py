"""Production SGLang adapters for persistent DeepSeek-V4 shared latent KV.

This is the only shared-latent module which knows the concrete SGLang cache
layout.  The CPU artifact controller owns token/epoch correctness and
``dsv4_shared_latent_gpu`` owns immutable device banks.  This adapter owns the
two transformations that must never be guessed by the generic store:

* packed SWA/C4/C128 records are captured after removing source RoPE and are
  restored by applying destination RoPE exactly once;
* Indexer K is captured as position-zero, pre-Hadamard BF16[128], then restored
  by applying destination RoPE, normalized Hadamard, and the normal FP8
  quantizer.

The public serving API is intentionally small.  Snapshot capture is chunked:
SWA has only a 128-row physical ring, so an 8K segment can never be recovered
by rereading that ring after the final prefill chunk::

    staging = begin_layer_capture_bundle(...)
    capture_chunk_components(staging, ...)  # before the next chunk overwrites SWA
    bundle = staging.finalize()
    bundle.stage_into(gpu_store, staged_epoch)

    adapter = build_layer_restore_adapter(...)
    validated = gpu_store.preflight_targets(
        prepared,
        layer_id=layer_id,
        targets=adapter.targets,
        target_slots=adapter.target_slots,
        kernels=adapter.kernels,
        positions=positions,
    )
    receipt = gpu_store.restore_clean(validated)

``build_layer_restore_adapter`` only binds existing cache tensors and slot
metadata.  It never invokes ``wkv``, the attention compressor, or the Indexer
compressor, and therefore cannot silently turn a clean restore into a
full-size online recomputation.

State records use grouped semantics.  One C4 restart record contains two
terminal compressor groups, each group containing four physical state rows.
One C128 record contains one group of either 128 rows (ordinary C128) or one
row (online-C128).  The scalar state target slot at an output anchor denotes
the *terminal group*: C4 restores ``[terminal-1, terminal]`` and C128 restores
``[terminal]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.attention.redknot.dsv4_offline_reuse_v2 import (
    _indexer_rotate_activation,
    _quantize_indexer_activation,
    _read_indexer_packed,
    _write_indexer_packed,
    compute_compressed_slots,
    compute_paged_compressed_slots,
    read_packed_kv,
    reposition_rope,
    write_packed_kv,
    write_rope_bf16,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_cache import (
    INDEXER_POSITION_SEMANTICS,
    PACKED_LATENT_BYTES,
    PACKED_LATENT_POSITION_SEMANTICS,
    build_dsv4_0731_shared_latent_spec,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
    DOMAIN_C128,
    DOMAIN_C128_ATTENTION_STATE,
    DOMAIN_C4,
    DOMAIN_C4_ATTENTION_STATE,
    DOMAIN_INDEXER,
    DOMAIN_INDEXER_STATE,
    DOMAIN_SWA,
    RESTORE_FAMILY_INDEXER,
    RESTORE_FAMILY_PACKED,
    RESTORE_FAMILY_STATE,
    restore_family_for_domain,
)


INDEXER_PACKED_RECORD_BYTES = 132
INDEXER_POSITIONLESS_RECORD_BYTES = 128 * 2  # pre-Hadamard BF16[128]
PACKED_NOPE_BYTES = 448


def _is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith(
        "sha256:"
    ):
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return value[7:] == value[7:].lower()
PACKED_ROPE_BYTES = 64 * 2
PACKED_NOPE_ROPE_BYTES = PACKED_NOPE_BYTES + PACKED_ROPE_BYTES

ATTENTION_STATE_POSITION_SEMANTICS = (
    "opaque_attention_compressor_restart_state_v1"
)
INDEXER_STATE_POSITION_SEMANTICS = "opaque_indexer_compressor_restart_state_v1"
CHECKPOINT_STRIDE_TOKENS = 512

_C4_DOMAINS = (
    DOMAIN_SWA,
    DOMAIN_C4,
    DOMAIN_INDEXER,
    DOMAIN_C4_ATTENTION_STATE,
    DOMAIN_INDEXER_STATE,
)
_C128_DOMAINS = (
    DOMAIN_SWA,
    DOMAIN_C128,
    DOMAIN_C128_ATTENTION_STATE,
)


def _as_contiguous_uint8_rows(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or values.ndim < 2:
        raise TypeError("shared-latent component must be a rank-2+ tensor")
    if not values.is_contiguous():
        values = values.contiguous()
    return values.view(torch.uint8).view(values.shape[0], -1)


def _as_uint8_group_rows_without_copy(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or values.ndim < 2:
        raise TypeError("compressor state target must be a tensor")
    if not values.is_contiguous():
        raise ValueError("compressor state target must be contiguous")
    return values.view(torch.uint8).view(values.shape[0], -1)


def _long_vector(
    values: torch.Tensor,
    *,
    name: str,
    expected_rows: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor")
    if expected_rows is not None and int(values.numel()) != int(expected_rows):
        raise ValueError(
            f"{name} has {values.numel()} rows; expected {expected_rows}"
        )
    return values.to(dtype=torch.long)


def _require_unique_nonnegative_slots(
    slots: torch.Tensor, *, expected: int, name: str
) -> None:
    """Offline-only alias check before a complete artifact is claimed."""

    slots = _long_vector(slots, name=name, expected_rows=expected)
    if bool((slots < 0).any().item()):
        raise ValueError(f"{name} contains a negative cache slot")
    if int(torch.unique(slots).numel()) != expected:
        raise ValueError(
            f"{name} aliases physical cache rows; capture must occur before "
            "the SWA/ring slot is overwritten or use explicit canonical rows"
        )


def _state_group_view(state_buffer: torch.Tensor, group_width: int) -> torch.Tensor:
    width = int(group_width)
    if (
        not isinstance(state_buffer, torch.Tensor)
        or state_buffer.ndim != 2
        or not state_buffer.is_contiguous()
        or width <= 0
        or int(state_buffer.shape[0]) % width
    ):
        raise ValueError("compressor state buffer/group width is invalid")
    return state_buffer.view(-1, width, state_buffer.shape[-1])


def _state_group_width(state_pool, *, ratio: int) -> int:
    width = min(int(ratio), int(state_pool.ring_size))
    if int(ratio) == 4 and width != 4:
        raise ValueError("C4 compressor state group width must be four")
    if int(ratio) == 128 and width not in (1, 128):
        raise ValueError("C128 state group width must be one or 128")
    return width


def _state_required_groups(domain: str) -> int:
    if domain in (DOMAIN_C4_ATTENTION_STATE, DOMAIN_INDEXER_STATE):
        return 2
    if domain == DOMAIN_C128_ATTENTION_STATE:
        return 1
    raise ValueError(f"{domain!r} is not a compressor state domain")


def _state_record_bytes(state_pool, *, ratio: int, domain: str) -> int:
    width = _state_group_width(state_pool, ratio=ratio)
    state = state_pool.kv_score_buffer.kv_score
    _state_group_view(state, width)
    return (
        _state_required_groups(domain)
        * width
        * int(state.shape[-1])
        * int(state.element_size())
    )


def _expected_domains(compress_ratio: int) -> Tuple[str, ...]:
    ratio = int(compress_ratio)
    if ratio == 4:
        return _C4_DOMAINS
    if ratio == 128:
        return _C128_DOMAINS
    raise ValueError("reusable middle layers must use C4 or C128")


def _object_field(value: object, name: str, *aliases: str):
    """Read one adapter field from either a Mapping or an attribute object."""

    names = (name,) + tuple(aliases)
    if isinstance(value, Mapping):
        for candidate in names:
            if candidate in value:
                return value[candidate]
    else:
        for candidate in names:
            if hasattr(value, candidate):
                return getattr(value, candidate)
    raise ValueError(f"adapter value is missing {name!r}")


def _checkpoint_anchors(length: int) -> Tuple[int, ...]:
    length = int(length)
    if length <= 0:
        raise ValueError("segment length must be positive")
    return tuple(
        range(CHECKPOINT_STRIDE_TOKENS, length, CHECKPOINT_STRIDE_TOKENS)
    ) + (length,)


@dataclass
class _RowCoverage:
    """CPU-only exact-once row coverage for a device staging tensor."""

    row_count: int
    _spans: list[Tuple[int, int]] = field(default_factory=list)

    def validate(self, begin: int, end: int, *, domain: str) -> None:
        begin = int(begin)
        end = int(end)
        if begin < 0 or end <= begin or end > self.row_count:
            raise ValueError(
                f"{domain} capture span [{begin},{end}) is outside "
                f"[0,{self.row_count})"
            )
        for prior_begin, prior_end in self._spans:
            if begin < prior_end and prior_begin < end:
                raise ValueError(f"{domain} capture rows were written twice")

    def add(self, begin: int, end: int, *, domain: str) -> None:
        self.validate(begin, end, domain=domain)
        self._spans.append((begin, end))
        self._spans.sort()

    @property
    def captured_rows(self) -> int:
        return sum(end - begin for begin, end in self._spans)

    @property
    def complete(self) -> bool:
        cursor = 0
        for begin, end in self._spans:
            if begin != cursor:
                return False
            cursor = end
        return cursor == self.row_count

    @property
    def missing_spans(self) -> Tuple[Tuple[int, int], ...]:
        result = []
        cursor = 0
        for begin, end in self._spans:
            if cursor < begin:
                result.append((cursor, begin))
            cursor = end
        if cursor < self.row_count:
            result.append((cursor, self.row_count))
        return tuple(result)


def build_runtime_shared_latent_spec(
    *,
    token_to_kv_pool,
    model_hash: str,
    policy_hash: str,
    segment_length: int,
    c4_layer_id: int,
    c128_layer_id: int,
):
    """Infer exact state widths from live pools for the 3+37+3 contract."""

    c4_attention = token_to_kv_pool.get_attention_compress_states(c4_layer_id)
    c4_indexer = token_to_kv_pool.get_indexer_compress_states(c4_layer_id)
    c128_attention = token_to_kv_pool.get_attention_compress_states(c128_layer_id)
    return build_dsv4_0731_shared_latent_spec(
        model_hash=str(model_hash),
        policy_hash=str(policy_hash),
        length=int(segment_length),
        c4_indexer_record_bytes=INDEXER_POSITIONLESS_RECORD_BYTES,
        c4_attention_terminal_state_bytes=_state_record_bytes(
            c4_attention, ratio=4, domain=DOMAIN_C4_ATTENTION_STATE
        ),
        c4_indexer_terminal_state_bytes=_state_record_bytes(
            c4_indexer, ratio=4, domain=DOMAIN_INDEXER_STATE
        ),
        c128_attention_terminal_state_bytes=_state_record_bytes(
            c128_attention, ratio=128, domain=DOMAIN_C128_ATTENTION_STATE
        ),
    )


@torch.no_grad()
def canonicalize_packed_latent(
    *,
    cache: torch.Tensor,
    slots: torch.Tensor,
    page_size: int,
    source_positions: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Capture packed KV with source-position RoPE removed exactly once."""

    slots = _long_vector(slots, name="packed capture slots")
    source_positions = _long_vector(
        source_positions,
        name="packed source positions",
        expected_rows=int(slots.numel()),
    ).to(device=cache.device)
    slots = slots.to(device=cache.device)
    if int(page_size) <= 0:
        raise ValueError("packed capture page size must be positive")
    packed = read_packed_kv(cache, slots, int(page_size)).contiguous()
    if (
        packed.dtype != torch.uint8
        or packed.ndim != 2
        or int(packed.shape[1]) != PACKED_LATENT_BYTES
    ):
        raise ValueError("DSV4 packed latent capture must contain 584-byte rows")

    # The packed artifact itself is a one-entry-page buffer.  Reading its RoPE
    # bytes through the canonical cache helper avoids assuming a BF16 stride.
    local_slots = torch.arange(
        int(packed.shape[0]), device=packed.device, dtype=torch.long
    )
    from sglang.srt.layers.attention.redknot.dsv4_offline_reuse_v2 import (
        read_rope_bf16,
    )

    source_rope = read_rope_bf16(packed, local_slots, 1)
    canonical_rope = reposition_rope(
        source_rope,
        source_positions,
        torch.zeros_like(source_positions),
        freqs_cis,
    )
    # Only the BF16 RoPE field is changed. FP8 no-PE bytes and all scale bytes
    # remain bit-identical to the native cache entry.
    write_rope_bf16(packed, local_slots, canonical_rope, 1)
    return packed


@torch.no_grad()
def canonicalize_indexer_key(
    *,
    cache: torch.Tensor,
    slots: torch.Tensor,
    page_size: int,
    source_positions: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Capture position-zero, pre-Hadamard BF16 Indexer K.

    Native Indexer cache stores ``FP8(H([nope64, RoPE(pos, rope64)]))``.  The
    normalized Hadamard transform is self-inverse, so capture must dequantize
    all 128 values, apply H once to recover the pre-Hadamard vector, remove
    RoPE from the final 64 values, and retain BF16 rather than requantizing.
    """

    slots = _long_vector(slots, name="Indexer capture slots")
    source_positions = _long_vector(
        source_positions,
        name="Indexer source positions",
        expected_rows=int(slots.numel()),
    ).to(device=cache.device)
    slots = slots.to(device=cache.device)
    packed = _read_indexer_packed(cache, slots, int(page_size))
    if (
        packed.dtype != torch.uint8
        or packed.ndim != 2
        or int(packed.shape[1]) != INDEXER_PACKED_RECORD_BYTES
    ):
        raise ValueError("native Indexer cache must contain 132-byte rows")
    count = int(packed.shape[0])
    scale = packed[:, 128:132].contiguous().view(torch.float32).reshape(count)
    if not bool((torch.isfinite(scale) & (scale > 0)).all().item()):
        raise ValueError("Indexer capture contains an invalid FP8 scale")
    quantized = packed[:, :128].contiguous().view(torch.float8_e4m3fn).float()
    post_hadamard = quantized * scale[:, None]
    # normalized H is its own inverse
    pre_hadamard = _indexer_rotate_activation(post_hadamard.contiguous())
    canonical_rope = reposition_rope(
        pre_hadamard[:, 64:].contiguous().to(torch.bfloat16),
        source_positions,
        torch.zeros_like(source_positions),
        freqs_cis,
    )
    canonical = torch.cat(
        (pre_hadamard[:, :64].to(torch.bfloat16), canonical_rope), dim=-1
    ).contiguous()
    result = _as_contiguous_uint8_rows(canonical)
    if tuple(result.shape) != (count, INDEXER_POSITIONLESS_RECORD_BYTES):
        raise AssertionError("canonical Indexer record width changed")
    return result


def _validate_capture_components(
    *, layer_id: int, compress_ratio: int, components: Mapping[str, torch.Tensor]
) -> None:
    if type(layer_id) is not int or layer_id < 0:
        raise ValueError("capture layer_id must be non-negative")
    expected = set(_expected_domains(compress_ratio))
    if set(components) != expected:
        raise ValueError(
            "layer capture domains are incomplete "
            f"(missing={sorted(expected-set(components))}, "
            f"extra={sorted(set(components)-expected)})"
        )
    for domain, tensor in components.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.dtype != torch.uint8
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"capture component {domain!r} must be contiguous uint8[rows,bytes]")
    swa = components[DOMAIN_SWA]
    if int(swa.shape[1]) != PACKED_LATENT_BYTES:
        raise ValueError("SWA capture rows must contain 584 bytes")
    compressed_domain = DOMAIN_C4 if int(compress_ratio) == 4 else DOMAIN_C128
    compressed = components[compressed_domain]
    if int(compressed.shape[1]) != PACKED_LATENT_BYTES:
        raise ValueError("compressed capture rows must contain 584 bytes")
    if int(swa.shape[0]) != int(compressed.shape[0]) * int(compress_ratio):
        raise ValueError("SWA/compressed capture row geometry differs")
    if int(compress_ratio) == 4:
        indexer = components[DOMAIN_INDEXER]
        if tuple(indexer.shape) != (
            int(compressed.shape[0]),
            INDEXER_POSITIONLESS_RECORD_BYTES,
        ):
            raise ValueError("Indexer canonical rows must be BF16[128] bytes")
        if int(components[DOMAIN_INDEXER_STATE].shape[0]) != int(
            components[DOMAIN_C4_ATTENTION_STATE].shape[0]
        ):
            raise ValueError("C4 attention/Indexer state anchors differ")


@dataclass(frozen=True)
class LayerCaptureBundle:
    """Complete canonical device components for one reusable model layer."""

    layer_id: int
    compress_ratio: int
    components: Mapping[str, torch.Tensor]
    position_semantics: Mapping[str, str]
    ready_events: Tuple[object, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_capture_components(
            layer_id=self.layer_id,
            compress_ratio=self.compress_ratio,
            components=self.components,
        )
        expected_semantics = {
            DOMAIN_SWA: PACKED_LATENT_POSITION_SEMANTICS,
            (DOMAIN_C4 if self.compress_ratio == 4 else DOMAIN_C128): (
                PACKED_LATENT_POSITION_SEMANTICS
            ),
            (
                DOMAIN_C4_ATTENTION_STATE
                if self.compress_ratio == 4
                else DOMAIN_C128_ATTENTION_STATE
            ): ATTENTION_STATE_POSITION_SEMANTICS,
        }
        if self.compress_ratio == 4:
            expected_semantics[DOMAIN_INDEXER] = INDEXER_POSITION_SEMANTICS
            expected_semantics[DOMAIN_INDEXER_STATE] = (
                INDEXER_STATE_POSITION_SEMANTICS
            )
        if dict(self.position_semantics) != expected_semantics:
            raise ValueError("capture position semantics are incomplete")

    @property
    def device_nbytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.components.values()
        )

    def stage_into(
        self,
        gpu_store,
        staged_epoch,
        *,
        stream: Optional[object] = None,
        non_blocking: bool = True,
    ) -> None:
        """Capture Tensor components directly into one persistent epoch slot."""

        if self.ready_events:
            first = self.components[DOMAIN_SWA]
            if first.device.type != "cuda":
                raise ValueError("CPU capture bundle cannot own CUDA ready events")
            copy_stream = (
                stream
                if stream is not None
                else torch.cuda.current_stream(first.device)
            )
            for event in self.ready_events:
                copy_stream.wait_event(event)

        for domain in _expected_domains(self.compress_ratio):
            gpu_store.capture_component(
                staged_epoch,
                domain=domain,
                layer_id=self.layer_id,
                payload=self.components[domain],
                stream=stream,
                non_blocking=non_blocking,
            )

    def export_cpu_components(self) -> Mapping[str, bytes]:
        """Explicit publication bridge for the CPU artifact controller.

        GPU publication should use :meth:`stage_into` and never take this
        device-to-host path.  Composite two-phase publication may call this
        method once to build the controller's bytes payload from the exact same
        finalized generation.  Rows retain staging order, including checkpoint
        anchors followed by the terminal state row.
        """

        exported = {}
        for event in self.ready_events:
            event.synchronize()
        for domain in _expected_domains(self.compress_ratio):
            host = (
                self.components[domain]
                .detach()
                .to(device="cpu", non_blocking=False)
                .contiguous()
            )
            exported[domain] = host.numpy().tobytes(order="C")
        return MappingProxyType(exported)


# Compatibility name used by the first adapter draft.
LayerCaptureComponents = LayerCaptureBundle


def make_layer_capture_bundle(
    *,
    layer_id: int,
    compress_ratio: int,
    components: Mapping[str, torch.Tensor],
    ready_events: Sequence[object] = (),
) -> LayerCaptureBundle:
    """Bind already-canonical model outputs without rereading an SWA ring."""

    ratio = int(compress_ratio)
    semantics = {
        DOMAIN_SWA: PACKED_LATENT_POSITION_SEMANTICS,
        (DOMAIN_C4 if ratio == 4 else DOMAIN_C128): (
            PACKED_LATENT_POSITION_SEMANTICS
        ),
        (
            DOMAIN_C4_ATTENTION_STATE
            if ratio == 4
            else DOMAIN_C128_ATTENTION_STATE
        ): ATTENTION_STATE_POSITION_SEMANTICS,
    }
    if ratio == 4:
        semantics[DOMAIN_INDEXER] = INDEXER_POSITION_SEMANTICS
        semantics[DOMAIN_INDEXER_STATE] = INDEXER_STATE_POSITION_SEMANTICS
    return LayerCaptureBundle(
        int(layer_id),
        ratio,
        MappingProxyType(dict(components)),
        MappingProxyType(semantics),
        tuple(ready_events),
    )


@dataclass(frozen=True)
class LayerCaptureChunkReceipt:
    """Rows copied by one overwrite-safe prefill capture hook."""

    layer_id: int
    swa_row_begin: int
    swa_row_end: int
    component_spans: Mapping[str, Tuple[int, int]]


@dataclass
class LayerCaptureStaging:
    """Full-segment device staging filled incrementally during prefill.

    The caller must invoke :func:`capture_chunk_components` after the native
    cache writer has produced a chunk and *before* a later chunk can reuse its
    SWA physical slots.  Coverage is tracked using CPU integers; checking or
    finalizing this object never scans a device mask or copies payload bytes to
    CPU.  The finalized tensors are handed directly to
    ``SharedLatentGPUStore.capture_component``.
    """

    layer_id: int
    compress_ratio: int
    segment_length: int
    components: Dict[str, torch.Tensor]
    _coverage: Dict[str, _RowCoverage] = field(repr=False)
    _ready_events: list[object] = field(default_factory=list, repr=False)
    _state: str = field(default="capturing", init=False, repr=False)

    def __post_init__(self) -> None:
        _expected_domains(self.compress_ratio)
        if self.segment_length <= 0 or self.segment_length % self.compress_ratio:
            raise ValueError("capture staging length is not compressor aligned")
        if set(self.components) != set(_expected_domains(self.compress_ratio)):
            raise ValueError("capture staging component domains are incomplete")
        if set(self._coverage) != set(self.components):
            raise ValueError("capture staging coverage domains are incomplete")
        for domain, tensor in self.components.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 2
                or tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"capture staging {domain!r} must be contiguous uint8[rows,bytes]"
                )
            if self._coverage[domain].row_count != int(tensor.shape[0]):
                raise ValueError(f"capture staging {domain!r} row geometry changed")

    @property
    def device_nbytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.components.values()
        )

    @property
    def missing_rows(self) -> Mapping[str, Tuple[Tuple[int, int], ...]]:
        return MappingProxyType(
            {
                domain: coverage.missing_spans
                for domain, coverage in self._coverage.items()
                if not coverage.complete
            }
        )

    def capture_canonical_rows(
        self, *, domain: str, row_begin: int, values: torch.Tensor
    ) -> Tuple[int, int]:
        """Copy already-canonical rows into this layer's device staging."""

        if self._state != "capturing":
            raise RuntimeError("capture staging is no longer writable")
        domain = str(domain)
        target = self.components.get(domain)
        coverage = self._coverage.get(domain)
        if target is None or coverage is None:
            raise ValueError(f"capture staging has no domain {domain!r}")
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 2
            or values.dtype != torch.uint8
            or not values.is_contiguous()
            or int(values.shape[1]) != int(target.shape[1])
        ):
            raise ValueError(
                f"canonical {domain!r} rows must be contiguous uint8 with "
                f"record width {int(target.shape[1])}"
            )
        begin = int(row_begin)
        end = begin + int(values.shape[0])
        coverage.validate(begin, end, domain=domain)
        target.narrow(0, begin, end - begin).copy_(values, non_blocking=True)
        if target.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(target.device))
            self._ready_events.append(event)
        coverage.add(begin, end, domain=domain)
        return begin, end

    def finalize(self) -> LayerCaptureBundle:
        """Seal a complete layer without reallocating or gathering payloads."""

        if self._state != "capturing":
            raise RuntimeError("capture staging was already finalized or aborted")
        missing = dict(self.missing_rows)
        if missing:
            raise ValueError(f"capture staging is incomplete: {missing}")
        bundle = make_layer_capture_bundle(
            layer_id=self.layer_id,
            compress_ratio=self.compress_ratio,
            components=self.components,
            ready_events=self._ready_events,
        )
        self._state = "finalized"
        return bundle

    def abort(self) -> None:
        if self._state == "finalized":
            raise RuntimeError("a finalized capture staging cannot be aborted")
        self._state = "aborted"


def begin_layer_capture_bundle(
    *,
    token_to_kv_pool,
    layer_id: int,
    compress_ratio: int,
    segment_length: int,
    device: Optional[object] = None,
) -> LayerCaptureStaging:
    """Allocate each layer/domain staging tensor once for a whole segment."""

    layer_id = int(layer_id)
    ratio = int(compress_ratio)
    length = int(segment_length)
    _expected_domains(ratio)
    if layer_id < 0 or length <= 0 or length % ratio:
        raise ValueError("capture layer/segment geometry is invalid")
    if device is None:
        device = token_to_kv_pool.get_swa_raw_key_buffer_radix(layer_id).device
    compressed_rows = length // ratio
    state_rows = len(_checkpoint_anchors(length))
    attention_domain = (
        DOMAIN_C4_ATTENTION_STATE
        if ratio == 4
        else DOMAIN_C128_ATTENTION_STATE
    )
    attention_pool = token_to_kv_pool.get_attention_compress_states(layer_id)
    geometry: Dict[str, Tuple[int, int]] = {
        DOMAIN_SWA: (length, PACKED_LATENT_BYTES),
        (DOMAIN_C4 if ratio == 4 else DOMAIN_C128): (
            compressed_rows,
            PACKED_LATENT_BYTES,
        ),
        attention_domain: (
            state_rows,
            _state_record_bytes(
                attention_pool, ratio=ratio, domain=attention_domain
            ),
        ),
    }
    if ratio == 4:
        indexer_pool = token_to_kv_pool.get_indexer_compress_states(layer_id)
        geometry[DOMAIN_INDEXER] = (
            compressed_rows,
            INDEXER_POSITIONLESS_RECORD_BYTES,
        )
        geometry[DOMAIN_INDEXER_STATE] = (
            state_rows,
            _state_record_bytes(
                indexer_pool, ratio=ratio, domain=DOMAIN_INDEXER_STATE
            ),
        )
    components = {
        domain: torch.empty(shape, dtype=torch.uint8, device=device)
        for domain, shape in geometry.items()
    }
    coverage = {
        domain: _RowCoverage(int(shape[0])) for domain, shape in geometry.items()
    }
    return LayerCaptureStaging(
        layer_id,
        ratio,
        length,
        components,
        coverage,
    )


def _capture_state_terminal_records(
    *,
    state_pool,
    ratio: int,
    domain: str,
    terminal_group_slots: torch.Tensor,
) -> torch.Tensor:
    """Gather restart records immediately when their anchor is produced."""

    terminals = _long_vector(
        terminal_group_slots, name=f"{domain} terminal state group slots"
    )
    required_groups = _state_required_groups(domain)
    group_width = _state_group_width(state_pool, ratio=ratio)
    groups = _state_group_view(state_pool.kv_score_buffer.kv_score, group_width)
    terminals = terminals.to(device=groups.device)
    offsets = torch.arange(
        1 - required_groups,
        1,
        device=groups.device,
        dtype=torch.long,
    )
    indices = terminals[:, None] + offsets[None, :]
    if bool((indices < 0).any().item()) or bool(
        (indices >= int(groups.shape[0])).any().item()
    ):
        raise ValueError(f"{domain} terminal state group slot is out of range")
    records = groups.index_select(0, indices.reshape(-1)).contiguous()
    records = records.view(int(terminals.numel()), -1).view(torch.uint8)
    expected = _state_record_bytes(state_pool, ratio=ratio, domain=domain)
    if tuple(records.shape) != (int(terminals.numel()), expected):
        raise AssertionError("incremental compressor state width changed")
    return records


@torch.no_grad()
def capture_compressed_components(
    staging: LayerCaptureStaging,
    *,
    token_to_kv_pool,
    freqs_cis: torch.Tensor,
    compressed_row_begin: int,
    compressed_slots: torch.Tensor,
    source_positions: Optional[torch.Tensor] = None,
    capture_core: bool = True,
    capture_indexer: Optional[bool] = None,
) -> Mapping[str, Tuple[int, int]]:
    """Capture completed C4/C128 and Indexer rows independently of SWA.

    C4 and Indexer writers can finish at different hook points.  Callers may
    first use ``capture_core=True, capture_indexer=False`` and later reverse
    the flags with the same row range; per-domain coverage remains exact-once.
    """

    if not isinstance(staging, LayerCaptureStaging):
        raise TypeError("compressed capture requires LayerCaptureStaging")
    ratio = int(staging.compress_ratio)
    if capture_indexer is None:
        capture_indexer = ratio == 4
    if type(capture_core) is not bool or type(capture_indexer) is not bool:
        raise TypeError("compressed capture selectors must be booleans")
    if ratio != 4 and capture_indexer:
        raise ValueError("C128 layers have no Indexer component")
    if not capture_core and not capture_indexer:
        raise ValueError("compressed capture selected no component")
    slots = _long_vector(compressed_slots, name=f"chunk C{ratio} slots")
    begin = int(compressed_row_begin)
    end = begin + int(slots.numel())
    if source_positions is None:
        source_positions = (
            torch.arange(begin, end, device=slots.device, dtype=torch.long) + 1
        ) * ratio - 1
    else:
        source_positions = _long_vector(
            source_positions,
            name="chunk compressor completion positions",
            expected_rows=int(slots.numel()),
        )
    _require_unique_nonnegative_slots(
        slots,
        expected=int(slots.numel()),
        name=f"chunk C{ratio} slots",
    )
    layer_id = int(staging.layer_id)
    spans = {}
    if capture_core:
        domain = DOMAIN_C4 if ratio == 4 else DOMAIN_C128
        rows = canonicalize_packed_latent(
            cache=token_to_kv_pool.get_extra_key_buffer(layer_id),
            slots=slots,
            page_size=int(token_to_kv_pool.get_extra_key_page_size(layer_id)),
            source_positions=source_positions,
            freqs_cis=freqs_cis,
        )
        spans[domain] = staging.capture_canonical_rows(
            domain=domain, row_begin=begin, values=rows
        )
    if capture_indexer:
        rows = canonicalize_indexer_key(
            cache=token_to_kv_pool.get_index_k_with_scale_buffer(layer_id),
            slots=slots,
            page_size=int(token_to_kv_pool.get_index_k_page_size()),
            source_positions=source_positions,
            freqs_cis=freqs_cis,
        )
        spans[DOMAIN_INDEXER] = staging.capture_canonical_rows(
            domain=DOMAIN_INDEXER, row_begin=begin, values=rows
        )
    return MappingProxyType(spans)


@torch.no_grad()
def capture_state_components(
    staging: LayerCaptureStaging,
    *,
    token_to_kv_pool,
    state_row_begin: int,
    attention_terminal_group_slots: Optional[torch.Tensor] = None,
    indexer_terminal_group_slots: Optional[torch.Tensor] = None,
) -> Mapping[str, Tuple[int, int]]:
    """Capture checkpoint/terminal recurrent state at its production hook."""

    if not isinstance(staging, LayerCaptureStaging):
        raise TypeError("state capture requires LayerCaptureStaging")
    if (
        attention_terminal_group_slots is None
        and indexer_terminal_group_slots is None
    ):
        raise ValueError("state capture selected no component")
    layer_id = int(staging.layer_id)
    ratio = int(staging.compress_ratio)
    begin = int(state_row_begin)
    spans = {}
    attention_domain = (
        DOMAIN_C4_ATTENTION_STATE
        if ratio == 4
        else DOMAIN_C128_ATTENTION_STATE
    )
    if attention_terminal_group_slots is not None:
        records = _capture_state_terminal_records(
            state_pool=token_to_kv_pool.get_attention_compress_states(layer_id),
            ratio=ratio,
            domain=attention_domain,
            terminal_group_slots=attention_terminal_group_slots,
        )
        spans[attention_domain] = staging.capture_canonical_rows(
            domain=attention_domain, row_begin=begin, values=records
        )
    if indexer_terminal_group_slots is not None:
        if ratio != 4:
            raise ValueError("only C4 layers own Indexer compressor state")
        records = _capture_state_terminal_records(
            state_pool=token_to_kv_pool.get_indexer_compress_states(layer_id),
            ratio=ratio,
            domain=DOMAIN_INDEXER_STATE,
            terminal_group_slots=indexer_terminal_group_slots,
        )
        spans[DOMAIN_INDEXER_STATE] = staging.capture_canonical_rows(
            domain=DOMAIN_INDEXER_STATE, row_begin=begin, values=records
        )
    return MappingProxyType(spans)


@torch.no_grad()
def capture_chunk_components(
    staging: LayerCaptureStaging,
    *,
    token_to_kv_pool,
    freqs_cis: torch.Tensor,
    row_begin: int,
    full_cache_slots: torch.Tensor,
    source_positions: Optional[torch.Tensor] = None,
    canonical_swa_rows: Optional[torch.Tensor] = None,
    compressed_row_begin: Optional[int] = None,
    compressed_slots: Optional[torch.Tensor] = None,
    compressed_source_positions: Optional[torch.Tensor] = None,
    state_row_begin: Optional[int] = None,
    attention_state_terminal_slots: Optional[torch.Tensor] = None,
    indexer_state_terminal_slots: Optional[torch.Tensor] = None,
) -> LayerCaptureChunkReceipt:
    """Capture one produced chunk before native SWA/ring rows are reused.

    ``full_cache_slots`` enumerates exactly the token rows beginning at
    ``row_begin``.  A fused writer which can alias its 128-row physical ring
    *within this very call* must pass its position-zero packed output as
    ``canonical_swa_rows``; those rows are copied directly and the ring is not
    read.  Without that argument the cache-read path remains available but
    fails closed on physical-slot aliases.  Compressed/state rows are optional
    and may be captured at their own completion hooks.  All canonical outputs
    remain device resident.
    """

    if not isinstance(staging, LayerCaptureStaging):
        raise TypeError("capture_chunk_components requires LayerCaptureStaging")
    if staging._state != "capturing":
        raise RuntimeError("capture staging is no longer writable")
    layer_id = int(staging.layer_id)
    full_slots = _long_vector(full_cache_slots, name="chunk full-cache slots")
    begin = int(row_begin)
    end = begin + int(full_slots.numel())
    if source_positions is None:
        source_positions = torch.arange(
            begin, end, device=full_slots.device, dtype=torch.long
        )
    else:
        source_positions = _long_vector(
            source_positions,
            name="chunk source positions",
            expected_rows=int(full_slots.numel()),
        )
    if canonical_swa_rows is not None:
        if (
            not isinstance(canonical_swa_rows, torch.Tensor)
            or canonical_swa_rows.ndim != 2
            or canonical_swa_rows.dtype != torch.uint8
            or not canonical_swa_rows.is_contiguous()
            or tuple(canonical_swa_rows.shape)
            != (int(full_slots.numel()), PACKED_LATENT_BYTES)
        ):
            raise ValueError(
                "canonical_swa_rows must be contiguous uint8[n,584]"
            )
        swa_rows = canonical_swa_rows
    else:
        swa_slots = token_to_kv_pool.translate_loc_from_full_to_swa(
            full_slots
        ).long()
        _require_unique_nonnegative_slots(
            swa_slots, expected=int(full_slots.numel()), name="chunk SWA slots"
        )
        swa_rows = canonicalize_packed_latent(
            cache=token_to_kv_pool.get_swa_raw_key_buffer_radix(layer_id),
            slots=swa_slots,
            page_size=int(token_to_kv_pool.swa_kv_pool.page_size),
            source_positions=source_positions,
            freqs_cis=freqs_cis,
        )
    spans: Dict[str, Tuple[int, int]] = {
        DOMAIN_SWA: staging.capture_canonical_rows(
            domain=DOMAIN_SWA, row_begin=begin, values=swa_rows
        )
    }

    has_compressed = compressed_slots is not None
    if has_compressed != (compressed_row_begin is not None):
        raise ValueError(
            "compressed_slots and compressed_row_begin must be supplied together"
        )
    if has_compressed:
        spans.update(
            capture_compressed_components(
                staging,
                token_to_kv_pool=token_to_kv_pool,
                freqs_cis=freqs_cis,
                compressed_row_begin=int(compressed_row_begin),
                compressed_slots=compressed_slots,
                source_positions=compressed_source_positions,
            )
        )

    supplied_state = (
        attention_state_terminal_slots is not None
        or indexer_state_terminal_slots is not None
    )
    if supplied_state != (state_row_begin is not None):
        raise ValueError(
            "state_row_begin and terminal state slots must be supplied together"
        )
    if supplied_state:
        spans.update(
            capture_state_components(
                staging,
                token_to_kv_pool=token_to_kv_pool,
                state_row_begin=int(state_row_begin),
                attention_terminal_group_slots=(
                    attention_state_terminal_slots
                ),
                indexer_terminal_group_slots=indexer_state_terminal_slots,
            )
        )
    return LayerCaptureChunkReceipt(
        layer_id,
        begin,
        end,
        MappingProxyType(spans),
    )


def finalize_layer_capture_bundle(
    staging: LayerCaptureStaging,
) -> LayerCaptureBundle:
    """Named finalization hook for backend/model integrations."""

    if not isinstance(staging, LayerCaptureStaging):
        raise TypeError("finalize requires LayerCaptureStaging")
    return staging.finalize()


def _capture_state_records(
    *,
    token_to_kv_pool,
    full_slots: torch.Tensor,
    state_pool,
    ratio: int,
    anchors: Tuple[int, ...],
    domain: str,
) -> torch.Tensor:
    group_width = _state_group_width(state_pool, ratio=ratio)
    state_groups = _state_group_view(
        state_pool.kv_score_buffer.kv_score, group_width
    )
    required_groups = _state_required_groups(domain)
    rows = []
    all_terminal_slots = []
    for anchor in anchors:
        state_slots = compute_compressed_slots(
            full_slots=full_slots[:anchor],
            full_to_swa=token_to_kv_pool.full_to_swa_index_mapping,
            swa_page_size=int(token_to_kv_pool.swa_page_size),
            ring_size=int(state_pool.ring_size),
            compress_ratio=int(ratio),
            seq_len=int(anchor),
            state_group_width=group_width,
        ).long()
        terminal = state_slots[-required_groups:]
        if int(terminal.numel()) != required_groups:
            raise ValueError(f"{domain} restart state at anchor {anchor} is incomplete")
        if required_groups == 2 and not bool(
            (terminal[1:] == terminal[:-1] + 1).all().item()
        ):
            raise ValueError("C4 terminal state groups are not consecutive")
        rows.append(state_groups.index_select(0, terminal).contiguous())
        all_terminal_slots.append(terminal)

    # If two historical anchors map to the same group, reading after the full
    # segment cannot recover both historical values.  Fail closed rather than
    # publishing a plausible-looking but temporally wrong checkpoint.
    flattened_slots = torch.cat(all_terminal_slots)
    if int(torch.unique(flattened_slots).numel()) != int(flattened_slots.numel()):
        raise ValueError(
            f"{domain} checkpoint groups alias after full prefill; capture "
            "checkpoints incrementally"
        )
    result = torch.stack(rows).contiguous().view(len(rows), -1)
    result = result.view(torch.uint8)
    expected_width = _state_record_bytes(
        state_pool, ratio=ratio, domain=domain
    )
    if tuple(result.shape) != (len(rows), expected_width):
        raise AssertionError("compressor state record width changed")
    return result


@torch.no_grad()
def capture_layer_bundle(
    *,
    token_to_kv_pool,
    req_to_token: torch.Tensor,
    page_table: torch.Tensor,
    request_pool_index: int,
    page_table_row: int,
    segment_length: int,
    layer_id: int,
    compress_ratio: int,
    freqs_cis: torch.Tensor,
) -> LayerCaptureBundle:
    """Compatibility capture for short/non-aliased cache layouts only.

    This entry point is valid only while every requested SWA/state slot still
    owns the corresponding source row.  Long-prefill ring aliasing is detected
    and rejected.  Production long-prefill must use
    :func:`begin_layer_capture_bundle` plus :func:`capture_chunk_components`
    before every overwrite; it must not call this function after the final
    chunk.
    """

    length = int(segment_length)
    ratio = int(compress_ratio)
    _expected_domains(ratio)
    if length <= 0 or length % ratio:
        raise ValueError("segment length must be positive and compressor aligned")
    full_slots = _long_vector(
        req_to_token[int(request_pool_index), :length],
        name="offline full-cache slots",
        expected_rows=length,
    )
    positions = torch.arange(length, device=full_slots.device, dtype=torch.long)

    swa_slots = token_to_kv_pool.translate_loc_from_full_to_swa(full_slots).long()
    _require_unique_nonnegative_slots(
        swa_slots, expected=length, name="offline SWA slots"
    )
    swa_cache = token_to_kv_pool.get_swa_raw_key_buffer_radix(int(layer_id))
    components: Dict[str, torch.Tensor] = {
        DOMAIN_SWA: canonicalize_packed_latent(
            cache=swa_cache,
            slots=swa_slots,
            page_size=int(token_to_kv_pool.swa_kv_pool.page_size),
            source_positions=positions,
            freqs_cis=freqs_cis,
        )
    }

    compressed_page_size = int(
        token_to_kv_pool.get_extra_key_page_size(int(layer_id))
    )
    compressed_slots = compute_paged_compressed_slots(
        page_table=page_table,
        req_idx=int(page_table_row),
        seq_len=length,
        compress_ratio=ratio,
        compressed_page_size=compressed_page_size,
    ).long()
    block_count = length // ratio
    _require_unique_nonnegative_slots(
        compressed_slots,
        expected=block_count,
        name=f"offline C{ratio} slots",
    )
    completion_positions = (
        torch.arange(
            1, block_count + 1, device=positions.device, dtype=torch.long
        )
        * ratio
        - 1
    )
    compressed_domain = DOMAIN_C4 if ratio == 4 else DOMAIN_C128
    components[compressed_domain] = canonicalize_packed_latent(
        cache=token_to_kv_pool.get_extra_key_buffer(int(layer_id)),
        slots=compressed_slots,
        page_size=compressed_page_size,
        source_positions=completion_positions,
        freqs_cis=freqs_cis,
    )
    if ratio == 4:
        components[DOMAIN_INDEXER] = canonicalize_indexer_key(
            cache=token_to_kv_pool.get_index_k_with_scale_buffer(int(layer_id)),
            slots=compressed_slots,
            page_size=int(token_to_kv_pool.get_index_k_page_size()),
            source_positions=completion_positions,
            freqs_cis=freqs_cis,
        )

    anchors = _checkpoint_anchors(length)
    attention_domain = (
        DOMAIN_C4_ATTENTION_STATE
        if ratio == 4
        else DOMAIN_C128_ATTENTION_STATE
    )
    components[attention_domain] = _capture_state_records(
        token_to_kv_pool=token_to_kv_pool,
        full_slots=full_slots,
        state_pool=token_to_kv_pool.get_attention_compress_states(int(layer_id)),
        ratio=ratio,
        anchors=anchors,
        domain=attention_domain,
    )
    if ratio == 4:
        components[DOMAIN_INDEXER_STATE] = _capture_state_records(
            token_to_kv_pool=token_to_kv_pool,
            full_slots=full_slots,
            state_pool=token_to_kv_pool.get_indexer_compress_states(int(layer_id)),
            ratio=ratio,
            anchors=anchors,
            domain=DOMAIN_INDEXER_STATE,
        )
    return make_layer_capture_bundle(
        layer_id=int(layer_id),
        compress_ratio=ratio,
        components=components,
    )


# Compatibility function used by the first adapter draft.
capture_layer_components = capture_layer_bundle


def _compute_state_target_slots(
    *, token_to_kv_pool, full_cache_slots: torch.Tensor, state_pool, ratio: int
) -> torch.Tensor:
    """Return one terminal *group* slot aligned with every forward row."""

    full_slots = _long_vector(full_cache_slots, name="state full-cache slots")
    swa_slots = token_to_kv_pool.translate_loc_from_full_to_swa(full_slots).long()
    ring_size = int(state_pool.ring_size)
    _state_group_width(state_pool, ratio=ratio)
    state_loc = (
        (swa_slots // int(token_to_kv_pool.swa_page_size)) * ring_size
        + swa_slots % ring_size
    )
    # This deliberately divides by compress_ratio, not group_width. It mirrors
    # create_paged_compressor_data/get_raw_loc, including online-C128.
    return (state_loc // int(ratio)).long()


def _metadata_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _metadata_int_tuple(values: object, name: str) -> Tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.ndim != 1:
            raise ValueError(f"{name} tensor must be one-dimensional CPU metadata")
        values = values.tolist()
    try:
        result = tuple(_metadata_int(value, name) for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer sequence") from error
    return result


@dataclass(frozen=True)
class DirtyCompressorIsland:
    """One compact online island, using flattened forward row coordinates."""

    flat_begin: int
    flat_end: int
    request_row_begin: int
    request_row_end: int
    token_begin: int
    token_end: int
    state_slot_indices: Tuple[int, ...]
    completion_output_rows: Tuple[int, ...]

    @property
    def state_slots(self) -> Tuple[int, ...]:
        return self.state_slot_indices

    @property
    def completion_rows(self) -> Tuple[int, ...]:
        return self.completion_output_rows


@dataclass(frozen=True)
class DirtyRequestWorkset:
    """Duck-compatible request workset consumed by dirty-only compressors."""

    request_index: int
    flat_row_offset: int
    row_count: int
    seq_len_before: int
    islands: Tuple[DirtyCompressorIsland, ...]


def _canonicalize_dirty_island(
    raw: object,
    *,
    request_index: int,
    flat_row_offset: int,
    row_count: int,
    seq_len_before: int,
    compress_ratio: int,
    state_slots_are_physical: bool,
) -> DirtyCompressorIsland:
    try:
        raw_state_slots = _object_field(
            raw, "state_slot_indices", "state_slots"
        )
    except ValueError:
        raw_state_slots = ()
    island = DirtyCompressorIsland(
        _metadata_int(_object_field(raw, "flat_begin"), "flat_begin"),
        _metadata_int(_object_field(raw, "flat_end"), "flat_end"),
        _metadata_int(
            _object_field(raw, "request_row_begin"), "request_row_begin"
        ),
        _metadata_int(_object_field(raw, "request_row_end"), "request_row_end"),
        _metadata_int(_object_field(raw, "token_begin"), "token_begin"),
        _metadata_int(_object_field(raw, "token_end"), "token_end"),
        _metadata_int_tuple(
            raw_state_slots,
            "state_slot_indices",
        ),
        _metadata_int_tuple(
            _object_field(raw, "completion_output_rows", "completion_rows"),
            "completion_output_rows",
        ),
    )
    if (
        island.flat_begin < flat_row_offset
        or island.flat_end <= island.flat_begin
        or island.flat_end > flat_row_offset + row_count
        or island.request_row_begin < 0
        or island.request_row_end <= island.request_row_begin
        or island.request_row_end > row_count
        or island.token_begin < 0
        or island.token_end <= island.token_begin
    ):
        raise ValueError(f"request {request_index} dirty island is out of range")
    lengths = {
        island.flat_end - island.flat_begin,
        island.request_row_end - island.request_row_begin,
        island.token_end - island.token_begin,
    }
    if len(lengths) != 1:
        raise ValueError("dirty island flat/request/token lengths differ")
    if island.flat_begin != flat_row_offset + island.request_row_begin:
        raise ValueError("dirty island request rows do not map to flat rows")
    if island.token_begin != seq_len_before + island.request_row_begin:
        raise ValueError("dirty island token range differs from request history")
    if (
        state_slots_are_physical
        and not island.state_slot_indices
        and island.token_begin != 0
    ):
        raise ValueError(
            "a non-initial dirty island requires restored compressor state slots"
        )
    if state_slots_are_physical and any(
        slot < 0 for slot in island.state_slot_indices
    ):
        raise ValueError("dirty island state slots must be non-negative")
    if state_slots_are_physical and island.state_slot_indices != tuple(
        sorted(set(island.state_slot_indices))
    ):
        raise ValueError("dirty island state slots must be increasing and unique")
    if state_slots_are_physical and island.token_begin > 0:
        if island.token_begin % int(compress_ratio):
            raise ValueError("dirty island restart must be compressor aligned")
        required_state_slots = 2 if int(compress_ratio) == 4 else 1
        if len(island.state_slot_indices) != required_state_slots:
            raise ValueError(
                f"C{compress_ratio} dirty island requires "
                f"{required_state_slots} terminal state slots"
            )
        if required_state_slots == 2 and (
            island.state_slot_indices[1] != island.state_slot_indices[0] + 1
        ):
            raise ValueError("C4 dirty island state slots must be consecutive")
    expected_completions = tuple(
        flat_row
        for flat_row in range(island.flat_begin, island.flat_end)
        if (
            island.token_begin + (flat_row - island.flat_begin) + 1
        )
        % int(compress_ratio)
        == 0
    )
    if island.completion_output_rows != expected_completions:
        raise ValueError(
            "dirty island completion rows are not exact compressor completions"
        )
    return island


def canonicalize_dirty_request_worksets(
    dirty_worksets: Sequence[object],
    *,
    q_rows: int,
    compress_ratio: int,
    state_slots_are_physical: bool = True,
) -> Tuple[DirtyRequestWorkset, ...]:
    """Validate and canonicalize compressor.py's public duck-typed metadata."""

    if type(state_slots_are_physical) is not bool:
        raise TypeError("state_slots_are_physical must be boolean")
    raw_items = tuple(dirty_worksets)
    result = []
    expected_offset = 0
    for expected_index, raw in enumerate(raw_items):
        request_index = _metadata_int(
            _object_field(raw, "request_index"), "request_index"
        )
        flat_row_offset = _metadata_int(
            _object_field(raw, "flat_row_offset"), "flat_row_offset"
        )
        row_count = _metadata_int(_object_field(raw, "row_count"), "row_count")
        seq_len_before = _metadata_int(
            _object_field(raw, "seq_len_before"), "seq_len_before"
        )
        if (
            request_index != expected_index
            or flat_row_offset != expected_offset
            or row_count < 0
            or seq_len_before < 0
        ):
            raise ValueError(
                "dirty request worksets must be ordered and tile the forward"
            )
        raw_islands = tuple(_object_field(raw, "islands"))
        islands = tuple(
            _canonicalize_dirty_island(
                island,
                request_index=request_index,
                flat_row_offset=flat_row_offset,
                row_count=row_count,
                seq_len_before=seq_len_before,
                compress_ratio=int(compress_ratio),
                state_slots_are_physical=bool(state_slots_are_physical),
            )
            for island in raw_islands
        )
        if islands != tuple(sorted(islands, key=lambda item: item.flat_begin)):
            raise ValueError("dirty islands must be ordered by flattened row")
        if any(
            left.flat_end > right.flat_begin
            for left, right in zip(islands, islands[1:])
        ):
            raise ValueError("dirty islands overlap")
        result.append(
            DirtyRequestWorkset(
                request_index,
                flat_row_offset,
                row_count,
                seq_len_before,
                islands,
            )
        )
        expected_offset += row_count
    if expected_offset != int(q_rows):
        raise ValueError("dirty request worksets do not cover every forward row")
    return tuple(result)


def _restore_tensor_identity(value: object, *, name: str) -> Tuple[object, ...]:
    """Return a mutation-sensitive identity without reading tensor payload."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    try:
        version: object = int(value._version)
    except RuntimeError:
        # Inference tensors have no version counter.  Their storage is bound to
        # the owning ForwardBatch generation by CompositeForwardResources.
        version = "inference-immutable"
    return (
        id(value),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        int(value.storage_offset()),
        version,
    )


def _restore_metadata_source_identity(
    value: Optional[object], *, name: str
) -> Tuple[object, ...]:
    """Bind tensor metadata without D2H, or copy an already-CPU sequence."""

    if value is None:
        return ("none",)
    if isinstance(value, torch.Tensor):
        return ("tensor",) + _restore_tensor_identity(value, name=name)
    return (
        "sequence",
        id(value),
        type(value).__qualname__,
        _metadata_int_tuple(value, name),
    )


def _dirty_state_slot_source_binding(
    *,
    token_to_kv_pool,
    full_cache_slots: torch.Tensor,
    req_to_token: Optional[torch.Tensor],
    request_pool_indices: Optional[object],
) -> Tuple[object, ...]:
    mapping = getattr(token_to_kv_pool, "full_to_swa_index_mapping", None)
    return (
        id(token_to_kv_pool),
        type(token_to_kv_pool).__qualname__,
        int(token_to_kv_pool.swa_page_size),
        _restore_tensor_identity(
            full_cache_slots, name="forward full-cache slots"
        ),
        _restore_metadata_source_identity(
            req_to_token, name="request-to-token mapping"
        ),
        _restore_metadata_source_identity(
            request_pool_indices, name="request pool indices"
        ),
        _restore_tensor_identity(mapping, name="full-to-SWA mapping"),
    )


def _dirty_state_pool_geometry(
    *, token_to_kv_pool, layer_id: int, compress_ratio: int
) -> Tuple[object, ...]:
    """Capture every pool field which can affect physical state addressing."""

    ratio = int(compress_ratio)
    page_size = int(token_to_kv_pool.swa_page_size)

    def pool_geometry(role: str, state_pool) -> Tuple[object, ...]:
        ring_size = int(state_pool.ring_size)
        pool_page_size = int(state_pool.swa_page_size)
        width = _state_group_width(state_pool, ratio=ratio)
        state = state_pool.kv_score_buffer.kv_score
        _state_group_view(state, width)
        if pool_page_size != page_size:
            raise ValueError(f"{role} state/SWA page geometry differs")
        return (
            role,
            ring_size,
            pool_page_size,
            width,
            tuple(int(item) for item in state.shape),
            str(state.dtype),
            str(state.device),
        )

    attention = token_to_kv_pool.get_attention_compress_states(int(layer_id))
    geometries = [pool_geometry("attention", attention)]
    if ratio == 4:
        indexer = token_to_kv_pool.get_indexer_compress_states(int(layer_id))
        indexer_geometry = pool_geometry("indexer", indexer)
        if indexer_geometry[1:4] != geometries[0][1:4]:
            raise ValueError("attention/Indexer state address geometry differs")
        geometries.append(indexer_geometry)
    return (page_size, tuple(geometries))


def _logical_dirty_worksets(
    worksets: Sequence[DirtyRequestWorkset],
) -> Tuple[DirtyRequestWorkset, ...]:
    """Erase only resolved physical slots, retaining exact island geometry."""

    return tuple(
        DirtyRequestWorkset(
            workset.request_index,
            workset.flat_row_offset,
            workset.row_count,
            workset.seq_len_before,
            tuple(
                DirtyCompressorIsland(
                    island.flat_begin,
                    island.flat_end,
                    island.request_row_begin,
                    island.request_row_end,
                    island.token_begin,
                    island.token_end,
                    (),
                    island.completion_output_rows,
                )
                for island in workset.islands
            ),
        )
        for workset in worksets
    )


@dataclass(frozen=True)
class DirtyStateSlotResolutionCertificate:
    """Forward/ratio proof for persistent physical dirty-state slot metadata."""

    forward_token: str
    compress_ratio: int
    q_rows: int
    logical_worksets: Tuple[DirtyRequestWorkset, ...]
    resolved_worksets: Tuple[DirtyRequestWorkset, ...]
    source_binding: Tuple[object, ...]
    pool_geometry: Tuple[object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.forward_token, str) or not self.forward_token:
            raise ValueError("dirty-state certificate needs a forward token")
        if self.compress_ratio not in (4, 128):
            raise ValueError("dirty-state certificate ratio is invalid")
        if type(self.q_rows) is not int or self.q_rows <= 0:
            raise ValueError("dirty-state certificate row count is invalid")
        logical = canonicalize_dirty_request_worksets(
            self.logical_worksets,
            q_rows=self.q_rows,
            compress_ratio=self.compress_ratio,
            state_slots_are_physical=False,
        )
        resolved = canonicalize_dirty_request_worksets(
            self.resolved_worksets,
            q_rows=self.q_rows,
            compress_ratio=self.compress_ratio,
            state_slots_are_physical=True,
        )
        if logical != self.logical_worksets or resolved != self.resolved_worksets:
            raise ValueError("dirty-state certificate worksets are not canonical")
        if _logical_dirty_worksets(resolved) != logical:
            raise ValueError("resolved dirty-state slots bind another workset")
        if any(
            island.token_begin == 0 and bool(island.state_slot_indices)
            for workset in resolved
            for island in workset.islands
        ):
            raise ValueError("initial dirty islands cannot restore prior state")
        if not isinstance(self.source_binding, tuple) or not self.source_binding:
            raise ValueError("dirty-state certificate source binding is absent")
        if not isinstance(self.pool_geometry, tuple) or not self.pool_geometry:
            raise ValueError("dirty-state certificate pool geometry is absent")

    def validate(
        self,
        *,
        forward_token: str,
        compress_ratio: int,
        q_rows: int,
        logical_worksets: Tuple[DirtyRequestWorkset, ...],
        source_binding: Tuple[object, ...],
        pool_geometry: Tuple[object, ...],
    ) -> None:
        if str(forward_token) != self.forward_token:
            raise ValueError("dirty-state certificate belongs to another forward")
        if int(compress_ratio) != self.compress_ratio or int(q_rows) != self.q_rows:
            raise ValueError("dirty-state certificate ratio/row geometry changed")
        if logical_worksets != self.logical_worksets:
            raise ValueError("dirty-state logical worksets changed across layers")
        if source_binding != self.source_binding:
            raise ValueError("dirty-state tensor identity/version changed")
        if pool_geometry != self.pool_geometry:
            raise ValueError("dirty-state pool geometry changed across layers")


@dataclass(frozen=True)
class LivePrefixStateContinuationAuthorization:
    """Proof that request-row zero may consume resident prefix state.

    A resolved physical state slot proves only its address.  The serving
    backend issues this separate authorization after the preceding
    microforward completed the full layer-3..39 final rendezvous.  Callers
    without that receipt remain fail-closed.
    """

    request_index: int
    flat_row_offset: int
    seq_len_before: int
    row_count: int
    prior_forward_token: str
    terminal_state_slots: Tuple[Tuple[int, str, int], ...]

    def __post_init__(self) -> None:
        if type(self.request_index) is not int or self.request_index < 0:
            raise ValueError("live-prefix authorization request index is invalid")
        if type(self.flat_row_offset) is not int or self.flat_row_offset < 0:
            raise ValueError("live-prefix authorization flat offset is invalid")
        if type(self.seq_len_before) is not int or self.seq_len_before <= 0:
            raise ValueError("live-prefix authorization needs a positive prefix")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("live-prefix authorization row count is invalid")
        if not _is_sha256_digest(self.prior_forward_token):
            raise ValueError("live-prefix authorization prior receipt is absent")
        if (
            type(self.terminal_state_slots) is not tuple
            or not self.terminal_state_slots
        ):
            raise ValueError("live-prefix authorization terminal slots are absent")
        normalized = []
        for value in self.terminal_state_slots:
            if not isinstance(value, tuple) or len(value) != 3:
                raise TypeError("live-prefix terminal slot binding is malformed")
            layer_id, domain, slot = value
            if type(layer_id) is not int or layer_id < 0:
                raise ValueError("live-prefix terminal slot layer is invalid")
            if domain not in (
                DOMAIN_C4_ATTENTION_STATE,
                DOMAIN_C128_ATTENTION_STATE,
                DOMAIN_INDEXER_STATE,
            ):
                raise ValueError("live-prefix terminal slot domain is invalid")
            if type(slot) is not int or slot < 0:
                raise ValueError("live-prefix terminal physical slot is invalid")
            normalized.append(value)
        if tuple(sorted(set(normalized))) != self.terminal_state_slots:
            raise ValueError("live-prefix terminal slot bindings are not canonical")

    def validate(
        self,
        *,
        workset: DirtyRequestWorkset,
        schedule,
        current_forward_token: str,
        layer_id: int,
        compress_ratio: int,
    ) -> int:
        if (
            int(workset.request_index) != self.request_index
            or int(workset.flat_row_offset) != self.flat_row_offset
            or int(workset.seq_len_before) != self.seq_len_before
            or int(workset.row_count) != self.row_count
        ):
            raise ValueError("live-prefix authorization geometry changed")
        positions = tuple(getattr(schedule, "positions", ()))
        if (
            len(positions) != self.row_count
            or any(type(value) is not int for value in positions)
            or positions[0] != self.seq_len_before
            or any(
                right != left + 1
                for left, right in zip(positions, positions[1:])
            )
        ):
            raise ValueError(
                "live-prefix schedule is not the certified continuation"
            )
        expected_forward = (
            f"{str(current_forward_token)}:request:{self.request_index}"
        )
        if str(getattr(schedule, "forward_id", "")) != expected_forward:
            raise ValueError("live-prefix schedule belongs to another forward")
        required_domains = (
            (DOMAIN_C4_ATTENTION_STATE, DOMAIN_INDEXER_STATE)
            if int(compress_ratio) == 4
            else (DOMAIN_C128_ATTENTION_STATE,)
        )
        bindings = tuple(
            (domain, slot)
            for bound_layer, domain, slot in self.terminal_state_slots
            if int(bound_layer) == int(layer_id)
        )
        if tuple(domain for domain, _ in bindings) != tuple(required_domains):
            raise ValueError("live-prefix terminal state domains changed")
        slots = {int(slot) for _, slot in bindings}
        if len(slots) != 1:
            raise ValueError("live-prefix terminal state domains disagree on slot")
        return next(iter(slots))


def resolve_dirty_state_slots(
    dirty_worksets: Sequence[object],
    *,
    token_to_kv_pool,
    layer_id: int,
    compress_ratio: int,
    full_cache_slots: torch.Tensor,
    req_to_token: Optional[torch.Tensor] = None,
    request_pool_indices: Optional[object] = None,
) -> Tuple[DirtyRequestWorkset, ...]:
    """Resolve logical island boundaries to physical compressor-state groups.

    The backend supplies request geometry, never ring/group arithmetic.  A
    boundary token already present in the packed extension is resolved through
    ``full_cache_slots``.  A boundary in the pre-existing prefix is resolved
    through ``req_to_token[request_pool_indices]``.  Only the one/two terminal
    completion slots are touched, independent of prefix length.
    """

    ratio = int(compress_ratio)
    _expected_domains(ratio)
    required = 2 if ratio == 4 else 1
    full_slots = _long_vector(full_cache_slots, name="forward full-cache slots")
    worksets = canonicalize_dirty_request_worksets(
        tuple(dirty_worksets),
        q_rows=int(full_slots.numel()),
        compress_ratio=ratio,
        state_slots_are_physical=False,
    )
    if req_to_token is not None and (
        not isinstance(req_to_token, torch.Tensor) or req_to_token.ndim != 2
    ):
        raise ValueError("req_to_token must be a rank-2 tensor")
    if request_pool_indices is not None:
        if isinstance(request_pool_indices, torch.Tensor):
            if request_pool_indices.ndim != 1:
                raise ValueError("request_pool_indices must be rank-1")
            request_pool_indices = tuple(
                int(value)
                for value in request_pool_indices.to(device="cpu").tolist()
            )
            pool_count = len(request_pool_indices)
        else:
            request_pool_indices = tuple(
                _metadata_int(value, "request pool index")
                for value in request_pool_indices
            )
            pool_count = len(request_pool_indices)
        if any(value < 0 for value in request_pool_indices):
            raise ValueError("request pool indices must be non-negative")
        if pool_count != len(worksets):
            raise ValueError("request pool indices do not cover dirty worksets")

    attention_pool = token_to_kv_pool.get_attention_compress_states(int(layer_id))
    indexer_pool = (
        token_to_kv_pool.get_indexer_compress_states(int(layer_id))
        if ratio == 4
        else None
    )
    pending_by_request = []
    attention_slot_tensors = []
    indexer_slot_tensors = []
    for workset in worksets:
        pending_islands = []
        for island in workset.islands:
            if island.token_begin == 0:
                pending_islands.append((island, 0))
                continue
            completed_blocks = island.token_begin // ratio
            if completed_blocks < required:
                raise ValueError(
                    f"C{ratio} dirty island has insufficient restart history"
                )
            boundary_tokens = tuple(
                (block + 1) * ratio - 1
                for block in range(
                    completed_blocks - required, completed_blocks
                )
            )
            boundary_full_slots = []
            for token_index in boundary_tokens:
                request_row = token_index - workset.seq_len_before
                if 0 <= request_row < workset.row_count:
                    flat_row = workset.flat_row_offset + request_row
                    boundary_full_slots.append(full_slots[flat_row])
                    continue
                if req_to_token is None or request_pool_indices is None:
                    raise ValueError(
                        "prefix-boundary state resolution requires req_to_token "
                        "and request_pool_indices"
                    )
                pool_index = request_pool_indices[workset.request_index]
                boundary_full_slots.append(
                    req_to_token[pool_index, int(token_index)]
                )
            boundary_full = torch.stack(boundary_full_slots).to(
                device=full_slots.device, dtype=torch.long
            )
            attention_slots = _compute_state_target_slots(
                token_to_kv_pool=token_to_kv_pool,
                full_cache_slots=boundary_full,
                state_pool=attention_pool,
                ratio=ratio,
            )
            attention_slot_tensors.append(attention_slots)
            if indexer_pool is not None:
                indexer_slot_tensors.append(
                    _compute_state_target_slots(
                        token_to_kv_pool=token_to_kv_pool,
                        full_cache_slots=boundary_full,
                        state_pool=indexer_pool,
                        ratio=ratio,
                    )
                )
            pending_islands.append((island, required))
        pending_by_request.append((workset, tuple(pending_islands)))

    if attention_slot_tensors:
        all_attention_slots = torch.cat(attention_slot_tensors)
        if indexer_slot_tensors:
            all_indexer_slots = torch.cat(indexer_slot_tensors)
            if not bool((all_attention_slots == all_indexer_slots).all().item()):
                raise ValueError("attention/Indexer physical restart slots differ")
        physical_values = tuple(
            int(value)
            for value in all_attention_slots.to(device="cpu").tolist()
        )
    else:
        physical_values = ()

    resolved_worksets = []
    cursor = 0
    for workset, pending_islands in pending_by_request:
        islands = []
        for island, slot_count in pending_islands:
            if slot_count:
                physical = physical_values[cursor : cursor + slot_count]
                cursor += slot_count
            else:
                physical = ()
            islands.append(
                DirtyCompressorIsland(
                    island.flat_begin,
                    island.flat_end,
                    island.request_row_begin,
                    island.request_row_end,
                    island.token_begin,
                    island.token_end,
                    physical,
                    island.completion_output_rows,
                )
            )
        resolved_worksets.append(
            DirtyRequestWorkset(
                workset.request_index,
                workset.flat_row_offset,
                workset.row_count,
                workset.seq_len_before,
                tuple(islands),
            )
        )
    if cursor != len(physical_values):
        raise AssertionError("physical state slot accounting changed")
    return canonicalize_dirty_request_worksets(
        tuple(resolved_worksets),
        q_rows=int(full_slots.numel()),
        compress_ratio=ratio,
        state_slots_are_physical=True,
    )


def resolve_dirty_state_slots_certified(
    dirty_worksets: Sequence[object],
    *,
    token_to_kv_pool,
    layer_id: int,
    compress_ratio: int,
    full_cache_slots: torch.Tensor,
    forward_token: str,
    req_to_token: Optional[torch.Tensor] = None,
    request_pool_indices: Optional[object] = None,
    certificate: Optional[DirtyStateSlotResolutionCertificate] = None,
) -> Tuple[
    Tuple[DirtyRequestWorkset, ...], DirtyStateSlotResolutionCertificate
]:
    """Resolve once per ratio, then revalidate immutable inputs without D2H."""

    ratio = int(compress_ratio)
    _expected_domains(ratio)
    full_slots = _long_vector(full_cache_slots, name="forward full-cache slots")
    q_rows = int(full_slots.numel())
    logical = canonicalize_dirty_request_worksets(
        tuple(dirty_worksets),
        q_rows=q_rows,
        compress_ratio=ratio,
        state_slots_are_physical=False,
    )
    source_binding = _dirty_state_slot_source_binding(
        token_to_kv_pool=token_to_kv_pool,
        full_cache_slots=full_cache_slots,
        req_to_token=req_to_token,
        request_pool_indices=request_pool_indices,
    )
    pool_geometry = _dirty_state_pool_geometry(
        token_to_kv_pool=token_to_kv_pool,
        layer_id=int(layer_id),
        compress_ratio=ratio,
    )
    if certificate is not None:
        if not isinstance(certificate, DirtyStateSlotResolutionCertificate):
            raise TypeError("dirty-state slot certificate has a foreign type")
        certificate.validate(
            forward_token=str(forward_token),
            compress_ratio=ratio,
            q_rows=q_rows,
            logical_worksets=logical,
            source_binding=source_binding,
            pool_geometry=pool_geometry,
        )
        return certificate.resolved_worksets, certificate

    resolved = resolve_dirty_state_slots(
        logical,
        token_to_kv_pool=token_to_kv_pool,
        layer_id=int(layer_id),
        compress_ratio=ratio,
        full_cache_slots=full_cache_slots,
        req_to_token=req_to_token,
        request_pool_indices=request_pool_indices,
    )
    created = DirtyStateSlotResolutionCertificate(
        str(forward_token),
        ratio,
        q_rows,
        logical,
        resolved,
        source_binding,
        pool_geometry,
    )
    return resolved, created


def _with_dirty_state_slot_overrides(
    slots: torch.Tensor,
    dirty_worksets: Sequence[DirtyRequestWorkset],
) -> torch.Tensor:
    """Bind checkpoint output rows to each island's terminal state group."""

    bindings = tuple(
        (island.flat_begin, island.state_slot_indices[-1])
        for workset in dirty_worksets
        for island in workset.islands
        if island.state_slot_indices
    )
    if not bindings:
        return slots
    rows = torch.tensor(
        [row for row, _ in bindings], dtype=torch.long, device=slots.device
    )
    values = torch.tensor(
        [slot for _, slot in bindings], dtype=torch.long, device=slots.device
    )
    result = slots.clone()
    result.index_copy_(0, rows, values)
    return result


@dataclass(frozen=True)
class RestoredStateBinding:
    """One restart-state binding for one dirty compressor island."""

    request_index: int
    token_begin: int
    state_slot_indices: Tuple[int, ...]

    @property
    def state_slots(self) -> Tuple[int, ...]:
        return self.state_slot_indices


@dataclass(frozen=True)
class RestoredStateReceipt:
    """Duck-compatible proof consumed by dirty core/Indexer compressors."""

    layer_id: int
    compress_ratio: int
    is_indexer: bool
    bindings: Tuple[RestoredStateBinding, ...]
    restore_token: str
    forward_token: str

    @property
    def ratio(self) -> int:
        return self.compress_ratio

    @property
    def entries(self) -> Tuple[RestoredStateBinding, ...]:
        return self.bindings

    @property
    def state_bindings(self) -> Tuple[RestoredStateBinding, ...]:
        return self.bindings

    @property
    def receipt_token(self) -> str:
        return self.restore_token

    @property
    def schedule_digest(self) -> str:
        return self.restore_token

    @property
    def forward_id(self) -> str:
        return self.forward_token


def _restored_state_receipt(
    *,
    layer_id: int,
    compress_ratio: int,
    dirty_worksets: Sequence[DirtyRequestWorkset],
    is_indexer: bool,
    restore_token: str,
    forward_token: str,
) -> RestoredStateReceipt:
    restore_token = str(restore_token)
    forward_token = str(forward_token)
    if not restore_token:
        raise ValueError("restored compressor state needs a restore token")
    bindings = tuple(
        RestoredStateBinding(
            workset.request_index,
            island.token_begin,
            island.state_slot_indices,
        )
        for workset in dirty_worksets
        for island in workset.islands
    )
    return RestoredStateReceipt(
        int(layer_id),
        int(compress_ratio),
        bool(is_indexer),
        bindings,
        restore_token,
        forward_token,
    )


def _raise_restore_slot_violation(
    keys: Sequence[Tuple[str, int]],
    violations: torch.Tensor,
) -> None:
    """Report the first exact slot-bounds failure after the fast device check."""

    reasons = violations.detach().to(device="cpu").tolist()
    for key, (has_negative, exceeds_capacity) in zip(keys, reasons):
        if bool(has_negative):
            raise ValueError(f"restore slots {key!r} contain a negative value")
        if bool(exceeds_capacity):
            raise ValueError(f"restore slots {key!r} exceed target capacity")
    # The caller reaches this helper only after the aggregated device predicate
    # reported a failure.  Never continue with an unexplained validation result.
    raise ValueError("restore slot bounds validation failed without a domain")


@dataclass(frozen=True)
class RestoreSlotBoundsBatchCertificate:
    """Proof that every registered full slot vector passed one device fence."""

    layer_ids: Tuple[int, ...]
    vector_count: int
    predicate_count: int


class RestoreSlotBoundsBatch:
    """Aggregate full-vector slot bounds without serializing every layer.

    Constructing a :class:`LayerRestoreTargets` normally performs one scalar
    device-to-host predicate read.  A forward-wide composite restore creates
    37 target sets before it can mutate any cache, so those reads needlessly
    serialize otherwise independent vector predicates.  This collector keeps
    the predicates on device and consumes them with one scalar read after all
    reusable layers have completed structural preflight.

    The exceptional path still copies the complete, tiny reason matrix and
    reports the first exact ``(domain, layer)`` failure.  Registration is
    one-shot: no target may be added after the certificate is issued.
    """

    def __init__(self) -> None:
        self._entries = []
        self._keys = set()
        self._device = None
        self._certificate = None

    @property
    def is_validated(self) -> bool:
        return self._certificate is not None

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def register(
        self,
        keys: Sequence[Tuple[str, int]],
        violations: torch.Tensor,
    ) -> None:
        if self._certificate is not None:
            raise RuntimeError(
                "restore slot bounds were registered after batch validation"
            )
        normalized_keys = tuple((str(domain), int(layer)) for domain, layer in keys)
        if not normalized_keys or len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("restore slot bounds keys are empty or duplicated")
        if self._keys.intersection(normalized_keys):
            raise ValueError("restore slot bounds batch repeats a domain/layer")
        if (
            not isinstance(violations, torch.Tensor)
            or violations.dtype != torch.bool
            or violations.ndim != 2
            or tuple(violations.shape) != (len(normalized_keys), 2)
        ):
            raise ValueError(
                "restore slot bounds predicates must be bool[num_vectors,2]"
            )
        device = str(violations.device)
        if self._device is None:
            self._device = device
        elif self._device != device:
            raise ValueError("restore slot bounds predicates use different devices")
        self._keys.update(normalized_keys)
        self._entries.append((normalized_keys, violations))

    def finalize(
        self,
        *,
        expected_layer_ids: Sequence[int],
    ) -> RestoreSlotBoundsBatchCertificate:
        expected_layers = tuple(int(value) for value in expected_layer_ids)
        if (
            not expected_layers
            or tuple(sorted(set(expected_layers))) != expected_layers
        ):
            raise ValueError("restore slot bounds expected layers are not canonical")
        if self._certificate is not None:
            if self._certificate.layer_ids != expected_layers:
                raise ValueError("restore slot bounds certificate covers other layers")
            return self._certificate
        observed_layers = tuple(sorted({layer for _, layer in self._keys}))
        if observed_layers != expected_layers:
            raise ValueError(
                "restore slot bounds batch does not cover the reusable layers"
            )
        if not self._entries:
            raise ValueError("restore slot bounds batch has no predicates")
        keys = tuple(
            key for entry_keys, _ in self._entries for key in entry_keys
        )
        predicates = torch.cat(
            tuple(violations for _, violations in self._entries), dim=0
        )
        # This is the only success-path D2H synchronization for every full
        # slot vector in the forward.  A failure takes one additional small
        # reason-matrix copy so the original exact error remains observable.
        if bool(predicates.any().item()):
            _raise_restore_slot_violation(keys, predicates)
        self._certificate = RestoreSlotBoundsBatchCertificate(
            layer_ids=observed_layers,
            vector_count=len(keys),
            predicate_count=int(predicates.numel()),
        )
        self._entries = []
        return self._certificate


@dataclass(frozen=True)
class LayerRestoreTargets:
    """Per-layer cache tensors, independent slot vectors, and kernel geometry."""

    layer_id: int
    compress_ratio: int
    targets: Mapping[Tuple[str, int], torch.Tensor]
    target_slots: Mapping[Tuple[str, int], torch.Tensor]
    packed_page_sizes: Mapping[Tuple[str, int], int]
    indexer_page_sizes: Mapping[int, int]
    state_group_widths: Mapping[Tuple[str, int], int]
    dirty_worksets: Tuple[DirtyRequestWorkset, ...] = ()
    forward_token: str = ""
    restore_token: str = ""
    dirty_state_slot_certificate: Optional[
        DirtyStateSlotResolutionCertificate
    ] = None
    slot_bounds_batch: Optional[RestoreSlotBoundsBatch] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        expected = {
            (domain, int(self.layer_id))
            for domain in _expected_domains(self.compress_ratio)
        }
        if set(self.targets) != expected or set(self.target_slots) != expected:
            raise ValueError("restore target cache/slot domains are incomplete")
        packed_keys = {
            (DOMAIN_SWA, self.layer_id),
            (self.compressed_domain, self.layer_id),
        }
        state_keys = {
            (
                DOMAIN_C4_ATTENTION_STATE
                if self.compress_ratio == 4
                else DOMAIN_C128_ATTENTION_STATE,
                self.layer_id,
            )
        }
        if self.compress_ratio == 4:
            state_keys.add((DOMAIN_INDEXER_STATE, self.layer_id))
        if set(self.packed_page_sizes) != packed_keys:
            raise ValueError("packed restore page-size domains are incomplete")
        if set(self.state_group_widths) != state_keys:
            raise ValueError("state restore group-width domains are incomplete")
        if set(self.indexer_page_sizes) != (
            {self.layer_id} if self.compress_ratio == 4 else set()
        ):
            raise ValueError("Indexer restore page-size domains are incomplete")
        row_counts = set()
        devices = set()
        slot_bounds = []
        for key in sorted(expected):
            target = self.targets[key]
            slots = self.target_slots[key]
            if not isinstance(target, torch.Tensor) or target.ndim != 2:
                raise TypeError(f"restore target {key!r} is not a tensor")
            if not isinstance(slots, torch.Tensor) or slots.ndim != 1:
                raise ValueError(f"restore slots {key!r} must be rank-1")
            if slots.dtype != torch.long:
                raise ValueError(f"restore slots {key!r} must use torch.long")
            row_counts.add(int(slots.numel()))
            devices.add(str(slots.device))
            if target.device != slots.device:
                raise ValueError(f"restore target/slots {key!r} use different devices")
            domain = key[0]
            if key in self.packed_page_sizes:
                capacity = int(target.shape[0]) * int(
                    self.packed_page_sizes[key]
                )
            elif domain == DOMAIN_INDEXER:
                capacity = int(target.shape[0]) * int(
                    self.indexer_page_sizes[self.layer_id]
                )
            else:
                width = int(self.state_group_widths.get(key, 0))
                if width <= 0 or int(target.shape[0]) % width:
                    raise ValueError(f"state target {key!r} group width changed")
                capacity = int(target.shape[0]) // width
            slot_bounds.append((key, slots, capacity))
        if len(row_counts) != 1 or len(devices) != 1:
            raise ValueError("per-domain restore slot vectors are not forward-aligned")
        # Keep every per-domain reduction on the device, then perform one
        # device-to-host predicate for the entire layer.  The exceptional slow
        # path copies the small reason matrix only when a bound actually fails.
        bound_violations = torch.stack(
            tuple(
                torch.stack(
                    ((slots < 0).any(), (slots >= capacity).any()), dim=0
                )
                for _, slots, capacity in slot_bounds
            ),
            dim=0,
        )
        bound_keys = tuple(key for key, _, _ in slot_bounds)
        if self.slot_bounds_batch is None:
            if bool(bound_violations.any().item()):
                _raise_restore_slot_violation(bound_keys, bound_violations)
        elif isinstance(self.slot_bounds_batch, RestoreSlotBoundsBatch):
            self.slot_bounds_batch.register(bound_keys, bound_violations)
        else:
            raise TypeError("slot_bounds_batch has a foreign type")
        if self.dirty_worksets:
            canonical = canonicalize_dirty_request_worksets(
                self.dirty_worksets,
                q_rows=next(iter(row_counts)),
                compress_ratio=self.compress_ratio,
            )
            if canonical != self.dirty_worksets:
                raise ValueError("dirty request worksets are not canonical")
        if not isinstance(self.forward_token, str) or not isinstance(
            self.restore_token, str
        ):
            raise TypeError("restore/forward tokens must be strings")
        certificate = self.dirty_state_slot_certificate
        if certificate is not None:
            if not isinstance(certificate, DirtyStateSlotResolutionCertificate):
                raise TypeError("restore target dirty-state certificate is invalid")
            if (
                certificate.forward_token != self.forward_token
                or certificate.compress_ratio != self.compress_ratio
                or certificate.q_rows != self.q_rows
                or certificate.resolved_worksets != self.dirty_worksets
            ):
                raise ValueError("restore target dirty-state certificate changed")

    @property
    def q_rows(self) -> int:
        return int(next(iter(self.target_slots.values())).numel())

    @property
    def target_caches(self) -> Mapping[Tuple[str, int], torch.Tensor]:
        return self.targets

    @property
    def compressed_domain(self) -> str:
        return DOMAIN_C4 if self.compress_ratio == 4 else DOMAIN_C128

    @property
    def full_compressed_target_loc(self) -> torch.Tensor:
        """Full flattened-forward loc consumed by dirty-only compressor APIs."""

        return self.target_slots[(self.compressed_domain, self.layer_id)]

    @property
    def full_indexer_target_loc(self) -> Optional[torch.Tensor]:
        if self.compress_ratio != 4:
            return None
        return self.target_slots[(DOMAIN_INDEXER, self.layer_id)]

    def restored_state_receipt(
        self,
        *,
        is_indexer: bool,
        restore_token: Optional[str] = None,
        forward_token: Optional[str] = None,
    ) -> RestoredStateReceipt:
        if is_indexer and self.compress_ratio != 4:
            raise ValueError("only C4 layers have Indexer restored state")
        return _restored_state_receipt(
            layer_id=self.layer_id,
            compress_ratio=self.compress_ratio,
            dirty_worksets=self.dirty_worksets,
            is_indexer=is_indexer,
            restore_token=(
                self.restore_token if restore_token is None else restore_token
            ),
            forward_token=(
                self.forward_token if forward_token is None else forward_token
            ),
        )


def build_layer_restore_targets(
    *,
    token_to_kv_pool,
    layer_id: int,
    compress_ratio: int,
    full_cache_slots: torch.Tensor,
    compressed_slots_by_output_row: torch.Tensor,
    dirty_worksets: Sequence[object] = (),
    forward_token: str = "",
    restore_token: str = "",
    attention_state_slots_by_output_row: Optional[torch.Tensor] = None,
    indexer_state_slots_by_output_row: Optional[torch.Tensor] = None,
    req_to_token: Optional[torch.Tensor] = None,
    request_pool_indices: Optional[object] = None,
    dirty_state_slot_certificate: Optional[
        DirtyStateSlotResolutionCertificate
    ] = None,
    enable_dirty_state_slot_certificate: bool = False,
    slot_bounds_batch: Optional[RestoreSlotBoundsBatch] = None,
) -> LayerRestoreTargets:
    """Bind native caches without running clean-row model/compressor compute.

    ``compressed_slots_by_output_row`` must be aligned to the complete forward
    row vector. Only compressor-completion rows are consumed by the restore
    schedule; values in other rows may be any in-range placeholder selected by
    the backend metadata builder.  Dirty worksets carry logical request/token
    geometry only: any supplied ``state_slot_indices`` are ignored and replaced
    with physical group slots derived here.  ``req_to_token`` plus
    ``request_pool_indices`` are required when an island's restart history lies
    before the packed ``full_cache_slots`` extension.
    """

    layer_id = int(layer_id)
    ratio = int(compress_ratio)
    _expected_domains(ratio)
    full_slots = _long_vector(full_cache_slots, name="forward full-cache slots")
    q_rows = int(full_slots.numel())
    compressed_slots = _long_vector(
        compressed_slots_by_output_row,
        name="compressed slots by output row",
        expected_rows=q_rows,
    ).to(device=full_slots.device)
    if compressed_slots.device != full_slots.device:
        raise ValueError("full and compressed cache slots use different devices")
    raw_dirty_worksets = tuple(dirty_worksets)
    state_slot_certificate = None
    if type(enable_dirty_state_slot_certificate) is not bool:
        raise TypeError("enable_dirty_state_slot_certificate must be boolean")
    if raw_dirty_worksets:
        if (
            enable_dirty_state_slot_certificate
            or dirty_state_slot_certificate is not None
        ):
            canonical_dirty, state_slot_certificate = (
                resolve_dirty_state_slots_certified(
                    raw_dirty_worksets,
                    token_to_kv_pool=token_to_kv_pool,
                    layer_id=layer_id,
                    compress_ratio=ratio,
                    full_cache_slots=full_slots,
                    forward_token=str(forward_token),
                    req_to_token=req_to_token,
                    request_pool_indices=request_pool_indices,
                    certificate=dirty_state_slot_certificate,
                )
            )
        else:
            canonical_dirty = resolve_dirty_state_slots(
                raw_dirty_worksets,
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                compress_ratio=ratio,
                full_cache_slots=full_slots,
                req_to_token=req_to_token,
                request_pool_indices=request_pool_indices,
            )
    else:
        if dirty_state_slot_certificate is not None:
            raise ValueError("dirty-state certificate has no dirty workset")
        canonical_dirty = ()

    key = lambda domain: (domain, layer_id)
    swa_slots = token_to_kv_pool.translate_loc_from_full_to_swa(full_slots).long()
    targets: Dict[Tuple[str, int], torch.Tensor] = {
        key(DOMAIN_SWA): token_to_kv_pool.get_swa_raw_key_buffer_radix(layer_id)
    }
    slots: Dict[Tuple[str, int], torch.Tensor] = {key(DOMAIN_SWA): swa_slots}
    packed_pages: Dict[Tuple[str, int], int] = {
        key(DOMAIN_SWA): int(token_to_kv_pool.swa_kv_pool.page_size)
    }
    indexer_pages: Dict[int, int] = {}
    state_widths: Dict[Tuple[str, int], int] = {}

    compressed_domain = DOMAIN_C4 if ratio == 4 else DOMAIN_C128
    targets[key(compressed_domain)] = token_to_kv_pool.get_extra_key_buffer(layer_id)
    slots[key(compressed_domain)] = compressed_slots
    packed_pages[key(compressed_domain)] = int(
        token_to_kv_pool.get_extra_key_page_size(layer_id)
    )

    attention_state_domain = (
        DOMAIN_C4_ATTENTION_STATE
        if ratio == 4
        else DOMAIN_C128_ATTENTION_STATE
    )
    attention_state_pool = token_to_kv_pool.get_attention_compress_states(layer_id)
    state_widths[key(attention_state_domain)] = _state_group_width(
        attention_state_pool, ratio=ratio
    )
    targets[key(attention_state_domain)] = (
        attention_state_pool.kv_score_buffer.kv_score
    )
    if attention_state_slots_by_output_row is None:
        attention_state_slots = _compute_state_target_slots(
            token_to_kv_pool=token_to_kv_pool,
            full_cache_slots=full_slots,
            state_pool=attention_state_pool,
            ratio=ratio,
        )
    else:
        attention_state_slots = _long_vector(
            attention_state_slots_by_output_row,
            name="attention state slots by output row",
            expected_rows=q_rows,
        ).to(device=full_slots.device)
    slots[key(attention_state_domain)] = _with_dirty_state_slot_overrides(
        attention_state_slots, canonical_dirty
    )

    if ratio == 4:
        targets[key(DOMAIN_INDEXER)] = (
            token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
        )
        slots[key(DOMAIN_INDEXER)] = compressed_slots
        indexer_pages[layer_id] = int(token_to_kv_pool.get_index_k_page_size())

        indexer_state_pool = token_to_kv_pool.get_indexer_compress_states(layer_id)
        state_widths[key(DOMAIN_INDEXER_STATE)] = _state_group_width(
            indexer_state_pool, ratio=ratio
        )
        targets[key(DOMAIN_INDEXER_STATE)] = (
            indexer_state_pool.kv_score_buffer.kv_score
        )
        if indexer_state_slots_by_output_row is None:
            indexer_state_slots = _compute_state_target_slots(
                token_to_kv_pool=token_to_kv_pool,
                full_cache_slots=full_slots,
                state_pool=indexer_state_pool,
                ratio=ratio,
            )
        else:
            indexer_state_slots = _long_vector(
                indexer_state_slots_by_output_row,
                name="Indexer state slots by output row",
                expected_rows=q_rows,
            ).to(device=full_slots.device)
        slots[key(DOMAIN_INDEXER_STATE)] = _with_dirty_state_slot_overrides(
            indexer_state_slots, canonical_dirty
        )
    elif indexer_state_slots_by_output_row is not None:
        raise ValueError("C128 layers cannot bind Indexer state slots")

    return LayerRestoreTargets(
        layer_id=layer_id,
        compress_ratio=ratio,
        targets=MappingProxyType(targets),
        target_slots=MappingProxyType(slots),
        packed_page_sizes=MappingProxyType(packed_pages),
        indexer_page_sizes=MappingProxyType(indexer_pages),
        state_group_widths=MappingProxyType(state_widths),
        dirty_worksets=canonical_dirty,
        forward_token=str(forward_token),
        restore_token=str(restore_token),
        dirty_state_slot_certificate=state_slot_certificate,
        slot_bounds_batch=slot_bounds_batch,
    )


def _apply_rope_from_canonical(
    rope: torch.Tensor,
    destination_positions: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply native DSV4 RoPE once to a position-zero canonical vector."""

    from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton

    work = rope.to(torch.bfloat16).clone().contiguous()
    positions = destination_positions.to(
        device=work.device, dtype=torch.int32
    ).contiguous()
    apply_rotary_emb_triton(
        work,
        freqs_cis.to(work.device),
        positions=positions,
        inverse=False,
    )
    return work.to(rope.dtype)


def _packed_restore_kernel(
    freqs_by_layer: Mapping[int, torch.Tensor],
    page_sizes: Mapping[Tuple[str, int], int],
):
    @torch.no_grad()
    def kernel(
        *,
        domain: str,
        layer_id: int,
        source_bank: torch.Tensor,
        source_indices: torch.Tensor,
        target_cache: torch.Tensor,
        target_slots: torch.Tensor,
        output_rows: torch.Tensor,
        positions: torch.Tensor,
        scratch: torch.Tensor,
        slot_scratch: torch.Tensor,
        position_semantics: str,
    ) -> None:
        key = (str(domain), int(layer_id))
        if position_semantics != PACKED_LATENT_POSITION_SEMANTICS:
            raise ValueError("packed restore received incompatible position semantics")
        page_size = int(page_sizes.get(key, 0))
        freqs_cis = freqs_by_layer.get(int(layer_id))
        if page_size <= 0 or freqs_cis is None:
            raise ValueError("packed restore is missing page/RoPE geometry")
        count = int(source_indices.numel())
        if count == 0:
            return
        if (
            source_bank.dtype != torch.uint8
            or int(source_bank.shape[1]) != PACKED_LATENT_BYTES
            or int(scratch.shape[0]) < count
            or int(scratch.shape[1]) != PACKED_LATENT_BYTES
            or int(slot_scratch.numel()) < count
        ):
            raise ValueError("packed restore workspace/source geometry changed")

        packed = scratch[:count]
        torch.index_select(source_bank, 0, source_indices, out=packed)
        # Materialize destination RoPE before the first target-cache mutation.
        canonical_rope = (
            packed[:, PACKED_NOPE_BYTES:PACKED_NOPE_ROPE_BYTES]
            .contiguous()
            .view(torch.bfloat16)
            .view(count, 64)
        )
        torch.index_select(positions, 0, output_rows, out=slot_scratch[:count])
        relocated_rope = _apply_rope_from_canonical(
            canonical_rope, slot_scratch[:count], freqs_cis
        )
        torch.index_select(target_slots, 0, output_rows, out=slot_scratch[:count])
        destination_slots = slot_scratch[:count]
        write_packed_kv(target_cache, destination_slots, packed, page_size)
        write_rope_bf16(
            target_cache, destination_slots, relocated_rope, page_size
        )

    return kernel


def _indexer_restore_kernel(
    freqs_by_layer: Mapping[int, torch.Tensor],
    page_sizes_by_layer: Mapping[int, int],
):
    @torch.no_grad()
    def kernel(
        *,
        domain: str,
        layer_id: int,
        source_bank: torch.Tensor,
        source_indices: torch.Tensor,
        target_cache: torch.Tensor,
        target_slots: torch.Tensor,
        output_rows: torch.Tensor,
        positions: torch.Tensor,
        scratch: torch.Tensor,
        slot_scratch: torch.Tensor,
        position_semantics: str,
    ) -> None:
        if domain != DOMAIN_INDEXER or position_semantics != INDEXER_POSITION_SEMANTICS:
            raise ValueError("Indexer restore received incompatible semantics")
        page_size = int(page_sizes_by_layer.get(int(layer_id), 0))
        freqs_cis = freqs_by_layer.get(int(layer_id))
        if page_size <= 0 or freqs_cis is None:
            raise ValueError("Indexer restore is missing page/RoPE geometry")
        count = int(source_indices.numel())
        if count == 0:
            return
        if (
            source_bank.dtype != torch.uint8
            or int(source_bank.shape[1]) != INDEXER_POSITIONLESS_RECORD_BYTES
            or int(scratch.shape[0]) < count
            or int(scratch.shape[1]) != INDEXER_POSITIONLESS_RECORD_BYTES
        ):
            raise ValueError("Indexer restore workspace/source geometry changed")
        canonical_bytes = scratch[:count]
        torch.index_select(
            source_bank, 0, source_indices, out=canonical_bytes
        )
        canonical = canonical_bytes.view(torch.bfloat16).view(count, 128)
        torch.index_select(positions, 0, output_rows, out=slot_scratch[:count])
        positioned_rope = _apply_rope_from_canonical(
            canonical[:, 64:].contiguous(),
            slot_scratch[:count],
            freqs_cis,
        )
        pre_hadamard = torch.cat(
            (canonical[:, :64], positioned_rope), dim=-1
        ).float().contiguous()
        post_hadamard = _indexer_rotate_activation(pre_hadamard)
        quantized, scale = _quantize_indexer_activation(
            post_hadamard.contiguous()
        )
        if tuple(quantized.shape) != (count, 128) or int(scale.numel()) != count:
            raise ValueError("native Indexer quantizer returned invalid geometry")
        packed = torch.empty(
            (count, INDEXER_PACKED_RECORD_BYTES),
            dtype=torch.uint8,
            device=post_hadamard.device,
        )
        packed[:, :128] = quantized.contiguous().view(torch.uint8)
        packed[:, 128:] = (
            scale.float().reshape(count, 1).contiguous().view(torch.uint8)
        )
        torch.index_select(target_slots, 0, output_rows, out=slot_scratch[:count])
        _write_indexer_packed(
            target_cache,
            slot_scratch[:count],
            packed,
            page_size,
        )

    return kernel


def _state_restore_kernel():
    @torch.no_grad()
    def kernel(
        *,
        domain: str,
        layer_id: int,
        source_bank: torch.Tensor,
        source_indices: torch.Tensor,
        target_cache: torch.Tensor,
        target_slots: torch.Tensor,
        output_rows: torch.Tensor,
        positions: torch.Tensor,
        scratch: torch.Tensor,
        slot_scratch: torch.Tensor,
        position_semantics: str,
    ) -> None:
        del layer_id, positions
        expected_semantics = (
            INDEXER_STATE_POSITION_SEMANTICS
            if domain == DOMAIN_INDEXER_STATE
            else ATTENTION_STATE_POSITION_SEMANTICS
        )
        if position_semantics != expected_semantics:
            raise ValueError("compressor state restore semantics changed")
        required_groups = _state_required_groups(domain)
        count = int(source_indices.numel())
        if count == 0:
            return
        if (
            source_bank.dtype != torch.uint8
            or target_cache.ndim != 2
            or not target_cache.is_contiguous()
            or int(scratch.shape[0]) < count
            or int(scratch.shape[1]) != int(source_bank.shape[1])
        ):
            raise ValueError("compressor state restore geometry changed")

        physical_row_bytes = int(target_cache.shape[-1]) * int(
            target_cache.element_size()
        )
        source_record_bytes = int(source_bank.shape[1])
        divisor = required_groups * physical_row_bytes
        if divisor <= 0 or source_record_bytes % divisor:
            raise ValueError("compressor state record cannot be grouped")
        group_width = source_record_bytes // divisor
        if domain in (DOMAIN_C4_ATTENTION_STATE, DOMAIN_INDEXER_STATE):
            if group_width != 4:
                raise ValueError("C4 state artifact must contain two four-row groups")
        elif group_width not in (1, 128):
            raise ValueError("C128 state artifact group width is incompatible")

        target_groups = _state_group_view(target_cache, group_width)
        target_group_bytes = _as_uint8_group_rows_without_copy(target_groups)
        bytes_per_group = int(target_group_bytes.shape[1])
        source = scratch[:count]
        torch.index_select(source_bank, 0, source_indices, out=source)
        torch.index_select(target_slots, 0, output_rows, out=slot_scratch[:count])
        terminal_group_slots = slot_scratch[:count]
        # C4 writes terminal-1 then terminal. C128 writes terminal only. Reuse
        # the persistent long scratch and never allocate a per-state index list.
        for group_index in range(required_groups):
            delta = required_groups - 1 - group_index
            if delta:
                terminal_group_slots.sub_(delta)
            begin = group_index * bytes_per_group
            end = begin + bytes_per_group
            target_group_bytes.index_copy_(
                0, terminal_group_slots, source[:, begin:end]
            )
            if delta:
                terminal_group_slots.add_(delta)

    return kernel


def build_restore_kernels(
    *,
    freqs_by_layer: Mapping[int, torch.Tensor],
    packed_page_sizes: Mapping[Tuple[str, int], int],
    indexer_page_sizes_by_layer: Mapping[int, int],
) -> Mapping[str, object]:
    """Build callbacks consumed by ``SharedLatentGPUStore.restore_clean``."""

    packed = _packed_restore_kernel(freqs_by_layer, packed_page_sizes)
    state = _state_restore_kernel()
    return MappingProxyType(
        {
            DOMAIN_SWA: packed,
            DOMAIN_C4: packed,
            DOMAIN_C128: packed,
            DOMAIN_INDEXER: _indexer_restore_kernel(
                freqs_by_layer, indexer_page_sizes_by_layer
            ),
            DOMAIN_C4_ATTENTION_STATE: state,
            DOMAIN_C128_ATTENTION_STATE: state,
            DOMAIN_INDEXER_STATE: state,
        }
    )


SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS = (
    "target_page_size",
    "target_row_bytes",
    "state_group_width",
    "state_required_groups",
    "freqs_cis_ptr",
    "freqs_cis_row_bytes",
    "freqs_cis_rows",
    "target_physical_rows",
)


@dataclass(frozen=True)
class SGLangRestoreBatchMetadata:
    """Immutable model geometry consumed by a pointer-table restore kernel.

    The generic GPU store owns source/target pointers and index vectors.  This
    model-owned suffix records exactly the remaining native layout facts.  A
    packed/Indexer job retains the layer's destination RoPE table; a state job
    retains its physical C4/C128 group geometry and carries no position data.
    """

    domain: str
    layer_id: int
    family: str
    target_page_size: int
    target_row_bytes: int
    target_physical_rows: int
    state_group_width: int
    state_required_groups: int
    freqs_cis: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.family != restore_family_for_domain(self.domain):
            raise ValueError("SGLang restore metadata crossed a launch family")
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError("SGLang restore metadata layer is invalid")
        for name, value in (
            ("target_page_size", self.target_page_size),
            ("target_row_bytes", self.target_row_bytes),
            ("target_physical_rows", self.target_physical_rows),
            ("state_group_width", self.state_group_width),
            ("state_required_groups", self.state_required_groups),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"SGLang restore {name} is invalid")
        if self.target_row_bytes <= 0 or self.target_physical_rows <= 0:
            raise ValueError("SGLang restore target geometry must be positive")
        if self.family in (RESTORE_FAMILY_PACKED, RESTORE_FAMILY_INDEXER):
            if self.target_page_size <= 0:
                raise ValueError("positioned restore requires a target page size")
            if self.state_group_width or self.state_required_groups:
                raise ValueError("positioned restore cannot carry state groups")
            if not isinstance(self.freqs_cis, torch.Tensor):
                raise TypeError("positioned restore requires a RoPE table")
            if (
                self.freqs_cis.ndim != 2
                or int(self.freqs_cis.shape[0]) <= 0
                or int(self.freqs_cis.shape[-1]) != 32
                or not self.freqs_cis.is_contiguous()
                or self.freqs_cis.dtype != torch.complex64
            ):
                raise ValueError(
                    "DSV4 restore RoPE table must be contiguous complex64"
                    "[positions,32]"
                )
            minimum_page_bytes = self.target_page_size * (
                PACKED_LATENT_BYTES
                if self.family == RESTORE_FAMILY_PACKED
                else INDEXER_PACKED_RECORD_BYTES
            )
            if self.target_row_bytes < minimum_page_bytes:
                raise ValueError("positioned restore target page is truncated")
        else:
            if self.target_page_size or self.freqs_cis is not None:
                raise ValueError("opaque state restore cannot carry RoPE/page data")
            if self.state_group_width <= 0 or self.state_required_groups <= 0:
                raise ValueError("opaque state restore needs physical group geometry")
            if self.state_required_groups != _state_required_groups(self.domain):
                raise ValueError("opaque state restore history width changed")
            if self.domain in (
                DOMAIN_C4_ATTENTION_STATE,
                DOMAIN_INDEXER_STATE,
            ):
                if self.state_group_width != 4:
                    raise ValueError("C4 state restore requires four-row groups")
            elif self.state_group_width not in (1, 128):
                raise ValueError("C128 state restore group width is incompatible")
            if self.target_physical_rows % self.state_group_width:
                raise ValueError("opaque state target cannot form physical groups")
            if (
                self.target_physical_rows // self.state_group_width
                < self.state_required_groups
            ):
                raise ValueError("opaque state target has no complete history")

    def batch_identity(self) -> Tuple[object, ...]:
        freq_identity = (
            _restore_tensor_identity(self.freqs_cis, name="batch RoPE table")
            if self.freqs_cis is not None
            else ()
        )
        return (
            self.domain,
            self.layer_id,
            self.family,
            self.target_page_size,
            self.target_row_bytes,
            self.target_physical_rows,
            self.state_group_width,
            self.state_required_groups,
            freq_identity,
        )

    def batch_descriptor_values(self) -> Tuple[int, ...]:
        freqs_ptr = 0
        freqs_row_bytes = 0
        freqs_rows = 0
        if self.freqs_cis is not None:
            freqs_ptr = int(self.freqs_cis.data_ptr())
            freqs_row_bytes = int(self.freqs_cis.stride(0)) * int(
                self.freqs_cis.element_size()
            )
            freqs_rows = int(self.freqs_cis.shape[0])
        return (
            self.target_page_size,
            self.target_row_bytes,
            self.state_group_width,
            self.state_required_groups,
            freqs_ptr,
            freqs_row_bytes,
            freqs_rows,
            self.target_physical_rows,
        )

    def validate_batch_geometry(
        self,
        *,
        domain: str,
        layer_id: int,
        source_bank: torch.Tensor,
        source_indices: torch.Tensor,
        target_cache: torch.Tensor,
        target_slots: torch.Tensor,
        output_rows: torch.Tensor,
        positions: torch.Tensor,
        record_bytes: int,
        position_semantics: str,
    ) -> None:
        if domain != self.domain or int(layer_id) != self.layer_id:
            raise ValueError("SGLang restore metadata belongs to another operation")
        count = int(source_indices.numel())
        if count <= 0 or int(output_rows.numel()) != count:
            raise ValueError("SGLang batch restore index geometry changed")
        if (
            source_bank.dtype != torch.uint8
            or source_bank.ndim != 2
            or int(source_bank.shape[1]) != int(record_bytes)
        ):
            raise ValueError("SGLang batch restore source bank changed")
        if (
            target_cache.ndim != 2
            or not target_cache.is_contiguous()
            or int(target_cache.shape[1]) * int(target_cache.element_size())
            != self.target_row_bytes
            or int(target_cache.shape[0]) != self.target_physical_rows
        ):
            raise ValueError("SGLang batch restore target row geometry changed")
        if (
            target_slots.dtype != torch.long
            or output_rows.dtype != torch.long
            or positions.dtype != torch.long
            or target_slots.device != target_cache.device
            or output_rows.device != target_cache.device
            or positions.device != target_cache.device
            or source_bank.device != target_cache.device
        ):
            raise ValueError("SGLang batch restore tensor device/dtype changed")
        if self.freqs_cis is not None and self.freqs_cis.device != target_cache.device:
            raise ValueError("SGLang batch restore RoPE table is on another device")

        if self.family == RESTORE_FAMILY_PACKED:
            if (
                domain not in (DOMAIN_SWA, DOMAIN_C4, DOMAIN_C128)
                or int(record_bytes) != PACKED_LATENT_BYTES
                or position_semantics != PACKED_LATENT_POSITION_SEMANTICS
                or int(target_cache.element_size()) != 1
            ):
                raise ValueError("packed batch restore semantics changed")
        elif self.family == RESTORE_FAMILY_INDEXER:
            if (
                domain != DOMAIN_INDEXER
                or int(record_bytes) != INDEXER_POSITIONLESS_RECORD_BYTES
                or position_semantics != INDEXER_POSITION_SEMANTICS
                or target_cache.dtype != torch.uint8
            ):
                raise ValueError("Indexer batch restore semantics changed")
        else:
            expected_semantics = (
                INDEXER_STATE_POSITION_SEMANTICS
                if domain == DOMAIN_INDEXER_STATE
                else ATTENTION_STATE_POSITION_SEMANTICS
            )
            if position_semantics != expected_semantics:
                raise ValueError("state batch restore semantics changed")
            expected_record_bytes = (
                self.state_required_groups
                * self.state_group_width
                * self.target_row_bytes
            )
            if int(record_bytes) != expected_record_bytes:
                raise ValueError("state batch restore physical group width changed")


def build_restore_batch_metadata(
    *,
    restore_targets: LayerRestoreTargets,
    freqs_cis: torch.Tensor,
) -> Mapping[Tuple[str, int], SGLangRestoreBatchMetadata]:
    """Bind every layer operation to persistent pointer-table metadata."""

    if not isinstance(restore_targets, LayerRestoreTargets):
        raise TypeError("batch metadata requires validated layer targets")
    if not isinstance(freqs_cis, torch.Tensor):
        raise TypeError("batch metadata requires a RoPE frequency tensor")
    layer_id = int(restore_targets.layer_id)
    result = {}
    for key, target in restore_targets.targets.items():
        domain, target_layer = key
        if int(target_layer) != layer_id:
            raise ValueError("batch metadata target layer changed")
        family = restore_family_for_domain(domain)
        page_size = 0
        group_width = 0
        required_groups = 0
        freqs = None
        if family == RESTORE_FAMILY_PACKED:
            page_size = int(restore_targets.packed_page_sizes[key])
            freqs = freqs_cis
        elif family == RESTORE_FAMILY_INDEXER:
            page_size = int(restore_targets.indexer_page_sizes[layer_id])
            freqs = freqs_cis
        else:
            group_width = int(restore_targets.state_group_widths[key])
            required_groups = _state_required_groups(domain)
        result[key] = SGLangRestoreBatchMetadata(
            domain=domain,
            layer_id=layer_id,
            family=family,
            target_page_size=page_size,
            target_row_bytes=int(target.shape[1]) * int(target.element_size()),
            target_physical_rows=int(target.shape[0]),
            state_group_width=group_width,
            state_required_groups=required_groups,
            freqs_cis=freqs,
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class LayerRestoreAdapter:
    """Ready-to-pass per-layer target and callback bundle."""

    layer_id: int
    compress_ratio: int
    targets: Mapping[Tuple[str, int], torch.Tensor]
    target_slots: Mapping[Tuple[str, int], torch.Tensor]
    kernels: Mapping[str, object]
    batch_metadata: Mapping[Tuple[str, int], SGLangRestoreBatchMetadata]
    restore_targets: LayerRestoreTargets
    live_prefix_state_authorizations: Tuple[
        LivePrefixStateContinuationAuthorization, ...
    ] = ()

    @property
    def dirty_worksets(self) -> Tuple[DirtyRequestWorkset, ...]:
        return self.restore_targets.dirty_worksets

    @property
    def target_caches(self) -> Mapping[Tuple[str, int], torch.Tensor]:
        return self.targets

    @property
    def full_compressed_target_loc(self) -> torch.Tensor:
        return self.restore_targets.full_compressed_target_loc

    @property
    def full_indexer_target_loc(self) -> Optional[torch.Tensor]:
        return self.restore_targets.full_indexer_target_loc

    def _request_worksets(
        self, request_index: Optional[int]
    ) -> Tuple[DirtyRequestWorkset, ...]:
        worksets = tuple(self.dirty_worksets)
        if request_index is None:
            if len(worksets) > 1:
                raise ValueError(
                    "request-scoped state schedule needs an explicit request index"
                )
            return worksets
        index = int(request_index)
        selected = tuple(
            workset
            for workset in worksets
            if int(workset.request_index) == index
        )
        if len(selected) != 1:
            raise ValueError(
                "state schedule has no unique request-scoped dirty workset"
            )
        return selected

    def _validate_state_receipt_schedule(
        self, schedule, *, request_index: Optional[int] = None
    ) -> None:
        worksets = self._request_worksets(request_index)
        certificate = self.restore_targets.dirty_state_slot_certificate
        certified_worksets = (
            {
                int(workset.request_index): workset
                for workset in certificate.resolved_worksets
            }
            if isinstance(certificate, DirtyStateSlotResolutionCertificate)
            else {}
        )
        authorizations = {
            int(item.request_index): item
            for item in self.live_prefix_state_authorizations
        }
        if len(authorizations) != len(self.live_prefix_state_authorizations):
            raise ValueError("live-prefix authorizations contain a duplicate request")
        expected_rows = set()
        for workset in worksets:
            certified_workset = certified_worksets.get(int(workset.request_index))
            authorization = authorizations.get(int(workset.request_index))
            authorized_terminal_slot = None
            if authorization is not None:
                authorized_terminal_slot = authorization.validate(
                    workset=workset,
                    schedule=schedule,
                    current_forward_token=self.restore_targets.forward_token,
                    layer_id=self.layer_id,
                    compress_ratio=self.compress_ratio,
                )
            for island in workset.islands:
                if not island.state_slot_indices:
                    continue
                # A dirty island at request-row zero resumes from compressor
                # state owned by the already materialized request prefix.  Its
                # physical slots were resolved through req_to_token and are
                # bound by DirtyStateSlotResolutionCertificate; they are not
                # an artifact scatter in this microforward.  Every other
                # restart state must still have an exact schedule receipt.
                live_prefix_carry = bool(
                    authorization is not None
                    and certified_workset == workset
                    and workset.seq_len_before > 0
                    and island.request_row_begin == 0
                    and island.flat_begin == workset.flat_row_offset
                    and island.token_begin == workset.seq_len_before
                    and island.state_slot_indices[-1]
                    == authorized_terminal_slot
                )
                if not live_prefix_carry:
                    # Artifact schedules are request-local even though the
                    # aggregate compressor workset is flattened batch-wide.
                    expected_rows.add(island.request_row_begin)
        if not expected_rows:
            return
        state_domains = (
            (DOMAIN_C4_ATTENTION_STATE, DOMAIN_INDEXER_STATE)
            if self.compress_ratio == 4
            else (DOMAIN_C128_ATTENTION_STATE,)
        )
        arena = tuple(schedule.index_arena)
        operations = schedule.operations_for_layer(self.layer_id)
        for domain in state_domains:
            restored_rows = set()
            for operation in operations:
                if operation.domain != domain:
                    continue
                restored_rows.update(
                    int(value)
                    for value in arena[
                        operation.output_rows.begin : operation.output_rows.end
                    ]
                )
            if not expected_rows.issubset(restored_rows):
                missing = tuple(sorted(expected_rows - restored_rows))
                raise ValueError(
                    f"{domain} restore schedule omits dirty island rows {missing}"
                )

    def preflight(
        self,
        gpu_store,
        prepared,
        *,
        positions: torch.Tensor,
        request_index: Optional[int] = None,
        target_slots: Optional[Mapping[Tuple[str, int], torch.Tensor]] = None,
    ):
        """Pin-check and validate only this layer's targets before mutation."""

        validate_open = getattr(getattr(prepared, "pin", None), "validate_open", None)
        if not callable(validate_open):
            raise TypeError("prepared restore has no open epoch pin")
        validate_open()
        self._validate_state_receipt_schedule(
            prepared.schedule, request_index=request_index
        )
        scoped_target_slots = (
            self.target_slots if target_slots is None else target_slots
        )
        return gpu_store.preflight_targets(
            prepared,
            targets=self.targets,
            target_slots=scoped_target_slots,
            kernels=self.kernels,
            positions=positions,
            layer_id=self.layer_id,
        )

    def preflight_batch_input(
        self,
        gpu_store,
        prepared,
        *,
        positions: torch.Tensor,
        request_index: int = -1,
    ):
        """Return a mutation-free contribution for a forward-wide restore."""

        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            DeviceRestoreBatchInput,
        )

        validated = self.preflight(
            gpu_store,
            prepared,
            positions=positions,
            request_index=(None if int(request_index) < 0 else int(request_index)),
        )
        return DeviceRestoreBatchInput(
            store=gpu_store,
            validated=validated,
            operation_metadata=self.batch_metadata,
            request_index=int(request_index),
            layer_id=self.layer_id,
        )

    def restore_layer(self, gpu_store, prepared, *, positions: torch.Tensor):
        """Restore one scheduled layer once and return dirty-builder receipts."""

        validated = self.preflight(gpu_store, prepared, positions=positions)
        device_receipt = gpu_store.restore_clean(validated)
        schedule = prepared.schedule
        return self.bind_batch_receipt(
            device_receipt=device_receipt,
            schedule=schedule,
        )

    def bind_batch_receipt(
        self,
        *,
        device_receipt,
        schedule,
        request_index: Optional[int] = None,
    ):
        """Convert one batch input receipt into the existing layer contract."""

        self._validate_state_receipt_schedule(
            schedule, request_index=request_index
        )
        expected = tuple(
            operation
            for operation in schedule.operations_for_layer(self.layer_id)
            if operation.count
        )
        expected_by_domain = {
            operation.domain: int(operation.count) for operation in expected
        }
        if (
            str(getattr(device_receipt, "forward_id", ""))
            != str(schedule.forward_id)
            or str(getattr(device_receipt, "schedule_digest", ""))
            != str(schedule.digest)
            or int(getattr(device_receipt, "operation_count", -1))
            != len(expected)
            or dict(getattr(device_receipt, "restored_by_domain", {}))
            != expected_by_domain
        ):
            raise ValueError("batch restore receipt differs from this layer schedule")
        restore_token = self.restore_targets.restore_token or str(schedule.digest)
        forward_token = self.restore_targets.forward_token or str(schedule.forward_id)
        attention_state = self.restore_targets.restored_state_receipt(
            is_indexer=False,
            restore_token=restore_token,
            forward_token=forward_token,
        )
        indexer_state = (
            self.restore_targets.restored_state_receipt(
                is_indexer=True,
                restore_token=restore_token,
                forward_token=forward_token,
            )
            if self.compress_ratio == 4
            else None
        )
        return LayerRestoreExecutionReceipt(
            self.layer_id,
            self.compress_ratio,
            device_receipt,
            self.dirty_worksets,
            self.full_compressed_target_loc,
            self.full_indexer_target_loc,
            attention_state,
            indexer_state,
            restore_token,
            forward_token,
        )


@dataclass(frozen=True)
class LayerRestoreExecutionReceipt:
    """Clean restore proof plus exact metadata for subsequent dirty compute."""

    layer_id: int
    compress_ratio: int
    device_receipt: object
    dirty_worksets: Tuple[DirtyRequestWorkset, ...]
    full_compressed_target_loc: torch.Tensor
    full_indexer_target_loc: Optional[torch.Tensor]
    attention_state: RestoredStateReceipt
    indexer_state: Optional[RestoredStateReceipt]
    restore_token: str
    forward_token: str

    @property
    def core_restored_state(self) -> RestoredStateReceipt:
        return self.attention_state

    @property
    def indexer_restored_state(self) -> Optional[RestoredStateReceipt]:
        return self.indexer_state


def build_layer_restore_adapter(
    *,
    token_to_kv_pool,
    layer_id: int,
    compress_ratio: int,
    full_cache_slots: torch.Tensor,
    compressed_slots_by_output_row: torch.Tensor,
    freqs_cis: torch.Tensor,
    dirty_worksets: Sequence[object] = (),
    forward_token: str = "",
    restore_token: str = "",
    attention_state_slots_by_output_row: Optional[torch.Tensor] = None,
    indexer_state_slots_by_output_row: Optional[torch.Tensor] = None,
    req_to_token: Optional[torch.Tensor] = None,
    request_pool_indices: Optional[object] = None,
    dirty_state_slot_certificate: Optional[
        DirtyStateSlotResolutionCertificate
    ] = None,
    enable_dirty_state_slot_certificate: bool = False,
    live_prefix_state_authorizations: Sequence[
        LivePrefixStateContinuationAuthorization
    ] = (),
    kernel_callbacks: Optional[Mapping[str, object]] = None,
    slot_bounds_batch: Optional[RestoreSlotBoundsBatch] = None,
) -> LayerRestoreAdapter:
    """Return complete per-layer arguments for GPU-store target preflight."""

    targets = build_layer_restore_targets(
        token_to_kv_pool=token_to_kv_pool,
        layer_id=int(layer_id),
        compress_ratio=int(compress_ratio),
        full_cache_slots=full_cache_slots,
        compressed_slots_by_output_row=compressed_slots_by_output_row,
        dirty_worksets=dirty_worksets,
        forward_token=forward_token,
        restore_token=restore_token,
        attention_state_slots_by_output_row=(
            attention_state_slots_by_output_row
        ),
        indexer_state_slots_by_output_row=indexer_state_slots_by_output_row,
        req_to_token=req_to_token,
        request_pool_indices=request_pool_indices,
        dirty_state_slot_certificate=dirty_state_slot_certificate,
        enable_dirty_state_slot_certificate=enable_dirty_state_slot_certificate,
        slot_bounds_batch=slot_bounds_batch,
    )
    if not isinstance(freqs_cis, torch.Tensor):
        raise TypeError("shared-latent restore requires a RoPE frequency tensor")
    default_kernels = build_restore_kernels(
        freqs_by_layer={int(layer_id): freqs_cis},
        packed_page_sizes=targets.packed_page_sizes,
        indexer_page_sizes_by_layer=targets.indexer_page_sizes,
    )
    kernels = dict(default_kernels)
    if kernel_callbacks is not None:
        if not isinstance(kernel_callbacks, Mapping):
            raise TypeError("kernel_callbacks must be a mapping")
        unknown = set(kernel_callbacks) - set(_expected_domains(int(compress_ratio)))
        if unknown:
            raise ValueError(f"restore kernel callbacks contain unknown domains {unknown}")
        if any(not callable(callback) for callback in kernel_callbacks.values()):
            raise TypeError("restore kernel callbacks must be callable")
        kernels.update(kernel_callbacks)
    batch_metadata = build_restore_batch_metadata(
        restore_targets=targets,
        freqs_cis=freqs_cis,
    )
    authorizations = tuple(live_prefix_state_authorizations)
    if any(
        not isinstance(item, LivePrefixStateContinuationAuthorization)
        for item in authorizations
    ):
        raise TypeError("live-prefix state authorizations have a foreign type")
    if len({item.request_index for item in authorizations}) != len(authorizations):
        raise ValueError("live-prefix state authorizations contain duplicates")
    return LayerRestoreAdapter(
        int(layer_id),
        int(compress_ratio),
        targets.targets,
        targets.target_slots,
        MappingProxyType(kernels),
        batch_metadata,
        targets,
        authorizations,
    )


__all__ = [
    "ATTENTION_STATE_POSITION_SEMANTICS",
    "CHECKPOINT_STRIDE_TOKENS",
    "INDEXER_PACKED_RECORD_BYTES",
    "INDEXER_POSITIONLESS_RECORD_BYTES",
    "INDEXER_STATE_POSITION_SEMANTICS",
    "DirtyCompressorIsland",
    "DirtyRequestWorkset",
    "DirtyStateSlotResolutionCertificate",
    "LivePrefixStateContinuationAuthorization",
    "LayerCaptureBundle",
    "LayerCaptureChunkReceipt",
    "LayerCaptureComponents",
    "LayerCaptureStaging",
    "LayerRestoreAdapter",
    "LayerRestoreExecutionReceipt",
    "LayerRestoreTargets",
    "RestoredStateBinding",
    "RestoredStateReceipt",
    "RestoreSlotBoundsBatch",
    "RestoreSlotBoundsBatchCertificate",
    "SGLANG_RESTORE_BATCH_DESCRIPTOR_COLUMNS",
    "SGLangRestoreBatchMetadata",
    "begin_layer_capture_bundle",
    "build_layer_restore_adapter",
    "build_layer_restore_targets",
    "build_restore_batch_metadata",
    "build_restore_kernels",
    "build_runtime_shared_latent_spec",
    "canonicalize_dirty_request_worksets",
    "canonicalize_indexer_key",
    "canonicalize_packed_latent",
    "capture_chunk_components",
    "capture_compressed_components",
    "capture_layer_bundle",
    "capture_layer_components",
    "capture_state_components",
    "finalize_layer_capture_bundle",
    "make_layer_capture_bundle",
    "resolve_dirty_state_slots",
    "resolve_dirty_state_slots_certified",
]
