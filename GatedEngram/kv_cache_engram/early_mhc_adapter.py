"""Per-layer masked sublayers with the *original full-row* decoder/mHC path.

Inactive attention/FFN outputs are zero, not stale cached hidden outputs. The
native decoder still runs both full-row norm/mHC calls; its next pre consumes
the preceding lazy post exactly once. This differs from interval identity carry.
No Torch, native module, model construction or CUDA work occurs on import.
"""

from __future__ import annotations

from functools import wraps
from types import MethodType

SPARSE_LAYERS = frozenset((6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18))
SOURCE_LAYERS = frozenset((2, 8, 14))
_MISSING = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _binding(metadata):
    marker = getattr(metadata, "canonical_request_binding", None)
    require(isinstance(marker, dict), "canonical request binding required")
    values = tuple(marker.get(k) for k in ("request_id", "phase", "start", "end"))
    rid, phase, start, end = values
    require(
        isinstance(rid, str)
        and rid
        and rid != "placeholder"
        and isinstance(phase, str)
        and phase
        and type(start) is int
        and type(end) is int
        and 0 <= start < end
        and metadata.is_prompt is True
        and metadata.num_reqs == 1
        and metadata.total_q == end - start,
        "invalid full prompt binding",
    )
    return values


def gather_packed_token_rows(hidden, ids, *, torch_module=None):
    """Copy data and packed SF token rows, preserving native padded SF ABI.

    SF has logical [tokens, packed_channels], token stride 1, channel stride
    roundup(selected_tokens, 4) int32 elements (16-byte column alignment).
    Padding is initialized to zero and no output aliases either input. This
    tensor-only helper is CPU testable; native dtype/device qualification lives
    in TorchMaskedOps.validate_full, not in a simulated CPU GEMM claim.
    """
    if torch_module is None:
        import torch as torch_module
    t = torch_module
    require(isinstance(hidden, tuple) and len(hidden) == 2, "packed data/SF tuple required")
    data, sf = hidden
    require(
        isinstance(data, t.Tensor)
        and isinstance(sf, t.Tensor)
        and data.ndim == sf.ndim == 2
        and sf.shape[0] == data.shape[0]
        and sf.dtype == t.int32
        and sf.device == data.device
        and sf.shape[1] > 0
        and not data.requires_grad
        and not sf.requires_grad,
        "packed data/SF input geometry/dtype invalid",
    )
    require(
        isinstance(ids, t.Tensor) and ids.ndim == 1 and ids.dtype == t.int64 and ids.device == data.device and ids.numel() > 0,
        "nonempty same-device token IDs required",
    )
    n = ids.numel()
    copied_data = data.index_select(0, ids)
    storage = t.zeros((sf.shape[1], (n + 3) // 4 * 4), dtype=t.int32, device=sf.device)
    copied_sf = storage[:, :n].transpose(0, 1)
    copied_sf.copy_(sf.index_select(0, ids))
    require(
        copied_data.dtype == data.dtype
        and copied_data.data_ptr() != data.data_ptr()
        and copied_sf.data_ptr() != sf.data_ptr()
        and copied_sf.stride(0) == 1
        and copied_sf.stride(1) == (n + 3) // 4 * 4,
        "packed gather ownership/layout invariant failed",
    )
    return copied_data, copied_sf


class TorchMaskedOps:
    """Native SM90 row gather, including packed FP8 scale-factor token rows."""

    native_execution = True

    def __init__(self):
        import torch

        self.torch = torch

    def check_runtime(self):
        from deepinfer import PLATFORM
        from deepinfer.model_executor.parallel_utils.parallel_state import PG

        require(
            PLATFORM.is_sm90 and PG.context_parallel_group().size() == 1 and not self.torch.cuda.is_current_stream_capturing(),
            "masked mHC requires eager SM90 CP1",
        )

    @staticmethod
    def unwrap(value):
        return value.move() if callable(getattr(value, "move", None)) else value

    @staticmethod
    def rows(value):
        return (value[0] if isinstance(value, tuple) else value).shape[0]

    def validate_full(self, norm, bf16, rows):
        t = self.torch
        x = norm[0] if isinstance(norm, tuple) else norm
        require(
            isinstance(bf16, t.Tensor)
            and bf16.ndim == 2
            and bf16.shape[0] == rows
            and bf16.is_cuda
            and bf16.dtype == t.bfloat16
            and not bf16.requires_grad,
            "full normalized BF16 rows required",
        )
        require(
            isinstance(x, t.Tensor) and x.shape == bf16.shape and x.device == bf16.device and not x.requires_grad,
            "full norm row geometry mismatch",
        )
        if isinstance(norm, tuple):
            require(
                len(norm) == 2
                and x.dtype == t.float8_e4m3fn
                and isinstance(norm[1], t.Tensor)
                and norm[1].dtype == t.int32
                and norm[1].shape == (rows, x.shape[1] // 128)
                and norm[1].device == x.device
                and not norm[1].requires_grad,
                "packed FP8 SF must have token-major logical rows",
            )
        else:
            require(x.dtype == t.bfloat16, "unsupported native norm representation")

    def validate_plan(self, plan, reference):
        t, ids = self.torch, plan.selected_rows
        require(
            type(plan.full_rows) is int
            and self.rows(reference) == plan.full_rows
            and type(plan.selected_count) is int
            and 1 <= plan.selected_count <= plan.full_rows
            and isinstance(ids, t.Tensor)
            and ids.shape == (plan.selected_count,)
            and ids.device == reference.device
            and ids.dtype == t.int64
            and ids.is_contiguous()
            and not ids.requires_grad,
            "invalid selected token device plan",
        )
        t._assert_async(((ids >= 0) & (ids < plan.full_rows)).all(), "selected token outside current chunk")
        if plan.selected_count > 1:
            t._assert_async((ids[1:] > ids[:-1]).all(), "selected token rows must be sorted and unique")

    def gather(self, value, ids):
        # Preserve the native column-major packed SF ABI for *both* Q GEMM
        # and the MoE dispatcher, including non-multiple-of-four active counts.
        return gather_packed_token_rows(value, ids, torch_module=self.torch) if isinstance(value, tuple) else value.index_select(0, ids)

    def scatter_zero(self, selected, reference, plan):
        require(
            selected.ndim == 2
            and selected.shape == (plan.selected_count, reference.shape[1])
            and selected.dtype == reference.dtype
            and selected.device == reference.device
            and not selected.requires_grad,
            "selected sublayer output geometry/dtype changed",
        )
        result = reference.new_zeros((plan.full_rows, reference.shape[1]))
        result.index_copy_(0, plan.selected_rows, selected)
        return result


class EarlyMHCAdapter:
    """Owned attention/MLP wrappers; never replace decoder or mHC methods.

    get_plan(attn, full_metadata) returns a DevicePlan or None. The externally
    pinned attention_forward/make_active_metadata callbacks implement full-KV
    sparse attention. No file is imported dynamically here. The caller owns
    source-code authentication, early sensitivity and offline-KV authentication.
    """

    def __init__(self, layers, get_plan, attention_forward, make_active_metadata, offline_provider, publish_native_source=None, ops=None):
        self.layers = tuple(layers)
        require(self.layers and len({x.layer_idx for x in self.layers}) == len(self.layers), "distinct decoder layers required")
        require(
            len({id(x.self_attn) for x in self.layers}) == len(self.layers) and len({id(x.mlp) for x in self.layers}) == len(self.layers),
            "sublayer instances cannot be shared between decoders",
        )
        require(all(callable(x) for x in (get_plan, attention_forward, make_active_metadata)), "masked adapter callbacks required")
        require(publish_native_source is None or callable(publish_native_source), "source publisher must be callable")
        self.get_plan, self.attention_forward = get_plan, attention_forward
        self.make_active_metadata, self.offline_provider = make_active_metadata, offline_provider
        self.publish_native_source, self.ops = publish_native_source, ops
        self.handles, self.pending, self.events = [], None, []
        self.closed, self.poisoned, self.installed = False, False, False
        self.counts = dict(
            native_attention_calls=0,
            native_mlp_calls=0,
            masked_layer_calls=0,
            full_rows=0,
            selected_rows=0,
            inactive_attention_zero_rows=0,
            inactive_ffn_zero_rows=0,
            source_publish_calls=0,
        )

    def _check(self):
        require(self.installed and not self.closed and not self.poisoned, "adapter not installed, closed or poisoned")

    def _patch(self, obj, name, function):
        old = obj.__dict__.get(name, _MISSING)
        bound = MethodType(function, obj)
        self.handles.append((obj, name, old, bound))
        setattr(obj, name, bound)

    def _validate_layer(self, layer, runtime):
        require(layer.layer_idx in SPARSE_LAYERS, "first six, sources, boundary and YOCO layers remain native")
        require(
            getattr(layer, "hc_enabled", False)
            and getattr(layer, "hc_mult", None) == 4
            and type(getattr(layer, "hc_shifted", None)) is bool,
            "native four-stream mHC required",
        )
        require(
            not getattr(layer, "use_mega_mhc", False) and not getattr(layer.mlp, "use_mega_moe", False),
            "mega mHC/MoE row compaction is unqualified",
        )
        require(not getattr(type(layer.mlp), "_staged_moe_hook_installed", False), "cannot stack token MoE/output-cache hooks")
        require(
            getattr(layer, "engram", None) is None
            and getattr(layer, "dflash_input_index", None) is None
            and not getattr(layer, "yoco_captures_shared_hidden", False),
            "special decoder remains native",
        )
        require(
            runtime is not None
            and getattr(runtime, "consumer_full_metadata", None) is None
            and getattr(runtime, "consumer_full_hidden", None) is None,
            "ordinary full-layout encoder required",
        )

    def install(self):
        require(not self.installed and not self.closed, "adapter cannot be reinstalled")
        for layer in self.layers:
            require(getattr(layer, "_sparse_interval_owner", None) is None, "cannot stack old interval carrier")
            for obj in (layer.self_attn, layer.mlp):
                require(getattr(obj, "_early_mhc_owner", None) is None, "another masked adapter owns sublayer")
            require(
                getattr(layer.self_attn, "_kv_cache_engram_native_hook", None) is None
                and getattr(layer.self_attn, "_swa_reuse_owner", None) is None,
                "conflicting SWA hook",
            )
        self.installed = True
        try:
            for layer in self.layers:
                self._install_layer(layer)
        except BaseException:
            self.poisoned = True
            self.close(allow_incomplete=True)
            raise
        return self

    def _install_layer(self, layer):
        owner, original_attn, original_mlp = self, layer.self_attn.forward, layer.mlp.forward

        @wraps(original_attn)
        def attention(attn, hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime=None, **kwargs):
            owner._check()
            try:
                require(owner.pending is None, "previous masked attention has no matching FFN")
                plan = owner.get_plan(attn, input_metadata) if input_metadata.is_prompt else None
                if plan is None:
                    result = original_attn(
                        hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime=cross_layer_runtime, **kwargs
                    )
                    owner.counts["native_attention_calls"] += 1
                    if input_metadata.is_prompt and attn.layer_idx in SOURCE_LAYERS and owner.publish_native_source:
                        owner.publish_native_source(attn, input_metadata, cross_layer_runtime)
                        owner.counts["source_publish_calls"] += 1
                    return result
                require(not kwargs, "unknown masked attention kwargs")
                owner._validate_layer(layer, cross_layer_runtime)
                binding = _binding(input_metadata)
                require(
                    binding == (plan.request_id, plan.phase, plan.chunk_start, plan.chunk_end) and plan.full_rows == input_metadata.total_q,
                    "foreign masked row plan",
                )
                if owner.ops is None:
                    owner.ops = TorchMaskedOps()
                ops = owner.ops
                ops.check_runtime()
                norm, bf16 = ops.unwrap(hidden_states), ops.unwrap(hidden_states_bf16)
                ops.validate_full(norm, bf16, plan.full_rows)
                ops.validate_plan(plan, bf16)
                active_meta = owner.make_active_metadata(attn, input_metadata, plan, kv_cache)
                require(
                    active_meta is not input_metadata and active_meta.total_q == plan.selected_count, "fresh selected metadata required"
                )
                result = owner.attention_forward(
                    attn,
                    ops.gather(norm, plan.selected_rows),
                    ops.gather(bf16, plan.selected_rows),
                    input_metadata,
                    active_meta,
                    plan,
                    cross_layer_runtime,
                    kv_cache,
                    owner.offline_provider,
                )
                require(
                    result.plan is plan and result.metadata is active_meta and result.full_rows == plan.full_rows,
                    "sparse attention changed plan or metadata",
                )
                output = ops.scatter_zero(result.output, bf16, plan)
                owner.pending = dict(layer=layer, metadata=input_metadata, active_metadata=active_meta, plan=plan, binding=binding)
                owner.counts["inactive_attention_zero_rows"] += plan.full_rows - plan.selected_count
                return output
            except BaseException:
                owner.poisoned = True
                raise

        @wraps(original_mlp)
        def mlp(module, hidden_states, hidden_states_bf16, input_metadata):
            owner._check()
            scope = owner.pending
            if scope is None:
                try:
                    result = original_mlp(hidden_states, hidden_states_bf16, input_metadata)
                    owner.counts["native_mlp_calls"] += 1
                    return result
                except BaseException:
                    owner.poisoned = True
                    raise
            try:
                require(
                    scope["layer"] is layer and input_metadata is scope["metadata"] and _binding(input_metadata) == scope["binding"],
                    "FFN does not match preceding attention",
                )
                ops, plan = owner.ops, scope["plan"]
                ops.validate_full(hidden_states, hidden_states_bf16, plan.full_rows)
                ops.validate_plan(plan, hidden_states_bf16)
                selected = original_mlp(
                    ops.gather(hidden_states, plan.selected_rows),
                    ops.gather(hidden_states_bf16, plan.selected_rows),
                    scope["active_metadata"],
                )
                output = ops.scatter_zero(selected, hidden_states_bf16, plan)
                owner.events.append(
                    dict(
                        layer=layer.layer_idx,
                        request_id=plan.request_id,
                        phase=plan.phase,
                        chunk_start=plan.chunk_start,
                        chunk_end=plan.chunk_end,
                        full_rows=plan.full_rows,
                        selected_rows=plan.selected_count,
                        plan_sha256=plan.plan_sha256,
                        inactive_sublayer_output="zero",
                        full_row_native_mhc=True,
                        cached_hidden_or_sublayer_output=False,
                    )
                )
                owner.counts["masked_layer_calls"] += 1
                owner.counts["full_rows"] += plan.full_rows
                owner.counts["selected_rows"] += plan.selected_count
                owner.counts["inactive_ffn_zero_rows"] += plan.full_rows - plan.selected_count
                owner.pending = None
                return output
            except BaseException:
                owner.poisoned = True
                raise

        self._patch(layer.self_attn, "forward", attention)
        self._patch(layer.mlp, "forward", mlp)
        layer.self_attn._early_mhc_owner = self
        layer.mlp._early_mhc_owner = self

    def report(self):
        require(self.installed and not self.poisoned and self.pending is None, "incomplete or failed masked layer")
        return dict(
            schema="kv-cache-engram-early-mhc-adapter-v1",
            counts=dict(self.counts),
            events=list(self.events),
            native_execution=None if self.ops is None else self.ops.native_execution,
            decoder_forward_unchanged=True,
            both_full_row_native_mhc_norm_calls_retained=True,
            per_layer_zero_inactive_attention_and_ffn=True,
            interval_frozen_carrier=False,
            inactive_hidden_equals_native=False,
            accuracy_guarantee=False,
            speedup_claim=False,
        )

    def close(self, *, allow_incomplete=False):
        if self.closed:
            return
        require(allow_incomplete or (self.pending is None and not self.poisoned), "failed/incomplete adapter needs explicit cleanup")
        for obj, name, old, bound in reversed(self.handles):
            require(obj.__dict__.get(name, _MISSING) is bound, "adapter patch ownership changed")
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        for layer in self.layers:
            for obj in (layer.self_attn, layer.mlp):
                if getattr(obj, "_early_mhc_owner", None) is self:
                    delattr(obj, "_early_mhc_owner")
        self.handles.clear()
        self.closed = True
