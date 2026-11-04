"""Isolated FFN row selection; attention, native YOCO and decoder/mHC stay native.

Plans are indexed into *the current native row layout*, including compact YOCO
rows. This adapter neither chooses tokens nor computes a sensitivity score. It
does not skip a distributed MoE call when the local selection is empty: that
would omit EP collectives. Empty selection needs a caller-supplied qualification
callback and still calls the original MoE with zero local input rows.

Importing this module does not import Torch or initialize a native/GPU runtime.
"""

from __future__ import annotations

from copy import copy
from functools import wraps
from types import MethodType

from .early_mhc_adapter import TorchMaskedOps

_MISSING = object()
_ROW_FIELDS = ("positions", "valid_token_mask", "gate_force_random", "images_seq_mask")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _binding(metadata):
    marker = getattr(metadata, "canonical_request_binding", None)
    require(isinstance(marker, dict), "canonical request binding required")
    rid, phase, start, end = (marker.get(k) for k in ("request_id", "phase", "start", "end"))
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
        and type(metadata.total_q) is int
        and metadata.total_q >= 0,
        "invalid FFN prompt binding",
    )
    # total_q can be less than end-start in native YOCO compact metadata.
    return rid, phase, start, end


def _snapshot(metadata):
    # Inference-mode tensors deliberately have no version counter. Accessing
    # Tensor._version raises there. Save these small metadata vectors on their
    # existing device instead; the later comparison uses an asynchronous assert.
    return (
        _binding(metadata),
        metadata.total_q,
        tuple((name, getattr(metadata, name, None), _copy_value(getattr(metadata, name, None))) for name in _ROW_FIELDS),
    )


def _copy_value(value):
    return value.clone() if callable(getattr(value, "clone", None)) else copy(value)


def _unchanged(metadata, snapshot, ops):
    binding, rows, fields = snapshot
    require(_binding(metadata) == binding and metadata.total_q == rows, "metadata provider changed native request binding/row count")
    for name, value, saved in fields:
        current = getattr(metadata, name, None)
        require(current is value, "metadata provider mutated native token metadata: " + name)
        if callable(getattr(value, "clone", None)):
            ops.assert_metadata_unchanged(value, saved, name)
        else:
            require(value == saved, "metadata provider mutated native token metadata: " + name)


class TorchFFNOnlyOps(TorchMaskedOps):
    """Native CP1 row operations, permitting explicitly qualified empty calls."""

    def validate_plan(self, plan, reference):
        if plan.selected_count:
            super().validate_plan(plan, reference)
            return
        t, ids = self.torch, plan.selected_rows
        require(
            type(plan.full_rows) is int
            and self.rows(reference) == plan.full_rows
            and type(plan.selected_count) is int
            and plan.selected_count == 0
            and isinstance(ids, t.Tensor)
            and ids.shape == (0,)
            and ids.dtype == t.int64
            and ids.device == reference.device
            and ids.is_contiguous()
            and not ids.requires_grad,
            "invalid empty selected token device plan",
        )

    def gather(self, value, ids):
        if not isinstance(value, tuple) or ids.numel():
            return super().gather(value, ids)
        # Native zero-local-row EP dispatch must still execute. The packed SF
        # view retains a real aligned column stride, although its logical T=0.
        t = self.torch
        data, sf = value
        storage = t.zeros((sf.shape[1], 4), dtype=sf.dtype, device=sf.device)
        return data.index_select(0, ids), storage[:, :0].transpose(0, 1)

    def validate_positions(self, positions, reference, rows):
        t = self.torch
        require(
            isinstance(positions, t.Tensor)
            and positions.shape == (rows,)
            and positions.device == reference.device
            and positions.dtype in (t.int32, t.int64)
            and not positions.requires_grad,
            "explicit current native position vector required",
        )

    def validate_selected_metadata(self, original, selected, plan):
        t = self.torch
        for name in _ROW_FIELDS:
            full, active = getattr(original, name, None), getattr(selected, name, None)
            if full is None:
                require(active is None, "selected metadata invented " + name)
                continue
            require(
                isinstance(full, t.Tensor) and full.ndim == 1 and full.shape[0] == plan.full_rows,
                "native token metadata has wrong layout: " + name,
            )
            require(
                isinstance(active, t.Tensor)
                and active.shape == (plan.selected_count,)
                and active.dtype == full.dtype
                and active.device == full.device
                and not active.requires_grad,
                "selected token metadata has wrong layout: " + name,
            )
            t._assert_async(
                (active == full.index_select(0, plan.selected_rows)).all(), "selected token metadata does not follow FFN row IDs: " + name
            )

    def assert_metadata_unchanged(self, value, saved, name):
        require(
            value.shape == saved.shape and value.dtype == saved.dtype and value.device == saved.device,
            "metadata provider changed native token geometry: " + name,
        )
        self.torch._assert_async((value == saved).all(), "metadata provider mutated native token metadata: " + name)


def make_selected_mlp_metadata(layer, metadata, plan):
    """Shallow-copy only the CP1 MoE-consumed per-token fields plus positions.

    The native MoE consumes is_prompt, valid_token_mask, gate_force_random and
    images_seq_mask; its token count comes from the tensors, not cu_seqlens_q.
    This is a private FFN-only view, never an attention/decoder metadata replacement.
    No host tensor construction or host transfer occurs here.
    """
    del layer
    result = copy(metadata)
    result.total_q = plan.selected_count
    for name in _ROW_FIELDS:
        value = getattr(metadata, name, None)
        if value is not None:
            setattr(result, name, value.index_select(0, plan.selected_rows))
    result.ffn_only_selected_view = True
    return result


class FFNOnlyAdapter:
    """Own only ``layer.mlp.forward``; never patch decoder or attention.

    get_plan(layer, metadata, bf16) returns None or a plan with full_rows,
    selected_rows (sorted int64 device indices), selected_count, current_positions
    (the identical metadata.positions object), request_id/phase/chunk_start/end,
    and plan_sha256. The caller owns sensitivity selection and GPU row evidence.

    make_selected_metadata(layer, metadata, plan) must return a fresh view with
    correctly gathered routing fields. Errors poison the adapter; they are never
    silently converted to full-row work or retried after distributed side effects.
    """

    def __init__(self, layers, get_plan, make_selected_metadata=make_selected_mlp_metadata, *, ops=None, allow_empty_native=None):
        self.layers = tuple(layers)
        require(
            self.layers
            and all(type(x.layer_idx) is int and x.layer_idx >= 0 for x in self.layers)
            and len({x.layer_idx for x in self.layers}) == len(self.layers)
            and len({id(x.mlp) for x in self.layers}) == len(self.layers),
            "distinct native decoder/MLP layers required",
        )
        require(callable(get_plan) and callable(make_selected_metadata), "FFN callbacks required")
        require(allow_empty_native is None or callable(allow_empty_native), "empty qualification callback invalid")
        self.get_plan, self.make_selected_metadata = get_plan, make_selected_metadata
        self.allow_empty_native, self.ops = allow_empty_native, ops
        self.handles, self.events = [], []
        self.installed = self.closed = self.poisoned = self.in_call = False
        self.counts = dict(
            native_mlp_calls=0,
            all_selected_mlp_calls=0,
            masked_mlp_calls=0,
            empty_native_mlp_calls=0,
            full_rows=0,
            selected_rows=0,
            inactive_ffn_zero_rows=0,
        )

    def _check(self):
        require(self.installed and not self.closed and not self.poisoned and not self.in_call, "FFN adapter unavailable or reentrant")

    def install(self):
        require(not self.installed and not self.closed, "FFN adapter cannot be reinstalled")
        for layer in self.layers:
            require(
                getattr(layer, "_sparse_interval_owner", None) is None
                and getattr(layer.mlp, "_early_mhc_owner", None) is None
                and getattr(layer.mlp, "_ffn_only_owner", None) is None
                and not getattr(type(layer.mlp), "_staged_moe_hook_installed", False),
                "conflicting FFN/interval owner",
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
        owner, original = self, layer.mlp.forward

        @wraps(original)
        def forward(module, hidden_states, hidden_states_bf16, input_metadata):
            owner._check()
            owner.in_call = True
            try:
                plan = owner.get_plan(layer, input_metadata, hidden_states_bf16) if input_metadata.is_prompt else None
                if plan is None:
                    result = original(hidden_states, hidden_states_bf16, input_metadata)
                    owner.counts["native_mlp_calls"] += 1
                    return result
                require(layer.layer_idx >= 6, "the first six native FFNs cannot be masked")
                require(not getattr(module, "use_mega_moe", False), "mega-MoE compact input ABI is unqualified")
                binding = _binding(input_metadata)
                require(
                    binding == (plan.request_id, plan.phase, plan.chunk_start, plan.chunk_end)
                    and type(plan.full_rows) is int
                    and plan.full_rows == input_metadata.total_q
                    and type(plan.selected_count) is int
                    and 0 <= plan.selected_count <= plan.full_rows,
                    "foreign FFN plan or current row count",
                )
                require(
                    isinstance(plan.plan_sha256, str)
                    and len(plan.plan_sha256) == 64
                    and all(c in "0123456789abcdef" for c in plan.plan_sha256),
                    "FFN plan seal required",
                )
                require(
                    getattr(plan, "current_positions", None) is getattr(input_metadata, "positions", None)
                    and getattr(plan, "current_positions", None) is not None,
                    "FFN plan must bind the actual current native position vector",
                )
                if owner.ops is None:
                    owner.ops = TorchFFNOnlyOps()
                ops = owner.ops
                ops.check_runtime()
                ops.validate_full(hidden_states, hidden_states_bf16, plan.full_rows)
                ops.validate_positions(plan.current_positions, hidden_states_bf16, plan.full_rows)
                ops.validate_plan(plan, hidden_states_bf16)
                all_selected = plan.selected_count == plan.full_rows
                if all_selected:
                    # Exact native aliasing, scale-factor storage, metadata and
                    # EP communication are retained when every row is selected.
                    output = original(hidden_states, hidden_states_bf16, input_metadata)
                    owner.counts["all_selected_mlp_calls"] += 1
                else:
                    empty = plan.selected_count == 0
                    require(
                        not empty
                        or (owner.allow_empty_native is not None and owner.allow_empty_native(layer, input_metadata, plan) is True),
                        "empty local FFN selection requires native EP qualification; collective cannot be skipped",
                    )
                    snapshot = _snapshot(input_metadata)
                    active = owner.make_selected_metadata(layer, input_metadata, plan)
                    _unchanged(input_metadata, snapshot, ops)
                    require(
                        active is not input_metadata
                        and active.total_q == plan.selected_count
                        and active.is_prompt is True
                        and active.num_reqs == input_metadata.num_reqs
                        and _binding(active) == binding,
                        "fresh scope-bound FFN metadata required",
                    )
                    ops.validate_selected_metadata(input_metadata, active, plan)
                    # Even zero-local-row calls enter original EP dispatch and
                    # combine; a rank may still receive work from other ranks.
                    selected = original(
                        ops.gather(hidden_states, plan.selected_rows), ops.gather(hidden_states_bf16, plan.selected_rows), active
                    )
                    output = ops.scatter_zero(selected, hidden_states_bf16, plan)
                    owner.counts["masked_mlp_calls"] += 1
                    owner.counts["empty_native_mlp_calls"] += int(empty)
                owner.counts["full_rows"] += plan.full_rows
                owner.counts["selected_rows"] += plan.selected_count
                owner.counts["inactive_ffn_zero_rows"] += plan.full_rows - plan.selected_count
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
                        all_selected_native=all_selected,
                        empty_native_collective=not all_selected and plan.selected_count == 0,
                        current_layout="native_current_rows",
                        inactive_ffn_output="zero",
                    )
                )
                return output
            except BaseException:
                owner.poisoned = True
                raise
            finally:
                owner.in_call = False

        old = layer.mlp.__dict__.get("forward", _MISSING)
        bound = MethodType(forward, layer.mlp)
        layer.mlp.forward = bound
        self.handles.append((layer.mlp, old, bound))
        layer.mlp._ffn_only_owner = self

    def report(self):
        require(self.installed and not self.poisoned and not self.in_call, "failed/incomplete FFN adapter")
        return dict(
            schema="kv-cache-engram-ffn-only-adapter-v1",
            counts=dict(self.counts),
            events=list(self.events),
            native_execution=None if self.ops is None else self.ops.native_execution,
            decoder_forward_unchanged=True,
            attention_forward_unchanged=True,
            native_mhc_unchanged=True,
            native_yoco_layout_unchanged=True,
            all_selected_delegates_native=True,
            empty_selection_skips_ep_collective=False,
            inactive_ffn_output="zero",
            cached_hidden_or_ffn_output=False,
            accuracy_guarantee=False,
            speedup_claim=False,
        )

    def close(self, *, allow_incomplete=False):
        if self.closed:
            return
        require(allow_incomplete or (not self.poisoned and not self.in_call), "failed/incomplete FFN adapter needs explicit cleanup")
        # Check every handle before restoring any: never partially unpatch a
        # stack whose ownership has changed behind us.
        require(
            all(
                obj.__dict__.get("forward", _MISSING) is bound and getattr(obj, "_ffn_only_owner", None) is self
                for obj, old, bound in self.handles
            ),
            "FFN patch ownership changed",
        )
        for obj, old, _bound in reversed(self.handles):
            if old is _MISSING:
                delattr(obj, "forward")
            else:
                obj.forward = old
            delattr(obj, "_ffn_only_owner")
        self.handles.clear()
        self.closed = True
