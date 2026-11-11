"""Isolated DeepInfer V4 SWA capture, never an online reuse hook.

The native capture point is the input of ``V4Attention._prepare_swa_kv``.
That input is projected but *not* normalized. NativeCaptureOps applies only
the native norm transform, explicitly without RoPE, casting or cache writes.
The original method still executes exactly once with its original arguments.

Importing this module imports neither Torch nor DeepInfer. The default operator
is a Python CPU reference, not a substitute for native numerical qualification.
Installing a session is an explicit, temporary instance-method interception on
caller-owned, isolated attention objects. No model is loaded or launched here.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import threading
from types import MappingProxyType, MethodType
from typing import Mapping

from .core import (CacheIdentity, ValidationError, integer,
                   normalize_rows, require, token_digest)


ENCODER_REUSE_LAYERS = frozenset(tuple(range(3, 8)) + tuple(range(9, 14)) + tuple(range(15, 19)))
CAPTURE_POINT = "V4Attention._prepare_swa_kv:projected-input/native-norm-only"
_OWNERS = {}
_OWNERS_LOCK = threading.Lock()


def _finite_epsilon(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            "positive finite RMSNorm epsilon required")
    return float(value)


class CPUReferenceCaptureOps:
    """Finite Python-row reference; test fixtures only, no device conversion."""

    label = "python-float64-rmsnorm-reference"
    native_operator = False
    accelerator_capture = False

    def finish(self):
        """CPU reference operations are synchronous."""

    def integer_matrix_shape(self, values):
        require(type(values) in (list, tuple) and bool(values), "integer matrix required")
        require(type(values[0]) in (list, tuple) and bool(values[0]), "integer matrix row required")
        width = len(values[0])
        require(all(type(row) in (list, tuple) and len(row) == width for row in values),
                "rectangular integer matrix required")
        return len(values), width

    def integer_matrix_rows(self, values, start, end):
        require(0 <= start < end <= len(values) and end - start <= 256,
                "bounded integer metadata slice required")
        result = tuple(tuple(row) for row in values[start:end])
        for row in result:
            for value in row:
                require(type(value) is int and value >= -1, "integer metadata value required")
        return result

    def row_shape(self, raw):
        require(type(raw) in (list, tuple) and bool(raw), "CPU row sequence required")
        require(type(raw[0]) in (list, tuple) and bool(raw[0]), "CPU row required")
        width = len(raw[0])
        require(all(type(row) in (list, tuple) and len(row) == width for row in raw),
                "rectangular CPU rows required")
        return len(raw), width

    def integers(self, values):
        require(type(values) in (list, tuple), "CPU integer sequence required")
        result = tuple(values)
        for value in result:
            integer(value, "metadata position")
        return result

    def normalize(self, module, raw):
        rows = normalize_rows(raw, width=module.qk_head_dim)
        weights = normalize_rows((module.kv_swa_norm.weight,), width=module.qk_head_dim)[0]
        eps = _finite_epsilon(module.kv_swa_norm.variance_epsilon)
        result = []
        for row in rows:
            mean_square = math.fsum(x * x for x in row) / len(row)
            require(math.isfinite(mean_square), "RMSNorm accumulation overflow")
            scale = 1.0 / math.sqrt(mean_square + eps)
            result.append(tuple(x * scale * w for x, w in zip(row, weights)))
        return normalize_rows(tuple(result), width=module.qk_head_dim)


class TorchCPUCaptureOps:
    """Actual Torch CPU tensors, FP32 RMSNorm reference; never calls CUDA APIs."""

    label = "torch-cpu-float32-rmsnorm-reference"
    native_operator = False
    accelerator_capture = False

    def __init__(self):
        import torch
        self.torch = torch

    def finish(self):
        """Torch CPU operations are synchronous."""

    def integer_matrix_shape(self, values):
        if type(values) in (list, tuple):
            return CPUReferenceCaptureOps().integer_matrix_shape(values)
        return tuple(self._tensor(values, ndim=2, floating=False).shape)

    def integer_matrix_rows(self, values, start, end):
        if type(values) in (list, tuple):
            return CPUReferenceCaptureOps().integer_matrix_rows(values, start, end)
        require(0 <= start < end <= values.shape[0] and end - start <= 256,
                "bounded integer metadata slice required")
        value = self._tensor(values, ndim=2, floating=False)
        rows = value[start:end].tolist()
        return CPUReferenceCaptureOps().integer_matrix_rows(rows, 0, len(rows))

    def _tensor(self, value, *, ndim, floating):
        torch = self.torch
        require(isinstance(value, torch.Tensor), "Torch tensor required")
        require(value.device.type == "cpu", "CPU ops reject accelerator tensors")
        require(value.layout == torch.strided and value.ndim == ndim, "dense tensor shape required")
        require(value.is_floating_point() == floating, "tensor dtype mismatch")
        if floating:
            require(bool(torch.isfinite(value).all()), "nonfinite tensor")
        else:
            require(value.dtype in (torch.int32, torch.int64), "integer metadata tensor required")
        return value.detach()

    def integers(self, values):
        if type(values) in (list, tuple):
            return CPUReferenceCaptureOps().integers(values)
        tensor = self._tensor(values, ndim=1, floating=False)
        return CPUReferenceCaptureOps().integers(tensor.tolist())

    def row_shape(self, raw):
        torch = self.torch
        require(isinstance(raw, torch.Tensor) and raw.device.type == "cpu"
                and raw.layout == torch.strided and raw.ndim == 2 and raw.is_floating_point(),
                "dense floating CPU projection required")
        return tuple(raw.shape)

    def normalize(self, module, raw):
        torch = self.torch
        value = self._tensor(raw, ndim=2, floating=True)
        weight = self._tensor(module.kv_swa_norm.weight, ndim=1, floating=True)
        require(value.shape[1] == module.qk_head_dim and weight.shape[0] == module.qk_head_dim,
                "native projection/norm width mismatch")
        eps = _finite_epsilon(module.kv_swa_norm.variance_epsilon)
        with torch.no_grad():
            value = value.float()
            mean_square = value.square().mean(dim=-1, keepdim=True)
            require(bool(torch.isfinite(mean_square).all()), "RMSNorm accumulation overflow")
            result = value * torch.rsqrt(mean_square + eps) * weight.float()
        return normalize_rows(result.tolist(), width=module.qk_head_dim)


class NativeCaptureOps:
    """Explicit offline native norm capture with synchronous device-to-CPU copy.

    ``from_deepinfer`` is never invoked implicitly. It does not create a model,
    GPU context, experiment or scheduler itself, but importing/calling a native
    runtime can initialize device libraries; callers must obtain execution
    authorization independently. Do not call it in CPU tests or against v8.
    Native outputs are converted to immutable Python rows for this diagnostic
    cache format; this is not a GPU cache-storage optimization.
    """

    # Dependency injection exercises contracts, not native provenance. Only
    # the explicit factory below upgrades this label and restricts CUDA use.
    label = "deepinfer-norm-dependency-fixture"
    native_operator = False
    accelerator_capture = False

    def __init__(self, *, torch_module, fused_transform, norm_args):
        require(callable(fused_transform) and callable(norm_args), "native operations required")
        self.torch = torch_module
        self.fused_transform = fused_transform
        self.norm_args = norm_args
        self._native_factory = False
        self._devices = set()

    @classmethod
    def from_deepinfer(cls, *, allow_accelerator_capture=False):
        require(allow_accelerator_capture is True,
                "native offline capture requires explicit accelerator opt-in")
        import torch
        from deepinfer import PLATFORM
        require(PLATFORM.is_sm90, "first native capture adapter supports the frozen SM90 path only")
        from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
            NormArgs, fused_norm_rope_hadamard_cast_store_kvcache,
        )
        result = cls(torch_module=torch,
                     fused_transform=fused_norm_rope_hadamard_cast_store_kvcache,
                     norm_args=NormArgs)
        result._native_factory = True
        result.label = "deepinfer-native-fused-rmsnorm-only"
        result.native_operator = result.accelerator_capture = True
        return result

    def finish(self):
        # The driver may enqueue more work after the final captured layer.
        # Publication must not precede completion of that work on any captured
        # device. A Python return alone is not a successful GPU completion.
        if self._native_factory:
            require(bool(self._devices), "native capture completed without a captured device")
            for device in sorted(self._devices, key=str):
                self.torch.cuda.synchronize(device)

    def _tensor(self, value, ndim):
        torch = self.torch
        require(isinstance(value, torch.Tensor) and value.ndim == ndim
                and value.layout == torch.strided, "native dense tensor required")
        require(self._native_factory or value.device.type == "cpu",
                "dependency fixtures reject non-CPU tensors")
        return value

    def integers(self, values):
        value = self._tensor(values, 1)
        require(value.dtype in (self.torch.int32, self.torch.int64), "integer native positions required")
        # Explicit offline synchronization: included in generation cost.
        return CPUReferenceCaptureOps().integers(value.detach().to(device="cpu").tolist())

    def integer_matrix_shape(self, values):
        value = self._tensor(values, 2)
        require(value.dtype in (self.torch.int32, self.torch.int64), "integer native matrix required")
        return tuple(value.shape)

    def integer_matrix_rows(self, values, start, end):
        self.integer_matrix_shape(values)
        require(0 <= start < end <= values.shape[0] and end - start <= 256,
                "bounded integer metadata slice required")
        rows = values[start:end].detach().to(device="cpu").tolist()
        return CPUReferenceCaptureOps().integer_matrix_rows(rows, 0, len(rows))

    def row_shape(self, raw):
        return tuple(self._tensor(raw, 2).shape)

    def normalize(self, module, raw):
        torch = self.torch
        value = self._tensor(raw, 2)
        weight = self._tensor(module.kv_swa_norm.weight, 1)
        require(value.device.type == ("cuda" if self._native_factory else "cpu"),
                "native factory requires CUDA; dependency fixtures require CPU")
        require(value.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16,
                "native canonical capture requires BF16 projection and norm weight")
        require(value.device == weight.device and value.shape[1] == module.qk_head_dim
                and weight.shape[0] == module.qk_head_dim, "native tensor contract mismatch")
        if value.device.type == "cuda":
            require(not torch.cuda.is_current_stream_capturing(), "offline capture cannot run inside a CUDA graph")
            self._devices.add(value.device)
        eps = _finite_epsilon(module.kv_swa_norm.variance_epsilon)
        with torch.no_grad():
            canonical = self.fused_transform(
                value, norm_args=self.norm_args(weight, eps),
                rope_args=None, cast_args=None, out_args=None,
            )
            require(isinstance(canonical, torch.Tensor) and canonical.shape == value.shape
                    and canonical.dtype == torch.bfloat16 and canonical.device == value.device,
                    "native norm-only output contract changed")
            cpu = canonical.detach().to(device="cpu", dtype=torch.float64)
        return normalize_rows(cpu.tolist(), width=module.qk_head_dim)


class DeepInferCaptureSession:
    """Capture one isolated, complete, single-request CP1 prefill.

    The caller's driver must attach ``session.chunk_marker(start, end)`` as
    ``input_metadata.canonical_request_binding`` for every actual chunk. The
    marker must originate from the same immutable input IDs supplied here;
    this adapter cannot prove a dishonest driver fed those IDs to the model.
    Each selected layer must visit all input positions exactly once and in
    increasing order. Only rows inside the declared document span are stored.
    Source/Engram/YOCO layers are outside this first adapter's reuse scope.
    """

    def __init__(self, attentions: Mapping[int, object], *, request_id: str,
                 phase: str, input_ids, identities: Mapping[int, CacheIdentity],
                 document_start: int, ops=None, sink=None,
                 max_retained_values: int = 1_048_576,
                 max_chunk_values: int = 8_388_608):
        require(isinstance(request_id, str) and bool(request_id.strip()), "request_id required")
        require(isinstance(phase, str) and bool(phase.strip()), "phase required")
        self.input_ids = tuple(input_ids)
        self.input_ids_sha256 = token_digest(self.input_ids)
        integer(document_start, "document_start")
        self.request_id, self.phase = request_id, phase
        self.document_start = document_start
        self.attentions = MappingProxyType(dict(attentions))
        self.identities = MappingProxyType(dict(identities))
        require(bool(self.attentions) and set(self.attentions) == set(self.identities),
                "matching nonempty attention/identity layer mappings required")
        require(all(type(layer) is int for layer in self.attentions), "integer layer keys required")
        require(set(self.attentions).issubset(ENCODER_REUSE_LAYERS), "source/Engram/YOCO layers forbidden")
        require(len(set(map(id, self.attentions.values()))) == len(self.attentions),
                "one attention object cannot own multiple layers")
        for layer, identity in self.identities.items():
            require(type(identity) is CacheIdentity and identity.layer == layer, "layer identity mismatch")
            require(document_start + identity.token_count <= len(self.input_ids), "document outside input")
            document_ids = self.input_ids[document_start:document_start + identity.token_count]
            require(token_digest(document_ids) == identity.tokens_sha256, "document token identity mismatch")
        common = [asdict(identity) for identity in self.identities.values()]
        for fields in common:
            fields.pop("layer")
        require(all(fields == common[0] for fields in common), "capture identities differ outside layer")
        self.document_end = document_start + next(iter(self.identities.values())).token_count
        require(sink is None or callable(sink), "capture sink must be callable")
        integer(max_retained_values, "max_retained_values", 1)
        integer(max_chunk_values, "max_chunk_values", 1)
        require(sink is not None or sum(i.token_count * i.width for i in self.identities.values())
                <= max_retained_values, "retained CPU KV budget exceeded; use a streaming sink")
        self._sink = sink
        self.max_chunk_values = max_chunk_values
        self.ops = CPUReferenceCaptureOps() if ops is None else ops
        require(callable(getattr(self.ops, "integers", None)) and callable(getattr(self.ops, "normalize", None))
                and callable(getattr(self.ops, "row_shape", None))
                and callable(getattr(self.ops, "integer_matrix_shape", None))
                and callable(getattr(self.ops, "integer_matrix_rows", None))
                and callable(getattr(self.ops, "finish", None))
                and isinstance(getattr(self.ops, "label", None), str)
                and type(getattr(self.ops, "native_operator", None)) is bool
                and type(getattr(self.ops, "accelerator_capture", None)) is bool, "capture operations required")
        self._rows = {layer: [] for layer in self.attentions}
        self._next = {layer: 0 for layer in self.attentions}
        self._chunks = []
        self._originals = {}
        self._wrappers = {}
        self._owner_thread = None
        self.state = "NEW"

    def chunk_marker(self, start, end):
        integer(start, "chunk start")
        integer(end, "chunk end", 1)
        require(start < end <= len(self.input_ids), "chunk outside complete input")
        return {"request_id": self.request_id, "phase": self.phase,
                "input_ids_sha256": self.input_ids_sha256, "start": start, "end": end}

    def _validate_module(self, layer, module):
        require(getattr(module, "layer_idx", None) == layer, "native layer index changed")
        require(getattr(module, "use_swa", None) is True, "SWA layer required")
        require(getattr(module, "qk_head_dim", None) == self.identities[layer].width, "native KV width mismatch")
        require(getattr(module, "window_size", None) == 128, "only the 128-token SWA contract is supported")
        for flag in ("is_kv_source", "is_index_source", "is_index_k_source",
                     "is_yoco_hidden_consumer_layer", "yoco_swa_share_hidden"):
            require(getattr(module, flag, None) is False, "source/YOCO or missing native flag: " + flag)
        norm = getattr(module, "kv_swa_norm", None)
        require(norm is not None and getattr(norm, "out_scale", 1.0) == 1.0,
                "unscaled SWA RMSNorm required")
        _finite_epsilon(norm.variance_epsilon)
        require(callable(getattr(module, "_prepare_swa_kv", None)), "native SWA prepare method required")
        for method in ("_prepare_swa_kv", "_project_q_and_swa", "forward"):
            require(method not in vars(module), "refusing existing instance hook: " + method)

    def __enter__(self):
        require(self.state == "NEW", "capture session is single-use")
        for layer, module in self.attentions.items():
            self._validate_module(layer, module)
        with _OWNERS_LOCK:
            require(all(id(module) not in _OWNERS for module in self.attentions.values()),
                    "attention is owned by another capture session")
            for module in self.attentions.values():
                _OWNERS[id(module)] = self
        self._owner_thread = threading.get_ident()
        try:
            for layer, module in self.attentions.items():
                original = getattr(module, "_prepare_swa_kv")
                self._originals[layer] = original

                def wrapper(owner, kv_swa, swa_cache, input_metadata, use_cp, _layer=layer):
                    return self._capture_call(_layer, owner, kv_swa, swa_cache, input_metadata, use_cp)

                wrapped = MethodType(wrapper, module)
                setattr(module, "_prepare_swa_kv", wrapped)
                self._wrappers[layer] = wrapped
            self.state = "ACTIVE"
            return self
        except BaseException:
            self.state = "ABORTED"
            self._restore()
            raise

    def _capture_call(self, layer, module, raw, swa_cache, metadata, use_cp):
        try:
            require(self.state == "ACTIVE" and threading.get_ident() == self._owner_thread,
                    "capture called outside its isolated owner thread")
            require(module is self.attentions[layer], "foreign attention instance")
            require(use_cp is False and getattr(metadata, "is_prompt", None) is True
                    and type(getattr(metadata, "num_reqs", None)) is int
                    and metadata.num_reqs == 1, "single-request CP1 prompt required")
            for name in ("redknot_binding", "head_intervention_binding", "attention_range_binding", "sparse_phase"):
                require(getattr(metadata, name, None) is None, "foreign intervention: " + name)
            marker = getattr(metadata, "canonical_request_binding", None)
            require(type(marker) is dict and type(marker.get("start")) is int and type(marker.get("end")) is int,
                    "canonical request chunk binding required")
            start, end = marker["start"], marker["end"]
            require(marker == self.chunk_marker(start, end), "foreign request/phase/input binding")
            require(start == self._next[layer], "gapped, duplicate or reordered capture chunk")
            positions = self.ops.integers(metadata.positions)
            require(positions == tuple(range(start, end)), "position shift or compacted rows")
            require(type(getattr(metadata, "total_q", None)) is int
                    and metadata.total_q == len(positions), "query row count mismatch")
            require(self.ops.integers(metadata.cu_seqlens_q) == (0, end - start), "request segments mismatch")
            require((end - start) * self.identities[layer].width <= self.max_chunk_values,
                    "capture chunk CPU copy budget exceeded")
            require(self.ops.row_shape(raw) == (end - start, self.identities[layer].width),
                    "raw projection shape differs from authenticated chunk")
            self._validate_full_swa_view(metadata, start, end)
            canonical = normalize_rows(self.ops.normalize(module, raw),
                                       width=self.identities[layer].width, count=end - start)
            # Publish no rows until the original native operation succeeds.
            output = self._originals[layer](raw, swa_cache, metadata, use_cp)
            lo, hi = max(start, self.document_start), min(end, self.document_end)
            if lo < hi:
                doc_rows = canonical[lo - start:hi - start]
                if self._sink is None:
                    self._rows[layer].extend(doc_rows)
                else:
                    self._sink(layer, lo - self.document_start, doc_rows)
            self._next[layer] = end
            self._chunks.append({"layer": layer, "start": start, "end": end,
                                 "document_rows": max(0, hi - lo)})
            return output
        except BaseException:
            self.state = "ABORTED"
            raise

    def _validate_full_swa_view(self, metadata, start, end):
        """Verify the *actual* native view, not unused YOCO candidate plans.

        prepare_tensor_input_v4.cpp builds these pivots for a single CP1
        request, W=128 and valid_start=0. compute_attn_info expands them before
        _prepare_swa_kv. Both representations must agree with the full causal
        window. Only 256 rows are transferred at once, never the whole index.
        """
        require(type(getattr(metadata, "total_k_swa", None)) is int
                and metadata.total_k_swa == 128, "single-request SWA history layout required")
        pivots = getattr(metadata, "kv_swa_full_index_pivots", None)
        expanded = getattr(metadata, "kv_swa_full_index", None)
        require(self.ops.integer_matrix_shape(pivots) == (end - start, 4),
                "full SWA pivot dimensions required")
        require(self.ops.integer_matrix_shape(expanded) == (end - start, 128),
                "expanded full SWA view required")
        for offset in range(0, end - start, 256):
            stop = min(offset + 256, end - start)
            pivot_rows = self.ops.integer_matrix_rows(pivots, offset, stop)
            index_rows = self.ops.integer_matrix_rows(expanded, offset, stop)
            for j, actual_pivot, actual_index in zip(range(offset, stop), pivot_rows, index_rows):
                abs_lo = max(start + j - 127, 0)
                start_slot = min(abs_lo, start) - (start - 128)
                local_lo = max(0, abs_lo - start)
                expected = (start_slot, 128 - start_slot, 128 + local_lo, j - local_lo + 1)
                require(actual_pivot == expected, "masked, zero-state or shifted SWA pivot")
                indices = (tuple(range(expected[0], expected[0] + expected[1]))
                           + tuple(range(expected[2], expected[2] + expected[3])))
                indices += (-1,) * (128 - len(indices))
                require(actual_index == indices, "masked or inconsistent expanded SWA view")

    def _restore(self):
        conflicts = []
        for layer, wrapper in self._wrappers.items():
            module = self.attentions[layer]
            if vars(module).get("_prepare_swa_kv") is wrapper:
                delattr(module, "_prepare_swa_kv")
            else:
                conflicts.append(layer)  # Never overwrite another actor's edit.
        with _OWNERS_LOCK:
            for module in self.attentions.values():
                if _OWNERS.get(id(module)) is self:
                    del _OWNERS[id(module)]
        self._wrappers.clear()
        if conflicts:
            self.state = "ABORTED"
            raise ValidationError("capture hook ownership changed at layers: " + repr(conflicts))

    def __exit__(self, exc_type, exc, traceback):
        complete = (exc_type is None and self.state == "ACTIVE"
                    and all(n == len(self.input_ids) for n in self._next.values()))
        if complete:
            try:
                self.ops.finish()
            except BaseException:
                self.state = "ABORTED"
                self._restore()
                raise
        self.state = "COMPLETE" if complete else "ABORTED"
        self._restore()
        if exc_type is None:
            require(complete, "incomplete or aborted capture cannot produce offline KV")
        return False

    @property
    def rows_by_layer(self):
        require(self.state == "COMPLETE", "only successful full coverage can be exported")
        require(self._sink is None, "streaming capture exports through its sink, not rows_by_layer")
        return MappingProxyType({layer: normalize_rows(tuple(rows),
                                 width=self.identities[layer].width,
                                 count=self.identities[layer].token_count)
                                 for layer, rows in self._rows.items()})

    @property
    def capture_provenance(self):
        require(self.state == "COMPLETE", "incomplete capture has no success provenance")
        return {"adapter": "deepinfer-v4-swa-capture-v1", "capture_point": CAPTURE_POINT,
                "request_id": self.request_id, "phase": self.phase,
                "input_ids_sha256": self.input_ids_sha256, "input_token_count": len(self.input_ids),
                "document_start": self.document_start, "document_end": self.document_end,
                "operator": self.ops.label, "native_operator": bool(self.ops.native_operator),
                "accelerator_capture": bool(self.ops.accelerator_capture),
                "streaming": self._sink is not None,
                "chunks": [dict(chunk) for chunk in self._chunks], "context_equivalence_certified": False,
                "native_numerical_alignment_measured": False,
                "identity_sha256_by_layer": {str(k): v.sha256 for k, v in self.identities.items()}}
