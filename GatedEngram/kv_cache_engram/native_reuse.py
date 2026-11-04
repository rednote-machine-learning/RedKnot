"""Isolated SM90 pilot: conditional SWA **KV projection** reuse, not sparse Q.

All Q, attention, residual and MoE rows follow the original forward. Only the
fourteen encoder global-KV consumer layers are eligible. Importing this module
does not import torch/DeepInfer, allocate a model, or launch a GPU. Native ops
are explicitly constructed on install; tests inject labelled CPU fixture ops.

This is an experimental, evidence-bound gate, not an accuracy certificate.
An accepted probe does not prove the remaining rows or the answer are correct.
Offline KV is post-RMSNorm, pre-RoPE BF16, never the CPU float64-reference format.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from types import MethodType, SimpleNamespace

from .core import digest, integer, require, token_digest, validate_digest
from .gate import GatePolicy

LAYERS = tuple(range(3, 8)) + tuple(range(9, 14)) + tuple(range(15, 19))
DOC, CHUNK, DIM, QDIM, KDIM = 32768, 8192, 512, 1280, 5120
REPRESENTATION = "swa-postnorm-prerope-v1"
NATIVE_DTYPE = "bfloat16-native-canonical"
ORACLE = {"relative_l2": 0.01, "absolute_error": 0.05}
MODES = ("native", "capture_independent", "shadow_full", "gated", "split_identity")
_ACTIVE = ContextVar("kv_cache_engram_native_frame", default=None)
_MISSING = object()


@dataclass(frozen=True)
class NativeCacheIdentity:
    model_sha256: str
    tokenizer_sha256: str
    execution_sha256: str
    tokens_sha256: str
    layer: int
    token_count: int = DOC
    width: int = DIM
    representation: str = REPRESENTATION
    storage_dtype: str = NATIVE_DTYPE

    def __post_init__(self):
        for key in ("model_sha256", "tokenizer_sha256", "execution_sha256", "tokens_sha256"):
            validate_digest(getattr(self, key), key)
        require(type(self.layer) is int and self.layer in LAYERS, "fourteen native consumer layers only")
        require(
            self.token_count == DOC and self.width == DIM and self.representation == REPRESENTATION and self.storage_dtype == NATIVE_DTYPE,
            "native canonical BF16 document geometry required",
        )

    @property
    def sha256(self):
        return digest(asdict(self))


def _scope(identity):
    require(type(identity) is NativeCacheIdentity, "native BF16 identity required")
    fields = asdict(identity)
    NativeCacheIdentity(**fields)
    del fields["tokens_sha256"], fields["token_count"]
    return digest(fields)


@dataclass(frozen=True)
class NativeCalibrationRecord:
    """Integrity binding to explicitly supplied pilot evidence, not certification."""

    scope_sha256: str
    policy_sha256: str
    evidence_sha256: str
    checksum: str

    @classmethod
    def bind(cls, identity, policy, evidence_sha256):
        require(type(policy) is GatePolicy, "GatePolicy required")
        policy.__post_init__()
        require(policy.enabled and bool(policy.calibration_id), "explicit enabled calibrated policy required")
        validate_digest(evidence_sha256, "calibration evidence")
        values = (_scope(identity), digest(asdict(policy)), evidence_sha256)
        return cls(*values, digest(["kce-native-pilot-calibration-v1", *values]))

    def validate(self, identity, policy):
        require(type(self) is NativeCalibrationRecord, "native calibration binding required")
        for value in (self.scope_sha256, self.policy_sha256, self.evidence_sha256, self.checksum):
            validate_digest(value, "calibration field")
        require(self == self.bind(identity, policy, self.evidence_sha256), "native calibration scope/policy mismatch")


def _policy(policy):
    require(type(policy) is GatePolicy, "explicit GatePolicy required")
    policy.__post_init__()
    require(
        policy.block_size == 512 and policy.head_guard == 128 and policy.tail_guard == 128,
        "pilot requires 512-row blocks and 128-row document guards",
    )
    return policy


def probe_plan(identity, start, end, policy):
    """Native-identity-bound stratified probes; no row values influence selection."""
    _policy(policy)
    require(start % 512 == 0 and end == min(start + 512, DOC), "complete aligned native block required")
    mandatory = tuple(i for i in range(start, end) if i < 128 or i >= DOC - 128)
    guard = set(mandatory)
    candidates = tuple(i for i in range(start, end) if i not in guard)
    count = min(policy.probe_count, len(candidates))
    probes = []
    for stratum in range(count):
        lo, hi = stratum * len(candidates) // count, (stratum + 1) * len(candidates) // count
        material = ["kce-native-stratified-probes-v1", policy.seed, identity.sha256, start, end, stratum, count, lo, hi]
        probes.append(candidates[lo + int(digest(material), 16) % (hi - lo)])
    return tuple(probes), mandatory


def document_plan(ids, documents, *, independent=False):
    ids = tuple(ids)
    token_digest(ids)
    require(all(i < 2**31 for i in ids), "native token IDs must fit int32")
    if independent:
        require(len(ids) == DOC and documents is None, "independent capture is one untemplated 32K document")
        return ({"start": 0, "end": DOC, "key": token_digest(ids)},)
    require(type(documents) in (list, tuple) and len(documents) == 8, "exact eight-document map required")
    plan = []
    for doc in documents:
        start, end = doc.get("input_start"), doc.get("input_end")
        require(
            type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(ids)
            and end - start == DOC
            and (not plan or start == plan[-1]["end"]),
            "ordered contiguous 32K input spans required",
        )
        key = token_digest(ids[start:end])
        require(
            doc.get("token_ids_sha256") == key and doc.get("document_id", key) == key and doc.get("offline_position", 0) == 0,
            "document token/position identity mismatch",
        )
        plan.append({"start": start, "end": end, "key": key})
    require(len({p["key"] for p in plan}) == 8, "pilot requires eight distinct independent documents")
    return tuple(plan)


def frame_parts(documents, start, end):
    """Partition intersecting document rows, making cross-chunk fragments explicit."""
    result = []
    for doc in documents:
        left, right = max(start, doc["start"]), min(end, doc["end"])
        local = left - doc["start"]
        stop = right - doc["start"]
        while local < stop:
            block_start = local // 512 * 512
            finish = min(stop, block_start + 512)
            result.append(
                dict(
                    document=doc,
                    start=local,
                    end=finish,
                    row_start=doc["start"] + local - start,
                    eligible=local == block_start and finish == block_start + 512,
                )
            )
            local = finish
    return result


class NativeReuseOps:
    """Lazy SM90 implementation. The constructor is not a GPU/model launcher."""

    backend = "deepinfer-sm90-native-bfloat16-kv-only-v1"
    native_execution = True

    @classmethod
    def from_deepinfer(cls):
        import torch
        from deepinfer import PLATFORM
        from deepinfer.model_executor.layers.linear import ReplicatedLinear
        from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
            CastArgs,
            NormArgs,
            RopeArgs,
            fused_norm_rope_hadamard_cast_store_kvcache,
        )
        from deepinfer.model_executor.parallel_utils.parallel_state import PG

        require(PLATFORM.is_sm90 and PG.context_parallel_group().size() == 1, "native KV-only pilot requires SM90 CP1")
        obj = cls()
        obj.torch, obj.Linear, obj.PG = torch, ReplicatedLinear, PG
        obj.NormArgs, obj.RopeArgs, obj.CastArgs = NormArgs, RopeArgs, CastArgs
        obj.transform = fused_norm_rope_hadamard_cast_store_kvcache
        obj.devices = set()
        return obj

    def make_projections(self, module):
        t, fused = self.torch, module.q_a_kv_swa_proj
        require(
            module.layer_idx in LAYERS
            and module.fuse_kv_swa
            and not module.yoco_swa_share_hidden
            and module.q_lora_rank == QDIM
            and module.qk_head_dim == DIM
            and fused.input_size == KDIM
            and fused.output_size == QDIM + DIM
            and fused.bias is None
            and not fused.use_fp32_output,
            "native fused Q/KV geometry changed",
        )
        w = fused.weight
        require(
            w.is_cuda
            and w.dtype == t.float8_e4m3fn
            and tuple(w.shape) == (QDIM + DIM, KDIM)
            and w.stride(-1) == 1
            and hasattr(w, "sf")
            and w.sf.dtype == t.int32
            and tuple(w.sf.shape) == (QDIM + DIM, KDIM // 128)
            and w.sf.device == w.device,
            "native FP8 rows and packed UE8M0 scales required",
        )
        result = []
        for lo, hi in ((0, QDIM), (QDIM, QDIM + DIM)):
            param = t.nn.Parameter(w[lo:hi], requires_grad=False)
            param.sf = w.sf[lo:hi]
            linear = self.Linear(KDIM, hi - lo, weight=param)
            require(
                linear.weight.data_ptr() == w[lo:hi].data_ptr()
                and linear.weight.stride() == w[lo:hi].stride()
                and linear.weight.sf.data_ptr() == w.sf[lo:hi].data_ptr()
                and linear.weight.sf.stride() == w.sf[lo:hi].stride(),
                "split projection must preserve native weight and SF storage views",
            )
            result.append(linear)
        return tuple(result)

    def validate(self, module, metadata, cross, marker):
        t = self.torch
        require(self.PG.context_parallel_group().size() == 1 and not t.cuda.is_current_stream_capturing(), "eager CP1 required")
        require(
            module.use_swa
            and module.has_packed_kv
            and module.is_sparse
            and module.use_shared_kv
            and not module.is_yoco_hidden_consumer_layer
            and not module.yoco_swa_share_hidden
            and not module.use_fused_norm_rope_attn_rope_cast
            and module.qk_rope_head_dim == 64
            and module.window_size == 128
            and not module.is_kv_source
            and not module.is_index_source
            and not module.is_index_k_source
            and not module.has_local_index
            and module.current_hidden_proj is None
            and module.shared_hidden_kv_proj is None,
            "only original fourteen full-view encoder KV/index consumers supported",
        )
        require(cross is not None and cross.consumer_full_metadata is None, "compact or alternate metadata views are forbidden")
        n = marker["end"] - marker["start"]
        require(
            metadata.is_prompt and metadata.num_reqs == 1 and metadata.total_q == n and metadata.canonical_request_binding == marker,
            "prefill metadata binding mismatch",
        )
        for key in ("redknot_binding", "head_intervention_binding", "attention_range_binding", "sparse_phase"):
            require(getattr(metadata, key, None) is None, "foreign attention intervention: " + key)
        p = metadata.positions
        require(
            isinstance(p, t.Tensor) and p.is_cuda and p.ndim == 1 and p.numel() == n and p.dtype in (t.int32, t.int64),
            "native full positions required",
        )
        t._assert_async(
            (p == t.arange(marker["start"], marker["end"], device=p.device, dtype=p.dtype)).all(),
            "native positions differ from verified full chunk",
        )
        self.devices.add(p.device)

    def validate_full_view(self, metadata, marker):
        t = self.torch
        n, start = marker["end"] - marker["start"], marker["start"]
        pivots = getattr(metadata, "kv_swa_full_index_pivots", None)
        index = getattr(metadata, "kv_swa_full_index", None)
        require(
            getattr(metadata, "total_k_swa", None) == 128
            and isinstance(pivots, t.Tensor)
            and tuple(pivots.shape) == (n, 4)
            and pivots.dtype in (t.int32, t.int64)
            and isinstance(index, t.Tensor)
            and tuple(index.shape) == (n, 128)
            and index.dtype in (t.int32, t.int64),
            "complete normal SWA view required",
        )
        j = t.arange(n, device=pivots.device, dtype=pivots.dtype)
        low = (start + j - 127).clamp(min=0)
        slot = t.minimum(low, t.full_like(low, start)) - (start - 128)
        local = (low - start).clamp(min=0)
        expected = t.stack((slot, 128 - slot, 128 + local, j - local + 1), dim=1)
        t._assert_async((pivots == expected).all(), "normal full SWA pivots changed")
        columns = t.arange(128, device=index.device, dtype=index.dtype)[None, :]
        exp = expected.to(index.device, dtype=index.dtype)
        normal = t.where(
            columns < exp[:, 1:2],
            columns + exp[:, 0:1],
            t.where(columns < exp[:, 1:2] + exp[:, 3:4], columns - exp[:, 1:2] + exp[:, 2:3], -1),
        )
        t._assert_async((index == normal).all(), "active SWA view is masked or compact")

    def rows(self, hidden):
        t = self.torch
        x = hidden[0] if isinstance(hidden, tuple) else hidden
        require(
            isinstance(x, t.Tensor) and x.is_cuda and x.ndim == 2 and x.shape[1] == KDIM, "native hidden projection shape/device mismatch"
        )
        if isinstance(hidden, tuple):
            require(
                len(hidden) == 2
                and x.dtype == t.float8_e4m3fn
                and hidden[1].dtype == t.int32
                and tuple(hidden[1].shape) == (x.shape[0], KDIM // 128)
                and hidden[1].device == x.device,
                "native packed input SF rows required",
            )
        else:
            require(x.dtype == t.bfloat16, "BF16 input or native FP8+SF tuple required")
        return x.shape[0]

    def take_hidden(self, hidden, positions):
        x = hidden[0] if isinstance(hidden, tuple) else hidden
        idx = self.torch.tensor(positions, device=x.device, dtype=self.torch.long)
        if isinstance(hidden, tuple):
            # SF is [token, packed-channel]; gather token rows, never channel axis.
            # SM90 gemm_nt passes both SF strides into its masked tl.load path;
            # unlike an SM100 TMA path, it accepts this row-major gathered SF
            # and any M (including 33/129). Split identity qualifies both.
            return tuple(v.index_select(0, idx) for v in hidden)
        return x.index_select(0, idx)

    def validate_tensor(self, value, rows=None):
        t = self.torch
        require(
            isinstance(value, t.Tensor)
            and value.is_cuda
            and value.dtype == t.bfloat16
            and value.ndim == 2
            and value.shape[1] == DIM
            and (rows is None or value.shape[0] == rows),
            "native canonical KV tensor shape/dtype/device mismatch",
        )
        t._assert_async(t.isfinite(value).all(), "nonfinite native canonical KV")
        self.devices.add(value.device)

    def normalize(self, module, raw):
        self.validate_tensor(raw)
        epsilon = module.kv_swa_norm.variance_epsilon
        require(
            type(epsilon) in (int, float)
            and math.isfinite(epsilon)
            and epsilon > 0
            and getattr(module.kv_swa_norm, "out_scale", 1.0) == 1.0,
            "finite positive native norm epsilon and unit output scale required",
        )
        # weight=None is a legal native unweighted RMSNorm, not a missing KV.
        result = self.transform(
            raw, norm_args=self.NormArgs(module.kv_swa_norm.weight, epsilon), rope_args=None, cast_args=None, out_args=None
        )
        self.validate_tensor(result, raw.shape[0])
        return result

    def allocate(self, rows, like):
        # DeepInfer executes its model step under inference_mode(). Resident
        # offline entries must nevertheless be ordinary tensors: their version
        # counters are the mutation guard checked before every reuse/export.
        # Disabling inference only around allocation retains that guard without
        # enabling autograd or changing the caller's inference-mode scope.
        with self.torch.inference_mode(False), self.torch.no_grad():
            return self.torch.empty((rows, DIM), device=like.device, dtype=self.torch.bfloat16, requires_grad=False)

    def copy(self, destination, value):
        destination.copy_(value)

    def take(self, value, positions):
        return value.index_select(0, self.torch.tensor(positions, device=value.device, dtype=self.torch.long))

    def scatter(self, destination, positions, values):
        destination.index_copy_(0, self.torch.tensor(positions, device=destination.device, dtype=self.torch.long), values)

    def slice(self, value, start, end):
        return value[start:end]

    def version(self, value):
        return value._version

    def postprocess(self, module, canonical, metadata):
        self.validate_tensor(canonical, metadata.total_q)
        result = self.transform(
            canonical,
            norm_args=None,
            rope_args=self.RopeArgs(metadata.positions, module.rotary_emb.cos_sin_cache, 64, module.rotary_emb.is_neox_style, offset=448),
            cast_args=self.CastArgs("e4m3", module.swa_cache_num_per_channels, round_sf=True, sf_format="ue8m0", cast_back=True),
            out_args=None,
        )
        self.validate_tensor(result, metadata.total_q)
        return result.reshape(result.shape[0], 1, DIM), None

    def errors(self, current, cached, floor):
        self.validate_tensor(current)
        self.validate_tensor(cached, current.shape[0])
        a, b = current.to(self.torch.float64), cached.to(self.torch.float64)
        delta = a - b
        relative = delta.norm(dim=1) / a.norm(dim=1).clamp(min=floor)
        absolute = delta.abs().amax(dim=1)
        return tuple(zip(relative.detach().cpu().tolist(), absolute.detach().cpu().tolist(), strict=False))

    def identity_error(self, actual, expected):
        t = self.torch
        require(
            actual.shape == expected.shape and actual.device == expected.device and actual.dtype == expected.dtype,
            "split identity shape/dtype/device mismatch",
        )
        t._assert_async(t.isfinite(actual).all() & t.isfinite(expected).all(), "nonfinite split identity")
        delta = (actual.to(t.float64) - expected.to(t.float64)).abs().max().item()
        require(delta == 0, "new split projection/gather differs from native computation")
        return delta

    def prepared_tensor(self, prepared):
        require(isinstance(prepared, tuple) and len(prepared) == 2 and prepared[1] is None, "native CP1 prepared SWA result required")
        return prepared[0]

    def synchronize(self):
        for device in self.devices:
            self.torch.cuda.synchronize(device)


@dataclass
class _Frame:
    owner: object
    module: object
    metadata: object
    marker: dict
    projected: bool = False
    prepared: bool = False
    canonical: object = None
    split_canonical: object = None


class KVCacheEngramHook:
    """Owned instance patches; no scheduling/model construction or native edits.

    ``ops`` is a complete injectable tensor boundary. Non-native fixture ops
    remain labelled fixture in reports and cannot qualify a GPU pilot.
    """

    def __init__(
        self,
        modules,
        *,
        model_sha256,
        tokenizer_sha256,
        execution_sha256,
        enabled=False,
        ops=None,
        allow_sync=None,
        max_cache_bytes=14 * 8 * DOC * DIM * 2,
        sink=None,
    ):
        for value in (model_sha256, tokenizer_sha256, execution_sha256):
            validate_digest(value, "native scope digest")
        integer(max_cache_bytes, "max_cache_bytes", 1)
        require(type(enabled) is bool, "enabled must be explicit")
        values = list(modules.values()) if isinstance(modules, dict) else list(modules)
        self.modules = {m.layer_idx: m for m in values if m.layer_idx in LAYERS}
        require(
            sum(m.layer_idx in LAYERS for m in values) == len(LAYERS)
            and tuple(sorted(self.modules)) == LAYERS
            and len({id(m) for m in self.modules.values()}) == len(LAYERS),
            "exact fourteen distinct native layer modules required",
        )
        self.scope = dict(
            model_sha256=model_sha256,
            tokenizer_sha256=tokenizer_sha256,
            execution_sha256=execution_sha256,
            representation=REPRESENTATION,
            storage_dtype=NATIVE_DTYPE,
            width=DIM,
            layers=list(LAYERS),
        )
        self.enabled, self.ops, self.allow_sync = enabled, ops, allow_sync or nullcontext
        self.max_cache_bytes, self.sink = max_cache_bytes, sink
        self.state, self.handles, self.projections, self.bank = "NEW", [], {}, {}
        self.request_id = None
        self.identity_qualified = False

    def _identity(self, key, layer):
        return NativeCacheIdentity(self.scope["model_sha256"], self.scope["tokenizer_sha256"], self.scope["execution_sha256"], key, layer)

    def install(self):
        require(self.state == "NEW", "hook already installed or closed")
        if self.ops is None:
            self.ops = NativeReuseOps.from_deepinfer()
        try:
            for layer, module in self.modules.items():
                require(getattr(module, "_kv_cache_engram_native_hook", None) is None, "another Engram hook owns native attention")
                for name in ("forward", "_project_q_and_swa", "_prepare_swa_kv"):
                    require(name not in module.__dict__, "foreign instance attention hook: " + name)
                self.projections[layer] = self.ops.make_projections(module)
            for module in self.modules.values():
                for name, handler in (
                    ("forward", self._forward),
                    ("_project_q_and_swa", self._project),
                    ("_prepare_swa_kv", self._prepare),
                ):
                    original = getattr(module, name)

                    def wrapped(mod, *args, _original=original, _handler=handler, **kwargs):
                        return _handler(mod, _original, *args, **kwargs)

                    wrapper = MethodType(wrapped, module)
                    setattr(module, name, wrapper)
                    self.handles.append((module, name, wrapper))
                module._kv_cache_engram_native_hook = self
            self.state = "IDLE"
            return self
        except BaseException:
            self._restore()
            self.state = "ABORTED"
            raise

    def reset(self, request_id, mode, *, input_ids, documents=None, phase, policies=None, calibrations=None, candidate_policies=()):
        require(self.state in ("IDLE", "COMPLETE"), "previous native request is incomplete/poisoned")
        require(isinstance(request_id, str) and request_id and isinstance(phase, str) and phase, "explicit request and phase required")
        require(mode in MODES and (self.enabled or mode == "native"), "disabled or unsupported native mode")
        ids = tuple(input_ids)
        plan = document_plan(ids, documents, independent=mode == "capture_independent")
        policies, calibrations = dict(policies or {}), dict(calibrations or {})
        require(set(policies) <= set(LAYERS) and set(calibrations) <= set(LAYERS), "foreign layer policy")
        for policy in policies.values():
            _policy(policy)
        require(
            mode != "gated" or not any(p.enabled for p in policies.values()) or self.identity_qualified,
            "full split/gather numerical qualification must complete before gated execution",
        )
        candidates = tuple(_policy(p) for p in candidate_policies)
        require(len({digest(asdict(p)) for p in candidates}) == len(candidates), "duplicate candidate policy")
        if mode == "shadow_full":
            require(bool(candidates), "shadow requires predeclared candidate policies")
        if mode in ("gated", "shadow_full"):
            for entry in self.bank.values():
                require(
                    entry["complete"] and self.ops.version(entry["tensor"]) == entry["version"],
                    "resident offline cache changed after capture",
                )
        if mode == "shadow_full":
            require(all((doc["key"], layer) in self.bank for doc in plan for layer in LAYERS), "complete offline bank required for shadow")
        if mode == "capture_independent":
            require(all((plan[0]["key"], layer) not in self.bank for layer in LAYERS), "offline cache may not be overwritten")
            require((len(self.bank) + len(LAYERS)) * DOC * DIM * 2 <= self.max_cache_bytes, "resident cache budget exceeded")
        self.request_id, self.phase, self.mode = request_id, phase, mode
        self.ids, self.input_sha256, self.documents = ids, token_digest(ids), plan
        self.policies, self.calibrations, self.candidates = policies, calibrations, candidates
        self.coverage, self.events, self.qualifications = {layer: 0 for layer in LAYERS}, [], []
        self.counts = {
            layer: dict(q_rows=0, native_fused_rows=0, split_q_rows=0, online_kv_rows=0, reused_kv_rows=0, probe_rows=0, fallback_rows=0)
            for layer in LAYERS
        }
        self.state = "RUNNING"

    def chunk_marker(self, start, end):
        return dict(request_id=self.request_id, phase=self.phase, start=start, end=end)

    def _forward(self, module, original, *args, **kwargs):
        metadata = kwargs.get("input_metadata", args[2] if len(args) > 2 else None)
        marker = getattr(metadata, "canonical_request_binding", None)
        if marker is None:
            # Native decode/EP placeholders have no authenticated real-prefill marker.
            return original(*args, **kwargs)
        try:
            require(
                self.state == "RUNNING" and type(marker) is dict and set(marker) == {"request_id", "phase", "start", "end"},
                "unexpected bound native request",
            )
            start, end = marker["start"], marker["end"]
            require(
                type(start) is int
                and type(end) is int
                and marker == self.chunk_marker(start, end)
                and start == self.coverage[module.layer_idx]
                and start < end <= len(self.ids)
                and end - start <= CHUNK,
                "native request/chunk binding or contiguous coverage mismatch",
            )
            require(_ACTIVE.get() is None, "nested attention intervention forbidden")
            cross = kwargs.get("cross_layer_runtime", args[4] if len(args) > 4 else None)
            # The timed standard arm is a GPU-operator passthrough: do not
            # burden its denominator with Engram position/index/finite kernels.
            # CPU ownership/coverage markers remain common instrumentation.
            if self.mode != "native":
                self.ops.validate(module, metadata, cross, marker)
            frame = _Frame(self, module, metadata, dict(marker))
            token = _ACTIVE.set(frame)
            try:
                result = original(*args, **kwargs)
                require(frame.projected and frame.prepared, "native forward skipped captured projection/preparation")
            finally:
                _ACTIVE.reset(token)
            self.coverage[module.layer_idx] = end
            return result
        except BaseException:
            self.state = "ABORTED"
            raise

    def _frame(self, module):
        frame = _ACTIVE.get()
        if frame is None:
            return None
        require(frame.owner is self and frame.module is module and self.state == "RUNNING", "foreign active frame")
        return frame

    def _project(self, module, original, hidden_states, shared_hidden_bf16):
        frame = self._frame(module)
        if frame is None:
            return original(hidden_states, shared_hidden_bf16)
        require(not frame.projected and shared_hidden_bf16 is None, "duplicate/YOCO projection forbidden")
        frame.projected = True
        n = self.ops.rows(hidden_states)
        require(n == frame.marker["end"] - frame.marker["start"], "hidden rows differ from full chunk")
        counts = self.counts[module.layer_idx]
        counts["q_rows"] += n
        if self.mode == "native":
            counts["native_fused_rows"] += n
            counts["online_kv_rows"] += n
            return original(hidden_states, shared_hidden_bf16)
        self.ops.validate_full_view(frame.metadata, frame.marker)
        if self.mode == "gated":
            output = self._gated(frame, hidden_states)
            if output is not None:
                return output
        q, raw = original(hidden_states, shared_hidden_bf16)
        counts["native_fused_rows"] += n
        counts["online_kv_rows"] += n
        self.ops.validate_tensor(raw, n)
        if self.mode == "split_identity":
            qproj, kvproj = self.projections[module.layer_idx]
            qsplit, kvsplit = qproj(hidden_states), kvproj(hidden_states)
            canonical = self.ops.normalize(module, raw)
            split_canonical = self.ops.normalize(module, kvsplit)
            with self.allow_sync():
                errors = {
                    "q": self.ops.identity_error(qsplit, q),
                    "kv_raw": self.ops.identity_error(kvsplit, raw),
                    "kv_canonical": self.ops.identity_error(split_canonical, canonical),
                }
                # Include noncontiguous packed-SF gathers at both declared probe densities.
                for count in (32, 128):
                    positions = tuple(sorted({i * n // min(count, n) for i in range(min(count, n))} | {0, n - 1}))
                    selected = self.ops.normalize(module, kvproj(self.ops.take_hidden(hidden_states, positions)))
                    errors["gather" + str(count)] = self.ops.identity_error(selected, self.ops.take(canonical, positions))
            frame.split_canonical = split_canonical
            self.qualifications.append(
                dict(layer=module.layer_idx, start=frame.marker["start"], end=frame.marker["end"], errors=errors, exact=True)
            )
        elif self.mode in ("capture_independent", "shadow_full"):
            canonical = self.ops.normalize(module, raw)
            if self.mode == "capture_independent":
                self._capture(frame, canonical)
            else:
                self._shadow(frame, canonical)
        return q, raw

    def _prepare(self, module, original, kv_swa, swa_cache, input_metadata, use_cp):
        frame = self._frame(module)
        if frame is None:
            return original(kv_swa, swa_cache, input_metadata, use_cp)
        require(
            frame.projected and not frame.prepared and input_metadata is frame.metadata and use_cp is False,
            "native prepare ownership/CP mismatch",
        )
        frame.prepared = True
        if frame.canonical is not None:
            require(kv_swa is frame.canonical, "hybrid canonical tensor replaced before preparation")
            return self.ops.postprocess(module, kv_swa, input_metadata)
        result = original(kv_swa, swa_cache, input_metadata, use_cp)
        if self.mode == "split_identity":
            restored = self.ops.postprocess(module, frame.split_canonical, input_metadata)
            with self.allow_sync():
                error = self.ops.identity_error(self.ops.prepared_tensor(restored), self.ops.prepared_tensor(result))
                self.qualifications[-1]["errors"]["post_rope_cast"] = error
                # The diagnostic path restores subsets at their actual absolute
                # positions. Qualify that operation separately from the whole
                # chunk: per-row FP8 casting must not depend on batch size/order.
                n = frame.marker["end"] - frame.marker["start"]
                native_positions = getattr(input_metadata, "positions", None)
                require(
                    native_positions is not None or not self.ops.native_execution, "native subset qualification requires absolute positions"
                )
                for count in (32, 128, 384, 512):
                    size = min(count, n)
                    positions = tuple(i * (n - 1) // (size - 1) for i in range(size)) if size > 1 else (0,)
                    partial = SimpleNamespace(
                        total_q=size, positions=(self.ops.take(native_positions, positions) if native_positions is not None else None)
                    )
                    sampled = self.ops.postprocess(module, self.ops.take(frame.split_canonical, positions), partial)
                    expected = self.ops.take(self.ops.prepared_tensor(result), positions)
                    self.qualifications[-1]["errors"]["post_rope_gather" + str(count)] = self.ops.identity_error(
                        self.ops.prepared_tensor(sampled), expected
                    )
        return result

    def _sink(self, layer, doc, local_start, canonical):
        if self.sink is not None:
            with self.allow_sync():
                self.sink(self.mode, layer, doc["key"], local_start, canonical)

    def _capture(self, frame, canonical):
        layer, start, end = frame.module.layer_idx, frame.marker["start"], frame.marker["end"]
        doc = self.documents[0]
        key = (doc["key"], layer)
        if key not in self.bank:
            require(start == 0, "offline capture must begin at document zero")
            self.bank[key] = dict(
                tensor=self.ops.allocate(DOC, canonical), cursor=0, complete=False, identity=self._identity(doc["key"], layer)
            )
        entry = self.bank[key]
        require(entry["cursor"] == start and not entry["complete"], "offline coverage is not contiguous")
        self.ops.copy(self.ops.slice(entry["tensor"], start, end), canonical)
        entry["cursor"] = end
        self._sink(layer, doc, start, canonical)

    def _event(self, frame, part, policy, errors, *, actual=False):
        layer, doc = frame.module.layer_idx, part["document"]
        start, end = part["start"], part["end"]
        identity = self._identity(doc["key"], layer)
        probes, mandatory = probe_plan(identity, start, end, policy) if part["eligible"] else ((), tuple(range(start, end)))
        keep = set(probes) | set(mandatory)
        candidates = tuple(i for i in range(start, end) if i not in keep)

        def maximum(positions):
            values = [errors[p - start] for p in positions]
            return (max(v[0] for v in values), max(v[1] for v in values)) if values else (None, None)

        nonfinite = sum(not (math.isfinite(r) and math.isfinite(a)) for r, a in errors)
        require(nonfinite == 0, "nonfinite gate/oracle metric")
        full = maximum(range(start, end))
        pr, ca = maximum(probes), maximum(candidates)
        accepted = bool(
            part["eligible"]
            and policy.enabled
            and probes
            and candidates
            and pr[0] <= policy.relative_tolerance
            and pr[1] <= policy.absolute_tolerance
        )
        bad = lambda positions: sum(
            errors[p - start][0] > ORACLE["relative_l2"] or errors[p - start][1] > ORACLE["absolute_error"] for p in positions
        )
        bad_candidates = bad(candidates)
        event = dict(
            layer=layer,
            document_sha256=doc["key"],
            document_local_start=start,
            document_local_end=end,
            request_id=self.request_id,
            phase=self.phase,
            input_sha256=self.input_sha256,
            candidate_id=digest(asdict(policy)),
            candidate_sha256=digest(asdict(policy)),
            eligible_for_reuse=part["eligible"],
            mode=self.mode if part["eligible"] else "full_recompute",
            fallback_reason=None if part["eligible"] else "cross_chunk_fragment",
            probe_count=len(probes),
            mandatory_count=len(mandatory),
            candidate_reusable_rows=len(candidates),
            reusable_rows=len(candidates),
            probe_relative_max=pr[0],
            probe_absolute_max=pr[1],
            candidate_relative_max=ca[0],
            candidate_absolute_max=ca[1],
            full_relative_max=full[0],
            full_absolute_max=full[1],
            nonfinite_count=0,
            nonfinite=False,
            accepted=accepted,
            bad_candidate_rows=bad_candidates,
            oracle_bad_rows=bad(range(start, end)),
            oracle_bad_reused_rows=bad_candidates if accepted else 0,
            would_reuse_rows=len(candidates) if accepted else 0,
            actual_reused_rows=len(candidates) if actual and accepted else 0,
        )
        return event

    def _shadow(self, frame, canonical):
        start, end, layer = frame.marker["start"], frame.marker["end"], frame.module.layer_idx
        # Compare once per policy norm floor per document/chunk; all-row results
        # stay observational and do not replace the native raw KV output.
        errors = {}
        for doc in self.documents:
            left, right = max(start, doc["start"]), min(end, doc["end"])
            if left >= right:
                continue
            local_start, local_end = left - doc["start"], right - doc["start"]
            current = self.ops.slice(canonical, left - start, right - start)
            cached = self.ops.slice(self.bank[(doc["key"], layer)]["tensor"], local_start, local_end)
            self._sink(layer, doc, local_start, current)
            for floor in {p.norm_floor for p in self.candidates}:
                with self.allow_sync():
                    errors[(doc["key"], floor)] = (local_start, self.ops.errors(current, cached, floor))
        for part in frame_parts(self.documents, start, end):
            for policy in self.candidates:
                offset, rows = errors[(part["document"]["key"], policy.norm_floor)]
                self.events.append(self._event(frame, part, policy, rows[part["start"] - offset : part["end"] - offset]))

    def _gated(self, frame, hidden):
        layer, start, end = frame.module.layer_idx, frame.marker["start"], frame.marker["end"]
        policy, record = self.policies.get(layer), self.calibrations.get(layer)
        if policy is None or not policy.enabled or record is None:
            return None
        parts = frame_parts(self.documents, start, end)
        eligible = []
        online = set(range(end - start))
        for part in parts:
            key = (part["document"]["key"], layer)
            if not part["eligible"] or key not in self.bank:
                continue
            entry = self.bank[key]
            record.validate(entry["identity"], policy)
            require(entry["complete"] and self.ops.version(entry["tensor"]) == entry["version"], "resident cache changed during request")
            probes, guards = probe_plan(entry["identity"], part["start"], part["end"], policy)
            protected = set(probes) | set(guards)
            candidates = tuple(i for i in range(part["start"], part["end"]) if i not in protected)
            if not candidates:
                continue
            base = part["document"]["start"] - start
            online.difference_update(base + i for i in candidates)
            eligible.append((part, probes, guards, candidates, entry))
        if not eligible:
            return None
        qproj, kvproj = self.projections[layer]
        q = qproj(hidden)
        positions = tuple(sorted(online))
        current = self.ops.normalize(frame.module, kvproj(self.ops.take_hidden(hidden, positions)))
        result = self.ops.allocate(end - start, current)
        self.ops.scatter(result, positions, current)
        # Batch probes into a single comparison/device-to-host decision transfer.
        all_probe_rows, all_cached, spans = [], [], []
        for part, probes, _guards, _candidates, entry in eligible:
            base = part["document"]["start"] - start
            spans.append((len(all_probe_rows), len(probes)))
            all_probe_rows.extend(base + i for i in probes)
            all_cached.append(self.ops.take(entry["tensor"], probes))
        cached_probes = self.ops.allocate(len(all_probe_rows), current)
        cursor = 0
        for value, (_, count) in zip(all_cached, spans, strict=False):
            self.ops.copy(self.ops.slice(cached_probes, cursor, cursor + count), value)
            cursor += count
        with self.allow_sync():
            observed = self.ops.errors(self.ops.take(result, tuple(all_probe_rows)), cached_probes, policy.norm_floor)
        fallback, reused = [], 0
        for (part, _probes, guards, candidates, entry), (offset, count) in zip(eligible, spans, strict=False):
            metrics = observed[offset : offset + count]
            require(metrics and all(math.isfinite(r) and math.isfinite(a) for r, a in metrics), "nonfinite probe")
            relative, absolute = max(v[0] for v in metrics), max(v[1] for v in metrics)
            accepted = relative <= policy.relative_tolerance and absolute <= policy.absolute_tolerance
            base = part["document"]["start"] - start
            target = tuple(base + i for i in candidates)
            if accepted:
                self.ops.scatter(result, target, self.ops.take(entry["tensor"], candidates))
                reused += len(candidates)
            else:
                fallback.extend(target)
            self.events.append(
                dict(
                    layer=layer,
                    document_sha256=part["document"]["key"],
                    document_local_start=part["start"],
                    document_local_end=part["end"],
                    request_id=self.request_id,
                    phase=self.phase,
                    input_sha256=self.input_sha256,
                    candidate_id=digest(asdict(policy)),
                    candidate_sha256=digest(asdict(policy)),
                    accepted=accepted,
                    probe_relative_max=relative,
                    probe_absolute_max=absolute,
                    probe_count=count,
                    mandatory_count=len(guards),
                    candidate_reusable_rows=len(candidates),
                    actual_reused_rows=len(candidates) if accepted else 0,
                    fallback_reason=None if accepted else "probe_threshold_exceeded",
                    oracle_measured=False,
                    current_hybrid_hidden_probes=True,
                )
            )
        if fallback:
            fallback = tuple(sorted(fallback))
            fresh = self.ops.normalize(frame.module, kvproj(self.ops.take_hidden(hidden, fallback)))
            self.ops.scatter(result, fallback, fresh)
        self.ops.validate_tensor(result, end - start)
        counts = self.counts[layer]
        counts["split_q_rows"] += end - start
        counts["online_kv_rows"] += len(positions) + len(fallback)
        counts["reused_kv_rows"] += reused
        counts["probe_rows"] += len(all_probe_rows)
        counts["fallback_rows"] += len(fallback)
        frame.canonical = result
        return q, result

    def finish(self):
        require(self.state == "RUNNING" and _ACTIVE.get() is None, "only successful complete requests may finish")
        try:
            require(all(end == len(self.ids) for end in self.coverage.values()), "incomplete native layer coverage")
            self.ops.synchronize()
            if self.mode == "capture_independent":
                for layer in LAYERS:
                    entry = self.bank[(self.documents[0]["key"], layer)]
                    require(entry["cursor"] == DOC, "incomplete offline document")
                    entry.update(complete=True, version=self.ops.version(entry["tensor"]))
            if self.mode == "split_identity":
                require(
                    bool(self.qualifications) and all(q["exact"] for q in self.qualifications),
                    "missing split/gather numerical qualification",
                )
                self.identity_qualified = True
            self.state = "COMPLETE"
        except BaseException:
            self.state = "ABORTED"
            raise

    def report(self):
        require(self.state == "COMPLETE", "only synchronized complete native reports may publish")
        return dict(
            schema="kv-cache-engram-native-hook-v1",
            complete=True,
            mode=self.mode,
            request_id=self.request_id,
            phase=self.phase,
            input_sha256=self.input_sha256,
            input_tokens=len(self.ids),
            scope=dict(self.scope),
            backend=self.ops.backend,
            native_execution_declared=bool(self.ops.native_execution),
            identity_exact=bool(self.mode == "split_identity" and self.qualifications and all(q["exact"] for q in self.qualifications)),
            # JSON object keys are strings. Canonicalize *before* the enclosing
            # phase is sealed, otherwise numeric sorting (3 before 10) changes
            # to lexical sorting after a JSON round trip and invalidates SHA.
            qualification=self.qualifications,
            coverage={str(layer): end for layer, end in self.coverage.items()},
            counts={str(layer): dict(values) for layer, values in self.counts.items()},
            events=self.events,
            oracle=dict(ORACLE),
            full_q_attention_moe_preserved=True,
            numerical_equivalence_claim=self.mode == "split_identity",
            downstream_accuracy_certified=False,
            calibration_binding_is_not_certificate=True,
            shared_cpu_binding_instrumentation=True,
            standard_gpu_operator_passthrough=self.mode == "native",
            cache_entries=len(self.bank),
            cache_bytes=len(self.bank) * DOC * DIM * 2,
        )

    def iter_cache_entries(self):
        require(self.state == "COMPLETE", "cache export requires completed request")
        for (key, layer), entry in sorted(self.bank.items()):
            require(entry["complete"] and self.ops.version(entry["tensor"]) == entry["version"], "invalid resident cache")
            yield key, layer, entry["tensor"]

    def cache_identities(self):
        require(self.state == "COMPLETE", "identities require completed capture")
        result = {}
        for key, layer, _ in self.iter_cache_entries():
            result.setdefault(layer, self._identity(key, layer))
        require(set(result) == set(LAYERS), "all layer identities required")
        return result

    def clear_cache(self):
        require(self.state in ("IDLE", "COMPLETE"), "cannot clear a live/poisoned cache")
        self.bank.clear()

    def _restore(self):
        for module, name, wrapper in reversed(self.handles):
            require(module.__dict__.get(name) is wrapper, "native hook ownership changed; refusing overwrite")
            delattr(module, name)
        self.handles.clear()
        for module in self.modules.values():
            if getattr(module, "_kv_cache_engram_native_hook", None) is self:
                delattr(module, "_kv_cache_engram_native_hook")

    def close(self):
        require(_ACTIVE.get() is None, "cannot close inside native attention")
        if self.state == "CLOSED":
            return
        self._restore()
        self.bank.clear()
        self.projections.clear()
        self.state = "CLOSED"
