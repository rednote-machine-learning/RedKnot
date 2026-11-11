"""Isolated A/B/C causal SWA-KV diagnostic; never an online admission policy.

A captures a bounded, immutable CPU bank of full-native prepared KV rows. B
forces cached KV into the same eligible current-chunk rows. C replaces one
preselected source block with A's already-positioned values. B/C return the
intervened context, so later native layers really consume the intervention.
All Q/KV projections, attention, residuals and FFN remain computed normally.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MethodType, SimpleNamespace

from .core import digest, require
from .native_reuse import CHUNK, DIM
from .sensitivity import SensitivityConfig, candidate_spans, memory_preflight, select_worst_span

ORACLE_LAYERS = (6, 7, 9, 10, 11, 12, 13, 16, 17, 18)
PROTECTED_LAYERS = (0, 1, 2, 3, 4, 5, 15)


@dataclass(frozen=True)
class OracleRepairConfig:
    """Fixed scientific choices; resource limits do not change those choices."""
    max_cpu_reference_bytes: int = 256 << 20
    max_extra_bytes: int = 4 << 30

    def __post_init__(self):
        require(type(self.max_cpu_reference_bytes) is int and self.max_cpu_reference_bytes > 0,
                "positive CPU oracle-bank limit required")
        require(type(self.max_extra_bytes) is int and self.max_extra_bytes > 0,
                "positive diagnostic reserve limit required")


def oracle_spans(documents, start, end):
    """B/C identical candidates, including native carry-tail protection."""
    spans = []
    for source in candidate_spans(documents, start, end):
        stop = min(source["input_end"], end-128)
        if source["input_start"] < stop:
            spans.append(dict(source, input_end=stop,
                document_local_end=source["document_local_start"]+stop-source["input_start"]))
    return spans


def _sealed(value):
    return dict(value, sha256=digest(value))


def _copy_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _tensor_digest(value, torch):
    """Hash exact contiguous CPU bytes without requiring NumPy/pickle."""
    require(value.device.type == "cpu" and value.is_contiguous(), "CPU contiguous oracle reference required")
    raw = value.view(torch.uint8).reshape(-1)
    checksum = hashlib.sha256()
    # Bounded Python conversion; at most 64 KiB of byte values per iteration.
    for start in range(0, raw.numel(), 65536):
        checksum.update(bytes(raw[start:start+65536].tolist()))
    return checksum.hexdigest()


class OracleRepairController:
    """Owned hooks for one A, then one B and C over the same frozen input.

    Install after KVCacheEngramHook. For each arm call hook.reset(...), reset(),
    execute the complete request, hook.finish(), then finish(). export_plan()
    is available only after complete A. close() restores owned instance methods.
    The private CPU bank is neither a persistent cache nor a calibrated gate.
    """
    def __init__(self, hook, config=None):
        self.hook, self.config = hook, config or OracleRepairConfig()
        self.handles, self.active, self.closed, self.failed = [], False, False, False
        self.plan, self.references, self.reference_bytes = None, {}, 0
        self.completed_modes, self.capture_records = set(), []

    def install(self):
        require(not self.handles and not self.closed, "oracle controller already installed/closed")
        try:
            for layer in ORACLE_LAYERS:
                module = self.hook.modules[layer]
                require("_prefill_attention" not in module.__dict__, "foreign prefill attention hook")
                original = module._prefill_attention
                def wrapped(mod, *args, _original=original, **kwargs):
                    try:
                        return self._attention(mod, _original, *args, **kwargs)
                    except BaseException:
                        self.failed = True
                        raise
                wrapper = MethodType(wrapped, module)
                setattr(module, "_prefill_attention", wrapper)
                self.handles.append((module, wrapper))
            return self
        except BaseException:
            self.close()
            raise

    def reset(self, mode):
        require(self.handles and not self.active and not self.closed and not self.failed,
                "oracle controller must be installed, idle and healthy")
        require(mode in ("capture", "reuse", "repair") and mode not in self.completed_modes,
                "each fixed diagnostic arm can run once")
        hook = self.hook
        require(hook.state == "RUNNING" and hook.mode in ("native", "shadow_full")
                and hook.identity_qualified, "oracle arms require qualified full-projection native hooks")
        require(hook.mode != "gated", "formal runtime gate may not be modified by this diagnostic")
        binding = dict(input_sha256=hook.input_sha256, input_tokens=len(hook.ids),
                       documents_sha256=digest(hook.documents))
        if mode == "capture":
            require(self.plan is None and not self.references and not self.capture_records,
                    "oracle A may not overwrite previous evidence")
            self.binding = binding
        else:
            require(self.plan is not None and "capture" in self.completed_modes
                    and binding == self.binding, "complete same-input A required before B/C")
            self._verify_references()
        self.mode, self.request_id, self.phase = mode, hook.request_id, hook.phase
        self.coverage = {layer: 0 for layer in ORACLE_LAYERS}
        self.records, self.memory_checks = [], []
        self.counts = {layer: dict(native_q_rows=0, native_kv_projection_rows=0,
            source_rows=0, injected_rows=0, oracle_restored_rows=0, reused_rows=0) for layer in ORACLE_LAYERS}
        self.active = True

    def _verify_references(self):
        require(self.plan is not None and self.plan["sha256"] == digest(
            {k: v for k, v in self.plan.items() if k != "sha256"}), "oracle plan was mutated")
        t = self.hook.ops.torch
        expected = {(r["layer"], r["start"], r["end"]): r for r in self.plan["records"]
                    if r["selected_span"] is not None}
        require(set(self.references) == set(expected), "oracle CPU reference coverage changed")
        observed_bytes = 0
        with self.hook.allow_sync():
            for key, entry in self.references.items():
                value = entry["tensor"]
                record = expected[key]
                size = value.numel()*value.element_size()
                require(value.device.type == "cpu" and list(value.shape) == record["oracle_reference_shape"]
                        and str(value.dtype) == record["oracle_reference_dtype"]
                        and size == record["oracle_reference_bytes"]
                        and entry["sha256"] == record["oracle_reference_sha256"]
                        and value._version == entry["version"] and _tensor_digest(value, t) == entry["sha256"],
                        "immutable CPU oracle reference was mutated")
                observed_bytes += size
        require(observed_bytes == self.reference_bytes == self.plan["cpu_reference_bytes"],
                "oracle CPU reference byte accounting changed")

    def _attention(self, module, original, q, kv, kv_ca_index, kv_swa_full_index,
                   total_k_swa_cached, total_k_swa_cur, metadata):
        call = lambda values: original(q, values, kv_ca_index, kv_swa_full_index,
                                       total_k_swa_cached, total_k_swa_cur, metadata)
        if not self.active:
            return call(kv)
        hook, t = self.hook, self.hook.ops.torch
        frame = hook._frame(module)
        if frame is None:  # Decode and EP placeholders have no real-prefill binding.
            return call(kv)
        require(hook.mode in ("native", "shadow_full") and hook.request_id == self.request_id
                and hook.phase == self.phase and hook.input_sha256 == self.binding["input_sha256"]
                and digest(hook.documents) == self.binding["documents_sha256"],
                "oracle arm/request/input changed")
        layer, start, end = module.layer_idx, frame.marker["start"], frame.marker["end"]
        n = end-start
        require(layer in ORACLE_LAYERS and start == self.coverage[layer]
                and 0 < n <= CHUNK and metadata is frame.metadata
                and metadata.canonical_request_binding == frame.marker
                and metadata.total_q == n and total_k_swa_cur == n
                and type(total_k_swa_cached) is int and total_k_swa_cached == 128,
                "oracle native chunk/binding/carry geometry mismatch")
        require(not module.use_fused_norm_rope_attn_rope_cast,
                "oracle diagnostic requires unfused native prefill")
        for name in ("redknot_binding", "head_intervention_binding", "attention_range_binding", "sparse_phase"):
            require(getattr(metadata, name, None) is None, "foreign oracle intervention: "+name)
        require(kv.ndim == 3 and tuple(kv.shape[1:]) == (1, DIM)
                and kv.shape[0] >= 128+n and q.ndim == 3 and q.shape[0] == n
                and q.shape[2] == DIM and q.device == kv.device,
                "oracle native Q/KV geometry changed")
        if hook.ops.native_execution:
            require(hook.ops.PG.context_parallel_group().size() == 1
                    and not t.cuda.is_current_stream_capturing()
                    and q.is_cuda and kv.is_cuda and q.dtype == kv.dtype == t.bfloat16
                    and metadata.is_prompt and metadata.num_reqs == 1
                    and kv_swa_full_index is metadata.kv_swa_full_index,
                    "oracle diagnostic requires authentic CP1 native BF16 full view")
            require(module.use_swa and module.has_packed_kv and module.is_sparse and module.use_shared_kv
                    and not module.is_yoco_hidden_consumer_layer and not module.yoco_swa_share_hidden
                    and module.qk_rope_head_dim == 64 and module.window_size == 128
                    and not module.is_kv_source and not module.is_index_source
                    and not module.is_index_k_source and not module.has_local_index
                    and module.current_hidden_proj is None and module.shared_hidden_kv_proj is None,
                    "oracle diagnostic requires original full-view KV/index consumer layers")
            hook.ops.validate_full_view(metadata, frame.marker)
            require(metadata.positions.ndim == 1 and metadata.positions.numel() == n
                    and metadata.positions.dtype in (t.int32, t.int64)
                    and metadata.positions.device == q.device, "oracle native absolute positions required")
            t._assert_async((metadata.positions == t.arange(start, end,
                device=q.device, dtype=metadata.positions.dtype)).all(), "oracle absolute positions changed")
        with hook.allow_sync():
            memory = memory_preflight(q, kv, config=SensitivityConfig(layers=ORACLE_LAYERS,
                max_extra_bytes=self.config.max_extra_bytes), torch=t)
        self.memory_checks.append(dict(memory, layer=layer, start=start, end=end))
        spans = oracle_spans(hook.documents, start, end)
        source_rows, span_sha = sum(s["input_end"]-s["input_start"] for s in spans), digest(spans)
        key = layer, start, end
        frozen = None
        if self.mode != "capture":
            frozen = self.plan_by_chunk.get(key)
            require(frozen is not None and frozen["source_spans"] == spans
                    and frozen["source_spans_sha256"] == span_sha,
                    "B/C candidates or chunk boundaries differ from A")
        hybrid = kv.clone() if spans else kv
        for span in spans:
            entry = hook.bank[(span["document_sha256"], layer)]
            require(entry["complete"] and hook.ops.version(entry["tensor"]) == entry["version"],
                    "offline canonical KV is incomplete/mutated")
            lo, hi = span["document_local_start"], span["document_local_end"]
            local = span["input_start"]-start
            partial = SimpleNamespace(total_q=hi-lo, positions=metadata.positions[local:local+hi-lo])
            cached, waiter = hook.ops.postprocess(module, entry["tensor"][lo:hi], partial)
            require(waiter is None and tuple(cached.shape) == (hi-lo, 1, DIM)
                    and cached.dtype == kv.dtype and cached.device == kv.device,
                    "cached KV postprocess geometry/dtype/device changed")
            hybrid[128+local:128+local+hi-lo].copy_(cached)
        if self.mode == "capture":
            scores = []
            with hook.allow_sync():
                for span in spans:
                    lo, hi = 128+span["input_start"]-start, 128+span["input_end"]-start
                    a, b = kv[lo:hi].float(), hybrid[lo:hi].float()
                    score = ((b-a).norm(dim=-1)/a.norm(dim=-1).clamp(min=1e-8)).max().item()
                    require(math.isfinite(score) and score >= 0, "nonfinite source KV ranking score")
                    scores.append(float(score))
            selected = select_worst_span(spans, scores) if spans else None
            frozen = dict(layer=layer, start=start, end=end, source_spans=spans,
                source_spans_sha256=span_sha, source_rows=source_rows,
                source_block_scores=[dict(input_start=s["input_start"], input_end=s["input_end"],
                    relative_l2_max=score) for s, score in zip(spans, scores)], selected_span=selected,
                oracle_reference_sha256=None, oracle_reference_shape=None,
                oracle_reference_dtype=None, oracle_reference_bytes=0)
            if selected is not None:
                lo, hi = 128+selected["input_start"]-start, 128+selected["input_end"]-start
                size = (hi-lo)*DIM*kv.element_size()
                require(self.reference_bytes+size <= self.config.max_cpu_reference_bytes,
                        "bounded oracle CPU-reference budget exceeded")
                with hook.allow_sync(), t.inference_mode(False), t.no_grad():
                    value = kv[lo:hi].detach().to(device="cpu", copy=True).contiguous()
                    checksum = _tensor_digest(value, t)
                require(not t.is_inference(value), "oracle CPU reference must have a version guard")
                self.references[key] = dict(tensor=value, version=value._version, sha256=checksum)
                self.reference_bytes += size
                frozen.update(oracle_reference_sha256=checksum, oracle_reference_shape=list(value.shape),
                              oracle_reference_dtype=str(value.dtype), oracle_reference_bytes=size)
            self.capture_records.append(frozen)
        selected = frozen["selected_span"]
        restored_rows = 0
        if self.mode == "repair" and selected is not None:
            reference = self.references[key]
            value = reference["tensor"]
            require(value._version == reference["version"]
                    and reference["sha256"] == frozen["oracle_reference_sha256"],
                    "oracle reference changed before repair")
            lo, hi = 128+selected["input_start"]-start, 128+selected["input_end"]-start
            # A stores already normalized, positioned and quantized values.
            # Never call postprocess/RoPE on these reference rows a second time.
            hybrid[lo:hi].copy_(value.to(device=kv.device, dtype=kv.dtype))
            restored_rows = hi-lo
        injected = source_rows if self.mode != "capture" else 0
        self.records.append(dict(layer=layer, start=start, end=end, source_spans_sha256=span_sha,
            source_rows=source_rows, injected_rows=injected, oracle_restored_rows=restored_rows,
            reused_rows=injected-restored_rows, selected_span=selected,
            oracle_reference_sha256=frozen["oracle_reference_sha256"]))
        counts = self.counts[layer]
        for name, count in (("native_q_rows", n), ("native_kv_projection_rows", n),
                            ("source_rows", source_rows), ("injected_rows", injected),
                            ("oracle_restored_rows", restored_rows), ("reused_rows", injected-restored_rows)):
            counts[name] += count
        result = call(kv if self.mode == "capture" else hybrid)
        require(isinstance(result, t.Tensor) and tuple(result.shape) == tuple(q.shape),
                "native prefill must return all query/head contexts")
        self.coverage[layer] = end
        return result

    def _flags(self):
        return dict(oracle_diagnostic=True, no_ttft_claim=True, all_q_native=True,
            all_kv_projections_native=True, all_ffn_native=True, global_kv_index_native=True,
            head_reuse=False, native_chunk_tail_guard=128, document_head_tail_guard=128,
            protected_layers=list(PROTECTED_LAYERS), formal_admission_unchanged=True,
            native_execution=self.hook.ops.native_execution, operator_backend=self.hook.ops.backend)

    def finish(self):
        require(self.active and not self.failed and self.hook.state == "COMPLETE"
                and all(n == self.binding["input_tokens"] for n in self.coverage.values()),
                "complete all-layer A/B/C native request required")
        coverage = {str(layer): self.coverage[layer] for layer in ORACLE_LAYERS}
        if self.mode == "capture":
            self.plan = _sealed(dict(schema="kv-cache-engram-oracle-plan-v1", complete=True,
                **self.binding, capture_request_id=self.request_id, capture_phase=self.phase,
                layers=list(ORACLE_LAYERS), coverage=coverage, records=self.capture_records,
                cpu_reference_bytes=self.reference_bytes, max_cpu_reference_bytes=self.config.max_cpu_reference_bytes,
                selection="max post-RoPE/cast relative L2; earliest source position breaks ties",
                norm_floor=1e-8, **self._flags()))
            self.plan_by_chunk = {(r["layer"], r["start"], r["end"]): r for r in self.capture_records}
            self._verify_references()
        report = _sealed(dict(schema="kv-cache-engram-oracle-arm-v1", complete=True,
            mode=self.mode, request_id=self.request_id, phase=self.phase, **self.binding,
            layers=list(ORACLE_LAYERS), coverage=coverage, plan_sha256=self.plan["sha256"],
            records=self.records, counts={str(k): dict(v) for k, v in self.counts.items()},
            memory_preflights=self.memory_checks, forced_unqualified_reuse=self.mode != "capture",
            propagates_to_later_layers=self.mode != "capture", **self._flags()))
        self.completed_modes.add(self.mode)
        self.active = False
        return _copy_json(report)

    def export_plan(self):
        require(self.plan is not None and "capture" in self.completed_modes and not self.failed,
                "complete frozen A plan required")
        self._verify_references()
        return _copy_json(self.plan)

    def close(self):
        changed = []
        for module, wrapper in reversed(self.handles):
            if module.__dict__.get("_prefill_attention") is wrapper:
                delattr(module, "_prefill_attention")
            else:
                changed.append(module.layer_idx)
        self.handles.clear()
        self.references.clear()
        self.closed, self.active = True, False
        require(not changed, "foreign oracle hooks were not overwritten")
