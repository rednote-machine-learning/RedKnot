"""Finite, real native EP qualification for zero-local-token FFN selection.

CPU import/validation never imports Torch. Only run_empty_ep_qualification runs
GPU kernels, under the already authenticated eight-rank native worker. This is
operator qualification, not a quality sample or a timing measurement.
"""
from __future__ import annotations

import hashlib
from types import MethodType, SimpleNamespace

from kv_cache_engram.core import digest, require, validate_digest

LAYERS = tuple(range(3,40))
PATTERNS = ("all_empty", "rank0_only", "rank0_empty", "alternating_empty")
FULL_ROWS = 128
ACTIVE_ROWS = 127
HIDDEN_SIZE = 5120
WORLD_SIZE = 8
NATIVE_SEQUENCE = ["dispatch", "routed_experts", "shared_experts", "combine"]
_MISSING = object()


def selected_count(pattern, rank):
    require(pattern in PATTERNS and type(rank) is int and 0 <= rank < WORLD_SIZE, "invalid empty EP case")
    active = {"all_empty": False, "rank0_only": rank == 0,
              "rank0_empty": rank != 0, "alternating_empty": rank % 2 == 1}[pattern]
    return ACTIVE_ROWS if active else 0


def seal(value):
    require("sha256" not in value, "cannot reseal empty EP proof")
    return dict(value, sha256=digest(value))


def _sealed(value):
    require(isinstance(value, dict) and value.get("sha256") == digest(
        {k: v for k, v in value.items() if k != "sha256"}), "empty EP proof seal mismatch")


def native_abi(layer):
    """Python-only native layout identity, safe to recheck inside forward."""
    module = layer.mlp
    result = dict(layer=layer.layer_idx, module=type(module).__module__, cls=type(module).__qualname__)
    for key in ("hidden_size", "ep_size", "num_physical_experts", "num_physical_topk",
                "max_num_tokens", "max_num_prefilling_tokens", "use_fp8_act", "use_mega_moe"):
        result[key] = getattr(module, key, None)
    weight = module.gate.weight
    result.update(gate_shape=list(weight.shape), gate_dtype=str(weight.dtype))
    require(layer.layer_idx in LAYERS and result["hidden_size"] == HIDDEN_SIZE
            and result["ep_size"] == WORLD_SIZE and result["use_fp8_act"] is True
            and result["use_mega_moe"] is False, "unsupported zero-local-token native MLP ABI")
    require(all(type(result[k]) is int and result[k] > 0 for k in (
        "num_physical_experts", "num_physical_topk", "max_num_tokens", "max_num_prefilling_tokens")),
        "incomplete native EP dimensions")
    return result


def validate_rank_report(value, *, rank, execution_sha256, model_sha256):
    _sealed(value)
    require(value.get("schema") == "index-hit20-empty-ep-rank-v1" and value.get("passed") is True
            and value.get("rank") == rank and value.get("world_size") == WORLD_SIZE
            and value.get("execution_sha256") == execution_sha256
            and value.get("model_sha256") == model_sha256
            and value.get("quality_eligible") is False and value.get("timing_valid") is False,
            "empty EP rank identity mismatch")
    require(value.get("layer_abis") and [a.get("layer") for a in value["layer_abis"]] == list(LAYERS),
            "empty EP layer ABI coverage incomplete")
    cases = value.get("cases")
    expected = [(layer, pattern) for layer in LAYERS for pattern in PATTERNS]
    require(isinstance(cases, list) and [(c.get("layer"), c.get("pattern")) for c in cases] == expected,
            "empty EP layer/pattern coverage incomplete")
    for case in cases:
        count = selected_count(case["pattern"], rank)
        require(case.get("full_rows") == FULL_ROWS and case.get("selected_rows") == count
                and case.get("original_input_nonempty") is True
                and case.get("exact_native_reference") is True and case.get("finite") is True
                and case.get("inactive_rows_exact_zero") is True
                and case.get("output_shape") == [FULL_ROWS, HIDDEN_SIZE]
                and case.get("output_dtype") == "torch.bfloat16"
                and case.get("strict_cuda_sync_debug_mode") == 2
                and case.get("strict_cuda_sync_mode_before_after") == [2, 2, 2, 2]
                and case.get("native_ep_called_when_empty") is (count == 0),
                "empty EP output/strict-call invariant failed")
        for key in ("reference_sha256", "candidate_sha256"):
            validate_digest(case.get(key), "empty EP output digest")
        require(case["reference_sha256"] == case["candidate_sha256"], "empty EP output bytes differ")
        for side in ("reference", "candidate"):
            trace = case.get(side + "_trace", {})
            require(trace.get("sequence") == NATIVE_SEQUENCE,
                    "native EP sequence not executed exactly once")
            received = trace.get("received_tokens")
            require(type(received) is int and received >= 0, "native received-token proof missing")
            if case["pattern"] == "all_empty":
                require(received == 0, "all-empty native EP unexpectedly received tokens")
        require(case["reference_trace"]["received_tokens"] == case["candidate_trace"]["received_tokens"],
                "native reference and candidate routed token counts differ")
    return True


def validate_all_ranks_report(value, *, execution_sha256, model_sha256):
    _sealed(value)
    validate_digest(execution_sha256, "execution identity")
    validate_digest(model_sha256, "model identity")
    require(value.get("schema") == "index-hit20-empty-ep-all-ranks-v1" and value.get("passed") is True
            and value.get("execution_sha256") == execution_sha256
            and value.get("model_sha256") == model_sha256
            and value.get("qualified_layers") == list(LAYERS) and value.get("patterns") == list(PATTERNS)
            and value.get("quality_eligible") is False and value.get("timing_valid") is False,
            "empty EP aggregate identity mismatch")
    reports = value.get("reports")
    require(isinstance(reports, list) and len(reports) == WORLD_SIZE, "all eight empty EP ranks required")
    remote_empty = 0
    remote_by_layer = {str(layer): 0 for layer in LAYERS}
    for rank, report in enumerate(reports):
        validate_rank_report(report, rank=rank, execution_sha256=execution_sha256, model_sha256=model_sha256)
        remote_empty += sum(c["selected_rows"] == 0 and c["candidate_trace"]["received_tokens"] > 0
                            for c in report["cases"] if c["pattern"] != "all_empty")
        for case in report["cases"]:
            if case["pattern"] != "all_empty" and case["selected_rows"] == 0 and case["candidate_trace"]["received_tokens"] > 0:
                remote_by_layer[str(case["layer"])] += 1
    require(remote_empty > 0 and value.get("empty_rank_remote_work_cases") == remote_empty,
            "no proven remote expert work on a zero-local-token rank")
    require(all(remote_by_layer.values()) and value.get("empty_rank_remote_work_by_layer") == remote_by_layer,
            "each granted layer needs proven remote expert work on an empty rank")
    require(value.get("checks") == WORLD_SIZE * len(LAYERS) * len(PATTERNS), "empty EP check count changed")
    return True


def build_all_ranks_report(reports, *, execution_sha256, model_sha256):
    require(isinstance(reports, list) and len(reports) == WORLD_SIZE, "eight native EP reports required")
    for rank, report in enumerate(reports):
        validate_rank_report(report, rank=rank, execution_sha256=execution_sha256, model_sha256=model_sha256)
    remote_empty = sum(c["selected_rows"] == 0 and c["candidate_trace"]["received_tokens"] > 0
                       for report in reports for c in report["cases"] if c["pattern"] != "all_empty")
    remote_by_layer = {str(layer): sum(c["selected_rows"] == 0 and c["candidate_trace"]["received_tokens"] > 0
        for report in reports for c in report["cases"] if c["layer"] == layer and c["pattern"] != "all_empty") for layer in LAYERS}
    value = seal(dict(schema="index-hit20-empty-ep-all-ranks-v1", passed=True,
        execution_sha256=execution_sha256, model_sha256=model_sha256, qualified_layers=list(LAYERS),
        patterns=list(PATTERNS), reports=reports, checks=WORLD_SIZE * len(LAYERS) * len(PATTERNS),
        empty_rank_remote_work_cases=remote_empty, empty_rank_remote_work_by_layer=remote_by_layer,
        quality_eligible=False, timing_valid=False))
    validate_all_ranks_report(value, execution_sha256=execution_sha256, model_sha256=model_sha256)
    return value


class EmptyEPCapability:
    """Permission tied to exact runtime objects, model/config and GPU evidence."""

    def __init__(self, layers, report, *, execution_sha256, model_sha256):
        validate_all_ranks_report(report, execution_sha256=execution_sha256, model_sha256=model_sha256)
        self.execution_sha256, self.model_sha256 = execution_sha256, model_sha256
        self.sha256 = report["sha256"]
        active = tuple(layer for layer in layers if layer.layer_idx in LAYERS)
        require(tuple(x.layer_idx for x in active) == LAYERS, "exact native layer set required")
        self.bindings = {}
        for layer in active:
            abi = native_abi(layer)
            require(all(r["layer_abis"][layer.layer_idx - 3] == abi for r in report["reports"]),
                    "native MLP ABI differs from qualified proof")
            self.bindings[layer.layer_idx] = (layer, layer.mlp, type(layer.mlp).forward, abi)

    def permits(self, layer, metadata, plan):
        bound = self.bindings.get(layer.layer_idx)
        require(bound is not None and bound[0] is layer and bound[1] is layer.mlp
                and bound[2] is type(layer.mlp).forward and bound[3] == native_abi(layer),
                "empty EP capability no longer owns qualified native layer")
        require(metadata.is_prompt is True and metadata.num_reqs == 1
                and metadata.total_q == plan.full_rows and plan.full_rows > 0
                and plan.selected_count == 0,
                "empty EP capability only permits bound zero-selected prompt subsets")
        return True


class _NativeTrace:
    """Owned, non-sync observer around the actual native communication calls."""

    def __init__(self, module):
        self.module, self.handles, self.sequence, self.received = module, [], [], None

    def __enter__(self):
        try:
            for name, label in (("dispatch", "dispatch"), ("routed_experts_forward", "routed_experts"),
                                ("shared_forward", "shared_experts"), ("combine", "combine")):
                require(name not in self.module.__dict__, "foreign native EP observer")
                original = getattr(self.module, name)
                def wrapper(module, *args, _original=original, _label=label, **kwargs):
                    self.sequence.append(_label)
                    if _label == "routed_experts":
                        psum = args[2] if len(args) > 2 else kwargs["psum_num_recv_tokens_per_expert"]
                        require(self.received is None, "repeated native routed expert execution")
                        self.received = psum.clone()
                    return _original(*args, **kwargs)
                bound = MethodType(wrapper, self.module)
                setattr(self.module, name, bound)
                self.handles.append((name, bound))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for name, bound in reversed(self.handles):
            require(self.module.__dict__.get(name) is bound, "native EP observer ownership changed")
            delattr(self.module, name)

    def result(self):
        require(self.sequence == NATIVE_SEQUENCE and self.received is not None,
                "native dispatch/routed/combine sequence missing")
        # Called only outside the strict forward region in allow_cuda_sync.
        values = self.received.detach().cpu().tolist()
        require(isinstance(values, list) and values and all(type(v) is int and v >= 0 for v in values)
                and all(a <= b for a, b in zip(values, values[1:])), "invalid native receive prefix sums")
        return dict(sequence=list(self.sequence), received_tokens=values[-1])


def _metadata(t, device, rows, pattern, layer, rank):
    return SimpleNamespace(is_prompt=True, num_reqs=1, total_q=rows,
        positions=t.arange(rows, dtype=t.int64, device=device), valid_token_mask=None,
        gate_force_random=None, images_seq_mask=None,
        canonical_request_binding=dict(request_id=f"empty-ep:{rank}:{layer}:{pattern}",
            phase="empty_ep_qualification", start=0, end=FULL_ROWS))


def run_empty_ep_qualification(layers, rank, torch_module, allow_sync, *, execution_sha256, model_sha256):
    """All ranks call identically, before installing the production controller.

Use deterministic synthetic BF16 normalized-range values with real native FP8
packing, router weights and EP buffers. No fake token is added to an empty
selected input. Every rank always calls the original native MoE and collectives.
    """
    from kv_cache_engram.index_hit20_adapter import IndexHit20Adapter as FFNOnlyAdapter
    from kv_cache_engram.ffn_only_adapter import TorchFFNOnlyOps, make_selected_mlp_metadata
    from deepinfer.model_executor.layers.linear import get_fp8_cast_params
    from tile_kernels.quant import per_token_cast
    t = torch_module
    require(type(rank) is int and 0 <= rank < WORLD_SIZE, "native EP rank required")
    validate_digest(execution_sha256, "execution identity"); validate_digest(model_sha256, "model identity")
    active = tuple(layer for layer in layers if layer.layer_idx in LAYERS)
    require(tuple(x.layer_idx for x in active) == LAYERS, "native layers 3..39 required")
    ops = TorchFFNOnlyOps(); ops.check_runtime()
    abis, cases = [native_abi(layer) for layer in active], []
    device = active[0].mlp.gate.weight.device
    require(device.type == "cuda", "empty EP qualification must run on actual CUDA")
    with t.inference_mode():
        for layer in active:
            for pattern in PATTERNS:
                count = selected_count(pattern, rank)
                with allow_sync():
                    values = t.arange(FULL_ROWS * HIDDEN_SIZE, device=device, dtype=t.float32).view(FULL_ROWS, HIDDEN_SIZE)
                    bf16 = (values.mul(.013).add(rank + layer.layer_idx * .17).sin()).to(t.bfloat16)
                    packed = per_token_cast(bf16, **get_fp8_cast_params())
                    ids = t.arange(count, dtype=t.int64, device=device)
                    metadata = _metadata(t, device, FULL_ROWS, pattern, layer.layer_idx, rank)
                    plan = SimpleNamespace(full_rows=FULL_ROWS, selected_rows=ids, selected_count=count,
                        current_positions=metadata.positions, request_id=metadata.canonical_request_binding["request_id"],
                        phase="empty_ep_qualification", chunk_start=0, chunk_end=FULL_ROWS,
                        plan_sha256=digest(dict(layer=layer.layer_idx, pattern=pattern, rank=rank, count=count)))
                    selected_metadata = make_selected_mlp_metadata(layer, metadata, plan)
                    reference_bf16 = ops.gather(bf16, ids)
                    # Nonempty reference independently quantizes the chosen
                    # BF16 rows; candidate gathers packed full-input SF rows.
                    reference_packed = (per_token_cast(reference_bf16, **get_fp8_cast_params())
                                        if count else ops.gather(packed, ids))
                    t.cuda.synchronize()
                old_mode = t.cuda.get_sync_debug_mode()
                try:
                    t.cuda.set_sync_debug_mode("error")
                    reference_before = t.cuda.get_sync_debug_mode()
                    require(reference_before == 2, "native reference strict CUDA mode not set")
                    with _NativeTrace(layer.mlp) as reference_trace:
                        reference = layer.mlp(reference_packed, reference_bf16, selected_metadata)
                    reference_after = t.cuda.get_sync_debug_mode()
                    require(reference_after == 2, "native reference disabled CUDA sync checks")
                finally:
                    t.cuda.set_sync_debug_mode(old_mode)
                with allow_sync():
                    t.cuda.synchronize()
                    expected = ops.scatter_zero(reference, bf16, plan)
                    reference_evidence = reference_trace.result()
                    # Build fresh inputs for candidate; native EP has ownership
                    # of its dispatched buffers and may release them in-place.
                    candidate_bf16 = bf16.clone()
                    candidate_packed = per_token_cast(candidate_bf16, **get_fp8_cast_params())
                    t.cuda.synchronize()
                def get_plan(actual_layer, actual_meta, _hidden):
                    require(actual_layer is layer and actual_meta is metadata, "foreign qualification plan")
                    return plan
                def qualification_permission(actual_layer, actual_meta, actual_plan):
                    # This permission lives only inside this one owned test,
                    # not in a production controller or a reusable capability.
                    return (actual_layer is layer and actual_meta is metadata and actual_plan is plan
                            and count == 0 and plan.full_rows == FULL_ROWS
                            and plan.phase == "empty_ep_qualification")
                adapter = FFNOnlyAdapter([layer], get_plan, ops=ops, allow_empty_native=qualification_permission)
                adapter.install()
                old_mode = t.cuda.get_sync_debug_mode()
                try:
                    t.cuda.set_sync_debug_mode("error")
                    candidate_before = t.cuda.get_sync_debug_mode()
                    require(candidate_before == 2, "adapter strict CUDA mode not set")
                    with _NativeTrace(layer.mlp) as candidate_trace:
                        actual = layer.mlp(candidate_packed, candidate_bf16, metadata)
                    candidate_after = t.cuda.get_sync_debug_mode()
                    require(candidate_after == 2, "adapter disabled CUDA sync checks")
                finally:
                    t.cuda.set_sync_debug_mode(old_mode)
                    adapter.close(allow_incomplete=True)
                with allow_sync():
                    t.cuda.synchronize()
                    require(actual.shape == expected.shape == (FULL_ROWS, HIDDEN_SIZE)
                            and actual.dtype == expected.dtype == t.bfloat16
                            and bool(t.isfinite(actual).all().item())
                            and bool((actual[count:] == 0).all().item()), "invalid zero-subset output")
                    reference_hash = hashlib.sha256(expected.contiguous().view(t.uint8).cpu().numpy().tobytes()).hexdigest()
                    candidate_hash = hashlib.sha256(actual.contiguous().view(t.uint8).cpu().numpy().tobytes()).hexdigest()
                    require(reference_hash == candidate_hash, "zero-subset native reference byte parity failed")
                    cases.append(dict(layer=layer.layer_idx, pattern=pattern, full_rows=FULL_ROWS, selected_rows=count,
                        original_input_nonempty=True, exact_native_reference=True, finite=True,
                        inactive_rows_exact_zero=True, output_shape=[FULL_ROWS, HIDDEN_SIZE],
                        output_dtype="torch.bfloat16", strict_cuda_sync_debug_mode=2,
                        strict_cuda_sync_mode_before_after=[reference_before, reference_after, candidate_before, candidate_after],
                        native_ep_called_when_empty=count == 0, reference_sha256=reference_hash,
                        candidate_sha256=candidate_hash, reference_trace=reference_evidence,
                        candidate_trace=candidate_trace.result()))
    value = seal(dict(schema="index-hit20-empty-ep-rank-v1", passed=True, rank=rank, world_size=WORLD_SIZE,
        execution_sha256=execution_sha256, model_sha256=model_sha256, layer_abis=abis,
        cases=cases, quality_eligible=False, timing_valid=False))
    validate_rank_report(value, rank=rank, execution_sha256=execution_sha256, model_sha256=model_sha256)
    return value
