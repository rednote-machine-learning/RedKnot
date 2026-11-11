"""Read-only, clean-input native attention sensitivity diagnostics.

This module never loads a model or imports Torch at import time. It replays only
the pure prefill-attention and output-projection operators, against private KV
copies, and returns the original full-native context to the live forward. The
measured projected attention update is NOT a complete decoder-block hidden
state. These diagnostics are deliberately forbidden on a timed/gated request.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MethodType, SimpleNamespace

from .core import digest, require
from .native_reuse import CHUNK, DIM, DOC, LAYERS, frame_parts


@dataclass(frozen=True)
class SensitivityConfig:
    layers: tuple = LAYERS
    restore_block: bool = True
    restore_half: bool = True
    restore_control: bool = False
    norm_floor: float = 1e-8
    metric_chunk_rows: int = 128
    top_tokens: int = 16
    max_extra_bytes: int = 4 << 30

    def __post_init__(self):
        require(type(self.layers) is tuple and bool(self.layers)
                and self.layers == tuple(sorted(set(self.layers)))
                and set(self.layers) <= set(LAYERS), "unique sorted supported diagnostic layers required")
        require(type(self.restore_block) is bool and type(self.restore_half) is bool
                and type(self.restore_control) is bool
                and (not (self.restore_half or self.restore_control) or self.restore_block),
                "half/control restoration requires block restoration")
        require(type(self.norm_floor) in (int, float) and math.isfinite(self.norm_floor)
                and self.norm_floor > 0, "positive finite diagnostic norm floor required")
        require(type(self.metric_chunk_rows) is int and 1 <= self.metric_chunk_rows <= 256,
                "bounded metric row chunks required")
        require(type(self.top_tokens) is int and 1 <= self.top_tokens <= 64, "bounded top-token count required")
        require(type(self.max_extra_bytes) is int and self.max_extra_bytes > 0,
                "explicit positive diagnostic memory budget required")


def candidate_spans(documents, start, end):
    """Actual reusable source intervals; no template, guard or chunk fragment."""
    spans = []
    for part in frame_parts(documents, start, end):
        if not part["eligible"]:
            continue
        lo, hi = max(part["start"], 128), min(part["end"], DOC-128)
        if lo >= hi:
            continue
        doc = part["document"]
        spans.append(dict(document_sha256=doc["key"], document_local_start=lo,
            document_local_end=hi, input_start=doc["start"]+lo,
            input_end=doc["start"]+hi, block_start=part["start"], block_end=part["end"]))
    return spans


def split_span(span):
    """Split at the original 512-row block's midpoint, preserving guards."""
    middle = (span["block_start"]+span["block_end"])//2
    result = []
    for lo, hi in ((span["document_local_start"], min(span["document_local_end"], middle)),
                   (max(span["document_local_start"], middle), span["document_local_end"])):
        if lo < hi:
            offset = span["input_start"]-span["document_local_start"]
            result.append(dict(span, document_local_start=lo, document_local_end=hi,
                               input_start=offset+lo, input_end=offset+hi))
    return result


def select_worst_span(spans, scores):
    require(len(spans) == len(scores) and bool(spans), "source scores must cover candidate spans")
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in scores),
            "finite nonnegative source scores required")
    return dict(spans[min(range(len(spans)), key=lambda i: (-scores[i], spans[i]["input_start"]))])


def memory_preflight(q, kv, *, config, torch):
    """Conservative tensor reserve, not a guarantee against allocator/kernel OOM.

    Called under the owner's explicit sync allowance. CPU fixtures never query
    CUDA, initialize a device or inspect device inventory. Physical free memory
    deliberately excludes allocator-reserved-but-reusable bytes: refusing a
    marginal diagnostic is preferable to changing the live execution settings.
    """
    contexts = 3*q.numel()*q.element_size()
    kv_copies = 2*kv.numel()*kv.element_size()
    scratch = 512 << 20
    estimate = contexts+kv_copies+scratch
    require(estimate <= config.max_extra_bytes, "diagnostic working-memory budget exceeded")
    free, total = None, None
    if q.is_cuda:
        require(kv.is_cuda and kv.device == q.device, "diagnostic CUDA tensor device mismatch")
        free, total = torch.cuda.mem_get_info(q.device)
        require(type(free) is int and type(total) is int and 0 <= free <= total,
                "invalid physical CUDA free-memory snapshot")
        require(free >= estimate, "insufficient physical CUDA memory for diagnostic reserve")
    return dict(estimated_reserve_bytes=estimate, configured_budget_bytes=config.max_extra_bytes,
        physical_free_bytes=free, physical_total_bytes=total,
        allocator_reserved_bytes_not_counted=True, allocation_success_guaranteed=False)


def tensor_statistics(reference, actual, *, start, config, torch):
    """All-row head/vector error statistics with bounded FP32 working memory.

    Inputs have [rows, heads, features] shape. Every row/head contributes; only
    the top query-token rows are retained in JSON. This never materializes the
    attention probability matrix or stores model-sized reference activations.
    """
    require(tuple(reference.shape) == tuple(actual.shape) and reference.ndim == 3
            and reference.shape[0] > 0 and reference.shape[1] > 0 and reference.shape[2] > 0,
            "paired nonempty three-dimensional diagnostic tensors required")
    require(reference.device == actual.device, "diagnostic devices differ")
    rows, heads, _ = reference.shape
    maximum = [0.] * heads
    absolute = [0.] * heads
    cosine = [0.] * heads
    sums = [0.] * heads
    worst = [start] * heads
    top = []
    blocks = {}
    for lo in range(0, rows, config.metric_chunk_rows):
        hi = min(rows, lo+config.metric_chunk_rows)
        a, b = reference[lo:hi].to(dtype=torch.float32), actual[lo:hi].to(dtype=torch.float32)
        require(bool(torch.isfinite(a).all().item()) and bool(torch.isfinite(b).all().item()),
                "nonfinite paired diagnostic tensor")
        delta = b-a
        na, nb = a.norm(dim=-1), b.norm(dim=-1)
        rel = delta.norm(dim=-1)/na.clamp(min=config.norm_floor)
        abs_error = delta.abs().amax(dim=-1)
        cos = (1-(a*b).sum(dim=-1)/(na*nb).clamp(min=config.norm_floor**2)).clamp(0, 2)
        cos = torch.where((na == 0) & (nb == 0), torch.zeros_like(cos), cos)
        require(bool(torch.isfinite(rel).all().item()) and bool(torch.isfinite(cos).all().item()),
                "nonfinite derived diagnostic metric")
        rmax, ridx = rel.max(dim=0)
        rmax, ridx = rmax.cpu().tolist(), ridx.cpu().tolist()
        amax, cmax = abs_error.amax(dim=0).cpu().tolist(), cos.amax(dim=0).cpu().tolist()
        rsum = rel.to(dtype=torch.float64).sum(dim=0).cpu().tolist()
        for head in range(heads):
            if rmax[head] > maximum[head]:
                maximum[head], worst[head] = rmax[head], start+lo+ridx[head]
            absolute[head] = max(absolute[head], amax[head])
            cosine[head] = max(cosine[head], cmax[head])
            sums[head] += rsum[head]
        # Query-position diagnostics are not labels for the source KV token.
        values = rel.amax(dim=1)
        abs_rows = abs_error.amax(dim=1)
        for index, (r, a) in enumerate(zip(values.cpu().tolist(), abs_rows.cpu().tolist())):
            position = start+lo+index
            key = position//512*512
            block = blocks.setdefault(key, dict(input_start=position, input_end=position,
                relative_l2_max=0., absolute_max=0., relative_l2_sum=0., rows=0))
            block["input_end"] = position+1
            block["relative_l2_max"] = max(block["relative_l2_max"], r)
            block["absolute_max"] = max(block["absolute_max"], a)
            block["relative_l2_sum"] += r
            block["rows"] += 1
        count = min(config.top_tokens, hi-lo)
        _, indices = torch.topk(values, count, largest=True, sorted=False)
        for index in indices.cpu().tolist():
            top.append(dict(token_index=start+lo+index,
                relative_l2_max=float(values[index].item()),
                absolute_max=float(abs_error[index].max().item())))
        top = sorted(top, key=lambda x: (-x["relative_l2_max"], x["token_index"]))[:config.top_tokens]
    result = [dict(head=head, rows=rows, relative_l2_max=maximum[head],
        absolute_max=absolute[head], relative_l2_mean=sums[head]/rows,
        cosine_distance_max=cosine[head], worst_token_index=worst[head]) for head in range(heads)]
    for block in blocks.values():
        block["relative_l2_mean"] = block.pop("relative_l2_sum")/block["rows"]
    return result, top, list(blocks.values())


class NativeSensitivityObserver:
    """Owned diagnostic hooks over a separately installed KVCacheEngramHook.

    reset() is explicit for each full-native/shadow request. All online quality
    and TTFT phases must leave this observer inactive. No diagnostic result is
    used to change the current forward or select cached rows in that request.
    """
    def __init__(self, hook, config=None):
        self.hook, self.config = hook, config or SensitivityConfig()
        self.handles, self.active, self.closed = [], False, False
        self.coverage, self.records, self.head_count, self.memory_checks = {}, [], None, []

    def install(self):
        require(not self.closed and not self.handles, "diagnostic observer already installed/closed")
        try:
            for layer in self.config.layers:
                module = self.hook.modules[layer]
                name = "_prefill_attention"
                require(name not in module.__dict__, "foreign prefill attention diagnostic hook")
                original = getattr(module, name)
                def wrapped(mod, *args, _original=original, **kwargs):
                    return self._attention(mod, _original, *args, **kwargs)
                wrapper = MethodType(wrapped, module)
                setattr(module, name, wrapper)
                self.handles.append((module, name, wrapper))
            return self
        except BaseException:
            self.close()
            raise

    def reset(self):
        require(self.handles and not self.closed and not self.active, "observer is not idle")
        require(self.hook.state == "RUNNING" and self.hook.mode in ("native", "shadow_full")
                and self.hook.identity_qualified, "diagnostics require qualified clean full-native inputs")
        self.request_id, self.phase = self.hook.request_id, self.hook.phase
        self.input_sha256, self.input_tokens = self.hook.input_sha256, len(self.hook.ids)
        self.coverage = {layer: 0 for layer in self.config.layers}
        self.records, self.head_count, self.active, self.memory_checks = [], None, True, []

    def _attention(self, module, original, q, kv, kv_ca_index, kv_swa_full_index,
                   total_k_swa_cached, total_k_swa_cur, metadata):
        if not self.active:
            return original(q, kv, kv_ca_index, kv_swa_full_index,
                            total_k_swa_cached, total_k_swa_cur, metadata)
        hook = self.hook
        frame = hook._frame(module)
        if frame is None:
            # Only authenticated real-prefill calls contribute, never EP placeholders/decode.
            return original(q, kv, kv_ca_index, kv_swa_full_index,
                            total_k_swa_cached, total_k_swa_cur, metadata)
        require(hook.mode in ("native", "shadow_full") and hook.request_id == self.request_id
                and hook.phase == self.phase and hook.input_sha256 == self.input_sha256,
                "diagnostic request changed or contains prior approximate KV")
        start, end = frame.marker["start"], frame.marker["end"]
        layer, n = module.layer_idx, end-start
        require(start == self.coverage[layer] and 0 < n <= CHUNK
                and total_k_swa_cur == n and metadata.total_q == n,
                "diagnostic full current-chunk coverage mismatch")
        require(not module.use_fused_norm_rope_attn_rope_cast
                and getattr(metadata, "redknot_binding", None) is None
                and getattr(metadata, "head_intervention_binding", None) is None
                and getattr(metadata, "attention_range_binding", None) is None,
                "diagnostic requires unmodified unfused native attention")
        t = hook.ops.torch
        require(kv.ndim == 3 and tuple(kv.shape[1:]) == (1, DIM)
                and q.ndim == 3 and q.shape[0] == n and q.shape[2] == DIM,
                "diagnostic native SWA and query geometry changed")
        heads = q.shape[1]
        if self.head_count is None:
            self.head_count = heads
        require(heads == self.head_count, "diagnostic head count changed")
        require(metadata is frame.metadata and type(total_k_swa_cached) is int
                and total_k_swa_cached == 128 and kv.shape[0] >= total_k_swa_cached+n,
                "diagnostic metadata or native cached/current SWA layout changed")
        with hook.allow_sync():
            memory = memory_preflight(q, kv, config=self.config, torch=t)
        self.memory_checks.append(dict(memory, layer=layer, start=start, end=end))
        call = lambda values: original(q, values, kv_ca_index, kv_swa_full_index,
                                       total_k_swa_cached, total_k_swa_cur, metadata)
        baseline = call(kv)
        require(isinstance(baseline, t.Tensor) and tuple(baseline.shape) == tuple(q.shape),
                "native context must retain every query head")
        spans = candidate_spans(hook.documents, start, end)
        with hook.allow_sync():
            baseline_update = module._wv_and_o_proj(baseline.clone(), metadata.positions).clone()
        require(isinstance(baseline_update, t.Tensor) and baseline_update.ndim == 2
                and baseline_update.shape[0] == n, "projected attention update geometry changed")
        hybrid = kv.clone()
        for span in spans:
            entry = hook.bank[(span["document_sha256"], layer)]
            require(entry["complete"] and hook.ops.version(entry["tensor"]) == entry["version"],
                    "diagnostic resident bank is incomplete or mutated")
            lo, hi = span["document_local_start"], span["document_local_end"]
            row_start = span["input_start"]-start
            partial = SimpleNamespace(total_q=hi-lo, positions=metadata.positions[row_start:row_start+hi-lo])
            restored, waiter = hook.ops.postprocess(module, entry["tensor"][lo:hi], partial)
            require(waiter is None and tuple(restored.shape) == (hi-lo, 1, DIM),
                    "diagnostic postprocess geometry/CP changed")
            hybrid[total_k_swa_cached+row_start:total_k_swa_cached+row_start+hi-lo].copy_(restored)
        total_reused = sum(s["input_end"]-s["input_start"] for s in spans)
        def score(span):
            lo = total_k_swa_cached+span["input_start"]-start
            hi = total_k_swa_cached+span["input_end"]-start
            with hook.allow_sync():
                a, b = kv[lo:hi].float(), hybrid[lo:hi].float()
                result = ((b-a).norm(dim=-1)/a.norm(dim=-1).clamp(min=self.config.norm_floor)).max().item()
            require(math.isfinite(result) and result >= 0, "nonfinite source KV-difference ranking score")
            return float(result)
        scores = [score(s) for s in spans]
        source_scores = [dict(input_start=s["input_start"], input_end=s["input_end"], relative_l2_max=v)
                         for s, v in zip(spans, scores)]
        unpatched_error = None
        def compare(operation, values, restored_spans, label):
            nonlocal unpatched_error
            context = baseline if total_reused == 0 else call(values)
            with hook.allow_sync():
                head_metrics, head_top, _ = tensor_statistics(baseline, context, start=start,
                    config=self.config, torch=t)
                update = (baseline_update if total_reused == 0 else
                          module._wv_and_o_proj(context.clone(), metadata.positions))
                update_metrics, update_top, query_blocks = tensor_statistics(baseline_update.unsqueeze(1), update.unsqueeze(1),
                    start=start, config=self.config, torch=t)
            metric = dict(update_metrics[0])
            del metric["head"]
            if operation == "reuse_current_chunk":
                unpatched_error = metric["relative_l2_max"]
            reduction = unpatched_error-metric["relative_l2_max"]
            restored_rows = sum(s["input_end"]-s["input_start"] for s in restored_spans)
            self.records.append(dict(layer=layer, start=start, end=end, operation=operation,
                source_spans=[dict(s) for s in spans], restored_spans=[dict(s) for s in restored_spans],
                source_block_scores=source_scores,
                restoration_label=label,
                source_rows=total_reused, reused_rows=total_reused-restored_rows, restored_rows=restored_rows,
                attention_update=metric, attention_update_top_query_tokens=update_top,
                attention_update_query_blocks=query_blocks,
                error_reduction_relative_l2_max=reduction,
                error_reduction_fraction=(reduction/unpatched_error if unpatched_error > 0 else None),
                heads=head_metrics, head_top_query_tokens=head_top, head_count=heads,
                same_clean_input=True, intervention_scope="current_chunk_swa_only",
                source_selection="maximum post-RoPE/cast KV relative L2; earliest source position breaks ties",
                full_block_hidden_measured=False, diagnostic_only=True, timing_eligible=False))
        compare("reuse_current_chunk", hybrid, [], "all_cached")
        if spans and self.config.restore_block:
            selected = select_worst_span(spans, scores)
            def repair(span):
                value = hybrid.clone()
                lo = total_k_swa_cached+span["input_start"]-start
                hi = total_k_swa_cached+span["input_end"]-start
                value[lo:hi].copy_(kv[lo:hi])
                return value
            compare("restore_block", repair(selected), [selected], "high_error_512")
            if self.config.restore_half:
                halves = split_span(selected)
                for half in halves:
                    label = ("left_half_256" if half["document_local_start"] <
                             (half["block_start"]+half["block_end"])//2 else "right_half_256")
                    compare("restore_half", repair(half), [half], label)
            if self.config.restore_control and len(spans) > 1:
                # Spatial control selected without inspecting error or the gold answer.
                control = min((s for s in spans if s != selected), key=lambda s: s["input_start"])
                compare("restore_block", repair(control), [control], "control_512")
        self.coverage[layer] = end
        return baseline

    def finish(self):
        require(self.active and self.hook.state == "COMPLETE", "complete native request required before diagnostics close")
        require(all(n == self.input_tokens for n in self.coverage.values()) and self.head_count is not None,
                "incomplete sensitivity layer/token coverage")
        result = dict(schema="kv-cache-engram-native-sensitivity-v1", complete=True,
            request_id=self.request_id, phase=self.phase, input_sha256=self.input_sha256,
            input_tokens=self.input_tokens, selected_layers=list(self.config.layers),
            operator_backend=self.hook.ops.backend, native_execution=self.hook.ops.native_execution,
            coverage={str(k): v for k, v in self.coverage.items()}, head_count=self.head_count,
            records=self.records, memory_preflights=self.memory_checks,
            source_token_vs_query_token_distinguished=True,
            downstream_native_unchanged=True, diagnostic_only=True, timing_eligible=False,
            full_block_hidden_measured=False, preserves_native_cached_prefix_and_global_kv=True,
            interpretation="single-layer clean-input current-chunk SWA intervention; projected attention update, not decoder-block hidden",
            diagnostics_do_not_authorize_runtime_reuse=True)
        result["sha256"] = digest(result)
        self.active = False
        return result

    def close(self):
        changed = []
        for module, name, wrapper in reversed(self.handles):
            if module.__dict__.get(name) is wrapper:
                delattr(module, name)
            else:
                changed.append((module.layer_idx, name))
        self.handles.clear()
        self.closed, self.active = True, False
        require(not changed, "diagnostic hook ownership changed; foreign methods were not overwritten")
