#!/usr/bin/env python3
"""End-to-end Qwen3 dense-cache versus managed-paged-cache validation.

Examples (run from the repository root)::

    python test/srt/redknot/validate_head_kv_qwen3.py --tiny --device cpu
    python test/srt/redknot/validate_head_kv_qwen3.py \
        --model-path /path/to/Qwen3-8B --device cuda --dtype bfloat16 \
        --output /path/to/results.json

The model's projections, Q/K norms, RoPE, output projections, MLPs, residuals,
embedding and LM head use the same weights in both runs. Only attention/cache
execution changes. Dense baseline caches are released before the managed run.
The managed forward has use_cache=False, never constructs a Hugging Face Cache,
and stores history only in HeadPagePool. Its temporary K/V projections cover
the *current* input chunk. Integrity checks export one page at a time to CPU;
they are outside timed model forwards and do not feed attention.

This is a batch-one correctness experiment, not a production scheduler adapter,
an approximate RedKnot head-policy quality experiment, or a throughput benchmark.
The temporary instance-level forward bindings are restored in all exit paths.
The supported Transformers source contract is exactly version 4.57.1.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
import types

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)


_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "python/sglang/srt/mem_cache"))
from head_kv.manager import HeadKVManager, SegmentWrite  # noqa: E402
from head_kv.attention import paged_attention  # noqa: E402
from head_kv.pool import HeadPagePool  # noqa: E402
from head_kv.segpaged import ManagedSegPagedKVCache  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tokens(start, count, vocab_size):
    return [3 + ((start + 29 * i) % (vocab_size - 3)) for i in range(count)]


def _timed_forward(model, tokens, device, **kwargs):
    inputs = torch.tensor([tokens], device=device, dtype=torch.long)
    _sync(device)
    started = time.perf_counter()
    output = model(input_ids=inputs, return_dict=True, **kwargs)
    _sync(device)
    elapsed = time.perf_counter() - started
    # Export is outside the model-forward time; all input-position logits are
    # compared, including the prefill chunk, rather than only its final row.
    logits = output.logits.detach().float().cpu()
    return output, logits, elapsed


def _dense_baseline(model, prompt, common, branches, device, attention_probe=None):
    def run(stage, tokens, **kwargs):
        if attention_probe is not None:
            attention_probe.stage = stage
        return _timed_forward(model, tokens, device, **kwargs)

    results = {}
    output, logits, elapsed = run("prefill", prompt, use_cache=True)
    cache = output.past_key_values
    results["prefill"] = dict(logits=logits, seconds=elapsed, input_tokens=prompt)
    for i, token in enumerate(common):
        output, logits, elapsed = run(f"shared_decode_{i}", [token], use_cache=True, past_key_values=cache)
        cache = output.past_key_values
        results[f"shared_decode_{i}"] = dict(logits=logits, seconds=elapsed, input_tokens=[token])
    _sync(device)
    started = time.perf_counter()
    fork_caches = {name: copy.deepcopy(cache) for name in branches}
    _sync(device)
    fork_seconds = time.perf_counter() - started
    for name, continuation in branches.items():
        branch_cache = fork_caches[name]
        for i, token in enumerate(continuation):
            output, logits, elapsed = run(
                f"{name}_decode_{i}", [token], use_cache=True, past_key_values=branch_cache,
            )
            branch_cache = output.past_key_values
            results[f"{name}_decode_{i}"] = dict(logits=logits, seconds=elapsed, input_tokens=[token])
    # No GPU cache/output escapes this function. Only CPU logits remain.
    return results, fork_seconds


def _dense_accum_attention(attention, query, key, value, attention_mask, *, accumulation_dtype=torch.float32):
    """Dense mathematical oracle; quantize only its final attention output.

    BF16/FP16 Q/K/V remain exactly the model's projected values. QK matmul,
    scaling, softmax and PV matmul use accumulation_dtype, unlike HF eager's
    low-precision QK output/scaling and low-precision probability tensor.
    """
    repeated_k = repeat_kv(key, attention.num_key_value_groups).to(accumulation_dtype)
    repeated_v = repeat_kv(value, attention.num_key_value_groups).to(accumulation_dtype)
    scores = torch.matmul(query.to(accumulation_dtype), repeated_k.transpose(2, 3)) * attention.scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key.shape[-2]].to(accumulation_dtype)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, repeated_v).to(query.dtype)
    return output.transpose(1, 2).contiguous(), None


def _tensor_error(left, right):
    delta = left.float() - right.float()
    return dict(max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
                relative_l2=float(torch.linalg.vector_norm(delta) /
                                  torch.linalg.vector_norm(right.float()).clamp_min(1e-12)),
                exact_element_fraction=float((left == right).float().mean()))


class _DenseQwen3AttentionProbe:
    """Opt-in dense baseline/precision probe; never executes in managed mode.

    No global Transformers attention registry is modified. When a probe is
    enabled, its temporary paged copy belongs solely to baseline diagnostics;
    it is discarded before the real managed model run. Diagnostic outputs do
    not replace the selected baseline output or feed its following layers.
    """

    def __init__(self, model, backend, *, probe=False, page_size=16):
        self.model, self.backend = model, backend
        self.probe, self.page_size = probe, page_size
        self.stage = "unspecified"
        self.records = []
        self._original = []

    def __enter__(self):
        attentions = [layer.self_attn for layer in self.model.model.layers]
        if not all(isinstance(attention, Qwen3Attention) for attention in attentions):
            raise TypeError("dense probe requires the pinned Qwen3Attention contract")
        for attention in attentions:
            self._original.append((attention, attention.forward))
            attention.forward = types.MethodType(self._forward, attention)
        return self

    def __exit__(self, *exc):
        for attention, original in self._original:
            attention.forward = original
        self._original.clear()

    def _forward(self, attention, hidden_states, position_embeddings, attention_mask,
                 past_key_values=None, cache_position=None, **kwargs):
        shape = hidden_states.shape[:-1]
        projected_shape = (*shape, -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(hidden_states).view(projected_shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(hidden_states).view(projected_shape)).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(projected_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None:
            cos, sin = position_embeddings
            k, v = past_key_values.update(k, v, attention.layer_idx,
                                          {"sin": sin, "cos": cos, "cache_position": cache_position})
        if self.backend == "dense_fp32_accum":
            output, weights = _dense_accum_attention(attention, q, k, v, attention_mask)
        else:
            interface = eager_attention_forward if self.backend == "eager" else ALL_ATTENTION_FUNCTIONS[self.backend]
            output, weights = interface(attention, q, k, v, attention_mask, dropout=0.0,
                                         scaling=attention.scaling, sliding_window=attention.sliding_window, **kwargs)
        if self.probe:
            self._record(attention, q, k, v, attention_mask, cache_position)
        output = output.reshape(*shape, -1).contiguous()
        return attention.o_proj(output), weights

    def _record(self, attention, q, k, v, mask, query_positions):
        if q.shape[0] != 1 or query_positions is None or k.shape[-2] != int(query_positions[-1]) + 1:
            raise ValueError("precision probe requires batch-one contiguous full-history dense K/V")
        eager, _ = eager_attention_forward(attention, q, k, v, mask,
                                           scaling=attention.scaling, dropout=0.0)
        dense32, _ = _dense_accum_attention(attention, q, k, v, mask)
        dense64, _ = _dense_accum_attention(attention, q, k, v, mask, accumulation_dtype=torch.float64)
        heads, length, dim = k.shape[1:]
        pages = math.ceil(length / self.page_size)
        # Only diagnostic baseline code packs its already-existing dense cache.
        # The managed forward never calls this method and never gathers history.
        kp = torch.empty((heads*pages, self.page_size, dim), device=k.device, dtype=k.dtype)
        vp = torch.empty_like(kp)
        slots = torch.arange(heads*pages, device=k.device, dtype=torch.int64).reshape(heads, pages)
        lengths = torch.full((heads, pages), self.page_size, device=k.device, dtype=torch.int64)
        lengths[:, -1] = length - (pages-1)*self.page_size
        positions = torch.arange(pages*self.page_size, device=k.device, dtype=torch.int64).reshape(1, pages, self.page_size).expand(heads, -1, -1)
        for h in range(heads):
            for page in range(pages):
                start = page*self.page_size
                stop = min(start+self.page_size, length)
                kp[h*pages+page, :stop-start].copy_(k[0, h, start:stop])
                vp[h*pages+page, :stop-start].copy_(v[0, h, start:stop])
        paged = paged_attention(q[0], kp, vp, slots, lengths, positions,
                                query_positions=query_positions, num_q_per_kv=attention.num_key_value_groups,
                                scale=attention.scaling, windows=[attention.sliding_window or 0]*heads)
        paged = paged.transpose(0, 1).unsqueeze(0)
        repeated_k = repeat_kv(k, attention.num_key_value_groups)
        eager_scores = torch.matmul(q, repeated_k.transpose(2, 3)) * attention.scaling
        dense_scores = torch.matmul(q.float(), repeated_k.float().transpose(2, 3)) * attention.scaling
        masked_eager = eager_scores
        if mask is not None:
            masked_eager = masked_eager + mask[:, :, :, :length]
        probabilities32 = torch.softmax(masked_eager, dim=-1, dtype=torch.float32)
        probabilities_low = probabilities32.to(q.dtype)
        self.records.append(dict(
            stage=self.stage, layer=attention.layer_idx, q_dtype=str(q.dtype),
            query_tokens=q.shape[-2], history_tokens=length,
            same_inputs=True, reference_output_rounding="All outputs compared in model Q dtype; dense FP32/FP64 and paged round at final output, while HF eager also rounds intermediate scores/probabilities",
            qk_rounding_eager_vs_fp32=_tensor_error(eager_scores, dense_scores),
            probability_cast_error=_tensor_error(probabilities_low, probabilities32),
            paged_vs_dense_fp32=_tensor_error(paged, dense32),
            eager_vs_dense_fp32=_tensor_error(eager, dense32),
            paged_vs_dense_fp64_rounded=_tensor_error(paged, dense64),
            eager_vs_dense_fp64_rounded=_tensor_error(eager, dense64),
        ))


def _page_hashes(manager, request_id):
    """Exact parent integrity proof; bounded one-page temporary CPU storage."""
    hashes = {}
    with manager.bind(request_id) as lease:
        for ref in sorted(lease.refs):
            digest = hashlib.sha256()
            k, v = manager.pool.read_page(ref)
            for tensor in (k, v):
                payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                digest.update(payload)
            hashes[ref.content_id] = digest.hexdigest()
    return hashes


def _request_refs(manager, request_id):
    return {p for s in manager.version(request_id).segments.values() for p in s.pages}


class _ManagedQwen3Probe:
    """Instance-scoped Qwen3 attention adapter; batch one, all model layers."""

    def __init__(self, model, manager):
        self.model, self.manager = model, manager
        self.caches = {}
        self.histories = {}
        self.request_id = None
        self.positions = ()
        self.forward_calls = 0
        self.hf_cache_objects_seen = 0
        self.max_current_chunk_tokens = 0
        self._original = []

    def __enter__(self):
        attentions = [layer.self_attn for layer in self.model.model.layers]
        if not all(isinstance(attention, Qwen3Attention) for attention in attentions):
            raise TypeError("only the Transformers 4.57.1 Qwen3Attention contract is supported")
        for attention in attentions:
            self._original.append((attention, attention.forward))
            attention.forward = types.MethodType(self._forward, attention)
        return self

    def __exit__(self, *exc):
        for attention, original in self._original:
            attention.forward = original
        self._original.clear()

    def add_request(self, request_id, prompt):
        context = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        self.manager.create_request(request_id, context_id=context, contract="qwen3-4.57.1-target-rope-v1")
        cache = ManagedSegPagedKVCache(self.manager, request_id,
                                      num_layers=self.model.config.num_hidden_layers,
                                      num_kv_heads=self.model.config.num_key_value_heads)
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if attn.sliding_window is not None:
                cache.set_head_policy(attn.layer_idx, 0, "local", window=attn.sliding_window)
                for head in range(1, cache.num_kv_heads):
                    cache.set_head_policy(attn.layer_idx, head, "local", window=attn.sliding_window)
        self.caches[request_id] = cache
        self.histories[request_id] = []

    def fork(self, source, target):
        self.caches[target] = self.caches[source].fork(target)
        self.histories[target] = list(self.histories[source])

    def run(self, request_id, tokens, device):
        self.request_id = request_id
        start = len(self.histories[request_id])
        self.positions = tuple(range(start, start + len(tokens)))
        self.histories[request_id].extend(tokens)
        self.max_current_chunk_tokens = max(self.max_current_chunk_tokens, len(tokens))
        positions = torch.tensor(self.positions, device=device, dtype=torch.long)
        try:
            output, logits, elapsed = _timed_forward(
                self.model, tokens, device,
                use_cache=False, past_key_values=None,
                cache_position=positions, position_ids=positions.unsqueeze(0),
                # The direct paged kernel applies the true positional mask.
                # Supplying this mapping avoids building an unused dense mask.
                attention_mask={"full_attention": None, "sliding_attention": None},
            )
            if output.past_key_values is not None:
                raise AssertionError("managed model unexpectedly returned an HF KV cache")
            return logits, elapsed
        finally:
            self.request_id = None
            self.positions = ()

    def _forward(self, attention, hidden_states, position_embeddings, attention_mask,
                 past_key_values=None, cache_position=None, **kwargs):
        self.forward_calls += 1
        if past_key_values is not None:
            self.hf_cache_objects_seen += 1
            raise AssertionError("managed forward must never receive a dense history Cache")
        if self.request_id is None or hidden_states.shape[0] != 1:
            raise ValueError("probe requires one explicitly selected request with batch size one")
        if hidden_states.shape[1] != len(self.positions):
            raise ValueError("query chunk and target positions disagree")
        # These projection/norm/RoPE operations follow the pinned upstream
        # Qwen3Attention.forward. No historical K/V tensor is formed here.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        version = self.manager.version(self.request_id)
        history_hash = hashlib.sha256(json.dumps(self.histories[self.request_id]).encode()).hexdigest()
        provenance = f"qwen3:{attention.layer_idx}:{history_hash}"
        keys = [(attention.layer_idx, h, "history") for h in range(k.shape[1])]
        existing = [key in version.segments for key in keys]
        if any(existing) and not all(existing):
            raise RuntimeError("partially populated layer cannot be advanced")
        if not any(existing):
            writes = [SegmentWrite(key, k[0, h], v[0, h], self.positions, provenance,
                                   position_basis="target-rope") for h, key in enumerate(keys)]
            self.manager.update(self.request_id, writes=writes)
        else:
            for h, key in enumerate(keys):
                self.manager.append(self.request_id, key, k[0, h], v[0, h], self.positions,
                                    provenance=provenance)
        out = self.caches[self.request_id].attention(
            q[0], layer=attention.layer_idx, query_positions=cache_position,
            num_q_per_kv=attention.num_key_value_groups, sm_scale=attention.scaling,
            use_fused=hidden_states.is_cuda, causal=True,
        )
        out = out.transpose(0, 1).unsqueeze(0).reshape(*input_shape, -1).contiguous()
        return attention.o_proj(out), None


def _comparison(name, baseline, logits, elapsed, tolerance):
    dense = baseline["logits"]
    error = (dense - logits).abs()
    dense_tokens, managed_tokens = dense.argmax(dim=-1), logits.argmax(dim=-1)
    max_error = float(error.max())
    token_matches = int((dense_tokens == managed_tokens).sum())
    finite = bool(torch.isfinite(logits).all())
    return dict(
        stage=name, input_tokens=baseline["input_tokens"], logits_shape=list(logits.shape),
        max_abs_logit_error=max_error, mean_abs_logit_error=float(error.mean()),
        relative_l2_error=float(torch.linalg.vector_norm(dense-logits) / torch.linalg.vector_norm(dense).clamp_min(1e-12)),
        compared_next_token_positions=dense_tokens.numel(), matching_next_token_positions=token_matches,
        dense_next_tokens=dense_tokens.flatten().tolist(), managed_next_tokens=managed_tokens.flatten().tolist(),
        all_next_tokens_match=token_matches == dense_tokens.numel(), all_logits_finite=finite,
        dense_forward_seconds=baseline["seconds"], managed_forward_seconds=elapsed,
        passed=finite and max_error <= tolerance and token_matches == dense_tokens.numel(),
    )


@torch.inference_mode()
def run_validation(args):
    if transformers.__version__ != "4.57.1":
        raise RuntimeError(f"expected Transformers 4.57.1, found {transformers.__version__}")
    if min(args.prompt_length, args.page_size, args.branch_steps) <= 0 or args.decode_steps < 0:
        raise ValueError("positive prompt/page/branch sizes and nonnegative shared decode steps required")
    if (args.prompt_length + args.decode_steps) % args.page_size == 0:
        raise ValueError("fork checkpoint must have a partial tail to exercise COW; change prompt/decode length")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported validation devices are cpu and cuda")
    dtype_name = args.dtype or ("bfloat16" if device.type == "cuda" else "float32")
    dtype = getattr(torch, dtype_name)
    tolerance = args.max_logit_error
    if tolerance is None:
        tolerance = {"float32": 3e-4, "float16": 0.05, "bfloat16": 0.25}[dtype_name]
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("--max-logit-error must be finite and nonnegative")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    load_started = time.perf_counter()
    hf_backend = "eager" if args.dense_backend == "dense_fp32_accum" else args.dense_backend
    if args.tiny:
        config = Qwen3Config(vocab_size=257, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=16, max_position_embeddings=4096, attention_dropout=0.0)
        config._attn_implementation = hf_backend
        model = Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()
        weights_path = None
    else:
        weights_path = str(Path(args.model_path).resolve())
        if not Path(weights_path).is_dir():
            raise ValueError("--model-path must be an existing local weight directory")
        model = Qwen3ForCausalLM.from_pretrained(weights_path, local_files_only=True,
                                                torch_dtype=dtype, attn_implementation=hf_backend).to(device).eval()
    _sync(device)
    load_seconds = time.perf_counter() - load_started
    config = model.config
    if config.model_type != "qwen3":
        raise ValueError("only Qwen3ForCausalLM weights are supported")
    if args.prompt_length + args.decode_steps + args.branch_steps > config.max_position_embeddings:
        raise ValueError("experiment sequence exceeds the model's declared maximum context")
    prompt = _tokens(11, args.prompt_length, config.vocab_size)
    common = _tokens(71, args.decode_steps, config.vocab_size)
    branches = {"left": _tokens(131, args.branch_steps, config.vocab_size),
                "right": _tokens(191, args.branch_steps, config.vocab_size)}
    if branches["left"] == branches["right"]:
        raise AssertionError("fork continuations must differ")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dense_probe = (_DenseQwen3AttentionProbe(model, args.dense_backend,
                                            probe=args.attention_probe, page_size=args.page_size)
                   if args.attention_probe or args.dense_backend == "dense_fp32_accum" else None)
    with dense_probe if dense_probe is not None else nullcontext():
        baseline, dense_fork_seconds = _dense_baseline(model, prompt, common, branches, device, dense_probe)
    precision_records = dense_probe.records if dense_probe is not None else []
    dense_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    max_length = args.prompt_length + args.decode_steps + args.branch_steps
    capacity = config.num_hidden_layers * config.num_key_value_heads * (3*math.ceil(max_length/args.page_size) + 2)
    pool = HeadPagePool(capacity, args.page_size, config.head_dim, dtype=dtype, device=device)
    manager = HeadKVManager(pool)
    stages, memory_snapshots = [], []
    parent_unchanged = []
    try:
        with _ManagedQwen3Probe(model, manager) as probe:
            probe.add_request("parent", prompt)
            logits, elapsed = probe.run("parent", prompt, device)
            stages.append(_comparison("prefill", baseline["prefill"], logits, elapsed, tolerance))
            for i, token in enumerate(common):
                name = f"shared_decode_{i}"
                logits, elapsed = probe.run("parent", [token], device)
                stages.append(_comparison(name, baseline[name], logits, elapsed, tolerance))
            memory_snapshots.append(dict(stage="before_fork", **manager.stats()))
            parent_hashes = _page_hashes(manager, "parent")
            source_refs = _request_refs(manager, "parent")
            _sync(device)
            started = time.perf_counter()
            for name in branches:
                probe.fork("parent", name)
            _sync(device)
            managed_fork_seconds = time.perf_counter() - started
            all_shared_at_fork = all(_request_refs(manager, name) == source_refs for name in branches)
            memory_snapshots.append(dict(stage="after_fork", **manager.stats()))
            for branch, continuation in branches.items():
                for i, token in enumerate(continuation):
                    name = f"{branch}_decode_{i}"
                    logits, elapsed = probe.run(branch, [token], device)
                    stages.append(_comparison(name, baseline[name], logits, elapsed, tolerance))
                parent_unchanged.append(dict(after_branch=branch, unchanged=_page_hashes(manager, "parent") == parent_hashes))
                memory_snapshots.append(dict(stage=f"after_{branch}", **manager.stats()))
            retained_after_branch = {name: len(_request_refs(manager, name) & source_refs) for name in branches}
            parent_segments = manager.version("parent").segments
            detached_tails = {
                name: all(manager.version(name).segments[key].pages[len(seg.pages)-1] != seg.pages[-1]
                          for key, seg in parent_segments.items())
                for name in branches
            }
            probe_metrics = dict(forward_calls=probe.forward_calls,
                                 expected_forward_calls=len(stages)*config.num_hidden_layers,
                                 hf_cache_objects_seen=probe.hf_cache_objects_seen,
                                 max_current_projection_chunk_tokens=probe.max_current_chunk_tokens)
        managed_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        pre_release = manager.stats()
    finally:
        # The experiment aborts the whole request on a failed model step; it
        # does not claim model-wide transactional rollback across all layers.
        for request_id in ("left", "right", "parent"):
            if request_id in manager._requests:
                manager.release_request(request_id)
        _sync(device)
        post_release = manager.stats()
    source_path = Path(inspect.getfile(Qwen3Attention))
    contract_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    passed = (all(x["passed"] for x in stages) and all_shared_at_fork
              and all(x["unchanged"] for x in parent_unchanged)
              and all(detached_tails.values())
              and probe_metrics["forward_calls"] == probe_metrics["expected_forward_calls"]
              and post_release["free_pages"] == capacity and probe_metrics["hf_cache_objects_seen"] == 0)
    return dict(
        schema="head-kv-qwen3-validation-v1", passed=passed,
        is_real_pretrained_model=not args.tiny, weights_path=weights_path,
        model_class=type(model).__name__, model_type=config.model_type,
        layers=config.num_hidden_layers, query_heads=config.num_attention_heads,
        kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
        parameter_count=sum(p.numel() for p in model.parameters()),
        model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        device=str(device), dtype=dtype_name, torch_version=torch.__version__,
        transformers_version=transformers.__version__, transformers_qwen3_source_sha256=contract_sha,
        dense_backend=args.dense_backend, managed_backend="direct_triton_paged" if device.type == "cuda" else "pagewise_torch",
        dense_backend_semantics=(
            "Same model-weight dtype and projected Q/K/V; dense QK, scaling, softmax and PV in FP32, final attention output cast to model dtype. This is an explicit mathematical oracle, not HF eager equivalence."
            if args.dense_backend == "dense_fp32_accum" else
            "Unmodified HF eager/SDPA numerical semantics; optional probe reproduces the pinned Qwen3 projection/cache forward and calls the original HF attention function."
        ),
        page_size=args.page_size, fixed_prompt_token_ids=prompt,
        fixed_shared_decode_tokens=common, fixed_branch_tokens=branches,
        max_abs_logit_error=max(x["max_abs_logit_error"] for x in stages),
        all_next_tokens_match=all(x["all_next_tokens_match"] for x in stages),
        predeclared_max_abs_logit_error=tolerance, stages=stages,
        all_source_pages_shared_at_fork=all_shared_at_fork, source_page_count=len(source_refs),
        source_pages_still_shared_after_branch=retained_after_branch,
        all_shared_tails_detached_by_branch=detached_tails,
        parent_integrity_checks=parent_unchanged, probe=probe_metrics,
        pool_before_release=pre_release, pool_after_release=post_release,
        pool_allocated_bytes=pre_release["allocated_bytes"], pool_reserved_bytes=pre_release["reserved_slab_bytes"],
        memory_snapshots=memory_snapshots,
        dense_peak_torch_allocated_bytes_including_weights=dense_peak,
        managed_peak_torch_allocated_bytes_including_weights=managed_peak,
        load_seconds=load_seconds, dense_fork_seconds=dense_fork_seconds,
        managed_fork_seconds=managed_fork_seconds,
        attention_precision_probe_enabled=args.attention_probe,
        attention_precision_probe_count=len(precision_records),
        attention_precision_probe_expected_count=(len(stages)*config.num_hidden_layers if args.attention_probe else 0),
        attention_precision_probe_summary=(dict(
            max_paged_vs_dense_fp32=max(x["paged_vs_dense_fp32"]["max_abs"] for x in precision_records),
            max_eager_vs_dense_fp32=max(x["eager_vs_dense_fp32"]["max_abs"] for x in precision_records),
            max_paged_vs_dense_fp64_rounded=max(x["paged_vs_dense_fp64_rounded"]["max_abs"] for x in precision_records),
            max_eager_vs_dense_fp64_rounded=max(x["eager_vs_dense_fp64_rounded"]["max_abs"] for x in precision_records),
        ) if precision_records else None),
        attention_precision_records=precision_records,
        timing_scope="Synchronized single model forward including Python manager, validation, copies and any first-use Triton compilation; excludes model load, CPU logit export and page integrity hashes. With --attention-probe, baseline times additionally include dense FP32/FP64 and temporary paged diagnostic evaluations and are not comparable. Single samples, no throughput/speedup claim.",
        coverage_limits=["Batch one and short fixed teacher-forced token inputs", "No approximate RedKnot reuse/head-policy quality claim",
                         "Temporary Qwen3 instance adapter, not SGLang scheduler integration", "Writes commit per manager operation; experiment aborts on partial model failure",
                         "No model-weight reload between dense and managed runs; dense historical KV released before managed execution",
                         "BF16/FP16 HF eager additionally rounds QK scores/scaling and probabilities; direct kernel keeps scores/probabilities in FP32",
                         "An explicit dense_fp32_accum pass does not erase or supersede a failed HF eager run",
                         "Optional same-input precision probes run only in baseline and do not feed the model; no dense history is constructed in managed mode"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true", help="random tiny Qwen3 architecture smoke, not pretrained quality")
    source.add_argument("--model-path", help="existing local Qwen3ForCausalLM weights; downloads are disabled")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--dense-backend", choices=("eager", "sdpa", "dense_fp32_accum"), default="eager",
                        help="eager remains the original default; dense_fp32_accum is an explicitly different mathematical oracle")
    parser.add_argument("--attention-probe", action="store_true",
                        help="during baseline only, compare each layer's identical Q/K/V using eager, dense FP32/FP64 and direct paged attention")
    parser.add_argument("--prompt-length", type=int, default=33)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--branch-steps", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-logit-error", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_validation(args)
    except Exception as exc:
        result = dict(schema="head-kv-qwen3-validation-v1", passed=False,
                      is_real_pretrained_model=not args.tiny, weights_path=args.model_path,
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    # Numerical failures must still produce a standards-compliant JSON report.
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    text = json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
