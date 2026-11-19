"""Actual eager CP1 S1 prefill adapter; imports remain CPU-only.

The candidate executes native layers0..2, then physically gathers the decoder
output and all four lazy mHC fields once. Layers3..39 retain only selected rows.
The dense policy reference deliberately retains dense decoder/MLP rows solely
for qualification. Native identity controls and decode keep native YOCO layout.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

from .core import digest, require, token_digest, validate_digest
from .early_mhc_adapter import gather_packed_token_rows
from .index_hit20 import fingerprint
from .native_reuse import document_plan
from .s1_plan import all_chunks

LAZY = ("residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix")
MODES = ("native", "qualified_native", "all_selected", "capture_independent", "reference_sparse", "sensitive")
POLICY = ("reference_sparse", "sensitive")
QUERY_FIELDS = ("positions", "valid_token_mask", "gate_force_random", "images_seq_mask", "ngram_token_ids",
                "slot_mapping_store_swa", "query_indices", "context_lens")
QUERY_MAPS = ("ks", "ke", "kv_ca_index", "kv_ca_index_len", "kv_ca_index_offset_len", "klen")
RUNTIME_CHECK_NAMES = ("decoder_hidden_gather", "four_mhc_fields_axis1", "query_metadata_preserves_key_axes",
                       "empty_packed_sf_abi", "noncontiguous_absolute_positions", "sampler_final_row_rebind",
                       "engram_embedding_gather_and_empty_hook_consumption")
_MISSING = object()


def unwrap(value):
    return value.move() if callable(getattr(value, "move", None)) else value


def gather_hidden(value, rows, torch):
    value = unwrap(value)
    if not isinstance(value, tuple):
        return value.index_select(0, rows)
    if rows.numel():
        return gather_packed_token_rows(value, rows, torch_module=torch)
    data, sf = value
    # Zero logical rows still retain a nonzero aligned SF column stride for
    # the native zero-local-row EP dispatcher. No fake hidden/token is added.
    storage = torch.zeros((sf.shape[1], 4), dtype=sf.dtype, device=sf.device)
    return data.index_select(0, rows), storage[:, :0].transpose(0, 1)


def gather_lazy(lazy, rows):
    require(lazy is not None and all(hasattr(lazy, name) for name in LAZY), "four native mHC lazy fields required")
    out = copy(lazy)
    for name in LAZY:
        value = getattr(lazy, name)
        if value is not None:
            require(value.ndim >= 2 and value.shape[0] == 1, "mHC batch/token axis changed")
            value = value.index_select(1, rows)
        setattr(out, name, value)
    return out


def compact_metadata(full, rows, cu_seqlens_q):
    """Gather query vectors only. Compression/store/load key axes stay full."""
    require(full.is_prompt is True and full.num_reqs == 1, "S1 requires one prompt in CP1")
    out = copy(full)
    out.total_q = rows.numel()
    out.cu_seqlens_q = cu_seqlens_q
    for name in QUERY_FIELDS:
        value = getattr(full, name, None)
        if value is not None:
            require(value.shape[0] == full.total_q, "query field has foreign row layout: " + name)
            setattr(out, name, value.index_select(0, rows))
    for name in QUERY_MAPS:
        values = getattr(full, name, None)
        if values is not None:
            setattr(out, name, {key: value.index_select(0, rows) for key, value in values.items()})
    # S1Attention supplies an explicit local-window matrix on its own KV bank;
    # the native contiguous-query pivots must never be expanded for sparse rows.
    out.kv_swa_full_index = None
    out.kv_swa_full_index_pivots = None
    out.flash_mla_sched_meta_dict = None
    out.mlp_relu_schedule_metadata = None
    out.s1_compact_query_view = True
    out.s1_full_key_metadata = full
    return out


def compact_cross_queries(cross, rows):
    for name in ("topk_indices", "block_candidate_idxs"):
        value = getattr(cross, name, None)
        if value is not None:
            setattr(cross, name, value.index_select(0, rows))
    if hasattr(cross, "candidate_mqa_lens"):
        cross.candidate_mqa_lens = None
    if hasattr(cross, "candidate_mqa_schedule_metadata"):
        cross.candidate_mqa_schedule_metadata = {}


def refresh_compact_metadata(ctx, torch):
    """Bind lazily expanded native attention fields AFTER layers0..2 ran."""
    ctx.compact_meta=compact_metadata(ctx.full_meta,ctx.selected_rows,ctx.prepared["cu_q"])
    torch._assert_async((ctx.compact_meta.positions==ctx.prepared["positions"]).all(),
                        "S1 refreshed absolute positions changed")
    ctx.positions=ctx.compact_meta.positions
    return ctx.compact_meta


def rebind_sampler_output(hidden, output_rows, destination, source):
    """Adapt output slots only; no unselected decoder/mHC rows are computed."""
    result = hidden.new_zeros((output_rows, *hidden.shape[1:]))
    if source.numel():
        result.index_copy_(0, destination, hidden.index_select(0, source))
    return result


def selected_embedding_hook(hook, rows, torch, consumed):
    """Consume the original distributed fetch exactly once, then gather rows.

    Hashes are generated from the original full input/positions. Rehashing a
    compact token sequence would incorrectly create n-grams across removed gaps.
    """
    def selected():
        require(not consumed[0], "Engram embedding hook consumed twice")
        value = hook()
        consumed[0] = True
        return gather_hidden(value, rows, torch)
    return selected


def s1_documents(ids, documents, *, expected_document_count=8):
    require(type(expected_document_count) is int and expected_document_count in (8, 10),
            "S1 requires explicit legacy eight or MuSiQue ten document count")
    if isinstance(documents, (list, tuple)) and len(documents) == 1:
        d = documents[0]
        start, end = d.get("input_start"), d.get("input_end")
        require(type(start) is type(end) is int and start==0 and end==len(ids)==16384,
                "S1 single-document qualification requires complete16K span")
        key = token_digest(ids[start:end])
        require(d.get("token_ids_sha256") == key and d.get("document_id",key) == key
                and d.get("offline_position",0) == 0, "S1 single-document token identity")
        return (dict(start=start,end=end,key=key),)
    return document_plan(ids, documents, natural_boundary_inputs=True,
                         expected_document_count=expected_document_count)


class S1Controller:
    def __init__(self, layers, *, native_model, model_sha256, execution_sha256, tokenizer_sha256,
                 allow_sync=None, dependency_root=None, empty_ep_capability=None,
                 expected_document_count=8, enable_dflash_tail=False, **unused):
        require(not unused, "unknown S1 controller options")
        require(type(expected_document_count) is int and expected_document_count in (8, 10),
                "S1 explicit document contract must be eight or ten")
        require(type(enable_dflash_tail) is bool, "explicit DFlash bridge opt-in required")
        self.enable_dflash_tail, self.dflash_bridge = enable_dflash_tail, None
        self.expected_document_count = expected_document_count
        self.layers = tuple(layers)
        require(tuple(x.layer_idx for x in self.layers) == tuple(range(40)), "S1 requires forty decoder layers")
        self.native_model = native_model
        require(tuple(native_model.layers) == self.layers, "S1 native model ownership")
        for value in (model_sha256, execution_sha256, tokenizer_sha256):
            validate_digest(value, "S1 scope")
        self.scope = dict(model_sha256=model_sha256, execution_sha256=execution_sha256, tokenizer_sha256=tokenizer_sha256)
        self.allow_sync = allow_sync or nullcontext
        if empty_ep_capability is not None:
            from gpu_pilot.index_hit20_empty_ep_qualification import EmptyEPCapability
            require(isinstance(empty_ep_capability,EmptyEPCapability)
                    and empty_ep_capability.execution_sha256==execution_sha256
                    and empty_ep_capability.model_sha256==model_sha256,"foreign S1 empty-EP capability")
        self.empty_ep_capability = empty_ep_capability
        self.dependency_root = dependency_root
        self.handles, self.state, self.mode = [], "NEW", None
        self.reference, self.sparse_reference = {}, {}
        self.reference_input = self.sparse_reference_input = None
        self.qualified = self.sparse_qualified = False
        self.bank_generation = 0
        self.case_binding = self.case_input = None
        self.ctx = self.capture_frame = None
        self.cache_output_dir = None

    def _patch(self, obj, name, function):
        require(name not in obj.__dict__, "foreign instance hook: " + name)
        method = MethodType(function, obj)
        self.handles.append((obj, name, method))
        setattr(obj, name, method)

    def install(self):
        require(self.state == "NEW", "S1 controller already installed")
        import torch
        from deepinfer.model_executor.parallel_utils.parallel_state import PG
        from .s1_cache import S1Cache
        from .s1_attention import S1Attention
        self.t = torch
        require(PG.context_parallel_group().size() == 1, "S1 requires native CP1")
        require(self.native_model.yoco_boundary_layer_idx == 20 and self.native_model.hc_enabled,
                "S1 requires original boundary20 and mHC")
        require((not self.native_model.dflash_input_size or self.enable_dflash_tail)
                and [x.layer_idx for x in self.layers if x.engram is not None] == [1,14],
                "S1 binds native Engram layers1/14; DFlash requires explicit tail bridge")
        require(all(layer.self_attn.yoco_swa_share_hidden is False for layer in self.layers),
                "S1 binds native yoco_swa_share_hidden=False")
        self.original_boundary = self.native_model.yoco_boundary_layer_idx
        original_model = self.native_model.forward
        self.native_symbols = original_model.__func__.__globals__
        for key in ("CrossLayerRuntime", "Movable", "CausalLMForwardOutput", "HCLayerMixin", "mhc_pre_apply_mix", "tile_kernels"):
            require(key in self.native_symbols, "native model ABI missing " + key)
        self.cache = S1Cache(layers=self.layers, scope=self.scope, torch_module=torch, allow_sync=self.allow_sync)
        self.attention = S1Attention(self.cache, torch_module=torch, allow_sync=self.allow_sync)
        try:
            if self.enable_dflash_tail:
                from .s1_dflash_bridge import S1DFlashBridge
                self.dflash_bridge = S1DFlashBridge(self).install()
            for layer in self.layers:
                require(getattr(layer.self_attn, "_kv_cache_engram_native_hook", None) is None
                        and getattr(layer, "_s1_owner", None) is None, "foreign S1 attention/decoder owner")
                self._install_decoder(layer)
                self._install_attention(layer.self_attn)
            self._install_model_forward(original_model)
            self.state = "IDLE"
            return self
        except BaseException:
            self.close()
            raise

    def _install_model_forward(self, original_model):
        owner = self
        def model_forward(model, kv_caches, input_metadata, hidden_states=None, dflash=None):
            marker = getattr(input_metadata, "canonical_request_binding", None)
            if (marker is None or not input_metadata.is_prompt
                    or owner.mode not in (*POLICY, "capture_independent")):
                return original_model(kv_caches, input_metadata, hidden_states, dflash)
            try:
                if owner.mode == "capture_independent":
                    start, end = owner._marker(marker)
                    require(input_metadata.num_reqs == 1 and input_metadata.total_q == end-start
                            and model.yoco_boundary_layer_idx is None,
                            "S1 capture requires full-row CP1 output")
                    owner._check_dflash(dflash)
                    # All forty native layers, every capture hook, and the native
                    # mHC head/output norm finish before any output row is removed.
                    output = original_model(kv_caches, input_metadata, hidden_states, dflash)
                    full = output.hidden_states
                    require(full.ndim == 2 and full.shape[0] == end-start,
                            "S1 capture model did not return every original chunk row")
                    # The engine's sampler still uses its compact YOCO140 ABI.
                    # Match prepare_sample.cpp, including its single real last-row
                    # placeholder for a chunk entirely before the final keep range.
                    clip = max(0, min(len(owner.ids)-140-start, end-start-1))
                    rebound = type(output)(hidden_states=full[clip:])
                    owner.capture_sampler_events.append(dict(chunk_start=start, chunk_end=end,
                        full_output_rows=end-start, native_sampler_rows=end-start-clip,
                        output_local_start=clip, output_last_absolute_position=end-1,
                        source="full40_post_head_norm", rebinding_after_capture=True))
                    return rebound
                return owner._policy_model_forward(model, kv_caches, input_metadata, hidden_states, dflash)
            except BaseException:
                owner.state = "FAILED"
                raise
        self._patch(self.native_model, "forward", model_forward)

    def _marker(self, marker):
        require(self.state == "RUNNING" and isinstance(marker, dict)
                and marker.get("request_id") == self.rid and marker.get("phase") == self.phase,
                "S1 request marker mismatch")
        start, end = marker.get("start"), marker.get("end")
        require(type(start) is type(end) is int and 0 <= start < end <= len(self.ids)
                and start % 8192 == 0 and end == min(start+8192, len(self.ids)), "S1 native chunk geometry")
        return start, end

    def _install_decoder(self, layer):
        original, owner = layer.forward, self
        def decoder(mod, hidden_states, kv_cache, input_metadata, residual, **kwargs):
            marker = getattr(input_metadata, "canonical_request_binding", None)
            if marker is None or not input_metadata.is_prompt:
                return original(hidden_states, kv_cache, input_metadata, residual, **kwargs)
            try:
                start, end = owner._marker(marker)
                if mod.layer_idx == 0:
                    require(start == owner.coverage, "S1 prompt chunk order changed")
                    owner.coverage = end
                if owner.mode == "all_selected" and mod.layer_idx >= 3:
                    # Identity control follows ORIGINAL native query layout,
                    # including its layer20 compaction, with every row online.
                    h = unwrap(hidden_states)
                    ids = owner.t.arange(h.shape[0], dtype=owner.t.long, device=h.device)
                    hidden_states = gather_hidden(h, ids, owner.t)
                    residual = gather_lazy(residual, ids)
                bridge = getattr(owner, "dflash_bridge", None)
                positions = (list(range(start, end)) if mod.layer_idx < 3 or owner.mode == "reference_sparse"
                             else owner.ctx.positions_cpu) if bridge is not None and owner.mode in POLICY else None
                with (bridge.layer_call(mod.layer_idx, positions) if positions is not None else nullcontext()):
                    result = original(hidden_states, kv_cache, input_metadata, residual, **kwargs)
                owner._observe_layer(mod.layer_idx, start, end, result, input_metadata)
                return result
            except BaseException:
                owner.state = "FAILED"
                raise
        self._patch(layer, "forward", decoder)

    def _install_attention(self, mod):
        original, owner = mod.forward, self
        def attention(attn, hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                      cross_layer_runtime, **kwargs):
            marker = getattr(input_metadata, "canonical_request_binding", None)
            if marker is None or not input_metadata.is_prompt:
                return original(hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                                cross_layer_runtime, **kwargs)
            owner._marker(marker)
            if owner.mode in POLICY and attn.layer_idx >= 3:
                ctx = owner.ctx
                require(ctx is not None, "S1 policy attention outside model frame")
                hidden, bf16 = unwrap(hidden_states), unwrap(hidden_states_bf16)
                ctx.reference_hidden = ctx.reference_bf16 = None
                if owner.mode == "reference_sparse":
                    ctx.reference_hidden, ctx.reference_bf16 = hidden, bf16
                    hidden = gather_hidden(hidden, ctx.selected_rows, owner.t)
                    bf16 = bf16.index_select(0, ctx.selected_rows)
                    if attn.layer_idx == 20 and cross_layer_runtime.shared_hidden is not None:
                        ctx.reference_shared_hidden = cross_layer_runtime.shared_hidden
                        cross_layer_runtime.shared_hidden = cross_layer_runtime.shared_hidden.index_select(0,ctx.selected_rows)
                result = owner.attention.forward(attn, hidden, bf16, kv_cache, ctx.compact_meta,
                                                cross_layer_runtime, ctx)
                require(tuple(result.shape) == (ctx.selected_count, bf16.shape[-1]), "S1 attention output geometry")
                if owner.mode == "reference_sparse":
                    dense = result.new_zeros((ctx.end-ctx.start, result.shape[-1]))
                    dense.index_copy_(0, ctx.selected_rows, result)
                    return dense
                return result
            owner.capture_frame = (attn, input_metadata) if owner.mode == "capture_independent" else None
            try:
                return original(hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                                cross_layer_runtime, **kwargs)
            finally:
                owner.capture_frame = None
        self._patch(mod, "forward", attention)
        original_swa = mod._prepare_swa_kv
        def prepare_swa(attn, raw, kv_cache, meta, use_cp):
            if owner.mode == "capture_independent" and getattr(meta, "canonical_request_binding", None) is not None:
                owner.cache.capture_swa(attn, raw, meta)
            return original_swa(raw, kv_cache, meta, use_cp)
        self._patch(mod, "_prepare_swa_kv", prepare_swa)
        if mod.layer_idx in (8, 14):
            projection = mod._project_current_hidden_inputs
            def current_projection(attn, *args, **kwargs):
                value = projection(*args, **kwargs)
                if owner.capture_frame is not None and owner.mode == "capture_independent":
                    owner.cache.capture_global(attn, value[0], owner.capture_frame[1])
                return value
            self._patch(mod, "_project_current_hidden_inputs", current_projection)
        if mod.layer_idx == 20:
            require(mod.shared_hidden_kv_proj is not None, "source20 shared-hidden projection missing")
            projection = mod.shared_hidden_kv_proj.forward
            def shared_projection(module, *args, **kwargs):
                value = projection(*args, **kwargs)
                if owner.capture_frame is not None and owner.mode == "capture_independent":
                    owner.cache.capture_global(mod, value, owner.capture_frame[1])
                return value
            self._patch(mod.shared_hidden_kv_proj, "forward", shared_projection)

    def _make_context(self, full_meta, start, end):
        prepared = self.prepared[start, end]
        p, rows = prepared["plan"], prepared["rows"]
        meta = compact_metadata(full_meta, rows, prepared["cu_q"])
        self.t._assert_async((meta.positions == prepared["positions"]).all(), "S1 absolute query positions changed")
        require(getattr(full_meta,"yoco_compaction_enabled",False) is True
                and full_meta.yoco_consumer_total_q == len(p["sampler_output_positions"]), "native sampler output layout changed")
        self.t._assert_async((full_meta.yoco_consumer_positions == prepared["sample_positions"]).all(),
                             "native final-consumer sampler positions changed")
        return SimpleNamespace(full_meta=full_meta, compact_meta=meta, start=start, end=end,
            positions=meta.positions, positions_cpu=p["selected_positions"], selected_rows=rows,
            selected_count=p["selected_rows"], docs=self.docs, input_tokens=len(self.ids),
            mode=self.mode, phase=self.phase, rid=self.rid, plan=p, prepared=prepared,
            verify_sparse=self.verify_sparse, verify=self._qualifying_sparse(), reference_hidden=None, reference_bf16=None,
            reference_selected_rows=rows)

    def _check_dflash(self, dflash):
        bridge = getattr(self, "dflash_bridge", None)
        if bridge is None:
            require(dflash is None and not self.native_model.dflash_input_size, "S1 MTP requires the tail bridge")
        else:
            bridge.check_native(dflash)

    def _commit_dflash_tail(self, meta, dflash, caches, end):
        bridge = getattr(self, "dflash_bridge", None)
        if bridge is not None and end == len(self.ids):
            bridge.finish(meta, dflash, caches)

    def _policy_model_forward(self, model, caches, meta, hidden, dflash):
        require(self.ctx is None, "S1 model frame overlap")
        self._check_dflash(dflash)
        start, end = self._marker(meta.canonical_request_binding)
        require(meta.total_q == end-start and meta.num_reqs == 1, "S1 full prompt metadata expected")
        g = self.native_symbols
        hidden = model.embed_tokens(meta.input_ids) if hidden is None else hidden
        residual, cross = None, g["CrossLayerRuntime"]()
        hooks = {}
        if model.engram_runtime is not None:
            require(meta.ngram_token_ids is not None, "native Engram ngrams missing")
            hashes = g["tile_kernels"].engram.engram_hash(meta.ngram_token_ids, model.ngram_multipliers,
                model.ngram_vocab_sizes, model.ngram_offsets, getattr(meta, "images_seq_mask", None))
            hooks = model.engram_runtime.fetch(hashes)
        ctx = self._make_context(meta, start, end)
        self.ctx = ctx
        source2_before = None
        try:
            for i, layer in enumerate(self.layers):
                if i == 3:
                    # InputMetadata.compute_attn_info populates klen and key
                    # maps lazily in native first3; an earlier shallow copy can
                    # retain None even after the full metadata has been filled.
                    refresh_compact_metadata(ctx,self.t)
                    if self.mode == "sensitive":
                        hidden, residual = gather_hidden(hidden, ctx.selected_rows, self.t), gather_lazy(residual, ctx.selected_rows)
                    compact_cross_queries(cross, ctx.selected_rows)
                    if self._qualifying_sparse():
                        with self.allow_sync():
                            source2_before = self.attention.source2_fingerprint(caches[2], meta, cross)
                current_meta = meta if i < 3 or self.mode == "reference_sparse" else ctx.compact_meta
                hook = hooks.get(i)
                consumed = [False]
                if i >= 3 and layer.engram is not None:
                    require(callable(hook), "native deep Engram embedding hook missing")
                    if self.mode == "sensitive":
                        hook = selected_embedding_hook(hook,ctx.selected_rows,self.t,consumed)
                    else:
                        original_hook = hook
                        def hook(_original=original_hook,_consumed=consumed):
                            require(not _consumed[0], "Engram reference hook consumed twice")
                            value = _original()
                            _consumed[0] = True
                            return value
                if i >= 3 and ctx.selected_count == 0 and self.mode == "sensitive":
                    if layer.engram is not None:
                        hook()  # drain distributed fetch even though no gate rows exist
                    hidden, residual = self._empty_decoder(layer, caches[i], ctx, cross, hidden, residual)
                    self._observe_layer(i, start, end, (hidden, residual), current_meta)
                else:
                    hidden, residual = layer(g["Movable"](hidden), caches[i], current_meta, residual,
                        embed_hook=hook, cross_layer_runtime=cross, dflash_input=None)
                if i >= 3 and layer.engram is not None:
                    require(consumed[0], "S1 must consume native deep Engram distributed fetch")
                    self.engram_events.append(dict(layer=i,chunk_start=start,chunk_end=end,
                        original_hash_rows=end-start,selected_positions=ctx.positions_cpu,
                        actual_engram_rows=end-start if self.mode=="reference_sparse" else ctx.selected_count,
                        hook_consumed_once=True,original_ngram_hashes_preserved=True,
                        empty_hook_consumed_without_fake_rows=ctx.selected_count==0 and self.mode=="sensitive"))
            if self._qualifying_sparse():
                with self.allow_sync():
                    after = self.attention.source2_fingerprint(caches[2], meta, cross)
                require(source2_before == after, "S1 deep policy changed native source2 main/index cache")
                self.source2_events.append(dict(chunk_start=start, chunk_end=end, before=source2_before, after=after, exact=True))
            if self.mode == "reference_sparse":
                hidden, residual = gather_hidden(hidden, ctx.selected_rows, self.t), gather_lazy(residual, ctx.selected_rows)
            if ctx.selected_count:
                if hidden.dim() == 2:
                    hidden = g["HCLayerMixin"].hc_post(hidden, residual)
                require(hidden.ndim == 3, "S1 final mHC trunk layout")
                if model.hc_shifted:
                    hidden = g["mhc_pre_apply_mix"](hidden, residual.pre_layer_mix)
                else:
                    hidden = model.hc_head(hidden)
                hidden = model.output_norm(hidden)
            else:
                hidden = hidden.new_empty((0, model.output_norm.weight.numel()))
            self._commit_dflash_tail(meta, dflash, caches, end)
            p = ctx.prepared
            # The engine's sampler selected_token_indices still use the native
            # final-consumer layout. Only output slots are rebound, after all
            # candidate work; the sampled last global token is always selected.
            hidden = rebind_sampler_output(hidden, len(ctx.plan["sampler_output_positions"]), p["sample_dst"], p["sample_src"])
            return g["CausalLMForwardOutput"](hidden_states=hidden)
        finally:
            self.ctx = None

    def _empty_decoder(self, layer, cache, ctx, cross, hidden, residual):
        require(self.empty_ep_capability is not None, "S1 zero-row chunk requires actual native empty EP qualification")
        permit = SimpleNamespace(full_rows=ctx.end-ctx.start, selected_count=0)
        self.empty_ep_capability.permits(layer, ctx.full_meta, permit)
        bf16 = hidden.new_empty((0, layer.mlp.hidden_size), dtype=self.t.bfloat16)
        sf = self.t.zeros((layer.mlp.hidden_size//128, 4), dtype=self.t.int32, device=hidden.device)
        packed = (self.t.empty_like(bf16, dtype=self.t.float8_e4m3fn), sf[:, :0].transpose(0, 1))
        self.attention.forward(layer.self_attn, packed, bf16, cache, ctx.compact_meta, cross, ctx)
        # Every rank must retain the native EP sequence even with zero local
        # rows. Empty mHC/normalization kernels are skipped; no dummy row exists.
        result = layer.mlp(packed, bf16, ctx.compact_meta)
        require(result.shape == bf16.shape, "native zero-local-row MLP output mismatch")
        return result, residual

    def _qualifying_sparse(self):
        return self.mode == "reference_sparse" or self.verify_sparse

    def _observe_layer(self, layer, start, end, result, meta):
        hidden, lazy = result
        rows = hidden.shape[0]
        selected = list(range(start, end)) if layer < 3 or self.mode not in POLICY else self.ctx.positions_cpu
        if self.mode in POLICY:
            actual = rows
            self.decoder_events.append(dict(layer=layer, chunk_start=start, chunk_end=end,
                full_rows=end-start, selected_positions=selected, selected_rows=len(selected),
                actual_attention_rows=(len(selected) if layer >= 3 else rows), actual_ffn_rows=actual,
                actual_residual_rows=actual, actual_mhc_rows=actual,
                dense_reference_only=self.mode == "reference_sparse" and layer >= 3,
                original_absolute_positions=True, empty_native_ep_call=layer >= 3 and actual == 0))
        if self.mode in ("qualified_native", "all_selected") or self._qualifying_sparse():
            if self.mode == "reference_sparse" and layer >= 3:
                hidden, lazy = gather_hidden(hidden, self.ctx.selected_rows, self.t), gather_lazy(lazy, self.ctx.selected_rows)
            with self.allow_sync():
                value = dict(output=fingerprint(hidden, self.t), lazy={k:fingerprint(getattr(lazy,k),self.t) for k in LAZY})
            key = (start, end, layer)
            target = self.observed if self.mode in ("qualified_native", "all_selected") else self.sparse_observed
            require(key not in target, "duplicate S1 decoder fingerprint")
            target[key] = value
            reference = self.reference if self.mode == "all_selected" else self.sparse_reference
            if self.mode == "all_selected" or self.verify_sparse:
                require(reference.get(key) == value, "S1 output/all-four-mHC byte parity failed: " + str(key))

    def set_cache_output_dir(self, path):
        require(self.state in ("IDLE", "COMPLETE"), "cannot change active cache destination")
        path = Path(path)
        path.mkdir(parents=False, exist_ok=False)
        self.cache_output_dir = path

    def clear_cache(self, *, reset_sparse_qualification=False):
        require(self.state in ("IDLE", "COMPLETE"), "cannot clear active S1 bank")
        self.cache.bank.clear()
        self.cache.capture_reports.clear()
        self.cache.global_bases.clear()
        self.cache.global_carry.clear()
        self.cache.global_events.clear()
        self.cache.preparation_s=0.0
        self.case_binding = self.case_input = None
        self.bank_generation += 1
        if reset_sparse_qualification:
            self.sparse_reference, self.sparse_reference_input, self.sparse_qualified = {}, None, False
            if getattr(self, "dflash_bridge", None) is not None:
                self.dflash_bridge.clear_reference()

    def prepare_case(self, *, input_ids, documents, index_device):
        require(self.state in ("IDLE", "COMPLETE"), "cannot bind active S1 case")
        docs = s1_documents(input_ids, documents, expected_document_count=self.expected_document_count)
        self.cache.prepare_case(documents=docs, input_tokens=len(input_ids), index_device=index_device)
        self.case_input = token_digest(input_ids)
        self.case_binding = tuple((key, id(e["tensor"]), e["version"]) for key,e in sorted(self.cache.bank.items()))

    def reset(self, rid, mode, *, arm, input_ids, documents=None, phase, fraction=None,
              index_device=None, verify_sparse=False):
        require(self.state in ("IDLE", "COMPLETE") and self.ctx is None and self.capture_frame is None, "unclean S1 request")
        require(self.native_model.yoco_boundary_layer_idx==self.original_boundary,
                "previous S1 capture layout was not restored")
        require(mode in MODES and fraction is None and type(verify_sparse) is bool, "S1 mode/fraction")
        require((arm == "recomputed" and mode == "native") or (arm == "s1" and mode in POLICY)
                or (arm == "qualification" and mode in ("qualified_native", "all_selected"))
                or (arm == "offline" and mode == "capture_independent"), "S1 arm/mode mismatch")
        require(not verify_sparse or mode == "sensitive", "invalid S1 sparse verification mode")
        require(type(rid) is str and rid and type(phase) is str and phase, "S1 request identity")
        self.rid, self.phase, self.mode, self.arm = rid, phase, mode, arm
        self.ids = tuple(input_ids)
        self.input_sha = token_digest(self.ids)
        self.verify_sparse = verify_sparse
        self.coverage = 0
        self.observed, self.sparse_observed = {}, {}
        self.decoder_events, self.source2_events, self.engram_events, self.capture_sampler_events = [], [], [], []
        self.prepared = {}
        if mode in ("qualified_native", "all_selected"):
            require(len(self.ids) == 16384, "S1 native identity requires16K")
            self.docs = ()
            if mode == "qualified_native":
                require(not self.reference, "native reference already populated")
                self.reference_input = self.input_sha
            else:
                require(self.reference_input == self.input_sha and len(self.reference) == 80, "fresh native reference missing")
        else:
            self.docs = (document_plan(self.ids, documents, independent=True, natural_boundary_inputs=True)
                         if mode == "capture_independent" else s1_documents(self.ids, documents,
                             expected_document_count=self.expected_document_count))
        if mode in POLICY:
            require(self.qualified, "S1 native identity gate not passed")
            require(self.case_input == self.input_sha and self.case_binding is not None, "S1 case bank not prepared")
            require(index_device is not None, "S1 explicit device required")
            if mode == "reference_sparse":
                require(not self.sparse_reference, "S1 sparse reference already exists")
                self.sparse_reference_input = self.input_sha
            else:
                require(self.sparse_qualified or (verify_sparse and self.sparse_reference_input == self.input_sha
                                                and self.sparse_reference), "fresh same-policy S1 qualification required")
            for p in all_chunks(self.docs, len(self.ids)):
                ids = lambda values, dtype=self.t.long: self.t.tensor(values, device=index_device, dtype=dtype)
                self.prepared[p["chunk_start"],p["chunk_end"]] = dict(plan=p,
                    rows=ids(p["selected_local_rows"]), positions=ids(p["selected_positions"]),
                    cu_q=ids([0,p["selected_rows"]],self.t.int32),
                    sample_positions=ids(p["sampler_output_positions"]),
                    sample_dst=ids(p["sampler_destination_rows"]), sample_src=ids(p["sampler_source_rows"]))
        if mode == "capture_independent":
            require(self.cache_output_dir is not None, "S1 capture output directory required")
            self.cache.begin_capture(self.docs[0]["key"], len(self.ids), output_dir=self.cache_output_dir)
            self.native_model.yoco_boundary_layer_idx = None
        elif self.case_binding is not None:
            require(tuple((key,id(e["tensor"]),e["version"]) for key,e in sorted(self.cache.bank.items())) == self.case_binding,
                    "same S1 GPU bank must remain resident across arms")
            for key in self.cache.bank:
                self.cache.entry(*key)
            self.cache.begin_phase(phase, build_global_base=mode in POLICY)
        self.attention.begin_phase()
        if getattr(self, "dflash_bridge", None) is not None:
            self.dflash_bridge.reset()
        self.state = "RUNNING"

    def finish(self):
        require(self.state == "RUNNING" and self.ctx is None and self.coverage == len(self.ids), "incomplete S1 request")
        self.native_model.yoco_boundary_layer_idx = self.original_boundary
        if self.mode == "capture_independent":
            self.cache.finish_capture()
        expected = 40*((len(self.ids)+8191)//8192)
        if self.mode in ("qualified_native", "all_selected"):
            require(len(self.observed) == expected == 80, "S1 native identity coverage")
            if self.mode == "qualified_native":
                self.reference = dict(self.observed)
            else:
                self.qualified = True
        if self._qualifying_sparse():
            require(len(self.sparse_observed) == expected and len(self.source2_events) == expected//40,
                    "S1 same-policy identity/source2 coverage")
            if self.mode == "reference_sparse":
                self.sparse_reference = dict(self.sparse_observed)
            else:
                self.sparse_qualified = True
        if self.mode in POLICY:
            require(len(self.decoder_events) == expected, "S1 actual decoder accounting incomplete")
        cache = self.cache.report()
        fingerprints = lambda values: [dict(key=list(k), value=v) for k,v in sorted(values.items())]
        bank_scope = digest(dict(scope=self.scope,documents=[dict(document_sha256=c["document_sha256"],
            document_tokens=c["document_tokens"]) for c in cache["capture_reports"]]))
        report = dict(schema="s1-controller-v1", mode=self.mode, arm=self.arm, phase=self.phase,
            input_ids_sha256=self.input_sha, full_input_tokens=len(self.ids), first_three_full=True,
            native_yoco_swa_share_hidden=False, decode_native=True, model_boundary_restored=True,
            selected_survive_layer39=self.mode in POLICY,
            yoco_layout_preserved=self.mode not in (*POLICY,"capture_independent"),
            unselected_deep_residual_mhc=self.mode == "reference_sparse", dense_reference_only=self.mode == "reference_sparse",
            extra_cross_document_tail_links=False, native_local_window=128,
            decoder_events=self.decoder_events, source2_identity_events=self.source2_events, engram_events=self.engram_events,
            capture_sampler_events=self.capture_sampler_events,
            source2_native_cache_preserved=True, source2_formal_byte_hash_claim=False,
            identity_native_query_layout_preserved=self.mode in ("qualified_native", "all_selected"),
            identity_all_kv_online=self.mode in ("qualified_native", "all_selected"),
            identity_exact=self.mode == "all_selected" and self.qualified,
            identity_check_count=len(self.observed), identity_fingerprints=fingerprints(self.observed),
            identity_fingerprints_sha256=digest(fingerprints(self.observed)) if self.observed else None,
            sparse_identity_exact=self.verify_sparse and self.sparse_qualified,
            sparse_identity_check_count=len(self.sparse_observed), sparse_identity_fingerprints=fingerprints(self.sparse_observed),
            sparse_identity_fingerprints_sha256=digest(fingerprints(self.sparse_observed)) if self.sparse_observed else None,
            bank_generation=self.bank_generation, phase_generation=cache["phase_generation"],offline_bank_scope_sha256=bank_scope,
            offline_bank_entry_count=cache["offline_bank_entry_count"], offline_bank_bytes=cache["offline_bank_bytes"],
            offline_swa_halo_rows=cache["offline_swa_halo_rows"], offline_global_raw_rows=cache["offline_global_raw_rows"],
            offline_bank_rows=cache["offline_swa_halo_rows"]+cache["offline_global_raw_rows"],
            document_token_lengths=[d["end"]-d["start"] for d in self.docs],
            expected_document_count=self.expected_document_count,
            offline_gpu_resident=True, gpu_bank_constant_across_arms=True,
            cache_storage_policy="same_s1_offline_gpu_bank_resident_both_arms", cache_scope=self.scope,
            cache_report=cache, attention_report=self.attention.report(),
            offline_payloads=cache["capture_reports"][-1]["payloads"] if self.mode == "capture_independent" else [],
            online_cache_prepare_s=cache["phase_preparation_s"],
            empty_ep_qualification_sha256=None if self.empty_ep_capability is None else self.empty_ep_capability.sha256)
        if getattr(self, "dflash_bridge", None) is not None:
            report["dflash_tail"] = self.dflash_bridge.report()
        self.last_report, self.state = report, "COMPLETE"
        return report

    def close(self):
        if self.state == "CLOSED":
            return
        if hasattr(self, "original_boundary"):
            self.native_model.yoco_boundary_layer_idx = self.original_boundary
        if getattr(self, "dflash_bridge", None) is not None:
            self.dflash_bridge.close()
        for obj, name, method in reversed(self.handles):
            require(obj.__dict__.get(name) is method, "S1 hook ownership changed")
            delattr(obj, name)
        self.handles.clear()
        self.state = "CLOSED"


def run_s1_runtime_qualification(layers, rank, torch, allow_sync, *, execution_sha256, model_sha256):
    """Real GPU row/metadata ABI tests on every rank; no synthetic CUDA claims."""
    require(type(rank) is int and 0 <= rank < 8, "S1 qualification rank")
    validate_digest(execution_sha256, "execution")
    validate_digest(model_sha256, "model")
    t = torch
    device = layers[0].self_attn.q_a_kv_swa_proj.weight.device
    require(device.type == "cuda", "S1 runtime qualification requires actual CUDA tensors")
    with allow_sync():
        width=layers[0].mlp.hidden_size
        require(width==5120,"S1 runtime hidden width changed")
        rows = t.tensor([0, 3, 8], dtype=t.long, device=device)
        base = t.arange(10*width, dtype=t.float32, device=device).remainder(127).reshape(10,width).to(t.bfloat16)
        gathered = gather_hidden(base, rows, t)
        require(t.equal(gathered, t.stack([base[0],base[3],base[8]])), "GPU hidden gather")
        lazy = SimpleNamespace(residual=base[None,:,None,:].expand(1,10,4,width).clone(),
            post_layer_mix=base[:,:4].reshape(1,10,4,1).float(),
            comb_res_mix=base[:,:16].reshape(1,10,4,4).float(),
            pre_layer_mix=base[:,:4].reshape(1,10,4,1).float())
        got = gather_lazy(lazy, rows)
        require(all(t.equal(getattr(got,k),getattr(lazy,k)[:,[0,3,8]]) for k in LAZY), "GPU four-field mHC gather")
        full = SimpleNamespace(is_prompt=True,num_reqs=1,total_q=10,positions=t.arange(100,110,device=device),
            ks={"2":t.zeros(10,dtype=t.int32,device=device)}, ke={"2":t.arange(10,dtype=t.int32,device=device)},
            positions_c={"2":t.arange(5,device=device)}, slot_mapping_store_ca={"2":t.arange(5,device=device)},
            cu_seqlens_k=t.tensor([0,110],dtype=t.int32,device=device))
        meta = compact_metadata(full,rows,t.tensor([0,3],dtype=t.int32,device=device))
        require(meta.positions_c is full.positions_c and meta.slot_mapping_store_ca is full.slot_mapping_store_ca
                and meta.cu_seqlens_k is full.cu_seqlens_k and t.equal(meta.ke["2"],full.ke["2"][rows]), "GPU metadata key/query axes")
        require(meta.positions.cpu().tolist()==[100,103,108] and full.total_q==10, "GPU absolute positions")
        empty = rows[:0]
        packed = (base.to(t.float8_e4m3fn), t.zeros((10,width//128),dtype=t.int32,device=device))
        zero = gather_hidden(packed,empty,t)
        require(zero[0].shape==(0,width) and zero[1].shape==(0,width//128) and zero[1].stride()==(1,4), "GPU empty packed SF ABI")
        output = rebind_sampler_output(gathered,2,t.tensor([1],dtype=t.long,device=device),t.tensor([2],dtype=t.long,device=device))
        require(t.equal(output[-1],base[8]) and t.count_nonzero(output[0]).item()==0, "GPU sampler selected final row")
        consumed = [False]
        hook = selected_embedding_hook(lambda:base,rows,t,consumed)
        require(t.equal(hook(),gathered) and consumed[0], "GPU Engram full-hash row gather")
        consumed = [False]
        hook = selected_embedding_hook(lambda:packed,empty,t,consumed)
        empty_embed = hook()
        require(consumed[0] and empty_embed[0].shape[0]==0 and empty_embed[1].stride()==(1,4),
                "GPU empty Engram hook was not consumed")
        t.cuda.synchronize(device)
        checks = [dict(name=name,passed=True,actual_rows=0 if name=="empty_packed_sf_abi" else 3,
                       cuda_tensor_device=str(device)) for name in RUNTIME_CHECK_NAMES]
        value = dict(schema="s1-runtime-rank-qualification-v1",rank=rank,world_size=8,
            execution_configuration_sha256=execution_sha256,model_sha256=model_sha256,
            passed=True,cuda_exercised=True,checks=checks,
            hidden_size=width,mhc_mult=4,packed_sf_columns=width//128,
            output_sha256=fingerprint(gathered,t)["sha256"], quality_eligible=False,timing_valid=False,
            native_empty_ep_requires_separate_all_ranks_proof=True)
        return dict(value,sha256=digest(value))
