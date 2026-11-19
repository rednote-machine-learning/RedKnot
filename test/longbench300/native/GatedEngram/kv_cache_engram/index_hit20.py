"""Independent index-hit-frequency token-level SWA/attention/FFN experiment.

Experimental SM90/CP1 adapter. Import is CPU-only. First three layers remain
native; subsequent prefill rows are selected from raw native index hit counts.
Global/index preparation stays full, selected SWA KV is online, and other SWA
rows are restored from the immutable independent-document bank. Selected main
queries and FFNs execute; inactive branch outputs are zero before native mHC.
Native decode and YOCO layouts are preserved. Same-policy dense reference and
all-selected native qualification are separate required controls.
"""

from __future__ import annotations

import copy
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

from .core import digest, require, token_digest, validate_digest
from .early_mhc_adapter import gather_packed_token_rows
from .index_hit20_adapter import IndexHit20Adapter, make_selected_mlp_metadata
from .native_reuse import NativeReuseOps, document_plan

CAPTURE_LAYERS = tuple(range(40))
FFN_LAYERS = tuple(range(3, 40))
MODES = ("native", "capture_independent", "qualified_native", "all_selected", "sensitive", "reference_sparse")
ACTIVE_MODES = ("all_selected", "sensitive", "reference_sparse")
_MISSING = object()
_LAZY = ("residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix")


def build_ffn_mask(torch_module, full_rows, selected_ids, *, all_selected=False):
    """Build the bool mask from caller-owned device indices without index_put.

    The caller supplies sorted, valid row IDs; all_selected means those IDs are
    exactly arange(full_rows). Keep scalar values in device fill kernels rather than
    Python advanced-index assignment, whose CUDA scalar path may synchronize.
    This helper alone does not certify GPU execution or validate a row plan.
    """
    t = torch_module
    if all_selected:
        return t.ones(full_rows, device=selected_ids.device, dtype=t.bool)
    return t.zeros(full_rows, device=selected_ids.device, dtype=t.bool).index_fill_(0, selected_ids, True)


def fingerprint(value, torch):
    if value is None:
        return None
    require(isinstance(value, torch.Tensor), "qualification tensor required")
    require(bool(torch.isfinite(value).all().item()), "nonfinite qualification tensor")
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class IndexHit20Controller:
    def __init__(
        self, layers, *, model_sha256, execution_sha256, tokenizer_sha256, dependency_root=None, allow_sync=None,
        native_model=None, empty_ep_capability=None, natural_boundary_inputs=False
    ):
        require(type(natural_boundary_inputs) is bool, "natural boundary opt-in must be a bool")
        self.natural_boundary_inputs = natural_boundary_inputs
        self.document_lengths = {}
        self.layers = tuple(layers)
        require(tuple(layer_index.layer_idx for layer_index in self.layers) == tuple(range(40)), "forty native layers required")
        for value in (model_sha256, execution_sha256, tokenizer_sha256):
            validate_digest(value, "cache identity")
        self.scope = dict(model_sha256=model_sha256, execution_sha256=execution_sha256, tokenizer_sha256=tokenizer_sha256)
        if empty_ep_capability is not None:
            from gpu_pilot.index_hit20_empty_ep_qualification import EmptyEPCapability
            require(isinstance(empty_ep_capability, EmptyEPCapability)
                    and empty_ep_capability.model_sha256 == model_sha256
                    and empty_ep_capability.execution_sha256 == execution_sha256,
                    "foreign empty EP qualification capability")
        self.empty_ep_capability = empty_ep_capability
        self.allow_sync = allow_sync or nullcontext
        self.state, self.mode = "NEW", None
        self.handles, self.bank, self.reference = [], {}, {}
        self.reference_input, self.cache_output_dir = None, None
        self.qualified, self.frame, self.adapter = False, None, None
        self.projections = {}
        self.native_model = native_model
        self.original_yoco_boundary = None
        self.offline_full_layout = False
        self.q_b_projections = {}
        self.sparse_reference = {}
        self.sparse_reference_input = None
        self.sparse_qualified = False
        self.verify_sparse = False
        self.native_prepares = {}
        self.index_output_dir = None

    def _patch(self, obj, name, function):
        old = obj.__dict__.get(name, _MISSING)
        require(old is _MISSING, "foreign instance hook: " + name)
        bound = MethodType(function, obj)
        self.handles.append((obj, name, old, bound))
        setattr(obj, name, bound)

    def _make_projections(self, attn):
        """Preserve native FP8 weight/SF views; sources are not skipped."""
        t, fused = self.t, attn.q_a_kv_swa_proj
        require(
            attn.use_swa
            and attn.q_lora_rank == 1280
            and attn.qk_head_dim == 512
            and attn.qk_rope_head_dim == 64
            and attn.window_size == 128,
            "unqualified attention geometry",
        )
        if attn.yoco_swa_share_hidden:
            require(not attn.fuse_kv_swa and attn.kv_swa_proj.output_size == 512, "shared hidden SWA geometry changed")
            return fused, attn.kv_swa_proj
        w = fused.weight
        require(
            attn.fuse_kv_swa
            and fused.input_size == 5120
            and fused.output_size == 1792
            and w.is_cuda
            and w.dtype == t.float8_e4m3fn
            and tuple(w.shape) == (1792, 5120)
            and fused.bias is None
            and not fused.use_fp32_output
            and w.sf.dtype == t.int32
            and tuple(w.sf.shape) == (1792, 40),
            "unsupported fused Q/SWA weight ABI",
        )
        result = []
        for lo, hi in ((0, 1280), (1280, 1792)):
            param = t.nn.Parameter(w[lo:hi], requires_grad=False)
            param.sf = w.sf[lo:hi]
            linear = self.ops.Linear(5120, hi - lo, weight=param)
            require(
                linear.weight.data_ptr() == w[lo:hi].data_ptr()
                and linear.weight.sf.data_ptr() == w.sf[lo:hi].data_ptr()
                and linear.weight.sf.stride() == w.sf[lo:hi].stride(),
                "projection storage changed",
            )
            result.append(linear)
        return tuple(result)

    def install(self):
        require(self.state == "NEW", "controller already installed")
        require(
            self.native_model is not None
            and tuple(self.native_model.layers) == self.layers
            and self.native_model.yoco_boundary_layer_idx == 20,
            "original native YOCO boundary/model ownership required",
        )
        self.original_yoco_boundary = self.native_model.yoco_boundary_layer_idx
        self.ops = NativeReuseOps.from_deepinfer()
        self.t = self.ops.torch
        try:
            for layer in self.layers:
                attn = layer.self_attn
                require(
                    getattr(attn, "_early_mhc_owner", None) is None and getattr(attn, "_kv_cache_engram_native_hook", None) is None,
                    "foreign attention controller",
                )
                if layer.layer_idx >= 3:
                    self.projections[layer.layer_idx] = self._make_projections(attn)
                    self.q_b_projections[layer.layer_idx] = self._make_q_b_projections(attn)
                original_decoder = layer.forward

                def decoder(mod, *args, _original=original_decoder, **kwargs):
                    meta = kwargs.get("input_metadata", args[2] if len(args) > 2 else None)
                    marker = getattr(meta, "canonical_request_binding", None)
                    try:
                        if marker is not None:
                            self._marker(marker)
                            if mod.layer_idx == 0:
                                require(marker["start"] == self.coverage, "noncontiguous prompt chunks")
                                self.coverage = marker["end"]
                        result = _original(*args, **kwargs)
                        if marker is not None and self.mode in ("qualified_native", "all_selected"):
                            hidden, lazy = result
                            with self.allow_sync():
                                value = dict(
                                    output=fingerprint(hidden, self.t), lazy={k: fingerprint(getattr(lazy, k), self.t) for k in _LAZY}
                                )
                            key = (marker["start"], marker["end"], mod.layer_idx)
                            require(key not in self.observed, "duplicate qualification layer")
                            self.observed[key] = value
                            if self.mode == "qualified_native":
                                self.reference[key] = value
                            else:
                                require(self.reference.get(key) == value, "decoder/mHC byte parity failed: " + str(key))
                        if marker is not None and (self.mode == "reference_sparse" or self.verify_sparse):
                            hidden, lazy = result
                            with self.allow_sync():
                                value = dict(output=fingerprint(hidden, self.t),
                                             lazy={k: fingerprint(getattr(lazy, k), self.t) for k in _LAZY})
                            key = (marker["start"], marker["end"], mod.layer_idx)
                            require(key not in self.sparse_observed, "duplicate sparse qualification layer")
                            self.sparse_observed[key] = value
                            if self.mode == "reference_sparse":
                                self.sparse_reference[key] = value
                            else:
                                require(self.sparse_reference.get(key) == value,
                                        "sparse reference output/mHC byte parity failed: " + str(key))
                        return result
                    except BaseException:
                        self.state = "FAILED"
                        self._restore_offline_layout()
                        raise

                self._patch(layer, "forward", decoder)
                self._install_attention(attn)
            self.adapter = IndexHit20Adapter(
                self.layers, self.get_ffn_plan, self._selected_metadata,
                allow_empty_native=None if self.empty_ep_capability is None else self.empty_ep_capability.permits,
            ).install()
            self.state = "IDLE"
            return self
        except BaseException:
            self.close()
            raise

    def _make_q_b_projections(self, attn):
        require(not attn.use_fused_norm_rope_attn_rope_cast, "unfused native Q required")
        if not attn.has_local_index:
            return attn.q_b_proj, None
        fused, t = attn.q_b_proj, self.t
        width = attn.num_heads * attn.qk_head_dim
        require(fused.input_size == attn.q_lora_rank and fused.bias is None
                and fused.output_size == width + attn.index_n_head * attn.index_head_dim,
                "main/index Q-B fused ABI changed")
        result = []
        for lo, hi in ((0, width), (width, fused.output_size)):
            weight = t.nn.Parameter(fused.weight[lo:hi], requires_grad=False)
            if hasattr(fused.weight, "sf"):
                weight.sf = fused.weight.sf[lo:hi]
            linear = self.ops.Linear(fused.input_size, hi-lo, weight=weight)
            require(linear.weight.data_ptr() == fused.weight[lo:hi].data_ptr(),
                    "Q-B split copied native weights")
            if hasattr(weight, "sf"):
                require(linear.weight.sf.data_ptr() == weight.sf.data_ptr(), "Q-B SF view changed")
            result.append(linear)
        return tuple(result)

    def _install_attention(self, attn):
        original_forward, original_prepare = attn.forward, attn._prepare_swa_kv
        self.native_prepares[attn.layer_idx] = original_prepare

        def prepare(mod, raw, cache, meta, use_cp):
            marker = getattr(meta, "canonical_request_binding", None)
            if marker is not None and self.mode == "capture_independent":
                self._capture(mod.layer_idx, marker, self.ops.normalize(mod, raw))
            return original_prepare(raw, cache, meta, use_cp)

        def forward(mod, hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                    cross_layer_runtime=None, **kwargs):
            marker = getattr(input_metadata, "canonical_request_binding", None)
            if marker is None or not input_metadata.is_prompt or mod.layer_idx < 3 or self.mode not in ACTIVE_MODES:
                return original_forward(hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                                        cross_layer_runtime=cross_layer_runtime, **kwargs)
            self._marker(marker)
            require(not kwargs and self.frame is None and self.ops.PG.context_parallel_group().size() == 1
                    and not self.t.cuda.is_current_stream_capturing(), "owned eager CP1 forward required")
            require(not self.offline_full_layout and self.native_model.yoco_boundary_layer_idx == 20,
                    "native online YOCO layout changed")
            require(all(getattr(input_metadata, k, None) is None for k in
                        ("redknot_binding", "head_intervention_binding", "attention_range_binding",
                         "sparse_phase", "transparent_swa_binding")), "foreign intervention")
            self.frame = dict(layer=mod.layer_idx, marker=marker, metadata=input_metadata)
            try:
                return self._selected_attention(mod, hidden_states.move(), kv_cache, input_metadata,
                                                hidden_states_bf16.move(), cross_layer_runtime)
            except BaseException:
                self.state = "FAILED"
                raise
            finally:
                self.frame = None

        self._patch(attn, "_prepare_swa_kv", prepare)
        self._patch(attn, "forward", forward)

    def _selected_attention(self, mod, hidden, cache, meta, bf16, cross):
        from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
        from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import store_kv_cache
        t, n, marker = self.t, meta.total_q, meta.canonical_request_binding
        require(mod.use_swa and mod.has_packed_kv and mod.is_sparse and cross is not None,
                "native joint SWA/global topology required")
        block_size = cache.ca_cache.shape[1] * mod.kv_compress_ratio
        meta.compute_attn_info(mod.max_model_len, mod.window_size, block_size)
        shared = mod._resolve_shared_hidden(bf16, cross.shared_hidden)
        projected, index_weights = mod._project_current_hidden_inputs(bf16)
        if cross.consumer_full_metadata is not None:
            mod._store_yoco_full_kv(cross, cache)
            waiter = None
        else:
            if mod.shared_hidden_kv_proj is not None:
                require(projected is None and shared is not None, "shared global source contract")
                projected = mod.shared_hidden_kv_proj(shared)
            waiter = None if projected is None else lambda: projected
        q_a_proj, swa_proj = self.projections[mod.layer_idx]
        main_q_b_proj, index_q_b_proj = self.q_b_projections[mod.layer_idx]
        dense_reference = self.mode == "reference_sparse"
        # Index-source Q-A is necessarily full: the index voting queries are all
        # native current rows. Consumer layers avoid even inactive Q-A work.
        q_a = q_a_proj(hidden) if (mod.has_local_index or dense_reference or self.mode == "all_selected") else None
        q_a_normal = mod.q_a_norm(q_a) if q_a is not None else None
        if mod.has_local_index:
            index_q = index_q_b_proj(q_a_normal)
        else:
            index_q = None
        global_index = mod._compute_kv_ca_index(waiter, index_weights, index_q, cache, meta, cross)
        require(global_index is not None and tuple(global_index.shape) == (n, 512),
                "native top512 index current query geometry changed")
        plan = self._index_plan(mod, meta, global_index)
        ids = plan["ids"]
        selected_positions = meta.positions.index_select(0, ids)
        selected_n = ids.numel()
        source = shared if mod.yoco_swa_share_hidden else hidden
        require(source is not None and self.ops.rows(source) == n, "native SWA input row geometry")
        if dense_reference and n:
            q_a_full, raw_full = mod._project_q_and_swa(hidden, shared)
            q_b_full, index_q_ref = mod._project_q_b(q_a_full, meta.positions)
            if mod.has_local_index:
                with self.allow_sync():
                    self.ops.identity_error(index_q, index_q_ref)
            raw = raw_full.index_select(0, ids)
            q_b = q_b_full
        elif dense_reference:
            raw = bf16.new_empty((0, 512))
            q_b = bf16.new_empty((0, mod.num_heads, mod.qk_head_dim))
        else:
            if q_a_normal is None:
                q_a_selected = q_a_proj(self._gather(hidden, ids)) if selected_n else bf16.new_empty((0, mod.q_lora_rank))
                q_a_selected = mod.q_a_norm(q_a_selected) if selected_n else q_a_selected
            else:
                q_a_selected = self._gather(q_a_normal, ids) if selected_n else None
            raw = swa_proj(self._gather(source, ids)) if selected_n else bf16.new_empty((0, 512))
            if selected_n:
                q_b = main_q_b_proj(q_a_selected).unflatten(-1, (mod.num_heads, mod.qk_head_dim))
                q_b = mod._q_b_norm_rope(q_b, selected_positions)
            else:
                q_b = bf16.new_empty((0, mod.num_heads, mod.qk_head_dim))
        if self.mode == "all_selected" and n:
            with self.allow_sync():
                native_q_a, native_raw = mod._project_q_and_swa(hidden, shared)
                native_q_b, native_index_q = mod._project_q_b(native_q_a, meta.positions)
                self.ops.identity_error(q_a, native_q_a)
                self.ops.identity_error(raw, native_raw)
                self.ops.identity_error(q_b, native_q_b)
                if mod.has_local_index:
                    self.ops.identity_error(index_q, native_index_q)
                probe = t.arange(min(127, n), device=meta.positions.device)
                self.ops.identity_error(swa_proj(self._gather(source, probe)), native_raw.index_select(0, probe))
            checks = 5 if mod.has_local_index else 4
            self.projection_checks += checks
            self.projection_events.append(dict(layer=mod.layer_idx, chunk_start=marker["start"],
                chunk_end=marker["end"], full_rows=n, checks=checks, max_absolute_error=0.0,
                checks_named=["split_q_a", "split_swa", "split_main_q_b", "packed_gather_swa127"]
                             + (["split_index_q_b"] if mod.has_local_index else [])))
        subset = copy.copy(meta)
        subset.total_q, subset.positions = selected_n, selected_positions
        prepare = self.native_prepares[mod.layer_idx]
        if self.mode == "all_selected":
            swa, wait = prepare(raw, cache.swa_cache, meta, False)
        else:
            canonical = self._canonical(mod.layer_idx, marker, meta.positions)
            swa, wait = self.ops.postprocess(mod, canonical, meta)
            if selected_n:
                updated, selected_wait = prepare(raw, cache.swa_cache, subset, False)
                require(selected_wait is None, "unexpected SWA subset CP wait")
                swa.index_copy_(0, ids, updated)
        require(wait is None and swa.shape == (n, 1, 512), "prepared SWA shape changed")
        cached, current = meta.total_k_swa, n
        global_rows = meta.total_k_compressed[mod.ratio_key]
        total = cached + current + global_rows
        kv = mod._prepare_prefill_kv(total, cached+current, meta.positions.device, cross)
        kv[cached:cached+current].copy_(swa)
        if not mod.use_shared_kv or mod.is_kv_source:
            load_paged_kv_cache(cache.ca_cache, meta.load_segments_ca[mod.ratio_key],
                               global_rows, out=kv[cached+current:])
            if mod.use_shared_kv and mod.is_kv_source:
                cross.loaded_prefill_kv = kv
                cross.loaded_prefill_kv_main_offset = cached+current
        load_paged_kv_cache(cache.swa_cache, meta.load_segments_swa, cached, out=kv[:cached])
        store_kv_cache(swa, cache.swa_cache, meta.slot_mapping_store_swa)
        if self.mode == "all_selected" and n:
            with self.allow_sync():
                restored, restored_wait = self.ops.postprocess(mod, self.ops.normalize(mod, raw), meta)
                require(restored_wait is None, "unexpected canonical restoration CP")
                self.ops.identity_error(restored, swa)
                self.ops.identity_error(kv[cached:cached+current], swa)
            self.restore_events.append(dict(layer=mod.layer_idx, chunk_start=marker["start"],
                chunk_end=marker["end"], full_rows=n, checks=2, max_absolute_error=0.0,
                checks_named=["canonical_norm_rope_cast", "assembled_swa_rows"]))
        if dense_reference and n:
            context = mod._prefill_attention(q_b, kv, global_index, meta.kv_swa_full_index, cached, current, meta)
            projected_out = mod._wv_and_o_proj(context, meta.positions).index_select(0, ids)
        elif selected_n:
            subset.kv_swa_full_index = meta.kv_swa_full_index.index_select(0, ids)
            context = mod._prefill_attention(q_b, kv, global_index.index_select(0, ids),
                                            subset.kv_swa_full_index, cached, current, subset)
            projected_out = mod._wv_and_o_proj(context, selected_positions)
        else:
            projected_out = bf16.new_empty((0, bf16.shape[-1]))
        output = bf16.new_zeros((n, bf16.shape[-1]))
        if selected_n:
            output.index_copy_(0, ids, projected_out)
        self.swa_events.append(dict(layer=mod.layer_idx, chunk_start=marker["start"], chunk_end=marker["end"],
            current_positions=meta.positions.clone(), online_local_rows=ids.clone(), online_rows=selected_n,
            reused_rows=n-selected_n, full_rows=n, compact=mod.layer_idx >= 20))
        self.attention_events.append(dict(layer=mod.layer_idx, chunk_start=marker["start"], chunk_end=marker["end"],
            full_rows=n, selected_rows=selected_n, main_q_a_rows=n if mod.has_local_index or dense_reference else selected_n,
            index_q_rows=n if mod.has_local_index else 0, main_q_b_rows=n if dense_reference else selected_n,
            joint_attention_rows=n if dense_reference else selected_n, output_projection_rows=n if dense_reference else selected_n,
            inactive_attention_zero_rows=n-selected_n, global_kv_rows=global_rows,
            global_compute_called=True, is_global_source=mod.is_kv_source, native_joint_attention=True,
            reference_dense=dense_reference, plan_sha256=plan["plan_sha256"]))
        return output

    def _index_plan(self, mod, meta, index):
        t, marker, n = self.t, meta.canonical_request_binding, meta.total_q
        ratio = mod.kv_compress_ratio
        require(ratio in (1, 2) and meta.num_reqs == 1, "single request C1/C2 index mapping required")
        require(meta.positions.dtype in (t.int32, t.int64) and meta.positions.shape == (n,),
                "native integral current position vector required")
        require(index.dtype == t.int32 and index.shape == (n, 512), "native int32 top512 matrix required")
        t._assert_async((meta.ks[mod.ratio_key] == 0).all(), "nonzero compressed request offset")
        t._assert_async(((index == -1) | ((index >= 0) & (index < meta.total_k_compressed[mod.ratio_key]))).all(),
                        "index outside complete global KV")
        t._assert_async(((index == -1) | (index < ((meta.positions+1)//ratio)[:, None])).all(),
                        "index refers to incomplete/future compressed group")
        local = (meta.positions-marker["start"]).long()
        t._assert_async(((local >= 0) & (local < marker["end"]-marker["start"])).all(), "query outside current chunk")
        prep = self.prepared[(marker["start"], marker["end"])]
        optional_mask = prep["optional_mask"].index_select(0, local)
        layout_key = (marker["start"], marker["end"], n)
        if layout_key not in self.layouts:
            with self.allow_sync():
                optional = t.nonzero(optional_mask, as_tuple=False).flatten()
                mandatory = t.nonzero(~optional_mask, as_tuple=False).flatten()
            self.layouts[layout_key] = dict(optional=optional, mandatory=mandatory, positions=meta.positions.clone())
        layout = self.layouts[layout_key]
        t._assert_async((layout["positions"] == meta.positions).all(), "native query positions changed")
        optional, mandatory = layout["optional"], layout["mandatory"]
        t._assert_async((meta.positions[1:] > meta.positions[:-1]).all(), "query positions must be strictly increasing")
        chunk_key = (marker["start"], marker["end"])
        if self.live_index_chunk != chunk_key:
            self.live_index_chunk, self.live_indices = chunk_key, []
        group_key = (marker["start"], marker["end"], id(index))
        if self.mode == "all_selected":
            ids = t.arange(n, device=meta.positions.device)
            hits = t.zeros(n, device=meta.positions.device, dtype=t.int64)
            group_id = None
        else:
            if group_key not in self.index_groups:
                columns = meta.total_k_compressed[mod.ratio_key]
                counts = t.zeros(columns, device=index.device, dtype=t.int64)
                flat = index.flatten().long()
                if columns:
                    counts.scatter_add_(0, flat.clamp(min=0), (flat >= 0).long())
                    token_keys = (meta.positions//ratio).long()
                    hits = counts.index_select(0, token_keys.clamp(max=columns-1))
                    hits = t.where(token_keys < columns, hits, t.zeros_like(hits))
                else:
                    require(n == 0, "nonempty query without global columns")
                    hits = t.empty(0, device=index.device, dtype=t.int64)
                group_id = "g" + str(len(self.index_groups))
                raw_cpu = None
                if self.mode == "reference_sparse" or self.verify_sparse:
                    require(self.index_output_dir is not None, "sparse qualification raw-index output directory required")
                    with self.allow_sync():
                        raw_cpu = index.detach().to(device="cpu", dtype=t.int32).contiguous()
                self.live_indices.append(index)
                source_layer = max(layer.layer_idx for layer in self.layers
                                   if layer.layer_idx <= mod.layer_idx and layer.self_attn.has_local_index)
                self.index_groups[group_key] = dict(group_id=group_id, tensor=raw_cpu,
                    source_layer=source_layer, chunk_start=marker["start"], chunk_end=marker["end"],
                    compression_ratio=ratio, request_kv_offset=0, rows=n, columns=512,
                    global_kv_rows=columns, hits=hits, positions=meta.positions.clone())
            group = self.index_groups[group_key]
            require(group["compression_ratio"] == ratio and group["rows"] == n, "shared index geometry changed")
            t._assert_async((group["positions"] == meta.positions).all(), "shared index query positions changed")
            hits, group_id = group["hits"], group["group_id"]
            count = (optional.numel()+4)//5
            ranking = t.argsort(hits.index_select(0, optional), descending=True, stable=True)
            ids = t.cat((mandatory, optional.index_select(0, ranking[:count]))).sort().values.contiguous()
        key = (marker["start"], marker["end"], mod.layer_idx)
        require(key not in self.plans, "duplicate per-layer index selection")
        plan = dict(ids=ids, positions=meta.positions.clone(), hits=hits, index_group_id=group_id,
                    compression_ratio=ratio, mandatory_rows=mandatory.numel(), optional_rows=optional.numel(),
                    optional_ids=optional, full_rows=n,
                    plan_sha256=digest(dict(input_sha256=self.input_sha, start=marker["start"], end=marker["end"],
                        layer=mod.layer_idx, group=group_id, mode=self.mode, fraction=.20,
                        policy="raw-index-hit-frequency-current-native-row-top20-position-tie")))
        self.plans[key] = plan
        return plan

    def _marker(self, marker):
        require(
            self.state == "RUNNING"
            and type(marker) is dict
            and set(marker) == {"request_id", "phase", "start", "end"}
            and marker["request_id"] == self.rid
            and marker["phase"] == self.phase
            and type(marker["start"]) is int
            and type(marker["end"]) is int
            and 0 <= marker["start"] < marker["end"] <= len(self.ids)
            and marker["end"] - marker["start"] <= 8192,
            "foreign request/chunk",
        )

    def _gather(self, value, ids):
        return gather_packed_token_rows(value, ids, torch_module=self.t) if isinstance(value, tuple) else value.index_select(0, ids)

    def reset(self, rid, mode, *, input_ids, documents=None, phase, fraction=None, index_device=None, verify_sparse=False):
        require(self.state in ("IDLE", "COMPLETE") and mode in MODES and self.frame is None, "unclean previous request")
        require(
            self.adapter is not None and not self.adapter.poisoned and not self.adapter.in_call,
            "failed/incomplete FFN adapter cannot start a new request",
        )
        require(
            not self.offline_full_layout and self.native_model.yoco_boundary_layer_idx == self.original_yoco_boundary,
            "previous offline layout was not restored",
        )
        require(type(rid) is str and rid and type(phase) is str and phase, "request identity required")
        ids = tuple(input_ids)
        key = token_digest(ids)
        require(
            (mode in ("sensitive", "reference_sparse") and self.qualified and fraction == .20)
            or (mode not in ("sensitive", "reference_sparse") and fraction is None),
            "fraction or qualification mismatch",
        )
        require(type(verify_sparse) is bool and (not verify_sparse or mode == "sensitive"), "invalid sparse parity mode")
        if mode == "reference_sparse":
            require(not self.sparse_reference, "sparse reference already bound")
            self.sparse_reference_input = key
        if mode == "sensitive":
            require(self.sparse_qualified or (verify_sparse and self.sparse_reference_input == key and self.sparse_reference),
                    "fresh same-policy sparse reference required before formal measurement")
        self.verify_sparse = verify_sparse
        if mode in ("qualified_native", "all_selected"):
            require(len(ids) == 16384, "two-chunk numerical qualification required")
            docs = ()
            if mode == "qualified_native":
                require(not self.reference, "qualification reference already set")
                self.reference_input = key
            else:
                require(self.reference_input == key and len(self.reference) == 80, "qualification reference absent")
        else:
            docs = document_plan(ids, documents, independent=mode == "capture_independent",
                                 natural_boundary_inputs=self.natural_boundary_inputs)
        self._bind_document_lengths(docs)
        if mode == "capture_independent":
            require(
                self.cache_output_dir is not None
                and len(self.bank) + 40 <= 320
                and all((docs[0]["key"], layer_index) not in self.bank for layer_index in CAPTURE_LAYERS),
                "cache overwrite/overflow",
            )
        self.rid, self.phase, self.mode, self.fraction = rid, phase, mode, fraction
        self.ids, self.docs, self.input_sha = ids, docs, key
        self.coverage, self.projection_checks = 0, 0
        self.projection_events = []
        self.restore_events, self.attention_events, self.sparse_observed = [], [], {}
        self.index_groups, self.live_indices, self.live_index_chunk = {}, [], None
        self.observed, self.scores, self.plans, self.layouts = {}, {}, {}, {}
        self.swa_events, self.ffn_layouts, self.prepared = [], [], {}
        self.capture_coverage = {layer_index: 0 for layer_index in CAPTURE_LAYERS}
        self.adapter.events.clear()
        for k in self.adapter.counts:
            self.adapter.counts[k] = 0
        if mode in ACTIVE_MODES:
            require(index_device is not None, "explicit device required before forward")
            if mode in ("sensitive", "reference_sparse"):
                for doc in docs:
                    for layer_index in CAPTURE_LAYERS:
                        require(
                            self._entry(doc["key"], layer_index)["tensor"].device == self.t.device(index_device), "bank device mismatch"
                        )
            for start in range(0, len(ids), 8192):
                end = min(start + 8192, len(ids))
                optional, mandatory, online = [], [], []
                for p in range(start, end):
                    doc = next((d for d in docs if d["start"] <= p < d["end"]), None)
                    (optional if doc is not None else mandatory).append(p - start)
                    if doc is None or p - doc["start"] < 127:
                        online.append(p - start)
                online_set = set(online)
                optional_set = set(optional)
                self.prepared[(start, end)] = dict(
                    optional_mask=self.t.tensor([i in optional_set for i in range(end-start)], device=index_device, dtype=self.t.bool),
                    optional=self.t.tensor(optional, device=index_device, dtype=self.t.long),
                    mandatory=self.t.tensor(mandatory, device=index_device, dtype=self.t.long),
                    online=self.t.tensor(online, device=index_device, dtype=self.t.long),
                    online_mask=self.t.tensor([i in online_set for i in range(end - start)], device=index_device, dtype=self.t.bool),
                )
        if mode == "capture_independent":
            # Offline generation needs all 32K positions at every layer. This
            # is a request-scoped layout choice, not an online model change.
            # All projections/attention/FFN and their weights remain native.
            self.native_model.yoco_boundary_layer_idx = None
            self.offline_full_layout = True
        self.state = "RUNNING"

    def get_ffn_plan(self, layer, meta, bf16):
        marker = getattr(meta, "canonical_request_binding", None)
        if not meta.is_prompt or marker is None or self.mode not in ACTIVE_MODES or layer.layer_idx < 3:
            return None
        self._marker(marker)
        key = (marker["start"], marker["end"], layer.layer_idx)
        require(key in self.plans, "attention index selection must precede FFN")
        base = self.plans[key]
        self.t._assert_async((base["positions"] == meta.positions).all(), "FFN positions differ from attention plan")
        ids = base["ids"]
        plan = SimpleNamespace(full_rows=meta.total_q, selected_count=ids.numel(), selected_rows=ids,
            current_positions=meta.positions, request_id=self.rid, phase=self.phase,
            chunk_start=marker["start"], chunk_end=marker["end"], plan_sha256=base["plan_sha256"],
            reference_dense=self.mode == "reference_sparse")
        self.ffn_layouts.append(dict(layer=layer.layer_idx, chunk_start=marker["start"], chunk_end=marker["end"],
            current_positions=meta.positions.clone(), selected_local_rows=ids.clone(), full_rows=meta.total_q,
            selected_rows=ids.numel(), plan_sha256=base["plan_sha256"]))
        return plan

    def set_index_output_dir(self, path):
        require(self.state in ("IDLE", "COMPLETE"), "cannot redirect active index evidence")
        path = Path(path)
        path.mkdir(parents=False, exist_ok=False)
        self.index_output_dir = path

    def _selected_metadata(self, layer, metadata, plan):
        return make_selected_mlp_metadata(layer, metadata, plan)

    def set_cache_output_dir(self, path):
        require(self.state in ("IDLE", "COMPLETE") and not self.bank, "clear bank before new destination")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.cache_output_dir = path

    def clear_cache(self):
        require(self.state in ("IDLE", "COMPLETE"), "cannot clear running bank")
        self.bank.clear()
        self.document_lengths.clear()
        self.cache_output_dir = None

    def _bind_document_lengths(self, documents):
        for doc in documents:
            length = doc["end"] - doc["start"]
            require(self.natural_boundary_inputs or length == 32768, "legacy document length changed")
            require(self.document_lengths.get(doc["key"], length) == length,
                    "document token identity changed length")
            self.document_lengths[doc["key"]] = length

    def _document_length(self, key):
        if not self.natural_boundary_inputs:
            return 32768
        require(key in self.document_lengths, "document length was not bound to input tokens")
        return self.document_lengths[key]

    def _entry_scope(self, key, layer):
        scope = dict(self.scope, document_sha256=key, layer=layer, representation="swa-postnorm-prerope-bf16")
        if self.natural_boundary_inputs:
            scope.update(source_document_tokens=self._document_length(key),
                         document_length_policy="natural_disjoint_token_spans")
        return scope

    def _capture(self, layer, marker, canonical):
        start, end = marker["start"], marker["end"]
        length = self._document_length(self.docs[0]["key"])
        require(
            len(self.ids) == length and 0 <= start < end <= length
            and self.capture_coverage[layer] == start and canonical.shape == (end - start, 512),
            "offline capture coverage changed",
        )
        key = (self.docs[0]["key"], layer)
        if start == 0:
            require(key not in self.bank, "immutable bank overwrite")
            self.bank[key] = dict(
                tensor=self.ops.allocate(length, canonical),
                complete=False,
                scope=self._entry_scope(key[0], layer),
            )
        entry = self.bank[key]
        entry["tensor"][start:end].copy_(canonical)
        self.capture_coverage[layer] = end
        if end == length:
            entry.update(complete=True, version=entry["tensor"]._version)

    def _entry(self, key, layer):
        value = self.bank.get((key, layer))
        require(
            value is not None
            and value["complete"]
            and value["tensor"]._version == value["version"]
            and value["scope"] == self._entry_scope(key, layer)
            and tuple(value["tensor"].shape) == (self._document_length(key), 512)
            and value["tensor"].dtype == self.t.bfloat16,
            "missing/mutated/foreign offline KV",
        )
        return value

    def _canonical(self, layer, marker, positions):
        start, end = marker["start"], marker["end"]
        full = self.t.zeros((end - start, 512), device=positions.device, dtype=self.t.bfloat16)
        for doc in self.docs:
            lo, hi = max(start, doc["start"]), min(end, doc["end"])
            if lo < hi:
                full[lo - start : hi - start].copy_(self._entry(doc["key"], layer)["tensor"][lo - doc["start"] : hi - doc["start"]])
        return full if positions.numel() == end - start else full.index_select(0, (positions - start).long())

    def finish(self):
        require(self.state == "RUNNING" and self.coverage == len(self.ids), "request did not close")
        self._restore_offline_layout()
        adapter = self.adapter.report()
        if self.mode in ("qualified_native", "all_selected"):
            require(len(self.observed) == 80, "qualification must cover all layers/chunks")
            if self.mode == "all_selected":
                require(self.observed == self.reference and self.projection_checks > 0, "split qualification missing")
        payloads = []
        if self.mode == "capture_independent":
            require(all(n == len(self.ids) for n in self.capture_coverage.values()), "offline full-layer capture incomplete")
            for layer_index in CAPTURE_LAYERS:
                entry = self._entry(self.docs[0]["key"], layer_index)
                raw = entry["tensor"].detach().contiguous().view(self.t.uint8).cpu().numpy().tobytes()
                filename = self.docs[0]["key"] + f".layer{layer_index}.bf16"
                with (self.cache_output_dir / filename).open("xb") as stream:
                    stream.write(raw)
                payloads.append(
                    dict(
                        scope=entry["scope"],
                        file=filename,
                        shape=[len(self.ids), 512],
                        dtype="bfloat16",
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
        def export(events):
            return [{k: (v.cpu().tolist() if isinstance(v, self.t.Tensor) else v) for k, v in e.items()} for e in events]
        from .index_hit_selector import select_by_hitcounts, raw_index_hitcounts
        groups = []
        for group in self.index_groups.values():
            positions = group["positions"].cpu().tolist()
            value = {k: group[k] for k in ("group_id", "source_layer", "chunk_start", "chunk_end",
                "compression_ratio", "request_kv_offset", "rows", "columns", "global_kv_rows")}
            value["query_positions"] = positions
            raw_tensor = group["tensor"]
            value["raw_indices_recorded"] = raw_tensor is not None
            value["hitcounts"] = group["hits"].cpu().tolist()
            if raw_tensor is not None:
                require(self.index_output_dir is not None, "raw index evidence destination missing")
                rows = raw_tensor.tolist()
                expected_hits = raw_index_hitcounts(rows, positions, positions, ratio=group["compression_ratio"],
                    global_kv_rows=group["global_kv_rows"], request_offset=group["request_kv_offset"])
                require(expected_hits == value["hitcounts"], "GPU index hitcount differs from CPU raw-index oracle")
                raw = raw_tensor.numpy().astype("<i4", copy=False).tobytes(order="C")
                filename = group["group_id"] + ".i32"
                with (self.index_output_dir/filename).open("xb") as handle:
                    handle.write(raw)
                value.update(file=self.index_output_dir.name + "/" + filename, bytes=len(raw),
                             sha256=hashlib.sha256(raw).hexdigest(), dtype="little-endian-int32",
                             cpu_raw_oracle_exact=True)
            groups.append(value)
        masks = []
        for (start, end, layer), value in sorted(self.plans.items()):
            ids, positions = value["ids"].cpu().tolist(), value["positions"].cpu().tolist()
            hits = value["hits"].cpu().tolist()
            expected = select_by_hitcounts(hits, positions, [(d["start"],d["end"]) for d in self.docs],
                                           all_selected=self.mode == "all_selected")
            require(ids == expected["selected_local_rows"], "GPU ranking differs from CPU oracle")
            masks.append(dict(start=start, end=end, layer=layer, full_rows=len(positions),
                current_positions=positions, selected_local_rows=ids, selected_rows=len(ids),
                selected_rows_sha256=digest(ids), hitcounts=hits, index_group_id=value["index_group_id"],
                compression_ratio=value["compression_ratio"], mandatory_rows=value["mandatory_rows"],
                optional_rows=value["optional_rows"], plan_sha256=value["plan_sha256"],
                zero_hit_document_rows=expected["zero_hit_document_rows"],
                selected_zero_hit_document_rows=expected["selected_zero_hit_document_rows"]))
        swa, ffn, attention = export(self.swa_events), export(self.ffn_layouts), list(self.attention_events)
        if self.mode in ACTIVE_MODES:
            expected = {(s,min(s+8192,len(self.ids)),i) for s in range(0,len(self.ids),8192) for i in FFN_LAYERS}
            for events in (swa, ffn, attention):
                require(len(events) == len(expected) and
                        {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in events} == expected,
                        "missing/duplicate per-layer sparse branch accounting")
            if self.mode == "all_selected":
                nonempty = {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in swa if e["full_rows"]}
                require(nonempty and {(e["chunk_start"],e["chunk_end"],e["layer"])
                        for e in self.projection_events} == nonempty and
                        {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in self.restore_events} == nonempty,
                        "nonempty split/cache identity coverage incomplete")
                self.qualified = True
        sparse_count = ((len(self.ids)+8191)//8192)*40
        if self.mode == "reference_sparse" or self.verify_sparse:
            require(len(self.sparse_observed) == sparse_count and groups
                    and all(g["raw_indices_recorded"] and g["cpu_raw_oracle_exact"] for g in groups),
                    "sparse reference index/output/mHC qualification incomplete")
        if self.verify_sparse:
            require(self.sparse_observed == self.sparse_reference and self.input_sha == self.sparse_reference_input,
                    "same-policy sparse reference input/output differs")
            self.sparse_qualified = True
        self.last_report = dict(schema="index-hit20-controller-v1", mode=self.mode, phase=self.phase,
            input_ids_sha256=self.input_sha, full_input_tokens=len(self.ids), first_three_full=True,
            actual_full_row_native_mhc=True, yoco_layout_preserved=True, ffn_layers=list(FFN_LAYERS),
            global_kv_native_online=True, decode_native=True, history_kv_rewritten=False,
            offline_payloads=payloads, selection=masks, index_groups=groups, adapter=adapter,
            swa_events=swa, ffn_layouts=ffn, attention_events=attention,
            projection_identity_checks=self.projection_checks, projection_identity_events=self.projection_events,
            cache_restore_identity_checks=sum(e["checks"] for e in self.restore_events),
            cache_restore_identity_events=self.restore_events,
            identity_exact=self.mode == "all_selected" and self.qualified,
            identity_check_count=len(self.observed),
            identity_fingerprints_sha256=digest([dict(key=list(k),value=v) for k,v in sorted(self.reference.items())])
                if self.reference else None,
            identity_fingerprints=[dict(key=list(k),value=v) for k,v in sorted(self.observed.items())],
            sparse_identity_exact=self.verify_sparse and self.sparse_qualified,
            sparse_identity_check_count=len(self.sparse_observed),
            sparse_identity_fingerprints_sha256=digest([dict(key=list(k),value=v) for k,v in sorted(self.sparse_reference.items())])
                if self.sparse_reference else None,
            sparse_identity_fingerprints=[dict(key=list(k),value=v) for k,v in sorted(self.sparse_observed.items())],
            raw_index_oracle_checked=self.mode == "reference_sparse" or self.verify_sparse,
            formal_raw_index_replay_claim=False, dynamic_mask_sync_in_online_timing=True,
            formal_admission_unchanged=True, diagnostic_reuse_not_accuracy_certified=True,
            offline_yoco_full_layout=self.mode == "capture_independent", model_boundary_restored=True,
            not_model_flops_fraction=True, selected_tokens_forced_through_yoco=False,
            empty_ep_qualification_sha256=None if self.empty_ep_capability is None else self.empty_ep_capability.sha256)
        self.live_indices.clear()
        self.state = "COMPLETE"
        return self.last_report

    def report(self):
        require(self.state == "COMPLETE", "finish before report")
        return self.last_report

    def close(self):
        if self.state == "CLOSED":
            return
        self._restore_offline_layout()
        if self.adapter is not None:
            self.adapter.close(allow_incomplete=True)
        for obj, name, old, bound in reversed(self.handles):
            require(obj.__dict__.get(name, _MISSING) is bound, "hook ownership changed")
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        self.handles.clear()
        self.bank.clear()
        self.state = "CLOSED"

    def _restore_offline_layout(self):
        if self.offline_full_layout:
            require(
                self.native_model is not None and self.native_model.yoco_boundary_layer_idx is None,
                "offline layout ownership changed; refusing foreign state overwrite",
            )
            self.native_model.yoco_boundary_layer_idx = self.original_yoco_boundary
            self.offline_full_layout = False
