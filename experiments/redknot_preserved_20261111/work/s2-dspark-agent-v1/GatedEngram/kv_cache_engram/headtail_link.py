"""Head/tail refresh with native local SWA and causal cross-document tail links.

Only native surviving head128/tail128/nondocument queries and FFNs execute.
Complete immutable independent-document SWA banks remain GPU resident. Current
layer boundary KV overlays are phase-local; missing YOCO rows keep offline KV.
First three layers, global/index computation, residual/mHC, and decode stay native.
"""
from __future__ import annotations

import copy

from .core import digest, require, token_digest
from .index_hit20 import IndexHit20Controller, CAPTURE_LAYERS, FFN_LAYERS, MODES
from .native_reuse import document_plan
from .tail_link import TailLinkController
from .headtail_link_plan import plan as memory_plan, rebind_layer

POLICY_MODES = ("reference_sparse", "sensitive")
HEAD, TAIL, DOCUMENT, WIDTH = 128, 128, 32768, 512
DYNAMIC_REFRESH_START_LAYER = 3
YOCO_KEEP_TOKENS = 140


def make_union_indices(t, native_index, current_positions, active, event):
    """Keep native columns exactly; append at most eight 128-row tail banks."""
    query = current_positions.index_select(0, active)
    native = native_index.index_select(0, active)
    out = t.full((active.numel(), event["index_width"]), -1,
                 device=query.device, dtype=t.int32)
    out[:, :128].copy_(native)
    count = len(event["tail_bank_positions"])
    if count:
        positions = t.tensor(event["tail_bank_positions"], device=query.device, dtype=query.dtype)
        ends = t.tensor(event["tail_bank_document_ends"], device=query.device, dtype=query.dtype)
        low = t.maximum(query - 127, t.full_like(query, max(
            0, event["native_valid_start"], event["native_yoco_rect_lo"])))
        low = t.minimum(low, query + 1)
        visible = (ends[None, :] <= query[:, None]) & ~(
            (positions[None, :] >= low[:, None]) & (positions[None, :] <= query[:, None]))
        slots = t.arange(count, device=query.device, dtype=t.int32) + event["tail_extra_offset"]
        out[:, 128:128 + count].copy_(t.where(visible, slots[None, :], -1))
    return out.contiguous()


def check_native_window(t, native_index, positions, event):
    """Verify the actual native mask, including effective valid-start and YOCO."""
    require(native_index.dtype == t.int32 and tuple(native_index.shape) == (positions.numel(), 128),
            "native SWA index geometry changed")
    low = t.maximum(positions - 127, t.full_like(positions, max(
        0, event["native_valid_start"], event["native_yoco_rect_lo"])))
    low = t.minimum(low, positions + 1)
    keys = low[:, None] + t.arange(128, device=positions.device, dtype=positions.dtype)[None, :]
    valid = keys <= positions[:, None]
    # Native current positions are contiguous after the optional YOCO prefix.
    current_slot = keys - positions[0] + 128
    old_slot = keys - (event["chunk_start"] - 128)
    expected = t.where(keys < event["chunk_start"], old_slot, current_slot)
    expected = t.where(valid, expected, -1).to(t.int32)
    t._assert_async((native_index == expected).all(),
                    "native SWA window/valid-start/YOCO differs from bound scalar contract")
    return expected


class HeadTailLinkController(IndexHit20Controller):
    # These helpers do not implement the old tail-only policy. They preserve
    # native global buffer sharing, bounded index transfers and ring readback.
    _joint_kv = TailLinkController._joint_kv
    _current_positions = TailLinkController._current_positions
    _upload_indices = TailLinkController._upload_indices
    _store_and_check_ring = TailLinkController._store_and_check_ring

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_yoco_swa_share_hidden = None
        self.phase_generation = 0
        self._clear_phase()

    def _verify_native_swa_binding(self):
        require(all(getattr(getattr(layer, "self_attn", None), "yoco_swa_share_hidden", None) is False
                    for layer in self.layers),
                "headtail experiment requires native_yoco_swa_share_hidden=False on all layers")

    def install(self):
        self._verify_native_swa_binding()
        result = super().install()
        self.native_yoco_swa_share_hidden = False
        return result

    def _clear_phase(self):
        self.boundaries = {}
        self.boundary_refreshed = {}
        self.boundary_chunk_end = {}
        self.headtail_swa_events = []
        self.headtail_memory_events = []
        self.boundary_restore_events = []
        self.union_oracle_events = []
        self.decode_ring_events = []
        self.ring = {}
        self.boundary_positions_cpu = []
        self.boundary_lookup = {}
        self.tail_boundary_rows = []
        self.tail_positions_cpu = []
        self.memory_layouts = {}
        self.oracle_chunk = None
        self.union_oracle_cache = {}

    def reset(self, rid, mode, *, input_ids, documents=None, phase, fraction=None,
              index_device=None, verify_sparse=False):
        require(fraction is None, "headtail policy has no index-hit fraction")
        require(self.state in ("IDLE", "COMPLETE") and self.frame is None,
                "cannot reset active/failed headtail phase")
        self._clear_phase()
        self.phase_generation += 1
        if mode not in POLICY_MODES:
            return super().reset(rid, mode, input_ids=input_ids, documents=documents,
                                 phase=phase, fraction=None, index_device=index_device,
                                 verify_sparse=verify_sparse)
        require(mode in MODES and self.adapter is not None and not self.adapter.poisoned
                and not self.adapter.in_call, "failed FFN adapter cannot reset")
        require(not self.offline_full_layout and self.native_model.yoco_boundary_layer_idx == 20,
                "native online YOCO boundary changed")
        require(type(rid) is str and rid and type(phase) is str and phase, "request identity required")
        require(self.qualified and index_device is not None, "native qualification/device required")
        require(type(verify_sparse) is bool and (not verify_sparse or mode == "sensitive"),
                "invalid sparse qualification mode")
        ids, key = tuple(input_ids), token_digest(input_ids)
        docs = document_plan(ids, documents, natural_boundary_inputs=self.natural_boundary_inputs)
        self._bind_document_lengths(docs)
        if mode == "reference_sparse":
            require(not self.sparse_reference, "sparse reference already bound")
            self.sparse_reference_input = key
        else:
            require(self.sparse_qualified or (verify_sparse and self.sparse_reference_input == key
                                            and bool(self.sparse_reference)),
                    "same-policy numerical qualification required")
        for doc in docs:
            for layer in CAPTURE_LAYERS:
                require(self._entry(doc["key"], layer)["tensor"].device == self.t.device(index_device),
                        "full immutable bank device mismatch")
        self.rid, self.phase, self.mode, self.fraction = rid, phase, mode, None
        self.ids, self.docs, self.input_sha, self.verify_sparse = ids, docs, key, verify_sparse
        self.coverage, self.projection_checks = 0, 0
        self.projection_events, self.restore_events, self.attention_events = [], [], []
        self.index_groups, self.live_indices, self.live_index_chunk = {}, [], None
        self.observed, self.plans, self.layouts, self.sparse_observed = {}, {}, {}, {}
        self.swa_events, self.ffn_layouts, self.prepared, self.scores = [], [], {}, {}
        self.capture_coverage = {layer: 0 for layer in CAPTURE_LAYERS}
        self.adapter.events.clear()
        for name in self.adapter.counts:
            self.adapter.counts[name] = 0
        self._set_boundary_positions(docs)
        self.state = "RUNNING"

    @staticmethod
    def _boundary_ranges(doc):
        length = doc["end"] - doc["start"]
        return [(0, length)] if length <= HEAD + TAIL else [(0, HEAD), (length - TAIL, length)]

    def _set_boundary_positions(self, docs):
        self.boundary_positions_cpu = [doc["start"] + p for doc in docs
            for lo, hi in self._boundary_ranges(doc) for p in range(lo, hi)]
        self.boundary_lookup = {p: i for i, p in enumerate(self.boundary_positions_cpu)}
        self.tail_positions_cpu = [p for doc in docs for p in range(max(doc["start"], doc["end"] - TAIL), doc["end"])]
        self.tail_boundary_rows = [self.boundary_lookup[p] for p in self.tail_positions_cpu]
        require(len(self.boundary_lookup) == len(self.boundary_positions_cpu), "boundary union contains duplicate rows")
        require(self.natural_boundary_inputs or (len(self.boundary_lookup) == 2048 and len(self.tail_boundary_rows) == 1024),
                "legacy eight complete document boundaries required")

    def _restored_boundaries(self, mod, meta):
        layer = mod.layer_idx
        if layer not in self.boundaries:
            canonical = self.t.cat([self._entry(doc["key"], layer)["tensor"][lo:hi]
                for doc in self.docs for lo, hi in self._boundary_ranges(doc)], dim=0)
            subset = copy.copy(meta)
            subset.total_q = len(self.boundary_positions_cpu)
            subset.positions = self._upload_indices(self.boundary_positions_cpu, meta.positions.device,
                                                     meta.positions.dtype)
            prepared, wait = self.ops.postprocess(mod, canonical, subset)
            require(wait is None and tuple(prepared.shape) == (len(self.boundary_positions_cpu), 1, WIDTH),
                    "boundary restore shape changed")
            checked = self.mode == "reference_sparse" or self.verify_sparse
            if checked:
                pieces = []
                for doc in self.docs:
                    # Independent per-document position gather, including a
                    # deduplicated overlap for documents shorter than 256.
                    for lo, hi in self._boundary_ranges(doc):
                        independent = copy.copy(meta)
                        independent.total_q = hi - lo
                        independent.positions = self.t.arange(doc["start"] + lo, doc["start"] + hi,
                            device=meta.positions.device, dtype=meta.positions.dtype)
                        value, other_wait = self.ops.postprocess(mod,
                            self._entry(doc["key"], layer)["tensor"][lo:hi], independent)
                        require(other_wait is None, "independent restore CP changed")
                        pieces.append(value)
                with self.allow_sync():
                    self.ops.identity_error(prepared, self.t.cat(pieces, dim=0))
            self.boundaries[layer] = prepared
            self.boundary_restore_events.append(dict(layer=layer, rows=len(self.boundary_positions_cpu), checked=checked,
                checks=1 if checked else 0, max_absolute_error=0.0 if checked else None,
                absolute_positions_sha256=digest(self.boundary_positions_cpu)))
        return self.boundaries[layer]

    def _refresh_boundaries(self, mod, meta, current, ids_cpu, prepared, boundaries, *, checked):
        marker, layer = meta.canonical_request_binding, mod.layer_idx
        require(layer >= 3 and self.boundary_chunk_end.get(layer, 0) == marker["start"],
                "boundary layer/chunk order changed")
        self.boundary_chunk_end[layer] = marker["end"]
        rows = [i for i, p in enumerate(current) if p in self.boundary_lookup]
        positions = [current[i] for i in rows]
        refreshed = self.boundary_refreshed.setdefault(layer, set())
        require(not refreshed.intersection(positions), "boundary KV refreshed twice in same layer/phase")
        selected_lookup = {row: i for i, row in enumerate(ids_cpu)}
        require(all(row in selected_lookup for row in rows), "boundary refresh missing selected query")
        bank_rows = [self.boundary_lookup[p] for p in positions]
        if rows:
            values = prepared.index_select(0, self._upload_indices(
                [selected_lookup[row] for row in rows], prepared.device))
            boundaries.index_copy_(0, self._upload_indices(bank_rows, boundaries.device), values)
            refreshed.update(positions)
        visible = [p for p in self.boundary_positions_cpu if p <= current[-1]]
        self.headtail_swa_events.append(dict(schema="headtail-link-dynamic-swa-v1", layer=layer,
            chunk_start=marker["start"], chunk_end=marker["end"], phase_generation=self.phase_generation,
            refresh_enabled=True, source="current_layer_attention_input", input_representation="native_packed_current",
            current_boundary_local_rows=rows, current_boundary_positions=positions,
            refreshed_local_rows=rows, refreshed_positions=positions, refreshed_bank_rows=bank_rows,
            available_refreshed_positions=sorted(refreshed),
            visible_refreshed_positions=[p for p in visible if p in refreshed],
            visible_offline_fallback_positions=[p for p in visible if p not in refreshed],
            projection_identity_checked=bool(rows) and checked, prepare_identity_checked=bool(rows) and checked,
            checks=2 if rows and checked else 0, max_absolute_error=0.0 if rows and checked else None))

    def _check_union_oracle(self, event, queries, actual):
        # Independent CPU integer oracle. Geometry is shared by encoder layers;
        # only the current chunk is cached, and every actual matrix is checked.
        chunk = event["chunk_start"], event["chunk_end"]
        if self.oracle_chunk != chunk:
            self.oracle_chunk, self.union_oracle_cache = chunk, {}
        key = event["layout_sha256"], tuple(queries)
        if key in self.union_oracle_cache:
            expected = self.union_oracle_cache[key]
            self.t._assert_async((actual == expected).all(), "GPU union differs from cached independent CPU oracle")
            return
        import numpy as np
        tails = np.asarray(event["tail_bank_positions"], dtype=np.int64)
        ends = np.asarray(event["tail_bank_document_ends"], dtype=np.int64)
        all_values = np.full((len(queries), event["index_width"]), -1, dtype=np.int32)
        for start in range(0, len(queries), 128):
            q = np.asarray(queries[start:start + 128], dtype=np.int64)
            low = np.minimum(q + 1, np.maximum(q - 127,
                max(0, event["native_valid_start"], event["native_yoco_rect_lo"])))
            keys = low[:, None] + np.arange(128)[None, :]
            native_slots = np.where(keys < event["chunk_start"], keys - event["chunk_start"] + 128,
                                    keys - event["current_positions"][0] + 128)
            native_slots = np.where(keys <= q[:, None], native_slots, -1)
            eligible = ends[None, :] <= q[:, None]
            duplicate = (tails[None, :] >= low[:, None]) & (tails[None, :] <= q[:, None])
            values = np.full((len(q), event["index_width"]), -1, dtype=np.int32)
            values[:, :128] = native_slots
            values[:, 128:128 + len(tails)] = np.where(eligible & ~duplicate,
                event["tail_extra_offset"] + np.arange(len(tails))[None, :], -1)
            all_values[start:start + len(q)] = values
        with self.allow_sync():
            expected = self.t.from_numpy(all_values).to(device=actual.device)
        self.union_oracle_cache[key] = expected
        self.t._assert_async((actual == expected).all(), "GPU local-plus-tail indices differ from scalar oracle")

    def _memory_plan(self, mod, meta, current):
        marker = meta.canonical_request_binding
        key = marker["start"], marker["end"], meta.total_q, mod.layer_idx >= 20
        if key not in self.memory_layouts:
            self.memory_layouts[key] = memory_plan(self.docs, current, marker["start"], marker["end"],
                                                  len(self.ids), layer=mod.layer_idx)
        return rebind_layer(self.memory_layouts[key], mod.layer_idx)

    def _selected_attention(self, mod, hidden, cache, meta, bf16, cross):
        if self.mode == "all_selected":
            return super()._selected_attention(mod, hidden, cache, meta, bf16, cross)
        from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
        from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import store_kv_cache
        t, n, marker = self.t, meta.total_q, meta.canonical_request_binding
        require(self.mode in POLICY_MODES and mod.use_swa and mod.has_packed_kv and mod.is_sparse
                and cross is not None and meta.num_reqs == 1 and mod.yoco_swa_share_hidden is False,
                "headtail sparse joint topology changed")
        meta.compute_attn_info(mod.max_model_len, mod.window_size, cache.ca_cache.shape[1] * mod.kv_compress_ratio)
        require(meta.total_k_swa == 128 and mod.window_size == 128, "native CP1 SWA history width changed")
        current = self._current_positions(meta, marker)
        event = self._memory_plan(mod, meta, current)
        check_native_window(t, meta.kv_swa_full_index, meta.positions, event)
        # Keep the native global projection, compression, index and sharing calls.
        shared = mod._resolve_shared_hidden(bf16, cross.shared_hidden)
        projected, index_weights = mod._project_current_hidden_inputs(bf16)
        if cross.consumer_full_metadata is not None:
            mod._store_yoco_full_kv(cross, cache)
            waiter = None
        else:
            if mod.shared_hidden_kv_proj is not None:
                require(projected is None and shared is not None, "native global source ownership")
                projected = mod.shared_hidden_kv_proj(shared)
            waiter = None if projected is None else lambda: projected
        q_a_proj, swa_proj = self.projections[mod.layer_idx]
        main_q_b_proj, index_q_b_proj = self.q_b_projections[mod.layer_idx]
        dense = self.mode == "reference_sparse"
        checked = dense or self.verify_sparse
        q_a = q_a_proj(hidden) if mod.has_local_index or dense else None
        q_a_normal = mod.q_a_norm(q_a) if q_a is not None else None
        index_q = index_q_b_proj(q_a_normal) if mod.has_local_index else None
        global_index = mod._compute_kv_ca_index(waiter, index_weights, index_q, cache, meta, cross)
        require(global_index is not None and tuple(global_index.shape) == (n, 512)
                and global_index.dtype == t.int32, "native global top512 geometry changed")
        t._assert_async((meta.ks[mod.ratio_key] == 0).all(), "nonzero compressed request offset")
        t._assert_async(((global_index == -1) | ((global_index >= 0)
            & (global_index < meta.total_k_compressed[mod.ratio_key])
            & (global_index < ((meta.positions + 1) // mod.kv_compress_ratio)[:, None]))).all(),
            "native global index refers to incomplete/future KV")
        ids_cpu = event["selected_local_rows"]
        ids = self._upload_indices(ids_cpu, meta.positions.device)
        key = marker["start"], marker["end"], mod.layer_idx
        require(key not in self.plans, "duplicate headtail layer plan")
        self.plans[key] = dict(ids=ids, positions=meta.positions.clone(), plan_sha256=event["plan_sha256"])
        selected_n = len(ids_cpu)
        subset = copy.copy(meta)
        subset.total_q, subset.positions = selected_n, meta.positions.index_select(0, ids)
        raw = swa_proj(self._gather(hidden, ids)) if selected_n else bf16.new_empty((0, WIDTH))
        if selected_n:
            updated, wait = self.native_prepares[mod.layer_idx](raw, cache.swa_cache, subset, False)
            require(wait is None and tuple(updated.shape) == (selected_n, 1, WIDTH), "selected native SWA prepare changed")
        else:
            updated = bf16.new_empty((0, 1, WIDTH))
        reference_q_b = None
        if dense or (checked and selected_n):
            full_q_a, full_raw = mod._project_q_and_swa(hidden, shared)
            if dense:
                reference_q_b, reference_index_q = mod._project_q_b(full_q_a, meta.positions)
                if mod.has_local_index:
                    with self.allow_sync():
                        self.ops.identity_error(index_q, reference_index_q)
        if checked and selected_n:
            expected_raw = full_raw.index_select(0, ids)
            expected, wait = self.ops.postprocess(mod, self.ops.normalize(mod, expected_raw), subset)
            require(wait is None, "SWA reference CP changed")
            with self.allow_sync():
                self.ops.identity_error(raw, expected_raw)
                self.ops.identity_error(updated, expected)
        canonical = self._canonical(mod.layer_idx, marker, meta.positions)
        swa, wait = self.ops.postprocess(mod, canonical, meta)
        require(wait is None and tuple(swa.shape) == (n, 1, WIDTH), "full cached SWA restoration changed")
        if selected_n:
            swa.index_copy_(0, ids, updated)
        boundaries = self._restored_boundaries(mod, meta)
        self._refresh_boundaries(mod, meta, current, ids_cpu, updated, boundaries, checked=checked)
        extra = boundaries.index_select(0, self._upload_indices(self.tail_boundary_rows, boundaries.device))
        memory = swa.new_empty((event["global_index_offset"], 1, WIDTH))
        # Load before store: the physical native SWA ring is overwritten below.
        load_paged_kv_cache(cache.swa_cache, meta.load_segments_swa, 128, out=memory[:128])
        memory[128:128 + n].copy_(swa)
        memory[128 + n:].copy_(extra)
        joint = self._joint_kv(mod, memory, cache, meta, cross, load_paged_kv_cache)
        all_rows = t.arange(n, device=meta.positions.device)
        self._store_and_check_ring(mod, cache, meta, current, all_rows, swa, store_kv_cache, load_paged_kv_cache)
        active = all_rows if dense else ids
        queries = current if dense else event["selected_positions"]
        query_positions = meta.positions.index_select(0, active)
        with self.allow_sync():
            indices = make_union_indices(t, meta.kv_swa_full_index, meta.positions, active, event)
        if checked:
            self._check_union_oracle(event, queries, indices)
        if active.numel():
            if dense:
                q_b = reference_q_b
            else:
                active_q_a = self._gather(q_a_normal, active) if q_a_normal is not None else mod.q_a_norm(
                    q_a_proj(self._gather(hidden, active)))
                q_b = main_q_b_proj(active_q_a).unflatten(-1, (mod.num_heads, mod.qk_head_dim))
                q_b = mod._q_b_norm_rope(q_b, query_positions)
            attention_meta = copy.copy(meta)
            attention_meta.total_q, attention_meta.positions = active.numel(), query_positions
            context = mod._prefill_attention(q_b, joint, global_index.index_select(0, active).contiguous(),
                indices, 128, n + len(self.tail_positions_cpu), attention_meta)
            projected_out = mod._wv_and_o_proj(context, query_positions)
            if dense:
                projected_out = projected_out.index_select(0, ids)
        else:
            projected_out = bf16.new_empty((0, bf16.shape[-1]))
        output = bf16.new_zeros((n, bf16.shape[-1]))
        if selected_n:
            output.index_copy_(0, ids, projected_out)
        event.update(query_positions=queries, actual_joint_query_rows=len(queries),
            reference_dense=dense, native_cache_store_rows=n, global_source_is_native=True,
            tail_slots_absolute_positions_sha256=digest(self.tail_positions_cpu),
            gpu_native_window_checked=True, gpu_union_oracle_checked=checked,
            native_valid_start_binding="native_indices_verified_zero_start")
        self.headtail_memory_events.append(event)
        if checked:
            self.union_oracle_events.append(dict(layer=mod.layer_idx, chunk_start=marker["start"],
                chunk_end=marker["end"], queries=len(queries), index_width=event["index_width"], checks=1))
        self.swa_events.append(dict(layer=mod.layer_idx, chunk_start=marker["start"], chunk_end=marker["end"],
            current_positions=meta.positions.clone(), online_local_rows=ids.clone(), online_rows=selected_n,
            reused_rows=n - selected_n, full_rows=n, compact=mod.layer_idx >= 20,
            native_cache_store_rows=n, head_rows=len(event["document_head_local_rows"]),
            tail_rows=len(event["document_tail_local_rows"]), global_index_offset=event["global_index_offset"]))
        self.attention_events.append(dict(layer=mod.layer_idx, chunk_start=marker["start"], chunk_end=marker["end"],
            full_rows=n, selected_rows=selected_n, main_q_a_rows=n if mod.has_local_index or dense else selected_n,
            index_q_rows=n if mod.has_local_index else 0, main_q_b_rows=n if dense else selected_n,
            joint_attention_rows=n if dense else selected_n, output_projection_rows=n if dense else selected_n,
            inactive_attention_zero_rows=n - selected_n, global_kv_rows=meta.total_k_compressed[mod.ratio_key],
            global_compute_called=True, is_global_source=mod.is_kv_source, native_joint_attention=True,
            native_local_window_preserved=True, extra_completed_document_tail_links=True,
            reference_dense=dense, plan_sha256=event["plan_sha256"]))
        return output

    def _report_fields(self, active):
        entries = len(self.bank)
        bank_rows = sum(self._document_length(key[0]) for key in self.bank)
        return dict(schema="headtail-link-controller-v1", head_tokens=HEAD, tail_tokens=TAIL,
            dynamic_refresh_start_layer=3, dynamic_document_swa=active,
            native_yoco_swa_share_hidden=self.native_yoco_swa_share_hidden,
            native_local_window_preserved=True, native_local_mask_preserved=True,
            extra_completed_document_tail_links=True,
            extra_links_cross_yoco_rectangle=True, native_valid_start=0, yoco_keep_tokens=140,
            offline_gpu_resident=True, offline_bank_representation="swa-postnorm-prerope-bf16",
            offline_bank_immutable=True, middle_kv_immutable=True, offline_bank_entry_count=entries,
            offline_bank_rows=bank_rows, offline_bank_bytes=bank_rows * WIDTH * 2,
            natural_boundary_inputs=self.natural_boundary_inputs,
            document_token_lengths=[d["end"] - d["start"] for d in getattr(self, "docs", ())],
            document_length_policy="natural_disjoint_token_spans" if self.natural_boundary_inputs else "fixed_32768",
            phase_generation=self.phase_generation, phase_overlay_fresh=True, cross_phase_overlay_reuse=False,
            phase_local_boundary_overlay=True,
            boundary_overlay_rows_per_layer=len(self.boundary_positions_cpu) if self.natural_boundary_inputs else 2048,
            staging=None, prefill_only_link=True, headtail_swa_events=self.headtail_swa_events,
            headtail_memory_events=self.headtail_memory_events,
            boundary_restore_identity_events=self.boundary_restore_events,
            union_oracle_events=self.union_oracle_events)

    def finish(self):
        require(self.native_yoco_swa_share_hidden is False, "native source binding was not verified")
        self._verify_native_swa_binding()
        if self.mode not in POLICY_MODES:
            report = super().finish()
            report.update(self._report_fields(False), decode_ring_positions=[])
            return report
        require(self.state == "RUNNING" and self.coverage == len(self.ids), "headtail request did not close")
        self._restore_offline_layout()
        expected = {(start, min(start + 8192, len(self.ids)), layer)
                    for start in range(0, len(self.ids), 8192) for layer in FFN_LAYERS}
        for events in (self.swa_events, self.ffn_layouts, self.attention_events,
                       self.headtail_memory_events, self.headtail_swa_events):
            require(len(events) == len(expected) and {
                (e["chunk_start"], e["chunk_end"], e["layer"]) for e in events} == expected,
                "incomplete per-layer/chunk headtail accounting")
        require({e["layer"] for e in self.decode_ring_events} == set(FFN_LAYERS), "decode ring closure incomplete")
        require(all(e["phase_generation"] == self.phase_generation for e in self.headtail_swa_events),
                "stale boundary overlay phase")
        checked = self.mode == "reference_sparse" or self.verify_sparse
        if checked:
            require(len(self.sparse_observed) == ((len(self.ids) + 8191) // 8192) * 40
                    and len(self.union_oracle_events) == len(expected)
                    and {e["layer"] for e in self.boundary_restore_events} == set(FFN_LAYERS)
                    and all(e["checked"] for e in self.boundary_restore_events),
                    "headtail same-policy qualification incomplete")
        if self.verify_sparse:
            require(self.sparse_reference_input == self.input_sha and self.sparse_observed == self.sparse_reference,
                    "same-policy output/mHC differs")
            self.sparse_qualified = True
        def export(events):
            return [{k: (v.cpu().tolist() if isinstance(v, self.t.Tensor) else v) for k, v in e.items()} for e in events]
        selection = [dict(start=e["chunk_start"], end=e["chunk_end"], layer=e["layer"], full_rows=e["full_rows"],
            current_positions=e["current_positions"], selected_local_rows=e["selected_local_rows"],
            selected_positions=e["selected_positions"], selected_rows=e["selected_rows"], plan_sha256=e["plan_sha256"])
            for e in self.headtail_memory_events]
        self.last_report = dict(mode=self.mode, phase=self.phase, input_ids_sha256=self.input_sha,
            full_input_tokens=len(self.ids), first_three_full=True, actual_full_row_native_mhc=True,
            yoco_layout_preserved=True, ffn_layers=list(FFN_LAYERS), global_kv_native_online=True,
            decode_native=True, history_kv_rewritten=False, offline_payloads=[], selection=selection,
            index_groups=[], adapter=self.adapter.report(), swa_events=export(self.swa_events),
            ffn_layouts=export(self.ffn_layouts), attention_events=self.attention_events,
            decode_ring_positions=export(self.decode_ring_events), identity_exact=False, identity_check_count=0,
            identity_fingerprints=[], projection_identity_checks=0, projection_identity_events=[],
            cache_restore_identity_checks=0, cache_restore_identity_events=[],
            sparse_identity_exact=self.verify_sparse and self.sparse_qualified,
            sparse_identity_check_count=len(self.sparse_observed),
            sparse_identity_fingerprints=[dict(key=list(k), value=v) for k, v in sorted(self.sparse_observed.items())],
            sparse_identity_fingerprints_sha256=digest([dict(key=list(k), value=v)
                for k, v in sorted(self.sparse_reference.items())]) if self.sparse_reference else None,
            raw_index_oracle_checked=False, formal_raw_index_replay_claim=False,
            same_policy_causal_oracle_checked=checked, dynamic_mask_sync_in_online_timing=True,
            formal_admission_unchanged=True, diagnostic_reuse_not_accuracy_certified=True,
            offline_yoco_full_layout=False, model_boundary_restored=True, not_model_flops_fraction=True,
            selected_tokens_forced_through_yoco=False,
            empty_ep_qualification_sha256=None if self.empty_ep_capability is None else self.empty_ep_capability.sha256,
            **self._report_fields(True))
        self.state = "COMPLETE"
        return self.last_report

    def clear_cache(self):
        require(self.state in ("IDLE", "COMPLETE"), "cannot clear active bank")
        self._clear_phase()
        return super().clear_cache()

    def close(self):
        self._clear_phase()
        return super().close()
