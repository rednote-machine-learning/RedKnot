"""Independent tail128 prefill memory; native first three layers and decode.

CPU-only import. This is a changed attention set, not native-equivalent SWA.
Only actual native/YOCO surviving tail or nondocument queries execute the main
attention/O/FFN branches. Full global/index preparation and native mHC remain.
The native all-selected control is intentionally separate from this policy.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from .core import digest, require, token_digest
from .index_hit20 import IndexHit20Controller, CAPTURE_LAYERS, FFN_LAYERS, MODES
from .native_reuse import document_plan
from .tail_link_plan import plan as memory_plan, tail_bank_rows, audit_plan_result
from .tail_link_staging import TailStaging

REPRESENTATION = "swa-tail128-postnorm-prerope-bf16"
TAIL, DOCUMENT, WIDTH = 128, 32768, 512
POLICY_MODES = ("reference_sparse", "sensitive")
DYNAMIC_REFRESH_START_LAYER = 4


def tail_scope(scope, document_sha256, layer):
    return dict(scope, document_sha256=document_sha256, layer=layer,
                representation=REPRESENTATION, source_local_start=DOCUMENT-TAIL,
                source_document_tokens=DOCUMENT, tail_tokens=TAIL)


def padded_memory_width(rows):
    require(type(rows) is int and rows >= 0, "nonnegative memory rows required")
    return max(128, ((rows+127)//128)*128)


def make_memory_indices(t, memory_positions, query_positions):
    """Actual GPU causal index construction; no per-query CPU mask transfers."""
    count = memory_positions.numel()
    width = padded_memory_width(count)
    columns = t.arange(width, device=query_positions.device, dtype=t.int32)
    if count:
        padded = t.empty(width, device=query_positions.device, dtype=memory_positions.dtype)
        padded[:count].copy_(memory_positions)
        padded[count:].fill_(2**31-1)
        visible = (columns[None, :] < count) & (padded[None, :] <= query_positions[:, None])
    else:
        visible = t.zeros((query_positions.numel(), width), device=query_positions.device, dtype=t.bool)
    return t.where(visible, columns[None, :], -1).to(t.int32).contiguous()


class TailLinkController(IndexHit20Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_generation = 0
        self.staging = None
        self.restored = {}
        self.dynamic_document_swa = False
        self.dynamic_swa_events = []
        self.dynamic_refreshed = {}
        self.dynamic_chunk_end = {}
        self.policy_qualifications = {}
        self.native_yoco_swa_share_hidden = None

    def _verify_native_swa_binding(self):
        # This experiment binds the actual HF parser/checkpoint configuration,
        # not a guess from the YOCO compaction boundary. The frozen native
        # configuration uses current-input SWA projections in all forty layers.
        require(all(getattr(getattr(layer, "self_attn", None),
                            "yoco_swa_share_hidden", None) is False for layer in self.layers),
                "fixed experiment requires native_yoco_swa_share_hidden=False on every layer")

    def install(self):
        self._verify_native_swa_binding()
        result = super().install()
        self.native_yoco_swa_share_hidden = False
        return result

    def _bind_swa_policy(self, dynamic):
        """Dense/sparse qualification is independent for the two policies."""
        require(type(dynamic) is bool, "dynamic_document_swa must be a bool")
        self.policy_qualifications[self.dynamic_document_swa] = dict(
            reference=self.sparse_reference, input=self.sparse_reference_input,
            qualified=self.sparse_qualified)
        selected = self.policy_qualifications.setdefault(dynamic,
            dict(reference={}, input=None, qualified=False))
        self.sparse_reference = selected["reference"]
        self.sparse_reference_input = selected["input"]
        self.sparse_qualified = selected["qualified"]
        self.dynamic_document_swa = dynamic

    def _clear_phase_gpu(self):
        if self.staging is not None:
            # Explicit lifecycle synchronization, never a hidden forward wait.
            # reset is timed from before entry; clear/close are after requests.
            with self.allow_sync():
                self.staging.close()
        self.staging = None
        self.restored = {}
        self.nondoc = {}
        self.ring = {}
        self.dynamic_refreshed = {}
        self.dynamic_chunk_end = {}
        self.dynamic_swa_events = []

    def reset(self, rid, mode, *, input_ids, documents=None, phase, fraction=None,
              index_device=None, verify_sparse=False, dynamic_document_swa=False):
        require(fraction is None, "tail-link has no sensitive-token percentage")
        require(type(dynamic_document_swa) is bool
                and (not dynamic_document_swa or mode in POLICY_MODES),
                "dynamic document SWA is only a qualified tail policy")
        require(self.state in ("IDLE", "COMPLETE") and self.frame is None,
                "cannot clear an active/failed tail phase on reset")
        self._bind_swa_policy(dynamic_document_swa)
        self._clear_phase_gpu()
        self.phase_generation += 1
        self.tail_memory_events, self.tail_restore_events = [], []
        self.memory_assembly_events, self.decode_ring_events = [], []
        self.cpu_plan_oracles = set()
        if mode in ("qualified_native", "all_selected", "native"):
            return super().reset(rid, mode, input_ids=input_ids, documents=documents,
                                 phase=phase, fraction=None, index_device=index_device,
                                 verify_sparse=verify_sparse)
        require(self.state in ("IDLE", "COMPLETE") and mode in MODES and self.frame is None,
                "unclean tail-link request")
        require(self.adapter is not None and not self.adapter.poisoned and not self.adapter.in_call,
                "failed FFN adapter cannot reset")
        require(not self.offline_full_layout and self.native_model.yoco_boundary_layer_idx == 20,
                "native online YOCO boundary changed")
        require(type(rid) is str and rid and type(phase) is str and phase, "request identity required")
        ids, key = tuple(input_ids), token_digest(input_ids)
        docs = document_plan(ids, documents, independent=mode == "capture_independent")
        require(type(verify_sparse) is bool and (not verify_sparse or mode == "sensitive"),
                "invalid same-policy verification mode")
        if mode in POLICY_MODES:
            require(self.qualified and index_device is not None, "native qualification/device required")
            if mode == "reference_sparse":
                require(not self.sparse_reference, "sparse reference already bound")
                self.sparse_reference_input = key
            else:
                require(self.sparse_qualified or (verify_sparse and self.sparse_reference_input == key
                                                  and bool(self.sparse_reference)),
                        "fresh same-policy qualification required")
            for doc in docs:
                for layer in CAPTURE_LAYERS:
                    self._entry(doc["key"], layer)
        else:
            require(mode == "capture_independent" and self.cache_output_dir is not None
                    and len(self.bank)+40 <= 320
                    and all((docs[0]["key"], layer) not in self.bank for layer in CAPTURE_LAYERS),
                    "tail bank overwrite/overflow")
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
        if mode == "capture_independent":
            self.native_model.yoco_boundary_layer_idx = None
            self.offline_full_layout = True
        else:
            rows = tail_bank_rows(docs, len(ids))
            require(len(rows) == 1024, "exact eight complete document tails required")
            self.tail_positions_cpu = [row["position"] for row in rows]
            self.tail_positions = self.t.tensor(self.tail_positions_cpu, device=index_device, dtype=self.t.int32)
            banks = {}
            for layer in FFN_LAYERS:
                with self.t.inference_mode(False), self.t.no_grad():
                    packed = self.t.empty((1024,512), device="cpu", dtype=self.t.bfloat16, pin_memory=True)
                for ordinal, doc in enumerate(docs):
                    packed[ordinal*128:(ordinal+1)*128].copy_(self._entry(doc["key"], layer)["tensor"])
                banks[layer] = packed
            self.staging = TailStaging(self.t, banks, device=index_device, generation=self.phase_generation)
        self.state = "RUNNING"

    def _capture(self, layer, marker, canonical):
        start, end = marker["start"], marker["end"]
        require(len(self.ids) == DOCUMENT and self.capture_coverage[layer] == start
                and canonical.shape == (end-start, WIDTH), "offline native full-document coverage changed")
        self.capture_coverage[layer] = end
        if end != DOCUMENT:
            return
        key = (self.docs[0]["key"], layer)
        require(key not in self.bank and start <= DOCUMENT-TAIL, "tail capture overwrite or incomplete tail")
        with self.t.inference_mode(False), self.t.no_grad():
            cpu = self.t.empty((TAIL,WIDTH), dtype=self.t.bfloat16, device="cpu", pin_memory=True)
        # Explicit offline-only D2H. No full document canonical tensor is retained.
        with self.allow_sync():
            cpu.copy_(canonical[DOCUMENT-TAIL-start:])
        self.bank[key] = dict(tensor=cpu, complete=True, version=cpu._version,
                              scope=tail_scope(self.scope, key[0], layer))

    def _entry(self, key, layer):
        value = self.bank.get((key, layer))
        require(value is not None and value["complete"]
                and value["tensor"]._version == value["version"]
                and value["scope"] == tail_scope(self.scope, key, layer)
                and tuple(value["tensor"].shape) == (TAIL,WIDTH)
                and value["tensor"].dtype == self.t.bfloat16
                and value["tensor"].device.type == "cpu" and value["tensor"].is_pinned(),
                "missing/mutated/nonpinned/foreign tail bank")
        return value

    def _current_positions(self, meta, marker):
        key = marker["start"], marker["end"], meta.total_q
        if key not in self.layouts:
            # One bounded native-position transfer per full/YOCO chunk layout;
            # this synchronization is explicitly inside online TTFT.
            with self.allow_sync():
                positions = meta.positions.detach().cpu().tolist()
            require(len(positions) == meta.total_q and positions == sorted(set(positions))
                    and all(marker["start"] <= p < marker["end"] for p in positions),
                    "native position layout invalid")
            self.layouts[key] = (positions, meta.positions.clone())
        positions, original = self.layouts[key]
        self.t._assert_async((original == meta.positions).all(), "native layout changed within chunk")
        return positions

    def _doc_ordinal(self, position):
        return next((i for i, doc in enumerate(self.docs) if doc["start"] <= position < doc["end"]), -1)

    def _upload_indices(self, values, device, dtype=None):
        # Deliberate, bounded host-index transfers are part of online TTFT.
        # Do not disable strict-sync-debug for native projection/attention/EP.
        # torch.tensor(host, device=CUDA) may synchronize on some builds.
        with self.allow_sync():
            return self.t.tensor(values, device=device, dtype=self.t.long if dtype is None else dtype)

    def _restored_tails(self, mod, meta):
        if mod.layer_idx not in self.restored:
            canonical = self.staging.acquire(mod.layer_idx)
            tail_meta = copy.copy(meta)
            tail_meta.total_q, tail_meta.positions = 1024, self.tail_positions
            restored, wait = self.ops.postprocess(mod, canonical, tail_meta)
            require(wait is None and restored.shape == (1024,1,512), "tail restore shape/CP changed")
            if self.mode == "reference_sparse" or self.verify_sparse:
                with self.allow_sync():
                    independent = self.t.cat([self._entry(doc["key"], mod.layer_idx)["tensor"].to(meta.positions.device)
                                              for doc in self.docs], dim=0)
                    self.ops.identity_error(canonical, independent)
                    # Independent per-document transform with independently
                    # constructed absolute target positions, not compact slots.
                    pieces = []
                    for ordinal, doc in enumerate(self.docs):
                        abs_pos = self.t.arange(doc["end"]-128, doc["end"], device=meta.positions.device,
                                                dtype=meta.positions.dtype)
                        value = self.ops.transform(independent[ordinal*128:(ordinal+1)*128], norm_args=None,
                            rope_args=self.ops.RopeArgs(abs_pos, mod.rotary_emb.cos_sin_cache, 64,
                                                       mod.rotary_emb.is_neox_style, offset=448),
                            cast_args=self.ops.CastArgs("e4m3", mod.swa_cache_num_per_channels,
                                                        round_sf=True, sf_format="ue8m0", cast_back=True),
                            out_args=None)
                        pieces.append(value.reshape(128,1,512))
                    self.ops.identity_error(restored, self.t.cat(pieces, dim=0))
                self.tail_restore_events.append(dict(layer=mod.layer_idx, checks=2,
                    checks_named=["dma_bytes", "absolute_rope_nativecast"], max_absolute_error=0.0,
                    absolute_positions_sha256=digest(self.tail_positions_cpu), rows=1024))
            self.restored[mod.layer_idx] = restored
            self.staging.release(mod.layer_idx)
        return self.restored[mod.layer_idx]

    def _refresh_document_tails(self, mod, hidden, bf16, shared, cache, meta, current, tails):
        """Overlay current-layer KV only for tails that really survived YOCO.

        Layer 3 is the unchanged first-link seed. Layers 4..39 consume this
        layer's attention-normalized INPUT, hence the previous layer's updated
        state. The online tail buffer is per layer and phase: earlier chunks'
        same-layer KV survives, while unavailable YOCO rows retain that layer's
        offline value. No earlier layer or future query result is substituted.
        """
        if not self.dynamic_document_swa:
            return []
        marker, layer, t = meta.canonical_request_binding, mod.layer_idx, self.t
        require(layer in FFN_LAYERS, "dynamic SWA cannot modify first three layers")
        prior_end = self.dynamic_chunk_end.get(layer, 0)
        require(prior_end == marker["start"],
                "dynamic SWA layer chunks missing, duplicated or out of order")
        self.dynamic_chunk_end[layer] = marker["end"]
        lookup = {position: i for i, position in enumerate(self.tail_positions_cpu)}
        current_rows = [i for i, position in enumerate(current) if position in lookup]
        enabled = layer >= DYNAMIC_REFRESH_START_LAYER
        rows = current_rows if enabled else []
        positions = [current[i] for i in rows]
        bank_rows = [lookup[position] for position in positions]
        refreshed = self.dynamic_refreshed.setdefault(layer, set())
        require(not refreshed.intersection(positions), "document tail KV rewritten twice in one layer/phase")
        checked = bool(rows) and (self.mode == "reference_sparse" or self.verify_sparse)
        if rows:
            ids = self._upload_indices(rows, meta.positions.device)
            # The native YOCO SWA projection otherwise uses a fixed boundary
            # hidden. Only DOCUMENT tails opt into the current layer's BF16
            # input here; non-document SWA and all global sharing stay native.
            source = bf16 if mod.yoco_swa_share_hidden else hidden
            require(source is not None and self.ops.rows(source) == meta.total_q,
                    "current-layer document SWA input row geometry changed")
            raw = self.projections[layer][1](self._gather(source, ids))
            subset = copy.copy(meta)
            subset.total_q, subset.positions = len(rows), meta.positions.index_select(0, ids)
            value, wait = self.native_prepares[layer](raw, cache.swa_cache, subset, False)
            require(wait is None and value.shape == (len(rows), 1, WIDTH),
                    "dynamic document SWA prepare shape/CP changed")
            if checked:
                # Independent full native projection followed by row selection,
                # then canonical norm + explicit postprocess instead of the
                # subset native fused preparation used by the actual overlay.
                _, full_raw = mod._project_q_and_swa(hidden, bf16 if mod.yoco_swa_share_hidden else shared)
                expected_raw = full_raw.index_select(0, ids)
                with self.allow_sync():
                    self.ops.identity_error(raw, expected_raw)
                    expected, expected_wait = self.ops.postprocess(mod, self.ops.normalize(mod, expected_raw), subset)
                    require(expected_wait is None, "dynamic reference CP changed")
                    self.ops.identity_error(value, expected)
            tails.index_copy_(0, self._upload_indices(bank_rows, tails.device), value)
            refreshed.update(positions)
        last_query = current[-1] if current else -1
        visible = [position for position in self.tail_positions_cpu if position <= last_query]
        self.dynamic_swa_events.append(dict(
            schema="tail-link-dynamic-swa-v1", layer=layer,
            chunk_start=marker["start"], chunk_end=marker["end"],
            phase_generation=self.phase_generation, refresh_enabled=enabled,
            source="current_layer_attention_input" if enabled else "offline_layer3_seed",
            input_representation="bf16_current" if mod.yoco_swa_share_hidden else "native_packed_current",
            current_tail_local_rows=current_rows, current_tail_positions=[current[i] for i in current_rows],
            refreshed_local_rows=rows, refreshed_positions=positions, refreshed_bank_rows=bank_rows,
            available_refreshed_positions=sorted(refreshed),
            visible_refreshed_positions=[position for position in visible if position in refreshed],
            visible_offline_fallback_positions=[position for position in visible if position not in refreshed],
            projection_identity_checked=checked, prepare_identity_checked=checked,
            checks=2 if checked else 0, max_absolute_error=0.0 if checked else None))
        return rows

    def _joint_kv(self, mod, memory, cache, meta, cross, load):
        rows, offset = meta.total_k_compressed[mod.ratio_key], memory.shape[0]
        # Layer3 is a consumer of native layer2. Its source buffer still has
        # native full-SWA layout; migrate only that already-computed global
        # slice when the compact tail prefix changes. Never rerun compression.
        if mod.is_kv_source or not mod.use_shared_kv:
            joint = memory.new_empty((offset+rows,1,512))
            load(cache.ca_cache, meta.load_segments_ca[mod.ratio_key], rows, out=joint[offset:])
        else:
            old, old_offset = cross.loaded_prefill_kv, cross.loaded_prefill_kv_main_offset
            require(old is not None and old.shape[0]-old_offset == rows, "native shared global cache absent")
            if old.shape[0] == offset+rows and old_offset == offset:
                joint = old
            else:
                joint = memory.new_empty((offset+rows,1,512))
                joint[offset:].copy_(old[old_offset:])
        joint[:offset].copy_(memory)
        if mod.use_shared_kv:
            cross.loaded_prefill_kv = joint
            cross.loaded_prefill_kv_main_offset = offset
        return joint

    def _assemble_memory(self, event, tails, nondoc_positions, nondoc_tensor):
        t, count = self.t, event["memory_count"]
        out = tails.new_empty((count,1,512))
        doc_dst, doc_src, other_dst, other_src = [], [], [], []
        non_lookup = {position:i for i, position in enumerate(nondoc_positions)}
        for row, (position, ordinal, bankrow) in enumerate(zip(event["memory_positions"],
                event["memory_doc_indices"], event["memory_tail_bank_rows"])):
            if ordinal >= 0:
                doc_dst.append(row); doc_src.append(ordinal*128+bankrow)
            else:
                require(position in non_lookup, "unmaterialized nondocument KV")
                other_dst.append(row); other_src.append(non_lookup[position])
        def idx(values): return self._upload_indices(values, tails.device)
        if doc_dst:
            out.index_copy_(0, idx(doc_dst), tails.index_select(0, idx(doc_src)))
        if other_dst:
            out.index_copy_(0, idx(other_dst), nondoc_tensor.index_select(0, idx(other_src)))
        if self.mode == "reference_sparse" or self.verify_sparse:
            # Different assembly algorithm: resolve absolute source positions
            # independently, then stack rows. Only qualification pays this cost.
            by_position = {position:tails[i] for i, position in enumerate(self.tail_positions_cpu)}
            by_position.update({position:nondoc_tensor[i] for i, position in enumerate(nondoc_positions)})
            expected = t.stack([by_position[p] for p in event["memory_positions"]]) if count else out.clone()
            if count:
                with self.allow_sync(): self.ops.identity_error(out, expected)
            self.memory_assembly_events.append(dict(schema="tail-link-memory-assembly-identity-v1",
                layer=event["layer"], chunk_start=event["chunk_start"], chunk_end=event["chunk_end"],
                checks=1, max_absolute_error=0.0, rows=count))
        return out

    def _store_and_check_ring(self, mod, cache, meta, positions, selected, values, store, load):
        t = self.t
        slots = meta.slot_mapping_store_swa.index_select(0, selected).contiguous()
        store(values, cache.swa_cache, slots)
        needed = set(range(max(0,len(self.ids)-128),len(self.ids)))
        ring = self.ring.setdefault(mod.layer_idx, {})
        for local, position in enumerate(positions):
            if position in needed:
                require(position not in ring, "duplicate final decode ring position")
                ring[position] = (slots[local].clone(), values[local].clone())
        if meta.canonical_request_binding["end"] == len(self.ids):
            require(set(ring) == needed, "native decode window has missing prompt KV")
            ordered = sorted(ring)
            selected_slots = t.stack([ring[p][0] for p in ordered]).to(t.int32)
            expected = t.stack([ring[p][1] for p in ordered])
            t._assert_async((selected_slots >= 0).all(), "required decode KV mapped to padding")
            # Every prompt KV in the final ring must own a different slot.
            t._assert_async((selected_slots[:,None] != selected_slots[None,:]).sum() == len(ordered)*(len(ordered)-1),
                            "decode ring has colliding physical slots")
            block = cache.swa_cache.shape[1]
            segments = t.stack((selected_slots//block, selected_slots%block,
                                 selected_slots%block+1,
                                 t.arange(len(ordered), device=selected_slots.device, dtype=t.int32)), dim=1).contiguous()
            actual = expected.new_empty(expected.shape)
            load(cache.swa_cache, segments, len(ordered), out=actual)
            t._assert_async((actual == expected).all() & t.isfinite(actual).all(), "decode ring byte/value readback differs")
            self.decode_ring_events.append(dict(layer=mod.layer_idx, positions=ordered,
                physical_slots=selected_slots, physical_readback_exact=True, checks=len(ordered)))

    def _selected_attention(self, mod, hidden, cache, meta, bf16, cross):
        if self.mode == "all_selected":
            return super()._selected_attention(mod, hidden, cache, meta, bf16, cross)
        from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
        from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import store_kv_cache
        t, n, marker = self.t, meta.total_q, meta.canonical_request_binding
        require(self.mode in POLICY_MODES and mod.use_swa and mod.has_packed_kv and mod.is_sparse
                and cross is not None and meta.num_reqs == 1, "owned sparse joint tail-link topology required")
        meta.compute_attn_info(mod.max_model_len, mod.window_size,
                               cache.ca_cache.shape[1]*mod.kv_compress_ratio)
        current = self._current_positions(meta, marker)
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
        q_a = q_a_proj(hidden) if mod.has_local_index or dense else None
        q_a_normal = mod.q_a_norm(q_a) if q_a is not None else None
        index_q = index_q_b_proj(q_a_normal) if mod.has_local_index else None
        global_index = mod._compute_kv_ca_index(waiter,index_weights,index_q,cache,meta,cross)
        require(global_index is not None and global_index.shape == (n,512) and global_index.dtype == t.int32,
                "native global top512 layout changed")
        t._assert_async((meta.ks[mod.ratio_key] == 0).all(), "nonzero compressed request offset")
        t._assert_async(((global_index == -1) | ((global_index >= 0)
            & (global_index < meta.total_k_compressed[mod.ratio_key])
            & (global_index < ((meta.positions+1)//mod.kv_compress_ratio)[:,None]))).all(),
            "native global index refers to future/compression-incomplete KV")
        tails = self._restored_tails(mod, meta)
        dynamic_rows_cpu = self._refresh_document_tails(mod, hidden, bf16, shared, cache, meta, current, tails)
        non_ids_cpu = [i for i,p in enumerate(current) if self._doc_ordinal(p) < 0]
        non_ids = self._upload_indices(non_ids_cpu, meta.positions.device)
        non_positions = [current[i] for i in non_ids_cpu]
        old_positions, old_values = self.nondoc.get(mod.layer_idx, ([], tails.new_empty((0,1,512))))
        if non_ids_cpu:
            source = shared if mod.yoco_swa_share_hidden else hidden
            require(source is not None and self.ops.rows(source) == n, "native online SWA hidden rows changed")
            raw = swa_proj(self._gather(source, non_ids))
            non_meta = copy.copy(meta)
            non_meta.total_q, non_meta.positions = len(non_ids_cpu), meta.positions.index_select(0,non_ids)
            online, wait = self.native_prepares[mod.layer_idx](raw,cache.swa_cache,non_meta,False)
            require(wait is None, "unexpected non-document CP gather")
            require(not old_positions or not non_positions or old_positions[-1] < non_positions[0],
                    "nondocument KV overwritten/out of order")
            available = old_positions + non_positions
            online_all = t.cat((old_values,online),dim=0)
        else:
            available, online_all = old_positions, old_values
        self.nondoc[mod.layer_idx] = (available,online_all)
        event = memory_plan(self.docs,current,marker["start"],marker["end"],len(self.ids),
                            available_nondoc_positions=available)
        if (self.mode == "reference_sparse" or self.verify_sparse) and event["plan_sha256"] not in self.cpu_plan_oracles:
            # Independent scalar span oracle before formal admission. Identical
            # layouts across layers share this immutable CPU verification only;
            # every actual GPU index matrix is still checked below.
            checked = audit_plan_result(event,self.docs,current,marker["start"],marker["end"],len(self.ids),
                                        available_nondoc_positions=available)
            require(checked["passed"], "independent tail-memory plan oracle failed: "+repr(checked["errors"]))
            self.cpu_plan_oracles.add(event["plan_sha256"])
        event["layer"] = mod.layer_idx
        ids = self._upload_indices(event["selected_local_rows"],meta.positions.device)
        self.plans[(marker["start"],marker["end"],mod.layer_idx)] = dict(ids=ids,
            positions=meta.positions.clone(), plan_sha256=event["plan_sha256"], event=event)
        selected_n = len(event["selected_local_rows"])
        active = t.arange(n,device=meta.positions.device) if dense else ids
        query_positions = meta.positions if dense else meta.positions.index_select(0,ids)
        if active.numel():
            selected_q_a = self._gather(q_a_normal,active) if q_a_normal is not None else mod.q_a_norm(q_a_proj(self._gather(hidden,active)))
            q_b = main_q_b_proj(selected_q_a).unflatten(-1,(mod.num_heads,mod.qk_head_dim))
            q_b = mod._q_b_norm_rope(q_b,query_positions)
        else:
            q_b = bf16.new_empty((0,mod.num_heads,mod.qk_head_dim))
        memory = self._assemble_memory(event,tails,available,online_all)
        joint = self._joint_kv(mod,memory,cache,meta,cross,load_paged_kv_cache)
        memory_positions = self._upload_indices(event["memory_positions"],meta.positions.device,meta.positions.dtype)
        indices = make_memory_indices(t,memory_positions,query_positions)
        if self.mode == "reference_sparse" or self.verify_sparse:
            expected_lengths = event["reference_memory_lengths"] if dense else event["selected_memory_lengths"]
            expected_lengths = self._upload_indices(expected_lengths,indices.device,t.int32)
            t._assert_async(((indices >= 0).sum(dim=1) == expected_lengths).all(), "causal GPU index/oracle length differs")
            cols = t.arange(indices.shape[1],device=indices.device,dtype=t.int32)
            t._assert_async((indices == t.where(cols[None,:] < expected_lengths[:,None],cols[None,:],-1)).all(),
                            "causal GPU index differs from independent prefix oracle")
        subset = copy.copy(meta)
        subset.total_q, subset.positions = active.numel(), query_positions
        if active.numel():
            context = mod._prefill_attention(q_b,joint,global_index.index_select(0,active).contiguous(),indices,
                                             0,event["memory_count"],subset)
            projected_out = mod._wv_and_o_proj(context,query_positions)
            if dense:
                projected_out = projected_out.index_select(0,ids)
        else:
            projected_out = bf16.new_empty((0,bf16.shape[-1]))
        output = bf16.new_zeros((n,bf16.shape[-1]))
        if selected_n:
            output.index_copy_(0,ids,projected_out)
        lookup = {p:i for i,p in enumerate(event["memory_positions"])}
        current_memory_rows = self._upload_indices([lookup[p] for p in event["selected_positions"]],memory.device)
        self._store_and_check_ring(mod,cache,meta,event["selected_positions"],ids,
                                  memory.index_select(0,current_memory_rows),store_kv_cache,load_paged_kv_cache)
        event.update(query_positions=current if dense else event["selected_positions"],
                     query_memory_lengths=event["reference_memory_lengths"] if dense else event["selected_memory_lengths"],
                     actual_joint_query_rows=n if dense else selected_n,
                     gpu_causal_oracle_checked=self.mode == "reference_sparse" or self.verify_sparse)
        self.tail_memory_events.append(event)
        online_rows_cpu = sorted(non_ids_cpu + dynamic_rows_cpu)
        online_rows = self._upload_indices(online_rows_cpu, meta.positions.device) if dynamic_rows_cpu else non_ids
        self.swa_events.append(dict(layer=mod.layer_idx,chunk_start=marker["start"],chunk_end=marker["end"],
            current_positions=meta.positions.clone(),online_local_rows=online_rows.clone(),online_rows=len(online_rows_cpu),
            non_document_online_rows=len(non_ids_cpu),document_online_rows=len(dynamic_rows_cpu),
            document_tail_rows=len(event["document_tail_local_rows"]),
            reused_rows=len(event["document_tail_local_rows"])-len(dynamic_rows_cpu),
            full_rows=n,compact=mod.layer_idx>=20))
        self.attention_events.append(dict(layer=mod.layer_idx,chunk_start=marker["start"],chunk_end=marker["end"],
            full_rows=n,selected_rows=selected_n,main_q_a_rows=n if mod.has_local_index or dense else selected_n,
            index_q_rows=n if mod.has_local_index else 0,main_q_b_rows=n if dense else selected_n,
            joint_attention_rows=n if dense else selected_n,output_projection_rows=n if dense else selected_n,
            inactive_attention_zero_rows=n-selected_n,global_kv_rows=meta.total_k_compressed[mod.ratio_key],
            global_compute_called=True,is_global_source=mod.is_kv_source,native_joint_attention=True,
            attention_memory_changed=True,reference_dense=dense,plan_sha256=event["plan_sha256"]))
        return output

    def finish(self):
        require(self.native_yoco_swa_share_hidden is False,
                "native SWA source binding was not installed and verified")
        self._verify_native_swa_binding()
        if self.mode in ("native","qualified_native","all_selected"):
            report = super().finish()
            report.update(schema="tail-link-controller-v1",staging=None,tail_memory_events=[],
                          tail_restore_identity_events=[],memory_assembly_identity_events=[],
                          decode_ring_positions=[],prefill_only_link=True,tail_tokens=128,
                          dynamic_document_swa=False,dynamic_refresh_start_layer=None,dynamic_swa_events=[],
                          native_yoco_swa_share_hidden=self.native_yoco_swa_share_hidden)
            return report
        require(self.state == "RUNNING" and self.coverage == len(self.ids), "tail request did not close")
        self._restore_offline_layout()
        payloads = []
        if self.mode == "capture_independent":
            require(all(value == DOCUMENT for value in self.capture_coverage.values()), "offline layer coverage incomplete")
            for layer in CAPTURE_LAYERS:
                entry = self._entry(self.docs[0]["key"],layer)
                raw = entry["tensor"].detach().contiguous().view(self.t.uint8).numpy().tobytes()
                filename = self.docs[0]["key"]+f".layer{layer}.tail128.bf16"
                with (self.cache_output_dir/filename).open("xb") as handle:
                    require(handle.write(raw) == len(raw), "short tail bank write")
                payloads.append(dict(scope=entry["scope"],file=filename,shape=[128,512],dtype="bfloat16",
                                     bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        else:
            expected = {(start,min(start+8192,len(self.ids)),layer)
                        for start in range(0,len(self.ids),8192) for layer in FFN_LAYERS}
            for events in (self.swa_events,self.ffn_layouts,self.attention_events,self.tail_memory_events):
                require(len(events) == len(expected) and {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in events} == expected,
                        "incomplete tail per-layer/chunk accounting")
            require({e["layer"] for e in self.decode_ring_events} == set(FFN_LAYERS), "decode ring closure incomplete")
            require((not self.dynamic_document_swa and not self.dynamic_swa_events) or
                    (self.dynamic_document_swa and len(self.dynamic_swa_events) == len(expected)
                     and {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in self.dynamic_swa_events} == expected
                     and all(e["phase_generation"] == self.phase_generation for e in self.dynamic_swa_events)),
                    "dynamic SWA per-layer/chunk/phase accounting incomplete")
        qualification = self.mode == "reference_sparse" or self.verify_sparse
        if qualification:
            require(len(self.sparse_observed) == ((len(self.ids)+8191)//8192)*40
                    and {e["layer"] for e in self.tail_restore_events} == set(FFN_LAYERS)
                    and len(self.memory_assembly_events) == len(self.tail_memory_events)
                    and all(e["gpu_causal_oracle_checked"] for e in self.tail_memory_events),
                    "same-policy tail numerical/causal qualification incomplete")
        if self.verify_sparse:
            require(self.sparse_reference_input == self.input_sha and self.sparse_observed == self.sparse_reference,
                    "same-policy output/mHC differs")
            self.sparse_qualified = True
        def export(events):
            return [{k:(v.cpu().tolist() if isinstance(v,self.t.Tensor) else v) for k,v in e.items()} for e in events]
        staging = None if self.staging is None else self.staging.report()
        if staging is not None:
            staging.update(cpu_pinned=True,document_keys=[doc["key"] for doc in self.docs])
        selection = [dict(start=e["chunk_start"],end=e["chunk_end"],layer=e["layer"],full_rows=e["full_rows"],
                          current_positions=e["current_positions"],selected_local_rows=e["selected_local_rows"],
                          selected_positions=e["selected_positions"],selected_rows=e["selected_rows"],
                          plan_sha256=e["plan_sha256"]) for e in self.tail_memory_events]
        self.last_report = dict(schema="tail-link-controller-v1",mode=self.mode,phase=self.phase,
            input_ids_sha256=self.input_sha,full_input_tokens=len(self.ids),first_three_full=True,
            actual_full_row_native_mhc=True,yoco_layout_preserved=True,ffn_layers=list(FFN_LAYERS),
            global_kv_native_online=True,decode_native=True,prefill_only_link=True,tail_tokens=128,
            native_yoco_swa_share_hidden=self.native_yoco_swa_share_hidden,
            dynamic_document_swa=self.dynamic_document_swa,
            dynamic_refresh_start_layer=DYNAMIC_REFRESH_START_LAYER if self.dynamic_document_swa else None,
            dynamic_swa_events=self.dynamic_swa_events,
            history_kv_rewritten=False,offline_payloads=payloads,selection=selection,index_groups=[],
            adapter=self.adapter.report(),swa_events=export(self.swa_events),ffn_layouts=export(self.ffn_layouts),
            attention_events=self.attention_events,tail_memory_events=self.tail_memory_events,
            tail_restore_identity_events=self.tail_restore_events,memory_assembly_identity_events=self.memory_assembly_events,
            decode_ring_positions=export(self.decode_ring_events),staging=staging,
            projection_identity_checks=0,projection_identity_events=[],cache_restore_identity_checks=0,
            cache_restore_identity_events=[],identity_exact=False,identity_check_count=0,identity_fingerprints=[],
            sparse_identity_exact=self.verify_sparse and self.sparse_qualified,
            sparse_identity_check_count=len(self.sparse_observed),
            sparse_identity_fingerprints=[dict(key=list(k),value=v) for k,v in sorted(self.sparse_observed.items())],
            sparse_identity_fingerprints_sha256=digest([dict(key=list(k),value=v) for k,v in sorted(self.sparse_reference.items())])
                if self.sparse_reference else None,
            raw_index_oracle_checked=False,formal_raw_index_replay_claim=False,
            same_policy_causal_oracle_checked=qualification,dynamic_mask_sync_in_online_timing=True,
            independent_cpu_plan_oracle_count=len(self.cpu_plan_oracles),
            formal_admission_unchanged=True,diagnostic_reuse_not_accuracy_certified=True,
            offline_yoco_full_layout=self.mode == "capture_independent",model_boundary_restored=True,
            not_model_flops_fraction=True,selected_tokens_forced_through_yoco=False,
            empty_ep_qualification_sha256=None if self.empty_ep_capability is None else self.empty_ep_capability.sha256)
        self.state = "COMPLETE"
        return self.last_report

    def clear_cache(self):
        require(self.state in ("IDLE","COMPLETE"), "cannot clear active tail bank")
        self._clear_phase_gpu()
        return super().clear_cache()

    def close(self):
        self._clear_phase_gpu()
        return super().close()
