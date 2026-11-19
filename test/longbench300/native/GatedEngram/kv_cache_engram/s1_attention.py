"""Executable S1 compact attention using native Q/KV/index/attention kernels."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext

from .core import digest, require
from .s1_cache import GLOBAL_SOURCES, fixed_selection


def local_memory_plan(documents, total, start, end, positions, previous_positions=()):
    """All native 128-window dependencies, with no extra nonlocal SWA links."""
    expected = [p for p in fixed_selection(documents, total) if start <= p < end]
    require(list(positions) == expected, "S1 current queries must be fixed tails plus nondoc")
    halo = [p for d in documents for p in range(d["end"]-256, d["end"]-128)]
    halo_lookup = {p: i for i,p in enumerate(halo)}
    current = {p: i for i,p in enumerate(positions)}
    previous = {p: i for i,p in enumerate(previous_positions)}
    needed = set()
    for q in positions:
        needed.update(range(max(0, q-127), q+1))
    memory = sorted(needed)
    kinds, rows = [], []
    for p in memory:
        if p in current:
            kinds.append("current"); rows.append(current[p])
        elif p in previous:
            kinds.append("previous"); rows.append(previous[p])
        elif p in halo_lookup:
            kinds.append("halo"); rows.append(halo_lookup[p])
        else:
            raise ValueError("S1 missing local KV dependency at absolute position " + str(p))
    result = dict(chunk_start=start, chunk_end=end, query_positions=list(positions),
        memory_positions=memory, source_kinds=kinds, source_rows=rows,
        native_window_size=128, extra_nonlocal_swa_links=False,
        local_lengths=[min(128,p+1) for p in positions])
    result["sha256"] = digest(result)
    return result


class S1Attention:
    def __init__(self, cache, *, torch_module=None, allow_sync=None):
        self.cache = cache
        self.t = cache.t if torch_module is None else torch_module
        self.allow_sync = allow_sync
        self.begin_phase()

    def begin_phase(self):
        self.halos, self.previous, self.geometry = {}, {}, {}
        self.events, self.ring_events = [], []
        self.next_chunk = {layer: 0 for layer in range(3, 40)}

    def _indices(self, values, device, dtype=None):
        with self.allow_sync() if self.allow_sync is not None else nullcontext():
            return self.t.tensor(values, device=device, dtype=self.t.long if dtype is None else dtype)

    def _project(self, mod, packed, bf16, positions):
        n = positions.numel()
        if n == 0:
            return dict(q=bf16.new_empty((0, mod.num_heads, 512)),
                swa=bf16.new_empty((0,512)),
                index_q=bf16.new_empty((0, mod.index_n_head*mod.index_head_dim)) if mod.has_local_index else None,
                weights=bf16.new_empty((0, mod.index_n_head)) if mod.has_local_index else None,
                global_raw=bf16.new_empty((0,512 if mod.kv_compress_ratio == 1 else 1024),
                    dtype=self.t.bfloat16 if mod.kv_compress_ratio == 1 else self.t.float32)
                    if mod.is_kv_source else None)
        require(mod.yoco_swa_share_hidden is False, "S1 requires current-layer SWA source")
        q_a, swa = mod._project_q_and_swa(packed, None)
        q, index_q = mod._project_q_b(q_a, positions)
        raw, weights = mod._project_current_hidden_inputs(bf16)
        if mod.is_kv_source and mod.shared_hidden_kv_proj is not None:
            require(mod.layer_idx == 20 and raw is None, "S1 source20 shared projection ownership")
            raw = mod.shared_hidden_kv_proj(bf16)
        return dict(q=q, swa=swa, index_q=index_q, weights=weights, global_raw=raw)

    def _projection(self, mod, packed, bf16, meta, ctx):
        dense = getattr(ctx, "reference_hidden", None) is not None
        checked = dense or getattr(ctx, "verify", False)
        current = self._project(mod, packed, bf16, meta.positions)
        checks = 0
        if dense:
            full = self._project(mod, ctx.reference_hidden, ctx.reference_bf16, ctx.full_meta.positions)
            ids = ctx.reference_selected_rows
            expected = {k: None if v is None else v.index_select(0, ids) for k,v in full.items()}
            for name, actual in current.items():
                other = expected[name]
                require((actual is None) == (other is None), "S1 reference projection presence")
                if actual is not None:
                    require(actual.dtype == other.dtype and actual.shape == other.shape and
                        self.t.equal(actual.contiguous().view(self.t.uint8),other.contiguous().view(self.t.uint8)),
                        "S1 native dense/gather projection byte mismatch: "+name)
                    checks += 1
            return expected, checks, checked
        return current, checks, checked

    def _halo(self, mod, device, position_dtype):
        if mod.layer_idx not in self.halos:
            positions, values = self.cache.halo(mod.layer_idx)
            pos = self._indices(positions, device, position_dtype)
            self.halos[mod.layer_idx] = self.cache.ops.prepare_swa(mod, values, pos)
        return self.halos[mod.layer_idx]

    def _memory(self, mod, prepared, meta, ctx):
        t, layer = self.t, mod.layer_idx
        previous_positions, previous = self.previous.get(layer, ([], prepared[:0]))
        key = ctx.start, ctx.end, tuple(previous_positions)
        if key not in self.geometry:
            self.geometry[key] = local_memory_plan(self.cache.docs, self.cache.total,
                ctx.start, ctx.end, ctx.positions_cpu, previous_positions)
        event = self.geometry[key]
        halo = self._halo(mod, prepared.device, meta.positions.dtype)
        out = prepared.new_empty((len(event["memory_positions"]),1,512))
        for kind, source in (("current", prepared), ("previous", previous), ("halo", halo)):
            dst = [i for i,k in enumerate(event["source_kinds"]) if k == kind]
            if dst:
                src = [event["source_rows"][i] for i in dst]
                out.index_copy_(0, self._indices(dst,out.device), source.index_select(0,self._indices(src,out.device)))
        # Keep only online positions that can be needed across the next chunk.
        candidates = previous_positions + list(ctx.positions_cpu)
        combined = t.cat((previous,prepared),dim=0)
        kept = [i for i,p in enumerate(candidates) if p >= ctx.end-127]
        self.previous[layer] = ([candidates[i] for i in kept], combined.index_select(0,self._indices(kept,out.device)))
        positions = self._indices(event["memory_positions"],out.device,meta.positions.dtype)
        query = meta.positions
        low = t.clamp(query-127,min=0)
        wanted = low[:,None] + t.arange(128,device=out.device,dtype=query.dtype)[None,:]
        valid = wanted <= query[:,None]
        if positions.numel():
            slots = t.searchsorted(positions,wanted.contiguous())
            safe = slots.clamp(max=positions.numel()-1)
            t._assert_async(((~valid) | ((slots < positions.numel()) &
                (positions[safe] == wanted))).all(), "S1 GPU local-window key missing")
            indices = t.where(valid,slots,-1).to(t.int32).contiguous()
        else:
            indices = t.empty((0,128),device=out.device,dtype=t.int32)
        return out, positions, indices, event

    def _joint(self, mod, memory, cache, meta, cross):
        from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
        n, offset = meta.total_k_compressed[mod.ratio_key], memory.shape[0]
        if mod.is_kv_source:
            joint = memory.new_empty((offset+n,1,512))
            load_paged_kv_cache(cache.ca_cache,meta.load_segments_ca[mod.ratio_key],n,out=joint[offset:])
        else:
            old, old_offset = cross.loaded_prefill_kv, cross.loaded_prefill_kv_main_offset
            require(old is not None and old.shape[0]-old_offset == n, "S1 native/shared global gather missing")
            if old.shape[0] == offset+n and old_offset == offset:
                joint = old
            else:
                joint = memory.new_empty((offset+n,1,512))
                joint[offset:].copy_(old[old_offset:])
        joint[:offset].copy_(memory)
        cross.loaded_prefill_kv, cross.loaded_prefill_kv_main_offset = joint, offset
        return joint

    def _store_ring(self, mod, memory, memory_positions, cache, full_meta, ctx, checked):
        from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import store_kv_cache
        from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
        t = self.t
        # Only the final ring must be complete; prefill local reads use explicit
        # online/halo memory and never consume uninitialized ring positions.
        if ctx.end != self.cache.total:
            return
        wanted_cpu = list(range(max(0,self.cache.total-128),self.cache.total))
        wanted = self._indices(wanted_cpu,memory.device,memory_positions.dtype)
        rows = t.searchsorted(memory_positions,wanted)
        t._assert_async((rows < memory_positions.numel()).all(), "S1 final ring missing rows")
        t._assert_async((memory_positions[rows] == wanted).all(), "S1 final ring absolute positions")
        values = memory.index_select(0,rows.long()).contiguous()
        block = cache.swa_cache.shape[1]
        last_slot = full_meta.slot_mapping_store_swa[-1]
        base = last_slot - ((ctx.end-1) % block)
        slots = (base + wanted.to(t.int32)%block).to(t.int32).contiguous()
        actual_current = full_meta.slot_mapping_store_swa
        t._assert_async(((actual_current == -1) |
            (actual_current == base + full_meta.positions.to(t.int32)%block)).all(),
                        "S1 native physical SWA ring addressing changed")
        store_kv_cache(values,cache.swa_cache,slots)
        if checked:
            segments = t.stack((slots//block,slots%block,slots%block+1,
                t.arange(len(wanted_cpu),device=slots.device,dtype=t.int32)),dim=1).contiguous()
            read = values.new_empty(values.shape)
            load_paged_kv_cache(cache.swa_cache,segments,len(wanted_cpu),out=read)
            require(t.equal(read,values), "S1 final physical SWA ring readback mismatch")
        self.ring_events.append(dict(layer=mod.layer_idx,positions=wanted_cpu,
            physical_store=True,physical_readback_exact=checked,checks=len(wanted_cpu) if checked else 0))

    def forward(self, mod, hidden_packed, hidden_bf16, kv_cache, meta, cross, ctx):
        checked = getattr(ctx,"reference_hidden",None) is not None or getattr(ctx,"verify",False)
        with self.allow_sync() if checked and self.allow_sync is not None else nullcontext():
            return self._forward(mod,hidden_packed,hidden_bf16,kv_cache,meta,cross,ctx)

    def _forward(self, mod, hidden_packed, hidden_bf16, kv_cache, meta, cross, ctx):
        t, layer = self.t, mod.layer_idx
        require(3 <= layer < 40 and self.next_chunk[layer] == ctx.start, "S1 layer/chunk order")
        require(meta.total_q == len(ctx.positions_cpu) and meta.positions.shape == (meta.total_q,), "S1 compact query rows")
        require(mod.use_swa and mod.has_packed_kv and mod.is_sparse and not mod.yoco_swa_share_hidden,
                "S1 native attention topology mismatch")
        require(mod.is_kv_source == (layer in GLOBAL_SOURCES)
                and mod.is_index_k_source == (layer in GLOBAL_SOURCES)
                and mod.has_local_index == (layer in (8,14,20,24,28,32,36))
                and mod.kv_compress_ratio == (2 if layer < 20 else 1)
                and mod.window_size == 128 and mod.num_heads == 64,
                "S1 checkpoint global ownership/ratio/head topology changed")
        full = ctx.full_meta
        full.compute_attn_info(mod.max_model_len,mod.window_size,kv_cache.ca_cache.shape[1]*mod.kv_compress_ratio)
        projected, checks, checked = self._projection(mod,hidden_packed,hidden_bf16,meta,ctx)
        if layer in GLOBAL_SOURCES:
            self.cache.global_update(mod,projected["global_raw"],kv_cache,full,ctx)
        if mod.has_local_index and meta.total_q:
            index = mod.index_attn(projected["weights"],projected["index_q"],None,
                kv_cache=kv_cache.ica_cache,state_cache=kv_cache.ica_state,input_metadata=meta,
                store_kv=False,cross_layer_runtime=cross)
            cross.topk_indices = index
        elif mod.has_local_index:
            index = t.empty((0,512),device=meta.positions.device,dtype=t.int32)
            cross.topk_indices = index
        else:
            index = cross.topk_indices
        require(index is not None and tuple(index.shape) == (meta.total_q,512), "S1 shared topk query rows")
        t._assert_async(((index == -1) | ((index >= 0) &
            (index < ((meta.positions+1)//mod.kv_compress_ratio)[:,None]))).all(),
            "S1 global key is incomplete/future")
        if meta.total_q:
            swa, waiter = mod._prepare_swa_kv(projected["swa"],kv_cache.swa_cache,meta,False)
            require(waiter is None, "S1 CP1 prepare returned communication")
        else:
            swa = hidden_bf16.new_empty((0,1,512))
        memory, memory_positions, indices, plan = self._memory(mod,swa,meta,ctx)
        require(full.total_k_swa == 128, "S1 native CP1 history width changed")
        native_indices = full.kv_swa_full_index.index_select(0,ctx.selected_rows)
        native_keys = t.where(native_indices < 128, native_indices+ctx.start-128,
                              native_indices-128+ctx.start)
        native_keys = t.where(native_indices >= 0,native_keys,-1)
        if memory_positions.numel():
            actual_keys = t.where(indices >= 0,memory_positions[indices.clamp(min=0).long()],-1)
        else:
            actual_keys = indices.to(meta.positions.dtype)
        t._assert_async((actual_keys == native_keys).all(),
                        "S1 local indices differ from actual native SWA mask")
        joint = self._joint(mod,memory,kv_cache,meta,cross)
        if meta.total_q:
            context = mod._prefill_attention(projected["q"],joint,index,indices,0,memory.shape[0],meta)
            result = mod._wv_and_o_proj(context,meta.positions)
        else:
            result = hidden_bf16.new_empty((0,hidden_bf16.shape[-1]))
        self._store_ring(mod,memory,memory_positions,kv_cache,full,ctx,checked)
        self.next_chunk[layer] = ctx.end
        self.events.append(dict(layer=layer,chunk_start=ctx.start,chunk_end=ctx.end,
            selected_positions=list(ctx.positions_cpu),selected_rows=meta.total_q,
            full_rows=ctx.end-ctx.start,memory_positions=plan["memory_positions"],
            offline_halo_rows=plan["source_kinds"].count("halo"),
            actual_attention_rows=meta.total_q,projection_identity_checks=checks,
            gpu_local_window_oracle_checked=True,actual_native_swa_indices_checked=True,
            global_causal_checked=True,
            no_unselected_hidden_fallback=True,no_extra_swa_links=True,
            plan_sha256=plan["sha256"],max_absolute_error=0.0 if checks else None))
        return result

    def source2_fingerprint(self, cache2, full_meta, cross=None):
        """Qualification-only host synchronization of valid main/index-K data."""
        with self.allow_sync() if self.allow_sync is not None else nullcontext():
            return self._source2_fingerprint(cache2,full_meta)

    def _source2_fingerprint(self,cache2,full_meta):
        from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
        from deepinfer.model_executor.layers.tilelang_ops.cache_common import get_kv_cache_format
        result = {}
        rows = full_meta.total_k_compressed["2"]
        for name,cache in (("main",cache2.ca_cache),("index_k",cache2.ica_cache)):
            loaded = load_paged_kv_cache(cache,full_meta.load_segments_ca["2"],rows,
                                         layout=get_kv_cache_format(cache))
            tensors = loaded if isinstance(loaded,tuple) else (loaded,)
            h = hashlib.sha256()
            for value in tensors:
                if value is not None:
                    h.update(value.detach().contiguous().view(self.t.uint8).cpu().numpy().tobytes())
            result[name+"_sha256"] = h.hexdigest()
        return dict(schema="s1-source2-cache-fingerprint-v1",rows=rows,**result)

    def report(self):
        return dict(schema="s1-attention-report-v1",attention_events=self.events,
            decode_ring_events=self.ring_events,full_swa_bank_used=False,
            only_tail_and_nondoc_queries=True,native_global_attention_kernel=True,
            selected_queries_survive_layers3_through39=True)
