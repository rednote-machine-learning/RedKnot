"""S1 immutable halo/raw bank and native global-cache assembly.

Only source projections and 128-row SWA halos are persistent. Position-dependent
global bases are built afresh at phase start and their cost is reported. No
unselected hidden/FFN values are stored or advanced by this module.
"""
from __future__ import annotations

import copy
import hashlib
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from .core import digest, require
from .tail_link_plan import normalize_documents

SWA_LAYERS = tuple(range(3, 40))
GLOBAL_SOURCES = (8, 14, 20)


def fixed_selection(documents, total):
    docs = normalize_documents(documents, total)
    require(all(d["end"] - d["start"] >= 256 for d in docs), "S1 requires document length >=256")
    hidden = {p for d in docs for p in range(d["start"], d["end"] - 128)}
    return [p for p in range(total) if p not in hidden]


class NativeCacheOps:
    """All numerical transforms use the installed native backend."""
    def __init__(self, torch_module, allow_sync=None):
        from .native_reuse import NativeReuseOps
        from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
            CastArgs, KVCacheOutArgs, NormArgs, RopeArgs,
            fused_norm_rope_hadamard_cast_store_kvcache,
        )
        self.t = torch_module
        self.allow_sync = allow_sync
        self.swa = NativeReuseOps.from_deepinfer()
        self.CastArgs, self.Out, self.Norm, self.Rope = CastArgs, KVCacheOutArgs, NormArgs, RopeArgs
        self.transform = fused_norm_rope_hadamard_cast_store_kvcache

    def normalize_swa(self, mod, raw):
        return self.swa.normalize(mod, raw)

    def prepare_swa(self, mod, canonical, positions):
        return self.swa.postprocess(mod, canonical, SimpleNamespace(total_q=len(positions), positions=positions))[0]

    def _compression_meta(self, raw, *, end=None, state_id=None, state=None):
        t, n = self.t, raw.shape[0]
        end = n if end is None else end
        begin = end - n
        count = end // 2 - begin // 2
        def ints(values):
            with self.allow_sync() if self.allow_sync is not None else nullcontext():
                return t.tensor(values, dtype=t.int32, device=raw.device)
        if state is None:
            state = t.zeros((1, 2, 2, 512), dtype=t.float32, device=raw.device)
        meta = SimpleNamespace(is_prompt=True, mirror_cp_reverse_index=None,
            state_block_idx=ints([0]) if state_id is None else state_id,
            compressed_batch_mapping={"2": t.zeros(max(1, count), dtype=t.int32, device=raw.device)},
            cu_seqlens_q=ints([0, n]), cu_seqlens_k=ints([0, end]),
            cu_compressed_seqlens_q={"2": ints([0, max(1, count)])})
        return meta, state, count

    def compress_canonical(self, mod, raw):
        """Native compressed BF16 BEFORE norm; preserve fused norm/RoPE rounding."""
        if mod.kv_compress_ratio == 2:
            require(raw.shape[0] % 2 == 0, "C2 input must contain complete pairs")
            if not raw.shape[0]:
                return raw.new_empty((0, 512), dtype=self.t.bfloat16)
            meta, state, count = self._compression_meta(raw)
            compressed = mod.PackedWkv.forward_pre_norm(raw, state, meta)[:count]
        else:
            compressed = raw
        return compressed

    def store_main_and_index(self, mod, canonical, cache, full_meta):
        from deepinfer.model_executor.layers.tilelang_ops.cache_common import get_kv_cache_format
        layout = get_kv_cache_format(cache.ca_cache)
        norm = mod.PackedWkv.norm
        normalized = self.t.empty_like(canonical).unsqueeze(1)
        self.transform(canonical.unsqueeze(1),
            norm_args=self.Norm(norm.weight,norm.variance_epsilon),
            rope_args=self.Rope(full_meta.positions_c[mod.ratio_key], mod.rotary_emb.cos_sin_cache,
                mod.rotary_emb.rotary_dim, mod.rotary_emb.is_neox_style,
                offset=512-mod.rotary_emb.rotary_dim),
            cast_args=self.CastArgs.from_cache_format(layout),
            out_args=self.Out(cache.ca_cache, full_meta.slot_mapping_store_ca[mod.ratio_key],
                layout=layout,out=normalized,out_stage="norm"))
        mod.index_attn.store_index_kv(normalized[:,0], cache.ica_cache, cache.ica_state, full_meta)

    def verify_native_store(self, mod, raw, cache, full_meta):
        """Qualification only: compare real packed payload/SF against original full store."""
        from deepinfer.model_executor.layers.tilelang_ops.cache_common import split_kv_cache, get_kv_cache_format
        slots = full_meta.slot_mapping_store_ca[mod.ratio_key].clamp(min=0)
        def snapshot(value):
            block, arrays = split_kv_cache(value,get_kv_cache_format(value))
            return [a.reshape(value.shape[0],block,-1)[slots//block,slots%block].contiguous().view(self.t.uint8).clone()
                    for a in arrays if a is not None]
        before = snapshot(cache.ca_cache)+snapshot(cache.ica_cache)
        normalized = mod.PackedWkv.forward_store_kv_cache(raw,cache.ca_cache,cache.ca_state,
            full_meta,return_pre_rope_kv=True)
        mod.index_attn.store_index_kv(normalized,cache.ica_cache,cache.ica_state,full_meta)
        after = snapshot(cache.ca_cache)+snapshot(cache.ica_cache)
        require(len(before) == len(after) and all(self.t.equal(a,b) for a,b in zip(before,after)),
                "S1 assembled global main/index packed payload differs from original native full store")

    def update_compressor_carry(self, mod, raw, cache, full_meta, end):
        if mod.kv_compress_ratio != 2:
            return
        # Reuse the original state-writing kernel on the final state-size rows.
        # This preserves an odd unpaired prompt token for subsequent decode.
        keep = min(raw.shape[0], cache.ca_state.shape[1])
        if not keep:
            return
        meta, _, _ = self._compression_meta(raw[-keep:].contiguous(), end=end,
            state_id=full_meta.state_block_idx, state=cache.ca_state)
        mod.PackedWkv.forward_pre_norm(raw[-keep:].contiguous(), cache.ca_state, meta)


class S1Cache:
    def __init__(self, *, layers, scope, torch_module=None, allow_sync=None, ops=None):
        if torch_module is None:
            import torch as torch_module
        self.t, self.layers, self.scope = torch_module, list(layers), dict(scope)
        self.allow_sync = allow_sync
        self.ops = ops if ops is not None else NativeCacheOps(self.t, allow_sync=allow_sync)
        self.bank, self.capture = {}, None
        self.docs, self.total = [], 0
        self.phase_generation = 0
        self.global_bases, self.global_carry, self.global_events = {}, {}, []
        self.capture_reports, self.preparation_s = [], 0.0

    def _allocate(self, rows, width, dtype, device):
        with self.t.inference_mode(False), self.t.no_grad():
            return self.t.empty((rows, width), dtype=dtype, device=device)

    def begin_capture(self, key, length, output_dir=None):
        require(self.capture is None and length >= 256, "S1 capture overlap/short document")
        require(not any(k[0] == key for k in self.bank), "S1 immutable capture overwrite")
        self.capture = dict(key=key, length=length, output_dir=output_dir, coverage={})

    @staticmethod
    def _bounds(meta):
        marker = meta.canonical_request_binding
        return marker["start"], marker["end"]

    def _capture_rows(self, kind, layer, value, start, end, source_start):
        c, t = self.capture, self.t
        require(c is not None, "capture not active")
        rows = 128 if kind == "swa_halo" else c["length"]
        dtype = t.bfloat16 if kind == "swa_halo" or layer == 20 else t.float32
        width = 1024 if kind == "global_raw" and layer != 20 else 512
        require(tuple(value.shape) == (end-start, width) and value.dtype == dtype, "S1 capture shape/dtype")
        key = c["key"], kind, layer
        if key not in self.bank:
            self.bank[key] = dict(tensor=self._allocate(rows, width, dtype, value.device),
                length=c["length"], local_start=source_start, scope=digest(self.scope), complete=False)
        entry = self.bank[key]
        cursor = c["coverage"].get((kind, layer), 0)
        require(start-source_start == cursor, "S1 capture coverage is not contiguous")
        entry["tensor"][cursor:cursor+end-start].copy_(value)
        c["coverage"][(kind, layer)] = cursor+end-start

    def capture_swa(self, mod, raw, meta):
        if self.capture is None or mod.layer_idx not in SWA_LAYERS:
            return
        start, end = self._bounds(meta)
        lo, hi = self.capture["length"]-256, self.capture["length"]-128
        left, right = max(start, lo), min(end, hi)
        if left < right:
            # Normalize only the captured rows, retaining native transform semantics.
            value = self.ops.normalize_swa(mod, raw[left-start:right-start].contiguous())
            self._capture_rows("swa_halo", mod.layer_idx, value, left, right, lo)

    def capture_global(self, mod, projected_raw, meta):
        if self.capture is None or mod.layer_idx not in GLOBAL_SOURCES:
            return
        start, end = self._bounds(meta)
        self._capture_rows("global_raw", mod.layer_idx, projected_raw, start, end, 0)

    def finish_capture(self):
        c = self.capture
        require(c is not None, "no active S1 capture")
        payloads = []
        for kind, layers, rows in (("swa_halo", SWA_LAYERS, 128),
                                   ("global_raw", GLOBAL_SOURCES, c["length"])):
            for layer in layers:
                require(c["coverage"].get((kind, layer)) == rows, "S1 offline coverage incomplete")
                entry = self.bank[c["key"], kind, layer]
                value = entry["tensor"]
                self.t._assert_async(self.t.isfinite(value).all(), "nonfinite S1 offline payload")
                entry.update(complete=True, version=value._version)
                item = dict(kind=kind, layer=layer, rows=rows, width=value.shape[1],
                    dtype=str(value.dtype).replace("torch.", ""), local_start=entry["local_start"],
                    bytes=value.numel()*value.element_size(), document_sha256=c["key"],
                    source_document_tokens=c["length"], scope_sha256=digest(self.scope))
                if c["output_dir"] is not None:
                    with self.allow_sync() if self.allow_sync is not None else nullcontext():
                        raw = value.detach().contiguous().view(self.t.uint8).cpu().numpy().tobytes()
                    path = Path(c["output_dir"]) / (c["key"]+f".{kind}.layer{layer}.bin")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(raw)
                    item.update(file=path.name, file_sha256=hashlib.sha256(raw).hexdigest())
                payloads.append(item)
        result = dict(schema="s1-offline-capture-v1", document_sha256=c["key"],
            document_tokens=c["length"], payloads=payloads,
            swa_halo_bytes=37*128*512*2, global_raw_bytes=c["length"]*9216,
            full_swa_bank_persisted=False, hidden_or_ffn_bank_persisted=False)
        self.capture_reports.append(result)
        self.capture = None
        return result

    def entry(self, key, kind, layer):
        e = self.bank.get((key, kind, layer))
        require(e is not None and e["complete"] and e["scope"] == digest(self.scope)
                and e["tensor"]._version == e["version"], "missing/mutated/foreign S1 bank entry")
        return e["tensor"]

    def prepare_case(self, *, documents, input_tokens=None, full_input_tokens=None, input_ids=None, index_device=None, **unused):
        require(not unused, "unknown S1 prepare_case fields")
        self.total = input_tokens if input_tokens is not None else full_input_tokens
        if self.total is None and input_ids is not None:
            self.total = len(input_ids)
        self.docs = normalize_documents(documents, self.total)
        self.selected = fixed_selection(self.docs, self.total)
        self.selected_set = set(self.selected)
        self.device = self.t.device(index_device) if index_device is not None else next(iter(self.bank.values()))["tensor"].device
        for d in self.docs:
            for kind, layers in (("swa_halo", SWA_LAYERS), ("global_raw", GLOBAL_SOURCES)):
                for layer in layers:
                    value = self.entry(d["key"], kind, layer)
                    require(value.device == self.device, "S1 bank device mismatch")
                    require(self.bank[d["key"], kind, layer]["length"] == d["end"]-d["start"], "S1 document length mismatch")

    def _offline_raw(self, layer, start, end):
        width = 512 if layer == 20 else 1024
        dtype = self.t.bfloat16 if layer == 20 else self.t.float32
        out = self.t.zeros((end-start, width), device=self.device, dtype=dtype)
        for d in self.docs:
            lo, hi = max(start, d["start"]), min(end, d["end"])
            if lo < hi:
                out[lo-start:hi-start].copy_(self.entry(d["key"], "global_raw", layer)[lo-d["start"]:hi-d["start"]])
        return out

    def _indices(self, values, device):
        with self.allow_sync() if self.allow_sync is not None else nullcontext():
            return self.t.tensor(values, device=device, dtype=self.t.long)

    def begin_phase(self, phase, *, build_global_base=True):
        self.phase, self.phase_generation = phase, self.phase_generation+1
        self.global_bases, self.global_carry, self.global_events = {}, {}, []
        self.next_chunk = {layer: 0 for layer in GLOBAL_SOURCES}
        begin = time.perf_counter()
        if build_global_base:
            for layer in GLOBAL_SOURCES:
                mod = self.layers[layer].self_attn
                raw = self._offline_raw(layer, 0, self.total)
                stop = self.total if layer == 20 else self.total//2*2
                self.global_bases[layer] = self.ops.compress_canonical(mod, raw[:stop])
        if build_global_base and self.device.type == "cuda":
            with self.allow_sync() if self.allow_sync is not None else nullcontext():
                self.t.cuda.synchronize(self.device)
        self.preparation_s = time.perf_counter()-begin
        return self.preparation_s

    def halo(self, layer):
        values = self.t.cat([self.entry(d["key"], "swa_halo", layer) for d in self.docs], dim=0)
        positions = [p for d in self.docs for p in range(d["end"]-256, d["end"]-128)]
        return positions, values

    def global_update(self, mod, online_raw, kv_cache, full_meta, ctx):
        layer, ratio = mod.layer_idx, mod.kv_compress_ratio
        require(layer in GLOBAL_SOURCES and self.next_chunk[layer] == ctx.start, "S1 global source/chunk order")
        require(layer in self.global_bases, "S1 phase global base not prepared")
        selected = list(ctx.positions_cpu)
        require(selected == [p for p in self.selected if ctx.start <= p < ctx.end], "global online selection differs")
        require(tuple(online_raw.shape) == (len(selected), 512 if ratio == 1 else 1024), "global projected shape")
        require(online_raw.dtype == (self.t.bfloat16 if ratio == 1 else self.t.float32),
                "S1 source raw projection precision changed")
        current = self._offline_raw(layer, ctx.start, ctx.end)
        ids = self._indices([p-ctx.start for p in selected],current.device)
        current.index_copy_(0, ids, online_raw)
        begin_id, end_id = ctx.start//ratio, ctx.end//ratio
        canonical = self.global_bases[layer][begin_id:end_id].clone()
        affected = sorted({p//ratio for p in selected if p//ratio < end_id})
        if ratio == 2 and ctx.start % 2 and ctx.start-1 in self.selected_set:
            affected = sorted(set(affected) | {ctx.start//2})
        if affected:
            positions = [p for j in affected for p in range(j*ratio, (j+1)*ratio)]
            raw = self._offline_raw(layer, begin_id*ratio, ctx.end)
            raw[ctx.start-begin_id*ratio:].copy_(current)
            if ctx.start % ratio:
                carry = self.global_carry.get(layer)
                require(carry is not None and carry[0] == ctx.start-1, "S1 missing C2 phase carry")
                raw[0].copy_(carry[1])
            member_ids = self._indices([p-begin_id*ratio for p in positions],raw.device)
            updated = self.ops.compress_canonical(mod, raw.index_select(0, member_ids).contiguous())
            dst = self._indices([j-begin_id for j in affected],raw.device)
            canonical.index_copy_(0, dst, updated)
        checks = 0
        if getattr(ctx, "verify", False) or getattr(ctx, "mode", "") == "reference_sparse":
            # Independent full-current assembly through the native compressor.
            whole = current
            if ctx.start % ratio:
                carry = self.global_carry.get(layer)
                require(carry is not None and carry[0] == ctx.start-1, "S1 reference carry absent")
                whole = self.t.cat((carry[1].unsqueeze(0), current), dim=0)
            stop = (ctx.end//ratio-begin_id)*ratio
            expected = self.ops.compress_canonical(mod, whole[:stop])
            require(self.t.equal(canonical, expected), "S1 affected-only/global full compression byte parity failed")
            checks = 1
        if ctx.end % ratio:
            self.global_carry[layer] = (ctx.end-1, current[-1].clone())
        else:
            self.global_carry.pop(layer, None)
        # Native metadata reserves one -1 slot when there is no complete C2 row.
        stored = canonical if end_id > begin_id else current.new_zeros((1, 512), dtype=self.t.bfloat16)
        require(stored.shape[0] == full_meta.positions_c[mod.ratio_key].numel(), "global native compressed metadata rows")
        self.ops.store_main_and_index(mod, stored, kv_cache, full_meta)
        if checks:
            self.ops.verify_native_store(mod,current,kv_cache,full_meta)
            checks += 1
        self.ops.update_compressor_carry(mod, current, kv_cache, full_meta, ctx.end)
        self.next_chunk[layer] = ctx.end
        event = dict(layer=layer, phase_generation=self.phase_generation, chunk_start=ctx.start, chunk_end=ctx.end,
            compression_ratio=ratio, online_projection_rows=len(selected), full_raw_rows=ctx.end-ctx.start,
            online_projection_positions=selected,
            complete_compressed_ids=list(range(begin_id, end_id)), affected_compressed_ids=affected,
            offline_base_rows=end_id-begin_id-len(affected), checks=checks,
            max_absolute_error=0.0 if checks else None, source2_untouched=True,
            main_and_index_k_native_store=True, global_base_representation="compressed_prenorm_bf16",
            packed_main_and_index_store_checked=bool(checks),native_compressor_carry_updated=ratio == 2)
        self.global_events.append(event)
        return stored

    def report(self):
        entries = list(self.bank.values())
        return dict(schema="s1-cache-report-v1", offline_bank_entry_count=len(entries),
            offline_bank_bytes=sum(e["tensor"].numel()*e["tensor"].element_size() for e in entries),
            offline_swa_halo_rows=sum(e["tensor"].shape[0] for k,e in self.bank.items() if k[1] == "swa_halo"),
            offline_global_raw_rows=sum(e["tensor"].shape[0] for k,e in self.bank.items() if k[1] == "global_raw"),
            offline_swa_halo_bytes=sum(e["tensor"].numel()*e["tensor"].element_size()
                for k,e in self.bank.items() if k[1] == "swa_halo"),
            offline_global_raw_bytes=sum(e["tensor"].numel()*e["tensor"].element_size()
                for k,e in self.bank.items() if k[1] == "global_raw"),
            full_swa_bank_persisted=False, hidden_or_ffn_bank_persisted=False,
            phase_generation=self.phase_generation, phase_preparation_s=self.preparation_s,
            phase_preparation_in_ttft_required=True, global_events=self.global_events,
            capture_reports=self.capture_reports, source2_native_prepass_unchanged=True)
