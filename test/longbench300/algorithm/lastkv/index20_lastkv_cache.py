"""Extra immutable layer39 attention inputs for a final-SWA replay experiment.

This stores post-mHC/post-attention-normalization BF16 and the exact native
packed input. It does not advance unselected hidden states through any layer.
"""
from __future__ import annotations

from index20_plan import digest, require

LAYER = 39
KINDS = ("last_attn_bf16", "last_attn_packed", "last_attn_sf")


def copy_native_fp8_rows(destination, destination_rows, source, source_rows, torch_module):
    """Copy packed FP8 bytes; CPU index_copy has no Float8 kernel.

    uint8 views preserve every native bit and avoid a BF16 dequant/requant
    roundtrip. The same implementation is used for CPU ownership tests and GPU.
    """
    t = torch_module
    require(destination.dtype == source.dtype == t.float8_e4m3fn
            and destination.ndim == source.ndim == 2
            and destination.shape[1] == source.shape[1]
            and destination.stride(-1) == source.stride(-1) == 1
            and destination.device == source.device,
            "last-KV FP8 byte-copy tensor layout")
    require(destination_rows.dtype == source_rows.dtype == t.long
            and destination_rows.ndim == source_rows.ndim == 1
            and destination_rows.shape == source_rows.shape
            and destination_rows.device == source_rows.device == destination.device,
            "last-KV FP8 byte-copy index geometry")
    destination.view(t.uint8).index_copy_(0,destination_rows,
        source.view(t.uint8).index_select(0,source_rows))


def index20_lastkv_cache_class(base_class):
    class LastKVCache(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lastkv_input_width = None

        def capture_last_attention_inputs(self, mod, packed, bf16, meta):
            require(self.capture is not None and mod.layer_idx == LAYER,
                    "last-KV inputs require active layer39 independent capture")
            c, t = self.capture, self.t
            start, end = self._bounds(meta)
            require(bf16.ndim == 2 and tuple(bf16.shape[:1]) == (end-start,)
                    and bf16.dtype == t.bfloat16 and bf16.is_contiguous(),
                    "last-KV BF16 attention input shape/type")
            require(isinstance(packed, tuple) and len(packed) == 2,
                    "last-KV replay requires exact native FP8/SF input")
            data, sf = packed
            width = bf16.shape[1]
            require(width % 128 == 0 and data.shape == bf16.shape
                    and data.dtype == t.float8_e4m3fn and sf.dtype == t.int32
                    and tuple(sf.shape) == (end-start,width//128)
                    and data.device == sf.device == bf16.device,
                    "last-KV native packed FP8/UE8M0 input layout")
            require(self.lastkv_input_width in (None,width), "last-KV input width changed")
            self.lastkv_input_width = width
            for kind,value in zip(KINDS,(bf16,data,sf)):
                key = c["key"],kind,LAYER
                if key not in self.bank:
                    self.bank[key] = dict(tensor=self._allocate(c["length"],value.shape[1],value.dtype,value.device),
                        length=c["length"],local_start=0,scope=digest(self.scope),complete=False)
                cursor = c["coverage"].get((kind,LAYER),0)
                require(cursor == start and 0 <= start < end <= c["length"],
                        "last-KV input capture must be complete and contiguous")
                self.bank[key]["tensor"][start:end].copy_(value)
                c["coverage"][(kind,LAYER)] = end

        def finish_capture(self):
            c = self.capture
            require(c is not None, "last-KV no active capture")
            payloads = []
            for kind in KINDS:
                require(c["coverage"].get((kind,LAYER)) == c["length"],
                        "last-KV attention input coverage incomplete")
                entry = self.bank[c["key"],kind,LAYER]
                value = entry["tensor"]
                if value.dtype != self.t.int32:
                    self.t._assert_async(self.t.isfinite(value.float()).all(),"nonfinite last-KV input")
                entry.update(complete=True,version=value._version)
                payloads.append(dict(kind=kind,layer=LAYER,rows=value.shape[0],width=value.shape[1],
                    dtype=str(value.dtype).replace("torch.",""),
                    bytes=value.numel()*value.element_size(),document_sha256=c["key"],
                    scope_sha256=digest(self.scope),disk_persisted=False))
            result = super().finish_capture()
            result["lastkv_input_payloads"] = payloads
            result["lastkv_input_bytes"] = sum(p["bytes"] for p in payloads)
            result["lastkv_inputs_are_post_mhc_attn_norm"] = True
            result["lastkv_unselected_online_hidden_layers_recomputed"] = False
            result["hidden_or_ffn_bank_persisted_scope"] = "disk_files_only"
            result["offline_attention_hidden_gpu_cached"] = True
            result["offline_ffn_output_gpu_cached"] = False
            result["lastkv_input_disk_persisted"] = False
            return result

        def prepare_case(self, **kwargs):
            result = super().prepare_case(**kwargs)
            for d in self.docs:
                shapes = []
                for kind in KINDS:
                    value = self.entry(d["key"],kind,LAYER)
                    require(value.shape[0] == d["end"]-d["start"] and value.device == self.device,
                            "last-KV bank source does not match exact document/device")
                    shapes.append(value.shape[1])
                require(shapes == [self.lastkv_input_width,self.lastkv_input_width,self.lastkv_input_width//128],
                        "last-KV bank width mismatch")
            return result

        def materialize_last_attention_inputs(self, start, end, selected, online_packed, online_bf16):
            """One bounded interval: every row comes from current online or exact offline input."""
            t = self.t
            require(type(start) is type(end) is int and 0 <= start < end <= self.total,
                    "last-KV materialization interval")
            selected = list(selected)
            require(selected == sorted(set(selected)) and isinstance(online_packed,tuple)
                    and all(type(p) is int and 0 <= p < self.total for p in selected)
                    and len(online_packed) == 2, "last-KV online row mapping")
            data,sf = online_packed
            width = self.lastkv_input_width
            require(tuple(online_bf16.shape) == tuple(data.shape) == (len(selected),width)
                    and tuple(sf.shape) == (len(selected),width//128)
                    and online_bf16.dtype == t.bfloat16 and data.dtype == t.float8_e4m3fn
                    and sf.dtype == t.int32 and data.device == sf.device == online_bf16.device == self.device,
                    "last-KV online packed/BF16 rows changed")
            n = end-start
            bf16 = t.empty((n,width),dtype=t.bfloat16,device=self.device)
            full_data = t.empty((n,width),dtype=t.float8_e4m3fn,device=self.device)
            # Logical values use native column-major aligned SF storage at GEMM.
            sf_owner = t.zeros((width//128,max(4,((n+3)//4)*4)),dtype=t.int32,device=self.device)
            full_sf = sf_owner[:,:n].transpose(0,1)
            covered = set()
            for d in self.docs:
                a,z = max(start,d["start"]),min(end,d["end"])
                if a >= z:
                    continue
                for kind,out in zip(KINDS,(bf16,full_data,full_sf)):
                    out[a-start:z-start].copy_(self.entry(d["key"],kind,LAYER)[a-d["start"]:z-d["start"]])
                covered.update(range(a,z))
            pairs = [(p-start,i) for i,p in enumerate(selected) if start <= p < end]
            if pairs:
                dst = self._indices([p for p,_ in pairs],self.device)
                src = self._indices([i for _,i in pairs],self.device)
                for out,value in zip((bf16,full_data,full_sf),(online_bf16,data,sf)):
                    if value.dtype == t.float8_e4m3fn:
                        copy_native_fp8_rows(out,dst,value,src,t)
                    else:
                        out.index_copy_(0,dst,value.index_select(0,src))
            online = {selected[i] for _,i in pairs}
            require(covered | online == set(range(start,end)),
                    "last-KV missing noncached online attention input")
            return (full_data,full_sf),bf16,dict(start=start,end=end,full_rows=n,
                online_input_rows=len(online),offline_input_rows=n-len(online),
                source_coverage_complete=True,post_mhc_attn_norm_input=True,
                native_packed_input_preserved=True,bf16_requantized=False)

        def original_last_swa_rows(self, positions):
            """Read original canonical layer39 values for diagnostic comparisons only."""
            t = self.t
            positions = list(positions)
            values = t.empty((len(positions),512),dtype=t.bfloat16,device=self.device)
            covered = 0
            for d in self.docs:
                pairs = [(i,p-d["start"]) for i,p in enumerate(positions) if d["start"] <= p < d["end"]]
                if pairs:
                    dst = self._indices([i for i,_ in pairs],self.device)
                    src = self._indices([p for _,p in pairs],self.device)
                    values.index_copy_(0,dst,self.entry(d["key"],"swa_full",LAYER).index_select(0,src))
                    covered += len(pairs)
            require(covered == len(positions), "last-KV diagnostic position outside source documents")
            return values

        def report(self):
            result = super().report()
            entries = [e for k,e in self.bank.items() if k[1] in KINDS]
            per_kind = {kind:sum(e["tensor"].numel()*e["tensor"].element_size()
                        for k,e in self.bank.items() if k[1] == kind) for kind in KINDS}
            result["lastkv_input_bank"] = dict(schema="index20-lastkv-input-bank-v1",layer_zero_based=LAYER,
                extra_entries=len(entries),extra_bytes=sum(per_kind.values()),bytes_by_kind=per_kind,
                width=self.lastkv_input_width,original_full_swa_bank_retained=True,
                captures_post_mhc_attn_norm_input=True,native_packed_fp8_and_sf_preserved=True,
                offline_attention_input_cached=True,offline_ffn_output_cached=False,
                unselected_online_hidden_layers_recomputed=False,disk_persisted=False)
            result["index20_full_swa_dependency_bank"]["no_offline_hidden_or_ffn_cache"] = False
            result["index20_full_swa_dependency_bank"]["extra_last_layer_attention_input_bank_present"] = True
            result["hidden_or_ffn_bank_persisted_scope"] = "disk_files_only"
            result["offline_attention_hidden_gpu_cached"] = bool(entries)
            result["offline_ffn_output_gpu_cached"] = False
            result["offline_bank_rows_by_kind"] = {
                kind:sum(e["tensor"].shape[0] for k,e in self.bank.items() if k[1] == kind)
                for kind in sorted({k[1] for k in self.bank})}
            result["offline_bank_total_tensor_rows"] = sum(result["offline_bank_rows_by_kind"].values())
            return result
    LastKVCache.__name__ = "LastKV"+base_class.__name__
    return LastKVCache


lastkv_cache_class = index20_lastkv_cache_class
