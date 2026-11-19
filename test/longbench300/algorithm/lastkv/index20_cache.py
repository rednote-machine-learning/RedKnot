"""Independent full-SWA dependency bank for index20 experiments.

The original halo/global entries and transforms remain unchanged. The extra
full-SWA bank is a correctness prototype cost, explicitly counted in reports.
"""
from __future__ import annotations
from index20_plan import digest, require, validate_selection, required_offline_positions

SWA_LAYERS = tuple(range(3, 40))


def index20_cache_class(base_class):
    class Index20Cache(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index20_selected = None
            self.index20_offline_positions = None
            self.index20_halo_groups = None

        def capture_swa(self, mod, raw, meta):
            # Preserve the original halo computation and its source bytes.
            super().capture_swa(mod, raw, meta)
            if self.capture is None or mod.layer_idx not in SWA_LAYERS:
                return
            c = self.capture
            start, end = self._bounds(meta)
            value = self.ops.normalize_swa(mod, raw.contiguous())
            require(tuple(value.shape)==(end-start,512) and value.dtype==self.t.bfloat16,
                    "full SWA capture shape/precision differs")
            key = c["key"], "swa_full", mod.layer_idx
            if key not in self.bank:
                self.bank[key] = dict(tensor=self._allocate(c["length"],512,self.t.bfloat16,value.device),
                    length=c["length"],local_start=0,scope=digest(self.scope),complete=False)
            cursor = c["coverage"].get(("swa_full",mod.layer_idx),0)
            require(cursor==start, "full SWA capture must be contiguous and complete")
            self.bank[key]["tensor"][start:end].copy_(value)
            c["coverage"][("swa_full",mod.layer_idx)] = end

        def finish_capture(self):
            c = self.capture
            require(c is not None, "no full-SWA capture active")
            payloads=[]
            for layer in SWA_LAYERS:
                require(c["coverage"].get(("swa_full",layer))==c["length"],
                        "full SWA layer coverage incomplete")
                e=self.bank[c["key"],"swa_full",layer];value=e["tensor"]
                self.t._assert_async(self.t.isfinite(value).all(),"nonfinite full SWA capture")
                e.update(complete=True,version=value._version)
                payloads.append(dict(kind="swa_full",layer=layer,rows=c["length"],width=512,
                    dtype="bfloat16",local_start=0,bytes=value.numel()*value.element_size(),
                    document_sha256=c["key"],source_document_tokens=c["length"],scope_sha256=digest(self.scope),
                    disk_persisted=False))
            # Native checks still validate every original halo and global entry.
            result=super().finish_capture()
            result["index20_full_swa_payloads"]=payloads
            result["index20_full_swa_bytes"]=sum(p["bytes"]for p in payloads)
            result["index20_full_swa_cached"]=True
            result["index20_full_swa_disk_persisted"]=False
            result["index20_capture_policy"]="original_halo_plus_global_unchanged_and_additional_full_swa"
            return result

        def prepare_case(self, **kwargs):
            result=super().prepare_case(**kwargs)
            self.bind_selection(None)
            for d in self.docs:
                for layer in SWA_LAYERS:
                    value=self.entry(d["key"],"swa_full",layer)
                    require(tuple(value.shape)==(d["end"]-d["start"],512) and value.device==self.device,
                            "full-SWA bank does not match current exact document")
            return result

        def bind_selection(self, selected):
            """Call before the first deep layer; None restores original S1 tails."""
            if selected is None:
                from index20_plan import mandatory_selection
                self.index20_selected=None
                self.selected=mandatory_selection(self.docs,self.total) if self.total else []
                self.selected_set=set(self.selected)
                self.index20_offline_positions=None
                self.index20_halo_groups=None
                return
            selected=validate_selection(self.docs,self.total,selected)
            self.index20_selected=list(selected)
            self.selected=list(selected);self.selected_set=set(selected)
            self.index20_offline_positions=required_offline_positions(self.docs,self.total,selected)
            groups=[]
            for d in self.docs:
                matches=[(i,p-d["start"])for i,p in enumerate(self.index20_offline_positions)
                         if d["start"]<=p<d["end"]]
                if matches:groups.append((d["key"],[a for a,b in matches],[b for a,b in matches]))
            require(sum(len(dst)for _,dst,_ in groups)==len(self.index20_offline_positions),
                    "full-SWA source mapping did not cover every offline dependency")
            self.index20_halo_groups=groups

        def halo(self, layer):
            if self.index20_selected is None:
                return super().halo(layer)
            require(layer in SWA_LAYERS,"invalid full-SWA layer")
            positions=self.index20_offline_positions
            values=self.t.empty((len(positions),512),device=self.device,dtype=self.t.bfloat16)
            for key,dst,src in self.index20_halo_groups:
                source=self.entry(key,"swa_full",layer)
                values.index_copy_(0,self._indices(dst,self.device),source.index_select(0,self._indices(src,self.device)))
            return list(positions),values

        def report(self):
            result=super().report()
            entries=[e for k,e in self.bank.items()if k[1]=="swa_full"]
            result["index20_full_swa_dependency_bank"]=dict(
                schema="index20-full-swa-bank-v1",extra_entries=len(entries),
                extra_rows=sum(e["tensor"].shape[0]for e in entries),
                extra_bytes=sum(e["tensor"].numel()*e["tensor"].element_size()for e in entries),
                current_selection_mode="original_s12"if self.index20_selected is None else"index20",
                selected_count=len(self.selected)if self.total else 0,
                offline_dependency_positions=len(self.index20_offline_positions or()),
                full_swa_cached=True,disk_persisted=False,
                original_halo_and_global_entries_retained=True,
                no_offline_hidden_or_ffn_cache=True,
                selection_sha256=None if self.index20_selected is None else digest(self.index20_selected))
            return result
    Index20Cache.__name__="Index20"+base_class.__name__
    return Index20Cache
