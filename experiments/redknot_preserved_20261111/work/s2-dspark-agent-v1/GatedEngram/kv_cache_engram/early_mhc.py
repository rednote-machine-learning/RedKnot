"""Independent early-six selector and per-layer mHC-only inactive path.

This module is an experimental controller, not the production Engram gate.
Imports do not initialize Torch/CUDA. No native/frozen source is edited. The
immutable offline bank contains canonical SWA KV, never hidden/FFN outputs.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType

from .core import require, digest, token_digest, validate_digest
from .early_sensitivity import EarlySensitivityConfig, METRIC, REPRESENTATION, plan_from_scores
from .early_mhc_adapter import EarlyMHCAdapter
from .native_reuse import NativeReuseOps, document_plan
from gpu_pilot.early_mhc_protocol import CAPTURE_LAYERS, SPARSE_LAYERS, DEPENDENCIES

MODES = ("native", "capture_independent", "qualified_native", "all_selected", "sensitive")
_MISSING = object()
_LAZY_FIELDS = ("residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix")


def verify_dependencies(root):
    root = Path(root)
    for name, expected in DEPENDENCIES.items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected,
                "frozen sparse operator dependency changed: "+name)
    return dict(DEPENDENCIES)


def tensor_fingerprint(value, torch):
    """Untimed strict bit-level qualification, not a sample or moment sketch."""
    if value is None:
        return None
    require(isinstance(value, torch.Tensor), "qualification expects real tensor")
    require(bool(torch.isfinite(value).all().item()), "nonfinite qualification activation")
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


class EarlyMHCController:
    def __init__(self, layers, *, allow_sync=None, model_sha256, execution_sha256,
                 tokenizer_sha256, dependency_root):
        self.layers = tuple(layers)
        require(tuple(l.layer_idx for l in self.layers) == tuple(range(40)), "actual forty decoder layers required")
        for value in (model_sha256, execution_sha256, tokenizer_sha256):
            validate_digest(value, "cache execution identity")
        self.scope = dict(model_sha256=model_sha256, execution_sha256=execution_sha256,
                          tokenizer_sha256=tokenizer_sha256)
        self.root, self.allow_sync = Path(dependency_root), allow_sync or nullcontext
        self.state, self.mode, self.handles = "NEW", None, []
        self.bank, self.reference, self.adapter = {}, {}, None
        self.prepared_indices, self.prepared_index_scope = {}, None
        self.reference_input = None
        self.qualified = False
        self.cache_output_dir = None
        self.config = EarlySensitivityConfig(metric=METRIC, optional_fractions=(.2, .5, 1.),
                                             representation=REPRESENTATION)

    def _patch(self, obj, name, function):
        old = obj.__dict__.get(name, _MISSING)
        bound = MethodType(function, obj)
        self.handles.append((obj, name, old, bound))
        setattr(obj, name, bound)

    def install(self):
        require(self.state == "NEW", "controller already installed")
        verify_dependencies(self.root)
        sys.path.insert(0, str(self.root/"sparse_query_20261030_v8"))
        import sparse_attention as sa
        import sparse_residual as sr
        require(Path(sa.__file__).resolve() == (self.root/"sparse_query_20261030_v8/sparse_attention.py").resolve()
                and Path(sr.__file__).resolve() == (self.root/"sparse_query_20261030_v8/sparse_residual.py").resolve(),
                "foreign sparse operator import")
        self.sa, self.DevicePlan = sa, sr.DevicePlan
        self.ops = NativeReuseOps.from_deepinfer()
        self.t = self.ops.torch
        try:
            for layer in self.layers:
                require("forward" not in layer.__dict__, "foreign decoder hook")
                original = layer.forward
                def decoder(mod, *args, _original=original, **kwargs):
                    metadata = kwargs.get("input_metadata", args[2] if len(args)>2 else None)
                    marker = getattr(metadata, "canonical_request_binding", None)
                    try:
                        if marker is not None:
                            self._marker(marker)
                            if mod.layer_idx == 0:
                                start, end = marker["start"], marker["end"]
                                require(start == self.coverage and end-start <= 8192, "request chunks not contiguous")
                                self.coverage = end
                        result = _original(*args, **kwargs)
                        if marker is not None and self.mode in ("qualified_native", "all_selected"):
                            with self.allow_sync():
                                hidden, lazy = result
                                entry = dict(output=tensor_fingerprint(hidden, self.t),
                                    lazy={name:tensor_fingerprint(getattr(lazy, name), self.t) for name in _LAZY_FIELDS})
                            key = (marker["start"], marker["end"], mod.layer_idx)
                            require(key not in self.identity_seen, "duplicate qualification layer/chunk")
                            self.identity_seen.add(key)
                            self.observed_identity[key] = entry
                            if self.mode == "qualified_native":
                                self.reference[key] = entry
                            else:
                                require(key in self.reference and entry == self.reference[key],
                                        "new all-selected decoder/mHC differs from native at "+str(key))
                        return result
                    except BaseException:
                        self.state = "FAILED"
                        raise
                self._patch(layer, "forward", decoder)
                if layer.layer_idx in CAPTURE_LAYERS:
                    attn, original_prepare = layer.self_attn, layer.self_attn._prepare_swa_kv
                    require("_prepare_swa_kv" not in attn.__dict__, "foreign SWA preparation hook")
                    def prepare(mod, raw, cache, metadata, use_cp, _original=original_prepare):
                        marker = getattr(metadata, "canonical_request_binding", None)
                        if marker is not None and metadata.is_prompt:
                            self._marker(marker)
                            if self.mode == "capture_independent" or (self.mode == "sensitive" and mod.layer_idx < 6):
                                require(not use_cp and raw.shape == (marker["end"]-marker["start"],512),
                                        "full canonical SWA row geometry changed")
                                canonical = self.ops.normalize(mod, raw)
                                if self.mode == "capture_independent":
                                    self._capture(mod.layer_idx, marker, canonical)
                                else:
                                    self._score(mod.layer_idx, marker, canonical)
                        return _original(raw, cache, metadata, use_cp)
                    self._patch(attn, "_prepare_swa_kv", prepare)
            self.adapter = EarlyMHCAdapter(self.layers, self.get_plan, sa.sparse_attention,
                sa.make_active_metadata, self.offline_provider, self._publish).install()
            self.state = "IDLE"
            return self
        except BaseException:
            self.close()
            raise

    def _marker(self, marker):
        require(self.state == "RUNNING" and type(marker) is dict
                and set(marker) == {"request_id","phase","start","end"}
                and marker["request_id"] == self.rid and marker["phase"] == self.phase
                and type(marker["start"]) is int and type(marker["end"]) is int
                and 0 <= marker["start"] < marker["end"] <= len(self.ids), "foreign active request/chunk")

    def reset(self, rid, mode, *, input_ids, documents=None, phase, fraction=None,
              index_device=None):
        """Prepare request-owned device indices before entering model forward.

        The caller must include this whole operation (including host-to-device
        transfers and any required synchronization) in online preparation time.
        No synchronization exception is installed inside the model forward.
        """
        require(self.state in ("IDLE", "COMPLETE") and mode in MODES, "previous request not cleanly complete")
        require(self.adapter.pending is None and not self.adapter.poisoned, "incomplete masked sublayer")
        require(type(rid) is str and rid and type(phase) is str and phase, "request identity required")
        ids = tuple(input_ids)
        key = token_digest(ids)
        require(mode != "sensitive" or (self.qualified and fraction in (.2,.5)), "new masked qualification required")
        require(mode == "sensitive" or fraction is None, "fraction only for diagnostic sensitive arms")
        if mode in ("qualified_native", "all_selected"):
            require(len(ids) == 16384, "fixed two-chunk qualification required")
            docs = ()
            if mode == "qualified_native":
                require(not self.reference and self.reference_input is None, "qualification reference cannot be overwritten")
                self.reference_input = key
            else:
                require(self.reference_input == key and len(self.reference) == 80, "missing native qualification reference")
        else:
            docs = document_plan(ids, documents, independent=mode=="capture_independent")
        if mode == "capture_independent":
            require(self.cache_output_dir is not None and all((docs[0]["key"],l) not in self.bank for l in CAPTURE_LAYERS),
                    "offline destination missing or cache overwrite")
            require(len(self.bank)+len(CAPTURE_LAYERS) <= 8*len(CAPTURE_LAYERS), "bounded eight-document bank")
        if mode == "sensitive":
            cache_device = None
            for doc in docs:
                for layer in CAPTURE_LAYERS:
                    device = self._entry(doc["key"],layer)["tensor"].device
                    if cache_device is None:
                        cache_device = device
                    require(device == cache_device, "offline bank spans multiple devices")
            require(index_device is None or self.t.device(index_device) == cache_device,
                    "prepared index device differs from authenticated offline bank")
            index_device = cache_device
        prepared = self._prepare_indices(mode, ids, docs, index_device)
        self.rid, self.mode, self.phase, self.fraction = rid, mode, phase, fraction
        self.ids, self.input_sha, self.docs = ids, key, docs
        self.prepared_indices = prepared
        self.prepared_index_scope = (rid, phase, key, mode, fraction)
        self.coverage, self.identity_seen, self.scores, self.plans = 0, set(), {}, {}
        self.observed_identity = {}
        self.capture_coverage = {l:0 for l in CAPTURE_LAYERS}
        self.adapter.events.clear()
        self.adapter.counts = {k:0 for k in self.adapter.counts}
        self.state = "RUNNING"

    def _prepare_indices(self, mode, ids, docs, device):
        """CPU geometry and fixed-size H2D happen only in request preparation."""
        require(self.state in ("IDLE", "COMPLETE"), "index preparation inside active forward is forbidden")
        if mode not in ("sensitive", "all_selected"):
            return {}
        require(device is not None, "explicit model device required before masked forward")
        device = self.t.device(device)
        prepared = {}
        for start in range(0, len(ids), 8192):
            end = min(start+8192, len(ids))
            n = end-start
            if mode == "all_selected":
                fi = self.t.arange(n, device=device, dtype=self.t.long)
                oi, mandatory, optional = None, n, 0
            else:
                geometry = plan_from_scores([0.]*n, start=start, end=end, input_tokens=len(ids),
                    document_spans=[(d["start"],d["end"]) for d in docs], config=self.config)
                fixed = [p-start for p in geometry["mandatory_positions"]]
                candidates = [p-start for p in geometry["optional_positions"]]
                mandatory, optional = len(fixed), len(candidates)
                fi = self.t.tensor(fixed, device=device, dtype=self.t.long)
                oi = self.t.tensor(candidates, device=device, dtype=self.t.long)
            prepared[(start,end)] = dict(fixed=fi, optional=oi,
                mandatory_rows=mandatory, optional_rows=optional)
        return prepared

    def set_cache_output_dir(self, path):
        require(self.state in ("IDLE", "COMPLETE") and not self.bank, "clear cache before changing destination")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.cache_output_dir = path

    def clear_cache(self):
        require(self.state in ("IDLE", "COMPLETE"), "cannot clear active bank")
        self.bank.clear()
        self.cache_output_dir = None

    def _capture(self, layer, marker, canonical):
        start, end = marker["start"],marker["end"]
        require(self.capture_coverage[layer] == start and len(self.ids)==32768, "noncontiguous offline KV capture")
        key = (self.docs[0]["key"],layer)
        if start == 0:
            require(key not in self.bank, "immutable KV overwrite")
            self.bank[key] = dict(tensor=self.ops.allocate(32768,canonical), complete=False,
                scope=dict(self.scope, document_sha256=key[0], layer=layer, representation="swa-postnorm-prerope-bf16"))
        entry = self.bank[key]
        entry["tensor"][start:end].copy_(canonical)
        self.capture_coverage[layer] = end
        if end == 32768:
            entry.update(complete=True, version=entry["tensor"]._version)

    def _entry(self, key, layer):
        entry = self.bank.get((key,layer))
        require(entry is not None and entry["complete"] and entry["tensor"]._version==entry["version"]
                and entry["scope"] == dict(self.scope, document_sha256=key,layer=layer,
                                           representation="swa-postnorm-prerope-bf16")
                and tuple(entry["tensor"].shape)==(32768,512)
                and entry["tensor"].dtype==self.t.bfloat16,
                "missing, foreign or mutated offline KV")
        return entry

    def _canonical(self, layer, start, end, device):
        result = self.t.zeros((end-start,512),dtype=self.t.bfloat16,device=device)
        for doc in self.docs:
            lo,hi = max(start,doc["start"]),min(end,doc["end"])
            if lo<hi:
                entry = self._entry(doc["key"],layer)
                result[lo-start:hi-start].copy_(entry["tensor"][lo-doc["start"]:hi-doc["start"]])
        return result

    def _score(self, layer, marker, canonical):
        key = (marker["start"],marker["end"])
        entry = self.scores.setdefault(key,{})
        require(layer not in entry and set(entry)==set(range(layer)), "early scores missing/out of order")
        cached = self._canonical(layer,*key,canonical.device)
        online, offline = canonical.to(self.t.float64), cached.to(self.t.float64)
        entry[layer] = (online-offline).norm(dim=1)/online.norm(dim=1).clamp(min=1e-8)

    def _publish(self, attn, metadata, runtime):
        if self.state=="RUNNING" and self.mode in ("sensitive","all_selected") and getattr(metadata,"canonical_request_binding",None):
            self.sa.publish_native_source(attn,metadata,runtime)

    def get_plan(self, attn, metadata):
        if self.mode not in ("sensitive","all_selected") or attn.layer_idx not in SPARSE_LAYERS:
            return None
        marker = getattr(metadata,"canonical_request_binding",None)
        if marker is None:
            return None
        self._marker(marker)
        start,end = marker["start"],marker["end"]
        key = (start,end)
        require(metadata.total_q==end-start and not attn.is_kv_source and not attn.has_local_index
                and not attn.is_yoco_hidden_consumer_layer, "protected or compact layer cannot be masked")
        require(self.prepared_index_scope == (self.rid, self.phase, self.input_sha, self.mode, self.fraction)
                and key in self.prepared_indices, "missing or stale request-prepared device indices")
        prepared = self.prepared_indices[key]
        fi, oi = prepared["fixed"], prepared["optional"]
        mandatory, optional = prepared["mandatory_rows"], prepared["optional_rows"]
        require(fi.device == metadata.positions.device and fi.shape == (mandatory,)
                and fi.dtype == self.t.long
                and (oi is None if self.mode == "all_selected" else
                     oi.device == fi.device and oi.shape == (optional,) and oi.dtype == self.t.long),
                "prepared device index geometry changed")
        if key not in self.plans:
            n = end-start
            if self.mode == "all_selected":
                ids, selected = fi, n
                score = None
            else:
                require(key in self.scores and set(self.scores[key])==set(range(6)), "six online scores required before reuse")
                score = self.t.stack([self.scores[key][l] for l in range(6)]).mean(0)
                self.t._assert_async(self.t.isfinite(score).all(), "nonfinite early score; abort instead of stale mask")
                keep = math.ceil(self.fraction*optional)
                require(score.device == fi.device, "score and request-prepared indices are on different devices")
                # All index payloads were transferred by reset, outside forward.
                # Ranking/gather remain device-side with CPU-known output sizes.
                ranking = self.t.argsort(score.index_select(0,oi),descending=True,stable=True)
                ids = self.t.cat((fi,oi.index_select(0,ranking[:keep]))).sort().values.contiguous()
                selected = mandatory+keep
            plan_sha = digest(dict(input_sha256=self.input_sha,start=start,end=end,mode=self.mode,
                                   fraction=self.fraction,score_protocol=self.config.protocol()))
            plan = self.DevicePlan(n,ids,selected,self.rid,self.phase,start,end,plan_sha,self.input_sha,(),None)
            self.plans[key] = dict(plan=plan,score=score,mandatory_rows=mandatory,optional_rows=optional)
        return self.plans[key]["plan"]

    def offline_provider(self, attn, metadata, plan):
        require(self.mode=="sensitive" and self.plans.get((plan.chunk_start,plan.chunk_end),{}).get("plan") is plan,
                "offline KV requested without current early selection")
        return self._canonical(attn.layer_idx,plan.chunk_start,plan.chunk_end,metadata.positions.device)

    def finish(self):
        require(self.state=="RUNNING" and self.coverage==len(self.ids), "incomplete full input coverage")
        adapter = self.adapter.report()
        if self.mode in ("qualified_native","all_selected"):
            require(len(self.identity_seen)==80, "qualification must cover all40 layers in both chunks")
            if self.mode=="all_selected":
                require(self.identity_seen==set(self.reference), "qualification reference coverage changed")
                self.qualified = True
        payloads = []
        if self.mode=="capture_independent":
            require(all(v==32768 for v in self.capture_coverage.values()), "incomplete offline capture")
            for layer in CAPTURE_LAYERS:
                entry = self._entry(self.docs[0]["key"],layer)
                raw = entry["tensor"].detach().contiguous().view(self.t.uint8).cpu().numpy().tobytes()
                name = self.docs[0]["key"]+f".layer{layer}.bf16"
                with (self.cache_output_dir/name).open("xb") as output:
                    output.write(raw)
                payloads.append(dict(scope=entry["scope"],file=name,shape=[32768,512],dtype="bfloat16",
                    bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                entry["payload_sha256"] = payloads[-1]["sha256"]
        masks=[]
        for (start,end),item in sorted(self.plans.items()):
            ids = item["plan"].selected_rows.cpu().tolist()
            record=dict(start=start,end=end,selected_local_rows=ids,selected_rows=len(ids),
                full_rows=end-start,mandatory_rows=item["mandatory_rows"],optional_rows=item["optional_rows"],
                selected_rows_sha256=digest(ids),plan_sha256=item["plan"].plan_sha256)
            record["plan_sha_binds_policy_and_input_not_gpu_row_payload"] = True
            if item["score"] is not None:
                scores=item["score"].cpu().tolist()
                # Save per-token scores and masks, not just medians. No gold is involved.
                record.update(mean_early_relative_l2=scores)
                ref=plan_from_scores(scores,start=start,end=end,input_tokens=len(self.ids),
                    document_spans=[(d["start"],d["end"]) for d in self.docs],config=self.config)
                expected=next(p for p in ref["selections"] if p["optional_fraction"]==self.fraction)
                require([start+i for i in ids]==expected["retained_positions"], "GPU ranking differs from independent CPU oracle")
            masks.append(record)
        if self.mode in ("sensitive","all_selected"):
            expected_chunks=math.ceil(len(self.ids)/8192)
            require(len(masks)==expected_chunks and len(adapter["events"])==expected_chunks*len(SPARSE_LAYERS),
                    "masked layers/chunks missing")
            expected=sum(p["selected_rows"] for p in masks)*len(SPARSE_LAYERS)
            require(adapter["counts"]["selected_rows"]==expected
                    and adapter["counts"]["full_rows"]==len(self.ids)*len(SPARSE_LAYERS), "masked row accounting mismatch")
        c=adapter["counts"]
        self.last_report=dict(schema="early-six-sensitive-mhc-controller-v1",mode=self.mode,phase=self.phase,
            input_ids_sha256=self.input_sha,full_input_tokens=len(self.ids),first_six_full=True,
            sparse_layers=list(SPARSE_LAYERS),offline_payloads=payloads,selection=masks,adapter=adapter,
            actual_swa_reuse_fraction_in_eleven_layers=(1-c["selected_rows"]/c["full_rows"]) if c["full_rows"] else 0.,
            identity_exact=self.mode=="all_selected" and self.qualified,identity_check_count=len(self.identity_seen),
            identity_fingerprints_sha256=digest([{ "key":list(k),"value":v} for k,v in sorted(self.reference.items())]) if self.reference else None,
            identity_fingerprints=[{"key":list(k),"value":v} for k,v in sorted(self.observed_identity.items())],
            actual_full_row_native_mhc=True,formal_admission_unchanged=True,
            diagnostic_reuse_not_accuracy_certified=True,not_model_flops_fraction=True)
        self.state="COMPLETE"
        return self.last_report

    def report(self):
        require(self.state=="COMPLETE", "finish before report")
        return self.last_report

    def close(self):
        if self.state=="CLOSED":
            return
        if self.adapter is not None:
            self.adapter.close(allow_incomplete=True)
        for obj,name,old,bound in reversed(self.handles):
            require(obj.__dict__.get(name,_MISSING) is bound, "controller hook ownership changed")
            if old is _MISSING: delattr(obj,name)
            else: setattr(obj,name,old)
        self.handles.clear()
        self.bank.clear()
        self.prepared_indices.clear()
        self.prepared_index_scope = None
        self.state="CLOSED"
