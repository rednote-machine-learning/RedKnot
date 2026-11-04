"""Independent adjacent-SWA selector with native attention and FFN-only masks.

Experimental SM90/CP1 adapter. Import is CPU-only. This does not modify native
sources, the old joint-sparse controller, or the production admission policy.
All live mutations are owned instance hooks and request-scoped state.
"""

from __future__ import annotations

import copy
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

from .core import digest, require, token_digest, validate_digest
from .early_mhc_adapter import gather_packed_token_rows
from .ffn_only_adapter import FFNOnlyAdapter, make_selected_mlp_metadata
from .native_reuse import NativeReuseOps, document_plan

CAPTURE_LAYERS = tuple(range(40))
FFN_LAYERS = tuple(range(6, 40))
SCORE_LAYERS = (2, 3, 4, 5)
MODES = ("native", "capture_independent", "qualified_native", "all_selected", "sensitive")
_MISSING = object()
_LAZY = ("residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix")


def fingerprint(value, torch):
    if value is None:
        return None
    require(isinstance(value, torch.Tensor), "qualification tensor required")
    require(bool(torch.isfinite(value).all().item()), "nonfinite qualification tensor")
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class FFNOnlyController:
    def __init__(
        self, layers, *, model_sha256, execution_sha256, tokenizer_sha256, dependency_root=None, allow_sync=None, native_model=None
    ):
        self.layers = tuple(layers)
        require(tuple(layer_index.layer_idx for layer_index in self.layers) == tuple(range(40)), "forty native layers required")
        for value in (model_sha256, execution_sha256, tokenizer_sha256):
            validate_digest(value, "cache identity")
        self.scope = dict(model_sha256=model_sha256, execution_sha256=execution_sha256, tokenizer_sha256=tokenizer_sha256)
        self.allow_sync = allow_sync or nullcontext
        self.state, self.mode = "NEW", None
        self.handles, self.bank, self.reference = [], {}, {}
        self.reference_input, self.cache_output_dir = None, None
        self.qualified, self.frame, self.adapter = False, None, None
        self.projections = {}
        self.native_model = native_model
        self.original_yoco_boundary = None
        self.offline_full_layout = False

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
                if layer.layer_idx >= 6:
                    self.projections[layer.layer_idx] = self._make_projections(attn)
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
                        return result
                    except BaseException:
                        self.state = "FAILED"
                        self._restore_offline_layout()
                        raise

                self._patch(layer, "forward", decoder)
                self._install_attention(attn)
            self.adapter = FFNOnlyAdapter(self.layers, self.get_ffn_plan, self._selected_metadata).install()
            self.state = "IDLE"
            return self
        except BaseException:
            self.close()
            raise

    def _install_attention(self, attn):
        original_forward, original_project, original_prepare = attn.forward, attn._project_q_and_swa, attn._prepare_swa_kv

        def forward(mod, hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime=None, **kwargs):
            marker = getattr(input_metadata, "canonical_request_binding", None)
            if marker is None or not input_metadata.is_prompt:
                return original_forward(
                    hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime=cross_layer_runtime, **kwargs
                )
            self._marker(marker)
            require(
                self.frame is None and self.ops.PG.context_parallel_group().size() == 1 and not self.t.cuda.is_current_stream_capturing(),
                "eager single-frame CP1 required",
            )
            require(
                not kwargs
                and all(
                    getattr(input_metadata, k, None) is None
                    for k in ("redknot_binding", "head_intervention_binding", "attention_range_binding", "sparse_phase")
                ),
                "foreign attention intervention",
            )
            frame = dict(layer=mod.layer_idx, metadata=input_metadata, marker=marker, projected=False, prepared=False)
            self.frame = frame
            try:
                if self.mode == "capture_independent":
                    require(
                        self.offline_full_layout
                        and self.native_model.yoco_boundary_layer_idx is None
                        and input_metadata.total_q == marker["end"] - marker["start"]
                        and getattr(cross_layer_runtime, "consumer_full_metadata", None) is None,
                        "offline capture must retain actual per-layer full hidden rows",
                    )
                else:
                    require(
                        not self.offline_full_layout and self.native_model.yoco_boundary_layer_idx == self.original_yoco_boundary,
                        "native online YOCO layout was not restored",
                    )
                result = original_forward(
                    hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime=cross_layer_runtime
                )
                if self.mode in ("sensitive", "all_selected") and mod.layer_idx >= 6:
                    require(frame["projected"] and frame["prepared"], "SWA path did not close")
                return result
            except BaseException:
                self.state = "FAILED"
                self._restore_offline_layout()
                raise
            finally:
                self.frame = None

        def project(mod, hidden, shared_hidden):
            f = self.frame
            if f is None or self.mode not in ("sensitive", "all_selected") or mod.layer_idx < 6:
                return original_project(hidden, shared_hidden)
            require(f["layer"] == mod.layer_idx and not f["projected"], "foreign or repeated SWA projection")
            meta = f["metadata"]
            q_proj, kv_proj = self.projections[mod.layer_idx]
            source = shared_hidden if mod.yoco_swa_share_hidden else hidden
            require(source is not None, "missing native shared hidden")
            n = meta.total_q
            require(self.ops.rows(source) == n and self.ops.rows(hidden) == n, "projection row layout changed")
            layout = self._layout(meta)
            ids = layout["online_ids"] if self.mode == "sensitive" else self.t.arange(n, device=meta.positions.device)
            q = q_proj(hidden)
            if ids.numel():
                selected = self._gather(source, ids)
                raw = kv_proj(selected)
            else:
                raw = q.new_empty((0, 512))
            f.update(projected=True, online_ids=ids, layout=layout)
            if self.mode == "all_selected" and n:
                with self.allow_sync():
                    q_ref, kv_ref = original_project(hidden, shared_hidden)
                    self.ops.identity_error(q, q_ref)
                    self.ops.identity_error(raw, kv_ref)
                    probe_ids = self.t.arange(min(127, n), device=meta.positions.device)
                    partial = kv_proj(self._gather(source, probe_ids))
                    self.ops.identity_error(partial, kv_ref.index_select(0, probe_ids))
                self.projection_checks += 3
                self.projection_events.append(
                    dict(
                        layer=mod.layer_idx,
                        chunk_start=f["marker"]["start"],
                        chunk_end=f["marker"]["end"],
                        full_rows=n,
                        checks=3,
                        max_absolute_error=0.0,
                        checks_named=["split_q", "split_swa", "packed_gather_swa127"],
                    )
                )
            return q, raw

        def prepare(mod, raw, cache, meta, use_cp):
            f = self.frame
            if f is None or not meta.is_prompt:
                return original_prepare(raw, cache, meta, use_cp)
            require(not use_cp and f["layer"] == mod.layer_idx, "unexpected SWA preparation frame")
            if self.mode == "capture_independent":
                self._capture(mod.layer_idx, f["marker"], self.ops.normalize(mod, raw))
            if self.mode == "sensitive" and mod.layer_idx in SCORE_LAYERS:
                self._score(mod.layer_idx, f["marker"], self.ops.normalize(mod, raw))
            if self.mode not in ("sensitive", "all_selected") or mod.layer_idx < 6:
                return original_prepare(raw, cache, meta, use_cp)
            require(f["projected"] and not f["prepared"], "SWA projection/preparation sequence changed")
            n, ids = meta.total_q, f["online_ids"]
            if self.mode == "all_selected" or n == 0:
                result = original_prepare(raw, cache, meta, use_cp)
            else:
                # Only offline rows undergo canonical target-position restoration.
                # The first127 online rows use the original fused norm/RoPE/cast
                # exactly once, not a second RoPE or an extra BF16 norm roundtrip.
                canonical = self._canonical(mod.layer_idx, f["marker"], meta.positions)
                result = self.ops.postprocess(mod, canonical, meta)
                if ids.numel():
                    subset = copy.copy(meta)
                    subset.total_q = ids.numel()
                    subset.positions = meta.positions.index_select(0, ids)
                    prepared, wait = original_prepare(raw, cache, subset, False)
                    require(wait is None and prepared.shape == (ids.numel(), 1, 512), "online SWA subset shape changed")
                    result[0].index_copy_(0, ids, prepared)
            f["prepared"] = True
            self.swa_events.append(
                dict(
                    layer=mod.layer_idx,
                    chunk_start=f["marker"]["start"],
                    chunk_end=f["marker"]["end"],
                    current_positions=meta.positions.clone(),
                    online_local_rows=ids.clone(),
                    online_rows=ids.numel(),
                    reused_rows=n - ids.numel(),
                    full_rows=n,
                    compact=mod.layer_idx >= 20,
                )
            )
            return result

        self._patch(attn, "forward", forward)
        self._patch(attn, "_project_q_and_swa", project)
        self._patch(attn, "_prepare_swa_kv", prepare)

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

    def reset(self, rid, mode, *, input_ids, documents=None, phase, fraction=None, index_device=None):
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
            (mode == "sensitive" and self.qualified and fraction == 0.15) or (mode != "sensitive" and fraction is None),
            "fraction or qualification mismatch",
        )
        if mode in ("qualified_native", "all_selected"):
            require(len(ids) == 16384, "two-chunk numerical qualification required")
            docs = ()
            if mode == "qualified_native":
                require(not self.reference, "qualification reference already set")
                self.reference_input = key
            else:
                require(self.reference_input == key and len(self.reference) == 80, "qualification reference absent")
        else:
            docs = document_plan(ids, documents, independent=mode == "capture_independent")
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
        self.observed, self.scores, self.plans, self.layouts = {}, {}, {}, {}
        self.swa_events, self.ffn_layouts, self.prepared = [], [], {}
        self.capture_coverage = {layer_index: 0 for layer_index in CAPTURE_LAYERS}
        self.adapter.events.clear()
        for k in self.adapter.counts:
            self.adapter.counts[k] = 0
        if mode in ("sensitive", "all_selected"):
            require(index_device is not None, "explicit device required before forward")
            if mode == "sensitive":
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
                self.prepared[(start, end)] = dict(
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

    def _layout(self, meta):
        marker = meta.canonical_request_binding
        self._marker(marker)
        start, end = marker["start"], marker["end"]
        p = meta.positions
        require(p.ndim == 1 and p.numel() == meta.total_q and p.dtype in (self.t.int32, self.t.int64), "invalid current positions")
        local = (p - start).long()
        self.t._assert_async(((local >= 0) & (local < end - start)).all(), "positions outside current input chunk")
        compact = meta.total_q != end - start
        key = (start, end, compact)
        if key not in self.layouts:
            prepared = self.prepared[(start, end)]
            if not compact:
                self.t._assert_async((local == self.t.arange(end - start, device=p.device)).all(), "full layout reordered")
                online_ids = prepared["online"]
            else:
                # One declared data-dependent intersection per compact chunk,
                # included in TTFT; never repeated once per upper layer.
                with self.allow_sync():
                    online_ids = self.t.nonzero(prepared["online_mask"].index_select(0, local), as_tuple=False).flatten()
            self.layouts[key] = dict(positions=p.clone(), local=local, online_ids=online_ids)
        layout = self.layouts[key]
        require(layout["positions"].shape == p.shape, "native layout size changed across layers")
        self.t._assert_async((layout["positions"] == p).all(), "native compact positions changed across layers")
        return layout

    def _score(self, layer, marker, canonical):
        key = (marker["start"], marker["end"])
        values = self.scores.setdefault(key, {})
        require(layer in SCORE_LAYERS and layer not in values and set(values) == set(range(2, layer)), "score layer order changed")
        values[layer] = canonical.clone()

    def _chunk_plan(self, marker):
        key = (marker["start"], marker["end"])
        if key not in self.plans:
            n = key[1] - key[0]
            prep = self.prepared[key]
            if self.mode == "all_selected":
                score = None
                deltas = None
                ids = self.t.arange(n, device=prep["mandatory"].device)
                mandatory, optional = n, 0
            else:
                require(set(self.scores.get(key, {})) == set(SCORE_LAYERS), "four independent SWA layers required")
                values = self.scores[key]
                deltas = []
                for layer_index in (2, 3, 4):
                    a, b = values[layer_index].double(), values[layer_index + 1].double()
                    deltas.append((b - a).norm(dim=1) / a.norm(dim=1).clamp(min=1e-8))
                score = self.t.stack(deltas).mean(0)
                self.t._assert_async(self.t.isfinite(score).all(), "nonfinite adjacent score")
                mandatory, optional = prep["mandatory"].numel(), prep["optional"].numel()
                count = (optional * 15 + 99) // 100
                rank = self.t.argsort(score.index_select(0, prep["optional"]), descending=True, stable=True)
                ids = self.t.cat((prep["mandatory"], prep["optional"].index_select(0, rank[:count]))).sort().values.contiguous()
                del self.scores[key]
            selected = self.t.zeros(n, device=ids.device, dtype=self.t.bool)
            selected[ids] = True
            self.plans[key] = dict(
                ids=ids,
                mask=selected,
                score=score,
                mandatory_rows=mandatory,
                transitions=deltas,
                optional_rows=optional,
                plan_sha256=digest(
                    dict(
                        input_sha256=self.input_sha, start=key[0], end=key[1], mode=self.mode, metric="adjacent-swa-l2-2..5", fraction=0.15
                    )
                ),
            )
        return self.plans[key]

    def get_ffn_plan(self, layer, meta, bf16):
        marker = getattr(meta, "canonical_request_binding", None)
        if not meta.is_prompt or marker is None or self.mode not in ("sensitive", "all_selected") or layer.layer_idx < 6:
            return None
        self._marker(marker)
        layout = self._layout(meta)
        base = self._chunk_plan(marker)
        if "ffn_ids" not in layout:
            if meta.total_q == marker["end"] - marker["start"]:
                layout["ffn_ids"] = base["ids"]
            elif self.mode == "all_selected":
                layout["ffn_ids"] = self.t.arange(meta.total_q, device=meta.positions.device)
            else:
                with self.allow_sync():
                    layout["ffn_ids"] = self.t.nonzero(base["mask"].index_select(0, layout["local"]), as_tuple=False).flatten()
        ids = layout["ffn_ids"]
        plan = SimpleNamespace(
            full_rows=meta.total_q,
            selected_count=ids.numel(),
            selected_rows=ids,
            current_positions=meta.positions,
            request_id=self.rid,
            phase=self.phase,
            chunk_start=marker["start"],
            chunk_end=marker["end"],
            plan_sha256=base["plan_sha256"],
        )
        self.ffn_layouts.append(
            dict(
                layer=layer.layer_idx,
                chunk_start=marker["start"],
                chunk_end=marker["end"],
                current_positions=meta.positions.clone(),
                selected_local_rows=ids.clone(),
                full_rows=meta.total_q,
                selected_rows=ids.numel(),
                plan_sha256=base["plan_sha256"],
            )
        )
        return plan

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
        self.cache_output_dir = None

    def _capture(self, layer, marker, canonical):
        start, end = marker["start"], marker["end"]
        require(
            len(self.ids) == 32768 and self.capture_coverage[layer] == start and canonical.shape == (end - start, 512),
            "offline capture coverage changed",
        )
        key = (self.docs[0]["key"], layer)
        if start == 0:
            require(key not in self.bank, "immutable bank overwrite")
            self.bank[key] = dict(
                tensor=self.ops.allocate(32768, canonical),
                complete=False,
                scope=dict(self.scope, document_sha256=key[0], layer=layer, representation="swa-postnorm-prerope-bf16"),
            )
        entry = self.bank[key]
        entry["tensor"][start:end].copy_(canonical)
        self.capture_coverage[layer] = end
        if end == 32768:
            entry.update(complete=True, version=entry["tensor"]._version)

    def _entry(self, key, layer):
        value = self.bank.get((key, layer))
        require(
            value is not None
            and value["complete"]
            and value["tensor"]._version == value["version"]
            and value["scope"] == dict(self.scope, document_sha256=key, layer=layer, representation="swa-postnorm-prerope-bf16")
            and tuple(value["tensor"].shape) == (32768, 512)
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
            require(all(n == 32768 for n in self.capture_coverage.values()), "offline full-layer capture incomplete")
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
                        shape=[32768, 512],
                        dtype="bfloat16",
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
        masks = []
        for (start, end), value in sorted(self.plans.items()):
            ids = value["ids"].cpu().tolist()
            score = None if value["score"] is None else value["score"].cpu().tolist()
            if score is not None:
                from .adjacent_kv_selector import select_chunk_by_scores

                expected = select_chunk_by_scores(
                    score,
                    input_tokens=len(self.ids),
                    chunk_start=start,
                    chunk_end=end,
                    document_spans=[(d["start"], d["end"]) for d in self.docs],
                )
                require([start + i for i in ids] == expected["ffn_selected_rows"], "GPU selection differs from CPU oracle")
            masks.append(
                dict(
                    start=start,
                    end=end,
                    full_rows=end - start,
                    selected_local_rows=ids,
                    selected_rows=len(ids),
                    selected_rows_sha256=digest(ids),
                    mean_adjacent_relative_l2=score,
                    adjacent_transition_relative_l2=None
                    if value["transitions"] is None
                    else [v.cpu().tolist() for v in value["transitions"]],
                    mandatory_rows=value["mandatory_rows"],
                    optional_rows=value["optional_rows"],
                    plan_sha256=value["plan_sha256"],
                )
            )

        def export(events):
            return [{k: (v.cpu().tolist() if isinstance(v, self.t.Tensor) else v) for k, v in e.items()} for e in events]

        swa, ffn = export(self.swa_events), export(self.ffn_layouts)
        if self.mode in ("sensitive", "all_selected"):
            expected = {(s, min(s + 8192, len(self.ids)), layer_index) for s in range(0, len(self.ids), 8192) for layer_index in FFN_LAYERS}
            for events in (swa, ffn):
                require(
                    len(events) == len(expected) and {(e["chunk_start"], e["chunk_end"], e["layer"]) for e in events} == expected,
                    "missing or duplicate SWA/FFN layer/chunk event",
                )
            if self.mode == "all_selected":
                # Native YOCO can leave a chunk with zero consumer rows. Only
                # nonempty calls can run projection/gather byte comparisons;
                # every such call must carry all three identity checks.
                nonempty = {(e["chunk_start"], e["chunk_end"], e["layer"]) for e in swa if e["full_rows"] > 0}
                actual = {(e["chunk_start"], e["chunk_end"], e["layer"]) for e in self.projection_events}
                require(
                    nonempty
                    and actual == nonempty
                    and len(self.projection_events) == len(nonempty)
                    and all(
                        e["checks"] == 3
                        and e["max_absolute_error"] == 0.0
                        and e["checks_named"] == ["split_q", "split_swa", "packed_gather_swa127"]
                        for e in self.projection_events
                    )
                    and self.projection_checks == 3 * len(nonempty),
                    "split qualification does not cover every nonempty native layer/chunk",
                )
                self.qualified = True
        self.last_report = dict(
            schema="adjacent-swa-ffn-only-controller-v1",
            mode=self.mode,
            phase=self.phase,
            input_ids_sha256=self.input_sha,
            full_input_tokens=len(self.ids),
            first_six_full=True,
            actual_full_row_native_mhc=True,
            all_query_attention_native=True,
            yoco_layout_preserved=True,
            ffn_layers=list(FFN_LAYERS),
            offline_payloads=payloads,
            selection=masks,
            adapter=adapter,
            swa_events=swa,
            ffn_layouts=ffn,
            projection_identity_checks=self.projection_checks,
            projection_identity_events=self.projection_events,
            identity_exact=self.mode == "all_selected" and self.qualified,
            identity_check_count=len(self.observed),
            identity_fingerprints_sha256=digest([dict(key=list(k), value=v) for k, v in sorted(self.reference.items())])
            if self.reference
            else None,
            identity_fingerprints=[dict(key=list(k), value=v) for k, v in sorted(self.observed.items())],
            formal_admission_unchanged=True,
            diagnostic_reuse_not_accuracy_certified=True,
            offline_yoco_full_layout=self.mode == "capture_independent",
            model_boundary_restored=True,
            not_model_flops_fraction=True,
            selected_tokens_forced_through_yoco=False,
        )
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
