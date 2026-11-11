"""Cached document SWA KV + online native global KV, with unchanged FFN15.

Only document KV generation is skipped: Q, native joint SWA/global attention,
global compression/index, and decode remain native.  No old controller is edited.
Imports are CPU-only.  Native RedKnotV4Attention is supported only without its
separate transparent_swa_binding session, avoiding stacked cache controllers.
"""
from __future__ import annotations

from types import MethodType

from .core import digest, require
from .ffn_only import FFNOnlyController


POLICY = "all-cached-document-rows-no-boundary-repair"


class SwaReuseQOnlyController(FFNOnlyController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_attention_events = []
        self.cache_restore_identity_events = []
        self.canonical_copy_events = []

    def _replace_owned_hook(self, obj, name, function):
        matches = [(i, h) for i, h in enumerate(self.handles) if h[0] is obj and h[1] == name]
        require(len(matches) == 1, "owned controller hook required: " + name)
        i, (_, _, old, installed) = matches[0]
        require(obj.__dict__.get(name) is installed, "foreign hook replacement: " + name)
        bound = MethodType(function, obj)
        self.handles[i] = (obj, name, old, bound)
        setattr(obj, name, bound)

    def _configure_document_reuse(self):
        """Reuse existing uploaded mandatory IDs; do not add a forward sync."""
        if self.mode != "sensitive":
            return
        for prepared in self.prepared.values():
            prepared["online"] = prepared["mandatory"]
            prepared["online_mask"].fill_(False).index_fill_(0, prepared["mandatory"], True)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        try:
            self._configure_document_reuse()
            self.joint_attention_events = []
            self.cache_restore_identity_events = []
            self.canonical_copy_events = []
        except BaseException:
            self.state = "FAILED"
            raise

    def _canonical(self, layer, marker, positions):
        result = super()._canonical(layer, marker, positions)
        copied = sum(max(0, min(marker["end"], d["end"])-max(marker["start"], d["start"])) for d in self.docs)
        self.canonical_copy_events.append(dict(layer=layer, chunk_start=marker["start"], chunk_end=marker["end"],
            copied_document_rows=copied, canonical_bf16_bytes=copied*512*2,
            restored_current_rows=positions.numel()))
        return result

    def _install_attention(self, attn):
        native_joint = attn._prefill_attention
        native_global = attn._compute_kv_ca_index
        super()._install_attention(attn)
        previous_forward, previous_prepare = attn.forward, attn._prepare_swa_kv

        def forward(mod, hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime=None, **kwargs):
            require(getattr(input_metadata, "transparent_swa_binding", None) is None,
                    "independent cache controller cannot overlap a native transparent SWA session")
            return previous_forward(hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                                    cross_layer_runtime=cross_layer_runtime, **kwargs)

        def prepare(mod, raw, cache, meta, use_cp):
            result = previous_prepare(raw, cache, meta, use_cp)
            frame = self.frame
            if frame is not None and meta.is_prompt and mod.layer_idx >= 6 and self.mode in ("all_selected", "sensitive"):
                require(isinstance(result, tuple) and len(result) == 2 and result[1] is None
                        and result[0].shape == (meta.total_q, 1, 512), "native CP1 prepared SWA contract changed")
                frame["reuse_prepared_swa"] = result[0]
                frame["reuse_restore_exact"] = False
                if self.mode == "all_selected" and meta.total_q:
                    with self.allow_sync():
                        restored, wait = self.ops.postprocess(mod, self.ops.normalize(mod, raw), meta)
                        require(wait is None, "canonical restore introduced CP wait")
                        self.ops.identity_error(restored, result[0])
                    frame["reuse_restore_exact"] = True
            return result

        def compute_global(mod, *args, **kwargs):
            result = native_global(*args, **kwargs)
            frame = self.frame
            if frame is not None and mod.layer_idx >= 6 and self.mode in ("all_selected", "sensitive"):
                require("reuse_global_index" not in frame and mod.has_packed_kv and result is not None,
                        "native online global index path missing or repeated")
                frame["reuse_global_index"] = result
            return result

        def joint(mod, q_b, kv, kv_ca_index, kv_swa_full_index, total_k_swa_cached, total_k_swa_cur, meta):
            frame = self.frame
            instrument = frame is not None and mod.layer_idx >= 6 and self.mode in ("all_selected", "sensitive")
            if instrument:
                require(mod.use_swa is True and mod.has_packed_kv is True
                        and frame.get("reuse_global_index") is kv_ca_index and kv_ca_index is not None
                        and kv_swa_full_index is meta.kv_swa_full_index and kv_swa_full_index is not None,
                        "native joint SWA/global inputs were replaced")
                prepared = frame.get("reuse_prepared_swa")
                require(prepared is not None and total_k_swa_cur == meta.total_q
                        and prepared.shape == (total_k_swa_cur,1,512)
                        and kv.shape == (total_k_swa_cached+total_k_swa_cur+meta.total_k_compressed[mod.ratio_key],1,512),
                        "assembled native local/global KV geometry changed")
                if self.mode == "all_selected" and meta.total_q:
                    require(frame["reuse_restore_exact"], "canonical restore byte qualification missing")
                    with self.allow_sync():
                        self.ops.identity_error(kv[total_k_swa_cached:total_k_swa_cached+total_k_swa_cur], prepared)
                    self.cache_restore_identity_events.append(dict(layer=mod.layer_idx,
                        chunk_start=frame["marker"]["start"], chunk_end=frame["marker"]["end"],
                        full_rows=meta.total_q, checks=2, max_absolute_error=0.0,
                        checks_named=["canonical_norm_rope_cast", "assembled_swa_rows"]))
                swa = self.swa_events[-1]
                require((swa["layer"],swa["chunk_start"],swa["chunk_end"],swa["full_rows"]) ==
                        (mod.layer_idx,frame["marker"]["start"],frame["marker"]["end"],meta.total_q),
                        "joint attention has no matching actual SWA preparation")
            result = native_joint(q_b, kv, kv_ca_index, kv_swa_full_index,
                                  total_k_swa_cached, total_k_swa_cur, meta)
            if instrument:
                self.joint_attention_events.append(dict(layer=mod.layer_idx,
                    chunk_start=frame["marker"]["start"], chunk_end=frame["marker"]["end"],
                    full_rows=meta.total_q, online_swa_rows=swa["online_rows"], reused_swa_rows=swa["reused_rows"],
                    swa_prefix_rows=total_k_swa_cached, current_swa_rows=total_k_swa_cur,
                    global_kv_rows=meta.total_k_compressed[mod.ratio_key], kv_rows=kv.shape[0],
                    use_swa=mod.use_swa, has_packed_kv=mod.has_packed_kv, global_compute_called=True,
                    native_joint_attention_called=True, native_swa_indices_preserved=True,
                    is_global_source=mod.is_kv_source, uses_shared_global_kv=mod.use_shared_kv))
            return result

        self._replace_owned_hook(attn, "forward", forward)
        self._replace_owned_hook(attn, "_prepare_swa_kv", prepare)
        self._patch(attn, "_compute_kv_ca_index", compute_global)
        self._patch(attn, "_prefill_attention", joint)

    def finish(self):
        try:
            report = super().finish()
            if self.mode in ("sensitive", "all_selected"):
                expected = {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in report["swa_events"]}
                require(len(self.joint_attention_events) == len(expected)
                        and {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in self.joint_attention_events} == expected,
                        "native joint attention call coverage missing")
                if self.mode == "all_selected":
                    nonempty = {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in report["swa_events"] if e["full_rows"]}
                    require(len(self.cache_restore_identity_events) == len(nonempty)
                            and {(e["chunk_start"],e["chunk_end"],e["layer"]) for e in self.cache_restore_identity_events} == nonempty,
                            "canonical cache restoration qualification incomplete")
            report.update(schema="swa-reuse-qonly-ffn15-controller-v1", swa_reuse_policy=POLICY,
                document_boundary_online_tokens=0, online_swa_scope="non-document-template-question-only",
                decode_native=True, joint_attention_native=True, global_kv_native_online=True,
                joint_attention_events=self.joint_attention_events,
                joint_attention_events_sha256=digest(self.joint_attention_events),
                cache_restore_identity_events=self.cache_restore_identity_events,
                cache_restore_identity_checks=sum(e["checks"] for e in self.cache_restore_identity_events),
                canonical_copy_events=self.canonical_copy_events,
                canonical_cache_copy_bytes=sum(e["canonical_bf16_bytes"] for e in self.canonical_copy_events),
                bytes_are_canonical_copies_not_measured_device_bandwidth=True,
                effective_cache_attention_rows=sum(e["reused_swa_rows"] for e in self.joint_attention_events))
            self.last_report = report
            return report
        except BaseException:
            self.state = "FAILED"
            raise
