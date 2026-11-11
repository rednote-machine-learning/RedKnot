"""One hook owner for native, static-tail, and dynamic-head/tail comparisons.

Both cached policies consume the same immutable GPU full-document canonical
bank. Static tail visibility/selection is unchanged from TailLinkController;
its historical pinned-CPU staging cost is intentionally absent in this new arm.
No second controller is installed and no qualification state crosses policies.
"""
from __future__ import annotations

import copy

from .core import digest, require
from .headtail_link import HeadTailLinkController, POLICY_MODES, DOCUMENT, WIDTH
from .tail_link import TailLinkController

ARMS = ("recomputed", "tail128_link", "headtail128_link")
CACHED_ARMS = ARMS[1:]
DIAGNOSTIC_LAYERS = frozenset((0, 1, 2, 3, 7, 13, 19, 20, 39))


class ComparisonController(HeadTailLinkController):
    _doc_ordinal = TailLinkController._doc_ordinal

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = None
        self.active_policy = None
        self.policy_qualifications = {}
        self.bank_generation = 0
        self.bank_case_binding = None

    def install(self):
        # Check actual host-side module configuration before any formal request;
        # a diagnostic mismatch must not first surface after fifteen phases.
        if self.natural_boundary_inputs:
            from gpu_pilot.natural_boundary_protocol import natural_boundary_protocol
            expected = natural_boundary_protocol()["diagnostics"]
        else:
            from gpu_pilot.headtail_scale20_protocol import headtail_scale20_protocol
            expected = headtail_scale20_protocol()["diagnostics"]
        modules = {layer.layer_idx:layer.self_attn for layer in self.layers[:3]
                   if getattr(layer.self_attn,"has_packed_kv",None) is True}
        require(sorted(modules) == expected["native_prefix_routing_layers"]
                and all(type(getattr(layer.self_attn,"has_packed_kv",None)) is bool
                        for layer in self.layers[:3]), "native diagnostic routing layer topology differs from protocol")
        require({str(layer):module.kv_compress_ratio for layer,module in modules.items()}
                == expected["native_prefix_routing_compress_ratios"],
                "native diagnostic compression ratios differ from protocol")
        require(all(type(module.is_sparse) is bool for module in modules.values())
                and {str(layer):"sparse_topk" if module.is_sparse else "dense_causal_enumeration"
                     for layer,module in modules.items()} == expected["native_prefix_routing_selection_kinds"],
                "native diagnostic selection kind differs from protocol")
        require(all(module.layer_cfg.get("index_topk") == expected["native_index_topk"]
                    for module in modules.values()), "native diagnostic topk width differs from protocol")
        require(tuple(expected["kv_error_layers"]) == tuple(sorted(DIAGNOSTIC_LAYERS))
                and all(getattr(self.layers[layer].self_attn,"use_swa",None) is True
                        for layer in expected["kv_error_layers"]),
                "native diagnostic layers require SWA projections")
        return super().install()

    def _clear_phase(self):
        super()._clear_phase()
        self.restored = {}
        self.nondoc = {}
        self.staging = None
        self.tail_memory_events = []
        self.tail_restore_events = []
        self.gpu_tail_restore_events = []
        self.memory_assembly_events = []
        self.cpu_plan_oracles = set()
        self.dynamic_document_swa = False
        self.dynamic_swa_events = []
        self.dynamic_refreshed = {}
        self.dynamic_chunk_end = {}
        self.diagnostic = False
        self.diagnostic_observer = None

    def _patch(self, obj, name, function):
        if name == "_prepare_swa_kv":
            original_hook = function
            owner = self
            def observed_prepare(mod, raw, cache, meta, use_cp):
                result = original_hook(mod, raw, cache, meta, use_cp)
                if (owner.diagnostic and owner.diagnostic_observer is not None
                        and getattr(meta, "is_prompt", False) and mod.layer_idx in DIAGNOSTIC_LAYERS
                        and getattr(meta, "canonical_request_binding", None) is not None and raw is not None):
                    require(owner.mode == "native" and owner.arm == "recomputed",
                            "diagnostic SWA observation escaped native-only phase")
                    owner.diagnostic_observer.observe_swa(mod, raw, meta)
                return result
            function = observed_prepare
        return super()._patch(obj, name, function)

    def _install_attention(self, attn):
        super()._install_attention(attn)
        if attn.layer_idx in (0, 1, 2):
            original_index = attn._compute_kv_ca_index
            owner = self
            def observed_index(mod, *args, **kwargs):
                result = original_index(*args, **kwargs)
                meta = kwargs.get("input_metadata", args[4] if len(args) > 4 else None)
                marker = getattr(meta, "canonical_request_binding", None)
                if (owner.diagnostic and owner.diagnostic_observer is not None and result is not None
                        and getattr(meta, "is_prompt", False) and marker is not None
                        and marker["end"] > max(doc["end"] for doc in owner.docs)):
                    require(owner.mode == "native" and owner.arm == "recomputed",
                            "diagnostic index observation escaped native-only phase")
                    # Observer gathers only query positions >= the final document
                    # end, and labels top-index incidence as a relevance proxy.
                    owner.diagnostic_observer.observe_global_index(mod, result, meta)
                return result
            self._patch(attn, "_compute_kv_ca_index", observed_index)

    def _bind_policy(self, arm):
        require(arm in CACHED_ARMS, "unknown cached comparison policy")
        if self.active_policy is not None:
            self.policy_qualifications[self.active_policy] = dict(
                reference=self.sparse_reference, input=self.sparse_reference_input,
                qualified=self.sparse_qualified)
        value = self.policy_qualifications.setdefault(arm, dict(reference={}, input=None, qualified=False))
        self.sparse_reference = value["reference"]
        self.sparse_reference_input = value["input"]
        self.sparse_qualified = value["qualified"]
        self.active_policy = arm

    @staticmethod
    def _validate_arm_mode(arm, mode):
        valid = ((mode == "native" and arm == "recomputed")
            or (mode in POLICY_MODES and arm in CACHED_ARMS)
            or (mode in ("qualified_native", "all_selected") and arm == "qualification")
            or (mode == "capture_independent" and arm == "offline"))
        require(valid, "comparison arm/mode mismatch")

    def reset(self, rid, mode, *, arm, input_ids, documents=None, phase, fraction=None,
              index_device=None, verify_sparse=False, diagnostic=False):
        self._validate_arm_mode(arm, mode)
        require(type(diagnostic) is bool and (not diagnostic or (mode == "native" and arm == "recomputed")),
                "diagnostics require a separate native recomputed phase")
        require(self.state in ("IDLE", "COMPLETE") and self.frame is None,
                "cannot switch comparison arm during an active or failed phase")
        require(fraction is None, "comparison policies have no fraction")
        if mode in POLICY_MODES:
            self._bind_policy(arm)
        self.arm = arm
        result = super().reset(rid, mode, input_ids=input_ids, documents=documents, phase=phase,
            fraction=None, index_device=index_device, verify_sparse=verify_sparse)
        if arm in ARMS:
            self._bind_case_bank()
        self.dynamic_document_swa = arm == "headtail128_link" and mode in POLICY_MODES
        self.diagnostic = diagnostic
        if diagnostic:
            from .early_token_diagnostics import EarlyTokenDiagnostics
            self.diagnostic_observer = EarlyTokenDiagnostics(controller=self)
        return result

    def _bind_case_bank(self):
        """Native denominator also retains the identical bank and memory load."""
        expected = {(doc["key"], layer) for doc in self.docs for layer in range(40)}
        require(len(expected) == 320 and set(self.bank) == expected,
                "all three formal arms require the same complete 320-entry bank")
        device = self.layers[0].self_attn.q_a_kv_swa_proj.weight.device
        binding = []
        for key in sorted(expected):
            entry = self._entry(*key)
            require(entry["tensor"].device == device, "shared full bank left the model device")
            binding.append((key, id(entry["tensor"]), entry["version"], digest(entry["scope"])))
        binding = (self.bank_generation, tuple(binding))
        require(self.bank_case_binding is None or self.bank_case_binding == binding,
                "formal comparison bank changed between arms or repeats")
        self.bank_case_binding = binding

    def _selected_attention(self, mod, hidden, cache, meta, bf16, cross):
        if self.mode in POLICY_MODES and self.arm == "tail128_link":
            return TailLinkController._selected_attention(self, mod, hidden, cache, meta, bf16, cross)
        require(self.mode == "all_selected" or (self.mode in POLICY_MODES and self.arm == "headtail128_link"),
                "unexpected active comparison attention arm")
        return super()._selected_attention(mod, hidden, cache, meta, bf16, cross)

    def _refresh_document_tails(self, mod, hidden, bf16, shared, cache, meta, current, tails):
        require(self.arm == "tail128_link" and self.mode in POLICY_MODES
                and not self.dynamic_document_swa, "static arm cannot refresh document KV")
        return []

    def _assemble_memory(self, event, tails, nondoc_positions, nondoc_tensor):
        if not self.natural_boundary_inputs:
            return TailLinkController._assemble_memory(self, event, tails, nondoc_positions, nondoc_tensor)
        out = tails.new_empty((event["memory_count"], 1, WIDTH))
        offsets, offset = [], 0
        for doc in self.docs:
            offsets.append(offset)
            offset += min(128, doc["end"] - doc["start"])
        require(offset == len(self.tail_positions_cpu) == tails.shape[0], "variable static tail bank geometry")
        non_lookup = {position:i for i, position in enumerate(nondoc_positions)}
        doc_dst, doc_src, other_dst, other_src = [], [], [], []
        for row, (position, ordinal, bankrow) in enumerate(zip(event["memory_positions"],
                event["memory_doc_indices"], event["memory_tail_bank_rows"])):
            if ordinal >= 0:
                doc_dst.append(row)
                doc_src.append(offsets[ordinal] + bankrow)
            else:
                require(position in non_lookup, "unmaterialized nondocument KV")
                other_dst.append(row)
                other_src.append(non_lookup[position])
        def idx(values): return self._upload_indices(values, tails.device)
        if doc_dst:
            out.index_copy_(0, idx(doc_dst), tails.index_select(0, idx(doc_src)))
        if other_dst:
            out.index_copy_(0, idx(other_dst), nondoc_tensor.index_select(0, idx(other_src)))
        if self.mode == "reference_sparse" or self.verify_sparse:
            by_position = {p:tails[i] for i,p in enumerate(self.tail_positions_cpu)}
            by_position.update({p:nondoc_tensor[i] for i,p in enumerate(nondoc_positions)})
            expected = self.t.stack([by_position[p] for p in event["memory_positions"]]) if len(out) else out.clone()
            if len(out):
                with self.allow_sync():
                    self.ops.identity_error(out, expected)
            self.memory_assembly_events.append(dict(schema="tail-link-memory-assembly-identity-v1",
                layer=event["layer"], chunk_start=event["chunk_start"], chunk_end=event["chunk_end"],
                checks=1, max_absolute_error=0.0, rows=len(out)))
        return out

    def _restored_tails(self, mod, meta):
        require(self.arm == "tail128_link" and self.mode in POLICY_MODES,
                "static GPU tail bank used by another arm")
        layer = mod.layer_idx
        if layer not in self.restored:
            entries = [self._entry(doc["key"], layer)["tensor"] for doc in self.docs]
            lengths = [doc["end"] - doc["start"] for doc in self.docs]
            tail_rows = sum(min(128, length) for length in lengths)
            require(len(entries) == 8 and all(tuple(value.shape) == (length, WIDTH)
                and value.device == meta.positions.device for value, length in zip(entries, lengths)),
                "static comparison requires the complete shared GPU bank")
            canonical = self.t.cat([value[-128:] for value in entries], dim=0)
            subset = copy.copy(meta)
            subset.total_q = tail_rows
            subset.positions = self._upload_indices(self.tail_positions_cpu, meta.positions.device, meta.positions.dtype)
            restored, wait = self.ops.postprocess(mod, canonical, subset)
            require(wait is None and tuple(restored.shape) == (tail_rows, 1, WIDTH),
                    "static GPU tail absolute-position restore changed")
            checked = self.mode == "reference_sparse" or self.verify_sparse
            if checked:
                gathers = [self.t.arange(max(0, length - 128), length, device=meta.positions.device)
                           for length in lengths]
                independent = self.t.cat([value.index_select(0, gather)
                                         for value, gather in zip(entries, gathers)], dim=0)
                pieces = []
                for doc, value, gather in zip(self.docs, entries, gathers):
                    positions = self.t.arange(max(doc["start"], doc["end"] - 128), doc["end"],
                        device=meta.positions.device, dtype=meta.positions.dtype)
                    transformed = self.ops.transform(value.index_select(0, gather), norm_args=None,
                        rope_args=self.ops.RopeArgs(positions, mod.rotary_emb.cos_sin_cache, 64,
                            mod.rotary_emb.is_neox_style, offset=448),
                        cast_args=self.ops.CastArgs("e4m3", mod.swa_cache_num_per_channels,
                            round_sf=True, sf_format="ue8m0", cast_back=True), out_args=None)
                    pieces.append(transformed.reshape(len(gather), 1, WIDTH))
                with self.allow_sync():
                    self.ops.identity_error(canonical, independent)
                    self.ops.identity_error(restored, self.t.cat(pieces, dim=0))
                self.tail_restore_events.append(dict(layer=layer, rows=tail_rows, checks=2,
                    checks_named=["gpu_full_bank_tail_slice", "absolute_rope_nativecast"],
                    max_absolute_error=0.0, absolute_positions_sha256=digest(self.tail_positions_cpu)))
            self.gpu_tail_restore_events.append(dict(layer=layer, rows=tail_rows, source="shared_gpu_full_canonical_bank",
                source_rows_per_document=lengths if self.natural_boundary_inputs else DOCUMENT,
                source_local_start=[max(0, length-128) for length in lengths] if self.natural_boundary_inputs else DOCUMENT-128,
                checked=checked, checks=2 if checked else 0,
                checks_named=["gpu_full_bank_tail_slice", "absolute_rope_nativecast"] if checked else [],
                max_absolute_error=0.0 if checked else None,
                absolute_positions_sha256=digest(self.tail_positions_cpu),
                phase_generation=self.phase_generation, host_to_device_copy=False))
            self.restored[layer] = restored
        return self.restored[layer]

    def _comparison_fields(self):
        count = len(self.bank)
        bank_rows = sum(self._document_length(key[0]) for key in self.bank)
        scopes = [dict(document=key[0], layer=key[1], scope=value["scope"])
                  for key, value in sorted(self.bank.items())]
        return dict(comparison_schema="headtail-natural-boundary-comparison-v1" if self.natural_boundary_inputs
                    else "headtail-three-arm-comparison-v1", comparison_arm=self.arm,
            single_installed_hook_owner=True, independent_policy_qualification=True,
            qualification_policy=self.active_policy if self.mode in POLICY_MODES else None,
            cache_storage_policy="shared_gpu_full_canonical_bank_all_arms",
            gpu_bank_constant_across_arms=True, offline_gpu_resident=True,
            offline_bank_representation="swa-postnorm-prerope-bf16", offline_bank_immutable=True,
            offline_bank_entry_count=count, offline_bank_rows=bank_rows,
            offline_bank_bytes=bank_rows*WIDTH*2, offline_bank_scope_sha256=digest(scopes),
            natural_boundary_inputs=self.natural_boundary_inputs,
            document_token_lengths=[d["end"] - d["start"] for d in getattr(self,"docs",())],
            document_length_policy="natural_disjoint_token_spans" if self.natural_boundary_inputs else "fixed_32768",
            bank_generation=self.bank_generation, phase_generation=self.phase_generation,
            cross_phase_overlay_reuse=False, phase_overlay_fresh=True,
            cached_document_kv_consumed=self.mode in POLICY_MODES,
            cold_cache_ttft_claim=False, historical_static_cpu_staging_comparable=False)

    def finish(self):
        if self.mode in POLICY_MODES and self.arm == "tail128_link":
            require(len(self.gpu_tail_restore_events) == 37
                    and {event["layer"] for event in self.gpu_tail_restore_events} == set(range(3, 40)),
                    "static GPU tail restoration coverage incomplete")
            report = TailLinkController.finish(self)
            report.update(schema="tail-link-gpu-bank-controller-v1", head_tokens=0,
                static_tail_source="shared_gpu_full_bank_tail128_slice", static_swa_projection_reused=True,
                native_local_window_preserved=False, native_local_mask_preserved=False,
                full_bank_retained_but_static_attention_tail_only=True,
                gpu_tail_restore_events=self.gpu_tail_restore_events,
                tail_bank_source_shape=[DOCUMENT, WIDTH], staging=None)
            if self.natural_boundary_inputs:
                report.pop("tail_bank_source_shape")
                report["tail_bank_source_shapes"] = [[doc["end"]-doc["start"], WIDTH] for doc in self.docs]
        else:
            report = super().finish()
        if self.mode not in POLICY_MODES:
            report.update(head_tokens=0, tail_tokens=0, dynamic_refresh_start_layer=None,
                dynamic_document_swa=False, extra_completed_document_tail_links=False,
                extra_links_cross_yoco_rectangle=False, phase_local_boundary_overlay=False,
                boundary_overlay_rows_per_layer=0)
        report.update(self._comparison_fields())
        report.update(diagnostic=self.diagnostic, early_token_diagnostics=None)
        if self.diagnostic:
            require(self.diagnostic_observer is not None and self.mode == "native" and self.arm == "recomputed",
                    "native diagnostic observer absent or attached to candidate")
            report["early_token_diagnostics"] = self.diagnostic_observer.finish()
        self.last_report = report
        return report

    def clear_cache(self):
        require(self.state in ("IDLE", "COMPLETE"), "cannot change bank during a comparison phase")
        result = super().clear_cache()
        self.bank_generation += 1
        self.bank_case_binding = None
        return result
