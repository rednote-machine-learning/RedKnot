"""Independent post-six-layer SWA-skipping ablation; never a KV-reuse claim.

The native and offline-capture paths are inherited unchanged.  The reference
global-only path computes native Q/SWA and assembles native KV, then restricts
core attention to a private global-KV view.  The optimized path does not compute
or read SWA KV at all after layer five.  Both variants also use global-only
decode, so no missing prompt SWA state is silently consumed by native decode.

The retained independent-document bank is *not* read by the ablated attention.
Its effective reuse is zero.  Importing this module does not import Torch or
DeepInfer.  No native source or weight is modified.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

from .core import digest, require
from .ffn_only import FFNOnlyController, FFN_LAYERS, _LAZY, fingerprint


VARIANTS = ("native", "global_reference", "global_only")


def expected_zos_decode_steps(output_tokens):
    """Native non-MTP ZOS: prompt synchronous, decode has one terminal tail."""
    require(type(output_tokens) is int and output_tokens >= 1, "actual emitted token count required")
    return output_tokens - 1 + int(output_tokens > 1)


def global_buffer_view(loaded, offset, rows):
    """Handle consumers 6/7 whose native source 2 still includes SWA storage."""
    require(type(offset) is int and offset >= 0 and type(rows) is int and rows >= 0,
            "invalid global buffer extent")
    require(loaded is not None and len(loaded.shape) == 3
            and tuple(loaded.shape[1:]) == (1, 512)
            and loaded.shape[0] == offset + rows, "global buffer source layout changed")
    return loaded[offset:]


def global_scheduler_key(attn):
    return f"{attn.kv_compress_ratio}_{attn.has_packed_kv}_False_{attn.layer_cfg.get('index_topk', 0)}"


class SwaSkipController(FFNOnlyController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_variant = "native"
        self.native_qualified = self.global_qualified = False
        self.native_reference_saved = None
        self.global_reference_started = False
        self.global_decode_reference = {}
        self.decode_observed = {}
        self.global_skip_events, self.decode_skip_events = [], []
        self.decode_step = -1
        self.schedule_identity_installed = False
        self.current_schedule = None
        self.scheduler_invocations = []
        self.placeholder_request_id = None
        self.rejected_schedule_identity = None
        self.real_seq_id = None

    def install_schedule_identity(self, model_runner, *, placeholder_request_id):
        """Bind a whole native forward to its CPU SequenceMetadata identity.

        DP ALL/ANY and terminal-cleanup placeholders are ordinary native
        forwards, not generated-token decode steps.  Never classify them from
        token positions, controller activity, or ``not metadata.is_prompt``.
        """
        require(not self.schedule_identity_installed and self.current_schedule is None,
                "scheduler identity already installed")
        require(type(placeholder_request_id) is str and placeholder_request_id,
                "native placeholder request constant required")
        original = model_runner.execute_main_model

        def execute(runner, seq_metadata_list, *args, **kwargs):
            if self.state != "RUNNING":
                return original(seq_metadata_list, *args, **kwargs)
            require(self.current_schedule is None, "overlapping native scheduled forwards")
            try:
                context = self._schedule_context(seq_metadata_list)
            except BaseException as exc:
                self.rejected_schedule_identity = dict(
                    error=type(exc).__name__ + ": " + str(exc),
                    sequences=[{k: getattr(seq, k, None) for k in
                                ("request_id", "seq_id", "is_placeholder", "is_prompt", "seqlen_q", "seqlen_k")}
                               for seq in seq_metadata_list] if isinstance(seq_metadata_list, (list, tuple)) else None)
                self.state = "FAILED"
                raise
            self.scheduler_invocations.append(context)
            self.current_schedule = context
            try:
                result = original(seq_metadata_list, *args, **kwargs)
                require(context["layer_indices"] == list(range(40)),
                        "native scheduled forward omitted/reordered layers")
                context["complete"] = True
                return result
            except BaseException as exc:
                context["error"] = type(exc).__name__ + ": " + str(exc)
                self.state = "FAILED"
                raise
            finally:
                self.current_schedule = None

        self._patch(model_runner, "execute_main_model", execute)
        self.placeholder_request_id = placeholder_request_id
        self.schedule_identity_installed = True
        return self

    def _schedule_context(self, sequences):
        require(isinstance(sequences, (tuple, list)) and len(sequences) == 1,
                "single native scheduled sequence required")
        seq = sequences[0]
        rid, placeholder = getattr(seq, "request_id", None), getattr(seq, "is_placeholder", None)
        prompt = getattr(seq, "is_prompt", None)
        q, k, seq_id = (getattr(seq, name, None) for name in ("seqlen_q", "seqlen_k", "seq_id"))
        require(type(placeholder) is bool and type(prompt) is bool
                and type(seq_id) is int and seq_id >= 0 and type(q) is int and type(k) is int
                and 0 < q <= k, "malformed native scheduled identity")
        if placeholder:
            require(rid == self.placeholder_request_id and seq_id == 0 and q == k == 1,
                    "native placeholder identity/geometry mismatch")
            kind = "placeholder"
        else:
            require(rid == self.rid and rid != self.placeholder_request_id,
                    "foreign real request in controlled native forward")
            require(self.real_seq_id is None or self.real_seq_id == seq_id,
                    "native sequence identity changed within request")
            self.real_seq_id = seq_id
            require((prompt and 0 <= k-q < k <= len(self.ids))
                    or (not prompt and q == 1), "real native request extent mismatch")
            kind = "request"
        return dict(invocation_index=len(self.scheduler_invocations), kind=kind,
                    request_id=rid, seq_id=seq_id, is_prompt=prompt, is_placeholder=placeholder,
                    seqlen_q=q, seqlen_k=k, phase=self.phase,
                    real_decode_step=None, layer_indices=[], layer_rows=[], complete=False)

    def _record_scheduled_layer(self, layer, meta):
        context = self.current_schedule
        require(self.schedule_identity_installed and context is not None,
                "layer has no authenticated native schedule context")
        require(context["phase"] == self.phase and context["is_prompt"] is meta.is_prompt
                and context["layer_indices"] == list(range(layer)), "native schedule/layer ordering changed")
        require(type(meta.total_q) is int and 0 <= meta.total_q <= context["seqlen_q"],
                "native scheduled row geometry changed")
        if layer == 0:
            require(meta.total_q == context["seqlen_q"], "native first-layer rows differ from schedule")
        if context["kind"] == "placeholder":
            require(meta.total_q == 1 and getattr(meta, "canonical_request_binding", None) is None,
                    "placeholder acquired real request metadata")
        elif not meta.is_prompt and layer == 0:
            self.decode_step += 1
            context["real_decode_step"] = self.decode_step
        context["layer_indices"].append(layer)
        context["layer_rows"].append(meta.total_q)
        return context

    def failure_diagnostics(self):
        """Host-only evidence is safe even after a CUDA/qualification error."""
        contexts = [dict(e, layer_indices=list(e["layer_indices"]), layer_rows=list(e["layer_rows"]))
                    for e in self.scheduler_invocations]
        return dict(schema="swa-skip-native-schedule-diagnostics-v1", state=self.state,
                    phase=getattr(self, "phase", None), request_id=getattr(self, "rid", None),
                    attention_variant=self.attention_variant, real_decode_steps=self.decode_step+1,
                    scheduler_invocations=contexts, scheduler_counts=self._scheduler_counts(),
                    rejected_schedule_identity=self.rejected_schedule_identity,
                    prefill_fingerprint_count=len(getattr(self, "observed", {})),
                    decode_fingerprint_count=len(self.decode_observed),
                    prefill_attention_events=len(self.global_skip_events),
                    decode_attention_events=len(self.decode_skip_events))

    def _scheduler_counts(self):
        return dict(total=len(self.scheduler_invocations),
                    real_prefill=sum(e["kind"] == "request" and e["is_prompt"] for e in self.scheduler_invocations),
                    real_decode=sum(e["kind"] == "request" and not e["is_prompt"] for e in self.scheduler_invocations),
                    placeholder_prefill=sum(e["kind"] == "placeholder" and e["is_prompt"] for e in self.scheduler_invocations),
                    placeholder_decode=sum(e["kind"] == "placeholder" and not e["is_prompt"] for e in self.scheduler_invocations))

    def _replace_owned_hook(self, obj, name, function):
        """Chain only our own hook, retaining one ownership record for close()."""
        matches = [(i, h) for i, h in enumerate(self.handles) if h[0] is obj and h[1] == name]
        require(len(matches) == 1, "missing/duplicate owned hook: " + name)
        i, (_, _, old, installed) = matches[0]
        require(obj.__dict__.get(name) is installed, "foreign replacement of owned hook: " + name)
        bound = MethodType(function, obj)
        self.handles[i] = (obj, name, old, bound)
        setattr(obj, name, bound)

    def install(self):
        try:
            super().install()
            for layer in self.layers:
                previous = layer.forward

                def decoder(mod, *args, _previous=previous, **kwargs):
                    meta = kwargs.get("input_metadata", args[2] if len(args) > 2 else None)
                    context = None
                    if self.state == "RUNNING":
                        context = self._record_scheduled_layer(mod.layer_idx, meta)
                    decode = (self.state == "RUNNING" and self.attention_variant != "native"
                              and context["kind"] == "request" and not meta.is_prompt)
                    try:
                        result = _previous(*args, **kwargs)
                        if decode and self.mode in ("qualified_native", "all_selected"):
                            hidden, lazy = result
                            with self.allow_sync():
                                value = dict(output=fingerprint(hidden, self.t),
                                             lazy={k: fingerprint(getattr(lazy, k), self.t) for k in _LAZY})
                            key = (self.decode_step, mod.layer_idx)
                            require(key not in self.decode_observed, "duplicate decode qualification layer")
                            self.decode_observed[key] = value
                            if self.attention_variant == "global_reference":
                                self.global_decode_reference[key] = value
                            else:
                                require(self.global_decode_reference.get(key) == value,
                                        "global-only decode/mHC byte parity failed: " + str(key))
                        return result
                    except BaseException:
                        self.state = "FAILED"
                        raise

                self._replace_owned_hook(layer, "forward", decoder)
            return self
        except BaseException:
            self.close()
            raise

    def _active(self, attn):
        return (self.state == "RUNNING" and self.attention_variant != "native" and attn.layer_idx >= 6
                and self.current_schedule is not None and self.current_schedule["kind"] == "request")

    @contextmanager
    def _without_swa(self, attn):
        """Restore only flags we own, including on a forward/qualification error."""
        require(attn.use_swa is True and attn.has_packed_kv, "native SWA/global topology changed")
        old_key, new_key = attn.sched_meta_key, global_scheduler_key(attn)
        attn.use_swa, attn.sched_meta_key = False, new_key
        try:
            yield
        finally:
            require(attn.use_swa is False and attn.sched_meta_key == new_key,
                    "foreign mutation of request-owned attention flags")
            attn.use_swa, attn.sched_meta_key = True, old_key

    def _install_attention(self, attn):
        native_forward = attn.forward
        native_prefill, native_decode = attn._prefill_attention, attn._decode_attention
        native_buffer = attn._prepare_prefill_kv
        super()._install_attention(attn)
        previous_forward, previous_project, previous_prepare = attn.forward, attn._project_q_and_swa, attn._prepare_swa_kv

        def forward(mod, hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                    cross_layer_runtime=None, **kwargs):
            if not self._active(mod) or self.attention_variant != "global_only":
                return previous_forward(hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                                        cross_layer_runtime=cross_layer_runtime, **kwargs)
            require(not kwargs and self.frame is None and self.ops.PG.context_parallel_group().size() == 1,
                    "global-only path requires owned CP1 frame")
            require(not self.t.cuda.is_current_stream_capturing(), "global-only path requires eager execution")
            require(all(getattr(input_metadata, k, None) is None for k in
                        ("redknot_binding", "head_intervention_binding", "attention_range_binding", "sparse_phase")),
                    "foreign attention intervention")
            marker = getattr(input_metadata, "canonical_request_binding", None)
            if input_metadata.is_prompt:
                require(marker is not None, "global-only prefill requires canonical marker")
                self._marker(marker)
            require(not self.offline_full_layout and self.native_model.yoco_boundary_layer_idx == self.original_yoco_boundary,
                    "global-only online YOCO layout changed")
            frame = dict(layer=mod.layer_idx, metadata=input_metadata, marker=marker,
                         projected=False, prepared=False, skip_optimized=True)
            self.frame = frame
            try:
                with self._without_swa(mod):
                    result = native_forward(hidden_states, kv_cache, input_metadata, hidden_states_bf16,
                                            cross_layer_runtime=cross_layer_runtime)
                require(frame["projected"] and frame["prepared"], "global-only Q/preparation path incomplete")
                return result
            except BaseException:
                self.state = "FAILED"
                raise
            finally:
                self.frame = None

        def project(mod, hidden, shared_hidden):
            if not (self._active(mod) and self.attention_variant == "global_only"):
                return previous_project(hidden, shared_hidden)
            frame = self.frame
            require(frame is not None and frame["layer"] == mod.layer_idx and not frame["projected"]
                    and mod.use_swa is False, "unowned/repeated global-only projection")
            q = self.projections[mod.layer_idx][0](hidden)
            require(self.ops.rows(q) == frame["metadata"].total_q, "global-only Q row count changed")
            frame["projected"] = True
            return q, None

        def prepare(mod, raw, cache, meta, use_cp):
            if not (self._active(mod) and self.attention_variant == "global_only"):
                return previous_prepare(raw, cache, meta, use_cp)
            frame = self.frame
            require(frame is not None and frame["layer"] == mod.layer_idx and frame["projected"]
                    and not frame["prepared"] and raw is None and not use_cp and mod.use_swa is False,
                    "global-only path must not create SWA KV")
            frame["prepared"] = True
            return None, None

        def buffer(mod, total_kv, main_kv_offset, device, cross_layer_runtime):
            if not (self._active(mod) and self.attention_variant == "global_only"):
                return native_buffer(total_kv, main_kv_offset, device, cross_layer_runtime)
            require(mod.use_swa is False and main_kv_offset == 0, "global-only buffer contains a SWA prefix")
            if mod.use_shared_kv and not mod.is_kv_source:
                return global_buffer_view(cross_layer_runtime.loaded_prefill_kv,
                                          cross_layer_runtime.loaded_prefill_kv_main_offset, total_kv)
            return native_buffer(total_kv, main_kv_offset, device, cross_layer_runtime)

        def prefill(mod, q_b, kv, kv_ca_index, kv_swa_full_index, total_k_swa_cached,
                    total_k_swa_cur, input_metadata):
            if not self._active(mod):
                return native_prefill(q_b, kv, kv_ca_index, kv_swa_full_index, total_k_swa_cached,
                                      total_k_swa_cur, input_metadata)
            require(kv_ca_index is not None and mod.has_packed_kv, "global-only attention requires global indices")
            if self.attention_variant == "global_reference":
                require(mod.use_swa is True, "reference path lost native SWA assembly")
                offset = total_k_swa_cached + total_k_swa_cur
                global_kv = global_buffer_view(kv, offset, kv.shape[0] - offset)
                with self._without_swa(mod):
                    result = native_prefill(q_b, global_kv, kv_ca_index, None, 0, 0, input_metadata)
            else:
                require(mod.use_swa is False and total_k_swa_cached == total_k_swa_cur == 0
                        and kv_swa_full_index is None, "optimized attention includes SWA keys")
                result = native_prefill(q_b, kv, kv_ca_index, None, 0, 0, input_metadata)
            self._record_attention(mod, input_metadata, decode=False)
            return result

        def decode(mod, q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata):
            if not self._active(mod):
                return native_decode(q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata)
            require(kv_ca_index is not None and mod.has_packed_kv, "global-only decode requires global indices")
            if self.attention_variant == "global_reference":
                with self._without_swa(mod):
                    result = native_decode(q_b, kv_ca_index, None, kv_cache, input_metadata)
            else:
                require(mod.use_swa is False and kv_swa_index is None
                        and mod.sched_meta_key == global_scheduler_key(mod), "optimized decode includes SWA state")
                result = native_decode(q_b, kv_ca_index, None, kv_cache, input_metadata)
            self._record_attention(mod, input_metadata, decode=True)
            return result

        self._replace_owned_hook(attn, "forward", forward)
        self._replace_owned_hook(attn, "_project_q_and_swa", project)
        self._replace_owned_hook(attn, "_prepare_swa_kv", prepare)
        self._patch(attn, "_prepare_prefill_kv", buffer)
        self._patch(attn, "_prefill_attention", prefill)
        self._patch(attn, "_decode_attention", decode)

    def _record_attention(self, mod, meta, *, decode):
        count = meta.total_q
        event = dict(layer=mod.layer_idx, current_positions=meta.positions.clone(), full_rows=count,
                     scheduler_invocation_index=self.current_schedule["invocation_index"],
                     global_query_rows=count, swa_projection_rows=count if self.attention_variant == "global_reference" else 0,
                     swa_attention_query_rows=0, swa_attention_kv_reads=0, offline_kv_attention_reads=0,
                     attention_variant=self.attention_variant)
        if decode:
            require(self.decode_step >= 0, "decode attention occurred outside layer sequence")
            event["decode_step"] = self.decode_step
            self.decode_skip_events.append(event)
        else:
            marker = meta.canonical_request_binding
            self._marker(marker)
            event.update(chunk_start=marker["start"], chunk_end=marker["end"])
            self.global_skip_events.append(event)

    def reset(self, rid, mode, *, attention_variant="native", **kwargs):
        require(self.state in ("IDLE", "COMPLETE"), "unclean previous global-only request")
        require(attention_variant in VARIANTS, "unknown attention variant")
        require((attention_variant == "native")
                or (attention_variant == "global_reference" and mode == "qualified_native")
                or (attention_variant == "global_only" and mode in ("all_selected", "sensitive")),
                "attention variant/mode mismatch")
        if attention_variant == "global_reference":
            require(self.native_qualified and not self.global_reference_started and len(self.reference) == 80,
                    "native qualification must precede independent global reference")
            self.native_reference_saved = (dict(self.reference), self.reference_input)
            self.reference, self.reference_input = {}, None
            self.global_reference_started = True
            self.global_decode_reference = {}
        elif attention_variant == "global_only":
            require(self.global_reference_started and len(self.reference) == 80
                    and len(self.global_decode_reference) == 80, "global reference prefill/decode incomplete")
            if mode == "sensitive":
                require(self.global_qualified, "global-only numerical qualification is required")
        elif mode in ("qualified_native", "all_selected"):
            require(not self.global_reference_started, "native qualification cannot overwrite global reference")
        super().reset(rid, mode, **kwargs)
        self.attention_variant = attention_variant
        self.global_skip_events, self.decode_skip_events, self.decode_observed = [], [], {}
        self.decode_step = -1
        self.scheduler_invocations = []
        self.rejected_schedule_identity = None
        self.real_seq_id = None

    def _export_events(self, events):
        return [{k: (v.cpu().tolist() if isinstance(v, self.t.Tensor) else v) for k, v in event.items()} for event in events]

    def finish(self, *, output_tokens=None):
        try:
            return self._finish_impl(output_tokens=output_tokens)
        except BaseException:
            self.state = "FAILED"
            raise

    def _finish_impl(self, *, output_tokens=None):
        if self.attention_variant == "native":
            report = super().finish()
            if self.mode == "all_selected":
                self.native_qualified = self.qualified
            report.update(attention_variant="native", global_identity_exact=False)
            report.update(self._schedule_report())
            self.last_report = report
            return report
        require(self.state == "RUNNING" and self.coverage == len(self.ids), "global-only request did not close")
        require(self.schedule_identity_installed and self.current_schedule is None
                and self.scheduler_invocations and all(e["complete"] and e["layer_indices"] == list(range(40))
                                                     for e in self.scheduler_invocations),
                "native schedule evidence incomplete")
        expected_steps = expected_zos_decode_steps(output_tokens)
        require(self.decode_step + 1 == expected_steps, "native ZOS decode/tail count differs from emitted output tokens")
        require(not self.offline_full_layout and self.native_model.yoco_boundary_layer_idx == self.original_yoco_boundary,
                "global-only request changed online layout")
        adapter = self.adapter.report()
        expected = {(s, min(s + 8192, len(self.ids)), layer) for s in range(0, len(self.ids), 8192) for layer in FFN_LAYERS}
        events, ffn = self._export_events(self.global_skip_events), self._export_events(self.ffn_layouts)
        actual = {(e["chunk_start"], e["chunk_end"], e["layer"]) for e in events}
        require(len(events) == len(expected) and actual == expected, "global-only prefill coverage incomplete")
        require(all(e["swa_attention_query_rows"] == e["swa_attention_kv_reads"] == e["offline_kv_attention_reads"] == 0
                    and e["global_query_rows"] == e["full_rows"]
                    and e["swa_projection_rows"] == (e["full_rows"] if self.attention_variant == "global_reference" else 0)
                    for e in events), "global-only attention counters invalid")
        decode = self._export_events(self.decode_skip_events)
        for event in decode:
            event["terminal_tail"] = bool(output_tokens > 1 and event["decode_step"] == output_tokens-1)
        expected_decode = {(step, layer) for step in range(self.decode_step + 1) for layer in FFN_LAYERS}
        require(len(decode) == len(expected_decode)
                and {(e["decode_step"], e["layer"]) for e in decode} == expected_decode,
                "global-only decode coverage incomplete")
        require(all(e["swa_attention_query_rows"] == e["swa_attention_kv_reads"] == e["offline_kv_attention_reads"] == 0
                    and e["global_query_rows"] == e["full_rows"]
                    and e["swa_projection_rows"] == (e["full_rows"] if self.attention_variant == "global_reference" else 0)
                    for e in decode), "global-only decode counters invalid")
        if self.mode in ("qualified_native", "all_selected"):
            require(output_tokens == 2 and len(self.observed) == 80 and len(self.decode_observed) == 80
                    and set(self.decode_observed) == {(step, layer) for step in range(2) for layer in range(40)},
                    "global-only qualification requires 80 prefill and 80 decode/tail layer fingerprints")
            if self.mode == "all_selected":
                require(self.observed == self.reference and self.decode_observed == self.global_decode_reference,
                        "global-only prefill/decode qualification byte parity failed")
                self.global_qualified = True
        if self.mode in ("all_selected", "sensitive"):
            require(len(ffn) == len(expected)
                    and {(e["chunk_start"], e["chunk_end"], e["layer"]) for e in ffn} == expected,
                    "global-only FFN layer/chunk coverage incomplete")
        else:
            require(not ffn and not self.plans, "reference qualification must use native full FFN")
        masks = []
        for (start, end), value in sorted(self.plans.items()):
            ids = value["ids"].cpu().tolist()
            score = None if value["score"] is None else value["score"].cpu().tolist()
            if score is not None:
                from .adjacent_kv_selector import select_chunk_by_scores
                check = select_chunk_by_scores(score, input_tokens=len(self.ids), chunk_start=start, chunk_end=end,
                                               document_spans=[(d["start"], d["end"]) for d in self.docs])
                require([start + i for i in ids] == check["ffn_selected_rows"], "GPU selection differs from CPU oracle")
            masks.append(dict(start=start, end=end, full_rows=end-start, selected_local_rows=ids, selected_rows=len(ids),
                              selected_rows_sha256=digest(ids), mean_adjacent_relative_l2=score,
                              adjacent_transition_relative_l2=None if value["transitions"] is None else [v.cpu().tolist() for v in value["transitions"]],
                              mandatory_rows=value["mandatory_rows"], optional_rows=value["optional_rows"], plan_sha256=value["plan_sha256"]))
        report = dict(schema="swa-skip-controller-v2", mode=self.mode, phase=self.phase,
                      **self._schedule_report(),
                      attention_variant=self.attention_variant, input_ids_sha256=self.input_sha, full_input_tokens=len(self.ids),
                      first_six_full=True, actual_full_row_native_mhc=True, all_query_attention_native=False,
                      yoco_layout_preserved=True, ffn_layers=list(FFN_LAYERS), offline_payloads=[],
                      selection=masks, adapter=adapter, swa_events=[], ffn_layouts=ffn,
                      global_skip_events=events, decode_skip_events=decode,
                      projection_identity_checks=0, projection_identity_events=[], identity_exact=False,
                      identity_check_count=len(self.observed),
                      identity_fingerprints_sha256=digest([dict(key=list(k), value=v) for k, v in sorted(self.observed.items())]),
                      identity_fingerprints=[dict(key=list(k), value=v) for k, v in sorted(self.observed.items())],
                      global_identity_exact=self.mode == "all_selected" and self.global_qualified,
                      global_reference_input_sha256=self.reference_input,
                      decode_identity_check_count=len(self.decode_observed),
                      decode_identity_fingerprints=[dict(key=list(k), value=v) for k, v in sorted(self.decode_observed.items())],
                      decode_identity_fingerprints_sha256=digest([dict(key=list(k), value=v) for k, v in sorted(self.decode_observed.items())]),
                      decode_reference_fingerprints_sha256=digest([dict(key=list(k), value=v) for k, v in sorted(self.global_decode_reference.items())]),
                      emitted_output_tokens=output_tokens, decode_steps=self.decode_step+1,
                      effective_decode_steps=output_tokens-1, terminal_tail_decode_steps=int(output_tokens>1),
                      terminal_tail_decode_step=(output_tokens-1) if output_tokens>1 else None,
                      decode_schedule=dict(schema="native-zos-mtp0-one-terminal-tail-after-decode-v1",
                                           native_zero_overhead_scheduling=True, output_tokens=output_tokens,
                                           output_driving_steps=output_tokens-1,
                                           terminal_tail_steps=int(output_tokens>1), actual_steps=expected_steps),
                      native_qualification_preserved=self.native_qualified,
                      formal_admission_unchanged=True, diagnostic_reuse_not_accuracy_certified=True,
                      offline_yoco_full_layout=False, model_boundary_restored=True, not_model_flops_fraction=True,
                      selected_tokens_forced_through_yoco=False,
                      empty_ep_qualification_sha256=None if self.empty_ep_capability is None else self.empty_ep_capability.sha256,
                      post6_global_only_prefill_and_decode=True, retained_offline_bank_entries=len(self.bank),
                      retained_offline_bank_attention_reads=0, effective_swa_kv_reuse_rate=0.0,
                      skip_is_not_effective_swa_kv_reuse=True)
        self.last_report, self.state = report, "COMPLETE"
        return report

    def _schedule_report(self):
        require(self.schedule_identity_installed and self.current_schedule is None
                and self.scheduler_invocations and all(e["complete"] and e["layer_indices"] == list(range(40))
                                                     for e in self.scheduler_invocations),
                "native schedule evidence incomplete")
        return dict(scheduler_identity_schema="native-sequence-whole-forward-v1",
                    request_id=self.rid, native_placeholder_request_id=self.placeholder_request_id,
                    scheduler_invocations=self.scheduler_invocations,
                    scheduler_invocations_sha256=digest(self.scheduler_invocations),
                    scheduler_counts=self._scheduler_counts())
