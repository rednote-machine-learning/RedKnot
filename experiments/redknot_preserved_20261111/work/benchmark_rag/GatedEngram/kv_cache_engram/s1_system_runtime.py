"""Owned S1 system adapter and explicitly synchronized DP worker batch loops.

The disabled path delegates to the original controller. Deferred deep batches
are executable only after eight worker-installation acknowledgements; formal
use additionally requires the original per-chunk numerical fingerprints.
"""
from __future__ import annotations

from types import FunctionType, MethodType, SimpleNamespace as NS

from .core import digest, require, token_digest, validate_digest
from .s1_runtime import (S1Controller, gather_hidden, gather_lazy,
                         compact_metadata, rebind_sampler_output)
from .s1_system_attention import S1SystemAttention
from .s1_system_cache import S1SystemCache
from .s1_system_batch import (CrossChunkDeepBatch, batch_schedule,
                              deferred_sampler_output, validate_schedule_agreement, REGISTERED_BATCH_GROUPS)


def inactive_empty_query_cu(metadata, torch):
    """No host-list transfer inside the native strict-sync forward scope."""
    return torch.zeros((2,),device=metadata.cu_seqlens_q.device,dtype=metadata.cu_seqlens_q.dtype)


class S1SystemController(S1Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configured_system = False
        self._configured_group = 1
        self.system_qualified = {width: False for width in (1, *REGISTERED_BATCH_GROUPS)}
        self.batch_schedule = self._batch = self._active_deep_batch = None
        self._batch_step_trace = []
        self._empty_group_case = None

    def install(self):
        super().install()
        try:
            require(not self.cache.bank and self.case_binding is None,
                    "system adapter must install before any bank/request")
            self.cache = S1SystemCache(layers=self.layers, scope=self.scope,
                torch_module=self.t, allow_sync=self.allow_sync)
            self.attention = S1SystemAttention(self.cache, torch_module=self.t,
                                              allow_sync=self.allow_sync)
            return self
        except BaseException:
            self.close()
            raise

    def configure(self, *, system_enabled, batch_group_chunks=1):
        require(type(system_enabled) is bool and type(batch_group_chunks) is int
                and batch_group_chunks in (1, *REGISTERED_BATCH_GROUPS), "invalid S1 system configuration")
        require(system_enabled or batch_group_chunks == 1, "batching requires system arm")
        require(self.state in ("NEW", "IDLE", "COMPLETE"), "cannot reconfigure active S1 request")
        self._configured_system, self._configured_group = system_enabled, batch_group_chunks
        return self

    @staticmethod
    def _empty_group_documents(ids, documents, source_full_input_ids_sha256):
        validate_digest(source_full_input_ids_sha256, "empty-group original full input")
        require(len(ids) == 32769 and isinstance(documents, (list, tuple)) and len(documents) == 1,
                "empty-group qualification requires one exact32769-token prefix")
        d = documents[0]
        key = token_digest(ids)
        require(d.get("input_start") == 0 and d.get("input_end") == 32769
                and d.get("token_ids_sha256") == key and d.get("document_id", key) == key
                and d.get("offline_position", 0) == 0, "empty-group document identity")
        return (dict(start=0, end=32769, key=key),)

    def prepare_case(self, *, input_ids, documents, index_device,
                     system_empty_group_qualification=False, source_full_input_ids_sha256=None):
        if not system_empty_group_qualification:
            self._empty_group_case = None
            return super().prepare_case(input_ids=input_ids, documents=documents, index_device=index_device)
        require(self.state in ("IDLE", "COMPLETE"), "cannot bind active empty-group qualification")
        docs = self._empty_group_documents(input_ids, documents, source_full_input_ids_sha256)
        self.cache.prepare_case(documents=docs, input_tokens=len(input_ids), index_device=index_device)
        self.case_input = token_digest(input_ids)
        self.case_binding = tuple((key, id(e["tensor"]), e["version"]) for key,e in sorted(self.cache.bank.items()))
        self._empty_group_case = self.case_input, source_full_input_ids_sha256

    def reset(self, *args, system_empty_group_qualification=False, source_full_input_ids_sha256=None, **kwargs):
        mode = args[1] if len(args) > 1 else kwargs.get("mode")
        require(not self._configured_system or mode == "sensitive",
                "only compact S1 may enable system changes; reference remains original")
        if self._configured_system and not kwargs.get("verify_sparse", False):
            require(self.system_qualified[self._configured_group],
                    "system configuration lacks actual per-chunk numerical qualification")
        self.cache.configure(enabled=self._configured_system)
        self.attention.configure(enabled=self._configured_system)
        if system_empty_group_qualification:
            require(mode in ("reference_sparse", "sensitive") and "empty_group" in kwargs.get("phase", ""),
                    "empty-group geometry is diagnostic-only")
            require(self._empty_group_case == (token_digest(kwargs["input_ids"]), source_full_input_ids_sha256),
                    "empty-group case/source binding differs")
            def validator(ids, documents, **unused):
                return self._empty_group_documents(ids, documents, source_full_input_ids_sha256)
            # Reuse the original reset bytecode with one isolated validator.
            # No native module/global is patched, including during execution.
            original = S1Controller.reset
            isolated = FunctionType(original.__code__, dict(original.__globals__, s1_documents=validator),
                                    original.__name__, original.__defaults__, original.__closure__)
            isolated.__kwdefaults__ = original.__kwdefaults__
            isolated(self, *args, **kwargs)
        else:
            super().reset(*args, **kwargs)
        if mode == "capture_independent" or self.case_binding is None:
            # The original controller does not begin a cache phase before a
            # case bank exists or during capture. Clear only phase workspace /
            # diagnostics here so a prior enabled case cannot leak its events.
            self.cache.begin_phase(self.phase, build_global_base=False)
        self.system_enabled, self.batch_group_chunks = self._configured_system, self._configured_group
        self.batch_schedule = self._batch = self._active_deep_batch = None
        self._batch_step_trace, self._source2_before_batch = [], []
        if self.batch_group_chunks > 1:
            self.batch_schedule = batch_schedule(self.docs, len(self.ids), request_id=self.rid,
                phase=self.phase, phase_generation=self.cache.phase_generation,
                input_ids_sha256=self.input_sha, group_chunks=self.batch_group_chunks)

    def bind_batch_schedule(self, schedule, receipts):
        require(self.state == "RUNNING" and self.coverage == 0 and self._batch is None
                and self.batch_schedule is not None and schedule == self.batch_schedule,
                "batch schedule must bind before the first native chunk")
        validate_schedule_agreement(schedule, receipts)
        self._batch = CrossChunkDeepBatch(torch_module=self.t, documents=self.docs,
            full_input_tokens=len(self.ids), request_id=self.rid, phase=self.phase,
            phase_generation=self.cache.phase_generation, input_ids_sha256=self.input_sha,
            group_chunks=self.batch_group_chunks, all_rank_receipts=receipts)

    def _marker(self, marker):
        group = self._active_deep_batch
        if group is None:
            return super()._marker(marker)
        require(self.state == "RUNNING" and isinstance(marker, dict)
                and marker.get("request_id") == self.rid and marker.get("phase") == self.phase
                and (marker.get("start"), marker.get("end")) == (group.start, group.end),
                "S1 grouped deep marker differs from current owned frame")
        return group.start, group.end

    def _observe_layer(self, layer, start, end, result, meta):
        group = self._active_deep_batch
        if group is None or layer < 3:
            return super()._observe_layer(layer, start, end, result, meta)
        require((start, end) == (group.start, group.end), "deep observation group mismatch")
        hidden, lazy = result
        original_ctx, offset = self.ctx, 0
        try:
            for ordinal, (a, b) in enumerate(group.member_chunks):
                positions = [p for p in group.positions_cpu if a <= p < b]
                if self._qualifying_sparse():
                    rows = self.t.arange(offset, offset+len(positions), device=hidden.device, dtype=self.t.long)
                    self.ctx = NS(**vars(original_ctx))
                    self.ctx.positions_cpu = positions
                    member = gather_hidden(hidden, rows, self.t), gather_lazy(lazy, rows)
                    super()._observe_layer(layer, a, b, member, meta)
                else:
                    # Formal accounting must not launch gathers of hidden and
                    # four lazy tensors just to describe the CPU-owned split.
                    self.decoder_events.append(dict(layer=layer, chunk_start=a, chunk_end=b,
                        full_rows=b-a, selected_positions=positions, selected_rows=len(positions),
                        actual_attention_rows=len(positions), actual_ffn_rows=len(positions),
                        actual_residual_rows=len(positions), actual_mhc_rows=len(positions),
                        dense_reference_only=False, original_absolute_positions=True))
                event = self.decoder_events[-1]
                owner = ordinal == len(group.member_chunks)-1
                event.update(system_deep_batch_member=True, logical_member_rows=True,
                    group_start=group.start, group_end=group.end, native_ep_call_owner=owner,
                    empty_native_ep_call=owner and not group.positions_cpu)
                offset += len(positions)
            require(offset == hidden.shape[0], "deep member output rows not exhausted")
        finally:
            self.ctx = original_ctx

    def _policy_model_forward(self, model, caches, meta, hidden, dflash):
        if self.batch_group_chunks == 1:
            return super()._policy_model_forward(model, caches, meta, hidden, dflash)
        require(self.system_enabled and self.mode == "sensitive" and self._batch is not None,
                "all-rank batch schedule was not installed")
        require(self.ctx is None and dflash is None and not model.dflash_input_size,
                "S1 batch model frame/MTP mismatch")
        start, end = self._marker(meta.canonical_request_binding)
        require(meta.total_q == end-start and meta.num_reqs == 1, "native batch member geometry")
        g, t = self.native_symbols, self.t
        hidden = model.embed_tokens(meta.input_ids) if hidden is None else hidden
        residual, cross = None, g["CrossLayerRuntime"]()
        require(model.engram_runtime is not None and meta.ngram_token_ids is not None,
                "batch requires original Engram history")
        hashes = g["tile_kernels"].engram.engram_hash(meta.ngram_token_ids, model.ngram_multipliers,
            model.ngram_vocab_sizes, model.ngram_offsets, getattr(meta, "images_seq_mask", None))
        hooks = model.engram_runtime.fetch(hashes)
        executed = []
        for i in range(3):
            hidden, residual = self.layers[i](g["Movable"](hidden), caches[i], meta, residual,
                embed_hook=hooks.get(i), cross_layer_runtime=cross, dflash_input=None)
            executed.append(i)
        member = self.prepared[start, end]
        if self._qualifying_sparse():
            with self.allow_sync():
                before = self.attention.source2_fingerprint(caches[2], meta, cross)
            self._source2_before_batch.append((start, end, before))
        group = self._batch.push(full_meta=meta, hidden_after_layer2=hidden,
            lazy_after_layer2=residual, cross_after_layer2=cross,
            selected_rows=member["rows"], selected_positions_cpu=member["plan"]["selected_positions"],
            engram14_hook=hooks.get(14))
        self.engram_events.append(dict(layer=14, chunk_start=start, chunk_end=end,
            original_hash_rows=end-start, selected_positions=member["plan"]["selected_positions"],
            actual_engram_rows=member["plan"]["selected_rows"], hook_consumed_once=True,
            original_ngram_hashes_preserved=True, system_deferred_gate=True,
            empty_hook_consumed_without_fake_rows=member["plan"]["selected_rows"] == 0))
        if group is None:
            output = deferred_sampler_output(hidden, full_input_tokens=len(self.ids), start=start, end=end)
            self._batch_step_trace.append(dict(start=start, end=end, flush=False, layers=executed,
                                               deep_native_ep_calls=0))
            return g["CausalLMForwardOutput"](hidden_states=output)

        self._active_deep_batch = group
        self.ctx = NS(full_meta=group.full_meta, compact_meta=group.compact_meta,
            start=group.start, end=group.end, positions=group.compact_meta.positions,
            positions_cpu=group.positions_cpu, selected_rows=group.selected_rows,
            selected_count=len(group.positions_cpu), docs=self.docs, input_tokens=len(self.ids),
            mode=self.mode, phase=self.phase, rid=self.rid, plan=group.plan, prepared=group.prepared,
            verify_sparse=self.verify_sparse, verify=self._qualifying_sparse(),
            reference_hidden=None, reference_bf16=None, reference_selected_rows=group.selected_rows)
        hidden, residual, cross = group.hidden, group.lazy, group.cross
        try:
            for i in range(3, 40):
                layer = self.layers[i]
                if not group.positions_cpu:
                    if i == 14: group.engram14_hook()
                    hidden, residual = self._empty_decoder(layer, caches[i], self.ctx, cross, hidden, residual)
                    self._observe_layer(i, group.start, group.end, (hidden, residual), group.compact_meta)
                else:
                    hidden, residual = layer(g["Movable"](hidden), caches[i], group.compact_meta, residual,
                        embed_hook=group.engram14_hook if i == 14 else None,
                        cross_layer_runtime=cross, dflash_input=None)
                executed.append(i)
            require(group.engram14_consumption[0], "merged Engram14 gate never consumed its embeddings")
            if self._qualifying_sparse():
                require(len(self._source2_before_batch) == len(group.member_metadata), "source2 member proof count")
                for (a, b, before), original_meta in zip(self._source2_before_batch, group.member_metadata):
                    with self.allow_sync():
                        after = self.attention.source2_fingerprint(caches[2], original_meta)
                    require(before == after, "system batch changed a native source2 prefix")
                    self.source2_events.append(dict(chunk_start=a, chunk_end=b, before=before, after=after, exact=True))
                self._source2_before_batch = []
            if group.positions_cpu:
                if hidden.dim() == 2:
                    hidden = g["HCLayerMixin"].hc_post(hidden, residual)
                require(hidden.ndim == 3, "S1 grouped final mHC trunk")
                hidden = g["mhc_pre_apply_mix"](hidden, residual.pre_layer_mix) if model.hc_shifted else model.hc_head(hidden)
                hidden = model.output_norm(hidden)
            else:
                hidden = hidden.new_empty((0, model.output_norm.weight.numel()))
            p = group.prepared
            hidden = rebind_sampler_output(hidden, p["sampler_output_rows"], p["sample_dst"], p["sample_src"])
            self._batch_step_trace.append(dict(start=start, end=end, flush=True, layers=executed,
                group_start=group.start, group_end=group.end, deep_native_ep_calls=37,
                empty_deep_native_ep_calls=37 if not group.positions_cpu else 0))
            return g["CausalLMForwardOutput"](hidden_states=hidden)
        finally:
            self.ctx = self._active_deep_batch = None

    def finish(self):
        batch = None
        if self.batch_schedule is not None:
            require(self._batch is not None and not self._source2_before_batch, "batch completion/proof missing")
            batch = self._batch.finish()
        report = super().finish()
        if self.system_enabled and self.verify_sparse:
            require(report["sparse_identity_exact"], "system qualification lacks original chunk byte parity")
            self.system_qualified[self.batch_group_chunks] = True
        report.update(system_enabled=self.system_enabled, batch_group_chunks=self.batch_group_chunks,
            system_numerically_qualified=self.system_qualified[self.batch_group_chunks] if self.system_enabled else None,
            system_batch_execution=batch, system_batch_step_trace=list(self._batch_step_trace),
            system_policy_changes=False, system_kernel_order_changed=self.batch_group_chunks > 1,
            system_original_per_chunk_fingerprints=True)
        return report

    def clear_cache(self, *, reset_sparse_qualification=False):
        super().clear_cache(reset_sparse_qualification=reset_sparse_qualification)
        if reset_sparse_qualification:
            self.system_qualified = {width: False for width in (1, *REGISTERED_BATCH_GROUPS)}
        self._empty_group_case = None

    def close(self):
        if self._batch is not None and self._batch.state == "ACTIVE":
            self._batch.discard()
        super().close()


def _validate_schedule(schedule):
    require(isinstance(schedule, dict) and schedule.get("schema") == "s1-system-batch-schedule-v1"
        and schedule.get("sha256") == digest({k:v for k,v in schedule.items() if k != "sha256"}),
        "worker batch schedule seal")
    n, width = schedule["full_input_tokens"], schedule["group_chunks"]
    require(type(n) is int and n > 0 and type(width) is int and width in REGISTERED_BATCH_GROUPS,
            "worker batch schedule geometry")
    steps = schedule["steps"]
    require(len(steps) == (n+8191)//8192, "worker batch step count")
    for i, step in enumerate(steps):
        start, end = i*8192, min((i+1)*8192, n)
        flush = (i+1)%width == 0 or end == n
        require(step["start"] == start and step["end"] == end and step["flush"] is flush
                and step["layers"] == (list(range(40)) if flush else [0,1,2]), "worker EP schedule order")


def classify_world_step(samples, schedule, ordinal, prefill_completed):
    """Classify actual CPU scheduler outputs, never placeholder tensor metadata.

    NORMAL scheduling can emit empty decode rounds before/between prefill
    chunks. They execute native placeholder work without advancing our queue.
    This fixed CP1/DP8 experiment owns exactly one real request on rank zero.
    """
    require(isinstance(samples, list) and len(samples) == 8, "eight scheduler observations required")
    for rank, sample in enumerate(samples):
        require(isinstance(sample, dict) and sample.get("rank") == rank
                and sample.get("schedule_sha256") == schedule["sha256"]
                and sample.get("ordinal") == ordinal
                and sample.get("prefill_completed") == prefill_completed,
                "world scheduler identity/progress mismatch")
        require(type(sample.get("prompt_run")) is bool and type(sample.get("is_empty")) is bool
                and isinstance(sample.get("sequences"), list)
                and sample["is_empty"] == (len(sample["sequences"]) == 0),
                "scheduler empty/sequence metadata mismatch")
    # This is guaranteed by the pinned scheduler's DP vote. A changed native
    # contract needs its own adapter; never relabel a decode tensor as prompt.
    require(all(s["prompt_run"] == samples[0]["prompt_run"] for s in samples),
            "native DP scheduler step types disagree")
    require(all(s["is_empty"] for s in samples[1:]), "real request on inactive batch rank")
    seqs = samples[0]["sequences"]
    phase, start, end = "idle", None, None
    if seqs:
        require(len(seqs) == 1, "batch schedule requires exactly one active request")
        seq = seqs[0]
        require(seq.get("request_id") == schedule["request_id"]
                and type(seq.get("is_prompt")) is bool
                and seq["is_prompt"] == samples[0]["prompt_run"]
                and type(seq.get("seqlen_q")) is int and seq["seqlen_q"] > 0
                and type(seq.get("seqlen_k")) is int and seq["seqlen_k"] >= seq["seqlen_q"],
                "active scheduler request identity/geometry mismatch")
        if seq["is_prompt"]:
            phase = "prefill"
            require(prefill_completed < len(schedule["steps"]), "extra actual prefill step")
            step = schedule["steps"][prefill_completed]
            start, end = seq["seqlen_k"]-seq["seqlen_q"], seq["seqlen_k"]
            require((start,end) == (step["start"],step["end"]), "actual prefill chunk differs from batch schedule")
        else:
            phase = "decode"
            require(prefill_completed == len(schedule["steps"]), "actual request decode before final batch flush")
    return dict(ordinal=ordinal, world_phase=phase, prefill_completed_before=prefill_completed,
                start=start, end=end, scheduler_prompt_run=samples[0]["prompt_run"],
                rank0_request_id=seqs[0]["request_id"] if seqs else None,
                scheduler_observations_sha256=digest(samples))


class WorkerBatchSchedule:
    """Install on all ranks; inactive DP ranks execute original placeholder first3.

    Only rank0 owns cache/query state. Other ranks drain their original Engram
    fetch each native step, and run zero-local-row deep MoE exactly on flush.
    """
    def __init__(self, model, *, rank, torch_module, empty_ep_capability, controller=None,
                 engine=None, phase_gather=None):
        require(type(rank) is int and 0 <= rank < 8, "worker rank must be 0..7")
        require((rank == 0) == (controller is not None), "only rank0 owns active S1 controller")
        from gpu_pilot.index_hit20_empty_ep_qualification import EmptyEPCapability
        require(isinstance(empty_ep_capability, EmptyEPCapability),
                "all workers require actual empty EP capability")
        self.model, self.rank, self.t, self.capability = model, rank, torch_module, empty_ep_capability
        self.controller, self.schedule, self.trace, self.closed = controller, None, [], False
        require(engine is not None and callable(phase_gather), "actual engine and CPU world scheduler gather required")
        require("_execute_model_async_step" not in engine.__dict__, "foreign engine execution hook")
        self.engine, self.phase_gather = engine, phase_gather
        self.world_trace, self._world_frame = [], None
        self.engine_original = engine._execute_model_async_step
        self.engine_hook = None
        self._phase_done = True
        self.original, self.hook = None, None
        if rank == 0:
            require(controller.native_model is model and controller.state != "NEW", "active controller not installed")
        else:
            require("forward" not in model.__dict__, "foreign inactive-worker model hook")
            self.original = model.forward
            self.symbols = self.original.__func__.__globals__
            owner = self
            def forward(module, kv_caches, input_metadata, hidden_states=None, dflash=None):
                return owner.forward(kv_caches, input_metadata, hidden_states, dflash)
            self.hook = MethodType(forward, model)
            model.forward = self.hook
        owner = self
        def execute(module, seq_metadata_list, scheduler_outputs):
            return owner.execute_step(seq_metadata_list, scheduler_outputs)
        self.engine_hook = MethodType(execute, engine)
        engine._execute_model_async_step = self.engine_hook

    def configure(self, schedule):
        require(not self.closed and self._phase_done, "worker previous phase not closed")
        self.schedule, self.trace, self._phase_done = schedule, [], False
        self.world_trace, self._world_frame = [], None
        if schedule is None:
            return dict(schema="s1-system-worker-disabled-v1", rank=self.rank, enabled=False)
        _validate_schedule(schedule)
        if self.rank == 0:
            require(self.controller.batch_schedule == schedule and self.controller.state == "RUNNING",
                    "active worker/controller schedule differs")
        return dict(schema="s1-system-batch-rank-installation-v1", rank=self.rank,
            schedule_sha256=schedule["sha256"], worker_loop_installed=True,
            native_first3_per_chunk=True, deep_only_on_flush=True, empty_deep_ep_on_flush=True,
            world_scheduler_phase_bound=True)

    def execute_step(self, seqs, outputs):
        if self.schedule is None:
            return self.engine_original(seqs, outputs)
        require(not self._phase_done and self._world_frame is None, "worker scheduler phase inactive or reentrant")
        trace = self.controller._batch_step_trace if self.rank == 0 else self.trace
        before = len(trace)
        sample = dict(rank=self.rank, schedule_sha256=self.schedule["sha256"],
            ordinal=len(self.world_trace), prefill_completed=before,
            prompt_run=outputs.prompt_run, is_empty=outputs.is_empty(),
            sequences=[dict(request_id=s.request_id, is_prompt=s.is_prompt,
                            seqlen_q=s.seqlen_q, seqlen_k=s.seqlen_k) for s in seqs])
        observations = self.phase_gather(sample)
        event = classify_world_step(observations, self.schedule, len(self.world_trace), before)
        if self.rank == 0:
            event["scheduler_observations"] = observations
        frame = dict(event, local_prompt_run=sample["prompt_run"],
                     local_is_empty=sample["is_empty"], model_consumed=False)
        self._world_frame = frame
        try:
            result = self.engine_original(seqs, outputs)
            expected = before + (event["world_phase"] == "prefill")
            require(len(trace) == expected, "actual model did not follow world scheduler phase")
            if self.rank:
                require(frame["model_consumed"], "inactive model bypassed world scheduler hook")
            self.world_trace.append(dict(event, local_prompt_run=sample["prompt_run"],
                local_is_empty=sample["is_empty"], model_execution_completed=True))
            return result
        finally:
            self._world_frame = None

    def forward(self, caches, meta, hidden, dflash):
        if self.schedule is None:
            return self.original(caches, meta, hidden, dflash)
        frame = self._world_frame
        require(frame is not None and not frame["model_consumed"], "model forward lacks unique world scheduler authorization")
        require(meta.is_prompt is frame["local_prompt_run"], "native placeholder step type differs from scheduler")
        frame["model_consumed"] = True
        if frame["world_phase"] != "prefill":
            return self.original(caches, meta, hidden, dflash)
        require(meta.is_prompt is True and frame["local_is_empty"] is True,
                "prefill companion must have native prompt placeholder metadata")
        require(not self._phase_done and len(self.trace) < len(self.schedule["steps"]), "extra worker prompt step")
        require(meta.num_reqs == 1 and meta.total_q == 1 and dflash is None
                and getattr(meta, "canonical_request_binding", None) is None,
                "inactive batch worker requires native one-row placeholder only")
        t, g, model = self.t, self.symbols, self.model
        for name in ("input_ids", "positions", "state_block_idx"):
            t._assert_async((getattr(meta, name) == 0).all(), "inactive worker received a real request: "+name)
        require(model.engram_runtime is not None and meta.ngram_token_ids is not None,
                "worker native Engram history missing")
        step = self.schedule["steps"][len(self.trace)]
        # The real DeepseekVLForCausalLM prefill wrapper has already called
        # get_input_embeddings, including its image-mask handling, before this
        # inner-model hook. Preserve that tensor exactly as DeepseekModel does.
        # The direct text-model entry remains valid and embeds only on None.
        hidden = model.embed_tokens(meta.input_ids) if hidden is None else hidden
        require(getattr(hidden, "ndim", None) == 2
                and tuple(hidden.shape) == (1, model.layers[0].mlp.hidden_size)
                and hidden.dtype == t.bfloat16 and hidden.device == meta.input_ids.device
                and hidden.is_contiguous(),
                "inactive native input embedding must be one contiguous BF16 row on the metadata device")
        residual, cross = None, g["CrossLayerRuntime"]()
        hashes = g["tile_kernels"].engram.engram_hash(meta.ngram_token_ids, model.ngram_multipliers,
            model.ngram_vocab_sizes, model.ngram_offsets, getattr(meta, "images_seq_mask", None))
        hooks = model.engram_runtime.fetch(hashes)
        layers_run = []
        for i in range(3):
            hidden, residual = model.layers[i](g["Movable"](hidden), caches[i], meta, residual,
                embed_hook=hooks.get(i), cross_layer_runtime=cross, dflash_input=None)
            layers_run.append(i)
        require(callable(hooks.get(14)), "worker Engram14 fetch missing")
        hooks[14]()
        if step["flush"]:
            rows = t.empty((0,), device=hidden.device, dtype=t.long)
            cu = inactive_empty_query_cu(meta,t)
            compact = compact_metadata(meta, rows, cu)
            for i in range(3, 40):
                layer = model.layers[i]
                self.capability.permits(layer, meta, NS(full_rows=1, selected_count=0))
                bf16 = hidden.new_empty((0, layer.mlp.hidden_size), dtype=t.bfloat16)
                sf = t.zeros((layer.mlp.hidden_size//128, 4), device=hidden.device, dtype=t.int32)
                packed = t.empty_like(bf16, dtype=t.float8_e4m3fn), sf[:, :0].transpose(0, 1)
                result = layer.mlp(packed, bf16, compact)
                require(result.shape == bf16.shape, "worker native empty MLP result geometry")
                layers_run.append(i)
        self.trace.append(dict(start=step["start"], end=step["end"], flush=step["flush"],
            layers=layers_run, original_placeholder_rows=1,
            deep_native_ep_calls=37 if step["flush"] else 0,
            empty_deep_native_ep_calls=37 if step["flush"] else 0,
            original_engram14_fetch_consumed=True))
        return model.allocate_output(1)

    def finish_phase(self):
        require(not self.closed and not self._phase_done, "worker phase not active")
        self._phase_done = True
        if self.schedule is None:
            return dict(schema="s1-system-worker-execution-v1", rank=self.rank, enabled=False, steps=[])
        trace = self.controller._batch_step_trace if self.rank == 0 else self.trace
        require(len(trace) == len(self.schedule["steps"]), "worker batch step coverage incomplete")
        for event, step in zip(trace, self.schedule["steps"]):
            require(all(event[k] == step[k] for k in ("start", "end", "flush", "layers")),
                    "worker actual EP loop differs from admitted schedule")
        return dict(schema="s1-system-worker-execution-v1", rank=self.rank, enabled=True,
            schedule_sha256=self.schedule["sha256"], steps=list(trace),
            actual_deep_ep_calls=sum(x["deep_native_ep_calls"] for x in trace),
            actual_empty_deep_ep_calls=sum(x.get("empty_deep_native_ep_calls",0) for x in trace),
            world_scheduler_phase_bound=True, world_step_trace=list(self.world_trace),
            gpu_numerical_parity_claim=False)

    def close(self):
        if self.closed: return
        require(self.engine.__dict__.get("_execute_model_async_step") is self.engine_hook,
                "worker engine hook ownership changed")
        if self.hook is not None:
            require(self.model.__dict__.get("forward") is self.hook, "worker model hook ownership changed")
            del self.model.forward
        del self.engine._execute_model_async_step
        self.closed = True


def install_worker_batch_schedule(native_model, *, rank, torch_module, empty_ep_capability, controller=None,
                                  engine=None, phase_gather=None):
    return WorkerBatchSchedule(native_model, rank=rank, torch_module=torch_module,
        empty_ep_capability=empty_ep_capability, controller=controller, engine=engine, phase_gather=phase_gather)
