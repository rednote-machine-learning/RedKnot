"""Regenerate layer39 SWA for every input row from live/restored attn inputs.

Selected queries retain their original compact Q path. The additional native
fused projection also computes unused q_a rows; this is reported, not hidden.
Layer39 consumes source20 global KV and does not own a global projection.
"""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from index20_plan import digest, require


def full_replay_memory_plan(start, end, positions, previous_positions):
    """Independent source ownership for selected queries and full regenerated KV."""
    positions,previous_positions = list(positions),list(previous_positions)
    require(positions == sorted(set(positions)) and all(start <= p < end for p in positions),
            "last-KV query positions outside execution group")
    require(previous_positions == list(range(max(0,start-127),start)),
            "last-KV previous regenerated window incomplete")
    needed = sorted({k for q in positions for k in range(max(0,q-127),q+1)})
    previous = {p:i for i,p in enumerate(previous_positions)}
    kinds,rows = [],[]
    for p in needed:
        if start <= p < end:
            kinds.append("current_regenerated");rows.append(p-start)
        else:
            require(p in previous,"last-KV missing preceding regenerated key")
            kinds.append("previous_regenerated");rows.append(previous[p])
    result = dict(chunk_start=start,chunk_end=end,query_positions=positions,
        memory_positions=needed,source_kinds=kinds,source_rows=rows,
        native_window_size=128,extra_nonlocal_swa_links=False,
        local_lengths=[min(128,p+1) for p in positions],
        full_layer39_swa_regenerated=True,all_dependencies_covered=True)
    return dict(result,sha256=digest(result))


def index20_lastkv_attention_class(base_class):
    class LastKVAttention(base_class):
        def configure_lastkv(self, enabled, diagnostic=False):
            require(type(enabled) is bool and type(diagnostic) is bool,"last-KV configure booleans")
            self._lastkv_configured,self._lastkv_diagnostic = enabled,diagnostic

        def begin_phase(self):
            result = super().begin_phase()
            self._lastkv_enabled = getattr(self,"_lastkv_configured",False)
            self._lastkv_events,self._lastkv_memory_events = [],[]
            self._lastkv_current = None
            self._lastkv_previous = None
            return result

        def _difference(self, actual, expected):
            require(actual.shape == expected.shape and actual.dtype == expected.dtype,
                    "last-KV diagnostic representation mismatch")
            with self.allow_sync() if self.allow_sync is not None else nullcontext():
                return dict(rows=actual.shape[0],elements=actual.numel(),
                    exact=bool(self.t.equal(actual,expected)),
                    max_abs_error=float((actual.float()-expected.float()).abs().max().item()) if actual.numel() else 0.)

        def _projection(self, mod, packed, bf16, meta, ctx):
            # Preserve selected Q_b/index/global/current projection and their
            # original row geometry. Other layers/disabled arms delegate fully.
            current,checks,checked = super()._projection(mod,packed,bf16,meta,ctx)
            if not self._lastkv_enabled or mod.layer_idx != 39:
                return current,checks,checked
            require(self._lastkv_current is None and mod.use_swa and not mod.yoco_swa_share_hidden
                    and not mod.is_kv_source and not mod.is_index_k_source and not mod.has_local_index
                    and mod.kv_compress_ratio == 1 and current["global_raw"] is None,
                    "last-KV layer39 global ownership or active replay changed")
            require(self.cache.index20_selected is not None,"last-KV requires a bound Index20 selection")
            selected = list(ctx.positions_cpu)
            require(selected == [p for p in self.cache.selected if ctx.start <= p < ctx.end],
                    "last-KV online input positions differ from frozen selection")
            lookup = {p:i for i,p in enumerate(selected)}
            t = self.t
            all_values = []
            diagnostic = getattr(self,"_lastkv_diagnostic",False)
            for start in range(ctx.start,ctx.end,8192):
                end = min(ctx.end,start+8192)
                full_packed,full_bf16,source = self.cache.materialize_last_attention_inputs(
                    start,end,selected,packed,bf16)
                # Same fused native weights/kernel path as original. q_a is an
                # unused byproduct for these extra rows, not a free operation.
                discarded_q_a,raw = mod._project_q_and_swa(full_packed,None)
                require(tuple(raw.shape) == (end-start,512),"last-KV native full raw shape")
                del discarded_q_a,full_packed,full_bf16
                positions = ctx.full_meta.positions[start-ctx.start:end-ctx.start]
                wanted = t.arange(start,end,device=positions.device,dtype=positions.dtype)
                t._assert_async((positions == wanted).all(),"last-KV full absolute positions changed")
                replay_meta = SimpleNamespace(is_prompt=True,positions=positions)
                prepared,waiter = mod._prepare_swa_kv(raw,None,replay_meta,False)
                require(waiter is None and tuple(prepared.shape) == (end-start,1,512),
                        "last-KV original native CP1 postprocess changed")
                t._assert_async(t.isfinite(prepared).all(),"nonfinite regenerated final SWA")
                all_values.append(prepared)
                live = [p for p in selected if start <= p < end]
                event = dict(source,start=start,end=end,projected_rows=end-start,
                    selected_positions=live,live_selected_rows=len(live),restored_offline_rows=end-start-len(live),
                    native_swa_projection=True,global_projection=False,positions_preserved=True,
                    fused_q_a_projection_rows=end-start,discarded_auxiliary_q_a_rows=end-start,
                    original_selected_query_projection_preserved=True,
                    all_row_ffn_or_residual_executed=False,
                    native_fused_norm_rope_cast=True,diagnostic_comparisons_performed=diagnostic)
                if diagnostic:
                    offline = [p for p in range(start,end) if p not in lookup]
                    if offline:
                        rows = self._indices([p-start for p in offline],raw.device)
                        saved = self.cache.original_last_swa_rows(offline)
                        canonical = self.cache.ops.normalize_swa(mod,raw.index_select(0,rows).contiguous())
                        pos = self._indices(offline,raw.device,positions.dtype)
                        saved_prepared = self.cache.ops.prepare_swa(mod,saved,pos)
                        event["unselected_canonical_vs_original_offline"] = self._difference(canonical,saved)
                        event["unselected_prepared_at_current_positions_vs_original"] = self._difference(
                            prepared.index_select(0,rows),saved_prepared)
                    if live:
                        full_rows = self._indices([p-start for p in live],raw.device)
                        live_rows = self._indices([lookup[p] for p in live],raw.device)
                        event["selected_raw_vs_original_compact_projection"] = self._difference(
                            raw.index_select(0,full_rows),current["swa"].index_select(0,live_rows))
                self._lastkv_events.append(event)
            self._lastkv_current = (ctx.start,ctx.end,t.cat(all_values,dim=0))
            return current,checks,checked

        def _memory(self, mod, prepared, meta, ctx):
            if not self._lastkv_enabled or mod.layer_idx != 39:
                return super()._memory(mod,prepared,meta,ctx)
            self._assert_selection_binding()
            require(self._lastkv_current is not None and self._lastkv_current[:2] == (ctx.start,ctx.end),
                    "last-KV full regeneration not bound to memory group")
            current = self._lastkv_current[2]
            previous_positions,previous = self._lastkv_previous or ([],current[:0])
            event = full_replay_memory_plan(ctx.start,ctx.end,ctx.positions_cpu,previous_positions)
            t = self.t
            memory = current.new_empty((len(event["memory_positions"]),1,512))
            for kind,source in (("current_regenerated",current),("previous_regenerated",previous)):
                dst = [i for i,k in enumerate(event["source_kinds"]) if k == kind]
                if dst:
                    src = [event["source_rows"][i] for i in dst]
                    memory.index_copy_(0,self._indices(dst,memory.device),source.index_select(0,self._indices(src,source.device)))
            absolute = self._indices(event["memory_positions"],memory.device,meta.positions.dtype)
            queries = meta.positions
            expected_queries = self._indices(ctx.positions_cpu,memory.device,meta.positions.dtype)
            t._assert_async((queries == expected_queries).all(),"last-KV selected query positions changed")
            low = t.clamp(queries-127,min=0)
            wanted = low[:,None]+t.arange(128,device=memory.device,dtype=queries.dtype)[None,:]
            valid = wanted <= queries[:,None]
            if absolute.numel():
                slots = t.searchsorted(absolute,wanted.contiguous())
                safe = slots.clamp(max=absolute.numel()-1)
                t._assert_async(((~valid)|((slots<absolute.numel())&(absolute[safe]==wanted))).all(),
                                "last-KV regenerated native window missing")
                indices = t.where(valid,slots,-1).to(t.int32).contiguous()
            else:
                indices = t.empty((0,128),dtype=t.int32,device=memory.device)
            # Full regenerated tail survives, including unselected preceding rows.
            combined = t.cat((previous,current),dim=0)
            kept_positions = list(range(max(0,ctx.end-127),ctx.end))
            self._lastkv_previous = (kept_positions,combined[-len(kept_positions):].clone())
            self._lastkv_current = None
            self._lastkv_memory_events.append(dict(start=ctx.start,end=ctx.end,
                query_rows=len(ctx.positions_cpu),regenerated_key_rows=ctx.end-ctx.start,
                local_memory_rows=len(event["memory_positions"]),
                previous_regenerated_rows=len(previous_positions),
                original_native_mask_oracle_runs_in_parent=True,final_ring_store_runs_in_parent=ctx.end==self.cache.total))
            return memory,absolute,indices,event

        def report(self):
            result = super().report()
            enabled = getattr(self,"_lastkv_enabled",False)
            require(self._lastkv_current is None,"last-KV regeneration not consumed")
            if enabled:
                result["attention_events"] = [dict(event,
                    no_unselected_hidden_fallback=False,
                    unselected_offline_attn_input_restored_for_kv_only=True,
                    unselected_full_decoder_fallback=False) if event.get("layer") == 39 else event
                    for event in result["attention_events"]]
                result["index20_dependency_policy"]["cached_swa_source_policy_layers_zero_based"] = list(range(3,39))
                result["index20_dependency_policy"]["layer39_sources"] = "newly_projected_from_live_or_offline_postnorm_attention_inputs"
                result["system_optimization"]["projection_scope_changed_by_lastkv"] = True
                result["system_optimization"]["selected_Q_and_joint_attention_unchanged"] = True
            result["index20_lastkv"] = dict(schema="index20-lastkv-attention-v1",enabled=enabled,
                layer_zero_based=39,events=list(self._lastkv_events),memory_events=list(self._lastkv_memory_events),
                total_projection_rows=sum(e["projected_rows"] for e in self._lastkv_events),
                fused_q_a_projection_rows=sum(e["fused_q_a_projection_rows"] for e in self._lastkv_events),
                auxiliary_q_a_is_discarded=True,projection_subchunk_limit=8192,
                selected_queries_unchanged=True,global_layer39_reprojected=False,
                global_source20_still_used=True,unselected_hidden_refreshed_online=False,
                unselected_attention_input_restored_from_offline=True,
                equivalence_or_accuracy_improvement_claim=False,
                complete_input_projection_coverage_claim=sum(e["projected_rows"] for e in self._lastkv_events)==self.cache.total if enabled else False,
                extra_selected_swa_postprocess_in_inherited_forward=True)
            return result
    LastKVAttention.__name__ = "LastKV"+base_class.__name__
    return LastKVAttention


lastkv_attention_class = index20_lastkv_attention_class
