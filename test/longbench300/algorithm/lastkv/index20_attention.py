"""Arbitrary-query SWA geometry with unchanged native attention arithmetic."""
from __future__ import annotations
from index20_plan import require, local_memory_plan, mandatory_selection


def index20_attention_class(base_class):
    class Index20Attention(base_class):
        def begin_phase(self):
            result=super().begin_phase()
            self._index20_bound_selection=None
            return result

        def _assert_selection_binding(self):
            signature=(self.cache.index20_selected is not None,tuple(self.cache.selected))
            if self._index20_bound_selection is None:self._index20_bound_selection=signature
            require(self._index20_bound_selection==signature,
                    "selection changed after deep attention began; start a fresh phase")

        def _candidate_event(self,ctx,previous_positions):
            require(self.cache.index20_selected is not None,"candidate selection absent")
            return local_memory_plan(self.cache.docs,self.cache.total,self.cache.selected,
                ctx.start,ctx.end,ctx.positions_cpu,previous_positions,
                offline_positions=self.cache.index20_offline_positions)

        def _mapping(self,prepared,meta,ctx,previous_positions):
            if self.cache.index20_selected is not None:
                key=ctx.start,ctx.end,tuple(ctx.positions_cpu),tuple(previous_positions)
                if key not in self.geometry:
                    self.geometry[key]=self._candidate_event(ctx,previous_positions)
            # The original method consumes the pre-populated CPU geometry and
            # constructs/validates exactly its existing GPU indices and buffers.
            return super()._mapping(prepared,meta,ctx,previous_positions)

        def _memory(self,mod,prepared,meta,ctx):
            self._assert_selection_binding()
            if self.cache.index20_selected is not None and not self.enabled:
                previous_positions,_=self.previous.get(mod.layer_idx,([],prepared[:0]))
                key=ctx.start,ctx.end,tuple(previous_positions)
                if key not in self.geometry:
                    self.geometry[key]=self._candidate_event(ctx,previous_positions)
            return super()._memory(mod,prepared,meta,ctx)

        def report(self):
            result=super().report()
            active=self.cache.index20_selected is not None
            result["full_swa_bank_used"]=active and bool(self.cache.index20_offline_positions)
            result["only_tail_and_nondoc_queries"]=(
                not active or self.cache.selected==mandatory_selection(self.cache.docs,self.cache.total))
            result["index20_dependency_policy"]=dict(
                schema="index20-native-swa-dependencies-v1",
                active=active,native_window=128,
                original_absolute_positions=True,extra_cross_document_links=False,
                cached_sources="full_normalized_document_swa_rows_for_unselected_dependencies",
                selected_sources="current_or_previous_online_swa_rows",
                projection_joint_attention_global_index_and_ring_unchanged=True)
            return result
    Index20Attention.__name__="Index20"+base_class.__name__
    return Index20Attention
