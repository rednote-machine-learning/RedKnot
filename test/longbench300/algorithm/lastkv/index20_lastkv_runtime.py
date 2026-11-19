"""Independent all-input final-layer SWA reprojection experiment.

The Index20 hidden policy is unchanged. Unselected positions restore the exact
offline last-layer attention inputs, including packed FP8/SF; selected positions
use live online inputs. This does not recreate missing online-context hidden.
Global layer39 has no independent projection and continues to consume source20.
"""
from __future__ import annotations

from types import MethodType
from index20_selector import require,seal


def take_owned(value):
    return value.move() if callable(getattr(value,"move",None)) else value


def index20_lastkv_controller_class(natural_class):
    from index20_runtime import index20_controller_class
    parent=index20_controller_class(natural_class)

    class Index20LastKVController(parent):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._configured_lastkv=False
            self.lastkv_enabled=False

        def install(self):
            super().install()
            try:
                from index20_lastkv_cache import index20_lastkv_cache_class
                from index20_lastkv_attention import index20_lastkv_attention_class
                require(not self.cache.bank and self.case_binding is None,
                        "last-layer input bank must install before offline capture")
                self.cache=index20_lastkv_cache_class(type(self.cache))(
                    layers=self.layers,scope=self.scope,torch_module=self.t,allow_sync=self.allow_sync)
                self.attention=index20_lastkv_attention_class(type(self.attention))(
                    self.cache,torch_module=self.t,allow_sync=self.allow_sync)
                return self
            except BaseException:
                self.close();raise

        def _install_attention(self,mod):
            super()._install_attention(mod)
            if mod.layer_idx!=39:return
            owner=self; original=mod.forward
            owners=[i for i,(obj,name,hook)in enumerate(self.handles)
                    if obj is mod and name=="forward" and hook is original]
            require(len(owners)==1 and mod.__dict__.get("forward") is original,
                    "last-layer capture hook requires the unique existing owned attention hook")
            def capture(attn,hidden_states,kv_cache,input_metadata,hidden_states_bf16,
                        cross_layer_runtime,**kwargs):
                marker=getattr(input_metadata,"canonical_request_binding",None)
                if owner.mode=="capture_independent" and input_metadata.is_prompt and marker is not None:
                    # Norm + previous lazy mHC post have already executed in
                    # the original decoder. Take each move-only input once,
                    # save exact representations, then restore ownership for
                    # the unchanged native attention implementation.
                    packed,bf16=take_owned(hidden_states),take_owned(hidden_states_bf16)
                    owner.cache.capture_last_attention_inputs(attn,packed,bf16,input_metadata)
                    movable=owner.native_symbols["Movable"]
                    hidden_states,hidden_states_bf16=movable(packed),movable(bf16)
                return original(hidden_states,kv_cache,input_metadata,hidden_states_bf16,
                                cross_layer_runtime,**kwargs)
            hooked=MethodType(capture,mod)
            # Replace the existing handle, not the class/global or a second
            # conflicting hook. Original controller.close removes it safely.
            self.handles[owners[0]]=(mod,"forward",hooked)
            mod.forward=hooked

        def configure_lastkv(self,enabled):
            require(type(enabled) is bool,"explicit all-input lastKV flag required")
            require(self.state in ("NEW","IDLE","COMPLETE"),"cannot change active lastKV policy")
            self._configured_lastkv=enabled
            return self

        def reset(self,*args,**kwargs):
            mode=args[1] if len(args)>1 else kwargs.get("mode")
            require(not self._configured_lastkv or (mode=="sensitive" and self._configured_index20),
                    "all-input lastKV requires an enabled Index20 hidden policy")
            self.lastkv_enabled=self._configured_lastkv
            self.attention.configure_lastkv(self.lastkv_enabled,diagnostic=
                bool(self.lastkv_enabled and self._index20_diagnostic))
            return super().reset(*args,**kwargs)

        def finish(self):
            report=super().finish()
            evidence=report["attention_report"]["index20_lastkv"]
            bank=report["cache_report"]["lastkv_input_bank"]
            require(evidence["enabled"] is self.lastkv_enabled,"lastKV phase flag drift")
            info=dict(schema="index20-all-input-layer39-swa-v1",enabled=self.lastkv_enabled,
                variant="all_input_layer39_swa_reprojection",input_tokens=len(self.ids),
                hidden_policy="unchanged_Index20_body_budget_and_full_noncached_tokens",
                unselected_hidden_source="offline_independent_document_layer39_attention_input",
                missing_online_context_hidden_recomputed=False,
                all_input_global_layer39_recomputed=False,
                global_layer39="reuse_source20_no_independent_projection",
                physical_swa_cache_policy="native_last128_ring_not_all_input_persistent_cache",
                extra_offline_input_bank=bank,attention=evidence,
                selected_hidden_ffn_attention_pruning=False,
                bitwise_kv_equivalence_claim=False,
                possible_numeric_difference_sources=["full_vs_compact_native_GEMM_row_geometry",
                    "native_fused_norm_rope_cast_vs_offline_normalized_BF16_intermediate_replay"],
                unselected_context_dependence_still_from_independent_offline_document=True)
            if self.lastkv_enabled:
                selected=report["index20"]["selection"]["selected_positions"]
                from index20_lastkv_audit import audit_projection_evidence
                proof=audit_projection_evidence(evidence,len(self.ids),selected)
                require(proof["status"]=="passed", "lastKV full-input projection evidence incomplete: "+str(proof))
                info.update(coverage=proof,all_input_rows_reprojected=True,
                    live_selected_rows=len(selected),restored_offline_rows=len(self.ids)-len(selected))
            else:
                info.update(all_input_rows_reprojected=False,live_selected_rows=0,restored_offline_rows=0)
            report["index20_lastkv"]=seal(info)
            self.last_report=report
            return report

    return Index20LastKVController
