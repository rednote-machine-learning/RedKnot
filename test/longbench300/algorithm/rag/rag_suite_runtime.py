"""Direct natural-document runtime for the RAG suite (no agent execution)."""
from pathlib import Path
import time

from rag_dspark_runtime import require
from rag_natural_runtime import natural_runtime_class
from global_base_cache import cached_global_base_class
from rag_capture_policy import capture_policy_cache_class


def suite_runtime_class(parent):
    natural=natural_runtime_class(parent)
    class RagSuiteRuntime(natural):
        def __init__(self,config,arm,output_dir,rank):
            super().__init__(config,arm,output_dir,rank)
            self.suite_global_reuse=bool(config['rag_suite_global_base_cache'])
            self.suite_persist_offline_tensors=config['rag_suite_persist_offline_tensors']
            self.suite_layout_policy=config['rag_suite_derived_layout_cache_policy']
            self.suite_progress=None
            self.suite_case_index=None
            self.suite_method=None

        def _initialize_s2(self):
            super()._initialize_s2()
            if self.controller is not None:
                c=self.controller
                require(not c.cache.bank and c.state in ('IDLE','COMPLETE'),'Install cache reuse before case capture')
                from kv_cache_engram.s1_system_cache import S1SystemCache
                from kv_cache_engram.s1_system_attention import S1SystemAttention
                cache_class=capture_policy_cache_class(S1SystemCache,persist_offline_tensors=self.suite_persist_offline_tensors)
                if self.suite_global_reuse:cache_class=cached_global_base_class(cache_class)
                c.cache=cache_class(layers=c.layers,scope=c.scope,torch_module=c.t,allow_sync=c.allow_sync)
                c.attention=S1SystemAttention(c.cache,torch_module=c.t,allow_sync=c.allow_sync)
                if self.suite_global_reuse:c.cache.configure_global_base_reuse(enabled=False,bank_generation=lambda:c.bank_generation)
            self.driver.barrier()

        def _reuse(self,enabled):
            if self.controller is not None and self.suite_global_reuse:
                self.controller.cache.configure_global_base_reuse(enabled=enabled,
                    bank_generation=lambda:self.controller.bank_generation,retain_cached_base=True)

        def _internal(self,request,phase,mode,*,verify_sparse=False,qualification_only=True):
            if self.rank==0 and self.suite_progress is not None:
                self.suite_progress(phase=phase,status='running',index=self.suite_case_index,
                    method=self.suite_method,input_tokens=len(request.prompt_token_ids),
                    qualification_only=qualification_only)
            result=super()._internal(request,phase,mode,verify_sparse=verify_sparse,qualification_only=qualification_only)
            if self.rank==0 and self.suite_progress is not None:
                record=result['record']
                self.suite_progress(phase=phase,status='phase_complete',index=self.suite_case_index,
                    method=self.suite_method,input_tokens=record['input_tokens'],
                    output_tokens=len(record['output_ids']),raw_ttft_s=record['raw_ttft_s'],
                    finish_reason=record['finish_reason'],phase_sha256=result['sha256'],
                    qualification_only=qualification_only)
            return result

        def capture_case(self,row,prefix):
            self.suite_method='offline_build'
            if self.controller is not None and self.suite_global_reuse:self.controller.cache.invalidate_global_base_reuse()
            self._reuse(False)
            return super().capture_case(row,prefix)

        def run_reference(self,row,name):
            self.suite_method='same_policy_dense_reference';self._reuse(False)
            return super().run_phase(row,name,'reference_sparse',qualification=True)

        def run_method(self,row,name,method,*,tokens=32,qualification=False,verify_sparse=False):
            require(method in ('Full','S1.2'),'Unknown RAG suite method')
            # This version's request clock includes the host-side switch too.
            began=time.perf_counter();self.suite_method=method
            plan=self.rag_case_plan
            require(row['input_ids_sha256']==plan['input_ids_sha256'],'Natural RAG case changed')
            if method=='S1.2' and self.suite_global_reuse and plan['effective_route']=='s12' and not qualification:
                require(self.suite_layout_policy=='cold_per_online_request','Formal suite forbids repeated-layout hot timing')
                if self.controller is not None:self.controller.cache.invalidate_global_base_reuse()
            self._reuse(method=='S1.2' and self.suite_global_reuse and plan['effective_route']=='s12')
            mode=None if method=='Full' or plan['effective_route']=='native_full' else 'sensitive'
            require(mode is not None or not verify_sparse,'Full fallback cannot claim sparse verification')
            from service.engine_bridge import TokenRequest
            request=TokenRequest(name,tuple(row['input_ids']),tokens,'full' if mode is None else 's2',began,
                documents=tuple(plan['effective_documents']))
            return self._internal(request,name,mode,verify_sparse=verify_sparse,qualification_only=qualification)
    return RagSuiteRuntime
