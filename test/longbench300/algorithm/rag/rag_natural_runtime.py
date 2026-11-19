"""Arbitrary natural RAG document count using unchanged fixed S1.2 selection.

This adapter is separate from the frozen original10-block pilot. Capacity
fallbacks execute every token normally; they are never labeled cache hits.
"""
from rag_dspark_runtime import require,runtime_class
from rag_natural_documents import plan_case,install_natural_controller_initializer


def natural_runtime_class(parent):
    fixed=runtime_class(parent)
    initializer=install_natural_controller_initializer(fixed,parent)
    class NaturalRagRuntime(fixed):
        _initialize_s2=initializer

        def capture_case(self,row,prefix):
            plan=plan_case(row['input_ids'],row['documents']);self.rag_case_plan=plan
            if self.controller is not None:
                self.controller.clear_cache(reset_sparse_qualification=True)
                self.controller.set_cache_output_dir(self.out/(prefix+'.offline-kv'))
            if plan['effective_route']=='native_full':
                self.driver.barrier()
                return 0.0,[]
            # Capture each exact content version once; repeated occurrences
            # retain their different absolute positions in the online prompt.
            import time
            self.driver.barrier();began=time.perf_counter();phases=[]
            for i,doc in enumerate(plan['unique_capture_documents']):
                offline=dict(input_ids=row['input_ids'][doc['input_start']:doc['input_end']])
                phases.append(super().run_phase(offline,prefix+'.offline_doc'+str(i),'capture_independent',tokens=1,qualification=True))
            if self.controller is not None:
                with self.allow_sync():self.controller.prepare_case(input_ids=row['input_ids'],documents=plan['effective_documents'],
                    index_device=self.layers[0].self_attn.q_a_kv_swa_proj.weight.device)
            self.driver.barrier()
            return max(self.driver.gather(time.perf_counter()-began)),phases

        def run_phase(self,row,name,mode,*,tokens=32,qualification=False,verify_sparse=False):
            plan=getattr(self,'rag_case_plan',None)
            if plan is not None:
                require(row['input_ids_sha256']==plan['input_ids_sha256'],'Natural RAG case binding changed')
                if plan['effective_route']=='native_full':
                    require(not qualification and not verify_sparse,'Native fallback has no sparse qualification claim')
                    mode=None
                row=dict(row,documents=plan['effective_documents'])
            return super().run_phase(row,name,mode,tokens=tokens,qualification=qualification,verify_sparse=verify_sparse)
    return NaturalRagRuntime
