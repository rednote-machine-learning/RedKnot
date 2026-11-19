"""Optional capture serialization policy; immutable GPU bank math is inherited."""
from rag_dspark_runtime import require


def capture_policy_cache_class(base_class,*,persist_offline_tensors):
    require(type(persist_offline_tensors) is bool,'Explicit offline tensor persistence policy required')
    class CapturePolicyCache(base_class):
        def begin_capture(self,key,length,output_dir=None):
            return super().begin_capture(key,length,output_dir=output_dir if persist_offline_tensors else None)

        def report(self):
            report=super().report()
            report['offline_bank_inventory']=[dict(document_sha256=key[0],kind=key[1],layer=key[2],
                source_document_tokens=entry['length'],local_start=entry['local_start'],scope_sha256=entry['scope'],
                complete=entry['complete'],recorded_version=entry.get('version'),actual_tensor_version=entry['tensor']._version,
                shape=list(entry['tensor'].shape),dtype=str(entry['tensor'].dtype),device=str(entry['tensor'].device),
                bytes=entry['tensor'].numel()*entry['tensor'].element_size()) for key,entry in sorted(self.bank.items())]
            report['capture_persistence_policy']=dict(schema='rag-offline-capture-persistence-v1',
                persist_offline_tensors=persist_offline_tensors,
                native_bank_allocations_and_version_scope_checks_unchanged=True,
                full_payload_counts_and_actual_bytes_recorded=True,
                disk_kv_load_measured=False,disk_kv_residency_measured=False,
                omitted_work='none' if persist_offline_tensors else 'gpu_to_cpu_file_serialization_only')
            return report
    CapturePolicyCache.__name__='PersistentCapture'+base_class.__name__ if persist_offline_tensors else 'MemoryCapture'+base_class.__name__
    return CapturePolicyCache
