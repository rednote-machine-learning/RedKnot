"""Private LastKV adapter: frozen Index20/S1.2/native sources remain unchanged."""
from types import FunctionType
import index20_bench_runtime as original_adapter
from index20_bench_runtime_v2 import bench_runtime_class as strategy_runtime_class
from rag_dspark_runtime import require
from rag_natural_documents import natural_controller_class

def bench_runtime_class(parent):
    base=strategy_runtime_class(parent)
    original=original_adapter.private_initializer(parent)
    def factory():
        from index20_lastkv_runtime import index20_lastkv_controller_class
        return index20_lastkv_controller_class(natural_controller_class())
    initializer=FunctionType(original.__code__,dict(original.__globals__,candidate_controller_class=factory),
        original.__name__,original.__defaults__,original.__closure__)
    class LastKVBenchRuntime(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.lastkv_requested=False;self.force_diagnostic=False
        def _initialize_s2(self):
            initializer(self)
            if self.controller is not None:
                from rag_capture_policy import capture_policy_cache_class
                c=self.controller
                require(not c.cache.bank and c.state in ('IDLE','COMPLETE'),'Install memory-only policy before offline capture')
                cls=capture_policy_cache_class(type(c.cache),persist_offline_tensors=False)
                c.cache=cls(layers=c.layers,scope=c.scope,torch_module=c.t,allow_sync=c.allow_sync)
                c.attention=type(c.attention)(c.cache,torch_module=c.t,allow_sync=c.allow_sync)
                c.configure_index20(enabled=False,fraction=self.index20_fraction)
                c.configure_lastkv(False)
            self.driver.barrier()
        def run_method(self,row,name,method,*,force_diagnostic=False,**kwargs):
            self.force_diagnostic=bool(force_diagnostic)
            self.lastkv_requested=method.startswith('Index20LastKV.')
            actual='Index20.'+method.split('.',1)[1]if self.lastkv_requested else method
            try:return super().run_method(row,name,actual,**kwargs)
            finally:self.lastkv_requested=False;self.force_diagnostic=False
        def select_arm(self,enabled,*,fraction=None,diagnostic_fingerprints=False):
            self.index20_active=bool(enabled)
            value=self.index20_fraction if fraction is None else float(fraction)
            if self.controller is None or(not enabled and self.controller.state=='FAILED'):return
            if not enabled:self.controller.configure_lastkv(False)
            self.controller.configure_index20(enabled=bool(enabled),fraction=value,strategy=self.index20_strategy,
                diagnostic_fingerprints=bool(enabled and(diagnostic_fingerprints or self.force_diagnostic)))
            if enabled:self.controller.configure_lastkv(self.lastkv_requested)
    return LastKVBenchRuntime
