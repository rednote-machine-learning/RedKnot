"""Strategy-tagged adapter; each request runs exactly one selected reducer."""
from index20_bench_runtime import bench_runtime_class as mean_runtime_class
from rag_dspark_runtime import require

STRATEGIES=('mean','max','softmax_mean')

def bench_runtime_class(parent):
    base=mean_runtime_class(parent)
    class StrategyRuntime(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.index20_strategy=self.config['index20_strategy']
            require(self.index20_strategy in STRATEGIES,'Unsupported prespecified Index20 strategy')
        def run_method(self,row,name,method,**kwargs):
            if method.startswith('Index20.'):
                strategy=method.split('.',1)[1]
                require(strategy in self.config['index20_strategies'],'Strategy not admitted in this frozen run')
                self.index20_strategy=strategy
                return super().run_method(row,name,'Index20',**kwargs)
            return super().run_method(row,name,method,**kwargs)
        def select_arm(self,enabled,*,fraction=None,diagnostic_fingerprints=False):
            self.index20_active=bool(enabled)
            value=self.index20_fraction if fraction is None else float(fraction)
            require(0<=value<=1,'Index20 selection fraction outside0..1')
            if self.controller is not None and not (not enabled and self.controller.state=='FAILED'):
                self.controller.configure_index20(enabled=bool(enabled),fraction=value,
                    strategy=self.index20_strategy,diagnostic_fingerprints=diagnostic_fingerprints)
    return StrategyRuntime
