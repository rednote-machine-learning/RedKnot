"""Private three-arm adapter; preserved RAG/native modules remain immutable."""
from __future__ import annotations
import ast, inspect, textwrap, time
from types import FunctionType

from rag_dspark_runtime import require, runtime_class
from rag_natural_documents import natural_controller_class
from rag_suite_runtime import suite_runtime_class


def private_initializer(parent):
    """Replace only controller and worker constructor imports in a private copy."""
    original=runtime_class(parent)._initialize_s2
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    changes=[]
    class Rewrite(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            names=[x.name for x in node.names]
            if node.module=='kv_cache_engram.s11_qualification'and names==['S11QualificationController']:
                changes.append('controller')
                return ast.copy_location(ast.parse('S11QualificationController = candidate_controller_class()').body[0],node)
            if node.module=='kv_cache_engram.s1_system_runtime'and names==['install_worker_batch_schedule']:
                changes.append('worker')
                return ast.copy_location(ast.ImportFrom(module='index20_worker',names=node.names,level=0),node)
            return node
    tree=Rewrite().visit(tree)
    require(sorted(changes)==['controller','worker'],'Native startup constructor boundaries changed')
    def candidate_controller_class():
        from index20_runtime import index20_controller_class
        return index20_controller_class(natural_controller_class())
    namespace=dict(original.__globals__,parent=parent,candidate_controller_class=candidate_controller_class)
    exec(compile(ast.fix_missing_locations(tree),'<index20-private-native-initializer>','exec'),namespace)
    return namespace[original.__name__]


def bench_runtime_class(parent):
    base=suite_runtime_class(parent)
    initializer=private_initializer(parent)
    from index20_worker import validate_candidate
    original_finish=parent.NativeRuntime._finish_phase
    finish=FunctionType(original_finish.__code__,dict(original_finish.__globals__,validate_candidate=validate_candidate),
        original_finish.__name__,original_finish.__defaults__,original_finish.__closure__)
    finish.__kwdefaults__=original_finish.__kwdefaults__
    class Index20BenchRuntime(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.index20_active=False
            self.index20_fraction=float(self.config['index20_fraction'])

        def _initialize_s2(self):
            initializer(self)
            if self.controller is not None:
                from rag_capture_policy import capture_policy_cache_class
                c=self.controller
                require(not c.cache.bank and c.state in ('IDLE','COMPLETE'),'Install full SWA before capture only')
                require(self.suite_global_reuse is False,'Index20 does not enable global-layout reuse')
                cls=capture_policy_cache_class(type(c.cache),persist_offline_tensors=self.suite_persist_offline_tensors)
                c.cache=cls(layers=c.layers,scope=c.scope,torch_module=c.t,allow_sync=c.allow_sync)
                c.attention=type(c.attention)(c.cache,torch_module=c.t,allow_sync=c.allow_sync)
                c.configure_index20(enabled=False,fraction=self.index20_fraction)
            self.driver.barrier()

        _finish_phase=finish

        def select_arm(self,enabled,*,fraction=None,diagnostic_fingerprints=False):
            self.index20_active=bool(enabled)
            value=self.index20_fraction if fraction is None else float(fraction)
            require(0<=value<=1,'Index20 selection fraction outside0..1')
            if self.controller is not None:
                self.controller.configure_index20(enabled=bool(enabled),fraction=value,diagnostic_fingerprints=diagnostic_fingerprints)

        def capture_case(self,row,prefix):
            self.select_arm(False)
            return super().capture_case(row,prefix)

        def run_reference(self,row,name):
            self.select_arm(False)
            return super().run_reference(row,name)

        def run_method(self,row,name,method,*,tokens=32,qualification=False,verify_sparse=False,fraction=None):
            require(method in ('Full','S1.2','Index20'),'Unknown three-arm method')
            if method!='Index20':
                self.select_arm(False)
                return super().run_method(row,name,method,tokens=tokens,qualification=qualification,verify_sparse=verify_sparse)
            require(not verify_sparse,'Index20 changes math policy; original S1 sparse identity is inapplicable')
            # Start the request clock BEFORE selecting the arm or constructing
            # the request. Selection is executed in the third-layer online pass.
            began=time.perf_counter()
            self.select_arm(True,fraction=fraction,diagnostic_fingerprints=qualification and fraction==0.);self.suite_method='Index20'
            plan=self.rag_case_plan
            require(row['input_ids_sha256']==plan['input_ids_sha256']and plan['effective_route']=='s12',
                    'Index20 requires unchanged original cacheable input, no Full fallback')
            from service.engine_bridge import TokenRequest
            request=TokenRequest(name,tuple(row['input_ids']),tokens,'s2',began,documents=tuple(plan['effective_documents']))
            try:return self._internal(request,name,'sensitive',qualification_only=qualification)
            finally:self.select_arm(False)

        def cache_accounting(self):
            report=None
            if self.controller is not None:
                with self.allow_sync():report=self.controller.cache.report()
            return self.driver.gather(report)[0]
    return Index20BenchRuntime
