"""Private binding adapter: eager prefill may coexist with native decode graphs."""
from __future__ import annotations
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

REFERENCE_SHA256='f25097ab9ebe51662c21b9652af472e43d7664e368276a4c317fc133d02e913e'


def _shape(node):
    return ast.dump(node,include_attributes=False)


def validate_native_prepare_source(source):
    """Assert the actual pinned native source separates prefill from decode graphs."""
    node=ast.parse(textwrap.dedent(source)).body[0]
    def assignment(name):
        return [v.value for v in node.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in v.targets)]
    expected={
        'is_prompt':'seq_metadata_list[0].is_prompt',
        'run_cuda_graph':'not is_prompt and self.use_cuda_graph and bool(self.cuda_graph_runner.graphs)',
    }
    for name,expression in expected.items():
        values=assignment(name)
        if len(values)!=1 or _shape(values[0])!=_shape(ast.parse(expression,mode='eval').body):
            raise ValueError('Native prefill/decode graph contract changed: '+name)
    branches=[v for v in node.body if isinstance(v,ast.If) and isinstance(v.test,ast.Name) and v.test.id=='run_cuda_graph']
    expected_eager=ast.parse('graph_size, cuda_graph_used = 0, None').body[0]
    expected_return=ast.parse('return input_metadata, total_q, run_cuda_graph, cuda_graph_used, graph_size').body[0]
    if len(branches)!=1 or not any(_shape(v)==_shape(expected_eager) for v in branches[0].orelse) or not any(_shape(v)==_shape(expected_return) for v in node.body):
        raise ValueError('Native eager metadata/return contract changed')
    return dict(prefill_uses_graph=False,decode_graphs_unchanged=True,
                native_prepare_source_sha256=hashlib.sha256(source.encode()).hexdigest())


def _private_installer(legacy):
    if hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()!=REFERENCE_SHA256:
        raise ValueError('Canonical binding reference source changed')
    node=ast.parse(textwrap.dedent(inspect.getsource(legacy.install_request_binding)))
    obsolete=ast.parse('getattr(runner, "use_cuda_graph", None) is not False',mode='eval').body
    removed=0
    class Rewrite(ast.NodeTransformer):
        def visit_BoolOp(self,value):
            nonlocal removed
            self.generic_visit(value)
            kept=[]
            for child in value.values:
                if isinstance(value.op,ast.Or) and _shape(child)==_shape(obsolete):removed+=1
                else:kept.append(child)
            value.values=kept
            return value
    node=Rewrite().visit(node)
    if removed!=1:raise ValueError('Expected exactly one legacy global graph guard')
    ast.fix_missing_locations(node)
    namespace=dict(vars(legacy))
    exec(compile(node,legacy.__file__,'exec'),namespace)
    return namespace['install_request_binding']


def install_request_binding(model_runner,get_request,*,native_ops=None):
    import canonical_request_binding as legacy
    # Never toggle use_cuda_graph: the real method and graph objects are untouched.
    proof=validate_native_prepare_source(inspect.getsource(model_runner._prepare_input_driver))
    handle=_private_installer(legacy)(model_runner,get_request,native_ops=native_ops)
    handle.native_decode_graph_contract=proof
    handle.reference_source_sha256=REFERENCE_SHA256
    return handle
