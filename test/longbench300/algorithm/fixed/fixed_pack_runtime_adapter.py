"""Private Cone15 fixed15 factory extension; original source and module globals are intact.

Wire this factory through the existing frozen DP4/admission bootstrap. Equal
native-frame batches retain the original admission/world schedule. Heterogeneous
frame batching needs the separately qualified delayed adapter and is not enabled
implicitly here. This module performs no model import, CUDA work or launch.
"""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
from types import FunctionType, SimpleNamespace
from fixed_pack_controller import require, digest
from fixed_pack_policy import ARM_FRACTIONS, canonical_packed_arm, fraction_for_method

OWN_FILES = ('cone15_ffn.py','cone15_ffn_native_proof.py','cone15_audit.py','cone15_plan.py','cone_query.py','cone_input.py', 'fixed15_geometry.py', 'fixed_pack_policy.py', 'fixed_pack_attention.py', 'fixed_pack_controller.py',
    'fixed_pack_audit.py', 'fixed_pack_runtime_adapter.py')
SHARED_FILES = ('runtime/fixed15_online.py','runtime/fixed15_online_triton.py','runtime/lastkv_packed_only.py', 'runtime/packed_attention.py', 'kvpack/__init__.py', 'kvpack/pack.py',
    'kvpack/kernels.py', 'kvpack/check_cuda.py', 'validation/tensor_checks.py')
CPU_MODULES = (
    ('native_source_root', 'redknot', 'redknot/__init__.py'),
    ('native_source_root', 'redknot.decode_evidence', 'redknot/decode_evidence.py'),
    ('gated_root', 'kv_cache_engram', 'kv_cache_engram/__init__.py'),
    *[('gated_root', 'kv_cache_engram.'+name, 'kv_cache_engram/'+name+'.py')
      for name in ('core', 'cache', 'calibration', 'gate', 'runtime')],
    ('frozen_runtime_root', 'index20_lastkv_audit', 'index20_lastkv_audit.py'),
    ('frozen_runtime_root', 'index20_lastkv_contract', 'index20_lastkv_contract.py'),
    ('rag_runtime_root', 'rag_dspark_runtime', 'rag_dspark_runtime.py'),
)


def canonical_extension(original):
    def canonical_method(method, selector_strategy='mean'):
        arm = canonical_packed_arm(method)
        if arm is not None:
            require(selector_strategy == 'max', 'Cone15 is native layer2 max only')
            return arm, True, 'max'
        require(method == 'Full', 'Only Full and Cone15 are enabled in this independent runtime')
        return original(method, selector_strategy)
    return canonical_method


def fixed_multiowner_controller_class(baseclass):
    from multiowner_controller import multiowner_controller_class
    from fixed_pack_controller import fixed_pack_controller_class
    from cone15_ffn import cone15_controller_class
    return fixed_pack_controller_class(multiowner_controller_class(cone15_controller_class(baseclass)))


def validate_candidate_phase(phase, row, *, expected_method):
    if expected_method in ARM_FRACTIONS:
        from fixed_pack_audit import validate_fixed_pack_phase
        return validate_fixed_pack_phase(phase, row, expected_method=expected_method)
    raise ValueError('Only the Cone15 candidate may use this phase auditor')


class FixedRuntimeRewrite(ast.NodeTransformer):
    def __init__(self):
        self.controller_imports = self.validator_imports = self.configurations = self.validators = self.fractions = 0

    def visit_ImportFrom(self, node):
        names = [(a.name, a.asname) for a in node.names]
        if node.module == 'multiowner_controller' and names == [('multiowner_controller_class', None)]:
            self.controller_imports += 1
            return ast.copy_location(ast.ImportFrom(module='fixed_pack_runtime_adapter', level=0,
                names=[ast.alias(name='fixed_multiowner_controller_class', asname='multiowner_controller_class')]), node)
        if node.module == 'index20_lastkv_contract' and names == [('validate_lastkv_phase', None)]:
            self.validator_imports += 1
            return ast.copy_location(ast.ImportFrom(module='fixed_pack_runtime_adapter', level=0,
                names=[ast.alias(name='validate_candidate_phase', asname='validate_lastkv_phase')]), node)
        return node

    def visit_Assign(self, node):
        shape = ast.dump(ast.parse("method == 'S1.3'", mode='eval').body, include_attributes=False)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'index20' \
                and ast.dump(node.value, include_attributes=False) == shape:
            self.configurations += 1
            node.value = ast.parse("method in FIXED_ARM_FRACTIONS", mode='eval').body
            extra = ast.parse("c.configure_fixed_pack(method in FIXED_ARM_FRACTIONS, method=method, "
                "kv_pack_budget_bytes=self._fixed_pack_budget_bytes)").body[0]
            return [node, ast.copy_location(extra, node)]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == 'c' and node.func.attr == 'configure_index20':
            enabled = next((k.value for k in node.keywords if k.arg == 'enabled'), None)
            if isinstance(enabled, ast.Constant) and enabled.value is False:
                # Original startup all-selected/native identity configuration.
                return node
            require(isinstance(enabled, ast.Name) and enabled.id == 'index20',
                'Unexpected original index configuration scope')
            fractions = [k for k in node.keywords if k.arg == 'fraction']
            require(len(fractions) == 1 and isinstance(fractions[0].value, ast.Constant)
                and fractions[0].value.value == .2, 'Original fixed index budget wiring changed')
            fractions[0].value = ast.parse('fixed_fraction_for_method(method)', mode='eval').body
            self.fractions += 1
        if isinstance(node.func, ast.Name) and node.func.id == 'validate_lastkv_phase':
            require(len(node.args) == 2 and not node.keywords, 'Original phase auditor call changed')
            self.validators += 1
            node.keywords.append(ast.keyword(arg='expected_method', value=ast.Name(id='arm', ctx=ast.Load())))
        return node

    def finish(self):
        require((self.controller_imports, self.validator_imports, self.configurations, self.validators, self.fractions) == (1,1,1,1,1),
            'Original controller/configuration/phase-validator structure changed')


def private_preserved_module(preserved):
    function = preserved.runtime_class
    visitor = FixedRuntimeRewrite()
    tree = visitor.visit(ast.parse(textwrap.dedent(inspect.getsource(function))))
    ast.fix_missing_locations(tree); visitor.finish()
    namespace = dict(function.__globals__, canonical_method=canonical_extension(preserved.canonical_method),
        FIXED_ARM_FRACTIONS=dict(ARM_FRACTIONS), fixed_fraction_for_method=fraction_for_method)
    exec(compile(tree, inspect.getsourcefile(function), 'exec'), namespace)
    return SimpleNamespace(**dict(vars(preserved), runtime_class=namespace[function.__name__],
        canonical_method=namespace['canonical_method']))


def runtime_class(parent, preserved, admission, cfg):
    """Usable factory for the existing native/admission runtime bootstrap.

    The caller still pins dependencies and supplies native qualification and
    two-arm comparison orchestration. No progressive cohort eligibility or
    production queue is borrowed. B4 only, max only; B8 remains unqualified.
    """
    private = private_preserved_module(preserved)
    base = admission.runtime_class(parent, private)

    class FixedPackRuntime(base):
        def __init__(self, *args, **kwargs):
            self._fixed_pack_budget_bytes = cfg.get('fixed_pack_budget_bytes', 1024**3)
            super().__init__(*args, **kwargs)

        def initialize(self):
            result = super().initialize()
            from cone15_ffn import collect_world_guard
            collect_world_guard(self)
            return result

        def _configure_many(self, requests, phase, mode, method, strategy, verify_sparse=False):
            if self.controller is not None:
                checker = None
                if self.rank in requests and method in ARM_FRACTIONS and phase.endswith('.qualification'):
                    from tensor_checks import QualificationChecks
                    checker = QualificationChecks(
                        input_ids_sha256=digest(list(requests[self.rank].prompt_token_ids)),
                        owner_rank=self.rank, qualification_only=True,
                        benchmark_request=False, torch_module=self.torch)
                self.controller.configure_fixed_pack_checks(checker)
            return super()._configure_many(requests, phase, mode, method, strategy, verify_sparse)

        def run_batch(self, method, input_rows, **kwargs):
            require(len(input_rows) == kwargs.get('request_batch_size') == 4,
                'Cone15 integration requires actual B4')
            require(kwargs.get('selector_strategy', 'mean') == 'max',
                'Matched Full/Cone15 comparison requires max identity')
            return super().run_batch(method, input_rows, **kwargs)

    return FixedPackRuntime


def pinned(cfg, path):
    path = Path(path).resolve()
    require(path.is_file() and cfg['source_sha256'].get(str(path)) ==
        hashlib.sha256(path.read_bytes()).hexdigest(), 'Unpinned fixed-pack dependency: '+str(path))
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies(cfg):
    """CPU-only source admission; exact module origins checked before bootstrap."""
    here = Path(__file__).resolve().parent
    require(Path(cfg['fixed_pack_runtime_root']).resolve() == here and
        cfg['multiowner_active_ranks'] == [0,1,2,3] and cfg['padding_allowed'] is False and
        cfg.get('selector_strategy') == 'max', 'Explicit fixed-pack source and B4 max policy required')
    require(type(cfg.get('fixed_pack_budget_bytes')) is int and cfg['fixed_pack_budget_bytes'] > 0,
        'Explicit positive pack memory budget required')
    old, ready = (Path(cfg[key]).resolve() for key in ('multiowner_v1_root', 'admission_v2_root'))
    shared = Path(cfg['fixed_pack_shared_root']).resolve()
    native = Path(cfg['native_source_root']).resolve()
    frozen = Path(cfg.get('frozen_runtime_root') or cfg['frozen_lastkv_runtime_root']).resolve()
    rag = Path(cfg['rag_runtime_root']).resolve()
    cpu_roots = dict(native_source_root=native, gated_root=native/'GatedEngram',
        frozen_runtime_root=frozen, rag_runtime_root=rag)
    for name in OWN_FILES:
        pinned(cfg, here/name)
    for name in SHARED_FILES:
        pinned(cfg, shared/name)
    for key, name, relative in CPU_MODULES:
        pinned(cfg, cpu_roots[key]/relative)
    for name in ('multiowner_runtime.py', 'multiowner_transport.py', 'multiowner_worker.py',
                 'multiowner_controller.py', 'world_plan_coordinator.py', 'gpu_batch_contract.py'):
        pinned(cfg, old/name)
    for name in ('multiowner_runtime_v2.py', 'admission_gate.py'):
        pinned(cfg, ready/name)
    for root in (*cpu_roots.values(), old, ready, shared, shared/'runtime', shared/'validation', here):
        require(root.is_dir(), 'Missing dependency source root: '+str(root))
        sys.path.insert(0, str(root))
    admission = load_module(ready/'multiowner_runtime_v2.py', 's15_fixed_preserved_admission_v2')
    preserved = admission.load_preserved(cfg)
    named = [(name[:-3], here/name) for name in OWN_FILES]
    named += [('packed_attention', shared/'runtime/packed_attention.py'),
        ('kvpack', shared/'kvpack/__init__.py'), ('kvpack.pack', shared/'kvpack/pack.py'),
        ('tensor_checks', shared/'validation/tensor_checks.py')]
    named += [(name, old/(name+'.py')) for name in ('world_plan_coordinator',
        'multiowner_worker', 'multiowner_controller', 'multiowner_transport', 'gpu_batch_contract')]
    named += [('admission_gate', ready/'admission_gate.py')]
    named += [(name, cpu_roots[key]/relative) for key, name, relative in CPU_MODULES]
    for name, path in named:
        actual = importlib.import_module(name)
        require(Path(actual.__file__).resolve() == pinned(cfg, path), 'Imported another dependency: '+name)
    return preserved, admission


@contextmanager
def create_runtime(cfg, *, rank, output_dir=None):
    """Existing eight-rank native launch calls this; no subprocess created here."""
    preserved, admission = load_dependencies(cfg)
    original = preserved.create_runtime.__wrapped__
    factory = lambda parent: runtime_class(parent, preserved, admission, cfg)
    private = FunctionType(original.__code__, dict(original.__globals__, runtime_class=factory),
        original.__name__, original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    with contextmanager(private)(cfg, rank=rank, output_dir=output_dir) as runtime:
        yield runtime
