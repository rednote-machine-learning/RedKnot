"""Private controller hierarchy with natural-document admission.

Frozen controller/operator method bytecode is retained. Only input validation,
request-boundary lifetime and optional capture serialization are adapted.
No imported frozen class or module globals are mutated.
"""
from __future__ import annotations

from functools import lru_cache
from types import FunctionType
import ast
import copy
import inspect
import textwrap

from kv_cache_engram.core import digest, require, token_digest
from kv_cache_engram.s1_runtime import S1Controller
from kv_cache_engram.s1_system_runtime import S1SystemController
from kv_cache_engram.s11_qualification import S11QualificationController
from kv_cache_engram.s1_system_cache import S1SystemCache
from kv_cache_engram.s1_cache import SWA_LAYERS, GLOBAL_SOURCES
from .admission import s2_documents, validate_plan

MEMORY_ONLY_CAPTURE = object()


class MemoryCaptureCache(S1SystemCache):
    def begin_capture(self, key, length, output_dir=None):
        return super().begin_capture(key, length,
            output_dir=None if output_dir is MEMORY_ONLY_CAPTURE else output_dir)


def _cell(value):
    return (lambda: value).__closure__[0]


def _clone_class(original, base, class_map, replacements):
    cloned = type('S2Private' + original.__name__, (base,), {'__module__': __name__, '__doc__': original.__doc__})
    class_map[original] = cloned
    for name, member in original.__dict__.items():
        descriptor = type(member) if isinstance(member, (staticmethod, classmethod)) else None
        fn = member.__func__ if descriptor else member
        if not isinstance(fn, FunctionType):
            if name not in ('__dict__', '__weakref__', '__module__', '__doc__'):
                setattr(cloned, name, member)
            continue
        namespace = dict(fn.__globals__)
        namespace.update({old.__name__: new for old, new in class_map.items()})
        namespace.update(replacements)
        closure = tuple(_cell(class_map.get(c.cell_contents, c.cell_contents))
                        if isinstance(c.cell_contents, type) else c for c in (fn.__closure__ or ())) or None
        copied = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, closure)
        copied.__kwdefaults__ = fn.__kwdefaults__
        copied.__annotations__ = dict(fn.__annotations__)
        setattr(cloned, name, descriptor(copied) if descriptor else copied)
    return cloned


def _admission_reset(original, owner, message, replacement):
    """Change exactly one admission statement in a private reset method.

    Compiling inside a temporary class retains zero-argument super semantics.
    No arithmetic/operator/row construction statement may be rewritten.
    """
    node = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    count = 0
    class Rewrite(ast.NodeTransformer):
        def visit_Expr(self, item):
            nonlocal count
            value = item.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'require' \
                    and len(value.args) == 2 and isinstance(value.args[1], ast.Constant) and value.args[1].value == message:
                count += 1
                return ast.copy_location(ast.parse(replacement).body[0], item)
            return self.generic_visit(item)
    node = Rewrite().visit(node)
    require(count == 1, 'frozen admission statement changed; refusing private rewrite')
    wrapped = ast.Module(body=[ast.ClassDef(name='_S2AdmissionTemplate', bases=[], keywords=[], body=[node], decorator_list=[])], type_ignores=[])
    namespace = dict(owner.reset.__globals__)
    exec(compile(ast.fix_missing_locations(wrapped), '<s2-explicit-session-admission>', 'exec'), namespace)
    fn = namespace['_S2AdmissionTemplate'].reset
    closure = tuple(_cell(owner) if isinstance(c.cell_contents, type) and c.cell_contents.__name__ == '_S2AdmissionTemplate' else c
                    for c in (fn.__closure__ or ())) or None
    result = FunctionType(fn.__code__, owner.reset.__globals__, original.__name__, original.__defaults__, closure)
    result.__kwdefaults__ = original.__kwdefaults__
    result.s2_admission_rewrite_count = count
    return result


@lru_cache(maxsize=1)
def adapted_controller_class():
    classes = {}
    replacements = dict(s1_documents=s2_documents, S1SystemCache=MemoryCaptureCache)
    base = _clone_class(S1Controller, object, classes, replacements)
    base.reset = _admission_reset(S1Controller.reset, base, 'fresh same-policy S1 qualification required',
                                 'self.require_sparse_admission(verify_sparse)')
    system = _clone_class(S1SystemController, base, classes, replacements)
    system.reset = _admission_reset(S1SystemController.reset, system,
        'system configuration lacks actual per-chunk numerical qualification',
        'self.require_system_admission(self._configured_group)')
    qualification = _clone_class(S11QualificationController, system, classes, replacements)

    class S2Controller(qualification):
        def __init__(self, *args, **kwargs):
            require('expected_document_count' not in kwargs, 'S2 document count comes from each natural request')
            # Constructor's legacy count controls no operator or tensor. Replace
            # it with the actual occurrence count before each request admission.
            super().__init__(*args, expected_document_count=8, **kwargs)
            self.expected_document_count = 0
            self.cache_output_dir = MEMORY_ONLY_CAPTURE
            self.s2_request_generation = 0
            self.s2_request_id = self.s2_plan = None
            self.session_certificate = None
            self._s2_admission_context = None
            self.session_admission_used = False

        def set_session_certificate(self, certificate, *, session_id, source_sha256):
            from .session import validate_certificate, validate_installation_process
            require(self.state in ('IDLE', 'COMPLETE') and self.cache.capture is None,
                    'cannot install session admission during an active request')
            validate_certificate(certificate, scope=self.scope, session_id=session_id, source_sha256=source_sha256)
            validate_installation_process(certificate)
            self.session_certificate = copy.deepcopy(certificate)

        def _session_admitted(self):
            context = self._s2_admission_context
            certificate = self.session_certificate
            if context is None or certificate is None or self.s2_plan is None:
                return False
            mode, rid, input_sha, verify_sparse = context
            plan = self.s2_plan
            valid = (mode == 'sensitive' and not verify_sparse and rid == self.s2_request_id
                     and input_sha == plan['input_ids_sha256'] and plan['route'] == 's2'
                     and plan['session_certificate_sha256'] == certificate['sha256']
                     and plan['session_supported_geometry'] == certificate['supported_geometry']
                     and plan['bank_budget_bytes'] == certificate['bank_budget_bytes'])
            if valid: self.session_admission_used = True
            return valid

        def require_sparse_admission(self, verify_sparse):
            require(self.sparse_qualified or (verify_sparse and self.sparse_reference_input == self.input_sha
                    and self.sparse_reference) or self._session_admitted(),
                    'fresh same-policy S1 qualification or explicit supported S2 session admission required')

        def require_system_admission(self, group):
            require(self.system_qualified[group] or group == 17 and self._session_admitted(),
                    'system configuration lacks per-request or explicit S2 session admission')

        def reset(self, *args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get('mode')
            rid = args[0] if args else kwargs.get('rid')
            self._s2_admission_context = (mode, rid, token_digest(kwargs['input_ids']), kwargs.get('verify_sparse', False))
            try:
                return super().reset(*args, **kwargs)
            finally:
                self._s2_admission_context = None

        def begin_request(self, plan, *, request_id):
            validate_plan(plan, scope=self.scope)
            require(type(request_id) is str and request_id and request_id != self.s2_request_id,
                    'fresh logical request identity required')
            require(self.state in ('IDLE', 'COMPLETE') and self.ctx is None and self.capture_frame is None
                    and self.cache.capture is None, 'cannot reuse a bank during an active/unfinished request')
            require(self._batch is None or self._batch.state != 'ACTIVE', 'pending deep batch cannot cross requests')
            resident = self.resident_bank_state()
            require(plan['resident_bank_bytes'] == resident['resident_bank_bytes']
                    and plan['resident_document_ids'] == resident['resident_document_ids'], 'plan resident bank inventory became stale')
            for key in self.cache.bank:
                self.cache.entry(*key)
            # Keep immutable document tensors, native qualification and hooks.
            # Prompt-specific references and global/phase workspaces never cross.
            self.case_binding = self.case_input = None
            self.sparse_reference, self.sparse_reference_input, self.sparse_qualified = {}, None, False
            self.system_qualified = {width: False for width in self.system_qualified}
            self.cache.global_bases, self.cache.global_carry, self.cache.global_events = {}, {}, []
            self.cache.preparation_s = 0.0
            self._empty_group_case = None
            self.expected_document_count = len(plan['effective_documents'])
            self.s2_request_generation += 1
            self.s2_request_id, self.s2_plan = request_id, plan
            self.session_admission_used = False
            return self.cache_misses(plan)

        def resident_bank_state(self):
            expected = {(kind, layer) for kind, layers in (('swa_halo', SWA_LAYERS), ('global_raw', GLOBAL_SOURCES))
                        for layer in layers}
            by_document = {}
            total = 0
            for key, kind, layer in self.cache.bank:
                tensor = self.cache.entry(key, kind, layer)
                total += tensor.numel() * tensor.element_size()
                by_document.setdefault(key, set()).add((kind, layer))
            require(all(parts == expected for parts in by_document.values()), 'unfinished shared capture cannot become resident inventory')
            return dict(resident_bank_bytes=total, resident_document_ids=sorted(by_document))

        def cache_misses(self, plan):
            validate_plan(plan, scope=self.scope)
            expected = {(kind, layer) for kind, layers in (('swa_halo', SWA_LAYERS), ('global_raw', GLOBAL_SOURCES))
                        for layer in layers}
            misses = []
            lengths = {d['document_id']: d['tokens'] for d in plan['effective_documents']}
            for key in plan['unique_document_ids']:
                present = {(kind, layer) for doc, kind, layer in self.cache.bank if doc == key}
                if not present:
                    misses.append(key)
                    continue
                require(present == expected, 'partial/foreign shared document bank is not a hit')
                for kind, layer in expected:
                    self.cache.entry(key, kind, layer)
                    require(self.cache.bank[key, kind, layer]['length'] == lengths[key], 'shared document token length changed')
            return misses

        def prepare_case(self, *, input_ids, documents, index_device, **kwargs):
            require(not kwargs and self.s2_plan is not None, 'S2 prepare requires a logical request plan')
            validate_plan(self.s2_plan, scope=self.scope, input_ids=input_ids)
            require(self.s2_plan['route'] == 's2' and documents == self.s2_plan['effective_documents'],
                    'only admitted natural cache spans may enter sparse execution')
            require(not self.cache_misses(self.s2_plan), 'capture all unique missing documents before request preparation')
            return super().prepare_case(input_ids=input_ids, documents=documents, index_device=index_device)

        def finish(self):
            report = super().finish()
            report.update(s2_adapter_schema='s2-natural-shared-bank-controller-v1',
                s2_request_generation=self.s2_request_generation, s2_request_id=self.s2_request_id,
                s2_request_plan_sha256=None if self.s2_plan is None else self.s2_plan['sha256'],
                gpu_bank_constant_across_arms=False, cache_storage_policy='s2_persistent_immutable_document_bank',
                capture_serialized_to_disk=self.cache_output_dir is not MEMORY_ONLY_CAPTURE,
                fresh_prompt_qualification_required=not self.session_admission_used,
                qualification_scope='representative_session' if self.session_admission_used else 'strict_request',
                session_certificate_sha256=self.session_certificate['sha256'] if self.session_admission_used else None,
                per_request_numeric_comparison=bool(self.verify_sparse),
                legacy_qualification_flags_forged=False, mathematical_policy_changed=False)
            return report

    S2Controller.private_base_class = base
    S2Controller.private_system_class = system
    S2Controller.private_qualification_class = qualification
    return S2Controller
