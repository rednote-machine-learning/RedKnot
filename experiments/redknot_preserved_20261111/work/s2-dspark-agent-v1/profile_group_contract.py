"""Isolated CPU audit contracts for same-policy 4/17-chunk system grouping.

Clone existing auditor functions into a private global namespace. Never mutate
the imported frozen modules or their function globals. Only the admitted group
size and its explicit geometry oracle change; all row/cache/mHC checks remain.
No Torch import or GPU execution occurs here.
"""
from __future__ import annotations

from functools import lru_cache
from types import FunctionType, SimpleNamespace

from gpu_pilot import read_s11_results as original_reader
from gpu_pilot import run_s11_pilot as original_runner


def clone_namespace(module):
    namespace = dict(module.__dict__)
    for name, value in module.__dict__.items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            cloned = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            cloned.__kwdefaults__ = value.__kwdefaults__
            cloned.__annotations__ = value.__annotations__.copy()
            cloned.__dict__.update(value.__dict__)
            cloned.__module__ = __name__
            namespace[name] = cloned
    return namespace


@lru_cache(maxsize=2)
def group_contract(group_chunks):
    original_reader.require(type(group_chunks) is int and group_chunks in (4, 17),
                            'only separately qualified group4/group17 candidates admitted')

    def phase_execution(arm, *, batch_group_chunks=None):
        controller_arm = original_reader.controller_arm(arm)
        enabled = arm == 's1_system'
        size = group_chunks if enabled else 1
        if batch_group_chunks is not None:
            original_reader.require(type(batch_group_chunks) is int and batch_group_chunks in (1, 2, 4, 17),
                                    'unregistered isolated group size')
            original_reader.require(enabled or batch_group_chunks == 1, 'baseline cannot enable deep grouping')
            size = batch_group_chunks
        return dict(controller_arm=controller_arm, system_enabled=enabled, batch_group_chunks=size)

    def group_oracle(chunks, size):
        original_reader.require(type(size) is int and size in (1, 2, 4, 17), 'foreign isolated deep batch size')
        groups = []
        for offset in range(0, len(chunks), size):
            members = chunks[offset:offset + size]
            groups.append(dict(chunk_start=members[0]['chunk_start'], chunk_end=members[-1]['chunk_end'],
                original_rows=sum(c['original_rows'] for c in members),
                selected_positions=[p for c in members for p in c['selected_positions']], members=members))
        return groups

    reader = clone_namespace(original_reader)
    reader.update(phase_execution=phase_execution, group_oracle=group_oracle)
    runner = clone_namespace(original_runner)
    runner.update(phase_execution=phase_execution)
    # Expose only explicitly usable functions. Helpers with local imports of
    # frozen modules are not exposed as polymorphic grouping entry points.
    return SimpleNamespace(group_chunks=group_chunks, phase_execution=phase_execution,
        group_oracle=group_oracle, verify_phase=runner['verify_phase'],
        sparse_identity=reader['sparse_identity'], audit_phase=reader['audit_phase'],
        audit_controller=reader['audit_controller'], audit_system_execution=reader['audit_system_execution'],
        binding=reader['binding'], comparison_scope=reader['comparison_scope'],
        validate_metadata_qualification=reader['validate_metadata_qualification'])
