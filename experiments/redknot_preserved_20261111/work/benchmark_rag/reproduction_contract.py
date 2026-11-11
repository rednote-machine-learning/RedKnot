"""Private 64K/128K CPU audit namespace for the frozen S1.2 group17 runtime.

No imported frozen module or function globals are changed. Geometry, source
partitioning, cache/state/mHC and native numerical checks remain inherited.
Only the declared input population is widened to the already pinned 64K rows.
"""
from functools import lru_cache
from types import FunctionType,SimpleNamespace
from profile_group_contract import group_contract
from gpu_pilot import read_s11_results as original

BAND_LIMITS={'64k':(57344,73728),'128k':(114688,147456)}
TARGET_INPUT_TOKENS={'64k':65536,'128k':131072}
BAND_INDICES={'64k':tuple(range(3,22,2)),'128k':tuple(range(2,21,2))}


def clone_scope(source):
    namespace=dict(source)
    for name,value in source.items():
        if isinstance(value,FunctionType) and value.__globals__ is source:
            copy=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            copy.__kwdefaults__=value.__kwdefaults__;copy.__annotations__=value.__annotations__.copy()
            copy.__dict__.update(value.__dict__);copy.__module__=__name__;namespace[name]=copy
    return namespace


@lru_cache(maxsize=1)
def reproduction_contract():
    grouped=group_contract(17)
    namespace=clone_scope(grouped.audit_controller.__globals__)
    namespace.update(LENGTH_BANDS=('64k','128k'),TARGET_INPUT_TOKENS=dict(TARGET_INPUT_TOKENS),
        BAND_LIMITS=dict(BAND_LIMITS),CASE_INDICES=tuple(range(2,22)),EVALUATION_INDICES=tuple(range(2,22)))

    def length_identity(row):
        band=row.get('length_band');index=row.get('index');total=row.get('input_tokens')
        original.require(band in BAND_INDICES and type(index) is int and index in BAND_INDICES[band],
            'foreign pinned 64K/128K input index or band')
        ordinal=(index-1)//2 if band=='64k' else index//2
        original.require(row.get('dataset')=='loong' and row.get('length_band_ordinal')==ordinal
            and row.get('target_input_tokens')==TARGET_INPUT_TOKENS[band] and type(total) is int
            and BAND_LIMITS[band][0]<=total<=BAND_LIMITS[band][1] and 270336-total>=4096,
            'actual original 64K/128K target, ordinal, token band or context capacity differs')
        return band

    def audit_length_inputs(rows):
        original.require(type(rows) is list and bool(rows) and len({row['index'] for row in rows})==len(rows),
            'selected reproduction inputs must be nonempty unique original rows')
        for row in rows:
            length_identity(row);namespace['audit_source_metadata'](row)
            ids=row.get('input_ids')
            original.require(type(ids) is list and len(ids)==row['input_tokens']
                and all(type(token) is int and 0<=token<2**31 for token in ids)
                and original.digest(ids)==row.get('input_ids_sha256'),'actual complete input token IDs differ')
            for lo,hi,key in namespace['documents'](row):
                original.require(original.digest(ids[lo:hi])==key,'original document token slice differs')
        return True

    namespace.update(length_identity=length_identity,audit_length_inputs=audit_length_inputs)
    names=('phase_execution','group_oracle','sparse_identity','audit_phase','audit_controller',
        'audit_system_execution','binding','comparison_scope','validate_metadata_qualification',
        'audit_source_metadata','documents','case_geometry','chunk_oracle','length_identity','audit_length_inputs')
    return SimpleNamespace(group_chunks=17,verify_phase=grouped.verify_phase,
        **{name:namespace[name] for name in names})
