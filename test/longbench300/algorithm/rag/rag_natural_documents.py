"""CPU-only RAG span admission; no agent markers or session certificate.

A cache block shorter than256 remains a native full-compute span because the
unchanged S1 policy requires both tail128 and the preceding halo128. The
prompt itself is never rewritten, truncated, padded, or re-tokenized here.
"""
from __future__ import annotations
from functools import lru_cache
import ast
import inspect
import textwrap

from rag_dspark_runtime import digest,require,seal


def natural_documents(input_ids,documents):
    require(isinstance(input_ids,(tuple,list)) and input_ids and all(type(v) is int and 0<=v<2**31 for v in input_ids),
            'Nonempty native int32 prompt token IDs required')
    require(isinstance(documents,(tuple,list)),'Explicit source document token spans required')
    result=[];previous=0
    for i,doc in enumerate(documents):
        a,z=doc.get('input_start'),doc.get('input_end')
        require(type(a) is type(z) is int and previous<=a<z<=len(input_ids),'Natural document spans overlap or leave input bounds')
        key=digest(list(input_ids[a:z]))
        require(doc.get('token_ids_sha256')==key and doc.get('document_id',key)==key
            and doc.get('tokens',z-a)==z-a and doc.get('offline_position',0)==0
            and doc.get('online_position',a)==a,'Natural source token/span identity changed')
        result.append(dict(doc,input_start=a,input_end=z,tokens=z-a,document_id=key,token_ids_sha256=key,
                           offline_position=0,online_position=a,source_document_ordinal=i))
        previous=z
    return result


def natural_cache_documents(ids,documents,*,expected_document_count=None):
    checked=natural_documents(ids,documents or ())
    require(expected_document_count is None or len(checked)==expected_document_count,'Natural cache occurrence count changed')
    require(all(d['tokens']>=256 for d in checked),'Short spans must remain native full-compute rows')
    return tuple(dict(start=d['input_start'],end=d['input_end'],key=d['document_id']) for d in checked)


def plan_case(input_ids,documents):
    checked=natural_documents(input_ids,documents);eligible=[d for d in checked if d['tokens']>=256]
    short=[d for d in checked if d['tokens']<256];total=len(input_ids)
    covered={p for d in eligible for p in range(d['input_start'],d['input_end'])}
    tails={p for d in eligible for p in range(d['input_end']-128,d['input_end'])}
    selected=sorted((set(range(total))-covered)|tails)
    chunks=[sum(a<=p<min(a+8192,total) for p in selected) for a in range(0,total,8192)]
    group_rows=[sum(chunks[i:i+17]) for i in range(0,len(chunks),17)]
    unique={d['document_id']:d for d in eligible}
    payload=sum(37*128*512*2+9216*d['tokens'] for d in unique.values())
    reason='no_cacheable_document' if not eligible else 'fixed_group17_selected_rows_exceed_8192' if max(group_rows)>8192 else None
    return seal(dict(schema='rag-natural-span-plan-v1',input_ids_sha256=digest(list(input_ids)),input_tokens=total,
        source_documents=checked,effective_documents=eligible if reason is None else [],short_documents=short,
        unique_capture_documents=list(unique.values()) if reason is None else [],effective_route='native_full' if reason else 's12',
        fallback_reason=reason,predicted_bank_bytes=payload,selected_tokens=len(selected),group_selected_rows=group_rows,
        padding_tokens_added=0,truncation_tokens=0,input_token_ids_rewritten=False,
        first_three_layers_full=True,minimum_cache_block_tokens=256,short_spans_full_compute=True,
        cache_build_is_separate=True,fresh_per_case_dense_qualification_required=True,
        session_certificate_required=False,agent_markers_required=False))


@lru_cache(maxsize=1)
def natural_controller_class():
    """Clone admission only; arithmetic bytecodes and original classes survive."""
    from redknot.controller import _clone_class
    from kv_cache_engram.s1_runtime import S1Controller
    from kv_cache_engram.s1_system_runtime import S1SystemController
    from kv_cache_engram.s11_qualification import S11QualificationController
    classes={};replacement=dict(s1_documents=natural_cache_documents)
    base=_clone_class(S1Controller,object,classes,replacement)
    system=_clone_class(S1SystemController,base,classes,replacement)
    qualification=_clone_class(S11QualificationController,system,classes,replacement)
    class NaturalRagController(qualification):
        def __init__(self,*args,**kwargs):
            kwargs.pop('expected_document_count',None)
            super().__init__(*args,expected_document_count=8,**kwargs)
            self.expected_document_count=0

        def prepare_case(self,*,input_ids,documents,index_device,**kwargs):
            require(not kwargs,'Natural RAG has no extra empty-group admission')
            self.expected_document_count=len(documents)
            return super().prepare_case(input_ids=input_ids,documents=documents,index_device=index_device)

        def reset(self,*args,**kwargs):
            mode=args[1] if len(args)>1 else kwargs.get('mode')
            if mode not in ('qualified_native','all_selected','capture_independent'):
                self.expected_document_count=len(kwargs.get('documents') or ())
            return super().reset(*args,**kwargs)
    return NaturalRagController


def install_natural_controller_initializer(fixed_class,parent):
    """Private initializer differs at exactly its controller constructor import.

    No qualification/admission check is removed; each new natural prompt still
    gets the full original same-policy dense32 vsgroup17 numerical comparison.
    """
    original=fixed_class._initialize_s2
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    changes=0
    class Rewrite(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            nonlocal changes
            if node.module=='kv_cache_engram.s11_qualification' and [x.name for x in node.names]==['S11QualificationController']:
                changes+=1
                return ast.copy_location(ast.parse('S11QualificationController = natural_controller_class()').body[0],node)
            return node
    tree=Rewrite().visit(tree);require(changes==1,'Preserved initializer controller import changed')
    namespace=dict(original.__globals__,parent=parent,natural_controller_class=natural_controller_class)
    exec(compile(ast.fix_missing_locations(tree),'<rag-private-natural-controller>','exec'),namespace)
    return namespace[original.__name__]
