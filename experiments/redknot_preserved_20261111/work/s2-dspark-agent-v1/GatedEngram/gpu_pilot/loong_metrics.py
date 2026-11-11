"""Deterministic, non-official Loong structural scoring; no model/API judge.

A format failure is scored zero for the format-sensitive exact metric, but is
explicitly *not* a verdict that every factual claim is wrong. Strings preserve
numeric signs, units, percent marks and punctuation; no implicit unit conversion.
"""
from collections import Counter
import ast
import json
import math
import re
import unicodedata

SCHEMA='loong-custom-structured-metrics-v1'
PRIMARY_METRIC='loong_custom_structured_exact'


def _require(ok,message):
    if not ok:raise ValueError(message)


def normalize_text(value):
    _require(type(value) is str,'string required')
    return ' '.join(unicodedata.normalize('NFC',value).split())


def gold_schema(value):
    if type(value) is dict and all(type(k) is str and type(v) is list and all(type(x) is str for x in v) for k,v in value.items()):
        return 'category_multisets'
    if type(value) is dict and all(type(k) is str and type(v) is str for k,v in value.items()):return 'string_mapping'
    if type(value) is list and all(type(x) is str for x in value):return 'ordered_string_list'
    if type(value) is str:return 'free_text'
    if type(value) in (int,float) and (type(value) is int or math.isfinite(value)):return 'numeric_scalar'
    raise ValueError('unsupported original Loong gold schema')


def metric_policy(dataset='loong'):
    _require(dataset=='loong','unsupported Loong dataset')
    return dict(schema=SCHEMA,dataset=dataset,primary_metrics=[PRIMARY_METRIC],
        official_loong_gpt4_score=False,external_judge_called=False,
        accepted_formats=['whole_json','whole_python_literal','one_complete_code_fence'],
        no_prose_fragment_extraction=True,duplicate_mapping_keys_rejected=True,
        dict_list_order='unordered_multiset',top_level_list_order='ordered',
        string_normalization='NFC_and_whitespace_only_case_sensitive_punctuation_units_preserved',
        numeric_policy='finite_literal_exact_no_string_coercion_no_unit_or_percent_conversion',
        parse_failure_is_semantic_incorrectness_claim=False,
        format_failure_structured_exact=0,free_text_primary_score=None,
        auxiliary_metrics=['normalized_text_exact','normalized_text_token_f1','legacy_answer_em','legacy_answer_token_f1'],
        field_metric='micro_and_macro_path_value_multiset_overlap_empty_groups_are_fields',
        free_text_auxiliary_is_factual_judge=False,agreement_is_gold_accuracy=False)


def _json_pairs(pairs):
    out={}
    for key,value in pairs:
        _require(key not in out,'duplicate mapping key');out[key]=value
    return out


def _bad_constant(value):raise ValueError('nonfinite literal')


def _valid_literal(value):
    if value is None or type(value) in (str,bool,int):return
    if type(value) is float:
        _require(math.isfinite(value),'nonfinite literal');return
    if type(value) is list:
        for item in value:_valid_literal(item)
        return
    if type(value) is dict:
        _require(all(type(k) is str for k in value),'mapping keys must be strings')
        for item in value.values():_valid_literal(item)
        return
    raise ValueError('only JSON-compatible literal types allowed')


def parse_output(text):
    _require(type(text) is str,'full prediction text required')
    body=text.strip();fenced=False
    if body.startswith('```'):
        match=re.fullmatch(r'```(?:json|python)?[ \t]*\r?\n([\s\S]*?)\r?\n```',body)
        if not match:return dict(success=False,value=None,mode=None,error='not_one_complete_supported_code_fence')
        body=match.group(1).strip();fenced=True
    try:
        value=json.loads(body,object_pairs_hook=_json_pairs,parse_constant=_bad_constant)
        _valid_literal(value)
        return dict(success=True,value=value,mode=('fenced_' if fenced else '')+'json',error=None)
    except (ValueError,TypeError,RecursionError,OverflowError):pass
    try:
        tree=ast.parse(body,mode='eval')
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                keys=[ast.literal_eval(k) for k in node.keys]
                _require(all(type(k) is str for k in keys) and len(keys)==len(set(keys)),'duplicate or invalid literal mapping key')
        value=ast.literal_eval(tree);_valid_literal(value)
        return dict(success=True,value=value,mode=('fenced_' if fenced else '')+'python_literal',error=None)
    except (ValueError,TypeError,SyntaxError,RecursionError,OverflowError):
        return dict(success=False,value=None,mode=None,error='not_one_complete_safe_literal')


def _atoms(value,schema):
    if schema=='category_multisets':
        _require(type(value) is dict and all(type(k) is str and type(v) is list and all(type(x) is str for x in v) for k,v in value.items()),'category object shape differs')
        # Keys remain byte-exact; empty categories carry a separate field atom.
        atoms=[]
        for key,items in value.items():
            atoms.extend((key,'member',normalize_text(x)) for x in items)
            if not items:atoms.append((key,'empty'))
        return Counter(atoms)
    if schema=='string_mapping':
        _require(type(value) is dict and all(type(k) is str and type(v) is str for k,v in value.items()),'string mapping shape differs')
        return Counter((k,normalize_text(v)) for k,v in value.items()) or Counter({('empty_mapping',):1})
    if schema=='ordered_string_list':
        _require(type(value) is list and all(type(x) is str for x in value),'ordered list shape differs')
        return Counter((i,normalize_text(x)) for i,x in enumerate(value)) or Counter({('empty_list',):1})
    if schema=='numeric_scalar':
        _require(type(value) in (int,float) and (type(value) is int or math.isfinite(value)),'numeric literal shape differs')
        return Counter({('numeric',value):1})
    raise ValueError('field atoms unavailable for free text')


def _overlap(gold,predicted):
    correct=sum((gold&predicted).values());ng=sum(gold.values());np=sum(predicted.values())
    precision=correct/np if np else float(not ng);recall=correct/ng if ng else float(not np)
    f1=2*correct/(ng+np) if ng+np else 1.
    return dict(field_true_positive=correct,field_predicted=np,field_gold=ng,
        field_precision=precision,field_recall=recall,field_f1=f1)


def _aux(text,gold):
    target=gold if type(gold) is str else json.dumps(gold,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    left,right=normalize_text(text),normalize_text(target)
    a,b=Counter(left.split()),Counter(right.split());n=sum((a&b).values())
    return dict(normalized_text_exact=float(left==right),
        normalized_text_token_f1=2*n/(sum(a.values())+sum(b.values())) if a or b else 1.)


def score_answer(text,gold):
    schema=gold_schema(gold);structured=schema!='free_text';parsed=parse_output(text) if structured else None
    result=dict(schema=SCHEMA,gold_schema=schema,structured_eligible=structured,
        parse_success=parsed['success'] if parsed else None,parse_mode=parsed['mode'] if parsed else None,
        parse_error=parsed['error'] if parsed else None,shape_compatible=None,
        structured_exact=None,semantic_assessment='not_assessed_free_text' if not structured else 'format_failure_semantic_unassessed',
        parsed_prediction=parsed['value'] if parsed and parsed['success'] else None,
        field_true_positive=None,field_predicted=None,field_gold=None,field_precision=None,field_recall=None,field_f1=None,
        official_loong_score=False,**_aux(text,gold))
    if not structured:return result
    gold_atoms=_atoms(gold,schema)
    try:
        _require(parsed['success'],'prediction parse failed');predicted_atoms=_atoms(parsed['value'],schema)
    except ValueError:
        result.update(shape_compatible=False,structured_exact=0.)
        if parsed['success']:result['semantic_assessment']='shape_failure_semantic_unassessed'
        return result
    result.update(shape_compatible=True,structured_exact=float(gold_atoms==predicted_atoms),
        semantic_assessment='custom_structural_match_only',**_overlap(gold_atoms,predicted_atoms))
    return result


def answer_agreement(left,right,gold):
    schema=gold_schema(gold);out=dict(schema='loong-answer-agreement-v1',gold_schema=schema,
        identical_display_text=left==right,normalized_text_equal=normalize_text(left)==normalize_text(right),
        structural_comparison_available=False,structural_equal=None,agreement_is_gold_correctness=False)
    if schema=='free_text':return out
    a,b=parse_output(left),parse_output(right)
    try:
        _require(a['success'] and b['success'],'parse failure')
        aa,bb=_atoms(a['value'],schema),_atoms(b['value'],schema)
    except ValueError:return out
    out.update(structural_comparison_available=True,structural_equal=aa==bb)
    return out


def summarize_scores(scores):
    _require(type(scores) is list and scores,'nonempty score list required')
    structured=[s for s in scores if s['structured_eligible']]
    valid=[s for s in structured if s['shape_compatible']]
    tp=sum(s['field_true_positive'] for s in valid);ng=sum(s['field_gold'] for s in valid);np=sum(s['field_predicted'] for s in valid)
    return dict(cases=len(scores),structured_cases=len(structured),free_text_cases=len(scores)-len(structured),
        structured_exact_percent=100*sum(s['structured_exact'] for s in structured)/len(structured) if structured else None,
        parse_success_count=sum(s['parse_success'] for s in structured),shape_compatible_count=len(valid),
        format_or_shape_failure_count=len(structured)-len(valid),
        valid_structure_field_macro_precision=sum(s['field_precision'] for s in valid)/len(valid) if valid else None,
        valid_structure_field_macro_recall=sum(s['field_recall'] for s in valid)/len(valid) if valid else None,
        valid_structure_field_macro_f1=sum(s['field_f1'] for s in valid)/len(valid) if valid else None,
        valid_structure_field_micro_precision=tp/np if np else (float(not ng) if valid else None),
        valid_structure_field_micro_recall=tp/ng if ng else (float(not np) if valid else None),
        valid_structure_field_micro_f1=2*tp/(ng+np) if ng+np else (1. if valid else None),
        field_metrics_condition_on_compatible_parse=True,
        normalized_text_exact_percent=100*sum(s['normalized_text_exact'] for s in scores)/len(scores),
        normalized_text_token_f1_percent=100*sum(s['normalized_text_token_f1'] for s in scores)/len(scores),
        official_loong_score=False)
