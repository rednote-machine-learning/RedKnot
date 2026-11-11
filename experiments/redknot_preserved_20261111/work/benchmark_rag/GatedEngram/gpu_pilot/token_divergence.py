"""CPU analysis of two recorded free-running greedy output trajectories.

Token IDs, including EOS, are authoritative. Independently decoded token pieces
are labels only and are never concatenated into purported answer text. A first
difference locates a trajectory divergence, not the first factual error or a
causal attribution to any layer, cache component, or subsequent token.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string

SCHEMA = 's1-free-running-token-divergence-v1'
TOKENIZER_SHA = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
METRICS_URL = 'https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py'
EVAL_URL = 'https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py'


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seal(value):
    body={k:v for k,v in value.items() if k != 'sha256'}
    return dict(body,sha256=digest(body))


def official_longbench_f1(text, references):
    """LongBench English QA normalization, best reference, no first-line cut.

    Deliberately separate from the legacy benchmark EM/F1: LongBench removes
    ASCII string.punctuation and returns zero when normalized overlap is empty.
    """
    require(type(text) is str and type(references) is list and references
            and all(type(s) is str for s in references),'official F1 needs full text and references')
    def words(value):
        value=value.lower().translate(str.maketrans('','',string.punctuation))
        return re.sub(r'\b(a|an|the)\b',' ',value).split()
    predicted=words(text);scores=[]
    for reference in references:
        gold=words(reference);overlap=sum((Counter(predicted)&Counter(gold)).values())
        scores.append(2.0*overlap/(len(predicted)+len(gold)) if overlap else 0.0)
    return max(scores)


def metric_policy(dataset):
    require(dataset in ('musique','qasper','hotpotqa','2wikimqa','multifieldqa_en'),'unsupported quality dataset')
    return dict(dataset=dataset,primary_metrics=['legacy_answer_em','legacy_answer_token_f1'] if dataset!='qasper'
                else ['official_longbench_f1'],legacy_scores_preserved=True,
        official_f1_sources=[METRICS_URL,EVAL_URL],official_f1_algorithm='ascii_punctuation_article_whitespace_normalization_max_reference_v1',
        prediction_preprocessing='entire original decoded output; no first-line clipping',
        official_f1_empty_overlap=0.0,official_f1_percent_rounding_digits=2,
        longbench_official_em_claim=False)


class TokenizerDecoder:
    def __init__(self, path, expected_sha):
        path=Path(path)
        raw=path.read_bytes();observed=hashlib.sha256(raw).hexdigest()
        require(observed==expected_sha==TOKENIZER_SHA,'actual checkpoint tokenizer SHA differs')
        obj=json.loads(raw)
        from tokenizers import Tokenizer, __version__
        self.backend=Tokenizer.from_str(raw.decode('utf-8'))
        self.eos_token_id=1
        require(self.backend.id_to_token(1)=='<｜end▁of▁sentence｜>','actual tokenizer EOS binding differs')
        self.special_ids=frozenset(token['id'] for token in obj.get('added_tokens',[]) if token.get('special') is True)
        require(1 in self.special_ids,'EOS must be a special token')
        self.identity=dict(schema='s1-actual-output-tokenizer-v1',path=str(path),tokenizer_json_sha256=observed,
            tokenizers_version=__version__,eos_token_id=1,special_token_ids=sorted(self.special_ids),
            decoder_sha256=digest(obj.get('decoder')),whole_sequence_decode=True,
            answer_skip_special_tokens=True,answer_cleanup_tokenization_spaces=False,
            standalone_token_pieces_are_not_joined=True)

    def decode(self, ids, *, skip_special_tokens):
        require(all(self.backend.id_to_token(token) is not None for token in ids),'output ID absent from actual tokenizer')
        return self.backend.decode(ids,skip_special_tokens=skip_special_tokens)

    def piece(self, token):
        value=self.backend.id_to_token(token)
        require(value is not None,'output ID absent from actual tokenizer')
        return value


def decoder_for_identity(identity):
    return TokenizerDecoder(Path(identity['checkpoint_path'])/'tokenizer.json',identity['tokenizer_json_sha256'])


def audit_decoded_record(record, decoder):
    ids=record.get('output_ids')
    require(type(ids) is list and ids and all(type(i) is int and i>=0 for i in ids)
        and record.get('output_ids_sha256')==digest(ids),'output IDs/hash invalid')
    require(type(record.get('text')) is str and decoder.decode(ids,skip_special_tokens=True)==record['text'],
            'whole output decode differs from actual raw quality text')
    eos=decoder.eos_token_id;where=[i for i,token in enumerate(ids) if token==eos]
    finish=record.get('finish_reason')
    require(record.get('natural_EOS') is True and finish in ('stop','length'),'natural quality termination required')
    require(where==[len(ids)-1] if finish=='stop' else where==[],
            'native EOS included/terminal position differs from finish reason')
    if finish=='length':
        require(record.get('truncated') is True and record.get('quality_limit_reason')=='engine_context_limit'
                and len(ids)==record.get('technical_max_new_tokens'), 'length stop is not the actual engine budget')
    else:
        require(record.get('truncated') is False and record.get('quality_output_complete') is True,
                'natural EOS output must be complete')
    content=sum(token not in decoder.special_ids for token in ids)
    return dict(raw_generated_tokens=len(ids),terminal_eos_included=finish=='stop',
        generated_tokens_excluding_terminal_eos=len(ids)-(finish=='stop'),non_special_generated_tokens=content,
        non_special_at_least50=content>=50,non_special_between50_and200=50<=content<=200,
        finish_reason=finish,length_bound=finish=='length',whole_output_text_verified=True)


def compare_quality_records(full, candidate, decoder, *, context_tokens=12):
    require(type(context_tokens) is int and context_tokens>0,'positive context width required')
    records={'recomputed':full,'s1':candidate}
    lengths={arm:audit_decoded_record(record,decoder) for arm,record in records.items()}
    left,right=full['output_ids'],candidate['output_ids'];common=0
    while common<min(len(left),len(right)) and left[common]==right[common]:common+=1
    identical=left==right;both_have=common<min(len(left),len(right))
    kind='identical' if identical else ('eos_vs_token' if both_have and decoder.eos_token_id in (left[common],right[common])
        else 'token_substitution' if both_have else 'length_bound_prefix_end')
    if not identical and not both_have:
        shorter='recomputed' if len(left)<len(right) else 's1'
        require(records[shorter]['finish_reason']=='length','raw strict prefix ending without EOS must be length-bound')
    else:shorter=None
    detail={}
    for arm,record in records.items():
        ids=record['output_ids'];token=ids[common] if common<len(ids) else None
        lo=max(0,common-context_tokens);hi=min(len(ids),common+context_tokens+1)
        detail[arm]=dict(token_id=token,token_kind='sequence_end' if token is None else 'eos' if token==decoder.eos_token_id else 'token',
            vocab_piece=None if token is None else decoder.piece(token),
            standalone_decode=None if token is None else decoder.decode([token],skip_special_tokens=False),
            standalone_decode_is_label_only=True,context_start_1based=lo+1,context_end_exclusive_1based=hi+1,
            context_token_ids=ids[lo:hi],context_text=decoder.decode(ids[lo:hi],skip_special_tokens=False),
            context_may_start_mid_byte_sequence=True,
            full_prefix_through_position_text=decoder.decode(ids[:min(len(ids),common+1)],skip_special_tokens=False))
    first=None if identical else common+1
    return seal(dict(schema=SCHEMA,comparison='free_running_greedy_trajectories',
        context_window_tokens=context_tokens,
        tokenizer=decoder.identity,output_ids_sha256={a:r['output_ids_sha256'] for a,r in records.items()},
        identical_output_ids=identical,identical_display_text=full['text']==candidate['text'],
        common_prefix_tokens_including_special=common,
        common_prefix_text=decoder.decode(left[:common],skip_special_tokens=False),
        first_difference_position_1based=first,
        first_different_generated_token_1based=first if both_have else None,
        difference_kind=kind,length_bound_prefix_ended_arm=shorter,
        first_difference=None if identical else detail,output_lengths=lengths,
        both_outputs_at_least50_non_special_tokens=all(x['non_special_at_least50'] for x in lengths.values()),
        eos_included_in_id_comparison=True,output_ids_preserved_in_parent_quality=True,
        first_difference_is_first_factual_error_claim=False,teacher_forced_comparison=False,
        causal_attribution_claim=False,post_divergence_conditioning_differs=not identical,
        limitation='First token difference locates a greedy trajectory divergence; it does not identify a factual error or causal model component. Later predictions condition on their own generated histories.'))
