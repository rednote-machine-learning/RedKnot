"""Resume twelve exact MuSiQue inputs and add six native longer-answer QASPER QAs.

CPU input preparation only. Original full contexts, questions, gold answers and
official prompts remain unchanged. Gold lengths select a cohort before inference;
they never impose a minimum generated length or permit selection by model results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from gpu_pilot import musique30_inputs as previous

MANIFEST_SCHEMA = 'di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA = 'kv-cache-engram-musique-longform-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA = 'musique-longform-input-protocol-v1'
INPUT_COMPANIONS = previous.INPUT_COMPANIONS
PREVIOUS_FILES = ('manifest.json', 'input-protocol.json', *INPUT_COMPANIONS)
PREVIOUS_MANIFEST_SHA256 = '177d0c621e7206a9ce6b693c41cd94b7335e68b4f00f5320274e4c422838a253'
PREVIOUS_PROTOCOL_SHA256 = 'cfa52f6866f93920797a9507c97d7158eb2b2a6bdd1a79dc58d7d03c1f5acbe6'
QASPER_SOURCE_SHA256 = '29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854'
OFFICIAL_ENCODING_SHA256 = previous.OFFICIAL_ENCODING_SHA256
OLD_MUSIQUE_INDICES = tuple(range(20, 32))
EVALUATION_INDICES = tuple(range(2, 20))
SEED = 20261107
NUM_CHUNKS = 10
MIN_BLOCK_TOKENS = 256
GOLD_MIN_TOKENS = 50
GOLD_MAX_TOKENS = 200
GOLD_PREFERRED_MIN_TOKENS = 80
require, digest, text_hash = previous.require, previous.digest, previous.text_hash
file_hash, seal, verify, read, publish = previous.file_hash, previous.seal, previous.verify, previous.read, previous.publish


def qasper_boundaries(context):
    """Only original newline boundaries, including whitespace without editing it."""
    require(type(context) is str and context, 'nonempty native QASPER context required')
    points = [0] + [m.end() for m in re.finditer('\n', context) if m.end() < len(context)] + [len(context)]
    require(len(points) >= 11, 'at least ten original QASPER newline segments required')
    return points


def partition_ten(boundaries, offsets, context_start):
    """Minimize squared token imbalance; short/empty atomic lines can be merged."""
    require(boundaries[0] == 0 and boundaries == sorted(set(boundaries)), 'invalid natural boundaries')
    arrays = previous.offset_arrays(offsets)
    spans = {}
    for begin in range(len(boundaries) - 1):
        for end in range(begin + 1, len(boundaries)):
            try:
                left, right = previous.exact_token_span(offsets, context_start + boundaries[begin],
                                                       context_start + boundaries[end], arrays=arrays)
            except ValueError:
                continue
            if right - left >= MIN_BLOCK_TOKENS:
                spans[begin, end] = (left, right)
    first, last = previous.exact_token_span(offsets, context_start, context_start + boundaries[-1], arrays=arrays)
    total = last - first
    dp = {(0, 0): (0, ())}
    for count in range(1, 11):
        for end in range(count, len(boundaries)):
            choices = []
            for begin in range(count - 1, end):
                if (count - 1, begin) in dp and (begin, end) in spans:
                    cost, cuts = dp[count - 1, begin]
                    left, right = spans[begin, end]
                    choices.append((cost + (10 * (right - left) - total) ** 2, cuts + (end,)))
            if choices:
                dp[count, end] = min(choices)
    require((10, len(boundaries) - 1) in dp, 'no ten complete natural blocks with minimum256 tokens')
    answer, begin = [], 0
    for end in dp[10, len(boundaries) - 1][1]:
        left, right = spans[begin, end]
        answer.append(dict(char_start=boundaries[begin], char_end=boundaries[end],
                           input_start=left, input_end=right, passage_start=begin, passage_end=end))
        begin = end
    return answer


def original_gold_encoding(raw, tokenizer):
    answers = raw.get('answers')
    require(type(answers) is list and answers and all(type(a) is str and a for a in answers),
            'nonempty original QASPER references required')
    ids = [tokenizer.encode(answer, add_special_tokens=False).ids for answer in answers]
    require(all(tokenizer.decode(tokens, skip_special_tokens=False) == answer for tokens, answer in zip(ids, answers)),
            'original gold tokenizer roundtrip differs')
    return dict(reference_token_ids=ids, reference_token_ids_sha256=digest(ids),
                reference_token_counts=[len(tokens) for tokens in ids], references_sha256=digest(answers))


def gold_is_eligible(gold):
    return all(GOLD_MIN_TOKENS <= n <= GOLD_MAX_TOKENS for n in gold['reference_token_counts'])


def build_qasper_row(raw, source_index, tokenizer, encode_messages, template, gold=None):
    require(raw.get('dataset') == 'qasper', 'native QASPER source required')
    gold = original_gold_encoding(raw, tokenizer) if gold is None else gold
    require(gold_is_eligible(gold), 'all original gold answers must have actual50..200 tokenizer tokens')
    context, question = raw['context'], raw['input']
    require(type(question) is str and question, 'nonempty original question required')
    rendered = encode_messages([dict(role='user', content=template.format(context=context, input=question))], thinking_mode='chat')
    require(rendered.count(context) == rendered.count(question) == 1, 'original context/question must appear exactly once')
    encoding = tokenizer.encode(rendered, add_special_tokens=False)
    ids, offsets = encoding.ids, encoding.offsets
    require(tokenizer.decode(ids, skip_special_tokens=False) == rendered, 'whole original prompt decode differs')
    require(tokenizer.encode(tokenizer.decode(ids, skip_special_tokens=False), add_special_tokens=False).ids == ids,
            'whole original prompt reencoding differs')
    start = rendered.index(context)
    boundaries = qasper_boundaries(context)
    groups = partition_ten(boundaries, offsets, start)
    documents = []
    for i, group in enumerate(groups):
        left, right = group['input_start'], group['input_end']
        sha = digest(ids[left:right])
        documents.append(dict(document_index=i, document_id=sha, token_ids_sha256=sha, tokens=right-left,
            input_start=left, input_end=right, offline_position=0, online_position=left,
            context_char_start=group['char_start'], context_char_end=group['char_end'],
            source_text_sha256=text_hash(context[group['char_start']:group['char_end']]),
            passage_start=group['passage_start'], passage_end=group['passage_end']))
    require(len({d['document_id'] for d in documents}) == 10, 'duplicate original QASPER cache block')
    prefix, suffix = documents[0]['input_start'], len(ids)-documents[-1]['input_end']
    row = dict(sample_id='qasper:'+raw['_id'], dataset='qasper', source_index=source_index, source_line=source_index+1,
        source_id=raw['_id'], source_row_sha256=digest(raw), question=question,
        references=raw['answers'], references_sha256=digest(raw['answers']), input_ids=ids,
        input_ids_sha256=digest(ids), input_tokens=len(ids), prompt=rendered, prompt_sha256=text_hash(rendered),
        context_text=context, context_sha256=text_hash(context), context_prompt_char_start=start,
        context_prompt_char_end=start+len(context), original_context_tokens=len(tokenizer.encode(context, add_special_tokens=False).ids),
        context_tokens=len(ids)-prefix-suffix, document_tokens_total=sum(d['tokens'] for d in documents),
        prefix_tokens=prefix, suffix_tokens=suffix, rendered_prefix=rendered[:start], rendered_suffix=rendered[start+len(context):],
        primary_context_truncated=False, natural_boundary_inputs=True, source_context_count=1,
        source_order_preserved=True, text_roundtrip_matches_ids=True,
        prompt_transport='whole rendered prompt tokenized once; exact token IDs authoritative',
        block_boundary_kind='original_newline_segment_start', documents=documents,
        input_non_document_tokens=len(ids)-sum(d['tokens'] for d in documents),
        support_retention='entire single original QA context unchanged; original question and references unchanged',
        longform_gold_token_counts=gold['reference_token_counts'])
    evidence = dict(sample_id=row['sample_id'], prompt_sha256=row['prompt_sha256'], input_ids_sha256=row['input_ids_sha256'],
        whole_prompt_single_encoding=True, decode_matches_rendered_prompt=True, reencode_matches_input_ids=True,
        token_offsets=[list(pair) for pair in offsets], original_passage_char_boundaries=boundaries,
        boundary_kind='original_newline_segment_start', full_source_context_preserved=True,
        full_source_question_preserved=True, additional_source_contexts=0, gold_encoding=gold)
    return row, evidence


def candidate_metadata(row, gold):
    return dict(source_index=row['source_index'], source_id=row['source_id'], sample_id=row['sample_id'],
        source_row_sha256=row['source_row_sha256'], context_sha256=row['context_sha256'],
        question_sha256=text_hash(row['question']), input_tokens=row['input_tokens'], input_ids_sha256=row['input_ids_sha256'],
        gold_encoding=gold, tie_sha256=digest(dict(seed=SEED, source_id=row['source_id'], context_sha256=row['context_sha256'])))


def selection_key(item):
    lengths = item['gold_encoding']['reference_token_counts']
    return (min(lengths) < GOLD_PREFERRED_MIN_TOKENS, -min(lengths), item['tie_sha256'])


def choose_candidates(candidates, excluded_contexts=(), excluded_questions=()):
    contexts, questions, chosen = set(excluded_contexts), set(excluded_questions), []
    for candidate in sorted(candidates, key=selection_key):
        if candidate['context_sha256'] in contexts or candidate['question_sha256'] in questions:
            continue
        chosen.append(candidate)
        contexts.add(candidate['context_sha256']); questions.add(candidate['question_sha256'])
        if len(chosen) == 6:
            break
    require(len(chosen) == 6, 'fewer than six distinct eligible native longer-answer QASPER contexts')
    return chosen


def validate_previous_bundle(bundle):
    """Anchor prior inputs to their already audited immutable full-manifest seal."""
    require(set(bundle) == set(PREVIOUS_FILES), 'previous exact input bundle incomplete')
    for value in bundle.values():
        verify(value)
    manifest, protocol = bundle['manifest.json'], bundle['input-protocol.json']
    require(manifest['sha256'] == PREVIOUS_MANIFEST_SHA256 and protocol['sha256'] == PREVIOUS_PROTOCOL_SHA256
            and manifest['input_protocol_sha256'] == protocol['sha256'], 'previous MuSiQue input identity differs')
    require(manifest['companion_sha256'] == {name:bundle[name]['sha256'] for name in INPUT_COMPANIONS},
            'previous input companion binding differs')
    for key, name in (('source_records_sha256','source-records.json'), ('selection_plan_sha256','selection-plan.json'),
                      ('tokenization_evidence_sha256','tokenization-evidence.json')):
        require(protocol[key] == bundle[name]['sha256'], 'previous protocol provenance differs')
    require([r['index'] for r in manifest['warmup']+manifest['cohort']] == list(range(32))
            and all(r['dataset']=='musique' and len(r['documents'])==10 for r in manifest['warmup']+manifest['cohort']),
            'expected exact32 natural10 MuSiQue previous inputs')
    require(manifest['warmup'][0]['input_tokens'] >= 16384, 'previous native16K calibration absent')
    return manifest


def copied_musique_row(original, index):
    return dict(copy.deepcopy(original), index=index, split='evaluation', original_mu_case_index=original['index'],
                original_mu_row_sha256=digest(original), original_mu_manifest_sha256=PREVIOUS_MANIFEST_SHA256)


def build_inputs(previous_bundle, qasper_source_bytes, tokenizer, encode_messages, *, tokenizer_identity,
                 official_encoding_source, official_prompt_file_content, official_prompt_source):
    prior = validate_previous_bundle(previous_bundle)
    require(prior['tokenizer'] == tokenizer_identity, 'checkpoint tokenizer/prompt identity differs from previous MuSiQue')
    require(hashlib.sha256(qasper_source_bytes).hexdigest() == QASPER_SOURCE_SHA256, 'QASPER acquisition bytes changed')
    require(text_hash(official_encoding_source) == OFFICIAL_ENCODING_SHA256, 'official chat renderer changed')
    require(text_hash(official_prompt_file_content) == official_prompt_source['sha256']
            == tokenizer_identity['official_longbench_prompt_sha256'], 'official LongBench prompt file differs')
    templates = json.loads(official_prompt_file_content)
    raw_rows = [json.loads(line) for line in qasper_source_bytes.splitlines() if line.strip()]
    require(len(raw_rows)==200 and len({r['_id'] for r in raw_rows})==200, 'pinned QASPER200 acquisition required')
    original_lookup = {r['index']:r for r in prior['cohort']}
    rows = copy.deepcopy(prior['warmup']) + [copied_musique_row(original_lookup[i], new) for new,i in enumerate(OLD_MUSIQUE_INDICES,2)]
    selected_ids = {r['sample_id'] for r in rows}
    sources = [copy.deepcopy(r) for r in previous_bundle['source-records.json']['records'] if r['sample_id'] in selected_ids]
    evidence = [copy.deepcopy(r) for r in previous_bundle['tokenization-evidence.json']['records'] if r['sample_id'] in selected_ids]
    eligible, rejected, lookup = [], [], {}
    for index, raw in enumerate(raw_rows):
        gold = original_gold_encoding(raw, tokenizer)
        attribution = dict(source_index=index, source_id=raw['_id'], source_row_sha256=digest(raw), gold_encoding=gold)
        if not gold_is_eligible(gold):
            rejected.append(dict(attribution, reason='original_gold_length_outside50..200'))
            continue
        try:
            row, ev = build_qasper_row(raw,index,tokenizer,encode_messages,templates['qasper'],gold)
            require(row['input_tokens'] < 270336, 'original prompt exceeds engine context capacity')
        except ValueError as exc:
            rejected.append(dict(attribution, reason='native_geometry: '+str(exc)))
            continue
        meta = candidate_metadata(row,gold); eligible.append(meta); lookup[meta['sample_id']] = (row,ev,raw)
    chosen = choose_candidates(eligible, (r['context_sha256'] for r in rows), (text_hash(r['question']) for r in rows))
    for index,item in enumerate(chosen,14):
        row, ev, raw = lookup[item['sample_id']]
        row = dict(row,index=index,split='evaluation'); rows.append(row); evidence.append(ev)
        sources.append(dict(sample_id=row['sample_id'],dataset='qasper',source_index=row['source_index'],
                            source_row_sha256=digest(raw),raw=raw))
    source = seal(dict(schema='musique-longform-source-records-v1', previous_inputs=previous_bundle,
        original_qasper_source_jsonl=qasper_source_bytes.decode('utf-8'), records=sources,
        files=prior['sources']+[dict(dataset='qasper',acquisition_sha256=QASPER_SOURCE_SHA256,bytes=len(qasper_source_bytes),records=200)]))
    selection = seal(dict(schema='musique-longform-selection-plan-v1', seed=SEED,
        no_model_outcomes_used=True, no_answer_semantic_selection=True, gold_length_selection=True,
        qasper_gold_tokens_min=50,qasper_gold_tokens_max=200,qasper_preferred_min_gold_tokens=80,
        qasper_ranking='all gold>=80 preferred; descending minimum gold token length; seeded source SHA tie-break',
        eligible_candidates=sorted(eligible,key=selection_key),rejected=rejected,
        previous_manifest_sha256=PREVIOUS_MANIFEST_SHA256, previous_mu_case_indices=list(OLD_MUSIQUE_INDICES),
        mu_index_mapping=[dict(index=new,original_mu_case_index=old) for new,old in enumerate(OLD_MUSIQUE_INDICES,2)],
        calibration_source='previous_musique_calibration_exact', calibration_indices=[0,1],
        evaluation_indices=list(EVALUATION_INDICES), selected_sample_ids=[r['sample_id'] for r in rows],
        generated_length_selection=False, minimum_generated_tokens=None, independent_holdout_claim=False,
        block_partition='minimum squared token-length imbalance at original newline boundaries; lexical cut tie-break'))
    tokens = seal(dict(schema='musique-longform-tokenization-evidence-v1',tokenizer=tokenizer_identity,
        official_prompt_source=official_prompt_source,official_prompt_file_content=official_prompt_file_content,
        official_encoding_source=official_encoding_source,official_prompt_templates={d:templates[d] for d in ('musique','qasper')},
        records=evidence))
    protocol = seal(dict(schema=INPUT_PROTOCOL_SCHEMA,seed=SEED,datasets=['musique','qasper'],
        composition='twelve_exact_previous_musique_plus_six_original_qasper_ten_natural_blocks',
        calibration_indices=[0,1],evaluation_indices=list(EVALUATION_INDICES),musique_indices=list(range(2,14)),
        qasper_indices=list(range(14,20)),previous_manifest_sha256=PREVIOUS_MANIFEST_SHA256,
        calibration_source='previous_musique_calibration_exact',num_chunks=10,min_block_tokens=256,
        equal_document_lengths_required=False,no_padding=True,no_truncation=True,no_cross_sample_contexts=True,
        preserve_original_order=True,single_prompt_tokenization=True,no_model_outcomes_used=True,
        no_answer_semantic_selection=True,gold_length_selection=True,generated_length_selection=False,
        qasper_gold_tokens_min=50,qasper_gold_tokens_max=200,qasper_preferred_min_gold_tokens=80,
        minimum_generated_tokens=None,natural_eos_preserved=True,tokenizer=tokenizer_identity,
        tokenizers_version='0.21.4',inference_seed=0,source_records_sha256=source['sha256'],
        selection_plan_sha256=selection['sha256'],tokenization_evidence_sha256=tokens['sha256']))
    companions = dict(zip(INPUT_COMPANIONS,(source,selection,tokens)))
    manifest = seal(dict(schema=MANIFEST_SCHEMA,contract_schema=CONTRACT_SCHEMA,artifact_kind='musique_longform_inputs_v1',
        natural_boundary_inputs=True,input_protocol_sha256=protocol['sha256'],tokenizer=tokenizer_identity,
        num_chunks=10,chunk_tokens=None,context_tokens=None,calibration_source='previous_musique_calibration_exact',
        sources=source['files'],warmup=rows[:2],cohort=rows[2:],companion_sha256={k:v['sha256'] for k,v in companions.items()},
        selection_plan_sha256=selection['sha256']))
    summary = seal(dict(schema='musique-longform-input-preparation-v1',status='complete',cuda_used=False,
        model_results_used=False,manifest_sha256=manifest['sha256'],input_protocol_sha256=protocol['sha256'],
        qasper_source_sha256=QASPER_SOURCE_SHA256,previous_manifest_sha256=PREVIOUS_MANIFEST_SHA256,
        qasper_eligible_count=len(eligible),evaluation_cases=18,musique_cases=12,qasper_cases=6,
        samples=[dict(index=r['index'],dataset=r['dataset'],source_index=r['source_index'],
            original_mu_case_index=r.get('original_mu_case_index'),input_tokens=r['input_tokens'],
            block_tokens=[d['tokens'] for d in r['documents']],gold_tokens=r.get('longform_gold_token_counts'),
            source_context_preserved=True,whole_prompt_roundtrip=True) for r in rows]))
    return dict(companions,**{'manifest.json':manifest,'input-protocol.json':protocol,'preparation.json':summary})


def audit_gold(gold, raw):
    ids = gold['reference_token_ids']
    require(type(ids) is list and len(ids)==len(raw['answers']) and all(type(v) is list and v for v in ids)
            and all(type(t) is int and 0<=t<2**31 for tokens in ids for t in tokens)
            and gold['reference_token_ids_sha256']==digest(ids)
            and gold['reference_token_counts']==[len(tokens) for tokens in ids]
            and gold['references_sha256']==digest(raw['answers']), 'original gold token evidence differs')


def validate_inputs(base, manifest, protocol, *, input_dir):
    """CPU-only independent source/copy/provenance/geometry audit; no Torch import."""
    verify(manifest); verify(protocol)
    require(manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('contract_schema')==CONTRACT_SCHEMA
            and manifest.get('artifact_kind')=='musique_longform_inputs_v1' and protocol.get('schema')==INPUT_PROTOCOL_SCHEMA
            and manifest.get('input_protocol_sha256')==protocol['sha256'], 'resume/longform input schema differs')
    expected = dict(seed=SEED,datasets=['musique','qasper'],calibration_indices=[0,1],evaluation_indices=list(EVALUATION_INDICES),
        musique_indices=list(range(2,14)),qasper_indices=list(range(14,20)),num_chunks=10,min_block_tokens=256,
        composition='twelve_exact_previous_musique_plus_six_original_qasper_ten_natural_blocks',
        previous_manifest_sha256=PREVIOUS_MANIFEST_SHA256,calibration_source='previous_musique_calibration_exact',
        equal_document_lengths_required=False,no_padding=True,no_truncation=True,no_cross_sample_contexts=True,
        preserve_original_order=True,single_prompt_tokenization=True,no_model_outcomes_used=True,
        no_answer_semantic_selection=True,gold_length_selection=True,generated_length_selection=False,
        qasper_gold_tokens_min=50,qasper_gold_tokens_max=200,qasper_preferred_min_gold_tokens=80,
        minimum_generated_tokens=None,natural_eos_preserved=True,tokenizers_version='0.21.4',inference_seed=0)
    require(all(k in protocol and protocol[k]==v for k,v in expected.items()), 'resume/longform fixed protocol differs')
    companions = {name:read(Path(input_dir)/name) for name in INPUT_COMPANIONS}
    require(set(manifest['companion_sha256'])==set(INPUT_COMPANIONS), 'input companion inventory differs')
    for name,value in companions.items():
        verify(value); require(manifest['companion_sha256'][name]==value['sha256'], 'input companion binding differs')
    source,selection,tokens = [companions[name] for name in INPUT_COMPANIONS]
    for key,value in (('source_records_sha256',source),('selection_plan_sha256',selection),('tokenization_evidence_sha256',tokens)):
        require(protocol[key]==value['sha256'], 'input provenance chain differs')
    require(source.get('schema')=='musique-longform-source-records-v1'
            and selection.get('schema')=='musique-longform-selection-plan-v1'
            and tokens.get('schema')=='musique-longform-tokenization-evidence-v1', 'companion schema differs')
    prior = validate_previous_bundle(source['previous_inputs'])
    require(manifest['tokenizer']==protocol['tokenizer']==tokens['tokenizer']==prior['tokenizer']
            and tokens['tokenizer']['encoding_sha256']==OFFICIAL_ENCODING_SHA256
            and tokens['tokenizer']['tokenizers_version']=='0.21.4', 'actual tokenizer identity differs')
    require(text_hash(tokens['official_prompt_file_content'])==tokens['official_prompt_source']['sha256']
            ==manifest['tokenizer']['official_longbench_prompt_sha256'], 'official prompt identity differs')
    templates = json.loads(tokens['official_prompt_file_content'])
    require(tokens['official_prompt_templates']=={d:templates[d] for d in ('musique','qasper')}, 'official task prompt changed')
    render = previous.frozen_renderer(tokens['official_encoding_source'])
    raw_bytes = source['original_qasper_source_jsonl'].encode('utf-8')
    require(hashlib.sha256(raw_bytes).hexdigest()==QASPER_SOURCE_SHA256, 'original QASPER acquisition changed')
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    require(len(raw_rows)==200 and len({r['_id'] for r in raw_rows})==200 and source['files']==prior['sources']+
            [dict(dataset='qasper',acquisition_sha256=QASPER_SOURCE_SHA256,bytes=len(raw_bytes),records=200)]
            and manifest['sources']==source['files'], 'original source file inventory differs')
    rows = manifest['warmup']+manifest['cohort']
    require(manifest['warmup']==prior['warmup'] and len(manifest['cohort'])==18
            and [r['index'] for r in rows]==list(range(20)) and manifest['num_chunks']==10
            and manifest['natural_boundary_inputs'] is True
            and manifest['calibration_source']==protocol['calibration_source']
            and manifest['selection_plan_sha256']==selection['sha256'], 'exact previous calibration2/new evaluation18 required')
    old_rows = {r['index']:r for r in prior['cohort']}
    for new,old in enumerate(OLD_MUSIQUE_INDICES,2):
        require(rows[new]==copied_musique_row(old_rows[old],new), 'resumed MuSiQue input changed beyond index/provenance')
    require(all(r['dataset']=='qasper' and r['split']=='evaluation' for r in rows[14:]), 'six QASPER evaluation cases required')
    require(len({r['sample_id'] for r in rows})==len({r['context_sha256'] for r in rows})
            ==len({text_hash(r['question']) for r in rows})==20, 'duplicate source context/question')
    for key in ('seed','no_model_outcomes_used','no_answer_semantic_selection','gold_length_selection',
                'generated_length_selection','qasper_gold_tokens_min','qasper_gold_tokens_max',
                'qasper_preferred_min_gold_tokens','minimum_generated_tokens','calibration_source',
                'calibration_indices','evaluation_indices','previous_manifest_sha256'):
        require(selection.get(key)==protocol[key], 'selection policy differs: '+key)
    require(selection['previous_mu_case_indices']==list(OLD_MUSIQUE_INDICES)
            and selection['mu_index_mapping']==[dict(index=new,original_mu_case_index=old) for new,old in enumerate(OLD_MUSIQUE_INDICES,2)]
            and selection['selected_sample_ids']==[r['sample_id'] for r in rows], 'source selection mapping differs')
    candidates,rejected = selection['eligible_candidates'],selection['rejected']
    require(sorted(v['source_index'] for v in candidates+rejected)==list(range(200)), 'every original QASPER source must appear once')
    for item in candidates+rejected:
        raw=raw_rows[item['source_index']]
        require(item['source_id']==raw['_id'] and item['source_row_sha256']==digest(raw), 'source selection attribution differs')
        audit_gold(item['gold_encoding'],raw)
    for item in candidates:
        raw=raw_rows[item['source_index']]
        require(gold_is_eligible(item['gold_encoding']) and type(item['input_tokens']) is int and 0<item['input_tokens']<270336
                and item['sample_id']=='qasper:'+raw['_id'] and item['context_sha256']==text_hash(raw['context'])
                and item['question_sha256']==text_hash(raw['input'])
                and item['tie_sha256']==digest(dict(seed=SEED,source_id=raw['_id'],context_sha256=text_hash(raw['context']))),
                'candidate ranking fields differ')
    for item in rejected:
        require((item['reason']=='original_gold_length_outside50..200' and not gold_is_eligible(item['gold_encoding']))
                or (item['reason'].startswith('native_geometry: ') and gold_is_eligible(item['gold_encoding'])),
                'source exclusion reason differs from gold evidence')
    require(candidates==sorted(candidates,key=selection_key), 'candidate ranking order differs')
    chosen=choose_candidates(candidates,(r['context_sha256'] for r in rows[:14]),(text_hash(r['question']) for r in rows[:14]))
    require([r['sample_id'] for r in chosen]==[r['sample_id'] for r in rows[14:]], 'selected QASPER cohort differs from fixed ranking')
    records={r['sample_id']:r for r in source['records']}; evidence={r['sample_id']:r for r in tokens['records']}
    require(len(records)==len(source['records'])==len(evidence)==len(tokens['records'])==20
            and set(records)==set(evidence)=={r['sample_id'] for r in rows}, 'source/token evidence inventory differs')
    old_source={r['sample_id']:r for r in source['previous_inputs']['source-records.json']['records']}
    old_tokens={r['sample_id']:r for r in source['previous_inputs']['tokenization-evidence.json']['records']}
    for row in rows[:14]:
        require(records[row['sample_id']]==old_source[row['sample_id']] and evidence[row['sample_id']]==old_tokens[row['sample_id']],
                'resumed MuSiQue raw source/token evidence changed')
    if base is not None:
        base.validate_manifest(manifest)
    for row in rows[14:]:
        rec,ev=records[row['sample_id']],evidence[row['sample_id']]
        raw=raw_rows[row['source_index']]; ids=row['input_ids']; context=raw['context']
        require(rec==dict(sample_id=row['sample_id'],dataset='qasper',source_index=row['source_index'],source_row_sha256=digest(raw),raw=raw)
                and row['source_row_sha256']==digest(raw) and row['source_id']==raw['_id']
                and row['source_line']==row['source_index']+1
                and row['sample_id']=='qasper:'+raw['_id'] and row['question']==raw['input']
                and row['references']==raw['answers'] and row['references_sha256']==digest(raw['answers'])
                and row['context_text']==context and row['context_sha256']==text_hash(context), 'original QASPER QA changed')
        require(row['input_tokens']==len(ids) and row['input_ids_sha256']==digest(ids)
                and all(type(t) is int and 0<=t<2**31 for t in ids), 'whole prompt token identity differs')
        expected_prompt=render([dict(role='user',content=templates['qasper'].format(context=context,input=raw['input']))],thinking_mode='chat')
        prompt=row['prompt']; start=row['context_prompt_char_start']; end=row['context_prompt_char_end']
        require(prompt==expected_prompt and prompt[start:end]==context and end==start+len(context)
                and prompt.count(context)==prompt.count(raw['input'])==1
                and row['prompt_sha256']==ev['prompt_sha256']==text_hash(prompt)
                and row['input_ids_sha256']==ev['input_ids_sha256'], 'whole original QASPER prompt differs')
        require(row['natural_boundary_inputs'] is True and row['primary_context_truncated'] is False
                and row['source_context_count']==1 and row['source_order_preserved'] is True
                and row['text_roundtrip_matches_ids'] is True
                and row['block_boundary_kind']==ev['boundary_kind']=='original_newline_segment_start', 'source transformation detected')
        for flag in ('whole_prompt_single_encoding','decode_matches_rendered_prompt','reencode_matches_input_ids',
                     'full_source_context_preserved','full_source_question_preserved'):
            require(ev[flag] is True, 'actual full prompt tokenization not proven')
        require(ev['additional_source_contexts']==0 and len(ev['token_offsets'])==len(ids), 'offset/foreign context mismatch')
        audit_gold(ev['gold_encoding'],raw)
        require(row['longform_gold_token_counts']==ev['gold_encoding']['reference_token_counts']
                and candidate_metadata(row,ev['gold_encoding']) in chosen, 'selected gold token evidence detached from ranking')
        boundaries=qasper_boundaries(context); groups=partition_ten(boundaries,ev['token_offsets'],start)
        require(boundaries==ev['original_passage_char_boundaries'] and len(row['documents'])==10, 'native newline boundary inventory differs')
        for i,(doc,group) in enumerate(zip(row['documents'],groups)):
            left,right=group['input_start'],group['input_end']
            require(doc==dict(document_index=i,document_id=digest(ids[left:right]),token_ids_sha256=digest(ids[left:right]),
                    tokens=right-left,input_start=left,input_end=right,offline_position=0,online_position=left,
                    context_char_start=group['char_start'],context_char_end=group['char_end'],
                    source_text_sha256=text_hash(context[group['char_start']:group['char_end']]),
                    passage_start=group['passage_start'],passage_end=group['passage_end']), 'natural10 QASPER block changed')
        require(''.join(context[d['context_char_start']:d['context_char_end']] for d in row['documents'])==context,
                'complete original context must be reconstructed without omission/repetition')
        require(row['document_tokens_total']==sum(d['tokens'] for d in row['documents'])
                and row['input_non_document_tokens']==len(ids)-row['document_tokens_total']
                and row['prefix_tokens']==row['documents'][0]['input_start']
                and row['suffix_tokens']==len(ids)-row['documents'][-1]['input_end']
                and row['context_tokens']==len(ids)-row['prefix_tokens']-row['suffix_tokens']
                and row['rendered_prefix']==prompt[:start] and row['rendered_suffix']==prompt[end:],
                'nondocument gap accounting differs')
    if base is not None:
        for row in rows:
            base.document_specs(row)
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-input-dir','qasper-source','tokenizer','official-encoding','prompt-file','prompt-source','output-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args(argv)
    require(not args.output_dir.exists(), 'never overwrite input preparation')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from tokenizers import Tokenizer
    import importlib.metadata
    require(importlib.metadata.version('tokenizers')=='0.21.4', 'exact tokenizers0.21.4 required')
    bundle={name:read(args.previous_input_dir/name) for name in PREVIOUS_FILES}
    previous.validate_inputs(None,bundle['manifest.json'],bundle['input-protocol.json'],input_dir=args.previous_input_dir)
    encoder=args.official_encoding.read_text(); render=previous.frozen_renderer(encoder)
    prompt_content=args.prompt_file.read_text(); prompt_source=read(args.prompt_source)
    identity=dict(tokenizer_json_sha256=file_hash(args.tokenizer),encoding_sha256=OFFICIAL_ENCODING_SHA256,
        thinking_mode='chat',loader='tokenizers.Tokenizer.from_file',prompt_renderer="official_encoding.encode_messages(thinking_mode='chat')",
        tokenizers_version='0.21.4',official_longbench_prompt_sha256=file_hash(args.prompt_file),
        official_longbench_prompt_revision=prompt_source['revision'])
    artifacts=build_inputs(bundle,args.qasper_source.read_bytes(),Tokenizer.from_file(str(args.tokenizer)),render,
        tokenizer_identity=identity,official_encoding_source=encoder,official_prompt_file_content=prompt_content,
        official_prompt_source=prompt_source)
    args.output_dir.mkdir(parents=True)
    for name,value in artifacts.items():
        publish(args.output_dir/name,value)
    validate_inputs(None,artifacts['manifest.json'],artifacts['input-protocol.json'],input_dir=args.output_dir)
    print(json.dumps(artifacts['preparation.json'],ensure_ascii=False))


if __name__=='__main__':
    main()
