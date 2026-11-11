"""Four native QA datasets, sixteen source QAs and paired official/complete prompts.

CPU preparation only. Selection never reads gold content, gold lengths or model
outputs. The complete original context is retained; cache boundaries use original
paragraphs and both prompt variants must have exactly the same document token IDs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from gpu_pilot import musique_longform_inputs as reuse
from gpu_pilot import musique30_inputs as common

DATASETS=('hotpotqa','2wikimqa','qasper','multifieldqa_en')
VARIANTS=('official','complete')
EVALUATION_INDICES=tuple(range(2,34))
INPUT_COMPANIONS=common.INPUT_COMPANIONS
PREVIOUS_FILES=reuse.PREVIOUS_FILES
MANIFEST_SCHEMA='di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA='kv-cache-engram-broadqa-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA='broadqa-input-protocol-v1'
HISTORY_INVENTORY_SHA256='c7170566bda2ed6c76a594223758e1db3835441a479a179f7850f20353e60afc'
SOURCE_SHA256={
 'hotpotqa':'a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f',
 '2wikimqa':'dda279cf93a99e1e5bfa3291fb199fd55978d10a1feb31822953cf77a1742e37',
 'qasper':'29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854',
 'multifieldqa_en':'0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f'}
SOURCE_COUNTS=dict(hotpotqa=200,**{'2wikimqa':200,'qasper':200,'multifieldqa_en':150})
SEED=20261107
NUM_CHUNKS=10
MIN_BLOCK_TOKENS=256
MAX_INPUT_TOKENS=24576
COMPLETE_INSTRUCTION=('Answer the question using only the provided context. Give a complete, direct answer: '
 'include every requested entity, item, comparison, quantity, and qualifier that is supported by the context. '
 'Be concise, but do not omit requested information merely to keep the answer short. '
 'Do not add unrelated background or unsupported claims. '
 'If the question cannot be answered from the context, answer "unanswerable". '
 'For a yes/no question, answer "yes", "no", or "unanswerable".')
require,digest,text_hash=common.require,common.digest,common.text_hash
file_hash,seal,verify,read,publish=common.file_hash,common.seal,common.verify,common.read,common.publish


def complete_template(dataset):
    # Preserve the official context's immediate whitespace tokenization seam.
    separator=' ' if dataset=='qasper' else '\n'
    return COMPLETE_INSTRUCTION+'\n\nContext:'+separator+'{context}\n\nQuestion: {input}\n\nAnswer:'


def natural_boundaries(dataset,context):
    require(dataset in DATASETS and type(context) is str and context,'native context required')
    if dataset=='qasper':return reuse.qasper_boundaries(context)
    points={0,len(context)}
    points.update(m.end() for m in re.finditer(r'\n[ \t\r]*\n(?:[ \t\r]*\n)*',context))
    if dataset in ('hotpotqa','2wikimqa'):
        points.update(m.start() for m in re.finditer(r'(?m)^Passage\s+\d+\s*:\s*\n',context))
    points=sorted(points)
    require(len(points)>=11,'at least ten native paragraph segments required')
    return points


def history_sets(history):
    verify(history)
    require(history.get('schema')=='broadqa-history-inventory-v1' and history['sha256']==HISTORY_INVENTORY_SHA256
        and history.get('model_outcomes_used') is False and history.get('provenance'), 'known-history inventory differs')
    require(all(history['source_files'].get(d)==SOURCE_SHA256[d] for d in DATASETS),'history source acquisition differs')
    return ({tuple(x) for x in history['source_identities']},set(history['context_sha256']),
            set(history['question_sha256']),set(history['normalized_question_sha256']))


def normalized_question_sha(question):return text_hash(' '.join(question.lower().split()))


def source_metadata(dataset,index,raw):
    context_sha=text_hash(raw['context']);question_sha=text_hash(raw['input'])
    pair_id=digest(dict(dataset=dataset,source_id=raw['_id'],context_sha256=context_sha,question_sha256=question_sha))
    return dict(dataset=dataset,source_index=index,source_id=raw['_id'],source_row_sha256=digest(raw),
        context_sha256=context_sha,question_sha256=question_sha,normalized_question_sha256=normalized_question_sha(raw['input']),
        pair_id=pair_id,tie_sha256=digest(dict(seed=SEED,dataset=dataset,source_id=raw['_id'],context_sha256=context_sha)))


def exclusion_reason(meta,sets):
    identities,contexts,questions,normalized=sets
    if (meta['dataset'],meta['source_id']) in identities:return 'known_history_source'
    if meta['context_sha256'] in contexts:return 'known_history_context'
    if meta['question_sha256'] in questions or meta['normalized_question_sha256'] in normalized:return 'known_history_question'
    return None


def documents_from_groups(context,ids,groups):
    answer=[]
    for i,g in enumerate(groups):
        left,right=g['input_start'],g['input_end'];sha=digest(ids[left:right])
        answer.append(dict(document_index=i,document_id=sha,token_ids_sha256=sha,tokens=right-left,
            input_start=left,input_end=right,offline_position=0,online_position=left,
            context_char_start=g['char_start'],context_char_end=g['char_end'],
            source_text_sha256=text_hash(context[g['char_start']:g['char_end']]),
            passage_start=g['passage_start'],passage_end=g['passage_end']))
    return answer


def build_variant(raw,dataset,index,variant,tokenizer,render,template,reference_groups=None):
    context,question=raw['context'],raw['input'];meta=source_metadata(dataset,index,raw)
    require(raw.get('dataset')==dataset and type(raw.get('answers')) is list and raw['answers'],'original QA source required')
    prompt=render([dict(role='user',content=template.format(context=context,input=question))],thinking_mode='chat')
    require(prompt.count(context)==prompt.count(question)==1,'original context/question must occur exactly once')
    encoded=tokenizer.encode(prompt,add_special_tokens=False);ids,offsets=encoded.ids,encoded.offsets
    require(len(ids)<=MAX_INPUT_TOKENS,'complete prompt exceeds declared24576 token selection cap')
    require(tokenizer.decode(ids,skip_special_tokens=False)==prompt and
        tokenizer.encode(tokenizer.decode(ids,skip_special_tokens=False),add_special_tokens=False).ids==ids,'whole prompt roundtrip differs')
    start=prompt.index(context);end=start+len(context);boundaries=natural_boundaries(dataset,context)
    if reference_groups is None:groups=reuse.partition_ten(boundaries,offsets,start)
    else:
        arrays=common.offset_arrays(offsets);groups=[]
        for g in reference_groups:
            left,right=common.exact_token_span(offsets,start+g['char_start'],start+g['char_end'],arrays=arrays)
            require(right-left>=256,'complete variant native block below256')
            groups.append(dict(g,input_start=left,input_end=right))
    documents=documents_from_groups(context,ids,groups)
    require(len({d['document_id'] for d in documents})==10,'duplicate native cache block')
    context_left,context_right=common.exact_token_span(offsets,start,end)
    prefix,suffix=documents[0]['input_start'],len(ids)-documents[-1]['input_end']
    row=dict(meta,sample_id='broadqa:'+dataset+':'+raw['_id']+':'+variant,source_line=index+1,
        prompt_variant=variant,question=question,references=raw['answers'],references_sha256=digest(raw['answers']),
        input_ids=ids,input_ids_sha256=digest(ids),input_tokens=len(ids),prompt=prompt,prompt_sha256=text_hash(prompt),
        context_text=context,context_prompt_char_start=start,context_prompt_char_end=end,
        original_context_tokens=len(tokenizer.encode(context,add_special_tokens=False).ids),
        context_tokens=len(ids)-prefix-suffix,document_tokens_total=sum(d['tokens'] for d in documents),
        prefix_tokens=prefix,suffix_tokens=suffix,rendered_prefix=prompt[:start],rendered_suffix=prompt[end:],
        context_semantic_token_ids_sha256=digest(ids[context_left:context_right]),
        primary_context_truncated=False,natural_boundary_inputs=True,source_context_count=1,source_order_preserved=True,
        text_roundtrip_matches_ids=True,prompt_transport='whole rendered prompt tokenized once; exact token IDs authoritative',
        block_boundary_kind='original_newline_segments' if dataset=='qasper' else 'original_paragraph_boundaries',
        documents=documents,input_non_document_tokens=len(ids)-sum(d['tokens'] for d in documents),
        support_retention='entire original QA context, question and references unchanged; prompt instructions are the only treatment')
    ev=dict(sample_id=row['sample_id'],prompt_sha256=row['prompt_sha256'],input_ids_sha256=row['input_ids_sha256'],
        whole_prompt_single_encoding=True,decode_matches_rendered_prompt=True,reencode_matches_input_ids=True,
        token_offsets=[list(v) for v in offsets],original_passage_char_boundaries=boundaries,
        full_source_context_preserved=True,full_source_question_preserved=True,additional_source_contexts=0)
    return row,ev,groups


def check_pair(a,b):
    for key in ('pair_id','dataset','source_id','source_index','source_row_sha256','context_sha256','context_text',
                'question','references','references_sha256','context_semantic_token_ids_sha256'):
        require(a[key]==b[key],'A/B original QA/context token identity differs: '+key)
    require(a['prompt_variant']=='official' and b['prompt_variant']=='complete' and
        a['prompt']!=b['prompt'] and a['input_ids']!=b['input_ids'],'two distinct prompt treatments required')
    require(len(a['documents'])==len(b['documents'])==10,'A/B ten native blocks required')
    for da,db in zip(a['documents'],b['documents']):
        for key in ('document_index','document_id','token_ids_sha256','tokens','context_char_start','context_char_end','source_text_sha256','passage_start','passage_end'):
            require(da[key]==db[key],'A/B natural block identity differs: '+key)
        require(a['input_ids'][da['input_start']:da['input_end']]==b['input_ids'][db['input_start']:db['input_end']],
            'A/B actual cached document token IDs differ')


def select_pairs(candidates):
    pools={d:sorted((m for m in candidates if m['dataset']==d),key=lambda m:(m['tie_sha256'],m['source_index'])) for d in DATASETS}
    contexts=set();questions=set();chosen=[]
    for ordinal in range(4):
        for dataset in DATASETS:
            item=next((m for m in pools[dataset] if m['context_sha256'] not in contexts and m['normalized_question_sha256'] not in questions),None)
            require(item is not None,'fewer than four distinct eligible original QAs: '+dataset)
            chosen.append(item);contexts.add(item['context_sha256']);questions.add(item['normalized_question_sha256'])
    return chosen


def build_inputs(previous_bundle,source_bytes,history,tokenizer,render,*,tokenizer_identity,
                 official_encoding_source,official_prompt_file_content,official_prompt_source):
    prior=reuse.validate_previous_bundle(previous_bundle);sets=history_sets(history)
    require(prior['tokenizer']==tokenizer_identity,'actual checkpoint tokenizer differs from original calibration')
    require(text_hash(official_encoding_source)==common.OFFICIAL_ENCODING_SHA256 and
        text_hash(official_prompt_file_content)==official_prompt_source['sha256']==tokenizer_identity['official_longbench_prompt_sha256'],
        'official renderer/prompt identity differs')
    templates=json.loads(official_prompt_file_content);eligible=[];rejected=[];lookup={};files=[]
    require(set(source_bytes)==set(DATASETS),'exact four source acquisitions required')
    for dataset in DATASETS:
        content=source_bytes[dataset]
        require(hashlib.sha256(content).hexdigest()==SOURCE_SHA256[dataset],'source acquisition differs: '+dataset)
        raws=[json.loads(s) for s in content.splitlines() if s.strip()]
        require(len(raws)==SOURCE_COUNTS[dataset] and len({r['_id'] for r in raws})==len(raws),'source inventory differs')
        files.append(dict(dataset=dataset,acquisition_sha256=SOURCE_SHA256[dataset],bytes=len(content),records=len(raws)))
        for index,raw in enumerate(raws):
            meta=source_metadata(dataset,index,raw);reason=exclusion_reason(meta,sets)
            if reason:rejected.append(dict(meta,reason=reason));continue
            try:
                a,ea,groups=build_variant(raw,dataset,index,'official',tokenizer,render,templates[dataset])
                b,eb,_=build_variant(raw,dataset,index,'complete',tokenizer,render,complete_template(dataset),groups)
                check_pair(a,b)
            except ValueError as exc:
                rejected.append(dict(meta,reason='native_geometry: '+str(exc)));continue
            item=dict(meta,variant_input_tokens={v:r['input_tokens'] for v,r in zip(VARIANTS,(a,b))},
                variant_input_ids_sha256={v:r['input_ids_sha256'] for v,r in zip(VARIANTS,(a,b))})
            eligible.append(item);lookup[meta['pair_id']]=(a,b,ea,eb,raw)
    chosen=select_pairs(eligible);rows=copy.deepcopy(prior['warmup']);sources=[];evidence=[]
    calids={r['sample_id'] for r in rows}
    sources.extend(copy.deepcopy(r) for r in previous_bundle['source-records.json']['records'] if r['sample_id'] in calids)
    evidence.extend(copy.deepcopy(r) for r in previous_bundle['tokenization-evidence.json']['records'] if r['sample_id'] in calids)
    for ordinal,item in enumerate(chosen,1):
        a,b,ea,eb,raw=lookup[item['pair_id']]
        for row,ev in ((a,ea),(b,eb)):
            row=dict(row,index=len(rows),split='evaluation',pair_ordinal=ordinal);rows.append(row);evidence.append(ev)
            sources.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],source_index=row['source_index'],source_row_sha256=digest(raw),raw=raw))
    source=seal(dict(schema='broadqa-source-records-v1',previous_inputs=previous_bundle,history_inventory=history,
        original_source_jsonl={d:source_bytes[d].decode('utf-8') for d in DATASETS},files=files,records=sources))
    selection=seal(dict(schema='broadqa-selection-plan-v1',seed=SEED,no_model_outcomes_used=True,gold_length_selection=False,
        no_answer_content_selection=True,generated_length_selection=False,max_complete_prompt_tokens=MAX_INPUT_TOKENS,
        history_inventory_sha256=history['sha256'],eligible_candidates=sorted(eligible,key=lambda m:(DATASETS.index(m['dataset']),m['tie_sha256'],m['source_index'])),
        rejected=rejected,selected_pair_ids=[m['pair_id'] for m in chosen],selected_sample_ids=[r['sample_id'] for r in rows],
        pair_order='dataset round-robin; official then complete adjacent; fixed source SHA ranking',
        known_history_scope=history['scope'],independent_model_training_holdout_claim=False))
    tokens=seal(dict(schema='broadqa-tokenization-evidence-v1',tokenizer=tokenizer_identity,
        official_prompt_source=official_prompt_source,official_prompt_file_content=official_prompt_file_content,
        official_encoding_source=official_encoding_source,official_prompt_templates={d:templates[d] for d in DATASETS},
        complete_prompt_templates={d:complete_template(d) for d in DATASETS},
        prompt_template_sha256={d:{'official':text_hash(templates[d]),'complete':text_hash(complete_template(d))} for d in DATASETS},records=evidence))
    protocol=seal(dict(schema=INPUT_PROTOCOL_SCHEMA,seed=SEED,datasets=list(DATASETS),source_qa_count=16,evaluation_cases=32,
        calibration_indices=[0,1],evaluation_indices=list(EVALUATION_INDICES),prompt_variants=list(VARIANTS),
        calibration_source='previous_musique_calibration_exact',previous_manifest_sha256=prior['sha256'],
        num_chunks=10,min_block_tokens=256,max_complete_prompt_tokens=MAX_INPUT_TOKENS,
        no_padding=True,no_truncation=True,no_cross_sample_contexts=True,preserve_original_order=True,
        single_prompt_tokenization=True,no_model_outcomes_used=True,no_answer_content_selection=True,
        gold_length_selection=False,generated_length_selection=False,minimum_generated_tokens=None,natural_eos_preserved=True,
        pair_context_and_document_tokens_identical=True,known_history_inventory_sha256=history['sha256'],
        known_history_scope=history['scope'],tokenizer=tokenizer_identity,tokenizers_version='0.21.4',inference_seed=0,
        source_records_sha256=source['sha256'],selection_plan_sha256=selection['sha256'],tokenization_evidence_sha256=tokens['sha256']))
    companions=dict(zip(INPUT_COMPANIONS,(source,selection,tokens)))
    manifest=seal(dict(schema=MANIFEST_SCHEMA,contract_schema=CONTRACT_SCHEMA,artifact_kind='broadqa_inputs_v1',natural_boundary_inputs=True,
        input_protocol_sha256=protocol['sha256'],tokenizer=tokenizer_identity,num_chunks=10,chunk_tokens=None,context_tokens=None,
        calibration_source='previous_musique_calibration_exact',sources=files,warmup=rows[:2],cohort=rows[2:],
        companion_sha256={k:v['sha256'] for k,v in companions.items()},selection_plan_sha256=selection['sha256']))
    summary=seal(dict(schema='broadqa-input-preparation-v1',status='complete',cuda_used=False,model_results_used=False,
        manifest_sha256=manifest['sha256'],input_protocol_sha256=protocol['sha256'],history_inventory_sha256=history['sha256'],
        source_qa_count=16,evaluation_cases=32,eligible_counts={d:sum(m['dataset']==d for m in eligible) for d in DATASETS},
        samples=[dict(index=r['index'],dataset=r['dataset'],source_index=r['source_index'],pair_id=r.get('pair_id'),
            pair_ordinal=r.get('pair_ordinal'),prompt_variant=r.get('prompt_variant'),input_tokens=r['input_tokens'],
            block_tokens=[d['tokens'] for d in r['documents']],source_context_preserved=True,whole_prompt_roundtrip=True) for r in rows]))
    return dict(companions,**{'manifest.json':manifest,'input-protocol.json':protocol,'preparation.json':summary})


def validate_inputs(base,manifest,protocol,*,input_dir):
    verify(manifest);verify(protocol)
    require(manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('contract_schema')==CONTRACT_SCHEMA
        and manifest.get('artifact_kind')=='broadqa_inputs_v1' and protocol.get('schema')==INPUT_PROTOCOL_SCHEMA
        and manifest['input_protocol_sha256']==protocol['sha256'],'BroadQA input schema/binding differs')
    fixed=dict(seed=SEED,datasets=list(DATASETS),source_qa_count=16,evaluation_cases=32,calibration_indices=[0,1],
        evaluation_indices=list(EVALUATION_INDICES),prompt_variants=list(VARIANTS),calibration_source='previous_musique_calibration_exact',
        num_chunks=10,min_block_tokens=256,max_complete_prompt_tokens=MAX_INPUT_TOKENS,no_padding=True,no_truncation=True,
        no_cross_sample_contexts=True,preserve_original_order=True,single_prompt_tokenization=True,no_model_outcomes_used=True,
        no_answer_content_selection=True,gold_length_selection=False,generated_length_selection=False,minimum_generated_tokens=None,
        natural_eos_preserved=True,pair_context_and_document_tokens_identical=True,tokenizers_version='0.21.4',inference_seed=0)
    require(all(k in protocol and protocol[k]==v for k,v in fixed.items()),'BroadQA fixed policy differs')
    companions={n:read(Path(input_dir)/n) for n in INPUT_COMPANIONS}
    require(set(manifest['companion_sha256'])==set(INPUT_COMPANIONS),'companion inventory differs')
    for n,v in companions.items():verify(v);require(manifest['companion_sha256'][n]==v['sha256'],'companion seal differs')
    source,selection,tokens=[companions[n] for n in INPUT_COMPANIONS]
    for key,v in (('source_records_sha256',source),('selection_plan_sha256',selection),('tokenization_evidence_sha256',tokens)):
        require(protocol[key]==v['sha256'],'provenance chain differs')
    require([v.get('schema') for v in (source,selection,tokens)]==['broadqa-source-records-v1','broadqa-selection-plan-v1','broadqa-tokenization-evidence-v1'],
        'BroadQA companion schema differs')
    prior=reuse.validate_previous_bundle(source['previous_inputs']);history=source['history_inventory'];sets=history_sets(history)
    require(protocol['known_history_inventory_sha256']==selection['history_inventory_sha256']==history['sha256']
        and protocol['known_history_scope']==selection['known_history_scope']==history['scope'],'known-history provenance differs')
    require(manifest['tokenizer']==protocol['tokenizer']==tokens['tokenizer']==prior['tokenizer']
        and tokens['tokenizer']['encoding_sha256']==common.OFFICIAL_ENCODING_SHA256
        and protocol['previous_manifest_sha256']==prior['sha256'],'actual tokenizer/calibration identity differs')
    templates=json.loads(tokens['official_prompt_file_content']);render=common.frozen_renderer(tokens['official_encoding_source'])
    require(text_hash(tokens['official_prompt_file_content'])==tokens['official_prompt_source']['sha256']
        ==manifest['tokenizer']['official_longbench_prompt_sha256'] and tokens['official_prompt_templates']=={d:templates[d] for d in DATASETS}
        and tokens['complete_prompt_templates']=={d:complete_template(d) for d in DATASETS}
        and tokens['prompt_template_sha256']=={d:{'official':text_hash(templates[d]),'complete':text_hash(complete_template(d))} for d in DATASETS},
        'official/complete prompt templates changed')
    raws={};files=[]
    require(set(source['original_source_jsonl'])==set(DATASETS),'exact four native source files required')
    for d in DATASETS:
        content=source['original_source_jsonl'][d].encode();require(hashlib.sha256(content).hexdigest()==SOURCE_SHA256[d],'original source bytes changed')
        raws[d]=[json.loads(s) for s in content.splitlines() if s.strip()]
        require(len(raws[d])==SOURCE_COUNTS[d] and len({r['_id'] for r in raws[d]})==len(raws[d]),'source record inventory differs')
        files.append(dict(dataset=d,acquisition_sha256=SOURCE_SHA256[d],bytes=len(content),records=len(raws[d])))
    require(source['files']==manifest['sources']==files,'source file inventory differs')
    rows=manifest['warmup']+manifest['cohort']
    require(manifest['warmup']==prior['warmup'] and len(manifest['cohort'])==32 and [r['index'] for r in rows]==list(range(34))
        and manifest['num_chunks']==10 and manifest['natural_boundary_inputs'] is True
        and manifest['calibration_source']==protocol['calibration_source']
        and manifest['selection_plan_sha256']==selection['sha256'],'calibration2/evaluation32 geometry differs')
    for k in ('seed','no_model_outcomes_used','no_answer_content_selection','gold_length_selection','generated_length_selection','max_complete_prompt_tokens'):
        require(selection.get(k)==protocol[k],'selection policy differs: '+k)
    candidates,rejected=selection['eligible_candidates'],selection['rejected']
    require(sorted((m['dataset'],m['source_index']) for m in candidates+rejected)==sorted((d,i) for d in DATASETS for i in range(SOURCE_COUNTS[d])),
        'every acquired source must appear exactly once in eligibility inventory')
    for item in candidates+rejected:
        raw=raws[item['dataset']][item['source_index']];meta=source_metadata(item['dataset'],item['source_index'],raw)
        require(all(item.get(k)==v for k,v in meta.items()),'candidate original source identity differs')
        reason=exclusion_reason(meta,sets)
        if item in candidates:
            require(reason is None and set(item['variant_input_tokens'])==set(VARIANTS)
                and all(type(n) is int and 0<n<=MAX_INPUT_TOKENS for n in item['variant_input_tokens'].values()),'ineligible source selected')
        else:require(item['reason']==reason if reason else item['reason'].startswith('native_geometry: '),'source exclusion reason differs')
    require(candidates==sorted(candidates,key=lambda m:(DATASETS.index(m['dataset']),m['tie_sha256'],m['source_index'])),'candidate ranking differs')
    chosen=select_pairs(candidates)
    require(selection['selected_pair_ids']==[m['pair_id'] for m in chosen]
        and selection['selected_sample_ids']==[r['sample_id'] for r in rows]
        and len({r['sample_id'] for r in rows})==34,'selected source/prompt order differs')
    records={r['sample_id']:r for r in source['records']};evidence={r['sample_id']:r for r in tokens['records']}
    require(len(records)==len(source['records'])==len(evidence)==len(tokens['records'])==34
        and set(records)==set(evidence)=={r['sample_id'] for r in rows},'source/token evidence inventory differs')
    old_sources={r['sample_id']:r for r in source['previous_inputs']['source-records.json']['records']}
    old_tokens={r['sample_id']:r for r in source['previous_inputs']['tokenization-evidence.json']['records']}
    for r in rows[:2]:require(records[r['sample_id']]==old_sources[r['sample_id']] and evidence[r['sample_id']]==old_tokens[r['sample_id']],
        'original calibration source/token evidence changed')
    for ordinal,item in enumerate(chosen,1):
        pair=rows[2+2*(ordinal-1):4+2*(ordinal-1)];official_groups=None
        for variant,row in zip(VARIANTS,pair):
            require(row['pair_ordinal']==ordinal and row['prompt_variant']==variant and row['split']=='evaluation'
                and all(row.get(k)==v for k,v in source_metadata(item['dataset'],item['source_index'],raws[item['dataset']][item['source_index']]).items()),
                'round-robin paired row/source binding differs')
            raw=raws[row['dataset']][row['source_index']];context=raw['context'];ev=evidence[row['sample_id']];ids=row['input_ids']
            require(row['sample_id']=='broadqa:'+row['dataset']+':'+raw['_id']+':'+variant
                and records[row['sample_id']]==dict(sample_id=row['sample_id'],dataset=row['dataset'],source_index=row['source_index'],source_row_sha256=digest(raw),raw=raw)
                and row['question']==raw['input'] and row['references']==raw['answers'] and row['references_sha256']==digest(raw['answers'])
                and row['context_text']==context,'original complete QA bytes changed')
            template=templates[row['dataset']] if variant=='official' else complete_template(row['dataset'])
            prompt=render([dict(role='user',content=template.format(context=context,input=raw['input']))],thinking_mode='chat')
            start=row['context_prompt_char_start'];end=row['context_prompt_char_end']
            require(row['prompt']==prompt and prompt[start:end]==context and end==start+len(context)
                and prompt.count(context)==prompt.count(raw['input'])==1 and row['prompt_sha256']==ev['prompt_sha256']==text_hash(prompt)
                and row['input_tokens']==len(ids)==item['variant_input_tokens'][variant]
                and row['input_ids_sha256']==ev['input_ids_sha256']==item['variant_input_ids_sha256'][variant]==digest(ids)
                and all(type(t) is int and 0<=t<2**31 for t in ids),'complete prompt/token identity differs')
            require(row['natural_boundary_inputs'] is True and row['primary_context_truncated'] is False
                and row['source_context_count']==1 and row['source_order_preserved'] is True and row['text_roundtrip_matches_ids'] is True,
                'source preservation claim differs')
            require(len(ev['token_offsets'])==len(ids) and ev['additional_source_contexts']==0
                and all(ev[k] is True for k in ('whole_prompt_single_encoding','decode_matches_rendered_prompt','reencode_matches_input_ids',
                    'full_source_context_preserved','full_source_question_preserved')),'actual tokenizer evidence differs')
            boundaries=natural_boundaries(row['dataset'],context);require(boundaries==ev['original_passage_char_boundaries'],'native boundaries differ')
            if variant=='official':groups=reuse.partition_ten(boundaries,ev['token_offsets'],start);official_groups=groups
            else:
                groups=[];arrays=common.offset_arrays(ev['token_offsets'])
                for g in official_groups:
                    left,right=common.exact_token_span(ev['token_offsets'],start+g['char_start'],start+g['char_end'],arrays=arrays)
                    groups.append(dict(g,input_start=left,input_end=right))
            require(row['documents']==documents_from_groups(context,ids,groups)
                and all(d['tokens']>=256 for d in row['documents'])
                and ''.join(context[d['context_char_start']:d['context_char_end']] for d in row['documents'])==context,
                'exact ten complete natural blocks differ')
            left,right=common.exact_token_span(ev['token_offsets'],start,end)
            require(row['context_semantic_token_ids_sha256']==digest(ids[left:right])
                and row['document_tokens_total']==sum(d['tokens'] for d in row['documents'])
                and row['input_non_document_tokens']==len(ids)-row['document_tokens_total']
                and row['prefix_tokens']==row['documents'][0]['input_start']
                and row['suffix_tokens']==len(ids)-row['documents'][-1]['input_end']
                and row['rendered_prefix']==prompt[:start] and row['rendered_suffix']==prompt[end:],'native token/gap accounting differs')
        check_pair(*pair)
    if base is not None:
        base.validate_manifest(manifest)
        for row in rows:base.document_specs(row)
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('previous-input-dir','source-dir','history-inventory','tokenizer','official-encoding','prompt-file','prompt-source','output-dir'):
        p.add_argument('--'+n,required=True,type=Path)
    args=p.parse_args(argv);require(not args.output_dir.exists(),'never overwrite input preparation')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from tokenizers import Tokenizer
    import importlib.metadata
    require(importlib.metadata.version('tokenizers')=='0.21.4','exact tokenizers0.21.4 required')
    bundle={n:read(args.previous_input_dir/n) for n in PREVIOUS_FILES}
    common.validate_inputs(None,bundle['manifest.json'],bundle['input-protocol.json'],input_dir=args.previous_input_dir)
    encoder=args.official_encoding.read_text();render=common.frozen_renderer(encoder);prompt=args.prompt_file.read_text();origin=read(args.prompt_source)
    identity=dict(tokenizer_json_sha256=file_hash(args.tokenizer),encoding_sha256=common.OFFICIAL_ENCODING_SHA256,thinking_mode='chat',
        loader='tokenizers.Tokenizer.from_file',prompt_renderer="official_encoding.encode_messages(thinking_mode='chat')",tokenizers_version='0.21.4',
        official_longbench_prompt_sha256=file_hash(args.prompt_file),official_longbench_prompt_revision=origin['revision'])
    result=build_inputs(bundle,{d:(args.source_dir/(d+'.jsonl')).read_bytes() for d in DATASETS},read(args.history_inventory),
        Tokenizer.from_file(str(args.tokenizer)),render,tokenizer_identity=identity,official_encoding_source=encoder,
        official_prompt_file_content=prompt,official_prompt_source=origin)
    args.output_dir.mkdir(parents=True)
    for n,v in result.items():publish(args.output_dir/n,v)
    validate_inputs(None,result['manifest.json'],result['input-protocol.json'],input_dir=args.output_dir)
    print(json.dumps(result['preparation.json'],ensure_ascii=False))


if __name__=='__main__':main()
