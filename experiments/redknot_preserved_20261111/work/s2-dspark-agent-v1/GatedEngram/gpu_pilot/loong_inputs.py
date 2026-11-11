"""Complete original Loong multi-document QAs; CPU input preparation only.

5..10 source documents stay intact and ordered. Ten cache blocks may subdivide
original blank paragraphs but never join across a source document boundary.
"""
from __future__ import annotations
import argparse
import bisect
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile
from gpu_pilot import musique30_inputs as common
from gpu_pilot import musique_longform_inputs as reuse
from gpu_pilot.broadqa_inputs import documents_from_groups

MANIFEST_SCHEMA='di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA='kv-cache-engram-loong-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA='loong-input-protocol-v1'
INPUT_COMPANIONS=common.INPUT_COMPANIONS
PREVIOUS_FILES=reuse.PREVIOUS_FILES
DATASETS=('loong',)
LENGTH_BANDS=('128k','64k')
TARGET_INPUT_TOKENS={'128k':131072,'64k':65536}
INPUT_TOKEN_RANGES={'128k':[112*1024,144*1024],'64k':[56*1024,72*1024]}
CASES_PER_BAND=10
EVALUATION_INDICES=tuple(range(2,22))
NUM_CHUNKS=10
MIN_BLOCK_TOKENS=256
SEED=20261109
SOURCE_REVISION='6d2115b8b48a3d19412ccb52a0c9c4ee37869af4'
SOURCE_SHA256='cd640d3ca0e41d3638153506e3caee109850b30de0a4e829862232a2b6302da8'
OFFICIAL_PROMPT_SOURCE_SHA256='459b41566258d59f41ae4d3a707bf45c09e275c4723cc20e9ad8f79c8bbc682d'
DOCUMENT_ARCHIVE_SHA256='ceb75be50f7f114f231e3a8069b51457745ea7db151c017f00de33d0789ab4cc'
DOCUMENT_CATALOG_SHA256='6655640a067adc1c8808f2db58a6746fdd3cd8f854c9a60099840a6063746c41'
CALIBRATION_SOURCE='previous_musique_calibration_exact'
require,digest,text_hash=common.require,common.digest,common.text_hash
file_hash,seal,verify,read,publish=common.file_hash,common.seal,common.verify,common.read,common.publish


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def source_rows(raw_bytes):
    require(hashlib.sha256(raw_bytes).hexdigest()==SOURCE_SHA256,'pinned Loong source JSONL differs')
    rows=[json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    require(len(rows)==1600 and len({r['id'] for r in rows})==1600,'complete original Loong1600 metadata required')
    for r in rows:
        require(r['type'] in ('paper','financial','legal') and r['language'] in ('en','zh')
            and type(r['level']) is int and r['level'] in (1,2,3,4)
            and type(r['doc']) is list and r['doc']
            and all(type(x) is str and x for x in r['doc'])
            and all(type(r[k]) is str for k in ('id','question','instruction','prompt_template')),
            'original Loong metadata invalid')
        canonical(r['answer'])  # Preserve JSON type; never select on its contents.
    return rows


def file_catalog(document_files):
    return {name:dict(sha256=text_hash(text),bytes=len(text.encode('utf-8'))) for name,text in sorted(document_files.items())}


def archive_document_files(archive,doc_dir=None):
    require(DOCUMENT_ARCHIVE_SHA256 is not None and file_hash(archive)==DOCUMENT_ARCHIVE_SHA256,'official document archive SHA differs')
    result={}
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():continue
            parts=Path(member.filename).parts
            found=next((i for i,p in enumerate(parts) if p in ('paper','financial','legal')),None)
            if found is None or '__MACOSX' in parts:continue
            name='/'.join(parts[found:])
            if not name.endswith(('.txt','.md','.json')):continue
            require('..' not in Path(name).parts and name not in result,'unsafe/duplicate archive source path')
            data=z.read(member);text=data.decode('utf-8')
            if doc_dir is not None:require((Path(doc_dir)/name).read_bytes()==data,'extracted source bytes differ: '+name)
            result[name]=text
    require(result and DOCUMENT_CATALOG_SHA256==digest(file_catalog(result)),'official archive document inventory differs')
    return result


def original_documents(raw,document_files):
    """Match pinned get_content formatting; make ambiguous glob matches fail closed."""
    out=[];domain=raw['type']
    for ordinal,name in enumerate(raw['doc']):
        if domain=='paper':
            path='paper/'+name;content=document_files[path]
            title=content.split('\n',1)[0].strip('#').strip()
            text=title+'\n'+content+'\n\n';part='entire_file'
        elif domain=='financial':
            paths=[p for p in document_files if p.startswith('financial/') and p.endswith('.txt')
                and name in Path(p).name and (raw['level']==4 or '2024-'+name in Path(p).name)]
            require(len(paths)==1,'financial document name has missing/ambiguous official file: '+name)
            path=paths[0];content=document_files[path]
            title=Path(path).stem.split('-')[-1];text='《'+title+'》\n'+content+'\n\n';part='entire_file'
        else:
            path='legal/legal.json';entry=_legal_records(document_files[path])[name]
            content_only=raw['level']==4 and '阅读以上判决文书，我将给你若干份判决结果：' in raw['instruction']
            content=entry['content'] if content_only else entry['content']+entry['result']
            text='《判决文书'+str(ordinal+1)+'》\n'+content+'\n\n';part='content_only_official_level4' if content_only else 'content_plus_result'
        out.append(dict(source_document_ordinal=ordinal,source_document_name=name,source_file=path,
            source_file_sha256=text_hash(document_files[path]),source_content_sha256=text_hash(content),
            source_content_selection=part,source_content_characters=len(content),rendered_doc_sha256=text_hash(text),text=text))
    return out


@lru_cache(maxsize=1)
def _legal_records(text):
    return json.loads(text)


def render_original(raw,documents,render):
    context=''.join(d['text'] for d in documents)
    values={'{docs}':context,'{question}':raw['question'],'{instruction}':raw['instruction']}
    template=raw['prompt_template']
    require(template.count('{docs}')==template.count('{instruction}')==1
        and (template.count('{question}')==1 or (template.count('{question}')==0 and raw['question']=='')),
        'original Loong template placeholders differ')
    content=re.sub(r'\{docs\}|\{question\}|\{instruction\}',lambda m:values[m.group()],template)
    prompt=render([dict(role='user',content=content)],thinking_mode='chat')
    require(prompt.count(context)==1,'complete source documents must appear once')
    return prompt,context


def paragraph_boundaries(text):
    return sorted({0,len(text),*(m.end() for m in re.finditer(r'\n[ \t\r]*\n(?:[ \t\r]*\n)*',text))})


def partition_document(points,offsets,start,count,arrays):
    """Feasible suffix DP, then deterministic balanced cuts; O(count*boundaries)."""
    token_starts,token_ends=arrays
    left=[bisect.bisect_left(token_starts,start+p) for p in points]
    right=[bisect.bisect_right(token_ends,start+p) for p in points]
    last=len(points)-1
    possible=[[False]*len(points) for _ in range(count+1)];possible[0][last]=True
    for k in range(1,count+1):
        valid=[j for j in range(1,last+1) if possible[k-1][j]]
        end_values=[right[j] for j in valid]
        for i in range(last):
            t=bisect.bisect_left(end_values,left[i]+MIN_BLOCK_TOKENS)
            possible[k][i]=t<len(valid) and valid[t]>i
    require(possible[count][0],'insufficient natural paragraph geometry for source document')
    total=right[last]-left[0];answer=[];begin=0
    for k in range(count):
        options=[j for j in range(begin+1,last+1) if possible[count-k-1][j] and right[j]-left[begin]>=MIN_BLOCK_TOKENS]
        end=min(options,key=lambda j:(abs(count*(right[j]-left[0])-(k+1)*total),j))
        answer.append(dict(char_start=points[begin],char_end=points[end],input_start=left[begin],input_end=right[end],
            passage_start=begin,passage_end=end));begin=end
    return answer


def cache_geometry(source_docs,offsets,context_start):
    require(5<=len(source_docs)<=10,'five to ten original documents required')
    arrays=common.offset_arrays(offsets);cursor=0;choices=[];original=[]
    total=0
    for d in source_docs:
        text=d['text'];points=paragraph_boundaries(text);start=context_start+cursor
        left,right=common.exact_token_span(offsets,start,start+len(text),arrays=arrays);total+=right-left
        candidates={}
        for count in range(1,NUM_CHUNKS-len(source_docs)+2):
            try:candidates[count]=partition_document(points,offsets,start,count,arrays)
            except ValueError:continue
        require(1 in candidates,'original document shorter than minimum256 surviving tokens')
        choices.append((cursor,candidates));original.append(dict({k:v for k,v in d.items() if k!='text'},
            context_char_start=cursor,context_char_end=cursor+len(text),prompt_char_start=start,prompt_char_end=start+len(text),
            input_start=left,input_end=right,tokens=right-left))
        cursor+=len(text)
    dp={0:(0,())}
    for _,candidates in choices:
        next_dp={}
        for used,(cost,allocation) in dp.items():
            for count,groups in candidates.items():
                if used+count>NUM_CHUNKS:continue
                candidate=(cost+sum((NUM_CHUNKS*(g['input_end']-g['input_start'])-total)**2 for g in groups),allocation+(count,))
                if used+count not in next_dp or candidate<next_dp[used+count]:next_dp[used+count]=candidate
        dp=next_dp
    require(NUM_CHUNKS in dp,'cannot make ten cache blocks without crossing original document boundaries')
    groups=[]
    for ordinal,((cursor,candidates),count) in enumerate(zip(choices,dp[NUM_CHUNKS][1])):
        for g in candidates[count]:
            groups.append(dict(g,char_start=g['char_start']+cursor,char_end=g['char_end']+cursor,
                source_document_ordinal=ordinal,source_document_name=source_docs[ordinal]['source_document_name'],
                source_doc_char_start=g['char_start'],source_doc_char_end=g['char_end']))
    return original,groups


def classify_length(n):
    return next((b for b in LENGTH_BANDS if INPUT_TOKEN_RANGES[b][0]<=n<=INPUT_TOKEN_RANGES[b][1]),None)


def source_metadata(raw,index):
    return dict(dataset='loong',source_index=index,source_id=raw['id'],source_row_sha256=digest(raw),
        source_domain=raw['type'],source_language=raw['language'],source_level=raw['level'],
        source_document_count=len(raw['doc']),original_document_count=len(raw['doc']),
        question_sha256=text_hash(raw['question']),source_qa_sha256=digest({k:raw[k] for k in ('doc','question','instruction','type','level')}),
        tie_sha256=digest(dict(seed=SEED,source_id=raw['id'])))


def build_row(raw,index,document_files,tokenizer,render):
    require(5<=len(raw['doc'])<=10,'original document count outside5..10')
    require(len(set(raw['doc']))==len(raw['doc']),'original QA repeats a source document; rejected without altering it')
    source_docs=original_documents(raw,document_files);prompt,context=render_original(raw,source_docs,render)
    encoded=tokenizer.encode(prompt,add_special_tokens=False);ids,offsets=encoded.ids,encoded.offsets
    band=classify_length(len(ids));require(band is not None,'complete prompt outside actual-token length bands')
    require(tokenizer.decode(ids,skip_special_tokens=False)==prompt and
        tokenizer.encode(tokenizer.decode(ids,skip_special_tokens=False),add_special_tokens=False).ids==ids,'whole prompt roundtrip differs')
    start=prompt.index(context);end=start+len(context);original,groups=cache_geometry(source_docs,offsets,start)
    docs=documents_from_groups(context,ids,groups)
    for d,g in zip(docs,groups):d.update({k:g[k] for k in ('source_document_ordinal','source_document_name','source_doc_char_start','source_doc_char_end')})
    require(len({d['document_id'] for d in docs})==10,'duplicate cache block token sequence')
    refs=[raw['answer'] if type(raw['answer']) is str else canonical(raw['answer'])]
    prefix,suffix=docs[0]['input_start'],len(ids)-docs[-1]['input_end']
    row=dict(source_metadata(raw,index),sample_id='loong:'+raw['id'],question=raw['question'],instruction=raw['instruction'],
        prompt_template=raw['prompt_template'],raw_answer=copy.deepcopy(raw['answer']),raw_answer_sha256=digest(raw['answer']),
        raw_answer_type=type(raw['answer']).__name__,references=refs,references_sha256=digest(refs),
        length_band=band,target_input_tokens=TARGET_INPUT_TOKENS[band],input_ids=ids,input_ids_sha256=digest(ids),input_tokens=len(ids),
        prompt=prompt,prompt_sha256=text_hash(prompt),context_text=context,context_sha256=text_hash(context),
        context_prompt_char_start=start,context_prompt_char_end=end,original_documents=original,
        document_tokens_total=sum(d['tokens'] for d in docs),context_tokens=len(ids)-prefix-suffix,prefix_tokens=prefix,suffix_tokens=suffix,
        rendered_prefix=prompt[:start],rendered_suffix=prompt[end:],natural_boundary_inputs=True,primary_context_truncated=False,
        source_context_count=len(source_docs),source_qa_count=1,source_order_preserved=True,text_roundtrip_matches_ids=True,
        prompt_transport='whole rendered prompt tokenized once; exact token IDs authoritative',
        block_boundary_kind='original_document_boundaries_and_internal_blank_paragraphs',cache_blocks_are_independent_articles=False,
        documents=docs,input_non_document_tokens=len(ids)-sum(d['tokens'] for d in docs),
        original_document_order=list(raw['doc']),official_random_shuffle_applied=False,official_length_based_document_omission_applied=False,
        support_retention='all original Loong documents/question/instruction/gold retained; only internal natural subdivisions')
    ev=dict(sample_id=row['sample_id'],prompt_sha256=row['prompt_sha256'],input_ids_sha256=row['input_ids_sha256'],
        whole_prompt_single_encoding=True,decode_matches_rendered_prompt=True,reencode_matches_input_ids=True,
        token_offsets=[list(x) for x in offsets],original_document_paragraph_boundaries=[paragraph_boundaries(d['text']) for d in source_docs],
        full_source_documents_preserved=True,full_source_question_preserved=True,full_source_instruction_preserved=True,
        full_source_template_preserved=True,additional_source_documents=0)
    return row,ev


def candidate_metadata(row):
    keys=('dataset','source_index','source_id','source_row_sha256','source_domain','source_language','source_level',
        'source_document_count','original_document_count','question_sha256','source_qa_sha256','tie_sha256','sample_id',
        'context_sha256','length_band','target_input_tokens','input_tokens','input_ids_sha256')
    return {k:row[k] for k in keys}


def choose_candidates(candidates):
    chosen={};seen=set()
    for band in LENGTH_BANDS:
        remaining=[c for c in candidates if c['length_band']==band];picked=[];domains={}
        while remaining and len(picked)<CASES_PER_BAND:
            c=min(remaining,key=lambda x:(x['source_level']==1,x['source_language']!='en',domains.get(x['source_domain'],0),
                abs(x['input_tokens']-TARGET_INPUT_TOKENS[band]),x['tie_sha256']))
            remaining.remove(c)
            if c['source_qa_sha256'] in seen:continue
            picked.append(c);seen.add(c['source_qa_sha256']);domains[c['source_domain']]=domains.get(c['source_domain'],0)+1
        require(len(picked)==CASES_PER_BAND,'insufficient distinct complete Loong QAs in '+band)
        chosen[band]=sorted(picked,key=lambda x:(-x['input_tokens'],x['tie_sha256']))
    return [chosen[b][i] for i in range(CASES_PER_BAND) for b in LENGTH_BANDS]


def fixed_policy():
    return dict(seed=SEED,datasets=list(DATASETS),evaluation_cases=20,source_qa_count=20,calibration_indices=[0,1],
        evaluation_indices=list(EVALUATION_INDICES),calibration_source=CALIBRATION_SOURCE,num_chunks=10,min_block_tokens=256,
        original_document_count_range=[5,10],length_bands=list(LENGTH_BANDS),target_input_tokens=TARGET_INPUT_TOKENS,
        input_token_ranges=INPUT_TOKEN_RANGES,cases_per_length_band=CASES_PER_BAND,source_revision=SOURCE_REVISION,
        source_jsonl_sha256=SOURCE_SHA256,document_archive_sha256=DOCUMENT_ARCHIVE_SHA256,document_catalog_sha256=DOCUMENT_CATALOG_SHA256,
        official_prompt_source_sha256=OFFICIAL_PROMPT_SOURCE_SHA256,tokenizers_version='0.21.4',inference_seed=0,
        no_padding=True,no_truncation=True,no_cross_sample_documents=True,no_cross_original_document_cache_blocks=True,
        preserve_original_order=True,single_prompt_tokenization=True,no_model_outcomes_used=True,no_answer_content_selection=True,
        gold_length_selection=False,generated_length_selection=False,minimum_generated_tokens=None,natural_eos_preserved=True,
        official_random_shuffle_applied=False,official_length_based_document_omission_applied=False,
        references_transport='string answer unchanged; other raw JSON encoded canonically for engine compatibility only',
        official_semantic_judge_executed=False,primary_metric='loong_custom_structured_exact',
        english_preferred=True,cross_document_levels_preferred=[2,3,4],domain_balanced_within_language_level_priority=True)


def build_inputs(previous_bundle,source_bytes,document_files,tokenizer,render,*,tokenizer_identity,
                 official_encoding_source,official_prompt_source,document_archive_sha256,progress=None):
    prior=reuse.validate_previous_bundle(previous_bundle);raws=source_rows(source_bytes)
    require(tokenizer_identity==prior['tokenizer'] and text_hash(official_encoding_source)==common.OFFICIAL_ENCODING_SHA256,
        'original calibration tokenizer/official chat renderer differs')
    require(text_hash(official_prompt_source)==OFFICIAL_PROMPT_SOURCE_SHA256 and
        document_archive_sha256==DOCUMENT_ARCHIVE_SHA256 and DOCUMENT_ARCHIVE_SHA256 is not None,
        'official Loong source identities differ')
    catalog=file_catalog(document_files);require(digest(catalog)==DOCUMENT_CATALOG_SHA256,'official document file inventory differs')
    eligible=[];rejected=[]
    for index,raw in enumerate(raws):
        meta=source_metadata(raw,index)
        try:row,_=build_row(raw,index,document_files,tokenizer,render)
        except (ValueError,KeyError) as exc:rejected.append(dict(meta,reason=str(exc)))
        else:eligible.append(candidate_metadata(row))
        if progress and (index%20==0 or index+1==len(raws)):progress(dict(processed=index+1,total=len(raws),eligible=len(eligible)))
    chosen=choose_candidates(eligible);rows=copy.deepcopy(prior['warmup']);calids={r['sample_id'] for r in rows}
    sources=[copy.deepcopy(r) for r in previous_bundle['source-records.json']['records'] if r['sample_id'] in calids]
    events=[copy.deepcopy(r) for r in previous_bundle['tokenization-evidence.json']['records'] if r['sample_id'] in calids]
    selected_files={}
    for pos,item in enumerate(chosen):
        raw=raws[item['source_index']];row,ev=build_row(raw,item['source_index'],document_files,tokenizer,render)
        require(candidate_metadata(row)==item,'repeat CPU tokenization changed')
        row.update(index=pos+2,split='evaluation',length_band_ordinal=pos//2+1);rows.append(row);events.append(ev)
        sources.append(dict(sample_id=row['sample_id'],dataset='loong',source_index=row['source_index'],source_row_sha256=digest(raw),raw=raw))
        for doc in row['original_documents']:selected_files[doc['source_file']]=document_files[doc['source_file']]
    source=seal(dict(schema='loong-source-records-v1',previous_inputs=previous_bundle,official_source_jsonl=source_bytes.decode('utf-8'),
        source_jsonl_sha256=SOURCE_SHA256,source_revision=SOURCE_REVISION,document_archive_sha256=document_archive_sha256,
        document_catalog=catalog,document_catalog_sha256=digest(catalog),document_files=selected_files,records=sources))
    selection=seal(dict(schema='loong-selection-plan-v1',seed=SEED,no_model_outcomes_used=True,no_answer_content_selection=True,
        gold_length_selection=False,generated_length_selection=False,eligible_candidates=eligible,rejected=rejected,
        selected_sample_ids=[r['sample_id'] for r in rows],
        selection_rule='per band: levels2/3/4 then English then least-used domain then closest actual length then fixed SHA; unique QA; descending length interleaved128k/64k',
        source_scope='all1600 pinned original rows; complete actual inputs only; no output-dependent selection',
        independent_model_training_holdout_claim=False))
    tokens=seal(dict(schema='loong-tokenization-evidence-v1',tokenizer=tokenizer_identity,
        official_encoding_source=official_encoding_source,official_prompt_source=official_prompt_source,records=events))
    protocol=seal(dict(schema=INPUT_PROTOCOL_SCHEMA,**fixed_policy(),tokenizer=tokenizer_identity,previous_manifest_sha256=prior['sha256'],
        source_records_sha256=source['sha256'],selection_plan_sha256=selection['sha256'],tokenization_evidence_sha256=tokens['sha256']))
    companions=dict(zip(INPUT_COMPANIONS,(source,selection,tokens)))
    manifest=seal(dict(schema=MANIFEST_SCHEMA,contract_schema=CONTRACT_SCHEMA,artifact_kind='loong_inputs_v1',natural_boundary_inputs=True,
        input_protocol_sha256=protocol['sha256'],tokenizer=tokenizer_identity,num_chunks=10,chunk_tokens=None,context_tokens=None,
        calibration_source=CALIBRATION_SOURCE,warmup=rows[:2],cohort=rows[2:],
        sources=[dict(dataset='loong',revision=SOURCE_REVISION,source_jsonl_sha256=SOURCE_SHA256,document_archive_sha256=document_archive_sha256)],
        companion_sha256={k:v['sha256'] for k,v in companions.items()},selection_plan_sha256=selection['sha256']))
    summary=seal(dict(schema='loong-input-preparation-v1',status='complete',cuda_used=False,model_results_used=False,
        manifest_sha256=manifest['sha256'],input_protocol_sha256=protocol['sha256'],evaluation_cases=20,source_qa_count=20,
        eligible_counts={b:sum(c['length_band']==b for c in eligible) for b in LENGTH_BANDS},
        samples=[dict(index=r['index'],dataset=r['dataset'],source_index=r['source_index'],source_id=r['source_id'],
            length_band=r.get('length_band'),length_band_ordinal=r.get('length_band_ordinal'),target_input_tokens=r.get('target_input_tokens'),
            input_tokens=r['input_tokens'],block_tokens=[d['tokens'] for d in r['documents']],
            source_domain=r.get('source_domain'),source_language=r.get('source_language'),source_level=r.get('source_level'),
            original_document_count=r.get('original_document_count'),source_context_preserved=True,whole_prompt_roundtrip=True) for r in rows]))
    return dict(companions,**{'manifest.json':manifest,'input-protocol.json':protocol,'preparation.json':summary})


def validate_inputs(base,manifest,protocol,*,input_dir):
    verify(manifest);verify(protocol)
    require(manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('contract_schema')==CONTRACT_SCHEMA
        and manifest.get('artifact_kind')=='loong_inputs_v1' and protocol.get('schema')==INPUT_PROTOCOL_SCHEMA
        and manifest['input_protocol_sha256']==protocol['sha256'],'Loong input schema/binding differs')
    require(all(k in protocol and protocol[k]==v for k,v in fixed_policy().items()),'Loong fixed policy differs')
    companions={n:read(Path(input_dir)/n) for n in INPUT_COMPANIONS}
    require(set(manifest['companion_sha256'])==set(INPUT_COMPANIONS),'companion inventory differs')
    for n,v in companions.items():verify(v);require(manifest['companion_sha256'][n]==v['sha256'],'companion seal differs')
    source,selection,tokens=[companions[n] for n in INPUT_COMPANIONS]
    require([v.get('schema') for v in (source,selection,tokens)]==['loong-source-records-v1','loong-selection-plan-v1','loong-tokenization-evidence-v1'],'companion schema differs')
    for key,v in (('source_records_sha256',source),('selection_plan_sha256',selection),('tokenization_evidence_sha256',tokens)):
        require(protocol[key]==v['sha256'],'provenance binding differs')
    prior=reuse.validate_previous_bundle(source['previous_inputs']);raws=source_rows(source['official_source_jsonl'].encode('utf-8'))
    require(source['source_jsonl_sha256']==SOURCE_SHA256 and source['source_revision']==SOURCE_REVISION
        and source['document_archive_sha256']==DOCUMENT_ARCHIVE_SHA256
        and source['document_catalog_sha256']==digest(source['document_catalog'])==DOCUMENT_CATALOG_SHA256,
        'pinned source/doc archive identities differ')
    files=source['document_files'];catalog=source['document_catalog']
    require(all(name in catalog and file_catalog({name:text})[name]==catalog[name] for name,text in files.items()),
        'selected original document bytes differ')
    require(manifest['tokenizer']==protocol['tokenizer']==tokens['tokenizer']==prior['tokenizer']
        and tokens['tokenizer']['encoding_sha256']==common.OFFICIAL_ENCODING_SHA256
        and protocol['previous_manifest_sha256']==prior['sha256'],'actual tokenizer/calibration identity differs')
    require(text_hash(tokens['official_prompt_source'])==OFFICIAL_PROMPT_SOURCE_SHA256,'official Loong renderer source differs')
    render=common.frozen_renderer(tokens['official_encoding_source']);rows=manifest['warmup']+manifest['cohort']
    require(manifest['warmup']==prior['warmup'] and [r['index'] for r in rows]==list(range(22)) and len(manifest['cohort'])==20
        and manifest['num_chunks']==10 and manifest['natural_boundary_inputs'] is True
        and manifest['calibration_source']==CALIBRATION_SOURCE and manifest['selection_plan_sha256']==selection['sha256'],
        'exact calibration2/evaluation20 geometry differs')
    require(manifest['sources']==[dict(dataset='loong',revision=SOURCE_REVISION,source_jsonl_sha256=SOURCE_SHA256,
        document_archive_sha256=DOCUMENT_ARCHIVE_SHA256)],'source inventory differs')
    require(selection['seed']==SEED and all(selection[k] is True for k in ('no_model_outcomes_used','no_answer_content_selection'))
        and all(selection[k] is False for k in ('gold_length_selection','generated_length_selection')),'selection policy differs')
    candidates,rejected=selection['eligible_candidates'],selection['rejected']
    require(sorted(x['source_index'] for x in candidates+rejected)==list(range(len(raws))),'candidate source coverage differs')
    for item in candidates+rejected:
        require(all(item.get(k)==v for k,v in source_metadata(raws[item['source_index']],item['source_index']).items()),'candidate source identity differs')
    for item in candidates:
        require(5<=item['source_document_count']<=10 and classify_length(item['input_tokens'])==item['length_band']
            and item['target_input_tokens']==TARGET_INPUT_TOKENS[item['length_band']],'candidate geometry differs')
    require(candidates==sorted(candidates,key=lambda x:x['source_index']),'candidate ordering differs')
    chosen=choose_candidates(candidates)
    require(selection['selected_sample_ids']==[r['sample_id'] for r in rows]
        and [r['sample_id'] for r in rows[2:]]==[c['sample_id'] for c in chosen],'selected original QA/order differs')
    records={x['sample_id']:x for x in source['records']};events={x['sample_id']:x for x in tokens['records']}
    require(len(records)==len(source['records'])==len(events)==len(tokens['records'])==22
        and set(records)==set(events)=={r['sample_id'] for r in rows},'source/token record inventory differs')
    oldsrc={x['sample_id']:x for x in source['previous_inputs']['source-records.json']['records']}
    oldev={x['sample_id']:x for x in source['previous_inputs']['tokenization-evidence.json']['records']}
    for r in rows[:2]:require(records[r['sample_id']]==oldsrc[r['sample_id']] and events[r['sample_id']]==oldev[r['sample_id']],'calibration evidence changed')
    used_files=set()
    for pos,(row,item) in enumerate(zip(rows[2:],chosen)):
        raw=raws[item['source_index']];ids=row['input_ids'];ev=events[row['sample_id']]
        require(row['length_band']==LENGTH_BANDS[pos%2] and row['length_band_ordinal']==pos//2+1 and row['split']=='evaluation'
            and candidate_metadata(row)==item,'case actual length/order/source differs')
        source_docs=original_documents(raw,files);used_files.update(d['source_file'] for d in source_docs)
        prompt,context=render_original(raw,source_docs,render);start=prompt.index(context);end=start+len(context)
        refs=[raw['answer'] if type(raw['answer']) is str else canonical(raw['answer'])]
        require(row['sample_id']=='loong:'+raw['id'] and row['question']==raw['question'] and row['instruction']==raw['instruction']
            and row['prompt_template']==raw['prompt_template'] and row['raw_answer']==raw['answer']
            and row['raw_answer_sha256']==digest(raw['answer']) and row['raw_answer_type']==type(raw['answer']).__name__
            and row['references']==refs and row['references_sha256']==digest(refs)
            and records[row['sample_id']]==dict(sample_id=row['sample_id'],dataset='loong',source_index=row['source_index'],source_row_sha256=digest(raw),raw=raw),
            'original question/instruction/template/gold changed')
        require(row['prompt']==prompt and row['context_text']==context and row['context_sha256']==text_hash(context)
            and row['context_prompt_char_start']==start and row['context_prompt_char_end']==end
            and row['prompt_sha256']==ev['prompt_sha256']==text_hash(prompt) and row['input_tokens']==len(ids)
            and row['input_ids_sha256']==ev['input_ids_sha256']==digest(ids)
            and all(type(t) is int and 0<=t<2**31 for t in ids),'whole original prompt/token identity differs')
        require(len(ev['token_offsets'])==len(ids) and ev['additional_source_documents']==0
            and all(ev[k] is True for k in ('whole_prompt_single_encoding','decode_matches_rendered_prompt','reencode_matches_input_ids',
                'full_source_documents_preserved','full_source_question_preserved','full_source_instruction_preserved','full_source_template_preserved')),
            'whole prompt token evidence differs')
        require(ev['original_document_paragraph_boundaries']==[paragraph_boundaries(d['text']) for d in source_docs], 'original paragraph boundaries differ')
        original,groups=cache_geometry(source_docs,ev['token_offsets'],start);docs=documents_from_groups(context,ids,groups)
        for d,g in zip(docs,groups):d.update({k:g[k] for k in ('source_document_ordinal','source_document_name','source_doc_char_start','source_doc_char_end')})
        require(row['original_documents']==original and row['documents']==docs
            and len({d['document_id'] for d in docs})==10 and all(d['tokens']>=256 for d in docs)
            and ''.join(context[d['context_char_start']:d['context_char_end']] for d in docs)==context,
            'ten complete blocks/original document boundaries differ')
        require(row['natural_boundary_inputs'] is True and row['primary_context_truncated'] is False
            and row['source_context_count']==row['source_document_count']==row['original_document_count']==len(source_docs)
            and row['source_qa_count']==1 and row['source_order_preserved'] is True and row['text_roundtrip_matches_ids'] is True
            and row['cache_blocks_are_independent_articles'] is False
            and row['block_boundary_kind']=='original_document_boundaries_and_internal_blank_paragraphs'
            and row['original_document_order']==raw['doc'] and row['official_random_shuffle_applied'] is False
            and row['official_length_based_document_omission_applied'] is False,'input preservation claims differ')
        require(row['document_tokens_total']==sum(d['tokens'] for d in docs)
            and row['input_non_document_tokens']==len(ids)-row['document_tokens_total']
            and row['prefix_tokens']==docs[0]['input_start'] and row['suffix_tokens']==len(ids)-docs[-1]['input_end']
            and row['context_tokens']==len(ids)-row['prefix_tokens']-row['suffix_tokens']
            and row['rendered_prefix']==prompt[:start] and row['rendered_suffix']==prompt[end:],'native gap accounting differs')
    require(set(files)==used_files,'unused/unbound document source payload')
    require(rows[2]['input_tokens']==max(r['input_tokens'] for r in rows[2:]),'case2 must be longest qualification input')
    if base is not None:
        base.validate_manifest(manifest)
        for row in rows:base.document_specs(row)
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-input-dir','source-jsonl','doc-dir','document-archive','prompt-source','tokenizer','official-encoding','output-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args(argv);require(not args.output_dir.exists(),'never overwrite input preparation')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import importlib.metadata
    from tokenizers import Tokenizer
    require(importlib.metadata.version('tokenizers')=='0.21.4','exact tokenizers0.21.4 required')
    bundle={n:read(args.previous_input_dir/n) for n in PREVIOUS_FILES}
    common.validate_inputs(None,bundle['manifest.json'],bundle['input-protocol.json'],input_dir=args.previous_input_dir)
    identity=bundle['manifest.json']['tokenizer'];require(file_hash(args.tokenizer)==identity['tokenizer_json_sha256'],'actual1017 tokenizer differs')
    encoder=args.official_encoding.read_text();files=archive_document_files(args.document_archive,args.doc_dir)
    import sys
    result=build_inputs(bundle,args.source_jsonl.read_bytes(),files,Tokenizer.from_file(str(args.tokenizer)),common.frozen_renderer(encoder),
        tokenizer_identity=identity,official_encoding_source=encoder,official_prompt_source=args.prompt_source.read_text(),
        document_archive_sha256=file_hash(args.document_archive),progress=lambda r:print(json.dumps(r),file=sys.stderr,flush=True))
    args.output_dir.mkdir(parents=True)
    for n,v in result.items():publish(args.output_dir/n,v)
    validate_inputs(None,result['manifest.json'],result['input-protocol.json'],input_dir=args.output_dir)
    print(json.dumps(result['preparation.json'],ensure_ascii=False))


if __name__=='__main__':main()
