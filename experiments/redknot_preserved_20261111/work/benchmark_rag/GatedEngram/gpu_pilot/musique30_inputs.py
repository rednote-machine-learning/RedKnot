"""Whole native LongBench QA inputs with ten natural, variable cache spans.

No model is imported. Full prompts are rendered once and tokenized once; natural
character boundaries only select cache rows, never rewrite the input sequence.
"""
from __future__ import annotations
import argparse
import bisect
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

DATASETS=('musique',)
EVALUATION_INDICES=tuple(range(2,32))
NUM_CHUNKS=10
SOURCE_SHA256='4ac69b91281c4ec6b21316cb7282e83fb6b4dda04fc68480bb8d8ed1e19ff7bd'
ORIGINAL_MANIFEST_SHA256='c430e8a69cc55b76f38ac03fb761c8c50d999d759af45db1597bf85266d70c57'
INPUT_COMPANIONS=('source-records.json','selection-plan.json','tokenization-evidence.json')
MANIFEST_SCHEMA='di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA='kv-cache-engram-musique30-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA='musique30-input-protocol-v1'
SEED=20261106
MIN_BLOCK_TOKENS=256
OFFICIAL_ENCODING_SHA256='13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2'

def require(ok,message):
    if not ok: raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def text_hash(value):return hashlib.sha256(value.encode()).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seal(value):return dict(value,sha256=digest(value))
def verify(value):
    require(value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'input evidence seal mismatch')
def read(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular input evidence required')
    return json.loads(p.read_text())
def publish(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False);f.write('\n')

def natural_boundaries(context):
    matches=list(re.finditer(r'(?m)^Passage\s+(\d+)\s*:\s*\n',context))
    require(len(matches)>=10,'at least ten original Passage boundaries required')
    require(matches[0].start()==0 and [int(m[1]) for m in matches]==list(range(1,len(matches)+1)),
        'original Passage numbering must start at1 and remain consecutive')
    return [m.start() for m in matches]+[len(context)]

def offset_arrays(offsets):
    starts=[a for a,b in offsets];ends=[b for a,b in offsets]
    require(starts==sorted(starts) and ends==sorted(ends),'token offsets must be monotonic')
    require(all(type(a) is int and type(b) is int and 0<=a<b for a,b in offsets),'nonempty original token offsets required')
    return starts,ends

def exact_token_span(offsets,char_start,char_end,*,arrays=None):
    starts,ends=offset_arrays(offsets) if arrays is None else arrays
    left=bisect.bisect_left(starts,char_start);right=bisect.bisect_right(ends,char_end)
    require(0<=left<right<=len(offsets),'empty natural token span')
    require(char_start<=starts[left] and ends[right-1]<=char_end,'cached token crosses natural character boundary')
    return left,right

def partition_ten(boundaries,offsets,context_start):
    """Globally minimize squared length imbalance, preserving all source text."""
    spans={};arrays=offset_arrays(offsets)
    for a in range(len(boundaries)-1):
        for b in range(a+1,len(boundaries)):
            left,right=exact_token_span(offsets,context_start+boundaries[a],context_start+boundaries[b],arrays=arrays)
            if right-left>=MIN_BLOCK_TOKENS:spans[a,b]=(left,right)
    first,last=exact_token_span(offsets,context_start,context_start+boundaries[-1],arrays=arrays);total=last-first
    dp={(0,0):(0,())}
    for count in range(1,11):
        for end in range(count,len(boundaries)):
            choices=[]
            for begin in range(count-1,end):
                if (count-1,begin) not in dp or (begin,end) not in spans:continue
                cost,cuts=dp[count-1,begin];left,right=spans[begin,end]
                choices.append((cost+(10*(right-left)-total)**2,cuts+(end,)))
            if choices:dp[count,end]=min(choices)
    require((10,len(boundaries)-1) in dp,'no ten natural blocks with minimum256 tokens')
    selected=dp[10,len(boundaries)-1][1];answer=[];begin=0
    for end in selected:
        left,right=spans[begin,end];answer.append(dict(char_start=boundaries[begin],char_end=boundaries[end],input_start=left,input_end=right,passage_start=begin,passage_end=end));begin=end
    return answer

def build_row(raw,dataset,source_index,tokenizer,encode_messages,template):
    require(dataset=='musique' and raw.get('dataset','musique')=='musique','MuSiQue source required')
    require(type(raw.get('answers')) is list and raw['answers'] and all(type(a) is str and a for a in raw['answers']),'original nonempty gold required')
    context=raw['context'];question=raw['input']
    require(type(context) is str and context and type(question) is str and question,'nonempty original QA required')
    user_text=template.format(context=context,input=question)
    rendered=encode_messages([dict(role='user',content=user_text)],thinking_mode='chat')
    require(rendered.count(context)==1 and rendered.count(question)==1,'source context/question must occur exactly once')
    encoding=tokenizer.encode(rendered,add_special_tokens=False);ids=encoding.ids;offsets=encoding.offsets
    require(tokenizer.decode(ids,skip_special_tokens=False)==rendered,'full prompt decode must reproduce exact rendered text')
    require(tokenizer.encode(tokenizer.decode(ids,skip_special_tokens=False),add_special_tokens=False).ids==ids,'whole prompt roundtrip differs')
    context_start=rendered.index(context);boundaries=natural_boundaries(context)
    groups=partition_ten(boundaries,offsets,context_start)
    documents=[]
    for i,group in enumerate(groups):
        left,right=group['input_start'],group['input_end'];sha=digest(ids[left:right])
        documents.append(dict(document_index=i,document_id=sha,token_ids_sha256=sha,tokens=right-left,
            input_start=left,input_end=right,offline_position=0,online_position=left,
            context_char_start=group['char_start'],context_char_end=group['char_end'],
            source_text_sha256=text_hash(context[group['char_start']:group['char_end']]),
            passage_start=group['passage_start'],passage_end=group['passage_end']))
    require(len({d['document_id'] for d in documents})==10,'duplicate cache block token sequence')
    prefix=documents[0]['input_start'];suffix=len(ids)-documents[-1]['input_end']
    row=dict(sample_id=dataset+':'+raw['_id'],dataset=dataset,source_index=source_index,source_line=source_index+1,source_id=raw['_id'],
        source_row_sha256=digest(raw),question=question,references=raw['answers'],references_sha256=digest(raw['answers']),
        input_ids=ids,input_ids_sha256=digest(ids),input_tokens=len(ids),prompt=rendered,prompt_sha256=text_hash(rendered),
        context_text=context,context_sha256=text_hash(context),context_prompt_char_start=context_start,
        context_prompt_char_end=context_start+len(context),original_context_tokens=len(tokenizer.encode(context,add_special_tokens=False).ids),
        context_tokens=len(ids)-prefix-suffix,document_tokens_total=sum(d['tokens'] for d in documents),
        prefix_tokens=prefix,suffix_tokens=suffix,rendered_prefix=rendered[:context_start],rendered_suffix=rendered[context_start+len(context):],
        primary_context_truncated=False,natural_boundary_inputs=True,source_context_count=1,source_order_preserved=True,
        text_roundtrip_matches_ids=True,prompt_transport='whole rendered prompt tokenized once; exact token IDs authoritative',
        block_boundary_kind='original_Passage_start',documents=documents,
        input_non_document_tokens=len(ids)-sum(d['tokens'] for d in documents),
        support_retention='entire single original QA context unchanged; original question and references unchanged')
    evidence=dict(sample_id=row['sample_id'],prompt_sha256=row['prompt_sha256'],input_ids_sha256=row['input_ids_sha256'],
        whole_prompt_single_encoding=True,decode_matches_rendered_prompt=True,reencode_matches_input_ids=True,
        token_offsets=[list(x) for x in offsets],original_passage_char_boundaries=boundaries,
        full_source_context_preserved=True,full_source_question_preserved=True,additional_source_contexts=0)
    return row,evidence

@functools.lru_cache(maxsize=2)
def frozen_renderer(source):
    require(text_hash(source)==OFFICIAL_ENCODING_SHA256,'unrecognized official chat encoder source')
    namespace={};exec(compile(source,'<pinned-official-v41-encoding>','exec'),namespace)
    return namespace['encode_messages']


def selection_key(item):
    return (-item['input_tokens'],item['tie_sha256'])


def candidate_metadata(row):
    return dict(source_index=row['source_index'],source_id=row['source_id'],
        sample_id=row['sample_id'],source_row_sha256=row['source_row_sha256'],
        context_sha256=row['context_sha256'],question_sha256=text_hash(row['question']),
        input_tokens=row['input_tokens'],input_ids_sha256=row['input_ids_sha256'],
        tie_sha256=digest(dict(seed=SEED,dataset='musique',source_id=row['source_id'],
            context_sha256=row['context_sha256'])))


def choose_candidates(candidates,use_original_calibration):
    """Length only, then seed/source SHA; gold and model outputs are never read."""
    ordered=sorted(candidates,key=selection_key);unique=[];contexts=set();questions=set()
    for item in ordered:
        if item['context_sha256'] in contexts or item['question_sha256'] in questions:continue
        unique.append(item);contexts.add(item['context_sha256']);questions.add(item['question_sha256'])
    required=30 if use_original_calibration else 32
    require(len(unique)>=required,'insufficient distinct natural10 MuSiQue sources')
    if not use_original_calibration:
        require(unique[0]['input_tokens']>=16384,'MuSiQue calibration requires actual prompt>=16384 tokens')
    return unique[:required]


def load_original_calibration(directory):
    from gpu_pilot import natural_boundary_inputs as old
    directory=Path(directory);manifest=read(directory/'manifest.json');protocol=read(directory/'input-protocol.json')
    old.validate_inputs(None,manifest,protocol,input_dir=directory)
    require(manifest['sha256']==ORIGINAL_MANIFEST_SHA256,'only exact original S1 calibration may be reused')
    source=read(directory/'source-records.json');tokens=read(directory/'tokenization-evidence.json')
    raw={r['sample_id']:r for r in source['records']};ev={r['sample_id']:r for r in tokens['records']}
    return dict(manifest=manifest,tokenizer=manifest['tokenizer'],
        source_records=[raw[r['sample_id']] for r in manifest['warmup']],
        evidence=[ev[r['sample_id']] for r in manifest['warmup']])


def build_inputs(source_bytes,tokenizer,encode_messages,*,tokenizer_identity,
                 official_encoding_source,official_prompt_file_content,official_prompt_source,
                 calibration_input_dir=None):
    """Pure CPU builder. Return named JSON objects; caller publishes once.

    Every eligible source is tokenized with the actual checkpoint tokenizer.
    The exact original JSONL bytes are embedded, allowing an independent reader
    to bind selected raw rows to the pinned acquisition without network access.
    """
    require(hashlib.sha256(source_bytes).hexdigest()==SOURCE_SHA256,'MuSiQue acquisition bytes changed')
    require(text_hash(official_encoding_source)==OFFICIAL_ENCODING_SHA256,'official renderer source changed')
    require(text_hash(official_prompt_file_content)==official_prompt_source['sha256'],
        'official LongBench prompt source changed')
    templates=json.loads(official_prompt_file_content);require('musique' in templates,'MuSiQue template absent')
    raw_rows=[json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    require(len(raw_rows)==200 and len({r['_id'] for r in raw_rows})==200,'pinned MuSiQue200 acquisition required')
    eligible=[];lookup={};rejected=[]
    for index,raw in enumerate(raw_rows):
        try:
            row,ev=build_row(raw,'musique',index,tokenizer,encode_messages,templates['musique'])
            require(row['input_tokens']<270336,'original prompt exceeds engine context capacity')
        except ValueError as exc:
            rejected.append(dict(source_index=index,source_id=raw['_id'],source_row_sha256=digest(raw),reason=str(exc)))
            continue
        meta=candidate_metadata(row);eligible.append(meta);lookup[row['sample_id']]=(row,ev,raw)
    # Prefer MuSiQue calibrations. The original two are only an explicit fallback
    # when the actual MuSiQue input geometry cannot supply the native16K probe.
    original=not(len(eligible)>=32 and max(x['input_tokens'] for x in eligible)>=16384)
    chosen=choose_candidates(eligible,original)
    rows=[];source_records=[];evidence=[];origin=None
    if original:
        require(calibration_input_dir is not None,'provide --calibration-input-dir: MuSiQue cannot supply native16K calibration')
        origin=load_original_calibration(calibration_input_dir)
        require(origin['tokenizer']==tokenizer_identity,'original calibration tokenizer/prompt renderer changed')
        rows.extend(origin['manifest']['warmup']);source_records.extend(origin['source_records']);evidence.extend(origin['evidence'])
    for ordinal,item in enumerate(chosen):
        row,ev,raw=lookup[item['sample_id']];row=dict(row)
        index=ordinal+2 if original else ordinal
        row.update(index=index,split='calibration' if index<2 else 'evaluation')
        rows.append(row);evidence.append(ev)
        source_records.append(dict(sample_id=row['sample_id'],dataset='musique',source_index=row['source_index'],
            source_row_sha256=digest(raw),raw=raw))
    require(len(rows)==32 and len({r['context_sha256'] for r in rows})==32
        and len({text_hash(r['question']) for r in rows})==32,'calibration/evaluation sources must be distinct')
    calibration_source='original_s1_warmup_exact' if original else 'musique_longest_two'
    source=seal(dict(schema='musique30-source-records-v1',files=[dict(dataset='musique',
        acquisition_sha256=SOURCE_SHA256,bytes=len(source_bytes),records=200)],
        original_source_jsonl=source_bytes.decode('utf-8'),records=source_records,
        calibration_origin=dict(manifest=origin['manifest']) if origin else None))
    selection=seal(dict(schema='musique30-selection-plan-v1',seed=SEED,no_model_outcomes_used=True,
        no_answer_content_selection=True,composition='single_original_qa_ten_natural_cache_blocks',
        evaluation_selection='descending actual complete prompt tokens; ascending seeded source SHA tie-break',
        calibration_source=calibration_source,calibration_indices=[0,1],evaluation_indices=list(EVALUATION_INDICES),
        eligible_candidates=sorted(eligible,key=selection_key),rejected=rejected,
        selected_sample_ids=[r['sample_id'] for r in rows],independent_holdout_claim=False,
        block_partition='minimum squared token-length imbalance among complete contiguous Passage groups; lexical cut tie-break'))
    used_datasets=sorted({r['dataset'] for r in rows})
    tokens=seal(dict(schema='musique30-tokenization-evidence-v1',tokenizer=tokenizer_identity,
        official_prompt_source=official_prompt_source,official_prompt_file_content=official_prompt_file_content,
        official_encoding_source=official_encoding_source,
        official_prompt_templates={d:templates[d] for d in used_datasets},records=evidence))
    protocol=seal(dict(schema=INPUT_PROTOCOL_SCHEMA,seed=SEED,
        composition='single_original_qa_ten_natural_cache_blocks',datasets=['musique'],
        calibration_indices=[0,1],evaluation_indices=list(EVALUATION_INDICES),calibration_source=calibration_source,
        num_chunks=10,min_block_tokens=256,equal_document_lengths_required=False,
        no_padding=True,no_truncation=True,no_cross_sample_contexts=True,preserve_original_order=True,
        single_prompt_tokenization=True,no_model_outcomes_used=True,no_answer_content_selection=True,
        source_records_sha256=source['sha256'],selection_plan_sha256=selection['sha256'],
        tokenization_evidence_sha256=tokens['sha256'],tokenizer=tokenizer_identity,
        tokenizers_version='0.21.4',inference_seed=0))
    companions=dict(zip(INPUT_COMPANIONS,(source,selection,tokens)))
    manifest=seal(dict(schema=MANIFEST_SCHEMA,contract_schema=CONTRACT_SCHEMA,artifact_kind='musique30_inputs_v1',
        natural_boundary_inputs=True,input_protocol_sha256=protocol['sha256'],tokenizer=tokenizer_identity,
        num_chunks=10,chunk_tokens=None,context_tokens=None,calibration_source=calibration_source,
        sources=source['files'],warmup=rows[:2],cohort=rows[2:],
        companion_sha256={k:v['sha256'] for k,v in companions.items()},selection_plan_sha256=selection['sha256']))
    summary=seal(dict(schema='musique30-input-preparation-v1',status='complete',cuda_used=False,
        model_results_used=False,manifest_sha256=manifest['sha256'],input_protocol_sha256=protocol['sha256'],
        source_file_sha256=SOURCE_SHA256,calibration_source=calibration_source,eligible_count=len(eligible),
        samples=[dict(index=r['index'],dataset=r['dataset'],source_index=r['source_index'],
            input_tokens=r['input_tokens'],block_tokens=[d['tokens'] for d in r['documents']],
            source_context_preserved=True,whole_prompt_roundtrip=True) for r in rows]))
    return dict(companions,**{'manifest.json':manifest,'input-protocol.json':protocol,'preparation.json':summary})


def validate_inputs(base,manifest,protocol,*,input_dir):
    """No Torch/tokenizers import; independently audit frozen geometry and source."""
    verify(manifest);verify(protocol)
    require(manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('contract_schema')==CONTRACT_SCHEMA
        and manifest.get('artifact_kind')=='musique30_inputs_v1' and manifest.get('num_chunks')==10
        and protocol.get('schema')==INPUT_PROTOCOL_SCHEMA and protocol.get('num_chunks')==10
        and manifest.get('input_protocol_sha256')==protocol['sha256'],'MuSiQue input contract differs')
    require(protocol.get('composition')=='single_original_qa_ten_natural_cache_blocks'
        and protocol.get('min_block_tokens')==256 and protocol.get('datasets')==['musique']
        and protocol.get('evaluation_indices')==list(EVALUATION_INDICES)
        and protocol.get('calibration_indices')==[0,1] and protocol.get('seed')==SEED
        and protocol.get('tokenizers_version')=='0.21.4' and protocol.get('inference_seed')==0,
        'MuSiQue fixed selection geometry differs')
    for flag in ('no_padding','no_truncation','no_cross_sample_contexts','preserve_original_order',
                 'single_prompt_tokenization','no_model_outcomes_used','no_answer_content_selection'):
        require(protocol.get(flag) is True,'source preservation/selection claim differs: '+flag)
    companions={name:read(Path(input_dir)/name) for name in INPUT_COMPANIONS}
    require(set(manifest['companion_sha256'])==set(INPUT_COMPANIONS),'companion inventory differs')
    for name,value in companions.items():verify(value);require(manifest['companion_sha256'][name]==value['sha256'],'companion binding differs')
    source,selection,tokens=[companions[n] for n in INPUT_COMPANIONS]
    for key,value in [('source_records_sha256',source),('selection_plan_sha256',selection),('tokenization_evidence_sha256',tokens)]:
        require(protocol[key]==value['sha256'],'input provenance chain differs')
    require(manifest['tokenizer']==tokens['tokenizer']==protocol['tokenizer']
        and manifest['tokenizer']['encoding_sha256']==OFFICIAL_ENCODING_SHA256
        and manifest['tokenizer']['tokenizers_version']=='0.21.4','tokenizer identity differs')
    require(source.get('schema')=='musique30-source-records-v1' and selection.get('schema')=='musique30-selection-plan-v1'
        and tokens.get('schema')=='musique30-tokenization-evidence-v1','MuSiQue companion schema differs')
    raw_bytes=source['original_source_jsonl'].encode('utf-8')
    require(hashlib.sha256(raw_bytes).hexdigest()==SOURCE_SHA256,'original acquisition bytes changed')
    raw_rows=[json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    require(len(raw_rows)==200 and source['files']==[dict(dataset='musique',acquisition_sha256=SOURCE_SHA256,
        bytes=len(raw_bytes),records=200)] and manifest['sources']==source['files'],'raw acquisition inventory differs')
    require(text_hash(tokens['official_prompt_file_content'])==tokens['official_prompt_source']['sha256']
        ==manifest['tokenizer']['official_longbench_prompt_sha256'],'official prompt source differs')
    templates=json.loads(tokens['official_prompt_file_content']);render=frozen_renderer(tokens['official_encoding_source'])
    rows=manifest['warmup']+manifest['cohort']
    require(len(manifest['warmup'])==2 and len(manifest['cohort'])==30
        and [r['index'] for r in rows]==list(range(32))
        and all(r['dataset']=='musique' and r['split']=='evaluation' for r in manifest['cohort']),
        'exact calibration2/evaluation30 MuSiQue required')
    require(tokens['official_prompt_templates']=={d:templates[d] for d in sorted({r['dataset'] for r in rows})},'selected task templates differ')
    require(selection.get('seed')==SEED and selection.get('no_model_outcomes_used') is True
        and selection.get('no_answer_content_selection') is True
        and selection.get('evaluation_indices')==list(EVALUATION_INDICES)
        and selection.get('selected_sample_ids')==[r['sample_id'] for r in rows], 'selection outcome/order differs')
    candidates=selection['eligible_candidates'];rejected=selection['rejected']
    indices=[v['source_index'] for v in candidates+rejected]
    require(sorted(indices)==list(range(200)),'every original source must appear once in eligibility inventory')
    for item in candidates+rejected:
        raw=raw_rows[item['source_index']]
        require(item['source_id']==raw['_id'] and item['source_row_sha256']==digest(raw),'source eligibility attribution differs')
    for item in candidates:
        raw=raw_rows[item['source_index']]
        require(type(item['input_tokens']) is int and 0<item['input_tokens']<270336
            and item['sample_id']=='musique:'+raw['_id'] and item['context_sha256']==text_hash(raw['context'])
            and item['question_sha256']==text_hash(raw['input'])
            and item['tie_sha256']==digest(dict(seed=SEED,dataset='musique',source_id=raw['_id'],context_sha256=text_hash(raw['context']))),
            'source ranking inputs differ')
    require(candidates==sorted(candidates,key=selection_key),'candidate ranking changed')
    original=selection['calibration_source']=='original_s1_warmup_exact'
    require(selection['calibration_source'] in ('original_s1_warmup_exact','musique_longest_two')
        and manifest['calibration_source']==protocol['calibration_source']==selection['calibration_source'],
        'calibration source differs')
    require(original==(not(len(candidates)>=32 and max(c['input_tokens'] for c in candidates)>=16384)),
        'original calibration fallback chosen despite eligible MuSiQue native16K source')
    chosen=choose_candidates(candidates,original)
    require([x['sample_id'] for x in chosen]==[r['sample_id'] for r in (manifest['cohort'] if original else rows)],
        'selected cases are not the declared longest actual prompts')
    if original:
        origin=source['calibration_origin']['manifest'];verify(origin)
        require(origin['sha256']==ORIGINAL_MANIFEST_SHA256 and manifest['warmup']==origin['warmup']
            and origin['tokenizer']==manifest['tokenizer'],'original calibration rows changed')
    else:require(source['calibration_origin'] is None,'foreign calibration evidence')
    raw_lookup={r['sample_id']:r for r in source['records']};evidence={e['sample_id']:e for e in tokens['records']}
    require(len(source['records'])==len(tokens['records'])==len(raw_lookup)==len(evidence)==32
        and len({r['context_sha256'] for r in rows})==len({text_hash(r['question']) for r in rows})==32,
        'duplicate/missing source, question, context or token evidence')
    if base is not None:base.validate_manifest(manifest)
    for row in rows:
        rec=raw_lookup[row['sample_id']];raw=rec['raw'];ev=evidence[row['sample_id']];ids=row['input_ids']
        require(row['source_row_sha256']==rec['source_row_sha256']==digest(raw)
            and row['source_index']==rec['source_index'] and row['source_id']==raw['_id']
            and row['sample_id']==row['dataset']+':'+raw['_id'] and rec['dataset']==row['dataset']
            and row['question']==raw['input'] and row['references']==raw['answers']
            and row['context_text']==raw['context'] and row['context_sha256']==text_hash(raw['context']), 'original QA changed')
        if row['dataset']=='musique':require(raw==raw_rows[row['source_index']], 'selected QA does not match pinned JSONL row')
        require(row['input_tokens']==len(ids) and row['input_ids_sha256']==digest(ids)
            and row['references_sha256']==digest(raw['answers']) and all(type(t) is int and 0<=t<2**31 for t in ids),
            'token/gold identity differs')
        if row['dataset']=='musique':
            require(candidate_metadata(row) in candidates,'selected actual token count/hash detached from ranking')
        prompt=row['prompt'];start=row['context_prompt_char_start'];end=row['context_prompt_char_end']
        expected=render([dict(role='user',content=templates[row['dataset']].format(context=raw['context'],input=raw['input']))],thinking_mode='chat')
        require(prompt==expected and prompt[start:end]==raw['context'] and end==start+len(raw['context'])
            and prompt.count(raw['context'])==prompt.count(raw['input'])==1
            and text_hash(prompt)==row['prompt_sha256']==ev['prompt_sha256']
            and row['input_ids_sha256']==ev['input_ids_sha256'],'whole original prompt/template differs')
        require(row['natural_boundary_inputs'] is True and row['primary_context_truncated'] is False
            and row['source_order_preserved'] is True and row['source_context_count']==1
            and row['text_roundtrip_matches_ids'] is True,'source transformation detected')
        offsets=ev['token_offsets'];require(len(offsets)==len(ids),'token offset count differs')
        for flag in ('whole_prompt_single_encoding','decode_matches_rendered_prompt','reencode_matches_input_ids',
                     'full_source_context_preserved','full_source_question_preserved'):
            require(ev[flag] is True,'whole prompt tokenization not proven')
        require(ev['additional_source_contexts']==0,'foreign context added')
        count=10
        if original and row['index']<2:
            from gpu_pilot.natural_boundary_inputs import natural_boundaries as old_boundaries,partition_eight
            boundaries=old_boundaries(raw['context']);groups=partition_eight(boundaries,offsets,start);count=8
        else:boundaries=natural_boundaries(raw['context']);groups=partition_ten(boundaries,offsets,start)
        require(boundaries==ev['original_passage_char_boundaries'] and len(row['documents'])==count,'natural boundary count differs')
        for i,(doc,group) in enumerate(zip(row['documents'],groups)):
            left,right=group['input_start'],group['input_end']
            require(doc['document_index']==i and doc['input_start']==left and doc['input_end']==right
                and doc['tokens']==right-left>=256 and doc['document_id']==doc['token_ids_sha256']==digest(ids[left:right])
                and doc['offline_position']==0 and doc['online_position']==left
                and doc['context_char_start']==group['char_start'] and doc['context_char_end']==group['char_end']
                and doc['passage_start']==group['passage_start'] and doc['passage_end']==group['passage_end']
                and doc['source_text_sha256']==text_hash(raw['context'][group['char_start']:group['char_end']]),
                'natural10 block geometry/source differs')
        require(''.join(raw['context'][d['context_char_start']:d['context_char_end']] for d in row['documents'])==raw['context'],
            'blocks must reconstruct full original context without omission/repetition')
        require(row['document_tokens_total']==sum(d['tokens'] for d in row['documents'])
            and row['input_non_document_tokens']==len(ids)-row['document_tokens_total'],'nondocument gap accounting differs')
        if base is not None:base.document_specs(row)
    require(rows[0]['input_tokens']>=16384,'native qualification needs original16K prompt')
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','tokenizer','official-encoding','prompt-file','prompt-source','output-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--calibration-input-dir',type=Path)
    args=parser.parse_args(argv);require(not args.output_dir.exists(),'never overwrite input preparation')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from tokenizers import Tokenizer
    import importlib.metadata
    require(importlib.metadata.version('tokenizers')=='0.21.4','exact tokenizers0.21.4 required')
    source=args.official_encoding.read_text();render=frozen_renderer(source)
    prompt_content=args.prompt_file.read_text();prompt_source=read(args.prompt_source)
    identity=dict(tokenizer_json_sha256=file_hash(args.tokenizer),encoding_sha256=OFFICIAL_ENCODING_SHA256,
        thinking_mode='chat',loader='tokenizers.Tokenizer.from_file',
        prompt_renderer="official_encoding.encode_messages(thinking_mode='chat')",tokenizers_version='0.21.4',
        official_longbench_prompt_sha256=file_hash(args.prompt_file),official_longbench_prompt_revision=prompt_source['revision'])
    artifacts=build_inputs((args.source_dir/'musique.jsonl').read_bytes(),Tokenizer.from_file(str(args.tokenizer)),render,
        tokenizer_identity=identity,official_encoding_source=source,official_prompt_file_content=prompt_content,
        official_prompt_source=prompt_source,calibration_input_dir=args.calibration_input_dir)
    args.output_dir.mkdir(parents=True)
    for name,value in artifacts.items():publish(args.output_dir/name,value)
    validate_inputs(None,artifacts['manifest.json'],artifacts['input-protocol.json'],input_dir=args.output_dir)
    print(json.dumps(artifacts['preparation.json'],ensure_ascii=False))


if __name__=='__main__':main()

