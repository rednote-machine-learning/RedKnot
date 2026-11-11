"""Whole native LongBench QA inputs with eight natural, variable cache spans.

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

DATASETS=('hotpotqa','2wikimqa')
INPUT_COMPANIONS=('source-records.json','selection-plan.json','tokenization-evidence.json')
MANIFEST_SCHEMA='di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA='kv-cache-engram-natural-boundary-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA='natural-boundary-input-protocol-v1'
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
    headers=[m.start() for m in re.finditer(r'(?m)^Passage\s+\d+\s*:',context)]
    require(len(headers)>=8,'at least eight original Passage boundaries required')
    return [0,*[x for x in headers[1:] if x>0],len(context)]

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

def partition_eight(boundaries,offsets,context_start):
    """Globally minimize squared length imbalance, preserving all source text."""
    spans={};arrays=offset_arrays(offsets)
    for a in range(len(boundaries)-1):
        for b in range(a+1,len(boundaries)):
            left,right=exact_token_span(offsets,context_start+boundaries[a],context_start+boundaries[b],arrays=arrays)
            if right-left>=MIN_BLOCK_TOKENS:spans[a,b]=(left,right)
    first,last=exact_token_span(offsets,context_start,context_start+boundaries[-1],arrays=arrays);total=last-first
    dp={(0,0):(0,())}
    for count in range(1,9):
        for end in range(count,len(boundaries)):
            choices=[]
            for begin in range(count-1,end):
                if (count-1,begin) not in dp or (begin,end) not in spans:continue
                cost,cuts=dp[count-1,begin];left,right=spans[begin,end]
                choices.append((cost+(8*(right-left)-total)**2,cuts+(end,)))
            if choices:dp[count,end]=min(choices)
    require((8,len(boundaries)-1) in dp,'no eight natural blocks with minimum256 tokens')
    selected=dp[8,len(boundaries)-1][1];answer=[];begin=0
    for end in selected:
        left,right=spans[begin,end];answer.append(dict(char_start=boundaries[begin],char_end=boundaries[end],input_start=left,input_end=right,passage_start=begin,passage_end=end));begin=end
    return answer

def build_row(raw,dataset,source_index,tokenizer,encode_messages,template):
    context=raw['context'];question=raw['input']
    require(type(context) is str and context and type(question) is str and question,'nonempty original QA required')
    user_text=template.format(context=context,input=question)
    rendered=encode_messages([dict(role='user',content=user_text)],thinking_mode='chat')
    require(rendered.count(context)==1 and rendered.count(question)==1,'source context/question must occur exactly once')
    encoding=tokenizer.encode(rendered,add_special_tokens=False);ids=encoding.ids;offsets=encoding.offsets
    require(tokenizer.decode(ids,skip_special_tokens=False)==rendered,'full prompt decode must reproduce exact rendered text')
    require(tokenizer.encode(tokenizer.decode(ids,skip_special_tokens=False),add_special_tokens=False).ids==ids,'whole prompt roundtrip differs')
    context_start=rendered.index(context);boundaries=natural_boundaries(context)
    groups=partition_eight(boundaries,offsets,context_start)
    documents=[]
    for i,group in enumerate(groups):
        left,right=group['input_start'],group['input_end'];sha=digest(ids[left:right])
        documents.append(dict(document_index=i,document_id=sha,token_ids_sha256=sha,tokens=right-left,
            input_start=left,input_end=right,offline_position=0,online_position=left,
            context_char_start=group['char_start'],context_char_end=group['char_end'],
            source_text_sha256=text_hash(context[group['char_start']:group['char_end']]),
            passage_start=group['passage_start'],passage_end=group['passage_end']))
    require(len({d['document_id'] for d in documents})==8,'duplicate cache block token sequence')
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

def validate_inputs(base,manifest,protocol,*,input_dir):
    verify(manifest);verify(protocol)
    require(manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('contract_schema')==CONTRACT_SCHEMA
        and manifest.get('artifact_kind')=='natural_boundary_inputs_v1'
        and protocol.get('schema')==INPUT_PROTOCOL_SCHEMA and manifest.get('input_protocol_sha256')==protocol['sha256'], 'natural input contract differs')
    require(protocol.get('composition')=='single_original_qa_eight_natural_cache_blocks'
        and protocol.get('no_padding') is True and protocol.get('no_truncation') is True
        and protocol.get('no_cross_sample_contexts') is True and protocol.get('preserve_original_order') is True,'native source preservation contract differs')
    directory=Path(input_dir)
    companions={name:read(directory/name) for name in INPUT_COMPANIONS}
    for name,value in companions.items():verify(value);require(manifest['companion_sha256'][name]==value['sha256'],'input companion binding differs')
    sources=companions['source-records.json'];selection=companions['selection-plan.json'];tokens=companions['tokenization-evidence.json']
    require(protocol['source_records_sha256']==sources['sha256'] and protocol['selection_plan_sha256']==selection['sha256']
        and protocol['tokenization_evidence_sha256']==tokens['sha256'] and manifest['tokenizer']==tokens['tokenizer'],'provenance chain differs')
    require(text_hash(tokens['official_prompt_file_content'])==tokens['official_prompt_source']['sha256']==manifest['tokenizer']['official_longbench_prompt_sha256'],
        'official prompt template source differs')
    templates=json.loads(tokens['official_prompt_file_content'])
    require(tokens['official_prompt_templates']=={k:templates[k] for k in DATASETS},'task prompt templates differ')
    render=frozen_renderer(tokens['official_encoding_source'])
    require(manifest['tokenizer']['encoding_sha256']==OFFICIAL_ENCODING_SHA256,'official renderer identity differs')
    require(selection['no_model_outcomes_used'] is True and selection['seed']==SEED,'outcome-dependent selection prohibited')
    rows=manifest['warmup']+manifest['cohort']
    require([r['index'] for r in rows]==list(range(22)) and [r['dataset'] for r in manifest['cohort']]==list(DATASETS)*10,'exact calibration2 and evaluation20 required')
    require([r['sample_id'] for r in rows]==selection['selected_sample_ids'],'sample selection binding differs')
    raw_lookup={r['sample_id']:r for r in sources['records']};evidence={r['sample_id']:r for r in tokens['records']}
    require(len(raw_lookup)==len(evidence)==len(rows)==22 and len({r['context_sha256'] for r in rows})==22,'selected sources must be distinct')
    if base is not None:base.validate_manifest(manifest)
    for row in rows:
        raw_record=raw_lookup[row['sample_id']];raw=raw_record['raw'];ev=evidence[row['sample_id']];ids=row['input_ids']
        require(row['source_row_sha256']==raw_record['source_row_sha256']==digest(raw)
            and row['source_index']==raw_record['source_index'] and row['source_id']==raw['_id']
            and row['question']==raw['input'] and row['references']==raw['answers']
            and row['context_text']==raw['context'] and row['context_sha256']==text_hash(raw['context']),'raw source QA was changed')
        require(row['input_tokens']==len(ids) and row['input_ids_sha256']==digest(ids) and row['references_sha256']==digest(raw['answers'])
            and all(type(t) is int and 0<=t<2**31 for t in ids),'token/reference hash differs')
        prompt=row['prompt'];start=row['context_prompt_char_start'];end=row['context_prompt_char_end']
        expected_prompt=render([dict(role='user',content=templates[row['dataset']].format(context=raw['context'],input=raw['input']))],thinking_mode='chat')
        require(prompt==expected_prompt,'prompt differs from exact original QA and official template')
        require(prompt[start:end]==raw['context'] and prompt.count(raw['context'])==prompt.count(raw['input'])==1
            and text_hash(prompt)==row['prompt_sha256']==ev['prompt_sha256'] and row['input_ids_sha256']==ev['input_ids_sha256'],'prompt content/identity differs')
        require(row['natural_boundary_inputs'] is True and row['primary_context_truncated'] is False
            and row['source_order_preserved'] is True and row['source_context_count']==1 and row['text_roundtrip_matches_ids'] is True,'source transformation detected')
        offsets=ev['token_offsets'];require(len(offsets)==len(ids),'token offset count differs')
        for flag in ('whole_prompt_single_encoding','decode_matches_rendered_prompt','reencode_matches_input_ids','full_source_context_preserved','full_source_question_preserved'):require(ev[flag] is True,'whole prompt identity not proven')
        boundaries=natural_boundaries(raw['context']);require(boundaries==ev['original_passage_char_boundaries'],'non-natural cut boundary')
        groups=partition_eight(boundaries,offsets,start);docs=row['documents'];require(len(docs)==8,'exactly8 cache blocks required')
        for i,(doc,group) in enumerate(zip(docs,groups)):
            left,right=exact_token_span(offsets,start+group['char_start'],start+group['char_end'])
            require(doc['document_index']==i and doc['input_start']==left and doc['input_end']==right
                and doc['tokens']==right-left>=MIN_BLOCK_TOKENS and doc['document_id']==doc['token_ids_sha256']==digest(ids[left:right])
                and doc['offline_position']==0 and doc['online_position']==left
                and doc['context_char_start']==group['char_start'] and doc['context_char_end']==group['char_end']
                and doc['source_text_sha256']==text_hash(raw['context'][group['char_start']:group['char_end']]),'natural block geometry/identity differs')
        require(''.join(raw['context'][x['context_char_start']:x['context_char_end']] for x in docs)==raw['context'],'blocks do not reconstruct entire original context')
        require(row['document_tokens_total']==sum(d['tokens'] for d in docs)
            and row['input_non_document_tokens']==len(ids)-row['document_tokens_total'],'online gap accounting differs')
        if base is not None:base.document_specs(row)
    require(manifest['warmup'][0]['input_tokens']>=16384,'native qualification requires an original16K input')
    return True

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','tokenizer','official-encoding','prompt-file','prompt-source','output-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--exclude-questions',type=Path)
    args=parser.parse_args(argv);require(not args.output_dir.exists(),'preserve prior input preparation')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from tokenizers import Tokenizer
    import importlib.metadata
    require(importlib.metadata.version('tokenizers')=='0.21.4','expected actual tokenizer library version')
    tok=Tokenizer.from_file(str(args.tokenizer));spec=importlib.util.spec_from_file_location('native_v41_encoding',args.official_encoding)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prompts=read(args.prompt_file);prompt_source=read(args.prompt_source)
    require(file_hash(args.prompt_file)==prompt_source['sha256'],'official prompt bytes differ')
    require(file_hash(args.official_encoding)==OFFICIAL_ENCODING_SHA256,'unrecognized official chat renderer')
    excludes=set(read(args.exclude_questions)) if args.exclude_questions else set()
    identity=dict(tokenizer_json_sha256=file_hash(args.tokenizer),encoding_sha256=file_hash(args.official_encoding),thinking_mode='chat',
        loader='tokenizers.Tokenizer.from_file',prompt_renderer="official_encoding.encode_messages(thinking_mode='chat')",tokenizers_version='0.21.4',
        official_longbench_prompt_sha256=file_hash(args.prompt_file),official_longbench_prompt_revision=prompt_source['revision'])
    pools={};rejections={};files=[]
    for dataset in DATASETS:
        path=args.source_dir/(dataset+'.jsonl');raw_bytes=path.read_bytes();files.append(dict(dataset=dataset,path=str(path),sha256=hashlib.sha256(raw_bytes).hexdigest()))
        eligible=[];rejected=[]
        for index,line in enumerate(raw_bytes.splitlines()):
            raw=json.loads(line)
            if text_hash(raw['input']) in excludes:rejected.append(dict(source_index=index,reason='previous_question'));continue
            count=len(tok.encode(raw['context'],add_special_tokens=False).ids)
            if count<8000:rejected.append(dict(source_index=index,reason='native_context_under8000',tokens=count));continue
            try:row,ev=build_row(raw,dataset,index,tok,module.encode_messages,prompts[dataset])
            except ValueError as exc:rejected.append(dict(source_index=index,reason=str(exc),tokens=count));continue
            eligible.append(dict(row=row,evidence=ev,raw=raw,order=digest(dict(seed=SEED,dataset=dataset,source_id=raw['_id'],context_sha256=text_hash(raw['context'])))))
        pools[dataset]=eligible;rejections[dataset]=rejected
    selected=[];seen_contexts=set();seen_questions=set()
    for ordinal,dataset in enumerate(DATASETS):
        eligible=pools[dataset]
        calibration=sorted(eligible,key=lambda x:(-x['row']['input_tokens'],x['order']))[0]
        calibration['row'].update(index=ordinal,split='calibration');selected.append(calibration)
        seen_contexts.add(calibration['row']['context_sha256']);seen_questions.add(text_hash(calibration['row']['question']))
    for ordinal,dataset in enumerate(DATASETS):
        chosen=[]
        for item in sorted(pools[dataset],key=lambda x:x['order']):
            row=item['row'];context_sha=row['context_sha256'];question_sha=text_hash(row['question'])
            if context_sha in seen_contexts or question_sha in seen_questions:continue
            row.update(index=2+2*len(chosen)+ordinal,split='evaluation');chosen.append(item)
            seen_contexts.add(context_sha);seen_questions.add(question_sha)
            if len(chosen)==10:break
        require(len(chosen)==10,'insufficient10 natural eight-block sources: '+dataset);selected.extend(chosen)
    selected.sort(key=lambda x:x['row']['index']);rows=[x['row'] for x in selected]
    sources=seal(dict(schema='natural-boundary-source-records-v1',files=files,
        records=[dict(sample_id=x['row']['sample_id'],dataset=x['row']['dataset'],source_index=x['row']['source_index'],source_row_sha256=digest(x['raw']),raw=x['raw']) for x in selected]))
    selection=seal(dict(schema='natural-boundary-selection-plan-v1',seed=SEED,no_model_outcomes_used=True,
        composition='single_original_qa_eight_natural_cache_blocks',eligibility='original context>=8000 tokens,>=8 Passage units,8 contiguous groups each>=256 tokens',
        block_partition='minimum squared token-length imbalance among all complete contiguous Passage groupings; lexical cut tie-break',
        calibration_selection='longest eligible original context per dataset; SHA tie-break',evaluation_selection='ascending SHA(seed,dataset,source_id,context_sha256),no repeated context/question',
        excluded_question_sha256=sorted(excludes),eligible_counts={k:len(v) for k,v in pools.items()},rejected=rejections,
        selected_sample_ids=[r['sample_id'] for r in rows],independent_holdout_claim=False))
    evidence=seal(dict(schema='natural-boundary-tokenization-evidence-v1',tokenizer=identity,official_prompt_source=prompt_source,
        official_prompt_file_content=args.prompt_file.read_text(),official_encoding_source=args.official_encoding.read_text(),
        official_prompt_templates={k:prompts[k] for k in DATASETS},records=[x['evidence'] for x in selected]))
    protocol=seal(dict(schema=INPUT_PROTOCOL_SCHEMA,seed=SEED,composition='single_original_qa_eight_natural_cache_blocks',
        calibration_indices=[0,1],evaluation_indices=list(range(2,22)),datasets=list(DATASETS),num_chunks=8,min_block_tokens=256,
        equal_document_lengths_required=False,no_padding=True,no_truncation=True,no_cross_sample_contexts=True,preserve_original_order=True,
        single_prompt_tokenization=True,no_model_outcomes_used=True,source_records_sha256=sources['sha256'],selection_plan_sha256=selection['sha256'],
        tokenization_evidence_sha256=evidence['sha256'],tokenizer=identity,tokenizers_version='0.21.4',inference_seed=0))
    companions=dict(zip(INPUT_COMPANIONS,(sources,selection,evidence)))
    manifest=seal(dict(schema=MANIFEST_SCHEMA,contract_schema=CONTRACT_SCHEMA,artifact_kind='natural_boundary_inputs_v1',natural_boundary_inputs=True,
        input_protocol_sha256=protocol['sha256'],tokenizer=identity,num_chunks=8,chunk_tokens=None,context_tokens=None,
        sources=files,warmup=rows[:2],cohort=rows[2:],companion_sha256={k:v['sha256'] for k,v in companions.items()},selection_plan_sha256=selection['sha256']))
    args.output_dir.mkdir(parents=True)
    for name,value in companions.items():publish(args.output_dir/name,value)
    publish(args.output_dir/'input-protocol.json',protocol);publish(args.output_dir/'manifest.json',manifest)
    validate_inputs(None,manifest,protocol,input_dir=args.output_dir)
    summary=seal(dict(schema='natural-boundary-input-preparation-v1',status='complete',cuda_used=False,model_results_used=False,
        manifest_sha256=manifest['sha256'],input_protocol_sha256=protocol['sha256'],eligible_counts=selection['eligible_counts'],
        samples=[dict(index=r['index'],dataset=r['dataset'],source_index=r['source_index'],question=r['question'],references=r['references'],
            input_tokens=r['input_tokens'],original_context_tokens=r['original_context_tokens'],block_tokens=[d['tokens'] for d in r['documents']],
            nondocument_tokens=r['input_non_document_tokens'],source_context_preserved=True,whole_prompt_roundtrip=True) for r in rows]))
    publish(args.output_dir/'preparation.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
