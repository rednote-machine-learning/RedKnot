"""Six complete original LongBench-v2 MC QAs in two actual-token length bands.

CPU only. Ten cache blocks follow original paragraph boundaries; no padding,
truncation, unrelated-context concatenation, answer-based or outcome-based choice.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from gpu_pilot import musique30_inputs as common
from gpu_pilot import musique_longform_inputs as reuse
from gpu_pilot.broadqa_inputs import documents_from_groups

MANIFEST_SCHEMA='di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA='kv-cache-engram-longcontext-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA='longcontext-input-protocol-v1'
INPUT_COMPANIONS=common.INPUT_COMPANIONS
PREVIOUS_FILES=reuse.PREVIOUS_FILES
DATASETS=('longbench_v2',)
LENGTH_BANDS=('128k','64k')
TARGET_INPUT_TOKENS={'128k':131072,'64k':65536}
INPUT_TOKEN_RANGES={'128k':[112*1024,144*1024],'64k':[56*1024,72*1024]}
EVALUATION_INDICES=tuple(range(2,8))
NUM_CHUNKS=10
MIN_BLOCK_TOKENS=256
SEED=20261109
SOURCE_REVISION='2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
SOURCE_POOL_SHA256='1917439a44b42305a9ff85c9891d74f7118fe9c1734e125e4b72295dcf64b20a'
OFFICIAL_PROMPT_REVISION='2e00731f8d0bff23dc4325161044d0ed8af94c1e'
OFFICIAL_PROMPT_SHA256='68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950'
OFFICIAL_SCORER_SHA256='ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138'
CALIBRATION_SOURCE='previous_musique_calibration_exact'
require,digest,text_hash=common.require,common.digest,common.text_hash
file_hash,seal,verify,read,publish=common.file_hash,common.seal,common.verify,common.read,common.publish


def validate_pool(pool):
    verify(pool)
    require(pool.get('schema')=='longcontext-official-candidate-pool-v1' and pool['sha256']==SOURCE_POOL_SHA256
        and pool.get('dataset')=='longbench_v2' and pool.get('source_revision')==SOURCE_REVISION,
        'pinned official candidate pool differs')
    require(pool.get('selection_uses_model_outputs') is False and pool.get('selection_uses_gold') is False,
        'candidate acquisition must not use outcomes or answers')
    rows=pool['rows']; require(rows and len({r['source_index'] for r in rows})==len(rows),'unique original source indices required')
    for item in rows:
        r=item['raw']; require(type(item['source_index']) is int and item['source_index']>=0
            and r.get('domain') in ('Single-Document QA','Multi-Document QA') and r.get('answer') in 'ABCD'
            and all(type(r.get(k)) is str and r[k] for k in ('_id','context','question','choice_A','choice_B','choice_C','choice_D')),
            'original document QA/options/gold required')
    return rows


def natural_boundaries(context):
    # Blank original paragraphs/pages only. A line wrapping a PDF sentence is not a boundary.
    require(type(context) is str and context,'nonempty original context required')
    points=sorted({0,len(context),*(m.end() for m in re.finditer(r'\n[ \t\r]*\n(?:[ \t\r]*\n)*',context))})
    require(len(points)>=11,'at least ten original paragraph segments required')
    return points


def partition_ten(boundaries,offsets,context_start):
    """Linear-in-boundaries deterministic balanced partition; all source characters survive."""
    require(boundaries[0]==0 and boundaries==sorted(set(boundaries)),'ordered unique natural boundaries required')
    arrays=common.offset_arrays(offsets)
    first,last=common.exact_token_span(offsets,context_start,context_start+boundaries[-1],arrays=arrays)
    total=last-first; groups=[]; begin=0
    for count in range(1,10):
        choices=[]
        for end in range(begin+1,len(boundaries)-(10-count)):
            try:
                left,right=common.exact_token_span(offsets,context_start+boundaries[begin],context_start+boundaries[end],arrays=arrays)
                rest_left,rest_right=common.exact_token_span(offsets,context_start+boundaries[end],context_start+boundaries[-1],arrays=arrays)
            except ValueError:continue
            if right-left>=MIN_BLOCK_TOKENS and rest_right-rest_left>=(10-count)*MIN_BLOCK_TOKENS:
                choices.append((abs(10*(right-first)-count*total),end,left,right))
        require(choices,'no ten natural blocks with minimum256 tokens')
        _,end,left,right=min(choices)
        groups.append(dict(char_start=boundaries[begin],char_end=boundaries[end],input_start=left,input_end=right,
            passage_start=begin,passage_end=end));begin=end
    left,right=common.exact_token_span(offsets,context_start+boundaries[begin],context_start+boundaries[-1],arrays=arrays)
    require(right-left>=MIN_BLOCK_TOKENS,'final natural block below minimum256 tokens')
    groups.append(dict(char_start=boundaries[begin],char_end=boundaries[-1],input_start=left,input_end=right,
        passage_start=begin,passage_end=len(boundaries)-1))
    return groups


def render_original(raw,render,template):
    # Replace placeholders once; original context/question/choices are never stripped or edited.
    values={'$DOC$':raw['context'],'$Q$':raw['question'],**{'$C_'+a+'$':raw['choice_'+a] for a in 'ABCD'}}
    require(all(template.count(k)==1 for k in values),'official template placeholders differ')
    content=re.sub(r'\$DOC\$|\$Q\$|\$C_[ABCD]\$',lambda m:values[m.group()],template)
    return render([dict(role='user',content=content)],thinking_mode='chat')


def source_metadata(raw,index):
    return dict(dataset='longbench_v2',source_index=index,source_id=raw['_id'],source_row_sha256=digest(raw),
        source_domain=raw['domain'],source_sub_domain=raw['sub_domain'],context_sha256=text_hash(raw['context']),
        question_sha256=text_hash(raw['question']),tie_sha256=digest(dict(seed=SEED,source_id=raw['_id'],context_sha256=text_hash(raw['context']))))


def classify_length(n):
    return next((band for band in LENGTH_BANDS if INPUT_TOKEN_RANGES[band][0]<=n<=INPUT_TOKEN_RANGES[band][1]),None)


def build_row(raw,index,tokenizer,render,template):
    prompt=render_original(raw,render,template);context=raw['context'];encoded=tokenizer.encode(prompt,add_special_tokens=False)
    ids,offsets=encoded.ids,encoded.offsets;band=classify_length(len(ids))
    require(band is not None,'complete prompt outside declared actual-token length bands')
    require(tokenizer.decode(ids,skip_special_tokens=False)==prompt and
        tokenizer.encode(tokenizer.decode(ids,skip_special_tokens=False),add_special_tokens=False).ids==ids,'whole prompt roundtrip differs')
    require(prompt.count(context)==1,'entire source context must appear exactly once')
    start=prompt.index(context);end=start+len(context);boundaries=natural_boundaries(context)
    groups=partition_ten(boundaries,offsets,start);docs=documents_from_groups(context,ids,groups)
    require(len({d['document_id'] for d in docs})==10,'duplicate natural cache block')
    prefix,suffix=docs[0]['input_start'],len(ids)-docs[-1]['input_end'];gold=raw['answer']
    row=dict(source_metadata(raw,index),sample_id='longbench_v2:'+raw['_id'],question=raw['question'],
        references=[gold],references_sha256=digest([gold]),gold_letter=gold,gold_text=raw['choice_'+gold],
        **{'choice_'+a:raw['choice_'+a] for a in 'ABCD'},
        length_band=band,target_input_tokens=TARGET_INPUT_TOKENS[band],input_ids=ids,input_ids_sha256=digest(ids),input_tokens=len(ids),
        prompt=prompt,prompt_sha256=text_hash(prompt),context_text=context,context_prompt_char_start=start,context_prompt_char_end=end,
        original_context_tokens=len(tokenizer.encode(context,add_special_tokens=False).ids),context_tokens=len(ids)-prefix-suffix,
        document_tokens_total=sum(d['tokens'] for d in docs),prefix_tokens=prefix,suffix_tokens=suffix,
        rendered_prefix=prompt[:start],rendered_suffix=prompt[end:],primary_context_truncated=False,natural_boundary_inputs=True,
        source_context_count=1,source_order_preserved=True,text_roundtrip_matches_ids=True,
        prompt_transport='whole rendered prompt tokenized once; exact token IDs authoritative',
        block_boundary_kind='original_blank_paragraph_boundaries',cache_blocks_are_independent_articles=False,
        documents=docs,input_non_document_tokens=len(ids)-sum(d['tokens'] for d in docs),
        support_retention='complete original LongBench-v2 QA context and choices unchanged; ten natural contiguous cache blocks')
    ev=dict(sample_id=row['sample_id'],prompt_sha256=row['prompt_sha256'],input_ids_sha256=row['input_ids_sha256'],
        whole_prompt_single_encoding=True,decode_matches_rendered_prompt=True,reencode_matches_input_ids=True,
        token_offsets=[list(x) for x in offsets],original_passage_char_boundaries=boundaries,
        full_source_context_preserved=True,full_source_question_preserved=True,full_source_options_preserved=True,additional_source_contexts=0)
    return row,ev


def candidate_metadata(row):
    keys=('dataset','source_index','source_id','source_row_sha256','source_domain','source_sub_domain','context_sha256',
          'question_sha256','tie_sha256','sample_id','length_band','target_input_tokens','input_tokens','input_ids_sha256')
    return {k:row[k] for k in keys}


def selection_key(item):
    return (item['source_domain']!='Multi-Document QA',abs(item['input_tokens']-TARGET_INPUT_TOKENS[item['length_band']]),item['tie_sha256'])


def choose_candidates(candidates):
    contexts=set();questions=set();chosen={}
    for band in LENGTH_BANDS:
        chosen[band]=[]
        for item in sorted((x for x in candidates if x['length_band']==band),key=selection_key):
            if item['context_sha256'] in contexts or item['question_sha256'] in questions:continue
            chosen[band].append(item);contexts.add(item['context_sha256']);questions.add(item['question_sha256'])
            if len(chosen[band])==3:break
        require(len(chosen[band])==3,'fewer than three distinct natural eligible contexts in '+band)
        chosen[band].sort(key=lambda x:(-x['input_tokens'],x['tie_sha256']))
    return [chosen[band][ordinal] for ordinal in range(3) for band in LENGTH_BANDS]


def fixed_policy():
    return dict(seed=SEED,datasets=list(DATASETS),evaluation_cases=6,source_qa_count=6,calibration_indices=[0,1],
        evaluation_indices=list(EVALUATION_INDICES),calibration_source=CALIBRATION_SOURCE,num_chunks=10,min_block_tokens=256,
        length_bands=list(LENGTH_BANDS),target_input_tokens=TARGET_INPUT_TOKENS,input_token_ranges=INPUT_TOKEN_RANGES,
        no_padding=True,no_truncation=True,no_cross_sample_contexts=True,preserve_original_order=True,single_prompt_tokenization=True,
        no_model_outcomes_used=True,no_answer_content_selection=True,gold_length_selection=False,generated_length_selection=False,
        minimum_generated_tokens=None,natural_eos_preserved=True,cache_blocks_are_independent_articles=False,
        source_revision=SOURCE_REVISION,source_pool_sha256=SOURCE_POOL_SHA256,tokenizers_version='0.21.4',inference_seed=0,
        official_prompt_revision=OFFICIAL_PROMPT_REVISION,official_prompt_sha256=OFFICIAL_PROMPT_SHA256,
        official_scorer_sha256=OFFICIAL_SCORER_SHA256,quality_task='multiple_choice',primary_metric='longbench_v2_official_mc_accuracy')


def build_inputs(previous_bundle,pool,tokenizer,render,*,tokenizer_identity,official_encoding_source,official_prompt_file_content):
    prior=reuse.validate_previous_bundle(previous_bundle);raws=validate_pool(pool)
    require(prior['tokenizer']==tokenizer_identity,'actual tokenizer identity differs from exact calibration')
    require(text_hash(official_encoding_source)==common.OFFICIAL_ENCODING_SHA256 and
        text_hash(official_prompt_file_content)==OFFICIAL_PROMPT_SHA256,'official renderer/template identity differs')
    eligible=[];rejected=[];lookup={}
    for item in raws:
        raw,index=item['raw'],item['source_index'];meta=source_metadata(raw,index)
        try:row,ev=build_row(raw,index,tokenizer,render,official_prompt_file_content)
        except ValueError as exc:rejected.append(dict(meta,reason=str(exc)));continue
        m=candidate_metadata(row);eligible.append(m);lookup[m['sample_id']]=(row,ev,raw)
    chosen=choose_candidates(eligible);rows=copy.deepcopy(prior['warmup']);calids={r['sample_id'] for r in rows}
    sources=[copy.deepcopy(r) for r in previous_bundle['source-records.json']['records'] if r['sample_id'] in calids]
    evidence=[copy.deepcopy(r) for r in previous_bundle['tokenization-evidence.json']['records'] if r['sample_id'] in calids]
    for i,item in enumerate(chosen):
        row,ev,raw=lookup[item['sample_id']];row=dict(row,index=i+2,split='evaluation',length_band_ordinal=i//2+1)
        rows.append(row);evidence.append(ev);sources.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],source_index=row['source_index'],source_row_sha256=digest(raw),raw=raw))
    source=seal(dict(schema='longcontext-source-records-v1',previous_inputs=previous_bundle,official_candidate_pool=pool,records=sources))
    selection=seal(dict(schema='longcontext-selection-plan-v1',seed=SEED,no_model_outcomes_used=True,no_answer_content_selection=True,
        gold_length_selection=False,generated_length_selection=False,source_pool_sha256=pool['sha256'],
        eligible_candidates=sorted(eligible,key=lambda x:x['source_index']),rejected=rejected,
        selected_sample_ids=[r['sample_id'] for r in rows],selection_rule='within band: native multidocument first, nearest target, fixed SHA; distinct context/question; descending lengths then 128k/64k interleaved',
        acquisition_scope='fixed official candidate prefix; not a representative random sample of all LongBench-v2',independent_model_training_holdout_claim=False))
    tokens=seal(dict(schema='longcontext-tokenization-evidence-v1',tokenizer=tokenizer_identity,
        official_prompt_file_content=official_prompt_file_content,official_prompt_revision=OFFICIAL_PROMPT_REVISION,
        official_prompt_sha256=OFFICIAL_PROMPT_SHA256,official_encoding_source=official_encoding_source,records=evidence))
    protocol=seal(dict(schema=INPUT_PROTOCOL_SCHEMA,**fixed_policy(),tokenizer=tokenizer_identity,previous_manifest_sha256=prior['sha256'],
        source_records_sha256=source['sha256'],selection_plan_sha256=selection['sha256'],tokenization_evidence_sha256=tokens['sha256']))
    companions=dict(zip(INPUT_COMPANIONS,(source,selection,tokens)))
    manifest=seal(dict(schema=MANIFEST_SCHEMA,contract_schema=CONTRACT_SCHEMA,artifact_kind='longcontext_inputs_v1',natural_boundary_inputs=True,
        input_protocol_sha256=protocol['sha256'],tokenizer=tokenizer_identity,num_chunks=10,chunk_tokens=None,context_tokens=None,
        calibration_source=CALIBRATION_SOURCE,warmup=rows[:2],cohort=rows[2:],
        sources=[dict(dataset='longbench_v2',revision=SOURCE_REVISION,source_pool_sha256=pool['sha256'])],
        companion_sha256={k:v['sha256'] for k,v in companions.items()},selection_plan_sha256=selection['sha256']))
    summary=seal(dict(schema='longcontext-input-preparation-v1',status='complete',cuda_used=False,model_results_used=False,
        manifest_sha256=manifest['sha256'],input_protocol_sha256=protocol['sha256'],evaluation_cases=6,source_qa_count=6,
        eligible_counts={b:sum(c['length_band']==b for c in eligible) for b in LENGTH_BANDS},
        samples=[dict(index=r['index'],dataset=r['dataset'],source_index=r['source_index'],source_id=r['source_id'],
            length_band=r.get('length_band'),length_band_ordinal=r.get('length_band_ordinal'),target_input_tokens=r.get('target_input_tokens'),
            input_tokens=r['input_tokens'],block_tokens=[d['tokens'] for d in r['documents']],
            source_domain=r.get('source_domain'),source_context_preserved=True,whole_prompt_roundtrip=True) for r in rows]))
    return dict(companions,**{'manifest.json':manifest,'input-protocol.json':protocol,'preparation.json':summary})


def validate_inputs(base,manifest,protocol,*,input_dir):
    verify(manifest);verify(protocol)
    require(manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('contract_schema')==CONTRACT_SCHEMA
        and manifest.get('artifact_kind')=='longcontext_inputs_v1' and protocol.get('schema')==INPUT_PROTOCOL_SCHEMA
        and manifest['input_protocol_sha256']==protocol['sha256'],'longcontext input schema/binding differs')
    require(all(k in protocol and protocol[k]==v for k,v in fixed_policy().items()),'longcontext fixed policy differs')
    companions={n:read(Path(input_dir)/n) for n in INPUT_COMPANIONS}
    require(set(manifest['companion_sha256'])==set(INPUT_COMPANIONS),'companion inventory differs')
    for n,v in companions.items():verify(v);require(manifest['companion_sha256'][n]==v['sha256'],'companion seal differs')
    source,selection,tokens=[companions[n] for n in INPUT_COMPANIONS]
    require([v.get('schema') for v in (source,selection,tokens)]==['longcontext-source-records-v1','longcontext-selection-plan-v1','longcontext-tokenization-evidence-v1'],'companion schema differs')
    for key,v in (('source_records_sha256',source),('selection_plan_sha256',selection),('tokenization_evidence_sha256',tokens)):
        require(protocol[key]==v['sha256'],'provenance binding differs')
    prior=reuse.validate_previous_bundle(source['previous_inputs']);rawitems=validate_pool(source['official_candidate_pool'])
    raws={x['source_index']:x['raw'] for x in rawitems}
    require(manifest['tokenizer']==protocol['tokenizer']==tokens['tokenizer']==prior['tokenizer']
        and tokens['tokenizer']['encoding_sha256']==common.OFFICIAL_ENCODING_SHA256
        and protocol['previous_manifest_sha256']==prior['sha256'],'actual tokenizer/calibration identity differs')
    require(tokens['official_prompt_revision']==OFFICIAL_PROMPT_REVISION and tokens['official_prompt_sha256']==OFFICIAL_PROMPT_SHA256
        and text_hash(tokens['official_prompt_file_content'])==OFFICIAL_PROMPT_SHA256,'official MC prompt differs')
    render=common.frozen_renderer(tokens['official_encoding_source']);rows=manifest['warmup']+manifest['cohort']
    require(manifest['warmup']==prior['warmup'] and [r['index'] for r in rows]==list(range(8)) and len(manifest['cohort'])==6
        and manifest['num_chunks']==10 and manifest['natural_boundary_inputs'] is True
        and manifest['calibration_source']==CALIBRATION_SOURCE and manifest['selection_plan_sha256']==selection['sha256'],
        'exact calibration2/evaluation6 geometry differs')
    require(manifest['sources']==[dict(dataset='longbench_v2',revision=SOURCE_REVISION,source_pool_sha256=SOURCE_POOL_SHA256)],'source inventory differs')
    require(selection['source_pool_sha256']==SOURCE_POOL_SHA256 and selection['seed']==SEED
        and all(selection[k] is True for k in ('no_model_outcomes_used','no_answer_content_selection'))
        and all(selection[k] is False for k in ('gold_length_selection','generated_length_selection')),'selection policy differs')
    candidates,rejected=selection['eligible_candidates'],selection['rejected']
    require(sorted(x['source_index'] for x in candidates+rejected)==sorted(raws),'candidate source coverage differs')
    for item in candidates+rejected:
        meta=source_metadata(raws[item['source_index']],item['source_index'])
        require(all(item.get(k)==v for k,v in meta.items()),'candidate original source differs')
    for item in candidates:
        require(classify_length(item['input_tokens'])==item['length_band'] and
            item['target_input_tokens']==TARGET_INPUT_TOKENS[item['length_band']],'candidate actual length band differs')
    require(candidates==sorted(candidates,key=lambda x:x['source_index']),'candidate order differs')
    chosen=choose_candidates(candidates)
    require(selection['selected_sample_ids']==[r['sample_id'] for r in rows]
        and [r['sample_id'] for r in rows[2:]]==[m['sample_id'] for m in chosen], 'selected candidate order differs')
    records={x['sample_id']:x for x in source['records']};evs={x['sample_id']:x for x in tokens['records']}
    require(len(records)==len(source['records'])==len(evs)==len(tokens['records'])==8
        and set(records)==set(evs)=={r['sample_id'] for r in rows},'source/token record inventory differs')
    oldsrc={x['sample_id']:x for x in source['previous_inputs']['source-records.json']['records']}
    oldev={x['sample_id']:x for x in source['previous_inputs']['tokenization-evidence.json']['records']}
    for r in rows[:2]:require(records[r['sample_id']]==oldsrc[r['sample_id']] and evs[r['sample_id']]==oldev[r['sample_id']], 'exact calibration evidence differs')
    for i,(row,item) in enumerate(zip(rows[2:],chosen)):
        raw=raws[item['source_index']];context=raw['context'];ids=row['input_ids'];ev=evs[row['sample_id']]
        require(row['length_band']==LENGTH_BANDS[i%2] and row['length_band_ordinal']==i//2+1 and row['split']=='evaluation'
            and candidate_metadata(row)==item,'case actual length/order/source binding differs')
        require(row['sample_id']=='longbench_v2:'+raw['_id'] and row['question']==raw['question'] and row['context_text']==context
            and row['references']==[raw['answer']] and row['references_sha256']==digest([raw['answer']])
            and row['gold_letter']==raw['answer'] and row['gold_text']==raw['choice_'+raw['answer']]
            and all(row['choice_'+a]==raw['choice_'+a] for a in 'ABCD')
            and records[row['sample_id']]==dict(sample_id=row['sample_id'],dataset='longbench_v2',source_index=row['source_index'],source_row_sha256=digest(raw),raw=raw),
            'original complete context/question/options/gold changed')
        prompt=render_original(raw,render,tokens['official_prompt_file_content']);start=row['context_prompt_char_start'];end=row['context_prompt_char_end']
        require(row['prompt']==prompt and prompt.count(context)==1 and prompt[start:end]==context and end==start+len(context)
            and row['prompt_sha256']==ev['prompt_sha256']==text_hash(prompt) and row['input_tokens']==len(ids)
            and row['input_ids_sha256']==ev['input_ids_sha256']==digest(ids)
            and all(type(t) is int and 0<=t<2**31 for t in ids),'whole native prompt/token identity differs')
        require(row['natural_boundary_inputs'] is True and row['primary_context_truncated'] is False and row['source_context_count']==1
            and row['source_order_preserved'] is True and row['text_roundtrip_matches_ids'] is True
            and row['cache_blocks_are_independent_articles'] is False and row['block_boundary_kind']=='original_blank_paragraph_boundaries',
            'native input preservation claims differ')
        require(len(ev['token_offsets'])==len(ids) and ev['additional_source_contexts']==0
            and all(ev[k] is True for k in ('whole_prompt_single_encoding','decode_matches_rendered_prompt','reencode_matches_input_ids',
                'full_source_context_preserved','full_source_question_preserved','full_source_options_preserved')),'tokenization evidence differs')
        boundaries=natural_boundaries(context);require(boundaries==ev['original_passage_char_boundaries'],'native paragraph boundaries differ')
        groups=partition_ten(boundaries,ev['token_offsets'],start);docs=documents_from_groups(context,ids,groups)
        require(row['documents']==docs and all(d['tokens']>=256 for d in docs)
            and ''.join(context[d['context_char_start']:d['context_char_end']] for d in docs)==context,'ten complete natural cache blocks differ')
        require(row['document_tokens_total']==sum(d['tokens'] for d in docs)
            and row['input_non_document_tokens']==len(ids)-row['document_tokens_total']
            and row['prefix_tokens']==docs[0]['input_start'] and row['suffix_tokens']==len(ids)-docs[-1]['input_end']
            and row['rendered_prefix']==prompt[:start] and row['rendered_suffix']==prompt[end:],'native gap accounting differs')
    require(rows[2]['input_tokens']==max(r['input_tokens'] for r in rows[2:]),'first evaluation must be longest qualification input')
    if base is not None:
        base.validate_manifest(manifest)
        for row in rows:base.document_specs(row)
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-input-dir','source-pool','tokenizer','official-encoding','prompt-file','output-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args(argv);require(not args.output_dir.exists(),'never overwrite input preparation')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import importlib.metadata
    from tokenizers import Tokenizer
    require(importlib.metadata.version('tokenizers')=='0.21.4','exact tokenizers0.21.4 required')
    bundle={n:read(args.previous_input_dir/n) for n in PREVIOUS_FILES}
    common.validate_inputs(None,bundle['manifest.json'],bundle['input-protocol.json'],input_dir=args.previous_input_dir)
    identity=bundle['manifest.json']['tokenizer']
    require(file_hash(args.tokenizer)==identity['tokenizer_json_sha256'],'actual checkpoint tokenizer bytes differ')
    encoder=args.official_encoding.read_text();template=args.prompt_file.read_text()
    result=build_inputs(bundle,read(args.source_pool),Tokenizer.from_file(str(args.tokenizer)),common.frozen_renderer(encoder),
        tokenizer_identity=identity,official_encoding_source=encoder,official_prompt_file_content=template)
    args.output_dir.mkdir(parents=True)
    for n,v in result.items():publish(args.output_dir/n,v)
    validate_inputs(None,result['manifest.json'],result['input-protocol.json'],input_dir=args.output_dir)
    print(json.dumps(result['preparation.json'],ensure_ascii=False))


if __name__=='__main__':main()
