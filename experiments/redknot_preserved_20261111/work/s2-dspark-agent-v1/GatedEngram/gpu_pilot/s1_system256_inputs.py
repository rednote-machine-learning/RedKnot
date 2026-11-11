"""Admit three already-tokenized natural Loong256K QAs, without reselection.

The original five-file CPU probe is reconstructed and its pinned seals checked.
No tokenizer, model, remote process, padding, truncation or new QA is used.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from gpu_pilot import loong_inputs as loong

require,digest,verify,seal,read,publish=loong.require,loong.digest,loong.verify,loong.seal,loong.read,loong.publish
MANIFEST_SCHEMA='di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA='kv-cache-engram-s1-system256-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA='s1-system256-input-protocol-v1'
INPUT_COMPANIONS=loong.INPUT_COMPANIONS
PREVIOUS_FILES=('manifest.json','input-protocol.json',*INPUT_COMPANIONS)
PROBE_FILES=('preparation.json','selected-rows.json',*INPUT_COMPANIONS)
PREVIOUS_MANIFEST_SHA256='fb6bba62937fb8005f05df9a3a213729c172932a30e5509b136000692881aa67'
PREVIOUS_PROTOCOL_SHA256='877db63dbaf0099536d64f011cb6ad08956d3970654a954adb5b8f541806529a'
PROBE_PREPARATION_SHA256='69e71aca68ce7ed7007c6c9608823656f07d463e7b537d90c7f7fb5bd5d0827f'
PROBE_SELECTED_ROWS_SHA256='6b3374e210452a92d5c9df2df4d2436b4733acc2af24a6cba3cf3f68c97b6e8b'
SELECTED_SOURCE_INDICES=(1039,1036,1035)
EVALUATION_INDICES=(2,3,4)
LENGTH_BANDS=('256k',)
INPUT_TOKEN_RANGE=(240*1024,260*1024)
TARGET_INPUT_TOKENS=256*1024
ENGINE_MAX_MODEL_LEN=270336
TOKENIZER_EXECUTION_FIELDS=('tokenizer_json_sha256','encoding_sha256','loader','prompt_renderer','thinking_mode','tokenizers_version')
CALIBRATION_PROMPT_PROVENANCE_FIELDS=('official_longbench_prompt_revision','official_longbench_prompt_sha256')
SCHEMAS={'source-records.json':'s1-system256-source-records-v1',
         'selection-plan.json':'s1-system256-selection-plan-v1',
         'tokenization-evidence.json':'s1-system256-tokenization-evidence-v1'}
PROBE_SCHEMAS={'source-records.json':'loong256k-source-records-v1',
              'selection-plan.json':'loong256k-selection-plan-v1',
              'tokenization-evidence.json':'loong256k-tokenization-evidence-v1'}


def fixed_policy():
    return dict(evaluation_cases=3,evaluation_indices=list(EVALUATION_INDICES),calibration_indices=[0,1],
        datasets=['loong'],length_bands=['256k'],target_input_tokens={'256k':TARGET_INPUT_TOKENS},
        input_token_ranges={'256k':list(INPUT_TOKEN_RANGE)},cases_per_length_band=3,
        num_chunks=10,min_block_tokens=256,original_document_count_range=[5,10],engine_max_model_len=ENGINE_MAX_MODEL_LEN,
        selected_source_indices=list(SELECTED_SOURCE_INDICES),reselection_performed=False,
        no_model_outcomes_used=True,no_padding=True,no_truncation=True,no_cross_sample_documents=True,
        no_cross_original_document_cache_blocks=True,preserve_original_order=True,
        original_question_instruction_gold_preserved=True,calibration_source=loong.CALIBRATION_SOURCE,
        calibration_in_quality_metrics=False,actual_tokenization_evidence_reused=True,new_tokenization_executed=False,
        tokenizer_execution_identity_fields=list(TOKENIZER_EXECUTION_FIELDS),
        calibration_prompt_provenance_fields=list(CALIBRATION_PROMPT_PROVENANCE_FIELDS),
        original_tokenizer_metadata_preserved=True,
        no_answer_content_selection=True,natural_eos_preserved=True,minimum_generated_tokens=None,
        official_semantic_judge_executed=False,primary_metric='loong_custom_structured_exact',
        source1036_original_gold_preserved_with_separate_unit_and_combined_field_annotation=True,
        source_quality_annotations=[dict(source_index=1036,original_gold='$1,740',automatic_scores_unchanged=True,
            supplemental_manual_annotation=True,selection_changed=False,
            source_review_sha256='0ac96cc81ae839334f1002a3256fe022ecd9f6c2de945cadefa66ac59428f20e',
            source_document_sha256='05153fe773999b618287ce2292e065a58386b633ae022b4c205bca07b73bec6a',
            note='Original balance sheet is in thousands; 1,740 labels Accounts payable and accrued expenses. The original gold is retained verbatim; it is not an unambiguous stand-alone dollar amount for Accounts payable.')],
        previous128_manifest_sha256=PREVIOUS_MANIFEST_SHA256,previous128_protocol_sha256=PREVIOUS_PROTOCOL_SHA256,
        probe_preparation_sha256=PROBE_PREPARATION_SHA256,probe_selected_rows_sha256=PROBE_SELECTED_ROWS_SHA256)


def tokenizer_execution_identity(value):
    """Separate tokenizer execution from optional historical calibration prompts.

    The two prompt-source fields describe the old LongBench calibration task,
    not tokenizer vocabulary, encoder code or rendering mode. Preserve them in
    the original calibration metadata; never add them to the sealed probe.
    """
    require(type(value) is dict and all(type(v) is str and v for v in value.values()),
        'tokenizer metadata must contain nonempty strings')
    fields=set(TOKENIZER_EXECUTION_FIELDS);legacy=set(CALIBRATION_PROMPT_PROVENANCE_FIELDS)
    require(set(value) in (fields,fields|legacy),'unknown or incomplete tokenizer metadata fields')
    return {key:value[key] for key in TOKENIZER_EXECUTION_FIELDS}


def validate_probe(probe):
    require(set(probe)==set(PROBE_FILES),'exact five original probe files required')
    for value in probe.values():verify(value)
    prep,chosen=probe['preparation.json'],probe['selected-rows.json']
    source,selection,tokens=[probe[n] for n in INPUT_COMPANIONS]
    require(prep.get('schema')=='loong256k-preparation-v1' and prep['sha256']==PROBE_PREPARATION_SHA256
        and chosen.get('schema')=='loong256k-selected-rows-v1' and chosen['sha256']==PROBE_SELECTED_ROWS_SHA256,
        'pinned original CPU preparation/selected rows differ')
    for name in INPUT_COMPANIONS:require(probe[name].get('schema')==PROBE_SCHEMAS[name],'probe companion schema')
    require(prep['selected_rows_sha256']==chosen['sha256'] and prep['status']=='complete'
        and prep['selected_count']==prep['requested_count']==3 and prep['cuda_used'] is False
        and prep['inference_run'] is False and prep['model_outputs_used'] is False,
        'probe must contain three pre-output CPU candidates')
    require(chosen['formal_experiment_manifest'] is False and chosen['calibration_rows_added'] is False,
        'probe cannot masquerade as earlier formal experiment')
    for key,value in (('source_records_sha256',source),('selection_plan_sha256',selection),('tokenization_evidence_sha256',tokens)):
        require(chosen[key]==value['sha256'],'probe evidence binding differs')
    rows=chosen['rows']
    require(len(rows)==3 and tuple(r['source_index'] for r in rows)==SELECTED_SOURCE_INDICES
        and [r['selection_ordinal'] for r in rows]==[1,2,3]
        and len({r['context_sha256'] for r in rows})==3,'original selected three/order differs')
    require(selection['selected_sample_ids']==[r['sample_id'] for r in rows]
        and selection['selected_count']==selection['requested_count']==3
        and selection['no_model_outcomes_used'] is True and selection['no_answer_content_selection'] is True,
        'original selection policy differs')
    require(tokens['tokenizer']==prep['actual_tokenizer_identity'],'probe actual tokenizer differs')
    return rows


def _previous(previous):
    require(set(previous)==set(PREVIOUS_FILES),'five original128 input files required')
    for v in previous.values():verify(v)
    manifest,protocol=previous['manifest.json'],previous['input-protocol.json']
    require(manifest['sha256']==PREVIOUS_MANIFEST_SHA256 and protocol['sha256']==PREVIOUS_PROTOCOL_SHA256
        and manifest['input_protocol_sha256']==protocol['sha256'],'original128 input pins differ')
    for name in INPUT_COMPANIONS:
        require(manifest['companion_sha256'][name]==previous[name]['sha256'],'original128 companion binding')
    ancestral=previous['source-records.json']['previous_inputs']
    prior=loong.reuse.validate_previous_bundle(ancestral)
    require(manifest['warmup']==prior['warmup'] and manifest['tokenizer']==prior['tokenizer'],
        'original128 calibration differs from its pinned ancestor')
    ids={r['sample_id'] for r in prior['warmup']}
    for name in ('source-records.json','tokenization-evidence.json'):
        old={r['sample_id']:r for r in ancestral[name]['records']}
        actual={r['sample_id']:r for r in previous[name]['records'] if r['sample_id'] in ids}
        require(actual=={k:old[k] for k in ids},'original128 calibration evidence changed')
    return prior,ancestral


def build_inputs(previous_bundle,probe_bundle):
    original_rows=validate_probe(probe_bundle);prior,ancestral=_previous(previous_bundle)
    actual_tokenizer=tokenizer_execution_identity(probe_bundle['tokenization-evidence.json']['tokenizer'])
    require(actual_tokenizer==tokenizer_execution_identity(prior['tokenizer']),
        'calibration and256K actual tokenizer differ')
    cohort=[]
    for ordinal,original in enumerate(original_rows,1):
        require(not any(k in original for k in ('index','split','length_band_ordinal')),'probe row already indexed')
        cohort.append(dict(copy.deepcopy(original),index=ordinal+1,split='evaluation',length_band_ordinal=ordinal))
    warmup=copy.deepcopy(previous_bundle['manifest.json']['warmup']);calids={r['sample_id'] for r in warmup}
    companions={}
    for name in INPUT_COMPANIONS:
        value={k:copy.deepcopy(v) for k,v in probe_bundle[name].items() if k!='sha256'}
        value['schema']=SCHEMAS[name]
        if name in ('source-records.json','tokenization-evidence.json'):
            value['records']=[copy.deepcopy(r) for r in previous_bundle[name]['records'] if r['sample_id'] in calids]+value['records']
        if name=='source-records.json':value['previous_inputs']=copy.deepcopy(ancestral)
        companions[name]=seal(value)
    protocol=seal(dict(schema=INPUT_PROTOCOL_SCHEMA,**fixed_policy(),tokenizer=prior['tokenizer'],
        actual_tokenizer_identity=actual_tokenizer,
        calibration_prompt_provenance={k:prior['tokenizer'][k] for k in CALIBRATION_PROMPT_PROVENANCE_FIELDS if k in prior['tokenizer']},
        previous_manifest_sha256=prior['sha256'],
        probe_preparation=copy.deepcopy(probe_bundle['preparation.json']),
        probe_selected_header={k:copy.deepcopy(v) for k,v in probe_bundle['selected-rows.json'].items() if k not in ('rows','sha256')},
        source_records_sha256=companions['source-records.json']['sha256'],
        selection_plan_sha256=companions['selection-plan.json']['sha256'],
        tokenization_evidence_sha256=companions['tokenization-evidence.json']['sha256']))
    manifest=seal(dict(schema=MANIFEST_SCHEMA,contract_schema=CONTRACT_SCHEMA,artifact_kind='s1_system256_inputs_v1',
        natural_boundary_inputs=True,input_protocol_sha256=protocol['sha256'],tokenizer=prior['tokenizer'],
        num_chunks=10,chunk_tokens=None,context_tokens=None,calibration_source=loong.CALIBRATION_SOURCE,
        warmup=warmup,cohort=cohort,sources=copy.deepcopy(previous_bundle['manifest.json']['sources']),
        companion_sha256={k:v['sha256'] for k,v in companions.items()},selection_plan_sha256=companions['selection-plan.json']['sha256']))
    summary=seal(dict(schema='s1-system256-input-preparation-v1',status='complete',cuda_used=False,inference_run=False,
        new_tokenization_executed=False,model_results_used=False,manifest_sha256=manifest['sha256'],
        input_protocol_sha256=protocol['sha256'],evaluation_cases=3,source_qa_count=3,
        probe_preparation_sha256=PROBE_PREPARATION_SHA256,probe_selected_rows_sha256=PROBE_SELECTED_ROWS_SHA256,
        samples=[dict(index=r['index'],source_index=r['source_index'],source_id=r['source_id'],input_tokens=r['input_tokens'],
            original_document_count=r['original_document_count'],block_tokens=[d['tokens'] for d in r['documents']],
            technical_max_new_tokens=ENGINE_MAX_MODEL_LEN-r['input_tokens']) for r in cohort]))
    result=dict(companions,**{'manifest.json':manifest,'input-protocol.json':protocol,'preparation.json':summary})
    _validate_bundle(None,manifest,protocol,companions)
    return result


def _validate_bundle(base,manifest,protocol,companions):
    verify(manifest);verify(protocol)
    require(manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('contract_schema')==CONTRACT_SCHEMA
        and manifest.get('artifact_kind')=='s1_system256_inputs_v1' and protocol.get('schema')==INPUT_PROTOCOL_SCHEMA
        and manifest['input_protocol_sha256']==protocol['sha256'],'system256 manifest/protocol schema or binding')
    require(all(protocol.get(k)==v for k,v in fixed_policy().items()),'system256 fixed policy differs')
    require(set(companions)==set(INPUT_COMPANIONS) and set(manifest['companion_sha256'])==set(INPUT_COMPANIONS),
        'system256 companion inventory')
    for name,value in companions.items():
        verify(value)
        require(value['schema']==SCHEMAS[name] and manifest['companion_sha256'][name]==value['sha256'],
            'system256 companion seal/schema differs')
    source,selection,tokens=[companions[n] for n in INPUT_COMPANIONS]
    for key,value in (('source_records_sha256',source),('selection_plan_sha256',selection),('tokenization_evidence_sha256',tokens)):
        require(protocol[key]==value['sha256'],'system256 evidence link differs')
    prior=loong.reuse.validate_previous_bundle(source['previous_inputs'])
    require(manifest['warmup']==prior['warmup'] and len(manifest['warmup'])==2
        and manifest['tokenizer']==protocol['tokenizer']==prior['tokenizer']
        and protocol['previous_manifest_sha256']==prior['sha256'],'exact calibration/tokenizer ancestry differs')
    require(tokenizer_execution_identity(tokens['tokenizer'])==tokenizer_execution_identity(prior['tokenizer'])
        ==protocol['actual_tokenizer_identity'],'calibration and256K actual tokenizer differ')
    require(protocol['calibration_prompt_provenance']=={k:prior['tokenizer'][k]
        for k in CALIBRATION_PROMPT_PROVENANCE_FIELDS if k in prior['tokenizer']},
        'original calibration prompt provenance differs')
    require(manifest['num_chunks']==10 and manifest['natural_boundary_inputs'] is True
        and manifest['calibration_source']==loong.CALIBRATION_SOURCE
        and manifest['selection_plan_sha256']==selection['sha256'],'natural geometry header differs')
    require(manifest['sources']==[dict(dataset='loong',revision=loong.SOURCE_REVISION,source_jsonl_sha256=loong.SOURCE_SHA256,
        document_archive_sha256=loong.DOCUMENT_ARCHIVE_SHA256)],'pinned source inventory differs')
    rows=manifest['warmup']+manifest['cohort'];calids={r['sample_id'] for r in manifest['warmup']}
    require([r['index'] for r in rows]==list(range(5)) and len(manifest['cohort'])==3,'exact2cal+3evaluation required')
    probe={}
    for name in INPUT_COMPANIONS:
        value={k:copy.deepcopy(v) for k,v in companions[name].items() if k!='sha256'}
        value['schema']=PROBE_SCHEMAS[name]
        if name in ('source-records.json','tokenization-evidence.json'):
            records=value['records'];by_id={r['sample_id']:r for r in records}
            require(len(by_id)==len(records)==5 and set(by_id)=={r['sample_id'] for r in rows},'source/token record inventory')
            old={r['sample_id']:r for r in source['previous_inputs'][name]['records']}
            require(all(by_id[k]==old[k] for k in calids),'calibration evidence changed')
            value['records']=[r for r in records if r['sample_id'] not in calids]
        if name=='source-records.json':value.pop('previous_inputs')
        probe[name]=seal(value)
    originals=[]
    for ordinal,row in enumerate(manifest['cohort'],1):
        require(row['split']=='evaluation' and row['length_band_ordinal']==ordinal,'evaluation ordinal differs')
        originals.append({k:copy.deepcopy(v) for k,v in row.items() if k not in ('index','split','length_band_ordinal')})
    probe['selected-rows.json']=seal(dict(protocol['probe_selected_header'],rows=originals))
    probe['preparation.json']=protocol['probe_preparation']
    validate_probe(probe)  # Reconstructed exact original seals bind every ID, offset and source byte.
    raw_rows=loong.source_rows(source['official_source_jsonl'].encode())
    require(source['source_revision']==loong.SOURCE_REVISION and source['source_jsonl_sha256']==loong.SOURCE_SHA256
        and source['document_archive_sha256']==loong.DOCUMENT_ARCHIVE_SHA256
        and source['document_catalog_sha256']==digest(source['document_catalog'])==loong.DOCUMENT_CATALOG_SHA256,
        'actual source/catalog identities differ')
    files=source['document_files'];catalog=source['document_catalog']
    require(all(name in catalog and loong.file_catalog({name:text})[name]==catalog[name] for name,text in files.items()),
        'selected complete document bytes differ')
    require(loong.text_hash(tokens['official_encoding_source'])==loong.common.OFFICIAL_ENCODING_SHA256
        and loong.text_hash(tokens['official_prompt_source'])==loong.OFFICIAL_PROMPT_SOURCE_SHA256,'official render source differs')
    render=loong.common.frozen_renderer(tokens['official_encoding_source'])
    events={e['sample_id']:e for e in tokens['records']};records={e['sample_id']:e for e in source['records']};used=set()
    for row in manifest['cohort']:
        n=row['input_tokens'];ids=row['input_ids'];raw=raw_rows[row['source_index']];ev=events[row['sample_id']]
        require(row['length_band']=='256k' and row['target_input_tokens']==TARGET_INPUT_TOKENS
            and INPUT_TOKEN_RANGE[0]<=n<=INPUT_TOKEN_RANGE[1] and row['technical_max_new_tokens']==ENGINE_MAX_MODEL_LEN-n>=4096,
            '256K natural length/remaining engine context differs')
        require(all(row.get(k)==v for k,v in loong.source_metadata(raw,row['source_index']).items()),'original QA identity differs')
        require(row['question']==raw['question'] and row['instruction']==raw['instruction']
            and row['prompt_template']==raw['prompt_template'] and row['raw_answer']==raw['answer']
            and row['raw_answer_sha256']==digest(raw['answer']),'original QA or gold changed')
        require(records[row['sample_id']]==dict(sample_id=row['sample_id'],source_index=row['source_index'],
            source_row_sha256=digest(raw),raw=raw),'original source record differs')
        originals=loong.original_documents(raw,files);used.update(d['source_file'] for d in originals)
        require(5<=len(originals)<=10 and len(set(raw['doc']))==len(raw['doc']),'original doc cardinality/duplicates')
        prompt,context=loong.render_original(raw,originals,render);start=prompt.index(context)
        require(row['prompt']==prompt and row['context_text']==context and row['prompt_sha256']==ev['prompt_sha256']==loong.text_hash(prompt)
            and n==len(ids) and row['input_ids_sha256']==ev['input_ids_sha256']==digest(ids)
            and all(type(x) is int and 0<=x<2**31 for x in ids),'complete actual prompt/IDs differ')
        require(len(ev['token_offsets'])==n and ev['additional_source_documents']==0
            and all(ev[k] is True for k in ('whole_prompt_single_encoding','decode_matches_rendered_prompt','reencode_matches_input_ids',
                'full_source_documents_preserved','full_source_question_preserved','full_source_instruction_preserved','full_source_template_preserved')),
            'preserved original tokenization proof differs')
        actual,groups=loong.cache_geometry(originals,ev['token_offsets'],start)
        docs=loong.documents_from_groups(context,ids,groups)
        for d,g in zip(docs,groups):d.update({k:g[k] for k in ('source_document_ordinal','source_document_name','source_doc_char_start','source_doc_char_end')})
        require(row['original_documents']==actual and row['documents']==docs and len(docs)==10
            and all(d['tokens']>=256 for d in docs)
            and ''.join(context[d['context_char_start']:d['context_char_end']] for d in docs)==context,
            'complete natural document subdivision differs')
        require(row['source_order_preserved'] is True and row['primary_context_truncated'] is False
            and row['natural_boundary_inputs'] is True and row['original_document_order']==raw['doc']
            and row['document_tokens_total']==sum(d['tokens'] for d in docs)
            and row['input_non_document_tokens']==n-row['document_tokens_total'],'original text/gap preservation differs')
    require(set(files)==used,'unbound document file')
    require(manifest['cohort'][0]['input_tokens']==max(r['input_tokens'] for r in manifest['cohort']),
        'case2 must be longest for full-input qualification')
    if base is not None:
        base.validate_manifest(manifest)
        for row in rows:base.document_specs(row)
    return True


def validate_inputs(base,manifest,protocol,*,input_dir):
    return _validate_bundle(base,manifest,protocol,{n:read(Path(input_dir)/n) for n in INPUT_COMPANIONS})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-input-dir','probe-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv)
    require(not args.output_dir.exists(),'never overwrite frozen or previous inputs')
    previous={n:read(args.previous_input_dir/n) for n in PREVIOUS_FILES}
    probe={n:read(args.probe_dir/n) for n in PROBE_FILES}
    result=build_inputs(previous,probe)
    args.output_dir.mkdir(parents=True)
    for name,value in result.items():publish(args.output_dir/name,value)
    print(json.dumps(result['preparation.json'],ensure_ascii=False))


if __name__=='__main__':main()
