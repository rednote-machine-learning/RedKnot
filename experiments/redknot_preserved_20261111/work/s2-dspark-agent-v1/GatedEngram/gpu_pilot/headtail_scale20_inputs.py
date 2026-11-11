"""CPU-only headtail extension preparation; never opens or initializes a model.

Outputs are new16 evaluations plus the exact original2 calibration records.
An existing4 result set is not manufactured or rerun by this input builder.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
DATASETS = ('hotpotqa', '2wikimqa')
SEED = 20261102
ORIGINAL_SEAL = '5125e6264397dc2bee18b23106250493868b1b82add31ccb07896524da4e0bc5'
ORIGINAL_FILE = 'da80011b807f587101c8370e33ee17073e9f52cc897f224349e5075477f24300'
DEPENDENCIES = ('prepare_rag256.py', 'prepare_attention_cohort.py',
                'prepare_attention_needles.py', 'pr148_native_runner.py')


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def sealed(body):
    return dict(body, sha256=digest(body))


def verify_seal(value):
    require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'record seal mismatch')


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular nonsymlink input required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class BlockAccelerators(importlib.abc.MetaPathFinder):
    forbidden = {'torch', 'tensorflow', 'jax', 'cupy', 'cuda', 'triton', 'deepinfer'}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.forbidden:
            raise RuntimeError('input preparation forbids accelerator/model import: ' + fullname)
        return None


def exposure(original):
    verify_seal(original)
    plan = original['selection_plan']
    contexts = set(plan['excluded_context_sha256'])
    questions = set(plan['excluded_question_sha256'])
    identities = set()
    for row in original['warmup'] + original['cohort']:
        questions.add(text_hash(row['question']))
        identities.add((row['dataset'], row['source_id']))
        require(row['context_segments'], 'old complete source segments required')
        for segment in row['context_segments']:
            if segment['kind'] != 'separator':
                contexts.add(segment['context_sha256'])
                identities.add((segment['dataset'], segment['source_id']))
    return contexts, questions, identities


def build_source_plan(sources, original):
    contexts, questions, identities = exposure(original)
    grouped = {}; selected_contexts = set(); selected_questions = set()
    for dataset_ordinal, dataset in enumerate(DATASETS):
        source = sources[dataset]; eligible = []; rejected = []; seen = set()
        for index, (_, raw) in enumerate(source):
            item = dict(dataset=dataset, source_index=index, source_id=raw['_id'],
                context_sha256=text_hash(raw['context']), question_sha256=text_hash(raw['input']),
                source_row_sha256=digest(raw))
            key = item['context_sha256']
            reason = ('old_context' if key in contexts else 'old_source_identity'
                if (dataset, raw['_id']) in identities else 'old_primary_question'
                if item['question_sha256'] in questions else 'duplicate_context' if key in seen else None)
            seen.add(key)
            (rejected if reason else eligible).append(dict(item, reason=reason) if reason else item)
        order = sorted(eligible, key=lambda row: (digest(dict(seed=SEED, dataset=dataset,
            context_sha256=row['context_sha256'], source_id=row['source_id'])), row['source_index']))
        selected = []
        for item in order:
            if item['context_sha256'] in selected_contexts or item['question_sha256'] in selected_questions:
                continue
            ordinal = len(selected)
            selected.append(dict(item, ordinal=ordinal, index=6+2*ordinal+dataset_ordinal,
                anchor_chunk=(1, 4, 6)[(dataset_ordinal+ordinal+1) % 3]))
            selected_contexts.add(item['context_sha256']); selected_questions.add(item['question_sha256'])
            if len(selected) == 8: break
        require(len(selected) == 8, 'insufficient8 unique unexposed primaries: ' + dataset)
        grouped[dataset] = dict(pool=eligible, rejected=rejected, selected=selected,
            source_rows=len(source), source_inventory_sha256=digest([digest(raw) for _, raw in source]))
    for dataset in DATASETS:
        donors = [r for r in grouped[dataset]['pool'] if r['context_sha256'] not in selected_contexts
                  and r['question_sha256'] not in selected_questions]
        require(donors, 'no distinct donor sources: ' + dataset)
        grouped[dataset]['donor_source_indices'] = [r['source_index'] for r in donors]
        grouped[dataset]['donor_context_sha256'] = [r['context_sha256'] for r in donors]
    return sealed(dict(schema='headtail-link-extension-source-plan-v1', seed=SEED, datasets=grouped,
        original_manifest_sha256=original['sha256'], excluded_context_sha256=sorted(contexts),
        excluded_question_sha256=sorted(questions), excluded_source_identities=[list(i) for i in sorted(identities)],
        partition='all exact-unexposed remaining sources; original calibration reused unchanged',
        primary_selection='ascending SHA256(seed,dataset,context_sha256,source_id); index tie-break; no outcomes',
        donor_selection='source order; no reserved primary context/question; distinct within each request',
        cross_new_case_donor_sharing_allowed=True, semantic_independence_claim=False,
        token_lengths_verified=False))


def input_protocol(original, plan, tokenizer_version):
    return sealed(dict(schema='headtail-link-extension-input-protocol-v1', seed=SEED,
        original_manifest_sha256=original['sha256'], source_plan_sha256=plan['sha256'],
        retained_evaluation_indices=[2, 3, 4, 5], new_evaluation_indices=list(range(6, 22)),
        calibration_indices=[0, 1], calibration_exact_copy=True, new_cases_per_dataset=8,
        datasets=list(DATASETS), num_chunks=8, chunk_tokens=32768, context_tokens=262144,
        whole_primary_required=True, fixed_original_gold=True, new_primary_questions_unique=True,
        prior_primary_and_donor_sources_excluded=True, original_six_doc_hashes_excluded=True,
        cross_new_case_donor_sharing_allowed=True, tokenizer=original['tokenizer'],
        tokenizers_version=tokenizer_version, model_inference_seed=0,
        original_four_result_reuse_requires_separate_complete_audit=True,
        cross_run_identity_and_seals_retained=True, statistical_generalization_claim=False))


def validate_extension(manifest, protocol, original, plan, sources, base):
    verify_seal(manifest); verify_seal(protocol); verify_seal(plan)
    base.validate_manifest(manifest)
    require(manifest.get('artifact_kind') == 'headtail_link_extension_inputs_v1'
        and manifest.get('extension_input_protocol_sha256') == protocol['sha256']
        and manifest.get('original_manifest_sha256') == original['sha256']
        and manifest.get('selection_plan') == plan
        and 'sensitivity_protocol_sha256' not in manifest, 'extension input provenance differs')
    require(manifest['warmup'] == original['warmup'], 'original calibration must be exact complete copies')
    rows = manifest['cohort']
    require([r['index'] for r in rows] == list(range(6, 22))
        and [r['dataset'] for r in rows] == list(DATASETS)*8, 'exact new16/balanced8+8 required')
    contexts, questions, identities = exposure(original)
    prior_inputs = {r['input_ids_sha256'] for r in original['warmup']+original['cohort']}
    prior_docs = {d['token_ids_sha256'] for r in original['warmup']+original['cohort'] for d in r['documents']}
    new_inputs = set(); new_questions = set()
    for row in rows:
        specs = plan['datasets'][row['dataset']]
        selected = next(s for s in specs['selected'] if s['index'] == row['index'])
        raw = sources[row['dataset']][selected['source_index']][1]
        require(row['source_index'] == selected['source_index'] and row['source_id'] == raw['_id']
            and row['source_row_sha256'] == digest(raw) and row['question'] == raw['input']
            and row['references'] == raw['answers'] and row['references_sha256'] == digest(raw['answers'])
            and row['primary_context_truncated'] is False and row['split'] == 'heldout',
            'primary identity/gold/whole-context binding differs')
        ids = row['input_ids']; docs = base.document_specs(row)
        require(len(docs) == 8 and len({d['token_ids_sha256'] for d in docs}) == 8
            and all(len(d['token_ids']) == 32768 for d in docs)
            and row['context_tokens'] == 262144
            and len(ids) == row['prefix_tokens'] + 262144 + row['suffix_tokens']
            and [d['input_start'] for d in docs] == [row['prefix_tokens']+i*32768 for i in range(8)]
            and not prior_docs.intersection(d['token_ids_sha256'] for d in docs), '8x32K/original document exclusion differs')
        require(row['input_ids_sha256'] not in prior_inputs | new_inputs
            and text_hash(row['question']) not in questions | new_questions, 'duplicate/exposed evaluation input/question')
        new_inputs.add(row['input_ids_sha256']); new_questions.add(text_hash(row['question']))
        seen = set(); primaries = []
        for segment in row['context_segments']:
            if segment['kind'] == 'separator': continue
            sha = segment['context_sha256']; identity = (segment['dataset'], segment['source_id'])
            require(sha not in contexts | seen and identity not in identities, 'prior/duplicate source material')
            seen.add(sha)
            raw_segment = sources[row['dataset']][segment['source_index']][1]
            require(segment['dataset'] == row['dataset'] and segment['source_id'] == raw_segment['_id']
                and sha == text_hash(raw_segment['context']), 'source segment provenance differs')
            if segment['kind'] == 'primary_full': primaries.append(segment)
            else: require(segment['source_index'] in specs['donor_source_indices'], 'unregistered donor')
        require(len(primaries) == 1 and primaries[0]['source_index'] == row['source_index']
            and primaries[0]['context_start'] == selected['anchor_chunk']*32768
            and primaries[0]['selected_tokens'] == row['original_context_tokens'], 'whole primary anchor differs')
    return True


def combined_protocol(original, extension, extension_protocol):
    return sealed(dict(schema='headtail-scale20-input-protocol-v1', seed=SEED,
        original_manifest_sha256=original['sha256'], extension_manifest_sha256=extension['sha256'],
        extension_input_protocol_sha256=extension_protocol['sha256'],
        source_plan_sha256=extension['selection_plan']['sha256'],
        calibration_indices=[0,1], evaluation_indices=list(range(2,22)),
        reused_original_input_indices=[2,3,4,5], newly_composed_input_indices=list(range(6,22)),
        datasets=list(DATASETS), evaluation_cases_per_dataset=10, num_chunks=8,
        chunk_tokens=32768, context_tokens=262144, tokenizer=original['tokenizer'],
        tokenizers_version='0.21.4', inference_seed=0,
        calibration_exact_copy=True, original_four_inputs_exact_copy=True,
        fresh_measurements_required_for_all_twenty=True, historical_measurement_reuse=False,
        original_four_previously_exposed=True, cross_new_case_donor_sharing_allowed=True,
        independent_holdout_or_statistical_generalization_claim=False))


def validate_inputs(base, manifest, protocol, *, input_dir):
    """New20 input contract shared by its new runner and independent reader."""
    directory=Path(input_dir)
    require(file_hash(directory/'source-manifest.json')==ORIGINAL_FILE,'original source manifest bytes changed')
    original=read(directory/'source-manifest.json');verify_seal(original)
    require(original['sha256']==ORIGINAL_SEAL,'foreign original six')
    extension=read(directory/'extension-manifest.json');verify_seal(extension)
    extension_protocol=read(directory/'extension-input-protocol.json');verify_seal(extension_protocol)
    source_plan=read(directory/'source-plan.json');verify_seal(source_plan)
    require(extension['selection_plan']==source_plan,'extension source plan differs')
    sources={}
    for item in original['sources']:
        require(file_hash(item['path'])==item['sha256'],'fixed LongBench source changed')
        with Path(item['path']).open() as handle:
            sources[item['dataset']]=[(i,json.loads(line)) for i,line in enumerate(handle,1) if line.strip()]
    require(source_plan==build_source_plan(sources,original),'outcome-blind source selection does not replay')
    require(extension_protocol==input_protocol(original,source_plan,'0.21.4'),'extension input protocol differs')
    validate_extension(extension,extension_protocol,original,source_plan,sources,base)
    verify_seal(protocol);verify_seal(manifest);base.validate_manifest(manifest)
    require(protocol==combined_protocol(original,extension,extension_protocol),'combined20 protocol differs')
    require(manifest.get('artifact_kind')=='headtail_scale20_inputs_v1'
        and manifest.get('input_protocol_sha256')==manifest.get('pilot_protocol_sha256')==protocol['sha256']
        and manifest.get('source_manifest_sha256')==original['sha256']
        and manifest.get('extension_manifest_sha256')==extension['sha256']
        and manifest.get('tokenizer')==original['tokenizer'] and manifest.get('sources')==original['sources']
        and manifest.get('selection_plan')==source_plan and 'sensitivity_protocol_sha256' not in manifest,
        'combined20 manifest provenance differs')
    require(manifest['warmup']==original['warmup']
        and manifest['cohort']==original['cohort']+extension['cohort'],
        'original calibration/four inputs or new16 were changed during assembly')
    rows=manifest['warmup']+manifest['cohort']
    require([r['index'] for r in rows]==list(range(22))
        and [r['dataset'] for r in rows]==list(DATASETS)*11
        and len({r['input_ids_sha256'] for r in rows})==22,'exact balanced2cal+20evaluation required')
    for row in rows:
        docs=base.document_specs(row)
        require(len(docs)==8 and all(len(d['token_ids'])==32768 for d in docs)
            and row['context_tokens']==262144 and row['primary_context_truncated'] is False,
            'combined20 exact8x32K/whole-primary contract changed')
    return rows


def assemble_inputs(args, original, base):
    extension=read(args.extension_dir/'manifest.json')
    extension_protocol=read(args.extension_dir/'input-protocol.json')
    source_plan=read(args.extension_dir/'source-plan.json')
    preparation=read(args.extension_dir/'preparation.json');verify_seal(preparation)
    require(preparation['schema']=='headtail-link-extension-input-preparation-v1'
        and preparation['status']=='complete' and preparation['actual_token_inputs_generated'] is True
        and preparation['new_evaluation_cases']==16 and preparation['cuda_initialized'] is False,
        'complete real16 input preparation required before20 assembly')
    for name in ('manifest.json','input-protocol.json','source-plan.json'):
        value=read(args.extension_dir/name);verify_seal(value)
        require(preparation['files'][name]['file_sha256']==file_hash(args.extension_dir/name)
            and preparation['files'][name]['seal_sha256']==value['sha256'],'extension prepared bytes changed')
    protocol=combined_protocol(original,extension,extension_protocol)
    manifest=sealed(dict(schema='redknot-deepinfer-real-rag-manifest-v1',artifact_kind='headtail_scale20_inputs_v1',
        tokenizer=copy.deepcopy(original['tokenizer']),sources=copy.deepcopy(original['sources']),
        warmup=copy.deepcopy(original['warmup']),cohort=copy.deepcopy(original['cohort']+extension['cohort']),
        source_manifest_sha256=original['sha256'],extension_manifest_sha256=extension['sha256'],
        input_protocol_sha256=protocol['sha256'],pilot_protocol_sha256=protocol['sha256'],
        selection_plan=source_plan,num_chunks=8,chunk_tokens=32768,context_tokens=262144,
        instruction=original['instruction'],cache_protocol=copy.deepcopy(original.get('cache_protocol',{})),
        fresh_measurements_required=True,historical_measurement_reuse=False,
        context_claim='LongBench source composites; not native256K questions or natural document boundaries',
        original_four_previously_exposed=True,cross_new_case_donor_sharing_allowed=True))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    companions={'source-manifest.json':args.source_manifest,'extension-manifest.json':args.extension_dir/'manifest.json',
        'extension-input-protocol.json':args.extension_dir/'input-protocol.json','source-plan.json':args.extension_dir/'source-plan.json'}
    refs={}
    for name,source in companions.items():
        path=args.output_dir/name
        with path.open('xb') as handle:handle.write(source.read_bytes())
        refs[name]=dict(file_sha256=file_hash(path),bytes=path.stat().st_size)
    validate_inputs(base,manifest,protocol,input_dir=args.output_dir)
    for name,value in (('manifest.json',manifest),('input-protocol.json',protocol)):
        path=args.output_dir/name
        with path.open('x',encoding='utf-8') as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
        refs[name]=dict(file_sha256=file_hash(path),seal_sha256=value['sha256'],bytes=path.stat().st_size)
    result=sealed(dict(schema='headtail-scale20-input-preparation-v1',status='complete',evaluation_cases=20,
        calibration_cases=2,indices=list(range(2,22)),original_four_exact=True,calibration_exact=True,
        new16_preparation_sha256=preparation['sha256'],files=refs,manifest_sha256=manifest['sha256'],
        input_protocol_sha256=protocol['sha256'],cuda_initialized=False,model_results_generated=False,
        fresh_measurements_required_for_all_twenty=True,historical_measurement_reuse=False))
    with (args.output_dir/'preparation.json').open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--dependencies-dir', type=Path, required=True)
    p.add_argument('--dependency-pins', type=Path, required=True)
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--official-encoding', type=Path)
    p.add_argument('--expected-tokenizers-version', default='0.21.4',choices=['0.21.4'])
    p.add_argument('--assemble-only',action='store_true')
    p.add_argument('--extension-dir',type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv); began = time.perf_counter()
    require(not args.output_dir.exists(), 'output directory must be new and independent')
    require(not any(n.split('.')[0] in BlockAccelerators.forbidden for n in sys.modules), 'model/accelerator already imported')
    sys.meta_path.insert(0, BlockAccelerators())
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['RAYON_NUM_THREADS'] = '1'
    require(file_hash(args.source_manifest) == ORIGINAL_FILE, 'original raw manifest changed')
    original = read(args.source_manifest); verify_seal(original)
    require(original['sha256'] == ORIGINAL_SEAL, 'original sealed manifest changed')
    pins = read(args.dependency_pins); verify_seal(pins)
    require(set(pins['files']) == set(DEPENDENCIES), 'preparation dependency inventory differs')
    for name, sha in pins['files'].items():
        require(file_hash(args.dependencies_dir/name) == sha, 'preparation source drift: '+name)
    sys.path.insert(0, str(args.dependencies_dir.resolve()))
    import pr148_native_runner as base
    from prepare_rag256 import OfficialTokenizer, load_source, INSTRUCTION
    from prepare_attention_cohort import make_composite
    base.validate_manifest(original)
    require([r['index'] for r in original['warmup']+original['cohort']] == list(range(6)), 'original six scope differs')
    if args.assemble_only:
        require(args.extension_dir is not None,'assembly requires real extension input directory')
        return assemble_inputs(args,original,base)
    require(all((args.source_dir,args.tokenizer,args.official_encoding)),'tokenization requires source/tokenizer/encoding paths')
    actual_version = importlib.metadata.version('tokenizers')
    require(actual_version == args.expected_tokenizers_version, 'tokenizers runtime version differs')
    sources = {}; metadata = copy.deepcopy(original['sources'])
    source_files = {}
    for info in metadata:
        dataset = info['dataset']; require(dataset in DATASETS, 'foreign dataset')
        path = args.source_dir/(dataset+'.jsonl')
        require(file_hash(path) == info['sha256'], 'LongBench source changed: '+dataset)
        sources[dataset] = load_source(path); source_files[dataset] = dict(path=str(path.resolve()), sha256=info['sha256'])
    require(set(sources) == set(DATASETS), 'both sources required')
    t = original['tokenizer']
    tokenizer = OfficialTokenizer(args.tokenizer, args.official_encoding, t['tokenizer_json_sha256'], t['encoding_sha256'])
    require(tokenizer.identity == t, 'official chat formatter changed')
    plan = build_source_plan(sources, original); protocol = input_protocol(original, plan, actual_version)
    rows = []
    for ordinal in range(8):
        for dataset in DATASETS:
            spec = plan['datasets'][dataset]; primary = spec['selected'][ordinal]
            row = make_composite(dataset, sources[dataset], primary['source_index'], 'heldout', tokenizer,
                donors=spec['donor_source_indices'], anchor_chunk=primary['anchor_chunk'],
                partition_sha256=plan['sha256'], seed=SEED, chunk_tokens=32768, num_chunks=8)
            row.update(index=primary['index'], sample_id=f"headtail-extension-v1:{dataset}:{row['source_id']}")
            rows.append(row)
    manifest = sealed(dict(schema='redknot-deepinfer-real-rag-manifest-v1',
        artifact_kind='headtail_link_extension_inputs_v1', tokenizer=tokenizer.identity, sources=metadata,
        warmup=copy.deepcopy(original['warmup']), cohort=rows, selection_plan=plan,
        original_manifest_sha256=original['sha256'], extension_input_protocol_sha256=protocol['sha256'],
        num_chunks=8, chunk_tokens=32768, context_tokens=262144, instruction=INSTRUCTION,
        cache_protocol=copy.deepcopy(original.get('cache_protocol', {})),
        context_claim='LongBench source composites; not native256K questions or natural document boundaries',
        source_exclusion_claim='all declared exact old context/ID/question identities; no semantic independence',
        cross_new_case_donor_sharing_allowed=True))
    validate_extension(manifest, protocol, original, plan, sources, base)
    require(not any(n.split('.')[0] in BlockAccelerators.forbidden for n in sys.modules), 'preparation imported accelerator')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    refs = {}
    for name, value in [('manifest.json', manifest), ('input-protocol.json', protocol), ('source-plan.json', plan)]:
        path = args.output_dir/name
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
        refs[name] = dict(file_sha256=file_hash(path), seal_sha256=value['sha256'], bytes=path.stat().st_size)
    result = sealed(dict(schema='headtail-link-extension-input-preparation-v1', status='complete',
        actual_token_inputs_generated=True, cuda_initialized=False, accelerator_modules_imported=False,
        new_evaluation_cases=16, calibration_cases=2, retained_evaluation_cases_in_other_run=4,
        source_manifest_file_sha256=ORIGINAL_FILE, source_manifest_sha256=ORIGINAL_SEAL,
        source_files=source_files, dependency_pins=pins, builder_sha256=file_hash(Path(__file__)),
        python=sys.version, tokenizers_version=actual_version, official_tokenizer=t, files=refs,
        cpu_thread_limits=dict(TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1'),
        row_inventory=[{k:r[k] for k in ('index','dataset','source_index','source_id','sample_id','input_tokens',
                                       'input_ids_sha256','references_sha256')} for r in rows],
        elapsed_s=time.perf_counter()-began, model_results_generated=False))
    with (args.output_dir/'preparation.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
