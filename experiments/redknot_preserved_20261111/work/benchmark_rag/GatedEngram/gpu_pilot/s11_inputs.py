"""Combine sealed, previously selected inputs for Full/S1.1 without retokenizing.

The remaining128 case is determined by a separately pinned completion receipt,
never by answer quality. Original calibration/evaluation rows retain all bytes,
indices, prompts, gold values and document boundaries. No CUDA is imported.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gpu_pilot import s1_system256_inputs as previous256

require, digest, verify, seal, read, publish = (
    previous256.require, previous256.digest, previous256.verify,
    previous256.seal, previous256.read, previous256.publish)
INPUT_COMPANIONS = previous256.INPUT_COMPANIONS
ORIGINAL_FILES = previous256.PREVIOUS_FILES
MANIFEST_SCHEMA = 'di-sg-real-rag-manifest-v1'
CONTRACT_SCHEMA = 'kv-cache-engram-s11-input-manifest-v1'
INPUT_PROTOCOL_SCHEMA = 's11-input-protocol-v1'
PREDECESSOR_SCHEMA = 's11-predecessor-admission-v1'
SYSTEM256_MANIFEST_SHA256 = '755b78a6f187cd18c16f0c0b34e186945a36f11551f12bd8b0ec81367a5399ee'
SYSTEM256_PROTOCOL_SHA256 = 'e5dbbf5aeba1b417c15e6c1204440ce26403e19270ff81453e5e6ae7d918df2f'
EVALUATION_INDICES = (20, 2, 3, 4)
STORED_EVALUATION_INDICES = (2, 3, 4, 20)
QUALIFICATION_ANCHOR_INDEX = 2
SCHEMAS = dict(zip(INPUT_COMPANIONS,
    ('s11-source-records-v1', 's11-selection-plan-v1', 's11-tokenization-evidence-v1')))


def _sha(value, label):
    require(type(value) is str and len(value) == 64
            and all(x in '0123456789abcdef' for x in value), label + ' must be a SHA256')


def validate_predecessor(receipt, original_manifest, *, expected_sha256):
    verify(receipt); _sha(expected_sha256, 'expected predecessor')
    require(receipt.get('schema') == PREDECESSOR_SCHEMA and receipt['sha256'] == expected_sha256,
            'independently pinned predecessor receipt differs')
    require(receipt.get('manifest_sha256') == original_manifest['sha256'],
            'predecessor original128 input identity differs')
    rows = {r['index']: r for r in original_manifest['cohort'] if r['length_band'] == '128k'}
    planned = list(range(2, 22, 2)); completed = planned[:-1]
    require(sorted(rows) == planned and receipt.get('registered_case_indices') == planned
            and receipt.get('completed_case_indices') == completed
            and receipt.get('pending_case_indices') == [20]
            and receipt.get('new_evaluation_case_indices') == list(EVALUATION_INDICES),
            'remaining128 must be the sole unaudited case20 of the original ten')
    require(receipt.get('status') == 'accepted_user_stop' and receipt.get('source_run_count') == 2
            and receipt.get('single_run') is False
            and receipt.get('exposed_arms') == ['recomputed', 's1_system']
            and receipt.get('user_requested_stop_authenticated') is True
            and receipt.get('physical_idle_verified') is True
            and type(receipt.get('stable_scope')) is dict and receipt['stable_scope'],
            'predecessor must bind the authenticated stop and two actual runs')
    def reference(value):
        require(type(value) is dict and type(value.get('path')) is str and value['path'],
                'predecessor evidence path missing')
        for key in ('file_sha256', 'seal_sha256'): _sha(value.get(key), key)
    evidence = receipt.get('audits')
    require(type(evidence) is list and len(evidence) == 2, 'two original audit sources required')
    accepted = []
    for item in evidence:
        require(type(item.get('run_id')) is str and item['run_id']
                and type(item.get('accepted_case_indices')) is list and item['accepted_case_indices'],
                'predecessor actual run/case scope missing')
        accepted.extend(item['accepted_case_indices'])
        for key in ('audit', 'preflight', 'workflow_terminal', 'lifecycle_terminal'): reference(item.get(key))
    require(len({x['run_id'] for x in evidence}) == 2 and sorted(accepted) == completed,
            'audited predecessor sources must partition the nine completed128 cases')
    for key in ('user_stop_request', 'physical_idle'): reference(receipt.get(key))
    return True


def _validate_originals(original128, system256):
    previous256._previous(original128)
    require(set(system256) == set(ORIGINAL_FILES), 'exact five original256 input files required')
    for value in system256.values(): verify(value)
    require(system256['manifest.json']['sha256'] == SYSTEM256_MANIFEST_SHA256
            and system256['input-protocol.json']['sha256'] == SYSTEM256_PROTOCOL_SHA256,
            'pinned original256 manifest/protocol differ')
    previous256._validate_bundle(None, system256['manifest.json'], system256['input-protocol.json'],
                                {name: system256[name] for name in INPUT_COMPANIONS})
    a, b = original128['manifest.json'], system256['manifest.json']
    require(a['warmup'] == b['warmup'] and a['tokenizer'] == b['tokenizer']
            and a['sources'] == b['sources'], 'calibration/tokenizer/source ancestry differs')
    require([r['index'] for r in a['cohort']] == list(range(2, 22)),
            'original128 complete twenty-input inventory differs')
    require([r['index'] for r in b['cohort']] == [2, 3, 4], 'original256 three-input inventory differs')


def fixed_policy():
    return dict(evaluation_cases=4, evaluation_indices=list(EVALUATION_INDICES),
        execution_indices=list(EVALUATION_INDICES), stored_evaluation_indices=list(STORED_EVALUATION_INDICES),
        qualification_anchor_index=QUALIFICATION_ANCHOR_INDEX, calibration_indices=[0, 1],
        datasets=['loong'], length_bands=['128k', '256k'],
        target_input_tokens={'128k': previous256.loong.TARGET_INPUT_TOKENS['128k'],
                             '256k': previous256.TARGET_INPUT_TOKENS},
        input_token_ranges={'128k': list(previous256.loong.INPUT_TOKEN_RANGES['128k']),
                            '256k': list(previous256.INPUT_TOKEN_RANGE)},
        evaluation_cases_by_length_band={'128k': 1, '256k': 3}, num_chunks=10, min_block_tokens=256,
        original_document_count_range=[5, 10], engine_max_model_len=previous256.ENGINE_MAX_MODEL_LEN,
        original_indices_preserved=True, original_rows_preserved=True, no_reselection=True,
        remaining128_selection_basis='independently_audited_completion_only',
        no_model_quality_used_for_selection=True, no_padding=True, no_truncation=True,
        no_cross_sample_documents=True, no_cross_original_document_cache_blocks=True,
        preserve_original_order=True, original_question_instruction_gold_preserved=True,
        calibration_source=previous256.loong.CALIBRATION_SOURCE, calibration_in_quality_metrics=False,
        actual_tokenization_evidence_reused=True, new_tokenization_executed=False,
        natural_eos_preserved=True, minimum_generated_tokens=None,
        official_semantic_judge_executed=False, primary_metric='loong_custom_structured_exact',
        previous128_manifest_sha256=previous256.PREVIOUS_MANIFEST_SHA256,
        previous128_protocol_sha256=previous256.PREVIOUS_PROTOCOL_SHA256,
        original256_manifest_sha256=SYSTEM256_MANIFEST_SHA256,
        original256_protocol_sha256=SYSTEM256_PROTOCOL_SHA256)


def _rows(original128, system256):
    m = system256['manifest.json']
    remaining = next(r for r in original128['manifest.json']['cohort'] if r['index'] == 20)
    return copy.deepcopy(m['warmup']), copy.deepcopy(m['cohort'] + [remaining])


def _merged_evidence(original128, system256, warmup, cohort):
    """Selected evidence is a literal subset of the two original sealed bundles."""
    rows = warmup + cohort
    selected_ids = [r['sample_id'] for r in rows]
    require(len(set(selected_ids)) == len(rows), 'selected original sample IDs overlap')
    output = {}
    for name in ('source-records.json', 'tokenization-evidence.json'):
        a, b = original128[name], system256[name]
        result = {k: copy.deepcopy(v) for k, v in a.items() if k != 'sha256'}
        result['schema'] = SCHEMAS[name]
        records = {r['sample_id']: r for r in b['records']}
        missing = set(selected_ids) - set(records)
        records.update({r['sample_id']: r for r in a['records'] if r['sample_id'] in missing})
        require(set(selected_ids) <= set(records), 'selected original source/token evidence missing')
        result['records'] = [copy.deepcopy(records[k]) for k in selected_ids]
        if name == 'source-records.json':
            for key in ('official_source_jsonl', 'source_jsonl_sha256', 'source_revision',
                        'document_archive_sha256', 'document_catalog_sha256', 'document_catalog'):
                require(a[key] == b[key], 'original source corpus/catalog differs: ' + key)
            files = {**a['document_files'], **b['document_files']}
            for key in set(a['document_files']) & set(b['document_files']):
                require(a['document_files'][key] == b['document_files'][key], 'shared original document bytes differ')
            used = {d['source_file'] for r in cohort for d in r['original_documents']}
            result['document_files'] = {k: files[k] for k in sorted(used)}
        else:
            for key in ('official_encoding_source', 'official_prompt_source'):
                require(a[key] == b[key], 'official original renderer differs')
            require(previous256.tokenizer_execution_identity(a['tokenizer']) ==
                    previous256.tokenizer_execution_identity(b['tokenizer']), 'actual tokenizer identity differs')
        output[name] = seal(result)
    return output


def build_inputs(original128, system256, predecessor, *, expected_predecessor_sha256):
    _validate_originals(original128, system256)
    validate_predecessor(predecessor, original128['manifest.json'], expected_sha256=expected_predecessor_sha256)
    warmup, cohort = _rows(original128, system256)
    companions = _merged_evidence(original128, system256, warmup, cohort)
    selection = seal(dict(schema=SCHEMAS['selection-plan.json'],
        original_input_bundles=copy.deepcopy(dict(original128=original128, system256=system256)),
        predecessor=copy.deepcopy(predecessor), expected_predecessor_sha256=expected_predecessor_sha256,
        stored_evaluation_indices=list(STORED_EVALUATION_INDICES), execution_indices=list(EVALUATION_INDICES),
        selected_sample_ids=[r['sample_id'] for r in cohort],
        selected_original_rows_sha256={str(r['index']): digest(r) for r in cohort},
        original_input_row_origin={str(r['index']): ('original128' if r['index'] == 20 else 'system256') for r in cohort},
        selection_basis='unfinished128_plus_preselected256_no_quality_selection'))
    companions['selection-plan.json'] = selection
    original_protocol = system256['input-protocol.json']
    protocol = seal(dict(schema=INPUT_PROTOCOL_SCHEMA, **fixed_policy(), tokenizer=copy.deepcopy(system256['manifest.json']['tokenizer']),
        actual_tokenizer_identity=copy.deepcopy(original_protocol['actual_tokenizer_identity']),
        source_quality_annotations=copy.deepcopy(original_protocol['source_quality_annotations']),
        expected_predecessor_sha256=expected_predecessor_sha256,
        source_records_sha256=companions['source-records.json']['sha256'],
        selection_plan_sha256=selection['sha256'],
        tokenization_evidence_sha256=companions['tokenization-evidence.json']['sha256']))
    manifest = seal(dict(schema=MANIFEST_SCHEMA, contract_schema=CONTRACT_SCHEMA, artifact_kind='s11_inputs_v1',
        natural_boundary_inputs=True, input_protocol_sha256=protocol['sha256'],
        tokenizer=copy.deepcopy(system256['manifest.json']['tokenizer']), num_chunks=10,
        chunk_tokens=None, context_tokens=None, calibration_source=previous256.loong.CALIBRATION_SOURCE,
        warmup=warmup, cohort=cohort, sources=copy.deepcopy(system256['manifest.json']['sources']),
        companion_sha256={n: companions[n]['sha256'] for n in INPUT_COMPANIONS}, selection_plan_sha256=selection['sha256']))
    preparation = seal(dict(schema='s11-input-preparation-v1', status='complete', cuda_used=False,
        inference_run=False, new_tokenization_executed=False, selection_uses_model_quality=False,
        manifest_sha256=manifest['sha256'], input_protocol_sha256=protocol['sha256'],
        expected_predecessor_sha256=expected_predecessor_sha256, evaluation_cases=4,
        execution_indices=list(EVALUATION_INDICES), qualification_anchor_index=2,
        samples=[dict(index=r['index'], length_band=r['length_band'], input_tokens=r['input_tokens'],
            input_ids_sha256=r['input_ids_sha256'], source_index=r['source_index'],
            original_document_count=r['original_document_count'], block_tokens=[d['tokens'] for d in r['documents']],
            technical_max_new_tokens=previous256.ENGINE_MAX_MODEL_LEN-r['input_tokens'])
            for i in EVALUATION_INDICES for r in cohort if r['index'] == i]))
    return dict(companions, **{'manifest.json': manifest, 'input-protocol.json': protocol, 'preparation.json': preparation})


def _validate_bundle(base, manifest, protocol, companions):
    verify(manifest); verify(protocol)
    require(set(companions) == set(INPUT_COMPANIONS), 'S1.1 exact companion inventory required')
    for value in companions.values(): verify(value)
    selection = companions['selection-plan.json']
    require(selection.get('schema') == SCHEMAS['selection-plan.json'], 'S1.1 selection schema differs')
    bundles = selection.get('original_input_bundles')
    require(type(bundles) is dict and set(bundles) == {'original128', 'system256'},
            'two complete original input bundles required')
    expected = build_inputs(bundles['original128'], bundles['system256'], selection['predecessor'],
                            expected_predecessor_sha256=selection['expected_predecessor_sha256'])
    require(manifest == expected['manifest.json'] and protocol == expected['input-protocol.json'],
            'S1.1 rows/protocol changed from pinned originals or completion receipt')
    for name in INPUT_COMPANIONS:
        require(companions[name] == expected[name], 'S1.1 source/token/selection evidence changed: ' + name)
    if base is not None:
        base.validate_manifest(manifest)
        for row in manifest['warmup'] + manifest['cohort']: base.document_specs(row)
    return True


def validate_inputs(base, manifest, protocol, *, input_dir):
    return _validate_bundle(base, manifest, protocol, {n: read(Path(input_dir)/n) for n in INPUT_COMPANIONS})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous128-input-dir', 'system256-input-dir', 'predecessor-receipt', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-predecessor-sha256', required=True)
    args = parser.parse_args(argv)
    require(not args.output_dir.exists(), 'never overwrite frozen or prior inputs')
    result = build_inputs({n: read(args.previous128_input_dir/n) for n in ORIGINAL_FILES},
        {n: read(args.system256_input_dir/n) for n in ORIGINAL_FILES}, read(args.predecessor_receipt),
        expected_predecessor_sha256=args.expected_predecessor_sha256)
    args.output_dir.mkdir(parents=True)
    for name, value in result.items(): publish(args.output_dir/name, value)
    print(json.dumps(result['preparation.json'], ensure_ascii=False))


if __name__ == '__main__': main()
