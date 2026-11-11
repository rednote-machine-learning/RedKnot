"""Independent per-request numerical gate, not a GPU qualification claim.

The service must additionally validate its fresh native/all-rank metadata and
idle/scheduler receipts. This module verifies the complete original-chunk
tensor, source2 and bounded generation comparison for the actual new prompt.
"""
from __future__ import annotations

from kv_cache_engram.core import digest, require, token_digest, validate_digest
from kv_cache_engram.s1_plan import all_chunks
from gpu_pilot.read_tail_link_results import _fingerprint
from .admission import validate_plan, s2_documents

LAZY = {'residual', 'post_layer_mix', 'comb_res_mix', 'pre_layer_mix'}


def verify_dflash_tail_pair(reference, candidate, *, input_ids_sha256, input_tokens):
    tails = [report.get('dflash_tail') for report in (reference, candidate)]
    positions = list(range(max(0, input_tokens-128), input_tokens))
    for index, tail in enumerate(tails):
        require(type(tail) is dict and tail.get('schema') == 's2-dflash-tail-runtime-v1'
            and tail.get('enabled') is True and tail.get('integrated_all_rank_runtime') is True
            and tail.get('input_ids_sha256') == input_ids_sha256
            and tail.get('input_tokens') == input_tokens and tail.get('window_size') == 128
            and tail.get('retained_positions') == positions
            and tail.get('native_main_cache_update_calls') == 1
            and tail.get('selected_positions_unchanged') is True
            and tail.get('sampler_placeholder_features_used') is False
            and tail.get('fingerprint_before_decode') is True
            and tail.get('same_policy_reference_exact') is bool(index),
            'actual S2 DFlash final-window initialization evidence differs')
        targets = tail.get('target_layer_ids')
        require(type(targets) is list and targets and len(set(targets)) == len(targets)
            and all(type(i) is int and 0 <= i < 40 for i in targets)
            and tail.get('coverage_by_target') == {str(k):len(positions) for k in targets}
            and type(tail.get('state_id')) is int and tail['state_id'] > 0,
            'complete real-row DFlash target/state coverage required')
        fingerprints = tail.get('qualification_fingerprints')
        require(type(fingerprints) is dict and set(fingerprints) == {'target_features','valid_main_swa'},
                'actual DFlash input and main-cache fingerprints required')
        _fingerprint(fingerprints['target_features'])
        require(fingerprints['target_features']['shape'][0] == len(positions), 'DFlash tail row axis differs')
        caches = fingerprints['valid_main_swa']
        require(type(caches) is list and caches
            and [item['layer'] for item in caches] == tail.get('native_dflash_layer_ids')
            and len(set(item['layer'] for item in caches)) == len(caches), 'all native DFlash layers required')
        for item in caches:
            _fingerprint(item['value'])
            require(item['value']['shape'][0] == len(positions), 'DFlash valid SWA row axis differs')
    for key in ('target_layer_ids','native_dflash_layer_ids','retained_positions','qualification_fingerprints'):
        require(tails[0][key] == tails[1][key], 'DFlash reference/candidate byte identity failed: '+key)
    return dict(retained_rows=len(positions), target_layer_ids=tails[0]['target_layer_ids'],
        native_dflash_layer_ids=tails[0]['native_dflash_layer_ids'],
        fingerprints_sha256=digest(tails[0]['qualification_fingerprints']),
        actual_prefill_tail_exact=True, complete_future_decode_equivalence_claim=False)


def _generation(record):
    ids = record.get('output_ids')
    require(type(ids) is list and 1 <= len(ids) <= 32
            and all(type(v) is int and v >= 0 for v in ids)
            and record.get('output_ids_sha256') == token_digest(ids)
            and type(record.get('text')) is str, 'complete bounded generation IDs/text required')
    reason = record.get('finish_reason')
    require((reason == 'stop' and ids[-1] == 1 and ids.count(1) == 1)
            or (reason == 'length' and len(ids) == 32 and 1 not in ids), 'bounded natural EOS or full32 length reason required')
    return {key: record[key] for key in ('output_ids', 'output_ids_sha256', 'text', 'finish_reason')}


def verify_sparse_pair(plan, input_ids, reference, candidate, reference_record, candidate_record):
    validate_plan(plan, input_ids=input_ids)
    require(plan['route'] == 's2', 'Full fallback has no sparse qualification')
    chunks = all_chunks(s2_documents(input_ids, plan['effective_documents']), len(input_ids))
    expected = {(c['chunk_start'], c['chunk_end'], layer): c for c in chunks for layer in range(40)}
    for report, mode, group in ((reference, 'reference_sparse', 1), (candidate, 'sensitive', 17)):
        require(report.get('schema') == 's1-controller-v1' and report.get('mode') == mode
                and report.get('system_enabled') is True and report.get('batch_group_chunks') == group
                and report.get('input_ids_sha256') == plan['input_ids_sha256']
                and report.get('full_input_tokens') == len(input_ids)
                and report.get('s2_request_plan_sha256') == plan['sha256']
                and report.get('cache_scope') == plan['scope'], 'same-request numerical scope differs')
        require(report.get('first_three_full') is True and report.get('decode_native') is True
                and report.get('source2_native_cache_preserved') is True
                and report.get('mathematical_policy_changed') is False, 'frozen mathematical policy scope differs')
        fingerprints = report.get('sparse_identity_fingerprints')
        require(type(fingerprints) is list and len(fingerprints) == len(expected)
                and report.get('sparse_identity_check_count') == len(expected)
                and report.get('sparse_identity_fingerprints_sha256') == digest(fingerprints), 'all original-chunk/layer tensors required')
        seen = set()
        for item in fingerprints:
            key = tuple(item.get('key', ()));value = item.get('value', {})
            require(key in expected and key not in seen and set(value) == {'output', 'lazy'}
                    and type(value['lazy']) is dict and set(value['lazy']) == LAZY, 'duplicate/missing output or mHC tensor')
            seen.add(key)
            rows = key[1]-key[0] if key[2] < 3 else expected[key]['selected_rows']
            require(value['output'] is not None, 'selected layer output missing')
            _fingerprint(value['output'])
            require(value['output']['shape'][0] == rows, 'selected output row count differs')
            for field in value['lazy'].values():
                _fingerprint(field)
                require(field is None or len(field['shape']) >= 2 and field['shape'][:2] == [1, rows], 'mHC token axis differs')
        source = report.get('source2_identity_events')
        require(type(source) is list and len(source) == len(chunks), 'source2 fingerprint coverage missing')
        for item, chunk in zip(source, chunks):
            require(item.get('chunk_start') == chunk['chunk_start'] and item.get('chunk_end') == chunk['chunk_end']
                    and item.get('exact') is True and item.get('before') == item.get('after'), 'source2 data changed')
            before = item['before']
            require(before.get('schema') == 's1-source2-cache-fingerprint-v1'
                    and before.get('rows') == chunk['chunk_end']//2, 'source2 prefix length differs')
            for key in ('main_sha256', 'index_k_sha256'):
                validate_digest(before.get(key), key)
    require(reference.get('sparse_identity_exact') is False and candidate.get('sparse_identity_exact') is True
            and candidate.get('system_numerically_qualified') is True, 'candidate must pass actual numerical comparison')
    for key in ('s2_request_id', 's2_request_generation', 'bank_generation', 'offline_bank_scope_sha256'):
        require(reference.get(key) is not None and reference.get(key) == candidate.get(key), 'reference/candidate request or bank differs')
    require(reference.get('phase_generation') != candidate.get('phase_generation'), 'reference phase state reused')
    require(reference['sparse_identity_fingerprints'] == candidate['sparse_identity_fingerprints'], 'output/four-mHC byte parity failed')
    require(_generation(reference_record) == _generation(candidate_record), 'complete bounded generation differs')
    dflash = verify_dflash_tail_pair(reference, candidate,
        input_ids_sha256=plan['input_ids_sha256'], input_tokens=len(input_ids))
    body = dict(schema='s2-request-sparse-comparison-v1', passed=True, request_plan_sha256=plan['sha256'],
                request_id=reference['s2_request_id'], input_ids_sha256=plan['input_ids_sha256'],
                reference_controller_sha256=digest(reference), candidate_controller_sha256=digest(candidate),
                original_chunk_layer_checks=len(expected), source2_chunk_checks=len(chunks),
                generated_tokens=len(reference_record['output_ids']), output_ids_sha256=reference_record['output_ids_sha256'],
                dflash_tail=dflash,
                full_model_equivalence_claim=False, all_rank_service_admission_still_required=True)
    return dict(body, sha256=digest(body))
