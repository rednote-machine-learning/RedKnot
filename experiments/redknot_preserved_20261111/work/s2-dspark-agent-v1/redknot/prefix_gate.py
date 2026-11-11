"""Independent CPU audit of native Prefix(copy) evidence.

This reads actual producer receipts; it never creates a GPU hit or substitutes
text similarity for copied KV/state and skipped native prefill rows.
"""
from kv_cache_engram.core import digest, token_digest, require, validate_digest
from prefix_decode_contract import validate_fresh_prefix_resume, validate_dflash_swa_coverage, validate_stable_graph_storage


def sealed(value):
    require(type(value) is dict and value.get('sha256') == digest({k:v for k,v in value.items() if k != 'sha256'}),
            'prefix evidence seal differs')
    return value


def integer(value, *, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), 'invalid nonnegative prefix counter')
    return value


def _fingerprints(value):
    require(type(value) is dict and set(value) == {'kv','state'}, 'both actual raw payload fingerprints required')
    for item in value.values():
        if type(item) is str:
            validate_digest(item, 'raw prefix payload SHA256')
        else:
            require(type(item) is dict and type(item.get('shape')) is list and item['shape']
                    and all(type(x) is int and x > 0 for x in item['shape'])
                    and type(item.get('dtype')) is str and item['dtype'], 'raw payload fingerprint geometry missing')
            validate_digest(item.get('sha256'), 'raw prefix payload SHA256')
    return value


def _rows(input_ids, reports, cached):
    require(type(reports) is list and len(reports) == 8, 'all eight native scheduled-row reports required')
    total = len(input_ids)
    request_id = None
    for rank, report in enumerate(reports):
        sealed(report)
        require(report.get('schema') == 's2-native-scheduled-rows-v1' and report.get('rank') == rank
                and report.get('input_tokens') == total and type(report.get('request_id')) is str,
                'native scheduled-row ownership differs')
        if request_id is None: request_id = report['request_id']
        require(report['request_id'] == request_id, 'rank request order/identity differs')
        integer(report['decode_steps'])
        initial = cached if rank == 0 else 0
        require(report['initial_cached_tokens'] == initial, 'actual native initial cached count differs')
        ranges = report['prefill_ranges']
        require(type(ranges) is list, 'actual native query ranges missing')
        if rank:
            require(ranges == [] and report['scheduled_prefill_rows'] == 0
                    and report['first_prefill_q_start'] is None, 'inactive rank reported real query rows')
            continue
        cursor = cached
        for item in ranges:
            a,b,n = item['q_start'],item['q_end'],item['rows']
            integer(a);integer(b, positive=True);integer(n, positive=True)
            require(a == cursor and b > a and b <= total and n == b-a,
                    'native prefill ranges overlap, skip, or exceed the prompt')
            cursor = b
        require(ranges and cursor == total and report['first_prefill_q_start'] == cached
                and report['scheduled_prefill_rows'] == total-cached,
                'actual native computed rows do not prove the claimed prefix reuse')
    return request_id


def audit_full_request(input_ids, scheduled_rows_reports):
    """Fresh Full/S2 admission has zero native prefix reuse on every rank."""
    request_id = _rows(input_ids, scheduled_rows_reports, 0)
    body = dict(schema='s2-full-no-prefix-row-audit-v1', request_id=request_id,
        input_ids_sha256=token_digest(input_ids), input_tokens=len(input_ids), reused_prefix_tokens=0,
        scheduled_rows_reports_sha256=[r['sha256'] for r in scheduled_rows_reports])
    return dict(body, sha256=digest(body))


def _capture(snapshot, input_ids, request_id, require_payload_fingerprints, contract_sha):
    n = integer(snapshot['prefix_tokens'], positive=True)
    require(n % 128 == 0 and n <= len(input_ids)-2 and snapshot.get('is_prompt') is True
            and snapshot.get('capture_request_id') == request_id
            and snapshot.get('prefix_ids_sha256') == token_digest(input_ids[:n])
            and snapshot.get('metadata_seqlen_k') == n and snapshot.get('committed_num_cached_tokens') == n,
            'snapshot is not the actual exact committed prompt checkpoint')
    integer(snapshot['metadata_seqlen_q'], positive=True)
    require(snapshot['metadata_seqlen_q'] <= n and snapshot.get('raw_state_complete') is True
            and snapshot.get('raw_kv_and_state_same_checkpoint') is True and snapshot.get('copy_complete') is True,
            'KV and complete state were not copied at the same checkpoint')
    require(type(snapshot.get('source_block_ids')) is list and snapshot['source_block_ids']
            and len(set(snapshot['source_block_ids'])) == len(snapshot['source_block_ids'])
            and all(type(x) is int and x > 0 for x in snapshot['source_block_ids']), 'source native KV ownership missing')
    integer(snapshot['source_state_id'], positive=True)
    integer(snapshot['swa_valid_start'])
    require(snapshot['swa_valid_start'] <= n, 'snapshot SWA valid boundary exceeds checkpoint')
    integer(snapshot['kv_bytes'], positive=True);integer(snapshot['state_bytes'], positive=True)
    committed = snapshot.get('committed_sequence')
    require(snapshot.get('decode_storage_contract_sha256') == contract_sha
        and snapshot.get('prefix_only_intermediate_checkpoint') is True
        and snapshot.get('old_proposal_pool_copied') is False
        and type(committed) is dict and committed.get('request_id') == request_id
        and committed.get('num_cached_tokens') == n and committed.get('output_tokens') == 0
        and snapshot.get('source_prompt_tokens') == committed.get('prompt_tokens') == len(input_ids)
        and snapshot.get('compact_keep_tokens') == 140
        and committed.get('mtp_has_draft_info') is False and committed.get('mtp_draft_count') == 0,
        'D-Spark snapshot must precede output/proposal production and bind the complete state layout')
    if require_payload_fingerprints: _fingerprints(snapshot.get('raw_fingerprints'))


def _decode_contract(report):
    contract = sealed(report.get('decode_storage_contract'))
    require(contract.get('schema') == 's2-prefix-dspark-storage-contract-v1'
        and contract.get('structural_only') is True and contract.get('numerical_qualification') is False
        and contract.get('old_proposal_pool_copied') is False
        and report.get('old_proposal_pool_copied') is False and report.get('prefix_checkpoint_only') is True
        and report.get('proposal_policy') == 'fresh_native_slot_then_native_suffix_and_decode',
        'fresh native D-Spark proposal policy and raw state layout required')
    coverage = validate_dflash_swa_coverage(contract['raw_states'], contract['dflash_swa_views'],
        contract['dflash_swa_coverage']['dflash_layer_ids'], window_tokens=128, backtrack_tokens=6)
    require(coverage == contract['dflash_swa_coverage'], 'DFlash SWA is not covered by the copied raw state')
    stable = validate_stable_graph_storage(contract['graph_storage'], contract['graph_storage'])
    require(report.get('graph_storage_stability') == stable, 'native captured storage was changed')
    return contract


def audit_prefix_request(input_ids, prefix_report, scheduled_rows_reports, *, require_payload_fingerprints=False,
                         source_snapshots=None):
    report = sealed(prefix_report)
    contract = _decode_contract(report)
    require(report.get('schema') == 's2-prefix-copy-request-v1' and report.get('method') == 'Prefix(copy)'
            and report.get('input_tokens') == len(input_ids) and report.get('input_ids_sha256') == token_digest(input_ids)
            and type(report.get('cache_hit')) is bool, 'prefix report input/method differs')
    reused = integer(report['reused_prefix_tokens'])
    require(report['request_id'] == _rows(input_ids, scheduled_rows_reports, reused), 'prefix and native request differ')
    cap = integer(report['bank_budget_bytes'], positive=True)
    require(integer(report['bank_bytes_before']) <= cap and integer(report['bank_bytes_after']) <= cap,
            'prefix bank exceeds declared actual cap')
    require(type(report.get('evictions')) in (list,int), 'cache eviction accounting missing')
    require(type(report.get('captures')) is list, 'actual capture accounting missing')
    for capture in report['captures']:
        if 'skipped' in capture:
            n = integer(capture.get('prefix_tokens'), positive=True)
            require(capture['skipped'] == 'snapshot_exceeds_budget' and n % 128 == 0 and n <= len(input_ids)
                    and n <= len(input_ids)-2 and integer(capture.get('required_bytes'), positive=True) > cap,
                    'unexplained skipped prefix capture')
            continue
        _capture(capture, input_ids, report['request_id'], require_payload_fingerprints, contract['sha256'])
    restore = report.get('restore')
    if not report['cache_hit']:
        require(reused == 0 and restore is None and report.get('matched_prefix_sha256') is None,
                'cache miss claims restored payload or skipped rows')
    else:
        require(reused > 0 and reused % 128 == 0 and reused <= len(input_ids)-2 and type(restore) is dict,
                'prefix hit lacks a supported exact checkpoint')
        require(restore.get('prefix_tokens') == reused == restore.get('restored_num_cached_tokens')
                == restore.get('capture_metadata_seqlen_k') == restore.get('capture_committed_num_cached_tokens')
                and restore.get('prefix_ids_sha256') == report.get('matched_prefix_sha256') == token_digest(input_ids[:reused]),
                'restored state/input boundary differs from actual matched prefix')
        require(type(restore.get('snapshot_id')) is str and restore['snapshot_id']
                and type(restore.get('capture_request_id')) is str and restore['capture_request_id']
                and restore.get('copy_complete') is True and restore.get('raw_state_complete') is True
                and restore.get('raw_kv_and_state_same_checkpoint') is True, 'actual raw snapshot restoration missing')
        for key in ('source_block_ids','destination_block_ids'):
            ids = restore.get(key)
            require(type(ids) is list and ids and len(set(ids)) == len(ids)
                    and all(type(x) is int and x > 0 for x in ids), 'native restored block ownership differs')
        require(len(restore['source_block_ids']) == len(restore['destination_block_ids']), 'KV copy block count differs')
        for key in ('source_state_id','destination_state_id','kv_bytes','state_bytes'): integer(restore[key], positive=True)
        require(not require_payload_fingerprints or source_snapshots is not None,
                'qualification hit must bind the previously observed capture')
        if source_snapshots is not None:
            require(restore['snapshot_id'] in source_snapshots, 'restored snapshot was not previously captured')
            entry = source_snapshots[restore['snapshot_id']];capture = entry['receipt']
            _capture(capture, entry['input_ids'], restore['capture_request_id'], require_payload_fingerprints, contract['sha256'])
            for key in ('snapshot_id','prefix_tokens','prefix_ids_sha256','capture_request_id',
                        'source_block_ids','source_state_id','kv_bytes','state_bytes'):
                require(capture.get(key) == restore.get(key), 'restore detached from source capture: '+key)
            require(capture['metadata_seqlen_k'] == restore['capture_metadata_seqlen_k']
                    and capture['committed_num_cached_tokens'] == restore['capture_committed_num_cached_tokens'],
                    'restored boundary detached from prior committed checkpoint')
            if require_payload_fingerprints:
                require(capture['raw_fingerprints'] == restore['source_fingerprints'],
                        'source snapshot payload changed after its original capture')
            checkpoint = dict(is_prompt=True, is_placeholder=False,
                seqlen_q=capture['metadata_seqlen_q'], seqlen_k=reused,
                committed_num_cached_tokens=reused, swa_valid_start=capture['swa_valid_start'],
                source_prompt_tokens=capture['source_prompt_tokens'], keep_tokens=140)
            admission = validate_fresh_prefix_resume(input_ids, input_ids[:reused], checkpoint,
                restore['fresh_sequence_before'], restore['fresh_sequence_after'])
            require(restore.get('fresh_resume_admission') == admission,
                    'fresh native proposal/rollback state differs on restoration')
        require(restore.get('decode_storage_contract_sha256') == contract['sha256']
            and restore.get('graph_storage_stability') == report['graph_storage_stability'],
            'restoration detached from actual native D-Spark storage')
        if require_payload_fingerprints:
            _fingerprints(restore.get('source_fingerprints'));_fingerprints(restore.get('restored_fingerprints'))
            require(restore['source_fingerprints'] == restore['restored_fingerprints'], 'actual restored GPU bytes differ')
    body = dict(schema='s2-prefix-copy-independent-request-audit-v1', request_id=report['request_id'],
        input_ids_sha256=token_digest(input_ids), prefix_report_sha256=report['sha256'],
        scheduled_rows_reports_sha256=[r['sha256'] for r in scheduled_rows_reports],
        reused_prefix_tokens=reused, raw_payload_fingerprints_verified=require_payload_fingerprints,
        source_capture_binding_verified=bool(report['cache_hit'] and source_snapshots is not None),
        decode_storage_contract_sha256=contract['sha256'], native_dflash_swa_structurally_covered=True,
        fresh_resume_independently_verified=bool(report['cache_hit'] and source_snapshots is not None),
        per_request_numerical_equivalence_claim=False)
    return dict(body, sha256=digest(body))


def _generation(record):
    ids = record.get('output_ids')
    require(type(ids) is list and 0 < len(ids) <= 32 and all(type(i) is int and i >= 0 for i in ids)
            and type(record.get('text')) is str, 'actual bounded generation missing')
    reason = record.get('finish_reason')
    require((reason == 'stop' and ids[-1] == 1 and ids.count(1) == 1)
            or (reason == 'length' and len(ids) == 32 and 1 not in ids), 'bounded EOS/length closure differs')
    return ids,record['text'],reason


def compare_full_prefix(input_ids, full_record, prefix_record, prefix_report,
                        scheduled_rows_reports, full_scheduled_rows_reports, *, source_snapshots=None):
    full = audit_full_request(input_ids, full_scheduled_rows_reports)
    candidate = audit_prefix_request(input_ids, prefix_report, scheduled_rows_reports, require_payload_fingerprints=True,
                                     source_snapshots=source_snapshots)
    for record, audit in ((full_record,full),(prefix_record,candidate)):
        require(record.get('request_id') == audit['request_id'] and record.get('input_ids_sha256') == token_digest(input_ids),
                'numerical generation detached from actual request')
    require(_generation(full_record) == _generation(prefix_record), 'Prefix(copy) generation differs from fresh Full')
    body = dict(schema='s2-prefix-copy-full-numerical-gate-v1', passed=True,
        input_ids_sha256=token_digest(input_ids), full=full, prefix=candidate,
        output_ids_sha256=token_digest(full_record['output_ids']), bounded_token_budget=32,
        complete_agent_quality_claim=False, universal_future_prompt_equivalence_claim=False)
    return dict(body,sha256=digest(body))
