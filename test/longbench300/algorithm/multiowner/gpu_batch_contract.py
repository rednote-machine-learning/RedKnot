"""CPU reader of actual per-request timing and positive-row concurrency evidence."""
import hashlib
import json
import math
from pathlib import Path

METHODS = ('Full', 'S1.2', 'S1.3')
STRATEGIES = ('mean', 'max', 'softmax_mean')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, sha256=digest(value))


def read(path):
    value = json.loads(Path(path).read_bytes())
    require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}), 'Invalid artifact seal')
    return value


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), file_sha256=file_sha(path), seal_sha256=read(path)['sha256'])


def read_ref(reference):
    require(file_sha(reference['path']) == reference['file_sha256'], 'Referenced bytes changed')
    value = read(reference['path'])
    require(value['sha256'] == reference['seal_sha256'], 'Referenced seal differs')
    return value


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value = seal(value)
    with path.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2); output.write('\n')
    return value


def validate_trace(trace, request_ids, batch_size):
    require(trace['schema'] == 's13-multiowner-world-trace-v1' and trace['steps'], 'Actual multiowner trace missing')
    peak = 0; overlap = []; step_ids = []; seen = set()
    for step in trace['steps']:
        require(step['kind'] in ('prefill', 'decode'), 'Unknown actual model step kind')
        require([r['rank'] for r in step['rank_rows']] == list(range(8)), 'Every actual world step needs eight rank observations')
        step_ids.append(step['step_id']); active = set()
        for rank in step['rank_rows']:
            require(len(rank['requests']) <= 1, 'Multiowner DP keeps native local request count at most one')
            for row in rank['requests']:
                require(row['request_id'] in request_ids and type(row['scheduled_token_rows']) is int
                    and row['scheduled_token_rows'] > 0, 'Only positive executed rows may count as active requests')
                active.add(row['request_id']); seen.add(row['request_id'])
        if step['kind'] == 'prefill':
            peak = max(peak, len(active))
            if len(active) == batch_size:
                overlap.append(step['step_id'])
    require(len(step_ids) == len(set(step_ids)) and step_ids == sorted(step_ids), 'Repeated or unordered native world steps')
    require(seen == request_ids, 'Missing actual request execution in world trace')
    require(overlap, 'No actual same-step full-batch prefill execution; queued requests are not GPU concurrency')
    return dict(peak_same_step_prefill_requests=peak, full_batch_prefill_step_ids=overlap,
        all_eight_ranks_observed=True, queued_requests_not_counted=True)


def validate_clock(proof):
    require(proof['schema'] == 's13-multiowner-clock-evidence-v1' and proof['passed'] is True
        and proof['timestamps_are_original_owner_samples'] is True and proof['timestamps_rewritten'] is False,
        'Actual unmodified owner clock provenance missing')
    ranks = proof['samples']
    require([r['rank'] for r in ranks] == list(range(8)), 'All eight clock observations required')
    require(len({r['hostname'] for r in ranks}) == len({r['boot_id'] for r in ranks}) == 1,
        'Cross-host owner times cannot be pooled without a separate calibrated protocol')
    for r in ranks:
        require(r['monotonic'] is True
            and 'CLOCK_MONOTONIC' in r['implementation'] and r['resolution'] > 0,
            'Only verified shared-kernel monotonic clock is admitted')
        require(all(type(t) is int for t in (proof['rank0_send_ns'], r['sample_ns'], proof['rank0_receive_ns']))
            and proof['rank0_send_ns'] <= r['sample_ns'] <= proof['rank0_receive_ns'],
            'Owner monotonic sample is outside rank0 measured probe bounds')
    return True


def validate_batch(batch, rows, *, method, phase, batch_size, b1=None, quality=None, reader=read_ref):
    require(batch['schema'] == 's13-native-multiowner-batch-v1' and batch['status'] == 'complete', 'Native batch incomplete')
    require(batch['method'] == method in METHODS and batch['phase'] == phase in ('quality', 'timing')
        and batch['request_batch_size'] == batch_size == len(rows), 'Native batch mode/geometry differs')
    require(batch['input_indices'] == [r['index'] for r in rows], 'Batch changed original input order')
    require(batch['clock_domain'] == 'same_host_perf_counter_ns', 'Unverified cross-rank clock domain')
    validate_clock(reader(batch['clock_evidence_ref']))
    requests = batch['requests']
    require([r['index'] for r in requests] == batch['input_indices'], 'Missing, reordered or repeated request results')
    ids = {r['request_id'] for r in requests}
    require(len(ids) == batch_size and len({r['owner_rank'] for r in requests}) == batch_size
        and all(type(r['owner_rank']) is int and 0 <= r['owner_rank'] < 8 for r in requests), 'Each request needs a distinct actual DP owner')
    start, end = batch['batch_started_ns'], batch['batch_finished_ns']
    require(type(start) is int and type(end) is int and 0 <= start < end, 'Invalid batch timing clock')
    metrics = []
    for row, request in zip(rows, requests):
        require(request['input_ids_sha256'] == row['input_ids_sha256'] and request['input_tokens'] == row['input_tokens'], 'Native request input differs')
        received, admitted, prepared, submitted, first, completed = (request[k] for k in
            ('received_ns', 'admitted_ns', 'prepared_ns', 'submitted_ns', 'first_token_ns', 'completed_ns'))
        require(all(type(t) is int for t in (received, admitted, prepared, submitted, first, completed))
            and abs(received - start) <= 256 and received <= admitted <= prepared <= submitted < first <= completed <= end,
            'Invalid original receipt/preparation/submission clock; online preparation must remain timed')
        require(request['output_ids'] and all(type(t) is int and t >= 0 for t in request['output_ids'])
            and type(request['text']) is str and request['finish_reason'] in ('stop', 'length'), 'Actual complete output envelope missing')
        actual = reader(request['phase_ref'])
        record = actual['record']
        for key in ('input_ids_sha256', 'input_tokens', 'output_ids', 'text', 'finish_reason'):
            require(record[key] == request[key], 'Request summary differs from actual phase: ' + key)
        require(actual['benchmark_request'] is True and actual['qualification_only'] is False, 'Qualification output cannot count as benchmark')
        proof = reader(request['qualification_ref'])
        require(proof['status'] == 'passed' and proof['request_id'] == request['request_id']
            and proof['input_ids_sha256'] == row['input_ids_sha256'] and proof['method'] == method,
            'Native request qualification missing or detached')
        if phase == 'quality' and b1 is not None:
            prior = b1[(method, row['index'])]
            require(request['output_ids'] == prior['output_ids'] and request['text'] == prior['text']
                and request['finish_reason'] == prior['finish_reason'],
                'Batched exact-output parity gate failed versus same-method B1; inspect raw evidence before continuing')
        if phase == 'timing':
            expected = quality[(batch_size, method, row['index'])]['output_ids']
            require(request['output_ids'] == expected[:len(request['output_ids'])], 'Timing token prefix differs from same-batch natural answer')
        metrics.append(dict(index=row['index'], request_id=request['request_id'], owner_rank=request['owner_rank'],
            ttft_s=(first - received) / 1e9, engine_ttft_s=(first - submitted) / 1e9,
            request_elapsed_s=(completed - received) / 1e9, engine_request_elapsed_s=(completed - submitted) / 1e9,
            online_prepare_and_admission_s=(submitted - received) / 1e9,
            submission_delay_s=(submitted - start) / 1e9, output_tokens=len(request['output_ids'])))
    native = reader(batch['native_evidence_ref'])
    require(native['status'] == 'passed' and native['batch_id'] == batch['batch_id']
        and set(native['request_ids']) == ids and native['all_request_qualifications_passed'] is True,
        'Actual native multiowner qualification missing')
    require(native['native_dspark_nextn'] == 5 and native['native_idle_bypass_installed'] is False,
        'All methods require native D-Spark NEXTN5 with native scheduling')
    graph = native['graph_replays_by_rank']
    require([g['rank'] for g in graph] == list(range(8)), 'Actual graph counters required from all eight ranks')
    for g in graph:
        require(g['main_captured_sizes'] and g['draft_captured_sizes']
            and type(g['main_replay_delta']) is int and g['main_replay_delta'] >= 0
            and type(g['draft_replay_delta']) is int and g['draft_replay_delta'] >= 0
            and g['native_cache_storage_unchanged'] is True, 'Native main/draft graph or cache-address evidence missing')
    for request in requests:
        if len(request['output_ids']) > 1:
            g = graph[request['owner_rank']]
            require(g['main_replay_delta'] > 0 and g['draft_replay_delta'] > 0, 'Owner did not replay both native decode graphs')
    trace = reader(batch['world_trace_ref']); concurrency = validate_trace(trace, ids, batch_size)
    collective = reader(batch['collective_trace_ref'])
    require(collective['status'] == 'passed' and collective['batch_id'] == batch['batch_id']
        and collective['all_eight_ranks_same_order'] is True and collective['empty_ep_participation_verified'] is True,
        'Actual eight-rank collective agreement missing')
    submitted0 = min(r['submitted_ns'] for r in requests); completed1 = max(r['completed_ns'] for r in requests)
    first1 = max(r['first_token_ns'] for r in requests)
    elapsed = (completed1 - start) / 1e9; horizon = (first1 - start) / 1e9
    engine_elapsed = (completed1 - submitted0) / 1e9
    input_tokens = sum(r['input_tokens'] for r in rows); output_tokens = sum(len(r['output_ids']) for r in requests)
    return dict(per_request=metrics, concurrency=concurrency, e2e_makespan_s=elapsed,
        all_first_tokens_horizon_s=horizon, prompts_per_s=batch_size / elapsed,
        input_tokens_per_s_e2e=input_tokens / elapsed, output_tokens_per_s_e2e=output_tokens / elapsed,
        input_tokens_per_s_to_all_first_tokens=input_tokens / horizon,
        engine_submission_makespan_s=engine_elapsed, engine_submission_prompts_per_s=batch_size / engine_elapsed,
        throughput_denominator='batch admission/receipt to final request completion; includes online preparation, prefill and decode',
        receipt_float_to_ns_max_rounding_tolerance_ns=256,
        first_token_horizon_is_not_pure_prefill_gpu_time=True)
