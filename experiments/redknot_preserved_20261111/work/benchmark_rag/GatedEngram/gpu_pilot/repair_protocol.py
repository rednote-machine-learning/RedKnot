"""Frozen CPU contracts for an explicitly unsafe, controlled SWA-KV diagnostic.

This experiment never grants runtime admission. All native Q/KV projections,
attention, hidden updates and FFN execute. Only prepared current-chunk SWA KV
rows are replaced, with A-derived oracle references restored in the C arm.
The four inputs were already exposed by v2; they are not an independent holdout.
"""
from __future__ import annotations

import math

from gpu_pilot import pilot_protocol as pp
from kv_cache_engram.core import digest, integer, require, validate_digest

PROTOCOL_SCHEMA = 'kv-cache-engram-oracle-repair-protocol-v1'
PLAN_SCHEMA = 'kv-cache-engram-oracle-plan-v1'
ARM_SCHEMA = 'kv-cache-engram-oracle-arm-v1'
PHASE_SCHEMA = 'kv-cache-engram-oracle-repair-phase-v1'
TRIPLE_SCHEMA = 'kv-cache-engram-oracle-repair-triple-v1'
RUN_SCHEMA = 'kv-cache-engram-oracle-repair-run-v1'
READOUT_SCHEMA = 'kv-cache-engram-oracle-repair-readout-v1'
BACKEND = 'deepinfer-sm90-native-bfloat16-kv-only-v1'
LAYERS = (6, 7, 9, 10, 11, 12, 13, 16, 17, 18)
PROTECTED_LAYERS = (0, 1, 2, 3, 4, 5, 15)
ARMS = ('recomputed', 'swa_reuse', 'swa_reuse_repair')
MODES = dict(zip(ARMS, ('capture', 'reuse', 'repair')))
CHUNK, DOC, BLOCK, GUARD, WIDTH = 8192, 32768, 512, 128, 512


def repair_protocol():
    return pp.seal_record(dict(schema=PROTOCOL_SCHEMA,
        arms=list(ARMS), arm_order=list(ARMS), input_indices=[2, 3, 4, 5],
        datasets=list(pp.DATASETS), cases_per_dataset=2,
        context_tokens=8*DOC, document_tokens=DOC, document_count=8,
        native_chunk_tokens=CHUNK, source_block_tokens=BLOCK,
        intervention_layers=list(LAYERS), protected_layers=list(PROTECTED_LAYERS),
        document_head_tail_guard=GUARD, native_chunk_tail_guard=GUARD,
        cross_native_chunk_fragments='native; never replace partial source blocks crossing native chunks',
        intervention='private current-chunk prepared SWA KV [T,1,512] only',
        oracle_selection='A clean prepared full KV versus position-restored offline KV; maximum relative L2 per source block, earliest input start tie',
        oracle_selection_budget='one source block per layer/native chunk with any candidates',
        oracle_reference='immutable CPU copy of A selected post-RoPE/cast KV, integrity checked before C restoration',
        oracle_plan_frozen_before_B=True, same_source_candidates_in_B_and_C=True,
        gold_or_generated_answers_used_for_selection=False,
        all_q_native=True, all_kv_projections_native=True, all_ffn_native=True,
        global_kv_index_native=True, head_reuse=False,
        quality=dict(max_new_tokens=64, natural_eos=True, metrics=['EM', 'F1'],
                     fixed_gold=True, retain_length_and_negative_results=True),
        performance_measured=False, no_ttft_claim=True,
        phase_wall_scope='diagnostic wall cost, including oracle capture/restore; not online TTFT or acceleration',
        previously_exposed_exploratory_cohort=True, independent_holdout_claimed=False,
        calibration_cases_in_quality=0, formal_admission_unchanged=True,
        forced_controlled_diagnostic=True, runtime_admission_created=False,
        accuracy_safety_claimed=False, model_flops_saved_claimed=False))


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'invalid '+name)
    return value


def validate_case(row):
    require(type(row) is dict and row.get('index') in (2, 3, 4, 5)
            and row.get('dataset') == pp.DATASETS[row['index'] % 2], 'wrong exploratory input identity')
    require(type(row.get('sample_id')) is str and bool(row['sample_id'])
            and row.get('case_id', row['sample_id']) == row['sample_id'], 'sample identity missing')
    validate_digest(row.get('input_sha256'), 'input digest')
    integer(row.get('input_tokens'), 'input length', 8*DOC+1)
    require(type(row.get('question')) is str and bool(row['question'])
            and type(row.get('references')) is list and bool(row['references'])
            and all(type(value) is str for value in row['references']), 'fixed question and gold required')
    require(row.get('previously_exposed_exploratory_cohort') is True,
            'previous exposure must be explicitly declared')
    spans = row.get('document_spans')
    require(type(spans) is list and len(spans) == 8, 'eight exact document spans required')
    previous, hashes = None, set()
    for span in spans:
        require(type(span) is dict, 'document span required')
        validate_digest(span.get('document_sha256'), 'document digest')
        integer(span.get('start'), 'document start'); integer(span.get('end'), 'document end', 1)
        require(span['end']-span['start'] == DOC and span.get('tokens') == DOC
                and span['end'] <= row['input_tokens']
                and (previous is None or span['start'] == previous), 'noncontiguous or wrong document geometry')
        require(span['document_sha256'] not in hashes, 'duplicate document')
        hashes.add(span['document_sha256']); previous = span['end']
    return row


def validate_cases(rows):
    require(type(rows) is list and len(rows) == 4 and [r.get('index') for r in rows] == [2, 3, 4, 5],
            'exact four exposed exploratory cases required; no old calibration cases')
    for row in rows:
        validate_case(row)
    require(len({r['input_sha256'] for r in rows}) == 4 and len({r['sample_id'] for r in rows}) == 4,
            'duplicate exploratory requests')
    return rows


def document_identity(row):
    return tuple(dict(start=s['start'], end=s['end'], key=s['document_sha256']) for s in row['document_spans'])


def candidate_spans(row, start, end):
    """Independent geometry derivation: no controller/model imports."""
    result = []
    for document in row['document_spans']:
        for block_start in range(0, DOC, BLOCK):
            block_end = block_start+BLOCK
            if document['start']+block_start < start or document['start']+block_end > end:
                continue
            lo = max(block_start, GUARD)
            hi = min(block_end, DOC-GUARD, end-GUARD-document['start'])
            if lo >= hi:
                continue
            result.append(dict(document_sha256=document['document_sha256'],
                document_local_start=lo, document_local_end=hi,
                input_start=document['start']+lo, input_end=document['start']+hi,
                block_start=block_start, block_end=block_end))
    return result


def _native_flags(value):
    require(value.get('native_execution') is True and value.get('operator_backend') == BACKEND
            and value.get('formal_admission_unchanged') is True,
            'native diagnostic provenance/formal admission binding missing')


def validate_plan(plan, row):
    pp._sealed(plan); validate_case(row); _native_flags(plan)
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('complete') is True
            and plan.get('input_sha256') == row['input_sha256'] and plan.get('input_tokens') == row['input_tokens']
            and plan.get('documents_sha256') == digest(document_identity(row))
            and plan.get('layers') == list(LAYERS)
            and plan.get('norm_floor') == 1e-8
            and plan.get('coverage') == {str(layer): row['input_tokens'] for layer in LAYERS},
            'oracle plan input/layer/coverage mismatch')
    require(type(plan.get('capture_request_id')) is str and bool(plan['capture_request_id'])
            and plan.get('capture_phase') == f"case{row['index']}.recomputed", 'oracle capture phase missing')
    records = plan.get('records')
    require(type(records) is list, 'oracle plan records required')
    expected = {(layer, start, min(start+CHUNK, row['input_tokens']))
                for layer in LAYERS for start in range(0, row['input_tokens'], CHUNK)}
    observed, result = set(), {}
    for record in records:
        key = record.get('layer'), record.get('start'), record.get('end')
        require(all(type(value) is int for value in key) and key in expected and key not in observed,
                'foreign/duplicate/malformed oracle chunk')
        observed.add(key)
        spans = candidate_spans(row, key[1], key[2])
        require(record.get('source_spans') == spans and record.get('source_spans_sha256') == digest(spans),
                'source candidates differ from frozen document/chunk guards')
        count = sum(s['input_end']-s['input_start'] for s in spans)
        require(record.get('source_rows') == count, 'source row accounting mismatch')
        scores = record.get('source_block_scores')
        require(type(scores) is list and len(scores) == len(spans), 'complete source scores required')
        for score, span in zip(scores, spans):
            require(type(score) is dict and score.get('input_start') == span['input_start']
                    and score.get('input_end') == span['input_end'], 'source score position mismatch')
            number(score.get('relative_l2_max'), 'A-only source KV error')
        if spans:
            selected = spans[min(range(len(spans)), key=lambda i: (-scores[i]['relative_l2_max'], spans[i]['input_start']))]
            require(record.get('selected_span') == selected, 'oracle source choice differs from frozen A-error rule')
            validate_digest(record.get('oracle_reference_sha256'), 'oracle reference payload')
            rows = selected['input_end']-selected['input_start']
            require(record.get('oracle_reference_shape') == [rows, 1, WIDTH]
                    and record.get('oracle_reference_dtype') == 'torch.bfloat16'
                    and record.get('oracle_reference_bytes') == rows*WIDTH*2,
                    'oracle reference native shape/dtype/bytes mismatch')
        else:
            require(all(record.get(field) is None for field in ('selected_span', 'oracle_reference_sha256',
                        'oracle_reference_shape', 'oracle_reference_dtype'))
                    and record.get('oracle_reference_bytes') == 0, 'empty chunk has phantom oracle reference')
        result[key] = record
    require(observed == expected, 'oracle plan lacks full ten-layer/native-chunk coverage')
    total_bytes = sum(record['oracle_reference_bytes'] for record in records)
    require(type(plan.get('cpu_reference_bytes')) is int and plan['cpu_reference_bytes'] == total_bytes
            and plan.get('max_cpu_reference_bytes') == 256*1024**2 and total_bytes <= 256*1024**2,
            'oracle reference aggregate bytes/budget mismatch')
    return result


def validate_intervention(report, plan, row, *, arm, request_id, phase):
    pp._sealed(report); records = validate_plan(plan, row); _native_flags(report)
    require(arm in MODES and report.get('schema') == ARM_SCHEMA and report.get('complete') is True
            and report.get('mode') == MODES[arm] and report.get('request_id') == request_id
            and report.get('phase') == phase and report.get('input_sha256') == row['input_sha256']
            and report.get('input_tokens') == row['input_tokens']
            and report.get('documents_sha256') == plan['documents_sha256']
            and report.get('plan_sha256') == plan['sha256'] and report.get('layers') == list(LAYERS)
            and report.get('coverage') == plan['coverage'], 'arm intervention/plan/request binding mismatch')
    for field in ('oracle_diagnostic', 'no_ttft_claim', 'all_q_native', 'all_kv_projections_native',
                  'all_ffn_native', 'global_kv_index_native', 'formal_admission_unchanged'):
        require(report.get(field) is True, 'missing intervention safeguard: '+field)
    require(report.get('forced_unqualified_reuse') is (arm != 'recomputed')
            and report.get('propagates_to_later_layers') is (arm != 'recomputed')
            and report.get('head_reuse') is False
            and report.get('native_chunk_tail_guard') == GUARD
            and report.get('document_head_tail_guard') == GUARD
            and report.get('protected_layers') == list(PROTECTED_LAYERS), 'intervention semantics changed')
    if arm == 'recomputed':
        require(request_id == plan['capture_request_id'] and phase == plan['capture_phase'], 'A capture identity mismatch')
    names = ('source_rows', 'injected_rows', 'oracle_restored_rows', 'reused_rows',
             'native_q_rows', 'native_kv_projection_rows')
    expected_counts = {str(layer): {name: 0 for name in names} for layer in LAYERS}
    entries = report.get('records')
    require(type(entries) is list, 'arm chunk records missing')
    seen = set()
    for item in entries:
        key = item.get('layer'), item.get('start'), item.get('end')
        require(all(type(value) is int for value in key) and key in records and key not in seen,
                'arm chunk missing/duplicated/foreign')
        seen.add(key); source = records[key]
        require(item.get('source_spans_sha256') == source['source_spans_sha256']
                and item.get('selected_span') == source['selected_span']
                and item.get('oracle_reference_sha256') == source['oracle_reference_sha256'],
                'B/C candidates or oracle reference differ from frozen A plan')
        count = source['source_rows']; selected = source['selected_span']
        restored = selected['input_end']-selected['input_start'] if selected and arm == 'swa_reuse_repair' else 0
        injected = count if arm != 'recomputed' else 0
        values = dict(source_rows=count, injected_rows=injected, oracle_restored_rows=restored,
                      reused_rows=injected-restored)
        require(all(type(item.get(name)) is int and item[name] == value for name, value in values.items()),
                'arm injection/restoration row accounting differs from plan')
        values.update(native_q_rows=key[2]-key[1], native_kv_projection_rows=key[2]-key[1])
        for name, value in values.items():
            expected_counts[str(key[0])][name] += value
    require(seen == set(records) and report.get('counts') == expected_counts,
            'arm counts lack complete native Q/KV or exact source row coverage')
    total = {name: sum(c[name] for c in expected_counts.values()) for name in names}
    total.update(ten_layer_prompt_row_denominator=len(LAYERS)*row['input_tokens'],
                 fourteen_layer_prompt_row_denominator=len(pp.LAYERS)*row['input_tokens'],
                 all_native_kv_projections_executed=True, not_saved_compute_or_flops=True)
    total['cache_rows_remaining_fraction_of_ten_layer_prompt_rows'] = total['reused_rows']/total['ten_layer_prompt_row_denominator']
    return total
