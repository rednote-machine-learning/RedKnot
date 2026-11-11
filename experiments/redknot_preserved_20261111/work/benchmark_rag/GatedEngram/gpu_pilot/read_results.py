"""Independent read-only CPU audit of the isolated native Engram pilot.

Only complete held-out quality + three paired C1 trials enter pooled metrics.
No model imports, remote access, GPU allocation, threshold tuning or writes are
performed unless the CLI explicitly receives a new exclusive output path.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_pilot import pilot_protocol as pp
from gpu_pilot import sensitivity_protocol as sp
from kv_cache_engram.core import ValidationError, digest, integer, require, validate_digest
from kv_cache_engram.gate import GatePolicy

SCHEMA = 'kv-cache-engram-native-pilot-readout-v1'
ARMS = ('recomputed', 'gated')


def _read(path, evidence):
    require(path.is_file() and not path.is_symlink(), 'missing or symlink evidence: '+path.name)
    require(path.stat().st_size <= 512*1024**2, 'JSON evidence exceeds512MiB budget')
    data = path.read_bytes()
    value = json.loads(data)
    pp._sealed(value)
    evidence[path.name] = dict(file_sha256=hashlib.sha256(data).hexdigest(), seal_sha256=value['sha256'])
    return value


def _number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'invalid '+name)
    return value


def _scores(text, references):
    def normalize(value):
        return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', re.sub(r'[^\w\s]', '', value.lower())).split())
    words, scores = normalize(text).split(), []
    for reference in references:
        gold = normalize(reference).split()
        common = sum((Counter(words) & Counter(gold)).values())
        scores.append((float(normalize(text) == normalize(reference)),
                       2*common/(len(words)+len(gold)) if words or gold else 1.))
    return dict(answer_em=max(score[0] for score in scores), answer_token_f1=max(score[1] for score in scores))


def _expected(preflight):
    require(preflight.get('protocol') == pp.pilot_protocol(), 'foreign preflight protocol')
    validate_digest(preflight.get('manifest_sha256'), 'manifest seal')
    rows = preflight.get('audit_cases')
    require(type(rows) is list and len(rows) == 6 and [row.get('index') for row in rows] == list(range(6)),
            'preflight requires all six fixed audit cases')
    require([row.get('dataset') for row in rows] == list(pp.DATASETS)*3, 'wrong fixed dataset order')
    pp._cases(rows[:2], 'calibration'); pp._cases(rows[2:], 'evaluation')
    for row in rows:
        require(type(row.get('question')) is str and bool(row['question'])
                and type(row.get('references')) is list and bool(row['references'])
                and all(type(ref) is str for ref in row['references']), 'fixed question/gold required')
        require(row.get('sample_id') == row['case_id'], 'sample ID/case ID mismatch')
        integer(row.get('input_tokens'), 'request length', 262145)
    sensitivity = preflight.get('sensitivity_protocol')
    identity_seal = preflight.get('identity', {}).get('sensitivity_protocol_sha256')
    if sensitivity is not None or identity_seal is not None:
        require(sensitivity == sp.sensitivity_protocol() and identity_seal == sensitivity['sha256'],
                'sensitivity protocol/execution identity mismatch')
    return rows


def _frozen(frozen, preflight, expected, *, protected_layers=()):
    identity = preflight['identity']
    scope = dict(model_sha256=identity['checkpoint_index_sha256'], tokenizer_sha256=identity['tokenizer_json_sha256'],
        execution_sha256=identity['execution_configuration_sha256'], representation='swa-postnorm-prerope-v1',
        storage_dtype='bfloat16-native-canonical', width=512, layers=list(pp.LAYERS))
    pp._scope(scope)
    require(frozen.get('schema') == 'kv-cache-engram-native-pilot-calibration-v1'
            and frozen.get('scope') == scope and frozen.get('oracle') == pp.ORACLE
            and frozen.get('protocol_sha256') == preflight['protocol']['sha256'], 'frozen calibration scope mismatch')
    projection = lambda row: {key: row[key] for key in ('case_id', 'dataset', 'split', 'input_sha256',
                                                       'source_document_ids', 'documents')}
    require(frozen.get('calibration_cases') == [projection(row) for row in expected[:2]], 'calibration input mismatch')
    require(type(frozen.get('calibration_report_sha256')) is list and len(frozen['calibration_report_sha256']) == 2,
            'both calibration evidence seals required')
    for seal in frozen['calibration_report_sha256']:
        validate_digest(seal, 'calibration report seal')
    declared = pp._policy_map(list(pp.candidate_policies()))
    require(set(frozen.get('layers', {})) == {str(layer) for layer in pp.LAYERS}, 'missing frozen layer')
    selected = {}
    for layer in pp.LAYERS:
        entry = frozen['layers'][str(layer)]
        candidates = entry.get('candidates')
        require(type(candidates) is list and len(candidates) == len(declared), 'missing frozen candidates')
        require({candidate['policy_sha256'] for candidate in candidates} == set(declared), 'undeclared frozen grid')
        for candidate in candidates:
            require(candidate['policy'] == asdict(declared[candidate['policy_sha256']]), 'frozen policy changed')
            summary = candidate['summary']
            for key in ('total_rows', 'blocks', 'would_reuse_rows', 'false_accept_blocks', 'bad_reused_rows',
                        'nonfinite_count', 'estimated_online_kv_callback_rows', 'actual_shadow_reused_rows'):
                integer(summary.get(key), key)
            require(summary['total_rows'] == 2*8*32768 and summary['actual_shadow_reused_rows'] == 0
                    and summary['would_reuse_rows'] <= summary['total_rows']
                    and summary['estimated_online_kv_callback_rows'] == summary['total_rows']-summary['would_reuse_rows'],
                    'invalid calibration row accounting')
            eligible = bool(summary['would_reuse_rows'] and not summary['false_accept_blocks']
                            and not summary['bad_reused_rows'] and not summary['nonfinite_count'])
            require(summary.get('eligible') is eligible, 'invalid calibration eligibility')
        eligible = [candidate for candidate in candidates if candidate['summary']['eligible']]
        choice = min(eligible, key=lambda c: (-c['summary']['would_reuse_rows'],
                     c['summary']['estimated_online_kv_callback_rows'], c['policy_sha256'])) if eligible else None
        if layer in protected_layers:
            choice = None
            require(entry.get('sensitivity_protected') is True, 'missing protected-layer marker')
        else:
            require(entry.get('sensitivity_protected') is not True, 'unapproved protected-layer marker')
        require(entry.get('selected') == choice, 'selection differs from fixed calibration-only rule')
        require(entry.get('mode') == ('pilot_gated_kv' if choice else 'full_recompute'), 'mode/selection mismatch')
        if choice:
            selected[layer] = choice['policy_sha256']
    # Source, donor, complete request and cache-document separation is checked
    # before any case becomes eligible, independently of progress.
    for field in ('input_sha256',):
        require(not {r[field] for r in expected[:2]} & {r[field] for r in expected[2:]}, 'request split leakage')
    require(not {s for r in expected[:2] for s in r['source_document_ids']}
            & {s for r in expected[2:] for s in r['source_document_ids']}, 'source/donor split leakage')
    require(not {d['document_sha256'] for r in expected[:2] for d in r['documents']}
            & {d['document_sha256'] for r in expected[2:] for d in r['documents']}, 'cache document split leakage')
    return scope, declared, selected


def _hook(hook, phase, row, scope, selected):
    require(type(hook) is dict and hook.get('schema') == 'kv-cache-engram-native-hook-v1'
            and hook.get('complete') is True and hook.get('native_execution_declared') is True
            and hook.get('backend') == 'deepinfer-sm90-native-bfloat16-kv-only-v1'
            and hook.get('full_q_attention_moe_preserved') is True and hook.get('scope') == scope
            and hook.get('input_sha256') == row['input_sha256'] and hook.get('input_tokens') == row['input_tokens']
            and hook.get('phase') == phase['name'] and hook.get('mode') == phase['mode'], 'invalid native hook binding')
    require(type(hook.get('request_id')) is str and bool(hook['request_id']), 'request ID missing')
    coverage = hook.get('coverage', {})
    counts = hook.get('counts', {})
    require(len(coverage) == len(pp.LAYERS) and {str(k): v for k, v in coverage.items()}
            == {str(layer): row['input_tokens'] for layer in pp.LAYERS}, 'incomplete native full-Q coverage')
    require(len(counts) == len(pp.LAYERS) and {str(k) for k in counts} == {str(layer) for layer in pp.LAYERS},
            'fourteen-layer accounting required')
    totals = dict(actual_reused_kv_rows=0, online_kv_rows=0, total_14_layer_prompt_kv_rows=0,
                  probe_rows=0, fallback_rows=0, native_fused_rows=0, split_q_rows=0)
    reused_by_layer = {}
    for layer in pp.LAYERS:
        count = counts.get(str(layer), counts.get(layer))
        for key in ('q_rows', 'native_fused_rows', 'split_q_rows', 'online_kv_rows', 'reused_kv_rows', 'probe_rows', 'fallback_rows'):
            integer(count.get(key), key)
        require(count['q_rows'] == row['input_tokens']
                and count['native_fused_rows']+count['split_q_rows'] == count['q_rows']
                and count['online_kv_rows']+count['reused_kv_rows'] == count['q_rows']
                and count['probe_rows'] <= count['online_kv_rows'] and count['fallback_rows'] <= count['online_kv_rows'],
                'inconsistent native row counters')
        require(count['reused_kv_rows'] <= 8*32768, 'reuse includes non-document tokens')
        if phase['mode'] != 'gated' or layer not in selected:
            require(count['reused_kv_rows'] == 0, 'unapproved/native layer reports reuse')
        reused_by_layer[layer] = count['reused_kv_rows']
        totals['actual_reused_kv_rows'] += count['reused_kv_rows']
        totals['online_kv_rows'] += count['online_kv_rows']
        totals['total_14_layer_prompt_kv_rows'] += count['q_rows']
        for key in ('probe_rows', 'fallback_rows', 'native_fused_rows', 'split_q_rows'):
            totals[key] += count[key]
    events = hook.get('events')
    require(type(events) is list, 'native event evidence required')
    if phase['mode'] == 'gated':
        actual = {layer: 0 for layer in pp.LAYERS}
        probes = {layer: 0 for layer in pp.LAYERS}
        fallbacks = {layer: 0 for layer in pp.LAYERS}
        seen = set()
        for event in events:
            layer = event.get('layer')
            require(layer in selected and event.get('candidate_id') == selected[layer]
                    and event.get('candidate_sha256') == selected[layer], 'unapproved runtime candidate')
            key = (layer, event.get('document_sha256'), event.get('document_local_start'), event.get('document_local_end'))
            require(key not in seen, 'duplicate runtime gate block')
            seen.add(key)
            require(key[1] in {doc['document_sha256'] for doc in row['documents']}, 'foreign runtime document')
            start, end = key[2:]
            integer(start, 'runtime block start'); integer(end, 'runtime block end', 1)
            require(start % 512 == 0 and end == start+512 and end <= 32768, 'invalid runtime whole block')
            policy = pp._policy_map(list(pp.candidate_policies()))[selected[layer]]
            mandatory = 128 if start in (0, 32256) else 0
            require(event.get('probe_count') == policy.probe_count and event.get('mandatory_count') == mandatory
                    and event.get('candidate_reusable_rows') == end-start-mandatory-policy.probe_count,
                    'runtime probe/guard geometry mismatch')
            relative = _number(event.get('probe_relative_max'), 'runtime relative error')
            absolute = _number(event.get('probe_absolute_max'), 'runtime absolute error')
            require(event.get('accepted') is (relative <= policy.relative_tolerance and absolute <= policy.absolute_tolerance),
                    'runtime acceptance violates frozen thresholds')
            for name in ('phase', 'request_id', 'input_sha256'):
                require(event.get(name) == hook[name], 'runtime gate request mismatch')
            integer(event.get('actual_reused_rows'), 'actual reused rows')
            require(type(event.get('accepted')) is bool
                    and event['actual_reused_rows'] == (event.get('candidate_reusable_rows') if event['accepted'] else 0),
                    'runtime acceptance/reuse mismatch')
            require(event.get('oracle_measured') is False and event.get('current_hybrid_hidden_probes') is True,
                    'runtime probes must not claim full-row oracle')
            actual[layer] += event['actual_reused_rows']
            probes[layer] += event['probe_count']
            fallbacks[layer] += 0 if event['accepted'] else event['candidate_reusable_rows']
        require(actual == reused_by_layer, 'gate events and actual counters disagree')
        for layer in pp.LAYERS:
            count = counts.get(str(layer), counts.get(layer))
            require(count['probe_rows'] == probes[layer] and count['fallback_rows'] == fallbacks[layer],
                    'gate probes/fallback counters disagree with events')
    elif phase['mode'] == 'native':
        require(not events, 'plain native arm unexpectedly gated')
        require(totals['split_q_rows'] == 0 and totals['probe_rows'] == 0 and totals['fallback_rows'] == 0,
                'standard arm must execute native fused computation')
    totals['actual_reuse_fraction_14_layer_prompt_rows'] = totals['actual_reused_kv_rows']/totals['total_14_layer_prompt_kv_rows']
    totals['document_row_denominator_14_layers'] = len(pp.LAYERS)*8*32768
    totals['actual_reuse_fraction_14_layer_document_rows'] = totals['actual_reused_kv_rows']/totals['document_row_denominator_14_layers']
    totals['all_full_recompute'] = totals['actual_reused_kv_rows'] == 0
    totals['not_full_model_compute_fraction'] = True
    return totals


def _no_diagnostics(value):
    require(not any(key in value for key in ('sensitivity', 'diagnostic_phase_wall_s',
                                             'diagnostic_wall_not_online_ttft')),
            'diagnostics forbidden on held-out, timed, or qualification phases')


def _phase(value, name, mode, row, preflight, scope, selected, *, quality):
    pp._sealed(value)
    _no_diagnostics(value)
    require(value.get('schema') == 'kv-cache-engram-native-phase-v1' and value.get('name') == name
            and value.get('mode') == mode and value.get('index') == row['index']
            and value.get('split') == 'heldout' and value.get('dataset') == row['dataset']
            and value.get('manifest_sha256') == preflight['manifest_sha256']
            and value.get('execution_configuration_sha256') == scope['execution_sha256']
            and value.get('input_ids_sha256') == row['input_sha256']
            and value.get('quality_eligible') is quality and value.get('timing_valid') is (not quality),
            'phase identity/quality/timing mismatch')
    require(value.get('online_gate_and_plan_included') is True and value.get('offline_build_included') is False,
            'online timing scope changed')
    rec = value.get('record')
    require(type(rec) is dict and rec.get('success') is True and rec.get('rank') == 0
            and rec.get('phase') == name and rec.get('input_ids_sha256') == row['input_sha256']
            and rec.get('sample_id') == row['sample_id'] and rec.get('dataset') == row['dataset']
            and rec.get('input_tokens') == row['input_tokens']
            and rec.get('request_id') == f"kce:{row['index']}:{name}:{row['sample_id']}"
            and value.get('hook', {}).get('request_id') == rec.get('request_id'),
            'incomplete or wrong request record')
    ids = rec.get('output_ids')
    require(type(ids) is list and bool(ids) and all(type(token) is int and token >= 0 for token in ids), 'bad output IDs')
    require(rec.get('output_ids_sha256') == digest(ids), 'output token seal mismatch')
    require(rec.get('finish_reason') in ('stop', 'length'), 'unknown generation stop reason')
    counts = _hook(value.get('hook'), value, row, scope, selected)
    if quality:
        require(rec.get('natural_EOS') is True and len(ids) <= 64 and type(rec.get('text')) is str
                and rec.get('references') == row['references']
                and rec.get('continuation_is_quality_evidence') is not False
                and rec.get('truncated') is (rec['finish_reason'] == 'length'), 'quality/gold/truncation mismatch')
        require(rec['finish_reason'] != 'length' or len(ids) == 64, 'length cap requires64 tokens')
        scores = _scores(rec['text'], row['references'])
        for key, expected in scores.items():
            _number(rec.get(key), key)
            require(math.isclose(rec[key], expected, rel_tol=0, abs_tol=1e-12), 'score differs from fixed gold')
        return dict(scores, text=rec['text'], references=row['references'], output_ids=ids,
                    output_tokens=len(ids), finish_reason=rec['finish_reason'], truncated=rec['truncated'],
                    row_accounting=counts)
    require(rec.get('natural_EOS') is False and len(ids) == 32 and rec['finish_reason'] == 'length', 'formal32-token trial required')
    ttft = _number(rec.get('online_ttft_s'), 'online TTFT', positive=True)
    elapsed = _number(rec.get('online_elapsed_s'), 'online elapsed', positive=True)
    window = _number(value.get('online_global_window_s'), 'synchronized online window', positive=True)
    prepare = _number(rec.get('online_prepare_to_submit_s'), 'online preparation')
    native_ttft = _number(rec.get('ttft_s'), 'native TTFT', positive=True)
    native_elapsed = _number(rec.get('elapsed_s'), 'native elapsed', positive=True)
    submit = _number(rec.get('submit_s'), 'native submit')
    finish = _number(rec.get('finish_s'), 'native finish', positive=True)
    require(math.isclose(ttft, prepare+native_ttft, rel_tol=1e-10, abs_tol=1e-6)
            and math.isclose(elapsed, prepare+native_elapsed, rel_tol=1e-10, abs_tol=1e-6)
            and math.isclose(native_elapsed, finish-submit, rel_tol=1e-10, abs_tol=1e-6),
            'online timing excludes preparation or disagrees with raw request clock')
    require(ttft <= elapsed and ttft <= window and elapsed <= window+1e-6, 'invalid online timing order')
    return dict(online_ttft_s=ttft, row_accounting=counts)


def _shadow_wrapper(phase, row, scope):
    hook = phase['hook']
    return pp.seal_record(dict(schema='kv-cache-engram-native-shadow-report-v1',
        case_id=row['case_id'], dataset=row['dataset'], split=row['split'], input_sha256=row['input_sha256'],
        request_id=hook['request_id'], phase=hook['phase'], scope=scope, mode='shadow_full', complete=True,
        native_execution_declared=True, oracle=hook['oracle'], events=hook['events'],
        source_phase_sha256=phase['sha256'], source_manifest_sha256=phase['manifest_sha256'],
        source_hook_sha256=digest(hook), provenance='caller-declared native backend; seals are not signatures'))


def _shadow(phase, row, scope, declared, selected):
    wrapper = _shadow_wrapper(phase, row, scope)
    summaries = pp._audit([wrapper], {row['case_id']: row}, scope, {layer: declared for layer in pp.LAYERS})
    values = {str(layer): summaries[(layer, checksum)] for layer, checksum in selected.items()}
    return dict(selected_layer_summaries=values, any_selected_layer=bool(selected),
        source_shadow_report_sha256=wrapper['sha256'],
        selected_candidate_kv_validation_pass=bool(selected) and all(s['eligible'] for s in values.values()),
        false_accept_blocks=sum(s['false_accept_blocks'] for s in values.values()),
        bad_predicted_reused_rows=sum(s['bad_reused_rows'] for s in values.values()),
        clean_full_native_shadow_not_runtime_hybrid_oracle=True, answer_accuracy_guaranteed=False)


def _projection(row):
    return {key: row[key] for key in ('case_id', 'dataset', 'split', 'input_sha256',
                                     'source_document_ids', 'documents')}


def _scope(preflight):
    identity = preflight['identity']
    value = dict(model_sha256=identity['checkpoint_index_sha256'],
        tokenizer_sha256=identity['tokenizer_json_sha256'],
        execution_sha256=identity['execution_configuration_sha256'],
        representation='swa-postnorm-prerope-v1', storage_dtype='bfloat16-native-canonical',
        width=512, layers=list(pp.LAYERS))
    pp._scope(value)
    return value


def _qualification(directory, preflight, row, evidence):
    """The new intervention path must prove every split/gather operation first."""
    raw = _read(directory/'qualification_split_identity.json', evidence)
    closure = _read(directory/'qualification.json', evidence)
    _no_diagnostics(raw)
    scope = _scope(preflight)
    name = 'qualification_split_identity'
    require(raw.get('schema') == 'kv-cache-engram-native-phase-v1' and raw.get('name') == name
            and raw.get('mode') == 'split_identity' and raw.get('split') == 'calibration'
            and raw.get('index') == row['index'] and raw.get('dataset') == row['dataset']
            and raw.get('manifest_sha256') == preflight['manifest_sha256']
            and raw.get('execution_configuration_sha256') == scope['execution_sha256']
            and raw.get('input_ids_sha256') == row['input_sha256']
            and raw.get('timing_valid') is False and raw.get('quality_eligible') is False,
            'qualification phase identity mismatch')
    rec, hook = raw.get('record', {}), raw.get('hook', {})
    require(rec.get('success') is True and rec.get('rank') == 0 and rec.get('phase') == name
            and rec.get('sample_id') == row['sample_id'] and rec.get('dataset') == row['dataset']
            and rec.get('input_tokens') == row['input_tokens']
            and rec.get('input_ids_sha256') == row['input_sha256']
            and rec.get('request_id') == f"kce:{row['index']}:{name}:{row['sample_id']}"
            and hook.get('request_id') == rec.get('request_id'), 'qualification request mismatch')
    ids = rec.get('output_ids')
    require(type(ids) is list and len(ids) == 1 and type(ids[0]) is int and ids[0] >= 0
            and rec.get('output_ids_sha256') == digest(ids) and rec.get('natural_EOS') is False,
            'qualification requires the complete one-token request')
    _hook(hook, raw, row, scope, {})
    require(hook.get('identity_exact') is True and hook.get('numerical_equivalence_claim') is True
            and not hook.get('events'), 'split numerical qualification did not pass')
    qualifications = hook.get('qualification')
    require(type(qualifications) is list and bool(qualifications), 'raw qualification rows missing')
    ends = {layer: 0 for layer in pp.LAYERS}
    error_names = {'q', 'kv_raw', 'kv_canonical', 'gather32', 'gather128', 'post_rope_cast',
                   'post_rope_gather32', 'post_rope_gather128', 'post_rope_gather384', 'post_rope_gather512'}
    for item in qualifications:
        layer = item.get('layer')
        require(type(layer) is int and layer in ends and item.get('start') == ends[layer]
                and item.get('exact') is True, 'duplicate or incomplete numerical qualification')
        end = item.get('end')
        integer(end, 'qualification end', 1)
        require(ends[layer] < end <= row['input_tokens'], 'qualification coverage out of range')
        errors = item.get('errors')
        require(type(errors) is dict and set(errors) == error_names, 'missing split/gather/post-RoPE check')
        require(all(_number(value, 'numerical identity error') == 0 for value in errors.values()),
                'nonzero numerical identity error')
        ends[layer] = end
    require(all(end == row['input_tokens'] for end in ends.values()), 'qualification lacks full fourteen-layer coverage')
    require(closure.get('passed') is True and closure.get('quality_claim') is False
            and closure.get('phase_sha256') == raw['sha256'], 'qualification closure differs from raw phase')
    return dict(passed=True, phase_sha256=raw['sha256'], closure_sha256=closure['sha256'],
                quality_claim=False, all_split_gather_post_rope_checks_exact=True)


def _sensitivity_calibration(directory, preflight, expected, frozen, evidence):
    """Rebuild all permissions from raw CAL2 evidence, never from held-out scores."""
    scope = _scope(preflight)
    reports, sensitivities = [], []
    for row in expected[:2]:
        name = f"case{row['index']}.reference_shadow"
        raw = _read(directory/(name+'.json'), evidence)
        require(raw.get('name') == name and raw.get('index') == row['index']
                and raw.get('mode') == 'shadow_full' and raw.get('quality_eligible') is True
                and raw.get('diagnostic_wall_not_online_ttft') is True,
                'calibration requires untimed full-shadow sensitivity evidence')
        _number(raw.get('diagnostic_phase_wall_s'), 'diagnostic wall time', positive=True)
        require(raw.get('hook', {}).get('request_id') == raw.get('record', {}).get('request_id'),
                'calibration hook/request mismatch')
        _hook(raw.get('hook'), raw, row, scope, {})
        require(raw['hook'].get('oracle') == pp.ORACLE, 'calibration oracle changed')
        wrapped = _shadow_wrapper(raw, row, scope)
        saved = _read(directory/f"case{row['index']}.calibration.json", evidence)
        require(saved.get('reference_sha256') == raw['sha256'] and saved.get('report') == wrapped,
                'saved numerical calibration wrapper differs from raw shadow')
        _number(saved.get('offline_build_s'), 'offline build time')
        validate_digest(saved.get('raw_pair_sha256'), 'paired offline/full KV snapshot seal')
        reports.append(wrapped)
        sensitivity = sp.wrap_phase(raw, _projection(row), manifest_sha256=preflight['manifest_sha256'],
                                    execution_sha256=scope['execution_sha256'])
        note = _read(directory/f"case{row['index']}.sensitivity.json", evidence)
        require(note == sensitivity, 'saved sensitivity report differs from raw phase')
        sensitivities.append(sensitivity)
    numerical = _read(directory/'calibration.numerical.json', evidence)
    require(numerical == pp.fit_shadow(reports, [_projection(row) for row in expected[:2]], scope),
            'numerical calibration differs from independently reconstructed raw evidence')
    admission = _read(directory/'sensitivity.admission.json', evidence)
    sp.validate_frozen_plan(admission, sensitivities, [_projection(row) for row in expected[:2]],
                           numerical_calibration_sha256=numerical['sha256'])
    effective = json.loads(json.dumps({key: value for key, value in numerical.items() if key != 'sha256'}))
    protected = admission['mandatory_full_layers']
    for layer in protected:
        if str(layer) in effective['layers']:
            effective['layers'][str(layer)].update(selected=None, mode='full_recompute', sensitivity_protected=True)
    effective.update(numerical_calibration_sha256=numerical['sha256'],
                     sensitivity_admission_plan_sha256=admission['sha256'])
    require(frozen == pp.seal_record(effective), 'effective calibration is not the protected numerical calibration')
    return admission, _frozen(frozen, preflight, expected, protected_layers=protected)


def _case(directory, paired, row, preflight, frozen, scope, declared, selected, evidence):
    require(paired.get('index') == row['index'] and paired.get('dataset') == row['dataset']
            and paired.get('sample_id') == row['sample_id'] and paired.get('references') == row['references']
            and paired.get('question') == row['question'] and paired.get('input_tokens') == row['input_tokens']
            and paired.get('calibration_sha256') == frozen['sha256']
            and paired.get('all_negative_results_retained') is True
            and paired.get('no_native_Q_attention_MoE_skipped') is True, 'paired case binding mismatch')
    if preflight.get('sensitivity_protocol') is not None:
        require(paired.get('sensitivity_admission_plan_sha256') == frozen.get('sensitivity_admission_plan_sha256'),
                'paired sensitivity admission binding mismatch')
        require(not (directory/f"case{row['index']}.sensitivity.json").exists(),
                'held-out sensitivity report would violate calibration-only selection')
        for arm in ARMS:
            warmup = directory/f"case{row['index']}.{arm}_warmup.json"
            if warmup.exists():
                _no_diagnostics(_read(warmup, evidence))
    quality = {}
    reference = None
    for key, arm, suffix, mode in (('reference', 'recomputed', 'reference_shadow', 'shadow_full'),
                                   ('gated', 'gated', 'gated_quality', 'gated')):
        name = f"case{row['index']}.{suffix}"
        raw = _read(directory/(name+'.json'), evidence)
        require(raw == paired.get(key), 'nested quality differs from raw sealed phase')
        quality[arm] = _phase(raw, name, mode, row, preflight, scope, selected, quality=True)
        if arm == 'recomputed':
            reference = raw
    phases = paired.get('timing_phases')
    require(type(phases) is list and len(phases) == 6, 'six formal phases required')
    schedule = [(repeat, arm) for repeat in range(3) for arm in
                (ARMS if (row['index']+row['index']//2+repeat) % 2 == 0 else ARMS[::-1])]
    timings = {arm: [] for arm in ARMS}
    trials = []
    for nested, (repeat, arm) in zip(phases, schedule):
        pp._sealed(nested)
        require(nested.get('arm') == arm and nested.get('repeat') == repeat, 'fixed formal order changed')
        name = f"case{row['index']}.{arm}_measure{repeat}"
        raw = _read(directory/(name+'.json'), evidence)
        require(nested.get('raw_phase_sha256') == raw['sha256'], 'formal raw seal mismatch')
        require({key: value for key, value in nested.items() if key not in ('sha256', 'arm', 'repeat', 'raw_phase_sha256')}
                == {key: value for key, value in raw.items() if key != 'sha256'}, 'nested formal differs from raw')
        data = _phase(raw, name, 'native' if arm == 'recomputed' else 'gated', row, preflight, scope, selected, quality=False)
        timings[arm].append(data['online_ttft_s'])
        trials.append(dict(arm=arm, repeat=repeat, **data))
    timing = _timing(timings)
    claimed = paired.get('timing_summary', {})
    for arm in ARMS:
        require(claimed.get(arm, {}).get('online_ttft_samples_s') == timings[arm]
                and claimed[arm].get('online_ttft_p50_s') == timing[arm]['online_ttft_p50_s'],
                'claimed timing summary differs from raw samples')
    for key in ('ttft_speedup', 'ttft_reduction_s', 'ttft_reduction_percent'):
        require(claimed.get(key) == timing[key], 'claimed timing comparison differs from raw')
    return dict(index=row['index'], dataset=row['dataset'], sample_id=row['sample_id'], question=row['question'],
        references=row['references'], input_tokens=row['input_tokens'], quality=quality, timing=timing,
        formal_trials=trials, shadow_full_audit=_shadow(reference, row, scope, declared, selected),
        paired_seal_sha256=paired['sha256'], complete=True)


def _timing(values):
    result = {arm: dict(online_ttft_samples_s=list(values[arm]),
                       online_ttft_p50_s=statistics.median(values[arm])) for arm in ARMS}
    a, b = [result[arm]['online_ttft_p50_s'] for arm in ARMS]
    result.update(ttft_speedup=a/b, ttft_reduction_s=a-b, ttft_reduction_percent=100*(1-b/a))
    return result


def _aggregate(cases):
    if not cases:
        return None
    quality = {arm: {metric: sum(case['quality'][arm][metric] for case in cases)/len(cases)
                     for metric in ('answer_em', 'answer_token_f1')} for arm in ARMS}
    losses = {metric: 100*(quality['recomputed'][metric]-quality['gated'][metric])
              for metric in ('answer_em', 'answer_token_f1')}
    trials = [trial for case in cases for trial in case['formal_trials']]
    timing = _timing({arm: [trial['online_ttft_s'] for trial in trials if trial['arm'] == arm] for arm in ARMS})
    rows = [trial['row_accounting'] for trial in trials if trial['arm'] == 'gated']
    total, reused, online = [sum(row[key] for row in rows) for key in
                            ('total_14_layer_prompt_kv_rows', 'actual_reused_kv_rows', 'online_kv_rows')]
    return dict(case_count=len(cases), quality=quality, loss_percentage_points=losses, c1_online_timing=timing,
        baseline_correct_to_gated_wrong=sum(c['quality']['recomputed']['answer_em'] == 1
                                           and c['quality']['gated']['answer_em'] == 0 for c in cases),
        baseline_wrong_to_gated_correct=sum(c['quality']['recomputed']['answer_em'] == 0
                                           and c['quality']['gated']['answer_em'] == 1 for c in cases),
        length_counts={arm: sum(c['quality'][arm]['truncated'] for c in cases) for arm in ARMS},
        gated_formal_row_accounting=dict(actual_reused_kv_rows=reused, online_kv_rows=online,
            total_14_layer_prompt_kv_rows=total, actual_reuse_fraction_14_layer_prompt_rows=reused/total,
            all_full_recompute=reused == 0, not_full_model_compute_fraction=True),
        false_accept_blocks_shadow=sum(c['shadow_full_audit']['false_accept_blocks'] for c in cases),
        bad_predicted_reused_rows_shadow=sum(c['shadow_full_audit']['bad_predicted_reused_rows'] for c in cases))


def read_results(out_dir):
    """Return complete-case-only paired results plus explicit pending/error state."""
    directory = Path(out_dir)
    require(directory.is_dir(), 'pilot output directory missing')
    evidence, failures, errors, completed, pending = {}, [], [], [], []
    if not (directory/'preflight.json').exists():
        return pp.seal_record(dict(schema=SCHEMA, status='awaiting_preflight', completed=0, planned=4,
                                   cases=[], by_dataset={d: dict(completed=0, planned=2) for d in pp.DATASETS}))
    preflight = _read(directory/'preflight.json', evidence)
    expected = _expected(preflight)
    for path in sorted(directory.glob('failure.rank*.json')):
        failure = _read(path, evidence)
        require(failure.get('schema') == 'kv-cache-engram-native-gpu-pilot-run-v1'
                and failure.get('status') == 'failed', 'invalid terminal failure record')
        failures.append(dict(rank=failure.get('rank'), error=failure.get('error'),
                             completed_indices=failure.get('completed_indices'), seal_sha256=failure['sha256']))
    frozen = admission = qualification = None
    sensitivity_enabled = preflight.get('sensitivity_protocol') is not None
    if sensitivity_enabled and (directory/'qualification.json').exists():
        qualification = _qualification(directory, preflight, expected[0], evidence)
    if (directory/'calibration.frozen.json').exists():
        frozen = _read(directory/'calibration.frozen.json', evidence)
        if sensitivity_enabled:
            require(qualification is not None, 'sensitivity calibration requires raw numerical qualification')
            admission, (scope, declared, selected) = _sensitivity_calibration(
                directory, preflight, expected, frozen, evidence)
        else:
            require(not any(key in frozen for key in ('sensitivity_admission_plan_sha256',
                                                       'numerical_calibration_sha256')),
                    'sensitivity calibration requires the explicit protocol')
            scope, declared, selected = _frozen(frozen, preflight, expected)
    for row in expected[2:]:
        name = f"case{row['index']}.paired.json"
        if not (directory/name).exists():
            pending.append(row['index'])
            continue
        try:
            require(frozen is not None, 'paired case exists before frozen calibration')
            paired = _read(directory/name, evidence)
            completed.append(_case(directory, paired, row, preflight, frozen, scope, declared, selected, evidence))
        except (ValidationError, ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(dict(index=row['index'], file=name, error=str(exc)))
    status = 'failed' if failures else 'running'
    terminal_audit = None
    if (directory/'run.json').exists():
        run = _read(directory/'run.json', evidence)
        require(run.get('schema') == 'kv-cache-engram-native-gpu-pilot-run-v1' and run.get('status') == 'complete'
                and run.get('identity') == preflight['identity'] and run.get('manifest_sha256') == preflight['manifest_sha256']
                and frozen is not None and run.get('calibration') == frozen, 'terminal run binding mismatch')
        if sensitivity_enabled:
            require(run.get('sensitivity_admission') == admission, 'terminal sensitivity admission mismatch')
        require([case['sha256'] for case in run.get('cases', [])] == [case['paired_seal_sha256'] for case in completed]
                and len(completed) == 4 and not errors and not failures, 'terminal complete lacks all four audited cases')
        terminal_audit = run.get('heldout_shadow_audit')
        pp._sealed(terminal_audit)
        require(terminal_audit.get('schema') == 'kv-cache-engram-native-pilot-heldout-v1'
                and terminal_audit.get('calibration_sha256') == frozen['sha256']
                and terminal_audit.get('evaluation_report_sha256') ==
                    [case['shadow_full_audit']['source_shadow_report_sha256'] for case in completed],
                'terminal shadow raw evidence mismatch')
        for layer, checksum in selected.items():
            parts = [case['shadow_full_audit']['selected_layer_summaries'][str(layer)] for case in completed]
            combined = {key: sum(part[key] for part in parts) for key in parts[0]
                        if key not in ('eligible', 'would_reuse_row_fraction')}
            combined['would_reuse_row_fraction'] = combined['would_reuse_rows']/combined['total_rows']
            combined['eligible'] = bool(combined['would_reuse_rows'] and not combined['false_accept_blocks']
                                        and not combined['bad_reused_rows'] and not combined['nonfinite_count'])
            require(terminal_audit['layers'][str(layer)]['summary'] == combined,
                    'terminal oracle summary differs from raw shadows')
        require(terminal_audit.get('kv_validation_pass') is
                (bool(selected) and all(terminal_audit['layers'][str(layer)]['summary']['eligible'] for layer in selected)),
                'terminal KV-validation decision mismatch')
        status = 'complete'
    if errors:
        status = 'evidence_invalid'
    return pp.seal_record(dict(schema=SCHEMA, status=status, completed=len(completed), planned=4,
        calibration_cases_excluded_from_quality=2, cases=completed, pending_case_indices=pending,
        global_summary=_aggregate(completed),
        by_dataset={dataset: dict(completed=sum(c['dataset'] == dataset for c in completed), planned=2,
            summary=_aggregate([c for c in completed if c['dataset'] == dataset])) for dataset in pp.DATASETS},
        failures=failures, audit_errors=errors, source_evidence=evidence, heldout_shadow_audit=terminal_audit,
        calibration_sha256=frozen['sha256'] if frozen else None, c1_only=True, c4_measured=False,
        sensitivity_enabled=sensitivity_enabled, numerical_qualification=qualification,
        sensitivity_admission=admission,
        ttft_scope='hot online gate/prepare/lookup/fallback included; model load/offline KV build excluded',
        native_provenance='caller-declared native backend; seals are not signatures',
        full_model_compute_fraction_claimed=False, population_accuracy_certified=False,
        runtime_admission_created=False, performance_winner_filter_applied=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--output', type=Path, help='Optional new exclusive JSON output; never overwrite.')
    args = parser.parse_args(argv)
    result = read_results(args.out_dir)
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output is not None:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(rendered)
    else:
        print(rendered, end='')
    return 0 if not result.get('audit_errors') else 2


if __name__ == '__main__':
    raise SystemExit(main())
