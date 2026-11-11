"""Read-only CPU audit of the exploratory A/B/C oracle-repair diagnostic.

Only complete raw three-arm closures enter scores. This reader never emits
TTFT, speedup, QPS, FLOPs saved, a runtime-admission certificate, or a holdout
claim. Native producer labels and seals are integrity evidence, not signatures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_pilot import pilot_protocol as pp
from gpu_pilot import repair_protocol as rp
from kv_cache_engram.core import ValidationError, digest, integer, require, validate_digest


def _read(path, evidence):
    require(path.is_file() and not path.is_symlink(), 'missing or symlink evidence: '+path.name)
    require(path.stat().st_size <= 512*1024**2, 'JSON evidence exceeds 512 MiB budget')
    payload = path.read_bytes(); value = json.loads(payload)
    pp._sealed(value)
    evidence[path.name] = dict(file_sha256=hashlib.sha256(payload).hexdigest(), seal_sha256=value['sha256'])
    return value


def _scores(text, references):
    def normalize(value):
        return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', re.sub(r'[^\w\s]', '', value.lower())).split())
    words, result = normalize(text).split(), []
    for reference in references:
        gold = normalize(reference).split()
        common = sum((Counter(words) & Counter(gold)).values())
        result.append((float(normalize(text) == normalize(reference)),
                       2*common/(len(words)+len(gold)) if words or gold else 1.))
    return dict(answer_em=max(x[0] for x in result), answer_token_f1=max(x[1] for x in result))


def _flags(value):
    for name in ('formal_admission_unchanged', 'forced_controlled_diagnostic', 'no_ttft_claim',
                 'previously_exposed_exploratory_cohort'):
        require(value.get(name) is True, 'missing diagnostic/exposure safeguard: '+name)


def _preflight(value):
    require(value.get('protocol') == rp.repair_protocol(), 'foreign repair protocol')
    _flags(value)
    validate_digest(value.get('manifest_sha256'), 'input manifest seal')
    identity = value.get('identity', {})
    require(identity.get('repair_protocol_sha256') == value['protocol']['sha256'],
            'execution identity does not bind repair protocol')
    for name in ('execution_configuration_sha256', 'checkpoint_index_sha256', 'tokenizer_json_sha256'):
        validate_digest(identity.get(name), name)
    return rp.validate_cases(value.get('audit_cases'))


def _hook(value, phase, row, identity, *, mode='native'):
    scope = dict(model_sha256=identity['checkpoint_index_sha256'],
        tokenizer_sha256=identity['tokenizer_json_sha256'], execution_sha256=identity['execution_configuration_sha256'],
        representation='swa-postnorm-prerope-v1', storage_dtype='bfloat16-native-canonical',
        width=512, layers=list(pp.LAYERS))
    require(type(value) is dict and value.get('schema') == 'kv-cache-engram-native-hook-v1'
            and value.get('complete') is True and value.get('mode') == mode
            and value.get('native_execution_declared') is True and value.get('backend') == rp.BACKEND
            and value.get('full_q_attention_moe_preserved') is True and value.get('scope') == scope
            and value.get('input_sha256') == row['input_sha256'] and value.get('input_tokens') == row['input_tokens']
            and value.get('request_id') == phase['record']['request_id'] and value.get('phase') == phase['name']
            and value.get('coverage') == {str(layer): row['input_tokens'] for layer in pp.LAYERS},
            'native full-compute hook identity/coverage mismatch')
    require(value.get('events') == [], 'formal numerical gate must not be changed or executed in this diagnostic')
    counts = value.get('counts')
    require(type(counts) is dict and set(counts) == {str(layer) for layer in pp.LAYERS}, 'all native layers required')
    for layer in pp.LAYERS:
        count = counts[str(layer)]
        for key in ('q_rows', 'native_fused_rows', 'online_kv_rows'):
            require(type(count.get(key)) is int and count[key] == row['input_tokens'],
                    'all native Q/KV projections must still execute')
        for key in ('split_q_rows', 'reused_kv_rows', 'probe_rows', 'fallback_rows'):
            require(type(count.get(key)) is int and count[key] == 0,
                    'projection sparsity/formal gate unexpectedly changed')


def _qualification(directory, preflight, evidence):
    """Audit all split/gather/positioned-cast checks, not just a passed label."""
    row = preflight.get('qualification_case', {})
    require(type(row) is dict and row.get('index') == 0 and row.get('split') == 'calibration'
            and row.get('dataset') == 'hotpotqa' and type(row.get('sample_id')) is str
            and bool(row['sample_id']), 'qualification input binding missing')
    validate_digest(row.get('input_sha256'), 'qualification input')
    integer(row.get('input_tokens'), 'qualification input tokens', 8*rp.DOC+1)
    raw = _read(directory/'qualification_split_identity.json', evidence)
    closure = _read(directory/'qualification.json', evidence)
    name = 'qualification_split_identity'
    rid = f"kce-repair:0:{name}:{row['sample_id']}"
    _flags(raw)
    require(raw.get('schema') == 'kv-cache-engram-native-phase-v1'
            and raw.get('name') == name and raw.get('mode') == 'split_identity'
            and raw.get('arm') is None and raw.get('intervention') is None and raw.get('plan_sha256') is None
            and raw.get('split') == 'calibration' and raw.get('index') == 0
            and raw.get('dataset') == row['dataset'] and raw.get('sample_id') == row['sample_id']
            and raw.get('request_id') == rid and raw.get('input_ids_sha256') == row['input_sha256']
            and raw.get('input_tokens') == row['input_tokens']
            and raw.get('manifest_sha256') == preflight['manifest_sha256']
            and raw.get('protocol_sha256') == preflight['protocol']['sha256']
            and raw.get('execution_configuration_sha256') == preflight['identity']['execution_configuration_sha256']
            and raw.get('timing_valid') is False and raw.get('quality_eligible') is False
            and 'sensitivity' not in raw, 'qualification phase identity mismatch')
    rec = raw.get('record', {})
    require(rec.get('success') is True and rec.get('rank') == 0 and rec.get('request_id') == rid
            and rec.get('phase') == name and rec.get('sample_id') == row['sample_id']
            and rec.get('dataset') == row['dataset'] and rec.get('input_ids_sha256') == row['input_sha256']
            and rec.get('input_tokens') == row['input_tokens'], 'qualification request mismatch')
    ids = rec.get('output_ids')
    require(type(ids) is list and len(ids) == 1 and type(ids[0]) is int and ids[0] >= 0
            and rec.get('output_ids_sha256') == digest(ids) and rec.get('natural_EOS') is False
            and rec.get('truncated') is False, 'qualification requires exactly one complete diagnostic token')
    hook = raw.get('hook', {})
    _hook(hook, raw, row, preflight['identity'], mode='split_identity')
    require(hook.get('identity_exact') is True and hook.get('numerical_equivalence_claim') is True,
            'split identity qualification did not pass')
    metrics = {'q', 'kv_raw', 'kv_canonical', 'gather32', 'gather128', 'post_rope_cast',
               'post_rope_gather32', 'post_rope_gather128', 'post_rope_gather384', 'post_rope_gather512'}
    expected = {(layer, start, min(start+rp.CHUNK, row['input_tokens']))
                for layer in pp.LAYERS for start in range(0, row['input_tokens'], rp.CHUNK)}
    qualifications, seen = hook.get('qualification'), set()
    require(type(qualifications) is list, 'raw qualification checks missing')
    for item in qualifications:
        key = item.get('layer'), item.get('start'), item.get('end')
        require(all(type(x) is int for x in key) and key in expected and key not in seen
                and item.get('exact') is True, 'qualification native chunk missing/duplicate/foreign')
        seen.add(key)
        errors = item.get('errors')
        require(type(errors) is dict and set(errors) == metrics, 'qualification missing split/gather/post-RoPE check')
        require(all(rp.number(value, 'numerical identity error') == 0 for value in errors.values()),
                'nonzero numerical identity error')
    require(seen == expected, 'qualification lacks complete fourteen-layer/native-chunk coverage')
    require(closure.get('passed') is True and closure.get('quality_claim') is False
            and closure.get('phase_sha256') == raw['sha256'], 'qualification closure differs from raw phase')
    return dict(passed=True, phase_sha256=raw['sha256'], closure_sha256=closure['sha256'],
        input_tokens=row['input_tokens'], layers=len(pp.LAYERS), entries=len(seen),
        numerical_checks=len(seen)*len(metrics), maximum_error=0., quality_claim=False,
        all_split_gather_post_rope_checks_exact=True)


def _phase(value, row, preflight, plan, arm):
    pp._sealed(value); _flags(value)
    name = f"case{row['index']}.{arm}"
    require(value.get('schema') == rp.PHASE_SCHEMA and value.get('arm') == arm and value.get('name') == name
            and value.get('index') == row['index'] and value.get('dataset') == row['dataset']
            and value.get('split') in ('heldout', 'exploratory')
            and value.get('manifest_sha256') == preflight['manifest_sha256']
            and value.get('execution_configuration_sha256') == preflight['identity']['execution_configuration_sha256']
            and value.get('protocol_sha256') == preflight['protocol']['sha256']
            and value.get('input_ids_sha256') == row['input_sha256'] and value.get('plan_sha256') == plan['sha256']
            and value.get('timing_valid') is False and value.get('quality_eligible') is True,
            'raw repair phase identity/plan/quality mismatch')
    require(not any(key in value for key in ('sensitivity', 'timing_summary', 'ttft_speedup', 'qps')),
            'foreign diagnostics or performance claim mixed into repair phase')
    wall = rp.number(value.get('phase_wall_s'), 'diagnostic phase wall time', positive=True)
    rec = value.get('record')
    require(type(rec) is dict and rec.get('success') is True and rec.get('rank') == 0
            and rec.get('phase') == name and rec.get('sample_id') == row['sample_id']
            and rec.get('dataset') == row['dataset'] and rec.get('input_ids_sha256') == row['input_sha256']
            and rec.get('input_tokens') == row['input_tokens'] and rec.get('references') == row['references']
            and type(rec.get('request_id')) is str and bool(rec['request_id']), 'raw quality request mismatch')
    ids = rec.get('output_ids')
    require(type(ids) is list and 0 < len(ids) <= 64 and all(type(x) is int and x >= 0 for x in ids)
            and rec.get('output_ids_sha256') == digest(ids) and rec.get('natural_EOS') is True
            and rec.get('finish_reason') in ('stop', 'length') and type(rec.get('text')) is str
            and rec.get('truncated') is (rec['finish_reason'] == 'length')
            and (rec['finish_reason'] != 'length' or len(ids) == 64)
            and rec.get('continuation_is_quality_evidence') is not False,
            'quality requires complete EOS/64-token generation with retained length flags')
    scores = _scores(rec['text'], row['references'])
    for metric, score in scores.items():
        rp.number(rec.get(metric), metric)
        require(math.isclose(rec[metric], score, rel_tol=0, abs_tol=1e-12), 'quality score differs from fixed gold')
    _hook(value.get('hook'), value, row, preflight['identity'])
    counts = rp.validate_intervention(value.get('intervention'), plan, row, arm=arm,
                                     request_id=rec['request_id'], phase=name)
    return dict(scores, text=rec['text'], references=row['references'], output_ids=ids,
        output_tokens=len(ids), finish_reason=rec['finish_reason'], truncated=rec['truncated'],
        row_accounting=counts, diagnostic_phase_wall_s=wall, phase_sha256=value['sha256'],
        phase_wall_is_not_online_ttft=True)


def comparisons(quality):
    result = {}
    for metric in ('answer_em', 'answer_token_f1'):
        a, b, c = (quality[arm][metric] for arm in rp.ARMS)
        result[metric] = dict(B_loss_vs_A_percentage_points=100*(a-b),
            C_gain_vs_B_percentage_points=100*(c-b), C_remaining_loss_vs_A_percentage_points=100*(a-c),
            gap_recovery_fraction=(c-b)/(a-b) if a > b else None,
            gap_recovery_applicable=a > b)
    return result


def transitions(cases):
    result = {}
    for first, second, label in ((rp.ARMS[0], rp.ARMS[1], 'A_to_B'),
                                 (rp.ARMS[1], rp.ARMS[2], 'B_to_C'),
                                 (rp.ARMS[0], rp.ARMS[2], 'A_to_C')):
        result[label] = dict(correct_to_wrong=sum(c['quality'][first]['answer_em'] == 1
                and c['quality'][second]['answer_em'] == 0 for c in cases),
            wrong_to_correct=sum(c['quality'][first]['answer_em'] == 0
                and c['quality'][second]['answer_em'] == 1 for c in cases),
            both_correct=sum(c['quality'][first]['answer_em'] == c['quality'][second]['answer_em'] == 1 for c in cases),
            both_wrong=sum(c['quality'][first]['answer_em'] == c['quality'][second]['answer_em'] == 0 for c in cases))
    return result


def aggregate(cases):
    if not cases:
        return None
    quality = {arm: {metric: sum(case['quality'][arm][metric] for case in cases)/len(cases)
                    for metric in ('answer_em', 'answer_token_f1')} for arm in rp.ARMS}
    accounting = {}
    for arm in rp.ARMS:
        rows = [case['quality'][arm]['row_accounting'] for case in cases]
        fields = ('source_rows', 'injected_rows', 'oracle_restored_rows', 'reused_rows',
                  'native_q_rows', 'native_kv_projection_rows', 'ten_layer_prompt_row_denominator',
                  'fourteen_layer_prompt_row_denominator')
        accounting[arm] = {key: sum(value[key] for value in rows) for key in fields}
        accounting[arm]['cache_rows_remaining_fraction_of_ten_layer_prompt_rows'] = (
            accounting[arm]['reused_rows']/accounting[arm]['ten_layer_prompt_row_denominator'])
        accounting[arm]['not_saved_compute_or_flops'] = True
    return dict(case_count=len(cases), quality=quality, comparisons=comparisons(quality),
        transitions=transitions(cases), row_accounting=accounting,
        length_counts={arm: sum(case['quality'][arm]['truncated'] for case in cases) for arm in rp.ARMS},
        diagnostic_phase_wall_s={arm: sum(case['quality'][arm]['diagnostic_phase_wall_s'] for case in cases)
                                 for arm in rp.ARMS},
        phase_wall_is_not_online_ttft=True, previously_exposed_exploratory_cohort=True)


def _case(directory, triple, row, preflight, evidence):
    _flags(triple)
    require(triple.get('schema') == rp.TRIPLE_SCHEMA and triple.get('index') == row['index']
            and triple.get('dataset') == row['dataset'] and triple.get('sample_id') == row['sample_id']
            and triple.get('input_ids_sha256') == row['input_sha256'] and triple.get('input_tokens') == row['input_tokens']
            and triple.get('manifest_sha256') == preflight['manifest_sha256']
            and triple.get('execution_configuration_sha256') == preflight['identity']['execution_configuration_sha256']
            and triple.get('protocol_sha256') == preflight['protocol']['sha256']
            and triple.get('all_negative_results_retained') is True, 'triple closure identity mismatch')
    plan = _read(directory/f"case{row['index']}.plan.json", evidence)
    rp.validate_plan(plan, row)
    require(triple.get('plan_sha256') == plan['sha256'] and type(triple.get('phase_sha256')) is dict
            and set(triple['phase_sha256']) == set(rp.ARMS), 'triple plan/three-arm closure missing')
    quality = {}
    for arm in rp.ARMS:
        raw = _read(directory/f"case{row['index']}.{arm}.json", evidence)
        require(triple['phase_sha256'][arm] == raw['sha256'], 'triple/raw phase seal mismatch')
        quality[arm] = _phase(raw, row, preflight, plan, arm)
    return dict(index=row['index'], dataset=row['dataset'], sample_id=row['sample_id'],
        question=row['question'], references=row['references'], input_tokens=row['input_tokens'],
        quality=quality, comparisons=comparisons(quality), plan_sha256=plan['sha256'],
        triple_sha256=triple['sha256'], complete=True, previously_exposed_exploratory_cohort=True)


def read_results(out_dir):
    directory = Path(out_dir)
    require(directory.is_dir(), 'repair output directory missing')
    evidence, cases, pending, errors, failures = {}, [], [], [], []
    if not (directory/'preflight.json').exists():
        return pp.seal_record(dict(schema=rp.READOUT_SCHEMA, status='awaiting_preflight', completed=0,
                                  planned=4, cases=[], no_ttft_claim=True))
    preflight = _read(directory/'preflight.json', evidence)
    expected = _preflight(preflight)
    qualification = None
    if (directory/'qualification.json').exists():
        qualification = _qualification(directory, preflight, evidence)
    for path in sorted(directory.glob('failure.rank*.json')):
        failure = _read(path, evidence)
        require(failure.get('schema') == rp.RUN_SCHEMA and failure.get('status') == 'failed',
                'invalid repair failure record')
        failures.append(dict(rank=failure.get('rank'), error=failure.get('error'), sha256=failure['sha256']))
    for row in expected:
        path = directory/f"case{row['index']}.triple.json"
        if not path.exists():
            pending.append(row['index']); continue
        require(qualification is not None, 'raw numerical qualification required before any triple')
        try:
            cases.append(_case(directory, _read(path, evidence), row, preflight, evidence))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(dict(index=row['index'], file=path.name, error=str(exc)))
    status = 'failed' if failures else 'running'
    if (directory/'run.json').exists():
        run = _read(directory/'run.json', evidence); _flags(run)
        require(run.get('schema') == rp.RUN_SCHEMA and run.get('status') == 'complete'
                and run.get('identity') == preflight['identity']
                and run.get('manifest_sha256') == preflight['manifest_sha256']
                and run.get('protocol_sha256') == preflight['protocol']['sha256']
                and qualification is not None
                and len(cases) == 4 and not errors and not failures
                and [value.get('sha256') for value in run.get('cases', [])] == [case['triple_sha256'] for case in cases],
                'terminal complete lacks four matched raw three-arm closures')
        status = 'complete'
    if errors:
        status = 'evidence_invalid'
    return pp.seal_record(dict(schema=rp.READOUT_SCHEMA, status=status, completed=len(cases), planned=4,
        cases=cases, pending_case_indices=pending, audit_errors=errors, failures=failures,
        global_summary=aggregate(cases),
        by_dataset={dataset: dict(completed=sum(case['dataset'] == dataset for case in cases), planned=2,
            summary=aggregate([case for case in cases if case['dataset'] == dataset])) for dataset in pp.DATASETS},
        source_evidence=evidence, protocol_sha256=preflight['protocol']['sha256'], numerical_qualification=qualification,
        formal_admission_unchanged=True, forced_controlled_diagnostic=True, no_ttft_claim=True,
        previously_exposed_exploratory_cohort=True, independent_holdout_claimed=False,
        calibration_cases_in_quality=0, runtime_admission_created=False, accuracy_safety_claimed=False,
        model_flops_saved_claimed=False, native_provenance='producer-declared native; seals are not signatures'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--output', type=Path, help='New exclusive output file only; never overwrite')
    args = parser.parse_args(argv)
    value = read_results(args.out_dir)
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output is None:
        print(text, end='')
    else:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(text)
    return 2 if value.get('audit_errors') else 0


if __name__ == '__main__':
    raise SystemExit(main())
