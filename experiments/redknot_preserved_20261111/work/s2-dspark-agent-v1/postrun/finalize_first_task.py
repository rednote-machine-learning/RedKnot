#!/usr/bin/env python3
"""Prepare official CPU grading commands after terminal + GPU cleanup with holder OFF.

Never executes a command, contacts a server, imports a model, or changes raw
experiment files. The printed plan uses the actual frozen framework environment.
"""
import argparse
import hashlib
import json
from pathlib import Path



def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, sealed=True):
    value = json.loads(Path(path).read_text())
    if sealed:
        require(value.get('sha256') == digest({k:v for k,v in value.items() if k != 'sha256'}),
                'artifact seal differs: ' + str(path))
    return value


def ref(path):
    return dict(path=str(Path(path).resolve()), file_sha256=sha(path))


def verify_reference(reference):
    require(sha(reference['path']) == reference['file_sha256'], 'raw reference changed')
    return read(reference['path'])


def official_rows(path, instance_id):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    require(all(isinstance(row, dict) and row.get('instance_id') == instance_id for row in rows),
            'official output task differs')
    return rows


def prepare(args):
    config_path = args.triplet_config.resolve()
    config, lifecycle = read(config_path), read(args.lifecycle_config)
    life_path = Path(lifecycle['output_dir']) / 'terminal.json'
    life = read(life_path)
    require(life['schema'] == 's2-dspark-agent-lifecycle-terminal-v1'
        and life['config_sha256'] == lifecycle['sha256']
        and life.get('physical_idle_verified') is True
        and life.get('holder_policy') == 'OFF' and life.get('holder_started') is False,
        'GPU cleanup and holder-OFF terminal required before grading')
    require(lifecycle['runner_argv'][-2:] == ['--config', str(config_path)]
        and lifecycle['source_sha256'].get(str(config_path)) == sha(config_path),
        'lifecycle is detached from this triplet configuration')
    root = Path(lifecycle['output_dir']) / 'benchmark'
    terminal_path = root / 'triplet-terminal.json'
    terminal = read(terminal_path)
    require(terminal['schema'] == 's2-agent-triplet-terminal-v1'
        and terminal['config_sha256'] == config['sha256'] and terminal['instance_id'] == config['instance_id'],
        'actual triplet terminal/config mismatch')
    require(terminal.get('released_unix') == life.get('released_unix')
        and terminal.get('absolute_deadline_unix') == life.get('deadline_unix'),
        'triplet and lifecycle release/deadline differ')
    framework = Path(config.get('framework_pinned_source_dir', Path(config['framework_script']).parent))
    grader = framework / 'offline_grader.py'
    require(config['source_sha256'].get(str(grader)) == sha(grader), 'official grader adapter changed since freeze')
    image = read(config['image_provenance'], sealed=False)
    base_ref = image['base_environment_admission']
    verify_reference(base_ref)
    registry = args.registry_manifest.resolve() if args.registry_manifest else framework.parent / 'container/oci/swebench_sweb.eval.x86_64.sphinx-doc_1776_sphinx-8638/manifest.raw.json'
    require(registry.is_file(), 'actual official OCI manifest file missing')
    output = args.output_dir.resolve()
    require(not output.exists(), 'new finalization directory required; never overwrite a prior plan/result')
    output.mkdir(parents=True)
    arms, commands, grades = [], [], []
    known = {a['arm']:a for a in terminal['arms']}
    require(len(known) == len(terminal['arms']), 'duplicate arm terminal')
    for arm in ('Full', 'Prefix', 'S2'):
        state = known.get(arm, {})
        arm_root = root / 'agent_runs' / arm / 'ordinal00'
        paths = list(arm_root.rglob('output.jsonl')) if arm_root.exists() else []
        result = dict(arm=arm, raw_arm_status=state.get('status', 'not_started'),
                      grading_prepared=False, official_resolved=None)
        framework_log = root / 'arms' / arm / 'framework.log'
        if framework_log.is_file():
            result['raw_framework_log'] = ref(framework_log)
        result['raw_error_outputs'] = [ref(p) for p in sorted(arm_root.rglob('output_errors.jsonl'))] if arm_root.exists() else []
        arms.append(result)
        if not paths:
            result['reason'] = 'no closed official output; do not invent a prediction or score'
            continue
        require(len(paths) == 1, 'ambiguous official output files for ' + arm)
        source = paths[0]
        result['raw_output'] = ref(source)
        try:
            rows = official_rows(source, config['instance_id'])
        except json.JSONDecodeError:
            result['reason'] = 'incomplete/corrupt raw output; preserve it without grading an earlier attempt'
            continue
        if state.get('outcome'):
            require(state['outcome']['file_sha256'] == sha(source), 'terminal-bound official output changed')
        if not rows:
            result.update(records=0, reason='empty official output; no closed model result, prediction or official score')
            continue
        row = rows[-1]
        patch_text = (row.get('test_result') or {}).get('git_patch')
        result.update(records=len(rows), selected_record='last official record, without outcome-based selection',
                      selected_attempt=row.get('attempt'), raw_agent_error=row.get('error'))
        if not isinstance(patch_text, str):
            result['reason'] = 'final official record has no patch string; do not substitute an earlier attempt'
            continue
        patch_sha = hashlib.sha256(patch_text.encode()).hexdigest()
        receipts = []
        for path in sorted((arm_root / 'patch-baseline').glob('*.json')):
            value = read(path)
            if value.get('schema') == 's2-environment-relative-patch-v1' and value.get('mode') == 'final' and value['model_patch_ref']['file_sha256'] == patch_sha:
                require(value['instance_id'] == config['instance_id'] and value['official_scoring_unchanged'] is True
                    and value['filename_or_hunk_filtering'] is False, 'patch extraction policy differs')
                for key in ('raw_patch_ref', 'model_patch_ref'):
                    require(sha(value[key]['path']) == value[key]['file_sha256'], 'preserved patch bytes changed')
                receipts.append(ref(path))
        require(receipts, 'final patch is not bound to an actual environment-relative extraction receipt: ' + arm)
        prediction = output / (arm + '.predictions.jsonl')
        with prediction.open('x') as f:
            f.write(json.dumps(dict(instance_id=config['instance_id'], model_name_or_path='DeepSeek-V4.1-Flash-' + arm,
                                   model_patch=patch_text), ensure_ascii=False) + '\n')
        run_id = 'task0-dspark-official-v1'
        argv = [config['framework_python'], '-B', str(grader), '--predictions', str(prediction),
                '--arm', arm, '--run-id', run_id, '--output-root', str(output / 'grades'),
                '--base-image-proof', base_ref['path'], '--registry-manifest', str(registry), '--run']
        environment = dict(config['environment']['framework'], CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        commands.append(dict(arm=arm, argv=argv, cwd=str(framework / 'benchmarks'), environment=environment,
                             execute_only_after_this_terminal_admission=True))
        report = output / 'grades' / arm / run_id / 'reports.json'
        grades += ['--grade', arm + '=' + str(report)]
        result.update(grading_prepared=True, prediction=ref(prediction), model_patch_sha256=patch_sha,
                      patch_extraction_receipts=receipts, expected_official_report=str(report),
                      incomplete_arm_remains_incomplete=state.get('status') != 'complete_awaiting_cpu_grading')
    reader = Path(config['cwd']) / 'report/read_triplet.py'
    plan = dict(schema='s2-first-task-postrun-plan-v3', status='commands_prepared_not_executed',
        raw_lifecycle_status=life['status'], raw_triplet_status=terminal['status'],
        round_released_unix=terminal.get('released_unix'),
        absolute_deadline_unix=terminal.get('absolute_deadline_unix'), holder_policy='OFF',
        lifecycle_terminal=ref(life_path), physical_idle=life['after'],
        triplet_terminal=ref(terminal_path), triplet_config=ref(config_path), arms=arms,
        grading_commands=commands, grader_source=ref(grader), reader_source=ref(reader),
        reader_command=[config['framework_python'], '-B', str(reader), '--directory', str(root),
                        '--output', str(output / 'triplet-results.json'), *grades],
        reader_instruction='Run after grading; omit --grade for a grader failure/no reports.json. Never replace failed or missing scores with resolved=false.',
        collect=[str(life_path.parent), str(root), str(output)],
        preserve=['raw failures/timeouts and not-started arms', 'native request/qualification journals and startup proofs',
                  'official output/history plus both patch forms and extraction receipts',
                  'official scores/test logs/reports and offline-grader-receipt.json for each attempted arm'],
        model_calls=0, gpu_calls=0, official_grading_executed=False)
    plan['sha256'] = digest(plan)
    path = output / 'postrun-plan.json'
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    return dict(status=plan['status'], plan=str(path), sha256=plan['sha256'],
                prepared_arms=[a['arm'] for a in arms if a['grading_prepared']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triplet-config', type=Path, required=True)
    parser.add_argument('--lifecycle-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--registry-manifest', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args), indent=2)); return 0
    except Exception as error:
        print(json.dumps(dict(status='not_ready_or_invalid', error=str(error), gpu_calls=0))); return 2


if __name__ == '__main__':
    raise SystemExit(main())
