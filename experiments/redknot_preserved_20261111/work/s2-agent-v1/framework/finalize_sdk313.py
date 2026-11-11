"""Finish the witnessed metadata-only prewarm failure without repeating installs.

The original failed receipt, logs, environment and artifacts are read-only. A
fresh directory receives selected wheel copies and an independently replayed
offline environment, then a new compatible success receipt referencing failure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib

from prewarm_sdk313 import SDK, Steps, seal, sha, snapshot, source_identity, wheel_inventory, write_json


def read_failed(path, expected_seal):
    raw = json.loads(path.read_text())
    body = dict(raw); digest = body.pop('sha256', None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != expected_seal or actual != digest:
        raise ValueError('Original failed receipt seal mismatch')
    if raw.get('schema') != 's2-sdk313-prewarm-v1' or raw.get('status') != 'failed' or not raw.get('error', '').startswith('ValueError: Ambiguous wheel metadata:'):
        raise ValueError('Only the witnessed wheel-metadata failure is recoverable here')
    if raw.get('locked_offline_sync_verified') is not True or raw.get('source_unchanged') is not True or raw['source'] != raw['source_after']:
        raise ValueError('Failed run did not verify frozen source/environment')
    if len({s['name'] for s in raw['steps']}) != len(raw['steps']):
        raise ValueError('Duplicate original preparation steps')
    steps = {s['name']: s for s in raw['steps']}
    required = ('create-env', 'export-frozen', 'locked-build-backends', 'download-locked-artifacts', 'verify-frozen-offline', 'build-locked-wheels', 'build-workspace-wheels')
    for name in required:
        if name not in steps or steps[name]['exit_code'] != 0 or steps[name]['timed_out']:
            raise ValueError('Required original step incomplete: ' + name)
    if not any(steps.get(n, {}).get('exit_code') == 0 and not steps[n]['timed_out'] for n in ('official-transformers-uv-attempt', 'official-transformers-pip-fallback')):
        raise ValueError('Original official Transformers install not completed')
    root = path.parent.resolve()
    for step in raw['steps']:
        log = Path(step['log']).resolve()
        if not log.is_relative_to(root) or sha(log) != step['log_sha256']:
            raise ValueError('Original log missing/changed: ' + step['name'])
    progress = json.loads((root / 'progress.json').read_text())
    if progress.get('steps') != raw['steps']:
        raise ValueError('Original progress differs from terminal failed receipt')
    return raw


def locked_downloads(root):
    lock = tomllib.loads((SDK / 'uv.lock').read_text())
    allowed = {}
    for package in lock['package']:
        for item in package.get('wheels', []) + ([package['sdist']] if package.get('sdist') else []):
            allowed.setdefault(item['hash'].removeprefix('sha256:'), []).append({'name': package['name'], 'version': package['version']})
    rows = []
    for path in sorted((root / 'downloads').iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Unexpected downloaded artifact')
        digest = sha(path)
        if digest not in allowed:
            raise ValueError('Original downloaded artifact not in frozen lock: ' + path.name)
        rows.append({'file': path.name, 'sha256': digest, 'bytes': path.stat().st_size, 'locked_packages': allowed[digest]})
    if not rows:
        raise ValueError('No original downloaded artifacts')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--failed-receipt', required=True)
    p.add_argument('--failed-receipt-seal', required=True)
    p.add_argument('--uv', required=True)
    p.add_argument('--managed-python-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--attempt-timeout', type=int, default=300)
    a = p.parse_args(argv)
    if not 1 <= a.attempt_timeout <= 300:
        p.error('Each offline step must be bounded to at most 300 seconds')
    from run_official import verify_inputs
    manifest, _ = verify_inputs()
    failed = Path(a.failed_receipt).resolve()
    original = read_failed(failed, a.failed_receipt_seal)
    before = source_identity()
    if before != original['source'] or before['commit'] != manifest['sources']['sdk']['commit']:
        raise ValueError('Actual pinned SDK source differs from original prewarm')
    parent = failed.parent
    locked_path, final_path = parent / 'sdk-locked-installed.json', parent / 'final-installed.json'
    locked = json.loads(locked_path.read_text()); final = json.loads(final_path.read_text())
    actual = snapshot(parent / 'env/bin/python')
    if actual != final:
        raise ValueError('Original installed environment no longer matches its saved final snapshot')
    if not actual['python'].startswith('3.13.18 ') or actual['machine'] != 'x86_64' or actual['platform'] != 'linux':
        raise ValueError('Unexpected actual Python/platform')
    changes = {name: {'locked': locked['packages'].get(name), 'final': final['packages'].get(name)} for name in sorted(set(locked['packages']) | set(final['packages'])) if locked['packages'].get(name) != final['packages'].get(name)}
    if changes:
        step = next((s for s in original['steps'] if s['name'] == 'download-resolved-extra-wheels'), None)
        if not step or step['exit_code'] != 0 or step['timed_out']:
            raise ValueError('Original resolved extra wheels were not completed')
    downloads = locked_downloads(parent)
    inventory = wheel_inventory(parent / 'wheels', final['packages'])
    root = Path(a.output_dir).resolve()
    if root == parent or parent.is_relative_to(root) or root.is_relative_to(SDK):
        raise ValueError('Finalization requires a fresh isolated output directory')
    root.mkdir(parents=True, exist_ok=False)
    result = {'schema': 's2-sdk313-prewarm-v1', 'status': 'running', 'source': before, 'gpu': False, 'model_calls': 0, 'attempt_timeout_s': a.attempt_timeout, 'derived_from_failed_preparation': {'path': str(failed), 'file_sha256': sha(failed), 'seal': original['sha256'], 'error': original['error'], 'original_receipt_preserved': True}, 'recovery_scope': 'Top-level wheel metadata identification; no dependency downloads or original environment installs repeated.', 'original_verified_steps': original['steps'], 'locked_downloaded_artifacts': downloads, 'original_snapshot_files': {str(x): sha(x) for x in (locked_path, final_path, parent / 'sdk-locked.requirements.txt')}}
    env = dict(os.environ)
    env.update(UV_PYTHON_INSTALL_DIR=str(Path(a.managed_python_dir).resolve()), UV_CACHE_DIR=str(root / 'uv-cache'), UV_PROJECT_ENVIRONMENT=str(root / 'replay-env'), UV_PYTHON_DOWNLOADS='never', UV_OFFLINE='1')
    steps = Steps(root, env, a.attempt_timeout)
    try:
        wheels = root / 'wheels'; wheels.mkdir()
        for row in inventory:
            target = wheels / row['file']
            shutil.copyfile(parent / 'wheels' / row['file'], target)
            if sha(target) != row['sha256']:
                raise ValueError('Wheel changed during local copy')
        req = root / 'offline-final.requirements.txt'
        req.write_text(''.join(row['name'] + '==' + row['version'] + ' --hash=sha256:' + row['sha256'] + '\n' for row in inventory))
        uv = str(Path(a.uv).absolute())
        steps.run('create-offline-replay-env', [uv, 'venv', '--offline', '--python-preference', 'only-managed', '--python', '3.13.18', root / 'replay-env'])
        replay = root / 'replay-env/bin/python'
        steps.run('replay-hashed-wheel-bundle', [uv, '--no-config', 'pip', 'install', '--offline', '--no-index', '--find-links', str(wheels), '--require-hashes', '--no-deps', '--python', str(replay), '-r', str(req)])
        replay_snapshot = snapshot(replay)
        if replay_snapshot['packages'] != final['packages']:
            raise ValueError('Fresh offline replay package set differs')
        steps.run('replay-dependency-check', [str(replay), '-m', 'pip', 'check'], seconds=30)
        write_json(root / 'replayed-installed.json', replay_snapshot)
        # Bind both original snapshots and the independently constructed bundle.
        write_json(root / 'sdk-locked-installed.json', locked)
        write_json(root / 'final-installed.json', final)
        if sha(failed) != result['derived_from_failed_preparation']['file_sha256'] or snapshot(parent / 'env/bin/python') != actual:
            raise ValueError('Original evidence/environment changed during finalization')
        result.update(status='complete', python=actual, locked_offline_sync_verified=True, locked_installed=locked, final_installed=final, official_transformers_constraint='>=4.56.0,<5', post_lock_dependency_changes=changes, final_environment_is_exact_original_lock=not bool(changes), wheel_inventory=inventory, wheel_directory=str(wheels), final_requirements=str(req), final_requirements_sha256=sha(req), frozen_export_sha256=sha(parent / 'sdk-locked.requirements.txt'), offline_replay_verified=True, dependency_check_passed=True, transport='Original hash-verified downloads and source-built wheels; local copy with SHA verification and independent offline replay.', remaining='Actual SIF package proof, guarded SDK wire, workspace isolation and official base environment admission remain required.')
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        after = source_identity()
        result.update(source_after=after, source_unchanged=before == after, steps=steps.records)
        if before != after:
            result.update(status='failed', error='Pinned SDK source changed during finalization')
        write_json(root / 'receipt.json', seal(result))
    if result['status'] != 'complete':
        raise RuntimeError(result['error'])
    print(json.dumps({'status': 'complete', 'receipt': str(root / 'receipt.json'), 'receipt_file_sha256': sha(root / 'receipt.json')}))


if __name__ == '__main__':
    main()
