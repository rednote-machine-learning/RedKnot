"""CPU-only, bounded dependency transport for the pinned official SDK.

No source/lock edits, no HOME override, no model calls. The UV attempt is allowed
to use its frozen artifact URLs; --default-index is not claimed to rewrite them.
The fallback uses the caller's pip configuration and verifies every locked hash.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import tomllib
from urllib.parse import urlsplit
import zipfile

B = Path(__file__).resolve().parent
SDK = B / 'benchmarks/vendor/software-agent-sdk'
WORKSPACES = ('openhands-sdk', 'openhands-tools', 'openhands-workspace', 'openhands-agent-server')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def seal(value):
    value = dict(value)
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def lock_bootstrap(lock):
    lines = []
    for name in ('setuptools', 'wheel', 'packaging'):
        packages = [p for p in lock['package'] if p['name'] == name]
        if len(packages) != 1:
            raise ValueError('Ambiguous locked build backend: ' + name)
        p = packages[0]
        hashes = sorted({a['hash'] for a in p.get('wheels', [])} | ({p['sdist']['hash']} if p.get('sdist') else set()))
        if not hashes or any(not h.startswith('sha256:') for h in hashes):
            raise ValueError('Missing locked SHA256 for ' + name)
        lines.append(name + '==' + p['version'] + ' ' + ' '.join('--hash=' + h for h in hashes))
    return '\n'.join(lines) + '\n'


def source_identity():
    commit = subprocess.check_output(['git', '-C', str(SDK), 'rev-parse', 'HEAD'], text=True).strip()
    paths = subprocess.check_output(['git', '-C', str(SDK), 'ls-files', '-z']).split(b'\0')
    files = {os.fsdecode(p): sha(SDK / os.fsdecode(p)) for p in paths if p and (SDK / os.fsdecode(p)).is_file()}
    return {'commit': commit, 'uv_lock_sha256': sha(SDK / 'uv.lock'), 'tracked_files_sha256': hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def index_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError('Index must be HTTP(S) without embedded credentials/query')
    return value


class Steps:
    def __init__(self, root, env, seconds):
        self.root, self.env, self.seconds, self.records = root, env, seconds, []

    def run(self, name, argv, *, required=True, seconds=None):
        limit = min(self.seconds, seconds or self.seconds)
        log = self.root / (name + '.log')
        start = time.time()
        with log.open('xb') as output:
            process = subprocess.Popen([str(x) for x in argv], cwd=SDK, env=self.env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                # This session is created by this exact unreaped child. Never
                # signal global package managers or an unrelated process group.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait(timeout=5)
        record = {'name': name, 'argv': [str(x) for x in argv], 'pid': process.pid, 'started_unix': start, 'elapsed_s': time.time() - start, 'timeout_s': limit, 'timed_out': timed_out, 'exit_code': code, 'log': str(log), 'log_sha256': sha(log)}
        self.records.append(record)
        write_json(self.root / 'progress.json', {'schema': 's2-sdk313-prewarm-progress-v1', 'steps': self.records})
        print(json.dumps({'step': name, 'exit_code': code, 'timed_out': timed_out}), flush=True)
        if required and (code != 0 or timed_out):
            raise RuntimeError('CPU preparation step failed: ' + name + '; inspect ' + str(log))
        return code == 0 and not timed_out


SNAPSHOT = """import importlib.metadata as m,json,platform,sys
p={}
for d in m.distributions():
 name=d.metadata['Name'].lower().replace('_','-').replace('.','-')
 if name in p: raise RuntimeError('Duplicate installed distribution: '+name)
 p[name]=d.version
print(json.dumps({'python':sys.version,'executable':sys.executable,'machine':platform.machine(),'platform':sys.platform,'packages':dict(sorted(p.items()))},sort_keys=True))
"""


def snapshot(python):
    return json.loads(subprocess.check_output([str(python), '-c', SNAPSHOT], text=True, timeout=30))


def wheel_inventory(directory, packages):
    available = {}
    for path in sorted(Path(directory).glob('*.whl')):
        with zipfile.ZipFile(path) as wheel:
            # A distribution may vendor other distributions' metadata as data
            # (py1.11.0 includes apipkg/iniconfig). Wheel metadata is top-level.
            metadata = [n for n in wheel.namelist() if len(Path(n).parts) == 2 and Path(n).parts[0].endswith('.dist-info') and Path(n).parts[1] == 'METADATA']
            if len(metadata) != 1:
                raise ValueError('Ambiguous wheel metadata: ' + path.name)
            message = email.message_from_bytes(wheel.read(metadata[0]))
        key = (canonical(message['Name']), message['Version'])
        if key in available:
            raise ValueError('Multiple wheels for one exact package: ' + repr(key))
        available[key] = {'name': key[0], 'version': key[1], 'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}
    selected = []
    for name, version in sorted(packages.items()):
        if (name, version) not in available:
            raise ValueError('No offline wheel for installed distribution: ' + name + '==' + version)
        selected.append(available[(name, version)])
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--uv', required=True)
    parser.add_argument('--pip-python', required=True, help='Existing host Python with pip; its configured indexes are inherited')
    parser.add_argument('--managed-python-dir', required=True)
    parser.add_argument('--output-dir', required=True, help='Fresh directory; never reuse a partial run')
    parser.add_argument('--attempt-timeout', type=int, default=300)
    parser.add_argument('--uv-probe-timeout', type=int, default=30, help='Short first UV network attempts; downloads keep --attempt-timeout')
    parser.add_argument('--default-index', type=index_url)
    parser.add_argument('--with-transformers', action='store_true', help='Resolve official extra constraint once, record any changes from the SDK lock')
    args = parser.parse_args(argv)
    if not 1 <= args.attempt_timeout <= 300:
        parser.error('Each attempt must be between 1 and 300 seconds')
    if not 1 <= args.uv_probe_timeout <= args.attempt_timeout:
        parser.error('UV probe timeout must be between 1 and --attempt-timeout')
    from run_official import verify_inputs
    manifest, _ = verify_inputs()
    before = source_identity()
    if before['commit'] != manifest['sources']['sdk']['commit']:
        raise ValueError('SDK is not the pinned revision')
    root = Path(args.output_dir).resolve()
    if root.is_relative_to(SDK):
        raise ValueError('Preparation outputs must be outside immutable SDK source')
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(root / 'env'), UV_PYTHON_INSTALL_DIR=str(Path(args.managed_python_dir).resolve()), UV_CACHE_DIR=str(root / 'uv-cache'), UV_HTTP_TIMEOUT='30', UV_HTTP_CONNECT_TIMEOUT='10', UV_HTTP_RETRIES='1', UV_PYTHON_DOWNLOADS='never', PIP_DISABLE_PIP_VERSION_CHECK='1')
    # Do not inspect/export pip credentials or override the caller's HOME/config.
    steps = Steps(root, env, args.attempt_timeout)
    result = {'schema': 's2-sdk313-prewarm-v1', 'source': before, 'gpu': False, 'model_calls': 0, 'attempt_timeout_s': args.attempt_timeout, 'frozen_url_policy': 'The UV attempt may use lockfile URLs; --default-index is not asserted to rewrite them.', 'status': 'running'}
    try:
        uv = str(Path(args.uv).resolve())
        steps.run('uv-version', [uv, '--version'], seconds=10)
        steps.run('create-env', [uv, 'venv', '--offline', '--python-preference', 'only-managed', '--python', '3.13.18', root / 'env'])
        python = root / 'env/bin/python'
        info = snapshot(python)
        if not info['python'].startswith('3.13.18 ') or info['platform'] != 'linux' or info['machine'] != 'x86_64':
            raise ValueError('Expected actual managed Linux x86_64 CPython 3.13.18')
        req = root / 'sdk-locked.requirements.txt'
        steps.run('export-frozen', [uv, 'export', '--frozen', '--offline', '--no-emit-workspace', '--extra', 'boto3', '--format', 'requirements.txt', '--output-file', req])
        if ' --hash=sha256:' not in req.read_text() and '\n    --hash=sha256:' not in req.read_text():
            raise ValueError('Frozen export did not contain hashes')
        lock = tomllib.loads((SDK / 'uv.lock').read_text())
        bootstrap = root / 'build-backends.requirements.txt'
        bootstrap.write_text(lock_bootstrap(lock))
        # Python uses the invocation path to identify a venv. Resolving its
        # symlink would silently switch to the system Python and lose pip.
        pip = [str(Path(args.pip_python).absolute()), '-m', 'pip', '--python', str(python)]
        steps.run('locked-build-backends', pip + ['install', '--require-hashes', '--no-deps', '-r', str(bootstrap)])
        sync = [uv, 'sync', '--frozen', '--no-editable', '--managed-python', '--extra', 'boto3', '--python', '3.13.18']
        online = sync + (['--default-index', args.default_index] if args.default_index else [])
        online_ok = steps.run('uv-frozen-online-attempt', online, required=False, seconds=args.uv_probe_timeout)
        downloads = root / 'downloads'; downloads.mkdir()
        # Pip chooses artifacts with the actual target interpreter and verifies
        # the SDK's original hashes, even when using a configured mirror.
        steps.run('download-locked-artifacts', pip + ['download', '--require-hashes', '--no-deps', '--no-build-isolation', '-r', str(req), '--dest', str(downloads)])
        if not online_ok:
            steps.run('pip-locked-offline-fallback', pip + ['install', '--no-index', '--find-links', str(downloads), '--require-hashes', '--no-deps', '--no-build-isolation', '-r', str(req)])
        steps.run('verify-frozen-offline', sync + ['--offline', '--no-build-isolation'])
        locked = snapshot(python)
        write_json(root / 'sdk-locked-installed.json', locked)
        result['locked_offline_sync_verified'] = True
        wheels = root / 'wheels'; wheels.mkdir()
        steps.run('build-locked-wheels', pip + ['wheel', '--no-index', '--find-links', str(downloads), '--require-hashes', '--no-deps', '--no-build-isolation', '-r', str(req), '--wheel-dir', str(wheels)])
        steps.run('build-workspace-wheels', pip + ['wheel', '--no-deps', '--no-build-isolation', '--wheel-dir', str(wheels)] + [str(SDK / p) for p in WORKSPACES])
        if args.with_transformers:
            extra = [uv, 'pip', 'install', '--python', str(python), 'transformers>=4.56.0,<5']
            if args.default_index:
                extra += ['--default-index', args.default_index]
            if not steps.run('official-transformers-uv-attempt', extra, required=False, seconds=args.uv_probe_timeout):
                steps.run('official-transformers-pip-fallback', pip + ['install', '--report', str(root / 'transformers-install-report.json'), 'transformers>=4.56.0,<5'])
        final = snapshot(python)
        write_json(root / 'final-installed.json', final)
        changes = {name: {'locked': locked['packages'].get(name), 'final': version} for name, version in final['packages'].items() if locked['packages'].get(name) != version}
        if changes:
            extra_req = root / 'resolved-extra.requirements.txt'
            extra_req.write_text(''.join(name + '==' + data['final'] + '\n' for name, data in sorted(changes.items())))
            steps.run('download-resolved-extra-wheels', pip + ['wheel', '--no-deps', '--no-build-isolation', '-r', str(extra_req), '--wheel-dir', str(wheels)])
        inventory = wheel_inventory(wheels, final['packages'])
        final_req = root / 'offline-final.requirements.txt'
        final_req.write_text(''.join(p['name'] + '==' + p['version'] + ' --hash=sha256:' + p['sha256'] + '\n' for p in inventory))
        # Independent fresh environment proves the wheel bundle itself works;
        # installed host packages or a warm UV cache cannot satisfy this check.
        steps.run('create-offline-replay-env', [uv, 'venv', '--offline', '--python-preference', 'only-managed', '--python', '3.13.18', root / 'replay-env'])
        replay = root / 'replay-env/bin/python'
        steps.run('replay-hashed-wheel-bundle', [uv, '--no-config', 'pip', 'install', '--offline', '--no-index', '--find-links', str(wheels), '--require-hashes', '--no-deps', '--python', str(replay), '-r', str(final_req)])
        if snapshot(replay)['packages'] != final['packages']:
            raise ValueError('Fresh offline replay package set differs')
        steps.run('replay-dependency-check', [str(replay), '-m', 'pip', 'check'], seconds=30)
        result.update(status='complete', python=info, locked_installed=locked, final_installed=final, official_transformers_constraint='>=4.56.0,<5' if args.with_transformers else None, post_lock_dependency_changes=changes, final_environment_is_exact_original_lock=not bool(changes), wheel_inventory=inventory, wheel_directory=str(wheels), final_requirements=str(final_req), final_requirements_sha256=sha(final_req), frozen_export_sha256=sha(req), offline_replay_verified=True, dependency_check_passed=True, transport='Exact lock hashes for upstream registry artifacts; wheels built from pinned workspace and hash-verified sdists; resolved official extra changes separately recorded.', remaining='Actual built SIF package metadata, guarded SDK wire probe, workspace isolation, and task base tree admission remain required.')
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        after = source_identity()
        result.update(source_after=after, source_unchanged=before == after, steps=steps.records)
        if before != after:
            result.update(status='failed', error='Pinned SDK tracked source changed during preparation')
        write_json(root / 'receipt.json', seal(result))
    if result['status'] != 'complete':
        raise RuntimeError(result.get('error', 'Preparation incomplete'))
    print(json.dumps({'status': 'complete', 'receipt': str(root / 'receipt.json'), 'receipt_file_sha256': sha(root / 'receipt.json')}))


if __name__ == '__main__':
    main()
