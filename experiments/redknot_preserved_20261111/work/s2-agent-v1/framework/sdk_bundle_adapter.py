"""Explicit dependency-install transport adapter; official SDK sources unchanged."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from offline_image_adapter import safe_path, sha_file


def validate_bundle(receipt_path, expected_sdk_commit, expected_lock_sha):
    path = safe_path(receipt_path)
    raw = json.loads(path.read_text())
    body = dict(raw); digest = body.pop('sha256', None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != actual:
        raise ValueError('Dependency bundle receipt seal mismatch')
    if raw.get('schema') != 's2-sdk313-prewarm-v1' or raw.get('status') != 'complete':
        raise ValueError('Dependency bundle preparation incomplete')
    for field in ('source_unchanged', 'locked_offline_sync_verified', 'offline_replay_verified', 'dependency_check_passed'):
        if raw.get(field) is not True:
            raise ValueError('Dependency bundle lacks ' + field)
    if raw['source']['commit'] != expected_sdk_commit or raw['source']['uv_lock_sha256'] != expected_lock_sha:
        raise ValueError('Dependency bundle SDK source mismatch')
    if raw.get('official_transformers_constraint') != '>=4.56.0,<5':
        raise ValueError('Official Transformers install must be resolved and included')
    wheels = safe_path(raw['wheel_directory']); requirements = safe_path(raw['final_requirements'])
    if not wheels.is_relative_to(path.parent) or not requirements.is_relative_to(path.parent):
        raise ValueError('Dependency bundle paths escape preparation root')
    if sha_file(requirements) != raw['final_requirements_sha256']:
        raise ValueError('Dependency requirements changed')
    packages = raw['final_installed']['packages']
    seen = {}
    lines = []
    for row in raw['wheel_inventory']:
        name = row['name']; wheel = wheels / row['file']
        if Path(row['file']).name != row['file'] or not row['file'].endswith('.whl'):
            raise ValueError('Invalid wheel filename')
        if name in seen or sha_file(wheel) != row['sha256'] or wheel.stat().st_size != row['bytes']:
            raise ValueError('Dependency wheel duplicate or changed: ' + name)
        seen[name] = row['version']
        lines.append(name + '==' + row['version'] + ' --hash=sha256:' + row['sha256'] + '\n')
    if seen != packages or requirements.read_text() != ''.join(lines):
        raise ValueError('Dependency bundle does not match exact final installed set')
    if raw['final_installed']['platform'] != 'linux' or raw['final_installed']['machine'] != 'x86_64' or not raw['final_installed']['python'].startswith('3.13.18 '):
        raise ValueError('Dependency bundle Python/platform mismatch')
    return {'receipt_path': str(path), 'receipt_file_sha256': sha_file(path), 'receipt_seal': digest, 'wheel_directory': str(wheels), 'requirements_path': str(requirements), 'requirements_sha256': sha_file(requirements), 'sdk_commit': expected_sdk_commit, 'uv_lock_sha256': expected_lock_sha, 'packages': packages, 'post_lock_dependency_changes': raw['post_lock_dependency_changes'], 'wheel_count': len(seen)}


def adapt_dependency_install(original, bundle):
    old_sync = 'uv sync --frozen --no-editable --managed-python --extra boto3'
    old_extra = 'uv pip install --python /agent-server/.venv/bin/python "transformers>=4.56.0,<5"'
    if original.count(old_sync) != 1 or original.count(old_extra) != 1 or original.count('%files\n') != 1:
        raise ValueError('Official dependency installer changed')
    verifier = safe_path(Path(__file__).with_name('verify_bundle_install.py'))
    additions = (f"    {safe_path(bundle['wheel_directory'])} /agent-server/.s2-deps/wheels\n"
                 f"    {safe_path(bundle['requirements_path'])} /agent-server/.s2-deps/offline-final.requirements.txt\n"
                 f"    {safe_path(bundle['receipt_path'])} /agent-server/.s2-deps/prewarm-receipt.json\n"
                 f"    {verifier} /agent-server/.s2-deps/verify_bundle_install.py\n")
    # All final versions and artifact hashes were independently replayed. The
    # SDK project's broad constraints must not be injected as unhashed inputs.
    install = 'uv --no-config pip install --python /agent-server/.venv/bin/python --offline --no-index --find-links /agent-server/.s2-deps/wheels --require-hashes --no-deps -r /agent-server/.s2-deps/offline-final.requirements.txt'
    verify = '/agent-server/.venv/bin/python /agent-server/.s2-deps/verify_bundle_install.py --receipt /agent-server/.s2-deps/prewarm-receipt.json --sdk-lock /agent-server/uv.lock --output /agent-server/.s2-deps/installed-proof.json'
    adapted = original.replace('%files\n', '%files\n' + additions, 1).replace(old_sync, install, 1).replace(old_extra, verify, 1)
    # Only the two package installation commands and declared transport files.
    reverted = adapted.replace('%files\n' + additions, '%files\n', 1).replace(install, old_sync, 1).replace(verify, old_extra, 1)
    if reverted != original:
        raise AssertionError('Unexpected dependency recipe mutation')
    for section in ('%environment', '%runscript'):
        if adapted.split(section, 1)[1] != original.split(section, 1)[1]:
            raise AssertionError('Runtime environment/runscript changed')
    return adapted, {'adapter': 'hash-pinned-offline-wheel-transport-v1', 'original_definition_sha256': hashlib.sha256(original.encode()).hexdigest(), 'adapted_definition_sha256': hashlib.sha256(adapted.encode()).hexdigest(), 'changed_install_commands': [old_sync, old_extra], 'runtime_environment_and_runscript_byte_identical': True, 'sdk_source_and_lock_unchanged': True, 'receipt_file_sha256': bundle['receipt_file_sha256'], 'verifier_sha256': sha_file(verifier), 'post_lock_dependency_changes': bundle['post_lock_dependency_changes']}
