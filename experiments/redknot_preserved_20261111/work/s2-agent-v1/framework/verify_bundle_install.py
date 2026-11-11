"""Run inside the real image after offline installation; no model imports."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import subprocess


def installed():
    packages = {}
    for dist in importlib.metadata.distributions():
        name = re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()
        if name in packages:
            raise ValueError('Duplicate installed distribution: ' + name)
        packages[name] = dist.version
    return dict(sorted(packages.items()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--receipt', required=True); p.add_argument('--sdk-lock', required=True); p.add_argument('--output', required=True); a = p.parse_args()
    receipt = json.loads(Path(a.receipt).read_text())
    body = dict(receipt); seal = body.pop('sha256')
    if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != seal:
        raise ValueError('Receipt seal mismatch')
    packages = installed()
    if packages != receipt['final_installed']['packages']:
        raise ValueError('Actual SIF package set differs from admitted offline replay')
    lock_sha = hashlib.sha256(Path(a.sdk_lock).read_bytes()).hexdigest()
    if lock_sha != receipt['source']['uv_lock_sha256']:
        raise ValueError('Actual SIF SDK lock differs')
    if sys.version_info[:3] != (3, 13, 18) or platform.machine() != 'x86_64' or sys.platform != 'linux':
        raise ValueError('Actual SIF Python/platform differs')
    check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=30)
    if check.returncode!=0:raise ValueError('Actual SIF dependency check failed: '+check.stdout+check.stderr)
    report = {'schema': 's2-sdk313-container-installed-v1', 'status': 'passed', 'packages': packages, 'python': sys.version, 'executable': sys.executable, 'uv_lock_sha256': lock_sha, 'prewarm_receipt_seal': seal, 'dependency_check':{'passed':True,'stdout':check.stdout,'stderr':check.stderr}, 'gpu': False, 'model_calls': 0}
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'package_count': len(packages), 'proof': a.output}))


if __name__ == '__main__':
    main()
