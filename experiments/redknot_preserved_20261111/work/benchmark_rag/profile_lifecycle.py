#!/usr/bin/env python3
"""Pinned wrapper for the existing holder-restoring GPU supervisor.

This file never runs inference or starts/stops a holder itself. The unchanged
glm53_gpu_lifecycle.py owns the shared lock, CPU-ready barrier, holder release,
7200-second timeout, authenticated cleanup and restoration. This is NOT an
independent watchdog; SIGKILL/host loss cannot be handled by that supervisor.
All imports in this wrapper are standard-library and do not initialize CUDA.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HOLDER_PYTHON = '/root/miniconda3/bin/python'
SHARED_LOCK = '/tmp/redknot-nonprefix-scale-supervisor.lock'
TIMEOUT = 7200
PREPARE_TIMEOUT = 600
GPU_IDS = list(range(8))
ORIGINAL_PINS = {
    'glm53_gpu_lifecycle.py': '0a98c5a48d9f09e274be571df1c15b5988e09c8130ab516ed36436ff3690cafb',
    'run_segpaged_experiment.py': '468375fbc05b7c69077663e4faef65ec11ac79297add994d087534723bd0accd',
    'gpu_hold_target.py': 'bc7469e90e85d868083c960d1d845a84c76b879bfa83dcec6fde9885e447db53',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    require('sha256' not in value, 'cannot reseal an existing artifact')
    return dict(value, sha256=digest(value))


def file_sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular non-symlink file required: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, sealed=False):
    value = json.loads(Path(path).read_bytes())
    if sealed:
        require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'invalid seal: ' + str(path))
    return value


def save(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'refusing to overwrite evidence: ' + str(path))
    with path.open('x', encoding='utf-8') as stream:
        json.dump(seal(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def reference(path):
    path = Path(path)
    value = read_json(path)
    result = dict(path=str(path.resolve()), file_sha256=file_sha(path))
    if isinstance(value, dict) and 'sha256' in value:
        read_json(path, sealed=True)
        result['seal_sha256'] = value['sha256']
    return result


def clean_holder_environment(environ=None):
    """Never propagate the model's CUDA/Python compatibility overrides."""
    source = os.environ if environ is None else environ
    keep = ('HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TZ', 'TMPDIR')
    env = {k: source[k] for k in keep if k in source}
    env.update(PATH='/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', GIT_OPTIONAL_LOCKS='0')
    return env


def proc_identity(pid):
    require(type(pid) is int and pid > 1, 'invalid process PID')
    directory = Path('/proc') / str(pid)
    a = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
    argv = [x.decode() for x in (directory / 'cmdline').read_bytes().split(b'\0') if x]
    b = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
    require(a[1:4] + [a[19]] == b[1:4] + [b[19]] and b[0] not in ('Z', 'X', 'x'),
        'unstable or exited process identity')
    return dict(pid=pid, ppid=int(b[1]), pgid=int(b[2]), sid=int(b[3]), start_ticks=int(b[19]), argv=argv)


def signal_exact(expected, signum):
    """Only forward interruption to the exact original supervisor, via pidfd."""
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'pidfd API required')
    try:
        fd = os.pidfd_open(expected['pid'])
    except ProcessLookupError:
        return False
    try:
        require(proc_identity(expected['pid']) == expected, 'supervisor PID/argv/starttime changed')
        signal.pidfd_send_signal(fd, signum)
        return True
    finally:
        os.close(fd)


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
            text=True, timeout=20)
    raw_gpu = query(['--query-gpu=index,uuid,utilization.gpu,memory.used'])
    raw_apps = query(['--query-compute-apps=pid,gpu_uuid,used_gpu_memory'])
    gpus = []
    for row in raw_gpu.splitlines():
        i, uuid, util, memory = [s.strip() for s in row.split(',')]
        gpus.append(dict(index=int(i), uuid=uuid, utilization=int(util), memory_mib=int(memory)))
    apps = []
    for row in raw_apps.splitlines():
        pid, uuid, memory = [s.strip() for s in row.split(',')]
        apps.append(dict(pid=int(pid), uuid=uuid, memory_mib=int(memory)))
    return dict(observed_unix=time.time(), gpus=gpus, compute_apps=apps)


def load_original(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_holder_capture(capture, directory, expected=None):
    metadata = capture['metadata']; identities = capture['process_identities']; snapshot = capture['snapshot']
    parent = metadata['pid']; workers = metadata['worker_pids']; directory = Path(directory)
    require(metadata['gpu_ids'] == GPU_IDS and len(workers) == 8 and len(set(workers)) == 8,
        'holder does not own exactly eight workers')
    require(metadata['target_util_percent'] == 80 and metadata['variation_amplitude_percent'] == 6
        and metadata['target_range_percent'] == [74, 86], 'original 80±6 holder policy changed')
    require(metadata['script'] == str(directory / 'gpu_hold_target.py'), 'wrong holder script')
    require(set(identities) == {str(x) for x in [parent, *workers]}, 'holder identity inventory differs')
    identity = identities[str(parent)]
    require(identity['pid'] == identity['pgid'] == identity['sid'] == metadata['pgid'] == parent,
        'holder parent is not its own authenticated process/session group')
    require(identity['argv'][0] == HOLDER_PYTHON and metadata['script'] in identity['argv']
        and '--run' in identity['argv'], 'holder interpreter/argv differs')
    for pid in workers:
        row = identities[str(pid)]
        require(row['pid'] == pid and row['ppid'] == parent and row['pgid'] == parent
            and row['sid'] == parent and row['start_ticks'] >= identity['start_ticks'],
            'holder worker ownership differs')
    gpus = snapshot['gpus']; apps = snapshot['compute_apps']
    require([row['index'] for row in gpus] == GPU_IDS and len({row['uuid'] for row in gpus}) == 8,
        'eight GPU UUID inventory differs')
    require(len(apps) == 8 and {x['pid'] for x in apps} == set(workers)
        and {x['uuid'] for x in apps} == {x['uuid'] for x in gpus}
        and all(x['memory_mib'] > 0 for x in apps), 'holder contexts missing, duplicated or foreign')
    if expected is not None:
        require(metadata == expected['metadata'] and identities == expected['process_identities'],
            'holder PID/starttime/argv/metadata changed since preparation')
        require({x['index']: x['uuid'] for x in gpus} ==
            {x['index']: x['uuid'] for x in expected['snapshot']['gpus']}, 'GPU UUID changed')
        require({(x['pid'], x['uuid']) for x in apps} ==
            {(x['pid'], x['uuid']) for x in expected['snapshot']['compute_apps']}, 'holder GPU ownership changed')
    return True


def capture_holder(config, expected=None):
    original = load_original(config['original_supervisor'], 'profile_original_lifecycle')
    helper = load_original(config['holder_helper'], 'profile_original_holder_helper')
    metadata, foreign = original.read_holder(helper, Path(config['holder_dir']), GPU_IDS)
    require(not foreign, 'foreign GPU application present; nothing stopped')
    identities = {str(pid): proc_identity(pid) for pid in [metadata['pid'], *metadata['worker_pids']]}
    snapshot = gpu_snapshot()
    require(read_json(Path(config['holder_dir']) / 'holder.json') == metadata,
        'holder metadata changed during snapshot')
    capture = dict(metadata=metadata, process_identities=identities, snapshot=snapshot)
    validate_holder_capture(capture, config['holder_dir'], expected)
    return capture


def validate_config(config, *, check_files=True):
    require(config.get('schema') == 's12-system-profile-lifecycle-config-v1'
        and config.get('sha256') == digest({k: v for k, v in config.items() if k != 'sha256'}), 'invalid lifecycle config')
    require(config['timeout_seconds'] == TIMEOUT and config['prepare_timeout_seconds'] == PREPARE_TIMEOUT,
        'fixed 7200-second/600-second budgets required')
    require(config['holder_python'] == HOLDER_PYTHON and config['lock_file'] == SHARED_LOCK,
        'original Python/shared lifecycle lock required')
    require(config['holder_policy'] == 'restore_original_holder_80_plusminus6_on_exit', 'holder restoration not authorized')
    require(config['independent_watchdog'] is False and config['sigkill_or_host_loss_recovery_claim'] is False,
        'cannot claim an unimplemented independent watchdog')
    command = config['runner_argv']
    require(type(command) is list and command and all(type(x) is str and x and '\0' not in x for x in command),
        'runner argv must be an explicit nonempty list')
    reserved = ('--output-dir', '--ready-file', '--release-file', '--out-dir', '--cpu-ready-file')
    require(not any(x.split('=', 1)[0] in reserved for x in command[1:]), 'supervisor owns barrier/output paths')
    require(Path(config['holder_helper']).parent == Path(config['holder_dir'])
        and Path(config['holder_script']) == Path(config['holder_dir']) / 'gpu_hold_target.py', 'holder sources detached')
    required = [config['original_supervisor'], config['holder_helper'], config['holder_script'],
        config['child_env_json'], command[0], str(Path(__file__).resolve())]
    require(set(required) <= set(config['source_sha256']), 'essential source pins missing')
    require(config['admission_refs'], 'CPU admission evidence required before release')
    for ref in config['admission_refs']:
        require(ref.get('required_fields') and type(ref['required_fields']) is dict, 'explicit admission fields required')
        if check_files:
            require(file_sha(ref['path']) == ref['file_sha256'], 'admission bytes changed')
            value = read_json(ref['path'], sealed=True)
            require(value['sha256'] == ref['seal_sha256'] and all(value.get(k) == v for k, v in ref['required_fields'].items()),
                'CPU admission not passed or detached')
    if check_files:
        for path, sha in config['source_sha256'].items():
            require(file_sha(path) == sha, 'pinned source changed: ' + path)
        for key in ('original_supervisor', 'holder_helper', 'holder_script'):
            path = Path(config[key]);require(file_sha(path) == ORIGINAL_PINS[path.name], 'original lifecycle source differs')
        require(Path(config['child_python']).is_file() and Path(config['cwd']).is_dir(), 'runtime/cwd missing')
        env = read_json(config['child_env_json'])
        require(type(env) is dict and all(type(k) is str and type(v) is str and k and '=' not in k and '\0' not in k + v for k,v in env.items()), 'invalid model environment')
        require('REDKNOT_GLM_RUN_ID' not in env, 'ownership marker is supervisor-reserved')
    validate_holder_capture(config['expected_holder'], config['holder_dir'])
    return True


def prepare_config(settings):
    config = dict(settings)
    for key in ('sha256', 'expected_holder'):
        require(key not in config, 'prepare obtains fresh evidence, not supplied holder identity')
    config.update(schema='s12-system-profile-lifecycle-config-v1', timeout_seconds=TIMEOUT,
        prepare_timeout_seconds=PREPARE_TIMEOUT, holder_python=HOLDER_PYTHON, lock_file=SHARED_LOCK,
        holder_policy='restore_original_holder_80_plusminus6_on_exit', independent_watchdog=False,
        sigkill_or_host_loss_recovery_claim=False, prepared_unix=time.time())
    config['source_sha256'] = dict(config.get('source_sha256', {}))
    for path in (config['original_supervisor'], config['holder_helper'], config['holder_script'],
                 config['child_env_json'], config['runner_argv'][0], str(Path(__file__).resolve())):
        actual = file_sha(path)
        require(path not in config['source_sha256'] or config['source_sha256'][path] == actual, 'supplied source pin changed')
        config['source_sha256'][path] = actual
    for key in ('original_supervisor','holder_helper','holder_script'):
        require(config['source_sha256'][config[key]] == ORIGINAL_PINS[Path(config[key]).name], 'original source pin mismatch')
    config['expected_holder'] = capture_holder(config)
    config = seal(config);validate_config(config)
    require(not Path(config['output_dir']).exists(), 'new output directory required')
    return config


def supervisor_command(config):
    return [HOLDER_PYTHON, '-B', '-u', config['original_supervisor'], '--python', config['child_python'],
        '--holder-dir', config['holder_dir'], '--holder-helper', config['holder_helper'],
        '--result-dir', str(supervisor_directory(config)), '--lock-file', SHARED_LOCK,
        '--child-env-json', config['child_env_json'], '--timeout', '7200', '--prepare-timeout', '600',
        '--gpus', '0,1,2,3,4,5,6,7', '--', *config['runner_argv']]


def supervisor_directory(config):
    return Path(config.get('supervisor_result_dir', str(Path(config['output_dir']) / 'OriginalSupervisor')))


def verify_restored(config, *, sleep=time.sleep, clock=time.monotonic):
    """Observe an already restored holder; this function never starts one."""
    capture = capture_holder(config)
    require({x['index']:x['uuid'] for x in capture['snapshot']['gpus']} ==
        {x['index']:x['uuid'] for x in config['expected_holder']['snapshot']['gpus']}, 'restored device UUID changed')
    warm=[];deadline=clock()+60
    while True:
        sample=capture_holder(config,capture)['snapshot'];warm.append(sample)
        if all(70<=x['utilization']<=90 for x in sample['gpus']):break
        require(clock()<deadline,'restored holder did not enter 70–90% band');sleep(5)
    samples=[];started=clock()
    for ordinal in range(37):
        sleep(max(0,started+ordinal*5-clock()))
        samples.append(capture_holder(config,capture)['snapshot'])
    summary={str(i):dict(min=min(s['gpus'][i]['utilization'] for s in samples),
        max=max(s['gpus'][i]['utilization'] for s in samples),
        in_band=sum(70<=s['gpus'][i]['utilization']<=90 for s in samples),samples=37) for i in GPU_IDS}
    return dict(schema='s12-original-holder-live-verification-v1', status='passed' if all(x['in_band']==37 for x in summary.values()) else 'outside_band',
        config_sha256=config['sha256'],read_only=True,holder=capture,stabilization_samples=warm,
        samples=samples,summary=summary,elapsed_s=samples[-1]['observed_unix']-samples[0]['observed_unix'],
        foreign_contexts_present=False)


def run(config):
    validate_config(config)
    require(os.path.realpath(sys.executable)==os.path.realpath(HOLDER_PYTHON), 'run wrapper with original holder Python')
    require(not Path(SHARED_LOCK).is_symlink(), 'shared lock must not be a symlink')
    output=Path(config['output_dir']);require(not output.exists() and not output.is_symlink(),'output already exists')
    require(not supervisor_directory(config).exists() and not supervisor_directory(config).is_symlink(),
        'original supervisor result path must be new')
    require(supervisor_directory(config) != output,'wrapper and original supervisor outputs must differ')
    # The original supervisor owns the shared lock during inference. This short
    # pre-check lock only excludes a competing owner while recording fresh state.
    with open(SHARED_LOCK,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=capture_holder(config,config['expected_holder'])
        output.mkdir(parents=True,exist_ok=False)
        save(output/'before.json',dict(schema='s12-profile-holder-before-v1',config_sha256=config['sha256'],holder=before))
    command=supervisor_command(config);child=None;identity=None;interrupted=[];error=None;released=False
    previous={}
    def forward(signum,frame):
        interrupted.append(dict(signal=signum,time_unix=time.time()))
        if identity is not None:signal_exact(identity,signal.SIGTERM)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,forward)
        with (output/'supervisor.log').open('x') as log:
            child=subprocess.Popen(command,cwd=config['cwd'],env=clean_holder_environment(),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=proc_identity(child.pid)
            require(identity['argv']==command and identity['pid']==identity['pgid']==identity['sid'],'supervisor argv/session differs')
            save(output/'supervisor-started.json',dict(schema='s12-original-supervisor-started-v1',identity=identity,
                command=command,config_sha256=config['sha256'],timeout_seconds=7200,independent_watchdog=False))
            if interrupted:signal_exact(identity,signal.SIGTERM)
            while child.poll() is None:
                release=supervisor_directory(config)/'release.json'
                if release.exists() and not released:
                    item=read_json(release);state=read_json(supervisor_directory(config)/'supervisor.json')
                    require(item['pid']==state['child_pid'],'original release/child mismatch')
                    require(proc_identity(identity['pid'])==identity,'supervisor identity changed during run')
                    save(output/'release-observed.json',dict(schema='s12-original-release-observed-v1',config_sha256=config['sha256'],
                        release=reference(release),released_unix=item['released_unix'],configured_timeout_seconds=7200,
                        nominal_deadline_unix=item['released_unix']+7200,
                        deadline_clock='original_supervisor_monotonic_loop_after_release_write',
                        loop_poll_interval_s=.5,independent_watchdog=False,supervisor=identity))
                    released=True
                time.sleep(.5)
    except BaseException as exc:
        error=repr(exc)
        if child is not None and child.poll() is None and identity is not None:
            try:signal_exact(identity,signal.SIGTERM);child.wait(timeout=150)
            except BaseException as cleanup: error+='; original supervisor cleanup pending: '+repr(cleanup)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    result=dict(schema='s12-profile-lifecycle-terminal-v1',config_sha256=config['sha256'],status='failed',
        original_supervisor_argv=command,original_supervisor_identity=identity,
        original_supervisor_exit=None if child is None else child.poll(),interruption_requests=interrupted,
        error=error,independent_watchdog=False,sigkill_or_host_loss_recovery_claim=False,
        holder_restored=False,holder_live_verification_passed=False,finished_unix=time.time())
    try:
        require(child is not None and child.poll() is not None,'original supervisor remains responsible for cleanup')
        state_path=supervisor_directory(config)/'supervisor.json';state=read_json(state_path)
        result['original_supervisor_state']=state;result['original_supervisor_reference']=reference(state_path)
        require(state.get('holder_restored') is True and not state.get('cleanup_error'),'original restoration did not pass')
        require(state['holder_helper_sha256']==ORIGINAL_PINS['run_segpaged_experiment.py']
            and state['supervisor_sha256']==ORIGINAL_PINS['glm53_gpu_lifecycle.py'], 'executed original supervisor sources differ')
        # Acquire only after the original process exits: terminal-file creation
        # alone is not evidence that its cleanup owner has relinquished the lock.
        with open(SHARED_LOCK,'a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            validate_config(config)
            verification=verify_restored(config)
            save(output/'holder-live-verification.json',verification)
        result.update(holder_restored=True,holder_live_verification_passed=verification['status']=='passed',
            holder_live_verification=reference(output/'holder-live-verification.json'))
        require(result['holder_live_verification_passed'],'holder restored but 180s utilization proof failed')
        result['status']='complete' if child.returncode==0 and error is None and not interrupted else 'failed_with_holder_restored'
    except BaseException as exc:
        result['restoration_verification_error']=repr(exc)
    result['finished_unix']=time.time();save(output/'terminal.json',result)
    print(json.dumps({k:result[k] for k in ('status','holder_restored','holder_live_verification_passed','original_supervisor_exit')}),flush=True)
    return 0 if result['status']=='complete' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-config',type=Path);group.add_argument('--config',type=Path)
    parser.add_argument('--output-config',type=Path);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    if args.prepare_config:
        require(args.output_config is not None and not args.check_only,'prepare requires --output-config')
        config=prepare_config(read_json(args.prepare_config));save(args.output_config,{k:v for k,v in config.items() if k!='sha256'})
        print(json.dumps(dict(status='prepared_no_gpu_release',config_sha256=config['sha256'],expected_holder_pid=config['expected_holder']['metadata']['pid'])));return 0
    config=read_json(args.config,sealed=True);validate_config(config)
    if args.check_only:
        print(json.dumps(dict(status='lifecycle_cpu_check_passed',config_sha256=config['sha256'],
            command=supervisor_command(config),gpu_released=False,independent_watchdog=False)));return 0
    return run(config)


if __name__=='__main__':raise SystemExit(main())
