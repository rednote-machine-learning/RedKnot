"""One finite native pilot under the pinned, original GPU-holder supervisor.

Importing this module and --check-only are CPU-only: neither queries a GPU nor
imports Torch.  The main entry point is the only launch path.  It never sends a
signal, restores a holder itself, retries, or resumes an old experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REMOTE_ROOT = Path('/mnt/tidal-alsh01/dataset/redone/RedKnot_DeepInfer_20261028')
RUNTIME = Path('/tmp/redknot-dsv41-sm90.6ZbGzo')
HOLDER = Path('/tmp/redknot-213-holder.4Ptbyc')
SUPERVISOR = RUNTIME / 'deepinfer_plan/glm53_gpu_lifecycle.py'
HELPER = HOLDER / 'run_segpaged_experiment.py'
SUPERVISOR_SHA = '0a98c5a48d9f09e274be571df1c15b5988e09c8130ab516ed36436ff3690cafb'
HELPER_SHA = '468375fbc05b7c69077663e4faef65ec11ac79297add994d087534723bd0accd'
ENV_SHA = '6c2ecb354f77e653b5711898c84bbe684d930bde75c5660dc9925d900906ddd2'
LOCK = Path('/tmp/redknot-nonprefix-scale-supervisor.lock')
MODEL = Path('/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4.1-Flash-1017')
PROTOCOL_SCHEMA = 'kv-cache-engram-native-pilot-protocol-v1'
PLAN_SCHEMA = 'kv-cache-engram-native-pilot-launch-v1'
GPU_IDS = list(range(8))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value):
    require('sha256' not in value, 'cannot reseal an already sealed object')
    return dict(value, sha256=digest(value))


def absolute_no_symlink(path):
    path = Path(os.path.abspath(os.fspath(path)))
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            'symlink path is not permitted: ' + str(path))
    return path


def file_digest(path):
    path = absolute_no_symlink(path)
    require(path.is_file(), 'regular file required: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate JSON key: ' + key)
        value[key] = item
    return value


def read_json(path, limit=512 << 20):
    path = absolute_no_symlink(path)
    require(path.is_file() and path.stat().st_size <= limit,
            'missing or oversized JSON: ' + str(path))
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object,
                       parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    require(isinstance(value, dict), 'JSON object required')
    return value


def read_sealed(path, schema=None):
    value = read_json(path)
    require(isinstance(value.get('sha256'), str) and
            value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'canonical SHA256 seal mismatch: ' + str(path))
    require(isinstance(value.get('schema'), str) and bool(value['schema']), 'schema required')
    if schema is not None:
        require(value['schema'] == schema, 'unexpected protocol schema')
    return value


def source_files(stage):
    stage = absolute_no_symlink(stage)
    require(stage.is_dir(), 'stage directory missing')
    result = {}
    for path in sorted(stage.rglob('*')):
        require(not path.is_symlink(), 'stage contains a symlink: ' + str(path))
        if path.is_file() and path.suffix == '.py':
            result[str(path.relative_to(stage))] = file_digest(path)
    require('gpu_pilot/run_pilot.py' in result and 'gpu_pilot/launch_pilot.py' in result,
            'independent pilot runner and launcher required')
    require(any(name.startswith('kv_cache_engram/') for name in result),
            'KV Cache Engram source tree missing')
    return result


def evidence_paths(root, name):
    result = root / 'results'
    return dict(result_dir=result / name,
                plan=result / (name + '.launch-plan.json'),
                before=result / (name + '.holder-before.json'),
                started=result / (name + '.supervisor-started.json'),
                after=result / (name + '.holder-after.json'),
                outcome=result / (name + '.launch-outcome.json'),
                log=result / (name + '.supervisor.log'))


def require_unused(paths):
    for path in paths.values():
        absolute_no_symlink(path)
        require(not path.exists() and not path.is_symlink(),
                'run name already used; never retry/overwrite: ' + str(path))


def fixed_runner_args(root, manifest, protocol, sensitivity_protocol=None):
    arguments = ['--experiment-root', str(root), '--manifest', str(manifest), '--pilot-protocol', str(protocol),
            '--model-path', str(MODEL), '--repo', str(root / 'DeepInfer-PR148'),
            '--mode', 'quality', '--arm', 'recomputed', '--concurrency', '1',
            '--max-new-tokens', '64', '--repeat', '1', '--ep', '8', '--cp', '1',
            '--dp', '8', '--max-model-len', '270336', '--max-num-batched-tokens', '8192',
            '--max-num-seqs', '1', '--gpu-memory-utilization', '0.60',
            '--sampler-force-topk', '1024', '--batch-invariant', '1',
            '--prefill-attempt-interval', '1', '--seed', '0', '--skip-warmup']
    if sensitivity_protocol is not None:
        arguments.extend(['--sensitivity-protocol', str(sensitivity_protocol)])
    return arguments


def build_plan(root, stage_dir, name, manifest, protocol, timeout=21600, prepare_timeout=300,
               sensitivity_protocol=None):
    require(isinstance(name, str) and re.fullmatch(r'(?:kce|kv-cache-engram)-[a-z0-9][a-z0-9-]{1,100}', name),
            'unique kce- or kv-cache-engram- lowercase pilot name required')
    require(type(timeout) is int and 1 <= timeout <= 21600, 'finite timeout 1..21600 required')
    require(type(prepare_timeout) is int and 1 <= prepare_timeout <= 1800,
            'finite prepare timeout 1..1800 required')
    root, stage = absolute_no_symlink(root), absolute_no_symlink(stage_dir)
    require(stage.name == 'GatedEngram' and root in stage.parents,
            'pilot stage must be a separate GatedEngram tree within the task root')
    require(not any(p.name.startswith('sparse_query_') for p in stage.parents),
            'never stage the pilot inside a frozen sparse experiment')
    manifest, protocol = absolute_no_symlink(manifest), absolute_no_symlink(protocol)
    require(manifest != protocol, 'manifest and protocol must be separate files')
    require(manifest.is_relative_to(stage) if hasattr(manifest, 'is_relative_to') else stage in manifest.parents,
            'pilot manifest must be independent inside GatedEngram')
    require(protocol.is_relative_to(stage) if hasattr(protocol, 'is_relative_to') else stage in protocol.parents,
            'pilot protocol must be independent inside GatedEngram')
    protocol_data = read_sealed(protocol, PROTOCOL_SCHEMA)
    manifest_data = read_sealed(manifest)
    require(manifest_data.get('artifact_kind') == 'kv_cache_engram_native_pilot'
            and manifest_data.get('pilot_protocol_sha256') == protocol_data['sha256'],
            'independent pilot manifest must bind this exact protocol; never v8')
    sensitivity = None
    if sensitivity_protocol is not None:
        sensitivity_protocol = absolute_no_symlink(sensitivity_protocol)
        require(stage in sensitivity_protocol.parents and sensitivity_protocol not in (manifest, protocol),
                'independent sensitivity protocol must be inside the new stage')
        from gpu_pilot.sensitivity_protocol import sensitivity_protocol as declared_sensitivity
        value = read_sealed(sensitivity_protocol, 'kv-cache-engram-sensitivity-protocol-v1')
        require(value == declared_sensitivity()
                and manifest_data.get('sensitivity_protocol_sha256') == value['sha256'],
                'sensitivity protocol differs from tested rules or manifest')
        sensitivity = dict(path=str(sensitivity_protocol), sha256=value['sha256'],
                           file_sha256=file_digest(sensitivity_protocol))
    else:
        require('sensitivity_protocol_sha256' not in manifest_data,
                'sensitivity inputs require an explicit sensitivity protocol')
    paths = evidence_paths(root, name)
    require_unused(paths)
    sources = source_files(stage)
    args = fixed_runner_args(root, manifest, protocol, sensitivity_protocol)
    command = ['/root/miniconda3/bin/python', str(SUPERVISOR), '--python',
               str(RUNTIME / 'di-venv/bin/python'), '--holder-dir', str(HOLDER),
               '--holder-helper', str(HELPER), '--result-dir', str(paths['result_dir']),
               '--lock-file', str(LOCK), '--child-env-json', str(root / 'pr148-native-env.json'),
               '--timeout', str(timeout), '--prepare-timeout', str(prepare_timeout),
               '--gpus', '0,1,2,3,4,5,6,7', '--', str(stage / 'gpu_pilot/run_pilot.py'), *args]
    return seal(dict(schema=PLAN_SCHEMA, name=name, root=str(root), stage_dir=str(stage),
        created_unix=time.time(), no_automatic_retry=True, preserves_paused_experiments=True,
        gpu_ids=GPU_IDS, timeout=timeout, prepare_timeout=prepare_timeout,
        paths={k: str(v) for k, v in paths.items()}, argv=command,
        protocol=dict(path=str(protocol), sha256=protocol_data['sha256'], file_sha256=file_digest(protocol)),
        manifest=dict(path=str(manifest), sha256=manifest_data['sha256'], file_sha256=file_digest(manifest)),
        sensitivity_protocol=sensitivity,
        stage_sources=sources, stage_sources_sha256=digest(sources),
        pinned_lifecycle={str(SUPERVISOR): SUPERVISOR_SHA, str(HELPER): HELPER_SHA,
                          str(root / 'pr148-native-env.json'): ENV_SHA},
        supervisor_owns_cpu_release_and_holder=True,
        check_only_scope='build and validate immutable CPU launch plan; not GPU or quality qualification'))


def verify_plan_inputs(plan):
    require(plan.get('sha256') == digest({k: v for k, v in plan.items() if k != 'sha256'}), 'plan seal mismatch')
    require(plan.get('schema') == PLAN_SCHEMA, 'wrong plan schema')
    require(source_files(Path(plan['stage_dir'])) == plan['stage_sources'], 'stage changed after plan freeze')
    for name in ('manifest', 'protocol'):
        evidence = plan[name]
        require(file_digest(evidence['path']) == evidence['file_sha256'], name + ' changed after freeze')
        read_sealed(evidence['path'], PROTOCOL_SCHEMA if name == 'protocol' else None)
    if plan.get('sensitivity_protocol') is not None:
        item = plan['sensitivity_protocol']
        require(file_digest(item['path']) == item['file_sha256'], 'sensitivity protocol changed after freeze')
        require(read_sealed(item['path'])['sha256'] == item['sha256'], 'sensitivity protocol seal changed')


def verify_lifecycle(root):
    require(absolute_no_symlink(root) == REMOTE_ROOT, 'actual launch only on the authorized remote root')
    for path, expected in ((SUPERVISOR, SUPERVISOR_SHA), (HELPER, HELPER_SHA),
                           (Path(root) / 'pr148-native-env.json', ENV_SHA)):
        require(file_digest(path) == expected, 'original lifecycle identity changed: ' + str(path))
    require(not LOCK.is_symlink() and LOCK.parent.is_dir(), 'invalid original shared lock path')


def proc_identity(pid):
    require(type(pid) is int and pid > 1, 'invalid process identity')
    folder = Path('/proc') / str(pid)
    fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
    require(fields[0] not in ('Z', 'X', 'x'), 'dead process identity')
    return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
                start_ticks=int(fields[19]), argv=[os.fsdecode(x) for x in
                    (folder / 'cmdline').read_bytes().split(b'\0') if x])


def validate_physical_holder(metadata, gpus, contexts):
    workers = metadata.get('worker_pids')
    require(isinstance(workers, list) and len(workers) == 8 and
            all(type(pid) is int and pid > 1 for pid in workers) and len(set(workers)) == 8,
            'exactly eight unique holder workers required')
    require(metadata.get('gpu_ids') == GPU_IDS, 'holder metadata must cover physical GPUs 0..7')
    require(len(gpus) == 8 and sorted(x['index'] for x in gpus) == GPU_IDS, 'physical eight-GPU set mismatch')
    uuids = {x['uuid'] for x in gpus}
    require(len(uuids) == 8 and all(isinstance(x, str) and x.startswith('GPU-') for x in uuids),
            'duplicate or invalid physical GPU UUID')
    require(len(contexts) == 8 and {x['pid'] for x in contexts} == set(workers)
            and {x['uuid'] for x in contexts} == uuids, 'foreign/missing/duplicate GPU contexts')
    require(all(type(x['memory_mib']) is int and x['memory_mib'] > 0 for x in contexts),
            'holder context has no positive measured occupancy')


def holder_snapshot(root):
    """Read-only pinned original identity checks surrounding physical telemetry."""
    verify_lifecycle(root)
    modules = []
    for label, path in (('pilot_original_supervisor', SUPERVISOR), ('pilot_original_holder_helper', HELPER)):
        spec = importlib.util.spec_from_file_location(label, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    lifecycle, helper = modules
    metadata, foreign = lifecycle.read_holder(helper, HOLDER, GPU_IDS)
    require(not foreign, 'foreign GPU processes present; nothing will be stopped')
    identities = {str(pid): proc_identity(pid) for pid in [metadata['pid'], *metadata['worker_pids']]}
    parent = metadata['pid']
    require(metadata['pgid'] == parent and identities[str(parent)]['pgid'] == parent
            and identities[str(parent)]['sid'] == parent, 'holder parent session/group mismatch')
    require(all(identities[str(pid)]['ppid'] == parent
                and identities[str(pid)]['pgid'] == parent
                and identities[str(pid)]['sid'] == parent for pid in metadata['worker_pids']),
            'holder worker parent/session/group mismatch')
    def query(arguments):
        output = subprocess.check_output(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
                                         text=True, timeout=20)
        return list(csv.reader(io.StringIO(output)))
    gpus = [dict(index=int(row[0]), uuid=row[1].strip(), utilization=int(row[2]), memory_mib=int(row[3]))
            for row in query(['--query-gpu=index,uuid,utilization.gpu,memory.used'])]
    contexts = [dict(uuid=row[0].strip(), pid=int(row[1]), memory_mib=int(row[2]))
                for row in query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'])]
    validate_physical_holder(metadata, gpus, contexts)
    after, foreign_after = lifecycle.read_holder(helper, HOLDER, GPU_IDS)
    require(after == metadata and not foreign_after, 'holder changed during physical verification')
    require(identities == {str(pid): proc_identity(pid) for pid in [metadata['pid'], *metadata['worker_pids']]},
            'PID/start-time/ownership changed during physical verification')
    verify_lifecycle(root)
    return seal(dict(schema='kv-cache-engram-pilot-holder-audit-v1', verified_unix=time.time(),
                     read_only=True, metadata=metadata, process_identities=identities,
                     gpus=gpus, contexts=contexts))


def write_exclusive(path, value):
    """Atomic readable publication without overwriting a previous result."""
    path = absolute_no_symlink(path)
    require(path.parent.is_dir(), 'evidence parent does not exist')
    require(not path.exists() and not path.is_symlink(), 'evidence already exists')
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def verify_supervisor_result(value, plan):
    require(value.get('schema') == 'redknot-glm53-gpu-lifecycle-v1', 'unexpected supervisor result')
    require(value.get('holder_restored') is True and value.get('gpu_ids') == GPU_IDS,
            'supervisor did not declare complete original holder restoration')
    require(not value.get('error') and not value.get('cleanup_error') and not value.get('cleanup_warnings'),
            'supervisor reported execution or cleanup errors')
    require(value.get('worker_exit') == 0 and type(value.get('finished_unix')) in (int, float),
            'supervisor worker did not finish successfully')
    require(value.get('supervisor_sha256') == SUPERVISOR_SHA and value.get('holder_helper_sha256') == HELPER_SHA
            and value.get('child_environment_sha256') == ENV_SHA
            and value.get('runner_sha256') == plan['stage_sources']['gpu_pilot/run_pilot.py'],
            'supervisor source or environment evidence mismatch')


def run_plan(plan):
    """One synchronous original-supervisor invocation; never retry or signal."""
    verify_plan_inputs(plan)
    root = Path(plan['root'])
    verify_lifecycle(root)
    require(root / 'DeepInfer-PR148' not in Path(plan['stage_dir']).parents,
            'actual GPU pilot must use a new stage outside the frozen native repo')
    paths = {k: Path(v) for k, v in plan['paths'].items()}
    require_unused(paths)
    require((root / 'results').is_dir(), 'existing task results directory required')
    write_exclusive(paths['plan'], plan)
    outcome = dict(schema='kv-cache-engram-pilot-launch-outcome-v1', plan_sha256=plan['sha256'],
                   started_unix=time.time(), supervisor_started=False, holder_physically_verified=False,
                   successful=False, automatic_retry=False)
    try:
        before = holder_snapshot(root)
        write_exclusive(paths['before'], before)
        verify_plan_inputs(plan)
        with paths['log'].open('x') as log:
            child = subprocess.Popen(plan['argv'], stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            outcome['supervisor_started'] = True
            outcome['supervisor_pid'] = child.pid
            started = seal(dict(schema='kv-cache-engram-pilot-supervisor-started-v1',
                                pid=child.pid, plan_sha256=plan['sha256'], started_unix=time.time(),
                                argv=plan['argv']))
            write_exclusive(paths['started'], started)
            print(json.dumps(dict(supervisor_pid=child.pid, log=str(paths['log']),
                                  result_dir=str(paths['result_dir']))), flush=True)
            # The original supervisor owns finite prepare/run timers and cleanup.
            # Do not add an outer timeout which would kill the restorer itself.
            outcome['supervisor_exit'] = child.wait()
        require(outcome['supervisor_exit'] == 0, 'original supervisor returned nonzero')
        verify_supervisor_result(read_json(paths['result_dir'] / 'supervisor.json'), plan)
        verify_plan_inputs(plan)
        outcome['successful'] = True
    except BaseException as exc:
        outcome['error'] = repr(exc)
        outcome['successful'] = False
    finally:
        try:
            after = holder_snapshot(root)
            write_exclusive(paths['after'], after)
            outcome['holder_physically_verified'] = True
            outcome['holder_audit_sha256'] = after['sha256']
        except BaseException as exc:
            outcome['holder_verification_error'] = repr(exc)
            outcome['successful'] = False
        outcome['finished_unix'] = time.time()
        write_exclusive(paths['outcome'], seal(outcome))
    return 0 if outcome['successful'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--root', type=Path, default=REMOTE_ROOT)
    parser.add_argument('--stage-dir', type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--pilot-protocol', required=True, type=Path)
    parser.add_argument('--sensitivity-protocol', type=Path)
    parser.add_argument('--timeout', type=int, default=21600)
    parser.add_argument('--prepare-timeout', type=int, default=300)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    stage = args.stage_dir or args.root / 'DeepInfer-PR148/GatedEngram'
    plan = build_plan(args.root, stage, args.name, args.manifest, args.pilot_protocol,
                      args.timeout, args.prepare_timeout, sensitivity_protocol=args.sensitivity_protocol)
    if args.check_only:
        print(json.dumps(dict(status='CPU launch plan validated; no launch or GPU query',
                              gpu_initialized=False, lifecycle_identity_checked=False, plan=plan), ensure_ascii=False))
        return 0
    return run_plan(plan)


if __name__ == '__main__':
    raise SystemExit(main())
