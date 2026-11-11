"""Detached CPU postrun for D-Spark three-arm task; no SSH, model call, or GPU action.

Derives actual directories from sealed configs. Actual GPU cleanup with holder OFF is required before official grading.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import finalize_first_task as finalizer

D = Path(__file__).resolve().parent
HELPER_SHA = '9015b385121061e8bb5b95d60716a3e367e4d6f8e72a73e49ee6caa699e7ed57'
PY = '/tmp/redknot-dsv41-sm90.6ZbGzo/di-venv/bin/python'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triplet-config', type=Path, required=True)
    parser.add_argument('--lifecycle-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--registry-manifest', type=Path)
    args = parser.parse_args(argv)
    state = dict(schema='s2-first-task-cpu-postrun-v3', status='waiting_for_gpu_lifecycle',
        started_unix=time.time(), gpu_actions=False, model_calls=0, steps=[], new_round_timer=False,
        holder_policy='OFF', waiting_deadline_unix=time.time()+9000)
    # Exclusive ownership prevents duplicate workers/official grading attempts.
    with (D / 'owner.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), argv=sys.argv, started_unix=state['started_unix'],
            worker_sha256=finalizer.sha(__file__), helper_sha256=HELPER_SHA), stream, indent=2)

    def publish():
        state['observed_unix'] = time.time()
        value = dict(state, sha256=finalizer.digest(state))
        temporary = D / 'status.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(temporary, D / 'status.json')

    def call(name, command, cwd, environment, timeout):
        log = D / (name + '.log')
        row = dict(name=name, argv=command, started_unix=time.time(), log=str(log))
        state['steps'].append(row); publish()
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            stat = Path('/proc', str(process.pid), 'stat').read_text().rsplit(') ', 1)[1].split()
            row.update(pid=process.pid, start_ticks=int(stat[19]), pgid=process.pid); publish()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # This is a fresh CPU grader child/session, never an inference
                # group or holder. Popen owns the unreaped leader PID throughout.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
                row.update(status='timeout', returncode=process.returncode, ended_unix=time.time()); publish()
                return False
        row.update(returncode=code, status='passed' if code == 0 else 'failed', ended_unix=time.time()); publish()
        return code == 0

    try:
        finalizer.require(finalizer.sha(D / 'finalize_first_task.py') == HELPER_SHA, 'postrun helper source changed')
        state['status'] = 'waiting_for_actual_configs'; publish()
        while not (args.lifecycle_config.is_file() and args.triplet_config.is_file()):
            if time.time() > state['waiting_deadline_unix']:
                raise TimeoutError('Actual B3 configs absent after original deadline allowance')
            time.sleep(5)
        lifecycle = finalizer.read(args.lifecycle_config)
        triplet = finalizer.read(args.triplet_config)
        finalizer.require(lifecycle['holder_policy']=='OFF' and lifecycle['timeout_seconds']==7200, 'Wrong GPU/holder policy')
        lifecycle_dir = Path(lifecycle['output_dir'])
        supervisor_dir = lifecycle_dir
        final = args.output_dir.resolve()
        state.update(lifecycle_config=finalizer.ref(args.lifecycle_config), triplet_config=finalizer.ref(args.triplet_config),
                     lifecycle_directory=str(lifecycle_dir), supervisor_directory=str(supervisor_dir), output_dir=str(final),
                     status='waiting_for_gpu_lifecycle')
        publish()
        while not (lifecycle_dir / 'terminal.json').exists():
            release = supervisor_dir / 'release.json'
            if release.exists() and 'retry_released_unix' not in state:
                value = finalizer.read(release, sealed=False)
                state.update(retry_released_unix=value['released_unix'], retry_release=finalizer.ref(release),waiting_deadline_unix=value['deadline_unix']+180); publish()
            if time.time() > state['waiting_deadline_unix']:
                raise TimeoutError('Lifecycle terminal absent after original cap+restoration allowance; no GPU/holder mutation')
            time.sleep(5)
        state['status'] = 'preparing_official_grading'; publish()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        prepare = [PY, '-B', str(D / 'finalize_first_task.py'), '--triplet-config', str(args.triplet_config.resolve()),
                   '--lifecycle-config', str(args.lifecycle_config.resolve()), '--output-dir', str(final)]
        if args.registry_manifest:
            prepare += ['--registry-manifest', str(args.registry_manifest.resolve())]
        if not call('prepare', prepare, str(D), env, 180):
            raise RuntimeError('D-Spark terminal/idle/patch admission failed; see prepare.log')
        plan = finalizer.read(final / 'postrun-plan.json')
        state['status'] = 'official_cpu_grading'; publish()
        accepted = []
        for command in plan['grading_commands']:
            arm = command['arm']
            ok = call('grade-' + arm, command['argv'], command['cwd'], dict(env, **command['environment']), 3900)
            row = next(a for a in plan['arms'] if a['arm'] == arm)
            report = Path(row['expected_official_report'])
            receipt = report.parent / 'offline-grader-receipt.json'
            if ok and report.exists() and receipt.exists() and finalizer.read(receipt).get('status') == 'official_evaluation_returned':
                accepted += ['--grade', arm + '=' + str(report)]
        reader = plan['reader_command']
        reader = reader[:reader.index('--grade')] if '--grade' in reader else reader
        state.update(status='reading_results', graded_arms=[accepted[i].split('=', 1)[0] for i in range(1, len(accepted), 2)]); publish()
        if not call('readout', reader + accepted, triplet['cwd'], env, 180):
            raise RuntimeError('Unchanged independent readout rejected raw evidence; see readout.log')
        results = finalizer.read(final / 'triplet-results.json')
        state.update(status='complete', result_path=str(final / 'triplet-results.json'),
            official_grading_attempted=True, missing_grades_preserved_as_null=True,
            raw_triplet_status=plan['raw_triplet_status'])
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
    state['finished_unix'] = time.time(); publish()
    print(json.dumps(state)); return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
