#!/usr/bin/env python3
"""Explicit one-command Full / S1.2 reproduction on the pinned H200 runtime.

No action occurs on import or without --check-only/--run. --check-only runs CPU
preflight without querying/releasing GPUs. --run is a future explicit request:
it performs fresh CPU preflight, uses the original holder-restoring supervisor,
then runs the independent CPU result audit after verified restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
DEFINITION_SCHEMA='benchmark-rag-redknot-definition-v1'
TIMEOUT_SECONDS=7200
HOLDER_PYTHON='/root/miniconda3/bin/python'


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seal(value):
    require('sha256' not in value,'already sealed')
    return dict(value,sha256=digest(value))


def file_sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular source file required: '+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sealed(path):
    value=json.loads(Path(path).read_bytes())
    require(type(value) is dict and value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),
        'invalid artifact seal: '+str(path))
    return value


def write_sealed(path,body):
    value=seal(body)
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return value


def reference(path):
    path=Path(path).resolve();value=read_sealed(path)
    return dict(path=str(path),file_sha256=file_sha(path),seal_sha256=value['sha256'])


def referenced(ref):
    require(type(ref) is dict and set(ref)=={'path','file_sha256','seal_sha256'},'complete source reference required')
    value=read_sealed(ref['path'])
    require(file_sha(ref['path'])==ref['file_sha256'] and value['sha256']==ref['seal_sha256'],
        'referenced proof changed')
    return value


def definition_path(value):
    path=Path(value)
    return path if path.is_absolute() else HERE/path


def load_definition(path):
    value=read_sealed(path)
    require(value.get('schema')==DEFINITION_SCHEMA and value.get('timeout_seconds')==TIMEOUT_SECONDS
        and value.get('holder_python')==HOLDER_PYTHON,'pinned runtime/holder timeout definition differs')
    required=('experiment_root','model_path','manifest','pilot_protocol','child_python','child_env_json',
        'original_supervisor','holder_dir','holder_helper','holder_script')
    require(all(type(value.get(key)) is str and value[key] for key in required),'runtime definition path missing')
    pins=value.get('package_source_sha256')
    require(type(pins) is dict and {'benchmark_rag_redknot.py','run_s12_reproduction.py',
        'reproduction_contract.py','read_s12_reproduction_results.py','profile_lifecycle.py',
        'idle_optimizer.py','profile_group_contract.py'}<=set(pins),'reproduction source pins missing')
    for name,expected in pins.items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'source pin escapes package')
        require(file_sha(HERE/relative)==expected,'package source changed: '+name)
    return value


def native_argv(definition,config_path):
    """The same pinned native bootstrap used by accepted Full / S1.2 runs."""
    paths=lambda key:str(definition_path(definition[key]).resolve())
    return [paths('child_python'),'-B',str(HERE/'run_s12_reproduction.py'),
        '--experiment-root',paths('experiment_root'),'--manifest',paths('manifest'),
        '--pilot-protocol',paths('pilot_protocol'),'--model-path',paths('model_path'),
        '--repo',str(definition_path(definition.get('native_snapshot','NativeSnapshot')).resolve()),
        '--mode','quality','--arm','recomputed','--concurrency','1','--max-new-tokens','64','--repeat','1',
        '--ep','8','--cp','1','--dp','8','--max-model-len','270336','--max-num-batched-tokens','8192',
        '--max-num-seqs','1','--gpu-memory-utilization','0.60','--sampler-force-topk','1024',
        '--batch-invariant','1','--prefill-attempt-interval','1','--seed','0','--skip-warmup',
        '--s12-config',str(config_path)]


def cpu_preflight(definition,definition_file,args,run_dir):
    import run_s12_reproduction as runner
    config=runner.build_config(definition_path(definition['manifest']),definition_path(definition['pilot_protocol']),
        length=args.length,cases_per_length=args.cases)
    config_path=run_dir/'reproduction-config.json'
    with config_path.open('x') as stream:json.dump(config,stream,indent=2,ensure_ascii=False);stream.write('\n')
    env=json.loads(definition_path(definition['child_env_json']).read_bytes())
    require(type(env) is dict and all(type(k) is str and type(v) is str for k,v in env.items()),'native environment invalid')
    require('REDKNOT_GLM_RUN_ID' not in env,'old process-ownership marker is forbidden')
    env['PYTHONPATH']=str(definition_path(definition.get('native_snapshot','NativeSnapshot')).resolve())
    env_path=run_dir/'child-env.json'
    with env_path.open('x') as stream:json.dump(env,stream,indent=2);stream.write('\n')
    actual_env=os.environ.copy();actual_env.update(env)
    argv=native_argv(definition,config_path)
    stdout=run_dir/'cpu-preflight.stdout';stderr=run_dir/'cpu-preflight.stderr'
    with stdout.open('x') as out,stderr.open('x') as err:
        result=subprocess.run([*argv,'--output-dir',str(run_dir/'CPUPreflightUnused'),'--check-only'],
            cwd=HERE,env=actual_env,stdout=out,stderr=err,timeout=600)
    require(result.returncode==0,'CPU preflight failed; inspect '+str(stderr))
    parsed=json.loads(stdout.read_text().strip().splitlines()[-1])
    require(parsed.get('status')=='cpu_preflight_passed' and parsed.get('gpu_initialized') is False,
        'actual CPU preflight did not pass')
    proof=run_dir/'cpu-preflight.json'
    write_sealed(proof,dict(schema='benchmark-rag-actual-cpu-preflight-v1',status='cpu_preflight_passed',
        gpu_initialized=False,preflight=parsed['preflight'],definition_ref=reference(definition_file),
        stdout_sha256=file_sha(stdout),runner_argv=argv,created_unix=time.time()))
    return dict(config=config,config_path=config_path,argv=argv,env_path=env_path,proof=proof)


def admitted_package_cpu_proof(definition):
    require('package_cpu_proof_ref' in definition,'--run requires the packaged focused CPU proof')
    ref=definition['package_cpu_proof_ref'];proof=referenced(ref)
    require(proof.get('passed') is True and proof.get('cuda_initialized') is False
        and type(proof.get('tests')) is int and proof['tests']>0
        and all(proof.get(key)==0 for key in ('failures','errors','skips')),
        'focused package CPU proof did not pass')
    require(all(proof.get('source_sha256',{}).get(name)==value
        for name,value in definition['package_source_sha256'].items()),'CPU proof is detached from packaged sources')
    return dict(**ref,required_fields=dict(passed=True,cuda_initialized=False,failures=0,errors=0,skips=0))


def request_supervisor_stop(process,wrapper_identity,config,run_dir,lifecycle):
    """Use this di-Python process's pidfd API, not the holder Python's API.

    The original supervisor owns inference cleanup and holder restoration.
    The wrapper receives no signal and can finish its normal restoration audit.
    """
    started_path=run_dir/'Lifecycle/supervisor-started.json'
    while process.poll() is None:
        require(lifecycle.proc_identity(wrapper_identity['pid'])==wrapper_identity,
            'wrapper identity changed while requesting interruption')
        if started_path.exists():
            try:started=read_sealed(started_path)
            except json.JSONDecodeError:
                time.sleep(.1);continue
            require(started.get('config_sha256')==config['sha256'],'started supervisor belongs to a different run')
            supervisor=started['identity']
            require(supervisor.get('ppid')==wrapper_identity['pid']
                and supervisor.get('argv')==lifecycle.supervisor_command(config),
                'original supervisor argv/parent ownership differs')
            sent=lifecycle.signal_exact(supervisor,signal.SIGTERM)
            write_sealed(run_dir/'interrupt-request.json',dict(schema='benchmark-rag-explicit-interrupt-v1',
                requested_unix=time.time(),lifecycle_config_sha256=config['sha256'],
                wrapper_identity=wrapper_identity,supervisor_identity=supervisor,
                signal_sent=sent,signal_number=int(signal.SIGTERM),
                signal_sent_by_inference_python=True,wrapper_signaled=False))
            return sent
        time.sleep(.1)
    return False


def explicit_run(definition,args,run_dir,prepared):
    import profile_lifecycle as lifecycle
    from gpu_pilot.launch_pilot import source_files
    from gpu_pilot.launch_index_hit20_pilot import snapshot_sources
    # No GPU query or holder operation is reachable before this explicit branch.
    cpu_ref=admitted_package_cpu_proof(definition)
    pins={str(HERE/name):value for name,value in definition['package_source_sha256'].items()}
    pins.update({str(HERE/'GatedEngram'/name):value for name,value in source_files(HERE/'GatedEngram').items()})
    native=definition_path(definition.get('native_snapshot','NativeSnapshot')).resolve()
    pins.update({str(native/name):value for name,value in snapshot_sources(native).items()})
    pins[str(prepared['config_path'])]=file_sha(prepared['config_path'])
    settings=dict(original_supervisor=str(definition_path(definition['original_supervisor'])),
        holder_dir=str(definition_path(definition['holder_dir'])),holder_helper=str(definition_path(definition['holder_helper'])),
        holder_script=str(definition_path(definition['holder_script'])),child_python=str(definition_path(definition['child_python'])),
        child_env_json=str(prepared['env_path']),cwd=str(HERE),runner_argv=prepared['argv'][2:],
        output_dir=str(run_dir/'Lifecycle'),supervisor_result_dir=str(run_dir/'Run'),source_sha256=pins,
        admission_refs=[cpu_ref,dict(**reference(prepared['proof']),
            required_fields=dict(status='cpu_preflight_passed',gpu_initialized=False))])
    config=lifecycle.prepare_config(settings)
    config_path=run_dir/'lifecycle-config.json'
    lifecycle.save(config_path,{k:v for k,v in config.items() if k!='sha256'})
    command=[HOLDER_PYTHON,'-B','-u',str(HERE/'profile_lifecycle.py'),'--config',str(config_path)]
    with (run_dir/'lifecycle.log').open('x') as log:
        process=subprocess.Popen(command,cwd=HERE,env=lifecycle.clean_holder_environment(),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=lifecycle.proc_identity(process.pid)
        write_sealed(run_dir/'dispatch.json',dict(schema='benchmark-rag-explicit-dispatch-v1',
            created_unix=time.time(),identity=identity,lifecycle_config_sha256=config['sha256'],
            explicit_run_requested=True,timeout_seconds=TIMEOUT_SECONDS))
        print(json.dumps(dict(status='dispatched',run_directory=str(run_dir),log=str(run_dir/'lifecycle.log'))),flush=True)
        try:code=process.wait()
        except KeyboardInterrupt:
            request_supervisor_stop(process,identity,config,run_dir,lifecycle)
            print('Stop requested from the authenticated original supervisor; waiting for holder restoration.',flush=True)
            code=process.wait()
    terminal=read_sealed(run_dir/'Lifecycle/terminal.json')
    require(terminal.get('holder_restored') is True and terminal.get('holder_live_verification_passed') is True,
        'inspect preserved lifecycle evidence: physical holder restoration is not verified')
    audit=run_dir/'result.json'
    with (run_dir/'audit.log').open('x') as log:
        result=subprocess.run([str(definition_path(definition['child_python'])),'-B',
            str(HERE/'read_s12_reproduction_results.py'),'--directory',str(run_dir/'Run/benchmark'),
            '--output',str(audit)],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    require(audit.exists(),'independent audit did not produce an artifact; inspect '+str(run_dir/'audit.log'))
    value=read_sealed(audit)
    print(json.dumps(dict(status=value['status'],result=str(audit),errors=value.get('errors'),
        holder_restored=True,holder_live_verification_passed=True),ensure_ascii=False),flush=True)
    return 0 if code==0 and result.returncode==0 and value.get('status')=='complete' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length',choices=('64k','128k','both'),default='128k')
    parser.add_argument('--cases',type=int,choices=range(1,11),default=1,help='questions per selected length band')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true',help='CPU preflight only; holder and GPU remain untouched')
    mode.add_argument('--run',action='store_true',help='explicitly execute a fresh bounded GPU reproduction')
    parser.add_argument('--definition',type=Path,default=HERE/'benchmark-definition.json')
    parser.add_argument('--output-root',type=Path,default=HERE/'runs')
    values=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(values)
    if not (args.check_only or args.run):parser.print_help();return 0
    definition=load_definition(args.definition)
    child_python=str(definition_path(definition['child_python']).resolve())
    require(Path(child_python).is_file(),'pinned server Python is missing: '+child_python)
    if os.path.realpath(sys.executable)!=os.path.realpath(child_python):
        os.execve(child_python,[child_python,'-B',str(Path(__file__).resolve()),*values],os.environ.copy())
    args.output_root.mkdir(parents=True,exist_ok=True)
    run_dir=Path(tempfile.mkdtemp(prefix=('cpu-check.' if args.check_only else 'reproduction.'),dir=args.output_root)).resolve()
    prepared=cpu_preflight(definition,args.definition,args,run_dir)
    if args.check_only:
        print(json.dumps(dict(status='cpu_preflight_passed',gpu_initialized=False,gpu_released=False,
            run_directory=str(run_dir),length=args.length,cases_per_length=args.cases,
            evaluation_case_indices=prepared['config']['evaluation_case_indices'])),flush=True)
        return 0
    return explicit_run(definition,args,run_dir,prepared)


if __name__=='__main__':raise SystemExit(main())
