"""Authenticated idle-to-idle lifecycle. This module has no holder start path.

A server-side watchdog is installed before release. All signals are restricted
by the recorded child session and process start ticks. No previous job is resumed.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE))
from gpu_pilot import launch_pilot as life
from gpu_pilot.launch_index_hit20_pilot import snapshot_sources
from gpu_pilot.launch_s1_pilot import protect_git_environment
require=life.require
POLICY='leave_stopped_never_restore'
REGISTERED_128_INDICES=list(range(2,22,2))
ARMS=['recomputed','s1','s1_system']
STABLE_IDENTITY_KEYS=('checkpoint_index_sha256','checkpoint_config_sha256','tokenizer_json_sha256',
    'core_source_sha256','core_policy_sha256','native_snapshot_sources_sha256','native_profile','fixed_env')
SYSTEM_FILES=tuple('kv_cache_engram/s1_system_'+name+'.py' for name in ('runtime','batch','attention','cache'))


def _pinned_reference(ref,plan,schema=None):
    require(type(ref) is dict and set(ref)=={'path','file_sha256','seal_sha256'},'incomplete predecessor artifact reference')
    path=Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),'unsafe predecessor artifact path')
    require(plan['file_sha256'].get(str(path))==ref['file_sha256']
        and life.file_digest(path)==ref['file_sha256'],'predecessor artifact not frozen in launch plan')
    value=life.read_sealed(path,schema)
    require(value['sha256']==ref['seal_sha256'],'predecessor artifact seal differs')
    return value


def _policy_scope(identity,protocol):
    require(protocol.get('sha256')==life.digest({k:v for k,v in protocol.items() if k!='sha256'}),
        'predecessor protocol seal differs')
    require(identity.get('s1_protocol_sha256')==protocol['sha256'],'protocol is detached from runtime identity')
    scope={key:identity[key] for key in STABLE_IDENTITY_KEYS}
    scope['system_source_sha256']={name:identity['pilot_source_sha256']['GatedEngram/'+name] for name in SYSTEM_FILES}
    require(protocol.get('baseline_core_policy_sha256')==scope['core_policy_sha256']
        and life.digest(protocol['baseline_core_policy'])==scope['core_policy_sha256'],'baseline policy changed')
    scope.update(quality=protocol['quality'],timing={k:protocol['timing'][k] for k in
        ('concurrency','warmups_per_arm','repeats','output_tokens')})
    return scope


def validate_predecessor_admission(plan):
    """Read sealed evidence; never turn a provisional case or an arm fragment into a result."""
    admission=_pinned_reference(plan['predecessor_admission'],plan,'s11-predecessor-admission-v1')
    require(admission.get('status')=='accepted_user_stop' and admission.get('holder_policy')==POLICY
        and admission.get('registered_case_indices')==REGISTERED_128_INDICES
        and admission.get('completed_case_indices')==REGISTERED_128_INDICES[:-1]
        and admission.get('pending_case_indices')==[20]
        and admission.get('new_evaluation_case_indices')==[20,2,3,4]
        and admission.get('user_requested_stop_authenticated') is True
        and admission.get('physical_idle_verified') is True
        and admission.get('source_run_count')==2 and admission.get('single_run') is False,
        'exact nine original128 cases and authenticated user stop must be accepted')
    entries=admission.get('audits')
    require(type(entries) is list and len(entries)==2,'two predecessor audit sets required')
    pre=plan['runtime_preflight']
    require(pre.get('status')=='cpu_preflight_passed' and pre.get('gpu_initialized') is False,
        'actual CPU preflight is required for current execution scope')
    require([r['index'] for r in pre['evaluation_cases']]==[20,2,3,4],'current preflight is not the remaining128 plus three256 population')
    command=plan['runner_argv'];protocol_path=Path(command[command.index('--s1-protocol')+1])
    require(protocol_path==STAGE/'s11-protocol.json'
        and plan['file_sha256'].get(str(protocol_path))==life.file_digest(protocol_path),'current protocol is not pinned')
    current_protocol=life.read_sealed(protocol_path)
    require(current_protocol['sha256']==plan['protocol_sha256']
        and current_protocol['input_manifest_sha256']==plan['manifest_sha256']
        and current_protocol['cases']==[20,2,3,4],'current 256K protocol/manifest mismatch')
    manifest_path=Path(command[command.index('--manifest')+1])
    require(plan['file_sha256'].get(str(manifest_path))==life.file_digest(manifest_path),'new combined manifest not frozen')
    current_manifest=life.read_sealed(manifest_path)
    require(current_manifest['sha256']==plan['manifest_sha256'],'combined manifest seal changed')
    current_row20=next(row for row in current_manifest['cohort'] if row['index']==20)
    current_scope=_policy_scope(pre['identity'],current_protocol)
    for name,sha in current_scope['system_source_sha256'].items():
        require(plan['stage_sources'].get(name)==sha,'current system implementation differs from CPU preflight')
    seen=[];run_ids=[];previous_audit=None;manifest_sha=None
    for ordinal,item in enumerate(entries):
        run_id=item.get('run_id')
        require(type(run_id) is str and run_id and run_id not in run_ids,'distinct predecessor run IDs required')
        run_ids.append(run_id)
        audit=_pinned_reference(item['audit'],plan)
        expected_schema=('s1-system128-independent-readout-v1','s1-system128-continuation-independent-readout-v1')[ordinal]
        require(audit.get('schema')==expected_schema,'predecessor audit role/schema differs')
        prior=_pinned_reference(item['preflight'],plan,'kv-cache-engram-s1-system128-pilot-run-v1')
        workflow=_pinned_reference(item['workflow_terminal'],plan,'s1-system-workflow-terminal-v1')
        terminal=_pinned_reference(item['lifecycle_terminal'],plan,'s1-system-idle-terminal-v1')
        require(workflow.get('status') == ('partial' if ordinal==0 else 'failed') and workflow.get('holder_policy')==POLICY
            and terminal.get('holder_policy')==POLICY and terminal.get('gpu_idle_verified') is True
            and not terminal.get('cleanup_error') and workflow.get('audit_sha256')==audit['sha256']
            and workflow.get('lifecycle_terminal_sha256')==terminal['sha256']
            and workflow.get('plan_sha256')==terminal.get('plan_sha256')
            and isinstance(workflow.get('plan_sha256'),str),'predecessor workflow/cleanup/audit closure differs')
        require(audit.get('errors')==[] and audit.get('qualification_passed') is True
            and audit.get('system_metadata_qualification_passed') is True
            and audit.get('system_exact_quality_preserved_for_all_completed_cases') is True
            and audit.get('arms')==ARMS,'predecessor did not pass all three-arm numerical and quality gates')
        indices=item.get('accepted_case_indices');expected=REGISTERED_128_INDICES[len(seen):]
        require(type(indices) is list and indices==expected[:len(indices)]
            and all(type(i) is int for i in indices) and len(indices)>0
            and audit.get('completed_case_indices')==indices and audit.get('complete_cases')==len(indices),
            'predecessor contribution is not a disjoint complete-case prefix')
        require(prior['evaluation_case_indices']==(REGISTERED_128_INDICES if ordinal==0 else expected)
            and audit.get('planned_cases')==len(prior['evaluation_case_indices']),'predecessor registered/execution population differs')
        if ordinal==0:
            require(audit.get('status')=='partial_complete' and len(indices)<10
                and audit.get('all_completed_cases_audited') is True
                and all(audit.get(k) is False for k in ('final_run_audited','overall_run_complete','all_registered_cases_audited')),
                'original run must retain its independently audited partial status')
        else:
            require(audit.get('status')=='partial_complete' and indices==expected[:-1]
                and audit.get('all_completed_cases_audited') is True
                and audit.get('all_execution_cases_audited') is False
                and all(audit.get(k) is False for k in ('final_run_audited','overall_run_complete','all_registered_cases_audited'))
                and audit.get('partial_completion_reason')=='runner_failure'
                and workflow.get('status')=='failed' and terminal.get('status')=='failed',
                'user-stopped continuation must retain its failed terminal and audited six-case prefix')
            require(audit.get('predecessor_audit_sha256')==previous_audit['sha256'],
                'continuation audit is not bound to the original accepted predecessor')
        identity=prior['identity']
        require(_policy_scope(identity,prior['protocol'])==current_scope,'model, native/core/system code or generation policy changed')
        require(audit.get('execution_configuration_sha256')==identity['execution_configuration_sha256']
            and audit.get('protocol_sha256')==prior['protocol']['sha256']
            and audit.get('manifest_sha256')==prior['manifest_sha256']
            and audit.get('frozen_sources_verified',{}).get('pilot_source_sha256')==life.digest(identity['pilot_source_sha256']),
            'predecessor audit is detached from its preflight and sources')
        if manifest_sha is None:manifest_sha=prior['manifest_sha256']
        prior_row20=next(row for row in prior['audit_cases'] if row['index']==20)
        require(all(current_row20.get(key)==prior_row20.get(key) for key in
            ('input_ids_sha256','input_tokens','documents','sample_id','dataset','question','references')),
            'remaining128 input/gold changed from the original registered case20')
        require(prior['manifest_sha256']==manifest_sha,'the two 128K runs use different input manifests')
        for key in ('core_source_sha256','core_policy_sha256','native_snapshot_sources_sha256'):
            require(audit.get(key)==identity[key],'predecessor audited core identity differs')
        cases=audit.get('cases');rows={r['index']:r for r in prior['audit_cases']}
        require(type(cases) is list and [c.get('index') for c in cases]==indices,'predecessor case inventory differs')
        for case in cases:
            i=case['index'];row=rows[i];ref=audit.get('evidence',{}).get(f'case{i}.paired.json')
            require(type(ref) is dict and re.fullmatch('[0-9a-f]{64}',ref.get('file_sha256',''))
                and ref.get('seal_sha256')==case.get('closure_sha256'),'closed paired evidence missing')
            require(all(case.get(k)==row[k] for k in ('input_tokens','dataset','question','references'))
                and case.get('cache_blocks')==row['documents']
                and case.get('document_content_sha256')==[d['token_ids_sha256'] for d in row['documents']],
                'predecessor paired input differs from actual preflight')
            require(set(case['quality'])==set(ARMS) and set(case['online_ttft_samples_s'])==set(ARMS)
                and all(len(case['online_ttft_samples_s'][a])==3 for a in ARMS)
                and case.get('system_quality_output_ids_equal_to_s1') is True
                and case['quality']['s1']['output_ids']==case['quality']['s1_system']['output_ids'],
                'predecessor contribution lacks a complete fresh three-arm case')
        seen.extend(indices);previous_audit=audit
    require(seen==REGISTERED_128_INDICES[:-1],'accepted predecessor population is not exactly nine cases')
    _validate_user_stop(admission,plan,entries[-1])
    return admission


def _validate_user_stop(admission,plan,entry):
    stop=_pinned_reference(admission['user_stop_request'],plan,'s1-system-user-directed-stop-v1')
    terminal=_pinned_reference(admission['physical_idle'],plan,'s1-system-idle-terminal-v1')
    require(admission['physical_idle']==entry['lifecycle_terminal'],
        'physical idle proof must be the actual stopped lifecycle terminal')
    require(stop.get('action')=='SIGTERM_authenticated_runner_leader_only_supervisor_owns_cleanup'
        and stop.get('source_or_result_modified') is False and stop.get('holder_policy')==POLICY
        and stop.get('plan_sha256')==terminal.get('plan_sha256')
        and terminal.get('runner_exit')==-15 and terminal.get('termination_reason')=='runner_failure'
        and terminal.get('status')=='failed' and terminal.get('gpu_idle_verified') is True
        and not terminal.get('cleanup_error'),'authenticated user stop/cleanup semantics differ')
    for role in ('child','supervisor'):
        expected=terminal.get(role,{});actual=stop.get(role,{})
        require(all(actual.get(key)==expected.get(key) and actual.get(key) is not None
            for key in ('pid','ppid','start_ticks','pgid','sid','argv')),
            'user stop did not authenticate the actual '+role)
    require(stop['child']['pid']==stop['child']['pgid']==stop['child']['sid']
        and stop['child']['ppid']==stop['supervisor']['pid'],'user stop target is not the actual session leader')
    audit=_pinned_reference(entry['audit'],plan)
    expected_files={f'case{i}.paired.json' for i in audit['completed_case_indices']}
    require(set(stop.get('completed_case_files',[]))==expected_files
        and len(stop['completed_case_files'])==len(expected_files),'stop receipt completed-case inventory differs')
    samples=terminal.get('after_snapshots')
    require(type(samples) is list and len(samples)==3,'three post-stop GPU observations required')
    previous=stop.get('requested_unix')
    require(type(previous) in (int,float) and terminal['released_unix']<=previous,
        'user stop predates actual GPU release')
    for sample in samples:
        rows=[row.strip().split(', ') for row in sample.get('gpu','').splitlines() if row.strip()]
        require(sample.get('idle') is True and not sample.get('compute_apps','').strip()
            and len(rows)==8 and sorted(int(row[0]) for row in rows)==list(range(8))
            and all(len(row)==5 and float(row[3])==0 and float(row[4])==0 for row in rows)
            and sample['time_unix']>=previous,'physical post-stop GPU state is not eight idle cards')
        previous=sample['time_unix']
    require(previous<=terminal['finished_unix'],'post-stop observation is after terminal seal time')
    return stop,terminal


def release_deadlines(plan, wall_now, monotonic_now):
    absolute=plan.get('absolute_deadline_unix')
    remaining=plan['timeout_seconds'] if absolute is None else min(plan['timeout_seconds'],absolute-wall_now)
    if remaining<=0:raise TimeoutError('original round deadline expired before GPU release')
    return wall_now+remaining,monotonic_now+remaining


def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('x') as f:json.dump(life.seal(value),f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def identity(pid):
    try:
        p=Path('/proc')/str(pid);text=(p/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        return dict(pid=int(pid),ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
            start_ticks=int(fields[19]),state=fields[0],argv=(p/'cmdline').read_bytes().split(b'\0')[:-1])
    except (FileNotFoundError,ProcessLookupError):return None


def serial_identity(pid):
    value=identity(pid)
    if value:value['argv']=[x.decode(errors='replace') for x in value['argv']]
    return value


def same_process(expected):
    actual=identity(expected['pid'])
    return bool(actual and actual['start_ticks']==expected['start_ticks'] and actual['sid']==expected['sid'])


def owned_members(child):
    """Session IDs cannot be reused while descendants of this session remain."""
    leader=identity(child['pid'])
    if leader is not None and leader['start_ticks']!=child['start_ticks']:
        return []  # Recycled leader PID: the old session is already gone.
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        v=identity(int(entry.name))
        if v and v['sid']==child['sid'] and v['start_ticks']>=child['start_ticks'] and v['state']!='Z':found.append(v)
    return found


def signal_member(member,sig):
    if not same_process(member):return
    # Bind the kernel process identity before the final /proc verification;
    # an intervening PID recycle must never redirect our signal.
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux pidfd signaling required')
    try:descriptor=os.pidfd_open(member['pid'])
    except ProcessLookupError:return
    try:
        if same_process(member):
            try:signal.pidfd_send_signal(descriptor,sig)
            except ProcessLookupError:pass
    finally:os.close(descriptor)


def cleanup(child):
    require(child['pid']>1 and child['pid']==child['sid']==child['pgid'],'child must own independent session')
    initial=owned_members(child)
    for member in initial:signal_member(member,signal.SIGTERM)
    deadline=time.monotonic()+8
    while owned_members(child) and time.monotonic()<deadline:time.sleep(.2)
    for member in owned_members(child):signal_member(member,signal.SIGKILL)
    deadline=time.monotonic()+10
    while owned_members(child) and time.monotonic()<deadline:time.sleep(.2)
    require(not owned_members(child),'owned child processes survived cleanup')
    return [dict(pid=x['pid'],start_ticks=x['start_ticks'],sid=x['sid']) for x in initial]


def gpu_snapshot():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader,nounits'],text=True,timeout=20)
    rows=[x.strip().split(', ') for x in gpu.splitlines() if x.strip()]
    require(len(rows)==8 and sorted(int(x[0]) for x in rows)==list(range(8)),'eight expected GPUs required')
    return dict(time_unix=time.time(),gpu=gpu,compute_apps=apps,idle=not apps.strip())


def validate_plan(path):
    plan=life.read_sealed(path,'s11-idle-lifecycle-plan-v1')
    require(plan['holder_policy']==POLICY and plan['timeout_seconds']==86400,'user lifecycle policy differs')
    require(plan.get('user_extension_scope')=='finish_10x128k_then_3x256k_leave_holder_off',
            'explicit completion-driven user extension required')
    require(Path(plan['stage'])==STAGE and Path(plan['root'])==life.REMOTE_ROOT,'stage/root mismatch')
    name=plan.get('experiment_name')
    require(type(name) is str and name=='kce-s11-full-two-arm-20261109-v1','explicit 256K experiment name required')
    require(Path(plan['result_dir'])==life.REMOTE_ROOT/'results'/name,'unexpected result directory')
    require(plan['prepare_timeout_seconds']==600,'bounded CPU preparation required')
    require(plan.get('absolute_deadline_unix') is None,'new completion-driven run must not inherit the old round deadline')
    require(life.source_files(STAGE)==plan['stage_sources'],'frozen source mismatch')
    require(snapshot_sources(STAGE.parent/'NativeSnapshot')==plan['native_sources'],'native snapshot mismatch')
    for name,sha in plan['file_sha256'].items():require(life.file_digest(name)==sha,'frozen file changed: '+name)
    validate_predecessor_admission(plan)
    proof=life.read_sealed(plan['cpu_proof'],'s11-cpu-proof-v1')
    require(proof['successful'] and not any(proof[k] for k in ('failures','errors','skipped','cuda_initialized'))
        and proof['cuda_init_guard_installed'] and proof['source_sha256']==plan['stage_sources'],'complete CPU admission required')
    require(proof['tests']>=1800 and len(set(proof['test_ids']))==proof['tests'],'complete unique test suite required')
    for family in ('test_s1_system_attention.','test_s1_system_cache.','test_s1_system_batch.','test_s1_system_runtime.','test_s1_system_protocol.','test_s1_system_lifecycle.',
                   'test_s11_inputs.','test_s11_protocol.','test_s11_results.','test_s11_qualification.','test_s11_lifecycle.'):
        require(any(family in x for x in proof['test_ids']),'CPU family absent: '+family)
    command=plan['runner_argv']
    require(command[:3]==[str(life.RUNTIME/'di-venv/bin/python'),'-B',str(STAGE/'gpu_pilot/run_s11_pilot.py')], 'unexpected runner')
    require('--allow-gpu-init' not in command and '--check-only' not in command,'authenticated fresh runner required')
    require(command[command.index('--repo')+1]==str(STAGE.parent/'NativeSnapshot'),'private native snapshot required')
    require(not any(x in command for x in ('--out-dir','--cpu-ready-file','--release-file')),'lifecycle owns readiness paths')
    return plan


def watchdog(plan_path,owner_path):
    plan=validate_plan(plan_path);owner=life.read_sealed(owner_path,'s1-system-owner-v1')
    require(owner['plan_sha256']==plan['sha256'],'watchdog owner/plan mismatch')
    root=Path(plan['result_dir']);child=owner['child'];parent=owner['supervisor'];deadline=None
    atomic(root/'watchdog-ready.json',dict(schema='s1-system-watchdog-ready-v1',identity=serial_identity(os.getpid()),owner_sha256=owner['sha256'],plan_sha256=plan['sha256']))
    reason='';started=time.monotonic()
    while True:
        if (root/'terminal.json').exists():
            terminal=life.read_sealed(root/'terminal.json','s1-system-idle-terminal-v1')
            require(terminal['plan_sha256']==plan['sha256'],'watchdog terminal/plan mismatch')
            return 0
        if not same_process(parent):reason='supervisor_disappeared';break
        release_path=root/'release.json'
        if release_path.exists() and deadline is None:
            release=life.read_sealed(release_path,'s1-system-authenticated-release-v1')
            require(release['pid']==child['pid'] and release['start_ticks']==child['start_ticks']
                and release['plan_sha256']==plan['sha256'] and release['holder_policy']==POLICY,'wrong release identity')
            deadline=release['deadline_monotonic']
        if deadline is not None and time.monotonic()>=deadline:reason='gpu_deadline';break
        if deadline is None and time.monotonic()-started>plan['prepare_timeout_seconds']+30:reason='prepare_deadline';break
        time.sleep(.5)
    killed=cleanup(child)
    atomic(root/'watchdog-stop.json',dict(schema='s1-system-watchdog-stop-v1',reason=reason,terminated=killed,
        plan_sha256=plan['sha256'],time_unix=time.time(),holder_policy=POLICY))
    return 0


def run(plan_path):
    plan=validate_plan(plan_path)
    require(plan.get('absolute_deadline_unix') is None or time.time()<plan['absolute_deadline_unix'],
            'declared experiment deadline already expired')
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux pidfd cleanup required before launch')
    result=Path(plan['result_dir']);require(not result.exists(),'never resume or overwrite an existing experiment')
    lock=open(life.LOCK,'a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=gpu_snapshot();require(before['idle'],'GPU currently occupied; will not kill another job')
    result.mkdir(parents=True,exist_ok=False)
    atomic(result/'before.json',dict(schema='s1-system-idle-before-v1',snapshot=before,holder_policy=POLICY,plan_sha256=plan['sha256']))
    child=None;watch=None;outcome=dict(schema='s1-system-idle-terminal-v1',status='failed',holder_policy=POLICY,
        plan_sha256=plan['sha256'],supervisor=serial_identity(os.getpid()),started_unix=time.time(),gpu_released=False)
    env=dict(os.environ);env.update(life.read_json(Path(plan['root'])/'pr148-native-env.json'))
    protect_git_environment(env);env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    argv=plan['runner_argv']+['--out-dir',str(result/'runner'),'--cpu-ready-file',str(result/'cpu-ready.json'),'--release-file',str(result/'release.json')]
    try:
        with (result/'runner.log').open('x') as log:
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_id=serial_identity(child.pid);require(child_id and child_id['sid']==child.pid,'child identity unavailable')
            outcome['child']=child_id
            atomic(result/'owner.json',dict(schema='s1-system-owner-v1',plan_sha256=plan['sha256'],child=child_id,supervisor=outcome['supervisor']))
            with (result/'watchdog.log').open('x') as wdlog:
                watch=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--plan',str(plan_path),'--watchdog',str(result/'owner.json')],env=env,stdin=subprocess.DEVNULL,stdout=wdlog,stderr=subprocess.STDOUT,start_new_session=True)
            prep_deadline=time.monotonic()+plan['prepare_timeout_seconds']
            while not ((result/'cpu-ready.json').exists() and (result/'watchdog-ready.json').exists()):
                require(child.poll() is None,'runner failed before GPU release')
                require(watch.poll() is None,'watchdog failed before GPU release')
                require(time.monotonic()<prep_deadline,'CPU prepare timeout')
                time.sleep(.25)
            ready=life.read_json(result/'cpu-ready.json');watch_ready=life.read_sealed(result/'watchdog-ready.json','s1-system-watchdog-ready-v1')
            require(ready['pid']==child.pid and ready['manifest_sha256']==plan['manifest_sha256']
                and ready['status']=='cpu-ready-no-torch-import','invalid CPU readiness')
            require(watch_ready['plan_sha256']==plan['sha256'] and watch_ready['identity']['pid']==watch.pid
                and watch_ready['owner_sha256']==life.read_sealed(result/'owner.json','s1-system-owner-v1')['sha256']
                and same_process(watch_ready['identity']),'watchdog identity mismatch')
            require(gpu_snapshot()['idle'],'GPU became occupied before release')
            release=time.time();wall_deadline,deadline=release_deadlines(plan,release,time.monotonic())
            atomic(result/'release.json',dict(schema='s1-system-authenticated-release-v1',pid=child.pid,start_ticks=child_id['start_ticks'],
                released_unix=release,deadline_unix=wall_deadline,deadline_monotonic=deadline,
                plan_sha256=plan['sha256'],holder_policy=POLICY,watchdog=watch_ready['identity']))
            outcome.update(gpu_released=True,released_unix=release,deadline_unix=wall_deadline)
            while child.poll() is None:
                require(watch.poll() is None,'watchdog exited while runner active')
                if time.monotonic()>=deadline:raise TimeoutError('24-hour emergency GPU execution limit')
                time.sleep(.5)
            outcome['runner_exit']=child.returncode
            require(child.returncode==0,'runner exited unsuccessfully')
            outcome['status']='runner_complete_pending_independent_audit'
    except BaseException as exc:
        outcome['error']=repr(exc);outcome['traceback']=traceback.format_exc()
        outcome['termination_reason']='gpu_deadline' if isinstance(exc,TimeoutError) else 'runner_failure'
    finally:
        try:
            if child is not None and outcome.get('child'):outcome['cleanup_signaled']=cleanup(outcome['child']);child.wait(timeout=10)
            snapshots=[]
            for _ in range(3):snapshots.append(gpu_snapshot());time.sleep(.5)
            outcome['after_snapshots']=snapshots;outcome['gpu_idle_verified']=all(x['idle'] for x in snapshots)
            require(outcome['gpu_idle_verified'],'GPU contexts remain; no unknown process will be killed')
        except BaseException as exc:outcome['cleanup_error']=repr(exc);outcome['status']='failed'
        outcome['finished_unix']=time.time();atomic(result/'terminal.json',outcome)
        if watch is not None:
            try:outcome['watchdog_exit']=watch.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return 0 if outcome['status']=='runner_complete_pending_independent_audit' and outcome.get('gpu_idle_verified') else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--watchdog',type=Path);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:plan=validate_plan(args.plan);print(json.dumps(dict(status='lifecycle_cpu_check_passed',plan_sha256=plan['sha256'],holder_policy=POLICY)));return 0
    return watchdog(args.plan,args.watchdog) if args.watchdog else run(args.plan)

if __name__=='__main__':raise SystemExit(main())
