"""Fresh twenty-case three-arm comparison under the original owner lifecycle.

No model/GPU imports or holder operations in check-only. Never retries, signals
unknown processes, mutates snapshots, or substitutes old performance results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE))
from gpu_pilot import launch_pilot as lifecycle
from gpu_pilot.launch_index_hit20_pilot import snapshot_sources

ROOT=lifecycle.REMOTE_ROOT
REMOTE_STAGE = Path('/mnt/tidal-alsh01/dataset/redone/RedKnot_DeepInfer_20261028/DeepInfer-PR148/experiments/kce_headtail_scale20_20261106.riyvkv5p/GatedEngram')
NAME='kce-headtail-three-arm-20case-20261106-v1'
PREDECESSOR='kce-headtail128-link-20261106-v1'
PREDECESSOR_RUNNER_SHA='3e4f9333aa73f33310d41ab3893ca043bfd38e657639648522d7eac1edbb3d18'
WORKFLOW_SCHEMA='kce-headtail-scale20-launch-v1'
TIMEOUT=86400
PREPARE_TIMEOUT=300
require=lifecycle.require
INPUT_COMPANIONS=("source-manifest.json","extension-manifest.json","extension-input-protocol.json","source-plan.json")


def predecessor_evidence(root):
    return verify_previous_fullmodel_terminal(root)


def verify_cpu_proof(plan):
    proof=lifecycle.read_sealed(plan['cpu_proof_path'])
    ids=proof.get('test_ids',[])
    require(proof.get('schema') == 'kv-cache-engram-headtail-scale20-cpu-proof-v1'
        and proof.get('successful') is True and proof.get('cuda_initialized') is False
        and proof.get('cuda_init_guard_installed') is True
        and proof.get('failures')==0 and proof.get('errors')==0 and proof.get('skipped')==0
        and proof.get('source_sha256')==plan['stage_sources']
        and type(proof.get('tests')) is int and proof['tests']>=1000
        and len(ids)==len(set(ids))==proof['tests']
        and all(any(family in name for name in ids) for family in (
            'test_headtail_link_plan.','test_headtail_comparison.','test_headtail_scale20_runner.',
            'test_headtail_scale20_results.','test_launch_headtail_scale20.','test_headtail_scale20_inputs.',
            'test_index_hit20_empty_ep_qualification.','test_early_token_diagnostics.')),
        'new complete CPU proof missing, stale, failed or skipped')


def build_plan(root,stage,manifest,pilot_protocol,comparison_protocol):
    from gpu_pilot.headtail_scale20_protocol import headtail_scale20_protocol
    root,stage=lifecycle.absolute_no_symlink(root),lifecycle.absolute_no_symlink(stage)
    require(stage.name == 'GatedEngram' and root in stage.parents,
            'independent comparison stage must remain inside authorized experiment root')
    manifest,pilot_protocol,comparison_protocol=(lifecycle.absolute_no_symlink(p)
        for p in (manifest,pilot_protocol,comparison_protocol))
    require(len({manifest,pilot_protocol,comparison_protocol}) == 3
        and all(stage in p.parents for p in (manifest,pilot_protocol,comparison_protocol)),
        'three independent input/protocol files required inside stage')
    data=lifecycle.read_sealed(manifest)
    inputs=lifecycle.read_sealed(pilot_protocol,'headtail-scale20-input-protocol-v1')
    declared=lifecycle.read_sealed(comparison_protocol)
    require(declared == headtail_scale20_protocol()
        and declared['input_manifest_sha256'] == data['sha256']
        and data.get('artifact_kind') == 'headtail_scale20_inputs_v1'
        and data.get('input_protocol_sha256') == inputs['sha256']
        and 'sensitivity_protocol_sha256' not in data, 'independent twenty-case inputs/protocol differ')
    paths=lifecycle.evidence_paths(root,NAME);lifecycle.require_unused(paths)
    sources=lifecycle.source_files(stage)
    required={'gpu_pilot/run_headtail_scale20_pilot.py','gpu_pilot/read_headtail_scale20_results.py',
        'gpu_pilot/headtail_scale20_protocol.py','gpu_pilot/headtail_scale20_inputs.py',
        'gpu_pilot/launch_headtail_scale20_pilot.py','kv_cache_engram/headtail_comparison.py',
        'kv_cache_engram/headtail_link.py','kv_cache_engram/headtail_link_plan.py',
        'kv_cache_engram/early_token_diagnostics.py'}
    require(required <= set(sources),'new comparison sources missing')
    native=stage.parent/'NativeSnapshot';native_files=snapshot_sources(native)
    ref=lambda p,v:dict(path=str(p),sha256=v['sha256'],file_sha256=lifecycle.file_digest(p))
    companions={name:lifecycle.file_digest(manifest.parent/name) for name in INPUT_COMPANIONS}
    body=dict(schema=lifecycle.PLAN_SCHEMA,workflow_schema=WORKFLOW_SCHEMA,name=NAME,
        root=str(root),stage_dir=str(stage),created_unix=time.time(),no_automatic_retry=True,
        preserves_paused_experiments=True,gpu_ids=list(range(8)),timeout=TIMEOUT,prepare_timeout=PREPARE_TIMEOUT,
        timeout_purpose='24hour failure backstop; normal completion exits immediately',
        previous_ninety_minute_limit_cancelled=True,completion_case_count=20,
        paths={k:str(v) for k,v in paths.items()},protocol=ref(pilot_protocol,inputs),manifest=ref(manifest,data),
        sensitivity_protocol=None,headtail_scale20_protocol=ref(comparison_protocol,declared),
        input_companion_files_sha256=companions,stage_sources=sources,stage_sources_sha256=lifecycle.digest(sources),
        pinned_lifecycle={str(lifecycle.SUPERVISOR):lifecycle.SUPERVISOR_SHA,
            str(lifecycle.HELPER):lifecycle.HELPER_SHA,str(root/'pr148-native-env.json'):lifecycle.ENV_SHA},
        native_snapshot=str(native),native_snapshot_sources=native_files,
        native_snapshot_sources_sha256=lifecycle.digest(native_files),predecessor_evidence=predecessor_evidence(root),
        evaluation_indices=list(range(2,22)),evaluation_arms=['recomputed','tail128_link','headtail128_link'],
        dynamic_refresh_start_layer=3,cpu_proof_path=str(stage/'HEADTAIL_SCALE20_CPU_TESTS-v1.json'),
        native_yoco_swa_share_hidden=False,first_full_layers=[0,1,2],tail_layers=list(range(3,40)),
        head_tokens=128,tail_tokens=128,global_kv_complete_native_online=True,native_yoco_keep140=True,
        decode_native=True,full_canonical_gpu_bank=True,all_online_preparation_in_ttft=True,
        fresh_all_twenty_cases_all_three_arms=True,cache_storage_policy='shared_gpu_full_canonical_bank_all_arms',
        new_native_and_same_policy_parity_required=True,real_8x32k_smoke_required=True,
        live_workspace_not_execution_source=True,supervisor_owns_cpu_release_and_holder=True,
        check_only_scope='validate immutable CPU launch plan; no GPU access, model import or holder action')
    body['argv']=expected_argv(body)
    return lifecycle.seal(body)


def expected_argv(plan):
    root,stage=Path(plan['root']),Path(plan['stage_dir'])
    args=lifecycle.fixed_runner_args(root,plan['manifest']['path'],plan['protocol']['path'])
    args[args.index('--repo')+1]=str(stage.parent/'NativeSnapshot')
    return ['/root/miniconda3/bin/python',str(lifecycle.SUPERVISOR),'--python',
        str(lifecycle.RUNTIME/'di-venv/bin/python'),'--holder-dir',str(lifecycle.HOLDER),
        '--holder-helper',str(lifecycle.HELPER),'--result-dir',
        str(lifecycle.evidence_paths(root,NAME)['result_dir']),'--lock-file',str(lifecycle.LOCK),
        '--child-env-json',str(root/'pr148-native-env.json'),'--timeout',str(TIMEOUT),
        '--prepare-timeout',str(PREPARE_TIMEOUT),'--gpus','0,1,2,3,4,5,6,7','--',
        str(stage/'gpu_pilot/run_headtail_scale20_pilot.py'),*args,
        '--headtail-scale20-protocol',plan['headtail_scale20_protocol']['path']]


def verify_plan(plan):
    from gpu_pilot.headtail_scale20_protocol import headtail_scale20_protocol as protocol_factory
    require(plan.get('sha256') == lifecycle.digest({k:v for k,v in plan.items() if k != 'sha256'})
        and plan.get('schema') == lifecycle.PLAN_SCHEMA, 'comparison launch plan seal/schema differs')
    require(lifecycle.source_files(Path(plan['stage_dir'])) == plan['stage_sources'], 'stage changed after freeze')
    for key in ('manifest','protocol'):
        item=plan[key]
        require(lifecycle.file_digest(item['path']) == item['file_sha256']
            and lifecycle.read_sealed(item['path'])['sha256'] == item['sha256'], key+' changed after freeze')
    require(lifecycle.read_sealed(plan['protocol']['path']).get('schema') == 'headtail-scale20-input-protocol-v1'
        and plan.get('sensitivity_protocol') is None, 'old pilot/sensitivity protocol cannot govern twenty cases')
    for name,sha in plan['input_companion_files_sha256'].items():
        require(name in INPUT_COMPANIONS and lifecycle.file_digest(Path(plan['manifest']['path']).parent/name) == sha,
                'input companion changed after freeze')
    require(set(plan['input_companion_files_sha256']) == set(INPUT_COMPANIONS),'missing source companion binding')
    require(plan.get('workflow_schema')==WORKFLOW_SCHEMA and plan.get('name')==NAME
        and plan.get('timeout')==TIMEOUT and plan.get('prepare_timeout')==PREPARE_TIMEOUT
        and plan.get('previous_ninety_minute_limit_cancelled') is True
        and plan.get('completion_case_count') == 20
        and plan.get('fresh_all_twenty_cases_all_three_arms') is True
        and plan.get('cache_storage_policy') == 'shared_gpu_full_canonical_bank_all_arms'
        and plan.get('timeout_purpose') == '24hour failure backstop; normal completion exits immediately'
        and plan.get('evaluation_indices')==list(range(2,22))
        and plan.get('evaluation_arms')==['recomputed','tail128_link','headtail128_link']
        and plan.get('dynamic_refresh_start_layer')==3
        and plan.get('native_yoco_swa_share_hidden') is False
        and plan.get('first_full_layers')==[0,1,2] and plan.get('tail_layers')==list(range(3,40))
        and plan.get('head_tokens')==plan.get('tail_tokens')==128
        and all(plan.get(k) is True for k in ('global_kv_complete_native_online','native_yoco_keep140',
            'decode_native','full_canonical_gpu_bank','all_online_preparation_in_ttft',
            'new_native_and_same_policy_parity_required','real_8x32k_smoke_required',
            'live_workspace_not_execution_source')),'launch policy differs')
    root,stage=Path(plan['root']),Path(plan['stage_dir'])
    native=stage.parent/'NativeSnapshot'
    require(plan.get('gpu_ids')==list(range(8))
        and all(plan.get(k) is True for k in ('no_automatic_retry','preserves_paused_experiments',
            'supervisor_owns_cpu_release_and_holder'))
        and plan.get('paths')=={k:str(v) for k,v in lifecycle.evidence_paths(root,NAME).items()}
        and plan.get('stage_sources_sha256')==lifecycle.digest(plan['stage_sources'])
        and plan.get('pinned_lifecycle')=={str(lifecycle.SUPERVISOR):lifecycle.SUPERVISOR_SHA,
            str(lifecycle.HELPER):lifecycle.HELPER_SHA,str(root/'pr148-native-env.json'):lifecycle.ENV_SHA},
        'lifecycle policy, sources or evidence destinations changed')
    require(all(stage in Path(plan[k]['path']).parents for k in
        ('manifest','protocol','headtail_scale20_protocol')),
        'inputs/protocols must remain inside independent stage')
    require(plan['native_snapshot']==str(native) and snapshot_sources(native)==plan['native_snapshot_sources']
        and lifecycle.digest(plan['native_snapshot_sources'])==plan['native_snapshot_sources_sha256'],
        'private native snapshot changed')
    protocol=plan['headtail_scale20_protocol']
    require(lifecycle.file_digest(protocol['path'])==protocol['file_sha256']
        and lifecycle.read_sealed(protocol['path'])==protocol_factory(),'new tail protocol changed')
    argv=plan['argv']
    require(argv==expected_argv(plan),'complete supervisor/runner command changed')
    require(predecessor_evidence(Path(plan['root']))==plan['predecessor_evidence'],'previous evidence changed')
    require(plan.get('cpu_proof_path') == str(stage/'HEADTAIL_SCALE20_CPU_TESTS-v1.json'), 'new headtail CPU proof required')
    verify_cpu_proof(plan)


def verify_terminal(state,plan):
    require(state.get('schema')=='redknot-glm53-gpu-lifecycle-v1'
        and state.get('status')=='runner_completed; consult benchmark quality/performance gates'
        and state.get('worker_exit')==0 and state.get('holder_restored') is True
        and state.get('finished_unix') and state.get('gpu_ids')==list(range(8))
        and not any(state.get(k) for k in ('error','cleanup_error','cleanup_warnings'))
        and state.get('supervisor_sha256')==lifecycle.SUPERVISOR_SHA
        and state.get('holder_helper_sha256')==lifecycle.HELPER_SHA
        and state.get('child_environment_sha256')==lifecycle.ENV_SHA
        and state.get('runner_sha256')==plan['stage_sources']['gpu_pilot/run_headtail_scale20_pilot.py'],
        'original supervisor did not report successful finite worker and cleanup')


def verify_previous_workflow_terminal(root):
    """A finished GPU child alone does not close its parent's CPU audit/holder check."""
    root=Path(root)
    plan=lifecycle.read_sealed(root/'results'/(PREDECESSOR+'.launch-plan.json'))
    previous_stage=Path(plan['stage_dir'])
    require(plan.get('name')==PREDECESSOR and plan.get('root')==str(root)
        and previous_stage.is_absolute() and previous_stage.name=='GatedEngram'
        and root/'DeepInfer-PR148/experiments' in previous_stage.parents
        and plan.get('stage_sources',{}).get('gpu_pilot/run_headtail_link_pilot.py')==PREDECESSOR_RUNNER_SHA,
        'previous workflow location/source is not bound to its launch plan')
    terminal=lifecycle.read_sealed(previous_stage.parent/'Workflow/terminal.json')
    require(terminal.get('schema')=='headtail-link-workflow-v1'
        and terminal.get('experiment_name')==PREDECESSOR
        and terminal.get('status') in ('complete','failed')
        and type(terminal.get('finished_at')) is str and bool(terminal['finished_at'])
        and not terminal.get('holder_error')
        and bool(terminal.get('holder_final_sha256') if terminal['status']=='complete'
                 else terminal.get('holder_on_error_sha256')),
        'previous workflow audit/holder verification has not terminated')
    return dict(workflow_terminal_sha256=terminal['sha256'],previous_launch_plan_sha256=plan['sha256'])


def verify_previous_fullmodel_terminal(root):
    """Require a terminal process report and independently absent prior workers."""
    workflow=verify_previous_workflow_terminal(root)
    result=Path(root)/'results'/PREDECESSOR
    state=lifecycle.read_json(result/'supervisor.json')
    require(state.get('schema') == 'redknot-glm53-gpu-lifecycle-v1'
        and state.get('status') in ('runner_completed; consult benchmark quality/performance gates','failed')
        and state.get('finished_unix') and state.get('holder_restored') is True
        and state.get('runner_sha256') == PREDECESSOR_RUNNER_SHA
        and state.get('supervisor_sha256') == lifecycle.SUPERVISOR_SHA
        and state.get('holder_helper_sha256') == lifecycle.HELPER_SHA
        and state.get('child_environment_sha256') == lifecycle.ENV_SHA
        and state.get('gpu_ids') == list(range(8)) and not state.get('cleanup_error'),
        'prior headtail pilot has not terminated with authenticated holder restoration')
    group=lifecycle.read_sealed(result/'benchmark/process-group.json')
    require(group.get('status') in ('complete','failed')
        and group.get('owned_ranks_still_alive_after_cleanup') == [] and len(group.get('workers',[])) == 8,
        'prior process group has not drained')
    for worker in group['workers']:
        require(worker.get('alive') is False and not Path('/proc',str(worker['pid'])).exists(),
                'prior worker still present')
    started=lifecycle.read_sealed(Path(root)/'results'/(PREDECESSOR+'.supervisor-started.json'))
    require(not Path('/proc',str(started['pid'])).exists(),'prior supervisor still present')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        require(not (any(b'read_headtail_link_results.py' in arg for arg in argv)
            and str(result/'benchmark').encode() in argv),'prior final CPU audit is still active')
    outcome=lifecycle.read_sealed(Path(root)/'results'/(PREDECESSOR+'.launch-outcome.json'))
    require(outcome.get('holder_physically_verified') is True,'prior physical restoration evidence absent')
    return dict(name=PREDECESSOR,previous_workers_exited=True,previous_scores_reused=False,
        supervisor_file_sha256=lifecycle.file_digest(result/'supervisor.json'),
        process_group_sha256=group['sha256'],outcome_sha256=outcome['sha256'],
        supervisor_started_sha256=started['sha256'],prior_status=state['status'],**workflow)


def run_plan(plan):
    verify_plan(plan)
    root,stage=Path(plan['root']),Path(plan['stage_dir'])
    require(root==ROOT and stage==REMOTE_STAGE and STAGE==REMOTE_STAGE,
        'execute only from authorized new remote stage')
    lifecycle.verify_lifecycle(root)
    verify_previous_fullmodel_terminal(root)
    paths={k:Path(v) for k,v in plan['paths'].items()}
    lifecycle.require_unused(paths)
    lifecycle.write_exclusive(paths['plan'],plan)
    outcome=dict(schema='tail-link-launch-outcome-v1',plan_sha256=plan['sha256'],
        started_unix=time.time(),successful=False,automatic_retry=False,
        supervisor_started=False,holder_physically_verified=False)
    try:
        lifecycle.write_exclusive(paths['before'],lifecycle.holder_snapshot(root))
        verify_plan(plan)
        with paths['log'].open('x') as log:
            child=subprocess.Popen(plan['argv'],stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            outcome['supervisor_started']=True
            try:
                identity=lifecycle.proc_identity(child.pid)
                require(identity['argv']==plan['argv'],'original supervisor argv differs')
                lifecycle.write_exclusive(paths['started'],lifecycle.seal(dict(
                    schema='tail-link-original-supervisor-started-v1',**identity,
                    started_unix=time.time(),plan_sha256=plan['sha256'])))
                print(json.dumps(dict(name=NAME,supervisor_pid=child.pid)),flush=True)
            finally:outcome['supervisor_exit']=child.wait()
        require(outcome['supervisor_exit']==0,'original supervisor failed')
        verify_terminal(lifecycle.read_json(paths['result_dir']/'supervisor.json'),plan)
        verify_plan(plan)
        final=lifecycle.read_sealed(paths['result_dir']/'benchmark/run.json')
        from gpu_pilot.run_headtail_scale20_pilot import SCHEMA
        require(final.get('schema')==SCHEMA and final.get('status')=='complete'
            and final.get('completed_case_indices')==list(range(2,22)),'all twenty fresh three-arm cases not complete')
        outcome.update(successful=True,run_sha256=final['sha256'],independent_raw_audit_required=True)
    except BaseException as exc:outcome['error']=repr(exc)
    finally:
        try:
            after=lifecycle.holder_snapshot(root)
            lifecycle.write_exclusive(paths['after'],after)
            outcome.update(holder_physically_verified=True,holder_audit_sha256=after['sha256'])
        except BaseException as exc:
            outcome.update(successful=False,holder_physically_verified=False,holder_error=repr(exc))
        outcome['finished_unix']=time.time()
        lifecycle.write_exclusive(paths['outcome'],lifecycle.seal(outcome))
    return outcome


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv)
    inputs=STAGE/'inputs-scale20-v1'
    plan=build_plan(ROOT,STAGE,inputs/'manifest.json',inputs/'input-protocol.json',
        STAGE/'headtail-scale20-protocol.json')
    verify_plan(plan)
    if args.check_only:
        print(json.dumps(plan,sort_keys=True,separators=(',',':')));return 0
    outcome=run_plan(plan);print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome['successful'] else 1


if __name__=='__main__':raise SystemExit(main())
