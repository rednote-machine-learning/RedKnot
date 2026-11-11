"""One independent headtail128 candidate run under the original owner lifecycle.

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
REMOTE_STAGE=ROOT/'DeepInfer-PR148/experiments/kce_headtail128_link_20261106.qw7l_gfl/GatedEngram'
NAME='kce-headtail128-link-20261106-v1'
PREDECESSOR='kce-tail-dynamic-three-arm-20261105-v2'
PREDECESSOR_RUNNER_SHA='8ad51c4f88214b03b94db9cd023357e55535d94e75102486b6867bab6f76ac3a'
WORKFLOW_SCHEMA='kce-headtail128-link-launch-v1'
TIMEOUT=5400
PREPARE_TIMEOUT=300
require=lifecycle.require


def predecessor_evidence(root):
    return verify_previous_fullmodel_terminal(root)


def verify_cpu_proof(plan):
    proof=lifecycle.read_sealed(plan['cpu_proof_path'])
    ids=proof.get('test_ids',[])
    require(proof.get('schema') == 'kv-cache-engram-headtail-link-cpu-proof-v1'
        and proof.get('successful') is True and proof.get('cuda_initialized') is False
        and proof.get('cuda_init_guard_installed') is True
        and proof.get('failures')==0 and proof.get('errors')==0 and proof.get('skipped')==0
        and proof.get('source_sha256')==plan['stage_sources']
        and type(proof.get('tests')) is int and proof['tests']>=1000
        and len(ids)==len(set(ids))==proof['tests']
        and all(any(family in name for name in ids) for family in (
            'test_headtail_link_plan.','test_headtail_link_controller.','test_headtail_link_runner.',
            'test_headtail_link_results.','test_launch_headtail_link.',
            'test_index_hit20_empty_ep_qualification.')),
        'new complete CPU proof missing, stale, failed or skipped')


def build_plan(root,stage,manifest,pilot_protocol,tail_protocol,sensitivity_protocol):
    from gpu_pilot.headtail_link_protocol import headtail_link_protocol as protocol_factory
    root,stage=lifecycle.absolute_no_symlink(root),lifecycle.absolute_no_symlink(stage)
    tail_protocol=lifecycle.absolute_no_symlink(tail_protocol)
    require(stage in tail_protocol.parents,'new protocol must be inside independent stage')
    declared=lifecycle.read_sealed(tail_protocol)
    require(declared==protocol_factory(),'tail-link protocol differs from tested implementation')
    base=lifecycle.build_plan(root,stage,NAME,manifest,pilot_protocol,timeout=TIMEOUT,
        prepare_timeout=PREPARE_TIMEOUT,sensitivity_protocol=sensitivity_protocol)
    body={k:v for k,v in base.items() if k!='sha256'}
    required={'gpu_pilot/run_headtail_link_pilot.py','gpu_pilot/read_headtail_link_results.py',
        'gpu_pilot/headtail_link_protocol.py','gpu_pilot/launch_headtail_link_pilot.py',
        'kv_cache_engram/headtail_link.py','kv_cache_engram/headtail_link_plan.py'}
    require(required<=set(body['stage_sources']),'new tail-link sources missing')
    native=stage.parent/'NativeSnapshot'
    native_files=snapshot_sources(native)
    argv=list(body['argv'])
    argv[argv.index('--')+1]=str(stage/'gpu_pilot/run_headtail_link_pilot.py')
    argv[argv.index('--repo')+1]=str(native)
    argv.extend(['--headtail-link-protocol',str(tail_protocol)])
    body.update(workflow_schema=WORKFLOW_SCHEMA,argv=argv,
        native_snapshot=str(native),native_snapshot_sources=native_files,
        native_snapshot_sources_sha256=lifecycle.digest(native_files),
        headtail_link_protocol=dict(path=str(tail_protocol),sha256=declared['sha256'],
            file_sha256=lifecycle.file_digest(tail_protocol)),
        predecessor_evidence=predecessor_evidence(root),evaluation_indices=[2,3,4,5],
        evaluation_arms=['headtail128_link'],dynamic_refresh_start_layer=3,
        cpu_proof_path=str(stage/'HEADTAIL_LINK_CPU_TESTS-v1.json'),
        native_yoco_swa_share_hidden=False,
        first_full_layers=[0,1,2],tail_layers=list(range(3,40)),head_tokens=128,tail_tokens=128,
        global_kv_complete_native_online=True,native_yoco_keep140=True,
        decode_native=True,full_canonical_gpu_bank=True,
        all_online_preparation_in_ttft=True,new_native_and_same_policy_parity_required=True,
        real_8x32k_smoke_required=True,live_workspace_not_execution_source=True)
    return lifecycle.seal(body)


def expected_argv(plan):
    root,stage=Path(plan['root']),Path(plan['stage_dir'])
    args=lifecycle.fixed_runner_args(root,plan['manifest']['path'],plan['protocol']['path'],
        plan['sensitivity_protocol']['path'])
    args[args.index('--repo')+1]=str(stage.parent/'NativeSnapshot')
    return ['/root/miniconda3/bin/python',str(lifecycle.SUPERVISOR),'--python',
        str(lifecycle.RUNTIME/'di-venv/bin/python'),'--holder-dir',str(lifecycle.HOLDER),
        '--holder-helper',str(lifecycle.HELPER),'--result-dir',
        str(lifecycle.evidence_paths(root,NAME)['result_dir']),'--lock-file',str(lifecycle.LOCK),
        '--child-env-json',str(root/'pr148-native-env.json'),'--timeout',str(TIMEOUT),
        '--prepare-timeout',str(PREPARE_TIMEOUT),'--gpus','0,1,2,3,4,5,6,7','--',
        str(stage/'gpu_pilot/run_headtail_link_pilot.py'),*args,
        '--headtail-link-protocol',plan['headtail_link_protocol']['path']]


def verify_plan(plan):
    from gpu_pilot.headtail_link_protocol import headtail_link_protocol as protocol_factory
    lifecycle.verify_plan_inputs(plan)
    require(plan.get('workflow_schema')==WORKFLOW_SCHEMA and plan.get('name')==NAME
        and plan.get('timeout')==TIMEOUT and plan.get('prepare_timeout')==PREPARE_TIMEOUT
        and plan.get('evaluation_indices')==[2,3,4,5]
        and plan.get('evaluation_arms')==['headtail128_link']
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
        ('manifest','protocol','sensitivity_protocol','headtail_link_protocol')),
        'inputs/protocols must remain inside independent stage')
    require(plan['native_snapshot']==str(native) and snapshot_sources(native)==plan['native_snapshot_sources']
        and lifecycle.digest(plan['native_snapshot_sources'])==plan['native_snapshot_sources_sha256'],
        'private native snapshot changed')
    protocol=plan['headtail_link_protocol']
    require(lifecycle.file_digest(protocol['path'])==protocol['file_sha256']
        and lifecycle.read_sealed(protocol['path'])==protocol_factory(),'new tail protocol changed')
    argv=plan['argv']
    require(argv==expected_argv(plan),'complete supervisor/runner command changed')
    require(predecessor_evidence(Path(plan['root']))==plan['predecessor_evidence'],'previous evidence changed')
    require(plan.get('cpu_proof_path') == str(stage/'HEADTAIL_LINK_CPU_TESTS-v1.json'), 'new headtail CPU proof required')
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
        and state.get('runner_sha256')==plan['stage_sources']['gpu_pilot/run_headtail_link_pilot.py'],
        'original supervisor did not report successful finite worker and cleanup')


def verify_previous_fullmodel_terminal(root):
    """The completed v2 run is historical only; its workers and reader must be gone."""
    result = Path(root)/'results'/PREDECESSOR
    state = lifecycle.read_json(result/'supervisor.json')
    require(state.get('schema') == 'redknot-glm53-gpu-lifecycle-v1'
        and state.get('status') == 'runner_completed; consult benchmark quality/performance gates'
        and state.get('finished_unix') and state.get('worker_exit') == 0 and state.get('holder_restored') is True
        and state.get('runner_sha256') == PREDECESSOR_RUNNER_SHA
        and state.get('supervisor_sha256') == lifecycle.SUPERVISOR_SHA
        and state.get('holder_helper_sha256') == lifecycle.HELPER_SHA
        and state.get('child_environment_sha256') == lifecycle.ENV_SHA
        and state.get('gpu_ids') == list(range(8))
        and not any(state.get(k) for k in ('error','cleanup_error','cleanup_warnings')),
        'previous full-model run did not restore the original holder')
    run = lifecycle.read_sealed(result/'benchmark/run.json')
    require(run.get('status') == 'complete' and run.get('completed_case_indices') == [2,3,4,5],
        'previous full-model run incomplete')
    group = lifecycle.read_sealed(result/'benchmark/process-group.json')
    require(group.get('status') == 'complete' and group.get('owned_ranks_still_alive_after_cleanup') == []
        and len(group.get('workers', [])) == 8, 'previous process group did not drain')
    for worker in group['workers']:
        require(worker.get('alive') is False and worker.get('exitcode') == 0
            and not Path('/proc',str(worker['pid'])).exists(), 'previous worker still present')
    # A new experiment must not overlap the old CPU raw audit and perturb timing.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        require(not (any(b'read_tail_link_results.py' in arg for arg in argv)
            and str(result/'benchmark').encode() in argv), 'previous final CPU audit is still active')
    outcome = lifecycle.read_sealed(Path(root)/'results'/(PREDECESSOR+'.launch-outcome.json'))
    require(outcome.get('successful') is True and outcome.get('holder_physically_verified') is True,
            'previous v2 physical restoration outcome missing')
    return dict(name=PREDECESSOR, previous_workers_exited=True, previous_scores_reused=False,
        supervisor_file_sha256=lifecycle.file_digest(result/'supervisor.json'),
        run_sha256=run['sha256'], process_group_sha256=group['sha256'], outcome_sha256=outcome['sha256'])


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
        from gpu_pilot.run_headtail_link_pilot import SCHEMA
        require(final.get('schema')==SCHEMA and final.get('status')=='complete'
            and final.get('completed_case_indices')==[2,3,4,5],'all four cases not complete')
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
    inputs=STAGE/'inputs-sensitivity-v1'
    plan=build_plan(ROOT,STAGE,inputs/'manifest.json',inputs/'protocol.json',
        STAGE/'headtail-link-protocol.json',inputs/'sensitivity-protocol.json')
    verify_plan(plan)
    if args.check_only:
        print(json.dumps(plan,sort_keys=True,separators=(',',':')));return 0
    outcome=run_plan(plan);print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome['successful'] else 1


if __name__=='__main__':raise SystemExit(main())
