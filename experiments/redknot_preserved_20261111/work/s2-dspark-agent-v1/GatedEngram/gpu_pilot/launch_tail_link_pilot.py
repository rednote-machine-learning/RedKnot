"""One independent full-model tail128 run under the original owner lifecycle.

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
REMOTE_STAGE=ROOT/'DeepInfer-PR148/experiments/kce_tail_dynamic_v2_20261105.j4d28sr5/GatedEngram'
NAME='kce-tail-dynamic-three-arm-20261105-v2'
PREVIOUS_DYNAMIC='kce-tail-dynamic-three-arm-20261105-v1'
PREVIOUS_DYNAMIC_RUNNER_SHA='d98b6fb5852673567dfeb72bb7ef884141135f98d32b832cc1eca4a3401db4e0'
PREDECESSOR='kce-tail-memory-128-operator-20261105-v1'
PREDECESSOR_RESULT_SEAL='dc5681160df25f9f3e9bbf31c8bf0abb09375da0c7aa06d6311014aae0f3729d'
PREDECESSOR_RUNNER_SHA='66a13d771d1427af8ee285045ac70398c357c76b76d80c075481b748059fe6e1'
WORKFLOW_SCHEMA='kce-tail-dynamic-three-arm-launch-v2'
TIMEOUT=21600
PREPARE_TIMEOUT=300
require=lifecycle.require


def predecessor_evidence(root):
    directory=Path(root)/'results'/PREDECESSOR
    state=lifecycle.read_json(directory/'supervisor.json')
    require(state.get('schema')=='redknot-glm53-gpu-lifecycle-v1'
        and state.get('status')=='runner_completed; consult benchmark quality/performance gates'
        and state.get('worker_exit')==0 and state.get('holder_restored') is True
        and state.get('finished_unix') and state.get('gpu_ids')==list(range(8))
        and state.get('runner_sha256')==PREDECESSOR_RUNNER_SHA
        and state.get('supervisor_sha256')==lifecycle.SUPERVISOR_SHA
        and state.get('holder_helper_sha256')==lifecycle.HELPER_SHA
        and state.get('child_environment_sha256')==lifecycle.ENV_SHA
        and not any(state.get(k) for k in ('error','cleanup_error','cleanup_warnings')),
        'previous operator did not finish with authenticated cleanup')
    result=lifecycle.read_sealed(directory/'benchmark/operator-result.json')
    require(result.get('sha256')==PREDECESSOR_RESULT_SEAL and result.get('status')=='complete'
        and result.get('synthetic_operator_only') is True
        and result.get('model_accuracy_measured') is False
        and result.get('end_to_end_ttft_measured') is False,
        'previous synthetic operator evidence changed; never substitute it for this model run')
    ids=result['lifecycle_identity']
    for pid in (ids['parent_pid'],ids['pid']):
        path=Path('/proc')/str(pid)/'stat'
        if path.exists():
            state_token=path.read_text().rsplit(') ',1)[1].split()[0]
            require(state_token in ('Z','X','x'),'previous operator process remains')
    return dict(name=PREDECESSOR,terminal=True,model_results_reused=False,
        state_file_sha256=lifecycle.file_digest(directory/'supervisor.json'),
        result_file_sha256=lifecycle.file_digest(directory/'benchmark/operator-result.json'),
        result_seal_sha256=result['sha256'])


def verify_cpu_proof(plan):
    proof=lifecycle.read_sealed(plan['cpu_proof_path'])
    ids=proof.get('test_ids',[])
    require(proof.get('successful') is True and proof.get('cuda_initialized') is False
        and proof.get('cuda_init_guard_installed') is True
        and proof.get('failures')==0 and proof.get('errors')==0 and proof.get('skipped')==0
        and proof.get('source_sha256')==plan['stage_sources']
        and type(proof.get('tests')) is int and proof['tests']>=1160
        and len(ids)==len(set(ids))==proof['tests']
        and all(any(family in name for name in ids) for family in (
            'test_tail_link_plan.','test_tail_link_controller.','test_tail_link_pilot.',
            'test_tail_link_results.','test_launch_tail_link.',
            'test_index_hit20_empty_ep_qualification.',
            'test_dynamic_tail_controller.','test_dynamic_tail_runner.','test_dynamic_tail_results.')),
        'new complete CPU proof missing, stale, failed or skipped')


def build_plan(root,stage,manifest,pilot_protocol,tail_protocol,sensitivity_protocol):
    from gpu_pilot.tail_link_protocol import tail_link_protocol as protocol_factory
    root,stage=lifecycle.absolute_no_symlink(root),lifecycle.absolute_no_symlink(stage)
    tail_protocol=lifecycle.absolute_no_symlink(tail_protocol)
    require(stage in tail_protocol.parents,'new protocol must be inside independent stage')
    declared=lifecycle.read_sealed(tail_protocol)
    require(declared==protocol_factory(),'tail-link protocol differs from tested implementation')
    base=lifecycle.build_plan(root,stage,NAME,manifest,pilot_protocol,timeout=TIMEOUT,
        prepare_timeout=PREPARE_TIMEOUT,sensitivity_protocol=sensitivity_protocol)
    body={k:v for k,v in base.items() if k!='sha256'}
    required={'gpu_pilot/run_tail_link_pilot.py','gpu_pilot/read_tail_link_results.py',
        'gpu_pilot/tail_link_protocol.py','gpu_pilot/launch_tail_link_pilot.py',
        'kv_cache_engram/tail_link.py','kv_cache_engram/tail_link_plan.py'}
    require(required<=set(body['stage_sources']),'new tail-link sources missing')
    native=stage.parent/'NativeSnapshot'
    native_files=snapshot_sources(native)
    argv=list(body['argv'])
    argv[argv.index('--')+1]=str(stage/'gpu_pilot/run_tail_link_pilot.py')
    argv[argv.index('--repo')+1]=str(native)
    argv.extend(['--tail-link-protocol',str(tail_protocol)])
    body.update(workflow_schema=WORKFLOW_SCHEMA,argv=argv,
        native_snapshot=str(native),native_snapshot_sources=native_files,
        native_snapshot_sources_sha256=lifecycle.digest(native_files),
        tail_link_protocol=dict(path=str(tail_protocol),sha256=declared['sha256'],
            file_sha256=lifecycle.file_digest(tail_protocol)),
        predecessor_evidence=predecessor_evidence(root),evaluation_indices=[2,3,4,5],
        evaluation_arms=['recomputed','tail128_link','tail128_dynamic'],dynamic_refresh_start_layer=4,
        cpu_proof_path=str(stage/'DYNAMIC_TAIL_CPU_TESTS-v2.json'),
        previous_dynamic_evidence=verify_previous_dynamic_terminal(root),
        native_yoco_swa_share_hidden=False,
        first_full_layers=[0,1,2],tail_layers=list(range(3,40)),tail_tokens=128,
        global_kv_complete_native_online=True,native_yoco_keep140=True,
        decode_native=True,cpu_cache_each_phase_fresh_gpu_staging=True,
        all_h2d_prefetch_waits_in_ttft=True,new_native_and_same_policy_parity_required=True,
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
        str(stage/'gpu_pilot/run_tail_link_pilot.py'),*args,
        '--tail-link-protocol',plan['tail_link_protocol']['path']]


def verify_plan(plan):
    from gpu_pilot.tail_link_protocol import tail_link_protocol as protocol_factory
    lifecycle.verify_plan_inputs(plan)
    require(plan.get('workflow_schema')==WORKFLOW_SCHEMA and plan.get('name')==NAME
        and plan.get('timeout')==TIMEOUT and plan.get('prepare_timeout')==PREPARE_TIMEOUT
        and plan.get('evaluation_indices')==[2,3,4,5]
        and plan.get('evaluation_arms')==['recomputed','tail128_link','tail128_dynamic']
        and plan.get('dynamic_refresh_start_layer')==4
        and plan.get('native_yoco_swa_share_hidden') is False
        and plan.get('first_full_layers')==[0,1,2] and plan.get('tail_layers')==list(range(3,40))
        and plan.get('tail_tokens')==128
        and all(plan.get(k) is True for k in ('global_kv_complete_native_online','native_yoco_keep140',
            'decode_native','cpu_cache_each_phase_fresh_gpu_staging','all_h2d_prefetch_waits_in_ttft',
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
        ('manifest','protocol','sensitivity_protocol','tail_link_protocol')),
        'inputs/protocols must remain inside independent stage')
    require(plan['native_snapshot']==str(native) and snapshot_sources(native)==plan['native_snapshot_sources']
        and lifecycle.digest(plan['native_snapshot_sources'])==plan['native_snapshot_sources_sha256'],
        'private native snapshot changed')
    protocol=plan['tail_link_protocol']
    require(lifecycle.file_digest(protocol['path'])==protocol['file_sha256']
        and lifecycle.read_sealed(protocol['path'])==protocol_factory(),'new tail protocol changed')
    argv=plan['argv']
    require(argv==expected_argv(plan),'complete supervisor/runner command changed')
    require(predecessor_evidence(Path(plan['root']))==plan['predecessor_evidence'],'previous evidence changed')
    require(plan.get('cpu_proof_path') == str(stage/'DYNAMIC_TAIL_CPU_TESTS-v2.json'), 'new v2 CPU proof required')
    require(verify_previous_dynamic_terminal(root)==plan.get('previous_dynamic_evidence'),
            'failed v1 terminal evidence changed')
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
        and state.get('runner_sha256')==plan['stage_sources']['gpu_pilot/run_tail_link_pilot.py'],
        'original supervisor did not report successful finite worker and cleanup')


def verify_previous_fullmodel_terminal(root):
    """Require the previous static run's workers exited; never reuse its scores."""
    result = Path(root)/'results/kce-tail-link-fullmodel-20261105-v1'
    state = lifecycle.read_json(result/'supervisor.json')
    require(state.get('worker_exit') == 0 and state.get('holder_restored') is True
        and state.get('runner_sha256') == 'c53bc6a0a7461f1d93ab2f8a71b802c1be6e97ad33e0e23efca853c4329e2586'
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
    return dict(previous_workers_exited=True, previous_scores_reused=False)


def verify_previous_dynamic_terminal(root):
    """Bind v1's failure and cleanup evidence, without treating it as success."""
    paths = lifecycle.evidence_paths(Path(root), PREVIOUS_DYNAMIC)
    directory = paths['result_dir']
    state = lifecycle.read_json(directory/'supervisor.json')
    warning = "_UnstableOwnership('refusing to signal unauthenticated process-group member 4022298')"
    require(state.get('schema') == 'redknot-glm53-gpu-lifecycle-v1'
        and state.get('status') == 'failed' and state.get('worker_exit') == 1
        and state.get('holder_restored') is True and state.get('finished_unix')
        and state.get('runner_sha256') == PREVIOUS_DYNAMIC_RUNNER_SHA
        and state.get('supervisor_sha256') == lifecycle.SUPERVISOR_SHA
        and state.get('holder_helper_sha256') == lifecycle.HELPER_SHA
        and state.get('child_environment_sha256') == lifecycle.ENV_SHA
        and state.get('gpu_ids') == list(range(8)) and not state.get('cleanup_error')
        and state.get('cleanup_warnings') == [warning], 'v1 failed supervisor terminal changed')
    group = lifecycle.read_sealed(directory/'benchmark/process-group.json')
    require(group.get('schema') == 'canonical-quality-owned-process-group-v1'
        and group.get('status') == 'failed' and group.get('reason') == 'worker_failure_marker'
        and group.get('owned_ranks_still_alive_after_cleanup') == []
        and group.get('only_owned_process_handles_signaled') is True
        and len(group.get('workers', [])) == 8
        and {w.get('rank') for w in group['workers']} == set(range(8)), 'v1 process group not drained')
    for worker in group['workers']:
        require(worker.get('alive') is False and type(worker.get('exitcode')) is int
            and not Path('/proc',str(worker['pid'])).exists(), 'v1 worker still present')
    for pid in (state['child_pid'], 4022298):
        require(not Path('/proc',str(pid)).exists(), 'v1 child or warned process still present')
    outcome = lifecycle.read_sealed(paths['outcome'])
    after = lifecycle.read_sealed(paths['after'])
    require(outcome.get('successful') is False and outcome.get('automatic_retry') is False
        and outcome.get('supervisor_exit') == 1 and outcome.get('finished_unix')
        and outcome.get('holder_physically_verified') is True
        and outcome.get('holder_audit_sha256') == after['sha256'], 'v1 failure/restoration outcome incomplete')
    return dict(name=PREVIOUS_DYNAMIC, status='failed', previous_workers_exited=True,
        previous_scores_reused=False, original_cleanup_warnings=state['cleanup_warnings'],
        supervisor_file_sha256=lifecycle.file_digest(directory/'supervisor.json'),
        process_group_sha256=group['sha256'], outcome_sha256=outcome['sha256'],
        holder_after_sha256=after['sha256'])


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
        from gpu_pilot.run_tail_link_pilot import SCHEMA
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
        STAGE/'tail-link-protocol.json',inputs/'sensitivity-protocol.json')
    verify_plan(plan)
    if args.check_only:
        print(json.dumps(plan,sort_keys=True,separators=(',',':')));return 0
    outcome=run_plan(plan);print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome['successful'] else 1


if __name__=='__main__':raise SystemExit(main())
