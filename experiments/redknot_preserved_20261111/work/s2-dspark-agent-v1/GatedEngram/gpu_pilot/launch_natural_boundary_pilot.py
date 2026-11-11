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
REMOTE_STAGE = Path('/mnt/tidal-alsh01/dataset/redone/RedKnot_DeepInfer_20261028/DeepInfer-PR148/experiments/kce_natural_boundary_20261106.mgrex1l_/GatedEngram')
NAME='kce-natural-boundary-three-arm-20261106-v1'
PREDECESSOR='kce-headtail-three-arm-20case-20261106-v1'
PREDECESSOR_RUNNER_SHA='8d7bb87e6e4766b31f303af6b7ad7df5bdc77d69bb01c2a20f74af734177cd1b'
STOP_REQUEST_SHA='8d7c45ffee82c4a22ca5f8b47feb6d51b86394cc2693b9016b4eac2b650560a5'
PREDECESSOR_WORKFLOW_SHA='1707f2e6834c6311f11cdd96b98f385f5fd2fa9ed709197746cc35838f60b605'
PREDECESSOR_HOLDER_SHA='84b07974ebb7639d50eb7d1fa7f1684606d3c53612c5fc07774e5459a6b6cf97'
PREDECESSOR_RECEIPT_SHA='eb0b9b9b2bc0c64420c20c0661de500b4c4d93d068516b410867c1c76afdcada'
PREDECESSOR_RECEIPT_PATH=STAGE.parent/'predecessor-evidence.json'
PROC_ROOT=Path('/proc')
WORKFLOW_SCHEMA='kce-natural-boundary-launch-v1'
TIMEOUT=86400
PREPARE_TIMEOUT=300
require=lifecycle.require
INPUT_COMPANIONS=("source-records.json","selection-plan.json","tokenization-evidence.json")


def predecessor_evidence(root):
    return verify_previous_fullmodel_terminal(root)


def verify_cpu_proof(plan):
    proof=lifecycle.read_sealed(plan['cpu_proof_path'])
    ids=proof.get('test_ids',[])
    require(proof.get('schema') == 'kv-cache-engram-natural-boundary-cpu-proof-v1'
        and proof.get('successful') is True and proof.get('cuda_initialized') is False
        and proof.get('cuda_init_guard_installed') is True
        and proof.get('failures')==0 and proof.get('errors')==0 and proof.get('skipped')==0
        and proof.get('source_sha256')==plan['stage_sources']
        and type(proof.get('tests')) is int and proof['tests']>=1000
        and len(ids)==len(set(ids))==proof['tests']
        and all(any(family in name for name in ids) for family in (
            'test_headtail_link_plan.','test_headtail_comparison.','test_natural_boundary_runner.',
            'test_natural_boundary_results.','test_launch_natural_boundary.','test_natural_boundary_inputs.',
            'test_index_hit20_empty_ep_qualification.','test_early_token_diagnostics.',
            'test_natural_boundary_controller.')),
        'new complete CPU proof missing, stale, failed or skipped')


def build_plan(root,stage,manifest,pilot_protocol,comparison_protocol):
    from gpu_pilot.natural_boundary_protocol import natural_boundary_protocol
    root,stage=lifecycle.absolute_no_symlink(root),lifecycle.absolute_no_symlink(stage)
    require(stage.name == 'GatedEngram' and root in stage.parents,
            'independent comparison stage must remain inside authorized experiment root')
    manifest,pilot_protocol,comparison_protocol=(lifecycle.absolute_no_symlink(p)
        for p in (manifest,pilot_protocol,comparison_protocol))
    require(len({manifest,pilot_protocol,comparison_protocol}) == 3
        and all(stage in p.parents for p in (manifest,pilot_protocol,comparison_protocol)),
        'three independent input/protocol files required inside stage')
    data=lifecycle.read_sealed(manifest)
    inputs=lifecycle.read_sealed(pilot_protocol,'natural-boundary-input-protocol-v1')
    declared=lifecycle.read_sealed(comparison_protocol)
    require(declared == natural_boundary_protocol()
        and declared['input_manifest_sha256'] == data['sha256']
        and data.get('schema') == 'di-sg-real-rag-manifest-v1'
        and data.get('artifact_kind') == 'natural_boundary_inputs_v1'
        and data.get('contract_schema') == 'kv-cache-engram-natural-boundary-input-manifest-v1'
        and data.get('input_protocol_sha256') == inputs['sha256']
        and 'sensitivity_protocol_sha256' not in data, 'independent twenty-case inputs/protocol differ')
    paths=lifecycle.evidence_paths(root,NAME);lifecycle.require_unused(paths)
    sources=lifecycle.source_files(stage)
    required={'validate_natural_boundary_cpu.py','gpu_pilot/run_natural_boundary_pilot.py','gpu_pilot/read_natural_boundary_results.py',
        'gpu_pilot/natural_boundary_protocol.py','gpu_pilot/natural_boundary_inputs.py',
        'gpu_pilot/launch_natural_boundary_pilot.py','kv_cache_engram/headtail_comparison.py',
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
        sensitivity_protocol=None,natural_boundary_protocol=ref(comparison_protocol,declared),
        input_companion_files_sha256=companions,stage_sources=sources,stage_sources_sha256=lifecycle.digest(sources),
        pinned_lifecycle={str(lifecycle.SUPERVISOR):lifecycle.SUPERVISOR_SHA,
            str(lifecycle.HELPER):lifecycle.HELPER_SHA,str(root/'pr148-native-env.json'):lifecycle.ENV_SHA},
        native_snapshot=str(native),native_snapshot_sources=native_files,
        native_snapshot_sources_sha256=lifecycle.digest(native_files),predecessor_evidence=predecessor_evidence(root),
        evaluation_indices=list(range(2,22)),evaluation_arms=['recomputed','tail128_link','headtail128_link'],
        dynamic_refresh_start_layer=3,cpu_proof_path=str(stage/'NATURAL_BOUNDARY_CPU_TESTS-v1.json'),
        native_yoco_swa_share_hidden=False,first_full_layers=[0,1,2],tail_layers=list(range(3,40)),
        head_tokens=128,tail_tokens=128,natural_boundary_inputs=True,global_kv_complete_native_online=True,native_yoco_keep140=True,
        decode_native=True,full_canonical_gpu_bank=True,all_online_preparation_in_ttft=True,
        fresh_all_twenty_cases_all_three_arms=True,cache_storage_policy='shared_gpu_full_canonical_bank_all_arms',
        new_native_and_same_policy_parity_required=True,real_variable_length_eight_block_smoke_required=True,
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
        str(stage/'gpu_pilot/run_natural_boundary_pilot.py'),*args,
        '--natural-boundary-protocol',plan['natural_boundary_protocol']['path']]


def verify_plan(plan):
    from gpu_pilot.natural_boundary_protocol import natural_boundary_protocol as protocol_factory
    require(plan.get('sha256') == lifecycle.digest({k:v for k,v in plan.items() if k != 'sha256'})
        and plan.get('schema') == lifecycle.PLAN_SCHEMA, 'comparison launch plan seal/schema differs')
    require(lifecycle.source_files(Path(plan['stage_dir'])) == plan['stage_sources'], 'stage changed after freeze')
    for key in ('manifest','protocol'):
        item=plan[key]
        require(lifecycle.file_digest(item['path']) == item['file_sha256']
            and lifecycle.read_sealed(item['path'])['sha256'] == item['sha256'], key+' changed after freeze')
    require(lifecycle.read_sealed(plan['protocol']['path']).get('schema') == 'natural-boundary-input-protocol-v1'
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
        and plan.get('natural_boundary_inputs') is True
        and all(plan.get(k) is True for k in ('global_kv_complete_native_online','native_yoco_keep140',
            'decode_native','full_canonical_gpu_bank','all_online_preparation_in_ttft',
            'new_native_and_same_policy_parity_required','real_variable_length_eight_block_smoke_required',
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
        ('manifest','protocol','natural_boundary_protocol')),
        'inputs/protocols must remain inside independent stage')
    require(plan['native_snapshot']==str(native) and snapshot_sources(native)==plan['native_snapshot_sources']
        and lifecycle.digest(plan['native_snapshot_sources'])==plan['native_snapshot_sources_sha256'],
        'private native snapshot changed')
    protocol=plan['natural_boundary_protocol']
    require(lifecycle.file_digest(protocol['path'])==protocol['file_sha256']
        and lifecycle.read_sealed(protocol['path'])==protocol_factory(),'new tail protocol changed')
    argv=plan['argv']
    require(argv==expected_argv(plan),'complete supervisor/runner command changed')
    require(predecessor_evidence(Path(plan['root']))==plan['predecessor_evidence'],'previous evidence changed')
    require(plan.get('cpu_proof_path') == str(stage/'NATURAL_BOUNDARY_CPU_TESTS-v1.json'), 'new headtail CPU proof required')
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
        and state.get('runner_sha256')==plan['stage_sources']['gpu_pilot/run_natural_boundary_pilot.py'],
        'original supervisor did not report successful finite worker and cleanup')


def verify_predecessor_receipt(receipt, root):
    """Bind the authenticated cancellation to real terminal restoration evidence."""
    root = Path(root)
    require(receipt.get('schema') == 'natural-boundary-predecessor-restoration-v1'
        and receipt.get('sha256') == PREDECESSOR_RECEIPT_SHA
        and receipt['sha256'] == lifecycle.digest({k:v for k,v in receipt.items() if k!='sha256'})
        and receipt.get('prior_experiment') == PREDECESSOR
        and receipt.get('termination_reason') == 'user_replaced_artificial_cohort'
        and receipt.get('all_old_owned_pids_absent') is True
        and receipt.get('only_restored_holder_contexts') is True,
        'authenticated user cancellation/restoration receipt differs')
    artifacts = receipt['artifacts']
    require(set(artifacts) == {'stop_request','supervisor','launch_plan','supervisor_started',
        'launch_outcome','workflow_terminal','holder_on_error'}, 'predecessor artifact closure differs')
    values = {key:item['value'] for key,item in artifacts.items()}
    for key,value in values.items():
        if key != 'supervisor':
            require(value.get('sha256') == lifecycle.digest({k:v for k,v in value.items() if k!='sha256'}),
                    'predecessor sealed artifact corrupt: '+key)
    plan, started, marker = (values[key] for key in ('launch_plan','supervisor_started','stop_request'))
    previous_stage = Path(plan['stage_dir'])
    require(plan.get('schema') == lifecycle.PLAN_SCHEMA and plan.get('name') == PREDECESSOR
        and plan.get('root') == str(root) and previous_stage.is_absolute()
        and previous_stage.name == 'GatedEngram' and root/'DeepInfer-PR148/experiments' in previous_stage.parents
        and plan.get('stage_sources',{}).get('gpu_pilot/run_headtail_scale20_pilot.py') == PREDECESSOR_RUNNER_SHA,
        'previous launch location/source is not authenticated')
    require(started.get('schema') == 'tail-link-original-supervisor-started-v1'
        and started.get('plan_sha256') == plan['sha256'] and started.get('argv') == plan.get('argv')
        and marker.get('schema') == 'user-authorized-natural-context-transition-v1'
        and marker.get('sha256') == STOP_REQUEST_SHA and marker.get('old_results_preserved') is True
        and all(marker.get('identity',{}).get(key) == started.get(key)
                for key in ('pid','start_ticks','pgid','sid','argv')),
        'stop request is not bound to the authenticated original supervisor')
    state, terminal, holder, outcome = (values[key] for key in
        ('supervisor','workflow_terminal','holder_on_error','launch_outcome'))
    require(state.get('schema') == 'redknot-glm53-gpu-lifecycle-v1'
        and state.get('status') == 'failed' and state.get('finished_unix')
        and state.get('holder_restored') is True and state.get('runner_sha256') == PREDECESSOR_RUNNER_SHA
        and state.get('supervisor_sha256') == lifecycle.SUPERVISOR_SHA
        and state.get('holder_helper_sha256') == lifecycle.HELPER_SHA
        and state.get('child_environment_sha256') == lifecycle.ENV_SHA
        and state.get('gpu_ids') == list(range(8)) and not state.get('cleanup_error')
        and state.get('error') == "KeyboardInterrupt('received signal 15')",
        'original supervisor has not closed the user-requested cancellation with holder restoration')
    require(terminal.get('schema') == 'headtail-scale20-workflow-v1'
        and terminal.get('sha256') == PREDECESSOR_WORKFLOW_SHA
        and terminal.get('experiment_name') == PREDECESSOR and terminal.get('status') == 'failed'
        and terminal.get('finished_at') and not terminal.get('holder_error')
        and terminal.get('holder_on_error_sha256') == holder.get('sha256') == PREDECESSOR_HOLDER_SHA,
        'previous workflow has not terminated with physical holder verification')
    require(outcome.get('successful') is False and outcome.get('supervisor_started') is True
        and outcome.get('holder_physically_verified') is True and outcome.get('plan_sha256') == plan['sha256']
        and outcome.get('finished_unix') and not outcome.get('holder_error'),
        'previous launch outcome lacks physical restoration')
    pids = receipt.get('old_owned_pids')
    require(type(pids) is list and len(pids) == len(set(pids)) == 21
        and all(type(pid) is int and pid > 1 for pid in pids)
        and started['pid'] in pids and state['child_pid'] in pids, 'owned-process absence coverage incomplete')
    holder_pids = holder.get('metadata',{}).get('worker_pids')
    require(type(holder_pids) is list and len(set(holder_pids)) == 8
        and len(holder.get('contexts',[])) == 8 and len(receipt.get('contexts',[])) == 8
        and {c['pid'] for c in holder['contexts']} == set(holder_pids)
        and {(c['pid'],c['uuid']) for c in holder['contexts']}
            == {(c['pid'],c['uuid']) for c in receipt['contexts']}
        and not set(holder_pids).intersection(pids), 'restored holder/old process identity overlap')
    result = root/'results'/PREDECESSOR
    expected_paths = dict(stop_request=root/'results'/(PREDECESSOR+'.natural4k-stop-request.json'),
        supervisor=result/'supervisor.json', launch_plan=root/'results'/(PREDECESSOR+'.launch-plan.json'),
        supervisor_started=root/'results'/(PREDECESSOR+'.supervisor-started.json'),
        launch_outcome=root/'results'/(PREDECESSOR+'.launch-outcome.json'),
        workflow_terminal=previous_stage.parent/'Workflow/terminal.json',
        holder_on_error=previous_stage.parent/'Workflow/holder-on-error.json')
    require(all(item.get('path') == str(expected_paths[key]) for key,item in artifacts.items()),
        'predecessor evidence path escaped original experiment')
    return values


def verify_previous_fullmodel_terminal(root):
    """Recheck immutable stop evidence and actual absence; do not require a successful old run."""
    receipt = lifecycle.read_sealed(PREDECESSOR_RECEIPT_PATH)
    values = verify_predecessor_receipt(receipt, root)
    for key,item in receipt['artifacts'].items():
        require(lifecycle.file_digest(item['path']) == item['file_sha256']
            and lifecycle.read_json(item['path']) == item['value'], 'previous artifact changed: '+key)
    for pid in receipt['old_owned_pids']:
        require(not (PROC_ROOT/str(pid)).exists(), 'previous owned PID is present: '+str(pid))
    previous_stage = Path(values['launch_plan']['stage_dir'])
    benchmark = Path(root)/'results'/PREDECESSOR/'benchmark'
    for proc in PROC_ROOT.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError):
            continue
        except PermissionError as exc:
            raise ValueError('cannot inspect process ownership for new admission') from exc
        require(not any(str(previous_stage).encode() in arg or str(benchmark).encode() in arg for arg in argv),
                'previous experiment or final audit process still active')
    return dict(name=PREDECESSOR, previous_workers_exited=True, previous_scores_reused=False,
        user_requested_cancellation=True, predecessor_restoration_receipt_sha256=receipt['sha256'],
        stop_request_sha256=STOP_REQUEST_SHA, workflow_terminal_sha256=PREDECESSOR_WORKFLOW_SHA,
        holder_on_error_sha256=PREDECESSOR_HOLDER_SHA,
        supervisor_file_sha256=receipt['artifacts']['supervisor']['file_sha256'],
        previous_launch_plan_sha256=values['launch_plan']['sha256'],
        outcome_sha256=values['launch_outcome']['sha256'],
        supervisor_started_sha256=values['supervisor_started']['sha256'],
        old_owned_pids=receipt['old_owned_pids'], prior_status=values['supervisor']['status'])


def verify_previous_workflow_terminal(root):
    return verify_previous_fullmodel_terminal(root)


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
        from gpu_pilot.run_natural_boundary_pilot import SCHEMA
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
    inputs=STAGE/'inputs-natural-v1'
    plan=build_plan(ROOT,STAGE,inputs/'manifest.json',inputs/'input-protocol.json',
        STAGE/'natural-boundary-protocol.json')
    verify_plan(plan)
    if args.check_only:
        print(json.dumps(plan,sort_keys=True,separators=(',',':')));return 0
    outcome=run_plan(plan);print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome['successful'] else 1


if __name__=='__main__':raise SystemExit(main())
