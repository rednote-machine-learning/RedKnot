"""Fresh natural Loong multi-document RAG S1 comparison under the original owner lifecycle.

No model/GPU imports or holder operations in check-only. Never retries, signals
unknown processes, mutates snapshots, or substitutes old performance results.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE))
from gpu_pilot import launch_pilot as lifecycle
from gpu_pilot.s1_protocol import EVALUATION_INDICES, DATASETS, LENGTH_BANDS, TARGET_INPUT_TOKENS
from gpu_pilot.launch_index_hit20_pilot import snapshot_sources

ROOT=lifecycle.REMOTE_ROOT
REMOTE_STAGE = Path('/mnt/tidal-alsh01/dataset/redone/RedKnot_DeepInfer_20261028/DeepInfer-PR148/experiments/kce_loong_rag64_128_20261109.hwndc9gp/GatedEngram')
NAME='kce-loong-rag64-128-20261109-v1'
PREDECESSOR='kce-longcontext64-128-20261109-v1'
PREDECESSOR_WORKFLOW_SHA='619d4f651047332a6748ca20e1a245d38fa9639b23f55b2493ea4b70170a2ef3'
PREDECESSOR_AUDIT_SHA='3559f8d8a6dcbb01952b0bf094bc10ee324526b5799128017f9bc4175e1b3a15'
PREDECESSOR_HOLDER_SHA='a6745e018bfd7d3a7efa4d2d167e8bb2caa82d97fa7d1ac22367c7fe506e63f5'
# Bind the actual sealed 37-sample predecessor observation and duration.
# This does not change the new holder verifier or reuse prior measurements.
PREDECESSOR_HOLDER_ELAPSED_S=180.03817653656006
PROC_ROOT=Path('/proc')
WORKFLOW_SCHEMA='kce-loong-s1-launch-v1'
TIMEOUT=7200
PREPARE_TIMEOUT=300
require=lifecycle.require
INPUT_COMPANIONS=("source-records.json","selection-plan.json","tokenization-evidence.json")


def protect_git_environment(env):
    """Keep native source-identity reads from refreshing the frozen Git index."""
    require(not env.get('GIT_CONFIG_PARAMETERS'), 'unsupported inherited Git parameters')
    raw=env.get('GIT_CONFIG_COUNT','0')
    require(isinstance(raw,str) and raw.isascii() and raw.isdigit(), 'invalid Git configuration count')
    count=int(raw)
    for i in range(count):
        require(isinstance(env.get(f'GIT_CONFIG_KEY_{i}'),str)
            and isinstance(env.get(f'GIT_CONFIG_VALUE_{i}'),str), 'incomplete Git configuration')
    if not (count and env[f'GIT_CONFIG_KEY_{count-1}'].lower()=='diff.autorefreshindex'
            and env[f'GIT_CONFIG_VALUE_{count-1}']=='false'):
        env[f'GIT_CONFIG_KEY_{count}']='diff.autoRefreshIndex'
        env[f'GIT_CONFIG_VALUE_{count}']='false'
        count+=1
    env['GIT_CONFIG_COUNT']=str(count)
    env['GIT_OPTIONAL_LOCKS']='0'


def supervisor_environment(root):
    additions=lifecycle.read_json(Path(root)/'pr148-native-env.json')
    require(additions.get('GIT_OPTIONAL_LOCKS','0')=='0'
        and not any(k in ('GIT_CONFIG_COUNT','GIT_CONFIG_PARAMETERS')
                    or k.startswith(('GIT_CONFIG_KEY_','GIT_CONFIG_VALUE_')) for k in additions),
        'native child environment must not override read-only Git configuration')
    env=dict(os.environ)
    protect_git_environment(env)
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env


def predecessor_evidence(root):
    return verify_previous_fullmodel_terminal(root)


def verify_cpu_proof(plan):
    proof=lifecycle.read_sealed(plan['cpu_proof_path'])
    ids=proof.get('test_ids',[])
    require(proof.get('schema') == 'kv-cache-engram-s1-cpu-proof-v1'
        and proof.get('successful') is True and proof.get('cuda_initialized') is False
        and proof.get('cuda_init_guard_installed') is True
        and proof.get('failures')==0 and proof.get('errors')==0 and proof.get('skipped')==0
        and proof.get('source_sha256')==plan['stage_sources']
        and type(proof.get('tests')) is int and proof['tests']>=1000
        and len(ids)==len(set(ids))==proof['tests']
        and all(any(family in name for name in ids) for family in (
            'test_s1_runner.','test_s1_results.','test_launch_s1.','test_s1_runtime.',
            'test_s1_cache_attention.','test_token_divergence.','test_loong_results.','test_loong_metrics.',
            'test_loong_inputs.','test_musique30_s1_geometry.','test_index_hit20_empty_ep_qualification.')),
        'new complete CPU proof missing, stale, failed or skipped')


def verify_core_identity(sources,native_sha,protocol):
    require(all(sources.get(name)==sha for name,sha in protocol['core_source_sha256'].items())
        and native_sha==protocol['native_snapshot_sources_sha256'],
        'six frozen S1 core modules or private native snapshot changed')


def build_plan(root,stage,manifest,pilot_protocol,comparison_protocol):
    from gpu_pilot.s1_protocol import s1_protocol
    root,stage=lifecycle.absolute_no_symlink(root),lifecycle.absolute_no_symlink(stage)
    require(stage.name == 'GatedEngram' and root in stage.parents,
            'independent comparison stage must remain inside authorized experiment root')
    manifest,pilot_protocol,comparison_protocol=(lifecycle.absolute_no_symlink(p)
        for p in (manifest,pilot_protocol,comparison_protocol))
    require(len({manifest,pilot_protocol,comparison_protocol}) == 3
        and all(stage in p.parents for p in (manifest,pilot_protocol,comparison_protocol)),
        'three independent input/protocol files required inside stage')
    data=lifecycle.read_sealed(manifest)
    inputs=lifecycle.read_sealed(pilot_protocol,'loong-input-protocol-v1')
    declared=lifecycle.read_sealed(comparison_protocol)
    require(declared == s1_protocol()
        and declared['input_manifest_sha256'] == data['sha256']
        and data.get('schema') == 'di-sg-real-rag-manifest-v1'
        and data.get('artifact_kind') == 'loong_inputs_v1'
        and data.get('contract_schema') == 'kv-cache-engram-loong-input-manifest-v1'
        and data.get('input_protocol_sha256') == inputs['sha256']
        and 'sensitivity_protocol_sha256' not in data, 'independent Loong multi-document RAG inputs/protocol differ')
    paths=lifecycle.evidence_paths(root,NAME);lifecycle.require_unused(paths)
    sources=lifecycle.source_files(stage)
    required={'validate_s1_cpu.py','gpu_pilot/run_s1_pilot.py','gpu_pilot/read_s1_results.py',
        'gpu_pilot/s1_protocol.py','gpu_pilot/loong_inputs.py','gpu_pilot/loong_metrics.py','gpu_pilot/token_divergence.py',
        'gpu_pilot/launch_s1_pilot.py','kv_cache_engram/s1_runtime.py'}
    require(required <= set(sources),'new comparison sources missing')
    native=stage.parent/'NativeSnapshot';native_files=snapshot_sources(native)
    verify_core_identity(sources,lifecycle.digest(native_files),declared)
    ref=lambda p,v:dict(path=str(p),sha256=v['sha256'],file_sha256=lifecycle.file_digest(p))
    companions={name:lifecycle.file_digest(manifest.parent/name) for name in INPUT_COMPANIONS}
    created=time.time()
    body=dict(schema=lifecycle.PLAN_SCHEMA,workflow_schema=WORKFLOW_SCHEMA,name=NAME,
        root=str(root),stage_dir=str(stage),created_unix=created,no_automatic_retry=True,
        preserves_paused_experiments=True,gpu_ids=list(range(8)),timeout=TIMEOUT,prepare_timeout=PREPARE_TIMEOUT,
        timeout_purpose='two-hour GPU execution bound; original supervisor restores holder after completion or timeout',
        timeout_origin='original supervisor GPU release',absolute_deadline_unix=None,
        completion_case_count=len(EVALUATION_INDICES),document_count=10,
        paths={k:str(v) for k,v in paths.items()},protocol=ref(pilot_protocol,inputs),manifest=ref(manifest,data),
        sensitivity_protocol=None,s1_protocol=ref(comparison_protocol,declared),
        input_companion_files_sha256=companions,stage_sources=sources,stage_sources_sha256=lifecycle.digest(sources),
        pinned_lifecycle={str(lifecycle.SUPERVISOR):lifecycle.SUPERVISOR_SHA,
            str(lifecycle.HELPER):lifecycle.HELPER_SHA,str(root/'pr148-native-env.json'):lifecycle.ENV_SHA},
        native_snapshot=str(native),native_snapshot_sources=native_files,
        native_snapshot_sources_sha256=lifecycle.digest(native_files),predecessor_evidence=predecessor_evidence(root),
        core_source_sha256=declared['core_source_sha256'],core_policy_sha256=declared['core_policy_sha256'],
        evaluation_indices=list(EVALUATION_INDICES),evaluation_arms=['recomputed','s1'],
        evaluation_datasets=list(DATASETS),
        length_band_order=[LENGTH_BANDS[i%len(LENGTH_BANDS)] for i in range(len(EVALUATION_INDICES))],
        target_input_tokens=dict(TARGET_INPUT_TOKENS),
        compact_start_layer=3,cpu_proof_path=str(stage/'S1_CPU_TESTS-v1.json'),
        native_yoco_swa_share_hidden=False,first_full_layers=[0,1,2],tail_layers=list(range(3,40)),
        head_tokens=0,tail_tokens=128,swa_halo_tokens=128,natural_boundary_inputs=True,native_first3_source2_preserved=True,selected_retained_through_layer39=True,
        decode_native=True,s1_offline_gpu_bank=True,all_online_preparation_in_ttft=True,
        fresh_all_cases_both_arms=True,cache_storage_policy='same_s1_offline_gpu_bank_resident_both_arms',
        new_native_and_same_policy_parity_required=True,real_variable_length_ten_block_smoke_required=True,
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
        '--child-env-json',str(root/'pr148-native-env.json'),'--timeout',str(plan['timeout']),
        '--prepare-timeout',str(PREPARE_TIMEOUT),'--gpus','0,1,2,3,4,5,6,7','--',
        str(stage/'gpu_pilot/run_s1_pilot.py'),*args,
        '--s1-protocol',plan['s1_protocol']['path']]


def verify_plan(plan):
    from gpu_pilot.s1_protocol import s1_protocol as protocol_factory
    require(plan.get('sha256') == lifecycle.digest({k:v for k,v in plan.items() if k != 'sha256'})
        and plan.get('schema') == lifecycle.PLAN_SCHEMA, 'comparison launch plan seal/schema differs')
    require(lifecycle.source_files(Path(plan['stage_dir'])) == plan['stage_sources'], 'stage changed after freeze')
    for key in ('manifest','protocol'):
        item=plan[key]
        require(lifecycle.file_digest(item['path']) == item['file_sha256']
            and lifecycle.read_sealed(item['path'])['sha256'] == item['sha256'], key+' changed after freeze')
    require(lifecycle.read_sealed(plan['protocol']['path']).get('schema') == 'loong-input-protocol-v1'
        and plan.get('sensitivity_protocol') is None, 'old pilot/sensitivity protocol cannot govern natural Loong multi-document RAG cases')
    for name,sha in plan['input_companion_files_sha256'].items():
        require(name in INPUT_COMPANIONS and lifecycle.file_digest(Path(plan['manifest']['path']).parent/name) == sha,
                'input companion changed after freeze')
    require(set(plan['input_companion_files_sha256']) == set(INPUT_COMPANIONS),'missing source companion binding')
    require(plan.get('workflow_schema')==WORKFLOW_SCHEMA and plan.get('name')==NAME
        and plan.get('timeout')==TIMEOUT and plan.get('prepare_timeout')==PREPARE_TIMEOUT
        and plan.get('absolute_deadline_unix') is None
        and not any(k in plan for k in ('hard_deadline_unix','stop_request_unix','holder_restore_reserve_seconds'))
        and plan.get('timeout_origin') == 'original supervisor GPU release' and plan.get('document_count') == 10
        and plan.get('completion_case_count') == len(EVALUATION_INDICES)
        and plan.get('fresh_all_cases_both_arms') is True
        and plan.get('cache_storage_policy') == 'same_s1_offline_gpu_bank_resident_both_arms'
        and plan.get('timeout_purpose') == 'two-hour GPU execution bound; original supervisor restores holder after completion or timeout'
        and plan.get('evaluation_indices')==list(EVALUATION_INDICES)
        and plan.get('evaluation_arms')==['recomputed','s1']
        and plan.get('evaluation_datasets')==list(DATASETS)
        and plan.get('length_band_order')==[LENGTH_BANDS[i%len(LENGTH_BANDS)] for i in range(len(EVALUATION_INDICES))]
        and plan.get('target_input_tokens')==TARGET_INPUT_TOKENS
        and plan.get('compact_start_layer')==3
        and plan.get('native_yoco_swa_share_hidden') is False
        and plan.get('first_full_layers')==[0,1,2] and plan.get('tail_layers')==list(range(3,40))
        and plan.get('head_tokens')==0 and plan.get('tail_tokens')==plan.get('swa_halo_tokens')==128
        and plan.get('natural_boundary_inputs') is True
        and all(plan.get(k) is True for k in ('native_first3_source2_preserved','selected_retained_through_layer39',
            'decode_native','s1_offline_gpu_bank','all_online_preparation_in_ttft',
            'new_native_and_same_policy_parity_required','real_variable_length_ten_block_smoke_required',
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
        ('manifest','protocol','s1_protocol')),
        'inputs/protocols must remain inside independent stage')
    require(plan['native_snapshot']==str(native) and snapshot_sources(native)==plan['native_snapshot_sources']
        and lifecycle.digest(plan['native_snapshot_sources'])==plan['native_snapshot_sources_sha256'],
        'private native snapshot changed')
    protocol=plan['s1_protocol']
    require(lifecycle.file_digest(protocol['path'])==protocol['file_sha256']
        and lifecycle.read_sealed(protocol['path'])==protocol_factory(),'new tail protocol changed')
    verify_core_identity(plan['stage_sources'],plan['native_snapshot_sources_sha256'],protocol_factory())
    require(plan.get('core_source_sha256')==protocol_factory()['core_source_sha256']
        and plan.get('core_policy_sha256')==protocol_factory()['core_policy_sha256'],'algorithm freeze evidence differs')
    argv=plan['argv']
    require(argv==expected_argv(plan),'complete supervisor/runner command changed')
    require(predecessor_evidence(Path(plan['root']))==plan['predecessor_evidence'],'previous evidence changed')
    require(plan.get('cpu_proof_path') == str(stage/'S1_CPU_TESTS-v1.json'), 'new headtail CPU proof required')
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
        and state.get('runner_sha256')==plan['stage_sources']['gpu_pilot/run_s1_pilot.py'],
        'original supervisor did not report successful finite worker and cleanup')






def verify_previous_fullmodel_terminal(root):
    """Bind the latest complete6 run and its verified physical restoration."""
    previous=Path(root)/"DeepInfer-PR148/experiments/kce_longcontext64_128_20261109.qzae9zix"
    terminal=lifecycle.read_sealed(previous/"Workflow/terminal.json")
    audit=lifecycle.read_sealed(previous/"Workflow/final-audit.json")
    holder=lifecycle.read_sealed(previous/"Workflow/holder-live-verification.json")
    require(terminal.get("sha256")==PREDECESSOR_WORKFLOW_SHA and terminal.get("status")=="complete"
        and terminal.get("experiment_name")==PREDECESSOR and terminal.get("holder_physically_verified") is True
        and terminal.get("launcher_successful") is True and terminal.get("supervisor_exit")==0
        and terminal.get("final_audit_sha256")==audit.get("sha256")==PREDECESSOR_AUDIT_SHA
        and audit.get("status")=="complete" and audit.get("errors")==[]
        and audit.get("complete_cases")==terminal.get("complete_cases")==6
        and audit.get("planned_cases")==terminal.get("planned_cases")==6
        and audit.get("completed_case_indices")==terminal.get("completed_case_indices")==list(range(2,8))
        and all(audit.get(k) is True for k in ('final_run_audited','overall_run_complete','all_registered_cases_audited')),
        "latest complete6 run is not independently closed")
    require(holder.get("sha256")==PREDECESSOR_HOLDER_SHA
        and terminal.get("holder_live_verification_sha256")==holder["sha256"]
        and holder.get("status")=="passed" and holder.get("experiment_contexts_present") is False
        and holder.get("elapsed_s")==PREDECESSOR_HOLDER_ELAPSED_S
        and len(holder.get("samples",[]))==37,
        "latest holder physical restoration proof differs")
    for proc in PROC_ROOT.iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/"cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError as exc:raise ValueError("cannot verify previous process absence") from exc
        require(not any(str(previous).encode() in item or str(Path(root)/"results"/PREDECESSOR/"benchmark").encode() in item for item in argv),
                "previous experiment or reader still active")
    return dict(name=PREDECESSOR,workflow_terminal_sha256=terminal["sha256"],final_audit_sha256=audit["sha256"],
        holder_live_verification_sha256=holder["sha256"],previous_workers_exited=True,previous_scores_reused=False,
        holder_observation_elapsed_s=holder["elapsed_s"],holder_sample_count=len(holder["samples"]),
        prior_status="complete",completed_case_indices=list(range(2,8)))


def verify_previous_workflow_terminal(root):
    return verify_previous_fullmodel_terminal(root)


def run_plan(plan):
    protect_git_environment(os.environ)
    verify_plan(plan)
    root,stage=Path(plan['root']),Path(plan['stage_dir'])
    require(root==ROOT and stage==REMOTE_STAGE and STAGE==REMOTE_STAGE,
        'execute only from authorized new remote stage')
    lifecycle.verify_lifecycle(root)
    verify_previous_fullmodel_terminal(root)
    child_env=supervisor_environment(root)
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
                stderr=subprocess.STDOUT,start_new_session=True,env=child_env)
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
        from gpu_pilot.run_s1_pilot import SCHEMA
        require(final.get('schema')==SCHEMA and final.get('status')=='complete'
            and final.get('completed_case_indices')==list(EVALUATION_INDICES),'all fresh Loong multi-document RAG two-arm cases not complete')
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
    protect_git_environment(os.environ)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv)
    inputs=STAGE/'inputs-loong-v1'
    plan=build_plan(ROOT,STAGE,inputs/'manifest.json',inputs/'input-protocol.json',
        STAGE/'s1-protocol.json')
    verify_plan(plan)
    if args.check_only:
        print(json.dumps(plan,sort_keys=True,separators=(',',':')));return 0
    outcome=run_plan(plan);print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome['successful'] else 1


if __name__=='__main__':raise SystemExit(main())
