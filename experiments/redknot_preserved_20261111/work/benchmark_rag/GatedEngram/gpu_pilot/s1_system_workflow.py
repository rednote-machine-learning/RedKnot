"""One detached experiment and one post-cleanup independent CPU audit; holder stays off."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
STAGE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(STAGE))
from gpu_pilot import launch_pilot as life
from gpu_pilot.s1_system_lifecycle import atomic,validate_plan,POLICY


def validate_final_audit(value,*,successful_runner):
    expected=[2,3,4];indices=value.get('completed_case_indices')
    life.require(value.get('schema')=='s1-system256-independent-readout-v1'
        and value.get('errors')==[] and value.get('planned_cases')==3
        and value.get('arms')==['recomputed','s1','s1_system']
        and value.get('population')=='256k_only' and value.get('other_128k_population_included') is False
        and type(indices) is list and indices==expected[:len(indices)]
        and value.get('complete_cases')==len(indices)
        and value.get('missing_case_indices')==expected[len(indices):]
        and value.get('all_completed_cases_audited') is True
        and value.get('qualification_passed') is True
        and value.get('system_metadata_qualification_passed') is True
        and value.get('system_exact_quality_preserved_for_all_completed_cases') is True,
        '256K final audit is not a qualified complete-case prefix')
    full=value.get('status')=='complete'
    life.require(full or value.get('status')=='partial_complete','invalid independent audit status')
    life.require((full and indices==expected) or (not full and len(indices)<3),
        'audit completion status contradicts the three-case inventory')
    life.require(all(value.get(k) is full for k in ('final_run_audited','overall_run_complete','all_registered_cases_audited')),
        'audit whole-run completion flags conflict')
    life.require(not successful_runner or full,'successful GPU run lacks the complete independent audit')
    return True


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args(argv)
    plan=validate_plan(args.plan);base=STAGE.parent;folder=base/'SystemWorkflow';folder.mkdir(exist_ok=False)
    result=Path(plan['result_dir']);record=dict(schema='s1-system-workflow-terminal-v1',plan_sha256=plan['sha256'],holder_policy=POLICY,started_unix=time.time(),status='failed')
    try:
        with (folder/'lifecycle.log').open('x') as log:
            child=subprocess.run([sys.executable,'-B',str(STAGE/'gpu_pilot/s1_system_lifecycle.py'),'--plan',str(args.plan)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        record['lifecycle_exit']=child.returncode
        terminal=life.read_sealed(result/'terminal.json','s1-system-idle-terminal-v1')
        life.require(terminal['plan_sha256']==plan['sha256'] and terminal['holder_policy']==POLICY,'lifecycle terminal belongs to another experiment')
        record['lifecycle_terminal_sha256']=terminal['sha256']
        life.require(terminal.get('gpu_idle_verified') is True and not terminal.get('cleanup_error'),'physical GPU cleanup must finish before audit')
        runner=result/'runner'
        if not (runner/'preflight.json').exists():raise ValueError('runner produced no preflight; no result audit to run')
        output=folder/'final-audit.json'
        argv=[sys.executable,'-B',str(STAGE/'gpu_pilot/read_s1_system256_results.py'),'--result-dir',str(runner),'--output',str(output)]
        stopped_by_deadline=terminal.get('termination_reason')=='gpu_deadline'
        if (result/'watchdog-stop.json').exists():
            stop=life.read_sealed(result/'watchdog-stop.json','s1-system-watchdog-stop-v1')
            life.require(stop['plan_sha256']==plan['sha256'],'watchdog stop belongs to another experiment')
            stopped_by_deadline=stopped_by_deadline or stop['reason']=='gpu_deadline'
        if child.returncode:argv+=['--allow-partial','--partial-reason','hard_deadline' if stopped_by_deadline else 'runner_failure']
        with (folder/'audit.log').open('x') as log:
            audit=subprocess.run(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        record['audit_exit']=audit.returncode
        life.require(audit.returncode==0 and output.is_file(),'independent CPU audit failed; preserve original evidence')
        value=life.read_sealed(output,'s1-system256-independent-readout-v1');record['audit_sha256']=value['sha256'];record['audit_status']=value.get('status')
        validate_final_audit(value,successful_runner=child.returncode==0)
        record['status']='complete' if child.returncode==0 and value.get('status')=='complete' else ('partial' if stopped_by_deadline and value.get('status')=='partial_complete' else 'failed')
    except BaseException as exc:record.update(error=repr(exc),traceback=traceback.format_exc())
    record['finished_unix']=time.time();atomic(folder/'terminal.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='traceback'}),flush=True)
    return 0 if record['status'] in ('complete','partial') else 1

if __name__=='__main__':raise SystemExit(main())
