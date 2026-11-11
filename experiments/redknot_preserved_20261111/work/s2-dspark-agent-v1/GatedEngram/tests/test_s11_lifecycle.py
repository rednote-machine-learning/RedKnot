"""Admission contracts and failure preservation only; no process/GPU operations."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from gpu_pilot import s11_lifecycle as lifecycle
from gpu_pilot import s11_workflow as workflow
life=lifecycle.life


class Fixture:
    def __init__(self,root,mutate=None):
        self.root=Path(root).resolve();self.stage=self.root/'GatedEngram';self.stage.mkdir()
        self.plan=dict(file_sha256={},stage_sources={p:life.digest(p) for p in lifecycle.SYSTEM_FILES})
        policy={'tail':128,'sampling':'greedy'}
        common=dict(schema='fixture-policy',baseline_core_policy=policy,baseline_core_policy_sha256=life.digest(policy),
            quality={'natural_eos':True},timing={'concurrency':1,'warmups_per_arm':1,'repeats':3,'output_tokens':32})
        identity={key:life.digest(key) for key in lifecycle.STABLE_IDENTITY_KEYS}
        identity.update(core_source_sha256={'core.py':'a'*64},core_policy_sha256=life.digest(policy),
            native_profile={'cp':1,'chunk':8192},fixed_env={'fixed':'same'},
            pilot_source_sha256={'GatedEngram/'+k:v for k,v in self.plan['stage_sources'].items()})
        self.entries=[];previous=None;row20=None
        for ordinal,indices in enumerate(([2,4,6],[8,10,12,14,16,18])):
            folder=self.root/str(ordinal);folder.mkdir();registered=list(range(2 if ordinal==0 else 8,22,2))
            protocol=life.seal(dict(common,cases=registered,input_manifest_sha256='c'*64))
            ident=dict(identity,s1_protocol_sha256=protocol['sha256'],execution_configuration_sha256=life.digest(ordinal))
            rows=[dict(index=i,input_tokens=128000,dataset='loong',sample_id='q'+str(i),question='Q'+str(i),references=['A'],
                input_ids_sha256=life.digest(['ids',i]),documents=[{'token_ids_sha256':life.digest(['doc',i])}]) for i in registered]
            row20=rows[-1]
            pre=dict(schema='kv-cache-engram-s1-system128-pilot-run-v1',identity=ident,protocol=protocol,
                manifest_sha256='c'*64,evaluation_case_indices=registered,audit_cases=rows)
            cases=[];evidence={}
            for i in indices:
                row=next(r for r in rows if r['index']==i)
                cases.append(dict(index=i,**{k:row[k] for k in ('input_tokens','dataset','question','references')},
                    cache_blocks=row['documents'],document_content_sha256=[row['documents'][0]['token_ids_sha256']],
                    closure_sha256=life.digest(['pair',i]),system_quality_output_ids_equal_to_s1=True,
                    quality={arm:{'output_ids':[12,1]} for arm in lifecycle.ARMS},
                    online_ttft_samples_s={arm:[1.,1.1,1.2] for arm in lifecycle.ARMS}))
                evidence[f'case{i}.paired.json']={'file_sha256':life.digest(['file',i]),'seal_sha256':cases[-1]['closure_sha256']}
            audit=dict(schema=('s1-system128-independent-readout-v1','s1-system128-continuation-independent-readout-v1')[ordinal],
                status='partial_complete',arms=lifecycle.ARMS,errors=[],qualification_passed=True,system_metadata_qualification_passed=True,
                system_exact_quality_preserved_for_all_completed_cases=True,planned_cases=len(registered),complete_cases=len(indices),
                completed_case_indices=indices,all_completed_cases_audited=True,final_run_audited=False,overall_run_complete=False,
                all_registered_cases_audited=False,all_execution_cases_audited=False,partial_completion_reason='runner_failure' if ordinal else 'hard_deadline',
                execution_configuration_sha256=ident['execution_configuration_sha256'],protocol_sha256=protocol['sha256'],manifest_sha256='c'*64,
                frozen_sources_verified={'pilot_source_sha256':life.digest(ident['pilot_source_sha256'])},cases=cases,evidence=evidence,
                **{k:ident[k] for k in ('core_source_sha256','core_policy_sha256','native_snapshot_sources_sha256')})
            if previous:audit['predecessor_audit_sha256']=previous['sha256']
            owner=dict(pid=123,ppid=100,start_ticks=456,pgid=123,sid=123,argv=['exact','frozen','runner'])
            gpu=''.join(f'{i}, GPU-{i}, L20X, 0, 0\n' for i in range(8))
            terminal=dict(schema='s1-system-idle-terminal-v1',status='failed',plan_sha256=life.digest(['plan',ordinal]),
                holder_policy=lifecycle.POLICY,gpu_idle_verified=True,child=owner,supervisor=dict(owner,pid=100,ppid=99),
                runner_exit=-15,termination_reason='runner_failure',released_unix=100.,finished_unix=210.,
                after_snapshots=[dict(time_unix=t,gpu=gpu,compute_apps='',idle=True) for t in (202.,203.,204.)])
            if mutate:mutate(ordinal,pre,audit,terminal)
            av=self.write(folder/'audit.json',audit);pv=self.write(folder/'preflight.json',pre);tv=self.write(folder/'life.json',terminal)
            wv=self.write(folder/'workflow.json',dict(schema='s1-system-workflow-terminal-v1',status='partial' if ordinal==0 else 'failed',
                plan_sha256=terminal['plan_sha256'],holder_policy=lifecycle.POLICY,audit_sha256=av['sha256'],lifecycle_terminal_sha256=tv['sha256']))
            self.entries.append(dict(run_id=str(ordinal),accepted_case_indices=indices,audit=self.ref(folder/'audit.json',av),
                preflight=self.ref(folder/'preflight.json',pv),workflow_terminal=self.ref(folder/'workflow.json',wv),
                lifecycle_terminal=self.ref(folder/'life.json',tv)))
            previous=av
        stop=self.write(self.root/'stop.json',dict(schema='s1-system-user-directed-stop-v1',
            action='SIGTERM_authenticated_runner_leader_only_supervisor_owns_cleanup',holder_policy=lifecycle.POLICY,
            source_or_result_modified=False,plan_sha256=terminal['plan_sha256'],child=terminal['child'],
            supervisor=terminal['supervisor'],requested_unix=200.,completed_case_files=[f'case{i}.paired.json' for i in (8,10,12,14,16,18)]))
        self.admission=self.write(self.root/'admission.json',dict(schema='s11-predecessor-admission-v1',status='accepted_user_stop',
            holder_policy=lifecycle.POLICY,registered_case_indices=list(range(2,22,2)),completed_case_indices=list(range(2,20,2)),
            pending_case_indices=[20],new_evaluation_case_indices=[20,2,3,4],source_run_count=2,single_run=False,
            user_requested_stop_authenticated=True,physical_idle_verified=True,audits=self.entries,
            user_stop_request=self.ref(self.root/'stop.json',stop),physical_idle=self.entries[-1]['lifecycle_terminal']))
        manifest=self.write(self.stage/'manifest.json',dict(schema='fixture-manifest-v1',cohort=[row20]))
        current=self.write(self.stage/'s11-protocol.json',dict(common,cases=[20,2,3,4],input_manifest_sha256=manifest['sha256']))
        self.plan.update(predecessor_admission=self.ref(self.root/'admission.json',self.admission),
            runtime_preflight=dict(status='cpu_preflight_passed',gpu_initialized=False,evaluation_cases=[{'index':i} for i in (20,2,3,4)],
                identity=dict(identity,s1_protocol_sha256=current['sha256'],execution_configuration_sha256='d'*64)),
            runner_argv=['python','-B','runner','--s1-protocol',str(self.stage/'s11-protocol.json'),'--manifest',str(self.stage/'manifest.json')],
            protocol_sha256=current['sha256'],manifest_sha256=manifest['sha256'])

    def write(self,path,value):
        value=life.seal({k:v for k,v in value.items() if k!='sha256'});path.write_text(json.dumps(value)+'\n')
        self.plan['file_sha256'][str(path)]=life.file_digest(path);return value
    def ref(self,path,value):return dict(path=str(path),file_sha256=life.file_digest(path),seal_sha256=value['sha256'])
    def validate(self):
        with patch.object(lifecycle,'STAGE',self.stage):return lifecycle.validate_predecessor_admission(self.plan)


def audit_value(indices,status):
    full=status=='complete';expected=[20,2,3,4]
    return dict(schema='s11-independent-readout-v1',status=status,arms=['recomputed','s1_system'],errors=[],planned_cases=4,
        population='remaining128_and_fresh256',historical_completed_cases_included=False,completed_case_indices=indices,
        complete_cases=len(indices),missing_case_indices=[i for i in expected if i not in indices],all_completed_cases_audited=True,
        qualification_passed=True,system_metadata_qualification_passed=True,retired_original_s1_formal_generated=False,
        same_policy_numerical_qualification_required=True,final_run_audited=full,overall_run_complete=full,all_registered_cases_audited=full)


class S11LifecycleTests(unittest.TestCase):
    def test_two_partial_predecessors_accept_exact_nine_cases_and_preserve_failed_terminal(self):
        with TemporaryDirectory() as folder:
            f=Fixture(folder);value=f.validate();self.assertEqual(value['pending_case_indices'],[20])
            terminal=life.read_sealed(f.entries[1]['workflow_terminal']['path'])
            self.assertEqual(terminal['status'],'failed')

    def test_missing_gate_or_resealed_failed_quality_cannot_become_valid_predecessor(self):
        for change in (lambda _,p,a,t:a.update(errors=['actual failure']),
                       lambda _,p,a,t:a.update(qualification_passed=False),
                       lambda _,p,a,t:a['cases'][0]['quality']['s1_system'].update(output_ids=[13,1]),
                       lambda _,p,a,t:t.update(gpu_idle_verified=False)):
            with TemporaryDirectory() as folder:
                f=Fixture(folder,change)
                with self.assertRaises(ValueError):f.validate()
        with TemporaryDirectory() as folder:
            f=Fixture(folder);f.plan['predecessor_admission']['seal_sha256']='0'*64
            with self.assertRaises(ValueError):f.validate()

    def test_real_stop_identity_interval_and_physical_contexts_are_required(self):
        for mutate in (lambda t:t.update(runner_exit=1),lambda t:t['after_snapshots'][0].update(compute_apps='999, GPU-0, inference'),
                       lambda t:t['after_snapshots'][0].update(gpu=''),lambda t:t.update(released_unix=201.),
                       lambda t:t['after_snapshots'][0].update(time_unix=199.)):
            with TemporaryDirectory() as folder:
                f=Fixture(folder,lambda i,p,a,t:mutate(t) if i==1 else None)
                with self.assertRaises(ValueError):f.validate()
        with TemporaryDirectory() as folder:
            f=Fixture(folder);path=Path(f.admission['user_stop_request']['path']);stop=life.read_sealed(path)
            stop['child']['start_ticks']+=1;stop=f.write(path,stop)
            f.admission['user_stop_request']=f.ref(path,stop)
            f.admission=f.write(Path(f.plan['predecessor_admission']['path']),f.admission)
            f.plan['predecessor_admission']=f.ref(Path(f.plan['predecessor_admission']['path']),f.admission)
            with self.assertRaisesRegex(ValueError,'authenticate'):f.validate()

    def test_fixed_emergency_clock_from_new_release_has_no_old_absolute_deadline(self):
        self.assertEqual(lifecycle.release_deadlines(dict(timeout_seconds=86400,absolute_deadline_unix=None),100.,200.),(86500.,86600.))

    def test_final_audit_requires_both_arms_and_honest_complete_prefix(self):
        self.assertTrue(workflow.validate_final_audit(audit_value([20,2,3,4],'complete'),successful_runner=True))
        self.assertTrue(workflow.validate_final_audit(audit_value([20],'partial_complete'),successful_runner=False))
        for mutate in (lambda a:a.update(arms=['recomputed','s1','s1_system']),lambda a:a.update(complete_cases=4),
                       lambda a:a.update(overall_run_complete=True),lambda a:a.update(errors=['bad']),
                       lambda a:a.update(retired_original_s1_formal_generated=True),lambda a:a.update(completed_case_indices=[2])):
            a=audit_value([20],'partial_complete');mutate(a)
            with self.assertRaises(ValueError):workflow.validate_final_audit(a,successful_runner=False)
        with self.assertRaises(ValueError):workflow.validate_final_audit(audit_value([20],'partial_complete'),successful_runner=True)

    def test_workflow_audits_only_after_verified_idle_and_never_turns_runtime_failure_into_success(self):
        for idle,deadline,expected_calls,expected_status in ((False,False,1,'failed'),(True,False,2,'failed'),(True,True,2,'partial')):
            with self.subTest(idle=idle,deadline=deadline),TemporaryDirectory() as folder:
                root=Path(folder).resolve();stage=root/'GatedEngram';stage.mkdir();result=root/'result';(result/'runner').mkdir(parents=True)
                (result/'runner'/'preflight.json').write_text('{}')
                plan=life.seal(dict(result_dir=str(result)));terminal=life.seal(dict(schema='s1-system-idle-terminal-v1',
                    plan_sha256=plan['sha256'],holder_policy=lifecycle.POLICY,gpu_idle_verified=idle,
                    termination_reason='gpu_deadline' if deadline else 'runner_failure'))
                (result/'terminal.json').write_text(json.dumps(terminal))
                calls=[]
                def execute(argv,**kwargs):
                    calls.append(argv)
                    if len(calls)==2:
                        self.assertEqual(Path(argv[2]).name,'read_s11_results.py')
                        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                        value=life.seal(audit_value([20],'partial_complete'))
                        Path(argv[argv.index('--output')+1]).write_text(json.dumps(value))
                    return SimpleNamespace(returncode=1 if len(calls)==1 else 0)
                with patch.object(workflow,'STAGE',stage),patch.object(workflow,'validate_plan',return_value=plan),\
                     patch.object(workflow.subprocess,'run',side_effect=execute):
                    code=workflow.main(['--plan',str(root/'plan.json')])
                value=life.read_sealed(root/'SystemWorkflow'/'terminal.json')
                self.assertEqual(len(calls),expected_calls);self.assertEqual(value['status'],expected_status)
                self.assertEqual(code,0 if expected_status=='partial' else 1)


if __name__=='__main__':unittest.main()
