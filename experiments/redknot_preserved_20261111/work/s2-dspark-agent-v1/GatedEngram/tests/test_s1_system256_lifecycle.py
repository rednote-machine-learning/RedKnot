"""Portable predecessor-admission and post-cleanup workflow failure regressions."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from gpu_pilot import s1_system_lifecycle as lifecycle
from gpu_pilot import s1_system_workflow as workflow

life=lifecycle.life


class PopulationFixture:
    def __init__(self,root,mutate=None):
        self.root=Path(root).resolve();self.stage=self.root/'GatedEngram';self.stage.mkdir()
        self.plan=dict(file_sha256={},stage_sources={name:life.digest(name) for name in lifecycle.SYSTEM_FILES})
        policy={'sampling':'greedy','tail':128}
        common=dict(schema='test-policy-v1',baseline_core_policy=policy,
            baseline_core_policy_sha256=life.digest(policy),quality={'natural_eos':True},timing={'repeats':3})
        self.protocol=self.write(self.stage/'s1-system256-protocol.json',dict(common,cases=[2,3,4],input_manifest_sha256='b'*64))
        identity={key:life.digest(key) for key in lifecycle.STABLE_IDENTITY_KEYS}
        identity.update(core_source_sha256={'core.py':'a'*64},core_policy_sha256=life.digest(policy),
            native_profile={'cp':1,'chunk':8192},fixed_env={'fixed':'same'},
            pilot_source_sha256={'GatedEngram/'+k:v for k,v in self.plan['stage_sources'].items()},
            s1_protocol_sha256=self.protocol['sha256'],execution_configuration_sha256='d'*64)
        self.plan.update(runtime_preflight=dict(status='cpu_preflight_passed',gpu_initialized=False,
            evaluation_cases=[dict(index=i) for i in (2,3,4)],identity=deepcopy(identity)),
            runner_argv=['python','-B','runner','--s1-protocol',str(self.stage/'s1-system256-protocol.json')],
            protocol_sha256=self.protocol['sha256'],manifest_sha256='b'*64)
        entries=[];previous=None
        for ordinal,indices in enumerate(([2,4,6],[8,10,12,14,16,18,20])):
            prefix=self.root/('old' if ordinal==0 else 'continuation');prefix.mkdir()
            registered=lifecycle.REGISTERED_128_INDICES if ordinal==0 else indices
            proto=life.seal(dict(common,cases=registered,input_manifest_sha256='c'*64))
            ident=deepcopy(identity);ident['s1_protocol_sha256']=proto['sha256'];ident['execution_configuration_sha256']=life.digest(ordinal)
            rows=[dict(index=i,input_tokens=128000,dataset='loong',question='Q'+str(i),references=['A'],
                input_ids_sha256=life.digest(['input',i]),documents=[{'token_ids_sha256':life.digest(['doc',i])}]) for i in registered]
            pre=dict(schema='kv-cache-engram-s1-system128-pilot-run-v1',identity=ident,protocol=proto,
                manifest_sha256='c'*64,evaluation_case_indices=registered,audit_cases=rows)
            cases=[];evidence={}
            for i in indices:
                row=next(r for r in rows if r['index']==i)
                cases.append(dict(index=i,**{k:row[k] for k in ('input_tokens','dataset','question','references')},
                    cache_blocks=row['documents'],document_content_sha256=[row['documents'][0]['token_ids_sha256']],
                    closure_sha256=life.digest(['pair',i]),system_quality_output_ids_equal_to_s1=True,
                    quality={arm:{'output_ids':[12,1]} for arm in lifecycle.ARMS},
                    online_ttft_samples_s={arm:[1.,1.1,1.2] for arm in lifecycle.ARMS}))
                evidence[f'case{i}.paired.json']=dict(file_sha256=life.digest(['file',i]),seal_sha256=cases[-1]['closure_sha256'])
            audit=dict(schema=('s1-system128-independent-readout-v1','s1-system128-continuation-independent-readout-v1')[ordinal],
                status='partial_complete' if ordinal==0 else 'complete',arms=lifecycle.ARMS,errors=[],
                qualification_passed=True,system_metadata_qualification_passed=True,
                system_exact_quality_preserved_for_all_completed_cases=True,planned_cases=len(registered),
                complete_cases=len(indices),completed_case_indices=indices,all_completed_cases_audited=True,
                final_run_audited=bool(ordinal),overall_run_complete=bool(ordinal),all_registered_cases_audited=False,
                all_execution_cases_audited=bool(ordinal),execution_configuration_sha256=ident['execution_configuration_sha256'],
                protocol_sha256=proto['sha256'],manifest_sha256='c'*64,
                frozen_sources_verified={'pilot_source_sha256':life.digest(ident['pilot_source_sha256'])},
                cases=cases,evidence=evidence,
                **{k:ident[k] for k in ('core_source_sha256','core_policy_sha256','native_snapshot_sources_sha256')})
            if previous:audit['predecessor_audit_sha256']=previous['sha256']
            terminal=dict(schema='s1-system-idle-terminal-v1',status='failed' if ordinal==0 else 'runner_complete_pending_independent_audit',
                plan_sha256=life.digest(['plan',ordinal]),holder_policy=lifecycle.POLICY,gpu_idle_verified=True)
            if mutate:mutate(ordinal,pre,audit,terminal)
            av=self.write(prefix/'audit.json',audit);pv=self.write(prefix/'preflight.json',pre);tv=self.write(prefix/'lifecycle.json',terminal)
            wv=self.write(prefix/'workflow.json',dict(schema='s1-system-workflow-terminal-v1',status='partial' if ordinal==0 else 'complete',
                plan_sha256=terminal['plan_sha256'],holder_policy=lifecycle.POLICY,
                audit_sha256=av['sha256'],lifecycle_terminal_sha256=tv['sha256']))
            entries.append(dict(run_id='original' if ordinal==0 else 'continuation',accepted_case_indices=indices,
                audit=self.ref(prefix/'audit.json',av),preflight=self.ref(prefix/'preflight.json',pv),
                workflow_terminal=self.ref(prefix/'workflow.json',wv),lifecycle_terminal=self.ref(prefix/'lifecycle.json',tv)))
            previous=av
        self.admission=self.write(self.root/'admission.json',dict(schema='s1-system128-population-admission-v1',
            status='complete',holder_policy=lifecycle.POLICY,registered_case_indices=lifecycle.REGISTERED_128_INDICES,
            completed_case_indices=lifecycle.REGISTERED_128_INDICES,source_run_count=2,single_run=False,audits=entries))
        self.plan['predecessor_population']=self.ref(self.root/'admission.json',self.admission)

    def write(self,path,value):
        path=Path(path).resolve()
        value=life.seal(value);Path(path).write_text(json.dumps(value)+'\n')
        self.plan['file_sha256'][str(path)]=life.file_digest(path)
        return value

    def ref(self,path,value):
        path=Path(path).resolve()
        return dict(path=str(path),file_sha256=life.file_digest(path),seal_sha256=value['sha256'])

    def validate(self):
        with patch.object(lifecycle,'STAGE',self.stage):
            return lifecycle.validate_predecessor_population(self.plan)


def final_audit(indices=(2,3,4)):
    complete=list(indices)==[2,3,4]
    return life.seal(dict(schema='s1-system256-independent-readout-v1',status='complete' if complete else 'partial_complete',
        errors=[],planned_cases=3,arms=lifecycle.ARMS,population='256k_only',other_128k_population_included=False,
        completed_case_indices=list(indices),complete_cases=len(indices),missing_case_indices=[i for i in (2,3,4) if i not in indices],
        all_completed_cases_audited=True,qualification_passed=True,system_metadata_qualification_passed=True,
        system_exact_quality_preserved_for_all_completed_cases=True,final_run_audited=complete,
        overall_run_complete=complete,all_registered_cases_audited=complete))


class PopulationAdmissionTests(unittest.TestCase):
    def test_real_file_refs_accept_two_complete_case_sets_and_different_256_manifest(self):
        with TemporaryDirectory() as root:
            fixture=PopulationFixture(root)
            self.assertEqual(fixture.validate()['completed_case_indices'],list(range(2,22,2)))

    def test_source_drift_rejected_even_when_all_seals_recomputed(self):
        def change(ordinal,pre,audit,terminal):
            if ordinal==1:pre['identity']['pilot_source_sha256']['GatedEngram/'+lifecycle.SYSTEM_FILES[0]]='f'*64
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'code or generation policy'):PopulationFixture(root,change).validate()

    def test_cleanup_failure_rejects_otherwise_complete_population(self):
        def change(ordinal,pre,audit,terminal):
            if ordinal==1:terminal['gpu_idle_verified']=False
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'cleanup'):PopulationFixture(root,change).validate()

    def test_resealed_partial_continuation_is_not_population_complete(self):
        def change(ordinal,pre,audit,terminal):
            if ordinal==1:audit['status']='partial_complete'
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'every remaining'):PopulationFixture(root,change).validate()

    def test_quality_mismatch_rejected_despite_summary_claim(self):
        def change(ordinal,pre,audit,terminal):
            if ordinal==1:audit['cases'][0]['quality']['s1_system']['output_ids']=[99,1]
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'three-arm case'):PopulationFixture(root,change).validate()

    def test_missing_repeat_rejected_without_rerunning_reader(self):
        def change(ordinal,pre,audit,terminal):
            if ordinal==0:audit['cases'][0]['online_ttft_samples_s']['recomputed']=[1.,2.]
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'three-arm case'):PopulationFixture(root,change).validate()

    def test_byte_drift_and_absent_plan_pin_rejected(self):
        with TemporaryDirectory() as root:
            fixture=PopulationFixture(root);path=Path(fixture.admission['audits'][0]['audit']['path'])
            original=path.read_bytes();path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'not frozen'):fixture.validate()
            path.write_bytes(original);del fixture.plan['file_sha256'][str(path)]
            with self.assertRaisesRegex(ValueError,'not frozen'):fixture.validate()

    def test_duplicate_contribution_cannot_hide_behind_complete_top_level(self):
        with TemporaryDirectory() as root:
            fixture=PopulationFixture(root);value=deepcopy(fixture.admission)
            value['audits'][1]['accepted_case_indices']=[6,8,10,12,14,16,18]
            value.pop('sha256');value=fixture.write(Path(root)/'admission.json',value)
            fixture.plan['predecessor_population']=fixture.ref(Path(root)/'admission.json',value)
            with self.assertRaisesRegex(ValueError,'disjoint'):fixture.validate()

    def test_new_window_is_relative_24_hour_emergency_only(self):
        self.assertEqual(lifecycle.release_deadlines({'timeout_seconds':86400,'absolute_deadline_unix':None},200.,300.),(86600.,86700.))


class WorkflowTests(unittest.TestCase):
    def test_actual_reader_completion_emitter_matches_workflow_gate(self):
        from gpu_pilot.read_s1_system256_results import completion_state
        for count in (0,1,2,3):
            results=[{'index':i,'shared_bank_identity':{'bank_generation':i}} for i in (2,3,4)[:count]]
            state=completion_state(results,[],complete=count==3,allow_partial=count<3,
                partial_reason='hard_deadline' if count<3 else None)
            audit=final_audit(tuple(c['index'] for c in results));audit.update(state)
            self.assertTrue(workflow.validate_final_audit(audit,successful_runner=count==3))

    def test_final_audit_complete_and_honest_prefix(self):
        self.assertTrue(workflow.validate_final_audit(final_audit(),successful_runner=True))
        self.assertTrue(workflow.validate_final_audit(final_audit((2,)),successful_runner=False))
        with self.assertRaises(ValueError):workflow.validate_final_audit(final_audit((2,)),successful_runner=True)
        bad=final_audit((2,));bad['overall_run_complete']=True
        with self.assertRaises(ValueError):workflow.validate_final_audit(bad,successful_runner=False)

    def run_workflow(self,root,*,exitcode=0,idle=True,deadline=False,indices=(2,3,4)):
        root=Path(root).resolve();stage=root/'GatedEngram';stage.mkdir();result=root/'result';result.mkdir()
        (result/'runner').mkdir();(result/'runner/preflight.json').write_text('{}')
        plan={'result_dir':str(result),'sha256':'a'*64}
        lifecycle.atomic(result/'terminal.json',dict(schema='s1-system-idle-terminal-v1',holder_policy=lifecycle.POLICY,
            plan_sha256=plan['sha256'],gpu_idle_verified=idle,termination_reason='gpu_deadline' if deadline else 'runner_failure'))
        calls=[]
        def invoke(argv,**kwargs):
            calls.append((argv,kwargs))
            if 'read_s1_system256_results.py' in argv[2]:
                Path(argv[argv.index('--output')+1]).write_text(json.dumps(final_audit(indices)))
                return SimpleNamespace(returncode=0)
            return SimpleNamespace(returncode=exitcode)
        with patch.object(workflow,'STAGE',stage),patch.object(workflow,'validate_plan',return_value=plan),patch.object(workflow.subprocess,'run',side_effect=invoke):
            rc=workflow.main(['--plan',str(root/'plan.json')])
        return rc,life.read_sealed(root/'SystemWorkflow/terminal.json'),calls

    def test_success_runs_one_cpu_audit_after_idle(self):
        with TemporaryDirectory() as root:
            rc,terminal,calls=self.run_workflow(root)
            self.assertEqual((rc,terminal['status'],len(calls)),(0,'complete',2))
            self.assertEqual(calls[1][1]['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertNotIn('--allow-partial',calls[1][0])

    def test_cleanup_failure_never_starts_audit(self):
        with TemporaryDirectory() as root:
            rc,terminal,calls=self.run_workflow(root,exitcode=1,idle=False)
            self.assertEqual((rc,terminal['status'],len(calls)),(1,'failed',1))

    def test_deadline_prefix_is_partial_but_runtime_failure_stays_failed(self):
        for deadline,status,rc in ((True,'partial',0),(False,'failed',1)):
            with self.subTest(deadline=deadline),TemporaryDirectory() as root:
                actual,terminal,calls=self.run_workflow(root,exitcode=1,deadline=deadline,indices=(2,))
                self.assertEqual((actual,terminal['status'],len(calls)),(rc,status,2))
                self.assertEqual(calls[1][0][-2:],['--partial-reason','hard_deadline' if deadline else 'runner_failure'])

    def test_cpu_validator_isolated_help_imports_no_gpu_runtime(self):
        path=Path(__file__).resolve().parents[1]/'validate_s1_system256_cpu.py'
        with TemporaryDirectory() as root:
            result=subprocess.run([sys.executable,'-I','-B','-X','importtime',str(path),'--help'],cwd=root,text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        imports=[x.rsplit('|',1)[-1].strip() for x in result.stderr.splitlines() if x.startswith('import time:')]
        self.assertFalse(any(x=='torch' or x.startswith('torch.') for x in imports))


if __name__=='__main__':unittest.main()
