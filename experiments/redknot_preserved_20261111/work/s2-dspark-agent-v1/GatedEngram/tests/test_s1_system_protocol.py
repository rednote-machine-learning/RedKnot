"""Three-arm scope, exact output parity and independent readout admission."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.core import digest
from gpu_pilot import s1_system_protocol as protocol
from gpu_pilot import run_s1_system_pilot as runner
from gpu_pilot import read_s1_system_results as reader
from tests.test_s1_runner import phase as original_phase
from tests.test_s1_results import seal,fixture as original_fixture


def phase(arm='s1',repeat=0,quality=False,tokens=32):
    value=original_phase('s1' if arm=='s1_system' else arm,repeat,quality,tokens)
    value.update(schema=runner.PHASE_SCHEMA,arm=arm,s1_policy=arm in protocol.POLICY_ARMS,
        **protocol.phase_execution(arm),**protocol.phase_policy(arm))
    return seal(value)


def execution_fields(value,*,enabled=False,group=1):
    value=copy.deepcopy(value);arm='s1_system' if enabled else 's1'
    value.update(schema=runner.PHASE_SCHEMA,arm=arm,**protocol.phase_execution(arm,batch_group_chunks=group))
    control=value['controller']
    control.update(system_enabled=enabled,batch_group_chunks=group,system_policy_changes=False,
        system_kernel_order_changed=group>1,system_original_per_chunk_fingerprints=True,
        system_numerically_qualified=True if enabled else None,system_batch_execution=None,system_batch_step_trace=[])
    control['cache_report']['system_global_update_diagnostics']=dict(
        schema='s1-system-global-update-diagnostics-v1',enabled=enabled,
        baseline_global_update_unchanged=not enabled,original_global_event_schema_preserved=True,
        full_raw_rows_event_is_logical_chunk_extent=True,
        numerical_qualification_uses_original_full_current_oracle=True,
        native_paged_store_and_compressor_state_kernels_unchanged=True,events=[])
    control['attention_report']['system_optimization']=dict(schema='s1-system-attention-diagnostics-v1',
        enabled=enabled,preparation_within_online_phase=True,cross_phase_device_reuse=False,
        offline_bank_mutated=False,original_projection_and_joint_attention=True,
        original_previous127_and_final_ring128=True,baseline_dispatches_original_implementation=not enabled,
        timing_measurements=None)
    value.update(system_batch_schedule=None,
        system_batch_configuration_receipts=[dict(schema='s1-system-worker-disabled-v1',rank=i,enabled=False) for i in range(8)],
        system_batch_worker_reports=[dict(schema='s1-system-worker-execution-v1',rank=i,enabled=False,steps=[]) for i in range(8)])
    return seal(value)


def world_traces(schedule):
    """Actual CPU classifier emitter with synthetic scheduler observations."""
    from kv_cache_engram.s1_system_runtime import classify_world_step
    traces=[[] for _ in range(8)];completed=0
    phases=['idle']
    for _ in schedule['steps']:phases+=['prefill','idle']
    phases+=['decode']
    for ordinal,kind in enumerate(phases):
        seqs=[]
        if kind=='prefill':
            step=schedule['steps'][completed]
            seqs=[dict(request_id=schedule['request_id'],is_prompt=True,
                seqlen_q=step['end']-step['start'],seqlen_k=step['end'])]
        elif kind=='decode':
            seqs=[dict(request_id=schedule['request_id'],is_prompt=False,
                seqlen_q=1,seqlen_k=schedule['full_input_tokens']+1)]
        samples=[dict(rank=rank,schedule_sha256=schedule['sha256'],ordinal=ordinal,
            prefill_completed=completed,prompt_run=kind=='prefill',
            is_empty=rank>0 or not seqs,sequences=seqs if rank==0 else []) for rank in range(8)]
        event=classify_world_step(samples,schedule,ordinal,completed)
        for rank,trace in enumerate(traces):
            trace.append(dict(event,local_prompt_run=samples[rank]['prompt_run'],
                local_is_empty=samples[rank]['is_empty'],model_execution_completed=True,
                **({'scheduler_observations':samples} if rank==0 else {})))
        if kind=='prefill':completed+=1
    return traces


class SystemProtocolTests(unittest.TestCase):
    @staticmethod
    def metadata_reports():
        # Contract fixture only; actual CUDA execution is a required runtime gate.
        identity=dict(execution_configuration_sha256='a'*64,checkpoint_index_sha256='b'*64,
            pilot_source_sha256={
                'GatedEngram/kv_cache_engram/s1_system_batch.py':'c'*64,
                'GatedEngram/kv_cache_engram/s1_system_runtime.py':'d'*64})
        report=dict(schema='s1-system-cuda-metadata-qualification-v1',passed=True,
            cuda_exercised=True,sync_debug_mode_during=2,sync_debug_mode_before=0,
            sync_debug_mode_after=0,gpu_numerical_parity_claim=False,
            checks=[dict(name='inactive_empty_cu',passed=True,rows=0,cu=[0,0]),
                dict(name='active_queue_group2',passed=True,input_tokens=32769,member_chunks=5,
                    deep_groups=3,empty_groups=1,selected_rows=128),
                dict(name='active_queue_group4',passed=True,input_tokens=32769,member_chunks=5,
                    deep_groups=2,empty_groups=0,selected_rows=128)])
        return identity,[dict(rank=i,operator_report=seal(copy.deepcopy(report))) for i in range(8)]

    def test_metadata_producer_reader_binds_eight_cuda_ranks_and_original_guard(self):
        identity,reports=self.metadata_reports()
        result=runner.metadata_qualification_report(reports,identity)
        self.assertIs(reader.validate_metadata_qualification(result,identity),result)
        self.assertFalse(result['gpu_numerical_parity_claim'])
        self.assertFalse(result['timing_valid']);self.assertFalse(result['quality_eligible'])
        for before in (1,2):
            changed=copy.deepcopy(reports)
            for item in changed:
                item['operator_report'].update(sync_debug_mode_before=before,sync_debug_mode_after=before)
                item['operator_report']=seal(item['operator_report'])
            runner.metadata_qualification_report(changed,identity)

    def test_metadata_reader_rejects_resealed_missing_ranks_guard_and_coverage(self):
        identity,reports=self.metadata_reports()
        mutators=(lambda x:x.pop(),lambda x:x[7].update(rank=0),
            lambda x:x[0]['operator_report'].update(cuda_exercised=False),
            lambda x:x[0]['operator_report'].update(sync_debug_mode_during=0),
            lambda x:x[0]['operator_report'].update(sync_debug_mode_after=2),
            lambda x:x[0]['operator_report']['checks'][1].update(empty_groups=0),
            lambda x:x[0]['operator_report']['checks'][2].update(deep_groups=3),
            lambda x:x[0]['operator_report'].update(gpu_numerical_parity_claim=True))
        for mutate in mutators:
            with self.subTest(mutation=mutate):
                changed=copy.deepcopy(reports);mutate(changed)
                for item in changed:item['operator_report']=seal(item['operator_report'])
                with self.assertRaises(ValueError):runner.metadata_qualification_report(changed,identity)
        value=runner.metadata_qualification_report(reports,identity)
        value['operator_source_sha256']['GatedEngram/kv_cache_engram/s1_system_batch.py']='e'*64
        with self.assertRaises(ValueError):reader.validate_metadata_qualification(seal(value),identity)

    def test_phase_requires_actual_metadata_gate_seal(self):
        qualification=dict(empty_ep_qualification_sha256='a'*64,system_metadata_qualification_sha256='b'*64)
        value=dict(empty_ep_qualification_sha256='a'*64,system_metadata_qualification_sha256='b'*64,
            controller=dict(empty_ep_qualification_sha256='a'*64))
        reader.empty_ep_binding(value,qualification)
        missing=dict(value);missing.pop('system_metadata_qualification_sha256')
        for bad in (missing,dict(value,system_metadata_qualification_sha256='c'*64)):
            with self.assertRaises(ValueError):reader.empty_ep_binding(bad,qualification)

    def test_missing_qualification_writes_sealed_failure_not_partial_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=seal(dict(schema=runner.SCHEMA,status='failed',rank=1,
                error='decode before final batch flush',traceback='actual exception text',completed_indices=[]))
            failure=root/'failure.rank1.json';failure.write_text(json.dumps(raw))
            output=root/'failed-audit.json'
            with patch.object(reader,'audit',side_effect=FileNotFoundError('s1_empty_chunk_qualification.json')):
                exitcode=reader.main(['--result-dir',str(root),'--output',str(output),
                    '--allow-partial','--partial-reason','runner_failure'])
            value=json.loads(output.read_text());reader.sealed(value)
            self.assertEqual(exitcode,1);self.assertEqual(value['status'],'invalid')
            self.assertEqual(value['failure_stage'],'qualification_incomplete')
            self.assertEqual(value['complete_cases'],0);self.assertFalse(value['formal_results_claimed'])
            self.assertFalse(value['qualification_passed']);self.assertFalse(value['overall_run_complete'])
            self.assertIn('s1_empty_chunk_qualification.json',value['missing_required_evidence'])
            item=value['runner_failures'][0];self.assertTrue(item['seal_verified'])
            self.assertEqual(item['sha256'],raw['sha256'])
            self.assertEqual(item['file_sha256'],hashlib.sha256(failure.read_bytes()).hexdigest())
            self.assertEqual(json.loads(failure.read_text()),raw)
            original=output.read_bytes()
            with patch.object(reader,'audit',side_effect=ValueError('still invalid')):
                with self.assertRaises(FileExistsError):reader.main(['--result-dir',str(root),'--output',str(output)])
            self.assertEqual(output.read_bytes(),original)

    def test_corrupt_failure_evidence_never_promoted_to_verified_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);value=seal(dict(schema=runner.SCHEMA,status='failed',rank=2,error='first'))
            value['error']='modified';(root/'failure.rank2.json').write_text(json.dumps(value))
            (root/'case2.paired.json').write_text('{}')
            result=reader.failed_audit(root,ValueError('qualification failed'))
            reader.sealed(result)
            self.assertFalse(result['runner_failures'][0]['seal_verified'])
            self.assertEqual(result['observed_paired_files'],['case2.paired.json'])
            self.assertFalse(result['observed_paired_files_are_audited_results'])
            self.assertEqual(result['cases'],[]);self.assertIsNone(result['summary'])
            self.assertEqual(len(result['errors']),2)

    def test_original_forty_layer_four_mhc_sparse_gate_is_preserved(self):
        data,ref=original_fixture(reference=True);_,candidate=original_fixture()
        ref=execution_fields(ref);candidate=execution_fields(candidate)
        result=reader.sparse_identity(ref,candidate,data,'a'*64)
        self.assertTrue(result['passed']);self.assertEqual(result['arm'],'s1')
        bad=copy.deepcopy(candidate);bad['arm']='s1_system';bad=seal(bad)
        with self.assertRaises(ValueError):reader.sparse_identity(ref,bad,data,'a'*64)
        bad=copy.deepcopy(candidate)
        bad['controller']['sparse_identity_fingerprints'][0]['value']['lazy']['pre_layer_mix']['sha256']='b'*64
        bad['controller']['sparse_identity_fingerprints_sha256']=digest(bad['controller']['sparse_identity_fingerprints'])
        with self.assertRaises(ValueError):reader.sparse_identity(ref,seal(bad),data,'a'*64)

    def test_actual_schedule_and_worker_report_emitters_bind_to_independent_reader(self):
        # CPU receipts and supplied loop traces exercise the real emitters; this
        # fixture makes no assertion that a CUDA kernel has actually executed.
        from types import SimpleNamespace as NS
        from kv_cache_engram.s1_system_batch import batch_schedule,validate_schedule_agreement
        from kv_cache_engram.s1_system_runtime import WorkerBatchSchedule
        from kv_cache_engram.s1_plan import plan
        data,value=original_fixture();value=execution_fields(value,enabled=True,group=4)
        control=value['controller'];value['record']['request_id']='fixture-request'
        docs=[dict(start=a,end=b,key=k) for a,b,k in reader.documents(data)]
        schedule=batch_schedule(docs,data['input_tokens'],request_id='fixture-request',phase=value['name'],
            phase_generation=control['phase_generation'],input_ids_sha256=data['input_ids_sha256'],group_chunks=4)
        workers=[];receipts=[]
        for rank in range(8):
            worker=object.__new__(WorkerBatchSchedule)
            worker.closed=False;worker._phase_done=True;worker.rank=rank
            worker.controller=NS(batch_schedule=schedule,state='RUNNING',_batch_step_trace=[])
            receipts.append(worker.configure(schedule));workers.append(worker)
        agreement=validate_schedule_agreement(schedule,receipts)
        group_events=[];all_selected=reader.position_oracle(reader.documents(data),data['input_tokens'])['selected_positions']
        begin=0
        for j,step in enumerate(schedule['steps']):
            if not step['flush']:continue
            members=schedule['steps'][begin:j+1];start=members[0]['start'];end=step['end']
            chosen=[p for p in all_selected if start<=p<end]
            event=dict(schema='s1-system-deep-batch-v1',schedule_sha256=schedule['sha256'],
                phase_generation=control['phase_generation'],start=start,end=end,
                member_chunks=[(x['start'],x['end']) for x in members],selected_positions=chosen,selected_rows=len(chosen),
                first3_native_calls=len(members),deep_layer_calls=37,deep_layer_calls_are_planned_not_measured=True,
                all_rank_installation=agreement,original_engram14_fetches_consumed=len(members),
                no_unselected_deep_rows=True,numerical_parity_claim=False,
                sampler_output_rows=len(plan(docs,data['input_tokens'],step['start'],end)['sampler_output_positions']))
            group_events.append(seal(event));begin=j+1
        reports=[]
        world=world_traces(schedule)
        for rank,worker in enumerate(workers):
            trace=[]
            for step in schedule['steps']:
                event={k:step[k] for k in ('start','end','flush','layers')}
                event.update(deep_native_ep_calls=37 if step['flush'] else 0,
                    empty_deep_native_ep_calls=37 if rank>0 and step['flush'] else 0)
                if rank>0:event.update(original_placeholder_rows=1,original_engram14_fetch_consumed=True)
                trace.append(event)
            worker.trace=trace;worker.controller._batch_step_trace=trace;worker.world_trace=world[rank]
            reports.append(worker.finish_phase())
        control['system_batch_execution']=dict(schedule=schedule,groups=group_events,
            all_rank_installation=agreement,numerical_parity_claim=False)
        control['system_batch_step_trace']=reports[0]['steps']
        value.update(system_batch_schedule=schedule,system_batch_configuration_receipts=receipts,
            system_batch_worker_reports=reports)
        self.assertTrue(reader.audit_system_execution(value,data))
        summary=reader.audit_world_scheduler(reports,schedule)
        self.assertEqual(summary['actual_prefill_steps'],len(schedule['steps']))
        self.assertEqual(summary['actual_idle_steps'],len(schedule['steps'])+1)
        for mutate in (lambda x:x['system_batch_worker_reports'][7]['steps'].pop(),
                       lambda x:x['system_batch_configuration_receipts'][4].update(rank=0),
                       lambda x:x['system_batch_worker_reports'][3].update(actual_deep_ep_calls=0),
                       lambda x:x['system_batch_worker_reports'][2]['world_step_trace'][0].update(world_phase='prefill'),
                       lambda x:x['system_batch_worker_reports'][0]['world_step_trace'][0]['scheduler_observations'][7].update(is_empty=False),
                       lambda x:x['system_batch_worker_reports'][0]['world_step_trace'][1].update(scheduler_observations_sha256='b'*64)):
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(ValueError):reader.audit_system_execution(bad,data)
    def test_full128k_subset_and_frozen_algorithm_policy(self):
        p=protocol.s1_system_protocol()
        self.assertEqual(p['cases'],list(range(2,22,2)))
        self.assertEqual(p['arms'],['recomputed','s1','s1_system'])
        self.assertEqual(p['length_band_order'],['128k']*10)
        self.assertEqual(p['system_optimization']['batch_group_chunks'],4)
        self.assertEqual(p['system_qualification']['candidate_groups'],[1,2,4])
        self.assertTrue(p['system_qualification']['cuda_metadata_operator_gate']['required'])
        self.assertEqual(p['system_qualification']['cuda_metadata_operator_gate']['ranks'],8)
        self.assertEqual(digest(p['baseline_core_policy']),p['core_policy_sha256'])
        self.assertFalse(p['restore_original_holder_after_exit'])
        self.assertEqual(p['supervisor_timeout_seconds'],7200)
        self.assertEqual(p['round_absolute_deadline_unix'],1794223026.9783676)
        self.assertEqual(p['timeout_origin'],'original v1 GPU release; fixed round deadline')
        self.assertTrue(p['system_optimization']['native_idle_steps_passthrough_without_batch_progress'])
        self.assertIsNone(p['quality']['max_tokens'])
        self.assertEqual(p['timing']['output_tokens'],32)
        for relative,expected in p['core_source_sha256'].items():
            self.assertEqual(hashlib.sha256((runner.STAGE/relative).read_bytes()).hexdigest(),expected)

    def test_world_reader_rejects_self_consistent_real_decode_before_prefill(self):
        from kv_cache_engram.s1_system_batch import batch_schedule
        schedule=batch_schedule([dict(start=0,end=16384,key='a'*64)],16384,
            request_id='request',phase='qualification',phase_generation=1,
            input_ids_sha256='b'*64,group_chunks=4)
        world=world_traces(schedule);samples=world[0][0]['scheduler_observations']
        samples[0].update(is_empty=False,sequences=[dict(request_id='request',is_prompt=False,
            seqlen_q=1,seqlen_k=16385)])
        # Recompute the hash and all eight reports: rejection must come from
        # actual scheduling semantics, not merely a mismatched digest.
        for rank,trace in enumerate(world):
            trace[0].update(world_phase='decode',rank0_request_id='request',
                scheduler_observations_sha256=digest(samples),local_is_empty=rank>0)
        reports=[dict(world_step_trace=trace) for trace in world]
        with self.assertRaisesRegex(ValueError,'real decode before final batch flush'):
            reader.audit_world_scheduler(reports,schedule)

    def test_nine_fresh_timings_per_case_rotate_all_three_arms(self):
        for index in protocol.EVALUATION_INDICES:
            schedule=runner.timing_schedule(index)
            self.assertEqual(len(schedule),9)
            for repeat in range(3):
                self.assertEqual({a for r,a in schedule if r==repeat},set(protocol.ARMS))
            if index<20:self.assertNotEqual(schedule[0][1],runner.timing_schedule(index+2)[0][1])
        for index in (0,1,3,21,22,True):
            with self.assertRaises(ValueError):runner.timing_schedule(index)

    def test_full_manifest_validated_before_unmodified_even_selection(self):
        rows=[]
        for index in range(2,22):
            band='128k' if index%2==0 else '64k';n=131072 if band=='128k' else 65536
            docs=[dict(input_start=n*i//10,input_end=n*(i+1)//10) for i in range(10)]
            rows.append(dict(index=index,dataset='loong',length_band=band,length_band_ordinal=(index-2)//2+1,
                target_input_tokens=n,input_tokens=n,source_document_count=5,original_document_count=5,
                raw_answer_sha256='a'*64,documents=docs))
        before=copy.deepcopy(rows)
        selected=runner.selected_rows(dict(cohort=rows))
        self.assertEqual(selected,rows[::2]);self.assertEqual(rows,before)
        self.assertTrue(all(a is b for a,b in zip(selected,rows[::2])))
        with self.assertRaises(ValueError):runner.selected_rows(dict(cohort=rows[::2]))
        bad=copy.deepcopy(rows);bad[1]['dataset']='foreign'
        with self.assertRaises(ValueError):runner.selected_rows(dict(cohort=bad))
        bad=copy.deepcopy(rows);bad[0]['documents']=[dict(input_start=0,input_end=131072)]
        with self.assertRaisesRegex(ValueError,'completely empty'):runner.selected_rows(dict(cohort=bad))

    def test_group2_empty_probe_is_exact_prefix_not_new_quality_input(self):
        source=dict(index=2,dataset='loong',input_ids=list(range(40000)),input_ids_sha256=digest(list(range(40000))))
        with patch.object(runner,'selected_rows',return_value=[source]):
            probe=runner.empty_group_qualification_row({})
        self.assertEqual(probe['input_ids'],source['input_ids'][:32769])
        self.assertEqual(probe['source_full_input_ids_sha256'],source['input_ids_sha256'])
        spans=reader.documents(probe)
        groups=reader.group_oracle(reader.chunk_oracle(spans,32769),2)
        self.assertEqual([len(g['selected_positions']) for g in groups],[0,127,1])
        self.assertEqual(probe['references'],['__diagnostic_not_rag_quality__'])
        with self.assertRaises(ValueError):reader.documents(dict(probe,input_tokens=32768))

    def test_formal_system_cannot_fallback_to_group1_or_group2(self):
        runner.verify_phase(phase('s1_system'),index=2,arm='s1_system',timing=True,quality=False)
        for group in (1,2):
            value=phase('s1_system',quality=True,tokens=200);value.update(batch_group_chunks=group);value=seal(value)
            with self.assertRaisesRegex(ValueError,'fallback'):runner.verify_phase(value,index=2,arm='s1_system',timing=False,quality=True)
        with self.assertRaises(ValueError):protocol.phase_execution('s1',batch_group_chunks=4)
        with self.assertRaises(ValueError):protocol.phase_execution('s1_system',batch_group_chunks=True)

    def test_independent_three_arm_timing_summary_and_system_speedup(self):
        phases=[];samples={a:[] for a in protocol.ARMS}
        for repeat,arm in runner.timing_schedule(2):
            value=phase(arm,repeat);sample={'recomputed':9.,'s1':6.,'s1_system':3.}[arm]+repeat*.1
            value['timing_valid']=True
            value['record'].update(online_ttft_s=sample,online_elapsed_s=sample+1,online_prepare_to_submit_s=.1)
            value['online_global_window_s']=sample+2
            phases.append(seal(value));samples[arm].append(sample)
        self.assertEqual(runner.summarize_timing(phases,2),reader.expected_timing_summary(samples))
        self.assertAlmostEqual(runner.summarize_timing(phases,2)['comparisons']['s1_system']['ttft_speedup_vs_s1'],6.1/3.1)
        with self.assertRaises(ValueError):runner.summarize_timing(phases[:-1],2)
        with self.assertRaises(ValueError):reader.expected_timing_summary({k:v for k,v in samples.items() if k!='s1_system'})

    def test_complete_output_tokens_required_not_first_token_or_equal_scores(self):
        phases={a:phase(a,quality=True,tokens=200) for a in protocol.ARMS}
        good=runner.system_quality_parity(phases,2)
        self.assertTrue(good['passed']);self.assertTrue(reader.audit_quality_parity(good,phases,2))
        # Scores and first token remain equal; a later actual generated token differs.
        changed=copy.deepcopy(phases);changed['s1_system']['record']['output_ids'][100]+=1
        changed['s1_system']['record']['output_ids_sha256']=digest(changed['s1_system']['record']['output_ids'])
        changed['s1_system']=seal(changed['s1_system'])
        failed=runner.system_quality_parity(changed,2)
        self.assertFalse(failed['passed']);self.assertFalse(reader.audit_quality_parity(failed,changed,2))
        with self.assertRaises(ValueError):reader.audit_quality_parity(seal(dict(failed,passed=True)),changed,2)
        with self.assertRaises(ValueError):runner.system_quality_parity({k:v for k,v in phases.items() if k!='recomputed'},2)
        changed['s1_system']['arm']='s1';changed['s1_system']=seal(changed['s1_system'])
        with self.assertRaises(ValueError):runner.system_quality_parity(changed,2)

    def test_partial_completion_is_declared_even_prefix_and_never_all_ten(self):
        rows=[dict(index=i,shared_bank_identity=dict(bank_generation=j+1)) for j,i in enumerate((2,4,6))]
        value=reader.completion_state(rows,[],complete=False,allow_partial=True,partial_reason='bounded_window')
        self.assertEqual(value['completed_case_indices'],[2,4,6]);self.assertEqual(value['planned_cases'],10)
        self.assertEqual(value['missing_case_indices'],[8,10,12,14,16,18,20])
        self.assertFalse(value['overall_run_complete'])
        with self.assertRaises(ValueError):reader.completion_state(rows[1:],[],complete=False,allow_partial=True,partial_reason='bounded_window')

    def test_cli_help_bootstraps_absolute_path_from_isolated_cwd(self):
        with tempfile.TemporaryDirectory() as cwd:
            result=subprocess.run([sys.executable,'-I','-B',str(Path(runner.__file__).resolve()),
                '--s1-protocol','unused.json','--help'],cwd=cwd,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--s1-protocol',result.stdout)


if __name__=='__main__':unittest.main()
