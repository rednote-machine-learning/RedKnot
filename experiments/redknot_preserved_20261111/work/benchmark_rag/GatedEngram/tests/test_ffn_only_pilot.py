"""CPU contract tests, not GPU correctness or accuracy measurements."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from kv_cache_engram.core import ValidationError,digest
from gpu_pilot.ffn_only_protocol import ffn_only_protocol,ARMS,FFN_LAYERS,CAPTURE_LAYERS,timing_schedule
from gpu_pilot import run_ffn_only_pilot as runner
from gpu_pilot import read_ffn_only_results as reader
from gpu_pilot import launch_ffn_only_pilot as launcher


def fixture():
    n=512; spans=[(2,258,"a"*64),(258,510,"b"*64)]
    scores=[float(i) for i in range(n)]
    ids,fixed,optional=reader.mask_reference(scores,0,n,spans,.15)
    row=dict(index=2,input_tokens=n,input_ids_sha256="c"*64,dataset="hotpotqa")
    phase=dict(arm="ffn15",mode="sensitive",name="phase",record=dict(request_id="request"))
    selection=dict(start=0,end=n,full_rows=n,selected_local_rows=ids,selected_rows=len(ids),
        selected_rows_sha256=digest(ids),mandatory_rows=fixed,optional_rows=optional,
        mean_adjacent_relative_l2=scores,plan_sha256="d"*64)
    selection["adjacent_transition_relative_l2"]=[list(scores) for _ in range(3)]
    events=[]; layouts=[]; swa=[]
    for layer in FFN_LAYERS:
        positions=list(range(n)) if layer<20 else [10,440,450,470,511]
        selected=[i for i,p in enumerate(positions) if p in ids]
        layout=dict(layer=layer,chunk_start=0,chunk_end=n,current_positions=positions,
            selected_local_rows=selected,full_rows=len(positions),selected_rows=len(selected),plan_sha256="e"*64)
        layouts.append(layout)
        events.append(dict(layer=layer,chunk_start=0,chunk_end=n,request_id="request",phase="phase",
            full_rows=len(positions),selected_rows=len(selected),plan_sha256="e"*64,current_layout="native_current_rows",
            inactive_ffn_output="zero",all_selected_native=len(positions)==len(selected),
            empty_native_collective=not selected and bool(positions)))
        online=[i for i,p in enumerate(positions) if not any(lo<=p<hi for lo,hi,_ in spans)
                or any(lo<=p<lo+127 for lo,hi,_ in spans)]
        swa.append(dict(layer=layer,chunk_start=0,chunk_end=n,current_positions=positions,
            online_local_rows=online,online_rows=len(online),reused_rows=len(positions)-len(online),
            full_rows=len(positions),compact=layer>=20))
    full=sum(v["full_rows"] for v in layouts); selected=sum(v["selected_rows"] for v in layouts)
    control=dict(schema="adjacent-swa-ffn-only-controller-v1",mode="sensitive",phase="phase",
        input_ids_sha256="c"*64,full_input_tokens=n,first_six_full=True,actual_full_row_native_mhc=True,
        all_query_attention_native=True,yoco_layout_preserved=True,formal_admission_unchanged=True,
        not_model_flops_fraction=True,selection=[selection],swa_events=swa,ffn_layouts=layouts,
        adapter=dict(schema="kv-cache-engram-ffn-only-adapter-v1",events=events,
        decoder_forward_unchanged=True,attention_forward_unchanged=True,native_mhc_unchanged=True,
        native_yoco_layout_unchanged=True,all_selected_delegates_native=True,
        empty_selection_skips_ep_collective=False,cached_hidden_or_ffn_output=False,
        accuracy_guarantee=False,speedup_claim=False,
        counts=dict(full_rows=full,selected_rows=selected,inactive_ffn_zero_rows=full-selected,
            empty_native_mlp_calls=sum(v["selected_rows"]==0 and v["full_rows"]>0 for v in layouts))))
    return control,row,phase,spans


class ProtocolTests(unittest.TestCase):
    def test_protocol_sealed_and_exact_user_choices(self):
        p=ffn_only_protocol()
        self.assertEqual(p["sha256"],digest({k:v for k,v in p.items() if k!="sha256"}))
        self.assertEqual(p["score_layers_zero_based"],[2,3,4,5])
        self.assertEqual(p["optional_fraction"],.15)
        self.assertTrue(p["all_query_attention_native"])
        self.assertFalse(p["selected_tokens_forced_through_all_layers"])
        self.assertFalse(p["doc_tail_or_native_chunk_boundary_guard"])
        self.assertTrue(p["offline_full_layout_for_kv_capture"])
        self.assertTrue(p["restore_model_yoco_boundary_before_online"])
        self.assertEqual(p["online_yoco_native_compact_keep_tokens"],140)
        self.assertEqual(CAPTURE_LAYERS,tuple(range(40)))
        self.assertEqual(FFN_LAYERS,tuple(range(6,40)))

    def test_only_two_arms_and_three_paired_rounds(self):
        self.assertEqual(ARMS,("recomputed","ffn15"))
        for case in (2,3,4,5):
            self.assertEqual(runner.timing_schedule(case),timing_schedule(case))
            self.assertEqual(len(timing_schedule(case)),6)
            for arm in ARMS:
                self.assertEqual(sum(a==arm for _,a in timing_schedule(case)),3)

    def test_source_identity_no_joint_attention_dependency(self):
        from gpu_pilot.ffn_only_protocol import DEPENDENCIES
        self.assertEqual(DEPENDENCIES,{})
        subprocess.run([sys.executable,"-B","-c",
            "import sys; from gpu_pilot import run_ffn_only_pilot, read_ffn_only_results, launch_ffn_only_pilot; assert 'torch' not in sys.modules"],check=True)

    def test_launcher_is_new_stage_old_user_stop(self):
        self.assertEqual(launcher.NAME,"kce-ffn-only-20261104-v3")
        self.assertEqual(launcher.PREDECESSOR,"kce-early-mhc-20261102-v2")
        self.assertEqual(launcher.FAILED_PREDECESSOR,"kce-ffn-only-20261104-v2")
        self.assertIn("gated_engram_ffn_only_20261104_v3",str(launcher.REMOTE_STAGE))


class FailedPredecessorTests(unittest.TestCase):
    """Pure CPU lifecycle contracts: no SSH, GPU query, or process signal."""

    def fixture(self):
        root=Path("/isolated-ffn-root")
        results=root/"results"
        name=launcher.FAILED_PREDECESSOR
        state=dict(status="failed",holder_restored=True,worker_exit=1,finished_unix=1,
            supervisor_sha256=launcher.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launcher.lifecycle.HELPER_SHA,
            child_environment_sha256=launcher.lifecycle.ENV_SHA,
            cleanup_warnings=["preserved original unauthenticated-member warning"])
        holder_path=results/(name+".terminal-holder-audit-20261104-1208.json")
        outcome_path=results/(name+".launch-outcome.json")
        raw_path=results/name/"benchmark/failure.rank0.json"
        holder=dict(sha256=launcher.FAILED_HOLDER_SHA,
            metadata=dict(worker_pids=list(range(100,108)),gpu_ids=list(range(8))),
            gpus=[dict(index=i,uuid="GPU-test-"+str(i)) for i in range(8)],
            contexts=[dict(uuid="GPU-test-"+str(i),pid=100+i,memory_mib=1000) for i in range(8)])
        outcome=dict(sha256=launcher.FAILED_OUTCOME_SHA,successful=False,automatic_retry=False,
            supervisor_started=True,supervisor_exit=1,holder_physically_verified=True)
        raw=dict(sha256=launcher.FAILED_RAW_SHA,status="failed",completed_indices=[],
            error=launcher.FAILED_ERROR)
        audit=dict(sha256=launcher.FAILED_AUDIT_SHA,schema="kce-ffn-only-terminal-smoke-failure-audit-v1",
            name=name,status="failed_during_partial_smoke",quality_completed_indices=[],
            partial_8x32k_smoke_passed=False,cuda_mask_gpu_qualification_passed=True,
            strict_16k_all_selected_gpu_qualification_passed=True,stage_sources_sha256=launcher.FAILED_STAGE_SHA,
            error=launcher.FAILED_ERROR,holder_physically_verified=True,automatic_retry=False,
            supervisor_file_sha256="s"*64,supervisor_cleanup_warnings=copy.deepcopy(state["cleanup_warnings"]),
            holder_audit_sha256=holder["sha256"],
            source_seals={outcome_path.name:outcome["sha256"],raw_path.name:raw["sha256"],holder_path.name:holder["sha256"]},
            all_original_process_ids_absent=list(launcher.FAILED_PROCESS_IDS))
        sealed={results/(name+".terminal-audit-20261104-1210.json"):audit,
            outcome_path:outcome,raw_path:raw,holder_path:holder}
        return root,sealed,state,audit,outcome,raw,holder

    def check(self,change=None,live=None):
        values=self.fixture()
        root,sealed,state,*_=values
        if change: change(values)
        def proc(pid):
            if live is not None and pid in launcher.FAILED_KNOWN_START_TICKS:
                return dict(start_ticks=live)
            raise FileNotFoundError(pid)
        with patch.object(launcher.lifecycle,"read_sealed",side_effect=lambda path:sealed[Path(path)]),\
             patch.object(launcher.lifecycle,"read_json",return_value=state),\
             patch.object(launcher.lifecycle,"file_digest",return_value="s"*64),\
             patch.object(launcher.lifecycle,"proc_identity",side_effect=proc):
            launcher.verify_failed_predecessor(root)

    def test_exact_failed_handoff_and_absent_processes_pass(self):
        self.check()

    def test_reused_pid_identity_is_not_mistaken_for_old_process(self):
        self.check(live=999999)

    def test_old_launcher_identity_still_live_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"still alive"):
            self.check(live=2883926072)

    def test_missing_or_altered_terminal_audit_is_rejected(self):
        for field,value in (("sha256","changed"),("name","another"),("automatic_retry",True),
                            ("quality_completed_indices",[2]),("status","running"),
                            ("stage_sources_sha256","changed"),("partial_8x32k_smoke_passed",True),
                            ("holder_physically_verified",False),("cuda_mask_gpu_qualification_passed",False),
                            ("strict_16k_all_selected_gpu_qualification_passed",False)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.check(lambda v:v[3].update({field:value}))

    def test_success_or_unrestored_outcome_cannot_authorize_repair_launch(self):
        for field,value in (("sha256","changed"),("successful",True),("automatic_retry",True),
                            ("holder_physically_verified",False),("supervisor_exit",0)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.check(lambda v:v[4].update({field:value}))

    def test_wrong_failure_or_raw_binding_is_rejected(self):
        for field,value in (("sha256","changed"),("error","other error"),("completed_indices",[2])):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.check(lambda v:v[5].update({field:value}))
        with self.assertRaises(ValueError):
            self.check(lambda v:v[3]["source_seals"].clear())

    def test_changed_supervisor_or_lifecycle_identity_is_rejected(self):
        for field,value in (("holder_restored",False),("status","running"),("worker_exit",0),
                            ("cleanup_error","unresolved"),("supervisor_sha256","changed")):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.check(lambda v:v[2].update({field:value}))

    def test_missing_exit_audit_and_unresolved_warning_are_rejected(self):
        for pid in launcher.FAILED_PROCESS_IDS:
            with self.subTest(pid=pid),self.assertRaises(ValueError):
                self.check(lambda v:v[3]["all_original_process_ids_absent"].remove(pid))

    def test_holder_evidence_requires_eight_unique_positive_contexts(self):
        with self.assertRaises(ValueError):
            self.check(lambda v:v[6]["contexts"].pop())
        with self.assertRaises(ValueError):
            self.check(lambda v:v[6]["contexts"][0].update(memory_mib=0))
        with self.assertRaises(ValueError):
            self.check(lambda v:v[6].update(sha256="changed"))

    def test_old_stop_and_oracle_pause_checks_remain_required(self):
        root=Path("/isolated-ffn-root")
        paused=dict(sha256="33e20e7b6b1ef85f3991c367200edeb25acbf4e306300118fc9cc924f856a54e",
            user_requested_stop=True,status="stopped_by_user",holder_physically_verified=True,automatic_retry=False,
            old_process_exit_checks={name:dict(exit_verified=True) for name in
                ("launcher_identity","supervisor_identity","child_identity")})
        state=dict(holder_restored=True,finished_unix=1,error="KeyboardInterrupt('received signal 15')",
            supervisor_sha256=launcher.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launcher.lifecycle.HELPER_SHA,
            child_environment_sha256=launcher.lifecycle.ENV_SHA)
        oracle=dict(exit_verified=True,no_gpu_supervisor_started=True)
        for oracle_ok,started in ((True,False),(False,False),(True,True)):
            oracle["exit_verified"]=oracle_ok
            with patch.object(launcher,"verify_failed_predecessor") as failed,\
                 patch.object(launcher.lifecycle,"read_sealed",side_effect=[paused,oracle]),\
                 patch.object(launcher.lifecycle,"read_json",return_value=state),\
                 patch.object(launcher.lifecycle,"proc_identity",side_effect=FileNotFoundError),\
                 patch.object(Path,"exists",return_value=started):
                if oracle_ok and not started:
                    launcher.verify_predecessor(root)
                    failed.assert_called_once_with(root)
                else:
                    with self.assertRaises(ValueError):launcher.verify_predecessor(root)


class MaskTests(unittest.TestCase):
    def test_per_token_rank_not_aggregate_and_no_tail_guard(self):
        scores=[0.]*300
        scores[200]=100
        ids,fixed,optional=reader.mask_reference(scores,0,300,[(0,300,"x")],.15)
        self.assertEqual(fixed,0)
        self.assertEqual(optional,300)
        self.assertEqual(len(ids),45)
        self.assertIn(200,ids)
        self.assertNotIn(299,ids)

    def test_tie_deterministic_and_prefix_counts_in_top15(self):
        ids,fixed,optional=reader.mask_reference([1.]*20,100,120,[(100,120,"x")],.15)
        self.assertEqual(ids,[0,1,2])
        self.assertEqual((fixed,optional),(0,20))

    def test_non_document_additive_only(self):
        ids,fixed,optional=reader.mask_reference([0.]*10,0,10,[(2,8,"x")],.15)
        self.assertEqual((fixed,optional),(4,6))
        self.assertEqual(ids,[0,1,2,8,9])

    def test_fraction_and_nonfinite_rejected(self):
        for scores,fraction in [([0.]*4,.2),([0.,math.nan,0.,0.],.15),
                                ([True,0.,0.,0.],.15),([0.,-1.,0.,0.],.15)]:
            with self.assertRaises(ValidationError):
                reader.mask_reference(scores,0,4,[(0,4,"x")],fraction)


class ControllerAuditTests(unittest.TestCase):
    def audit(self,change=None):
        values=fixture()
        if change: change(values[0])
        return reader.audit_controller(*values)

    def test_decoupled_full_query_partial_ffn_native_yoco(self):
        result=self.audit()
        self.assertGreater(result["ffn_bypassed_rows"],0)
        self.assertGreater(result["swa_reused_rows"],0)
        self.assertNotEqual(result["ffn_bypass_fraction"],result["swa_reuse_fraction"])

    def test_ffn_masks_cannot_be_used_for_swa_projection(self):
        def change(c):
            c["swa_events"][0]["online_local_rows"]=c["ffn_layouts"][0]["selected_local_rows"]
        with self.assertRaises(ValidationError): self.audit(change)

    def test_no_hidden_chunk_or_tail_guards(self):
        def change(c):
            c["swa_events"][0]["online_local_rows"].append(509)
        with self.assertRaises(ValidationError): self.audit(change)

    def test_yoco_selected_intersection_required(self):
        def change(c):
            c["ffn_layouts"][-1]["selected_local_rows"]=[0]
        with self.assertRaises(ValidationError): self.audit(change)

    def test_full_row_encoder_must_not_compact(self):
        def change(c):
            c["ffn_layouts"][0]["current_positions"]=[1,2]
        with self.assertRaises(ValidationError): self.audit(change)

    def test_missing_or_duplicate_event_is_invalid(self):
        for field in ("ffn_layouts","swa_events"):
            with self.subTest(field=field),self.assertRaises(ValidationError):
                self.audit(lambda c:c[field].pop())
            with self.subTest(field=field),self.assertRaises(ValidationError):
                self.audit(lambda c:c[field].append(c[field][0]))

    def test_changed_event_count_or_binding_rejected(self):
        for field,value in (("request_id","foreign"),("selected_rows",0),("plan_sha256","f"*64)):
            with self.subTest(field=field),self.assertRaises(ValidationError):
                self.audit(lambda c:c["adapter"]["events"][0].update({field:value}))

    def test_shared_swa_and_global_or_attention_sparsity_claim_rejected(self):
        for field in ("all_query_attention_native","yoco_layout_preserved","first_six_full"):
            with self.subTest(field=field),self.assertRaises(ValidationError):
                self.audit(lambda c:c.update({field:False}))

    def test_zero_native_upper_rows_supported(self):
        control,row,phase,spans=fixture()
        for collection in (control["swa_events"],control["ffn_layouts"],control["adapter"]["events"]):
            for event in collection:
                if event["layer"]<20: continue
                for field in ("current_positions","online_local_rows","selected_local_rows"):
                    if field in event:event[field]=[]
                for field in ("online_rows","reused_rows","selected_rows","full_rows"):
                    if field in event:event[field]=0
                if "all_selected_native" in event:event["all_selected_native"]=True
                if "empty_native_collective" in event:event["empty_native_collective"]=False
        full=sum(v["full_rows"] for v in control["ffn_layouts"])
        selected=sum(v["selected_rows"] for v in control["ffn_layouts"])
        control["adapter"]["counts"]=dict(full_rows=full,selected_rows=selected,inactive_ffn_zero_rows=full-selected,
            empty_native_mlp_calls=0)
        self.assertGreater(reader.audit_controller(control,row,phase,spans)["ffn_bypassed_rows"],0)

    def test_partial_smoke_producer_checks_same_contract(self):
        control,row,phase,spans=fixture()
        row["documents"]=[{}]*8
        phase.update(index=2,schema=runner.PHASE_SCHEMA,input_ids_sha256="c"*64,partial_smoke=True,
            fraction=.15,timing_valid=False,quality_eligible=False,controller=control,
            empty_ep_qualification_sha256="a"*64)
        phase["record"].update(success=True,rank=0,input_ids_sha256="c"*64,
            output_ids=[7],output_ids_sha256=digest([7]),natural_EOS=False)
        phase["sha256"]=digest(phase)
        with patch.object(reader,"documents",return_value=spans):
            result=runner.verify_partial_smoke([phase],row,dict(passed=True,identity_check_count=80,sha256="f"*64,
                empty_ep_qualification_sha256="a"*64))
        self.assertTrue(result["passed"])


def outcome(em0,em1,f10,f11,t0,t1):
    return dict(dataset="hotpotqa",quality={
        "recomputed":dict(answer_em=em0,answer_token_f1=f10,truncated=False),
        "ffn15":dict(answer_em=em1,answer_token_f1=f11,truncated=False)},
        online_ttft_samples_s={"recomputed":[t0]*3,"ffn15":[t1]*3})


class SummaryTests(unittest.TestCase):
    def test_tradeoff_keeps_both_points(self):
        result=reader.summarize_cases([outcome(1,0,1,0,80,50)])
        self.assertEqual(result["empirical_frontiers"]["em_percent_vs_ttft"],list(ARMS))
        self.assertEqual(result["comparison"]["correct_to_wrong"],1)
        self.assertEqual(result["comparison"]["ttft_speedup"],1.6)

    def test_equal_quality_faster_dominates(self):
        result=reader.summarize_cases([outcome(1,1,1,1,80,50)])
        self.assertEqual(result["empirical_frontiers"]["em_percent_vs_ttft"],["ffn15"])

    def test_equal_points_both_retained(self):
        result=reader.summarize_cases([outcome(1,1,1,1,80,80)])
        self.assertEqual(result["empirical_frontiers"]["em_percent_vs_ttft"],list(ARMS))

    def test_worse_and_slower_candidate_dominated(self):
        result=reader.summarize_cases([outcome(1,0,1,0,80,90)])
        self.assertEqual(result["empirical_frontiers"]["f1_percent_vs_ttft"],["recomputed"])
        self.assertLess(result["comparison"]["ttft_reduction_s"],0)

    def test_loss_negative_if_improves_no_clip(self):
        result=reader.summarize_cases([outcome(0,1,0,1,80,70)])
        self.assertEqual(result["comparison"]["em_loss_pp"],-100)
        self.assertEqual(result["comparison"]["wrong_to_correct"],1)
        self.assertFalse(result["global_frontier_claim"])


def split_fixture():
    swa=[]; events=[]
    for start in (0,8192):
        end=start+8192
        for layer in FFN_LAYERS:
            positions=list(range(start,end)) if layer<20 else ([] if start==0 else list(range(end-140,end)))
            swa.append(dict(chunk_start=start,chunk_end=end,layer=layer,current_positions=positions,
                full_rows=len(positions),online_rows=len(positions),online_local_rows=list(range(len(positions))),reused_rows=0))
            if positions:
                events.append(dict(chunk_start=start,chunk_end=end,layer=layer,full_rows=len(positions),checks=3,
                    max_absolute_error=0.,checks_named=["split_q","split_swa","packed_gather_swa127"]))
    return dict(swa_events=swa,projection_identity_events=events,projection_identity_checks=len(events)*3)


class ProjectionQualificationTests(unittest.TestCase):
    def test_every_nonempty_native_layout_checks_q_kv_and_gather(self):
        self.assertEqual(reader.projection_identity(split_fixture()),144)

    def test_nonzero_error_rejected(self):
        value=split_fixture(); value["projection_identity_events"][0]["max_absolute_error"]=1e-12
        with self.assertRaises(ValidationError):reader.projection_identity(value)

    def test_missing_and_duplicate_checks_rejected(self):
        for mutate in (lambda v:v["projection_identity_events"].pop(),
                       lambda v:v["projection_identity_events"].append(v["projection_identity_events"][0])):
            value=split_fixture();mutate(value)
            with self.assertRaises(ValidationError):reader.projection_identity(value)

    def test_count_does_not_replace_raw_evidence(self):
        value=split_fixture();value["projection_identity_events"]=[]
        with self.assertRaises(ValidationError):reader.projection_identity(value)

    def test_offline_reuse_forbidden_in_identity_qualification(self):
        value=split_fixture();value["swa_events"][0]["reused_rows"]=1
        with self.assertRaises(ValidationError):reader.projection_identity(value)

    def test_all_layers_all_chunks_required_even_if_zero_query(self):
        value=split_fixture();value["swa_events"].pop(15)
        with self.assertRaises(ValidationError):reader.projection_identity(value)

    def test_named_checks_and_schema_are_exact(self):
        value=split_fixture();value["projection_identity_events"][0]["checks_named"]=["allclose"]
        with self.assertRaises(ValidationError):reader.projection_identity(value)

    def test_adjacent_transition_mutation_rejected(self):
        value,row,phase,spans=fixture()
        value["selection"][0]["adjacent_transition_relative_l2"][1][300]+=1
        with self.assertRaises(ValidationError):reader.audit_controller(value,row,phase,spans)

