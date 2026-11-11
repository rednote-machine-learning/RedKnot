"""CPU-only regression tests for the independent tail128 pilot and reader."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from gpu_pilot import tail_link_protocol as protocol
from gpu_pilot import read_tail_link_results as reader
from gpu_pilot import run_tail_link_pilot as runner
from kv_cache_engram.core import digest


def spans():
    return [(5+i*32768, 5+(i+1)*32768, str(i)*64) for i in range(8)]


def staging():
    return dict(schema="tail-link-h2d-staging-v1", phase_generation=3, double_buffers=2,
                cpu_pinned=True, cross_phase_gpu_reuse=False, copies_after_reset=True,
                same_phase_restored_reuse=True, copy_ms_is_not_wall_or_ttft=True,
                document_keys=[k for _, _, k in spans()],
                copies=[dict(layer=l, slot=l%2, generation=3, bytes=1048576, cuda_copy_ms=.3)
                        for l in range(3, 40)], copy_bytes=38797312, copy_count=37)


class ProtocolTests(unittest.TestCase):
    def test_new_identity(self):
        p = protocol.tail_link_protocol()
        self.assertEqual(p["sha256"], digest({k:v for k,v in p.items() if k != "sha256"}))
        self.assertEqual(p["arms"], ["recomputed", "tail128_link", "tail128_dynamic"])
        self.assertIsNone(p["optional_fraction"])

    def test_no_fraction_or_index_selection(self):
        self.assertEqual(protocol.FRACTIONS, dict(recomputed=None, tail128_link=None, tail128_dynamic=None))
        self.assertFalse(protocol.tail_link_protocol()["selection_depends_on_index_hits"])

    def test_qualifications_are_fresh(self):
        p = protocol.tail_link_protocol()
        self.assertEqual(p["empty_ep_qualification"]["checks"], 37*4*8)
        self.assertEqual(p["sparse_parity"]["layer_chunk_checks"], 40*33)
        self.assertFalse(p["sparse_parity"]["equivalence_to_full_model_claim"])

    def test_tail_storage_40_mib(self):
        self.assertEqual(protocol.tail_link_protocol()["offline_payload_bytes_per_case"], 40*1024**2)

    def test_three_formal_pairs_and_rotation(self):
        for case in (2, 3, 4, 5):
            schedule = runner.timing_schedule(case)
            self.assertEqual(schedule, protocol.timing_schedule(case))
            self.assertEqual(len(schedule), 9)
            for rep in range(3):
                self.assertEqual(set(a for r,a in schedule if r == rep), set(protocol.ARMS))
        self.assertNotEqual(runner.timing_schedule(2), runner.timing_schedule(3))

    def test_unregistered_case_rejected(self):
        for case in (True, 0, 6, "2"):
            with self.assertRaises(ValueError):
                runner.timing_schedule(case)

    def test_source_pins_must_be_frozen(self):
        with mock.patch.object(protocol, "DEPENDENCIES", {}):
            with self.assertRaisesRegex(ValueError, "must be frozen"):
                runner.source_hashes(Path("/unused"))

    def test_clock_precedes_reset_and_no_raw_index_export(self):
        source = (STAGE/"gpu_pilot/run_tail_link_pilot.py").read_text()
        phase = source[source.index("        def phase("):source.index("        check_row =")]
        self.assertLess(phase.index("began = time.perf_counter()"), phase.index("controller.reset("))
        self.assertNotIn("set_index_output_dir", phase)
        self.assertIn("phase_local_gpu_staging=True", phase)

    def test_private_nested_live_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); live = root/"DeepInfer-PR148"
            stage = live/"experiments/tail/GatedEngram"; native = stage.parent/"NativeSnapshot"
            native.mkdir(parents=True); stage.mkdir()
            args = type("Args", (), dict(repo=native, experiment_root=root))()
            before = list(sys.path)
            try:
                sys.path[:] = [str(live), str(live/"examples"), str(stage), str(native)]
                with mock.patch.object(runner, "STAGE", stage), mock.patch.object(runner, "NATIVE_SNAPSHOT", native):
                    runner.bind_native_snapshot(args)
                self.assertIn(str(stage), sys.path)
                self.assertIn(str(native), sys.path)
                self.assertNotIn(str(live), sys.path)
                self.assertNotIn(str(live/"examples"), sys.path)
            finally:
                sys.path[:] = before


class ReaderTests(unittest.TestCase):
    def test_tail_selection_boundaries(self):
        docs = spans(); lo, hi, _ = docs[0]
        positions = [0, lo, hi-129, hi-128, hi-1, hi]
        selected, non_doc, tails = reader.mask_reference(positions, 0, hi+1, docs)
        self.assertEqual(selected, [0, 3, 4])
        self.assertEqual(non_doc, [0])
        self.assertEqual(tails, [3, 4])

    def test_query_template_selected(self):
        end = spans()[-1][1]
        self.assertEqual(reader.mask_reference([end, end+1], end, end+2, spans()), ([0,1],[0,1],[]))

    def test_empty_native_layout_not_forced(self):
        self.assertEqual(reader.mask_reference([], 8192, 16384, spans()), ([],[],[]))

    def test_positions_are_sorted_unique(self):
        for positions in ([2,1],[1,1],[True],[-1],[10]):
            with self.assertRaises(ValueError):
                reader.mask_reference(positions, 0, 10, spans())

    def test_staging_transfer_exact(self):
        self.assertEqual(reader.audit_staging(staging(), spans()), 37*1048576)

    def test_staging_cross_phase_rejected(self):
        p = staging(); p["cross_phase_gpu_reuse"] = True
        with self.assertRaises(ValueError): reader.audit_staging(p, spans())

    def test_staging_wrong_generation_rejected(self):
        p = staging(); p["copies"][0]["generation"] = 2
        with self.assertRaises(ValueError): reader.audit_staging(p, spans())

    def test_staging_missing_copy_rejected(self):
        p = staging(); p["copies"].pop()
        with self.assertRaises(ValueError): reader.audit_staging(p, spans())

    def test_staging_duplicate_layer_rejected(self):
        p = staging(); p["copies"][-1]["layer"] = 3
        with self.assertRaises(ValueError): reader.audit_staging(p, spans())

    def test_staging_nonfinite_timing_rejected(self):
        p = staging(); p["copies"][0]["cuda_copy_ms"] = float("nan")
        with self.assertRaises(ValueError): reader.audit_staging(p, spans())

    def test_staging_wrong_bytes_rejected(self):
        p = staging(); p["copies"][0]["bytes"] -= 1
        with self.assertRaises(ValueError): reader.audit_staging(p, spans())

    def test_raw_io_seal_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"result.json"; value = dict(n=1); value["sha256"] = digest(value)
            runner.write_json(path, value)
            evidence = {}; self.assertEqual(reader.read(path, evidence), value)
            self.assertEqual(evidence[path.name]["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError): runner.write_json(path, value)

    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError): reader._unique([("a",1),("a",2)])

    def test_quality_independent_gold(self):
        self.assertEqual(reader.score("The Blue Whale", ["blue whale"]), dict(answer_em=1., answer_token_f1=1.))
        self.assertEqual(reader.score("red whale", ["blue whale"]), dict(answer_em=0., answer_token_f1=.5))

    def test_event_coverage_not_counts_only(self):
        expected = {(0, 8192, 3), (0, 8192, 4)}
        events = [dict(start=0,end=8192,layer=3),dict(start=0,end=8192,layer=3)]
        with self.assertRaises(ValueError): reader._event_map(events, expected, "test")

    def memory_event(self, current, available, dense=False):
        from kv_cache_engram.tail_link_plan import plan
        docs = [dict(start=lo,end=hi,key=key) for lo,hi,key in spans()]
        end = spans()[-1][1]+2
        event = plan(docs,current,0,end,end,available_nondoc_positions=available)
        event.update(layer=3, query_positions=current if dense else event["selected_positions"],
                     query_memory_lengths=event["reference_memory_lengths"] if dense else event["selected_memory_lengths"],
                     actual_joint_query_rows=len(current) if dense else event["selected_rows"],
                     gpu_causal_oracle_checked=dense)
        return event

    def test_independent_memory_matches_planner(self):
        end = spans()[-1][1]
        current = [0,1,32772, end-128,end-1,end,end+1]
        event = self.memory_event(current,[0,1,end,end+1])
        selected = reader.mask_reference(current,0,end+2,spans())[0]
        self.assertEqual(reader.audit_memory_event(event,current,selected,spans(),[0,1,end,end+1],False), (1024,4))

    def test_causal_memory_no_future_own_tail(self):
        current = [32700, 32701]
        event = self.memory_event(current,[])
        selected = reader.mask_reference(current,0,spans()[-1][1]+2,spans())[0]
        reader.audit_memory_event(event,current,selected,spans(),[],False)
        event["memory_positions"].append(32702)
        with self.assertRaises(ValueError): reader.audit_memory_event(event,current,selected,spans(),[],False)

    def test_memory_wrong_source_offset_rejected(self):
        current = [32771,32772]
        event = self.memory_event(current,[])
        event["memory_doc_token_offsets"][0] = 0
        with self.assertRaises(ValueError): reader.audit_memory_event(event,current,[0,1],spans(),[],False)

    def test_memory_empty_yoco_layout_has_no_memory(self):
        event = self.memory_event([], [0,1])
        self.assertEqual(reader.audit_memory_event(event,[],[],spans(),[0,1],False), (0,0))

    def test_memory_plan_hash_recomputed(self):
        event = self.memory_event([0,1], [0,1])
        event["plan_sha256"] = "a"*64
        with self.assertRaises(ValueError): reader.audit_memory_event(event,[0,1],[0,1],spans(),[0,1],False)

    def test_memory_dense_reference_queries_include_inactive(self):
        current = [0, 200, 32771, 32772]
        event = self.memory_event(current,[0],dense=True)
        selected = reader.mask_reference(current,0,spans()[-1][1]+2,spans())[0]
        reader.audit_memory_event(event,current,selected,spans(),[0],True)
        with self.assertRaises(ValueError): reader.audit_memory_event(event,current,selected,spans(),[0],False)

    def test_bank_bounded_finite_and_sha(self):
        raw = b"\x80\x3f"*(128*512)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"tail.bf16"
            with path.open("xb") as handle: handle.write(raw)
            sha = hashlib.sha256(raw).hexdigest()
            self.assertEqual(reader.audit_bank_payload(path,sha)["bytes"],len(raw))
            with self.assertRaises(ValueError): reader.audit_bank_payload(path,"0"*64)

    def test_bank_nonfinite_rejected_even_valid_sha(self):
        raw = b"\x80\x7f"+b"\x00\x00"*(128*512-1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"tail.bf16"
            with path.open("xb") as handle: handle.write(raw)
            with self.assertRaisesRegex(ValueError,"nonfinite"):
                reader.audit_bank_payload(path,hashlib.sha256(raw).hexdigest())

    def test_bank_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"tail.bf16"; link = Path(tmp)/"link.bf16"
            with path.open("xb") as handle: handle.write(b"\0"*(128*512*2))
            link.symlink_to(path)
            with self.assertRaises(ValueError): reader.audit_bank_payload(link,"0"*64)

    def controller_fixture(self):
        """One layer with all33 real-sized chunks, exercising compact YOCO rows.

        The production reader still requires37 layers; only this fixture's
        explicit test patch reduces layer count. Transfer helper tested above.
        """
        from kv_cache_engram.tail_link_plan import plan
        total = spans()[-1][1]+50
        row = dict(input_tokens=total,input_ids_sha256="a"*64)
        phase = dict(mode="sensitive",name="test",arm="tail128_link",fraction=None, dynamic_document_swa=False,
                     native_yoco_swa_share_hidden=False,
                     verify_sparse=False,record=dict(request_id="r"))
        adapter = dict(schema="kv-cache-engram-index-hit20-adapter-v1",events=[],counts={})
        for name in ("decoder_forward_unchanged","attention_forward_unchanged","native_mhc_unchanged",
                     "native_yoco_layout_unchanged","all_selected_delegates_native"):
            adapter[name] = True
        for name in ("empty_selection_skips_ep_collective","cached_hidden_or_ffn_output","accuracy_guarantee","speedup_claim"):
            adapter[name] = False
        c = dict(schema="tail-link-controller-v1",mode="sensitive",phase="test",input_ids_sha256="a"*64,
                 dynamic_document_swa=False,dynamic_refresh_start_layer=None,dynamic_swa_events=[],
                 native_yoco_swa_share_hidden=False,
                 full_input_tokens=total,ffn_layers=[20],selected_tokens_forced_through_yoco=False,
                 adapter=adapter,selection=[],ffn_layouts=[],swa_events=[],attention_events=[],tail_memory_events=[],
                 staging=staging(),decode_ring_positions=[dict(layer=20,positions=list(range(total-128,total)),
                    physical_readback_exact=True,checks=128,physical_slots=list(range(128)))])
        for name in ("first_three_full","actual_full_row_native_mhc","yoco_layout_preserved","formal_admission_unchanged",
                     "decode_native","global_kv_native_online","not_model_flops_fraction","prefill_only_link"):
            c[name] = True
        available = []; full = selected = 0
        docs = [dict(start=lo,end=hi,key=key) for lo,hi,key in spans()]
        for start in range(0,total,8192):
            end = min(start+8192,total); positions = list(range(max(start,end-140),end))
            available += [p for p in positions if not any(lo<=p<hi for lo,hi,_ in spans())]
            event = plan(docs,positions,start,end,total,available_nondoc_positions=available)
            n,k = len(positions),event["selected_rows"]
            m = dict(start=start,end=end,layer=20,full_rows=n,selected_rows=k,current_positions=positions,
                     selected_local_rows=event["selected_local_rows"],selected_positions=event["selected_positions"],
                     plan_sha256=event["plan_sha256"])
            c["selection"].append(m)
            layout = {**m,"chunk_start":start,"chunk_end":end}
            c["ffn_layouts"].append(layout)
            adapter["events"].append(dict(chunk_start=start,chunk_end=end,layer=20,full_rows=n,selected_rows=k,
                actual_mlp_rows=k,reference_dense=False,plan_sha256=event["plan_sha256"],request_id="r",phase="test",
                inactive_ffn_output="zero",empty_native_collective=(k==0 and n>0)))
            non = event["mandatory_nondoc_local_rows"]; tails = event["document_tail_local_rows"]
            c["swa_events"].append(dict(chunk_start=start,chunk_end=end,layer=20,current_positions=positions,
                full_rows=n,document_online_rows=0,document_tail_rows=len(tails),non_document_online_rows=len(non),
                reused_rows=len(tails),online_local_rows=non,online_rows=len(non),compact=True))
            c["attention_events"].append(dict(chunk_start=start,chunk_end=end,layer=20,full_rows=n,selected_rows=k,
                main_q_a_rows=n,index_q_rows=n,main_q_b_rows=k,joint_attention_rows=k,output_projection_rows=k,
                global_compute_called=True,native_joint_attention=True,reference_dense=False,is_global_source=True,
                global_kv_rows=end,plan_sha256=event["plan_sha256"]))
            event.update(layer=20,query_positions=event["selected_positions"],
                         query_memory_lengths=event["selected_memory_lengths"],actual_joint_query_rows=k,
                         gpu_causal_oracle_checked=False)
            c["tail_memory_events"].append(event)
            full += n; selected += k
        adapter["counts"].update(full_rows=full,selected_rows=selected,actual_mlp_rows=selected,
                                  inactive_ffn_zero_rows=full-selected)
        return c,row,phase

    def check_fixture(self,c,row,phase):
        with mock.patch.object(reader,"FFN_LAYERS",(20,)),mock.patch.object(reader,"audit_staging",return_value=38797312):
            return reader.audit_controller(c,row,phase,spans())

    def test_complete_controller_contract(self):
        c,row,phase = self.controller_fixture()
        result = self.check_fixture(c,row,phase)
        self.assertGreater(result["ffn_bypassed_rows"],0)
        self.assertEqual(result["swa_online_rows"],50)
        self.assertGreater(result["consumed_memory_cached_tail_pairs"],result["materialized_memory_cached_tail_rows"])
        self.assertEqual(result["ffn_selected_rows"],result["swa_reused_rows"]+50)

    def test_controller_missing_layer_chunk_rejected(self):
        c,row,phase = self.controller_fixture(); c["attention_events"].pop()
        with self.assertRaises(ValueError): self.check_fixture(c,row,phase)

    def test_controller_global_offline_rejected(self):
        c,row,phase = self.controller_fixture(); c["attention_events"][0]["global_compute_called"] = False
        with self.assertRaises(ValueError): self.check_fixture(c,row,phase)

    def test_controller_doc_recompute_rejected(self):
        c,row,phase = self.controller_fixture(); c["swa_events"][0]["document_online_rows"] = 1
        with self.assertRaises(ValueError): self.check_fixture(c,row,phase)

    def test_controller_forged_decode_ring_rejected(self):
        c,row,phase = self.controller_fixture(); c["decode_ring_positions"][0]["physical_readback_exact"] = False
        with self.assertRaises(ValueError): self.check_fixture(c,row,phase)

    def test_controller_forged_selection_rejected(self):
        c,row,phase = self.controller_fixture(); c["selection"][0]["selected_local_rows"] = [0]
        with self.assertRaises(ValueError): self.check_fixture(c,row,phase)

    def run_closure_fixture(self):
        pre = dict(identity=dict(run="new"),manifest_sha256="a"*64)
        qual,smoke = dict(sha256="b"*64),dict(sha256="c"*64)
        dynamic_smoke = dict(sha256="f"*64)
        closures = [dict(index=i,sha256=str(i)*64) for i in (2,3,4,5)]
        evidence = {f"case{i}.paired.json":dict(file_sha256="d"*64,seal_sha256=str(i)*64) for i in (2,3,4,5)}
        refs = [dict(index=i,file=f"case{i}.paired.json",file_sha256="d"*64,seal_sha256=str(i)*64) for i in (2,3,4,5)]
        run = dict(schema=reader.RUN_SCHEMA,status="complete",completed_case_indices=[2,3,4,5],
                   identity=pre["identity"],manifest_sha256=pre["manifest_sha256"],qualification=qual,
                   partial_smoke_sha256=smoke["sha256"],dynamic_partial_smoke_sha256=dynamic_smoke["sha256"],cases=refs)
        run["sha256"] = digest(run)
        return run,pre,qual,smoke,closures,[],Path("/audit"),evidence,dynamic_smoke

    def test_final_run_complete_requires_all_audited_cases(self):
        args = self.run_closure_fixture()
        self.assertTrue(reader.validate_closed_run(*args))
        args[4].pop()
        with self.assertRaises(ValueError): reader.validate_closed_run(*args)

    def test_final_run_claim_does_not_override_errors(self):
        args = self.run_closure_fixture(); args[5].append(dict(index=2,error="bad raw"))
        with self.assertRaises(ValueError): reader.validate_closed_run(*args)

    def test_final_run_claim_does_not_override_raw_sha(self):
        args = self.run_closure_fixture(); args[7]["case2.paired.json"]["file_sha256"] = "e"*64
        with self.assertRaises(ValueError): reader.validate_closed_run(*args)

    def test_final_run_wrong_order_rejected(self):
        args = self.run_closure_fixture(); args[4].reverse()
        with self.assertRaises(ValueError): reader.validate_closed_run(*args)

    def test_final_run_foreign_identity_rejected(self):
        args = self.run_closure_fixture(); args[0]["identity"] = dict(run="old")
        args[0]["sha256"] = digest({k:v for k,v in args[0].items() if k != "sha256"})
        with self.assertRaises(ValueError): reader.validate_closed_run(*args)

    def test_final_run_corrupt_seal_rejected(self):
        args = self.run_closure_fixture(); args[0]["sha256"] = "0"*64
        with self.assertRaises(ValueError): reader.validate_closed_run(*args)

    def test_summary_uses_all_formal_samples_not_median_of_medians(self):
        cases = []
        for full,candidate in (([1.,2.,100.],[1.,1.,1.]),([4.,5.,6.],[2.,2.,100.])):
            cases.append(dict(quality={a:dict(answer_em=0.,answer_token_f1=0.,truncated=False) for a in protocol.ARMS},
                online_ttft_samples_s=dict(recomputed=full,tail128_link=candidate,tail128_dynamic=candidate),
                actual_rows_and_swa_reuse=dict(tail128_link={},tail128_dynamic={})))
        result = reader.summarize_cases(cases)
        self.assertEqual(result["arms"]["recomputed"]["online_ttft_p50_s"],4.5)
        self.assertEqual(result["arms"]["tail128_link"]["online_ttft_p50_s"],1.5)
        self.assertEqual(result["comparison"]["ttft_speedup"],3.)
        self.assertEqual(result["comparison"]["em_loss_pp"],0.)
        self.assertFalse(result["statistical_significance_claim"])

    def test_reader_no_model_imports(self):
        tree = ast.parse((STAGE/"gpu_pilot/read_tail_link_results.py").read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom): names.append(node.module or "")
        self.assertFalse(any(n.startswith(("torch", "deepinfer", "kv_cache_engram.tail_link")) for n in names))


if __name__ == "__main__":
    unittest.main()
