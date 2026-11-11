"""CPU-only contracts for literal SWA skip, sealed references and publication."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.core import ValidationError, digest
from gpu_pilot import run_swa_skip_pilot as runner
from gpu_pilot import read_swa_skip_results as reader
from gpu_pilot.swa_skip_protocol import swa_skip_protocol, timing_schedule, FFN_LAYERS


def seal(value):
    return dict(value, sha256=digest(value))


def global_events(n, *, reference=False, outputs=1, request_id="r", phase="p", placeholder_count=1):
    events = []
    for start in range(0, n, 8192):
        end = min(start+8192, n)
        for layer in FFN_LAYERS:
            positions = list(range(start,end)) if layer < 20 else [end-1]
            events.append(dict(layer=layer, chunk_start=start, chunk_end=end,
                current_positions=positions, full_rows=len(positions), global_query_rows=len(positions),
                swa_projection_rows=len(positions) if reference else 0,
                swa_attention_query_rows=0, swa_attention_kv_reads=0, offline_kv_attention_reads=0))
    decode = [dict(layer=layer, decode_step=step, current_positions=[n+step], full_rows=1,
        global_query_rows=1, swa_projection_rows=1 if reference else 0,
        swa_attention_query_rows=0, swa_attention_kv_reads=0, offline_kv_attention_reads=0,
        terminal_tail=bool(outputs>1 and step==outputs-1))
        for step in range(outputs-1+int(outputs>1)) for layer in FFN_LAYERS]
    calls=[]
    for _ in range(placeholder_count):
        calls.append(dict(invocation_index=len(calls),kind="placeholder",request_id="placeholder",seq_id=0,
            is_prompt=False,is_placeholder=True,seqlen_q=1,seqlen_k=1,phase=phase,real_decode_step=None,
            layer_indices=list(range(40)),layer_rows=[1]*40,complete=True))
    for start in range(0,n,8192):
        end=min(start+8192,n);idx=len(calls)
        calls.append(dict(invocation_index=idx,kind="request",request_id=request_id,seq_id=17,
            is_prompt=True,is_placeholder=False,seqlen_q=end-start,seqlen_k=end,phase=phase,real_decode_step=None,
            layer_indices=list(range(40)),layer_rows=[end-start]*20+[1]*20,complete=True))
        for event in events:
            if event["chunk_start"]==start:event["scheduler_invocation_index"]=idx
    for step in range(outputs-1+int(outputs>1)):
        idx=len(calls)
        calls.append(dict(invocation_index=idx,kind="request",request_id=request_id,seq_id=17,
            is_prompt=False,is_placeholder=False,seqlen_q=1,seqlen_k=n+step+1,phase=phase,real_decode_step=step,
            layer_indices=list(range(40)),layer_rows=[1]*40,complete=True))
        for event in decode:
            if event["decode_step"]==step:event["scheduler_invocation_index"]=idx
    return dict(schema="swa-skip-controller-v2", attention_variant="global_reference" if reference else "global_only",
        request_id=request_id,phase=phase,scheduler_identity_schema="native-sequence-whole-forward-v1",
        native_placeholder_request_id="placeholder",scheduler_invocations=calls,scheduler_invocations_sha256=digest(calls),
        scheduler_counts=dict(total=len(calls),real_prefill=(n+8191)//8192,
            real_decode=outputs-1+int(outputs>1),placeholder_prefill=0,placeholder_decode=placeholder_count),
        all_query_attention_native=False, global_skip_events=events, decode_skip_events=decode,
        emitted_output_tokens=outputs, decode_steps=outputs-1+int(outputs>1), effective_decode_steps=outputs-1,
        terminal_tail_decode_steps=int(outputs>1), terminal_tail_decode_step=outputs-1 if outputs>1 else None,
        decode_schedule=dict(schema="native-zos-mtp0-one-terminal-tail-after-decode-v1",
            native_zero_overhead_scheduling=True, output_tokens=outputs, output_driving_steps=outputs-1,
            terminal_tail_steps=int(outputs>1), actual_steps=outputs-1+int(outputs>1)))


def global_pair():
    result=[]
    fp=dict(shape=[1,2],dtype="torch.bfloat16",bytes=4,sha256="a"*64)
    value=dict(output=fp,lazy={k:fp for k in ("residual","post_layer_mix","comb_res_mix","pre_layer_mix")})
    prefill=[dict(key=[s,s+8192,l],value=copy.deepcopy(value)) for s in (0,8192) for l in range(40)]
    decode=[dict(key=[step,l],value=copy.deepcopy(value)) for step in range(2) for l in range(40)]
    for reference in (True,False):
        name="qualification."+("global_reference" if reference else "global_only")
        rid="swa-skip:0:"+name+":qualification-"+"b"*64
        control=global_events(16384,reference=reference,outputs=2,request_id=rid,phase=name,
            placeholder_count=1 if reference else 2)
        mode="qualified_native" if reference else "all_selected"
        control.update(input_ids_sha256="b"*64,full_input_tokens=16384,identity_check_count=80,identity_fingerprints=prefill,
            phase=name,mode=mode,decode_identity_check_count=80,
            identity_fingerprints_sha256=digest(prefill),decode_identity_fingerprints=decode,
            decode_identity_fingerprints_sha256=digest(decode),global_identity_exact=not reference)
        result.append(seal(dict(schema=runner.PHASE_SCHEMA,index=0,arm="qualification",
            name=name,repeat=None,mode=mode,request_id=rid,
            attention_variant=control["attention_variant"],timing_valid=False,quality_eligible=False,
            input_ids_sha256="b"*64,execution_configuration_sha256="c"*64,
            swa_skip_protocol_sha256="d"*64,manifest_sha256="e"*64,empty_ep_qualification_sha256="f"*64,
            record=dict(success=True,rank=0,phase=name,request_id=rid,input_tokens=16384,input_ids_sha256="b"*64,output_ids=[4,5],
                output_ids_sha256=digest([4,5]),natural_EOS=False),controller=control)))
    return result


class ProtocolTests(unittest.TestCase):
    def test_exact_new_scheme_and_unchanged_ffn_fraction(self):
        p=swa_skip_protocol()
        self.assertEqual(p["sha256"],digest({k:v for k,v in p.items() if k!="sha256"}))
        self.assertEqual(p["first_full_layers"],list(range(6)))
        self.assertEqual(p["score_layers_zero_based"],[2,3,4,5])
        self.assertEqual(p["optional_fraction"],.15)
        self.assertFalse(p["post6_swa_projection"])
        self.assertFalse(p["post6_swa_attention"])
        self.assertTrue(p["includes_decode_swa_skip"])
        self.assertTrue(p["decode_ffn_native"])
        self.assertEqual(p["offline_swa_kv_attention_consumed_rows"],0)
        self.assertFalse(p["selected_tokens_forced_through_all_layers"])
        self.assertEqual(p["online_yoco_native_compact_keep_tokens"],140)

    def test_fixed_two_arm_three_pair_timing(self):
        for index in (2,3,4,5):
            self.assertEqual(runner.timing_schedule(index),timing_schedule(index))
            self.assertEqual(len(timing_schedule(index)),6)

    def test_imports_are_cpu_only(self):
        subprocess.run([sys.executable,"-B","-c",
            "import sys; from gpu_pilot import run_swa_skip_pilot,read_swa_skip_results,swa_skip_protocol; assert 'torch' not in sys.modules"],check=True)

    def test_new_cli_removed_from_inherited_parser(self):
        with patch.object(runner.support,"parse_args",return_value=type("Args",(),{})()) as parse:
            args=runner.parse_args(["--swa-skip-protocol","/tmp/new.json","--manifest","/tmp/m.json"])
        self.assertEqual(args.swa_skip_protocol,Path("/tmp/new.json"))
        parse.assert_called_once_with(["--manifest","/tmp/m.json"])


class AtomicEvidenceTests(unittest.TestCase):
    def test_atomic_compact_publish_and_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw.json"; value=seal(dict(numbers=[1.,2.,3.]))
            runner.write_json(path,value)
            evidence={}; read=reader.read(path,evidence)
            ref=runner.evidence_ref(folder,path.name,value["sha256"])
            reader.verify_reference(ref,path,read,evidence)
            self.assertEqual(read,value)
            self.assertEqual(list(Path(folder).glob("*.pending-*")),[])

    def test_existing_evidence_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw.json"; value=seal(dict(n=1))
            runner.write_json(path,value)
            before=path.read_bytes()
            with self.assertRaises(ValueError):runner.write_json(path,seal(dict(n=2)))
            self.assertEqual(path.read_bytes(),before)

    def test_interrupted_write_has_no_final_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw.json"
            with patch.object(runner.json,"dump",side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):runner.write_json(path,dict(a=1))
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(folder).glob(".raw.json.pending-*"))),1)

    def test_reference_must_bind_raw_filename_bytes_and_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw.json"; value=seal(dict(n=1)); runner.write_json(path,value)
            evidence={}; reader.read(path,evidence)
            ref=runner.evidence_ref(folder,path.name,value["sha256"])
            for key,bad in (("file","../raw.json"),("file_sha256","0"*64),("seal_sha256","1"*64)):
                changed=dict(ref);changed[key]=bad
                with self.assertRaises(ValidationError):reader.verify_reference(changed,path,value,evidence)

    def test_embedded_phase_or_unexpected_reference_fields_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw.json";value=seal(dict(n=1));runner.write_json(path,value)
            evidence={};reader.read(path,evidence)
            ref=runner.evidence_ref(folder,path.name,value["sha256"]);ref["phase"]=value
            with self.assertRaises(ValidationError):reader.verify_reference(ref,path,value,evidence)

    def test_duplicate_json_keys_rejected_even_if_resealed(self):
        with self.assertRaises(ValidationError):reader._unique([("a",1),("a",1)])


class FailureDiagnosticsTests(unittest.TestCase):
    def test_output_progress_survives_finish_failure(self):
        state=dict(request_id="r", output_ids=[], emitted_output_tokens=0)
        emit=runner.phase_observer(state)
        emit(dict(event="arrival",request_id="r",token_id=7,time_s=1.5,finished=False,finish_reason=None))
        emit(dict(event="arrival",request_id="r",token_id=8,time_s=1.6,finished=True,finish_reason="stop"))
        controller=type("C",(),{"failure_diagnostics":lambda self:dict(real_decode=2,placeholder=3)})()
        diagnostic=runner.failure_diagnostics(controller,state)
        self.assertEqual(diagnostic["phase"]["output_ids"],[7,8])
        self.assertEqual(diagnostic["phase"]["emitted_output_tokens"],2)
        self.assertTrue(diagnostic["phase"]["terminal_output_seen"])
        self.assertEqual(diagnostic["controller"]["placeholder"],3)

    def test_foreign_output_is_not_silently_counted(self):
        state=dict(request_id="r",output_ids=[])
        with self.assertRaises(ValueError):
            runner.phase_observer(state)(dict(event="arrival",request_id="foreign",token_id=4))

    def test_diagnostic_error_never_masks_original_exception(self):
        def broken(self):raise RuntimeError("diagnostic failed")
        controller=type("C",(),{"failure_diagnostics":broken,"global_skip_events":[1],"decode_skip_events":[2,3]})()
        diagnostic=runner.failure_diagnostics(controller,dict(emitted_output_tokens=2))
        self.assertIn("diagnostic failed",diagnostic["controller_diagnostic_error"])
        self.assertEqual(diagnostic["controller"]["real_decode_attention_events"],2)


class GlobalOnlyQualificationTests(unittest.TestCase):
    def test_two_independent_exact_paths_include_real_decode(self):
        summary=reader.validate_global_pair(*global_pair())
        self.assertEqual(summary["prefill_identity_check_count"],80)
        self.assertEqual(summary["decode_identity_check_count"],80)
        self.assertEqual(summary["terminal_tail_decode_steps"],1)
        self.assertFalse(summary["native_full_model_equivalence_claim"])

    def test_changed_decoder_output_or_any_mhc_fails_even_resealed(self):
        for field in ("identity_fingerprints","decode_identity_fingerprints"):
            pair=global_pair();candidate=copy.deepcopy(pair[1])
            candidate["controller"][field][0]["value"]["lazy"]["residual"]["sha256"]="0"*64
            candidate["controller"][field+"_sha256"]=digest(candidate["controller"][field])
            candidate=seal({k:v for k,v in candidate.items() if k!="sha256"})
            with self.assertRaises(ValidationError):reader.validate_global_pair(pair[0],candidate)

    def test_missing_fingerprint_cannot_be_replaced_by_count(self):
        pair=global_pair();candidate=copy.deepcopy(pair[1]);candidate["controller"]["decode_identity_fingerprints"].pop()
        candidate=seal({k:v for k,v in candidate.items() if k!="sha256"})
        with self.assertRaises(ValidationError):reader.validate_global_pair(pair[0],candidate)

    def test_one_token_smoke_not_a_decode_qualification(self):
        pair=global_pair();candidate=copy.deepcopy(pair[1]);candidate["record"]["output_ids"]=[4]
        candidate["record"]["output_ids_sha256"]=digest([4]);candidate=seal({k:v for k,v in candidate.items() if k!="sha256"})
        with self.assertRaises(ValidationError):reader.validate_global_pair(pair[0],candidate)

    def test_record_and_controller_must_bind_actual_qualification_input(self):
        for field in ("record","controller"):
            pair=global_pair();candidate=copy.deepcopy(pair[1]);candidate[field]["input_ids_sha256"]="0"*64
            candidate=seal({k:v for k,v in candidate.items() if k!="sha256"})
            with self.assertRaises(ValidationError):reader.validate_global_pair(pair[0],candidate)

    def test_qualification_name_and_repeat_are_not_interchangeable(self):
        for key,bad in (("name","qualification.native"),("repeat",0)):
            pair=global_pair();candidate=copy.deepcopy(pair[1]);candidate[key]=bad
            candidate=seal({k:v for k,v in candidate.items() if k!="sha256"})
            with self.assertRaises(ValidationError):reader.validate_global_pair(pair[0],candidate)

    def test_qualification_must_bind_manifest_prefix_in_preflight(self):
        phase=global_pair()[0]
        reader.qualification_input_binding(phase,dict(qualification_input_ids_sha256="b"*64))
        with self.assertRaises(ValidationError):
            reader.qualification_input_binding(phase,dict(qualification_input_ids_sha256="0"*64))

    def test_zos_terminal_tail_cannot_be_dropped_or_arbitrarily_extended(self):
        value=global_events(10,outputs=2)
        value["decode_skip_events"]=[e for e in value["decode_skip_events"] if not e["terminal_tail"]]
        with self.assertRaises(ValidationError):reader.audit_global_events(value,10,2)
        value=global_events(10,outputs=2);value["decode_schedule"]["actual_steps"]=3
        with self.assertRaises(ValidationError):reader.audit_global_events(value,10,2)

    def test_zos_has_no_tail_after_single_prefill_output(self):
        value=global_events(10,outputs=1)
        self.assertEqual(reader.audit_global_events(value,10,1),sum(e["full_rows"] for e in value["global_skip_events"]))
        value["terminal_tail_decode_steps"]=1
        with self.assertRaises(ValidationError):reader.audit_global_events(value,10,1)

    def test_full_projection_cannot_masquerade_as_optimized(self):
        value=global_events(10);value["global_skip_events"][0]["swa_projection_rows"]=10
        with self.assertRaises(ValidationError):reader.audit_global_events(value,10,1)

    def test_any_local_attention_or_cache_consumption_rejected(self):
        for key in ("swa_attention_query_rows","swa_attention_kv_reads","offline_kv_attention_reads"):
            value=global_events(10);value["global_skip_events"][0][key]=1
            with self.assertRaises(ValidationError):reader.audit_global_events(value,10,1)

    def test_missing_or_duplicate_prefill_event_fails(self):
        for duplicate in (False,True):
            value=global_events(10)
            if duplicate:value["global_skip_events"].append(value["global_skip_events"][0])
            else:value["global_skip_events"].pop()
            with self.assertRaises(ValidationError):reader.audit_global_events(value,10,1)

    def test_decode_cannot_silently_restore_native_swa(self):
        value=global_events(10,outputs=2);value["decode_skip_events"][0]["swa_attention_query_rows"]=1
        with self.assertRaises(ValidationError):reader.audit_global_events(value,10,2)

    def test_decode_step_and_position_are_bound(self):
        for key,bad in (("decode_step",1),("current_positions",[9])):
            value=global_events(10,outputs=2);value["decode_skip_events"][0][key]=bad
            with self.assertRaises(ValidationError):reader.audit_global_events(value,10,2)

    def test_reference_must_really_project_but_not_attend_swa(self):
        value=global_events(10,reference=True);value["global_skip_events"][0]["swa_projection_rows"]=0
        with self.assertRaises(ValidationError):reader.audit_global_events(value,10,1,reference=True)


class SchedulerIdentityTests(unittest.TestCase):
    def test_placeholders_are_retained_but_never_real_tokens(self):
        value=global_events(10,outputs=2,placeholder_count=7)
        result=reader.audit_scheduler_identity(value,10)
        self.assertEqual(len(result["decodes"]),2)
        self.assertEqual(len(result["placeholders"]),7)
        reader.audit_global_events(value,10,2)

    def test_placeholder_count_can_differ_between_numerical_paths(self):
        pair=global_pair()
        self.assertNotEqual(pair[0]["controller"]["scheduler_counts"]["placeholder_decode"],
                            pair[1]["controller"]["scheduler_counts"]["placeholder_decode"])
        reader.validate_global_pair(*pair)

    def test_unknown_request_or_inconsistent_placeholder_fails_resealed(self):
        for field,bad in (("request_id","foreign"),("is_placeholder",False),("seq_id",1),("seqlen_k",2)):
            value=global_events(10);value["scheduler_invocations"][0][field]=bad
            value["scheduler_invocations_sha256"]=digest(value["scheduler_invocations"])
            with self.assertRaises(ValidationError):reader.audit_scheduler_identity(value,10)

    def test_missing_layer_in_placeholder_cannot_be_hidden_by_totals(self):
        value=global_events(10);value["scheduler_invocations"][0]["layer_indices"].pop()
        value["scheduler_invocations_sha256"]=digest(value["scheduler_invocations"])
        with self.assertRaises(ValidationError):reader.audit_scheduler_identity(value,10)

    def test_real_attention_event_cannot_bind_placeholder(self):
        value=global_events(10);value["global_skip_events"][0]["scheduler_invocation_index"]=0
        with self.assertRaises(ValidationError):reader.audit_global_events(value,10,1)

    def test_real_decode_position_and_request_sequence_are_bound(self):
        for field,bad in (("seqlen_k",50),("seq_id",99),("real_decode_step",5),("is_placeholder",True)):
            value=global_events(10,outputs=2);value["scheduler_invocations"][-1][field]=bad
            value["scheduler_invocations_sha256"]=digest(value["scheduler_invocations"])
            with self.assertRaises(ValidationError):reader.audit_scheduler_identity(value,10)

    def test_prompt_placeholder_is_separate_and_still_native_complete(self):
        value=global_events(10);value["scheduler_invocations"][0]["is_prompt"]=True
        value["scheduler_invocations_sha256"]=digest(value["scheduler_invocations"])
        value["scheduler_counts"].update(placeholder_prefill=1,placeholder_decode=0)
        self.assertEqual(len(reader.audit_scheduler_identity(value,10)["prompts"]),1)


class SnapshotTests(unittest.TestCase):
    def test_wrong_native_root_rejected_without_import(self):
        args=type("Args",(),dict(repo=Path("/tmp/not-the-isolated-snapshot"),experiment_root=Path("/tmp")))()
        with self.assertRaises(ValueError):runner.bind_native_snapshot(args)

    def test_foreign_native_import_rejected(self):
        foreign=type("Module",(),{"__file__":"/tmp/live-native/deepinfer/model.py"})()
        with patch.dict(sys.modules,{"deepinfer.foreign_test":foreign}):
            with self.assertRaises(ValueError):runner.verify_native_imports(Path("/tmp/frozen-native"))


if __name__=="__main__":unittest.main()
