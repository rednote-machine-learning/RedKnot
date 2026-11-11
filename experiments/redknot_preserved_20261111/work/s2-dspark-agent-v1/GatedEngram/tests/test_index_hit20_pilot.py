"""Independent protocol/reader CPU tests; never GPU qualification claims."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gpu_pilot import index_hit20_protocol as protocol
from gpu_pilot import read_index_hit20_results as reader
from gpu_pilot import run_index_hit20_pilot as runner
from kv_cache_engram.core import ValidationError,digest


def seal(value):
    return dict(value,sha256=digest(value))


def controller_fixture():
    row=dict(index=2,input_tokens=20,input_ids_sha256="a"*64)
    phase=dict(mode="sensitive",arm="index20",name="test",record=dict(request_id="test-request"))
    spans=[(1,19,"doc")]
    masks=[];groups={};swa=[];layouts=[];attn=[];events=[]
    for layer in range(3,40):
        source=max(i for i in (2,8,14,20,24,28,32,36) if i<=layer)
        positions=list(range(20)) if layer<20 else list(range(15,20))
        ratio=2 if layer<20 else 1
        hits=[p%3 for p in positions]
        ident="g"+str(source)
        groups.setdefault(ident,dict(group_id=ident,source_layer=source,chunk_start=0,chunk_end=20,
            rows=len(positions),columns=512,compression_ratio=ratio,request_kv_offset=0,
            global_kv_rows=20//ratio,query_positions=positions,hitcounts=hits,raw_indices_recorded=False))
        ids,mandatory,optional=reader.mask_reference(hits,positions,0,20,spans)
        plan=digest(dict(input_sha256=row["input_ids_sha256"],start=0,end=20,layer=layer,
            group=ident,mode="sensitive",fraction=.2,policy="raw-index-hit-frequency-current-native-row-top20-position-tie"))
        doc_ids=[i for i,p in enumerate(positions) if 1<=p<19]
        m=dict(start=0,end=20,layer=layer,full_rows=len(positions),current_positions=positions,
            selected_local_rows=ids,selected_rows=len(ids),selected_rows_sha256=digest(ids),hitcounts=hits,
            index_group_id=ident,compression_ratio=ratio,mandatory_rows=mandatory,optional_rows=optional,
            plan_sha256=plan,zero_hit_document_rows=sum(hits[i]==0 for i in doc_ids),
            selected_zero_hit_document_rows=sum(hits[i]==0 for i in ids if i in doc_ids))
        masks.append(m)
        layout={k:m[k] for k in ("layer","full_rows","current_positions","selected_local_rows","selected_rows","plan_sha256")}
        layout.update(chunk_start=0,chunk_end=20);layouts.append(layout)
        swa.append(dict(layer=layer,chunk_start=0,chunk_end=20,full_rows=len(positions),current_positions=positions,
            online_local_rows=ids,online_rows=len(ids),reused_rows=len(positions)-len(ids),compact=layer>=20))
        local=layer in (8,14,20,24,28,32,36)
        attn.append(dict(layer=layer,chunk_start=0,chunk_end=20,full_rows=len(positions),selected_rows=len(ids),
            main_q_a_rows=len(positions) if local else len(ids),index_q_rows=len(positions) if local else 0,
            main_q_b_rows=len(ids),joint_attention_rows=len(ids),output_projection_rows=len(ids),
            inactive_attention_zero_rows=len(positions)-len(ids),global_kv_rows=20//ratio,
            global_compute_called=True,is_global_source=layer in (8,14,20),native_joint_attention=True,
            reference_dense=False,plan_sha256=plan))
        events.append(dict(layer=layer,request_id="test-request",phase="test",chunk_start=0,chunk_end=20,
            full_rows=len(positions),selected_rows=len(ids),actual_mlp_rows=len(ids),reference_dense=False,
            plan_sha256=plan,current_layout="native_current_rows",inactive_ffn_output="zero",
            all_selected_native=len(ids)==len(positions),empty_native_collective=False))
    full=sum(m["full_rows"] for m in masks);selected=sum(m["selected_rows"] for m in masks)
    adapter=dict(schema="kv-cache-engram-index-hit20-adapter-v1",events=events,
        counts=dict(full_rows=full,selected_rows=selected,actual_mlp_rows=selected,inactive_ffn_zero_rows=full-selected,
            empty_native_mlp_calls=0,reference_dense_mlp_calls=0))
    for k in ("decoder_forward_unchanged","attention_forward_unchanged","native_mhc_unchanged",
              "native_yoco_layout_unchanged","all_selected_delegates_native"):adapter[k]=True
    for k in ("empty_selection_skips_ep_collective","cached_hidden_or_ffn_output","accuracy_guarantee","speedup_claim"):adapter[k]=False
    control=dict(schema="index-hit20-controller-v1",mode="sensitive",phase="test",input_ids_sha256=row["input_ids_sha256"],
        full_input_tokens=20,ffn_layers=list(range(3,40)),history_kv_rewritten=False,
        selected_tokens_forced_through_yoco=False,formal_raw_index_replay_claim=False,selection=masks,
        index_groups=list(groups.values()),swa_events=swa,ffn_layouts=layouts,attention_events=attn,adapter=adapter)
    for k in ("first_three_full","actual_full_row_native_mhc","yoco_layout_preserved","formal_admission_unchanged",
              "decode_native","global_kv_native_online","not_model_flops_fraction","dynamic_mask_sync_in_online_timing"):
        control[k]=True
    return control,row,phase,spans


def raw_index_fixture(root, mutation=None):
    import numpy as np
    sub=root/"qualification.index";sub.mkdir()
    values=np.full((3,512),-1,dtype="<i4")
    values[0,0]=0;values[1,0]=0;values[2,0]=1
    if mutation is not None:mutation(values)
    raw=values.tobytes();(sub/"g0.i32").write_bytes(raw)
    return dict(raw_index_oracle_checked=True,index_groups=[dict(group_id="g0",
        file="qualification.index/g0.i32",rows=3,columns=512,bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),raw_indices_recorded=True,cpu_raw_oracle_exact=True,
        dtype="little-endian-int32",query_positions=[1,2,3],compression_ratio=2,
        global_kv_rows=2,request_kv_offset=0,hitcounts=[2,1,1])])


class IndexHit20PilotTests(unittest.TestCase):
    def test_protocol_is_first3_and20_not_previous15(self):
        value=protocol.index_hit20_protocol()
        self.assertEqual(value["first_full_layers"],[0,1,2])
        self.assertEqual(value["optional_fraction"],.20)
        self.assertEqual(value["ffn_layers"],list(range(3,40)))
        self.assertEqual(value["arms"],["recomputed","index20"])
        self.assertTrue(value["global_kv_complete_online"])
        self.assertTrue(value["decoder_prefill_selected_by_hit_frequency"])
        self.assertFalse(value["historical_chunk_kv_rewritten"])
        self.assertEqual(value["empty_ep_qualification"]["checks"],1184)
        self.assertFalse(value["selection_audit"]["formal_raw_topk_rescan_claim"])
        self.assertEqual(value["sha256"],digest({k:v for k,v in value.items() if k!="sha256"}))

    def test_no_gpu_import(self):
        code="import sys;import gpu_pilot.index_hit20_protocol,gpu_pilot.read_index_hit20_results,gpu_pilot.run_index_hit20_pilot;assert 'torch' not in sys.modules;assert not any(x.startswith('deepinfer') for x in sys.modules)"
        result=subprocess.run([sys.executable,"-B","-c",code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_original_four_cases_two_datasets(self):
        self.assertEqual(runner.EVALUATION_INDICES,(2,3,4,5))
        self.assertEqual(reader.CASE_INDICES,(2,3,4,5))
        self.assertEqual(runner.ARMS,("recomputed","index20"))

    def test_rotation_three_repeats(self):
        for index in runner.EVALUATION_INDICES:
            schedule=runner.timing_schedule(index)
            self.assertEqual(schedule,protocol.timing_schedule(index))
            for arm in runner.ARMS:self.assertEqual(sum(a==arm for _,a in schedule),3)

    def test_new_cli_consumed(self):
        from unittest.mock import patch
        with patch.object(runner.support,"parse_args",return_value=type("Args",(),{})()) as parsed:
            args=runner.parse_args(["--index-hit20-protocol","new.json","--check-only"])
        self.assertEqual(args.index_hit20_protocol,Path("new.json"))
        self.assertEqual(parsed.call_args.args[0],["--check-only"])

    def test_integer_frequency_not_intersection(self):
        ids,mandatory,optional=reader.mask_reference([0,2,5,1,0],list(range(5)),0,5,[(0,5,"a")])
        self.assertEqual((ids,mandatory,optional),([2],0,5))

    def test_ties_absolute_position_and_ceil(self):
        ids,_,_=reader.mask_reference([4]*6,[1,3,5,7,9,11],0,12,[(0,12,"a")])
        self.assertEqual(ids,[0,1])

    def test_mandatory_is_extra_not20_budget(self):
        ids,mandatory,optional=reader.mask_reference([99,8,7,6,5,4,0],list(range(7)),0,7,[(1,6,"a")])
        self.assertEqual((ids,mandatory,optional),([0,1,6],2,5))

    def test_yoco_ranks_survivors_not_full_document(self):
        ids,_,optional=reader.mask_reference([1,100,2,4,3],[70,90,95,98,99],0,100,[(0,100,"a")])
        self.assertEqual(ids,[1]);self.assertEqual(optional,5)

    def test_zero_frequency_is_deterministic_not_random(self):
        ids,_,_=reader.mask_reference([0]*5,[11,12,13,14,15],10,20,[(10,20,"a")])
        self.assertEqual(ids,[0])

    def test_no_doc_rows_preserves_templates(self):
        self.assertEqual(reader.mask_reference([0,0],[8,9],0,10,[]),([0,1],2,0))

    def test_empty_native_layout_allowed(self):
        self.assertEqual(reader.mask_reference([],[],0,10,[(0,10,"a")]),([],0,0))

    def test_invalid_counts_positions_fraction_rejected(self):
        for counts,pos,fraction in (([1.0],[0],.2),([-1],[0],.2),([True],[0],.2),([0],[0],.15),([0,1],[0],.2),([0,1],[1,0],.2),([0],[10],.2)):
            with self.assertRaises(ValidationError):reader.mask_reference(counts,pos,0,10,[(0,10,"a")],fraction)

    def test_seal_duplicate_nonfinite_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p=root/"bad.json"
            for raw in ('{"a":1,"a":2,"sha256":"x"}','{"a":NaN,"sha256":"x"}','{"a":1,"sha256":"x"}'):
                p.write_text(raw)
                with self.assertRaises((ValueError,ValidationError)):reader.read(p,{})
            p.write_text(json.dumps(seal({"a":1})))
            link=root/"link.json";link.symlink_to(p)
            with self.assertRaises(ValidationError):reader.read(link,{})

    def test_atomic_publishing_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"result.json"
            runner.write_json(p,seal({"a":1}))
            before=p.read_bytes()
            with self.assertRaises(ValueError):runner.write_json(p,seal({"a":2}))
            self.assertEqual(p.read_bytes(),before)

    def test_raw_reference_binds_exact_bytes_and_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"result.json";body=seal({"a":1});runner.write_json(p,body)
            ref=runner.evidence_ref(folder,p.name,body["sha256"]);evidence={};value=reader.read(p,evidence)
            reader.verify_reference(ref,p,value,evidence)
            bad=dict(ref,file_sha256="0"*64)
            with self.assertRaises(ValidationError):reader.verify_reference(bad,p,value,evidence)

    def test_quality_zero_and_truncation_not_filtered(self):
        self.assertEqual(reader.score("wrong",["gold"]),dict(answer_em=0.,answer_token_f1=0.))
        self.assertEqual(reader.score("the answer",["answer"]),dict(answer_em=1.,answer_token_f1=1.))

    def test_json_limit_preserves_large_file_requirement(self):
        self.assertEqual(reader.MAX_JSON_BYTES,8*1024**3)

    def test_actual_layer_accounting_independent_audit(self):
        values=controller_fixture();result=reader.audit_controller(*values)
        self.assertGreater(result["swa_reused_rows"],0)
        self.assertEqual(result["ffn_selected_rows"],result["swa_online_rows"])
        self.assertEqual(result["ffn_selected_rows"],result["attention_selected_rows"])
        self.assertGreater(result["decoder_selected_rows"],0)

    def test_ranked_mask_mutation_rejected(self):
        control,row,phase,spans=controller_fixture()
        control["selection"][0]["selected_local_rows"]=[]
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_count_vector_group_binding_rejected(self):
        control,row,phase,spans=controller_fixture()
        control["selection"][0]["hitcounts"]=list(control["selection"][0]["hitcounts"])
        control["selection"][0]["hitcounts"][0]+=1
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_missing_decoder_layer_rejected(self):
        control,row,phase,spans=controller_fixture();control["selection"].pop()
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_old15_or_early_six_cannot_pass(self):
        for field,value in (("first_three_full",False),("ffn_layers",list(range(6,40)))):
            control,row,phase,spans=controller_fixture();control[field]=value
            with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_full_attention_fallback_cannot_be_called_sparse(self):
        control,row,phase,spans=controller_fixture()
        control["attention_events"][0]["joint_attention_rows"]=20
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_full_ffn_fallback_cannot_be_called_sparse(self):
        control,row,phase,spans=controller_fixture()
        control["adapter"]["events"][0]["actual_mlp_rows"]=20
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_extra_swa_recomputation_rejected(self):
        control,row,phase,spans=controller_fixture();control["swa_events"][0]["online_rows"]+=1
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_index_source_and_ratio_must_match_native_map(self):
        for field,value in (("source_layer",3),("compression_ratio",1),("request_kv_offset",1)):
            control,row,phase,spans=controller_fixture();control["index_groups"][0][field]=value
            with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_same_total_cannot_hide_duplicate_event(self):
        control,row,phase,spans=controller_fixture()
        control["swa_events"][-1]=copy.deepcopy(control["swa_events"][0])
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_raw_topk_qualification_cpu_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);control=raw_index_fixture(root);evidence={}
            self.assertEqual(reader.audit_raw_index_oracle(root,control,evidence),1)
            self.assertIn("qualification.index/g0.i32",evidence)

    def test_raw_topk_changed_byte_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);control=raw_index_fixture(root)
            path=root/"qualification.index/g0.i32";raw=bytearray(path.read_bytes());raw[0]=1;path.write_bytes(raw)
            with self.assertRaises(ValidationError):reader.audit_raw_index_oracle(root,control,{})

    def test_raw_topk_noncausal_and_duplicate_rejected(self):
        for mutation in (lambda a:a.__setitem__((0,0),1),lambda a:a.__setitem__((0,1),0)):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);control=raw_index_fixture(root,mutation)
                with self.assertRaises(ValidationError):reader.audit_raw_index_oracle(root,control,{})

    def test_raw_topk_count_summary_not_trusted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);control=raw_index_fixture(root);control["index_groups"][0]["hitcounts"][0]=1
            with self.assertRaises(ValidationError):reader.audit_raw_index_oracle(root,control,{})

    def test_raw_topk_path_escape_and_missing_raw_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);control=raw_index_fixture(root)
            for field,value in (("file","../g0.i32"),("file","/tmp/g0.i32"),("raw_indices_recorded",False)):
                bad=copy.deepcopy(control);bad["index_groups"][0][field]=value
                with self.assertRaises(ValidationError):reader.audit_raw_index_oracle(root,bad,{})


if __name__=="__main__":unittest.main()
