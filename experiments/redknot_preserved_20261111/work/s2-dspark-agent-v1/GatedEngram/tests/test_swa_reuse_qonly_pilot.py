"""CPU contracts for cached SWA Q-only hits and unchanged native joint attention."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.core import ValidationError,digest
from gpu_pilot import run_swa_reuse_qonly_pilot as runner
from gpu_pilot import read_swa_reuse_qonly_results as reader
from gpu_pilot.swa_reuse_qonly_protocol import swa_reuse_qonly_protocol,ARMS,FFN_LAYERS,timing_schedule
from tests.test_ffn_only_pilot import fixture as original_fixture, split_fixture


def seal(value):return dict(value,sha256=digest(value))


def joint_events(swa):
    return [dict(layer=e["layer"],chunk_start=e["chunk_start"],chunk_end=e["chunk_end"],
        full_rows=e["full_rows"],online_swa_rows=e["online_rows"],reused_swa_rows=e["reused_rows"],
        swa_prefix_rows=0,current_swa_rows=e["full_rows"],global_kv_rows=2,kv_rows=e["full_rows"]+2,
        use_swa=True,has_packed_kv=True,global_compute_called=True,native_joint_attention_called=True,
        native_swa_indices_preserved=True,is_global_source=e["layer"] in (8,14,20),uses_shared_global_kv=True) for e in swa]


def fixture():
    control,row,phase,spans=original_fixture()
    control.update(schema="swa-reuse-qonly-ffn15-controller-v1",
        swa_reuse_policy="all-cached-document-rows-no-boundary-repair",document_boundary_online_tokens=0,
        online_swa_scope="non-document-template-question-only",decode_native=True,
        joint_attention_native=True,global_kv_native_online=True)
    for e in control["swa_events"]:
        online=[i for i,p in enumerate(e["current_positions"]) if not any(lo<=p<hi for lo,hi,_ in spans)]
        e.update(online_local_rows=online,online_rows=len(online),reused_rows=e["full_rows"]-len(online))
    joint=joint_events(control["swa_events"])
    copies=[dict(layer=e["layer"],chunk_start=0,chunk_end=row["input_tokens"],copied_document_rows=508,
        canonical_bf16_bytes=508*512*2,restored_current_rows=e["full_rows"]) for e in control["swa_events"] if e["full_rows"]]
    control.update(joint_attention_events=joint,joint_attention_events_sha256=digest(joint),
        effective_cache_attention_rows=sum(e["reused_swa_rows"] for e in joint),canonical_copy_events=copies,
        canonical_cache_copy_bytes=sum(e["canonical_bf16_bytes"] for e in copies),
        bytes_are_canonical_copies_not_measured_device_bandwidth=True)
    return control,row,phase,spans


def qualification_fixture():
    control=split_fixture()
    events=[dict(layer=e["layer"],chunk_start=e["chunk_start"],chunk_end=e["chunk_end"],full_rows=e["full_rows"],
        checks=2,max_absolute_error=0.,checks_named=["canonical_norm_rope_cast","assembled_swa_rows"])
        for e in control["swa_events"] if e["full_rows"]]
    joint=joint_events(control["swa_events"])
    control.update(cache_restore_identity_events=events,cache_restore_identity_checks=len(events)*2,
        joint_attention_events=joint,joint_attention_events_sha256=digest(joint),effective_cache_attention_rows=0)
    return control


class ProtocolTests(unittest.TestCase):
    def test_literal_cached_documents_and_unchanged_ffn15(self):
        p=swa_reuse_qonly_protocol()
        self.assertEqual(p["sha256"],digest({k:v for k,v in p.items() if k!="sha256"}))
        self.assertEqual(p["first_full_layers"],list(range(6)))
        self.assertEqual(p["score_layers_zero_based"],[2,3,4,5])
        self.assertEqual(p["optional_fraction"],.15)
        self.assertEqual(p["document_prefix_repair_tokens"],0)
        self.assertTrue(p["native_joint_swa_global_attention"])
        self.assertTrue(p["native_decode_unchanged"])
        self.assertTrue(p["all_query_attention_native"])
        self.assertNotIn("new_global_only_parity",p)

    def test_two_arms_three_paired_repetitions(self):
        self.assertEqual(ARMS,("recomputed","ffn15"))
        for n in (2,3,4,5):
            self.assertEqual(runner.timing_schedule(n),timing_schedule(n))
            self.assertEqual(len(timing_schedule(n)),6)

    def test_cli_new_flag_does_not_leak_to_old_parser(self):
        with patch.object(runner.support,"parse_args",return_value=type("A",(),{})()) as parse:
            args=runner.parse_args(["--swa-reuse-qonly-protocol","/tmp/p.json","--manifest","/tmp/m.json"])
        self.assertEqual(args.swa_reuse_qonly_protocol,Path("/tmp/p.json"))
        parse.assert_called_once_with(["--manifest","/tmp/m.json"])

    def test_cpu_import_never_imports_native_or_torch(self):
        subprocess.run([sys.executable,"-B","-c","import sys; from gpu_pilot import run_swa_reuse_qonly_pilot,read_swa_reuse_qonly_results; assert 'torch' not in sys.modules; assert not any(k.startswith('deepinfer') for k in sys.modules)"],check=True)

    def test_qualification_prefix_is_exact_manifest_prefix(self):
        ids=list(range(20000));row=runner.qualification_row(dict(warmup=[dict(index=0,dataset="hotpotqa",input_ids=ids)]))
        self.assertEqual(row["input_ids"],ids[:16384])
        self.assertEqual(row["input_ids_sha256"],digest(ids[:16384]))


class CacheAndAttentionTests(unittest.TestCase):
    def test_documents_all_reused_nondoc_online_ffn_mask_unchanged(self):
        control,row,phase,spans=fixture();result=reader.audit_controller(control,row,phase,spans)
        self.assertEqual(result["swa_reused_rows"],7192)
        self.assertEqual(result["swa_online_rows"],76)
        self.assertGreater(result["ffn_bypassed_rows"],0)
        self.assertEqual(result["effective_cache_attention_rows"],result["swa_reused_rows"])
        self.assertEqual(control["selection"],original_fixture()[0]["selection"])

    def test_old_prefix127_repair_is_rejected(self):
        control,row,phase,spans=fixture();old=original_fixture()[0]
        control["swa_events"]=old["swa_events"]
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_sensitive15_must_not_be_added_to_swa_online_rows(self):
        control,row,phase,spans=fixture();e=control["swa_events"][0]
        e["online_local_rows"].append(400);e["online_rows"]+=1;e["reused_rows"]-=1
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_native_joint_attention_must_consume_both_banks(self):
        for field in ("use_swa","has_packed_kv","global_compute_called","native_joint_attention_called","native_swa_indices_preserved"):
            control,row,phase,spans=fixture();control["joint_attention_events"][0][field]=False
            control["joint_attention_events_sha256"]=digest(control["joint_attention_events"])
            with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_missing_duplicate_or_wrong_joint_rows_fail(self):
        for change in (lambda x:x.pop(),lambda x:x.append(x[0]),lambda x:x[0].update(reused_swa_rows=0),lambda x:x[0].update(kv_rows=1)):
            control,row,phase,spans=fixture();change(control["joint_attention_events"])
            control["joint_attention_events_sha256"]=digest(control["joint_attention_events"])
            with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_changed_score_or_ffn_mask_still_fails(self):
        control,row,phase,spans=fixture();control["selection"][0]["adjacent_transition_relative_l2"][0][100]+=1
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_canonical_copy_bytes_are_independently_rebuilt(self):
        control,row,phase,spans=fixture();control["canonical_copy_events"][0]["canonical_bf16_bytes"]-=1024
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)

    def test_missing_copy_event_cannot_be_hidden_by_total(self):
        control,row,phase,spans=fixture();control["canonical_copy_events"].pop()
        with self.assertRaises(ValidationError):reader.audit_controller(control,row,phase,spans)


class NumericalQualificationTests(unittest.TestCase):
    def test_original_split_and_new_restore_are_exact(self):
        control=qualification_fixture()
        self.assertEqual(reader.projection_identity(control),144)
        self.assertEqual(reader.cache_restore_identity(control),96)
        self.assertEqual(reader.audit_joint_attention(control),0)

    def test_any_nonzero_restore_error_fails(self):
        control=qualification_fixture();control["cache_restore_identity_events"][0]["max_absolute_error"]=1e-30
        with self.assertRaises(ValidationError):reader.cache_restore_identity(control)

    def test_restore_count_cannot_replace_raw_checks(self):
        control=qualification_fixture();control["cache_restore_identity_events"]=[]
        with self.assertRaises(ValidationError):reader.cache_restore_identity(control)

    def test_missing_duplicate_wrong_named_restore_checks_fail(self):
        for change in (lambda x:x.pop(),lambda x:x.append(x[0]),lambda x:x[0].update(checks_named=["allclose"])):
            control=qualification_fixture();change(control["cache_restore_identity_events"])
            with self.assertRaises(ValidationError):reader.cache_restore_identity(control)

    def test_native_phases_cannot_claim_hidden_cached_intervention(self):
        value=dict(swa_events=[],joint_attention_events=[],joint_attention_events_sha256=digest([]),
            effective_cache_attention_rows=0,canonical_copy_events=[],cache_restore_identity_events=[],
            canonical_cache_copy_bytes=0,cache_restore_identity_checks=0)
        reader.audit_native_no_cache(value)
        for field,bad in (("canonical_copy_events",[{}]),("cache_restore_identity_events",[{}]),
                           ("canonical_cache_copy_bytes",1024),("cache_restore_identity_checks",2)):
            changed=copy.deepcopy(value);changed[field]=bad
            with self.assertRaises(ValidationError):reader.audit_native_no_cache(changed)


class EvidenceAndIsolationTests(unittest.TestCase):
    def test_compact_atomic_file_reference_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"phase.json";value=seal(dict(values=[1.,2.]))
            runner.write_json(path,value);evidence={};actual=reader.read(path,evidence)
            reader.verify_reference(runner.evidence_ref(folder,path.name,value["sha256"]),path,actual,evidence)
            self.assertEqual(actual,value)

    def test_publish_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"phase.json";runner.write_json(path,seal(dict(n=1)));before=path.read_bytes()
            with self.assertRaises(ValueError):runner.write_json(path,seal(dict(n=2)))
            self.assertEqual(path.read_bytes(),before)

    def test_failed_write_preserves_pending_without_final(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"phase.json"
            with patch.object(runner.json,"dump",side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):runner.write_json(path,dict(n=1))
            self.assertFalse(path.exists());self.assertEqual(len(list(Path(folder).glob(".phase.json.pending-*"))),1)

    def test_reference_cannot_inline_data_or_change_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"phase.json";value=seal(dict(n=1));runner.write_json(path,value);evidence={};reader.read(path,evidence)
            original=runner.evidence_ref(folder,path.name,value["sha256"])
            for changes in (dict(file="../phase.json"),dict(file_sha256="0"*64),dict(seal_sha256="1"*64),dict(phase=value)):
                with self.assertRaises(ValidationError):reader.verify_reference(dict(original,**changes),path,value,evidence)

    def test_json_limit_is_8gib_and_fail_closed_when_exceeded(self):
        self.assertEqual(reader.MAX_JSON_BYTES,8*1024**3)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"phase.json";runner.write_json(path,seal(dict(n=1)))
            with patch.object(reader,"MAX_JSON_BYTES",10):
                with self.assertRaises(ValidationError):reader.read(path,{})

    def test_symlink_and_bad_seal_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"phase.json";runner.write_json(path,seal(dict(n=1)))
            alias=Path(folder)/"alias.json";alias.symlink_to(path)
            with self.assertRaises(ValidationError):reader.read(alias,{})
            path.write_text('{"n":1,"sha256":"bad"}')
            with self.assertRaises(ValidationError):reader.read(path,{})

    def test_duplicate_keys_and_nonfinite_numbers_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"phase.json"
            for content in ('{"n":1,"n":2}','{"n":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):reader.read(path,{})

    def test_foreign_native_root_or_import_is_rejected(self):
        args=type("A",(),dict(repo=Path("/tmp/live"),experiment_root=Path("/tmp")))()
        with self.assertRaises(ValueError):runner.bind_native_snapshot(args)
        foreign=type("M",(),dict(__file__="/tmp/live/deepinfer/x.py"))()
        with patch.dict(sys.modules,{"deepinfer.foreign_test":foreign}):
            with self.assertRaises(ValueError):runner.verify_native_imports(Path("/tmp/frozen"))


if __name__=="__main__":unittest.main()
