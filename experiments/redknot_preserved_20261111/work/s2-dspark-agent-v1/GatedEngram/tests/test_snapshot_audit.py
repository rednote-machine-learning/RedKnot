"""CPU-only fake-driver capture-to-audit integration, not real model evidence."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from kv_cache_engram.core import ValidationError, digest
from kv_cache_engram.gate import GatePolicy
from kv_cache_engram.offline import (CAPTURE_POINT, CaptureSpec, Snapshot,
    SnapshotWriter, capture_snapshot)
from kv_cache_engram.snapshot_audit import (audit_snapshots, pair_from_snapshots,
    snapshot_pair_provenance)


class FakeAttention:
    use_swa=True
    window_size=128
    is_kv_source=False
    is_index_source=False
    is_index_k_source=False
    is_yoco_hidden_consumer_layer=False
    yoco_swa_share_hidden=False

    def __init__(self,layer,width):
        self.layer_idx=layer;self.qk_head_dim=width
        self.kv_swa_norm=SimpleNamespace(weight=[1.]*width,variance_epsilon=1e-5,out_scale=1.)
        self.calls=0;self.output=object()

    def _prepare_swa_kv(self,raw,cache,metadata,use_cp):
        self.calls+=1
        return self.output


def driver(spec,session):
    # Token-derived synthetic projection stays equal under prefix insertion;
    # unlike a real model, this fake deliberately has no contextual mixing.
    for start in range(0,len(spec.input_ids),3):
        end=min(start+3,len(spec.input_ids))
        # Independently enumerate the causal window from absolute key IDs;
        # do not call the implementation-under-test's pivot/index validator.
        pivots=[];indices=[]
        for query in range(start,end):
            visible=range(max(0,query-127),query+1)
            history=[key-(start-128) for key in visible if key<start]
            fresh=[128+key-start for key in visible if key>=start]
            pivots.append((history[0] if history else 128,len(history),fresh[0],len(fresh)))
            indices.append(tuple(history+fresh+[-1]*(128-len(history)-len(fresh))))
        metadata=SimpleNamespace(is_prompt=True,num_reqs=1,total_q=end-start,
            positions=list(range(start,end)),cu_seqlens_q=[0,end-start],
            total_k_swa=128,kv_swa_full_index_pivots=tuple(pivots),
            kv_swa_full_index=tuple(indices),
            canonical_request_binding=session.chunk_marker(start,end))
        for layer,attention in session.attentions.items():
            raw=[[float(token),float(layer)] for token in spec.input_ids[start:end]]
            assert attention._prepare_swa_kv(raw,None,metadata,False) is attention.output


def provenance(spec,operator="python-float64-rmsnorm-reference"):
    native=operator=="deepinfer-native-fused-rmsnorm-only"
    return dict(adapter="deepinfer-v4-swa-capture-v1",capture_point=CAPTURE_POINT,
        request_id=spec.request_id,phase=spec.phase,input_ids_sha256=spec.request_sha256,
        input_token_count=len(spec.input_ids),document_start=spec.document_start,
        document_end=spec.document_end,operator=operator,native_operator=native,
        accelerator_capture=native,streaming=True,
        chunks=[dict(layer=layer,start=0,end=len(spec.input_ids),document_rows=len(spec.document_ids))
                for layer in spec.layers],context_equivalence_certified=False,
        native_numerical_alignment_measured=False,
        identity_sha256_by_layer={str(k):v.sha256 for k,v in spec.identities.items()})


class SnapshotAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.offline_spec=CaptureSpec("offline-request","offline-cpu","1"*64,"2"*64,"3"*64,
            tuple(range(10,26)),0,16,layers=(3,4),width=2)
        self.online_spec=replace(self.offline_spec,request_id="online-request",phase="online-cpu",
            input_ids=(99,98)+self.offline_spec.input_ids+(97,),document_start=2,document_end=18,
            kind="full_context_reference")
        self.policy=GatePolicy(enabled=True,block_size=8,probe_count=1,head_guard=1,tail_guard=1,
            relative_tolerance=0.,absolute_tolerance=0.,calibration_id="synthetic-fixture-only")

    def capture(self,name,spec):
        attentions={layer:FakeAttention(layer,spec.width) for layer in spec.layers}
        snapshot=capture_snapshot(self.root/name,spec,attentions,driver)
        self.addCleanup(snapshot.close)
        self.assertTrue(all("_prepare_swa_kv" not in vars(a) for a in attentions.values()))
        return snapshot

    def manual(self,name,spec,operator):
        writer=SnapshotWriter(self.root/name,spec)
        for layer in spec.layers:writer.append(layer,0,tuple((1.,2.) for _ in spec.document_ids))
        writer.finalize(provenance(spec,operator))
        snapshot=Snapshot(self.root/name);self.addCleanup(snapshot.close)
        return snapshot

    def pair(self):
        return self.capture("offline",self.offline_spec),self.capture("online",self.online_spec)

    def test_fake_full_prefill_capture_offset_bridge_and_gate_end_to_end(self):
        offline,online=self.pair()
        pair=pair_from_snapshots(offline,online,3,max_values=64)
        self.assertEqual(pair.document_start,2)
        self.assertEqual(pair.online_rows,pair.offline.rows)
        self.assertEqual(pair.data_origin,"synthetic")
        report=audit_snapshots(offline,online,3,self.policy,max_values=64,
            oracle_relative_tolerance=0.,oracle_absolute_tolerance=0.)
        self.assertEqual(report["summary"]["reused_rows"],12)
        self.assertEqual(report["summary"]["bad_reused_rows"],0)
        self.assertEqual(report["snapshot_sources"]["offline"]["seal"],offline.seal)
        self.assertEqual(report["snapshot_sources"]["online"]["seal"],online.seal)
        self.assertEqual(report["capture_sha256_declared"],digest(report["snapshot_sources"]))
        self.assertFalse(report["runtime_admission_created"])
        self.assertEqual(offline.read_rows(3,0,1),pair.offline.rows[:1])

    def test_combined_value_budget_checked_before_materializing_either_side(self):
        offline,online=self.pair()
        with mock.patch.object(offline,"as_cache",side_effect=AssertionError("must not load")), \
             mock.patch.object(online,"read_rows",side_effect=AssertionError("must not load")):
            with self.assertRaises(ValidationError):pair_from_snapshots(offline,online,3,max_values=63)
        for budget in (True,0,-1,1.5):
            with self.subTest(budget=budget),self.assertRaises(ValidationError):
                pair_from_snapshots(offline,online,3,max_values=budget)

    def test_reversed_kinds_and_same_request_id_rejected(self):
        offline,online=self.pair()
        with self.assertRaises(ValidationError):pair_from_snapshots(online,offline,3)
        same_id=self.capture("same-id",replace(self.online_spec,request_id=self.offline_spec.request_id))
        with self.assertRaises(ValidationError):pair_from_snapshots(offline,same_id,3)

    def test_wrong_document_and_model_config_rejected(self):
        offline=self.capture("offline",self.offline_spec)
        changes=[dict(input_ids=(99,98)+tuple(range(11,27))+(97,)),
                 dict(model_sha256="4"*64),dict(tokenizer_sha256="4"*64),
                 dict(execution_sha256="4"*64)]
        for index,change in enumerate(changes):
            online=self.capture(str(index),replace(self.online_spec,**change))
            with self.subTest(change=change),self.assertRaises(ValidationError):
                pair_from_snapshots(offline,online,3)

    def test_missing_or_invalid_layer_rejected(self):
        offline,online=self.pair()
        for layer in (5,True,-1,"3"):
            with self.subTest(layer=layer),self.assertRaises(ValidationError):
                pair_from_snapshots(offline,online,layer)
        only4=self.capture("only4",replace(self.online_spec,layers=(4,)))
        with self.assertRaises(ValidationError):pair_from_snapshots(offline,only4,3)

    def test_mixed_reference_operators_and_native_origins_rejected(self):
        offline=self.manual("offline",self.offline_spec,"python-float64-rmsnorm-reference")
        for index,operator in enumerate(("torch-cpu-float32-rmsnorm-reference",
                                         "deepinfer-native-fused-rmsnorm-only")):
            online=self.manual(str(index),self.online_spec,operator)
            with self.subTest(operator=operator),self.assertRaises(ValidationError):
                pair_from_snapshots(offline,online,3)

    def test_native_labels_are_only_declarations_not_proof_of_model_execution(self):
        # CPU-written manifests intentionally demonstrate that a source flag is
        # not authentication. No NativeCaptureOps or GPU code is executed.
        operator="deepinfer-native-fused-rmsnorm-only"
        offline=self.manual("offline",self.offline_spec,operator)
        online=self.manual("online",self.online_spec,operator)
        pair=pair_from_snapshots(offline,online,3)
        source=snapshot_pair_provenance(offline,online,3)
        self.assertEqual(pair.data_origin,"model_capture")
        self.assertFalse(source["origin_authenticated"])
        self.assertFalse(source["task_accuracy_certified"])

    def test_dependency_injection_source_cannot_become_model_capture(self):
        operator="deepinfer-norm-dependency-fixture"
        offline=self.manual("offline",self.offline_spec,operator)
        online=self.manual("online",self.online_spec,operator)
        pair=pair_from_snapshots(offline,online,3)
        self.assertEqual(pair.data_origin,"synthetic")
        source=snapshot_pair_provenance(offline,online,3)
        self.assertFalse(source["native_operator_declared"])
        self.assertFalse(source["accelerator_capture_declared"])

    def test_fake_driver_enumerates_full_causal_window_beyond_128_positions(self):
        offline_spec=replace(self.offline_spec,input_ids=tuple(range(10,270)),
                             document_end=260,layers=(3,))
        online_spec=replace(self.online_spec,input_ids=(99,98)+offline_spec.input_ids+(97,),
                            document_end=262,layers=(3,))
        offline=self.capture("offline",offline_spec)
        online=self.capture("online",online_spec)
        pair=pair_from_snapshots(offline,online,3,max_values=1040)
        self.assertEqual(pair.online_rows,pair.offline.rows)
        self.assertEqual(len(pair.online_rows),260)

    def test_closed_snapshot_rejected(self):
        offline,online=self.pair();online.close()
        with self.assertRaises(ValidationError):pair_from_snapshots(offline,online,3)

    def test_payload_corruption_after_open_rejected(self):
        offline,online=self.pair()
        path=online.directory/"layer3.f64le"
        data=path.read_bytes();path.write_bytes(bytes([data[0]^1])+data[1:])
        with self.assertRaises(ValidationError):pair_from_snapshots(offline,online,3)

    def test_changed_in_memory_seal_and_spec_rejected(self):
        offline,online=self.pair()
        online._manifest["provenance"]["operator"]="unknown"
        with self.assertRaises(ValidationError):pair_from_snapshots(offline,online,3)
        online._manifest["seal"]=digest({k:v for k,v in online._manifest.items() if k!="seal"})
        with self.assertRaises(ValidationError):pair_from_snapshots(offline,online,3)

    def test_provenance_detached_and_binds_both_seals_and_layer(self):
        offline,online=self.pair()
        source=snapshot_pair_provenance(offline,online,3)
        first=pair_from_snapshots(offline,online,3)
        source["offline"]["seal"]="f"*64
        self.assertEqual(first.capture_sha256,digest(snapshot_pair_provenance(offline,online,3)))
        second=pair_from_snapshots(offline,online,4)
        self.assertNotEqual(first.capture_sha256,second.capture_sha256)


if __name__=="__main__":
    unittest.main()
