"""Synthetic CPU-only contracts, never real-model sensitivity evidence."""
from copy import deepcopy
import json
import unittest

from gpu_pilot import pilot_protocol as pp
from gpu_pilot import sensitivity_protocol as sp
from kv_cache_engram.core import ValidationError, digest


def cases():
    return [dict(case_id="case"+str(i), dataset=pp.DATASETS[i], split="calibration",
        input_sha256=digest([i, "input"]), source_document_ids=[digest([i, "source"])],
        documents=[dict(document_sha256=digest([i, j, "doc"]), tokens=32768) for j in range(8)])
        for i in range(2)]


def metric(error, head=None):
    value = dict(rows=1024, relative_l2_max=error, relative_l2_mean=error/2,
                 absolute_max=error/4, cosine_distance_max=min(error, 2.), worst_token_index=4)
    if head is not None:
        value["head"] = head
    return value


def observer(case, worst_layer=7, *, repairs=True):
    records = []
    span = dict(document_sha256=case["documents"][0]["document_sha256"],
                document_local_start=128, document_local_end=512, input_start=128, input_end=512,
                block_start=0, block_end=512)
    second = dict(span, document_local_start=512, document_local_end=1024, input_start=512, input_end=1024,
                  block_start=512, block_end=1024)
    for layer in sp.LAYERS:
        error = 1. if layer == worst_layer else .1
        record = dict(layer=layer, start=0, end=1024, operation="reuse_current_chunk",
            source_spans=[span, second], reused_rows=896, restored_rows=0,
            source_block_scores=[dict(input_start=128, input_end=512, relative_l2_max=1.),
                                 dict(input_start=512, input_end=1024, relative_l2_max=.2)],
            attention_update=metric(error), heads=[metric(error, head=h) for h in range(2)],
            same_clean_input=True, intervention_scope="current_chunk_swa_only",
            full_block_hidden_measured=False)
        records.append(record)
        if repairs:
            for label, operation, a, b, after in (
                ("high_error_512", "restore_block", 128, 512, .5),
                ("left_half_256", "restore_half", 128, 256, 1.2),
                ("right_half_256", "restore_half", 256, 512, .6)):
                restored = dict(span, document_local_start=a, document_local_end=b, input_start=a, input_end=b)
                records.append(dict(record, operation=operation, restoration_label=label,
                    restored_spans=[restored], restored_rows=b-a, reused_rows=896-(b-a),
                    attention_update=metric(after), heads=[metric(after, head=h) for h in range(2)]))
    return pp.seal_record(dict(schema=sp.OBSERVER_SCHEMA, complete=True, request_id="r:"+case["case_id"],
        phase="diag:"+case["case_id"], input_sha256=case["input_sha256"],
        input_tokens=1024, native_execution=True, operator_backend="deepinfer-sm90-native-bfloat16-kv-only-v1",
        selected_layers=list(sp.LAYERS), head_count=2,
        coverage={str(layer): 1024 for layer in sp.LAYERS}, records=records,
        diagnostic_only=True, timing_eligible=False, downstream_native_unchanged=True,
        source_token_vs_query_token_distinguished=True, full_block_hidden_measured=False,
        preserves_native_cached_prefix_and_global_kv=True, diagnostics_do_not_authorize_runtime_reuse=True))


def reseal(value):
    return pp.seal_record({key: item for key, item in value.items() if key != "sha256"})


def wrap(value, case, **overrides):
    args = dict(input_tokens=1024, source_phase_sha256=digest([case["case_id"], "phase"]),
                execution_sha256="e"*64, manifest_sha256="f"*64)
    args.update(overrides)
    return sp.wrap_observer(value, case, **args)


class SensitivityProtocolTests(unittest.TestCase):
    def validate(self, value):
        return sp.validate_observer(reseal(value), input_sha256=cases()[0]["input_sha256"], input_tokens=1024)

    def test_protocol_is_json_roundtrip_stable_and_not_accuracy_certificate(self):
        protocol = sp.sensitivity_protocol()
        self.assertEqual(protocol, json.loads(json.dumps(protocol)))
        pp._sealed(json.loads(json.dumps(protocol)))
        self.assertFalse(protocol["em_f1_guarantee"])
        self.assertTrue(protocol["head_diagnostic_only"])
        self.assertEqual(protocol["early_full_layers"], list(range(6)))

    def test_all_tokens_heads_and_layers_counted(self):
        summary = self.validate(observer(cases()[0]))
        self.assertEqual(set(summary["layers"]), {str(layer) for layer in sp.LAYERS})
        self.assertEqual(summary["layers"]["7"]["rows"], 1024)
        self.assertEqual(summary["layers"]["7"]["affected_query_token_index"], 4)
        self.assertFalse(summary["head_control_enabled"])
        self.assertFalse(summary["full_block_hidden_measured"])

    def test_both_halves_and_negative_recovery_are_retained(self):
        summary = self.validate(observer(cases()[0], repairs=True))
        restored = {r["restoration_label"]: r for r in summary["restoration"] if r["layer"] == 7}
        self.assertEqual(len(restored), 3)
        self.assertLess(restored["left_half_256"]["signed_error_reduction"], 0)
        self.assertLess(restored["left_half_256"]["gap_recovery_fraction"], 0)

    def test_zero_denominator_is_not_forced_to_success(self):
        value = observer(cases()[0], repairs=True)
        for row in value["records"]:
            if row["layer"] == 7 and row["operation"] == "reuse_current_chunk":
                row["attention_update"] = metric(0.)
        summary = self.validate(reseal(value))
        self.assertTrue(all(r["gap_recovery_fraction"] is None for r in summary["restoration"] if r["layer"] == 7))

    def test_missing_or_duplicate_layer_chunk_fails(self):
        for duplicate in (False, True):
            value = observer(cases()[0])
            value["records"] = value["records"][:-1] if not duplicate else value["records"]+[value["records"][0]]
            with self.assertRaises(ValueError):
                self.validate(value)

    def test_malformed_head_and_nonfinite_error_fail_closed(self):
        for mode in ("head", "nonfinite", "outside", "mean"):
            value = observer(cases()[0])
            if mode == "head":
                value["records"][0]["heads"].pop()
            elif mode == "nonfinite":
                value["records"][0]["attention_update"]["relative_l2_max"] = float("nan")
            elif mode == "outside":
                value["records"][0]["attention_update"]["worst_token_index"] = 1024
            else:
                value["records"][0]["attention_update"]["relative_l2_mean"] = 100.
            with self.assertRaises(ValueError):
                self.validate(value)

    def test_unsafe_scope_claims_fail(self):
        for key, changed in (("timing_eligible", True), ("downstream_native_unchanged", False),
                             ("source_token_vs_query_token_distinguished", False), ("complete", False)):
            value = observer(cases()[0]); value[key] = changed
            with self.assertRaises(ValidationError):
                self.validate(value)
        value = observer(cases()[0]); value["records"][0]["same_clean_input"] = False
        with self.assertRaises(ValidationError):
            self.validate(value)

    def test_wrong_source_accounting_guard_and_restoration_identity_rejected(self):
        for mode in ("count", "guard", "foreign", "duplicate"):
            value = observer(cases()[0], repairs=True)
            row = next(r for r in value["records"] if r["operation"] == "restore_block")
            if mode == "count":
                row["restored_rows"] += 1
            elif mode == "guard":
                row["restored_spans"][0]["document_local_start"] = 0
            elif mode == "foreign":
                row["restored_spans"][0]["document_sha256"] = "a"*64
            else:
                value["records"].append(row)
            with self.assertRaises(ValidationError):
                self.validate(value)

    def test_freeze_requires_both_calibration_cases_and_keeps_numeric_gate(self):
        cal = cases()
        reports = [wrap(observer(case, worst), case) for case, worst in zip(cal, (7, 12))]
        frozen = sp.freeze_admission_plan(reports, cal, numerical_calibration_sha256="d"*64)
        self.assertEqual(frozen["sensitivity_full_layers"], [7, 12])
        self.assertEqual(frozen["mandatory_full_layers"], [0, 1, 2, 3, 4, 5, 7, 12])
        self.assertNotIn(7, frozen["eligible_reuse_layers"])
        self.assertTrue(frozen["original_numerical_gate_required"])
        self.assertFalse(frozen["source_blocks_transferred_to_heldout"])
        self.assertFalse(frozen["head_specific_kv_recompute"])
        self.assertEqual(tuple(frozen["eligible_reuse_layers"]), sp.validate_frozen_plan(
            json.loads(json.dumps(frozen)), reports, cal, numerical_calibration_sha256="d"*64))
        with self.assertRaises(ValidationError):
            sp.freeze_admission_plan(reports[:1], cal, numerical_calibration_sha256="d"*64)

    def test_selection_is_deterministic_and_early_layers_cannot_consume_extra_budget(self):
        cal = cases()
        reports = [wrap(observer(case, 3), case) for case in cal]
        frozen = sp.freeze_admission_plan(reports, cal, numerical_calibration_sha256="d"*64)
        self.assertEqual(frozen["sensitivity_full_layers"], [6])
        reverse = sp.freeze_admission_plan(reports[::-1], cal, numerical_calibration_sha256="d"*64)
        self.assertEqual(frozen, reverse)

    def test_mixed_manifest_execution_and_tampered_summary_fail(self):
        cal = cases()
        for mode in ("manifest", "execution", "summary"):
            reports = [wrap(observer(case), case) for case in cal]
            report = dict(reports[1]); report.pop("sha256")
            if mode == "manifest":
                report["manifest_sha256"] = "a"*64
            elif mode == "execution":
                report["execution_sha256"] = "a"*64
            else:
                report["summary"] = dict(report["summary"], head_control_enabled=True)
            reports[1] = pp.seal_record(report)
            with self.assertRaises(ValidationError):
                sp.freeze_admission_plan(reports, cal, numerical_calibration_sha256="d"*64)

    def test_heldout_observer_may_not_tune_admission(self):
        case = cases()[0]; case["split"] = "evaluation"
        with self.assertRaises(ValidationError):
            wrap(observer(case), case)

    def test_raw_phase_binding_is_rechecked(self):
        case = cases()[0]
        obs = observer(case)
        phase = dict(schema="kv-cache-engram-native-phase-v1", mode="shadow_full",
            split="calibration", timing_valid=False, input_ids_sha256=case["input_sha256"],
            dataset=case["dataset"], name=obs["phase"], manifest_sha256="f"*64,
            execution_configuration_sha256="e"*64, sensitivity=obs,
            record=dict(success=True, sample_id=case["case_id"], request_id=obs["request_id"],
                        input_ids_sha256=case["input_sha256"], input_tokens=1024))
        wrapped = sp.wrap_phase(pp.seal_record(phase), case, manifest_sha256="f"*64, execution_sha256="e"*64)
        self.assertEqual(wrapped["observer"], obs)
        for field, changed in (("timing_valid", True), ("mode", "gated"), ("split", "heldout")):
            bad = dict(phase, **{field: changed})
            with self.assertRaises(ValidationError):
                sp.wrap_phase(pp.seal_record(bad), case, manifest_sha256="f"*64, execution_sha256="e"*64)

    def test_incomplete_halves_and_unverifiable_source_selection_fail(self):
        for mode in ("missing_half", "wrong_scores", "missing_scores", "wrong_block"):
            value = observer(cases()[0])
            if mode == "missing_half":
                value["records"] = [r for r in value["records"]
                                    if not (r["layer"] == 7 and r.get("restoration_label") == "right_half_256")]
            elif mode == "wrong_scores":
                for row in value["records"]:
                    if row["layer"] == 7:
                        row["source_block_scores"][1]["relative_l2_max"] = 5.
            elif mode == "missing_scores":
                value["records"][0].pop("source_block_scores")
            else:
                value["records"][0]["source_spans"][0]["block_end"] = 1024
            with self.assertRaises(ValidationError):
                self.validate(value)

    def test_signed_gain_cannot_be_clipped(self):
        value = observer(cases()[0])
        row = next(r for r in value["records"] if r.get("restoration_label") == "left_half_256")
        row["error_reduction_relative_l2_max"] = 0.
        with self.assertRaises(ValidationError):
            self.validate(value)

    def test_tampered_nested_observer_seal_fails(self):
        value = observer(cases()[0])
        value["records"][0]["attention_update"]["absolute_max"] = 7.
        with self.assertRaises(ValidationError):
            sp.validate_observer(value, input_sha256=cases()[0]["input_sha256"], input_tokens=1024)

    def test_foreign_source_document_cannot_be_wrapped(self):
        case = cases()[0]
        foreign = observer(cases()[1])
        foreign["input_sha256"] = case["input_sha256"]
        with self.assertRaises(ValidationError):
            wrap(reseal(foreign), case)


if __name__ == "__main__":
    unittest.main()
