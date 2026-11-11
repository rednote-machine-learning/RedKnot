"""CPU-only adversarial reader fixtures for the independent repaired GPU pilot."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.core import ValidationError, digest
from gpu_pilot import read_early_mhc_results as reader
from gpu_pilot.early_mhc_protocol import SPARSE_LAYERS, early_mhc_protocol


def seal(body):
    body = {key: value for key, value in body.items() if key != "sha256"}
    return dict(body, sha256=digest(body))


def preflight():
    docs = [dict(input_start=3+i*32768, input_end=3+(i+1)*32768,
                 token_ids_sha256=digest([i])) for i in range(8)]
    rows = [dict(index=index, dataset="hotpotqa" if index % 2 == 0 else "2wikimqa",
        input_tokens=262147, input_ids_sha256=digest([index]), documents=deepcopy(docs),
        sample_id=f"sample{index}", question="fixture question", references=["answer"])
        for index in reader.CASE_INDICES]
    return seal(dict(schema=reader.RUN_SCHEMA, protocol=early_mhc_protocol(),
        evaluation_case_indices=list(reader.CASE_INDICES), audit_cases=rows,
        manifest_sha256="a"*64, identity=dict(execution_configuration_sha256="b"*64,
        checkpoint_index_sha256="c"*64, tokenizer_json_sha256="d"*64)))


def phase_fixture(pf, arm):
    row = pf["audit_cases"][0]
    name = f"case2.{arm}_partial_smoke"
    rid = f"early-mhc:2:{name}:{row['sample_id']}"
    selection, events = [], []
    full, selected = 0, 0
    for start in range(0, row["input_tokens"], 8192):
        end = min(start+8192, row["input_tokens"])
        scores = [0.]*(end-start)
        ids, mandatory, optional = reader.mask_reference(scores, start, end, reader.documents(row), reader.FRACTIONS[arm])
        plan_hash = digest([arm, start, end])
        selection.append(dict(start=start, end=end, selected_local_rows=ids,
            selected_rows=len(ids), full_rows=end-start, mandatory_rows=mandatory,
            optional_rows=optional, selected_rows_sha256=digest(ids),
            plan_sha256=plan_hash, mean_early_relative_l2=scores))
        for layer in SPARSE_LAYERS:
            events.append(dict(layer=layer, request_id=rid, phase=name, chunk_start=start,
                chunk_end=end, full_rows=end-start, selected_rows=len(ids), plan_sha256=plan_hash,
                inactive_sublayer_output="zero", full_row_native_mhc=True, cached_hidden_or_sublayer_output=False))
        full += (end-start)*len(SPARSE_LAYERS)
        selected += len(ids)*len(SPARSE_LAYERS)
    control = dict(schema="early-six-sensitive-mhc-controller-v1", mode="sensitive", phase=name,
        input_ids_sha256=row["input_ids_sha256"], full_input_tokens=row["input_tokens"],
        sparse_layers=list(SPARSE_LAYERS), first_six_full=True, actual_full_row_native_mhc=True,
        formal_admission_unchanged=True, not_model_flops_fraction=True, selection=selection,
        actual_swa_reuse_fraction_in_eleven_layers=1-selected/full,
        adapter=dict(schema="kv-cache-engram-early-mhc-adapter-v1", native_execution=True,
            both_full_row_native_mhc_norm_calls_retained=True,
            per_layer_zero_inactive_attention_and_ffn=True, interval_frozen_carrier=False,
            events=events, counts=dict(full_rows=full, selected_rows=selected, masked_layer_calls=len(events),
                inactive_attention_zero_rows=full-selected, inactive_ffn_zero_rows=full-selected)))
    return seal(dict(schema=reader.PHASE_SCHEMA, index=2, dataset=row["dataset"], name=name,
        arm=arm, mode="sensitive", fraction=reader.FRACTIONS[arm], repeat=None,
        partial_smoke=True,
        timing_valid=False, quality_eligible=False, input_ids_sha256=row["input_ids_sha256"],
        manifest_sha256=pf["manifest_sha256"], early_mhc_protocol_sha256=pf["protocol"]["sha256"],
        execution_configuration_sha256=pf["identity"]["execution_configuration_sha256"],
        online_early6_scout_and_plan_included=True, offline_build_included=False,
        native_global_kv_and_index_preserved=True, controller=control,
        record=dict(success=True, rank=0, input_ids_sha256=row["input_ids_sha256"],
            input_tokens=row["input_tokens"], phase=name, sample_id=row["sample_id"],
            dataset=row["dataset"], request_id=rid, output_ids=[42], output_ids_sha256=digest([42]),
            natural_EOS=False, truncated=False)))


class SmokeReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = preflight()
        cls.phases = {arm: phase_fixture(cls.pf, arm) for arm in ("early20", "early50")}
        cls.qual = seal(dict(schema="test-qualification", passed=True))

    def run_smoke(self, phases=None, closure_changes=None, omit=None):
        phases = self.phases if phases is None else phases
        row = self.pf["audit_cases"][0]
        body = dict(schema="early-mhc-partial-smoke-v2", passed=True, index=2,
            phase_sha256={arm: phase["sha256"] for arm, phase in phases.items()},
            input_ids_sha256=row["input_ids_sha256"], input_tokens=row["input_tokens"],
            qualification_sha256=self.qual["sha256"], quality_eligible=False, timing_valid=False,
            accuracy_guarantee=False, full_input_original_eight_documents=True,
            dynamic_early_score_and_partial_q_ffn_mhc_exercised=True, output_tokens_per_arm=1)
        body.update(closure_changes or {})
        closure = seal(body)
        values = {"partial_smoke.json": closure}
        values.update({phase["name"]+".json": phase for phase in phases.values()})
        if omit is not None:
            values.pop(omit)
        def read_fixture(path, evidence):
            if path.name not in values:
                raise ValidationError("missing raw fixture")
            return reader.sealed(values[path.name])
        with patch.object(reader, "read", side_effect=read_fixture):
            return reader.partial_smoke(Path("/never-read"), self.pf, self.qual, {})

    def test_two_actual_partial_arms_cover_every_layer_and_chunk(self):
        _, summary = self.run_smoke()
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["numerical_identity_claim"])
        for arm in ("early20", "early50"):
            count = summary["actual_rows"][arm]
            self.assertEqual(count["full_rows"], 262147*11)
            self.assertEqual(count["actual_partial_chunks"], 32)
            self.assertGreater(count["bypass_rows"], 0)

    def test_smoke_missing_phase_or_wrong_qualification_rejected(self):
        with self.assertRaises(ValidationError):
            self.run_smoke(omit="case2.early50_partial_smoke.json")
        with self.assertRaises(ValidationError):
            self.run_smoke(closure_changes=dict(qualification_sha256="f"*64))

    def test_smoke_foreign_input_or_phase_hash_rejected(self):
        for changes in (dict(input_ids_sha256="f"*64), dict(input_tokens=32768),
                        dict(quality_eligible=True), dict(full_input_original_eight_documents=False),
                        dict(phase_sha256={"early20":"f"*64,"early50":"e"*64})):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.run_smoke(closure_changes=changes)

    def test_smoke_cannot_be_quality_or_32token_timing(self):
        for field in ("quality_eligible", "timing_valid"):
            phases = deepcopy(self.phases)
            phases["early20"][field] = True
            phases["early20"] = seal(phases["early20"])
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.run_smoke(phases)
        phases = deepcopy(self.phases)
        phases["early20"]["partial_smoke"] = False
        phases["early20"] = seal(phases["early20"])
        with self.assertRaises(ValidationError):
            self.run_smoke(phases)
        phases = deepcopy(self.phases)
        phases["early20"]["record"].update(output_ids=[42]*32, output_ids_sha256=digest([42]*32))
        phases["early20"] = seal(phases["early20"])
        with self.assertRaises(ValidationError):
            self.run_smoke(phases)

    def test_missing_duplicate_or_omitted_later_chunk_events_rejected(self):
        for mutation in ("missing", "duplicate", "later_chunk"):
            phases = deepcopy(self.phases)
            events = phases["early20"]["controller"]["adapter"]["events"]
            if mutation == "missing":
                events.pop()
            elif mutation == "duplicate":
                events[-1] = deepcopy(events[0])
            else:
                phases["early20"]["controller"]["selection"].pop()
            phases["early20"] = seal(phases["early20"])
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                self.run_smoke(phases)

    def test_all_selected_smoke_cannot_pass_as_partial(self):
        phases = deepcopy(self.phases)
        item = phases["early20"]["controller"]["selection"][0]
        item["selected_local_rows"] = list(range(item["full_rows"]))
        item["selected_rows"] = item["full_rows"]
        item["selected_rows_sha256"] = digest(item["selected_local_rows"])
        phases["early20"] = seal(phases["early20"])
        with self.assertRaises(ValidationError):
            self.run_smoke(phases)

    def test_nonfinite_score_and_changed_scope_rejected(self):
        # NaN is rejected by canonical sealing as well as by the numeric mask oracle.
        with self.assertRaises(ValidationError):
            reader.mask_reference([float("nan")]*8192,0,8192,reader.documents(self.pf["audit_cases"][0]),.2)
        phases = deepcopy(self.phases)
        phases["early20"]["controller"]["adapter"]["events"][0]["request_id"] = "foreign"
        phases["early20"] = seal(phases["early20"])
        with self.assertRaises(ValidationError):
            self.run_smoke(phases)

    def test_nested_masks_checked_across_the_two_smoke_arms(self):
        phases = deepcopy(self.phases)
        phase = phases["early50"]
        item = phase["controller"]["selection"][0]
        scores = list(map(float, range(8192)))
        ids, _, _ = reader.mask_reference(scores,0,8192,reader.documents(self.pf["audit_cases"][0]),.5)
        item.update(mean_early_relative_l2=scores, selected_local_rows=ids, selected_rows_sha256=digest(ids))
        phases["early50"] = seal(phase)
        with self.assertRaisesRegex(ValidationError, "nested"):
            self.run_smoke(phases)


class SubsetAuditTests(unittest.TestCase):
    def make_directory(self, directory, indices=(), final=False):
        path = Path(directory)
        pf = preflight()
        (path/"preflight.json").write_text(json.dumps(pf))
        for name in ("qualification.json", "partial_smoke.json"):
            (path/name).write_text("unused mocked closure")
        for index in indices:
            (path/f"case{index}.paired.json").write_text("unused mocked case")
        if final:
            (path/"run.json").write_text("must not read a final run during subset audit")
        return path, pf

    def test_requested_case_only_does_not_reaudit_old_banks_or_claim_final(self):
        with tempfile.TemporaryDirectory() as directory:
            path, pf = self.make_directory(directory, indices=(2,3,4,5), final=True)
            qual, smoke = seal(dict(qualified=True)), seal(dict(partial=True))
            def one_case(directory, row, preflight, q, s, evidence):
                self.assertEqual(row["index"],4)
                return dict(index=4), dict(index=4,dataset="hotpotqa")
            with patch.object(reader,"qualification",return_value=qual), \
                 patch.object(reader,"partial_smoke",return_value=(smoke,dict(passed=True))), \
                 patch.object(reader,"audit_case",side_effect=one_case) as audit_case:
                result = reader.audit(path,case_index=4)
            self.assertEqual(audit_case.call_count,1)
            self.assertEqual(result["status"],"subset_complete")
            self.assertEqual(result["requested_case_indices"],[4])
            self.assertEqual(result["planned_cases"],4)
            self.assertFalse(result["overall_run_complete"])
            self.assertFalse(result["final_run_audited"])
            self.assertNotIn("run.json",result["evidence"])

    def test_formal_case_before_smoke_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,pf = self.make_directory(directory,indices=(2,))
            (path/"partial_smoke.json").unlink()
            with patch.object(reader,"qualification",return_value=seal(dict(qualified=True))):
                with self.assertRaisesRegex(ValidationError,"before qualified partial smoke"):
                    reader.audit(path,case_index=2)

    def test_missing_qualification_with_smoke_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_ = self.make_directory(directory)
            (path/"qualification.json").unlink()
            with self.assertRaisesRegex(ValidationError,"without strict"):
                reader.audit(path)

    def test_preflight_requires_all_original_four_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path,pf = self.make_directory(directory)
            pf["evaluation_case_indices"] = [2,3]
            pf["audit_cases"] = pf["audit_cases"][:2]
            (path/"preflight.json").write_text(json.dumps(seal(pf)))
            with self.assertRaises(ValidationError):
                reader.audit(path)

    def test_empty_subset_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_ = self.make_directory(directory,indices=(2,3))
            with patch.object(reader,"qualification",return_value=seal(dict(qualified=True))), \
                 patch.object(reader,"partial_smoke",return_value=(seal(dict(partial=True)),dict(passed=True))):
                result=reader.audit(path,case_index=5)
            self.assertEqual(result["status"],"incomplete")
            self.assertEqual(result["complete_cases"],0)
            self.assertFalse(result["overall_run_complete"])

    def test_full_final_audit_requires_all_four_and_links_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            path,pf = self.make_directory(directory,indices=(2,3,4,5))
            qual,smoke = seal(dict(qualified=True)),seal(dict(partial=True))
            cases=[dict(index=i) for i in reader.CASE_INDICES]
            run=seal(dict(schema=reader.RUN_SCHEMA,status="complete",completed_case_indices=list(reader.CASE_INDICES),
                identity=pf["identity"],manifest_sha256=pf["manifest_sha256"],qualification=qual,
                partial_smoke_sha256=smoke["sha256"],cases=cases))
            (path/"run.json").write_text(json.dumps(run))
            def one_case(directory,row,preflight,q,s,evidence):
                return dict(index=row["index"]),dict(index=row["index"],dataset=row["dataset"])
            with patch.object(reader,"qualification",return_value=qual), \
                 patch.object(reader,"partial_smoke",return_value=(smoke,dict(passed=True))), \
                 patch.object(reader,"audit_case",side_effect=one_case) as audit_case:
                result=reader.audit(path)
                self.assertEqual(audit_case.call_count,4)
                self.assertEqual(result["status"],"complete")
                self.assertTrue(result["final_run_audited"])
                run["partial_smoke_sha256"]="f"*64
                (path/"run.json").write_text(json.dumps(seal(run)))
                with self.assertRaises(ValidationError):
                    reader.audit(path)

    def test_cli_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"audit.json"
            result=seal(dict(errors=[],status="subset_complete"))
            with patch.object(reader,"audit",return_value=result):
                self.assertEqual(reader.main(["--result-dir",directory,"--case-index","4","--output",str(path)]),0)
                with self.assertRaises(FileExistsError):
                    reader.main(["--result-dir",directory,"--case-index","4","--output",str(path)])


if __name__ == "__main__":
    unittest.main()
