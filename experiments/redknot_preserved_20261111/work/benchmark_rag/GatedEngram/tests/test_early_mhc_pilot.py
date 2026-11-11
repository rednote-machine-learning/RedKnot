"""CPU-only early-mHC protocol/timing/lifecycle regression fixtures."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from gpu_pilot import run_early_mhc_pilot as runner
from gpu_pilot import launch_early_mhc_pilot as launcher


def seal(value):
    return dict(value, sha256=runner._digest(value))


def phase(index=2, arm="recomputed", repeat=0, timing=True, quality=False, **updates):
    ids = list(range(32 if timing else 1))
    record = dict(success=True, rank=0, input_ids_sha256="i"*64, output_ids=ids,
        output_ids_sha256=runner._digest(ids), online_ttft_s=80. if arm == "recomputed" else 60.,
        online_elapsed_s=100., online_prepare_to_submit_s=.1, natural_EOS=quality,
        finish_reason="stop", truncated=False, answer_em=1., answer_token_f1=1.)
    body = dict(schema=runner.PHASE_SCHEMA, name="fixture", index=index, arm=arm, repeat=repeat,
        timing_valid=timing, quality_eligible=quality, input_ids_sha256="i"*64,
        execution_configuration_sha256="e"*64, manifest_sha256="m"*64,
        early_mhc_protocol_sha256="p"*64, record=record, online_global_window_s=101.,
        mode="native", controller={})
    body.update(updates)
    return seal(body)


def changed(value, **updates):
    body = {k: copy.deepcopy(v) for k, v in value.items() if k != "sha256"}
    body.update(updates)
    return seal(body)


def smoke_fixture():
    # Small CPU report fixture only: native input geometry is separately
    # enforced by frozen input preflight and the real controller.
    row = dict(index=2, documents=[{} for _ in range(8)], input_tokens=8193, input_ids_sha256="i"*64)
    qualification = seal(dict(passed=True, identity_check_count=80))
    phases = []
    for arm in runner.ARMS[1:]:
        fraction = runner.FRACTIONS[arm]
        masks, events = [], []
        for start, end in ((0, 8192), (8192, 8193)):
            full = end-start
            mandatory, optional = min(2, full), max(0, full-2)
            selected = mandatory + math.ceil(fraction*optional)
            ids = list(range(selected))
            mask = dict(start=start, end=end, selected_local_rows=ids, full_rows=full,
                selected_rows=selected, mandatory_rows=mandatory, optional_rows=optional,
                selected_rows_sha256=runner._digest(ids), plan_sha256="p"*64,
                mean_early_relative_l2=[.1]*full)
            masks.append(mask)
            for layer in runner.SPARSE_LAYERS:
                events.append(dict(layer=layer, chunk_start=start, chunk_end=end, full_rows=full,
                    selected_rows=selected, plan_sha256="p"*64, full_row_native_mhc=True,
                    inactive_sublayer_output="zero", cached_hidden_or_sublayer_output=False))
        full = row["input_tokens"]*len(runner.SPARSE_LAYERS)
        selected = sum(m["selected_rows"] for m in masks)*len(runner.SPARSE_LAYERS)
        controller = dict(mode="sensitive", input_ids_sha256="i"*64, full_input_tokens=row["input_tokens"],
            first_six_full=True, actual_full_row_native_mhc=True, sparse_layers=list(runner.SPARSE_LAYERS),
            selection=masks, adapter=dict(native_execution=True, both_full_row_native_mhc_norm_calls_retained=True,
                events=events, counts=dict(masked_layer_calls=len(events), full_rows=full, selected_rows=selected,
                    inactive_attention_zero_rows=full-selected, inactive_ffn_zero_rows=full-selected)))
        phases.append(phase(index=2, arm=arm, timing=False, mode="sensitive", fraction=fraction,
                            partial_smoke=True, controller=controller))
    return row, qualification, phases


class RunnerTests(unittest.TestCase):
    def test_source_freeze_includes_exact_external_dependencies(self):
        import hashlib
        from gpu_pilot import early_mhc_protocol
        with tempfile.TemporaryDirectory(prefix="early-dependency-test-") as folder:
            root = Path(folder)
            payload = root/"fixed.py"
            payload.touch()
            expected = hashlib.sha256(b"").hexdigest()
            with patch.object(early_mhc_protocol, "DEPENDENCIES", {"fixed.py": expected}), \
                 patch.object(runner.support, "source_hashes", return_value={"stage.py": "s"*64}):
                result = runner.source_hashes(root)
                self.assertEqual(result["fixed.py"], expected)
            with patch.object(early_mhc_protocol, "DEPENDENCIES", {"fixed.py": "a"*64}), \
                 patch.object(runner.support, "source_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "dependency changed"):
                    runner.source_hashes(root)

    def test_import_does_not_initialize_accelerator_runtime(self):
        command = ("import sys; from gpu_pilot import run_early_mhc_pilot, launch_early_mhc_pilot; "
                   "assert 'torch' not in sys.modules; "
                   "assert not any(n == 'deepinfer' or n.startswith('deepinfer.') for n in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", command],
            cwd=Path(runner.__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_rotating_schedule_three_rounds_each_arm(self):
        for index in (2, 3, 4, 5):
            schedule = runner.timing_schedule(index)
            self.assertEqual(len(schedule), 9)
            for repeat in range(3):
                self.assertEqual([arm for r, arm in schedule if r == repeat],
                                 [runner.ARMS[(index+repeat+s)%3] for s in range(3)])
            for arm in runner.ARMS:
                self.assertEqual(sum(a == arm for _, a in schedule), 3)
        with self.assertRaises(ValueError): runner.timing_schedule(6)

    def test_qualification_prefix_has_new_bound_hash_and_no_quality_gold(self):
        original = dict(index=0, dataset="hotpotqa", input_ids=list(range(20000)), references=["answer"])
        result = runner.qualification_row(dict(warmup=[original]))
        self.assertEqual(result["input_tokens"], 16384)
        self.assertEqual(result["input_ids"], original["input_ids"][:16384])
        self.assertEqual(result["input_ids_sha256"], runner._digest(result["input_ids"]))
        self.assertIsNone(result["documents"])
        self.assertNotIn("answer", result["references"])
        self.assertEqual(len(original["input_ids"]), 20000)

    def test_evaluation_contains_all_four_registered_cases(self):
        rows = [dict(index=i, dataset="hotpotqa" if i % 2 == 0 else "2wikimqa") for i in range(2, 6)]
        self.assertEqual([r["index"] for r in runner.selected_rows(dict(cohort=rows))], [2, 3, 4, 5])
        bad = copy.deepcopy(rows); bad[1]["dataset"] = "hotpotqa"
        with self.assertRaises(ValueError): runner.selected_rows(dict(cohort=bad))
        with self.assertRaises(ValueError): runner.selected_rows(dict(cohort=rows[:2]))

    def test_partial_smoke_requires_both_real_partial_paths_and_is_not_quality_or_timing(self):
        row, qualification, phases = smoke_fixture()
        result = runner.verify_partial_smoke(phases, row, qualification)
        self.assertTrue(result["passed"])
        self.assertEqual(result["schema"], "early-mhc-partial-smoke-v2")
        self.assertFalse(result["quality_eligible"])
        self.assertFalse(result["timing_valid"])
        self.assertEqual(result["phase_sha256"], {p["arm"]: p["sha256"] for p in phases})
        for bad in (phases[:1], list(reversed(phases)),
                    [changed(phases[0], timing_valid=True)]+phases[1:],
                    [changed(phases[0], input_ids_sha256="j"*64)]+phases[1:],
                    [changed(phases[0], mode="all_selected")]+phases[1:],
                    [changed(phases[0], partial_smoke=False)]+phases[1:]):
            with self.assertRaises(ValueError): runner.verify_partial_smoke(bad, row, qualification)
        with self.assertRaises(ValueError): runner.verify_partial_smoke(phases, row, dict(qualification, passed=False))

    def test_partial_smoke_rejects_missing_layer_chunk_or_ffn_or_dynamic_scores(self):
        row, qualification, phases = smoke_fixture()
        for edit in (lambda c: c["adapter"]["events"].pop(),
                     lambda c: c["adapter"]["counts"].update(inactive_ffn_zero_rows=0),
                     lambda c: c["adapter"].update(native_execution=False),
                     lambda c: c["selection"][0].pop("mean_early_relative_l2"),
                     lambda c: c["selection"][0].update(mean_early_relative_l2=[.1]),
                     lambda c: c["adapter"]["events"][0].update(full_row_native_mhc=False)):
            report = copy.deepcopy(phases[0]["controller"])
            edit(report)
            with self.assertRaises(ValueError):
                runner.verify_partial_smoke([changed(phases[0], controller=report)]+phases[1:], row, qualification)

    def test_partial_smoke_cannot_substitute_all_selected_or_extra_output(self):
        row, qualification, phases = smoke_fixture()
        report = copy.deepcopy(phases[0]["controller"])
        for mask in report["selection"]:
            ids = list(range(mask["full_rows"]))
            mask.update(selected_local_rows=ids, selected_rows=len(ids), mandatory_rows=len(ids), optional_rows=0,
                        selected_rows_sha256=runner._digest(ids))
        with self.assertRaisesRegex(ValueError, "genuinely partial"):
            runner.verify_partial_smoke([changed(phases[0], controller=report)]+phases[1:], row, qualification)
        record = dict(phases[0]["record"], output_ids=[0, 1], output_ids_sha256=runner._digest([0, 1]))
        with self.assertRaises(ValueError):
            runner.verify_partial_smoke([changed(phases[0], record=record)]+phases[1:], row, qualification)

    def test_online_prepare_clock_covers_precomputed_index_reset(self):
        code = Path(runner.__file__).read_text()
        phase_code = code[code.index("        def phase("):code.index("        check_row = qualification_row")]
        self.assertLess(phase_code.index("began = time.perf_counter()"), phase_code.index("controller.reset("))
        self.assertLess(phase_code.index("controller.reset("), phase_code.index("base.run_phase("))
        self.assertIn("with allow_cuda_sync():\n                    controller.reset", phase_code)
        self.assertIn("index_device=index_device", phase_code)

    def test_timing_summary_includes_actual_slowdowns(self):
        phases = [phase(arm=a, repeat=r) for r, a in runner.timing_schedule(2)]
        report = runner.summarize_timing(phases, 2)
        self.assertAlmostEqual(report["arms"]["early20"]["ttft_speedup"], 4/3)
        self.assertTrue(report["online_early6_scout_and_plan_included"])
        changed_phases = []
        for p in phases:
            if p["arm"] == "early50":
                rec = dict(p["record"], online_ttft_s=90.)
                p = changed(p, record=rec)
            changed_phases.append(p)
        report = runner.summarize_timing(changed_phases, 2)
        self.assertLess(report["arms"]["early50"]["ttft_reduction_percent"], 0)

    def test_incomplete_reordered_or_unpaired_timings_rejected(self):
        phases = [phase(arm=a, repeat=r) for r, a in runner.timing_schedule(2)]
        for bad in (phases[:-1], list(reversed(phases)),
                    [changed(phases[0], execution_configuration_sha256="x"*64)]+phases[1:]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                runner.summarize_timing(bad, 2)

    def test_formal_length_or_clock_order_invalid_rejected(self):
        p = phase()
        for rec in (dict(p["record"], online_ttft_s=0), dict(p["record"], online_ttft_s=102),
                    dict(p["record"], online_prepare_to_submit_s=float("nan")),
                    dict(p["record"], output_ids=[1], output_ids_sha256=runner._digest([1]))):
            if not all(not isinstance(v, float) or v == v for v in rec.values()):
                with self.assertRaises(ValueError): changed(p, record=rec)
            else:
                with self.assertRaises(ValueError): runner.verify_phase(changed(p, record=rec), index=2,
                    arm="recomputed", timing=True, quality=False)

    def test_tampered_seal_rejected(self):
        p = phase(); p["record"]["online_ttft_s"] = 1.
        with self.assertRaisesRegex(ValueError, "seal"):
            runner.verify_phase(p, index=2, arm="recomputed", timing=True, quality=False)

    def test_quality_length_truncation_is_not_deleted(self):
        p = phase(timing=False, quality=True)
        rec = dict(p["record"], output_ids=list(range(64)), output_ids_sha256=runner._digest(list(range(64))),
                   finish_reason="length", truncated=True, answer_em=0., answer_token_f1=0.)
        result = runner.verify_phase(changed(p, record=rec), index=2, arm="recomputed", timing=False, quality=True)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["answer_em"], 0)
        with self.assertRaises(ValueError): runner.verify_phase(changed(p, record=dict(rec, truncated=False)),
            index=2, arm="recomputed", timing=False, quality=True)

    def test_new_qualification_requires_exact_checks_not_output_only(self):
        reference = phase(index=0, arm="qualification", timing=False, mode="qualified_native",
                          controller=dict(identity_check_count=80, identity_fingerprints_sha256="f"*64))
        candidate = phase(index=0, arm="qualification", timing=False, mode="all_selected",
                          controller=dict(identity_exact=True, identity_check_count=80, identity_fingerprints_sha256="f"*64))
        self.assertTrue(runner.qualification_passed(reference, candidate))
        for report in ({}, dict(identity_exact=False, identity_check_count=80),
                       dict(identity_exact=True, identity_check_count=0)):
            with self.assertRaises(ValueError): runner.qualification_passed(reference, changed(candidate, controller=report))
        with self.assertRaises(ValueError): runner.qualification_passed(reference,
            changed(candidate, early_mhc_protocol_sha256="x"*64))
        with self.assertRaises(ValueError): runner.qualification_passed(reference,
            changed(candidate, controller=dict(candidate["controller"], identity_fingerprints_sha256="d"*64)))

    def test_all_selected_qualification_does_not_pass_sensitive_fraction(self):
        code = Path(runner.__file__).read_text()
        self.assertIn('phase(check_row, "qualification.all_selected", "all_selected", arm="qualification")', code)
        self.assertNotIn('arm="qualification", fraction=1.', code)

    def test_layer_denominators_are_seventeen_capture_eleven_sparse(self):
        self.assertEqual(len(runner.CAPTURE_LAYERS), 17)
        self.assertEqual(len(runner.SPARSE_LAYERS), 11)
        self.assertEqual(set(range(6)) & set(runner.SPARSE_LAYERS), set())
        self.assertTrue(set(runner.SPARSE_LAYERS) <= set(runner.CAPTURE_LAYERS))


class LauncherTests(unittest.TestCase):
    def proof(self, sources=None):
        names = [f"test_early_mhc_pilot.Case.test_{i}" for i in range(10)]
        return dict(successful=True, cuda_initialized=False, failures=0, errors=0, skipped=0,
                    source_sha256=sources or {"runner.py": "a"*64}, tests=10, test_ids=names)

    def test_cpu_proof_requires_actual_new_tests_zero_skip_and_frozen_source(self):
        plan = dict(cpu_proof_path="not-read", stage_sources={"runner.py": "a"*64})
        good = self.proof()
        with patch.object(launcher.lifecycle, "read_sealed", return_value=good):
            launcher.verify_cpu_proof(plan)
        for update in (dict(skipped=1), dict(cuda_initialized=True), dict(failures=1),
                       dict(source_sha256={}), dict(test_ids=["old.test"]*10), dict(tests=11)):
            with self.subTest(update=update), patch.object(launcher.lifecycle, "read_sealed", return_value=dict(good, **update)):
                with self.assertRaises(ValueError): launcher.verify_cpu_proof(plan)

    def test_terminal_runner_sha_must_match_new_runner(self):
        plan = dict(stage_sources={"gpu_pilot/run_early_mhc_pilot.py": "a"*64})
        value = dict(schema="redknot-glm53-gpu-lifecycle-v1", holder_restored=True, worker_exit=0,
            gpu_ids=list(range(8)), finished_unix=1., supervisor_sha256=launcher.lifecycle.SUPERVISOR_SHA,
            holder_helper_sha256=launcher.lifecycle.HELPER_SHA, child_environment_sha256=launcher.lifecycle.ENV_SHA,
            runner_sha256="a"*64)
        launcher.verify_terminal(value, plan)
        with self.assertRaises(ValueError): launcher.verify_terminal(dict(value, runner_sha256="b"*64), plan)
        with self.assertRaises(ValueError): launcher.verify_terminal(dict(value, cleanup_warnings=["lost child"]), plan)

    def test_post_popen_identity_error_still_waits_before_holder_audit(self):
        events = []
        with tempfile.TemporaryDirectory(prefix="early-lifecycle-test-") as folder:
            root = Path(folder); stage = root/"independent/GatedEngram"; stage.mkdir(parents=True)
            result = root/"results"; result.mkdir()
            paths = launcher.lifecycle.evidence_paths(root, launcher.NAME)
            plan = dict(root=str(root), stage_dir=str(stage), paths={k:str(v) for k,v in paths.items()},
                        sha256="p"*64, argv=["mock-supervisor"])
            class Process:
                pid = 99
                def wait(self): events.append("wait"); return 1
            def snapshot(_): events.append("holder"); return dict(sha256="h"*64)
            def write(path, value): events.append("write:"+path.name)
            with patch.multiple(launcher, ROOT=root, REMOTE_STAGE=stage, STAGE=stage), \
                 patch.object(launcher, "verify_plan"), patch.object(launcher, "verify_predecessor"), \
                 patch.object(launcher.lifecycle, "verify_lifecycle"), patch.object(launcher.lifecycle, "require_unused"), \
                 patch.object(launcher.lifecycle, "write_exclusive", side_effect=write), \
                 patch.object(launcher.lifecycle, "holder_snapshot", side_effect=snapshot), \
                 patch.object(launcher.lifecycle, "proc_identity", side_effect=RuntimeError("evidence failure")), \
                 patch.object(launcher.subprocess, "Popen", return_value=Process()):
                outcome = launcher.run_plan(plan)
            self.assertFalse(outcome["successful"])
            self.assertEqual(events.count("wait"), 1)
            self.assertEqual(events.count("holder"), 2)
            self.assertLess(events.index("wait"), max(i for i, v in enumerate(events) if v == "holder"))

    def test_actual_launch_rejects_local_repo_stage(self):
        with patch.object(launcher, "verify_plan"):
            with self.assertRaisesRegex(ValueError, "independent authorized"):
                launcher.run_plan(dict(root="/tmp/fixture", stage_dir="/tmp/fixture/DeepInfer-PR148/GatedEngram"))


if __name__ == "__main__":
    unittest.main()
