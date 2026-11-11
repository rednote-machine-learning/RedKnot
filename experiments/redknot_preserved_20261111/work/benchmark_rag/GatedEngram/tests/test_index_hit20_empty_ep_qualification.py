"""CPU proof/ownership tests. They never certify actual GPU EP execution."""
import copy
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from gpu_pilot import index_hit20_empty_ep_qualification as q
from kv_cache_engram.core import ValidationError

EXECUTION = "a" * 64
MODEL = "b" * 64


class FakeNativeMLP:
    def __init__(self):
        self.hidden_size = q.HIDDEN_SIZE
        self.ep_size = 8
        self.num_physical_experts = 256
        self.num_physical_topk = 8
        self.max_num_tokens = self.max_num_prefilling_tokens = 8192
        self.use_fp8_act, self.use_mega_moe = True, False
        self.gate = SimpleNamespace(weight=SimpleNamespace(shape=(256, 5120), dtype="torch.bfloat16"))

    def forward(self, *_):
        raise AssertionError("CPU contract must not run native MLP")


def fake_layers():
    return [SimpleNamespace(layer_idx=i, mlp=FakeNativeMLP()) for i in range(40)]


def rank_report(rank, layers):
    cases = []
    for layer in q.LAYERS:
        for pattern in q.PATTERNS:
            count = q.selected_count(pattern, rank)
            trace = dict(sequence=list(q.NATIVE_SEQUENCE),
                         received_tokens=0 if pattern == "all_empty" else 7)
            cases.append(dict(layer=layer, pattern=pattern, full_rows=q.FULL_ROWS,
                selected_rows=count, original_input_nonempty=True, exact_native_reference=True,
                finite=True, inactive_rows_exact_zero=True, output_shape=[q.FULL_ROWS, q.HIDDEN_SIZE],
                output_dtype="torch.bfloat16", strict_cuda_sync_debug_mode=2,
                strict_cuda_sync_mode_before_after=[2, 2, 2, 2],
                native_ep_called_when_empty=count == 0, reference_sha256="c" * 64,
                candidate_sha256="c" * 64, reference_trace=trace.copy(), candidate_trace=trace.copy()))
    return q.seal(dict(schema="index-hit20-empty-ep-rank-v1", passed=True, rank=rank, world_size=8,
        execution_sha256=EXECUTION, model_sha256=MODEL,
        layer_abis=[q.native_abi(layers[i]) for i in q.LAYERS], cases=cases,
        quality_eligible=False, timing_valid=False))


def proof(layers=None):
    layers = fake_layers() if layers is None else layers
    return q.build_all_ranks_report([rank_report(r, layers) for r in range(8)],
                                    execution_sha256=EXECUTION, model_sha256=MODEL)


def reseal(value):
    value.pop("sha256", None)
    return q.seal(value)


class EmptyEPProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers = fake_layers()
        cls.valid = proof(cls.layers)

    def test_import_is_cpu_only(self):
        code = "import sys; import gpu_pilot.index_hit20_empty_ep_qualification; assert 'torch' not in sys.modules; assert not any(x.startswith('deepinfer') for x in sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_eight_rank_four_pattern_all_layer_coverage(self):
        self.assertTrue(q.validate_all_ranks_report(self.valid, execution_sha256=EXECUTION, model_sha256=MODEL))
        self.assertEqual(self.valid["checks"], 1184)
        self.assertGreater(self.valid["empty_rank_remote_work_cases"], 0)

    def test_zero_selection_patterns_do_not_insert_dummy_rows(self):
        self.assertEqual([q.selected_count("all_empty", r) for r in range(8)], [0] * 8)
        self.assertEqual([q.selected_count("rank0_only", r) for r in range(8)], [127] + [0] * 7)
        self.assertEqual([q.selected_count("rank0_empty", r) for r in range(8)], [0] + [127] * 7)
        self.assertEqual([q.selected_count("alternating_empty", r) for r in range(8)], [0, 127] * 4)

    def test_invalid_rank_pattern_rejected(self):
        for pattern, rank in (("unknown", 0), ("all_empty", -1), ("all_empty", 8), ("all_empty", True)):
            with self.assertRaises(ValidationError):
                q.selected_count(pattern, rank)

    def test_no_report_or_missing_rank_cannot_issue_capability(self):
        for reports in ([], self.valid["reports"][:-1], self.valid["reports"] + [self.valid["reports"][0]]):
            with self.assertRaises(ValidationError):
                q.build_all_ranks_report(reports, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_case_failures_rejected_even_when_resealed(self):
        edits = {"selected_rows": 1, "full_rows": 0, "exact_native_reference": False, "finite": False,
                 "inactive_rows_exact_zero": False, "output_shape": [0, 5120], "output_dtype": "torch.float32",
                 "strict_cuda_sync_debug_mode": 0, "native_ep_called_when_empty": False,
                 "candidate_sha256": "d" * 64, "original_input_nonempty": False}
        for key, bad in edits.items():
            with self.subTest(key=key):
                report = copy.deepcopy(self.valid["reports"][0])
                report["cases"][0][key] = bad
                with self.assertRaises(ValidationError):
                    q.validate_rank_report(reseal(report), rank=0, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_each_collective_and_routed_call_required(self):
        for side in ("reference_trace", "candidate_trace"):
            for sequence in ([], ["dispatch", "combine"], ["combine", "routed_experts", "dispatch"],
                             ["dispatch", "routed_experts", "combine", "combine"]):
                report = copy.deepcopy(self.valid["reports"][0])
                report["cases"][0][side]["sequence"] = sequence
                with self.assertRaises(ValidationError):
                    q.validate_rank_report(reseal(report), rank=0, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_all_empty_must_receive_no_real_tokens(self):
        report = copy.deepcopy(self.valid["reports"][0])
        report["cases"][0]["candidate_trace"]["received_tokens"] = 1
        with self.assertRaises(ValidationError):
            q.validate_rank_report(reseal(report), rank=0, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_remote_expert_work_must_be_observed_on_empty_rank(self):
        reports = copy.deepcopy(self.valid["reports"])
        for report in reports:
            for case in report["cases"]:
                if case["selected_rows"] == 0:
                    case["candidate_trace"]["received_tokens"] = 0
                    case["reference_trace"]["received_tokens"] = 0
            report.update(reseal(report))
        with self.assertRaises(ValidationError):
            q.build_all_ranks_report(reports, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_missing_or_duplicate_layer_pattern_rejected(self):
        for remove in (True, False):
            report = copy.deepcopy(self.valid["reports"][0])
            if remove:
                report["cases"].pop()
            else:
                report["cases"][-1] = copy.deepcopy(report["cases"][0])
            with self.assertRaises(ValidationError):
                q.validate_rank_report(reseal(report), rank=0, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_seal_and_execution_identity_enforced(self):
        report = copy.deepcopy(self.valid)
        report["passed"] = False
        with self.assertRaises(ValidationError):
            q.validate_all_ranks_report(report, execution_sha256=EXECUTION, model_sha256=MODEL)
        for execution, model in (("d" * 64, MODEL), (EXECUTION, "e" * 64)):
            with self.assertRaises(ValidationError):
                q.EmptyEPCapability(self.layers, self.valid, execution_sha256=execution, model_sha256=model)

    def test_rank_reorder_is_not_allowed(self):
        reports = copy.deepcopy(self.valid["reports"])
        reports.reverse()
        with self.assertRaises(ValidationError):
            q.build_all_ranks_report(reports, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_proof_not_eligible_for_quality_or_timing(self):
        for key in ("quality_eligible", "timing_valid"):
            report = copy.deepcopy(self.valid)
            report[key] = True
            with self.assertRaises(ValidationError):
                q.validate_all_ranks_report(reseal(report), execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_actual_runtime_objects_bound(self):
        capability = q.EmptyEPCapability(self.layers, self.valid, execution_sha256=EXECUTION, model_sha256=MODEL)
        meta = SimpleNamespace(is_prompt=True, num_reqs=1, total_q=140)
        plan = SimpleNamespace(full_rows=140, selected_count=0)
        self.assertTrue(capability.permits(self.layers[20], meta, plan))
        with self.assertRaises(ValidationError):
            capability.permits(fake_layers()[20], meta, plan)

    def test_native_abi_change_revokes_capability(self):
        layers = fake_layers()
        capability = q.EmptyEPCapability(layers, self.valid, execution_sha256=EXECUTION, model_sha256=MODEL)
        layers[20].mlp.num_physical_topk += 1
        with self.assertRaises(ValidationError):
            capability.permits(layers[20], SimpleNamespace(is_prompt=True, num_reqs=1, total_q=140),
                               SimpleNamespace(full_rows=140, selected_count=0))

    def test_capability_only_zero_selected_nonempty_prompt(self):
        capability = q.EmptyEPCapability(self.layers, self.valid, execution_sha256=EXECUTION, model_sha256=MODEL)
        for rows, count, prompt, requests in ((0, 0, True, 1), (140, 1, True, 1), (140, 0, False, 1), (140, 0, True, 2)):
            with self.assertRaises(ValidationError):
                capability.permits(self.layers[20], SimpleNamespace(is_prompt=prompt, num_reqs=requests, total_q=rows),
                                   SimpleNamespace(full_rows=rows, selected_count=count))

    def test_native_gpu_function_contains_no_skipped_collective_fallback(self):
        source = inspect.getsource(q.run_empty_ep_qualification)
        self.assertIn('t.cuda.set_sync_debug_mode("error")', source)
        self.assertIn("reference = layer.mlp(", source)
        self.assertIn("actual = layer.mlp(", source)
        self.assertIn("t.inference_mode()", source)
        self.assertIn("per_token_cast(reference_bf16", source)

    def test_reference_candidate_must_have_identical_native_receive_counts(self):
        report = copy.deepcopy(self.valid["reports"][0])
        report["cases"][1]["candidate_trace"]["received_tokens"] += 1
        with self.assertRaises(ValidationError):
            q.validate_rank_report(reseal(report), rank=0, execution_sha256=EXECUTION, model_sha256=MODEL)

    def test_partial_observer_install_restores_owned_methods(self):
        class Native:
            def dispatch(self, *_):
                return None
            def routed_experts_forward(self, *_):
                return None
            def combine(self, *_):
                return None
            def shared_forward(self, *_):
                return None
        native = Native()
        native.combine = lambda *args: "foreign"
        with self.assertRaises(ValidationError):
            with q._NativeTrace(native):
                pass
        self.assertNotIn("dispatch", native.__dict__)
        self.assertNotIn("routed_experts_forward", native.__dict__)
        self.assertEqual(native.combine(), "foreign")

    def test_each_layer_needs_actual_remote_work_on_empty_rank(self):
        reports = copy.deepcopy(self.valid["reports"])
        for report in reports:
            for case in report["cases"]:
                if case["selected_rows"] == 0 and case["layer"] == 39:
                    case["candidate_trace"]["received_tokens"] = 0
                    case["reference_trace"]["received_tokens"] = 0
            report.update(reseal(report))
        with self.assertRaises(ValidationError):
            q.build_all_ranks_report(reports, execution_sha256=EXECUTION, model_sha256=MODEL)


if __name__ == "__main__":
    unittest.main()
