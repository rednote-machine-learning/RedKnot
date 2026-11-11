"""CPU contracts and optional Torch operator fixtures, not model accuracy tests."""
from contextlib import nullcontext
import importlib.util
import json
from types import SimpleNamespace
import unittest

from kv_cache_engram.core import ValidationError, digest
from kv_cache_engram.native_reuse import DOC
from kv_cache_engram.sensitivity import (
    NativeSensitivityObserver, SensitivityConfig, candidate_spans,
    memory_preflight, select_worst_span, split_span, tensor_statistics,
)


class PlanTests(unittest.TestCase):
    def test_guards_and_chunk_fragments_are_not_candidates(self):
        doc = dict(start=7, end=7+DOC, key="a"*64)
        spans = candidate_spans([doc], 0, 8192)
        self.assertEqual(spans[0]["document_local_start"], 128)
        self.assertEqual(spans[0]["input_start"], 135)
        self.assertEqual(spans[-1]["document_local_end"], 7680)
        self.assertTrue(all(s["input_end"] <= 8192 for s in spans))

    def test_tail_guard(self):
        doc = dict(start=0, end=DOC, key="a"*64)
        spans = candidate_spans([doc], DOC-8192, DOC)
        self.assertEqual(spans[-1]["document_local_end"], DOC-128)

    def test_split_keeps_two_nonempty_guard_clipped_halves(self):
        span = candidate_spans([dict(start=0, end=DOC, key="a"*64)], 0, 512)[0]
        halves = split_span(span)
        self.assertEqual([(s["input_start"], s["input_end"]) for s in halves], [(128, 256), (256, 512)])

    def test_source_ranking_tie_uses_position(self):
        spans = candidate_spans([dict(start=0, end=DOC, key="a"*64)], 0, 1024)
        self.assertEqual(select_worst_span(list(reversed(spans)), [1., 1.]), spans[0])
        self.assertEqual(select_worst_span(spans, [0., 2.]), spans[1])

    def test_source_ranking_rejects_nonfinite_or_missing_scores(self):
        spans = candidate_spans([dict(start=0, end=DOC, key="a"*64)], 0, 512)
        for values in ([], [float("nan")], [float("inf")], [-1.]):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                select_worst_span(spans, values)

    def test_invalid_configuration(self):
        for kwargs in (dict(layers=(9, 3)), dict(layers=(0,)), dict(layers=(3, 3)),
                       dict(norm_floor=0), dict(metric_chunk_rows=1000), dict(top_tokens=0),
                       dict(restore_block=False), dict(max_extra_bytes=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                SensitivityConfig(**kwargs)


class MemoryPreflightTests(unittest.TestCase):
    def tensors(self, cuda=False):
        value = lambda n: SimpleNamespace(is_cuda=cuda, device="cuda:0" if cuda else "cpu",
            numel=lambda: n, element_size=lambda: 2)
        return value(1024*2*512), value(1152*512)

    def test_cpu_never_reads_cuda(self):
        q, kv = self.tensors()
        result = memory_preflight(q, kv, config=SensitivityConfig(), torch=SimpleNamespace())
        self.assertIsNone(result["physical_free_bytes"])
        self.assertFalse(result["allocation_success_guaranteed"])

    def test_cuda_queries_the_explicit_device_and_retains_evidence(self):
        q, kv = self.tensors(cuda=True)
        devices = []
        def query(device):
            devices.append(device)
            return 8 << 30, 80 << 30
        t = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=query))
        result = memory_preflight(q, kv, config=SensitivityConfig(), torch=t)
        self.assertEqual(devices, ["cuda:0"])
        self.assertEqual(result["physical_free_bytes"], 8 << 30)

    def test_low_free_memory_fails_closed_without_changing_budget(self):
        q, kv = self.tensors(cuda=True)
        t = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (1, 80 << 30)))
        with self.assertRaises(ValidationError):
            memory_preflight(q, kv, config=SensitivityConfig(), torch=t)

    def test_configured_limit_checked_before_cuda_query(self):
        q, kv = self.tensors(cuda=True)
        with self.assertRaises(ValidationError):
            memory_preflight(q, kv, config=SensitivityConfig(max_extra_bytes=1), torch=SimpleNamespace())


TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "optional Torch CPU fixture dependency is absent")
class TorchDiagnosticTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.config = SensitivityConfig(layers=(3,), metric_chunk_rows=128, top_tokens=4,
                                         restore_control=True)

    def fixture(self, *, rows=1024, documents=True):
        t = self.t
        class Module:
            layer_idx = 3
            use_fused_norm_rope_attn_rope_cast = False
            def __init__(self):
                self.attention_calls = 0
                self.projection_calls = 0
            def _prefill_attention(self, q, kv, ca, swa, cached, current, metadata):
                self.attention_calls += 1
                # Deliberately mixes source rows into later query-token outputs.
                current_kv = kv[cached:cached+current, 0]
                mean = current_kv.cumsum(dim=0)/t.arange(1, current+1)[:, None]
                return t.stack((mean, mean*2), dim=1)
            def _wv_and_o_proj(self, context, positions):
                self.projection_calls += 1
                context.add_(2)  # Native inverse RoPE also mutates its input.
                return context.sum(dim=1)
        module = Module()
        class Ops:
            native_execution = False
            backend = "torch-cpu-fixture-not-native"
            torch = t
            @staticmethod
            def version(value):
                return value._version
            @staticmethod
            def postprocess(module, value, metadata):
                assert value.shape[0] == metadata.total_q
                return value.clone().unsqueeze(1), None
        tensor = t.zeros(DOC, 512)
        marker = dict(request_id="fixture", phase="diagnostic", start=0, end=rows)
        metadata = SimpleNamespace(total_q=rows, positions=t.arange(rows), canonical_request_binding=marker)
        frame = SimpleNamespace(marker=marker, module=module, metadata=metadata)
        hook = SimpleNamespace(modules={3: module}, state="RUNNING", mode="native",
            identity_qualified=True, request_id="fixture", phase="diagnostic", input_sha256="a"*64,
            ids=range(rows), ops=Ops(), allow_sync=nullcontext,
            documents=[dict(start=0, end=DOC, key="b"*64)] if documents else [],
            bank={("b"*64, 3):dict(tensor=tensor, complete=True, version=tensor._version)},
            _frame=lambda m: frame)
        q = t.zeros(rows, 2, 512)
        kv = t.ones(rows+128, 1, 512)
        args = (q, kv, None, None, 128, rows, metadata)
        return module, hook, args

    def test_all_rows_heads_and_signed_cosine_metrics(self):
        reference = self.t.ones(5, 2, 3)
        actual = reference.clone()
        actual[4, 1] = -1
        metrics, top, blocks = tensor_statistics(reference, actual, start=510,
            config=self.config, torch=self.t)
        self.assertEqual(metrics[0]["relative_l2_max"], 0)
        self.assertAlmostEqual(metrics[1]["relative_l2_max"], 2)
        self.assertAlmostEqual(metrics[1]["cosine_distance_max"], 2)
        self.assertEqual(metrics[1]["worst_token_index"], 514)
        self.assertEqual(top[0]["token_index"], 514)
        self.assertEqual(sum(x["rows"] for x in blocks), 5)
        self.assertEqual(len(blocks), 2)

    def test_zero_vectors_have_zero_cosine_distance(self):
        value = self.t.zeros(2, 1, 3)
        metrics, _, _ = tensor_statistics(value, value, start=0, config=self.config, torch=self.t)
        self.assertEqual(metrics[0]["cosine_distance_max"], 0)

    def test_nonfinite_is_rejected(self):
        a = self.t.ones(2, 1, 3)
        b = a.clone()
        b[1, 0, 0] = float("nan")
        with self.assertRaises(ValidationError):
            tensor_statistics(a, b, start=0, config=self.config, torch=self.t)

    def test_counterfactual_does_not_mutate_native_context_kv_or_bank(self):
        module, hook, args = self.fixture()
        kv_before, bank_before = args[1].clone(), hook.bank[("b"*64, 3)]["tensor"].clone()
        expected = module._prefill_attention(*args)
        observer = NativeSensitivityObserver(hook, self.config).install()
        observer.reset()
        actual = module._prefill_attention(*args)
        self.assertTrue(self.t.equal(expected, actual))
        self.assertTrue(self.t.equal(kv_before, args[1]))
        self.assertTrue(self.t.equal(bank_before, hook.bank[("b"*64, 3)]["tensor"]))
        hook.state = "COMPLETE"
        result = observer.finish()
        self.assertEqual(result["coverage"], {"3": 1024})
        self.assertEqual(result["head_count"], 2)
        self.assertEqual([r["restoration_label"] for r in result["records"]],
            ["all_cached", "high_error_512", "left_half_256", "right_half_256", "control_512"])
        for record in result["records"]:
            self.assertEqual(record["source_rows"], record["reused_rows"]+record["restored_rows"])
            self.assertEqual(len(record["heads"]), 2)
            self.assertEqual(record["attention_update"]["rows"], 1024)
        decoded = json.loads(json.dumps(result))
        checksum = decoded.pop("sha256")
        self.assertEqual(digest(decoded), checksum)
        observer.close()
        self.assertNotIn("_prefill_attention", module.__dict__)

    def test_inactive_observer_is_operator_passthrough(self):
        module, hook, args = self.fixture()
        observer = NativeSensitivityObserver(hook, self.config).install()
        module._prefill_attention(*args)
        self.assertEqual(module.attention_calls, 1)
        self.assertEqual(module.projection_calls, 0)
        self.assertEqual(observer.records, [])
        observer.close()

    def test_finish_schema_integrates_with_strict_reader_without_claiming_fixture_native(self):
        from gpu_pilot.sensitivity_protocol import validate_observer
        module, hook, args = self.fixture()
        observer = NativeSensitivityObserver(hook, self.config).install()
        self.addCleanup(observer.close)
        observer.reset()
        module._prefill_attention(*args)
        hook.state = "COMPLETE"
        result = observer.finish()
        # Real fixture output must be rejected as native numerical evidence.
        with self.assertRaises(ValidationError):
            validate_observer(result, input_sha256="a"*64, input_tokens=1024, expected_layers=(3,))
        # Explicit provenance stub tests only serialization/schema compatibility.
        # It is not emitted as an experiment result or a native qualification.
        declared = json.loads(json.dumps(result))
        declared.pop("sha256")
        declared["native_execution"] = True
        declared["operator_backend"] = "deepinfer-sm90-native-bfloat16-kv-only-v1"
        declared["sha256"] = digest(declared)
        summary = validate_observer(declared, input_sha256="a"*64,
                                    input_tokens=1024, expected_layers=(3,))
        self.assertIsNotNone(summary)

    def test_gated_request_is_rejected(self):
        module, hook, args = self.fixture()
        observer = NativeSensitivityObserver(hook, self.config).install()
        hook.mode = "gated"
        with self.assertRaises(ValidationError):
            observer.reset()
        observer.close()

    def test_zero_candidate_chunk_has_record_not_fake_reuse(self):
        module, hook, args = self.fixture(documents=False)
        observer = NativeSensitivityObserver(hook, self.config).install()
        observer.reset()
        module._prefill_attention(*args)
        hook.state = "COMPLETE"
        result = observer.finish()
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["reused_rows"], 0)
        self.assertEqual(result["records"][0]["attention_update"]["relative_l2_max"], 0)
        observer.close()

    def test_incomplete_coverage_rejected(self):
        module, hook, args = self.fixture()
        observer = NativeSensitivityObserver(hook, self.config).install()
        observer.reset()
        hook.state = "COMPLETE"
        with self.assertRaises(ValidationError):
            observer.finish()
        observer.close()

    def test_mutated_bank_rejected(self):
        module, hook, args = self.fixture()
        observer = NativeSensitivityObserver(hook, self.config).install()
        observer.reset()
        hook.bank[("b"*64, 3)]["tensor"].add_(1)
        with self.assertRaises(ValidationError):
            module._prefill_attention(*args)
        observer.close()

    def test_foreign_hook_is_not_overwritten(self):
        module, hook, args = self.fixture()
        observer = NativeSensitivityObserver(hook, self.config).install()
        replacement = lambda *a: None
        module._prefill_attention = replacement
        with self.assertRaises(ValidationError):
            observer.close()
        self.assertIs(module._prefill_attention, replacement)


if __name__ == "__main__":
    unittest.main()
