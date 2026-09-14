"""Benchmark test contract, written before implementation.

1. Purpose: compare actual dense and reused hot prefills without inventing hits.
2. I/O: tokenized cases produce paired raw generations, direct TTFT metrics,
   separately timed capture/e2e, English token F1, and evidence qualification.
3. Failures: discarded prompt tokens, unequal pairs, warmup contamination,
   mixed timestamp clocks, fabricated accuracy/cache hits, and file overwrite.
4. Cheapest coverage: stdlib FakeLLM and metric fixtures; no model or GPU startup.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_redknot.py"
SPEC = importlib.util.spec_from_file_location("redknot_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class FakeClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        value = self.calls * 0.25
        self.calls += 1
        return value


class FakeLLM:
    def __init__(self, *, hits=True, metrics=True, rpc=True):
        self.hits, self.metrics, self.rpc = hits, metrics, rpc
        self.calls = []
        self.rpc_calls = []
        self.counters = Counter()
        self.cache_hits = 0
        self.worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                redknot_runtime=SimpleNamespace(stats=self.stats)
            )
        )

    def stats(self):
        return {"runtime": dict(self.counters), "cache": {"hits": self.cache_hits}}

    def collective_rpc(self, method, timeout):
        self.rpc_calls.append((method, timeout))
        if not self.rpc:
            raise TimeoutError("fake unavailable worker")
        return [method(self.worker)]

    def generate(self, prompts, sampling_params, *, use_tqdm):
        self.assert_single_prompt(prompts, use_tqdm)
        plan = sampling_params.extra_args["redknot"]
        mode = plan["mode"]
        prompt = list(prompts[0]["prompt_token_ids"])
        self.calls.append({"tokens": prompt, "params": sampling_params, "mode": mode})
        if mode == "capture":
            self.counters["capture_committed"] += 1
        elif mode == "reuse" and self.hits:
            self.cache_hits += 1
            self.counters["reuse_steps"] += 1
            self.counters["reused_local_query_rows"] += 12
            self.counters["restored_local_kv_rows"] += 6
        state = (
            SimpleNamespace(
                first_token_latency=0.4 if mode == "recomputed" else 0.2,
                arrival_time=1_800_000_000.0,
                first_token_ts=42.0,
                is_corrupted=False,
            )
            if self.metrics
            else None
        )
        completion = SimpleNamespace(
            text="cat dog" if mode == "recomputed" else "cat",
            token_ids=list(range(sampling_params.max_tokens)),
            finish_reason="length",
            stop_reason=None,
        )
        return [
            SimpleNamespace(
                request_id=str(len(self.calls)),
                prompt_token_ids=prompt,
                outputs=[completion],
                metrics=state,
                finished=True,
            )
        ]

    @staticmethod
    def assert_single_prompt(prompts, use_tqdm):
        assert len(prompts) == 1 and use_tqdm is False


def cases(references=True):
    return benchmark.parse_cases(
        {
            "cases": [
                {
                    "id": "case-a",
                    "chunks": [[11, 12], [11, 12], [21]],
                    "query": [31, 32],
                    "references": ["cat dog"] if references else [],
                }
            ]
        }
    )


def run(llm=None, **kwargs):
    return benchmark.run_benchmark(
        llm or FakeLLM(),
        SimpleNamespace,
        cases(),
        max_model_len=256,
        max_num_batched_tokens=256,
        namespace="unit-test",
        **kwargs,
    )


class BenchmarkTest(unittest.TestCase):
    def test_captures_unique_chunks_and_preserves_every_paired_prompt_token(self):
        llm = FakeLLM()
        report = run(llm)
        captures = [call for call in llm.calls if call["mode"] == "capture"]
        self.assertEqual([call["tokens"] for call in captures], [[11, 12], [21]])
        self.assertEqual(len(report["captures"]), 2)
        for call in captures:
            params = call["params"]
            self.assertEqual(params.max_tokens, 1)
            self.assertEqual(
                params.extra_args["redknot"]["chunks"],
                [{"start": 0, "end": len(call["tokens"])}],
            )
        for call in llm.calls[2:]:
            self.assertEqual(call["tokens"], [11, 12, 11, 12, 21, 31, 32])
            self.assertEqual(call["params"].temperature, 0.0)
            self.assertEqual(call["params"].max_tokens, 128)
            self.assertEqual(call["params"].min_tokens, 128)
            self.assertTrue(call["params"].ignore_eos)
            if call["mode"] == "reuse":
                self.assertTrue(
                    call["params"].extra_args["redknot"]["allow_approximate"]
                )
                self.assertEqual(
                    call["params"].extra_args["redknot"]["chunks"],
                    [
                        {"start": 0, "end": 2},
                        {"start": 2, "end": 4},
                        {"start": 4, "end": 5},
                    ],
                )

    def test_warmups_are_untimed_and_excluded_from_paired_aggregates(self):
        llm, clock = FakeLLM(), FakeClock()
        report = run(llm, clock=clock)
        self.assertEqual(len(report["warmups"]), 6)
        self.assertEqual(len(report["measurements"]), 20)
        self.assertEqual(clock.calls, 2 * (2 + 20))
        for row in report["warmups"]:
            self.assertIsNone(row["ttft_seconds"])
            self.assertIsNone(row["e2e_seconds"])
        for rows in (report["warmups"], report["measurements"]):
            previous = None
            for offset in range(0, len(rows), 2):
                pair = rows[offset : offset + 2]
                order = [row["mode"] for row in pair]
                self.assertEqual(set(order), {"dense", "reuse"})
                if previous is not None:
                    self.assertEqual(order, previous[::-1])
                previous = order
                self.assertEqual(
                    pair[0]["prompt_token_ids"], pair[1]["prompt_token_ids"]
                )
        self.assertEqual(report["summary"]["measured_pairs"], 10)

    def test_ttft_uses_latency_field_without_subtracting_different_clocks(self):
        report = run(clock=FakeClock())
        for row in report["measurements"]:
            self.assertEqual(row["e2e_seconds"], 0.25)
            self.assertEqual(
                row["ttft_seconds"], 0.4 if row["mode"] == "dense" else 0.2
            )
        ratios = report["summary"]["hot_ttft_ratio_dense_over_reuse"]
        self.assertEqual(ratios, {"p50": 2.0, "p95": 2.0})
        self.assertEqual(report["capture_cost"]["e2e_seconds_total"], 0.5)

    def test_english_f1_aggregation_reports_absolute_percentage_point_drop(self):
        report = run()
        summary = report["summary"]
        self.assertEqual(summary["mean_f1"]["dense"], 1.0)
        self.assertAlmostEqual(summary["mean_f1"]["reuse"], 2 / 3)
        self.assertAlmostEqual(summary["f1_drop_percentage_points"], 100 / 3)
        self.assertAlmostEqual(benchmark.token_f1("The cat, cat!", "cat dog"), 0.5)
        self.assertEqual(benchmark.reference_f1("a cat", ["dog", "the cat"]), 1.0)
        self.assertIsNone(benchmark.reference_f1("anything", []))
        self.assertAlmostEqual(benchmark.percentile([1, 2, 3, 4], 95), 3.85)

    def test_worker_rpc_deltas_prove_measured_reuse_not_warmup_hits(self):
        llm = FakeLLM()
        report = run(llm)
        self.assertTrue(report["qualified"], report["qualification_reasons"])
        for method, timeout in llm.rpc_calls:
            self.assertIs(method, benchmark.read_stats)
            self.assertEqual(timeout, 10)
        summary = report["summary"]["measured_reuse_evidence"]
        self.assertEqual(summary["cache_hits"], 10)
        self.assertEqual(summary["reuse_steps"], 10)
        self.assertEqual(summary["reused_local_query_rows"], 120)

    def test_missing_rpc_or_zero_reuse_cannot_qualify(self):
        for llm, reason in (
            (FakeLLM(rpc=False), "worker_stats_unavailable"),
            (FakeLLM(hits=False), "no_cache_hit"),
        ):
            with self.subTest(reason=reason):
                report = run(llm)
                self.assertFalse(report["qualified"])
                self.assertTrue(
                    any(reason in r for r in report["qualification_reasons"])
                )
        reply = benchmark.read_stats(SimpleNamespace(model_runner=SimpleNamespace()))
        self.assertFalse(reply["available"])

    def test_cache_hits_without_selected_head_rows_do_not_qualify(self):
        class NoRowsLLM(FakeLLM):
            def generate(self, *args, **kwargs):
                outputs = super().generate(*args, **kwargs)
                self.counters["reused_local_query_rows"] = 0
                return outputs

        report = run(NoRowsLLM())
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(
                "no_selected_query_head_rows_reused" in reason
                for reason in report["qualification_reasons"]
            )
        )
        self.assertEqual(report["summary"]["measured_reuse_evidence"]["cache_hits"], 10)

    def test_dense_requests_with_actual_reuse_do_not_qualify_as_dense(self):
        class ReusedDenseLLM(FakeLLM):
            def generate(self, prompts, sampling_params, **kwargs):
                outputs = super().generate(prompts, sampling_params, **kwargs)
                if sampling_params.extra_args["redknot"]["mode"] == "recomputed":
                    self.counters["reused_local_query_rows"] += 1
                return outputs

        report = run(ReusedDenseLLM())
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(
                "dense_unexpected_reuse" in reason
                for reason in report["qualification_reasons"]
            )
        )

    def test_missing_references_or_ttft_are_unavailable_not_perfect_accuracy(self):
        report = benchmark.run_benchmark(
            FakeLLM(metrics=False),
            SimpleNamespace,
            cases(references=False),
            max_model_len=256,
            max_num_batched_tokens=256,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(report["summary"]["mean_f1"], {"dense": None, "reuse": None})
        self.assertIsNone(report["summary"]["f1_drop_percentage_points"])
        self.assertEqual(
            report["summary"]["hot_ttft_ratio_dense_over_reuse"],
            {
                "p50": None,
                "p95": None,
            },
        )

    def test_case_validation_rejects_invalid_tokens_and_more_than_eight_chunks(self):
        for bad in (True, -1, 1.5, "token"):
            with self.subTest(token=bad), self.assertRaises(ValueError):
                benchmark.parse_cases(
                    {
                        "cases": [
                            {
                                "id": "x",
                                "chunks": [[bad]],
                                "query": [],
                            }
                        ]
                    }
                )
        with self.assertRaises(ValueError):
            benchmark.parse_cases(
                {
                    "cases": [
                        {
                            "id": "x",
                            "chunks": [[1]] * 9,
                            "query": [],
                        }
                    ]
                }
            )

    def test_minimum_pairs_lengths_and_generation_budget_are_enforced(self):
        for kwargs in (
            {"warmup_pairs": 2},
            {"measured_pairs": 9},
            {"output_tokens": 49},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run(**kwargs)
        with self.assertRaises(ValueError):
            benchmark.run_benchmark(
                FakeLLM(),
                SimpleNamespace,
                cases(),
                max_model_len=134,
                max_num_batched_tokens=134,
            )

    def test_cli_prepares_strict_local_engine_kwargs_before_importing_vllm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"architectures": ["Qwen3ForCausalLM"]})
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "model_revision": "local-test",
                        "local_heads": {"0": [0]},
                        "allow_approximate": True,
                    }
                ),
                encoding="utf-8",
            )
            args = benchmark.build_parser().parse_args(
                [
                    "--model",
                    str(model),
                    "--config",
                    str(config),
                    "--cases",
                    str(root / "cases.json"),
                    "--output",
                    str(root / "out.json"),
                    "--max-model-len",
                    "256",
                    "--max-num-batched-tokens",
                    "256",
                ]
            )
            with patch.dict("os.environ", {}, clear=True):
                kwargs, _ = benchmark.prepare_engine(args)
                import os

                self.assertEqual(
                    os.environ["VLLM_REDKNOT_CONFIG"], str(config.resolve())
                )
                self.assertEqual(os.environ["VLLM_PLUGINS"], "redknot")
                self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "0")
            self.assertFalse(kwargs["disable_log_stats"])
            self.assertTrue(kwargs["enforce_eager"])
            self.assertFalse(kwargs["enable_prefix_caching"])
            self.assertFalse(kwargs["enable_chunked_prefill"])
            self.assertEqual(kwargs["max_num_seqs"], 1)
            self.assertEqual(kwargs["tensor_parallel_size"], 1)
            self.assertEqual(kwargs["attention_config"], {"backend": "CUSTOM"})
            config.write_text(
                json.dumps(
                    {
                        "model_revision": "local-test",
                        "local_heads": {"0": [0]},
                        "allow_approximate": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                benchmark.prepare_engine(args)

    def test_flash_engine_uses_native_backend_not_custom_and_requires_bf16(self):
        from vllm_redknot.dsv4_runner import FLASH_REVISION

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["DeepseekV4ForCausalLM"],
                        "num_hidden_layers": 43,
                        "num_attention_heads": 64,
                        "head_dim": 512,
                        "qk_rope_head_dim": 64,
                        "o_groups": 8,
                    }
                )
            )
            config = root / "policy.json"
            config.write_text(
                json.dumps(
                    {
                        "engine_family": "deepseek_v4_flash",
                        "model_revision": FLASH_REVISION,
                        "local_heads": {"3": [1, 63]},
                        "allow_approximate": True,
                    }
                )
            )
            args = benchmark.build_parser().parse_args(
                [
                    "--model",
                    str(model),
                    "--config",
                    str(config),
                    "--cases",
                    str(root / "cases.json"),
                    "--output",
                    str(root / "result.json"),
                    "--max-model-len",
                    "4096",
                    "--max-num-batched-tokens",
                    "4096",
                ]
            )
            with patch.dict("os.environ", {}, clear=True):
                kwargs, metadata = benchmark.prepare_engine(args)
                self.assertEqual(
                    kwargs["attention_config"], {"backend": "FLASHMLA_SPARSE_DSV4"}
                )
                self.assertEqual(kwargs["dtype"], "bfloat16")
                self.assertEqual(kwargs["tensor_parallel_size"], 1)
                self.assertEqual(metadata["engine_family"], "deepseek_v4_flash")
                args.dtype = "float16"
                with self.assertRaisesRegex(ValueError, "bfloat16"):
                    benchmark.prepare_engine(args)

    def test_flash_qualification_requires_projected_and_native_state_evidence(self):
        report = run(engine_family="deepseek_v4_flash")
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(
                "no_clean_zoff_merge" in reason
                for reason in report["qualification_reasons"]
            )
        )

        class FakeFlashLLM(FakeLLM):
            def generate(self, prompts, sampling_params, **kwargs):
                output = super().generate(prompts, sampling_params, **kwargs)
                if sampling_params.extra_args["redknot"]["mode"] == "reuse":
                    self.counters["restored_local_kv_rows"] = 0
                    self.counters["reused_projected_token_rows"] += 3
                    self.counters["native_state_token_rows"] += 7
                    self.counters["launched_sparse_head_rows"] += 112
                return output

        report = run(FakeFlashLLM(), engine_family="deepseek_v4_flash")
        self.assertTrue(report["qualified"], report["qualification_reasons"])
        evidence = report["summary"]["measured_reuse_evidence"]
        self.assertEqual(evidence["restored_local_kv_rows"], 0)
        self.assertEqual(evidence["reused_projected_token_rows"], 30)
        self.assertEqual(evidence["native_state_token_rows"], 70)
        self.assertEqual(evidence["launched_sparse_head_rows"], 1120)
        self.assertTrue(
            any("no wo_a compute saving" in text for text in report["limitations"])
        )

    def test_output_is_exclusive_and_retains_raw_generations(self):
        report = run()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            benchmark.write_report(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["measurements"][0]["output"]["text"], "cat")
            with self.assertRaises(FileExistsError):
                benchmark.write_report(path, {"overwrite": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), saved)


if __name__ == "__main__":
    unittest.main()
