"""CPU-only contracts: explicit opt-in, actual native evidence, untruncated inputs.

No tests load weights, import GPU packages, or treat mocks as model qualification.
"""

import builtins
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/check_dsv4_model_smoke.py"
SPEC = importlib.util.spec_from_file_location("redknot_model_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


@contextmanager
def forbid_gpu_imports():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "triton", "vllm"}:
            raise AssertionError(f"unexpected GPU dependency import: {name}")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        yield


class TokenizerWithoutOffsets:
    """Only documented encode/chat-template interfaces; no fast offsets method."""

    def __init__(self):
        self.vocabulary = {}
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.calls.append(text)
        return [
            self.vocabulary.setdefault(word, len(self.vocabulary))
            for word in re.findall(r"\S+", text)
        ]

    def apply_chat_template(self, messages, *, tokenize, enable_thinking):
        assert enable_thinking is False
        assert len(messages) == 1 and messages[0]["role"] == "user"
        rendered = "<bos> <user> " + messages[0]["content"] + " <assistant>"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class FakeLLM:
    def __init__(self):
        self.tokenizer = TokenizerWithoutOffsets()
        self.counters = Counter()
        self.cache = {"hits": 0, "entries": 0, "bytes": 0, "max_bytes": 1024}
        self.calls = []
        self.replies = 1
        self.mutate = lambda output: None
        self.on_generate = lambda: None
        runtime_class = type(
            "DSV4Runtime",
            (),
            {
                "__module__": "vllm_redknot.dsv4_runtime",
                "stats": lambda _: self.stats(),
            },
        )
        runtime = runtime_class()
        runtime.settings = SimpleNamespace(model_revision=smoke.FLASH_REVISION)
        self.worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                redknot_runtime=runtime, _redknot_dsv4_installed=True
            ),
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    model="/models/fixed", dtype="torch.bfloat16"
                ),
                parallel_config=SimpleNamespace(
                    **dict.fromkeys(smoke.PARALLEL_FIELDS, 1)
                ),
                attention_config=SimpleNamespace(
                    backend=SimpleNamespace(name="FLASHMLA_SPARSE_DSV4")
                ),
            ),
        )

    def stats(self):
        return {"runtime": dict(self.counters), "cache": dict(self.cache)}

    def get_tokenizer(self):
        self.calls.append("tokenizer")
        return self.tokenizer

    def collective_rpc(self, method, timeout):
        assert timeout == 10
        self.calls.append("rpc")
        return [method(self.worker) for _ in range(self.replies)]

    def generate(self, prompts, sampling_params, *, use_tqdm):
        self.calls.append("generate")
        self.params = sampling_params
        self.prompts = prompts
        assert use_tqdm is False and len(prompts) == 1
        self.counters["recomputed_steps"] += 1
        self.on_generate()
        output = SimpleNamespace(
            prompt_token_ids=list(prompts[0]["prompt_token_ids"]),
            finished=True,
            metrics=SimpleNamespace(is_corrupted=False),
            outputs=[
                SimpleNamespace(
                    text="A triangle has three sides.",
                    token_ids=list(range(8)),
                    finish_reason="length",
                )
            ],
        )
        self.mutate(output)
        return [output]


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.argv = [
            "--run",
            "--verified-model",
            "--model",
            str(self.directory / "model"),
            "--config",
            str(self.directory / "config.json"),
            "--cases-output",
            str(self.directory / "cases.json"),
            "--report-output",
            str(self.directory / "report.json"),
        ]
        environment = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def args(self, argv=None):
        return smoke.build_parser().parse_args(self.argv if argv is None else argv)

    def test_import_and_help_do_not_import_gpu_libraries(self):
        spec = importlib.util.spec_from_file_location("fresh_smoke", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        with forbid_gpu_imports(), redirect_stdout(io.StringIO()) as output:
            spec.loader.exec_module(module)
            with self.assertRaises(SystemExit) as caught:
                module.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--verified-model", output.getvalue())
        self.assertIn("operator", output.getvalue())

    def test_both_opt_ins_required_before_any_preparation_or_gpu_import(self):
        for flag in ("--run", "--verified-model"):
            argv = [item for item in self.argv if item != flag]
            with (
                self.subTest(flag=flag),
                forbid_gpu_imports(),
                patch.object(smoke, "prepare_engine") as prepare,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                smoke.main(argv)
            self.assertEqual(caught.exception.code, 2)
            prepare.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_visible_device_and_distributed_guards(self):
        for device in ("", "0,1", "-1", "cuda:0", "MIG-test", "GPU-not-a-uuid"):
            with (
                self.subTest(device=device),
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=device),
            ):
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    smoke.validate_args(self.args())
        for name, value in (
            ("VLLM_DP_SIZE", "2"),
            ("VLLM_DP_RANK", "1"),
            ("WORLD_SIZE", "2"),
        ):
            with self.subTest(name=name), patch.dict(os.environ, {name: value}):
                with self.assertRaisesRegex(ValueError, "distributed"):
                    smoke.validate_args(self.args())
        with patch.dict(
            os.environ, CUDA_VISIBLE_DEVICES="GPU-9ce76ce0-362e-fc10-cecd-68afb62b9dfb"
        ):
            smoke.validate_args(self.args())

    def test_budget_and_output_paths_checked_before_engine(self):
        for option, value in (
            ("--max-model-len", "128"),
            ("--gpu-memory-utilization", "nan"),
            ("--gpu-memory-utilization", "0"),
        ):
            with (
                self.subTest(option=option, value=value),
                self.assertRaises(ValueError),
            ):
                smoke.validate_args(self.args(self.argv + [option, value]))
        args = self.args()
        args.report_output = args.cases_output
        with self.assertRaisesRegex(ValueError, "distinct"):
            smoke.validate_args(args)
        broken = self.directory / "broken"
        broken.symlink_to(self.directory / "does-not-exist")
        with self.assertRaises(FileExistsError):
            smoke.new_output_path(broken)

    def test_native_generation_is_exactly_once_recomputed_with_rpc_evidence(self):
        llm = FakeLLM()
        ticks = iter((10.0, 11.5))
        with forbid_gpu_imports():
            result, tokenizer = smoke.run_native(
                llm, SimpleNamespace, max_model_len=2048, clock=lambda: next(ticks)
            )
        self.assertEqual(llm.calls, ["tokenizer", "rpc", "generate", "rpc"])
        plan = llm.params.extra_args["redknot"]
        self.assertEqual(plan["mode"], "recomputed")
        self.assertFalse(plan["allow_approximate"])
        self.assertEqual(
            plan["chunks"], [{"start": 0, "end": len(result["prompt_token_ids"])}]
        )
        self.assertEqual(llm.params.min_tokens, 8)
        self.assertEqual(llm.params.max_tokens, 8)
        self.assertTrue(llm.params.ignore_eos)
        self.assertEqual(result["output_token_ids"], list(range(8)))
        self.assertEqual(result["e2e_seconds"], 1.5)
        self.assertEqual(result["worker_delta"]["runtime"], {"recomputed_steps": 1})
        self.assertIs(tokenizer, llm.tokenizer)
        self.assertFalse(result["quality_assessed"])
        self.assertFalse(result["reuse_assessed"])

    def test_native_rejects_missing_rpc_or_non_tp1_before_generation(self):
        llm = FakeLLM()
        llm.replies = 2
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            smoke.run_native(llm, SimpleNamespace, max_model_len=2048)
        self.assertNotIn("generate", llm.calls)
        llm = FakeLLM()
        llm.worker.vllm_config.parallel_config.data_parallel_size = 2
        with self.assertRaisesRegex(RuntimeError, "parallelism"):
            smoke.run_native(llm, SimpleNamespace, max_model_len=2048)
        self.assertNotIn("generate", llm.calls)

    def test_native_rejects_capture_reuse_or_missing_recomputed(self):
        for key in (
            "capture_layers",
            "reuse_steps",
            "reused_local_query_rows",
            "launched_sparse_head_rows",
        ):
            llm = FakeLLM()
            llm.on_generate = lambda key=key: llm.counters.update({key: 1})
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "capture/reuse"),
            ):
                smoke.run_native(llm, SimpleNamespace, max_model_len=2048)
        llm = FakeLLM()
        llm.on_generate = llm.counters.clear
        with self.assertRaisesRegex(RuntimeError, "recomputed"):
            smoke.run_native(llm, SimpleNamespace, max_model_len=2048)

    def test_native_rejects_truncated_empty_corrupted_or_short_output(self):
        changes = (
            lambda out: out.prompt_token_ids.pop(),
            lambda out: out.outputs[0].token_ids.pop(),
            lambda out: setattr(out.outputs[0], "text", ""),
            lambda out: setattr(out.metrics, "is_corrupted", True),
            lambda out: setattr(out, "finished", False),
        )
        for change in changes:
            llm = FakeLLM()
            llm.mutate = change
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                smoke.run_native(llm, SimpleNamespace, max_model_len=2048)

    def test_cases_have_two_long_original_docs_no_references_and_all_tokens(self):
        tokenizer = TokenizerWithoutOffsets()
        with forbid_gpu_imports():
            document = smoke.build_cases(
                tokenizer, boundary_tokens=128, max_model_len=2048
            )
        case = document["cases"][0]
        provenance = document["provenance"]
        self.assertEqual(len(case["chunks"]), 2)
        self.assertTrue(all(len(chunk) > 128 for chunk in case["chunks"]))
        self.assertTrue(all(size > 128 for size in provenance["document_token_counts"]))
        self.assertEqual(case["references"], [])
        self.assertEqual(
            case["chunks"][0] + case["chunks"][1] + case["query"],
            provenance["full_prompt_token_ids"],
        )
        self.assertEqual(
            "".join(provenance["fragment_texts"]), provenance["rendered_text"]
        )
        self.assertEqual(provenance["truncated_tokens"], 0)
        self.assertIn("not a quality", provenance["purpose"])

    def test_cases_refuse_too_short_documents_or_budget_instead_of_truncating(self):
        tokenizer = TokenizerWithoutOffsets()
        with self.assertRaisesRegex(ValueError, "no truncation"):
            smoke.build_cases(tokenizer, boundary_tokens=128, max_model_len=256)
        with self.assertRaisesRegex(ValueError, "dirty boundary"):
            smoke.build_cases(tokenizer, boundary_tokens=500, max_model_len=4096)

    def test_new_json_only_never_overwrites(self):
        path = self.directory / "new.json"
        smoke.write_json(path, {"status": "test-only"})
        with self.assertRaises(FileExistsError):
            smoke.write_json(path, {"status": "replacement"})
        self.assertEqual(json.loads(path.read_text()), {"status": "test-only"})

    def mocked_main(self, llm, *, case_error=None):
        shutdown = Mock()
        llm.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown))
        factories = SimpleNamespace(
            LLM=Mock(return_value=llm), SamplingParams=SimpleNamespace
        )
        original_cases = smoke.build_cases
        with (
            patch.dict(sys.modules, {"vllm": factories}),
            patch.object(
                smoke,
                "prepare_engine",
                return_value=({}, {"configuration": {"boundary_tokens": 128}}),
            ),
            patch.object(
                smoke,
                "build_cases",
                side_effect=case_error or original_cases,
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            result = smoke.main(self.argv)
        shutdown.assert_called_once_with(timeout=30)
        return result, json.loads((self.directory / "report.json").read_text())

    def test_main_writes_cases_only_after_native_success_and_shuts_down_owned_engine(
        self,
    ):
        llm = FakeLLM()
        code, report = self.mocked_main(llm)
        self.assertEqual(code, 0)
        self.assertTrue(report["native_smoke_passed"])
        self.assertFalse(report["full_redknot_benchmark_passed"])
        self.assertTrue(report["shutdown"]["call_returned"])
        self.assertFalse(report["shutdown"]["gpu_free_independently_verified"])
        self.assertEqual(report["native"]["output_text"], "A triangle has three sides.")
        self.assertTrue((self.directory / "cases.json").is_file())
        self.assertEqual(llm.calls.count("generate"), 1)

    def test_native_failure_does_not_write_cases_and_still_shuts_down(self):
        llm = FakeLLM()
        llm.mutate = lambda output: output.outputs[0].token_ids.pop()
        code, report = self.mocked_main(llm)
        self.assertEqual(code, 2)
        self.assertFalse(report["native_smoke_passed"])
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse((self.directory / "cases.json").exists())

    def test_case_failure_still_shuts_down_and_does_not_claim_complete_smoke(self):
        code, report = self.mocked_main(FakeLLM(), case_error=ValueError("case failed"))
        self.assertEqual(code, 2)
        self.assertTrue(report["native_smoke_passed"])
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error"]["message"], "case failed")
        self.assertFalse((self.directory / "cases.json").exists())


if __name__ == "__main__":
    unittest.main()
