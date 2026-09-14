"""CPU-only data, source policy and failure-before-GPU migration contracts."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks import multimodel_rag as rag
from vllm_redknot.config import validate_config


def encode(text):
    return list(map(ord, text))


def decode(tokens):
    return "".join(map(chr, tokens))


def row(index, context, answer="A"):
    return {
        "source_row_index": index,
        "input": f"Question {index}?",
        "context": context,
        "answers": [answer],
    }


def prepare(key, rows, **overrides):
    settings = {
        "samples": 2,
        "chunk_tokens": 80,
        "max_context_tokens": 320,
        "num_chunks": 4 if key in {"mistral", "qwen35_397b"} else 0,
        "seed": 0,
    }
    settings.update(overrides)
    return rag.prepare_rows(key, rows, "test", encode, decode, **settings)


class MultiModelRagTests(unittest.TestCase):
    def test_all_four_entrypoints_default_cpu_only(self):
        names = ("Mistral_RAG", "Llama3.3_RAG", "Qwen3_RAG", "Qwen35_397B_RAG")
        for suffix in names:
            script = rag.HERE / f"benchmark_RedKnot_{suffix}.py"
            code = (
                "import runpy,sys; sys.argv=[sys.argv[1]]; "
                "try:\n runpy.run_path(sys.argv[0],run_name='__main__')\n"
                "except SystemExit as e:\n assert e.code==0\n"
                "assert 'torch' not in sys.modules and 'vllm' not in sys.modules "
                "and 'sglang' not in sys.modules"
            )
            # A direct newline before try is required by Python compound grammar.
            code = code.replace("; try:", "\ntry:")
            result = subprocess.run(
                [sys.executable, "-c", code, str(script)],
                cwd=rag.HERE.parent,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["gpu_initialized"])

    def test_qwen_actual_profile_does_not_trust_stale_ratios(self):
        plan = rag.source_policy_plan("qwen3", 16000)
        self.assertEqual(
            plan["source_head_classes"],
            {"local_full": 435, "global": 48, "retrieval": 29},
        )
        self.assertIn("local_full", plan["untranslated_classes"])
        self.assertEqual(plan["head_identity_candidates"], {})

    def test_llama_source_sweetspot_and_adaptive_window_preserved(self):
        plan = rag.source_policy_plan("llama33", 32000)
        self.assertEqual(plan["source_head_classes"], {"local": 576, "global": 64})
        self.assertEqual(plan["source_effective_window"], 4096)
        self.assertFalse(
            plan["source_disabled_fixed_window_fallback"]["active_by_default"]
        )
        self.assertEqual(plan["source_disabled_fixed_window_fallback"]["ratio"], 0.5)
        self.assertEqual(plan["source_ffn_or_linear"]["mass_thresh_deep"], 0.05)

    def test_mistral_uses_swa_not_companion_head_json(self):
        plan = rag.source_policy_plan("mistral", 30000)
        self.assertEqual(plan["source_native_swa_window"], 4096)
        self.assertEqual(plan["source_recompute_ratio"], 0.20)
        self.assertFalse(plan["companion_head_profile_used_by_source_benchmark"])

    def test_qwen35_397b_geometry_not_35b_defaults(self):
        plan = rag.source_policy_plan("qwen35_397b", 32000)
        self.assertEqual(len(plan["full_attention_layers"]), 15)
        self.assertEqual(plan["dense_full_layers"], 9)
        self.assertEqual(plan["sparse_full_layers"], [39, 43, 47, 51, 55, 59])
        self.assertEqual(plan["source_ffn_or_linear"]["moe"]["mass_thresh"], 0.7)
        self.assertEqual(plan["source_ffn_or_linear"]["safety"], 2.0)

    def test_qwen_longest_first_and_cap_are_explicit(self):
        source = [row(0, "a" * 120), row(1, "b" * 220)]
        result = prepare("qwen3", source, samples=1, max_context_tokens=160)
        self.assertEqual(result["cases"][0]["id"], "qwen3:test:row1")
        self.assertEqual(
            result["preparation"][0]["source_cap_or_tail_omitted_tokens"], 60
        )
        self.assertEqual(result["cases"][0]["references"], ["A"])
        self.assertIn(
            "shortest exact answer span", result["preparation"][0]["query_text"]
        )

    def test_llama_source_order_and_short_tail_drop(self):
        result = prepare("llama33", [row(0, "a" * 170), row(1, "b" * 300)], samples=1)
        self.assertEqual(result["cases"][0]["id"], "llama33:test:row0")
        self.assertEqual(
            result["preparation"][0]["source_cap_or_tail_omitted_tokens"], 10
        )
        self.assertEqual(len(result["cases"][0]["chunks"]), 2)

    def test_mistral_middle_cap_even_split_and_instruction_wrap(self):
        result = prepare(
            "mistral",
            [row(0, "a" * 100 + "x" * 100 + "b" * 100)],
            max_context_tokens=200,
            instruction_wrap=True,
        )
        fragments = result["preparation"][0]["document_fragments"]
        self.assertEqual(fragments[0], "[INST] " + "a" * 50)
        self.assertEqual(fragments[-1], "b" * 50)
        self.assertNotIn("x", "".join(fragments))
        self.assertTrue(result["preparation"][0]["query_text"].endswith("[/INST]"))

    def test_qwen35_shuffle_and_distractors_are_deterministic(self):
        source = [row(i, chr(65 + i) * 100) for i in range(6)]
        first = prepare("qwen35_397b", source)
        second = prepare("qwen35_397b", source)
        self.assertEqual(first, second)
        self.assertEqual(len(first["cases"][0]["chunks"]), 4)
        self.assertTrue(first["preparation"][0]["distractor_row_indices"])
        self.assertEqual(
            first["preparation"][0]["source_cap_or_tail_omitted_tokens"], 80
        )

    def test_qwen35_fails_if_corpus_cannot_fill_target(self):
        with self.assertRaisesRegex(ValueError, "no source-profile"):
            prepare("qwen35_397b", [row(0, "a" * 100)])

    def test_explicit_nonprefix_documents_and_gold_survive(self):
        source = [
            {
                "source_row_index": 0,
                "question": "Which?",
                "documents": ["Document B", "Document A"],
                "answers": ["A"],
            }
        ]
        result = prepare("qwen3", source)
        self.assertEqual(decode(result["cases"][0]["chunks"][0]), "Document B\n\n")
        self.assertEqual(result["cases"][0]["references"], ["A"])
        self.assertTrue(result["preparation"][0]["matches_single_text_encoding"])

    def test_preparation_does_not_mutate_input_rows(self):
        source = [row(i, "a" * 200) for i in range(3)]
        before = copy.deepcopy(source)
        prepare("qwen35_397b", source)
        self.assertEqual(source, before)

    def test_more_than_eight_source_chunks_is_prepared_but_not_runnable(self):
        result = prepare("qwen3", [row(0, "a" * 900)], max_context_tokens=900)
        self.assertGreater(len(result["cases"][0]["chunks"]), 8)
        plan = rag.data_preflight(result, 2000, 128)
        self.assertTrue(any("1..8" in value for value in plan["blockers"]))

    def test_model_context_overflow_is_not_silently_capped_at_run(self):
        result = prepare("qwen3", [row(0, "a" * 200)])
        plan = rag.data_preflight(result, 100, 128)
        self.assertTrue(
            any("no runtime truncation" in value for value in plan["blockers"])
        )

    def test_symmetric_metrics_and_unscored_missing_references(self):
        value = rag.score_answer(
            '<think>x</think>Answer: "The Cat."\nMore', ["cat"], "qwen3"
        )
        self.assertEqual(value["f1"], 1.0)
        self.assertEqual(value["em"], 1.0)
        self.assertIsNone(rag.score_answer("A", [], "mistral")["f1"])

    def test_unsupported_backends_fail_before_data_model_or_gpu(self):
        for key in ("mistral", "qwen35_397b"):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as result,
            ):
                rag.main(key, ["--run"])
            self.assertEqual(result.exception.code, 2)

    def test_source_profile_cannot_silently_become_compatible_variant(self):
        for key in ("llama33", "qwen3"):
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                rag.main(key, ["--run"])
            self.assertIn("run-compatible-variant", error.getvalue())

    def test_runtime_rope_and_quantization_preflight_is_real_worker_guard(self):
        policy = validate_config(
            {
                "model_revision": "local-verified",
                "rope_theta": 10000,
                "local_heads": {"0": [0]},
                "allow_approximate": True,
            }
        )
        checkpoint = {
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "hidden_size": 5120,
            "head_dim": 128,
            "rope_theta": 10000,
        }
        self.assertTrue(
            rag.checkpoint_preflight("qwen3", checkpoint, policy)[
                "cpu_checkpoint_contract_passed"
            ]
        )
        for field, value, message in (
            ("rope_scaling", {"type": "yarn"}, "static RoPE"),
            ("quantization_config", {"quant_method": "nf4"}, "unquantized"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                rag.checkpoint_preflight("qwen3", {**checkpoint, field: value}, policy)

    def test_native_llama33_scaled_rope_is_not_disabled_to_make_run_pass(self):
        policy = validate_config(
            {
                "model_revision": "native-llama",
                "rope_theta": 500000,
                "local_heads": {"0": [0]},
                "allow_approximate": True,
            }
        )
        checkpoint = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "hidden_size": 8192,
            "rope_theta": 500000,
            "rope_scaling": {"rope_type": "llama3"},
        }
        with self.assertRaisesRegex(ValueError, "static RoPE"):
            rag.checkpoint_preflight("llama33", checkpoint, policy)

    def test_jsonl_source_row_ids_and_json_array(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.json"
            path.write_text(json.dumps([{"input": "Q"}, {"input": "P"}]))
            self.assertEqual(
                [r["source_row_index"] for r in rag.read_rows(path)], [0, 1]
            )
            path.write_text('{"input":"Q"}\n{"input":"P"}\n')
            self.assertEqual(len(rag.read_rows(path)), 2)


if __name__ == "__main__":
    unittest.main()
