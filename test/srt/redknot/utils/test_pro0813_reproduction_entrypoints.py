"""CPU-only contracts for the formal Pro-0813 reproduction entry points."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
HTTP_ENTRY = HERE / "benchmark_dsv4_pro0813_redknot_http.py"
RAG_DRIVER = HERE / "benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py"
ONE_TARGET = HERE / "run_deepseek_v4_pro0813_reproduction.sh"
SUPERVISOR = HERE / "run_pro0813_combined_supervisor.sh"
ALL_TARGETS = HERE / "run_deepseek_v4_pro0813_all_targets.sh"
PROBE_GUARD = HERE / "run_sm103_probe_with_holder_guard.sh"
SERVER_START = HERE.parents[3] / "server/start_server_redknot_pro0813.sh"
TRITON_FUSED = (
    HERE.parents[3]
    / "python/sglang/srt/layers/attention/nsa/triton_decode/"
    "triton_mla_kernels_decode_fused.py"
)
TRITON_H1_PROBE = HERE / "probe_pro0813_triton_h1_sm103.py"


def _load_http_entry():
    spec = importlib.util.spec_from_file_location(
        "_redknot_pro0813_http_entrypoint_contract_test", HTTP_ENTRY
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HTTP_ENTRY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_functions(
    path: Path,
    names: tuple[str, ...],
    *,
    namespace: dict | None = None,
):
    """Load selected dependency-free functions without importing CUDA code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    if set(functions) != set(names):
        missing = set(names) - set(functions)
        raise AssertionError(f"missing functions in {path}: {missing}")
    namespace = dict(namespace or {})
    selected = [functions[name] for name in names]
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename=str(path),
            mode="exec",
        ),
        namespace,
    )
    return namespace


class TestPro0813B300TritonConfigContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded = _load_source_functions(
            TRITON_FUSED,
            (
                "_prune_head_tile_configs",
                "_sm103_head_tile_prune_enabled",
                "_prune_splitk_configs",
            ),
            namespace={
                "os": os,
                "_SM103_HEAD_TILE_PRUNE_ENV": (
                    "REDKNOT_DSV4_SM103_HEAD_TILE_PRUNE"
                ),
            },
        )
        cls.prune = staticmethod(loaded["_prune_splitk_configs"])
        cls.select_scenarios = staticmethod(
            _load_source_functions(
                TRITON_H1_PROBE,
                ("_select_scenarios",),
            )["_select_scenarios"]
        )

    @staticmethod
    def _configs():
        return [
            SimpleNamespace(kwargs={"BLOCK_H": block_h, "BLOCK_N": block_n})
            for block_h in (16, 64, 128)
            for block_n in (64, 128)
        ]

    def _block_heads(
        self, hardware_profile: str, h_q: int, total_tokens_bucket: int
    ):
        with mock.patch.dict(
            os.environ,
            {
                "REDKNOT_HARDWARE_PROFILE": hardware_profile,
                "REDKNOT_DSV4_SM103_HEAD_TILE_PRUNE": "1",
            },
            clear=False,
        ):
            selected = self.prune(
                self._configs(),
                {"h_q": h_q, "total_tokens_bucket": total_tokens_bucket},
            )
        return sorted({config.kwargs["BLOCK_H"] for config in selected})

    def test_tp_local_h14_keeps_only_b300_safe_head_tile(self):
        self.assertEqual(self._block_heads("b300", 14, 16), [16])

    def test_existing_h64_and_h128_large_batch_pruning_is_preserved(self):
        self.assertEqual(self._block_heads("b300", 64, 16), [64])
        self.assertEqual(self._block_heads("b300", 128, 16), [64, 128])
        self.assertEqual(self._block_heads("b300", 128, 8), [16, 64, 128])

    def test_h200_keeps_the_original_splitk_candidate_set(self):
        self.assertEqual(self._block_heads("h200", 14, 16), [64, 128])
        self.assertEqual(self._block_heads("h200", 64, 16), [64, 128])
        self.assertEqual(self._block_heads("h200", 128, 8), [16, 64, 128])

    def test_probe_scenarios_can_be_selected_independently(self):
        available = tuple(
            SimpleNamespace(name=name)
            for name in ("splitk_h2", "splitk_h14", "c128_h14")
        )
        selected = self.select_scenarios(["c128_h14"], available)
        self.assertEqual([scenario.name for scenario in selected], ["c128_h14"])
        self.assertEqual(self.select_scenarios(None, available), available)
        with self.assertRaises(ValueError):
            self.select_scenarios(["splitk_h14", "splitk_h14"], available)
        with self.assertRaises(ValueError):
            self.select_scenarios(["missing"], available)


class TestPro0813FormalCliDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = _load_http_entry()

    def _performance_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        self.entry._add_performance_qualification_args(parser)
        return parser

    def test_http_entry_defaults_to_formal_qps_sampling(self):
        args = self._performance_parser().parse_args([])
        self.assertTrue(args.measure_qps)
        self.assertEqual(args.qps_concurrencies, "1")
        self.assertEqual(args.qps_warmup_waves, 3)
        self.assertEqual(args.qps_waves, 10)
        self.assertTrue(args.strict_performance)

    def test_diagnostic_performance_is_an_explicit_mutually_exclusive_opt_out(self):
        parser = self._performance_parser()
        args = parser.parse_args(
            ["--diagnostic-performance", "--no-measure-qps"]
        )
        self.assertFalse(args.strict_performance)
        self.assertFalse(args.measure_qps)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["--strict-performance", "--diagnostic-performance"]
                )

    def test_formal_validation_rejects_every_sampling_shortcut(self):
        parser = argparse.ArgumentParser(add_help=False)
        base = {
            "strict_performance": True,
            "ttft_warmup": 3,
            "ttft_iters": 10,
            "measure_qps": True,
            "qps_warmup_waves": 3,
            "qps_waves": 10,
            "quality_repeats": 3,
        }
        self.entry._validate_performance_qualification_args(
            SimpleNamespace(**base), parser
        )
        for field, value in (
            ("ttft_warmup", 2),
            ("ttft_iters", 9),
            ("measure_qps", False),
            ("qps_warmup_waves", 2),
            ("qps_waves", 9),
            ("quality_repeats", 2),
        ):
            with self.subTest(field=field):
                values = dict(base)
                values[field] = value
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.entry._validate_performance_qualification_args(
                            SimpleNamespace(**values), parser
                        )

    def test_explicit_diagnostic_mode_allows_short_sampling(self):
        parser = argparse.ArgumentParser(add_help=False)
        self.entry._validate_performance_qualification_args(
            SimpleNamespace(
                strict_performance=False,
                ttft_warmup=0,
                ttft_iters=1,
                measure_qps=False,
                qps_warmup_waves=0,
                qps_waves=1,
                quality_repeats=1,
            ),
            parser,
        )

    def test_standalone_canonical_cli_defaults_to_three_quality_repeats(self):
        source = HTTP_ENTRY.read_text(encoding="utf-8")
        self.assertIn(
            'parser.add_argument("--quality-repeats", type=int, default=3)',
            source,
        )
        self.assertIn("or args.quality_repeats < 3", source)

    def test_python_source_pin_discards_materialized_ambient_entries(self):
        original_path = list(sys.path)
        with tempfile.TemporaryDirectory() as directory:
            poisoned = Path(directory).resolve()
            legacy_flash = Path("/workspace/RedKnot/python").resolve()
            try:
                with mock.patch.dict(
                    os.environ,
                    {
                        "PYTHONPATH": (
                            f"{poisoned}{os.pathsep}{legacy_flash}"
                        ),
                        "REDKNOT_FLASHMLA_SM103_ROOT": (
                            "/data/temp/FlashMLA-sm103-src"
                        ),
                    },
                    clear=False,
                ):
                    sys.path[:] = [str(poisoned), str(legacy_flash), *original_path]
                    pro_root, flash_root = (
                        self.entry._pin_pro0813_python_sources()
                    )
                    self.assertEqual(
                        sys.path[:2], [str(pro_root), str(flash_root)]
                    )
                    materialized = {
                        Path(item or os.getcwd()).expanduser().resolve()
                        for item in sys.path
                    }
                    self.assertNotIn(poisoned, materialized)
                    if legacy_flash != pro_root:
                        self.assertNotIn(legacy_flash, materialized)
                    self.assertEqual(
                        os.environ["PYTHONPATH"],
                        f"{pro_root}{os.pathsep}{flash_root}",
                    )
                    self.assertEqual(os.environ["PYTHONNOUSERSITE"], "1")
                    self.assertEqual(os.environ["PYTHONSAFEPATH"], "1")
            finally:
                sys.path[:] = original_path

    def test_formal_short_targets_reject_arbitrary_qualification_profiles(self):
        for target_tokens in (65536, 131072, 262144):
            with self.subTest(target_tokens=target_tokens):
                with self.assertRaisesRegex(ValueError, "built-in frozen"):
                    self.entry._validate_qualification_profile_target_policy(
                        SimpleNamespace(
                            strict_performance=True,
                            target_tokens=target_tokens,
                            qualification_profile="/tmp/arbitrary.json",
                        )
                    )
        for target_tokens in (450560, 524288):
            self.entry._validate_qualification_profile_target_policy(
                SimpleNamespace(
                    strict_performance=True,
                    target_tokens=target_tokens,
                    qualification_profile="/tmp/frozen.json",
                )
            )
        self.entry._validate_qualification_profile_target_policy(
            SimpleNamespace(
                strict_performance=False,
                target_tokens=65536,
                qualification_profile="/tmp/diagnostic.json",
            )
        )

    def test_formal_redknot_requires_full_combined_execution(self):
        self.entry._validate_formal_execution_profile_args(
            SimpleNamespace(
                mode="baseline",
                strict_performance=True,
                combined_headsplit_row_sparse=False,
            )
        )
        self.entry._validate_formal_execution_profile_args(
            SimpleNamespace(
                mode="redknot",
                strict_performance=True,
                combined_headsplit_row_sparse=True,
            )
        )
        for label, values in (
            (
                "pure",
                {
                    "mode": "redknot",
                    "strict_performance": True,
                    "combined_headsplit_row_sparse": False,
                },
            ),
            (
                "standalone_row_sparse",
                {
                    "mode": "redknot",
                    "strict_performance": True,
                    "combined_headsplit_row_sparse": False,
                },
            ),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "full combined"
            ):
                self.entry._validate_formal_execution_profile_args(
                    SimpleNamespace(**values)
                )
        self.entry._validate_formal_execution_profile_args(
            SimpleNamespace(
                mode="redknot",
                strict_performance=False,
                combined_headsplit_row_sparse=False,
            )
        )

    def test_builtin_short_profile_has_stable_provenance_identity(self):
        args = SimpleNamespace(
            target_tokens=65536,
            qualification_profile="",
        )
        first = self.entry._resolve_redknot_target_profile(args)
        second = self.entry._resolve_redknot_target_profile(args)
        self.assertEqual(
            first["qualification_profile_path"], "builtin:pro0813:65536"
        )
        self.assertRegex(
            first["qualification_profile_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            first["qualification_profile_sha256"],
            second["qualification_profile_sha256"],
        )

    def _fake_long_profile_document(self):
        digest = "1" * 64
        return {
            "format": "redknot_multidataset_profile_v2",
            "target_tokens": 450560,
            "num_chunks": 8,
            "chunk_tokens": 56320,
            "query_start": 450560,
            "dataset": "hotpotqa",
            "num_queries": 1,
            "query_row_ids": [7],
            "cases": [
                {
                    "text_sha256": f"sha256:{digest}",
                    "full_input_ids_sha256": f"sha256:{digest}",
                }
            ],
            "long_output_target_tokens": 0,
            "data_manifest": "data_selection.json",
            "data_selection_sha256": digest,
            "prompt_manifest": "prompt_manifest.json",
            "prompt_manifest_sha256": digest,
            "max_total_tokens": 450561,
            "intended_execution_profile": "full_combined_production_v1",
            "diagnostic_execution_profiles": ["zoff_only"],
        }

    def _resolve_with_fake_verifier(self, source, verification):
        fake_module = SimpleNamespace(
            verify_profile=lambda *args, **kwargs: verification()
        )
        fake_spec = SimpleNamespace(
            loader=SimpleNamespace(exec_module=lambda module: None)
        )
        args = SimpleNamespace(
            target_tokens=450560,
            qualification_profile=str(source),
        )
        with mock.patch.object(
            self.entry.importlib.util,
            "spec_from_file_location",
            return_value=fake_spec,
        ), mock.patch.object(
            self.entry.importlib.util,
            "module_from_spec",
            return_value=fake_module,
        ):
            return self.entry._resolve_redknot_target_profile(args)

    def test_consumer_uses_verified_bytes_after_profile_path_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "profile.json"
            document = self._fake_long_profile_document()
            verified_bytes = json.dumps(document, sort_keys=True).encode("utf-8")
            source.write_bytes(verified_bytes)
            digest = hashlib.sha256(verified_bytes).hexdigest()

            def replace_then_return():
                tampered = dict(document)
                tampered["intended_execution_profile"] = "tampered_after_verify"
                source.write_text(json.dumps(tampered), encoding="utf-8")
                return {
                    "profile": str(source),
                    "profile_sha256": digest,
                    "_verified_profile_bytes": verified_bytes,
                }

            resolved = self._resolve_with_fake_verifier(
                source, replace_then_return
            )
            self.assertEqual(
                resolved["intended_execution_profile"],
                "full_combined_production_v1",
            )
            self.assertEqual(resolved["qualification_profile_sha256"], digest)
            self.assertEqual(
                resolved["qualification_profile_path"], str(source)
            )
            self.assertEqual(
                json.loads(source.read_text(encoding="utf-8"))[
                    "intended_execution_profile"
                ],
                "tampered_after_verify",
            )

    def test_consumer_rejects_verifier_bytes_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "profile.json"
            verified_bytes = json.dumps(
                self._fake_long_profile_document(), sort_keys=True
            ).encode("utf-8")
            source.write_bytes(verified_bytes)

            def mismatched_verification():
                return {
                    "profile": str(source),
                    "profile_sha256": "0" * 64,
                    "_verified_profile_bytes": verified_bytes,
                }

            with self.assertRaisesRegex(RuntimeError, "different SHA-256"):
                self._resolve_with_fake_verifier(
                    source, mismatched_verification
                )


class TestPro0813ShellEntrypointContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.one_target = ONE_TARGET.read_text(encoding="utf-8")
        cls.supervisor = SUPERVISOR.read_text(encoding="utf-8")
        cls.all_targets = ALL_TARGETS.read_text(encoding="utf-8")
        cls.http_entry = HTTP_ENTRY.read_text(encoding="utf-8")
        cls.rag_driver = RAG_DRIVER.read_text(encoding="utf-8")
        cls.server_start = SERVER_START.read_text(encoding="utf-8")
        cls.probe_guard = PROBE_GUARD.read_text(encoding="utf-8")

    def test_shell_files_parse(self):
        subprocess.run(
            [
                "bash",
                "-n",
                str(ONE_TARGET),
                str(SUPERVISOR),
                str(ALL_TARGETS),
                str(PROBE_GUARD),
                str(SERVER_START),
            ],
            check=True,
        )

    def test_server_start_is_fail_closed_for_b300_sm103_kernels(self):
        self.assertIn("SGLANG_USE_JIT_RMSNORM=1", self.server_start)
        self.assertIn("FLASHINFER_CUDA_ARCH_LIST=10.3a", self.server_start)
        self.assertIn(
            "NVRTC_LIB=/usr/local/cuda/lib64/libnvrtc.so.12",
            self.server_start,
        )
        self.assertIn(
            'LD_PRELOAD="$NVRTC_LIB:$NVJITLINK_LIB', self.server_start
        )
        self.assertIn(
            "test/srt/redknot/utils/probe_pro0813_jit_rmsnorm_sm103.py",
            self.server_start,
        )
        self.assertIn(
            "test/srt/redknot/utils/probe_pro0813_triton_h1_sm103.py",
            self.server_start,
        )
        self.assertIn(
            "REDKNOT_PRO0813_B300_TRITON_H1_PROBE=1", self.server_start
        )
        self.assertIn(
            "requires exactly 8x NVIDIA B300 compute_cap=10.3",
            self.server_start,
        )
        self.assertLess(
            self.server_start.index("verify_pro0813_official_model.py"),
            self.server_start.index("nvidia-smi"),
        )

    def test_formal_process_tree_replaces_ambient_pythonpath(self):
        expected_server = (
            'export PYTHONPATH="$REDKNOT_ROOT/python:$FLASHMLA_SM103_ROOT"'
        )
        expected_runner = (
            'export PYTHONPATH="$repo/python:$flashmla_sm103_root"'
        )
        self.assertIn(expected_server, self.server_start)
        self.assertIn(expected_runner, self.one_target)
        self.assertIn(expected_runner, self.supervisor)
        self.assertIn(
            'PYTHONPATH="$repo/python:$flashmla_sm103_root"',
            self.supervisor,
        )
        self.assertIn(
            "PYTHONPATH=$repo/python:/data/temp/FlashMLA-sm103-src",
            self.all_targets,
        )
        for label, source in (
            ("server", self.server_start),
            ("one-target", self.one_target),
            ("supervisor", self.supervisor),
            ("all-targets", self.all_targets),
        ):
            with self.subTest(label=label):
                self.assertNotIn("${PYTHONPATH:+:$PYTHONPATH}", source)
                self.assertIn("PYTHONNOUSERSITE=1", source)
                self.assertIn("PYTHONSAFEPATH=1", source)

    def test_rag_driver_pins_before_torch_and_checks_sglang_origin(self):
        pin_call = (
            "PRO0813_PYTHON_ROOT, PRO0813_FLASHMLA_ROOT = "
            "_pin_pro0813_python_sources()"
        )
        self.assertIn(pin_call, self.rag_driver)
        self.assertLess(
            self.rag_driver.index(pin_call),
            self.rag_driver.index("import torch"),
        )
        self.assertIn(
            "_COMMIT_SOURCE.is_relative_to(PRO0813_PYTHON_ROOT / \"sglang\")",
            self.rag_driver,
        )

    def test_standalone_holder_guard_runs_both_b300_startup_oracles(self):
        self.assertIn("SGLANG_USE_JIT_RMSNORM=1", self.probe_guard)
        self.assertIn("FLASHINFER_CUDA_ARCH_LIST=10.3a", self.probe_guard)
        self.assertIn(
            "NVRTC_LIB=/usr/local/cuda/lib64/libnvrtc.so.12",
            self.probe_guard,
        )
        self.assertIn(
            'LD_PRELOAD="$NVRTC_LIB:$NVJITLINK_LIB', self.probe_guard
        )
        self.assertIn(
            "HOLDER_DIR=/workspace/RedKnot/test/srt/redknot/utils",
            self.probe_guard,
        )
        for probe in (
            "probe_pro0813_jit_rmsnorm_sm103.py",
            "probe_pro0813_triton_h1_sm103.py",
            "probe_pro0813_groups2_sm103.py",
            "probe_flashmla_pro0813_sm103.py",
            "probe_pro0813_shared_latent_batch_sm103.py",
            "probe_pro0813_rope_reloc_sm103.py",
        ):
            self.assertIn(
                f"test/srt/redknot/utils/{probe}", self.probe_guard
            )
        self.assertIn(
            "--expected-source-root /workspace/RedKnot/python",
            self.probe_guard,
        )

    def test_wrapper_has_formal_defaults_and_target_derived_names(self):
        self.assertIn('${REDKNOT_QPS_WARMUP_WAVES:-3}', self.one_target)
        self.assertIn('${REDKNOT_QPS_WAVES:-10}', self.one_target)
        self.assertIn('${REDKNOT_QPS_CONCURRENCIES:-1}', self.one_target)
        self.assertRegex(self.one_target, r"65536\) target_label=64k")
        self.assertRegex(self.one_target, r"131072\) target_label=128k")
        self.assertRegex(self.one_target, r"262144\) target_label=256k")
        self.assertRegex(self.one_target, r"450560\) target_label=440k")
        self.assertRegex(self.one_target, r"524288\) target_label=512k")
        self.assertIn('pro0813-${target_label}-$(date +%Y%m%d-%H%M%S)', self.one_target)
        self.assertLess(
            self.one_target.index(
                "validating frozen Pro-0813 target assets before model gate"
            ),
            self.one_target.index(
                "validating complete official Pro-0813 model before holder lookup"
            ),
        )
        self.assertLess(
            self.one_target.index(
                "validating complete official Pro-0813 model before holder lookup"
            ),
            self.one_target.index("pgrep -f '[g]pu_hold.py'"),
        )

    def test_supervisor_explicitly_binds_formal_or_diagnostic_mode(self):
        self.assertRegex(self.supervisor, r"ttft_warmup=\$\{4:-3\}")
        self.assertRegex(self.supervisor, r"ttft_iters=\$\{5:-10\}")
        self.assertRegex(self.supervisor, r"quality_repeats=\$\{19:-3\}")
        self.assertIn(
            "formal performance requires QUALITY_REPEATS >= 3",
            self.supervisor,
        )
        self.assertRegex(self.supervisor, r"qps_warmup_waves=\$\{13:-3\}")
        self.assertRegex(self.supervisor, r"qps_waves=\$\{14:-10\}")
        self.assertRegex(self.supervisor, r"qps_concurrencies=\$\{11:-1\}")
        self.assertIn("benchmark_extra_args+=(--strict-performance)", self.supervisor)
        self.assertIn("benchmark_extra_args+=(--diagnostic-performance)", self.supervisor)
        self.assertIn(
            "--combined-headsplit-row-sparse-diagnostic-zoff-only",
            self.supervisor,
        )
        self.assertIn(
            "diagnostic_zoff_only=${REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY:-0}",
            self.supervisor,
        )
        self.assertIn(
            'if [[ "$diagnostic_zoff_only" == 1 ]]; then',
            self.supervisor,
        )
        self.assertNotIn(
            'if [[ "$diagnostic_performance" == 1 ]]; then\n'
            "    benchmark_extra_args+=(\n"
            "      --combined-headsplit-row-sparse-diagnostic-zoff-only",
            self.supervisor,
        )
        self.assertIn("benchmark_extra_args+=(--no-measure-qps)", self.supervisor)
        self.assertIn("REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE", self.supervisor)
        self.assertIn(
            "REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256",
            self.supervisor,
        )
        self.assertLess(
            self.supervisor.index(
                "validating frozen Pro-0813 qualification profile before model gate"
            ),
            self.supervisor.index(
                'model_gate="$script_dir/verify_pro0813_official_model.py"'
            ),
        )
        self.assertLess(
            self.supervisor.index('model_gate="$script_dir/verify_pro0813_official_model.py"'),
            self.supervisor.index("nvidia-smi"),
        )

    def test_supervisor_gpu_handoff_is_fail_closed_and_physically_bound(self):
        self.assertIn("GPU_QUERY_ATTEMPTS=3", self.supervisor)
        self.assertIn(
            "--query-compute-apps=gpu_uuid,pid", self.supervisor
        )
        self.assertIn(
            "--query-gpu=index,uuid,utilization.gpu", self.supervisor
        )
        self.assertEqual(
            len(re.findall(r"active_gpu_pids=\$\(gpu_pids\)", self.supervisor)),
            3,
        )
        self.assertEqual(
            len(
                re.findall(
                    r"if ! active_gpu_pids=\$\(gpu_pids\); then",
                    self.supervisor,
                )
            ),
            3,
        )
        self.assertIn("HOLDER_UTIL_MIN_PERCENT=90", self.supervisor)
        self.assertIn("HOLDER_UTIL_REQUIRED_SAMPLES=3", self.supervisor)
        self.assertIn("HOLDER_UTIL_MAX_SAMPLES=15", self.supervisor)
        self.assertIn("verify_holder_process_identity", self.supervisor)
        self.assertIn("verify_gpu_group_coverage_once", self.supervisor)
        self.assertIn("verify_worker_gpu_coverage_once", self.supervisor)
        self.assertIn(
            "bootstrap holder identity failed during worker barrier",
            self.supervisor,
        )
        self.assertIn(
            "bootstrap holder GPU coverage failed during worker barrier",
            self.supervisor,
        )
        self.assertIn(
            'ln "$worker_go_tmp" "$worker_go_file"', self.supervisor
        )
        self.assertIn(
            '"$worker_go_identity" != "$worker_go_tmp_identity"',
            self.supervisor,
        )
        self.assertIn(
            '"$worker_go_record" != release_workers=8', self.supervisor
        )
        self.assertIn(
            "full holder only after it can prove no residual GPU process",
            self.supervisor,
        )

    def test_formal_and_diagnostic_combined_paths_cannot_be_confused(self):
        self.assertIn('"--combined-headsplit-row-sparse"', self.http_entry)
        self.assertIn(
            '"--combined-headsplit-row-sparse-diagnostic-zoff-only"',
            self.http_entry,
        )
        self.assertIn(
            "--combined-headsplit-row-sparse-diagnostic-zoff-only requires",
            self.http_entry,
        )
        self.assertIn("PRO0813_COMBINED_FULL_PROFILE", self.http_entry)
        self.assertIn("allow_diagnostic=not IH_STRICT_PERFORMANCE_CLAIMS", self.rag_driver)
        self.assertIn(
            "the all-target entry point cannot run a zoff-only diagnostic",
            self.all_targets,
        )
        self.assertEqual(
            self.http_entry.count("_resolve_redknot_target_profile(args)"),
            2,
            "profile resolution must occur only in its definition and once in main",
        )

    def test_all_target_sequence_is_exact_and_profiles_are_target_bound(self):
        tokens = re.search(
            r"targets=\(([^)]*)\)", self.all_targets, flags=re.MULTILINE
        )
        labels = re.search(
            r"labels=\(([^)]*)\)", self.all_targets, flags=re.MULTILINE
        )
        self.assertIsNotNone(tokens)
        self.assertIsNotNone(labels)
        self.assertEqual(
            tokens.group(1).split(),
            ["65536", "131072", "262144", "450560", "524288"],
        )
        self.assertEqual(
            labels.group(1).split(), ["64k", "128k", "256k", "440k", "512k"]
        )
        self.assertIn("REDKNOT_QUALIFICATION_PROFILE_440K", self.all_targets)
        self.assertIn("REDKNOT_QUALIFICATION_PROFILE_512K", self.all_targets)
        self.assertIn("pro0813_440k_hotpotqa_10q/profile.json", self.all_targets)
        self.assertIn("pro0813_512k_hotpotqa_10q/profile.json", self.all_targets)
        self.assertNotRegex(
            self.all_targets,
            r"qualification_profile_440k=\$\{REDKNOT_QUALIFICATION_PROFILE_440K",
        )
        self.assertNotRegex(
            self.all_targets,
            r"qualification_profile_512k=\$\{REDKNOT_QUALIFICATION_PROFILE_512K",
        )
        self.assertIn("formal_unset_args=(", self.all_targets)
        self.assertIn("-u REDKNOT_QUALIFICATION_PROFILE", self.all_targets)
        self.assertIn("REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=0", self.all_targets)
        self.assertIn(
            "REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256",
            self.all_targets,
        )
        for expected_hash in (
            "52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b",
            "e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d",
        ):
            self.assertIn(expected_hash, self.all_targets)
        self.assertIn("verify_pro0813_formal_assets.py", self.all_targets)
        self.assertIn("formal_assets.preflight.json", self.all_targets)
        canonical_formal_environment = {
            "REDKNOT_TIMING": "1",
            "REDKNOT_TTFT_WARMUP": "3",
            "REDKNOT_TTFT_ITERS": "10",
            "REDKNOT_QPS_CONCURRENCIES": "1",
            "REDKNOT_QPS_WARMUP_WAVES": "3",
            "REDKNOT_QPS_WAVES": "10",
            "REDKNOT_QUALITY_REPEATS": "3",
            "REDKNOT_ROW_SPARSE_ACTIVE_RATIO": "0.20",
            "REDKNOT_GENERALIZED_ADAPTIVE_CONTROLLER": "0",
            "REDKNOT_COMBINED_HEADSPLIT_ROW_SPARSE": "1",
            "REDKNOT_RELEASE_ADAPTIVE_TOPK_MASS": "0.50",
            "REDKNOT_RELEASE_ADAPTIVE_TOPK_BUCKETS": "3,4,5,6",
            "REDKNOT_MOE_RUNNER_BACKEND": "flashinfer_mxfp4",
            "SGLANG_OPT_USE_TILELANG_MHC_PRE": "0",
            "SGLANG_OPT_USE_TILELANG_MHC_POST": "0",
            "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
            "REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH": "0",
            "SGLANG_USE_JIT_RMSNORM": "1",
            "FLASHINFER_CUDA_ARCH_LIST": "10.3a",
        }
        for name, value in canonical_formal_environment.items():
            with self.subTest(formal_environment=name):
                self.assertIn(f"{name}={value}", self.all_targets)
        for unset_name in (
            "REDKNOT_HEAD_CFG",
            "REDKNOT_SWA_FULL_TOKENS_RATIO",
            "REDKNOT_SERVER_POLICY_MANIFEST_OUT",
            "REDKNOT_SERVER_INSTANCE_NONCE",
        ):
            self.assertIn(f"-u {unset_name}", self.all_targets)
        self.assertIn("profile_sha256", self.all_targets)
        self.assertLess(
            self.all_targets.index("prevalidating all five selection/dataset/profile"),
            self.all_targets.index('mkdir -p "$root_run_dir"'),
        )
        self.assertNotIn("release_440k_15case", self.all_targets)

    def test_all_target_sequence_invokes_one_runner_per_target_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_dir = root / "redknot"
            script_dir.mkdir()
            sequence = script_dir / ALL_TARGETS.name
            sequence.write_text(
                ALL_TARGETS.read_text(encoding="utf-8").replace(
                    "runtime_python=/workspace/RedKnot/.venv_sm103/bin/python",
                    f"runtime_python={shlex.quote(sys.executable)}",
                ),
                encoding="utf-8",
            )
            sequence.chmod(0o755)
            verifier = script_dir / "verify_pro0813_qualification_profile.py"
            verifier.write_text(
                "import argparse, json\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('profile')\n"
                "p.add_argument('--expected-target-tokens', type=int, required=True)\n"
                "p.add_argument('--expected-profile-sha256', required=True)\n"
                "a = p.parse_args()\n"
                "print(json.dumps({'pass': True, 'profile': a.profile, "
                "'profile_sha256': a.expected_profile_sha256, "
                "'target_tokens': a.expected_target_tokens}))\n",
                encoding="utf-8",
            )
            formal_asset_verifier = (
                script_dir / "verify_pro0813_formal_assets.py"
            )
            formal_asset_verifier.write_text(
                "import json\n"
                "print(json.dumps({'format': "
                "'redknot_pro0813_formal_asset_preflight_v1', "
                "'pass': True, 'assets': ["
                "{'target_tokens': 450560, 'profile_sha256': "
                "'52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b'},"
                "{'target_tokens': 524288, 'profile_sha256': "
                "'e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d'}"
                "]}))\n",
                encoding="utf-8",
            )
            summary_writer = script_dir / "write_pro0813_all_targets_summary.py"
            summary_writer.write_text(
                "import json\n"
                "print(json.dumps({'overall_exit_code': 0}))\n",
                encoding="utf-8",
            )
            runner = script_dir / ONE_TARGET.name
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "[[ \"$REDKNOT_TIMING\" == 1 ]]\n"
                "[[ \"$REDKNOT_TTFT_WARMUP\" == 3 ]]\n"
                "[[ \"$REDKNOT_TTFT_ITERS\" == 10 ]]\n"
                "[[ \"$REDKNOT_QPS_CONCURRENCIES\" == 1 ]]\n"
                "[[ \"$REDKNOT_QPS_WARMUP_WAVES\" == 3 ]]\n"
                "[[ \"$REDKNOT_QPS_WAVES\" == 10 ]]\n"
                "[[ \"$REDKNOT_QUALITY_REPEATS\" == 3 ]]\n"
                "[[ \"$REDKNOT_ROW_SPARSE_ACTIVE_RATIO\" == 0.20 ]]\n"
                "[[ \"$REDKNOT_GENERALIZED_ADAPTIVE_CONTROLLER\" == 0 ]]\n"
                "[[ \"$REDKNOT_COMBINED_HEADSPLIT_ROW_SPARSE\" == 1 ]]\n"
                "[[ \"$REDKNOT_RELEASE_ADAPTIVE_TOPK_MASS\" == 0.50 ]]\n"
                "[[ \"$REDKNOT_RELEASE_ADAPTIVE_TOPK_BUCKETS\" == 3,4,5,6 ]]\n"
                "[[ \"$REDKNOT_MOE_RUNNER_BACKEND\" == flashinfer_mxfp4 ]]\n"
                "[[ \"$SGLANG_OPT_USE_TILELANG_MHC_PRE\" == 0 ]]\n"
                "[[ \"$SGLANG_OPT_USE_TILELANG_MHC_POST\" == 0 ]]\n"
                "[[ \"$SGLANG_OPT_DEEPGEMM_HC_PRENORM\" == 0 ]]\n"
                "[[ \"$REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH\" == 0 ]]\n"
                "[[ \"$SGLANG_USE_JIT_RMSNORM\" == 1 ]]\n"
                "[[ \"$FLASHINFER_CUDA_ARCH_LIST\" == 10.3a ]]\n"
                "[[ -z \"${REDKNOT_HEAD_CFG+x}\" ]]\n"
                "[[ -z \"${REDKNOT_SWA_FULL_TOKENS_RATIO+x}\" ]]\n"
                "[[ -z \"${REDKNOT_SERVER_POLICY_MANIFEST_OUT+x}\" ]]\n"
                "[[ -z \"${REDKNOT_SERVER_INSTANCE_NONCE+x}\" ]]\n"
                "printf '%s|%s|%s|%s|%s\\n' \"$1\" \"$2\" "
                '"${REDKNOT_QUALIFICATION_PROFILE:-}" '
                '"${REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256:-}" '
                '"${REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE:-}" '
                '>> "$FAKE_RUN_LOG"\n',
                encoding="utf-8",
            )
            runner.chmod(0o755)
            profile_440k = (
                script_dir.parent
                / "qualification_profiles/pro0813_440k_hotpotqa_10q/profile.json"
            )
            profile_512k = (
                script_dir.parent
                / "qualification_profiles/pro0813_512k_hotpotqa_10q/profile.json"
            )
            profile_440k.parent.mkdir(parents=True)
            profile_512k.parent.mkdir(parents=True)
            profile_440k.write_text("{}\n", encoding="utf-8")
            profile_512k.write_text("{}\n", encoding="utf-8")
            run_root = root / "results"
            run_log = root / "runs.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "FAKE_RUN_LOG": str(run_log),
                    "REDKNOT_QUALIFICATION_PROFILE": "/must/be/unset",
                    "REDKNOT_TIMING": "0",
                    "REDKNOT_TTFT_WARMUP": "1",
                    "REDKNOT_TTFT_ITERS": "2",
                    "REDKNOT_QPS_CONCURRENCIES": "8",
                    "REDKNOT_QPS_WARMUP_WAVES": "1",
                    "REDKNOT_QPS_WAVES": "2",
                    "REDKNOT_QUALITY_REPEATS": "1",
                    "REDKNOT_ROW_SPARSE_ACTIVE_RATIO": "0.84",
                    "REDKNOT_GENERALIZED_ADAPTIVE_CONTROLLER": "1",
                    "REDKNOT_COMBINED_HEADSPLIT_ROW_SPARSE": "0",
                    "REDKNOT_RELEASE_ADAPTIVE_TOPK_MASS": "0.10",
                    "REDKNOT_RELEASE_ADAPTIVE_TOPK_BUCKETS": "1,2",
                    "REDKNOT_MOE_RUNNER_BACKEND": "polluted_backend",
                    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "1",
                    "SGLANG_OPT_USE_TILELANG_MHC_POST": "1",
                    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "1",
                    "REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH": "1",
                    "SGLANG_USE_JIT_RMSNORM": "0",
                    "FLASHINFER_CUDA_ARCH_LIST": "10.3",
                    "REDKNOT_HEAD_CFG": "/tmp/polluted-head-cfg.json",
                    "REDKNOT_SWA_FULL_TOKENS_RATIO": "0.99",
                    "REDKNOT_SERVER_POLICY_MANIFEST_OUT": "/tmp/polluted.json",
                    "REDKNOT_SERVER_INSTANCE_NONCE": "polluted",
                }
            )
            subprocess.run(
                ["bash", str(sequence), str(run_root)],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                run_log.read_text(encoding="utf-8").splitlines(),
                [
                    f"{run_root / '64k'}|65536|||0",
                    f"{run_root / '128k'}|131072|||0",
                    f"{run_root / '256k'}|262144|||0",
                    f"{run_root / '440k'}|450560|{profile_440k.resolve()}|"
                    "52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b|0",
                    f"{run_root / '512k'}|524288|{profile_512k.resolve()}|"
                    "e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d|0",
                ],
            )
            plan_lines = (run_root / "sequence.plan.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(plan_lines), 6)
            self.assertEqual(plan_lines[1].split("\t")[3], str(run_root / "64k"))
            self.assertEqual(plan_lines[5].split("\t")[3], str(run_root / "512k"))
            self.assertEqual(
                plan_lines[4].split("\t")[5],
                "52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b",
            )
            self.assertEqual(
                plan_lines[5].split("\t")[5],
                "e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d",
            )


if __name__ == "__main__":
    unittest.main()
