"""CPU-only CLI contracts for the opt-in native DSV4 projection micro-check."""

import builtins
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[1] / "benchmarks/check_dsv4_projection_kernel.py"
)
GPU_PACKAGES = {"torch", "triton", "vllm"}


@contextmanager
def forbid_gpu_imports():
    """Catch imports even when another test already imported a GPU package."""
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    attempts = []

    def check(name):
        if name.split(".", 1)[0] in GPU_PACKAGES:
            attempts.append(name)
            raise AssertionError(f"unexpected GPU dependency import: {name}")

    def guarded_import(name, *args, **kwargs):
        check(name)
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        check(name)
        return original_import_module(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=guarded_import),
        patch("importlib.import_module", side_effect=guarded_import_module),
    ):
        yield attempts


class ProjectionKernelCLITests(unittest.TestCase):
    def setUp(self):
        name = f"_redknot_projection_kernel_cli_{id(self)}"
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        self.benchmark = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.benchmark
        self.addCleanup(sys.modules.pop, name, None)
        with forbid_gpu_imports() as attempts:
            spec.loader.exec_module(self.benchmark)
        self.assertEqual(attempts, [])

    def assert_rejected(self, argv):
        output, errors = io.StringIO(), io.StringIO()
        with (
            forbid_gpu_imports() as attempts,
            patch.object(self.benchmark, "run") as run,
            redirect_stdout(output),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as caught,
        ):
            self.benchmark.main(argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("error:", errors.getvalue())
        self.assertEqual(attempts, [])
        run.assert_not_called()

    def mocked_main(self, argv, *, report=None, error=None):
        output, errors = io.StringIO(), io.StringIO()
        with (
            forbid_gpu_imports() as attempts,
            patch.object(
                self.benchmark, "run", return_value=report, side_effect=error
            ) as run,
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = self.benchmark.main(argv)
        self.assertEqual(attempts, [])
        self.assertEqual(errors.getvalue(), "")
        run.assert_called_once()
        return result, json.loads(output.getvalue()), run.call_args.args[0]

    def test_import_in_fresh_process_is_silent_without_gpu_dependencies(self):
        code = """
import importlib.util
import sys

class RejectGPUImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'triton', 'vllm'}:
            raise AssertionError('unexpected GPU dependency import: ' + fullname)

sys.meta_path.insert(0, RejectGPUImports())
name = '_redknot_projection_kernel_isolated_import'
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
assert not {'torch', 'triton', 'vllm'} & sys.modules.keys()
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(SCRIPT)],
            cwd=SCRIPT.parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_help_exits_without_importing_gpu_dependencies_or_running(self):
        output = io.StringIO()
        with (
            forbid_gpu_imports() as attempts,
            patch.object(self.benchmark, "run") as run,
            redirect_stdout(output),
            self.assertRaises(SystemExit) as caught,
        ):
            self.benchmark.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--run", output.getvalue())
        self.assertIn("--device", output.getvalue())
        self.assertIn("--local-head-ids", output.getvalue())
        self.assertEqual(attempts, [])
        run.assert_not_called()

    def test_gpu_execution_requires_both_opt_in_and_explicit_device(self):
        for argv in ([], ["--run"], ["--device", "cuda:0"]):
            with self.subTest(argv=argv):
                self.assert_rejected(argv)

    def test_device_must_be_an_exact_nonnegative_cuda_index(self):
        for device in (
            "cpu",
            "cuda",
            "cuda:",
            "cuda:-1",
            "cuda:+1",
            "cuda:one",
            "cuda:1.0",
            "cuda:0x0",
            "cuda:0junk",
            " cuda:0",
            "cuda:0 ",
            "cuda:0\n",
        ):
            with self.subTest(device=device):
                self.assert_rejected(["--run", "--device", device])

    def test_workload_counts_and_rank_are_checked_before_gpu_import(self):
        invalid = {
            "--tokens": ("-1", "0", "3", "4097", "1.5"),
            "--rank": ("0", "64", "129", "512", "1025"),
            "--warmup": ("-1", "0", "1.5"),
            "--iterations": ("-1", "0", "1.5"),
            "--seed": ("-1", "1.5"),
        }
        for option, values in invalid.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    self.assert_rejected(
                        ["--run", "--device", "cuda:0", f"{option}={value}"]
                    )

    def test_tolerances_must_be_positive_and_finite_before_gpu_import(self):
        for option in ("--relative-rms", "--max-absolute"):
            for value in ("0", "-0.1", "nan", "inf", "-inf", "1e999"):
                with self.subTest(option=option, value=value):
                    self.assert_rejected(
                        ["--run", "--device", "cuda:0", f"{option}={value}"]
                    )

    def test_local_heads_are_nonempty_unique_integer_ids_in_range(self):
        for value in ("", "1,1", "-1", "64", "0,64", "1,,2", "1,", "1.5", "one"):
            with self.subTest(value=value):
                self.assert_rejected(
                    ["--run", "--device", "cuda:0", f"--local-head-ids={value}"]
                )

    def test_parser_defaults_do_not_authorize_gpu_execution(self):
        with forbid_gpu_imports() as attempts:
            args = self.benchmark._parser().parse_args([])
        self.assertEqual(attempts, [])
        self.assertFalse(args.run)
        self.assertIsNone(args.device)
        self.assertEqual(args.tokens, 64)
        self.assertEqual(args.rank, 128)
        self.assertEqual(args.warmup, 3)
        self.assertEqual(args.iterations, 10)
        self.assertIsInstance(args.seed, int)
        self.assertGreaterEqual(args.seed, 0)
        self.assertEqual(args.relative_rms, 0.02)
        self.assertEqual(args.max_absolute, 0.05)
        self.assertEqual(
            tuple(args.local_head_ids), tuple(head for head in range(64) if head % 8)
        )

    def test_explicit_run_returns_success_json_with_default_options(self):
        report = {"passed": True, "scope": "mock native projection check"}
        result, actual, args = self.mocked_main(
            ["--run", "--device", "cuda:0"], report=report
        )
        self.assertEqual(result, 0)
        self.assertEqual(actual, report)
        self.assertTrue(args.run)
        self.assertEqual(args.device, "cuda:0")

    def test_valid_boundaries_and_custom_head_order_reach_mocked_run(self):
        for tokens, rank in ((4, 1024), (4096, 256)):
            with self.subTest(tokens=tokens, rank=rank):
                result, report, args = self.mocked_main(
                    [
                        "--run",
                        "--device=cuda:17",
                        f"--tokens={tokens}",
                        f"--rank={rank}",
                        "--warmup=1",
                        "--iterations=1",
                        "--seed=0",
                        "--relative-rms=1e-9",
                        "--max-absolute=1e-9",
                        "--local-head-ids=63,0,17",
                    ],
                    report={"passed": True},
                )
                self.assertEqual(result, 0)
                self.assertTrue(report["passed"])
                self.assertEqual(args.tokens, tokens)
                self.assertEqual(args.rank, rank)
                self.assertEqual(args.device, "cuda:17")
                self.assertEqual(args.warmup, 1)
                self.assertEqual(args.iterations, 1)
                self.assertEqual(args.seed, 0)
                self.assertEqual(args.relative_rms, 1e-9)
                self.assertEqual(args.max_absolute, 1e-9)
                self.assertEqual(tuple(args.local_head_ids), (63, 0, 17))

    def test_failed_check_is_preserved_as_json_and_nonzero_exit(self):
        report = {"passed": False, "failures": ["native projection mismatch"]}
        result, actual, _ = self.mocked_main(
            ["--run", "--device", "cuda:0"], report=report
        )
        self.assertEqual(result, 1)
        self.assertEqual(actual, report)

    def test_runtime_exception_becomes_failed_json_and_nonzero_exit(self):
        result, report, _ = self.mocked_main(
            ["--run", "--device", "cuda:0"],
            error=RuntimeError("native projection unavailable"),
        )
        self.assertEqual(result, 1)
        self.assertIs(report["passed"], False)
        self.assertIn("RuntimeError", report["error"])
        self.assertIn("native projection unavailable", report["error"])


if __name__ == "__main__":
    unittest.main()
