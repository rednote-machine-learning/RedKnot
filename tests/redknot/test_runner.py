from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

from benchmarks.redknot.contract import SCHEMA, digest, file_digest, seal, tokenizer_identity, write_json
from benchmarks.redknot.runner import (
    _configure_process_group,
    _loaded_module_identity,
    _select_devices,
    _stop_workers,
    aggregate_ranks,
    preflight_model,
    run_benchmark,
    source_identity,
)


def make_case(sample_id: str, tokens: list[int]) -> dict:
    references = ["Mercury"]
    return {
        "sample_id": sample_id,
        "dataset": "hotpotqa",
        "input_ids": tokens,
        "input_ids_sha256": digest(tokens),
        "references": references,
        "references_sha256": digest(references),
        "max_new_tokens": 2,
        "scored": True,
        "ignore_eos": False,
    }


def make_manifest(checkpoint: Path) -> dict:
    return seal(
        {
            "schema": SCHEMA,
            "model": {
                "profile": "deepseek-v4.1-flash",
                "model_id": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "tokenizer": tokenizer_identity(checkpoint),
            },
            "suite": {"name": "unit-test"},
            "warmup": [make_case("warmup", [1, 2])],
            "cases": [make_case("case", [3, 4])],
        }
    )


def rank_result(rank: int, manifest: dict, iterations: int) -> dict:
    phases = []
    for iteration in range(iterations):
        records = []
        for case in manifest["cases"]:
            records.append(
                {
                    **{
                        key: case[key]
                        for key in ("sample_id", "dataset", "input_ids_sha256", "references", "max_new_tokens", "scored", "ignore_eos")
                    },
                    "iteration": iteration,
                    "dp_rank": rank,
                    "output_token_ids": [31, 32],
                    "text": "Mercury",
                    "ttft_s": 0.5,
                    "latency_s": 1.0,
                    "finished": True,
                    "finish_reason": "length",
                }
            )
        phases.append({"iteration": iteration, "elapsed_s": 2.0 + rank, "completed_requests": len(records), "requests": records})
    return {
        "rank": rank,
        "status": "complete",
        "phases": phases,
        "runtime_versions": {"torch": "test-only", "cuda": "test-only", "deepinfer": "test-only"},
        "native_engine": {"module": "unit-test-fixture", "module_sha256": "b" * 64},
    }


def batched_rank_result(rank: int, manifest: dict, iterations: int, batch_size: int) -> dict:
    result = rank_result(rank, manifest, iterations)
    ids = [case["sample_id"] for case in manifest["cases"]]
    for phase in result["phases"]:
        phase["batches"] = []
        for batch_index, start in enumerate(range(0, len(ids), batch_size)):
            sample_ids = ids[start : start + batch_size]
            phase["batches"].append(
                {
                    "batch_index": batch_index,
                    "sample_ids": sample_ids,
                    "elapsed_s": 2.0 + 3.0 * ((batch_index + rank) % 2),
                    "completed_requests": len(sample_ids),
                }
            )
            for record in phase["requests"]:
                if record["sample_id"] in sample_ids:
                    record["batch_index"] = batch_index
        phase["elapsed_s"] = sum(batch["elapsed_s"] for batch in phase["batches"])
    return result


class FakeProcess:
    def __init__(self, *, args: tuple, fail: bool = False, **kwargs: Any) -> None:
        self.rank, self.manifest, self.spec = args
        self.fail = fail
        self.pid = None
        self.exitcode = None

    def start(self) -> None:
        directory = Path(self.spec["output_dir"])
        if self.fail:
            write_json(directory / f"rank-{self.rank}.failure.json", {"status": "failed", "error": "test failure"})
            self.exitcode = 1
        else:
            result = (
                batched_rank_result(self.rank, self.manifest, self.spec["iterations"], self.spec["admission_batch_size"])
                if "admission_batch_size" in self.spec
                else rank_result(self.rank, self.manifest, self.spec["iterations"])
            )
            write_json(directory / f"rank-{self.rank}.json", result)
            self.exitcode = 0

    def join(self, timeout: float | None = None) -> None:
        pass


class FakeContext:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def Process(self, **kwargs: Any) -> FakeProcess:
        return FakeProcess(fail=self.fail, **kwargs)


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        for filename in ("examples/_config.py", "deepinfer/engine/llm_engine.py"):
            path = self.runtime / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# CPU-only runtime layout fixture; never imported.\n")
        self.checkpoint = self.root / "model"
        self.checkpoint.mkdir()
        (self.checkpoint / "tokenizer.json").write_text('{"unit_test":true}')
        self.config = {"architectures": ["DeepseekV41ForCausalLM"], "vocab_size": 64, "max_position_embeddings": 128}
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        self.manifest = make_manifest(self.checkpoint)
        self.hardware = {
            "hostname": "unit-test",
            "gpus": [{"index": index, "compute_capability": "10.0", "name": "fake-test-device"} for index in range(8)],
        }

    def execute(self, *, fail: bool = False, environment: dict[str, str] | None = None, **kwargs: Any) -> Path:
        with (
            patch.dict(os.environ, environment or {}, clear=True),
            patch("benchmarks.redknot.runner.REPO_ROOT", self.runtime),
            patch("benchmarks.redknot.runner._hardware_inventory", return_value=copy.deepcopy(self.hardware)),
            patch("benchmarks.redknot.runner.source_identity", return_value={"commit": "a" * 40, "dirty": False}),
            patch("benchmarks.redknot.runner._free_master_port", return_value=25000),
            patch("benchmarks.redknot.runner.multiprocessing.get_context", return_value=FakeContext(fail=fail)),
        ):
            return run_benchmark(self.manifest, self.checkpoint, self.root / "run", ep=2, cp=1, dp=2, iterations=2, **kwargs)

    def test_preflight_checks_tokenizer_architecture_and_context(self) -> None:
        self.assertEqual(preflight_model(self.manifest, self.checkpoint)["required_context"], 4)
        for changes, message in (
            ({"architectures": ["Qwen3ForCausalLM"]}, "supports only"),
            ({"vocab_size": 3}, "outside"),
            ({"max_position_embeddings": 3}, "exceeding"),
        ):
            (self.checkpoint / "config.json").write_text(json.dumps({**self.config, **changes}))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                preflight_model(self.manifest, self.checkpoint)
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        (self.checkpoint / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            preflight_model(self.manifest, self.checkpoint)

    def test_selected_runtime_git_trust_is_process_local_and_exactly_scoped(self) -> None:
        checkout = self.root / "mounted runtime"
        checkout.mkdir()
        subprocess.run(["git", "-C", str(checkout), "init", "-q"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark-test@example.invalid",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "CPU ownership fixture",
            ],
            check=True,
        )
        config_path = checkout / ".git" / "config"
        original_config = config_path.read_bytes()
        check_output = subprocess.check_output
        with patch.dict(os.environ, {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}):
            untrusted = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, check=False)
            if untrusted.returncode == 0:
                self.skipTest("installed Git does not enforce GIT_TEST_ASSUME_DIFFERENT_OWNER")
            self.assertIn(b"dubious ownership", untrusted.stderr)
            with patch("benchmarks.redknot.runner.subprocess.check_output", wraps=check_output) as git:
                identity = source_identity(checkout)
            for invocation in git.call_args_list:
                self.assertEqual(
                    invocation.args[0][:5],
                    ["git", "-c", f"safe.directory={checkout.resolve()}", "-C", str(checkout.resolve())],
                )
        self.assertEqual(len(identity["commit"]), 40)
        self.assertFalse(identity["dirty"])
        self.assertEqual(config_path.read_bytes(), original_config)

    def test_source_identity_keeps_bounded_git_error_diagnostics(self) -> None:
        failure = subprocess.CalledProcessError(128, ["git"], stderr=b"fatal: selected runtime unavailable\n")
        with (
            patch("benchmarks.redknot.runner.subprocess.check_output", side_effect=failure),
            self.assertRaisesRegex(ValueError, "selected runtime unavailable"),
        ):
            source_identity(self.runtime)

    def test_configuration_only_profile_cannot_run_under_a_supported_checkpoint(self) -> None:
        for changes, message in (
            ({"profile": "qwen3-32b", "model_id": "Qwen/Qwen3-32B"}, "configuration-only"),
            ({"model_id": "Qwen/Qwen3-32B"}, "does not match"),
        ):
            forged = seal({**self.manifest, "model": {**self.manifest["model"], **changes}})
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                preflight_model(forged, self.checkpoint)

    def test_global_qps_uses_slowest_rank_per_phase(self) -> None:
        path = self.execute()
        report = json.loads(path.read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["mode"], "baseline")
        self.assertEqual(report["elapsed_s"], 6.0)
        self.assertEqual(report["completed_requests"], 4)
        self.assertAlmostEqual(report["summary"]["completed_requests_per_s"], 4 / 6)
        self.assertEqual(report["metadata"]["settings"]["replication"], "full_cohort_per_dp_rank")
        self.assertTrue(report["comparison_eligible"])
        self.assertFalse(report["activation"]["redknot_enabled"])

    def test_cold_smoke_is_explicit_and_not_comparison_eligible(self) -> None:
        report = json.loads(self.execute(warmup_iterations=0).read_text())
        self.assertEqual(report["metadata"]["warmup"]["state"], "cold")
        self.assertFalse(report["metadata"]["warmup"]["complete"])
        self.assertFalse(report["comparison_eligible"])

    def test_baseline_disables_inherited_artificial_expert_routing(self) -> None:
        inherited = {"DEEPINFER_FORCE_LOAD_BALANCE": "1", "DEEPINFER_FORCE_IMBALANCE_RATIO": "1.8"}
        path = self.execute(environment=inherited)
        report = json.loads(path.read_text())
        launch = json.loads((path.parent / "launch.json").read_text())
        for key in inherited:
            self.assertEqual(launch["environment"][key], "0")
            self.assertEqual(report["metadata"]["settings"]["environment"][key], "0")

    def test_explicit_sm90_kernel_setting_preserves_eager_and_prefill_environment(self) -> None:
        inherited = {
            "DEEPINFER_BATCH_INVARIANT": "1",
            "DEEPINFER_ENFORCE_EAGER": "1",
            "DEEPINFER_DFLASH_ENFORCE_EAGER": "1",
            "DEEPINFER_PREFILL_ATTEMPT_INTERVAL": "1",
            "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL": "1",
        }
        path = self.execute(environment=inherited, batch_invariant=False)
        report = json.loads(path.read_text())
        launch = json.loads((path.parent / "launch.json").read_text())
        self.assertIs(report["metadata"]["settings"]["batch_invariant"], False)
        self.assertIs(launch["batch_invariant"], False)
        self.assertEqual(launch["max_num_seqs"], 1)
        for key, value in {**inherited, "DEEPINFER_BATCH_INVARIANT": "0"}.items():
            self.assertEqual(launch["environment"][key], value)
            self.assertEqual(report["metadata"]["settings"]["environment"][key], value)

    def test_default_batch_invariance_overrides_inherited_environment(self) -> None:
        report = json.loads(self.execute(environment={"DEEPINFER_BATCH_INVARIANT": "0"}).read_text())
        self.assertIs(report["metadata"]["settings"]["batch_invariant"], True)
        self.assertEqual(report["metadata"]["settings"]["environment"]["DEEPINFER_BATCH_INVARIANT"], "1")
        with self.assertRaisesRegex(ValueError, "batch_invariant must be boolean"):
            run_benchmark(self.manifest, self.checkpoint, self.root / "invalid", batch_invariant=1)

    def test_supervision_marker_allowed_only_in_explicit_supervised_mode(self) -> None:
        marker = "unit-test-authenticated-lifecycle-marker"
        with self.assertRaisesRegex(ValueError, "RedKnot environment"):
            self.execute(environment={"REDKNOT_GLM_RUN_ID": marker})
        path = self.execute(environment={"REDKNOT_GLM_RUN_ID": marker}, supervised_process_group=True)
        report = json.loads(path.read_text())
        self.assertIs(report["metadata"]["settings"]["supervised_process_group"], True)
        self.assertEqual(report["metadata"]["settings"]["process_group_owner"], "external")
        self.assertNotIn(marker, path.read_text())
        self.assertNotIn(marker, (path.parent / "launch.json").read_text())
        for key in ("REDKNOT_GLM_RUN_ID_SUFFIX", "DEEPINFER_REDKNOT_ENABLED"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "RedKnot environment"):
                self.execute(environment={key: "1"}, supervised_process_group=True)

    def test_process_group_setup_and_cleanup_respect_supervisor_ownership(self) -> None:
        for supervised in (False, True):
            process = SimpleNamespace(pid=12345, is_alive=Mock(return_value=True), terminate=Mock(), kill=Mock(), join=Mock())
            with self.subTest(supervised=supervised), patch("os.setsid") as setsid, patch("os.killpg") as killpg:
                _configure_process_group(supervised_process_group=supervised)
                _stop_workers([SimpleNamespace(pid=None), process], supervised_process_group=supervised)
                if supervised:
                    setsid.assert_not_called()
                    killpg.assert_not_called()
                    process.terminate.assert_called_once_with()
                    process.kill.assert_called_once_with()
                else:
                    setsid.assert_called_once_with()
                    self.assertEqual(killpg.call_args_list, [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)])
                    process.terminate.assert_not_called()
                    process.kill.assert_not_called()

    def test_rank_failure_writes_failure_not_success(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            self.execute(fail=True)
        self.assertFalse((self.root / "run" / "report.json").exists())
        self.assertEqual(json.loads((self.root / "run" / "failure.json").read_text())["status"], "failed")

    def test_engine_binary_identity_hashes_loaded_extension_and_package_children(self) -> None:
        binary = self.root / "engine.so"
        binary.write_bytes(b"unit-test binary fixture")
        module = SimpleNamespace(__name__="fixture.engine", __file__=str(binary))
        identity = _loaded_module_identity(module)
        self.assertEqual(identity["kind"], "extension")
        self.assertEqual(identity["loaded_extensions"][module.__name__]["sha256"], file_digest(binary))
        wrapper = self.root / "__init__.py"
        wrapper.write_text("# unit-test package wrapper\n")
        package = SimpleNamespace(__name__="fixture.engine", __file__=str(wrapper), __path__=[str(self.root)])
        with patch.dict(sys.modules, {"fixture.engine._native": module}):
            identity = _loaded_module_identity(package)
        self.assertEqual(identity["kind"], "package")
        self.assertEqual(identity["loaded_extensions"]["fixture.engine._native"]["sha256"], file_digest(binary))
        self.assertEqual(identity["module_sha256"], file_digest(wrapper))

    def test_inconsistent_engine_binaries_never_produce_success(self) -> None:
        original = rank_result

        def differing_binary(rank: int, manifest: dict, iterations: int) -> dict:
            result = original(rank, manifest, iterations)
            result["native_engine"]["module_sha256"] = str(rank) * 64
            return result

        with patch(f"{__name__}.rank_result", side_effect=differing_binary), self.assertRaisesRegex(RuntimeError, "binary identities"):
            self.execute()
        self.assertFalse((self.root / "run" / "report.json").exists())

    def test_warmup_and_parameters_fail_before_hardware(self) -> None:
        self.manifest = seal({**self.manifest, "warmup": []})
        with patch("benchmarks.redknot.runner._hardware_inventory") as hardware:
            with self.assertRaisesRegex(ValueError, "warmup"):
                run_benchmark(self.manifest, self.checkpoint, self.root / "run")
            hardware.assert_not_called()
        with self.assertRaisesRegex(ValueError, "divisible"):
            run_benchmark(self.manifest, self.checkpoint, self.root / "run", ep=3, cp=2, dp=1)

    def test_output_directory_is_never_overwritten(self) -> None:
        self.execute()
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_admission_batches_bound_engine_and_preserve_every_rank_cohort(self) -> None:
        self.manifest = seal({**self.manifest, "cases": [*self.manifest["cases"], make_case("second", [5, 6])]})
        path = self.execute(admission_batch_size=1, timeout_s=20000)
        result = json.loads(path.read_text())
        settings = result["metadata"]["settings"]
        self.assertEqual(settings["max_num_seqs"], 1)
        self.assertEqual(settings["admission_batch_size"], 1)
        self.assertEqual(settings["replication"], "full_cohort_per_dp_rank")
        self.assertEqual(result["completed_requests"], 8)
        self.assertEqual(result["elapsed_s"], 20.0)
        self.assertEqual(result["phases"][0]["rank_elapsed_s"], [7.0, 7.0])
        self.assertEqual(result["phases"][0]["elapsed_s"], 10.0)
        launch = json.loads((path.parent / "launch.json").read_text())
        self.assertEqual(launch["timeout_seconds"], 20000)
        self.assertEqual(launch["max_num_seqs"], 1)

    def test_admission_aggregate_rejects_missing_reordered_and_mislabeled_batches(self) -> None:
        manifest = seal({**self.manifest, "cases": [*self.manifest["cases"], make_case("second", [5, 6])]})
        originals = [batched_rank_result(rank, manifest, 1, 1) for rank in range(2)]
        mutations = [
            lambda rows: rows[0]["phases"][0]["batches"].pop(),
            lambda rows: rows[0]["phases"][0]["batches"].reverse(),
            lambda rows: rows[0]["phases"][0]["batches"][0].update(sample_ids=["second"]),
            lambda rows: rows[0]["phases"][0]["requests"][0].update(batch_index=1),
            lambda rows: rows[0]["phases"][0]["requests"][0].update(latency_s=3.0),
            lambda rows: rows[0]["phases"][0].update(elapsed_s=99.0),
        ]
        for mutate in mutations:
            results = copy.deepcopy(originals)
            mutate(results)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                aggregate_ranks(results, manifest, iterations=1, dp_size=2, admission_batch_size=1)
        with self.assertRaisesRegex(RuntimeError, "undeclared"):
            aggregate_ranks(originals, manifest, iterations=1, dp_size=2)

    def test_invalid_admission_bound_fails_before_hardware(self) -> None:
        for size in (0, -1, True, 1.5):
            with patch("benchmarks.redknot.runner._hardware_inventory") as hardware, self.assertRaisesRegex(ValueError, "admission"):
                run_benchmark(self.manifest, self.checkpoint, self.root / "run", admission_batch_size=size)
            hardware.assert_not_called()

    def test_admission_limit_uses_actual_warmup_batches_and_skips_cold_warmup(self) -> None:
        self.manifest = seal({**self.manifest, "warmup": [make_case(f"warmup-{i}", [10 + i]) for i in range(3)]})
        path = self.execute(admission_batch_size=2)
        self.assertEqual(json.loads(path.read_text())["metadata"]["settings"]["max_num_seqs"], 2)
        # Use a new directory; output files must never be overwritten.
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("benchmarks.redknot.runner.REPO_ROOT", self.runtime),
            patch("benchmarks.redknot.runner._hardware_inventory", return_value=copy.deepcopy(self.hardware)),
            patch("benchmarks.redknot.runner.source_identity", return_value={"commit": "a" * 40, "dirty": False}),
            patch("benchmarks.redknot.runner._free_master_port", return_value=25000),
            patch("benchmarks.redknot.runner.multiprocessing.get_context", return_value=FakeContext()),
        ):
            cold_path = run_benchmark(
                self.manifest,
                self.checkpoint,
                self.root / "cold-run",
                ep=2,
                dp=2,
                iterations=1,
                warmup_iterations=0,
                admission_batch_size=2,
            )
        cold = json.loads(cold_path.read_text())
        self.assertEqual(cold["metadata"]["settings"]["max_num_seqs"], 1)
        self.assertFalse(cold["comparison_eligible"])

    def test_aggregate_rejects_missing_duplicate_and_nonfinite_results(self) -> None:
        original = [rank_result(rank, self.manifest, 2) for rank in range(2)]
        mutations = [
            lambda rows: rows.pop(),
            lambda rows: rows[1].update(rank=0),
            lambda rows: rows[0]["phases"].pop(),
            lambda rows: rows[0]["phases"][0]["requests"].clear(),
            lambda rows: rows[0]["phases"][0].update(elapsed_s=float("nan")),
            lambda rows: rows[0]["phases"][0]["requests"][0].update(finished=False),
        ]
        for mutate in mutations:
            rows = copy.deepcopy(original)
            mutate(rows)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                aggregate_ranks(rows, self.manifest, iterations=2, dp_size=2)

    def test_device_selection_preserves_contiguous_user_block(self) -> None:
        self.assertEqual(_select_devices(self.hardware, 2, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}), [4, 5])
        for environment in ({"CUDA_VISIBLE_DEVICES": "0,2"}, {"CUDA_VISIBLE_DEVICES": "0,0"}, {"CUDA_VISIBLE_DEVICES": "GPU-uuid"}):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                _select_devices(self.hardware, 2, environment)

    def test_imports_do_not_load_accelerator_libraries(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import sys; import benchmarks.redknot.native, benchmarks.redknot.runner; assert not {'torch', 'deepinfer', 'numpy'} & set(sys.modules)",
        ]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    unittest.main()
