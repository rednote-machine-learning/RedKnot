"""Single-node, supervised native DeepInfer benchmark execution.

CPU preparation and report inspection never import DeepInfer or torch. Each DP
rank runs the same cohort, explicitly recorded as replication. A failed rank
invalidates the entire run and its sibling workers are stopped, so a distributed
collective cannot silently hang after a peer error.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any

from .catalog import get_model_profile
from .contract import digest, file_digest, seal, tokenizer_identity, validate_manifest, write_json
from .native import ADMISSION_POLICY, BOUNDARY, run_admission_phase

RUN_SCHEMA = "redknot-deepinfer-benchmark-run-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ARCHITECTURES = frozenset({"DeepseekV41ForCausalLM"})
BASELINE_ENV = {
    "DEEPINFER_USE_NEXTN_DECODE": "0",
    "DEEPINFER_ENABLE_MTP_VARLEN": "0",
    "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
    "DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE": "0",
    "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",
    "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": "off",
    "DEEPINFER_FORCE_LOAD_BALANCE": "0",
    "DEEPINFER_FORCE_IMBALANCE_RATIO": "0",
}


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def preflight_model(manifest: dict, model_path: str | Path) -> dict:
    """Validate architecture and exact tokenizer identity before GPU imports."""
    validate_manifest(manifest)
    profile = get_model_profile(manifest["model"]["profile"])
    if not profile["native_supported"]:
        raise ValueError(f"model profile {profile['profile']} is configuration-only; its native DeepInfer port is pending")
    if manifest["model"]["model_id"] != profile["model_id"]:
        raise ValueError("manifest model_id does not match its declared model profile")
    checkpoint = Path(model_path).expanduser().resolve()
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError("--model-path must be a local checkpoint containing config.json")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures or any(arch not in SUPPORTED_ARCHITECTURES for arch in architectures):
        raise ValueError(
            f"native DeepInfer supports only {sorted(SUPPORTED_ARCHITECTURES)}; checkpoint declares {architectures!r}. "
            "Migrating a benchmark profile does not add a model implementation."
        )
    tokenizer = tokenizer_identity(checkpoint)
    if tokenizer["fingerprint"] != manifest["model"]["tokenizer"]["fingerprint"]:
        raise ValueError("checkpoint tokenizer fingerprint differs from the frozen manifest; prepare inputs with this tokenizer")
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is int:
        for case in manifest["warmup"] + manifest["cases"]:
            if max(case["input_ids"]) >= vocab_size:
                raise ValueError(f"{case['sample_id']}: frozen token ID is outside checkpoint vocabulary")
    required_context = max(len(case["input_ids"]) + case["max_new_tokens"] for case in manifest["warmup"] + manifest["cases"])
    context_limit = config.get("max_position_embeddings")
    if type(context_limit) is int and required_context > context_limit:
        raise ValueError(f"manifest requires context {required_context}, exceeding checkpoint limit {context_limit}")
    return {
        "checkpoint_path": str(checkpoint),
        "config_sha256": file_digest(config_path),
        "architectures": architectures,
        "required_context": required_context,
        "tokenizer": tokenizer,
        "weight_files": {
            path.name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(checkpoint.glob("*.safetensors"))
        },
        "weight_index_sha256": {path.name: file_digest(path) for path in sorted(checkpoint.glob("*.index.json"))},
    }


def source_identity(root: str | Path) -> dict:
    """Record commit plus hashes of modified/untracked runtime source files."""
    root = Path(root).resolve()

    def git(*args: str) -> bytes:
        # The caller selected this checkout. Trust exactly its resolved path
        # for this command, without changing persistent Git configuration.
        return subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", str(root), *args], stderr=subprocess.PIPE)

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        tree = git("rev-parse", "HEAD^{tree}").decode().strip()
        # Include tracked changes and untracked source, not generated logs or weights.
        changed = set(git("diff", "--name-only", "HEAD", "-z").decode().split("\0"))
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
        source_roots = {"deepinfer", "deep_cluster", "csrc", "scripts", "examples", "benchmarks", "tests"}
        changed.update(
            path
            for path in untracked
            if Path(path).parts
            and Path(path).parts[0] in source_roots
            and Path(path).suffix in {".py", ".sh", ".cpp", ".cu", ".h", ".hpp", ".json", ".toml"}
        )
        files = {}
        for name in sorted(changed - {""}):
            path = root / name
            files[name] = file_digest(path) if path.is_file() else "deleted-or-nonregular"
        submodules = git("submodule", "status", "--recursive").decode().splitlines()
        return {
            "root": str(root),
            "commit": commit,
            "tree": tree,
            "dirty": bool(files),
            "changed_files": files,
            "changes_sha256": digest(files),
            "submodules": submodules,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        detail = " ".join(detail.split())[:1000]
        raise ValueError(f"runtime root must be an identifiable Git checkout: {root}: {detail}") from None


def _loaded_module_identity(module: Any) -> dict:
    """Hash the imported engine file and any loaded package child extensions.

    A Python package wrapper is distinguished from an extension. If a wrapper
    loads its binary outside Python's module system, that absence is recorded
    explicitly instead of claiming that a wrapper hash identifies the binary.
    """
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("loaded native engine module has no identifiable file")
    path = Path(module_file).resolve()
    files = {}
    for name, child in list(sys.modules.items()):
        if name != module.__name__ and not name.startswith(module.__name__ + "."):
            continue
        filename = getattr(child, "__file__", None)
        if filename and any(filename.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
            binary = Path(filename).resolve()
            files[name] = {"path": str(binary), "sha256": file_digest(binary)}
    if any(str(path).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        kind = "extension"
        files[module.__name__] = {"path": str(path), "sha256": file_digest(path)}
    else:
        kind = "package" if hasattr(module, "__path__") else "python_module"
    return {
        "module": module.__name__,
        "kind": kind,
        "module_path": str(path),
        "module_sha256": file_digest(path),
        "loaded_extensions": files,
        "binary_provenance": "loaded_python_extensions" if files else "no_loaded_python_extension_identified",
    }


def _hardware_inventory() -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("native GPU execution requires Linux; CPU prepare/validate/compare commands remain available")
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("this benchmark launcher currently requires NVIDIA GPUs and working nvidia-smi") from exc
    devices = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise RuntimeError("unexpected nvidia-smi hardware inventory")
        index, name, uuid, memory, driver, capability = (field.strip() for field in row)
        devices.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "memory_total_mib": int(memory),
                "driver": driver,
                "compute_capability": capability,
            }
        )
    if not devices:
        raise RuntimeError("no NVIDIA GPU found")
    return {"hostname": socket.gethostname(), "os": platform.platform(), "python": platform.python_version(), "gpus": devices}


def _select_devices(hardware: dict, world_size: int, environment: dict[str, str]) -> list[int]:
    available = [item["index"] for item in hardware["gpus"]]
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        try:
            selected = [int(part.strip()) for part in visible.split(",")]
        except ValueError as exc:
            raise ValueError("this native launcher requires contiguous numeric CUDA_VISIBLE_DEVICES, not UUIDs") from exc
    else:
        offset = int(environment.get("DEVICE_OFFSET", "0"))
        selected = available[offset:]
    if len(selected) < world_size or len(set(selected)) != len(selected) or any(index not in available for index in selected):
        raise ValueError("selected physical devices do not cover dp_size × cp_size")
    selected = selected[:world_size]
    if selected != list(range(selected[0], selected[0] + world_size)):
        raise ValueError("upstream create_llm requires a contiguous GPU block; refusing to remap the user's device selection")
    return selected


def _free_master_port() -> int:
    # DeepInfer's rendezvous is MASTER_PORT - 50. Choose that actual port first.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        actual = listener.getsockname()[1]
    if actual > 65485:
        return _free_master_port()
    return actual + 50


class _NativeEngineBridge:
    def __init__(self, engine: Any, array_factory: Any, barrier: Any) -> None:
        self.engine = engine
        self.array_factory = array_factory
        self.barrier = barrier

    def prepare_input_ids(self, tokens: list[int]) -> Any:
        return self.array_factory(tokens)

    def before_phase(self) -> None:
        self.barrier()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)


def _configure_process_group(*, supervised_process_group: bool) -> None:
    # A lifecycle supervisor must retain every descendant in its owned group.
    if not supervised_process_group and hasattr(os, "setsid"):
        os.setsid()


def _rank_worker(rank: int, manifest: dict, spec: dict) -> None:
    """Spawn target; everything accelerator-dependent is imported in this child."""
    directory = Path(spec["output_dir"])
    llm = None
    try:
        _configure_process_group(supervised_process_group=spec["supervised_process_group"])
        with (directory / f"rank-{rank}.log").open("x", encoding="utf-8") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            os.environ.update(spec["environment"])
            runtime_root = Path(spec["runtime_root"])
            sys.path.insert(0, str(runtime_root))
            config_spec = importlib.util.spec_from_file_location("_redknot_runtime_config", runtime_root / "examples" / "_config.py")
            if config_spec is None or config_spec.loader is None:
                raise RuntimeError("cannot load upstream examples/_config.py")
            config_module = importlib.util.module_from_spec(config_spec)
            config_spec.loader.exec_module(config_module)
            config_module.setup_common_env(
                model_name=spec["model_path"], sampler_force_topk=4096, set_override_experts=False, enable_eplb=False
            )
            llm = config_module.create_llm(
                spec["model_path"],
                rank,
                ep_size=spec["ep"],
                cp_size=spec["cp"],
                data_parallel_size=spec["dp"],
                max_num_batched_tokens=spec["max_num_batched_tokens"],
                max_model_len=spec["max_model_len"],
                max_num_seqs=spec["max_num_seqs"],
                gpu_memory_utilization=spec["memory_utilization"],
                seed=spec["seed"],
                lookahead_tokens=0,
            )
            import numpy as np
            import torch
            import torch.distributed as dist

            import deepinfer
            import deepinfer._C.engine as native_engine
            from deepinfer.model_executor.parallel_utils.parallel_state import PG
            from deepinfer.sampling_params import SamplingParams

            if Path(deepinfer.__file__).resolve().parent != runtime_root / "deepinfer":
                raise RuntimeError("imported DeepInfer does not belong to the selected runtime root")
            native_engine_identity = _loaded_module_identity(native_engine)
            RequestOutput = native_engine.RequestOutput
            group = PG.cpu_data_parallel_group()

            def barrier() -> None:
                torch.cuda.synchronize()
                if spec["dp"] > 1:
                    dist.barrier(group=group)

            bridge = _NativeEngineBridge(llm.llm_engine, lambda ids: np.asarray(ids, dtype=np.int32), barrier)

            def sampling(case: dict) -> Any:
                sequence_seed = (spec["seed"] + int(digest(case["sample_id"])[:16], 16)) % 2**64
                return SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=case["max_new_tokens"],
                    ignore_eos=case["ignore_eos"],
                    seq_seed=sequence_seed,
                )

            def read_output(raw: Any) -> dict:
                return {
                    "request_id": str(RequestOutput.get_request_id(raw)),
                    "token_id": int(RequestOutput.get_token_id(raw)),
                    "finished": bool(RequestOutput.get_finished(raw)),
                    "finish_reason": str(RequestOutput.get_finish_reason(raw)),
                }

            def progress(role: str, iteration: int, batch: Any) -> None:
                print(
                    json.dumps({"event": "redknot_batch_complete", "phase": role, "iteration": iteration, "dp_rank": rank, **batch}),
                    flush=True,
                )

            for iteration in range(spec["warmup_iterations"]):
                run_admission_phase(
                    bridge,
                    manifest["warmup"],
                    sampling_factory=sampling,
                    output_reader=read_output,
                    admission_batch_size=spec.get("admission_batch_size"),
                    progress_callback=lambda batch: progress("warmup", iteration, batch),
                )
                barrier()

            phases = []
            for iteration in range(spec["iterations"]):
                phase = run_admission_phase(
                    bridge,
                    manifest["cases"],
                    sampling_factory=sampling,
                    output_reader=read_output,
                    admission_batch_size=spec.get("admission_batch_size"),
                    progress_callback=lambda batch: progress("measurement", iteration, batch),
                )
                barrier()
                cases_by_id = {case["sample_id"]: case for case in manifest["cases"]}
                for record in phase["requests"]:
                    case = cases_by_id[record["sample_id"]]
                    record.update(
                        {key: case[key] for key in ("dataset", "references", "max_new_tokens", "scored", "ignore_eos", "input_ids_sha256")}
                    )
                    record.update(
                        {
                            "iteration": iteration,
                            "dp_rank": rank,
                            "text": llm.llm_engine.tokenizer.decode(record["output_token_ids"], skip_special_tokens=True),
                        }
                    )
                phase["iteration"] = iteration
                phases.append(phase)
            write_json(
                directory / f"rank-{rank}.json",
                {
                    "rank": rank,
                    "status": "complete",
                    "phases": phases,
                    "runtime_versions": {
                        "torch": str(torch.__version__),
                        "cuda": str(torch.version.cuda),
                        "deepinfer": str(deepinfer.__version__),
                    },
                    "native_engine": native_engine_identity,
                },
            )
            llm.close()
            llm = None
    except BaseException as exc:
        failure = {"rank": rank, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        failure_path = directory / f"rank-{rank}.failure.json"
        if not failure_path.exists():
            write_json(failure_path, failure)
        # Do not block reporting this failure on a collective close operation.
        raise


def _stop_workers(processes: list[Any], *, supervised_process_group: bool = False) -> None:
    """Stop owned ranks; an external supervisor owns descendant group cleanup."""
    for process in processes:
        if process.pid is None:
            continue
        if supervised_process_group:
            if process.is_alive():
                process.terminate()
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
    deadline = time.monotonic() + 2
    for process in processes:
        if process.pid is not None:
            process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.pid is None:
            continue
        if supervised_process_group:
            if process.is_alive():
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.is_alive():
                    process.kill()
        process.join(timeout=1)


def aggregate_ranks(
    rank_results: list[dict],
    manifest: dict,
    *,
    iterations: int,
    dp_size: int,
    admission_batch_size: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Validate complete rank/iteration coverage before producing a report."""
    if len(rank_results) != dp_size or {result.get("rank") for result in rank_results} != set(range(dp_size)):
        raise RuntimeError("missing or duplicate DP rank results")
    expected_ids = {case["sample_id"] for case in manifest["cases"]}
    ordered_ids = [case["sample_id"] for case in manifest["cases"]]
    if admission_batch_size is not None:
        _positive_int(admission_batch_size, "admission_batch_size")
    expected_batches = (
        [ordered_ids[start : start + admission_batch_size] for start in range(0, len(ordered_ids), admission_batch_size)]
        if admission_batch_size is not None
        else None
    )
    phase_map: dict[tuple[int, int], dict] = {}
    for result in rank_results:
        if result.get("status") != "complete" or len(result.get("phases", [])) != iterations:
            raise RuntimeError("rank did not complete every iteration")
        for phase in result["phases"]:
            key = (result["rank"], phase.get("iteration"))
            if key in phase_map or key[1] not in range(iterations):
                raise RuntimeError("duplicate or unexpected iteration")
            duration = phase.get("elapsed_s")
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise RuntimeError("invalid phase duration")
            requests = phase.get("requests", [])
            if len(requests) != len(expected_ids) or {record.get("sample_id") for record in requests} != expected_ids:
                raise RuntimeError("rank result omitted or duplicated a manifest sample")
            if phase.get("completed_requests") != len(expected_ids) or not all(record.get("finished") is True for record in requests):
                raise RuntimeError("rank result contains incomplete requests")
            if expected_batches is None:
                if "batches" in phase or any("batch_index" in record for record in requests):
                    raise RuntimeError("undeclared admission batches in rank result")
            else:
                batches = phase.get("batches")
                if not isinstance(batches, list) or len(batches) != len(expected_batches):
                    raise RuntimeError("rank result omitted an admission batch")
                for batch_index, (batch, sample_ids) in enumerate(zip(batches, expected_batches, strict=True)):
                    if not isinstance(batch, dict) or type(batch.get("batch_index")) is not int or batch["batch_index"] != batch_index:
                        raise RuntimeError("rank admission batches are out of order")
                    if batch.get("sample_ids") != sample_ids or batch.get("completed_requests") != len(sample_ids):
                        raise RuntimeError("rank admission batch coverage differs from manifest order")
                    elapsed = batch.get("elapsed_s")
                    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError("invalid rank admission batch duration")
                    selected = [record for record in requests if record.get("batch_index") == batch_index]
                    if [record.get("sample_id") for record in selected] != sample_ids:
                        raise RuntimeError("rank request order differs from admission batches")
                    for record in selected:
                        latency = record.get("latency_s")
                        if type(latency) not in (int, float) or not math.isfinite(latency) or not 0 < latency <= elapsed + 1e-8:
                            raise RuntimeError("request latency exceeds its rank admission batch")
                if not math.isclose(duration, sum(batch["elapsed_s"] for batch in batches), rel_tol=1e-8, abs_tol=1e-8):
                    raise RuntimeError("rank phase duration differs from its admission batch sum")
            phase_map[key] = phase
    phases, requests = [], []
    for iteration in range(iterations):
        rank_phases = [phase_map[rank, iteration] for rank in range(dp_size)]
        durations = [phase["elapsed_s"] for phase in rank_phases]
        phase = {
            "iteration": iteration,
            "elapsed_s": max(durations),
            "rank_elapsed_s": durations,
            "completed_requests": len(expected_ids) * dp_size,
        }
        if expected_batches is not None:
            phase["batches"] = [
                {
                    "batch_index": batch_index,
                    "sample_ids": sample_ids,
                    "rank_elapsed_s": [item["batches"][batch_index]["elapsed_s"] for item in rank_phases],
                    "elapsed_s": max(item["batches"][batch_index]["elapsed_s"] for item in rank_phases),
                    "completed_requests": len(sample_ids) * dp_size,
                }
                for batch_index, sample_ids in enumerate(expected_batches)
            ]
            phase["elapsed_s"] = sum(batch["elapsed_s"] for batch in phase["batches"])
        phases.append(phase)
        for rank_phase in rank_phases:
            requests.extend(rank_phase["requests"])
    return phases, requests


def run_benchmark(
    manifest: dict,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    ep: int = 8,
    cp: int = 1,
    dp: int | None = None,
    warmup_iterations: int = 3,
    iterations: int = 10,
    max_num_batched_tokens: int = 32768,
    admission_batch_size: int | None = None,
    memory_utilization: float = 0.85,
    batch_invariant: bool = True,
    supervised_process_group: bool = False,
    seed: int = 0,
    runtime_root: str | Path | None = None,
    timeout_s: float = 7200,
) -> Path:
    """Execute the fully recomputed native baseline and write report.json.

    Only single-node NVIDIA execution is implemented. An explicit runtime_root
    permits an independently installed SM90 port, whose exact source identity
    is recorded; selecting it is not a claim of verified hardware support.
    """
    checkpoint = preflight_model(manifest, model_path)
    for value, name in ((ep, "ep"), (cp, "cp"), (iterations, "iterations"), (max_num_batched_tokens, "max_num_batched_tokens")):
        _positive_int(value, name)
    _positive_int(warmup_iterations, "warmup_iterations", allow_zero=True)
    _positive_int(seed, "seed", allow_zero=True)
    if admission_batch_size is not None:
        _positive_int(admission_batch_size, "admission_batch_size")
    if type(batch_invariant) is not bool:
        raise ValueError("batch_invariant must be boolean")
    if type(supervised_process_group) is not bool:
        raise ValueError("supervised_process_group must be boolean")
    dp = max(ep // cp, 1) if dp is None else _positive_int(dp, "dp")
    if dp * cp % ep:
        raise ValueError("dp × cp must be divisible by ep")
    if warmup_iterations and not manifest["warmup"]:
        raise ValueError("warmup_iterations > 0 requires explicit, disjoint warmup cases")
    if not isinstance(memory_utilization, (int, float)) or not math.isfinite(memory_utilization) or not 0 < memory_utilization < 1:
        raise ValueError("memory_utilization must be finite and strictly between 0 and 1")
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0:
        raise ValueError("this benchmark launcher supports one node; multi-node result aggregation is not implemented")
    enabled_redknot = {
        key: value
        for key, value in os.environ.items()
        if "REDKNOT" in key.upper()
        and value.lower() not in {"", "0", "false", "off", "none"}
        and not (supervised_process_group and key == "REDKNOT_GLM_RUN_ID")
    }
    if enabled_redknot:
        raise ValueError(f"baseline refuses active RedKnot environment overrides: {sorted(enabled_redknot)}")
    root = Path(runtime_root or REPO_ROOT).expanduser().resolve()
    if not (root / "examples" / "_config.py").is_file() or not (root / "deepinfer" / "engine" / "llm_engine.py").is_file():
        raise ValueError("runtime_root must contain native DeepInfer and examples/_config.py")
    runtime = source_identity(root)
    runtime["benchmark_harness_sha256"] = {path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    runtime["explicit_runtime_override"] = root != REPO_ROOT
    runtime["hardware_support"] = "caller_selected_runtime_unvalidated" if root != REPO_ROOT else "upstream_nvidia_sm100"
    hardware = _hardware_inventory()
    devices = _select_devices(hardware, dp * cp, os.environ.copy())
    selected_hardware = [gpu for gpu in hardware["gpus"] if gpu["index"] in devices]
    if root == REPO_ROOT and any(not gpu["compute_capability"].startswith("10.") for gpu in selected_hardware):
        raise RuntimeError(
            "this checkout's native runtime targets SM100; SM90/H200 requires a separately validated port via --runtime-root"
        )
    hardware["selected_device_indices"] = devices
    effective_environment = {key: value for key, value in os.environ.items() if key.startswith("DEEPINFER_")}
    effective_environment.update(BASELINE_ENV)
    effective_environment.update(
        {
            "DEEPINFER_BATCH_INVARIANT": str(int(batch_invariant)),
            "DEEPINFER_ACCELERATOR": "nvidia",
            "DEEPINFER_SAMPLER_FORCE_TOPK": "4096",
            "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE": "3",
        }
    )
    environment = {
        **effective_environment,
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_free_master_port()),
        "DEVICE_OFFSET": str(devices[0]),
    }
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    spec = {
        "output_dir": str(directory),
        "runtime_root": str(root),
        "model_path": checkpoint["checkpoint_path"],
        "ep": ep,
        "cp": cp,
        "dp": dp,
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": checkpoint["required_context"],
        "max_num_seqs": max(len(manifest["warmup"]), len(manifest["cases"])),
        "memory_utilization": float(memory_utilization),
        "batch_invariant": batch_invariant,
        "supervised_process_group": supervised_process_group,
        "seed": seed,
        "environment": environment,
    }
    if admission_batch_size is not None:
        spec.update(admission_batch_size=admission_batch_size, admission_policy=ADMISSION_POLICY)
        active_warmup_count = len(manifest["warmup"]) if warmup_iterations else 0
        spec["max_num_seqs"] = min(admission_batch_size, max(active_warmup_count, len(manifest["cases"])))
    spec["timeout_seconds"] = float(timeout_s)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "launch.json", spec)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_rank_worker, args=(rank, manifest, spec), daemon=False) for rank in range(dp)]
    try:
        for process in processes:
            process.start()
        started = time.monotonic()
        while True:
            failures = list(directory.glob("rank-*.failure.json"))
            failed_exits = [process.exitcode for process in processes if process.exitcode not in (None, 0)]
            if failures or failed_exits:
                raise RuntimeError(f"native benchmark worker failed; inspect {directory}/rank-*.log and failure JSON")
            if all(process.exitcode == 0 for process in processes):
                break
            if time.monotonic() - started > timeout_s:
                raise TimeoutError(f"native benchmark exceeded {timeout_s}s; no successful report was produced")
            time.sleep(0.1)
        for process in processes:
            process.join()
        rank_results = [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(dp)]
        versions = [result.get("runtime_versions") for result in rank_results]
        if not versions[0] or any(version != versions[0] for version in versions):
            raise RuntimeError("ranks reported missing or inconsistent runtime versions")
        runtime["versions"] = versions[0]
        native_engines = [result.get("native_engine") for result in rank_results]
        if not native_engines[0] or any(engine != native_engines[0] for engine in native_engines):
            raise RuntimeError("ranks reported missing or inconsistent native engine binary identities")
        runtime["native_engine"] = native_engines[0]
        phases, requests = aggregate_ranks(
            rank_results,
            manifest,
            iterations=iterations,
            dp_size=dp,
            admission_batch_size=admission_batch_size,
        )
        elapsed_s = sum(phase["elapsed_s"] for phase in phases)
        report = {
            "schema": RUN_SCHEMA,
            "status": "complete",
            "mode": "baseline",
            "manifest_sha256": manifest["sha256"],
            "comparison_eligible": warmup_iterations > 0,
            "activation": {"redknot_enabled": False, "verified": True, "implementation": "deepinfer-native"},
            "metadata": {
                "model": {
                    **manifest["model"],
                    "checkpoint_config_sha256": checkpoint["config_sha256"],
                    "weight_files": checkpoint["weight_files"],
                    "weight_index_sha256": checkpoint["weight_index_sha256"],
                    "weight_content_hashing": "not_performed",
                },
                "tokenizer": checkpoint["tokenizer"],
                "runtime": runtime,
                "hardware": hardware,
                "settings": {
                    "ep_size": ep,
                    "cp_size": cp,
                    "dp_size": dp,
                    "iterations": iterations,
                    "replication": "full_cohort_per_dp_rank",
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "max_model_len": checkpoint["required_context"],
                    "max_num_seqs": spec["max_num_seqs"],
                    "memory_utilization": float(memory_utilization),
                    "batch_invariant": batch_invariant,
                    "supervised_process_group": supervised_process_group,
                    "process_group_owner": "external" if supervised_process_group else "runner",
                    "environment": effective_environment,
                },
                "sampling": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "seed": seed,
                    "seq_seed_method": "seed_plus_first_64_bits_of_sample_id_digest",
                    "max_new_tokens": "per_case",
                    "ignore_eos": "per_case",
                },
                "warmup": {
                    "iterations": warmup_iterations,
                    "cases_per_rank": len(manifest["warmup"]),
                    "complete": warmup_iterations > 0,
                    "state": "warm" if warmup_iterations else "cold",
                },
                "measurement_boundary": BOUNDARY,
            },
            "phases": phases,
            "requests": requests,
            "elapsed_s": elapsed_s,
            "completed_requests": len(requests),
        }
        from .metrics import summarize_requests

        if admission_batch_size is not None:
            report["metadata"]["settings"].update(admission_batch_size=admission_batch_size, admission_policy=ADMISSION_POLICY)
        report["summary"] = summarize_requests(requests, elapsed_s=elapsed_s)
        from .compare import validate_run

        report = seal(report)
        validate_run(report, manifest, mode="baseline", require_warmup=warmup_iterations > 0)
        return write_json(directory / "report.json", report)
    except BaseException as exc:
        _stop_workers(processes, supervised_process_group=supervised_process_group)
        failure_path = directory / "failure.json"
        if not failure_path.exists():
            write_json(
                failure_path,
                {"schema": RUN_SCHEMA, "status": "failed", "mode": "baseline", "manifest_sha256": manifest["sha256"], "error": str(exc)},
            )
        raise
