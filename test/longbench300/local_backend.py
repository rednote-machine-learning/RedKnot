"""Run paired Full/S1.6 inference on frozen LongBench questions on eight local SM90 GPUs.

The default uses this checkout's S1.6 implementation and a local checkpoint.
Use Python 3.12 with DeepInfer's GPU dependencies installed. For an optional
separate S1.6 installation --runtime-root accepts its directory layout or a directory containing
redknot-runtime.json. For a custom layout this index maps "producer", "runner",
"kernel", "template", and "source_root" to local paths relative to the index.
There is no network transport. CPU preflight never requests GPU ownership.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

REVISION = "s16-sm90-runtime-m-asm-ask-v1"


def load_v2_policy_config_class():
    name = "redknot_v2_policy"
    module = sys.modules.get(name)
    if module is None:
        fixed_root = Path(__file__).resolve().parent / "algorithm" / "fixed"
        spec = importlib.util.spec_from_file_location(name, fixed_root / "v2_policy.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return module.V2PolicyConfig


def normalize_v2_config(v2_config):
    cls = load_v2_policy_config_class()
    if v2_config is None:
        return cls.default()
    if isinstance(v2_config, cls):
        return v2_config
    if isinstance(v2_config, dict):
        return cls.from_dict(v2_config)
    if hasattr(v2_config, "to_dict"):
        return cls.from_dict(v2_config.to_dict())
    raise ValueError(f"Unsupported RedKnot V2 policy configuration: {type(v2_config).__name__}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def save(path, obj):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(f.name, path)


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover(root):
    root = Path(root).resolve(strict=True)
    descriptor = root / "redknot-runtime.json"
    if descriptor.exists():
        values = read(descriptor)
        return {k: (root / values[k]).resolve(strict=True)
                for k in ("producer", "runner", "kernel", "template", "source_root")}
    suffix = "dynamic-gemm-v1/code-r1/production/run_fixed.py"
    candidates = [p for depth in ("", "*/", "*/*/") for p in root.glob(depth + suffix)]
    require(len(candidates) == 1, "Select one installed LongBench S1.6 runtime, or supply its redknot-runtime.json")
    producer = candidates[0].resolve()
    bench = producer.parents[3]
    return dict(producer=producer, runner=bench/"sources/runtime/run_s16.py",
                kernel=producer.parents[1]/"kernel", source_root=bench.parent.parent,
                template=bench.parent/"rag-cone15-finance-v2/queue-v1/attempts/finance-000/runner-config.json")


def relocate(value, mappings):
    if isinstance(value, dict):
        return {relocate(k, mappings): relocate(v, mappings) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, mappings) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        for old, new in mappings:
            if Path(value).is_relative_to(old):
                return str(new / Path(value).relative_to(old))
    return value


def verify_checkpoint(model, service):
    for name, expected in service["checkpoint_sha256"].items():
        require(hash_file(model/name) == expected, f"Checkpoint identity mismatch: {name}")
    if service.get("encoder_sha256"):
        require(hash_file(model/"encoding/encoding.py") == service["encoder_sha256"], "Checkpoint encoder mismatch")
    weights = set(read(model/"model.safetensors.index.json")["weight_map"].values())
    require(weights and all((model/w).is_file() and (model/w).stat().st_size > 0 for w in weights),
            "Checkpoint is missing weight shards")


def bundled_runtime(model, output, *, v2_config=None, selected_groups=None):
    if __package__:
        from .native_assets import load_native_assets
    else:
        from native_assets import load_native_assets
    algorithm = Path(__file__).resolve().parent / "algorithm"
    native_root, service = load_native_assets(model, output)
    native_root = Path(native_root).resolve(strict=True)
    verify_checkpoint(model, service)
    # Record live file digests for run provenance without blocking open-source source edits.
    source_hashes = {}
    for source in sorted(algorithm.rglob("*.py")):
        resolved = source.resolve(strict=True)
        source_hashes[str(resolved)] = hash_file(resolved)
    for rel in (
        "GatedEngram/gpu_pilot/cuda_mask_qualification.py",
        "GatedEngram/gpu_pilot/index_hit20_empty_ep_qualification.py",
        "GatedEngram/gpu_pilot/read_s11_results.py",
        "GatedEngram/kv_cache_engram/__init__.py",
        "GatedEngram/kv_cache_engram/cache.py",
        "GatedEngram/kv_cache_engram/calibration.py",
        "GatedEngram/kv_cache_engram/core.py",
        "GatedEngram/kv_cache_engram/gate.py",
        "GatedEngram/kv_cache_engram/runtime.py",
        "NativeSnapshot/deepinfer/model_executor/layers/sm90/gemm.py",
        "redknot/__init__.py",
        "redknot/decode_evidence.py",
    ):
        source = (native_root / rel).resolve(strict=True)
        source_hashes[str(source)] = hash_file(source)
    launcher = (Path(service["experiment_root"]) / "kv_recompression_20261029/quality_process_group.py").resolve(strict=True)
    require(launcher.is_file(), "Missing local multi-process launcher")
    policy_cfg = normalize_v2_config(v2_config)
    config = dict(service_config=service, output_dir=str(output), source_sha256=source_hashes,
                  strict_source_hashes=False,
                  native_source_root=str(native_root), fixed_pack_runtime_root=str(algorithm/"fixed"),
                  fixed_pack_shared_root=str(algorithm/"shared"), frozen_runtime_root=str(algorithm/"lastkv"),
                  rag_runtime_root=str(algorithm/"rag"), multiowner_v1_root=str(algorithm/"multiowner"),
                  admission_v2_root=str(algorithm/"admission"), fixed_pack_budget_bytes=1024**3,
                  multiowner_active_ranks=[0, 1, 2, 3], target_batch_sizes=[4], padding_allowed=False,
                  selector_strategy="max", dspark_nextn=5, main_and_draft_graphs_required=True,
                  redknot_v2=policy_cfg.to_dict())
    if selected_groups is not None:
        config["selected_groups"] = list(selected_groups)
    config["sha256"] = digest(config)
    locations = dict(producer=str(algorithm/"producer/run_fixed.py"), runner=str(algorithm/"producer/run_s16.py"),
                     kernel=str(algorithm/"kernel"), launcher=str(launcher))
    return config, locations


def load_runtime(root, model, output, *, v2_config=None, selected_groups=None):
    model = Path(model).resolve(strict=True)
    if root is None:
        return bundled_runtime(model, output, v2_config=v2_config, selected_groups=selected_groups)
    paths = discover(root)
    for filename in (paths["producer"], paths["runner"],
                     paths["kernel"]/"install_dynamic_gemm.py",
                     paths["kernel"]/"dynamic_kernel.py"):
        require(filename.is_file(), f"Missing runtime source file: {filename}")
    producer = imported(paths["producer"], "redknot_installed_producer")
    base = producer.load_base(paths["runner"])
    original = base.load_template(paths["template"])
    old_source = Path(original["native_source_root"]).parent
    mapping = [(old_source, paths["source_root"])]
    ref = relocate(original["service_config_ref"], mapping)
    service = read(ref["path"])
    mappings = [(Path(service["model_path"]), model), *mapping]
    service = relocate(service, mappings)
    verify_checkpoint(model, service)
    svcroot = Path(service["experiment_root"])
    launcher = (svcroot / "kv_recompression_20261029" / "quality_process_group.py").resolve(strict=True)
    require(launcher.is_file(), "Missing local native eight-rank launcher")
    relocated = relocate(original, mappings)
    # Keep runtime and numerical settings. Historical evaluation and queue metadata
    # are not native model inputs. Source keys refer to the local installation.
    keys = ("native_source_root", "fixed_pack_runtime_root", "fixed_pack_shared_root",
            "frozen_runtime_root", "rag_runtime_root", "multiowner_v1_root", "admission_v2_root",
            "fixed_pack_budget_bytes", "multiowner_active_ranks", "target_batch_sizes", "padding_allowed",
            "selector_strategy", "dspark_nextn", "main_and_draft_graphs_required")
    config = {k: relocated[k] for k in keys}
    config.update(service_config=service, output_dir=str(output),
                  source_sha256=relocated.get("source_sha256", {}),
                  strict_source_hashes=False)
    if v2_config is not None:
        config["redknot_v2"] = normalize_v2_config(v2_config).to_dict()
    if selected_groups is not None:
        config["selected_groups"] = list(selected_groups)
    config["sha256"] = digest(config)
    locations = {k: str(v) for k, v in paths.items() if k != "source_root"}
    locations["launcher"] = str(launcher)
    return config, locations


def prepare_native(config):
    native = Path(config["native_source_root"])
    sys.path[:0] = [str(native/"NativeSnapshot"), str(native), str(native/"GatedEngram"), config["fixed_pack_runtime_root"]]
    import fixed_pack_runtime_adapter as adapter
    preserved, _ = adapter.load_dependencies(config)
    adapter.private_preserved_module(preserved)
    require("torch" not in sys.modules and "deepinfer" not in sys.modules, "Model import before GPU release")


def local_modules(config, locations):
    """Bind portable template I/O; all actual model computation belongs to the installed producer."""
    producer = imported(locations["producer"], "redknot_installed_producer")
    base = producer.load_base(locations["runner"])

    def local_template(path):
        value = read(path)
        signed = value.pop("sha256")
        require(signed == config["sha256"] == digest(value), "Local execution settings changed")
        value["sha256"] = signed
        return value

    base.load_template = local_template
    producer.load_base = lambda path: base
    original_identity = producer.identity

    def local_identity(base, runner, kernel, data):
        return dict(original_identity(base, runner, kernel, data),
                    local_config_sha256=config["sha256"], local_adapter_sha256=hash_file(__file__))

    producer.identity = local_identity
    return base, producer


# Both workloads use the same native model and scheduler. Bind the logical
# request/target order and reusable token spans independently of record compression.
WORKLOADS = {
    300: (75, "0deffb2e243e784cf02476ff090c18aaa4d66dae400a0bfcfaf9b9ac75020955",
          "b483befdfba1e2985fd8044829a27d1582e881ed089d50d8d9cd1d111a9f8852"),
    2100: (527, "bcb1559ffde66ab95ec6b7b13d7889b90bce272cae6b4b53a71f6dfb99a0cebe",
           "fcaccd37391e186ce090d4afa20669733cf2225e657e6afb7a7027c1a336be5f"),
}


def check_dataset(base, data, target_count=300):
    group_count, plan_digest, documents_digest = WORKLOADS[target_count]
    terminal = read(data / "complete.json")
    require((terminal["group_count"], terminal["target_count"]) == (group_count, target_count),
            f"Expected {target_count} targets in {group_count} original batches")
    names = [f"lb-{i:04d}" for i in range(group_count)]
    require(sorted(p.stem for p in (data/"groups").glob("*.json")) == names, "Invalid original B4 inventory")
    plan, documents_by_id, targets = [], {}, set()
    for name in names:
        group, rows = base.group_rows(data, data/"groups"/(name+".json"))
        ids = [row["index"] for row in rows]
        require(targets.isdisjoint(group["target_indices"]), f"Target counted twice in {name}")
        targets.update(group["target_indices"])
        plan.append([name, ids, group["target_indices"], group["native_chunk_counts"]])
        for row in rows:
            spans = [[d["input_start"], d["input_end"], d["token_ids_sha256"]] for d in row["documents"]]
            item = [row["index"], row["input_ids_sha256"], spans]
            require(documents_by_id.setdefault(row["index"], item) == item, "Companion input mismatch")
    expected = {1000 * dataset + slot for dataset in range(1, target_count // 100 + 1) for slot in range(100)}
    require(targets == expected == set(documents_by_id) and digest(plan) == plan_digest,
            "Unexpected target membership or original group order")
    require(digest([documents_by_id[i] for i in sorted(documents_by_id)]) == documents_digest,
            "Frozen prefill token IDs or reusable document spans changed")
    return group_count


def rank_worker(options, locations, contract, attempt, world, rank):
    cfg = read(options["template"])
    prepare_native(cfg)
    _, producer = local_modules(cfg, locations)
    producer.worker(options, contract, attempt, world, rank)


def driver_compute_capabilities():
    """Read all CUDA-visible devices through the driver without creating a CUDA context."""
    driver = ctypes.CDLL("libcuda.so.1")
    int_pointer = ctypes.POINTER(ctypes.c_int)
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [int_pointer],
        "cuDeviceGet": [int_pointer, ctypes.c_int],
        "cuDeviceGetAttribute": [int_pointer, ctypes.c_int, ctypes.c_int],
    }
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int

    def call(name, *arguments):
        result = getattr(driver, name)(*arguments)
        if result != 0:
            raise RuntimeError(f"CUDA driver {name} failed with CUresult {result}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    require(count.value == 8, f"CUDA driver sees {count.value} devices; this benchmark requires eight")
    capabilities = []
    for ordinal in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(device), ordinal)
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR from cuda.h.
        call("cuDeviceGetAttribute", ctypes.byref(major), 75, device)
        call("cuDeviceGetAttribute", ctypes.byref(minor), 76, device)
        capabilities.append((major.value, minor.value))
    return capabilities


def ensure_eight_idle():
    def query(field):
        cmd = ["nvidia-smi", field, "--format=csv,noheader,nounits"]
        return subprocess.check_output(cmd, text=True, timeout=12).splitlines()
    physical = dict(map(str.strip, row.split(",")) for row in query("--query-gpu=index,uuid"))
    # The pinned native launcher assigns rank r to CUDA device r. Its environment
    # setup replaces CUDA_VISIBLE_DEVICES; validate exactly the devices it will use.
    require(len(physical) == len(set(physical.values())) == 8 and int(os.environ.get("DEVICE_OFFSET", "0")) == 0,
            "Expose exactly eight assigned GPUs to nvidia-smi and use DEVICE_OFFSET=0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    request = [s.strip() for s in visible.split(",")] if visible is not None else list(physical)
    selected = [physical.get(x, x) for x in request]
    require(len(selected) == 8 and set(selected) == set(physical.values()),
            "This benchmark requires exactly eight visible GPUs; check CUDA_VISIBLE_DEVICES")
    capabilities = driver_compute_capabilities()
    require(all(cc == (9, 0) for cc in capabilities),
            f"This benchmark requires SM90 on all eight devices; CUDA driver reports {capabilities}")
    active = [(pid, uuid) for pid, uuid in (map(str.strip, line.split(","))
              for line in query("--query-compute-apps=pid,gpu_uuid")) if uuid in selected]
    require(not active, f"Selected GPUs are busy. No jobs were stopped. Active compute processes: {active}")


def native_child(config_path, locations_path, data, output, attempt):
    cfg, loc = read(config_path), read(locations_path)
    prepare_native(cfg)
    base, producer = local_modules(cfg, loc)
    stamp = producer.identity(base, Path(loc["runner"]), Path(loc["kernel"]), data)
    contract = base.digest(stamp)
    ready, release = attempt/"ready.json", attempt/"release.json"
    save(ready, dict(pid=os.getpid(), contract_id=contract))
    deadline = time.monotonic() + 120
    parent = os.getppid()
    while not release.is_file():
        require(os.getppid() == parent and os.getppid() != 1 and time.monotonic() < deadline,
                "No matching GPU release received")
        time.sleep(.2)
    require(read(release) == read(ready), "Wrong GPU release identity")
    sys.path.insert(0, str(Path(loc["launcher"]).parent))
    launch = imported(loc["launcher"], "redknot_singlehost").launch
    meta = output/"gemm-revision.json"
    if meta.exists():
        old = read(meta)
        seal = old.pop("sha256")
        require(seal == base.digest(old) and old == dict(contract_id=contract, **stamp), "Incompatible prior outputs")
    else:
        require(not (output/"groups").exists(), "Unbound existing output groups")
        base.save(meta, dict(contract_id=contract, **stamp))
    options = dict(template=str(config_path), runner=loc["runner"], kernel=loc["kernel"], data=str(data), out=str(output))
    launch(functools.partial(rank_worker, options, loc, contract, attempt), 8, attempt)
    terminal = read(output/"run.json")
    selected_groups = cfg.get("selected_groups")
    if selected_groups is not None:
        require((terminal["status"], terminal["group_count"], terminal["contract_id"])
                == ("complete", len(selected_groups), contract), "Incomplete native GPU subset run")
    else:
        manifest = read(data/"complete.json")
        require((terminal["status"], terminal["group_count"], terminal["target_count"], terminal["contract_id"])
                == ("complete", manifest["group_count"], manifest["target_count"], contract), "Incomplete native GPU run")


def release_child(config_path, locations_path, data, output, contract):
    # A shared lock among invocations on this machine and a last-moment GPU
    # occupancy check keep the runner from displacing other compute workloads.
    lockpath = Path(tempfile.gettempdir()) / "redknot-s16-singlehost-gpu.lock"
    with lockpath.open("a+") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_eight_idle()
        attempt = output/"attempts"/f"run-{time.time_ns()}"
        attempt.mkdir(parents=True)
        cmd = [sys.executable, "-B", "-u", __file__, "_native", str(config_path), str(locations_path),
               str(data), str(output), str(attempt)]
        native_root = Path(read(config_path)["native_source_root"])
        env = dict(os.environ, DEEPINFER_ACCELERATOR="nvidia")
        search = [str(native_root/"NativeSnapshot"), str(native_root), str(native_root/"GatedEngram")]
        if env.get("PYTHONPATH"):
            search.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(search)
        with (attempt/"worker.log").open("xb") as log:
            # SIGTERM must unwind the owner so only this fresh process group is drained.
            def cancelled(signum, frame):
                raise InterruptedError("Local benchmark terminated")
            old_handler = None
            if threading.current_thread() is threading.main_thread():
                old_handler = signal.signal(signal.SIGTERM, cancelled)
            proc = None
            try:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
                deadline = time.monotonic() + 600
                while not (attempt/"ready.json").is_file():
                    require(proc.poll() is None and time.monotonic() < deadline, f"Preflight failed; see {log.name}")
                    time.sleep(.25)
                ready = read(attempt/"ready.json")
                require(ready == dict(pid=proc.pid, contract_id=contract), "Child identity changed")
                ensure_eight_idle()
                save(attempt/"release.json", ready)
                require(proc.wait() == 0, f"Native GPU inference failed; see {log.name}")
            finally:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                if old_handler is not None:
                    signal.signal(signal.SIGTERM, old_handler)


def main():
    if sys.argv[1:2] == ["_native"]:
        native_child(*map(Path, sys.argv[2:]))
        return
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runtime-root", type=Path, default=os.environ.get("REDKNOT_RUNTIME_ROOT"))
    cli.add_argument("--data", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--model-path", required=True, type=Path)
    cli.add_argument("--check-only", action="store_true")
    cli.add_argument("--redknot-mode", "--mode", dest="redknot_mode", choices=("v2.1-fast", "s1.6"), default="v2.1-fast")
    cli.add_argument("--two-stage-l14", action="store_true", default=False)
    cli.add_argument("--preset", choices=("v2.1-fast", "short20_long15", "v2.3_12", "v2.3_10", "m16", "s1.6"), default=None)
    cli.add_argument("--span-min-tokens", type=int, default=48_000)
    cli.add_argument("--host-prefix-offload", choices=("auto", "on", "off"), default="auto")
    cli.add_argument("--audit-halo", action="store_true", default=False)
    opt = cli.parse_args()
    v2_cls = load_v2_policy_config_class()
    v2_cfg = v2_cls.from_cli(
        mode=opt.redknot_mode,
        two_stage_l14=opt.two_stage_l14,
        preset=opt.preset,
        span_min_tokens=opt.span_min_tokens,
        host_prefix_offload=opt.host_prefix_offload,
        audit_halo=opt.audit_halo,
    )
    print(json.dumps(run_local(opt.runtime_root, opt.model_path, opt.data, opt.output, opt.check_only, v2_config=v2_cfg)))


def run_local(runtime_root, model_path, data, output, check_only=False, *, target_count=300, v2_config=None, selected_groups=None):
    """Validate or execute the local benchmark. Raw Full and sparse pairs are under output/groups/."""
    require(model_path, "Set --model-path or REDKNOT_MODEL_PATH to the matching local DeepSeek V4.1 Flash checkpoint")
    data, out = Path(data).resolve(strict=True), Path(output).resolve()
    require(not out.is_relative_to(data) and not data.is_relative_to(out), "Choose independent data and output paths")
    if selected_groups is not None:
        selected_groups = [str(g).strip() for g in selected_groups if str(g).strip()]
        require(bool(selected_groups) and len(selected_groups) == len(set(selected_groups)),
                "selected_groups must be a nonempty list of distinct group IDs")
        for gid in selected_groups:
            require((data / "groups" / f"{gid}.json").is_file(), f"Unknown B4 group ID: {gid}")
    out.mkdir(parents=True, exist_ok=True)
    with (out/".runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (out/"run.json").exists() or check_only, "Completed run already exists; choose a new output")
        cfg, loc = load_runtime(runtime_root, model_path, out, v2_config=v2_config, selected_groups=selected_groups)
        prepare_native(cfg)
        base, producer = local_modules(cfg, loc)
        group_count = check_dataset(base, data, target_count)
        effective_groups = len(selected_groups) if selected_groups is not None else group_count
        contract = base.digest(producer.identity(base, Path(loc["runner"]), Path(loc["kernel"]), data))
        if __package__:
            from .sm90_source import build_extension, extension_path
        else:
            from sm90_source import build_extension, extension_path
        snapshot = Path(cfg["native_source_root"]) / "NativeSnapshot"
        extension = extension_path(snapshot)
        extension_ready = extension.is_file() and extension.stat().st_size > 0
        if check_only:
            return dict(status="cpu_pass", group_count=effective_groups, target_count=target_count, model_calls=0, contract_id=contract,
                        sm90_extension_ready=extension_ready, redknot_v2=cfg.get("redknot_v2"), selected_groups=selected_groups)
        if not extension_ready and runtime_root is None:
            build_extension(snapshot)
        require(extension.is_file() and extension.stat().st_size > 0,
                f"The installed DeepInfer runtime needs the SM90 extension for this Python: {extension}")
        config_path, locations_path = out/"local-config.json", out/"runtime-locations.json"
        save(config_path, cfg)
        save(locations_path, loc)
        release_child(config_path, locations_path, data, out, contract)
        return dict(status="complete", group_count=effective_groups, target_count=target_count, contract_id=contract,
                    results=str(out), redknot_v2=cfg.get("redknot_v2"), selected_groups=selected_groups)


if __name__ == "__main__":
    main()
