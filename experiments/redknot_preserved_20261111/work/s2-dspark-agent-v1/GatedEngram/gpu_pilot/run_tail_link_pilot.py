"""Finite independent tail-link token-level B pilot.

Import and check-only are CPU-only. The original supervisor releases CUDA only
after immutable preflight. No v8 module or native source is modified.
"""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
import tempfile

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from gpu_pilot import run_pilot as support

SCHEMA = "kv-cache-engram-tail-link-pilot-run-v1"
PHASE_SCHEMA = "kv-cache-engram-tail-link-phase-v1"
ARMS = ("recomputed", "tail128_link", "tail128_dynamic")
FRACTIONS = dict.fromkeys(ARMS)
EVALUATION_INDICES = (2, 3, 4, 5)
CAPTURE_LAYERS = tuple(range(40))
FFN_LAYERS = tuple(range(3,40))
NATIVE_SNAPSHOT = STAGE.parent / "NativeSnapshot"


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    """Publish immutable evidence atomically; existing evidence is never replaced."""
    path = Path(path)
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink(),
            "evidence target already exists or parent is missing")
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # link is no-replace, unlike rename/replace. Same-directory temporary
        # gives a single-filesystem atomic publication and leaves failed data.
        os.link(temporary, path)
        os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Preserve the unique pending file for diagnostics/recovery.
        raise


def evidence_ref(directory, name, expected_seal):
    require(type(name) is str and Path(name).name == name and name.endswith(".json"),
            "evidence reference must be a plain JSON filename")
    path = Path(directory)/name
    require(path.is_file() and not path.is_symlink(), "raw evidence must exist before closure")
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024**2), b""):
            sha.update(block)
    return dict(file=name, file_sha256=sha.hexdigest(), seal_sha256=expected_seal)


def bind_native_snapshot(args):
    require(args.repo.resolve() == NATIVE_SNAPSHOT.resolve()
            and args.repo.is_dir() and not args.repo.is_symlink(),
            "new pilot requires its isolated native snapshot")
    live = (args.experiment_root/"DeepInfer-PR148").resolve()
    def live_path(item):
        if not item:
            return False
        path = Path(item).resolve()
        allowed = (STAGE.resolve(), args.repo.resolve())
        if any(path == root or root in path.parents for root in allowed):
            return False
        return path == live or live in path.parents
    sys.path[:] = [p for p in sys.path if not live_path(p)]
    for folder in (args.repo/"examples", args.repo):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    verify_native_imports(args.repo)


def verify_native_imports(repo):
    root = Path(repo).resolve()
    for name, module in list(sys.modules.items()):
        if name == "deepinfer" or name.startswith("deepinfer.") or name == "_config":
            origin = getattr(module, "__file__", None)
            if origin is None:
                continue
            require(root in Path(origin).resolve().parents,
                    "native module imported outside isolated snapshot: "+name)


def native_yoco_swa_configuration(repo, model_path):
    """Read the frozen HF default and every config override without importing CUDA."""
    field = "yoco_swa_share_hidden"
    relative = "deepinfer/hugging_face/config.py"
    source = (Path(repo)/relative).read_bytes()
    tree = ast.parse(source)
    defaults = [node.value for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(isinstance(target, ast.Name) and target.id == "_GPT_DEFAULTS"
                        for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    require(len(defaults) == 1 and isinstance(defaults[0], ast.Dict),
            "frozen HF parser must have one literal _GPT_DEFAULTS dictionary")
    entries = [(key, value) for key, value in zip(defaults[0].keys, defaults[0].values)
               if isinstance(key, ast.Constant) and key.value == field]
    require(len(entries) == 1 and isinstance(entries[0][1], ast.Constant)
            and entries[0][1].value is False, "frozen HF native YOCO SWA default must be false")
    # The pinned parser has exactly one default consumer and no field-specific
    # override. Reject future parser rewrites instead of guessing precedence.
    references = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
                  and node.id == "_GPT_DEFAULTS" and isinstance(node.ctx, ast.Load)]
    consumers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "AttrDict"
                 and len(node.args) == 1 and node.args[0] in references
                 and all(keyword.arg is not None for keyword in node.keywords)]
    require(len(references) == len(consumers) == 1,
            "unreviewed HF defaults consumer or dynamic override")
    require(sum(isinstance(node, ast.Constant) and node.value == field for node in ast.walk(tree)) == 1
            and not any((isinstance(node, ast.keyword) and node.arg == field)
                        or (isinstance(node, ast.Attribute) and node.attr == field) for node in ast.walk(tree)),
            "unreviewed HF parser YOCO SWA override")
    config_raw = (Path(model_path)/"config.json").read_bytes()
    config = json.loads(config_raw)
    require(isinstance(config, dict) and config.get("model_type") == "deepseek_v41"
            and isinstance(config.get("text_config"), dict), "expected frozen DeepSeek V4.1 HF config")
    overrides = []
    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                child = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key == field:
                    require(item is False, "checkpoint YOCO SWA override must be false: " + child)
                    overrides.append(dict(path=child, value=False))
                visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + "/" + str(index))
    visit(config, "")
    return dict(schema="tail-link-native-yoco-swa-configuration-v1",
                native_yoco_swa_share_hidden=False, hf_default=False,
                hf_parser_file=relative, hf_parser_sha256=hashlib.sha256(source).hexdigest(),
                checkpoint_config_sha256=hashlib.sha256(config_raw).hexdigest(),
                config_overrides=overrides, parser_override_absent=True, verified=True)


def source_hashes(root):
    from gpu_pilot.tail_link_protocol import DEPENDENCIES
    require(DEPENDENCIES, "new tail-link dependency pins must be frozen before preflight")
    hashes = support.source_hashes(root)
    for name, expected in DEPENDENCIES.items():
        path = (STAGE/name.removeprefix("GatedEngram/")) if name.startswith("GatedEngram/") else Path(root)/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, "frozen sparse operator dependency changed: "+name)
        hashes[name] = actual
    return hashes


def parse_args(argv=None):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--tail-link-protocol", required=True, type=Path)
    known, rest = extra.parse_known_args(argv)
    args = support.parse_args(rest)
    args.tail_link_protocol = known.tail_link_protocol
    return args


def validate_protocol(args, manifest):
    from gpu_pilot.tail_link_protocol import tail_link_protocol
    value = json.loads(args.tail_link_protocol.read_bytes())
    require(value == tail_link_protocol(), "tail-link protocol differs from frozen CPU-tested rules")
    require(manifest.get("sha256") == value["input_manifest_sha256"],
            "tail-link pilot must use the original fixed exploratory input manifest")
    if getattr(args, "sensitivity_protocol", None) is not None:
        support.validate_sensitivity_inputs(args, manifest)
    else:
        require("sensitivity_protocol_sha256" not in manifest,
                "inherited sensitivity manifest requires explicit source protocol")
    return value


def selected_rows(manifest):
    by_index = {row["index"]: row for row in manifest["cohort"]}
    require(all(i in by_index for i in EVALUATION_INDICES), "fixed tail-link pilot cases absent")
    rows = [by_index[i] for i in EVALUATION_INDICES]
    require([r["dataset"] for r in rows] == ["hotpotqa", "2wikimqa"] * 2, "fixed dataset order changed")
    return rows


def timing_schedule(index):
    require(type(index) is int and index in EVALUATION_INDICES, "unregistered evaluation case")
    return [(repeat, ARMS[(index + repeat + slot) % len(ARMS)])
            for repeat in range(3) for slot in range(len(ARMS))]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def qualification_row(manifest):
    original = manifest["warmup"][0]
    require(original["index"] == 0 and len(original["input_ids"]) >= 16384,
            "qualification requires the declared calibration case0 prefix")
    ids = original["input_ids"][:16384]
    token_hash = _digest(ids)
    return dict(index=0, dataset=original["dataset"], input_ids=ids, input_tokens=16384,
                input_ids_sha256=token_hash, sample_id="qualification-"+token_hash,
                references=["__operator_qualification_not_quality__"], documents=None)


def verify_phase(value, *, index, arm, timing, quality):
    require(isinstance(value, dict) and value.get("sha256") == _digest(
        {k: v for k, v in value.items() if k != "sha256"}), "phase seal mismatch")
    require(value.get("schema") == PHASE_SCHEMA and value.get("index") == index
            and value.get("arm") == arm and value.get("timing_valid") is timing
            and value.get("quality_eligible") is quality, "phase identity/eligibility mismatch")
    require(value.get("dynamic_document_swa") is (arm == "tail128_dynamic"),
            "phase dynamic policy does not match arm")
    require(value.get("native_yoco_swa_share_hidden") is False,
            "phase native YOCO SWA configuration differs")
    record = value["record"]
    require(record.get("success") is True and record.get("rank") == 0
            and record.get("input_ids_sha256") == value.get("input_ids_sha256"),
            "phase has no complete bound rank-zero request")
    require(record.get("output_ids_sha256") == _digest(record.get("output_ids")), "output token hash changed")
    if timing:
        require(len(record["output_ids"]) == 32 and record.get("natural_EOS") is False,
                "formal timing must generate exactly32 tokens")
        for name in ("online_ttft_s", "online_elapsed_s", "online_prepare_to_submit_s"):
            number = record.get(name)
            require(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                    "invalid online timing: " + name)
        require(0 < record["online_ttft_s"] <= record["online_elapsed_s"]
                <= value["online_global_window_s"], "online timing ordering invalid")
    if quality:
        require(record.get("natural_EOS") is True and 0 < len(record["output_ids"]) <= 64
                and record.get("truncated") is (record.get("finish_reason") == "length"),
                "quality EOS/length binding mismatch")
        for name in ("answer_em", "answer_token_f1"):
            require(type(record.get(name)) in (int, float) and math.isfinite(record[name])
                    and 0 <= record[name] <= 1, "quality score invalid")
    return record


def qualification_passed(reference, candidate):
    from gpu_pilot.read_index_hit20_results import projection_identity, cache_restore_identity, audit_joint_attention
    require(reference.get("mode") == "qualified_native" and candidate.get("mode") == "all_selected",
            "new full/native qualification pair required")
    for value in (reference, candidate):
        verify_phase(value, index=0, arm="qualification", timing=False, quality=False)
    require(reference["input_ids_sha256"] == candidate["input_ids_sha256"]
            and reference["record"]["output_ids"] == candidate["record"]["output_ids"],
            "all-selected output differs from native qualification")
    for key in ("execution_configuration_sha256", "tail_link_protocol_sha256", "manifest_sha256"):
        require(reference[key] == candidate[key], "qualification execution/protocol changed")
    fingerprint = reference["controller"].get("identity_fingerprints_sha256")
    require(candidate["controller"].get("identity_exact") is True
            and type(candidate["controller"].get("identity_check_count")) is int
            and candidate["controller"]["identity_check_count"] == 80
            and reference["controller"].get("identity_check_count") == 80
            and isinstance(fingerprint, str) and len(fingerprint) == 64
            and candidate["controller"].get("identity_fingerprints_sha256") == fingerprint,
            "new per-layer output/lazy identity checks did not pass")
    projection_identity(candidate["controller"])
    cache_restore_identity(candidate["controller"])
    require(audit_joint_attention(candidate["controller"])==0,"all-selected qualification reused cached SWA")
    return True


def verify_partial_smoke(phases, row, qualification, sparse_qualification, *, arm="tail128_link"):
    """Require full input, independent SWA and FFN accounting before quality."""
    from gpu_pilot.read_tail_link_results import audit_controller, documents
    require(row["index"] == EVALUATION_INDICES[0] and len(row["documents"]) == 8,
            "smoke must use original complete eight-document case2")
    require(qualification.get("passed") is True and qualification.get("identity_check_count") == 80,
            "new strict numerical qualification must precede smoke")
    require(sparse_qualification.get("passed") is True
            and sparse_qualification.get("input_ids_sha256") == row["input_ids_sha256"],
            "independent sparse numerical qualification must precede smoke closure")
    require(arm in ARMS[1:] and len(phases) == 1 and phases[0].get("arm") == arm,
            "one registered tail policy smoke required")
    phase = phases[0]
    require(sparse_qualification.get("candidate_sha256") == phase.get("sha256")
            and sparse_qualification.get("qualification_sha256") == qualification.get("sha256"),
            "smoke is not bound to its own fresh same-policy numerical qualification")
    require(phase.get("empty_ep_qualification_sha256") == qualification.get("empty_ep_qualification_sha256")
            and isinstance(phase.get("empty_ep_qualification_sha256"), str),
            "partial smoke must bind the new native empty-EP qualification")
    record = verify_phase(phase, index=row["index"], arm=arm, timing=False, quality=False)
    require(phase.get("mode") == "sensitive" and phase.get("fraction") is None
            and phase.get("partial_smoke") is True
            and phase.get("input_ids_sha256") == row["input_ids_sha256"]
            and len(record["output_ids"]) == 1 and record.get("natural_EOS") is False,
            "partial smoke input/mode/output policy changed")
    require(phase["controller"].get("dynamic_document_swa") is (arm == "tail128_dynamic"),
            "smoke controller policy does not match arm")
    actual = audit_controller(phase["controller"], row, phase, documents(row))
    require(actual["ffn_bypassed_rows"] > 0 and actual["swa_reused_rows"] > 0,
            "smoke did not exercise real independent FFN and SWA reuse paths")
    if arm == "tail128_dynamic":
        require(actual.get("dynamic_refreshed_rows", 0) > 0,
                "dynamic smoke did not refresh actual document SWA rows")
    body = dict(schema="tail-link-partial-smoke-v1", passed=True, index=row["index"],
        phase_sha256={arm:phase["sha256"]}, input_ids_sha256=row["input_ids_sha256"],
        input_tokens=row["input_tokens"], qualification_sha256=qualification["sha256"],
        quality_eligible=False, timing_valid=False, accuracy_guarantee=False,
        full_input_original_eight_documents=True, independent_swa_and_ffn_paths_exercised=True,
        output_tokens_per_arm=1)
    body["sparse_qualification_sha256"] = sparse_qualification["sha256"]
    return dict(body, sha256=_digest(body))


def summarize_timing(phases, index):
    require([(p["repeat"], p["arm"]) for p in phases] == timing_schedule(index),
            "all three arms need three paired rotating repetitions")
    summaries = {}
    bindings = set()
    for arm in ARMS:
        records = []
        for phase in (p for p in phases if p["arm"] == arm):
            records.append(verify_phase(phase, index=index, arm=arm, timing=True, quality=False))
            bindings.add((phase["input_ids_sha256"], phase["execution_configuration_sha256"],
                          phase["manifest_sha256"], phase["tail_link_protocol_sha256"]))
        samples = [r["online_ttft_s"] for r in records]
        summaries[arm] = dict(online_ttft_samples_s=samples, online_ttft_p50_s=statistics.median(samples),
                              repetitions=3, fixed_output_tokens=32)
    require(len(bindings) == 1, "timing arms do not share frozen input/execution")
    baseline = summaries["recomputed"]["online_ttft_p50_s"]
    for arm in ARMS[1:]:
        candidate = summaries[arm]["online_ttft_p50_s"]
        summaries[arm].update(ttft_speedup=baseline/candidate, ttft_reduction_s=baseline-candidate,
                             ttft_reduction_percent=100*(1-candidate/baseline))
    return dict(arms=summaries, concurrency=1, active_dp_ranks=1, participating_ep_ranks=8,
                online_cpu_tail_prefetch_included=True, online_gather_scatter_and_fallback_included=True,
                model_load_included=False, offline_build_included=False, saturated_capacity_claim=False)


def verify_frozen(args, manifest, identity, profile, base, *, native_source=False):
    require(identity.get("native_yoco_swa_share_hidden") is False
            and native_yoco_swa_configuration(args.repo, args.model_path)
            == identity.get("native_yoco_swa_configuration"), "native YOCO SWA configuration changed")
    require(source_hashes(args.experiment_root) == identity["pilot_source_sha256"],
            "tail-link pilot source changed after preflight")
    require(profile.identity(args) == identity["native_profile"] and base.effective_env(args) == identity["fixed_env"],
            "native configuration drift")
    disk = base.read_json(args.manifest)
    support.validate_inputs(base, disk, base.read_json(args.pilot_protocol))
    protocol = validate_protocol(args, disk)
    require(disk["sha256"] == manifest["sha256"]
            and protocol["sha256"] == identity["tail_link_protocol_sha256"], "input/protocol changed")
    for filename, key in (("config.json", "checkpoint_config_sha256"),
                          ("model.safetensors.index.json", "checkpoint_index_sha256"),
                          ("tokenizer.json", "tokenizer_json_sha256")):
        require(base.file_digest(args.model_path/filename) == identity[key], "model/tokenizer identity changed")
    if native_source:
        verify_native_imports(args.repo)
        require(all(identity[k] == v for k, v in base.source_identity(args.repo).items()), "native source changed")


def worker(args, manifest, identity, dp_size, rank):
    bind_native_snapshot(args)
    base, native, NativeProfile, scoped_base_profile = support.load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args, apply_environment=True):
        require(not native.gpu_runtime_imported(), "GPU imported before guarded worker")
        verify_frozen(args, manifest, identity, profile, base)
        return worker_impl(args, manifest, identity, dp_size, rank, profile, base, native)


def worker_impl(args, manifest, identity, dp_size, rank, profile, base, native):
    sys.path.insert(0, str(args.repo/"examples")); sys.path.insert(0, str(args.repo))
    llm = controller = binder = None
    completed = []
    try:
        from _config import create_llm, setup_common_env
        setup_common_env(model_name=str(args.model_path), sampler_force_topk=args.sampler_force_topk,
                         set_override_experts=False, enable_eplb=False)
        started = time.perf_counter()
        llm = create_llm(str(args.model_path), rank, ep_size=8, cp_size=1,
            env={"DEEPINFER_LOGGING_INTERVAL_STEP": "0"}, data_parallel_size=8,
            max_num_batched_tokens=8192, max_model_len=270336, max_num_seqs=1,
            gpu_memory_utilization=.60, **profile.llm_kwargs())
        init_s = time.perf_counter()-started
        verify_native_imports(args.repo)
        import torch
        from deepinfer import envs
        from deepinfer.utils import allow_cuda_sync
        from canonical_request_binding import install_request_binding
        from kv_cache_engram.tail_link import TailLinkController
        torch.backends.cuda.matmul.allow_tf32 = False

        class Driver(base.NativeDriver):
            request_binding = None
            phase = "native"

            def submit(self, rid, ids):
                require(self.request_binding is None, "overlapping request state")
                self.request_binding = dict(request_id=rid, phase=self.phase, token_ids=list(ids))
                return super().submit(rid, ids)

            def finish(self, rid):
                super().finish(rid)
                require(self.request_binding is not None and self.request_binding["request_id"] == rid,
                        "request completion ownership mismatch")
                self.request_binding = None

        driver = Driver(llm, args, diagnostic_rank=rank)
        model_runner = driver.engine.driver_worker.model_runner
        runtime, validation = native.actual_runtime(profile, args, model_runner, torch, envs,
                                                    cache_config=driver.engine.cache_config)
        write_json(args.out_dir/f"native-runtime.rank{rank}.json", base.seal(dict(observed=runtime, validation=validation)))
        from gpu_pilot.cuda_mask_qualification import run_cuda_mask_qualification, validate_report
        mask_reports = driver.gather(run_cuda_mask_qualification(torch, allow_cuda_sync))
        require(len(mask_reports) == 8, "all eight ranks require strict CUDA mask qualification")
        for report in mask_reports:
            validate_report(report)
        cuda_mask_qualification = base.seal(dict(
            schema="ffn-only-cuda-mask-all-ranks-v1", passed=True,
            execution_configuration_sha256=identity["execution_configuration_sha256"],
            reports=[dict(rank=i, report=report) for i, report in enumerate(mask_reports)],
            quality_eligible=False, timing_valid=False))
        if rank == 0:
            write_json(args.out_dir/"cuda_mask_qualification.json", cuda_mask_qualification)
            print(json.dumps(dict(phase="cuda_mask_qualification", status="complete", ranks=8)), flush=True)
        driver.barrier()
        layers = list(model_runner.model.model.layers)
        require([m.layer_idx for m in layers] == list(range(40)) and dp_size == 8, "native40 layer DP8 geometry required")
        from gpu_pilot.index_hit20_empty_ep_qualification import (
            run_empty_ep_qualification, build_all_ranks_report, EmptyEPCapability,
        )
        empty_ep_reports = driver.gather(run_empty_ep_qualification(
            layers, rank, torch, allow_cuda_sync,
            execution_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"]))
        empty_ep_qualification = build_all_ranks_report(empty_ep_reports,
            execution_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"])
        if rank == 0:
            write_json(args.out_dir/"empty_ep_qualification.json", empty_ep_qualification)
        driver.barrier()
        binder = install_request_binding(model_runner, lambda: driver.request_binding if rank == 0 else None)
        if rank == 0:
            empty_ep_capability = EmptyEPCapability(layers, empty_ep_qualification,
                execution_sha256=identity["execution_configuration_sha256"],
                model_sha256=identity["checkpoint_index_sha256"])
            controller = TailLinkController(layers, allow_sync=allow_cuda_sync,
                native_model=model_runner.model.model,
                model_sha256=identity["checkpoint_index_sha256"], tokenizer_sha256=identity["tokenizer_json_sha256"],
                execution_sha256=identity["execution_configuration_sha256"], dependency_root=args.experiment_root,
                empty_ep_capability=empty_ep_capability)
            controller.install()
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]

        def phase(row, name, mode, *, arm, quality=False, tokens=1, timing=False, repeat=None, fraction=None,
                  partial_smoke=False, verify_sparse=False):
            began = time.perf_counter()
            args.mode, args.max_new_tokens = ("quality" if quality else "throughput"), tokens
            driver.phase = name
            rid = f"tail-link:{row.get('index', 'doc')}:{name}:{row['sample_id']}"
            if controller is not None:
                # CPU cache lookup, fresh staging allocation, prefetch H2D and
                # event waits belong to TTFT: clock starts before this reset.
                index_device = layers[0].self_attn.q_a_kv_swa_proj.weight.device if mode in ("sensitive", "reference_sparse", "all_selected") else None
                with allow_cuda_sync():
                    controller.reset(rid, mode, input_ids=row["input_ids"], documents=row.get("documents"),
                                     phase=name, fraction=fraction, index_device=index_device,
                                     verify_sparse=verify_sparse, dynamic_document_swa=(arm == "tail128_dynamic"))
            driver.barrier()
            if rank == 0:
                print(json.dumps(dict(case=row.get("index"), phase=name, status="starting", mode=mode, arm=arm)), flush=True)
            records, window = base.run_phase(driver, [(rid, row)] if rank == 0 else [],
                                             int(rank == 0), rank, epoch, tokens, lambda _: None)
            report = None
            if controller is not None:
                with allow_cuda_sync():
                    report = controller.finish()
            require(len(records) == 1 and records[0].rank == 0 and records[0].success(args.mode),
                    "phase lacks exactly one complete native request")
            rec = base.canonical_record(records[0], driver.engine.tokenizer, args.mode, name)
            require(quality or len(rec["output_ids"]) == tokens, "fixed generation length mismatch")
            start = driver.gather(began-epoch)[0]
            rec.update(online_prepare_to_submit_s=rec["submit_s"]-start,
                online_ttft_s=rec["submit_s"]+rec["ttft_s"]-start,
                online_elapsed_s=rec["finish_s"]-start, natural_EOS=quality,
                truncated=bool(quality and rec.get("finish_reason") == "length"))
            reports = driver.gather(report)
            require(isinstance(reports[0], dict) and all(p is None for p in reports[1:]),
                    "only rank0 may own tail-link state")
            result = base.seal(dict(schema=PHASE_SCHEMA, name=name, mode=mode, arm=arm, repeat=repeat,
                fraction=fraction, record=rec, controller=reports[0], timing_valid=timing, quality_eligible=quality,
                partial_smoke=partial_smoke,
                verify_sparse=verify_sparse, dynamic_document_swa=(arm == "tail128_dynamic"),
                native_yoco_swa_share_hidden=identity["native_yoco_swa_share_hidden"],
                online_global_window_s=window["end_s"]-start, index=row.get("index"), dataset=row["dataset"],
                input_ids_sha256=row["input_ids_sha256"], manifest_sha256=manifest["sha256"],
                execution_configuration_sha256=identity["execution_configuration_sha256"],
                empty_ep_qualification_sha256=empty_ep_qualification["sha256"],
                tail_link_protocol_sha256=identity["tail_link_protocol_sha256"],
                online_cpu_tail_prefetch_included=True, online_clock_started_before_reset=True,
                phase_local_gpu_staging=True, offline_build_included=False,
                native_global_kv_and_index_preserved=True))
            if rank == 0:
                write_json(args.out_dir/f"{name}.json", result)
                print(json.dumps(dict(case=row.get("index"), phase=name, status="complete", arm=arm,
                    em=rec.get("answer_em") if quality else None, f1=rec.get("answer_token_f1") if quality else None,
                    online_ttft_s=rec["online_ttft_s"] if timing else None)), flush=True)
            driver.barrier()
            return result

        check_row = qualification_row(manifest)
        reference = phase(check_row, "qualification.native", "qualified_native", arm="qualification")
        candidate = phase(check_row, "qualification.all_selected", "all_selected", arm="qualification")
        qualification_passed(reference, candidate)
        qualification = base.seal(dict(schema="tail-link-qualification-v1", passed=True,
            cuda_mask_qualification_sha256=cuda_mask_qualification["sha256"],
            empty_ep_qualification_sha256=empty_ep_qualification["sha256"],
            reference_sha256=reference["sha256"], candidate_sha256=candidate["sha256"],
            identity_check_count=candidate["controller"]["identity_check_count"], quality_claim=False,
            projection_identity_checks=candidate["controller"]["projection_identity_checks"],
            cache_restore_identity_checks=candidate["controller"]["cache_restore_identity_checks"],
            qualification_input_tokens=16384, full_256k_numerical_parity_claim=False,
            old_qualification_reused=False))
        if rank == 0:
            write_json(args.out_dir/"qualification.json", qualification)
        partial_smoke = dynamic_partial_smoke = None
        sparse_qualifications = {}
        for row in selected_rows(manifest):
            index = row["index"]
            if controller is not None:
                controller.clear_cache()
                controller.set_cache_output_dir(args.out_dir/f"case{index}.offline-kv")
            build_start = time.perf_counter()
            builds = [phase(doc, f"case{index}.offline_doc{doc['offline_document_index']}",
                            "capture_independent", arm="offline") for doc in support.document_jobs(base, row)]
            driver.barrier()
            build_s = max(driver.gather(time.perf_counter()-build_start))
            if index == EVALUATION_INDICES[0]:
                from gpu_pilot.read_tail_link_results import sparse_identity
                smoke_closures = {}
                for arm in ARMS[1:]:
                    # A policy cannot borrow another policy's numerical gate.
                    # Controller keeps qualification fingerprints keyed by policy.
                    sparse_reference = phase(row, f"case2.{arm}_sparse_reference", "reference_sparse",
                        arm=arm, tokens=1, fraction=None)
                    smoke_phase = phase(row, f"case{index}.{arm}_partial_smoke", "sensitive", arm=arm,
                        tokens=1, fraction=FRACTIONS[arm], partial_smoke=True, verify_sparse=True)
                    sparse_qualifications[arm] = sparse_identity(
                        sparse_reference, smoke_phase, row, qualification["sha256"])
                    smoke_closures[arm] = verify_partial_smoke(
                        [smoke_phase], row, qualification, sparse_qualifications[arm], arm=arm)
                    if rank == 0:
                        prefix = "dynamic_" if arm == "tail128_dynamic" else ""
                        write_json(args.out_dir/f"{prefix}sparse_qualification.json", sparse_qualifications[arm])
                        write_json(args.out_dir/f"{prefix}partial_smoke.json", smoke_closures[arm])
                partial_smoke = smoke_closures["tail128_link"]
                dynamic_partial_smoke = smoke_closures["tail128_dynamic"]
            require(partial_smoke is not None and partial_smoke["passed"]
                    and dynamic_partial_smoke is not None and dynamic_partial_smoke["passed"],
                    "both static and dynamic full-input execution smokes required")
            quality = {}
            for arm in ARMS:
                quality[arm] = phase(row, f"case{index}.{arm}_quality", "native" if arm == "recomputed" else "sensitive",
                    arm=arm, quality=True, tokens=64, fraction=FRACTIONS[arm])
                verify_phase(quality[arm], index=index, arm=arm, timing=False, quality=True)
            for _, arm in timing_schedule(index)[:len(ARMS)]:
                phase(row, f"case{index}.{arm}_warmup", "native" if arm == "recomputed" else "sensitive",
                      arm=arm, tokens=32, fraction=FRACTIONS[arm])
            measured = [phase(row, f"case{index}.{arm}_measure{repeat}", "native" if arm == "recomputed" else "sensitive",
                              arm=arm, tokens=32, timing=True, repeat=repeat, fraction=FRACTIONS[arm])
                        for repeat, arm in timing_schedule(index)]
            result = base.seal(dict(schema="tail-link-case-paired-v1", index=index, dataset=row["dataset"],
                sample_id=row["sample_id"], references=row["references"], question=row["question"],
                input_tokens=row["input_tokens"], input_ids_sha256=row["input_ids_sha256"],
                quality={}, timing_phases=[], timing_summary=summarize_timing(measured, index),
                offline_build_s=build_s, offline_phase_sha256=[p["sha256"] for p in builds],
                qualification_sha256=qualification["sha256"], ffn_layers=list(FFN_LAYERS),
                partial_smoke_sha256=partial_smoke["sha256"],
                dynamic_partial_smoke_sha256=dynamic_partial_smoke["sha256"],
                offline_capture_layers=list(CAPTURE_LAYERS), all_negative_results_retained=True,
                exploratory_previously_exposed_inputs=True, calibration_claim=False))
            if rank == 0:
                body = {k:v for k,v in result.items() if k != "sha256"}
                body["quality"] = {arm: evidence_ref(args.out_dir,
                    f"case{index}.{arm}_quality.json", value["sha256"]) for arm,value in quality.items()}
                body["timing_phases"] = [dict(repeat=value["repeat"], arm=value["arm"],
                    **evidence_ref(args.out_dir, value["name"]+".json", value["sha256"])) for value in measured]
                result = base.seal(body)
                write_json(args.out_dir/f"case{index}.paired.json", result)
                completed.append(dict(index=index, **evidence_ref(args.out_dir,
                    f"case{index}.paired.json", result["sha256"])))
            else:
                completed.append(dict(index=index))
            driver.barrier()
        args.mode, args.max_new_tokens = "quality", 64
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        if rank == 0:
            write_json(args.out_dir/"run.json", base.seal(dict(schema=SCHEMA, status="complete",
                identity=identity, manifest_sha256=manifest["sha256"], qualification=qualification, cases=completed,
                partial_smoke_sha256=partial_smoke["sha256"],
                dynamic_partial_smoke_sha256=dynamic_partial_smoke["sha256"],
                completed_case_indices=list(EVALUATION_INDICES), engine_init_s=init_s,
                calibration_claim=False, population_accuracy_certified=False, v8_changed=False,
                other_experiments_resumed=False, no_automatic_retry=True)))
        driver.barrier()
    except BaseException as exc:
        # These are already-host Python scope fields, not tensor reads. Keep
        # the exact failed row layout without adding a synchronization path.
        failure_context = None
        if controller is not None:
            failure_context = getattr(getattr(controller, "adapter", None), "last_failure_context", None)
        write_json(args.out_dir/f"failure.rank{rank}.json", base.seal(dict(schema=SCHEMA, status="failed", rank=rank,
            error=repr(exc), traceback=traceback.format_exc()[-32768:], completed_indices=[r["index"] for r in completed],
            ffn_failure_context=failure_context)))
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup = []
        for obj, method in ((controller, "close"), (binder, "restore"), (llm, "close")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except BaseException as exc:
                    cleanup.append(f"{method}: {exc!r}")
        if cleanup and not active_error:
            raise RuntimeError("tail-link cleanup failed: " + "; ".join(cleanup))


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bind_native_snapshot(args)
    base, native, NativeProfile, scoped_base_profile = support.load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args):
        support.validate_args(args)
        manifest, identity = base.preflight(args)
        support.validate_inputs(base, manifest, base.read_json(args.pilot_protocol))
        protocol = validate_protocol(args, manifest)
        rows = selected_rows(manifest)
        identity.update(pilot_source_sha256=source_hashes(args.experiment_root),
            native_yoco_swa_share_hidden=False,
            native_yoco_swa_configuration=native_yoco_swa_configuration(args.repo, args.model_path),
            native_profile=profile.identity(args), tail_link_protocol_sha256=protocol["sha256"],
            workflow_stage=SCHEMA, no_redknot=False,
            experiment_root=str(args.experiment_root.resolve()),
            isolated_native_snapshot=str(args.repo.resolve()))
        identity["execution_configuration_sha256"] = base.digest(dict(native_profile=identity["native_profile"],
            fixed_env=identity["fixed_env"], native_source=base.source_identity(args.repo),
            pilot_source_sha256=identity["pilot_source_sha256"], protocol_sha256=protocol["sha256"],
            native_yoco_swa_configuration=identity["native_yoco_swa_configuration"]))
        require(not native.gpu_runtime_imported(), "CPU preflight imported GPU runtime")
        require(shutil.disk_usage(args.experiment_root).free >= 4*1024**3, "at least4GiB free tail-cache/evidence space required")
        if args.check_only:
            print(json.dumps(dict(status="cpu_preflight_passed", gpu_initialized=False, identity=identity,
                evaluation_cases=[dict(index=r["index"], dataset=r["dataset"], input_tokens=r["input_tokens"]) for r in rows],
                qualification_case=0, qualification_input_tokens=16384, calibrated=False)))
            return 0
        args.out_dir.mkdir(parents=True, exist_ok=False)
        # Preserve the complete original sealed input, not a self-asserted list
        # of gold strings. The reader independently rehashes IDs and documents.
        write_json(args.out_dir/"input_manifest.json", manifest)
        write_json(args.out_dir/"preflight.json", base.seal(dict(schema=SCHEMA, identity=identity,
            native_yoco_swa_share_hidden=False,
            manifest_sha256=manifest["sha256"], protocol=protocol, evaluation_case_indices=list(EVALUATION_INDICES),
            qualification_input_ids_sha256=qualification_row(manifest)["input_ids_sha256"],
            partial_smoke=dict(index=EVALUATION_INDICES[0], arms=list(ARMS[1:]), mode="sensitive", output_tokens=1,
                               original_full_input=True, quality_eligible=False, timing_valid=False),
            audit_cases=[dict(index=r["index"], dataset=r["dataset"], sample_id=r["sample_id"], references=r["references"],
                question=r["question"], input_tokens=r["input_tokens"], input_ids_sha256=r["input_ids_sha256"],
                documents=r["documents"]) for r in rows])))
        base.wait_for_release(args, manifest["sha256"])
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        from quality_process_group import launch
        launch(functools.partial(worker, args, manifest, identity), args.dp, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

