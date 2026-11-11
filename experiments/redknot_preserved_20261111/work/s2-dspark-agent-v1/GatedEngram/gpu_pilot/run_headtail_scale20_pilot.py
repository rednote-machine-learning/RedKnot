"""Finite independent headtail-scale20 token-level B pilot.

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
from gpu_pilot.headtail_scale20_inputs import validate_inputs
from gpu_pilot.headtail_scale20_protocol import phase_policy

SCHEMA = "kv-cache-engram-headtail-scale20-pilot-run-v1"
PHASE_SCHEMA = "kv-cache-engram-headtail-scale20-phase-v1"
ARMS = ("recomputed", "tail128_link", "headtail128_link")
POLICY_ARMS = ARMS[1:]
FRACTIONS = dict.fromkeys(ARMS)
EVALUATION_INDICES = tuple(range(2, 22))
CAPTURE_LAYERS = tuple(range(40))
FFN_LAYERS = tuple(range(3,40))
NATIVE_SNAPSHOT = STAGE.parent / "NativeSnapshot"
INPUT_COMPANIONS = ("source-manifest.json", "extension-manifest.json", "extension-input-protocol.json", "source-plan.json")


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
    from gpu_pilot.headtail_scale20_protocol import DEPENDENCIES
    require(DEPENDENCIES, "new headtail-scale20 dependency pins must be frozen before preflight")
    hashes = support.source_hashes(root)
    for name, expected in DEPENDENCIES.items():
        path = (STAGE/name.removeprefix("GatedEngram/")) if name.startswith("GatedEngram/") else Path(root)/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, "frozen sparse operator dependency changed: "+name)
        hashes[name] = actual
    return hashes


def parse_args(argv=None):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--headtail-scale20-protocol", required=True, type=Path)
    known, rest = extra.parse_known_args(argv)
    args = support.parse_args(rest)
    args.headtail_scale20_protocol = known.headtail_scale20_protocol
    return args


def validate_protocol(args, manifest):
    from gpu_pilot.headtail_scale20_protocol import headtail_scale20_protocol
    value = json.loads(args.headtail_scale20_protocol.read_bytes())
    require(value == headtail_scale20_protocol(), "twenty-case comparison protocol differs")
    require(type(value.get("input_manifest_sha256")) is str
            and manifest.get("sha256") == value["input_manifest_sha256"],
            "comparison must bind the frozen assembled twenty-case manifest")
    require(getattr(args, "sensitivity_protocol", None) is None
            and "sensitivity_protocol_sha256" not in manifest,
            "old sensitivity20% protocol cannot govern this comparison")
    return value


def selected_rows(manifest):
    by_index = {row["index"]: row for row in manifest["cohort"]}
    require(all(i in by_index for i in EVALUATION_INDICES), "fixed headtail-scale20 pilot cases absent")
    rows = [by_index[i] for i in EVALUATION_INDICES]
    require([r["dataset"] for r in rows] == ["hotpotqa", "2wikimqa"] * 10, "fixed dataset order changed")
    return rows


def timing_schedule(index):
    from gpu_pilot.headtail_scale20_protocol import timing_schedule as declared
    return declared(index)


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


def verify_phase(value, *, index, arm, timing, quality, diagnostic=False):
    require(isinstance(value, dict) and value.get("sha256") == _digest(
        {k: v for k, v in value.items() if k != "sha256"}), "phase seal mismatch")
    require(value.get("schema") == PHASE_SCHEMA and value.get("index") == index
            and value.get("arm") == arm and value.get("timing_valid") is timing
            and value.get("quality_eligible") is quality, "phase identity/eligibility mismatch")
    require(value.get("dynamic_document_swa") is (arm == "headtail128_link"),
            "phase dynamic policy does not match arm")
    require(value.get("native_yoco_swa_share_hidden") is False,
            "phase native YOCO SWA configuration differs")
    require(all(value.get(k) == v for k,v in phase_policy(arm).items()),
            "phase head/tail scope or refresh layer differs")
    require(value.get("diagnostic") is diagnostic and not (diagnostic and (timing or quality)),
            "diagnostic cannot enter formal timing or quality")
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
    for key in ("execution_configuration_sha256", "headtail_scale20_protocol_sha256", "manifest_sha256"):
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


def verify_partial_smoke(phases, row, qualification, sparse_qualification, *, arm="headtail128_link"):
    """Require full input, independent SWA and FFN accounting before quality."""
    from gpu_pilot.read_headtail_scale20_results import audit_controller, documents
    require(row["index"] == EVALUATION_INDICES[0] and len(row["documents"]) == 8,
            "smoke must use original complete eight-document case2")
    require(qualification.get("passed") is True and qualification.get("identity_check_count") == 80,
            "new strict numerical qualification must precede smoke")
    require(sparse_qualification.get("passed") is True
            and sparse_qualification.get("arm") == arm
            and sparse_qualification.get("input_ids_sha256") == row["input_ids_sha256"],
            "independent sparse numerical qualification must precede smoke closure")
    require(arm in POLICY_ARMS and len(phases) == 1 and phases[0].get("arm") == arm,
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
    require(phase["controller"].get("dynamic_document_swa") is (arm == "headtail128_link"),
            "smoke controller policy does not match arm")
    actual = audit_controller(phase["controller"], row, phase, documents(row))
    require(actual["ffn_bypassed_rows"] > 0, "smoke did not exercise real FFN bypass")
    require(actual.get("consumed_cached_pairs", 0) > 0,
            "smoke did not consume offline SWA memory in actual query attention")
    if arm == "headtail128_link":
        require(actual.get("dynamic_refreshed_rows", 0) > 0,
                "dynamic smoke did not refresh actual document SWA rows")
    body = dict(schema="headtail-scale20-partial-smoke-v1", passed=True, arm=arm, index=row["index"],
        phase_sha256={arm:phase["sha256"]}, input_ids_sha256=row["input_ids_sha256"],
        input_tokens=row["input_tokens"], qualification_sha256=qualification["sha256"],
        quality_eligible=False, timing_valid=False, accuracy_guarantee=False,
        full_input_original_eight_documents=True, independent_swa_and_ffn_paths_exercised=True,
        output_tokens_per_arm=1)
    body["sparse_qualification_sha256"] = sparse_qualification["sha256"]
    return dict(body, sha256=_digest(body))


def summarize_timing(phases, index):
    require([(p["repeat"], p["arm"]) for p in phases] == timing_schedule(index),
            "candidate needs exactly three registered repetitions")
    summaries = {}
    bindings = set()
    for arm in ARMS:
        records = []
        for phase in (p for p in phases if p["arm"] == arm):
            records.append(verify_phase(phase, index=index, arm=arm, timing=True, quality=False))
            bindings.add((phase["input_ids_sha256"], phase["execution_configuration_sha256"],
                          phase["manifest_sha256"], phase["headtail_scale20_protocol_sha256"]))
        samples = [r["online_ttft_s"] for r in records]
        summaries[arm] = dict(online_ttft_samples_s=samples, online_ttft_p50_s=statistics.median(samples),
                              repetitions=3, fixed_output_tokens=32)
    require(len(bindings) == 1, "timing arms do not share frozen input/execution")
    baseline = summaries["recomputed"]["online_ttft_p50_s"]
    comparisons = {arm:dict(ttft_speedup=baseline/summaries[arm]["online_ttft_p50_s"],
        paired_repeat_speedups=[]) for arm in POLICY_ARMS}
    by_repeat = {(p["repeat"],p["arm"]):p["record"]["online_ttft_s"] for p in phases}
    for arm in POLICY_ARMS:
        comparisons[arm]["paired_repeat_speedups"] = [by_repeat[(r,"recomputed")]/by_repeat[(r,arm)] for r in range(3)]
    return dict(arms=summaries, comparisons=comparisons, concurrency=1, active_dp_ranks=1, participating_ep_ranks=8,
                online_full_bank_gpu_resident=True, online_gather_scatter_and_fallback_included=True,
                model_load_included=False, offline_build_included=False, saturated_capacity_claim=False,
                fresh_paired_baseline=True, historical_results_used_for_admission=False)


def verify_diagnostic(value, full_quality, index):
    record = verify_phase(value,index=index,arm="recomputed",timing=False,quality=False,diagnostic=True)
    require(value.get("name") == f"case{index}.diagnostic_recomputed"
            and value.get("mode") == "native" and value.get("partial_smoke") is False
            and value.get("repeat") is None and record.get("natural_EOS") is False
            and len(record.get("output_ids",[])) == 1, "independent native one-token diagnostic required")
    require(full_quality.get("arm") == "recomputed" and full_quality.get("diagnostic") is False
            and full_quality.get("quality_eligible") is True
            and all(value.get(k) == full_quality.get(k) for k in
                ("index","input_ids_sha256","manifest_sha256","execution_configuration_sha256",
                 "headtail_scale20_protocol_sha256")), "diagnostic/source quality binding differs")
    require(record["output_ids"] == full_quality["record"]["output_ids"][:1],
            "diagnostic observer changed the native first output token")
    require(value.get("controller",{}).get("diagnostic") is True
            and type(value["controller"].get("early_token_diagnostics")) is dict
            and value["controller"]["early_token_diagnostics"].get("schema") == "early-token-diagnostics-v1",
            "diagnostic phase did not close its independent observer evidence")
    return True


def verify_shared_bank(phases):
    require(len(phases) == 15, "all three quality/warmup/three-timing phases required")
    fields = ("bank_generation", "offline_bank_scope_sha256", "offline_bank_entry_count",
              "offline_bank_rows", "offline_bank_bytes")
    observations = []
    for phase in phases:
        control = phase["controller"]
        require(control.get("comparison_schema") == "headtail-three-arm-comparison-v1"
                and control.get("comparison_arm") == phase["arm"]
                and control.get("cache_storage_policy") == "shared_gpu_full_canonical_bank_all_arms"
                and control.get("gpu_bank_constant_across_arms") is True
                and control.get("offline_gpu_resident") is True, "unmatched three-arm GPU bank policy")
        observation = {k:control.get(k) for k in fields}
        require(type(observation["bank_generation"]) is int and observation["bank_generation"] > 0
                and type(observation["offline_bank_scope_sha256"]) is str
                and len(observation["offline_bank_scope_sha256"]) == 64
                and observation["offline_bank_entry_count"] == 320
                and observation["offline_bank_rows"] == 320*32768
                and observation["offline_bank_bytes"] == 10*1024**3, "incomplete shared GPU bank")
        observations.append(observation)
    require(all(v == observations[0] for v in observations), "GPU bank changed across arms/phases")
    return observations[0]


def verify_diagnostic_native_identity(identity, protocol):
    """Fail on the CPU before model loading if diagnostic topology pins drift."""
    pins = protocol["diagnostics"]
    require(identity.get("checkpoint_config_sha256") == pins["checkpoint_config_sha256"]
            and all(identity.get("runtime_source_files_sha256", {}).get(name) == sha
                    for name, sha in pins["native_source_files_sha256"].items()),
            "diagnostic native source/config pins differ before GPU admission")
    require(pins["native_prefix_routing_layers"] == [2]
            and pins["native_prefix_routing_compress_ratios"] == {"2": 2}
            and pins["native_prefix_routing_selection_kinds"] == {"2": "sparse_topk"}
            and pins["native_index_topk"] == 512, "diagnostic prefix topology differs")


def verify_frozen(args, manifest, identity, profile, base, *, native_source=False):
    require(identity.get("native_yoco_swa_share_hidden") is False
            and native_yoco_swa_configuration(args.repo, args.model_path)
            == identity.get("native_yoco_swa_configuration"), "native YOCO SWA configuration changed")
    require(source_hashes(args.experiment_root) == identity["pilot_source_sha256"],
            "headtail-scale20 pilot source changed after preflight")
    require(profile.identity(args) == identity["native_profile"] and base.effective_env(args) == identity["fixed_env"],
            "native configuration drift")
    disk = base.read_json(args.manifest)
    validate_inputs(base, disk, base.read_json(args.pilot_protocol), input_dir=args.manifest.parent)
    protocol = validate_protocol(args, disk)
    verify_diagnostic_native_identity(identity, protocol)
    require(disk["sha256"] == manifest["sha256"]
            and protocol["sha256"] == identity["headtail_scale20_protocol_sha256"], "input/protocol changed")
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
    diagnostics = []
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
        from kv_cache_engram.headtail_comparison import ComparisonController
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
            controller = ComparisonController(layers, allow_sync=allow_cuda_sync,
                native_model=model_runner.model.model,
                model_sha256=identity["checkpoint_index_sha256"], tokenizer_sha256=identity["tokenizer_json_sha256"],
                execution_sha256=identity["execution_configuration_sha256"], dependency_root=args.experiment_root,
                empty_ep_capability=empty_ep_capability)
            controller.install()
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]

        def phase(row, name, mode, *, arm, quality=False, tokens=1, timing=False, repeat=None, fraction=None,
                  partial_smoke=False, verify_sparse=False, diagnostic=False):
            began = time.perf_counter()
            args.mode, args.max_new_tokens = ("quality" if quality else "throughput"), tokens
            driver.phase = name
            rid = f"headtail-scale20:{row.get('index', 'doc')}:{name}:{row['sample_id']}"
            if controller is not None:
                # Fresh phase state, full-bank lookup and online memory preparation
                # belong to TTFT: clock starts before this reset.
                index_device = layers[0].self_attn.q_a_kv_swa_proj.weight.device if mode in ("sensitive", "reference_sparse", "all_selected") else None
                with allow_cuda_sync():
                    controller.reset(rid, mode, arm=arm, diagnostic=diagnostic, input_ids=row["input_ids"], documents=row.get("documents"),
                                     phase=name, fraction=fraction, index_device=index_device,
                                     verify_sparse=verify_sparse)
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
                    "only rank0 may own headtail-scale20 state")
            require(reports[0].get("diagnostic") is diagnostic
                    and (diagnostic or reports[0].get("early_token_diagnostics") is None),
                    "diagnostic observer must remain disabled during all formal phases")
            result = base.seal(dict(schema=PHASE_SCHEMA, name=name, mode=mode, arm=arm, repeat=repeat,
                fraction=fraction, record=rec, controller=reports[0], timing_valid=timing, quality_eligible=quality,
                partial_smoke=partial_smoke, diagnostic=diagnostic,
                verify_sparse=verify_sparse, dynamic_document_swa=(arm == "headtail128_link"),
                native_yoco_swa_share_hidden=identity["native_yoco_swa_share_hidden"],
                **phase_policy(arm),
                online_global_window_s=window["end_s"]-start, index=row.get("index"), dataset=row["dataset"],
                input_ids_sha256=row["input_ids_sha256"], manifest_sha256=manifest["sha256"],
                execution_configuration_sha256=identity["execution_configuration_sha256"],
                empty_ep_qualification_sha256=empty_ep_qualification["sha256"],
                headtail_scale20_protocol_sha256=identity["headtail_scale20_protocol_sha256"],
                online_full_bank_gpu_resident=(arm in ARMS), online_clock_started_before_reset=True,
                phase_local_memory_rebuilt=True, offline_build_included=False,
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
        qualification = base.seal(dict(schema="headtail-scale20-qualification-v1", passed=True,
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
        partial_smokes = {}
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
                from gpu_pilot.read_headtail_scale20_results import sparse_identity
                for arm in POLICY_ARMS:
                    sparse_reference = phase(row, f"case2.{arm}_sparse_reference", "reference_sparse",
                        arm=arm, tokens=1, fraction=None)
                    smoke_phase = phase(row, f"case2.{arm}_partial_smoke", "sensitive", arm=arm,
                        tokens=1, fraction=None, partial_smoke=True, verify_sparse=True)
                    smoke_result = None
                    if rank == 0:
                        sparse_qualification = sparse_identity(
                            sparse_reference, smoke_phase, row, qualification["sha256"])
                        smoke_result = verify_partial_smoke(
                            [smoke_phase], row, qualification, sparse_qualification, arm=arm)
                        write_json(args.out_dir/f"{arm}.sparse_qualification.json", sparse_qualification)
                        write_json(args.out_dir/f"{arm}.partial_smoke.json", smoke_result)
                    partial_smokes[arm] = driver.gather(smoke_result)[0]
            require(set(partial_smokes) == set(POLICY_ARMS)
                    and all(s["passed"] for s in partial_smokes.values()),
                    "both fresh full-input policy qualifications and smokes required")
            quality = {}
            for _, arm in timing_schedule(index)[:len(ARMS)]:
                quality[arm] = phase(row, f"case{index}.{arm}_quality", "native" if arm == "recomputed" else "sensitive",
                    arm=arm, quality=True, tokens=64, fraction=FRACTIONS[arm])
                verify_phase(quality[arm], index=index, arm=arm, timing=False, quality=True)
            warmups = []
            for _, arm in timing_schedule(index)[:len(ARMS)]:
                warmups.append(phase(row, f"case{index}.{arm}_warmup", "native" if arm == "recomputed" else "sensitive",
                      arm=arm, tokens=32, fraction=FRACTIONS[arm]))
            measured = [phase(row, f"case{index}.{arm}_measure{repeat}", "native" if arm == "recomputed" else "sensitive",
                              arm=arm, tokens=32, timing=True, repeat=repeat, fraction=FRACTIONS[arm])
                        for repeat, arm in timing_schedule(index)]
            shared_bank = verify_shared_bank([*quality.values(), *warmups, *measured])
            timing_summary = summarize_timing(measured, index)
            diagnostic_began = time.perf_counter()
            diagnostic = phase(row, f"case{index}.diagnostic_recomputed", "native",
                arm="recomputed", tokens=1, diagnostic=True)
            verify_diagnostic(diagnostic, quality["recomputed"], index)
            diagnostic_elapsed_s = max(driver.gather(time.perf_counter()-diagnostic_began))
            result = base.seal(dict(schema="headtail-scale20-case-paired-v1", index=index, dataset=row["dataset"],
                sample_id=row["sample_id"], references=row["references"], question=row["question"],
                input_tokens=row["input_tokens"], input_ids_sha256=row["input_ids_sha256"],
                quality={}, timing_phases=[], timing_summary=timing_summary,
                diagnostic_recomputed=None, diagnostic_elapsed_s=diagnostic_elapsed_s,
                diagnostic_excluded_from_formal_metrics=True,
                shared_bank_identity=shared_bank, fresh_paired_baseline=True,
                offline_build_s=build_s, offline_phase_sha256=[p["sha256"] for p in builds],
                qualification_sha256=qualification["sha256"], ffn_layers=list(FFN_LAYERS),
                partial_smoke_sha256={arm:smoke["sha256"] for arm,smoke in partial_smokes.items()},
                offline_capture_layers=list(CAPTURE_LAYERS), all_negative_results_retained=True,
                exploratory_previously_exposed_inputs=True, calibration_claim=False))
            if rank == 0:
                body = {k:v for k,v in result.items() if k != "sha256"}
                body["diagnostic_recomputed"] = evidence_ref(args.out_dir,
                    f"case{index}.diagnostic_recomputed.json", diagnostic["sha256"])
                diagnostics.append(dict(index=index, **body["diagnostic_recomputed"]))
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
                identity=identity, manifest_sha256=manifest["sha256"], qualification=qualification, cases=completed, diagnostics=diagnostics,
                diagnostic_excluded_from_formal_metrics=True,
                partial_smoke_sha256={arm:smoke["sha256"] for arm,smoke in partial_smokes.items()},
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
            raise RuntimeError("headtail-scale20 cleanup failed: " + "; ".join(cleanup))


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bind_native_snapshot(args)
    base, native, NativeProfile, scoped_base_profile = support.load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args):
        support.validate_args(args)
        manifest, identity = base.preflight(args)
        validate_inputs(base, manifest, base.read_json(args.pilot_protocol), input_dir=args.manifest.parent)
        protocol = validate_protocol(args, manifest)
        verify_diagnostic_native_identity(identity, protocol)
        rows = selected_rows(manifest)
        identity.update(pilot_source_sha256=source_hashes(args.experiment_root),
            native_yoco_swa_share_hidden=False,
            native_yoco_swa_configuration=native_yoco_swa_configuration(args.repo, args.model_path),
            native_profile=profile.identity(args), headtail_scale20_protocol_sha256=protocol["sha256"],
            workflow_stage=SCHEMA, no_redknot=False,
            experiment_root=str(args.experiment_root.resolve()),
            isolated_native_snapshot=str(args.repo.resolve()))
        identity["execution_configuration_sha256"] = base.digest(dict(native_profile=identity["native_profile"],
            fixed_env=identity["fixed_env"], native_source=base.source_identity(args.repo),
            pilot_source_sha256=identity["pilot_source_sha256"], protocol_sha256=protocol["sha256"],
            native_yoco_swa_configuration=identity["native_yoco_swa_configuration"]))
        require(not native.gpu_runtime_imported(), "CPU preflight imported GPU runtime")
        require(shutil.disk_usage(args.experiment_root).free >= 320*1024**3, "at least320GiB free full-cache/evidence space required")
        if args.check_only:
            print(json.dumps(dict(status="cpu_preflight_passed", gpu_initialized=False, identity=identity,
                evaluation_cases=[dict(index=r["index"], dataset=r["dataset"], input_tokens=r["input_tokens"]) for r in rows],
                qualification_case=0, qualification_input_tokens=16384, calibrated=False)))
            return 0
        args.out_dir.mkdir(parents=True, exist_ok=False)
        # Preserve the complete original sealed input, not a self-asserted list
        # of gold strings. The reader independently rehashes IDs and documents.
        write_json(args.out_dir/"input_manifest.json", manifest)
        for name in INPUT_COMPANIONS:
            source = args.manifest.parent/name
            require(source.is_file() and not source.is_symlink(), "missing/unsafe input companion: "+name)
            with (args.out_dir/name).open("xb") as handle:
                handle.write(source.read_bytes())
        with (args.out_dir/"input-protocol.json").open("xb") as handle:
            handle.write(args.pilot_protocol.read_bytes())
        write_json(args.out_dir/"preflight.json", base.seal(dict(schema=SCHEMA, identity=identity,
            native_yoco_swa_share_hidden=False,
            manifest_sha256=manifest["sha256"], protocol=protocol, evaluation_case_indices=list(EVALUATION_INDICES),
            qualification_input_ids_sha256=qualification_row(manifest)["input_ids_sha256"],
            partial_smoke=dict(index=EVALUATION_INDICES[0], arms=list(POLICY_ARMS), mode="sensitive", output_tokens=1,
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

