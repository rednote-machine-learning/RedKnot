"""Finite, isolated native KV Cache Engram pilot. CUDA only after release.

Two source-disjoint calibration requests, four fixed evaluation requests.
The standard arm is unmodified full native computation. Candidate retains
every Q/attention/MoE row; only approved SWA KV projections may be bypassed.
This script never imports v8 or changes a native source file.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
LAYERS = tuple(range(3, 8)) + tuple(range(9, 14)) + tuple(range(15, 19))
SCHEMA = "kv-cache-engram-native-gpu-pilot-run-v1"
CACHE_BYTES = 14 * 8 * 32768 * 512 * 2
DEPENDENCIES = (
    "benchmark/pr148_native_runner.py", "benchmark/prepare_rag256.py",
    "strong_native_20261029/strong_native_runner.py", "strong_native_20261029/native_profile.py",
    "kv_recompression_20261029/canonical_request_binding.py",
    "kv_recompression_20261029/quality_process_group.py",
)


def load_runtime(root):
    for folder in ("benchmark", "strong_native_20261029", "kv_recompression_20261029",
                   "direct_global_20261029", "cross_layer_mla_repair_20261030",
                   "kv_compute_20261029", "moe_stage_20261028"):
        sys.path.insert(0, str(root / folder))
    import pr148_native_runner as base
    import strong_native_runner as native
    from native_profile import NativeProfile, scoped_base_profile
    return base, native, NativeProfile, scoped_base_profile


def parse_args(argv=None):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--experiment-root", type=Path, required=True)
    extra.add_argument("--pilot-protocol", type=Path, required=True)
    extra.add_argument("--sensitivity-protocol", type=Path)
    known, rest = extra.parse_known_args(argv)
    base, _, _, _ = load_runtime(known.experiment_root)
    args = base.parse_args(rest)
    vars(args).update(vars(known))
    return args


def source_hashes(root):
    paths = sorted(p for p in STAGE.rglob("*.py") if "__pycache__" not in p.parts)
    result = {"GatedEngram/" + str(p.relative_to(STAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    for name in DEPENDENCIES:
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return result


def validate_args(args):
    if ((args.dp, args.ep, args.cp, args.concurrency, args.max_num_seqs) != (8, 8, 1, 1, 1)
            or args.mode != "quality" or args.arm != "recomputed" or args.max_new_tokens != 64
            or args.max_num_batched_tokens != 8192 or args.max_model_len != 270336
            or args.gpu_memory_utilization != .60 or not args.skip_warmup or args.repeat != 1
            or args.evaluate_calibration or args.profile_limit is not None or args.limit is not None
            or args.sample_id or args.head_profile or args.intervention_policy or args.cache_dir
            or args.head_cache_representation != "expanded" or args.allow_gpu_init
            or args.batch_invariant != 1 or args.sampler_force_topk != 1024
            or args.prefill_attempt_interval != 1 or args.seed != 0
            or any((args.diagnostic_progress, args.diagnostic_ep_sync, args.diagnostic_finite,
                    args.diagnostic_stacks_seconds, args.diagnostic_prefill_breakdown))):
        raise ValueError("fixed independent C1 DP8/EP8/CP1 native pilot arguments required")
    if not args.check_only and (args.cpu_ready_file is None or args.release_file is None):
        raise ValueError("original supervisor ready/release gate is mandatory")


def validate_inputs(base, manifest, protocol):
    from gpu_pilot.pilot_protocol import pilot_protocol
    if protocol != pilot_protocol():
        raise ValueError("pilot protocol differs from CPU-tested frozen experiment")
    base.validate_manifest(manifest)
    if manifest.get("artifact_kind") != "kv_cache_engram_native_pilot":
        raise ValueError("independent pilot manifest required, never v8")
    if manifest.get("pilot_protocol_sha256") != protocol["sha256"]:
        raise ValueError("pilot protocol/manifest mismatch")
    rows = manifest["warmup"] + manifest["cohort"]
    if ([r["dataset"] for r in rows] != ["hotpotqa", "2wikimqa"] * 3
            or len(manifest["warmup"]) != 2 or len(manifest["cohort"]) != 4
            or [r.get("index") for r in rows] != list(range(6))):
        raise ValueError("two calibration + four evaluation fixed cases required")
    split_contexts = []
    previous = set(manifest["selection_plan"]["excluded_context_sha256"])
    for split, group in (("calibration", manifest["warmup"]), ("heldout", manifest["cohort"])):
        contexts = set()
        for row in group:
            docs = base.document_specs(row)
            if (row.get("split") != split or row.get("primary_context_truncated") is not False
                    or len(docs) != 8 or any(len(d["token_ids"]) != 32768 for d in docs)
                    or row.get("context_tokens") != 262144
                    or row["input_tokens"] != row["prefix_tokens"] + 262144 + row["suffix_tokens"]
                    or [d["input_start"] for d in docs] != [row["prefix_tokens"] + j*32768 for j in range(8)]):
                raise ValueError("whole primary and exact 8x32768 context required")
            segments = row["context_segments"]
            actual = {s["context_sha256"] for s in segments if "context_sha256" in s}
            if not actual or actual & previous:
                raise ValueError("pilot reused registered exposed source material")
            contexts.update(actual)
        split_contexts.append(contexts)
    if split_contexts[0] & split_contexts[1]:
        raise ValueError("calibration/evaluation source material overlap")
    if len({r["input_ids_sha256"] for r in rows}) != 6 or len({r["question"] for r in rows}) != 6:
        raise ValueError("duplicate pilot inputs/questions")
    return rows


def document_jobs(base, row):
    for i, doc in enumerate(base.document_specs(row)):
        yield dict(sample_id=doc["token_ids_sha256"], dataset=row["dataset"],
            input_ids=list(doc["token_ids"]), input_tokens=32768,
            input_ids_sha256=doc["token_ids_sha256"], references=["__offline_not_quality__"],
            offline_document_index=i)


def timing_schedule(index):
    return [(repeat, arm) for repeat in range(3)
            for arm in (("recomputed", "gated") if (index + index // 2 + repeat) % 2 == 0
                        else ("gated", "recomputed"))]


def summarize_timing(phases, index):
    if [(p["repeat"], p["arm"]) for p in phases] != timing_schedule(index):
        raise ValueError("complete three paired timing repetitions required")
    summary = {}
    for arm in ("recomputed", "gated"):
        selected = [p for p in phases if p["arm"] == arm]
        values = []
        for p in selected:
            r = p["record"]
            if (p["timing_valid"] is not True or not r["success"] or len(r["output_ids"]) != 32
                    or not math.isfinite(r["online_ttft_s"]) or r["online_ttft_s"] <= 0
                    or not math.isfinite(p["online_global_window_s"]) or p["online_global_window_s"] <= 0
                    or r["online_ttft_s"] > p["online_global_window_s"]
                    or not math.isfinite(r["online_prepare_to_submit_s"]) or r["online_prepare_to_submit_s"] < 0
                    or not math.isfinite(r["online_elapsed_s"]) or r["online_elapsed_s"] < r["online_ttft_s"]):
                raise ValueError("invalid online paired timing")
            values.append(r["online_ttft_s"])
        summary[arm] = dict(online_ttft_samples_s=values, online_ttft_p50_s=statistics.median(values),
            fixed_output_tokens=32, repetitions=3)
    a, b = (summary[k]["online_ttft_p50_s"] for k in ("recomputed", "gated"))
    summary.update(ttft_speedup=a/b, ttft_reduction_s=a-b, ttft_reduction_percent=100*(1-b/a),
        concurrency=1, active_dp_ranks=1, participating_ep_ranks=8,
        online_gate_preparation_included=True, offline_build_included=False,
        model_load_included=False, saturated_capacity_claim=False)
    return summary


def verify_frozen(args, manifest, identity, profile, base, *, native_source=False):
    if source_hashes(args.experiment_root) != identity["pilot_source_sha256"]:
        raise RuntimeError("pilot executable changed after preflight")
    if profile.identity(args) != identity["native_profile"] or base.effective_env(args) != identity["fixed_env"]:
        raise RuntimeError("native configuration drift")
    disk = base.read_json(args.manifest)
    protocol = base.read_json(args.pilot_protocol)
    validate_inputs(base, disk, protocol)
    validate_sensitivity_inputs(args, disk, identity=identity)
    if disk["sha256"] != manifest["sha256"]:
        raise RuntimeError("pilot input changed")
    for filename, key in (("config.json", "checkpoint_config_sha256"),
                          ("model.safetensors.index.json", "checkpoint_index_sha256"),
                          ("tokenizer.json", "tokenizer_json_sha256")):
        if base.file_digest(args.model_path/filename) != identity[key]:
            raise RuntimeError("model/tokenizer identity changed")
    if native_source:
        for key, value in base.source_identity(args.repo).items():
            if identity[key] != value:
                raise RuntimeError("native source changed")


def validate_sensitivity_inputs(args, manifest, *, identity=None):
    """Optional new finite experiment, explicitly bound without changing v1 data."""
    path = getattr(args, "sensitivity_protocol", None)
    if path is None:
        if "sensitivity_protocol_sha256" in manifest:
            raise ValueError("sensitivity manifest requires its explicit protocol")
        return None
    from gpu_pilot.sensitivity_protocol import sensitivity_protocol
    value = json.loads(path.read_bytes())
    if value != sensitivity_protocol() or manifest.get("sensitivity_protocol_sha256") != value["sha256"]:
        raise ValueError("frozen sensitivity protocol/manifest mismatch")
    if identity is not None and identity.get("sensitivity_protocol_sha256") != value["sha256"]:
        raise ValueError("sensitivity execution identity mismatch")
    return value


def protect_calibrated_layers(numerical, admission):
    """Only remove permissions; retain every original candidate and audit statistic."""
    from gpu_pilot import pilot_protocol as pp
    pp._sealed(numerical); pp._sealed(admission)
    if admission.get("numerical_calibration_sha256") != numerical["sha256"]:
        raise ValueError("sensitivity plan belongs to another numerical calibration")
    value = json.loads(json.dumps({k: v for k, v in numerical.items() if k != "sha256"}))
    for layer in admission["mandatory_full_layers"]:
        if str(layer) in value["layers"]:
            value["layers"][str(layer)].update(selected=None, mode="full_recompute",
                sensitivity_protected=True)
    value.update(numerical_calibration_sha256=numerical["sha256"],
        sensitivity_admission_plan_sha256=admission["sha256"])
    return pp.seal_record(value)


def worker(args, manifest, identity, dp_size, rank):
    base, native, NativeProfile, scoped_base_profile = load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args, apply_environment=True):
        if native.gpu_runtime_imported():
            raise RuntimeError("GPU runtime imported before guarded worker")
        verify_frozen(args, manifest, identity, profile, base)
        return worker_impl(args, manifest, identity, dp_size, rank, profile, base, native)


def worker_impl(args, manifest, identity, dp_size, rank, profile, base, native):
    sys.path.insert(0, str(args.repo/"examples"))
    sys.path.insert(0, str(args.repo))
    llm = hook = binder = raw_writer = None
    completed, calibration_reports, evaluation_reports, sensitivity_reports = [], [], [], []
    try:
        from _config import create_llm, setup_common_env
        setup_common_env(model_name=str(args.model_path), sampler_force_topk=args.sampler_force_topk,
                         set_override_experts=False, enable_eplb=False)
        started = time.perf_counter()
        llm = create_llm(str(args.model_path), rank, ep_size=8, cp_size=1,
            env={"DEEPINFER_LOGGING_INTERVAL_STEP": "0"}, data_parallel_size=8,
            max_num_batched_tokens=8192, max_model_len=270336, max_num_seqs=1,
            gpu_memory_utilization=.60, **profile.llm_kwargs())
        init_s = time.perf_counter() - started
        import torch
        from deepinfer import envs
        from deepinfer.utils import allow_cuda_sync
        from canonical_request_binding import install_request_binding
        from kv_cache_engram.native_reuse import KVCacheEngramHook
        from gpu_pilot.raw_pairs import RawPairWriter
        from gpu_pilot import pilot_protocol as pp
        torch.backends.cuda.matmul.allow_tf32 = False

        class Driver(base.NativeDriver):
            request_binding = None
            phase = "native"

            def submit(self, rid, ids):
                if self.request_binding is not None:
                    raise RuntimeError("overlapping request state")
                self.request_binding = dict(request_id=rid, phase=self.phase, token_ids=list(ids))
                return super().submit(rid, ids)

            def finish(self, rid):
                super().finish(rid)
                if self.request_binding is None or self.request_binding["request_id"] != rid:
                    raise RuntimeError("request completion ownership mismatch")
                self.request_binding = None

        driver = Driver(llm, args, diagnostic_rank=rank)
        model_runner = driver.engine.driver_worker.model_runner
        runtime, validation = native.actual_runtime(profile, args, model_runner, torch, envs,
                                                    cache_config=driver.engine.cache_config)
        base.write_json(args.out_dir/f"native-runtime.rank{rank}.json", base.seal(dict(
            observed=runtime, validation=validation)))
        modules = [layer.self_attn for layer in model_runner.model.model.layers]
        if [m.layer_idx for m in modules] != list(range(40)) or dp_size != 8:
            raise RuntimeError("native forty-layer DP8 geometry required")
        binder = install_request_binding(model_runner, lambda: driver.request_binding if rank == 0 else None)
        if rank == 0:
            hook = KVCacheEngramHook(modules, model_sha256=identity["checkpoint_index_sha256"],
                tokenizer_sha256=identity["tokenizer_json_sha256"],
                execution_sha256=identity["execution_configuration_sha256"], enabled=True,
                allow_sync=allow_cuda_sync, max_cache_bytes=CACHE_BYTES)
            hook.install()
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]
        candidates = pp.candidate_policies()
        frozen = admission = None
        sensitivity_enabled = getattr(args, "sensitivity_protocol", None) is not None

        def phase(row, name, mode, *, quality=False, tokens=1, timing=False,
                  policies=None, calibrations=None, sink=None, diagnostic=False):
            if diagnostic and (timing or mode != "shadow_full" or row.get("split") != "calibration"):
                raise ValueError("sensitivity is calibration-only, full-native and untimed")
            began = time.perf_counter()
            args.mode, args.max_new_tokens = ("quality" if quality else "throughput"), tokens
            driver.phase = name
            rid = f"kce:{row.get('index', 'doc')}:{name}:{row['sample_id']}"
            if hook is not None:
                hook.sink = sink
                hook.reset(rid, mode, input_ids=row["input_ids"], documents=row.get("documents"),
                    phase=name, policies=policies or {}, calibrations=calibrations or {},
                    candidate_policies=candidates if mode == "shadow_full" else ())
            observer = None
            if diagnostic and hook is not None:
                from kv_cache_engram.sensitivity import NativeSensitivityObserver, SensitivityConfig
                observer = NativeSensitivityObserver(hook, SensitivityConfig(layers=LAYERS,
                    restore_block=True, restore_half=True, restore_control=False)).install()
                try:
                    observer.reset()
                except BaseException:
                    observer.close()
                    raise
            sensitivity_report = None
            try:
                driver.barrier()
                if rank == 0:
                    print(json.dumps(dict(case=row.get("index"), phase=name, status="starting",
                        mode=mode, input_tokens=row["input_tokens"], timing=timing)), flush=True)
                records, window = base.run_phase(driver, [(rid, row)] if rank == 0 else [],
                    int(rank == 0), rank, epoch, tokens, lambda event: None)
                if hook is not None:
                    with allow_cuda_sync():
                        hook.finish()
                        if observer is not None:
                            sensitivity_report = observer.finish()
            finally:
                if observer is not None:
                    observer.close()
            if len(records) != 1 or records[0].rank != 0 or not records[0].success(args.mode):
                raise RuntimeError("phase lacks exactly one complete real request")
            rec = base.canonical_record(records[0], driver.engine.tokenizer, args.mode, name)
            if not quality and len(rec["output_ids"]) != tokens:
                raise RuntimeError("fixed generation length mismatch")
            start = driver.gather(began-epoch)[0]
            rec.update(online_prepare_to_submit_s=rec["submit_s"]-start,
                online_ttft_s=rec["submit_s"]+rec["ttft_s"]-start,
                online_elapsed_s=rec["finish_s"]-start,
                natural_EOS=quality, truncated=bool(quality and rec.get("finish_reason") == "length"))
            report = None
            if hook is not None:
                with allow_cuda_sync():
                    report = hook.report()
            reports = driver.gather(report)
            if reports[0] is None or any(p is not None for p in reports[1:]):
                raise RuntimeError("rank0 exclusively owns Engram state")
            sensitivity_reports_by_rank = driver.gather(sensitivity_report)
            if diagnostic and (sensitivity_reports_by_rank[0] is None
                              or any(p is not None for p in sensitivity_reports_by_rank[1:])):
                raise RuntimeError("rank0 exclusively owns complete sensitivity diagnostics")
            result_body = dict(schema="kv-cache-engram-native-phase-v1", name=name, mode=mode,
                record=rec, hook=reports[0], timing_valid=timing, quality_eligible=quality,
                online_global_window_s=window["end_s"]-start,
                manifest_sha256=manifest["sha256"], execution_configuration_sha256=identity["execution_configuration_sha256"],
                input_ids_sha256=row["input_ids_sha256"], index=row.get("index"),
                split=row.get("split", "offline"), dataset=row["dataset"],
                online_gate_and_plan_included=True, offline_build_included=False)
            if diagnostic:
                result_body.update(sensitivity=sensitivity_reports_by_rank[0],
                    diagnostic_phase_wall_s=max(driver.gather(time.perf_counter()-began)),
                    diagnostic_wall_not_online_ttft=True)
            result = base.seal(result_body)
            if rank == 0:
                base.write_json(args.out_dir/f"{name}.json", result)
                print(json.dumps(dict(case=row.get("index"), phase=name, status="complete",
                    text=rec["text"] if quality else None, em=rec.get("answer_em") if quality else None,
                    f1=rec.get("answer_token_f1") if quality else None,
                    online_ttft_s=rec["online_ttft_s"] if timing else None)), flush=True)
            driver.barrier()
            return result

        # New split projection/gather path qualification; v8 alignment is not reused.
        qualification = phase(manifest["warmup"][0], "qualification_split_identity", "split_identity")
        if qualification["hook"].get("identity_exact") is not True:
            raise RuntimeError("new Q/KV split and gathered KV qualification did not pass")
        if rank == 0:
            base.write_json(args.out_dir/"qualification.json", base.seal(dict(
                passed=True, phase_sha256=qualification["sha256"], quality_claim=False)))

        for row in manifest["warmup"] + manifest["cohort"]:
            index, is_cal = row["index"], row["split"] == "calibration"
            if hook is not None:
                hook.clear_cache()
            if rank == 0 and is_cal:
                raw_writer = RawPairWriter(args.out_dir/f"case{index}.paired-kv",
                    documents=[d["document_id"] for d in row["documents"]], layers=LAYERS,
                    identity=dict(input_ids_sha256=row["input_ids_sha256"],
                        model_sha256=identity["checkpoint_index_sha256"],
                        execution_sha256=identity["execution_configuration_sha256"],
                        source_exposure=row["source_exposure"], split=row["split"]))
            build_start = time.perf_counter()
            builds = []
            for doc in document_jobs(base, row):
                builds.append(phase(doc, f"case{index}.offline_doc{doc['offline_document_index']}",
                                    "capture_independent", sink=raw_writer))
            driver.barrier()
            build_s = max(driver.gather(time.perf_counter()-build_start))
            reference = phase(row, f"case{index}.reference_shadow", "shadow_full", quality=True,
                              tokens=64, sink=raw_writer, diagnostic=sensitivity_enabled and is_cal)
            raw_manifest = None
            if raw_writer is not None:
                raw_manifest = raw_writer.finish()
                raw_writer = None
            # Shadow is a clean, full-native reference. Its gate decisions never change its KV.
            wrapped = pp.wrap_shadow(reference, row, manifest, identity)
            if is_cal:
                calibration_reports.append(wrapped)
                if rank == 0:
                    base.write_json(args.out_dir/f"case{index}.calibration.json", base.seal(dict(
                        reference_sha256=reference["sha256"], report=wrapped,
                        offline_build_s=build_s, raw_pair_sha256=raw_manifest["sha256"] if raw_manifest else None)))
                if sensitivity_enabled:
                    from gpu_pilot import sensitivity_protocol as sp
                    sensitivity = sp.wrap_phase(reference, pp.expected_case(row),
                        manifest_sha256=manifest["sha256"],
                        execution_sha256=identity["execution_configuration_sha256"])
                    sensitivity_reports.append(sensitivity)
                    if rank == 0:
                        base.write_json(args.out_dir/f"case{index}.sensitivity.json", sensitivity)
                if index == 1:
                    frozen = pp.fit_shadow(calibration_reports,
                        expected_cases=[pp.expected_case(r) for r in manifest["warmup"]],
                        scope=wrapped["scope"])
                    if sensitivity_enabled:
                        numerical = frozen
                        admission = sp.freeze_admission_plan(sensitivity_reports,
                            [pp.expected_case(r) for r in manifest["warmup"]],
                            numerical_calibration_sha256=numerical["sha256"])
                        frozen = protect_calibrated_layers(numerical, admission)
                        if rank == 0:
                            base.write_json(args.out_dir/"calibration.numerical.json", numerical)
                            base.write_json(args.out_dir/"sensitivity.admission.json", admission)
                    if rank == 0:
                        base.write_json(args.out_dir/"calibration.frozen.json", frozen)
                continue
            if frozen is None:
                raise RuntimeError("no sealed calibration before evaluation")
            evaluation_reports.append(wrapped)
            policies, calibrations = {}, {}
            if hook is not None:
                policies, calibrations = pp.make_pilot_records(frozen, hook.cache_identities())
            gated = phase(row, f"case{index}.gated_quality", "gated", quality=True, tokens=64,
                          policies=policies, calibrations=calibrations)
            for _, arm in timing_schedule(index)[:2]:
                phase(row, f"case{index}.{arm}_warmup", "native" if arm == "recomputed" else "gated",
                      tokens=32, policies=policies, calibrations=calibrations)
            measured = []
            for repeat, arm in timing_schedule(index):
                item = phase(row, f"case{index}.{arm}_measure{repeat}", "native" if arm == "recomputed" else "gated",
                    tokens=32, timing=True, policies=policies, calibrations=calibrations)
                measured.append(base.seal(dict({k: v for k, v in item.items() if k != "sha256"},
                    arm=arm, repeat=repeat, raw_phase_sha256=item["sha256"])))
            result = base.seal(dict(index=index, dataset=row["dataset"], sample_id=row["sample_id"],
                references=row["references"], question=row["question"], input_tokens=row["input_tokens"],
                reference=reference, gated=gated, timing_phases=measured,
                timing_summary=summarize_timing(measured, index), offline_build_s=build_s,
                calibration_sha256=frozen["sha256"], all_negative_results_retained=True,
                sensitivity_admission_plan_sha256=admission["sha256"] if admission else None,
                no_native_Q_attention_MoE_skipped=True))
            completed.append(result)
            if rank == 0:
                base.write_json(args.out_dir/f"case{index}.paired.json", result)
        final_audit = pp.validate_shadow(frozen, evaluation_reports,
                                        expected_cases=[pp.expected_case(r) for r in manifest["cohort"]])
        args.mode, args.max_new_tokens = "quality", 64
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        if rank == 0:
            base.write_json(args.out_dir/"run.json", base.seal(dict(schema=SCHEMA, status="complete",
                identity=identity, manifest_sha256=manifest["sha256"], calibration=frozen,
                heldout_shadow_audit=final_audit, cases=completed, engine_init_s=init_s,
                sensitivity_admission=admission,
                model_load_included=False, offline_build_included=False,
                population_accuracy_certified=False, candidate_is_KV_only=True,
                v8_changed=False, other_experiments_resumed=False)))
        driver.barrier()
    except BaseException as exc:
        failure = dict(schema=SCHEMA, status="failed", rank=rank, error=repr(exc),
                       traceback=traceback.format_exc()[-32768:], completed_indices=[r["index"] for r in completed])
        base.write_json(args.out_dir/f"failure.rank{rank}.json", base.seal(failure))
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup = []
        for obj, method in ((raw_writer, "close"), (hook, "close"), (binder, "restore"), (llm, "close")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except BaseException as exc:
                    cleanup.append(f"{method}: {exc!r}")
        if cleanup and not active_error:
            raise RuntimeError("pilot cleanup failed: " + "; ".join(cleanup))


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    base, native, NativeProfile, scoped_base_profile = load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args):
        validate_args(args)
        manifest, identity = base.preflight(args)
        protocol = base.read_json(args.pilot_protocol)
        rows = validate_inputs(base, manifest, protocol)
        sensitivity = validate_sensitivity_inputs(args, manifest)
        identity.update(pilot_source_sha256=source_hashes(args.experiment_root),
            native_profile=profile.identity(args), pilot_protocol_sha256=protocol["sha256"],
            workflow_stage=SCHEMA, no_redknot=False,
            execution_configuration_sha256=base.digest(dict(native_profile=profile.identity(args),
                fixed_env=identity["fixed_env"], native_source=base.source_identity(args.repo),
                pilot_source_sha256=source_hashes(args.experiment_root), protocol_sha256=protocol["sha256"])))
        if sensitivity is not None:
            identity["sensitivity_protocol_sha256"] = sensitivity["sha256"]
            identity["execution_configuration_sha256"] = base.digest(dict(
                native_and_pilot_execution_sha256=identity["execution_configuration_sha256"],
                sensitivity_protocol_sha256=sensitivity["sha256"]))
        if native.gpu_runtime_imported():
            raise RuntimeError("CPU preflight imported accelerator runtime")
        # Two full14-layer BF16 paired calibration snapshots =14GiB; allow ample margin.
        if shutil.disk_usage(args.experiment_root).free < 40 * 1024**3:
            raise RuntimeError("pilot requires at least40GiB free for immutable paired evidence")
        if args.check_only:
            print(json.dumps(dict(status="cpu_preflight_passed", gpu_initialized=False,
                cases=[dict(index=r["index"], dataset=r["dataset"], split=r["split"],
                    input_tokens=r["input_tokens"]) for r in rows], identity=identity)))
            return 0
        args.out_dir.mkdir(parents=True, exist_ok=False)
        from gpu_pilot.pilot_protocol import expected_case
        audit_cases = [dict(expected_case(r), index=r["index"], sample_id=r["sample_id"],
            question=r["question"], references=r["references"], input_tokens=r["input_tokens"])
            for r in rows]
        base.write_json(args.out_dir/"preflight.json", base.seal(dict(identity=identity,
            manifest_sha256=manifest["sha256"], protocol=protocol, audit_cases=audit_cases,
            sensitivity_protocol=sensitivity)))
        base.wait_for_release(args, manifest["sha256"])
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        from quality_process_group import launch
        launch(functools.partial(worker, args, manifest, identity), args.dp, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
