"""Finite independent early-six scout / per-layer mHC sparse pilot.

Import and check-only are CPU-only. The original supervisor releases CUDA only
after immutable preflight. No v8 module or native source is modified.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from gpu_pilot import run_pilot as support

SCHEMA = "kv-cache-engram-early-mhc-pilot-run-v2"
PHASE_SCHEMA = "kv-cache-engram-early-mhc-phase-v2"
ARMS = ("recomputed", "early20", "early50")
FRACTIONS = {"recomputed": None, "early20": .2, "early50": .5}
EVALUATION_INDICES = (2, 3, 4, 5)
CAPTURE_LAYERS = tuple(range(8)) + tuple(range(9, 14)) + tuple(range(15, 19))
SPARSE_LAYERS = (6, 7) + tuple(range(9, 14)) + tuple(range(15, 19))


def require(value, message):
    if not value:
        raise ValueError(message)


def source_hashes(root):
    from gpu_pilot.early_mhc_protocol import DEPENDENCIES
    hashes = support.source_hashes(root)
    for name, expected in DEPENDENCIES.items():
        actual = hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
        require(actual == expected, "frozen sparse operator dependency changed: "+name)
        hashes[name] = actual
    return hashes


def parse_args(argv=None):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--early-mhc-protocol", required=True, type=Path)
    known, rest = extra.parse_known_args(argv)
    args = support.parse_args(rest)
    args.early_mhc_protocol = known.early_mhc_protocol
    return args


def validate_protocol(args, manifest):
    from gpu_pilot.early_mhc_protocol import early_mhc_protocol
    value = json.loads(args.early_mhc_protocol.read_bytes())
    require(value == early_mhc_protocol(), "early-mHC protocol differs from frozen CPU-tested rules")
    if getattr(args, "sensitivity_protocol", None) is not None:
        support.validate_sensitivity_inputs(args, manifest)
    else:
        require("sensitivity_protocol_sha256" not in manifest,
                "inherited sensitivity manifest requires explicit source protocol")
    return value


def selected_rows(manifest):
    by_index = {row["index"]: row for row in manifest["cohort"]}
    require(all(i in by_index for i in EVALUATION_INDICES), "fixed early pilot cases absent")
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
    require(reference.get("mode") == "qualified_native" and candidate.get("mode") == "all_selected",
            "new full/native qualification pair required")
    for value in (reference, candidate):
        verify_phase(value, index=0, arm="qualification", timing=False, quality=False)
    require(reference["input_ids_sha256"] == candidate["input_ids_sha256"]
            and reference["record"]["output_ids"] == candidate["record"]["output_ids"],
            "all-selected output differs from native qualification")
    for key in ("execution_configuration_sha256", "early_mhc_protocol_sha256", "manifest_sha256"):
        require(reference[key] == candidate[key], "qualification execution/protocol changed")
    fingerprint = reference["controller"].get("identity_fingerprints_sha256")
    require(candidate["controller"].get("identity_exact") is True
            and type(candidate["controller"].get("identity_check_count")) is int
            and candidate["controller"]["identity_check_count"] == 80
            and reference["controller"].get("identity_check_count") == 80
            and isinstance(fingerprint, str) and len(fingerprint) == 64
            and candidate["controller"].get("identity_fingerprints_sha256") == fingerprint,
            "new per-layer output/lazy identity checks did not pass")
    return True


def verify_partial_smoke(phases, row, qualification):
    """Prove real partial execution coverage, not accuracy or speed parity."""
    require(row["index"] == EVALUATION_INDICES[0] and len(row["documents"]) == 8,
            "partial smoke uses the original first complete eight-document evaluation input")
    require(qualification.get("passed") is True and qualification.get("identity_check_count") == 80,
            "strict numerical qualification must precede partial smoke")
    require([value.get("arm") for value in phases] == list(ARMS[1:]),
            "both partial fractions must execute in declared smoke order")
    n = row["input_tokens"]
    chunks = [(start, min(start+8192, n)) for start in range(0, n, 8192)]
    bindings = set()
    for value in phases:
        arm = value["arm"]
        record = verify_phase(value, index=row["index"], arm=arm, timing=False, quality=False)
        require(value.get("mode") == "sensitive" and value.get("fraction") == FRACTIONS[arm]
                and value.get("partial_smoke") is True
                and value["input_ids_sha256"] == row["input_ids_sha256"]
                and len(record["output_ids"]) == 1 and record.get("natural_EOS") is False,
                "partial smoke input/mode/one-token scope changed")
        bindings.add((value["execution_configuration_sha256"], value["manifest_sha256"],
                      value["early_mhc_protocol_sha256"]))
        report = value["controller"]
        require(report.get("mode") == "sensitive" and report.get("input_ids_sha256") == row["input_ids_sha256"]
                and report.get("full_input_tokens") == n and report.get("first_six_full") is True
                and report.get("actual_full_row_native_mhc") is True
                and report.get("sparse_layers") == list(SPARSE_LAYERS), "partial smoke controller scope changed")
        masks = report["selection"]
        require([(m["start"], m["end"]) for m in masks] == chunks, "partial smoke chunk coverage incomplete")
        for m in masks:
            ids, scores, full = m["selected_local_rows"], m.get("mean_early_relative_l2"), m["end"]-m["start"]
            require(m["full_rows"] == full and type(ids) is list and ids == sorted(set(ids))
                    and all(type(i) is int and 0 <= i < full for i in ids)
                    and len(ids) == m["selected_rows"] == m["mandatory_rows"]+math.ceil(FRACTIONS[arm]*m["optional_rows"])
                    and m["mandatory_rows"]+m["optional_rows"] == full
                    and m["selected_rows_sha256"] == _digest(ids), "partial smoke selection/count identity invalid")
            require(type(scores) is list and len(scores) == full
                    and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in scores),
                    "partial smoke must execute finite dynamic early-six scoring")
        selected = sum(m["selected_rows"] for m in masks)*len(SPARSE_LAYERS)
        full = n*len(SPARSE_LAYERS)
        require(0 < selected < full, "smoke did not exercise any genuinely partial token path")
        adapter, counts = report["adapter"], report["adapter"]["counts"]
        require(adapter.get("native_execution") is True
                and adapter.get("both_full_row_native_mhc_norm_calls_retained") is True
                and counts.get("masked_layer_calls") == len(chunks)*len(SPARSE_LAYERS)
                and counts.get("full_rows") == full and counts.get("selected_rows") == selected
                and counts.get("inactive_attention_zero_rows") == full-selected
                and counts.get("inactive_ffn_zero_rows") == full-selected,
                "partial Q/attention/FFN/mHC execution accounting incomplete")
        expected = {(start, end, layer) for start, end in chunks for layer in SPARSE_LAYERS}
        events = adapter["events"]
        actual = {(event["chunk_start"], event["chunk_end"], event["layer"]) for event in events}
        require(actual == expected and len(events) == len(expected), "partial smoke layer/chunk events incomplete")
        by_chunk = {(m["start"], m["end"]): m for m in masks}
        for event in events:
            mask = by_chunk[(event["chunk_start"], event["chunk_end"])]
            require(event.get("full_row_native_mhc") is True and event.get("inactive_sublayer_output") == "zero"
                    and event.get("cached_hidden_or_sublayer_output") is False
                    and event["full_rows"] == mask["full_rows"] and event["selected_rows"] == mask["selected_rows"]
                    and event["plan_sha256"] == mask["plan_sha256"], "partial smoke event/mask mismatch")
    require(len(bindings) == 1, "partial smoke execution/manifest/protocol changed across arms")
    body = dict(schema="early-mhc-partial-smoke-v2", passed=True, index=row["index"],
        phase_sha256={value["arm"]: value["sha256"] for value in phases},
        input_ids_sha256=row["input_ids_sha256"], input_tokens=n,
        qualification_sha256=qualification["sha256"], quality_eligible=False, timing_valid=False,
        accuracy_guarantee=False, full_input_original_eight_documents=True,
        dynamic_early_score_and_partial_q_ffn_mhc_exercised=True, output_tokens_per_arm=1)
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
                          phase["manifest_sha256"], phase["early_mhc_protocol_sha256"]))
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
                online_early6_scout_and_plan_included=True, online_gather_scatter_and_fallback_included=True,
                model_load_included=False, offline_build_included=False, saturated_capacity_claim=False)


def verify_frozen(args, manifest, identity, profile, base, *, native_source=False):
    require(source_hashes(args.experiment_root) == identity["pilot_source_sha256"],
            "early pilot source changed after preflight")
    require(profile.identity(args) == identity["native_profile"] and base.effective_env(args) == identity["fixed_env"],
            "native configuration drift")
    disk = base.read_json(args.manifest)
    support.validate_inputs(base, disk, base.read_json(args.pilot_protocol))
    protocol = validate_protocol(args, disk)
    require(disk["sha256"] == manifest["sha256"]
            and protocol["sha256"] == identity["early_mhc_protocol_sha256"], "input/protocol changed")
    for filename, key in (("config.json", "checkpoint_config_sha256"),
                          ("model.safetensors.index.json", "checkpoint_index_sha256"),
                          ("tokenizer.json", "tokenizer_json_sha256")):
        require(base.file_digest(args.model_path/filename) == identity[key], "model/tokenizer identity changed")
    if native_source:
        require(all(identity[k] == v for k, v in base.source_identity(args.repo).items()), "native source changed")


def worker(args, manifest, identity, dp_size, rank):
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
        import torch
        from deepinfer import envs
        from deepinfer.utils import allow_cuda_sync
        from canonical_request_binding import install_request_binding
        from kv_cache_engram.early_mhc import EarlyMHCController
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
        base.write_json(args.out_dir/f"native-runtime.rank{rank}.json", base.seal(dict(observed=runtime, validation=validation)))
        layers = list(model_runner.model.model.layers)
        require([m.layer_idx for m in layers] == list(range(40)) and dp_size == 8, "native40 layer DP8 geometry required")
        binder = install_request_binding(model_runner, lambda: driver.request_binding if rank == 0 else None)
        if rank == 0:
            controller = EarlyMHCController(layers, allow_sync=allow_cuda_sync,
                model_sha256=identity["checkpoint_index_sha256"], tokenizer_sha256=identity["tokenizer_json_sha256"],
                execution_sha256=identity["execution_configuration_sha256"], dependency_root=args.experiment_root)
            controller.install()
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]

        def phase(row, name, mode, *, arm, quality=False, tokens=1, timing=False, repeat=None, fraction=None,
                  partial_smoke=False):
            began = time.perf_counter()
            args.mode, args.max_new_tokens = ("quality" if quality else "throughput"), tokens
            driver.phase = name
            rid = f"early-mhc:{row.get('index', 'doc')}:{name}:{row['sample_id']}"
            if controller is not None:
                # CPU-known row-index H2D and explicit synchronization belong
                # to online preparation: the clock starts before this reset.
                index_device = layers[0].self_attn.q_a_kv_swa_proj.weight.device if mode in ("sensitive", "all_selected") else None
                with allow_cuda_sync():
                    controller.reset(rid, mode, input_ids=row["input_ids"], documents=row.get("documents"),
                                     phase=name, fraction=fraction, index_device=index_device)
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
                    "only rank0 may own early-mHC state")
            result = base.seal(dict(schema=PHASE_SCHEMA, name=name, mode=mode, arm=arm, repeat=repeat,
                fraction=fraction, record=rec, controller=reports[0], timing_valid=timing, quality_eligible=quality,
                partial_smoke=partial_smoke,
                online_global_window_s=window["end_s"]-start, index=row.get("index"), dataset=row["dataset"],
                input_ids_sha256=row["input_ids_sha256"], manifest_sha256=manifest["sha256"],
                execution_configuration_sha256=identity["execution_configuration_sha256"],
                early_mhc_protocol_sha256=identity["early_mhc_protocol_sha256"],
                online_early6_scout_and_plan_included=True, offline_build_included=False,
                native_global_kv_and_index_preserved=True))
            if rank == 0:
                base.write_json(args.out_dir/f"{name}.json", result)
                print(json.dumps(dict(case=row.get("index"), phase=name, status="complete", arm=arm,
                    em=rec.get("answer_em") if quality else None, f1=rec.get("answer_token_f1") if quality else None,
                    online_ttft_s=rec["online_ttft_s"] if timing else None)), flush=True)
            driver.barrier()
            return result

        check_row = qualification_row(manifest)
        reference = phase(check_row, "qualification.native", "qualified_native", arm="qualification")
        candidate = phase(check_row, "qualification.all_selected", "all_selected", arm="qualification")
        qualification_passed(reference, candidate)
        qualification = base.seal(dict(schema="early-mhc-qualification-v1", passed=True,
            reference_sha256=reference["sha256"], candidate_sha256=candidate["sha256"],
            identity_check_count=candidate["controller"]["identity_check_count"], quality_claim=False,
            qualification_input_tokens=16384, full_256k_numerical_parity_claim=False,
            old_qualification_reused=False))
        if rank == 0:
            base.write_json(args.out_dir/"qualification.json", qualification)
        partial_smoke = None
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
                smoke_phases = [phase(row, f"case{index}.{arm}_partial_smoke", "sensitive", arm=arm,
                    tokens=1, fraction=FRACTIONS[arm], partial_smoke=True) for arm in ARMS[1:]]
                partial_smoke = verify_partial_smoke(smoke_phases, row, qualification)
                if rank == 0:
                    base.write_json(args.out_dir/"partial_smoke.json", partial_smoke)
            require(partial_smoke is not None and partial_smoke["passed"], "partial execution smoke missing")
            quality = {}
            for arm in ARMS:
                quality[arm] = phase(row, f"case{index}.{arm}_quality", "native" if arm == "recomputed" else "sensitive",
                    arm=arm, quality=True, tokens=64, fraction=FRACTIONS[arm])
                verify_phase(quality[arm], index=index, arm=arm, timing=False, quality=True)
            for _, arm in timing_schedule(index)[:3]:
                phase(row, f"case{index}.{arm}_warmup", "native" if arm == "recomputed" else "sensitive",
                      arm=arm, tokens=32, fraction=FRACTIONS[arm])
            measured = [phase(row, f"case{index}.{arm}_measure{repeat}", "native" if arm == "recomputed" else "sensitive",
                              arm=arm, tokens=32, timing=True, repeat=repeat, fraction=FRACTIONS[arm])
                        for repeat, arm in timing_schedule(index)]
            result = base.seal(dict(schema="early-mhc-case-paired-v1", index=index, dataset=row["dataset"],
                sample_id=row["sample_id"], references=row["references"], question=row["question"],
                input_tokens=row["input_tokens"], input_ids_sha256=row["input_ids_sha256"],
                quality=quality, timing_phases=measured, timing_summary=summarize_timing(measured, index),
                offline_build_s=build_s, offline_phase_sha256=[p["sha256"] for p in builds],
                qualification_sha256=qualification["sha256"], sparse_layers=list(SPARSE_LAYERS),
                partial_smoke_sha256=partial_smoke["sha256"],
                offline_capture_layers=list(CAPTURE_LAYERS), all_negative_results_retained=True,
                exploratory_previously_exposed_inputs=True, calibration_claim=False))
            completed.append(result)
            if rank == 0:
                base.write_json(args.out_dir/f"case{index}.paired.json", result)
        args.mode, args.max_new_tokens = "quality", 64
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        if rank == 0:
            base.write_json(args.out_dir/"run.json", base.seal(dict(schema=SCHEMA, status="complete",
                identity=identity, manifest_sha256=manifest["sha256"], qualification=qualification, cases=completed,
                partial_smoke_sha256=partial_smoke["sha256"],
                completed_case_indices=list(EVALUATION_INDICES), engine_init_s=init_s,
                calibration_claim=False, population_accuracy_certified=False, v8_changed=False,
                other_experiments_resumed=False, no_automatic_retry=True)))
        driver.barrier()
    except BaseException as exc:
        base.write_json(args.out_dir/f"failure.rank{rank}.json", base.seal(dict(schema=SCHEMA, status="failed", rank=rank,
            error=repr(exc), traceback=traceback.format_exc()[-32768:], completed_indices=[r["index"] for r in completed])))
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
            raise RuntimeError("early-mHC cleanup failed: " + "; ".join(cleanup))


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    base, native, NativeProfile, scoped_base_profile = support.load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args):
        support.validate_args(args)
        manifest, identity = base.preflight(args)
        support.validate_inputs(base, manifest, base.read_json(args.pilot_protocol))
        protocol = validate_protocol(args, manifest)
        rows = selected_rows(manifest)
        identity.update(pilot_source_sha256=source_hashes(args.experiment_root),
            native_profile=profile.identity(args), early_mhc_protocol_sha256=protocol["sha256"],
            workflow_stage=SCHEMA, no_redknot=False)
        identity["execution_configuration_sha256"] = base.digest(dict(native_profile=identity["native_profile"],
            fixed_env=identity["fixed_env"], native_source=base.source_identity(args.repo),
            pilot_source_sha256=identity["pilot_source_sha256"], protocol_sha256=protocol["sha256"]))
        require(not native.gpu_runtime_imported(), "CPU preflight imported GPU runtime")
        require(shutil.disk_usage(args.experiment_root).free >= 28*1024**3, "at least28GiB free evidence space required")
        if args.check_only:
            print(json.dumps(dict(status="cpu_preflight_passed", gpu_initialized=False, identity=identity,
                evaluation_cases=[dict(index=r["index"], dataset=r["dataset"], input_tokens=r["input_tokens"]) for r in rows],
                qualification_case=0, qualification_input_tokens=16384, calibrated=False)))
            return 0
        args.out_dir.mkdir(parents=True, exist_ok=False)
        base.write_json(args.out_dir/"preflight.json", base.seal(dict(schema=SCHEMA, identity=identity,
            manifest_sha256=manifest["sha256"], protocol=protocol, evaluation_case_indices=list(EVALUATION_INDICES),
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
