"""Three-arm, controlled SWA KV replacement and oracle repair diagnostic.

This is deliberately NOT an admission policy or a speed experiment. All native
projections execute. A supplies a frozen, answer-blind repair plan; B/C change
only current-chunk prepared SWA KV and propagate the resulting attention output.
GPU runtime is imported only after the original supervisor release gate.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from gpu_pilot import run_pilot as old
from gpu_pilot import repair_protocol as rp
from gpu_pilot import pilot_protocol as pp

FLAGS = dict(formal_admission_unchanged=True, forced_controlled_diagnostic=True,
    no_ttft_claim=True, previously_exposed_exploratory_cohort=True,
    all_negative_results_retained=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repair-protocol", type=Path, required=True)
    known, rest = parser.parse_known_args(argv)
    args = old.parse_args(rest)
    args.repair_protocol = known.repair_protocol
    return args


def audit_cases(base, manifest):
    rows = []
    for row in manifest["cohort"]:
        rows.append(dict(pp.expected_case(row), index=row["index"],
            sample_id=row["sample_id"], question=row["question"],
            references=row["references"], input_tokens=row["input_tokens"],
            previously_exposed_exploratory_cohort=True,
            document_spans=[dict(document_sha256=d["token_ids_sha256"],
                start=d["input_start"], end=d["input_start"]+len(d["token_ids"]),
                tokens=len(d["token_ids"])) for d in base.document_specs(row)]))
    rp.validate_cases(rows)
    return rows


def validate_repair(args, base, identity=None):
    protocol = base.read_json(args.repair_protocol)
    if protocol != rp.repair_protocol():
        raise ValueError("oracle repair protocol is not the frozen CPU-tested diagnostic")
    if getattr(args, "sensitivity_protocol", None) is None:
        raise ValueError("original input sensitivity binding is required")
    if identity is not None and identity.get("repair_protocol_sha256") != protocol["sha256"]:
        raise ValueError("repair execution identity drift")
    return protocol


def verify_frozen(args, manifest, identity, profile, base, *, native_source=False):
    old.verify_frozen(args, manifest, identity, profile, base, native_source=native_source)
    validate_repair(args, base, identity)


def build_triple(row, manifest, identity, protocol, plan, phases, offline_build_s):
    if [p.get("arm") for p in phases] != list(rp.ARMS):
        raise ValueError("all three arms in frozen order required")
    for arm, phase in zip(rp.ARMS, phases):
        pp._sealed(phase)
        if (phase["name"] != f"case{row['index']}.{arm}"
                or phase["plan_sha256"] != plan["sha256"]
                or phase["input_ids_sha256"] != row["input_ids_sha256"]
                or phase["timing_valid"] is not False or phase["quality_eligible"] is not True):
            raise ValueError("triple phases do not bind to the same input/plan")
    return pp.seal_record(dict(schema=rp.TRIPLE_SCHEMA, index=row["index"],
        dataset=row["dataset"], sample_id=row["sample_id"],
        input_ids_sha256=row["input_ids_sha256"], input_tokens=row["input_tokens"],
        manifest_sha256=manifest["sha256"],
        execution_configuration_sha256=identity["execution_configuration_sha256"],
        protocol_sha256=protocol["sha256"], plan_sha256=plan["sha256"],
        phase_sha256={p["arm"]: p["sha256"] for p in phases},
        offline_build_s=offline_build_s, **FLAGS))


def worker(args, manifest, identity, dp_size, rank):
    base, native, NativeProfile, scoped_base_profile = old.load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args, apply_environment=True):
        if native.gpu_runtime_imported():
            raise RuntimeError("accelerator runtime imported before guarded worker")
        verify_frozen(args, manifest, identity, profile, base)
        return worker_impl(args, manifest, identity, dp_size, rank, profile, base, native)


def worker_impl(args, manifest, identity, dp_size, rank, profile, base, native):
    sys.path.insert(0, str(args.repo/"examples"))
    sys.path.insert(0, str(args.repo))
    llm = hook = binder = controller = None
    completed = []
    protocol = validate_repair(args, base, identity)
    rows_by_index = {r["index"]: r for r in audit_cases(base, manifest)}
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
        from kv_cache_engram.native_reuse import KVCacheEngramHook
        from kv_cache_engram.oracle_repair import OracleRepairController, OracleRepairConfig
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
                    raise RuntimeError("request ownership mismatch")
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
                allow_sync=allow_cuda_sync, max_cache_bytes=old.CACHE_BYTES)
            hook.install()
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]

        def phase(row, name, mode, *, arm=None):
            began = time.perf_counter()
            quality = arm is not None
            tokens = 64 if quality else 1
            args.mode, args.max_new_tokens = ("quality" if quality else "throughput"), tokens
            driver.phase = name
            rid = f"kce-repair:{row.get('index', 'doc')}:{name}:{row['sample_id']}"
            if hook is not None:
                hook.sink = None
                hook.reset(rid, mode, input_ids=row["input_ids"], documents=row.get("documents"),
                           phase=name, policies={}, calibrations={}, candidate_policies=())
                if quality:
                    controller.reset(mode=rp.MODES[arm])
            driver.barrier()
            if rank == 0:
                print(json.dumps(dict(case=row.get("index"), phase=name, status="starting",
                    arm=arm, input_tokens=row["input_tokens"], no_ttft_claim=True)), flush=True)
            records, _ = base.run_phase(driver, [(rid, row)] if rank == 0 else [],
                int(rank == 0), rank, epoch, tokens, lambda event: None)
            report = intervention = plan = None
            if hook is not None:
                with allow_cuda_sync():
                    hook.finish()
                    report = hook.report()
                    if quality:
                        intervention = controller.finish()
                        plan = controller.export_plan()
                        rp.validate_intervention(intervention, plan, rows_by_index[row["index"]],
                            arm=arm, request_id=rid, phase=name)
            if len(records) != 1 or records[0].rank != 0 or not records[0].success(args.mode):
                raise RuntimeError("phase lacks exactly one successful native request")
            rec = base.canonical_record(records[0], driver.engine.tokenizer, args.mode, name)
            if not quality and len(rec["output_ids"]) != tokens:
                raise RuntimeError("qualification/offline output length mismatch")
            rec.update(natural_EOS=quality, truncated=bool(quality and rec.get("finish_reason") == "length"))
            gathered = driver.gather((report, intervention, plan))
            if gathered[0][0] is None or any(item != (None, None, None) for item in gathered[1:]):
                raise RuntimeError("rank0 must exclusively own diagnostic state")
            report, intervention, plan = gathered[0]
            wall_s = max(driver.gather(time.perf_counter()-began))
            result = base.seal(dict(schema=rp.PHASE_SCHEMA if quality else "kv-cache-engram-native-phase-v1",
                name=name, mode=mode, arm=arm, record=rec, hook=report, intervention=intervention,
                request_id=rid, plan_sha256=plan["sha256"] if plan else None,
                timing_valid=False, quality_eligible=quality, phase_wall_s=wall_s,
                manifest_sha256=manifest["sha256"], protocol_sha256=protocol["sha256"],
                execution_configuration_sha256=identity["execution_configuration_sha256"],
                input_ids_sha256=row["input_ids_sha256"], input_tokens=row["input_tokens"],
                index=row.get("index"), sample_id=row["sample_id"],
                split=row.get("split", "offline"), dataset=row["dataset"], **FLAGS))
            if rank == 0:
                if arm == "recomputed":
                    base.write_json(args.out_dir/f"case{row['index']}.plan.json", plan)
                base.write_json(args.out_dir/f"{name}.json", result)
                print(json.dumps(dict(case=row.get("index"), phase=name, status="complete",
                    text=rec["text"] if quality else None,
                    em=rec.get("answer_em") if quality else None,
                    f1=rec.get("answer_token_f1") if quality else None,
                    diagnostic_wall_s=wall_s, no_ttft_claim=True)), flush=True)
            driver.barrier()
            return result, plan

        qualification, _ = phase(manifest["warmup"][0], "qualification_split_identity", "split_identity")
        if qualification["hook"].get("identity_exact") is not True:
            raise RuntimeError("strict native split/gather/post-RoPE qualification failed")
        if rank == 0:
            base.write_json(args.out_dir/"qualification.json", base.seal(dict(
                passed=True, phase_sha256=qualification["sha256"], quality_claim=False)))
        for row in manifest["cohort"]:
            if hook is not None:
                hook.clear_cache()
            began = time.perf_counter()
            for doc in old.document_jobs(base, row):
                phase(doc, f"case{row['index']}.offline_doc{doc['offline_document_index']}",
                      "capture_independent")
            driver.barrier()
            offline_build_s = max(driver.gather(time.perf_counter()-began))
            if hook is not None:
                controller = OracleRepairController(hook, config=OracleRepairConfig()).install()
            phases, plan = [], None
            for arm in rp.ARMS:
                result, current = phase(row, f"case{row['index']}.{arm}", "native", arm=arm)
                if plan is not None and plan != current:
                    raise RuntimeError("oracle plan changed between A/B/C")
                plan = current
                phases.append(result)
            result = build_triple(row, manifest, identity, protocol, plan, phases, offline_build_s)
            completed.append(result)
            if rank == 0:
                base.write_json(args.out_dir/f"case{row['index']}.triple.json", result)
            if controller is not None:
                controller.close()
                controller = None
            driver.barrier()
        args.mode, args.max_new_tokens = "quality", 64
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        if rank == 0:
            base.write_json(args.out_dir/"run.json", base.seal(dict(schema=rp.RUN_SCHEMA,
                status="complete", identity=identity, manifest_sha256=manifest["sha256"],
                protocol_sha256=protocol["sha256"], cases=completed, engine_init_s=init_s,
                v8_changed=False, other_experiments_resumed=False, **FLAGS)))
        driver.barrier()
    except BaseException as exc:
        base.write_json(args.out_dir/f"failure.rank{rank}.json", base.seal(dict(
            schema=rp.RUN_SCHEMA, status="failed", rank=rank, error=repr(exc),
            traceback=traceback.format_exc()[-32768:], completed_indices=[r["index"] for r in completed])))
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup = []
        for obj, method in ((controller, "close"), (hook, "close"), (binder, "restore"), (llm, "close")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except BaseException as exc:
                    cleanup.append(f"{method}: {exc!r}")
        if cleanup and not active_error:
            raise RuntimeError("oracle repair cleanup failed: "+"; ".join(cleanup))


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    base, native, NativeProfile, scoped_base_profile = old.load_runtime(args.experiment_root)
    profile = NativeProfile(mtp_tokens=0, keep_tokens=140)
    with scoped_base_profile(base, profile, args=args):
        old.validate_args(args)
        manifest, identity = base.preflight(args)
        pilot = base.read_json(args.pilot_protocol)
        old.validate_inputs(base, manifest, pilot)
        sensitivity = old.validate_sensitivity_inputs(args, manifest)
        protocol = validate_repair(args, base)
        rows = audit_cases(base, manifest)
        sources = old.source_hashes(args.experiment_root)
        identity.update(pilot_source_sha256=sources, native_profile=profile.identity(args),
            pilot_protocol_sha256=pilot["sha256"], sensitivity_protocol_sha256=sensitivity["sha256"],
            repair_protocol_sha256=protocol["sha256"], workflow_stage=rp.RUN_SCHEMA, no_redknot=False)
        identity["execution_configuration_sha256"] = base.digest(dict(
            native_profile=profile.identity(args), fixed_env=identity["fixed_env"],
            native_source=base.source_identity(args.repo), pilot_source_sha256=sources,
            pilot_protocol_sha256=pilot["sha256"], sensitivity_protocol_sha256=sensitivity["sha256"],
            repair_protocol_sha256=protocol["sha256"]))
        if native.gpu_runtime_imported():
            raise RuntimeError("CPU preflight imported accelerator runtime")
        if shutil.disk_usage(args.experiment_root).free < 8*1024**3:
            raise RuntimeError("diagnostic requires8GiB free for immutable evidence")
        if args.check_only:
            print(json.dumps(dict(status="cpu_preflight_passed", gpu_initialized=False,
                cases=[dict(index=r["index"], dataset=r["dataset"], input_tokens=r["input_tokens"])
                       for r in rows], identity=identity, protocol=protocol, **FLAGS)))
            return 0
        args.out_dir.mkdir(parents=True, exist_ok=False)
        base.write_json(args.out_dir/"preflight.json", base.seal(dict(identity=identity,
            manifest_sha256=manifest["sha256"], protocol=protocol, audit_cases=rows,
            qualification_case=dict(pp.expected_case(manifest["warmup"][0]), index=0,
                sample_id=manifest["warmup"][0]["sample_id"],
                input_tokens=manifest["warmup"][0]["input_tokens"]),
            pilot_protocol=pilot, sensitivity_protocol=sensitivity, **FLAGS)))
        base.wait_for_release(args, manifest["sha256"])
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        from quality_process_group import launch
        launch(functools.partial(worker, args, manifest, identity), args.dp, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
