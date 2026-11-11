#!/usr/bin/env python3
"""Real fixed-input DeepInfer PR148 runner, independent of PR92 and other engines.

Native enqueue-to-first-observed-token TTFT includes native scheduling, excludes
tokenization/model initialization, and is not a kernel-only latency. QPS is the
number of successfully completed requests divided by the complete phase window.
No GPU package is imported until the authenticated external release barrier.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

PINNED_COMMIT = "731508b4282cee87f526a492c6bad8cabcc2f35c"
BOUNDARY = "native_engine_enqueue_to_native_token_observation"
PROJECTED_HEAD_POLICY = "grouped_wv_dequant_fp32_accum_bf16_z_cache_native_o_proj_v1"
FIXED_ENV = {
    "DEEPINFER_USE_NEXTN_DECODE": "0", "DEEPINFER_ENABLE_MTP_VARLEN": "0",
    "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
    "DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE": "0",
    "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",
    "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": "off",
    "DEEPINFER_ENFORCE_EAGER": "1",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seal(value):
    return {**value, "sha256": digest(value)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_compressed_json(path, value):
    """Large calibration arrays are not replicated into ordinary run reports."""
    import gzip
    with gzip.open(path, "xt", encoding="utf-8", compresslevel=3) as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def validate_manifest(manifest):
    if manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("manifest seal mismatch")
    if manifest.get("schema") not in ("di-sg-real-rag-manifest-v1", "redknot-deepinfer-real-rag-manifest-v1"):
        raise ValueError("unsupported manifest schema")
    if not manifest.get("cohort") or not isinstance(manifest.get("warmup"), list):
        raise ValueError("manifest requires cohort and explicit warmup list")
    seen = set()
    for row in manifest["cohort"] + manifest["warmup"]:
        sid, ids = row["sample_id"], row["input_ids"]
        if sid in seen:
            raise ValueError("duplicate sample or warmup overlap")
        seen.add(sid)
        if (not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 or x >= 2**31 for x in ids)
                or len(ids) != row["input_tokens"] or digest(ids) != row["input_ids_sha256"]
                or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]
                or digest(row["references"]) != row["references_sha256"]):
            raise ValueError(f"invalid input/hash for {sid}")
        if not row["references"] or any(not isinstance(x, str) for x in row["references"]):
            raise ValueError("nonempty string references required")


def active_redknot_features(environment):
    return {k: v for k, v in environment.items() if "REDKNOT" in k.upper()
            and k != "REDKNOT_GLM_RUN_ID" and v.lower() not in ("", "0", "false", "off", "none")}


def source_identity(repo):
    git = ["git", "-c", f"safe.directory={repo.resolve()}", "-C", str(repo)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != PINNED_COMMIT:
        raise ValueError(f"source revision {head} != pinned {PINNED_COMMIT}")
    diff = subprocess.check_output([*git, "diff", "--no-ext-diff", "HEAD"])
    names = subprocess.check_output([*git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    files = {}
    for raw in sorted(set(names.split(b"\0"))):
        if not raw:
            continue
        name = os.fsdecode(raw)
        path = repo / name
        if path.is_file() and path.suffix in (".py", ".pyi", ".cpp", ".hpp", ".cu", ".h", ".toml"):
            files[name] = file_digest(path)
    return {"commit": head, "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_is_modified": bool(diff), "runtime_source_files_sha256": files,
            "runtime_source_tree_sha256": digest(files),
            "includes_untracked_runtime_source": True}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--out-dir", "--output-dir", required=True, type=Path)
    p.add_argument("--mode", choices=("quality", "throughput"), default="quality")
    p.add_argument("--arm", choices=("recomputed", "profile", "attention-profile", "head-intervention", "redknot"), default="recomputed",
                   help="profile=offline/online head-output stability; attention-profile=native attention range, no offline capture")
    p.add_argument("--head-profile", type=Path)
    p.add_argument("--intervention-policy", type=Path,
                   help="sealed calibration-only candidate policy for a quality intervention, not a reuse profile")
    p.add_argument("--evaluate-calibration", action="store_true",
                   help="quality-only native/intervention parameter selection on warmup samples; never heldout certification")
    p.add_argument("--intervention-query-chunk-rows", type=int, default=256)
    p.add_argument("--head-boundary-tokens", type=int, default=128,
                   help="Profile/reuse only: fully recompute all heads for the first N tokens of every document (default 128)")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--head-cache-representation", choices=("expanded", "projected"), default="expanded",
                   help="projected: opt-in C1 quality-only grouped local-head z cache; not native MLA KV and not a qualified performance path")
    p.add_argument("--max-cache-gib", type=float, default=256)
    p.add_argument("--max-staging-mib", type=float, default=64)
    p.add_argument("--profile-max-gib", type=float, default=2)
    p.add_argument("--profile-samples-per-document", type=int, default=32)
    p.add_argument("--profile-relative-l2", type=float, default=0.05)
    p.add_argument("--profile-min-cosine", type=float, default=0.999)
    p.add_argument("--profile-min-samples", type=int, default=32)
    p.add_argument("--profile-limit", type=int)
    p.add_argument("--attention-samples-per-document", type=int, default=32)
    p.add_argument("--attention-boundary-tokens", type=int, default=128)
    p.add_argument("--attention-sink-tokens", type=int, default=4)
    p.add_argument("--attention-recent-tokens", type=int, default=512)
    p.add_argument("--diagnostic-stacks-seconds", type=float, default=0,
                   help="Explicit opt-in periodic per-rank Python stack dump (default disabled); diagnostic timing is ineligible")
    p.add_argument("--diagnostic-progress", action="store_true",
                   help="Log bounded per-rank native step entry/exit events; diagnostic runs are not benchmarks")
    p.add_argument("--diagnostic-progress-limit", type=int, default=5000)
    p.add_argument("--diagnostic-prefill-breakdown", action="store_true",
                   help="Opt-in bounded CPU/CUDA inclusive scopes; no hot-path synchronization; instrumented timings are not performance benchmarks")
    p.add_argument("--diagnostic-ep-sync", action="store_true",
                   help="Diagnostic-only CUDA sync and CPU-DP barrier before each MoE dispatch; implies progress, not periodic stack dumps")
    p.add_argument("--diagnostic-finite", action="store_true",
                   help="Diagnostic-only layer-0 finite-value probes; stops at the first nonfinite stage, never a benchmark")
    p.add_argument("--diagnostic-embedding-guard", action="store_true",
                   help="Diagnostic-only token-105818 embedding sentinel at every layer and indexer address audit; implies finite probes")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int)
    p.add_argument("--sample-id", help="Explicit single frozen heldout row for bounded-cache quality diagnostics; not a full-cohort benchmark")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--ep", type=int, default=8)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--dp", type=int)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--sampler-force-topk", type=int, default=1024)
    p.add_argument("--batch-invariant", choices=(0, 1), type=int, default=1)
    p.add_argument("--prefill-attempt-interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--cpu-ready-file", "--ready-file", type=Path)
    p.add_argument("--release-file", type=Path)
    p.add_argument("--gate-timeout", type=float, default=86400)
    p.add_argument("--allow-gpu-init", action="store_true")
    args = p.parse_args(argv)
    if args.diagnostic_embedding_guard:
        args.diagnostic_finite = True
    if args.diagnostic_ep_sync or args.diagnostic_finite:
        args.diagnostic_progress = True
        # Repeated failures coincided with periodic faulthandler ticks during
        # JIT stack inspection. Causation is not established; never introduce
        # those asynchronous dumps implicitly into another diagnostic mode.
    return args


def effective_env(args):
    return {**FIXED_ENV, "DEEPINFER_BATCH_INVARIANT": str(args.batch_invariant),
            "DEEPINFER_PREFILL_ATTEMPT_INTERVAL": str(args.prefill_attempt_interval),
            "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL": str(args.prefill_attempt_interval)}


def document_specs(row):
    """Validate authoritative token spans; never retokenize decoded text."""
    docs = row.get("documents")
    if not isinstance(docs, list) or not docs:
        raise ValueError("head profiling/reuse requires frozen document spans")
    result, previous = [], 0
    for doc in docs:
        start, end = doc["input_start"], doc["input_end"]
        if (type(start) is not int or type(end) is not int or not previous <= start < end <= len(row["input_ids"])
                or doc.get("offline_position") != 0 or doc.get("online_position") != start
                or doc.get("tokens") != end - start):
            raise ValueError("invalid independent-document positions/coverage")
        ids = tuple(row["input_ids"][start:end])
        if digest(ids) != doc["token_ids_sha256"] or doc["document_id"] != doc["token_ids_sha256"]:
            raise ValueError("frozen document token hash mismatch")
        result.append({"token_ids": ids, "input_start": start, "token_ids_sha256": digest(ids)})
        previous = end
    return result


def attention_request_metadata(row):
    document_specs(row)  # Validate frozen token hashes, spans and independent-document provenance.
    documents = tuple((doc["input_start"], doc["input_end"]) for doc in row["documents"])
    needles = []
    previous_end = 0
    for span in row.get("needle_spans", []):
        if isinstance(span, dict):
            start, end = span.get("input_start"), span.get("input_end")
        elif isinstance(span, (list, tuple)) and len(span) == 2:
            start, end = span
        else:
            raise ValueError("needle span must be an explicit full-request token [start,end) pair")
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(row["input_ids"]):
            raise ValueError("needle span outside the frozen full-request token layout")
        needles.append((start, end))
        previous_end = end
    result = {"token_ids": tuple(row["input_ids"]), "documents": documents, "needle_spans": tuple(needles)}
    if row.get("answer_query_span") is not None or row.get("trace_kind") == "teacher_forced_answer":
        span = row.get("answer_query_span")
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(x) is not int for x in span)
                or not documents[-1][1] <= span[0] < span[1] <= len(row["input_ids"])
                or row.get("trace_kind") != "teacher_forced_answer"):
            raise ValueError("invalid teacher-forced answer prediction span")
        prompt_end = row.get("original_prompt_end")
        answer_ids = row.get("teacher_forced_answer_ids")
        if (type(prompt_end) is not int or not documents[-1][1] < prompt_end < len(row["input_ids"])
                or row.get("answer_token_span") != [prompt_end, len(row["input_ids"])]
                or list(span) != [prompt_end-1, len(row["input_ids"])-1]
                or not isinstance(answer_ids, list) or answer_ids != row["input_ids"][prompt_end:]
                or row.get("original_prompt_ids_sha256") != digest(row["input_ids"][:prompt_end])
                or row.get("continuation_is_quality_evidence") is not False):
            raise ValueError("answer queries must predict exactly the verified appended answer, excluding EOS prediction")
        result["answer_query_span"] = tuple(span)
    return result


def validate_attention_profile_args(args, manifest):
    dp = args.dp if args.dp is not None else 1
    if (args.cp != 1 or not 1 <= args.concurrency <= dp <= 8
            or args.max_num_seqs != 1 or args.mode != "quality"):
        raise ValueError("attention-profile requires CP1, quality mode, 1<=concurrency<=DP<=8 and --max-num-seqs 1")
    if args.skip_warmup or args.limit is not None or args.repeat != 1:
        raise ValueError("attention-profile uses calibration-only --profile-limit; no heldout --limit/repeat/skip-warmup")
    if not manifest["warmup"] or (args.profile_limit is not None and not 1 <= args.profile_limit <= len(manifest["warmup"])):
        raise ValueError("attention-profile requires a nonempty declared calibration prefix")
    if args.head_profile is not None or args.cache_dir is not None:
        raise ValueError("attention-profile performs native all-head execution, no head profile or offline cache")
    if (args.attention_samples_per_document < 1 or args.attention_boundary_tokens < 0
            or args.attention_sink_tokens < 0 or args.attention_recent_tokens < 1):
        raise ValueError("invalid attention-profile sampling or streaming-window settings")
    for row in manifest["warmup"][:args.profile_limit]:
        attention_request_metadata(row)


def validate_intervention_policy(value, identity, manifest, *, calibration=False):
    # Standard-library-only provenance validation before the GPU release gate.
    from select_head_intervention_candidates import validate_policy
    return validate_policy(value, expected_identity=identity, evaluation_manifest=manifest,
                           split="calibration" if calibration else "heldout")


def validate_head_profile(value, identity, manifest, *, boundary_tokens=None):
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("head profile seal mismatch")
    if value.get("schema") != "redknot-deepinfer-head-profile-v1":
        raise ValueError("unsupported head profile schema")
    if boundary_tokens is not None and value.get("head_boundary_tokens") != boundary_tokens:
        raise ValueError("head profile boundary policy differs; reprofile with the requested --head-boundary-tokens")
    for key in ("checkpoint_config_sha256", "checkpoint_index_sha256", "tokenizer_json_sha256"):
        if value.get(key) != identity[key]:
            raise ValueError(f"head profile {key} does not identify this checkpoint")
    calibration = value.get("calibration_sample_ids")
    allowed = {r["sample_id"] for r in manifest["warmup"]}
    heldout = {r["sample_id"] for r in manifest["cohort"]}
    if not calibration or len(set(calibration)) != len(calibration) or set(calibration) - allowed or set(calibration) & heldout:
        raise ValueError("profile calibration must use disjoint declared warmup samples")
    if value.get("calibration_manifest_sha256") != manifest["sha256"]:
        raise ValueError("head profile must bind the frozen calibration manifest")
    layers = value.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("head profile requires every model layer")
    for i, layer in enumerate(layers):
        heads, local = layer.get("num_heads"), layer.get("local_heads")
        if (layer.get("layer") != i or type(heads) is not int or heads < 1 or not isinstance(local, list)
                or any(type(h) is not int or not 0 <= h < heads for h in local) or len(set(local)) != len(local)):
            raise ValueError("invalid explicit per-(layer,head) profile")
        if (i < 3 or i >= len(layers) - 3) and local:
            raise ValueError("first three and last three layers must fully recompute")
    if not any(x["local_heads"] for x in layers):
        raise ValueError("measured profile has no reusable heads; cannot claim a reuse arm")
    return value


def validate_head_cache_representation(args):
    """CPU-only admission before holder release; old callers retain expansion."""
    representation = getattr(args, "head_cache_representation", "expanded")
    if representation not in ("expanded", "projected"):
        raise ValueError("unknown head cache representation")
    if representation == "projected" and (
            args.arm != "redknot" or args.mode != "quality" or args.concurrency != 1
            or args.cp != 1 or args.max_num_seqs != 1 or args.head_profile is None):
        raise ValueError("projected head cache requires measured-profile redknot quality/C1/CP1 and max-num-seqs1")
    return representation


def projected_dimensions(cfg, heads):
    groups, rank = cfg.get("o_groups"), cfg.get("o_lora_rank")
    if (type(groups) is not int or groups < 1 or type(rank) is not int or rank < 1
            or type(heads) is not int or heads < 1 or heads % groups):
        raise ValueError("projected head cache requires valid loaded o_groups/o_lora_rank geometry")
    return groups, rank


def projected_checkpoint_text_config(config):
    """Select the HF text geometry exactly where DeepInfer's loader reads it.

    The loaded model.cfg is already flattened, but V4.1's HF config.json wraps
    these fields in text_config. Never fill missing text fields from vision or
    silently choose between contradictory outer/text geometry declarations.
    """
    if not isinstance(config, dict):
        raise ValueError("projected checkpoint config must be an object")
    if "text_config" not in config:
        if "vision_config" in config:
            raise ValueError("wrapped projected checkpoint requires text_config")
        return config
    text = config["text_config"]
    if not isinstance(text, dict) or not text or "text_config" in text:
        raise ValueError("projected checkpoint text_config must be a nonempty flat object")
    for key in ("num_hidden_layers", "num_attention_heads", "head_dim", "o_groups", "o_lora_rank"):
        if key in config and (key not in text or type(config[key]) is not type(text[key]) or config[key] != text[key]):
            raise ValueError(f"conflicting outer/text_config projected geometry: {key}")
    return text


def projected_storage_geometry(cfg, calibration_geometry, cache_abi):
    """Actual payload ABI is distinct from the unchanged calibration geometry."""
    groups, rank = projected_dimensions(cfg, calibration_geometry["heads"])
    if not isinstance(cache_abi, str) or not cache_abi or cache_abi == calibration_geometry["cache_abi"]:
        raise ValueError("projected storage needs a distinct explicit cache ABI")
    return {"representation": "projected", "cache_abi": cache_abi,
            "shape_tail": [groups, rank], "num_output_groups": groups, "projection_dim": rank,
            "dtype": "bfloat16", "bytes_per_token_per_reused_layer": groups * rank * 2,
            "boundary": "post_inverse_rope_grouped_wv_local_contribution_before_native_o_proj",
            "projection_policy": PROJECTED_HEAD_POLICY,
            "offline_rounding": "FP32 grouped local contribution rounded once to BF16 cache",
            "online_merge": "cached BF16 local contribution plus FP32 online contribution; one final BF16 round before native output projection",
            "is_native_mla_kv": False}


def selected_heldout_row(args, manifest):
    """Select an original sealed row, never rewrite the manifest or its split."""
    sample_id = getattr(args, "sample_id", None)
    if sample_id is None:
        return None
    if (not isinstance(sample_id, str) or not sample_id or args.mode != "quality"
            or args.arm not in ("recomputed", "redknot") or args.concurrency != 1
            or args.cp != 1 or args.max_num_seqs != 1 or not args.skip_warmup
            or args.limit is not None or args.repeat != 1 or args.evaluate_calibration
            or args.profile_limit is not None):
        raise ValueError("sample-id requires single heldout quality C1/CP1, max-num-seqs1, skip-warmup and no other selectors/repeats")
    rows = [row for row in manifest["cohort"] if row["sample_id"] == sample_id]
    if (len(rows) != 1 or any(row["sample_id"] == sample_id for row in manifest["warmup"])
            or rows[0].get("split") != "heldout" or rows[0].get("synthetic_probe", False)
            or rows[0].get("trace_kind") == "teacher_forced_answer"
            or rows[0].get("continuation_is_quality_evidence") is False):
        raise ValueError("sample-id must identify exactly one natural heldout row in the frozen manifest")
    return rows[0]


def preflight(args):
    representation = validate_head_cache_representation(args)
    manifest = read_json(args.manifest)
    validate_manifest(manifest)
    selected = selected_heldout_row(args, manifest)
    if min(args.ep, args.cp, args.concurrency, args.max_new_tokens, args.repeat,
           args.max_num_seqs, args.max_num_batched_tokens, args.prefill_attempt_interval) < 1:
        raise ValueError("all capacities/counts must be positive")
    if not math.isfinite(args.diagnostic_stacks_seconds) or args.diagnostic_stacks_seconds < 0 or args.diagnostic_progress_limit < 1:
        raise ValueError("diagnostic stack period must be nonnegative and progress limit positive")
    if (args.diagnostic_ep_sync or args.diagnostic_finite) and args.cp != 1:
        raise ValueError("EP synchronization diagnostics currently require CP1")
    if getattr(args, "diagnostic_prefill_breakdown", False) and (
            args.cp != 1 or args.mode != "quality" or args.concurrency != 1
            or args.arm not in ("recomputed", "redknot")):
        raise ValueError("prefill breakdown requires native/reuse quality C1/CP1")
    args.dp = args.dp if args.dp is not None else max(args.ep // args.cp, 1)
    if args.dp < 1 or args.dp * args.cp % args.ep:
        raise ValueError("dp*cp must be a positive multiple of ep")
    if args.max_num_seqs < math.ceil(args.concurrency / args.dp):
        raise ValueError("per-rank max_num_seqs insufficient for requested concurrency")
    if args.mode == "quality" and args.concurrency != 1 and args.arm not in ("attention-profile", "profile"):
        raise ValueError("initial quality gate requires C1")
    if args.evaluate_calibration:
        if (args.arm not in ("recomputed", "head-intervention", "redknot") or args.mode != "quality"
                or args.skip_warmup or args.limit is not None or args.repeat != 1
                or not manifest["warmup"]):
            raise ValueError("calibration evaluation requires native/intervention/reuse quality mode and a declared warmup split")
        if args.profile_limit is not None and not 1 <= args.profile_limit <= len(manifest["warmup"]):
            raise ValueError("calibration profile-limit outside warmup split")
    if args.intervention_policy is not None and args.arm != "head-intervention":
        raise ValueError("only head-intervention may consume an intervention policy")
    if args.limit is not None and not 1 <= args.limit <= len(manifest["cohort"]):
        raise ValueError("limit outside source-order cohort")
    if args.max_model_len < max(32768, max(r["input_tokens"] for r in manifest["cohort"] + manifest["warmup"]) + args.max_new_tokens):
        raise ValueError("max_model_len below native model minimum or request length")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be in (0,1)")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0:
        raise ValueError("one-node driver requires one shared monotonic clock")
    active = active_redknot_features(os.environ)
    if active:
        raise ValueError(f"native recomputed baseline refuses active RedKnot keys: {sorted(active)}")
    tokenizer = args.model_path / "tokenizer.json"
    expected = manifest["tokenizer"].get("tokenizer_json_sha256")
    if not expected or not tokenizer.is_file() or file_digest(tokenizer) != expected:
        raise ValueError("checkpoint tokenizer does not match frozen manifest")
    config_path = args.model_path / "config.json"
    config = read_json(config_path)
    if "v4" not in str(config.get("model_type", "")).lower():
        raise ValueError("checkpoint is not a DeepSeek V4-family configuration")
    index_path = args.model_path / "model.safetensors.index.json"
    index = read_json(index_path)
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (args.model_path / x).is_file() or (args.model_path / x).stat().st_size == 0 for x in shards):
        raise ValueError("checkpoint has missing or empty safetensors shards")
    identity = {**source_identity(args.repo), "engine": "deepinfer-pr148", "arm": args.arm,
                      "checkpoint_path": str(args.model_path.resolve()),
                      "checkpoint_config_sha256": file_digest(config_path),
                      "checkpoint_index_sha256": file_digest(index_path), "checkpoint_shards": len(shards),
                      "weight_identity_limit": "index hash and shard presence/size checked; full weight hashing not performed",
                      "tokenizer_json_sha256": expected, "fixed_env": effective_env(args),
                      "manifest_artifact_kind": manifest.get("artifact_kind", "unspecified"),
                      "calibration_contains_synthetic_probes": any(row.get("synthetic_probe", False) for row in manifest["warmup"]),
                      "layout": {"ep": args.ep, "cp": args.cp, "dp": args.dp, "gpus": args.dp * args.cp},
                      "adapter_sha256": file_digest(__file__), "sampling": sampling(args),
                      "argv": sys.argv, "no_redknot": args.arm in ("recomputed", "attention-profile"), "model_init_in_measurement": False,
                      "evaluation_split": "calibration" if args.evaluate_calibration or args.arm in ("profile", "attention-profile") else "heldout",
                      "diagnostics": {"stacks_seconds": args.diagnostic_stacks_seconds,
                                      "progress": args.diagnostic_progress,
                                      "ep_sync": args.diagnostic_ep_sync,
                                      "finite": args.diagnostic_finite,
                                      "embedding_guard": args.diagnostic_embedding_guard,
                                      "progress_event_limit_per_rank": args.diagnostic_progress_limit},
                      "cache_policy": "fresh native requests, no cross-request prefix reuse; native within-request decode cache retained",
                      "scheduling_policy": "fixed disclosed prefill attempt intervals, identical between compared arms; not a maximum-throughput tuning claim"}
    if selected is not None:
        identity["sample_selection"] = {"policy": "explicit_frozen_sample_id_v1", "split": "heldout",
            "sample_ids": [selected["sample_id"]], "parent_manifest_sha256": manifest["sha256"]}
    if getattr(args, "diagnostic_prefill_breakdown", False):
        identity["diagnostics"]["prefill_breakdown"] = True
        identity["prefill_diagnostics_sha256"] = file_digest(Path(__file__).with_name("pr148_prefill_diagnostics.py"))
    if args.arm in ("profile", "redknot"):
        if args.head_boundary_tokens < 0:
            raise ValueError("head-boundary-tokens must be nonnegative")
        if args.cp != 1 or args.concurrency > args.dp or args.max_num_seqs != 1:
            raise ValueError("experimental head reuse requires CP1, concurrency <= DP and --max-num-seqs 1")
        if args.arm == "profile" and (args.mode != "quality" or args.skip_warmup
                                      or not 1 <= args.concurrency <= args.dp <= 8):
            raise ValueError("profiling requires quality mode, 1<=concurrency<=DP<=8 and declared calibration warmup")
        if args.arm == "profile" and (args.limit is not None or args.repeat != 1):
            raise ValueError("profiling uses --profile-limit on calibration samples, not heldout --limit/--repeat")
        if args.profile_limit is not None and not 1 <= args.profile_limit <= len(manifest["warmup"]):
            raise ValueError("profile-limit outside calibration sample count")
        if (any(not math.isfinite(x) or x <= 0 for x in (args.max_cache_gib, args.max_staging_mib, args.profile_max_gib))
                or args.profile_samples_per_document < 1 or args.profile_min_samples < 1
                or not math.isfinite(args.profile_relative_l2) or args.profile_relative_l2 < 0
                or not -1 <= args.profile_min_cosine <= 1):
            raise ValueError("invalid head cache/profile budgets or thresholds")
        for row in manifest["warmup"] + manifest["cohort"]:
            document_specs(row)
        if args.arm == "redknot":
            if args.head_profile is None:
                raise ValueError("RedKnot requires a measured checkpoint-bound --head-profile")
            args.head_profile_data = validate_head_profile(read_json(args.head_profile), identity, manifest,
                                                          boundary_tokens=args.head_boundary_tokens)
            identity["head_profile_file_sha256"] = file_digest(args.head_profile)
            identity["head_profile_seal"] = args.head_profile_data["sha256"]
        elif args.head_profile is not None:
            raise ValueError("profiling executes all heads; do not supply a reuse head profile")
        identity["reuse_scope"] = "query-head output reuse; MLA/SWA/index KV, projections and MoE remain native"
        identity["head_boundary_tokens"] = args.head_boundary_tokens
        identity["cache_policy"] = "independent position-zero document capture, no native prefix shortcut; hot online includes head restoration/merge"
        if representation == "projected":
            text_config = projected_checkpoint_text_config(config)
            projected_dimensions(text_config, text_config.get("num_attention_heads"))
            identity.update(head_cache_representation="projected", head_projection_policy=PROJECTED_HEAD_POLICY,
                            reuse_scope="projected local-head z reuse with selected online grouped-Wv contributions; native MLA/SWA/index KV and MoE recomputed; not compact latent restoration",
                            cache_representation_qualified=False,
                            cache_policy="independent position-zero document capture of BF16 grouped local-head z; hot online includes restore and FP32 contribution merge")
    elif args.arm == "attention-profile":
        validate_attention_profile_args(args, manifest)
        identity["attention_profile_policy"] = {
            "samples_per_document": args.attention_samples_per_document, "boundary_tokens": args.attention_boundary_tokens,
            "sink_tokens": args.attention_sink_tokens, "recent_tokens": args.attention_recent_tokens,
            "all_layers_and_heads_native": True, "offline_head_capture": False,
            "classification": "observations only; no automatic reusable-head selection"}
    elif args.arm == "head-intervention":
        if (args.cp != 1 or args.concurrency != 1 or args.max_num_seqs != 1
                or args.mode != "quality" or args.head_profile is not None or args.cache_dir is not None
                or args.intervention_policy is None or args.intervention_query_chunk_rows < 1):
            raise ValueError("head-intervention requires a policy, quality/C1/CP1, max-num-seqs1 and no cache/reuse profile")
        if not math.isfinite(args.max_staging_mib) or args.max_staging_mib <= 0:
            raise ValueError("head-intervention staging budget must be positive")
        for row in manifest["warmup"] + manifest["cohort"]:
            document_specs(row)
        args.intervention_policy_data = validate_intervention_policy(
            read_json(args.intervention_policy), identity, manifest,
            calibration=args.evaluate_calibration)
        identity["head_intervention_policy_seal"] = args.intervention_policy_data["sha256"]
        identity["head_intervention_policy_file_sha256"] = file_digest(args.intervention_policy)
        identity["head_intervention_scope"] = "document interior prefill only; native prompt boundaries, question and decode; MoE unchanged; NOT independent-document reuse"
        identity["no_redknot"] = True
    elif args.head_profile is not None:
        raise ValueError("recomputed arm cannot use a head profile")
    return manifest, identity


def sampling(args):
    return {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "max_new_tokens": args.max_new_tokens, "ignore_eos": args.mode == "throughput",
            "seq_seed": args.seed}


def wait_for_release(args, manifest_sha):
    if args.allow_gpu_init:
        if args.cpu_ready_file or args.release_file:
            raise ValueError("choose explicit bypass OR the authenticated release pair")
        return
    if not args.cpu_ready_file or not args.release_file:
        raise ValueError("GPU initialization requires CPU-ready/release paths")
    if args.release_file.exists():
        raise ValueError("stale release sentinel already exists")
    ready_time = time.time()
    write_json(args.cpu_ready_file, {"status": "cpu-ready-no-torch-import", "phase": "pre_cuda_ready",
                                  "pid": os.getpid(), "manifest_sha256": manifest_sha, "time_unix_s": ready_time})
    print(f"CPU_READY {args.cpu_ready_file}", flush=True)
    deadline = time.monotonic() + args.gate_timeout
    while not args.release_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU release gate timed out without GPU initialization")
        time.sleep(0.25)
    release = read_json(args.release_file)
    if release.get("pid") != os.getpid() or type(release.get("released_unix")) not in (int, float):
        raise ValueError("release does not authorize this process")
    if not math.isfinite(release["released_unix"]) or release["released_unix"] < ready_time:
        raise ValueError("release timestamp stale or invalid")


def rank_budget(concurrency, dp, rank):
    active = min(concurrency, dp)
    return 0 if rank >= active else concurrency // active + int(rank < concurrency % active)


def patch_dispatch_for_diagnostics(moe_class, *, synchronize, barrier, allow_sync, emit):
    """Subprocess-local diagnostic patch; dependency injection keeps CPU tests real.

    Synchronization precedes the native EP launch so an earlier CUDA failure or
    stall is not misreported as another rank's secondary dispatch timeout. No
    change to the original dispatch arguments, outputs, or communicator timeouts.
    """
    original = moe_class.dispatch
    if getattr(original, "_redknot_diagnostic_ep_sync", False):
        raise ValueError("EP diagnostic patch is already installed")
    call_count = 0

    @functools.wraps(original)
    def wrapped(module, hidden_states, topk_idx, topk_weights, input_metadata, dispatch_pre_evt):
        nonlocal call_count
        call_count += 1
        tensor = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        fields = {"dispatch_index": call_count, "layer": int(module.layer_idx),
                  "is_prompt": bool(input_metadata.is_prompt), "qshape": list(tensor.shape),
                  "topk_shape": list(topk_idx.shape)}
        stage = "ep_sync_enter"
        try:
            emit(stage, **fields)
            with allow_sync():
                synchronize()
            stage = "ep_sync_done"
            emit(stage, **fields)
            stage = "ep_cpu_barrier_enter"
            emit(stage, **fields)
            barrier()
            stage = "ep_cpu_barrier_done"
            emit(stage, **fields)
            stage = "ep_native_dispatch_enter"
            emit(stage, **fields)
            result = original(module, hidden_states, topk_idx, topk_weights, input_metadata, dispatch_pre_evt)
            emit("ep_native_dispatch_returned", **fields)
            return result
        except BaseException as exc:
            emit("ep_diagnostic_failure", failed_stage=stage, error=f"{type(exc).__name__}: {exc}", **fields)
            raise

    wrapped._redknot_diagnostic_ep_sync = True
    moe_class.dispatch = wrapped

    def restore():
        if moe_class.dispatch is not wrapped:
            raise RuntimeError("MoE dispatch changed after installing diagnostic patch; refusing to overwrite another patch")
        moe_class.dispatch = original

    return restore


def install_ep_sync_diagnostic(stream, args, rank):
    """Call only after rank-specific CUDA_VISIBLE_DEVICES has been configured."""
    import torch
    import torch.distributed as dist
    from deepinfer.model_executor.models.v4_modules.mlp import DeepseekMoE
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    from deepinfer.utils import allow_cuda_sync
    events = 0

    def emit(stage, **values):
        nonlocal events
        if events > args.diagnostic_progress_limit:
            return
        limited = events == args.diagnostic_progress_limit
        events += 1
        item = {"stage": "event_limit_reached" if limited else stage, "rank": rank,
                "pid": os.getpid(), "time_monotonic_s": time.perf_counter(),
                **({} if limited else values)}
        stream.write(json.dumps(item, allow_nan=False) + "\n")
        stream.flush()

    def barrier():
        group = PG.cpu_data_parallel_group()
        if group.size() > 1:
            dist.barrier(group=group)

    return patch_dispatch_for_diagnostics(DeepseekMoE,
        synchronize=lambda: torch.cuda.current_stream().synchronize(), barrier=barrier,
        allow_sync=allow_cuda_sync, emit=emit)


def patch_finite_method_for_diagnostics(owner, name, *, stage, before, after, check, inner_hook=None, predicate=None):
    """Layer-0-only, read-only diagnostic wrapper, testable without importing CUDA."""
    import inspect
    original = getattr(owner, name)
    if getattr(original, "_redknot_diagnostic_finite", False):
        raise ValueError("finite diagnostic patch is already installed")
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(module, *args, **kwargs):
        selected = predicate(module) if predicate is not None else getattr(module, "layer_idx", None) == 0
        if not selected:
            return original(module, *args, **kwargs)
        bound = signature.bind(module, *args, **kwargs).arguments
        if before is not None:
            check(stage + ".input", before(bound), bound)
        hook = inner_hook(module, bound) if inner_hook is not None else None
        try:
            output = original(module, *args, **kwargs)
            if after is not None:
                check(stage + ".output", after(output), bound)
            return output
        finally:
            if hook is not None:
                hook.remove()

    wrapped._redknot_diagnostic_finite = True
    setattr(owner, name, wrapped)

    def restore():
        if getattr(owner, name) is not wrapped:
            raise RuntimeError("finite-probed method changed; refusing to overwrite another patch")
        setattr(owner, name, original)
    return restore


class FiniteMetadataContext:
    """VL prefill embeds before DeepseekModel.forward; count that call once."""
    def __init__(self):
        self.metadata = None
        self.model_calls = 0
        self.pending_vl_model = False

    def bind(self, stage, metadata):
        if stage == "vl_embedding.input":
            self.model_calls += 1
            self.pending_vl_model = True
        elif stage == "model.input":
            if not (self.pending_vl_model and metadata is self.metadata):
                self.model_calls += 1
            self.pending_vl_model = False
        self.metadata = metadata


def historical_i32_output_overlap(out_ptr, rows, width, stride, element_size, target_ptr, target_bytes):
    """Audit (OUT + int32(row * stride)) + columns, respecting left association.

    Only row*stride is signed-int32 arithmetic. Once added to OUT, columns
    advance a pointer; crossing 2**31 within that row does not wrap again.
    """
    if min(rows, width, stride, target_bytes) < 0 or element_size < 1:
        raise ValueError("invalid diagnostic address geometry")
    modulus, half = 1 << 32, 1 << 31
    hits, overflow_rows = [], 0
    first = None
    for row in range(rows):
        raw_start = row * stride
        wrapped_start = (raw_start + half) % modulus - half
        if wrapped_start == raw_start or width == 0:
            continue
        overflow_rows += 1
        if first is None:
            first = row
        byte_start = out_ptr + wrapped_start * element_size
        byte_end = byte_start + width * element_size
        lo, hi = max(byte_start, target_ptr), min(byte_end, target_ptr + target_bytes)
        if lo < hi:
            hits.append({"output_row": row, "raw_row_offset": raw_start, "wrapped_row_offset": wrapped_start,
                         "column_start": 0, "column_end_exclusive": width,
                         "wrapped_byte_start": byte_start, "wrapped_byte_end_exclusive": byte_end,
                         "target_overlap_bytes": hi - lo})
    return {"signed_i32_limit": half, "overflow_output_rows": overflow_rows,
            "expression": "(OUT + int32(row * stride)) + columns",
            "first_overflow_output_row": first, "target_overlap_rows": len(hits),
            "target_overlap_bytes_sum": sum(hit["target_overlap_bytes"] for hit in hits), "overlaps": hits}


def install_finite_diagnostic(stream, args, rank):
    """Bounded scratch, full row reporting; no repair, no extra DP collectives.

    Called before model initialization, after CUDA device visibility is set.
    An offending rank raises before the next kernel; the external supervisor
    terminates the other TP/EP ranks. This avoids conditional collective hangs.
    """
    import torch
    from deepinfer.model_executor.models.decoder_layer import DeepseekDecoderLayer
    from deepinfer.model_executor.models.deepseek import DeepseekModel
    from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM
    from deepinfer.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from deepinfer.model_executor.models.v4_modules.hc import HCLayerMixin
    from deepinfer.model_executor.models.v4_modules.attn import V4Attention, V4AttentionBase
    from deepinfer.model_executor.models.v4_modules.mlp import DeepseekMoE
    from deepinfer.utils import allow_cuda_sync
    restores, metadata_context, counters = [], FiniteMetadataContext(), {"event": 0, "layer0_call": 0, "model_call": 0}
    embedding_context = {}
    guard_state = {"token_id": 105818, "checks": 0, "failures": 0, "baseline_sha256": None, "active_layer": None}

    def emit_guard(item, *, always=False):
        if always or counters["event"] < args.diagnostic_progress_limit:
            stream.write(json.dumps({"rank": rank, "pid": os.getpid(), "time_monotonic_s": time.perf_counter(),
                                     "model_call": counters["model_call"], **item}, allow_nan=False) + "\n")
            stream.flush()
        counters["event"] += 1

    def embedding_guard(stage, module=None):
        if not args.diagnostic_embedding_guard:
            return
        module = module or embedding_context.get("module")
        if module is None:
            return
        token = guard_state["token_id"]
        if token >= module.weight.shape[0]:
            raise ValueError("embedding sentinel token outside loaded vocabulary")
        with allow_cuda_sync():
            snapshot = row_snapshot(module.weight[token])
        sha = hashlib.sha256(bytes.fromhex(snapshot["bytes_hex"])).hexdigest()
        if guard_state["baseline_sha256"] is None:
            guard_state["baseline_sha256"] = sha
        guard_state["checks"] += 1
        bad = snapshot["nonfinite_elements"] != 0 or sha != guard_state["baseline_sha256"]
        guard_state["failures"] += int(bad)
        item = {"stage": "embedding_weight_guard", "boundary": stage, "layer": guard_state["active_layer"],
                "token_id": token, "weight_ptr": module.weight.data_ptr(),
                "row_ptr": module.weight[token].data_ptr(), "row_bytes": module.weight.shape[1] * module.weight.element_size(),
                "row_sha256": sha, "baseline_sha256": guard_state["baseline_sha256"],
                "nonfinite_elements": snapshot["nonfinite_elements"], "unchanged_and_finite": not bad,
                "check_index": guard_state["checks"]}
        if bad:
            path = args.out_dir / f"diagnostic.embedding.guard.rank{rank}.check{guard_state['checks']}.json"
            write_json(path, {**item, "row_snapshot": snapshot})
            item["artifact"] = path.name
        emit_guard(item, always=bad)
        if bad:
            raise RuntimeError(f"embedding weight sentinel corrupted at {stage}, layer {guard_state['active_layer']}")

    def row_snapshot(tensor):
        value = tensor.detach().contiguous().cpu()
        check_value = value.float()
        finite = torch.isfinite(check_value)
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "nonfinite_elements": int((~finite).sum()),
                "nan_elements": int(torch.isnan(check_value).sum()),
                "positive_inf_elements": int(torch.isposinf(check_value).sum()),
                "negative_inf_elements": int(torch.isneginf(check_value).sum()),
                "max_abs_finite": float(torch.where(finite, check_value.abs(), 0).amax()),
                "bytes_hex": bytes(value.view(torch.uint8).reshape(-1).tolist()).hex()}

    def embedding_failure_evidence(tensor, bad_rows):
        """Only on failure: preserve at most 8 raw output/source rows (~240 KiB)."""
        from safetensors import safe_open
        result = {"source": "actual pre-gather input IDs, not inferred manifest positions",
                  "bad_rows_total": len(bad_rows), "raw_rows_limit": 8, "rows": []}
        module, ids = embedding_context.get("module"), embedding_context.get("input_ids")
        if module is None or ids is None or tensor.shape[0] != len(ids):
            result["unavailable"] = "no matching embedding input snapshot"
            return result
        index = read_json(args.model_path / "model.safetensors.index.json")["weight_map"]
        names = [name for name in ("embed.weight", "model.embed.weight") if name in index]
        if len(names) != 1:
            raise ValueError("embedding diagnostic needs an unambiguous HF embed.weight")
        name = names[0]
        shard = args.model_path / index[name]
        result.update({"checkpoint_tensor": name, "checkpoint_shard": str(shard),
                       "runtime_weight_shape": list(module.weight.shape), "runtime_weight_dtype": str(module.weight.dtype),
                       "runtime_weight_stride": list(module.weight.stride()),
                       "runtime_weight_has_sf": hasattr(module.weight, "sf"),
                       "native_embedding_uses_sf": False})
        with allow_cuda_sync(), safe_open(str(shard), framework="pt", device="cpu") as source:
            view = source.get_slice(name)
            for row in bad_rows[:8]:
                token = int(ids[row])
                entry = {"row": row, "actual_token_id": token,
                         "query_position_before_embedding": embedding_context.get("positions", [None] * len(ids))[row],
                         "output": row_snapshot(tensor[row])}
                if 0 <= token < module.weight.shape[0]:
                    entry["runtime_weight"] = row_snapshot(module.weight[token])
                    entry["checkpoint_weight"] = row_snapshot(view[token:token + 1].reshape(-1))
                    entry["runtime_weight_matches_checkpoint_bytes"] = (
                        entry["runtime_weight"]["bytes_hex"] == entry["checkpoint_weight"]["bytes_hex"])
                    entry["output_matches_runtime_weight_bytes"] = (
                        entry["output"]["bytes_hex"] == entry["runtime_weight"]["bytes_hex"])
                    if hasattr(module.weight, "sf"):
                        sf = module.weight.sf
                        if isinstance(sf, torch.Tensor):
                            entry["scale_metadata"] = {"shape": list(sf.shape), "dtype": str(sf.dtype)}
                            if sf.ndim and sf.shape[0] == module.weight.shape[0]:
                                entry["scale_row"] = row_snapshot(sf[token])
                else:
                    entry["token_out_of_range"] = True
                result["rows"].append(entry)
        path = args.out_dir / f"diagnostic.embedding.rows.rank{rank}.call{counters['model_call']}.json"
        write_json(path, result)
        return {"artifact": path.name, "rows": [{k: v for k, v in r.items()
                if k not in ("output", "runtime_weight", "checkpoint_weight", "scale_row")} for r in result["rows"]]}

    def leaves(value, label):
        # Movable.move() consumes the input. Read its slot without moving it.
        if hasattr(value, "_value"):
            value = value._value
        if isinstance(value, torch.Tensor):
            yield label, value
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                yield from leaves(child, f"{label}.{index}")

    def check(stage, values, bound):
        layer = getattr(bound.get("self"), "layer_idx", None)
        if stage.startswith("decoder."):
            guard_state["active_layer"] = layer
            if layer != 0:
                embedding_guard(stage)
                return  # Other layers only inspect the 10-KiB immutable sentinel.
        if stage in ("vl_embedding.input", "model.input", "decoder.input"):
            metadata_context.bind(stage, bound.get("input_metadata"))
            counters["model_call"] = metadata_context.model_calls
        if stage == "decoder.input":
            counters["layer0_call"] += 1
        metadata = bound.get("input_metadata", metadata_context.metadata)
        positions = bound.get("positions", getattr(metadata, "positions", None))
        item = {"stage": stage, "rank": rank, "pid": os.getpid(), "layer": 0,
                "layer0_call": counters["layer0_call"], "time_monotonic_s": time.perf_counter(),
                "model_call": counters["model_call"], "metadata_object_id": id(metadata),
                "is_prompt": bool(getattr(metadata, "is_prompt", False)), "tensors": []}
        bad = False
        with allow_cuda_sync():
            if stage == "embedding.input":
                input_ids = bound["input_"].detach().reshape(-1).cpu().tolist()
                embedding_context.clear()
                embedding_context.update(module=bound["self"], input_ids=input_ids)
                if isinstance(positions, torch.Tensor) and positions.numel() == len(input_ids):
                    embedding_context["positions"] = positions.detach().reshape(-1).cpu().tolist()
                item["input_ids"] = {"count": len(input_ids), "min": min(input_ids) if input_ids else None,
                                     "max": max(input_ids) if input_ids else None, "sha256": digest(input_ids),
                                     "first_16": input_ids[:16], "last_16": input_ids[-16:],
                                     "vocab_size": int(bound["self"].weight.shape[0])}
                if embedding_context.get("positions"):
                    item["query_positions"] = {"first": embedding_context["positions"][0],
                                               "last": embedding_context["positions"][-1]}
            for label, value in values.items():
                for leaf_label, tensor in leaves(value, label):
                    if not tensor.is_floating_point():
                        continue  # Packed integer payloads/scales have no NaN representation.
                    rows = int(tensor.shape[0]) if tensor.ndim else 1
                    flat = tensor.detach().reshape(rows, -1) if rows else tensor.detach().reshape(0, 0)
                    counts, maxima = [], []
                    # Never materialize a full fp32 Q/context copy (512 MiB BF16).
                    for start in range(0, rows, 128):
                        chunk = flat[start:start + 128].float()
                        finite = torch.isfinite(chunk)
                        counts.append((~finite).sum(dim=1, dtype=torch.int64))
                        maxima.append(torch.where(finite, chunk.abs(), 0).amax())
                    row_counts = torch.cat(counts).cpu().tolist() if counts else []
                    max_abs = float(torch.stack(maxima).amax().cpu()) if maxima else 0.0
                    bad_rows = [i for i, count in enumerate(row_counts) if count]
                    report = {"name": leaf_label, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                              "nonfinite_elements": sum(row_counts), "bad_row_count": len(bad_rows),
                              "bad_rows": bad_rows, "bad_elements_per_row": [row_counts[i] for i in bad_rows],
                              "max_abs_finite": max_abs}
                    # KV cache rows are cache slots, not query token positions.
                    if (bad_rows and not label.startswith("kv") and isinstance(positions, torch.Tensor)
                            and positions.numel() == rows):
                        pos = positions.detach().reshape(-1).cpu().tolist()
                        report["bad_query_positions"] = [int(pos[i]) for i in bad_rows]
                    if bad_rows and rows == len(embedding_context.get("input_ids", [])) and not label.startswith("kv"):
                        report["actual_input_token_ids"] = [int(embedding_context["input_ids"][i]) for i in bad_rows]
                    if bad_rows and stage in ("embedding.output", "vl_embedding.output", "decoder.input"):
                        try:
                            report["embedding_failure_evidence"] = embedding_failure_evidence(tensor, bad_rows)
                        except Exception as exc:
                            report["embedding_failure_evidence_error"] = f"{type(exc).__name__}: {exc}"
                    item["tensors"].append(report)
                    bad |= bool(bad_rows)
        item["all_finite"] = not bad
        embedding_guard(stage)
        # Always emit failures even after the healthy-event log budget expires.
        if counters["event"] < args.diagnostic_progress_limit or bad:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
        counters["event"] += 1
        if bad:
            raise RuntimeError(f"diagnostic nonfinite at layer 0 {stage}; see diagnostic.finite.rank{rank}.jsonl")

    def patch(owner, name, stage, before=None, after=None, inner_hook=None, predicate=None):
        restores.append(patch_finite_method_for_diagnostics(owner, name, stage=stage,
                        before=before, after=after, check=check, inner_hook=inner_hook, predicate=predicate))

    hidden = lambda b: {"hidden": b["hidden_states"]}
    try:
        patch(DeepseekVLForCausalLM, "get_input_embeddings", "vl_embedding", lambda b: {},
              lambda out: {"output": out}, predicate=lambda module: True)
        patch(DeepseekModel, "forward", "model", lambda b: {}, lambda out: {}, predicate=lambda module: True)
        patch(VocabParallelEmbedding, "forward", "embedding", lambda b: {"input_ids": b["input_"]},
              lambda out: {"output": out}, predicate=lambda module: True)
        patch(DeepseekDecoderLayer, "forward", "decoder", hidden, lambda out: {},
              predicate=(lambda module: True) if args.diagnostic_embedding_guard else None)
        for name in ("attn_norm_forward", "ffn_norm_forward"):
            patch(HCLayerMixin, name, name, hidden, lambda out: {"normalized_bf16": out[1]})
        patch(V4Attention, "forward", "attention", lambda b: {"hidden_bf16": b["hidden_states_bf16"]},
              lambda out: {"output": out})
        patch(V4Attention, "_prefill_attention", "prefill_attention",
              lambda b: {"q_b": b["q_b"], "kv": b["kv"]}, lambda out: {"context": out})
        patch(V4AttentionBase, "_wv_and_o_proj", "wv_o_projection",
              lambda b: {"context": b["context"]}, lambda out: {"output": out})
        patch(DeepseekMoE, "gate_forward", "moe_gate", hidden,
              inner_hook=lambda module, bound: module.gate.register_forward_hook(
                  lambda gate, inputs, output: check("moe_gate.logits", {"logits": output}, bound)))
        if args.diagnostic_embedding_guard:
            from deepinfer.model_executor.layers.sm90 import indexer as indexer_module
            original_mqa_logits = indexer_module.mqa_logits

            @functools.wraps(original_mqa_logits)
            def mqa_with_address_audit(*call_args, **call_kwargs):
                embedding_guard("mqa_logits.before")
                out = original_mqa_logits(*call_args, **call_kwargs)
                module = embedding_context.get("module")
                if module is not None:
                    weight_row = module.weight[guard_state["token_id"]]
                    item = {"stage": "indexer_output_address_audit", "layer": guard_state["active_layer"],
                            "output_ptr": out.data_ptr(), "output_shape": list(out.shape), "output_stride": list(out.stride()),
                            "output_element_bytes": out.element_size(), "sentinel_row_ptr": weight_row.data_ptr(),
                            "sentinel_row_bytes": weight_row.numel() * weight_row.element_size(),
                            "interpretation": "hypothetical historical signed-int32 stores; fixed kernel must not use these addresses"}
                    item["historical_i32_overlap"] = historical_i32_output_overlap(
                        out.data_ptr(), out.shape[0], out.shape[1], out.stride(0), out.element_size(),
                        weight_row.data_ptr(), weight_row.numel() * weight_row.element_size())
                    emit_guard(item, always=bool(item["historical_i32_overlap"]["overflow_output_rows"]))
                embedding_guard("mqa_logits.after")
                return out

            indexer_module.mqa_logits = mqa_with_address_audit
            def restore_mqa():
                if indexer_module.mqa_logits is not mqa_with_address_audit:
                    raise RuntimeError("mqa_logits changed after diagnostic patch")
                indexer_module.mqa_logits = original_mqa_logits
            restores.append(restore_mqa)
    except BaseException:
        for restore in reversed(restores):
            restore()
        raise

    def restore_all():
        if args.diagnostic_embedding_guard:
            emit_guard({"stage": "embedding_weight_guard_summary", **guard_state}, always=True)
        for restore in reversed(restores):
            restore()
    restore_all.embedding_guard = embedding_guard
    return restore_all


@dataclass
class Record:
    request_id: str
    sample: dict
    rank: int
    submit_s: float
    max_tokens: int
    token_ids: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    arrivals: list = field(default_factory=list)
    finish_s: float | None = None
    finish_reason: str = ""
    prompt_len: int | None = None
    error: str = ""

    def observe(self, now, output):
        if self.finish_s is not None:
            raise RuntimeError("tokens received after terminal notification")
        if self.prompt_len is not None and self.prompt_len != output["prompt_len"]:
            self.error = "prompt length changed during generation"
        self.prompt_len = output["prompt_len"]
        self.token_ids.append(output["token_id"])
        self.logprobs.append(output["logprob"])
        self.arrivals.append((now, len(self.token_ids)))
        if output["finished"]:
            self.finish_s, self.finish_reason = now, output["finish_reason"]

    def success(self, mode):
        return (not self.error and self.finish_s is not None and self.finish_reason in ("length", "stop")
                and self.prompt_len == self.sample["input_tokens"] and 0 < len(self.token_ids) <= self.max_tokens
                and all(math.isfinite(x) for x in self.logprobs)
                and (mode != "throughput" or (len(self.token_ids) == self.max_tokens and self.finish_reason == "length")))


class NativeDriver:
    """PR148 LLM._run_engine protocol plus copied scalar fields before next step."""
    def __init__(self, llm, args, diagnostic_stream=None, diagnostic_rank=None):
        import numpy as np
        import torch
        import torch.distributed as dist
        from deepinfer._C.engine import RequestOutput
        from deepinfer.model_executor.parallel_utils.parallel_state import PG
        from deepinfer.sampling_params import SamplingParams
        self.np, self.torch, self.dist, self.ro, self.params = np, torch, dist, RequestOutput, SamplingParams
        self.engine, self.args = llm.llm_engine, args
        self.group = PG.cpu_data_parallel_group()
        self.dp_size = self.group.size()
        self.last_arrival = None
        self.redknot_controller = None
        self.attention_profile_controller = None
        self.head_intervention_controller = None
        self.request_specs = {}
        self.diagnostic_stream = diagnostic_stream
        self.diagnostic_events = 0
        self.diagnostic_step = 0
        self.diagnostic_rank = diagnostic_rank

    def diagnostic(self, stage, **values):
        stream = getattr(self, "diagnostic_stream", None)
        if stream is None:
            return
        count = self.diagnostic_events
        limit = self.args.diagnostic_progress_limit
        if count > limit:
            return
        self.diagnostic_events += 1
        event = {"stage": "event_limit_reached" if count == limit else stage,
                 "step": self.diagnostic_step, "time_monotonic_s": time.perf_counter(),
                 "pid": os.getpid(), "rank": getattr(self, "diagnostic_rank", None),
                 **({} if count == limit else values)}
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()

    def submit(self, rid, ids):
        self.diagnostic("submit_enter", request_id=rid, input_tokens=len(ids))
        spec = self.request_specs.get(rid)
        if spec is not None:
            if spec["mode"] == "attention_profile":
                self.attention_profile_controller.register(rid, spec)
            elif spec["mode"] == "head_intervention":
                self.head_intervention_controller.register(rid, spec)
            else:
                self.redknot_controller.register(rid, spec)
        is_capture = spec is not None and spec["mode"] in ("capture", "profile_capture")
        params = self.params(temperature=0.0, top_p=1.0, top_k=-1,
                             max_tokens=1 if is_capture else self.args.max_new_tokens,
                             ignore_eos=is_capture or self.args.mode == "throughput", seq_seed=self.args.seed)
        mm = dict(image_bytes=[], image_sizes=[]) if self.engine.multi_modal_processor is not None else None
        seq = self.engine.add_request(user_id="0", request_id=rid, sampling_params=params,
                                      prompt_token_ids=self.np.asarray(ids, dtype=self.np.int32), multi_modal_data=mm)
        if seq is None:
            self.finish(rid)
            raise RuntimeError(f"native engine rejected request {rid}")
        self.diagnostic("submit_exit", request_id=rid)

    def finish(self, rid):
        if self.redknot_controller is not None:
            self.redknot_controller.runner.clear_redknot_request(rid)
        if getattr(self, "attention_profile_controller", None) is not None:
            self.attention_profile_controller.runner.clear_attention_range_request(rid)
        if getattr(self, "head_intervention_controller", None) is not None:
            self.head_intervention_controller.runner.clear_head_intervention_request(rid)

    def advance(self, unfinished):
        self.diagnostic_step += 1
        self.diagnostic("step_enter" if unfinished else "flush_enter", unfinished=bool(unfinished))
        guard = getattr(self, "embedding_guard", None)
        if guard is not None:
            guard("native_step.before")
        ptrs = self.engine.step() if unfinished else self.engine.flush_inflight_reqs()
        if guard is not None:
            guard("native_step.after")
        self.last_arrival = time.perf_counter()
        self.diagnostic("step_exit" if unfinished else "flush_exit", unfinished=bool(unfinished), outputs=int(len(ptrs)))
        ro = self.ro
        return [{"request_id": ro.get_request_id(p), "token_id": ro.get_token_id(p),
                 "finished": ro.get_finished(p), "finish_reason": ro.get_finish_reason(p),
                 "prompt_len": ro.get_prompt_len(p), "logprob": ro.get_token_logprobs(p)} for p in ptrs.tolist()]

    def has_unfinished(self):
        self.diagnostic("has_unfinished_enter")
        result = self.engine.has_unfinished_requests()
        self.diagnostic("has_unfinished_exit", unfinished=bool(result))
        return result

    def any_rank(self, flag):
        if self.dp_size == 1:
            return flag
        value = self.torch.tensor([int(flag)], dtype=self.torch.int64)
        self.dist.all_reduce(value, op=self.dist.ReduceOp.MAX, group=self.group)
        return bool(value.item())

    def barrier(self):
        if self.dp_size > 1:
            self.dist.barrier(group=self.group)

    def gather(self, value):
        if self.dp_size == 1:
            return [value]
        results = [None] * self.dp_size
        self.dist.all_gather_object(results, value, group=self.group)
        return results


def phase_repetitions(rows, phase, args):
    if phase == "measured":
        return args.repeat
    if phase == "warmup" and rows:
        # All active DP workers must see a real calibration prompt before
        # measurement. Offline RedKnot capture cannot substitute for an equal
        # explicit warmup policy on the native recomputed baseline.
        return max(1, math.ceil(min(args.concurrency, args.dp) / len(rows)))
    return 1


def benchmark_phases(args, manifest):
    selected = selected_heldout_row(args, manifest)
    if selected is not None:
        return (("measured", [selected]),)
    if args.arm in ("profile", "attention-profile") or getattr(args, "evaluate_calibration", False):
        return (("calibration", manifest["warmup"][:args.profile_limit]),)
    return (("warmup", [] if args.skip_warmup else manifest["warmup"]), ("measured", manifest["cohort"][:args.limit]))


def phase_jobs(rows, phase, args, rank):
    jobs = []
    for repetition in range(phase_repetitions(rows, phase, args)):
        for i, row in enumerate(rows):
            if (repetition * len(rows) + i) % min(args.concurrency, args.dp) == rank:
                jobs.append((f"{phase}-r{repetition}-{i:03d}", row))
    return jobs


def run_phase(driver, jobs, budget, rank, epoch, max_tokens, emit, clock=time.perf_counter):
    if budget < 0 or (jobs and budget == 0):
        raise ValueError("jobs require an active concurrency slot")
    pending, records, inflight = deque(jobs), {}, set()
    start = clock() - epoch

    def fill():
        count = 0
        while pending and len(inflight) < budget:
            rid, row = pending.popleft()
            if rid in records:
                raise ValueError("duplicate request id")
            rec = Record(rid, row, rank, clock() - epoch, max_tokens)
            records[rid] = rec
            driver.submit(rid, row["input_ids"])
            inflight.add(rid)
            emit({"event": "submit", "request_id": rid, "sample_id": row["sample_id"], "time_s": rec.submit_s})
            count += 1
        return count

    fill()
    while True:
        unfinished = driver.has_unfinished()
        outputs = driver.advance(unfinished)
        now = driver.last_arrival - epoch
        for output in outputs:
            rid = output["request_id"]
            if rid not in records:
                raise RuntimeError(f"unexpected native request output {rid}")
            records[rid].observe(now, output)
            emit({"event": "arrival", "time_s": now, **output})
            if output["finished"]:
                inflight.remove(rid)
                if hasattr(driver, "finish"):
                    driver.finish(rid)
        submitted = fill()
        if unfinished:
            continue
        if inflight and not outputs and not submitted:
            for rid in inflight:
                records[rid].error = "engine drained without terminal notification"
                records[rid].finish_s = now
            inflight.clear()
        if not driver.any_rank(bool(inflight or pending)):
            break
    parts = driver.gather({"records": list(records.values()), "start_s": start, "end_s": clock() - epoch})
    return [r for p in parts for r in p["records"]], {"start_s": min(p["start_s"] for p in parts), "end_s": max(p["end_s"] for p in parts)}


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", text.lower()))
    return " ".join(text.split())


def answer_scores(text, references):
    candidate = normalize(text).split()
    scores = []
    for ref in references:
        gold = normalize(ref).split()
        common = sum((Counter(candidate) & Counter(gold)).values())
        f1 = 2 * common / (len(candidate) + len(gold)) if candidate or gold else 1.0
        scores.append((float(normalize(text) == normalize(ref)), f1))
    return {"answer_em": max(x[0] for x in scores), "answer_token_f1": max(x[1] for x in scores)}


def canonical_record(rec, tokenizer, mode, phase):
    text = tokenizer.decode(rec.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    success = rec.success(mode)
    return {"request_id": rec.request_id, "sample_id": rec.sample["sample_id"], "dataset": rec.sample["dataset"],
            "rank": rec.rank, "phase": phase, "input_tokens": rec.sample["input_tokens"],
            "input_ids_sha256": rec.sample["input_ids_sha256"], "references": rec.sample["references"],
            "output_ids": rec.token_ids, "output_ids_sha256": digest(rec.token_ids), "text": text,
            "logprobs": [x if math.isfinite(x) else None for x in rec.logprobs],
            "submit_s": rec.submit_s, "finish_s": rec.finish_s,
            "ttft_s": rec.arrivals[0][0] - rec.submit_s if rec.arrivals else None,
            "elapsed_s": rec.finish_s - rec.submit_s if rec.finish_s is not None else None,
            "arrivals": rec.arrivals, "finish_reason": rec.finish_reason,
            "success": success, "error": rec.error or ("" if success else "native output validation failed"),
            **({"quality_role": "teacher_forced_diagnostic_not_independent_retrieval_test",
                "continuation_is_quality_evidence": False} if rec.sample.get("trace_kind") == "teacher_forced_answer" else {}),
            **(answer_scores(text, rec.sample["references"]) if success and mode == "quality"
               and rec.sample.get("trace_kind") != "teacher_forced_answer" else {})}


def summarize(records, window):
    good = [r for r in records if r["success"]]
    scored = [r for r in good if "answer_em" in r and "answer_token_f1" in r]
    wall = window["end_s"] - window["start_s"]
    ttft = sorted(r["ttft_s"] for r in good if r["ttft_s"] is not None)
    return {"measured": len(records), "ok": len(good), "wall_s": wall,
            "valid_complete_phase": bool(records) and len(good) == len(records),
            "completed_requests_per_s": len(good) / wall if wall > 0 else None,
            "qps_comparison_eligible": bool(records) and len(good) == len(records),
            "total_output_tok_s": sum(len(r["output_ids"]) for r in good) / wall if wall > 0 else None,
            "ttft_s_p50": statistics.median(ttft) if ttft else None,
            "ttft_s_p95": ttft[max(0, math.ceil(0.95 * len(ttft)) - 1)] if ttft else None,
            "quality_scored_requests": len(scored),
            "answer_em": statistics.fmean(r["answer_em"] for r in scored) if scored else None,
            "answer_token_f1": statistics.fmean(r["answer_token_f1"] for r in scored) if scored else None}


def evidence_eligibility(args, manifest, summary):
    """A smoke result is useful evidence of execution, never a throughput claim."""
    complete_cohort = args.limit in (None, len(manifest["cohort"]))
    exploratory_profile = args.arm == "redknot" and bool(getattr(args, "head_profile_data", {}).get("exploratory_only"))
    projected = getattr(args, "head_cache_representation", "expanded") == "projected"
    eligible = (not getattr(args, "diagnostic_prefill_breakdown", False) and getattr(args, "sample_id", None) is None and not projected and not exploratory_profile and not getattr(args, "evaluate_calibration", False) and args.arm not in ("profile", "attention-profile", "head-intervention") and not args.diagnostic_finite and not args.diagnostic_ep_sync and not args.diagnostic_progress and args.diagnostic_stacks_seconds == 0
                and manifest.get("artifact_kind") != "smoke" and manifest.get("benchmark_eligible", True)
                and not args.skip_warmup and bool(manifest["warmup"]) and complete_cohort
                and len(manifest["cohort"]) * args.repeat >= args.concurrency
                and summary["valid_complete_phase"])
    result = {"benchmark_eligible": bool(eligible),
            "qps_comparison_eligible": bool(eligible and args.mode == "throughput"),
            "exploratory_profile": exploratory_profile,
            "profile_qualification_exclusion": (
                "exploratory_only head profile: minimal chain probe, not quality or performance qualified"
                if exploratory_profile else None),
            "throughput_semantics": "controlled_fixed_output_length_ignore_eos" if args.mode == "throughput" else "natural_eos_quality_only"}
    if projected:
        result["cache_representation_qualification_exclusion"] = (
            "projected BF16 z-cache path is an opt-in quality smoke; separate numeric and model quality validation required")
    if getattr(args, "sample_id", None) is not None:
        result["sample_selection_qualification_exclusion"] = "single preselected heldout diagnostic, not full-cohort or controlled-throughput evidence"
    return result


def candidate_profile(report, geometry, args, identity, manifest):
    """Absolute calibration thresholds; no guessed local-head percentage."""
    if (report.get("boundary_tokens") != args.head_boundary_tokens
            or report.get("boundary_policy") != "recompute_document_prefix_all_placements_v1"
            or report.get("sampling_domain") != "document_interior"):
        raise ValueError("profiler report does not match requested document-boundary sampling policy")
    metrics = {(x["layer"], x["head"]): x for x in report["heads"]}
    layers = []
    for layer in range(geometry["layers"]):
        local = []
        if layer in geometry["enabled_layers"]:
            for head in range(geometry["heads"]):
                point = metrics.get((layer, head))
                if (point is not None and point["samples"] >= args.profile_min_samples
                        and point.get("nonfinite_samples", 0) == 0
                        and point["relative_l2"] is not None and point["mean_cosine"] is not None
                        and math.isfinite(point["relative_l2"]) and math.isfinite(point["mean_cosine"])
                        and point["relative_l2"] <= args.profile_relative_l2
                        and point["mean_cosine"] >= args.profile_min_cosine
                        and point.get("max_sample_relative_l2") is not None
                        and point.get("min_sample_cosine") is not None
                        and math.isfinite(point["max_sample_relative_l2"]) and math.isfinite(point["min_sample_cosine"])
                        and point["max_sample_relative_l2"] <= args.profile_relative_l2
                        and point["min_sample_cosine"] >= args.profile_min_cosine):
                    local.append(head)
        layers.append({"layer": layer, "num_heads": geometry["heads"], "local_heads": local})
    docs = {}
    for row in manifest["warmup"] + manifest["cohort"]:
        for doc in document_specs(row):
            docs[doc["token_ids_sha256"]] = len(doc["token_ids"])
    selected = sum(len(x["local_heads"]) for x in layers)
    return seal({"schema": "redknot-deepinfer-head-profile-v1", "layers": layers,
                 **{key: identity[key] for key in ("checkpoint_config_sha256", "checkpoint_index_sha256", "tokenizer_json_sha256")},
                 "calibration_manifest_sha256": manifest["sha256"],
                 "calibration_sample_ids": [r["sample_id"] for r in manifest["warmup"][:args.profile_limit]],
                 "profiling_report_sha256": digest(report), "runtime_geometry": geometry,
                 "head_boundary_tokens": args.head_boundary_tokens,
                 "selection": {"relative_l2_max": args.profile_relative_l2, "mean_cosine_min": args.profile_min_cosine,
                               "minimum_sample_comparisons": args.profile_min_samples,
                               "samples_per_document": args.profile_samples_per_document,
                               "boundary_sampling": "eligible document interior only; boundary rows fully recompute all heads online",
                               "worst_sample_guards": "same thresholds apply to max_sample_relative_l2 and min_sample_cosine; nonfinite samples disqualify",
                               "policy": "fixed absolute per-head calibration stability thresholds; no heldout selection or local ratio target"},
                 "selected_layer_heads": selected, "total_layer_heads": geometry["layers"] * geometry["heads"],
                 "estimated_one_256k_request_context_bytes": 262144 * selected * geometry["head_dim"] * 2,
                 "estimated_full_manifest_unique_context_bytes": sum(docs.values()) * selected * geometry["head_dim"] * 2,
                 "artifact_scope": "post-inverse-RoPE query-head outputs, not compact MLA latent KV",
                 "is_accuracy_certificate": False,
                 "claim_limit": "heuristic calibration candidate; requires independent heldout end-task quality and complete timing tests"})


def head_runtime_geometry(layers, heads, dim, yoco_consumer_start, cache_abi, boundary_tokens=128):
    """Query outputs are layer-owned, even when native KV is YOCO-shared."""
    if type(boundary_tokens) is not int or boundary_tokens < 0:
        raise ValueError("head boundary policy requires a nonnegative integer")
    return {"layers": layers, "heads": heads, "head_dim": dim,
            "enabled_layers": list(range(3, layers - 3)),
            "yoco_consumer_start": yoco_consumer_start,
            "protected_first_layers": 3, "protected_last_layers": 3,
            "cache_abi": cache_abi,
            "head_boundary_policy": {"tokens_per_document": boundary_tokens,
                                     "policy_id": "recompute_document_prefix_all_placements_v1",
                                     "boundary_heads": "fully_recomputed",
                                     "first_document_at_offset_zero": "same_boundary_protection",
                                     "question_and_gaps": "all_heads_recomputed",
                                     "scope": "query-head outputs only; not compact MLA checkpoint repair"},
            "layout_policy": {"context_parallel_size": 1, "eager": True,
                              "max_live_requests_per_worker": 1,
                              "query_layout": "full_untruncated",
                              "yoco_prefill_truncation_mode": "off",
                              "shared_hidden_kv_index_compressor": "native"}}


class HeadInterventionController:
    """Causal prefill quality probe; does not capture/reuse offline states."""
    def __init__(self, driver, args, identity, runtime_factory=None):
        if runtime_factory is None:
            from deepinfer.model_executor.diagnostics.head_intervention import HeadInterventionRuntime
            runtime_factory = HeadInterventionRuntime
        self.runner = driver.engine.driver_worker.model_runner
        cfg = self.runner.model.cfg
        geometry = {"layers": int(cfg.num_hidden_layers), "query_heads": int(cfg.num_attention_heads)}
        policy = args.intervention_policy_data
        if any(policy.get("model_geometry", {}).get(key) != value for key, value in geometry.items()):
            raise ValueError("head intervention policy geometry differs from the loaded model")
        self.runtime = runtime_factory.from_policy(
            policy, expected_identity=identity,
            query_chunk_rows=args.intervention_query_chunk_rows,
            max_staging_bytes=int(args.max_staging_mib * 1024**2))
        self.runner.attach_head_intervention_runtime(self.runtime)
        driver.head_intervention_controller = self

    def online_specs(self, jobs):
        return {rid: {"mode": "head_intervention", "row": row} for rid, row in jobs}

    def register(self, rid, spec):
        row = spec["row"]
        document_specs(row)
        self.runner.register_head_intervention_request(
            rid, token_ids=tuple(row["input_ids"]),
            documents=tuple((doc["input_start"], doc["input_end"]) for doc in row["documents"]))

    def report(self):
        return self.runtime.report()

    def close(self):
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()


class AttentionProfileController:
    """Read-only native attention observations; no offline capture or reuse plan."""
    def __init__(self, driver, args, runtime_factory=None):
        if runtime_factory is None:
            from deepinfer.model_executor.diagnostics.attention_observer import AttentionRangeRuntime
            runtime_factory = AttentionRangeRuntime
        self.args, self.runner = args, driver.engine.driver_worker.model_runner
        cfg = self.runner.model.cfg
        self.geometry = {"layers": int(cfg.num_hidden_layers), "query_heads": int(cfg.num_attention_heads)}
        self.runtime = runtime_factory(samples_per_document=args.attention_samples_per_document,
                                       boundary_tokens=args.attention_boundary_tokens,
                                       sink_tokens=args.attention_sink_tokens, recent_tokens=args.attention_recent_tokens,
                                       layers=self.geometry["layers"], query_heads=self.geometry["query_heads"])
        self.runner.attach_attention_range_runtime(self.runtime)
        self.request_bindings = {}
        driver.attention_profile_controller = self

    def online_specs(self, jobs):
        return {rid: {"mode": "attention_profile", "row": row} for rid, row in jobs}

    def register(self, rid, spec):
        row = spec["row"]
        self.runner.register_attention_range_request(rid, **attention_request_metadata(row))
        evidence = row.get("baseline_evidence", {})
        self.request_bindings[rid] = {
            "request_id": rid, "sample_id": row["sample_id"], "data_split": "calibration",
            "input_ids_sha256": row["input_ids_sha256"],
            "baseline_needle_pass": row.get("baseline_needle_pass", False),
            "baseline_trace_sha256": evidence.get("run_sha256"),
            "baseline_evidence": evidence, "trace_kind": row.get("trace_kind"),
        }

    def report(self):
        return {**self.runtime.report(), "range_request_bindings": list(self.request_bindings.values())}

    def close(self):
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()


def attention_range_artifact(report, geometry, args, identity, manifest):
    """Bind observational runtime output to calibration, never derive local heads."""
    from attention_profile_contract import PROTOCOL, SCHEMA
    if report.get("measurement_kind") != "native_attention_probability_mass":
        raise ValueError("attention runtime must identify actual native probability measurements")
    if "protocol" in report and report["protocol"] != PROTOCOL:
        raise ValueError("runtime attention mapping/probability protocol differs from the declared protocol")
    return seal({**report, "schema": SCHEMA, "manifest_sha256": manifest["sha256"],
                 "data_split": "calibration", "heldout_used": False,
                 "manifest_artifact_kind": manifest.get("artifact_kind", "unspecified"),
                 "calibration_contains_synthetic_probes": any(
                     row.get("synthetic_probe", False) for row in manifest["warmup"][:args.profile_limit]),
                 "calibration_sample_ids": [row["sample_id"] for row in manifest["warmup"][:args.profile_limit]],
                 "quality_qualified": False, "performance_qualified": False, "reusable_head_selection": None,
                 "protocol": PROTOCOL, "model_geometry": geometry,
                 "head_semantics": {"classification_axes": ["layer", "query_output_head"],
                                    "logical_heads_per_layer": geometry["query_heads"],
                                    "physical_mla_kv_heads": 1,
                                    "physical_kv_evidence": "profile_sample validates KV shape [capacity,1,query_dim]; no latent-axis head classification"},
                 "contract_sha256": file_digest(Path(__file__).with_name("attention_profile_contract.py")),
                 "evidence_stages": {"attention_range": "observational_only", "fixed_layer_head_mask": "not_run",
                                     "end_to_end_needle_intervention": "not_run", "heldout_rag": "not_run",
                                     **report.get("evidence_stages", {})},
                 "sink_mass_semantics": "learned attention-sink scalar, distinct from the first sink_tokens prompt positions",
                 "identity": {key: identity[key] for key in (
                     "checkpoint_config_sha256", "checkpoint_index_sha256", "tokenizer_json_sha256",
                     "runtime_source_tree_sha256", "adapter_sha256")},
                 "sampling_policy": {"samples_per_document": args.attention_samples_per_document,
                                     "boundary_tokens": args.attention_boundary_tokens,
                                     "sink_tokens": args.attention_sink_tokens, "recent_tokens": args.attention_recent_tokens},
                 "claim_limit": "observational attention range only; fixed-head interventions, end-to-end needles and heldout RAG are separate evidence"})


def write_attention_profile_shard(controller, args, identity, manifest, rank):
    """Persist local observations once; only this small index is all-gathered."""
    name = f"attention-range.rank{rank}.json.gz"
    payload = seal({"schema": "redknot-attention-range-rank-shard-v1", "rank": rank,
                    "manifest_sha256": manifest["sha256"], "model_geometry": controller.geometry,
                    "identity": {key: identity[key] for key in (
                        "checkpoint_config_sha256", "checkpoint_index_sha256", "tokenizer_json_sha256",
                        "runtime_source_tree_sha256", "adapter_sha256")},
                    "report": controller.report()})
    path = args.out_dir / name
    write_compressed_json(path, payload)
    return {"rank": rank, "file": name, "file_sha256": file_digest(path),
            "payload_sha256": payload["sha256"], "bytes": path.stat().st_size}


def gather_attention_profile_shards(driver, controller, args, identity, manifest, rank):
    """Every rank enters the same status collective even if its local I/O fails."""
    try:
        index = write_attention_profile_shard(controller, args, identity, manifest, rank)
        local = {"rank": rank, "ok": True, "index": index}
    except Exception as exc:
        local = {"rank": rank, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    statuses = driver.gather(local)
    if (len(statuses) != args.dp or {s.get("rank") for s in statuses} != set(range(args.dp))
            or any(s.get("ok") is not True for s in statuses)):
        failures = [{k: v for k, v in status.items() if k != "index"} for status in statuses if status.get("ok") is not True]
        raise RuntimeError(f"attention shard publication failed collectively: {failures}")
    return [status["index"] for status in sorted(statuses, key=lambda s: s["rank"])]


def publish_attention_profile_report(driver, indices, geometry, args, identity, manifest, rank):
    """Rank0 publishes; all ranks learn success/failure before a later barrier."""
    local = {"rank": rank, "ok": True, "role": "waiting"}
    if rank == 0:
        try:
            from attention_profile_contract import validate_attention_range_report
            merged = merge_attention_profile_shards(indices, geometry, args, identity, manifest)
            artifact = attention_range_artifact(merged, geometry, args, identity, manifest)
            write_compressed_json(args.out_dir / "attention-range.unvalidated.json.gz", artifact)
            validate_attention_range_report(artifact, manifest)
            write_compressed_json(args.out_dir / "attention-range.json.gz", artifact)
            local.update(role="publisher", artifact_sha256=artifact["sha256"])
        except Exception as exc:
            local.update(ok=False, role="publisher", error=f"{type(exc).__name__}: {exc}"[:2000])
    statuses = driver.gather(local)
    roots = [status for status in statuses if status.get("rank") == 0]
    if (len(statuses) != args.dp or {s.get("rank") for s in statuses} != set(range(args.dp))
            or len(roots) != 1 or roots[0].get("role") != "publisher"
            or any(s.get("ok") is not True for s in statuses)):
        raise RuntimeError(f"attention merged report publication failed collectively: {statuses}")
    return roots[0]["artifact_sha256"]


def _attention_expected_samples(row, args):
    """Mirror the frozen endpoint-inclusive CPU sampling contract, without CUDA."""
    metadata = attention_request_metadata(row)
    samples, cursor = {}, 0

    def add(lo, hi, label):
        count = min(args.attention_samples_per_document, hi - lo)
        if count > 0:
            for i in range(count):
                position = hi - 1 if count == 1 else lo + i * (hi - lo - 1) // (count - 1)
                samples[position] = label

    for ordinal, (lo, hi) in enumerate(metadata["documents"]):
        add(cursor, lo, "question_or_gap")
        boundary = min(hi, lo + args.attention_boundary_tokens)
        label = "first_document" if ordinal == 0 else "later_document"
        add(lo, boundary, label + "_boundary")
        add(boundary, hi, label + "_interior")
        cursor = hi
    add(cursor, len(metadata["token_ids"]), "question_or_gap")
    if metadata.get("answer_query_span") is not None:
        for position in range(*metadata["answer_query_span"]):
            samples[position] = "question_or_gap"
    return samples


def merge_attention_profile_shards(indices, geometry, args, identity, manifest):
    """CPU-only, fail-closed merge of rank-local evidence, never GPU arrays.

    All rank envelopes, ownership, input hashes, sampled positions and heads are
    checked before publishing a merged artifact. Empty ranks remain explicit.
    Weighted sufficient statistics preserve query counts, not rank averages.
    """
    import gzip
    from attention_profile_contract import MASS_FIELDS, STRATA, validate_attention_range_report

    def require(ok, message):
        if not ok:
            raise ValueError("attention shard: " + message)

    require(len(indices) == args.dp and {x.get("rank") for x in indices} == set(range(args.dp)),
            "missing or duplicate rank")
    expected_identity = {key: identity[key] for key in (
        "checkpoint_config_sha256", "checkpoint_index_sha256", "tokenizer_json_sha256",
        "runtime_source_tree_sha256", "adapter_sha256")}
    rows = manifest["warmup"][:args.profile_limit]
    merged, accumulators, seen_bindings, seen_observations = None, {}, set(), set()
    total_layer_rows, peak_scratch = 0, 0
    layer_metadata = {}
    all_bindings, all_observations = [], []
    for entry in sorted(indices, key=lambda x: x["rank"]):
        rank = entry["rank"]
        name = f"attention-range.rank{rank}.json.gz"
        require(entry.get("file") == name, "unexpected shard filename")
        path = args.out_dir / name
        require(not path.is_symlink() and path.is_file(), "missing or symlinked shard")
        require(path.stat().st_size == entry.get("bytes") and file_digest(path) == entry.get("file_sha256"),
                "file identity mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        require(payload.get("sha256") == entry.get("payload_sha256") == digest(
            {k: v for k, v in payload.items() if k != "sha256"}), "payload seal mismatch")
        require(payload.get("schema") == "redknot-attention-range-rank-shard-v1"
                and payload.get("rank") == rank and payload.get("manifest_sha256") == manifest["sha256"]
                and payload.get("model_geometry") == geometry and payload.get("identity") == expected_identity,
                "runtime/model/manifest/rank identity mismatch")
        report = payload["report"]
        jobs = dict(phase_jobs(rows, "calibration", args, rank))
        bindings = report.get("range_request_bindings", [])
        require(len(bindings) == len(jobs) and {b.get("request_id") for b in bindings} == set(jobs),
                "missing, duplicate or foreign request binding")
        for binding in bindings:
            rid = binding["request_id"]
            row = jobs[rid]
            require(rid not in seen_bindings and binding.get("sample_id") == row["sample_id"]
                    and binding.get("input_ids_sha256") == row["input_ids_sha256"]
                    and binding.get("data_split") == "calibration", "request/input ownership mismatch")
            seen_bindings.add(rid)
        observations = report.get("range_observations", [])
        expected_samples = {rid: _attention_expected_samples(row, args) for rid, row in jobs.items()}
        expected_keys = {(rid, layer, position) for rid, positions in expected_samples.items()
                         for layer in range(geometry["layers"]) for position in positions}
        local_seen = set()
        stratum_counts = Counter()
        windows = report.get("window_candidates", [])
        require(isinstance(windows, list) and windows and all(type(w) is int and w > 0 for w in windows)
                and windows == sorted(set(windows)), "invalid window candidates")
        for observation in observations:
            key = (observation.get("request_id"), observation.get("layer"), observation.get("query_position"))
            require(key in expected_keys and key not in local_seen and key not in seen_observations,
                    "duplicate, foreign or unscheduled layer/query observation")
            require(observation.get("query_stratum") == expected_samples[key[0]][key[2]]
                    and observation.get("current_context_tokens") == key[2] + 1,
                    "sampling stratum/position mismatch")
            valid = observation.get("window_coverage_valid")
            require(isinstance(valid, list) and len(valid) == geometry["query_heads"]
                    and all(type(v) is bool for v in valid), "window validity must cover every logical head")
            for field in ("conditional_unknown_key_mass", "virtual_sink_mass", "real_key_mass"):
                require(isinstance(observation.get(field), list)
                        and len(observation[field]) == geometry["query_heads"], "mass must cover every logical head")
            for field in ("recent_coverage_lower", "recent_coverage_upper", "sink_recent_coverage_lower", "sink_recent_coverage_upper"):
                matrix = observation.get(field)
                require(isinstance(matrix, list) and len(matrix) == geometry["query_heads"]
                        and all(isinstance(values, list) and len(values) == len(windows) for values in matrix),
                        "window matrix must cover every logical head/window")
            stratum_counts[key[1], observation["query_stratum"]] += 1
            local_seen.add(key)
        require(local_seen == expected_keys and report.get("sampled_layer_rows") == len(expected_keys),
                "incomplete sampled layer/query coverage")
        seen_observations.update(local_seen)
        if not jobs:
            require(not observations and not bindings and report.get("sampled_layer_rows") == 0,
                    "inactive rank contains observations")
            continue
        require(report.get("head_axis") == "query_output_head"
                and report.get("logical_query_heads") == geometry["query_heads"]
                and report.get("physical_kv_heads") == 1
                and type(report.get("qk_head_dim")) is int and report["qk_head_dim"] > 0,
                "actual logical/physical head geometry mismatch")
        require(report.get("query_stage") == "native_post_norm_post_rope"
                and report.get("kv_stage") == "native_assembled_post_rope_cache_values"
                and report.get("attention_support") == "native_selected_entries"
                and report.get("coverage_conditioning") == "real_keys_excluding_virtual_sink",
                "native logical-head observation stage mismatch")
        # The existing strict contract validates complete (layer,head) and
        # strata grids, probability mass, and bounds before any aggregation.
        validate_attention_range_report(attention_range_artifact(report, geometry, args, identity, manifest), manifest)
        metadata_rows = report.get("layer_metadata", [])
        require(len(metadata_rows) == geometry["layers"]
                and {item.get("layer") for item in metadata_rows} == set(range(geometry["layers"])),
                "incomplete or duplicate layer metadata")
        for metadata in metadata_rows:
            layer = metadata["layer"]
            for field in ("logical_query_heads", "physical_kv_heads", "qk_head_dim"):
                require(metadata.get(field) == report[field], "layer/logical head geometry mismatch")
            width_min, width_max = metadata.get("selected_index_width_min"), metadata.get("selected_index_width_max")
            require(type(width_min) is int and type(width_max) is int and 0 <= width_min <= width_max,
                    "invalid native selected-index width bounds")
            if layer not in layer_metadata:
                layer_metadata[layer] = dict(metadata)
            else:
                target = layer_metadata[layer]
                require(target.keys() == metadata.keys(), "inconsistent layer metadata fields")
                for field, value in metadata.items():
                    if "selected" in field and "width" in field and field.endswith(("_min", "_max")):
                        require(type(value) is int and value >= 0 and type(target[field]) is int,
                                "invalid native selected-index width")
                        target[field] = (min if field.endswith("_min") else max)(target[field], value)
                    else:
                        require(target[field] == value, f"inconsistent layer metadata {field}")
        if merged is None:
            merged = {key: value for key, value in report.items()
                      if key not in ("heads", "range_request_bindings", "range_observations")}
        else:
            variable_fields = {"heads", "range_request_bindings", "range_observations", "layer_metadata",
                               "sampled_layer_rows", "peak_diagnostic_scratch_estimate_bytes"}
            require(set(report) - variable_fields == set(merged) - variable_fields,
                    "inconsistent report metadata fields")
            for key in set(report) - variable_fields:
                require(report.get(key) == merged.get(key), f"inconsistent {key}")
        total_layer_rows += report["sampled_layer_rows"]
        peak_scratch = max(peak_scratch, report.get("peak_diagnostic_scratch_estimate_bytes", 0))
        for item in report["heads"]:
            for source in item["strata"]:
                require(source["valid_sampled_queries"] + source["excluded_queries"]
                        == stratum_counts[item["layer"], source["stratum"]],
                        "head/stratum counts disagree with sampled queries")
                key = (item["layer"], item["head"], source["stratum"])
                target = accumulators.setdefault(key, {"count": 0, "excluded": 0,
                    "mass": dict.fromkeys(MASS_FIELDS, 0.0), "needle_count": 0,
                    "needle_lower": 0.0, "needle_upper": 0.0, "counterfactuals": {}})
                count = source["valid_sampled_queries"]
                target["count"] += count
                target["excluded"] += source["excluded_queries"]
                if not count:
                    continue
                for field in MASS_FIELDS:
                    target["mass"][field] += source["mass_mean"][field] * count
                n = source.get("needle_sample_count", 0)
                require(type(n) is int and 0 <= n <= count, "invalid needle count")
                target["needle_count"] += n
                if n:
                    target["needle_lower"] += source["needle_mass_lower_mean"] * n
                    target["needle_upper"] += source["needle_mass_upper_mean"] * n
                for name, counter in source.get("fixed_layer_counterfactuals", {}).items():
                    combined = target["counterfactuals"].setdefault(name, {"samples": 0, "excluded": 0,
                        "relative_l2_sum": 0.0, "relative_l2_max": 0.0, "cosine_sum": 0.0})
                    n = counter["samples"]
                    require(type(n) is int and 0 <= n <= count and type(counter["excluded"]) is int
                            and counter["excluded"] >= 0, "invalid counterfactual count")
                    combined["samples"] += n
                    combined["excluded"] += counter["excluded"]
                    if n:
                        require(all(isinstance(counter[field], (int, float)) and math.isfinite(counter[field])
                                    for field in ("relative_l2_mean", "relative_l2_max", "cosine_mean")),
                                "nonfinite counterfactual")
                        combined["relative_l2_sum"] += n * counter["relative_l2_mean"]
                        combined["relative_l2_max"] = max(combined["relative_l2_max"], counter["relative_l2_max"])
                        combined["cosine_sum"] += n * counter["cosine_mean"]
        all_bindings.extend(bindings)
        all_observations.extend(observations)
    require(merged is not None and len(seen_bindings) == len(rows), "incomplete global calibration coverage")
    heads = []
    for layer in range(geometry["layers"]):
        for head in range(geometry["query_heads"]):
            strata = []
            for stratum in STRATA:
                point = accumulators[layer, head, stratum]
                count, needle_count = point["count"], point["needle_count"]
                item = {"stratum": stratum, "valid_sampled_queries": count,
                        "excluded_queries": point["excluded"], "mass_mean": None}
                if count:
                    mass = {key: value / count for key, value in point["mass"].items()}
                    item.update(mass_mean=mass, cross_context_mass_lower_mean=mass["other_document"]+mass["outside_documents"],
                                cross_context_mass_upper_mean=mass["other_document"]+mass["outside_documents"]+mass["mixed_or_unknown"],
                                needle_sample_count=needle_count,
                                needle_mass_lower_mean=point["needle_lower"]/needle_count if needle_count else None,
                                needle_mass_upper_mean=point["needle_upper"]/needle_count if needle_count else None,
                                fixed_layer_counterfactuals={})
                    for name, counter in point["counterfactuals"].items():
                        n = counter["samples"]
                        item["fixed_layer_counterfactuals"][name] = {
                            "samples": n, "excluded": counter["excluded"],
                            "relative_l2_mean": counter["relative_l2_sum"]/n if n else None,
                            "relative_l2_max": counter["relative_l2_max"],
                            "cosine_mean": counter["cosine_sum"]/n if n else None}
                else:
                    item["unavailable_reason"] = "no valid sampled queries in this stratum"
                strata.append(item)
            heads.append({"layer": layer, "head": head, "strata": strata})
    merged.update(heads=heads, range_request_bindings=sorted(all_bindings, key=lambda b: b["request_id"]),
                  range_observations=sorted(all_observations, key=lambda o: (o["request_id"], o["layer"], o["query_position"])),
                  sampled_layer_rows=total_layer_rows, peak_diagnostic_scratch_estimate_bytes=peak_scratch,
                  layer_metadata=[layer_metadata[k] for k in sorted(layer_metadata)],
                  rank_shards=sorted(indices, key=lambda x: x["rank"]),
                  parallel_calibration={"global_concurrency_limit": args.concurrency, "dp": args.dp,
                                        "max_live_requests_per_rank": 1, "unique_calibration_requests": len(rows),
                                        "assignment": "source-order round-robin; no duplicated rank copies"})
    return merged


class HeadReuseController:
    """One CP1 worker's explicit request registry and bounded offline storage."""
    def __init__(self, driver, args, identity, rows, rank):
        from deepinfer.redknot.plan import CacheStamp, DocumentArtifact, DocumentPlacement, HeadProfile, LayerHeads, build_reuse_plan
        from deepinfer.redknot.runtime import BoundedHeadStore, HeadSignalProfiler, RedKnotRuntime, HEAD_OUTPUT_CACHE_ABI
        import torch
        self.args, self.runner = args, driver.engine.driver_worker.model_runner
        self.DocumentPlacement, self.build_plan = DocumentPlacement, build_reuse_plan
        cfg = self.runner.model.cfg
        layers, heads, dim = int(cfg.num_hidden_layers), int(cfg.num_attention_heads), int(cfg.head_dim)
        self.representation = validate_head_cache_representation(args)
        self.calibration_geometry = head_runtime_geometry(layers, heads, dim, int(cfg.get("n_prefill_layer", layers)),
                                                          HEAD_OUTPUT_CACHE_ABI, args.head_boundary_tokens)
        self.geometry = self.calibration_geometry
        self.storage_geometry = None
        cache_abi = HEAD_OUTPUT_CACHE_ABI
        if self.representation == "projected":
            from deepinfer.redknot.runtime import BoundedProjectedStore, HEAD_PROJECTED_CACHE_ABI
            cache_abi = HEAD_PROJECTED_CACHE_ABI
            self.storage_geometry = projected_storage_geometry(cfg, self.calibration_geometry, cache_abi)
            self.geometry = {**self.calibration_geometry, "cache_abi": cache_abi,
                             "representation": "projected", "storage_geometry": self.storage_geometry}
        enabled = tuple(self.geometry["enabled_layers"])
        if args.arm == "profile":
            # Candidate set only: both offline and online execute ALL heads.
            profile = HeadProfile(tuple(LayerHeads(i, heads, tuple(range(heads)) if i in enabled else ()) for i in range(layers)))
        else:
            if args.head_profile_data.get("runtime_geometry") != self.calibration_geometry:
                raise ValueError("measured head profile differs from the loaded cache ABI/layout/layer policy; reprofile required")
            source = args.head_profile_data["layers"]
            if len(source) != layers or any(x["num_heads"] != heads for x in source):
                raise ValueError("measured head profile differs from effective loaded model geometry")
            if any(x["local_heads"] and x["layer"] not in enabled for x in source):
                raise ValueError("head profile requests protected layer reuse")
            profile = HeadProfile(tuple(LayerHeads(x["layer"], x["num_heads"], tuple(x["local_heads"])) for x in source))
        payload_identity = {"abi": cache_abi, "commit": PINNED_COMMIT,
                            "geometry": self.geometry, "dtype": "bfloat16"}
        if self.representation == "projected":
            payload_identity.update(calibration_profile_seal=args.head_profile_data["sha256"],
                                    calibration_profile_geometry=self.calibration_geometry)
        self.cache_abi_sha256 = digest(payload_identity)
        stamp = CacheStamp(digest({k: identity[k] for k in ("checkpoint_config_sha256", "checkpoint_index_sha256")}),
                           identity["tokenizer_json_sha256"], profile.sha256,
                           self.cache_abi_sha256)
        self.artifacts, self.row_documents = {}, {}
        self.profile_row_bindings = []
        for row in rows:
            if row["sample_id"] in self.row_documents:
                continue
            placements = []
            for doc in document_specs(row):
                key = doc["token_ids_sha256"]
                if key not in self.artifacts:
                    # Native KV/compression stays native. Head-output artifacts
                    # themselves have no compressed-token alignment requirement.
                    self.artifacts[key] = DocumentArtifact(stamp, doc["token_ids"], compression_ratios=(1,))
                placements.append(DocumentPlacement(self.artifacts[key], doc["input_start"]))
            self.row_documents[row["sample_id"]] = tuple(placements)
            self.profile_row_bindings.append({"sample_id": row["sample_id"],
                "input_ids_sha256": row["input_ids_sha256"]})
        parent = (args.cache_dir or args.out_dir / "offline-cache") / f"rank{rank}"
        store_kwargs = {"max_cache_bytes": int(args.max_cache_gib * 1024**3),
                        "max_staging_bytes": int(args.max_staging_mib * 1024**2)}
        if self.representation == "projected":
            self.store = BoundedProjectedStore(parent, **store_kwargs,
                num_groups=self.storage_geometry["num_output_groups"],
                projection_dim=self.storage_geometry["projection_dim"], cache_dtype=torch.bfloat16)
        else:
            self.store = BoundedHeadStore(parent, **store_kwargs)
        try:
            self.runtime = RedKnotRuntime(profile=profile, stamp=stamp, store=self.store, enabled_layers=enabled,
                profiler=HeadSignalProfiler(samples_per_document=args.profile_samples_per_document,
                                            max_bytes=int(args.profile_max_gib * 1024**3),
                                            boundary_tokens=args.head_boundary_tokens))
            if args.arm == "profile":
                samples = sum(min(args.profile_samples_per_document, max(0, len(a.token_ids) - args.head_boundary_tokens))
                              for a in self.artifacts.values())
                estimated = samples * len(enabled) * heads * dim * 2
                if estimated > int(args.profile_max_gib * 1024**3):
                    raise MemoryError(f"profiling samples need {estimated} bytes, above host sample budget")
                self.preflight = {"profile_sample_bytes": estimated, "documents": len(self.artifacts),
                                  "head_boundary_tokens": args.head_boundary_tokens,
                                  "scope": "full-head diagnostics, not sparse execution"}
            else:
                self.preflight = self.runtime.preflight(tuple(self.artifacts.values()), head_dim=dim, dtype=torch.bfloat16)
            self.runner.attach_redknot_runtime(self.runtime)
        except BaseException:
            self.store.close(remove_files=True)
            raise
        driver.redknot_controller = self

    def offline_jobs(self, rank):
        jobs, specs = [], {}
        mode = "profile_capture" if self.args.arm == "profile" else "capture"
        for i, (key, artifact) in enumerate(self.artifacts.items()):
            rid = f"offline-rank{rank}-d{i:03d}"
            row = {"sample_id": f"document:{key}", "input_ids": artifact.token_ids, "input_tokens": len(artifact.token_ids)}
            jobs.append((rid, row))
            specs[rid] = {"mode": mode, "artifact": artifact}
        return jobs, specs

    def online_specs(self, jobs):
        mode = "profile_compare" if self.args.arm == "profile" else "reuse"
        return {rid: {"mode": mode, "row": row} for rid, row in jobs}

    def register(self, rid, spec):
        if "artifact" in spec:
            self.runner.register_redknot_request(rid, mode=spec["mode"], artifact=spec["artifact"])
        else:
            row = spec["row"]
            # This request validation/plan construction occurs in submit(),
            # inside measured TTFT/QPS rather than being hidden in offline time.
            tokens = tuple(row["input_ids"])
            plan = self.build_plan(request_token_ids=tokens, placements=self.row_documents[row["sample_id"]],
                                   expected_stamp=self.runtime.stamp, profile=self.runtime.profile,
                                   boundary_tokens=self.args.head_boundary_tokens)
            self.runner.register_redknot_request(rid, mode=spec["mode"], plan=plan, request_token_ids=tokens)

    def report(self):
        result = {"geometry": self.geometry, "preflight": self.preflight,
                "profile_row_bindings": self.profile_row_bindings,
                "offline_cache_bytes": self.store.disk_bytes, "peak_transfer_bytes": self.store.peak_transfer_bytes,
                "work": self.runtime.work, "native_fallback_layers": sorted(self.runtime.native_fallback_layers),
                "profiling": self.runtime.profiler.report() if self.args.arm == "profile" else None,
                "cache_scope": "query-head outputs only; shared MLA KV/index/compressor and MoE recomputed",
                "online_transfer_included": True, "is_total_flops_measurement": False}
        if self.representation == "projected":
            result.update(cache_representation="projected", storage_geometry=self.storage_geometry,
                          calibration_profile_geometry=self.calibration_geometry,
                          calibration_profile_seal=self.args.head_profile_data["sha256"],
                          cache_abi_sha256=self.cache_abi_sha256,
                          projection_policy=PROJECTED_HEAD_POLICY,
                          cache_scope="projected local-head z contributions only; native MLA KV/index/compressor and MoE recomputed",
                          quality_qualified=False, performance_qualified=False)
        return result

    def close(self):
        # Only files owned by this newly-created runtime; all requests must
        # already have drained or the model must have closed before this call.
        self.store.close(remove_files=True)


def worker(args, manifest, identity, dp_size, rank):
    sys.path.insert(0, str(args.repo / "examples"))
    sys.path.insert(0, str(args.repo))
    os.environ.update(effective_env(args))
    llm = None
    controller = None
    stack_stream = None
    diagnostic_stream = None
    ep_diagnostic_stream = None
    restore_ep_dispatch = None
    finite_diagnostic_stream = None
    restore_finite = None
    prefill_diagnostics = None
    try:
        if args.diagnostic_progress:
            diagnostic_stream = (args.out_dir / f"diagnostic.progress.rank{rank}.jsonl").open("x", encoding="utf-8", buffering=1)
            diagnostic_stream.write(json.dumps({"stage": "worker_start", "rank": rank, "pid": os.getpid(),
                                               "time_monotonic_s": time.perf_counter()}) + "\n")
        if args.diagnostic_stacks_seconds:
            import faulthandler
            stack_stream = (args.out_dir / f"diagnostic.stacks.rank{rank}.log").open("x", encoding="utf-8", buffering=1)
            stack_stream.write(f"rank={rank} pid={os.getpid()} periodic_seconds={args.diagnostic_stacks_seconds}\n")
            stack_stream.flush()
            faulthandler.dump_traceback_later(args.diagnostic_stacks_seconds, repeat=True, file=stack_stream, exit=False)
        from _config import create_llm, setup_common_env
        setup_common_env(model_name=str(args.model_path), sampler_force_topk=args.sampler_force_topk,
                         set_override_experts=False, enable_eplb=False)
        if args.diagnostic_ep_sync or args.diagnostic_finite:
            # create_llm normally performs this before its first torch import.
            # Do it identically here so the diagnostic hook is installed for
            # initialization/profile-run dispatches without binding all ranks
            # to the first GPU. create_llm repeats this idempotent setup.
            from _config import get_actual_num_gpus, setup_cuda_visible_devices
            setup_cuda_visible_devices(rank, args.cp, get_actual_num_gpus())
            os.environ["DEEPINFER_LOGGING_INTERVAL_STEP"] = "0"
        if args.diagnostic_ep_sync:
            ep_diagnostic_stream = (args.out_dir / f"diagnostic.ep.rank{rank}.jsonl").open("x", encoding="utf-8", buffering=1)
            restore_ep_dispatch = install_ep_sync_diagnostic(ep_diagnostic_stream, args, rank)
        if args.diagnostic_finite:
            finite_diagnostic_stream = (args.out_dir / f"diagnostic.finite.rank{rank}.jsonl").open("x", encoding="utf-8", buffering=1)
            restore_finite = install_finite_diagnostic(finite_diagnostic_stream, args, rank)
        init_start = time.perf_counter()
        llm = create_llm(str(args.model_path), rank, ep_size=args.ep, cp_size=args.cp,
                         env={"DEEPINFER_LOGGING_INTERVAL_STEP": "0"}, data_parallel_size=dp_size,
                         max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
                         max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
                         lookahead_tokens=0, decode_sequence=False)
        init_s = time.perf_counter() - init_start
        driver = NativeDriver(llm, args, diagnostic_stream, rank)
        if getattr(args, "diagnostic_prefill_breakdown", False):
            from pr148_prefill_diagnostics import PrefillDiagnostics
            prefill_diagnostics = PrefillDiagnostics(enabled=True, rank=rank, ranks=None,
                max_records=100000, max_cuda_pairs=100000)
            prefill_diagnostics.install_deepinfer()
        driver.diagnostic("model_ready", rank=rank, engine_init_s=init_s)
        if args.diagnostic_embedding_guard:
            driver.embedding_guard = restore_finite.embedding_guard
            model = driver.engine.driver_worker.model_runner.model
            driver.embedding_guard("model_ready", module=model.model.embed_tokens)
        if driver.dp_size != dp_size:
            raise RuntimeError("native DP group differs from configured DP")
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]
        phases = benchmark_phases(args, manifest)
        if args.arm in ("profile", "redknot"):
            owned_rows = [row for phase, rows in phases for _, row in phase_jobs(rows, phase, args, rank)]
            controller = HeadReuseController(driver, args, identity, owned_rows, rank)
            offline_jobs, driver.request_specs = controller.offline_jobs(rank)
            write_json(args.out_dir / f"cache-preflight.rank{rank}.json", seal(controller.report()))
            driver.barrier()
            offline_events = []
            if prefill_diagnostics is not None:
                prefill_diagnostics.begin_phase("offline")
            try:
                offline_records, offline_window = run_phase(driver, offline_jobs, int(bool(offline_jobs)),
                    rank, epoch, 1, offline_events.append)
            finally:
                if prefill_diagnostics is not None:
                    write_json(args.out_dir / f"prefill-breakdown.offline.rank{rank}.json",
                               seal(prefill_diagnostics.end_phase()))
                for event in offline_events:
                    if "logprob" in event and not math.isfinite(event["logprob"]):
                        event["logprob"] = None
                write_json(args.out_dir / f"offline.events.rank{rank}.json", seal({"events": offline_events}))
            expected_offline = sum(driver.gather(len(offline_jobs)))
            if len(offline_records) != expected_offline or any(not r.success("quality") for r in offline_records):
                raise RuntimeError("offline document capture failed; refusing online head reuse")
            driver.request_specs = {}
            captures = driver.gather(controller.report())
            if rank == 0:
                write_json(args.out_dir / "offline.json", seal({
                    "window": offline_window, "wall_s": offline_window["end_s"] - offline_window["start_s"],
                    "documents": len(offline_records), "ranks": captures,
                    "scope": "independent position-zero document generation including 1 sampled token each; excluded from hot online TTFT",
                    "is_online_performance": False,
                    "records": [{"request_id": r.request_id, "sample_id": r.sample["sample_id"],
                                 "input_tokens": r.sample["input_tokens"], "success": r.success("quality"),
                                 "output_ids": r.token_ids, "elapsed_s": r.finish_s - r.submit_s}
                                for r in offline_records]}))
            driver.barrier()
        elif args.arm == "attention-profile":
            controller = AttentionProfileController(driver, args)
        elif args.arm == "head-intervention":
            controller = HeadInterventionController(driver, args, identity)
        for phase, rows in phases:
            events = []
            jobs = phase_jobs(rows, phase, args, rank)
            if controller is not None:
                driver.request_specs = controller.online_specs(jobs)
            driver.barrier()
            if prefill_diagnostics is not None:
                prefill_diagnostics.begin_phase(phase)
            try:
                records, window = run_phase(driver, jobs,
                                            rank_budget(args.concurrency, dp_size, rank), rank, epoch,
                                            args.max_new_tokens, events.append)
            finally:
                if prefill_diagnostics is not None:
                    write_json(args.out_dir / f"prefill-breakdown.{phase}.rank{rank}.json",
                               seal(prefill_diagnostics.end_phase()))
                # No filesystem logging in the timing window. Preserve raw observations on failure.
                with (args.out_dir / f"{phase}.events.rank{rank}.jsonl").open("x", encoding="utf-8") as stream:
                    for event in events:
                        if "logprob" in event and not math.isfinite(event["logprob"]):
                            event["logprob"] = None
                        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            expected = len(rows) * phase_repetitions(rows, phase, args)
            failure = len(records) != expected or any(not r.success(args.mode) for r in records)
            attention_shards = None
            if args.arm == "attention-profile" and not failure:
                attention_shards = gather_attention_profile_shards(driver, controller, args, identity, manifest, rank)
            if args.arm == "attention-profile":
                # Keep this diagnostics-only gather compact as well. Heavy
                # arrays stay in per-rank files and are read on rank0 only.
                runtime_reports = driver.gather({"rank": rank,
                    "sampled_layer_rows": controller.runtime.sampled_layer_rows,
                    "registered_requests": len(controller.request_bindings)})
            else:
                runtime_reports = driver.gather(controller.report()) if controller is not None else None
            if args.arm == "profile" and not failure:
                from parallel_head_signals import merge_reports
                # Small sufficient statistics only; raw sampled contexts never
                # traverse the collective or get counted as independent cases.
                merged_signals = merge_reports(runtime_reports, controller.geometry, args, manifest)
                publish_status = {"rank": rank, "ok": True}
                if rank == 0:
                    try:
                        write_json(args.out_dir / "head-signals.json", seal(merged_signals))
                        candidate = candidate_profile(merged_signals, controller.geometry, args, identity, manifest)
                        write_json(args.out_dir / "candidate-head-profile.json", candidate)
                        print(json.dumps({"head_profile_candidate": "candidate-head-profile.json",
                            "selected_layer_heads": candidate["selected_layer_heads"],
                            "is_accuracy_certificate": False}), flush=True)
                    except Exception as exc:
                        publish_status.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                statuses = driver.gather(publish_status)
                if any(not item["ok"] for item in statuses):
                    raise RuntimeError(f"head signal publication failed collectively: {statuses}")
            if args.arm == "attention-profile" and not failure:
                publish_attention_profile_report(driver, attention_shards, controller.geometry,
                                                 args, identity, manifest, rank)
            if rank == 0:
                import torch
                canonical = [canonical_record(r, llm.llm_engine.tokenizer, args.mode, phase) for r in records]
                canonical.sort(key=lambda x: x["request_id"])
                write_json(args.out_dir / f"{phase}.json", seal({"records": canonical, "window": window}))
                if phase in ("measured", "calibration"):
                    summary = summarize(canonical, window)
                    eligibility = evidence_eligibility(args, manifest, summary)
                    summary["qps_comparison_eligible"] = eligibility["qps_comparison_eligible"]
                    result = {"schema": "redknot-deepinfer-native-run-v1", "status": "failed" if failure else "complete",
                              "engine": "deepinfer-pr148", "arm": args.arm, "mode": args.mode,
                              "concurrency": args.concurrency, "manifest_sha256": manifest["sha256"],
                              "measurement_boundary": BOUNDARY, "records": canonical, "window": window,
                              "summary": summary, **eligibility,
                              "head_reuse_runtime": runtime_reports if args.arm in ("profile", "redknot") else None,
                              "attention_range_runtime": runtime_reports if args.arm == "attention-profile" else None,
                              "head_intervention_runtime": runtime_reports if args.arm == "head-intervention" else None,
                              "meta": {**identity, "engine_init_s": init_s,
                              "warmup_requests": 0 if args.skip_warmup or args.evaluate_calibration or args.arm in ("profile", "attention-profile") else
                                  len(manifest["warmup"]) * phase_repetitions(manifest["warmup"], "warmup", args),
                              "warmup_unique_samples": 0 if args.skip_warmup or args.evaluate_calibration or args.arm in ("profile", "attention-profile") else len(manifest["warmup"]),
                              "warmup_rank_request_counts": [0 if args.skip_warmup or args.evaluate_calibration or args.arm in ("profile", "attention-profile") else
                                  len(phase_jobs(manifest["warmup"], "warmup", args, owner)) for owner in range(dp_size)],
                              "warmup_policy": "source-order calibration repeats until every active DP rank receives a real prompt; heldout never used",
                              "max_model_len": args.max_model_len, "max_num_batched_tokens": args.max_num_batched_tokens,
                              "max_num_seqs": args.max_num_seqs, "gpu_memory_utilization": args.gpu_memory_utilization,
                              "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                              "torch": torch.__version__, "cuda": torch.version.cuda, "scope": "pilot; not quality certification",
                              "timing_semantics": "host enqueue to first native step/flush token; native scheduler included; no HTTP; initialization and tokenization excluded",
                              "quality_semantics": (
                                  "teacher-forced answer attention diagnostic; continuation is NOT independent retrieval accuracy"
                                  if manifest.get("artifact_kind") == "synthetic_teacher_forced_answer_calibration" else
                                  "normalized answer EM/token F1 against frozen references; synthetic probes are disclosed; not official LongBench scoring")}}
                    write_json(args.out_dir / "run.json", seal(result))
                    print(json.dumps({"status": result["status"], "summary": result["summary"]}), flush=True)
            if failure:
                raise RuntimeError(f"{phase} request validation failed; refusing further dispatch")
            driver.barrier()
    except Exception as exc:
        failure_path = args.out_dir / f"failure.rank{rank}.json"
        if not failure_path.exists():
            write_json(failure_path, seal({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "rank": rank}))
        raise
    finally:
        try:
            if llm is not None:
                llm.close()
                if controller is not None:
                    controller.close()
        finally:
            try:
                if prefill_diagnostics is not None:
                    prefill_diagnostics.close()
                if restore_finite is not None:
                    restore_finite()
                if restore_ep_dispatch is not None:
                    restore_ep_dispatch()
            finally:
                if ep_diagnostic_stream is not None:
                    ep_diagnostic_stream.close()
                if finite_diagnostic_stream is not None:
                    finite_diagnostic_stream.close()
                if stack_stream is not None:
                    import faulthandler
                    faulthandler.cancel_dump_traceback_later()
                    stack_stream.close()
                if diagnostic_stream is not None:
                    diagnostic_stream.close()


def main(argv=None):
    args = parse_args(argv)
    manifest, identity = preflight(args)
    imported = any(x in sys.modules for x in ("torch", "deepinfer"))
    if imported:
        raise RuntimeError("preflight imported GPU runtime")
    report = {"status": "cpu-preflight-passed", "manifest_sha256": manifest["sha256"],
              "identity": identity, "gpu_initialized": False, "imported_torch_or_deepinfer": imported}
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "preflight.json", seal(report))
    wait_for_release(args, manifest["sha256"])
    os.environ.update(effective_env(args))
    sys.path.insert(0, str(args.repo / "examples"))
    from _launch import launch_dp
    launch_dp(functools.partial(worker, args, manifest, identity), args.dp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
