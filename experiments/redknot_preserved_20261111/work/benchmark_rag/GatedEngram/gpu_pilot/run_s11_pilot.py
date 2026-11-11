"""Independent Full / S1.1 comparison: remaining 128K and three natural 256K cases.

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
from gpu_pilot.s11_protocol import (phase_policy, phase_execution, controller_arm, EVALUATION_INDICES, QUALIFICATION_CASE_INDEX, SYSTEM_BATCH_GROUP_CHUNKS)

SCHEMA = "kv-cache-engram-s11-pilot-run-v1"
PHASE_SCHEMA = "kv-cache-engram-s11-phase-v1"
ARMS = ("recomputed", "s1_system")
POLICY_ARMS = ARMS[1:]
FRACTIONS = dict.fromkeys(ARMS)
CAPTURE_LAYERS = tuple(range(3,40))
FFN_LAYERS = tuple(range(3,40))
NATIVE_SNAPSHOT = STAGE.parent / "NativeSnapshot"
INPUT_COMPANIONS = ("source-records.json", "selection-plan.json", "tokenization-evidence.json")


def validate_inputs(base, manifest, protocol, *, input_dir):
    from gpu_pilot.s11_inputs import validate_inputs as validate
    return validate(base, manifest, protocol, input_dir=input_dir)


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
    from gpu_pilot.s11_protocol import DEPENDENCIES, REQUIRED_NEW_DEPENDENCIES
    require(DEPENDENCIES and all(k in DEPENDENCIES for k in REQUIRED_NEW_DEPENDENCIES),
            "new S1 dependency pins must be frozen before preflight")
    hashes = support.source_hashes(root)
    for name, expected in DEPENDENCIES.items():
        path = (STAGE/name.removeprefix("GatedEngram/")) if name.startswith("GatedEngram/") else Path(root)/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, "frozen sparse operator dependency changed: "+name)
        hashes[name] = actual
    return hashes


def parse_args(argv=None):
    extra = argparse.ArgumentParser(description=__doc__)
    extra.add_argument("--s1-protocol", required=True, type=Path)
    known, rest = extra.parse_known_args(argv)
    args = support.parse_args(rest)
    args.s1_protocol = known.s1_protocol
    return args


def validate_protocol(args, manifest):
    from gpu_pilot.s11_protocol import s1_protocol
    value = json.loads(args.s1_protocol.read_bytes())
    require(value == s1_protocol(), "Loong multi-document RAG comparison protocol differs")
    require(type(value.get("input_manifest_sha256")) is str
            and manifest.get("sha256") == value["input_manifest_sha256"],
            "comparison must bind the frozen assembled Loong multi-document RAG manifest")
    require(getattr(args, "sensitivity_protocol", None) is None
            and "sensitivity_protocol_sha256" not in manifest,
            "old sensitivity20% protocol cannot govern this comparison")
    return value


def case_scope(row):
    return {key:row.get(key) for key in ("length_band","length_band_ordinal","target_input_tokens",
        "source_document_count","original_document_count","source_domain","source_language",
        "source_level","raw_answer_sha256")}


def selected_rows(manifest):
    from gpu_pilot.s11_protocol import DATASETS, INPUT_TOKEN_RANGES, TARGET_INPUT_TOKENS
    cohort=manifest.get('cohort')
    require(type(cohort) is list and len(cohort)==len(EVALUATION_INDICES)
            and {r.get('index') for r in cohort}==set(EVALUATION_INDICES),
            'exact remaining128 plus three256 cohort required')
    by_index={r['index']:r for r in cohort}
    rows=[by_index[index] for index in EVALUATION_INDICES]
    for row in rows:
        band='128k' if row['index']==20 else '256k'
        n=row.get('input_tokens');lo,hi=INPUT_TOKEN_RANGES[band]
        require(row.get('dataset') in DATASETS and row.get('length_band')==band
            and row.get('target_input_tokens')==TARGET_INPUT_TOKENS[band]
            and type(n) is int and lo<=n<=hi and n<=270336-4096,
            'actual token length/band/context headroom differs')
        require(type(row.get('length_band_ordinal')) is int
            and (row['length_band_ordinal']==10 if band=='128k' else row['length_band_ordinal']==row['index']-1),
            'original source band ordinal changed')
        require(len(row.get('input_ids',[]))==n and _digest(row['input_ids'])==row.get('input_ids_sha256'),
            'full input IDs differ')
        document_lengths(row)
        selected=set(range(n))
        for doc in row['documents']:
            selected.difference_update(range(doc['input_start'],doc['input_end']-128))
        counts=[sum(start<=p<min(start+8192*SYSTEM_BATCH_GROUP_CHUNKS,n) for p in selected)
                for start in range(0,n,8192*SYSTEM_BATCH_GROUP_CHUNKS)]
        require(all(0<count<=8192 for count in counts),
            'group4 needs nonempty selected queries within native8192 EP buffer capacity')
    require(by_index[QUALIFICATION_CASE_INDEX]['input_tokens']==max(r['input_tokens'] for r in rows),
            'case2 must remain the longest full-input numerical qualification anchor')
    return rows


def qualification_anchor(manifest):
    return next(row for row in selected_rows(manifest) if row['index']==QUALIFICATION_CASE_INDEX)


def timing_schedule(index):
    from gpu_pilot.s11_protocol import timing_schedule as declared
    return declared(index)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def metadata_qualification_report(reports, identity):
    """Bind actual per-rank operator evidence to this frozen execution."""
    from gpu_pilot.read_s11_results import validate_metadata_qualification
    paths = ("GatedEngram/kv_cache_engram/s1_system_batch.py",
             "GatedEngram/kv_cache_engram/s1_system_runtime.py")
    body = dict(schema="s1-system-cuda-metadata-all-ranks-v1", passed=True,
        execution_configuration_sha256=identity["execution_configuration_sha256"],
        model_sha256=identity["checkpoint_index_sha256"],
        operator_source_sha256={p:identity["pilot_source_sha256"][p] for p in paths},
        reports=reports, quality_eligible=False, timing_valid=False,
        gpu_numerical_parity_claim=False)
    result = dict(body, sha256=_digest(body))
    validate_metadata_qualification(result, identity)
    return result


def qualification_row(manifest):
    original = manifest["warmup"][0]
    require(original["index"] == 0 and len(original["input_ids"]) >= 16384,
            "qualification requires the declared calibration case0 prefix")
    ids = original["input_ids"][:16384]
    token_hash = _digest(ids)
    return dict(index=0, dataset=original["dataset"], input_ids=ids, input_tokens=16384,
                input_ids_sha256=token_hash, sample_id="qualification-"+token_hash,
                references=["__operator_qualification_not_quality__"], documents=None)


def empty_qualification_row(manifest):
    row=qualification_row(manifest);key=row['input_ids_sha256']
    row.update(sample_id='s1-empty-'+key,references=['__same_policy_qualification_not_quality__'],
        s1_empty_chunk_qualification=True,documents=[dict(document_id=key,token_ids_sha256=key,
            input_start=0,input_end=16384,tokens=16384,offline_position=0,online_position=0)])
    return row


def empty_group_qualification_row(manifest):
    source=qualification_anchor(manifest)
    ids=list(source['input_ids'][:32769])
    require(len(ids)==32769,'empty group diagnostic requires actual case2 prefix')
    key=_digest(ids)
    return dict(index=0,dataset=source['dataset'],input_ids=ids,input_tokens=len(ids),input_ids_sha256=key,
        sample_id='s1-empty-group-'+key,references=['__diagnostic_not_rag_quality__'],
        system_empty_group_qualification=True,source_original_case_index=source['index'],
        source_full_input_ids_sha256=source['input_ids_sha256'],
        documents=[dict(document_id=key,token_ids_sha256=key,input_start=0,input_end=len(ids),
            tokens=len(ids),offline_position=0,online_position=0)])


def quality_token_budget(input_tokens, engine_max_model_len=270336):
    """Native SamplingParams requires an integer; only engine context bounds quality."""
    require(type(input_tokens) is int and type(engine_max_model_len) is int
            and engine_max_model_len == 270336 and 0 < input_tokens < engine_max_model_len,
            "quality request must fit the frozen engine context capacity")
    return engine_max_model_len-input_tokens


def document_lengths(row):
    """Read actual token spans; separators remain mandatory nondocument rows."""
    docs, ids = row.get("documents"), row.get("input_ids")
    require(type(docs) is list and len(docs) == 10 and type(ids) is list
            and row.get("input_tokens") == len(ids), "ten declared cache-block spans required")
    previous, lengths = 0, []
    for doc in docs:
        start, end = doc.get("input_start"), doc.get("input_end")
        require(type(start) is int and type(end) is int and previous <= start < end <= len(ids),
                "document spans must be ordered, disjoint and inside the input")
        require(doc.get("tokens") == end-start and doc.get("offline_position") == 0
                and doc.get("online_position") == start
                and end-start >= 256
                and doc.get("document_id") == doc.get("token_ids_sha256") == _digest(ids[start:end]),
                "document geometry or token identity changed")
        lengths.append(end-start)
        previous = end
    return lengths


def case_geometry(row):
    lengths = document_lengths(row)
    tokens = row["input_tokens"]
    return dict(index=row["index"], input_tokens=tokens,
        document_token_lengths=lengths, document_tokens=sum(lengths),
        nondocument_tokens=tokens-sum(lengths), native_prefill_chunks=(tokens+8191)//8192,
        layer_chunk_checks=40*((tokens+8191)//8192), offline_bank_entry_count=400,
        offline_swa_halo_rows=37*10*128,offline_global_raw_rows=3*sum(lengths),
        offline_bank_rows=37*10*128+3*sum(lengths),
        offline_bank_bytes=37*10*128*512*2+9216*sum(lengths))


def manifest_geometry(manifest, protocol):
    cases = [case_geometry(row) for row in selected_rows(manifest)]
    anchor = case_geometry(qualification_anchor(manifest))
    payload = sum(row["offline_bank_bytes"] for row in cases)+anchor["offline_bank_bytes"]
    policy = protocol["free_disk_policy"]
    numerator, denominator = policy["payload_multiplier_numerator"], policy["payload_multiplier_denominator"]
    required = max(policy["minimum_bytes"], (payload*numerator+denominator-1)//denominator
                   + policy["evidence_reserve_bytes"])
    return dict(schema="s11-input-geometry-v1", cases=cases,
        sparse_qualification=dict(index=anchor["index"], input_tokens=anchor["input_tokens"],
            layer_chunk_checks=anchor["layer_chunk_checks"]),
        offline_payload_bytes_all_cases=payload-anchor["offline_bank_bytes"],
        qualification_anchor_payload_bytes=anchor["offline_bank_bytes"],
        offline_payload_bytes_including_qualification=payload, free_disk_required_bytes=required)


def document_jobs(base, row):
    document_lengths(row)
    for index, doc in enumerate(base.document_specs(row)):
        ids = list(doc["token_ids"])
        yield dict(sample_id=doc["token_ids_sha256"], dataset=row["dataset"],
            input_ids=ids, input_tokens=len(ids), input_ids_sha256=doc["token_ids_sha256"],
            references=["__offline_not_quality__"], offline_document_index=index, **case_scope(row))


def verify_phase(value, *, index, arm, timing, quality, diagnostic=False):
    require(isinstance(value, dict) and value.get("sha256") == _digest(
        {k: v for k, v in value.items() if k != "sha256"}), "phase seal mismatch")
    require(value.get("schema") == PHASE_SCHEMA and value.get("index") == index
            and value.get("arm") == arm and value.get("timing_valid") is timing
            and value.get("quality_eligible") is quality, "phase identity/eligibility mismatch")
    require(value.get("s1_policy") is (arm in POLICY_ARMS),
            "phase dynamic policy does not match arm")
    require(all(value.get(k) == v for k,v in phase_execution(arm, batch_group_chunks=value.get("batch_group_chunks")).items()), "phase system execution mapping differs")
    require(not (timing or quality) or all(value.get(k) == v for k,v in phase_execution(arm).items()), "formal batch fallback is forbidden")
    require(value.get("natural_boundary_inputs") is True, "phase natural boundary scope differs")
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
        budget = quality_token_budget(record.get("input_tokens"))
        finish = record.get("finish_reason")
        require("quality_requested_max_new_tokens" in value
                and value["quality_requested_max_new_tokens"] is None
                and value.get("technical_max_new_tokens") == record.get("technical_max_new_tokens") == budget
                and record.get("natural_EOS") is True and 0 < len(record["output_ids"]) <= budget
                and finish in ("stop", "length")
                and record.get("truncated") is (finish == "length")
                and record.get("quality_output_complete") is (finish == "stop")
                and record.get("quality_limit_reason") == ("engine_context_limit" if finish == "length" else None)
                and (finish != "length" or len(record["output_ids"]) == budget),
                "quality natural EOS/engine capacity binding mismatch")
        for name in ("answer_em", "answer_token_f1"):
            require(type(record.get(name)) in (int, float) and math.isfinite(record[name])
                    and 0 <= record[name] <= 1, "quality score invalid")
    return record


def qualification_passed(reference, candidate):
    from gpu_pilot.read_s11_results import native_identity
    for value in (reference,candidate):
        verify_phase(value,index=0,arm="qualification",timing=False,quality=False)
    return native_identity(reference,candidate)



def verify_partial_smoke(phases, row, qualification, sparse_qualification, *, arm="s1_system"):
    """Require full input, independent SWA and FFN accounting before quality."""
    from gpu_pilot.read_s11_results import audit_controller, documents
    require(row["index"] == QUALIFICATION_CASE_INDEX and len(row["documents"]) == 10,
            "smoke must use the complete ten-cache-block longest 256K case2")
    require(qualification.get("passed") is True and qualification.get("identity_check_count") == 80,
            "new strict numerical qualification must precede smoke")
    require(sparse_qualification.get("passed") is True
            and sparse_qualification.get("arm") == arm
            and sparse_qualification.get("input_ids_sha256") == row["input_ids_sha256"]
            and sparse_qualification.get("input_tokens") == row["input_tokens"]
            and sparse_qualification.get("identity_check_count") == case_geometry(row)["layer_chunk_checks"],
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
    require(phase.get("s1_policy") is (arm in POLICY_ARMS),
            "smoke controller policy does not match arm")
    actual = audit_controller(phase["controller"], row, phase, documents(row))
    require(all(actual.get(k,0)>0 for k in ("ffn_bypassed_rows","residual_bypassed_rows","mhc_bypassed_rows")),
            "smoke did not exercise complete S1 row bypass")
    require(actual.get("consumed_cached_pairs", 0) > 0,
            "smoke did not consume offline SWA memory in actual query attention")
    if arm in POLICY_ARMS:
        require(actual.get("dynamic_refreshed_rows", 0) > 0,
                "dynamic smoke did not refresh actual document SWA rows")
    body = dict(schema="s11-partial-smoke-v1", passed=True, arm=arm, index=row["index"],
        phase_sha256={arm:phase["sha256"]}, input_ids_sha256=row["input_ids_sha256"],
        input_tokens=row["input_tokens"], qualification_sha256=qualification["sha256"],
        quality_eligible=False, timing_valid=False, accuracy_guarantee=False,
        full_input_original_ten_documents=False, full_input_ten_natural_cache_blocks=True,
        original_document_count=row['original_document_count'], independent_swa_and_ffn_paths_exercised=True,
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
                          phase["manifest_sha256"], phase["s1_protocol_sha256"]))
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


def verify_shared_bank(phases, row):
    from gpu_pilot.read_s11_results import comparison_scope
    require(len(phases)==10,"both quality/warmup/three-repeat arms required")
    observations=[comparison_scope(p["controller"],p,spans=[(d["input_start"],d["input_end"],d["token_ids_sha256"]) for d in row["documents"]]) for p in phases]
    require(all(v==observations[0] for v in observations),"S1 offline GPU bank changed across arms")
    return observations[0]



def frozen_core_identity(repo, protocol):
    from gpu_pilot.launch_index_hit20_pilot import snapshot_sources
    actual={name:hashlib.sha256((STAGE/name).read_bytes()).hexdigest()
            for name in protocol['core_source_sha256']}
    native_sha=_digest(snapshot_sources(Path(repo)))
    require(actual==protocol['core_source_sha256']
        and native_sha==protocol['native_snapshot_sources_sha256'],
        'frozen S1 core/native source changed before execution')
    return dict(core_source_sha256=actual,core_policy_sha256=protocol['core_policy_sha256'],
                native_snapshot_sources_sha256=native_sha)


def verify_diagnostic_native_identity(identity, protocol):
    """Fail on the CPU before model loading if diagnostic topology pins drift."""
    pins = protocol["native_topology"]
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
            "natural-boundary pilot source changed after preflight")
    require(profile.identity(args) == identity["native_profile"] and base.effective_env(args) == identity["fixed_env"],
            "native configuration drift")
    disk = base.read_json(args.manifest)
    validate_inputs(base, disk, base.read_json(args.pilot_protocol), input_dir=args.manifest.parent)
    protocol = validate_protocol(args, disk)
    verify_diagnostic_native_identity(identity, protocol)
    require(disk["sha256"] == manifest["sha256"]
            and protocol["sha256"] == identity["s1_protocol_sha256"], "input/protocol changed")
    for filename, key in (("config.json", "checkpoint_config_sha256"),
                          ("model.safetensors.index.json", "checkpoint_index_sha256"),
                          ("tokenizer.json", "tokenizer_json_sha256")):
        require(base.file_digest(args.model_path/filename) == identity[key], "model/tokenizer identity changed")
    if native_source:
        require(all(identity.get(k)==v for k,v in frozen_core_identity(args.repo,protocol).items()),
                "frozen S1 core evidence changed after preflight")
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
    llm = controller = binder = batch_worker = None
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
        from kv_cache_engram.s1_system_runtime import install_worker_batch_schedule
        from kv_cache_engram.s11_qualification import S11QualificationController
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
        from kv_cache_engram.s1_runtime import run_s1_runtime_qualification
        from gpu_pilot.read_s11_results import validate_runtime_qualification
        runtime_reports=driver.gather(run_s1_runtime_qualification(layers,rank,torch,allow_cuda_sync,
            execution_sha256=identity["execution_configuration_sha256"],model_sha256=identity["checkpoint_index_sha256"]))
        runtime_qualification=base.seal(dict(schema="s1-runtime-all-ranks-qualification-v1",passed=True,
            execution_configuration_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"],reports=runtime_reports,
            empty_ep_qualification_sha256=empty_ep_qualification["sha256"],quality_eligible=False,timing_valid=False))
        validate_runtime_qualification(runtime_qualification,identity,empty_ep_qualification["sha256"])
        if rank==0:write_json(args.out_dir/"s1_runtime_qualification.json",runtime_qualification)
        driver.barrier()
        from kv_cache_engram.s1_system_batch import run_s1_system_metadata_qualification
        metadata_reports = driver.gather(dict(rank=rank, operator_report=
            run_s1_system_metadata_qualification(torch, allow_cuda_sync, device="cuda")))
        metadata_qualification = metadata_qualification_report(metadata_reports, identity)
        if rank == 0:
            write_json(args.out_dir/"s1_system_metadata_qualification.json", metadata_qualification)
            print(json.dumps(dict(phase="s1_system_metadata_qualification", status="complete", ranks=8)), flush=True)
        driver.barrier()
        binder = install_request_binding(model_runner, lambda: driver.request_binding if rank == 0 else None)
        empty_ep_capability = EmptyEPCapability(layers, empty_ep_qualification,
            execution_sha256=identity["execution_configuration_sha256"],
            model_sha256=identity["checkpoint_index_sha256"])
        if rank == 0:
            controller = S11QualificationController(layers, allow_sync=allow_cuda_sync,
                native_model=model_runner.model.model, expected_document_count=10,
                model_sha256=identity["checkpoint_index_sha256"], tokenizer_sha256=identity["tokenizer_json_sha256"],
                execution_sha256=identity["execution_configuration_sha256"], dependency_root=args.experiment_root,
                empty_ep_capability=empty_ep_capability)
            controller.install()
        batch_worker = install_worker_batch_schedule(model_runner.model.model, rank=rank,
            torch_module=torch, empty_ep_capability=empty_ep_capability, controller=controller,
            engine=driver.engine, phase_gather=driver.gather)
        driver.barrier()
        epoch = driver.gather(time.perf_counter())[0]

        def phase(row, name, mode, *, arm, quality=False, tokens=None, timing=False, repeat=None, fraction=None,
                  partial_smoke=False, verify_sparse=False, diagnostic=False, batch_group_chunks=None):
            began = time.perf_counter()
            execution = phase_execution(arm, batch_group_chunks=batch_group_chunks)
            require(not (timing or quality) or execution == phase_execution(arm), "formal system execution cannot fallback")
            if quality:
                require(tokens is None, "quality cannot receive an artificial output-token cap")
                tokens = quality_token_budget(row["input_tokens"], args.max_model_len)
            elif tokens is None:
                tokens = 1
            args.mode, args.max_new_tokens = ("quality" if quality else "throughput"), tokens
            driver.phase = name
            rid = f"s1:{row.get('index', 'doc')}:{name}:{row['sample_id']}"
            if controller is not None:
                # Fresh phase state, full-bank lookup and online memory preparation
                # belong to TTFT: clock starts before this reset.
                index_device = layers[0].self_attn.q_a_kv_swa_proj.weight.device if mode in ("sensitive", "reference_sparse", "all_selected") else None
                with allow_cuda_sync():
                    controller.configure(system_enabled=execution["system_enabled"], batch_group_chunks=execution["batch_group_chunks"])
                    special = dict(system_empty_group_qualification=True,
                        source_full_input_ids_sha256=row['source_full_input_ids_sha256']) if row.get('system_empty_group_qualification') else {}
                    controller.reset(rid, mode, arm=execution["controller_arm"], input_ids=row["input_ids"], documents=row.get("documents"),
                                     phase=name, fraction=fraction, index_device=index_device,
                                     verify_sparse=verify_sparse,**special)
            schedule = driver.gather(controller.batch_schedule if controller is not None else None)[0]
            batch_receipts = driver.gather(batch_worker.configure(schedule))
            if schedule is not None and controller is not None:
                controller.bind_batch_schedule(schedule, batch_receipts)
            driver.barrier()
            if rank == 0:
                print(json.dumps(dict(case=row.get("index"), phase=name, status="starting", mode=mode, arm=arm)), flush=True)
            records, window = base.run_phase(driver, [(rid, row)] if rank == 0 else [],
                                             int(rank == 0), rank, epoch, tokens, lambda _: None)
            batch_worker_reports = driver.gather(batch_worker.finish_phase())
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
            quality_scope = {}
            if quality:
                quality_scope = dict(quality_requested_max_new_tokens=None, technical_max_new_tokens=tokens)
                rec.update(technical_max_new_tokens=tokens,
                    quality_output_complete=rec.get("finish_reason") == "stop",
                    quality_limit_reason="engine_context_limit" if rec.get("finish_reason") == "length" else None)
            reports = driver.gather(report)
            require(isinstance(reports[0], dict) and all(p is None for p in reports[1:]),
                    "only rank0 may own natural-boundary state")
            require(diagnostic is False,"S1 formal runner does not execute inherited diagnostics")
            result = base.seal(dict(schema=PHASE_SCHEMA, name=name, mode=mode, arm=arm, repeat=repeat,
                fraction=fraction, record=rec, controller=reports[0], timing_valid=timing, quality_eligible=quality,
                **execution,
                system_batch_schedule=schedule, system_batch_configuration_receipts=batch_receipts,
                system_batch_worker_reports=batch_worker_reports,
                system_empty_group_qualification=row.get('system_empty_group_qualification',False),
                source_original_case_index=row.get('source_original_case_index'),
                source_full_input_ids_sha256=row.get('source_full_input_ids_sha256'),
                **quality_scope,
                partial_smoke=partial_smoke, diagnostic=diagnostic, natural_boundary_inputs=True,
                verify_sparse=verify_sparse, s1_policy=(arm in POLICY_ARMS),
                native_yoco_swa_share_hidden=identity["native_yoco_swa_share_hidden"],
                **phase_policy(arm),
                online_global_window_s=window["end_s"]-start, index=row.get("index"), dataset=row["dataset"],
                **case_scope(row),
                input_ids_sha256=row["input_ids_sha256"], manifest_sha256=manifest["sha256"],
                execution_configuration_sha256=identity["execution_configuration_sha256"],
                empty_ep_qualification_sha256=empty_ep_qualification["sha256"],
                runtime_qualification_sha256=runtime_qualification["sha256"],
                system_metadata_qualification_sha256=metadata_qualification["sha256"],
                s1_protocol_sha256=identity["s1_protocol_sha256"],
                online_full_bank_gpu_resident=(arm in ARMS), online_clock_started_before_reset=True,
                phase_local_memory_rebuilt=True, offline_build_included=False,
                native_first3_source2_preserved=True))
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
        qualification = base.seal(dict(schema="s11-qualification-v1", passed=True,
            cuda_mask_qualification_sha256=cuda_mask_qualification["sha256"],
            empty_ep_qualification_sha256=empty_ep_qualification["sha256"],
            runtime_qualification_sha256=runtime_qualification["sha256"],
            system_metadata_qualification_sha256=metadata_qualification["sha256"],
            reference_sha256=reference["sha256"], candidate_sha256=candidate["sha256"],
            identity_check_count=candidate["controller"]["identity_check_count"], quality_claim=False,
            qualification_input_tokens=16384, full_input_numerical_equivalence_claim=False,
            old_qualification_reused=False))
        if rank == 0:
            write_json(args.out_dir/"qualification.json", qualification)
        # A real one-document calibration request has an entirely empty first deep chunk.
        # It exercises native cache/state/EP handling in both policy paths before case2.
        empty_row=empty_qualification_row(manifest)
        if controller is not None:
            controller.clear_cache()
            controller.set_cache_output_dir(args.out_dir/"qualification.empty_s1.offline-kv")
        empty_doc=dict(sample_id=empty_row["input_ids_sha256"],dataset=empty_row["dataset"],
            input_ids=empty_row["input_ids"],input_tokens=16384,input_ids_sha256=empty_row["input_ids_sha256"],
            references=["__offline_not_quality__"])
        empty_capture=phase(empty_doc,"qualification.empty_s1.offline_doc0","capture_independent",arm="offline")
        if controller is not None:
            with allow_cuda_sync():
                controller.prepare_case(input_ids=empty_row["input_ids"],documents=empty_row["documents"],
                    index_device=layers[0].self_attn.q_a_kv_swa_proj.weight.device)
        driver.barrier()
        empty_reference=phase(empty_row,"qualification.empty_s1.reference","reference_sparse",arm="s1_system", batch_group_chunks=1)
        empty_group1=phase(empty_row,"qualification.empty_s1.system_group1","sensitive",
            arm="s1_system",verify_sparse=True,batch_group_chunks=1)
        empty_system=phase(empty_row,"qualification.empty_s1.system_target","sensitive",
            arm="s1_system",verify_sparse=True)
        empty_closure=None
        if rank==0:
            from gpu_pilot.read_s11_results import sparse_identity
            empty_system_group1=sparse_identity(empty_reference,empty_group1,empty_row,qualification["sha256"])
            empty_system_target=sparse_identity(empty_reference,empty_system,empty_row,qualification["sha256"])
            require(empty_system_group1["empty_chunk_exercised"] is True,"real empty chunk was not exercised")
            empty_closure=base.seal(dict(schema="s11-empty-chunk-qualification-v1",passed=True,
                qualification_sha256=qualification["sha256"],input_ids_sha256=empty_row["input_ids_sha256"],
                input_tokens=16384,first_chunk_selected_rows=0,second_chunk_selected_rows=128,
                deep_empty_layer_calls=37,quality_eligible=False,timing_valid=False,
                system_group1_qualification=empty_system_group1,
                system_target_qualification=empty_system_target,
                offline_phase_sha256=empty_capture["sha256"]))
            write_json(args.out_dir/"s1_empty_chunk_qualification.json",empty_closure)
        empty_closure=driver.gather(empty_closure)[0]
        if controller is not None:controller.clear_cache(reset_sparse_qualification=True)
        probe=empty_group_qualification_row(manifest)
        if controller is not None:controller.set_cache_output_dir(args.out_dir/'qualification.empty_group.offline-kv')
        probe_doc=dict(sample_id=probe['input_ids_sha256'],dataset=probe['dataset'],input_ids=probe['input_ids'],
            input_tokens=probe['input_tokens'],input_ids_sha256=probe['input_ids_sha256'],references=['__offline_not_quality__'])
        probe_capture=phase(probe_doc,'qualification.empty_group.offline_doc0','capture_independent',arm='offline')
        if controller is not None:
            with allow_cuda_sync():
                controller.prepare_case(input_ids=probe['input_ids'],documents=probe['documents'],
                    index_device=layers[0].self_attn.q_a_kv_swa_proj.weight.device,
                    system_empty_group_qualification=True,source_full_input_ids_sha256=probe['source_full_input_ids_sha256'])
        driver.barrier()
        probe_reference=phase(probe,'qualification.empty_group.reference','reference_sparse',arm='s1_system', batch_group_chunks=1)
        probe_candidate=phase(probe,'qualification.empty_group.system_group2','sensitive',arm='s1_system',
            batch_group_chunks=2,verify_sparse=True)
        probe_closure=None
        if rank==0:
            from gpu_pilot.read_s11_results import sparse_identity
            probe_sparse=sparse_identity(probe_reference,probe_candidate,probe,qualification['sha256'])
            probe_closure=base.seal(dict(schema='s11-empty-group-qualification-v1',passed=True,
                qualification_sha256=qualification['sha256'],source_original_case_index=2,
                source_full_input_ids_sha256=probe['source_full_input_ids_sha256'],
                input_ids_sha256=probe['input_ids_sha256'],input_tokens=32769,batch_group_chunks=2,
                groups_selected_rows=[0,127,1],actual_deep_empty_layer_calls=37,
                quality_eligible=False,timing_valid=False,actual_rag_quality_claim=False,
                sparse_qualification=probe_sparse,offline_phase_sha256=probe_capture['sha256']))
            write_json(args.out_dir/'s1_system_empty_group_qualification.json',probe_closure)
        probe_closure=driver.gather(probe_closure)[0]
        if controller is not None:controller.clear_cache(reset_sparse_qualification=True)
        def offline_bank(row, label):
            if controller is not None:
                controller.clear_cache()
                controller.set_cache_output_dir(args.out_dir/f"{label}.offline-kv")
            began=time.perf_counter()
            builds=[phase(doc,f"{label}.offline_doc{doc['offline_document_index']}",
                          "capture_independent",arm="offline") for doc in document_jobs(base,row)]
            if controller is not None:
                with allow_cuda_sync():
                    controller.prepare_case(input_ids=row["input_ids"],documents=row["documents"],
                        index_device=layers[0].self_attn.q_a_kv_swa_proj.weight.device)
            driver.barrier()
            elapsed=max(driver.gather(time.perf_counter()-began))
            return builds,elapsed

        anchor=qualification_anchor(manifest)
        anchor_builds,anchor_build_s=offline_bank(anchor,"qualification.anchor")
        anchor_build=None
        if rank==0:
            anchor_build=base.seal(dict(schema="s11-qualification-anchor-offline-build-v1",
                index=anchor["index"],input_ids_sha256=anchor["input_ids_sha256"],input_tokens=anchor["input_tokens"],
                qualification_sha256=qualification["sha256"],offline_build_s=anchor_build_s,
                offline_phase_sha256=[v["sha256"] for v in anchor_builds],
                phase_refs=[evidence_ref(args.out_dir,v["name"]+".json",v["sha256"]) for v in anchor_builds],
                timing_valid=False,quality_eligible=False,formal_bank_rebuilt_separately=True))
            write_json(args.out_dir/"qualification_anchor_offline_build.json",anchor_build)
        anchor_build=driver.gather(anchor_build)[0]
        from gpu_pilot.read_s11_results import sparse_identity
        sparse_reference=phase(anchor,"case2.s1_system_sparse_reference","reference_sparse",
            arm="s1_system",tokens=1,batch_group_chunks=1)
        for group in (1,2):
            group_phase=phase(anchor,f"case2.s1_system_group{group}_partial_smoke","sensitive",arm="s1_system",
                tokens=1,partial_smoke=True,verify_sparse=True,batch_group_chunks=group)
            if rank==0:
                group_qualification=sparse_identity(sparse_reference,group_phase,anchor,qualification["sha256"])
                write_json(args.out_dir/f"s1_system.group{group}_sparse_qualification.json",group_qualification)
        smoke_phase=phase(anchor,"case2.s1_system_partial_smoke","sensitive",arm="s1_system",
            tokens=1,partial_smoke=True,verify_sparse=True)
        smoke_result=None
        if rank==0:
            sparse_qualification=sparse_identity(sparse_reference,smoke_phase,anchor,qualification["sha256"])
            smoke_result=verify_partial_smoke([smoke_phase],anchor,qualification,sparse_qualification,arm="s1_system")
            write_json(args.out_dir/"s1_system.sparse_qualification.json",sparse_qualification)
            write_json(args.out_dir/"s1_system.partial_smoke.json",smoke_result)
        partial_smokes={"s1_system":driver.gather(smoke_result)[0]}
        for row in selected_rows(manifest):
            index=row["index"]
            builds,build_s=offline_bank(row,f"case{index}")
            require(set(partial_smokes) == set(POLICY_ARMS)
                    and all(s["passed"] for s in partial_smokes.values()),
                    "fresh full-input S1 dense/compact qualification and smoke required")
            quality = {}
            for _, arm in timing_schedule(index)[:len(ARMS)]:
                quality[arm] = phase(row, f"case{index}.{arm}_quality", "native" if arm == "recomputed" else "sensitive",
                    arm=arm, quality=True, fraction=FRACTIONS[arm])
                verify_phase(quality[arm], index=index, arm=arm, timing=False, quality=True)
            warmups = []
            for _, arm in timing_schedule(index)[:len(ARMS)]:
                warmups.append(phase(row, f"case{index}.{arm}_warmup", "native" if arm == "recomputed" else "sensitive",
                      arm=arm, tokens=32, fraction=FRACTIONS[arm]))
            measured = [phase(row, f"case{index}.{arm}_measure{repeat}", "native" if arm == "recomputed" else "sensitive",
                              arm=arm, tokens=32, timing=True, repeat=repeat, fraction=FRACTIONS[arm])
                        for repeat, arm in timing_schedule(index)]
            shared_bank = verify_shared_bank([*quality.values(), *warmups, *measured], row)
            timing_summary = summarize_timing(measured, index)
            result = base.seal(dict(schema="s11-case-paired-v1", index=index, dataset=row["dataset"],
                **case_scope(row),
                sample_id=row["sample_id"], references=row["references"], question=row["question"],
                input_tokens=row["input_tokens"], input_ids_sha256=row["input_ids_sha256"],
                quality={}, timing_phases=[], timing_summary=timing_summary,
                shared_bank_identity=shared_bank, fresh_paired_baseline=True,
                offline_build_s=build_s, offline_phase_sha256=[p["sha256"] for p in builds],
                qualification_sha256=qualification["sha256"], empty_chunk_qualification_sha256=empty_closure["sha256"], ffn_layers=list(FFN_LAYERS),
                system_empty_group_qualification_sha256=probe_closure['sha256'],
                qualification_anchor_offline_build_sha256=anchor_build["sha256"],
                original_s1_benchmark_executed=False,display_names={"recomputed":"Full","s1_system":"S1.1"},
                partial_smoke_sha256={arm:smoke["sha256"] for arm,smoke in partial_smokes.items()},
                offline_swa_layers=list(CAPTURE_LAYERS),offline_global_source_layers=[8,14,20], all_negative_results_retained=True,
                natural_boundary_inputs=True, input_geometry=case_geometry(row), calibration_claim=False))
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
        # Restore the fixed bootstrap/preflight identity after requests have ended.
        # This value never supplies the actual quality request budget above.
        args.mode, args.max_new_tokens = "quality", 64
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        if rank == 0:
            write_json(args.out_dir/"run.json", base.seal(dict(schema=SCHEMA, status="complete",
                identity=identity, manifest_sha256=manifest["sha256"], qualification=qualification, empty_chunk_qualification_sha256=empty_closure["sha256"], cases=completed,
                system_metadata_qualification_sha256=metadata_qualification["sha256"],
                system_empty_group_qualification_sha256=probe_closure['sha256'],
                partial_smoke_sha256={arm:smoke["sha256"] for arm,smoke in partial_smokes.items()},
                completed_case_indices=list(EVALUATION_INDICES), engine_init_s=init_s,
                qualification_case_index=QUALIFICATION_CASE_INDEX,
                qualification_anchor_offline_build_sha256=anchor_build["sha256"],
                original_s1_benchmark_executed=False,
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
        for obj, method in ((batch_worker, "close"), (controller, "close"), (binder, "restore"), (llm, "close")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except BaseException as exc:
                    cleanup.append(f"{method}: {exc!r}")
        if cleanup and not active_error:
            raise RuntimeError("natural-boundary cleanup failed: " + "; ".join(cleanup))


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
        identity.update(frozen_core_identity(args.repo,protocol))
        verify_diagnostic_native_identity(identity, protocol)
        rows = selected_rows(manifest)
        geometry = manifest_geometry(manifest, protocol)
        identity.update(pilot_source_sha256=source_hashes(args.experiment_root),
            native_yoco_swa_share_hidden=False,
            native_yoco_swa_configuration=native_yoco_swa_configuration(args.repo, args.model_path),
            native_profile=profile.identity(args), s1_protocol_sha256=protocol["sha256"],
            workflow_stage=SCHEMA, no_redknot=False,
            experiment_root=str(args.experiment_root.resolve()),
            isolated_native_snapshot=str(args.repo.resolve()))
        identity["execution_configuration_sha256"] = base.digest(dict(native_profile=identity["native_profile"],
            fixed_env=identity["fixed_env"], native_source=base.source_identity(args.repo),
            pilot_source_sha256=identity["pilot_source_sha256"], protocol_sha256=protocol["sha256"],
            native_yoco_swa_configuration=identity["native_yoco_swa_configuration"]))
        require(not native.gpu_runtime_imported(), "CPU preflight imported GPU runtime")
        require(shutil.disk_usage(args.experiment_root).free >= geometry["free_disk_required_bytes"],
                "insufficient space for actual full-cache payloads and evidence reserve")
        if args.check_only:
            print(json.dumps(dict(status="cpu_preflight_passed", gpu_initialized=False, identity=identity,
                evaluation_cases=[dict(index=r["index"], dataset=r["dataset"], input_tokens=r["input_tokens"], **case_scope(r)) for r in rows],
                qualification_case=0, qualification_input_tokens=16384, input_geometry=geometry, calibrated=False)))
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
            native_yoco_swa_share_hidden=False, natural_boundary_inputs=True, input_geometry=geometry,
            qualification_input_tokens=16384,qualification_case_index=QUALIFICATION_CASE_INDEX,
            manifest_sha256=manifest["sha256"], protocol=protocol, evaluation_case_indices=list(EVALUATION_INDICES),
            qualification_input_ids_sha256=qualification_row(manifest)["input_ids_sha256"],
            partial_smoke=dict(index=QUALIFICATION_CASE_INDEX, arms=list(POLICY_ARMS), mode="sensitive", output_tokens=1,
                               original_full_input=True, quality_eligible=False, timing_valid=False),
            audit_cases=[dict(index=r["index"], dataset=r["dataset"], sample_id=r["sample_id"], references=r["references"],
                **case_scope(r),
                question=r["question"], input_tokens=r["input_tokens"], input_ids_sha256=r["input_ids_sha256"],
                documents=r["documents"]) for r in rows])))
        base.wait_for_release(args, manifest["sha256"])
        verify_frozen(args, manifest, identity, profile, base, native_source=True)
        from quality_process_group import launch
        launch(functools.partial(worker, args, manifest, identity), args.dp, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

