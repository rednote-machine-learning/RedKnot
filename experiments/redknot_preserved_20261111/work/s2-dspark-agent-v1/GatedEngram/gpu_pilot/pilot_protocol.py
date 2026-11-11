"""CPU planning and fixed-candidate shadow auditing for the isolated GPU pilot.

No Torch/model/accelerator imports or launches occur here. Engineering KV error
limits are not EM/F1 tolerances. Native provenance flags are declarations whose
truth must be established by the isolated runner, not by a checksum.
"""
from dataclasses import asdict
import hashlib
import math

from kv_cache_engram.core import digest, integer, require, validate_digest
from kv_cache_engram.gate import GatePolicy


DATASETS = ("hotpotqa", "2wikimqa")
LAYERS = tuple(range(3, 8)) + tuple(range(9, 14)) + tuple(range(15, 19))
SEED = 20261102
ORACLE = {"relative_l2": .01, "absolute_error": .05}
SCOPE_FIELDS = {"model_sha256", "tokenizer_sha256", "execution_sha256",
                "representation", "storage_dtype", "width", "layers"}


def seal_record(value):
    require(type(value) is dict and "sha256" not in value, "seal expects an unsealed record")
    return dict(value, sha256=digest(value))


def _sealed(value):
    require(type(value) is dict and value.get("sha256") == digest(
        {key: item for key, item in value.items() if key != "sha256"}), "record seal mismatch")


def _text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidate_policies():
    """Predeclared engineering grid. These are not deployment defaults."""
    return tuple(GatePolicy(enabled=True, block_size=512, probe_count=probes,
        head_guard=128, tail_guard=128, relative_tolerance=relative,
        absolute_tolerance=absolute, norm_floor=1e-8,
        calibration_id="kce-native-pilot-20261102-candidate-not-qualified",
        seed="kce-native-pilot-20261102")
        for probes in (32, 128)
        for relative, absolute in ((0., 0.), (.001, .005), (.003, .015), (.01, .05)))


def pilot_protocol():
    return seal_record({"schema": "kv-cache-engram-native-pilot-protocol-v1",
        "seed": SEED, "datasets": list(DATASETS), "calibration_cases_per_dataset": 1,
        "evaluation_cases_per_dataset": 2, "num_chunks": 8, "chunk_tokens": 32768,
        "context_tokens": 262144, "input_shape_includes_additional_template_and_question": True,
        "input_claim": "LongBench-derived expanded composites, not native 256K questions",
        "layers": list(LAYERS), "reuse_scope": "SWA KV only; full Q/attention/hidden/MoE",
        "protected": "native global KV/index sources, template/question, document guards, chunk fragments",
        "candidate_policies": [asdict(policy) for policy in candidate_policies()],
        "oracle": dict(ORACLE), "max_false_accept_blocks": 0, "max_bad_reused_rows": 0,
        "selection": "calibration only; maximum shadow predicted reusable rows, then estimated KV callback row cost, then policy SHA",
        "heldout_selection": "never reselect or tune on heldout results",
        "fallback": "no eligible layer uses native full recomputation; committed-state errors abort and restart clean",
        "quality": {"max_new_tokens": 64, "stop_on_eos": True, "metrics": ["EM", "F1"],
                    "score_against": "fixed gold", "truncations_retained": True},
        "performance": {"concurrency": 1, "formal_repeats": 3, "fixed_output_tokens": 32,
                        "arm_order": "recomputed-first iff (case_index + case_index//2 + repeat) % 2 == 0; balanced within each heldout dataset",
                        "main_metric": "hot online TTFT including gate/plan/cache lookup/restore/fallback",
                        "offline_build_included": False, "model_load_included": False,
                        "c4_measured": False, "cold_start_claimed": False},
        "raw_storage": {"dtype": "bfloat16", "format": "separately sealed native canonical format",
                        "finite_required": True, "payload_sha256_required": True,
                        "float64_snapshot_format_reused": False},
        "runtime_admission_automatic": False, "em_f1_guarantee": False,
        "statistical_generalization_claimed": False})


def _sources(sources):
    require(type(sources) is dict and set(sources) == set(DATASETS), "exact pilot datasets required")
    result = {}
    for dataset in DATASETS:
        values = sources[dataset]
        require(type(values) in (list, tuple) and bool(values), "nonempty source records required")
        rows, ids = [], set()
        for index, item in enumerate(values):
            row = item[1] if type(item) in (tuple, list) and len(item) == 2 else item
            require(type(row) is dict, "source row must be a mapping")
            for key in ("_id", "context", "input"):
                require(type(row.get(key)) is str and bool(row[key]), "invalid source " + key)
            require(row["_id"] not in ids, "duplicate source ID")
            ids.add(row["_id"])
            rows.append({"dataset": dataset, "source_index": index, "source_id": row["_id"],
                "context_sha256": _text(row["context"]), "question_sha256": _text(row["input"]),
                "source_row_sha256": digest(row)})
        result[dataset] = rows
    return result


def build_source_plan(sources, previous_manifests, *, seed=SEED):
    """Split complete source contexts before composing 8x32K requests.

    No tokenizer is called and no token-fit claim is made. The caller must use
    the pinned official tokenizer, preserve the full primary, and fail if a
    frozen split lacks enough distinct donor tokens. No replacement by repeats.
    """
    require(type(seed) is int and seed == SEED, "pilot seed is frozen")
    rows = _sources(sources)
    require(type(previous_manifests) in (list, tuple) and bool(previous_manifests),
            "explicit old manifests required")
    exposed, questions, inherited = set(), set(), set()
    previous_hashes = []
    for manifest in previous_manifests:
        _sealed(manifest)
        previous_hashes.append(manifest["sha256"])
        old = manifest.get("evaluation_protocol", {}).get("old_exposure", {})
        exposed.update(old.get("context_sha256", []))
        questions.update(old.get("primary_question_sha256", []))
        inherited.update(old.get("manifest_sha256", []))
        for split in ("warmup", "cohort"):
            for row in manifest.get(split, []):
                require(type(row.get("question")) is str and bool(row["question"]), "old question missing")
                questions.add(_text(row["question"]))
                segments = row.get("context_segments")
                require(type(segments) is list and bool(segments), "complete old donor exposure required")
                for segment in segments:
                    if segment.get("kind") == "separator":
                        continue
                    validate_digest(segment.get("context_sha256"), "old context digest")
                    exposed.add(segment["context_sha256"])
    for checksum in exposed | questions | inherited:
        validate_digest(checksum, "old exposure fingerprint")
    grouped, selected_hashes, selected_questions = {}, set(), set()
    for dataset_index, dataset in enumerate(DATASETS):
        eligible, rejected, seen = [], [], set()
        for row in rows[dataset]:
            sha = row["context_sha256"]
            reason = ("previous_context_exposure" if sha in exposed else
                      "previous_primary_question" if row["question_sha256"] in questions else
                      "duplicate_context" if sha in seen else None)
            seen.add(sha)
            if reason:
                rejected.append(dict(row, reason=reason))
            else:
                eligible.append(row)
        pools = {"calibration": [], "evaluation": []}
        for row in eligible:
            bucket = int(digest({"seed": seed, "context_sha256": row["context_sha256"]}), 16) % 2
            pools["calibration" if bucket == 0 else "evaluation"].append(row)
        grouped[dataset] = {"source_rows": len(rows[dataset]), "source_inventory_sha256": digest(rows[dataset]),
                            "eligible_exact_unexposed_sources": len(eligible), "rejected": rejected, "splits": {}}
        for split in ("calibration", "evaluation"):
            count = 1 if split == "calibration" else 2
            order = sorted(pools[split], key=lambda row: (digest({"seed": seed, "dataset": dataset,
                "context_sha256": row["context_sha256"], "source_id": row["source_id"]}), row["source_index"]))
            selected = []
            for row in order:
                if row["context_sha256"] in selected_hashes or row["question_sha256"] in selected_questions:
                    continue
                ordinal = len(selected)
                anchor = (1, 4, 6)[(dataset_index + ordinal + (split == "evaluation")) % 3]
                selected.append(dict(row, split=split, ordinal=ordinal, anchor_chunk=anchor))
                selected_hashes.add(row["context_sha256"])
                selected_questions.add(row["question_sha256"])
                if len(selected) == count:
                    break
            require(len(selected) == count, "insufficient unique pilot primary candidates")
            grouped[dataset]["splits"][split] = {"pool": pools[split], "selected": selected}
    for dataset in DATASETS:
        for split in ("calibration", "evaluation"):
            values = grouped[dataset]["splits"][split]
            donors = [row for row in values["pool"] if row["context_sha256"] not in selected_hashes]
            require(bool(donors), "split has no donors; no automatic reuse of other split")
            values["donor_source_indices"] = [row["source_index"] for row in donors]
            values["donor_context_sha256"] = [row["context_sha256"] for row in donors]
    return seal_record({"schema": "kv-cache-engram-pilot-source-plan-v1", "seed": seed,
        "datasets": grouped, "previous_manifest_sha256": sorted(set(previous_hashes)),
        "inherited_previous_manifest_sha256": sorted(inherited),
        "excluded_context_sha256": sorted(exposed), "excluded_question_sha256": sorted(questions),
        "token_lengths_verified": False, "num_chunks": 8, "chunk_tokens": 32768,
        "calibration_evaluation_source_partition": "SHA256(seed,context_sha256) modulo2 globally before primary/donor selection",
        "primary_selection": "fixed SHA256(seed,dataset,context_sha256,source_id) ascending; no outcomes",
        "donor_selection": "source order within its own split, excluding all six reserved primaries",
        "claim_limit": "exact identities in declared prior manifests only; no semantic-independence or statistical guarantee"})


def _scope(scope):
    require(type(scope) is dict and set(scope) == SCOPE_FIELDS, "explicit exact native scope required")
    for key in ("model_sha256", "tokenizer_sha256", "execution_sha256"):
        validate_digest(scope[key], key)
    require(scope["representation"] == "swa-postnorm-prerope-v1"
            and scope["storage_dtype"] == "bfloat16-native-canonical" and scope["width"] == 512,
            "native BF16 canonical scope required")
    require(scope["layers"] == list(LAYERS), "exact fourteen pilot layers required")


def _cases(cases, split):
    require(type(cases) in (list, tuple) and bool(cases), "expected cases required")
    expected, counts = {}, {dataset: 0 for dataset in DATASETS}
    for case in cases:
        require(type(case) is dict and case.get("split") == split and case.get("dataset") in DATASETS,
                "case split/dataset mismatch")
        case_id = case.get("case_id")
        require(type(case_id) is str and bool(case_id) and case_id not in expected, "unique case IDs required")
        validate_digest(case.get("input_sha256"), "case input digest")
        require(type(case.get("source_document_ids")) is list and bool(case["source_document_ids"])
                and len(set(case["source_document_ids"])) == len(case["source_document_ids"]),
                "complete unique primary/donor source hashes required")
        for source in case["source_document_ids"]:
            validate_digest(source, "source context digest")
        docs = case.get("documents")
        require(type(docs) is list and len(docs) == 8, "eight cache documents required")
        require(len({doc["document_sha256"] for doc in docs}) == 8, "duplicate cache document in request")
        for doc in docs:
            validate_digest(doc["document_sha256"], "document digest")
            require(type(doc["tokens"]) is int and doc["tokens"] == 32768, "every document must be 32768 tokens")
        expected[case_id] = case
        counts[case["dataset"]] += 1
    require(all(count == (1 if split == "calibration" else 2) for count in counts.values()),
            "pilot requires one calibration or two evaluation cases per dataset")
    require(len({case["input_sha256"] for case in cases}) == len(cases), "duplicate full input")
    return expected


def _metric(value, name, *, optional=False):
    if value is None and optional:
        return
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            name + " must be finite and nonnegative")


def expected_case(row):
    """Minimal exact coverage binding from a fully tokenized frozen manifest row."""
    require(type(row) is dict, "manifest case required")
    split = {"calibration": "calibration", "heldout": "evaluation",
             "evaluation": "evaluation"}.get(row.get("split"))
    require(split is not None, "unknown manifest split")
    require(type(row.get("context_segments")) is list and bool(row["context_segments"]),
            "complete primary/donor source exposure required")
    sources = set()
    for segment in row["context_segments"]:
        require(type(segment) is dict, "source segment required")
        if segment.get("kind") == "separator":
            continue
        validate_digest(segment.get("context_sha256"), "source context digest")
        sources.add(segment["context_sha256"])
    require(bool(sources), "no source context evidence")
    docs = row.get("documents")
    require(type(docs) is list and len(docs) == 8, "eight exact tokenized documents required")
    result_docs = []
    for doc in docs:
        require(type(doc) is dict and doc.get("document_id") == doc.get("token_ids_sha256"),
                "document ID/token fingerprint mismatch")
        validate_digest(doc.get("token_ids_sha256"), "document tokens digest")
        require(type(doc.get("tokens")) is int and doc["tokens"] == 32768,
                "document must have exactly 32768 tokens")
        result_docs.append({"document_sha256": doc["token_ids_sha256"], "tokens": doc["tokens"]})
    require(len({doc["document_sha256"] for doc in result_docs}) == 8, "duplicate cache document")
    validate_digest(row.get("input_ids_sha256"), "full request digest")
    require(type(row.get("sample_id")) is str and bool(row["sample_id"])
            and row.get("dataset") in DATASETS, "case ID/dataset required")
    return {"case_id": row["sample_id"], "dataset": row["dataset"], "split": split,
            "input_sha256": row["input_ids_sha256"], "source_document_ids": sorted(sources),
            "documents": result_docs}


def wrap_shadow(reference_phase, row, manifest, identity):
    """Bind a completed native hook report to its sealed phase and manifest.

    Backend/native labels remain producer declarations, not authentication by
    hash. This wrapper neither creates an admission record nor changes a gate.
    """
    _sealed(reference_phase)
    _sealed(manifest)
    require(manifest.get("artifact_kind") == "kv_cache_engram_native_pilot"
            and manifest.get("pilot_protocol_sha256") == pilot_protocol()["sha256"],
            "fixed independent pilot manifest required")
    require(any(row == item for key in ("warmup", "cohort") for item in manifest.get(key, [])),
            "case does not match sealed manifest")
    case = expected_case(row)
    require(type(identity) is dict, "native execution identity required")
    expected_scope = dict(model_sha256=identity.get("checkpoint_index_sha256"),
        tokenizer_sha256=identity.get("tokenizer_json_sha256"),
        execution_sha256=identity.get("execution_configuration_sha256"),
        representation="swa-postnorm-prerope-v1", storage_dtype="bfloat16-native-canonical",
        width=512, layers=list(LAYERS))
    _scope(expected_scope)
    require(reference_phase.get("schema") == "kv-cache-engram-native-phase-v1"
            and reference_phase.get("mode") == "shadow_full"
            and reference_phase.get("manifest_sha256") == manifest["sha256"]
            and reference_phase.get("execution_configuration_sha256") == expected_scope["execution_sha256"]
            and reference_phase.get("input_ids_sha256") == case["input_sha256"]
            and reference_phase.get("dataset") == row["dataset"]
            and reference_phase.get("split") == row["split"]
            and reference_phase.get("index") == row.get("index"), "phase/case/manifest binding mismatch")
    hook = reference_phase.get("hook")
    require(type(hook) is dict and hook.get("schema") == "kv-cache-engram-native-hook-v1"
            and hook.get("mode") == "shadow_full" and hook.get("complete") is True
            and hook.get("native_execution_declared") is True
            and hook.get("backend") == "deepinfer-sm90-native-bfloat16-kv-only-v1"
            and hook.get("full_q_attention_moe_preserved") is True,
            "complete declared native KV-only shadow hook required")
    require(hook.get("scope") == expected_scope and hook.get("oracle") == ORACLE
            and hook.get("input_sha256") == case["input_sha256"]
            and hook.get("phase") == reference_phase.get("name"), "hook scope/request/phase mismatch")
    integer(row.get("input_tokens"), "request token count", 262145)
    require(hook.get("input_tokens") == row["input_tokens"], "hook request length mismatch")
    coverage = hook.get("coverage")
    require(type(coverage) is dict and len(coverage) == len(LAYERS)
            and {str(layer): end for layer, end in coverage.items()}
                == {str(layer): row["input_tokens"] for layer in LAYERS}, "incomplete full prompt coverage")
    for key in ("request_id", "phase"):
        require(type(hook.get(key)) is str and bool(hook[key]), "hook request/phase binding missing")
    require(type(hook.get("events")) is list and bool(hook["events"]), "shadow block observations missing")
    return seal_record(dict(schema="kv-cache-engram-native-shadow-report-v1", **{
        key: case[key] for key in ("case_id", "dataset", "split", "input_sha256")},
        request_id=hook["request_id"], phase=hook["phase"], scope=expected_scope,
        mode="shadow_full", complete=True, native_execution_declared=True,
        oracle=dict(ORACLE), events=hook["events"],
        source_phase_sha256=reference_phase["sha256"], source_manifest_sha256=manifest["sha256"],
        source_hook_sha256=digest(hook), provenance="caller-declared native backend; seals are not signatures"))


def _policy_map(policies):
    require(type(policies) in (list, tuple) and bool(policies), "fixed candidate policies required")
    result = {}
    for policy in policies:
        require(type(policy) is GatePolicy, "GatePolicy required")
        policy.__post_init__()
        require(policy.block_size == 512 and policy.head_guard == policy.tail_guard == 128,
                "pilot guard/block geometry is frozen")
        checksum = digest(asdict(policy))
        require(checksum not in result, "duplicate candidate policy")
        result[checksum] = policy
    return result


def _audit(reports, expected, scope, policies_by_layer):
    require(type(reports) in (list, tuple) and len(reports) == len(expected), "all case reports required")
    observed, summaries = set(), {}
    for layer, policies in policies_by_layer.items():
        for checksum in policies:
            summaries[(layer, checksum)] = dict(total_rows=0, blocks=0, would_reuse_rows=0,
                false_accept_blocks=0, bad_reused_rows=0, nonfinite_count=0,
                estimated_online_kv_callback_rows=0, oracle_bad_rows=0, actual_shadow_reused_rows=0)
    for report in reports:
        _sealed(report)
        require(report.get("schema") == "kv-cache-engram-native-shadow-report-v1"
                and report.get("mode") == "shadow_full" and report.get("complete") is True
                and report.get("native_execution_declared") is True,
                "complete declared native shadow report required")
        case_id = report.get("case_id")
        require(case_id in expected and case_id not in observed, "unknown/duplicate case report")
        observed.add(case_id)
        case = expected[case_id]
        require(report.get("scope") == scope and report.get("oracle") == ORACLE,
                "shadow scope or predeclared oracle mismatch")
        for key in ("dataset", "split", "input_sha256"):
            require(report.get(key) == case[key], "shadow case binding mismatch: " + key)
        for key in ("request_id", "phase"):
            require(type(report.get(key)) is str and bool(report[key]), "request/phase binding missing")
        docs = {doc["document_sha256"]: doc["tokens"] for doc in case["documents"]}
        partitions = {(layer, checksum, doc): [] for layer, policies in policies_by_layer.items()
                      for checksum in policies for doc in docs}
        require(type(report.get("events")) is list, "shadow events required")
        for event in report["events"]:
            key = (event.get("layer"), event.get("candidate_id"), event.get("document_sha256"))
            require(key in partitions and event.get("candidate_sha256") == key[1], "foreign layer/policy/document event")
            for name in ("input_sha256", "request_id", "phase"):
                require(event.get(name) == report[name], "event binding mismatch")
            start, end = event.get("document_local_start"), event.get("document_local_end")
            integer(start, "block start"); integer(end, "block end", 1)
            require(start < end <= docs[key[2]], "block outside complete document")
            partitions[key].append((start, end))
            for name in ("probe_count", "mandatory_count", "candidate_reusable_rows", "would_reuse_rows",
                         "bad_candidate_rows", "oracle_bad_reused_rows", "oracle_bad_rows", "nonfinite_count"):
                integer(event.get(name), name)
            require(type(event.get("accepted")) is bool and type(event.get("nonfinite")) is bool
                    and type(event.get("eligible_for_reuse")) is bool, "explicit decision flags required")
            require(event["nonfinite"] == bool(event["nonfinite_count"]), "nonfinite count/flag mismatch")
            reusable = event["candidate_reusable_rows"]
            require(reusable <= end-start
                    and event["bad_candidate_rows"] <= reusable
                    and event["bad_candidate_rows"] <= event["oracle_bad_rows"] <= end-start,
                    "invalid candidate/oracle row coverage")
            optional = bool(event["nonfinite_count"]) or not event["eligible_for_reuse"]
            for name in ("probe_relative_max", "probe_absolute_max", "candidate_relative_max",
                         "candidate_absolute_max", "full_relative_max", "full_absolute_max"):
                _metric(event.get(name), name, optional=optional or ("candidate" in name and reusable == 0))
            policy = policies_by_layer[key[0]][key[1]]
            if event["eligible_for_reuse"]:
                require(start % 512 == 0 and end == min(start+512, docs[key[2]]), "only whole aligned blocks may gate")
                mandatory = sum(p < 128 or p >= docs[key[2]]-128 for p in range(start, end))
                probes = min(policy.probe_count, end-start-mandatory)
                require(event["mandatory_count"] == mandatory and event["probe_count"] == probes
                        and reusable == end-start-mandatory-probes, "probe/guard/candidate coverage mismatch")
                accepted = bool(policy.enabled and reusable and probes and not event["nonfinite_count"]
                    and event["probe_relative_max"] <= policy.relative_tolerance
                    and event["probe_absolute_max"] <= policy.absolute_tolerance)
                require(event["accepted"] == accepted, "decision does not match frozen gate")
                if reusable and not event["nonfinite_count"]:
                    oracle_bad = event["candidate_relative_max"] > ORACLE["relative_l2"] or event["candidate_absolute_max"] > ORACLE["absolute_error"]
                    require(bool(event["bad_candidate_rows"]) == oracle_bad, "candidate oracle maxima/count mismatch")
            else:
                require(not event["accepted"] and reusable == 0 and event["probe_count"] == 0
                        and event.get("mode") == "full_recompute" and bool(event.get("fallback_reason")),
                        "ineligible fragment must remain full recompute")
            bad = event["bad_candidate_rows"] if event["accepted"] else 0
            require(event["oracle_bad_reused_rows"] == bad, "bad reused count mismatch")
            summary = summaries[key[:2]]
            reused = reusable if event["accepted"] else 0
            require(event["would_reuse_rows"] == reused, "shadow predicted reuse count mismatch")
            for name in ("actual_reused_rows", "reused_rows"):
                if name in event:
                    require(type(event[name]) is int and event[name] == 0,
                            "shadow full reference may not actually reuse cached KV")
            summary["total_rows"] += end-start
            summary["blocks"] += 1
            summary["would_reuse_rows"] += reused
            summary["bad_reused_rows"] += bad
            summary["false_accept_blocks"] += bool(bad)
            summary["nonfinite_count"] += event["nonfinite_count"]
            summary["oracle_bad_rows"] += event["oracle_bad_rows"]
            # Native fallback fills only the rejected candidate rows; probes
            # and guards already computed are not projected a second time.
            summary["estimated_online_kv_callback_rows"] += end-start-reused
        for key, spans in partitions.items():
            cursor = 0
            for start, end in sorted(spans):
                require(start == cursor, "shadow document gap/overlap/duplicate")
                cursor = end
            require(cursor == docs[key[2]], "incomplete shadow document coverage")
    require(observed == set(expected), "missing case report")
    for summary in summaries.values():
        summary["would_reuse_row_fraction"] = summary["would_reuse_rows"] / summary["total_rows"]
        summary["eligible"] = bool(summary["would_reuse_rows"] and not summary["false_accept_blocks"]
                                   and not summary["bad_reused_rows"] and not summary["nonfinite_count"])
    return summaries


def fit_shadow(calibration_reports, expected_cases, scope, policies=None):
    """Select by layer using every complete calibration block and explicit oracle."""
    _scope(scope)
    expected = _cases(expected_cases, "calibration")
    policy_map = _policy_map(list(candidate_policies()) if policies is None else policies)
    require(set(policy_map) == set(_policy_map(list(candidate_policies()))),
            "candidate grid differs from fixed pilot protocol")
    summaries = _audit(calibration_reports, expected, scope, {layer: policy_map for layer in LAYERS})
    layers = {}
    for layer in LAYERS:
        candidates = [{"policy_sha256": checksum, "policy": asdict(policy_map[checksum]),
                       "summary": summaries[(layer, checksum)]} for checksum in sorted(policy_map)]
        eligible = [entry for entry in candidates if entry["summary"]["eligible"]]
        selected = min(eligible, key=lambda entry: (-entry["summary"]["would_reuse_rows"],
            entry["summary"]["estimated_online_kv_callback_rows"], entry["policy_sha256"])) if eligible else None
        layers[str(layer)] = {"candidates": candidates, "selected": selected,
                              "mode": "pilot_gated_kv" if selected else "full_recompute"}
    return seal_record({"schema": "kv-cache-engram-native-pilot-calibration-v1",
        "protocol_sha256": pilot_protocol()["sha256"], "scope": scope, "oracle": dict(ORACLE),
        "calibration_cases": list(expected_cases),
        "calibration_report_sha256": [report["sha256"] for report in calibration_reports],
        "layers": layers, "runtime_admission_automatic": False,
        "evidence_claim": "declared native full-row KV error calibration; not EM/F1/TTFT qualification"})


def validate_shadow(frozen, evaluation_reports, expected_cases):
    """Validate selected candidates on disjoint sources; no candidate reselection."""
    _sealed(frozen)
    require(frozen.get("schema") == "kv-cache-engram-native-pilot-calibration-v1"
            and frozen.get("protocol_sha256") == pilot_protocol()["sha256"]
            and frozen.get("oracle") == ORACLE, "foreign frozen calibration")
    scope = frozen["scope"]
    _scope(scope)
    old = _cases(frozen["calibration_cases"], "calibration")
    expected = _cases(expected_cases, "evaluation")
    for field in ("input_sha256",):
        require(not {case[field] for case in old.values()} & {case[field] for case in expected.values()},
                "full request split leakage")
    require(not {source for case in old.values() for source in case["source_document_ids"]}
            & {source for case in expected.values() for source in case["source_document_ids"]},
            "source primary/donor split leakage")
    require(not {doc["document_sha256"] for case in old.values() for doc in case["documents"]}
            & {doc["document_sha256"] for case in expected.values() for doc in case["documents"]},
            "cache document split leakage")
    selected, all_candidates = {}, {}
    require(type(frozen.get("layers")) is dict and set(frozen["layers"]) == {str(layer) for layer in LAYERS},
            "all calibrated layers required")
    for layer in LAYERS:
        entry = frozen["layers"][str(layer)]
        require(type(entry.get("candidates")) is list and bool(entry["candidates"]), "all frozen candidates required")
        candidate_map = _policy_map([GatePolicy(**candidate["policy"]) for candidate in entry["candidates"]])
        require(set(candidate_map) == set(_policy_map(list(candidate_policies()))),
                "frozen candidate grid differs from protocol")
        require(all(candidate["policy_sha256"] == digest(candidate["policy"])
                    for candidate in entry["candidates"]), "candidate fingerprint mismatch")
        all_candidates[layer] = candidate_map
        chosen = entry["selected"]
        if chosen is not None:
            policy = GatePolicy(**chosen["policy"])
            require(digest(asdict(policy)) == chosen["policy_sha256"] and chosen["summary"]["eligible"] is True,
                    "selected policy fingerprint/eligibility mismatch")
            require(chosen in entry["candidates"], "selected candidate absent from frozen grid")
            selected[layer] = {chosen["policy_sha256"]: policy}
    # Validate every observed candidate's shape/coverage, including unselected
    # layers; malformed evidence is never silently filtered. Only the frozen
    # selected candidate contributes to the held-out decision, never reselection.
    summaries = _audit(evaluation_reports, expected, scope, all_candidates)
    layer_reports = {}
    for layer in LAYERS:
        if layer not in selected:
            layer_reports[str(layer)] = {"mode": "full_recompute", "kv_validation_pass": False,
                                         "reason": "no_eligible_calibration_candidate"}
        else:
            summary = summaries[(layer, next(iter(selected[layer])))]
            layer_reports[str(layer)] = {"mode": "pilot_gated_kv", "kv_validation_pass": summary["eligible"],
                                         "summary": summary}
    return seal_record({"schema": "kv-cache-engram-native-pilot-heldout-v1",
        "calibration_sha256": frozen["sha256"], "evaluation_cases": list(expected_cases),
        "evaluation_report_sha256": [report["sha256"] for report in evaluation_reports],
        "layers": layer_reports, "any_gated_layer": bool(selected),
        "kv_validation_pass": bool(selected) and all(value["kv_validation_pass"] for layer, value
                              in layer_reports.items() if int(layer) in selected),
        "candidate_reselection_performed": False, "runtime_admission_automatic": False,
        "em_f1_guarantee": False})


def make_pilot_records(frozen, identity_by_layer):
    """Explicit pilot-only evidence binding; never a deployment certificate.

    The runner, not this function, authenticates the native capture producer and
    enforces isolated-pilot scope. Calling this does not load or run any model.
    """
    from kv_cache_engram.native_reuse import NativeCalibrationRecord, NativeCacheIdentity
    _sealed(frozen)
    require(frozen.get("schema") == "kv-cache-engram-native-pilot-calibration-v1"
            and frozen.get("protocol_sha256") == pilot_protocol()["sha256"]
            and frozen.get("oracle") == ORACLE, "foreign pilot calibration")
    _scope(frozen["scope"])
    require(type(identity_by_layer) is dict and set(identity_by_layer) == set(LAYERS), "all native identity templates required")
    policies, result = {}, {}
    for layer in LAYERS:
        identity = identity_by_layer[layer]
        require(type(identity) is NativeCacheIdentity and identity.layer == layer, "native identity/layer mismatch")
        for key in SCOPE_FIELDS - {"layers"}:
            require(getattr(identity, key) == frozen["scope"][key], "native calibration scope mismatch")
        entry = frozen["layers"][str(layer)]
        candidate_map = _policy_map([GatePolicy(**candidate["policy"]) for candidate in entry["candidates"]])
        require(set(candidate_map) == set(_policy_map(list(candidate_policies()))),
                "frozen candidate grid differs from protocol")
        chosen = entry["selected"]
        if chosen is None:
            continue
        policy = GatePolicy(**chosen["policy"])
        require(chosen in entry["candidates"] and chosen["summary"]["eligible"] is True
                and chosen["policy_sha256"] == digest(asdict(policy)),
                "unqualified or mismatched selected policy")
        policies[layer] = policy
        result[layer] = NativeCalibrationRecord.bind(identity, policy, frozen["sha256"])
    return policies, result
