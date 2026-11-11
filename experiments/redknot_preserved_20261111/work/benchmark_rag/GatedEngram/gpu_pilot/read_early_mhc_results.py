"""Independent CPU-only audit of sealed early-six/mHC pilot evidence.

Only closed three-arm quality + three C1 repetitions enter scores. This module
does not import the controller, runner, Torch, native code or any GPU runtime.
It reconstructs token selection from all stored scores and actual document
boundaries, not from a producer's selected-row count or median summary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_cache_engram.core import digest, require, validate_digest
from gpu_pilot.early_mhc_protocol import ARMS, CAPTURE_LAYERS, SPARSE_LAYERS, early_mhc_protocol

SCHEMA = "early-six-mhc-independent-readout-v2"
PHASE_SCHEMA = "kv-cache-engram-early-mhc-phase-v2"
RUN_SCHEMA = "kv-cache-engram-early-mhc-pilot-run-v2"
CASE_INDICES = (2, 3, 4, 5)
FRACTIONS = {"recomputed": None, "early20": .2, "early50": .5}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def sealed(value):
    require(type(value) is dict and value.get("sha256") == digest(
        {key: item for key, item in value.items() if key != "sha256"}), "canonical seal mismatch")
    return value


def read(path, evidence):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 512*1024**2,
            "missing, symlink or oversized evidence: "+path.name)
    payload = path.read_bytes()
    value = sealed(json.loads(payload, object_pairs_hook=_unique))
    evidence[path.name] = dict(file_sha256=hashlib.sha256(payload).hexdigest(), seal_sha256=value["sha256"])
    return value


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "invalid "+name)
    return value


def score(text, references):
    require(type(text) is str and type(references) is list and bool(references)
            and all(type(item) is str for item in references), "fixed answer/gold required")
    def norm(value):
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", value.lower())).split())
    words, outcomes = norm(text).split(), []
    for answer in references:
        gold = norm(answer).split()
        common = sum((Counter(words) & Counter(gold)).values())
        outcomes.append((float(norm(text) == norm(answer)),
                         2*common/(len(words)+len(gold)) if words or gold else 1.))
    return dict(answer_em=max(v[0] for v in outcomes), answer_token_f1=max(v[1] for v in outcomes))


def documents(row):
    docs = row.get("documents")
    require(type(docs) is list and len(docs) == 8, "independent audit requires eight raw document bindings")
    previous, keys, spans = None, set(), []
    for doc in docs:
        lo, hi, key = doc.get("input_start"), doc.get("input_end"), doc.get("token_ids_sha256")
        validate_digest(key, "document token SHA")
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= row["input_tokens"]
                and hi-lo == 32768 and (previous is None or lo == previous)
                and key not in keys, "document support, distinctness or length mismatch")
        previous = hi
        keys.add(key)
        spans.append((lo, hi, key))
    return spans


def mask_reference(scores, start, end, spans, fraction):
    require(type(start) is int and type(end) is int and 0 <= start < end
            and type(scores) is list and len(scores) == end-start,
            "all-token sensitivity scores must cover chunk")
    for value in scores:
        number(value, "sensitivity score")
    optional = [pos for pos in range(start+128, max(start+128, end-128))
                if any(lo+128 <= pos < hi-128 for lo, hi, _ in spans)]
    fixed = set(range(start, end))-set(optional)
    ranked = sorted(optional, key=lambda pos: (-scores[pos-start], pos))
    keep = math.ceil(fraction*len(optional))
    return sorted(pos-start for pos in fixed | set(ranked[:keep])), len(fixed), len(optional)


def _fingerprint(value):
    if value is None:
        return
    require(type(value) is dict and set(value) == {"shape", "dtype", "bytes", "sha256"},
            "raw activation fingerprint missing fields")
    shape = value["shape"]
    require(type(shape) is list and bool(shape) and all(type(n) is int and n >= 0 for n in shape)
            and type(value["dtype"]) is str and type(value["bytes"]) is int and value["bytes"] >= 0,
            "invalid activation geometry fingerprint")
    validate_digest(value["sha256"], "activation bytes")


def qualification(directory, preflight, evidence):
    raw = [read(directory/name, evidence) for name in
           ("qualification.native.json", "qualification.all_selected.json")]
    closure = read(directory/"qualification.json", evidence)
    require(closure.get("schema") == "early-mhc-qualification-v1" and closure.get("passed") is True
            and closure.get("reference_sha256") == raw[0]["sha256"]
            and closure.get("candidate_sha256") == raw[1]["sha256"]
            and closure.get("qualification_input_tokens") == 16384
            and closure.get("identity_check_count") == 80
            and closure.get("full_256k_numerical_parity_claim") is False
            and closure.get("old_qualification_reused") is False, "qualification closure mismatch")
    fingerprints = []
    for phase, mode in zip(raw, ("qualified_native", "all_selected")):
        require(phase.get("schema") == PHASE_SCHEMA and phase.get("mode") == mode
                and phase.get("index") == 0 and phase.get("arm") == "qualification"
                and phase.get("timing_valid") is False and phase.get("quality_eligible") is False,
                "foreign qualification phase")
        binding(phase, preflight)
        record, control = phase["record"], phase["controller"]
        require(record.get("success") is True and record.get("rank") == 0
                and record.get("input_tokens") == 16384 and len(record.get("output_ids", [])) == 1
                and record.get("output_ids_sha256") == digest(record["output_ids"]),
                "qualification output/length mismatch")
        rows = control.get("identity_fingerprints")
        expected = {(start, start+8192, layer) for start in (0, 8192) for layer in range(40)}
        require(type(rows) is list and len(rows) == 80 and control.get("identity_check_count") == 80,
                "all eighty raw layer/chunk fingerprints required")
        seen = set()
        for row in rows:
            key = tuple(row.get("key", []))
            require(key in expected and key not in seen, "foreign or duplicate qualification layer/chunk")
            seen.add(key)
            value = row["value"]
            require(set(value) == {"output", "lazy"}
                    and set(value["lazy"]) == {"residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix"},
                    "every native lazy field must be fingerprinted")
            require(value["output"] is not None, "decoder output missing")
            _fingerprint(value["output"])
            for item in value["lazy"].values():
                _fingerprint(item)
        require(digest(rows) == control.get("identity_fingerprints_sha256"), "fingerprint aggregate mismatch")
        fingerprints.append(rows)
    require(raw[0]["input_ids_sha256"] == raw[1]["input_ids_sha256"]
            and raw[0]["record"]["output_ids"] == raw[1]["record"]["output_ids"]
            and fingerprints[0] == fingerprints[1]
            and raw[1]["controller"].get("identity_exact") is True, "all-selected differs from native")
    return closure


def binding(phase, preflight):
    require(phase.get("manifest_sha256") == preflight["manifest_sha256"]
            and phase.get("early_mhc_protocol_sha256") == preflight["protocol"]["sha256"]
            and phase.get("execution_configuration_sha256") ==
                preflight["identity"]["execution_configuration_sha256"], "phase execution identity drift")


def audit_controller(control, row, phase, spans):
    require(control.get("schema") == "early-six-sensitive-mhc-controller-v1"
            and control.get("mode") == phase["mode"] and control.get("phase") == phase["name"]
            and control.get("input_ids_sha256") == row["input_ids_sha256"]
            and control.get("full_input_tokens") == row["input_tokens"]
            and control.get("sparse_layers") == list(SPARSE_LAYERS)
            and control.get("first_six_full") is True
            and control.get("actual_full_row_native_mhc") is True
            and control.get("formal_admission_unchanged") is True
            and control.get("not_model_flops_fraction") is True, "controller scope/safeguard mismatch")
    selection, adapter = control.get("selection"), control.get("adapter", {})
    require(type(selection) is list and adapter.get("schema") == "kv-cache-engram-early-mhc-adapter-v1"
            and adapter.get("both_full_row_native_mhc_norm_calls_retained") is True
            and adapter.get("per_layer_zero_inactive_attention_and_ffn") is True
            and adapter.get("interval_frozen_carrier") is False, "native mHC path safeguard missing")
    events, counts = adapter.get("events"), adapter.get("counts", {})
    require(type(events) is list, "raw adapter events required")
    if phase["arm"] == "recomputed":
        require(selection == [] and events == [] and counts.get("full_rows") == 0
                and counts.get("selected_rows") == 0, "baseline must remain native full computation")
        return dict(full_rows=0, selected_rows=0, bypass_rows=0, fraction=0.,
                    optional_chunks=0, actual_partial_chunks=0)
    fraction = FRACTIONS[phase["arm"]]
    require(adapter.get("native_execution") is True, "sparse path is not authenticated native execution")
    expected_chunks = [(start, min(start+8192, row["input_tokens"]))
                       for start in range(0, row["input_tokens"], 8192)]
    require([(item.get("start"), item.get("end")) for item in selection] == expected_chunks,
            "all exact native chunk masks required")
    event_expected, full, selected = {}, 0, 0
    optional_chunks, actual_partial_chunks = 0, 0
    for item in selection:
        start, end = item["start"], item["end"]
        ids, mandatory, optional = mask_reference(item.get("mean_early_relative_l2"), start, end, spans, fraction)
        require(item.get("selected_local_rows") == ids and item.get("selected_rows_sha256") == digest(ids)
                and item.get("full_rows") == end-start and item.get("selected_rows") == len(ids)
                and item.get("mandatory_rows") == mandatory and item.get("optional_rows") == optional,
                "actual token mask differs from six-layer score ranking / protected boundaries")
        validate_digest(item.get("plan_sha256"), "device plan")
        optional_chunks += int(optional > 0)
        actual_partial_chunks += int(0 < len(ids) < end-start)
        for layer in SPARSE_LAYERS:
            event_expected[(start, end, layer)] = item
        full += (end-start)*len(SPARSE_LAYERS)
        selected += len(ids)*len(SPARSE_LAYERS)
    seen = set()
    for event in events:
        key = event.get("chunk_start"), event.get("chunk_end"), event.get("layer")
        require(key in event_expected and key not in seen, "foreign or duplicate masked layer event")
        seen.add(key)
        item = event_expected[key]
        require(event.get("selected_rows") == item["selected_rows"] and event.get("full_rows") == item["full_rows"]
                and event.get("plan_sha256") == item["plan_sha256"]
                and event.get("request_id") == phase["record"]["request_id"] and event.get("phase") == phase["name"]
                and event.get("inactive_sublayer_output") == "zero" and event.get("full_row_native_mhc") is True
                and event.get("cached_hidden_or_sublayer_output") is False, "masked event binding differs")
    require(seen == set(event_expected) and counts.get("full_rows") == full
            and counts.get("selected_rows") == selected and counts.get("masked_layer_calls") == len(events)
            and counts.get("inactive_attention_zero_rows") == full-selected
            and counts.get("inactive_ffn_zero_rows") == full-selected,
            "raw selected/bypassed row counts inconsistent")
    actual = 1-selected/full
    require(control.get("actual_swa_reuse_fraction_in_eleven_layers") == actual, "reuse fraction differs from actual masks")
    return dict(full_rows=full, selected_rows=selected, bypass_rows=full-selected, fraction=actual,
                optional_chunks=optional_chunks, actual_partial_chunks=actual_partial_chunks)


def audit_phase(phase, row, preflight, *, arm, timing, quality, fixed_tokens=32):
    sealed(phase)
    binding(phase, preflight)
    require(phase.get("schema") == PHASE_SCHEMA and phase.get("index") == row["index"]
            and phase.get("dataset") == row["dataset"] and phase.get("arm") == arm
            and phase.get("mode") == ("native" if arm == "recomputed" else "sensitive")
            and phase.get("fraction") == FRACTIONS[arm] and phase.get("timing_valid") is timing
            and phase.get("quality_eligible") is quality and phase.get("input_ids_sha256") == row["input_ids_sha256"]
            and phase.get("online_early6_scout_and_plan_included") is True
            and phase.get("offline_build_included") is False
            and phase.get("native_global_kv_and_index_preserved") is True, "phase scope mismatch")
    require(phase.get("partial_smoke") is (not quality and not timing and fixed_tokens == 1),
            "partial smoke must remain separate from quality and formal timing")
    record = phase["record"]
    require(record.get("success") is True and record.get("rank") == 0
            and record.get("input_ids_sha256") == row["input_ids_sha256"]
            and record.get("input_tokens") == row["input_tokens"]
            and record.get("phase") == phase["name"] and record.get("sample_id") == row["sample_id"]
            and record.get("dataset") == row["dataset"]
            and record.get("request_id") == f"early-mhc:{row['index']}:{phase['name']}:{row['sample_id']}",
            "complete request scope mismatch")
    ids = record.get("output_ids")
    require(type(ids) is list and bool(ids) and all(type(x) is int and x >= 0 for x in ids)
            and digest(ids) == record.get("output_ids_sha256"), "output token evidence mismatch")
    if quality:
        require(record.get("natural_EOS") is True and len(ids) <= 64
                and record.get("truncated") is (record.get("finish_reason") == "length"), "quality EOS/cap mismatch")
        actual = score(record.get("text"), row["references"])
        require(all(abs(record.get(key, -9)-value) <= 1e-12 for key, value in actual.items()),
                "quality score not independently reproducible against fixed gold")
    else:
        require(type(fixed_tokens) is int and fixed_tokens in (1, 32)
                and (not timing or fixed_tokens == 32)
                and len(ids) == fixed_tokens and record.get("natural_EOS") is False
                and record.get("truncated") is False, "wrong fixed output length or EOS policy")
    if timing:
        ttft = number(record.get("online_ttft_s"), "online TTFT", True)
        elapsed = number(record.get("online_elapsed_s"), "online elapsed", True)
        number(record.get("online_prepare_to_submit_s"), "online preparation")
        require(ttft <= elapsed <= number(phase.get("online_global_window_s"), "common wall", True),
                "invalid online timing ordering")
    return audit_controller(phase["controller"], row, phase, documents(row))


def partial_smoke(directory, preflight, qual, evidence):
    """Full case2, actual partial masks in both arms; not another all-selected test."""
    row = next(item for item in preflight["audit_cases"] if item["index"] == 2)
    closure = read(directory/"partial_smoke.json", evidence)
    require(closure.get("schema") == "early-mhc-partial-smoke-v2"
            and closure.get("passed") is True and closure.get("index") == 2
            and closure.get("qualification_sha256") == qual["sha256"]
            and closure.get("input_ids_sha256") == row["input_ids_sha256"]
            and closure.get("input_tokens") == row["input_tokens"]
            and closure.get("quality_eligible") is False and closure.get("timing_valid") is False
            and closure.get("accuracy_guarantee") is False
            and closure.get("full_input_original_eight_documents") is True
            and closure.get("dynamic_early_score_and_partial_q_ffn_mhc_exercised") is True
            and closure.get("output_tokens_per_arm") == 1
            and type(closure.get("phase_sha256")) is dict
            and set(closure["phase_sha256"]) == {"early20", "early50"},
            "partial smoke closure missing, foreign or not qualified")
    actual_rows = {}
    phases = {}
    for arm in ("early20", "early50"):
        name = f"case2.{arm}_partial_smoke"
        phase = read(directory/(name+".json"), evidence)
        require(phase.get("name") == name and phase.get("repeat") is None
                and phase["sha256"] == closure["phase_sha256"][arm], "partial smoke raw closure mismatch")
        rows = audit_phase(phase, row, preflight, arm=arm, timing=False, quality=False, fixed_tokens=1)
        require(0 < rows["selected_rows"] < rows["full_rows"] and rows["bypass_rows"] > 0
                and rows["optional_chunks"] > 0 and rows["actual_partial_chunks"] > 0,
                "partial smoke did not execute any real inactive row")
        # Every full native 8K chunk containing enough optional rows must be
        # partially selected, not only one early chunk with the rest all native.
        for item in phase["controller"]["selection"]:
            if item["optional_rows"] >= 2:
                require(0 < item["selected_rows"] < item["full_rows"],
                        "eligible smoke native chunk silently ran all-selected")
        phases[arm] = phase
        actual_rows[arm] = rows
    lower = phases["early20"]["controller"]["selection"]
    upper = phases["early50"]["controller"]["selection"]
    require(len(lower) == len(upper), "smoke arms cover different native chunks")
    for a, b in zip(lower, upper):
        require((a["start"], a["end"]) == (b["start"], b["end"])
                and set(a["selected_local_rows"]) <= set(b["selected_local_rows"]),
                "early20 smoke mask is not nested inside early50")
    return closure, dict(passed=True, index=2, input_tokens=row["input_tokens"],
        input_ids_sha256=row["input_ids_sha256"], closure_sha256=closure["sha256"],
        actual_rows=actual_rows, fixed_output_tokens=1, numerical_identity_claim=False,
        quality_eligible=False, timing_valid=False, all_eleven_layers_all_native_chunks_audited=True)


def audit_case(directory, row, preflight, qual, smoke, evidence):
    index = row["index"]
    pair = read(directory/f"case{index}.paired.json", evidence)
    require(pair.get("schema") == "early-mhc-case-paired-v1"
            and pair.get("qualification_sha256") == qual["sha256"]
            and pair.get("partial_smoke_sha256") == smoke["sha256"]
            and all(pair.get(key) == row[key] for key in
                    ("index", "dataset", "sample_id", "references", "question", "input_tokens", "input_ids_sha256"))
            and pair.get("sparse_layers") == list(SPARSE_LAYERS)
            and pair.get("offline_capture_layers") == list(CAPTURE_LAYERS)
            and pair.get("all_negative_results_retained") is True
            and pair.get("exploratory_previously_exposed_inputs") is True, "three-arm closure identity mismatch")
    quality, timing, masks = {}, {arm: [] for arm in ARMS}, {}
    require(set(pair.get("quality", {})) == set(ARMS), "all three quality arms required")
    for arm in ARMS:
        phase = read(directory/f"case{index}.{arm}_quality.json", evidence)
        require(phase == pair["quality"][arm], "closure quality differs from raw phase")
        masks[arm] = audit_phase(phase, row, preflight, arm=arm, timing=False, quality=True)
        rec = phase["record"]
        quality[arm] = {key: rec[key] for key in ("text", "answer_em", "answer_token_f1", "truncated", "finish_reason")}
        quality[arm]["generated_tokens"] = len(rec["output_ids"])
        warmup = read(directory/f"case{index}.{arm}_warmup.json", evidence)
        audit_phase(warmup, row, preflight, arm=arm, timing=False, quality=False)
    expected = [(repeat, ARMS[(index+repeat+slot)%3]) for repeat in range(3) for slot in range(3)]
    phases = pair.get("timing_phases")
    require(type(phases) is list and [(p.get("repeat"), p.get("arm")) for p in phases] == expected,
            "three rotating same-case paired C1 rounds required")
    for recorded, (repeat, arm) in zip(phases, expected):
        phase = read(directory/f"case{index}.{arm}_measure{repeat}.json", evidence)
        require(phase == recorded, "closure timing differs from raw phase")
        audit_phase(phase, row, preflight, arm=arm, timing=True, quality=False)
        timing[arm].append(phase["record"]["online_ttft_s"])
    builds = []
    for doc_index, (_, _, doc_key) in enumerate(documents(row)):
        phase = read(directory/f"case{index}.offline_doc{doc_index}.json", evidence)
        binding(phase, preflight)
        require(phase.get("arm") == "offline" and phase.get("mode") == "capture_independent"
                and phase.get("input_ids_sha256") == doc_key, "offline document phase mismatch")
        payloads = phase["controller"].get("offline_payloads")
        require(type(payloads) is list and len(payloads) == len(CAPTURE_LAYERS), "all offline layers required")
        by_layer = {item["scope"]["layer"]: item for item in payloads}
        require(set(by_layer) == set(CAPTURE_LAYERS), "offline layer coverage changed")
        for layer, item in by_layer.items():
            name = doc_key+f".layer{layer}.bf16"
            scope = dict(model_sha256=preflight["identity"]["checkpoint_index_sha256"],
                tokenizer_sha256=preflight["identity"]["tokenizer_json_sha256"],
                execution_sha256=preflight["identity"]["execution_configuration_sha256"],
                document_sha256=doc_key, layer=layer, representation="swa-postnorm-prerope-bf16")
            require(item.get("scope") == scope and item.get("file") == name
                    and item.get("shape") == [32768, 512] and item.get("dtype") == "bfloat16"
                    and item.get("bytes") == 32768*512*2, "offline KV identity/geometry mismatch")
            path = directory/f"case{index}.offline-kv"/name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"],
                    "offline KV payload missing/truncated")
            sha = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4*1024**2), b""):
                    sha.update(chunk)
            require(sha.hexdigest() == item["sha256"], "offline KV payload bytes changed")
        builds.append(phase["sha256"])
    require(builds == pair.get("offline_phase_sha256"), "offline construction closure mismatch")
    medians = {arm: statistics.median(samples) for arm, samples in timing.items()}
    baseline = medians["recomputed"]
    comparisons = {arm: dict(em_loss_pp=100*(quality["recomputed"]["answer_em"]-quality[arm]["answer_em"]),
        f1_loss_pp=100*(quality["recomputed"]["answer_token_f1"]-quality[arm]["answer_token_f1"]),
        ttft_reduction_s=baseline-medians[arm], ttft_reduction_percent=100*(1-medians[arm]/baseline),
        ttft_speedup=baseline/medians[arm]) for arm in ARMS[1:]}
    return pair, dict(index=index, dataset=row["dataset"], references=row["references"], question=row["question"],
        input_tokens=row["input_tokens"], quality=quality, online_ttft_samples_s=timing,
        online_ttft_p50_s=medians, actual_rows_and_swa_reuse=masks, comparisons=comparisons,
        offline_build_s=number(pair.get("offline_build_s"), "offline build seconds"), closure_sha256=pair["sha256"])


def audit(directory, case_index=None):
    """Audit new closed cases without rereading prior 4.25-GiB offline banks.

    ``case_index`` deliberately emits a SUBSET result and never a final complete
    run claim, even if run.json already exists. The final audit without this
    option still checks every case, every bank and the complete run closure.
    """
    directory = Path(directory)
    require(case_index is None or type(case_index) is int and case_index in CASE_INDICES,
            "requested subset case is not registered")
    evidence, errors, results, closures = {}, [], [], []
    preflight = read(directory/"preflight.json", evidence)
    require(preflight.get("schema") == RUN_SCHEMA and preflight.get("protocol") == early_mhc_protocol()
            and preflight["protocol"].get("cases") == list(CASE_INDICES)
            and preflight.get("evaluation_case_indices") == list(CASE_INDICES), "foreign preflight/protocol")
    rows = preflight.get("audit_cases")
    require(type(rows) is list and [r.get("index") for r in rows] == list(CASE_INDICES)
            and [r.get("dataset") for r in rows] == ["hotpotqa", "2wikimqa"]*2,
            "fixed four-case cohort required; no score-based case selection")
    for row in rows:
        require(type(row.get("input_tokens")) is int and row["input_tokens"] >= 262145, "8x32K input required")
        validate_digest(row.get("input_ids_sha256"), "case input")
        documents(row)
    qual, smoke, smoke_summary = None, None, None
    if (directory/"qualification.json").exists():
        qual = qualification(directory, preflight, evidence)
    if (directory/"partial_smoke.json").exists():
        require(qual is not None, "partial smoke exists without strict native qualification")
        smoke, smoke_summary = partial_smoke(directory, preflight, qual, evidence)
    requested = rows if case_index is None else [r for r in rows if r["index"] == case_index]
    if any((directory/f"case{r['index']}.paired.json").exists() for r in requested):
        require(qual is not None and smoke is not None, "formal case exists before qualified partial smoke")
    if smoke is not None:
        for row in requested:
            if not (directory/f"case{row['index']}.paired.json").exists():
                continue
            try:
                closure, result = audit_case(directory, row, preflight, qual, smoke, evidence)
                closures.append(closure)
                results.append(result)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(dict(index=row["index"], error=str(exc)))
    complete = False
    if case_index is None and (directory/"run.json").exists():
        run = read(directory/"run.json", evidence)
        require(not errors and len(results) == 4 and run.get("schema") == RUN_SCHEMA
                and run.get("status") == "complete" and run.get("completed_case_indices") == list(CASE_INDICES)
                and run.get("identity") == preflight["identity"] and run.get("manifest_sha256") == preflight["manifest_sha256"]
                and run.get("qualification") == qual and run.get("partial_smoke_sha256") == smoke["sha256"]
                and run.get("cases") == closures,
                "final run does not close all four independently audited cases")
        complete = True
    subset_complete = case_index is not None and not errors and len(results) == 1
    status = "invalid" if errors else "complete" if complete else "subset_complete" if subset_complete else "incomplete"
    output = dict(schema=SCHEMA, status=status,
        completed_case_indices=[item["index"] for item in results], complete_cases=len(results), planned_cases=4,
        audit_scope="all_registered_cases" if case_index is None else "explicit_single_case_subset",
        requested_case_indices=[r["index"] for r in requested], final_run_audited=complete,
        overall_run_complete=complete, all_registered_cases_audited=complete,
        qualification_passed=qual is not None, partial_smoke=smoke_summary,
        cases=results, errors=errors, evidence=evidence,
        manifest_sha256=preflight["manifest_sha256"], execution_configuration_sha256=preflight["identity"]["execution_configuration_sha256"],
        protocol_sha256=preflight["protocol"]["sha256"], cuda_initialized=False,
        exploratory_previously_exposed_inputs=True, statistical_generalization=False,
        cold_start_claim=False, no_flops_fraction_claim=True, c4_measured=False)
    return dict(output, sha256=digest(output))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-index", type=int, choices=CASE_INDICES,
                        help="independent single-case subset; never a final run-complete claim")
    args = parser.parse_args(argv)
    result = audit(args.result_dir, case_index=args.case_index)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload, end="")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
