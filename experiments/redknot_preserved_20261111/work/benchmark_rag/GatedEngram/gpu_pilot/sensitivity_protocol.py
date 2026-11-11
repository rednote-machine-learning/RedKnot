"""CPU-only contracts for the finite, clean-input sensitivity diagnostic.

The observer perturbs only current native-chunk SWA KV and then restores native
execution. These records are not a whole-request approximate rollout, post-MoE
hidden measurements, or a runtime accuracy certificate. No model is imported.
"""
from __future__ import annotations

import math

from gpu_pilot import pilot_protocol as pp
from kv_cache_engram.core import digest, integer, require, validate_digest


OBSERVER_SCHEMA = "kv-cache-engram-native-sensitivity-v1"
PROTOCOL_SCHEMA = "kv-cache-engram-sensitivity-protocol-v1"
REPORT_SCHEMA = "kv-cache-engram-sensitivity-calibration-report-v1"
FROZEN_SCHEMA = "kv-cache-engram-sensitivity-admission-plan-v1"
LAYERS = (3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18)
EARLY_FULL_LAYERS = tuple(range(6))
OPERATIONS = ("reuse_current_chunk", "restore_block", "restore_half")


def sensitivity_protocol():
    """Fixed selection rule; calibration changes neither thresholds nor budget."""
    return pp.seal_record(dict(
        schema=PROTOCOL_SCHEMA,
        diagnostic_layers=list(LAYERS), early_full_layers=list(EARLY_FULL_LAYERS),
        diagnostic_cases="two calibration requests only; never tune on heldout",
        intervention="one layer and current native chunk at a time, same clean input hidden",
        operations=list(OPERATIONS),
        restoration_candidate="highest per-row post-RoPE/cast KV relative error; input position breaks ties",
        restoration_block_tokens=512, restoration_half_tokens=256,
        restoration_halves="both left and right halves, separately; never choose a favourable half",
        control_restoration="optional fixed nonselected 512-token block, diagnostic only",
        layer_ranking="maximum per-token projected attention-update relative L2, lower layer ID breaks ties",
        full_recompute_selection="one worst positive-error layer outside early-full region per calibration request; union",
        original_numerical_gate_required=True, new_thresholds_inferred=False,
        head_diagnostic_only=True, post_moe_hidden_measured=False,
        source_blocks_transferred_to_heldout=False,
        observer_changes_downstream_native=False,
        diagnostic_cost_in_online_ttft=False,
        diagnostic_cost_reported_separately=True,
        statistical_generalization_claimed=False, em_f1_guarantee=False))


def _number(value, name, *, upper=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            name + " must be finite and nonnegative")
    if upper is not None:
        require(value <= upper, name + " exceeds bound")
    return value


def _metrics(value, start, end):
    require(type(value) is dict and value.get("rows") == end-start,
            "metrics must cover exactly this query-token chunk")
    for key in ("relative_l2_max", "absolute_max", "relative_l2_mean"):
        _number(value.get(key), key)
    require(value["relative_l2_mean"] <= value["relative_l2_max"] + 1e-12,
            "mean error exceeds maximum")
    if "cosine_distance_max" in value:
        _number(value["cosine_distance_max"], "cosine distance", upper=2.000001)
    integer(value.get("worst_token_index"), "worst query-token index")
    require(start <= value["worst_token_index"] < end,
            "worst affected query token is outside chunk")


def _spans(values, start, end):
    require(type(values) is list, "source KV intervals required")
    intervals = []
    for span in values:
        require(type(span) is dict, "source interval must be a mapping")
        validate_digest(span.get("document_sha256"), "source cache document")
        for name in ("document_local_start", "document_local_end", "input_start", "input_end"):
            integer(span.get(name), name)
        a, b = span["input_start"], span["input_end"]
        local_a, local_b = span["document_local_start"], span["document_local_end"]
        require(start <= a < b <= end and 128 <= local_a < local_b <= 32768-128
                and b-a == local_b-local_a,
                "source KV interval includes foreign, guard, or non-document positions")
        block_start, block_end = span.get("block_start"), span.get("block_end")
        integer(block_start, "source block start"); integer(block_end, "source block end", 1)
        require(block_start % 512 == 0 and block_end == block_start+512
                and block_start <= local_a < local_b <= block_end,
                "source interval must remain within its aligned 512-token block")
        intervals.append((a, b))
    intervals.sort()
    require(all(a >= previous_b for (_, previous_b), (a, _) in zip(intervals, intervals[1:])),
            "source KV intervals overlap")
    return sum(b-a for a, b in intervals)


def validate_observer(value, *, input_sha256, input_tokens, expected_layers=LAYERS):
    """Validate complete chunk coverage and distinguish source KV/query outputs.

    Returned summaries retain negative restoration gains; high-error *query*
    positions are never relabelled as source KV positions. Hashes prove integrity,
    not that producer declarations were honestly executed on a real model.
    """
    validate_digest(input_sha256, "request digest")
    integer(input_tokens, "input tokens", 1)
    require(type(expected_layers) in (list, tuple) and bool(expected_layers)
            and len(set(expected_layers)) == len(expected_layers)
            and all(type(layer) is int and layer in LAYERS for layer in expected_layers),
            "unsupported/duplicate diagnostic layers")
    expected_layers = tuple(expected_layers)
    pp._sealed(value)
    require(type(value) is dict and value.get("schema") == OBSERVER_SCHEMA
            and value.get("complete") is True and value.get("input_sha256") == input_sha256
            and value.get("input_tokens") == input_tokens
            and value.get("selected_layers") == list(expected_layers)
            and value.get("diagnostic_only") is True and value.get("timing_eligible") is False
            and value.get("downstream_native_unchanged") is True
            and value.get("source_token_vs_query_token_distinguished") is True
            and value.get("full_block_hidden_measured") is False
            and value.get("preserves_native_cached_prefix_and_global_kv") is True
            and value.get("diagnostics_do_not_authorize_runtime_reuse") is True
            and value.get("native_execution") is True
            and value.get("operator_backend") == "deepinfer-sm90-native-bfloat16-kv-only-v1",
            "incomplete or foreign sensitivity observer")
    for name in ("request_id", "phase"):
        require(type(value.get(name)) is str and bool(value[name]), "observer binding missing")
    integer(value.get("head_count"), "head count", 1)
    head_count = value["head_count"]
    require(value.get("coverage") == {str(layer): input_tokens for layer in expected_layers},
            "observer coverage must be JSON-safe and complete")
    records = value.get("records")
    require(type(records) is list and bool(records), "diagnostic records missing")
    chunks, repairs, document_offsets = {}, [], {}
    for record in records:
        require(type(record) is dict and record.get("layer") in expected_layers
                and record.get("operation") in OPERATIONS
                and record.get("same_clean_input") is True
                and record.get("intervention_scope") == "current_chunk_swa_only"
                and record.get("full_block_hidden_measured") is False,
                "unknown or unsafe diagnostic operation")
        layer, start, end = record["layer"], record.get("start"), record.get("end")
        integer(start, "chunk start"); integer(end, "chunk end", 1)
        require(start < end <= input_tokens, "invalid diagnostic chunk")
        _metrics(record.get("attention_update"), start, end)
        heads = record.get("heads")
        require(type(heads) is list and len(heads) == head_count
                and [head.get("head") for head in heads] == list(range(head_count)),
                "complete ordered head diagnostics required")
        for head in heads:
            _metrics(head, start, end)
        source_rows = _spans(record.get("source_spans"), start, end)
        for span in record["source_spans"]:
            document = span["document_sha256"]
            offset = span["input_start"]-span["document_local_start"]
            require(offset >= 0, "source document starts outside request")
            require(document not in document_offsets or document_offsets[document] == offset,
                    "source cache document changed request position")
            document_offsets[document] = offset
        scores = record.get("source_block_scores")
        require(type(scores) is list and len(scores) == len(record["source_spans"]),
                "source KV block scores missing")
        for score, span in zip(scores, record["source_spans"]):
            require(type(score) is dict and score.get("input_start") == span["input_start"]
                    and score.get("input_end") == span["input_end"], "source block score/identity mismatch")
            _number(score.get("relative_l2_max"), "source KV error score")
        integer(record.get("reused_rows"), "reused source rows")
        integer(record.get("restored_rows"), "restored source rows")
        require(record["reused_rows"] + record["restored_rows"] == source_rows,
                "source row accounting mismatch")
        if "source_rows" in record:
            require(record["source_rows"] == source_rows, "declared source row count mismatch")
        if "attention_update_query_blocks" in record:
            blocks = record["attention_update_query_blocks"]
            require(type(blocks) is list and bool(blocks), "query block error coverage missing")
            cursor = start
            for block in blocks:
                require(type(block) is dict and block.get("input_start") == cursor,
                        "query block gap/overlap")
                stop = block.get("input_end")
                integer(stop, "query block end", 1)
                require(cursor < stop <= end and block.get("rows") == stop-cursor,
                        "query block row accounting mismatch")
                for key in ("relative_l2_max", "relative_l2_mean", "absolute_max"):
                    _number(block.get(key), key)
                require(block["relative_l2_mean"] <= block["relative_l2_max"] + 1e-12,
                        "query block mean error exceeds maximum")
                cursor = stop
            require(cursor == end, "query block coverage is incomplete")
            require(max(block["relative_l2_max"] for block in blocks)
                    == record["attention_update"]["relative_l2_max"], "query block maximum mismatch")
        key = layer, start, end
        if record["operation"] == "reuse_current_chunk":
            require(key not in chunks and record["restored_rows"] == 0,
                    "duplicate full-current-chunk intervention")
            require(record.get("restoration_label", "all_cached") == "all_cached",
                    "uncorrected intervention mislabelled as a repair")
            chunks[key] = record
        else:
            require(record["restored_rows"] > 0, "restoration must restore actual source KV rows")
            repairs.append(record)
    layer_summaries = {}
    for layer in expected_layers:
        selected = sorted((record for key, record in chunks.items() if key[0] == layer),
                          key=lambda record: record["start"])
        cursor = 0
        for record in selected:
            require(record["start"] == cursor, "diagnostic chunk gap/overlap")
            cursor = record["end"]
        require(cursor == input_tokens, "incomplete diagnostic layer")
        worst = max(selected, key=lambda r: (r["attention_update"]["relative_l2_max"], -r["start"]))
        layer_summaries[str(layer)] = dict(
            rows=input_tokens, chunks=len(selected),
            attention_update_relative_l2_max=worst["attention_update"]["relative_l2_max"],
            attention_update_absolute_max=max(r["attention_update"]["absolute_max"] for r in selected),
            affected_query_token_index=worst["attention_update"]["worst_token_index"],
            source_kv_rows_perturbed=sum(r["reused_rows"] for r in selected),
            head_relative_l2_max=[max(r["heads"][head]["relative_l2_max"] for r in selected)
                                 for head in range(head_count)])
    restoration = []
    seen_repairs = set()
    for record in repairs:
        key = record["layer"], record["start"], record["end"]
        require(key in chunks, "restoration has no paired uncorrected intervention")
        original = chunks[key]
        label = record.get("restoration_label")
        require(type(label) is str and label in ("high_error_512", "left_half_256", "right_half_256", "control_512"),
                "restoration identity missing")
        require(record["operation"] == ("restore_half" if "half" in label else "restore_block"),
                "restoration label/operation mismatch")
        identity = (*key, label)
        require(identity not in seen_repairs, "duplicate restoration intervention")
        seen_repairs.add(identity)
        require(record["source_spans"] == original["source_spans"]
                and record["reused_rows"] + record["restored_rows"] == original["reused_rows"],
                "restoration is not paired to exactly the same source KV candidate set")
        restored_spans = record.get("restored_spans")
        require(_spans(restored_spans, key[1], key[2]) == record["restored_rows"],
                "restored source intervals/count mismatch")
        for span in restored_spans:
            require(any(span["document_sha256"] == source["document_sha256"]
                        and source["input_start"] <= span["input_start"] < span["input_end"] <= source["input_end"]
                        and span["input_start"]-span["document_local_start"]
                            == source["input_start"]-source["document_local_start"]
                        for source in original["source_spans"]),
                    "restoration includes a foreign source KV row")
        before = original["attention_update"]["relative_l2_max"]
        after = record["attention_update"]["relative_l2_max"]
        if "error_reduction_relative_l2_max" in record:
            require(record["error_reduction_relative_l2_max"] == before-after,
                    "restoration gain is inconsistent or clipped")
        if "error_reduction_fraction" in record:
            require(record["error_reduction_fraction"] == ((before-after)/before if before > 0 else None),
                    "restoration recovery ratio is inconsistent or clipped")
        restoration.append(dict(layer=key[0], start=key[1], end=key[2], operation=record["operation"],
            restoration_label=label, restored_spans=restored_spans,
            restored_rows=record["restored_rows"], relative_l2_max_before=before,
            relative_l2_max_after=after, signed_error_reduction=before-after,
            gap_recovery_fraction=(before-after)/before if before > 0 else None,
            causal_scope="this current chunk, this layer, fixed clean input; not entire rollout"))
    for key, original in chunks.items():
        labels = {identity[-1] for identity in seen_repairs if identity[:-1] == key}
        if original["reused_rows"]:
            require({"high_error_512", "left_half_256", "right_half_256"} <= labels,
                    "every nonempty intervention requires its full-block and both-half restorations")
            scores, spans = original["source_block_scores"], original["source_spans"]
            selected_index = min(range(len(spans)),
                                 key=lambda i: (-scores[i]["relative_l2_max"], spans[i]["input_start"]))
            selected = spans[selected_index]
            block_start, block_end = selected.get("block_start"), selected.get("block_end")
            integer(block_start, "source original block start")
            integer(block_end, "source original block end", 1)
            require(block_start % 512 == 0 and block_end == block_start+512
                    and block_start <= selected["document_local_start"] < selected["document_local_end"] <= block_end,
                    "restoration must target an original 512-token source block")
            middle = block_start+256
            offset = selected["input_start"]-selected["document_local_start"]
            left_end = min(selected["document_local_end"], middle)
            right_start = max(selected["document_local_start"], middle)
            expected_repairs = {
                "high_error_512": selected,
                "left_half_256": dict(selected, document_local_end=left_end, input_end=offset+left_end),
                "right_half_256": dict(selected, document_local_start=right_start, input_start=offset+right_start)}
            if "control_512" in labels:
                others = [span for span in spans if span != selected]
                require(bool(others), "control requires a different source block")
                expected_repairs["control_512"] = min(others, key=lambda span: span["input_start"])
            for record in repairs:
                if (record["layer"], record["start"], record["end"]) == key:
                    require(record["source_block_scores"] == scores
                            and record["restored_spans"] == [expected_repairs[record["restoration_label"]]],
                            "restoration selection differs from fixed source-score/half/control rule")
        else:
            require(not labels, "zero-reuse chunk cannot have a restoration")
    return dict(layers=layer_summaries, restoration=restoration,
                source_token_vs_query_token_distinguished=True, head_control_enabled=False,
                full_block_hidden_measured=False, population_accuracy_certified=False)


def wrap_observer(observer, case, *, input_tokens, source_phase_sha256, execution_sha256,
                  manifest_sha256):
    """Bind an observer to its source raw phase. No native claim is synthesized."""
    require(type(case) is dict and case.get("split") == "calibration"
            and case.get("dataset") in pp.DATASETS, "only calibration sensitivity can select policy")
    validate_digest(source_phase_sha256, "source phase seal")
    validate_digest(execution_sha256, "execution identity")
    validate_digest(manifest_sha256, "manifest seal")
    summary = validate_observer(observer, input_sha256=case.get("input_sha256"), input_tokens=input_tokens)
    documents = {document["document_sha256"] for document in case["documents"]}
    require(all(span["document_sha256"] in documents for record in observer["records"]
                for span in record["source_spans"]), "sensitivity source cache document is not part of this case")
    return pp.seal_record(dict(schema=REPORT_SCHEMA, case_id=case["case_id"], dataset=case["dataset"],
        split="calibration", input_sha256=case["input_sha256"], input_tokens=input_tokens,
        execution_sha256=execution_sha256, manifest_sha256=manifest_sha256,
        source_phase_sha256=source_phase_sha256,
        sensitivity_protocol_sha256=sensitivity_protocol()["sha256"],
        observer=observer, summary=summary))


def freeze_admission_plan(reports, expected_cases, *, numerical_calibration_sha256):
    """Conservative layer protection; never replaces the existing numeric gate.

    One worst positive-error layer outside the first six is protected per
    calibration request. Exact calibration source blocks are diagnostics only;
    their offsets must never be transferred to different evaluation documents.
    """
    validate_digest(numerical_calibration_sha256, "numerical gate calibration seal")
    expected = pp._cases(expected_cases, "calibration")
    require(type(reports) in (list, tuple) and len(reports) == len(expected),
            "both complete calibration observers required")
    seen, choices, source_seals, execution, manifest = set(), [], [], None, None
    for report in reports:
        pp._sealed(report)
        require(report.get("schema") == REPORT_SCHEMA
                and report.get("sensitivity_protocol_sha256") == sensitivity_protocol()["sha256"],
                "foreign sensitivity report")
        case_id = report.get("case_id")
        require(case_id in expected and case_id not in seen, "foreign/duplicate sensitivity case")
        seen.add(case_id)
        case = expected[case_id]
        for field in ("dataset", "split", "input_sha256"):
            require(report.get(field) == case[field], "sensitivity case identity mismatch")
        validate_digest(report.get("execution_sha256"), "execution identity")
        if execution is None:
            execution = report["execution_sha256"]
        require(report["execution_sha256"] == execution, "mixed execution configurations")
        validate_digest(report.get("manifest_sha256"), "manifest seal")
        if manifest is None:
            manifest = report["manifest_sha256"]
        require(report["manifest_sha256"] == manifest, "mixed manifests")
        validate_digest(report.get("source_phase_sha256"), "source phase seal")
        summary = validate_observer(report.get("observer"), input_sha256=case["input_sha256"],
                                    input_tokens=report.get("input_tokens"))
        require(report.get("summary") == summary, "derived sensitivity summary mismatch")
        candidates = [(layer, summary["layers"][str(layer)]["attention_update_relative_l2_max"])
                      for layer in LAYERS if layer not in EARLY_FULL_LAYERS]
        layer, score = min(candidates, key=lambda item: (-item[1], item[0]))
        choices.append(dict(case_id=case_id, layer=layer if score > 0 else None,
                            observed_max_error=score, report_sha256=report["sha256"]))
        source_seals.append(report["source_phase_sha256"])
    require(seen == set(expected), "missing calibration sensitivity")
    selected = sorted({choice["layer"] for choice in choices if choice["layer"] is not None})
    protected = sorted(set(EARLY_FULL_LAYERS) | set(selected))
    return pp.seal_record(dict(schema=FROZEN_SCHEMA,
        sensitivity_protocol_sha256=sensitivity_protocol()["sha256"],
        numerical_calibration_sha256=numerical_calibration_sha256, execution_sha256=execution,
        manifest_sha256=manifest,
        calibration_cases=list(expected_cases),
        calibration_report_sha256=sorted(report["sha256"] for report in reports),
        source_phase_sha256=sorted(source_seals), choices=sorted(choices, key=lambda item: item["case_id"]),
        early_full_layers=list(EARLY_FULL_LAYERS), sensitivity_full_layers=selected,
        mandatory_full_layers=protected,
        eligible_reuse_layers=[layer for layer in LAYERS if layer not in protected],
        original_numerical_gate_required=True, candidate_reselection_on_heldout=False,
        source_blocks_transferred_to_heldout=False, head_specific_kv_recompute=False,
        runtime_admission_automatic=False, em_f1_guarantee=False))


def validate_frozen_plan(frozen, reports, expected_cases, *, numerical_calibration_sha256):
    pp._sealed(frozen)
    require(frozen == freeze_admission_plan(reports, expected_cases,
                numerical_calibration_sha256=numerical_calibration_sha256),
            "frozen sensitivity policy differs from full calibration evidence")
    return tuple(frozen["eligible_reuse_layers"])


def wrap_phase(phase, case, *, manifest_sha256, execution_sha256):
    """Independent reader/producer entry: validate raw full-reference binding."""
    pp._sealed(phase)
    require(phase.get("schema") == "kv-cache-engram-native-phase-v1"
            and phase.get("split") == "calibration"
            and phase.get("mode") in ("native", "shadow_full")
            and phase.get("timing_valid") is False
            and phase.get("input_ids_sha256") == case.get("input_sha256")
            and phase.get("dataset") == case.get("dataset")
            and phase.get("manifest_sha256") == manifest_sha256
            and phase.get("execution_configuration_sha256") == execution_sha256,
            "diagnostics must belong to an untimed clean full-reference calibration phase")
    record, observer = phase.get("record"), phase.get("sensitivity")
    require(type(record) is dict and record.get("success") is True
            and record.get("sample_id") == case.get("case_id")
            and record.get("input_ids_sha256") == case.get("input_sha256")
            and type(observer) is dict and observer.get("request_id") == record.get("request_id")
            and observer.get("phase") == phase.get("name"),
            "diagnostic observer/raw request mismatch")
    return wrap_observer(observer, case, input_tokens=record.get("input_tokens"),
        source_phase_sha256=phase["sha256"], execution_sha256=execution_sha256,
        manifest_sha256=manifest_sha256)
