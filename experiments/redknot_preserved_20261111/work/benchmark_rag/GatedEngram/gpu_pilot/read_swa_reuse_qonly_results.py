"""Independent CPU-only audit of sealed early-six/mHC pilot evidence.

Only closed two-arm quality + three C1 repetitions enter scores. This module
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
import os
from pathlib import Path
import re
import statistics
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_cache_engram.core import digest, require, validate_digest
from gpu_pilot.swa_reuse_qonly_protocol import ARMS, CAPTURE_LAYERS, FFN_LAYERS, swa_reuse_qonly_protocol

SCHEMA = "swa-reuse-qonly-independent-readout-v1"
PHASE_SCHEMA = "kv-cache-engram-swa-reuse-qonly-phase-v1"
RUN_SCHEMA = "kv-cache-engram-swa-reuse-qonly-pilot-run-v1"
CASE_INDICES = (2, 3, 4, 5)
FRACTIONS = {"recomputed": None, "ffn15": .15}
MAX_JSON_BYTES = 8*1024**3


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
    path = Path(path)
    def identity(info):
        return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_JSON_BYTES,
            "missing, symlink, nonregular or oversized evidence: "+path.name)
    fd = os.open(path, os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
    with os.fdopen(fd,"rb") as handle:
        opened = os.fstat(handle.fileno())
        require(stat.S_ISREG(opened.st_mode) and identity(opened) == identity(before), "evidence changed while opening")
        payload = handle.read(before.st_size+1)
        after = os.fstat(handle.fileno())
    require(len(payload) == before.st_size and identity(after) == identity(before)
            and identity(path.lstat()) == identity(before), "evidence changed during bounded read")
    def reject_constant(value):
        raise ValueError("non-finite JSON constant: "+value)
    value = sealed(json.loads(payload, object_pairs_hook=_unique, parse_constant=reject_constant))
    evidence[path.name] = dict(file_sha256=hashlib.sha256(payload).hexdigest(), seal_sha256=value["sha256"])
    return value


def verify_reference(reference, path, value, evidence, *, extras=()):
    expected = {"file", "file_sha256", "seal_sha256"}|set(extras)
    require(type(reference) is dict and set(reference) == expected
            and reference.get("file") == path.name and Path(reference["file"]).name == reference["file"]
            and reference.get("file_sha256") == evidence[path.name]["file_sha256"]
            and reference.get("seal_sha256") == value["sha256"],
            "closure reference does not bind original raw bytes and seal")


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
            and type(scores) is list and len(scores) == end-start and fraction == .15,
            "all-token adjacent-SWA scores must cover chunk")
    for value in scores:
        number(value, "per-token adjacent-SWA score")
    optional = [pos for pos in range(start, end) if any(lo <= pos < hi for lo, hi, _ in spans)]
    fixed = set(range(start,end))-set(optional)
    ranked = sorted(optional, key=lambda pos: (-scores[pos-start],pos))
    keep = (15*len(optional)+99)//100
    return sorted(pos-start for pos in fixed|set(ranked[:keep])), len(fixed), len(optional)


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


def projection_identity(control):
    """Independently bind all nonempty split/gather checks to native layouts."""
    swa=control.get("swa_events")
    events=control.get("projection_identity_events")
    require(type(swa) is list and type(events) is list, "split projection raw evidence required")
    expected={(s,s+8192,l) for s in (0,8192) for l in FFN_LAYERS}
    by_key={}
    for event in swa:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in by_key,"qualification SWA coverage mismatch")
        positions=_positions(event.get("current_positions"),key[0],key[1],full_required=key[2]<20)
        require(event.get("full_rows")==len(positions) and event.get("online_rows")==len(positions)
                and event.get("online_local_rows")==list(range(len(positions)))
                and event.get("reused_rows")==0,"qualification must recompute every native SWA row")
        by_key[key]=event
    require(set(by_key)==expected,"qualification missing layer/chunk SWA evidence")
    nonempty={k for k,v in by_key.items() if v["full_rows"]>0}
    seen=set()
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in nonempty and key not in seen and event.get("full_rows")==by_key[key]["full_rows"]
                and event.get("checks")==3 and type(event.get("max_absolute_error")) in (int,float)
                and event["max_absolute_error"]==0
                and event.get("checks_named")==["split_q","split_swa","packed_gather_swa127"],
                "split Q/KV/gather exact-zero checks missing or failed")
        seen.add(key)
    count=3*len(nonempty)
    require(count>0 and seen==nonempty and control.get("projection_identity_checks")==count,
            "projection qualification count/coverage mismatch")
    return count


def cache_restore_identity(control):
    """Every nonempty native layout must prove exact canonical/RoPE/cast assembly."""
    source = control.get("swa_events")
    events = control.get("cache_restore_identity_events")
    require(type(source) is list and type(events) is list, "cache restoration numerical evidence missing")
    expected = {(e["chunk_start"],e["chunk_end"],e["layer"]):e for e in source if e["full_rows"]>0}
    seen=set()
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen and event.get("full_rows")==expected[key]["full_rows"]
                and event.get("checks")==2 and type(event.get("max_absolute_error")) in (int,float)
                and event["max_absolute_error"]==0
                and event.get("checks_named")==["canonical_norm_rope_cast","assembled_swa_rows"],
                "cache normalization/RoPE/cast/assembly qualification is not exact")
        seen.add(key)
    count=2*len(expected)
    require(count>0 and seen==set(expected) and control.get("cache_restore_identity_checks")==count,
            "cache restoration numerical coverage/count mismatch")
    return count


def audit_joint_attention(control):
    events,source=control.get("joint_attention_events"),control.get("swa_events")
    require(type(events) is list and type(source) is list
            and control.get("joint_attention_events_sha256")==digest(events),
            "native joint-attention raw call evidence missing")
    expected={(e["chunk_start"],e["chunk_end"],e["layer"]):e for e in source}
    require(len(expected)==len(source),"duplicate SWA source event")
    seen=set();reused=0
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen,"joint attention call coverage mismatch")
        seen.add(key);swa=expected[key]
        for field in ("swa_prefix_rows","current_swa_rows","global_kv_rows","kv_rows"):
            require(type(event.get(field)) is int and event[field]>=0,"invalid joint KV geometry")
        require(event.get("full_rows")==swa["full_rows"]==event["current_swa_rows"]
                and event.get("online_swa_rows")==swa["online_rows"]
                and event.get("reused_swa_rows")==swa["reused_rows"]
                and event["kv_rows"]==event["swa_prefix_rows"]+event["current_swa_rows"]+event["global_kv_rows"]
                and all(event.get(k) is True for k in ("use_swa","has_packed_kv","global_compute_called",
                    "native_joint_attention_called","native_swa_indices_preserved"))
                and type(event.get("is_global_source")) is bool and type(event.get("uses_shared_global_kv")) is bool,
                "cached SWA and online global KV did not enter unchanged native joint attention")
        reused+=event["reused_swa_rows"]
    require(seen==set(expected) and control.get("effective_cache_attention_rows")==reused,
            "joint-attention effective cache consumption differs from actual SWA rows")
    return reused


def audit_native_no_cache(control):
    require(audit_joint_attention(control)==0
            and control.get("joint_attention_events")==[]
            and control.get("canonical_copy_events")==[]
            and control.get("cache_restore_identity_events")==[]
            and control.get("canonical_cache_copy_bytes")==0
            and control.get("cache_restore_identity_checks")==0,
            "native or offline capture phase cannot claim cached KV intervention")


def cuda_mask_qualification(directory, preflight, evidence):
    from gpu_pilot.cuda_mask_qualification import validate_report
    value = read(directory/"cuda_mask_qualification.json", evidence)
    require(value.get("schema") == "ffn-only-cuda-mask-all-ranks-v1" and value.get("passed") is True
            and value.get("execution_configuration_sha256") == preflight["identity"]["execution_configuration_sha256"]
            and value.get("quality_eligible") is False and value.get("timing_valid") is False,
            "CUDA mask qualification not bound to this execution")
    reports = value.get("reports")
    require(type(reports) is list and len(reports) == 8
            and [r.get("rank") for r in reports] == list(range(8)), "all eight CUDA rank checks required")
    for item in reports:
        validate_report(item["report"])
    return value


def qualification(directory, preflight, evidence):
    mask = cuda_mask_qualification(directory, preflight, evidence)
    from gpu_pilot.empty_ep_qualification import validate_all_ranks_report
    empty_ep = read(directory/"empty_ep_qualification.json", evidence)
    validate_all_ranks_report(empty_ep,
        execution_sha256=preflight["identity"]["execution_configuration_sha256"],
        model_sha256=preflight["identity"]["checkpoint_index_sha256"])
    raw = [read(directory/name, evidence) for name in
           ("qualification.native.json", "qualification.all_selected.json")]
    closure = read(directory/"qualification.json", evidence)
    require(closure.get("cuda_mask_qualification_sha256") == mask["sha256"],
            "strict numerical qualification must bind completed CUDA mask checks")
    require(closure.get("empty_ep_qualification_sha256") == empty_ep["sha256"],
            "strict numerical qualification must bind current native empty-EP proof")
    require(closure.get("schema") == "ffn-only-qualification-v1" and closure.get("passed") is True
            and closure.get("reference_sha256") == raw[0]["sha256"]
            and closure.get("candidate_sha256") == raw[1]["sha256"]
            and closure.get("qualification_input_tokens") == 16384
            and closure.get("identity_check_count") == 80
            and closure.get("full_256k_numerical_parity_claim") is False
            and closure.get("old_qualification_reused") is False, "qualification closure mismatch")
    fingerprints = []
    for phase, mode, name in zip(raw, ("qualified_native", "all_selected"),
                                 ("qualification.native","qualification.all_selected")):
        empty_ep_binding(phase, closure)
        require(phase.get("schema") == PHASE_SCHEMA and phase.get("mode") == mode
                and phase.get("index") == 0 and phase.get("arm") == "qualification"
                and phase.get("name") == name and phase.get("repeat") is None
                and phase.get("timing_valid") is False and phase.get("quality_eligible") is False,
                "foreign qualification phase")
        binding(phase, preflight)
        record, control = phase["record"], phase["controller"]
        if mode == "qualified_native":
            audit_native_no_cache(control)
        expected_input=preflight.get("qualification_input_ids_sha256")
        validate_digest(expected_input,"qualification manifest prefix")
        require(record.get("success") is True and record.get("rank") == 0
                and phase.get("input_ids_sha256")==record.get("input_ids_sha256")==control.get("input_ids_sha256")==expected_input
                and control.get("full_input_tokens")==16384
                and record.get("phase")==control.get("phase")==name and control.get("mode")==mode
                and record.get("natural_EOS") is False
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
    checks=projection_identity(raw[1]["controller"])
    require(closure.get("projection_identity_checks")==checks,"split qualification closure mismatch")
    require(closure.get("cache_restore_identity_checks")==cache_restore_identity(raw[1]["controller"]),
            "cache restoration qualification closure mismatch")
    require(audit_joint_attention(raw[1]["controller"])==0,"all-selected qualification cannot reuse cached rows")
    return closure


def empty_ep_binding(phase, qualification):
    proof = qualification.get("empty_ep_qualification_sha256")
    validate_digest(proof, "native empty-EP qualification")
    require(phase.get("empty_ep_qualification_sha256") == proof
            and phase.get("controller", {}).get("empty_ep_qualification_sha256") == proof,
            "phase/controller missing or foreign native empty-EP capability")


def binding(phase, preflight):
    require(phase.get("manifest_sha256") == preflight["manifest_sha256"]
            and phase.get("swa_reuse_qonly_protocol_sha256") == preflight["protocol"]["sha256"]
            and phase.get("execution_configuration_sha256") ==
                preflight["identity"]["execution_configuration_sha256"], "phase execution identity drift")


def _positions(value, start, end, *, full_required=False):
    require(type(value) is list and all(type(p) is int and start <= p < end for p in value)
            and value == sorted(set(value)), "native row positions must be unique, sorted, absolute and in chunk")
    if full_required:
        require(value == list(range(start,end)), "encoder native row layout unexpectedly compacted")
    return value


def audit_controller(control, row, phase, spans):
    require(control.get("schema") == "swa-reuse-qonly-ffn15-controller-v1"
            and control.get("mode") == phase["mode"] and control.get("phase") == phase["name"]
            and control.get("input_ids_sha256") == row["input_ids_sha256"]
            and control.get("full_input_tokens") == row["input_tokens"]
            and control.get("first_six_full") is True
            and control.get("actual_full_row_native_mhc") is True
            and control.get("all_query_attention_native") is True
            and control.get("yoco_layout_preserved") is True
            and control.get("formal_admission_unchanged") is True
            and control.get("swa_reuse_policy")=="all-cached-document-rows-no-boundary-repair"
            and control.get("document_boundary_online_tokens")==0
            and control.get("online_swa_scope")=="non-document-template-question-only"
            and control.get("decode_native") is True and control.get("joint_attention_native") is True
            and control.get("global_kv_native_online") is True
            and control.get("not_model_flops_fraction") is True, "controller identity/design safeguards differ")
    selection = control.get("selection")
    swa = control.get("swa_events")
    layouts = control.get("ffn_layouts")
    adapter = control.get("adapter", {})
    counts, events = adapter.get("counts", {}), adapter.get("events")
    require(type(selection) is list and type(swa) is list and type(layouts) is list
            and type(events) is list
            and adapter.get("schema") == "kv-cache-engram-ffn-only-adapter-v1"
            and all(adapter.get(k) is True for k in ("decoder_forward_unchanged", "attention_forward_unchanged",
                "native_mhc_unchanged", "native_yoco_layout_unchanged", "all_selected_delegates_native"))
            and all(adapter.get(k) is False for k in ("empty_selection_skips_ep_collective",
                "cached_hidden_or_ffn_output", "accuracy_guarantee", "speedup_claim")),
            "independent SWA/FFN evidence missing")
    if phase["arm"] == "recomputed":
        audit_native_no_cache(control)
        require(not selection and not swa and not layouts and not events
                and counts.get("full_rows") == 0 and counts.get("selected_rows") == 0,
                "baseline must not execute sparse FFN or cached SWA")
        return dict(ffn_full_rows=0,ffn_selected_rows=0,ffn_bypassed_rows=0,ffn_bypass_fraction=0.,
                    swa_full_rows=0,swa_online_rows=0,swa_reused_rows=0,swa_reuse_fraction=0.,
                    optional_chunks=0,actual_partial_chunks=0)
    require(phase["arm"] == "ffn15", "unknown candidate")
    chunks=[(s,min(s+8192,row["input_tokens"])) for s in range(0,row["input_tokens"],8192)]
    require([(m.get("start"),m.get("end")) for m in selection] == chunks,
            "every exact native chunk needs one frozen early selection")
    masks={}
    for m in selection:
        start,end=m["start"],m["end"]
        ids,mandatory,optional=mask_reference(m.get("mean_adjacent_relative_l2"),start,end,spans,.15)
        transitions=m.get("adjacent_transition_relative_l2")
        require(type(transitions) is list and len(transitions)==3
                and all(type(v) is list and len(v)==end-start for v in transitions),
                "three adjacent per-token transitions required")
        for i,mean in enumerate(m["mean_adjacent_relative_l2"]):
            values=[number(v[i],"adjacent transition") for v in transitions]
            require(math.isclose(mean,math.fsum(values)/3,rel_tol=1e-12,abs_tol=1e-14),
                    "stored per-token mean differs from adjacent transitions")
        require(m.get("selected_local_rows")==ids and m.get("selected_rows_sha256")==digest(ids)
                and m.get("selected_rows")==len(ids) and m.get("full_rows")==end-start
                and m.get("mandatory_rows")==mandatory and m.get("optional_rows")==optional,
                "FFN mask differs from independent per-token top15 ranking")
        validate_digest(m.get("plan_sha256"),"selection policy binding")
        masks[(start,end)]=m
    expected={(s,e,l) for s,e in chunks for l in FFN_LAYERS}
    by_layout={}
    compact={}
    for layout in layouts:
        key=(layout.get("chunk_start"),layout.get("chunk_end"),layout.get("layer"))
        require(key in expected and key not in by_layout,"foreign/duplicate FFN layout")
        start,end,layer=key
        positions=_positions(layout.get("current_positions"),start,end,full_required=layer<20)
        if layer>=20:
            old=compact.setdefault((start,end),positions)
            require(positions==old,"native YOCO query layout changed between upper layers")
        selected=set(masks[(start,end)]["selected_local_rows"])
        ids=[i for i,p in enumerate(positions) if p-start in selected]
        require(layout.get("selected_local_rows")==ids and layout.get("full_rows")==len(positions)
                and layout.get("selected_rows")==len(ids),"FFN selected rows differ from surviving native position intersection")
        validate_digest(layout.get("plan_sha256"),"FFN layout plan")
        by_layout[key]=layout
    require(set(by_layout)==expected,"FFN layer/chunk coverage incomplete")
    seen=set()
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen,"foreign/duplicate FFN event")
        seen.add(key); layout=by_layout[key]
        require(event.get("request_id")==phase["record"]["request_id"] and event.get("phase")==phase["name"]
                and event.get("full_rows")==layout["full_rows"] and event.get("selected_rows")==layout["selected_rows"]
                and event.get("plan_sha256")==layout["plan_sha256"]
                and event.get("current_layout")=="native_current_rows"
                and event.get("inactive_ffn_output")=="zero"
                and event.get("all_selected_native") is (layout["selected_rows"]==layout["full_rows"])
                and event.get("empty_native_collective") is (layout["selected_rows"]==0 and layout["full_rows"]>0),
                "FFN execution event not bound to actual native layout")
    require(seen==expected,"FFN execution coverage incomplete")
    full=sum(v["full_rows"] for v in layouts); selected=sum(v["selected_rows"] for v in layouts)
    require(counts.get("full_rows")==full and counts.get("selected_rows")==selected
            and counts.get("inactive_ffn_zero_rows")==full-selected,"FFN execution row accounting mismatch")
    require(counts.get("empty_native_mlp_calls") == sum(
        v["selected_rows"] == 0 and v["full_rows"] > 0 for v in layouts),
        "empty native EP call accounting mismatch")
    seen=set(); swa_full=swa_online=0
    for event in swa:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen,"foreign/duplicate SWA event")
        seen.add(key); start,end,layer=key
        positions=_positions(event.get("current_positions"),start,end,full_required=layer<20)
        require(positions==by_layout[key]["current_positions"],"SWA and FFN use different native query layout")
        online=[]
        for i,pos in enumerate(positions):
            doc=next(((lo,hi) for lo,hi,_ in spans if lo<=pos<hi),None)
            if doc is None:
                online.append(i)
        require(event.get("online_local_rows")==online and event.get("full_rows")==len(positions)
                and event.get("online_rows")==len(online) and event.get("reused_rows")==len(positions)-len(online)
                and event.get("compact") is (layer>=20),
                "SWA recomputation must be exactly non-document misses; document prefix repair is forbidden")
        swa_full+=len(positions); swa_online+=len(online)
    require(seen==expected,"SWA layer/chunk coverage incomplete")
    require(audit_joint_attention(control)==swa_full-swa_online,"effective cache reuse accounting differs")
    copy_events=control.get("canonical_copy_events")
    require(type(copy_events) is list,"canonical copy evidence required")
    nonempty={k:v for k,v in by_layout.items() if v["full_rows"]>0}
    copy_seen=set();copy_bytes=0
    for event in copy_events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in nonempty and key not in copy_seen,"canonical copy coverage differs")
        copy_seen.add(key)
        copied=sum(max(0,min(key[1],hi)-max(key[0],lo)) for lo,hi,_ in spans)
        require(event.get("copied_document_rows")==copied
                and event.get("canonical_bf16_bytes")==copied*512*2
                and event.get("restored_current_rows")==nonempty[key]["full_rows"],
                "canonical cache copy byte accounting differs")
        copy_bytes+=event["canonical_bf16_bytes"]
    require(copy_seen==set(nonempty) and control.get("canonical_cache_copy_bytes")==copy_bytes
            and control.get("bytes_are_canonical_copies_not_measured_device_bandwidth") is True,
            "canonical copies missing or misrepresented as measured bandwidth")
    return dict(ffn_full_rows=full,ffn_selected_rows=selected,ffn_bypassed_rows=full-selected,
        ffn_bypass_fraction=(full-selected)/full if full else 0.,
        swa_full_rows=swa_full,swa_online_rows=swa_online,swa_reused_rows=swa_full-swa_online,
        swa_reuse_fraction=(swa_full-swa_online)/swa_full if swa_full else 0.,
        effective_cache_attention_rows=swa_full-swa_online, canonical_cache_copy_bytes=copy_bytes,
        cache_bytes_not_measured_device_bandwidth=True,
        optional_chunks=sum(m["optional_rows"]>0 for m in selection),
        actual_partial_chunks=sum(0<m["selected_rows"]<m["full_rows"] for m in selection))


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
            and record.get("request_id") == f"swa-reuse-qonly:{row['index']}:{phase['name']}:{row['sample_id']}",
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
    row=next(item for item in preflight["audit_cases"] if item["index"]==2)
    closure=read(directory/"partial_smoke.json",evidence)
    require(closure.get("schema")=="swa-reuse-qonly-partial-smoke-v1" and closure.get("passed") is True
            and closure.get("index")==2 and closure.get("qualification_sha256")==qual["sha256"]
            and closure.get("input_ids_sha256")==row["input_ids_sha256"]
            and closure.get("input_tokens")==row["input_tokens"]
            and closure.get("quality_eligible") is False and closure.get("timing_valid") is False
            and closure.get("accuracy_guarantee") is False
            and closure.get("full_input_original_eight_documents") is True
            and closure.get("independent_swa_and_ffn_paths_exercised") is True
            and closure.get("output_tokens_per_arm")==1
            and set(closure.get("phase_sha256",{}))=={"ffn15"},"partial smoke closure incomplete")
    name="case2.ffn15_partial_smoke"
    phase=read(directory/(name+".json"),evidence)
    empty_ep_binding(phase, qual)
    require(phase.get("name")==name and phase.get("repeat") is None
            and phase["sha256"]==closure["phase_sha256"]["ffn15"],"partial smoke raw mismatch")
    rows=audit_phase(phase,row,preflight,arm="ffn15",timing=False,quality=False,fixed_tokens=1)
    require(rows["ffn_bypassed_rows"]>0 and rows["swa_reused_rows"]>0
            and rows["actual_partial_chunks"]>0,"smoke did not exercise independent partial paths")
    return closure,dict(passed=True,index=2,input_tokens=row["input_tokens"],actual_rows=rows,
        closure_sha256=closure["sha256"],fixed_output_tokens=1,quality_eligible=False,timing_valid=False,
        numerical_identity_claim=False,all_declared_layers_and_native_layouts_audited=True)


def audit_case(directory, row, preflight, qual, smoke, evidence):
    index = row["index"]
    pair = read(directory/f"case{index}.paired.json", evidence)
    require(pair.get("schema") == "swa-reuse-qonly-case-paired-v1"
            and pair.get("qualification_sha256") == qual["sha256"]
            and pair.get("partial_smoke_sha256") == smoke["sha256"]
            and all(pair.get(key) == row[key] for key in
                    ("index", "dataset", "sample_id", "references", "question", "input_tokens", "input_ids_sha256"))
            and pair.get("ffn_layers") == list(FFN_LAYERS)
            and pair.get("offline_capture_layers") == list(CAPTURE_LAYERS)
            and pair.get("all_negative_results_retained") is True
            and pair.get("exploratory_previously_exposed_inputs") is True, "two-arm closure identity mismatch")
    quality, timing, masks = {}, {arm: [] for arm in ARMS}, {}
    require(set(pair.get("quality", {})) == set(ARMS), "both quality arms required")
    for arm in ARMS:
        phase = read(directory/f"case{index}.{arm}_quality.json", evidence)
        empty_ep_binding(phase, qual)
        verify_reference(pair["quality"][arm], directory/f"case{index}.{arm}_quality.json", phase, evidence)
        require(phase.get("name") == f"case{index}.{arm}_quality" and phase.get("repeat") is None,
                "quality phase filename/repeat mismatch")
        masks[arm] = audit_phase(phase, row, preflight, arm=arm, timing=False, quality=True)
        rec = phase["record"]
        quality[arm] = {key: rec[key] for key in ("text", "answer_em", "answer_token_f1", "truncated", "finish_reason")}
        quality[arm]["generated_tokens"] = len(rec["output_ids"])
        warmup = read(directory/f"case{index}.{arm}_warmup.json", evidence)
        empty_ep_binding(warmup, qual)
        require(warmup.get("name") == f"case{index}.{arm}_warmup" and warmup.get("repeat") is None,
                "warmup phase filename/repeat mismatch")
        audit_phase(warmup, row, preflight, arm=arm, timing=False, quality=False)
    expected = [(repeat, ARMS[(index+repeat+slot)%2]) for repeat in range(3) for slot in range(2)]
    phases = pair.get("timing_phases")
    require(type(phases) is list and [(p.get("repeat"), p.get("arm")) for p in phases] == expected,
            "three rotating same-case paired C1 rounds required")
    for recorded, (repeat, arm) in zip(phases, expected):
        phase = read(directory/f"case{index}.{arm}_measure{repeat}.json", evidence)
        empty_ep_binding(phase, qual)
        verify_reference(recorded, directory/f"case{index}.{arm}_measure{repeat}.json", phase, evidence,
                         extras=("repeat","arm"))
        require(phase.get("name") == f"case{index}.{arm}_measure{repeat}"
                and phase.get("repeat") == repeat and phase.get("arm") == arm, "formal timing schedule mismatch")
        audit_phase(phase, row, preflight, arm=arm, timing=True, quality=False)
        timing[arm].append(phase["record"]["online_ttft_s"])
    builds = []
    for doc_index, (_, _, doc_key) in enumerate(documents(row)):
        phase = read(directory/f"case{index}.offline_doc{doc_index}.json", evidence)
        empty_ep_binding(phase, qual)
        binding(phase, preflight)
        require(phase.get("arm") == "offline" and phase.get("mode") == "capture_independent"
                and phase.get("input_ids_sha256") == doc_key, "offline document phase mismatch")
        require(phase["controller"].get("offline_yoco_full_layout") is True
                and phase["controller"].get("model_boundary_restored") is True,
                "offline all-layer capture must restore native model YOCO before online runs")
        audit_native_no_cache(phase["controller"])
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


def summarize_cases(cases):
    """Same closed cases only; two empirical points, never a global frontier."""
    if not cases:
        return None
    arms = {}
    for arm in ARMS:
        samples = [sample for case in cases for sample in case["online_ttft_samples_s"][arm]]
        arms[arm] = dict(n=len(cases),
            em_percent=100*sum(case["quality"][arm]["answer_em"] for case in cases)/len(cases),
            f1_percent=100*sum(case["quality"][arm]["answer_token_f1"] for case in cases)/len(cases),
            online_ttft_p50_s=statistics.median(samples), online_ttft_samples_s=samples,
            truncated_count=sum(case["quality"][arm]["truncated"] for case in cases))
    baseline, candidate = arms["recomputed"], arms["ffn15"]
    compare = dict(em_loss_pp=baseline["em_percent"]-candidate["em_percent"],
        f1_loss_pp=baseline["f1_percent"]-candidate["f1_percent"],
        ttft_reduction_s=baseline["online_ttft_p50_s"]-candidate["online_ttft_p50_s"],
        ttft_reduction_percent=100*(1-candidate["online_ttft_p50_s"]/baseline["online_ttft_p50_s"]),
        ttft_speedup=baseline["online_ttft_p50_s"]/candidate["online_ttft_p50_s"],
        correct_to_wrong=sum(case["quality"]["recomputed"]["answer_em"] == 1 and
                             case["quality"]["ffn15"]["answer_em"] == 0 for case in cases),
        wrong_to_correct=sum(case["quality"]["recomputed"]["answer_em"] == 0 and
                             case["quality"]["ffn15"]["answer_em"] == 1 for case in cases))
    frontiers = {}
    for metric in ("em_percent", "f1_percent"):
        def dominates(a, b):
            return (arms[a][metric] >= arms[b][metric]
                    and arms[a]["online_ttft_p50_s"] <= arms[b]["online_ttft_p50_s"]
                    and (arms[a][metric] > arms[b][metric]
                         or arms[a]["online_ttft_p50_s"] < arms[b]["online_ttft_p50_s"]))
        frontiers[metric+"_vs_ttft"] = [a for a in ARMS if not any(dominates(b,a) for b in ARMS if a != b)]
    return dict(n=len(cases), arms=arms, comparison=compare, empirical_frontiers=frontiers,
                point_set=list(ARMS), global_frontier_claim=False, statistical_significance_claim=False)


def audit(directory, case_index=None):
    """Audit new closed cases without rereading prior 10-GiB offline banks.

    ``case_index`` deliberately emits a SUBSET result and never a final complete
    run claim, even if run.json already exists. The final audit without this
    option still checks every case, every bank and the complete run closure.
    """
    directory = Path(directory)
    require(case_index is None or type(case_index) is int and case_index in CASE_INDICES,
            "requested subset case is not registered")
    evidence, errors, results, closures = {}, [], [], []
    preflight = read(directory/"preflight.json", evidence)
    require(preflight.get("schema") == RUN_SCHEMA and preflight.get("protocol") == swa_reuse_qonly_protocol()
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
                and type(run.get("cases")) is list and len(run["cases"]) == len(closures),
                "final run does not close all four independently audited cases")
        for ref,closure in zip(run["cases"],closures):
            require(ref.get("index") == closure["index"], "final case order mismatch")
            verify_reference(ref, directory/f"case{closure['index']}.paired.json", closure,evidence,extras=("index",))
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
        summary=summarize_cases(results),
        by_dataset={name:summarize_cases([c for c in results if c["dataset"]==name])
                    for name in ("hotpotqa", "2wikimqa")},
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
    if sys.platform.startswith("linux"):
        import resource
        soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        bound=96*1024**3
        if soft != resource.RLIM_INFINITY:bound=min(bound,soft)
        if hard != resource.RLIM_INFINITY:bound=min(bound,hard)
        resource.setrlimit(resource.RLIMIT_AS,(bound,hard))
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
