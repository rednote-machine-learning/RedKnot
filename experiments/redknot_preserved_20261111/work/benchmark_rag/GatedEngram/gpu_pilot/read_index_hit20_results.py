"""Independent CPU-only audit of sealed index-hit20 B pilot evidence.

Only closed two-arm quality + three C1 repetitions enter scores. This module
does not import the controller, runner, Torch, native code or any GPU runtime.
It reconstructs token selection from full integer hit vectors and actual native
layouts. Sparse qualification additionally replays the original top-k arrays;
formal count vectors come from that frozen qualified selector, not a raw-index
rescan claim. Producer selected-row totals and median summaries are not trusted.
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
from gpu_pilot.index_hit20_protocol import ARMS, CAPTURE_LAYERS, FFN_LAYERS, index_hit20_protocol

SCHEMA = "index-hit20-independent-readout-v1"
PHASE_SCHEMA = "kv-cache-engram-index-hit20-phase-v1"
RUN_SCHEMA = "kv-cache-engram-index-hit20-pilot-run-v1"
CASE_INDICES = (2, 3, 4, 5)
FRACTIONS = {"recomputed": None, "index20": .20}
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


def mask_reference(hit_counts, positions, start, end, spans, fraction=.20):
    """Independent integer hit ranking over actual native query rows."""
    require(type(start) is int and type(end) is int and 0 <= start < end
            and fraction == .20, "invalid current chunk/fraction")
    _positions(positions,start,end)
    require(type(hit_counts) is list and len(hit_counts)==len(positions)
            and all(type(v) is int and v>=0 for v in hit_counts),
            "complete nonnegative integer hit vector required")
    optional=[i for i,pos in enumerate(positions) if any(lo<=pos<hi for lo,hi,_ in spans)]
    mandatory=set(range(len(positions)))-set(optional)
    ranked=sorted(optional,key=lambda i:(-hit_counts[i],positions[i]))
    keep=(len(optional)+4)//5
    return sorted(mandatory|set(ranked[:keep])),len(mandatory),len(optional)


def sparse_identity(reference, candidate, row, qualification_sha256):
    """Strict same-policy sparse parity; never an equivalence-to-full claim."""
    validate_digest(qualification_sha256,"native all-selected qualification")
    for phase,mode in ((reference,"reference_sparse"),(candidate,"sensitive")):
        sealed(phase)
        require(phase.get("schema")==PHASE_SCHEMA and phase.get("mode")==mode
                and phase.get("arm")=="index20" and phase.get("fraction")==.20
                and phase.get("quality_eligible") is False and phase.get("timing_valid") is False
                and phase.get("index")==row["index"]
                and phase.get("input_ids_sha256")==row["input_ids_sha256"],
                "sparse reference/candidate phase identity mismatch")
        record=phase["record"]; control=phase["controller"]
        require(control.get("raw_index_oracle_checked") is True,
                "sparse implementation qualification omitted CPU raw-index oracle")
        audit_controller(control,row,phase,documents(row))
        require(record.get("success") is True and record.get("rank")==0
                and record.get("input_tokens")==row["input_tokens"]
                and record.get("input_ids_sha256")==row["input_ids_sha256"]
                and len(record.get("output_ids",[]))==1 and record.get("natural_EOS") is False
                and record.get("output_ids_sha256")==digest(record["output_ids"]),
                "sparse qualification output/input binding mismatch")
        fingerprints=control.get("sparse_identity_fingerprints")
        chunks=[(s,min(s+8192,row["input_tokens"])) for s in range(0,row["input_tokens"],8192)]
        expected={(s,e,l) for s,e in chunks for l in range(40)}
        require(type(fingerprints) is list and len(fingerprints)==len(expected)
                and control.get("sparse_identity_check_count")==len(expected)
                and control.get("sparse_identity_fingerprints_sha256")==digest(fingerprints),
                "full sparse layer/chunk fingerprints required")
        seen=set()
        for item in fingerprints:
            key=tuple(item.get("key",[])); value=item.get("value",{})
            require(key in expected and key not in seen
                    and set(value)=={"output","lazy"}
                    and set(value["lazy"])=={"residual","post_layer_mix","comb_res_mix","pre_layer_mix"},
                    "sparse output/all4 mHC field coverage differs")
            seen.add(key)
            require(value["output"] is not None,"sparse decoder output absent")
            _fingerprint(value["output"])
            for v in value["lazy"].values(): _fingerprint(v)
    for name in ("manifest_sha256","execution_configuration_sha256","index_hit20_protocol_sha256",
                 "empty_ep_qualification_sha256"):
        require(reference.get(name)==candidate.get(name),"sparse qualification identity changed")
    require(candidate.get("verify_sparse") is True
            and candidate["controller"].get("sparse_identity_exact") is True
            and reference["controller"]["sparse_identity_fingerprints"]
                ==candidate["controller"]["sparse_identity_fingerprints"]
            and reference["record"]["output_ids"]==candidate["record"]["output_ids"],
            "sparse same-policy reference byte parity failed")
    fields=("start","end","layer","current_positions","hitcounts","selected_local_rows")
    require([{k:m[k] for k in fields} for m in reference["controller"]["selection"]]
            ==[{k:m[k] for k in fields} for m in candidate["controller"]["selection"]],
            "sparse reference and production masks differ")
    body=dict(schema="index-hit20-sparse-qualification-v1",passed=True,
        index=row["index"],input_tokens=row["input_tokens"],input_ids_sha256=row["input_ids_sha256"],
        qualification_sha256=qualification_sha256,reference_sha256=reference["sha256"],
        candidate_sha256=candidate["sha256"],
        identity_check_count=candidate["controller"]["sparse_identity_check_count"],
        identity_fingerprints_sha256=candidate["controller"]["sparse_identity_fingerprints_sha256"],
        quality_eligible=False,timing_valid=False,equivalence_to_full_model_claim=False,
        exact_same_policy_reference=True)
    return dict(body,sha256=digest(body))


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
        names=["split_q_a","split_swa","split_main_q_b","packed_gather_swa127"]
        if key[2] in (8,14,20,24,28,32,36): names.append("split_index_q_b")
        require(key in nonempty and key not in seen and event.get("full_rows")==by_key[key]["full_rows"]
                and event.get("checks")==len(names) and type(event.get("max_absolute_error")) in (int,float)
                and event["max_absolute_error"]==0
                and event.get("checks_named")==names,
                "split Q/KV/gather exact-zero checks missing or failed")
        seen.add(key)
    count=sum(5 if key[2] in (8,14,20,24,28,32,36) else 4 for key in nonempty)
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
    """Audit actual selected query execution, not merely cached-row counts."""
    source=control.get("swa_events");events=control.get("attention_events")
    require(type(source) is list and type(events) is list,"SWA/attention raw accounting absent")
    expected={(e["chunk_start"],e["chunk_end"],e["layer"]):e for e in source}
    require(len(expected)==len(source),"duplicate SWA event")
    seen=set();reused=0
    dense=control.get("mode")=="reference_sparse"
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen,"attention coverage differs")
        seen.add(key);swa=expected[key];n=swa["full_rows"];k=swa["online_rows"];layer=key[2]
        local=layer in (8,14,20,24,28,32,36)
        expected_actual=n if dense else k
        require(event.get("full_rows")==n and event.get("selected_rows")==k
                and event.get("main_q_a_rows")== (n if local or dense else k)
                and event.get("index_q_rows")== (n if local else 0)
                and event.get("main_q_b_rows")==expected_actual
                and event.get("joint_attention_rows")==expected_actual
                and event.get("output_projection_rows")==expected_actual
                and event.get("inactive_attention_zero_rows")==n-k
                and event.get("reference_dense") is dense
                and event.get("global_compute_called") is True
                and event.get("native_joint_attention") is True
                and event.get("is_global_source") is (layer in (8,14,20)),
                "actual Q/attention/O execution differs from B selection")
        require(type(event.get("global_kv_rows")) is int and event["global_kv_rows"]>=0,
                "complete global KV geometry missing")
        validate_digest(event.get("plan_sha256"),"attention plan")
        reused+=swa["reused_rows"]
    require(seen==set(expected),"attention layers missing")
    return reused


def audit_native_no_cache(control):
    require(all(control.get(k)==[] for k in
                ("selection","index_groups","swa_events","ffn_layouts","attention_events",
                 "projection_identity_events","cache_restore_identity_events"))
            and control.get("projection_identity_checks")==0
            and control.get("cache_restore_identity_checks")==0
            and control.get("adapter",{}).get("events")==[],
            "native/offline phase cannot claim sparse or cached intervention")


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
    from gpu_pilot.index_hit20_empty_ep_qualification import validate_all_ranks_report
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
    require(closure.get("schema") == "index-hit20-qualification-v1" and closure.get("passed") is True
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
            and phase.get("index_hit20_protocol_sha256") == preflight["protocol"]["sha256"]
            and phase.get("execution_configuration_sha256") ==
                preflight["identity"]["execution_configuration_sha256"], "phase execution identity drift")


def _positions(value, start, end, *, full_required=False):
    require(type(value) is list and all(type(p) is int and start <= p < end for p in value)
            and value == sorted(set(value)), "native row positions must be unique, sorted, absolute and in chunk")
    if full_required:
        require(value == list(range(start,end)), "encoder native row layout unexpectedly compacted")
    return value


def audit_controller(control, row, phase, spans):
    require(control.get("schema")=="index-hit20-controller-v1"
            and control.get("mode")==phase["mode"] and control.get("phase")==phase["name"]
            and control.get("input_ids_sha256")==row["input_ids_sha256"]
            and control.get("full_input_tokens")==row["input_tokens"]
            and control.get("ffn_layers")==list(FFN_LAYERS)
            and all(control.get(k) is True for k in ("first_three_full","actual_full_row_native_mhc",
                "yoco_layout_preserved","formal_admission_unchanged","decode_native",
                "global_kv_native_online","not_model_flops_fraction","dynamic_mask_sync_in_online_timing"))
            and control.get("history_kv_rewritten") is False
            and control.get("selected_tokens_forced_through_yoco") is False
            and control.get("formal_raw_index_replay_claim") is False,
            "controller scope or B design safeguards differ")
    selection=control.get("selection");swa=control.get("swa_events")
    layouts=control.get("ffn_layouts");groups=control.get("index_groups")
    adapter=control.get("adapter",{});counts=adapter.get("counts",{});events=adapter.get("events")
    require(all(type(v) is list for v in (selection,swa,layouts,groups,events))
            and adapter.get("schema")=="kv-cache-engram-index-hit20-adapter-v1"
            and all(adapter.get(k) is True for k in ("decoder_forward_unchanged","attention_forward_unchanged",
                "native_mhc_unchanged","native_yoco_layout_unchanged","all_selected_delegates_native"))
            and all(adapter.get(k) is False for k in ("empty_selection_skips_ep_collective",
                "cached_hidden_or_ffn_output","accuracy_guarantee","speedup_claim")),
            "actual execution adapter evidence missing")
    if phase["arm"]=="recomputed":
        audit_native_no_cache(control)
        require(counts.get("full_rows")==0 and counts.get("selected_rows")==0,
                "baseline must not use sparse controller")
        return dict(ffn_full_rows=0,ffn_selected_rows=0,ffn_bypassed_rows=0,ffn_bypass_fraction=0.,
            swa_full_rows=0,swa_online_rows=0,swa_reused_rows=0,swa_reuse_fraction=0.,
            attention_full_rows=0,attention_selected_rows=0,attention_bypassed_rows=0,
            optional_chunks=0,actual_partial_chunks=0)
    require(phase["arm"]=="index20" and phase["mode"] in ("sensitive","reference_sparse"),
            "unknown candidate policy")
    chunks=[(s,min(s+8192,row["input_tokens"])) for s in range(0,row["input_tokens"],8192)]
    expected={(s,e,l) for s,e in chunks for l in FFN_LAYERS}
    by_group={}
    for group in groups:
        ident=group.get("group_id")
        require(type(ident) is str and re.fullmatch(r"g[0-9]+",ident)
                and ident not in by_group and group.get("source_layer") in (2,8,14,20,24,28,32,36)
                and type(group.get("rows")) is int and group["rows"]>=0
                and group.get("columns")==512 and group.get("request_kv_offset")==0
                and type(group.get("global_kv_rows")) is int and group["global_kv_rows"]>=0
                and group.get("compression_ratio") in (1,2),
                "index group identity/geometry invalid")
        positions=_positions(group.get("query_positions"),group["chunk_start"],group["chunk_end"])
        require(len(positions)==group["rows"] and type(group.get("hitcounts")) is list
                and len(group["hitcounts"])==len(positions)
                and all(type(v) is int and 0<=v<=group["rows"] for v in group["hitcounts"]),
                "index frequency vector incomplete")
        by_group[ident]=group
    masks={};compact={};used=set()
    for m in selection:
        key=(m.get("start"),m.get("end"),m.get("layer"))
        require(key in expected and key not in masks,"selection layer/chunk coverage differs")
        start,end,layer=key
        positions=_positions(m.get("current_positions"),start,end,full_required=layer<20)
        if layer>=20:
            old=compact.setdefault((start,end),positions)
            require(old==positions,"native YOCO survivor layout changed")
        group=by_group.get(m.get("index_group_id"))
        require(group is not None,"selection has no native index group")
        used.add(group["group_id"])
        source=max(i for i in (2,8,14,20,24,28,32,36) if i<=layer)
        require((group["chunk_start"],group["chunk_end"])==(start,end)
                and group["source_layer"]==source and group["query_positions"]==positions
                and group["compression_ratio"]==(2 if layer<20 else 1)
                and m.get("compression_ratio")==group["compression_ratio"]
                and m.get("hitcounts")==group["hitcounts"],
                "selection detached from native source index or compressed request layout")
        ids,mandatory,optional=mask_reference(m["hitcounts"],positions,start,end,spans)
        optional_ids=[i for i,p in enumerate(positions) if any(lo<=p<hi for lo,hi,_ in spans)]
        require(m.get("selected_local_rows")==ids and m.get("selected_rows_sha256")==digest(ids)
                and m.get("selected_rows")==len(ids) and m.get("full_rows")==len(positions)
                and m.get("mandatory_rows")==mandatory and m.get("optional_rows")==optional
                and m.get("zero_hit_document_rows")==sum(m["hitcounts"][i]==0 for i in optional_ids)
                and m.get("selected_zero_hit_document_rows")==sum(m["hitcounts"][i]==0 for i in ids if i in set(optional_ids)),
                "selection differs from independent hit-frequency top20 rank")
        expected_plan=digest(dict(input_sha256=row["input_ids_sha256"],start=start,end=end,
            layer=layer,group=group["group_id"],mode=phase["mode"],fraction=.20,
            policy="raw-index-hit-frequency-current-native-row-top20-position-tie"))
        require(m.get("plan_sha256")==expected_plan,"selection plan identity changed")
        masks[key]=m
    require(set(masks)==expected and used==set(by_group),"missing or extraneous selection/index group")
    by_layout={}
    for layout in layouts:
        key=(layout.get("chunk_start"),layout.get("chunk_end"),layout.get("layer"))
        require(key in expected and key not in by_layout,"FFN layout duplicated or foreign")
        m=masks[key]
        require(all(layout.get(k)==m[k] for k in ("current_positions","selected_local_rows",
                "full_rows","selected_rows","plan_sha256")),"FFN differs from index-selected query rows")
        by_layout[key]=layout
    require(set(by_layout)==expected,"FFN layouts incomplete")
    dense=phase["mode"]=="reference_sparse"
    seen=set();actual_mlp=0
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen,"FFN execution event coverage differs")
        seen.add(key);m=masks[key];full=m["full_rows"];selected=m["selected_rows"]
        actual=full if dense else selected
        require(event.get("request_id")==phase["record"]["request_id"] and event.get("phase")==phase["name"]
                and event.get("full_rows")==full and event.get("selected_rows")==selected
                and event.get("actual_mlp_rows")==actual and event.get("reference_dense") is dense
                and event.get("plan_sha256")==m["plan_sha256"]
                and event.get("current_layout")=="native_current_rows"
                and event.get("inactive_ffn_output")=="zero"
                and event.get("all_selected_native") is (selected==full)
                and event.get("empty_native_collective") is (selected==0 and full>0),
                "FFN execution not bound to B plan")
        actual_mlp+=actual
    full=sum(m["full_rows"] for m in masks.values());selected=sum(m["selected_rows"] for m in masks.values())
    require(seen==expected and counts.get("full_rows")==full and counts.get("selected_rows")==selected
            and counts.get("actual_mlp_rows")==actual_mlp
            and counts.get("inactive_ffn_zero_rows")==full-selected
            and counts.get("reference_dense_mlp_calls")== (len(expected) if dense else 0)
            and counts.get("empty_native_mlp_calls")== (0 if dense else sum(m["selected_rows"]==0 and m["full_rows"]>0 for m in masks.values())),
            "FFN row accounting differs")
    seen=set()
    for event in swa:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen,"SWA execution event coverage differs")
        seen.add(key);m=masks[key]
        require(event.get("current_positions")==m["current_positions"]
                and event.get("online_local_rows")==m["selected_local_rows"]
                and event.get("full_rows")==m["full_rows"] and event.get("online_rows")==m["selected_rows"]
                and event.get("reused_rows")==m["full_rows"]-m["selected_rows"]
                and event.get("compact") is (key[2]>=20),
                "SWA online/reused rows do not match B selection")
    require(seen==expected and audit_joint_attention(control)==full-selected,
            "SWA/joint attention rows incomplete")
    by_attention={(e["chunk_start"],e["chunk_end"],e["layer"]):e for e in control["attention_events"]}
    require(all(by_attention[k]["plan_sha256"]==m["plan_sha256"] for k,m in masks.items()),
            "attention plan differs from FFN/SWA plan")
    encoder=[m for k,m in masks.items() if k[2]<20];decoder=[m for k,m in masks.items() if k[2]>=20]
    return dict(ffn_full_rows=full,ffn_selected_rows=selected,ffn_bypassed_rows=full-selected,
        ffn_actual_computed_rows=actual_mlp,ffn_bypass_fraction=(full-selected)/full if full else 0.,
        swa_full_rows=full,swa_online_rows=selected,swa_reused_rows=full-selected,
        swa_reuse_fraction=(full-selected)/full if full else 0.,
        attention_full_rows=full,attention_selected_rows=selected,attention_bypassed_rows=full-selected,
        attention_actual_computed_rows=sum(e["joint_attention_rows"] for e in control["attention_events"]),
        optional_chunks=sum(m["optional_rows"]>0 for m in masks.values()),
        actual_partial_chunks=sum(0<m["selected_rows"]<m["full_rows"] for m in masks.values()),
        zero_hit_document_rows=sum(m["zero_hit_document_rows"] for m in masks.values()),
        selected_zero_hit_document_rows=sum(m["selected_zero_hit_document_rows"] for m in masks.values()),
        encoder_rows=sum(m["full_rows"] for m in encoder),encoder_selected_rows=sum(m["selected_rows"] for m in encoder),
        decoder_rows=sum(m["full_rows"] for m in decoder),decoder_selected_rows=sum(m["selected_rows"] for m in decoder),
        formal_raw_topk_independently_rescanned=False,
        frequency_vectors_from_frozen_qualified_selector=True,selection_independently_reranked=True)


def audit_phase(phase, row, preflight, *, arm, timing, quality, fixed_tokens=32):
    sealed(phase)
    binding(phase, preflight)
    require(phase.get("schema") == PHASE_SCHEMA and phase.get("index") == row["index"]
            and phase.get("dataset") == row["dataset"] and phase.get("arm") == arm
            and phase.get("mode") == ("native" if arm == "recomputed" else "sensitive")
            and phase.get("fraction") == FRACTIONS[arm] and phase.get("timing_valid") is timing
            and phase.get("quality_eligible") is quality and phase.get("input_ids_sha256") == row["input_ids_sha256"]
            and phase.get("online_index_frequency_selection_included") is True
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
            and record.get("request_id") == f"index-hit20:{row['index']}:{phase['name']}:{row['sample_id']}",
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


def audit_raw_index_oracle(directory, control, evidence):
    """CPU-only replay of actual native topk for untimed sparse qualification.

    Formal requests deliberately retain full hit vectors, not enormous index
    matrices. Their masks are independently reranked by audit_controller.
    """
    import numpy as np
    require(control.get("raw_index_oracle_checked") is True,"sparse qualification needs raw-index oracle")
    groups=control.get("index_groups")
    require(type(groups) is list and groups,"qualification raw index groups absent")
    checked=0
    root=Path(directory).resolve()
    for group in groups:
        relative=group.get("file")
        require(type(relative) is str and not Path(relative).is_absolute()
                and len(Path(relative).parts)==2 and all(p not in (".","..") for p in Path(relative).parts)
                and Path(relative).suffix==".i32" and group.get("raw_indices_recorded") is True
                and group.get("cpu_raw_oracle_exact") is True
                and group.get("dtype")=="little-endian-int32","raw index artifact identity invalid")
        path=root/relative
        require(not path.parent.is_symlink() and path.parent.is_dir() and path.parent.resolve().parent==root,
                "raw index directory escapes result root")
        before=path.lstat()
        rows,columns=group["rows"],group["columns"]
        size=rows*columns*4
        require(stat.S_ISREG(before.st_mode) and before.st_size==group.get("bytes")==size
                and 0<=size<=128*1024**2,"raw index shape/size mismatch")
        fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
        with os.fdopen(fd,"rb") as handle:
            opened=os.fstat(handle.fileno())
            raw=handle.read(size+1);after=os.fstat(handle.fileno())
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(len(raw)==size and identity(before)==identity(opened)==identity(after)==identity(path.lstat()),
                "raw index changed during audit")
        sha=hashlib.sha256(raw).hexdigest()
        require(sha==group.get("sha256"),"raw index SHA mismatch")
        evidence[relative]=dict(file_sha256=sha,bytes=size,qualification_only=True)
        values=np.frombuffer(raw,dtype="<i4").reshape(rows,columns)
        positions=np.asarray(group["query_positions"],dtype=np.int64)
        ratio=group["compression_ratio"];total=group["global_kv_rows"]
        require(group["request_kv_offset"]==0 and values.shape==(rows,512)
                and positions.shape==(rows,) and ratio in (1,2),"native index layout changed")
        require(bool(np.all((values==-1)|((values>=0)&(values<total)))),"raw topk outside global KV")
        require(bool(np.all((values==-1)|(values<((positions+1)//ratio)[:,None]))),
                "raw topk violates complete compressed-group causal support")
        ordered=np.sort(values,axis=1)
        require(not bool(np.any((ordered[:,1:]==ordered[:,:-1])&(ordered[:,1:]>=0))),
                "duplicate key in native per-query topk")
        valid=values[values>=0]
        histogram=np.bincount(valid,minlength=total)
        token_keys=positions//ratio
        counts=[int(histogram[k]) if k<total else 0 for k in token_keys]
        require(counts==group.get("hitcounts"),"qualification GPU hit vector differs from independent raw CPU replay")
        checked+=1
    return checked


def partial_smoke(directory, preflight, qual, evidence):
    row=next(item for item in preflight["audit_cases"] if item["index"]==2)
    closure=read(directory/"partial_smoke.json",evidence)
    require(closure.get("schema")=="index-hit20-partial-smoke-v1" and closure.get("passed") is True
            and closure.get("index")==2 and closure.get("qualification_sha256")==qual["sha256"]
            and closure.get("input_ids_sha256")==row["input_ids_sha256"]
            and closure.get("input_tokens")==row["input_tokens"]
            and closure.get("quality_eligible") is False and closure.get("timing_valid") is False
            and closure.get("accuracy_guarantee") is False
            and closure.get("full_input_original_eight_documents") is True
            and closure.get("independent_swa_and_ffn_paths_exercised") is True
            and closure.get("output_tokens_per_arm")==1
            and set(closure.get("phase_sha256",{}))=={"index20"},"partial smoke closure incomplete")
    name="case2.index20_partial_smoke"
    phase=read(directory/(name+".json"),evidence)
    empty_ep_binding(phase, qual)
    require(phase.get("name")==name and phase.get("repeat") is None
            and phase["sha256"]==closure["phase_sha256"]["index20"],"partial smoke raw mismatch")
    rows=audit_phase(phase,row,preflight,arm="index20",timing=False,quality=False,fixed_tokens=1)
    sparse_reference=read(directory/"case2.index20_sparse_reference.json",evidence)
    sparse_closure=read(directory/"sparse_qualification.json",evidence)
    binding(sparse_reference,preflight)
    empty_ep_binding(sparse_reference,qual)
    require(sparse_reference.get("name")=="case2.index20_sparse_reference"
            and sparse_reference.get("repeat") is None and sparse_reference.get("partial_smoke") is False,
            "same-policy sparse reference phase scope mismatch")
    audit_controller(sparse_reference["controller"],row,sparse_reference,documents(row))
    expected_sparse=sparse_identity(sparse_reference,phase,row,qual["sha256"])
    require(sparse_closure==expected_sparse
            and closure.get("sparse_qualification_sha256")==sparse_closure["sha256"],
            "sparse same-policy numerical closure differs")
    raw_index_groups=sum(audit_raw_index_oracle(directory,p["controller"],evidence)
                         for p in (sparse_reference,phase))
    require(rows["ffn_bypassed_rows"]>0 and rows["swa_reused_rows"]>0
            and rows["actual_partial_chunks"]>0,"smoke did not exercise independent partial paths")
    return closure,dict(passed=True,index=2,input_tokens=row["input_tokens"],actual_rows=rows,
        closure_sha256=closure["sha256"],fixed_output_tokens=1,quality_eligible=False,timing_valid=False,
        numerical_identity_claim=False,same_policy_sparse_byteexact=True,
        sparse_qualification_sha256=sparse_closure["sha256"],raw_index_groups_audited=raw_index_groups,
        all_declared_layers_and_native_layouts_audited=True)


def audit_case(directory, row, preflight, qual, smoke, evidence):
    index = row["index"]
    pair = read(directory/f"case{index}.paired.json", evidence)
    require(pair.get("schema") == "index-hit20-case-paired-v1"
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
    baseline, candidate = arms["recomputed"], arms["index20"]
    totals={name:sum(case["actual_rows_and_swa_reuse"]["index20"].get(name,0) for case in cases)
            for name in ("swa_full_rows","swa_online_rows","swa_reused_rows","ffn_full_rows",
                         "ffn_selected_rows","attention_full_rows","attention_selected_rows",
                         "encoder_rows","encoder_selected_rows","decoder_rows","decoder_selected_rows",
                         "zero_hit_document_rows","selected_zero_hit_document_rows")}
    rates=dict(totals,
        swa_reuse_fraction=totals["swa_reused_rows"]/totals["swa_full_rows"] if totals["swa_full_rows"] else 0.,
        ffn_selected_fraction=totals["ffn_selected_rows"]/totals["ffn_full_rows"] if totals["ffn_full_rows"] else 0.,
        attention_selected_fraction=totals["attention_selected_rows"]/totals["attention_full_rows"] if totals["attention_full_rows"] else 0.,
        denominator="sum quality-phase native/YOCO surviving prefill rows in layers3..39; first3 excluded",
        not_model_flops_fraction=True,baseline_zero_counters_mean_controller_absent=True)
    compare = dict(em_loss_pp=baseline["em_percent"]-candidate["em_percent"],
        f1_loss_pp=baseline["f1_percent"]-candidate["f1_percent"],
        ttft_reduction_s=baseline["online_ttft_p50_s"]-candidate["online_ttft_p50_s"],
        ttft_reduction_percent=100*(1-candidate["online_ttft_p50_s"]/baseline["online_ttft_p50_s"]),
        ttft_speedup=baseline["online_ttft_p50_s"]/candidate["online_ttft_p50_s"],
        correct_to_wrong=sum(case["quality"]["recomputed"]["answer_em"] == 1 and
                             case["quality"]["index20"]["answer_em"] == 0 for case in cases),
        wrong_to_correct=sum(case["quality"]["recomputed"]["answer_em"] == 0 and
                             case["quality"]["index20"]["answer_em"] == 1 for case in cases))
    frontiers = {}
    for metric in ("em_percent", "f1_percent"):
        def dominates(a, b):
            return (arms[a][metric] >= arms[b][metric]
                    and arms[a]["online_ttft_p50_s"] <= arms[b]["online_ttft_p50_s"]
                    and (arms[a][metric] > arms[b][metric]
                         or arms[a]["online_ttft_p50_s"] < arms[b]["online_ttft_p50_s"]))
        frontiers[metric+"_vs_ttft"] = [a for a in ARMS if not any(dominates(b,a) for b in ARMS if a != b)]
    return dict(n=len(cases), arms=arms, comparison=compare, actual_rates=rates, empirical_frontiers=frontiers,
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
    require(preflight.get("schema") == RUN_SCHEMA and preflight.get("protocol") == index_hit20_protocol()
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
