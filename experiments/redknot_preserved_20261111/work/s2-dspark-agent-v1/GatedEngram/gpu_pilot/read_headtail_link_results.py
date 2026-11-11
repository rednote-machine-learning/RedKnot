"""Independent CPU audit of head/tail updates and native-window plus tail links.

The oracle never imports the controller or its planner. Historical tail-only
results cannot enter this candidate's statistics. Generic sealed IO and native
qualification primitives are shared with the earlier audited reader.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_cache_engram.core import digest, require, validate_digest
from gpu_pilot import read_tail_link_results as common
from gpu_pilot.headtail_link_protocol import headtail_link_protocol

SCHEMA = "headtail-link-independent-readout-v1"
PHASE_SCHEMA = "kv-cache-engram-headtail-link-phase-v1"
RUN_SCHEMA = "kv-cache-engram-headtail-link-pilot-run-v1"
ARM = "headtail128_link"
ARMS = (ARM,)
CASE_INDICES = (2, 3, 4, 5)
FFN_LAYERS = tuple(range(3, 40))
CAPTURE_LAYERS = tuple(range(40))
DOCUMENT, WIDTH, WINDOW, KEEP = 32768, 512, 128, 140
read, sealed, number = common.read, common.sealed, common.number
documents, score, verify_reference = common.documents, common.score, common.verify_reference
empty_ep_binding = common.empty_ep_binding
_positions, _event_map, _fingerprint = common._positions, common._event_map, common._fingerprint
audit_native_swa_configuration = common.audit_native_swa_configuration


def _fields(actual, expected, context):
    for field, value in expected.items():
        matches = actual.get(field) is value if type(value) is bool else actual.get(field) == value
        require(matches, context + ": " + field + " differs from independent expectation")


def normalized_spans(spans, total):
    result, previous = [], 0
    require(type(spans) in (tuple, list) and bool(spans), "nonempty document spans required")
    for occurrence, (lo, hi, key) in enumerate(spans):
        require(type(lo) is int and type(hi) is int and previous <= lo < hi <= total,
                "ordered disjoint in-bounds document spans required")
        result.append(dict(occurrence=occurrence, key=key, start=lo, end=hi,
            tail_start=max(lo, hi-128), tail_end=hi, tail_rows=min(hi-lo, 128),
            head_end=min(hi, lo+128)))
        previous = hi
    return result


def native_positions_reference(start, end, total, layer):
    require(type(start) is int and type(end) is int and type(total) is int
            and 0 <= start < end <= total and type(layer) is int and 0 <= layer < 40,
            "invalid native chunk/layer")
    if layer < 20:
        return list(range(start, end))
    surviving = list(range(max(start, total-KEEP), end))
    return surviving if surviving else [end-1]


def mask_reference(positions, start, end, spans):
    _positions(positions, start, end)
    selected, nondoc, heads, tails, inactive = [], [], [], [], []
    for row, position in enumerate(positions):
        owner = next(((lo, hi) for lo, hi, _ in spans if lo <= position < hi), None)
        if owner is None:
            selected.append(row); nondoc.append(row)
        else:
            lo, hi = owner
            head, tail = position < min(hi, lo+128), position >= max(lo, hi-128)
            if head: heads.append(row)
            if tail: tails.append(row)
            (selected if head or tail else inactive).append(row)
    return selected, nondoc, heads, tails, inactive


def expected_plan(positions, start, end, total, layer, spans):
    """Reconstruct exact geometry from source spans and pinned native layout.

    Interval intersections / sorted prefix counts are independent of the
    producer's per-tail Boolean scans. Native windows include the current key.
    Additional links are allowed across the YOCO rectangle, only after a whole
    document is complete, and never duplicate a key already in the local view.
    """
    docs = normalized_spans(spans, total)
    require(positions == native_positions_reference(start, end, total, layer),
            "native surviving positions changed or omitted hidden rows revived")
    selected, nondoc, heads, tails, inactive = mask_reference(positions, start, end, spans)
    bank = [(p, d["end"], d["occurrence"], p-d["start"])
            for d in docs for p in range(d["tail_start"], d["end"])]
    tail_positions = [item[0] for item in bank]
    ends = [d["end"] for d in docs]
    cumulative = [0]
    for doc in docs: cumulative.append(cumulative[-1]+doc["tail_rows"])
    rect = max(0, total-KEEP) if layer >= 20 else 0
    lows, lengths, completed, extras = [], [], [], []
    for query in positions:
        lo = min(query+1, max(0, query-WINDOW+1, rect))
        count = bisect_right(ends, query)
        count_rows = cumulative[count]
        overlap = min(count_rows, bisect_right(tail_positions, query))-min(count_rows, bisect_left(tail_positions, lo))
        lows.append(lo); lengths.append(query-lo+1)
        completed.append(count); extras.append(count_rows-overlap)
    expected = dict(schema="headtail-link-plan-v1", layer=layer, chunk_start=start, chunk_end=end,
        full_input_tokens=total, head_tokens=128, tail_tokens=128, native_window_size=WINDOW,
        native_valid_start=0, native_yoco_rect_lo=rect, yoco_keep_tokens=KEEP,
        documents=docs, current_positions=positions, full_rows=len(positions),
        selected_local_rows=selected, selected_positions=[positions[i] for i in selected],
        selected_rows=len(selected), document_head_local_rows=heads, document_tail_local_rows=tails,
        mandatory_nondoc_local_rows=nondoc, inactive_local_rows=inactive,
        native_window_lows=lows, native_window_lengths=lengths, completed_document_counts=completed,
        extra_tail_lengths=extras, union_lengths=[a+b for a,b in zip(lengths,extras)],
        tail_bank_positions=tail_positions, tail_bank_document_ends=[i[1] for i in bank],
        tail_bank_document_indices=[i[2] for i in bank], tail_bank_document_offsets=[i[3] for i in bank],
        tail_extra_offset=WINDOW+len(positions), global_index_offset=WINDOW+len(positions)+len(bank),
        index_width=max(128, ((WINDOW+len(bank)+127)//128)*128),
        extra_links_cross_yoco_rectangle=True, native_local_mask_preserved=True,
        native_yoco_surviving_positions_preserved=True,
        local_extra_deduplicated=True, completed_document_rule="document_end_le_query_position")
    layout_sha = digest({k:v for k,v in expected.items() if k != "layer"})
    return dict(expected, layout_sha256=layout_sha,
                plan_sha256=digest(dict(layout_sha256=layout_sha,layer=layer)))


def query_keys_reference(query, layer, total, spans):
    """Slow independent set oracle for bounded negative tests / qualification."""
    rect = max(0, total-KEEP) if layer >= 20 else 0
    local = list(range(min(query+1, max(0, query-WINDOW+1, rect)), query+1))
    seen = set(local)
    extra = [p for lo, hi, _ in spans if hi <= query
             for p in range(max(lo, hi-128), hi) if p not in seen]
    return local, extra


def audit_observed_indices(event, queries, indices):
    """Validate raw index rows without using the producer's index materializer."""
    positions, width = event["current_positions"], event["index_width"]
    lookup = {p: i for i,p in enumerate(positions)}
    spans = [(d["start"],d["end"],d["key"]) for d in event["documents"]]
    require(type(indices) is list and len(indices) == len(queries), "union index query count differs")
    for query, row in zip(queries, indices):
        require(query in lookup and type(row) is list and len(row) == width
                and all(type(slot) is int for slot in row), "invalid union index row")
        local, extra = query_keys_reference(query, event["layer"], event["full_input_tokens"], spans)
        native_slots = []
        for key in local:
            if key < event["chunk_start"]:
                slot = key-(event["chunk_start"]-WINDOW)
                require(0 <= slot < WINDOW, "historical local key outside native ring")
            else:
                require(key in lookup, "native local key lacks current hidden/KV layout")
                slot = WINDOW+lookup[key]
            native_slots.append(slot)
        expected = native_slots+[-1]*(WINDOW-len(native_slots))
        extra_set = set(extra)
        expected += [event["tail_extra_offset"]+i if p in extra_set else -1
                     for i,p in enumerate(event["tail_bank_positions"])]
        expected += [-1]*(width-len(expected))
        require(row == expected, "union index mapping/order/padding/window/causality differs")
    return True


def boundary_positions(spans):
    return sorted({p for lo,hi,_ in spans for a,b in ((lo,min(hi,lo+128)),(max(lo,hi-128),hi))
                   for p in range(a,b)})


def audit_refresh_event(event, positions, spans, accumulated, *, generation, qualification):
    layer, start, end = event.get("layer"), event.get("chunk_start"), event.get("chunk_end")
    require(type(layer) is int and layer in FFN_LAYERS and type(generation) is int and generation > 0,
            "headtail refresh layer/generation invalid")
    _positions(positions,start,end)
    boundary = boundary_positions(spans)
    slots = {p:i for i,p in enumerate(boundary)}
    rows = [i for i,p in enumerate(positions) if p in slots]
    refreshed = [positions[i] for i in rows]
    require(not set(refreshed) & accumulated, "same-layer boundary refresh occurred twice")
    available = accumulated | set(refreshed)
    cutoff = positions[-1] if positions else -1
    visible = {p for p in boundary if p <= cutoff}
    expected = dict(layer=layer, chunk_start=start, chunk_end=end, phase_generation=generation,
        schema="headtail-link-dynamic-swa-v1",
        source="current_layer_attention_input", input_representation="native_packed_current", refresh_enabled=True,
        current_boundary_local_rows=rows, current_boundary_positions=refreshed,
        refreshed_local_rows=rows, refreshed_positions=refreshed,
        refreshed_bank_rows=[slots[p] for p in refreshed], available_refreshed_positions=sorted(available),
        visible_refreshed_positions=sorted(visible & available),
        visible_offline_fallback_positions=sorted(visible-available))
    _fields(event, expected, f"headtail refresh layer={layer} chunk=[{start},{end})")
    require(type(event.get("phase_generation")) is int, "refresh generation must be an integer")
    checked = qualification and bool(rows)
    _fields(event, dict(projection_identity_checked=checked, prepare_identity_checked=checked,
        checks=2 if checked else 0), "headtail projection/prepare evidence")
    require((type(event.get("max_absolute_error")) in (int,float) and event["max_absolute_error"] == 0)
            if checked else event.get("max_absolute_error") is None, "headtail exact projection/RoPE/cast proof absent")
    accumulated.update(refreshed)
    return len(refreshed)


def _finite_bfloat16(raw):
    """Exact exponent check using bounded byte operations, no numeric runtime."""
    require(len(raw)%2 == 0, "unaligned BF16 bytes")
    high = bytes(128 if (value & 127) == 127 else 0 for value in range(256))
    return (int.from_bytes(raw[::2], "little") & int.from_bytes(raw[1::2].translate(high), "little")) == 0


def audit_bank_payload(path, expected_sha256):
    path = Path(path)
    validate_digest(expected_sha256, "full canonical payload SHA")
    size = DOCUMENT*WIDTH*2
    ident = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size == size,
            "full canonical bank must contain32768x512BF16, never tail-only")
    fd = os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    checksum, count = hashlib.sha256(), 0
    with os.fdopen(fd,"rb") as handle:
        opened = os.fstat(handle.fileno())
        while True:
            block = handle.read(2*1024**2)
            if not block: break
            count += len(block); checksum.update(block)
            require(count <= size and _finite_bfloat16(block), "nonfinite/oversized full canonical bank")
        after = os.fstat(handle.fileno())
    require(count == size and ident(before)==ident(opened)==ident(after)==ident(path.lstat()),
            "canonical bank changed during read")
    require(checksum.hexdigest()==expected_sha256,"canonical payload bytes changed")
    return dict(file_sha256=expected_sha256,bytes=size,finite_bfloat16=True)


def binding(phase, preflight):
    audit_native_swa_configuration(preflight)
    _fields(phase, dict(manifest_sha256=preflight["manifest_sha256"],
        headtail_link_protocol_sha256=preflight["protocol"]["sha256"],
        execution_configuration_sha256=preflight["identity"]["execution_configuration_sha256"],
        head_tokens=128,tail_tokens=128,dynamic_refresh_start_layer=3,
        native_yoco_swa_share_hidden=False), "headtail phase execution binding")
    require(phase.get("controller",{}).get("native_yoco_swa_share_hidden") is False,
            "controller native configuration is not verified False")


def memory_accounting(event, spans, refreshed, available_nondoc, *, dense):
    """Count consumed sources through interval intersections, never Qx1024 loops."""
    positions = event["current_positions"]
    queries = list(range(len(positions))) if dense else event["selected_local_rows"]
    ref = sorted(refreshed)
    tail_ref = sorted(p for p in refreshed if any(max(lo,hi-128) <= p < hi for lo,hi,_ in spans))
    doc_ends = [hi for _,hi,_ in spans]
    counts = dict(consumed_local_pairs=0,consumed_extra_tail_pairs=0,consumed_cached_pairs=0,
                  consumed_refreshed_pairs=0,consumed_online_nondoc_pairs=0)
    intervals = []
    for index in queries:
        q, low = positions[index], event["native_window_lows"][index]
        local = event["native_window_lengths"][index]
        doc_rows = sum(max(0,min(hi,q+1)-max(lo,low)) for lo,hi,_ in spans)
        non_rows = local-doc_rows
        require(bisect_right(available_nondoc,q)-bisect_left(available_nondoc,low) == non_rows,
                "native local window refers to ungenerated nondocument KV")
        local_ref = bisect_right(ref,q)-bisect_left(ref,low)
        completed = bisect_right(doc_ends,q)
        completed_end = doc_ends[completed-1] if completed else 0
        extra_ref = bisect_left(tail_ref,completed_end)
        extra_ref -= max(0,bisect_left(tail_ref,min(completed_end,q+1))-bisect_left(tail_ref,low))
        extra = event["extra_tail_lengths"][index]
        require(0 <= local_ref <= doc_rows and 0 <= extra_ref <= extra,
                "refresh source accounting escaped native/extra visibility")
        counts["consumed_local_pairs"] += local
        counts["consumed_extra_tail_pairs"] += extra
        counts["consumed_cached_pairs"] += doc_rows+extra-local_ref-extra_ref
        counts["consumed_refreshed_pairs"] += local_ref+extra_ref
        counts["consumed_online_nondoc_pairs"] += non_rows
        if local: intervals.append((low,q+1))
    if queries:
        last = positions[queries[-1]]
        intervals += [(max(lo,hi-128),hi) for lo,hi,_ in spans if hi <= last]
    merged = []
    for lo,hi in sorted(intervals):
        if merged and lo <= merged[-1][1]: merged[-1] = (merged[-1][0],max(merged[-1][1],hi))
        else: merged.append((lo,hi))
    counts["materialized_offline_rows"] = sum(
        sum(max(0,min(hi,b)-max(lo,a)) for lo,hi,_ in spans)
        -(bisect_left(ref,b)-bisect_left(ref,a)) for a,b in merged)
    return counts


def audit_controller(control, row, phase, spans):
    require(phase.get("arm") == ARM and phase.get("mode") in ("sensitive","reference_sparse")
            and phase.get("fraction") is None and phase.get("dynamic_document_swa") is True
            and phase.get("native_yoco_swa_share_hidden") is False,
            "foreign/static/native policy cannot satisfy headtail audit")
    _fields(control, dict(schema="headtail-link-controller-v1", mode=phase["mode"], phase=phase["name"],
        input_ids_sha256=row["input_ids_sha256"], full_input_tokens=row["input_tokens"],
        ffn_layers=list(FFN_LAYERS), first_three_full=True, actual_full_row_native_mhc=True,
        yoco_layout_preserved=True, global_kv_native_online=True, decode_native=True,
        formal_admission_unchanged=True, not_model_flops_fraction=True, prefill_only_link=True,
        selected_tokens_forced_through_yoco=False, native_yoco_swa_share_hidden=False,
        head_tokens=128,tail_tokens=128,dynamic_refresh_start_layer=3,dynamic_document_swa=True,
        native_local_window_preserved=True,extra_completed_document_tail_links=True,
        native_local_mask_preserved=True,
        extra_links_cross_yoco_rectangle=True,native_valid_start=0,yoco_keep_tokens=140,
        offline_gpu_resident=True,offline_bank_representation="swa-postnorm-prerope-bf16",
        phase_local_boundary_overlay=True,boundary_overlay_rows_per_layer=len(boundary_positions(spans)),
        staging=None,offline_bank_immutable=True,middle_kv_immutable=True,
        phase_overlay_fresh=True,cross_phase_overlay_reuse=False,
        offline_bank_entry_count=40*8,offline_bank_rows=40*8*DOCUMENT,
        offline_bank_bytes=40*8*DOCUMENT*WIDTH*2), "headtail controller scope")
    generation = control.get("phase_generation")
    require(type(generation) is int and generation > 0,"positive phase-local overlay generation required")
    adapter = control.get("adapter",{})
    _fields(adapter, dict(schema="kv-cache-engram-index-hit20-adapter-v1",
        decoder_forward_unchanged=True,attention_forward_unchanged=True,native_mhc_unchanged=True,
        native_yoco_layout_unchanged=True,all_selected_delegates_native=True,
        empty_selection_skips_ep_collective=False,cached_hidden_or_ffn_output=False,
        accuracy_guarantee=False,speedup_claim=False),"native FFN adapter")
    chunks = [(s,min(s+8192,row["input_tokens"])) for s in range(0,row["input_tokens"],8192)]
    expected_keys = {(s,e,l) for s,e in chunks for l in FFN_LAYERS}
    selection = _event_map(control.get("selection"),expected_keys,"selection")
    layouts = _event_map(control.get("ffn_layouts"),expected_keys,"FFN layout")
    ffn = _event_map(adapter.get("events"),expected_keys,"FFN compute")
    swa = _event_map(control.get("swa_events"),expected_keys,"SWA compute/store")
    attention = _event_map(control.get("attention_events"),expected_keys,"attention compute")
    memories = _event_map(control.get("headtail_memory_events"),expected_keys,"native-window/extra-tail union")
    refreshes = _event_map(control.get("headtail_swa_events"),expected_keys,"current-layer headtail refresh")
    dense = phase["mode"] == "reference_sparse"
    qualified = dense or phase.get("verify_sparse") is True
    union_checks = _event_map(control.get("union_oracle_events"),expected_keys,"GPU union oracle") if qualified else {}
    if not qualified: require(control.get("union_oracle_events") == [],"formal timing claims qualification-only union checks")
    restores = control.get("boundary_restore_identity_events")
    require(type(restores) is list and len(restores)==len(FFN_LAYERS),"all boundary restores required")
    restore_layers = set()
    for event in restores:
        layer = event.get("layer")
        require(layer in FFN_LAYERS and layer not in restore_layers,"missing/duplicate boundary restore")
        restore_layers.add(layer)
        _fields(event,dict(rows=len(boundary_positions(spans)),checked=qualified,checks=1 if qualified else 0,
            absolute_positions_sha256=digest(boundary_positions(spans))),"boundary absolute-RoPE restore")
        require((type(event.get("max_absolute_error")) in (int,float) and event["max_absolute_error"]==0)
                if qualified else event.get("max_absolute_error") is None,"boundary restore is not exact")
    refreshed = {l:set() for l in FFN_LAYERS}
    nondoc = {l:set() for l in FFN_LAYERS}
    templates, statistics_cache = {}, {}
    totals = dict(ffn_full_rows=0,ffn_selected_rows=0,ffn_actual_computed_rows=0,
        attention_full_rows=0,attention_selected_rows=0,attention_actual_computed_rows=0,
        swa_online_rows=0,swa_reused_rows=0,native_cache_store_rows=0,
        dynamic_refreshed_rows=0,actual_partial_chunks=0,head_selected_rows=0,tail_selected_rows=0,
        encoder_rows=0,encoder_selected_rows=0,decoder_rows=0,decoder_selected_rows=0,
        consumed_local_pairs=0,consumed_extra_tail_pairs=0,consumed_cached_pairs=0,
        consumed_refreshed_pairs=0,consumed_online_nondoc_pairs=0,materialized_offline_rows=0,
        main_q_a_rows=0,index_q_rows=0,global_compute_calls=0)
    for start,end,layer in sorted(expected_keys):
        key = start,end,layer
        m, fe, se, ae, event = selection[key],ffn[key],swa[key],attention[key],memories[key]
        positions = m.get("current_positions")
        template_key = start,end,layer>=20
        if template_key not in templates:
            templates[template_key] = expected_plan(positions,start,end,row["input_tokens"],layer,spans)
        template = templates[template_key]
        expected = dict(template,layer=layer,
            plan_sha256=digest(dict(layout_sha256=template["layout_sha256"],layer=layer)))
        _fields(event,expected,f"union plan layer={layer} chunk=[{start},{end})")
        ids, non, heads, tails = (expected[k] for k in
            ("selected_local_rows","mandatory_nondoc_local_rows","document_head_local_rows","document_tail_local_rows"))
        n,k = len(positions),len(ids)
        _fields(m,dict(current_positions=expected["current_positions"],selected_local_rows=ids,
            selected_positions=[positions[i] for i in ids],full_rows=n,selected_rows=k,
            plan_sha256=expected["plan_sha256"]),"attention query selection")
        _fields(layouts[key],{field:m[field] for field in
            ("current_positions","selected_local_rows","full_rows","selected_rows","plan_sha256")},"FFN layout")
        actual = n if dense else k
        _fields(fe,dict(full_rows=n,selected_rows=k,actual_mlp_rows=actual,reference_dense=dense,
            plan_sha256=expected["plan_sha256"],request_id=phase["record"]["request_id"],phase=phase["name"],
            inactive_ffn_output="zero",empty_native_collective=k==0 and n>0),"actual native FFN execution")
        local_index = layer in (8,14,20,24,28,32,36)
        _fields(ae,dict(full_rows=n,selected_rows=k,main_q_a_rows=n if local_index or dense else k,
            index_q_rows=n if local_index else 0,main_q_b_rows=actual,joint_attention_rows=actual,
            output_projection_rows=actual,inactive_attention_zero_rows=n-k,global_compute_called=True,
            native_joint_attention=True,is_global_source=layer in (8,14,20),reference_dense=dense,
            native_local_window_preserved=True,extra_completed_document_tail_links=True,
            plan_sha256=expected["plan_sha256"]),"actual Q/global/attention execution")
        require(type(ae.get("global_kv_rows")) is int and ae["global_kv_rows"] >= 0,"global KV geometry absent")
        refreshed_count = audit_refresh_event(refreshes[key],positions,spans,refreshed[layer],
            generation=generation,qualification=qualified)
        _fields(se,dict(current_positions=positions,online_local_rows=ids,online_rows=k,reused_rows=n-k,
            full_rows=n,compact=layer>=20,native_cache_store_rows=n,head_rows=len(heads),tail_rows=len(tails),
            global_index_offset=expected["global_index_offset"]),"SWA full-bank reuse and cache store")
        queries = positions if dense else [positions[i] for i in ids]
        _fields(event,dict(query_positions=queries,actual_joint_query_rows=actual,reference_dense=dense,
            native_cache_store_rows=n,global_source_is_native=True,
            tail_slots_absolute_positions_sha256=digest(expected["tail_bank_positions"]),
            gpu_native_window_checked=True,gpu_union_oracle_checked=qualified,
            native_valid_start_binding="native_indices_verified_zero_start"),"native-window/global-offset GPU binding")
        if qualified:
            _fields(union_checks[key],dict(queries=len(queries),index_width=expected["index_width"],checks=1),
                    "exact full union GPU qualification")
        nondoc[layer].update(positions[i] for i in non)
        stats_key = template_key,dense,frozenset(refreshed[layer]),frozenset(nondoc[layer])
        if stats_key not in statistics_cache:
            statistics_cache[stats_key] = memory_accounting(expected,spans,refreshed[layer],sorted(nondoc[layer]),dense=dense)
        for name,value in statistics_cache[stats_key].items(): totals[name] += value
        for name,value in dict(ffn_full_rows=n,ffn_selected_rows=k,ffn_actual_computed_rows=actual,
            attention_full_rows=n,attention_selected_rows=k,attention_actual_computed_rows=actual,
            swa_online_rows=k,swa_reused_rows=n-k,native_cache_store_rows=n,
            dynamic_refreshed_rows=refreshed_count,actual_partial_chunks=int(0<k<n),
            head_selected_rows=len(heads),tail_selected_rows=len(tails),
            main_q_a_rows=ae["main_q_a_rows"],index_q_rows=ae["index_q_rows"],global_compute_calls=1).items():
            totals[name] += value
        category = "encoder" if layer<20 else "decoder"
        totals[category+"_rows"] += n; totals[category+"_selected_rows"] += k
    _fields(adapter.get("counts",{}),dict(full_rows=totals["ffn_full_rows"],selected_rows=totals["ffn_selected_rows"],
        actual_mlp_rows=totals["ffn_actual_computed_rows"],
        inactive_ffn_zero_rows=totals["ffn_full_rows"]-totals["ffn_selected_rows"]),"FFN aggregate")
    rings = control.get("decode_ring_positions")
    require(type(rings) is list and len(rings)==len(FFN_LAYERS)
            and {e.get("layer") for e in rings}==set(FFN_LAYERS),"physical decode ring layer coverage changed")
    for event in rings:
        _fields(event,dict(positions=list(range(row["input_tokens"]-WINDOW,row["input_tokens"])),
            physical_readback_exact=True,checks=WINDOW),"physical decode ring")
        slots = event.get("physical_slots")
        require(type(slots) is list and len(slots)==len(set(slots))==WINDOW
                and all(type(slot) is int and slot>=0 for slot in slots),"decode ring physical slot collision")
    capacity = len(FFN_LAYERS)*len(boundary_positions(spans))
    totals.update(ffn_bypassed_rows=totals["ffn_full_rows"]-totals["ffn_selected_rows"],
        attention_bypassed_rows=totals["attention_full_rows"]-totals["attention_selected_rows"],
        boundary_slot_capacity=capacity,unrefreshed_boundary_slots=capacity-totals["dynamic_refreshed_rows"],
        dynamic_refreshed_boundary_fraction=totals["dynamic_refreshed_rows"]/capacity,
        dynamic_refresh_start_layer=3,layer3_offline_seed=False,
        phase_generation=generation,first_three_layers_excluded=True,not_model_flops_fraction=True)
    return totals


# Native qualification checks retain the existing exact-zero contracts.
projection_identity = common.projection_identity
cache_restore_identity = common.cache_restore_identity
audit_joint_attention = common.audit_joint_attention
audit_native_no_cache = common.audit_native_no_cache
cuda_mask_qualification = common.cuda_mask_qualification


def frozen_sources(preflight):
    """Recheck every bound Python/dependency/native source, without importing it."""
    from gpu_pilot.headtail_link_protocol import DEPENDENCIES
    require(DEPENDENCIES, "new reader dependency pins have not been frozen")
    stage = Path(__file__).resolve().parents[1]
    identity = preflight["identity"]
    root = Path(identity["experiment_root"])
    native = Path(identity["isolated_native_snapshot"])
    require(root.is_absolute() and native.is_absolute() and native.resolve() == stage.parent/"NativeSnapshot",
            "source roots differ from independent frozen deployment")
    recorded = identity.get("pilot_source_sha256")
    require(type(recorded) is dict and all(recorded.get(k) == v for k,v in DEPENDENCIES.items()),
            "preflight dependency pins differ from reader")
    paths = {"GatedEngram/"+str(p.relative_to(stage)):p for p in stage.rglob("*.py")
             if "__pycache__" not in p.parts}
    expected_stage = {k for k in recorded if k.startswith("GatedEngram/")}
    require(set(paths) == expected_stage, "frozen stage Python inventory changed")
    for name, expected in recorded.items():
        validate_digest(expected, "frozen source SHA")
        path = paths[name] if name in paths else root/name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "frozen source changed: "+name)
    native_files = identity["runtime_source_files_sha256"]
    require(digest(native_files) == identity["runtime_source_tree_sha256"], "native source aggregate differs")
    require(native.is_dir() and not native.is_symlink(), "native snapshot must be a real directory")
    git = ["git", "-c", "safe.directory="+str(native.resolve()), "-C", str(native)]
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], env=git_env).decode().strip()
    diff = subprocess.check_output([*git, "diff", "--no-ext-diff", "HEAD"], env=git_env)
    names = subprocess.check_output([*git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], env=git_env)
    current_names = set()
    for raw in set(names.split(b"\0")):
        if raw:
            name = os.fsdecode(raw)
            path = native/name
            if path.is_file() and path.suffix in (".py", ".pyi", ".cpp", ".hpp", ".cu", ".h", ".toml"):
                require(not path.is_symlink(), "native runtime source symlink rejected")
                current_names.add(name)
    require(current_names == set(native_files) and head == identity["commit"]
            and hashlib.sha256(diff).hexdigest() == identity["source_diff_sha256"]
            and bool(diff) is identity["source_is_modified"], "native source inventory/commit/diff changed")
    for name, expected in native_files.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "native source path escapes snapshot")
        path = native/name
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected, "private native source changed: "+name)
    return dict(pilot_source_sha256=digest(recorded), native_source_sha256=digest(native_files))


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
    require(closure.get("schema") == "headtail-link-qualification-v1" and closure.get("passed") is True
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


def sparse_identity(reference, candidate, row, qualification_sha256, *, controller_results=None):
    """Strict same-policy sparse parity; never an equivalence-to-full claim."""
    validate_digest(qualification_sha256,"native all-selected qualification")
    arm = reference.get("arm")
    require(arm in ARMS and candidate.get("arm") == arm, "same-policy qualification arm mismatch")
    for phase,mode in ((reference,"reference_sparse"),(candidate,"sensitive")):
        sealed(phase)
        require(phase.get("schema")==PHASE_SCHEMA and phase.get("mode")==mode
                and phase.get("dynamic_document_swa") is True
                and phase.get("arm")==arm and phase.get("fraction") is None
                and phase.get("quality_eligible") is False and phase.get("timing_valid") is False
                and phase.get("index")==row["index"]
                and phase.get("input_ids_sha256")==row["input_ids_sha256"],
                "sparse reference/candidate phase identity mismatch")
        record=phase["record"]; control=phase["controller"]
        require(control.get("same_policy_causal_oracle_checked") is True,
                "same-policy qualification omitted tail mapping/restore oracle")
        audited = audit_controller(control,row,phase,documents(row))
        if controller_results is not None:
            controller_results[phase["sha256"]] = audited
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
    for name in ("manifest_sha256","execution_configuration_sha256","headtail_link_protocol_sha256",
                 "empty_ep_qualification_sha256"):
        require(reference.get(name)==candidate.get(name),"sparse qualification identity changed")
    require(candidate.get("verify_sparse") is True
            and candidate["controller"].get("sparse_identity_exact") is True
            and reference["controller"]["sparse_identity_fingerprints"]
                ==candidate["controller"]["sparse_identity_fingerprints"]
            and reference["record"]["output_ids"]==candidate["record"]["output_ids"],
            "sparse same-policy reference byte parity failed")
    fields=("start","end","layer","current_positions","selected_local_rows")
    require([{k:m[k] for k in fields} for m in reference["controller"]["selection"]]
            ==[{k:m[k] for k in fields} for m in candidate["controller"]["selection"]],
            "sparse reference and production masks differ")
    body=dict(schema="headtail-link-sparse-qualification-v1",passed=True,arm=arm,
        index=row["index"],input_tokens=row["input_tokens"],input_ids_sha256=row["input_ids_sha256"],
        qualification_sha256=qualification_sha256,reference_sha256=reference["sha256"],
        candidate_sha256=candidate["sha256"],
        identity_check_count=candidate["controller"]["sparse_identity_check_count"],
        identity_fingerprints_sha256=candidate["controller"]["sparse_identity_fingerprints_sha256"],
        quality_eligible=False,timing_valid=False,equivalence_to_full_model_claim=False,
        exact_same_policy_reference=True)
    return dict(body,sha256=digest(body))


def audit_phase(phase,row,preflight,*,arm=ARM,timing,quality,fixed_tokens=32,controller_result=None):
    sealed(phase); binding(phase,preflight)
    require(arm==ARM,"historical/native arm is not a fresh candidate observation")
    _fields(phase,dict(schema=PHASE_SCHEMA,index=row["index"],dataset=row["dataset"],arm=ARM,
        dynamic_document_swa=True,mode="sensitive",fraction=None,timing_valid=timing,
        quality_eligible=quality,input_ids_sha256=row["input_ids_sha256"],
        online_full_bank_gpu_resident=True,online_clock_started_before_reset=True,
        phase_local_memory_rebuilt=True,offline_build_included=False,
        native_global_kv_and_index_preserved=True,
        partial_smoke=not quality and not timing and fixed_tokens==1,
        verify_sparse=not quality and not timing and fixed_tokens==1),"headtail phase scope")
    rec=phase["record"]
    _fields(rec,dict(success=True,rank=0,input_ids_sha256=row["input_ids_sha256"],
        input_tokens=row["input_tokens"],phase=phase["name"],sample_id=row["sample_id"],dataset=row["dataset"],
        request_id=f"headtail-link:{row['index']}:{phase['name']}:{row['sample_id']}"),"request/output scope")
    ids=rec.get("output_ids")
    require(type(ids) is list and bool(ids) and all(type(i) is int and i>=0 for i in ids)
            and digest(ids)==rec.get("output_ids_sha256"),"raw output token binding differs")
    if quality:
        require(rec.get("natural_EOS") is True and len(ids)<=64
                and rec.get("finish_reason") in ("length","stop")
                and rec.get("truncated") is (rec.get("finish_reason")=="length"),"quality EOS/cap mismatch")
        scores=score(rec.get("text"),row["references"])
        require(all(type(rec.get(k)) in (float,int) and math.isfinite(rec[k])
                    and abs(rec[k]-v)<=1e-12 for k,v in scores.items()),"fixed-gold quality is not reproducible")
    else:
        require(type(fixed_tokens) is int and fixed_tokens in (1,32) and (not timing or fixed_tokens==32)
                and len(ids)==fixed_tokens and rec.get("natural_EOS") is False
                and rec.get("finish_reason")=="length" and rec.get("truncated") is False,
                "formal/warmup/smoke token policy changed")
    ttft=number(rec.get("online_ttft_s"),"online TTFT",True)
    elapsed=number(rec.get("online_elapsed_s"),"online elapsed",True)
    prepared=number(rec.get("online_prepare_to_submit_s"),"online preparation")
    require(prepared<=ttft<=elapsed<=number(phase.get("online_global_window_s"),"common wall",True),
            "online timing ordering changed")
    return (audit_controller(phase["controller"],row,phase,documents(row))
            if controller_result is None else controller_result)


def partial_smoke(directory,preflight,qual,evidence):
    row=next(r for r in preflight["audit_cases"] if r["index"]==2)
    closure=read(directory/"partial_smoke.json",evidence)
    _fields(closure,dict(schema="headtail-link-partial-smoke-v1",passed=True,index=2,
        qualification_sha256=qual["sha256"],input_ids_sha256=row["input_ids_sha256"],
        input_tokens=row["input_tokens"],quality_eligible=False,timing_valid=False,accuracy_guarantee=False,
        full_input_original_eight_documents=True,independent_swa_and_ffn_paths_exercised=True,
        output_tokens_per_arm=1),"new full-input smoke closure")
    require(set(closure.get("phase_sha256",{}))=={ARM},"smoke closure contains a historical arm")
    phase=read(directory/f"case2.{ARM}_partial_smoke.json",evidence)
    reference=read(directory/f"case2.{ARM}_sparse_reference.json",evidence)
    sparse=read(directory/"sparse_qualification.json",evidence)
    for value,name in ((phase,f"case2.{ARM}_partial_smoke"),(reference,f"case2.{ARM}_sparse_reference")):
        binding(value,preflight);empty_ep_binding(value,qual)
        require(value.get("name")==name and value.get("repeat") is None,"same-policy smoke/reference name changed")
    require(reference.get("partial_smoke") is False and phase["sha256"]==closure["phase_sha256"][ARM],
            "smoke raw phase seal mismatch")
    controller_results={}
    expected=sparse_identity(reference,phase,row,qual["sha256"],controller_results=controller_results)
    require(sparse==expected and closure.get("sparse_qualification_sha256")==sparse["sha256"],
            "fresh same-policy numerical closure differs")
    rows=audit_phase(phase,row,preflight,timing=False,quality=False,fixed_tokens=1,
                     controller_result=controller_results[phase["sha256"]])
    require(rows["ffn_bypassed_rows"]>0 and rows["dynamic_refreshed_rows"]>0
            and rows["consumed_cached_pairs"]>0 and rows["actual_partial_chunks"]>0,
            "smoke did not execute boundary refresh, full-bank reuse and partial FFN")
    return closure,dict(passed=True,arm=ARM,index=2,input_tokens=row["input_tokens"],actual_rows=rows,
        closure_sha256=closure["sha256"],quality_eligible=False,timing_valid=False,fixed_output_tokens=1,
        same_policy_sparse_byteexact=True,sparse_qualification_sha256=sparse["sha256"],
        first_three_reference_sparse_fingerprints_sha256=digest([r for r in reference["controller"]["sparse_identity_fingerprints"] if r["key"][2]<3]),
        equivalence_to_full_model_claim=False,all_declared_layers_and_native_layouts_audited=True)


def audit_timing_summary(summary, samples):
    require(type(samples) is list and len(samples)==3,"three raw formal samples required")
    for sample in samples:number(sample,"formal TTFT",True)
    expected=dict(arms={ARM:dict(online_ttft_samples_s=samples,
        online_ttft_p50_s=statistics.median(samples),repetitions=3,fixed_output_tokens=32)},
        concurrency=1,active_dp_ranks=1,participating_ep_ranks=8,
        online_full_bank_gpu_resident=True,online_gather_scatter_and_fallback_included=True,
        model_load_included=False,offline_build_included=False,saturated_capacity_claim=False,
        fresh_paired_baseline=False,historical_results_used_for_admission=False)
    require(type(summary) is dict and set(summary)==set(expected),"candidate timing summary scope differs")
    _fields(summary,expected,"raw three-repeat timing summary")


def audit_offline_phase(phase,row,ordinal,key,preflight,qual):
    empty_ep_binding(phase,qual);binding(phase,preflight)
    name=f"case{row['index']}.offline_doc{ordinal}"
    _fields(phase,dict(schema=PHASE_SCHEMA,name=name,index=None,dataset=row["dataset"],arm="offline",
        mode="capture_independent",input_ids_sha256=key,repeat=None,fraction=None,
        timing_valid=False,quality_eligible=False,partial_smoke=False,verify_sparse=False,
        dynamic_document_swa=False,online_full_bank_gpu_resident=False,
        online_clock_started_before_reset=True,phase_local_memory_rebuilt=True,
        offline_build_included=False,native_global_kv_and_index_preserved=True),"offline document phase")
    rec=phase["record"]
    _fields(rec,dict(success=True,rank=0,input_ids_sha256=key,input_tokens=DOCUMENT,phase=name,
        sample_id=key,dataset=row["dataset"],request_id=f"headtail-link:doc:{name}:{key}",
        natural_EOS=False,finish_reason="length",truncated=False),"offline document request")
    ids=rec.get("output_ids")
    require(type(ids) is list and len(ids)==1 and type(ids[0]) is int and ids[0]>=0
            and rec.get("output_ids_sha256")==digest(ids),"offline document fixed output binding differs")
    control=phase["controller"]
    _fields(control,dict(mode="capture_independent",phase=name,input_ids_sha256=key,
        full_input_tokens=DOCUMENT,offline_yoco_full_layout=True,model_boundary_restored=True,
        dynamic_document_swa=False,headtail_swa_events=[],headtail_memory_events=[],
        union_oracle_events=[],boundary_restore_identity_events=[]),"independent full canonical capture")
    audit_native_no_cache(control)
    return control


def audit_case(directory,row,preflight,qual,smoke,evidence):
    index=row["index"]
    pair=read(directory/f"case{index}.candidate.json",evidence)
    _fields(pair,dict(schema="headtail-link-case-v1",qualification_sha256=qual["sha256"],
        partial_smoke_sha256=smoke["sha256"],ffn_layers=list(FFN_LAYERS),
        offline_capture_layers=list(CAPTURE_LAYERS),all_negative_results_retained=True,
        exploratory_previously_exposed_inputs=True,calibration_claim=False),"candidate case closure")
    _fields(pair,{k:row[k] for k in ("index","dataset","sample_id","references","question","input_tokens","input_ids_sha256")},
            "candidate source identity")
    require(set(pair.get("quality",{}))=={ARM},"only one fresh candidate quality result is permitted")
    masks,quality,timing={}, {}, {ARM:[]}
    generations=set()
    def fresh(value):
        generation=value["controller"]["phase_generation"]
        require(generation not in generations,"phase reused a boundary overlay generation")
        generations.add(generation)
    phase=read(directory/f"case{index}.{ARM}_quality.json",evidence)
    empty_ep_binding(phase,qual)
    verify_reference(pair["quality"][ARM],directory/f"case{index}.{ARM}_quality.json",phase,evidence)
    require(phase.get("name")==f"case{index}.{ARM}_quality" and phase.get("repeat") is None,"quality filename scope changed")
    masks[ARM]=audit_phase(phase,row,preflight,timing=False,quality=True);fresh(phase)
    rec=phase["record"]
    quality[ARM]={k:rec[k] for k in ("text","answer_em","answer_token_f1","truncated","finish_reason")}
    quality[ARM]["generated_tokens"]=len(rec["output_ids"])
    warmup=read(directory/f"case{index}.{ARM}_warmup.json",evidence)
    empty_ep_binding(warmup,qual)
    require(warmup.get("name")==f"case{index}.{ARM}_warmup" and warmup.get("repeat") is None,"warmup scope changed")
    audit_phase(warmup,row,preflight,timing=False,quality=False);fresh(warmup)
    recorded=pair.get("timing_phases")
    require(type(recorded) is list and [(p.get("repeat"),p.get("arm")) for p in recorded]==[(i,ARM) for i in range(3)],
            "exactly three fresh candidate timing repeats required")
    for repeat,ref in enumerate(recorded):
        name=f"case{index}.{ARM}_measure{repeat}.json"
        value=read(directory/name,evidence)
        empty_ep_binding(value,qual);verify_reference(ref,directory/name,value,evidence,extras=("repeat","arm"))
        require(value.get("name")==name[:-5] and value.get("repeat")==repeat,"formal timing filename/repeat changed")
        audit_phase(value,row,preflight,timing=True,quality=False);fresh(value)
        timing[ARM].append(value["record"]["online_ttft_s"])
    audit_timing_summary(pair.get("timing_summary"),timing[ARM])
    builds=[]
    for ordinal,(_,_,key) in enumerate(documents(row)):
        phase=read(directory/f"case{index}.offline_doc{ordinal}.json",evidence)
        control=audit_offline_phase(phase,row,ordinal,key,preflight,qual)
        payloads=control.get("offline_payloads")
        require(type(payloads) is list and len(payloads)==len(CAPTURE_LAYERS),"full canonical capture layer coverage missing")
        by_layer={item["scope"]["layer"]:item for item in payloads}
        require(set(by_layer)==set(CAPTURE_LAYERS),"duplicate/foreign offline layer")
        for layer,item in by_layer.items():
            filename=key+f".layer{layer}.bf16"
            expected_scope=dict(model_sha256=preflight["identity"]["checkpoint_index_sha256"],
                tokenizer_sha256=preflight["identity"]["tokenizer_json_sha256"],
                execution_sha256=preflight["identity"]["execution_configuration_sha256"],
                document_sha256=key,layer=layer,representation="swa-postnorm-prerope-bf16")
            _fields(item,dict(scope=expected_scope,file=filename,shape=[DOCUMENT,WIDTH],dtype="bfloat16",
                bytes=DOCUMENT*WIDTH*2),"immutable full canonical payload")
            relative=f"case{index}.offline-kv/"+filename
            path=directory/relative
            require(path.parent.is_dir() and not path.parent.is_symlink(),"canonical directory symlink rejected")
            require(relative not in evidence,"full canonical payload must be hashed exactly once")
            evidence[relative]=audit_bank_payload(path,item["sha256"])
        builds.append(phase["sha256"])
    require(builds==pair.get("offline_phase_sha256"),"offline build closure mismatch")
    return pair,dict(index=index,dataset=row["dataset"],references=row["references"],question=row["question"],
        input_tokens=row["input_tokens"],quality=quality,online_ttft_samples_s=timing,
        online_ttft_p50_s={ARM:statistics.median(timing[ARM])},actual_rows_and_swa_reuse=masks,
        offline_build_s=number(pair.get("offline_build_s"),"offline build seconds"),closure_sha256=pair["sha256"],
        fresh_paired_baseline=False,historical_results_used_for_admission=False)


def summarize_cases(cases):
    if not cases:return None
    samples=[s for case in cases for s in case["online_ttft_samples_s"][ARM]]
    rates=[case["actual_rows_and_swa_reuse"][ARM] for case in cases]
    totals={key:sum(rate.get(key,0) for rate in rates) for key in rates[0]
            if type(rates[0][key]) is int and key not in ("phase_generation","dynamic_refresh_start_layer")}
    totals.update(ffn_selected_fraction=totals["ffn_selected_rows"]/totals["ffn_full_rows"],
        attention_selected_fraction=totals["attention_selected_rows"]/totals["attention_full_rows"],
        dynamic_refreshed_boundary_fraction=totals["dynamic_refreshed_rows"]/totals["boundary_slot_capacity"],
        not_model_flops_fraction=True,compute_denominator="quality-phase native/YOCO surviving prefill rows, zero-based3..39")
    return dict(n=len(cases),arms={ARM:dict(n=len(cases),online_ttft_p50_s=statistics.median(samples),
        online_ttft_samples_s=samples,em_percent=100*sum(c["quality"][ARM]["answer_em"] for c in cases)/len(cases),
        f1_percent=100*sum(c["quality"][ARM]["answer_token_f1"] for c in cases)/len(cases),
        truncated_count=sum(c["quality"][ARM]["truncated"] for c in cases))},
        actual_rates_by_arm={ARM:totals},fresh_paired_baseline=False,historical_results_used_for_admission=False,
        statistical_significance_claim=False)


def validate_closed_run(run,preflight,qual,smoke,closures,errors,directory,evidence):
    sealed(run)
    require(not errors and len(closures)==len(CASE_INDICES) and qual is not None and smoke is not None
            and run.get("schema")==RUN_SCHEMA and run.get("status")=="complete"
            and run.get("completed_case_indices")==list(CASE_INDICES)
            and [c.get("index") for c in closures]==list(CASE_INDICES)
            and run.get("identity")==preflight["identity"] and run.get("manifest_sha256")==preflight["manifest_sha256"]
            and run.get("qualification")==qual and run.get("partial_smoke_sha256")==smoke["sha256"]
            and type(run.get("cases")) is list and len(run["cases"])==len(CASE_INDICES),
            "final run does not close all four new candidate cases")
    for ref,closure in zip(run["cases"],closures):
        require(ref.get("index")==closure["index"],"final candidate case order changed")
        verify_reference(ref,Path(directory)/f"case{closure['index']}.candidate.json",closure,evidence,extras=("index",))
    return True


def audit(directory,case_index=None):
    directory=Path(directory)
    require(case_index is None or type(case_index) is int and case_index in CASE_INDICES,"unknown subset case")
    evidence,errors,results,closures={},[],[],[]
    preflight=read(directory/"preflight.json",evidence)
    require(preflight.get("schema")==RUN_SCHEMA and preflight.get("protocol")==headtail_link_protocol()
            and preflight.get("evaluation_case_indices")==list(CASE_INDICES),"foreign headtail preflight/protocol")
    audit_native_swa_configuration(preflight)
    source_before=frozen_sources(preflight)
    original=read(directory/"input_manifest.json",evidence)
    require(original["sha256"]==preflight["manifest_sha256"]==preflight["protocol"]["input_manifest_sha256"],
            "fixed input/gold manifest differs")
    by_index={r["index"]:r for r in original["cohort"]}
    rows=preflight.get("audit_cases")
    require(type(rows) is list and [r.get("index") for r in rows]==list(CASE_INDICES)
            and [r.get("dataset") for r in rows]==["hotpotqa","2wikimqa"]*2,"fixed four-case cohort changed")
    for row in rows:
        require(type(row.get("input_tokens")) is int and row["input_tokens"]>=262145,"8x32K input required")
        validate_digest(row.get("input_ids_sha256"),"case input")
        source=by_index[row["index"]]
        require(all(row[k]==source[k] for k in row),"preflight source/gold changed")
        ids=source["input_ids"]
        require(type(ids) is list and len(ids)==row["input_tokens"] and all(type(t) is int and t>=0 for t in ids)
                and digest(ids)==row["input_ids_sha256"],"raw input token hash mismatch")
        for lo,hi,key in documents(row):require(digest(ids[lo:hi])==key,"document token/span hash mismatch")
    qual=qualification(directory,preflight,evidence) if (directory/"qualification.json").exists() else None
    smoke=smoke_summary=None
    if (directory/"partial_smoke.json").exists():
        require(qual is not None,"smoke exists before fresh native qualification")
        smoke,smoke_summary=partial_smoke(directory,preflight,qual,evidence)
    requested=rows if case_index is None else [r for r in rows if r["index"]==case_index]
    for row in requested:
        if not (directory/f"case{row['index']}.candidate.json").exists():continue
        require(qual is not None and smoke is not None,"candidate case precedes qualified full-input smoke")
        try:
            closure,result=audit_case(directory,row,preflight,qual,smoke,evidence)
            closures.append(closure);results.append(result)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            errors.append(dict(index=row["index"],error=str(exc)))
    complete=False
    if case_index is None and (directory/"run.json").exists():
        complete=validate_closed_run(read(directory/"run.json",evidence),preflight,qual,smoke,closures,errors,directory,evidence)
    subset=case_index is not None and not errors and len(results)==1
    status="invalid" if errors else "complete" if complete else "subset_complete" if subset else "incomplete"
    require(frozen_sources(preflight)==source_before,"frozen source changed during independent audit")
    output=dict(schema=SCHEMA,status=status,frozen_sources_verified=source_before,
        completed_case_indices=[r["index"] for r in results],complete_cases=len(results),planned_cases=len(CASE_INDICES),
        audit_scope="all_registered_cases" if case_index is None else "explicit_single_case_subset",
        requested_case_indices=[r["index"] for r in requested],final_run_audited=complete,overall_run_complete=complete,
        all_registered_cases_audited=complete,qualification_passed=qual is not None,partial_smoke=smoke_summary,
        cases=results,errors=errors,evidence=evidence,summary=summarize_cases(results),
        by_dataset={d:summarize_cases([c for c in results if c["dataset"]==d]) for d in ("hotpotqa","2wikimqa")},
        manifest_sha256=preflight["manifest_sha256"],execution_configuration_sha256=preflight["identity"]["execution_configuration_sha256"],
        protocol_sha256=preflight["protocol"]["sha256"],cuda_initialized=False,
        fresh_paired_baseline=False,historical_results_used_for_admission=False,
        exploratory_previously_exposed_inputs=True,statistical_generalization=False,cold_start_claim=False,
        no_flops_fraction_claim=True,c4_measured=False)
    return dict(output,sha256=digest(output))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--case-index",type=int,choices=CASE_INDICES)
    args=parser.parse_args(argv)
    result=audit(args.result_dir,args.case_index)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as stream:stream.write(payload)
    else:print(payload,end="")
    return 0 if result["status"] in ("complete","subset_complete") and not result["errors"] else 1


if __name__=="__main__":raise SystemExit(main())
