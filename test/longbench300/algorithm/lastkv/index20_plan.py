"""CPU geometry for arbitrary index-selected query rows with complete SWA reads.

This module does not choose scores. It validates a frozen absolute-position set
supplied by the third-layer selector and preserves mandatory tail/new rows.
"""
from __future__ import annotations

import hashlib
import json

WINDOW = TAIL = 128


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def normalize_documents(documents, total):
    require(type(total) is int and total > 0, "positive input length required")
    result = []
    previous = 0
    for d in documents:
        start, end = d.get("start"), d.get("end")
        require(type(start) is type(end) is int and previous <= start < end <= total,
                "cache documents must be sorted, disjoint, and in bounds")
        require(end-start >= 256, "short documents must be noncache full-compute rows")
        require(type(d.get("key")) is str and d["key"], "cache document key missing")
        result.append(dict(start=start, end=end, key=d["key"]))
        previous = end
    return result


def mandatory_selection(documents, total):
    docs = normalize_documents(documents, total)
    covered = {p for d in docs for p in range(d["start"], d["end"])}
    tails = {p for d in docs for p in range(d["end"]-TAIL, d["end"])}
    nondoc = set(range(total))-covered
    return sorted(tails | nondoc)


def validate_selection(documents, total, selected):
    docs = normalize_documents(documents, total)
    selected = list(selected)
    require(all(type(p) is int and 0 <= p < total for p in selected),
            "selected positions must be in-bounds Python integers")
    require(selected == sorted(set(selected)), "selected positions must be sorted and unique")
    require(set(mandatory_selection(docs, total)) <= set(selected),
            "index selection dropped a mandatory tail/new/short-document position")
    require(selected[-1] == total-1, "sampler final prompt token must remain selected")
    return selected


def required_offline_positions(documents, total, selected):
    """The exact union of native SWA windows minus online query positions."""
    docs = normalize_documents(documents, total)
    selected = validate_selection(docs, total, selected)
    selected_set = set(selected)
    needed = set()
    for p in selected:
        needed.update(range(max(0, p-WINDOW+1), p+1))
    offline = needed-selected_set
    covered = {p for d in docs for p in range(d["start"], d["end"])}
    require(offline <= covered, "uncached non-document dependency must be selected online")
    return sorted(offline)


def make_plan(documents, total, selected, start, end):
    docs = normalize_documents(documents, total)
    selected = validate_selection(docs, total, selected)
    require(type(start) is type(end) is int and 0 <= start < end <= total,
            "invalid execution frame")
    positions = [p for p in selected if start <= p < end]
    selected_set = set(selected)
    tails = {p for d in docs for p in range(d["end"]-TAIL, d["end"])}
    covered = {p for d in docs for p in range(d["start"], d["end"])}
    nondoc = set(range(total))-covered
    missing = set()
    for p in positions:
        missing.update(set(range(max(0, p-WINDOW+1), p+1))-selected_set)
    require(missing <= covered, "offline dependency outside admitted document")
    old_start = max(start, min(end-1, max(0, total-140)))
    output_positions = list(range(old_start, end))
    row_of = {p:i for i,p in enumerate(positions)}
    pairs = [(i, row_of[p]) for i,p in enumerate(output_positions) if p in row_of]
    value = dict(schema="index20-chunk-plan-v1", chunk_start=start, chunk_end=end,
        full_input_tokens=total, full_rows=end-start, selected_positions=positions,
        selected_local_rows=[p-start for p in positions], selected_rows=len(positions),
        tail_positions=[p for p in positions if p in tails],
        nondoc_positions=[p for p in positions if p in nondoc],
        index_added_positions=[p for p in positions if p not in tails and p not in nondoc],
        required_offline_halo_positions=sorted(missing),
        sampler_output_positions=output_positions,
        sampler_destination_rows=[a for a,b in pairs], sampler_source_rows=[b for a,b in pairs],
        native_local_window=128, extra_cross_document_links=False,
        selected_survive_layer39=True, native_yoco140_query_intersection=False,
        original_absolute_positions=True, unselected_deep_residual_mhc=False,
        offline_swa_source="full_normalized_document_swa_bank",
        selection_sha256=digest(selected))
    return dict(value, sha256=digest(value))


def local_memory_plan(documents, total, selected, start, end, positions,
                      previous_positions=(), offline_positions=None):
    """Return one source for every key in every selected query's causal window."""
    selected = validate_selection(documents, total, selected)
    expected = [p for p in selected if start <= p < end]
    require(list(positions) == expected, "current query positions differ from frozen selection")
    previous_positions = list(previous_positions)
    require(previous_positions == sorted(set(previous_positions)), "previous positions order")
    require(all(max(0, start-127) <= p < start and p in set(selected)
                for p in previous_positions), "previous online position outside native dependency tail")
    if offline_positions is None:
        offline_positions = required_offline_positions(documents, total, selected)
    require(list(offline_positions) == sorted(set(offline_positions)), "offline source order")
    require(not set(offline_positions) & set(selected), "offline source shadows online selected rows")
    halo = {p:i for i,p in enumerate(offline_positions)}
    current = {p:i for i,p in enumerate(positions)}
    previous = {p:i for i,p in enumerate(previous_positions)}
    needed = set()
    for p in positions:
        needed.update(range(max(0, p-WINDOW+1), p+1))
    memory = sorted(needed)
    kinds, rows = [], []
    for p in memory:
        if p in current: kind, row = "current", current[p]
        elif p in previous: kind, row = "previous", previous[p]
        elif p in halo: kind, row = "halo", halo[p]
        else: raise ValueError("missing native SWA dependency at absolute position "+str(p))
        kinds.append(kind); rows.append(row)
    result = dict(chunk_start=start, chunk_end=end, query_positions=list(positions),
        memory_positions=memory, source_kinds=kinds, source_rows=rows,
        native_window_size=128, extra_nonlocal_swa_links=False,
        local_lengths=[min(128,p+1) for p in positions],
        offline_swa_source="full_normalized_document_swa_bank",
        selection_sha256=digest(selected), all_dependencies_covered=True)
    return dict(result, sha256=digest(result))


def all_chunks(documents, total, selected, chunk_size=8192):
    require(type(chunk_size) is int and chunk_size > 0, "chunk size must be positive")
    return [make_plan(documents,total,selected,a,min(a+chunk_size,total))
            for a in range(0,total,chunk_size)]
