"""Pure-CPU contract for native local SWA plus completed-document tail links.

The native local window (including YOCO clipping) is retained. Additional
document tails may cross that clipping boundary, but queries are never revived.
The index width depends on the window plus document tails, never prompt length.
"""
from __future__ import annotations

import hashlib
import json

from .tail_link_plan import normalize_documents

SCHEMA = "headtail-link-plan-v1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def native_positions(chunk_start, chunk_end, full_input_tokens, layer, yoco_keep_tokens=140):
    """CP1 native compact layout, including the one-row early-chunk placeholder."""
    for value, name in ((chunk_start, "start"), (chunk_end, "end"),
                        (full_input_tokens, "tokens"), (layer, "layer")):
        _integer(value, name)
    _integer(yoco_keep_tokens, "keep", 1)
    if not 0 <= chunk_start < chunk_end <= full_input_tokens or layer >= 40:
        raise ValueError("invalid native chunk/layer")
    first = chunk_start
    if layer >= 20:
        first = max(chunk_start, min(chunk_end - 1, full_input_tokens - yoco_keep_tokens))
    return list(range(first, chunk_end))


def plan(documents, current_positions, chunk_start, chunk_end, full_input_tokens, *, layer,
         head_tokens=128, tail_tokens=128, window_size=128, yoco_keep_tokens=140,
         native_valid_start=0):
    for value, name in ((head_tokens, "head"), (tail_tokens, "tail"),
                        (window_size, "window"), (yoco_keep_tokens, "keep")):
        _integer(value, name, 1)
    _integer(native_valid_start, "native valid start")
    if native_valid_start > chunk_start:
        raise ValueError("native valid start exceeds chunk start")
    spans = normalize_documents(documents, full_input_tokens, tail_tokens)
    positions = list(current_positions)
    if any(type(p) is not int for p in positions) or positions != native_positions(
            chunk_start, chunk_end, full_input_tokens, layer, yoco_keep_tokens):
        raise ValueError("current positions differ from actual native compact layout")
    for doc in spans:
        doc["head_end"] = min(doc["end"], doc["start"] + head_tokens)
    selected, heads, tails, nondoc, inactive = [], [], [], [], []
    for row, position in enumerate(positions):
        doc = next((d for d in spans if d["start"] <= position < d["end"]), None)
        if doc is None:
            nondoc.append(row)
            selected.append(row)
        else:
            head, tail = position < doc["head_end"], position >= doc["tail_start"]
            if head:
                heads.append(row)
            if tail:
                tails.append(row)
            (selected if head or tail else inactive).append(row)
    bank = [(p, doc["end"], doc["occurrence"], p - doc["start"])
            for doc in spans for p in range(doc["tail_start"], doc["end"])]
    rect = max(0, full_input_tokens - yoco_keep_tokens) if layer >= 20 else 0
    lows, lengths, completed, extras = [], [], [], []
    for query in positions:
        low = min(query + 1, max(0, query - window_size + 1, rect, native_valid_start))
        lows.append(low)
        lengths.append(query - low + 1)
        completed.append(sum(d["end"] <= query for d in spans))
        # Eight interval intersections, never query x 1024 Python iteration.
        extras.append(sum(d["tail_rows"] - max(0, min(query + 1, d["end"]) - max(low, d["tail_start"]))
                          for d in spans if d["end"] <= query))
    offset = window_size + len(positions)
    result = dict(
        schema=SCHEMA, layer=layer, chunk_start=chunk_start, chunk_end=chunk_end,
        full_input_tokens=full_input_tokens, head_tokens=head_tokens, tail_tokens=tail_tokens,
        native_window_size=window_size, native_valid_start=native_valid_start,
        native_yoco_rect_lo=rect, yoco_keep_tokens=yoco_keep_tokens,
        documents=spans, current_positions=positions, full_rows=len(positions),
        selected_local_rows=selected, selected_positions=[positions[i] for i in selected],
        selected_rows=len(selected), document_head_local_rows=heads,
        document_tail_local_rows=tails, mandatory_nondoc_local_rows=nondoc,
        inactive_local_rows=inactive, native_window_lows=lows, native_window_lengths=lengths,
        completed_document_counts=completed, extra_tail_lengths=extras,
        union_lengths=[a + b for a, b in zip(lengths, extras)],
        tail_bank_positions=[p for p, _, _, _ in bank],
        tail_bank_document_ends=[end for _, end, _, _ in bank],
        tail_bank_document_indices=[i for _, _, i, _ in bank],
        tail_bank_document_offsets=[p for _, _, _, p in bank],
        tail_extra_offset=offset, global_index_offset=offset + len(bank),
        index_width=max(128, ((window_size + len(bank) + 127) // 128) * 128),
        extra_links_cross_yoco_rectangle=True, native_yoco_surviving_positions_preserved=True,
        native_local_mask_preserved=True,
        local_extra_deduplicated=True, completed_document_rule="document_end_le_query_position",
    )
    layout = {k: v for k, v in result.items() if k != "layer"}
    result["layout_sha256"] = _digest(layout)
    return rebind_layer(result, layer)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rebind_layer(event, layer):
    """Share immutable layout arrays across layers without repeated planning."""
    _integer(layer, "layer")
    if (layer >= 20) != (event["layer"] >= 20) or layer >= 40:
        raise ValueError("cannot rebind across native YOCO layouts")
    result = dict(event, layer=layer)
    result["plan_sha256"] = _digest(dict(layout_sha256=result["layout_sha256"], layer=layer))
    return result


def query_key_positions(event, query):
    """Independent scalar union in native-then-extra column order."""
    if query not in event["current_positions"]:
        raise ValueError("query is not a native survivor")
    low = min(query + 1, max(0, query - event["native_window_size"] + 1,
                           event["native_valid_start"], event["native_yoco_rect_lo"]))
    local = list(range(low, query + 1))
    extra = [p for p, end in zip(event["tail_bank_positions"], event["tail_bank_document_ends"])
             if end <= query and p not in local]
    return local + extra


def materialize_indices(event, queries=None):
    """Small fixtures / bounded qualification batches, never the timed path."""
    current = event["current_positions"]
    rows = {p: i for i, p in enumerate(current)}
    start, window = event["chunk_start"], event["native_window_size"]
    result = []
    for query in current if queries is None else queries:
        if query not in rows:
            raise ValueError("query is not a native survivor")
        low = min(query + 1, max(0, query - window + 1,
                               event["native_valid_start"], event["native_yoco_rect_lo"]))
        native = []
        for p in range(low, query + 1):
            if p < start:
                slot = p - (start - window)
                if not 0 <= slot < window:
                    raise ValueError("native historical key outside ring view")
            else:
                if p not in rows:
                    raise ValueError("native local key removed by YOCO")
                slot = window + rows[p]
            native.append(slot)
        output = native + [-1] * (window - len(native))
        for slot, (p, end) in enumerate(zip(event["tail_bank_positions"], event["tail_bank_document_ends"])):
            output.append(event["tail_extra_offset"] + slot if end <= query and not low <= p <= query else -1)
        output.extend([-1] * (event["index_width"] - len(output)))
        result.append(output)
    return result


def audit_plan_result(result, documents, current_positions, chunk_start, chunk_end,
                      full_input_tokens, *, layer, **kwargs):
    """Convenience CPU check. Formal reader implements its own scalar oracle."""
    expected = plan(documents, current_positions, chunk_start, chunk_end,
                    full_input_tokens, layer=layer, **kwargs)
    errors = [key for key in expected if result.get(key) != expected[key]]
    return dict(passed=not errors, errors=errors)
