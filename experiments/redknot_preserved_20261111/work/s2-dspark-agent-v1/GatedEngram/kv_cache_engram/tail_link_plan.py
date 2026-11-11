"""CPU-only absolute-position contract for the independent tail-link experiment.

This is NOT native sliding-window attention or B20. Document-tail memory has
explicit nonlocal visibility. Compact KV slots are never RoPE positions. No
torch/model/GPU imports are permitted here so readers can use the independent
oracle below without loading a model.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Mapping


SCHEMA = "kce-tail-link-plan-v1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positions(values, name):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    out = [_integer(value, name) for value in values]
    if any(left >= right for left, right in zip(out, out[1:])):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return out


def normalize_documents(documents, full_input_tokens, tail_tokens=128):
    """Return occurrence-aware spans; identical content at disjoint spans is legal."""
    _integer(full_input_tokens, "full_input_tokens", 1)
    _integer(tail_tokens, "tail_tokens", 1)
    if not isinstance(documents, (list, tuple)) or not documents:
        raise ValueError("at least one document span is required")
    result = []
    previous_end = 0
    for occurrence, doc in enumerate(documents):
        if not isinstance(doc, Mapping):
            raise ValueError("document must be a mapping")
        start = _integer(doc.get("start"), "document start")
        end = _integer(doc.get("end"), "document end", 1)
        if start >= end or end > full_input_tokens or start < previous_end:
            raise ValueError("document spans must be ordered, nonempty, disjoint and in bounds")
        key = doc.get("key")
        if key is not None and not isinstance(key, str):
            raise ValueError("document key must be a string or None")
        result.append(dict(occurrence=occurrence, key=key, start=start, end=end,
                           tail_start=max(start, end - tail_tokens),
                           tail_end=end, tail_rows=min(end - start, tail_tokens)))
        previous_end = end
    return result


def tail_bank_rows(documents, full_input_tokens, tail_tokens=128):
    """All occurrence-specific tail positions, independent of the current chunk."""
    spans = normalize_documents(documents, full_input_tokens, tail_tokens)
    rows = []
    for doc in spans:
        for position in range(doc["tail_start"], doc["end"]):
            rows.append(dict(position=position, document_index=doc["occurrence"],
                             document_key=doc["key"],
                             document_token_offset=position - doc["start"],
                             tail_bank_row=position - doc["tail_start"]))
    return rows


def _inputs(documents, current_positions, chunk_start, chunk_end, full_input_tokens,
            tail_tokens, available_nondoc_positions):
    spans = normalize_documents(documents, full_input_tokens, tail_tokens)
    _integer(chunk_start, "chunk_start")
    _integer(chunk_end, "chunk_end")
    if not chunk_start <= chunk_end <= full_input_tokens:
        raise ValueError("chunk bounds are invalid")
    positions = _positions(current_positions, "current_positions")
    if any(not chunk_start <= value < chunk_end for value in positions):
        raise ValueError("native surviving positions must lie in the current chunk")
    available = None
    if available_nondoc_positions is not None:
        available = _positions(available_nondoc_positions, "available_nondoc_positions")
        if any(value >= chunk_end for value in available):
            raise ValueError("available nondocument KV cannot be from future chunks")
        if any(doc["start"] <= value < doc["end"] for value in available for doc in spans):
            raise ValueError("available_nondoc_positions contains a document position")
    return spans, positions, available


def plan(documents, current_positions, chunk_start, chunk_end, full_input_tokens,
         tail_tokens=128, available_nondoc_positions=None):
    """Plan actual native surviving queries without forcing omitted YOCO rows.

    ``available_nondoc_positions=None`` describes the full mathematical memory.
    A controller must instead supply the sorted union of its actually generated
    historical and current nondocument KV positions; missing KV is not zero KV.
    Such an explicit availability set is shared by sparse and dense-reference
    executions and recorded in the report. All actual nondocument queries are
    mandatory, whether or not their KV has already been generated.

    Sorted memory slots make each query's valid indices a prefix. Length arrays
    avoid allocating a CPU [queries, memory] matrix on every layer. Device code
    may construct arange(memory_count), masking columns >= each length to -1.
    """
    spans, positions, available = _inputs(
        documents, current_positions, chunk_start, chunk_end, full_input_tokens,
        tail_tokens, available_nondoc_positions)
    starts = [doc["start"] for doc in spans]

    def owner(position):
        index = bisect.bisect_right(starts, position) - 1
        return index if index >= 0 and position < spans[index]["end"] else -1

    selected, mandatory, tail_selected, inactive = [], [], [], []
    for local, position in enumerate(positions):
        doc_index = owner(position)
        if doc_index < 0:
            selected.append(local)
            mandatory.append(local)
        elif position >= spans[doc_index]["tail_start"]:
            selected.append(local)
            tail_selected.append(local)
        else:
            inactive.append(local)
    last_query = positions[-1] if positions else -1
    candidates = []
    for doc in spans:
        candidates.extend(range(doc["tail_start"], min(doc["end"], last_query + 1)))
    if available is None:
        cursor = 0
        for doc in spans:
            candidates.extend(range(cursor, min(doc["start"], last_query + 1)))
            cursor = doc["end"]
        candidates.extend(range(cursor, min(full_input_tokens, last_query + 1)))
    else:
        candidates.extend(available[:bisect.bisect_right(available, last_query)])
    memory = sorted(candidates)
    if len(set(memory)) != len(memory):
        raise AssertionError("memory construction produced duplicate absolute positions")
    local_by_position = {position: local for local, position in enumerate(positions)}
    doc_indices, doc_offsets, bank_rows, kinds, memory_local = [], [], [], [], []
    for position in memory:
        index = owner(position)
        doc_indices.append(index)
        doc_offsets.append(position - spans[index]["start"] if index >= 0 else -1)
        bank_rows.append(position - spans[index]["tail_start"] if index >= 0 else -1)
        kinds.append("document_tail" if index >= 0 else "online_nondocument")
        memory_local.append(local_by_position.get(position, -1))
    reference_lengths = [bisect.bisect_right(memory, position) for position in positions]
    result = dict(
        schema=SCHEMA, full_input_tokens=full_input_tokens,
        chunk_start=chunk_start, chunk_end=chunk_end, tail_tokens=tail_tokens,
        documents=spans, current_positions=positions,
        selected_local_rows=selected, selected_positions=[positions[i] for i in selected],
        document_tail_local_rows=tail_selected, mandatory_nondoc_local_rows=mandatory,
        inactive_local_rows=inactive, selected_rows=len(selected), full_rows=len(positions),
        memory_positions=memory, memory_count=len(memory), memory_kinds=kinds,
        memory_doc_indices=doc_indices, memory_doc_token_offsets=doc_offsets,
        memory_tail_bank_rows=bank_rows, memory_current_local_rows=memory_local,
        selected_memory_lengths=[reference_lengths[i] for i in selected],
        reference_memory_lengths=reference_lengths,
        available_nondoc_positions=available,
        nondocument_availability="all_mathematical" if available is None else "explicit_generated_only",
        compact_slots_are_absolute_positions=False, global_index_offset=len(memory),
        native_yoco_surviving_positions_preserved=True,
    )
    result["plan_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def materialize_indices(memory_lengths, memory_count, padded_width=None):
    """Small-fixture/reference utility, not the GPU production construction."""
    _integer(memory_count, "memory_count")
    width = memory_count if padded_width is None else _integer(padded_width, "padded_width")
    if width < memory_count:
        raise ValueError("padded index width is smaller than memory_count")
    output = []
    for length in memory_lengths:
        _integer(length, "memory length")
        if length > memory_count:
            raise ValueError("query memory length exceeds cache")
        output.append(list(range(length)) + [-1] * (width - length))
    return output


def audit_observed_indices(memory_positions, query_positions, observed_indices):
    """Independently check exact causal visibility, including padding and order."""
    memory = _positions(memory_positions, "memory_positions")
    queries = _positions(query_positions, "query_positions")
    errors = []
    if len(observed_indices) != len(queries):
        return dict(passed=False, errors=["query/index row count mismatch"])
    width = None
    for row, (query, values) in enumerate(zip(queries, observed_indices)):
        if not isinstance(values, (list, tuple)):
            errors.append(f"row {row}: index row must be a list or tuple")
            continue
        if width is None:
            width = len(values)
        if len(values) != width:
            errors.append(f"row {row}: ragged index matrix")
        if any(type(value) is not int or value < -1 for value in values):
            errors.append(f"row {row}: invalid index value")
            continue
        valid = [value for value in values if value != -1]
        expected = [slot for slot, position in enumerate(memory) if position <= query]
        if valid != expected or list(values) != valid + [-1] * (len(values) - len(valid)):
            errors.append(f"row {row}: wrong causal slots/order/padding")
    return dict(passed=not errors, errors=errors, queries=len(queries), memory_rows=len(memory))


def audit_plan_result(result, documents, current_positions, chunk_start, chunk_end,
                      full_input_tokens, tail_tokens=128, available_nondoc_positions=None):
    """Independent scalar oracle for reader use; never calls ``plan``.

    Validation is shared, but visibility and selection are rederived directly
    from source intervals, rather than trusting controller reports or hashes.
    """
    spans, positions, available = _inputs(
        documents, current_positions, chunk_start, chunk_end, full_input_tokens,
        tail_tokens, available_nondoc_positions)
    available_set = set(available) if available is not None else None
    selected, mandatory, tails, inactive = [], [], [], []
    for row, position in enumerate(positions):
        containing = [doc for doc in spans if doc["start"] <= position < doc["end"]]
        if not containing:
            selected.append(row)
            mandatory.append(row)
        elif containing[0]["end"] - position <= tail_tokens:
            selected.append(row)
            tails.append(row)
        else:
            inactive.append(row)
    memory, indices, offsets, bank_rows, kinds = [], [], [], [], []
    for position in range(max(positions) + 1 if positions else 0):
        containing = [(i, doc) for i, doc in enumerate(spans) if doc["start"] <= position < doc["end"]]
        if containing:
            index, doc = containing[0]
            if doc["end"] - position > tail_tokens:
                continue
            indices.append(index)
            offsets.append(position - doc["start"])
            bank_rows.append(position - max(doc["start"], doc["end"] - tail_tokens))
            kinds.append("document_tail")
        else:
            if available_set is not None and position not in available_set:
                continue
            indices.append(-1)
            offsets.append(-1)
            bank_rows.append(-1)
            kinds.append("online_nondocument")
        memory.append(position)
    lengths = [sum(position <= query for position in memory) for query in positions]
    expected = dict(
        schema=SCHEMA, full_input_tokens=full_input_tokens, chunk_start=chunk_start,
        chunk_end=chunk_end, tail_tokens=tail_tokens, documents=spans,
        current_positions=positions, selected_local_rows=selected,
        selected_positions=[positions[row] for row in selected],
        document_tail_local_rows=tails, mandatory_nondoc_local_rows=mandatory,
        inactive_local_rows=inactive, selected_rows=len(selected), full_rows=len(positions),
        memory_positions=memory, memory_count=len(memory), memory_kinds=kinds,
        memory_doc_indices=indices, memory_doc_token_offsets=offsets,
        memory_tail_bank_rows=bank_rows,
        memory_current_local_rows=[positions.index(position) if position in positions else -1 for position in memory],
        selected_memory_lengths=[lengths[row] for row in selected], reference_memory_lengths=lengths,
        available_nondoc_positions=available,
        nondocument_availability="all_mathematical" if available is None else "explicit_generated_only",
        compact_slots_are_absolute_positions=False, global_index_offset=len(memory),
        native_yoco_surviving_positions_preserved=True,
    )
    def same_typed(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(right, list):
            return len(left) == len(right) and all(same_typed(a, b) for a, b in zip(left, right))
        if isinstance(right, dict):
            return left.keys() == right.keys() and all(same_typed(left[key], right[key]) for key in right)
        return left == right

    errors = [f"{key}: independent oracle mismatch" for key, value in expected.items()
              if not same_typed(result.get(key), value)]
    payload = {key: value for key, value in result.items() if key != "plan_sha256"}
    actual_hash = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if result.get("plan_sha256") != actual_hash:
        errors.append("plan_sha256: digest mismatch")
    return dict(schema="kce-tail-link-plan-audit-v1", passed=not errors, errors=errors,
                checked_query_rows=len(positions), selected_rows=len(selected), memory_rows=len(memory))


def canonical_bank_bytes(documents, full_input_tokens, layers=40, tail_tokens=128,
                         head_dim=512, bytes_per_element=2):
    """Logical canonical bytes only; no GPU replication/scale/staging overhead."""
    spans = normalize_documents(documents, full_input_tokens, tail_tokens)
    for value, name in ((layers, "layers"), (head_dim, "head_dim"),
                        (bytes_per_element, "bytes_per_element")):
        _integer(value, name, 1)
    rows = sum(doc["tail_rows"] for doc in spans)
    full_rows = sum(doc["end"] - doc["start"] for doc in spans)
    per_row = head_dim * bytes_per_element
    return dict(tail_rows=rows, full_document_rows=full_rows, layers=layers,
                bytes_per_row=per_row, tail_bytes_per_layer=rows * per_row,
                full_bytes_per_layer=full_rows * per_row,
                tail_bank_bytes=rows * per_row * layers,
                full_bank_bytes=full_rows * per_row * layers,
                logical_only=True, excludes_replication_staging_metadata=True)


def audit_decode_memory_coverage(memory_positions, full_input_tokens, window_tokens=128):
    """Check logical final-window coverage, NOT physical ring writes or values.

    The runtime must separately prove slot mapping, the actual stored KV and
    native backtracking/replay requirements (which may exceed window_tokens).
    """
    _integer(full_input_tokens, "full_input_tokens", 1)
    _integer(window_tokens, "window_tokens", 1)
    positions = _positions(memory_positions, "memory_positions")
    if any(position >= full_input_tokens for position in positions):
        raise ValueError("decode memory contains a position beyond the input")
    expected = list(range(max(0, full_input_tokens - window_tokens), full_input_tokens))
    present = set(positions)
    missing = [position for position in expected if position not in present]
    return dict(passed=not missing, required_positions=expected, missing_positions=missing,
                logical_only=True, physical_ring_values_verified=False,
                backtracking_safety_verified=False)
