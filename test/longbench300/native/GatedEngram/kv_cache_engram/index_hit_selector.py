"""CPU-only specification for B's raw index-frequency ranking (not accuracy)."""
from collections import Counter
from .core import require


def select_by_hitcounts(hitcounts, positions, document_spans, *, all_selected=False):
    require(len(hitcounts) == len(positions), "score/position length mismatch")
    require(all(type(x) is int and x >= 0 for x in hitcounts), "nonnegative integer hits required")
    require(all(type(p) is int and p >= 0 for p in positions)
            and all(a < b for a, b in zip(positions, positions[1:])), "positions must be strictly increasing")
    require(all(type(a) is int and type(b) is int and 0 <= a < b for a,b in document_spans), "document intervals invalid")
    require(all(a[1] <= b[0] for a,b in zip(document_spans, document_spans[1:])), "overlapping/unordered documents")
    optional = [i for i,p in enumerate(positions) if any(a <= p < b for a,b in document_spans)]
    optional_set = set(optional)
    mandatory = [i for i,p in enumerate(positions) if not any(a <= p < b for a,b in document_spans)]
    chosen = list(range(len(positions))) if all_selected else sorted(
        mandatory + sorted(optional, key=lambda i: (-hitcounts[i], positions[i]))[:(len(optional)+4)//5])
    return dict(selected_local_rows=chosen, optional_rows=len(optional), mandatory_rows=len(mandatory),
                zero_hit_document_rows=sum(hitcounts[i] == 0 for i in optional),
                selected_zero_hit_document_rows=sum(hitcounts[i] == 0 for i in chosen if i in optional_set))


def raw_index_hitcounts(index_rows, query_positions, target_positions, *, ratio, global_kv_rows, request_offset=0):
    """C1/C2 non-overlap groups are aligned to the whole request, not documents.

    Native prefill's index values are flattened global-array columns. Single
    request CP1 has offset zero. Each query votes once for each valid selected
    column; the vote goes to every original position represented by that group.
    """
    require(ratio in (1,2) and type(global_kv_rows) is int and global_kv_rows >= 0
            and type(request_offset) is int and request_offset >= 0, "invalid compression geometry")
    require(len(index_rows) == len(query_positions), "index/query count mismatch")
    counts = Counter()
    for row, p in zip(index_rows, query_positions):
        require(type(p) is int and p >= 0, "invalid query position")
        seen = set()
        for value in row:
            require(type(value) is int, "integer top-k index required")
            if value == -1:
                continue
            key = value-request_offset
            require(0 <= key < global_kv_rows and key < (p+1)//ratio, "invalid or noncausal compressed index")
            require(key not in seen, "duplicate column inside one query top-k")
            seen.add(key)
            counts[key] += 1
    require(all(type(p) is int and p >= 0 for p in target_positions), "invalid target position")
    return [counts[p//ratio] for p in target_positions]
