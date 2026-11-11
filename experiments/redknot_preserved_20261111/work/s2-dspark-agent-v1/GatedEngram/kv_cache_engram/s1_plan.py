"""CPU geometry for S1: online document tail128 plus every nondocument row."""
from __future__ import annotations

from .core import digest, require
from .tail_link_plan import normalize_documents

WINDOW = TAIL = HALO = 128


def plan(documents, full_input_tokens, chunk_start, chunk_end):
    require(type(full_input_tokens) is int and full_input_tokens > 0, "positive prompt length required")
    require(type(chunk_start) is type(chunk_end) is int
            and 0 <= chunk_start < chunk_end <= full_input_tokens, "invalid S1 chunk")
    docs = normalize_documents(documents, full_input_tokens, TAIL)
    require(all(d["end"]-d["start"] >= TAIL+HALO for d in docs), "S1 blocks require at least256 tokens")
    tails = {p for d in docs for p in range(d["end"]-TAIL, d["end"])}
    covered = {p for d in docs for p in range(d["start"], d["end"])}
    nondoc = set(range(full_input_tokens))-covered
    all_selected = tails | nondoc
    positions = [p for p in range(chunk_start, chunk_end) if p in all_selected]
    halo = {p for d in docs for p in range(d["end"]-TAIL-HALO, d["end"]-TAIL)}
    required_offline = set()
    for query in positions:
        missing = set(range(max(0, query-WINDOW+1), query+1))-all_selected
        require(missing <= halo, "selected query requires uncached SWA outside declared halo128")
        required_offline.update(missing)
    # Native sampler indices refer to the old native YOCO output layout.  Only
    # the final original prompt row is sampled; keep that slot available without
    # computing any unselected hidden rows merely for output-shape compatibility.
    old_start = max(chunk_start, min(chunk_end-1, max(0, full_input_tokens-140)))
    output_positions = list(range(old_start, chunk_end))
    selected_map = {p: i for i, p in enumerate(positions)}
    output_pairs = [(j, selected_map[p]) for j, p in enumerate(output_positions) if p in selected_map]
    if chunk_end == full_input_tokens:
        require(full_input_tokens-1 in all_selected, "final prompt token must be selected")
    value = dict(schema="s1-chunk-plan-v1", chunk_start=chunk_start, chunk_end=chunk_end,
        full_input_tokens=full_input_tokens, full_rows=chunk_end-chunk_start, selected_positions=positions,
        selected_local_rows=[p-chunk_start for p in positions], selected_rows=len(positions),
        tail_positions=[p for p in positions if p in tails], nondoc_positions=[p for p in positions if p in nondoc],
        required_offline_halo_positions=sorted(required_offline),
        sampler_output_positions=output_positions, sampler_destination_rows=[a for a, _ in output_pairs],
        sampler_source_rows=[b for _, b in output_pairs],
        native_local_window=128, extra_cross_document_links=False,
        selected_survive_layer39=True, native_yoco140_query_intersection=False,
        original_absolute_positions=True, unselected_deep_residual_mhc=False)
    return dict(value, sha256=digest(value))


def all_chunks(documents, full_input_tokens):
    return [plan(documents, full_input_tokens, start, min(start+8192, full_input_tokens))
            for start in range(0, full_input_tokens, 8192)]
