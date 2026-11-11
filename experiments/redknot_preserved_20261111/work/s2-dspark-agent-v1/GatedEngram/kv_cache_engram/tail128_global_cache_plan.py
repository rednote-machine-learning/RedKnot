"""CPU contracts for mixed online/offline global KV, not a GPU implementation.

C2 groups are aligned to absolute request positions, never document starts.
The bank stores token projections before compression so odd document offsets
and groups containing both online and offline members remain representable.
Native source 2 belongs to the unchanged first-three-layer prepass. Sources
8/14/20 are the only later main-KV and index-K writers for this checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math

from .tail_link_plan import normalize_documents

SCHEMA = "tail128-global-cache-plan-v1"
BANK_SCHEMA = "tail128-global-canonical-bank-layout-v1"
MAIN_SOURCES = (2, 8, 14, 20)
INDEX_QUERY_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
ONLINE_MAIN_SOURCES = (8, 14, 20)
MAIN_DIM = 512
INDEX_DIM = 128


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _positions(values, name, total):
    values = list(values)
    if any(type(p) is not int or not 0 <= p < total for p in values):
        raise ValueError(name + " contains an invalid absolute position")
    if values != sorted(set(values)):
        raise ValueError(name + " must be sorted and unique")
    return values


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def topology_for_layer(layer):
    """Fixed checkpoint ownership; index-query sources are not index-K sources."""
    _int(layer, "layer")
    if layer >= 40:
        raise ValueError("layer outside the forty-layer model")
    main = max((s for s in MAIN_SOURCES if s <= layer), default=None)
    index_query = max((s for s in INDEX_QUERY_SOURCES if s <= layer), default=None)
    return dict(layer=layer, main_kv_source_layer=main, index_k_source_layer=main,
                index_query_source_layer=index_query,
                compression_ratio=0 if main is None else (1 if main == 20 else 2),
                writes_main_kv=layer in MAIN_SOURCES,
                writes_index_k=layer in MAIN_SOURCES,
                recomputes_index_queries=layer in INDEX_QUERY_SOURCES,
                main_cache_from_native_prepass=main == 2,
                main_hidden_source=(None if main is None else
                    "source20_boundary_attn_normalized_bf16" if main == 20 else
                    "current_source_attn_normalized_bf16"))


def canonical_bank_layout(documents, full_input_tokens, *, scope_sha256):
    """Specify required raw arrays; their values still need a real native capture.

    ``scope_sha256`` must bind checkpoint, source weights, native source, dtype,
    norm/configuration and capture policy. This function validates its encoding,
    not the external evidence behind that digest. Repeated content occurrences
    may share a raw entry, but never share absolute-position cache placement.
    """
    _sha(scope_sha256, "scope_sha256")
    docs = normalize_documents(documents, full_input_tokens)
    lengths = {}
    entries = []
    for doc in docs:
        key = _sha(doc["key"], "document key")
        n = doc["end"] - doc["start"]
        if key in lengths:
            if lengths[key] != n:
                raise ValueError("same document digest has inconsistent lengths")
            continue
        lengths[key] = n
        for source in ONLINE_MAIN_SOURCES:
            c2 = source < 20
            width, element_bytes = (2 * MAIN_DIM, 4) if c2 else (MAIN_DIM, 2)
            entries.append(dict(document_sha256=key, source_layer=source,
                scope_sha256=scope_sha256, rows=n, shape=[n, width],
                dtype="float32" if c2 else "bfloat16",
                representation="gate_then_latent_precompression" if c2 else "latent_pre_norm",
                gate_slice=[0, MAIN_DIM] if c2 else None,
                latent_slice=[MAIN_DIM, 2 * MAIN_DIM] if c2 else [0, MAIN_DIM],
                bytes=n * width * element_bytes,
                offline_coordinates="document_local_token_positions",
                no_rope_no_cache_quantization=True))
    result = dict(schema=BANK_SCHEMA, documents=docs, full_input_tokens=full_input_tokens,
        scope_sha256=scope_sha256, entries=entries, entry_count=len(entries),
        raw_bank_bytes=sum(e["bytes"] for e in entries),
        source_layers=list(ONLINE_MAIN_SOURCES), native_prepass_source_layer=2,
        native_prepass_main_and_index_k_preserved=True,
        source2_offline_override_allowed=False, offline_raw_bank_immutable=True,
        compression_alignment="absolute_request_position_mod_ratio",
        local_precompressed_rows_not_relocated=True,
        position_dependent_cache_is_phase_local=True,
        main_norm_then_main_rope_then_native_cache_cast=True,
        index_k_from_main_postnorm_prerope=True,
        index_k_projection_owner_matches_main_source=True,
        index_norm_then_compressed_position_rope_then_fp4_ue8m0=True)
    result["sha256"] = _hash(result)
    return result


def source_update_plan(documents, full_input_tokens, selected_positions, *, layer,
                       phase_generation, chunk_start=0, chunk_end=None,
                       current_projection_positions=None,
                       previous_projection_positions=(), previous_projection_scope=None):
    """Plan complete groups and member provenance at one actual source layer.

    Online positions refer to this source's raw projections, not hidden from
    another layer and not already-compressed KV. ``previous`` is same-phase,
    same-source carry from an earlier chunk. A selected member may never fall
    back silently to offline data. An incomplete final C2 group is pending and
    must not become a visible compressed key.
    """
    docs = normalize_documents(documents, full_input_tokens)
    _int(phase_generation, "phase_generation", 1)
    topo = topology_for_layer(layer)
    if layer not in ONLINE_MAIN_SOURCES:
        raise ValueError("only sources 8/14/20 update mixed global KV; source2 stays native")
    if chunk_end is None:
        chunk_end = full_input_tokens
    _int(chunk_start, "chunk_start")
    _int(chunk_end, "chunk_end", 1)
    if not chunk_start < chunk_end <= full_input_tokens:
        raise ValueError("invalid source chunk")
    selected = _positions(selected_positions, "selected_positions", full_input_tokens)
    selected_set = set(selected)
    owner = {}
    for doc in docs:
        _sha(doc["key"], "document key")
        for pos in range(doc["start"], doc["end"]):
            owner[pos] = doc
    nondoc = [p for p in range(full_input_tokens) if p not in owner]
    if not set(nondoc).issubset(selected_set):
        raise ValueError("all nondocument positions must be selected; no offline nondoc bank")
    required_current = [p for p in selected if chunk_start <= p < chunk_end]
    current = required_current if current_projection_positions is None else _positions(
        current_projection_positions, "current_projection_positions", full_input_tokens)
    previous = _positions(previous_projection_positions, "previous_projection_positions", full_input_tokens)
    if current != required_current:
        raise ValueError("current projection positions must exactly cover selected rows in this chunk")
    if any(p >= chunk_start or p not in selected_set for p in previous):
        raise ValueError("previous projection must be a selected position from an earlier chunk")
    expected_scope = dict(source_layer=layer, phase_generation=phase_generation)
    if previous and previous_projection_scope != expected_scope:
        raise ValueError("previous projection scope must bind the same source and phase")
    current_set, previous_set = set(current), set(previous)
    ratio = topo["compression_ratio"]

    def member(pos):
        doc = owner.get(pos)
        if pos in current_set:
            origin = "current_online_projection"
        elif pos in previous_set:
            origin = "same_phase_source_previous_projection"
        elif pos in selected_set:
            raise ValueError("selected C2 carry is absent; offline fallback is prohibited")
        else:
            origin = "immutable_offline_projection"
        if origin == "immutable_offline_projection" and doc is None:
            raise ValueError("offline projection does not cover nondocument position")
        return dict(position=pos, origin=origin, source_layer=layer,
            phase_generation=None if origin == "immutable_offline_projection" else phase_generation,
            document_occurrence=None if doc is None else doc["occurrence"],
            document_sha256=None if doc is None else doc["key"],
            document_local_row=None if doc is None else pos - doc["start"])

    blocks = []
    for compressed_id in range(chunk_start // ratio, chunk_end // ratio):
        positions = list(range(compressed_id * ratio, (compressed_id + 1) * ratio))
        members = [member(p) for p in positions]
        affected = any(p in selected_set for p in positions)
        blocks.append(dict(compressed_id=compressed_id, rope_position=compressed_id * ratio,
            causal_visible_from_query=(compressed_id + 1) * ratio - 1,
            member_positions=positions, members=members, affected=affected,
            action="overlay_recompress_mixed_members" if affected else "reuse_phase_offline_base",
            main_kv_source_layer=layer, index_k_source_layer=layer))
    pending = [member(p) for p in range(chunk_end // ratio * ratio, chunk_end)]
    result = dict(schema=SCHEMA, topology=topo, source_layer=layer,
        full_input_tokens=full_input_tokens, phase_generation=phase_generation,
        chunk_start=chunk_start, chunk_end=chunk_end, compression_ratio=ratio,
        selected_positions=selected, current_projection_positions=current,
        previous_projection_positions=previous, mandatory_nondoc_positions=nondoc,
        online_projection_scope=expected_scope,
        blocks=blocks, new_complete_compressed_ids=[b["compressed_id"] for b in blocks],
        affected_compressed_ids=[b["compressed_id"] for b in blocks if b["affected"]],
        offline_base_compressed_ids=[b["compressed_id"] for b in blocks if not b["affected"]],
        pending_incomplete_members=pending,
        visible_compressed_rows_after_chunk=chunk_end // ratio,
        source2_native_cache_unchanged=True, offline_raw_bank_immutable=True,
        phase_local_main_and_index_k_overlay=True,
        compressed_position_is_block_start=True, selected_offline_fallback_allowed=False,
        native_main_cast_layout_must_be_bound=True,
        index_k_recomputed_only_for_affected_main_blocks=True)
    result["gpu_execution_implemented"] = False
    result["numerical_qualification"] = "cpu_geometry_and_unquantized_reference_only"
    result["sha256"] = _hash(result)
    return result


def gated_compress_reference(members, *, ratio):
    """Unquantized float64 CPU mathematical oracle, NOT native byte parity.

    The native C2 kernel first computes this per-feature gated mixture, then
    rounds to BF16 before source RMSNorm/RoPE/cache conversion. Neither that
    rounding nor native projection/normalization/quantization is emulated here.
    """
    if type(ratio) is not int or ratio not in (1, 2) or len(members) != ratio:
        raise ValueError("one complete C1/C2 group required")
    width = len(members[0]["latent"])
    if not width or any(len(m["latent"]) != width for m in members):
        raise ValueError("latent widths differ or are empty")
    latent = [[float(x) for x in m["latent"]] for m in members]
    if any(not math.isfinite(x) for row in latent for x in row):
        raise ValueError("nonfinite latent")
    if ratio == 1:
        if members[0].get("gate") is not None:
            raise ValueError("C1 has no gate")
        return list(latent[0])
    if any(m.get("gate") is None or len(m["gate"]) != width for m in members):
        raise ValueError("C2 requires a gate for each feature/member")
    gates = [[float(x) for x in m["gate"]] for m in members]
    if any(not math.isfinite(x) for row in gates for x in row):
        raise ValueError("nonfinite gate")
    result = []
    for j in range(width):
        maximum = max(g[j] for g in gates)
        weights = [math.exp(g[j] - maximum) for g in gates]
        result.append(sum(w * row[j] for w, row in zip(weights, latent)) / sum(weights))
    return result
