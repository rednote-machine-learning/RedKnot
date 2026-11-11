"""Observational KV error collection, enabled only in an untimed native phase.

The immutable independent-document bank and online SWA are compared after the
same native RMSNorm and before RoPE/FP8. This is a cache-error diagnostic, not a
claim that large error or an index hit is a token's causal answer importance.
"""
from __future__ import annotations

import base64
import hashlib
import zlib

from .core import digest, require
from .headtail_link_plan import native_positions

LAYERS = (0, 1, 2, 3, 7, 13, 19, 20, 39)
BIN_EDGES = (0, 128, 256, 384, 512, 1024, 4096, 8192, 16384, 32640, 32768)
SCHEMA = "early-token-diagnostics-v1"
CHUNK_TOKENS, YOCO_KEEP_TOKENS = 8192, 140


def pack_rows(positions, values):
    """Lossless little-endian scalar evidence; never serialize model KV vectors."""
    import numpy as np
    p = np.asarray(positions, dtype="<u4")
    v = np.asarray(values, dtype="<f4")
    require(p.ndim == 1 and v.shape == (len(p), 3), "diagnostic scalar geometry")
    raw = p.tobytes() + v.tobytes()
    return dict(encoding="zlib-base64", position_dtype="<u4", value_dtype="<f4",
                layout="positions_then_row_major_values", columns=["error_l2_squared", "reference_l2_squared", "cosine_distance"],
                rows=len(p), raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b64encode(zlib.compress(raw, 6)).decode("ascii"))


def unpack_rows(record):
    import numpy as np
    require(record["encoding"] == "zlib-base64" and record["position_dtype"] == "<u4"
            and record["value_dtype"] == "<f4" and record["layout"] == "positions_then_row_major_values",
            "diagnostic encoding changed")
    raw = zlib.decompress(base64.b64decode(record["data"], validate=True))
    n = record["rows"]
    require(type(n) is int and n >= 0 and len(raw) == record["raw_bytes"] == n * 16
            and hashlib.sha256(raw).hexdigest() == record["raw_sha256"], "diagnostic scalar hash/length")
    p = np.frombuffer(raw, dtype="<u4", count=n)
    v = np.frombuffer(raw, dtype="<f4", offset=n*4).reshape(n, 3)
    require(np.isfinite(v).all() and (v >= 0).all() and (p[1:] > p[:-1]).all(), "invalid diagnostic scalars")
    return p, v


def scalar_summary(positions, values, document_tokens=32768):
    import numpy as np
    p, v = np.asarray(positions), np.asarray(values, dtype=np.float64)
    require(len(p) and v.shape == (len(p), 3), "empty diagnostic summary")
    require(type(document_tokens) is int and document_tokens > 0 and p.min() >= 0 and p.max() < document_tokens,
            "diagnostic positions outside actual document")
    energy, ref = v[:, 0], v[:, 1]
    relative = np.sqrt(energy / np.maximum(ref, 1e-12))
    total = float(energy.sum())
    def group(mask):
        n = int(mask.sum())
        return dict(rows=n, error_energy=float(energy[mask].sum()),
                    mean_relative_l2=float(relative[mask].mean()) if n else None,
                    mean_cosine_distance=float(v[mask, 2].mean()) if n else None,
                    error_energy_fraction=float(energy[mask].sum()/total) if total else None)
    edges = sorted(set([0, document_tokens, max(0, document_tokens-128)]
                       + [edge for edge in BIN_EDGES if edge < document_tokens]))
    bins = [dict(local_start=lo, local_end=hi, **group((p >= lo) & (p < hi)))
            for lo, hi in zip(edges, edges[1:])]
    boundaries = {}
    for prefix in (128, 256, 384, 512):
        boundaries[f"head{prefix}"] = group(p < prefix)
        boundaries[f"head{prefix}_tail128"] = group((p < prefix) | (p >= document_tokens-128))
    boundaries["tail128"] = group(p >= document_tokens-128)
    budgets = []
    for k in (128, 256, 384, 512):
        k = min(k, len(p))
        # Stable positional tie break; top-error is an oracle diagnostic only.
        top = np.lexsort((p, -energy))[:k]
        budgets.append(dict(budget=k, top_error_energy_fraction=float(energy[top].sum()/total) if total else None,
                            uniform_random_expected_energy_fraction=k/len(p)))
    return dict(observed_rows=len(p), possible_rows=document_tokens, coverage=len(p)/document_tokens,
                error_energy=total, reference_energy=float(ref.sum()),
                relative_frobenius=float(np.sqrt(total/max(float(ref.sum()), 1e-12))),
                relative_l2_percentiles={str(q):float(np.percentile(relative,q)) for q in (50,90,95,99,100)},
                bins=bins, boundaries=boundaries, oracle_budgets=budgets)


def _average_ranks(values):
    import numpy as np
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat((starts + ends - 1) / 2, ends-starts)
    return ranks


def rank_comparison(a_positions, a_values, b_positions, b_values):
    import numpy as np
    positions, ia, ib = np.intersect1d(a_positions, b_positions, return_indices=True)
    if len(positions) < 2:
        return dict(common_rows=len(positions), spearman=None, top256_overlap_fraction=None)
    a = np.sqrt(a_values[ia,0] / np.maximum(a_values[ia,1], 1e-12))
    b = np.sqrt(b_values[ib,0] / np.maximum(b_values[ib,1], 1e-12))
    ra, rb = _average_ranks(a), _average_ranks(b)
    correlation = float(np.corrcoef(ra,rb)[0,1]) if ra.std() and rb.std() else None
    k = min(256, len(positions))
    ta, tb = np.lexsort((positions,-a))[:k], np.lexsort((positions,-b))[:k]
    return dict(common_rows=len(positions), spearman=correlation,
                top256_overlap_fraction=len(np.intersect1d(ta,tb))/k, top_budget=k)


def native_row_groups(positions, values, *, document_start, total_tokens, layer):
    """Separate native full rows, YOCO placeholders and the final rectangle."""
    import numpy as np
    p, v = np.asarray(positions), np.asarray(values, dtype=np.float64)
    rect = max(0, total_tokens-YOCO_KEEP_TOKENS)
    full = np.full(len(p), layer < 20, dtype=bool)
    placeholder = ~full & (p+document_start < rect)
    final = ~full & ~placeholder
    def group(mask):
        count = int(mask.sum())
        error, reference = float(v[mask,0].sum()), float(v[mask,1].sum())
        return dict(rows=count, error_energy=error, reference_energy=reference,
            relative_frobenius=float(np.sqrt(error/max(reference,1e-12))) if count else None,
            mean_relative_l2=float(np.sqrt(v[mask,0]/np.maximum(v[mask,1],1e-12)).mean()) if count else None,
            mean_cosine_distance=float(v[mask,2].mean()) if count else None)
    return {name:group(mask) for name,mask in
            (("full_layout",full),("yoco_placeholder",placeholder),("yoco_final_rectangle",final))}


class EarlyTokenDiagnostics:
    def __init__(self, *, controller):
        require(controller.mode == "native" and controller.arm == "recomputed" and controller.diagnostic,
                "diagnostics require an independent native recomputed phase")
        self.c = controller
        self.rows = {}
        self.chunks = set()
        self.index_events = []
        require(tuple(layer.layer_idx for layer in controller.layers[:3]) == (0,1,2),
                "diagnostic routing requires the actual first three modules")
        self.routing_modules = {}
        for layer in controller.layers[:3]:
            module = layer.self_attn
            require(type(module.has_packed_kv) is bool and type(module.is_sparse) is bool,
                    "native global topology flags missing")
            if module.has_packed_kv:
                require(type(module.kv_compress_ratio) is int and module.kv_compress_ratio in (1,2),
                        "unsupported native global compression ratio")
                self.routing_modules[layer.layer_idx] = module
        self.bank_versions = {(d["key"], l):controller._entry(d["key"],l)["version"]
                              for d in controller.docs for l in LAYERS}

    def observe_swa(self, module, raw, metadata):
        import numpy as np
        c, t = self.c, self.c.t
        layer = module.layer_idx
        require(layer in LAYERS and metadata.is_prompt, "unexpected diagnostic layer/phase")
        marker = metadata.canonical_request_binding
        key = layer, marker["start"], marker["end"]
        require(key not in self.chunks, "duplicate diagnostic layer/chunk")
        self.chunks.add(key)
        current = c._current_positions(metadata, marker)
        canonical = c.ops.normalize(module, raw)
        require(tuple(canonical.shape) == (len(current),512), "diagnostic online canonical shape")
        positions = np.asarray(current, dtype=np.int64)
        for ordinal, doc in enumerate(c.docs):
            mask = (positions >= doc["start"]) & (positions < doc["end"])
            row_ids = np.flatnonzero(mask)
            if not len(row_ids):
                continue
            local = positions[row_ids] - doc["start"]
            selected = c._upload_indices(row_ids.tolist(), canonical.device)
            offline_ids = c._upload_indices(local.tolist(), canonical.device)
            online = canonical.index_select(0, selected).float()
            offline = c._entry(doc["key"],layer)["tensor"].index_select(0,offline_ids).float()
            error = ((online-offline)**2).sum(dim=1)
            ref_energy, online_energy = (offline**2).sum(dim=1), (online**2).sum(dim=1)
            cosine = (1-(online*offline).sum(dim=1)/(ref_energy*online_energy).sqrt().clamp_min(1e-12)).clamp(0,2)
            t._assert_async(t.isfinite(error).all() & t.isfinite(cosine).all(), "nonfinite diagnostic difference")
            with c.allow_sync():
                values = t.stack((error,ref_energy,cosine),dim=1).detach().cpu().numpy().copy()
            self.rows.setdefault((layer,ordinal), []).append((local.copy(),values))

    def observe_global_index(self, module, indices, metadata):
        # Raw native compressed-block IDs are retained. Mapping them to tokens
        # must respect the source block's native compression semantics.
        c, t = self.c, self.c.t
        require(module.layer_idx in (0,1,2) and metadata.is_prompt, "unexpected routing diagnostic")
        require(self.routing_modules.get(module.layer_idx) is module,
                "routing diagnostic module differs from native topology")
        require(c.ops.PG.context_parallel_group().size() == 1 and metadata.num_reqs == 1,
                "routing diagnostics require native CP1 single-request indices")
        marker = metadata.canonical_request_binding
        current = c._current_positions(metadata, marker)
        question_start = max(d["end"] for d in c.docs)
        selected = [i for i,p in enumerate(current) if p >= question_start]
        if not selected:
            return
        require(indices.ndim == 2 and indices.shape[0] == len(current) == metadata.total_q
                and indices.dtype == t.int32, "global index diagnostic layout")
        if module.is_sparse:
            topk = module.layer_cfg.get("index_topk")
            require(type(topk) is int and topk > 0 and indices.shape[1] == topk,
                    "native sparse global index width differs from layer configuration")
        ids = c._upload_indices(selected, indices.device)
        ratio = module.kv_compress_ratio
        ratio_key = str(ratio)
        if module.is_sparse:
            starts, ends = metadata.ks[ratio_key], metadata.ke[ratio_key]
            require(tuple(starts.shape) == tuple(ends.shape) == (len(current),),
                    "native sparse compressed bounds shape")
        else:
            bounds = metadata.kv_ca_index_offset_len[ratio_key]
            require(tuple(bounds.shape) == (len(current),2), "native dense compressed bounds shape")
            starts, ends = bounds[:,0], bounds[:,0]+bounds[:,1]
        with c.allow_sync():
            values = indices.index_select(0,ids).detach().cpu().tolist()
            key_starts = starts.index_select(0,ids).detach().cpu().tolist()
            key_ends = ends.index_select(0,ids).detach().cpu().tolist()
        query_positions = [current[i] for i in selected]
        expected_ends = [(p+1)//ratio for p in query_positions]
        require(key_starts == [0]*len(selected) and key_ends == expected_ends,
                "native compressed indices are not single-request absolute blocks")
        require(all(value == -1 or 0 <= value < limit
                    for row,limit in zip(values,expected_ends) for value in row),
                "native compressed index is invalid or noncausal")
        self.index_events.append(dict(layer=module.layer_idx, chunk_start=marker["start"], chunk_end=marker["end"],
            query_absolute_positions=query_positions, question_suffix_start=question_start,
            kv_compress_ratio=ratio, native_global_indices=values,
            selection_kind="sparse_topk" if module.is_sparse else "dense_causal_enumeration",
            coordinate_system="loaded_compressed_cache_rows_single_request", invalid_index=-1,
            token_interval="[index*ratio,(index+1)*ratio)", context_parallel_size=1, num_reqs=1,
            native_compressed_key_starts=key_starts, native_compressed_key_ends=key_ends,
            interpretation="native_compressed_block_routing_proxy_not_attention_weights",
            includes_question_suffix_template_tokens=True))

    def finish(self):
        import numpy as np
        c = self.c
        records, merged = [], {}
        natural = getattr(c, "natural_boundary_inputs", False)
        total = len(c.ids)
        chunks = [(start,min(start+CHUNK_TOKENS,total)) for start in range(0,total,CHUNK_TOKENS)]
        require(self.chunks == {(layer,start,end) for layer in LAYERS for start,end in chunks},
                "diagnostic layer/chunk coverage incomplete")
        for layer in LAYERS:
            expected_absolute = np.asarray([p for start,end in chunks
                for p in native_positions(start,end,total,layer)], dtype=np.int64)
            for ordinal, doc in enumerate(c.docs):
                expected = expected_absolute[(expected_absolute >= doc["start"])
                    & (expected_absolute < doc["end"])]-doc["start"]
                pieces = self.rows.get((layer,ordinal),[])
                require(pieces or (natural and not len(expected)), "missing diagnostic layer/document")
                positions = np.concatenate([p for p,v in pieces]) if pieces else np.empty(0, dtype=np.int64)
                values = np.concatenate([v for p,v in pieces]) if pieces else np.empty((0,3), dtype=np.float32)
                require(not len(positions) or ((positions[1:] > positions[:-1]).all() and positions.min() >= 0
                        and positions.max() < doc["end"]-doc["start"]), "diagnostic position overlap/range")
                require(np.array_equal(positions,expected), "diagnostic native document position coverage differs")
                require(c._entry(doc["key"],layer)["version"] == self.bank_versions[(doc["key"],layer)],
                        "diagnostic mutated offline bank")
                merged[layer,ordinal] = positions,values
                record = dict(layer=layer, document_ordinal=ordinal, document_sha256=doc["key"],
                    document_start=doc["start"], document_end=doc["end"],
                    native_yoco_rectangle_lo=max(0,total-YOCO_KEEP_TOKENS) if layer >= 20 else None,
                    native_row_groups=native_row_groups(positions,values,
                        document_start=doc["start"],total_tokens=total,layer=layer) if len(positions) else None,
                    summary=scalar_summary(positions,values,doc["end"]-doc["start"]) if len(positions) else None,
                    scalars=pack_rows(positions,values))
                if natural:
                    record.update(available=bool(len(expected)), expected_rows=len(expected),
                        document_tokens=doc["end"]-doc["start"],
                        unavailable_reason=None if len(expected) else "no_native_surviving_rows")
                records.append(record)
        question_start = max(doc["end"] for doc in c.docs)
        expected_routing = {(layer,start,end):[p for p in native_positions(start,end,total,layer)
            if p >= question_start] for layer in self.routing_modules for start,end in chunks
            if end > question_start}
        actual_routing = {(event["layer"],event["chunk_start"],event["chunk_end"]):event["query_absolute_positions"]
                          for event in self.index_events}
        require(len(actual_routing) == len(self.index_events) and actual_routing == expected_routing,
                "diagnostic native question routing coverage incomplete")
        correlations = []
        for early in (1,2,3):
            for late in (7,13,19,20,39):
                for ordinal in range(len(c.docs)):
                    comparison = dict(early_layer=early,late_layer=late,document_ordinal=ordinal,
                        **rank_comparison(*merged[early,ordinal],*merged[late,ordinal]))
                    if natural:
                        comparison["available"] = comparison["common_rows"] >= 2
                    correlations.append(comparison)
        report = dict(schema="early-token-diagnostics-natural-boundary-v1" if natural else SCHEMA,
            complete=True, phase=c.phase, input_ids_sha256=c.input_sha,
            mode="native", arm="recomputed", included_in_formal_timing=False,
            representation="swa-postnorm-prerope-bf16", layers=list(LAYERS),
            metrics=["error_l2_squared","reference_l2_squared","cosine_distance"],
            full_kv_vectors_exported=False, attention_weights_collected=False,
            only_actual_native_surviving_rows=True, offline_bank_mutated=False,
            native_position_coverage_checked=True, prompt_chunk_tokens=CHUNK_TOKENS,
            native_yoco_keep_tokens=YOCO_KEEP_TOKENS,
            bin_edges=list(BIN_EDGES), records=records, early_late_rank_comparisons=correlations,
            question_routing_events=self.index_events,
            question_routing_layers=sorted(self.routing_modules),
            question_routing_compress_ratios={str(layer):module.kv_compress_ratio
                                            for layer,module in self.routing_modules.items()},
            question_routing_selection_kinds={str(layer):"sparse_topk" if module.is_sparse else "dense_causal_enumeration"
                                            for layer,module in self.routing_modules.items()},
            limitations=["KV deviation is not causal answer importance.",
                "Question routing indices are compressed-block selection, not attention mass.",
                "YOCO records cover actual surviving rows only; missing rows were not reconstructed.",
                "YOCO placeholders have empty native local SWA visibility and do not write the native SWA ring.",
                "Question-suffix early-layer routing arrives after preceding chunks traverse deeper layers.",
                "Prefix budget summaries are counterfactual error coverage, not measured answer quality."])
        if natural:
            report.update(natural_boundary_inputs=True,
                document_token_lengths=[d["end"]-d["start"] for d in c.docs],
                bin_edges_policy="clip_to_document_length_include_tail128")
        report["sha256"] = digest(report)
        self.rows.clear()
        return report
