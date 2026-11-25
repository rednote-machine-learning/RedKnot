"""Opt-in two-stage L2->L14 progressive selection, local backfill, and 256K pinned host offload.

Verified on H200 (SM90) Round 10-15 and B300 (SM10.3) LongBench21:
  - Stage 1 (L2): selects stage1_fraction (e.g. 12% / 9% / 7% / 16%) + Online_Head + tail128.
  - Shallow pass (L3..L14): runs Stage-1 rows along the existing 8-rank EP group,
    caching hidden, SWA, mHC LAZY residual, and L8 global_raw/topk in PrefixBank
    (or HostPrefixBank for >=192K / explicit host offload), while computing L14
    suffix-to-offline-key scores via native mqa_logits.
  - Stage 2 (L14 promotion + L3..L13 local backfill): promotes extra unselected
    valid C2 keys up to the exact target body budget, then executes ONLY the newly
    promoted `affected[layer] == extra` rows across L3..L13 (L14 strict locality law),
    restoring Stage-1 rows directly from PrefixBank/HostPrefixBank.
  - L14..L39 execute natively on the full union selection (with MinimalCone post-20).

This module performs NO top-level torch/numpy/deepinfer imports so CPU preflight
remains completely free of accelerator imports.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import copy
import math
from types import MethodType, SimpleNamespace as NS

from index20_selector import seal as default_seal, span_values


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def affected_positions(initial, extra, selected, mid=14, total=None):
    """Return per-layer affected position sets for layers 3..mid-1.

    L14 strict locality law (empirically verified: 0/12,640 out-of-cone changes
    at Layer 7 on H200 Round 10):
      For mid == 14, affected[layer] == set(extra) for all 3 <= layer < 14,
      strictly disjoint from Stage-1 `initial`.
    """
    initial_set = set(initial)
    extra_set = set(extra)
    selected_list = list(selected)
    require(extra_set.isdisjoint(initial_set), "Promoted extra positions overlap Stage-1 selection")
    require(initial_set | extra_set == set(selected_list),
            "Stage-1 union extra differs from final selected positions")
    require(mid in (8, 14), f"Unsupported two-stage mid layer: {mid}")
    if total is None:
        total = (max(selected_list) + 1) if selected_list else 0
    result = {}
    if mid == 14:
        for layer in range(3, mid):
            layer_set = set(extra_set)
            require(extra_set <= layer_set and layer_set.isdisjoint(initial_set),
                    "L14 strict locality invariant violated")
            result[layer] = layer_set
        return result
    # Legacy mid==8 causal 128-token SWA propagation.
    dirty = [False] * total
    for pos in extra_set:
        require(0 <= pos < total, "Promoted position out of bounds")
        dirty[pos] = True
    for layer in range(3, mid):
        prefix = [0] * (total + 1)
        acc = 0
        for idx, flag in enumerate(dirty):
            acc += int(flag)
            prefix[idx + 1] = acc
        dirty = [False] * total
        layer_set = set()
        for pos in selected_list:
            if prefix[pos + 1] > prefix[max(0, pos - 127)]:
                dirty[pos] = True
                layer_set.add(pos)
        require(extra_set <= layer_set, "Affected layer set dropped a promoted position")
        result[layer] = layer_set
    return result


def promote_stage2(
    stage1_choice,
    docs,
    total,
    values,
    valid_mask,
    *,
    target_fraction=None,
    target_body_tokens=None,
    span_window=(0, 0),
    seal_fn=default_seal,
):
    """Promote unselected valid C2 document positions using Stage-2 (L14) scores.

    Preserves all Stage-1 mandatory/head/ranked positions, applies optional
    intra-document asymmetric C2 Span smoothing to eligible keys, and breaks
    ties in ascending absolute token position order.
    """
    require(target_fraction is not None or target_body_tokens is not None,
            "Specify either target_fraction or target_body_tokens")
    n_keys = total // 2
    scores_list = [float(v) for v in values]
    valid_list = [bool(v) for v in valid_mask]
    require(len(scores_list) == n_keys and len(valid_list) == n_keys,
            "Stage-2 score or valid mask length differs from total // 2")

    start_set = set(stage1_choice["selected_positions"])
    body_set = set(stage1_choice["selected_body_positions"])
    body_tokens = int(stage1_choice["body_tokens"])

    raw_scores = list(scores_list)
    eligible_pairs = [
        valid_list[k] and (2 * k not in start_set) and (2 * k + 1 not in start_set)
        for k in range(n_keys)
    ]
    counts = [1 if ok else 0 for ok in eligible_pairs]
    left, right = span_window
    if left or right:
        scores_list, counts = span_values(docs, scores_list, counts, left, right)

    primary = [p for k, ok in enumerate(eligible_pairs) if ok for p in (2 * k, 2 * k + 1)]
    primary_set = set(primary)
    # Include remaining unselected in-document body tokens (half-occupied C2 pairs and
    # odd-boundary edge tokens) as a lower-priority fallback tier so extreme budgets up
    # to 100% (e.g. 50+50) never exhaust candidates, while keeping normal budgets exact.
    fallback = [
        p
        for doc in docs
        for p in range(doc["start"], doc["end"])
        if p not in start_set and p not in primary_set
    ]
    require(all(p not in start_set for p in primary), "Eligible promotion token already in Stage-1")
    ranked = sorted(primary, key=lambda p: (-scores_list[p // 2], p))
    if fallback:
        ranked.extend(
            sorted(
                fallback,
                key=lambda p: (
                    -(raw_scores[p // 2] if (p // 2 < n_keys and valid_list[p // 2]) else -math.inf),
                    p,
                ),
            )
        )

    if target_body_tokens is not None:
        budget = max(len(body_set), int(target_body_tokens))
    else:
        budget = max(len(body_set), math.ceil(float(target_fraction) * body_tokens))
    needed = budget - len(body_set)
    require(len(ranked) >= needed,
            f"Not enough eligible document body tokens ({len(ranked)}) to promote {needed} rows")

    extra = sorted(ranked[:needed])
    extra_set = set(extra)
    final_positions = sorted(start_set | extra_set)
    new_body = sorted(body_set | extra_set)

    updated = {k: v for k, v in stage1_choice.items() if k != "sha256"}
    stage1_fraction = float(stage1_choice.get("stage1_fraction", stage1_choice["fraction"]))
    reported_fraction = (
        float(target_fraction) if target_fraction is not None else float(stage1_choice["fraction"])
    )
    updated.update(
        fraction=reported_fraction,
        stage1_fraction=stage1_fraction,
        two_stage_l14=True,
        stage1_selected_tokens=len(start_set),
        stage1_selected_body_tokens=len(body_set),
        promoted_tokens=len(extra),
        selected_positions=final_positions,
        selected_tokens=len(final_positions),
        selected_body_positions=new_body,
        selected_body_tokens=len(new_body),
        ranked_body_positions=sorted(set(stage1_choice["ranked_body_positions"]) | extra_set),
        added_tokens=int(stage1_choice["added_tokens"]) + len(extra),
        actual_body_fraction=len(new_body) / body_tokens if body_tokens else 0.0,
        actual_selected_fraction=len(final_positions) / total,
        body_budget_exact=(
            len(new_body) == int(target_body_tokens)
            if target_body_tokens is not None
            else len(new_body) == math.ceil(float(target_fraction) * body_tokens)
        ),
    )
    return seal_fn(updated), extra


def _pin_vector(positions, device, torch_module=None):
    t = torch_module
    if t is None:
        import torch as t
    if getattr(device, "type", str(device)) == "cuda":
        return t.tensor(positions, dtype=t.long, device="cpu", pin_memory=True).to(device, non_blocking=True)
    return t.tensor(positions, dtype=t.long, device=device)


def empty_mlp(worker, meta, layer_idx, torch_module=None):
    t = torch_module
    if t is None:
        import torch as t
    from kv_cache_engram.s1_runtime import compact_metadata
    from kv_cache_engram.s1_system_runtime import inactive_empty_query_cu

    dev = meta.positions.device
    compact = compact_metadata(meta, t.empty(0, device=dev, dtype=t.long), inactive_empty_query_cu(meta, t))
    layer = worker.model.layers[layer_idx]
    worker.capability.permits(layer, meta, NS(full_rows=meta.total_q, selected_count=0))
    bf = t.empty((0, layer.mlp.hidden_size), device=dev, dtype=t.bfloat16)
    sf = t.zeros((layer.mlp.hidden_size // 128, 4), device=dev, dtype=t.int32)
    out = layer.mlp((bf.to(t.float8_e4m3fn), sf[:, :0].T), bf, compact)
    require(out.shape == bf.shape, "Zero-row EP MLP returned unexpected shape")


class PrefixBank:
    """GPU-resident Stage-1 shallow state bank for L3..L13 reuse during L2->L14 backfill."""

    def __init__(self, positions):
        self.positions = list(positions)
        self.index = {v: i for i, v in enumerate(self.positions)}
        self.data = {}
        self.saved_bytes = 0
        self.restored_bytes = 0

    def save(self, key, val, begin, end, axis=0):
        if val is None:
            require(self.data.get(key) is None, f"PrefixBank mixed None and tensor for {key}")
            self.data[key] = None
            return
        if key not in self.data:
            shape = list(val.shape)
            shape[axis] = len(self.positions)
            self.data[key] = val.new_empty(shape)
        self.data[key].narrow(axis, begin, end - begin).copy_(val)

    def combine(self, key, new, maps, axis=0):
        cached = self.data[key]
        if cached is None:
            require(new is None, f"PrefixBank expected None new tensor for {key}")
            return None
        shape = list(cached.shape)
        shape[axis] = maps["n"]
        out = cached.new_empty(shape)
        if maps["old_dst"].numel():
            out.index_copy_(axis, maps["old_dst"], cached.index_select(axis, maps["old_src"]))
        if maps["new_dst"].numel():
            require(new is not None and new.shape[axis] == maps["new_dst"].numel(),
                    f"PrefixBank new tensor row mismatch for {key}")
            out.index_copy_(axis, maps["new_dst"], new)
        return out

    def clear(self):
        self.data.clear()


class HostPrefixBank(PrefixBank):
    """256K OOM-safe PrefixBank offloading L3..L8 mHC residuals to pinned CPU RAM.

    Verified 8/8 Bit-Exact on H200 Round 13: only L3..L8 'residual' tensors are
    copied asynchronously to pin_memory=True CPU buffers and synchronized via
    CUDA events; backfill restores only each frame's live `[lo, hi)` interval.
    """

    OFFLOAD_LAYERS = frozenset((3, 4, 5, 6, 7, 8))

    def __init__(self, positions, *, torch_module=None):
        super().__init__(positions)
        self._t = torch_module
        self.events = {}

    def _torch(self):
        if self._t is None:
            import torch
            self._t = torch
        return self._t

    def save(self, key, val, begin, end, axis=0):
        if val is None or key[1] != "residual" or key[0] not in self.OFFLOAD_LAYERS:
            return super().save(key, val, begin, end, axis)
        t = self._torch()
        rows = val.movedim(axis, 0)
        require(rows.shape[0] == end - begin and rows.is_cuda,
                "HostPrefixBank requires CUDA residual rows matching [begin, end)")
        if key not in self.data:
            self.data[key] = t.empty(
                (len(self.positions), *rows.shape[1:]),
                dtype=rows.dtype,
                device="cpu",
                pin_memory=True,
            )
            self.saved_bytes += self.data[key].numel() * rows.element_size()
        slot = self.data[key]
        stream = t.cuda.current_stream(rows.device)
        if key in self.events:
            stream.wait_event(self.events[key])
        slot[begin:end].copy_(rows.contiguous(), non_blocking=True)
        event = t.cuda.Event()
        event.record(stream)
        self.events[key] = event

    def combine(self, key, new, maps, axis=0):
        if key not in self.events:
            return super().combine(key, new, maps, axis)
        t = self._torch()
        slot = self.data[key]
        dst = maps["old_dst"]
        dev = dst.device
        stream = t.cuda.current_stream(dev)
        stream.wait_event(self.events[key])
        shape = list(slot.shape)
        shape[0] = maps["n"]
        result = t.empty(shape, device=dev, dtype=slot.dtype).movedim(0, axis)
        if dst.numel():
            lo, hi = maps["source_interval"]
            require(0 <= lo < hi <= len(self.positions), "Invalid HostPrefixBank source interval")
            block = slot[lo:hi].to(dev, non_blocking=True).movedim(0, axis)
            result.index_copy_(axis, dst, block.index_select(axis, maps["old_src"] - lo))
            self.restored_bytes += block.numel() * block.element_size()
        if maps["new_dst"].numel():
            require(new is not None and new.shape[axis] == maps["new_dst"].numel(),
                    f"HostPrefixBank new tensor row mismatch for {key}")
            result.index_copy_(axis, maps["new_dst"], new)
        return result

    def clear(self):
        self.events.clear()
        super().clear()


def build_backfill_indices(run, layer, positions, device, torch_module=None):
    """Build GPU destination/source index maps for Stage-1 reuse and backfill rows."""
    states = run["states"]
    changed = run["affected"][layer]
    ids = [k for k, v in enumerate(positions) if v in changed]
    stable = [k for k, v in enumerate(positions) if v not in changed]
    require(all(positions[k] in states.index for k in stable),
            "Unchanged backfill row missing from Stage-1 PrefixBank")
    maps = dict(
        n=len(positions),
        new_dst=_pin_vector(ids, device, torch_module),
        old_dst=_pin_vector(stable, device, torch_module),
        old_src=_pin_vector([states.index[positions[k]] for k in stable], device, torch_module),
    )
    if isinstance(states, HostPrefixBank) and stable:
        first = positions[stable[0]]
        last = positions[stable[-1]]
        maps["source_interval"] = (states.index[first], states.index[last] + 1)
    return maps


def _doc_bounds_and_key(doc):
    if "start" in doc:
        return doc["start"], doc["end"], doc["key"]
    return doc["input_start"], doc["input_end"], doc["token_ids_sha256"]


def project_offline_l14_keys(native, documents, total, mid=14):
    """Project offline cached global_raw at layer `mid` into index_k + RoPE keys."""
    import torch

    ctrl = native.controller
    cache = ctrl.cache
    ops = cache.ops
    n_keys = total // 2
    layer_attn = native.layers[mid].self_attn
    index_mod = layer_attn.index_attn
    dev = layer_attn.q_a_kv_swa_proj.weight.device
    packed = torch.zeros((n_keys, 64), dtype=torch.int8, device=dev)
    scales = torch.zeros((n_keys, 1), dtype=torch.int32, device=dev)
    valid = [False] * n_keys
    for doc in documents:
        start, end, key = _doc_bounds_and_key(doc)
        lo, hi = (start + 1) // 2, end // 2
        if hi <= lo:
            continue
        values = cache.entry(key, "global_raw", mid)
        strip = values[2 * lo - start : 2 * hi - start].contiguous()
        compressed = ops.compress_canonical(layer_attn, strip)
        normed = layer_attn.PackedWkv.norm(compressed)
        ik = index_mod.index_k_module.forward_pre_norm(normed)
        ip = torch.arange(lo, hi, device=dev, dtype=torch.int32) * 2
        val, sf = ops.transform(
            ik,
            norm_args=ops.Norm(
                index_mod.index_k_module.norm.weight,
                index_mod.index_k_module.norm.variance_epsilon,
            ),
            rope_args=ops.Rope(
                ip,
                index_mod.rotary_emb.cos_sin_cache,
                index_mod.rotary_emb.rotary_dim,
                index_mod.rotary_emb.is_neox_style,
                offset=ik.shape[-1] - index_mod.rotary_emb.rotary_dim,
            ),
            cast_args=ops.CastArgs.from_cache_format(index_mod.cache_format),
        )
        packed[lo:hi] = val.reshape(hi - lo, 64)
        scales[lo:hi] = sf.reshape(hi - lo, 1)
        for k in range(lo, hi):
            valid[k] = True
    return {mid: (packed, scales, valid)}


def _install_suffix_observers(stack, native, docs, keys):
    import torch
    from deepinfer.model_executor.layers.sm90.indexer import mqa_logits

    _, suffix, _ = _doc_bounds_and_key(docs[-1])
    scores = {}
    seen = {}
    for layer_idx, (k_packed, k_scales, _valid) in keys.items():
        mod = native.layers[layer_idx].self_attn.index_attn
        orig = mod._compute_prefill_p_sum
        maxima = k_packed.new_full((len(k_packed),), -float("inf"), dtype=torch.float32)
        scores[layer_idx] = maxima
        seen[layer_idx] = []

        def make_hook(_orig, _k, _sf, _l, _score):
            def hook(self_mod, q, kv, weights, meta, chunk, blocks, bs, runtime):
                original = _orig(q, kv, weights, meta, chunk, blocks, bs, runtime)
                with native.allow_sync():
                    absolute = meta.positions[chunk]
                    ind = torch.where(absolute >= suffix)[0]
                    if len(ind):
                        pos = absolute.index_select(0, ind)
                        seen[_l].extend(pos.cpu().tolist())
                        qs = tuple(v[chunk].index_select(0, ind).contiguous() for v in q)
                        w = weights[chunk].index_select(0, ind).contiguous()
                        ends = (pos + 1) // 2
                        vv = mqa_logits(
                            qs,
                            (_k, _sf),
                            w,
                            torch.zeros_like(ends, dtype=torch.int32),
                            ends.int(),
                            len(_k),
                        )
                        _score.copy_(torch.maximum(_score, vv.float().amax(0)))
                return original

            return MethodType(hook, mod)

        hooked = make_hook(orig, k_packed, k_scales, layer_idx, maxima)
        had = "_compute_prefill_p_sum" in mod.__dict__
        prev = mod.__dict__.get("_compute_prefill_p_sum")
        mod._compute_prefill_p_sum = hooked

        def _restore(_mod=mod, _had=had, _prev=prev):
            if _had:
                _mod._compute_prefill_p_sum = _prev
            else:
                delattr(_mod, "_compute_prefill_p_sum")

        stack.callback(_restore)
    return scores, seen, suffix


def _group_context(ctrl, g):
    return NS(
        full_meta=g.full_meta,
        compact_meta=g.compact_meta,
        start=g.start,
        end=g.end,
        positions=g.compact_meta.positions,
        positions_cpu=g.positions_cpu,
        selected_rows=g.selected_rows,
        selected_count=len(g.positions_cpu),
        docs=ctrl.docs,
        input_tokens=len(ctrl.ids),
        mode=ctrl.mode,
        phase=ctrl.phase,
        rid=ctrl.rid,
        plan=g.plan,
        prepared=g.prepared,
        verify_sparse=False,
        verify=False,
        reference_hidden=None,
        reference_bf16=None,
        reference_selected_rows=g.selected_rows,
    )


def run_side_pass(native, run, initial=None, *, host_prefix_offload=False):
    """Execute synchronized 8-rank EP shallow pass (L3..mid) for Stage-1 rows."""
    import torch
    from index20_runtime import partition_native_frames
    from kv_cache_engram.s1_runtime import LAZY

    worker = native.batch_worker
    ctrl = native.controller
    mid = run["mid"]
    caches, meta = run["frame"]
    groups = []
    if initial is not None:
        pos = initial["selected_positions"]
        frames = ctrl._index20_frames
        groups = partition_native_frames([(f.start, f.end) for f in frames], pos)
        by_start = {f.start: f for f in frames}

    world = worker.phase_gather(dict(groups=len(groups), mid=mid, prefix="stage1"))
    require(
        len(world) == 8 and all(v["mid"] == mid and v["prefix"] == "stage1" for v in world),
        "Incomplete 8-rank EP agreement for two-stage shallow pass",
    )
    count = max(v["groups"] for v in world)
    require(count > 0, "Two-stage shallow pass invoked with zero Stage-1 groups across world")

    if initial is not None:
        bank = HostPrefixBank(pos, torch_module=torch) if host_prefix_offload else PrefixBank(pos)
        run["states"] = bank
        run["side_mlp_rows"] = []

    with ExitStack() as scope:
        if initial is not None:
            cache = copy(ctrl.cache)
            vars(cache).pop("bind_selection", None)
            cache.global_carry = {}
            cache.global_events = []
            cache.system_global_update_events = []
            cache.next_chunk = ctrl.cache.next_chunk.copy()
            cache.bind_selection(pos)
            aux = copy(ctrl.attention)
            for key in ("_memory", "begin_phase", "_candidate_event", "_projection"):
                vars(aux).pop(key, None)
            aux.cache = cache
            type(aux).begin_phase(aux)
            probe = run["keys"]
            scores, seen, suffix = _install_suffix_observers(scope, native, ctrl.docs, probe)
            cursor = 0

        for j in range(count):
            if j >= len(groups):
                for i in range(3, mid):
                    empty_mlp(worker, meta, i, torch_module=torch)
                continue
            g = ctrl._make_deep_group([by_start[a] for a, _z in groups[j]], pos)
            ctx = _group_context(ctrl, g)
            h, lazy, cross = g.hidden, g.lazy, g.cross
            n = len(g.positions_cpu)
            require(n > 0 and pos[cursor : cursor + n] == g.positions_cpu,
                    "Stage-1 group positions drifted from selection")
            for i in range(3, mid + 1):
                layer = native.layers[i]
                attn_mod = layer.self_attn
                hook = None
                if layer.engram is not None:
                    require(i == 14, "Unexpected Engram layer in shallow pass")
                    hook = lambda trunk: layer.engram(
                        trunk, g.engram14_hook, getattr(g.compact_meta, "images_seq_mask", None)
                    )
                packed, bf, carry = layer.attn_norm_forward(h, lazy, hook)
                project = aux._projection

                def capture(*args, _project=project, _i=i, _cursor=cursor, _n=n):
                    v = _project(*args)
                    if _i < mid:
                        bank.save((_i, "swa"), v[0]["swa"], _cursor, _cursor + _n)
                        if _i == 8:
                            bank.save((_i, "global"), v[0]["global_raw"], _cursor, _cursor + _n)
                    return v

                prev_proj = aux.__dict__.get("_projection")
                had_proj = "_projection" in aux.__dict__
                aux._projection = capture
                try:
                    att = aux.forward(attn_mod, packed, bf, caches[i], g.compact_meta, cross, ctx)
                finally:
                    if had_proj:
                        aux._projection = prev_proj
                    else:
                        vars(aux).pop("_projection", None)
                if i == mid:
                    break  # Final attention already recorded Stage-2 scores.
                packed, bf, lazy = layer.ffn_norm_forward(att, carry)
                h = layer.mlp(packed, bf, g.compact_meta)
                run["side_mlp_rows"].append((i, n))
                if i < mid:
                    bank.save((i, "hidden"), h, cursor, cursor + n)
                    for key in LAZY:
                        bank.save((i, key), getattr(lazy, key), cursor, cursor + n, 1)
                    if i == 8:
                        bank.save((i, "topk"), cross.topk_indices, cursor, cursor + n)
            cursor += n

        if initial is not None:
            require(cursor == len(pos), "Incomplete Stage-1 shallow pass cursor")
            with native.allow_sync():
                target = list(range(suffix, len(ctrl.ids)))
                require(all(sorted(seen[i]) == target for i in probe),
                        "L14 suffix observer did not cover every suffix token once")
                run["scores"] = {}
                for layer_idx, tensor in scores.items():
                    vals = tensor.detach().cpu().tolist()
                    valid_mask = probe[layer_idx][2]
                    require(
                        all(math.isfinite(vals[k]) for k, ok in enumerate(valid_mask) if ok),
                        f"Nonfinite Stage-2 L{layer_idx} score on valid offline key",
                    )
                    run["scores"][layer_idx] = vals


def narrow_group(ctrl, native, layer, hidden, kv, meta, lazy, cross, run):
    """Run L3..L13 local backfill on ONLY affected rows, reusing Stage-1 states."""
    import torch
    from kv_cache_engram.s1_runtime import LAZY, compact_metadata, gather_lazy, unwrap
    from kv_cache_engram.s1_system_batch import device_cumulative_pair

    i = layer.layer_idx
    attn_mod = layer.self_attn
    att = ctrl.attention
    ctx = ctrl.ctx
    worker = native.batch_worker
    old = run["states"]
    pos = ctx.positions_cpu
    dev = meta.positions.device
    maps = build_backfill_indices(run, i, pos, dev, torch_module=torch)
    ids = maps["new_dst"]
    k = ids.numel()
    view = compact_metadata(meta, ids, device_cumulative_pair(meta.cu_seqlens_q, k, torch))
    h = unwrap(hidden)
    if k:
        packed, bf, carry = layer.attn_norm_forward(h.index_select(0, ids), gather_lazy(lazy, ids), None)
        projected = att._project(attn_mod, packed, bf, view.positions)
    else:
        bf = h.new_empty((0, layer.mlp.hidden_size), dtype=torch.bfloat16)
        packed = None
        projected = att._project(attn_mod, None, bf, view.positions)
    projected["swa"] = old.combine((i, "swa"), projected["swa"], maps)
    if attn_mod.is_kv_source:
        require(i == 8, "Unexpected global KV source layer in L3..L13 backfill")
        projected["global_raw"] = old.combine((i, "global"), projected["global_raw"], maps)

    orig_core = attn_mod._prefill_attention
    orig_out = attn_mod._wv_and_o_proj
    calls = {}

    def core(_self, q, mem, global_idx, local_idx, cached, n, full):
        require(full is meta and q.shape[0] == k, "Backfill attention core query geometry mismatch")
        calls["core"] = k
        if k:
            return orig_core(
                q,
                mem,
                global_idx.index_select(0, ids),
                local_idx.index_select(0, ids),
                cached,
                n,
                view,
            )
        return None

    def output(_self, context, positions):
        require(positions is meta.positions, "Backfill output projection position mismatch")
        return orig_out(context, view.positions) if k else bf

    def _swap(obj, name, replacement):
        had = name in obj.__dict__
        prev = obj.__dict__.get(name)
        setattr(obj, name, replacement)
        return obj, name, had, prev

    swaps = []
    try:
        swaps.append(_swap(att, "_projection", lambda *args: (projected, 0, False)))
        swaps.append(_swap(attn_mod, "_prefill_attention", MethodType(core, attn_mod)))
        swaps.append(_swap(attn_mod, "_wv_and_o_proj", MethodType(output, attn_mod)))
        if attn_mod.has_local_index:
            require(i == 8, "Unexpected local index layer in L3..L13 backfill")
            orig_index = attn_mod.index_attn.forward

            def index(_self, weights, query, key, **kw):
                require(kw["input_metadata"] is meta, "Backfill index metadata mismatch")
                kw["input_metadata"] = view
                kw["cross_layer_runtime"] = copy(kw["cross_layer_runtime"])
                idx = orig_index(weights, query, key, **kw) if k else cross.topk_indices[:0]
                return old.combine((i, "topk"), idx, maps)

            swaps.append(_swap(attn_mod.index_attn, "forward", MethodType(index, attn_mod.index_attn)))
        out = att.forward(attn_mod, packed, bf, kv, meta, cross, ctx)
    finally:
        for obj, name, had, prev in reversed(swaps):
            if had:
                setattr(obj, name, prev)
            else:
                delattr(obj, name)

    require(calls.get("core") == k and out.shape[0] == k, "Incomplete backfill attention execution")
    if k:
        packed, bf, new_lazy = layer.ffn_norm_forward(out, carry)
        result = layer.mlp(packed, bf, view)
    else:
        empty_mlp(worker, meta, i, torch_module=torch)
        new_lazy = None
        result = None
    hnew = old.combine((i, "hidden"), result, maps)
    merged = copy(new_lazy if k else lazy)
    for key in LAZY:
        setattr(merged, key, old.combine((i, key), getattr(new_lazy, key, None), maps, 1))
    run["backfill_rows"].append(
        dict(
            layer=i,
            start=ctx.start,
            end=ctx.end,
            union=len(pos),
            actual_query_and_mlp=k,
            cached_old=len(pos) - k,
        )
    )
    before = len(ctrl.decoder_events)
    ctrl._observe_layer(i, ctx.start, ctx.end, (hnew, merged), meta)
    ae = att.events[-1]
    require(ae["layer"] == i and ae["selected_positions"] == pos, "Backfill attention event drift")

    def mark_counts(event, fields):
        physical = [p for p in event["selected_positions"] if p in run["affected"][i]]
        event["backfill_positions"] = physical
        event["cached_stage1_rows"] = len(event["selected_positions"]) - len(physical)
        for field in fields:
            event[field] = len(physical)

    mark_counts(ae, ("actual_attention_rows",))
    for event in ctrl.decoder_events[before:]:
        mark_counts(event, ("actual_attention_rows", "actual_ffn_rows", "actual_residual_rows", "actual_mhc_rows"))
    return hnew, merged


class TwoStageL14Coordinator:
    """Lifecycle coordinator attaching/detaching L2->L14 hooks around one multiowner phase."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._active = False
        self._stack = None
        self._run = None
        self._key_cache_id = None
        self._cached_keys = None
        self.last_summary = None

    def _get_offline_keys(self, mid=14):
        ctrl = self.runtime.controller
        cache_id = (ctrl.case_input, ctrl.case_binding, mid)
        if self._key_cache_id != cache_id:
            with self.runtime.allow_sync():
                self._cached_keys = project_offline_l14_keys(self.runtime, ctrl.docs, len(ctrl.ids), mid=mid)
            self._key_cache_id = cache_id
        return self._cached_keys

    def make_stage2_callback(self, policy):
        """Return a collector callback executed inside NativeIndexScoreCollector.finish()."""
        runtime = self.runtime
        coord = self

        def callback(stage1_choice, target_body_budget, docs, total, seal_fn):
            run = coord._run
            require(run is not None, "Two-stage callback invoked outside an active two-stage phase")
            if not (policy is not None and policy.two_stage_l14):
                # Mixed boundary batch: this owner is single-stage (<48K), but another
                # owner in the same B4 group entered two-stage -> participate as empty EP.
                run_side_pass(runtime, run, initial=None)
                return stage1_choice

            mid = policy.two_stage_mid_layer
            run["keys"] = coord._get_offline_keys(mid=mid)
            run_side_pass(runtime, run, stage1_choice, host_prefix_offload=policy.host_prefix_offload)
            chosen, extra = promote_stage2(
                stage1_choice,
                docs,
                total,
                run["scores"][mid],
                run["keys"][mid][2],
                target_fraction=policy.fraction,
                target_body_tokens=target_body_budget,
                span_window=policy.window,
                seal_fn=seal_fn,
            )
            old = set(stage1_choice["selected_positions"])
            require(not old.intersection(extra) and old.union(extra) == set(chosen["selected_positions"]),
                    "Two-stage union invariant failed")
            run["affected"] = affected_positions(old, extra, chosen["selected_positions"], mid=mid, total=total)
            run["stage1_tokens"] = len(old)
            run["promoted_tokens"] = len(extra)
            run["target_body_budget"] = target_body_budget
            runtime.controller._v2_two_stage_state = run
            return chosen

        return callback

    def begin_phase(self, policy, *, world_two_stage: bool):
        """Install per-phase worker and L3..L13 backfill hooks if any rank runs two-stage."""
        require(not self._active, "TwoStageL14Coordinator already active")
        self.last_summary = None
        if not world_two_stage:
            return

        runtime = self.runtime
        worker = runtime.batch_worker
        ctrl = runtime.controller
        mid = policy.two_stage_mid_layer if policy is not None else 14
        run = dict(
            mid=mid,
            backfill_rows=[],
            side_mlp_rows=[],
            host_prefix_offload=bool(policy is not None and policy.host_prefix_offload),
        )
        stack = ExitStack()

        def _patch_attr(obj, name, replacement):
            had = name in obj.__dict__
            prev = obj.__dict__.get(name)
            setattr(obj, name, replacement)

            def _restore():
                if had:
                    setattr(obj, name, prev)
                else:
                    delattr(obj, name)

            stack.callback(_restore)

        orig_forward = worker.forward

        def remember_frame(_self, caches, meta, hidden, dflash):
            run["frame"] = (caches, meta)
            try:
                return orig_forward(caches, meta, hidden, dflash)
            finally:
                run.pop("frame", None)

        _patch_attr(worker, "forward", MethodType(remember_frame, worker))

        if ctrl is None:
            orig_agree = worker.agree_flush

            def flush_with_side_pass(_self, count):
                require(count is None, "Inactive rank received non-None flush group count")
                run_side_pass(runtime, run, initial=None)
                return orig_agree(count)

            _patch_attr(worker, "agree_flush", MethodType(flush_with_side_pass, worker))
        elif policy is not None and policy.two_stage_l14:
            for layer_idx in range(3, mid):
                layer = runtime.layers[layer_idx]
                orig_layer_forward = layer.forward

                def make_layer_call(_layer, _orig):
                    def call(_self, hidden_states, kv_cache, input_metadata, residual, **kw):
                        if (
                            not input_metadata.is_prompt
                            or not ctrl.index20_enabled
                            or ctrl.ctx is None
                            or not run.get("affected")
                        ):
                            return _orig(hidden_states, kv_cache, input_metadata, residual, **kw)
                        require(
                            _layer.engram is None and kw.get("embed_hook") is None,
                            "Unexpected Engram hook inside L3..L13 backfill",
                        )
                        return narrow_group(
                            ctrl,
                            runtime,
                            _layer,
                            hidden_states,
                            kv_cache,
                            input_metadata,
                            residual,
                            kw["cross_layer_runtime"],
                            run,
                        )

                    return MethodType(call, _layer)

                _patch_attr(layer, "forward", make_layer_call(layer, orig_layer_forward))

        self._run = run
        self._stack = stack
        self._active = True

    def finalize_summary(self):
        """Extract verified execution summary before tearing down phase buffers."""
        if not self._active or self._run is None:
            return None
        run = self._run
        ctrl = self.runtime.controller
        if ctrl is None or "affected" not in run:
            return None
        mid = run["mid"]
        observed_layers = {l for l, _n in run["side_mlp_rows"]}
        require(observed_layers == set(range(3, mid)),
                f"Incomplete Stage-1 shallow pass layers: {sorted(observed_layers)}")
        for layer_idx in range(3, mid):
            executed = sum(x["actual_query_and_mlp"] for x in run["backfill_rows"] if x["layer"] == layer_idx)
            require(executed == len(run["affected"][layer_idx]),
                    f"Layer {layer_idx} backfill executed {executed} rows != {len(run['affected'][layer_idx])}")
        states = run.get("states")
        summary = dict(
            mid_layer=mid,
            stage1_tokens=run.get("stage1_tokens"),
            promoted_tokens=run.get("promoted_tokens"),
            target_body_budget=run.get("target_body_budget"),
            side_mlp_layers_rows=list(run["side_mlp_rows"]),
            backfill_rows=list(run["backfill_rows"]),
            host_prefix_offload=bool(run.get("host_prefix_offload", False)),
            host_prefix_saved_bytes=getattr(states, "saved_bytes", 0),
            host_prefix_restored_bytes=getattr(states, "restored_bytes", 0),
        )
        self.last_summary = summary
        return summary

    def end_phase(self):
        """Always restore all hooks and free GPU/pinned PrefixBank storage."""
        if not self._active:
            return
        try:
            if self._stack is not None:
                self._stack.close()
        finally:
            if self._run is not None:
                states = self._run.get("states")
                if states is not None:
                    states.clear()
                self._run.clear()
            ctrl = self.runtime.controller
            if ctrl is not None:
                ctrl._v2_two_stage_state = None
            self._run = None
            self._stack = None
            self._active = False

    def clear_key_cache(self):
        self._key_cache_id = None
        self._cached_keys = None
