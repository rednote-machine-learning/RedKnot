"""Third-layer native index-logit selection; no model-output or gold inputs.

The native indexer still returns exactly its original top512.  A private hook
observes the full pre-topk score matrix and reduces only mandatory query rows
(or pure post-document suffix rows under RedKnot V2). Selection itself happens
once all first-three-layer chunks have completed.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import time
from types import MethodType


def require(value, message):
    if not value:
        raise ValueError(message)


def seal(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return dict(value, sha256=hashlib.sha256(encoded).hexdigest())


def mandatory_positions(documents, total):
    covered = set()
    tails = set()
    previous = 0
    for d in documents:
        a, z = d["start"], d["end"]
        require(0 <= previous <= a < z <= total and z-a >= 256, "invalid cacheable document")
        covered.update(range(a, z))
        tails.update(range(z-128, z))
        previous = z
    return sorted((set(range(total))-covered) | tails)


def validate_frame_capacity(selected_positions, total, *, frame_size=8192):
    """Verify selected positions stay in-bounds, sorted, unique, and within 8192-frame capacity."""
    require(type(total) is int and total > 0, "positive total tokens required")
    require(type(frame_size) is int and frame_size > 0, "positive frame_size required")
    frame_rows = [0] * ((total + frame_size - 1) // frame_size)
    prev = -1
    for p in selected_positions:
        require(type(p) is int and 0 <= p < total, "selected position out of bounds")
        require(p > prev, "selected positions must be strictly sorted and unique")
        prev = p
        frame_rows[p // frame_size] += 1
    for i, count in enumerate(frame_rows):
        cap = min(frame_size, total - frame_size * i)
        require(count <= cap, f"frame {i} selected rows {count} exceed frame capacity {cap}")
    return frame_rows


def span_values(documents, values, counts, before, after):
    """CPU reference intra-document asymmetric C2 Span smoothing.

    Spreads each observed peak at complete C2 key `s` strictly within its
    document `[a, z) = [(start + 1) // 2, end // 2)` to `[s - before, s + after]`,
    i.e., target key `k` takes the max over observed `s in [k - after, k + before]`.
    Cross-document peak leakage is strictly prevented.
    """
    require(type(before) is int and type(after) is int and before >= 0 and after >= 0,
            "span window entries must be nonnegative integers")
    result = [float(v) for v in values]
    seen = [int(c) for c in counts]
    if not (before or after):
        return result, seen
    n_keys = len(result)
    require(len(seen) == n_keys, "span_values length mismatch between values and counts")
    for doc in documents:
        first = (doc["start"] + 1) // 2
        end = doc["end"] // 2
        if end <= first:
            continue
        window = deque()
        right = first
        for k in range(first, end):
            limit = min(end, k + before + 1)
            while right < limit:
                if counts[right]:
                    while window and values[window[-1]] <= values[right]:
                        window.pop()
                    window.append(right)
                right += 1
            while window and window[0] < k - after:
                window.popleft()
            if window:
                result[k] = float(values[window[0]])
                seen[k] = max(1, int(counts[k]))
            else:
                result[k] = 0.0
                seen[k] = 0
    return result, seen


def span_values_gpu(documents, values, counts, before, after, *, torch_module=None, functional_module=None):
    """GPU F.pad + F.max_pool1d intra-document asymmetric C2 Span smoothing.

    Mathematically identical to CPU `span_values`:
      - Unobserved keys (`counts == 0`) are filled with `-math.inf` BEFORE pooling
        so they never pollute negative observed scores.
      - Padding is `(after, before)` so target key `k` pools over `[k - after, k + before]`,
        spreading a source peak at `s` to `[s - before, s + after]`.
    """
    require(type(before) is int and type(after) is int and before >= 0 and after >= 0,
            "span window entries must be nonnegative integers")
    t = torch_module
    if t is None:
        import torch as t
    F = functional_module if functional_module is not None else t.nn.functional
    masked = t.where(counts > 0, values, -math.inf)
    if not (before or after):
        return masked
    smooth = masked.clone()
    kernel_size = 1 + before + after
    for doc in documents:
        a = (doc["start"] + 1) // 2
        z = doc["end"] // 2
        if z > a:
            chunk = F.pad(masked[a:z].view(1, 1, -1), (after, before), value=-math.inf)
            smooth[a:z] = F.max_pool1d(chunk, kernel_size, stride=1).view(-1)
    return smooth


def select_from_reduced_scores(
    documents,
    total,
    sums,
    counts,
    *,
    fraction=0.2,
    ratio=2,
    strategy="mean",
    span_window=(0, 0),
    head_width=0,
    allow_budget_floor=False,
):
    """CPU-only deterministic choice after one explicit GPU readback.

    Mandatory tails reserve their share of the total cacheable-body budget.
    Both positions in a C2 pair inherit the same score; absolute token
    position breaks ties.  Unobserved positions sort after every observed one.
    """
    require(type(fraction) in (int, float) and 0 <= fraction <= 1, "fraction outside0..1")
    require(strategy in ("mean","max","softmax_mean"),"unknown native indexer strategy")
    require(ratio == 2 and len(sums) == len(counts) == total//ratio, "source2 C2 score geometry")
    require(all(type(n) is int and n >= 0 for n in counts), "invalid score counts")
    require(all(math.isfinite(float(v)) for v in sums), "nonfinite reduced native index score")
    before, after = span_window
    require(type(before) is int and type(after) is int and before >= 0 and after >= 0, "invalid span_window")
    require(type(head_width) is int and 0 <= head_width <= 128, "head_width must be in 0..128")
    v2_features = bool(before or after or head_width > 0 or allow_budget_floor)
    if before or after:
        sums, counts = span_values(documents, sums, counts, before, after)
    mandatory = mandatory_positions(documents, total)
    body = [p for d in documents for p in range(d["start"], d["end"])]
    tails={p for d in documents for p in range(d["end"]-128,d["end"])}
    budget = math.ceil(len(body)*fraction)
    if allow_budget_floor:
        effective_budget = max(budget, len(tails))
    else:
        require(fraction==0 or budget>=len(tails),"body budget cannot contain all mandatory tail128 positions")
        effective_budget = budget
    additional_budget=0 if fraction==0 else effective_budget-len(tails)
    def rank(p):
        key = p//ratio
        observed = key < len(counts) and counts[key] > 0
        value=float(sums[key]) if strategy=="max" else (float(sums[key])/counts[key] if observed else 0.0)
        return (not observed, -value if observed else 0.0, p)
    chosen = sorted(sorted([p for p in body if p not in tails], key=rank)[:additional_budget])
    if head_width > 0:
        heads = {p for d in documents for p in range(d["start"], min(d["start"] + head_width, d["end"]))}
        selected = sorted(set(mandatory) | set(chosen) | heads)
        selected_body = sorted(tails | set(chosen) | heads)
        ranked_body = sorted(set(chosen) | (heads - tails))
        body_budget_exact = False
    else:
        selected = sorted(set(mandatory) | set(chosen))
        selected_body=sorted(tails|set(chosen))
        ranked_body = chosen
        body_budget_exact = len(selected_body)==budget if fraction>0 else None
    if v2_features:
        validate_frame_capacity(selected, total)
    payload = dict(schema="index20-selection-v1", fraction=float(fraction),
        strategy=strategy,aggregation={
            "mean":"arithmetic_mean_native_pre_topk_index_logit_over_causally_valid_mandatory_queries",
            "max":"maximum_native_pre_topk_index_logit_over_causally_valid_mandatory_queries",
            "softmax_mean":"arithmetic_mean_per_query_causal_native_index_logit_softmax_probability"}[strategy],
        score_formula="native sum_heads(relu(q_index dot k_index) * query_head_weight)",
        query_policy="original_tail128_plus_noncached_tokens", source_layer_zero_based=2,
        compressed_key_to_token_rule="C2 key k maps to absolute tokens2k and2k+1",
        budget_denominator="all_cacheable_document_body_tokens_including_mandatory_tails",
        budget_policy="reserve_tail128_then_rank_remaining_non_tail_body_within_total_budget",
        body_tokens=len(body),requested_body_budget_tokens=budget,
        requested_ranked_tokens=additional_budget,ranked_body_positions=ranked_body,
        mandatory_body_tail_tokens=len(tails),selected_body_positions=selected_body,
        selected_body_tokens=len(selected_body),actual_body_fraction=len(selected_body)/len(body) if body else 0.,
        mandatory_noncached_tokens=len(mandatory)-len(tails),
        diagnostic_tail_only_fraction_zero=fraction==0,
        body_budget_exact=body_budget_exact,
        mandatory_positions=mandatory, selected_positions=selected,
        selected_tokens=len(selected), added_tokens=len(set(selected)-set(mandatory)),
        actual_selected_fraction=len(selected)/total, tie_break="absolute_token_position_ascending",
        observed_compressed_keys=sum(n>0 for n in counts),
        unobserved_compressed_keys=sum(n==0 for n in counts),
        observed_zero_mean_keys=sum(n>0 and v==0 for v,n in zip(sums,counts)),
        observed_negative_mean_keys=sum(n>0 and v<0 for v,n in zip(sums,counts)),
        zero_mass_or_unobserved_not_silently_dropped=True,
        score_accumulator_sha256=seal({"values":sums})["sha256"],
        score_sums_sha256=seal({"values":sums})["sha256"] if strategy!="max" else None,
        score_counts_sha256=seal({"values":counts})["sha256"],
        gold_or_model_output_used=False, top512_frequency_proxy=False)
    if v2_features:
        payload.update(head_width=head_width, span_window=[before, after])
    return seal(payload)


def select_from_reduced_scores_gpu(
    documents,
    total,
    sums,
    counts,
    *,
    fraction=0.2,
    stage1_fraction=None,
    ratio=2,
    strategy="max",
    span_window=(0, 0),
    head_width=0,
    torch_module=None,
    functional_module=None,
    geometry_cache=None,
):
    """Opt-3 GPU stable-sort selection, bit-equivalent to CPU select_from_reduced_scores.

    Eliminates full-array sums/counts `.cpu().tolist()` readback and Python sorting.
    Because `eligible` is strictly ascending in absolute token position and
    `torch.argsort(..., descending=True, stable=True)` preserves relative order
    for ties (including unobserved `-math.inf` keys), tie-breaking is 100%
    identical to CPU `select_from_reduced_scores()`.

    Returns `(choice_dict, target_body_budget)`.
    """
    t = torch_module
    if t is None:
        import torch as t
    F = functional_module if functional_module is not None else t.nn.functional
    require(strategy == "max", "Opt-3 GPU selector requires max strategy")
    require(ratio == 2 and int(sums.shape[0]) == int(counts.shape[0]) == total // ratio,
            "source2 C2 score geometry")
    require(type(fraction) in (int, float) and 0 <= fraction <= 1, "fraction outside 0..1")
    before, after = span_window
    require(type(before) is int and type(after) is int and before >= 0 and after >= 0, "invalid span_window")
    require(type(head_width) is int and 0 <= head_width <= 128, "head_width must be in 0..128")
    if stage1_fraction is not None:
        require(type(stage1_fraction) in (int, float) and 0 <= stage1_fraction <= fraction,
                "stage1_fraction must lie within 0..fraction")

    doc_bounds = tuple((d["start"], d["end"]) for d in documents)
    device = sums.device
    cache_key = (total, doc_bounds, head_width, str(device))
    cached = geometry_cache.get("entry") if isinstance(geometry_cache, dict) else None
    if cached is None or cached[0] != cache_key:
        previous = 0
        body_set = set()
        tail_set = set()
        head_set = set()
        for start, end in doc_bounds:
            require(type(start) is int and type(end) is int and 0 <= previous <= start < end <= total
                    and end - start >= 256, "invalid cacheable document")
            previous = end
            body_set.update(range(start, end))
            tail_set.update(range(end - 128, end))
            if head_width > 0:
                head_set.update(range(start, min(start + head_width, end)))
        mandatory_set = (set(range(total)) - body_set) | tail_set
        mandatory_list = sorted(mandatory_set)
        eligible_list = sorted(body_set - tail_set)
        eligible_gpu = t.as_tensor(eligible_list, device=device, dtype=t.long)
        cached = (
            cache_key,
            body_set,
            tail_set,
            head_set,
            mandatory_set,
            mandatory_list,
            eligible_gpu,
            len(body_set),
            len(tail_set),
        )
        if isinstance(geometry_cache, dict):
            geometry_cache["entry"] = cached
    _, body_set, tail_set, head_set, mandatory_set, mandatory_list, eligible_gpu, nbody, ntail = cached

    smooth = span_values_gpu(
        documents, sums, counts, before, after, torch_module=t, functional_module=F
    )
    requested = math.ceil(nbody * fraction)
    top = 0 if fraction == 0 else max(0, requested - ntail)
    padded = F.pad(smooth, (0, 1), value=-math.inf)
    scores = padded[eligible_gpu // 2]
    if top > 0 and int(eligible_gpu.shape[0]) > 0:
        order = t.argsort(scores, descending=True, stable=True)[:top]
        ranked_all = eligible_gpu[order].cpu().tolist()
    else:
        ranked_all = []

    full_body = tail_set | head_set | set(ranked_all)
    target_body_budget = len(full_body)

    if stage1_fraction is not None:
        stage_requested = math.ceil(nbody * stage1_fraction)
        nfirst = 0 if stage1_fraction == 0 else max(0, stage_requested - ntail)
        stage_ranked = ranked_all[:nfirst]
        eff_fraction = float(stage1_fraction)
    else:
        stage_requested = requested
        nfirst = top
        stage_ranked = ranked_all
        eff_fraction = float(fraction)

    chosen_set = mandatory_set | head_set | set(stage_ranked)
    selected_positions = sorted(chosen_set)
    selected_body = sorted(chosen_set & body_set)
    ranked_body = sorted((chosen_set & body_set) - tail_set)
    validate_frame_capacity(selected_positions, total)

    choice = seal(dict(
        schema="redknot-v2-device-selection-v1",
        strategy="max",
        fraction=eff_fraction,
        target_fraction=float(fraction),
        stage1_fraction=eff_fraction,
        query_policy="suffix_only_post_documents",
        head_width=head_width,
        span_window=[before, after],
        body_tokens=nbody,
        requested_body_budget_tokens=stage_requested,
        requested_ranked_tokens=nfirst,
        mandatory_body_tail_tokens=ntail,
        mandatory_noncached_tokens=len(mandatory_list) - ntail,
        mandatory_positions=mandatory_list,
        selected_positions=selected_positions,
        selected_tokens=len(selected_positions),
        selected_body_positions=selected_body,
        selected_body_tokens=len(selected_body),
        ranked_body_positions=ranked_body,
        added_tokens=len(selected_positions) - len(mandatory_list),
        body_budget_exact=(len(selected_body) == stage_requested) if head_width == 0 and eff_fraction > 0 else False,
        actual_body_fraction=len(selected_body) / nbody if nbody else 0.0,
        actual_selected_fraction=len(selected_positions) / total,
        tie_break="absolute_token_position_ascending",
        original_absolute_tie_break=True,
        full_score_readbacks=0,
        gold_or_model_output_used=False,
        top512_frequency_proxy=False,
    ))
    return choice, target_body_budget


class NativeIndexScoreCollector:
    """Own a reversible instance hook; preserve the original p_sum object."""
    def __init__(self, module, *, torch_module, allow_sync):
        require(module.layer_idx == 2 and module.kv_compress_ratio == 2,
                "selector requires actual third-layer C2 native index source")
        require(not module.use_candidate_topk, "third-layer selection needs complete logits")
        require("_compute_prefill_p_sum" not in module.__dict__, "index score hook already owned")
        self.module, self.t, self.allow_sync = module, torch_module, allow_sync
        self.original = module._compute_prefill_p_sum
        self.active = False
        self._v2_config = None
        self._v2_geometry_cache = {}
        self._suffix_only = False
        self._head_width = 0
        self._span_window = (0, 0)
        self._gpu_select = False
        self._stage1_fraction = None
        self._two_stage_callback = None
        owner = self
        def observe(mod, q, kv, weights, input_metadata, chunk, candidate_blocks,
                    candidate_block_size, cross_layer_runtime):
            result = owner.original(q, kv, weights, input_metadata, chunk,
                candidate_blocks, candidate_block_size, cross_layer_runtime)
            if owner.active:
                began=time.perf_counter()
                owner.observe(result, input_metadata, chunk, candidate_blocks, kv[0].shape[0])
                owner.host_enqueue_s+=time.perf_counter()-began
            return result
        self.hook = MethodType(observe, module)
        module._compute_prefill_p_sum = self.hook

    def configure_v2(self, config=None):
        """Configure RedKnot V2 scoring/selection options before begin() (None = legacy S1.6)."""
        require(not self.active, "cannot reconfigure active index collector")
        self._v2_config = None if config is None else dict(config)
        return self

    def begin(
        self,
        documents,
        total,
        device,
        fraction=0.2,
        strategy="mean",
        *,
        suffix_only=None,
        head_width=None,
        span_window=None,
        gpu_select=None,
        stage1_fraction=None,
        two_stage_callback=None,
    ):
        require(not self.active, "index collector already active")
        require(strategy in ("mean","max","softmax_mean"),"unknown native indexer strategy")
        t=self.t
        cfg = self._v2_config or {}
        self._suffix_only = bool(cfg.get("suffix_only", False) if suffix_only is None else suffix_only)
        self._head_width = int(cfg.get("head_width", 0) if head_width is None else head_width)
        self._span_window = tuple(cfg.get("span_window", (0, 0)) if span_window is None else span_window)
        self._gpu_select = bool(cfg.get("gpu_select", False) if gpu_select is None else gpu_select)
        self._stage1_fraction = cfg.get("stage1_fraction", None) if stage1_fraction is None else stage1_fraction
        self._two_stage_callback = cfg.get("two_stage_callback", None) if two_stage_callback is None else two_stage_callback
        self.documents, self.total, self.fraction,self.strategy = tuple(documents), total, fraction,strategy
        if self._suffix_only:
            require(bool(self.documents), "suffix-only scoring requires at least one document")
            suffix_start = self.documents[-1]["end"]
            require(0 <= suffix_start < total, "invalid post-document suffix start")
            mandatory = list(range(suffix_start, total))
            self.mandatory = mandatory
            self.query_mask = t.zeros(total, dtype=t.bool, device=device)
            self.query_mask[suffix_start:] = True
        else:
            mandatory=mandatory_positions(documents,total)
            self.mandatory=mandatory
            self.query_mask=t.zeros(total,dtype=t.bool,device=device)
            self.query_mask[t.tensor(mandatory,dtype=t.long,device=device)]=True
        self.sums=t.zeros(total//2,dtype=t.float32,device=device)
        if strategy=="max":self.sums.fill_(-math.inf)
        self.counts=t.zeros(total//2,dtype=t.int64,device=device)
        self.query_seen=t.zeros(total,dtype=t.int32,device=device)
        self.calls=0;self.score_query_rows=0;self.score_value_elements=0;self.host_enqueue_s=0.;self.active=True

    def observe(self, logits, meta, chunk, candidates, loaded_keys):
        t=self.t
        require(candidates is None and meta.is_prompt and meta.num_reqs==1,
                "source2 logits changed to candidate/multirequest/decode geometry")
        from kv_cache_engram.s1_system_batch import device_integer_values
        a,z,stride=chunk.indices(meta.total_q)
        require(stride==1 and logits.shape[0]==z-a,"native score/query geometry mismatch")
        marker=meta.canonical_request_binding
        require(isinstance(marker,dict) and marker['end']-marker['start']==meta.total_q,
                "source2 native request marker missing")
        absolute_start=marker['start']+a
        wanted=[p for p in self.mandatory if absolute_start<=p<marker['start']+z]
        rows=device_integer_values([p-absolute_start for p in wanted],meta.positions,t).long()
        positions=meta.positions[chunk].index_select(0,rows).long()
        expected=device_integer_values(wanted,positions,t)
        t._assert_async((positions==expected).all(),"source2 absolute query positions changed")
        self.calls+=1
        if not wanted:return
        # ks is zero for the sole real request. Prefix key positions are global.
        t._assert_async((meta.ks['2'][chunk].index_select(0,rows)==0).all(), "source2 key coordinate is not request-local")
        valid_end=meta.ke['2'][chunk].index_select(0,rows).long()
        width=min(logits.shape[1],self.total//2,loaded_keys)
        if width:
            scores=logits[:,:width].index_select(0,rows).float()
            columns=t.arange(width,device=logits.device,dtype=t.long)
            valid=columns[None,:]<valid_end[:,None]
            values=t.where(valid,scores,0.0)
            t._assert_async(t.isfinite(values).all(), "nonfinite valid native source2 score")
            if self.strategy=="max":
                maximum=t.where(valid,scores,-math.inf).amax(dim=0)
                self.sums[:width].copy_(t.maximum(self.sums[:width],maximum))
            else:
                if self.strategy=="softmax_mean":
                    maximum=t.where(valid,scores,-math.inf).amax(dim=1,keepdim=True)
                    maximum=t.where(valid_end[:,None]>0,maximum,0.)
                    mass=t.where(valid,t.exp(scores-maximum),0.)
                    values=mass/mass.sum(dim=1,keepdim=True).clamp_min(1.e-30)
                self.sums[:width].add_(values.sum(dim=0))
            self.counts[:width].add_(valid.sum(dim=0))
        self.query_seen.index_add_(0,positions,t.ones_like(positions,dtype=t.int32))
        self.score_query_rows+=len(wanted)
        self.score_value_elements+=len(wanted)*width

    def finish(self):
        require(self.active and self.calls>0, "no native index observations")
        began=time.perf_counter()
        t=self.t
        t._assert_async((self.query_seen==self.query_mask.to(t.int32)).all(),
                        "mandatory queries missing or counted more than once")
        score_bytes = self.sums.numel()*self.sums.element_size()+self.counts.numel()*self.counts.element_size()
        if self._gpu_select:
            with self.allow_sync():
                result, target_body_budget = select_from_reduced_scores_gpu(
                    self.documents,
                    self.total,
                    self.sums,
                    self.counts,
                    fraction=self.fraction,
                    stage1_fraction=self._stage1_fraction,
                    strategy=self.strategy,
                    span_window=self._span_window,
                    head_width=self._head_width,
                    torch_module=t,
                    geometry_cache=self._v2_geometry_cache,
                )
            readback_done = time.perf_counter()
            if self._two_stage_callback is not None:
                result = self._two_stage_callback(
                    result, target_body_budget, self.documents, self.total, seal
                )
            ranked = time.perf_counter()
        else:
            with self.allow_sync():
                sums=t.where(self.counts>0,self.sums,0.).cpu().tolist();counts=self.counts.cpu().tolist()
            readback_done=time.perf_counter()
            if self._suffix_only or self._head_width > 0 or any(self._span_window):
                result=select_from_reduced_scores(
                    self.documents,self.total,sums,counts,
                    fraction=self.fraction,strategy=self.strategy,
                    span_window=self._span_window,head_width=self._head_width,
                    allow_budget_floor=True,
                )
            else:
                result=select_from_reduced_scores(self.documents,self.total,sums,counts,fraction=self.fraction,strategy=self.strategy)
            ranked=time.perf_counter()
        result=dict(result, native_logit_chunk_calls=self.calls,
            score_reduction_query_rows=self.score_query_rows,
            score_reduction_value_elements=self.score_value_elements,
            reduction_excludes_nonmandatory_query_rows=True,
            reduction_host_enqueue_wall_s=self.host_enqueue_s,
            score_readback_wall_s=readback_done-began,selection_cpu_rank_wall_s=ranked-readback_done,
            reduction_cuda_ms=None,reduction_cuda_timing_status="not_separately_profiled_in_formal_TTFT",
            complete_selection_cost_in_online_ttft=True,other_strategy_statistics_computed=False,
            online_score_collection_and_selection=True, all_mandatory_queries_observed_once=True,
            score_storage_bytes=score_bytes)
        result.pop('sha256');result=seal(result)
        self.active=False
        self.query_mask=self.sums=self.counts=self.query_seen=None
        return result

    def close(self):
        require(self.module.__dict__.get('_compute_prefill_p_sum') is self.hook, "index hook ownership changed")
        delattr(self.module,'_compute_prefill_p_sum')
        self.active=False
