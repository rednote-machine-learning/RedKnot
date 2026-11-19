"""Third-layer native index-logit selection; no model-output or gold inputs.

The native indexer still returns exactly its original top512.  A private hook
observes the full pre-topk score matrix and reduces only mandatory query rows.
Selection itself happens once all first-three-layer chunks have completed.
"""
from __future__ import annotations

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


def select_from_reduced_scores(documents, total, sums, counts, *, fraction=0.2, ratio=2, strategy="mean"):
    """CPU-only deterministic choice after one explicit GPU readback.

    Mandatory tails reserve their share of the total cacheable-body budget.
    Both positions in a C2 pair inherit the same mean score; absolute token
    position breaks ties.  Unobserved positions sort after every observed one.
    """
    require(type(fraction) in (int, float) and 0 <= fraction <= 1, "fraction outside0..1")
    require(strategy in ("mean","max","softmax_mean"),"unknown native indexer strategy")
    require(ratio == 2 and len(sums) == len(counts) == total//ratio, "source2 C2 score geometry")
    require(all(type(n) is int and n >= 0 for n in counts), "invalid score counts")
    require(all(math.isfinite(float(v)) for v in sums), "nonfinite reduced native index score")
    mandatory = mandatory_positions(documents, total)
    body = [p for d in documents for p in range(d["start"], d["end"])]
    tails={p for d in documents for p in range(d["end"]-128,d["end"])}
    budget = math.ceil(len(body)*fraction)
    require(fraction==0 or budget>=len(tails),"body budget cannot contain all mandatory tail128 positions")
    additional_budget=0 if fraction==0 else budget-len(tails)
    def rank(p):
        key = p//ratio
        observed = key < len(counts) and counts[key] > 0
        value=float(sums[key]) if strategy=="max" else (float(sums[key])/counts[key] if observed else 0.0)
        return (not observed, -value if observed else 0.0, p)
    chosen = sorted(sorted([p for p in body if p not in tails], key=rank)[:additional_budget])
    selected = sorted(set(mandatory) | set(chosen))
    selected_body=sorted(tails|set(chosen))
    return seal(dict(schema="index20-selection-v1", fraction=float(fraction),
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
        requested_ranked_tokens=additional_budget,ranked_body_positions=chosen,
        mandatory_body_tail_tokens=len(tails),selected_body_positions=selected_body,
        selected_body_tokens=len(selected_body),actual_body_fraction=len(selected_body)/len(body) if body else 0.,
        mandatory_noncached_tokens=len(mandatory)-len(tails),
        diagnostic_tail_only_fraction_zero=fraction==0,
        body_budget_exact=len(selected_body)==budget if fraction>0 else None,
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
        gold_or_model_output_used=False, top512_frequency_proxy=False))


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

    def begin(self, documents, total, device, fraction=0.2, strategy="mean"):
        require(not self.active, "index collector already active")
        require(strategy in ("mean","max","softmax_mean"),"unknown native indexer strategy")
        t=self.t
        self.documents, self.total, self.fraction,self.strategy = tuple(documents), total, fraction,strategy
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
        with self.allow_sync():
            sums=t.where(self.counts>0,self.sums,0.).cpu().tolist();counts=self.counts.cpu().tolist()
        readback_done=time.perf_counter()
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
            score_storage_bytes=self.sums.numel()*self.sums.element_size()+self.counts.numel()*self.counts.element_size())
        result.pop('sha256');result=seal(result)
        self.active=False
        self.query_mask=self.sums=self.counts=self.query_seen=None
        return result

    def close(self):
        require(self.module.__dict__.get('_compute_prefill_p_sum') is self.hook, "index hook ownership changed")
        delattr(self.module,'_compute_prefill_p_sum')
        self.active=False
