"""CPU-only protocol for independent index-hit-frequency B experiment."""
from kv_cache_engram.core import digest

ARMS = ("recomputed", "index20")
FFN_LAYERS = tuple(range(3, 40))
CAPTURE_LAYERS = tuple(range(40))
FRACTIONS = {"recomputed": None, "index20": .20}
DEPENDENCIES = {}


def index_hit20_protocol():
    body = dict(
        schema="index-hit20-b-pilot-protocol-v1",
        cases=[2,3,4,5], datasets=["hotpotqa","2wikimqa"],
        exploratory_previously_exposed=True, independent_holdout_claim=False,
        calibration_cases_in_quality_denominator=False,
        input_claim="LongBench-expanded 8x32K plus template/question; not native256K or new holdout",
        context_tokens=262144, independent_documents=8, document_tokens=32768,
        native_chunk_tokens=8192, arms=list(ARMS), first_full_layers=[0,1,2],
        ffn_layers=list(FFN_LAYERS), attention_layers=list(FFN_LAYERS),
        selection="native index hit frequency mapped from compressed global entries to original document positions",
        selection_statistic="unweighted integer hit count; no intersection or exposure normalization",
        routing_causality="chunk-wide known-prompt routing: later prompt queries may affect earlier row selection; attention mask remains causal",
        strict_sequential_causal_equivalence_claim=False,
        labels_or_future_generated_tokens_used_for_selection=False,
        optional_fraction=.20, optional_count_rounding="ceil",
        rank="per native chunk/current YOCO layout: descending hit count, ties ascending absolute position",
        optional_rows="document rows currently present in the native query layout",
        mandatory_rows="all non-document template/question query rows",
        index_refresh="preserve native index source layers and consumer sharing; rank the actual current index",
        selected_swa="recompute selected document rows and mandatory non-document rows",
        unselected_swa="restore offline canonical SWA KV to online positions/native cast",
        selected_compute=["main Q","joint SWA/global attention","output projection","FFN/MoE"],
        inactive_attention_and_ffn_output="zero contribution; native norm/residual/mHC still executes",
        global_kv_index_native=True, global_kv_complete_online=True,
        global_source_layers_zero_based=[2,8,14,20],
        historical_chunk_kv_rewritten=False,
        doc_tail_or_native_chunk_boundary_guard=False,
        no_cached_attention_or_ffn_output=True,
        native_decode_unchanged=True,
        yoco_layout="native compact keep140; rank current surviving document query rows, no forced token retention",
        decoder_prefill_selected_by_hit_frequency=True,
        cache_representation="canonical postnorm/preRoPE SWA KV; target RoPE/native cast on restore",
        offline_full_layout_for_kv_capture=True,
        offline_capture_layout="eight independent32K docs, all40 SWA layers; temporarily disable YOCO layout compression and restore before online",
        restore_model_yoco_boundary_before_online=True, online_yoco_native_compact_keep_tokens=140,
        formal_admission_policy_unchanged=True, diagnostic_unqualified_reuse=True,
        all_online_prepare_and_dynamic_selection_in_ttft=True,
        empty_ep_qualification=dict(layers=list(FFN_LAYERS), patterns=4, ranks=8, checks=1184,
            native_dispatch_routed_shared_combine_even_zero=True, byteexact=True, sync_debug=2),
        new_parity=dict(input_tokens=16384, source="first16384 CALcase0 tokens", output_tokens=1,
            compares="native versus all-selected all-online candidate",
            check="all80 layer/chunk outputs and all4 mHC fields byteexact",
            split_projections_and_gather_tolerance="exact zero",
            cache_restore_tolerance="exact zero", failure="abort; no threshold relaxation"),
        sparse_parity=dict(case_index=2, input="original8x32K+template/question", output_tokens=1,
            reference_mode="reference_sparse", candidate_mode="sensitive",
            compares="independent full-compute then mask reference against packed selected-row production",
            check="all layer/chunk outputs and all4 mHC fields byteexact; same index masks",
            quality_eligible=False, timing_valid=False, failure="abort; no threshold relaxation"),
        partial_smoke=dict(case_index=2, arms=["index20"], output_tokens=1,
            context_tokens=262144, mode="sensitive", timing_valid=False, quality_eligible=False,
            must_precede_formal=True, requires_sparse_parity=True,
            must_have_inactive_attention_rows=True, must_have_inactive_ffn_rows=True,
            must_have_swa_reuse_rows=True, failure="abort without automatic retry"),
        quality=dict(max_tokens=64, eos=True, gold_metrics=["EM","F1"], keep_truncations=True),
        timing=dict(concurrency=1,warmups_per_arm=1,repeats=3,output_tokens=32,
            order="rotate(recomputed,index20) by (case_index+repeat)%2",
            includes="all online index/hit counting/ranking/dynamic-shape sync/index upload/projections/KV restore/gather/scatter/mHC/communication",
            excludes="model loading, independent offline KV construction, qualifications and smoke",
            no_c4=True,no_qps_claim=True,no_cold_start_claim=True),
        rates=dict(denominator="sum actual native/YOCO surviving prefill rows in layers3..39; first3 excluded",
            separate_swa_attention_ffn=True, not_model_flops=True),
        selection_audit=dict(qualification="independent raw native topk reconstruction plus exact slow-reference output parity",
            formal="all per-row count vectors from frozen qualified selector; independent reranking and branch-row audits",
            formal_raw_topk_rescan_claim=False),
        pareto=dict(points=list(ARMS),global_optimum_claim=False,statistical_significance_claim=False),
        no_accuracy_or_speed_guarantee=True,no_statistical_generalization=True,
        numerical_fault="abort after committed-state fault; never continue with stale selection")
    return dict(body,sha256=digest(body))


def timing_schedule(index):
    return [(repeat,ARMS[(index+repeat+slot)%2]) for repeat in range(3) for slot in range(2)]
