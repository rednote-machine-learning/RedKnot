"""Independent literal post-six SWA-skip / FFN15 exploratory protocol.

No native, Torch, model or GPU imports occur here. Old pilot files are not edited.
"""
from kv_cache_engram.core import digest

ARMS = ("recomputed", "ffn15")
FFN_LAYERS = tuple(range(6, 40))
CAPTURE_LAYERS = tuple(range(40))
SCORE_LAYERS = (2, 3, 4, 5)
FRACTIONS = {"recomputed": None, "ffn15": .15}
# Existing request binding/runtime dependencies remain byte-pinned by run_pilot.
# No old joint Q/attention sparsity dependency is required by this workflow.
DEPENDENCIES = {}


def swa_skip_protocol():
    body = dict(schema="swa-skip-ffn15-pilot-protocol-v2",
        cases=[2, 3, 4, 5], datasets=["hotpotqa", "2wikimqa"],
        exploratory_previously_exposed=True, independent_holdout_claim=False,
        calibration_cases_in_quality_denominator=False, context_tokens=262144,
        independent_documents=8, document_tokens=32768, native_chunk_tokens=8192,
        input_claim="LongBench-expanded 8x32K plus template/question, not native256K questions",
        arms=list(ARMS), first_full_layers=list(range(6)), ffn_layers=list(FFN_LAYERS),
        score_layers_zero_based=list(SCORE_LAYERS), score_layers_human=[3,4,5,6],
        score="mean_l=2,3,4(norm(KV_l+1-KV_l)/max(norm(KV_l),1e-8)) per token",
        score_representation="independent-layer SWA postnorm/preRoPE BF16, float64 reduction",
        score_is_reuse_safety_certificate=False, global_shared_kv_not_used_for_scoring=True,
        optional_fraction=.15, optional_count_rounding="ceil",
        rank="each native8K chunk descending per-token score; ties ascending absolute position",
        optional_rows="document tokens only; includes document prefix and tail",
        mandatory_ffn="non-document template/question tokens only",
        mandatory_swa="none after layer5, including template/question; first6 remain native",
        doc_tail_or_native_chunk_boundary_guard=False, masks_decoupled=True,
        all_query_attention_native=False, selected_ffn_only=True,
        post6_swa_projection=False, post6_swa_attention=False,
        post6_global_only_attention=True, includes_decode_swa_skip=True,
        decode_ffn_native=True, offline_swa_bank_retained=True,
        offline_swa_kv_attention_consumed_rows=0,
        unused_offline_cache_not_claimed_as_effective_reuse=True,
        global_only_renormalization_is_approximation=True,
        inactive_ffn_output="zero; actual native mHC/norm still runs on native present rows",
        global_kv_index_native=True, first6_source_layers_native=True,
        post6_source_layer_ffn_masked=True,
        yoco_layout="preserve native compact; FFN selection intersect native surviving query positions",
        offline_full_layout_for_kv_capture=True,
        offline_capture_layout="independent32K documents, temporarily disable only model YOCO boundary compaction; capture all40 current-layer SWA",
        restore_model_yoco_boundary_before_online=True,
        online_yoco_native_compact_keep_tokens=140,
        selected_tokens_forced_through_all_layers=False,
        offline_hidden_or_ffn_output_reused=False,
        formal_admission_policy_unchanged=True, diagnostic_unqualified_reuse=False,
        diagnostic_swa_branch_ablation=True,
        scheduler_identity=dict(schema="swa-skip-native-request-identity-v2",
            source="native CPU SequenceMetadata at every execute_main_model invocation",
            placeholder_request_id="placeholder", placeholder_seq_id=0,
            placeholder_geometry="one native prompt or decode sequence with seqlen_q=seqlen_k=1 and is_placeholder true",
            foreign_or_mixed_requests="fail closed", all40_layer_callbacks_required=True,
            real_token_position_binding=True, preserve_native_placeholder_ep_communication=True),
        all_online_prepare_and_dynamic_selection_in_ttft=True,
        new_parity=dict(input_tokens=16384, source="first16384 CALcase0 tokens", output_tokens=1,
            check="all40 decoder outputs and all4 lazy mHC fields exact across two native chunks",
            compares="native versus all-selected FFN with all-online split Q/SWA projections",
            failure="abort; no threshold relaxation", full256k_equivalence_claim=False),
        new_global_only_parity=dict(input_tokens=16384, output_tokens=2,
            compares="global-only native combined-projection reference versus optimized Q-only path; full FFN in both",
            check="80 real prefill layer/chunk outputs plus every native-identity-bound real decode/tail layer output and all4 lazy mHC fields byteexact",
            native_zos_schedule="classify real request versus placeholder from native identity before intervention; all calls retained, never infer total forward count from emitted output count",
            placeholder_policy="native communication intact; separately retained, never counted as a real token or used to fill missing identity checks",
            full_model_equivalence_claim=False, failure="abort; no threshold relaxation"),
        partial_smoke=dict(case_index=2, arms=["ffn15"], output_tokens=1,
            context_tokens=262144, mode="sensitive", timing_valid=False, quality_eligible=False,
            must_precede_formal=True, must_have_inactive_ffn_rows=True,
            must_have_global_only_rows=True, must_have_zero_post6_swa_reads=True,
            failure="abort without changing protocol or automatic retry"),
        quality=dict(max_tokens=64, eos=True, gold_metrics=["EM","F1"], keep_truncations=True),
        timing=dict(concurrency=1, warmups_per_arm=1, repeats=3, output_tokens=32,
            order="rotate(recomputed,ffn15) by (case_index+repeat)%2",
            includes="online score/reduction/rank, index upload, Q-only projection, global attention, FFN gather/scatter, mHC and sync",
            excludes="model loading, offline KV build, untimed qualifications",
            no_c4=True, no_cold_start_claim=True),
        pareto=dict(points=["recomputed","ffn15"], axes=["EM versus C1 TTFT", "F1 versus C1 TTFT"],
            dominance="all dimensions no worse and at least one strictly better",
            retain_exact_ties=True, global_optimum_claim=False, statistical_significance_claim=False),
        no_accuracy_or_speed_guarantee=True, no_statistical_generalization=True,
        numerical_fault="fail closed; abort after committed-state fault; never continue with stale selection")
    return dict(body, sha256=digest(body))


def timing_schedule(index):
    return [(repeat, ARMS[(index+repeat+slot)%2])
            for repeat in range(3) for slot in range(2)]


