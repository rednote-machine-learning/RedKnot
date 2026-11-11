"""Frozen, exploratory early-six sensitivity / full-row mHC pilot contract."""
from kv_cache_engram.core import digest

ARMS = ("recomputed", "early20", "early50")
SPARSE_LAYERS = (6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18)
CAPTURE_LAYERS = tuple(range(6)) + SPARSE_LAYERS
DEPENDENCIES = {
    "sparse_query_20261030_v8/sparse_attention.py": "2e52b893dcf59427072add5cbbbdcc6d8596ff177b0cd98650b4011e6e68e45c",
    "sparse_query_20261030_v8/sparse_residual.py": "4be63054371c4dd5810f76e5a5f140a8706c69be44d76def6d0b9090ba26f19b",
    "post_kv_sparse_20261029/post_kv_attention.py": "8165ea64ed9666eaf7d633ba8e646bd1cf906b7ccc141bc1b2e49db714ce3e20",
}


def early_mhc_protocol():
    body = dict(schema="early-six-sensitive-mhc-pilot-protocol-v2", cases=[2, 3, 4, 5],
        datasets=["hotpotqa", "2wikimqa"], exploratory_previously_exposed=True,
        calibration_cases_in_quality_denominator=False, context_tokens=262144,
        independent_documents=8, document_tokens=32768, native_chunk_tokens=8192,
        input_claim="LongBench-expanded 8x32K plus template/question, not native 256K questions",
        arms=list(ARMS), first_full_layers=list(range(6)), sparse_layers=list(SPARSE_LAYERS),
        all_other_layers_native=True, all_global_index_source_layers_native_including_swa=True,
        score="mean_l0..5(norm(KV_online_l-KV_offline_l)/max(norm(KV_online_l),1e-8))",
        score_representation="same-layer SWA postnorm/preRoPE BF16, float64 reduction",
        optional_fractions=[.2, .5], optional_count_rounding="ceil",
        rank="per-native-chunk descending score; ties ascending input position",
        mandatory="template/question; each actual 32K document first/last128; each native chunk first/last128",
        mandatory_rows_additive=True, scout_extra_model_pass=False,
        fixed_geometry_preparation="CPU-known boundary indices uploaded before forward; complete cost included in online TTFT",
        later_layers_use_same_early_mask=True, mask_never_uses_gold_or_generation=True,
        inactive="offline SWA KV restored at target positions; zero attention and FFN outputs; actual native full-row mHC pre/post/norm at every layer",
        active="all native Q/SWA KV/attention/output projection/FFN-MoE computation",
        global_kv_index_native=True, offline_hidden_or_ffn_output_reused=False,
        formal_admission_policy_unchanged=True, diagnostic_unqualified_reuse=True,
        new_parity=dict(input_tokens=16384, source="first16384 CALcase0 tokens", output_tokens=1,
            check="exact byte fingerprints of every decoder output and four lazy mHC fields across two native chunks and all40 layers",
            compares="native versus all-selected new masked path", failure="abort; no threshold relaxation",
            full256k_equivalence_claim=False),
        partial_smoke=dict(case_index=2, arms=["early20", "early50"], output_tokens=1,
            context_tokens=262144, mode="sensitive", timing_valid=False, quality_eligible=False,
            must_precede_formal=True, must_have_inactive_rows=True,
            failure="abort without changing source/protocol or automatic retry"),
        quality=dict(max_tokens=64, eos=True, gold_metrics=["EM", "F1"], keep_truncations=True),
        timing=dict(concurrency=1, warmups_per_arm=1, repeats=3, output_tokens=32,
            order="rotate(recomputed,early20,early50) by (case_index+repeat)%3",
            includes="online score normalization/reduction/ranking, plans, gathers/scatters, RoPE/cast and mHC",
            excludes="model loading, independent offline KV build, untimed parity fingerprints",
            no_c4=True, no_cold_start_claim=True),
        numerical_fault="fail closed; abort after any committed-state fault, never continue with stale mask",
        no_accuracy_or_speed_guarantee=True, no_statistical_generalization=True,
        frozen_dependencies=dict(DEPENDENCIES))
    return dict(body, sha256=digest(body))


def timing_schedule(index):
    return [(repeat, arm) for repeat in range(3)
            for arm in ARMS[(index+repeat)%3:]+ARMS[:(index+repeat)%3]]
