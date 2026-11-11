"""Fixed head128+tail128 candidate and native-window/previous-tail union memory."""
from kv_cache_engram.core import digest

ARMS = ("headtail128_link",)
FFN_LAYERS = tuple(range(3, 40))
CAPTURE_LAYERS = tuple(range(40))
FRACTIONS = dict.fromkeys(ARMS)
# Root freezes only this new protocol's dependencies before remote admission.
DEPENDENCIES = {'benchmark/pr148_native_runner.py': '615315c76ee54320201daa918d462676cd3c59cb05000769396e3253f0886a47', 'benchmark/prepare_rag256.py': '839b268e295e460eb49440f9a59c2951e7f8721707f379551553f76c8565133e', 'strong_native_20261029/strong_native_runner.py': '45f25f878ea62741cef12ca186618483531e0c968a50dc8722051b4177411a7a', 'strong_native_20261029/native_profile.py': 'e68d7d3b39c8255ef1f06d4ba42bb26e2e6121b4f141ace5da06c41060ba0717', 'kv_recompression_20261029/canonical_request_binding.py': 'f25097ab9ebe51662c21b9652af472e43d7664e368276a4c317fc133d02e913e', 'kv_recompression_20261029/quality_process_group.py': 'a7d503f23d1df3f41044c98107b25ca2a09debc1e66384d17d26a9f9f33ba9db', 'GatedEngram/kv_cache_engram/tail_link.py': 'cafe17fb5c29038b3f8de243f46b931d8c1000b5d9403bcecb0b8dde74fff5d7', 'GatedEngram/kv_cache_engram/tail_link_staging.py': '4f5bf3d745127382e792c04a6d3f50352187843559cef89ead0de9c560d524b7', 'GatedEngram/kv_cache_engram/tail_link_plan.py': 'e62c35018784618114b52cb2d10fe47911319397e252dc5e4ca3028b73a98a1c', 'GatedEngram/kv_cache_engram/index_hit20.py': '10a699cc6b69d12f01ba76e40abe8a072b5ae531e0df60643de3dc1b0147619d', 'GatedEngram/kv_cache_engram/index_hit20_adapter.py': '9dd49e07b85b610eeea3c3d6e2f177ec23973c4740e151ab128bb3a2c9cfd2ea', 'GatedEngram/kv_cache_engram/index_hit_selector.py': '6e1f930ec446186e4da9092bd97e7845b0dd82b434efcbed22faaf265f3b6220', 'GatedEngram/kv_cache_engram/native_reuse.py': '5bd7d2dea0c71eee299c39993c4f068e86af735d05663d54ab554dc2197631a7', 'GatedEngram/kv_cache_engram/early_mhc_adapter.py': '7a19b8775c1cf9a93c9dcf41dbd8e0e54816e32ccdad0a3ae3c0fb47e20b9d00', 'GatedEngram/kv_cache_engram/ffn_only_adapter.py': 'a5fe50b1bbfdc6e66488c9616b63c74e95f862e29af9c7b2ce071987e368a9c0', 'GatedEngram/kv_cache_engram/core.py': '5bd8a4201b8756b7772fc8c6a9c2b0155e8edf8a7283c369a741f2bffa3f0f41', 'GatedEngram/gpu_pilot/index_hit20_empty_ep_qualification.py': '3a8c95fcd85fc712b7eaee2747d3cb0f14dd773d904e6d4fa229c0652b80e148', 'GatedEngram/gpu_pilot/cuda_mask_qualification.py': '910254c71c5b3d97e7c8e605e65306151e5a94464bb1d737848283e83949d6fc', 'GatedEngram/gpu_pilot/run_pilot.py': '67cef9b955fd00fe7c4ee08125ee13bc637d9dfaee83439a26d18bf215d6667f', 'GatedEngram/gpu_pilot/run_tail_link_pilot.py': '8ad51c4f88214b03b94db9cd023357e55535d94e75102486b6867bab6f76ac3a', 'GatedEngram/gpu_pilot/read_tail_link_results.py': 'c43346cb94fdac0c16b8d3eb91c67d67bfcc8aa3dd7c5e6680d9941d386f470a', 'GatedEngram/gpu_pilot/launch_tail_link_pilot.py': 'fb688f5982993c612b2669a537304f9ad0bfe0eeedffbf31f50d738e39a3d257', 'GatedEngram/gpu_pilot/launch_index_hit20_pilot.py': '31be2c72d9fb00af2787fb0acd7e6730da18058cc07e8473fdd203201e57c022', 'GatedEngram/gpu_pilot/launch_pilot.py': '980c9cd1eb1082412522f6fc9c49ec1db886051a3ddb3a8d3fec1039ac12f464', 'GatedEngram/kv_cache_engram/headtail_link.py': '7d1b317d989534efe30570b54a88377f4cf3ea9a88ca6dc7edbaef6f70af38f1', 'GatedEngram/kv_cache_engram/headtail_link_plan.py': 'c0b76174afe344418de623aaa2aa28a74c48f92a591836c91a580d549319ea8c', 'GatedEngram/gpu_pilot/run_headtail_link_pilot.py': '3e4f9333aa73f33310d41ab3893ca043bfd38e657639648522d7eac1edbb3d18', 'GatedEngram/gpu_pilot/read_headtail_link_results.py': '441dae217fd22d68865a9b432084457ee79b2276428462422bcd816e67e7f946', 'GatedEngram/gpu_pilot/launch_headtail_link_pilot.py': 'c50c563a0ce27b92d86f7ace2c6fea5789cd812269d3f6a7e7abeefdc2db9fe1', 'GatedEngram/validate_headtail_cpu.py': '9ae58094eaaefb6ac5fbd91ab01518a643a8e0ac27b0c9d99edc16e74a05c81b'}


def headtail_link_protocol():
    body = dict(schema="headtail-link-pilot-protocol-v1", cases=[2, 3, 4, 5],
        datasets=["hotpotqa", "2wikimqa"], arms=list(ARMS), optional_fraction=None,
        input_manifest_sha256="5125e6264397dc2bee18b23106250493868b1b82add31ccb07896524da4e0bc5",
        exploratory_previously_exposed=True, independent_holdout_claim=False,
        input_claim="LongBench-expanded eight independent32K documents plus template/question; not native256K",
        context_tokens=262144, independent_documents=8, document_tokens=32768,
        head_tokens=128, tail_tokens=128, first_full_layers=[0, 1, 2],
        ffn_layers=list(FFN_LAYERS), attention_layers=list(FFN_LAYERS), native_chunk_tokens=8192,
        selection="actual native-surviving document head128 or tail128, plus non-document rows",
        selection_depends_on_index_hits=False, selected_compute=["Q", "joint attention", "FFN/MoE"],
        unselected_compute="native residual/mHC; zero attention and FFN contribution",
        memory="per-query deduplicated union of native local SWA window and prior-document tail128",
        native_local_mask_preserved=True, extra_links_cross_yoco_rectangle=True,
        native_swa_window=128, prior_document_rule="document end <= query absolute position",
        causal_rule="each query uses only keys at absolute positions <= query; union has no duplicate keys",
        dynamic_document_swa=True, dynamic_refresh_start_layer=3, dynamic_layer3_static_link_seed=False,
        dynamic_swa_source="current layer attention input; never same-layer output or fixed shared_hidden",
        dynamic_refresh_scope="native-surviving head128 and tail128 only; no forced YOCO retention",
        dynamic_missing_yoco_rows="same-layer offline canonical fallback, explicitly counted",
        dynamic_cross_chunk="same-layer, same-request refreshed KV persists across chunks only",
        non_document_swa_online=True, global_kv_complete_online=True, global_index_native=True,
        global_source_layers_zero_based=[2, 8, 14, 20], preserve_native_global_consumer_sharing=True,
        native_yoco_keep140=True, native_yoco_swa_share_hidden=False, forced_survivor_retention=False,
        decode_native=True, prefill_only_link=True, final_decode_ring_last128_complete=True,
        offline_capture_layers=list(CAPTURE_LAYERS), offline_yoco_full_layout=True,
        offline_representation="swa-postnorm-prerope-bf16", offline_shape=[32768, 512],
        offline_payload_bytes_per_case=40*8*32768*512*2, cache_media="GPU full canonical bank",
        phase_local_memory_rebuilt=True, full_bank_immutable_between_phases=True,
        empty_ep_qualification=dict(layers=list(FFN_LAYERS), patterns=4, ranks=8, checks=1184,
                                    byteexact=True, native_collectives_even_zero=True, sync_debug=2),
        new_parity=dict(input_tokens=16384, output_tokens=1, layer_chunk_checks=80,
                       all_four_mhc_fields=True, byteexact=True, split_restore_exact_zero=True),
        sparse_parity=dict(case_index=2, input_tokens=262194, output_tokens=1,
                          layer_chunk_checks=1320, all_four_mhc_fields=True, byteexact=True,
                          same_policy_slow_reference=True, arms=list(ARMS), equivalence_to_full_model_claim=False),
        partial_smoke=dict(case_index=2, arms=list(ARMS), output_tokens=1,
                           quality_eligible=False, timing_valid=False, must_precede_formal=True),
        quality=dict(max_tokens=64, eos=True, gold_metrics=["EM", "F1"], keep_truncations=True),
        timing=dict(concurrency=1, warmups_per_arm=1, repeats=3, output_tokens=32,
                    start="before controller reset", order="candidate repeats0,1,2",
                    includes="reset,full-bank lookup,memory union,dynamic projection,RoPE,cast,Q,attention,FFN,mHC,communication",
                    excludes="model load,offline construction,qualification,smoke,post-generation evidence export",
                    fresh_paired_baseline=False, historical_comparisons_only=True),
        supervisor_timeout_seconds=5400, supervisor_timeout_start="after GPU release",
        prepare_timeout_seconds=300, no_accuracy_or_speed_guarantee=True,
        failure="abort without automatic retry or tolerance relaxation")
    return dict(body, sha256=digest(body))


def timing_schedule(index):
    if type(index) is not int or index not in (2, 3, 4, 5):
        raise ValueError("unregistered case")
    return [(repeat, ARMS[0]) for repeat in range(3)]
