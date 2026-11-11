"""Independent tail128-link protocol. Pure CPU and no mutable algorithm flags."""
from kv_cache_engram.core import digest

ARMS = ("recomputed", "tail128_link", "tail128_dynamic")
FFN_LAYERS = tuple(range(3, 40))
CAPTURE_LAYERS = tuple(range(40))
FRACTIONS = dict.fromkeys(ARMS)
# Frozen reviewed executable dependencies; the complete stage is also bound by preflight.
DEPENDENCIES = {'benchmark/pr148_native_runner.py': '615315c76ee54320201daa918d462676cd3c59cb05000769396e3253f0886a47', 'benchmark/prepare_rag256.py': '839b268e295e460eb49440f9a59c2951e7f8721707f379551553f76c8565133e', 'strong_native_20261029/strong_native_runner.py': '45f25f878ea62741cef12ca186618483531e0c968a50dc8722051b4177411a7a', 'strong_native_20261029/native_profile.py': 'e68d7d3b39c8255ef1f06d4ba42bb26e2e6121b4f141ace5da06c41060ba0717', 'kv_recompression_20261029/canonical_request_binding.py': 'f25097ab9ebe51662c21b9652af472e43d7664e368276a4c317fc133d02e913e', 'kv_recompression_20261029/quality_process_group.py': 'a7d503f23d1df3f41044c98107b25ca2a09debc1e66384d17d26a9f9f33ba9db', 'GatedEngram/kv_cache_engram/tail_link.py': 'cafe17fb5c29038b3f8de243f46b931d8c1000b5d9403bcecb0b8dde74fff5d7', 'GatedEngram/kv_cache_engram/tail_link_staging.py': '4f5bf3d745127382e792c04a6d3f50352187843559cef89ead0de9c560d524b7', 'GatedEngram/kv_cache_engram/tail_link_plan.py': 'e62c35018784618114b52cb2d10fe47911319397e252dc5e4ca3028b73a98a1c', 'GatedEngram/kv_cache_engram/index_hit20.py': '10a699cc6b69d12f01ba76e40abe8a072b5ae531e0df60643de3dc1b0147619d', 'GatedEngram/kv_cache_engram/index_hit20_adapter.py': '9dd49e07b85b610eeea3c3d6e2f177ec23973c4740e151ab128bb3a2c9cfd2ea', 'GatedEngram/kv_cache_engram/index_hit_selector.py': '6e1f930ec446186e4da9092bd97e7845b0dd82b434efcbed22faaf265f3b6220', 'GatedEngram/kv_cache_engram/native_reuse.py': '5bd7d2dea0c71eee299c39993c4f068e86af735d05663d54ab554dc2197631a7', 'GatedEngram/kv_cache_engram/early_mhc_adapter.py': '7a19b8775c1cf9a93c9dcf41dbd8e0e54816e32ccdad0a3ae3c0fb47e20b9d00', 'GatedEngram/kv_cache_engram/ffn_only_adapter.py': 'a5fe50b1bbfdc6e66488c9616b63c74e95f862e29af9c7b2ce071987e368a9c0', 'GatedEngram/kv_cache_engram/core.py': '5bd8a4201b8756b7772fc8c6a9c2b0155e8edf8a7283c369a741f2bffa3f0f41', 'GatedEngram/gpu_pilot/index_hit20_empty_ep_qualification.py': '3a8c95fcd85fc712b7eaee2747d3cb0f14dd773d904e6d4fa229c0652b80e148', 'GatedEngram/gpu_pilot/cuda_mask_qualification.py': '910254c71c5b3d97e7c8e605e65306151e5a94464bb1d737848283e83949d6fc', 'GatedEngram/gpu_pilot/run_pilot.py': '67cef9b955fd00fe7c4ee08125ee13bc637d9dfaee83439a26d18bf215d6667f', 'GatedEngram/gpu_pilot/run_tail_link_pilot.py': '8ad51c4f88214b03b94db9cd023357e55535d94e75102486b6867bab6f76ac3a', 'GatedEngram/gpu_pilot/read_tail_link_results.py': 'c43346cb94fdac0c16b8d3eb91c67d67bfcc8aa3dd7c5e6680d9941d386f470a', 'GatedEngram/gpu_pilot/launch_tail_link_pilot.py': 'fb688f5982993c612b2669a537304f9ad0bfe0eeedffbf31f50d738e39a3d257', 'GatedEngram/gpu_pilot/launch_index_hit20_pilot.py': '31be2c72d9fb00af2787fb0acd7e6730da18058cc07e8473fdd203201e57c022', 'GatedEngram/gpu_pilot/launch_pilot.py': '980c9cd1eb1082412522f6fc9c49ec1db886051a3ddb3a8d3fec1039ac12f464'}


def tail_link_protocol():
    body = dict(
        schema="tail-link-pilot-protocol-v3", cases=[2, 3, 4, 5],
        datasets=["hotpotqa", "2wikimqa"], arms=list(ARMS), optional_fraction=None,
        input_manifest_sha256="5125e6264397dc2bee18b23106250493868b1b82add31ccb07896524da4e0bc5",
        exploratory_previously_exposed=True, independent_holdout_claim=False,
        input_claim="LongBench-expanded eight independent32K documents plus template/question; not native256K",
        context_tokens=262144, independent_documents=8, document_tokens=32768,
        tail_tokens=128, first_full_layers=[0, 1, 2], ffn_layers=list(FFN_LAYERS),
        attention_layers=list(FFN_LAYERS), native_chunk_tokens=8192,
        selection="document last128 positions and non-document rows actually surviving native YOCO",
        selection_depends_on_index_hits=False, selected_compute=["Q", "joint attention", "FFN/MoE"],
        unselected_compute="native full-row residual/mHC; zero attention and FFN contribution",
        memory="causally available document tail128 keys plus actually generated online non-document keys",
        causal_rule="each selected query consumes only keys at absolute positions <= query position",
        online_document_swa_projection_by_arm=dict(recomputed="native", tail128_link=False, tail128_dynamic=True),
        dynamic_refresh_start_layer=4, dynamic_layer3_static_link_seed=True,
        dynamic_swa_source="current layer input hidden; never current layer post-attention hidden or YOCO shared_hidden",
        dynamic_refresh_scope="actual native-surviving document tail rows only; no forced YOCO retention",
        dynamic_missing_yoco_rows="same-layer offline canonical tail fallback, explicitly counted",
        dynamic_cross_chunk="retain refreshed values only within the same layer and request phase",
        dynamic_qualification_independent_of_static=True, non_document_swa_online=True,
        global_kv_complete_online=True, global_index_native=True,
        global_source_layers_zero_based=[2, 8, 14, 20],
        preserve_native_global_consumer_sharing=True, no_cached_hidden_or_attention_output=True,
        native_yoco_keep140=True, native_yoco_swa_share_hidden=False,
        native_yoco_swa_configuration_verified="frozen HF parser AST default plus every checkpoint config override",
        forced_survivor_retention=False,
        decode_native=True, prefill_only_link=True, final_decode_ring_last128_complete=True,
        offline_capture_layers=list(CAPTURE_LAYERS), offline_yoco_full_layout=True,
        offline_representation="swa-tail128-postnorm-prerope-bf16",
        offline_shape=[128, 512], offline_source_local_start=32640,
        offline_payload_bytes_per_case=40*8*128*512*2,
        cache_media="CPU pinned memory", gpu_staging="bounded per-layer prefetch",
        warmup_gpu_payload_reused_by_formal=False, phase_local_staging=True,
        empty_ep_qualification=dict(layers=list(FFN_LAYERS), patterns=4, ranks=8, checks=1184,
                                    byteexact=True, native_collectives_even_zero=True, sync_debug=2),
        new_parity=dict(input_tokens=16384, output_tokens=1, layer_chunk_checks=80,
                       all_four_mhc_fields=True, byteexact=True, split_restore_exact_zero=True),
        sparse_parity=dict(case_index=2, input_tokens=262194, output_tokens=1,
                          layer_chunk_checks=1320, all_four_mhc_fields=True, byteexact=True,
                          same_policy_slow_reference=True, arms=list(ARMS[1:]), equivalence_to_full_model_claim=False,
                          independent_causal_tail_memory_audit=True),
        partial_smoke=dict(case_index=2, arms=list(ARMS[1:]), output_tokens=1,
                           quality_eligible=False, timing_valid=False, must_precede_formal=True),
        quality=dict(max_tokens=64, eos=True, gold_metrics=["EM", "F1"], keep_truncations=True),
        timing=dict(concurrency=1, warmups_per_arm=1, repeats=3, output_tokens=32,
                    order="rotate(recomputed,tail128_link,tail128_dynamic) by (case_index+repeat)%3",
                    start="before controller reset and first phase-local H2D",
                    includes="reset,lookup,pinned H2D,prefetch,event waits,dynamic tail projection,RoPE,cast,Q,attention,FFN,mHC,communication",
                    excludes="model load,offline construction,qualification,smoke,post-generation evidence export",
                    no_qps_claim=True, no_cold_cache_miss_claim=True),
        rates=dict(compute_denominator="sum actual native/YOCO surviving prefill rows layers3..39",
                   consumed_memory_denominator="actual selected-query SWA query-key pairs, cached tail versus online non-document; global KV excluded",
                   materialized_memory_denominator="distinct SWA memory entries materialized per layer/chunk, not consumed query-key pairs",
                   distinct_storage_denominator="8*32768 canonical document rows per layer",
                   not_flops=True),
        no_accuracy_or_speed_guarantee=True, no_statistical_generalization=True,
        failure="abort without automatic retry or tolerance relaxation")
    return dict(body, sha256=digest(body))


def timing_schedule(index):
    if type(index) is not int or index not in (2, 3, 4, 5):
        raise ValueError("unregistered case")
    return [(repeat, ARMS[(index+repeat+slot)%len(ARMS)]) for repeat in range(3) for slot in range(len(ARMS))]
