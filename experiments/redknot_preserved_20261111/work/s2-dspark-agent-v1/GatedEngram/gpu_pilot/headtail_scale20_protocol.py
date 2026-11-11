"""Fresh twenty-case three-arm comparison with one shared immutable GPU bank."""
from kv_cache_engram.core import digest

ARMS = ("recomputed", "tail128_link", "headtail128_link")
POLICY_ARMS = ARMS[1:]
EVALUATION_INDICES = tuple(range(2, 22))
FFN_LAYERS = tuple(range(3, 40))
CAPTURE_LAYERS = tuple(range(40))
FRACTIONS = dict.fromkeys(ARMS)
DEPENDENCIES = {'benchmark/pr148_native_runner.py': '615315c76ee54320201daa918d462676cd3c59cb05000769396e3253f0886a47', 'benchmark/prepare_rag256.py': '839b268e295e460eb49440f9a59c2951e7f8721707f379551553f76c8565133e', 'strong_native_20261029/strong_native_runner.py': '45f25f878ea62741cef12ca186618483531e0c968a50dc8722051b4177411a7a', 'strong_native_20261029/native_profile.py': 'e68d7d3b39c8255ef1f06d4ba42bb26e2e6121b4f141ace5da06c41060ba0717', 'kv_recompression_20261029/canonical_request_binding.py': 'f25097ab9ebe51662c21b9652af472e43d7664e368276a4c317fc133d02e913e', 'kv_recompression_20261029/quality_process_group.py': 'a7d503f23d1df3f41044c98107b25ca2a09debc1e66384d17d26a9f9f33ba9db', 'GatedEngram/kv_cache_engram/tail_link.py': 'cafe17fb5c29038b3f8de243f46b931d8c1000b5d9403bcecb0b8dde74fff5d7', 'GatedEngram/kv_cache_engram/tail_link_staging.py': '4f5bf3d745127382e792c04a6d3f50352187843559cef89ead0de9c560d524b7', 'GatedEngram/kv_cache_engram/tail_link_plan.py': 'e62c35018784618114b52cb2d10fe47911319397e252dc5e4ca3028b73a98a1c', 'GatedEngram/kv_cache_engram/index_hit20.py': '10a699cc6b69d12f01ba76e40abe8a072b5ae531e0df60643de3dc1b0147619d', 'GatedEngram/kv_cache_engram/index_hit20_adapter.py': '9dd49e07b85b610eeea3c3d6e2f177ec23973c4740e151ab128bb3a2c9cfd2ea', 'GatedEngram/kv_cache_engram/index_hit_selector.py': '6e1f930ec446186e4da9092bd97e7845b0dd82b434efcbed22faaf265f3b6220', 'GatedEngram/kv_cache_engram/native_reuse.py': '5bd7d2dea0c71eee299c39993c4f068e86af735d05663d54ab554dc2197631a7', 'GatedEngram/kv_cache_engram/early_mhc_adapter.py': '7a19b8775c1cf9a93c9dcf41dbd8e0e54816e32ccdad0a3ae3c0fb47e20b9d00', 'GatedEngram/kv_cache_engram/ffn_only_adapter.py': 'a5fe50b1bbfdc6e66488c9616b63c74e95f862e29af9c7b2ce071987e368a9c0', 'GatedEngram/kv_cache_engram/core.py': '5bd8a4201b8756b7772fc8c6a9c2b0155e8edf8a7283c369a741f2bffa3f0f41', 'GatedEngram/gpu_pilot/index_hit20_empty_ep_qualification.py': '3a8c95fcd85fc712b7eaee2747d3cb0f14dd773d904e6d4fa229c0652b80e148', 'GatedEngram/gpu_pilot/cuda_mask_qualification.py': '910254c71c5b3d97e7c8e605e65306151e5a94464bb1d737848283e83949d6fc', 'GatedEngram/gpu_pilot/run_pilot.py': '67cef9b955fd00fe7c4ee08125ee13bc637d9dfaee83439a26d18bf215d6667f', 'GatedEngram/gpu_pilot/run_tail_link_pilot.py': '8ad51c4f88214b03b94db9cd023357e55535d94e75102486b6867bab6f76ac3a', 'GatedEngram/gpu_pilot/read_tail_link_results.py': 'c43346cb94fdac0c16b8d3eb91c67d67bfcc8aa3dd7c5e6680d9941d386f470a', 'GatedEngram/gpu_pilot/launch_tail_link_pilot.py': 'fb688f5982993c612b2669a537304f9ad0bfe0eeedffbf31f50d738e39a3d257', 'GatedEngram/gpu_pilot/launch_index_hit20_pilot.py': '31be2c72d9fb00af2787fb0acd7e6730da18058cc07e8473fdd203201e57c022', 'GatedEngram/gpu_pilot/launch_pilot.py': '980c9cd1eb1082412522f6fc9c49ec1db886051a3ddb3a8d3fec1039ac12f464', 'GatedEngram/kv_cache_engram/headtail_link.py': '7d1b317d989534efe30570b54a88377f4cf3ea9a88ca6dc7edbaef6f70af38f1', 'GatedEngram/kv_cache_engram/headtail_link_plan.py': 'c0b76174afe344418de623aaa2aa28a74c48f92a591836c91a580d549319ea8c', 'GatedEngram/gpu_pilot/run_headtail_link_pilot.py': '3e4f9333aa73f33310d41ab3893ca043bfd38e657639648522d7eac1edbb3d18', 'GatedEngram/gpu_pilot/read_headtail_link_results.py': '441dae217fd22d68865a9b432084457ee79b2276428462422bcd816e67e7f946', 'GatedEngram/gpu_pilot/launch_headtail_link_pilot.py': 'c50c563a0ce27b92d86f7ace2c6fea5789cd812269d3f6a7e7abeefdc2db9fe1', 'GatedEngram/validate_headtail_cpu.py': '9ae58094eaaefb6ac5fbd91ab01518a643a8e0ac27b0c9d99edc16e74a05c81b', 'GatedEngram/kv_cache_engram/headtail_comparison.py': '2942239363b01f97d35e7032dd08f526ead158f4b2ece6680b51df8916104379', 'GatedEngram/kv_cache_engram/early_token_diagnostics.py': '6630411109a3794448d9f92bf4e5a1511d47aa657d62c73c6a3aae58663832b7', 'GatedEngram/gpu_pilot/run_headtail_scale20_pilot.py': '8d7bb87e6e4766b31f303af6b7ad7df5bdc77d69bb01c2a20f74af734177cd1b', 'GatedEngram/gpu_pilot/read_headtail_scale20_results.py': 'f22037e63258c3ff0450be1475485db6283ae2b93ffba52278bf0d2db72205a8', 'GatedEngram/gpu_pilot/launch_headtail_scale20_pilot.py': '54c973a50d51d50ed75406a4401277bb25b1de2b8e8e0896e0096c89b54ae316', 'GatedEngram/gpu_pilot/headtail_scale20_inputs.py': 'fd7b881ad02a29b0a75b60b5b40c3a829894c63451465509659b5952c7e1ae37', 'GatedEngram/validate_headtail_scale20_cpu.py': '0c08b3538ffd107712fe6f6704c1c803be989ff5472bee349af472755be54810'}
INPUT_MANIFEST_SHA256 = 'df2aff92762fdd7f407adb01652205e1ed7cae242996314748e32cdb16b47360'
DIAGNOSTIC_NATIVE_SOURCE_SHA256 = {
    "deepinfer/model_executor/models/v4_modules/attn.py": "01a807d70b71203fa45ff85d5557f2ddacabf2661f562d4f24907c742ad1bb42",
    "deepinfer/model_executor/models/v4_modules/index_attn.py": "416d09a87233034f93c42788c209516bf0b9ef9527ce662e8f0f34482449b8fc",
    "deepinfer/model_executor/models/v4_modules/yoco.py": "7dc8ad7b6ca4902bcef67b1cbc4804592995c5b6af9ebaa00b59cc338c1aa6ec",
    "deepinfer/model_executor/input_metadata/yoco.py": "882a33de3da0fba81644adde9e9e1a90ec3d3ff34a8f3ad4dda18faafebd6460",
    "deepinfer/model_executor/layers/tilelang_ops/fused_transform.py": "3483e3155ac685c317fd21f7f12d89bf24196d9f613987d7f40d34fd3c5dba68",
    "deepinfer/model_executor/layers/tilelang_ops/cuda/compress.py": "f29ab1fc97c73537bfd9b0e5be6a9f4014e9b17a1a1952c876f2dea4cf4c8b9c",
    "csrc/engine/input_metadata/prepare_tensor_input_v4.cpp": "93a6b1dd533db116d2f7540c78d6bb5419bdb8c1d2063b031ab8b1a8cc138fe2",
    "deepinfer/hugging_face/config.py": "978c863928ece0e7f81c3e619c02d1287ade442f9282d7eb67bc739411595e49",
}


def phase_policy(arm):
    if arm not in (*ARMS, "qualification", "offline"):
        raise ValueError("unregistered comparison arm")
    return dict(head_tokens=128 if arm == "headtail128_link" else 0,
                tail_tokens=128 if arm in POLICY_ARMS else 0,
                dynamic_refresh_start_layer=3 if arm == "headtail128_link" else None)


def timing_schedule(index):
    if type(index) is not int or index not in EVALUATION_INDICES:
        raise ValueError("unregistered evaluation case")
    return [(repeat, ARMS[(index-2+repeat+slot) % 3])
            for repeat in range(3) for slot in range(3)]


def headtail_scale20_protocol():
    body = dict(schema="headtail-scale20-pilot-protocol-v1", cases=list(EVALUATION_INDICES),
        datasets=["hotpotqa", "2wikimqa"], arms=list(ARMS), input_manifest_sha256=INPUT_MANIFEST_SHA256,
        calibration_indices=[0, 1], preserved_evaluation_indices=[2, 3, 4, 5], new_evaluation_indices=list(range(6,22)),
        fresh_all_twenty_cases_all_three_arms=True, historical_measurements_used=False,
        original_four_inputs_previously_exposed=True, independent_holdout_claim=False,
        expanded_donor_documents_may_repeat_within_new_sixteen=True,
        context_tokens=262144, independent_documents=8, document_tokens=32768,
        input_claim="LongBench-expanded eight32K documents plus template/question; not native256K",
        first_full_layers=[0,1,2], ffn_layers=list(FFN_LAYERS), attention_layers=list(FFN_LAYERS),
        native_chunk_tokens=8192, native_yoco_keep140=True, native_yoco_swa_share_hidden=False,
        native_global_path_preserved=True, global_source_layers_zero_based=[2,8,14,20], decode_native=True,
        policies={
            "recomputed": dict(selection="all native surviving rows", memory="native", dynamic=False),
            "tail128_link": dict(selection="native surviving tail128 plus nondocument",
                memory="static offline document tail128 plus online nondocument; no local middle memory", dynamic=False),
            "headtail128_link": dict(selection="native surviving head128 union tail128 plus nondocument",
                memory="native local SWA window union prior-completed-document tail128", dynamic=True,
                dynamic_refresh_start_layer=3, native_local_mask_preserved=True,
                extra_links_cross_yoco_rectangle=True, missing_yoco_rows="same-layer offline canonical fallback")},
        phase_policy={arm:phase_policy(arm) for arm in (*ARMS,"qualification","offline")},
        no_forced_survivor_retention=True, current_layer_hidden_and_weights_for_refresh=True,
        no_hidden_or_ffn_output_cache=True, no_adaptive_token_ablation=True,
        offline_capture_layers=list(CAPTURE_LAYERS), offline_yoco_full_layout=True,
        offline_representation="swa-postnorm-prerope-bf16", offline_shape=[32768,512],
        offline_payload_bytes_per_case=10*1024**3, cache_storage_policy="shared_gpu_full_canonical_bank_all_arms",
        bank_constant_across_arms=True, static_tail_bank_gpu_resident=True,
        phase_overlay_rebuilt=True, offline_build_in_online_ttft=False, free_disk_minimum_bytes=320*1024**3,
        empty_ep_qualification=dict(layers=list(FFN_LAYERS),patterns=4,ranks=8,checks=1184,byteexact=True),
        new_parity=dict(input_tokens=16384,output_tokens=1,layer_chunk_checks=80,all_four_mhc_fields=True,byteexact=True),
        sparse_parity=dict(case_index=2,arms=list(POLICY_ARMS),input_tokens=262194,output_tokens=1,
            layer_chunk_checks=1320,all_four_mhc_fields=True,byteexact=True,independent_policy_state=True,
            equivalence_to_full_model_claim=False),
        partial_smoke=dict(case_index=2,arms=list(POLICY_ARMS),output_tokens=1,must_precede_formal=True),
        quality=dict(max_tokens=64,eos=True,keep_truncations=True,gold_metrics=["EM","F1"]),
        diagnostics=dict(schema="early-token-diagnostics-v1", phase="separate native one-token after all15 formal phases",
            kv_error_layers=[0,1,2,3,7,13,19,20,39],prompt_chunk_tokens=8192,native_yoco_keep_tokens=140,
            native_prefix_routing_layers=[2],native_prefix_routing_compress_ratios={"2":2},
            native_prefix_routing_selection_kinds={"2":"sparse_topk"},native_index_topk=512,
            checkpoint_config_sha256="8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
            native_source_files_sha256=dict(DIAGNOSTIC_NATIVE_SOURCE_SHA256),
            first_token_must_equal_full_quality=True,excluded_from_formal_metrics=True),
        timing=dict(concurrency=1,warmups_per_arm=1,repeats=3,output_tokens=32,
            start="before controller reset",order="rotate by case then repeat; all3 arms once per repeat",
            includes="all online reset,lookup,union,projection,attention,FFN,mHC,communication",
            excludes="model load,offline construction,qualification,post-generation evidence export",
            fresh_paired_baseline=True,same_bank_resident_for_baseline=True),
        diagnostic=dict(per_case=True,after_all_formal_phases=True,arm="recomputed",mode="native",
            output_tokens=1,timing_valid=False,quality_eligible=False,
            native_first_output_token_matches_quality=True,excluded_from_formal_metrics=True,
            no_selector_or_ablation=True,overhead_separately_reported=True),
        requested_completion_cases=20, previous_ninety_minute_limit_cancelled=True,
        supervisor_timeout_seconds=86400, timeout_purpose="24hour failure backstop; normal completion exits immediately",
        supervisor_timeout_start="after GPU release",prepare_timeout_seconds=300,
        holder_during_gpu=False,restore_original_holder_after_exit=True,
        failure="abort without automatic retry or tolerance relaxation")
    return dict(body,sha256=digest(body))
