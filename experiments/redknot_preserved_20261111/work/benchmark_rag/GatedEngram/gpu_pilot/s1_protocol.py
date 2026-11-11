"""Frozen contract for S1 tail128 + preceding SWA halo128, two fresh arms."""
from kv_cache_engram.core import digest, require
from gpu_pilot.natural_boundary_protocol import (
    DIAGNOSTIC_NATIVE_SOURCE_SHA256,
)

INPUT_MANIFEST_SHA256 = 'fb6bba62937fb8005f05df9a3a213729c172932a30e5509b136000692881aa67'
DOCUMENT_COUNT = 10

ARMS = ("recomputed", "s1")
POLICY_ARMS = ("s1",)
EVALUATION_INDICES = tuple(range(2, 22))
DATASETS = ("loong",)
LENGTH_BANDS = ("128k", "64k")
TARGET_INPUT_TOKENS = {"128k": 131072, "64k": 65536}
INPUT_TOKEN_RANGES = {"128k": (112*1024, 144*1024), "64k": (56*1024, 72*1024)}
FFN_LAYERS = tuple(range(3, 40))
CAPTURE_LAYERS = tuple(range(3,40))
DEPENDENCIES = {'GatedEngram/gpu_pilot/cuda_mask_qualification.py': '910254c71c5b3d97e7c8e605e65306151e5a94464bb1d737848283e83949d6fc', 'GatedEngram/gpu_pilot/headtail_scale20_inputs.py': 'fd7b881ad02a29b0a75b60b5b40c3a829894c63451465509659b5952c7e1ae37', 'GatedEngram/gpu_pilot/index_hit20_empty_ep_qualification.py': '3a8c95fcd85fc712b7eaee2747d3cb0f14dd773d904e6d4fa229c0652b80e148', 'GatedEngram/gpu_pilot/launch_headtail_link_pilot.py': 'c50c563a0ce27b92d86f7ace2c6fea5789cd812269d3f6a7e7abeefdc2db9fe1', 'GatedEngram/gpu_pilot/launch_headtail_scale20_pilot.py': '54c973a50d51d50ed75406a4401277bb25b1de2b8e8e0896e0096c89b54ae316', 'GatedEngram/gpu_pilot/launch_index_hit20_pilot.py': '31be2c72d9fb00af2787fb0acd7e6730da18058cc07e8473fdd203201e57c022', 'GatedEngram/gpu_pilot/launch_natural_boundary_pilot.py': '9f4aad3970b03c60f833ce8e5a91b3f5db5ae276d74f878d70b016fcc19d2a69', 'GatedEngram/gpu_pilot/launch_pilot.py': '980c9cd1eb1082412522f6fc9c49ec1db886051a3ddb3a8d3fec1039ac12f464', 'GatedEngram/gpu_pilot/launch_tail_link_pilot.py': 'fb688f5982993c612b2669a537304f9ad0bfe0eeedffbf31f50d738e39a3d257', 'GatedEngram/gpu_pilot/natural_boundary_inputs.py': '7177eb018e6a1756f116a98754af0c941417cf6de0567256c90a3617d61ba224', 'GatedEngram/gpu_pilot/read_headtail_link_results.py': '441dae217fd22d68865a9b432084457ee79b2276428462422bcd816e67e7f946', 'GatedEngram/gpu_pilot/read_headtail_scale20_results.py': 'f22037e63258c3ff0450be1475485db6283ae2b93ffba52278bf0d2db72205a8', 'GatedEngram/gpu_pilot/read_natural_boundary_results.py': 'c8366e31103fe3ed6399435fda36ae9ea2ad877b88ade90955ed8576f55d41d3', 'GatedEngram/gpu_pilot/read_tail_link_results.py': 'c43346cb94fdac0c16b8d3eb91c67d67bfcc8aa3dd7c5e6680d9941d386f470a', 'GatedEngram/gpu_pilot/run_headtail_link_pilot.py': '3e4f9333aa73f33310d41ab3893ca043bfd38e657639648522d7eac1edbb3d18', 'GatedEngram/gpu_pilot/run_headtail_scale20_pilot.py': '8d7bb87e6e4766b31f303af6b7ad7df5bdc77d69bb01c2a20f74af734177cd1b', 'GatedEngram/gpu_pilot/run_natural_boundary_pilot.py': '385081d71f4f2cc03ad0d5bf2cae7b47450ac982285d4093a2a78788bbd7902f', 'GatedEngram/gpu_pilot/run_pilot.py': '67cef9b955fd00fe7c4ee08125ee13bc637d9dfaee83439a26d18bf215d6667f', 'GatedEngram/gpu_pilot/run_tail_link_pilot.py': '8ad51c4f88214b03b94db9cd023357e55535d94e75102486b6867bab6f76ac3a', 'GatedEngram/kv_cache_engram/core.py': '5bd8a4201b8756b7772fc8c6a9c2b0155e8edf8a7283c369a741f2bffa3f0f41', 'GatedEngram/kv_cache_engram/early_mhc_adapter.py': '7a19b8775c1cf9a93c9dcf41dbd8e0e54816e32ccdad0a3ae3c0fb47e20b9d00', 'GatedEngram/kv_cache_engram/early_token_diagnostics.py': '8f8589248c90fca6ed38f9c80d20e202ed96506cae201acc4252dbfd6fbb50cd', 'GatedEngram/kv_cache_engram/ffn_only_adapter.py': 'a5fe50b1bbfdc6e66488c9616b63c74e95f862e29af9c7b2ce071987e368a9c0', 'GatedEngram/kv_cache_engram/headtail_comparison.py': 'a28f7985e5d5682d0305211407d0baad4e4c62948f0f4b354f2b9acdba5d9ac5', 'GatedEngram/kv_cache_engram/headtail_link.py': '8b214f3f84ca2a07b2beb768bc05130e2f6b47c74f51c9ff9bddd3fdcaeb564a', 'GatedEngram/kv_cache_engram/headtail_link_plan.py': 'c0b76174afe344418de623aaa2aa28a74c48f92a591836c91a580d549319ea8c', 'GatedEngram/kv_cache_engram/index_hit20.py': '0dd0f5c8d2df642c2139a0478e9246ef145091e8b905e12c4ba173aa62c061df', 'GatedEngram/kv_cache_engram/index_hit20_adapter.py': '9dd49e07b85b610eeea3c3d6e2f177ec23973c4740e151ab128bb3a2c9cfd2ea', 'GatedEngram/kv_cache_engram/index_hit_selector.py': '6e1f930ec446186e4da9092bd97e7845b0dd82b434efcbed22faaf265f3b6220', 'GatedEngram/kv_cache_engram/native_reuse.py': '7d7b0f2b301d46a862190142089706724eb6794bbd1dd26f15e224b31cc1ea43', 'GatedEngram/kv_cache_engram/tail_link.py': 'cafe17fb5c29038b3f8de243f46b931d8c1000b5d9403bcecb0b8dde74fff5d7', 'GatedEngram/kv_cache_engram/tail_link_plan.py': 'e62c35018784618114b52cb2d10fe47911319397e252dc5e4ca3028b73a98a1c', 'GatedEngram/kv_cache_engram/tail_link_staging.py': '4f5bf3d745127382e792c04a6d3f50352187843559cef89ead0de9c560d524b7', 'GatedEngram/validate_headtail_cpu.py': '9ae58094eaaefb6ac5fbd91ab01518a643a8e0ac27b0c9d99edc16e74a05c81b', 'GatedEngram/validate_headtail_scale20_cpu.py': '0c08b3538ffd107712fe6f6704c1c803be989ff5472bee349af472755be54810', 'GatedEngram/validate_natural_boundary_cpu.py': '6ccf0b8f8e3b9ca292c43a78d2d81bcbab3758ede5e03c33ca7789936520d219', 'benchmark/pr148_native_runner.py': '615315c76ee54320201daa918d462676cd3c59cb05000769396e3253f0886a47', 'benchmark/prepare_rag256.py': '839b268e295e460eb49440f9a59c2951e7f8721707f379551553f76c8565133e', 'kv_recompression_20261029/canonical_request_binding.py': 'f25097ab9ebe51662c21b9652af472e43d7664e368276a4c317fc133d02e913e', 'kv_recompression_20261029/quality_process_group.py': 'a7d503f23d1df3f41044c98107b25ca2a09debc1e66384d17d26a9f9f33ba9db', 'strong_native_20261029/native_profile.py': 'e68d7d3b39c8255ef1f06d4ba42bb26e2e6121b4f141ace5da06c41060ba0717', 'strong_native_20261029/strong_native_runner.py': '45f25f878ea62741cef12ca186618483531e0c968a50dc8722051b4177411a7a', 'GatedEngram/kv_cache_engram/s1_runtime.py': '7fcdde36fffe69da9d902bd449b7094bd91798c8904cca655599123e12d9ca0b', 'GatedEngram/kv_cache_engram/s1_attention.py': '3c47ef7119ee16a2764e1203d10b2af86eb00935a36af5566eb00db225cabebb', 'GatedEngram/kv_cache_engram/s1_cache.py': '8eab07b22e97aa7688c5d5235da0d3a0df7ef71358ad03eecb81f87074087b4f', 'GatedEngram/kv_cache_engram/s1_plan.py': 'b7a576332933ee11bf39ffe080856d1a5c9b889b468161cc0957099d478f9fc3', 'GatedEngram/gpu_pilot/read_s1_results.py': 'c711921ad41252034d2f29af39fca3e130360ec855828de4a9de2a686224c52d', 'GatedEngram/gpu_pilot/run_s1_pilot.py': '3a13f3c140d2e043a2c4fb8bf0dfdf62a8604b3b37058e72ebe82ad0e679cd79', 'GatedEngram/gpu_pilot/launch_s1_pilot.py': '93d51ea40e89086116318052bace353097493d57736720daa0fc962f889249b7', 'GatedEngram/validate_s1_cpu.py': '275b346384217b128291b778d7415372a77fe86dd19de5190e203977492bed8d', 'GatedEngram/gpu_pilot/musique30_inputs.py': '6baaac6dcbd71adffbf6cccf5dd115e43731bdc6b6cdd2c907aaefea14db611a', 'GatedEngram/gpu_pilot/musique_longform_inputs.py': '7afafb767055f8ee35152d306bc678899a5a2b5491ddccf588ac688da4647f6c', 'GatedEngram/gpu_pilot/token_divergence.py': 'b9ca38135ad6b63ae2efb50be538ecd1d6944ae7aac6a1ffa16afe523854423f', 'GatedEngram/gpu_pilot/broadqa_inputs.py': '399aded51fb3333e3e0a6fc79c6c9cac27b77fc46997819e224309eb23659791', 'GatedEngram/gpu_pilot/longcontext_inputs.py': 'eeed83420cc207e22a4d51c1aefd26be3f73df847e37e49541c030bc4c176016', 'GatedEngram/gpu_pilot/longcontext_metrics.py': 'ef9eb0f298a4ddc860d82ac349f4ab17621f516b25cc27176200f5810130f664', 'GatedEngram/gpu_pilot/loong_metrics.py': '85429b7a82dd1326f41488eac2ae87b055485614eab5861b26b6578a650b636a', 'GatedEngram/gpu_pilot/loong_inputs.py': '1377015414ced165fa47641952b1ec0d4051cdfc1ec4eb9c443050f5d91ffdf7'}
CORE_SOURCE_SHA256 = {'kv_cache_engram/s1_runtime.py': '7fcdde36fffe69da9d902bd449b7094bd91798c8904cca655599123e12d9ca0b', 'kv_cache_engram/s1_attention.py': '3c47ef7119ee16a2764e1203d10b2af86eb00935a36af5566eb00db225cabebb', 'kv_cache_engram/s1_cache.py': '8eab07b22e97aa7688c5d5235da0d3a0df7ef71358ad03eecb81f87074087b4f', 'kv_cache_engram/s1_plan.py': 'b7a576332933ee11bf39ffe080856d1a5c9b889b468161cc0957099d478f9fc3', 'kv_cache_engram/native_reuse.py': '7d7b0f2b301d46a862190142089706724eb6794bbd1dd26f15e224b31cc1ea43', 'kv_cache_engram/tail128_global_cache_plan.py': '29225efcc150e75b5df276abb9d486d44ad6a4360896748e59436390a50887b7'}
NATIVE_SNAPSHOT_SHA256 = 'd89dad6c23f9df603b7ba27ca3d07717c315bc9d4dcbf7471346552b300cb94f'
CORE_POLICY_KEYS = ('scheme', 'arms', 'document_count', 'first_full_layers', 'compact_layers', 'phase_policy', 'selection', 'selection_is_adaptive', 'tail_tokens', 'swa_halo_tokens', 'selected_rows_preserved_through_layer39', 'native_yoco140_intersection_applied', 'unselected_deep_attention_rows', 'unselected_deep_ffn_rows', 'unselected_deep_residual_rows', 'unselected_deep_mhc_rows', 'full_hidden_scatter_after_each_layer', 'native_engram_layers', 'deep_engram_selected_rows_only', 'empty_engram_distributed_hook_consumed_without_fake_rows', 'local_swa_mask', 'extra_cross_document_tail_links', 'local_kv_coverage_required_for_all_selected_queries', 'offline_swa_scope', 'offline_swa_layers', 'offline_swa_dtype', 'offline_swa_width', 'offline_global_source_layers', 'offline_global_dtypes', 'offline_global_widths', 'offline_bytes_formula', 'online_swa_scope', 'native_first3_source2_preserved', 'global_source_layers_zero_based', 'global_source2', 'deep_global_sources', 'offline_global', 'affected_global_groups', 'global_compression_alignment', 'source20_compression_ratio', 'source8_14_compression_ratio', 'selected_global_member_offline_fallback_allowed', 'native_chunk_tokens', 'native_yoco_swa_share_hidden', 'native_topology', 'empty_ep_qualification', 'new_parity', 'sparse_parity', 'required_runtime_qualification_patterns', 'qualification_must_precede_formal', 'quality', 'timing', 'cache_storage_policy')
CORE_POLICY_SHA256 = '82273f5646f23e7b59fb4830ec1c3d7ac155e4326a08fdb57763b9379e4438f9'

REQUIRED_NEW_DEPENDENCIES = (
    "GatedEngram/kv_cache_engram/s1_runtime.py",
    "GatedEngram/kv_cache_engram/s1_plan.py",
    "GatedEngram/kv_cache_engram/s1_cache.py",
    "GatedEngram/kv_cache_engram/s1_attention.py",
    "GatedEngram/gpu_pilot/loong_inputs.py",
    "GatedEngram/gpu_pilot/loong_metrics.py",
    "GatedEngram/gpu_pilot/token_divergence.py",
    "GatedEngram/gpu_pilot/run_s1_pilot.py",
    "GatedEngram/gpu_pilot/read_s1_results.py",
    "GatedEngram/gpu_pilot/launch_s1_pilot.py",
    "GatedEngram/validate_s1_cpu.py",
)


def phase_policy(arm):
    if arm not in (*ARMS, "qualification", "offline"):
        raise ValueError("unregistered S1 arm")
    return dict(head_tokens=0, tail_tokens=128 if arm == "s1" else 0,
                swa_halo_tokens=128 if arm == "s1" else 0,
                compact_start_layer=3 if arm == "s1" else None)


def timing_schedule(index):
    if type(index) is not int or index not in EVALUATION_INDICES:
        raise ValueError("unregistered S1 evaluation case")
    return [(repeat, ARMS[(index-2+repeat+slot) % 2])
            for repeat in range(3) for slot in range(2)]


def s1_protocol():
    body = dict(schema="loong-s1-pilot-protocol-v1", scheme="S1", arms=list(ARMS),
        cases=list(EVALUATION_INDICES), calibration_indices=[0, 1],
        datasets=list(DATASETS), document_count=DOCUMENT_COUNT, input_manifest_sha256=INPUT_MANIFEST_SHA256,
        input_contract="one original Loong QA with 5-10 original documents subdivided into ten natural cache blocks",
        original_document_count_range=[5,10], cache_block_count=10,
        cache_blocks_are_original_documents=False, cache_blocks_cross_original_document_boundaries=False,
        natural_boundary_inputs=True, full_original_context_preserved=True,
        filler_added=False, content_truncated=False, historical_measurements_used=False,
        fresh_all_cases_both_arms=True, independent_holdout_claim=False,
        first_full_layers=[0,1,2], compact_layers=list(FFN_LAYERS),
        phase_policy={arm:phase_policy(arm) for arm in (*ARMS,"qualification","offline")},
        selection="each block last128 original positions union all nondocument positions",
        selection_is_adaptive=False, tail_tokens=128, swa_halo_tokens=128,
        selected_rows_preserved_through_layer39=True, native_yoco140_intersection_applied=False,
        unselected_deep_attention_rows=0, unselected_deep_ffn_rows=0,
        unselected_deep_residual_rows=0, unselected_deep_mhc_rows=0,
        full_hidden_scatter_after_each_layer=False,
        native_engram_layers=[1,14],deep_engram_selected_rows_only=True,
        empty_engram_distributed_hook_consumed_without_fake_rows=True,
        local_swa_mask="causal original absolute window128; no cross-document extra links",
        extra_cross_document_tail_links=False, local_kv_coverage_required_for_all_selected_queries=True,
        offline_swa_scope="per-layer preceding128 halo, retained at original absolute positions",
        offline_swa_layers=list(FFN_LAYERS),offline_swa_dtype="bfloat16",offline_swa_width=512,
        offline_global_source_layers=[8,14,20],offline_global_dtypes={"8":"float32","14":"float32","20":"bfloat16"},
        offline_global_widths={"8":1024,"14":1024,"20":512},
        offline_bytes_formula="37 * 10 * 128 * 512 * 2 + sum(document_tokens) * 9216",
        online_swa_scope="current-layer selected input hidden and current-layer weights",
        native_first3_source2_preserved=True, global_source_layers_zero_based=[2,8,14,20],
        global_source2="immutable native first3 full-request cache",
        deep_global_sources=[8,14,20], offline_global="per-source raw projections before compression",
        affected_global_groups="selected current-source members + immutable offline unselected members",
        global_compression_alignment="absolute request coordinates; never reset at document boundary",
        source20_compression_ratio=1, source8_14_compression_ratio=2,
        selected_global_member_offline_fallback_allowed=False,
        full_model_numerical_equivalence_claim=False, same_policy_parity_is_quality_guarantee=False,
        native_chunk_tokens=8192, native_yoco_swa_share_hidden=False,
        native_topology=dict(checkpoint_config_sha256="8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
            native_source_files_sha256=dict(DIAGNOSTIC_NATIVE_SOURCE_SHA256),
            native_prefix_routing_layers=[2], native_prefix_routing_compress_ratios={"2":2},
            native_prefix_routing_selection_kinds={"2":"sparse_topk"},native_index_topk=512),
        empty_ep_qualification=dict(layers=list(FFN_LAYERS),patterns=4,ranks=8,checks=1184,byteexact=True),
        new_parity=dict(input_tokens=16384,output_tokens=1,layer_chunk_checks=80,
            all_four_mhc_fields=True,byteexact=True),
        sparse_parity=dict(case_index=2,output_tokens=1,geometry="full input and selected original positions",
            all_four_mhc_fields=True,byteexact=True,source2_immutable=True,
            equivalence_to_full_model_claim=False),
        required_runtime_qualification_patterns=["nonempty_compact_chunk","empty_compact_chunk",
            "nondocument_gaps","absolute_causality","mixed_compression_group"],
        qualification_must_precede_formal=True,
        quality=dict(max_tokens=None,eos=True,stop_policy="natural_eos_or_engine_context_limit",
            engine_max_model_len=270336,model_max_position_embeddings=1048576,
            technical_max_new_tokens_formula="engine_max_model_len - actual_input_tokens",
            bootstrap_validation_max_new_tokens=64,bootstrap_is_quality_output_cap=False,
            preserve_full_output_text=True,preserve_full_output_ids=True,keep_truncations=True,gold_metrics=["EM","F1"]),
        token_comparison=dict(source="actual quality output_ids including terminal EOS",
            analysis="CPU only; no extra model phase", compute_first_token_match=True, compute_first_divergence=True,
            same_prefix_or_length_end_distinguished=True, quality_guarantee=False),
        length_bands=list(LENGTH_BANDS),
        length_band_order=[LENGTH_BANDS[i % len(LENGTH_BANDS)] for i in range(len(EVALUATION_INDICES))],
        cases_per_length_band=len(EVALUATION_INDICES)//len(LENGTH_BANDS),
        target_input_tokens=dict(TARGET_INPUT_TOKENS),
        input_token_ranges={band:list(bounds) for band,bounds in INPUT_TOKEN_RANGES.items()},
        length_band_basis="actual checkpoint-tokenized complete prompt; no padding or truncation",
        exact_target_length_claim=False,paired_across_length_bands=False,
        longest_length_band_qualified_first=True,
        question_format="original Loong question, instruction and prompt template; original structured gold preserved",
        source_dataset="Loong", official_external_model_judge_invoked=False,
        generated_length_guaranteed=False,
        timing=dict(concurrency=1,warmups_per_arm=1,repeats=3,output_tokens=32,
            order="rotate case and repeat, both arms once per repeat",start="before controller reset",
            includes="all online setup, cache assembly, compaction, attention, FFN, residual, mHC and first token",
            excludes="model load, offline bank construction, qualification, post-generation evidence export",
            fresh_paired_baseline=True,same_bank_resident_for_baseline=True),
        offline_build_in_online_ttft=False, offline_build_cost_reported=True,
        offline_storage_bytes_reported=True, cold_cache_ttft_claim=False,
        diagnostics_repeated=False, first_case_reference_conflict_annotation=False, historical_gold_conflict_annotations=False,
        reference_conflict_changes_automatic_scores=False,
        cache_storage_policy="same_s1_offline_gpu_bank_resident_both_arms",
        free_disk_policy=dict(minimum_bytes=32*1024**3,payload_multiplier_numerator=5,
            payload_multiplier_denominator=4,evidence_reserve_bytes=8*1024**3),
        supervisor_timeout_seconds=7200,supervisor_timeout_is_maximum=True,prepare_timeout_seconds=300,
        timeout_origin="original supervisor GPU release", absolute_deadline_unix=None,
        timeout_purpose="two-hour GPU execution bound; original supervisor restores holder after completion or timeout",
        core_source_sha256=dict(CORE_SOURCE_SHA256),core_policy_sha256=CORE_POLICY_SHA256,
        native_snapshot_sources_sha256=NATIVE_SNAPSHOT_SHA256,
        generation_sampling=dict(temperature=0.0,top_p=1.0,top_k=-1,seq_seed=0),
        holder_during_gpu=False,
        restore_original_holder_after_exit=True,failure="abort; no automatic retry or tolerance relaxation")
    require(digest({key:body[key] for key in CORE_POLICY_KEYS})==CORE_POLICY_SHA256,
            "previously completed S1 algorithm/quality/timing policy changed")
    return dict(body,sha256=digest(body))
