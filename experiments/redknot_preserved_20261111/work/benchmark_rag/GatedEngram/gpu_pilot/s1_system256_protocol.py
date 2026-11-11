"""Independent three-arm system optimization contract; original S1 stays frozen."""
from copy import deepcopy

from kv_cache_engram.core import digest, require
from gpu_pilot import s1_protocol as original

INPUT_MANIFEST_SHA256 = '755b78a6f187cd18c16f0c0b34e186945a36f11551f12bd8b0ec81367a5399ee'
DOCUMENT_COUNT = 10
ARMS = ("recomputed", "s1", "s1_system")
POLICY_ARMS = ARMS[1:]
EVALUATION_INDICES = (2, 3, 4)
DATASETS = ("loong",)
LENGTH_BANDS = ("256k",)
TARGET_INPUT_TOKENS = {"256k": 262144}
INPUT_TOKEN_RANGES = {"256k": (240*1024, 260*1024)}
SYSTEM_BATCH_GROUP_CHUNKS = 4
SYSTEM_QUALIFICATION_GROUPS = (1, 2, 4)
CAPTURE_LAYERS = FFN_LAYERS = tuple(range(3, 40))
CORE_SOURCE_SHA256 = original.CORE_SOURCE_SHA256
CORE_POLICY_SHA256 = original.CORE_POLICY_SHA256
NATIVE_SNAPSHOT_SHA256 = original.NATIVE_SNAPSHOT_SHA256
CORE_POLICY_KEYS = original.CORE_POLICY_KEYS
# Root freezes all new and inherited source pins only after interfaces settle.
DEPENDENCIES = {'GatedEngram/gpu_pilot/cuda_mask_qualification.py': '910254c71c5b3d97e7c8e605e65306151e5a94464bb1d737848283e83949d6fc', 'GatedEngram/gpu_pilot/headtail_scale20_inputs.py': 'fd7b881ad02a29b0a75b60b5b40c3a829894c63451465509659b5952c7e1ae37', 'GatedEngram/gpu_pilot/index_hit20_empty_ep_qualification.py': '3a8c95fcd85fc712b7eaee2747d3cb0f14dd773d904e6d4fa229c0652b80e148', 'GatedEngram/gpu_pilot/launch_headtail_link_pilot.py': 'c50c563a0ce27b92d86f7ace2c6fea5789cd812269d3f6a7e7abeefdc2db9fe1', 'GatedEngram/gpu_pilot/launch_headtail_scale20_pilot.py': '54c973a50d51d50ed75406a4401277bb25b1de2b8e8e0896e0096c89b54ae316', 'GatedEngram/gpu_pilot/launch_index_hit20_pilot.py': '31be2c72d9fb00af2787fb0acd7e6730da18058cc07e8473fdd203201e57c022', 'GatedEngram/gpu_pilot/launch_natural_boundary_pilot.py': '9f4aad3970b03c60f833ce8e5a91b3f5db5ae276d74f878d70b016fcc19d2a69', 'GatedEngram/gpu_pilot/launch_pilot.py': '980c9cd1eb1082412522f6fc9c49ec1db886051a3ddb3a8d3fec1039ac12f464', 'GatedEngram/gpu_pilot/launch_tail_link_pilot.py': 'fb688f5982993c612b2669a537304f9ad0bfe0eeedffbf31f50d738e39a3d257', 'GatedEngram/gpu_pilot/natural_boundary_inputs.py': '7177eb018e6a1756f116a98754af0c941417cf6de0567256c90a3617d61ba224', 'GatedEngram/gpu_pilot/read_headtail_link_results.py': '441dae217fd22d68865a9b432084457ee79b2276428462422bcd816e67e7f946', 'GatedEngram/gpu_pilot/read_headtail_scale20_results.py': 'f22037e63258c3ff0450be1475485db6283ae2b93ffba52278bf0d2db72205a8', 'GatedEngram/gpu_pilot/read_natural_boundary_results.py': 'c8366e31103fe3ed6399435fda36ae9ea2ad877b88ade90955ed8576f55d41d3', 'GatedEngram/gpu_pilot/read_tail_link_results.py': 'c43346cb94fdac0c16b8d3eb91c67d67bfcc8aa3dd7c5e6680d9941d386f470a', 'GatedEngram/gpu_pilot/run_headtail_link_pilot.py': '3e4f9333aa73f33310d41ab3893ca043bfd38e657639648522d7eac1edbb3d18', 'GatedEngram/gpu_pilot/run_headtail_scale20_pilot.py': '8d7bb87e6e4766b31f303af6b7ad7df5bdc77d69bb01c2a20f74af734177cd1b', 'GatedEngram/gpu_pilot/run_natural_boundary_pilot.py': '385081d71f4f2cc03ad0d5bf2cae7b47450ac982285d4093a2a78788bbd7902f', 'GatedEngram/gpu_pilot/run_pilot.py': '67cef9b955fd00fe7c4ee08125ee13bc637d9dfaee83439a26d18bf215d6667f', 'GatedEngram/gpu_pilot/run_tail_link_pilot.py': '8ad51c4f88214b03b94db9cd023357e55535d94e75102486b6867bab6f76ac3a', 'GatedEngram/kv_cache_engram/core.py': '5bd8a4201b8756b7772fc8c6a9c2b0155e8edf8a7283c369a741f2bffa3f0f41', 'GatedEngram/kv_cache_engram/early_mhc_adapter.py': '7a19b8775c1cf9a93c9dcf41dbd8e0e54816e32ccdad0a3ae3c0fb47e20b9d00', 'GatedEngram/kv_cache_engram/early_token_diagnostics.py': '8f8589248c90fca6ed38f9c80d20e202ed96506cae201acc4252dbfd6fbb50cd', 'GatedEngram/kv_cache_engram/ffn_only_adapter.py': 'a5fe50b1bbfdc6e66488c9616b63c74e95f862e29af9c7b2ce071987e368a9c0', 'GatedEngram/kv_cache_engram/headtail_comparison.py': 'a28f7985e5d5682d0305211407d0baad4e4c62948f0f4b354f2b9acdba5d9ac5', 'GatedEngram/kv_cache_engram/headtail_link.py': '8b214f3f84ca2a07b2beb768bc05130e2f6b47c74f51c9ff9bddd3fdcaeb564a', 'GatedEngram/kv_cache_engram/headtail_link_plan.py': 'c0b76174afe344418de623aaa2aa28a74c48f92a591836c91a580d549319ea8c', 'GatedEngram/kv_cache_engram/index_hit20.py': '0dd0f5c8d2df642c2139a0478e9246ef145091e8b905e12c4ba173aa62c061df', 'GatedEngram/kv_cache_engram/index_hit20_adapter.py': '9dd49e07b85b610eeea3c3d6e2f177ec23973c4740e151ab128bb3a2c9cfd2ea', 'GatedEngram/kv_cache_engram/index_hit_selector.py': '6e1f930ec446186e4da9092bd97e7845b0dd82b434efcbed22faaf265f3b6220', 'GatedEngram/kv_cache_engram/native_reuse.py': '7d7b0f2b301d46a862190142089706724eb6794bbd1dd26f15e224b31cc1ea43', 'GatedEngram/kv_cache_engram/tail_link.py': 'cafe17fb5c29038b3f8de243f46b931d8c1000b5d9403bcecb0b8dde74fff5d7', 'GatedEngram/kv_cache_engram/tail_link_plan.py': 'e62c35018784618114b52cb2d10fe47911319397e252dc5e4ca3028b73a98a1c', 'GatedEngram/kv_cache_engram/tail_link_staging.py': '4f5bf3d745127382e792c04a6d3f50352187843559cef89ead0de9c560d524b7', 'GatedEngram/validate_headtail_cpu.py': '9ae58094eaaefb6ac5fbd91ab01518a643a8e0ac27b0c9d99edc16e74a05c81b', 'GatedEngram/validate_headtail_scale20_cpu.py': '0c08b3538ffd107712fe6f6704c1c803be989ff5472bee349af472755be54810', 'GatedEngram/validate_natural_boundary_cpu.py': '6ccf0b8f8e3b9ca292c43a78d2d81bcbab3758ede5e03c33ca7789936520d219', 'benchmark/pr148_native_runner.py': '615315c76ee54320201daa918d462676cd3c59cb05000769396e3253f0886a47', 'benchmark/prepare_rag256.py': '839b268e295e460eb49440f9a59c2951e7f8721707f379551553f76c8565133e', 'kv_recompression_20261029/canonical_request_binding.py': 'f25097ab9ebe51662c21b9652af472e43d7664e368276a4c317fc133d02e913e', 'kv_recompression_20261029/quality_process_group.py': 'a7d503f23d1df3f41044c98107b25ca2a09debc1e66384d17d26a9f9f33ba9db', 'strong_native_20261029/native_profile.py': 'e68d7d3b39c8255ef1f06d4ba42bb26e2e6121b4f141ace5da06c41060ba0717', 'strong_native_20261029/strong_native_runner.py': '45f25f878ea62741cef12ca186618483531e0c968a50dc8722051b4177411a7a', 'GatedEngram/kv_cache_engram/s1_runtime.py': '7fcdde36fffe69da9d902bd449b7094bd91798c8904cca655599123e12d9ca0b', 'GatedEngram/kv_cache_engram/s1_attention.py': '3c47ef7119ee16a2764e1203d10b2af86eb00935a36af5566eb00db225cabebb', 'GatedEngram/kv_cache_engram/s1_cache.py': '8eab07b22e97aa7688c5d5235da0d3a0df7ef71358ad03eecb81f87074087b4f', 'GatedEngram/kv_cache_engram/s1_plan.py': 'b7a576332933ee11bf39ffe080856d1a5c9b889b468161cc0957099d478f9fc3', 'GatedEngram/gpu_pilot/read_s1_results.py': 'c711921ad41252034d2f29af39fca3e130360ec855828de4a9de2a686224c52d', 'GatedEngram/gpu_pilot/run_s1_pilot.py': '3a13f3c140d2e043a2c4fb8bf0dfdf62a8604b3b37058e72ebe82ad0e679cd79', 'GatedEngram/gpu_pilot/launch_s1_pilot.py': '93d51ea40e89086116318052bace353097493d57736720daa0fc962f889249b7', 'GatedEngram/validate_s1_cpu.py': '275b346384217b128291b778d7415372a77fe86dd19de5190e203977492bed8d', 'GatedEngram/gpu_pilot/musique30_inputs.py': '6baaac6dcbd71adffbf6cccf5dd115e43731bdc6b6cdd2c907aaefea14db611a', 'GatedEngram/gpu_pilot/musique_longform_inputs.py': '7afafb767055f8ee35152d306bc678899a5a2b5491ddccf588ac688da4647f6c', 'GatedEngram/gpu_pilot/token_divergence.py': 'b9ca38135ad6b63ae2efb50be538ecd1d6944ae7aac6a1ffa16afe523854423f', 'GatedEngram/gpu_pilot/broadqa_inputs.py': '399aded51fb3333e3e0a6fc79c6c9cac27b77fc46997819e224309eb23659791', 'GatedEngram/gpu_pilot/longcontext_inputs.py': 'eeed83420cc207e22a4d51c1aefd26be3f73df847e37e49541c030bc4c176016', 'GatedEngram/gpu_pilot/longcontext_metrics.py': 'ef9eb0f298a4ddc860d82ac349f4ab17621f516b25cc27176200f5810130f664', 'GatedEngram/gpu_pilot/loong_metrics.py': '85429b7a82dd1326f41488eac2ae87b055485614eab5861b26b6578a650b636a', 'GatedEngram/gpu_pilot/loong_inputs.py': '1377015414ced165fa47641952b1ec0d4051cdfc1ec4eb9c443050f5d91ffdf7', 'GatedEngram/tests/test_s1_system_protocol.py': 'c799656894aa7447cefe1231abcdf137028331a7089466640d6f113df19fefac', 'GatedEngram/gpu_pilot/read_s1_system_results.py': '12d93ceb14fd7ef1605f558b16994f2ea06c4ecf1ef601d7fd50561c7b850857', 'GatedEngram/gpu_pilot/s1_system_lifecycle.py': 'ea4d89175dc27a0496e7187fb52eb738e435b3a1d50e8fe4ae9925eb43bf7591', 'GatedEngram/kv_cache_engram/s1_system_batch.py': 'fd694134cae34da2cb8baac966c7020f7e4970599ea72a115452c7f832b77149', 'GatedEngram/gpu_pilot/s1_system_workflow.py': '3299372dfb46495029db2fdd40057cc154c713ff585b1a7653e6d0ab841966e0', 'GatedEngram/gpu_pilot/run_s1_system_pilot.py': 'ce5952f6d63f3974dd35c13c66edaddbd6cd3ee8ba21e1f9b6a7cb5df1815992', 'GatedEngram/kv_cache_engram/s1_system_runtime.py': '2817532ebbe643892b154afc0c5de2fe8ef1b4d91f5fd1d2bd596417dce00b34', 'GatedEngram/kv_cache_engram/s1_system_cache.py': 'cbcf2959e25db57c811eebd800c06adc6552a4de13ab5c7e81bd00935f4a93d9', 'GatedEngram/tests/test_s1_system_lifecycle.py': '0b8a75333bd22e604590b8c579b86a4500bb4256ee5d004f7832ffe27d18a8b7', 'GatedEngram/tests/test_s1_system_cache.py': 'ff57599b38b4af11303cc5a3b7527dc110c7c3fb6e5fae452a26541e8c34b687', 'GatedEngram/tests/test_s1_system_runtime.py': 'b043656936f085501a5c507ce191e22daf18331cd6670f3797bc92f8fab84432', 'GatedEngram/tests/test_s1_system_batch.py': 'bd3d0fa242461c80b13389478d40d08abed65b705cc9ec81dac4eb7545872cc4', 'GatedEngram/validate_s1_system_cpu.py': '4f87c29bfddeab9a75cba33d819f03a068216de04954f8aedead4eebef378be4', 'GatedEngram/kv_cache_engram/s1_system_attention.py': '1b4260e7e986faaae090193260700dab0f4daa43ac22a4e2bbd1ff586fbd8ef9', 'GatedEngram/tests/test_s1_system_attention.py': 'cfcf9882c22de7a00b77c30a62a9a25a20df5bc7fded01e590c3bd79360ad76d', 'GatedEngram/gpu_pilot/read_s1_system256_results.py': 'e8dd17b0e46e0770824857a2db9442da176d172b50ef7a87a46bd7d247e507cc', 'GatedEngram/tests/test_s1_system256_lifecycle.py': 'bd484a5044e24474da1df93043b334646d09d3e7c5b2145d6d90a2d75479b4bc', 'GatedEngram/tests/test_s1_system256_results.py': 'c4040ea7637a7542e2c321265b0572d9b5afdec384711001bb691011420269cd', 'GatedEngram/validate_s1_system256_cpu.py': '846291f4636c73459b3d86d29771a67e0e5657b896a75ca397588715791f7f63', 'GatedEngram/gpu_pilot/run_s1_system256_pilot.py': 'd6dd2232f46856296eaaa24307f20fc3ab1404468cc9c2e30314ec32ec98cef3', 'GatedEngram/gpu_pilot/s1_system256_inputs.py': '209befcbc8fe00ef872d26790b1ef68ccf7b73bba30bbf30c8988ee69bf620c7', 'GatedEngram/tests/test_s1_system256_protocol.py': '57567dc927a0c68bc82edb1e5a8f898a866a116c75fa6c8e0a5ee2c705cac8b8', 'GatedEngram/tests/test_s1_system256_inputs.py': '32a35d585182780577155cbacc696dc2065ecadf36994194a6adc6dacd8519b8'}
REQUIRED_NEW_DEPENDENCIES = (*original.REQUIRED_NEW_DEPENDENCIES,
    "GatedEngram/kv_cache_engram/s1_system_cache.py",
    "GatedEngram/kv_cache_engram/s1_system_attention.py",
    "GatedEngram/kv_cache_engram/s1_system_runtime.py",
    "GatedEngram/kv_cache_engram/s1_system_batch.py",
    "GatedEngram/gpu_pilot/s1_system_lifecycle.py",
    "GatedEngram/gpu_pilot/run_s1_system256_pilot.py",
    "GatedEngram/gpu_pilot/read_s1_system256_results.py",
    "GatedEngram/gpu_pilot/s1_system256_inputs.py",
    "GatedEngram/validate_s1_system256_cpu.py",
    "GatedEngram/tests/test_s1_system256_inputs.py",
    "GatedEngram/tests/test_s1_system256_protocol.py",
    "GatedEngram/tests/test_s1_system256_results.py")


def controller_arm(arm):
    require(arm in (*ARMS, "qualification", "offline"), "unregistered system experiment arm")
    return "s1" if arm == "s1_system" else arm


def phase_policy(arm):
    return original.phase_policy(controller_arm(arm))


def phase_execution(arm, *, batch_group_chunks=None):
    controller_arm(arm)
    enabled = arm == "s1_system"
    group = SYSTEM_BATCH_GROUP_CHUNKS if enabled else 1
    if batch_group_chunks is not None:
        require(type(batch_group_chunks) is int and batch_group_chunks in SYSTEM_QUALIFICATION_GROUPS,
                "unregistered system batch group")
        require(enabled or batch_group_chunks == 1, "baseline cannot enable chunk grouping")
        group = batch_group_chunks
    return dict(controller_arm=controller_arm(arm), system_enabled=enabled, batch_group_chunks=group)


def timing_schedule(index):
    require(type(index) is int and index in EVALUATION_INDICES, "unregistered system evaluation case")
    ordinal = EVALUATION_INDICES.index(index)
    return [(repeat, ARMS[(ordinal+repeat+slot) % len(ARMS)])
            for repeat in range(3) for slot in range(len(ARMS))]


def s1_system256_protocol():
    previous = original.s1_protocol()
    baseline_policy = {key:deepcopy(previous[key]) for key in CORE_POLICY_KEYS}
    require(digest(baseline_policy) == CORE_POLICY_SHA256, "original S1 policy changed")
    body = {key:deepcopy(value) for key,value in previous.items() if key != "sha256"}
    body.update(schema="s1-system256-pilot-protocol-v1", arms=list(ARMS),
        cases=list(EVALUATION_INDICES), length_bands=list(LENGTH_BANDS),
        length_band_order=["256k"]*len(EVALUATION_INDICES), cases_per_length_band=3,
        target_input_tokens=TARGET_INPUT_TOKENS,
        input_token_ranges={key:list(value) for key,value in INPUT_TOKEN_RANGES.items()},
        input_manifest_sha256=INPUT_MANIFEST_SHA256,source_case_indices=[1039,1036,1035],
        selection_from_original_manifest="three sealed natural 256K QA rows; no content or gold change",
        source_probe_fully_validated=True, baseline_core_policy=baseline_policy,
        baseline_core_policy_sha256=CORE_POLICY_SHA256,
        fresh_all_cases_both_arms=False, fresh_all_cases_three_arms=True,
        phase_policy={arm:phase_policy(arm) for arm in (*ARMS, "qualification", "offline")},
        system_optimization=dict(enabled_arm="s1_system",batch_group_chunks=SYSTEM_BATCH_GROUP_CHUNKS,
            index_geometry_reuse=True, global_raw_affected_members_only=True,
            source20_direct_base_overlay=True,
            batch_must_pass_same_policy_byteexact_before_formal=True,
            world_scheduler_phase_binding_required=True,
            native_idle_steps_passthrough_without_batch_progress=True,
            scheduler_observation_gather_within_online_ttft=True,
            silent_batch_fallback_allowed=False, original_six_core_files_unchanged=True,
            disabled_arms=["recomputed", "s1", "qualification", "offline"]),
        system_qualification=dict(reference_arm="s1",reference_mode="reference_sparse",
            cuda_metadata_operator_gate=dict(required=True,ranks=8,
                artifact="s1_system_metadata_qualification.json",
                sync_debug_mode_during=2,original_sync_debug_mode_restored=True,
                inactive_empty_cu=True,active_queue_group_chunks=[2,4],
                input_tokens=32769,operator_metadata_only=True,gpu_numerical_parity_claim=False),
            reference_system_enabled=False,reference_batch_group_chunks=1,
            original_compact_required=True, optimized_group1_required=True,
            optimized_target_group_required=True, empty_chunk_and_longest_case_required=True,
            all_40_layers_all_four_mhc_fields=True, native_chunk_fingerprints_required=True,
            output_token_identity_required=True, tolerance=0.0, candidate_groups=list(SYSTEM_QUALIFICATION_GROUPS),
            empty_group_probe=dict(input_tokens=32769,batch_group_chunks=2,
                source="exact contiguous prefix of original case2",quality_eligible=False,timing_valid=False,
                actual_rag_quality_claim=False,group4_empty_execution_qualified=False),
            formal_group4_requires_no_completely_empty_group=True),
        restore_original_holder_after_exit=False, holder_after_exit="off per explicit user instruction",
        supervisor_timeout_seconds=86400,
        timeout_origin="new 256K GPU release; completion-driven with emergency ceiling",
        round_absolute_deadline_unix=None,
        timeout_purpose="finish all three cases; 24-hour fault ceiling, holder remains off",
        cache_storage_policy="same_s1_offline_gpu_bank_resident_both_arms")
    body["timing"]["order"] = "rotate case ordinal and repeat; all three arms once per repeat"
    body["timing"]["fresh_three_arm_baseline"] = True
    body["timing"]["system_gain_baseline"] = "fresh original s1 with identical input, bank and generation policy"
    return dict(body, sha256=digest(body))


# Keep the old CLI attribute spelling while dispatch uses the independent file.
s1_protocol = s1_system256_protocol
