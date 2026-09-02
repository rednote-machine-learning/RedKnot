#!/usr/bin/env python3
"""CPU-only fail-closed checks for the isolated Pro-0813 RedKnot path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
    DeepSeekV4MLAHeadConfig,
)
from sglang.srt.layers.attention.redknot.dsv4_fused_z_merge import (
    PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN,
)
from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
    MLA_OFF_REQUIRED_LAYER_IDS,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_latent_cache import (
    DEFAULT_REQUIRED_LAYER_IDS,
    build_dsv4_pro0813_shared_latent_spec,
)
from sglang.srt.layers.attention.redknot.dsv4_shared_snapshot_runtime import (
    REUSABLE_LAYER_IDS,
)
from sglang.srt.layers.attention.redknot.pro0813.profile import (
    inspect_pro0813_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/workspace/Models/DeepSeek-V4-Pro-0813/config.json",
    )
    parser.add_argument("--flash-config")
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    geometry = inspect_pro0813_config(raw, tp_size=8)
    expected_reusable = tuple(range(3, 58))
    assert geometry.reusable_layer_ids == expected_reusable
    assert geometry.dense_layer_ids == (0, 1, 2, 58, 59, 60)
    assert geometry.heads_per_tp_rank == 16
    assert geometry.o_groups_per_tp_rank == 2
    assert geometry.q_lora_rank == 1536
    assert geometry.index_topk == 1024
    assert geometry.n_routed_experts == 384
    assert geometry.dspark_markov_rank == 512
    assert geometry.target_compress_ratios.count(4) == 30
    assert geometry.target_compress_ratios.count(128) == 31

    head_config = DeepSeekV4MLAHeadConfig.from_model_config(
        SimpleNamespace(**raw),
        dense_prefix_layers=3,
        dense_suffix_layers=3,
        local_window=128,
        global_head_stride=8,
    )
    assert head_config.variant == geometry.variant
    assert head_config.geometry_digest == geometry.geometry_digest
    assert head_config.dense_layer_ids == geometry.dense_layer_ids
    assert head_config.offline_online_layer_ids == expected_reusable
    assert head_config.summary() == {
        "dense": 6 * 128,
        "global": 55 * 16,
        "local": 55 * 112,
        "total": 61 * 128,
        "physical_kv_heads": 1,
    }

    spec = build_dsv4_pro0813_shared_latent_spec(
        model_hash="model",
        policy_hash="policy",
        length=512,
        c4_indexer_record_bytes=128,
        c4_attention_terminal_state_bytes=256,
        c4_indexer_terminal_state_bytes=256,
        c128_attention_terminal_state_bytes=256,
    )
    assert spec.required_layer_ids == expected_reusable
    assert len(spec.layers) == 55
    assert sum(layer.compress_ratio == 4 for layer in spec.layers) == 27
    assert sum(layer.compress_ratio == 128 for layer in spec.layers) == 28
    assert DEFAULT_REQUIRED_LAYER_IDS == expected_reusable
    assert REUSABLE_LAYER_IDS == expected_reusable
    assert MLA_OFF_REQUIRED_LAYER_IDS == expected_reusable
    assert PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN == (
        "dsv4_pro0813_headsplit_woa_groups2_blockdiag_persistent_merge:v3"
    )

    bytes_64k = 55 * 65_536 * 2 * 1_024 * 2
    assert bytes_64k == 14_763_950_080

    if args.flash_config:
        flash_raw = json.loads(Path(args.flash_config).read_text(encoding="utf-8"))
        try:
            inspect_pro0813_config(flash_raw, tp_size=8)
        except ValueError:
            pass
        else:
            raise AssertionError("Flash-0731 config entered the Pro-0813 path")

    print(
        json.dumps(
            {
                "status": "pass",
                "variant": geometry.variant,
                "geometry_digest": geometry.geometry_digest,
                "official_config_sha256": geometry.official_config_sha256,
                "reusable_layers": [
                    geometry.first_reusable_layer,
                    geometry.last_reusable_layer,
                    len(geometry.reusable_layer_ids),
                ],
                "heads_per_tp_rank": geometry.heads_per_tp_rank,
                "o_groups_per_tp_rank": geometry.o_groups_per_tp_rank,
                "zoff_64k_bytes_per_rank": bytes_64k,
                "kernel_token": PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
