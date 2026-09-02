from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
PROFILE_PATH = (
    ROOT
    / "python/sglang/srt/layers/attention/redknot/pro0813/profile.py"
)
SPEC = importlib.util.spec_from_file_location("redknot_pro0813_profile", PROFILE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


def _official_config() -> dict:
    path = Path(__file__).with_name("deepseek_v4_pro0813_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_official_pro_geometry_isolated_contract() -> None:
    geometry = PROFILE.inspect_pro0813_config(_official_config(), tp_size=8)

    assert geometry.variant == "deepseek_v4_pro_0813"
    assert geometry.official_config_sha256 == (
        "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
    )
    assert geometry.num_target_layers == 61
    assert geometry.num_attention_heads == 128
    assert geometry.heads_per_tp_rank == 16
    assert geometry.o_groups_per_tp_rank == 2
    assert geometry.q_lora_rank == 1536
    assert geometry.o_lora_rank == 1024
    assert geometry.index_topk == 1024
    assert geometry.n_routed_experts == 384
    assert geometry.dspark_markov_rank == 512
    assert geometry.dense_layer_ids == (0, 1, 2, 58, 59, 60)
    assert geometry.reusable_layer_ids == tuple(range(3, 58))
    assert geometry.first_reusable_layer == 3
    assert geometry.last_reusable_layer == 57
    assert geometry.dspark_target_layer_ids == (58, 59, 60)
    assert geometry.geometry_digest.startswith("sha256:")
    assert len(geometry.geometry_digest) == 71


def test_flash_config_is_rejected_in_pro_path() -> None:
    config = _official_config()
    config.update(
        num_hidden_layers=43,
        num_attention_heads=64,
        hidden_size=4096,
        q_lora_rank=1024,
        o_groups=8,
        index_topk=512,
        n_routed_experts=256,
        dspark_target_layer_ids=[40, 41, 42],
    )
    with pytest.raises(ValueError, match="does not match DeepSeek-V4-Pro-0813"):
        PROFILE.inspect_pro0813_config(config, tp_size=8)


@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 5, 6, 7, 16])
def test_invalid_tensor_parallel_shapes_fail_closed(tp_size: int) -> None:
    with pytest.raises(ValueError, match="requires TP=8"):
        PROFILE.inspect_pro0813_config(_official_config(), tp_size=tp_size)


def test_boundary_contract_cannot_be_weakened() -> None:
    with pytest.raises(ValueError, match="dense prefix/suffix"):
        PROFILE.inspect_pro0813_config(
            _official_config(),
            tp_size=8,
            dense_prefix_layers=2,
            dense_suffix_layers=3,
        )


def test_compression_layer_order_is_part_of_the_fingerprint() -> None:
    config = _official_config()
    config["compress_ratios"][2], config["compress_ratios"][3] = (
        config["compress_ratios"][3],
        config["compress_ratios"][2],
    )
    with pytest.raises(ValueError, match="layer order"):
        PROFILE.inspect_pro0813_config(config, tp_size=8)


def test_dspark_rank_is_part_of_the_fingerprint() -> None:
    config = _official_config()
    config["dspark_markov_rank"] = 256
    with pytest.raises(ValueError, match="does not match"):
        PROFILE.inspect_pro0813_config(config, tp_size=8)
