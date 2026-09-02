from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
REDKNOT = ROOT / "python/sglang/srt/layers/attention/redknot"
if str(REDKNOT) not in sys.path:
    sys.path.insert(0, str(REDKNOT))

import mla_head_drift_analyze as analyze  # noqa: E402
import mla_head_drift_profiler as profiler  # noqa: E402
import mla_head_drift_runtime as runtime  # noqa: E402
from pro0813 import profile  # noqa: E402


def _metrics(head: int) -> profiler.HeadDriftMetrics:
    # Stable ordering deliberately puts all group-0 heads before group-1 heads.
    risk = float(head % 16) * 1.0e-6
    return profiler.HeadDriftMetrics(
        rms=risk,
        relative_rms=risk,
        row_p95=risk,
        row_p99=risk,
        row_max=risk,
        cosine=1.0,
        count_rows=1,
        count_values=1,
        worst_pair_relative_rms=risk,
        worst_pair_row_p99=risk,
        worst_pair_row_max=risk,
        worst_pair_cosine=1.0,
    )


def _report() -> profiler.MLAHeadDriftReport:
    layers = profile.PRO0813_TP8_GEOMETRY.num_target_layers
    heads = profile.PRO0813_TP8_GEOMETRY.num_attention_heads
    metrics = tuple(_metrics(head) for head in range(heads))
    pairs = tuple(
        profiler.DriftPairMetrics(
            layer_id=layer,
            segment_id="seg0",
            oracle_id="oracle0",
            row_count=1,
            row_ids_sha256="0" * 64,
            head_metrics=metrics,
        )
        for layer in range(layers)
    )
    pair_gram = torch.zeros((layers, 8, 16, 16), dtype=torch.float64)
    pair_reference = torch.ones((layers, 8), dtype=torch.float64)
    return profiler.MLAHeadDriftReport(
        variant=profile.PRO0813_VARIANT,
        geometry_digest=profile.PRO0813_TP8_GEOMETRY_DIGEST,
        official_config_sha256=profile.PRO0813_OFFICIAL_CONFIG_SHA256,
        num_layers=layers,
        num_heads=heads,
        num_output_groups=16,
        tp_size=8,
        tp_world_size=8,
        represented_tp_ranks=tuple(range(8)),
        calibration_id="cpu-test",
        dense_prefix_layers=3,
        dense_suffix_layers=3,
        expected_segments=("seg0",),
        expected_oracles=("oracle0",),
        aggregate_metrics=tuple(metrics for _ in range(layers)),
        pair_metrics=pairs,
        rank_gram=pair_gram.clone(),
        rank_reference_energy=pair_reference.clone(),
        pair_rank_gram=pair_gram,
        pair_rank_reference_energy=pair_reference,
    )


def _thresholds() -> profiler.HeadSelectionThresholds:
    return profiler.HeadSelectionThresholds(
        max_head_relative_rms=1.0,
        max_head_row_p99=1.0,
        max_head_row_max=1.0,
        min_head_cosine=0.0,
        max_rank_no_cancel_error=1.0,
    )


def _assert_pro_group_mix(selection: profiler.TPBalancedHeadSelection) -> None:
    for layer, selected in enumerate(selection.local_heads_by_layer):
        if layer < 3 or layer >= 58:
            assert selected == ()
            continue
        for rank in range(8):
            rank_heads = [head for head in selected if head // 16 == rank]
            assert len(rank_heads) == selection.target_local_heads_per_rank
            counts = [
                sum((head % 16) // 8 == group for head in rank_heads)
                for group in range(2)
            ]
            assert all(1 <= count <= 7 for count in counts)


def _load_current_head_config_class():
    """Load only the current consumer and its two small dependencies."""

    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.redknot",
        "sglang.srt.layers.attention.redknot.pro0813",
    )
    dependency_names = (
        "sglang.srt.layers.attention.redknot.head_config",
        "sglang.srt.layers.attention.redknot.pro0813.profile",
    )
    saved = {name: sys.modules.get(name) for name in package_names + dependency_names}
    try:
        for name in package_names:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

        head_config = types.ModuleType(dependency_names[0])
        head_config.DEFAULT_LOCAL_WINDOW = 512
        head_config.DEFAULT_SINK_SIZE = 4
        head_config.HEAD_DENSE = "dense"
        head_config.HEAD_GLOBAL = "global"
        head_config.HEAD_LOCAL = "local"
        sys.modules[dependency_names[0]] = head_config

        pro_profile = types.ModuleType(dependency_names[1])
        pro_profile.PRO0813_OFFICIAL_CONFIG_SHA256 = (
            profile.PRO0813_OFFICIAL_CONFIG_SHA256
        )
        pro_profile.PRO0813_TP8_GEOMETRY = profile.PRO0813_TP8_GEOMETRY
        pro_profile.PRO0813_TP8_GEOMETRY_DIGEST = (
            profile.PRO0813_TP8_GEOMETRY_DIGEST
        )
        pro_profile.PRO0813_VARIANT = profile.PRO0813_VARIANT
        pro_profile.inspect_pro0813_config = profile.inspect_pro0813_config
        sys.modules[dependency_names[1]] = pro_profile

        spec = importlib.util.spec_from_file_location(
            "_pro0813_deepseek_v4_mla_consumer_under_test",
            REDKNOT / "deepseek_v4_mla.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.DeepSeekV4MLAHeadConfig
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_grouped_wo_a_decomposition_is_block_diagonal() -> None:
    output = torch.randn(3, 4, 5, dtype=torch.float64)
    weight = torch.randn(2, 7, 10, dtype=torch.float64)
    contributions = profiler.decompose_wo_a_per_head(output, weight)
    reconstructed = profiler.sum_wo_a_head_contributions(
        contributions, num_output_groups=2
    )
    direct = torch.einsum("tgi,gri->tgr", output.reshape(3, 2, 10), weight)
    torch.testing.assert_close(reconstructed, direct)

    changed = output.clone()
    changed[:, :2].add_(100.0)
    changed_projection = profiler.decompose_wo_a_per_head(changed, weight)
    torch.testing.assert_close(changed_projection[:, 2:], contributions[:, 2:])

    with pytest.raises(ValueError, match="group_input_dim"):
        profiler.decompose_wo_a_per_head(
            output, torch.randn(2, 7, 20, dtype=torch.float64)
        )


@pytest.mark.parametrize(
    "selector",
    (
        profiler.select_tp_balanced_nested,
        profiler.select_tp_balanced_fixed_count_exploratory,
    ),
)
def test_groups2_selector_is_nested_and_fail_closed(selector) -> None:
    report = _report()
    selections = selector(
        report,
        thresholds=_thresholds(),
        target_local_heads_per_rank=(2, 14),
    )
    assert selections[0].eligible_non_dense_heads == 55 * 128
    for selection in selections:
        _assert_pro_group_mix(selection)
    for layer in range(61):
        assert set(selections[0].local_heads_by_layer[layer]).issubset(
            selections[1].local_heads_by_layer[layer]
        )

    for invalid in (1, 15, 16):
        with pytest.raises(ValueError, match="targets must be"):
            selector(
                report,
                thresholds=_thresholds(),
                target_local_heads_per_rank=(invalid,),
            )


def test_v3_builder_round_trips_through_current_pro_loader(tmp_path: Path) -> None:
    selection = profiler.select_tp_balanced_nested(
        _report(),
        thresholds=_thresholds(),
        target_local_heads_per_rank=(2,),
    )[0]
    payload = profiler.build_deepseek_v4_head_config_dict(selection)
    assert payload["format"] == profiler.HEAD_CONFIG_FORMAT
    assert payload["variant"] == profile.PRO0813_VARIANT
    assert payload["geometry_digest"] == profile.PRO0813_TP8_GEOMETRY_DIGEST
    assert (
        payload["official_config_sha256"]
        == profile.PRO0813_OFFICIAL_CONFIG_SHA256
    )
    assert (
        payload["tp_size"],
        payload["heads_per_rank"],
        payload["num_output_groups"],
        payload["groups_per_rank"],
        payload["heads_per_output_group"],
    ) == (8, 16, 16, 2, 8)
    assert payload["dense_layer_ids"] == [0, 1, 2, 58, 59, 60]
    assert payload["offline_online_layer_ids"] == list(range(3, 58))
    assert all(
        value == "dense"
        for layer in (0, 1, 2, 58, 59, 60)
        for value in payload["mla_head_classification"][layer]
    )

    config_path = tmp_path / "candidate.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    loader = _load_current_head_config_class()
    consumer_module = sys.modules[loader.__module__]
    assert consumer_module.PRO0813_MLA_HEAD_CONFIG_FORMAT == (
        profiler.HEAD_CONFIG_FORMAT
    )
    loaded = loader.from_json(str(config_path))
    assert loaded.num_layers == 61
    assert loaded.num_attention_heads == 128
    assert loaded.dense_layer_ids == (0, 1, 2, 58, 59, 60)
    assert loaded.offline_online_layer_ids == tuple(range(3, 58))
    roundtrip = loaded.to_dict()
    assert roundtrip["official_config_sha256"] == (
        profile.PRO0813_OFFICIAL_CONFIG_SHA256
    )
    assert (
        roundtrip["tp_size"],
        roundtrip["heads_per_rank"],
        roundtrip["num_output_groups"],
        roundtrip["groups_per_rank"],
        roundtrip["heads_per_output_group"],
    ) == (8, 16, 16, 2, 8)
    config_path.write_text(json.dumps(roundtrip), encoding="utf-8")
    reloaded = loader.from_json(str(config_path))
    assert reloaded.geometry_digest == profile.PRO0813_TP8_GEOMETRY_DIGEST


def test_pro_v3_consumer_rejects_metadata_and_policy_mutations(
    tmp_path: Path,
) -> None:
    selection = profiler.select_tp_balanced_nested(
        _report(),
        thresholds=_thresholds(),
        target_local_heads_per_rank=(2,),
    )[0]
    payload = profiler.build_deepseek_v4_head_config_dict(selection)
    config_path = tmp_path / "candidate.json"
    loader = _load_current_head_config_class()

    mutations = (
        ("format", "redknot_deepseek_v4_mla_head_config_v2"),
        ("variant", "deepseek_v4_flash_0731"),
        ("geometry_digest", "sha256:" + "0" * 64),
        ("official_config_sha256", "0" * 64),
        ("num_layers", 43),
        ("num_attention_heads", 64),
        ("physical_kv_heads", 2),
        ("tp_size", 4),
        ("tp_size", True),
        ("heads_per_rank", 8),
        ("num_output_groups", 8),
        ("groups_per_rank", 1),
        ("heads_per_output_group", 16),
        ("dense_prefix_layers", 2),
        ("dense_suffix_layers", 2),
        ("dense_layer_ids", [0, 1, 2]),
        ("offline_online_layer_ids", list(range(3, 61))),
    )
    for field, value in mutations:
        bad = copy.deepcopy(payload)
        bad[field] = value
        config_path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises((TypeError, ValueError)):
            loader.from_json(str(config_path))

    bad = copy.deepcopy(payload)
    bad["unexpected"] = True
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid field set"):
        loader.from_json(str(config_path))

    bad = copy.deepcopy(payload)
    del bad["official_config_sha256"]
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid field set"):
        loader.from_json(str(config_path))

    bad = copy.deepcopy(payload)
    bad["profiling_meta"] = []
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="profiling_meta"):
        loader.from_json(str(config_path))

    config_path.write_text('{"format": "one", "format": "two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        loader.from_json(str(config_path))

    bad = copy.deepcopy(payload)
    bad["profiling_meta"] = {"invalid": float("nan")}
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden JSON constant"):
        loader.from_json(str(config_path))

    for override in (2, True):
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="runtime override"):
            loader.from_json(str(config_path), dense_prefix_layers=override)
        with pytest.raises(ValueError, match="runtime override"):
            loader.from_json(str(config_path), dense_suffix_layers=override)

    # The selector picked one local head in each group. Removing one creates a
    # plausible-looking TP8 config whose topology metadata is still correct.
    bad = copy.deepcopy(payload)
    bad["mla_head_classification"][3][8] = "global"
    bad["mla_head_max_distance"][3][8] = -1
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="every W_OA group"):
        loader.from_json(str(config_path))

    bad = copy.deepcopy(payload)
    bad["mla_head_classification"][58][0] = "global"
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="dense fence"):
        loader.from_json(str(config_path))

    local_head = payload["mla_head_classification"][3].index("local")
    bad = copy.deepcopy(payload)
    bad["mla_head_max_distance"][3][local_head] = -1
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="local head distance"):
        loader.from_json(str(config_path))


def test_builder_rejects_one_sided_rank_local_groups() -> None:
    local_by_layer = []
    counts = []
    errors = []
    for layer in range(61):
        if layer < 3 or layer >= 58:
            selected = ()
        else:
            selected = tuple(
                head
                for rank in range(8)
                for head in (rank * 16, rank * 16 + 1)
            )
        local_by_layer.append(selected)
        counts.append(2 if selected else 0)
        errors.append((0.0,) * 8)
    malformed = profiler.TPBalancedHeadSelection(
        num_layers=61,
        num_heads=128,
        num_output_groups=16,
        tp_size=8,
        dense_prefix_layers=3,
        dense_suffix_layers=3,
        target_local_heads_per_rank=2,
        local_heads_by_layer=tuple(local_by_layer),
        local_heads_per_rank_by_layer=tuple(counts),
        conservative_rank_error_by_layer=tuple(errors),
        thresholds=_thresholds(),
    )
    with pytest.raises(ValueError, match="must retain local and global heads"):
        profiler.build_deepseek_v4_head_config_dict(malformed)


def _manifest_payload() -> dict:
    segment_ids = [f"seg{index}" for index in range(4)]
    segments = [
        {
            "id": segment_id,
            "length": 1,
            "sample_rows": [0],
            "token_ids_sha256": analyze.token_ids_sha256([index]),
        }
        for index, segment_id in enumerate(segment_ids)
    ]
    oracles = []
    for rotation in range(4):
        order = segment_ids[rotation:] + segment_ids[:rotation]
        oracles.append(
            {
                "id": f"oracle{rotation}",
                "logical_seq_len": 4,
                "segments": [
                    {"id": segment_id, "slot": slot, "start": slot, "length": 1}
                    for slot, segment_id in enumerate(order)
                ],
            }
        )
    rows_per_rank = 4 + 4 * 4
    estimated = 8 * 61 * rows_per_rank * 16 * 1024 * 4
    return {
        "schema": analyze.CALIBRATION_PAYLOAD_SCHEMA,
        "run_id": "cpu-test",
        "model": {
            "variant": profile.PRO0813_VARIANT,
            "geometry_digest": profile.PRO0813_TP8_GEOMETRY_DIGEST,
            "config_sha256": profile.PRO0813_OFFICIAL_CONFIG_SHA256,
            "num_layers": 61,
            "num_attention_heads": 128,
            "o_lora_rank": 1024,
        },
        "topology": {
            "tp_size": 8,
            "heads_per_rank": 16,
            "num_output_groups": 16,
            "groups_per_rank": 2,
            "heads_per_output_group": 8,
            "dense_prefix_layers": 3,
            "dense_suffix_layers": 3,
        },
        "segments": segments,
        "oracles": oracles,
        "resource_guard": {
            "estimated_capture_tensor_bytes": estimated,
            "max_capture_tensor_bytes": estimated,
        },
    }


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "format": analyze.CALIBRATION_MANIFEST_FORMAT,
                "digest_scope": analyze.CALIBRATION_DIGEST_SCOPE,
                "canonical_payload": payload,
                "calibration_digest": analyze.canonical_manifest_digest(payload),
                "output_dir": str(path.parent.resolve()),
            }
        ),
        encoding="utf-8",
    )


def test_manifest_binds_pro_identity_and_geometry(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = _manifest_payload()
    _write_manifest(path, payload)
    manifest = analyze.load_calibration_manifest(path)
    assert manifest.variant == profile.PRO0813_VARIANT
    assert manifest.geometry_digest == profile.PRO0813_TP8_GEOMETRY_DIGEST
    assert manifest.official_config_sha256 == profile.PRO0813_OFFICIAL_CONFIG_SHA256
    assert manifest.dense_prefix_layers == 3
    assert manifest.dense_suffix_layers == 3

    mutations = (
        ("variant", "flash"),
        ("geometry_digest", "sha256:" + "0" * 64),
        ("config_sha256", "0" * 64),
        ("o_lora_rank", 1),
    )
    for field, value in mutations:
        bad = copy.deepcopy(payload)
        bad["model"][field] = value
        _write_manifest(path, bad)
        with pytest.raises(ValueError):
            analyze.load_calibration_manifest(path)


def test_rank_capture_binds_pro_identity_groups_and_dense_fence(
    monkeypatch,
) -> None:
    rank = 3
    segment = analyze.CaptureSegment(
        segment_id="seg0",
        start=0,
        length=1,
        sample_rows=(0,),
        token_ids_sha256=analyze.token_ids_sha256([7]),
    )
    rows = torch.tensor([0], dtype=torch.int64)
    projection = torch.zeros((1, 16, 1024), dtype=torch.float32)
    raw = {
        "format": analyze.RANK_CAPTURE_FORMAT,
        "variant": profile.PRO0813_VARIANT,
        "geometry_digest": profile.PRO0813_TP8_GEOMETRY_DIGEST,
        "official_config_sha256": profile.PRO0813_OFFICIAL_CONFIG_SHA256,
        "role": "snapshot",
        "run_id": "cpu-test",
        "calibration_id": "cpu-test",
        "calibration_digest": "0" * 64,
        "capture_digest": analyze._capture_digest((segment,)),
        "tp_rank": rank,
        "tp_size": 8,
        "source_tp_rank": rank,
        "source_tp_world_size": 8,
        "represented_tp_ranks": [rank],
        "num_layers": 61,
        "num_attention_heads": 128,
        "local_head_start": rank * 16,
        "local_head_end": (rank + 1) * 16,
        "global_head_ids": list(range(rank * 16, (rank + 1) * 16)),
        "heads_per_rank": 16,
        "num_local_heads": 16,
        "n_local_groups": 2,
        "num_output_groups": 16,
        "local_output_group_ids": [rank * 2, rank * 2 + 1],
        "heads_per_output_group": 8,
        "dense_prefix_layers": 3,
        "dense_suffix_layers": 3,
        "dense_layer_ids": [0, 1, 2, 58, 59, 60],
        "offline_online_layer_ids": list(range(3, 58)),
        "projection_rank": 1024,
        "source_dtype": "torch.bfloat16",
        "logical_seq_len": 1,
        "segments": [
            {
                "id": segment.segment_id,
                "start": segment.start,
                "length": segment.length,
                "sample_rows": list(segment.sample_rows),
                "token_ids_sha256": segment.token_ids_sha256,
            }
        ],
        "layers": {
            layer: {"seg0": {"row_ids": rows, "projection": projection}}
            for layer in range(61)
        },
    }
    current = {"raw": raw}
    monkeypatch.setattr(
        analyze, "_load_weights_only_tree", lambda _path: current["raw"]
    )
    loaded = analyze.load_rank_capture("ignored.pt", expected_role="snapshot")
    assert loaded.tp_rank == rank
    assert loaded.geometry_digest == profile.PRO0813_TP8_GEOMETRY_DIGEST

    mutations = (
        ("variant", "flash"),
        ("geometry_digest", "sha256:" + "0" * 64),
        ("official_config_sha256", "0" * 64),
        ("n_local_groups", 1),
        ("num_output_groups", 8),
        ("local_output_group_ids", [rank]),
        ("dense_suffix_layers", 0),
        ("dense_layer_ids", [0, 1, 2]),
        ("offline_online_layer_ids", list(range(3, 61))),
        ("projection_rank", 1),
    )
    for field, value in mutations:
        bad = dict(raw)
        bad[field] = value
        current["raw"] = bad
        with pytest.raises(ValueError):
            analyze.load_rank_capture("ignored.pt", expected_role="snapshot")


def test_runtime_plan_requires_exact_pro_identity(tmp_path: Path) -> None:
    assert runtime.RANK_CAPTURE_FORMAT == analyze.RANK_CAPTURE_FORMAT
    plan = {
        "mode": runtime.DRIFT_PROFILE_MODE,
        "variant": profile.PRO0813_VARIANT,
        "geometry_digest": profile.PRO0813_TP8_GEOMETRY_DIGEST,
        "official_config_sha256": profile.PRO0813_OFFICIAL_CONFIG_SHA256,
        "run_id": "cpu-test",
        "calibration_digest": "0" * 64,
        "role": "snapshot",
        "out_path": str(tmp_path / "snapshot-rank{tp_rank}.pt"),
        "segments": [{"id": "seg0", "start": 0, "length": 1}],
        "sample_rows": {"seg0": [0]},
    }
    kwargs = {
        "positions": torch.tensor([0]),
        "input_ids": torch.tensor([7]),
        "logical_seq_len": 1,
        "batch_size": 1,
        "is_extend": True,
        "tp_rank": 0,
        "tp_size": 8,
        "pp_size": 1,
        "num_layers": 61,
        "fp8_wo_a": False,
        "attention_backend": "dsv4",
        "sparse_ffn": False,
    }
    parsed = runtime.parse_drift_profile_plan(plan, **kwargs)
    assert parsed.geometry_digest == profile.PRO0813_TP8_GEOMETRY_DIGEST

    for field, value in (
        ("variant", "flash"),
        ("geometry_digest", "sha256:" + "0" * 64),
        ("official_config_sha256", "0" * 64),
    ):
        bad = dict(plan)
        bad[field] = value
        with pytest.raises(ValueError, match=field):
            runtime.parse_drift_profile_plan(bad, **kwargs)
