#!/usr/bin/env python3
"""CPU-only sparse-Q geometry and production-call-chain checks for Pro-0813."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
REDKNOT = ROOT / "python/sglang/srt/layers/attention/redknot"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _expect_value_error(call, fragment: str) -> None:
    try:
        call()
    except ValueError as error:
        assert fragment in str(error), (fragment, str(error))
    else:
        raise AssertionError(f"expected ValueError containing {fragment!r}")


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _verify_production_calls() -> int:
    required_keywords = {
        "tp_rank",
        "tp_size",
        "total_attention_heads",
        "local_attention_head_axes",
        "total_rows",
        "online_rows",
        "layer_id",
        "head_dim",
        "q_lora_rank",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
    }
    paths = (
        ROOT / "python/sglang/srt/models/deepseek_v4.py",
        ROOT / "python/sglang/srt/layers/attention/redknot_mla_backend.py",
    )
    pro_calls = []
    generic_calls = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "build_pro0813_sparse_q_plan":
                pro_calls.append((path, node))
            elif name == "build_sparse_q_plan":
                generic_calls.append((path, node))
    assert len(pro_calls) == 3, [str(path) for path, _ in pro_calls]
    assert not generic_calls, [str(path) for path, _ in generic_calls]
    for path, node in pro_calls:
        assert not node.args, f"{path}:{node.lineno} uses positional Pro geometry"
        keywords = {item.arg for item in node.keywords}
        assert keywords == required_keywords, (
            f"{path}:{node.lineno}",
            sorted(required_keywords - keywords),
            sorted(keywords - required_keywords),
        )
    return len(pro_calls)


def main() -> None:
    profile = _load(
        "redknot_pro0813_profile_sparse_q_test",
        REDKNOT / "pro0813/profile.py",
    )
    sparse_q = _load(
        "redknot_pro0813_sparse_q_test",
        REDKNOT / "dsv4_sparse_q.py",
    )
    runtime = _load(
        "redknot_pro0813_sparse_q_runtime_test",
        REDKNOT / "dsv4_sparse_q_runtime.py",
    )

    geometry = profile.PRO0813_TP8_GEOMETRY
    contract = sparse_q.PRO0813_SPARSE_Q_CONTRACT
    assert sparse_q.SPARSE_Q_PLAN_FORMAT_VERSION == 2
    assert sparse_q.DEFAULT_ATTENTION_HEADS == 128
    assert sparse_q.DEFAULT_LOGICAL_HEADS == 128
    assert sparse_q.DEFAULT_INDEX_HEADS == 64
    generic_signature = inspect.signature(sparse_q.build_sparse_q_plan)
    assert (
        generic_signature.parameters["geometry_contract"].default
        is inspect.Parameter.empty
    )
    assert contract.variant == geometry.variant
    assert contract.model_geometry_digest == geometry.geometry_digest
    assert contract.official_config_sha256 == geometry.official_config_sha256
    assert contract.num_layers == geometry.num_target_layers == 61
    assert contract.reusable_layer_ids == geometry.reusable_layer_ids
    assert contract.num_attention_heads == geometry.num_attention_heads == 128
    assert contract.tp_size == geometry.tp_size == 8
    assert contract.heads_per_tp_rank == geometry.heads_per_tp_rank == 16
    assert contract.head_dim == geometry.head_dim == 512
    assert contract.q_lora_rank == geometry.q_lora_rank == 1536
    # These are indexer heads, not attention heads.
    assert contract.index_n_heads == geometry.index_n_heads == 64
    assert contract.index_head_dim == geometry.index_head_dim == 128
    assert contract.index_topk == geometry.index_topk == 1024
    assert contract.index_n_heads != contract.num_attention_heads

    pro_plan = sparse_q.build_pro0813_sparse_q_plan(
        tp_rank=7,
        tp_size=8,
        total_attention_heads=128,
        local_attention_head_axes=(0, 15),
        total_rows=64,
        online_rows=(62, 63),
        layer_id=57,
        head_dim=512,
        q_lora_rank=1536,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
    )
    pro_plan.validate()
    assert pro_plan.owned_logical_heads == tuple(range(112, 128))
    assert pro_plan.owned_head_count == 16
    assert pro_plan.output_shape == (64, 16, 512)
    payload = pro_plan.as_payload()
    assert payload["total_attention_heads"] == 128
    assert "total_logical_heads" not in payload
    assert payload["geometry_contract"]["q_lora_rank"] == 1536
    assert payload["geometry_contract"]["index_n_heads"] == 64
    assert payload["geometry_contract"]["num_attention_heads"] == 128
    assert payload["geometry_contract_digest"] == contract.digest

    common = dict(
        tp_rank=0,
        tp_size=8,
        total_attention_heads=128,
        local_attention_head_axes=(0, 15),
        total_rows=8,
        online_rows=(7,),
        layer_id=3,
        head_dim=512,
        q_lora_rank=1536,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
    )
    for field, wrong in (
        ("total_attention_heads", 64),
        ("tp_size", 4),
        ("head_dim", 128),
        ("q_lora_rank", 1024),
        ("index_n_heads", 128),
        ("index_head_dim", 64),
        ("index_topk", 512),
    ):
        bad = dict(common)
        bad[field] = wrong
        _expect_value_error(
            lambda bad=bad: sparse_q.build_pro0813_sparse_q_plan(**bad),
            "does not match deepseek_v4_pro_0813",
        )
    dense = dict(common)
    dense["layer_id"] = 58
    _expect_value_error(
        lambda: sparse_q.build_pro0813_sparse_q_plan(**dense),
        "outside the contract's reusable layer domain",
    )

    # The lower-level builder remains generic, but every caller must supply an
    # explicit contract; its entire attention/indexer geometry enters digest.
    synthetic = sparse_q.SparseQGeometryContract(
        variant="synthetic_dsv4",
        model_geometry_digest="sha256:" + "1" * 64,
        official_config_sha256="2" * 64,
        num_layers=4,
        reusable_layer_ids=(1, 2),
        num_attention_heads=32,
        tp_size=4,
        heads_per_tp_rank=8,
        head_dim=256,
        q_lora_rank=768,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=128,
    )
    generic_plan = sparse_q.build_sparse_q_plan(
        1,
        4,
        32,
        (0, 7),
        16,
        (15,),
        geometry_contract=synthetic,
        layer_id=1,
        head_dim=256,
    )
    generic_plan.validate()
    assert generic_plan.owned_logical_heads == tuple(range(8, 16))
    changed_indexer = replace(synthetic, index_topk=256)
    changed_plan = sparse_q.build_sparse_q_plan(
        1,
        4,
        32,
        (0, 7),
        16,
        (15,),
        geometry_contract=changed_indexer,
        layer_id=2,
        head_dim=256,
    )
    assert generic_plan.digest != changed_plan.digest
    _expect_value_error(
        lambda: runtime.SequentialPackedQArena(
            plans=(generic_plan, changed_plan),
            backing=None,
            arena_token="mixed-geometry-must-fail-before-backing",
        ),
        "mix model geometry contracts",
    )

    production_call_count = _verify_production_calls()
    print(
        json.dumps(
            {
                "status": "pass",
                "variant": contract.variant,
                "model_geometry_digest": contract.model_geometry_digest,
                "sparse_q_contract_digest": contract.digest,
                "plan_digest": pro_plan.digest,
                "attention_heads": contract.num_attention_heads,
                "heads_per_tp_rank": contract.heads_per_tp_rank,
                "q_lora_rank": contract.q_lora_rank,
                "index_geometry": [
                    contract.index_n_heads,
                    contract.index_head_dim,
                    contract.index_topk,
                ],
                "production_call_count": production_call_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
