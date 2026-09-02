"""Unit tests for DeepSeek V4 RedKnot topology parsing."""

import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

import sglang.srt.layers.attention.redknot_mla_backend as redknot_mla_backend
import sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime as dsv4_reuse_backend_runtime
import sglang.srt.layers.attention.redknot.dsv4_shared_latent_sglang as dsv4_shared_latent_sglang
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    _redknot_mla_off_preserves_online_kv,
)
from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
    DeepSeekV4MLAHeadConfig,
    deepseek_v4_mla_cache_descriptor,
    deepseek_v4_redknot_topology,
)
from sglang.srt.layers.attention.redknot.head_config import (
    HEAD_DENSE,
    HEAD_GLOBAL,
    HEAD_LOCAL,
)
from sglang.srt.layers.attention.redknot.mla_head_profiler import (
    MLAHeadLocalityCollector,
    MLAHeadProfileConfig,
)
from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
    DSV4MLAOffController,
    MLA_OFF_EXECUTION_PROFILE,
    MLAOffLayerSpec,
    MLAOffRestoreRows,
    MLAOffRuntimeContext,
    build_restore_rows,
)
from sglang.srt.layers.attention.redknot.dsv4_context_identity import (
    NATIVE_FULL_SCOPE_POLICY,
    context_segment_sha256,
    token_ids_sha256,
)
from sglang.srt.layers.attention.redknot.dsv4_reuse_backend_runtime import (
    CompositeForwardResources,
    ForwardCompositeCommitCoordinator,
    LayerCompositeCommitBuilder,
    _active_restore_plan_fallback_reason,
)
from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
    ForwardCommitCertificate,
    ForwardCommitSession,
    OmissionAuthorization,
)
from sglang.srt.layers.attention.redknot_mla_backend import (
    _build_dual_layer_pass_plan,
    _validate_pure_headsplit_plan_contract,
    _validate_forced_local_window,
)
from sglang.srt.layers.attention.redknot.v4.config import RedKnotV4Config
from sglang.srt.layers.attention.redknot.v4.reuse_planner import (
    MLA_OFF_HEAD_SCOPE_POLICY,
    MLA_OFF_INDEPENDENT_RELOCATION_PROFILE,
    validate_mla_off_plan,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.srt.speculative.dspark_utils import parse_dspark_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_CONTEXT_MODEL_HASH = "a" * 64
_CONTEXT_POLICY_HASH = "b" * 64


class TestMLAOffForwardSystemCaches(CustomTestCase):
    @staticmethod
    def _independent_restore_certificate_plan():
        first = (11, 12, 13, 14)
        second = (21, 22, 23, 24)
        return first, second, {
            "mode": "restore",
            "mla_off_execution_profile": (
                "pure_headsplit_independent_rope_relocation_fullscope_"
                "boundary128_3_37_3_v1"
            ),
            "query_start": 8,
            "total_tokens": 10,
            "radix_prefix_role": "consume",
            "radix_prefix_tokens": 4,
            "radix_prefix_input_hash": token_ids_sha256(first),
            "segments": [
                {
                    "global_offset": 0,
                    "length": 4,
                    "token_hash": token_ids_sha256(first),
                },
                {
                    "global_offset": 4,
                    "length": 4,
                    "token_hash": token_ids_sha256(second),
                },
            ],
        }

    def test_independent_restore_certificate_accepts_exact_radix_document(self):
        first, second, plan = self._independent_restore_certificate_plan()
        phase = dsv4_reuse_backend_runtime._validate_independent_restore_chunk(
            plan=plan,
            request_positions=(4, 5, 6, 7),
            request_tokens=second,
            trusted_cached_prefix_tokens=first,
        )
        self.assertEqual(phase, "segment")

    def test_independent_restore_certificate_rejects_token_or_prefix_tamper(self):
        first, second, plan = self._independent_restore_certificate_plan()
        with self.assertRaisesRegex(ValueError, "document token hash"):
            dsv4_reuse_backend_runtime._validate_independent_restore_chunk(
                plan=plan,
                request_positions=(4, 5, 6, 7),
                request_tokens=second[:-1] + (99,),
                trusted_cached_prefix_tokens=first,
            )
        with self.assertRaisesRegex(ValueError, "cached first document"):
            dsv4_reuse_backend_runtime._validate_independent_restore_chunk(
                plan=plan,
                request_positions=(4, 5, 6, 7),
                request_tokens=second,
                trusted_cached_prefix_tokens=first[:-1] + (99,),
            )

    def test_independent_restore_certificate_marks_online_suffix(self):
        first, _, plan = self._independent_restore_certificate_plan()
        phase = dsv4_reuse_backend_runtime._validate_independent_restore_chunk(
            plan=plan,
            request_positions=(8, 9),
            request_tokens=(31, 32),
            trusted_cached_prefix_tokens=first,
        )
        self.assertEqual(phase, "suffix_complete")

    def test_independent_restore_certificate_rejects_partial_or_cross_boundary(self):
        first, second, plan = self._independent_restore_certificate_plan()
        with self.assertRaisesRegex(ValueError, "partial document"):
            dsv4_reuse_backend_runtime._validate_independent_restore_chunk(
                plan=plan,
                request_positions=(4, 5),
                request_tokens=second[:2],
                trusted_cached_prefix_tokens=first,
            )
        with self.assertRaisesRegex(ValueError, "crosses the query boundary"):
            dsv4_reuse_backend_runtime._validate_independent_restore_chunk(
                plan=plan,
                request_positions=(7, 8),
                request_tokens=(second[-1], 31),
                trusted_cached_prefix_tokens=first,
            )
        foreign = dict(plan)
        foreign["mla_off_execution_profile"] = "foreign"
        with self.assertRaisesRegex(ValueError, "foreign profile"):
            dsv4_reuse_backend_runtime._validate_independent_restore_chunk(
                plan=foreign,
                request_positions=(4, 5, 6, 7),
                request_tokens=second,
                trusted_cached_prefix_tokens=first,
            )

    def test_independent_restore_certificate_does_not_require_context_registry(self):
        source = inspect.getsource(
            dsv4_reuse_backend_runtime._prepare_composite_restore_context_impl
        )
        certificate_branch = source[
            source.index("cached_prefixes = getattr") : source.index(
                "if context_phase not in"
            )
        ]
        independent_block, context_bound_block = certificate_branch.split(
            "                    else:\n", 1
        )
        self.assertNotIn("_redknot_context_token_streams", independent_block)
        self.assertIn("_redknot_context_token_streams", context_bound_block)

    def test_independent_pure_restore_bypasses_legacy_boundary_replay(self):
        plan = {
            "mode": "restore",
            "reuse_mla_off": True,
            "capture_mla_off": False,
            "skip_forward": False,
            "mla_off_execution_profile": (
                MLA_OFF_INDEPENDENT_RELOCATION_PROFILE
            ),
        }
        forward_batch = SimpleNamespace(
            redknot_reuse_plan=[plan],
            redknot_v4_boundary_replay=None,
            redknot_v4_compressor_schedules=None,
        )

        self.assertTrue(
            ModelRunner._prepare_redknot_v4_boundary_replay(
                SimpleNamespace(), forward_batch
            )
        )
        self.assertIs(forward_batch.redknot_reuse_plan[0], plan)

    def test_plan_tree_freezes_once_and_live_guard_checks_owner_identity(self):
        plans = [
            {
                "mode": "restore",
                "segments": [{"seg_hash": "segment", "length": 8}],
            }
        ]
        positions = torch.arange(8, dtype=torch.long)
        input_ids = torch.arange(8, dtype=torch.long)
        lengths = [8]
        totals = [8]
        extents = [8]
        source = dsv4_reuse_backend_runtime.CompositeGeometrySourceBinding.capture(
            forward_generation=("generation", 1),
            raw_plans=plans,
            positions=positions,
            input_ids=input_ids,
            ragged_lengths=(8,),
            ragged_source=lengths,
            scheduler_totals_source=totals,
            scheduler_extents_source=extents,
            batch_size=1,
            q_rows=8,
        )
        frozen = dsv4_reuse_backend_runtime._freeze_plan_payload(plans[0])
        plans[0]["segments"][0]["length"] = 99
        self.assertEqual(frozen["segments"][0]["length"], 8)

        source.validate_live(
            forward_generation=("generation", 1),
            raw_plans=plans,
            positions=positions,
            input_ids=input_ids,
            ragged_lengths=(8,),
            ragged_source=lengths,
            scheduler_totals_source=totals,
            scheduler_extents_source=extents,
            batch_size=1,
            q_rows=8,
        )
        plans[0] = dict(plans[0])
        with self.assertRaisesRegex(ValueError, "plan object identity changed"):
            source.validate_live(
                forward_generation=("generation", 1),
                raw_plans=plans,
                positions=positions,
                input_ids=input_ids,
                ragged_lengths=(8,),
                ragged_source=lengths,
                scheduler_totals_source=totals,
                scheduler_extents_source=extents,
                batch_size=1,
                q_rows=8,
            )

    def test_restore_rows_cache_one_validated_projection_geometry(self):
        output = torch.arange(16, dtype=torch.long)
        local = torch.arange(32, 48, dtype=torch.long)
        rows = MLAOffRestoreRows(
            seg_hash="segment",
            output_rows_cpu=output,
            local_positions_cpu=local,
        )

        self.assertIs(rows.output_rows, rows.output_rows)
        self.assertIs(rows.local_positions, rows.local_positions)
        self.assertIs(rows.projection_geometry, rows.projection_geometry)
        self.assertEqual(rows.output_rows, tuple(range(16)))
        self.assertEqual(rows.local_positions, tuple(range(32, 48)))
        self.assertEqual(
            rows.projection_geometry.output_rows,
            rows.output_rows,
        )
        self.assertEqual(
            rows.projection_geometry.local_rows,
            rows.local_positions,
        )

    def test_restore_rows_reject_invalid_geometry_before_layer_loop(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            MLAOffRestoreRows(
                seg_hash="segment",
                output_rows_cpu=torch.tensor([0, 0], dtype=torch.long),
                local_positions_cpu=torch.tensor([0, 1], dtype=torch.long),
            )

    def test_token_row_values_validate_once_and_identity_rechecks_every_layer(self):
        geometry = object.__new__(
            dsv4_reuse_backend_runtime.CompositeForwardGeometry
        )
        geometry._token_rows_binding_identities = None
        rows = SimpleNamespace(
            local_positions_cpu=torch.arange(4, dtype=torch.long)
        )
        token_ids = torch.arange(4, dtype=torch.long)
        first_view = SimpleNamespace(layer_id=3)
        second_view = SimpleNamespace(layer_id=4)
        controller = SimpleNamespace(
            validate_view_token_rows=MagicMock(),
            token_rows_binding_identity=MagicMock(
                return_value=("sealed-token-rows",)
            ),
        )

        geometry.validate_or_install_token_rows_bindings(
            controller, ((first_view, rows, token_ids),)
        )
        geometry.validate_or_install_token_rows_bindings(
            controller, ((second_view, rows, token_ids),)
        )

        controller.validate_view_token_rows.assert_called_once_with(
            first_view,
            local_positions=rows.local_positions_cpu,
            token_ids=token_ids,
        )
        self.assertEqual(controller.token_rows_binding_identity.call_count, 2)

        controller.token_rows_binding_identity.return_value = (
            "changed-token-rows",
        )
        with self.assertRaisesRegex(ValueError, "artifact identity changed"):
            geometry.validate_or_install_token_rows_bindings(
                controller, ((second_view, rows, token_ids),)
            )

    def test_shared_latent_forward_template_hashes_schedule_once(self):
        mirror = SimpleNamespace(commit_epoch=1, layout_fingerprint="layout")
        state = SimpleNamespace(
            validate=MagicMock(),
            reusable=True,
            request_index=0,
            flat_row_offset=0,
            row_count=2,
            schedule=SimpleNamespace(
                layout_fingerprint="layout",
                positions=(0, 1),
                query_start=2,
                artifact_epochs={"segment": 1},
                index_arena=(),
                operations=(),
            ),
            cpu_plan=SimpleNamespace(
                spec=SimpleNamespace(
                    model_hash="model",
                    policy_hash="policy",
                    length=2,
                ),
                artifacts={},
            ),
            pin=SimpleNamespace(mirrors={"segment": mirror}),
            dirty_rows=(1,),
        )
        resources = SimpleNamespace(
            batch_digest="sha256:" + "1" * 64,
            shared_states=(state,),
            _shared_latent_template=None,
        )
        plan = SimpleNamespace(local_clean_rows=(0,), local_dirty_rows=(1,))
        context3 = SimpleNamespace(layer_id=3, batched_reuse_plan=plan)
        context4 = SimpleNamespace(layer_id=4, batched_reuse_plan=plan)

        first = dsv4_reuse_backend_runtime._build_shared_latent_binding(
            context3, resources, compression_ratio=128
        )
        second = dsv4_reuse_backend_runtime._build_shared_latent_binding(
            context4, resources, compression_ratio=4
        )

        state.validate.assert_called_once_with()
        self.assertEqual(first.spec_digest, second.spec_digest)
        self.assertEqual(first.restore_plan_digest, second.restore_plan_digest)
        self.assertIs(first.artifacts, second.artifacts)
        self.assertEqual(first.layer_ids, (3,))
        self.assertEqual(second.layer_ids, (4,))
        self.assertEqual(first.compression_by_layer, {3: 128})
        self.assertEqual(second.compression_by_layer, {4: 4})

        resources._shared_latent_template = replace(first)
        with self.assertRaisesRegex(ValueError, "template changed"):
            dsv4_reuse_backend_runtime._build_shared_latent_binding(
                context4, resources, compression_ratio=4
            )
        resources._shared_latent_template = first

        context4.batched_reuse_plan = SimpleNamespace(
            local_clean_rows=(0, 1),
            local_dirty_rows=(),
        )
        with self.assertRaisesRegex(ValueError, "row geometry changed"):
            dsv4_reuse_backend_runtime._build_shared_latent_binding(
                context4, resources, compression_ratio=4
            )

    def test_ragged_forward_geometry_constructs_once_and_rejects_replacement(self):
        digest = "sha256:" + "2" * 64
        request = SimpleNamespace(
            request_index=0,
            request_token="request",
            flat_row_start=0,
            row_count=8,
            logical_positions=tuple(range(8)),
            query_start=6,
        )
        batch_plan = SimpleNamespace(
            requests=(request,),
            q_rows=8,
            validate=MagicMock(),
        )
        resources = SimpleNamespace(
            batch_digest=digest,
            request_token_digests=("sha256:" + "3" * 64,),
            forward_id="forward",
            total_rows=8,
            _ragged_geometry_template=None,
            _ragged_geometry_template_identity=None,
        )
        context = SimpleNamespace(batched_reuse_plan=batch_plan)
        with patch.object(
            dsv4_reuse_backend_runtime,
            "_batched_geometry_digest",
            return_value=digest,
        ) as geometry_digest:
            first = dsv4_reuse_backend_runtime._build_ragged_geometry(
                context, resources
            )
            second = dsv4_reuse_backend_runtime._build_ragged_geometry(
                SimpleNamespace(batched_reuse_plan=object()), resources
            )
        self.assertIs(first, second)
        batch_plan.validate.assert_called_once_with()
        geometry_digest.assert_called_once_with(batch_plan)

        resources._ragged_geometry_template = replace(first)
        with self.assertRaisesRegex(ValueError, "template changed"):
            dsv4_reuse_backend_runtime._build_ragged_geometry(
                context, resources
            )


def _context_restore_plan(chunk_tokens, num_chunks, *, suffix=(1, 2, 3)):
    prefix = []
    segments = []
    for index in range(num_chunks):
        tokens = tuple(
            (index * int(chunk_tokens) + offset) % 100_000
            for offset in range(int(chunk_tokens))
        )
        source_start = len(prefix)
        prefix_hash = token_ids_sha256(prefix)
        prefix.extend(tokens)
        full_hash = token_ids_sha256(prefix)
        token_hash = token_ids_sha256(tokens)
        contract = {
            "token_hash": token_hash,
            "prefix_input_hash": prefix_hash,
            "full_input_hash": full_hash,
            "source_start": source_start,
            "source_end": len(prefix),
            "global_offset": source_start,
            "length": len(tokens),
            "canonical_start_pos": 0,
            "skip_first": 0,
        }
        contract["seg_hash"] = context_segment_sha256(
            execution_profile=MLA_OFF_EXECUTION_PROFILE,
            head_scope_policy=NATIVE_FULL_SCOPE_POLICY,
            model_compat_hash=_CONTEXT_MODEL_HASH,
            head_policy_hash=_CONTEXT_POLICY_HASH,
            **{
                key: contract[key]
                for key in (
                    "token_hash",
                    "prefix_input_hash",
                    "full_input_hash",
                    "source_start",
                    "source_end",
                    "length",
                    "canonical_start_pos",
                )
            },
        )
        segments.append(contract)
    full_request = tuple(prefix) + tuple(suffix)
    return {
        "mode": "restore",
        "reuse_mla_off": True,
        "allow_approximate": False,
        "mla_off_execution_profile": MLA_OFF_EXECUTION_PROFILE,
        "mla_off_head_scope_policy": NATIVE_FULL_SCOPE_POLICY,
        "model_compat_hash": _CONTEXT_MODEL_HASH,
        "head_policy_hash": _CONTEXT_POLICY_HASH,
        "query_start": len(prefix),
        "total_tokens": len(full_request),
        "offline_prefix_hash": segments[-1]["full_input_hash"],
        "request_input_hash": token_ids_sha256(full_request),
        "segments": segments,
        "mla_off_qualification_only": True,
    }


def _config(**overrides):
    values = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "hidden_size": 4096,
        "vocab_size": 129280,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "compress_ratios": [0, 0] + [x for _ in range(20) for x in (4, 128)] + [4],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _head_config(head_class, windows, *, layer_id=0):
    layer_id = int(layer_id)
    filler_heads = [HEAD_GLOBAL] * len(head_class)
    filler_windows = [-1] * len(head_class)
    return DeepSeekV4MLAHeadConfig(
        head_class=[list(filler_heads) for _ in range(layer_id)]
        + [head_class],
        head_max_distance=[list(filler_windows) for _ in range(layer_id)]
        + [windows],
        num_layers=layer_id + 1,
        num_attention_heads=len(head_class),
        physical_kv_heads=1,
        dense_prefix_layers=0,
        local_default_window=128,
    )


def _load_redknot_benchmark_module():
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent
            / "test"
            / "srt"
            / "redknot"
            / "utils"
            / "benchmark_RedKnot_DeepSeekV4_Flash_RAG.py"
        )
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "_redknot_dsv4_benchmark_runtime_test", candidate
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("DeepSeek V4 RedKnot benchmark module was not found")


class TestDeepSeekV4RedKnotTopology(CustomTestCase):
    def test_0731_auxiliary_layers_are_not_target_layers(self):
        config = _config(
            compress_ratios=_config().compress_ratios + [0, 0, 0],
            dspark_block_size=5,
            dspark_target_layer_ids=[40, 41, 42],
        )

        topology = deepseek_v4_redknot_topology(config)

        self.assertEqual(topology["num_target_layers"], 43)
        self.assertEqual(topology["num_swa_only_layers"], 2)
        self.assertEqual(topology["num_c4_layers"], 21)
        self.assertEqual(topology["num_c128_layers"], 20)
        self.assertEqual(topology["aux_compress_ratios"], (0, 0, 0))
        self.assertEqual(topology["num_dspark_stages"], 3)

        descriptor = deepseek_v4_mla_cache_descriptor(config)
        self.assertEqual(descriptor["num_layers"], 43)
        self.assertEqual(descriptor["auxiliary_layers"], 3)

    def test_missing_target_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "42 entries for 43 target layers"):
            deepseek_v4_redknot_topology(
                _config(compress_ratios=_config().compress_ratios[:-1])
            )

    def test_transformers_five_layer_types_are_supported(self):
        ratios = _config().compress_ratios
        ratio_to_type = {
            0: "sliding_attention",
            4: "compressed_sparse_attention",
            128: "heavily_compressed_attention",
        }
        config = _config(
            compress_ratios=None,
            layer_types=[ratio_to_type[x] for x in ratios],
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
            dspark_block_size=5,
            dspark_target_layer_ids=[40, 41, 42],
        )

        topology = deepseek_v4_redknot_topology(config)

        self.assertEqual(topology["target_compress_ratios"], tuple(ratios))
        self.assertEqual(topology["aux_compress_ratios"], (0, 0, 0))
        self.assertEqual(topology["num_dspark_stages"], 3)

    def test_invalid_dspark_target_layer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside target model"):
            deepseek_v4_redknot_topology(
                _config(dspark_block_size=5, dspark_target_layer_ids=[43])
            )

    def test_dspark_runtime_contract(self):
        config = _config(
            dspark_block_size=5,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[40, 41, 42],
            dspark_markov_rank=256,
        )

        dspark = parse_dspark_config(config)

        self.assertEqual(dspark.block_size, 5)
        self.assertEqual(dspark.num_verify_tokens, 6)
        self.assertEqual(dspark.num_stages, 3)
        self.assertEqual(dspark.target_hidden_size, 3 * 4096)

    def test_dspark_noise_token_is_validated(self):
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            parse_dspark_config(
                _config(
                    dspark_block_size=5,
                    dspark_noise_token_id=129280,
                    dspark_target_layer_ids=[40, 41, 42],
                    dspark_markov_rank=256,
                )
            )

    def test_dspark_target_layers_must_be_ordered(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_dspark_config(
                _config(
                    dspark_block_size=5,
                    dspark_noise_token_id=128799,
                    dspark_target_layer_ids=[42, 40, 41],
                    dspark_markov_rank=256,
                )
            )


class TestPureMLAHeadSplit3373(CustomTestCase):
    EXPECTED_OFFLINE_LAYERS = tuple(range(3, 40))
    EXPECTED_DENSE_LAYERS = (0, 1, 2, 40, 41, 42)

    def test_execution_profile_is_context_bound_fullscope_v3(self):
        self.assertEqual(
            MLA_OFF_EXECUTION_PROFILE,
            "pure_headsplit_context_bound_fullscope_3_37_3_v1",
        )
        self.assertEqual(
            MLA_OFF_HEAD_SCOPE_POLICY,
            "native_dsv4_full_candidate_scope_v1",
        )

    def test_default_policy_has_exact_3_37_3_layer_fence(self):
        cfg = DeepSeekV4MLAHeadConfig.from_model_config(
            _config(),
            local_window=128,
            global_head_stride=8,
            global_layer_stride=0,
        )

        self.assertEqual(cfg.dense_prefix_layers, 3)
        self.assertEqual(cfg.dense_suffix_layers, 3)
        self.assertEqual(cfg.dense_layer_ids, self.EXPECTED_DENSE_LAYERS)
        self.assertEqual(
            cfg.offline_online_layer_ids, self.EXPECTED_OFFLINE_LAYERS
        )
        self.assertEqual(len(cfg.offline_online_layer_ids), 37)
        for layer in self.EXPECTED_DENSE_LAYERS:
            self.assertTrue(
                all(
                    cfg.get_strategy(layer, head).head_type == HEAD_DENSE
                    for head in range(64)
                )
            )
        for layer in self.EXPECTED_OFFLINE_LAYERS:
            strategies = [cfg.get_strategy(layer, head) for head in range(64)]
            self.assertEqual(sum(item.is_local() for item in strategies), 56)
            self.assertEqual(sum(item.is_global() for item in strategies), 8)

    def test_external_json_cannot_weaken_dense_boundary(self):
        payload = {
            "format": "redknot_deepseek_v4_mla_head_config_v1",
            "num_layers": 43,
            "num_attention_heads": 64,
            "physical_kv_heads": 1,
            "dense_prefix_layers": 0,
            "dense_suffix_layers": 0,
            "local_default_window": 128,
            "mla_head_classification": [[HEAD_LOCAL] * 64 for _ in range(43)],
            "mla_head_max_distance": [[128] * 64 for _ in range(43)],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "heads.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = DeepSeekV4MLAHeadConfig.from_json(
                str(path), dense_prefix_layers=3, dense_suffix_layers=3
            )

        self.assertEqual(cfg.dense_layer_ids, self.EXPECTED_DENSE_LAYERS)
        for layer in self.EXPECTED_DENSE_LAYERS:
            self.assertTrue(
                all(
                    cfg.get_strategy(layer, head).head_type == HEAD_DENSE
                    for head in range(64)
                )
            )
        self.assertTrue(cfg.get_strategy(3, 0).is_local())
        self.assertTrue(cfg.get_strategy(39, 63).is_local())

    def test_pure_context_bound_exact_8x8k_restore_geometry(self):
        chunk_tokens = 8192
        num_chunks = 8
        document_tokens = chunk_tokens * num_chunks
        plan = _context_restore_plan(chunk_tokens, num_chunks)
        _validate_pure_headsplit_plan_contract(plan)
        rows, reusable = build_restore_rows(
            plan=plan,
            positions_cpu=torch.arange(document_tokens + 3, dtype=torch.long),
            refresh_layer=False,
        )

        self.assertEqual(int(reusable[:document_tokens].sum().item()), 65536)
        self.assertEqual(int((~reusable[:document_tokens]).sum().item()), 0)
        self.assertFalse(bool(reusable[document_tokens:].any().item()))
        self.assertEqual(
            torch.nonzero(~reusable[:document_tokens], as_tuple=False)
            .flatten()
            .tolist(),
            [],
        )
        self.assertEqual(sum(item.output_rows_cpu.numel() for item in rows), 65536)
        self.assertEqual(
            tuple(item.output_rows_cpu.numel() for item in rows),
            (8192,) * 8,
        )
        for item in rows:
            self.assertEqual(item.local_positions_cpu[0].item(), 0)

    def test_pure_context_bound_rejects_nonzero_segment_boundaries(self):
        base = _context_restore_plan(256, 2, suffix=())
        for segment_index, skip_first in ((0, 128), (1, 1), (1, 127), (1, 128)):
            plan = {
                **base,
                "segments": [dict(segment) for segment in base["segments"]],
            }
            plan["segments"][segment_index]["skip_first"] = skip_first
            with self.subTest(segment=segment_index, skip_first=skip_first):
                with self.assertRaisesRegex(ValueError, "skip_first"):
                    _validate_pure_headsplit_plan_contract(plan)
                with self.assertRaisesRegex(ValueError, "boundary"):
                    build_restore_rows(
                        plan=plan,
                        positions_cpu=torch.arange(512, dtype=torch.long),
                        refresh_layer=False,
                    )

    def test_planner_accepts_only_exact_context_bound_contract(self):
        base = _context_restore_plan(8192, 2, suffix=())
        config = RedKnotV4Config(mode="aggressive")
        self.assertEqual(config.boundary_replay_tokens, 128)
        accepted = validate_mla_off_plan(base, config=config)
        self.assertTrue(accepted.valid, accepted.detail)

        for segment_index, skip_first in ((0, 128), (1, 1), (1, 127), (1, 128)):
            plan = {
                **base,
                "segments": [dict(segment) for segment in base["segments"]],
            }
            plan["segments"][segment_index]["skip_first"] = skip_first
            with self.subTest(segment=segment_index, skip_first=skip_first):
                rejected = validate_mla_off_plan(plan, config=config)
                self.assertFalse(rejected.valid)
                self.assertIn("restore every context-qualified row", rejected.detail)

    def test_pure_plan_rejects_every_selected_row_overlay(self):
        base = _context_restore_plan(256, 1, suffix=())
        contaminations = {
            "mla_off_use_indexer_hot": True,
            "selection_policy": "checkpoint_islands",
            "hot_frac": 0.5,
            "checkpoint_stride_tokens": 1024,
            "checkpoint_islands": [(0, 128)],
            "mla_off_refresh_layer_stride": 2,
            "mla_off_dirty_ranges": [(0, 1)],
        }
        for field, value in contaminations.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "pure headsplit"
            ):
                _validate_pure_headsplit_plan_contract(
                    {**base, field: value}
                )

    def test_pure_plan_accepts_only_certified_merged_prefill_groups(self):
        config = RedKnotV4Config(mode="aggressive")
        base = _context_restore_plan(8192, 8, suffix=())
        merged = {**base, "merged_prefill_tokens": 32768}
        _validate_pure_headsplit_plan_contract(merged)
        accepted = validate_mla_off_plan(merged, config=config)
        self.assertTrue(accepted.valid, accepted.detail)

        two_segment_group = {**base, "merged_prefill_tokens": 16384}
        _validate_pure_headsplit_plan_contract(two_segment_group)
        accepted_two_segment_group = validate_mla_off_plan(
            two_segment_group, config=config
        )
        self.assertTrue(
            accepted_two_segment_group.valid,
            accepted_two_segment_group.detail,
        )

        rows, reusable = build_restore_rows(
            plan=merged,
            positions_cpu=torch.arange(32768, dtype=torch.long),
            refresh_layer=False,
        )
        self.assertEqual(tuple(item.output_rows_cpu.numel() for item in rows), (8192,) * 4)
        self.assertTrue(bool(reusable.all().item()))

        invalid = {**base, "merged_prefill_tokens": 24576}
        with self.assertRaisesRegex(ValueError, "invalid pure merged-prefill"):
            _validate_pure_headsplit_plan_contract(invalid)
        rejected = validate_mla_off_plan(invalid, config=config)
        self.assertFalse(rejected.valid)
        self.assertIn("invalid pure merged-prefill", rejected.detail)

    def test_restore_bitmap_rejects_dirty_indexer_and_hot_overlays(self):
        base = _context_restore_plan(256, 1, suffix=())
        contaminations = {
            "mla_off_use_indexer_hot": True,
            "selection_policy": "checkpoint_islands",
            "hot_frac": 0.5,
            "mla_off_hot_expand_tokens": 128,
            "mla_off_refresh_layer_stride": 1,
            "mla_off_dirty_ranges": [(0, 1)],
        }
        for field, value in contaminations.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "pure headsplit"
            ):
                build_restore_rows(
                    plan={**base, field: value},
                    positions_cpu=torch.arange(256, dtype=torch.long),
                    refresh_layer=False,
                )


class TestMLADualLayerPassPlan(CustomTestCase):
    def test_heterogeneous_local_windows_are_separate_groups(self):
        cfg = _head_config(
            [HEAD_LOCAL, HEAD_LOCAL, HEAD_GLOBAL],
            [64, 128, -1],
        )

        plan = _build_dual_layer_pass_plan(cfg, layer_id=0, swa_capacity=128)

        self.assertEqual(plan.local_groups, ((64, (0,)), (128, (1,))))
        self.assertEqual(plan.global_heads, (2,))
        self.assertEqual(plan.promoted_heads, ())
        self.assertEqual(plan.effective_local_heads, 2)

    def test_all_local_layer_does_not_require_global_pass(self):
        cfg = _head_config(
            [HEAD_LOCAL, HEAD_LOCAL, HEAD_LOCAL],
            [64, 128, 64],
        )

        plan = _build_dual_layer_pass_plan(cfg, layer_id=0, swa_capacity=128)

        self.assertEqual(plan.local_groups, ((64, (0, 2)), (128, (1,))))
        self.assertEqual(plan.global_heads, ())
        self.assertEqual(plan.effective_local_heads, 3)

    def test_window_beyond_swa_is_explicitly_promoted(self):
        cfg = _head_config(
            [HEAD_LOCAL, HEAD_LOCAL, HEAD_GLOBAL],
            [64, 256, -1],
        )

        plan = _build_dual_layer_pass_plan(cfg, layer_id=0, swa_capacity=128)

        self.assertEqual(plan.local_groups, ((64, (0,)),))
        self.assertEqual(plan.global_heads, (1, 2))
        self.assertEqual(plan.promoted_heads, ((1, 256),))

    def test_forced_local_rejects_unexecutable_window(self):
        self.assertEqual(_validate_forced_local_window(128, 128), 128)
        with self.assertRaisesRegex(ValueError, "physical SWA cache"):
            _validate_forced_local_window(256, 128)


class TestMLAOffLegacyKVIsolation(CustomTestCase):
    def test_only_full_row_mla_restore_preserves_online_kv(self):
        self.assertTrue(
            _redknot_mla_off_preserves_online_kv(
                {"mode": "restore", "reuse_mla_off": True}
            )
        )
        self.assertTrue(
            _redknot_mla_off_preserves_online_kv(
                {
                    "mode": "restore",
                    "reuse_mla_off": True,
                    "skip_forward": False,
                }
            )
        )
        for plan in (
            None,
            {},
            {"mode": "snapshot", "capture_mla_off": True},
            {"mode": "restore", "reuse_mla_off": False},
            {"mode": "restore", "reuse_mla_off": True, "skip_forward": True},
        ):
            self.assertFalse(_redknot_mla_off_preserves_online_kv(plan))

    @patch(
        "sglang.srt.layers.attention.redknot.dsv4_offline_reuse_v2."
        "get_offline_reuse_controller_v2"
    )
    @patch(
        "sglang.srt.layers.attention.redknot.dsv4_offline_reuse."
        "get_offline_reuse_controller"
    )
    def test_mla_restore_hook_does_not_touch_legacy_kv(
        self, get_controller, get_controller_v2
    ):
        backend = object.__new__(DeepseekV4AttnBackend)
        forward_batch = SimpleNamespace(
            forward_mode=redknot_mla_backend.ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([8]),
            redknot_reuse_plan=[
                {
                    "mode": "restore",
                    "reuse_mla_off": True,
                    "skip_forward": False,
                }
            ],
        )
        token_pool = MagicMock()

        backend._maybe_redknot_reuse_hook(2, forward_batch, token_pool)

        get_controller.assert_not_called()
        get_controller_v2.assert_not_called()
        self.assertEqual(token_pool.mock_calls, [])


class TestDeepSeekV4DeterministicRouter(CustomTestCase):
    def test_deterministic_router_preserves_dsv4_fp32_contract(self):
        gate = MoEGate(
            SimpleNamespace(
                n_routed_experts=4,
                hidden_size=8,
                topk_method="hash",
            ),
            quant_config=None,
            is_hash_moe=True,
            is_deepseek_v4=True,
        )
        gate.weight = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16))
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16)

        with patch(
            "sglang.srt.models.deepseek_v2.get_global_server_args",
            return_value=SimpleNamespace(enable_deterministic_inference=True),
        ), patch(
            "sglang.srt.models.deepseek_v2.use_intel_amx_backend",
            return_value=False,
        ):
            logits = gate(hidden_states)

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(
            logits, torch.nn.functional.linear(hidden_states, gate.weight).float()
        )


class TestMLAOffTPConsensus(CustomTestCase):
    class ScriptedGroup:
        def __init__(self, expected_local, reduced, *, dtype=torch.int64):
            self.expected_local = torch.tensor(expected_local, dtype=dtype)
            self.reduced = torch.tensor(reduced, dtype=dtype)
            self.calls = 0

        def all_reduce(self, value):
            self.calls += 1
            if value.dtype != self.expected_local.dtype or not torch.equal(
                value.cpu(), self.expected_local
            ):
                raise AssertionError(
                    f"unexpected local vote: {value.cpu().tolist()}"
                )
            return self.reduced.to(device=value.device)

    class ScalarSequenceGroup:
        def __init__(self, reduced_values):
            self.reduced_values = list(reduced_values)
            self.calls = 0

        def all_reduce(self, value):
            if value.dtype != torch.int32 or value.numel() != 1:
                raise AssertionError("expected one int32 finalize vote")
            if int(value.item()) != 1:
                raise AssertionError("this test rank must vote success")
            reduced = self.reduced_values[self.calls]
            self.calls += 1
            return value.new_tensor([reduced])

    @staticmethod
    def _backend(tp_size=2):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_tp_size = tp_size
        backend._redknot_runtime_counters = Counter()
        backend._redknot_mla_off_logged_failures = set()
        return backend

    @staticmethod
    def _preflight_signal(mode, plan):
        modes = ("none", "snapshot", "restore", "invalid")
        values = [0] * len(modes)
        values[modes.index(mode)] = 1
        if mode == "none":
            digest_a = digest_b = 0
        else:
            payload = json.dumps(
                plan,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            digest_a = int(digest[:5], 16)
            digest_b = int(digest[5:10], 16)
        return values + [
            digest_a,
            digest_a * digest_a,
            digest_b,
            digest_b * digest_b,
        ]

    @staticmethod
    def _startup_signal(enabled, model_hash, policy_hash, local_layers):
        layer_payload = json.dumps(
            {
                "local_layers": local_layers,
            },
            sort_keys=True,
        )
        layer_hash = hashlib.sha256(layer_payload.encode("utf-8")).hexdigest()
        values = [int(enabled)]
        provider_hash = hashlib.sha256(
            b"restore-provider-unavailable"
        ).hexdigest()
        for digest in (model_hash, policy_hash, layer_hash, provider_hash):
            first = int(digest[:5], 16)
            second = int(digest[5:10], 16)
            values.extend((first, first * first, second, second * second))
        return values

    def test_control_all_reduce_bypasses_data_plane_custom_kernel(self):
        backend = self._backend()
        device_group = object()
        group = SimpleNamespace(device_group=device_group, all_reduce=MagicMock())
        signal = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ), patch.object(
            redknot_mla_backend.torch.distributed,
            "all_reduce",
        ) as native_all_reduce:
            reduced = backend._mla_off_control_all_reduce(signal)

        self.assertIs(reduced, signal)
        native_all_reduce.assert_called_once_with(signal, group=device_group)
        group.all_reduce.assert_not_called()

    def test_decode_bypasses_mla_off_without_evidence_or_collective(self):
        backend = self._backend(tp_size=2)
        backend._redknot_mla_off_enabled = True
        backend._redknot_mla_off_local_layer_ids = (3,)
        forward_batch = SimpleNamespace(
            forward_mode=redknot_mla_backend.ForwardMode.DECODE,
            redknot_reuse_plan=[
                {
                    "mode": "restore",
                    "reuse_mla_off": True,
                    "benchmark_request_id": "decode-must-bypass",
                }
            ],
        )

        with patch.object(
            backend, "_mla_off_control_all_reduce"
        ) as control_reduce, patch.object(
            redknot_mla_backend.logger, "info"
        ) as log_info:
            context = backend.prepare_mla_off_context(
                layer_id=3,
                positions=torch.tensor([0]),
                forward_batch=forward_batch,
                q_head_count=8,
                q_row_count=1,
                n_local_heads=8,
                n_local_groups=1,
                head_dim=1,
                o_lora_rank=1,
                fp8_wo_a=False,
                device=torch.device("cpu"),
                projection_dtype=torch.float32,
            )

        self.assertIsNone(context)
        self.assertEqual(
            backend.runtime_counters(), {"mla_off.non_prefill_bypass": 1}
        )
        control_reduce.assert_not_called()
        log_info.assert_not_called()

    def test_auxiliary_layer_bypasses_mla_off_context_preparation(self):
        backend = self._backend(tp_size=1)

        context = backend.prepare_mla_off_context(
            layer_id=43,
            positions=torch.tensor([0]),
            forward_batch=SimpleNamespace(),
            q_head_count=8,
            q_row_count=1,
            n_local_heads=8,
            n_local_groups=1,
            head_dim=1,
            o_lora_rank=1,
            fp8_wo_a=False,
            device=torch.device("cpu"),
            projection_dtype=torch.float32,
        )

        self.assertIsNone(context)

    def test_context_reuse_requires_an_explicit_certified_range(self):
        backend = self._backend(tp_size=1)
        forward_batch = SimpleNamespace(
            orig_seq_lens=torch.tensor([1024], dtype=torch.int32)
        )

        backend._redknot_mla_off_certified_max_context_tokens = 0
        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {"total_tokens": 1024}, forward_batch
            ),
            "uncertified_context",
        )

        backend._redknot_mla_off_certified_max_context_tokens = 2048
        for missing in (None, True, 0, "not-an-int", 1024.0):
            self.assertEqual(
                backend._mla_off_context_safety_reason(
                    {"total_tokens": missing}, forward_batch
                ),
                "missing_total_tokens",
            )
        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {"total_tokens": 2049},
                SimpleNamespace(
                    orig_seq_lens=torch.tensor([2049], dtype=torch.int32)
                ),
            ),
            "context_exceeds_certification",
        )
        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {"total_tokens": 2048},
                SimpleNamespace(
                    orig_seq_lens=torch.tensor([2048], dtype=torch.int32)
                ),
            ),
            "",
        )

    def test_qualification_context_requires_server_and_plan_marker(self):
        backend = self._backend(tp_size=1)
        backend._redknot_mla_off_certified_max_context_tokens = 0
        backend._redknot_mla_off_qualification_only = True
        backend._redknot_mla_off_qualification_max_context_tokens = 2048
        forward_batch = SimpleNamespace(
            orig_seq_lens=torch.tensor([1024], dtype=torch.int32)
        )

        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {"total_tokens": 1024}, forward_batch
            ),
            "qualification_plan_marker_required",
        )
        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {
                    "total_tokens": 1024,
                    "mla_off_qualification_only": True,
                },
                forward_batch,
            ),
            "",
        )
        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {
                    "total_tokens": 2049,
                    "mla_off_qualification_only": True,
                },
                SimpleNamespace(
                    orig_seq_lens=torch.tensor([2049], dtype=torch.int32)
                ),
            ),
            "context_exceeds_qualification",
        )

        backend._redknot_mla_off_qualification_only = False
        backend._redknot_mla_off_qualification_max_context_tokens = 0
        backend._redknot_mla_off_certified_max_context_tokens = 2048
        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {
                    "total_tokens": 1024,
                    "mla_off_qualification_only": True,
                },
                forward_batch,
            ),
            "qualification_plan_marker_forbidden",
        )

    def test_chunked_restore_uses_the_same_qualification_cap_gate(self):
        backend = self._backend(tp_size=1)
        backend._redknot_mla_off_certified_max_context_tokens = 0
        backend._redknot_mla_off_qualification_only = True
        backend._redknot_mla_off_qualification_max_context_tokens = 2048
        backend._redknot_mla_off_model_hash = _CONTEXT_MODEL_HASH
        backend._redknot_mla_off_policy_hash = _CONTEXT_POLICY_HASH
        base_plan = _context_restore_plan(
            1024, 1, suffix=tuple(range(32))
        )
        base_plan.pop("mla_off_qualification_only")
        common = {
            "scheduler_total_tokens": 1056,
            "scheduler_current_extent": 512,
            "request_positions": tuple(range(512)),
        }

        self.assertEqual(
            _active_restore_plan_fallback_reason(
                backend, plan=base_plan, **common
            ),
            "qualification_plan_marker_required",
        )
        self.assertEqual(
            _active_restore_plan_fallback_reason(
                backend,
                plan={**base_plan, "mla_off_qualification_only": True},
                **common,
            ),
            "",
        )
        for diagnostic_ablation in ("zoff_only", "shared_only"):
            qualification_plan = {
                **base_plan,
                "mla_off_qualification_only": True,
                "mla_off_diagnostic_ablation": diagnostic_ablation,
            }
            with self.subTest(diagnostic_ablation=diagnostic_ablation):
                self.assertEqual(
                    backend._mla_off_resolve_context_cap(qualification_plan),
                    (
                        0,
                        "qualification_requires_full_diagnostic_ablation",
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError, "requires diagnostic_ablation=full"
                ):
                    _validate_pure_headsplit_plan_contract(qualification_plan)
                self.assertEqual(
                    _active_restore_plan_fallback_reason(
                        backend,
                        plan=qualification_plan,
                        **common,
                    ),
                    "pure_headsplit_contract:ValueError",
                )
        with self.assertRaisesRegex(ValueError, "marker must be a boolean"):
            _validate_pure_headsplit_plan_contract(
                {**base_plan, "mla_off_qualification_only": 1}
            )

    def test_context_certification_binds_scheduler_owned_length(self):
        backend = self._backend(tp_size=1)
        backend._redknot_mla_off_certified_max_context_tokens = 4096

        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {"total_tokens": 1024}, SimpleNamespace()
            ),
            "missing_actual_total_tokens",
        )

        cached_batch = SimpleNamespace(
            orig_seq_lens=torch.tensor([1024], dtype=torch.int32),
            _redknot_forward_generation_id=(123, 1),
        )
        self.assertTrue(
            backend._mla_off_prepare_forward_generation(cached_batch)
        )
        self.assertEqual(
            backend._mla_off_scheduler_total_tokens(cached_batch), 1024
        )
        self.assertIs(
            cached_batch._redknot_mla_off_scheduler_total_tokens_cache[1],
            cached_batch.orig_seq_lens,
        )
        self.assertEqual(
            backend._mla_off_scheduler_total_tokens(cached_batch), 1024
        )

        self.assertEqual(
            backend._mla_off_context_safety_reason(
                {"total_tokens": 1024},
                SimpleNamespace(
                    orig_seq_lens=torch.tensor([2048], dtype=torch.int32)
                ),
            ),
            "total_tokens_mismatch",
        )
        for invalid_actual in (
            torch.tensor([1024.0]),
            torch.tensor([1024, 1024], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([64], dtype=torch.int8),
            [1024],
        ):
            self.assertEqual(
                backend._mla_off_context_safety_reason(
                    {"total_tokens": 1024},
                    SimpleNamespace(orig_seq_lens=invalid_actual),
                ),
                "missing_actual_total_tokens",
            )

    def test_explicit_refresh_precedes_context_certification(self):
        backend = self._backend(tp_size=1)
        backend._redknot_mla_off_certified_max_context_tokens = 0

        explicit, reason = backend._mla_off_restore_safety(
            plan={"mla_off_refresh_layers": [2], "total_tokens": 1024},
            layer_id=2,
            forward_batch=SimpleNamespace(),
        )
        self.assertTrue(explicit)
        self.assertEqual(reason, "")

        explicit, reason = backend._mla_off_restore_safety(
            plan={"mla_off_refresh_layers": [2], "total_tokens": 1024},
            layer_id=3,
            forward_batch=SimpleNamespace(),
        )
        self.assertFalse(explicit)
        self.assertEqual(reason, "uncertified_context")

    def test_forward_identity_is_stable_per_forward_invocation(self):
        backend = self._backend(tp_size=1)
        plan = {"benchmark_request_id": "request-a"}
        first_batch = SimpleNamespace(
            forward_mode=redknot_mla_backend.ForwardMode.EXTEND
        )
        second_batch = SimpleNamespace(
            forward_mode=redknot_mla_backend.ForwardMode.MIXED
        )

        first = backend._mla_off_forward_evidence(
            forward_batch=first_batch,
            plan=plan,
            positions=torch.arange(512),
            q_rows=512,
        )
        first_again = backend._mla_off_forward_evidence(
            forward_batch=first_batch,
            plan=plan,
            positions=torch.arange(512),
            q_rows=512,
        )
        reused_batch = backend._mla_off_forward_evidence(
            forward_batch=first_batch,
            plan=plan,
            positions=torch.arange(512, 529),
            q_rows=17,
        )
        same_shape_reused_batch = backend._mla_off_forward_evidence(
            forward_batch=first_batch,
            plan=plan,
            positions=torch.arange(512, 1024),
            q_rows=512,
        )
        second = backend._mla_off_forward_evidence(
            forward_batch=second_batch,
            plan=plan,
            positions=torch.arange(512, 529),
            q_rows=17,
        )

        self.assertEqual(first[0], "request-a")
        self.assertRegex(first[1], r"^f[0-9a-f]{16}$")
        self.assertEqual(first[2:], ("extend", 512, 0, 512))
        self.assertEqual(first_again, first)
        self.assertNotEqual(reused_batch[1], first[1])
        self.assertEqual(reused_batch[2:], ("extend", 17, 512, 529))
        self.assertNotEqual(same_shape_reused_batch[1], first[1])
        self.assertEqual(
            same_shape_reused_batch[2:], ("extend", 512, 512, 1024)
        )
        self.assertNotEqual(second[1], first[1])
        self.assertEqual(second[0], "request-a")
        self.assertEqual(second[2:], ("mixed", 17, 512, 529))

    def test_strict_forward_identity_rechecks_mutated_inference_positions(self):
        backend = self._backend(tp_size=1)
        backend._redknot_mla_off_strict_row_verify = True
        plan = {"benchmark_request_id": "request-a"}
        forward_batch = SimpleNamespace(
            forward_mode=redknot_mla_backend.ForwardMode.EXTEND
        )

        with torch.inference_mode():
            positions = torch.arange(4)
            first = backend._mla_off_forward_evidence(
                forward_batch=forward_batch,
                plan=plan,
                positions=positions,
                q_rows=4,
            )
            positions.add_(4)
            second = backend._mla_off_forward_evidence(
                forward_batch=forward_batch,
                plan=plan,
                positions=positions,
                q_rows=4,
            )

        self.assertNotEqual(second[1], first[1])
        self.assertEqual(first[2:], ("extend", 4, 0, 4))
        self.assertEqual(second[2:], ("extend", 4, 4, 8))

    def test_restore_layout_certificate_binds_all_layers_and_mask(self):
        reusable = torch.tensor([True, False, True, False])
        dirty = torch.tensor([1, 3], dtype=torch.long)
        layout_key = (123, (7, 11), "positions", "tokens", None)
        certificate = (
            redknot_mla_backend._MLAOffRestoreLayoutCertificate(
                layout_key=layout_key,
                certified_layer_ids=(2, 3),
                restore_rows=(),
                reusable_cpu=reusable,
                reuse_mask_digest=(17, 29),
                reused_count=2,
                dirty_rows_cpu=dirty,
                segments_by_hash={},
                reusable_identity=(
                    redknot_mla_backend._mla_off_control_tensor_identity(
                        reusable
                    )
                ),
                dirty_identity=(
                    redknot_mla_backend._mla_off_control_tensor_identity(dirty)
                ),
            )
        )

        for layer_id in (2, 3):
            certificate.validate(
                layer_id=layer_id,
                layout_key=layout_key,
                reusable_cpu=reusable,
                dirty_rows_cpu=dirty,
                reuse_mask_digest=(17, 29),
                q_rows=4,
                reused_count=2,
                online_count=2,
            )
        with self.assertRaisesRegex(ValueError, "not certified for this layer"):
            certificate.validate(
                layer_id=4,
                layout_key=layout_key,
                reusable_cpu=reusable,
                dirty_rows_cpu=dirty,
                reuse_mask_digest=(17, 29),
                q_rows=4,
            )
        with self.assertRaisesRegex(ValueError, "plan/input certificate"):
            certificate.validate(
                layer_id=2,
                layout_key=(124, (7, 11), "positions", "tokens", None),
                reusable_cpu=reusable,
                dirty_rows_cpu=dirty,
                reuse_mask_digest=(17, 29),
                q_rows=4,
            )

        reusable[0] = False
        with self.assertRaisesRegex(ValueError, "mask certificate is stale"):
            certificate.validate(
                layer_id=2,
                layout_key=layout_key,
                reusable_cpu=reusable,
                dirty_rows_cpu=dirty,
                reuse_mask_digest=(17, 29),
                q_rows=4,
            )

    def test_full_local_rows_are_counted_in_online_denominator(self):
        backend = self._backend(tp_size=1)

        with patch.dict(
            os.environ, {"REDKNOT_MLA_OFF_METRICS": "1"}
        ), patch.object(redknot_mla_backend.logger, "info") as log_info:
            backend._mla_off_record_runtime_rows(
                request_id="request-a",
                forward_id="f2",
                forward_mode="extend",
                q_rows=22,
                layer_id=2,
                reused_local_head_rows=0,
                online_local_head_rows=154,
                online_global_head_rows=22,
            )

        self.assertEqual(
            backend.runtime_counters(),
            {
                "mla_off.reused_local_head_rows": 0,
                "mla_off.online_local_head_rows": 154,
                "mla_off.online_global_head_rows": 22,
            },
        )
        self.assertEqual(len(log_info.call_args_list), 1)
        self.assertEqual(
            log_info.call_args.args[1:],
            ("request-a", "f2", "extend", 22, 2, 0, 154, 22, "full"),
        )

    def test_restore_consensus_uses_native_int64_collective(self):
        backend = self._backend()
        context = SimpleNamespace(reuse_mask_digest=(3, 7))
        device_group = object()
        group = SimpleNamespace(device_group=device_group, all_reduce=MagicMock())

        def native_reduce(signal, *, group):
            self.assertIs(group, device_group)
            self.assertEqual(signal.dtype, torch.int64)
            self.assertEqual(signal.tolist(), [1, 1, 3, 9, 7, 49])
            signal.copy_(signal.new_tensor([2, 2, 6, 18, 14, 98]))

        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ), patch.object(
            redknot_mla_backend.torch.distributed,
            "all_reduce",
            side_effect=native_reduce,
        ) as native_all_reduce:
            resolved, reason = backend._mla_off_resolve_restore_context(
                context,
                intentional_full_local=False,
                device=torch.device("cpu"),
            )

        self.assertIs(resolved, context)
        self.assertEqual(reason, "")
        self.assertEqual(native_all_reduce.call_count, 1)
        group.all_reduce.assert_not_called()

    def test_startup_consensus_rejects_policy_drift_on_every_rank(self):
        backend = self._backend()
        backend.device = torch.device("cpu")
        backend._redknot_mla_off_enabled = True
        backend._redknot_mla_off_local_layer_ids = (2, 3)
        backend._redknot_mla_off_rank_local_layer_ids = (2, 3)
        backend._redknot_mla_off_model_hash = hashlib.sha256(b"model").hexdigest()
        backend._redknot_mla_off_policy_hash = hashlib.sha256(b"policy-a").hexdigest()
        backend._redknot_shared_latent_enabled = False
        local_signal = self._startup_signal(
            True,
            backend._redknot_mla_off_model_hash,
            backend._redknot_mla_off_policy_hash,
            (2, 3),
        )
        peer_signal = self._startup_signal(
            True,
            backend._redknot_mla_off_model_hash,
            hashlib.sha256(b"policy-b").hexdigest(),
            (2, 3),
        )
        group = self.ScriptedGroup(
            local_signal,
            [left + right for left, right in zip(local_signal, peer_signal)],
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            with self.assertRaisesRegex(RuntimeError, "startup contract drift"):
                backend._mla_off_initialize_tp_consensus()
        self.assertFalse(backend._redknot_mla_off_enabled)
        self.assertEqual(group.calls, 1)

    def test_preflight_rejects_missing_plan_before_restore_votes(self):
        backend = self._backend()
        group = self.ScriptedGroup(
            self._preflight_signal("none", None),
            [1, 0, 1, 0, 0, 0, 0, 0],
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            mode, reason = backend._mla_off_preflight_mode(
                "none", None, torch.device("cpu")
            )
        self.assertIsNone(mode)
        self.assertIn("mode differs", reason)
        self.assertEqual(group.calls, 1)

    def test_preflight_rejects_same_mode_with_different_plans(self):
        backend = self._backend()
        local_plan = {"mode": "restore"}
        peer_plan = {"mode": "restore", "query_start": 8}
        local_signal = self._preflight_signal("restore", local_plan)
        peer_signal = self._preflight_signal("restore", peer_plan)
        group = self.ScriptedGroup(
            local_signal,
            [left + right for left, right in zip(local_signal, peer_signal)],
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            mode, reason = backend._mla_off_preflight_mode(
                "restore", local_plan, torch.device("cpu")
            )
        self.assertIsNone(mode)
        self.assertIn("plan differs", reason)
        self.assertEqual(group.calls, 1)

    def test_preflight_accepts_identical_restore_plan(self):
        backend = self._backend()

        plan = {"mode": "restore"}
        local_signal = self._preflight_signal("restore", plan)
        group = self.ScriptedGroup(
            local_signal, [2 * value for value in local_signal]
        )

        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            mode, reason = backend._mla_off_preflight_mode(
                "restore", plan, torch.device("cpu")
            )
        self.assertEqual(mode, "restore")
        self.assertEqual(reason, "")
        self.assertEqual(group.calls, 1)

    def test_preflight_reuses_certified_local_plan_digest(self):
        backend = self._backend()
        plan = {"mode": "restore", "segments": []}
        digest = backend._mla_off_plan_digest(plan)
        local_signal = self._preflight_signal("restore", plan)
        group = self.ScriptedGroup(
            local_signal, [2 * value for value in local_signal]
        )

        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ), patch.object(
            backend,
            "_mla_off_plan_digest",
            side_effect=AssertionError("plan was serialized twice"),
        ):
            mode, reason = backend._mla_off_preflight_mode(
                "restore",
                plan,
                torch.device("cpu"),
                local_plan_digest=digest,
            )
        self.assertEqual(mode, "restore")
        self.assertEqual(reason, "")
        self.assertEqual(group.calls, 1)

    def test_preflight_turns_non_json_plan_into_collective_invalid_vote(self):
        backend = self._backend()
        group = self.ScriptedGroup(
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0],
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            mode, reason = backend._mla_off_preflight_mode(
                "restore", {"bad": object()}, torch.device("cpu")
            )
        self.assertIsNone(mode)
        self.assertIn("invalid", reason)
        self.assertEqual(group.calls, 1)

    def test_restore_mixed_context_forces_full_online(self):
        backend = self._backend()
        context = object()
        group = self.ScriptedGroup(
            [1, 1, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0]
        )

        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            resolved, reason = backend._mla_off_resolve_restore_context(
                context,
                intentional_full_local=False,
                device=torch.device("cpu"),
            )

        self.assertIsNone(resolved)
        self.assertEqual(reason, "restore_context_mismatch")
        self.assertEqual(group.calls, 1)

    def test_restore_not_ready_stops_before_context_vote(self):
        backend = self._backend()
        group = self.ScriptedGroup(
            [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]
        )

        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            resolved, reason = backend._mla_off_resolve_restore_context(
                None,
                intentional_full_local=False,
                device=torch.device("cpu"),
            )

        self.assertIsNone(resolved)
        self.assertEqual(reason, "restore_not_ready")

    def test_post_attention_consumer_stage_requires_all_rank_success(self):
        backend = self._backend()
        group = self.ScriptedGroup(
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 1],
            dtype=torch.int32,
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            accepted, reason = backend.resolve_mla_off_consumer_stage(
                stage="indexed_pipeline",
                local_success=False,
                device=torch.device("cpu"),
            )
        self.assertFalse(accepted)
        self.assertEqual(reason, "consumer_stage_failed")
        self.assertEqual(group.calls, 1)

    def test_post_attention_consumer_stage_detects_rank_path_drift(self):
        backend = self._backend()
        group = self.ScriptedGroup(
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 2],
            dtype=torch.int32,
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            accepted, reason = backend.resolve_mla_off_consumer_stage(
                stage="indexed_pipeline",
                local_success=True,
                device=torch.device("cpu"),
            )
        self.assertFalse(accepted)
        self.assertEqual(reason, "consumer_stage_mismatch")
        self.assertEqual(group.calls, 1)

    def test_post_attention_consumer_stage_supports_tp1(self):
        backend = self._backend(tp_size=1)
        accepted, reason = backend.resolve_mla_off_consumer_stage(
            stage="projection_merge",
            local_success=True,
            device=torch.device("cpu"),
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "")

        accepted, reason = backend.resolve_mla_off_consumer_stage(
            stage="projection_merge",
            local_success=False,
            device=torch.device("cpu"),
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "consumer_stage_failed")

        accepted, reason = backend.resolve_mla_off_consumer_stage(
            stage="pure_headsplit_projection_merge",
            local_success=True,
            device=torch.device("cpu"),
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "")

    def test_attention_application_vote_includes_local_failure(self):
        backend = self._backend(tp_size=2)
        with patch.object(
            backend, "_mla_off_vote_count", return_value=1
        ) as vote_count:
            accepted, reason = backend.resolve_mla_off_attention_application(
                local_success=False, device=torch.device("cpu")
            )
        self.assertFalse(accepted)
        self.assertEqual(reason, "attention_application_failed")
        vote_count.assert_called_once_with(False, torch.device("cpu"))

        with patch.object(
            backend, "_mla_off_vote_count", return_value=2
        ):
            accepted, reason = backend.resolve_mla_off_attention_application(
                local_success=True, device=torch.device("cpu")
            )
        self.assertTrue(accepted)
        self.assertEqual(reason, "")

    def test_pre_attention_restore_requires_uniform_compact_eligibility(self):
        backend = self._backend(tp_size=2)
        with patch.object(
            backend,
            "_mla_off_vote_count",
            side_effect=[2, 1],
        ) as vote_count:
            ready, row_skip, compact_woa, reason = (
                backend._mla_off_resolve_pre_attention_restore(
                    local_ready=True,
                    local_compact_eligible=True,
                    device=torch.device("cpu"),
                )
            )
        self.assertFalse(ready)
        self.assertFalse(row_skip)
        self.assertFalse(compact_woa)
        self.assertEqual(reason, "compact_eligibility_mismatch")
        self.assertEqual(vote_count.call_count, 2)

        with patch.object(
            backend,
            "_mla_off_vote_count",
            side_effect=[1],
        ) as vote_count:
            ready, row_skip, compact_woa, reason = (
                backend._mla_off_resolve_pre_attention_restore(
                    local_ready=False,
                    local_compact_eligible=True,
                    device=torch.device("cpu"),
                )
            )
        self.assertFalse(ready)
        self.assertFalse(row_skip)
        self.assertFalse(compact_woa)
        self.assertEqual(reason, "pre_attention_recheck_failed")
        self.assertEqual(vote_count.call_count, 1)

        with patch.object(
            backend,
            "_mla_off_vote_count",
            side_effect=[2, 2, 2],
        ) as vote_count:
            ready, row_skip, compact_woa, reason = (
                backend._mla_off_resolve_pre_attention_restore(
                    local_ready=True,
                    local_compact_eligible=True,
                    device=torch.device("cpu"),
                )
            )
        self.assertTrue(ready)
        self.assertTrue(row_skip)
        self.assertTrue(compact_woa)
        self.assertEqual(reason, "")
        self.assertEqual(vote_count.call_count, 3)

        # The control arm keeps the certified attention-row skip while every
        # TP rank selects the legacy full-row inverse-RoPE/wo_a consumer.
        with patch.object(
            backend,
            "_mla_off_vote_count",
            side_effect=[2, 2, 0],
        ) as vote_count:
            ready, row_skip, compact_woa, reason = (
                backend._mla_off_resolve_pre_attention_restore(
                    local_ready=True,
                    local_compact_eligible=True,
                    local_compact_requested=False,
                    device=torch.device("cpu"),
                )
            )
        self.assertTrue(ready)
        self.assertTrue(row_skip)
        self.assertFalse(compact_woa)
        self.assertEqual(reason, "")
        self.assertEqual(vote_count.call_count, 3)

        with patch.object(
            backend,
            "_mla_off_vote_count",
            side_effect=[2, 2, 1],
        ):
            ready, row_skip, compact_woa, reason = (
                backend._mla_off_resolve_pre_attention_restore(
                    local_ready=True,
                    local_compact_eligible=True,
                    local_compact_requested=True,
                    device=torch.device("cpu"),
                )
            )
        self.assertFalse(ready)
        self.assertFalse(row_skip)
        self.assertFalse(compact_woa)
        self.assertEqual(reason, "compact_mode_mismatch")

    def test_compact_preflight_failure_enters_restore_not_ready_fallback(self):
        backend = self._backend(tp_size=1)
        reuse_mask = torch.tensor([True, False], dtype=torch.bool)
        dirty_rows = torch.tensor([1], dtype=torch.long)
        digest = (17, 29)
        layout_key = ("forward", "layout")
        layout = redknot_mla_backend._MLAOffRestoreLayoutCertificate(
            layout_key=layout_key,
            certified_layer_ids=(2,),
            restore_rows=(),
            reusable_cpu=reuse_mask,
            reuse_mask_digest=digest,
            reused_count=1,
            dirty_rows_cpu=dirty_rows,
            segments_by_hash={},
            reusable_identity=(
                redknot_mla_backend._mla_off_control_tensor_identity(
                    reuse_mask
                )
            ),
            dirty_identity=(
                redknot_mla_backend._mla_off_control_tensor_identity(
                    dirty_rows
                )
            ),
        )
        controller = DSV4MLAOffController(
            max_cache_bytes=1024, max_device_cache_bytes=0
        )
        rows_certificate = controller.prepare_device_indices(
            dirty_rows,
            device=torch.device("cpu"),
            role="online_local_rows",
            semantic_digest=digest,
            upper_bound=2,
        )
        context = MLAOffRuntimeContext(
            mode="restore",
            layer_id=2,
            spec=MLAOffLayerSpec(
                layer_id=2,
                tp_rank=0,
                tp_size=1,
                owned_logical_heads=(0, 1),
                offline_local_heads=(0, 1),
                num_output_groups=1,
                heads_per_group=2,
                head_dim=4,
                o_lora_rank=3,
                model_compat_hash="model",
                head_policy_hash="policy",
            ),
            local_head_axes=(0, 1),
            controller=controller,
            # Deliberately one row short: this must fail before a TP rank can
            # vote its restore context ready or enter attention.
            offline_projection=torch.zeros((1, 1, 3)),
            reuse_mask=reuse_mask,
            online_local_row_indices=rows_certificate.device_indices,
            reused_row_count=1,
            online_local_row_count=1,
            reuse_mask_digest=digest,
            online_local_row_indices_cpu=dirty_rows,
            online_local_row_indices_certificate=rows_certificate,
            restore_layout_certificate=layout,
        )

        with self.assertRaisesRegex(ValueError, "offline projection"):
            context.prevalidate_compact_woa(
                torch.arange(2, dtype=torch.long),
                total_rows=2,
                device=torch.device("cpu"),
                projection_dtype=torch.float32,
            )
        resolved, reason = backend._mla_off_resolve_restore_context(
            None,
            intentional_full_local=False,
            device=torch.device("cpu"),
        )
        self.assertIsNone(resolved)
        self.assertEqual(reason, "restore_not_ready")

    def test_restore_reuse_mask_mismatch_forces_full_online(self):
        backend = self._backend()
        context = SimpleNamespace(reuse_mask_digest=(3, 7))
        group = self.ScriptedGroup(
            [1, 1, 3, 9, 7, 49], [2, 2, 7, 25, 14, 98]
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            resolved, reason = backend._mla_off_resolve_restore_context(
                context,
                intentional_full_local=False,
                device=torch.device("cpu"),
            )
        self.assertIsNone(resolved)
        self.assertEqual(reason, "restore_mask_mismatch")
        self.assertEqual(group.calls, 1)

    def test_snapshot_input_layout_mismatch_aborts_before_attention(self):
        backend = self._backend()
        context = SimpleNamespace(input_layout_digest=(3, 7))
        group = self.ScriptedGroup(
            [1, 3, 9, 7, 49], [2, 7, 25, 14, 98]
        )
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            resolved, reason = backend._mla_off_resolve_snapshot_context(
                context, device=torch.device("cpu")
            )
        self.assertIsNone(resolved)
        self.assertEqual(reason, "snapshot_layout_mismatch")
        self.assertEqual(group.calls, 1)

    def test_snapshot_finalize_confirms_only_after_all_publish(self):
        backend = self._backend()
        receipt = object()

        class Context:
            is_snapshot = True
            controller = SimpleNamespace(clear=MagicMock())

            @staticmethod
            def snapshot_complete():
                return True

            publish_snapshot = MagicMock(return_value=receipt)
            validate_snapshot_confirmation = MagicMock()
            confirm_snapshot = MagicMock()
            rollback_snapshot = MagicMock()
            abort_snapshot = MagicMock()

        context = Context()
        group = self.ScalarSequenceGroup([2, 2, 2, 2])
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            ready = backend.finalize_mla_off_snapshot_context(
                context,
                capture_succeeded=True,
                device=torch.device("cpu"),
            )
        self.assertTrue(ready)
        self.assertEqual(group.calls, 4)
        context.validate_snapshot_confirmation.assert_called_once_with(receipt)
        context.confirm_snapshot.assert_called_once_with(receipt)
        context.rollback_snapshot.assert_not_called()

    def test_snapshot_finalize_rolls_back_partial_publish(self):
        backend = self._backend()
        receipt = object()

        class Context:
            is_snapshot = True
            controller = SimpleNamespace(clear=MagicMock())

            @staticmethod
            def snapshot_complete():
                return True

            publish_snapshot = MagicMock(return_value=receipt)
            validate_snapshot_confirmation = MagicMock()
            confirm_snapshot = MagicMock()
            rollback_snapshot = MagicMock()
            abort_snapshot = MagicMock()

        context = Context()
        group = self.ScalarSequenceGroup([2, 2, 1, 2])
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            ready = backend.finalize_mla_off_snapshot_context(
                context,
                capture_succeeded=True,
                device=torch.device("cpu"),
            )
        self.assertFalse(ready)
        self.assertEqual(group.calls, 4)
        context.rollback_snapshot.assert_called_once_with(receipt)
        context.abort_snapshot.assert_called_once()
        context.validate_snapshot_confirmation.assert_not_called()
        context.confirm_snapshot.assert_not_called()

    def test_snapshot_finalize_clears_cache_if_publish_vote_raises(self):
        backend = self._backend()
        receipt = object()

        class Context:
            is_snapshot = True
            controller = SimpleNamespace(clear=MagicMock())

            @staticmethod
            def snapshot_complete():
                return True

            publish_snapshot = MagicMock(return_value=receipt)
            validate_snapshot_confirmation = MagicMock()
            confirm_snapshot = MagicMock()
            rollback_snapshot = MagicMock()
            abort_snapshot = MagicMock()

        class FailingPublishVoteGroup:
            def __init__(self):
                self.calls = 0

            def all_reduce(self, value):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("injected TP publish-vote failure")
                return value.new_tensor([2])

        context = Context()
        group = FailingPublishVoteGroup()
        with patch.object(
            redknot_mla_backend,
            "get_attention_tp_group",
            return_value=group,
        ):
            with self.assertRaisesRegex(RuntimeError, "publish-vote"):
                backend.finalize_mla_off_snapshot_context(
                    context,
                    capture_succeeded=True,
                    device=torch.device("cpu"),
                )
        self.assertEqual(group.calls, 3)
        context.publish_snapshot.assert_called_once()
        context.controller.clear.assert_called_once()
        context.validate_snapshot_confirmation.assert_not_called()
        context.confirm_snapshot.assert_not_called()

    def test_shared_snapshot_post_confirm_observability_cannot_split_tp(self):
        backend = self._backend(tp_size=1)
        backend._redknot_tp_rank = 0
        backend._redknot_mla_off_model_hash = _CONTEXT_MODEL_HASH
        backend._redknot_mla_off_policy_hash = _CONTEXT_POLICY_HASH
        backend._redknot_mla_off_execution_profile = MLA_OFF_EXECUTION_PROFILE
        backend._redknot_mla_off_rank_local_layer_ids = tuple(range(3, 40))
        stage_key = ("segment", "generation")
        backend._redknot_shared_snapshot_stages = {
            stage_key: {
                "seg_hash": "sha256:" + "1" * 64,
                "token_hash": "sha256:" + "2" * 64,
                "generation_id": "generation",
            }
        }

        local_prepared = SimpleNamespace(
            digest="sha256:" + "3" * 64,
            shared_latent_digest="sha256:" + "4" * 64,
        )
        prepared_publish = object()
        published = object()
        adapter = SimpleNamespace(
            prepare_local=MagicMock(return_value=local_prepared),
            prepare_publish=MagicMock(return_value=prepared_publish),
            publish=MagicMock(return_value=published),
            validate_confirmation=MagicMock(
                return_value="sha256:" + "5" * 64
            ),
            confirm=MagicMock(),
            rollback=MagicMock(),
        )
        prepared_context_publication = SimpleNamespace(
            receipt="sha256:" + "6" * 64,
            commit_noexcept=MagicMock(),
            poison_noexcept=MagicMock(),
        )
        backend._redknot_context_token_streams = SimpleNamespace(
            prepare_snapshot_publication=MagicMock(
                return_value=prepared_context_publication
            )
        )
        backend._mla_off_vote_restore_ready = MagicMock(return_value=True)

        def digest_vote(*, local_digest, **_kwargs):
            return True, (local_digest,), "sha256:" + "7" * 64

        backend._mla_off_snapshot_digest_vote = MagicMock(
            side_effect=digest_vote
        )
        backend._mla_off_quarantine_shared_latent = MagicMock()
        backend._count = MagicMock(
            side_effect=RuntimeError("injected post-confirm metric failure")
        )
        backend._mla_off_maybe_emit_shared_snapshot_audit = MagicMock(
            side_effect=SystemExit("injected post-confirm audit failure")
        )
        context = SimpleNamespace(
            is_snapshot=True,
            shared_snapshot_enabled=True,
            shared_snapshot_adapter=adapter,
            shared_snapshot_session=object(),
            shared_snapshot_stage_key=stage_key,
            context_snapshot_request_binding=("request-binding",),
            layer_id=39,
            length=8192,
        )

        with patch.object(
            redknot_mla_backend.torch.cuda,
            "current_stream",
            return_value=object(),
        ):
            ready = backend.finalize_mla_off_snapshot_context(
                context,
                capture_succeeded=True,
                device=torch.device("cpu"),
            )

        self.assertTrue(ready)
        prepared_context_publication.commit_noexcept.assert_called_once_with()
        backend._count.assert_called_once_with(
            "mla_off.shared_snapshot_published"
        )
        backend._mla_off_maybe_emit_shared_snapshot_audit.assert_called_once_with(
            adapter=adapter, published=published
        )
        backend._mla_off_quarantine_shared_latent.assert_not_called()
        prepared_context_publication.poison_noexcept.assert_not_called()

        # If the irreversible adapter confirmation fails on this rank, the
        # final TP vote takes every rank into quarantine and the already-
        # prepared context receipt must be poisoned, not left unevictable.
        backend._redknot_shared_snapshot_stages[stage_key] = {
            "seg_hash": "sha256:" + "1" * 64,
            "token_hash": "sha256:" + "2" * 64,
            "generation_id": "generation",
        }
        adapter.confirm.side_effect = RuntimeError(
            "injected irreversible confirmation failure"
        )
        backend._count.side_effect = None
        backend._mla_off_maybe_emit_shared_snapshot_audit.side_effect = None
        backend._mla_off_vote_restore_ready = MagicMock(
            side_effect=(True, False)
        )
        prepared_context_publication.commit_noexcept.reset_mock()
        with patch.object(
            redknot_mla_backend.torch.cuda,
            "current_stream",
            return_value=object(),
        ), self.assertRaisesRegex(RuntimeError, "visibility gate"):
            backend.finalize_mla_off_snapshot_context(
                context,
                capture_succeeded=True,
                device=torch.device("cpu"),
            )
        prepared_context_publication.poison_noexcept.assert_called_once_with()
        prepared_context_publication.commit_noexcept.assert_not_called()
        backend._mla_off_quarantine_shared_latent.assert_called_once()


class TestMLADualForwardContract(CustomTestCase):
    def test_mixed_windows_promotions_and_single_cache_hook(self):
        class FakeDSV4AttnMetadata:
            pass

        class FakeTokenToKVPool:
            pass

        class FakeForwardMode:
            def is_idle(self):
                return False

            def is_draft_extend(self, include_v2=False):
                return False

        head_cfg = _head_config(
            [HEAD_LOCAL, HEAD_LOCAL, HEAD_LOCAL, HEAD_GLOBAL],
            [64, 128, 256, -1],
            layer_id=3,
        )
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend.redknot_mla_pass_mode = "dual"
        backend.redknot_mla_head_cfg = head_cfg
        backend._redknot_swa_capacity = 128
        backend._redknot_dual_layer_plans = (None, None, None) + (
            _build_dual_layer_pass_plan(head_cfg, layer_id=3, swa_capacity=128),
        )
        backend._redknot_runtime_counters = Counter()
        backend._redknot_logged_paths = set()
        backend._redknot_trace_actual_passes = False
        backend.mtp_enabled = False
        backend.head_dim_v = 1
        backend.softmax_scale = 1.0
        backend._maybe_upgrade_forward_metadata = MagicMock()
        backend.store_cache = MagicMock()
        backend._maybe_redknot_reuse_hook = MagicMock()

        global_metadata = object()
        core_metadata = FakeDSV4AttnMetadata()
        core_metadata.swa_page_indices = torch.arange(128, dtype=torch.int32).reshape(
            1, 128
        )
        core_metadata.swa_topk_lengths = torch.tensor([128], dtype=torch.int32)
        core_metadata.c4_sparse_page_indices = torch.arange(
            64, dtype=torch.int32
        ).reshape(1, 64)
        core_metadata.c4_sparse_topk_lengths = torch.tensor([7], dtype=torch.int32)
        core_metadata.get_flashmla_metadata = MagicMock(return_value=global_metadata)
        backend.forward_metadata = SimpleNamespace(core_attn_metadata=core_metadata)

        token_pool = FakeTokenToKVPool()
        token_pool.swa_window_size = 128
        token_pool.page_size = 256
        token_pool.swa_kv_pool = SimpleNamespace(kv_cache_total_dim=1)
        token_pool.get_swa_key_buffer_radix = MagicMock(
            return_value=torch.zeros(1, 128)
        )
        token_pool.get_extra_key_buffer = MagicMock(return_value=torch.zeros(1, 64))
        backend.token_to_kv_pool = token_pool

        flash_calls = []

        def fake_flash_mla_with_kvcache(**kwargs):
            flash_calls.append(kwargs)
            is_global = kwargs["extra_k_cache"] is not None
            value = 1000.0 if is_global else float(kwargs["topk_length"].max().item())
            q_arg = kwargs["q"]
            output = torch.full(
                (q_arg.shape[0], q_arg.shape[1], q_arg.shape[2], 1),
                value,
                dtype=q_arg.dtype,
            )
            return (output,)

        fake_flash_mla = SimpleNamespace(
            flash_mla_with_kvcache=MagicMock(side_effect=fake_flash_mla_with_kvcache)
        )
        fake_envs = SimpleNamespace(
            SGLANG_REDKNOT_OFFLINE_REUSE=SimpleNamespace(get=lambda: True),
            SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(
                get=lambda: False
            ),
        )
        local_metadata = [object(), object()]
        layer = SimpleNamespace(layer_id=3, v_head_dim=1, tp_q_head_num=4)
        forward_batch = SimpleNamespace(forward_mode=FakeForwardMode())
        q = torch.zeros(1, 4, 1)
        kv = torch.zeros(1, 1)
        attn_sink = torch.zeros(4)

        with patch.object(
            redknot_mla_backend, "DSV4AttnMetadata", FakeDSV4AttnMetadata
        ), patch.object(
            redknot_mla_backend,
            "DeepSeekV4TokenToKVPool",
            FakeTokenToKVPool,
        ), patch.object(
            redknot_mla_backend,
            "_create_flashmla_metadata",
            side_effect=local_metadata,
        ) as create_local_metadata, patch.object(
            redknot_mla_backend, "envs", fake_envs
        ), patch.dict(
            sys.modules, {"flash_mla": fake_flash_mla}
        ), patch.dict(
            os.environ, {"REDKNOT_C4_TOPK_CLAMP": "5"}
        ):
            output = backend.forward(
                q=q,
                k=kv,
                v=kv,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=4,
                save_kv_cache=True,
                attn_sink=attn_sink,
            )

        # Heads 0/1 retain their own 64/128-token local results. Head 2's
        # unexecutable 256-token window is promoted, and head 3 is configured
        # global, so both retain the global output.
        self.assertTrue(
            torch.equal(
                output[0, :, 0],
                torch.tensor([64.0, 128.0, 1000.0, 1000.0]),
            )
        )
        self.assertEqual(fake_flash_mla.flash_mla_with_kvcache.call_count, 3)
        self.assertEqual(create_local_metadata.call_count, 2)
        self.assertIs(flash_calls[0]["tile_scheduler_metadata"], global_metadata)
        self.assertIs(flash_calls[1]["tile_scheduler_metadata"], local_metadata[0])
        self.assertIs(flash_calls[2]["tile_scheduler_metadata"], local_metadata[1])
        self.assertIsNotNone(flash_calls[0]["extra_k_cache"])
        self.assertEqual(flash_calls[0]["extra_topk_length"].tolist(), [5])
        self.assertIsNone(flash_calls[1]["extra_k_cache"])
        self.assertIsNone(flash_calls[2]["extra_k_cache"])
        self.assertTrue(
            torch.equal(
                flash_calls[1]["topk_length"],
                torch.tensor([64], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                flash_calls[2]["topk_length"],
                torch.tensor([128], dtype=torch.int32),
            )
        )

        backend.store_cache.assert_called_once_with(3, kv, forward_batch)
        backend._maybe_redknot_reuse_hook.assert_called_once_with(
            3, forward_batch, token_pool
        )
        token_pool.get_extra_key_buffer.assert_called_once_with(3)
        core_metadata.get_flashmla_metadata.assert_called_once_with(4)
        self.assertEqual(
            backend.runtime_counters(),
            {
                "forwards": 1,
                "path.dual_mixed": 1,
                "global_flashmla_calls": 1,
                "local_flashmla_calls": 2,
                "policy_local_head_outputs": 2,
                "policy_global_head_outputs": 2,
                "policy_promoted_head_outputs": 1,
            },
        )


class TestMLAHeadwiseForwardContract(CustomTestCase):
    def test_padded_flashmla_h64_slices_merged_rows_before_padding(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend.head_dim_v = 512
        backend.softmax_scale = 512**-0.5
        backend._redknot_runtime_counters = Counter()
        backend._redknot_mla_off_global_q_workspace = None

        calls = []

        def fake_flash_mla_with_kvcache(**kwargs):
            calls.append(
                {
                    "q": kwargs["q"].clone(),
                    "indices": kwargs["indices"].clone(),
                    "topk_length": kwargs["topk_length"].clone(),
                    "extra_indices": kwargs[
                        "extra_indices_in_kvcache"
                    ].clone(),
                    "extra_lengths": kwargs["extra_topk_length"].clone(),
                    "metadata": kwargs["tile_scheduler_metadata"],
                }
            )
            padded_q = kwargs["q"]
            output = padded_q.new_zeros((padded_q.shape[0], 1, 64, 512))
            output[:, :, :1, :].copy_(padded_q[:, :, :1, :])
            return output, torch.zeros((padded_q.shape[0], 64))

        q = torch.randn(5, 1, 1, 512, dtype=torch.bfloat16)
        swa_indices = torch.arange(5, dtype=torch.int32).reshape(5, 1, 1)
        swa_lengths = torch.arange(1, 6, dtype=torch.int32)
        extra_indices = (swa_indices + 10).clone()
        extra_lengths = (swa_lengths + 10).clone()
        metadata = (object(), object(), object())
        fake_flash = SimpleNamespace(
            flash_mla_with_kvcache=fake_flash_mla_with_kvcache
        )

        with patch.dict(sys.modules, {"flash_mla": fake_flash}), patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), patch.object(
            redknot_mla_backend, "FLASHMLA_MAX_BATCH_ROWS", 2
        ), patch.object(
            redknot_mla_backend,
            "_create_flashmla_metadata",
            side_effect=metadata,
        ):
            output = backend._mla_off_forward_global_padded_flashmla_h64(
                layer_id=3,
                q_part=q,
                sink_part=torch.tensor([0.25]),
                swa_k_cache=torch.zeros(1, 128),
                swa_page_indices=swa_indices,
                swa_topk_lengths=swa_lengths,
                extra_k_cache=torch.zeros(1, 64),
                extra_indices=extra_indices,
                extra_topk_lengths=extra_lengths,
            )

        self.assertTrue(torch.equal(output, q))
        self.assertEqual([call["q"].shape[0] for call in calls], [2, 2, 1])
        self.assertEqual([call["metadata"] for call in calls], list(metadata))
        self.assertEqual(
            torch.cat([call["indices"] for call in calls], dim=0).tolist(),
            swa_indices.tolist(),
        )
        self.assertEqual(
            torch.cat([call["topk_length"] for call in calls]).tolist(),
            swa_lengths.tolist(),
        )
        self.assertEqual(
            torch.cat([call["extra_indices"] for call in calls], dim=0).tolist(),
            extra_indices.tolist(),
        )
        self.assertEqual(
            torch.cat([call["extra_lengths"] for call in calls]).tolist(),
            extra_lengths.tolist(),
        )
        self.assertEqual(
            tuple(backend._redknot_mla_off_global_q_workspace.shape),
            (2, 1, 64, 512),
        )
        counters = backend.runtime_counters()
        self.assertEqual(counters["global_padded_flashmla_h64_calls"], 1)
        self.assertEqual(counters["global_padded_flashmla_h64_row_chunks"], 3)
        self.assertEqual(
            counters["global_padded_flashmla_physical_head_rows"], 5 * 64
        )

    def test_padded_flashmla_h64_reuses_workspace_but_refreshes_metadata(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend.head_dim_v = 512
        backend.softmax_scale = 512**-0.5
        backend._redknot_runtime_counters = Counter()
        backend._redknot_mla_off_global_q_workspace = None

        calls = []

        def fake_flash_mla_with_kvcache(**kwargs):
            calls.append(kwargs)
            padded_q = kwargs["q"]
            output = padded_q.new_zeros((padded_q.shape[0], 1, 64, 512))
            output[:, :, :1, :].copy_(padded_q[:, :, :1, :])
            lse = torch.zeros(
                (padded_q.shape[0], 64), dtype=torch.float32
            )
            return output, lse

        fake_flash = SimpleNamespace(
            flash_mla_with_kvcache=fake_flash_mla_with_kvcache
        )
        q = torch.randn(3, 1, 1, 512, dtype=torch.bfloat16)
        sink = torch.tensor([0.25], dtype=torch.float32)
        swa_cache = torch.zeros(1, 128)
        swa_indices = torch.zeros(3, 1, 128, dtype=torch.int32)
        swa_lengths = torch.full((3,), 128, dtype=torch.int32)
        extra_cache = torch.zeros(1, 64)
        extra_indices = torch.zeros(3, 1, 64, dtype=torch.int32)
        extra_lengths = torch.full((3,), 64, dtype=torch.int32)
        metadata = (object(), object(), object())

        with patch.dict(sys.modules, {"flash_mla": fake_flash}), patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), patch.object(
            redknot_mla_backend,
            "_create_flashmla_metadata",
            side_effect=metadata,
        ) as create_metadata:
            first = backend._mla_off_forward_global_padded_flashmla_h64(
                layer_id=3,
                q_part=q,
                sink_part=sink,
                swa_k_cache=swa_cache,
                swa_page_indices=swa_indices,
                swa_topk_lengths=swa_lengths,
                extra_k_cache=extra_cache,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_lengths,
            )
            workspace_pointer = (
                backend._redknot_mla_off_global_q_workspace.data_ptr()
            )
            second_q = torch.randn_like(q)
            second = backend._mla_off_forward_global_padded_flashmla_h64(
                layer_id=3,
                q_part=second_q,
                sink_part=sink,
                swa_k_cache=swa_cache,
                swa_page_indices=swa_indices,
                swa_topk_lengths=swa_lengths,
                extra_k_cache=extra_cache,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_lengths,
            )
            third = backend._mla_off_forward_global_padded_flashmla_h64(
                layer_id=4,
                q_part=second_q,
                sink_part=sink,
                swa_k_cache=swa_cache,
                swa_page_indices=swa_indices,
                swa_topk_lengths=swa_lengths,
                extra_k_cache=extra_cache,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_lengths,
            )

        self.assertTrue(torch.equal(first, q))
        self.assertTrue(torch.equal(second, second_q))
        self.assertTrue(torch.equal(third, second_q))
        self.assertEqual(
            backend._redknot_mla_off_global_q_workspace.data_ptr(),
            workspace_pointer,
        )
        self.assertEqual(create_metadata.call_count, 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(tuple(calls[0]["q"].shape), (3, 1, 64, 512))
        self.assertTrue(torch.count_nonzero(calls[0]["q"][:, :, 1:, :]) == 0)
        self.assertEqual(calls[0]["attn_sink"].tolist(), [0.25] + [0.0] * 63)
        self.assertIs(calls[0]["tile_scheduler_metadata"], metadata[0])
        self.assertIs(calls[1]["tile_scheduler_metadata"], metadata[1])
        self.assertIs(calls[2]["tile_scheduler_metadata"], metadata[2])
        self.assertEqual(
            backend.runtime_counters()["global_padded_flashmla_h64_calls"],
            3,
        )

    def test_uniform_window_fuses_arbitrary_owned_scope_masks(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend.redknot_mla_head_cfg = _head_config(
            [HEAD_GLOBAL] + [HEAD_LOCAL] * 7,
            [-1] + [128] * 7,
        )
        backend._redknot_swa_capacity = 128
        backend._redknot_runtime_counters = Counter()
        backend._redknot_logged_paths = set()
        backend._redknot_trace_actual_passes = False
        backend.head_dim_v = 1
        backend.softmax_scale = 1.0

        q = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
        attn_sink = torch.arange(8, dtype=torch.float32)
        swa_cache = torch.zeros(1, 128)
        swa_indices = torch.arange(128, dtype=torch.int32).reshape(1, 128)
        swa_lengths = torch.tensor([100], dtype=torch.int32)
        extra_cache = torch.zeros(1, 64)
        extra_indices = torch.arange(64, dtype=torch.int32).reshape(1, 64)
        extra_lengths = torch.tensor([11], dtype=torch.int32)
        owned = (tuple(range(8)), tuple(range(8)))
        calls = []

        def fake_triton_fp8_attention_fwd(**kwargs):
            calls.append(kwargs)
            output = kwargs["q"][..., :1].clone()
            return output, torch.zeros(output.shape[:-1])

        fake_triton_module = SimpleNamespace(
            triton_fp8_attention_fwd=fake_triton_fp8_attention_fwd
        )

        def run(plan, **mla_off_kwargs):
            calls.clear()
            if int(mla_off_kwargs.get("mla_off_reused_row_count", 0)) > 0:
                reuse_mask = mla_off_kwargs["mla_off_reuse_mask"]
                dirty_rows_cpu = mla_off_kwargs[
                    "mla_off_online_local_rows_cpu"
                ]
                reuse_mask_digest = (17, 29)
                layout_key = (
                    id(reuse_mask),
                    (7, 11),
                    "positions",
                    "tokens",
                    None,
                )
                layout_certificate = (
                    redknot_mla_backend._MLAOffRestoreLayoutCertificate(
                        layout_key=layout_key,
                        certified_layer_ids=(0,),
                        restore_rows=(),
                        reusable_cpu=reuse_mask,
                        reuse_mask_digest=reuse_mask_digest,
                        reused_count=int(
                            mla_off_kwargs["mla_off_reused_row_count"]
                        ),
                        dirty_rows_cpu=dirty_rows_cpu,
                        segments_by_hash={},
                        reusable_identity=(
                            redknot_mla_backend._mla_off_control_tensor_identity(
                                reuse_mask
                            )
                        ),
                        dirty_identity=(
                            redknot_mla_backend._mla_off_control_tensor_identity(
                                dirty_rows_cpu
                            )
                        ),
                    )
                )
                controller = DSV4MLAOffController(
                    max_cache_bytes=1024, max_device_cache_bytes=0
                )
                rows_certificate = controller.prepare_device_indices(
                    dirty_rows_cpu,
                    device=q.device,
                    role="online_local_rows",
                    semantic_digest=reuse_mask_digest,
                    upper_bound=q.shape[0],
                )
                mla_off_kwargs.update(
                    mla_off_online_local_rows=(
                        controller.device_indices_from_certificate(
                            rows_certificate,
                            cpu_indices=dirty_rows_cpu,
                            device=q.device,
                            role="online_local_rows",
                            semantic_digest=reuse_mask_digest,
                            upper_bound=q.shape[0],
                        )
                    ),
                    mla_off_online_local_rows_certificate=rows_certificate,
                    mla_off_restore_layout_certificate=layout_certificate,
                    mla_off_controller=controller,
                    mla_off_reuse_mask_digest=reuse_mask_digest,
                )
            with patch.dict(
                sys.modules,
                {"sglang.srt.layers.attention.nsa.triton_decode": fake_triton_module},
            ):
                output = backend._forward_headwise(
                    q=q,
                    attn_sink=attn_sink,
                    plan=plan,
                    layer_id=0,
                    owned_view=owned,
                    swa_k_cache=swa_cache,
                    swa_page_indices=swa_indices,
                    swa_topk_lengths=swa_lengths,
                    extra_k_cache=extra_cache,
                    extra_indices=extra_indices,
                    extra_topk_lengths=extra_lengths,
                    **mla_off_kwargs,
                )
            self.assertEqual(tuple(output.shape), (1, 8, 1))
            self.assertEqual(len(calls), 1)
            return calls[0]

        mixed = redknot_mla_backend._DualLayerPassPlan(
            local_groups=((128, tuple(range(1, 8))),),
            global_heads=(0,),
            promoted_heads=(),
        )
        call = run(mixed)
        self.assertEqual(call["extra_head_mask"].tolist(), [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(
            call["main_head_lengths"].tolist(),
            [128, 128, 128, 128, 128, 128, 128, 128],
        )
        self.assertIs(call["k_cache"], swa_cache)
        self.assertIs(call["extra_k_cache"], extra_cache)

        all_global = redknot_mla_backend._DualLayerPassPlan(
            local_groups=(),
            global_heads=tuple(range(8)),
            promoted_heads=(),
        )
        call = run(all_global)
        self.assertEqual(call["extra_head_mask"].tolist(), [1] * 8)
        self.assertEqual(call["main_head_lengths"].tolist(), [128] * 8)

        all_local = redknot_mla_backend._DualLayerPassPlan(
            local_groups=((128, tuple(range(8))),),
            global_heads=(),
            promoted_heads=(),
        )
        call = run(all_local)
        self.assertIsNone(call["extra_k_cache"])
        self.assertEqual(call["topk_length"].tolist(), [100])
        self.assertNotIn("extra_head_mask", call)
        self.assertNotIn("main_head_lengths", call)

        # A caller cannot make an all-local restore skip attention rows merely
        # by supplying a mask/index certificate. Production ``forward`` sets
        # this internal gate only after the runtime context revalidates its
        # opaque compact proof immediately before entering this function.
        with self.assertRaisesRegex(
            ValueError, "all-local attention skip requires compact preflight"
        ):
            run(
                all_local,
                mla_off_reuse_mask=torch.tensor([True]),
                mla_off_online_local_rows=torch.empty(0, dtype=torch.long),
                mla_off_online_local_rows_cpu=torch.empty(0, dtype=torch.long),
                mla_off_reused_row_count=1,
                mla_off_online_local_row_count=0,
            )

        # Reuse eligibility and attention scope are separate in the
        # accuracy-first mode.  A mixed local/global policy with no restored
        # rows must execute all owned heads together using the full native DSV4
        # candidate scope, exactly like the all-global headwise oracle.
        backend._redknot_reuse_heads_full_scope = True
        call = run(mixed)
        self.assertEqual(tuple(call["q"].shape), (1, 1, 8, 1))
        self.assertIs(call["extra_k_cache"], extra_cache)
        self.assertEqual(call["topk_length"].tolist(), [100])
        self.assertNotIn("extra_head_mask", call)
        self.assertNotIn("main_head_lengths", call)
        backend._redknot_reuse_heads_full_scope = False

        with patch.dict(
            os.environ, {"REDKNOT_MLA_OFF_METRICS": "1"}
        ), patch.object(redknot_mla_backend.logger, "info") as log_info:
            call = run(
                mixed,
                mla_off_reuse_mask=torch.tensor([True]),
                mla_off_online_local_rows=torch.empty(0, dtype=torch.long),
                mla_off_online_local_rows_cpu=torch.empty(0, dtype=torch.long),
                mla_off_reused_row_count=1,
                mla_off_online_local_row_count=0,
                mla_off_benchmark_request_id="request-scope-test",
                mla_off_benchmark_forward_id="forward-scope-test",
                mla_off_benchmark_forward_mode="extend",
                mla_off_benchmark_q_rows=1,
            )
        self.assertIsNotNone(call["extra_k_cache"])
        metric_calls = [
            item
            for item in log_info.call_args_list
            if item.args and str(item.args[0]).startswith("REDKNOT_MLA_OFF_METRIC")
        ]
        self.assertEqual(len(metric_calls), 1)
        self.assertEqual(metric_calls[0].args[1], "request-scope-test")
        self.assertEqual(
            metric_calls[0].args[2:],
            ("forward-scope-test", "extend", 1, 0, 7, 0, 1, "full"),
        )

        deferred_rows = {}
        with patch.dict(
            os.environ, {"REDKNOT_MLA_OFF_METRICS": "1"}
        ), patch.object(redknot_mla_backend.logger, "info") as deferred_log:
            run(
                mixed,
                mla_off_reuse_mask=torch.tensor([True]),
                mla_off_online_local_rows=torch.empty(0, dtype=torch.long),
                mla_off_online_local_rows_cpu=torch.empty(0, dtype=torch.long),
                mla_off_reused_row_count=1,
                mla_off_online_local_row_count=0,
                mla_off_benchmark_request_id="request-scope-test",
                mla_off_benchmark_forward_id="forward-scope-test",
                mla_off_benchmark_forward_mode="extend",
                mla_off_benchmark_q_rows=1,
                mla_off_runtime_rows_out=deferred_rows,
            )
        self.assertEqual(
            deferred_rows,
            {
                "reused_local_head_rows": 7,
                "online_local_head_rows": 0,
                "online_global_head_rows": 1,
            },
        )
        self.assertFalse(
            any(
                item.args
                and str(item.args[0]).startswith("REDKNOT_MLA_OFF_METRIC")
                for item in deferred_log.call_args_list
            )
        )

    def test_full_scope_reuse_recomputes_dirty_local_rows_with_extra_kv(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend.redknot_mla_head_cfg = _head_config(
            [HEAD_GLOBAL] + [HEAD_LOCAL] * 7,
            [-1] + [128] * 7,
        )
        backend._redknot_runtime_counters = Counter()
        backend._redknot_logged_paths = set()
        backend._redknot_trace_actual_passes = False
        backend._redknot_reuse_heads_full_scope = True
        backend._redknot_mla_off_strict_row_verify = True
        backend.head_dim_v = 1
        backend.softmax_scale = 1.0

        q = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8, 1)
        swa_cache = torch.zeros(1, 128)
        swa_indices = torch.arange(256, dtype=torch.int32).reshape(2, 128)
        swa_lengths = torch.tensor([100, 96], dtype=torch.int32)
        extra_cache = torch.zeros(1, 64)
        extra_indices = torch.arange(128, dtype=torch.int32).reshape(2, 64)
        extra_lengths = torch.tensor([11, 9], dtype=torch.int32)
        calls = []

        def fake_triton_fp8_attention_fwd(**kwargs):
            calls.append(kwargs)
            output = kwargs["q"][..., :1].clone()
            return output, torch.zeros(output.shape[:-1])

        fake_triton_module = SimpleNamespace(
            triton_fp8_attention_fwd=fake_triton_fp8_attention_fwd
        )
        reuse_mask = torch.tensor([True, False])
        dirty_rows_cpu = torch.tensor([1], dtype=torch.long)
        reuse_mask_digest = (17, 29)
        layout_key = (123, (7, 11), "positions", "tokens", None)
        layout_certificate = (
            redknot_mla_backend._MLAOffRestoreLayoutCertificate(
                layout_key=layout_key,
                certified_layer_ids=(0,),
                restore_rows=(),
                reusable_cpu=reuse_mask,
                reuse_mask_digest=reuse_mask_digest,
                reused_count=1,
                dirty_rows_cpu=dirty_rows_cpu,
                segments_by_hash={},
                reusable_identity=(
                    redknot_mla_backend._mla_off_control_tensor_identity(
                        reuse_mask
                    )
                ),
                dirty_identity=(
                    redknot_mla_backend._mla_off_control_tensor_identity(
                        dirty_rows_cpu
                    )
                ),
            )
        )
        controller = DSV4MLAOffController(
            max_cache_bytes=1024, max_device_cache_bytes=0
        )
        dirty_rows_certificate = controller.prepare_device_indices(
            dirty_rows_cpu,
            device=q.device,
            role="online_local_rows",
            semantic_digest=reuse_mask_digest,
            upper_bound=q.shape[0],
        )
        dirty_rows_device = controller.device_indices_from_certificate(
            dirty_rows_certificate,
            cpu_indices=dirty_rows_cpu,
            device=q.device,
            role="online_local_rows",
            semantic_digest=reuse_mask_digest,
            upper_bound=q.shape[0],
        )
        runtime_rows = {}
        with patch.dict(
            sys.modules,
            {"sglang.srt.layers.attention.nsa.triton_decode": fake_triton_module},
        ):
            output = backend._forward_headwise(
                q=q,
                attn_sink=torch.arange(8, dtype=torch.float32),
                plan=redknot_mla_backend._DualLayerPassPlan(
                    local_groups=(
                        (64, (1, 2, 3)),
                        (128, (4, 5, 6, 7)),
                    ),
                    global_heads=(0,),
                    promoted_heads=(),
                ),
                layer_id=0,
                owned_view=(tuple(range(8)), tuple(range(8))),
                swa_k_cache=swa_cache,
                swa_page_indices=swa_indices,
                swa_topk_lengths=swa_lengths,
                extra_k_cache=extra_cache,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_lengths,
                mla_off_reuse_mask=reuse_mask,
                mla_off_online_local_rows=dirty_rows_device,
                mla_off_online_local_rows_cpu=dirty_rows_cpu,
                mla_off_online_local_rows_certificate=(
                    dirty_rows_certificate
                ),
                mla_off_restore_layout_certificate=layout_certificate,
                mla_off_controller=controller,
                mla_off_reuse_mask_digest=reuse_mask_digest,
                mla_off_reused_row_count=1,
                mla_off_online_local_row_count=1,
                mla_off_runtime_rows_out=runtime_rows,
            )

        self.assertEqual(tuple(output.shape), (2, 8, 1))
        self.assertEqual(len(calls), 2)
        global_call, dirty_local_call = calls
        self.assertEqual(tuple(global_call["q"].shape), (2, 1, 1, 1))
        self.assertEqual(tuple(dirty_local_call["q"].shape), (1, 1, 7, 1))
        self.assertIs(global_call["extra_k_cache"], extra_cache)
        self.assertIs(dirty_local_call["extra_k_cache"], extra_cache)
        self.assertEqual(global_call["topk_length"].tolist(), [100, 96])
        self.assertEqual(dirty_local_call["topk_length"].tolist(), [96])
        self.assertEqual(dirty_local_call["extra_topk_length"].tolist(), [9])
        self.assertEqual(
            runtime_rows,
            {
                "reused_local_head_rows": 7,
                "online_local_head_rows": 7,
                "online_global_head_rows": 2,
            },
        )

    def test_tp_owned_heads_share_one_kv_and_keep_per_head_scope(self):
        class FakeDSV4AttnMetadata:
            pass

        class FakeTokenToKVPool:
            pass

        class FakeForwardMode:
            def is_idle(self):
                return False

            def is_draft_extend(self, include_v2=False):
                return False

        # TP rank 1/2 owns logical heads 4..7.  The other four Q slots are
        # intentionally present because DSV4 pads Q to its global 64-head view.
        head_cfg = _head_config(
            [
                HEAD_GLOBAL,
                HEAD_GLOBAL,
                HEAD_GLOBAL,
                HEAD_GLOBAL,
                HEAD_LOCAL,
                HEAD_GLOBAL,
                HEAD_LOCAL,
                HEAD_GLOBAL,
            ],
            [-1, -1, -1, -1, 64, -1, 128, -1],
            layer_id=3,
        )
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend.redknot_mla_pass_mode = "headwise"
        backend.redknot_mla_head_cfg = head_cfg
        backend._redknot_swa_capacity = 128
        backend._redknot_dual_layer_plans = (None, None, None) + (
            _build_dual_layer_pass_plan(head_cfg, layer_id=3, swa_capacity=128),
        )
        backend._redknot_tp_rank = 1
        backend._redknot_tp_size = 2
        backend._redknot_runtime_counters = Counter()
        backend._redknot_logged_paths = set()
        backend._redknot_trace_actual_passes = False
        backend.mtp_enabled = False
        backend.head_dim_v = 1
        backend.softmax_scale = 1.0
        backend._maybe_upgrade_forward_metadata = MagicMock()
        backend.store_cache = MagicMock()
        backend._maybe_redknot_reuse_hook = MagicMock()

        core_metadata = FakeDSV4AttnMetadata()
        core_metadata.swa_page_indices = torch.arange(128, dtype=torch.int32).reshape(
            1, 128
        )
        core_metadata.swa_topk_lengths = torch.tensor([128], dtype=torch.int32)
        core_metadata.c4_sparse_page_indices = torch.arange(
            64, dtype=torch.int32
        ).reshape(1, 64)
        core_metadata.c4_sparse_topk_lengths = torch.tensor([7], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(core_attn_metadata=core_metadata)

        shared_swa = torch.zeros(1, 128)
        token_pool = FakeTokenToKVPool()
        token_pool.swa_window_size = 128
        token_pool.page_size = 256
        token_pool.swa_kv_pool = SimpleNamespace(kv_cache_total_dim=1)
        token_pool.get_swa_key_buffer_radix = MagicMock(return_value=shared_swa)
        token_pool.get_extra_key_buffer = MagicMock(return_value=torch.zeros(1, 64))
        backend.token_to_kv_pool = token_pool

        triton_calls = []

        def fake_triton_fp8_attention_fwd(**kwargs):
            triton_calls.append(kwargs)
            if "extra_head_mask" in kwargs:
                mask = kwargs["extra_head_mask"].bool().reshape(1, 1, -1, 1)
                local_base = kwargs["main_head_lengths"].reshape(1, 1, -1, 1)
                base = torch.where(mask, 1000.0, local_base.float())
            else:
                base = (
                    1000.0
                    if kwargs["extra_k_cache"] is not None
                    else float(kwargs["topk_length"].max().item())
                )
            # Preserve the selected Q value so the assertion proves logical
            # head identity, not merely the number of heads in each group.
            output = kwargs["q"][..., :1] + base
            return output, torch.zeros(output.shape[:-1])

        fake_triton_module = SimpleNamespace(
            triton_fp8_attention_fwd=MagicMock(
                side_effect=fake_triton_fp8_attention_fwd
            )
        )
        fake_envs = SimpleNamespace(
            SGLANG_REDKNOT_OFFLINE_REUSE=SimpleNamespace(get=lambda: True),
            SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(
                get=lambda: False
            ),
        )
        layer = SimpleNamespace(layer_id=3, v_head_dim=1, tp_q_head_num=4)
        forward_batch = SimpleNamespace(forward_mode=FakeForwardMode())
        q = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1)
        kv = torch.zeros(1, 1)
        attn_sink = torch.arange(8, dtype=torch.float32)

        with patch.object(
            redknot_mla_backend, "DSV4AttnMetadata", FakeDSV4AttnMetadata
        ), patch.object(
            redknot_mla_backend,
            "DeepSeekV4TokenToKVPool",
            FakeTokenToKVPool,
        ), patch.object(
            redknot_mla_backend, "envs", fake_envs
        ), patch.dict(
            sys.modules,
            {"sglang.srt.layers.attention.nsa.triton_decode": (fake_triton_module)},
        ):
            output = backend.forward(
                q=q,
                k=kv,
                v=kv,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=4,
                save_kv_cache=True,
                attn_sink=attn_sink,
            )

        self.assertTrue(
            torch.equal(
                output[0, :, 0],
                torch.tensor([0.0, 0.0, 0.0, 0.0, 69.0, 1006.0, 135.0, 1008.0]),
            )
        )
        self.assertEqual([call["q"].shape[2] for call in triton_calls], [4])
        self.assertEqual(
            [call["attn_sink"].tolist() for call in triton_calls],
            [[4.0, 5.0, 6.0, 7.0]],
        )
        self.assertIsNotNone(triton_calls[0]["extra_k_cache"])
        self.assertEqual(triton_calls[0]["extra_head_mask"].tolist(), [0, 1, 0, 1])
        self.assertEqual(
            triton_calls[0]["main_head_lengths"].tolist(), [64, 128, 128, 128]
        )
        self.assertEqual(
            {call["k_cache"].data_ptr() for call in triton_calls},
            {shared_swa.data_ptr()},
        )
        backend.store_cache.assert_called_once_with(3, kv, forward_batch)
        backend._maybe_redknot_reuse_hook.assert_called_once_with(
            3, forward_batch, token_pool
        )
        self.assertEqual(
            backend.runtime_counters(),
            {
                "forwards": 1,
                "path.headwise_mixed": 1,
                "fused_scope_triton_calls": 1,
                "policy_local_head_outputs": 2,
                "policy_global_head_outputs": 2,
                "headwise_owned_q_heads": 4,
            },
        )

        # With full-scope reuse enabled and no restore context, the policy is
        # eligibility metadata only: production must keep native FlashMLA and
        # must not enter the arbitrary-head Triton path.
        native_output = torch.full((1, 8, 1), 42.0)
        backend._redknot_reuse_heads_full_scope = True
        backend._native_forward = MagicMock(return_value=native_output)
        triton_call_count = len(triton_calls)
        with patch.object(
            redknot_mla_backend, "DSV4AttnMetadata", FakeDSV4AttnMetadata
        ), patch.object(
            redknot_mla_backend,
            "DeepSeekV4TokenToKVPool",
            FakeTokenToKVPool,
        ):
            full_scope_output = backend.forward(
                q=q,
                k=kv,
                v=kv,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=4,
                save_kv_cache=True,
                attn_sink=attn_sink,
            )
        self.assertIs(full_scope_output, native_output)
        self.assertEqual(len(triton_calls), triton_call_count)
        backend._native_forward.assert_called_once()
        self.assertEqual(
            backend.runtime_counters()["path.headwise_reuse_full_scope_native"],
            1,
        )

        snapshot_context = SimpleNamespace(
            is_restore=False,
            is_full_local=False,
            backend_applied=False,
        )
        backend._mla_off_vote_count = MagicMock(return_value=2)
        backend._native_forward.reset_mock()
        with patch.object(
            redknot_mla_backend, "DSV4AttnMetadata", FakeDSV4AttnMetadata
        ), patch.object(
            redknot_mla_backend,
            "DeepSeekV4TokenToKVPool",
            FakeTokenToKVPool,
        ):
            snapshot_output = backend.forward(
                q=q,
                k=kv,
                v=kv,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=4,
                save_kv_cache=False,
                attn_sink=attn_sink,
                mla_off_context=snapshot_context,
            )
        self.assertIs(snapshot_output, native_output)
        self.assertTrue(snapshot_context.backend_applied)
        backend._native_forward.assert_called_once()
        backend._mla_off_vote_count.assert_called_once_with(True, q.device)
        self.assertEqual(len(triton_calls), triton_call_count)

        full_local_context = SimpleNamespace(
            is_restore=False,
            is_full_local=True,
            backend_applied=False,
            local_head_axes=(0, 1),
            spec=SimpleNamespace(owned_logical_heads=(0, 1, 2, 3)),
            benchmark_q_rows=1,
            benchmark_request_id="query-suffix",
            benchmark_forward_id="query-suffix-forward",
            benchmark_forward_mode="extend",
        )
        backend._mla_off_record_runtime_rows = MagicMock()
        backend._native_forward.reset_mock()
        backend._mla_off_vote_count.reset_mock(return_value=True)
        backend._mla_off_vote_count.return_value = 2
        with patch.object(
            redknot_mla_backend, "DSV4AttnMetadata", FakeDSV4AttnMetadata
        ), patch.object(
            redknot_mla_backend,
            "DeepSeekV4TokenToKVPool",
            FakeTokenToKVPool,
        ):
            suffix_output = backend.forward(
                q=q,
                k=kv,
                v=kv,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=4,
                save_kv_cache=False,
                attn_sink=attn_sink,
                mla_off_context=full_local_context,
            )
        self.assertIs(suffix_output, native_output)
        self.assertTrue(full_local_context.backend_applied)
        backend._native_forward.assert_called_once()
        backend._mla_off_record_runtime_rows.assert_called_once()
        self.assertEqual(len(triton_calls), triton_call_count)

        restore_mask = torch.tensor([True])
        restore_dirty_cpu = torch.empty(0, dtype=torch.long)
        restore_mask_digest = (17, 29)
        restore_layout = redknot_mla_backend._MLAOffRestoreLayoutCertificate(
            layout_key=(123, (7, 11), "positions", "tokens", None),
            certified_layer_ids=(3,),
            restore_rows=(),
            reusable_cpu=restore_mask,
            reuse_mask_digest=restore_mask_digest,
            reused_count=1,
            dirty_rows_cpu=restore_dirty_cpu,
            segments_by_hash={},
            reusable_identity=(
                redknot_mla_backend._mla_off_control_tensor_identity(
                    restore_mask
                )
            ),
            dirty_identity=(
                redknot_mla_backend._mla_off_control_tensor_identity(
                    restore_dirty_cpu
                )
            ),
        )
        restore_controller = DSV4MLAOffController(
            max_cache_bytes=1024, max_device_cache_bytes=0
        )
        restore_rows_certificate = restore_controller.prepare_device_indices(
            restore_dirty_cpu,
            device=q.device,
            role="online_local_rows",
            semantic_digest=restore_mask_digest,
            upper_bound=1,
        )
        restore_context = SimpleNamespace(
            is_restore=True,
            is_full_local=False,
            layer_id=3,
            reuse_mask=restore_mask,
            reuse_mask_digest=restore_mask_digest,
            online_local_row_indices=(
                restore_controller.device_indices_from_certificate(
                    restore_rows_certificate,
                    cpu_indices=restore_dirty_cpu,
                    device=q.device,
                    role="online_local_rows",
                    semantic_digest=restore_mask_digest,
                    upper_bound=1,
                )
            ),
            online_local_row_indices_cpu=restore_dirty_cpu,
            online_local_row_indices_certificate=restore_rows_certificate,
            restore_layout_certificate=restore_layout,
            controller=restore_controller,
            reused_row_count=1,
            online_local_row_count=0,
            benchmark_request_id="top-level-restore",
            benchmark_forward_id="f0",
            benchmark_forward_mode="extend",
            benchmark_q_rows=1,
            backend_applied=False,
        )
        forward_batch._redknot_mla_off_restore_layout = restore_layout
        backend._native_forward.reset_mock()
        backend._mla_off_vote_count.reset_mock()
        backend._mla_off_record_runtime_rows = MagicMock()
        with patch.object(
            redknot_mla_backend, "DSV4AttnMetadata", FakeDSV4AttnMetadata
        ), patch.object(
            redknot_mla_backend,
            "DeepSeekV4TokenToKVPool",
            FakeTokenToKVPool,
        ), patch.object(
            redknot_mla_backend, "envs", fake_envs
        ), patch.dict(
            sys.modules,
            {"sglang.srt.layers.attention.nsa.triton_decode": fake_triton_module},
        ):
            backend.forward(
                q=q,
                k=kv,
                v=kv,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=4,
                save_kv_cache=False,
                attn_sink=attn_sink,
                mla_off_context=restore_context,
            )
        backend._native_forward.assert_not_called()
        self.assertTrue(restore_context.backend_applied)
        self.assertEqual(
            backend._mla_off_vote_count.call_args_list,
            [
                call(True, q.device),
                call(False, q.device),
                # The safe default keeps compact wo_a disabled unless the
                # experiment opts in explicitly.
                call(False, q.device),
                call(True, q.device),
            ],
        )
        self.assertGreater(len(triton_calls), triton_call_count)

        # A rank-local attention exception must still enter the same fixed
        # attention_application vote with False; peers therefore abort instead
        # of waiting forever at a success-only applied-count collective.
        restore_context.backend_applied = False
        backend._mla_off_vote_count = MagicMock(
            side_effect=[2, 0, 2, 1]
        )
        with patch.object(
            redknot_mla_backend, "DSV4AttnMetadata", FakeDSV4AttnMetadata
        ), patch.object(
            redknot_mla_backend,
            "DeepSeekV4TokenToKVPool",
            FakeTokenToKVPool,
        ), patch.object(
            redknot_mla_backend, "envs", fake_envs
        ), patch.object(
            backend,
            "_forward_headwise",
            side_effect=RuntimeError("injected attention failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "attention_application failed"
            ):
                backend.forward(
                    q=q,
                    k=kv,
                    v=kv,
                    layer=layer,
                    forward_batch=forward_batch,
                    compress_ratio=4,
                    save_kv_cache=False,
                    attn_sink=attn_sink,
                    mla_off_context=restore_context,
                )
        self.assertEqual(
            backend._mla_off_vote_count.call_args_list[-1],
            call(False, q.device),
        )
        self.assertFalse(restore_context.backend_applied)


class TestMLAHeadLocalityCollector(CustomTestCase):
    def test_chunked_request_uses_complete_k_and_ignores_short_warmup(self):
        tokens = 12
        chunk_size = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = MLAHeadProfileConfig(
            num_layers=1,
            num_heads=2,
            expected_context=tokens,
            coverage=0.95,
            sample_queries=4,
            global_window_ratio=0.5,
            window_safety=1.0,
            window_round_to=1,
            window_min=1,
            dense_prefix_layers=0,
        )
        collector = MLAHeadLocalityCollector(cfg)
        baseline = MLAHeadLocalityCollector(
            MLAHeadProfileConfig(
                num_layers=1,
                num_heads=2,
                coverage=cfg.coverage,
                sample_queries=cfg.sample_queries,
                global_window_ratio=cfg.global_window_ratio,
                window_safety=cfg.window_safety,
                window_round_to=cfg.window_round_to,
                window_min=cfg.window_min,
                dense_prefix_layers=cfg.dense_prefix_layers,
            )
        )

        latent_k = torch.eye(tokens, device=device)
        q = torch.zeros(tokens, 2, tokens, device=device)
        q[:, 0, :] = 20.0 * torch.eye(tokens, device=device)
        self.assertTrue(baseline.observe_layer(0, q, latent_k, 1.0))

        # A short engine warmup begins at a real request boundary, but cannot
        # seal or contribute any observations because it never reaches the
        # configured context length.
        self.assertFalse(
            collector.observe_layer(
                0,
                q[:2],
                latent_k[:2],
                1.0,
                prefix_len=0,
                extend_len=2,
                seq_len=2,
            )
        )

        completed = False
        for prefix in range(0, tokens, chunk_size):
            end = min(tokens, prefix + chunk_size)
            # SGLang may pad the token dimension. Extreme sentinels verify the
            # profiler uses extend_len's real CPU count and cannot accidentally
            # include padding in either Q sampling or the reconstructed K.
            q_chunk = torch.cat(
                (q[prefix:end], torch.full((2, 2, tokens), 1e4, device=device))
            )
            k_chunk = torch.cat(
                (latent_k[prefix:end], torch.full((2, tokens), -1e4, device=device))
            )
            completed = collector.observe_layer(
                0,
                q_chunk,
                k_chunk,
                1.0,
                prefix_len=prefix,
                extend_len=end - prefix,
                seq_len=end,
            )
            self.assertEqual(completed, end == tokens)

        head_cfg = collector.build_head_config()
        self.assertEqual(collector.max_context, tokens)
        self.assertEqual(head_cfg.head_class[0][0], HEAD_LOCAL)
        self.assertEqual(head_cfg.head_class[0][1], HEAD_GLOBAL)
        evidence = collector.distance_profile()
        self.assertEqual(evidence["query_rows_per_layer"], [4.0])
        self.assertEqual(evidence["sampled_query_positions"], [7, 8, 9, 11])
        self.assertGreaterEqual(evidence["coverage_windows"]["d95"][0][1], 8)
        baseline_evidence = baseline.distance_profile()
        self.assertEqual(
            evidence["coverage_windows"], baseline_evidence["coverage_windows"]
        )
        self.assertEqual(
            evidence["per_query_coverage_window_quantiles"],
            baseline_evidence["per_query_coverage_window_quantiles"],
        )
        self.assertTrue(
            torch.allclose(
                collector._g_mass,
                baseline._g_mass,
                rtol=0.0,
                atol=0.0,
            )
        )
        self.assertTrue(
            torch.equal(collector._g_window_counts, baseline._g_window_counts)
        )
        self.assertTrue(torch.equal(collector._g_sum, baseline._g_sum))
        self.assertTrue(torch.equal(collector._g_sq, baseline._g_sq))

    def test_chunked_completion_requires_same_generation_on_every_layer(self):
        tokens = 8
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        collector = MLAHeadLocalityCollector(
            MLAHeadProfileConfig(
                num_layers=2,
                num_heads=2,
                expected_context=tokens,
                coverage=0.95,
                sample_queries=4,
                dense_prefix_layers=0,
            )
        )
        latent_k = torch.eye(tokens, device=device)
        q = torch.zeros(tokens, 2, tokens, device=device)
        q[:, 0, :] = 20.0 * latent_k

        # Incomplete generation one on both layers.
        for layer in range(2):
            self.assertFalse(
                collector.observe_layer(
                    layer,
                    q[:2],
                    latent_k[:2],
                    1.0,
                    prefix_len=0,
                    extend_len=2,
                    seq_len=2,
                )
            )

        completed = []
        for prefix in (0, 4):
            for layer in range(2):
                completed.append(
                    collector.observe_layer(
                        layer,
                        q[prefix : prefix + 4],
                        latent_k[prefix : prefix + 4],
                        1.0,
                        prefix_len=prefix,
                        extend_len=4,
                        seq_len=prefix + 4,
                    )
                )
        self.assertEqual(completed, [False, False, False, True])
        self.assertEqual(collector.max_context, tokens)

    def test_real_softmax_distance_mass_separates_local_and_global_heads(self):
        tokens = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = MLAHeadProfileConfig(
            num_layers=1,
            num_heads=2,
            coverage=0.95,
            sample_queries=tokens,
            global_window_ratio=0.5,
            window_safety=1.0,
            window_round_to=1,
            window_min=1,
            dense_prefix_layers=0,
        )
        collector = MLAHeadLocalityCollector(cfg)

        # The shared latent key is an identity basis.  Head 0's query selects
        # the current key almost exclusively (distance 0); head 1 has uniform
        # causal attention and therefore needs roughly the whole context.
        latent_k = torch.eye(tokens, device=device)
        q = torch.zeros(tokens, 2, tokens, device=device)
        q[:, 0, :] = 20.0 * torch.eye(tokens, device=device)
        collector.observe_layer(0, q, latent_k, softmax_scale=1.0)

        head_cfg = collector.build_head_config()
        self.assertEqual(head_cfg.head_class[0][0], HEAD_LOCAL)
        self.assertEqual(head_cfg.head_max_distance[0][0], 1)
        self.assertEqual(head_cfg.head_class[0][1], HEAD_GLOBAL)

        evidence = collector.distance_profile()
        self.assertEqual(evidence["coverage_windows"]["d95"][0][0], 0)
        self.assertGreaterEqual(evidence["coverage_windows"]["d95"][0][1], 32)
        self.assertEqual(
            evidence["per_query_coverage_window_quantiles"]["p90"][0][0], 0
        )
        self.assertGreaterEqual(
            evidence["per_query_coverage_window_quantiles"]["p90"][0][1], 32
        )
        self.assertEqual(evidence["sampled_query_positions"][-1], tokens - 1)
        self.assertGreater(evidence["query_rows_per_layer"][0], 0)

    def test_profile_distance_is_converted_to_token_window(self):
        tokens = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = MLAHeadProfileConfig(
            num_layers=1,
            num_heads=1,
            coverage=0.95,
            sample_queries=tokens,
            global_window_ratio=2.0,
            window_safety=1.0,
            window_round_to=1,
            window_min=1,
            dense_prefix_layers=0,
        )
        collector = MLAHeadLocalityCollector(cfg)

        # Every meaningful sampled query attends to exactly the key at causal
        # distance four. A max distance of four requires five retained tokens:
        # distances 0, 1, 2, 3, and 4.
        latent_k = torch.eye(tokens, device=device)
        q = torch.zeros(tokens, 1, tokens, device=device)
        rows = torch.arange(tokens, device=device)
        q[rows, 0, (rows - 4).clamp_min(0)] = 20.0
        collector.observe_layer(0, q, latent_k, softmax_scale=1.0)

        head_cfg = collector.build_head_config()
        self.assertEqual(head_cfg.head_class[0][0], HEAD_LOCAL)
        self.assertEqual(head_cfg.head_max_distance[0][0], 5)
        self.assertEqual(
            collector.distance_profile()["coverage_windows"]["d95"][0][0], 4
        )


class TestMLAOffForwardGeneration(CustomTestCase):
    def test_generation_invalidates_forward_batch_caches_and_keys_inference_tensor(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_mla_off_strict_row_verify = False
        forward_batch = SimpleNamespace(
            _redknot_forward_generation_id=(123, 1),
            _redknot_mla_off_disabled=True,
            _redknot_mla_off_positions_cpu=("stale",),
            _redknot_mla_off_restore_layout=object(),
            _redknot_mla_off_verified_tokens={"stale"},
            _redknot_rank_local_q_authorizations={3: True},
        )
        self.assertTrue(
            backend._mla_off_prepare_forward_generation(forward_batch)
        )
        self.assertFalse(hasattr(forward_batch, "_redknot_mla_off_disabled"))
        self.assertFalse(
            hasattr(forward_batch, "_redknot_mla_off_positions_cpu")
        )
        self.assertFalse(
            hasattr(forward_batch, "_redknot_mla_off_restore_layout")
        )
        self.assertFalse(
            hasattr(forward_batch, "_redknot_rank_local_q_authorizations")
        )

        with torch.inference_mode():
            tensor = torch.arange(4, dtype=torch.long)
            first_key = backend._mla_off_forward_tensor_cache_key(
                forward_batch, tensor
            )
            forward_batch._redknot_mla_off_positions_cpu = ("current",)
            self.assertTrue(
                backend._mla_off_prepare_forward_generation(forward_batch)
            )
            self.assertTrue(
                hasattr(forward_batch, "_redknot_mla_off_positions_cpu")
            )
            tensor.add_(10)
            forward_batch._redknot_forward_generation_id = (123, 2)
            self.assertTrue(
                backend._mla_off_prepare_forward_generation(forward_batch)
            )
            second_key = backend._mla_off_forward_tensor_cache_key(
                forward_batch, tensor
            )

        self.assertNotEqual(first_key[0], second_key[0])
        self.assertEqual(first_key[1][0], second_key[1][0])
        self.assertFalse(
            hasattr(forward_batch, "_redknot_mla_off_positions_cpu")
        )

    def test_missing_generation_uses_strict_inference_tensor_path(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_mla_off_strict_row_verify = False
        with torch.inference_mode():
            tensor = torch.arange(2, dtype=torch.long)
            self.assertIsNone(
                backend._mla_off_forward_tensor_cache_key(
                    SimpleNamespace(), tensor
                )
            )

    def test_rank_local_q_requires_forward_scoped_restore_authorization(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend.redknot_mla_pass_mode = "headwise"
        mode = SimpleNamespace(is_draft_extend=lambda include_v2=True: False)
        batch = SimpleNamespace(forward_mode=mode)
        self.assertFalse(backend.accepts_rank_local_q(layer_id=3, forward_batch=batch))
        batch._redknot_rank_local_q_authorizations = {3: False}
        self.assertFalse(backend.accepts_rank_local_q(layer_id=3, forward_batch=batch))
        batch._redknot_rank_local_q_authorizations[3] = True
        self.assertTrue(backend.accepts_rank_local_q(layer_id=3, forward_batch=batch))
        self.assertFalse(backend.accepts_rank_local_q(layer_id=4, forward_batch=batch))


class TestContextBoundWiringOrder(CustomTestCase):
    def test_restore_observer_runs_before_geometry_and_artifact_pins(self):
        source = inspect.getsource(
            dsv4_reuse_backend_runtime._prepare_composite_restore_context_impl
        )
        observer = source.index("observe_restore_chunk")
        geometry = source.index("_build_forward_composite_geometry", observer)
        pins = source.index("_prepare_shared_request_states", geometry)
        self.assertLess(observer, geometry)
        self.assertLess(geometry, pins)

    def test_snapshot_prefix_returns_online_before_capture_staging(self):
        source = inspect.getsource(
            redknot_mla_backend.RedKnotMLAAttnBackend.prepare_mla_off_context
        )
        observer = source.index("observe_snapshot_chunk")
        prefix_gate = source.index('if snapshot_phase == "prefix"', observer)
        generation = source.index("_mla_off_snapshot_generation_id", prefix_gate)
        self.assertLess(observer, prefix_gate)
        self.assertLess(prefix_gate, generation)

    def test_failed_snapshot_context_resolution_poisons_before_rollback(self):
        source = inspect.getsource(
            redknot_mla_backend.RedKnotMLAAttnBackend.prepare_mla_off_context
        )
        snapshot = source.index('if mode == "snapshot"')
        failure_gate = source.index("if local_context is None:", snapshot)
        poison = source.index(
            "_mla_off_poison_context_snapshot", failure_gate
        )
        rollback = source.index(
            "_mla_off_rollback_snapshot_staging", failure_gate
        )
        self.assertLess(poison, rollback)


class TestMLAOffTransferAuditEmission(CustomTestCase):
    @staticmethod
    def _stats(artifact_bytes, artifact_calls):
        return {
            "online_artifact_h2d_bytes": artifact_bytes,
            "online_artifact_h2d_calls": artifact_calls,
            "online_index_h2d_bytes": 0,
            "online_total_h2d_bytes": artifact_bytes,
            "device_cache_enabled": 0,
            "reserved_device_bytes": 0,
            "allocated_device_bytes": 0,
            "max_device_cache_bytes": 0,
        }

    def test_forward_deltas_do_not_accumulate_and_emit_once_on_last_layer(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_mla_off_rank_local_layer_ids = (2, 3)
        backend._redknot_tp_rank = 1
        backend._redknot_tp_size = 2
        backend._redknot_runtime_counters = Counter()
        controller = MagicMock()
        controller.snapshot_stats.side_effect = [
            self._stats(10, 1),
            self._stats(18, 2),
            self._stats(18, 2),
            self._stats(22, 3),
        ]
        forward_batch = SimpleNamespace()

        def run_forward(request_id, forward_id):
            state = backend._mla_off_begin_transfer_audit(
                forward_batch=forward_batch,
                controller=controller,
                layer_id=2,
                request_id=request_id,
                forward_id=forward_id,
                forward_mode="extend",
                q_rows=4,
            )
            context = SimpleNamespace(
                transfer_audit_state=state,
                controller=controller,
                benchmark_request_id=request_id,
                benchmark_forward_id=forward_id,
                benchmark_forward_mode="extend",
                benchmark_q_rows=4,
            )
            backend._mla_off_maybe_emit_transfer_audit(
                layer_id=2, context=context
            )
            backend._mla_off_maybe_emit_transfer_audit(
                layer_id=3, context=context
            )
            return context

        with patch.dict(os.environ, {"REDKNOT_MLA_OFF_METRICS": "1"}), patch(
            "builtins.print"
        ) as output:
            first = run_forward("request", "f1")
            second = run_forward("request", "f2")
            self.assertEqual(output.call_count, 2)
            first_payload = json.loads(
                output.call_args_list[0].args[0].split(" ", 1)[1]
            )
            second_payload = json.loads(
                output.call_args_list[1].args[0].split(" ", 1)[1]
            )
            self.assertEqual(
                first_payload["counter_delta"]["online_artifact_h2d_bytes"], 8
            )
            self.assertEqual(
                second_payload["counter_delta"]["online_artifact_h2d_bytes"], 4
            )
            self.assertEqual(first_payload["tp_rank"], 1)
            self.assertEqual(first_payload["tp_size"], 2)
            backend._mla_off_maybe_emit_transfer_audit(
                layer_id=3, context=second
            )
            self.assertTrue(first.transfer_audit_state["emitted"])
            self.assertEqual(
                backend.runtime_counters()[
                    "mla_off.transfer_audit_publish_failures"
                ],
                1,
            )

    def test_forward_wide_composite_audit_binds_all_layers_and_restore_counts(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_mla_off_rank_local_layer_ids = tuple(range(3, 40))
        backend._redknot_tp_rank = 0
        backend._redknot_tp_size = 8
        backend._redknot_runtime_counters = Counter()
        controller = MagicMock()
        controller.snapshot_stats.side_effect = [
            self._stats(10, 1),
            self._stats(18, 2),
        ]
        forward_batch = SimpleNamespace()
        contexts = [
            SimpleNamespace(
                controller=controller,
                benchmark_request_id="request",
                benchmark_forward_id="forward",
                benchmark_forward_mode="extend",
                benchmark_q_rows=8192,
            )
            for _ in range(37)
        ]
        with patch.dict(
            os.environ, {"REDKNOT_MLA_OFF_METRICS": "1"}
        ), patch(
            "sglang.srt.layers.attention.redknot.dsv4_mla_offload."
            "get_dsv4_mla_off_controller",
            return_value=controller,
        ), patch("builtins.print") as output:
            state = backend._mla_off_begin_composite_transfer_audit(
                forward_batch=forward_batch,
                layer_id=3,
            )
            for context in contexts:
                backend._mla_off_bind_composite_transfer_audit(state, context)
            backend._count("mla_off.shared_device_restore_calls", 1)
            backend._count("mla_off.shared_device_restore_operations", 147)
            backend._count("mla_off.shared_device_values_restored", 4096)
            backend._mla_off_maybe_emit_transfer_audit(
                layer_id=3, context=contexts[0]
            )
            backend._mla_off_maybe_emit_transfer_audit(
                layer_id=39, context=contexts[-1]
            )

        self.assertTrue(all(context.transfer_audit_state is state for context in contexts))
        output.assert_called_once()
        payload = json.loads(output.call_args.args[0].split(" ", 1)[1])
        self.assertEqual(
            payload["shared_restore"]["counter_delta"],
            {
                "shared_device_restore_calls": 1,
                "shared_device_restore_operations": 147,
                "shared_device_values_restored": 4096,
            },
        )
        self.assertTrue(state["emitted"])

    def test_single_rank_transfer_print_failure_is_observational(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_mla_off_rank_local_layer_ids = (2, 3)
        backend._redknot_tp_rank = 0
        backend._redknot_tp_size = 2
        backend._redknot_runtime_counters = Counter()
        controller = MagicMock()
        controller.snapshot_stats.side_effect = [
            self._stats(10, 1),
            self._stats(18, 2),
        ]
        forward_batch = SimpleNamespace()
        with patch.dict(
            os.environ, {"REDKNOT_MLA_OFF_METRICS": "1"}
        ):
            state = backend._mla_off_begin_transfer_audit(
                forward_batch=forward_batch,
                controller=controller,
                layer_id=2,
                request_id="request",
                forward_id="f1",
                forward_mode="extend",
                q_rows=4,
            )
            context = SimpleNamespace(
                transfer_audit_state=state,
                controller=controller,
                benchmark_request_id="request",
                benchmark_forward_id="f1",
                benchmark_forward_mode="extend",
                benchmark_q_rows=4,
            )
        with patch("builtins.print", side_effect=RuntimeError("broken stdout")):
            backend._mla_off_maybe_emit_transfer_audit(
                layer_id=3, context=context
            )

        self.assertFalse(state["emitted"])
        self.assertEqual(
            backend.runtime_counters()[
                "mla_off.transfer_audit_publish_failures"
            ],
            1,
        )

    def test_single_rank_runtime_metric_logger_failure_is_observational(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_runtime_counters = Counter()
        with patch.dict(
            os.environ, {"REDKNOT_MLA_OFF_METRICS": "1"}
        ), patch.object(
            redknot_mla_backend.logger,
            "info",
            side_effect=RuntimeError("broken logger"),
        ):
            backend._mla_off_record_runtime_rows(
                request_id="request",
                forward_id="f1",
                forward_mode="extend",
                q_rows=4,
                layer_id=3,
                reused_local_head_rows=24,
                online_local_head_rows=8,
                online_global_head_rows=0,
            )

        counters = backend.runtime_counters()
        self.assertEqual(counters["mla_off.reused_local_head_rows"], 24)
        self.assertEqual(
            counters["mla_off.runtime_metric_publish_failures"], 1
        )


class TestMLAOffCompositeForwardManifest(CustomTestCase):
    @staticmethod
    def _context(*position_groups):
        requests = tuple(
            SimpleNamespace(logical_positions=tuple(positions))
            for positions in position_groups
        )
        return SimpleNamespace(
            benchmark_request_id="request",
            benchmark_forward_id="forward",
            benchmark_forward_mode="extend",
            benchmark_q_rows=len(position_groups[0]),
            diagnostic_ablation="full",
            _redknot_composite_forward_resources=SimpleNamespace(
                geometry=SimpleNamespace(requests=requests)
            ),
        )

    def test_single_request_manifest_preserves_exact_position_span(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._mla_off_log_forward_start = MagicMock()

        backend._mla_off_log_composite_forward_manifest(
            self._context((8192, 8193, 8194))
        )

        backend._mla_off_log_forward_start.assert_called_once_with(
            request_id="request",
            forward_id="forward",
            forward_mode="extend",
            q_rows=3,
            position_start=8192,
            position_end=8195,
            position_contiguous=True,
            plan_mode="restore",
            diagnostic_ablation="full",
        )

    def test_multi_request_geometry_does_not_forge_one_manifest(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._mla_off_log_forward_start = MagicMock()

        backend._mla_off_log_composite_forward_manifest(
            self._context((0, 1), (0, 1))
        )

        backend._mla_off_log_forward_start.assert_not_called()

    def test_forward_transaction_full_local_emits_every_layer_status(self):
        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        backend._redknot_runtime_counters = Counter()
        backend._mla_off_log_request_status = MagicMock()
        contexts = {
            layer_id: SimpleNamespace(
                is_full_local=True,
                intentional_full_local_reason="query_suffix_only",
                benchmark_forward_id="forward",
                benchmark_forward_mode="extend",
                benchmark_q_rows=49,
            )
            for layer_id in range(3, 40)
        }
        transaction = SimpleNamespace(
            context_for=lambda layer_id: contexts[int(layer_id)]
        )
        plan = {
            "mode": "restore",
            "reuse_mla_off": True,
            "benchmark_request_id": "request",
        }

        backend._mla_off_log_forward_transaction_full_local_statuses(
            transaction=transaction,
            active_plans=(plan,),
            expected_layers=tuple(range(3, 40)),
        )

        self.assertEqual(backend._mla_off_log_request_status.call_count, 37)
        self.assertEqual(
            [call.kwargs["layer_id"] for call in backend._mla_off_log_request_status.call_args_list],
            list(range(3, 40)),
        )
        for status_call in backend._mla_off_log_request_status.call_args_list:
            self.assertEqual(status_call.kwargs["status"], "full_local")
            self.assertEqual(status_call.kwargs["reason"], "query_suffix_only")
            self.assertEqual(status_call.kwargs["forward_id"], "forward")
            self.assertEqual(status_call.kwargs["q_rows"], 49)
        self.assertEqual(
            backend.runtime_counters()["mla_off.intentional_full_local_layers"],
            37,
        )


class TestQualificationTerminalStateBatch(CustomTestCase):
    def test_real_layer_ratio_layout_copies_exactly_55_slots_once(self):
        from sglang.srt.layers.attention.redknot.dsv4_shared_latent_gpu import (
            DOMAIN_C128_ATTENTION_STATE,
            DOMAIN_C4_ATTENTION_STATE,
            DOMAIN_INDEXER_STATE,
        )

        positions = tuple(range(8))
        prepared_layers = {}
        expected_slot_values = []
        next_slot = 1000
        for layer_id in range(3, 40):
            # DeepSeek-V4 target ratios are layer3=C128, layer4=C4, ...
            ratio = 128 if layer_id % 2 else 4
            required_domains = (
                (DOMAIN_C128_ATTENTION_STATE,)
                if ratio == 128
                else (DOMAIN_C4_ATTENTION_STATE, DOMAIN_INDEXER_STATE)
            )
            arena = tuple(7 for _ in required_domains)
            operations = tuple(
                SimpleNamespace(
                    domain=domain,
                    output_rows=SimpleNamespace(begin=index, end=index + 1),
                )
                for index, domain in enumerate(required_domains)
            )

            def operations_for_layer(requested, *, current=layer_id, ops=operations):
                return ops if int(requested) == int(current) else ()

            schedule = SimpleNamespace(
                positions=positions,
                index_arena=arena,
                digest="sha256:" + f"{layer_id:064x}",
                operations_for_layer=operations_for_layer,
            )
            target_slots = {}
            for domain in required_domains:
                slot_value = next_slot
                next_slot += 1
                expected_slot_values.append(slot_value)
                target_slots[(domain, layer_id)] = torch.full(
                    (len(positions),), slot_value, dtype=torch.long
                )
            prepared_layers[layer_id] = SimpleNamespace(
                restore_adapter=SimpleNamespace(
                    compress_ratio=ratio,
                    target_slots=target_slots,
                ),
                validated_restores=(
                    SimpleNamespace(
                        request_index=0,
                        validated=SimpleNamespace(
                            prepared=SimpleNamespace(schedule=schedule)
                        ),
                    ),
                ),
            )

        backend = object.__new__(redknot_mla_backend.RedKnotMLAAttnBackend)
        geometry = SimpleNamespace(benchmark_request_id="request")
        request = SimpleNamespace(
            row_count=len(positions),
            logical_positions=positions,
            request_token="request",
        )
        continuation = (
            geometry,
            request,
            0,
            len(positions),
            ("request", 7, 100, "sha256:" + "1" * 64),
        )
        resources = SimpleNamespace(expected_layer_ids=tuple(range(3, 40)))
        transaction = SimpleNamespace(
            restore_batch_receipt=object(),
            prepared_layers=prepared_layers,
        )
        with patch.object(
            backend,
            "_redknot_qualification_continuation_geometry",
            return_value=continuation,
        ), patch.object(
            redknot_mla_backend.torch, "stack", wraps=torch.stack
        ) as stack:
            proof = backend._redknot_prepare_qualification_prefix_receipt(
                resources=resources,
                forward_batch=SimpleNamespace(),
                transaction=transaction,
            )

        self.assertEqual(stack.call_count, 1)
        self.assertEqual(len(stack.call_args.args[0]), 55)
        materialization = proof[-1]
        self.assertEqual(len(materialization), 37)
        self.assertEqual(
            sum(len(layer_proof[3]) for layer_proof in materialization), 55
        )
        observed_slot_values = [
            domain_proof[2]
            for layer_proof in materialization
            for domain_proof in layer_proof[3]
        ]
        self.assertEqual(observed_slot_values, expected_slot_values)
        self.assertEqual(len(set(observed_slot_values)), 55)


class TestPureOfficialPromptManifest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_redknot_benchmark_module()

    class _Tokenizer:
        source_identity = {}

        def __call__(self, _text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [index % 997 for index in range(65633)]}

        @staticmethod
        def decode(input_ids, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(map(str, input_ids[:4]))

    def test_pure_request_schema_rejects_every_legacy_field(self):
        chunk_tokens = int(self.benchmark.IH_CHUNK_TOKENS)
        chunks = tuple(
            tuple(range(index * chunk_tokens, (index + 1) * chunk_tokens))
            for index in range(int(self.benchmark.IH_NUM_CHUNKS))
        )
        contracts = self.benchmark._ih_build_context_segment_contracts(
            chunks,
            model_compat_hash=_CONTEXT_MODEL_HASH,
            head_policy_hash=_CONTEXT_POLICY_HASH,
        )
        _, snapshot = self.benchmark._ih_build_context_snapshot_request(
            chunks,
            contracts,
            index=0,
            model_compat_hash=_CONTEXT_MODEL_HASH,
            head_policy_hash=_CONTEXT_POLICY_HASH,
        )
        composed_prefix = tuple(
            token for chunk in chunks for token in chunk
        )
        restore = self.benchmark._ih_build_context_restore_plan(
            composed_prefix,
            (1, 2, 3),
            contracts,
            model_compat_hash=_CONTEXT_MODEL_HASH,
            head_policy_hash=_CONTEXT_POLICY_HASH,
        )
        self.benchmark._ih_validate_exact_pure_plan(snapshot, mode="snapshot")
        snapshot["reuse_window_kv"] = False
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.benchmark._ih_validate_exact_pure_plan(
                snapshot, mode="snapshot"
            )

        self.benchmark._ih_validate_exact_pure_plan(restore, mode="restore")
        restore["selection_policy"] = "checkpoint_islands"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.benchmark._ih_validate_exact_pure_plan(
                restore, mode="restore"
            )

        config = {
            key: "legacy"
            for key in self.benchmark._IH_PURE_LEGACY_CONFIG_KEYS
        }
        config["reuse_strategy"] = "pure_mla_offline_online_headsplit_merge"
        self.benchmark._ih_finalize_pure_result_config(
            config, offline_chunk_order=range(8)
        )
        self.assertFalse(
            self.benchmark._IH_PURE_LEGACY_CONFIG_KEYS.intersection(config)
        )
        self.assertEqual(config["offline_chunk_order"], list(range(8)))

    def test_prompt_manifest_publication_never_overwrites(self):
        manifest = {"format": "redknot_pure_mla_prompt_v1", "value": 1}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "prompt.json"
            self.benchmark._ih_write_data_manifest(destination, manifest)
            original = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                self.benchmark._ih_write_data_manifest(
                    destination, {"format": "tampered", "value": 2}
                )
            self.assertEqual(destination.read_bytes(), original)

    def test_official_prompt_is_frozen_before_split(self):
        chunks = [
            [chunk_index + 1] * 8192 for chunk_index in range(8)
        ]
        queries = [("question", [41, 42], ["answer"], 2)]
        selection = {
            "format": self.benchmark._IH_DATA_MANIFEST_FORMAT,
            "selection_sha256": "selection-digest",
            "dataset": {
                "name": "musique",
                "bytes": 123,
                "sha256": "dataset-digest",
                "row_id_base": 0,
            },
            "selection": {"queries": [{"row_id": 68}]},
        }
        tokenizer = self._Tokenizer()
        tokenizer.source_identity = {
            "path": str(
                (Path(self.benchmark.MODEL_PATH) / "tokenizer.json").resolve()
            ),
            "sha256": "sha256:" + "b" * 64,
        }
        with patch.object(
            self.benchmark, "_IHFastTokenizer", self._Tokenizer
        ), patch.object(
            self.benchmark, "IH_PROMPT_MANIFEST", ""
        ), patch.object(
            self.benchmark, "IH_PROMPT_MANIFEST_OUT", "/tmp/prompt.json"
        ), patch.object(
            self.benchmark, "IH_EXPECTED_DATA_SELECTION_SHA256", "selection-digest"
        ), patch.object(
            self.benchmark, "IH_EXPECTED_QUERY_ROW_ID", 68
        ), patch.object(
            self.benchmark, "IH_EXPECTED_QUERY_ROW_IDS", (68,)
        ), patch.object(
            self.benchmark,
            "_OFFICIAL_ENCODER_IDENTITY",
            {"path": "/model/encoding_dsv4.py", "sha256": "sha256:" + "a" * 64},
        ), patch.object(
            self.benchmark,
            "_encode_pure_official_rag_prompt",
            return_value="official-prompt",
        ), patch.object(
            self.benchmark, "_ih_sha256_file", return_value="a" * 64
        ), patch.object(
            Path, "is_file", return_value=True
        ), patch.object(
            self.benchmark, "_ih_write_data_manifest"
        ) as publish:
            prompt_chunks, prompt_queries, manifest = (
                self.benchmark._ih_build_official_pure_prompt(
                    tokenizer,
                    chunks,
                    queries,
                    selection,
                    chunk_tokens=8192,
                )
            )

        self.assertEqual([len(chunk) for chunk in prompt_chunks], [8192] * 8)
        self.assertEqual(len(prompt_queries[0][1]), 97)
        self.assertEqual(prompt_queries[0][3], 2)
        self.assertEqual(manifest["geometry"]["total_tokens"], 65633)
        self.assertEqual(manifest["geometry"]["online_suffix_tokens"], 97)
        self.assertNotIn("query_score_by_source_chunk", manifest["source"])
        self.assertEqual(
            manifest["prompt"]["full_input_ids_sha256"],
            self.benchmark._ih_chunk_hash(
                [index % 997 for index in range(65633)]
            ),
        )
        publish.assert_called_once_with("/tmp/prompt.json", manifest)
        with patch.object(
            self.benchmark,
            "IH_EXPECTED_PROMPT_TEXT_SHA256",
            manifest["prompt"]["text_sha256"],
        ), patch.object(
            self.benchmark,
            "IH_EXPECTED_FULL_INPUT_IDS_SHA256",
            manifest["prompt"]["full_input_ids_sha256"],
        ), patch.object(
            self.benchmark, "IH_EXPECTED_FULL_INPUT_TOKENS", 65633
        ):
            self.benchmark._ih_validate_official_prompt_run_identity(
                manifest, 65633
            )
            for invalid_cap in (65632, 65634):
                with self.assertRaisesRegex(ValueError, "qualification cap"):
                    self.benchmark._ih_validate_official_prompt_run_identity(
                        manifest, invalid_cap
                    )

    def test_musique_row68_one_pass_golden(self):
        fixture = Path(__file__).with_name(
            "musique_pure_prompt_selection_v1.json"
        )
        tokenizer_json = Path(self.benchmark.MODEL_PATH) / "tokenizer.json"
        encoder = (
            Path(self.benchmark.MODEL_PATH)
            / "encoding"
            / "encoding_dsv4.py"
        )
        dataset_dir = Path(self.benchmark.LONGBENCH_DIR)
        if not (
            fixture.is_file()
            and tokenizer_json.is_file()
            and encoder.is_file()
            and (dataset_dir / "musique.jsonl").is_file()
        ):
            self.skipTest("cluster-real MuSiQue prompt fixtures are unavailable")

        tokenizer = self.benchmark._ih_load_tokenizer()
        chunks, queries, selection = self.benchmark._ih_load(
            tokenizer,
            8192,
            8,
            1,
            row_offset=0,
            manifest_path=str(fixture),
            exclude_manifest_paths=(),
            return_manifest=True,
            dataset_name="musique",
            dataset_dir=str(dataset_dir),
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.benchmark,
            "IH_EXPECTED_DATA_SELECTION_SHA256",
            "586fd683bfe043e1a6aaa1d07c7236ea9d956d99be739be743c4a2ec1728bcd8",
        ), patch.object(
            self.benchmark, "IH_EXPECTED_QUERY_ROW_ID", 68
        ), patch.object(
            self.benchmark, "IH_EXPECTED_QUERY_ROW_IDS", (68,)
        ), patch.object(
            self.benchmark, "IH_PROMPT_MANIFEST", ""
        ), patch.object(
            self.benchmark,
            "IH_PROMPT_MANIFEST_OUT",
            str(Path(directory) / "prompt.json"),
        ), patch.object(
            self.benchmark,
            "_ih_query_chunk_weights",
            side_effect=AssertionError("pure prompt must not score chunks"),
        ):
            prompt_chunks, prompt_queries, manifest = (
                self.benchmark._ih_build_official_pure_prompt(
                    tokenizer,
                    chunks,
                    queries,
                    selection,
                    chunk_tokens=8192,
                )
            )

        full_ids = [
            token for chunk in prompt_chunks for token in chunk
        ] + list(prompt_queries[0][1])
        self.assertEqual(len(full_ids), 65585)
        self.assertEqual([len(chunk) for chunk in prompt_chunks], [8192] * 8)
        self.assertEqual(len(prompt_queries[0][1]), 49)
        self.assertEqual(full_ids[:2], [0, 128803])
        self.assertEqual(full_ids[-2:], [128804, 128822])
        self.assertEqual(
            manifest["prompt"]["text_sha256"],
            "sha256:fa33caccb16d22f9df544239de3229c74bf6ce6847148ddeccbdbde371db11c8",
        )
        self.assertEqual(
            manifest["prompt"]["full_input_ids_sha256"],
            "sha256:9329590a5c2bb87e7689d5d8b81edbadf50394a89f268df97268debd82bea891",
        )
        self.assertEqual(
            manifest["prompt"]["offline_prefix_hash"],
            "sha256:dfeca9a8db5bdab9a80a7ed5450ebbaf083f9dca0d80d5a03d6ebe7bfad1aeec",
        )
        self.assertEqual(
            manifest["prompt"]["offline_chunk_hashes"],
            [
                "sha256:f8890146a40a5c3979a217c828147be120d25f92e65cb08847e0a1b3b2153374",
                "sha256:d74acac1eeb8f147efaff9dda0a897fc4b132d0d97d02dd73178599f46fd0ad5",
                "sha256:f1b820e3d1ec926439779820d66d6ab72f62ce01730cf410fb4517273de2ab14",
                "sha256:cdf0deebc1c77400ef7e4068c684844d2457edddc601bf61501a76f721e88b93",
                "sha256:60f8a98edc90d1bc05caa1007b55b76a896d3bc3616c5a6581a2e4bd095ba933",
                "sha256:2f15b07cd14fcd9ca3fec5c78b354b92afa8c2c1eee10ae90032f52701e46cf8",
                "sha256:c3d887caf5195ff14be6ef6814b74fbceedb3afb997e1612ba6281f873308bb9",
                "sha256:ef7f82887e447d13e0680794ac4c2f5e7f5078de18ca02c7ad1c7618b4c68657",
            ],
        )
        self.assertEqual(
            manifest["prompt"]["online_suffix_hash"],
            "sha256:e564a68cea703929b18249d9988553550fc8e20bd12be928a511dc1edde036b3",
        )
        self.assertEqual(
            manifest["protocol"]["encoder_sha256"],
            "sha256:abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6",
        )
        self.assertEqual(
            manifest["protocol"]["tokenizer_sha256"],
            "sha256:8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
        )
        self.assertEqual(
            manifest["protocol"]["tokenizer_config_sha256"],
            "sha256:6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
        )

    def test_musique_row68_128k_one_pass_golden(self):
        fixture = Path(__file__).with_name(
            "musique_pure_prompt_selection_128k_v1.json"
        )
        tokenizer_json = Path(self.benchmark.MODEL_PATH) / "tokenizer.json"
        encoder = (
            Path(self.benchmark.MODEL_PATH)
            / "encoding"
            / "encoding_dsv4.py"
        )
        dataset_dir = Path(self.benchmark.LONGBENCH_DIR)
        if not (
            fixture.is_file()
            and tokenizer_json.is_file()
            and encoder.is_file()
            and (dataset_dir / "musique.jsonl").is_file()
        ):
            self.skipTest("cluster-real 128K MuSiQue prompt fixtures are unavailable")

        tokenizer = self.benchmark._ih_load_tokenizer()
        chunks, queries, selection = self.benchmark._ih_load(
            tokenizer,
            8192,
            16,
            1,
            row_offset=0,
            manifest_path=str(fixture),
            exclude_manifest_paths=(),
            return_manifest=True,
            dataset_name="musique",
            dataset_dir=str(dataset_dir),
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.benchmark,
            "IH_EXPECTED_DATA_SELECTION_SHA256",
            "caf99890880e0de190f845d0a38e600d760d2153cd1961888bd7776a2044f040",
        ), patch.object(
            self.benchmark, "IH_EXPECTED_QUERY_ROW_ID", 68
        ), patch.object(
            self.benchmark, "IH_EXPECTED_QUERY_ROW_IDS", (68,)
        ), patch.object(
            self.benchmark, "IH_PROMPT_MANIFEST", ""
        ), patch.object(
            self.benchmark,
            "IH_PROMPT_MANIFEST_OUT",
            str(Path(directory) / "prompt.json"),
        ), patch.object(
            self.benchmark,
            "_ih_query_chunk_weights",
            side_effect=AssertionError("pure prompt must not score chunks"),
        ):
            prompt_chunks, prompt_queries, manifest = (
                self.benchmark._ih_build_official_pure_prompt(
                    tokenizer,
                    chunks,
                    queries,
                    selection,
                    chunk_tokens=8192,
                )
            )

        full_ids = [
            token for chunk in prompt_chunks for token in chunk
        ] + list(prompt_queries[0][1])
        self.assertEqual(len(full_ids), 131128)
        self.assertEqual([len(chunk) for chunk in prompt_chunks], [8192] * 16)
        self.assertEqual(len(prompt_queries[0][1]), 56)
        self.assertEqual(full_ids[:2], [0, 128803])
        self.assertEqual(full_ids[-2:], [128804, 128822])
        self.assertEqual(
            manifest["prompt"]["text_sha256"],
            "sha256:9959bc0f32f7eb29a4cf61e7d7a20ca8fda937166057510f49ae74056576f4b1",
        )
        self.assertEqual(
            manifest["prompt"]["full_input_ids_sha256"],
            "sha256:3b1ee37110db315a9ba84a3ae55adfce61b2aaa61520fc4c68511313cf96dd87",
        )
        self.assertEqual(
            manifest["prompt"]["offline_prefix_hash"],
            "sha256:d86610b99677ff52011067ff3b6f5a435d3152c20b61dc5d023f274c2fcc1aa1",
        )
        self.assertEqual(
            manifest["prompt"]["offline_chunk_hashes"],
            [
                "sha256:f8890146a40a5c3979a217c828147be120d25f92e65cb08847e0a1b3b2153374",
                "sha256:d74acac1eeb8f147efaff9dda0a897fc4b132d0d97d02dd73178599f46fd0ad5",
                "sha256:f1b820e3d1ec926439779820d66d6ab72f62ce01730cf410fb4517273de2ab14",
                "sha256:cdf0deebc1c77400ef7e4068c684844d2457edddc601bf61501a76f721e88b93",
                "sha256:60f8a98edc90d1bc05caa1007b55b76a896d3bc3616c5a6581a2e4bd095ba933",
                "sha256:2f15b07cd14fcd9ca3fec5c78b354b92afa8c2c1eee10ae90032f52701e46cf8",
                "sha256:c3d887caf5195ff14be6ef6814b74fbceedb3afb997e1612ba6281f873308bb9",
                "sha256:ef7f82887e447d13e0680794ac4c2f5e7f5078de18ca02c7ad1c7618b4c68657",
                "sha256:e99177b04fdb92757e1ddf1dd286979bcd96eb8044dbea87b5871ce82c80f461",
                "sha256:cc46447ce35f6ad861578ea0d21de7d4b78e0c8edc5230b93e95eb63f90b7b92",
                "sha256:f9a721dae8c69315a9c5566092a9ea6604a0a0fd4bc69ce3375b366cf27a17ea",
                "sha256:e9715089fcc3da052bb28240de3d4f2d6d94b00c37019dec5826d43ac38bc5c6",
                "sha256:627c5016020a9de9d8d7d4166ea460a03e16fc652e0bef6f5568352ab2191adf",
                "sha256:8f532650cf378faf8cd19af21480fc50a76793a8edefe09221e225b550c9d9ef",
                "sha256:97f20e876d3fa500ab486115916cb3a2c0094839681a32cedac1bc34c6d8158e",
                "sha256:b8d58bbc88b6b2b7b49a15774930534793a1391380614fe4e5d1dfe4ba9de85f",
            ],
        )
        self.assertEqual(
            manifest["prompt"]["online_suffix_hash"],
            "sha256:abbf06f4aea7f9fe690c12cc0f08270b1764264cb8ef5603b0bb98f1c2db2d88",
        )


class TestMLAOffBenchmarkRuntimeEvidence(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_redknot_benchmark_module()

    @staticmethod
    def _forward(request_id, forward_id, q_rows, position_start=0):
        return (
            "REDKNOT_MLA_OFF_FORWARD "
            f"request_id={request_id} forward_id={forward_id} "
            f"forward_mode=extend q_rows={q_rows} "
            f"position_start={position_start} "
            f"position_end={position_start + q_rows} "
            "position_contiguous=1 plan_mode=restore "
            "diagnostic_ablation=full\n"
        )

    @staticmethod
    def _metric(
        request_id,
        forward_id,
        q_rows,
        layer_id,
        reused,
        online_local,
        online_global,
    ):
        return (
            "REDKNOT_MLA_OFF_METRIC "
            f"request_id={request_id} forward_id={forward_id} "
            f"forward_mode=extend q_rows={q_rows} layer={layer_id} "
            f"reused_local_head_rows={reused} "
            f"online_local_head_rows={online_local} "
            f"online_global_head_rows={online_global} "
            "diagnostic_ablation=full\n"
        )

    @staticmethod
    def _status(request_id, forward_id, q_rows, layer_id, status, reason):
        return (
            "REDKNOT_MLA_OFF_REQUEST "
            f"request_id={request_id} forward_id={forward_id} "
            f"forward_mode=extend q_rows={q_rows} layer={layer_id} "
            f"status={status} reason={reason} "
            "diagnostic_ablation=full\n"
        )

    def _parse(self, lines):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rank0.log"
            path.write_text("".join(lines), encoding="utf-8")
            with patch.object(
                self.benchmark, "_ih_runtime_log_paths", return_value=[str(path)]
            ):
                return self.benchmark._ih_read_runtime_metrics({str(path): 0})

    def _complete_two_forward_log(self):
        request_id = "measured-request"
        lines = [self._forward(request_id, "f1", 2)]
        for layer_id in (2, 3):
            lines.append(
                self._metric(request_id, "f1", 2, layer_id, 7, 7, 2)
            )
        lines.append(self._forward(request_id, "f2", 1, position_start=2))
        for layer_id in (2, 3):
            lines.append(
                self._metric(request_id, "f2", 1, layer_id, 0, 7, 1)
            )
            lines.append(
                self._status(
                    request_id,
                    "f2",
                    1,
                    layer_id,
                    "full_local",
                    "no_reusable_rows",
                )
            )
        return request_id, lines

    def _validate(
        self,
        metrics,
        request_id,
        *,
        expected_q_rows=3,
        max_q_rows_per_forward=2,
        full_local_sanity=False,
    ):
        return self.benchmark._ih_validate_mla_runtime_metrics(
            metrics,
            expected_request_ids={request_id},
            expected_q_rows_by_request={request_id: expected_q_rows},
            expected_layer_ids=[2, 3],
            expected_request_count=1,
            expected_head_counts_by_layer={
                "2": {"local": 7, "global": 1},
                "3": {"local": 7, "global": 1},
            },
            max_q_rows_per_forward=max_q_rows_per_forward,
            full_local_sanity=full_local_sanity,
        )

    def test_full_local_forward_is_counted_and_validated(self):
        request_id, lines = self._complete_two_forward_log()

        metrics = self._parse(lines)
        validation = self._validate(metrics, request_id)

        self.assertEqual(metrics["mla_off_samples"], 4)
        self.assertEqual(metrics["reused_local_head_rows"], 14)
        self.assertEqual(metrics["online_local_head_rows"], 28)
        self.assertEqual(metrics["online_global_head_rows"], 6)
        self.assertAlmostEqual(metrics["mla_head_row_saving"], 14 / 48)
        self.assertEqual(metrics["mla_evidence_errors"], [])
        self.assertEqual(validation["expected_mla_forward_count"], 2)
        self.assertEqual(validation["observed_mla_forward_count"], 2)
        self.assertEqual(validation["measured_mla_off_samples"], 4)
        self.assertEqual(validation["minimum_mla_off_samples"], 4)
        self.assertTrue(validation["pass"])

    def test_runtime_uses_verified_server_chunk_cap(self):
        request_id = "external-server-request"
        q_rows = 534
        lines = [self._forward(request_id, "f1", q_rows)]
        for layer_id in (2, 3):
            lines.append(
                self._metric(
                    request_id,
                    "f1",
                    q_rows,
                    layer_id,
                    400 * 7,
                    (q_rows - 400) * 7,
                    q_rows,
                )
            )

        metrics = self._parse(lines)
        validation = self._validate(
            metrics,
            request_id,
            expected_q_rows=q_rows,
            max_q_rows_per_forward=4096,
        )
        self.assertEqual(validation["expected_mla_forward_count"], 1)
        self.assertTrue(validation["pass"])

        too_small = self._validate(
            metrics,
            request_id,
            expected_q_rows=q_rows,
            max_q_rows_per_forward=512,
        )
        self.assertEqual(too_small["expected_mla_forward_count"], 2)
        self.assertIn(f"{request_id}/f1", too_small["invalid_mla_forwards"])
        self.assertFalse(too_small["pass"])

    def test_refresh_all_is_evidence_not_formal_reuse(self):
        request_id = "full-local-control"
        q_rows = 3
        lines = [self._forward(request_id, "f1", q_rows)]
        for layer_id in (2, 3):
            lines.extend(
                [
                    self._metric(
                        request_id,
                        "f1",
                        q_rows,
                        layer_id,
                        0,
                        q_rows * 7,
                        q_rows,
                    ),
                    self._status(
                        request_id,
                        "f1",
                        q_rows,
                        layer_id,
                        "full_local",
                        "refresh_layer",
                    ),
                ]
            )

        metrics = self._parse(lines)
        validation = self._validate(
            metrics,
            request_id,
            max_q_rows_per_forward=q_rows,
            full_local_sanity=True,
        )
        self.assertTrue(validation["runtime_evidence_pass"])
        self.assertTrue(validation["full_local_sanity_pass"])
        self.assertFalse(validation["formal_reuse_pass"])
        self.assertFalse(validation["pass"])
        self.assertEqual(validation["measured_mla_head_row_saving"], 0.0)
        self.assertEqual(validation["runtime_evidence_failed_request_ids"], [])
        self.assertEqual(validation["full_local_sanity_failed_request_ids"], [])
        self.assertEqual(validation["formal_failed_mla_request_ids"], [request_id])

    def test_full_local_sanity_rejects_non_refresh_and_duplicate_status(self):
        request_id = "bad-full-local-control"
        q_rows = 3
        lines = [self._forward(request_id, "f1", q_rows)]
        for layer_id in (2, 3):
            status = self._status(
                request_id,
                "f1",
                q_rows,
                layer_id,
                "full_local",
                "no_reusable_rows",
            )
            lines.extend(
                [
                    self._metric(
                        request_id,
                        "f1",
                        q_rows,
                        layer_id,
                        0,
                        q_rows * 7,
                        q_rows,
                    ),
                    status,
                ]
            )
            if layer_id == 2:
                lines.append(status)

        metrics = self._parse(lines)
        validation = self._validate(
            metrics,
            request_id,
            max_q_rows_per_forward=q_rows,
            full_local_sanity=True,
        )
        self.assertIn(request_id, metrics["mla_evidence_error_request_ids"])
        self.assertFalse(validation["runtime_evidence_pass"])
        self.assertFalse(validation["full_local_sanity_pass"])
        self.assertIn(
            f"{request_id}/f1/layer=2",
            validation["full_local_sanity_errors"],
        )

    def test_context_certification_fallback_is_valid_evidence_not_reuse(self):
        request_id = "context-safety-fallback"
        q_rows = 3
        lines = [self._forward(request_id, "f1", q_rows)]
        for layer_id in (2, 3):
            lines.extend(
                [
                    self._metric(
                        request_id,
                        "f1",
                        q_rows,
                        layer_id,
                        0,
                        q_rows * 7,
                        q_rows,
                    ),
                    self._status(
                        request_id,
                        "f1",
                        q_rows,
                        layer_id,
                        "full_local",
                        "context_exceeds_certification",
                    ),
                ]
            )

        metrics = self._parse(lines)
        validation = self._validate(
            metrics,
            request_id,
            max_q_rows_per_forward=q_rows,
        )
        self.assertEqual(metrics["mla_evidence_errors"], [])
        self.assertTrue(validation["runtime_evidence_pass"])
        self.assertFalse(validation["formal_reuse_pass"])
        self.assertFalse(validation["pass"])
        self.assertEqual(validation["formal_failed_mla_request_ids"], [request_id])

    def test_one_complete_chunk_cannot_hide_another_chunks_missing_layer(self):
        request_id, lines = self._complete_two_forward_log()
        missing_line = self._metric(request_id, "f2", 1, 3, 0, 7, 1)
        lines.remove(missing_line)
        lines.remove(
            self._status(
                request_id,
                "f2",
                1,
                3,
                "full_local",
                "no_reusable_rows",
            )
        )

        metrics = self._parse(lines)
        validation = self._validate(metrics, request_id)

        self.assertEqual(metrics["mla_request_layers"][request_id], [2, 3])
        self.assertEqual(
            validation["missing_mla_forward_layers"],
            {f"{request_id}/f2": [3]},
        )
        self.assertFalse(validation["pass"])

    def test_missing_whole_chunk_fails_exact_position_coverage(self):
        request_id, lines = self._complete_two_forward_log()
        lines = [line for line in lines if "forward_id=f2" not in line]

        metrics = self._parse(lines)
        validation = self._validate(metrics, request_id)

        self.assertEqual(metrics["mla_request_layers"][request_id], [2, 3])
        self.assertIn(request_id, validation["mla_position_coverage_errors"])
        self.assertFalse(validation["pass"])

    def test_duplicate_manifest_and_bad_row_geometry_fail_closed(self):
        request_id, lines = self._complete_two_forward_log()
        lines.insert(1, lines[0])
        good_metric = self._metric(request_id, "f1", 2, 2, 7, 7, 2)
        lines.remove(good_metric)
        lines.append(self._metric(request_id, "f1", 2, 2, 7, 6, 2))

        metrics = self._parse(lines)
        validation = self._validate(metrics, request_id)

        self.assertIn(request_id, metrics["mla_evidence_error_request_ids"])
        self.assertIn(
            f"{request_id}/f1/layer=2",
            validation["mla_row_geometry_errors"],
        )
        self.assertFalse(validation["pass"])

    def test_zero_reuse_requires_full_local_status_and_cannot_be_whole_request(
        self,
    ):
        request_id = "measured-request"
        lines = [self._forward(request_id, "f1", 3)]
        for layer_id in (2, 3):
            lines.append(
                self._metric(request_id, "f1", 3, layer_id, 0, 21, 3)
            )

        metrics = self._parse(lines)
        validation = self._validate(metrics, request_id)

        self.assertEqual(
            validation["mla_requests_without_reuse"], [request_id]
        )
        self.assertIn(
            f"{request_id}/f1/layer=2",
            validation["mla_row_geometry_errors"],
        )
        self.assertFalse(validation["pass"])

    def test_unexpected_layer_cannot_hide_policy_drift(self):
        request_id, lines = self._complete_two_forward_log()
        lines.append(self._metric(request_id, "f1", 2, 4, 7, 7, 2))

        metrics = self._parse(lines)
        validation = self._validate(metrics, request_id)

        self.assertEqual(
            validation["unexpected_mla_forward_layers"],
            {f"{request_id}/f1": [4]},
        )
        self.assertFalse(validation["pass"])

    def test_measured_fallback_fails_but_another_requests_fallback_is_scoped(self):
        request_id, lines = self._complete_two_forward_log()
        lines.extend(
            [
                self._forward("other-request", "f9", 1),
                self._status(
                    "other-request",
                    "f9",
                    1,
                    2,
                    "fallback",
                    "restore_not_ready",
                ),
            ]
        )
        metrics = self._parse(lines)
        scoped = self._validate(metrics, request_id)
        self.assertTrue(scoped["pass"])

        lines.append(
            self._status(
                request_id,
                "f2",
                1,
                3,
                "fallback",
                "restore_not_ready",
            )
        )
        metrics = self._parse(lines)
        measured = self._validate(metrics, request_id)
        self.assertEqual(
            measured["measured_mla_fallback_request_ids"], [request_id]
        )
        self.assertFalse(measured["pass"])


class TestMLAOffQualificationClaimGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_redknot_benchmark_module()

    def test_qualification_result_is_always_claim_ineligible(self):
        source = {
            "eligible": True,
            "reasons": [],
            "runtime_evidence": {"minimum_observed_qps_speedup": 9.0},
        }
        gated = self.benchmark._ih_apply_qualification_only_claim_gate(
            source, True
        )
        self.assertFalse(gated["eligible"])
        self.assertEqual(gated["claim_status"], "qualification_only")
        self.assertIn("pure_mla_qualification_only", gated["reasons"])
        self.assertTrue(gated["qualification_only"])
        self.assertTrue(source["eligible"])

    def test_owned_launcher_binds_qualification_environment(self):
        proc = SimpleNamespace(pid=12345)
        with tempfile.TemporaryDirectory() as tmpdir, patch.multiple(
            self.benchmark,
            IH_MLA_OFFLOAD=True,
            IH_MLA_OFF_QUALIFICATION_ONLY=True,
            IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS=0,
            IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS=65568,
            IH_SERVER_LOG=str(Path(tmpdir) / "server.log"),
            IH_RANK_LOG_DIR=str(Path(tmpdir) / "ranks"),
        ), patch.object(
            self.benchmark, "_ih_assert_gpu_capacity"
        ), patch(
            "socket.socket"
        ) as socket_ctor, patch.object(
            self.benchmark.subprocess, "Popen", return_value=proc
        ) as popen:
            socket_ctor.return_value.__enter__.return_value.connect_ex.return_value = 1
            launched = self.benchmark._ih_launch_server()

        self.assertIs(launched, proc)
        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(
            child_env["REDKNOT_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS"], "0"
        )
        self.assertEqual(
            child_env["REDKNOT_MLA_OFF_QUALIFICATION_ONLY"], "1"
        )
        self.assertEqual(
            child_env["REDKNOT_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS"],
            "65568",
        )
        self.assertEqual(
            child_env["REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL"], "triton_h1"
        )


class TestMLAOffBenchmarkServerPolicy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_redknot_benchmark_module()

    def _valid_manifest(self, **overrides):
        _, start_time_ticks = self.benchmark._ih_linux_process_identity(os.getpid())
        kernel_path = (
            self.benchmark.REPO
            / "python/sglang/srt/layers/attention/redknot/"
            "dsv4_shared_latent_batch_kernels.py"
        )
        oracle_path = (
            self.benchmark.REPO
            / "python/sglang/srt/layers/attention/redknot/"
            "probe_dsv4_shared_latent_batch_kernels.py"
        )
        manifest = {
            "format": "redknot_mla_server_policy_v4",
            "execution_profile": MLA_OFF_EXECUTION_PROFILE,
            "dense_prefix_layers": 3,
            "dense_suffix_layers": 3,
            "dense_layer_ids": [0, 1, 2, 40, 41, 42],
            "offline_online_layer_ids": list(range(3, 40)),
            "selected_row_enabled": False,
            "indexer_hot_enabled": False,
            "disable_radix_cache": True,
            "radix_eviction_policy": "lru",
            "runtime_local_layer_ids": list(range(3, 40)),
            "q_projection_scope": (
                "q_a_full_rows_native_dsv4_fullscope_skip0_v1"
            ),
            "head_scope_policy": NATIVE_FULL_SCOPE_POLICY,
            "reuse_heads_full_scope": True,
            "kv_projection_scope": (
                "shared_clean_rows_gpu_restore_dirty_rows_wkv_v1"
            ),
            "compressor_projection_scope": (
                "shared_clean_blocks_gpu_restore_dirty_islands_online_v1"
            ),
            "shared_latent_restore_scope": (
                "persistent_gpu_ragged_fused_scatter_v1"
            ),
            "tp_commit_scope": (
                "composite_forward_prepare_and_full_layer_final_v6"
            ),
            "wo_a_projection_scope": "true_head_column_slices_v1",
            "performance_claim_status": "unverified",
            "backend_ready": True,
            "server_instance_nonce": "unit-test-nonce",
            "pid": os.getpid(),
            "pid_start_time_ticks": start_time_ticks,
            "port": 31998,
            "chunked_prefill_size": 8192,
            "max_prefill_tokens": 8192,
            "mla_off_certified_max_context_tokens": 0,
            "mla_off_qualification_only": False,
            "mla_off_qualification_max_context_tokens": 0,
            "mla_off_effective_restore_max_context_tokens": 0,
            "mla_off_device_cache_enabled": False,
            "mla_off_device_max_bytes": 0,
            "model_compat_hash": "a" * 64,
            "batch_restore_provider_ready": True,
            "batch_restore_provider_error": "",
            "batch_restore_provider_common_token": "sha256:" + "b" * 64,
            "batch_restore_provider_local_token": "sha256:" + "c" * 64,
            "batch_restore_oracle_evidence": {
                "strict_pass": True,
                "kernel_source_sha256": (
                    "sha256:" + self.benchmark._ih_sha256_file(kernel_path)
                ),
                "oracle_source_sha256": (
                    "sha256:" + self.benchmark._ih_sha256_file(oracle_path)
                ),
                "torch_version": "2.0",
                "triton_version": "3.0",
                "cuda_runtime_version": "12.0",
                "device_name": "unit-test-device",
                "device_capability": [8, 9],
            },
        }
        manifest.update(overrides)
        return manifest

    def _verify(self, manifest, *, expected=None):
        if expected is None:
            expected = {
                "format": "redknot_mla_server_policy_v4",
                "backend_ready": True,
                "server_instance_nonce": "unit-test-nonce",
                "port": 31998,
            }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "server_policy_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return self.benchmark._ih_verify_runtime_server_policy(
                {"server_policy_manifest": expected}, str(path)
            )

    def test_valid_v3_manifest_returns_resolved_scheduler_limits(self):
        verified = self._verify(self._valid_manifest())
        self.assertEqual(verified["chunked_prefill_size"], 8192)
        self.assertEqual(verified["max_prefill_tokens"], 8192)
        self.assertEqual(verified["port"], 31998)

    def test_schema_fields_types_and_port_fail_closed(self):
        cases = {
            "old schema": {"format": "redknot_mla_server_policy_v3"},
            "missing port": {"port": None},
            "wrong port": {"port": 31999},
            "missing chunk cap": {"chunked_prefill_size": None},
            "missing prefill cap": {"max_prefill_tokens": None},
            "missing start time": {"pid_start_time_ticks": None},
            "integer backend ready": {"backend_ready": 1},
            "radix cache enabled": {"disable_radix_cache": False},
            "bool chunk cap": {"chunked_prefill_size": True},
            "string chunk cap": {"chunked_prefill_size": "8192"},
            "zero chunk cap": {"chunked_prefill_size": 0},
            "chunk greater than prefill": {
                "chunked_prefill_size": 8192,
                "max_prefill_tokens": 4096,
            },
            "missing pid": {"pid": None},
            "bool pid": {"pid": True},
            "bool start time": {"pid_start_time_ticks": True},
            "bool certified cap": {
                "mla_off_certified_max_context_tokens": True
            },
            "negative certified cap": {
                "mla_off_certified_max_context_tokens": -1
            },
            "integer qualification-only": {
                "mla_off_qualification_only": 1
            },
            "negative qualification cap": {
                "mla_off_qualification_max_context_tokens": -1
            },
            "formal qualification cap": {
                "mla_off_qualification_max_context_tokens": 4096
            },
            "formal effective cap mismatch": {
                "mla_off_effective_restore_max_context_tokens": 4096
            },
            "integer device enable": {"mla_off_device_cache_enabled": 0},
            "negative device cap": {"mla_off_device_max_bytes": -1},
            "device enable/cap mismatch": {
                "mla_off_device_cache_enabled": True,
                "mla_off_device_max_bytes": 0,
            },
        }
        for name, overrides in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._verify(self._valid_manifest(**overrides))

        expected = {
            "format": "redknot_mla_server_policy_v4",
            "backend_ready": True,
            "server_instance_nonce": "unit-test-nonce",
            "port": 65536,
        }
        with self.assertRaisesRegex(ValueError, "outside"):
            self._verify(self._valid_manifest(port=65536), expected=expected)

    def test_qualification_manifest_is_bound_and_claim_ineligible(self):
        manifest = self._valid_manifest(
            mla_off_qualification_only=True,
            mla_off_qualification_max_context_tokens=65568,
            mla_off_effective_restore_max_context_tokens=65568,
            performance_claim_status="qualification_only_claim_ineligible",
        )
        verified = self._verify(manifest)
        self.assertIs(verified["mla_off_qualification_only"], True)

        invalid = (
            {"mla_off_certified_max_context_tokens": 1},
            {"mla_off_qualification_max_context_tokens": 0},
            {"mla_off_effective_restore_max_context_tokens": 1},
            {"performance_claim_status": "unverified"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._verify({**manifest, **overrides})

    def test_zombie_and_pid_reuse_are_rejected(self):
        manifest = self._valid_manifest()
        start_time_ticks = manifest["pid_start_time_ticks"]
        with patch.object(
            self.benchmark,
            "_ih_linux_process_identity",
            return_value=("Z", start_time_ticks),
        ), self.assertRaisesRegex(ValueError, "dead worker"):
            self._verify(manifest)

        with patch.object(
            self.benchmark,
            "_ih_linux_process_identity",
            return_value=("S", start_time_ticks + 1),
        ), self.assertRaisesRegex(ValueError, "start time"):
            self._verify(manifest)


class TestMLAOffRuntimeManifestWriter(CustomTestCase):
    def test_manifest_publishes_resolved_scheduler_and_process_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text("{}\n", encoding="utf-8")
            manifest_path = Path(tmpdir) / "server_policy_manifest.json"
            model_runner = SimpleNamespace(
                model_config=SimpleNamespace(
                    model_path=str(model_path),
                    hf_config=SimpleNamespace(o_groups=8, o_lora_rank=1024),
                )
            )
            server_args = SimpleNamespace(
                port=32021,
                chunked_prefill_size=8192,
                max_prefill_tokens=8192,
                disable_radix_cache=True,
                radix_eviction_policy="lru",
                redknot_head_config_path="",
                redknot_mla_dense_prefix_layers=3,
                redknot_mla_dense_suffix_layers=3,
                redknot_mla_local_window=128,
                redknot_mla_global_head_stride=8,
                redknot_mla_global_layer_stride=0,
            )
            backend = redknot_mla_backend.RedKnotMLAAttnBackend.__new__(
                redknot_mla_backend.RedKnotMLAAttnBackend
            )
            backend._redknot_tp_rank = 0
            backend._redknot_tp_size = 8
            backend.redknot_mla_pass_mode = "headwise"
            backend._redknot_reuse_heads_full_scope = True
            backend._redknot_mla_off_global_attention_impl = "triton_h1"
            backend._redknot_mla_off_execution_profile = (
                MLA_OFF_EXECUTION_PROFILE
            )
            backend._redknot_mla_off_certified_max_context_tokens = 2048
            backend._redknot_mla_off_qualification_only = False
            backend._redknot_mla_off_qualification_max_context_tokens = 0
            backend._redknot_mla_off_enabled = True
            backend._redknot_disable_radix_cache = True
            backend._redknot_mla_prefix_materialization = False
            backend._redknot_radix_eviction_policy = "lru"
            backend._redknot_mla_off_compact_woa = False
            backend.redknot_v4_mode = "aggressive"
            # The manifest writer is intentionally exercised on an object
            # built with __new__, so mirror the closure fields that __init__
            # always installs in production.
            backend._redknot_three_way_closure = False
            backend._redknot_token_sparse_dense_suffix_layers = 0
            backend._redknot_token_sparse_boundary_tokens = 0
            backend._redknot_mla_off_dp_size = 1
            backend._redknot_mla_off_dp_attention = False
            backend._redknot_mla_off_cp_size = 1
            backend._redknot_mla_off_pp_size = 1
            backend._redknot_restore_pipeline_group_layers = 0
            backend._redknot_swa_capacity = 128
            backend._redknot_mla_off_policy_hash = "b" * 64
            backend._redknot_mla_off_model_hash = "c" * 64
            backend._redknot_mla_off_rank_local_layer_ids = tuple(range(3, 40))

            with patch.dict(
                os.environ,
                {
                    "REDKNOT_SERVER_POLICY_MANIFEST_OUT": str(manifest_path),
                    "REDKNOT_SERVER_INSTANCE_NONCE": "writer-unit-test",
                    "REDKNOT_MLA_OFF_MAX_BYTES": "1073741824",
                },
            ):
                backend._mla_off_write_runtime_manifest(
                    model_runner=model_runner, server_args=server_args
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state, start_time_ticks = redknot_mla_backend._linux_process_identity(
                os.getpid()
            )
            self.assertNotIn(state, {"Z", "X", "x"})
            self.assertEqual(manifest["format"], "redknot_mla_server_policy_v4")
            self.assertEqual(manifest["pid"], os.getpid())
            self.assertEqual(manifest["pid_start_time_ticks"], start_time_ticks)
            self.assertEqual(manifest["port"], 32021)
            self.assertEqual(manifest["chunked_prefill_size"], 8192)
            self.assertEqual(manifest["max_prefill_tokens"], 8192)
            self.assertIs(manifest["disable_radix_cache"], True)
            self.assertIs(manifest["prefix_materialization"], False)
            self.assertEqual(manifest["radix_eviction_policy"], "lru")
            self.assertEqual(
                manifest["mla_off_certified_max_context_tokens"], 2048
            )
            self.assertIs(manifest["mla_off_qualification_only"], False)
            self.assertEqual(
                manifest["mla_off_qualification_max_context_tokens"], 0
            )
            self.assertEqual(
                manifest["mla_off_effective_restore_max_context_tokens"],
                2048,
            )
            self.assertIs(manifest["reuse_heads_full_scope"], True)
            self.assertIs(manifest["mla_off_compact_woa_enabled"], False)
            self.assertEqual(manifest["dense_prefix_layers"], 3)
            self.assertEqual(manifest["dense_suffix_layers"], 3)
            self.assertEqual(
                manifest["runtime_local_layer_ids"], list(range(3, 40))
            )
            self.assertEqual(
                manifest["execution_profile"], MLA_OFF_EXECUTION_PROFILE
            )
            self.assertIs(manifest["selected_row_enabled"], False)
            self.assertIs(manifest["indexer_hot_enabled"], False)
            self.assertEqual(
                manifest["wo_a_projection_scope"],
                "true_head_column_slices_v1",
            )
            self.assertEqual(
                manifest["q_projection_scope"],
                "q_a_full_rows_native_dsv4_fullscope_skip0_v1",
            )
            self.assertEqual(
                manifest["head_scope_policy"], NATIVE_FULL_SCOPE_POLICY
            )
            self.assertEqual(
                manifest["online_global_attention_impl"], "triton_h1"
            )
            self.assertEqual(
                manifest["tp_commit_scope"],
                "composite_forward_prepare_and_full_layer_final_v6",
            )
            self.assertEqual(
                manifest["performance_claim_status"], "unverified"
            )


class TestForwardCompositeReceiptIdentity(CustomTestCase):
    """CPU-only coverage for the forward-certificate/layer-receipt boundary."""

    @staticmethod
    def _uninitialized(cls):
        # These protocol objects are used only as opaque identity tokens here.
        # Their semantic validation belongs to dsv4_composite_commit tests; this
        # fixture isolates the integration ordering bug without tensors or TP.
        return object.__new__(cls)

    def _make_bound_builder(self):
        certificate = self._uninitialized(ForwardCommitCertificate)
        session = self._uninitialized(ForwardCommitSession)
        session.certificate = certificate
        authorization = self._uninitialized(OmissionAuthorization)
        batch_plan = SimpleNamespace(validate=MagicMock())
        persistent_plan = object()
        context = SimpleNamespace(
            layer_id=3,
            batched_reuse_plan=batch_plan,
            sparse_q_committed=False,
            composite_irreversible=True,
            composite_commit_session=session,
            composite_certificate=certificate,
            composite_omission_authorization=authorization,
            composite_layer_execution_receipt=None,
            validate_persistent_projection_commit=MagicMock(
                return_value=persistent_plan
            ),
        )
        owner = SimpleNamespace(
            session=session,
            authorization=authorization,
            committed=True,
            ledger=SimpleNamespace(
                session=session,
                authorization=authorization,
            ),
            _bound_contexts={3: context},
        )
        resources = SimpleNamespace(
            validate=MagicMock(),
            _contexts={3: context},
            batch_digest="batch-digest",
            forward_id="forward-generation",
            total_rows=1,
            forward_commit_coordinator=owner,
        )
        cache_domain = SimpleNamespace(
            restore_plan_digest="restore-plan",
            builder_preflight_token="cache-preflight",
        )
        proposal = SimpleNamespace(
            ragged=SimpleNamespace(total_rows=1),
            shared_latent=SimpleNamespace(restore_plan_digest="restore-plan"),
            cache_domains=(cache_domain,),
            z_off_views=(SimpleNamespace(artifact_digest="z-off-artifact"),),
        )
        packed_sparse_q = SimpleNamespace(
            validate=MagicMock(),
            projection_token="packed-q-projection",
        )
        builder = self._uninitialized(LayerCompositeCommitBuilder)
        builder.context = context
        builder.resources = resources
        builder.proposal = proposal
        builder.packed_sparse_q = packed_sparse_q
        builder._packed_identity = ("packed-q",)
        builder._persistent_identity = ("persistent-z",)
        builder._cache_live_tensors = ()
        builder.builder_epoch_token = "layer-builder-epoch"
        builder.committed_forward_session = session
        builder.committed_forward_authorization = authorization
        return (
            builder,
            context,
            resources,
            session,
            certificate,
            authorization,
            persistent_plan,
        )

    @staticmethod
    def _coordinator(resources, session, authorization):
        coordinator = object.__new__(ForwardCompositeCommitCoordinator)
        coordinator.resources = resources
        coordinator.session = session
        coordinator.authorization = authorization
        coordinator.outcome = SimpleNamespace(committed=True)
        coordinator.ledger = MagicMock()
        coordinator.ledger.session = session
        coordinator.ledger.authorization = authorization
        coordinator._bound_contexts = dict(resources._contexts)
        coordinator.bind_context = MagicMock()
        resources.forward_commit_coordinator = coordinator
        return coordinator

    def _identity_patches(self, persistent_plan):
        return patch.multiple(
            dsv4_reuse_backend_runtime,
            _batched_geometry_digest=MagicMock(return_value="batch-digest"),
            _packed_sparse_q_identity=MagicMock(return_value=("packed-q",)),
            _persistent_projection_identity=MagicMock(
                side_effect=lambda value: (
                    ("persistent-z",)
                    if value is persistent_plan
                    else ("foreign-persistent-z",)
                )
            ),
        )

    def _make_unbound_coordinator(self):
        certificate = self._uninitialized(ForwardCommitCertificate)
        session = self._uninitialized(ForwardCommitSession)
        session.certificate = certificate
        authorization = self._uninitialized(OmissionAuthorization)
        context = SimpleNamespace(
            layer_id=3,
            composite_irreversible=False,
            composite_commit_session=None,
            composite_certificate=None,
            composite_omission_authorization=None,
            composite_layer_execution_receipt=None,
            sparse_q_committed=False,
        )
        resources = SimpleNamespace(
            validate=MagicMock(),
            _contexts={3: context},
        )
        coordinator = object.__new__(ForwardCompositeCommitCoordinator)
        coordinator.resources = resources
        coordinator.session = session
        coordinator.authorization = authorization
        coordinator.outcome = SimpleNamespace(committed=True)
        coordinator.ledger = SimpleNamespace(
            session=session,
            authorization=authorization,
        )
        coordinator._bound_contexts = {}
        resources.forward_commit_coordinator = coordinator
        return coordinator, context, session, certificate, authorization

    def test_forward_context_first_bind_requires_and_updates_pristine_state(self):
        (
            coordinator,
            context,
            session,
            certificate,
            authorization,
        ) = self._make_unbound_coordinator()

        coordinator.bind_context(context)

        self.assertIs(context.composite_irreversible, True)
        self.assertIs(context.composite_commit_session, session)
        self.assertIs(context.composite_certificate, certificate)
        self.assertIs(context.composite_omission_authorization, authorization)
        self.assertIs(coordinator._bound_contexts[3], context)

    def test_forward_context_first_bind_rejects_foreign_partial_state(self):
        for field, foreign_type in (
            ("composite_commit_session", ForwardCommitSession),
            ("composite_certificate", ForwardCommitCertificate),
            ("composite_omission_authorization", OmissionAuthorization),
        ):
            with self.subTest(field=field):
                coordinator, context, *_ = self._make_unbound_coordinator()
                foreign = self._uninitialized(foreign_type)
                setattr(context, field, foreign)

                with self.assertRaises(RuntimeError):
                    coordinator.bind_context(context)

                self.assertIs(getattr(context, field), foreign)
                self.assertEqual(coordinator._bound_contexts, {})

    def test_forward_context_second_exact_bind_is_idempotent(self):
        coordinator, context, session, certificate, authorization = (
            self._make_unbound_coordinator()
        )
        coordinator.bind_context(context)

        coordinator.bind_context(context)

        self.assertIs(context.composite_commit_session, session)
        self.assertIs(context.composite_certificate, certificate)
        self.assertIs(context.composite_omission_authorization, authorization)
        self.assertEqual(coordinator._bound_contexts, {3: context})

    def test_forward_context_second_mutated_bind_is_rejected(self):
        coordinator, context, *_ = self._make_unbound_coordinator()
        coordinator.bind_context(context)
        foreign = self._uninitialized(ForwardCommitCertificate)
        context.composite_certificate = foreign

        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            coordinator.bind_context(context)

        self.assertIs(context.composite_certificate, foreign)
        self.assertIs(coordinator._bound_contexts[3], context)

    def test_forward_receipt_accepts_exact_bound_identity_without_overwrite(self):
        (
            builder,
            context,
            resources,
            session,
            certificate,
            authorization,
            persistent_plan,
        ) = self._make_bound_builder()
        coordinator = self._coordinator(resources, session, authorization)
        self.assertIs(resources.forward_commit_coordinator, coordinator)
        receipt = object()
        coordinator.ledger.record_layer_execution.return_value = receipt
        observed = SimpleNamespace(digest="layer-reservation")

        with self._identity_patches(persistent_plan), patch.object(
            dsv4_reuse_backend_runtime,
            "build_layer_reservation_binding",
            return_value=observed,
        ):
            actual = coordinator.record_layer_builder(builder)

        self.assertIs(actual, receipt)
        self.assertIs(context.composite_commit_session, session)
        self.assertIs(context.composite_certificate, certificate)
        self.assertIs(context.composite_omission_authorization, authorization)
        self.assertIs(context.composite_layer_execution_receipt, receipt)
        coordinator.ledger.record_failure.assert_not_called()
        coordinator.bind_context.assert_called_once_with(context)

    def test_prepared_full_receipt_reuses_forward_bindings_without_plan_rehash(self):
        (
            _builder,
            context,
            resources,
            session,
            _certificate,
            authorization,
            _persistent_plan,
        ) = self._make_bound_builder()
        coordinator = self._coordinator(resources, session, authorization)
        receipt = object()
        coordinator.ledger.record_layer_execution.return_value = receipt
        sparse = SimpleNamespace(
            layer_id=3,
            projection_token="packed-q-projection",
            packed_projection_view=SimpleNamespace(device_index=0),
        )
        cache = SimpleNamespace(
            layer_id=3,
            builder_preflight_token="cache-preflight",
        )
        zoff = SimpleNamespace(layer_id=3, artifact_digest="z-off-artifact")
        coordinator.proposal = SimpleNamespace(
            omission_profile="full",
            sparse_q=(sparse,),
            cache_domains=(cache,),
            z_off_views=(zoff,),
            shared_latent=SimpleNamespace(restore_plan_digest="restore-plan"),
            persistent_zoff_arena_token="persistent-arena",
            fused_merge_kernel_token="merge-kernel",
        )
        observed = SimpleNamespace(digest="layer-reservation")
        packed_sparse_q = SimpleNamespace(
            projection_token="packed-q-projection"
        )
        context.batched_reuse_plan.validate.reset_mock()

        with patch.multiple(
            dsv4_reuse_backend_runtime,
            _build_sparse_q_binding=MagicMock(return_value=sparse),
            _layer_compression_ratio=MagicMock(return_value=4),
            _build_cache_domain_bindings=MagicMock(
                return_value=((cache,), ())
            ),
            _build_zoff_binding=MagicMock(return_value=(zoff, ("zoff",))),
            build_layer_reservation_binding=MagicMock(return_value=observed),
            _batched_geometry_digest=MagicMock(
                side_effect=AssertionError("prepared receipt rehashed geometry")
            ),
        ):
            actual = coordinator.record_prepared_full_layer(
                context=context,
                packed_sparse_q=packed_sparse_q,
                cache_domains=(cache,),
                builder_epoch_token="layer-builder-epoch",
            )

        self.assertIs(actual, receipt)
        self.assertIs(context.composite_layer_execution_receipt, receipt)
        context.batched_reuse_plan.validate.assert_not_called()
        coordinator.ledger.record_failure.assert_not_called()
        coordinator.bind_context.assert_called_once_with(context)

    def test_prepared_full_receipt_rejects_changed_live_binding(self):
        (
            _builder,
            context,
            resources,
            session,
            _certificate,
            authorization,
            _persistent_plan,
        ) = self._make_bound_builder()
        coordinator = self._coordinator(resources, session, authorization)
        reserved_sparse = SimpleNamespace(
            layer_id=3,
            projection_token="reserved",
            packed_projection_view=SimpleNamespace(device_index=0),
        )
        live_sparse = SimpleNamespace(
            layer_id=3,
            projection_token="changed",
            packed_projection_view=SimpleNamespace(device_index=0),
        )
        cache = SimpleNamespace(layer_id=3, builder_preflight_token="cache")
        zoff = SimpleNamespace(layer_id=3, artifact_digest="zoff")
        coordinator.proposal = SimpleNamespace(
            omission_profile="full",
            sparse_q=(reserved_sparse,),
            cache_domains=(cache,),
            z_off_views=(zoff,),
            shared_latent=SimpleNamespace(restore_plan_digest="restore-plan"),
            persistent_zoff_arena_token="persistent-arena",
            fused_merge_kernel_token="merge-kernel",
        )

        with patch.multiple(
            dsv4_reuse_backend_runtime,
            _build_sparse_q_binding=MagicMock(return_value=live_sparse),
            _layer_compression_ratio=MagicMock(return_value=4),
            _build_cache_domain_bindings=MagicMock(
                return_value=((cache,), ())
            ),
            _build_zoff_binding=MagicMock(return_value=(zoff, ("zoff",))),
        ):
            actual = coordinator.record_prepared_full_layer(
                context=context,
                packed_sparse_q=SimpleNamespace(projection_token="changed"),
                cache_domains=(cache,),
                builder_epoch_token="layer-builder-epoch",
            )

        self.assertIsNone(actual)
        coordinator.ledger.record_layer_execution.assert_not_called()
        coordinator.ledger.record_failure.assert_called_once()
        self.assertEqual(
            coordinator.ledger.record_failure.call_args.kwargs["stage"],
            "prepared_full_layer_live_preflight",
        )

    def test_forward_receipt_construction_preserves_coordinator_session(self):
        certificate = self._uninitialized(ForwardCommitCertificate)
        session = self._uninitialized(ForwardCommitSession)
        session.certificate = certificate
        authorization = self._uninitialized(OmissionAuthorization)
        resources = object.__new__(CompositeForwardResources)
        resources.validate = MagicMock()
        resources.forward_id = "forward-generation"
        resources.register_commit_builder = MagicMock()
        context = SimpleNamespace(
            layer_id=3,
            spec=SimpleNamespace(
                model_compat_hash="model",
                head_policy_hash="policy",
                tp_size=1,
                tp_rank=0,
            ),
            batched_reuse_plan=SimpleNamespace(digest="batch-plan"),
            persistent_projection_plan=SimpleNamespace(digest="persistent-plan"),
            sparse_q_committed=False,
            composite_irreversible=True,
            composite_commit_session=session,
            composite_certificate=certificate,
            composite_omission_authorization=authorization,
            composite_layer_execution_receipt=None,
            _redknot_composite_forward_resources=resources,
        )
        owner = SimpleNamespace(
            session=session,
            authorization=authorization,
            committed=True,
            _bound_contexts={3: context},
        )
        resources._contexts = {3: context}
        resources._forward_commit_coordinator = owner
        sparse = SimpleNamespace(
            packed_projection_view=SimpleNamespace(device_index=0)
        )
        z_off = SimpleNamespace(gpu_view=SimpleNamespace(device_index=0))
        proposal = object()
        builder = SimpleNamespace(session=self._uninitialized(ForwardCommitSession))
        builder_factory = MagicMock(return_value=builder)

        with patch.multiple(
            dsv4_reuse_backend_runtime,
            _layer_compression_ratio=MagicMock(return_value=4),
            _build_ragged_geometry=MagicMock(return_value=object()),
            _build_shared_latent_binding=MagicMock(
                return_value=SimpleNamespace(
                    restore_plan_digest="restore-plan"
                )
            ),
            _build_sparse_q_binding=MagicMock(return_value=sparse),
            _build_zoff_binding=MagicMock(
                return_value=(z_off, ("persistent-z",))
            ),
            _build_cache_domain_bindings=MagicMock(return_value=((), ())),
            CompositeForwardProposal=MagicMock(return_value=proposal),
            LayerCompositeCommitBuilder=builder_factory,
        ):
            actual = dsv4_reuse_backend_runtime._begin_layer_composite_commit_impl(
                context,
                cache_domains=(),
                packed_sparse_q=object(),
                forward_ordinal=0,
                builder_epoch_token="layer-builder-epoch",
                committed_forward_session=session,
                committed_forward_authorization=authorization,
            )

        self.assertIs(actual, builder)
        self.assertIs(context.composite_commit_session, session)
        self.assertIsNot(context.composite_commit_session, builder.session)
        resources.register_commit_builder.assert_called_once_with(3, builder)
        builder_call = builder_factory.call_args.kwargs
        self.assertIs(builder_call["committed_forward_session"], session)
        self.assertIs(
            builder_call["committed_forward_authorization"], authorization
        )

    def test_forward_receipt_rejects_wrong_bound_identity_fail_closed(self):
        for field, foreign_type in (
            ("composite_commit_session", ForwardCommitSession),
            ("composite_certificate", ForwardCommitCertificate),
            ("composite_omission_authorization", OmissionAuthorization),
        ):
            with self.subTest(field=field):
                (
                    builder,
                    context,
                    resources,
                    session,
                    _certificate,
                    authorization,
                    persistent_plan,
                ) = self._make_bound_builder()
                setattr(context, field, self._uninitialized(foreign_type))
                coordinator = self._coordinator(resources, session, authorization)

                with self._identity_patches(persistent_plan):
                    actual = coordinator.record_layer_builder(builder)

                self.assertIsNone(actual)
                coordinator.ledger.record_layer_execution.assert_not_called()
                coordinator.ledger.record_failure.assert_called_once()
                failure = coordinator.ledger.record_failure.call_args.kwargs
                self.assertEqual(failure["layer_id"], 3)
                self.assertEqual(failure["stage"], "layer_live_preflight")

    def test_forward_receipt_rejects_replaced_active_coordinator_fail_closed(self):
        (
            builder,
            _context,
            resources,
            session,
            _certificate,
            authorization,
            persistent_plan,
        ) = self._make_bound_builder()
        coordinator = self._coordinator(resources, session, authorization)
        resources.forward_commit_coordinator = SimpleNamespace(
            session=session,
            authorization=authorization,
            committed=True,
            ledger=coordinator.ledger,
        )

        with self._identity_patches(persistent_plan):
            actual = coordinator.record_layer_builder(builder)

        self.assertIsNone(actual)
        coordinator.ledger.record_layer_execution.assert_not_called()
        coordinator.ledger.record_failure.assert_called_once()
        failure = coordinator.ledger.record_failure.call_args.kwargs
        self.assertEqual(failure["layer_id"], 3)
        self.assertEqual(failure["stage"], "layer_live_preflight")
        self.assertIn("coordinator binding changed", failure["detail"])

    def test_legacy_layer_commit_still_rejects_preinstalled_certificate(self):
        (
            builder,
            _context,
            _resources,
            _session,
            _certificate,
            _authorization,
            persistent_plan,
        ) = self._make_bound_builder()
        builder.committed_forward_session = None
        builder.committed_forward_authorization = None

        with self._identity_patches(persistent_plan), self.assertRaisesRegex(
            RuntimeError, "before TP commit"
        ):
            builder._validate_precommit()


class TestForwardWideRestoreSlotBounds(CustomTestCase):
    @staticmethod
    def _c128_targets(*, c128_slots, slot_bounds_batch=None):
        layer_id = 3
        swa = dsv4_shared_latent_sglang.DOMAIN_SWA
        c128 = dsv4_shared_latent_sglang.DOMAIN_C128
        state = dsv4_shared_latent_sglang.DOMAIN_C128_ATTENTION_STATE
        keys = ((swa, layer_id), (c128, layer_id), (state, layer_id))
        targets = {key: torch.zeros((2, 1)) for key in keys}
        target_slots = {
            (swa, layer_id): torch.tensor((0, 1), dtype=torch.long),
            (c128, layer_id): torch.tensor(c128_slots, dtype=torch.long),
            (state, layer_id): torch.tensor((0, 1), dtype=torch.long),
        }
        return dsv4_shared_latent_sglang.LayerRestoreTargets(
            layer_id=layer_id,
            compress_ratio=128,
            targets=targets,
            target_slots=target_slots,
            packed_page_sizes={
                (swa, layer_id): 1,
                (c128, layer_id): 1,
            },
            indexer_page_sizes={},
            state_group_widths={(state, layer_id): 1},
            slot_bounds_batch=slot_bounds_batch,
        )

    def test_direct_target_construction_keeps_immediate_fail_closed_check(self):
        with self.assertRaisesRegex(
            ValueError,
            r"restore slots \('c128', 3\) exceed target capacity",
        ):
            self._c128_targets(c128_slots=(0, 2))

    def test_forward_batch_defers_only_the_scalar_fence(self):
        bounds = dsv4_shared_latent_sglang.RestoreSlotBoundsBatch()
        targets = self._c128_targets(
            c128_slots=(0, 2), slot_bounds_batch=bounds
        )
        self.assertEqual(targets.q_rows, 2)
        self.assertFalse(bounds.is_validated)
        self.assertEqual(bounds.entry_count, 1)
        with self.assertRaisesRegex(
            ValueError,
            r"restore slots \('c128', 3\) exceed target capacity",
        ):
            bounds.finalize(expected_layer_ids=(3,))
        self.assertFalse(bounds.is_validated)

    def test_all_147_vectors_use_one_aggregate_predicate(self):
        bounds = dsv4_shared_latent_sglang.RestoreSlotBoundsBatch()
        expected_layers = tuple(range(3, 40))
        expected_vectors = 0
        for layer_id in expected_layers:
            # DeepSeek-V4 layers 3..39 contain 19 C128 layers with three
            # target domains and 18 C4 layers with five target domains.
            domains = (
                ("swa", "c4", "indexer", "attention_state", "indexer_state")
                if layer_id % 2 == 0
                else ("swa", "c128", "attention_state")
            )
            expected_vectors += len(domains)
            bounds.register(
                tuple((domain, layer_id) for domain in domains),
                torch.zeros((len(domains), 2), dtype=torch.bool),
            )

        with patch.object(
            dsv4_shared_latent_sglang.torch,
            "cat",
            wraps=torch.cat,
        ) as concatenate:
            certificate = bounds.finalize(
                expected_layer_ids=expected_layers
            )
            repeated = bounds.finalize(expected_layer_ids=expected_layers)

        self.assertIs(repeated, certificate)
        self.assertEqual(expected_vectors, 147)
        self.assertEqual(certificate.layer_ids, expected_layers)
        self.assertEqual(certificate.vector_count, 147)
        self.assertEqual(certificate.predicate_count, 294)
        concatenate.assert_called_once()

    def test_aggregate_failure_preserves_exact_layer_and_domain(self):
        bounds = dsv4_shared_latent_sglang.RestoreSlotBoundsBatch()
        bounds.register(
            (("swa", 3),),
            torch.tensor(((False, False),), dtype=torch.bool),
        )
        bounds.register(
            (("c128", 4),),
            torch.tensor(((False, True),), dtype=torch.bool),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"restore slots \('c128', 4\) exceed target capacity",
        ):
            bounds.finalize(expected_layer_ids=(3, 4))

    def test_missing_layer_and_post_certificate_registration_fail_closed(self):
        incomplete = dsv4_shared_latent_sglang.RestoreSlotBoundsBatch()
        incomplete.register(
            (("swa", 3),), torch.zeros((1, 2), dtype=torch.bool)
        )
        with self.assertRaisesRegex(ValueError, "does not cover"):
            incomplete.finalize(expected_layer_ids=(3, 4))

        complete = dsv4_shared_latent_sglang.RestoreSlotBoundsBatch()
        complete.register(
            (("swa", 3),), torch.zeros((1, 2), dtype=torch.bool)
        )
        complete.finalize(expected_layer_ids=(3,))
        with self.assertRaisesRegex(RuntimeError, "after batch validation"):
            complete.register(
                (("c128", 4),), torch.zeros((1, 2), dtype=torch.bool)
            )

    def test_backend_fences_bounds_before_descriptor_preflight(self):
        source = inspect.getsource(
            redknot_mla_backend.RedKnotMLAAttnBackend.begin_mla_off_forward_transaction
        )
        finalize = source.index("slot_bounds_batch.finalize(")
        descriptor_preflight = source.index("preflight_device_restore_batch(")
        transaction_publish = source.index(
            "forward_batch._redknot_mla_off_forward_transaction = transaction",
            finalize,
        )
        self.assertLess(finalize, descriptor_preflight)
        self.assertLess(finalize, transaction_publish)
        self.assertIn(
            "slot_bounds_certificate=slot_bounds_certificate",
            source,
        )


class TestSharedLatentStateScheduleProvenance(CustomTestCase):
    @staticmethod
    def _workset_and_certificate(*, ratio: int, request_row_begin: int):
        row_count = 8192
        seq_len_before = 8192
        span = 128
        flat_begin = int(request_row_begin)
        flat_end = flat_begin + span
        token_begin = seq_len_before + int(request_row_begin)
        token_end = token_begin + span
        completions = tuple(
            flat_begin + offset
            for offset, token in enumerate(range(token_begin, token_end))
            if (token + 1) % int(ratio) == 0
        )
        logical_island = dsv4_shared_latent_sglang.DirtyCompressorIsland(
            flat_begin=flat_begin,
            flat_end=flat_end,
            request_row_begin=int(request_row_begin),
            request_row_end=int(request_row_begin) + span,
            token_begin=token_begin,
            token_end=token_end,
            state_slot_indices=(),
            completion_output_rows=completions,
        )
        slot_count = 2 if int(ratio) == 4 else 1
        resolved_island = dsv4_shared_latent_sglang.DirtyCompressorIsland(
            flat_begin=logical_island.flat_begin,
            flat_end=logical_island.flat_end,
            request_row_begin=logical_island.request_row_begin,
            request_row_end=logical_island.request_row_end,
            token_begin=logical_island.token_begin,
            token_end=logical_island.token_end,
            state_slot_indices=tuple(range(100, 100 + slot_count)),
            completion_output_rows=logical_island.completion_output_rows,
        )
        logical = dsv4_shared_latent_sglang.DirtyRequestWorkset(
            request_index=0,
            flat_row_offset=0,
            row_count=row_count,
            seq_len_before=seq_len_before,
            islands=(logical_island,),
        )
        resolved = dsv4_shared_latent_sglang.DirtyRequestWorkset(
            request_index=0,
            flat_row_offset=0,
            row_count=row_count,
            seq_len_before=seq_len_before,
            islands=(resolved_island,),
        )
        certificate = (
            dsv4_shared_latent_sglang.DirtyStateSlotResolutionCertificate(
                forward_token="forward-prefix-state",
                compress_ratio=int(ratio),
                q_rows=row_count,
                logical_worksets=(logical,),
                resolved_worksets=(resolved,),
                source_binding=("req-to-token-bound-source",),
                pool_geometry=("state-pool-geometry",),
            )
        )
        return resolved, certificate

    @staticmethod
    def _adapter(*, ratio: int, workset, certificate, authorized=False):
        restore_targets = SimpleNamespace(
            dirty_worksets=(workset,),
            dirty_state_slot_certificate=certificate,
            forward_token="forward-prefix-state",
        )
        authorizations = (
            (
                dsv4_shared_latent_sglang.LivePrefixStateContinuationAuthorization(
                    request_index=int(workset.request_index),
                    flat_row_offset=int(workset.flat_row_offset),
                    seq_len_before=int(workset.seq_len_before),
                    row_count=int(workset.row_count),
                    prior_forward_token="sha256:" + "a" * 64,
                    terminal_state_slots=tuple(
                        (
                            3,
                            domain,
                            int(workset.islands[0].state_slot_indices[-1]),
                        )
                        for domain in (
                            (
                                dsv4_shared_latent_sglang.DOMAIN_C4_ATTENTION_STATE,
                                dsv4_shared_latent_sglang.DOMAIN_INDEXER_STATE,
                            )
                            if int(ratio) == 4
                            else (
                                dsv4_shared_latent_sglang.DOMAIN_C128_ATTENTION_STATE,
                            )
                        )
                    ),
                ),
            )
            if authorized
            else ()
        )
        return dsv4_shared_latent_sglang.LayerRestoreAdapter(
            layer_id=3,
            compress_ratio=int(ratio),
            targets={},
            target_slots={},
            kernels={},
            batch_metadata={},
            restore_targets=restore_targets,
            live_prefix_state_authorizations=authorizations,
        )

    @staticmethod
    def _schedule(
        *,
        ratio: int,
        output_rows=(),
        position_start: int = 8192,
        row_count: int = 8192,
        forward_id: str = "forward-prefix-state:request:0",
    ):
        arena = tuple(int(value) for value in output_rows)
        domains = (
            (
                dsv4_shared_latent_sglang.DOMAIN_C4_ATTENTION_STATE,
                dsv4_shared_latent_sglang.DOMAIN_INDEXER_STATE,
            )
            if int(ratio) == 4
            else (dsv4_shared_latent_sglang.DOMAIN_C128_ATTENTION_STATE,)
        )
        operations = tuple(
            SimpleNamespace(
                domain=domain,
                output_rows=SimpleNamespace(begin=0, end=len(arena)),
            )
            for domain in domains
        )
        return SimpleNamespace(
            index_arena=arena,
            positions=tuple(range(position_start, position_start + row_count)),
            forward_id=forward_id,
            operations_for_layer=lambda layer_id: operations,
        )

    def test_certificate_alone_does_not_authorize_live_prefix_state(self):
        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                workset, certificate = self._workset_and_certificate(
                    ratio=ratio,
                    request_row_begin=0,
                )
                adapter = self._adapter(
                    ratio=ratio,
                    workset=workset,
                    certificate=certificate,
                )
                with self.assertRaisesRegex(
                    ValueError, "restore schedule omits dirty island rows"
                ):
                    adapter._validate_state_receipt_schedule(
                        self._schedule(ratio=ratio)
                    )

    def test_exact_prior_receipt_authorizes_live_prefix_state(self):
        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                workset, certificate = self._workset_and_certificate(
                    ratio=ratio,
                    request_row_begin=0,
                )
                adapter = self._adapter(
                    ratio=ratio,
                    workset=workset,
                    certificate=certificate,
                    authorized=True,
                )
                adapter._validate_state_receipt_schedule(
                    self._schedule(ratio=ratio)
                )

    def test_live_prefix_authorization_rejects_wrong_start_or_forward(self):
        workset, certificate = self._workset_and_certificate(
            ratio=128,
            request_row_begin=0,
        )
        adapter = self._adapter(
            ratio=128,
            workset=workset,
            certificate=certificate,
            authorized=True,
        )
        with self.assertRaisesRegex(ValueError, "certified continuation"):
            adapter._validate_state_receipt_schedule(
                self._schedule(ratio=128, position_start=8193)
            )
        with self.assertRaisesRegex(ValueError, "another forward"):
            adapter._validate_state_receipt_schedule(
                self._schedule(ratio=128, forward_id="foreign:request:0")
            )

    def test_live_prefix_authorization_rejects_foreign_terminal_slot(self):
        workset, certificate = self._workset_and_certificate(
            ratio=128,
            request_row_begin=0,
        )
        adapter = self._adapter(
            ratio=128,
            workset=workset,
            certificate=certificate,
            authorized=True,
        )
        authorization = replace(
            adapter.live_prefix_state_authorizations[0],
            terminal_state_slots=(
                (
                    3,
                    dsv4_shared_latent_sglang.DOMAIN_C128_ATTENTION_STATE,
                    999,
                ),
            ),
        )
        adapter = replace(
            adapter,
            live_prefix_state_authorizations=(authorization,),
        )
        with self.assertRaisesRegex(
            ValueError, "restore schedule omits dirty island rows"
        ):
            adapter._validate_state_receipt_schedule(
                self._schedule(ratio=128)
            )

    def test_uncertified_prefix_state_still_fails_closed(self):
        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                workset, _ = self._workset_and_certificate(
                    ratio=ratio,
                    request_row_begin=0,
                )
                adapter = self._adapter(
                    ratio=ratio,
                    workset=workset,
                    certificate=None,
                )
                with self.assertRaisesRegex(
                    ValueError, "restore schedule omits dirty island rows"
                ):
                    adapter._validate_state_receipt_schedule(
                        self._schedule(ratio=ratio)
                    )

    def test_in_forward_restart_still_requires_state_scatter(self):
        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                workset, certificate = self._workset_and_certificate(
                    ratio=ratio,
                    request_row_begin=512,
                )
                adapter = self._adapter(
                    ratio=ratio,
                    workset=workset,
                    certificate=certificate,
                )
                with self.assertRaisesRegex(
                    ValueError, "restore schedule omits dirty island rows"
                ):
                    adapter._validate_state_receipt_schedule(
                        self._schedule(ratio=ratio)
                    )
                adapter._validate_state_receipt_schedule(
                    self._schedule(ratio=ratio, output_rows=(512,))
                )

    def test_request_one_uses_request_local_schedule_rows(self):
        first, _ = self._workset_and_certificate(
            ratio=128, request_row_begin=0
        )
        second_base, _ = self._workset_and_certificate(
            ratio=128, request_row_begin=512
        )
        second_island = replace(
            second_base.islands[0],
            flat_begin=8192 + 512,
            flat_end=8192 + 640,
        )
        second = replace(
            second_base,
            request_index=1,
            flat_row_offset=8192,
            islands=(second_island,),
        )
        restore_targets = SimpleNamespace(
            dirty_worksets=(first, second),
            dirty_state_slot_certificate=None,
            forward_token="forward-prefix-state",
        )
        adapter = dsv4_shared_latent_sglang.LayerRestoreAdapter(
            layer_id=3,
            compress_ratio=128,
            targets={},
            target_slots={},
            kernels={},
            batch_metadata={},
            restore_targets=restore_targets,
        )
        adapter._validate_state_receipt_schedule(
            self._schedule(
                ratio=128,
                output_rows=(512,),
                forward_id="forward-prefix-state:request:1",
            ),
            request_index=1,
        )

    def test_preflight_rejects_missing_receipt_before_store_call(self):
        workset, certificate = self._workset_and_certificate(
            ratio=128, request_row_begin=0
        )
        adapter = self._adapter(
            ratio=128,
            workset=workset,
            certificate=certificate,
        )
        calls = []
        store = SimpleNamespace(
            preflight_targets=lambda *args, **kwargs: calls.append((args, kwargs))
        )
        prepared = SimpleNamespace(
            pin=SimpleNamespace(validate_open=lambda: None),
            schedule=self._schedule(ratio=128),
        )
        with self.assertRaisesRegex(
            ValueError, "restore schedule omits dirty island rows"
        ):
            adapter.preflight(
                store,
                prepared,
                positions=torch.arange(8192, dtype=torch.long),
                request_index=0,
                target_slots={},
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
