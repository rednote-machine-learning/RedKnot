"""CPU-only extraction contracts; not a model or kernel correctness certificate.

The extraction must preserve source algorithms, exclude native SGLang imports,
keep package imports free of GPU initialization, and retain representative
transaction, identity, projection, boundary, and memory-accounting contracts.
"""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from vllm_redknot.core.dsv4_context_identity import (
    NATIVE_FULL_SCOPE_POLICY,
    context_segment_sha256,
    token_ids_sha256,
)
from vllm_redknot.core.dsv4_fused_z_merge import ProjectionSpanGeometry
from vllm_redknot.core.dsv4_shared_latent_cache import (
    PACKED_LATENT_BYTES,
    DSV4SharedLatentController,
    LayerComponentSpec,
    SharedLatentSpec,
)
from vllm_redknot.core.dsv4_sparse_q import (
    build_rank_local_sparse_q_plan,
    rank_owned_logical_heads,
)
from vllm_redknot.core.native_segment_pages import NativeIndexerBucketPolicy
from vllm_redknot.core.pro0813.scale_policy import (
    flash0731_zoff_bytes_per_rank,
    pro0813_zoff_bytes_per_rank,
)
from vllm_redknot.core.v4.boundary_replay import build_boundary_replay
from vllm_redknot.core.v4.segmented_compressor import (
    build_segmented_compressor_schedule,
    validate_complete_online_row_coverage,
)

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT / "vllm_redknot" / "core"
MANIFEST = PROJECT / "docs" / "core_provenance.json"
MARKER = (
    "# REDKNOT-CORE: portable source extraction; "
    "runtime integration is tracked separately.\n"
)


class ExtractionBoundaryTest(unittest.TestCase):
    def test_manifest_covers_every_extracted_file_with_exact_sha(self):
        manifest = json.loads(MANIFEST.read_text())
        recorded = {entry["target_path"] for entry in manifest["files"]}
        actual = {str(path.relative_to(PROJECT)) for path in CORE.rglob("*.py")}
        self.assertEqual(recorded, actual)
        for entry in manifest["files"]:
            with self.subTest(path=entry["target_path"]):
                self.assertFalse(entry["runtime_integrated"])
                self.assertEqual(
                    hashlib.sha256(
                        (PROJECT / entry["target_path"]).read_bytes()
                    ).hexdigest(),
                    entry["target_sha256"],
                )

    def test_source_hash_and_mechanical_extraction_when_snapshot_available(self):
        manifest = json.loads(MANIFEST.read_text())
        configured = os.environ.get("REDKNOT_SOURCE_ROOT")
        source = (
            Path(configured)
            if configured
            else PROJECT / manifest["source_snapshot_relative_to_project"]
        )
        if not source.is_dir():
            self.skipTest("original source snapshot is not part of this installation")
        for entry in manifest["files"] + manifest["excluded"]:
            with self.subTest(source=entry["source_path"]):
                original = (source / entry["source_path"]).read_bytes()
                self.assertEqual(
                    hashlib.sha256(original).hexdigest(), entry["source_sha256"]
                )
                if entry["status"] != "portable_code_extracted_runtime_wiring_pending":
                    continue
                expected = original.decode().replace(
                    "sglang.srt.layers.attention.redknot", "vllm_redknot.core"
                )
                for name in ("dsv4_fused_z_merge", "mla_head_drift_profiler"):
                    expected = expected.replace(
                        f"from {name} import",
                        f"from vllm_redknot.core.{name} import",
                    )
                expected = MARKER + expected.rstrip() + "\n"
                self.assertEqual((PROJECT / entry["target_path"]).read_text(), expected)

    def test_import_graph_has_no_sglang_and_no_missing_core_dependencies(self):
        for path in CORE.rglob("*.py"):
            with self.subTest(path=str(path.relative_to(CORE))):
                tree = ast.parse(path.read_text(), filename=str(path))
                package_parts = path.relative_to(PROJECT).with_suffix("").parts
                package = ".".join(package_parts[:-1])
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        name = node.module or ""
                        if node.level:
                            name = importlib.util.resolve_name(
                                "." * node.level + name, package
                            )
                        names = [name]
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                        # Also reject a dynamic import string pointing at SGLang.
                        if node.value.startswith(("sglang.", "sgl_kernel.")):
                            self.fail(
                                f"native engine reference in {path}:{node.lineno}"
                            )
                    for name in names:
                        self.assertNotIn(name.split(".")[0], ("sglang", "sgl_kernel"))
                        if name.startswith("vllm_redknot.core"):
                            module = PROJECT.joinpath(*name.split("."))
                            self.assertTrue(
                                module.with_suffix(".py").is_file()
                                or (module / "__init__.py").is_file(),
                                f"missing extracted dependency {name}",
                            )

    def test_lightweight_packages_and_cpu_contracts_without_tensor_runtimes(self):
        modules = (
            "",
            ".pro0813",
            ".v4",
            ".segpaged_v2",
            ".dsv4_composite_commit",
            ".dsv4_context_identity",
            ".dsv4_fused_z_merge",
            ".dsv4_reuse_batch",
            ".dsv4_shared_latent_cache",
            ".dsv4_shared_latent_gpu",
            ".dsv4_shared_snapshot_runtime",
            ".dsv4_sparse_q",
            ".dsv4_sparse_q_runtime",
            ".eval_harness",
            ".native_segment_pages",
            ".pro0813.profile",
            ".pro0813.scale_policy",
            ".v4.boundary_replay",
            ".v4.compatibility",
            ".v4.merged_prefill",
            ".v4.request_selector",
            ".v4.segmented_compressor",
            ".v4.state_composer",
            ".v4.types",
        )
        code = """
import importlib, importlib.abc, sys
blocked = {'torch', 'triton', 'sglang', 'sgl_kernel', 'vllm', 'flash_attn'}
class BlockTensorRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockTensorRuntime())
for suffix in MODULES:
    importlib.import_module('vllm_redknot.core' + suffix)
assert not any(name.split('.')[0] in blocked for name in sys.modules)
""".replace("MODULES", repr(modules))
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=PROJECT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
        "Torch/Triton are optional and absent on controller-only hosts",
    )
    def test_all_modules_import_without_creating_a_cuda_context(self):
        modules = tuple(
            ".".join(path.relative_to(PROJECT).with_suffix("").parts)
            for path in CORE.rglob("*.py")
            if path.name != "__init__.py"
        )
        code = """
import importlib, importlib.abc, sys, torch
assert not torch.cuda.is_initialized()
class NoNativeSGLang(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sglang', 'sgl_kernel'}:
            raise AssertionError('native SGLang import attempted: ' + fullname)
sys.meta_path.insert(0, NoNativeSGLang())
for name in MODULES:
    importlib.import_module(name)
assert not torch.cuda.is_initialized(), 'import initialized CUDA'
""".replace("MODULES", repr(modules))
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=PROJECT,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class PortableAlgorithmContractTest(unittest.TestCase):
    def test_context_identity_changes_for_prefix_not_just_chunk_body(self):
        digest = token_ids_sha256((100, 200))
        kwargs = {
            "execution_profile": "context-bound-test",
            "head_scope_policy": NATIVE_FULL_SCOPE_POLICY,
            "model_compat_hash": "a" * 64,
            "head_policy_hash": "b" * 64,
            "token_hash": digest,
            "prefix_input_hash": token_ids_sha256((1,)),
            "full_input_hash": token_ids_sha256((1, 100, 200)),
            "source_start": 1,
            "source_end": 3,
            "length": 2,
            "canonical_start_pos": 0,
        }
        first = context_segment_sha256(**kwargs)
        kwargs["prefix_input_hash"] = token_ids_sha256((2,))
        kwargs["full_input_hash"] = token_ids_sha256((2, 100, 200))
        self.assertNotEqual(first, context_segment_sha256(**kwargs))
        with self.assertRaises(TypeError):
            token_ids_sha256((True,))

    def test_sparse_q_partition_conserves_head_rows_without_claiming_flops(self):
        plan = build_rank_local_sparse_q_plan(
            layer_id=3,
            tp_rank=2,
            tp_size=8,
            q_rows=100,
            offline_local_logical_heads=(17, 18, 19, 20, 21, 22, 23),
            online_local_rows=(98, 99),
        )
        self.assertEqual(plan.owned_logical_heads, tuple(range(16, 24)))
        self.assertEqual(plan.global_head_axes, (0,))
        self.assertEqual(plan.projected_head_rows, 114)
        self.assertEqual(plan.omitted_head_rows, 686)
        self.assertEqual(plan.full_owned_head_rows, 800)
        plan.validate()
        with self.assertRaises(ValueError):
            rank_owned_logical_heads(total_logical_heads=65, tp_size=8, tp_rank=0)

    def test_native_bucket_caps_documents_but_not_online_suffix(self):
        policy = NativeIndexerBucketPolicy(
            documents=2,
            document_compressed_rows=4,
            per_document_cap=1,
            indexer_topk=6,
        )
        self.assertEqual(
            policy.retain_reference((0, 1, 4, 5, 8, 9, 10, -1)),
            (0, 4, 8, 9, 10),
        )

    def test_boundary_replay_keeps_prefix_query_and_chunk_boundaries_online(self):
        plan = build_boundary_replay(
            segments=(
                {"global_offset": 128, "length": 256},
                {"global_offset": 384, "length": 256},
            ),
            total_tokens=650,
        )
        self.assertEqual(plan.online_prefix_range, (0, 128))
        self.assertEqual(plan.online_query_range, (640, 650))
        self.assertEqual(plan.online_token_count, 394)
        self.assertEqual(plan.segments[1].offline_token_range, (512, 640))
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_boundary_replay(
                segments=(
                    {"global_offset": 0, "length": 256},
                    {"global_offset": 128, "length": 256},
                ),
                total_tokens=384,
            )

    def test_projection_span_rejects_invalid_or_out_of_bounds_rows(self):
        span = ProjectionSpanGeometry((10, 11), (0, 1))
        self.assertTrue(span.is_unit_stride)
        span.validate(total_rows=12, segment_rows=2)
        with self.assertRaises(ValueError):
            span.validate(total_rows=11, segment_rows=2)
        with self.assertRaises(ValueError):
            ProjectionSpanGeometry((10, 10), (0, 1))

    def test_segmented_compressor_splits_sparse_positions_without_lost_rows(self):
        replay = build_boundary_replay(segments=(), total_tokens=20)
        schedule = build_segmented_compressor_schedule(
            replay=replay,
            positions=(1, 2, 9, 10),
            compress_ratio=4,
            include_all_present_rows=True,
        )
        self.assertEqual(
            tuple((event.token_begin, event.token_end) for event in schedule.events),
            ((1, 3), (9, 11)),
        )
        self.assertEqual(schedule.online_rows, 4)
        validate_complete_online_row_coverage(schedule, total_rows=4)

    def test_persistent_layout_reports_exact_banked_bytes_without_allocation(self):
        from vllm_redknot.core.dsv4_shared_latent_gpu import (
            build_shared_latent_device_layout,
            shared_latent_device_nbytes,
        )

        spec = SharedLatentSpec(
            model_hash="model",
            policy_hash="policy",
            length=2,
            layers=(LayerComponentSpec(layer_id=3, compress_ratio=0),),
            required_layer_ids=(3,),
        )
        layout = build_shared_latent_device_layout(spec)
        self.assertEqual(layout.bytes_per_segment, 2 * PACKED_LATENT_BYTES)
        self.assertEqual(
            shared_latent_device_nbytes(layout, segment_epoch_capacity=3),
            6 * PACKED_LATENT_BYTES,
        )

    def test_pro_storage_scales_layers_and_groups_not_checkpoint_bytes(self):
        tokens = 128
        self.assertEqual(
            pro0813_zoff_bytes_per_rank(tokens), 55 * tokens * 2 * 1024 * 2
        )
        self.assertEqual(flash0731_zoff_bytes_per_rank(tokens), 37 * tokens * 1024 * 2)

    def test_shared_latent_publish_is_complete_immutable_and_rollback_capable(self):
        controller = DSV4SharedLatentController()
        spec = SharedLatentSpec(
            model_hash="model",
            policy_hash="policy",
            length=2,
            layers=(LayerComponentSpec(layer_id=3, compress_ratio=0),),
            required_layer_ids=(3,),
        )
        kwargs = {"seg_hash": "segment", "generation_id": "first"}
        controller.begin_capture(**kwargs, token_ids=(10, 11), spec=spec)
        controller.capture_swa_rows(
            **kwargs,
            layer_id=3,
            local_rows=(0,),
            positionless_packed=b"a" * PACKED_LATENT_BYTES,
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            controller.commit_capture(**kwargs)
        with self.assertRaises(KeyError):
            controller.get_committed("segment")
        controller.capture_swa_rows(
            **kwargs,
            layer_id=3,
            local_rows=(1,),
            positionless_packed=b"b" * PACKED_LATENT_BYTES,
        )
        old = controller.commit_capture(**kwargs)
        with self.assertRaises(TypeError):
            old.layers[4] = old.layers[3]
        kwargs["generation_id"] = "second"
        controller.begin_capture(**kwargs, token_ids=(10, 11), spec=spec)
        controller.capture_swa_rows(
            **kwargs,
            layer_id=3,
            local_rows=(0, 1),
            positionless_packed=b"c" * (2 * PACKED_LATENT_BYTES),
        )
        receipt = controller.publish_capture(**kwargs)
        self.assertGreater(receipt.artifact.commit_epoch, old.commit_epoch)
        controller.rollback_publish(receipt)
        self.assertIs(controller.get_committed("segment"), old)
        with self.assertRaises(ValueError):
            controller.confirm_publish(receipt)


if __name__ == "__main__":
    unittest.main()
