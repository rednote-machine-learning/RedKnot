from __future__ import annotations

import dataclasses
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).with_name("verify_pro0813_official_model.py")
SPEC = importlib.util.spec_from_file_location("verify_pro0813_official_model", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: object) -> bytes:
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _fixture(tmp_path: Path) -> tuple[GATE.GatePolicy, dict[str, object]]:
    target = tmp_path / "data" / "DeepSeek-V4-Pro-0813"
    target.mkdir(parents=True)
    # macOS exposes /tmp and /var through /private; mirror the production
    # contract by storing the target's canonical absolute path in the policy.
    target = target.resolve(strict=True)
    link = tmp_path / "workspace" / "DeepSeek-V4-Pro-0813"
    link.parent.mkdir()
    link.symlink_to(target)

    config = b'{"model_type":"deepseek_v4"}\n'
    index = b'{"metadata":{"total_size":7},"weight_map":{}}\n'
    shard = b"weights"
    files = {
        "config.json": {"size": len(config), "sha256": None},
        "model.safetensors.index.json": {"size": len(index), "sha256": None},
        "model-00001-of-00001.safetensors": {
            "size": len(shard),
            "sha256": _sha256(shard),
        },
    }
    for name, data in {
        "config.json": config,
        "model.safetensors.index.json": index,
        "model-00001-of-00001.safetensors": shard,
    }.items():
        (target / name).write_bytes(data)

    revision = "a" * 40
    manifest: dict[str, object] = {
        "repo_id": "test/DeepSeek-V4-Pro-0813",
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files.values()),
        "files": files,
    }
    manifest_raw = _write_json(target / GATE.MANIFEST_NAME, manifest)
    status_path = tmp_path / "data" / "pro0813_download_status.json"
    _write_json(
        status_path,
        {
            "status": "complete",
            "revision": revision,
            "file_count": len(files),
            "total_bytes": manifest["total_bytes"],
            "path": str(target),
        },
    )
    policy = GATE.GatePolicy(
        model_link=link,
        model_target=target,
        status_path=status_path,
        repo_id=manifest["repo_id"],
        revision=revision,
        manifest_sha256=_sha256(manifest_raw),
        expected_file_count=len(files),
        expected_shard_count=1,
        expected_total_bytes=manifest["total_bytes"],
        config_sha256=_sha256(config),
        index_sha256=_sha256(index),
    )
    return policy, manifest


def _rewrite_manifest(
    policy: GATE.GatePolicy, manifest: dict[str, object]
) -> GATE.GatePolicy:
    raw = _write_json(policy.model_target / GATE.MANIFEST_NAME, manifest)
    return dataclasses.replace(policy, manifest_sha256=_sha256(raw))


class OfficialModelGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_complete_fixture_passes(self) -> None:
        policy, _ = _fixture(self.root)
        GATE.validate_model_layout(policy)

    def test_model_path_must_itself_be_symlink(self) -> None:
        policy, _ = _fixture(self.root)
        policy.model_link.unlink()
        policy.model_link.mkdir()
        with self.assertRaises(GATE.ModelLinkError):
            GATE.validate_model_layout(policy)

    def test_model_link_must_resolve_to_exact_target(self) -> None:
        policy, _ = _fixture(self.root)
        other = self.root / "other"
        other.mkdir()
        policy.model_link.unlink()
        policy.model_link.symlink_to(other)
        with self.assertRaises(GATE.ModelLinkError):
            GATE.validate_model_layout(policy)

    def test_manifest_hash_is_pinned(self) -> None:
        policy, manifest = _fixture(self.root)
        manifest["extra"] = "mutation"
        _write_json(policy.model_target / GATE.MANIFEST_NAME, manifest)
        with self.assertRaises(GATE.ManifestHashError):
            GATE.validate_model_layout(policy)

    def test_manifest_summary_fields_are_exact(self) -> None:
        cases = (
            ("repo_id", "wrong/repo"),
            ("revision", "b" * 40),
            ("file_count", 4),
            ("file_count", 3.0),
            ("total_bytes", 999),
        )
        for index, (field, value) in enumerate(cases):
            with self.subTest(field=field):
                case_root = self.root / str(index)
                policy, manifest = _fixture(case_root)
                manifest[field] = value
                policy = _rewrite_manifest(policy, manifest)
                with self.assertRaises(GATE.ManifestSchemaError):
                    GATE.validate_model_layout(policy)

    def test_manifest_path_traversal_is_rejected(self) -> None:
        unsafe_names = ("../escape", "/absolute", "a/../b", "a\\b")
        for index, unsafe_name in enumerate(unsafe_names):
            with self.subTest(name=unsafe_name):
                case_root = self.root / str(index)
                policy, manifest = _fixture(case_root)
                files = manifest["files"]
                self.assertIsInstance(files, dict)
                metadata = files.pop("model-00001-of-00001.safetensors")
                files[unsafe_name] = metadata
                policy = _rewrite_manifest(policy, manifest)
                with self.assertRaises(GATE.UnsafeManifestPathError):
                    GATE.validate_model_layout(policy)

    def test_missing_manifest_target_fails(self) -> None:
        policy, _ = _fixture(self.root)
        (policy.model_target / "model-00001-of-00001.safetensors").unlink()
        with self.assertRaises(GATE.ModelFileError):
            GATE.validate_model_layout(policy)

    def test_manifest_target_must_not_be_symlink(self) -> None:
        policy, _ = _fixture(self.root)
        target = policy.model_target / "model-00001-of-00001.safetensors"
        target.unlink()
        target.symlink_to(policy.model_target / "config.json")
        with self.assertRaises(GATE.ModelFileError):
            GATE.validate_model_layout(policy)

    def test_manifest_target_size_is_exact(self) -> None:
        policy, _ = _fixture(self.root)
        target = policy.model_target / "model-00001-of-00001.safetensors"
        target.write_bytes(b"short")
        with self.assertRaises(GATE.ModelFileSizeError):
            GATE.validate_model_layout(policy)

    def test_config_hash_is_pinned(self) -> None:
        policy, manifest = _fixture(self.root)
        changed = b'{"model_type":"tampered!!"}\n'
        (policy.model_target / "config.json").write_bytes(changed)
        files = manifest["files"]
        self.assertIsInstance(files, dict)
        files["config.json"]["size"] = len(changed)
        manifest["total_bytes"] = sum(item["size"] for item in files.values())
        policy = dataclasses.replace(
            policy, expected_total_bytes=manifest["total_bytes"]
        )
        policy = _rewrite_manifest(policy, manifest)
        status = json.loads(policy.status_path.read_text())
        status["total_bytes"] = policy.expected_total_bytes
        _write_json(policy.status_path, status)
        with self.assertRaises(GATE.ConfigHashError):
            GATE.validate_model_layout(policy)

    def test_index_hash_is_pinned(self) -> None:
        policy, manifest = _fixture(self.root)
        changed = b'{"metadata":{"total_size":8},"weight_map":{}}\n'
        (policy.model_target / "model.safetensors.index.json").write_bytes(changed)
        files = manifest["files"]
        self.assertIsInstance(files, dict)
        files["model.safetensors.index.json"]["size"] = len(changed)
        manifest["total_bytes"] = sum(item["size"] for item in files.values())
        policy = dataclasses.replace(
            policy, expected_total_bytes=manifest["total_bytes"]
        )
        policy = _rewrite_manifest(policy, manifest)
        status = json.loads(policy.status_path.read_text())
        status["total_bytes"] = policy.expected_total_bytes
        _write_json(policy.status_path, status)
        with self.assertRaises(GATE.IndexHashError):
            GATE.validate_model_layout(policy)

    def test_completion_status_fields_are_exact(self) -> None:
        cases = (
            ("status", "hashing"),
            ("revision", "b" * 40),
            ("file_count", 2),
            ("file_count", 3.0),
            ("total_bytes", 1),
            ("path", "/wrong/model"),
        )
        for index, (field, value) in enumerate(cases):
            with self.subTest(field=field):
                case_root = self.root / str(index)
                policy, _ = _fixture(case_root)
                status = json.loads(policy.status_path.read_text())
                status[field] = value
                _write_json(policy.status_path, status)
                with self.assertRaises(GATE.StatusSchemaError):
                    GATE.validate_model_layout(policy)

    def test_completion_status_must_not_be_symlink(self) -> None:
        policy, _ = _fixture(self.root)
        alternate = self.root / "alternate-status.json"
        alternate.write_bytes(policy.status_path.read_bytes())
        policy.status_path.unlink()
        policy.status_path.symlink_to(alternate)
        with self.assertRaises(GATE.StatusFileError):
            GATE.validate_model_layout(policy)

    def test_manifest_must_not_be_symlink(self) -> None:
        policy, _ = _fixture(self.root)
        manifest_path = policy.model_target / GATE.MANIFEST_NAME
        alternate = self.root / "alternate-manifest.json"
        alternate.write_bytes(manifest_path.read_bytes())
        manifest_path.unlink()
        manifest_path.symlink_to(alternate)
        with self.assertRaises(GATE.ManifestFileError):
            GATE.validate_model_layout(policy)

    def test_production_policy_is_fully_pinned(self) -> None:
        policy = GATE.PRODUCTION_POLICY
        self.assertEqual(
            policy.model_link, Path("/workspace/Models/DeepSeek-V4-Pro-0813")
        )
        self.assertEqual(
            policy.model_target, Path("/data/temp/Models/DeepSeek-V4-Pro-0813")
        )
        self.assertEqual(
            policy.status_path,
            Path("/data/temp/Models/pro0813_download_status.json"),
        )
        self.assertEqual(policy.repo_id, "deepseek-ai/DeepSeek-V4-Pro-0813")
        self.assertEqual(
            policy.revision, "72e1d3230f6c080a530b0a1d46f8eb4602340597"
        )
        self.assertEqual(
            policy.manifest_sha256,
            "27c3ef953c3baeb1a69f8ba0c6fd8c55814b1c89a4aa768ff28486c27bebb1f5",
        )
        self.assertEqual(policy.expected_file_count, 92)
        self.assertEqual(policy.expected_shard_count, 66)
        self.assertEqual(policy.expected_total_bytes, 892_762_497_859)
        self.assertEqual(
            policy.config_sha256,
            "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0",
        )
        self.assertEqual(
            policy.index_sha256,
            "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a",
        )

    def test_main_uses_only_production_policy_and_emits_safe_json(self) -> None:
        observed: list[object] = []
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"REDKNOT_MODEL_PATH": "/unsafe"}), mock.patch.object(
            sys, "argv", [str(MODULE_PATH), "--model", "/unsafe"]
        ), mock.patch.object(GATE, "validate_model_layout", observed.append), contextlib.redirect_stdout(
            output
        ):
            self.assertEqual(GATE.main(), 0)
        self.assertEqual(observed, [GATE.PRODUCTION_POLICY])
        self.assertEqual(json.loads(output.getvalue()), {"status": "pass"})

    def test_main_failure_json_does_not_expose_exception_message(self) -> None:
        def fail(_policy: GATE.GatePolicy) -> None:
            raise GATE.ManifestSchemaError("signed-url-or-secret")

        output = io.StringIO()
        with mock.patch.object(
            GATE, "validate_model_layout", fail
        ), contextlib.redirect_stdout(output):
            self.assertEqual(GATE.main(), 1)
        self.assertNotIn("signed-url-or-secret", output.getvalue())
        self.assertEqual(
            json.loads(output.getvalue()),
            {"error_type": "ManifestSchemaError", "status": "fail"},
        )


if __name__ == "__main__":
    unittest.main()
