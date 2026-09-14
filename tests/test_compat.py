"""Source-hash contract: detect missing/drifting APIs before runtime patching."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_redknot.compat import verify_vllm_sources


class CompatibilityTests(unittest.TestCase):
    def test_family_checks_only_its_own_source_contract(self):
        digest = hashlib.sha256(b"verified").hexdigest()
        manifest = {
            "commit": "fixture",
            "files": {"mha.py": digest},
            "family_files": {"deepseek_v4_flash": {"dsv4.py": digest}},
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "dsv4.py").write_bytes(b"verified")
            with patch("vllm_redknot.compat.json.loads", return_value=manifest):
                result = verify_vllm_sources(
                    directory, engine_family="deepseek_v4_flash"
                )
                self.assertEqual(result["engine_family"], "deepseek_v4_flash")
                with self.assertRaisesRegex(RuntimeError, "mha.py: missing"):
                    verify_vllm_sources(directory)
                Path(directory, "dsv4.py").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "dsv4.py: hash mismatch"):
                    verify_vllm_sources(directory, engine_family="deepseek_v4_flash")

    def test_dsv4_without_family_manifest_does_not_fall_back_to_mha(self):
        with patch("vllm_redknot.compat.json.loads", return_value={"files": {}}):
            with self.assertRaisesRegex(RuntimeError, "No source contract"):
                verify_vllm_sources("/unused", engine_family="deepseek_v4_flash")

    def test_unknown_family_is_rejected_before_import_discovery(self):
        with patch("vllm_redknot.compat.importlib.util.find_spec") as discover:
            for family in ("dsv4", None, []):
                with self.subTest(family=family), self.assertRaises(ValueError):
                    verify_vllm_sources(engine_family=family)
            discover.assert_not_called()

    def test_dsv4_manifest_covers_native_state_and_built_kernel_interface(self):
        import json

        manifest = json.loads(
            (Path(__file__).parents[1] / "vllm_redknot/pinned_files.json").read_text()
        )
        files = manifest["family_files"]["deepseek_v4_flash"]
        required = (
            "models/deepseek_v4/attention.py",
            "models/deepseek_v4/nvidia/flashmla.py",
            "models/deepseek_v4/nvidia/ops/o_proj.py",
            "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
            "models/deepseek_v4/common/rope.py",
            "models/deepseek_v4/compressor.py",
            "models/deepseek_v4/sparse_mla.py",
            "models/deepseek_v4/nvidia/model.py",
            "v1/attention/backends/mla/sparse_swa.py",
            "v1/attention/ops/flashmla.py",
            "third_party/flashmla/flash_mla_interface.py",
        )
        for name in required:
            self.assertRegex(files[name], r"^[0-9a-f]{64}$")
        self.assertNotIn("model_executor/models/qwen2.py", files)
        self.assertNotIn("v1/attention/backends/flash_attn.py", files)

    def test_exact_hash_is_required(self):
        manifest = {
            "commit": "fixture",
            "files": {"module.py": hashlib.sha256(b"verified").hexdigest()},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "module.py"
            with patch("vllm_redknot.compat.json.loads", return_value=manifest):
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    verify_vllm_sources(directory)
                path.write_bytes(b"verified")
                self.assertEqual(
                    verify_vllm_sources(directory)["status"], "SOURCE_CONTRACT_OK"
                )
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    verify_vllm_sources(directory)


if __name__ == "__main__":
    unittest.main()
