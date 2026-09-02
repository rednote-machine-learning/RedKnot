"""CPU-only contracts for the stable DeepSeek-V4-Pro-0813 CLI alias."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


HERE = Path(__file__).resolve().parent
ENTRYPOINT = HERE / "benchmark-redknot-deepseekv4-pro.py"


def _load_entrypoint() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_redknot_deepseek_v4_pro_stable_cli_test", ENTRYPOINT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDeepSeekV4ProStandaloneEntrypoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entry = _load_entrypoint()

    def test_entrypoint_is_executable_and_pro_only(self) -> None:
        self.assertTrue(os.access(ENTRYPOINT, os.X_OK))
        source = ENTRYPOINT.read_text(encoding="utf-8")
        self.assertLess(len(source.encode("utf-8")), 12 * 1024)
        self.assertIn("benchmark_dsv4_pro0813_redknot_http.py", source)
        self.assertNotIn("benchmark_RedKnot_DeepSeekV4_Flash_RAG.py", source)

    def test_identity_pin_is_exactly_pro0813(self) -> None:
        self.assertEqual(
            self.entry.PINNED_MODEL,
            Path("/workspace/Models/DeepSeek-V4-Pro-0813"),
        )
        self.assertEqual(self.entry.PINNED_VARIANT, "deepseek_v4_pro_0813")
        self.assertEqual(
            self.entry.PINNED_CONFIG_SHA256,
            "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0",
        )
        self.assertNotIn("0831", str(self.entry.PINNED_MODEL))

    def test_real_canonical_entrypoint_satisfies_wrapper_identity(self) -> None:
        canonical = self.entry._load_canonical()
        self.entry._validate_canonical_identity(canonical)

    def test_main_delegates_once_to_canonical_main(self) -> None:
        canonical = self.entry._load_canonical()
        canonical.main = mock.Mock()
        with mock.patch.object(
            self.entry, "_load_canonical", return_value=canonical
        ):
            self.entry.main()
        canonical.main.assert_called_once_with()

    def test_every_identity_field_drift_fails_before_delegation(self) -> None:
        for field, _expected in self.entry._IDENTITY_FIELDS:
            with self.subTest(field=field):
                canonical = self.entry._load_canonical()
                canonical.main = mock.Mock()
                setattr(canonical, field, object())
                with mock.patch.object(
                    self.entry, "_load_canonical", return_value=canonical
                ), self.assertRaises(self.entry.CanonicalEntrypointError):
                    self.entry.main()
                canonical.main.assert_not_called()

    def test_canonical_module_source_drift_fails_before_delegation(self) -> None:
        canonical = self.entry._load_canonical()
        canonical.main = mock.Mock()
        canonical.__file__ = str(HERE / "lookalike_pro_entrypoint.py")
        with mock.patch.object(
            self.entry, "_load_canonical", return_value=canonical
        ), self.assertRaises(self.entry.CanonicalEntrypointError):
            self.entry.main()
        canonical.main.assert_not_called()

    def test_help_is_served_by_canonical_safe_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(ENTRYPOINT), "--help"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("DeepSeek-V4-Pro-0813", completed.stdout)
        self.assertIn("--contract-only", completed.stdout)
        self.assertIn("--combined-headsplit-row-sparse", completed.stdout)
        self.assertNotIn("Traceback", completed.stderr)

    def test_contract_only_reports_the_pinned_pro0813_identity(self) -> None:
        config = HERE / "deepseek_v4_pro0813_config.json"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ENTRYPOINT),
                    "--contract-only",
                    "--contract-config",
                    str(config),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        contract = json.loads(completed.stdout)
        self.assertEqual(contract["variant"], "deepseek_v4_pro_0813")
        self.assertEqual(
            contract["official_config_sha256"], self.entry.PINNED_CONFIG_SHA256
        )
        self.assertEqual(contract["geometry_digest"], self.entry.PINNED_GEOMETRY_DIGEST)
        self.assertEqual(contract["tp_size"], 8)
        self.assertEqual(contract["num_layers"], 61)
        self.assertEqual(contract["num_attention_heads"], 128)
        self.assertEqual(contract["index_topk"], 1024)

    def test_unknown_option_is_rejected_before_benchmark_execution(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ENTRYPOINT), "--not-a-real-option"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unrecognized arguments", completed.stderr)


if __name__ == "__main__":
    unittest.main()
