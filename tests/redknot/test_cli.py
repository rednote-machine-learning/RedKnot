from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.redknot.contract import seal, write_json
from benchmarks.redknot.metrics import summarize_requests
from tests.redknot.test_compare import manifest_fixture, run_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="redknot cli ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = manifest_fixture()
        self.manifest_path = write_json(self.root / "manifest.json", self.manifest)

    def invoke(self, *arguments: str | Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "benchmarks.redknot", *map(str, arguments)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_catalog_exposes_native_and_pending_profiles(self) -> None:
        result = self.invoke("catalog")
        self.assertEqual(result.returncode, 0, result.stderr)
        catalog = json.loads(result.stdout)
        supported = [name for name, profile in catalog["models"].items() if profile["native_supported"]]
        self.assertEqual(supported, ["deepseek-v4.1-flash"])
        self.assertEqual(catalog["models"]["qwen3-32b"]["status"], "configuration_only")
        self.assertEqual(set(catalog["suites"]), {"64k", "128k", "256k", "440k"})

    def test_validate_checks_manifest_seal_and_missing_files(self) -> None:
        result = self.invoke("validate", self.manifest_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["valid"])
        self.manifest["cases"][0]["input_ids"][0] += 1
        tampered = write_json(self.root / "tampered.json", self.manifest)
        for path in (tampered, self.root / "missing.json"):
            with self.subTest(path=path):
                result = self.invoke("validate", path)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("redknot benchmark:", result.stderr)

    def test_summarize_rejects_complete_but_missing_manifest_case(self) -> None:
        report = run_fixture(self.manifest)
        report["requests"] = [row for row in report["requests"] if row["sample_id"] != "long"]
        report["completed_requests"] = len(report["requests"])
        # These requests are independently valid and their cached summary is
        # consistent; only full manifest coverage can detect the omission.
        report["summary"] = summarize_requests(report["requests"], elapsed_s=report["elapsed_s"])
        path = write_json(self.root / "incomplete.json", seal(report))
        result = self.invoke("summarize", path, "--manifest", self.manifest_path)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("missing or extra completed requests", result.stderr)

    def test_summarize_accepts_complete_cold_smoke_report(self) -> None:
        report = run_fixture(self.manifest)
        report["metadata"]["warmup"].update(iterations=0, complete=False, state="cold")
        report["comparison_eligible"] = False
        path = write_json(self.root / "cold.json", seal(report))
        result = self.invoke("summarize", path, "--manifest", self.manifest_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), report["summary"])


class WrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="redknot shell ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        self.log = self.root / "calls.jsonl"
        self.python = self.root / "fake python"
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['REDKNOT_TEST_CALLS'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        self.python.chmod(0o755)

    def invoke(self, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "/bin/bash",
                str(REPO_ROOT / "scripts" / "run_redknot_benchmarks.sh"),
                "--model-path",
                str(self.model),
                "--output-root",
                str(output),
                *arguments,
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHON_BIN": str(self.python), "REDKNOT_TEST_CALLS": str(self.log)},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_prepare_only_never_invokes_gpu_run_and_preserves_paths(self) -> None:
        result = self.invoke(self.root / "new output", "--prepare-only", "--suites", "64k 128k", "--trust-remote-code")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([call[:3] for call in calls], [["-m", "benchmarks.redknot", command] for command in ("prepare", "validate") * 2])
        for call in calls[::2]:
            self.assertEqual(Path(call[call.index("--tokenizer") + 1]).resolve(), self.model.resolve())
            self.assertIn("--trust-remote-code", call)

    def test_empty_or_invalid_suites_fail_before_any_python_action(self) -> None:
        for index, suites in enumerate(("", "   ", "invalid", "64k invalid")):
            output = self.root / f"invalid-output-{index}"
            with self.subTest(suites=suites):
                result = self.invoke(output, "--suites", suites)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.log.exists())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
