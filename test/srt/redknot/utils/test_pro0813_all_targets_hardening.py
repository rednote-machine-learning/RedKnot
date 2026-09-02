"""CPU/static tests for formal asset preflight and five-target collection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ASSET_VERIFIER = HERE / "verify_pro0813_formal_assets.py"
SUMMARY_WRITER = HERE / "write_pro0813_all_targets_summary.py"
ALL_TARGETS = HERE / "run_deepseek_v4_pro0813_all_targets.sh"
ONE_TARGET = HERE / "run_deepseek_v4_pro0813_reproduction.sh"
SUPERVISOR = HERE / "run_pro0813_combined_supervisor.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestFormalAssetVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = _load(ASSET_VERIFIER, "_pro0813_asset_hardening_test")

    def _fixture(self, root: Path):
        data_dir = root / "data"
        data_dir.mkdir()
        dataset = data_dir / "musique.jsonl"
        dataset.write_bytes(
            b'{"input":"question","context":"context","answers":["answer"]}\n'
        )
        dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        manifest = {
            "format": "redknot_indexer_hot_data_selection_v1",
            "dataset": {
                "name": "musique",
                "bytes": dataset.stat().st_size,
                "sha256": dataset_digest,
                "row_id_base": 0,
            },
            "geometry": {
                "chunk_tokens": 8192,
                "num_chunks": 8,
                "num_queries": 1,
            },
            "selection": {
                "mode": "offset",
                "row_offset": 0,
                "excluded_row_ids": [],
                "excluded_selection_sha256": [],
                "chunks": [],
                "queries": [{"query_index": 0, "row_id": 68}],
            },
        }
        digest = self.assets._canonical_sha256(manifest)
        manifest["selection_sha256"] = digest
        manifest_path = root / "selection.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        target_profile = {
            "selection_sha256": digest,
            "data_manifest": manifest_path,
            "chunk_tokens": 8192,
            "num_chunks": 8,
            "query_row_id": 68,
        }
        return data_dir, dataset, manifest_path, target_profile

    def test_short_target_closes_manifest_and_whole_dataset_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, dataset, _, target_profile = self._fixture(root)
            result = self.assets._verify_short_target_asset(
                65536, target_profile, data_dir=data_dir
            )
            self.assertTrue(result["pass"])
            self.assertEqual(result["selection_sha256"], target_profile["selection_sha256"])
            self.assertEqual(
                result["dataset_sha256"],
                hashlib.sha256(dataset.read_bytes()).hexdigest(),
            )

    def test_short_target_rejects_manifest_semantic_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, _, manifest_path, target_profile = self._fixture(root)
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            document["geometry"]["num_chunks"] = 7
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                self.assets.FormalAssetVerificationError, "canonical digest"
            ):
                self.assets._verify_short_target_asset(
                    65536, target_profile, data_dir=data_dir
                )

    def test_short_target_rejects_dataset_byte_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, dataset, _, target_profile = self._fixture(root)
            dataset.write_bytes(dataset.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(
                self.assets.FormalAssetVerificationError, "byte identity differs"
            ):
                self.assets._verify_short_target_asset(
                    65536, target_profile, data_dir=data_dir
                )

    def test_short_target_rejects_symlink_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, _, manifest_path, target_profile = self._fixture(root)
            real_manifest = root / "real-selection.json"
            manifest_path.rename(real_manifest)
            manifest_path.symlink_to(real_manifest)
            with self.assertRaisesRegex(
                self.assets.FormalAssetVerificationError, "non-symlink"
            ):
                self.assets._verify_short_target_asset(
                    65536, target_profile, data_dir=data_dir
                )

    def test_canonical_long_profile_byte_pins_match_tree(self):
        self.assertEqual(set(self.assets.CANONICAL_LONG_PROFILES), {450560, 524288})
        for profile_path, expected_digest in self.assets.CANONICAL_LONG_PROFILES.values():
            with self.subTest(profile=profile_path):
                self.assertTrue(profile_path.is_file())
                self.assertFalse(profile_path.is_symlink())
                self.assertEqual(
                    hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                    expected_digest,
                )

    def test_three_short_target_identity_roots_are_literal_and_complete(self):
        self.assertEqual(
            set(self.assets.CANONICAL_SHORT_TARGETS), {65536, 131072, 262144}
        )
        expected = {
            65536: (
                "musique_pure_prompt_selection_v1.json",
                "586fd683bfe043e1a6aaa1d07c7236ea9d956d99be739be743c4a2ec1728bcd8",
            ),
            131072: (
                "musique_pure_prompt_selection_128k_v1.json",
                "caf99890880e0de190f845d0a38e600d760d2153cd1961888bd7776a2044f040",
            ),
            262144: (
                "musique_pure_prompt_selection_256k_32k_v1.json",
                "a2524b87a6ff0a91e7f5aef104d3b8eb14b9aa55e2b8c6b5db34ef0dbe1477cc",
            ),
        }
        for target_tokens, (name, digest) in expected.items():
            with self.subTest(target_tokens=target_tokens):
                record = self.assets.CANONICAL_SHORT_TARGETS[target_tokens]
                self.assertEqual(record["manifest"].name, name)
                self.assertEqual(record["selection_sha256"], digest)


class TestAllTargetsSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = _load(SUMMARY_WRITER, "_pro0813_summary_hardening_test")

    @staticmethod
    def _restored_line(pid: int) -> str:
        return (
            f"holder_restarted pid={pid} workers=8 util_threshold_pct=90 "
            "util_good_samples=3/15 mapping=0:GPU-a:100 1:GPU-b:101\n"
        )

    def _root_with_receipts(self, root: Path) -> None:
        for index, (label, _) in enumerate(self.summary.TARGETS, start=1):
            run_dir = root / label
            run_dir.mkdir()
            (run_dir / "holder_restore_status").write_text(
                self._restored_line(1000 + index), encoding="utf-8"
            )

    def test_all_success_requires_five_supervisor_restore_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self._root_with_receipts(root)
            result = self.summary.build_summary(root, [0, 0, 0, 0, 0])
            self.assertTrue(result["pass"])
            self.assertEqual(result["overall_exit_code"], 0)
            self.assertEqual(len(result["targets"]), 5)
            self.assertTrue(all(item["holder"]["proven"] for item in result["targets"]))

    def test_first_failure_is_recorded_but_later_targets_remain_collectable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self._root_with_receipts(root)
            result = self.summary.build_summary(root, [17, 0, 23, 0, 0])
            self.assertFalse(result["pass"])
            self.assertEqual(result["overall_exit_code"], 17)
            self.assertEqual(
                [item["exit_code"] for item in result["targets"]],
                [17, 0, 23, 0, 0],
            )
            self.assertTrue(result["continued_after_target_failure"])

    def test_zero_exit_without_restore_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self._root_with_receipts(root)
            (root / "256k/holder_restore_status").unlink()
            result = self.summary.build_summary(root, [0, 0, 0, 0, 0])
            self.assertFalse(result["pass"])
            self.assertEqual(result["overall_exit_code"], 74)
            self.assertEqual(result["targets"][2]["holder"]["state"], "no_handoff_receipt")

    def test_summary_publication_is_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self._root_with_receipts(root)
            result = self.summary.build_summary(root, [0, 0, 0, 0, 0])
            destination = root / "all_targets_summary.json"
            self.summary.write_summary(destination, result)
            parsed = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(parsed, result)
            with self.assertRaises(FileExistsError):
                self.summary.write_summary(destination, result)


class TestShellIntegration(unittest.TestCase):
    def test_shell_entrypoints_parse(self):
        for script in (ALL_TARGETS, ONE_TARGET, SUPERVISOR):
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_asset_preflight_precedes_every_holder_lookup(self):
        one_target = ONE_TARGET.read_text(encoding="utf-8")
        self.assertLess(
            one_target.index("verify_pro0813_formal_assets.py"),
            one_target.index("pgrep -f '[g]pu_hold.py'"),
        )
        all_targets = ALL_TARGETS.read_text(encoding="utf-8")
        self.assertLess(
            all_targets.index('"$runtime_python" "$formal_asset_verifier"'),
            all_targets.index('mkdir -p "$root_run_dir"'),
        )
        supervisor = SUPERVISOR.read_text(encoding="utf-8")
        self.assertLess(
            supervisor.index('"$runtime_python" "$formal_asset_verifier"'),
            supervisor.index('if [[ ! -r "/proc/$holder_pid/cmdline" ]]'),
        )

    def test_target_commands_are_guarded_and_summary_controls_exit(self):
        source = ALL_TARGETS.read_text(encoding="utf-8")
        self.assertEqual(source.count('if env "${formal_unset_args[@]}"'), 3)
        self.assertIn('target_exit_codes+=("$target_exit_code")', source)
        self.assertIn('"$runtime_python" "$summary_writer"', source)
        self.assertIn('exit "$overall_exit_code"', source)

    def test_real_sweep_continues_after_failures_and_returns_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script_dir = root / "redknot"
            script_dir.mkdir()
            sequence = script_dir / ALL_TARGETS.name
            sequence.write_text(
                ALL_TARGETS.read_text(encoding="utf-8").replace(
                    "runtime_python=/workspace/RedKnot/.venv_sm103/bin/python",
                    f"runtime_python={shlex.quote(sys.executable)}",
                ),
                encoding="utf-8",
            )
            sequence.chmod(0o755)
            (script_dir / "verify_pro0813_qualification_profile.py").write_text(
                "# fixture: readability is the contract exercised here\n",
                encoding="utf-8",
            )
            (script_dir / "verify_pro0813_formal_assets.py").write_text(
                "import json\n"
                "print(json.dumps({'pass': True, 'assets': ["
                "{'target_tokens': 450560, 'profile_sha256': "
                "'52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b'},"
                "{'target_tokens': 524288, 'profile_sha256': "
                "'e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d'}"
                "]}))\n",
                encoding="utf-8",
            )
            (script_dir / "write_pro0813_all_targets_summary.py").write_text(
                SUMMARY_WRITER.read_text(encoding="utf-8"), encoding="utf-8"
            )
            runner = script_dir / ONE_TARGET.name
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "mkdir -p \"$1\"\n"
                "printf 'holder_restarted pid=123 workers=8 util_threshold_pct=90 "
                "util_good_samples=3/15 mapping=0:GPU-a:100\\n' "
                "> \"$1/holder_restore_status\"\n"
                "printf '%s\\n' \"$2\" >> \"$FAKE_TARGET_LOG\"\n"
                "case \"$2\" in 65536) exit 7 ;; 262144) exit 9 ;; *) exit 0 ;; esac\n",
                encoding="utf-8",
            )
            runner.chmod(0o755)
            for relative in (
                "qualification_profiles/pro0813_440k_hotpotqa_10q/profile.json",
                "qualification_profiles/pro0813_512k_hotpotqa_10q/profile.json",
            ):
                profile = script_dir.parent / relative
                profile.parent.mkdir(parents=True, exist_ok=True)
                profile.write_text("{}\n", encoding="utf-8")
            run_root = root / "results"
            target_log = root / "targets.log"
            environment = os.environ.copy()
            environment["FAKE_TARGET_LOG"] = str(target_log)
            completed = subprocess.run(
                ["bash", str(sequence), str(run_root)],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 7, completed.stderr)
            self.assertEqual(
                target_log.read_text(encoding="utf-8").splitlines(),
                ["65536", "131072", "262144", "450560", "524288"],
            )
            summary = json.loads(
                (run_root / "all_targets_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [record["exit_code"] for record in summary["targets"]],
                [7, 0, 9, 0, 0],
            )
            self.assertEqual(summary["overall_exit_code"], 7)
            self.assertFalse(summary["pass"])

    def test_supervisor_proves_retained_holder_and_prioritizes_restore_failure(self):
        source = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn("holder_retained pid=%s workers=8", source)
        self.assertIn('final_status=$restore_status', source)
        self.assertNotIn('if (( final_status == 0 )); then\n      final_status=$restore_status', source)


if __name__ == "__main__":
    unittest.main()
