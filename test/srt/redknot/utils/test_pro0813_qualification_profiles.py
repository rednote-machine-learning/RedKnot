#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
REDKNOT_ROOT = HERE.parent
PROFILE_ROOT = REDKNOT_ROOT / "qualification_profiles"
PROFILES = {
    450560: {
        "directory": "pro0813_440k_hotpotqa_10q",
        "sha256": "52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b",
        "chunk_tokens": 56320,
        "cohort_index": 44000,
        "query_row_ids": [0, 4, 9, 14, 19, 24, 30, 34, 1, 5],
    },
    524288: {
        "directory": "pro0813_512k_hotpotqa_10q",
        "sha256": "e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d",
        "chunk_tokens": 65536,
        "cohort_index": 51200,
        "query_row_ids": [0, 5, 11, 16, 22, 29, 34, 39, 1, 6],
    },
}
DEFAULT_MODEL_ROOT = Path(
    os.environ.get(
        "REDKNOT_PRO0813_PROFILE_TEST_MODEL_ROOT",
        "/workspace/Models/DeepSeek-V4-Pro-0813",
    )
)
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "REDKNOT_PRO0813_PROFILE_TEST_DATA_DIR",
        str(REDKNOT_ROOT / "datasets/LongBench/data"),
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_module(
    "pro0813_qualification_verifier",
    HERE / "verify_pro0813_qualification_profile.py",
)


class TestPro0813QualificationProfiles(unittest.TestCase):
    def _profile(self, target: int) -> Path:
        return (
            PROFILE_ROOT / str(PROFILES[target]["directory"]) / "profile.json"
        ).resolve()

    def _require_generation_inputs(self) -> None:
        required = [
            DEFAULT_MODEL_ROOT / "tokenizer.json",
            DEFAULT_MODEL_ROOT / "tokenizer_config.json",
            DEFAULT_MODEL_ROOT / "encoding/encoding_dsv4.py",
            DEFAULT_DATA_DIR / "hotpotqa.jsonl",
        ]
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            self.skipTest("profile generation inputs are absent: " + ", ".join(absent))

    def test_frozen_profile_hashes_quotas_and_execution_intent(self) -> None:
        observed_selection_hashes = set()
        observed_prompt_hashes = set()
        for target, expected in PROFILES.items():
            profile_path = self._profile(target)
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(_sha256(profile_path), expected["sha256"])
            self.assertEqual(
                profile_path.with_name("profile.json.sha256").read_text(
                    encoding="ascii"
                ),
                f"{expected['sha256']}  profile.json\n",
            )
            self.assertEqual(profile["format"], "redknot_multidataset_profile_v2")
            self.assertEqual(profile["target_tokens"], target)
            self.assertEqual(profile["query_start"], target)
            self.assertEqual(profile["chunk_tokens"], expected["chunk_tokens"])
            self.assertEqual(profile["num_chunks"], 8)
            self.assertEqual(profile["cohort_index"], expected["cohort_index"])
            self.assertEqual(profile["dataset"], "hotpotqa")
            self.assertEqual(profile["num_queries"], 10)
            self.assertEqual(profile["query_row_ids"], expected["query_row_ids"])
            self.assertEqual(
                profile["dataset_quotas"],
                [
                    {
                        "dataset": "hotpotqa",
                        "num_queries": 10,
                        "query_row_ids": expected["query_row_ids"],
                    }
                ],
            )
            self.assertEqual(
                profile["intended_execution_profile"],
                "full_combined_production_v1",
            )
            self.assertEqual(profile["diagnostic_execution_profiles"], ["zoff_only"])
            self.assertEqual(profile["data_manifest"], "data_selection.json")
            self.assertEqual(profile["prompt_manifest"], "prompt_manifest.json")
            self.assertEqual(len(profile["cases"]), 10)
            observed_selection_hashes.add(profile["data_selection_sha256"])
            observed_prompt_hashes.add(profile["prompt_manifest_sha256"])
        self.assertEqual(len(observed_selection_hashes), 2)
        self.assertEqual(len(observed_prompt_hashes), 2)

    def test_stdlib_verifier_closes_both_profiles(self) -> None:
        self._require_generation_inputs()
        for target, expected in PROFILES.items():
            result = VERIFIER.verify_profile(
                self._profile(target),
                expected_target_tokens=target,
                model_root=DEFAULT_MODEL_ROOT,
                data_dir=DEFAULT_DATA_DIR,
                expected_profile_sha256=str(expected["sha256"]),
                include_profile_bytes=True,
            )
            self.assertTrue(result["pass"])
            self.assertEqual(result["target_tokens"], target)
            self.assertEqual(result["num_cases"], 10)
            self.assertEqual(result["profile_sha256"], expected["sha256"])
            self.assertEqual(
                hashlib.sha256(result["_verified_profile_bytes"]).hexdigest(),
                expected["sha256"],
            )

    def test_verifier_rejects_artifact_tamper_and_path_escape(self) -> None:
        self._require_generation_inputs()
        source_dir = self._profile(450560).parent
        with tempfile.TemporaryDirectory(prefix="pro0813-profile-tamper-") as tmp:
            copied = Path(tmp) / "profile"
            shutil.copytree(source_dir, copied)
            prompt = copied / "prompt_manifest.json"
            prompt.write_bytes(prompt.read_bytes() + b" ")
            with self.assertRaisesRegex(
                VERIFIER.ProfileVerificationError, "artifact byte digest mismatch"
            ):
                VERIFIER.verify_profile(
                    (copied / "profile.json").resolve(),
                    expected_target_tokens=450560,
                    model_root=DEFAULT_MODEL_ROOT,
                    data_dir=DEFAULT_DATA_DIR,
                )

            shutil.rmtree(copied)
            shutil.copytree(source_dir, copied)
            profile_path = copied / "profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["data_manifest"] = "../data_selection.json"
            profile_path.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            digest = _sha256(profile_path)
            profile_path.with_name("profile.json.sha256").write_text(
                f"{digest}  profile.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                VERIFIER.ProfileVerificationError,
                "same-directory relative filename",
            ):
                VERIFIER.verify_profile(
                    profile_path.resolve(),
                    expected_target_tokens=450560,
                    model_root=DEFAULT_MODEL_ROOT,
                    data_dir=DEFAULT_DATA_DIR,
                )

            shutil.rmtree(copied)
            shutil.copytree(source_dir, copied)
            prompt = copied / "prompt_manifest.json"
            outside = Path(tmp) / "outside-prompt.json"
            shutil.copyfile(prompt, outside)
            prompt.unlink()
            prompt.symlink_to(outside)
            with self.assertRaisesRegex(
                VERIFIER.ProfileVerificationError, "non-symlink regular file"
            ):
                VERIFIER.verify_profile(
                    (copied / "profile.json").resolve(),
                    expected_target_tokens=450560,
                    model_root=DEFAULT_MODEL_ROOT,
                    data_dir=DEFAULT_DATA_DIR,
                )

    def test_verifier_rejects_cross_target_binding(self) -> None:
        with self.assertRaisesRegex(
            VERIFIER.ProfileVerificationError, "invalid target_tokens"
        ):
            VERIFIER.verify_profile(
                self._profile(450560),
                expected_target_tokens=524288,
                model_root=DEFAULT_MODEL_ROOT,
                data_dir=DEFAULT_DATA_DIR,
            )

    def test_profiles_rebuild_byte_for_byte(self) -> None:
        self._require_generation_inputs()
        builder = HERE / "prepare_multidataset_512k_manifests.py"
        core = HERE / "benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py"
        for target, expected in PROFILES.items():
            with tempfile.TemporaryDirectory(
                prefix=f"pro0813-profile-{target}-"
            ) as tmp:
                output = Path(tmp)
                command = [
                    sys.executable,
                    str(builder),
                    "--core",
                    str(core),
                    "--dataset",
                    "hotpotqa",
                    "--data-dir",
                    str(DEFAULT_DATA_DIR),
                    "--model",
                    str(DEFAULT_MODEL_ROOT),
                    "--num-queries",
                    "10",
                    "--row-offset",
                    "0",
                    "--cohort-index",
                    str(expected["cohort_index"]),
                    "--chunk-tokens",
                    str(expected["chunk_tokens"]),
                    "--num-chunks",
                    "8",
                    "--data-out",
                    str(output / "data_selection.json"),
                    "--prompt-out",
                    str(output / "prompt_manifest.json"),
                    "--profile-out",
                    str(output / "profile.json"),
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)
                frozen = self._profile(target).parent
                for name in (
                    "data_selection.json",
                    "prompt_manifest.json",
                    "profile.json",
                    "profile.json.sha256",
                ):
                    self.assertEqual(
                        (output / name).read_bytes(),
                        (frozen / name).read_bytes(),
                        f"deterministic rebuild drifted for {target}/{name}",
                    )

    def test_http_consumer_resolves_only_verified_relative_artifacts(self) -> None:
        self._require_generation_inputs()
        benchmark = _load_module(
            "pro0813_http_profile_consumer",
            HERE / "benchmark_dsv4_pro0813_redknot_http.py",
        )
        for target in PROFILES:
            args = SimpleNamespace(
                target_tokens=target,
                qualification_profile=str(self._profile(target)),
            )
            with mock.patch.object(benchmark, "MODEL", DEFAULT_MODEL_ROOT), mock.patch.object(
                benchmark, "DATA_DIR", DEFAULT_DATA_DIR
            ):
                resolved = benchmark._resolve_redknot_target_profile(args)
            self.assertTrue(resolved["data_manifest"].is_absolute())
            self.assertTrue(Path(resolved["prompt_manifest"]).is_absolute())
            self.assertEqual(
                resolved["qualification_profile_sha256"],
                PROFILES[target]["sha256"],
            )
            self.assertEqual(
                resolved["intended_execution_profile"],
                "full_combined_production_v1",
            )


if __name__ == "__main__":
    unittest.main()
