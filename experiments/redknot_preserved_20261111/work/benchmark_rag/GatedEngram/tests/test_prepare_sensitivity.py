"""CPU cloning tests with synthetic exact-length tokens, never model evidence."""
import builtins
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gpu_pilot import launch_pilot as launch
from gpu_pilot import pilot_protocol as pp
from gpu_pilot import prepare_sensitivity as prep
from gpu_pilot import run_pilot
from gpu_pilot.sensitivity_protocol import sensitivity_protocol


def synthetic_manifest():
    """Real geometry for input validation only; no factual/model claims."""
    source_plan = pp.seal_record(dict(schema="cpu-fixture-source-plan-v1",
        excluded_context_sha256=[launch.digest("old exposed source")] ))
    rows = []
    for index in range(6):
        documents, ids = [], [999]
        for doc_index in range(8):
            tokens = [100 + 10 * index + doc_index] * 32768
            start = len(ids)
            ids.extend(tokens)
            token_hash = launch.digest(tokens)
            documents.append(dict(document_id=token_hash, token_ids_sha256=token_hash,
                input_start=start, input_end=len(ids), tokens=32768,
                online_position=start, offline_position=0))
        ids.append(1000)
        prompt, refs = "synthetic prompt " + str(index), ["synthetic gold " + str(index)]
        rows.append(dict(index=index, sample_id="synthetic:" + str(index),
            dataset=("hotpotqa", "2wikimqa")[index % 2],
            split="calibration" if index < 2 else "heldout", question="question " + str(index),
            input_ids=ids, input_tokens=len(ids), input_ids_sha256=launch.digest(ids),
            prefix_tokens=1, suffix_tokens=1, context_tokens=262144,
            primary_context_truncated=False, documents=documents,
            prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            references=refs, references_sha256=launch.digest(refs),
            context_segments=[dict(context_sha256=launch.digest(["source", index]))]))
    return pp.seal_record(dict(schema="redknot-deepinfer-real-rag-manifest-v1",
        artifact_kind="kv_cache_engram_native_pilot", warmup=rows[:2], cohort=rows[2:],
        sources=[dict(dataset="hotpotqa", path="/unchanged/provenance/hotpotqa.jsonl"),
                 dict(dataset="2wikimqa", path="relative-provenance/2wikimqa.jsonl")],
        selection_plan=source_plan, pilot_protocol_sha256=pp.pilot_protocol()["sha256"],
        num_chunks=8, chunk_tokens=32768, context_tokens=262144,
        fixture_kind="synthetic CPU geometry only; not native/GPU or quality evidence"))


class PrepareSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_template = synthetic_manifest()
        cls.manifest_bytes = json.dumps(cls.manifest_template, ensure_ascii=False).encode() + b"\n"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source, self.output = self.root / "inputs-v2", self.root / "inputs-sensitivity"
        self.source.mkdir()
        (self.source / "manifest.json").write_bytes(self.manifest_bytes)
        # Deliberately noncanonical whitespace must survive cloning unchanged.
        (self.source / "protocol.json").write_text(json.dumps(pp.pilot_protocol(), indent=3) + "\n\n")
        (self.source / "source-plan.json").write_text(
            json.dumps(self.manifest_template["selection_plan"], indent=1) + "\n")
        (self.source / "assets").mkdir()
        (self.source / "assets/opaque.bin").write_bytes(b"\x00\xff\x80CPU-fixture\r\n")
        (self.source / "empty-folder").mkdir()

    def write_manifest(self, value):
        value = {key: item for key, item in value.items() if key != "sha256"}
        (self.source / "manifest.json").write_text(json.dumps(pp.seal_record(value)))

    def test_exact_clone_preserves_samples_paths_and_all_other_bytes(self):
        before = prep._inventory(self.source)
        report = prep.prepare(self.source, self.output)
        self.assertEqual(prep._inventory(self.source), before)
        self.assertEqual((self.output / "parent-manifest.json").read_bytes(), self.manifest_bytes)
        for name in ("protocol.json", "source-plan.json", "assets/opaque.bin"):
            self.assertEqual((self.output / name).read_bytes(), (self.source / name).read_bytes())
        self.assertTrue((self.output / "empty-folder").is_dir())
        new = launch.read_sealed(self.output / "manifest.json")
        original = self.manifest_template
        self.assertEqual(new["parent_manifest_sha256"], original["sha256"])
        self.assertEqual(new["sensitivity_protocol_sha256"], sensitivity_protocol()["sha256"])
        self.assertEqual({k: v for k, v in new.items()
                          if k not in ("sha256", "parent_manifest_sha256", "sensitivity_protocol_sha256")},
                         {k: v for k, v in original.items() if k != "sha256"})
        rows = run_pilot.validate_inputs(prep._ManifestChecks, new,
                                        launch.read_json(self.output / "protocol.json"))
        self.assertEqual([row["index"] for row in rows], list(range(6)))
        self.assertEqual([row["context_tokens"] for row in rows], [262144] * 6)
        self.assertEqual(launch.read_sealed(self.output / "sensitivity-protocol.json"), sensitivity_protocol())
        self.assertEqual(launch.read_sealed(self.output / "preparation.json"), report)
        self.assertTrue(report["complete"])
        for name, digest in report["output_files_sha256"].items():
            self.assertEqual(launch.file_digest(self.output / name), digest)

    def test_cli_has_no_model_import_or_process_or_gpu_access(self):
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "deepinfer", "_config", "transformers"):
                raise AssertionError("CPU preparation imported " + name)
            return real_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=guarded), \
             mock.patch.object(launch.subprocess, "Popen", side_effect=AssertionError("no launch")) as process, \
             mock.patch.object(launch.subprocess, "check_output", side_effect=AssertionError("no query")) as query, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = prep.main(["--source-inputs", str(self.source), "--output", str(self.output)])
        self.assertEqual(result, 0)
        summary = json.loads(stdout.getvalue())
        self.assertFalse(summary["gpu_initialized"])
        self.assertEqual(summary["calibration_cases"], 2)
        self.assertEqual(summary["evaluation_cases"], 4)
        report = launch.read_sealed(self.output / "preparation.json")
        for field in ("gpu_initialized", "model_imported", "tokenizer_called", "samples_reselected"):
            self.assertFalse(report[field])
        process.assert_not_called(); query.assert_not_called()

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_bytes(b"existing evidence")
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            prep.prepare(self.source, self.output)
        self.assertEqual(sentinel.read_bytes(), b"existing evidence")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_nested_and_same_directories_fail_without_mutation(self):
        before = prep._inventory(self.source)
        for output in (self.source, self.source / "nested", self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "not nested"):
                prep.prepare(self.source, output)
        self.assertEqual(prep._inventory(self.source), before)

    def test_source_and_destination_symlinks_are_rejected(self):
        source_alias = self.root / "source-alias"
        source_alias.symlink_to(self.source, target_is_directory=True)
        output_alias = self.root / "output-alias"
        output_alias.symlink_to(self.output, target_is_directory=True)
        for source, output in ((source_alias, self.output), (self.source, output_alias)):
            with self.subTest(source=source, output=output), self.assertRaisesRegex(ValueError, "symlink"):
                prep.prepare(source, output)
        self.assertFalse(self.output.exists())

    def test_nested_input_symlink_is_not_traversed_or_copied(self):
        (self.source / "assets/alias").symlink_to(self.source / "protocol.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            prep.prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_original_source_plan_fails_before_output_creation(self):
        (self.source / "source-plan.json").rename(self.source / "different-name.json")
        with self.assertRaisesRegex(ValueError, "source-plan.json"):
            prep.prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_reserved_names_prevent_silent_reextension(self):
        for name in prep.RESERVED:
            path = self.source / name
            path.write_bytes(b"existing evidence")
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "output names"):
                    prep.prepare(self.source, self.output)
            finally:
                path.unlink()
        self.assertFalse(self.output.exists())

    def test_bad_seal_is_not_resealed_into_valid_new_evidence(self):
        value = copy.deepcopy(self.manifest_template)
        value["cohort"][0]["input_ids"][1] += 1
        (self.source / "manifest.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "seal"):
            prep.prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_resealed_token_tampering_and_source_overlap_still_fail(self):
        mutations = (
            lambda m: m["cohort"][0]["input_ids"].__setitem__(1, 0),
            lambda m: m["cohort"][0].update(context_segments=m["warmup"][0]["context_segments"]),
            lambda m: m["cohort"][0]["documents"][0].update(input_end=100),
        )
        for mutate in mutations:
            value = copy.deepcopy(self.manifest_template)
            mutate(value)
            self.write_manifest(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                prep.prepare(self.source, self.output)
            self.assertFalse(self.output.exists())

    def test_source_plan_mismatch_fails_even_if_both_are_sealed(self):
        value = dict(self.manifest_template["selection_plan"])
        value.pop("sha256")
        value["changed"] = True
        (self.source / "source-plan.json").write_text(json.dumps(pp.seal_record(value)))
        with self.assertRaisesRegex(ValueError, "source plan and manifest differ"):
            prep.prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_copy_detects_changed_bytes_and_preserves_partial_output(self):
        real_copy = prep._copy_exclusive
        def change_then_copy(source, target, expected):
            if source.name == "opaque.bin":
                source.write_bytes(b"simulated concurrent source change")
            real_copy(source, target, expected)
        with mock.patch.object(prep, "_copy_exclusive", side_effect=change_then_copy), \
             self.assertRaisesRegex(ValueError, "source changed while copying"):
            prep.prepare(self.source, self.output)
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "preparation.json").exists())
        self.assertFalse((self.output / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            prep.prepare(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
