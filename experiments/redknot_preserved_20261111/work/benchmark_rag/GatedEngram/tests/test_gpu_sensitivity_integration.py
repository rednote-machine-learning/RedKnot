"""CPU-only wiring guards for the optional, independently frozen diagnostic.

These tests do not instantiate a model, import Torch, query a GPU, or launch a
worker. Sealed synthetic records below exercise orchestration, not accuracy.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_pilot import launch_pilot as launch
from gpu_pilot import pilot_protocol as pp
from gpu_pilot import run_pilot as runner
from gpu_pilot.sensitivity_protocol import sensitivity_protocol


class SensitivityLaunchIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage = self.root / "new_sensitivity_cpu_fixture/GatedEngram"
        for folder in ("gpu_pilot", "kv_cache_engram", "tests"):
            (self.stage / folder).mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir()
        for filename in ("gpu_pilot/run_pilot.py", "gpu_pilot/launch_pilot.py",
                         "kv_cache_engram/core.py", "tests/test_cpu.py"):
            (self.stage / filename).write_text("# immutable CPU fixture\n")
        self.manifest = self.stage / "pilot.manifest.json"
        self.protocol = self.stage / "pilot.protocol.json"
        self.extension = self.stage / "sensitivity.protocol.json"
        self.write(self.protocol, pp.pilot_protocol())
        self.write(self.extension, sensitivity_protocol())
        self.manifest_data = dict(schema="independent-native-pilot-input-v1",
            artifact_kind="kv_cache_engram_native_pilot",
            pilot_protocol_sha256=pp.pilot_protocol()["sha256"],
            sensitivity_protocol_sha256=sensitivity_protocol()["sha256"])
        self.write(self.manifest, pp.seal_record(self.manifest_data))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True))

    def plan(self, **changes):
        args = dict(root=self.root, stage_dir=self.stage, name="kce-sensitivity-cpu-only",
            manifest=self.manifest, protocol=self.protocol, sensitivity_protocol=self.extension)
        args.update(changes)
        return launch.build_plan(**args)

    def test_extension_is_bound_by_path_file_and_canonical_digest(self):
        plan = self.plan()
        item = plan["sensitivity_protocol"]
        self.assertEqual(item["path"], str(self.extension))
        self.assertEqual(item["sha256"], sensitivity_protocol()["sha256"])
        self.assertEqual(item["file_sha256"], launch.file_digest(self.extension))
        self.assertEqual(plan["manifest"]["sha256"], launch.read_json(self.manifest)["sha256"])
        self.assertEqual(plan["protocol"]["sha256"], pp.pilot_protocol()["sha256"])
        self.assertEqual(plan["argv"].count("--sensitivity-protocol"), 1)
        self.assertEqual(plan["argv"][plan["argv"].index("--sensitivity-protocol") + 1],
                         str(self.extension))
        launch.verify_plan_inputs(plan)
        self.assertEqual(list((self.root / "results").iterdir()), [])

    def test_v1_arguments_remain_a_prefix_and_no_extension_means_no_flag(self):
        original = launch.fixed_runner_args(self.root, self.manifest, self.protocol)
        extended = launch.fixed_runner_args(self.root, self.manifest, self.protocol, self.extension)
        self.assertEqual(extended, original + ["--sensitivity-protocol", str(self.extension)])
        self.assertNotIn("--sensitivity-protocol", original)
        for flag, value in (("--dp", "8"), ("--ep", "8"), ("--cp", "1"),
                            ("--max-model-len", "270336"), ("--max-num-batched-tokens", "8192")):
            self.assertEqual(extended[extended.index(flag) + 1], value)
        self.assertNotIn("--allow-gpu-init", extended)

    def test_legacy_plan_still_accepts_legacy_manifest_without_extension(self):
        legacy = dict(self.manifest_data)
        legacy.pop("sensitivity_protocol_sha256")
        self.write(self.manifest, pp.seal_record(legacy))
        plan = self.plan(sensitivity_protocol=None)
        self.assertIsNone(plan["sensitivity_protocol"])
        self.assertNotIn("--sensitivity-protocol", plan["argv"])
        launch.verify_plan_inputs(plan)

    def test_sensitivity_manifest_cannot_silently_run_legacy_worker(self):
        with self.assertRaisesRegex(ValueError, "explicit sensitivity protocol"):
            self.plan(sensitivity_protocol=None)

    def test_extension_requires_its_manifest_binding(self):
        for binding in (None, "0" * 64):
            value = dict(self.manifest_data)
            if binding is None:
                value.pop("sensitivity_protocol_sha256")
            else:
                value["sensitivity_protocol_sha256"] = binding
            self.write(self.manifest, pp.seal_record(value))
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                self.plan()

    def test_extension_must_be_a_distinct_file_inside_independent_stage(self):
        outside = self.root / "outside.protocol.json"
        self.write(outside, sensitivity_protocol())
        for path in (outside, self.manifest, self.protocol):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.plan(sensitivity_protocol=path)

    def test_extension_symlink_is_rejected(self):
        alias = self.stage / "sensitivity.alias.json"
        alias.symlink_to(self.extension)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.plan(sensitivity_protocol=alias)

    def test_changed_seal_and_even_resealed_different_rules_are_rejected(self):
        changed = sensitivity_protocol()
        changed["early_full_layers"] = [0]
        self.write(self.extension, changed)
        with self.assertRaises(ValueError):
            self.plan()
        changed.pop("sha256")
        changed = pp.seal_record(changed)
        self.write(self.extension, changed)
        self.write(self.manifest, pp.seal_record(dict(self.manifest_data,
            sensitivity_protocol_sha256=changed["sha256"])))
        with self.assertRaisesRegex(ValueError, "tested rules"):
            self.plan()

    def test_post_freeze_extension_byte_drift_is_rejected(self):
        plan = self.plan()
        self.extension.write_text(self.extension.read_text() + "\n")
        # Same parsed JSON and canonical seal is not the same frozen file.
        self.assertEqual(launch.read_json(self.extension), sensitivity_protocol())
        with self.assertRaisesRegex(ValueError, "sensitivity protocol changed after freeze"):
            launch.verify_plan_inputs(plan)

    def test_post_freeze_source_or_manifest_drift_is_still_rejected(self):
        plan = self.plan()
        self.write(self.manifest, pp.seal_record(dict(self.manifest_data, injected="drift")))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            launch.verify_plan_inputs(plan)
        self.write(self.manifest, pp.seal_record(self.manifest_data))
        (self.stage / "kv_cache_engram/core.py").write_text("# changed CPU fixture\n")
        with self.assertRaisesRegex(ValueError, "stage changed"):
            launch.verify_plan_inputs(plan)

    def test_check_only_accepts_extension_without_launch_or_hardware_access(self):
        with mock.patch.object(launch, "run_plan", side_effect=AssertionError("no launch")) as run, \
             mock.patch.object(launch, "holder_snapshot", side_effect=AssertionError("no GPU query")) as holder, \
             mock.patch.object(launch.subprocess, "Popen", side_effect=AssertionError("no process")) as process, \
             mock.patch.object(launch.subprocess, "check_output", side_effect=AssertionError("no process")) as query, \
             contextlib.redirect_stdout(io.StringIO()) as out:
            result = launch.main(["--name", "kce-sensitivity-check-only", "--root", str(self.root),
                "--stage-dir", str(self.stage), "--manifest", str(self.manifest),
                "--pilot-protocol", str(self.protocol), "--sensitivity-protocol", str(self.extension),
                "--check-only"])
        self.assertEqual(result, 0)
        record = json.loads(out.getvalue())
        self.assertFalse(record["gpu_initialized"])
        self.assertFalse(record["lifecycle_identity_checked"])
        self.assertEqual(record["plan"]["sensitivity_protocol"]["sha256"],
                         sensitivity_protocol()["sha256"])
        for mocked in (run, holder, process, query):
            mocked.assert_not_called()
        self.assertEqual(list((self.root / "results").iterdir()), [])


class SensitivityRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "sensitivity.protocol.json"
        self.path.write_text(json.dumps(sensitivity_protocol()))
        self.args = SimpleNamespace(sensitivity_protocol=self.path)
        self.manifest = {"sensitivity_protocol_sha256": sensitivity_protocol()["sha256"]}

    def test_runner_parses_extension_without_leaking_flags_to_native_parser(self):
        base = SimpleNamespace(parse_args=mock.Mock(return_value=SimpleNamespace(cpu_fixture=True)))
        with mock.patch.object(runner, "load_runtime", return_value=(base, None, None, None)) as load:
            args = runner.parse_args(["--experiment-root", "/independent", "--pilot-protocol",
                "/independent/pilot.json", "--sensitivity-protocol", str(self.path),
                "--manifest", "/independent/manifest.json", "--check-only"])
        load.assert_called_once_with(Path("/independent"))
        base.parse_args.assert_called_once_with(["--manifest", "/independent/manifest.json", "--check-only"])
        self.assertEqual(args.sensitivity_protocol, self.path)
        self.assertTrue(args.cpu_fixture)

    def test_runner_legacy_absence_is_explicit_and_compatible(self):
        self.assertIsNone(runner.validate_sensitivity_inputs(SimpleNamespace(), {}))
        self.assertIsNone(runner.validate_sensitivity_inputs(SimpleNamespace(sensitivity_protocol=None), {}))
        with self.assertRaisesRegex(ValueError, "explicit protocol"):
            runner.validate_sensitivity_inputs(SimpleNamespace(), self.manifest)

    def test_runner_accepts_exact_protocol_manifest_and_execution_identity(self):
        expected = sensitivity_protocol()
        self.assertEqual(runner.validate_sensitivity_inputs(self.args, self.manifest), expected)
        self.assertEqual(runner.validate_sensitivity_inputs(self.args, self.manifest,
                         identity={"sensitivity_protocol_sha256": expected["sha256"]}), expected)

    def test_runner_rejects_missing_or_foreign_manifest_and_execution_identity(self):
        for manifest in ({}, {"sensitivity_protocol_sha256": "0" * 64}):
            with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, "manifest mismatch"):
                runner.validate_sensitivity_inputs(self.args, manifest)
        for identity in ({}, {"sensitivity_protocol_sha256": "0" * 64}):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "identity mismatch"):
                runner.validate_sensitivity_inputs(self.args, self.manifest, identity=identity)

    def test_runner_rejects_a_resealed_modified_protocol(self):
        value = sensitivity_protocol()
        value.pop("sha256")
        value["original_numerical_gate_required"] = False
        value = pp.seal_record(value)
        self.path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "protocol/manifest mismatch"):
            runner.validate_sensitivity_inputs(self.args, {"sensitivity_protocol_sha256": value["sha256"]})


class SensitivityProtectionIntegrationTests(unittest.TestCase):
    def records(self):
        # Deliberately small synthetic calibration: do not mistake for model evidence.
        numerical = pp.seal_record(dict(schema="cpu-fixture-numerical-calibration", layers={
            "3": dict(mode="gated", selected={"candidate": "strict"}, candidates=[{"error": 0}]),
            "6": dict(mode="gated", selected={"candidate": "strict"}, candidates=[{"error": .01}]),
            "7": dict(mode="full_recompute", selected=None, candidates=[{"error": .2}]),
            "18": dict(mode="gated", selected={"candidate": "loose"}, candidates=[{"error": .001}])},
            calibration_case_indices=[0, 1], heldout_used=False))
        admission = pp.seal_record(dict(schema="cpu-fixture-admission-plan",
            numerical_calibration_sha256=numerical["sha256"], mandatory_full_layers=[0, 1, 2, 3, 4, 5, 18]))
        return numerical, admission

    def test_protection_only_revokes_selected_layers_and_retains_audit(self):
        numerical, admission = self.records()
        original = copy.deepcopy(numerical)
        result = runner.protect_calibrated_layers(numerical, admission)
        self.assertEqual(numerical, original)
        self.assertEqual(set(result["layers"]), set(numerical["layers"]))
        for key in ("3", "18"):
            self.assertIsNone(result["layers"][key]["selected"])
            self.assertEqual(result["layers"][key]["mode"], "full_recompute")
            self.assertTrue(result["layers"][key]["sensitivity_protected"])
            self.assertEqual(result["layers"][key]["candidates"], numerical["layers"][key]["candidates"])
        for key in ("6", "7"):
            self.assertEqual(result["layers"][key], numerical["layers"][key])
        self.assertEqual(result["calibration_case_indices"], [0, 1])
        self.assertFalse(result["heldout_used"])

    def test_protection_hash_binds_both_parents_and_survives_json(self):
        numerical, admission = self.records()
        result = runner.protect_calibrated_layers(numerical, admission)
        self.assertEqual(result["numerical_calibration_sha256"], numerical["sha256"])
        self.assertEqual(result["sensitivity_admission_plan_sha256"], admission["sha256"])
        self.assertNotEqual(result["sha256"], numerical["sha256"])
        pp._sealed(json.loads(json.dumps(result)))

    def test_protection_output_does_not_alias_numerical_candidates(self):
        numerical, admission = self.records()
        result = runner.protect_calibrated_layers(numerical, admission)
        result["layers"]["6"]["candidates"][0]["error"] = 999
        self.assertEqual(numerical["layers"]["6"]["candidates"][0]["error"], .01)
        result["layers"]["3"]["candidates"].append({"error": 999})
        self.assertEqual(len(numerical["layers"]["3"]["candidates"]), 1)

    def test_foreign_numerical_parent_is_rejected(self):
        numerical, admission = self.records()
        admission.pop("sha256")
        admission["numerical_calibration_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "another numerical calibration"):
            runner.protect_calibrated_layers(numerical, pp.seal_record(admission))

    def test_tampering_either_parent_seal_is_rejected(self):
        for parent in ("numerical", "admission"):
            numerical, admission = self.records()
            (numerical if parent == "numerical" else admission)["extra"] = "tampered"
            with self.subTest(parent=parent), self.assertRaises(ValueError):
                runner.protect_calibrated_layers(numerical, admission)


if __name__ == "__main__":
    unittest.main()
