"""CPU tests for fresh three-arm comparison admission and phase identity."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_pilot import run_tail_link_pilot as runner
from gpu_pilot import tail_link_protocol as protocol
from gpu_pilot import read_tail_link_results as reader


def seal(body):
    body = {k: v for k, v in body.items() if k != "sha256"}
    return dict(body, sha256=runner._digest(body))


def timing_phase(arm, repeat):
    ttft = {"recomputed": 80., "tail128_link": 55., "tail128_dynamic": 50.}[arm] + repeat
    ids = list(range(32))
    return seal(dict(
        schema=runner.PHASE_SCHEMA, index=2, arm=arm, repeat=repeat,
        dynamic_document_swa=arm == "tail128_dynamic", native_yoco_swa_share_hidden=False, timing_valid=True,
        quality_eligible=False, input_ids_sha256="a" * 64,
        execution_configuration_sha256="b" * 64, manifest_sha256="c" * 64,
        tail_link_protocol_sha256="d" * 64, online_global_window_s=ttft+3,
        record=dict(success=True, rank=0, input_ids_sha256="a" * 64,
                    output_ids=ids, output_ids_sha256=runner._digest(ids), natural_EOS=False,
                    online_ttft_s=ttft, online_elapsed_s=ttft+1, online_prepare_to_submit_s=.1)))


class DynamicProtocolTests(unittest.TestCase):
    def test_link_seed_precedes_refresh_and_first3_global_stay_native(self):
        p = protocol.tail_link_protocol()
        self.assertEqual(p["first_full_layers"], [0, 1, 2])
        self.assertTrue(p["dynamic_layer3_static_link_seed"])
        self.assertEqual(p["dynamic_refresh_start_layer"], 4)
        self.assertEqual(p["global_source_layers_zero_based"], [2, 8, 14, 20])
        self.assertTrue(p["global_kv_complete_online"])
        self.assertTrue(p["preserve_native_global_consumer_sharing"])
        self.assertFalse(p["forced_survivor_retention"])
        self.assertIs(p["native_yoco_swa_share_hidden"], False)
        self.assertTrue(p["dynamic_qualification_independent_of_static"])

    def test_both_partial_policies_require_fresh_numerical_gates(self):
        p = protocol.tail_link_protocol()
        self.assertEqual(p["sparse_parity"]["arms"], list(runner.ARMS[1:]))
        self.assertEqual(p["partial_smoke"]["arms"], list(runner.ARMS[1:]))
        self.assertEqual(p["sparse_parity"]["layer_chunk_checks"], 1320)
        self.assertTrue(p["sparse_parity"]["byteexact"])
        self.assertFalse(p["warmup_gpu_payload_reused_by_formal"])

    def test_balanced_three_arm_rotation(self):
        for index in runner.EVALUATION_INDICES:
            schedule = runner.timing_schedule(index)
            self.assertEqual(schedule, protocol.timing_schedule(index))
            self.assertEqual(len(schedule), 9)
            for arm in runner.ARMS:
                self.assertEqual([repeat for repeat, a in schedule if a == arm], [0, 1, 2])
                self.assertEqual({schedule[3*r+s][1] for r in range(3) for s in [0]}, set(runner.ARMS))


class ThreeArmTimingTests(unittest.TestCase):
    def phases(self):
        return [timing_phase(arm, repeat) for repeat, arm in runner.timing_schedule(2)]

    def test_all_three_formal_arms_produce_fresh_speed_comparisons(self):
        summary = runner.summarize_timing(self.phases(), 2)
        self.assertEqual(set(summary["arms"]), set(runner.ARMS))
        self.assertEqual(summary["arms"]["recomputed"]["online_ttft_p50_s"], 81.)
        self.assertEqual(summary["arms"]["tail128_link"]["online_ttft_p50_s"], 56.)
        self.assertEqual(summary["arms"]["tail128_dynamic"]["online_ttft_p50_s"], 51.)
        self.assertAlmostEqual(summary["arms"]["tail128_dynamic"]["ttft_speedup"], 81/51)

    def test_old_two_arm_results_cannot_close_new_comparison(self):
        phases = [p for p in self.phases() if p["arm"] != "tail128_dynamic"]
        with self.assertRaises(ValueError):
            runner.summarize_timing(phases, 2)

    def test_missing_repeat_cannot_close_comparison(self):
        phases = self.phases()
        phases.pop()
        with self.assertRaises(ValueError):
            runner.summarize_timing(phases, 2)

    def test_static_execution_cannot_be_relabelled_dynamic(self):
        phases = self.phases()
        idx = next(i for i, p in enumerate(phases) if p["arm"] == "tail128_dynamic")
        phases[idx] = seal(dict(phases[idx], dynamic_document_swa=False))
        with self.assertRaisesRegex(ValueError, "dynamic policy"):
            runner.summarize_timing(phases, 2)

    def test_dynamic_execution_cannot_be_relabelled_static(self):
        phases = self.phases()
        idx = next(i for i, p in enumerate(phases) if p["arm"] == "tail128_link")
        phases[idx] = seal(dict(phases[idx], dynamic_document_swa=True))
        with self.assertRaisesRegex(ValueError, "dynamic policy"):
            runner.summarize_timing(phases, 2)

    def test_foreign_input_or_execution_rejected(self):
        for field in ("execution_configuration_sha256", "manifest_sha256", "tail_link_protocol_sha256"):
            phases = self.phases()
            phases[-1] = seal(dict(phases[-1], **{field: "e" * 64}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.summarize_timing(phases, 2)


class PolicySmokeTests(unittest.TestCase):
    def fixture(self, arm="tail128_dynamic"):
        row = dict(index=2, input_ids_sha256="a" * 64, input_tokens=262194, documents=[{}] * 8)
        qual = dict(passed=True, identity_check_count=80, sha256="b" * 64,
                    empty_ep_qualification_sha256="c" * 64)
        p = seal(dict(schema=runner.PHASE_SCHEMA, index=2, arm=arm, fraction=None,
            mode="sensitive", dynamic_document_swa=arm == "tail128_dynamic", native_yoco_swa_share_hidden=False,
            timing_valid=False, quality_eligible=False, partial_smoke=True,
            input_ids_sha256=row["input_ids_sha256"],
            empty_ep_qualification_sha256=qual["empty_ep_qualification_sha256"],
            controller=dict(dynamic_document_swa=arm == "tail128_dynamic", native_yoco_swa_share_hidden=False),
            record=dict(success=True, rank=0, input_ids_sha256=row["input_ids_sha256"],
                        output_ids=[1], output_ids_sha256=runner._digest([1]), natural_EOS=False)))
        sparse = dict(passed=True, sha256="d" * 64, input_ids_sha256=row["input_ids_sha256"],
                      candidate_sha256=p["sha256"], qualification_sha256=qual["sha256"])
        return p, row, qual, sparse

    def check(self, p, row, qual, sparse, arm="tail128_dynamic"):
        with mock.patch.object(reader, "documents", return_value=[]), \
             mock.patch.object(reader, "audit_controller", return_value={
                 "ffn_bypassed_rows": 10, "swa_reused_rows": 5,
                 "dynamic_refreshed_rows": 10, "dynamic_fallback_memory_rows": 5}):
            return runner.verify_partial_smoke([p], row, qual, sparse, arm=arm)

    def test_dynamic_closure_binds_its_own_numerical_candidate(self):
        p, row, qual, sparse = self.fixture()
        closure = self.check(p, row, qual, sparse)
        self.assertEqual(closure["phase_sha256"], {"tail128_dynamic": p["sha256"]})
        self.assertEqual(closure["sparse_qualification_sha256"], sparse["sha256"])
        self.assertFalse(closure["timing_valid"])

    def test_other_policy_gate_cannot_admit_dynamic_smoke(self):
        p, row, qual, sparse = self.fixture()
        with self.assertRaisesRegex(ValueError, "own fresh"):
            self.check(p, row, qual, dict(sparse, candidate_sha256="e" * 64))

    def test_dynamic_smoke_must_execute_real_document_refresh(self):
        p, row, qual, sparse = self.fixture()
        with mock.patch.object(reader, "documents", return_value=[]), \
             mock.patch.object(reader, "audit_controller", return_value={
                 "ffn_bypassed_rows": 10, "swa_reused_rows": 5, "dynamic_refreshed_rows": 0}):
            with self.assertRaisesRegex(ValueError, "did not refresh"):
                runner.verify_partial_smoke([p], row, qual, sparse, arm="tail128_dynamic")

    def test_wrong_smoke_arm_is_rejected(self):
        p, row, qual, sparse = self.fixture("tail128_link")
        with self.assertRaises(ValueError):
            self.check(p, row, qual, sparse)


class NativeYocoConfigurationTests(unittest.TestCase):
    SOURCE = "import torch\n_GPT_DEFAULTS: dict = {'yoco_swa_share_hidden': False}\ndef load():\n    return AttrDict(_GPT_DEFAULTS, hidden_size=1)\n"

    def check(self, *, source=None, config=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = root / "deepinfer/hugging_face/config.py"
            parser.parent.mkdir(parents=True)
            parser.write_text(self.SOURCE if source is None else source)
            (root / "config.json").write_text(json.dumps(
                dict(model_type="deepseek_v41", text_config={}) if config is None else config))
            return runner.native_yoco_swa_configuration(root, root)

    def test_cpu_ast_uses_frozen_false_without_executing_parser(self):
        result = self.check(source=self.SOURCE + "\nraise RuntimeError('must not execute')\n")
        self.assertTrue(result["verified"])
        self.assertIs(result["native_yoco_swa_share_hidden"], False)
        self.assertEqual(result["config_overrides"], [])
        self.assertEqual(len(result["hf_parser_sha256"]), 64)

    def test_all_nested_false_overrides_are_recorded(self):
        config = dict(model_type="deepseek_v41", text_config=dict(yoco_swa_share_hidden=False),
                      layer_configs=[dict(yoco_swa_share_hidden=False)])
        result = self.check(config=config)
        self.assertEqual({x["path"] for x in result["config_overrides"]},
                         {"/text_config/yoco_swa_share_hidden", "/layer_configs/0/yoco_swa_share_hidden"})

    def test_true_or_non_boolean_overrides_rejected_everywhere(self):
        for value in (True, 0, "false", None):
            for location in ("root", "text", "layer"):
                config = dict(model_type="deepseek_v41", text_config={})
                target = config if location == "root" else config["text_config"]
                if location == "layer":
                    config["layer_configs"] = [{}]; target = config["layer_configs"][0]
                target["yoco_swa_share_hidden"] = value
                with self.subTest(value=value, location=location), self.assertRaises(ValueError):
                    self.check(config=config)

    def test_true_default_or_parser_override_or_unknown_consumer_rejected(self):
        for source in (self.SOURCE.replace(": False", ": True"),
                       self.SOURCE.replace("hidden_size=1", "yoco_swa_share_hidden=True"),
                       self.SOURCE + "\n_GPT_DEFAULTS.update(other)\n",
                       self.SOURCE.replace("hidden_size=1", "**other"),
                       self.SOURCE.replace("AttrDict(_GPT_DEFAULTS", "Other(_GPT_DEFAULTS")):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.check(source=source)

    def test_foreign_model_config_rejected(self):
        with self.assertRaises(ValueError):
            self.check(config=dict(model_type="other", text_config={}))

    def test_phase_missing_or_true_native_setting_rejected(self):
        for value in (True, None):
            phase = timing_phase("tail128_dynamic", 0)
            phase["native_yoco_swa_share_hidden"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "native YOCO"):
                runner.verify_phase(seal(phase), index=2, arm="tail128_dynamic", timing=True, quality=False)


if __name__ == "__main__":
    unittest.main()
