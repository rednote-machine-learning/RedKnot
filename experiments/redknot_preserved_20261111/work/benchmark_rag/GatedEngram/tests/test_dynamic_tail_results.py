"""Independent dynamic refresh and frozen native configuration checks."""
import copy
import random
import unittest
from bisect import bisect_right

from gpu_pilot import read_tail_link_results as reader


class DynamicSWAEventTests(unittest.TestCase):
    spans = [(1, 32769, "a"*64), (32769, 65537, "b"*64)]

    def fixture(self, *, layer=21, positions=None, start=32768, end=40960,
                previous=None, qualification=True, native_yoco_swa_share_hidden=False):
        positions = [32768, 40959] if positions is None else positions
        previous = set() if previous is None else set(previous)
        tail_slots = {p: ordinal*128+p-(hi-128)
                      for ordinal, (_, hi, _) in enumerate(self.spans) for p in range(hi-128, hi)}
        tail_ids = [i for i, p in enumerate(positions) if p in tail_slots]
        enabled = layer >= 4
        ids = tail_ids if enabled else []
        refreshed = [positions[i] for i in ids]
        available = previous | set(refreshed)
        cutoff = positions[-1] if positions else -1
        visible = {p for p in tail_slots if p <= cutoff}
        checked = qualification and bool(ids)
        event = dict(schema="tail-link-dynamic-swa-v1", layer=layer,
            chunk_start=start, chunk_end=end, phase_generation=17,
            refresh_enabled=enabled,
            source="current_layer_attention_input" if enabled else "offline_layer3_seed",
            input_representation=("bf16_current" if native_yoco_swa_share_hidden and layer >= 20
                                  else "native_packed_current"),
            current_tail_local_rows=tail_ids, current_tail_positions=[positions[i] for i in tail_ids],
            refreshed_local_rows=ids, refreshed_positions=refreshed,
            refreshed_bank_rows=[tail_slots[p] for p in refreshed],
            available_refreshed_positions=sorted(available),
            visible_refreshed_positions=sorted(visible & available),
            visible_offline_fallback_positions=sorted(visible-available),
            projection_identity_checked=checked, prepare_identity_checked=checked,
            checks=2 if checked else 0, max_absolute_error=0.0 if checked else None)
        return event, positions, tail_ids, previous, qualification

    def audit(self, fixture, *, native_yoco_swa_share_hidden=False):
        event, positions, tail_ids, prior, qualification = fixture
        return reader.audit_dynamic_swa_event(event, positions, tail_ids,
            self.spans, prior, generation=17, qualification=qualification,
            native_yoco_swa_share_hidden=native_yoco_swa_share_hidden)

    def test_surviving_yoco_rows_do_not_imply_shared_bf16_swa_input(self):
        for layer in (20, 24, 39):
            for shared in (False, True):
                with self.subTest(layer=layer, shared=shared):
                    fixture = self.fixture(layer=layer, native_yoco_swa_share_hidden=shared)
                    self.assertEqual(self.audit(fixture, native_yoco_swa_share_hidden=shared)["refreshed_rows"], 1)
                    fixture = self.fixture(layer=layer, native_yoco_swa_share_hidden=shared)
                    with self.assertRaisesRegex(ValueError, f"layer={layer}.*input_representation"):
                        self.audit(fixture, native_yoco_swa_share_hidden=not shared)

    def test_representation_verified_even_when_current_chunk_has_no_tail(self):
        for layer in (20, 24, 39):
            fixture = self.fixture(layer=layer, start=0, end=8192, positions=[8191])
            self.assertEqual(self.audit(fixture)["refreshed_rows"], 0)
            fixture[0]["input_representation"] = "bf16_current"
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "input_representation"):
                self.audit(fixture)

    def test_helper_native_configuration_requires_real_boolean(self):
        for flag in (None, 0, 1, "false"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "explicit bool"):
                self.audit(self.fixture(), native_yoco_swa_share_hidden=flag)

    def test_shared_configuration_still_uses_packed_input_before_yoco_boundary(self):
        fixture = self.fixture(layer=19, native_yoco_swa_share_hidden=True)
        self.assertEqual(fixture[0]["input_representation"], "native_packed_current")
        self.assertEqual(self.audit(fixture, native_yoco_swa_share_hidden=True)["refreshed_rows"], 1)

    def test_surviving_tail_refresh_keeps_yoco_missing_tail_as_offline(self):
        fixture = self.fixture()
        result = self.audit(fixture)
        self.assertEqual(result, dict(refreshed_rows=1, visible_refreshed_rows=1,
                                    visible_offline_fallback_rows=127))
        self.assertEqual(fixture[3], {32768})

    def test_layer3_uses_seed_and_never_updates_even_current_tail(self):
        fixture = self.fixture(layer=3)
        result = self.audit(fixture)
        self.assertEqual(result["refreshed_rows"], 0)
        self.assertEqual(result["visible_offline_fallback_rows"], 128)
        self.assertEqual(fixture[3], set())

    def test_prior_chunk_current_layer_refresh_remains_visible(self):
        fixture = self.fixture(previous={32767})
        result = self.audit(fixture)
        self.assertEqual(result["visible_refreshed_rows"], 2)
        self.assertEqual(result["visible_offline_fallback_rows"], 126)

    def test_empty_current_layout_does_not_fabricate_visible_keys(self):
        fixture = self.fixture(positions=[], previous={32767})
        result = self.audit(fixture)
        self.assertEqual(result, dict(refreshed_rows=0, visible_refreshed_rows=0,
                                    visible_offline_fallback_rows=0))

    def test_formal_refresh_does_not_claim_qualification_checks(self):
        fixture = self.fixture(qualification=False)
        self.assertEqual(self.audit(fixture)["refreshed_rows"], 1)

    def test_wrong_policy_source_mapping_and_lifecycle_are_rejected(self):
        changes = {
            "source": "boundary_shared_hidden",
            "phase_generation": 16,
            "input_representation": "bf16_current",
            "refresh_enabled": False,
            "refreshed_local_rows": [],
            "refreshed_positions": [32767],
            "refreshed_bank_rows": [0],
            "available_refreshed_positions": [32767, 32768],
            "visible_refreshed_positions": [32767, 32768],
            "visible_offline_fallback_positions": [],
            "projection_identity_checked": False,
            "prepare_identity_checked": False,
            "checks": 1,
            "max_absolute_error": 1e-12,
        }
        for key, value in changes.items():
            fixture = self.fixture(); fixture[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(fixture)

    def test_same_phase_position_cannot_be_rewritten_twice(self):
        fixture = self.fixture(previous={32768})
        with self.assertRaises(ValueError): self.audit(fixture)

    def test_seed_cannot_claim_online_refresh(self):
        fixture = self.fixture(layer=3)
        fixture[0]["refreshed_local_rows"] = fixture[2]
        with self.assertRaises(ValueError): self.audit(fixture)

    def test_future_tail_cannot_be_made_visible(self):
        fixture = self.fixture()
        fixture[0]["visible_refreshed_positions"].append(65536)
        with self.assertRaises(ValueError): self.audit(fixture)

    def test_unsorted_or_duplicate_current_positions_rejected(self):
        for positions in ([32768, 32768], [40959, 32768]):
            fixture = self.fixture(positions=positions)
            with self.subTest(positions=positions), self.assertRaises(ValueError): self.audit(fixture)


class DynamicControllerAuditTests(unittest.TestCase):
    def fixture(self):
        from tests.test_tail_link_pilot import ReaderTests, spans
        fixture_owner = ReaderTests()
        control, row, phase = fixture_owner.controller_fixture()
        phase["arm"] = "tail128_dynamic"
        phase["dynamic_document_swa"] = True
        control.update(dynamic_document_swa=True, dynamic_refresh_start_layer=4, dynamic_swa_events=[],
                       native_yoco_swa_share_hidden=False)
        accumulated = set()
        slot = {p: ordinal*128+p-(hi-128) for ordinal, (_, hi, _) in enumerate(spans())
                for p in range(hi-128, hi)}
        for selection, swa in zip(control["selection"], control["swa_events"]):
            positions = selection["current_positions"]
            tail_ids = [i for i, p in enumerate(positions) if p in slot]
            tail_positions = [positions[i] for i in tail_ids]
            accumulated.update(tail_positions)
            cutoff = positions[-1] if positions else -1
            visible = {p for p in slot if p <= cutoff}
            event = dict(schema="tail-link-dynamic-swa-v1", layer=20,
                chunk_start=selection["start"], chunk_end=selection["end"],
                phase_generation=control["staging"]["phase_generation"],
                refresh_enabled=True, source="current_layer_attention_input", input_representation="native_packed_current",
                current_tail_local_rows=tail_ids, current_tail_positions=tail_positions,
                refreshed_local_rows=tail_ids, refreshed_positions=tail_positions,
                refreshed_bank_rows=[slot[p] for p in tail_positions],
                available_refreshed_positions=sorted(accumulated),
                visible_refreshed_positions=sorted(visible & accumulated),
                visible_offline_fallback_positions=sorted(visible-accumulated),
                projection_identity_checked=False, prepare_identity_checked=False,
                checks=0, max_absolute_error=None)
            control["dynamic_swa_events"].append(event)
            swa["online_local_rows"] = sorted(swa["online_local_rows"] + tail_ids)
            swa["online_rows"] += len(tail_ids)
            swa["document_online_rows"] = len(tail_ids)
            swa["reused_rows"] = 0
        return fixture_owner, control, row, phase

    def test_dynamic_audit_counts_actual_refresh_and_offline_fallback(self):
        owner, control, row, phase = self.fixture()
        result = owner.check_fixture(control, row, phase)
        self.assertGreater(result["dynamic_refreshed_rows"], 0)
        self.assertEqual(result["dynamic_refreshed_rows"],
                         sum(len(e["refreshed_positions"]) for e in control["dynamic_swa_events"]))
        self.assertGreaterEqual(result["dynamic_unrefreshed_yoco_tail_slots"], 0)
        self.assertLessEqual(result["dynamic_refreshed_tail_slot_fraction"], 1)
        self.assertGreater(result["consumed_memory_online_document_pairs"], 0)

    def test_missing_dynamic_chunk_proof_rejected(self):
        owner, control, row, phase = self.fixture()
        control["dynamic_swa_events"].pop()
        with self.assertRaises(ValueError): owner.check_fixture(control, row, phase)

    def test_formal_controller_rejects_unverified_or_nonfalse_native_configuration(self):
        for location in ("phase", "controller"):
            for flag in (True, None, 0, "false"):
                owner, control, row, phase = self.fixture()
                target = phase if location == "phase" else control
                target["native_yoco_swa_share_hidden"] = flag
                with self.subTest(location=location, flag=flag), \
                     self.assertRaisesRegex(ValueError, "native_yoco_swa_share_hidden"):
                    owner.check_fixture(control, row, phase)
        owner, control, row, phase = self.fixture()
        del control["native_yoco_swa_share_hidden"]
        with self.assertRaisesRegex(ValueError, "native_yoco_swa_share_hidden"):
            owner.check_fixture(control, row, phase)

    def test_false_configuration_cannot_accept_uniformly_bf16_labeled_events(self):
        owner, control, row, phase = self.fixture()
        for event in control["dynamic_swa_events"]:
            event["input_representation"] = "bf16_current"
        with self.assertRaisesRegex(ValueError, "layer=20.*input_representation"):
            owner.check_fixture(control, row, phase)

    def test_static_counters_cannot_hide_dynamic_compute(self):
        owner, control, row, phase = self.fixture()
        changed = next(event for event in control["swa_events"] if event["document_online_rows"])
        changed["document_online_rows"] = 0
        with self.assertRaises(ValueError): owner.check_fixture(control, row, phase)

    def test_missing_dynamic_smoke_prevents_final_completion(self):
        from tests.test_tail_link_pilot import ReaderTests
        args = list(ReaderTests().run_closure_fixture())
        args[-1] = None
        with self.assertRaises(ValueError): reader.validate_closed_run(*args)


class NativeConfigurationBindingTests(unittest.TestCase):
    def fixture(self):
        proof = dict(schema="tail-link-native-yoco-swa-configuration-v1",
            native_yoco_swa_share_hidden=False, hf_default=False,
            hf_parser_file="deepinfer/hugging_face/config.py", hf_parser_sha256="a"*64,
            checkpoint_config_sha256="b"*64, config_overrides=[], parser_override_absent=True, verified=True)
        preflight = dict(native_yoco_swa_share_hidden=False, manifest_sha256="c"*64,
            protocol=dict(native_yoco_swa_share_hidden=False, sha256="d"*64),
            identity=dict(native_yoco_swa_share_hidden=False, native_yoco_swa_configuration=proof,
                runtime_source_files_sha256={"deepinfer/hugging_face/config.py": "a"*64},
                checkpoint_config_sha256="b"*64, execution_configuration_sha256="e"*64))
        phase = dict(native_yoco_swa_share_hidden=False,
            controller=dict(native_yoco_swa_share_hidden=False), manifest_sha256="c"*64,
            tail_link_protocol_sha256="d"*64, execution_configuration_sha256="e"*64)
        return preflight, phase

    def test_binding_accepts_exact_false_with_frozen_parser_checkpoint_proof(self):
        preflight, phase = self.fixture()
        self.assertIs(reader.audit_native_swa_configuration(preflight), False)
        reader.binding(phase, preflight)
        preflight["identity"]["native_yoco_swa_configuration"]["config_overrides"] = [
            dict(path="/yoco_swa_share_hidden", value=False),
            dict(path="/nested/gpt/yoco_swa_share_hidden", value=False)]
        reader.binding(phase, preflight)

    def test_all_binding_copies_must_explicitly_be_false(self):
        for location in ("preflight", "protocol", "identity", "proof", "phase", "controller"):
            for invalid in (True, None, 0, "false", "missing"):
                preflight, phase = self.fixture()
                target = {"preflight": preflight, "protocol": preflight["protocol"],
                    "identity": preflight["identity"],
                    "proof": preflight["identity"]["native_yoco_swa_configuration"],
                    "phase": phase, "controller": phase["controller"]}[location]
                if invalid == "missing": del target["native_yoco_swa_share_hidden"]
                else: target["native_yoco_swa_share_hidden"] = invalid
                with self.subTest(location=location, invalid=invalid), self.assertRaises(ValueError):
                    reader.binding(phase, preflight)

    def test_false_claim_cannot_hide_wrong_or_unbound_configuration_evidence(self):
        changes = dict(schema="old", hf_default=True, parser_override_absent=False, verified=False,
            hf_parser_file="other.py", hf_parser_sha256="f"*64, checkpoint_config_sha256="f"*64,
            config_overrides=[dict(path="/yoco_swa_share_hidden", value=True)])
        for field, invalid in changes.items():
            preflight, phase = self.fixture()
            preflight["identity"]["native_yoco_swa_configuration"][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError): reader.binding(phase, preflight)

    def test_override_evidence_requires_valid_unique_paths_and_boolean_false(self):
        for overrides in (None, [dict(path="/x", value=False)],
                          [dict(path="/yoco_swa_share_hidden", value=0)],
                          [dict(path="/yoco_swa_share_hidden", value=False)]*2):
            preflight, phase = self.fixture()
            preflight["identity"]["native_yoco_swa_configuration"]["config_overrides"] = overrides
            with self.subTest(overrides=overrides), self.assertRaises(ValueError): reader.binding(phase, preflight)


class SortedCausalOracleTests(unittest.TestCase):
    def test_binary_prefix_count_equals_scalar_inclusive_causal_rule(self):
        rng = random.Random(148)
        for count in (0, 1, 2, 127, 128, 1024):
            keys = sorted(rng.sample(range(262145), count))
            queries = [-1, 0, 262144, 262145]
            queries += [x+d for x in keys[:20] for d in (-1, 0, 1)]
            queries += [rng.randrange(-1, 262146) for _ in range(300)]
            self.assertEqual([bisect_right(keys, query) for query in queries],
                             [sum(key <= query for key in keys) for query in queries])


if __name__ == "__main__": unittest.main()
