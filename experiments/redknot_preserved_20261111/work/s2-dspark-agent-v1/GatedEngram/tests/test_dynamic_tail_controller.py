"""Dynamic document SWA contracts; CPU mocks are not GPU qualification."""
import copy
import importlib.util
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from kv_cache_engram.core import ValidationError
from kv_cache_engram.index_hit20 import IndexHit20Controller
from tests.test_tail_link_controller import controller
from tests import test_tail_link_controller as static_fixtures


class DynamicPolicyTests(unittest.TestCase):
    def installation_fixture(self):
        c = controller()
        for layer in c.layers:
            layer.self_attn = NS(yoco_swa_share_hidden=False)
        return c

    def test_install_binds_actual_false_flags_before_native_hooks(self):
        c = self.installation_fixture()
        with patch.object(IndexHit20Controller, 'install', return_value=c) as install:
            self.assertIs(c.install(), c)
        install.assert_called_once_with()
        self.assertIs(c.native_yoco_swa_share_hidden, False)

    def test_install_rejects_true_missing_or_nonboolean_model_flags(self):
        for layer_index in (0, 3, 20, 39):
            for flag in (True, 0, None, 'false'):
                c = self.installation_fixture()
                c.layers[layer_index].self_attn.yoco_swa_share_hidden = flag
                with self.subTest(layer=layer_index, flag=flag), \
                     patch.object(IndexHit20Controller, 'install') as install, \
                     self.assertRaisesRegex(ValidationError, 'native_yoco_swa_share_hidden=False'):
                    c.install()
                install.assert_not_called()
                self.assertIsNone(c.native_yoco_swa_share_hidden)
        c = self.installation_fixture()
        del c.layers[24].self_attn.yoco_swa_share_hidden
        with self.assertRaises(ValidationError): c.install()

    def test_native_report_exports_verified_binding_and_rejects_later_drift(self):
        c = self.installation_fixture()
        with patch.object(IndexHit20Controller, 'install', return_value=c): c.install()
        c.mode = 'native'
        with patch.object(IndexHit20Controller, 'finish', return_value={}):
            self.assertIs(c.finish()['native_yoco_swa_share_hidden'], False)
            c.layers[20].self_attn.yoco_swa_share_hidden = True
            with self.assertRaises(ValidationError): c.finish()

    def test_real_controller_placeholder_event_is_accepted_by_reader_for_explicit_flag(self):
        from gpu_pilot.read_tail_link_results import audit_dynamic_swa_event
        spans = [(0, 32768, 'a'*64)]
        for layer in (20, 24, 39):
            for actual_flag in (False, True):
                c = controller(); c.t = None; c.dynamic_document_swa = True
                c.phase_generation = 17; c.mode = 'reference_sparse'; c.verify_sparse = False
                c.tail_positions_cpu = list(range(32640, 32768))
                mod = NS(layer_idx=layer, yoco_swa_share_hidden=actual_flag)
                meta = NS(canonical_request_binding={'start': 0, 'end': 8192})
                c._refresh_document_tails(mod, None, None, None, None, meta, [8191], None)
                event = c.dynamic_swa_events[0]
                with self.subTest(layer=layer, flag=actual_flag):
                    result = audit_dynamic_swa_event(event, [8191], [], spans, set(),
                        generation=17, qualification=True,
                        native_yoco_swa_share_hidden=actual_flag)
                    self.assertEqual(result['refreshed_rows'], 0)
                    with self.assertRaises(ValueError):
                        audit_dynamic_swa_event(event, [8191], [], spans, set(),
                            generation=17, qualification=True,
                            native_yoco_swa_share_hidden=not actual_flag)

    def test_policy_qualification_cannot_admit_other_policy(self):
        c = controller()
        c.sparse_reference = {(0, 8, 3): 'static'}
        c.sparse_reference_input = 'static-input'
        c.sparse_qualified = True
        c._bind_swa_policy(True)
        self.assertEqual(c.sparse_reference, {})
        self.assertIsNone(c.sparse_reference_input)
        self.assertFalse(c.sparse_qualified)
        c.sparse_reference = {(0, 8, 3): 'dynamic'}
        c.sparse_reference_input = 'dynamic-input'
        c.sparse_qualified = True
        c._bind_swa_policy(False)
        self.assertEqual(c.sparse_reference, {(0, 8, 3): 'static'})
        self.assertEqual(c.sparse_reference_input, 'static-input')
        self.assertTrue(c.sparse_qualified)
        c._bind_swa_policy(True)
        self.assertEqual(c.sparse_reference, {(0, 8, 3): 'dynamic'})
        self.assertEqual(c.sparse_reference_input, 'dynamic-input')

    def test_invalid_policy_rejected_before_mutation(self):
        c = controller()
        for value in [1, 'dynamic', None]:
            with self.assertRaises(ValidationError):
                c.reset('r', 'sensitive', input_ids=[1], phase='p', dynamic_document_swa=value)
        with self.assertRaises(ValidationError):
            c.reset('r', 'native', input_ids=[1], phase='p', dynamic_document_swa=True)
        self.assertFalse(c.dynamic_document_swa)

    def test_phase_cleanup_removes_all_online_overlays(self):
        c = controller()
        c.dynamic_refreshed = {4: {1, 2}}
        c.dynamic_chunk_end = {4: 8192}
        c.dynamic_swa_events = [object()]
        c.restored = {4: object()}
        c._clear_phase_gpu()
        self.assertEqual(c.dynamic_refreshed, {})
        self.assertEqual(c.dynamic_chunk_end, {})
        self.assertEqual(c.dynamic_swa_events, [])
        self.assertEqual(c.restored, {})


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,
                     'requires CPU torch; must pass unskipped on remote')
class DynamicTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch

    def refresh_fixture(self, *, layer=4, yoco=False, qualification=False):
        t = self.t
        c = controller(); c.t = t
        c.dynamic_document_swa = True
        c.phase_generation = 9
        c.mode = 'reference_sparse' if qualification else 'sensitive'
        c.verify_sparse = False
        c.tail_positions_cpu = [10, 11, 20, 21]
        positions = [9, 10, 11]
        meta = NS(total_q=3, positions=t.tensor(positions, dtype=t.int32),
                  canonical_request_binding={'start': 0, 'end': 12})
        mod = NS(layer_idx=layer, yoco_swa_share_hidden=yoco)
        hidden = t.full((3, 512), 3, dtype=t.bfloat16)
        bf16 = t.full((3, 512), 5, dtype=t.bfloat16)
        shared = t.full((3, 512), 7, dtype=t.bfloat16)
        projection_calls = []
        def projection(value):
            projection_calls.append(value.clone())
            return value.clone()
        c.projections[layer] = (None, projection)
        def prepare(raw, cache, md, use_cp):
            return (raw * 2 + (md.positions % 7)[:, None]).reshape(-1, 1, 512), None
        c.native_prepares[layer] = prepare
        def full_projection(current_hidden, current_shared):
            return None, (current_shared if yoco else current_hidden).clone()
        mod._project_q_and_swa = full_projection
        def identity(a, b):
            if not t.equal(a, b):
                raise ValidationError('independent equality failed')
        c.ops = NS(rows=lambda value: len(value), identity_error=identity,
                   normalize=lambda mod, raw: raw * 2,
                   postprocess=lambda mod, canonical, md:
                       ((canonical + (md.positions % 7)[:, None]).reshape(-1, 1, 512), None))
        cache = NS(swa_cache=None)
        tails = t.tensor([30, 31, 32, 33], dtype=t.bfloat16)[:, None, None].repeat(1, 1, 512)
        return c, mod, hidden, bf16, shared, cache, meta, positions, tails, projection_calls

    def apply(self, fixture):
        c, mod, hidden, bf16, shared, cache, meta, positions, tails, _ = fixture
        return c._refresh_document_tails(mod, hidden, bf16, shared, cache, meta, positions, tails)

    def test_layer3_offline_seed_remains_identical(self):
        f = self.refresh_fixture(layer=3); original = f[8].clone()
        self.assertEqual(self.apply(f), [])
        self.assertTrue(self.t.equal(f[8], original)); self.assertEqual(f[9], [])
        event = f[0].dynamic_swa_events[0]
        self.assertEqual(event['source'], 'offline_layer3_seed')
        self.assertEqual(event['visible_offline_fallback_positions'], [10, 11])
        self.assertEqual(event['refreshed_positions'], [])

    def test_layer4_current_input_norm_absolute_rope_and_cast_path(self):
        f = self.refresh_fixture(qualification=True)
        self.assertEqual(self.apply(f), [1, 2])
        self.assertEqual(f[8][:, 0, 0].tolist(), [9, 10, 32, 33])
        event = f[0].dynamic_swa_events[0]
        self.assertEqual(event['refreshed_positions'], [10, 11])
        self.assertEqual(event['refreshed_bank_rows'], [0, 1])
        self.assertEqual(event['checks'], 2)
        self.assertTrue(event['prepare_identity_checked'])
        self.assertEqual(event['visible_offline_fallback_positions'], [])

    def test_yoco_tail_uses_current_bf16_not_fixed_shared_hidden(self):
        f = self.refresh_fixture(layer=24, yoco=True, qualification=True)
        self.apply(f)
        self.assertEqual(f[8][:, 0, 0].tolist(), [13, 14, 32, 33])
        self.assertTrue(self.t.equal(f[9][0], f[3][1:]))
        self.assertFalse(self.t.equal(f[9][0], f[4][1:]))
        self.assertEqual(f[0].dynamic_swa_events[0]['input_representation'], 'bf16_current')

    def test_independent_full_projection_detects_wrong_subset_projection(self):
        f = self.refresh_fixture(qualification=True)
        f[0].projections[4] = (None, lambda value: value + 1)
        with self.assertRaises(ValidationError): self.apply(f)

    def test_independent_norm_rope_path_detects_wrong_position_transform(self):
        f = self.refresh_fixture(qualification=True)
        f[0].native_prepares[4] = lambda raw, cache, md, cp: (raw.reshape(-1, 1, 512), None)
        with self.assertRaises(ValidationError): self.apply(f)

    def test_no_refresh_when_yoco_surviving_placeholder_is_not_tail(self):
        f = list(self.refresh_fixture(layer=24, yoco=True))
        f[6].total_q = 1; f[6].positions = self.t.tensor([9], dtype=self.t.int32)
        f[7] = [9]
        f[2] = f[2][:1]; f[3] = f[3][:1]
        original = f[8].clone()
        self.assertEqual(self.apply(f), [])
        self.assertTrue(self.t.equal(f[8], original))
        self.assertEqual(f[0].dynamic_swa_events[0]['refreshed_positions'], [])

    def test_missing_yoco_tail_positions_retain_layer_specific_offline_value(self):
        f = list(self.refresh_fixture(layer=24, yoco=True))
        f[6].positions = self.t.tensor([9, 11, 12], dtype=self.t.int32)
        f[6].canonical_request_binding['end'] = 13
        f[7] = [9, 11, 12]
        self.apply(f)
        self.assertEqual(f[8][:, 0, 0].tolist(), [30, 14, 32, 33])
        self.assertEqual(f[0].dynamic_swa_events[0]['visible_offline_fallback_positions'], [10])

    def test_prior_chunk_refreshed_values_survive_and_future_values_not_replaced(self):
        f = list(self.refresh_fixture()); self.apply(f)
        first = f[8][:2].clone()
        f[6].positions = self.t.tensor([12, 20, 21], dtype=self.t.int32)
        f[6].canonical_request_binding = {'start': 12, 'end': 22}
        f[7] = [12, 20, 21]; f[2] = f[2] + 10
        self.apply(f)
        self.assertTrue(self.t.equal(f[8][:2], first))
        self.assertEqual(f[8][2:, 0, 0].tolist(), [32, 26])
        event = f[0].dynamic_swa_events[-1]
        self.assertEqual(event['available_refreshed_positions'], [10, 11, 20, 21])
        self.assertEqual(event['phase_generation'], 9)

    def test_duplicate_or_skipped_chunk_cannot_rewrite_history(self):
        f = self.refresh_fixture(); self.apply(f)
        with self.assertRaises(ValidationError): self.apply(f)
        f = self.refresh_fixture()
        f[6].canonical_request_binding['start'] = 1
        with self.assertRaises(ValidationError): self.apply(f)

    def test_layers_do_not_reuse_other_layers_refreshed_values(self):
        f = list(self.refresh_fixture()); self.apply(f)
        f[1] = copy.copy(f[1]); f[1].layer_idx = 5
        f[0].projections[5] = f[0].projections[4]
        f[0].native_prepares[5] = f[0].native_prepares[4]
        f[8] = self.t.full_like(f[8], 90)
        self.apply(f)
        self.assertEqual(f[8][:, 0, 0].tolist(), [9, 10, 90, 90])

    def test_dynamic_full_forward_keeps_nondoc_shared_and_global_native(self):
        c, mod, meta, cache, cross, h, calls, modules = static_fixtures.TorchForwardTests.fixture(self)
        old_layer = mod.layer_idx; mod.layer_idx = 24; mod.yoco_swa_share_hidden = True
        c.projections[24] = c.projections[old_layer]
        c.q_b_projections[24] = c.q_b_projections[old_layer]
        c.native_prepares[24] = c.native_prepares[old_layer]
        c.dynamic_document_swa = True
        c.dynamic_chunk_end[24] = meta.canonical_request_binding['start']
        cross.shared_hidden = self.t.full_like(h, 99)
        bf16 = self.t.full_like(h, 42)
        with patch.dict(sys.modules, modules):
            c._selected_attention(mod, h, cache, meta, bf16, cross)
        self.assertIn(('global_project', 5), calls)
        self.assertIn(('global_index', 5), calls)
        stored = next(row[1] for row in calls if row[0] == 'store')
        self.assertEqual(stored[:, 0, 0].tolist(), [42, 42, 99])
        self.assertEqual(c.swa_events[0]['document_online_rows'], 2)
        self.assertEqual(c.swa_events[0]['non_document_online_rows'], 1)
        self.assertEqual(c.swa_events[0]['online_local_rows'].tolist(), [2, 3, 4])
        self.assertEqual(c.swa_events[0]['reused_rows'], 0)
        joint = next(row for row in calls if row[0] == 'joint')
        # Query at first tail position sees only its own causal prefix, even
        # though the overlay has already prepared the later same-chunk tail.
        self.assertEqual((joint[3][0] >= 0).sum().item(), 1)

    def test_real_refresh_events_roundtrip_reader_actual_tail_and_cross_chunk(self):
        from gpu_pilot.read_tail_link_results import audit_dynamic_swa_event
        # Small synthetic row fixtures exercise the event contract; the real
        # 8x32K GPU gate remains separately required by the launcher.
        spans = [(0, 32768, 'a'*64)]
        for layer in (20, 24, 39):
            for actual_flag in (False, True):
                f = list(self.refresh_fixture(layer=layer, yoco=actual_flag, qualification=True))
                c, mod, h, bf16, shared, cache, meta, _, _, calls = f
                c.tail_positions_cpu = list(range(32640, 32768))
                meta.positions = self.t.tensor([32639, 32640, 32641], dtype=self.t.int32)
                meta.canonical_request_binding = {'start': 0, 'end': 32642}
                f[7] = [32639, 32640, 32641]
                f[8] = self.t.full((128, 1, 512), 90, dtype=self.t.bfloat16)
                self.apply(f)
                accumulated = set()
                event = c.dynamic_swa_events[-1]
                with self.subTest(layer=layer, flag=actual_flag, chunk=0):
                    result = audit_dynamic_swa_event(event, f[7], [1, 2], spans, accumulated,
                        generation=9, qualification=True,
                        native_yoco_swa_share_hidden=actual_flag)
                    self.assertEqual(result['refreshed_rows'], 2)
                    self.assertEqual(accumulated, {32640, 32641})
                first_values = f[8][:2].clone()
                meta.positions = self.t.tensor([32642, 32766, 32767], dtype=self.t.int32)
                meta.canonical_request_binding = {'start': 32642, 'end': 32768}
                f[7] = [32642, 32766, 32767]
                self.apply(f)
                self.assertTrue(self.t.equal(f[8][:2], first_values))
                event = c.dynamic_swa_events[-1]
                with self.subTest(layer=layer, flag=actual_flag, chunk=1):
                    result = audit_dynamic_swa_event(event, f[7], [0, 1, 2], spans, accumulated,
                        generation=9, qualification=True,
                        native_yoco_swa_share_hidden=actual_flag)
                    self.assertEqual(result['refreshed_rows'], 3)
                    self.assertEqual(result['visible_offline_fallback_rows'], 123)
                    self.assertEqual(accumulated, {32640, 32641, 32642, 32766, 32767})


if __name__ == '__main__':
    unittest.main()
