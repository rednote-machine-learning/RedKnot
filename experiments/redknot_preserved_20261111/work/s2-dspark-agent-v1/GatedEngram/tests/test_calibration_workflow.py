"""Deterministic CPU fixtures only; no model/answer accuracy claims."""
from dataclasses import asdict, replace
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.cache import CacheArtifact
from kv_cache_engram.calibration_workflow import (
    CalibrationExample, FrozenCalibration, evaluate_frozen, fit_candidates,
    from_paired_capture)
from kv_cache_engram.core import CacheIdentity, ValidationError, digest, token_digest
from kv_cache_engram.gate import GatePolicy, make_plan
from kv_cache_engram.reuse_audit import PairedKV


def example(offset=0, *, source=None, split=None, count=32, layer=3):
    doc = tuple(range(offset, offset+count))
    identity = CacheIdentity('a'*64, 'b'*64, 'c'*64, token_digest(doc), layer, count, 2)
    rows = ((1., 1.),)*count
    pair = PairedKV(CacheArtifact.build(identity, rows), identity, doc,
                    (99999,)+doc+(99998,), 1, rows, 'request-'+str(offset), 'synthetic')
    return CalibrationExample(pair, source or 'source-'+str(offset), split)


def policy(**changes):
    params = dict(enabled=True, block_size=16, probe_count=2, head_guard=2,
                  tail_guard=2, relative_tolerance=.1, absolute_tolerance=.1,
                  calibration_id='cpu-workflow-test-not-runtime')
    params.update(changes)
    return GatePolicy(**params)


def fit(examples=None, policies=None, **changes):
    limits = dict(oracle_relative_tolerance=.1, oracle_absolute_tolerance=.1,
                  max_false_accept_blocks=0, max_bad_reused_rows=0)
    limits.update(changes)
    return fit_candidates([example()] if examples is None else examples,
                          [policy()] if policies is None else policies, **limits)


def hidden_spike(item, config, *, overflow=False):
    pair = item.pair
    plan = make_plan(pair.binding(0, min(config.block_size, pair.online_identity.token_count)), config)
    pos = next(p for p in plan.positions if p not in plan.probe_positions
               and p not in plan.mandatory_positions)
    rows = list(pair.online_rows)
    rows[pos] = (1e308, 1e308) if overflow else (20., 20.)
    offline = pair.offline
    if overflow:
        cached = list(offline.rows)
        cached[pos] = (-1e308, -1e308)
        offline = CacheArtifact.build(pair.online_identity, cached)
    return replace(item, pair=replace(pair, online_rows=rows, offline=offline))


class CalibrationWorkflowTests(unittest.TestCase):
    def test_fit_freezes_full_evidence_and_all_policies(self):
        candidates = [policy(probe_count=4), policy()]
        frozen = fit([example(), example(100)], candidates)
        data = frozen.to_dict()['payload']
        self.assertEqual(frozen.mode, 'gated_reuse_candidate')
        self.assertEqual(frozen.selected_policy, policy())
        self.assertEqual(len(data['candidate_reports']), 2)
        self.assertEqual(len(data['calibration_inputs']), 2)
        for report in data['candidate_reports']:
            self.assertEqual(len(report['examples']), 2)
            self.assertEqual(len(report['examples'][0]['audit']['tokens']), 32)
        self.assertFalse(data['runtime_admission_created'])
        self.assertEqual(data['scope']['layer'], 3)
        self.assertEqual(data['calibration_inputs'][0]['online_rows_sha256'], digest(example().pair.online_rows))

    def test_largest_actual_reuse_is_selected_not_nominal_tolerance(self):
        less = policy(probe_count=6, relative_tolerance=100., absolute_tolerance=100.)
        frozen = fit(policies=[less, policy()])
        self.assertEqual(frozen.selected_policy, policy())
        self.assertEqual(frozen.to_dict()['payload']['candidate_reports'][0]['summary']['token_rows'], 32)

    def test_candidate_order_does_not_change_selection_or_artifact(self):
        a = policy(calibration_id='a')
        b = policy(calibration_id='b')
        first = fit(policies=[a, b])
        second = fit(policies=[b, a])
        self.assertEqual(first.seal_sha256, second.seal_sha256)
        expected = min((a, b), key=lambda p: digest(asdict(p)))
        self.assertEqual(first.selected_policy, expected)

    def test_callback_cost_breaks_equal_reuse_tie_before_hash(self):
        from kv_cache_engram.calibration_workflow import _selected
        expensive = dict(eligible=True, policy_sha256='0'*64,
                         summary=dict(reused_rows=20, online_callback_rows=18))
        cheap = dict(eligible=True, policy_sha256='f'*64,
                     summary=dict(reused_rows=20, online_callback_rows=15))
        self.assertIs(_selected([expensive, cheap]), cheap)

    def test_sparse_false_accept_is_rejected_by_full_oracle(self):
        bad = hidden_spike(example(), policy())
        frozen = fit([bad])
        candidate = frozen.to_dict()['payload']['candidate_reports'][0]
        self.assertEqual(candidate['summary']['false_accept_blocks'], 1)
        self.assertEqual(candidate['summary']['bad_reused_rows'], 1)
        self.assertFalse(candidate['eligible'])
        self.assertEqual(frozen.mode, 'full_recompute')
        self.assertIsNone(frozen.selected_policy)

    def test_bad_examples_retained_not_filtered_to_force_a_winner(self):
        frozen = fit([example(), hidden_spike(example(100), policy())])
        candidate = frozen.to_dict()['payload']['candidate_reports'][0]
        self.assertEqual(len(candidate['examples']), 2)
        self.assertEqual(candidate['summary']['token_rows'], 64)
        self.assertEqual(frozen.mode, 'full_recompute')

    def test_explicit_count_budgets_allow_only_requested_finite_violations(self):
        bad = hidden_spike(example(), policy())
        self.assertEqual(fit([bad], max_false_accept_blocks=1).mode, 'full_recompute')
        self.assertEqual(fit([bad], max_bad_reused_rows=1).mode, 'full_recompute')
        allowed = fit([bad], max_false_accept_blocks=1, max_bad_reused_rows=1)
        self.assertEqual(allowed.mode, 'gated_reuse_candidate')

    def test_heldout_failure_rejects_without_reselection_or_mutation(self):
        frozen = fit(policies=[policy(), policy(probe_count=32)])
        before = frozen.to_dict()
        result = evaluate_frozen(frozen, [hidden_spike(example(100), policy())])
        self.assertEqual(result['validation_result'], 'rejected')
        self.assertFalse(result['kv_validation_pass'])
        self.assertFalse(result['candidate_reselection_performed'])
        self.assertEqual(result['report']['summary']['bad_reused_rows'], 1)
        self.assertEqual(frozen.to_dict(), before)

    def test_heldout_pass_preserves_every_example_and_bounds(self):
        result = evaluate_frozen(fit(), [example(100), example(200)])
        self.assertEqual(result['validation_result'], 'kv_validation_pass')
        self.assertTrue(result['kv_validation_pass'])
        self.assertEqual(result['report']['summary']['reused_rows'], 48)
        self.assertEqual(len(result['report']['examples']), 2)
        self.assertFalse(result['runtime_admission_created'])

    def test_no_eligible_candidate_evaluates_disabled_full_recompute(self):
        frozen = fit(policies=[GatePolicy()])
        result = evaluate_frozen(frozen, [example(100)])
        self.assertFalse(result['kv_validation_pass'])
        self.assertEqual(result['reason'], 'no_eligible_candidate')
        self.assertEqual(result['report']['summary']['online_callback_rows'], 32)
        self.assertEqual(result['report']['summary']['reused_rows'], 0)

    def test_no_actual_reuse_on_heldout_is_not_validation_pass(self):
        result = evaluate_frozen(fit(), [example(100, count=3)])
        self.assertFalse(result['kv_validation_pass'])
        self.assertIn('no_actual_reuse', result['report']['rejection_reasons'])

    def test_same_source_group_rejected_even_with_different_tokens_and_context(self):
        with self.assertRaisesRegex(ValidationError, 'source document'):
            evaluate_frozen(fit(), [example(100, source='source-0')])

    def test_same_token_document_rejected_under_renamed_source_and_request(self):
        item = example(source='renamed')
        item = replace(item, pair=replace(item.pair, request_id='renamed-request',
            request_token_ids=(88888,)+item.pair.document_token_ids+(77777,)))
        with self.assertRaisesRegex(ValidationError, 'token hash'):
            evaluate_frozen(fit(), [item])

    def test_same_full_request_different_document_slices_rejected(self):
        train, heldout = example(), example(100)
        request = train.pair.document_token_ids+heldout.pair.document_token_ids+(99998,)
        train = replace(train, pair=replace(train.pair, request_token_ids=request, document_start=0))
        heldout = replace(heldout, pair=replace(heldout.pair, request_token_ids=request, document_start=32))
        with self.assertRaisesRegex(ValidationError, 'request context'):
            evaluate_frozen(fit([train]), [heldout])

    def test_nonfinite_oracle_arithmetic_not_waived_by_error_budgets(self):
        limits = dict(max_false_accept_blocks=99, max_bad_reused_rows=99)
        bad = hidden_spike(example(100), policy(), overflow=True)
        frozen = fit(**limits)
        result = evaluate_frozen(frozen, [bad])
        self.assertFalse(result['kv_validation_pass'])
        self.assertEqual(result['report']['summary']['nonfinite_oracle_metrics'], 1)
        self.assertIn('nonfinite_oracle_metrics', result['report']['rejection_reasons'])
        self.assertEqual(fit([bad], **limits).mode, 'full_recompute')

    def test_invalid_or_nonfinite_budget_or_tolerance_rejected(self):
        for name in ('max_false_accept_blocks', 'max_bad_reused_rows'):
            for value in (-1, 1.5, True, None, float('nan'), float('inf'), '0'):
                with self.subTest(name=name, value=value), self.assertRaises(ValidationError):
                    fit(**{name: value})
        for name in ('oracle_relative_tolerance', 'oracle_absolute_tolerance'):
            for value in (-1, True, None, float('nan'), float('inf'), 10**1000, '0'):
                with self.subTest(name=name, value=value), self.assertRaises(ValidationError):
                    fit(**{name: value})

    def test_mixed_model_execution_layer_and_source_scope_rejected(self):
        original = example(100)
        for key, value in [('model_sha256', 'd'*64), ('execution_sha256', 'd'*64), ('layer', 4)]:
            identity = replace(original.pair.online_identity, **{key: value})
            pair = replace(original.pair, online_identity=identity,
                           offline=CacheArtifact.build(identity, original.pair.offline.rows))
            changed = replace(original, pair=pair)
            with self.subTest(key=key), self.assertRaises(ValidationError):
                fit([example(), changed])
            with self.subTest(key=key), self.assertRaises(ValidationError):
                evaluate_frozen(fit(), [changed])
        native = replace(original, pair=replace(original.pair, data_origin='model_capture', capture_sha256='e'*64))
        with self.assertRaises(ValidationError): fit([example(), native])
        with self.assertRaises(ValidationError): evaluate_frozen(fit(), [native])

    def test_declared_split_must_match_operation(self):
        with self.assertRaises(ValidationError): fit([example(split='evaluation')])
        with self.assertRaises(ValidationError): evaluate_frozen(fit(), [example(100, split='calibration')])
        frozen = fit([example(split='calibration')])
        self.assertTrue(evaluate_frozen(frozen, [example(100, split='evaluation')])['kv_validation_pass'])

    def test_empty_invalid_duplicate_inputs_fail_whole_operation(self):
        for inputs in ([], [example(), None], [example(), example()]):
            with self.subTest(inputs=inputs), self.assertRaises(ValidationError): fit(inputs)
        for inputs in ([], [example(100), None], [example(100), example(100)]):
            with self.subTest(inputs=inputs), self.assertRaises(ValidationError): evaluate_frozen(fit(), inputs)
        for candidates in ([], [policy(), None], [policy(), policy()]):
            with self.subTest(candidates=candidates), self.assertRaises(ValidationError): fit(policies=candidates)

    def test_damaged_pair_raises_without_dropping_example(self):
        bad = example(100)
        object.__setattr__(bad.pair.offline, 'checksum', '0'*64)
        with self.assertRaises(ValidationError): fit([example(), bad])
        with self.assertRaises(ValidationError): evaluate_frozen(fit(), [example(200), bad])

    def test_artifact_roundtrip_and_copies_are_immutable(self):
        frozen = fit()
        envelope = frozen.to_dict()
        restored = FrozenCalibration.from_dict(json.loads(json.dumps(envelope)))
        self.assertEqual(restored, frozen)
        envelope['payload']['selected_policy']['enabled'] = False
        self.assertTrue(frozen.selected_policy.enabled)
        with self.assertRaises(ValidationError): FrozenCalibration.from_dict(envelope)

    def test_artifact_tampering_and_invalid_envelopes_rejected(self):
        frozen = fit()
        for key, value in [('scope', {}), ('selected_policy', None), ('limits', {})]:
            envelope = frozen.to_dict()
            envelope['payload'][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError): FrozenCalibration.from_dict(envelope)
        for envelope in ({}, {'payload': {}, 'seal_sha256': 'x'}, {'payload': {}, 'seal_sha256': 'a'*64}):
            with self.subTest(envelope=envelope), self.assertRaises(ValidationError): FrozenCalibration.from_dict(envelope)

    def test_resealed_inconsistent_selection_and_dropped_example_rejected(self):
        envelope = fit().to_dict()
        envelope['payload']['selected_policy'] = None
        envelope['seal_sha256'] = digest(envelope['payload'])
        with self.assertRaises(ValidationError): FrozenCalibration.from_dict(envelope)
        envelope = fit().to_dict()
        envelope['payload']['candidate_reports'][0]['examples'] = []
        envelope['seal_sha256'] = digest(envelope['payload'])
        with self.assertRaises(ValidationError): FrozenCalibration.from_dict(envelope)

    def test_fit_does_not_access_heldout_and_evaluate_does_not_refit(self):
        # The only pair supplied to fit is calibration; all audit calls bind it.
        from kv_cache_engram.calibration_workflow import audit_pair as real_audit
        train = example()
        with patch('kv_cache_engram.calibration_workflow.audit_pair', wraps=real_audit) as calls:
            frozen = fit([train], [policy(), policy(probe_count=3)])
            self.assertEqual(len(calls.call_args_list), 2)
            self.assertTrue(all(call.args[0] is train.pair for call in calls.call_args_list))
        with patch('kv_cache_engram.calibration_workflow.fit_candidates', side_effect=AssertionError('no refit')):
            self.assertTrue(evaluate_frozen(frozen, [example(100)])['kv_validation_pass'])

    def test_model_capture_is_declared_not_authenticated_or_runtime_admitted(self):
        train, heldout = example(), example(100)
        def labelled(item):
            return replace(item, pair=replace(item.pair, data_origin='model_capture', capture_sha256='f'*64))
        result = evaluate_frozen(fit([labelled(train)]), [labelled(heldout)])
        self.assertTrue(result['kv_validation_pass'])
        self.assertFalse(result['runtime_admission_created'])
        self.assertIn('not authentication', ' '.join(result['limitations']))

    def test_example_requires_explicit_nonblank_source_and_valid_capture_binding(self):
        for source in ('', ' ', ' leading', None, 12):
            with self.subTest(source=source), self.assertRaises(ValidationError): CalibrationExample(example().pair, source)
        with self.assertRaises(ValidationError): replace(example(), paired_capture_sha256='e'*64)
        with self.assertRaises(ValidationError): replace(example(), declared_split='test')


class PairedCaptureImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def capture(self, name, *, split='calibration', offset=0):
        from kv_cache_engram.paired_capture import capture_pair
        from test_paired_capture import pair_spec, descriptor
        original = pair_spec()
        doc = tuple(token+offset for token in original.independent.input_ids)
        independent = replace(original.independent, input_ids=doc,
                              request_id='independent-'+name)
        reference = replace(original.reference, input_ids=(101,102)+doc+(103,),
                            request_id='reference-'+name)
        spec = replace(original, pair_id=name, source_document_id='document-'+name,
                       split=split, independent=independent, reference=reference)
        @contextmanager
        def factory(side):
            yield descriptor(side)
        return capture_pair(self.path/name, spec, factory)

    def test_cpu_capture_import_freeze_and_validate_end_to_end(self):
        with self.capture('cal') as capture:
            train = from_paired_capture(capture, 3)
            self.assertEqual(train.source_document_id, 'document-cal')
            self.assertEqual(train.declared_split, 'calibration')
            self.assertEqual(train.paired_capture_sha256, capture.seal)
            self.assertEqual(train.pair.data_origin, 'synthetic')
        with self.capture('eval', split='evaluation', offset=30) as capture:
            heldout = from_paired_capture(capture, 3)
        config = policy(block_size=4, probe_count=1, head_guard=0, tail_guard=0,
                        relative_tolerance=100., absolute_tolerance=100.)
        frozen = fit([train], [config], oracle_relative_tolerance=100., oracle_absolute_tolerance=100.)
        result = evaluate_frozen(frozen, [heldout])
        self.assertTrue(result['kv_validation_pass'])
        self.assertFalse(result['runtime_admission_created'])
        self.assertEqual(result['heldout_inputs'][0]['paired_capture_sha256'], heldout.paired_capture_sha256)

    def test_imported_declared_evaluation_split_cannot_fit(self):
        with self.capture('eval', split='evaluation') as capture:
            item = from_paired_capture(capture, 3)
        with self.assertRaisesRegex(ValidationError, 'split'):
            fit([item])

    def test_import_requires_open_bound_valid_capture_and_explicit_budget(self):
        with self.assertRaises(ValidationError): from_paired_capture(None, 3)
        with self.capture('cal') as capture:
            with self.assertRaises(ValidationError): from_paired_capture(capture, 3, max_values=15)
            with self.assertRaises(ValidationError): from_paired_capture(capture, 18)
            altered = replace(capture.spec, source_document_id='forged-source')
            original = capture.spec
            capture.spec = altered
            with self.assertRaises(ValidationError): from_paired_capture(capture, 3)
            capture.spec = original
            self.assertEqual(from_paired_capture(capture, 3, max_values=16).source_document_id, 'document-cal')
        with self.assertRaises(ValidationError): from_paired_capture(capture, 3)


if __name__ == '__main__':
    unittest.main()
