"""Synthetic CPU vectors; these tests do not measure a model or task accuracy."""
from dataclasses import replace
import json
import unittest

from kv_cache_engram.cache import CacheArtifact
from kv_cache_engram.core import CacheIdentity, ValidationError, token_digest
from kv_cache_engram.gate import GatePolicy, make_plan
from kv_cache_engram.reuse_audit import PairedKV, audit_pair, audit_splits


def make_pair(count=32, width=2, token_offset=0, document_start=3, layer=3):
    document = tuple(range(token_offset, token_offset+count))
    request = (999,) * document_start + document + (1000, 1001)
    identity = CacheIdentity('a'*64, 'b'*64, 'c'*64,
        token_digest(document), layer, count, width)
    rows = tuple((1.,) * width for _ in document)
    return PairedKV(CacheArtifact.build(identity, rows), identity, document,
        request, document_start, rows, 'synthetic-request', 'synthetic')


def policy(**changes):
    values = dict(enabled=True, block_size=16, probe_count=2, head_guard=2,
        tail_guard=2, relative_tolerance=.1, absolute_tolerance=.1,
        calibration_id='synthetic-test-only')
    values.update(changes)
    return GatePolicy(**values)


def audit(pair=None, config=None, **changes):
    values = dict(oracle_relative_tolerance=.1, oracle_absolute_tolerance=.1)
    values.update(changes)
    return audit_pair(make_pair() if pair is None else pair,
        policy() if config is None else config, **values)


def change_row(pair, position, value):
    rows = list(pair.online_rows)
    rows[position] = (value,) * pair.online_identity.width
    return replace(pair, online_rows=rows)


class PairContractTests(unittest.TestCase):
    def test_exact_pair_checks_full_identity_tokenization_position_and_shape(self):
        pair = make_pair()
        for name, value in [('model_sha256','f'*64), ('tokenizer_sha256','f'*64),
                            ('execution_sha256','f'*64), ('layer',4), ('width',3)]:
            with self.subTest(name=name), self.assertRaises(ValidationError):
                replace(pair, online_identity=replace(pair.online_identity, **{name:value}))
        for changes in [dict(document_token_ids=(9,)*32), dict(document_start=4),
                        dict(request_token_ids=(999,)*37), dict(online_rows=[(1.,)]*32),
                        dict(online_rows=[(1.,1.)]*31), dict(document_start=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                replace(pair, **changes)

    def test_rejects_invalid_tokens_outside_document_slice(self):
        pair=make_pair()
        request=list(pair.request_token_ids);request[-1]=True
        with self.assertRaises(ValidationError):replace(pair,request_token_ids=request)

    def test_requires_explicit_origin_and_real_capture_evidence_digest(self):
        pair=make_pair()
        with self.assertRaises(ValidationError):replace(pair,data_origin='unknown')
        with self.assertRaises(ValidationError):replace(pair,data_origin='model_capture')
        with self.assertRaises(ValidationError):replace(pair,capture_sha256='not-a-hash')
        observed=replace(pair,data_origin='model_capture',capture_sha256='e'*64)
        self.assertEqual(audit(observed)['data_origin_declared'],'model_capture')

    def test_damaged_cache_and_nonfinite_online_rows_rejected(self):
        pair=make_pair()
        with self.assertRaises(ValidationError):
            replace(pair,offline=replace(pair.offline,checksum='0'*64))
        for value in (float('nan'),float('inf'),True):
            with self.subTest(value=value),self.assertRaises(ValidationError):
                change_row(pair,1,value)

    def test_copies_rows_and_ids(self):
        pair=make_pair();rows=[list(row) for row in pair.online_rows]
        doc=list(pair.document_token_ids);request=list(pair.request_token_ids)
        copied=replace(pair,online_rows=rows,document_token_ids=doc,request_token_ids=request)
        rows[0][0]=44;doc[0]=55;request[0]=66
        self.assertEqual(copied.online_rows,pair.online_rows)
        self.assertEqual(copied.document_token_ids,pair.document_token_ids)
        self.assertEqual(copied.request_token_ids,pair.request_token_ids)


class ReuseAuditTests(unittest.TestCase):
    def test_matching_all_rows_reports_candidates_reused_and_fresh_probes_guards(self):
        result=audit()
        self.assertEqual(result['summary']['reused_rows'],24)
        self.assertEqual(result['summary']['online_callback_rows'],8)
        self.assertEqual(result['summary']['false_accept_blocks'],0)
        self.assertEqual(result['summary']['false_reject_blocks'],0)
        self.assertEqual(result['summary']['reused_row_fraction'],.75)
        self.assertEqual(result['by_boundary']['head_guard']['rows'],2)
        self.assertEqual(result['by_boundary']['tail_guard']['rows'],2)
        self.assertEqual(result['by_boundary']['interior']['rows'],28)
        self.assertEqual(result['tokens'][0]['absolute_position'],3)
        self.assertEqual(result['tokens'][31]['absolute_position'],34)
        self.assertEqual(sum(b['rows'] for b in result['by_position_decile'].values()),32)

    def test_unsampled_bad_row_demonstrates_false_acceptance(self):
        pair=make_pair();plan=make_plan(pair.binding(0,16),policy())
        missed=next(p for p in plan.positions if p not in plan.probe_positions and p not in plan.mandatory_positions)
        result=audit(change_row(pair,missed,20.))
        self.assertEqual(result['summary']['false_accept_blocks'],1)
        self.assertEqual(result['summary']['bad_reused_rows'],1)
        self.assertIn(missed,result['blocks'][0]['bad_reused_positions'])
        self.assertEqual(result['blocks'][0]['gate']['max_absolute_error'],0)
        self.assertEqual(result['blocks'][0]['unsampled_candidate_errors']['max_absolute_error'],19)

    def test_bad_probe_but_good_candidates_is_false_rejection_and_counts_probe_overhead(self):
        pair=make_pair();plan=make_plan(pair.binding(0,16),policy())
        result=audit(change_row(pair,plan.probe_positions[0],20.))
        block=result['blocks'][0]
        self.assertTrue(block['false_reject'])
        self.assertEqual(block['online_callback_row_counts'],[2,16])
        self.assertEqual(result['summary']['online_callback_rows'],22)
        self.assertEqual(result['summary']['fallback_probe_overhead_rows'],2)
        self.assertEqual(result['summary']['bad_reused_rows'],0)
        self.assertFalse(result['tokens'][plan.probe_positions[0]]['reused'])

    def test_probe_and_candidate_bad_are_true_rejection(self):
        pair=make_pair();plan=make_plan(pair.binding(0,16),policy())
        candidate=next(p for p in plan.positions if p not in plan.probe_positions and p not in plan.mandatory_positions)
        pair=change_row(change_row(pair,plan.probe_positions[0],20.),candidate,20.)
        result=audit(pair)
        self.assertFalse(result['blocks'][0]['oracle_allows_candidate_reuse'])
        self.assertFalse(result['blocks'][0]['false_accept'])
        self.assertFalse(result['blocks'][0]['false_reject'])

    def test_bad_guard_is_reported_but_not_counted_as_bad_reuse(self):
        result=audit(change_row(make_pair(),0,20.))
        self.assertEqual(result['by_boundary']['head_guard']['outside_oracle_tolerances'],1)
        self.assertEqual(result['summary']['bad_reused_rows'],0)
        self.assertEqual(result['summary']['false_accept_blocks'],0)
        self.assertFalse(result['tokens'][0]['reused'])

    def test_disabled_policy_counts_full_without_spurious_false_rejection(self):
        result=audit(config=policy(enabled=False))
        self.assertEqual(result['summary']['online_callback_rows'],32)
        self.assertEqual(result['summary']['reused_rows'],0)
        self.assertEqual(result['summary']['false_reject_blocks'],0)
        self.assertEqual(result['summary']['compared_blocks'],0)

    def test_all_guarded_short_document_never_claims_reuse(self):
        result=audit(make_pair(count=3),policy(head_guard=2,tail_guard=2))
        self.assertEqual(result['summary']['online_callback_rows'],3)
        self.assertEqual(result['summary']['reused_rows'],0)
        self.assertEqual(result['by_boundary']['both_guards']['rows'],1)
        self.assertEqual(result['summary']['compared_blocks'],0)

    def test_all_probed_not_mislabeled_reuse(self):
        result=audit(make_pair(count=3),policy(head_guard=0,tail_guard=0,probe_count=9))
        self.assertEqual(result['summary']['online_callback_rows'],3)
        self.assertEqual(result['summary']['reused_rows'],0)
        self.assertEqual(result['blocks'][0]['mode'],'full_recompute')
        self.assertEqual(result['summary']['compared_blocks'],0)

    def test_tail_block_complete_and_layer_positions_retained(self):
        result=audit(make_pair(count=19,document_start=127,layer=18))
        self.assertEqual(result['blocks'][-1]['end'],19)
        self.assertEqual(result['blocks'][-1]['absolute_end'],146)
        self.assertEqual(result['tokens'][-1]['layer'],18)
        self.assertEqual(len(result['tokens']),19)

    def test_oracle_thresholds_explicit_separate_from_gate_and_inclusive(self):
        pair=make_pair();rows=tuple((2.,2.) for _ in pair.online_rows)
        pair=replace(pair,online_rows=rows)
        result=audit(pair,policy(relative_tolerance=1.,absolute_tolerance=2.),
            oracle_relative_tolerance=.5,oracle_absolute_tolerance=1.)
        self.assertEqual(result['summary']['bad_reused_rows'],0)
        stricter=audit(pair,policy(relative_tolerance=1.,absolute_tolerance=2.),
            oracle_relative_tolerance=.49,oracle_absolute_tolerance=1.)
        self.assertEqual(stricter['summary']['false_accept_blocks'],2)
        for name in ('oracle_relative_tolerance','oracle_absolute_tolerance'):
            for value in (-1.,float('nan'),float('inf'),True,None,10**1000):
                with self.subTest(name=name,value=value),self.assertRaises(ValidationError):
                    audit(**{name:value})

    def test_zero_online_norm_uses_explicit_floor(self):
        pair=make_pair();pair=replace(pair,online_rows=tuple((0.,0.) for _ in pair.online_rows))
        result=audit(pair,policy(norm_floor=.5))
        self.assertAlmostEqual(result['tokens'][0]['relative_l2'],2**.5/.5)

    def test_finite_input_metric_overflow_denies_and_serializes_strict_json(self):
        pair=make_pair();offline=CacheArtifact.build(pair.online_identity,tuple((-1e308,-1e308) for _ in pair.online_rows))
        pair=replace(pair,offline=offline,online_rows=tuple((1e308,1e308) for _ in pair.online_rows))
        result=audit(pair)
        self.assertEqual(result['summary']['reused_rows'],0)
        self.assertEqual(result['summary']['all_row_errors']['nonfinite_metrics'],32)
        self.assertIsNone(result['summary']['all_row_errors']['max_relative_l2'])
        self.assertEqual(result['blocks'][0]['gate']['reason'],'nonfinite_metric')
        json.dumps(result,allow_nan=False)

    def test_fixed_probes_do_not_change_when_data_changes(self):
        first=audit();changed=audit(change_row(make_pair(),10,30.))
        self.assertEqual([b['probe_positions'] for b in first['blocks']],
                         [b['probe_positions'] for b in changed['blocks']])

    def test_fallback_extra_rows_counted_when_all_blocks_rejected(self):
        pair=make_pair();pair=replace(pair,online_rows=tuple((20.,20.) for _ in pair.online_rows))
        result=audit(pair)
        self.assertEqual(result['summary']['online_callback_rows'],36)
        self.assertEqual(result['summary']['extra_callback_rows_over_full'],4)
        self.assertEqual(result['summary']['fallback_probe_overhead_rows'],4)


class SplitAuditTests(unittest.TestCase):
    def run_splits(self,calibration,evaluation):
        return audit_splits(calibration,evaluation,policy(),
            oracle_relative_tolerance=.1,oracle_absolute_tolerance=.1)

    def test_disjoint_splits_retain_every_pair_and_aggregate_by_layer(self):
        pairs=[make_pair(token_offset=100),make_pair(token_offset=200,layer=18)]
        result=self.run_splits([make_pair()],pairs)
        self.assertFalse(result['thresholds_fitted'])
        self.assertEqual(result['splits']['evaluation']['pair_count'],2)
        self.assertEqual(set(result['splits']['evaluation']['by_layer']),{'3','18'})
        self.assertEqual(result['splits']['evaluation']['by_layer']['18']['reused_rows'],24)
        json.dumps(result,allow_nan=False)

    def test_same_document_across_context_or_layer_cannot_leak(self):
        for pair in [make_pair(document_start=7),make_pair(layer=18)]:
            with self.subTest(pair=pair),self.assertRaises(ValidationError):
                self.run_splits([make_pair()],[pair])

    def test_duplicates_or_empty_splits_are_rejected(self):
        pair=make_pair(token_offset=100)
        for left,right in [([],[pair]),([make_pair()],[]),([make_pair()],[pair,pair])]:
            with self.subTest(left=left,right=right),self.assertRaises(ValidationError):
                self.run_splits(left,right)

    def test_multiple_contexts_of_same_document_within_split_are_retained(self):
        result=self.run_splits([make_pair()],
            [make_pair(token_offset=100),make_pair(token_offset=100,document_start=7)])
        self.assertEqual(result['splits']['evaluation']['pair_count'],2)
        self.assertEqual(result['splits']['evaluation']['document_count'],1)

    def test_mixed_execution_regimes_are_not_aggregated(self):
        pair=make_pair(token_offset=100)
        identity=replace(pair.online_identity,execution_sha256='f'*64)
        changed=replace(pair,online_identity=identity,
                        offline=CacheArtifact.build(identity,pair.offline.rows))
        with self.assertRaises(ValidationError):self.run_splits([make_pair()],[changed])

    def test_synthetic_and_declared_model_data_not_aggregated(self):
        pair=replace(make_pair(token_offset=100),data_origin='model_capture',capture_sha256='e'*64)
        with self.assertRaises(ValidationError):self.run_splits([make_pair()],[pair])


if __name__ == '__main__':
    unittest.main()
