"""CPU-only synthetic shadow contracts; these fixtures are not native evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

from gpu_pilot import pilot_protocol as pp
from kv_cache_engram.core import ValidationError, digest


def scope():
    return dict(model_sha256='a'*64, tokenizer_sha256='b'*64, execution_sha256='c'*64,
                representation='swa-postnorm-prerope-v1', storage_dtype='bfloat16-native-canonical',
                width=512, layers=list(pp.LAYERS))


def cases(split):
    return [dict(case_id=split+str(i), dataset=pp.DATASETS[i % 2], split=split,
                 input_sha256=digest([split, i, 'request']),
                 source_document_ids=[digest([split, i, 'source'])],
                 documents=[dict(document_sha256=digest([split, i, j, 'doc']), tokens=32768)
                            for j in range(8)])
            for i in range(2 if split == 'calibration' else 4)]


def report(case, policies):
    events = []
    for layer in pp.LAYERS:
        for policy in policies:
            checksum = digest(asdict(policy))
            for doc in case['documents']:
                for start in range(0, 32768, 512):
                    guards = 128 if start in (0, 32256) else 0
                    count = 512-guards-policy.probe_count
                    events.append(dict(layer=layer, document_sha256=doc['document_sha256'],
                        document_local_start=start, document_local_end=start+512,
                        candidate_id=checksum, candidate_sha256=checksum,
                        input_sha256=case['input_sha256'], request_id='request:'+case['case_id'],
                        phase='shadow:'+case['case_id'], probe_relative_max=0., probe_absolute_max=0.,
                        candidate_relative_max=0., candidate_absolute_max=0., full_relative_max=0.,
                        full_absolute_max=0., nonfinite_count=0, nonfinite=False,
                        candidate_reusable_rows=count, would_reuse_rows=count, actual_reused_rows=0,
                        accepted=True, bad_candidate_rows=0, oracle_bad_reused_rows=0, oracle_bad_rows=0,
                        probe_count=policy.probe_count, mandatory_count=guards, eligible_for_reuse=True,
                        mode='shadow_full', fallback_reason=None))
    return pp.seal_record(dict(schema='kv-cache-engram-native-shadow-report-v1',
        case_id=case['case_id'], dataset=case['dataset'], split=case['split'],
        input_sha256=case['input_sha256'], request_id='request:'+case['case_id'],
        phase='shadow:'+case['case_id'], scope=scope(), mode='shadow_full', complete=True,
        native_execution_declared=True, oracle=dict(pp.ORACLE), events=events))


def reseal(value):
    value = deepcopy(value)
    value.pop('sha256', None)
    return pp.seal_record(value)


def reports(split, policies):
    return [report(case, policies) for case in cases(split)]


class PilotProtocolTests(unittest.TestCase):
    def setUp(self):
        # Partition/gating contracts are exercised on one layer for cheap CPU
        # fixtures; a separate test fixes the actual fourteen-layer protocol.
        self.layer_patch = patch.object(pp, 'LAYERS', (3,))
        self.layer_patch.start()
        self.addCleanup(self.layer_patch.stop)
        self.policy = pp.candidate_policies()[0]
        self.grid = [self.policy]
        self.grid_patch = patch.object(pp, 'candidate_policies', side_effect=lambda: tuple(self.grid))
        self.grid_patch.start()
        self.addCleanup(self.grid_patch.stop)
        self.cal = reports('calibration', [self.policy])

    def fit(self, reports_=None, policies=None):
        self.grid = sorted([self.policy] if policies is None else policies, key=lambda p: digest(asdict(p)))
        return pp.fit_shadow(self.cal if reports_ is None else reports_, cases('calibration'), scope(),
                             [self.policy] if policies is None else policies)

    def test_protocol_grid_has_predeclared_engineering_limits(self):
        self.layer_patch.stop()
        self.grid_patch.stop()
        protocol = pp.pilot_protocol()
        self.assertEqual(protocol['layers'], [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18])
        self.assertEqual(len(protocol['candidate_policies']), 8)
        self.assertEqual(protocol['oracle'], {'relative_l2': .01, 'absolute_error': .05})
        self.assertFalse(protocol['em_f1_guarantee'])
        self.assertFalse(protocol['performance']['c4_measured'])

    def test_undeclared_policy_grid_cannot_claim_fixed_protocol(self):
        other = replace(self.policy, seed='not-predeclared')
        with self.assertRaises(ValidationError):
            pp.fit_shadow(reports('calibration', [other]), cases('calibration'), scope(), [other])

    def test_exact_shadow_selects_and_binds_full_case_coverage(self):
        frozen = self.fit()
        chosen = frozen['layers']['3']['selected']
        self.assertEqual(chosen['policy_sha256'], digest(asdict(self.policy)))
        self.assertEqual(chosen['summary']['total_rows'], 2*8*32768)
        self.assertEqual(chosen['summary']['actual_shadow_reused_rows'], 0)
        self.assertGreater(chosen['summary']['would_reuse_row_fraction'], .9)
        self.assertEqual(frozen['calibration_report_sha256'], [r['sha256'] for r in self.cal])
        self.assertFalse(frozen['runtime_admission_automatic'])

    def test_candidate_order_is_irrelevant_to_frozen_selection(self):
        other = replace(self.policy, seed='different')
        evidence = reports('calibration', [self.policy, other])
        a = self.fit(evidence, [self.policy, other])
        b = self.fit(evidence, [other, self.policy])
        self.assertEqual(a, b)
        self.assertEqual(a['layers']['3']['selected']['policy_sha256'],
                         min(digest(asdict(p)) for p in (self.policy, other)))

    def test_fewer_probes_wins_only_with_calibration_oracle_pass(self):
        many = replace(self.policy, probe_count=128)
        evidence = reports('calibration', [self.policy, many])
        self.assertEqual(self.fit(evidence, [many, self.policy])['layers']['3']['selected']['policy'],
                         asdict(self.policy))
        bad = evidence[0]['events'][0]
        bad.update(candidate_relative_max=.2, full_relative_max=.2, bad_candidate_rows=1,
                   oracle_bad_rows=1, oracle_bad_reused_rows=1)
        evidence[0] = reseal(evidence[0])
        frozen = self.fit(evidence, [many, self.policy])
        self.assertEqual(frozen['layers']['3']['selected']['policy'], asdict(many))
        self.assertEqual(len(frozen['layers']['3']['candidates']), 2)

    def test_unsampled_spike_false_accept_prevents_selection(self):
        bad = self.cal[0]['events'][0]
        bad.update(candidate_absolute_max=1., full_absolute_max=1., bad_candidate_rows=1,
                   oracle_bad_rows=1, oracle_bad_reused_rows=1)
        self.cal[0] = reseal(self.cal[0])
        frozen = self.fit()
        self.assertIsNone(frozen['layers']['3']['selected'])
        self.assertEqual(frozen['layers']['3']['mode'], 'full_recompute')

    def test_guard_only_difference_is_not_false_accept(self):
        event = self.cal[0]['events'][0]
        event.update(full_relative_max=.5, full_absolute_max=1., oracle_bad_rows=128)
        self.cal[0] = reseal(self.cal[0])
        chosen = self.fit()['layers']['3']['selected']
        self.assertEqual(chosen['summary']['false_accept_blocks'], 0)
        self.assertEqual(chosen['summary']['oracle_bad_rows'], 128)

    def test_nonfinite_cannot_be_waived(self):
        event = self.cal[0]['events'][0]
        event.update(nonfinite=True, nonfinite_count=1, accepted=False, would_reuse_rows=0,
                     probe_relative_max=None, probe_absolute_max=None)
        self.cal[0] = reseal(self.cal[0])
        self.assertIsNone(self.fit()['layers']['3']['selected'])

    def test_shadow_may_not_claim_actual_reuse(self):
        self.cal[0]['events'][0]['actual_reused_rows'] = 1
        self.cal[0] = reseal(self.cal[0])
        with self.assertRaises(ValidationError):
            self.fit()

    def test_gap_overlap_duplicate_and_missing_case_fail_closed(self):
        for mode in ('gap', 'duplicate', 'case'):
            with self.subTest(mode=mode):
                evidence = deepcopy(self.cal)
                if mode == 'case':
                    evidence.pop()
                elif mode == 'gap':
                    evidence[0]['events'].pop(0)
                else:
                    evidence[0]['events'].append(deepcopy(evidence[0]['events'][0]))
                if evidence:
                    evidence[0] = reseal(evidence[0])
                with self.assertRaises(ValidationError):
                    self.fit(evidence)

    def test_fragment_full_recompute_counts_no_candidate_rows(self):
        old = self.cal[0]['events'].pop(0)
        fragments = []
        for start, end in ((0, 485), (485, 512)):
            part = dict(old, document_local_start=start, document_local_end=end,
                        eligible_for_reuse=False, mode='full_recompute', fallback_reason='cross_chunk_fragment',
                        probe_count=0, mandatory_count=end-start, candidate_reusable_rows=0,
                        would_reuse_rows=0, accepted=False, probe_relative_max=None,
                        probe_absolute_max=None, candidate_relative_max=None, candidate_absolute_max=None)
            fragments.append(part)
        self.cal[0]['events'].extend(fragments)
        self.cal[0] = reseal(self.cal[0])
        summary = self.fit()['layers']['3']['selected']['summary']
        self.assertEqual(summary['blocks'], 2*8*64+1)
        self.assertEqual(summary['total_rows'], 2*8*32768)

    def test_wrong_request_or_policy_or_scope_or_seal_is_rejected(self):
        for field in ('request_id', 'candidate_sha256', 'scope', 'seal'):
            with self.subTest(field=field):
                evidence = deepcopy(self.cal)
                if field == 'scope':
                    evidence[0]['scope']['model_sha256'] = 'd'*64
                elif field == 'seal':
                    evidence[0]['complete'] = False
                else:
                    evidence[0]['events'][0][field] = 'f'*64
                if field != 'seal':
                    evidence[0] = reseal(evidence[0])
                with self.assertRaises(ValidationError):
                    self.fit(evidence)

    def test_probe_decision_must_follow_both_frozen_thresholds(self):
        self.cal[0]['events'][0]['probe_absolute_max'] = .001
        self.cal[0] = reseal(self.cal[0])
        with self.assertRaises(ValidationError):
            self.fit()

    def test_rejected_block_does_not_double_count_probe_projection(self):
        event = self.cal[0]['events'][0]
        event.update(probe_absolute_max=.001, full_absolute_max=.001, accepted=False, would_reuse_rows=0)
        self.cal[0] = reseal(self.cal[0])
        summary = self.fit()['layers']['3']['selected']['summary']
        self.assertEqual(summary['estimated_online_kv_callback_rows'],
                         summary['total_rows']-summary['would_reuse_rows'])

    def test_reusable_oracle_maxima_must_match_bad_count(self):
        self.cal[0]['events'][0]['candidate_relative_max'] = .2
        self.cal[0] = reseal(self.cal[0])
        with self.assertRaises(ValidationError):
            self.fit()

    def test_heldout_pass_has_no_reselection_or_runtime_admission(self):
        result = pp.validate_shadow(self.fit(), reports('evaluation', [self.policy]), cases('evaluation'))
        self.assertTrue(result['kv_validation_pass'])
        self.assertFalse(result['candidate_reselection_performed'])
        self.assertFalse(result['runtime_admission_automatic'])
        self.assertFalse(result['em_f1_guarantee'])

    def test_heldout_failure_is_reported_without_retuning(self):
        evaluation = reports('evaluation', [self.policy])
        bad = evaluation[0]['events'][0]
        bad.update(candidate_relative_max=.2, full_relative_max=.2, bad_candidate_rows=1,
                   oracle_bad_rows=1, oracle_bad_reused_rows=1)
        evaluation[0] = reseal(evaluation[0])
        frozen = self.fit()
        result = pp.validate_shadow(frozen, evaluation, cases('evaluation'))
        self.assertFalse(result['kv_validation_pass'])
        self.assertEqual(frozen['layers']['3']['selected']['policy'], asdict(self.policy))
        self.assertEqual(result['layers']['3']['summary']['false_accept_blocks'], 1)

    def test_no_eligible_calibration_stays_native_not_false_pass(self):
        event = self.cal[0]['events'][0]
        event.update(candidate_absolute_max=1., full_absolute_max=1., bad_candidate_rows=1,
                     oracle_bad_rows=1, oracle_bad_reused_rows=1)
        self.cal[0] = reseal(self.cal[0])
        result = pp.validate_shadow(self.fit(), reports('evaluation', [self.policy]), cases('evaluation'))
        self.assertFalse(result['any_gated_layer'])
        self.assertFalse(result['kv_validation_pass'])

    def test_each_split_leak_is_rejected(self):
        frozen = self.fit()
        for field in ('input_sha256', 'source_document_ids', 'documents'):
            with self.subTest(field=field):
                expected = cases('evaluation')
                expected[0][field] = deepcopy(cases('calibration')[0][field])
                with self.assertRaises(ValidationError):
                    pp.validate_shadow(frozen, reports('evaluation', [self.policy]), expected)

    def test_evaluation_count_and_artifact_tamper_rejected(self):
        frozen = self.fit()
        with self.assertRaises(ValidationError):
            pp.validate_shadow(frozen, [], [])
        frozen['oracle']['relative_l2'] = 2
        with self.assertRaises(ValidationError):
            pp.validate_shadow(frozen, reports('evaluation', [self.policy]), cases('evaluation'))

    def test_expected_case_translates_heldout_and_keeps_source_exposure(self):
        expected = cases('evaluation')[0]
        row = manifest_row(expected)
        self.assertEqual(pp.expected_case(row), expected)
        row['documents'][0]['document_id'] = 'd'*64
        with self.assertRaises(ValidationError):
            pp.expected_case(row)

    def test_phase_bridge_then_fit_then_native_record_binding_cpu_only(self):
        identity, manifest, phases = phase_bundle(self.cal)
        wrapped = [pp.wrap_shadow(phase, row, manifest, identity)
                   for phase, row in zip(phases, manifest['warmup'])]
        frozen = pp.fit_shadow(wrapped, [pp.expected_case(row) for row in manifest['warmup']],
                               wrapped[0]['scope'], [self.policy])
        from kv_cache_engram.native_reuse import NativeCacheIdentity
        cache_identity = NativeCacheIdentity('a'*64, 'b'*64, 'c'*64, 'd'*64, 3)
        policies, records = pp.make_pilot_records(frozen, {3: cache_identity})
        self.assertEqual(policies, {3: self.policy})
        records[3].validate(cache_identity, self.policy)
        self.assertEqual(records[3].evidence_sha256, frozen['sha256'])
        self.assertEqual(wrapped[0]['source_manifest_sha256'], manifest['sha256'])
        self.assertEqual(wrapped[0]['source_phase_sha256'], phases[0]['sha256'])
        # Native identity scopes the same model/config/layer across document IDs.
        records[3].validate(replace(cache_identity, tokens_sha256='e'*64), self.policy)
        with self.assertRaises(ValidationError):
            pp.make_pilot_records(frozen, {3: replace(cache_identity, execution_sha256='f'*64)})

    def test_bridge_rejects_fixture_backend_bad_binding_incomplete_or_tamper(self):
        identity, manifest, phases = phase_bundle(self.cal)
        for fault in ('fixture', 'native', 'scope', 'phase', 'coverage', 'tokens', 'manifest', 'seal'):
            with self.subTest(fault=fault):
                phase = deepcopy(phases[0])
                if fault == 'fixture':
                    phase['hook']['backend'] = 'cpu-fixture'
                elif fault == 'native':
                    phase['hook']['native_execution_declared'] = False
                elif fault == 'scope':
                    phase['hook']['scope']['execution_sha256'] = 'd'*64
                elif fault == 'phase':
                    phase['hook']['phase'] = 'different'
                elif fault == 'coverage':
                    phase['hook']['coverage'][3] -= 1
                elif fault == 'tokens':
                    phase['input_ids_sha256'] = 'e'*64
                elif fault == 'manifest':
                    phase['manifest_sha256'] = 'f'*64
                else:
                    phase['name'] = 'tampered'
                if fault != 'seal':
                    phase = reseal(phase)
                with self.assertRaises(ValidationError):
                    pp.wrap_shadow(phase, manifest['warmup'][0], manifest, identity)

    def test_bridge_refuses_case_not_in_manifest(self):
        identity, manifest, phases = phase_bundle(self.cal)
        row = deepcopy(manifest['warmup'][0])
        row['sample_id'] = 'other'
        with self.assertRaises(ValidationError):
            pp.wrap_shadow(phases[0], row, manifest, identity)

    def test_all_native_fallback_returns_two_empty_maps(self):
        bad = self.cal[0]['events'][0]
        bad.update(candidate_relative_max=.2, full_relative_max=.2, bad_candidate_rows=1,
                   oracle_bad_rows=1, oracle_bad_reused_rows=1)
        self.cal[0] = reseal(self.cal[0])
        from kv_cache_engram.native_reuse import NativeCacheIdentity
        identity = NativeCacheIdentity('a'*64, 'b'*64, 'c'*64, 'd'*64, 3)
        self.assertEqual(pp.make_pilot_records(self.fit(), {3: identity}), ({}, {}))

    def test_unselected_evaluation_malformed_data_cannot_be_filtered_out(self):
        other = replace(self.policy, probe_count=128)
        frozen = self.fit(reports('calibration', [self.policy, other]), [self.policy, other])
        evaluation = reports('evaluation', [self.policy, other])
        checksum = digest(asdict(other))
        evaluation[0]['events'] = [event for event in evaluation[0]['events']
                                  if not (event['candidate_id'] == checksum and event['document_local_start'] == 0)]
        evaluation[0] = reseal(evaluation[0])
        with self.assertRaises(ValidationError):
            pp.validate_shadow(frozen, evaluation, cases('evaluation'))


def manifest_row(case):
    return dict(sample_id=case['case_id'], dataset=case['dataset'],
        split='heldout' if case['split'] == 'evaluation' else case['split'],
        input_ids_sha256=case['input_sha256'],
        documents=[dict(document_id=d['document_sha256'], token_ids_sha256=d['document_sha256'],
                        tokens=d['tokens']) for d in case['documents']],
        context_segments=[dict(kind='primary_full', context_sha256=s) for s in case['source_document_ids']])


def phase_bundle(shadows):
    identity = dict(checkpoint_index_sha256='a'*64, tokenizer_json_sha256='b'*64,
                    execution_configuration_sha256='c'*64)
    rows = [dict(manifest_row(case), index=i, input_tokens=262180)
            for i, case in enumerate(cases('calibration'))]
    manifest = pp.seal_record(dict(artifact_kind='kv_cache_engram_native_pilot',
        pilot_protocol_sha256=pp.pilot_protocol()['sha256'], warmup=rows, cohort=[]))
    phases = []
    for shadow, row in zip(shadows, rows):
        hook = {key: deepcopy(shadow[key]) for key in ('mode', 'complete', 'native_execution_declared',
                    'scope', 'oracle', 'input_sha256', 'request_id', 'phase', 'events')}
        hook.update(schema='kv-cache-engram-native-hook-v1',
                    backend='deepinfer-sm90-native-bfloat16-kv-only-v1', full_q_attention_moe_preserved=True,
                    coverage={layer: row['input_tokens'] for layer in pp.LAYERS}, input_tokens=row['input_tokens'])
        phases.append(pp.seal_record(dict(schema='kv-cache-engram-native-phase-v1', mode='shadow_full',
            manifest_sha256=manifest['sha256'], execution_configuration_sha256='c'*64,
            input_ids_sha256=row['input_ids_sha256'], dataset=row['dataset'], split=row['split'],
            index=row['index'], name=hook['phase'], hook=hook)))
    return identity, manifest, phases


class SourcePlanTests(unittest.TestCase):
    def setUp(self):
        self.sources = {dataset: [dict(_id=dataset+str(i), context=dataset+' context '+str(i),
                                      input=dataset+' question '+str(i), answers=['fixture'])
                                 for i in range(80)] for dataset in pp.DATASETS}
        self.previous = pp.seal_record(dict(warmup=[], cohort=[], evaluation_protocol=dict(old_exposure={})))

    def test_source_partition_excludes_exposure_and_all_reserved_primaries(self):
        exposed = pp._text(self.sources['hotpotqa'][0]['context'])
        previous = pp.seal_record(dict(warmup=[], cohort=[], evaluation_protocol=dict(
            old_exposure=dict(context_sha256=[exposed]))))
        plan = pp.build_source_plan(self.sources, [previous])
        self.assertFalse(plan['token_lengths_verified'])
        split_hashes = {'calibration': set(), 'evaluation': set()}
        primary_hashes = set()
        for dataset in pp.DATASETS:
            for split, entry in plan['datasets'][dataset]['splits'].items():
                split_hashes[split].update(row['context_sha256'] for row in entry['pool'])
                primary_hashes.update(row['context_sha256'] for row in entry['selected'])
                self.assertNotIn(exposed, [row['context_sha256'] for row in entry['pool']])
                self.assertEqual(len(entry['selected']), 1 if split == 'calibration' else 2)
        self.assertFalse(split_hashes['calibration'] & split_hashes['evaluation'])
        for dataset in pp.DATASETS:
            for entry in plan['datasets'][dataset]['splits'].values():
                self.assertFalse(primary_hashes & set(entry['donor_context_sha256']))

    def test_answers_do_not_influence_primary_or_donor_selection(self):
        before = pp.build_source_plan(self.sources, [self.previous])
        for values in self.sources.values():
            for row in values:
                row['answers'] = ['different gold must never select inputs']
        after = pp.build_source_plan(self.sources, [self.previous])
        for dataset in pp.DATASETS:
            for split in ('calibration', 'evaluation'):
                a, b = before['datasets'][dataset]['splits'][split], after['datasets'][dataset]['splits'][split]
                self.assertEqual([r['source_index'] for r in a['selected']], [r['source_index'] for r in b['selected']])
                self.assertEqual(a['donor_source_indices'], b['donor_source_indices'])
        self.assertNotEqual(before['sha256'], after['sha256'])  # Source bytes remain bound.

    def test_old_manifest_tamper_or_missing_exposure_fails(self):
        self.previous['cohort'].append(dict(question='old question'))
        with self.assertRaises(ValidationError):
            pp.build_source_plan(self.sources, [self.previous])
        bad = reseal(self.previous)
        with self.assertRaises(ValidationError):
            pp.build_source_plan(self.sources, [bad])

    def test_fixed_seed_and_insufficient_source_fail_without_padding(self):
        with self.assertRaises(ValidationError):
            pp.build_source_plan(self.sources, [self.previous], seed=123)
        with self.assertRaises(ValidationError):
            pp.build_source_plan({dataset: rows[:2] for dataset, rows in self.sources.items()}, [self.previous])


if __name__ == '__main__':
    unittest.main()
