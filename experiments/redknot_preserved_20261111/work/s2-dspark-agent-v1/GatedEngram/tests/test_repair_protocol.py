"""Synthetic CPU-only contracts, not native execution or repair-effect evidence."""
from copy import deepcopy
import json
import unittest

from gpu_pilot import pilot_protocol as pp
from gpu_pilot import repair_protocol as rp
from kv_cache_engram.core import ValidationError, digest


def seal(value):
    return pp.seal_record({key: item for key, item in value.items() if key != 'sha256'})


def case(index=2):
    return dict(index=index, dataset=pp.DATASETS[index % 2], sample_id=f'exposed:{index}',
        case_id=f'exposed:{index}', input_sha256=digest(['input', index]), input_tokens=8*32768+53,
        question='Which item?', references=['Gold'], previously_exposed_exploratory_cohort=True,
        document_spans=[dict(document_sha256=digest(['doc', index, j]), start=27+j*32768,
            end=27+(j+1)*32768, tokens=32768) for j in range(8)])


def plan(row):
    records = []
    for start in range(0, row['input_tokens'], rp.CHUNK):
        end = min(start+rp.CHUNK, row['input_tokens'])
        spans = rp.candidate_spans(row, start, end)
        for layer in rp.LAYERS:
            selected = spans[0] if spans else None
            rows = selected['input_end']-selected['input_start'] if selected else 0
            records.append(dict(layer=layer, start=start, end=end, source_spans=deepcopy(spans),
                source_spans_sha256=digest(spans), source_rows=sum(s['input_end']-s['input_start'] for s in spans),
                source_block_scores=[dict(input_start=s['input_start'], input_end=s['input_end'], relative_l2_max=1.) for s in spans],
                selected_span=deepcopy(selected), oracle_reference_sha256=digest([layer, start]) if selected else None,
                oracle_reference_shape=[rows, 1, 512] if selected else None,
                oracle_reference_dtype='torch.bfloat16' if selected else None,
                oracle_reference_bytes=rows*512*2))
    return seal(dict(schema=rp.PLAN_SCHEMA, complete=True, input_sha256=row['input_sha256'],
        input_tokens=row['input_tokens'], documents_sha256=digest(rp.document_identity(row)),
        capture_request_id=f"repair:{row['index']}:recomputed", capture_phase=f"case{row['index']}.recomputed",
        layers=list(rp.LAYERS), coverage={str(layer): row['input_tokens'] for layer in rp.LAYERS}, records=records,
        norm_floor=1e-8,cpu_reference_bytes=sum(r['oracle_reference_bytes'] for r in records),
        max_cpu_reference_bytes=256*1024**2,
        native_execution=True, operator_backend=rp.BACKEND, formal_admission_unchanged=True))


def report(row, frozen, arm):
    records, counts = [], {str(layer): dict(source_rows=0, injected_rows=0, oracle_restored_rows=0,
        reused_rows=0, native_q_rows=0, native_kv_projection_rows=0) for layer in rp.LAYERS}
    for source in frozen['records']:
        span = source['selected_span']; rows = source['source_rows']
        injected = rows if arm != 'recomputed' else 0
        restored = span['input_end']-span['input_start'] if span and arm == 'swa_reuse_repair' else 0
        item = {key: deepcopy(source[key]) for key in ('layer', 'start', 'end', 'source_spans_sha256',
                                                     'source_rows', 'selected_span', 'oracle_reference_sha256')}
        item.update(injected_rows=injected, oracle_restored_rows=restored, reused_rows=injected-restored)
        records.append(item)
        count = counts[str(source['layer'])]
        for key in ('source_rows', 'injected_rows', 'oracle_restored_rows', 'reused_rows'):
            count[key] += item[key]
        count['native_q_rows'] += source['end']-source['start']
        count['native_kv_projection_rows'] += source['end']-source['start']
    return seal(dict(schema=rp.ARM_SCHEMA, mode=rp.MODES[arm], complete=True,
        request_id=f"repair:{row['index']}:{arm}", phase=f"case{row['index']}.{arm}",
        input_sha256=row['input_sha256'], input_tokens=row['input_tokens'], documents_sha256=frozen['documents_sha256'],
        layers=list(rp.LAYERS), coverage=frozen['coverage'], plan_sha256=frozen['sha256'], records=records, counts=counts,
        oracle_diagnostic=True, forced_unqualified_reuse=arm != 'recomputed', propagates_to_later_layers=arm != 'recomputed',
        no_ttft_claim=True, all_q_native=True, all_kv_projections_native=True, all_ffn_native=True,
        global_kv_index_native=True, head_reuse=False, native_chunk_tail_guard=128, document_head_tail_guard=128,
        protected_layers=list(rp.PROTECTED_LAYERS), native_execution=True, operator_backend=rp.BACKEND,
        formal_admission_unchanged=True))


class RepairProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = case()
        cls.frozen = plan(cls.row)

    def test_fixed_protocol_has_no_runtime_admission_or_performance_claim(self):
        value = rp.repair_protocol()
        self.assertEqual(value, json.loads(json.dumps(value)))
        self.assertEqual(value['input_indices'], [2, 3, 4, 5])
        self.assertEqual(value['intervention_layers'], [6,7,9,10,11,12,13,16,17,18])
        self.assertTrue(value['formal_admission_unchanged'])
        self.assertTrue(value['previously_exposed_exploratory_cohort'])
        self.assertFalse(value['performance_measured'])
        self.assertFalse(value['gold_or_generated_answers_used_for_selection'])

    def test_four_explicitly_exposed_cases_only(self):
        rows = [case(i) for i in range(2, 6)]
        rp.validate_cases(rows)
        for value in (rows[:3], [dict(rows[0], index=0)]+rows[1:],
                      [dict(rows[0], previously_exposed_exploratory_cohort=False)]+rows[1:]):
            with self.assertRaises(ValidationError): rp.validate_cases(value)

    def test_guard_geometry_preserves_both_document_and_native_chunk_tails(self):
        spans = rp.candidate_spans(self.row, 0, 8192)
        self.assertEqual(spans[0]['document_local_start'], 128)
        self.assertTrue(all(s['input_end'] <= 8192-128 for s in spans))
        self.assertTrue(all(27+s['block_end'] <= 8192 for s in spans))
        self.assertTrue(all(s['document_local_end'] <= 32768-128 for s in spans))
        chunks = rp.validate_plan(self.frozen, self.row)
        self.assertEqual(len(chunks), len(rp.LAYERS)*33)

    def test_complete_native_A_B_C_row_accounting(self):
        result = {}
        for arm in rp.ARMS:
            value = report(self.row, self.frozen, arm)
            result[arm] = rp.validate_intervention(value, self.frozen, self.row, arm=arm,
                request_id=value['request_id'], phase=value['phase'])
        self.assertEqual(result['recomputed']['reused_rows'], 0)
        self.assertEqual(result['swa_reuse']['injected_rows'], result['swa_reuse_repair']['injected_rows'])
        self.assertEqual(result['swa_reuse']['reused_rows'], result['swa_reuse_repair']['reused_rows']
                         + result['swa_reuse_repair']['oracle_restored_rows'])
        self.assertEqual(result['swa_reuse']['native_kv_projection_rows'], len(rp.LAYERS)*self.row['input_tokens'])

    def test_missing_or_duplicate_chunk_rejected(self):
        for records in (self.frozen['records'][:-1], self.frozen['records']+[self.frozen['records'][0]]):
            with self.assertRaises(ValidationError):
                rp.validate_plan(seal(dict(self.frozen, records=records)), self.row)

    def test_candidate_geometry_cannot_include_native_tail_or_template(self):
        for field, value in (('input_start', 0), ('input_end', 8191)):
            frozen = deepcopy(self.frozen)
            frozen['records'][0]['source_spans'][0][field] = value
            frozen['records'][0]['source_spans_sha256'] = digest(frozen['records'][0]['source_spans'])
            with self.assertRaisesRegex(ValidationError, 'source candidates'):
                rp.validate_plan(seal(frozen), self.row)

    def test_oracle_selection_is_A_error_argmax_not_a_later_preferred_block(self):
        frozen = deepcopy(self.frozen)
        frozen['records'][0]['source_block_scores'][1]['relative_l2_max'] = 2.
        with self.assertRaisesRegex(ValidationError, 'A-error rule'):
            rp.validate_plan(seal(frozen), self.row)

    def test_reference_payload_shape_dtype_and_plan_seal_are_binding(self):
        for key, value in (('oracle_reference_shape', [1,512]), ('oracle_reference_dtype', 'torch.float32'),
                           ('oracle_reference_bytes', 0), ('oracle_reference_sha256', 'bad')):
            frozen = deepcopy(self.frozen); frozen['records'][0][key] = value
            with self.assertRaises(ValidationError): rp.validate_plan(seal(frozen), self.row)

    def test_oracle_reference_aggregate_budget_and_norm_are_binding(self):
        for key,value in (('cpu_reference_bytes',0),('max_cpu_reference_bytes',512*1024**2),('norm_floor',1e-6)):
            with self.assertRaises(ValidationError):
                rp.validate_plan(seal(dict(self.frozen,**{key:value})),self.row)

    def test_B_C_cannot_change_candidates_restore_count_or_native_compute(self):
        for fault in ('source', 'count', 'projection', 'ffn', 'plan'):
            value = report(self.row, self.frozen, 'swa_reuse_repair')
            if fault == 'source': value['records'][0]['source_spans_sha256'] = '0'*64
            elif fault == 'count': value['records'][0]['oracle_restored_rows'] = 0
            elif fault == 'projection': value['counts'][str(rp.LAYERS[0])]['native_kv_projection_rows'] -= 1
            elif fault == 'ffn': value['all_ffn_native'] = False
            else: value['plan_sha256'] = '0'*64
            with self.assertRaises(ValidationError):
                rp.validate_intervention(seal(value), self.frozen, self.row, arm='swa_reuse_repair',
                                        request_id=value['request_id'], phase=value['phase'])

    def test_cpu_fixture_or_forged_real_reuse_safety_label_is_rejected(self):
        for field in ('native_execution', 'formal_admission_unchanged'):
            frozen = seal(dict(self.frozen, **{field:False}))
            with self.assertRaises(ValidationError): rp.validate_plan(frozen, self.row)


if __name__ == '__main__':
    unittest.main()
