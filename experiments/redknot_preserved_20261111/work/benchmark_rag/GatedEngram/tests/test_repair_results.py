"""Synthetic CPU raw-file triple audit; no model, Torch, GPU or network calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from gpu_pilot import pilot_protocol as pp
from gpu_pilot import repair_protocol as rp
from gpu_pilot import read_repair_results as reader
from kv_cache_engram.core import ValidationError, digest
from test_repair_protocol import case, plan, report, seal


def flags():
    return dict(formal_admission_unchanged=True, forced_controlled_diagnostic=True,
                no_ttft_claim=True, previously_exposed_exploratory_cohort=True)


def write(root, name, value):
    (root/name).write_text(json.dumps(value, ensure_ascii=False))


class Fixture:
    def __init__(self, path):
        self.path = path
        self.rows = [case(i) for i in range(2, 6)]
        self.identity = dict(checkpoint_index_sha256='a'*64, tokenizer_json_sha256='b'*64,
            execution_configuration_sha256='c'*64, repair_protocol_sha256=rp.repair_protocol()['sha256'])
        self.qualification_row = dict(case(0), split='calibration')
        self.preflight = seal(dict(identity=self.identity, manifest_sha256='d'*64,
            protocol=rp.repair_protocol(), audit_cases=self.rows, qualification_case=self.qualification_row, **flags()))
        write(path, 'preflight.json', self.preflight)
        self.qualification()
        self.triples = {}

    def qualification(self):
        row = self.qualification_row
        name = 'qualification_split_identity'
        rid = f"kce-repair:0:{name}:{row['sample_id']}"
        errors = {key:0. for key in ('q','kv_raw','kv_canonical','gather32','gather128','post_rope_cast',
            'post_rope_gather32','post_rope_gather128','post_rope_gather384','post_rope_gather512')}
        hook = dict(schema='kv-cache-engram-native-hook-v1', mode='split_identity', complete=True,
            native_execution_declared=True, backend=rp.BACKEND, full_q_attention_moe_preserved=True,
            request_id=rid, phase=name, input_sha256=row['input_sha256'], input_tokens=row['input_tokens'],
            scope=dict(model_sha256='a'*64, tokenizer_sha256='b'*64, execution_sha256='c'*64,
                representation='swa-postnorm-prerope-v1', storage_dtype='bfloat16-native-canonical',
                width=512, layers=list(pp.LAYERS)), events=[],
            coverage={str(layer):row['input_tokens'] for layer in pp.LAYERS},
            counts={str(layer):dict(q_rows=row['input_tokens'], native_fused_rows=row['input_tokens'],
                online_kv_rows=row['input_tokens'], split_q_rows=0,reused_kv_rows=0,probe_rows=0,fallback_rows=0)
                for layer in pp.LAYERS}, identity_exact=True, numerical_equivalence_claim=True,
            qualification=[dict(layer=layer,start=start,end=min(start+rp.CHUNK,row['input_tokens']),exact=True,
                errors=deepcopy(errors)) for start in range(0,row['input_tokens'],rp.CHUNK) for layer in pp.LAYERS])
        rec=dict(success=True,rank=0,request_id=rid,phase=name,sample_id=row['sample_id'],dataset=row['dataset'],
            input_ids_sha256=row['input_sha256'],input_tokens=row['input_tokens'],output_ids=[1],
            output_ids_sha256=digest([1]),natural_EOS=False,truncated=False)
        raw=seal(dict(schema='kv-cache-engram-native-phase-v1',name=name,mode='split_identity',arm=None,
            intervention=None,plan_sha256=None,split='calibration',index=0,dataset=row['dataset'],
            sample_id=row['sample_id'],request_id=rid,input_ids_sha256=row['input_sha256'],input_tokens=row['input_tokens'],
            manifest_sha256='d'*64,protocol_sha256=self.preflight['protocol']['sha256'],
            execution_configuration_sha256='c'*64,timing_valid=False,quality_eligible=False,record=rec,hook=hook,**flags()))
        self.qualified=raw
        write(self.path,name+'.json',raw)
        write(self.path,'qualification.json',seal(dict(passed=True,quality_claim=False,phase_sha256=raw['sha256'])))

    def create(self, index, texts=('Gold', 'Wrong', 'Gold'), length=False):
        row = self.rows[index-2]; frozen = plan(row)
        write(self.path, f'case{index}.plan.json', frozen)
        seals = {}
        for arm, text in zip(rp.ARMS, texts):
            intervention = report(row, frozen, arm)
            name, rid = intervention['phase'], intervention['request_id']
            ids = list(range(64 if length else 2))
            rec = dict(success=True, rank=0, phase=name, request_id=rid,
                sample_id=row['sample_id'], dataset=row['dataset'], input_ids_sha256=row['input_sha256'],
                input_tokens=row['input_tokens'], references=row['references'], text=text,
                output_ids=ids, output_ids_sha256=digest(ids), natural_EOS=True,
                finish_reason='length' if length else 'stop', truncated=length,
                # Native metadata may contain timings. Reader must never display them as an optimization result.
                ttft_s=.1, online_ttft_s=.2, **reader._scores(text, row['references']))
            hook = dict(schema='kv-cache-engram-native-hook-v1', mode='native', complete=True,
                native_execution_declared=True, backend=rp.BACKEND, full_q_attention_moe_preserved=True,
                request_id=rid, phase=name, input_sha256=row['input_sha256'], input_tokens=row['input_tokens'],
                scope=dict(model_sha256='a'*64, tokenizer_sha256='b'*64, execution_sha256='c'*64,
                    representation='swa-postnorm-prerope-v1', storage_dtype='bfloat16-native-canonical',
                    width=512, layers=list(pp.LAYERS)), events=[],
                coverage={str(layer):row['input_tokens'] for layer in pp.LAYERS},
                counts={str(layer):dict(q_rows=row['input_tokens'], native_fused_rows=row['input_tokens'],
                    online_kv_rows=row['input_tokens'], split_q_rows=0,reused_kv_rows=0,probe_rows=0,fallback_rows=0)
                    for layer in pp.LAYERS})
            raw = seal(dict(schema=rp.PHASE_SCHEMA, index=index, arm=arm, name=name,
                dataset=row['dataset'], split='heldout', manifest_sha256='d'*64,
                execution_configuration_sha256='c'*64, protocol_sha256=self.preflight['protocol']['sha256'],
                input_ids_sha256=row['input_sha256'], plan_sha256=frozen['sha256'],
                record=rec, hook=hook, intervention=intervention, timing_valid=False,
                quality_eligible=True, phase_wall_s=15., **flags()))
            write(self.path, name+'.json', raw); seals[arm] = raw['sha256']
        triple = seal(dict(schema=rp.TRIPLE_SCHEMA, index=index, dataset=row['dataset'],sample_id=row['sample_id'],
            input_ids_sha256=row['input_sha256'],input_tokens=row['input_tokens'], manifest_sha256='d'*64,
            execution_configuration_sha256='c'*64,protocol_sha256=self.preflight['protocol']['sha256'],
            plan_sha256=frozen['sha256'],phase_sha256=seals,all_negative_results_retained=True,**flags()))
        write(self.path, f'case{index}.triple.json',triple); self.triples[index] = triple
        return triple

    def mutate(self, index, arm, callback):
        path = self.path/f'case{index}.{arm}.json'
        raw = json.loads(path.read_text()); callback(raw); raw = seal(raw)
        write(self.path, path.name, raw)
        triple = deepcopy(self.triples[index]); triple['phase_sha256'][arm] = raw['sha256']
        triple = seal(triple); self.triples[index] = triple
        write(self.path, f'case{index}.triple.json', triple)


class RepairResultsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name); self.fixture = Fixture(self.path)

    def test_first_complete_exploratory_triple_has_accuracy_comparison_not_ttft(self):
        self.fixture.create(2)
        result = reader.read_results(self.path)
        self.assertEqual(result['completed'],1)
        self.assertEqual(result['by_dataset']['hotpotqa']['completed'],1)
        self.assertEqual(result['by_dataset']['2wikimqa']['completed'],0)
        comparison=result['global_summary']['comparisons']['answer_em']
        self.assertEqual(comparison,dict(B_loss_vs_A_percentage_points=100.,C_gain_vs_B_percentage_points=100.,
            C_remaining_loss_vs_A_percentage_points=0.,gap_recovery_fraction=1.,gap_recovery_applicable=True))
        self.assertTrue(result['previously_exposed_exploratory_cohort'])
        self.assertFalse(result['independent_holdout_claimed'])
        self.assertNotIn('online_ttft_s', json.dumps(result))
        self.assertNotIn('ttft_speedup', json.dumps(result))
        self.assertEqual(result['global_summary']['transitions']['B_to_C']['wrong_to_correct'],1)
        self.assertEqual(result['numerical_qualification']['entries'],462)
        self.assertEqual(result['numerical_qualification']['numerical_checks'],4620)

    def test_qualification_cannot_be_replaced_by_passed_flag(self):
        self.fixture.create(2)
        (self.path/'qualification_split_identity.json').unlink()
        with self.assertRaisesRegex(ValidationError,'missing'): reader.read_results(self.path)
        (self.path/'qualification.json').unlink()
        with self.assertRaisesRegex(ValidationError,'qualification required'): reader.read_results(self.path)

    def test_every_qualification_chunk_and_all_ten_checks_are_required(self):
        self.fixture.create(2)
        for fault in ('nonzero','missingmetric','chunk','duplicate','wronginput','wrongrequest','claim'):
            raw=deepcopy(self.fixture.qualified)
            if fault=='nonzero': raw['hook']['qualification'][0]['errors']['post_rope_gather512']=.001
            elif fault=='missingmetric': raw['hook']['qualification'][0]['errors'].pop('post_rope_gather384')
            elif fault=='chunk': raw['hook']['qualification'].pop()
            elif fault=='duplicate': raw['hook']['qualification'].append(raw['hook']['qualification'][0])
            elif fault=='wronginput': raw['input_ids_sha256']='0'*64
            elif fault=='wrongrequest': raw['record']['request_id']='other'
            else: raw['hook']['identity_exact']=False
            raw=seal(raw)
            write(self.path,'qualification_split_identity.json',raw)
            write(self.path,'qualification.json',seal(dict(passed=True,quality_claim=False,phase_sha256=raw['sha256'])))
            with self.subTest(fault=fault),self.assertRaises(ValidationError): reader.read_results(self.path)

    def test_length_zero_scores_and_negative_repair_are_retained(self):
        self.fixture.create(2,('Gold','Gold','Wrong'),length=True)
        result=reader.read_results(self.path)
        self.assertEqual(result['completed'],1)
        comparison=result['global_summary']['comparisons']['answer_em']
        self.assertEqual(comparison['C_gain_vs_B_percentage_points'],-100.)
        self.assertIsNone(comparison['gap_recovery_fraction'])
        self.assertEqual(result['global_summary']['length_counts'],dict.fromkeys(rp.ARMS,1))

    def test_recovery_can_be_negative_or_over_one_without_clipping(self):
        for values, expected in (((.6,.4,.9),2.5),((.6,.4,.1),-1.5)):
            quality={arm:dict(answer_em=value,answer_token_f1=value) for arm,value in zip(rp.ARMS,values)}
            self.assertAlmostEqual(reader.comparisons(quality)['answer_token_f1']['gap_recovery_fraction'],expected)
        quality={arm:dict(answer_em=.5,answer_token_f1=.5) for arm in rp.ARMS}
        self.assertIsNone(reader.comparisons(quality)['answer_em']['gap_recovery_fraction'])

    def test_partial_or_missing_raw_never_enters_results(self):
        self.fixture.create(2)
        (self.path/'case2.swa_reuse_repair.json').unlink()
        result=reader.read_results(self.path)
        self.assertEqual(result['completed'],0)
        self.assertEqual(result['status'],'evidence_invalid')
        self.assertIsNone(result['global_summary'])

    def test_quality_gold_is_scored_independently_of_producer(self):
        self.fixture.create(2)
        self.fixture.mutate(2,'swa_reuse',lambda raw:raw['record'].update(answer_em=1.))
        result=reader.read_results(self.path)
        self.assertEqual(result['completed'],0)
        self.assertIn('fixed gold',result['audit_errors'][0]['error'])

    def test_gate_projection_or_speed_claim_cannot_sneak_into_diagnostic(self):
        for fault in ('gate','projection','timing'):
            self.fixture.create(2)
            def change(raw):
                if fault=='gate': raw['hook']['events']=[{}]
                elif fault=='projection': raw['hook']['counts']['6']['online_kv_rows']-=1
                else: raw['timing_valid']=True
            self.fixture.mutate(2,'swa_reuse',change)
            self.assertEqual(reader.read_results(self.path)['completed'],0)

    def test_B_C_must_share_frozen_candidates_and_selected_reference(self):
        self.fixture.create(2)
        def change(raw):
            raw['intervention']['records'][0]['oracle_reference_sha256']='0'*64
            raw['intervention']=seal(raw['intervention'])
        self.fixture.mutate(2,'swa_reuse_repair',change)
        result=reader.read_results(self.path)
        self.assertEqual(result['completed'],0)
        self.assertIn('frozen A plan',result['audit_errors'][0]['error'])

    def test_stale_seal_and_foreign_plan_fail_closed(self):
        self.fixture.create(2)
        value=json.loads((self.path/'case2.plan.json').read_text())
        value['records'][0]['source_rows']+=1
        write(self.path,'case2.plan.json',value)
        self.assertEqual(reader.read_results(self.path)['completed'],0)

    def test_four_complete_triples_terminal_does_not_include_old_calibration(self):
        for index in range(2,6):self.fixture.create(index)
        run=seal(dict(schema=rp.RUN_SCHEMA,status='complete',identity=self.fixture.identity,
            manifest_sha256='d'*64,protocol_sha256=rp.repair_protocol()['sha256'],
            cases=list(self.fixture.triples.values()),**flags()))
        write(self.path,'run.json',run)
        result=reader.read_results(self.path)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['completed'],4)
        self.assertEqual([result['by_dataset'][d]['completed'] for d in pp.DATASETS],[2,2])
        self.assertEqual(result['calibration_cases_in_quality'],0)

    def test_complete_claim_rejects_missing_or_invalid_triple(self):
        self.fixture.create(2)
        run=seal(dict(schema=rp.RUN_SCHEMA,status='complete',identity=self.fixture.identity,
            manifest_sha256='d'*64,protocol_sha256=rp.repair_protocol()['sha256'],
            cases=list(self.fixture.triples.values()),**flags()))
        write(self.path,'run.json',run)
        with self.assertRaisesRegex(ValidationError,'four matched'):reader.read_results(self.path)

    def test_exposure_label_cannot_be_omitted_to_claim_new_holdout(self):
        value=deepcopy(self.fixture.preflight);value['previously_exposed_exploratory_cohort']=False
        write(self.path,'preflight.json',seal(value))
        with self.assertRaisesRegex(ValidationError,'exposure safeguard'):reader.read_results(self.path)

    def test_failed_run_keeps_completed_triples_and_output_is_exclusive(self):
        self.fixture.create(2)
        write(self.path,'failure.rank0.json',seal(dict(schema=rp.RUN_SCHEMA,status='failed',rank=0,error='fixture')))
        self.assertEqual(reader.read_results(self.path)['status'],'failed')
        target=self.path/'audit.json'
        self.assertEqual(reader.main([str(self.path),'--output',str(target)]),0)
        with self.assertRaises(FileExistsError):reader.main([str(self.path),'--output',str(target)])


if __name__=='__main__':
    unittest.main()
