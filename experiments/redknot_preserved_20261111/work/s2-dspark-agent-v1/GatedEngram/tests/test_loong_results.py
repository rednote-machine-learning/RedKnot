"""Synthetic length/source/metric contract checks; never GPU observations."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from gpu_pilot import read_s1_results as r


def inputs():
    rows=[]
    for index in range(2,22):
        band=r.LENGTH_BANDS[(index-2)%2];n=r.TARGET_INPUT_TOKENS[band]
        ids=list(range(index*200000,index*200000+n));cuts=[1+(n-2)*j//10 for j in range(11)]
        docs=[dict(input_start=lo,input_end=hi,token_ids_sha256=r.digest(ids[lo:hi]),
            context_char_start=j*100,context_char_end=(j+1)*100,source_document_ordinal=j//2,
            source_document_name='doc'+str(j//2),source_doc_char_start=(j%2)*100,source_doc_char_end=(j%2+1)*100)
            for j,(lo,hi) in enumerate(zip(cuts,cuts[1:]))]
        originals=[dict(source_document_ordinal=j,source_document_name='doc'+str(j),context_char_start=j*200,context_char_end=(j+1)*200) for j in range(5)]
        gold={'type':['A','B']}
        rows.append(dict(index=index,dataset='loong',length_band=band,length_band_ordinal=(index-2)//2+1,
            target_input_tokens=n,input_tokens=n,input_ids=ids,documents=docs,source_id='synthetic-'+str(index),
            references=[json.dumps(gold,sort_keys=True,separators=(',',':'))],raw_answer=gold,raw_answer_sha256=r.digest(gold),
            source_domain='paper',source_language='en',source_level=3,source_document_count=5,original_document_count=5,original_documents=originals))
    return rows

class LoongResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=inputs()
    def test_real_band_geometry_and_interleaved_order(self):
        self.assertTrue(r.audit_length_inputs(self.rows));self.assertEqual([r.length_identity(x) for x in self.rows],['128k','64k']*10)
    def test_exact_actual_band_limits(self):
        for original in self.rows[:2]:
            row=copy.deepcopy(original);lo,hi=r.BAND_LIMITS[row['length_band']]
            for n in (lo,hi):row['input_tokens']=n;self.assertEqual(r.length_identity(row),row['length_band'])
            for n in (lo-1,hi+1):
                row['input_tokens']=n
                with self.assertRaises(ValueError):r.length_identity(row)
    def test_raw_gold_source_count_or_length_mutation_rejected(self):
        for key,value in [('raw_answer',{'wrong':[]}),('references',['C']),('length_band','64k'),('target_input_tokens',65536),('source_id','synthetic-3'),('source_document_count',10)]:
            rows=list(self.rows);rows[0]=copy.deepcopy(rows[0]);rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.audit_length_inputs(rows)
    def test_actual_body_token_mutation_rejected(self):
        rows=list(self.rows);rows[0]=copy.deepcopy(rows[0]);rows[0]['input_ids'][10]+=1
        with self.assertRaisesRegex(ValueError,'actual input IDs'):r.audit_length_inputs(rows)
    def test_original_docs_and_internal_cache_blocks_not_conflated(self):
        for change in [lambda row:row['documents'][1].update(source_document_ordinal=1),lambda row:row['documents'][0].update(source_doc_char_end=201),lambda row:row['original_documents'][0].update(context_char_end=199)]:
            row=copy.deepcopy(self.rows[0]);change(row)
            with self.assertRaises(ValueError):r.audit_source_metadata(row)
    def test_existing_clocks_are_nested_not_additive(self):
        phase=dict(controller=dict(online_cache_prepare_s=.2),record=dict(online_prepare_to_submit_s=.5,ttft_s=2.,online_ttft_s=2.5))
        self.assertEqual(r.existing_timing_breakdown(phase)['engine_ttft_s'],2.)
        for target,key,value in [('controller','online_cache_prepare_s',.8),('record','online_ttft_s',2.7),('record','ttft_s',float('nan'))]:
            bad=copy.deepcopy(phase);bad[target][key]=value
            with self.assertRaises(ValueError):r.existing_timing_breakdown(bad)
    def test_twenty_complete_and_prefix_partial(self):
        rows=[dict(index=i,shared_bank_identity=dict(bank_generation=i)) for i in range(2,22)]
        self.assertEqual(r.completion_state(rows,[],complete=True)['planned_cases'],20)
        partial=r.completion_state(rows[:3],[],complete=False,allow_partial=True,partial_reason='bounded_window')
        self.assertEqual(partial['status'],'partial_complete');self.assertEqual(partial['missing_case_indices'],list(range(5,22)))
        with self.assertRaises(ValueError):r.completion_state(rows[1:3],[],complete=False,allow_partial=True,partial_reason='bounded_window')
    def test_actual_producer_twenty_rows_reader_and_runner_scope(self):
        from unittest.mock import patch
        from tests.test_loong_inputs import LoongInputs
        from gpu_pilot import loong_inputs as producer
        from gpu_pilot.run_s1_pilot import case_scope
        LoongInputs.setUpClass();rows=LoongInputs.artifacts['manifest.json']['cohort']
        with LoongInputs.pins(),patch.object(r,'BAND_LIMITS',producer.INPUT_TOKEN_RANGES),patch.object(r,'TARGET_INPUT_TOKENS',producer.TARGET_INPUT_TOKENS):
            self.assertTrue(r.audit_length_inputs(rows))
            for row in rows:
                self.assertEqual(case_scope(row),{k:row[k] for k in r.SCOPE_FIELDS})
                self.assertEqual(r.position_oracle(r.documents(row),row['input_tokens'])['selected_tokens'],1280+row['input_non_document_tokens'])
                text=row['raw_answer'] if type(row['raw_answer']) is str else json.dumps(row['raw_answer'],ensure_ascii=False)
                result=r.score_answer(text,row['raw_answer'])
                if result['structured_eligible']:self.assertEqual(result['structured_exact'],1)
    def test_runner_case_scope_and_six_original_core_bindings(self):
        from gpu_pilot.run_s1_pilot import case_scope
        for row in self.rows:
            self.assertEqual(case_scope(row),{k:row[k] for k in r.SCOPE_FIELDS})
        root=Path(__file__).resolve().parents[1]
        for name,expected in r.CORE_SHA256.items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),expected)
        identity=dict(core_source_sha256=r.CORE_SHA256,core_policy_sha256=r.CORE_POLICY_SHA256,native_snapshot_sources_sha256=r.NATIVE_SNAPSHOT_SHA256)
        self.assertEqual(r.verify_core_bindings(identity,identity),identity)
        with self.assertRaises(ValueError):r.verify_core_bindings(dict(identity,core_policy_sha256='0'*64),identity)

if __name__=='__main__':unittest.main()
