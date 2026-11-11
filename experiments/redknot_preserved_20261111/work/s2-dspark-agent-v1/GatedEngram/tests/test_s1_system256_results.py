"""Three independent 256K QAs; no128K gates or results substituted."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from gpu_pilot import read_s1_system256_results as r
from gpu_pilot import s1_system256_protocol as p
from gpu_pilot import run_s1_system256_pilot as runner


def rows():
    values=[]
    for index,(n,source) in enumerate(zip((260899,254000,248000),(1039,1036,1035)),2):
        ids=list(range(n));cuts=[3+(n-20)*j//10 for j in range(11)]
        docs=[]
        for j,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
            docs.append(dict(input_start=lo,input_end=hi,token_ids_sha256=r.digest(ids[lo:hi]),
                context_char_start=j*100,context_char_end=(j+1)*100,source_document_ordinal=j//2,
                source_document_name='doc'+str(j//2),source_doc_char_start=(j%2)*100,source_doc_char_end=(j%2+1)*100))
        originals=[dict(source_document_ordinal=j,source_document_name='doc'+str(j),
            context_char_start=j*200,context_char_end=(j+1)*200) for j in range(5)]
        gold='$1,740' if source==1036 else {'kind':['A','B']}
        values.append(dict(index=index,dataset='loong',length_band='256k',length_band_ordinal=index-1,
            target_input_tokens=262144,input_tokens=n,input_ids=ids,input_ids_sha256=r.digest(ids),documents=docs,
            technical_max_new_tokens=270336-n,source_index=source,context_sha256=r.digest(['synthetic-context',source]),
            raw_answer=gold,raw_answer_sha256=r.digest(gold),references=[gold if type(gold) is str else json.dumps(gold,separators=(',',':'))],
            source_domain='financial',source_language='en',source_level=2,source_document_count=5,
            original_document_count=5,original_documents=originals))
    return values


class System256ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=rows()

    def test_three_longest_first_separate_population(self):
        self.assertEqual(r.CASE_INDICES,(2,3,4));self.assertEqual(r.LENGTH_BANDS,('256k',))
        self.assertTrue(r.audit_length_inputs(self.rows))
        self.assertEqual(r.SCHEMA,'s1-system256-independent-readout-v1')
        self.assertEqual(r.RUN_SCHEMA,runner.SCHEMA);self.assertEqual(r.PHASE_SCHEMA,runner.PHASE_SCHEMA)

    def test_actual_band_boundaries_and_engine_headroom(self):
        row=dict(self.rows[0])
        for n in (245760,266240):row['input_tokens']=n;self.assertEqual(r.length_identity(row),'256k')
        for n in (245759,266241,131072,270336):
            row['input_tokens']=n
            with self.subTest(n=n),self.assertRaises(ValueError):r.length_identity(row)
        self.assertEqual(270336-260899,9437)

    def test_qualify_actual_32_chunks_and_keep_10_cache_blocks(self):
        geo=r.case_geometry(self.rows[0])
        self.assertEqual(geo['native_prefill_chunks'],32);self.assertEqual(geo['layer_chunk_checks'],1280)
        self.assertEqual(geo['offline_bank_entry_count'],400)
        self.assertEqual(geo['offline_bank_bytes'],37*10*128*512*2+9216*geo['document_tokens'])
        self.assertEqual(len(r.documents(self.rows[0])),10)

    def test_source_order_population_and_context_reuse_rejected(self):
        for key,value in (('source_index',1036),('index',8),('length_band','128k'),
                          ('length_band_ordinal',2),('technical_max_new_tokens',100),
                          ('context_sha256',self.rows[1]['context_sha256'])):
            data=list(self.rows);data[0]=dict(data[0],**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):r.audit_length_inputs(data)
        for data in (self.rows[:2],self.rows+[self.rows[0]],list(reversed(self.rows))):
            with self.assertRaises(ValueError):r.audit_length_inputs(data)

    def test_raw_gold_units_not_rewritten_by_reader(self):
        row=self.rows[1];self.assertTrue(r.audit_source_metadata(row));self.assertEqual(row['raw_answer'],'$1,740')
        for key,value in (('raw_answer','$1,740,000'),('references',['$1,740,000'])):
            with self.assertRaises(ValueError):r.audit_source_metadata(dict(row,**{key:value}))

    def test_actual_input_and_document_ids_verified_independently(self):
        bad=list(self.rows);bad[0]=dict(bad[0],input_ids=list(bad[0]['input_ids']))
        bad[0]['input_ids'][10]+=1
        with self.assertRaisesRegex(ValueError,'actual full input IDs'):r.audit_length_inputs(bad)
        bad[0]['input_ids_sha256']=r.digest(bad[0]['input_ids'])
        with self.assertRaisesRegex(ValueError,'document token slice'):r.audit_length_inputs(bad)

    def test_no_cross_original_document_boundary_or_omitted_source(self):
        row=copy.deepcopy(self.rows[0]);row['documents'][1]['source_document_ordinal']=1
        with self.assertRaises(ValueError):r.audit_source_metadata(row)
        row=copy.deepcopy(self.rows[0]);row['original_documents'][0]['context_char_end']-=1
        with self.assertRaises(ValueError):r.audit_source_metadata(row)

    def test_completion_counts_are_three_not_ten(self):
        values=[dict(index=i,shared_bank_identity=dict(bank_generation=i)) for i in (2,3,4)]
        complete=r.completion_state(values,[],complete=True)
        self.assertEqual(complete['planned_cases'],3);self.assertEqual(complete['status'],'complete')
        partial=r.completion_state(values[:1],[],complete=False,allow_partial=True,partial_reason='runner_failure')
        self.assertEqual(partial['missing_case_indices'],[3,4]);self.assertFalse(partial['overall_run_complete'])
        with self.assertRaises(ValueError):r.completion_state(values[1:],[],complete=False,
            allow_partial=True,partial_reason='runner_failure')

    def test_exact_three_arm_nine_repeat_schedule(self):
        for index in (2,3,4):
            schedule=r.timing_schedule(index);self.assertEqual(schedule,runner.timing_schedule(index))
            self.assertEqual(len(schedule),9)
            for repeat in range(3):self.assertEqual({a for k,a in schedule if k==repeat},set(r.ARMS))
        with self.assertRaises(ValueError):r.timing_schedule(6)

    def test_existing_clock_is_nested_cache_region_not_sum_three(self):
        phase=dict(controller=dict(online_cache_prepare_s=.2),record=dict(online_prepare_to_submit_s=.4,ttft_s=15,online_ttft_s=15.4))
        self.assertEqual(r.existing_timing_breakdown(phase)['engine_ttft_s'],15)
        phase['record']['online_ttft_s']=15.6
        with self.assertRaises(ValueError):r.existing_timing_breakdown(phase)

    def test_original_core_and_system_implementation_bytes_preserved(self):
        stage=Path(__file__).resolve().parents[1]
        for path,sha in r.CORE_SHA256.items():self.assertEqual(hashlib.sha256((stage/path).read_bytes()).hexdigest(),sha)
        contract=p.s1_protocol();self.assertEqual(contract['cases'],[2,3,4])
        self.assertIsNone(contract['quality']['max_tokens']);self.assertEqual(contract['quality']['engine_max_model_len'],270336)
        self.assertEqual(contract['system_optimization']['batch_group_chunks'],4)

    def test_actual_input_builder_metadata_reader_and_runner_link(self):
        from tests.test_s1_system256_inputs import System256Inputs
        from gpu_pilot import s1_system256_inputs as producer
        System256Inputs.setUpClass();data=System256Inputs.artifacts['manifest.json']['cohort']
        with System256Inputs.pins(),patch.object(r,'BAND_LIMITS',{'256k':(8000,11000)}),\
             patch.object(r,'TARGET_INPUT_TOKENS',{'256k':9000}):
            self.assertTrue(r.audit_length_inputs(data))
            for row in data:
                self.assertEqual(runner.case_scope(row),{k:row[k] for k in r.SCOPE_FIELDS})
                self.assertEqual(r.case_geometry(row)['document_tokens'],row['document_tokens_total'])

if __name__=='__main__':unittest.main()
