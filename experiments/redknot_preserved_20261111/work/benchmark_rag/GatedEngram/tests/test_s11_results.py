"""Independent mixed-length two-arm audit contracts; no GPU numerical claim."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
from gpu_pilot import read_s11_results as r
from gpu_pilot import s11_protocol as p
from gpu_pilot import run_s11_pilot as runner
from tests.test_s1_system256_results import rows as original_rows


def rows():
    values=original_rows();row=copy.deepcopy(values[-1]);n=130000;ids=list(range(n))
    row.update(index=20,length_band='128k',length_band_ordinal=10,target_input_tokens=131072,
        input_tokens=n,input_ids=ids,input_ids_sha256=r.digest(ids),source_index=111,
        context_sha256=r.digest('another128context'))
    row.pop('technical_max_new_tokens')
    for j,doc in enumerate(row['documents']):
        lo=3+(n-20)*j//10;hi=3+(n-20)*(j+1)//10
        doc.update(input_start=lo,input_end=hi,token_ids_sha256=r.digest(ids[lo:hi]))
    return values+[row]


def reference_fixture():
    from tests.test_s11_qualification import S11QualificationTests
    from kv_cache_engram.s1_runtime import S1Controller
    c,args=S11QualificationTests().make();c.reset('r','reference_sparse',**args)
    with patch.object(S1Controller,'finish',return_value={'sparse_identity_exact':False}):control=c.finish()
    phase=dict(mode='reference_sparse',arm='s1_system',system_enabled=True,batch_group_chunks=1,
        verify_sparse=False,timing_valid=False,quality_eligible=False)
    return phase,control


class S11ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=rows()

    def test_population_preserves_storage_and_distinct_execution_order(self):
        self.assertTrue(r.audit_length_inputs(self.rows))
        self.assertEqual(r.CASE_INDICES,(20,2,3,4));self.assertEqual(r.ARMS,('recomputed','s1_system'))
        self.assertEqual(r.RUN_SCHEMA,runner.SCHEMA);self.assertEqual(r.PHASE_SCHEMA,runner.PHASE_SCHEMA)
        self.assertEqual(r.SCHEMA,'s11-independent-readout-v1')
        with self.assertRaises(ValueError):r.audit_length_inputs([self.rows[3],*self.rows[:3]])

    def test_per_band_original_identity_and_longest_qualification(self):
        self.assertEqual(r.length_identity(self.rows[-1]),'128k')
        self.assertEqual(r.case_geometry(self.rows[0])['layer_chunk_checks'],1280)
        for index,key,value in ((3,'length_band_ordinal',1),(0,'source_index',1040),(0,'length_band','128k')):
            bad=copy.deepcopy(self.rows);bad[index][key]=value
            with self.assertRaises(ValueError):r.audit_length_inputs(bad)
        bad=copy.deepcopy(self.rows);bad[-1]['input_ids'][5]+=1
        with self.assertRaisesRegex(ValueError,'full input IDs'):r.audit_length_inputs(bad)

    def test_natural_eos_capacity_not_old64_token_limit(self):
        self.assertIsNone(p.s11_protocol()['quality']['max_tokens'])
        self.assertEqual(p.s11_protocol()['quality']['engine_max_model_len']-self.rows[0]['input_tokens'],9437)
        for row in self.rows:self.assertGreaterEqual(270336-row['input_tokens'],4096)

    def test_actual_adapter_dense_reference_is_unqualified(self):
        phase,control=reference_fixture();r.audit_qualification_adapter(control,phase)
        self.assertTrue(control['system_enabled']);self.assertFalse(control['system_numerically_qualified'])
        self.assertFalse(control['sparse_identity_exact'])
        for key,value in [('system_numerically_qualified',True),('sparse_identity_exact',True),
                          ('system_enabled',False),('s11_original_s1_benchmark_executed',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):r.audit_qualification_adapter(dict(control,**{key:value}),phase)

    def test_reference_cannot_use_retired_arm_batch_or_self_verification(self):
        phase,control=reference_fixture()
        for key,value in [('arm','s1'),('batch_group_chunks',4),('verify_sparse',True),('timing_valid',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):r.audit_qualification_adapter(control,dict(phase,**{key:value}))
        with self.assertRaises(ValueError):r.audit_qualification_adapter(control,dict(phase,mode='sensitive',arm='s1'))

    def test_full_and_system_only_six_timings(self):
        for index in r.CASE_INDICES:
            self.assertEqual(r.timing_schedule(index),runner.timing_schedule(index))
            self.assertEqual(len(r.timing_schedule(index)),6)
        sample={'recomputed':[12.,15.,18.],'s1_system':[4.,5.,6.]}
        summary=r.expected_timing_summary(sample);self.assertEqual(summary['comparisons']['s1_system']['ttft_speedup'],3.)
        self.assertNotIn('s1',summary['arms'])
        for bad in [dict(sample,s1=[7.,8.,9.]),dict(sample,s1_system=[4.,5.])]:
            with self.assertRaises(ValueError):r.expected_timing_summary(bad)
        with self.assertRaises(ValueError):p.phase_execution('s1')

    def test_partial_prefix_follows_execution_not_numeric_order(self):
        cases=[dict(index=index,shared_bank_identity={'bank_generation':ordinal+10}) for ordinal,index in enumerate(r.CASE_INDICES)]
        result=r.completion_state(cases,[],complete=True)
        self.assertEqual(result['planned_cases'],4);self.assertEqual(result['completed_case_indices'],[20,2,3,4])
        partial=r.completion_state(cases[:2],[],complete=False,allow_partial=True,partial_reason='runner_failure')
        self.assertEqual(partial['missing_case_indices'],[3,4]);self.assertFalse(partial['overall_run_complete'])
        for bad in [cases[1:],list(reversed(cases)),[cases[0],cases[2]]]:
            with self.assertRaises(ValueError):r.completion_state(bad,[],complete=False,allow_partial=True,partial_reason='runner_failure')

    def test_qualification_anchor_capture_is_separate_and_all_ten_are_bound(self):
        row=self.rows[0];qual={'sha256':'a'*64};phases={}
        for ordinal in range(10):
            phase={'name':f'qualification.anchor.offline_doc{ordinal}','sha256':r.digest(ordinal)}
            phases[phase['name']+'.json']=phase
        closure=dict(schema='s11-qualification-anchor-offline-build-v1',index=2,input_ids_sha256=row['input_ids_sha256'],
            input_tokens=row['input_tokens'],qualification_sha256=qual['sha256'],offline_build_s=1.,timing_valid=False,
            quality_eligible=False,formal_bank_rebuilt_separately=True,phase_refs=[{'file':n} for n in phases],
            offline_phase_sha256=[v['sha256'] for v in phases.values()])
        phases['qualification_anchor_offline_build.json']=closure;seen=[]
        def offline(phase,*args,**kw):
            self.assertEqual(kw['prefix'],'qualification.anchor');seen.append(phase['name']);return {'bank_generation':9}
        def check(ref,path,phase,evidence):self.assertEqual(ref['file'],path.name)
        with patch.object(r,'read',side_effect=lambda path,evidence:phases[path.name]),\
             patch.object(r,'audit_offline_phase',side_effect=offline),patch.object(r,'verify_reference',side_effect=check):
            self.assertEqual(r.qualification_anchor_build(Path('/fixture'),{},qual,row,{})[1],9)
            self.assertEqual(len(seen),10)
            closure['formal_bank_rebuilt_separately']=False
            with self.assertRaises(ValueError):r.qualification_anchor_build(Path('/fixture'),{},qual,row,{})

    def test_existing_clock_cache_host_region_is_nested(self):
        phase=dict(controller=dict(online_cache_prepare_s=.2),record=dict(online_prepare_to_submit_s=.4,ttft_s=15,online_ttft_s=15.4))
        self.assertEqual(r.existing_timing_breakdown(phase)['engine_ttft_s'],15)
        phase['record']['online_ttft_s']=15.6
        with self.assertRaises(ValueError):r.existing_timing_breakdown(phase)

    def test_immutable_core_bytes_and_original_53_policy(self):
        stage=Path(__file__).resolve().parents[1]
        for path,sha in r.CORE_SHA256.items():self.assertEqual(hashlib.sha256((stage/path).read_bytes()).hexdigest(),sha)
        contract=p.s11_protocol();self.assertEqual(contract['core_policy_sha256'],r.CORE_POLICY_SHA256)
        self.assertEqual(r.digest(contract['baseline_core_policy']),r.CORE_POLICY_SHA256)

    def test_actual_input_producer_reader_runner_link(self):
        from tests.test_s11_inputs import S11Inputs
        S11Inputs.setUpClass();data=S11Inputs.artifacts['manifest.json']['cohort']
        # Producer fixtures have natural short CPU-only geometry, never claim actual bands.
        with S11Inputs.pins(),patch.object(r,'BAND_LIMITS',{'128k':(100,20000),'256k':(8000,11000)}),\
             patch.object(r,'TARGET_INPUT_TOKENS',{row['length_band']:row['target_input_tokens'] for row in data}),\
             patch.object(r,'SOURCE_256_INDICES',tuple(row['source_index'] for row in data[:3])):
            self.assertTrue(r.audit_length_inputs(data))
            for row in data:self.assertEqual(runner.case_scope(row),{k:row[k] for k in r.SCOPE_FIELDS})
            self.assertEqual(S11Inputs.artifacts['input-protocol.json']['expected_predecessor_sha256'],S11Inputs.predecessor['sha256'])

if __name__=='__main__':unittest.main()
