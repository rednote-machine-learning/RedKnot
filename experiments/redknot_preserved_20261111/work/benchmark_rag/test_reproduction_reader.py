"""CPU-only checks for private 64K/128K reproduction audit admission."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import read_s12_reproduction_results as reader
from reproduction_contract import reproduction_contract,BAND_LIMITS
from profile_group_contract import group_contract

class ReproductionReaderTests(unittest.TestCase):
    def test_private_namespace_does_not_mutate_frozen_modules_or_group17(self):
        original=reader.inherited;grouped=group_contract(17)
        before=(dict(original.BAND_LIMITS),original.CASE_INDICES,dict(grouped.audit_controller.__globals__['BAND_LIMITS']))
        contract=reproduction_contract();scope=contract.audit_phase.__globals__
        self.assertIsNot(scope,grouped.audit_phase.__globals__)
        self.assertIs(scope['audit_controller'],contract.audit_controller)
        self.assertIs(scope['length_identity'],contract.length_identity)
        self.assertEqual(scope['BAND_LIMITS'],BAND_LIMITS)
        self.assertIs(contract.audit_controller.__code__,grouped.audit_controller.__code__)
        self.assertIs(contract.comparison_scope.__code__,grouped.comparison_scope.__code__)
        self.assertEqual(before,(dict(original.BAND_LIMITS),original.CASE_INDICES,dict(grouped.audit_controller.__globals__['BAND_LIMITS'])))

    def test_actual_band_index_target_and_ordinal_are_checked(self):
        contract=reproduction_contract()
        for band,index,n,target,ordinal in [('64k',3,67394,65536,1),('64k',21,63482,65536,10),('128k',2,132360,131072,1),('128k',20,120389,131072,10)]:
            row=dict(length_band=band,index=index,input_tokens=n,target_input_tokens=target,length_band_ordinal=ordinal,dataset='loong')
            self.assertEqual(contract.length_identity(row),band)
            for change in ({'target_input_tokens':262144},{'length_band_ordinal':0},{'index':0},{'input_tokens':50000}):
                with self.subTest(change=change),self.assertRaises(ValueError):contract.length_identity(dict(row,**change))

    def test_deterministic_length_count_and_both_order(self):
        self.assertEqual(reader.selected_indices('64k',3),[3,5,7]);self.assertEqual(reader.selected_indices('128k',3),[2,4,6])
        self.assertEqual(reader.selected_indices('both',3),[3,5,7,2,4,6]);self.assertEqual(len(reader.selected_indices('both',10)),20)
        for length,count in [('256k',1),('both',True),('128k',0),('64k',11)]:
            with self.subTest(length=length,count=count),self.assertRaises(ValueError):reader.selected_indices(length,count)

    def test_manifest_seal_and_selected_order_are_independently_bound(self):
        value=reader.seal(dict(cohort=[dict(index=i) for i in range(2,22)]))
        config=dict(length='both',cases_per_length=2,evaluation_case_indices=[3,5,2,4]);contract=Mock()
        with patch.object(reader,'ORIGINAL_MANIFEST_SHA256',value['sha256']),patch.object(reader,'reproduction_contract',return_value=contract):
            rows=reader.original_selection(value,config);self.assertEqual([r['index'] for r in rows],[3,5,2,4])
            contract.audit_length_inputs.assert_called_once_with(rows)
            with self.assertRaises(ValueError):reader.original_selection(value,dict(config,evaluation_case_indices=[2,4,3,5]))
        with self.assertRaises(ValueError):reader.original_selection(value,config)

    def test_timing_ordinal_uses_this_run_selection(self):
        indices=reader.selected_indices('both',2)
        for ordinal,index in enumerate(indices):
            schedule=reader.timing_schedule(index,indices)
            self.assertEqual(schedule[0],(0,reader.ARMS[ordinal%2]))
            self.assertEqual(set(schedule),{(r,a) for r in range(3) for a in reader.ARMS})
        with self.assertRaises(ValueError):reader.timing_schedule(21,indices)

    def test_budget32_early_eos_is_not_quality(self):
        phase=dict(timing_valid=False,quality_eligible=False,diagnostic=False,diagnostic_gpu_profile=False,qualification_only=True,
            qualification_budget=True,eos_enabled=True,generation_token_budget=32,record=dict(output_ids=[100,1],
            output_ids_sha256=reader.digest([100,1]),success=True,rank=0,natural_EOS=True,finish_reason='stop',truncated=False,text='answer'))
        self.assertEqual(reader.qualification_output(phase),[100,1])
        with self.assertRaises(ValueError):reader.qualification_output(dict(phase,quality_eligible=True))
        bad=deepcopy(phase);bad['record'].update(finish_reason='length',truncated=True)
        with self.assertRaises(ValueError):reader.qualification_output(bad)

    def test_cohorts_are_summarized_separately(self):
        cases=[dict(index=i,length_band=band,context_sha256=str(i),full_over_s12_speedup=full/s12,
            clean_timing={'arms':{'recomputed':{'online_ttft_p50_s':full},'s1_system':{'online_ttft_p50_s':s12}}})
            for i,band,full,s12 in [(3,'64k',20.,5.),(2,'128k',36.,8.)]]
        value=reader.summarize_lengths(cases)
        self.assertEqual(value['64k']['completed_case_indices'],[3]);self.assertEqual(value['128k']['completed_case_indices'],[2])
        self.assertEqual(value['64k']['full_over_s12_geomean_speedup'],4.);self.assertEqual(value['128k']['full_over_s12_geomean_speedup'],4.5)

    def test_import_is_cpu_only_and_does_not_load_current_runner(self):
        subprocess.run([sys.executable,'-B','-c','import sys;import read_s12_reproduction_results;assert not any(x=="torch" or x.startswith("torch.") or x=="deepinfer" or x=="run_s12_reproduction" for x in sys.modules)'],cwd=Path(reader.__file__).resolve().parent,check=True,capture_output=True)

    def test_no_actual_evidence_never_claims_success(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'result.json';self.assertEqual(reader.main(['--directory',temp,'--output',str(output)]),1)
            value=json.loads(output.read_bytes());self.assertEqual(value['status'],'invalid');self.assertFalse(value['performance_accepted'])

if __name__=='__main__':unittest.main()
