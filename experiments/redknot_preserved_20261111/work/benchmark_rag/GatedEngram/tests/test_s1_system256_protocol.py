"""256K three-case orchestration contracts, independent of CUDA execution."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.core import digest
from gpu_pilot import s1_system256_protocol as protocol
from gpu_pilot import run_s1_system256_pilot as runner
from gpu_pilot import s1_system256_inputs as inputs
from tests import test_s1_system256_inputs as input_tests


class System256ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        input_tests.System256Inputs.setUpClass()

    def source_fixture(self):
        return copy.deepcopy(input_tests.System256Inputs.artifacts['manifest.json'])

    def select_fixture(self,manifest):
        with patch.object(protocol,'INPUT_TOKEN_RANGES',{'256k':(8000,11000)}),\
             patch.object(protocol,'TARGET_INPUT_TOKENS',{'256k':9000}):
            return runner.selected_rows(manifest)

    def test_actual_input_producer_to_new_runner_does_not_use_old_twenty_row_selector(self):
        manifest=self.source_fixture()
        with patch('gpu_pilot.run_s1_pilot.selected_rows',side_effect=AssertionError('old20selector must not run')):
            rows=self.select_fixture(manifest)
        self.assertEqual([r['index'] for r in rows],[2,3,4])
        for row in rows:
            geometry=runner.case_geometry(row)
            self.assertEqual(geometry['document_token_lengths'],[d['tokens'] for d in row['documents']])
            self.assertEqual(geometry['layer_chunk_checks'],40*((row['input_tokens']+8191)//8192))
            self.assertEqual(geometry['offline_bank_entry_count'],400)
            self.assertEqual(set(runner.case_scope(row)),{'length_band','length_band_ordinal','target_input_tokens',
                'source_document_count','original_document_count','source_domain','source_language','source_level','raw_answer_sha256'})

    def test_selection_rejects_extra_cases_reorder_ids_and_wrong_band(self):
        mutators=(lambda m:m['cohort'].append(copy.deepcopy(m['cohort'][0])),
                  lambda m:m['cohort'].reverse(),
                  lambda m:m['cohort'][0]['input_ids'].__setitem__(0,99),
                  lambda m:m['cohort'][0].update(length_band='128k'),
                  lambda m:m['cohort'][0]['documents'][0].update(tokens=1))
        for mutate in mutators:
            value=self.source_fixture();mutate(value)
            with self.assertRaises(ValueError):self.select_fixture(value)

    def test_longest_anchor_and_ep_selected_row_capacity_are_cpu_gates(self):
        rows=self.source_fixture()['cohort']
        rows[0],rows[1]=rows[1],rows[0]
        for ordinal,row in enumerate(rows,1):row.update(index=ordinal+1,length_band_ordinal=ordinal)
        if rows[0]['input_tokens']<rows[1]['input_tokens']:
            with self.assertRaisesRegex(ValueError,'longest'):self.select_fixture({'cohort':rows})
        # Exact input/body geometry is valid, but too many non-document rows
        # would overflow the unchanged native8192 selected EP/Engram buffer.
        n=40000;ids=list(range(n));docs=[]
        for j in range(10):
            lo=j*256;hi=lo+256;key=digest(ids[lo:hi])
            docs.append(dict(input_start=lo,input_end=hi,tokens=256,offline_position=0,online_position=lo,
                document_id=key,token_ids_sha256=key))
        cases=[dict(index=i,dataset='loong',length_band='256k',length_band_ordinal=i-1,
            target_input_tokens=40000,input_tokens=n,input_ids=ids,input_ids_sha256=digest(ids),documents=docs)
            for i in (2,3,4)]
        with patch.object(protocol,'INPUT_TOKEN_RANGES',{'256k':(39000,41000)}),\
             patch.object(protocol,'TARGET_INPUT_TOKENS',{'256k':40000}),self.assertRaisesRegex(ValueError,'8192'):
            runner.selected_rows({'cohort':cases})

    def test_registered_actual_lengths_and_eos_context_budget(self):
        for n,chunks,budget in ((260899,32,9437),(255192,32,15144),(252778,31,17558)):
            self.assertEqual((n+8191)//8192,chunks)
            self.assertEqual(runner.quality_token_budget(n),budget)
        for n in (270336,270337,0,True):
            with self.assertRaises(ValueError):runner.quality_token_budget(n)
        p=protocol.s1_system256_protocol()
        self.assertEqual(p['schema'],'s1-system256-pilot-protocol-v1')
        self.assertEqual(p['cases'],[2,3,4]);self.assertEqual(p['source_case_indices'],[1039,1036,1035])
        self.assertEqual(p['length_band_order'],['256k']*3)
        self.assertEqual(p['input_token_ranges'],{'256k':[245760,266240]})
        self.assertEqual(p['system_optimization']['batch_group_chunks'],4)
        self.assertEqual(p['system_qualification']['candidate_groups'],[1,2,4])
        self.assertIsNone(p['quality']['max_tokens']);self.assertEqual(p['timing']['output_tokens'],32)
        self.assertEqual(p['supervisor_timeout_seconds'],86400);self.assertIsNone(p['round_absolute_deadline_unix'])
        self.assertFalse(p['restore_original_holder_after_exit'])
        self.assertEqual(digest(p['baseline_core_policy']),p['core_policy_sha256'])
        for path,sha in p['core_source_sha256'].items():
            self.assertEqual(hashlib.sha256((runner.STAGE/path).read_bytes()).hexdigest(),sha)

    def test_all_three_fresh_arms_once_each_repeat(self):
        for i in (2,3,4):
            schedule=runner.timing_schedule(i);self.assertEqual(len(schedule),9)
            for repeat in range(3):self.assertEqual({a for r,a in schedule if r==repeat},set(protocol.ARMS))
        self.assertEqual([runner.timing_schedule(i)[0][1] for i in (2,3,4)],list(protocol.ARMS))
        for i in (0,1,5,20,True):
            with self.assertRaises(ValueError):runner.timing_schedule(i)

    def test_new_entrypoint_help_isolated_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-I','-B',str(Path(runner.__file__)),
                '--s1-protocol','unused.json','--help'],cwd=folder,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('usage:',result.stdout)


if __name__=='__main__':unittest.main()
