"""New two-arm orchestration: no retired S1 phase, fresh longest-input gates."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.core import digest
from gpu_pilot import s11_protocol as protocol
from gpu_pilot import run_s11_pilot as runner
from tests.test_s1_runner import phase as original_phase


def seal(value):
    body={k:v for k,v in value.items() if k!='sha256'}
    return dict(body,sha256=digest(body))


def fixture():
    rows=[]
    for index,n in ((2,40000),(3,38000),(4,36000),(20,20000)):
        ids=list(range(n));docs=[];start=2
        for j in range(10):
            end=2+(n-30)*(j+1)//10;key=digest(ids[start:end])
            docs.append(dict(input_start=start,input_end=end,tokens=end-start,offline_position=0,
                online_position=start,token_ids_sha256=key,document_id=key));start=end
        band='128k' if index==20 else '256k'
        rows.append(dict(index=index,dataset='loong',length_band=band,
            length_band_ordinal=10 if index==20 else index-1,target_input_tokens=20000 if index==20 else 40000,
            input_tokens=n,input_ids=ids,input_ids_sha256=digest(ids),documents=docs))
    return dict(cohort=rows)


def phase(arm,repeat,index=20,quality=False):
    p=original_phase('s1' if arm=='s1_system' else arm,repeat,quality,32)
    p.update(schema=runner.PHASE_SCHEMA,index=index,arm=arm,s1_policy=arm=='s1_system',
        **protocol.phase_execution(arm),**protocol.phase_policy(arm))
    return seal(p)


class S11ProtocolTests(unittest.TestCase):
    def selected(self,manifest):
        with patch.object(protocol,'INPUT_TOKEN_RANGES',{'128k':(19000,21000),'256k':(35000,41000)}),\
             patch.object(protocol,'TARGET_INPUT_TOKENS',{'128k':20000,'256k':40000}):
            return runner.selected_rows(manifest)

    def test_registry_and_enabled_dense_reference_preserve_original_policy_and_sources(self):
        p=protocol.s11_protocol()
        self.assertEqual(p['arms'],['recomputed','s1_system'])
        self.assertEqual(p['cases'],[20,2,3,4]);self.assertEqual(p['qualification_case_index'],2)
        self.assertEqual(p['display_names'],{'recomputed':'Full','s1_system':'S1.1'})
        self.assertTrue(p['formal_original_s1_retired']);self.assertFalse(p['original_s1_reference_executed'])
        self.assertTrue(p['system_qualification']['reference_system_enabled'])
        self.assertEqual(p['system_qualification']['reference_arm'],'s1_system')
        self.assertEqual(p['system_qualification']['reference_batch_group_chunks'],1)
        self.assertEqual(p['system_qualification']['candidate_groups'],[1,2,4])
        self.assertEqual(digest(p['baseline_core_policy']),p['core_policy_sha256'])
        for path,sha in p['core_source_sha256'].items():
            self.assertEqual(hashlib.sha256((runner.STAGE/path).read_bytes()).hexdigest(),sha)
        self.assertEqual(p['supervisor_timeout_seconds'],86400)
        self.assertIsNone(p['round_absolute_deadline_unix']);self.assertFalse(p['restore_original_holder_after_exit'])
        self.assertIsNone(p['quality']['max_tokens']);self.assertEqual(p['timing']['output_tokens'],32)

    def test_retired_arm_cannot_enter_execution_or_timing(self):
        for group in (None,1,2,4):
            with self.assertRaises(ValueError):protocol.phase_execution('s1',batch_group_chunks=group)
        for i in protocol.EVALUATION_INDICES:
            schedule=runner.timing_schedule(i);self.assertEqual(len(schedule),6)
            for repeat in range(3):self.assertEqual([a for r,a in schedule if r==repeat].count('s1_system'),1)
            self.assertEqual({a for _,a in schedule},{'recomputed','s1_system'})
        self.assertNotEqual(runner.timing_schedule(20)[0][1],runner.timing_schedule(2)[0][1])
        for index in (0,1,6,18,True):
            with self.assertRaises(ValueError):runner.timing_schedule(index)

    def test_storage_order_is_not_execution_order_and_anchor_is_not_first_case(self):
        manifest=fixture();rows=self.selected(manifest)
        self.assertEqual([r['index'] for r in rows],[20,2,3,4])
        with patch.object(runner,'selected_rows',return_value=rows):
            self.assertEqual(runner.qualification_anchor(manifest)['index'],2)
            empty=runner.empty_group_qualification_row(manifest)
            self.assertEqual(empty['source_original_case_index'],2)
            self.assertEqual(empty['input_ids'],rows[1]['input_ids'][:32769])
            geometry=runner.manifest_geometry(manifest,protocol.s11_protocol())
        self.assertEqual(geometry['sparse_qualification']['index'],2)
        self.assertEqual(geometry['offline_payload_bytes_including_qualification'],
            geometry['offline_payload_bytes_all_cases']+geometry['qualification_anchor_payload_bytes'])
        self.assertEqual(geometry['sparse_qualification']['layer_chunk_checks'],40*((40000+8191)//8192))

    def test_wrong_population_token_identity_band_and_anchor_rejected(self):
        mutations=(lambda m:m['cohort'].pop(),lambda m:m['cohort'].append(m['cohort'][0]),
            lambda m:m['cohort'][0].update(index=6),lambda m:m['cohort'][3].update(length_band='256k'),
            lambda m:m['cohort'][0]['input_ids'].__setitem__(0,99),
            lambda m:m['cohort'][0]['documents'][0].update(tokens=1))
        for mutate in mutations:
            m=fixture();mutate(m)
            with self.assertRaises(ValueError):self.selected(m)
        m=fixture();m['cohort'][0],m['cohort'][1]=m['cohort'][1],m['cohort'][0]
        m['cohort'][0].update(index=2,length_band_ordinal=1);m['cohort'][1].update(index=3,length_band_ordinal=2)
        with self.assertRaisesRegex(ValueError,'longest'):self.selected(m)

    def test_two_arm_timing_summary_has_no_retired_baseline(self):
        phases=[phase(a,r) for r,a in runner.timing_schedule(20)]
        value=runner.summarize_timing(phases,20)
        self.assertEqual(set(value['arms']),set(protocol.ARMS))
        self.assertEqual(set(value['comparisons']),{'s1_system'})
        self.assertNotIn('ttft_speedup_vs_s1',value['comparisons']['s1_system'])
        for bad in (phases[:-1],phases[::-1]):
            with self.assertRaises(ValueError):runner.summarize_timing(bad,20)
        bad=copy.deepcopy(phases);bad[0]['arm']='s1';bad[0]=seal(bad[0])
        with self.assertRaises(ValueError):runner.summarize_timing(bad,20)

    def test_formal_group_fallback_is_rejected_and_quality_has_only_context_bound(self):
        p=phase('s1_system',0);p.update(**protocol.phase_execution('s1_system',batch_group_chunks=1))
        with self.assertRaisesRegex(ValueError,'fallback'):runner.verify_phase(seal(p),index=20,arm='s1_system',timing=True,quality=False)
        for n,budget in ((260899,9437),(255192,15144),(252778,17558)):
            self.assertEqual(runner.quality_token_budget(n),budget)

    def test_new_entrypoint_help_never_imports_runtime_from_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-I','-B',str(Path(runner.__file__)),
                '--s1-protocol','unused.json','--help'],cwd=folder,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('usage:',result.stdout)


if __name__=='__main__':unittest.main()
