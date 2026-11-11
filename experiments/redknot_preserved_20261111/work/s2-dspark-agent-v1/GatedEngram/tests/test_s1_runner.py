"""S1 source/phase/qualification admission regressions; no CUDA imports."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gpu_pilot import run_s1_pilot as runner
from gpu_pilot import read_s1_results as reader
from gpu_pilot import s1_protocol as protocol
from tests.test_natural_boundary_runner import row as old_row,phase as old_phase
from tests.test_s1_results import fixture,seal


def row():
    data=old_row();ids=list(range(5300));docs=[];start=3
    for ordinal in range(10):
        end=start+300+ordinal*30;key=runner._digest(ids[start:end])
        docs.append(dict(input_start=start,input_end=end,tokens=end-start,offline_position=0,
            online_position=start,document_id=key,token_ids_sha256=key))
        start=end+2
    data.update(dataset='musique',input_ids=ids,input_tokens=len(ids),input_ids_sha256=runner._digest(ids),documents=docs)
    return data


def phase(arm='s1',repeat=0,quality=False,tokens=32):
    p=old_phase('headtail128_link' if arm=='s1' else 'recomputed',repeat,quality=quality,tokens=tokens)
    p.update(schema=runner.PHASE_SCHEMA,arm=arm,s1_policy=arm=='s1',s1_protocol_sha256='d'*64,
        **protocol.phase_policy(arm))
    p.pop('natural_boundary_protocol_sha256');p.pop('dynamic_document_swa');p.pop('dynamic_refresh_start_layer')
    return seal(p)


class S1RunnerTests(unittest.TestCase):
    def test_two_fresh_arms_and_explicit_policy_change(self):
        p=protocol.s1_protocol()
        self.assertEqual(p['arms'],['recomputed','s1']);self.assertEqual(p['cases'],list(protocol.EVALUATION_INDICES))
        self.assertFalse(p['selection_is_adaptive']);self.assertFalse(p['extra_cross_document_tail_links'])
        self.assertTrue(p['selected_rows_preserved_through_layer39'])
        self.assertFalse(p['native_yoco140_intersection_applied'])
        self.assertEqual(p['unselected_deep_residual_rows'],0);self.assertEqual(p['unselected_deep_mhc_rows'],0)
        self.assertFalse(p['full_model_numerical_equivalence_claim'])
        self.assertEqual(p['native_engram_layers'],[1,14])
        self.assertEqual(p['datasets'],['loong']);self.assertEqual(p['document_count'],10)
        self.assertFalse(p['first_case_reference_conflict_annotation'])
        self.assertEqual(p['supervisor_timeout_seconds'],7200)
        self.assertEqual(p['timeout_origin'],'original supervisor GPU release')
        self.assertIsNone(p['absolute_deadline_unix'])
        self.assertNotIn('hard_deadline_unix',p)
        self.assertTrue(p['quality']['preserve_full_output_ids'])
        self.assertEqual(p['length_band_order'],['128k','64k']*10)
        self.assertEqual(p['original_document_count_range'],[5,10])
        self.assertFalse(p['cache_blocks_are_original_documents'])
        self.assertFalse(p['cache_blocks_cross_original_document_boundaries'])
        self.assertEqual(p['cache_block_count'],10)

    def test_case_scope_preserves_source_documents_and_original_gold_without_mc_fields(self):
        data=dict(length_band="128k",length_band_ordinal=1,target_input_tokens=131072,
            source_document_count=7,original_document_count=7,source_domain="paper",
            source_language="en",source_level=3,raw_answer_sha256="a"*64)
        self.assertEqual(runner.case_scope(data),data)
        data.update(gold_letter="A",choice_A="legacy MC",raw_answer={"original":"nested answer"})
        self.assertNotIn("gold_letter",runner.case_scope(data))
        self.assertNotIn("raw_answer",runner.case_scope(data))

    def test_quality_natural_eos_above64_and_technical_bound_only(self):
        runner.verify_phase(phase(quality=True,tokens=200),index=2,arm='s1',timing=False,quality=True)
        self.assertIsNone(protocol.s1_protocol()['quality']['max_tokens'])
        self.assertEqual(protocol.s1_protocol()['timing']['output_tokens'],32)
        p=phase(quality=True,tokens=64);p['record'].update(finish_reason='length',truncated=True,
            quality_output_complete=False,quality_limit_reason='engine_context_limit');p=seal(p)
        with self.assertRaises(ValueError):runner.verify_phase(p,index=2,arm='s1',timing=False,quality=True)

    def test_three_repeats_each_arm_and_counterbalanced_order(self):
        for i in protocol.EVALUATION_INDICES:
            schedule=runner.timing_schedule(i);self.assertEqual(len(schedule),6)
            for repeat in range(3):self.assertEqual({a for r,a in schedule if r==repeat},set(runner.ARMS))
            if i<protocol.EVALUATION_INDICES[-1]:self.assertNotEqual(schedule[0][1],runner.timing_schedule(i+1)[0][1])
        for i in (0,1,protocol.EVALUATION_INDICES[-1]+1,True):
            with self.assertRaises(ValueError):runner.timing_schedule(i)

    def test_exact_natural_length_bands_and_interleaving_required(self):
        rows=[dict(index=i,dataset='loong',length_band=protocol.LENGTH_BANDS[(i-2)%2],
            length_band_ordinal=(i-2)//2+1,target_input_tokens=protocol.TARGET_INPUT_TOKENS[protocol.LENGTH_BANDS[(i-2)%2]],
            input_tokens=protocol.TARGET_INPUT_TOKENS[protocol.LENGTH_BANDS[(i-2)%2]],
            source_document_count=5,original_document_count=5,raw_answer_sha256='a'*64) for i in protocol.EVALUATION_INDICES]
        self.assertEqual(runner.selected_rows(dict(cohort=rows)),rows)
        for mutate in (lambda x:x.pop(),lambda x:x.append(dict(x[0])),lambda x:x[-1].update(index=2),
                       lambda x:x[1].update(length_band='128k'),lambda x:x[2].update(dataset='hotpotqa'),
                       lambda x:x[0].update(input_tokens=16384),lambda x:x[-1].update(length_band_ordinal=2),
                       lambda x:x[0].update(source_document_count=4),lambda x:x[0].update(original_document_count=10),
                       lambda x:x[0].update(raw_answer_sha256=None),lambda x:x[2].update(input_tokens=131073),
                       lambda x:x[0].update(target_input_tokens=65536),lambda x:x[0].update(input_tokens=147457)):
            bad=copy.deepcopy(rows);mutate(bad)
            with self.assertRaises(ValueError):runner.selected_rows(dict(cohort=bad))
        with self.assertRaises(ValueError):runner.case_geometry(old_row())
        self.assertEqual(runner.case_geometry(row())['offline_bank_entry_count'],400)

    def test_long_context_uses_actual_chunks_bank_and_remaining_context(self):
        for n in (65536,65537,131072,131073):
            data=row();ids=list(range(n));docs=[];start=1
            for i in range(10):
                end=1+(n-2)*(i+1)//10;key=runner._digest(ids[start:end])
                docs.append(dict(input_start=start,input_end=end,tokens=end-start,offline_position=0,
                    online_position=start,document_id=key,token_ids_sha256=key));start=end
            data.update(input_ids=ids,input_tokens=n,input_ids_sha256=runner._digest(ids),documents=docs)
            geometry=runner.case_geometry(data)
            self.assertEqual(geometry,reader.case_geometry(data))
            self.assertEqual(geometry['native_prefill_chunks'],(n+8191)//8192)
            self.assertEqual(geometry['layer_chunk_checks'],40*((n+8191)//8192))
            self.assertEqual(geometry['offline_bank_bytes'],37*10*128*512*2+(n-2)*9216)
            self.assertEqual(runner.quality_token_budget(n),270336-n)
            chunks=reader.chunk_oracle(reader.documents(data),n)
            self.assertEqual(sum(len(c['selected_positions']) for c in chunks),1280+2)
            if n>=131072:self.assertTrue(any(not c['selected_positions'] for c in chunks))

    def test_core_algorithm_and_quality_policy_are_frozen_to_previous_run(self):
        p=protocol.s1_protocol()
        self.assertEqual(len(p['core_source_sha256']),6)
        self.assertEqual(p['native_snapshot_sources_sha256'],'d89dad6c23f9df603b7ba27ca3d07717c315bc9d4dcbf7471346552b300cb94f')
        self.assertEqual(runner._digest({k:p[k] for k in protocol.CORE_POLICY_KEYS}),protocol.CORE_POLICY_SHA256)
        with mock.patch.object(protocol,'phase_policy',side_effect=lambda arm:dict(head_tokens=128)):
            with self.assertRaisesRegex(ValueError,'policy changed'):protocol.s1_protocol()
        self.assertTrue(p['longest_length_band_qualified_first'])
        self.assertNotIn('prompt_effect_requires_both_variant_closures',p)

    def test_runner_summary_matches_independent_reader(self):
        values=[phase(arm,repeat) for repeat,arm in runner.timing_schedule(2)]
        for p,(repeat,arm) in zip(values,runner.timing_schedule(2)):
            p.update(repeat=repeat,arm=arm);p.update(sha256=runner._digest({k:v for k,v in p.items() if k!='sha256'}))
        samples={arm:[p['record']['online_ttft_s'] for p in values if p['arm']==arm] for arm in runner.ARMS}
        self.assertEqual(runner.summarize_timing(values,2),reader.expected_timing_summary(samples))
        with self.assertRaises(ValueError):runner.summarize_timing(values[::-1],2)

    def test_variable_geometry_matches_independent_storage_oracle(self):
        data=row();self.assertEqual(runner.case_geometry(data),reader.case_geometry(data))
        self.assertEqual(runner.case_geometry(data)['offline_swa_halo_rows'],37*10*128)
        for mutation in (lambda x:x['documents'][0].update(tokens=32768),
                         lambda x:x['documents'][1].update(input_start=x['documents'][0]['input_end']-1)):
            bad=copy.deepcopy(data);mutation(bad)
            with self.assertRaises(ValueError):runner.case_geometry(bad)

    def test_calibration_empty_chunk_uses_exact_original_ids_and_one_document(self):
        ids=list(range(17000));manifest=dict(warmup=[dict(index=0,input_ids=ids,dataset='musique')])
        original=copy.deepcopy(manifest);q=runner.empty_qualification_row(manifest)
        self.assertEqual(q['input_ids'],ids[:16384]);self.assertEqual(manifest,original)
        chunks=reader.chunk_oracle(reader.documents(q),16384)
        self.assertEqual([len(c['selected_positions']) for c in chunks],[0,128])
        from kv_cache_engram.s1_runtime import s1_documents
        self.assertEqual(len(s1_documents(q['input_ids'],q['documents'])),1)

    def test_ten_formal_phases_share_one_immutable_bank(self):
        data,p=fixture(qualified=False);phases=[]
        for i in range(10):
            value=copy.deepcopy(p);value['arm']='recomputed' if i%2 else 's1';value['controller']['arm']=value['arm'];phases.append(value)
        self.assertEqual(runner.verify_shared_bank(phases,data)['bank_generation'],1)
        phases[-1]['controller']['bank_generation']=2
        with self.assertRaises(ValueError):runner.verify_shared_bank(phases,data)
        with self.assertRaises(ValueError):runner.verify_shared_bank(phases[:-1],data)

    def test_incomplete_new_dependency_pins_fail_before_runtime(self):
        with mock.patch.object(protocol,'DEPENDENCIES',{}),self.assertRaisesRegex(ValueError,'must be frozen'):
            runner.source_hashes(Path('/unused'))

    def test_evidence_publish_never_overwrites_existing_json(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a.json';runner.write_json(p,dict(x=1))
            original=p.read_bytes()
            with self.assertRaises(ValueError):runner.write_json(p,dict(x=2))
            self.assertEqual(p.read_bytes(),original)

    def test_s1_policy_and_timing_scope_mutation_rejected(self):
        for update in (dict(s1_policy=False),dict(compact_start_layer=4),dict(tail_tokens=256),
                       dict(swa_halo_tokens=0),dict(diagnostic=True),dict(schema='foreign')):
            p=phase();p.update(update);p=seal(p)
            with self.subTest(update=update),self.assertRaises(ValueError):runner.verify_phase(p,index=2,arm='s1',timing=True,quality=False)

    def test_runtime_gpu_qualification_requires_all_ranks_and_each_real_check(self):
        identity=dict(execution_configuration_sha256='a'*64,checkpoint_index_sha256='b'*64);reports=[]
        for rank in range(8):
            reports.append(seal(dict(schema='s1-runtime-rank-qualification-v1',rank=rank,world_size=8,
                **dict(execution_configuration_sha256=identity['execution_configuration_sha256'],model_sha256=identity['checkpoint_index_sha256']),
                passed=True,cuda_exercised=True,quality_eligible=False,timing_valid=False,
                native_empty_ep_requires_separate_all_ranks_proof=True,output_sha256='c'*64,
                checks=[dict(name=n,passed=True,actual_rows=0 if n=='empty_packed_sf_abi' else 3,
                    cuda_tensor_device='cuda:0') for n in reader.RUNTIME_CHECK_NAMES])))
        value=seal(dict(schema='s1-runtime-all-ranks-qualification-v1',passed=True,
            execution_configuration_sha256='a'*64,model_sha256='b'*64,reports=reports,
            empty_ep_qualification_sha256='d'*64,quality_eligible=False,timing_valid=False))
        reader.validate_runtime_qualification(value,identity,'d'*64)
        for mutate in (lambda x:x['reports'].pop(),lambda x:x['reports'][0].update(cuda_exercised=False),
                       lambda x:x['reports'][0]['checks'][0].update(cuda_tensor_device='cpu')):
            bad=copy.deepcopy(value);mutate(bad);bad['reports']=[seal(x) for x in bad['reports']];bad=seal(bad)
            with self.assertRaises(ValueError):reader.validate_runtime_qualification(bad,identity,'d'*64)


if __name__=='__main__':unittest.main()
