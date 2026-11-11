"""CPU regressions for immutable continuation inputs and native long-answer QA."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import musique30_inputs as old
from gpu_pilot import musique_longform_inputs as m
from tests.test_musique30_inputs import source_bytes, ENCODER, render, TEMPLATES, IDENTITY
from tests.test_natural_boundary_inputs import CharacterTokenizer


def qasper_row(index, *, gold_length=90, paragraphs=14):
    return dict(_id='qasper-id-'+str(index), dataset='qasper', input=f'What mechanisms are used in native paper {index}?',
        answers=[('original '+str(index)+' ' + 'x'*gold_length)[:gold_length]],
        context='\n'.join(f'Section {index}.{i}: '+chr(65+i)*350 for i in range(paragraphs)),
        length=1, language='en', all_classes=None)


def reseal(data):
    for name in m.INPUT_COMPANIONS:
        data[name]=m.seal({k:v for k,v in data[name].items() if k!='sha256'})
    p=data['input-protocol.json']
    for key,name in (('source_records_sha256','source-records.json'),('selection_plan_sha256','selection-plan.json'),
                     ('tokenization_evidence_sha256','tokenization-evidence.json')):
        p[key]=data[name]['sha256']
    data['input-protocol.json']=m.seal({k:v for k,v in p.items() if k!='sha256'})
    manifest=data['manifest.json']
    manifest['input_protocol_sha256']=data['input-protocol.json']['sha256']
    manifest['companion_sha256']={n:data[n]['sha256'] for n in m.INPUT_COMPANIONS}
    manifest['selection_plan_sha256']=data['selection-plan.json']['sha256']
    data['manifest.json']=m.seal({k:v for k,v in manifest.items() if k!='sha256'})


class MuSiQueLongformInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        templates=dict(TEMPLATES,qasper='Read the native scientific article:\n{context}\nQuestion: {input}\nAnswer:')
        cls.prompts=json.dumps(templates,sort_keys=True)
        cls.prompt_source=dict(sha256=m.text_hash(cls.prompts),revision='fixture-fixed-revision')
        cls.identity=dict(IDENTITY,official_longbench_prompt_sha256=m.text_hash(cls.prompts))
        mu=source_bytes()
        with patch.object(old,'SOURCE_SHA256',hashlib.sha256(mu).hexdigest()),\
                patch.object(old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)):
            prior=old.build_inputs(mu,CharacterTokenizer(),render,tokenizer_identity=cls.identity,
                official_encoding_source=ENCODER,official_prompt_file_content=cls.prompts,official_prompt_source=cls.prompt_source)
        cls.prior={name:prior[name] for name in m.PREVIOUS_FILES}
        cls.raw_rows=[qasper_row(i,gold_length=(60+i*5 if i<12 else 5)) for i in range(200)]
        cls.raw=('\n'.join(json.dumps(r) for r in cls.raw_rows)+'\n').encode()
        cls.source_sha=hashlib.sha256(cls.raw).hexdigest()
        with cls.pins():
            cls.artifacts=m.build_inputs(cls.prior,cls.raw,CharacterTokenizer(),render,tokenizer_identity=cls.identity,
                official_encoding_source=ENCODER,official_prompt_file_content=cls.prompts,official_prompt_source=cls.prompt_source)

    @classmethod
    def pins(cls):
        import contextlib
        stack=contextlib.ExitStack()
        for module,name,value in ((m,'PREVIOUS_MANIFEST_SHA256',cls.prior['manifest.json']['sha256']),
                (m,'PREVIOUS_PROTOCOL_SHA256',cls.prior['input-protocol.json']['sha256']),
                (m,'QASPER_SOURCE_SHA256',cls.source_sha),(m,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)),
                (old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER))):
            stack.enter_context(patch.object(module,name,value))
        return stack

    def check(self,data=None):
        data=self.artifacts if data is None else data
        with tempfile.TemporaryDirectory() as temp,self.pins():
            for name in m.INPUT_COMPANIONS:
                Path(temp,name).write_text(json.dumps(data[name]))
            return m.validate_inputs(None,data['manifest.json'],data['input-protocol.json'],input_dir=temp)

    def test_builder_full20_validator_and_exact12_continuation(self):
        self.assertTrue(self.check())
        manifest=self.artifacts['manifest.json'];prior=self.prior['manifest.json']
        self.assertEqual(manifest['warmup'],prior['warmup'])
        self.assertEqual([r['index'] for r in manifest['cohort']],list(range(2,20)))
        self.assertEqual([r['dataset'] for r in manifest['cohort']],['musique']*12+['qasper']*6)
        for row,original in zip(manifest['cohort'][:12],prior['cohort'][18:]):
            for key in original:
                if key!='index':
                    self.assertEqual(row[key],original[key],key)
            self.assertEqual(row['original_mu_case_index'],original['index'])
            self.assertEqual(row['original_mu_row_sha256'],m.digest(original))

    def test_qasper_natural_lines_preserve_full_context_and_original_gold(self):
        for row in self.artifacts['manifest.json']['cohort'][12:]:
            original=self.raw_rows[row['source_index']]
            self.assertEqual(row['references'],original['answers'])
            self.assertEqual(row['question'],original['input'])
            self.assertEqual(row['input_ids'],list(map(ord,row['prompt'])))
            self.assertEqual(''.join(row['context_text'][d['context_char_start']:d['context_char_end']]
                for d in row['documents']),original['context'])
            self.assertEqual(len(row['documents']),10)
            self.assertTrue(all(d['tokens']>=256 for d in row['documents']))

    def test_all_gold_lengths_not_only_longest_and_no_output_minimum(self):
        raw=qasper_row(900)
        for answers in (['x'*49,'y'*100],['x'*50,'y'*201]):
            raw['answers']=answers
            self.assertFalse(m.gold_is_eligible(m.original_gold_encoding(raw,CharacterTokenizer())))
        raw['answers']=['x'*50,'y'*200]
        self.assertTrue(m.gold_is_eligible(m.original_gold_encoding(raw,CharacterTokenizer())))
        protocol=self.artifacts['input-protocol.json']
        self.assertIsNone(protocol['minimum_generated_tokens'])
        self.assertFalse(protocol['generated_length_selection'])
        self.assertTrue(protocol['natural_eos_preserved'])

    def test_actual_gold_length_ranking_ignores_raw_length_and_model_outputs(self):
        selected=self.artifacts['manifest.json']['cohort'][12:]
        self.assertEqual([r['source_index'] for r in selected],[11,10,9,8,7,6])
        self.assertTrue(all(self.raw_rows[r['source_index']]['length']==1 for r in selected))
        candidates=copy.deepcopy(self.artifacts['selection-plan.json']['eligible_candidates'])
        for item in candidates:
            item['model_output']='must not affect source choice'
        chosen=m.choose_candidates(candidates)
        self.assertEqual([r['sample_id'] for r in chosen],[r['sample_id'] for r in selected])

    def test_duplicate_qasper_contexts_or_questions_are_not_selected_twice(self):
        candidates=copy.deepcopy(self.artifacts['selection-plan.json']['eligible_candidates'])
        candidates[1]['context_sha256']=candidates[0]['context_sha256']
        candidates[2]['question_sha256']=candidates[0]['question_sha256']
        chosen=m.choose_candidates(candidates)
        self.assertEqual(len({r['context_sha256'] for r in chosen}),6)
        self.assertEqual(len({r['question_sha256'] for r in chosen}),6)
        self.assertNotIn(candidates[1],chosen);self.assertNotIn(candidates[2],chosen)

    def test_short_empty_lines_merge_without_truncation_or_filler(self):
        context='\n'.join('Title\n\n'+chr(65+i)*400 for i in range(12))
        offsets=[(i,i+1) for i in range(len(context))]
        groups=m.partition_ten(m.qasper_boundaries(context),offsets,0)
        self.assertEqual(len(groups),10)
        self.assertEqual(''.join(context[g['char_start']:g['char_end']] for g in groups),context)
        self.assertTrue(all(g['input_end']-g['input_start']>=256 for g in groups))
        with self.assertRaises(ValueError):
            m.qasper_boundaries('no original paragraph boundary')
        with self.assertRaises(ValueError):
            m.partition_ten(list(range(11)),[(i,i+1) for i in range(10)],0)

    def test_boundary_crossing_tokens_remain_uncached_not_dropped(self):
        self.assertEqual(old.exact_token_span([(0,2),(2,5),(5,8),(8,10)],3,10),(2,4))
        self.assertEqual(old.exact_token_span([(0,2),(2,5),(5,8),(8,10)],0,3),(0,1))

    def test_resealed_original_musique_token_question_or_span_mutation_rejected(self):
        for kind in ('question','input_ids','documents'):
            data=copy.deepcopy(self.artifacts);row=data['manifest.json']['cohort'][0]
            if kind=='question':row[kind]+=' altered'
            elif kind=='input_ids':row[kind][0]+=1;row['input_ids_sha256']=m.digest(row[kind])
            else:row[kind][0]['input_start']+=1
            reseal(data)
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'resumed MuSiQue'):
                self.check(data)

    def test_resealed_calibration_or_previous_bundle_mutation_rejected(self):
        data=copy.deepcopy(self.artifacts);data['manifest.json']['warmup'][0]['question']+=' altered';reseal(data)
        with self.assertRaises(ValueError):self.check(data)
        data=copy.deepcopy(self.artifacts);prior=data['source-records.json']['previous_inputs']['manifest.json']
        prior['warmup'][0]['question']+=' altered'
        data['source-records.json']['previous_inputs']['manifest.json']=m.seal({k:v for k,v in prior.items() if k!='sha256'})
        reseal(data)
        with self.assertRaisesRegex(ValueError,'previous MuSiQue input identity'):
            self.check(data)

    def test_resealed_qasper_gold_and_whole_context_mutations_rejected(self):
        for field in ('references','context_text'):
            data=copy.deepcopy(self.artifacts);row=data['manifest.json']['cohort'][12]
            if field=='references':row[field]=['new longer answer'*10];row['references_sha256']=m.digest(row[field])
            else:row[field]=row[field][:-1];row['context_sha256']=m.text_hash(row[field])
            reseal(data)
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(data)

    def test_resealed_qasper_offsets_gold_counts_and_rank_mutations_rejected(self):
        for target in ('offset','gold_count','rank'):
            data=copy.deepcopy(self.artifacts)
            if target=='offset':data['tokenization-evidence.json']['records'][-1]['token_offsets'].pop()
            elif target=='gold_count':data['selection-plan.json']['eligible_candidates'][0]['gold_encoding']['reference_token_counts']=[500]
            else:data['selection-plan.json']['eligible_candidates'].reverse()
            reseal(data)
            with self.subTest(target=target),self.assertRaises(ValueError):self.check(data)

    def test_resealed_source_bytes_missing_eligibility_and_companion_rejected(self):
        for target in ('source','inventory','companion'):
            data=copy.deepcopy(self.artifacts)
            if target=='source':data['source-records.json']['original_qasper_source_jsonl']+=' '
            elif target=='inventory':data['selection-plan.json']['rejected'].pop()
            else:data['tokenization-evidence.json']['records'].pop()
            reseal(data)
            with self.subTest(target=target),self.assertRaises(ValueError):self.check(data)


if __name__=='__main__':
    unittest.main()
