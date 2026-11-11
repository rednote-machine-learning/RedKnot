"""CPU-only tests of paired native QA preservation and outcome-blind selection."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gpu_pilot import broadqa_inputs as m
from gpu_pilot import musique30_inputs as old
from gpu_pilot import musique_longform_inputs as reuse
from tests.test_musique30_inputs import source_bytes, ENCODER, render, IDENTITY
from tests.test_natural_boundary_inputs import CharacterTokenizer


def reseal(data):
    for n in m.INPUT_COMPANIONS:data[n]=m.seal({k:v for k,v in data[n].items() if k!='sha256'})
    p=data['input-protocol.json']
    for k,n in (('source_records_sha256','source-records.json'),('selection_plan_sha256','selection-plan.json'),('tokenization_evidence_sha256','tokenization-evidence.json')):p[k]=data[n]['sha256']
    data['input-protocol.json']=m.seal({k:v for k,v in p.items() if k!='sha256'})
    v=data['manifest.json'];v['input_protocol_sha256']=data['input-protocol.json']['sha256'];v['selection_plan_sha256']=data['selection-plan.json']['sha256'];v['companion_sha256']={n:data[n]['sha256'] for n in m.INPUT_COMPANIONS}
    data['manifest.json']=m.seal({k:x for k,x in v.items() if k!='sha256'})


class BroadQAInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        templates={d:'Official question instructions.\n\n{context}\n\nQuestion: {input}\nAnswer:' for d in (*m.DATASETS,'musique')}
        cls.prompts=json.dumps(templates,sort_keys=True);cls.origin=dict(sha256=m.text_hash(cls.prompts),revision='fixture-fixed-revision')
        cls.identity=dict(IDENTITY,official_longbench_prompt_sha256=m.text_hash(cls.prompts));mu=source_bytes()
        with patch.object(old,'SOURCE_SHA256',hashlib.sha256(mu).hexdigest()),patch.object(old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)):
            p=old.build_inputs(mu,CharacterTokenizer(),render,tokenizer_identity=cls.identity,official_encoding_source=ENCODER,
                official_prompt_file_content=cls.prompts,official_prompt_source=cls.origin)
        cls.prior={n:p[n] for n in m.PREVIOUS_FILES};cls.raws={};cls.contents={}
        for d in m.DATASETS:
            cls.raws[d]=[dict(_id=d+'-id-'+str(i),dataset=d,input=f'What does original {d} source {i} describe?',answers=['unchanged gold '+str(i)],
                context=('\n' if d=='qasper' else '\n\n').join(f'Original {d}/{i}/{j} '+chr(65+j)*350 for j in range(12)),length=1) for i in range(6)]
            cls.contents[d]=('\n'.join(json.dumps(r) for r in cls.raws[d])+'\n').encode()
        cls.shas={d:hashlib.sha256(v).hexdigest() for d,v in cls.contents.items()}
        cls.history=m.seal(dict(schema='broadqa-history-inventory-v1',scope='fixture known sources only',model_outcomes_used=False,
            provenance=[dict(path='fixture',file_sha256='0'*64,seal_sha256=None)],source_files=cls.shas,
            source_identities=[[d,cls.raws[d][0]['_id']] for d in m.DATASETS],context_sha256=[],question_sha256=[],normalized_question_sha256=[]))
        with cls.pins():
            cls.artifacts=m.build_inputs(cls.prior,cls.contents,cls.history,CharacterTokenizer(),render,
                tokenizer_identity=cls.identity,official_encoding_source=ENCODER,official_prompt_file_content=cls.prompts,official_prompt_source=cls.origin)

    @classmethod
    def pins(cls):
        stack=contextlib.ExitStack()
        for module,key,val in ((m,'SOURCE_SHA256',cls.shas),(m,'SOURCE_COUNTS',{d:6 for d in m.DATASETS}),
            (m,'HISTORY_INVENTORY_SHA256',cls.history['sha256']),(old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)),
            (reuse,'PREVIOUS_MANIFEST_SHA256',cls.prior['manifest.json']['sha256']),
            (reuse,'PREVIOUS_PROTOCOL_SHA256',cls.prior['input-protocol.json']['sha256'])):
            stack.enter_context(patch.object(module,key,val))
        return stack

    def check(self,data=None):
        data=self.artifacts if data is None else data
        with tempfile.TemporaryDirectory() as temp,self.pins():
            for n in m.INPUT_COMPANIONS:Path(temp,n).write_text(json.dumps(data[n]))
            return m.validate_inputs(None,data['manifest.json'],data['input-protocol.json'],input_dir=temp)

    def test_full_builder_validator_pair_order_and_calibration_exact(self):
        self.assertTrue(self.check());manifest=self.artifacts['manifest.json']
        self.assertEqual(manifest['warmup'],self.prior['manifest.json']['warmup'])
        rows=manifest['cohort'];self.assertEqual([r['index'] for r in rows],list(range(2,34)))
        self.assertEqual(len({r['pair_id'] for r in rows}),16)
        for k in range(16):
            a,b=rows[2*k:2*k+2];m.check_pair(a,b)
            self.assertEqual((a['pair_ordinal'],b['pair_ordinal']),(k+1,k+1))
            self.assertEqual(a['dataset'],m.DATASETS[k%4]);self.assertNotEqual(a['source_index'],0)
            self.assertEqual(a['input_ids'],list(map(ord,a['prompt'])))
            self.assertEqual(b['input_ids'],list(map(ord,b['prompt'])))

    def test_full_source_context_gold_and_block_tokens_identical(self):
        for a,b in zip(self.artifacts['manifest.json']['cohort'][::2],self.artifacts['manifest.json']['cohort'][1::2]):
            raw=self.raws[a['dataset']][a['source_index']]
            self.assertEqual(a['references'],raw['answers']);self.assertEqual(b['references'],raw['answers'])
            self.assertEqual(''.join(a['context_text'][d['context_char_start']:d['context_char_end']] for d in a['documents']),raw['context'])
            self.assertTrue(all(d['tokens']>=256 for d in a['documents']))
            for da,db in zip(a['documents'],b['documents']):
                self.assertEqual(a['input_ids'][da['input_start']:da['input_end']],b['input_ids'][db['input_start']:db['input_end']])

    def test_gold_content_and_gold_length_do_not_affect_ranking(self):
        raw=copy.deepcopy(self.raws['qasper'][1]);first=m.source_metadata('qasper',1,raw)
        raw['answers']=['x'*10000];second=m.source_metadata('qasper',1,raw)
        self.assertEqual(first['tie_sha256'],second['tie_sha256']);self.assertEqual(first['pair_id'],second['pair_id'])
        self.assertNotEqual(first['source_row_sha256'],second['source_row_sha256'])
        p=self.artifacts['input-protocol.json'];self.assertFalse(p['gold_length_selection']);self.assertFalse(p['generated_length_selection']);self.assertIsNone(p['minimum_generated_tokens'])

    def test_history_source_context_raw_and_normalized_question_exclusion(self):
        meta=m.source_metadata('hotpotqa',1,self.raws['hotpotqa'][1])
        for sets,reason in (({('hotpotqa',meta['source_id'])},'known_history_source'),({meta['context_sha256']},'known_history_context'),
                           ({meta['question_sha256']},'known_history_question'),({meta['normalized_question_sha256']},'known_history_question')):
            i=('known_history_source','known_history_context','known_history_question').index(reason) if reason!='known_history_question' else (3 if meta['normalized_question_sha256'] in sets else 2)
            parts=[set(),set(),set(),set()];parts[i]=sets;self.assertEqual(m.exclusion_reason(meta,parts),reason)

    def test_natural_context_cannot_be_filled_or_token_truncated(self):
        with self.assertRaises(ValueError):m.natural_boundaries('multifieldqa_en','single paragraph')
        raw=copy.deepcopy(self.raws['hotpotqa'][1]);raw['context']='\n\n'.join('x' for _ in range(12))
        with self.assertRaises(ValueError):m.build_variant(raw,'hotpotqa',1,'official',CharacterTokenizer(),render,'{context}\n{input}')
        raw=copy.deepcopy(self.raws['hotpotqa'][1]);raw['context']=raw['context']*8
        with self.assertRaisesRegex(ValueError,'24576'):m.build_variant(raw,'hotpotqa',1,'official',CharacterTokenizer(),render,'{context}\n{input}')

    def test_resealed_pair_content_or_gold_mutation_rejected(self):
        for key in ('context_text','references','documents'):
            data=copy.deepcopy(self.artifacts);r=data['manifest.json']['cohort'][1]
            if key=='context_text':r[key]+=' omitted evidence'
            elif key=='references':r[key]=['replacement gold'];r['references_sha256']=m.digest(r[key])
            else:r[key][0]['context_char_start']+=1
            reseal(data)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(data)

    def test_resealed_calibration_history_and_template_mutations_rejected(self):
        for target in ('calibration','history','template'):
            data=copy.deepcopy(self.artifacts)
            if target=='calibration':data['manifest.json']['warmup'][0]['question']+=' changed'
            elif target=='history':
                h=data['source-records.json']['history_inventory'];h['source_identities']=[];h['sha256']=m.digest({k:v for k,v in h.items() if k!='sha256'})
            else:data['tokenization-evidence.json']['complete_prompt_templates']['qasper']='Answer '+str(self.raws['qasper'][1]['answers'])
            reseal(data)
            with self.subTest(target=target),self.assertRaises(ValueError):self.check(data)

    def test_resealed_missing_candidate_or_pair_order_rejected(self):
        for target in ('candidate','order','offset'):
            data=copy.deepcopy(self.artifacts)
            if target=='candidate':data['selection-plan.json']['rejected'].pop()
            elif target=='order':data['manifest.json']['cohort'][0]['prompt_variant']='complete'
            else:data['tokenization-evidence.json']['records'][-1]['token_offsets'].pop()
            reseal(data)
            with self.subTest(target=target),self.assertRaises(ValueError):self.check(data)


if __name__=='__main__':unittest.main()
