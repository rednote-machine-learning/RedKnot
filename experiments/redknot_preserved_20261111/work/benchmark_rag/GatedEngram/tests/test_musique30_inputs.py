"""CPU-only source, selection and geometry tests; no model-result fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import musique30_inputs as m
from gpu_pilot import natural_boundary_inputs as old
from tests.test_natural_boundary_inputs import CharacterTokenizer

ENCODER="def encode_messages(messages, thinking_mode):\n    assert thinking_mode == 'chat'\n    return '<user>'+messages[0]['content']+'</user><assistant>'\n"
TEMPLATES={d:'Read the original articles:\n{context}\nQuestion: {input}\nAnswer:'
           for d in ('musique','hotpotqa','2wikimqa')}
PROMPTS=json.dumps(TEMPLATES,sort_keys=True)
PROMPT_SOURCE=dict(sha256=m.text_hash(PROMPTS),revision='fixture-fixed-revision')
IDENTITY=dict(tokenizer_json_sha256='a'*64,encoding_sha256=m.text_hash(ENCODER),
    thinking_mode='chat',loader='tokenizers.Tokenizer.from_file',
    prompt_renderer="official_encoding.encode_messages(thinking_mode='chat')",tokenizers_version='0.21.4',
    official_longbench_prompt_sha256=m.text_hash(PROMPTS),official_longbench_prompt_revision='fixture-fixed-revision')


def render(messages,thinking_mode):
    assert thinking_mode=='chat'
    return '<user>'+messages[0]['content']+'</user><assistant>'


def raw_source(index,*,length=1660,passages=10,dataset='musique'):
    return dict(_id='id'+str(index),dataset=dataset,input='Which entity belongs to question '+str(index)+'?',
        answers=['original answer '+str(index)],context='\n\n'.join(
            f'Passage {i+1}:\nTitle {index} / {i}\n'+chr(97+i%26)*(length+i)
            for i in range(passages)),length=1,language='en',all_classes=None)


def source_bytes(*,short=False):
    rows=[raw_source(i,length=(300 if short else 1660)+i,passages=10 if i<40 else 3)
          for i in range(200)]
    return ('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n').encode()


def build(raw,**kwargs):
    return m.build_inputs(raw,CharacterTokenizer(),render,tokenizer_identity=IDENTITY,
        official_encoding_source=ENCODER,official_prompt_file_content=PROMPTS,
        official_prompt_source=PROMPT_SOURCE,**kwargs)


def rebind(artifacts):
    """Reseal to test semantic guards rather than only outer hash mismatch."""
    for name in m.INPUT_COMPANIONS:
        artifacts[name]=m.seal({k:v for k,v in artifacts[name].items() if k!='sha256'})
    protocol=artifacts['input-protocol.json']
    for field,name in [('source_records_sha256','source-records.json'),
                       ('selection_plan_sha256','selection-plan.json'),
                       ('tokenization_evidence_sha256','tokenization-evidence.json')]:
        protocol[field]=artifacts[name]['sha256']
    artifacts['input-protocol.json']=m.seal({k:v for k,v in protocol.items() if k!='sha256'})
    manifest=artifacts['manifest.json'];manifest['input_protocol_sha256']=artifacts['input-protocol.json']['sha256']
    manifest['selection_plan_sha256']=artifacts['selection-plan.json']['sha256']
    manifest['companion_sha256']={n:artifacts[n]['sha256'] for n in m.INPUT_COMPANIONS}
    artifacts['manifest.json']=m.seal({k:v for k,v in manifest.items() if k!='sha256'})


class MuSiQue30Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=source_bytes();cls.source_sha=hashlib.sha256(cls.raw).hexdigest()
        with patch.object(m,'SOURCE_SHA256',cls.source_sha),patch.object(m,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)):
            cls.artifacts=build(cls.raw)

    def check(self,artifacts=None,*,source_sha=None,original_sha=None):
        data=artifacts or self.artifacts
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'SOURCE_SHA256',source_sha or self.source_sha),\
                patch.object(m,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)),\
                patch.object(m,'ORIGINAL_MANIFEST_SHA256',original_sha or m.ORIGINAL_MANIFEST_SHA256):
            for name in m.INPUT_COMPANIONS:Path(temp,name).write_text(json.dumps(data[name]))
            return m.validate_inputs(None,data['manifest.json'],data['input-protocol.json'],input_dir=temp)

    def test_full_builder_and_independent_validator(self):
        self.assertTrue(self.check())
        data=self.artifacts;manifest=data['manifest.json'];rows=manifest['warmup']+manifest['cohort']
        self.assertEqual([r['index'] for r in rows],list(range(32)))
        self.assertEqual(len(manifest['cohort']),30)
        self.assertEqual(manifest['calibration_source'],'musique_longest_two')
        self.assertGreaterEqual(rows[0]['input_tokens'],16384)
        self.assertEqual([r['input_tokens'] for r in rows],sorted((r['input_tokens'] for r in rows),reverse=True))
        self.assertEqual(data['source-records.json']['original_source_jsonl'].encode(),self.raw)
        for row in rows:
            self.assertEqual(len(row['documents']),10)
            self.assertTrue(all(d['tokens']>=256 for d in row['documents']))
            self.assertEqual(''.join(row['context_text'][d['context_char_start']:d['context_char_end']]
                for d in row['documents']),row['context_text'])
            self.assertEqual(row['input_ids'],list(map(ord,row['prompt'])))

    def test_ten_groups_merge_all_original_passages_without_donor_or_tail_drop(self):
        raw=raw_source(500,length=300,passages=13)
        row,ev=m.build_row(raw,'musique',0,CharacterTokenizer(),render,TEMPLATES['musique'])
        self.assertEqual(len(row['documents']),10)
        self.assertEqual(row['documents'][-1]['passage_end'],13)
        self.assertIn('Title 500 / 12',row['context_text'])
        self.assertEqual(row['prompt'].count(raw['context']),1)
        self.assertEqual(row['references'],raw['answers'])
        self.assertEqual(ev['additional_source_contexts'],0)

    def test_fewer_than_ten_or_too_short_rejected_not_filled(self):
        for raw in (raw_source(0,passages=9),raw_source(0,passages=10,length=10)):
            with self.assertRaises(ValueError):m.build_row(raw,'musique',0,CharacterTokenizer(),render,TEMPLATES['musique'])

    def test_original_header_numbering_and_boundaries_required(self):
        raw=raw_source(0)
        for text in ('prefix'+raw['context'],raw['context'].replace('Passage 2:','Passage 7:')):
            with self.assertRaises(ValueError):m.natural_boundaries(text)

    def test_boundary_crossing_token_is_kept_as_online_gap(self):
        self.assertEqual(m.exact_token_span([(0,2),(2,5),(5,8),(8,10)],3,10),(2,4))
        self.assertEqual(m.exact_token_span([(0,2),(2,5),(5,8),(8,10)],0,3),(0,1))

    def test_fixed_ranking_uses_actual_prompt_length_not_source_length_or_gold(self):
        chosen=m.choose_candidates(self.artifacts['selection-plan.json']['eligible_candidates'],False)
        self.assertEqual([x['sample_id'] for x in chosen],self.artifacts['selection-plan.json']['selected_sample_ids'])
        source=self.artifacts['source-records.json'];self.assertTrue(all(r['raw']['length']==1 for r in source['records']))
        row=self.artifacts['manifest.json']['cohort'][0];changed=dict(row,references=['a completely different answer'])
        self.assertEqual(m.candidate_metadata(row),m.candidate_metadata(changed))

    def test_sample_order_mutation_even_resealed_is_rejected(self):
        data=copy.deepcopy(self.artifacts);rows=data['manifest.json']['cohort'];rows[0],rows[1]=rows[1],rows[0]
        rows[0]['index'],rows[1]['index']=2,3
        data['selection-plan.json']['selected_sample_ids']=[r['sample_id'] for r in data['manifest.json']['warmup']+rows]
        rebind(data)
        with self.assertRaises(ValueError):self.check(data)

    def test_missing_eligibility_inventory_source_is_rejected(self):
        data=copy.deepcopy(self.artifacts);data['selection-plan.json']['rejected'].pop();rebind(data)
        with self.assertRaises(ValueError):self.check(data)

    def test_source_jsonl_mutation_resealed_is_rejected(self):
        data=copy.deepcopy(self.artifacts);data['source-records.json']['original_source_jsonl']+=' ';rebind(data)
        with self.assertRaises(ValueError):self.check(data)

    def test_gold_mutation_resealed_is_rejected(self):
        data=copy.deepcopy(self.artifacts);row=data['manifest.json']['cohort'][0]
        row['references']=['leaked replacement'];row['references_sha256']=m.digest(row['references']);rebind(data)
        with self.assertRaises(ValueError):self.check(data)

    def test_missing_last_passage_and_wrong_block_count_rejected(self):
        for field in ('context_char_end','input_start'):
            data=copy.deepcopy(self.artifacts);data['manifest.json']['cohort'][0]['documents'][-1][field]-=1;rebind(data)
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(data)
        data=copy.deepcopy(self.artifacts);data['manifest.json']['cohort'][0]['documents'].pop();rebind(data)
        with self.assertRaises(ValueError):self.check(data)

    def test_token_offset_or_native_template_mutation_rejected(self):
        for mutate in (lambda d:d['tokenization-evidence.json']['records'][-1]['token_offsets'].pop(),
                       lambda d:d['tokenization-evidence.json']['official_prompt_templates'].update(musique='foreign')):
            data=copy.deepcopy(self.artifacts);mutate(data);rebind(data)
            with self.assertRaises(ValueError):self.check(data)

    def test_companion_missing_or_changed_seal_rejected(self):
        data=copy.deepcopy(self.artifacts);data['source-records.json']['sha256']='0'*64
        with self.assertRaises(ValueError):self.check(data)

    def test_short_musique_uses_exact_original_calibration_and_excludes_it_from_eval(self):
        raw=source_bytes(short=True);sha=hashlib.sha256(raw).hexdigest()
        calrows=[];source=[];evidence=[]
        for i,dataset in enumerate(('hotpotqa','2wikimqa')):
            original=raw_source(1000+i,dataset=dataset,passages=8,length=2200)
            row,ev=old.build_row(original,dataset,0,CharacterTokenizer(),render,TEMPLATES[dataset]);row.update(index=i,split='calibration')
            calrows.append(row);evidence.append(ev);source.append(dict(sample_id=row['sample_id'],dataset=dataset,
                source_index=0,source_row_sha256=m.digest(original),raw=original))
        original=m.seal(dict(warmup=calrows,tokenizer=IDENTITY))
        origin=dict(manifest=original,tokenizer=IDENTITY,source_records=source,evidence=evidence)
        with patch.object(m,'SOURCE_SHA256',sha),patch.object(m,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)),\
                patch.object(m,'load_original_calibration',return_value=origin):
            data=build(raw,calibration_input_dir='fixture-original')
        self.assertEqual(data['manifest.json']['warmup'],calrows)
        self.assertTrue(all(x['dataset']=='musique' and len(x['documents'])==10 for x in data['manifest.json']['cohort']))
        self.assertTrue(self.check(data,source_sha=sha,original_sha=original['sha256']))
        data['manifest.json']['warmup'][0]['question']='modified calibration';rebind(data)
        with self.assertRaises(ValueError):self.check(data,source_sha=sha,original_sha=original['sha256'])

    def test_short_musique_without_native16k_fallback_refuses_build(self):
        raw=source_bytes(short=True)
        with patch.object(m,'SOURCE_SHA256',hashlib.sha256(raw).hexdigest()),\
                patch.object(m,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)):
            with self.assertRaisesRegex(ValueError,'calibration-input-dir'):build(raw)


if __name__=='__main__':unittest.main()
