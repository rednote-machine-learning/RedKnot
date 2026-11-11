"""CPU source-preservation and negative audit tests; no GPU/tokenizer downloads."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import loong_inputs as m
from gpu_pilot import musique30_inputs as old
from gpu_pilot import musique_longform_inputs as reuse
from tests.test_musique30_inputs import source_bytes, ENCODER, render, IDENTITY, PROMPTS, PROMPT_SOURCE, rebind
from tests.test_natural_boundary_inputs import CharacterTokenizer

PROMPT_SOURCE_TEXT='# immutable official prompt-source fixture\n'
TEMPLATE='# Papers:\n{docs}\n{instruction}\nOriginal question: {question}'


def raw_row(i, names):
    answers=({'Reference':['Title A','Title B'],'Citation':[]},['A','B'],'original free text',{'A':'unchanged case result'})
    return dict(id='original-loong-'+str(i),type='paper',language='en',level=2+i%3,doc=names,
        question='What does original source '+str(i)+' establish?',instruction='Use these original documents only.',
        prompt_template=TEMPLATE,answer=copy.deepcopy(answers[i%4]),shuffle_doc=True,length='original metadata',set='fixture')


class LoongInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mu=source_bytes()
        with patch.object(old,'SOURCE_SHA256',hashlib.sha256(mu).hexdigest()),patch.object(old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)):
            p=old.build_inputs(mu,CharacterTokenizer(),render,tokenizer_identity=IDENTITY,official_encoding_source=ENCODER,
                official_prompt_file_content=PROMPTS,official_prompt_source=PROMPT_SOURCE)
        cls.prior={n:p[n] for n in m.PREVIOUS_FILES};cls.files={};cls.raws=[]
        for i in range(22):
            names=[]
            for d in range(5):
                name=f'{i}-{d}.md';names.append(name)
                n=(830 if i%2==0 else 390)+i
                cls.files['paper/'+name]=f'# Original paper {i}/{d}\n\n'+ '\n\n'.join(
                    f'Original paragraph {i}/{d}/{j}: '+chr(65+d)*n for j in range(2))
            cls.raws.append(raw_row(i,names))
        # A native empty-question template is meaningful: the task is in instruction.
        cls.raws[0]['question']='';cls.raws[0]['prompt_template']='# Papers:\n{docs}\n{instruction}'
        cls.raws[22:]=[raw_row(i,['missing-only-one-document.md']) for i in range(22,1600)]
        cls.raws[23]['doc']=['0-0.md']*5  # Original duplicates stay in source but are ineligible.
        cls.raw= ('\n'.join(json.dumps(r,ensure_ascii=False) for r in cls.raws)+'\n').encode()
        cls.sha=hashlib.sha256(cls.raw).hexdigest()
        with cls.pins():
            cls.artifacts=m.build_inputs(cls.prior,cls.raw,cls.files,CharacterTokenizer(),render,tokenizer_identity=IDENTITY,
                official_encoding_source=ENCODER,official_prompt_source=PROMPT_SOURCE_TEXT,document_archive_sha256='d'*64)

    @classmethod
    def pins(cls):
        stack=contextlib.ExitStack()
        for module,key,value in ((m,'SOURCE_SHA256',cls.sha),(m,'OFFICIAL_PROMPT_SOURCE_SHA256',m.text_hash(PROMPT_SOURCE_TEXT)),
            (m,'DOCUMENT_ARCHIVE_SHA256','d'*64),(m,'DOCUMENT_CATALOG_SHA256',m.digest(m.file_catalog(cls.files))),
            (m,'INPUT_TOKEN_RANGES',{'128k':[8000,11000],'64k':[4000,6500]}),
            (m,'TARGET_INPUT_TOKENS',{'128k':9000,'64k':4500}),
            (old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)),
            (reuse,'PREVIOUS_MANIFEST_SHA256',cls.prior['manifest.json']['sha256']),
            (reuse,'PREVIOUS_PROTOCOL_SHA256',cls.prior['input-protocol.json']['sha256'])):
            stack.enter_context(patch.object(module,key,value))
        return stack

    def check(self,data=None):
        data=self.artifacts if data is None else data
        with self.pins(),tempfile.TemporaryDirectory() as temp:
            for n in m.INPUT_COMPANIONS:Path(temp,n).write_text(json.dumps(data[n]))
            return m.validate_inputs(None,data['manifest.json'],data['input-protocol.json'],input_dir=temp)

    def test_builder_validator_twenty_unique_original_qas_and_exact_calibration(self):
        self.assertTrue(self.check());manifest=self.artifacts['manifest.json'];rows=manifest['cohort']
        self.assertEqual(manifest['warmup'],self.prior['manifest.json']['warmup'])
        self.assertEqual([r['index'] for r in rows],list(range(2,22)))
        self.assertEqual([r['length_band'] for r in rows],['128k','64k']*10)
        self.assertEqual([r['length_band_ordinal'] for r in rows],[x for i in range(1,11) for x in (i,i)])
        self.assertEqual(len({r['source_qa_sha256'] for r in rows}),20)
        self.assertEqual(rows[0]['input_tokens'],max(r['input_tokens'] for r in rows))

    def test_whole_prompt_original_gold_and_ten_blocks_never_cross_original_documents(self):
        for row in self.artifacts['manifest.json']['cohort']:
            raw=self.raws[row['source_index']];self.assertEqual(row['raw_answer'],raw['answer'])
            self.assertEqual(row['references'],[raw['answer'] if type(raw['answer']) is str else m.canonical(raw['answer'])])
            self.assertEqual(row['input_ids'],list(map(ord,row['prompt'])))
            self.assertEqual(row['original_document_order'],raw['doc']);self.assertEqual(len(row['documents']),10)
            self.assertEqual(''.join(row['context_text'][d['context_char_start']:d['context_char_end']] for d in row['documents']),row['context_text'])
            for original in row['original_documents']:
                blocks=[d for d in row['documents'] if d['source_document_ordinal']==original['source_document_ordinal']]
                self.assertTrue(blocks);self.assertEqual(blocks[0]['source_doc_char_start'],0)
                self.assertEqual(blocks[-1]['source_doc_char_end'],original['context_char_end']-original['context_char_start'])
                self.assertTrue(all(d['tokens']>=256 and d['source_document_name']==original['source_document_name'] for d in blocks))

    def test_original_duplicates_reject_candidate_without_aborting_or_normalizing_corpus(self):
        with self.pins():
            rows=m.source_rows(self.raw);self.assertEqual(rows[23]['doc'],['0-0.md']*5)
            with self.assertRaisesRegex(ValueError,'repeats'):m.build_row(rows[23],23,self.files,CharacterTokenizer(),render)
        rejects=self.artifacts['selection-plan.json']['rejected']
        self.assertTrue(any(r['source_index']==23 and 'repeats' in r['reason'] for r in rejects))

    def test_native_empty_question_and_literal_placeholders_are_preserved(self):
        raw=copy.deepcopy(self.raws[0]);raw['instruction']='Analyze literal {question} and {docs} examples.'
        docs=m.original_documents(raw,self.files);prompt,context=m.render_original(raw,docs,render)
        self.assertIn(raw['instruction'],prompt);self.assertEqual(prompt.count(context),1)
        raw['question']='nonempty question omitted by broken template'
        with self.assertRaisesRegex(ValueError,'placeholders'):m.render_original(raw,docs,render)

    def test_natural_paragraph_partition_cannot_cut_sentences_to_fill_ten(self):
        docs=[dict(source_document_ordinal=i,source_document_name=str(i),text='single uninterrupted sentence '+str(i)+'x'*900) for i in range(5)]
        text=''.join(d['text'] for d in docs);offsets=[(i,i+1) for i in range(len(text))]
        with self.assertRaisesRegex(ValueError,'ten cache blocks'):m.cache_geometry(docs,offsets,0)
        self.assertEqual(m.paragraph_boundaries('wrapped\nline\nwithout blank'),[0,26])

    def test_boundary_crossing_tokens_stay_online_and_are_not_duplicated(self):
        offsets=[(0,2),(2,5),(5,8),(8,10)]
        self.assertEqual(old.exact_token_span(offsets,0,3),(0,1));self.assertEqual(old.exact_token_span(offsets,3,10),(2,4))

    def test_ambiguous_or_missing_financial_source_refused_not_arbitrarily_chosen(self):
        raw=dict(self.raws[0],type='financial',doc=['Company'],level=2)
        for files in ({},{'financial/2024-Company-j.txt':'x','financial/2024-Company-n.txt':'y'}):
            with self.assertRaisesRegex(ValueError,'missing/ambiguous'):m.original_documents(raw,files)
        docs=m.original_documents(raw,{'financial/2024-Company-n.txt':'original bytes'})
        self.assertEqual(docs[0]['text'],'《n》\noriginal bytes\n\n')

    def test_official_legal_level4_task_mask_and_source_order_are_exact(self):
        files={'legal/legal.json':json.dumps({'a':{'content':'original body','result':'original result'}})}
        raw=dict(self.raws[0],type='legal',doc=['a'],level=4,instruction='阅读以上判决文书，我将给你若干份判决结果：')
        doc=m.original_documents(raw,files)[0];self.assertEqual(doc['text'],'《判决文书1》\noriginal body\n\n')
        self.assertEqual(doc['source_content_selection'],'content_only_official_level4')
        raw['level']=3;self.assertIn('original bodyoriginal result',m.original_documents(raw,files)[0]['text'])

    def test_length_outside_band_refused_without_padding_or_truncation(self):
        raw=self.raws[0]
        with self.pins(),patch.object(m,'INPUT_TOKEN_RANGES',{'128k':[100000,100001],'64k':[1,2]}):
            with self.assertRaisesRegex(ValueError,'length bands'):m.build_row(raw,0,self.files,CharacterTokenizer(),render)

    def test_selection_does_not_depend_on_gold_or_model_outputs(self):
        row=self.artifacts['manifest.json']['cohort'][0];changed=dict(row,raw_answer='changed',references=['long '*100000],generated_output='answer')
        self.assertEqual(m.candidate_metadata(row),m.candidate_metadata(changed))
        raw=copy.deepcopy(self.raws[0]);before=m.source_metadata(raw,0);raw['answer']='unrelated replacement'
        after=m.source_metadata(raw,0)
        self.assertEqual(before['tie_sha256'],after['tie_sha256']);self.assertEqual(before['source_qa_sha256'],after['source_qa_sha256'])

    def test_resealed_gold_question_instruction_template_and_raw_type_mutations_fail(self):
        for key in ('raw_answer','references','question','instruction','prompt_template','raw_answer_type'):
            data=copy.deepcopy(self.artifacts);r=data['manifest.json']['cohort'][0]
            r[key]=['changed'] if key=='references' else 'changed';rebind(data)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(data)

    def test_resealed_document_bytes_ordinals_or_last_boundary_mutations_fail(self):
        for kind in ('bytes','ordinal','end','source_count'):
            data=copy.deepcopy(self.artifacts);r=data['manifest.json']['cohort'][0]
            if kind=='bytes':
                files=data['source-records.json']['document_files'];files[next(iter(files))]+='added filler'
            elif kind=='ordinal':r['documents'][0]['source_document_ordinal']=1
            elif kind=='end':r['documents'][-1]['context_char_end']-=1
            else:r['original_document_count']=4
            rebind(data)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check(data)

    def test_resealed_calibration_offsets_source_coverage_and_band_mutations_fail(self):
        for kind in ('calibration','offsets','source','band'):
            data=copy.deepcopy(self.artifacts)
            if kind=='calibration':data['manifest.json']['warmup'][0]['question']+='changed'
            elif kind=='offsets':data['tokenization-evidence.json']['records'][-1]['token_offsets'].pop()
            elif kind=='source':data['selection-plan.json']['rejected'].pop()
            else:data['manifest.json']['cohort'][0]['length_band']='64k'
            rebind(data)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check(data)


if __name__=='__main__':unittest.main()
