"""Whole-context length-band selection and source/audit negative regressions."""
import contextlib,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from gpu_pilot import longcontext_inputs as m
from gpu_pilot import musique30_inputs as old
from gpu_pilot import musique_longform_inputs as reuse
from tests.test_musique30_inputs import source_bytes,ENCODER,render,IDENTITY,PROMPTS,PROMPT_SOURCE,rebind
from tests.test_natural_boundary_inputs import CharacterTokenizer
TEMPLATE='Read original text:\n\n$DOC$\n\nQuestion: $Q$\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\nThe correct answer is '
def raw_source(i,n,domain='Multi-Document QA'):
    return dict(_id='long-native-'+str(i),domain=domain,sub_domain='Academic',difficulty='hard',length='medium',
        context='\n\n'.join(f'Chapter {i}.{j}. '+chr(65+j)*n for j in range(12)),
        question='What was reported in original text '+str(i)+'?',choice_A='First original option',
        choice_B='Second original option',choice_C='Third original option',choice_D='Fourth original option',answer='B')
class LongContextInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mu=source_bytes()
        with patch.object(old,'SOURCE_SHA256',hashlib.sha256(mu).hexdigest()),patch.object(old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)):
            prior=old.build_inputs(mu,CharacterTokenizer(),render,tokenizer_identity=IDENTITY,official_encoding_source=ENCODER,
                official_prompt_file_content=PROMPTS,official_prompt_source=PROMPT_SOURCE)
        cls.prior={n:prior[n] for n in m.PREVIOUS_FILES}
        cls.raws=[raw_source(i,720+i) for i in range(3)]+[raw_source(i,340+i) for i in range(3,6)]
        cls.raws += [raw_source(6,340,'Single-Document QA'),raw_source(7,30),raw_source(8,400)]
        cls.raws[-1]['context']='single original paragraph '*200
        cls.pool=m.seal(dict(schema='longcontext-official-candidate-pool-v1',dataset='longbench_v2',source_revision=m.SOURCE_REVISION,
            selection_uses_model_outputs=False,selection_uses_gold=False,rows=[dict(source_index=i,raw=r) for i,r in enumerate(cls.raws)]))
        with cls.pins():
            cls.artifacts=m.build_inputs(cls.prior,cls.pool,CharacterTokenizer(),render,tokenizer_identity=IDENTITY,
                official_encoding_source=ENCODER,official_prompt_file_content=TEMPLATE)
    @classmethod
    def pins(cls):
        s=contextlib.ExitStack()
        for module,k,v in ((m,'SOURCE_POOL_SHA256',cls.pool['sha256']),(m,'OFFICIAL_PROMPT_SHA256',m.text_hash(TEMPLATE)),
            (m,'INPUT_TOKEN_RANGES',{'128k':[8000,11000],'64k':[4000,6500]}),(m,'TARGET_INPUT_TOKENS',{'128k':9000,'64k':4500}),
            (old,'OFFICIAL_ENCODING_SHA256',m.text_hash(ENCODER)),(reuse,'PREVIOUS_MANIFEST_SHA256',cls.prior['manifest.json']['sha256']),
            (reuse,'PREVIOUS_PROTOCOL_SHA256',cls.prior['input-protocol.json']['sha256'])):s.enter_context(patch.object(module,k,v))
        return s
    def check(self,data=None):
        data=self.artifacts if data is None else data
        with tempfile.TemporaryDirectory() as temp,self.pins():
            for n in m.INPUT_COMPANIONS:Path(temp,n).write_text(json.dumps(data[n]))
            return m.validate_inputs(None,data['manifest.json'],data['input-protocol.json'],input_dir=temp)
    def test_real_builder_validator_and_exact_calibration(self):
        self.assertTrue(self.check());r=self.artifacts['manifest.json'];self.assertEqual(r['warmup'],self.prior['manifest.json']['warmup'])
        self.assertEqual([x['index'] for x in r['cohort']],list(range(2,8)))
        self.assertEqual([x['length_band'] for x in r['cohort']],['128k','64k']*3)
        self.assertEqual([x['length_band_ordinal'] for x in r['cohort']],[1,1,2,2,3,3])
        self.assertEqual(r['cohort'][0]['input_tokens'],max(x['input_tokens'] for x in r['cohort']))
    def test_ten_natural_blocks_preserve_original_context_options_gold(self):
        for r in self.artifacts['manifest.json']['cohort']:
            raw=self.raws[r['source_index']];self.assertEqual(r['input_ids'],list(map(ord,r['prompt'])))
            self.assertEqual(''.join(r['context_text'][d['context_char_start']:d['context_char_end']] for d in r['documents']),raw['context'])
            self.assertEqual(len(r['documents']),10);self.assertTrue(all(d['tokens']>=256 for d in r['documents']))
            self.assertEqual(r['references'],['B']);self.assertEqual(r['gold_text'],raw['choice_B'])
            self.assertTrue(all(r['choice_'+a]==raw['choice_'+a] for a in 'ABCD'));self.assertFalse(r['cache_blocks_are_independent_articles'])
    def test_only_blank_paragraph_boundaries_not_wrapped_lines(self):
        with self.assertRaisesRegex(ValueError,'paragraph'):m.natural_boundaries('\n'.join('A sentence wraps here' for _ in range(20)))
        raw=raw_source(100,350);points=m.natural_boundaries(raw['context']);self.assertEqual(len(points),13)
        for n in points[1:-1]:self.assertEqual(raw['context'][n-2:n],'\n\n')
    def test_short_or_long_source_rejected_without_fill_or_truncation(self):
        with self.pins():
            for raw in (raw_source(20,20),raw_source(21,2000)):
                with self.assertRaisesRegex(ValueError,'length bands'):m.build_row(raw,0,CharacterTokenizer(),render,TEMPLATE)
    def test_selection_domain_actual_length_dedup_not_gold(self):
        eligible=self.artifacts['selection-plan.json']['eligible_candidates']
        with self.pins():chosen=m.choose_candidates(eligible)
        self.assertTrue(all(x['source_domain']=='Multi-Document QA' for x in chosen))
        raw=copy.deepcopy(self.raws[0]);meta=m.source_metadata(raw,0);raw['answer']='D';raw['choice_D']='x'*200000
        changed=m.source_metadata(raw,0);self.assertEqual(meta['tie_sha256'],changed['tie_sha256'])
        self.assertNotEqual(meta['source_row_sha256'],changed['source_row_sha256'])
        with self.pins(),self.assertRaisesRegex(ValueError,'distinct'):m.choose_candidates([dict(x,context_sha256='same') for x in eligible])
    def test_resealed_context_options_and_gold_mutations_rejected(self):
        for key in ('context_text','choice_A','gold_letter','gold_text','references'):
            data=copy.deepcopy(self.artifacts);r=data['manifest.json']['cohort'][0]
            r[key]=['D'] if key=='references' else 'changed original';rebind(data)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(data)
    def test_resealed_wrong_length_band_or_order_rejected(self):
        for key in ('length_band','length_band_ordinal','target_input_tokens'):
            data=copy.deepcopy(self.artifacts);r=data['manifest.json']['cohort'][0];r[key]='64k' if key=='length_band' else 999;rebind(data)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(data)
    def test_resealed_missing_source_candidate_offsets_or_boundary_rejected(self):
        for kind in ('source','candidate','offsets','boundary'):
            data=copy.deepcopy(self.artifacts)
            if kind=='source':data['source-records.json']['official_candidate_pool']['rows'][0]['raw']['context']+='fill'
            elif kind=='candidate':data['selection-plan.json']['rejected'].pop()
            elif kind=='offsets':data['tokenization-evidence.json']['records'][-1]['token_offsets'].pop()
            else:data['manifest.json']['cohort'][-1]['documents'][-1]['context_char_end']-=1
            rebind(data)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check(data)
    def test_resealed_template_and_calibration_mutations_rejected(self):
        for kind in ('template','calibration'):
            data=copy.deepcopy(self.artifacts)
            if kind=='template':data['tokenization-evidence.json']['official_prompt_file_content']+='The gold is B'
            else:data['manifest.json']['warmup'][0]['question']+='changed'
            rebind(data)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check(data)
    def test_original_placeholder_text_not_recursively_replaced(self):
        raw=copy.deepcopy(self.raws[0]);raw['context']+='$Q$ and $C_A$ are literal original strings.'
        p=m.render_original(raw,render,TEMPLATE);self.assertEqual(p.count(raw['context']),1);self.assertIn('$Q$ and $C_A$',p)
    def test_boundary_crossing_tokens_stay_online(self):
        offsets=[(0,2),(2,5),(5,8),(8,10)]
        self.assertEqual(old.exact_token_span(offsets,0,3),(0,1));self.assertEqual(old.exact_token_span(offsets,3,10),(2,4))
if __name__=='__main__':unittest.main()
