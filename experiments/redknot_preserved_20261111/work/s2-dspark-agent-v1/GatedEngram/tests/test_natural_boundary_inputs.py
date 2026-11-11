import types
import unittest
from gpu_pilot.natural_boundary_inputs import build_row,exact_token_span,natural_boundaries,partition_eight

class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        return types.SimpleNamespace(ids=list(map(ord,text)),offsets=[(i,i+1) for i in range(len(text))])
    def decode(self,ids,skip_special_tokens=False):return ''.join(map(chr,ids))

class NaturalBoundaryInputs(unittest.TestCase):
    def source(self,lengths):
        return dict(_id='natural',context='\n\n'.join('Passage '+str(i+1)+':\n'+chr(97+i)*n for i,n in enumerate(lengths)),input='Which original entity?',answers=['original entity'])
    def make(self,raw):
        return build_row(raw,'hotpotqa',7,CharacterTokenizer(),lambda messages,thinking_mode:'<user>'+messages[0]['content']+'</user><assistant>','Passages:\n{context}\nQuestion: {input}\nAnswer:')
    def test_whole_native_context_and_question_preserved(self):
        raw=self.source([350,500,900,310,760,400,800,650,530,450]);row,evidence=self.make(raw)
        self.assertEqual(''.join(raw['context'][d['context_char_start']:d['context_char_end']] for d in row['documents']),raw['context'])
        self.assertEqual(row['prompt'].count(raw['context']),1);self.assertEqual(row['prompt'].count(raw['input']),1)
        self.assertEqual(row['references'],raw['answers']);self.assertEqual(row['input_ids'],list(map(ord,row['prompt'])))
        self.assertEqual(len(row['documents']),8);self.assertGreater(len(set(d['tokens'] for d in row['documents'])),1)
        self.assertTrue(evidence['whole_prompt_single_encoding'])
    def test_only_existing_passage_boundaries(self):
        raw=self.source([500]*10);row,_=self.make(raw);cuts=natural_boundaries(raw['context'])
        for d in row['documents']:
            self.assertIn(d['context_char_start'],cuts);self.assertIn(d['context_char_end'],cuts)
            self.assertEqual(row['input_ids'][d['input_start']:d['input_end']],list(map(ord,raw['context'][d['context_char_start']:d['context_char_end']])))
    def test_crossing_boundary_token_remains_online(self):
        offsets=[(0,2),(2,5),(5,8),(8,10)]
        self.assertEqual(exact_token_span(offsets,3,10),(2,4))
        self.assertEqual(exact_token_span(offsets,0,3),(0,1))
    def test_short_material_is_rejected_not_padded(self):
        with self.assertRaisesRegex(ValueError,'minimum256'):self.make(self.source([10]*8))
    def test_not_enough_articles_is_rejected_not_token_split(self):
        with self.assertRaisesRegex(ValueError,'eight original'):self.make(self.source([2000]*7))
    def test_partition_is_deterministic_and_ordered(self):
        raw=self.source([300,310,900,950,700,800,350,600,400,810]);row,ev=self.make(raw)
        groups=partition_eight(ev['original_passage_char_boundaries'],ev['token_offsets'],row['context_prompt_char_start'])
        self.assertEqual(groups,partition_eight(ev['original_passage_char_boundaries'],ev['token_offsets'],row['context_prompt_char_start']))
        self.assertEqual(groups[0]['char_start'],0);self.assertEqual(groups[-1]['char_end'],len(raw['context']))
        self.assertTrue(all(a['char_end']==b['char_start'] for a,b in zip(groups,groups[1:])))

if __name__=='__main__':unittest.main()
