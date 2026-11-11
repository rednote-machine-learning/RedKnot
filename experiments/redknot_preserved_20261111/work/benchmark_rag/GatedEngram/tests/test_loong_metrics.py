import unittest
from gpu_pilot import loong_metrics as m

class LoongMetrics(unittest.TestCase):
    def test_category_order_ignores_list_order_but_counts_duplicates(self):
        gold={'high':['Alpha','Beta'],'low':[]}
        self.assertEqual(m.score_answer('{"low":[],"high":["Beta","Alpha"]}',gold)['structured_exact'],1)
        bad=m.score_answer('{"high":["Alpha","Alpha"],"low":[]}',gold)
        self.assertEqual(bad['structured_exact'],0);self.assertEqual(bad['field_true_positive'],2)
        self.assertAlmostEqual(bad['field_f1'],2/3)
    def test_ordered_list_ranking_and_empty_category_are_fields(self):
        self.assertEqual(m.score_answer('["B","A"]',['A','B'])['field_true_positive'],0)
        self.assertEqual(m.score_answer('{}',{'empty':[]})['structured_exact'],0)
        self.assertEqual(m.score_answer('{}',{})['structured_exact'],1)
        self.assertEqual(m.score_answer('[]',[])['field_f1'],1)
    def test_whole_json_python_literal_and_single_fence_modes(self):
        for text,mode in [("{'x': ['A']}",'python_literal'),('```json\n{"x":["A"]}\n```','fenced_json'),("```python\n{'x':['A']}\n```",'fenced_python_literal')]:
            q=m.score_answer(text,{'x':['A']});self.assertEqual(q['parse_mode'],mode);self.assertEqual(q['structured_exact'],1)
    def test_prose_duplicate_keys_nonfinite_code_and_unsafe_shapes_rejected(self):
        for text in ['Answer: {"x":[]}','{"x":[],"x":["B"]}',"{'x': [], 'x': ['B']}",'NaN','1e999','__import__("os").system("true")','```json\n{}\n```\n```json\n{}\n```',"{1: []}"]:
            with self.subTest(text=text):self.assertFalse(m.parse_output(text)['success'])
        self.assertFalse(m.score_answer('{"x":"A"}',{'x':['A']})['shape_compatible'])
        self.assertFalse(m.score_answer('True',1)['shape_compatible'])
    def test_units_percent_sign_minus_and_case_are_not_erased(self):
        for prediction,gold in [('50%','0.5'),('$4','4'),('0.04','-0.04'),('A','a')]:
            self.assertEqual(m.score_answer(prediction,gold)['normalized_text_exact'],0)
        self.assertEqual(m.score_answer('500',.5)['structured_exact'],0)
        self.assertEqual(m.score_answer('"0.5"',.5)['structured_exact'],0)
    def test_free_text_has_no_structural_or_semantic_judgment(self):
        q=m.score_answer('144,480.00','144,480.00')
        self.assertIsNone(q['structured_exact']);self.assertIsNone(q['parse_success'])
        self.assertEqual(q['semantic_assessment'],'not_assessed_free_text')
        self.assertEqual(q['normalized_text_exact'],1)
    def test_format_failure_is_zero_exact_but_field_denominator_explicit(self):
        a=m.score_answer('not JSON',{'x':['A']});b=m.score_answer('{"x":["A"]}',{'x':['A']})
        self.assertEqual(a['semantic_assessment'],'format_failure_semantic_unassessed')
        self.assertIsNone(a['field_f1']);summary=m.summarize_scores([a,b,m.score_answer('x','x')])
        self.assertEqual(summary['structured_cases'],2);self.assertEqual(summary['structured_exact_percent'],50)
        self.assertEqual(summary['shape_compatible_count'],1);self.assertEqual(summary['valid_structure_field_micro_f1'],1)
    def test_agreement_does_not_mean_gold_correct(self):
        q=m.answer_agreement('["wrong"]','["wrong"]',['right'])
        self.assertTrue(q['structural_equal']);self.assertFalse(q['agreement_is_gold_correctness'])
        self.assertFalse(m.answer_agreement('Answer: A','Answer: A',['A'])['structural_comparison_available'])
    def test_macro_and_micro_are_distinct(self):
        scores=[m.score_answer('["wrong"]',['A']),m.score_answer('["B","C","D"]',['B','C','D'])]
        q=m.summarize_scores(scores);self.assertEqual(q['valid_structure_field_macro_f1'],.5)
        self.assertEqual(q['valid_structure_field_micro_f1'],.75)
    def test_legal_document_to_verdict_mapping(self):
        gold={'《文书1》':'判决结果5','《文书2》':'判决结果1'}
        self.assertEqual(m.score_answer(str(gold),gold)['structured_exact'],1)
        q=m.score_answer("{'《文书1》':'判决结果1','《文书2》':'判决结果1'}",gold)
        self.assertEqual(q['gold_schema'],'string_mapping');self.assertEqual(q['field_f1'],.5)
        self.assertEqual(m.score_answer("{'《文书1》':['判决结果5']}",gold)['shape_compatible'],False)
    def test_policy_is_explicitly_non_official(self):
        p=m.metric_policy();self.assertFalse(p['official_loong_gpt4_score']);self.assertFalse(p['external_judge_called'])
        with self.assertRaises(ValueError):m.gold_schema({'x':1})

if __name__=='__main__':unittest.main()
