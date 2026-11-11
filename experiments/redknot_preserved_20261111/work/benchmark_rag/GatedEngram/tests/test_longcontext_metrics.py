"""Real strict parser edge cases, including official regex precedence."""
import unittest
from gpu_pilot.longcontext_metrics import extract_answer,mc_quality,metric_policy

class LongcontextMetricTests(unittest.TestCase):
    def test_official_formats_and_asterisk_removal(self):
        for text in ('The correct answer is (B)','**The correct answer is (B)**','The correct answer is B.'):
            self.assertEqual(extract_answer(text),'B');self.assertTrue(mc_quality(text,'B')['mc_correct'])
    def test_case_format_and_letters_not_relaxed(self):
        for text in ('B','(B)','the correct answer is (B)','The correct answer is (b)','The correct answer is (E)','unanswerable'):
            self.assertIsNone(extract_answer(text));self.assertFalse(mc_quality(text,'B')['mc_correct'])
    def test_parenthesized_pass_precedes_earlier_unparenthesized_match(self):
        self.assertEqual(extract_answer('The correct answer is A. Later: The correct answer is (C).'),'C')
        self.assertEqual(extract_answer('The correct answer is (D). The correct answer is (B).'),'D')
    def test_multiline_full_response_and_no_first_line_clipping(self):
        self.assertEqual(extract_answer('Some explanation\nThe correct answer is (C)\nDetails.'),'C')
    def test_wrong_but_parseable_is_not_correct(self):
        self.assertEqual(mc_quality('The correct answer is (A)','D'),dict(mc_predicted_letter='A',mc_parse_success=True,mc_correct=False,mc_gold_letter='D'))
    def test_provenance_and_mc_primary_not_legacy_f1(self):
        p=metric_policy('longbench_v2');self.assertEqual(p['primary_metrics'],['longbench_v2_official_mc_accuracy'])
        self.assertFalse(p['generation_matches_official_benchmark_settings'])
        with self.assertRaises(ValueError):metric_policy('qasper')
        with self.assertRaises(ValueError):mc_quality('A','E')

if __name__=='__main__':unittest.main()
