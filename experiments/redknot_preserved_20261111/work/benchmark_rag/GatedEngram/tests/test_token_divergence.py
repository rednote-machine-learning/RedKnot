"""Synthetic CPU tokenizer/trajectory regressions, never experimental data."""
import copy
import unittest
from gpu_pilot import token_divergence as t


class ByteDecoder:
    eos_token_id=1
    special_ids=frozenset((0,1,99))
    identity={'schema':'synthetic-test-tokenizer','tokenizer_json_sha256':'a'*64}
    table={0:b'<BOS>',1:b'<EOS>',2:b'a',3:b'b',4:b' ',5:b'c',10:b'\xc3',11:b'\xa9',12:b'\xc3\xa9',99:b'<SPECIAL>'}
    def decode(self,ids,*,skip_special_tokens):
        return b''.join(self.table[i] for i in ids if not skip_special_tokens or i not in self.special_ids).decode('utf-8','replace')
    def piece(self,token):return repr(self.table[token])


def record(ids,finish='stop'):
    return dict(output_ids=list(ids),output_ids_sha256=t.digest(list(ids)),text=ByteDecoder().decode(ids,skip_special_tokens=True),
        finish_reason=finish,natural_EOS=True,truncated=finish=='length',technical_max_new_tokens=len(ids) if finish=='length' else 999,
        quality_output_complete=finish=='stop',quality_limit_reason='engine_context_limit' if finish=='length' else None)


class TokenDivergenceTests(unittest.TestCase):
    def compare(self,a,b):return t.compare_quality_records(a,b,ByteDecoder())

    def test_identical_includes_terminal_eos(self):
        result=self.compare(record([2,3,1]),record([2,3,1]))
        self.assertTrue(result['identical_output_ids']);self.assertEqual(result['common_prefix_tokens_including_special'],3)
        self.assertIsNone(result['first_difference_position_1based']);self.assertIsNone(result['first_difference'])
        self.assertEqual(result['output_lengths']['s1']['raw_generated_tokens'],3)
        self.assertEqual(result['output_lengths']['s1']['non_special_generated_tokens'],2)
        self.assertEqual(result['sha256'],t.digest({k:v for k,v in result.items() if k!='sha256'}))

    def test_first_and_later_different_generated_token_one_based(self):
        for left,right,position in [([2,1],[3,1],1),([2,4,3,1],[2,4,5,1],3)]:
            value=self.compare(record(left),record(right))
            self.assertEqual(value['first_different_generated_token_1based'],position)
            self.assertEqual(value['common_prefix_tokens_including_special'],position-1)
            self.assertEqual(value['first_difference']['recomputed']['token_id'],left[position-1])
            self.assertFalse(value['first_difference_is_first_factual_error_claim'])
            self.assertFalse(value['causal_attribution_claim'])

    def test_eos_versus_continuation_is_actual_token_difference(self):
        value=self.compare(record([2,1]),record([2,3,1]))
        self.assertEqual(value['difference_kind'],'eos_vs_token')
        self.assertEqual(value['first_different_generated_token_1based'],2)
        self.assertEqual(value['first_difference']['recomputed']['token_kind'],'eos')
        self.assertEqual(value['first_difference']['recomputed']['vocab_piece'],"b'<EOS>'")

    def test_length_bound_prefix_end_is_not_fabricated_generated_token(self):
        value=self.compare(record([2,3],finish='length'),record([2,3,5,1]))
        self.assertEqual(value['difference_kind'],'length_bound_prefix_end')
        self.assertEqual(value['first_difference_position_1based'],3)
        self.assertIsNone(value['first_different_generated_token_1based'])
        self.assertEqual(value['length_bound_prefix_ended_arm'],'recomputed')
        self.assertIsNone(value['first_difference']['recomputed']['token_id'])

    def test_byte_tokens_never_join_standalone_decodes_into_answer(self):
        decoder=ByteDecoder();left=record([10,11,1]);right=record([12,1])
        self.assertEqual(left['text'],'é')
        self.assertNotEqual(decoder.decode([10],skip_special_tokens=True)+decoder.decode([11],skip_special_tokens=True),'é')
        value=self.compare(left,right)
        self.assertTrue(value['identical_display_text']);self.assertFalse(value['identical_output_ids'])
        self.assertEqual(value['first_different_generated_token_1based'],1)
        self.assertIn('é',value['first_difference']['recomputed']['context_text'])
        self.assertTrue(value['first_difference']['recomputed']['standalone_decode_is_label_only'])

    def test_actual_long_output_counts_exclude_eos_and_specials(self):
        value=self.compare(record([2]*49+[99,1]),record([3]*50+[1]))
        self.assertEqual(value['output_lengths']['recomputed']['raw_generated_tokens'],51)
        self.assertEqual(value['output_lengths']['recomputed']['non_special_generated_tokens'],49)
        self.assertFalse(value['both_outputs_at_least50_non_special_tokens'])
        self.assertTrue(value['output_lengths']['s1']['non_special_between50_and200'])

    def test_empty_display_eos_only_is_valid_and_not_long(self):
        value=self.compare(record([1]),record([1]))
        self.assertEqual(value['output_lengths']['s1']['non_special_generated_tokens'],0)
        self.assertFalse(value['both_outputs_at_least50_non_special_tokens'])

    def test_eos_position_and_technical_budget_rejections(self):
        for bad in (record([2]),record([1,2,1]),record([2,1],finish='length')):
            with self.assertRaisesRegex(ValueError,'EOS included'):self.compare(bad,record([1]))
        bad=record([2],finish='length');bad['technical_max_new_tokens']=8
        with self.assertRaisesRegex(ValueError,'actual engine budget'):self.compare(bad,record([1]))

    def test_hash_and_actual_full_decode_are_required(self):
        bad=record([2,1]);bad['output_ids_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'IDs/hash'):self.compare(bad,record([1]))
        bad=record([2,1]);bad['text']='fabricated'
        with self.assertRaisesRegex(ValueError,'whole output decode'):self.compare(bad,record([1]))

    def test_official_f1_ascii_punctuation_and_empty_not_legacy_equivalent(self):
        self.assertEqual(t.official_longbench_f1('The red_book',['redbook']),1.)
        self.assertEqual(t.official_longbench_f1('“red”',['red']),0.)
        self.assertEqual(t.official_longbench_f1('the',['a']),0.)
        self.assertEqual(t.official_longbench_f1('two red',['blue','two red']),1.)

    def test_official_f1_does_not_first_line_clip_qasper(self):
        self.assertEqual(t.official_longbench_f1('answer\nextra',['answer']),2/3)
        self.assertEqual(t.metric_policy('qasper')['primary_metrics'],['official_longbench_f1'])
        self.assertEqual(t.metric_policy('musique')['primary_metrics'],['legacy_answer_em','legacy_answer_token_f1'])
        with self.assertRaises(ValueError):t.metric_policy('unsupported')

    def test_decoder_rejects_unpinned_actual_tokenizer_before_import(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tokenizer.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'tokenizer SHA'):t.TokenizerDecoder(path,'0'*64)


if __name__=='__main__':unittest.main()
