"""CPU-only LongBench v2 answer extraction, matching the pinned official source.

This is the official extraction/accuracy rule, not a claim that our generation
settings reproduce official temperature, truncation, or output-token caps.
"""
import re

SOURCE_REVISION='2e00731f8d0bff23dc4325161044d0ed8af94c1e'
OFFICIAL_PRED_SHA256='ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138'
OFFICIAL_PROMPT_SHA256='68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950'
SOURCE_URL='https://github.com/THUDM/LongBench/blob/'+SOURCE_REVISION+'/pred.py'


def extract_answer(response):
    if type(response) is not str:raise ValueError('whole decoded response must be a string')
    response=response.replace('*','')
    match=re.search(r'The correct answer is \(([A-D])\)',response)
    if match:return match.group(1)
    match=re.search(r'The correct answer is ([A-D])',response)
    return match.group(1) if match else None


def mc_quality(text,gold_letter):
    if gold_letter not in ('A','B','C','D'):raise ValueError('original gold letter A-D required')
    prediction=extract_answer(text)
    return dict(mc_predicted_letter=prediction,mc_parse_success=prediction is not None,
        mc_correct=prediction==gold_letter,mc_gold_letter=gold_letter)


def metric_policy(dataset):
    if dataset!='longbench_v2':raise ValueError('LongBench v2 multiple-choice dataset required')
    return dict(dataset=dataset,primary_metrics=['longbench_v2_official_mc_accuracy'],
        secondary_metrics=['mc_parse_success'],legacy_scores_preserved=True,
        legacy_em_f1_are_auxiliary_not_official_mc_metrics=True,
        official_parser_source=SOURCE_URL,official_parser_source_sha256=OFFICIAL_PRED_SHA256,
        official_prompt_sha256=OFFICIAL_PROMPT_SHA256,
        parser_policy='remove asterisks; first parenthesized A-D match, otherwise first unparenthesized match; exact case',
        prediction_preprocessing='whole original decoded response; no first-line clipping',
        unparsed_is_incorrect=True,generation_matches_official_benchmark_settings=False,
        greedy_natural_eos_preserved=True)
